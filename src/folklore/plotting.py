"""
Diagnostic plots.

All plotting functions share the same keyword arguments:

norm : matplotlib.colors.Normalize, False or None
    Image normalization. None (default) computes one with
    :func:`astropy.visualization.simple_norm` (log stretch, 25–99.98
    percentiles of the positive pixels); False disables normalization; a
    Normalize instance is used as given.
norm_kwargs : dict, optional
    Keyword arguments for ``simple_norm`` when ``norm=None``.
show : bool
    Display the figure (``plt.show()``). If False the figure is closed.
save : bool
    Save the figure.
output_path : str or pathlib.Path, optional
    Where to save. A path with a suffix (``"out/fig.png"``) is used as the
    file name; a path without suffix is treated as a directory and the
    default file name of each function is used inside it.
dpi : int
    Resolution of the saved figure.

The reference image of the multi-filter plots is the first filter of the
cube.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import simple_norm
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

from .detection import (
    _galaxy_and_center_masks,
    build_contrast_map,
    refine_hierarchical,
    select_candidate_pixels,
)
from .properties import classify_clumps

__all__ = [
    "plot_galaxy_masks",
    "plot_masks_filters",
    "plot_contrast_steps",
    "plot_contrast_levels",
    "plot_contrast_levels_filters",
    "plot_candidate_pixels",
    "plot_clumps_filters",
    "plot_final_clumps",
    "plot_union_clumps",
    "plot_galaxy_region_levels",
    "plot_classify_clumps",
    "plot_clump_rings",
    "plot_expanded_clumps",
]

DEFAULT_NORM_KWARGS = dict(stretch="log", min_percent=25, max_percent=99.98)

#: Contour colors of the significance levels (lowest level first).
DEFAULT_LEVEL_COLORS = [
    "tab:green", "tab:red", "tab:olive", "tab:cyan", "tab:purple", "tab:orange"]


def _finalize_plot(fig, *, show, save, output_path, filename, dpi):
    """Optionally save and/or display a Matplotlib figure."""
    if save:
        path = Path(output_path) if output_path is not None else Path(filename)
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            target = path
        else:
            path.mkdir(parents=True, exist_ok=True)
            target = path / filename
        fig.savefig(target, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _resolve_norm(image, norm=None, norm_kwargs=None):
    """Return the normalization to use for ``image`` (see module docstring)."""
    if norm is False:
        return None
    if norm is not None:
        return norm
    if norm_kwargs is None:
        norm_kwargs = DEFAULT_NORM_KWARGS
    good = ~np.logical_or(np.isnan(image), image <= 0)
    return simple_norm(image[good], **norm_kwargs)


def _reference_image(data_cube, filters):
    """Return (name, image) of the first filter of the cube."""
    ref_filter = list(filters.keys())[0]
    return ref_filter, data_cube[filters[ref_filter]]


def _mask_from_pixels(dfp, shape):
    """Boolean mask from a table with ``x`` and ``y`` columns."""
    mask = np.zeros(shape, dtype=bool)
    mask[dfp["y"].values.astype(int), dfp["x"].values.astype(int)] = True
    return mask


def plot_galaxy_masks(
    galaxy_masks,
    norm=None,
    norm_kwargs=None,
    show=True,
    save=False,
    output_path=None,
    dpi=300):
    """
    Plot the collapsed galaxy image with the percentile masks overlaid.

    Parameters
    ----------
    galaxy_masks : dict
        Output of :func:`folklore.galaxy.build_galaxy_masks`.
    norm, norm_kwargs, show, save, output_path, dpi
        See :mod:`folklore.plotting`. Default file name:
        ``plot_galaxy_masks.png``.
    """

    image = galaxy_masks["image"]
    masks = galaxy_masks["masks"]

    norm_use = _resolve_norm(image, norm, norm_kwargs)

    fig = plt.figure(figsize=(8, 8))

    plt.imshow(image, origin="lower", cmap="gray", norm=norm_use)

    colors = plt.cm.cool(np.linspace(0.2, 0.8, len(masks)))

    for (p, mask), color in zip(masks.items(), colors):

        plt.contour(mask, levels=[0.5], colors=[color], linewidths=1.5)

        plt.plot([], [], color=color, label=f"{p}th percentile")

    plt.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5))

    plt.axis("off")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    _finalize_plot(
        fig, show=show, save=save, output_path=output_path,
        filename="plot_galaxy_masks.png", dpi=dpi)


def plot_masks_filters(
    data_cube,
    filters,
    galaxy_masks,
    norm=None,
    norm_kwargs=None,
    show=True,
    save=False,
    output_path=None,
    dpi=300):
    """
    Plot all filters of the cube with the galaxy masks overlaid.

    The masks are computed from the collapsed image and are therefore
    identical for all filters. This plot is intended as a diagnostic to
    verify that the masks properly cover the galaxy in all bands.

    Parameters
    ----------
    data_cube : ndarray
        Data cube with shape (n_filters, ny, nx).
    filters : dict
        Dictionary mapping filter_name → cube index.
    galaxy_masks : dict
        Output of :func:`folklore.galaxy.build_galaxy_masks`.
    norm, norm_kwargs, show, save, output_path, dpi
        See :mod:`folklore.plotting`. The normalization is computed per
        filter. Default file name: ``plot_masks_filters.png``.
    """

    masks = galaxy_masks["masks"]

    n_filters = len(filters)

    ncols = int(np.ceil(np.sqrt(n_filters)))
    nrows = int(np.ceil(n_filters / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4*ncols, 4*nrows))

    axes = np.atleast_1d(axes).flatten()

    colors = plt.cm.cool(np.linspace(0.2, 0.8, len(masks)))
    legend_lines = [
        Line2D([0], [0], color=color, lw=2, label=f"{p}th percentile")
        for color, p in zip(colors, sorted(masks.keys()))]

    for i, (filt, idx) in enumerate(filters.items()):

        ax = axes[i]

        data = data_cube[idx]

        ax.imshow(
            data,
            origin="lower",
            cmap="gray",
            norm=_resolve_norm(data, norm, norm_kwargs))

        for (p, mask), color in zip(sorted(masks.items()), colors):

            ax.contour(mask, levels=[0.5], colors=[color], linewidths=2)

        ax.set_title(filt)
        ax.axis("off")

    for j in range(n_filters, len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    fig.legend(
        handles=legend_lines,
        loc="lower center",
        ncol=len(legend_lines),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02))

    _finalize_plot(
        fig, show=show, save=save, output_path=output_path,
        filename="plot_masks_filters.png", dpi=dpi)


def plot_contrast_steps(
    imdata,
    gal_mask,
    filt_name=None,
    sigma_smooth=None,
    bkg_size=20,
    sigma_size=20,
    vmax_contrast=10,
    norm=None,
    norm_kwargs=None,
    show=True,
    save=False,
    output_path=None,
    dpi=300):
    """
    Plot the steps of the local contrast computation.

    Panels:

    1. Original image
    2. Estimated background
    3. Background-subtracted image
    4. Local noise (sigma)
    5. Local contrast map

    Parameters
    ----------
    imdata : ndarray
        Image data for a single filter.
    gal_mask : ndarray of bool
        Mask defining the galaxy region.
    filt_name : str, optional
        Filter name for the plot title.
    sigma_smooth, bkg_size, sigma_size
        Passed to :func:`folklore.detection.build_contrast_map`.
    vmax_contrast : float, optional
        Upper limit of the contrast map color scale.
    norm, norm_kwargs, show, save, output_path, dpi
        See :mod:`folklore.plotting`. ``norm`` applies to the original
        image; the background and subtracted images always use
        ``simple_norm`` with ``norm_kwargs``. Default file name:
        ``plot_contrast_steps.png``.
    """

    contrast_map, background, image_sub, sigma_local = build_contrast_map(
        imdata,
        gal_mask,
        sigma_smooth=sigma_smooth,
        bkg_size=bkg_size,
        sigma_size=sigma_size)

    norm1 = _resolve_norm(imdata, norm, norm_kwargs)
    norm2 = _resolve_norm(background, None, norm_kwargs)
    norm3 = _resolve_norm(image_sub, None, norm_kwargs)

    fig, axs = plt.subplots(3, 2, figsize=(12, 16))

    axs[0, 0].imshow(imdata, origin="lower", cmap="gray", norm=norm1)
    title = "original image" if filt_name is None else f"{filt_name}: original image"
    axs[0, 0].set_title(title)

    axs[0, 1].imshow(background, origin="lower", cmap="gray", norm=norm2)
    axs[0, 1].set_title("background")

    axs[1, 0].imshow(image_sub, origin="lower", cmap="gray", norm=norm3)
    axs[1, 0].set_title("image − background")

    im1 = axs[1, 1].imshow(sigma_local, origin="lower", cmap="viridis")
    axs[1, 1].set_title("local sigma")
    plt.colorbar(im1, ax=axs[1, 1])

    im2 = axs[2, 0].imshow(
        contrast_map,
        origin="lower",
        cmap="inferno",
        vmin=0,
        vmax=vmax_contrast)
    axs[2, 0].set_title("local contrast")
    plt.colorbar(im2, ax=axs[2, 0])

    axs[2, 1].axis("off")

    plt.tight_layout()
    _finalize_plot(
        fig, show=show, save=save, output_path=output_path,
        filename="plot_contrast_steps.png", dpi=dpi)


def plot_contrast_levels(
    imdata,
    gal_mask,
    filt_name=None,
    levels=(2, 3, 4, 5),
    colors=None,
    sigma_smooth=None,
    bkg_size=20,
    sigma_size=20,
    norm=None,
    norm_kwargs=None,
    show=True,
    save=False,
    output_path=None,
    dpi=300):
    """
    Plot contrast significance contours over the original image.

    Helps choosing the detection thresholds: it shows which structures
    are detected at different significance levels.

    Parameters
    ----------
    imdata : ndarray
        2D image for a single filter.
    gal_mask : ndarray of bool
        Binary mask defining the galaxy region.
    filt_name : str, optional
        Filter name used for the plot title.
    levels : sequence of float, optional
        Significance levels (sigma) of the contours.
    colors : sequence of str, optional
        Colors used for each level. If None, green, red, olive, ... are
        used, in this order.
    sigma_smooth, bkg_size, sigma_size
        Passed to :func:`folklore.detection.build_contrast_map`.
    norm, norm_kwargs, show, save, output_path, dpi
        See :mod:`folklore.plotting`. Default file name:
        ``plot_contrast_levels.png``.
    """

    contrast_map, _, _, _ = build_contrast_map(
        imdata,
        gal_mask,
        sigma_smooth=sigma_smooth,
        bkg_size=bkg_size,
        sigma_size=sigma_size)

    norm_use = _resolve_norm(imdata, norm, norm_kwargs)

    if colors is None:
        colors = [DEFAULT_LEVEL_COLORS[i % len(DEFAULT_LEVEL_COLORS)]
                  for i in range(len(levels))]

    fig = plt.figure(figsize=(7, 7))

    plt.imshow(imdata, origin="lower", cmap="gray", norm=norm_use)

    plt.contour(contrast_map, levels=levels, colors=colors, linewidths=1)

    plt.contour(gal_mask, levels=[0.5], colors="blue", linewidths=1)

    legend_lines = [
        Line2D([0], [0], color=c, lw=2, label=f"{lvl}σ")
        for c, lvl in zip(colors, levels)]

    legend_lines.append(
        Line2D([0], [0], color="blue", lw=2, label="galaxy mask"))

    plt.legend(
        frameon=False,
        handles=legend_lines,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0)

    plt.title(filt_name if filt_name is not None else " ")

    plt.tight_layout()
    _finalize_plot(
        fig, show=show, save=save, output_path=output_path,
        filename="plot_contrast_levels.png", dpi=dpi)


def plot_contrast_levels_filters(
    data_cube,
    filters,
    galaxy_masks,
    levels=(3, 5, 7),
    colors=None,
    ncols=2,
    sigma_smooth=None,
    bkg_size=20,
    sigma_size=20,
    norm=None,
    norm_kwargs=None,
    show=True,
    save=False,
    output_path=None,
    dpi=300):
    """
    Plot contrast significance contours for every filter of the cube.

    Multi-filter version of :func:`plot_contrast_levels`, with one panel
    per filter. The contrast map of each filter is computed inside the
    galaxy region (the lowest-percentile mask of ``galaxy_masks``), as in
    :func:`folklore.detection.build_clump_dataframes`.

    Parameters
    ----------
    data_cube : ndarray
        Data cube with shape (n_filters, ny, nx).
    filters : dict
        Dictionary mapping filter_name → cube index.
    galaxy_masks : dict
        Output of :func:`folklore.galaxy.build_galaxy_masks`.
    levels : sequence of float, optional
        Significance levels (sigma) of the contours.
    colors : sequence of str, optional
        Colors used for each level. If None, green, red, olive, ... are
        used, in this order.
    ncols : int, optional
        Number of panel columns. Default is 2.
    sigma_smooth, bkg_size, sigma_size
        Passed to :func:`folklore.detection.build_contrast_map`.
    norm, norm_kwargs, show, save, output_path, dpi
        See :mod:`folklore.plotting`. The normalization is computed per
        filter. Default file name: ``plot_contrast_levels_filters.png``.
    """

    gal_mask, _ = _galaxy_and_center_masks(galaxy_masks)

    if colors is None:
        colors = [DEFAULT_LEVEL_COLORS[i % len(DEFAULT_LEVEL_COLORS)]
                  for i in range(len(levels))]

    n_filters = len(filters)
    nrows = int(np.ceil(n_filters / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4*ncols, 4*nrows))

    axes = np.atleast_1d(axes).flatten()

    for i, (filt, idx) in enumerate(filters.items()):

        ax = axes[i]

        data = data_cube[idx]

        contrast_map, _, _, _ = build_contrast_map(
            data,
            gal_mask,
            sigma_smooth=sigma_smooth,
            bkg_size=bkg_size,
            sigma_size=sigma_size)

        ax.imshow(
            data,
            origin="lower",
            cmap="gray",
            norm=_resolve_norm(data, norm, norm_kwargs))

        ax.contour(contrast_map, levels=levels, colors=colors, linewidths=1)

        ax.set_title(filt)
        ax.axis("off")

    for j in range(n_filters, len(axes)):
        axes[j].axis("off")

    legend_lines = [
        Line2D([0], [0], color=c, lw=2, label=f"{lvl}σ")
        for c, lvl in zip(colors, levels)]

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    fig.legend(
        handles=legend_lines,
        loc="lower center",
        ncol=len(legend_lines),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01))

    _finalize_plot(
        fig, show=show, save=save, output_path=output_path,
        filename="plot_contrast_levels_filters.png", dpi=dpi)


def plot_candidate_pixels(
    data_cube,
    filters,
    galaxy_masks,
    sigma_center=3.5,
    sigma_outer=4.0,
    sigma_split=6.0,
    sigma_smooth=None,
    bkg_size=20,
    sigma_size=20,
    norm=None,
    norm_kwargs=None,
    show=True,
    save=False,
    output_path=None,
    dpi=300):
    """
    Plot candidate clumps detected from the local contrast map.

    One figure is produced per filter.

    Parameters
    ----------
    data_cube : ndarray
        Data cube (n_filters, ny, nx).
    filters : dict
        Mapping ``{filter_name: cube_index}``.
    galaxy_masks : dict
        Output of :func:`folklore.galaxy.build_galaxy_masks`.
    sigma_center, sigma_outer, sigma_split
        Detection thresholds (see
        :func:`folklore.detection.build_clump_dataframes`).
    sigma_smooth, bkg_size, sigma_size
        Passed to :func:`folklore.detection.build_contrast_map`.
    norm, norm_kwargs, show, save, output_path, dpi
        See :mod:`folklore.plotting`. The filter name is appended to the
        file name, e.g. ``candidate_pixels.png`` →
        ``candidate_pixels_F090W.png``. Default file name:
        ``candidate_pixels_<filter>.png``.
    """

    gal_mask, gal_mask_center = _galaxy_and_center_masks(galaxy_masks)

    for filt, idx in filters.items():

        imdata = data_cube[idx]

        contrast_map, _, _, _ = build_contrast_map(
            imdata,
            gal_mask,
            sigma_smooth=sigma_smooth,
            bkg_size=bkg_size,
            sigma_size=sigma_size)

        candidate_pixels = select_candidate_pixels(
            contrast_map,
            gal_mask,
            gal_mask_center,
            sigma_center=sigma_center,
            sigma_outer=sigma_outer)

        labels_refined = refine_hierarchical(
            contrast_map,
            candidate_pixels,
            sigma_split=sigma_split)

        fig = plt.figure(figsize=(7, 7))

        plt.imshow(
            imdata,
            origin="lower",
            cmap="gray",
            norm=_resolve_norm(imdata, norm, norm_kwargs))

        for lab in np.unique(labels_refined):
            if lab == 0:
                continue

            plt.contour(
                labels_refined == lab,
                levels=[0.5],
                colors="red",
                linewidths=1)

        plt.title(f"{filt} — candidate clumps")

        plt.axis("off")
        plt.tight_layout()

        # one file per filter
        filt_path = output_path
        if output_path is not None and Path(output_path).suffix:
            p = Path(output_path)
            filt_path = p.with_name(f"{p.stem}_{filt}{p.suffix}")

        _finalize_plot(
            fig, show=show, save=save, output_path=filt_path,
            filename=f"candidate_pixels_{filt}.png", dpi=dpi)


def plot_clumps_filters(
    data_cube,
    filters,
    pixels_by_filter,
    norm=None,
    norm_kwargs=None,
    show=True,
    save=False,
    output_path=None,
    dpi=300):
    """
    Plot the clumps of every filter over the reference image.

    Parameters
    ----------
    data_cube : ndarray
        Data cube with shape (n_filters, ny, nx).
    filters : dict
        Dictionary mapping filter_name → cube index.
    pixels_by_filter : dict
        Output of :func:`folklore.detection.build_clump_dataframes`.
    norm, norm_kwargs, show, save, output_path, dpi
        See :mod:`folklore.plotting`. Default file name:
        ``plot_clumps_filters.png``.
    """

    ref_filter, im_ref = _reference_image(data_cube, filters)

    cmap = plt.get_cmap("tab20")

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(
        im_ref,
        origin="lower",
        cmap="gray",
        norm=_resolve_norm(im_ref, norm, norm_kwargs))

    legend_lines = []

    for i, filt in enumerate(filters):

        dfp = pixels_by_filter.get(filt)

        if dfp is None or dfp.empty:
            continue

        color = cmap(i % 20)

        plt.contour(
            _mask_from_pixels(dfp, im_ref.shape),
            levels=[0.5],
            colors=[color],
            linewidths=1)

        legend_lines.append(Line2D([0], [0], color=color, lw=2, label=filt))

    plt.legend(
        handles=legend_lines,
        fontsize=8,
        ncol=2,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5))

    plt.title(ref_filter)

    plt.axis("off")
    plt.tight_layout()
    _finalize_plot(
        fig, show=show, save=save, output_path=output_path,
        filename="plot_clumps_filters.png", dpi=dpi)


def plot_final_clumps(
    data_cube,
    filters,
    df_pixels_final,
    norm=None,
    norm_kwargs=None,
    show=True,
    save=False,
    output_path=None,
    dpi=300):
    """
    Plot the per-filter detections of the final physical clumps.

    Parameters
    ----------
    data_cube : ndarray
        Data cube with shape (n_filters, ny, nx).
    filters : dict
        Dictionary mapping filter_name → cube index.
    df_pixels_final : pandas.DataFrame
        Pixel table of validated physical clumps (output of
        :func:`folklore.matching.build_final_clump_dataframes`).
    norm, norm_kwargs, show, save, output_path, dpi
        See :mod:`folklore.plotting`. Default file name:
        ``plot_final_clumps.png``.
    """

    ref_filter, imdata = _reference_image(data_cube, filters)

    cmap = plt.get_cmap("tab20")

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(
        imdata,
        origin="lower",
        cmap="gray",
        norm=_resolve_norm(imdata, norm, norm_kwargs))

    legend_lines = []

    for i, filt in enumerate(filters):

        dfp = df_pixels_final[df_pixels_final["filter"] == filt]

        if dfp.empty:
            continue

        color = cmap(i % 20)

        plt.contour(
            _mask_from_pixels(dfp, imdata.shape),
            levels=[0.5],
            colors=[color],
            linewidths=1)

        legend_lines.append(Line2D([0], [0], color=color, lw=2, label=filt))

    plt.legend(
        handles=legend_lines,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False)

    plt.title(ref_filter)

    plt.axis("off")
    plt.tight_layout()
    _finalize_plot(
        fig, show=show, save=save, output_path=output_path,
        filename="plot_final_clumps.png", dpi=dpi)


def plot_union_clumps(
    data_cube,
    filters,
    union_masks,
    norm=None,
    norm_kwargs=None,
    show=True,
    save=False,
    output_path=None,
    dpi=300):
    """
    Plot the union masks of the final clumps over the reference image.

    Parameters
    ----------
    data_cube : ndarray
        Data cube with shape (n_filters, ny, nx).
    filters : dict
        Dictionary mapping filter_name → cube index.
    union_masks : dict
        ``{clump_id: boolean mask}``.
    norm, norm_kwargs, show, save, output_path, dpi
        See :mod:`folklore.plotting`. Default file name:
        ``plot_union_clumps.png``.
    """

    ref_filter, imdata = _reference_image(data_cube, filters)

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(
        imdata,
        origin="lower",
        cmap="gray",
        norm=_resolve_norm(imdata, norm, norm_kwargs))

    for mask_union in union_masks.values():

        plt.contour(mask_union, levels=[0.5], linewidths=1)

    plt.title(ref_filter)
    plt.axis("off")
    plt.tight_layout()
    _finalize_plot(
        fig, show=show, save=save, output_path=output_path,
        filename="plot_union_clumps.png", dpi=dpi)


def plot_galaxy_region_levels(
    image,
    region,
    union_masks,
    df_union_clumps,
    n_show=8,
    ncols=4,
    norm=None,
    norm_kwargs=None,
    show=True,
    save=False,
    output_path=None,
    dpi=300):
    """
    Plot the last levels of the galaxy region and the resulting classification.

    Helps choosing the galaxy body mask by eye: each panel shows one level
    of :func:`folklore.galaxy.build_galaxy_region` (white), the clumps
    classified as disk (green) or outside (red) with that mask, the mask
    area and the number of clumps in each class. Panels are labelled with
    the negative index to use, e.g. ``region["masks"][-2]``.

    Parameters
    ----------
    image : ndarray
        2D image used to build the region (the collapsed cube).
    region : dict
        Output of :func:`folklore.galaxy.build_galaxy_region`.
    union_masks : dict
        ``{clump_id: boolean mask}``.
    df_union_clumps : pandas.DataFrame
        Clump catalog with ``clump_id``, ``x0`` and ``y0`` (output of
        :func:`folklore.properties.compute_clump_geometry`).
    n_show : int, optional
        Number of levels shown, counted from the last one. Default is 8.
    ncols : int, optional
        Number of panel columns. Default is 4.
    norm, norm_kwargs, show, save, output_path, dpi
        See :mod:`folklore.plotting`. Default file name:
        ``plot_galaxy_region_levels.png``.
    """

    masks = region["masks"]
    n_levels = len(masks)
    indices = range(max(0, n_levels - n_show), n_levels)

    norm_use = _resolve_norm(image, norm, norm_kwargs)

    nrows = int(np.ceil(len(indices) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.5*ncols, 4.2*nrows))

    axes = np.atleast_1d(axes).flatten()

    for ax, i in zip(axes, indices):

        mask = masks[i]

        df = classify_clumps(df_union_clumps, mask).set_index("clump_id")

        ax.imshow(image, origin="lower", cmap="gray", norm=norm_use)

        ax.contour(mask, levels=[0.5], colors="white", linewidths=1.5)

        for cid, mask_clump in union_masks.items():
            color = "lime" if df.loc[cid, "component"] == "disk" else "red"
            ax.contour(mask_clump, levels=[0.5], colors=[color], linewidths=0.8)

        n_disk = int((df["component"] == "disk").sum())

        ax.set_title(
            f"level {i - n_levels} ({i}/{n_levels - 1})\n"
            f"area = {int(mask.sum())} px | "
            f"disk = {n_disk}, outside = {len(df) - n_disk}",
            fontsize=9)
        ax.axis("off")

    for ax in axes[len(indices):]:
        ax.axis("off")

    outline = [pe.Stroke(linewidth=3.5, foreground="black"), pe.Normal()]
    legend_lines = [
        Line2D([0], [0], color="white", lw=2, label="galaxy mask",
               path_effects=outline),
        Line2D([0], [0], color="lime", lw=2, label="disk"),
        Line2D([0], [0], color="red", lw=2, label="outside")]

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    fig.legend(
        handles=legend_lines,
        loc="lower center",
        ncol=len(legend_lines),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01))

    _finalize_plot(
        fig, show=show, save=save, output_path=output_path,
        filename="plot_galaxy_region_levels.png", dpi=dpi)


def plot_classify_clumps(
    data_cube,
    filters,
    union_masks,
    galaxy_mask,
    df_union_clumps,
    show_ids=True,
    norm=None,
    norm_kwargs=None,
    show=True,
    save=False,
    output_path=None,
    dpi=300,
):
    """
    Plot the galaxy body region and the classified clumps.

    Disk clumps are drawn in green, outside clumps in red and the galaxy
    region in white.

    Parameters
    ----------
    data_cube : ndarray
        Data cube with shape (n_filters, ny, nx).
    filters : dict
        Dictionary mapping filter_name → cube index.
    union_masks : dict
        ``{clump_id: boolean mask}``.
    galaxy_mask : ndarray of bool
        Galaxy body mask used for the classification.
    df_union_clumps : pandas.DataFrame
        Output of :func:`folklore.properties.classify_clumps`.
    show_ids : bool, optional
        Write the clump IDs at the clump centroids.
    norm, norm_kwargs, show, save, output_path, dpi
        See :mod:`folklore.plotting`. Default file name:
        ``galaxy_region_clumps.png``.
    """

    ref_filter, imdata = _reference_image(data_cube, filters)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(
        imdata,
        origin="lower",
        cmap="gray",
        norm=_resolve_norm(imdata, norm, norm_kwargs),
    )

    ax.contour(
        galaxy_mask,
        levels=[0.5],
        colors="white",
        linewidths=2,
    )

    clump_lookup = df_union_clumps.set_index("clump_id")

    for cid, mask in union_masks.items():

        row = clump_lookup.loc[cid]

        color = "lime" if row["component"] == "disk" else "red"

        ax.contour(mask, levels=[0.5], colors=color, linewidths=1.5)

        if show_ids:
            ax.text(
                row["x0"],
                row["y0"],
                str(cid),
                color="cyan",
                fontsize=8,
                ha="center",
                va="center",
            )

    legend = [
        Line2D([0], [0], color="white", lw=2, label="galaxy region",
               path_effects=[pe.Stroke(linewidth=3.5, foreground="black"),
                             pe.Normal()]),
        Line2D([0], [0], color="lime", lw=2, label="disk"),
        Line2D([0], [0], color="red", lw=2, label="outside"),
    ]

    ax.legend(
        handles=legend,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5))

    ax.set_title(ref_filter)
    ax.axis("off")

    plt.tight_layout()

    _finalize_plot(
        fig, show=show, save=save, output_path=output_path,
        filename="galaxy_region_clumps.png", dpi=dpi)


def plot_clump_rings(
    data_cube,
    filters,
    union_masks,
    ring_masks,
    norm=None,
    norm_kwargs=None,
    show=True,
    save=False,
    output_path=None,
    dpi=300):
    """
    Plot the clumps (red) and their background rings (cyan).

    Parameters
    ----------
    data_cube : ndarray
        Data cube with shape (n_filters, ny, nx).
    filters : dict
        Dictionary mapping filter_name → cube index.
    union_masks : dict
        ``{clump_id: boolean mask}``.
    ring_masks : dict
        Output of :func:`folklore.photometry.build_background_rings`.
    norm, norm_kwargs, show, save, output_path, dpi
        See :mod:`folklore.plotting`. Default file name:
        ``plot_clump_rings.png``.
    """

    ref_filter, imdata = _reference_image(data_cube, filters)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()

    ax.imshow(
        imdata,
        origin="lower",
        cmap="gray",
        norm=_resolve_norm(imdata, norm, norm_kwargs))

    for cid, clump in union_masks.items():

        ax.contour(clump, levels=[0.5], colors="red", linewidths=2)

        ax.contour(ring_masks[cid], levels=[0.5], colors="cyan", linewidths=0.7)

    ax.set_title(ref_filter)

    plt.tight_layout()
    _finalize_plot(
        fig, show=show, save=save, output_path=output_path,
        filename="plot_clump_rings.png", dpi=dpi)


def plot_expanded_clumps(
    data_cube,
    filters,
    union_masks,
    expanded_masks,
    norm=None,
    norm_kwargs=None,
    show=True,
    save=False,
    output_path=None,
    dpi=300):
    """
    Plot the original clumps (red) and their expanded regions (cyan).

    Parameters
    ----------
    data_cube : ndarray
        Data cube with shape (n_filters, ny, nx).
    filters : dict
        Dictionary mapping filter_name → cube index.
    union_masks : dict
        ``{clump_id: boolean mask}``.
    expanded_masks : dict
        Output of :func:`folklore.photometry.build_expanded_clump_masks`.
    norm, norm_kwargs, show, save, output_path, dpi
        See :mod:`folklore.plotting`. Default file name:
        ``plot_expanded_clumps.png``.
    """

    ref_filter, imdata = _reference_image(data_cube, filters)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()

    ax.imshow(
        imdata,
        origin="lower",
        cmap="gray",
        norm=_resolve_norm(imdata, norm, norm_kwargs)
    )

    for cid, clump in union_masks.items():

        ax.contour(clump, levels=[0.5], colors="red", linewidths=1.5)

        ax.contour(expanded_masks[cid], levels=[0.5], colors="cyan", linewidths=1.0)

    ax.set_title(f"{ref_filter} — clumps + expanded regions")

    plt.tight_layout()

    _finalize_plot(
        fig, show=show, save=save, output_path=output_path,
        filename="plot_expanded_clumps.png", dpi=dpi)
