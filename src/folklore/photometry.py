"""
Clump photometry.

The clump masks found on the detection cube are transferred to any other
cube (e.g. HST or the full JWST set, with different pixel grids) through
the sky coordinates of their pixels. Fluxes are summed inside each mask
and a local background, estimated as the median in a ring around the
clump, is subtracted.
"""

import astropy.units as u
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from skimage.morphology import disk

from .io import get_celestial_wcs

__all__ = [
    "DEFAULT_FILTER_ORDER",
    "reproject_clump_masks_sky",
    "build_background_rings",
    "build_expanded_clump_masks",
    "compute_clump_photometry",
    "sed_fitting_dataframe",
]

#: HST and JWST/NIRCam filters sorted by pivot wavelength.
DEFAULT_FILTER_ORDER = [
    "F225W", "F275W", "F336W",
    "F435W", "F606W", "F070W", "F814W", "F090W",
    "F105W", "F115W", "F125W", "F140W", "F140M",
    "F150W", "F160W", "F162M", "F182M", "F200W",
    "F210M", "F250M", "F277W", "F300M", "F335M",
    "F356W", "F360M", "F410M", "F430M", "F444W",
    "F460M", "F480M",
]


def reproject_clump_masks_sky(
    df_union_pixels,
    new_wcs,
    new_shape,
    grow=0,
):
    """
    Reconstruct clump masks in a new image using sky coordinates.

    Each clump pixel is placed at the nearest pixel of the target grid.

    Parameters
    ----------
    df_union_pixels : pandas.DataFrame
        DataFrame containing ``clump_id``, ``ra`` and ``dec`` (sky
        coordinates of each clump pixel), as returned by
        :func:`folklore.properties.compute_clump_geometry`.
    new_wcs : astropy.wcs.WCS
        WCS of the target image or cube.
    new_shape : tuple
        Shape of the target image (ny, nx).
    grow : int, optional
        Number of binary dilation iterations after reprojection. Useful
        when the target grid is finer than the detection grid.

    Returns
    -------
    union_masks : dict
        Dictionary ``{clump_id: boolean mask}`` on the new grid.
    """

    new_wcs = get_celestial_wcs(new_wcs)

    ny, nx = new_shape

    union_masks = {}

    x_new, y_new = new_wcs.world_to_pixel_values(
        df_union_pixels["ra"].values,
        df_union_pixels["dec"].values,
    )

    x_new = np.floor(x_new + 0.5).astype(int)
    y_new = np.floor(y_new + 0.5).astype(int)

    valid = (
        (x_new >= 0)
        & (x_new < nx)
        & (y_new >= 0)
        & (y_new < ny)
    )

    df_reprojected = df_union_pixels.copy()

    df_reprojected["x_new"] = x_new
    df_reprojected["y_new"] = y_new
    df_reprojected["valid"] = valid

    for cid, group in df_reprojected.groupby("clump_id"):

        mask = np.zeros((ny, nx), dtype=bool)

        good = group["valid"]

        mask[
            group.loc[good, "y_new"],
            group.loc[good, "x_new"],
        ] = True

        if grow > 0:
            mask = binary_dilation(mask, iterations=grow)

        union_masks[cid] = mask

    return union_masks


def build_background_rings(
    union_masks,
    inner_radius=1,
    outer_radius=4,
    exclude_clumps=True):
    """
    Build background rings around each clump.

    The ring is the clump mask dilated ``outer_radius`` times minus the
    clump mask dilated ``inner_radius`` times.

    Parameters
    ----------
    union_masks : dict
        Dictionary clump_id → boolean mask of clump pixels.
    inner_radius : int, optional
        Inner dilation radius in pixels. 0 means the ring starts right at
        the clump border.
    outer_radius : int, optional
        Outer dilation radius in pixels.
    exclude_clumps : bool, optional
        If True, remove pixels belonging to any clump from the ring.

    Returns
    -------
    ring_masks : dict
        Dictionary clump_id → boolean mask of the background ring.
    """

    ring_masks = {}

    if exclude_clumps:
        all_clumps_mask = np.zeros_like(
            next(iter(union_masks.values())),
            dtype=bool
        )
        for mask in union_masks.values():
            all_clumps_mask |= mask

    for cid, clump_mask in union_masks.items():

        if inner_radius == 0:
            inner = clump_mask.copy()
        else:
            inner = binary_dilation(clump_mask, iterations=inner_radius)

        outer = binary_dilation(clump_mask, iterations=outer_radius)

        ring = outer & ~inner

        if exclude_clumps:
            ring &= ~all_clumps_mask

        ring_masks[cid] = ring

    return ring_masks


def build_expanded_clump_masks(
    union_masks,
    expansion_radius=4):
    """
    Build expanded masks around each clump.

    Each mask is dilated with a disk-shaped structuring element of radius
    ``expansion_radius``. Used, for example, to cut per-clump cubes with
    :func:`folklore.io.save_individual_clump_cubes`.

    Parameters
    ----------
    union_masks : dict
        Dictionary ``{clump_id: boolean mask}`` containing the original
        clump masks.
    expansion_radius : int, optional
        Radius in pixels by which each clump mask is expanded.

    Returns
    -------
    expanded_masks : dict
        Dictionary ``{clump_id: boolean mask}`` containing the expanded
        clump regions.
    """

    expanded_masks = {}

    structure = disk(expansion_radius)

    for cid, clump_mask in union_masks.items():

        expanded_masks[cid] = binary_dilation(clump_mask, structure=structure)

    return expanded_masks


def compute_clump_photometry(
    union_masks,
    ring_masks,
    data_cube,
    filters,
    flux_unit):
    """
    Compute clump photometry with local background subtraction.

    For every clump and filter::

        flux     = sum(clump pixels) - median(ring pixels) * n_pix
        flux_err = std(ring pixels) * sqrt(n_pix)

    converted to Jy.

    Parameters
    ----------
    union_masks : dict
        clump_id → clump mask.
    ring_masks : dict
        clump_id → background ring mask (see :func:`build_background_rings`).
    data_cube : ndarray
        Image cube (n_filters, ny, nx).
    filters : dict
        filter_name → cube index.
    flux_unit : astropy.units.Unit or astropy.units.Quantity
        Flux unit of one pixel value, compatible with Jy (e.g. ``u.nJy`` or
        ``10 * u.nJy``).

    Returns
    -------
    df_flux : pandas.DataFrame
        Long-format table with columns
        ``clump_id, filter, flux, flux_err, n_pix`` (fluxes in Jy).
    """

    flux_rows = []

    if isinstance(flux_unit, u.Quantity):
        flux_scale = flux_unit.value
        flux_base_unit = flux_unit.unit
    else:
        flux_scale = 1.0
        flux_base_unit = flux_unit

    to_jy = (flux_scale * flux_base_unit).to(u.Jy).value

    for filt, index in filters.items():

        imdata = data_cube[index]

        for cid, clump_mask in union_masks.items():

            ring_mask = ring_masks[cid]

            clump_pixels = imdata[clump_mask]
            ring_pixels = imdata[ring_mask]

            n_pix = np.sum(clump_mask)

            flux_raw = np.nansum(clump_pixels)

            bkg_median = np.nanmedian(ring_pixels)

            flux_net = flux_raw - bkg_median * n_pix

            bkg_std = np.nanstd(ring_pixels)
            flux_err = bkg_std * np.sqrt(n_pix)

            flux_rows.append({
                "clump_id": cid,
                "filter": filt,
                "flux": flux_net * to_jy,
                "flux_err": flux_err * to_jy,
                "n_pix": n_pix})

    df_flux = pd.DataFrame(
        flux_rows,
        columns=["clump_id", "filter", "flux", "flux_err", "n_pix"])

    return df_flux


def sed_fitting_dataframe(
    df_flux,
    filter_order=None):
    """
    Prepare a wide-format table for SED fitting.

    Parameters
    ----------
    df_flux : pandas.DataFrame
        Long-format table with columns ``clump_id, filter, flux,
        flux_err`` (output of :func:`compute_clump_photometry`, possibly
        concatenated over several cubes).
    filter_order : list, optional
        Desired order of filters (e.g. :data:`DEFAULT_FILTER_ORDER`).
        Filters not present in ``df_flux`` are ignored; filters missing
        from the list are dropped.

    Returns
    -------
    df_sed : pandas.DataFrame
        One row per clump with columns ``clump_id``, one flux column per
        filter and one ``err_<filter>`` column per filter.
    """

    df_flux_wide = df_flux.pivot(
        index="clump_id",
        columns="filter",
        values="flux"
    )

    df_err_wide = df_flux.pivot(
        index="clump_id",
        columns="filter",
        values="flux_err"
    )

    if filter_order is not None:

        flux_order = [f for f in filter_order if f in df_flux_wide.columns]
        err_order = [f for f in filter_order if f in df_err_wide.columns]

        df_flux_wide = df_flux_wide[flux_order]
        df_err_wide = df_err_wide[err_order]

    df_err_wide = df_err_wide.add_prefix("err_")

    df_sed = pd.concat([df_flux_wide, df_err_wide], axis=1)

    df_sed = df_sed.reset_index()

    df_sed.columns.name = None

    return df_sed
