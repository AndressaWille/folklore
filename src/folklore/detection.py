"""
Per-filter clump detection.

The core of the detection is the local contrast map: each pixel value is
expressed as the local excess over a median-filtered background in units
of the local noise::

    contrast = (image - background) / sigma_local

For every filter of the cube:

1. build the local contrast map (:func:`build_contrast_map`);
2. select candidate pixels above a significance threshold, stricter in the
   galaxy centre (:func:`select_candidate_pixels`);
3. label connected regions, optionally splitting regions that contain
   several brighter peaks (:func:`refine_hierarchical`);
4. discard regions smaller than ``min_area`` and tabulate the rest
   (:func:`build_clump_dataframes`).
"""

import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass, gaussian_filter, label, median_filter

__all__ = [
    "build_contrast_map",
    "select_candidate_pixels",
    "refine_hierarchical",
    "build_clump_dataframes",
]

CLUMP_COLUMNS = ["filter", "label", "area_pix", "x0", "y0"]
PIXEL_COLUMNS = ["filter", "label", "x", "y"]


def _galaxy_and_center_masks(galaxy_masks):
    """Return (galaxy mask, central mask) from ``build_galaxy_masks`` output."""
    masks = galaxy_masks["masks"]
    gal_mask = masks[min(masks.keys())]
    gal_mask_center = masks[max(masks.keys())]
    return gal_mask, gal_mask_center


def build_contrast_map(
    imdata,
    gal_mask,
    sigma_smooth=None,
    bkg_size=20,
    sigma_size=20):
    """
    Build a local contrast map for a single image.

    Parameters
    ----------
    imdata : ndarray
        2D input image.
    gal_mask : ndarray of bool
        Galaxy mask defining the valid region. The contrast map is NaN
        outside it.
    sigma_smooth : float, optional
        Gaussian smoothing applied before background estimation. None or 0
        disables smoothing.
    bkg_size : int, optional
        Window size (pixels) of the median filter used to estimate the
        background.
    sigma_size : int, optional
        Window size (pixels) used to compute the local noise. The local
        noise is the square root of the median-filtered squared residuals.

    Returns
    -------
    contrast_map : ndarray
        Local contrast map (image_sub / sigma_local).
    background : ndarray
        Estimated background.
    image_sub : ndarray
        Background-subtracted image.
    sigma_local : ndarray
        Local noise estimate.
    """

    # optional smoothing
    if sigma_smooth is not None and sigma_smooth > 0:
        image_smooth = gaussian_filter(imdata, sigma=sigma_smooth)
    else:
        image_smooth = imdata

    # background estimation
    background = median_filter(image_smooth, size=bkg_size)

    # background subtraction
    image_sub = image_smooth - background

    # local noise estimate
    sigma_local = np.sqrt(
        median_filter(image_sub**2, size=sigma_size))

    sigma_local[sigma_local == 0] = np.nan

    # contrast map
    contrast_map = image_sub / sigma_local

    # restrict to galaxy region
    contrast_map[~gal_mask] = np.nan

    return contrast_map, background, image_sub, sigma_local


def select_candidate_pixels(
    contrast_map,
    gal_mask,
    gal_mask_center,
    sigma_center=3.5,
    sigma_outer=4.0):
    """
    Select candidate pixels based on local contrast significance.

    Different sigma thresholds are used for the galaxy center and for the
    outer regions.

    Parameters
    ----------
    contrast_map : ndarray
        Local contrast map (signal / local noise).
    gal_mask : ndarray of bool
        Mask defining the full galaxy region.
    gal_mask_center : ndarray of bool
        Mask defining the central region of the galaxy.
    sigma_center : float, optional
        Detection threshold (sigma) used inside the central region.
    sigma_outer : float, optional
        Detection threshold (sigma) used in the outer galaxy region.

    Returns
    -------
    candidate_pixels : ndarray of bool
        Boolean map marking candidate clump pixels.
    """

    # ensure boolean masks
    gal_mask = gal_mask.astype(bool)
    gal_mask_center = gal_mask_center.astype(bool)

    # ignore invalid contrast values
    valid = np.isfinite(contrast_map)

    # central region candidates
    center_candidates = (
        (contrast_map > sigma_center)
        & gal_mask_center
        & valid)

    # outer region mask
    outer_mask = gal_mask & (~gal_mask_center)

    # outer region candidates
    outer_candidates = (
        (contrast_map > sigma_outer)
        & outer_mask
        & valid)

    # combine candidates
    candidate_pixels = center_candidates | outer_candidates

    return candidate_pixels


def refine_hierarchical(
    contrast_map,
    candidate_pixels,
    sigma_split=6.0):
    """
    Refine candidate clumps using hierarchical thresholding.

    Each connected candidate region is re-thresholded at ``sigma_split``.
    If this produces two or more separate peaks, the region is replaced by
    those peaks; otherwise the original region is kept.

    Parameters
    ----------
    contrast_map : ndarray
        2D contrast map (e.g., (image - background) / sigma).
    candidate_pixels : ndarray of bool
        Binary mask of candidate clump pixels.
    sigma_split : float or None, optional
        Threshold to subdivide regions. If None, no subdivision is applied
        and the connected regions are simply labelled.

    Returns
    -------
    refined : ndarray of int
        Labeled map of clumps (0 = background).
    """

    if sigma_split is None:
        labels, _ = label(candidate_pixels)
        return labels

    labels_low, n_low = label(candidate_pixels)

    refined = np.zeros_like(candidate_pixels, dtype=int)
    current_id = 1

    for lab in range(1, n_low + 1):

        region = (labels_low == lab)

        mask_high = (contrast_map > sigma_split) & region
        labels_high, n_high = label(mask_high)

        if n_high >= 2:
            # subdivide
            for sub in range(1, n_high + 1):
                refined[labels_high == sub] = current_id
                current_id += 1
        else:
            # keep the original region
            refined[region] = current_id
            current_id += 1

    return refined


def build_clump_dataframes(
    data_cube,
    filters,
    galaxy_masks,
    sigma_center=3.5,
    sigma_outer=4.0,
    sigma_split=6.0,
    min_area=3,
    sigma_smooth=None,
    bkg_size=20,
    sigma_size=20,
    verbose=True):
    """
    Detect clumps in every filter and tabulate them and their pixels.

    Parameters
    ----------
    data_cube : ndarray
        Data cube with shape (n_filters, ny, nx).
    filters : dict
        Dictionary mapping filter_name → cube index.
    galaxy_masks : dict
        Output from :func:`folklore.galaxy.build_galaxy_masks`. The lowest
        percentile mask is the search region and the highest one is the
        central region.
    sigma_center : float, optional
        Sigma threshold for the galaxy center.
    sigma_outer : float, optional
        Sigma threshold for outer galaxy regions.
    sigma_split : float or None, optional
        Threshold for hierarchical splitting (see
        :func:`refine_hierarchical`). None disables splitting.
    min_area : int, optional
        Minimum number of pixels required for a clump.
    sigma_smooth : float or None, optional
        Gaussian smoothing applied before contrast estimation.
    bkg_size : int, optional
        Window size for background estimation.
    sigma_size : int, optional
        Window size for local noise estimation.
    verbose : bool, optional
        Print the number of candidate regions per filter.

    Returns
    -------
    clumps_by_filter : dict
        ``{filter_name: DataFrame}`` with one row per clump and columns
        ``filter, label, area_pix, x0, y0``.
    pixels_by_filter : dict
        ``{filter_name: DataFrame}`` with one row per clump pixel and
        columns ``filter, label, x, y``.
    pixels_all_filters : pandas.DataFrame
        Concatenation of all tables in ``pixels_by_filter``.
    """

    gal_mask, gal_mask_center = _galaxy_and_center_masks(galaxy_masks)

    clumps_by_filter = {}
    pixels_by_filter = {}

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

        labeled_map = refine_hierarchical(
            contrast_map,
            candidate_pixels,
            sigma_split=sigma_split)

        n_labels = labeled_map.max()

        if verbose:
            print(f"{filt}: candidate regions = {n_labels}")

        clump_rows = []
        pixel_tables = []

        for lab in range(1, n_labels + 1):

            mask = labeled_map == lab
            area = np.sum(mask)

            if area < min_area:
                continue

            y_pix, x_pix = np.nonzero(mask)

            pixel_tables.append(pd.DataFrame({
                "filter": filt,
                "label": lab,
                "x": x_pix,
                "y": y_pix}))

            y0, x0 = center_of_mass(mask)

            clump_rows.append({
                "filter": filt,
                "label": lab,
                "area_pix": area,
                "x0": x0,
                "y0": y0})

        clumps_by_filter[filt] = pd.DataFrame(clump_rows, columns=CLUMP_COLUMNS)

        if pixel_tables:
            pixels_by_filter[filt] = pd.concat(pixel_tables, ignore_index=True)
        else:
            pixels_by_filter[filt] = pd.DataFrame(columns=PIXEL_COLUMNS)

    pixels_all_filters = pd.concat(pixels_by_filter.values(), ignore_index=True)

    return clumps_by_filter, pixels_by_filter, pixels_all_filters
