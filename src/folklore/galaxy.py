"""
Galaxy-scale masks.

Two kinds of masks are built here:

* **Percentile masks** (:func:`build_galaxy_masks`) on the collapsed cube.
  The lowest percentile defines the region where clumps are searched for;
  the highest one defines the bright central region, where a stricter
  detection threshold is used.
* **Galaxy body region** (:func:`build_galaxy_region`), grown from the
  galaxy centre by lowering an intensity threshold. It is used to classify
  clumps as belonging to the disk or lying outside it. It requires the
  coordinates of the galaxy centre; build it on the collapsed cube.
"""

import numpy as np
import pandas as pd
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter, label

__all__ = [
    "collapse_cube",
    "percentile_mask",
    "build_galaxy_masks",
    "build_galaxy_region",
]


def collapse_cube(data_cube, method="median"):
    """
    Collapse a photometric data cube into a single 2D image.

    The cube is collapsed along the filter axis (axis=0) using either the
    median or the sum of the pixel values across filters. This collapsed
    image is typically used to estimate global galaxy properties such as
    masks or morphological structures.

    Parameters
    ----------
    data_cube : ndarray
        3D array with shape (n_filters, ny, nx) containing the image in
        each filter.
    method : {"median", "sum"}, optional
        Method used to collapse the cube:

        - "median": median value across filters (default). More robust
          against outliers and noise.
        - "sum": sum of flux across filters.

    Returns
    -------
    image : ndarray
        2D array (ny, nx) representing the collapsed image.

    Raises
    ------
    ValueError
        If ``method`` is not one of the allowed options.
    """

    if method == "median":
        image = np.nanmedian(data_cube, axis=0)

    elif method == "sum":
        image = np.nansum(data_cube, axis=0)

    else:
        raise ValueError("method must be 'median' or 'sum'")

    return image


def percentile_mask(
    image,
    percentile,
    sigma=3,
    opening_iter=2,
    closing_iter=3):
    """
    Build a binary mask based on a percentile threshold of a smoothed image.

    The input image is first smoothed using a Gaussian filter to suppress
    small-scale noise. A threshold is then defined using the specified
    percentile of the smoothed pixel values. Pixels above this threshold
    are included in the mask. Morphological closing and opening operations
    are applied to fill small holes, remove small artifacts and smooth the
    mask boundaries.

    Parameters
    ----------
    image : ndarray
        2D input image.
    percentile : float
        Percentile used to define the intensity threshold. Pixels with
        values above this percentile (in the smoothed image) are included
        in the mask.
    sigma : float, optional
        Standard deviation of the Gaussian kernel used to smooth the image.
        Default is 3.
    opening_iter : int, optional
        Number of iterations of the morphological opening operation used
        to remove small isolated structures. Default is 2.
    closing_iter : int, optional
        Number of iterations of the morphological closing operation used
        to fill small holes in the mask. Default is 3.

    Returns
    -------
    mask : ndarray of bool
        Binary mask identifying pixels above the percentile threshold.
    smooth : ndarray
        Smoothed version of the input image.
    threshold : float
        Pixel value corresponding to the chosen percentile in the smoothed
        image.
    """

    smooth = gaussian_filter(image, sigma=sigma)

    threshold = np.nanpercentile(smooth, percentile)

    mask = smooth > threshold

    mask = binary_closing(mask, iterations=closing_iter)
    mask = binary_opening(mask, iterations=opening_iter)

    return mask, smooth, threshold


def build_galaxy_masks(
    image,
    percentiles=(50, 95),
    sigma=3,
    opening_iter=2,
    closing_iter=3):
    """
    Generate galaxy masks from a collapsed image using percentile thresholds.

    Multiple masks are constructed from the same image using different
    percentile thresholds. The detection functions use the mask of the
    **lowest** percentile as the galaxy region and the mask of the
    **highest** percentile as the central region.

    Parameters
    ----------
    image : ndarray
        2D collapsed image of the galaxy (see :func:`collapse_cube`).
    percentiles : sequence of float, optional
        Percentile values used to generate masks. Each percentile will
        produce one mask. Default is (50, 95).
    sigma : float, optional
        Gaussian smoothing kernel standard deviation used before
        thresholding. Default is 3.
    opening_iter : int, optional
        Number of iterations for morphological opening. Default is 2.
    closing_iter : int, optional
        Number of iterations for morphological closing. Default is 3.

    Returns
    -------
    results : dict
        Dictionary containing:

        - "image": the input collapsed image
        - "smooth": the smoothed image
        - "masks": dictionary mapping percentile → mask
        - "thresholds": dictionary mapping percentile → threshold value
    """

    masks = {}
    thresholds = {}

    for p in percentiles:

        mask, smooth, thr = percentile_mask(
            image,
            percentile=p,
            sigma=sigma,
            opening_iter=opening_iter,
            closing_iter=closing_iter)

        masks[p] = mask
        thresholds[p] = thr

    results = {
        "image": image,
        "smooth": smooth,
        "masks": masks,
        "thresholds": thresholds}

    return results


def build_galaxy_region(
    image,
    center,
    sigma_smooth=2,
    nsigma=2,
    nlevels=60,
    connectivity=8,
    min_fraction=None,
    growth_factor=None,
    min_area_check=100,
):
    """
    Build the galaxy region by progressively lowering the intensity threshold.

    Starting from a known galaxy center, the function follows the connected
    component containing that center while decreasing the threshold from the
    image peak toward the background, and returns one mask per threshold
    level.

    Two ways of choosing the final mask:

    * **By eye (default).** The lowest threshold is background +
      ``nsigma`` × std. Inspect the last levels with
      :func:`folklore.plotting.plot_galaxy_region_levels` and pick one,
      e.g. ``region["masks"][-2]``. ``region["mask"]`` is the last level.
    * **Automatic (optional).** Set ``min_fraction`` and/or
      ``growth_factor``: the lowest threshold becomes a fraction of the
      peak, and the growth stops at the level before the area increases by
      more than ``growth_factor`` times (the mask leaking into the tail or
      the background). ``region["mask"]`` is then the selected mask. Check
      the result: a single pair of parameters does not suit every galaxy.

    Use the collapsed cube (:func:`collapse_cube`) as ``image``.

    Parameters
    ----------
    image : ndarray
        2D image.
    center : tuple of float
        (x, y) pixel coordinates of the galaxy centre
        (see :func:`folklore.io.skycoord_to_pixel`).
    sigma_smooth : float, optional
        Gaussian smoothing applied to the image before thresholding.
    nsigma : float, optional
        Lowest threshold = background + nsigma * std, with background and
        std from sigma-clipped statistics of the smoothed image.
    nlevels : int, optional
        Number of threshold levels between the image peak and the lowest
        threshold.
    connectivity : {4, 8}, optional
        Pixel connectivity used to label connected regions.
    min_fraction : float, optional
        If given, the lowest threshold is ``min_fraction`` × the peak of
        the smoothed image instead of background + ``nsigma`` × std
        (e.g. 0.02).
    growth_factor : float, optional
        If given, stop at the level before the mask area grows by more
        than this factor from one level to the next (e.g. 1.5).
    min_area_check : int, optional
        The ``growth_factor`` criterion is only applied once the mask has
        at least this many pixels; near the peak the area naturally doubles
        between levels. Default is 100.

    Returns
    -------
    region : dict
        Dictionary containing:

        - "mask": the selected mask (the last level, or the level before
          the area jump when ``growth_factor`` is used); None if no mask
          contains the centre
        - "stop_reason": "area jump" or "last level"
        - "masks": list of boolean masks, from the highest to the lowest
          threshold (only levels where the centre is inside a region; with
          ``growth_factor``, only up to the selected mask)
        - "properties": DataFrame with columns ``level``, ``threshold``
          and ``area`` for each mask
        - "thresholds": all threshold values tested
    """

    img = gaussian_filter(image, sigma_smooth)

    threshold_max = np.nanmax(img)

    if min_fraction is not None:
        threshold_min = threshold_max * min_fraction
    else:
        _, median, std = sigma_clipped_stats(img)
        threshold_min = median + nsigma * std

    thresholds = np.linspace(
        threshold_max,
        threshold_min,
        nlevels,
    )

    x0 = int(round(center[0]))
    y0 = int(round(center[1]))

    if connectivity == 8:
        structure = np.ones((3, 3), dtype=int)
    else:
        structure = np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]]
        )

    masks = []
    rows = []
    stop_reason = "last level"

    center_inside = 0 <= x0 < img.shape[1] and 0 <= y0 < img.shape[0]

    for i, threshold in enumerate(thresholds):

        if not center_inside:
            break

        binary = img >= threshold

        labels, _ = label(binary, structure=structure)

        lab = labels[y0, x0]

        if lab == 0:
            continue

        mask = labels == lab

        if (
            growth_factor is not None
            and masks
            and rows[-1]["area"] >= min_area_check
            and mask.sum() > rows[-1]["area"] * growth_factor
        ):
            stop_reason = "area jump"
            break

        rows.append(dict(
            level=i,
            threshold=threshold,
            area=mask.sum(),
        ))

        masks.append(mask)

    properties = pd.DataFrame(rows, columns=["level", "threshold", "area"])

    return dict(
        mask=masks[-1] if masks else None,
        stop_reason=stop_reason,
        masks=masks,
        properties=properties,
        thresholds=thresholds,
    )
