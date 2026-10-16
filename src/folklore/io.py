"""
Reading and writing FITS data cubes, and WCS helpers.

FOLKLORE works on photometric data cubes with shape ``(n_filters, ny, nx)``
whose header lists the filter of every slice in the keywords ``FILTER1``,
``FILTER2``, ... (the format produced by
`lightstack <https://lightstack.readthedocs.io>`_).
"""

import numbers
import re
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

__all__ = [
    "find_ext",
    "load_datacube",
    "save_individual_clump_cubes",
    "save_masked_clump_cubes",
    "get_celestial_wcs",
    "skycoord_to_pixel",
    "pixel_scale_arcsec",
]


def find_ext(hdul):
    """
    Find the first FITS extension containing valid image data.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        Opened FITS file.

    Returns
    -------
    ext : int or None
        Index of the HDU containing image data. Returns None if not found.
    """
    for i, hdu in enumerate(hdul):
        if hdu.data is not None and isinstance(hdu.data, np.ndarray):
            return i
    return None


def load_datacube(filename):
    """
    Load a FITS data cube and extract the filter-to-slice mapping.

    The filter names are read from the header keywords ``FILTER1``,
    ``FILTER2``, ..., where ``FILTERn`` is the filter of slice ``n - 1``.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the FITS data cube.

    Returns
    -------
    data_cube : ndarray
        Array with shape (n_filters, ny, nx).
    header : astropy.io.fits.Header
        Header of the HDU holding the cube.
    wcs : astropy.wcs.WCS
        WCS built from ``header``.
    filters : dict
        Mapping ``{filter_name: index_in_cube}``.

    Raises
    ------
    ValueError
        If no HDU with image data is found.
    """

    with fits.open(filename) as hdul:

        ext = find_ext(hdul)

        if ext is None:
            raise ValueError("No datacube found")

        # copy into memory so the array stays valid after the file is closed
        data_cube = np.array(hdul[ext].data)
        header = hdul[ext].header.copy()
        wcs = WCS(header)

    filters = {}

    for key in header:
        m = re.match(r"FILTER(\d+)$", key)
        if m:
            idx = int(m.group(1)) - 1
            filters[header[key].strip()] = idx

    return data_cube, header, wcs, filters


def save_masked_clump_cubes(
    data_cube,
    header,
    expanded_masks,
    output_dir,
    clump_ids=None):
    """
    Save one full-size masked photometric cube for each clump.

    Every cube has the same shape and header as the input cube, with all
    pixels outside the clump mask set to NaN. See
    :func:`save_individual_clump_cubes` for a cropped version.

    Parameters
    ----------
    data_cube : ndarray
        Photometric data cube with shape (n_filters, ny, nx).
    header : astropy.io.fits.Header
        FITS header from the original photometric cube.
    expanded_masks : dict
        Dictionary ``{clump_id: boolean mask}`` (for example the output of
        :func:`folklore.photometry.build_expanded_clump_masks`).
    output_dir : str or pathlib.Path
        Directory where the individual FITS files will be saved.
    clump_ids : iterable, optional
        Clump IDs to save. If None, all clumps are saved.

    Returns
    -------
    output_files : dict
        Dictionary ``{clump_id: output_path}``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if clump_ids is None:
        clump_ids = expanded_masks.keys()

    output_files = {}

    for cid in clump_ids:

        mask = expanded_masks[cid]

        masked_cube = data_cube.copy()
        masked_cube[:, ~mask] = np.nan

        output_path = output_dir / f"clump_{cid}.fits"

        hdu = fits.PrimaryHDU(data=masked_cube, header=header.copy())
        hdu.writeto(output_path, overwrite=True)

        output_files[cid] = output_path

        print(f"Clump {cid}: {np.sum(mask)} pixels → {output_path}")

    return output_files


def save_individual_clump_cubes(
    data_cube,
    header,
    expanded_masks,
    output_dir,
    clump_ids=None,
    padding=2):
    """
    Save one cropped photometric cube for each clump (bounding-box cutout).

    The cube is cropped to the bounding box of the clump mask (plus
    ``padding`` pixels), pixels outside the mask are set to NaN and the
    header reference pixel (``CRPIX1``/``CRPIX2``) is shifted so the WCS
    stays valid.

    Parameters
    ----------
    data_cube : ndarray
        Photometric data cube with shape (n_filters, ny, nx).
    header : astropy.io.fits.Header
        FITS header from the original photometric cube.
    expanded_masks : dict
        Dictionary ``{clump_id: boolean mask}``.
    output_dir : str or pathlib.Path
        Directory where the individual FITS files will be saved.
    clump_ids : iterable, optional
        Clump IDs to save. If None, all clumps are saved.
    padding : int, optional
        Extra pixels around the bounding box. Default is 2.

    Returns
    -------
    output_files : dict
        Dictionary ``{clump_id: output_path}``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if clump_ids is None:
        clump_ids = expanded_masks.keys()

    output_files = {}

    for cid in clump_ids:

        mask = expanded_masks[cid]

        # bounding box of the mask pixels
        rows, cols = np.where(mask)

        if len(rows) == 0:
            print(f"Clump {cid}: empty mask, skipping.")
            continue

        ny, nx = mask.shape

        y_min = max(rows.min() - padding, 0)
        y_max = min(rows.max() + padding + 1, ny)
        x_min = max(cols.min() - padding, 0)
        x_max = min(cols.max() + padding + 1, nx)

        # crop cube and mask
        cropped_cube = data_cube[:, y_min:y_max, x_min:x_max].copy()
        cropped_mask = mask[y_min:y_max, x_min:x_max]

        # apply the mask inside the cutout
        cropped_cube[:, ~cropped_mask] = np.nan

        # shift the WCS reference pixel (FITS axis 1 = columns, axis 2 = rows)
        new_header = header.copy()
        if "CRPIX1" in new_header:
            new_header["CRPIX1"] -= x_min
        if "CRPIX2" in new_header:
            new_header["CRPIX2"] -= y_min

        output_path = output_dir / f"clump_{cid}.fits"

        hdu = fits.PrimaryHDU(data=cropped_cube, header=new_header)
        hdu.writeto(output_path, overwrite=True)

        output_files[cid] = output_path

        print(
            f"Clump {cid}: {np.sum(mask)} pixels | "
            f"cutout {cropped_cube.shape} | "
            f"→ {output_path}"
        )

    return output_files


def get_celestial_wcs(wcs):
    """
    Return a 2D celestial WCS.

    Data cubes usually carry a 3D WCS (RA, Dec, filter). This function drops
    the non-celestial axes. If the input WCS is already 2D, it is returned
    unchanged.

    Parameters
    ----------
    wcs : astropy.wcs.WCS or None
        WCS of an image (2D) or data cube (3D).

    Returns
    -------
    astropy.wcs.WCS or None
        Celestial (RA, Dec) part of ``wcs``; None if ``wcs`` is None.
    """

    if wcs is None:
        return None

    if wcs.pixel_n_dim > 2:
        return wcs.celestial

    return wcs


def skycoord_to_pixel(ra, dec, wcs):
    """
    Convert sky coordinates (RA, Dec) to pixel coordinates.

    Parameters
    ----------
    ra, dec : float or str
        Sky coordinates. Numbers are interpreted as decimal degrees and
        strings as sexagesimal (RA in hours, e.g. ``'0:14:22.3841'``;
        Dec in degrees, e.g. ``'-30:23:50.2'``).
    wcs : astropy.wcs.WCS
        WCS of the image or cube.

    Returns
    -------
    x, y : float
        Pixel coordinates (0-based).
    """

    if isinstance(ra, numbers.Number):
        ra_unit = u.deg
    else:
        ra_unit = u.hourangle

    coord = SkyCoord(ra, dec, unit=(ra_unit, u.deg), frame="icrs")

    wcs = get_celestial_wcs(wcs)

    x, y = wcs.world_to_pixel_values(coord.ra.deg, coord.dec.deg)

    return float(x), float(y)


def pixel_scale_arcsec(wcs):
    """
    Pixel scale of an image or cube in arcsec/pixel.

    Uses the mean of the two celestial axes, so it works for both
    ``CDELT``/``PC`` and ``CD`` style headers.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        WCS of the image or cube.

    Returns
    -------
    float
        Pixel scale in arcsec/pixel.
    """

    scales_deg = proj_plane_pixel_scales(get_celestial_wcs(wcs))

    return float(np.mean(scales_deg) * 3600)
