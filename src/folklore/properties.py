"""
Clump properties: sizes, positions and disk/outside classification.
"""

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.cosmology import LambdaCDM
from scipy.ndimage import center_of_mass

__all__ = [
    "DEFAULT_COSMO",
    "compute_clump_geometry",
    "add_physical_sizes",
    "classify_clumps",
]

#: Default cosmology (flat LambdaCDM with H0=70, Om0=0.3, Ode0=0.7).
DEFAULT_COSMO = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)


def compute_clump_geometry(
    union_masks,
    pixel_scale_arcsec,
    wcs=None,
):
    """
    Compute geometric properties of the final physical clumps.

    Parameters
    ----------
    union_masks : dict
        Dictionary mapping clump_id → boolean mask.
    pixel_scale_arcsec : float
        Pixel scale in arcsec/pixel (see
        :func:`folklore.io.pixel_scale_arcsec`).
    wcs : astropy.wcs.WCS, optional
        2D celestial WCS of the image. If provided, RA and Dec are computed
        for every pixel belonging to each clump (needed later by
        :func:`folklore.photometry.reproject_clump_masks_sky`).

    Returns
    -------
    df_union_pixels : pandas.DataFrame
        Pixels belonging to each clump, with columns
        ``clump_id, x, y, ra, dec`` (``ra``/``dec`` are NaN without WCS).
    df_union_clumps : pandas.DataFrame
        One row per clump with columns
        ``clump_id, area_pix, area_arcsec2, r_eff_arcsec, x0, y0``, where
        ``r_eff_arcsec = sqrt(area / pi)`` and ``(x0, y0)`` is the centroid
        of the mask.
    """

    pixel_tables = []
    clump_rows = []

    for cid, mask_union in union_masks.items():

        y_pix, x_pix = np.nonzero(mask_union)

        if wcs is not None:
            ra, dec = wcs.pixel_to_world_values(x_pix, y_pix)
        else:
            ra = np.full(len(x_pix), np.nan)
            dec = np.full(len(x_pix), np.nan)

        pixel_tables.append(pd.DataFrame({
            "clump_id": cid,
            "x": x_pix,
            "y": y_pix,
            "ra": ra,
            "dec": dec,
        }))

        area_pix = len(x_pix)
        area_arcsec2 = area_pix * pixel_scale_arcsec**2

        r_eff_arcsec = np.sqrt(area_arcsec2 / np.pi)

        y0, x0 = center_of_mass(mask_union)

        clump_rows.append({
            "clump_id": cid,
            "area_pix": area_pix,
            "area_arcsec2": area_arcsec2,
            "r_eff_arcsec": r_eff_arcsec,
            "x0": x0,
            "y0": y0,
        })

    if pixel_tables:
        df_union_pixels = pd.concat(pixel_tables, ignore_index=True)
    else:
        df_union_pixels = pd.DataFrame(
            columns=["clump_id", "x", "y", "ra", "dec"])

    df_union_clumps = pd.DataFrame(
        clump_rows,
        columns=["clump_id", "area_pix", "area_arcsec2",
                 "r_eff_arcsec", "x0", "y0"])

    return df_union_pixels, df_union_clumps


def add_physical_sizes(
    df_union_clumps,
    z=None,
    cosmo=DEFAULT_COSMO):
    """
    Convert clump sizes from angular to physical units.

    Parameters
    ----------
    df_union_clumps : pandas.DataFrame
        Clump catalog containing ``area_arcsec2`` and ``r_eff_arcsec``.
    z : float
        Redshift of the galaxy (required).
    cosmo : astropy.cosmology.Cosmology, optional
        Cosmology used for the conversion. Default is :data:`DEFAULT_COSMO`.

    Returns
    -------
    pandas.DataFrame
        Copy of the catalog with the new columns ``area_kpc2`` and
        ``r_eff_kpc``.
    """

    if z is None:
        raise ValueError("Redshift 'z' must be provided.")

    if "area_arcsec2" not in df_union_clumps.columns:
        raise ValueError("Column 'area_arcsec2' missing.")

    if "r_eff_arcsec" not in df_union_clumps.columns:
        raise ValueError("Column 'r_eff_arcsec' missing.")

    DA = cosmo.angular_diameter_distance(z).to(u.kpc)

    kpc_per_arcsec = (DA * np.deg2rad(1/3600)).value

    df_union_clumps = df_union_clumps.copy()

    df_union_clumps["area_kpc2"] = df_union_clumps["area_arcsec2"] * kpc_per_arcsec**2
    df_union_clumps["r_eff_kpc"] = df_union_clumps["r_eff_arcsec"] * kpc_per_arcsec

    return df_union_clumps


def classify_clumps(
    df_union_clumps,
    galaxy_mask,
):
    """
    Classify clumps as belonging to the galaxy body or outside it.

    A clump is "disk" when its centroid ``(x0, y0)`` falls inside
    ``galaxy_mask`` and "outside" otherwise.

    Parameters
    ----------
    df_union_clumps : pandas.DataFrame
        Table containing at least the columns ``clump_id``, ``x0`` and
        ``y0``.
    galaxy_mask : ndarray of bool
        Mask defining the galaxy body (e.g. the last mask from
        :func:`folklore.galaxy.build_galaxy_region`).

    Returns
    -------
    pandas.DataFrame
        Copy of ``df_union_clumps`` with the columns ``inside`` (bool) and
        ``component`` ("disk" or "outside").
    """

    ny, nx = galaxy_mask.shape

    inside = []

    for _, row in df_union_clumps.iterrows():

        x = int(round(row["x0"]))
        y = int(round(row["y0"]))

        if (
            x < 0 or x >= nx or
            y < 0 or y >= ny
        ):
            inside.append(False)
            continue

        inside.append(bool(galaxy_mask[y, x]))

    df = df_union_clumps.copy()

    df["inside"] = np.array(inside, dtype=bool)

    df["component"] = np.where(
        df["inside"],
        "disk",
        "outside",
    )

    return df
