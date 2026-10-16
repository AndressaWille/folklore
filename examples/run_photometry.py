"""
Clump photometry over a list of galaxies.

Reads the clump pixel tables written by ``run_detection.py`` and measures
background-subtracted fluxes in every cube listed for the galaxy (cubes
may have different pixel grids: the clumps are transferred through their
sky coordinates). Writes ``<galaxy>_sed_input.csv`` with one row per clump
and one flux/error column per filter (in Jy).

Edit ``PARAMS``, ``GALAXIES`` and ``OUTPUT_ROOT`` and run::

    python run_photometry.py
"""

from pathlib import Path

import astropy.units as u
import pandas as pd

import folklore as fm

PARAMS = {
    # background ring, in pixels of each cube
    "inner_radius": 0,
    "outer_radius": 2,
    # flux of one pixel value; check the units of each cube!
    "flux_unit": 10 * u.nJy,
}

DETECTION_ROOT = Path("output/detection/abells1063")
OUTPUT_ROOT = Path("output/photometry/abells1063")

# Abell S1063
GALAXIES = {
    "J1": {
        "cubes": {
            "wfc3_ir": "datacubes/abells1063/J1/datacube_J1_hst_wfc3_ir.fits",
            "jwst": "datacubes/abells1063/J1/datacube_J1_jwst_nircam_all.fits",
        },
    },
    "J2": {
        "cubes": {
            "wfc3_ir": "datacubes/abells1063/J2/datacube_J2_hst_wfc3_ir.fits",
            "jwst": "datacubes/abells1063/J2/datacube_J2_jwst_nircam_all.fits",
        },
    },
}

PLOT = dict(show=False, save=True, dpi=200)


def measure_galaxy(galaxy, info, params, output_dir):

    df_union_pixels = pd.read_csv(DETECTION_ROOT / galaxy / f"{galaxy}_pixels.csv")

    all_fluxes = []

    for cube_name, cube_path in info["cubes"].items():

        if cube_path is None:
            continue

        print(f"Processing {cube_name}")

        data_cube, header, wcs, filters = fm.load_datacube(cube_path)

        union_masks = fm.reproject_clump_masks_sky(
            df_union_pixels, wcs, data_cube.shape[1:])

        fm.plot_union_clumps(
            data_cube, filters, union_masks,
            output_path=output_dir / f"{cube_name}_union_clumps.png", **PLOT)

        ring_masks = fm.build_background_rings(
            union_masks,
            inner_radius=params["inner_radius"],
            outer_radius=params["outer_radius"])

        fm.plot_clump_rings(
            data_cube, filters, union_masks, ring_masks,
            output_path=output_dir / f"{cube_name}_rings.png", **PLOT)

        df_flux = fm.compute_clump_photometry(
            union_masks, ring_masks, data_cube, filters,
            flux_unit=params["flux_unit"])

        df_flux["instrument"] = cube_name

        all_fluxes.append(df_flux)

    df_flux_all = pd.concat(all_fluxes, ignore_index=True)

    df_sed = fm.sed_fitting_dataframe(
        df_flux_all, filter_order=fm.DEFAULT_FILTER_ORDER)

    df_sed.to_csv(output_dir / f"{galaxy}_sed_input.csv", index=False)

    print(df_sed)


if __name__ == "__main__":

    for galaxy, info in GALAXIES.items():

        print("=" * 60)
        print(f"Processing {galaxy}")
        print("=" * 60)

        params = {**PARAMS, **info.get("params", {})}

        output_dir = OUTPUT_ROOT / galaxy
        output_dir.mkdir(parents=True, exist_ok=True)

        measure_galaxy(galaxy, info, params, output_dir)
