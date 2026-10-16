"""
Clump detection over a list of galaxies.

For every galaxy: build the galaxy masks, detect clumps in each filter,
match them across filters, measure sizes, classify them as disk/outside
and save

* ``<galaxy>_properties.csv`` – one row per clump
* ``<galaxy>_pixels.csv``     – one row per clump pixel (x, y, ra, dec)
* ``clumps_<galaxy>_pixels.geojson`` / ``clumps_<galaxy>_sky.geojson``
* diagnostic figures

Edit ``PARAMS``, ``GALAXIES`` and ``OUTPUT_ROOT`` and run::

    python run_detection.py
"""

from pathlib import Path

import folklore as fm

PARAMS = {
    # galaxy masks (lowest percentile = search region, highest = centre)
    "percentiles": (30, 90),

    # galaxy body region (disk/outside classification)
    "region_sigma_smooth": 1.5,
    "region_nsigma": 3,
    "region_nlevels": 60,
    "region_connectivity": 4,

    # contrast map
    "sigma_smooth": None,
    "bkg_size": 15,
    "sigma_size": 15,
    "vmax_contrast": 10,

    # detection
    "sigma_center": 3.0,
    "sigma_outer": 2.5,
    "sigma_split": None,
    "min_area": 3,

    # cross-filter matching
    "valid_filters": None,  # e.g. ["F070W", "F090W", "F115W", "F150W"]
    "min_filters": 3,
    "min_overlap": 0.20,
}

# Abell S1063
GALAXIES = {
    "J1": {
        "z": 0.352,
        "ra": "22:48:40.15",
        "dec": "-44:30:50.26",
        "file": "datacubes/abells1063/J1/datacube_J1_jwst_nircam_sw_psfmatched.fits",
        # "mask_level": -2,  # galaxy body mask level (default -1)
        # "params": {"sigma_outer": 3.0},  # per-galaxy overrides
    },
    "J2": {
        "z": 0.355,
        "ra": "22:48:48.897",
        "dec": "-44:32:32.202",
        "file": "datacubes/abells1063/J2/datacube_J2_jwst_nircam_sw_psfmatched.fits",
    },
}

OUTPUT_ROOT = Path("output/detection/abells1063")

PLOT = dict(show=False, save=True, dpi=200)


def detect_galaxy(galaxy, info, params, output_dir):

    data_cube, header, wcs, filters = fm.load_datacube(info["file"])

    # 1. galaxy masks
    galaxy_masks = fm.build_galaxy_masks(
        fm.collapse_cube(data_cube),
        percentiles=params["percentiles"])

    fm.plot_galaxy_masks(
        galaxy_masks, output_path=output_dir / "galaxy_masks.png", **PLOT)
    fm.plot_masks_filters(
        data_cube, filters, galaxy_masks,
        output_path=output_dir / "galaxy_masks_filters.png", **PLOT)

    # 2. contrast diagnostics
    masks = galaxy_masks["masks"]
    gal_mask = masks[min(masks.keys())]

    contrast_kw = dict(
        sigma_smooth=params["sigma_smooth"],
        bkg_size=params["bkg_size"],
        sigma_size=params["sigma_size"])

    for filt, idx in filters.items():

        fm.plot_contrast_steps(
            data_cube[idx], gal_mask, filt_name=filt,
            vmax_contrast=params["vmax_contrast"], **contrast_kw,
            output_path=output_dir / f"contrast_map_{filt}.png", **PLOT)

        fm.plot_contrast_levels(
            data_cube[idx], gal_mask, filt_name=filt,
            levels=[2, 4, 6, 8], colors=["pink", "cyan", "red", "green"],
            **contrast_kw,
            output_path=output_dir / f"contrast_levels_{filt}.png", **PLOT)

    # 3. per-filter detection
    detection_kw = dict(
        sigma_center=params["sigma_center"],
        sigma_outer=params["sigma_outer"],
        sigma_split=params["sigma_split"],
        **contrast_kw)

    fm.plot_candidate_pixels(
        data_cube, filters, galaxy_masks, **detection_kw,
        output_path=output_dir / "candidate_pixels.png", **PLOT)

    clumps_by_filter, pixels_by_filter, pixels_all_filters = fm.build_clump_dataframes(
        data_cube, filters, galaxy_masks,
        min_area=params["min_area"], **detection_kw)

    fm.plot_clumps_filters(
        data_cube, filters, pixels_by_filter,
        output_path=output_dir / "clump_filters.png", **PLOT)

    # 4. cross-filter matching
    valid_filters = params["valid_filters"] or list(filters)

    pixels_sel = pixels_all_filters[pixels_all_filters["filter"].isin(valid_filters)]

    df_overlap = fm.compute_clump_overlaps(
        fm.build_clump_pixel_sets(pixels_sel),
        min_overlap=params["min_overlap"])

    df_components = fm.find_final_clumps(
        fm.build_overlap_graph(df_overlap),
        min_filters=params["min_filters"])

    _, df_pixels_final = fm.build_final_clump_dataframes(
        clumps_by_filter, pixels_by_filter, df_components)

    fm.plot_final_clumps(
        data_cube, filters, df_pixels_final,
        output_path=output_dir / "final_clumps.png", **PLOT)

    union_masks = fm.build_union_masks(df_pixels_final, data_cube.shape[1:])

    fm.plot_union_clumps(
        data_cube, filters, union_masks,
        output_path=output_dir / "union_clumps.png", **PLOT)

    # 5. properties and classification
    df_union_pixels, df_union_clumps = fm.compute_clump_geometry(
        union_masks,
        pixel_scale_arcsec=fm.pixel_scale_arcsec(wcs),
        wcs=fm.get_celestial_wcs(wcs))

    df_union_clumps = fm.add_physical_sizes(df_union_clumps, z=info["z"])

    center = fm.skycoord_to_pixel(info["ra"], info["dec"], wcs)

    image = fm.collapse_cube(data_cube)

    the_galaxy = fm.build_galaxy_region(
        image,
        center=center,
        sigma_smooth=params["region_sigma_smooth"],
        nsigma=params["region_nsigma"],
        nlevels=params["region_nlevels"],
        connectivity=params["region_connectivity"])

    fm.plot_galaxy_region_levels(
        image, the_galaxy, union_masks, df_union_clumps,
        output_path=output_dir / "galaxy_region_levels.png", **PLOT)

    # level of the galaxy body mask, chosen by eye in
    # galaxy_region_levels.png ("mask_level" in GALAXIES; -1 = last level)
    the_galaxy_mask = the_galaxy["masks"][info.get("mask_level", -1)]

    df_union_clumps = fm.classify_clumps(df_union_clumps, the_galaxy_mask)

    fm.plot_classify_clumps(
        data_cube, filters, union_masks, the_galaxy_mask, df_union_clumps,
        output_path=output_dir / "classify_clumps.png", **PLOT)

    # 6. outputs
    fm.build_clump_geodataframe(union_masks, df_union_clumps).to_file(
        output_dir / f"clumps_{galaxy}_pixels.geojson", driver="GeoJSON")

    fm.build_clump_geodataframe_sky(union_masks, df_union_clumps, wcs).to_file(
        output_dir / f"clumps_{galaxy}_sky.geojson", driver="GeoJSON")

    df_union_clumps.to_csv(output_dir / f"{galaxy}_properties.csv", index=False)
    df_union_pixels.to_csv(output_dir / f"{galaxy}_pixels.csv", index=False)

    print(f"{galaxy}: {len(df_union_clumps)} clumps")


if __name__ == "__main__":

    for galaxy, info in GALAXIES.items():

        print("=" * 60)
        print(f"Processing {galaxy}")
        print("=" * 60)

        params = {**PARAMS, **info.get("params", {})}

        output_dir = OUTPUT_ROOT / galaxy
        output_dir.mkdir(parents=True, exist_ok=True)

        detect_galaxy(galaxy, info, params, output_dir)
