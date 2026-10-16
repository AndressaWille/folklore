"""
End-to-end tests on a synthetic data cube.

The cube contains an exponential disk plus a few Gaussian clumps (the same
in every filter) on top of Gaussian noise.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

import folklore as fm

NY, NX = 120, 120
FILTERS = ["F090W", "F150W", "F200W"]
CLUMPS_XY = [(45, 60), (75, 52), (60, 85)]


def _make_wcs():
    w = WCS(naxis=3)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN", "WAVE"]
    w.wcs.crval = [150.0, 2.0, 1.0]
    w.wcs.crpix = [NX / 2, NY / 2, 1]
    w.wcs.cdelt = [-0.04 / 3600, 0.04 / 3600, 1]
    w.wcs.cunit = ["deg", "deg", ""]
    return w


@pytest.fixture(scope="module")
def cube_file(tmp_path_factory):
    rng = np.random.default_rng(42)
    y, x = np.mgrid[:NY, :NX]

    disk = 5 * np.exp(-np.hypot(x - 60, y - 60) / 15)
    clumps = sum(
        20 * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * 1.5**2))
        for cx, cy in CLUMPS_XY
    )

    cube = np.array([
        disk + clumps + rng.normal(0, 0.2, (NY, NX)) for _ in FILTERS
    ])

    header = _make_wcs().to_header()
    for i, f in enumerate(FILTERS):
        header[f"FILTER{i + 1}"] = f

    path = tmp_path_factory.mktemp("data") / "cube.fits"
    fits.PrimaryHDU(cube, header=header).writeto(path)
    return path


@pytest.fixture(scope="module")
def detection(cube_file):
    data_cube, header, wcs, filters = fm.load_datacube(cube_file)

    image = fm.collapse_cube(data_cube)
    galaxy_masks = fm.build_galaxy_masks(image, percentiles=(30, 90))

    clumps_by_filter, pixels_by_filter, pixels_all = fm.build_clump_dataframes(
        data_cube, filters, galaxy_masks,
        sigma_center=3.0, sigma_outer=2.5, sigma_split=None, min_area=3,
        bkg_size=15, sigma_size=15, verbose=False,
    )

    clump_pixels = fm.build_clump_pixel_sets(pixels_all)
    df_overlap = fm.compute_clump_overlaps(clump_pixels, min_overlap=0.2)
    G = fm.build_overlap_graph(df_overlap)
    df_components = fm.find_final_clumps(G, min_filters=3)
    _, df_pixels_final = fm.build_final_clump_dataframes(
        clumps_by_filter, pixels_by_filter, df_components)

    union_masks = fm.build_union_masks(df_pixels_final, data_cube.shape[1:])

    return dict(
        data_cube=data_cube, header=header, wcs=wcs, filters=filters,
        galaxy_masks=galaxy_masks, union_masks=union_masks,
    )


def test_load_datacube(cube_file):
    data_cube, header, wcs, filters = fm.load_datacube(cube_file)
    assert data_cube.shape == (len(FILTERS), NY, NX)
    assert filters == {f: i for i, f in enumerate(FILTERS)}


def test_pixel_scale(detection):
    assert fm.pixel_scale_arcsec(detection["wcs"]) == pytest.approx(0.04)


def test_skycoord_to_pixel_degrees_and_sexagesimal(detection):
    wcs = detection["wcs"]
    x1, y1 = fm.skycoord_to_pixel(150.0, 2.0, wcs)
    x2, y2 = fm.skycoord_to_pixel("10:00:00", "2:00:00", wcs)
    assert x1 == pytest.approx(NX / 2 - 1)
    assert y1 == pytest.approx(NY / 2 - 1)
    assert (x1, y1) == pytest.approx((x2, y2))


def test_detects_injected_clumps(detection):
    union_masks = detection["union_masks"]
    # the disk centre may also be picked up, so only check recovery
    assert len(union_masks) >= len(CLUMPS_XY)
    for cx, cy in CLUMPS_XY:
        assert any(m[cy, cx] for m in union_masks.values())


def test_geometry_and_classification(detection):
    union_masks = detection["union_masks"]
    pix_scale = fm.pixel_scale_arcsec(detection["wcs"])

    df_pixels, df_clumps = fm.compute_clump_geometry(
        union_masks, pix_scale, wcs=fm.get_celestial_wcs(detection["wcs"]))
    assert np.isfinite(df_pixels[["ra", "dec"]].values).all()

    df_clumps = fm.add_physical_sizes(df_clumps, z=0.3)
    assert (df_clumps["r_eff_kpc"] > 0).all()

    image = detection["data_cube"][0]
    region = fm.build_galaxy_region(image, center=(60, 60), sigma_smooth=1.5,
                                    nsigma=3, connectivity=4)
    assert region["mask"] is region["masks"][-1]
    assert region["stop_reason"] == "last level"
    df_clumps = fm.classify_clumps(df_clumps, region["masks"][-1])
    assert set(df_clumps["component"]) <= {"disk", "outside"}

    auto = fm.build_galaxy_region(image, center=(60, 60), sigma_smooth=1.5,
                                  connectivity=4, min_fraction=0.02,
                                  growth_factor=1.5, min_area_check=10)
    assert auto["mask"] is not None and auto["mask"].sum() >= 10
    assert auto["stop_reason"] in {"area jump", "last level"}

    gdf_pix = fm.build_clump_geodataframe(union_masks, df_clumps)
    gdf_sky = fm.build_clump_geodataframe_sky(union_masks, df_clumps,
                                              detection["wcs"])
    assert len(gdf_pix) == len(gdf_sky) == len(union_masks)
    minx, miny, maxx, maxy = gdf_sky.total_bounds
    assert 149.99 < minx < maxx < 150.01
    assert 1.99 < miny < maxy < 2.01


def test_photometry_roundtrip(detection):
    union_masks = detection["union_masks"]
    df_pixels, _ = fm.compute_clump_geometry(
        union_masks, 0.04, wcs=fm.get_celestial_wcs(detection["wcs"]))

    data_cube = detection["data_cube"]
    reproj = fm.reproject_clump_masks_sky(df_pixels, detection["wcs"],
                                          data_cube.shape[1:])
    for cid, mask in union_masks.items():
        np.testing.assert_array_equal(reproj[cid], mask)

    rings = fm.build_background_rings(reproj, inner_radius=0, outer_radius=2)
    df_flux = fm.compute_clump_photometry(reproj, rings, data_cube,
                                          detection["filters"], 10 * u.nJy)
    assert (df_flux["flux"] > 0).all()

    df_sed = fm.sed_fitting_dataframe(df_flux,
                                      filter_order=fm.DEFAULT_FILTER_ORDER)
    assert list(df_sed.columns) == (
        ["clump_id"] + FILTERS + [f"err_{f}" for f in FILTERS])


def test_save_clump_cubes(detection, tmp_path):
    expanded = fm.build_expanded_clump_masks(detection["union_masks"], 3)
    files = fm.save_individual_clump_cubes(
        detection["data_cube"], detection["header"], expanded, tmp_path)
    assert len(files) == len(expanded)


def test_empty_detection_does_not_crash():
    df_overlap = fm.compute_clump_overlaps({}, min_overlap=0.2)
    df_components = fm.find_final_clumps(fm.build_overlap_graph(df_overlap))
    assert df_components.empty
    assert list(df_components.columns) == ["clump_id", "filter", "label"]


def test_plots_run(detection, tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    d = detection
    kw = dict(show=False, save=True)
    fm.plot_galaxy_masks(d["galaxy_masks"], output_path=tmp_path, **kw)
    fm.plot_union_clumps(d["data_cube"], d["filters"], d["union_masks"],
                         output_path=tmp_path / "union.png", **kw)
    fm.plot_candidate_pixels(d["data_cube"], d["filters"], d["galaxy_masks"],
                             output_path=tmp_path / "cand.png", **kw)
    fm.plot_contrast_levels_filters(d["data_cube"], d["filters"], d["galaxy_masks"],
                                    output_path=tmp_path, **kw)
    image = d["data_cube"][0]
    region = fm.build_galaxy_region(image, center=(60, 60), sigma_smooth=1.5)
    _, df_clumps = fm.compute_clump_geometry(d["union_masks"], 0.03)
    fm.plot_galaxy_region_levels(image, region, d["union_masks"], df_clumps,
                                 output_path=tmp_path, **kw)

    assert (tmp_path / "plot_galaxy_masks.png").exists()
    assert (tmp_path / "union.png").exists()
    assert (tmp_path / "plot_contrast_levels_filters.png").exists()
    assert (tmp_path / "plot_galaxy_region_levels.png").exists()
    for f in FILTERS:
        assert (tmp_path / f"cand_{f}.png").exists()
