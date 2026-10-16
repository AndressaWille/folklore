Changelog
=========

0.1.0 (unreleased)
------------------

First packaged version, built from ``folklore_test_module.py``.

* Functions split into the submodules ``io``, ``galaxy``, ``detection``,
  ``matching``, ``properties``, ``polygons``, ``photometry`` and
  ``plotting``; all are also available from the top-level ``folklore``
  namespace.
* New helper ``pixel_scale_arcsec`` and new plot
  ``plot_contrast_levels_filters`` (significance contours in every filter).
* ``build_galaxy_region`` returns the selected ``mask`` and can choose it
  automatically (``min_fraction``, ``growth_factor``); new plot
  ``plot_galaxy_region_levels`` to choose the level by eye.
* ``save_individual_clump_cubes_`` (full-frame masked cubes) renamed to
  ``save_masked_clump_cubes``.
* ``build_clump_dataframes`` gained a ``verbose`` option.
* Consistent detection defaults in all functions: ``sigma_center=3.5``,
  ``sigma_outer=4.0``, ``sigma_split=6.0``, ``min_area=3``.
* Fixes: ``skycoord_to_pixel`` accepts RA in decimal degrees;
  ``build_clump_geodataframe_sky`` handles clumps with several
  disconnected parts; ``plot_candidate_pixels`` writes one file per
  filter instead of overwriting the same file; empty detections return
  empty tables with the right columns; compatibility with
  Matplotlib ≥ 3.9 (``get_cmap`` removal).
