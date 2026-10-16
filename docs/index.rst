FOLKLORE
========

**Finding Overdensities and Luminous Knots via LOcal contRast Estimation**

FOLKLORE detects star-forming clumps in multi-filter photometric data
cubes, matches them across filters, measures their sizes and positions and
extracts background-subtracted photometry ready for SED fitting.

It was developed for clumps in jellyfish galaxies observed with JWST/NIRCam
and HST, but works on any FITS data cube with a WCS and the filter names in
the header.

Features
--------

* Galaxy masks from percentile thresholds of the collapsed cube
* Local-contrast detection with different thresholds for the centre and
  the outskirts, plus optional hierarchical splitting of blended clumps
* Cross-filter matching through an overlap graph
* Clump sizes (arcsec and kpc) and disk/outside classification
* Clump masks stored in sky coordinates and exported as GeoJSON
* Photometry on any cube (other instruments, other pixel grids) with local
  background rings
* Diagnostic plots for every step

Quick example
-------------

.. code-block:: python

   import folklore as fm

   data_cube, header, wcs, filters = fm.load_datacube("cube.fits")

   galaxy_masks = fm.build_galaxy_masks(fm.collapse_cube(data_cube),
                                        percentiles=(30, 90))

   clumps_by_filter, pixels_by_filter, pixels_all = fm.build_clump_dataframes(
       data_cube, filters, galaxy_masks,
       sigma_center=3.5, sigma_outer=4.0, min_area=3)

   overlaps = fm.compute_clump_overlaps(fm.build_clump_pixel_sets(pixels_all))
   components = fm.find_final_clumps(fm.build_overlap_graph(overlaps),
                                     min_filters=3)
   _, df_pixels_final = fm.build_final_clump_dataframes(
       clumps_by_filter, pixels_by_filter, components)

   union_masks = fm.build_union_masks(df_pixels_final, data_cube.shape[1:])
   fm.plot_union_clumps(data_cube, filters, union_masks)

.. figure:: _static/workflow_folklore.png
   :width: 100%
   :alt: FOLKLORE workflow

See :doc:`workflow` for the full pipeline.

Source code: https://github.com/AndressaWille/folklore

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   workflow
   tutorials/index
   api/index
   changelog

How to cite
-----------

If you use FOLKLORE in your research, please cite it through its Zenodo
record: `doi:10.5281/zenodo.23407081 <https://doi.org/10.5281/zenodo.23407081>`_. This DOI always points to
the latest version; the Zenodo page also lists a DOI for each individual
version.

Contact
-------

If you find a problem, have suggestions or would like a new feature,
please open an issue on GitHub or write to andressaw2@gmail.com.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
