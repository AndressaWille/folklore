Workflow
========

.. figure:: _static/workflow_folklore.png
   :width: 100%
   :alt: FOLKLORE detection and combination steps

FOLKLORE works in two stages. Clumps are **detected** on a PSF-matched
data cube and stored in sky coordinates; their **photometry** can then be
measured on any cube with a WCS (other instruments, other pixel grids).

1. **Galaxy masks.** The cube is collapsed and thresholded at two
   percentiles: the search region and the bright centre
   (:func:`~folklore.galaxy.build_galaxy_masks`).
2. **Local contrast.** In every filter, each pixel is expressed as its
   excess over a running-median background in units of the local noise,
   :math:`C = (I - B)/\sigma_\mathrm{local}`
   (:func:`~folklore.detection.build_contrast_map`).
3. **Detection.** Pixels above a significance threshold (different in the
   centre and in the outskirts) are grouped into clumps in each filter
   (:func:`~folklore.detection.build_clump_dataframes`).
4. **Cross-filter matching.** Detections that overlap in different filters
   are linked; clumps found in at least ``min_filters`` filters are kept,
   and their final mask is the union over all filters
   (:mod:`folklore.matching`).
5. **Properties and classification.** Sizes in arcsec and kpc, and
   disk/outside classification with a galaxy body mask grown from the
   galaxy centre (:mod:`folklore.properties`,
   :func:`~folklore.galaxy.build_galaxy_region`). This step needs the
   coordinates of the galaxy centre and its redshift.
6. **Outputs.** Clump catalogue, pixel table and polygons in pixel and sky
   coordinates (GeoJSON) (:mod:`folklore.polygons`).
7. **Photometry.** The clumps are transferred to another cube through their
   sky coordinates, and background-subtracted fluxes are measured and
   arranged for SED fitting (:mod:`folklore.photometry`).

Every step has diagnostic plots in :mod:`folklore.plotting`. See the
:doc:`tutorial <tutorials/folklore_tutorial>` for a complete example and
the :doc:`API reference <api/index>` for the details of each function.
