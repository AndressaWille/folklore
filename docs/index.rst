FOLKLORE
========

**Finding Overdensities and Luminous Knots via LOcal contRast Estimation**

FOLKLORE detects star-forming clumps in multi-filter photometric data
cubes, matches them across filters, measures their sizes and positions and
extracts background-subtracted photometry ready for SED fitting.

It was developed for clumps in jellyfish galaxies observed with JWST/NIRCam
and HST, but works on any FITS data cube with a WCS and the filter names in
the header.

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
record: `doi:10.5281/zenodo.23407081
<https://doi.org/10.5281/zenodo.23407081>`_. This DOI always points to the
latest version; the Zenodo page also lists a DOI for each individual
version.

Contact
-------

If you find a problem, have suggestions or would like a new feature,
please open an issue on GitHub or write to andressaw2@gmail.com.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
