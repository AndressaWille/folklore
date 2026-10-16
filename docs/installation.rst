Installation
============

With pip
--------

.. code-block:: bash

    pip install folklore-clumps

The package is imported as ``folklore`` (``import folklore as fm``).

From source
-----------

.. code-block:: bash

    git clone https://github.com/AndressaWille/folklore.git
    cd folklore
    pip install .

For development (editable install, plus test and documentation tools):

.. code-block:: bash

    pip install -e ".[test,docs]"

Dependencies
------------

FOLKLORE requires Python ≥ 3.9 and

* numpy, scipy, pandas
* astropy
* matplotlib
* scikit-image
* networkx
* shapely ≥ 2.0 and geopandas (clump polygons / GeoJSON output)

They are installed automatically by ``pip``.

Input data
----------

FOLKLORE works on photometric data cubes: FITS files with an image HDU of
shape ``(n_filters, ny, nx)``, a celestial WCS and one ``FILTERn`` keyword
per slice giving the filter name (``FILTER1`` is slice 0). Cubes built with
`lightstack <https://lightstack.readthedocs.io>`_ already follow this format.

For detection, the filters should be PSF-matched (all slices with the same
spatial resolution).

Running the tests
-----------------

.. code-block:: bash

    pytest

The tests build a small synthetic cube with injected clumps and run the
whole pipeline on it.
