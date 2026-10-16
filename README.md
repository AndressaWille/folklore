<p align="center">
  <img src="https://raw.githubusercontent.com/AndressaWille/folklore/master/docs/_static/folklore_logo.png" alt="FOLKLORE logo" width="220">
</p>

# FOLKLORE

**Finding Overdensities and Luminous Knots via LOcal contRast Estimation**
*Drawing stars around galactic scars.*

[![Documentation Status](https://readthedocs.org/projects/folklore/badge/?version=latest)](https://folklore.readthedocs.io/en/latest/)
[![Tests](https://github.com/AndressaWille/folklore/actions/workflows/tests.yml/badge.svg)](https://github.com/AndressaWille/folklore/actions/workflows/tests.yml)

Detection, cross-filter matching and photometry of star-forming clumps in
multi-filter photometric data cubes (JWST, HST, ...).

## Features

- Galaxy masks from the collapsed cube
- Local-contrast clump detection in every filter
- Cross-filter matching through an overlap graph
- Clump sizes (arcsec, kpc) and disk/outside classification
- Clump polygons in pixel and sky coordinates (GeoJSON)
- Background-subtracted photometry on any cube, ready for SED fitting
- Diagnostic plots for every step

## Installation

```bash
pip install folklore-clumps
```

or, from source:

```bash
git clone https://github.com/AndressaWille/folklore.git
cd folklore
pip install .
```

The package is imported as `folklore`.

## Quick start

```python
import folklore as fm

data_cube, header, wcs, filters = fm.load_datacube("cube.fits")

galaxy_masks = fm.build_galaxy_masks(fm.collapse_cube(data_cube), percentiles=(30, 90))

clumps_by_filter, pixels_by_filter, pixels_all = fm.build_clump_dataframes(
    data_cube, filters, galaxy_masks, sigma_center=3.5, sigma_outer=4.0, min_area=3)

overlaps = fm.compute_clump_overlaps(fm.build_clump_pixel_sets(pixels_all))
components = fm.find_final_clumps(fm.build_overlap_graph(overlaps), min_filters=3)
_, df_pixels_final = fm.build_final_clump_dataframes(clumps_by_filter, pixels_by_filter, components)

union_masks = fm.build_union_masks(df_pixels_final, data_cube.shape[1:])
fm.plot_union_clumps(data_cube, filters, union_masks)
```

A tutorial notebook is in [`docs/tutorials/`](docs/tutorials/folklore_tutorial.ipynb) and complete scripts are in [`examples/`](examples/).

## Documentation

https://folklore.readthedocs.io

## Input data

FITS data cubes with shape `(n_filters, ny, nx)`, a celestial WCS and one
`FILTERn` header keyword per slice — the format produced by
[lightstack](https://github.com/AndressaWille/lightstack).

## Contact

Open an issue or write to andressaw2@gmail.com.
