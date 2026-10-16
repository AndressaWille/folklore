<img src="https://raw.githubusercontent.com/AndressaWille/folklore/master/docs/_static/folklore_logo.png" alt="FOLKLORE logo" width="170" align="left">

# FOLKLORE

**Finding Overdensities and Luminous Knots via LOcal contRast Estimation**

<br clear="left">

Detection, cross-filter matching and photometry of star-forming clumps in
multi-filter photometric data cubes (JWST, HST, ...).

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

## Documentation

https://folklore.readthedocs.io

A tutorial notebook is in [`docs/tutorials/`](docs/tutorials/folklore_tutorial.ipynb).

## Input data

FITS data cubes with shape `(n_filters, ny, nx)`, a celestial WCS and one
`FILTERn` header keyword per slice — the format produced by
[lightstack](https://github.com/AndressaWille/lightstack).

## How to cite

If you use FOLKLORE in your research, please cite it through its Zenodo
record: [doi:10.5281/zenodo.23407081](https://doi.org/10.5281/zenodo.23407081).

This DOI always points to the latest version; the Zenodo page also lists
a DOI for each individual version.

## Contact

Open an issue or write to andressaw2@gmail.com.
