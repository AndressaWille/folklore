# Examples

| File | What it does |
|------|--------------|
| `run_detection.py` | Detects clumps in a list of galaxies and writes the clump tables (`*_properties.csv`, `*_pixels.csv`), GeoJSON polygons and diagnostic figures. |
| `run_photometry.py` | Reads the `*_pixels.csv` tables and measures background-subtracted fluxes in any set of cubes (HST, JWST, ...), writing one SED-fitting table per galaxy. |

A step-by-step tutorial notebook is in
[`docs/tutorials/folklore_tutorial.ipynb`](../docs/tutorials/folklore_tutorial.ipynb).

Edit the `PARAMS` and `GALAXIES` dictionaries (and the data paths) at the
top of each script before running.

