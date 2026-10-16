# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

import folklore  # noqa: E402

# -- Project information -----------------------------------------------------

project = "FOLKLORE"
copyright = "2026, Andressa Wille"
author = "Andressa Wille"
release = folklore.__version__
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints", "**/output"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# autodoc / napoleon
autodoc_member_order = "bysource"
autodoc_typehints = "none"
autosummary_generate = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "ndarray": "numpy.ndarray",
    "DataFrame": "pandas.DataFrame",
    "shapely.geometry.Polygon": "shapely.Polygon",
    "shapely.geometry.MultiPolygon": "shapely.MultiPolygon",
}
# generic words used as types in the docstrings
nitpick_ignore = [("py:class", "iterable"), ("py:class", "sequence")]

# notebooks are rendered with their saved outputs (data are not on RTD)
nb_execution_mode = "off"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/folklore_logo.png"
html_title = f"FOLKLORE {release}"
