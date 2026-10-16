"""
FOLKLORE: Finding Overdensities and Luminous Knots via LOcal contRast Estimation.

Detection, cross-filter matching and photometry of star-forming clumps in
multi-filter photometric data cubes.

Every public function is available directly from the top-level package::

    import folklore as fm

    data_cube, header, wcs, filters = fm.load_datacube("cube.fits")

or from its submodule (``folklore.detection.build_clump_dataframes``).
"""

__version__ = "0.1.0"

from . import (
    detection,
    galaxy,
    io,
    matching,
    photometry,
    plotting,
    polygons,
    properties,
)
from .detection import *  # noqa: F401,F403
from .galaxy import *  # noqa: F401,F403
from .io import *  # noqa: F401,F403
from .matching import *  # noqa: F401,F403
from .photometry import *  # noqa: F401,F403
from .plotting import *  # noqa: F401,F403
from .polygons import *  # noqa: F401,F403
from .properties import *  # noqa: F401,F403

__all__ = (
    io.__all__
    + galaxy.__all__
    + detection.__all__
    + matching.__all__
    + properties.__all__
    + polygons.__all__
    + photometry.__all__
    + plotting.__all__
)
