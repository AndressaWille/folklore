Tutorials
=========

The tutorial notebook goes through every step of FOLKLORE for one
jellyfish galaxy (detection, cross-filter matching, disk/outside
classification, polygons and photometry) and then runs the same steps in
a loop over several galaxies.

The data cubes used in the notebook are not distributed with the package:
the notebook shows how to call the functions and what each step produces.
Use it as a template with your own data cubes. It is rendered here with
its saved outputs.

.. toctree::
   :maxdepth: 1

   folklore_tutorial

The complete detection and photometry scripts are in the ``examples/``
folder of the repository:

* ``examples/run_detection.py``
* ``examples/run_photometry.py``
