Overview
========

What are grid cells
-------------------

Grid cells are a type of cells found in some mammalian species that have very
regular spatial firing fields. When these animals forage in experimental
arenas, grid cells are active only in certain places and the locations where
they are active form a hexagonal lattice. More information can be found in
[MOSER2007]_ or in [HAFTING2005]_.


What does :mod:`gridcells` do
-----------------------------

:mod:`gridcells` is a simple Python library that aims to provide an open source
code repository related to analysis of grid cell related experiments and
simulation of models of grid cells.



Download
--------

:mod:`gridcells` can be downloaded from https://github.com/lsolanka/gridcells.


Dependencies
------------

There are a number of dependencies needed for the python version:
    - setuptools (>= 3.6)

    - SWIG  (ideally >= 3.0; earlier versions not tested)

    - numpy (>= 1.8)

    - scipy (>= 0.13.3)

    - matplotlib (>= 1.3.1)

For Linux, simply install these using the package manager.For Mac OS the
easiest way is probably to use `homebrew <http://brew.sh/>`_. This package has
not been tested on Windows but if you manage to install the dependencies there
should be no problems.


Installation
------------

After installing the dependencies, run::

    python setup.py install

.. note::

    Currently, installing with ``pip`` through the Python Package Index has
    some difficulties with the dependency on numpy. Thus it is advisable to
    have ``numpy`` installed beforehand.


License
-------

:mod:`gridcells` is distributed under the GPL license. See LICENSE.txt in the
root of the source directory.


References
----------

.. [MOSER2007] Edvard Moser and May-Britt Moser (2007). Grid cells.
               Scholarpedia, 2(7):3394.

.. [HAFTING2005] Hafting, T. et al., 2005. Microstructure of a spatial map in
                 the entorhinal cortex. Nature, 436(7052), pp.801–806.
