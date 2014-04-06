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

:mod:`gridcells` is a simple C++/Python library that aims to provide an open
source code repository related to analysis and simulations of grid cell models.



Download
--------

:mod:`gridcells` can be downloaded from bitbucket at
https://bitbucket.org/lsolanka/gridanalysis.


Dependencies
------------

There are a number of dependencies needed for the python version:
    - CMake (>= 2.8)

    - `armadillo <http://arma.sourceforge.net/>`_ (>= 4.100)

    - SWIG (>= 2.0)

    - numpy (>= 1.6)

    - scipy (>= 0.13)

For Linux, simply install these using the package manager.For Mac OS the
easiest way is probably to use `homebrew <http://brew.sh/>`_. This package has
not been tested on Windows but if you manage to install the dependencies there
should be no problems.


Installation
------------

After installing all the dependencies, perform the following steps. If you only
want to use the package from the source directory, follow steps 1. -- 3. and
set your ``PYTHONPATH`` appropriately. Otherwise, follow all the steps.

    1. In the root directory, create the ``build`` directory and enter it.

    2. Run cmake. Currently the armadillo path has to be specified explicitly,
       i.e. ``cmake .. -DARMADILLO_INCLUDE_DIR=<path_to_armadillo>``. Replace
       ``<path_to_armadillo>`` with the appropriate directory containing the
       header files (for instance ``/usr/local/include``).

       In case you do not want to install system-wide, set the prefix parameter
       when running cmake: ``-DCMAKE_INSTALL_PREFIX=<install_path>``.

    3. Run ``make``. This will compile all the C++ files and copy the SWIG
       generated python modules into the original source.

    4. Run ``sudo make install``.

    5. Optionally run tests in the ``gridcells/tests`` directory by running
       ``python -m unittest discover``.


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
