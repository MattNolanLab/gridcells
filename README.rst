GridCells
=========

What are grid cells
-------------------

Grid cells are a type of cells found in some mammalian species that have very
regular spatial firing fields. When these animals forage in experimental
arenas, grid cells are active only in certain places and the locations where
they are active form a hexagonal lattice. More information can be found in
[MOSER2007]_ or in [HAFTING2005]_.


What does `gridcells` do
-----------------------------

`gridcells` is a simple Python library that aims to provide an open source code
repository related to analysis of grid cell related experiments and simulation
of models of grid cells.



Download
--------

`gridcells` can be downloaded from https://github.com/lsolanka/gridcells.


Dependencies
------------

There are a number of dependencies you need to install this package:

    - setuptools (>= 3.6)

    - SWIG  (ideally >= 3.0; earlier versions not tested)

    - numpy (>= 1.8)

    - scipy (>= 0.13.3)

    - matplotlib (>= 1.3.1)

For Linux, simply install these using the package manager.For Mac OS the
easiest way is probably to use `homebrew <http://brew.sh/>`_ and pip. This
package has not been tested on Windows but if you manage to install the
dependencies there should be no problems.


Installation
------------

After installing ``SWIG``, run::

    python setup.py install


License
-------

`gridcells` is distributed under the GPL license. See LICENSE.txt in the root
of the source directory. `armadillo`, which is part of this source, is
distributed under the `Mozilla Public License 2.0
<http://arma.sourceforge.net/license.html>`_. This packages also uses a
modified version of `armanpy <http://sourceforge.net/p/armanpy/wiki/Home/>`_.
`armanpy` is distributed under the LGPL license.


References
----------

.. [MOSER2007] Edvard Moser and May-Britt Moser (2007). Grid cells.
               Scholarpedia, 2(7):3394.

.. [HAFTING2005] Hafting, T. et al., 2005. Microstructure of a spatial map in
                 the entorhinal cortex. Nature, 436(7052), pp.801–806.

Build Status
------------
.. image:: https://travis-ci.org/lsolanka/gridcells.svg?branch=master
    :target: https://travis-ci.org/lsolanka/gridcells
