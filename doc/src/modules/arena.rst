.. :module:: gridcells.arena

=======================
arena - Defining arenas
=======================

The :mod:`~gridcells.arena` module provides class definitions of arenas. These
can subsequently be used as input to process spiking data and generate spatial
firing fields/autocorrelations.

These types of arenas are currently defined:

  1. An abstract :cpp:class:`~grids::Arena` class

  2. :cpp:class:`~grids::RectangularArena` to define rectangular arenas

  3. :cpp:class:`~grids::SquareArena` to define square arenas, which are a
     special case of rectangular arenas

  4. :cpp:class:`~grids::CircularArena` to define circular arenas,
     parameterized by their radius. Any data outside the radius are invalid.


C++/SWIG API reference
----------------------

These classes provide SWIG python wrappers and can therefore be used directly
from within the module. The ``arma::vec`` (column vector) and ``arma::mat`` (2D
array) classes are mapped directly to numpy.ndarray.

.. doxygenclass:: grids::Arena
    :members:

.. doxygenclass:: grids::RectangularArena
    :members:

.. doxygenclass:: grids::SquareArena
    :members:


