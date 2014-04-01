=====================================
gridsCore - common/shared definitions
=====================================

gridsCore is mostly a wrapper around the basic classes and typedefs that are
shared among other modules. Currently there are these classes:

    1. ``VecPair`` - a wrapper containing two numpy vectors, ``x`` and ``y`` to
       give them a semantic meaning.

    2. ``Size2D`` - A 2D size variable.

    3. ``Position2D`` - A subclass of ``VecPair`` that also contains
       positional data indentically spaced in time.


.. todo::

    Improve the documentation transferred from doxygen.


C++/SWIG API reference
----------------------

These classes provide SWIG python wrappers and can therefore be used directly
from within the module. The ``arma::vec`` (column vector) and ``arma::mat`` (2D
array) classes are mapped directly to numpy.ndarray.

.. doxygentypedef:: grids::VecPair

.. doxygentypedef:: grids::Size2D

.. doxygenclass:: grids::Position2D
    :members:
