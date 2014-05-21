'''
========================================================
:mod:`gridcells.core.common` - common/shared definitions
========================================================

The :mod:`~gridcells.core.common` module is a collection of basic classes used
throughout the package:

.. autosummary::

    Pair2D
    Position2D

'''
import numpy as np

class Pair2D(object):
    '''A pair of ``x`` and ``y`` attributes.'''
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def copy(self):
        return Pair2D(np.copy(self.x), np.copy(self.y))

    def __repr__(self):
        return "<Pair2D\n\tx: %s\n\ty: %s>" % (self.x, self.y)

    def __eq__(self, other):
        return np.all(self.x == other.x) and np.all(self.y == other.y)

    def __ne__(self, other):
        return not self.__eq__(other)


class Position2D(Pair2D):
    '''Positional information with a constant time step.'''
    def __init__(self, x, y, dt):
        self.x = x
        self.y = y
        self.dt = dt

        if len(x) != len(y):
            raise ValueError("'x' and 'y' lengths must match: (%d, %d)" % (
                             len(x), len(y)))

    def copy(self):
        return Position2D(np.copy(self.x), np.copy(self.y), self.dt)

    def __len__(self):
        return len(self.x)

    def __repr__(self):
        return "<Position2D\n\tx: %s\n\ty: %s\n\tdt: %s>" % (self.x, self.y,
                                                             self.dt)

    def __eq__(self, other):
        return super(Position2D, self).__eq__(other) and self.dt == other.dt

    def __ne__(self, other):
        return not self.__eq__(other)

