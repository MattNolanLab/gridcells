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

class Pair2D(object):
    '''A pair of ``x`` and ``y`` attributes.'''
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "<Pair2D\n\tx: %s\n\ty: %s>" % (self.x, self.y)


class Position2D(Pair2D):
    '''Positional information with a constant time step.'''
    def __init__(self, x, y, dt):
        self.x = x
        self.y = y
        self.dt = dt

        if len(x) != len(y):
            raise ValueError("'x' and 'y' lengths must match: (%d, %d)" % (
                             len(x), len(y)))

    def __len__(self):
        return len(self.x)

    def __repr__(self):
        return "<Position2D\n\tx: %s\n\ty: %s\n\tdt: %s>" % (self.x, self.y,
                                                             self.dt)

