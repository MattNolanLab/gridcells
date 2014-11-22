'''
========================================================
:mod:`gridcells.core.common` - common/shared definitions
========================================================

The :mod:`~gridcells.core.common` module is a collection of basic classes used
throughout the package:

.. autosummary::

    Pair2D
    Position2D
    twisted_torus_distance

'''
from __future__ import absolute_import, print_function

import os

import numpy as np

# Do not import when in RDT environment
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not on_rtd:
    from . import _common
    from ._common import divisor_mod
else:
    divisor_mod = None

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


def twisted_torus_distance(a, others, dim):
    ''' Calculate a distance between ``a`` and ``others`` on a twisted torus.

    Take ``a`` which is a 2D position and others, which is a vector of 2D
    positions and compute the distances between them based on the topology of
    the twisted torus.

    If you just want to remap a function of (X, Y), set a==[[0, 0]].

    Parameters
    ----------

    a : :class:`Pair2D`
        Specifies the initial position. ``a.x`` and ``a.y`` must be convertible
        to floats
    others : :class:`Pair2D`
        Positions for which to compute the distance.
    dim : :class:`Pair2D`
        Dimensions of the torus. ``dim.x`` and ``dim.y`` must be convertible to
        floats.

    Returns
    -------
    An array of positions, always of the length of others
    '''
    return _common.twisted_torus_distance(
            a.x, a.y,
            others.x, others.y,
            dim.x, dim.y)

