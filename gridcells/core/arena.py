'''
==============================================
:mod:`gridcells.core.arena` - Defining arenas
==============================================

The :mod:`~gridcells.core.arena` module provides class definitions of arenas. These
can subsequently be used as input to process spiking data and generate spatial
firing fields/autocorrelations.

These types of arenas are currently defined:
--------------------------------------------
.. autosummary::

    Arena
    CircularArena
    RectangularArena
    SquareArena
'''
from __future__ import absolute_import, print_function, division

import numpy as np

from .common import Pair2D

##############################################################################


class Arena(object):
    '''An abstract class for arenas.

    This class is an interface for obtaining discretisations of the arenas and
    masks when the shape is not rectangular.
    '''

    def getDiscretisation(self):
        '''Obtain the discretisation of this arena.

        Returns
        =======
        d : gridcells.core.Pair2D
            A pair of x and y coordinates for the positions in the arena. Units
            are arbitrary.
        '''
        raise NotImplementedError()

    def getMask(self):
        '''Return mask (a 2D ``np.ndarray``) of where the positions in the
        arena are valid.

        For isntance with a circular arena, all positions outside its radius
        are invalid.
        '''
        raise NotImplementedError()


class RectangularArena(Arena):
    '''A rectangular arena.

    Use :class:`~gridcells.core.RectangularArena` when you need to work with
    rectangular arenas.

    .. note::
        The origin (0, 0) of the coordinate system in all the arenas is in the
        bottom-left corner of the arena.
    '''
    def __init__(self, size, discretisation):
        self._sz = size
        self._q = discretisation

    def getDiscretisation(self):
        numX = self._sz.x / self._q.x + 1
        numY = self._sz.y / self._q.y + 1
        xedges = np.linspace(0., self._sz.x, numX)
        yedges = np.linspace(0., self._sz.y, numY)
        return Pair2D(xedges, yedges)

    def getDiscretisationSteps(self):
        return self._q

    def getMask(self):
        return None

    def getSize(self):
        return self._sz

    @property
    def sz(self):
        '''Return the size of the arena. Equivalent to
        :meth:`~RectangularArena.getSize`.
        '''
        return self._sz

    @property
    def bounds(self):
        return Pair2D(
                    (0., self._sz.x),
                    (0., self._sz.y)
               )


class SquareArena(RectangularArena):
    '''A square arena.'''
    def __init__(self, size, discretisation):
        tmpSz = Pair2D(size, size)
        super(SquareArena, self).__init__(tmpSz, discretisation)


class CircularArena(SquareArena):
    '''A circular arena.'''
    def __init__(self, radius, discretisation):
        super(CircularArena, self).__init__(radius*2., discretisation)
        self.radius = radius

    def getMask(self):
        edges = self.getDiscretisation()
        X, Y = np.meshgrid(edges.x, edges.y)
        return np.sqrt(X**2 + Y**2) > self.radius

