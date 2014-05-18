'''
======================================================================
:mod:`gridcells.analysis.registration` - Positional data registration.
======================================================================

Use the classes here to align (register) positional data of several recordings
with the specified arena coordinates.

Classes
-------
.. autosummary::

    RegistrationEngine

'''
from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.optimize import minimize

from ..core import Pair2D, Position2D


class RegistrationEngine(object):
    '''Register positional data to zero-coordinates of an arena.

    The actual positional data recordings are prone to outliers. This
    registration engine ensures that the positional data from different
    recordings are "aligned" with respect to the arena coordinates. This is
    accomplished by optimising the positional offsets with respect to the
    number of outliers.

    .. todo::

        Deal with rotations.
    '''

    class Result(object):
        '''A holder for registered data.

        Contains two attributes: ``positions`` and estimated ``offsets`` in the
        arena.
        '''
        def __init__(self, positions, offsets):
            self.positions = positions
            self.offsets = offsets


    def __init__(self, arena=None):
        '''Initialise with an ``arena`` agains which to register the data.

        Also use :meth:`set_arena` to change the specific
        arena.
        '''
        self._arena = arena

    def set_arena(self, arena):
        '''Set the arena for registration.
        
        All subsequent calls to :meth:`register` will be performed on this
        arena.
        '''
        self._arena = arena

    def register(self, positions):
        '''Register the positional data against the current arena.

        Parameters
        ----------
        positions : :class:`~gridcells.core.common.Position2D`
            Positional data.

        Returns
        -------
        res : :class:`Result`
            The result object, containing new positional data and the
            determined offsets.
        '''
        arena_sz = self._arena.getSize()
        def count_outliers(offsets):
            offx = offsets[0]
            offy = offsets[1]
            out_idx = np.logical_or(positions.x < offx,
                        np.logical_or(positions.x > arena_sz.x + offx,
                            np.logical_or(positions.y < offy,
                                          positions.y > arena_sz.y + offy)))
            return np.count_nonzero(out_idx)

        if self._arena is None:
            raise InitialisationError("Arena must be set for the registration "
                                      "process.")

        offsets0 = [np.nanmean(positions.x) - arena_sz.x/2.,
                    np.nanmean(positions.y) - arena_sz.y/2.]
        res = minimize(count_outliers, offsets0, method='Nelder-Mead')

        offsets = Pair2D(res.x[0], res.x[1])
        registered_pos = Position2D(positions.x - offsets.x,
                                    positions.y - offsets.y,
                                    positions.dt)
        return self.Result(registered_pos, offsets)

        

        
