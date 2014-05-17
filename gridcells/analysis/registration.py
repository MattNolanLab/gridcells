'''Arena registration.

The actual positional data recordings are prone to outliers. The classes here
ensure that given an arena with a specified size, the majority of positional
data fit into the arena and the rest are classified as outside values.
'''
from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.optimize import minimize

from ..core import Pair2D, Position2D


class RegistrationEngine(object):
    class Result(object):
        '''A holder for registered data.

        Contains two attributes: ``positions`` and estimated ``offsets`` in the
        arena.
        '''
        def __init__(self, positions, offsets):
            self.positions = positions
            self.offsets = offsets


    def __init__(self, arena=None):
        self._arena = arena

    def set_arena(self, arena):
        self._arena = arena

    def register(self, positions):
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

        

        
