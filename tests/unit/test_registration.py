from __future__ import absolute_import, division, print_function

import pytest
import numpy as np

from gridcells.core import Pair2D, Position2D, SquareArena
from gridcells.analysis import ArenaOriginRegistration


@pytest.fixture(scope='module')
def npos():
    return 10000

@pytest.fixture(scope='module')
def ntests():
    return 100

@pytest.fixture(scope='module')
def position_generator(arena, npos, ntests):
    return UrandPositionGenerator(arena, npos, ntests)

@pytest.fixture(scope='module', params=range(10, 200, 10))
def arena(request):
    return SquareArena(request.param, None)

@pytest.fixture(scope='module')
def registration_engine():
    return ArenaOriginRegistration()

@pytest.fixture(scope='module')
def rtol():
    return 1e-1



class UrandPositionGenerator(object):
    '''Works only with rectangular arenas for now.'''

    class Data(object):
        def __init__(self, positions, offsets):
            self.positions = positions
            self.offsets = offsets

        def __repr__(self):
            return "<Data object:\n\tPositions: %s\n\tOffsets: %s>" % (self.positions,
                                                   self.offsets)

    max_offset = .4         # relative to arena size
    max_outlier_range = .1  # relative to arena size
    outlier_fraction = .1   # n_outliers / npos
    dt = 20                 # just a stub, not used here

    def __init__(self, arena, npos, ntests):
        self.arena = arena
        self.npos = npos
        self.ntests = ntests

    def _gen_offsets(self):
        sz = self.arena.getSize()
        return Pair2D(np.random.rand() * sz.x * self.max_offset,
                      np.random.rand() * sz.y * self.max_offset)

    def _inject_outliers(self, pos, offsets):
        sz = len(pos)
        n_outliers = sz*self.outlier_fraction
        idx_list = np.random.choice(np.arange(sz),
                                    size=n_outliers,
                                    replace=False)
        outlier_range = Pair2D(self.arena.getSize().x * self.max_outlier_range,
                               self.arena.getSize().y * self.max_outlier_range)
        pos.x[idx_list[0:n_outliers/4]] = (
                offsets.x - np.random.rand(n_outliers/4)*outlier_range.x)
        pos.x[idx_list[n_outliers/4:n_outliers/2]] = (
                self.arena.getSize().x + offsets.x +
                np.random.rand(n_outliers/4)*outlier_range.x)
        pos.y[idx_list[n_outliers/2:n_outliers/4*3]] = ( 
                offsets.y - np.random.rand(n_outliers/4)*outlier_range.y)
        pos.y[idx_list[n_outliers/4*3:]] = (
                self.arena.getSize().y + offsets.y +
                np.random.rand(n_outliers/4)*outlier_range.y)

    def all_data(self):
        for it in range(self.ntests):
            offsets = self._gen_offsets()
            arena_sz = self.arena.getSize()
            pos = Position2D(arena_sz.x * np.random.rand(self.npos) + offsets.x,
                             arena_sz.y * np.random.rand(self.npos) + offsets.y,
                             self.dt)
            self._inject_outliers(pos, offsets)
            yield self.Data(pos, offsets)
            

class TestRegistration(object):
    '''Test the whole registration process.

    This works only with 2D arenas for now.
    '''

    def test_origin_registration(self, position_generator, arena,
                                 registration_engine, rtol):
        pg = position_generator
        rege = registration_engine
        atol = rtol * pg.max_outlier_range * max(arena.sz.x, arena.sz.y)

        rege.set_arena(arena)
        for ref_data in pg.all_data():
            reg_result = rege.register(ref_data.positions)
            offsets = ref_data.offsets
            np.testing.assert_allclose(reg_result.offsets.x, offsets.x,
                                       rtol=0, atol=atol)
            np.testing.assert_allclose(reg_result.offsets.y, offsets.y,
                                       rtol=0, atol=atol)
            np.testing.assert_allclose(
                    reg_result.positions.x + reg_result.offsets.x,
                    ref_data.positions.x,
                    rtol=rtol)
            np.testing.assert_allclose(
                    reg_result.positions.y + reg_result.offsets.y,
                    ref_data.positions.y,
                    rtol=rtol)

