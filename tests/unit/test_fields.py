from __future__ import absolute_import, division, print_function

import pytest
import numpy as np
from gridcells.core import Position2D
from gridcells.core import CircularArena, Pair2D
from gridcells.analysis import spatialRateMap, extractSpikePositions
from gridcells.analysis import occupancy_prob_dist
import fields_ref_impl as refimp


@pytest.fixture(scope='module')
def reference_data():
    class InputData(object):
        def __init__(self, spike_times, pos_x, pos_y, pos_dt, arena_diam, h):
            self.spike_times = spike_times
            self.pos_x = pos_x
            self.pos_y = pos_y
            self.pos_dt = pos_dt
            self.arena_diam = arena_diam
            self.h = h
    data_dir = "tests/data"
    arena_diam = 180.0
    ref_data = InputData(
        spike_times=np.loadtxt("%s/spikeTimes.txt" % data_dir),
        pos_x=np.loadtxt("%s/pos_x.txt" % data_dir) + arena_diam/2.,
        pos_y=np.loadtxt("%s/pos_y.txt" % data_dir) + arena_diam/2.,
        pos_dt=20,
        arena_diam=arena_diam,
        h=3)
    return ref_data


@pytest.fixture(scope='module')
def reference_spatial_map(reference_data):
    d = reference_data
    ref_rate_map, ref_xedges, ref_yedges = refimp.SNSpatialRate2D(
        d.spike_times,
        d.pos_x, d.pos_y, d.pos_dt,
        d.arena_diam,
        d.h)
    return d, ref_rate_map, ref_xedges, ref_yedges


class TestSpatialRateMap(object):
    '''Test the correctness of firing field analysis.'''
    rtol = 1e-3
    arenaR = 90.    # cm
    sigma = 3.      # cm

    def test_naive_version(self, reference_spatial_map):
        d, ref_rate_map, ref_xedges, ref_yedges = reference_spatial_map

        # Tested code
        ar = CircularArena(self.arenaR, Pair2D(self.sigma, self.sigma))
        pos = Position2D(d.pos_x, d.pos_y, d.pos_dt)
        their_rate_map = spatialRateMap(d.spike_times, pos, ar, self.sigma)
        their_edges = ar.getDiscretisation()
        # print(np.max(np.abs(their_rate_map - ref_rate_map)))
        np.testing.assert_allclose(their_rate_map, ref_rate_map, self.rtol)
        np.testing.assert_allclose(their_edges.x, ref_xedges, self.rtol)
        np.testing.assert_allclose(their_edges.y, ref_yedges, self.rtol)


@pytest.fixture(scope='module')
def npos():
    return 10000


@pytest.fixture(scope='module')
def nspikes():
    return 10


@pytest.fixture(scope='module')
def dt_fixture():
    return 20.


@pytest.fixture(scope='module')
def incorrect_spikes_p():
    return .1


@pytest.fixture
def positions(npos, dt_fixture):
    x = np.arange(npos, dtype=np.float)
    y = np.arange(npos, dtype=np.float) + 10.
    return Position2D(x, y, dt_fixture)


class TestSpikePosExtraction(object):
    '''Test conversion of spike times into spike positions in the arena'''

    def get_correct_spike_times(self, nspikes, positions):
        tstart = 0.
        tend = len(positions)-1 * positions.dt
        return np.sort(np.random.rand(nspikes) * (tend-tstart))

    def reference_spike_pos(self, spike_times, positions):
        ref_spike_idx = np.array(spike_times // positions.dt, dtype=np.int)
        ref_spike_pos = Pair2D(positions.x[ref_spike_idx],
                               positions.y[ref_spike_idx])
        return ref_spike_pos

    def inject_at_random_pos(self, p, nspikes, positions, times_val,
                             spike_pos_val):
        corr_times = self.get_correct_spike_times(nspikes, positions)
        corr_spike_pos = self.reference_spike_pos(corr_times, positions)
        times_idx = np.random.choice(len(corr_times), size=len(corr_times)*p)
        inj_times = np.copy(corr_times)
        inj_times[times_idx] = times_val
        inj_spike_pos = corr_spike_pos.copy()
        inj_spike_pos.x[times_idx] = spike_pos_val
        inj_spike_pos.y[times_idx] = spike_pos_val
        return inj_times, inj_spike_pos

    def test_correct_extraction(self, nspikes, positions):
        spike_times = self.get_correct_spike_times(nspikes, positions)
        spike_pos, m_i = extractSpikePositions(spike_times, positions)
        # TODO: get rid of code duplication
        ref_spike_pos = self.reference_spike_pos(spike_times, positions)
        assert spike_pos == ref_spike_pos

    def test_negative_idx(self, nspikes, incorrect_spikes_p, positions):
        if nspikes == 0:
            return
        # Prepare negative times
        neg_times, neg_spike_pos = self.inject_at_random_pos(
            incorrect_spikes_p,
            nspikes,
            positions,
            -10,
            np.nan)

        # Do actual test - must be element-wise to test for NaNs
        their_spike_pos, m_i = extractSpikePositions(neg_times, positions)
        np.testing.assert_equal(their_spike_pos.x, neg_spike_pos.x)
        np.testing.assert_equal(their_spike_pos.y, neg_spike_pos.y)

    def test_out_of_bounds_idx(self, nspikes, incorrect_spikes_p, positions):
        if nspikes == 0:
            return
        # Prepare out of bound times
        out_times, out_spike_pos = self.inject_at_random_pos(
            incorrect_spikes_p,
            nspikes,
            positions,
            len(positions)*positions.dt,
            np.nan)

        # Do actual test - must be element-wise to test for NaNs
        their_spike_pos, m_i = extractSpikePositions(out_times, positions)
        np.testing.assert_equal(their_spike_pos.x, out_spike_pos.x)
        np.testing.assert_equal(their_spike_pos.y, out_spike_pos.y)


@pytest.fixture(scope='function')
def fix_pos_data(fix_arena):
    ar = fix_arena

    pos_x = np.random.rand(100000) * ar.sz.x / 2.
    pos_y = np.random.rand(100000) * ar.sz.y / 2.
    return Position2D(pos_x, pos_y, 0.02)


@pytest.fixture(scope='module')
def fix_arena():
    arenaR = 90.    # cm
    sigma = 3.      # cm
    return CircularArena(arenaR, Pair2D(sigma, sigma))

class TestOccupancyPDF(object):
    '''Tests for Occupancy Probability density function computation'''

    def test_output_shape(self, fix_arena, fix_pos_data):
        arena = fix_arena
        pos = fix_pos_data

        edges = arena.getDiscretisation()

        pdf = occupancy_prob_dist(arena, pos)
        assert len(edges.x) == pdf.shape[0]
        assert len(edges.y) == pdf.shape[1]
        np.testing.assert_allclose(np.sum(np.ravel(pdf)), 1.)
