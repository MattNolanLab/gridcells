import pytest
import numpy as np
from gridcells.core import Position2D
from gridcells.fields import spatialRateMap
from gridcells.core import CircularArena, Pair2D
import fields_ref_impl as refimp


@pytest.fixture(scope='module')
def reference_data():
    class InputData(object):
        def __init__(self, spikeTimes, pos_x, pos_y, pos_dt, arenaDiam, h):
            self.spikeTimes = spikeTimes
            self.pos_x = pos_x
            self.pos_y = pos_y
            self.pos_dt = pos_dt
            self.arenaDiam = arenaDiam
            self.h = h
    dataDir = "tests/data"
    ref_data = InputData(
        spikeTimes = np.loadtxt("%s/spikeTimes.txt" % dataDir),
        pos_x      = np.loadtxt("%s/pos_x.txt" % dataDir),
        pos_y      = np.loadtxt("%s/pos_y.txt" % dataDir),
        pos_dt     = 20,
        arenaDiam  = 180.0,
        h          = 3)
    return ref_data


@pytest.fixture(scope='module')
def reference_spatial_map(reference_data):
    d = reference_data
    ref_rate_map, ref_xedges, ref_yedges = refimp.SNSpatialRate2D(
            d.spikeTimes,
            d.pos_x, d.pos_y, d.pos_dt,
            d.arenaDiam,
            d.h)
    return d, ref_rate_map, ref_xedges, ref_yedges


class TestSpatialRateMap(object):
    '''Test the correctness of firing field analysis.'''
    rtol = 1e-3
    arenaR = 90. # cm
    sigma  = 3.  # cm

    def test_NaiveVersion(self, reference_spatial_map):
        d, ref_rate_map, ref_xedges, ref_yedges = reference_spatial_map

        # Tested code
        ar = CircularArena(self.arenaR, Pair2D(self.sigma, self.sigma))
        pos = Position2D(d.pos_x, d.pos_y, d.pos_dt)
        theirRateMap = spatialRateMap(d.spikeTimes, pos, ar, self.sigma)
        theirEdges = ar.getDiscretisation()
        #print(np.max(np.abs(theirRateMap - ref_rate_map)))
        np.testing.assert_allclose(theirRateMap, ref_rate_map, self.rtol)
        np.testing.assert_allclose(theirEdges.x, ref_xedges, self.rtol)
        np.testing.assert_allclose(theirEdges.y, ref_yedges, self.rtol)

