'''
.. currentmodule: gridcells.tests.test_fields

The :mod:`~gridcells.test.test_fields` module defines a set of classes for unit
testing of the firing field analysis code.  The module is currently based on
the unittest module.

A list of currently supported tests:

.. autosummary::
'''
import unittest
import numpy as np
from gridcells import fields, _cppfields
import fields_ref_impl as refimp

notImplMsg = "Not implemented"

class TestRateFields(unittest.TestCase):
    '''Test the correctness of firing field analysis.'''
    rtol = 1e-9

    def setUp(self):
        self.refData = None
        self.refRateMap = None

    class InputData(object):
        def __init__(self, spikeTimes, pos_x, pos_y, pos_dt, arenaDiam, h):
            self.spikeTimes = spikeTimes
            self.pos_x = pos_x
            self.pos_y = pos_y
            self.pos_dt = pos_dt
            self.arenaDiam = arenaDiam
            self.h = h


    def loadRealDataSample(self):
        if self.refData is None:
            dataDir = "profiling/data"
            d = self.InputData(
                    spikeTimes = np.loadtxt("%s/spikeTimes.txt" % dataDir),
                    pos_x      = np.loadtxt("%s/pos_x.txt" % dataDir),
                    pos_y      = np.loadtxt("%s/pos_y.txt" % dataDir),
                    pos_dt     = 20,
                    arenaDiam  = 180.0,
                    h          = 3)
            refRateMap, refXedges, refYedges = \
                refimp.SNSpatialRate2D(d.spikeTimes, d.pos_x, d.pos_y,
                        d.pos_dt, d.arenaDiam, d.h)
            self.refData = d
            self.refRateMap = refRateMap
            self.refXedges = refXedges
            self.refYedges = refYedges
        return self.refData, self.refRateMap, self.refXedges, self.refYedges


    def test_PythonVersion(self):
        d, refRateMap, refXE, refYE = self.loadRealDataSample()
        theirRateMap, theirXE, theirYE = fields.SNSpatialRate2D(d.spikeTimes,
                d.pos_x, d.pos_y, d.pos_dt, d.arenaDiam, d.h)
        np.testing.assert_allclose(theirRateMap, refRateMap, self.rtol)
        np.testing.assert_allclose(theirXE, refXE, self.rtol)
        np.testing.assert_allclose(theirYE, refYE, self.rtol)



    @unittest.skip(notImplMsg)
    def test_CppVersion(self):
        pass


    @unittest.skip(notImplMsg)
    def test_CppOptimisedVersion(self):
        pass
