'''Test the analysis.signal module.'''
from __future__ import absolute_import, print_function, division
import unittest
import numpy as np

import gridcells.analysis.signal as asignal

notImplMsg = "Not implemented"


class TestCorrelation(unittest.TestCase):
    '''
    Test the analysis.signal.corr function (and effectively the core of the
    autoCorrelation) function.
    '''
    def setUp(self):
        self.places = 10
        self.maxN = 1000
        self.maxLoops = 1000

    def assertSequenceAlmostEqual(self, first, second, places=None, msg=None,
                                  delta=None):
        """
        Fail if the two objects are unequal as determined by the difference
        between all of their values, rounded to the given number of decimal
        places (default 7) and comparing to zero, or by comparing that the
        differences between any two items of the two objects is more than the
        given delta.

        The test will fail if the conditions for any of the elements are not
        met.

        Note that decimal places (from zero) are usually not the same
        as significant digits (measured from the most signficant digit).

        If the two objects compare equal then they will automatically
        compare almost equal.
        """
        if delta is not None and places is not None:
            raise TypeError("specify delta or places not both")

        if delta is not None:
            if np.all(np.abs(first - second) <= delta):
                return

            standardMsg = '%s != %s within %s delta' % (first, second, delta)
        else:
            if places is None:
                places = 7

            if np.all(np.round(np.abs(second - first), places) == 0):
                return

            standardMsg = '%s != %s within %r places' % (first, second, places)
        msg = self._formatMessage(msg, standardMsg)
        raise self.failureException(msg)

    def checkCppNumpyCorr(self, a1, a2):
        '''Check whether the cpp version gives approximately equal results when
        compared to numpy.'''
        c_cpp = asignal.corr(a1, a2, mode='twosided')
        c_np    = np.correlate(a1, a2, mode='full')[::-1]
        self.assertSequenceAlmostEqual(c_cpp, c_np, places=self.places)

    def test_cpp(self):
        '''Test the c++ vs. numpy version.'''
        for _ in range(self.maxLoops):
            N1 = np.random.randint(self.maxN) + 1
            N2 = np.random.randint(self.maxN) + 1
            if N1 == 0 and N2 == 0:
                continue

            a1 = np.random.rand(N1)
            a2 = np.random.rand(N2)

            self.checkCppNumpyCorr(a1, a2)

    @unittest.skip(notImplMsg)
    def test_onesided(self):
        '''Test the one-sided version of ``corr``.'''
        pass

    @unittest.skip(notImplMsg)
    def test_twosided(self):
        '''Test the two-sided version of ``corr``.'''
        pass

    @unittest.skip(notImplMsg)
    def test_range(self):
        '''Test the ranged version of ``corr``.'''
        pass

    def test_zero_len(self):
        '''Test that an exception is raised when inputs have zero length.'''
        a1 = np.array([])
        a2 = np.arange(10)

        # corr(a1, a2)
        lag_start = 0
        lag_end   = 0
        for mode in ("onesided", "twosided", "range"):
            self.assertRaises(TypeError, asignal.corr, a1, a2, mode, lag_start,
                              lag_end)
            self.assertRaises(TypeError, asignal.corr, a2, a1, mode, lag_start,
                              lag_end)
            self.assertRaises(TypeError, asignal.corr, a1, a1, mode, lag_start,
                              lag_end)
