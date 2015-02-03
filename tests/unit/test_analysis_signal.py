'''Test the analysis.signal module.'''
from __future__ import absolute_import, print_function, division

import pytest
import numpy as np
import gridcells.analysis.signal as asignal

import base

notImplMsg = "Not implemented"


class TestCorrelation(object):
    '''
    Test the analysis.signal.corr function (and effectively the core of the
    autoCorrelation) function.
    '''
    rtol = 1e-10
    maxN = 1000
    maxLoops = 1000

    def checkCppNumpyCorr(self, a1, a2):
        '''Check whether the cpp version gives approximately equal results when
        compared to numpy.'''
        c_cpp = asignal.corr(a1, a2, mode='twosided')
        c_np    = np.correlate(a1, a2, mode='full')[::-1]
        np.testing.assert_allclose(c_cpp, c_np, rtol=self.rtol)

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

    @base.notimpl
    def test_onesided(self):
        '''Test the one-sided version of ``corr``.'''
        pass

    @base.notimpl
    def test_twosided(self):
        '''Test the two-sided version of ``corr``.'''
        pass

    @base.notimpl
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
            with pytest.raises(TypeError):
                asignal.corr(a1, a2, mode, lag_start, lag_end)
            with pytest.raises(TypeError):
                asignal.corr(a2, a1, mode, lag_start, lag_end)
            with pytest.raises(TypeError):
                asignal.corr(a1, a1, mode, lag_start, lag_end)
