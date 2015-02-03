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

    def _data_generator(self, n_items, sz):
        '''Generate pairs of test vectors.'''
        it = 0
        while it < n_items:
            N1 = np.random.randint(self.maxN) + 1
            N2 = np.random.randint(self.maxN) + 1
            if N1 == 0 and N2 == 0:
                continue

            a1 = np.random.rand(N1)
            a2 = np.random.rand(N2)

            yield (a1, a2)

            it += 1

    def test_onesided(self):
        '''Test the one-sided version of ``corr``.'''
        for a1, a2 in self._data_generator(self.maxLoops, self.maxN):
            c_cpp = asignal.corr(a1, a2, mode='onesided')
            c_np = np.correlate(a1, a2, mode='full')[::-1][a1.size - 1:]
            np.testing.assert_allclose(c_cpp, c_np, rtol=self.rtol)

    def test_twosided(self):
        '''Test the two-sided version of ``corr``.'''
        for a1, a2 in self._data_generator(self.maxLoops, self.maxN):
            c_cpp = asignal.corr(a1, a2, mode='twosided')
            c_np = np.correlate(a1, a2, mode='full')[::-1]
            np.testing.assert_allclose(c_cpp, c_np, rtol=self.rtol)

    def test_range(self):
        '''Test the ranged version of ``corr``.'''
        # Half the range of both signals
        for a1, a2 in self._data_generator(self.maxLoops, self.maxN):
            if a1.size <= 1 or a2.size <= 1:
                continue
            lag_start =  - (a1.size // 2)
            lag_end = a2.size // 2
            c_np_centre = a1.size - 1
            c_cpp = asignal.corr(a1, a2, mode='range', lag_start=lag_start,
                                 lag_end = lag_end)
            c_np = np.correlate(a1, a2, mode='full')[::-1]
            np.testing.assert_allclose(
                c_cpp,
                c_np[c_np_centre + lag_start:c_np_centre + lag_end + 1],
                rtol=self.rtol)


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

    def test_non_double(self):
        '''Test the corr function when dtype is not double.'''
        a1 = np.array([1, 2, 3], dtype=int)
        asignal.corr(a1, a1, mode='twosided')
