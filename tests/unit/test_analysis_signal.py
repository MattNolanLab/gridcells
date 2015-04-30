'''Test the analysis.signal module.'''
from __future__ import absolute_import, print_function, division

import pytest
import numpy as np
import gridcells.analysis.signal as asignal
from gridcells.analysis.signal import local_extrema, local_maxima, local_minima


RTOL = 1e-10


def _data_generator(n_items, sz):
    '''Generate pairs of test vectors.'''
    it = 0
    while it < n_items:
        N1 = np.random.randint(sz) + 1
        N2 = np.random.randint(sz) + 1
        if N1 == 0 and N2 == 0:
            continue

        a1 = np.random.rand(N1)
        a2 = np.random.rand(N2)

        yield (a1, a2)

        it += 1


class TestCorrelation(object):
    '''
    Test the analysis.signal.corr function (and effectively the core of the
    autoCorrelation) function.
    '''
    maxN = 500
    maxLoops = 1000

    def test_onesided(self):
        '''Test the one-sided version of ``corr``.'''
        for a1, a2 in _data_generator(self.maxLoops, self.maxN):
            c_cpp = asignal.corr(a1, a2, mode='onesided')
            c_np = np.correlate(a1, a2, mode='full')[::-1][a1.size - 1:]
            np.testing.assert_allclose(c_cpp, c_np, rtol=RTOL)

    def test_twosided(self):
        '''Test the two-sided version of ``corr``.'''
        for a1, a2 in _data_generator(self.maxLoops, self.maxN):
            c_cpp = asignal.corr(a1, a2, mode='twosided')
            c_np = np.correlate(a1, a2, mode='full')[::-1]
            np.testing.assert_allclose(c_cpp, c_np, rtol=RTOL)

    def test_range(self):
        '''Test the ranged version of ``corr``.'''
        # Half the range of both signals
        for a1, a2 in _data_generator(self.maxLoops, self.maxN):
            if a1.size <= 1 or a2.size <= 1:
                continue
            lag_start = - (a1.size // 2)
            lag_end = a2.size // 2
            c_np_centre = a1.size - 1
            c_cpp = asignal.corr(a1, a2, mode='range', lag_start=lag_start,
                                 lag_end=lag_end)
            c_np = np.correlate(a1, a2, mode='full')[::-1]
            np.testing.assert_allclose(
                c_cpp,
                c_np[c_np_centre + lag_start:c_np_centre + lag_end + 1],
                rtol=RTOL)

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


class TestAutoCorrelation(object):
    '''Test the acorr function.'''
    maxN = 500
    maxLoops = 1000

    def test_default_params(self):
        '''Test default parameters.'''
        a = np.arange(10)
        c_cpp = asignal.acorr(a)
        c_np = np.correlate(a, a, mode='full')[::-1][a.size - 1:]
        np.testing.assert_allclose(c_cpp, c_np, rtol=RTOL)

    def test_onesided(self):
        '''Test the one-sided version of ``corr``.'''
        a = np.arange(10)
        c_cpp = asignal.acorr(a, mode='onesided', max_lag=5)
        c_np = np.correlate(a, a, mode='full')[::-1][a.size - 1:a.size - 1 + 6]
        np.testing.assert_allclose(c_cpp, c_np, rtol=RTOL)

    def test_twosided(self):
        '''Test the two-sided version of ``corr``.'''
        a = np.arange(10)
        c_cpp = asignal.acorr(a, mode='twosided', max_lag=5)
        c_np = np.correlate(a, a, mode='full')[::-1][a.size - 6:a.size + 5]
        np.testing.assert_allclose(c_cpp, c_np, rtol=RTOL)

    def test_norm(self):
        '''Test normalization.'''
        # Simple array
        a = np.arange(10)
        c_cpp = asignal.acorr(a, mode='twosided', norm=True)
        c_np = np.correlate(a, a, mode='full')[::-1]
        np.testing.assert_allclose(c_cpp, c_np / np.max(c_np), rtol=RTOL)

        # A zero array will return zero
        zero_array = np.zeros(13)
        c_cpp = asignal.acorr(zero_array, mode='twosided', norm=True)
        assert np.all(c_cpp == 0.)


def generate_sin(n_half_cycles, resolution=100):
    '''Generate a sine function with a number of (full) half cycles.

    Note that the positions of the extrema might be shifted +/- 1 with respect
    to the actual real sin because of possible rounding errors.

    Parameters
    ----------
    n_half_cycles : int
        Number of half cycles to generate. Does not have to be even.
    resolution : int
        Number of data points for each half cycle.
    '''
    if n_half_cycles < 1:
        raise ValueError()
    if resolution < 1:
        raise ValueError()

    f = 1. / (2 * resolution)
    t = np.arange(n_half_cycles * resolution, dtype=float)
    sig = np.sin(2 * np.pi * f * t)
    extrema_positions = np.array(np.arange(n_half_cycles) * resolution +
                                 resolution / 2,
                                 dtype=int)

    extrema_types = []
    current_type = 1
    for _ in range(n_half_cycles):
        extrema_types.append(current_type)
        current_type = -1 if current_type == 1 else 1

    return (sig, extrema_positions, np.array(extrema_types))


class TestLocalExtrema(object):
    '''Test computation of local extrema.'''

    def test_local_extrema(self):
        for n_extrema in [1, 2, 51]:
            sig, extrema_idx, extrema_types = generate_sin(n_extrema)
            extrema, etypes = local_extrema(sig)
            assert len(extrema) == n_extrema
            assert len(etypes) == n_extrema
            assert np.all(extrema_idx == extrema)
            assert np.all(extrema_types == etypes)

    def test_zero_array(self):
        extrema, etypes = local_extrema(np.empty(0))
        assert len(extrema) == 0
        assert len(etypes) == 0

        for func in [local_maxima, local_minima]:
            extrema = func(np.empty(0))
            assert len(extrema) == 0

    def test_single_item(self):
        '''This should return a zero length array.'''
        extrema, etypes = local_extrema(np.array([1.]))
        assert len(extrema) == 0
        assert len(etypes) == 0

        for func in [local_maxima, local_minima]:
            extrema = func(np.array([1.]))
            assert len(extrema) == 0

    def test_maxima(self):
        # One maximum only
        for n_extrema in [1, 2]:
            sig, extrema_idx, extrema_types = generate_sin(n_extrema)
            maxima = local_maxima(sig)
            assert len(maxima) == 1
            assert np.all(extrema_idx[extrema_types == 1] == maxima)

        # 2 maxima
        for n_extrema in [3, 4]:
            sig, extrema_idx, extrema_types = generate_sin(n_extrema)
            maxima = local_maxima(sig)
            assert len(maxima) == 2
            assert np.all(extrema_idx[extrema_types == 1] == maxima)

    def test_minima(self):
        # Only one maximum so should return empty
        n_extrema = 1
        sig, extrema_idx, extrema_types = generate_sin(n_extrema)
        minima = local_minima(sig)
        assert len(minima) == 0
        assert np.all(extrema_idx[extrema_types == -1] == minima)

        # One maximum and minimum
        n_extrema = 2
        sig, extrema_idx, extrema_types = generate_sin(n_extrema)
        minima = local_minima(sig)
        assert len(minima) == 1
        assert np.all(extrema_idx[extrema_types == -1] == minima)

        # 2 minima
        for n_extrema in [4, 5]:
            sig, extrema_idx, extrema_types = generate_sin(n_extrema)
            minima = local_minima(sig)
            assert len(minima) == 2
            assert np.all(extrema_idx[extrema_types == -1] == minima)
