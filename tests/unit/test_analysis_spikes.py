import collections
import numpy as np
import pytest

from gridcells.analysis.spikes import PopulationSpikes
from base.spikes import RateDefinedSpikeGenerator

import base

not_impl_msg = "Not implemented yet."


##############################################################################
# Spike analysis tests (analysis.spikes)
def _compute_output_sequence(train1, train2):
    res = np.array([])
    for t1 in train1:
        res = np.hstack((res, train2 - t1))
    return res


def _create_test_sequence(train_size, n_trains):
    '''
    Create a test sequence of ``n_trains`` spike trains with exactly
    ``train_size`` number of spikes. Spike times are random, but **time
    sorted**.
    '''
    senders = np.repeat(np.arange(n_trains), train_size)
    np.random.shuffle(senders)
    times = np.random.rand(train_size * n_trains)
    times.sort()
    sp = PopulationSpikes(n_trains, senders, times)
    return senders, times, sp


@pytest.fixture(params=[100., 200., 300.])
def fix_spike_gen(request):
    return RateDefinedSpikeGenerator(0, 100., request.param)


class TestSlidingFiringRate:
    def test_empty(self):
        for N in range(0, 4):
            pop = PopulationSpikes(N, [], [])
            r, rt = pop.sliding_firing_rate(0, 1, .1, .2)
            assert np.all(r == 0)

    def test_rates(self, fix_spike_gen):
        gen = fix_spike_gen

        # Single neuron
        item = gen.get_test_vector([[0, 1, 2]])
        r, rt = item.pop.sliding_firing_rate(gen.tstart, item.tend, gen.dt,
                                             gen.winlen)
        np.testing.assert_allclose(r, item.test_rates)
        np.testing.assert_allclose(rt, np.arange(3) * gen.dt)

        # Single neuron, len = 4
        item = gen.get_test_vector([[0, 1, 2, 6]])
        r, rt = item.pop.sliding_firing_rate(gen.tstart, item.tend, gen.dt,
                                             gen.winlen)
        np.testing.assert_allclose(r, item.test_rates)
        np.testing.assert_allclose(rt, np.arange(4) * gen.dt)

        # 3 neurons, this should be enough to triangulate
        item = gen.get_test_vector(
            [[0, 1, 2],
             [1, 2, 3],
             [3, 2, 1]])
        r, rt = item.pop.sliding_firing_rate(gen.tstart, item.tend, gen.dt,
                                             gen.winlen)
        np.testing.assert_allclose(r, item.test_rates)
        np.testing.assert_allclose(rt, np.arange(3) * gen.dt)


class TestPopulationSpikes:
    '''
    Unit tests of :class:`analysis.spikes.PopulationSpikes`.
    '''

    def test_negative_n(self):
        with pytest.raises(ValueError):
            PopulationSpikes(-10, [], [])

    def test_zero_n(self):
        PopulationSpikes(0, [], [])

    @base.notimpl
    def test_avg_firing_rate(self):
        pass

    def test_lists(self):
        train_size = 100
        n = 10
        senders = list(np.random.randint(n, size=train_size * n))
        times = list(np.random.rand(train_size * n))
        sp = PopulationSpikes(n, senders, times)

        # try to retrieve spike trains
        for nIdx in xrange(n):
            train = sp[nIdx]
            assert train is not None

        # Try to run all the methods, None should raise an exception
        sp.avg_firing_rate(0, 1)
        sp.sliding_firing_rate(0, 1, 0.05, 0.1)
        sp.windowed((0, 1))
        sp.raster_data()
        sp.spike_train_difference(range(n))


class TestSpikeTrainDifference:

    def test_full(self):
        # full must be True
        n = 10
        senders, times, sp = _create_test_sequence(0, n)
        with pytest.raises(NotImplementedError):
            sp.spike_train_difference(1, 10, False)

    def test_empty(self):
        # Empty spike trains
        n = 10
        senders, times, sp = _create_test_sequence(0, n)
        std = sp.spike_train_difference
        res = std(1, 2, True)
        assert isinstance(res, collections.Sequence)
        assert len(res) == 1
        assert res[0].shape[0] == 0

    def test_out_of_bounds(self):
        # Out of bounds spike trains
        train_size = 100
        n = 100
        senders, times, sp = _create_test_sequence(train_size, n)

        with pytest.raises(IndexError):
            sp.spike_train_difference(n, 1)
        with pytest.raises(IndexError):
            sp.spike_train_difference(1, n)

        # boundaries
        sp.spike_train_difference(n - 1, 1)
        sp.spike_train_difference(1, n - 1)
        sp.spike_train_difference(-1, 0)
        sp.spike_train_difference(0, -1)

    def test_result_length(self):
        train_size = 100
        n = 100
        senders, times, sp = _create_test_sequence(train_size, n)
        std = sp.spike_train_difference

        # result length must be correct
        train_lens = [np.count_nonzero(senders == x) for x in xrange(n)]
        res = std(range(n), None, True)
        assert len(res) == n
        for nIdx in xrange(n):
            assert len(res[nIdx]) == n

        for n1 in xrange(n):
            for n2 in xrange(n):
                expected_len = train_lens[n1] * train_lens[n2]
                assert len(res[n1][n2]) == expected_len

    def test_correct_values(self):
        train_size = 100
        n = 50
        senders, times, sp = _create_test_sequence(train_size, n)
        std = sp.spike_train_difference
        res = std(range(n), None, True)

        for n1 in xrange(n):
            train1 = times[senders == n1]
            for n2 in xrange(n):
                train2 = times[senders == n2]
                diff = res[n1][n2]
                expected_diff = _compute_output_sequence(train1, train2)
                assert np.all(diff == expected_diff)


class TestSpikeTrainXCorrelation:

    def test_bin_edges(self):
        train_size = 100
        n = 50
        bins = 37
        senders, times, sp = _create_test_sequence(train_size, n)
        xcf = sp.spike_train_xcorr

        # trainLens = [np.count_nonzero(senders == x) for x in xrange(n)]
        res, bin_centers, bin_edges = xcf(range(n), None, (0, 1), bins)
        assert bins == len(bin_edges) - 1
        assert len(bin_centers) == bins
        for n1 in xrange(n):
            for n2 in xrange(n):
                assert len(res[n1][n2]) == bins

    @base.notimpl
    def test_correct_values(self):
        '''
        Since we are running this on numpy.histogram, it should be ok for
        these purposes.
        '''
        pass


class TestISI:

    def test_empty(self):
        # empty spike trains
        n = 100
        for nSpikes in [0, 1]:
            senders, times, sp = _create_test_sequence(nSpikes, n)
            res = sp.isi()
            assert len(res) == n
            for ISIs in res:
                assert len(ISIs) == 0

    def test_results_length(self):
        train_size = 101
        n = 100
        senders, times, sp = _create_test_sequence(train_size, n)
        res = sp.isi()
        for ISIs in res:
            assert len(ISIs) == train_size - 1

    def test_positive(self):
        train_size = 1000
        n = 100
        senders, times, sp = _create_test_sequence(train_size, n)
        res = sp.isi()
        for ISIs in res:
            assert np.all(ISIs >= 0)

    def test_constant_isi(self):
        '''
        .. todo::

            this will only work if dt = 2^x. For now it should be enough.
        '''
        max_size = 1011
        dt = 0.25
        for train_size in xrange(2, max_size):
            senders = [0] * train_size
            times = np.arange(train_size, dtype=float) * dt
            sp = PopulationSpikes(1, senders, times)
            res = sp.isi(n=0)
            assert np.all(res[0] == dt)


class TestISICV:

    def test_empty(self):
        # empty spike trains
        n = 137
        for nSpikes in [0, 1]:
            senders, times, sp = _create_test_sequence(nSpikes, n)
            res = sp.isi_cv()
            assert len(res) == n

    def test_results_length(self):
        train_size = 101
        n = 137
        senders, times, sp = _create_test_sequence(train_size, n)
        res = sp.isi_cv()
        assert len(res) == n
        for CV in res:
            assert isinstance(CV, int) or isinstance(CV, float)

    def test_positive(self):
        train_size = 10
        n = 137
        senders, times, sp = _create_test_sequence(train_size, n)
        res = sp.isi_cv()
        assert np.all(np.asarray(res) >= 0)
