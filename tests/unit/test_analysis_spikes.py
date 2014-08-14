import collections
import numpy as np
import unittest

from gridcells.analysis.spikes import PopulationSpikes

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


class TestPopulationSpikes(unittest.TestCase):
    '''
    Unit tests of :class:`analysis.spikes.PopulationSpikes`.
    '''

    def test_negative_n(self):
        self.assertRaises(ValueError, PopulationSpikes, -10, [], [])

    def test_zero_n(self):
        PopulationSpikes(0, [], [])

    @unittest.skip(not_impl_msg)
    def test_avg_firing_rate(self):
        pass

    @unittest.skip(not_impl_msg)
    def test_sliding_firing_rate(self):
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


class TestSpikeTrainDifference(unittest.TestCase):

    def test_full(self):
        # full must be True
        n = 10
        senders, times, sp = _create_test_sequence(0, n)
        std = sp.spike_train_difference
        self.assertRaises(NotImplementedError, std, 1, 10, False)

    def test_empty(self):
        # Empty spike trains
        n = 10
        senders, times, sp = _create_test_sequence(0, n)
        std = sp.spike_train_difference
        res = std(1, 2, True)
        self.assertIsInstance(res, collections.Sequence)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].shape[0], 0)

    def test_out_of_bounds(self):
        # Out of bounds spike trains
        train_size = 100
        n = 100
        senders, times, sp = _create_test_sequence(train_size, n)
        std = sp.spike_train_difference

        self.assertRaises(IndexError, std, n, 1)
        self.assertRaises(IndexError, std, 1, n)

        # boundaries
        std(n - 1, 1)
        std(1, n - 1)
        std(-1, 0)
        std(0, -1)

    def test_result_length(self):
        train_size = 100
        n = 100
        senders, times, sp = _create_test_sequence(train_size, n)
        std = sp.spike_train_difference

        # result length must be correct
        train_lens = [np.count_nonzero(senders == x) for x in xrange(n)]
        res = std(range(n), None, True)
        self.assertEqual(len(res), n)
        for nIdx in xrange(n):
            self.assertEqual(len(res[nIdx]), n)

        for n1 in xrange(n):
            for n2 in xrange(n):
                expected_len = train_lens[n1] * train_lens[n2]
                self.assertEqual(len(res[n1][n2]), expected_len)

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
                self.assertTrue(np.all(diff == expected_diff))


class TestSpikeTrainXCorrelation(unittest.TestCase):

    def test_bin_edges(self):
        train_size = 100
        n = 50
        bins = 37
        senders, times, sp = _create_test_sequence(train_size, n)
        xcf = sp.spike_train_xcorr

        # trainLens = [np.count_nonzero(senders == x) for x in xrange(n)]
        res, bin_centers, bin_edges = xcf(range(n), None, (0, 1), bins)
        self.assertEqual(bins, len(bin_edges) - 1)
        self.assertEqual(len(bin_centers), bins)
        for n1 in xrange(n):
            for n2 in xrange(n):
                self.assertEqual(len(res[n1][n2]), bins)

    @unittest.skip(not_impl_msg)
    def test_correct_values(self):
        '''
        Since we are running this on numpy.histogram, it should be ok for
        these purposes.
        '''
        pass


class TestISI(unittest.TestCase):

    def test_empty(self):
        # empty spike trains
        n = 100
        for nSpikes in [0, 1]:
            senders, times, sp = _create_test_sequence(nSpikes, n)
            res = sp.isi()
            self.assertEqual(len(res), n)
            for ISIs in res:
                self.assertEqual(len(ISIs), 0)

    def test_results_length(self):
        train_size = 101
        n = 100
        senders, times, sp = _create_test_sequence(train_size, n)
        res = sp.isi()
        for ISIs in res:
            self.assertEqual(len(ISIs), train_size - 1)

    def test_positive(self):
        train_size = 1000
        n = 100
        senders, times, sp = _create_test_sequence(train_size, n)
        res = sp.isi()
        for ISIs in res:
            self.assertTrue(np.all(ISIs >= 0))

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
            self.assertTrue(np.all(res[0] == dt))


class TestISICV(unittest.TestCase):

    def test_empty(self):
        # empty spike trains
        n = 137
        for nSpikes in [0, 1]:
            senders, times, sp = _create_test_sequence(nSpikes, n)
            res = sp.isi_cv()
            self.assertEqual(len(res), n)

    def test_results_length(self):
        train_size = 101
        n = 137
        senders, times, sp = _create_test_sequence(train_size, n)
        res = sp.isi_cv()
        self.assertEqual(len(res), n)
        for CV in res:
            self.assertTrue(isinstance(CV, int) or isinstance(CV, float))

    def test_positive(self):
        train_size = 10
        n = 137
        senders, times, sp = _create_test_sequence(train_size, n)
        res = sp.isi_cv()
        self.assertTrue(np.all(np.asarray(res >= 0)))
