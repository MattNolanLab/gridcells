import collections
import numpy as np
import unittest

from gridcells.analysis.spikes import PopulationSpikes


##############################################################################
# Spike analysis tests (analysis.spikes)

def _computeOutputSequence(train1, train2):
    res = np.array([])
    for t1 in train1:
        res = np.hstack((res, train2 - t1))
    return res

def _createTestSequence(trainSize, N):
    '''
    Create a test sequence of ``N`` spike trains with exactly ``trainSize``
    number of spikes. Spike times are random, but **time sorted**.
    '''
    senders = np.repeat(np.arange(N), trainSize)
    np.random.shuffle(senders)
    times   = np.random.rand(trainSize * N)
    times.sort()
    sp      = PopulationSpikes(N, senders, times)
    return senders, times, sp


class TestPopulationSpikes(unittest.TestCase):
    '''
    Unit tests of :class:`analysis.spikes.PopulationSpikes`.
    '''

    def test_negative_N(self):
        self.assertRaises(ValueError, PopulationSpikes, -10, [], [])

    def test_zero_N(self):
        PopulationSpikes(0, [], [])
    

    #@unittest.skip(notImplMsg)
    #def testAvgFiringRate(self):
    #    pass


    #@unittest.skip(notImplMsg)
    #def test_slidingFiringRate(self):
    #    pass


    
    def test_lists(self):
        trainSize = 100
        N = 10
        senders = list(np.random.randint(N, size=trainSize * N))
        times   = list(np.random.rand(trainSize * N))
        sp      = PopulationSpikes(N, senders, times)

        # try to retrieve spike trains
        for nIdx in xrange(N):
            train = sp[nIdx]

        # Try to run all the methods, None should raise an exception
        sp.avgFiringRate(0, 1)
        sp.slidingFiringRate(0, 1, 0.05, 0.1)
        sp.windowed((0, 1))
        sp.rasterData()
        sp.spikeTrainDifference(range(N))




class TestSpikeTrainDifference(unittest.TestCase):

    def test_full(self):
        # full must be True
        N       = 10
        senders, times, sp = _createTestSequence(0, N)
        std     = sp.spikeTrainDifference
        self.assertRaises(NotImplementedError, std, 1, 10, False)

    def test_empty(self):
        # Empty spike trains
        N = 10
        senders, times, sp = _createTestSequence(0, N)
        std = sp.spikeTrainDifference
        res = std(1, 2, True)
        self.assertIsInstance(res, collections.Sequence)
        self.assertEqual(len(res), 1) #
        self.assertEqual(res[0].shape[0], 0)

    def test_out_of_bounds(self):
        # Out of bounds spike trains
        trainSize = 100
        N         = 100
        senders, times, sp = _createTestSequence(trainSize, N)
        std       = sp.spikeTrainDifference

        self.assertRaises(IndexError, std, N, 1)
        self.assertRaises(IndexError, std, 1, N)

        # boundaries
        std(N - 1, 1)
        std(1, N-1)
        std(-1, 0)
        std(0, -1)


    def test_result_length(self):
        trainSize = 100
        N         = 100
        senders, times, sp = _createTestSequence(trainSize, N)
        std       = sp.spikeTrainDifference

        # result length must be correct
        trainLengths = [np.count_nonzero(senders == x) for x in xrange(N)]
        res = std(range(N), None, True)
        self.assertEqual(len(res), N)
        for nIdx in xrange(N):
            self.assertEqual(len(res[nIdx]), N)

        for n1 in xrange(N):
            for n2 in xrange(N):
                expectedLen = trainLengths[n1] * trainLengths[n2]
                self.assertEqual(len(res[n1][n2]), expectedLen)

    def test_correct_values(self):
        trainSize = 100
        N         = 50
        senders, times, sp = _createTestSequence(trainSize, N)
        std       = sp.spikeTrainDifference
        res       = std(range(N), None, True)

        for n1 in xrange(N):
            train1 = times[senders == n1]
            for n2 in xrange(N):
                train2 = times[senders == n2]
                diff = res[n1][n2]
                expectedDiff = _computeOutputSequence(train1, train2)
                self.assertTrue(np.all(diff == expectedDiff))


class TestSpikeTrainXCorrelation(unittest.TestCase):

    def test_bin_edges(self):
        trainSize = 100
        N = 50
        bins = 37
        senders, times, sp = _createTestSequence(trainSize, N)
        xcf = sp.spikeTrainXCorrelation

        trainLens = [np.count_nonzero(senders == x) for x in xrange(N)]
        res, bin_centers, bin_edges = xcf(range(N), None, (0, 1), bins)
        self.assertEqual(bins, len(bin_edges) - 1)
        self.assertEqual(len(bin_centers), bins)
        for n1 in xrange(N):
            for n2 in xrange(N):
                self.assertEqual(len(res[n1][n2]), bins)


    #@unittest.skip(notImplMsg)
    #def test_correct_values(self):
    #    '''
    #    Since we are running this on numpy.histogram, it should be ok for these
    #    purposes.
    #    '''
    #    pass


class TestISI(unittest.TestCase):

    def test_empty(self):
        # empty spike trains
        N       = 100
        for nSpikes in [0, 1]:
            senders, times, sp = _createTestSequence(nSpikes, N)
            res = sp.ISI()
            self.assertEqual(len(res), N)
            for ISIs in res:
                self.assertEqual(len(ISIs), 0)


    def test_results_length(self):
        trainSize = 101
        N       = 100
        senders, times, sp = _createTestSequence(trainSize, N)
        res = sp.ISI()
        for ISIs in res:
            self.assertEqual(len(ISIs), trainSize-1)


    def test_positive(self):
        trainSize = 1000
        N         = 100
        senders, times, sp = _createTestSequence(trainSize, N)
        res = sp.ISI()
        for ISIs in res:
            self.assertTrue(np.all(ISIs >=0))


    def test_constant_ISI(self):
        '''
        .. todo::

            this will only work if dt = 2^x. For now it should be enough.
        '''
        maxSize = 1011
        dt = 0.25
        for trainSize in xrange(2, maxSize):
            senders = [0] * trainSize
            times   = np.arange(trainSize, dtype=float) * dt
            sp = PopulationSpikes(1, senders, times)
            res = sp.ISI(n=0)
            self.assertTrue(np.all(res[0] == dt))



class TestISICV(unittest.TestCase):

    def test_empty(self):
        # empty spike trains
        N       = 137
        for nSpikes in [0, 1]:
            senders, times, sp = _createTestSequence(nSpikes, N)
            res = sp.ISICV()
            self.assertEqual(len(res), N)


    def test_results_length(self):
        trainSize = 101
        N       = 137
        senders, times, sp = _createTestSequence(trainSize, N)
        res = sp.ISICV()
        self.assertEqual(len(res), N)
        for CV in res:
            self.assertTrue(isinstance(CV, int) or isinstance(CV, float))


    def test_positive(self):
        trainSize = 10
        N         = 137
        senders, times, sp = _createTestSequence(trainSize, N)
        res = sp.ISICV()
        self.assertTrue(np.all(np.asarray(res >= 0)))
