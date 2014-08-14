'''
===============================================================
:mod:`gridcells.analysis.spikes` - spike train analysis
===============================================================

Classes
-------
.. autosummary::

    SpikeTrain
    PopulationSpikes
    TorusPopulationSpikes
    TwistedTorusSpikes
'''
import os

import numpy as np
import scipy
import collections
from scipy import weave

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not on_rtd:
    from . import _spikes

__all__ = ['SpikeTrain', 'PopulationSpikes', 'TorusPopulationSpikes',
           'TwistedTorusSpikes']


def slidingFiringRateTuple(spikes, N, tstart, tend, dt, winLen):
    '''
    Compute a firing rate with a sliding window from a tuple of spike data:
    spikes is a tuple(n_id, times), in which n_id is a list/array of neuron id
    and times is a list/array of spike times

    Parameters
    ----------
    spikes : A pair (n_id, spikes)
        Spike times
    N : int
        Total number of neurons
    tstart : float
        When the firing rate will start (ms)
    tend : float
        End time of firing rate (ms)
    dt : float
        Sliding window dt - not related to simulation time (ms)
    winLen : float
        Length of the sliding window (ms). Must be >= dt.

    Returns
    -------
    rates :  np.ndarray
        An array of shape (N, int((tend-tstart)/dt)+1.
    '''
    tstart = float(tstart)
    tend   = float(tend)
    dt     = float(dt)
    winLen = float(winLen)
    
    szRate      = int((tend-tstart)/dt)+1
    n_ids       = np.array(spikes[0])
    spikeTimes  = np.array(spikes[1])
    lenSpikes   = len(spikeTimes)
    bitSpikes   = np.zeros((N, szRate))
    fr          = np.zeros((N, szRate))
    dtWlen      = int(winLen/dt)
    times       = np.arange(tstart, tend+dt, dt)
    N           = int(N)

    #print "max(n_ids): ", np.max(n_ids)
    #print 'szRate: ', szRate
    #print 'N: ', N
    #print 'dtWlen: ', dtWlen

    code = """
        for (int i = 0; i < lenSpikes; i++)
        {
            int spikeSteps = (spikeTimes(i) - tstart) / dt;
            if (spikeSteps >= 0 && spikeSteps < szRate)
            {
                int n_id = n_ids(i);
                bitSpikes(n_id, spikeSteps) += 1;
            }
        }

        for (int n_id = 0; n_id < N; n_id++)
            for (int t = 0; t < szRate; t++)
            {
                fr(n_id, t) = .0;
                for (int s = 0; s < dtWlen; s++)
                    if ((t+s) < szRate)
                        fr(n_id, t) += bitSpikes(n_id, t+s);
            }
        """

    err = weave.inline(code,
            ['N', 'szRate', 'dtWlen', 'lenSpikes', 'n_ids', 'spikeTimes',
                'tstart', 'dt', 'bitSpikes', 'fr'],
            type_converters=weave.converters.blitz,
            compiler='gcc',
            extra_compile_args=['-O3'],
            verbose=2)

    return fr/(winLen*1e-3), times


class SpikeTrain(object):
    '''
    A base class for handling spike trains.
    '''
    def __init__(self):
        raise NotImplementedError()


class PopulationSpikes(SpikeTrain, collections.Sequence):
    '''
    Class to handle a population of spikes and a set of methods to do analysis
    on the *whole* population.
    '''
    def __init__(self, N, senders, times):
        '''
        Parameters
        ----------
        N : int
            Number of neurons in the population
        senders : 1D array
            Neuron numbers corresponding to the spikes
        times : 1D array
            Spike times. The shape of this array must be the same as for
            `senders`.
        '''
        self._N        = N
        if (N < 0):
            msg = "Number of neurons in the spike train must be " +\
                    "non-negative! Got {0}."
            raise ValueError(msg.format(N))

        # We are expecting senders and times as numpy arrays, if they are not,
        # convert them. Moreover, senders.dtype must be int, for indexing.
        self._senders  = np.asarray(senders, dtype=int)
        self._times    = np.asarray(times)
        self._unpacked = [None] * self._N # unpacked version of spikes

    @property
    def N(self):
        '''
        Number of neurons in the population
        '''
        return self._N

    def avgFiringRate(self, tStart, tEnd):
        '''
        Compute and average firing rate for all the neurons between 'tstart'
        and 'tend'. Return an array of firing rates, one item for each neuron
        in the population.

        Parameters
        ----------
        tStart : float (ms)
            Start time.
        tEnd   : float (ms)
            End time.

        Returns
        -------
        output : numpy.ndarray
            Firing rate in Hz for each neuron in the population.
        '''
        result  = np.zeros((self._N, ))
        times   = self._times
        senders = self._senders
        N       = int(self._N)
        ts      = float(tStart)
        te      = float(tEnd)
        code = '''
            for (int i = 0; i < senders.size(); i++)
            {
                int t = times(i);
                int s = senders(i);
                if (s >= 0 && s < N && t >= ts && t <= te)
                    result(s)++; 
                else if (s < 0 || s >= N)
                    std::cout << "senders is outside range <0, N)" << 
                            std::endl;
            }
        '''

        err = weave.inline(code,
                ['N', 'times', 'senders', 'ts', 'te', 'result'],
                type_converters=weave.converters.blitz,
                compiler='gcc',
                extra_compile_args=['-O3'],
                verbose=2)
        return 1e3 * result / (tEnd - tStart)

    def slidingFiringRate(self, tStart, tEnd, dt, winLen):
        '''
        Compute a sliding firing rate over the population of spikes, by taking
        a rectangular window of specified length.

        Parameters
        ----------
        tStart : float
            Start time of the firing rate analysis.
        tEnd : float
            End time of the analysis
        dt : float
            Firing rate window time step
        winLen : float
            Lengths of the windowing function (rectangle)

        Returns
        -------
        output : tuple
            A pair (F, t), specifying the vector of firing rates and
            corresponding times. F is a 2D array of the shape (N, Ntimes), in
            which N is the number of neurons and Ntimes is the number of time
            steps. 't' is a vector of times corresponding to the time windows
            taken.
        '''
        spikes = (self._senders, self._times)
        return slidingFiringRateTuple(spikes, self._N, tStart, tEnd, dt,
                winLen)

    def windowed(self, tLimits):
        '''
        Return population spikes restricted to tLimits.

        Parameters
        ----------
        tLimits : a pair
            A tuple (tStart, tEnd). The spikes in the population must satisfy
            tStart >= t <= tEnd.

        Returns
        -------
        output : PopulationSpikes
            A copy of self with only a subset of spikes, limited by the time
            window.
        '''
        tStart = tLimits[0]
        tEnd   = tLimits[1]
        tIdx = np.logical_and(self._times >= tStart, self._times <= tEnd)
        return PopulationSpikes(self._N, self._senders[tIdx],
                self._times[tIdx])

    def rasterData(self, neuronList=None):
        '''
        Extract the senders and corresponding spike times for a raster plot.
        TODO: implement neuronList

        Parameters
        ==========
        neuronList : list, optional
            Extract only neurons given in this list

        Returns
        -------
        output : a tuple
            A pair containing (senders, times).
        '''
        if (neuronList is not None):
            raise NotImplementedError()

        return self._senders, self._times

    def spikeTrainDifference(self, idx1, idx2=None, full=True, reduceFun=None):
        '''
        Compute time differences between pairs of spikes of two neurons or a
        list of neurons.
        
        Parameters
        ----------
        idx1 : int, or a sequence of ints
            Index of the first neuron or a list of neurons for which to compute
            the correlation histogram.
        idx2 : int, or a sequence of ints, or None
            Index of the second neuron or a list of indexes for the second set
            of spike trains.
        full : bool, optional
            Not fully implemented yet. Must be set to True.
        reduceFun : callable, optional
            Any callable object that computes a function over an array of each
            spike train difference. The function must take one input argument,
            which will be the array of spike time differences for a pair of
            neurons. The output of this function will be stored instead of the
            default output.

        Returns
        -------
        output : A 2D or 1D array of spike train autocorrelation histograms for all
            the pairs of neurons.
        
        The computation takes the following steps:
        
         * If ``idx1`` or ``idx2`` are integers, they will be converted to a list
           of size 1.
         * If ``idx2`` is None, then the result will be a list of lists of pairs
           of cross-correlations between the neurons. Even if there is only one
           neuron. If ``full == True``, the output will be an upper triangular
           matrix of all the pairs, i.e. it will exclude the duplicated.
           Otherwise there will be cross correlation histograms between all the
           pairs.
         * if ``idx2`` is not None, then ``idx1`` and ``idx2`` must be arrays
           of the same length, specifying the pairs to compute autocorrelation
           for
        '''
        if (full == False):
            raise NotImplementedError()

        if (reduceFun is None):
            reduceFun = lambda x: x

        if (not isinstance(idx1, collections.Iterable)):
            idx1 = [idx1]

        if (idx2 is None):
            idx2 = idx1
            res = [[] for x in idx1]
            for n1 in idx1:
                for n2 in idx2:
                    #print n1, n2, len(self[n1]), len(self[n2])
                    res[n1].append(reduceFun(_spikes.spike_time_diff(self[n1],
                        self[n2])))
            return res
        elif (not isinstance(idx2, collections.Iterable)):
            idx2 = [idx2]

        # Two arrays of pairs
        if (len(idx1) != len(idx2)):
            raise TypeError('Length of neuron indexes do not match!')

        res = [None] * len(idx1)
        for n in xrange(len(idx1)):
            res[n] = reduceFun(_spikes.spike_time_diff(self[idx1[n]],
                self[idx2[n]]))

        return res


    class CallableHistogram(object):
        def __init__(self, **kw):
            self.kw = kw

        def __call__(self, x):
            '''
            Perform the histogram on x and return the result of
            numpy.histogram, without bin_edges
            '''
            res, _ = np.histogram(x, **self.kw)
            return res

        def get_bin_edges(self):
            _, bin_edges = np.histogram([], **self.kw)
            return bin_edges

    def spikeTrainXCorrelation(self, idx1, idx2, range, bins=50,
            **kw):
        '''
        Compute the spike train crosscorrelation function for all pairs of
        spike trains in the population.

        For explanation of how ``idx1`` and ``idx2`` are treated, see
        :meth:`~PopulationSpikes.spikeTrainDifference`.

        Parameters
        ----------
        idx1 : int, or a sequence of ints
            Index of the first neuron or a list of neurons for which to compute
            the correlation histogram.
        idx2 : int, or a sequence of ints, or None
            Index of the second neuron or a list of indexes for the second set
            of spike trains.
        range : (lag_start, lag_end)
            Limits of the cross-correlation function. The bins will always be
            **centered** on the values.
        bins : int, optional
            Number of bins
        kw : dict
            Keyword arguments passed on to the numpy.histogram function

        Returns
        -------
        output : a 2D or 1D list
            see :meth:`~PopulationSpikes.spikeTrainDifference`.
        '''
        lag_start = range[0]
        lag_end   = range[1]
        binWidth = (lag_end - lag_start) / (bins - 1)
        bin_edges = np.linspace(lag_start - binWidth/2.0, lag_end +
                binWidth/2.0, bins+1) 
        h = self.CallableHistogram(bins=bin_edges, **kw)
        XC = self.spikeTrainDifference(idx1, idx2, full=True, reduceFun=h)
        bin_edges = h.get_bin_edges()
        bin_centers = (bin_edges[0:-1] + bin_edges[1:])/2.0
        return XC, bin_centers, bin_edges
        
    def ISINeuron(self, n):
        '''
        Compute all interspike intervals of one neuron with ID ``n``. If the
        number of spikes is less than 2, returns an empty array.
        .. todo::
            
            Works on sorted spike trains only!

        .. note::
            If you get negative interspike intervals, you will need to sort
            your spike times (per each neuron).
        '''
        spikes = self[n]
        if (len(spikes) < 2):
            return np.array([])
        return spikes[1:] - spikes[0:-1]

    def ISI(self, n=None, reduceFun=None):
        '''
        Return interspike interval of one or more neurons.

        Parameters
        ----------
        n : None, int, or sequence 
            Neuron numbers. If ``n`` is None, then compute ISI stats for all
            neurons in the population. If ``n`` is an int, compute ISIs for
            just neuron indexed by ``n``. Otherwise ``n`` is expected to be a
            sequence of neuron indices.
        reduceFun : callable or None
            A reduction function (callable object) that performs an operation
            on all the ISIs of the population. If ``None``, nothing is done.
            The callable has to take one input parameter, which is the sequence
            of ISIs. This allows to cascade data processing without the need
            for duplicating spike timing data.

        Returns
        -------
        output: list
            A list of outputs (depending on parameters) for each neuron, even if ``n``
            is an int.
        '''
        if (reduceFun is None):
            reduceFun = lambda x: x

        res = []
        if (n is None):
            for n_id in xrange(len(self)):
                res.append(reduceFun(self.ISINeuron(n_id)))
        elif (isinstance(n, int)):
            res.append(reduceFun(self.ISINeuron(n)))
        else:
            for n_id in n:
                res.append(reduceFun(self.ISINeuron(n_id)))

        return res

    def ISICV(self, n=None, winLen=None):
        '''
        Coefficients of variation of inter-spike intervals of one or more
        neurons in the population. For the description of parameters and
        outputs and their semantics see also :meth:`~PopulationSpikes.ISI`.

        Parameters
        ----------
        ``winLen`` : float, list of floats, or ``None``
            Specify the maximal ISI value, i.e. use windowed coefficient of
            variation. If ``None``, use the whole range.
        '''
        cvfunc = scipy.stats.variation
        if (winLen is None):
            f = scipy.stats.variation
        elif (isinstance(winLen, collections.Sequence) or
                isinstance(winLen, np.ndarray)):
            f = lambda x: np.asarray([cvfunc(x[x <= wl]) for wl in winLen])
        else:
            f = lambda x: cvfunc(x[x <= winLen])
        return self.ISI(n, f)



    #######################################################################
    # Methods implementing collections.Sequence
    def __getitem__(self, key):
        '''Retrieve a spike train for one neuron.'''
        if self._unpacked[key] is not None:
            return self._unpacked[key]
        ret = self._times[self._senders == key]
        self._unpacked[key] = ret
        return ret

    def __len__(self):
        return self._N


class TorusPopulationSpikes(PopulationSpikes):
    '''
    Spikes of a population of neurons on a twisted torus.
    '''
    def __init__(self, senders, times, sheetSize):
        self._sheetSize = sheetSize
        N = sheetSize[0]*sheetSize[1]
        PopulationSpikes.__init__(self, N, senders, times)

    def getXSize(self):
        return self._sheetSize[0]
    def getYSize(self):
        return self._sheetSize[1]
    def getDimensions(self):
        return self._sheetSize

    Nx = property(fget=getXSize, doc='Horizontal size of the torus')
    Ny = property(fget=getYSize, doc='Vertical size of the torus')
    dimensions = property(fget=getDimensions, doc='Dimensions of the torus (X, Y)')

    def avgFiringRate(self, tStart, tEnd):
        F = super(TorusPopulationSpikes, self).avgFiringRate(tStart, tEnd)
        return np.reshape(F, (self.Ny, self.Nx))

    def populationVector(self, tStart, tEnd, dt, winLen):
        '''
        Compute the population vector on a torus, from the spikes present. Note
        that this method will have a limited functionality on a twisted torus,
        but can be used if the population activity translates in the X
        dimension only.

        Parameters
        ----------
        tStart : float
            Start time of analysis
        tEnd : float
            End time of analysis
        dt : float
            Time step of the (rectangular) windowing function
        winLen : float
            Length of the windowing function

        Returns
        -------
        output : tuple
            A pair (r, t) in which r is a 2D vector of shape
            (int((tEnd-tStart)/dt)+1), 2), corresponding to the population
            vector for each time step of the windowing function, and t is a
            vector of times, of length the first dimension of r.
        '''
        sheetSize_x = self.getXSize()
        sheetSize_y = self.getYSize()
        N = sheetSize_x*sheetSize_y
        
        F, tsteps = PopulationSpikes.slidingFiringRate(self, tStart, tEnd, dt,
                winLen)
        P = np.ndarray((len(tsteps), 2), dtype=complex)
        X, Y = np.meshgrid(np.arange(sheetSize_x), np.arange(sheetSize_y))
        X = np.exp(1j*(X - sheetSize_x/2)/sheetSize_x*2*np.pi).ravel()
        Y = np.exp(1j*(Y - sheetSize_y/2)/sheetSize_y*2*np.pi).ravel()
        for t_it in xrange(len(tsteps)):
            P[t_it, 0] = np.dot(F[:, t_it], X)
            P[t_it, 1] = np.dot(F[:, t_it], Y)

        return (np.angle(P)/2/np.pi*self._sheetSize, tsteps)

    def slidingFiringRate(self, tStart, tEnd, dt, winLen):
        '''
        Compute a sliding firing rate over the population of spikes, by taking
        a rectangular window of specified length. However, unlike the ancestor
        method (PopulationSpikes.slidingFiringRate), return a 3D array, a
        succession of 2D population firing rates in time.

        Parameters
        ----------
        tStart : float
            Start time of the firing rate analysis.
        tEnd : float
            End time of the analysis
        dt : float
            Firing rate window time step
        winLen : float
            Lengths of the windowing function (rectangle)

        Returns
        -------
        output : a tuple
            A pair (F, t), specifying the vector of firing rates and
            corresponding times. F is a 3D array of the shape (Nx, Ny, Ntimes),
            in which Nx/Ny are the number of neurons in X and Y dimensions,
            respectively, and Ntimes is the number of time steps. 't' is a
            vector of times corresponding to the time windows taken.
        '''
        spikes = (self._senders, self._times)
        F, Ft = slidingFiringRateTuple(spikes, self._N, tStart, tEnd, dt,
                winLen)
        Nx = self.getXSize()
        Ny = self.getYSize()
        return np.reshape(F, (Ny, Nx, len(Ft))), Ft


class TwistedTorusSpikes(TorusPopulationSpikes):
    '''
    Spikes arranged on twisted torus. The torus is twisted in the X direction.
    '''
    def __init__(self, senders, times, sheetSize):
        super(TwistedTorusSpikes, self).__init__(senders, times, sheetSize)

    def populationVector(self, tStart, tEnd, dt, winLen):
        msg = 'populationVector() has not been implemented yet for {}. Note'+\
                ' that this method is different for the regular torus ' +\
                '(TorusPopulationSpikes).'
        raise NotImplementedError(msg.format(self.__class__.__name__))
