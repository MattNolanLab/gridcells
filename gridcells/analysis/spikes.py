'''
===============================================================
:mod:`gridcells.analysis.spikes` - spike train analysis
===============================================================

Classes
-------
.. inheritance-diagram:: gridcells.analysis.spikes
                         gridcells.analysis.bumps.SingleBumpPopulation
    :parts: 2

.. autosummary::

    PopulationSpikes
    TorusPopulationSpikes
    TwistedTorusSpikes
'''
from __future__ import absolute_import, division, print_function

import os

import numpy as np
import scipy
import collections

# Do not import when in RDT environment
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not on_rtd:
    from . import _spikes

__all__ = [
    'PopulationSpikes',
    'TorusPopulationSpikes',
    'TwistedTorusSpikes'
]


def sliding_firing_rate_tuple(spikes, n, tstart, tend, dt, win_len):
    '''
    Compute a firing rate with a sliding window from a tuple of spike data:
    spikes is a tuple(n_id, times), in which n_id is a list/array of neuron id
    and times is a list/array of spike times

    Parameters
    ----------
    spikes : np.ndarray
        A pair (n_id, spikes)
    n : int
        Total number of neurons
    tstart : float
        When the firing rate will start (ms)
    tend : float
        End time of firing rate (ms)
    dt : float
        Sliding window dt - not related to simulation time (ms)
    win_len : float
        Length of the sliding window (ms). Must be >= dt.

    Returns
    -------
    output : np.ndarray
        An array of shape (n, int((tend-tstart)/dt)+1
    '''
    rate = _spikes.sliding_firing_rate_base(spikes[0], spikes[1], n, tstart,
                                            tend, dt, win_len)
    rate_t = _spikes.sliding_times(tstart, tend, dt)
    return (rate, rate_t)


class PopulationSpikes(collections.Sequence):
    '''Abstraction of a population of spikes.'''
    def __init__(self, n, senders, times):
        '''
        Parameters
        ----------
        n : int
            Number of neurons in the population
        senders : 1D array
            Neuron numbers corresponding to the spikes
        times : 1D array
            Spike times. The shape of this array must be the same as for
            `senders`.
        '''
        self._N = n
        if n < 0:
            msg = "Number of neurons in the spike train must be " +\
                  "non-negative! Got {0}."
            raise ValueError(msg.format(n))

        # We are expecting senders and times as numpy arrays, if they are not,
        # convert them. Moreover, senders.dtype must be int, for indexing.
        self._senders = np.ascontiguousarray(senders, dtype=np.int)
        self._times = np.ascontiguousarray(times)
        self._unpacked = [None] * self._N  # unpacked version of spikes

    @property
    def n(self):
        '''
        Number of neurons in the population
        '''
        return self._N

    def avg_firing_rate(self, tstart, tend):
        '''
        Compute and average firing rate for all the neurons between 'tstart'
        and 'tend'. Return an array of firing rates, one item for each neuron
        in the population.

        Parameters
        ----------
        tstart : float (ms)
            Start time.
        tend   : float (ms)
            End time.

        Returns
        -------
        output : numpy array
            Firing rate in Hz for each neuron in the population.
        '''
        if tend < tstart:
            raise ValueError('tstart must be <= tend.')
        return _spikes.avg_fr(self._senders, self._times, self._N, tstart,
                              tend) * 1e3

    def sliding_firing_rate(self, tstart, tend, dt, win_len):
        '''
        Compute a sliding firing rate over the population of spikes, by taking
        a rectangular window of specified length.

        Parameters
        ----------
        tstart : float
            Start time of the firing rate analysis.
        tend : float
            End time of the analysis
        dt : float
            Firing rate window time step
        win_len : float
            Lengths of the windowing function (rectangle)

        Returns
        -------
        output : a tuple
            A pair (F, t), specifying the vector of firing rates and
            corresponding times. F is a 2D array of the shape (n, Ntimes), in
            which n is the number of neurons and Ntimes is the number of time
            steps. 't' is a vector of times corresponding to the time windows
            taken.
        '''
        spikes = (self._senders, self._times)
        return sliding_firing_rate_tuple(spikes, self._N, tstart, tend, dt,
                                         win_len)

    def windowed(self, tlimits):
        '''
        Return population spikes restricted to tlimits.

        Parameters
        ----------
        tlimits : a pair
            A tuple (tstart, tend). The spikes in the population must satisfy
            tstart >= t <= tend.

        Returns
        -------
        output : PopulationSpikes instance
            A copy of self with only a subset of spikes, limited by the time
            window.
        '''
        tstart = tlimits[0]
        tend = tlimits[1]
        tidx = np.logical_and(self._times >= tstart, self._times <= tend)
        return PopulationSpikes(self._N, self._senders[tidx],
                                self._times[tidx])

    def raster_data(self, neuron_list=None):
        '''
        Extract the senders and corresponding spike times for a raster plot.

        .. todo::

            implement neuron_list

        Parameters
        ----------
        neuron_list : list, optional
            Extract only neurons given in this list

        Returns
        -------
        output : a tuple
            A pair containing (senders, times).
        '''
        if neuron_list is not None:
            raise NotImplementedError()

        return self._senders, self._times

    def spike_train_difference(self, idx1, idx2=None, full=True,
                               reduce_fun=None):
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
        reduce_fun : callable, optional
            Any callable object that computes a function over an array of each
            spike train difference. The function must take one input argument,
            which will be the array of spike time differences for a pair of
            neurons. The output of this function will be stored instead of the
            default output.

        Returns
        -------
        output : A 2D or 1D array
            Spike train autocorrelation histograms for all the pairs of
            neurons.

        The computation takes the following steps:

         * If ``idx1`` or ``idx2`` are integers, they will be converted to a
           list of size 1.
         * If ``idx2`` is None, then the result will be a list of lists of
           pairs of cross-correlations between the neurons. Even if there is
           only one neuron. If ``full == True``, the output will be an upper
           triangular matrix of all the pairs, i.e. it will exclude the
           duplicated.  Otherwise there will be cross correlation histograms
           between all the pairs.
         * if ``idx2`` is not None, then ``idx1`` and ``idx2`` must be arrays
           of the same length, specifying the pairs to compute autocorrelation
           for
        '''
        if not full:
            raise NotImplementedError()

        if reduce_fun is None:
            reduce_fun = lambda x: x

        if not isinstance(idx1, collections.Iterable):
            idx1 = [idx1]

        if idx2 is None:
            idx2 = idx1
            res = [[] for x in idx1]
            for n1 in idx1:
                for n2 in idx2:
                    # print n1, n2, len(self[n1]), len(self[n2])
                    res[n1].append(
                        reduce_fun(
                            _spikes.spike_time_diff(self[n1], self[n2])))
            return res
        elif not isinstance(idx2, collections.Iterable):
            idx2 = [idx2]

        # Two arrays of pairs
        if len(idx1) != len(idx2):
            raise TypeError('Length of neuron indexes do not match!')

        res = [None] * len(idx1)
        for n in range(len(idx1)):
            res[n] = reduce_fun(_spikes.spike_time_diff(self[idx1[n]],
                                                        self[idx2[n]]))

        return res

    class CallableHistogram(object):
        '''Callable class that computes a histogram.'''
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
            '''Calculate bin edges.'''
            _, bin_edges = np.histogram([], **self.kw)
            return bin_edges

    def spike_train_xcorr(self, idx1, idx2, lag_range, bins=50, **kw):
        '''
        Compute the spike train crosscorrelation function for all pairs of
        spike trains in the population.

        For explanation of how ``idx1`` and ``idx2`` are treated, see
        :meth:`~PopulationSpikes.spike_train_difference`.

        Parameters
        ----------
        idx1 : int, or a sequence of ints
            Index of the first neuron or a list of neurons for which to compute
            the correlation histogram.
        idx2 : int, or a sequence of ints, or None
            Index of the second neuron or a list of indexes for the second set
            of spike trains.
        lag_range : (lag_start, lag_end)
            Limits of the cross-correlation function. The bins will always be
            **centered** on the values.
        bins : int, optional
            Number of bins
        kw : dict
            Keyword arguments passed on to the numpy.histogram function

        Returns
        -------
        output : a 2D or 1D list
            See :meth:`~PopulationSpikes.spike_train_difference`.
        '''
        lag_start, lag_end = lag_range
        bin_width = (lag_end - lag_start) / (bins - 1)
        bin_edges = np.linspace(lag_start - bin_width / 2.0, lag_end +
                                bin_width / 2.0, bins + 1)
        h = self.CallableHistogram(bins=bin_edges, **kw)
        xc = self.spike_train_difference(idx1, idx2, full=True, reduce_fun=h)
        bin_edges = h.get_bin_edges()
        bin_centers = (bin_edges[0:-1] + bin_edges[1:]) / 2.0
        return xc, bin_centers, bin_edges

    def isi_neuron(self, n):
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
        if len(spikes) < 2:
            return np.array([])
        return spikes[1:] - spikes[0:-1]

    def isi(self, n=None, reduce_fun=None):
        '''
        Return interspike interval of one or more neurons.

        Parameters
        ----------
        n : None, int, or sequence
            Neuron numbers. If ``n`` is None, then compute ISI stats for all
            neurons in the population. If ``n`` is an int, compute ISIs for
            just neuron indexed by ``n``. Otherwise ``n`` is expected to be a
            sequence of neuron indices.
        reduce_fun : callable or None
            A reduction function (callable object) that performs an operation
            on all the ISIs of the population. If ``None``, nothing is done.
            The callable has to take one input parameter, which is the sequence
            of ISIs. This allows to cascade data processing without the need
            for duplicating spike timing data.

        Returns
        -------
        output: list
            A list of outputs (depending on parameters) for each neuron, even
            if ``n`` is an int.
        '''
        if reduce_fun is None:
            reduce_fun = lambda x: x

        res = []
        if n is None:
            for n_id in range(len(self)):
                res.append(reduce_fun(self.isi_neuron(n_id)))
        elif isinstance(n, int):
            res.append(reduce_fun(self.isi_neuron(n)))
        else:
            for n_id in n:
                res.append(reduce_fun(self.isi_neuron(n_id)))

        return res

    def isi_cv(self, n=None, win_len=None):
        '''
        Coefficients of variation of inter-spike intervals of one or more
        neurons in the population. For the description of parameters and
        outputs and their semantics see also :meth:`~PopulationSpikes.ISI`.

        Parameters
        ----------
        win_len : float, list of floats, or ``None``
            Specify the maximal ISI value, i.e. use windowed coefficient of
            variation. If ``None``, use the whole range.
        '''
        cvfunc = scipy.stats.variation
        if win_len is None:
            f = scipy.stats.variation
        elif (isinstance(win_len, collections.Sequence) or
              isinstance(win_len, np.ndarray)):
            f = lambda x: np.asarray([cvfunc(x[x <= wl]) for wl in win_len])
        else:
            f = lambda x: cvfunc(x[x <= win_len])
        return self.isi(n, f)

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
    def __init__(self, senders, times, sheet_size):
        self._sheetSize = sheet_size
        n = sheet_size[0] * sheet_size[1]
        PopulationSpikes.__init__(self, n, senders, times)

    def get_x_size(self):
        '''Horizontal size of the torus.'''
        return self._sheetSize[0]

    def get_y_size(self):
        '''Vertical size of the torus.'''
        return self._sheetSize[1]

    def get_dimensions(self):
        '''Size of the torus.'''
        return self._sheetSize

    nx = property(fget=get_x_size, doc='Horizontal size of the torus')
    ny = property(fget=get_y_size, doc='Vertical size of the torus')
    dimensions = property(fget=get_dimensions,
                          doc='Dimensions of the torus (X, Y)')

    def avg_firing_rate(self, tstart, tend):
        rate = super(TorusPopulationSpikes, self).avg_firing_rate(tstart, tend)
        return np.reshape(rate, (self.ny, self.nx))

    def population_vector(self, tstart, tend, dt, win_len):
        '''
        Compute the population vector on a torus, from the spikes present. Note
        that this method will have a limited functionality on a twisted torus,
        but can be used if the population activity translates in the X
        dimension only.

        Parameters
        ----------
        tstart : float
            Start time of analysis
        tend : float
            End time of analysis
        dt : float
            Time step of the (rectangular) windowing function
        win_len : float
            Length of the windowing function

        Returns
        -------
        output : tuple
            A pair (r, t) in which r is a 2D vector of shape
            (int((tend-tstart)/dt)+1), 2), corresponding to the population
            vector for each time step of the windowing function, and t is a
            vector of times, of length the first dimension of r.
        '''
        sheet_size_x = self.get_x_size()
        sheet_size_y = self.get_y_size()
        # n = sheet_size_x * sheet_size_y

        F, tsteps = PopulationSpikes.sliding_firing_rate(self, tstart, tend,
                                                         dt, win_len)
        p = np.ndarray((len(tsteps), 2), dtype=complex)
        x, y = np.meshgrid(np.arange(sheet_size_x), np.arange(sheet_size_y))
        x = np.exp(1j *
                   (x - sheet_size_x / 2) / sheet_size_x * 2 * np.pi).ravel()
        y = np.exp(1j *
                   (y - sheet_size_y / 2) / sheet_size_y * 2 * np.pi).ravel()
        for t_it in range(len(tsteps)):
            p[t_it, 0] = np.dot(F[:, t_it], x)
            p[t_it, 1] = np.dot(F[:, t_it], y)

        return (np.angle(p) / 2 / np.pi * self._sheetSize, tsteps)

    def sliding_firing_rate(self, tstart, tend, dt, win_len):
        '''
        Compute a sliding firing rate over the population of spikes, by taking
        a rectangular window of specified length. However, unlike the ancestor
        method (PopulationSpikes.sliding_firing_rate), return a 3D array, a
        succession of 2D population firing rates in time.

        Parameters
        ----------
        tstart : float
            Start time of the firing rate analysis.
        tend : float
            End time of the analysis
        dt : float
            Firing rate window time step
        win_len : float
            Lengths of the windowing function (rectangle)

        Returns
        -------
        output : a tuple
            A pair (F, t), specifying the vector of firing rates and
            corresponding times. F is a 3D array of the shape (nx, ny, Ntimes),
            in which nx/ny are the number of neurons in X and Y dimensions,
            respectively, and Ntimes is the number of time steps. 't' is a
            vector of times corresponding to the time windows taken.
        '''
        spikes = (self._senders, self._times)
        F, Ft = sliding_firing_rate_tuple(spikes, self._N, tstart, tend, dt,
                                          win_len)
        nx = self.get_x_size()
        ny = self.get_y_size()
        return np.reshape(F, (ny, nx, len(Ft))), Ft


class TwistedTorusSpikes(TorusPopulationSpikes):
    '''
    Spikes arranged on twisted torus. The torus is twisted in the X direction.
    '''
    def __init__(self, senders, times, sheet_size):
        super(TwistedTorusSpikes, self).__init__(senders, times, sheet_size)

    def population_vector(self, tstart, tend, dt, win_len):
        msg = ('population_vector() has not been implemented yet for {}. ' +
               'Note that this method is different for the regular torus ' +
               '(TorusPopulationSpikes).')
        raise NotImplementedError(msg.format(self.__class__.__name__))
