'''
===============================================================
:mod:`gridcells.analysis.signal` - signal analysis
===============================================================

The can be e.g. filtering, slicing, correlation analysis, up/down-sampling, etc.

.. autosummary::

    acorr
    corr
    ExtremumTypes
    local_extrema
    local_maxima
    local_minima
    LocalExtrema
'''
from __future__ import absolute_import, print_function, division

import os
from enum import IntEnum

import numpy as np

__all__ = [
    'acorr',
    'corr',
    'local_extrema',
    'local_minima',
    'local_maxima',
    'ExtremumTypes',
    'LocalExtrema',
]


# Do not import when in RDT environment
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not on_rtd:
    from . import _signal

class ExtremumTypes(IntEnum):
    '''Specifies types of extrema.'''
    UNDEFINED = 0
    '''Undefined.'''
    MIN = 1
    '''Local minimum.'''
    MAX = 2
    '''Local maximum.'''


def corr(a, b, mode='onesided', lag_start=None, lag_end=None):
    '''
    An enhanced correlation function of two real signals, based on a custom C++
    code.

    This function uses dot product instead of FFT to compute a correlation
    function with range restricted lags.

    Thus, for a long-range of lags and big arrays it can be slower than the
    numpy.correlate (which uses fft-based convolution). However, for arrays in
    which the number of lags << max(a.size, b.size) the computation time might
    be much shorter than using convolution to calculate the full correlation
    function and taking a slice of it.

    Parameters:

    a, b : ndarray
        One dimensional numpy arrays (in the current implementation, they will
        be converted to dtype=double if not already of that type.

    mode : str, optional
        A string indicating the size of the output:

          ``onesided`` : range of lags is [0, b.size - 1]

          ``twosided`` : range of lags is [-(a.size - 1), b.size - 1]

          ``range``    : range of lags is [-lag_start, lag_end]

    lag_start, lag_end : int, optional
        Initial and final lag value. Only used when mode == 'range'

    output : numpy.ndarray with shape (1, ) and dtype.float
        A 1D array of size depending on mode

    .. note::

        This function always returns a numpy array with dtype=float.

    .. seealso:: :py:func:`acorr`
    '''
    a = np.require(a, np.float, 'F')
    b = np.require(b, np.float, 'F')
    sz1 = a.size
    sz2 = b.size
    if sz1 == 0 or sz2 == 0:
        raise TypeError("Both input arrays must have non-zero size!")

    if mode == 'onesided':
        return _signal.correlation_function(a, b, 0, sz2 - 1)
    elif mode == 'twosided':
        return _signal.correlation_function(a, b, -(sz1 - 1), sz2 - 1)
    elif mode == 'range':
        lag_start = int(lag_start)
        lag_end   = int(lag_end)
        if lag_start <= -sz1 or lag_end >= sz2:
            raise ValueError("Lag range must be in the range "
                             "[{0}, {1}]".format(-(sz1 - 1), sz2 - 1))
        return _signal.correlation_function(a, b, lag_start, lag_end)
    else:
        raise ValueError("mode must be one of 'onesided', 'twosided', or "
                         "'range'")


def acorr(sig, max_lag=None, norm=False, mode='onesided'):
    '''
    Compute an autocorrelation function of a real signal.

    Parameters
    ----------
    sig : numpy.ndarray
        The signal, 1D vector, to compute an autocorrelation of.

    max_lag : int, optional
        Maximal number of lags. If mode == 'onesided', the range of lags will be
        [0, max_lag], i.e. the size of the output will be (max_lag+1). If
        mode == 'twosided', the lags will be in the range [-max_lag, max_lag],
        and so the size of the output will be 2*max_lag + 1.

        If max_lag is None, then max_lag will be set to len(sig)-1

    norm : bool, optional
        Whether to normalize the auto correlation result, so that res(0) = 1

    mode : string, optional
        ``onesided`` or ``twosided``. See description of max_lag

    output : numpy.ndarray
        A 1D array, size depends on ``max_lag`` and ``mode`` parameters.

    Notes
    -----
    If the normalisation constant is zero (i.e. the input array is zero), this
    function will return a zero array.
    '''
    if max_lag is None:
        max_lag = len(sig) - 1
    if mode == 'onesided':
        c = corr(sig, sig, mode='range', lag_start=0, lag_end=max_lag)
    elif mode == 'twosided':
        c = corr(sig, sig, mode='range', lag_start=-max_lag, lag_end=max_lag)
    else:
        raise ValueError("mode can be either 'onesided' or 'twosided'!")

    if norm:
        maximum = np.max(np.abs(c))
        if maximum != 0.:
            c /= maximum

    return c


def local_extrema(sig):
    '''Find all local extrema using the derivative approach.

    Parameters
    ----------
    sig : numpy.ndarray
        A 1D numpy array

    Returns
    -------
    extrema : :class:`~LocalExtrema`
        An object containing the local extrema of the signal ``sig``.

    See also
    --------
    local_minima : Finds local minima.
    local_maxima : Finds local maxima.

    Notes
    -----
    This method is not suitable to find local extrema of functions where the
    extremum is flat, i.e. as in square pulses.
    '''
    sz = len(sig)
    szDiff = sz - 1

    der = np.diff(sig)
    der0 = (der[0:szDiff - 1] * der[1:szDiff]) < 0.
    ext_idx = np.nonzero(der0)[0]

    dder = np.diff(der)[ext_idx]
    ext_idx += 1    # Correction for a peak position
    ext_t = np.ndarray((dder.size, ), dtype=int)
    ext_t[dder < 0] = ExtremumTypes.MAX
    ext_t[dder > 0] = ExtremumTypes.MIN

    return LocalExtrema(ext_idx, ext_t)


def local_maxima(sig):
    '''Find all local maxima using the derivative approach

    Parameters
    ----------
    sig : numpy.ndarray
        A 1D numpy array

    Returns
    -------
    maxima : np.ndarray
        An array of indices into ``sig`` of the local maxima.

    See also
    --------
    local_extrema : Finds local extrema.
    local_minima: Finds local minima.
    '''
    return local_extrema(sig).get_type(ExtremumTypes.MAX)


def local_minima(sig):
    '''Find all local minima using the derivative approach

    Parameters
    ----------
    sig : numpy.ndarray
        A 1D numpy array

    Returns
    -------
    maxima : np.ndarray
        An array of indices into ``sig`` of the local minima.

    See also
    --------
    local_extrema : Finds local extrema.
    local_maxima: Finds local maxima.
    '''
    return local_extrema(sig).get_type(ExtremumTypes.MIN)


class LocalExtrema(object):
    '''A class representing local extrema for a particular object.

    This is only a helper class. Users should not instantiate this class
    directly

    .. note::
        For now only 1D extrema are supported.

    Parameters
    ----------
    extrema : 1D numpy array
        Positions of the extrema in a signal. The user must track the source.
    types : 1D numpy array
        Types of the extrema held in this object. Contains values from
        :class:`~ExtremumTypes`.
    '''
    def __init__(self, extrema, types):
        self._extrema = np.asanyarray(extrema)
        self._types = np.asanyarray(types)
        if len(extrema) != len(types):
            raise IndexError('Number of extrema and number of their types '
                             'must match.')

    def __len__(self):
        return len(self._types)

    def get_type(self, extremum_type):
        '''Get all the local extrema that are of type ``extremum_type``.

        Parameters
        ----------
        extremum_type : :class:`~ExtremumTypes`
            The type of the extremum to retrieve.

        Returns
        -------
        extrema : iterable
            An iterable that will contain the extrema with the specified type.
            If no extremum of the requested type is present, returns an empty
            array.
        '''
        return self._extrema[self._types == extremum_type]
