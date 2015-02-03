#
#   signal.py
#
#   Signal analysis tools: ffts, cwt, etc. specific to GridCells
#
#       Copyright (C) 2012  Lukas Solanka <l.solanka@sms.ed.ac.uk>
#
#       This program is free software: you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation, either version 3 of the License, or
#       (at your option) any later version.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
'''
The :mod:`analysis.signal` module contains functions and classes for continuous signal
analysis

The can be e.g. filtering, slicing, correlation analysis, up/down-sampling, etc.

.. currentmodule:: analysis.signal
.. autosummary::
    autoCorrelation
    corr
    localExtrema

----------------------


'''

import numpy as np
import scipy.signal

from numpy.fft.fftpack import fft
from Wavelets import Morlet

import _signal


#__all__ = ['butterHighPass', 'butterBandPass', 'spikePhaseTrialRaster',
#        'splitSigToThetaCycles', 'getChargeTheta', 'phaseCWT', 'CWT',
#        'fft_real_freq', 'relativePower']


def butterHighPass(sig, dt, f_pass):
    nyq_f = 1./dt/2
    norm_f_pass = f_pass/nyq_f

    # Low pass filter
    b, a = scipy.signal.butter(3, norm_f_pass, btype='high')
    return scipy.signal.filtfilt(b, a, sig)


def butterBandPass(sig, dt, f_start, f_stop):
    '''Band pass filter a signal, with f_start and f_stop frequencies'''
    nyq_f = 1./dt/2
    norm_f_start = f_start/ nyq_f
    norm_f_stop  = f_stop / nyq_f
    b, a = scipy.signal.butter(3, [norm_f_start, norm_f_stop], btype='band')
    return scipy.signal.filtfilt(b, a, sig)


def spikePhaseTrialRaster(spikeTimes, f, start_t=0):
    '''Here assuming that phase(t=0) = 0'''
    spikeTimes -= start_t
    trials = np.floor(f*spikeTimes)
    phases = np.mod(2*np.pi*f*spikeTimes, 2*np.pi)
    times  = np.mod(spikeTimes, 1./f)
    return (phases, times, trials)




## Take a 1D signal and rescale it to signals of individual theta cycles.
#
# Each row of the result contains one theta cycle and it is assumed that theta
# is generated continuously.
#
# The last, unaligned part of the signal will be discarded.
#
# Phase(sig, t=0) must be 0, no phase shifts!
#
# @param sig   Signal with dt.
# @param thetaT Theta period. MUST be a multiple of dt
# @param dt     Time resolution of the signal.
def splitSigToThetaCycles(sig, thetaT, dt):
    n_ph = thetaT / dt
    q_ph = len(sig) // n_ph
    return np.reshape(sig[0:q_ph*n_ph], (q_ph, n_ph))



def getChargeTheta(sig_theta_sliced, dt):
    '''
    For each theta cycle, find the total charge of synaptic current.
    Each row of sig_theta_sliced is one theta cycle
    '''
    return np.trapz(sig_theta_sliced, dx=dt, axis=1)



def phaseCWT(sig, Tph, dt, maxF, dF=2):
    '''
    Calculate Morlet wavelet transform of a signal, but as a function of
    phase. Unaligned phase at the end will be discarded, and ph(t=0) must be 0,
    i.e. no phase shifts!
    '''
    n_ph = Tph/dt
    N = len(sig)
    q_ph = np.floor(N/n_ph)

    minF = 1./(len(sig)/2 * Morlet.fourierwl * dt)
    F = np.linspace(minF, maxF, (maxF-minF)/dF+1)
    scales = 1/F * 1/Morlet.fourierwl * 1/dt

    w = Morlet(sig, scales, scaling='direct')
    w_cwt_ph = np.ndarray((w.nscale, n_ph))

    for sc_it in xrange(w.nscale):
        w_ph = np.reshape(np.abs(w.cwt[sc_it, :][0:q_ph*n_ph])**2, (q_ph, n_ph))
        w_cwt_ph[sc_it, :] = np.mean(w_ph, 0)

    sig_ph = np.reshape(sig[0:q_ph*n_ph], (q_ph, n_ph))
    phases = 1. * np.arange(n_ph) / n_ph * 2*np.pi - np.pi
    return phases, w_cwt_ph, 1./(w.scales*w.fourierwl*dt), sig_ph



def CWT(sig, dt, maxF, dF=2):
    '''
    Calculate a Morlet wavelet transfrom of a signal.
    '''
    N = len(sig)

    minF = 1./(len(sig)/2 * Morlet.fourierwl * dt)
    F = np.linspace(minF, maxF, (maxF-minF)/dF+1)
    scales = 1/F * 1/Morlet.fourierwl * 1/dt

    w = Morlet(sig, scales, scaling='direct')
    return np.abs(w.cwt)**2, 1./(w.scales*w.fourierwl*dt)



## Compute a DFT of a real signal and return an array of frequencies and
# Fourier coefficients.
#
# @param sig    The signal
# @param dt     Sampling rate of the signal
# @return A tuple (F, sig~). F is an array of frequencies, sig~ is the fourier
#         transform
#
def fft_real_freq(sig, dt):
    S = fft(sig)
    S_F = np.linspace(0, 1, len(S)/2) / dt / 2.0

    return S_F, S[0:len(S_F)]





## Compute power from FFT data in a specified frequency range, relative to the
# total power.
#
# This function will throw an error if the desired frequency range is out of the
# range of the actual signal.
#
# @param Pxx     A Power spectral density vector
# @param F       Frequencies corresponding to Pxx (Hz).
# @param Frange  A tuple containing the frequency range (Hz).
# @return Relative power in the specified freqency range.
#
def relativePower(Pxx, F, Frange):
    Fidx = np.nonzero(np.logical_and(F >= Frange[0], F <= Frange[1]))[0]
    rangeP = sum(Pxx[Fidx])
    return rangeP / sum(Pxx)




# Get frequency with maximum power
#
# @param Pxx     Power spectral density of the signal.
# @param F       A corresponding array of frequencies
# @param Frange  A tuple containing frequency range to restrict the analysis to.
#
# @return An index to F, the frequency with maximum power.
#
def maxPowerFrequency(Pxx, F, Frange=None):
    if (Frange == None):
        return F[np.argmax(Pxx)]
    else:
        range = np.logical_and(F >= Frange[0], F <= Frange[1])
        return F[range][np.argmax(Pxx[range])]



def corr(a, b, mode='onesided', lag_start=None, lag_end=None):
    '''
    An enhanced correlation function of two real signals, based on blitz++.

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

    .. seealso:: :py:func:`autoCorrelation`
    '''
    sz1 = a.size
    sz2 = b.size
    if (sz1 == 0 or sz2 == 0):
        raise TypeError("Both input arrays must have non-zero size!")

    if (mode == 'onesided'):
        return _signal.correlation_function(a, b, 0, sz2 - 1)
    elif (mode == 'twosided'):
        return _signal.correlation_function(a, b, -(sz1 - 1), sz2 - 1)
    elif (mode == 'range'):
        lag_start = int(lag_start)
        lag_end   = int(lag_end)
        if (lag_start <= -sz1 or lag_end >= sz2):
            raise ValueError("Lag range must be in the range [{0}, {1}]".format(-(sz1 - 1), sz2 - 1))
        return _signal.correlation_function(a, b, lag_start, lag_end)
    else:
        raise ValueError("mode must be one of 'onesided', 'twosided', or 'range'")



def autoCorrelation(sig, max_lag=None, norm=False, mode='onesided'):
    '''
    Compute an autocorrelation function of a real signal.

    Parameters
    ----------
    sig : numpy.ndarray
        The signal, 1D vector, to compute an autocorrelation of.

    max_lag : int, optional
        Maximal number of lags. If mode == 'onesided', the range of lags will
        be [0, max_lag], i.e. the size of the output will be (max_lag+1). If mode ==
        'twosided', the lags will be in the range [-max_lag, max_lag], and so the size
        of the output will be 2*max_lag + 1.

        If max_lag is None, then max_lag will be set to len(sig)-1

    norm : bool, optional
        Whether to normalize the auto correlation result, so that res(0) = 1

    mode : string, optional
        ``onesided`` or ``twosided``. See description of max_lag

    output : numpy.ndarray
        A 1D array, size depends on ``max_lag`` and ``mode`` parameters.
    '''
    if (mode == 'onesided'):
        c = corr(sig, sig, mode='range', lag_start=0, lag_end=max_lag)
    elif (mode == 'twosided'):
        c = corr(sig, sig, mode='range', lag_start=-max_lag, lag_end=max_lag)
    else:
        raise ValueError("mode can be either 'onesided' or 'twosided'!")

    if (norm):
        c /= max(c)

    return c




###############################################################################
#                           Extrema analysis
###############################################################################
def localExtrema(sig):
    '''
    Find all local extrema using the derivative approach.

    Parameters
    ----------
    sig : numpy.ndarray
        A 1D numpy array

    output : (numpy.ndarray, numpy.ndarray)
        A pair (idx, types) containing the positions of local extrema iniside
        ``sig`` and the type of the extrema:
            * type > 0 means local maximum
            * type < 0 is local minimum
    '''
    sz = len(sig)
    szDiff = sz - 1
    der = np.diff(sig)
    der0 = (der[0:szDiff - 1] * der[1:szDiff]) < 0.
    ext_idx = np.nonzero(der0)[0]
    dder = np.diff(der)[ext_idx]
    ext_idx += 1    # Correction for a peak position
    ext_t = np.ndarray((dder.size, ), dtype=int)
    ext_t[dder < 0] = 1
    ext_t[dder > 0] = -1
    return (ext_idx, ext_t)



def globalExtremum(sig, func):
    '''
    Return global maximum of a signal.

    ``sig``
        A 1D array, in which case a global extremum will be returned as a single
        value, or a 2D array, for which an array of extrema will be returned,
        one extremum for **each row**. Numpy arrays accepted only.

    ``func``
        Numpy extremum function. Must accept the 'axis' parameter
    '''
    shpLen = len(sig.shape)
    if (shpLen == 1):
        return func(sig)
    elif (shpLen == 2):
        return func(sig, axis=1)
    else:
        raise TypeError("signal must be either 1D or a 2D numpy array!")


def relativePeakHeight(localExtrema, cmpFun):
    sz = len(localExtrema)
    if (sz == 0):
        raise TypeError("Cannot compute relative peak heights in an empty array.")
    elif (sz == 1):
        raise TypeError("Cannot compute relative peak heights in an array with only one element")

    res = np.ndarray((sz, ))
    cmp = np.ndarray((2, sz-2))

    hr = np.abs(localExtrema[0:sz-1] - localExtrema[1:])
    hl = np.abs(localExtrema[1:]     - localExtrema[0:sz-1])


    cmp[0, :] = hr[1:]
    cmp[1, :] = hl[0:hl.size - 1]

    res[0]  = hr[0]
    res[-1] = hl[-1]
    res[1:sz-1] = cmpFun(cmp, axis=0)


    return res



###############################################################################
#                         (Re-)Sampling signals
###############################################################################

def downSample(sig, factor, X=None):
    '''
    Downsample a signal by a specified factor.

    Parameters
    ----------
    sig : numpy.ndarray
        Signal to down-sample.

    factor : int
        Down-sampling factor. If the factor is not int, it will be converted to
        int.

    X : numpy.ndarray, optional
        Optional X value of the signal (i.e. time, space) that will be
        downsampled as well. If ``None``, [0, len(sig)-1] will be used.

    output : numpy.ndarray
        A tuple (sig_d, X_d), where
        sig_d is the downsampled signal and X_d is the down-sampled X
        coordinate of the signal. The size of the output will depend on whether
        the size of ``sig`` is an integer multiple of ``factor``.
    '''
    if (X is None):
        X = np.arange(len(sig))
    idx = np.arange(0, len(sig), 10)
    return (sig[idx], X[idx])


###############################################################################
#                                     Other
###############################################################################
def sliceSignal(t, sig, tStart, tEnd):
    idx = np.logical_and(t >= tStart, t <= tEnd)
    return t[idx], sig[idx], idx

