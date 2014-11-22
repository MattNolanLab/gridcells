'''
===============================================
:mod:`gridcells.analysis.bumps` - bump tracking
===============================================

Classes and functions for processing data related to bump attractors.

Classes
-------
.. inheritance-diagram:: gridcells.analysis.bumps
    :parts: 2


.. autosummary::

    MLFit
    MLFitList
    MLGaussianFit
    MLGaussianFitList
    SingleBumpPopulation
    SymmetricGaussianParams

Functions
---------

.. autosummary::

    fit_gaussian_tt
    fit_gaussian_bump_tt
    fit_maximum_lh
'''
from __future__ import absolute_import, division, print_function

import collections
import logging

import numpy as np
import scipy.optimize

from . import spikes
from ..core.common import Pair2D, twisted_torus_distance

LOGGER = logging.getLogger(__name__)


class SymmetricGaussianParams(object):
    '''Parameters for the symmetric Gaussian function.'''

    def __init__(self, amplitude, mu_x, mu_y, sigma, err2):
        self.A = amplitude
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.sigma = sigma
        self.err2 = err2


class MLFit(object):
    '''Maximum likelihood fit data holer.'''

    def __init__(self, mu, sigma2, ln_lh, err2):
        self.mu = mu
        self.sigma2 = sigma2
        self.ln_lh = ln_lh
        self.err2 = err2


class MLFitList(MLFit, collections.Sequence):
    '''A container for holding results of maximum likelihood fitting.

    Can be accessed as a Sequence object.
    '''

    def __init__(self, mu=None, sigma2=None, ln_lh=None, err2=None,
                 times=None):
        if mu is None:
            mu = []
        if sigma2 is None:
            sigma2 = []
        if ln_lh is None:
            ln_lh = []
        if err2 is None:
            err2 = []
        if times is None:
            times = []
        super(MLFitList, self).__init__(mu, sigma2, ln_lh, err2)
        self.times = times

        if not self._consistent():
            raise ValueError('All input arguments mus have same length')

    def _consistent(self):
        '''Check if the data is consistent.'''
        return len(self.mu) == len(self.sigma2) and \
            len(self.mu) == len(self.ln_lh) and   \
            len(self.mu) == len(self.err2) and    \
            len(self.mu) == len(self.times)

    def __getitem__(self, key):
        return (MLFit(self.mu[key], self.sigma2[key], self.ln_lh[key],
                      self.err2),
                self.times)

    def __len__(self):
        return len(self.mu)

    def append_data(self, d, t):
        '''`d` must be an instance of :class:`MLFit`'''
        if not isinstance(d, MLFit):
            raise TypeError('ML data must be an instance of MLFit')
        self.mu.append(d.mu)
        self.sigma2.append(d.sigma2)
        self.ln_lh.append(d.ln_lh)
        self.err2.append(d.err2)
        self.times.append(t)


class MLGaussianFit(SymmetricGaussianParams):
    '''Gaussian fit performed by applying maximum likelihood estimator.'''

    def __init__(self, amplitude, mu_x, mu_y, sigma, err2, ln_lh,
                 lh_precision):
        super(MLGaussianFit, self).__init__(amplitude, mu_x, mu_y, sigma, err2)
        self.ln_lh = ln_lh
        self.lh_precision = lh_precision


class MLGaussianFitList(MLGaussianFit, collections.Sequence):
    '''A container for holding maximum likelihood Gaussian fits.

    Can be accessed as a Sequence.
    '''

    def __init__(self, amplitude=None, mu_x=None, mu_y=None, sigma=None,
                 err2=None, ln_lh=None, lh_precision=None, times=None):
        if amplitude is None:
            amplitude = []
        if mu_x is None:
            mu_x = []
        if mu_y is None:
            mu_y = []
        if sigma is None:
            sigma = []
        if err2 is None:
            err2 = []
        if ln_lh is None:
            ln_lh = []
        if lh_precision is None:
            lh_precision = []
        if times is None:
            times = []

        super(MLGaussianFitList, self).__init__(amplitude, mu_x, mu_y, sigma,
                                                err2, ln_lh,
                                                lh_precision)
        self.times = times

        if not self._consistent():
            raise ValueError('All input arguments mus have same length')

    def _consistent(self):
        '''Check if the data is consistent.'''
        return \
            len(self.A) == len(self.mu_x) and         \
            len(self.A) == len(self.mu_y) and         \
            len(self.A) == len(self.sigma) and        \
            len(self.A) == len(self.err2) and          \
            len(self.A) == len(self.ln_lh) and         \
            len(self.A) == len(self.lh_precision) and \
            len(self.A) == len(self.times)

    def append_data(self, d, t):
        '''`d` must be an instance of :class:`MLGaussianFit`'''
        if not isinstance(d, MLGaussianFit):
            raise TypeError('Data must be an instance of MLGaussianFit')

        self.A.append(d.A)
        self.mu_x.append(d.mu_x)
        self.mu_y.append(d.mu_y)
        self.sigma.append(d.sigma)
        self.err2.append(d.err2)
        self.ln_lh.append(d.ln_lh)
        self.lh_precision.append(d.lh_precision)
        self.times.append(t)

    def __getitem__(self, key):
        return MLGaussianFit(self.A[key],
                             self.mu_x[key],
                             self.mu_y[key],
                             self.sigma[key],
                             self.err2[key],
                             self.ln_lh,
                             self.lh_precision), \
            self.times[key]

    def __len__(self):
        return len(self.A)  # All same length


def fit_gaussian_tt(sig_f, i):
    r'''Fit a 2D circular Gaussian function to a 2D signal using a maximum
    likelihood estimator.

    The Gaussian is not generic: :math:`\sigma_x = \sigma_y = \sigma`, i.e.
    it is circular only.

    The function fitted looks like this:

    .. math::
        f(\mathbf{X}) = |A| \exp\left\{\frac{-|\mathbf{X} -
                        \mathbf{\mu}|^2}{2\sigma^2}\right\}

    where :math:`|\cdot|` is a distance metric on the twisted torus.

    Parameters
    ----------
    sig_f : np.ndarray
        A 2D array that specified the signal to fit the Gaussian onto. The
        dimensions of the torus will be inferred from the shape of `sig_f`:
        (dim.y, dim.x) = `sig_f.shape`.
    i : SymmetricGaussianParams
        Guassian initialisation parameters. The `err2` field will be ignored.

    Returns
    -------
    :class:`MLGaussianFit`
        Estimated values, together with maximum likelihood value and precision
        (inverse variance of noise: *NOT* of the fitted Gaussian).
    '''
    # Fit the Gaussian using least squares
    dim = Pair2D(sig_f.shape[1], sig_f.shape[0])
    X, Y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(dim.x, dtype=np.double),
        np.arange(dim.y, dtype=np.double))
    others = Pair2D(X.flatten(), Y.flatten())
    a = Pair2D(None, None)

    def gaussian_diff(x):
        '''Compute error.'''
        a.x = x[1]  # mu_x
        a.y = x[2]  # mu_y
        dist = twisted_torus_distance(a, others, dim)
        #                A                              sigma
        #                |                                |
        return (np.abs(x[0]) * np.exp(-dist ** 2 / 2. / x[3] ** 2) -
                sig_f.ravel())

    xest, _ = scipy.optimize.leastsq(gaussian_diff,
                                     np.array([i.A, i.mu_x, i.mu_y, i.sigma]))
    err2 = gaussian_diff(xest) ** 2

    # Remap the values modulo torus size
    xest[1] = xest[1] % dim.x
    xest[2] = xest[2] % dim.y

    # Compute the log-likelihood
    n = dim.x * dim.y
    aic_correction = 5  # Number of optimized parameters
    beta = 1.0 / (np.mean(err2))
    ln_lh = -beta / 2. * np.sum(err2) +  \
        n / 2. * np.log(beta) -     \
        n / 2. * np.log(2 * np.pi) -  \
        aic_correction

    return MLGaussianFit(xest[0], xest[1], xest[2], xest[3], err2, ln_lh, beta)


def fit_gaussian_bump_tt(sig):
    '''Fit a 2D Gaussian onto a (potential) firing rate bump on the twisted
    torus.

    Parameters
    ----------
    sig : np.ndarray
        2D firing rate map to fit. Axis 0 is the Y position. This will be
        passed directly to :func:`~analysis.image.fit_gaussian_tt`.

    Returns
    -------
    :class:`analysis.image.MLGaussianFit`
        Estimated values of the fit.

    Notes
    -----
    The function initialises the Gaussian fitting parameters to a position at
    the maximum of `sig`.
    '''
    mu0_y, mu0_x = np.unravel_index(np.argmax(sig), sig.shape)
    a0 = sig[mu0_y, mu0_x]
    sigma0 = np.max(sig.shape) / 4.
    init = SymmetricGaussianParams(a0, mu0_x, mu0_y, sigma0, None)
    return fit_gaussian_tt(sig, init)


def fit_maximum_lh(sig):
    '''Fit a maximum likelihood solution under Gaussian noise.

    Parameters
    ----------
    sig : np.ndarray
        A vector containing the samples

    Returns
    fit : MLFit
        Maximum likelihood parameters
    '''
    sig = sig.flatten()
    mu = np.mean(sig)
    sigma2 = np.var(sig)
    err2 = (sig - mu) ** 2

    if sigma2 == 0:
        ln_lh = np.inf
    else:
        n = len(sig)
        aic_correction = 2
        ln_lh = -.5 / sigma2 * np.sum((sig - mu) ** 2) - \
            .5 * n * np.log(sigma2) -             \
            .5 * n * np.log(2 * np.pi) -            \
            aic_correction

    return MLFit(mu, sigma2, ln_lh, err2)


class SingleBumpPopulation(spikes.TwistedTorusSpikes):
    '''
    A population of neurons that is supposed to form a bump on a twisted torus.

    Parameters
    ----------
    senders : array_like
        A an array of neurons' IDs.
    times : array_like
        An array of spike times. Length must be the same as as <senders>.
    sheet_size : A pair
        A pair of X and Y dimensions of the torus.
    '''

    def __init__(self, senders, times, sheet_size):
        super(SingleBumpPopulation, self).__init__(senders, times, sheet_size)

    def _perform_fit(self, tstart, tend, dt, win_len, fit_callable, list_cls,
                     full_err=True):
        '''Perform the fit given the requested ``fit_callable``.'''
        F, Ft = self.sliding_firing_rate(tstart, tend, dt, win_len)
        res = list_cls()
        for tIdx in range(len(Ft)):
            LOGGER.debug('%s:: fitting: %d/%d, %.3f/%.3f ',
                         fit_callable.__name__, tIdx + 1, len(Ft), Ft[tIdx],
                         Ft[-1])
            fit_params = fit_callable(F[:, :, tIdx])

            if not full_err:
                fit_params.err2 = np.sum(fit_params.err2)

            res.append_data(fit_params, Ft[tIdx])
        return res

    def bump_position(self, tstart, tend, dt, win_len, full_err=True):
        '''Estimate bump positions during the simulation time:

        1. Estimates population firing rate for each bin.

        2. Apply the bump position estimation procedure to each of the
           population activity items.

        Parameters
        ----------
        tstart, tend, dt, win_len : float
            Start and end time, time step, and window length. See also
            :meth:`~gridcells.analysis.spikes.PopulationSpikes.sliding_firing_rate`.
        full_err : bool
            If ``True``, save the full error of fit. Otherwise a sum only.

        Returns
        -------
        pos:list :class:`MLGaussianFitList`
            A list of fitted Gaussian parameters

        Notes
        -----
        This method uses the Maximum likelihood estimator to fit the Gaussian
        function (:meth:`~fit_gaussian_bump_tt`)
        '''
        return self._perform_fit(tstart, tend, dt, win_len,
                                 fit_gaussian_bump_tt, MLGaussianFitList,
                                 full_err=full_err)

    def uniform_fit(self, tstart, tend, dt, win_len, full_err=True):
        '''Estimate the mean firing rate using maximum likelihood estimator
        (:func:`~gridcells.analysis.image.fit_maximum_lh`)

            1. Uses :meth:`sliding_firing_rate`.

            2. Apply the estimator.

        Parameters
        ----------
        tstart, tend, dt, win_len
            As in :py:meth:`~analysis.spikes.sliding_firing_rate`.
        full_err : bool
            If ``True``, save the full error of fit. Otherwise a sum only.

        Returns
        -------
        MLFitList
            A list of fitted parameters.
        '''
        return self._perform_fit(tstart, tend, dt, win_len, fit_maximum_lh,
                                 MLFitList, full_err=full_err)
