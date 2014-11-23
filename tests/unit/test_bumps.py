'''Test bump fitting procedures.'''
from __future__ import absolute_import, print_function, division

import numpy as np

from gridcells.core import Pair2D, twisted_torus_distance
from gridcells.analysis.bumps import (fit_gaussian_bump_tt, fit_maximum_lh,
                                      SingleBumpPopulation, MLGaussianFitList,
                                      MLFitList)

from base import notimpl


def gen_gaussian_tt(ampl, mu_x, mu_y, sigma, x, y):
    '''Generate a Gaussian function on a twisted torus.'''
    dim = Pair2D(x.shape[1], x.shape[0])
    first = Pair2D(mu_x, mu_y)
    second = Pair2D(x.ravel(), y.ravel())
    dist = twisted_torus_distance(first, second, dim)
    g = ampl * np.exp(-dist ** 2 / (2 * sigma ** 2))
    return np.reshape(g, (dim.y, dim.x))


class TestFittingTT(object):

    '''Test all implemented functions/classes that require fitting a Gaussian
    on the twisted torus.

    Notes
    -----
    `decimalAlmostEqual` and `noiseDeltaFrac` have been manually fine-tuned by
    inspecting the data. In general, it turns that the fitting procedure will
    not be better than having 10% error, i.e. one digit.

    .. todo::

        `decimalAlmostEqual` should be relative to the size of the torus. If
        the torus is too small this setting will effectively force all the
        tests to pass even if the values will differ.

        Also, log-likelihood tests would be useful
    '''
    decimalAlmostEqual = 1
    noiseDeltaFrac = 2e-1

    gaussianAMax = 40
    nIter = 1000
    minSigma = 1.
    maxFailures = .02
    maxNoiseSigma = 1e-3

    def test_bump_fitting(self):
        '''Test fitting of single bump on the twisted torus.'''
        dim = Pair2D(34, 30)
        x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
            np.arange(dim.x, dtype=float), np.arange(dim.y, dtype=float))

        failures = 0
        for _ in range(self.nIter):
            a = np.random.rand() * self.gaussianAMax
            mu_x = np.random.rand() * dim.x
            mu_y = np.random.rand() * dim.y
            sigma = self.minSigma + \
                np.random.rand() * (np.min((dim.x, dim.y)) / 4 - self.minSigma)
            noise_sigma = np.random.rand() * self.maxNoiseSigma * a

            est_params = fit_gaussian_bump_tt(
                (gen_gaussian_tt(a, mu_x, mu_y, sigma, x, y) +
                 np.random.randn(dim.y, dim.x) * noise_sigma))
            est_params.A = np.abs(est_params.A)
            est_params.sigma = np.abs(est_params.sigma)

            try:
                e = est_params
                first  = np.array([a,     mu_x,   mu_y,   sigma])
                second = np.array([e.A, e.mu_x, e.mu_y, e.sigma])
                np.testing.assert_almost_equal(first, second,
                                               self.decimalAlmostEqual)
                # print noise_sigma, np.sqrt(1./e.lh_precision)
                # print noise_sigma*self.noiseDeltaFrac
                np.testing.assert_allclose(
                    noise_sigma, np.sqrt(1. / e.lh_precision),
                    rtol=self.noiseDeltaFrac)
                assert isinstance(e.ln_lh, float) or isinstance(e.ln_lh, int)
            except AssertionError as e:
                failures += 1
                # print failures
                # print('%s != \n%s within %r places' % (first, second,
                #    self.decimalAlmostEqual))
                if failures / float(self.nIter) > self.maxFailures:
                    print('%.1f%%  fitting errors reached.' % (self.maxFailures
                                                               * 100))
                    assert False

    def test_maximum_lh(self):
        '''Test computation of maximum likelihood estimator.'''
        nvals = 100000
        aic_correction = 2
        for _ in range(self.nIter):
            mu = np.random.rand() * self.gaussianAMax
            sigma = np.random.rand() * np.sqrt(mu)
            our_data = sigma * np.random.randn(nvals) + mu
            ml_fit = fit_maximum_lh(our_data)

            # print ml_fit.mu, ml_fit.sigma2, ml_fit.ln_lh
            # print mu, sigma

            np.testing.assert_allclose(ml_fit.mu, mu, rtol=self.noiseDeltaFrac)
            np.testing.assert_allclose(np.sqrt(ml_fit.sigma2), sigma,
                                       rtol=self.noiseDeltaFrac)
            correct_ln_lh = - nvals / 2. * (1 + np.log(ml_fit.sigma2) +
                                            np.log(2 * np.pi)) - aic_correction
            np.testing.assert_allclose(ml_fit.ln_lh, correct_ln_lh, atol=1e-9,
                                       rtol=0)


class TestSingleBumpPopulation(object):
    '''Tests of SingleBumpPopulation.'''

    def test_interface(self):
        '''Test interfaces.'''
        tstart = 0.
        dt = 100.
        win_len = 200.

        pop = SingleBumpPopulation([], [], [10, 10])

        # One item
        tend = 0
        fits_gauss = pop.bump_position(tstart, tend, dt, win_len)
        fits_uni = pop.uniform_fit(tstart, tend, dt, win_len)
        assert isinstance(fits_gauss, MLGaussianFitList)
        assert isinstance(fits_uni, MLFitList)
        assert len(fits_gauss) == 1
        assert len(fits_uni) == 1

        # 11 items
        tend = 1e3
        fits_gauss = pop.bump_position(tstart, tend, dt, win_len)
        fits_uni = pop.uniform_fit(tstart, tend, dt, win_len)
        assert len(fits_gauss) == 11
        assert len(fits_uni) == 11

    @notimpl
    def test_bump_position(self):
        '''Test consecutive bump positions.'''
        pass

    @notimpl
    def test_uniform_fit(self):
        '''Test consecutive uniform fits.'''
        pass
