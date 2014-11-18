import unittest
import numpy as np

from gridcells.core import Pair2D, twisted_torus_distance
from gridcells.analysis.bumps import fit_gaussian_bump_tt, fit_maximum_lh

notImplMsg = "Not implemented"


##############################################################################
# Gaussian fitting on the twisted torus

def gen_gaussian_tt(ampl, mu_x, mu_y, sigma, x, y):
    dim = Pair2D(x.shape[1], x.shape[0])
    first = Pair2D(mu_x, mu_y)
    second = Pair2D(x.ravel(), y.ravel())
    dist = twisted_torus_distance(first, second, dim)
    g = ampl * np.exp(-dist ** 2 / (2 * sigma ** 2))
    return np.reshape(g, (dim.y, dim.x))


class TestFittingTT(unittest.TestCase):

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
    noiseDeltaFrac = 1e-1

    gaussianAMax = 40
    nIter = 1000
    minSigma = 1.
    maxFailures = .02
    maxNoiseSigma = 1e-3

    def assert_ndarray_almost_equal(self, first, second, msg=None):
        # print first, second
        np.testing.assert_almost_equal(first, second,
                                       self.decimalAlmostEqual)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, self.assert_ndarray_almost_equal)

    def test_bump_fitting(self):
        dim = Pair2D(34, 30)
        x, y = np.meshgrid(np.arange(dim.x, dtype=float),
                           np.arange(dim.y, dtype=float))
        failures = 0
        for it in xrange(self.nIter):
            a = np.random.rand() * self.gaussianAMax
            mu_x = np.random.rand() * dim.x
            mu_y = np.random.rand() * dim.y
            sigma = self.minSigma + \
                np.random.rand() * (np.min((dim.x, dim.y)) / 4 - self.minSigma)
            noise_sigma = np.random.rand() * self.maxNoiseSigma * a

            our_g = gen_gaussian_tt(a, mu_x, mu_y, sigma, x, y) + \
                np.random.randn(dim.y, dim.x) * noise_sigma
            est_params = fit_gaussian_bump_tt(our_g)
            est_params.A = np.abs(est_params.A)
            est_params.sigma = np.abs(est_params.sigma)

            try:
                e = est_params
                first  = np.array([a,   mu_x,   mu_y,   sigma])
                second = np.array([e.A, e.mu_x, e.mu_y, e.sigma])
                self.assert_ndarray_almost_equal(first, second)
                # print noise_sigma, np.sqrt(1./e.lh_precision)
                # print noise_sigma*self.noiseDeltaFrac
                self.assertAlmostEqual(noise_sigma, np.sqrt(1. /
                                       e.lh_precision), delta=noise_sigma *
                                       self.noiseDeltaFrac)
                self.assertTrue(isinstance(e.ln_lh, float) or
                                isinstance(e.ln_lh, int),
                                msg='Log likelihood must be either float or ' +
                                    'int!')
            except (AssertionError, self.failureException) as e:
                failures += 1
                # print failures
                # print('%s != \n%s within %r places' % (first, second,
                #    self.decimalAlmostEqual))
                if failures / float(self.nIter) > self.maxFailures:
                    msg = '%.1f%%  fitting errors reached.' % (
                        self.maxFailures * 100)
                    raise self.failureException(msg)

    def test_maximum_lh(self):
        nvals = 100000
        aic_correction = 2
        for it in xrange(self.nIter):
            mu = np.random.rand() * self.gaussianAMax
            sigma = np.random.rand() * np.sqrt(mu)
            our_data = sigma * np.random.randn(nvals) + mu
            ml_fit = fit_maximum_lh(our_data)

            # print ml_fit.mu, ml_fit.sigma2, ml_fit.ln_lh
            # print mu, sigma

            delta_mu = mu * self.noiseDeltaFrac
            delta_sigma = sigma * self.noiseDeltaFrac
            self.assertAlmostEqual(ml_fit.mu, mu, delta=delta_mu)
            self.assertAlmostEqual(
                np.sqrt(ml_fit.sigma2), sigma, delta=delta_sigma)
            correct_ln_lh = - nvals / 2. * (1 + np.log(ml_fit.sigma2) +
                                            np.log(2 * np.pi)) - aic_correction
            self.assertAlmostEqual(ml_fit.ln_lh, correct_ln_lh, delta=1e-9)
