import unittest
import numpy as np

from gridcells.core import Pair2D, twisted_torus_distance
from gridcells.analysis.bumps import fitGaussianBumpTT, fitMaximumLikelihood

notImplMsg = "Not implemented"


##############################################################################
# Gaussian fitting on the twisted torus

def generateGaussianTT(A, mu_x, mu_y, sigma, X, Y):
    dim = Pair2D(X.shape[1], X.shape[0])
    a = Pair2D(mu_x, mu_y)
    others = Pair2D(X.ravel(), Y.ravel())
    dist = twisted_torus_distance(a, others, dim)
    G = A * np.exp( -dist**2 / (2*sigma**2)  )
    return np.reshape(G, (dim.y, dim.x))


class Test_FittingTT(unittest.TestCase):
    '''Test all implemented functions/classes that require fitting a Gaussian
    on the twisted torus.

    Notes
    -----
    `decimalAlmostEqual` and `noiseDeltaFrac` have been manually fine-tuned by
    inspecting the data. In general, it turns that the fitting procedure will
    no be better than 10%, i.e. one digit.

    .. todo::

        `decimalAlmostEqual` should be relative to the size of the torus. If
        the torus is too small this setting will effectively force all the
        tests to pass even if the values will differ.

        Also, log-likelihood tests would be useful
    '''
    decimalAlmostEqual = 1
    noiseDeltaFrac     = 1e-1

    gaussianAMax  = 40
    nIter         = 1000
    minSigma      = 1.
    maxFailures   = .02
    maxNoiseSigma = 1e-3


    def assertNdarrayAlmostEqual(self, first, second, msg=None):
        #print first, second
        np.testing.assert_almost_equal(first, second,
                self.decimalAlmostEqual)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, self.assertNdarrayAlmostEqual)


    def test_bumpFitting(self):
        dim = Pair2D(34, 30)
        X, Y = np.meshgrid(np.arange(dim.x, dtype=float),
                           np.arange(dim.y, dtype=float))
        failures = 0
        for it in xrange(self.nIter):
            A     = np.random.rand() * self.gaussianAMax
            mu_x  = np.random.rand() * dim.x
            mu_y  = np.random.rand() * dim.y
            sigma = self.minSigma + np.random.rand() * (np.min((dim.x, dim.y)) / 4 - self.minSigma)
            noise_sigma = np.random.rand() * self.maxNoiseSigma * A

            ourG = generateGaussianTT(A, mu_x, mu_y, sigma, X, Y) + \
                    np.random.randn(dim.y, dim.x)*noise_sigma
            estParams = fitGaussianBumpTT(ourG)
            estParams.A = np.abs(estParams.A)
            estParams.sigma = np.abs(estParams.sigma)

            try:
                e = estParams
                first  = np.array([  A,   mu_x,   mu_y,   sigma])
                second = np.array([e.A, e.mu_x, e.mu_y, e.sigma])
                self.assertNdarrayAlmostEqual(first, second)
                #print noise_sigma, np.sqrt(1./e.lh_precision)
                #print noise_sigma*self.noiseDeltaFrac
                self.assertAlmostEqual(noise_sigma, np.sqrt(1./e.lh_precision),
                        delta=noise_sigma*self.noiseDeltaFrac)
                self.assertTrue(isinstance(e.ln_L, float) or isinstance(e.ln_L,
                    int), msg='Log likelihood must be either float or int!')
            except (AssertionError, self.failureException) as e:
                failures += 1
                #print failures
                #print('%s != \n%s within %r places' % (first, second,
                #    self.decimalAlmostEqual))
                if failures / float(self.nIter) > self.maxFailures:
                    msg = '%.1f%%  fitting errors reached.' % (self.maxFailures*100)
                    raise self.failureException(msg)
                

    def test_maximumLikelihood(self):
        nVals = 100000
        AIC_correction = 2
        for it in xrange(self.nIter):
            mu = np.random.rand() * self.gaussianAMax
            sigma = np.random.rand() * np.sqrt(mu)
            ourData = sigma * np.random.randn(nVals) + mu
            mlFit = fitMaximumLikelihood(ourData)

            #print mlFit.mu, mlFit.sigma2, mlFit.ln_L
            #print mu, sigma

            deltaMu     = mu * self.noiseDeltaFrac
            deltaSigma  = sigma * self.noiseDeltaFrac
            self.assertAlmostEqual(mlFit.mu, mu, delta=deltaMu)
            self.assertAlmostEqual(np.sqrt(mlFit.sigma2), sigma, delta=deltaSigma)
            correct_ln_L = - nVals / 2. * (1 + np.log(mlFit.sigma2) +
                np.log(2*np.pi)) - AIC_correction
            self.assertAlmostEqual(mlFit.ln_L, correct_ln_L, delta=1e-9)







