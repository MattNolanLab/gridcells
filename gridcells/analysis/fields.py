'''
===============================================================
:mod:`gridcells.analysis.fields` - grid field related analysis
===============================================================

The :mod:`~gridcells.analysis.fields` module contains routines to analyse
spiking data either from experiments involoving a rodent running in an arena or
simulations involving an animat running in a simulated arena.

Functions
---------
.. autosummary::

    gridnessScore
    spatialAutoCorrelation
    spatialRateMap

'''
from __future__ import absolute_import, division, print_function

import numpy    as np
import numpy.ma as ma

from scipy.integrate             import trapz
from scipy.signal                import correlate2d
from scipy.ndimage.interpolation import rotate

from . import _fields
from ..core import Pair2D


def spatialRateMap(spikeTimes, positions, arena, sigma):
    '''Compute spatial rate map for spikes of a given neuron.

    Preprocess neuron spike times into a smoothed spatial rate map, given arena
    parameters.  Both spike times and positional data must be aligned in time!
    The rate map will be smoothed by a gaussian kernel.

    Parameters
    ----------
    spikeTimes : np.ndarray
        Spike times for a given neuron.
    positions : gridcells.core.Position2D
        Positional data for these spikes. The timing must be aligned with
        ``spikeTimes``
    arena : gridcells.core.Arena
        The specification of the arena in which movement was carried out.
    sigma : float
        Standard deviation of the Gaussian smoothing kernel.

    Returns
    -------
    rateMap : np.ma.MaskedArray
        The 2D spatial firing rate map. The shape will be determined by the
        arena type.
    '''
    spikeTimes = np.asarray(spikeTimes, dtype=np.double)
    edges = arena.getDiscretisation()
    rateMap = _fields.spatialRateMap(spikeTimes,
                                     positions.x, positions.y, positions.dt, 
                                     edges.x, edges.y,
                                     sigma)
    # Mask values which are outside the arena
    rateMap = np.ma.MaskedArray(rateMap, mask=arena.getMask(), copy=False)
    return  rateMap.T



def spatialAutoCorrelation(rateMap, arenaDiam, h):
    '''Compute autocorrelation function of the spatial firing rate map.

    This function assumes that the arena is a circle and masks all values of
    the autocorrelation that are outside the `arenaDiam`.

    .. warning::

        This function will undergo serious interface changes in the future.

    Parameters
    ----------
    rateMap : np.ndarray
        Spatial firing rate map (2D). The shape should be `(arenadiam/h+1,
        arenadiam/2+1)`.
    arenaDiam : float
        Diameter of the arena.
    h : float
        Precision of the spatial firing rate map.

    Returns
    -------
    corr : np.ndarray
        The autocorrelation function, of shape `(arenadiam/h*2+1,
        arenaDiam/h*2+1)`
    xedges, yedges : np.ndarray
        Values of the spatial lags for the correlation function. The same shape
        as `corr.shape[0]`.
    '''
    precision = arenaDiam/h
    xedges = np.linspace(-arenaDiam, arenaDiam, precision*2 + 1)
    yedges = np.linspace(-arenaDiam, arenaDiam, precision*2 + 1)
    X, Y = np.meshgrid(xedges, yedges)

    corr = ma.masked_array(correlate2d(rateMap, rateMap), mask = np.sqrt(X**2 + Y**2) > arenaDiam)

    return corr, xedges, yedges


def gridnessScore(rateMap, arenaDiam, h, corr_cutRmin):
    '''Calculate gridness score of a spatial firing rate map.

    Parameters
    ----------
    rateMap : np.ndarray
        Spatial firing rate map.
    arenaDiam : float
        The diameter of the arena.
    h : float
        Precision of the spatial firing rate map.

    Returns
    -------
    G : float
        Gridness score.
    crossCorr : np.ndarray
        An array containing cross correlation values of the rotated
        autocorrelations, with the original autocorrelation.
    angles : np.ndarray
        An array of angles corresponding to the `crossCorr` array.


    Notes
    -----
    This function computes gridness score accoring to [1]_. The auto
    correlation of the firing rate map is rotated in 3 degree steps. The
    resulting gridness score is the difference between a minimum of cross
    correlations at 60 and 90 degrees, and a maximum of cross correlations at
    30, 90 and 150 degrees.
    
    The center of the auto correlation map (given by corr_cutRmin) is removed
    from the map.

    References
    ----------
    .. [1] Hafting, T. et al., 2005. Microstructure of a spatial map in the
       entorhinal cortex. Nature, 436(7052), pp.801-806.
    '''
    rateMap_mean = rateMap - np.mean(np.reshape(rateMap, (1, rateMap.size)))
    autoCorr, autoC_xedges, autoC_yedges = SNAutoCorr(rateMap_mean, arenaDiam, h)
    
    # Remove the center point and
    X, Y = np.meshgrid(autoC_xedges, autoC_yedges)
    autoCorr[np.sqrt(X**2 + Y**2) < corr_cutRmin] = 0
    
    da = 3
    angles = range(0, 180+da, da)
    crossCorr = []
    # Rotate and compute correlation coefficient
    for angle in angles:
        autoCorrRot = rotate(autoCorr, angle, reshape=False)
        C = np.corrcoef(np.reshape(autoCorr, (1, autoCorr.size)),
            np.reshape(autoCorrRot, (1, autoCorrRot.size)))
        crossCorr.append(C[0, 1])

    max_angles_i = np.array([30, 90, 150]) / da
    min_angles_i = np.array([60, 120]) / da

    maxima = np.max(np.array(crossCorr)[max_angles_i])
    minima = np.min(np.array(crossCorr)[min_angles_i])
    G = minima - maxima

    return G, np.array(crossCorr), angles


def extractSpikePositions(spikeTimes, positions):
    spikeIdx = spikeTimes // positions.dt
    return (Pair2D(positions.x[spikeIdx], positions.y[spikeIdx]),
            np.max(spikeIdx))


