'''
.. currentmodule:: gridcells.fields

The :mod:`~gridcells.fields` module contains routines to analyse spiking data
either from experiments involoving a rodent running in an arena or simulations
involving an animat running in a simulated arena.

Functions
---------
.. autosummary::

    gaussianFilter
    extractSpikePositions2D
    SNSpatialRate2D
    SNAutoCorr
    cellGridnessScore

'''
import numpy    as np
import numpy.ma as ma

from scipy.integrate             import trapz
from scipy.signal                import correlate2d
from scipy.ndimage.interpolation import rotate

#__all__ = ['gaussianFilter', 'extractSpikePositions2D', 'SNSpatialRate2D',
#        'SNFiringRate', 'motionDirection', 'SNAutoCorr', 'cellGridnessScore']


def gaussianFilter(X, sigma):
    '''Simple Gaussian filter.

    Parameters
    ----------
    X : a numeric value or any sequence
        The X parameter for the Gaussian
    sigma : float
        Standard deviation of the Gaussian

    Returns
    -------
    type(X)
        The value of the Gaussian filter for the input argument. The shape is the
        same as `X`.
    '''
    return np.exp(-X**2/ 2.0 / sigma**2)



def extractSpikePositions2D(spikeTimes, pos_x, pos_y, dt):
    '''Extract positions of the animal for each spike.

    Extract spike positions from the animal tracking data and neuron spike times.
    Both animal positions positions and spikes must be *aligned in time*.

    Parameters
    ----------
    spikeTimes : np.ndarray
        A vector that contains spike times for the neuron.
    pos_x, pos_y : np.ndarray
        An array containing the animal/animat positional information. The
        positions must be evenly spaced in time.
    dt : float
        Time step of the positional information in `pos_x` and `pos_y`.

    Returns
    -------
    A tuple (npos_x, npos_y, max_index)
        A tuple containing positional information and an index into the
        positional data, of the last spike in `spikeTimes`.
    '''
    neuronPos_i = np.array(spikeTimes/dt, dtype=int)
    neuronPos_x = pos_x[neuronPos_i]
    neuronPos_y = pos_y[neuronPos_i]
    return (neuronPos_x, neuronPos_y, np.max(neuronPos_i))



def SNSpatialRate2D(spikeTimes, pos_x, pos_y, dt, arenaDiam, h):
    '''Compute spatial rate map for spikes of a given neuron.

    Preprocess neuron spike times into a smoothed spatial rate map, given arena
    parameters.  Both spike times and positional data must be aligned in time!
    The rate map will be smoothed by a gaussian kernel.

    Parameters
    ----------
    spikeTimes : np.ndarray
        Spike times for a given neuron.
    pos_x, pos_y : np.ndarray
        Vectors of positional data of where the animal was located.
    dt : float
        Time step of the positional data. The units must be the same as for
        `spikeTimes`.
    arenaDiam : float
        Diameter of the arena. The result will be a masked array in which all
        values outside the arena will be invalid.
    h : float
        Standard deviation of the Gaussian smoothing kernel.

    Returns
    -------
    rateMap : np.ndarray
        The 2D spatial firing rate map. The shape will be `(arenaDiam/h + 1,
        arenaDiam/h + 1)`.
    xedges, yedges : np.ndarray
        Values of the spatial lags for the correlation function. The same shape
        as `rateMap.shape[0]`.
    '''
    precision = arenaDiam/h
    xedges = np.linspace(-arenaDiam/2, arenaDiam/2, precision+1)
    yedges = np.linspace(-arenaDiam/2, arenaDiam/2, precision+1)

    rateMap = np.zeros((len(xedges), len(yedges)))

    for x_i in xrange(len(xedges)):
        for y_i in xrange(len(yedges)):
            x = xedges[x_i]
            y = yedges[y_i]
            isNearTrack = np.count_nonzero(np.sqrt((pos_x - x)**2 + (pos_y - y)**2) <= h) > 0

            if isNearTrack:
                normConst = trapz(gaussianFilter(np.sqrt((pos_x - x)**2 + (pos_y - y)**2), sigma=h), dx=dt)
                neuronPos_x, neuronPos_y, m_i = extractSpikePositions2D(spikeTimes, pos_x, pos_y, dt)
                spikes = np.sum(gaussianFilter(np.sqrt((neuronPos_x - x)**2 + (neuronPos_y - y)**2), sigma=h))
                rateMap[x_i, y_i] = spikes/normConst

    # Mask values which are outside the arena
    X, Y = np.meshgrid(xedges, yedges)
    rateMap = ma.masked_array(rateMap, mask = np.sqrt(X**2 + Y**2) > arenaDiam/2.0)

    return  rateMap.T, xedges, yedges



def SNAutoCorr(rateMap, arenaDiam, h):
    '''Compute autocorrelation function of the spatial firing rate map.

    This function assumes that the arena is a circle and masks all values of
    the autocorrelation that are outside the `arenaDiam`.

    .. todo::

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


def SNFiringRate(spikeTimes, tend, dt, winLen):
    '''
    Compute a windowed firing rate from action potential times
    spikeTimes  Spike timestamps (should be ordered)
    dt          Sliding window step (s)
    winLen      Sliding windown length (s)
    '''
    szRate = int((tend)/dt)+1
    r = np.ndarray((szRate, ))
    times = np.ndarray(szRate)
    for t_i in xrange(szRate):
        t = t_i*dt
        r[t_i] = np.sum(np.logical_and(spikeTimes > t-winLen/2, spikeTimes <
            t+winLen/2))
        times[t_i] = t

    return (r/winLen, times)


def motionDirection(pos_x, pos_y, pos_dt, tend, winLen):
    '''
    Estimate the direction of motion as an average angle of the
    directional vector in the windown of winLen.
    pos_x, pos_y    Tracking data
    pos_dt          Sampling rate of tracking data
    tend            End time to consider
    winLen          Window length (s)
    '''
    sz = int(tend/pos_dt) + 1
    angles = np.ndarray(sz)
    avg_spd = np.ndarray(sz)
    times = np.ndarray(sz)

    vel_x = np.diff(pos_x) / pos_dt
    vel_y = np.diff(pos_y) / pos_dt

    for t_i in xrange(sz):
        times[t_i] = t_i*pos_dt
        if t_i < len(vel_x):
            vel_x_win = np.mean(vel_x[t_i:t_i+winLen/pos_dt])
            vel_y_win = np.mean(vel_y[t_i:t_i+winLen/pos_dt])
            angles[t_i] = np.arctan2(vel_y_win, vel_x_win)
            avg_spd[t_i] = np.sqrt(vel_x_win**2 + vel_y_win**2)
        else:
            angles[t_i] = 0.0
            avg_spd[t_i] = 0.0

    return angles, times, avg_spd



def cellGridnessScore(rateMap, arenaDiam, h, corr_cutRmin):
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



