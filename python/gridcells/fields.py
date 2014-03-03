#
#   grid_cells.py
#
#   Grid cell analysis module. Use this to analyse spikes/membrane potentials
#   of grid cell models.
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

import numpy    as np
import numpy.ma as ma

from scipy.integrate             import trapz
from scipy.signal                import correlate2d
from scipy.ndimage.interpolation import rotate

__all__ = ['gaussianFilter', 'extractSpikePositions2D', 'SNSpatialRate2D',
        'SNFiringRate', 'motionDirection', 'SNAutoCorr', 'cellGridnessScore']



## Simple Gaussian function
#
# @param X      The X parameter for the Gaussian
# @param sigma  Std. deviation of the Gaussian
def gaussianFilter(X, sigma):
    return np.exp(-X**2/ 2.0 / sigma**2)



def extractSpikePositions2D(spikeTimes, rat_pos_x, rat_pos_y, dt):
    '''
    Extract spike positions from the rat tracking data and cell spike times.
    Both positions and spikes must be aligned!
    '''
    neuronPos_i = np.array(spikeTimes/dt, dtype=int)
    neuronPos_x = rat_pos_x[neuronPos_i]
    neuronPos_y = rat_pos_y[neuronPos_i]

    return (neuronPos_x, neuronPos_y, np.max(neuronPos_i))


def SNSpatialRate2D(spikeTimes, rat_pos_x, rat_pos_y, dt, arenaDiam, h):
    '''
    Preprocess neuron spike times into a spatial rate map, given arena parameters.
    Both spike times and rat tracking data must be aligned in time!
    '''
    precision = arenaDiam/h
    xedges = np.linspace(-arenaDiam/2, arenaDiam/2, precision+1)
    yedges = np.linspace(-arenaDiam/2, arenaDiam/2, precision+1)

    rateMap = np.zeros((len(xedges), len(yedges)))

    for x_i in xrange(len(xedges)):
        for y_i in xrange(len(yedges)):
            x = xedges[x_i]
            y = yedges[y_i]
            isNearTrack = np.count_nonzero(np.sqrt((rat_pos_x - x)**2 + (rat_pos_y - y)**2) <= h) > 0

            if isNearTrack:
                normConst = trapz(gaussianFilter(np.sqrt((rat_pos_x - x)**2 + (rat_pos_y - y)**2), sigma=h), dx=dt)
                neuronPos_x, neuronPos_y, m_i = extractSpikePositions2D(spikeTimes, rat_pos_x, rat_pos_y, dt)
                spikes = np.sum(gaussianFilter(np.sqrt((neuronPos_x - x)**2 + (neuronPos_y - y)**2), sigma=h))
                rateMap[x_i, y_i] = spikes/normConst

    # Mask values which are outside the arena
    X, Y = np.meshgrid(xedges, yedges)
    rateMap = ma.masked_array(rateMap, mask = np.sqrt(X**2 + Y**2) > arenaDiam/2.0)

    return  rateMap.T, xedges, yedges



def SNAutoCorr(rateMap, arenaDiam, h):
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
    '''
    Compute a cell gridness score by taking the auto correlation of the
    firing rate map, rotating it, and subtracting maxima of the
    correlation coefficients of the former and latter, at 30, 90 and 150 (max),
    and 60 and 120 deg. (minima). This gives the gridness score.

    The center of the auto correlation map (given by corr_cutRmin) is removed
    from the map
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



