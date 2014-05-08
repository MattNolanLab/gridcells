'''Grid field plotting'''
from __future__ import absolute_import, division, print_function

import numpy as np
import numpy.ma as ma

from ..fields import extractSpikePositions
from .low_level import xScaleBar

lim_factor = 1.1


def gridScaleBar(scaleLen, scaleText, ax):
    if (scaleLen is not None):
        if (scaleText):
            unitsText = 'cm'
        else:
            unitsText = None
        xScaleBar(scaleLen, x=0.7, y=-0.00, ax=ax, height=0.015,
                unitsText=unitsText, size='small')



def plotSpikes2D(spikeTimes, pos, arena, ax, titleStr='',
                 scaleBar=None, scaleText=True, spikeDotSize=5):
    '''Plot spike positions into the figure.
    
    Both positions and spikes must be aligned!

    Parameters
    ==========
    spikeTimes : np.ndarray
        Spike times to plot on top of the trajectories
    pos : gridcells.Position2D
        Positional data for the spike times.
    arena : gridcells.Arena
        Arena in which the recording was performed.
    ax : Axis to plot into

    Keyword arguments:
    ==================
    titleStr
    scaleBar
    scaleText
    spikeDotSize

    .. todo::
        Figure out the way how to document kwargs properly
    '''
    neuronPos, m_i = extractSpikePositions(spikeTimes, pos)

    ax.plot(pos.x, pos.y)
    ax.hold('on')
    ax.plot(neuronPos.x, neuronPos.y, 'or', markersize=spikeDotSize)
    ax.axis('off')
    ax.axis('scaled')
    ax.set_title(titleStr, va='bottom')
    sz = arena.getSize()
    ax.set_xlim([-lim_factor*sz.x/2., lim_factor*sz.x/2.])
    ax.set_ylim([-lim_factor*sz.y/2., lim_factor*sz.y/2.])
    gridScaleBar(scaleBar, scaleText, ax)



def plotSpatialRateMap(rateMap, arena, ax, titleStr="", scaleBar=None,
        scaleText=True, maxRate=True, G=None, **kw):
    '''
    Plot the grid-like rate map into the current axis
    '''
    #rateMap = ma.masked_array(rateMap, mask = np.sqrt(X**2 + Y**2) > diam/2.0)
    edges = arena.getDiscretisation()
    X, Y = np.meshgrid(edges.x, edges.y)
    ax.pcolormesh(X, Y, rateMap, **kw)
    ax.axis('scaled')
    ax.axis('off')
    ax.set_title(titleStr, va='bottom')
    sz = arena.getSize()
    ax.set_xlim([-lim_factor*sz.x/2., lim_factor*sz.x/2.])
    ax.set_ylim([-lim_factor*sz.y/2., lim_factor*sz.y/2.])
    gridScaleBar(scaleBar, scaleText, ax)
    if (maxRate):
        rStr = '{0:.1f} Hz'.format(np.max(rateMap.flatten()))
        ax.text(1.0, 1.025, rStr, ha="right", va='bottom', fontsize='xx-small',
                transform=ax.transAxes)
    if (G is not None):
        if (int(G*100)/100.0 == int(G)):
            gStr = '{0}'.format(int(G))
        else:
            gStr = '{0:.2f}'.format(G)
        ax.text(0, 1.025, gStr, ha="left", va='bottom', fontsize='xx-small',
                transform=ax.transAxes)


 
#def plotAutoCorrelation(ac, X, Y, diam, ax, titleStr="",
#        scaleBar=None, scaleText=True, **kw):
#    ac = ma.masked_array(ac, mask = np.sqrt(X**2 + Y**2) > diam)
#    ax.pcolormesh(X, Y, ac, **kw)
#    ax.axis('scaled')
#    ax.axis('off')
#    ax.set_title(titleStr, va='bottom')
#    if (diam != np.inf):
#        ax.set_xlim([-lim_factor*diam, lim_factor*diam])
#        ax.set_ylim([-lim_factor*diam, lim_factor*diam])
#    gridScaleBar(scaleBar, scaleText, ax)


