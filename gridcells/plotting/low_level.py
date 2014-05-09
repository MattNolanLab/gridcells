'''
.. currentmodule:: plotting.low_level

Low level plotting sub-routines.
'''
import numpy as np

import matplotlib.pyplot     as m
import matplotlib.transforms as transforms
import matplotlib.patches    as patches

from matplotlib import rcParams as rcp

def xScaleBar(scaleLen, x, y, ax=None, height=0.02, color='black',
        unitsText='ms', size='medium', textYOffset=0.075):
    '''
    Plot a horizontal (X) scale bar into the axes.

    **Parameters:**
    scaleLen : float
        Size of the scale (X) in data coordinates.
    ax : mpl.axes.Axes
        Axes object. If unspecified, use the current axes.
    x  : float, optional
        Left end of the scale bar, in axis coordinates.
    y : float, optional
        Bottom position of the scale bar, in axis units. This excludes the scale
        text
    height : float, optional
        Height of the scale bar, in relative axis units.
    color
        Color of the bar.
    unitsText : string
        Units drawn below the scale bar.
    size 
        Size of the text below the scale bar.
    textYOffset : float
        Offset of the text from the scale bar. Positive value is a downward
        offset.
    '''
    if ax is None: ax = m.gca()

    (left, right) = ax.get_xlim()
    axisLen = scaleLen / (right - left)
    scaleCenter = x + 0.5*axisLen
    rect = patches.Rectangle((x,y), width=axisLen, height=height,
            transform=ax.transAxes, color=color)
    rect.set_clip_on(False)
    ax.add_patch(rect)
    if (unitsText is not None):
        textTemplate = '{0}'
        if (unitsText != ''):
            textTemplate += ' {1}'
        ax.text(scaleCenter, y - textYOffset,
                textTemplate.format(scaleLen, unitsText),
                va='top', ha='center', transform=ax.transAxes, size=size)


def yScaleBar(scaleLen, x, y, ax=None, width=0.0075, color='black',
        unitsText='ms', size='medium', textXOffset=0.075):
    '''
    Plot a vertical (Y) scale bar into the axes.

    **Parameters:**
    scaleLen : float
        Size of the scale (X) in data coordinates.
    ax : mpl.axes.Axes
        Axes object. If unspecified, use the current axes.
    x  : float, optional
        Left position of the scale bar, in axis coordinates. This excludes the
        scale text
    y : float, optional
        Bottom end of the scale bar, in axis units.
    width : float, optional
        Width of the scale bar, in relative axis units.
    color
        Color of the bar.
    unitsText : string
        Units drawn below the scale bar.
    size 
        Size of the text below the scale bar.
    textXOffset : float
        Offset of the text from the scale bar. Positive value is a leftward
        offset.
    '''
    if ax is None: ax = m.gca()

    (bottom, top) = ax.get_ylim()
    axisHeight = scaleLen / (top - bottom)
    scaleCenter = y + 0.5*axisHeight
    rect = patches.Rectangle((x,y), width=width, height=axisHeight,
            transform=ax.transAxes, color=color)
    rect.set_clip_on(False)
    ax.add_patch(rect)
    if (unitsText is not None):
        textTemplate = '{0}'
        if (unitsText != ''):
            textTemplate += ' {1}'
        ax.text(x - textXOffset, scaleCenter,
                textTemplate.format(scaleLen, unitsText),
                va='center', ha='left', transform=ax.transAxes, size=size,
                rotation=90)


def removeAllSpines(ax):
    for spine in ax.spines.itervalues():
        spine.set_visible(False)


def zeroLines(ax, which='both'):
    color  = rcp['grid.color']
    ls     = rcp['grid.linestyle']
    lw     = rcp['grid.linewidth']
    alphsa = rcp['grid.alpha']

    if (which == 'x' or which == 'both'):
        ax.axvline(0, ls=ls, lw=lw, color=color, zorder=-10)

    if (which == 'y' or which == 'both'):
        ax.axhline(0, ls=ls, lw=lw, color=color, zorder=-10)


def symmetricDataLimits(data):
    absmax = np.max(np.abs(data.flatten()))
    return (-absmax, absmax)

