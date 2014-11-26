'''
.. currentmodule:: plotting.low_level

Low level plotting sub-routines.
'''
import numpy as np

import matplotlib.pyplot as m
import matplotlib.patches as patches
from matplotlib import rcParams


def hscalebar(scalelen, x, y, ax=None, height=0.02, color='black',
              unitstext='ms', size='medium', textyoffset=0.075):
    '''
    Plot a horizontal (X) scale bar into the axes.

    **Parameters:**
    scalelen : float
        Size of the scale (X) in data coordinates.
    ax : mpl.axes.Axes
        Axes object. If unspecified, use the current axes.
    x  : float, optional
        Left end of the scale bar, in axis coordinates.
    y : float, optional
        Bottom position of the scale bar, in axis units. This excludes the
        scale text
    height : float, optional
        Height of the scale bar, in relative axis units.
    color
        Color of the bar.
    unitstext : string
        Units drawn below the scale bar.
    size
        Size of the text below the scale bar.
    textyoffset : float
        Offset of the text from the scale bar. Positive value is a downward
        offset.
    '''
    if ax is None:
        ax = m.gca()

    (left, right) = ax.get_xlim()
    axislen = scalelen / (right - left)
    scalecenter = x + 0.5*axislen
    rect = patches.Rectangle((x, y), width=axislen, height=height,
                             transform=ax.transAxes, color=color)
    rect.set_clip_on(False)
    ax.add_patch(rect)
    if (unitstext is not None):
        texttemplate = '{0}'
        if (unitstext != ''):
            texttemplate += ' {1}'
        ax.text(scalecenter, y - textyoffset,
                texttemplate.format(scalelen, unitstext),
                va='top', ha='center', transform=ax.transAxes, size=size)


def vscalebar(scalelen, x, y, ax=None, width=0.0075, color='black',
              unitstext='ms', size='medium', textxoffset=0.075):
    '''
    Plot a vertical (Y) scale bar into the axes.

    **Parameters:**
    scalelen : float
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
    unitstext : string
        Units drawn below the scale bar.
    size
        Size of the text below the scale bar.
    textxoffset : float
        Offset of the text from the scale bar. Positive value is a leftward
        offset.
    '''
    if ax is None:
        ax = m.gca()

    (bottom, top) = ax.get_ylim()
    axisheight = scalelen / (top - bottom)
    scalecenter = y + 0.5*axisheight
    rect = patches.Rectangle((x, y), width=width, height=axisheight,
                             transform=ax.transAxes, color=color)
    rect.set_clip_on(False)
    ax.add_patch(rect)
    if (unitstext is not None):
        texttemplate = '{0}'
        if (unitstext != ''):
            texttemplate += ' {1}'
        ax.text(x - textxoffset, scalecenter,
                texttemplate.format(scalelen, unitstext),
                va='center', ha='left', transform=ax.transAxes, size=size,
                rotation=90)


def remove_all_spines(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)


def zero_lines(ax, which='both'):
    color = rcParams['grid.color']
    ls = rcParams['grid.linestyle']
    lw = rcParams['grid.linewidth']

    if (which == 'x' or which == 'both'):
        ax.axvline(0, ls=ls, lw=lw, color=color, zorder=-10)

    if (which == 'y' or which == 'both'):
        ax.axhline(0, ls=ls, lw=lw, color=color, zorder=-10)


def symmetric_data_limits(data):
    absmax = np.max(np.abs(data.flatten()))
    return (-absmax, absmax)
