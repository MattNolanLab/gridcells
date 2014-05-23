'''
==============================================================
:mod:`gridcells.plotting.fields` - grid field related plotting
==============================================================

The :mod:`~gridcells.plotting.fields` module contains routines to create
matplotlib plots of spatial firing fields and similar commonly used structures.


How to plot
-----------

The plotting is currently realized as a subclass of ``matplotlib.axes.Axes``
and is used via a ``projection="gridcells_arena"`` keyword argument. Since a
custom ``Axes`` class is not part of standard matplotlib, before using the
``projection``, you have to first register the plotting class with matplotlib
by importing the plotting module::

    import matplotlib.pyplot as plt
    from gridcells.plotting import fields
    from gridcells.core import SquareArena, Pair2D

Next, create an Axes object that you can plot to::

    fig = plt.figure()
    arena = SquareArena(100., Pair2D(1., 1.))
    ax = fig.add_subplot(111, projection="gridcells_arena", arena=arena)

The ``add_subplot`` method takes a keyword argument
``projection="gridcells_arena"`` that specifies the type of Axes to use. Here,
we also have to specify the ``arena`` parameter in the form of
:class:`~gridcells.core.arena.Arena` instance. In our case we have created a
square arena with size 100x100 and a discretisation of 1x1 (in arbitrary
units).

Next, for illustration purposes, we create a random spatial rate map with size
compatible with the current arena, and plot to the axes, by calling
:meth:`~gridcells.plotting.fields.GridArenaAxes.spatial_rate_map`::

    sz = arena.getDiscretisation()
    rate_map = np.random.rand(len(sz.x), len(sz.y))
    ax.spatial_rate_map(rate_map)


Custom grid cell plotting Axes
------------------------------
.. autosummary::

    GridArenaAxes

'''
from __future__ import absolute_import, division, print_function

__all__ = ['GridArenaAxes']

import numpy as np

import matplotlib as mpl
from matplotlib.axes import Axes as MplAxes

from ..analysis import extractSpikePositions
from .low_level import xScaleBar

default_margin = .1


def _scale_bar(scaleLen, scaleText, ax):
    if (scaleLen is not None):
        if scaleText:
            unitsText = scaleText
        else:
            unitsText = None
        xScaleBar(scaleLen, x=0.6, y=-.05, ax=ax, height=0.015,
                  unitsText=unitsText, size='small')


def _set_arena_limits(arena, margin_factor, ax):
    margin_x = arena.getSize().x * margin_factor
    margin_y = arena.getSize().y * margin_factor
    ax.set_xlim([arena.bounds.x[0] - margin_x, arena.bounds.x[1] + margin_x])
    ax.set_ylim([arena.bounds.y[0] - margin_y, arena.bounds.y[1] + margin_y])


class GridArenaAxes(MplAxes):
    '''A custom matplotlib Axes that allows to plot figures in the shape of
    arenas.

    '''
    name = "gridcells_arena"

    def __init__(self, *args, **kwargs):
        self._arena = kwargs.pop('arena')
        MplAxes.__init__(self, *args, **kwargs)

    @property
    def arena(self):
        return self._arena

    @arena.setter
    def arena(self, a):
        self._arena = a

    def fft2(self, rate_map, scalebar=None, scaletext='$cm^{-1}$', FFTN=None,
             subtractmean=True):
        '''Plot a 2D Fourier transform (power) of a spatial rate map.

        Parameters
        ==========
        rate_map : np.ndarray
            The rate map as 2D array. Rows determine the Y coordinate, columns
            the X coordinate. Masked items will be ignored.
        scalebar : float, optional
            The length of the scale bar that will be plotted as horizontal
            line. Must be in data units.
        scaletext : str, optional
            Text after the scale bar number, i.e. units.
        FFTN : int, optional
            Size of the array that the Fourier transform is actually computed
            from. If ``None`` it will be ``max(rate_map.shape)``. Otherwise the
            ``rate_map`` will be padded with zeros.
        subtractmean : bool, optional
            Whether to subtract the mean of the signal before computing the
            FFT. This will remove any constant component in the centre of the
            spectrogram.
        '''
        if FFTN is None:
            FFTN = np.max(rate_map.shape)

        rate_map = np.copy(rate_map)
        rate_map[np.isnan(rate_map)] = 0
        if subtractmean:
            rate_map -= np.mean(rate_map)

        ds = self._arena.getDiscretisationSteps()
        FsX = 1. / ds.x  # units: specified by caller
        FsY = 1. / ds.y

        rateMap_pad = np.zeros((FFTN, FFTN))
        rateMap_pad[0:rate_map.shape[0], 0:rate_map.shape[0]] = rate_map
        FT = np.fft.fft2(rateMap_pad)
        iFT = np.fft.ifft2(FT)[:rate_map.shape[0], :rate_map.shape[1]]

        fxy = np.linspace(-1.0, 1.0, FFTN)
        FX, FY = np.meshgrid(fxy, fxy)
        FX *= FsX/2.0
        FY *= FsY/2.0

        PSD_centered = np.abs(np.fft.fftshift(FT))**2
        self.pcolormesh(FX, FY, PSD_centered)
        self.axis('scaled')
        self.axis('off')

        _scale_bar(scalebar, scaletext, self)

        return FT, iFT

    def spatial_rate_map(self, rate_map, scalebar=None, scaletext='cm',
                         maxrate=True, G=None, **kwargs):
        '''
        Plot the spatial rate map in the specified arena.

        Parameters
        ==========
        rate_map : np.ndarray
            The rate map as 2D array. Rows determine the Y coordinate, columns
            the X coordinate. Masked items will be ignored.
        scalebar : float, optional
            The length of the scale bar that will be plotted as horizontal
            line. Must be in data units.
        scaletext : str, optional
            Text after the scale bar number, i.e. units.
        maxrate : bool, optional
            Whether to print the max firing rate (top right corner)
        G : float, optional
            Grid score for this spatial rate map. If ``None``, plot nothing.
        kwargs : kwargs
            Optional kwargs that will be passed to matplotlib's pcolormesh.
        '''
        edges = self._arena.getDiscretisation()
        X, Y = np.meshgrid(edges.x, edges.y)
        self.pcolormesh(X, Y, rate_map, **kwargs)
        self.axis('scaled')
        self.axis('off')
        _set_arena_limits(self._arena, default_margin, self)
        _scale_bar(scalebar, scaletext, self)
        if (maxrate):
            rStr = '{0:.1f} Hz'.format(np.max(rate_map.flatten()))
            self.text(1.0, 1.025, rStr, ha="right", va='bottom',
                      fontsize='xx-small', transform=self.transAxes)
        if (G is not None):
            if (int(G*100)/100.0 == int(G)):
                gStr = '{0}'.format(int(G))
            else:
                gStr = '{0:.2f}'.format(G)
            self.text(0, 1.025, gStr, ha="left", va='bottom',
                      fontsize='xx-small', transform=self.transAxes)

    def spikes(self, spike_times, pos, dotsize=5, scalebar=None,
               scaletext='cm'):
        '''Plot spike positions.

        Both positions and spikes must be aligned!

        Parameters
        ==========
        spike_times : np.ndarray
            Spike times to plot on top of the trajectories
        pos : gridcells.Position2D
            Positional data for the spike times.

        Keyword arguments:
        ==================
        scalebar
        scaletext
        dotsize

        .. todo::
            Document kwargs properly
        '''
        neuronPos, m_i = extractSpikePositions(spike_times, pos)

        self.plot(pos.x, pos.y)
        self.hold('on')
        self.plot(neuronPos.x, neuronPos.y, 'or', markersize=dotsize)
        self.axis('off')
        self.axis('scaled')
        _set_arena_limits(self._arena, default_margin, self)
        _scale_bar(scalebar, scaletext, self)


#   def plotAutoCorrelation(ac, X, Y, diam, ax, titleStr="",
#           scaleBar=None, scaleText=True, **kw):
#       ac = ma.masked_array(ac, mask = np.sqrt(X**2 + Y**2) > diam)
#       ax.pcolormesh(X, Y, ac, **kw)
#       ax.axis('scaled')
#       ax.axis('off')
#       ax.set_title(titleStr, va='bottom')
#       if (diam != np.inf):
#           ax.set_xlim([-lim_factor*diam, lim_factor*diam])
#           ax.set_ylim([-lim_factor*diam, lim_factor*diam])
#       _scale_bar(scaleBar, scaleText, ax)


# Register with matplotlib
mpl.projections.register_projection(GridArenaAxes)
