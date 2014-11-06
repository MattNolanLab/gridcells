'''
=================================================================
:mod:`gridcells.analysis.info` - Information-theoretical analysis
=================================================================

The :mod:`~gridcells.analysis.info` module contains routines related to
information-theoretic analysis of data related to grid cells.

Functions
---------
.. autosummary::

    information_rate
    information_specificity

'''
from __future__ import absolute_import, division, print_function

import numpy as np


def _inf_rate(rate_map, px):
    '''A helper function for information rate.'''
    tmp_map = np.ma.array(rate_map, copy=True)
    tmp_map[np.isnan(tmp_map)] = 0
    avg_rate = np.sum(np.ravel(tmp_map * px))
    return (np.nansum(np.ravel(tmp_map * np.log2(tmp_map/avg_rate) * px)),
            avg_rate)

def information_rate(rate_map, px):
    '''Compute information rate of a cell given variable x.

    A simple algorithm devised by [1]_. This computes the spatial information
    rate of cell spikes given variable x (e.g. position, head direction) in
    bits/second.

    Parameters
    ----------
    rate_map : numpy.ndarray
        A firing rate map, any number of dimensions. If units are in Hz, then
        the information rate will be in bits/s.
    px : numpy.ndarray
        Probability density function for variable ``x``. ``px.shape`` must be
        equal ``rate_maps.shape``

    Returns
    -------
    I : float
        Information rate.

    Notes
    -----
    If you need information in bits/spike, you need to divide the information
    rate by the average firing rate of the cell.

    The firing rate map, in positions that are valid within the arena, may
    contains NaN numbers. In that case, the firing rate in these positions in
    ``rate_map`` will be set to 0.

    References
    ----------
    .. [1] Skaggs, W.E. et al., 1993. An Information-Theoretic Approach to Deciphering
       the Hippocampal Code. In Advances in Neural Information Processing
       Systems 5. pp. 1030-1037.
    '''
    return _inf_rate(rate_map, px)[0]


def information_specificity(rate_map, px):
    '''Compute the 'specificity' of the cell firing rate to a variable X.

    Compute :func:`information_rate` for this cell and divide by the average
    firing rate of the cell. See [1]_ for more information.

    Parameters
    ----------
    rate_map : numpy.ndarray
        A firing rate map, any number of dimensions.
    px : numpy.ndarray
        Probability density function for variable ``x``. ``px.shape`` must be
        equal ``rate_maps.shape``

    Returns
    -------
    I : float
        Information in bits/spike.

    References
    ----------
    .. [1] Skaggs, W.E. et al., 1993. An Information-Theoretic Approach to Deciphering
       the Hippocampal Code. In Advances in Neural Information Processing
       Systems 5. pp. 1030-1037.
    '''
    I, avg_rate = _inf_rate(rate_map, px)
    return I / avg_rate
