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

    References
    ----------
    .. [1] Skaggs, W.E. et al., 1993. An Information-Theoretic Approach to Deciphering
       the Hippocampal Code. In Advances in Neural Information Processing
       Systems 5. pp. 1030-1037.
    '''
    avg_rate = np.sum(np.ravel(rate_map * px))
    return np.nansum(np.ravel(rate_map * np.log2(rate_map/avg_rate) * px))


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
    avg_rate = np.sum(np.ravel(rate_map * px))
    I = np.nansum(np.ravel(rate_map * np.log2(rate_map/avg_rate) * px))
    return I / avg_rate
