from __future__ import absolute_import, division, print_function

import numpy as np
from gridcells.analysis import information_rate, information_specificity


class TestInformationRate(object):
    '''Test computation of the information rate'''
    def test_basic(self):
        # Spikes in half of the arena, uniform occupancy
        rate_map = np.array([10., 0.])
        px = np.array([.5, .5])
        rate = information_rate(rate_map, px)
        spec = information_specificity(rate_map, px)
        assert rate == 5  # bits/s
        assert spec == 1  # bits/spike

        rate_map = np.array([
                [10., 0.],
                [10., 0.],
        ])
        px = np.array([
            [.25, .25],
            [.25, .25],
        ])
        rate = information_rate(rate_map, px)
        spec = information_specificity(rate_map, px)
        assert rate == 5  # bits/s
        assert spec == 1  # bits/spike

        
        # Spikes in 1/4 of the arena, uniform occupancy
        rate_map = np.array([10., 0., 0., 0.])
        px = np.array([.25, .25, .25, .25])
        rate = information_rate(rate_map, px)
        spec = information_specificity(rate_map, px)
        assert rate == 5  # bits/s
        assert spec == 2  # bits/spike

        rate_map = np.array([
                [10., 0.],
                [ 0., 0.],
        ])
        px = np.array([
            [.25, .25],
            [.25, .25],
        ])
        rate = information_rate(rate_map, px)
        spec = information_specificity(rate_map, px)
        assert rate == 5  # bits/s
        assert spec == 2  # bits/spike

    def test_nans(self):
        # Spikes in 1/4 of the arena, uniform occupancy
        rate_map = np.array([10., 0., 0., np.nan])
        px = np.array([.25, .25, .25, .25])
        rate = information_rate(rate_map, px)
        spec = information_specificity(rate_map, px)
        assert rate == 5  # bits/s
        assert spec == 2  # bits/spike
