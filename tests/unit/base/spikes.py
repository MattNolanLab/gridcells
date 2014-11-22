'''Spike testing common code.'''
from __future__ import absolute_import, print_function, division

import numpy as np

from gridcells.analysis.spikes import PopulationSpikes


class RateDefinedSpikeGenerator(object):
    '''A spike generator that emits spikes according to an array of rates.'''

    class PopulationRateTestItem(object):
        '''Test and input vectors for rate testing.'''
        def __init__(self, pop, orig_nspikes, test_rates, tend):
            self.pop = pop
            self.orig_nspikes = orig_nspikes
            self.test_rates = test_rates
            self.tend = tend

    def __init__(self, tstart, dt, winlen):
        self.tstart = tstart
        self.dt = dt
        self.winlen = winlen

    def get_test_vector(self, nspikes):
        '''Generate test vector.

        Generates a test vector in which every time step spikes are generated
        according to ``nspikes``.

        All times are in units of ms, except firing rates, which are in Hz.
        '''
        nspikes = np.asarray(nspikes)
        nneurons, nt = nspikes.shape
        spikes = np.array([], dtype=np.float)
        senders = np.array([], dtype=np.int)

        # Generate spikes and senders
        t = self.tstart
        for t_i in range(nt):
            for n_i in range(nneurons):
                sample = t + np.random.rand(nspikes[n_i, t_i]) * self.dt
                spikes = np.hstack((spikes, sample))
                senders = np.hstack((senders, [n_i] * len(sample)))
            t += self.dt
        to_sort = np.array(list(zip(senders, spikes)), dtype=[('senders', 'f8'),
                                                        ('spikes', 'f8')])
        sorted_spikes = np.sort(to_sort, order='spikes')
        pop = PopulationSpikes(nneurons, sorted_spikes['senders'],
                               sorted_spikes['spikes'])

        # Adjust nspikes to accomodate for winlen
        test_rates = np.array(nspikes, dtype=np.float, copy=True)
        ndt_winlen = int(self.winlen / self.dt)
        for t_i in range(nt):
            test_rates[:, t_i] = np.sum(nspikes[:, t_i:t_i + ndt_winlen],
                                        axis=1) / self.winlen

        tend = self.tstart + (nt - 1) * self.dt
        test_rates *= 1e3  # Adjust timing to Hz
        return self.PopulationRateTestItem(pop, nspikes, test_rates, tend)


