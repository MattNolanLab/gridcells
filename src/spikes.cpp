
#include <armadillo>
#include "spikes.hpp"

using arma::Col;
using arma::vec;
using arma::mat;
using arma::linspace;

namespace grids
{

vec* spike_time_diff(const vec &train1, const vec &train2)
{
    int sz1 = train1.n_elem;
    int sz2 = train2.n_elem;

    int szRes = sz1 * sz2;
    vec* res = new vec(szRes);
    int resIdx = 0;
    for (int tIdx1 = 0; tIdx1 < sz1; tIdx1++) {
        for (int tIdx2 = 0; tIdx2 < sz2; tIdx2++) {
            (*res)(resIdx) = train2(tIdx2) - train1(tIdx1);
            resIdx++;
        }
    }

    return res;
}


/**
 * Compute a sliding window firing rate over a population of neurons.
 *
 * All times are in ms, firing rate is in Hz.
 *
 * @param n_ids A vector of neuron IDs that emitted the spikes.
 * @param spike_t A vector of spike times.
 * @param n_neurons Total number of neurons in the population
 * @param tstart Start time
 * @param tend End time
 * @param dt Window time step
 * @param win_len Window length
 * @return A matrix of firing rates. Size of fr is (n_neurons,
 *         int((tend-tstart) / dt)) + 1).
 */
mat*
sliding_firing_rate_base(const Col<long> &n_ids,
                         const vec &spike_t,
                         unsigned n_neurons,
                         double tstart,
                         double tend,
                         double dt,
                         double win_len)
{
    unsigned sz_rate = static_cast<unsigned>((tend - tstart) / dt) + 1;
    mat* bit_spikes = new mat(n_neurons, sz_rate);
    mat* fr = new mat(n_neurons, sz_rate);
    unsigned dt_wlen = static_cast<unsigned>(win_len / dt);

    bit_spikes->zeros();

    for (int i = 0; i < spike_t.n_elem; i++)
    {
        int spikeSteps = (spike_t(i) - tstart) / dt;
        if (spikeSteps >= 0 && spikeSteps < sz_rate)
        {
            int n_id = n_ids(i);
            (*bit_spikes)(n_id, spikeSteps) += 1;
        }
    }

    for (int n_id = 0; n_id < n_neurons; n_id++)
    {
        for (int t = 0; t < sz_rate; t++)
        {
            std::cout << "n_id: " << n_id << ", t: " << t << std::endl;
            std::cout << "dt_wlen: " << dt_wlen << std::endl;
            (*fr)(n_id, t) = .0;
            for (int s = 0; s < dt_wlen; s++) {
                if ((t+s) < sz_rate) {
                    std::cout << "s: " << s << std::endl;
                    (*fr)(n_id, t) += (*bit_spikes)(n_id, t+s);
                }
            }
        }
    }

    *fr /= win_len * 1e-3;  // msecond --> Hz
    delete bit_spikes;
    return fr;
}


/**
 * Return a vector of times corresponding to the calculations of the sliding
 * firing rate.
 *
 * Starts at tstart and ends at tend.
 *
 * @param tstart Start time
 * @param tend End time
 * @param dt Window time step
 * @return A vector of firing rate times, the size will be
 *         int((tend-tstart) / dt)) + 1).
 */
vec sliding_times(double tstart, double tend, double dt)
{
    unsigned sz = static_cast<unsigned>((tend - tstart) / dt) + 1;
    return linspace<vec>(tstart, tend, sz);
}



/**
 * Compute an average firing rate for each neuron in the population.
 *
 * @param n_ids Neurons IDs
 * @param spike_t Spike times.
 * @param n_neurons Total number of neurons in the population.
 * @param tstart Start time
 * @param tend End time.
 * @return An array of firing rates for all neurons. In units specified by the
 *         time parameters.
 */
vec* avg_fr(const Col<long> &n_ids,
            const vec &spike_t,
            unsigned n_neurons,
            double tstart,
            double tend)
{
    vec* rates = new vec(n_neurons);
    rates->zeros();

    for (int i = 0; i < n_ids.size(); i++)
    {
        int t = spike_t(i);
        int s = n_ids(i);
        if (s >= 0 && s < n_neurons && t >= tstart && t <= tend) {
            (*rates)(s)++;
        } else if (s < 0 || s >= n_neurons) {
            std::cout << "n_ids is outside range <0, n)" << std::endl;
        }
    }

    (*rates) /= (tend - tstart);
    return rates;
}

} // namespace grids
