/*
 *   spikes.hpp
 *
 *   Spike analysis and manipulation
 *
 */

#ifndef SPIKES_H
#define SPIKES_H

#include <cmath>
#include <armadillo>


namespace grids {

/**
 * Compute the distribution of the spike time differences of all pairs of
 * spikes of two spike trains (1D arrays) (t2 - t1). The order thus matters: if
 * t1 precedes t2, then the result will be positive.
 *
 * @param train1 First spike train.
 * @param train2 Second spike train.
 * @return An array of spike time differences for each spike pair.
 */
arma::vec* spike_time_diff(const arma::vec &train1, const arma::vec &train2);

} // namespace grids

#endif // SPIKES_H
