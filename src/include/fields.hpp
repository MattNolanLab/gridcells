/** @file fields.hpp
 *
 * Grid field / autocorrelation / other calculations.
 *
 */

#ifndef FIELDS_HPP
#define FIELDS_HPP

#include <armadillo>

namespace grids
{


/**
 * Compute a spatial firing rate of a neuron.
 *
 * @param spikeTimes A vector of spike times of the neuron. Units must be the
 *          same as for pos.dt.
 * @param pos Positional data, i.e. where the subject was located at which
 *          time.
 * @param arena Arena in which the experiment happened
 * @param sigma  Smoothing factor (i.e. std. dev. of the smoothing Gaussian
 *          function).
 * @returns A 2D array containing the firing rate in time units specified by
 *          the user data.
 */
arma::mat*
spatialRateMap(const arma::vec& spikeTimes,
               const arma::vec& posX,
               const arma::vec& posY,
               double pos_dt,
               const arma::vec& xedges,
               const arma::vec& yedges,
               double sigma);

arma::vec
extractSpikePos(const arma::vec& spikePosIdx,
                const arma::vec& posData);

} // namespace grids

#endif // FIELDS_HPP
