#ifndef FIELDS_HPP
#define FIELDS_HPP

#include <armadillo>

#include "gridsCore.hpp"

namespace grids
{


/**
 * Compute a spatial firing rate in a 2D environment.
 */
arma::mat*
spatialRateMap(const arma::vec& spikeTimes,
               const Position2D& pos,
               const arma::vec& xedges, const arma::vec& yedges,
               double sigma);


} // namespace grids

#endif // FIELDS_HPP
