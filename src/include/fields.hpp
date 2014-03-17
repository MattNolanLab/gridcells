#ifndef FIELDS_HPP
#define FIELDS_HPP

#include <armadillo>

namespace grids
{

arma::vec
extractSpikePos(const arma::vec& spikePosIdx, const arma::vec& posData, double dt);

arma::mat
SNSpatialRate2D(const arma::vec& spikeTimes, const arma::vec& pos_x, const
        arma::vec& pos_y, double dt, double arenaDiam, double h);

} // namespace grids

#endif // FIELDS_HPP
