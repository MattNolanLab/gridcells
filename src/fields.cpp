
#include <cmath>

#include "fields.hpp"

using arma::vec;
using arma::mat;
using arma::uword;

namespace grids
{

vec
extractSpikePos(const vec& spikePosIdx, const vec& posData)
{
    vec res(spikePosIdx.n_elem);
    for (int i = 0; i < spikePosIdx.n_elem; i++) {
        if (spikePosIdx(i) >= posData.n_elem || spikePosIdx(i) < 0) {
            res(i) = NAN;
        } else {
            res(i) = posData(floor(spikePosIdx(i)));
        }
    }
    return res;
}


double nansum(const arma::vec& v)
{
    double sum = 0;
    for (size_t i = 0; i < v.n_elem; i++) {
        double elem = v(i);
        if (!isnan(elem)) {
            sum += elem;
        }
    }

    return sum;
}


double trapz(const vec& f)
{
    double partialSum = 0;
    int starti = 0;
    int endi   = f.n_elem - 1;
    int nansz  = -1;

    if (f.n_elem == 0)
        return NAN;

    // Correct for NaNs
    while (isnan(f(starti)) && starti < f.n_elem) {
        starti++;
    }
    while (isnan(f(endi)) && endi >= 0) {
        endi--;
    }
    if (starti < endi) {
        nansz = endi - starti + 1;
    } else {
        return NAN;
    }
    
    if (nansz > 2) {
        partialSum = nansum(f.subvec(starti+1, endi-1));
    }

    return .5 * (f(starti) + f(endi)) + partialSum;
}


mat*
spatialRateMap(const vec& spikeTimes,
               const vec& posX,
               const vec& posY,
               double pos_dt,
               const vec& xedges,
               const vec& yedges,
               double sigma)
{
    mat* rateMap = new mat(xedges.n_elem, yedges.n_elem);
    rateMap->zeros();
    vec spikePosIdx = spikeTimes / pos_dt;
    vec neuronPos_x = extractSpikePos(spikePosIdx, posX);
    vec neuronPos_y = extractSpikePos(spikePosIdx, posY);


    int nIter = 0;
    for (int x_i = 0; x_i < xedges.n_elem; x_i++) {
        for (int y_i =0; y_i < yedges.n_elem; y_i++) {
            double x = xedges(x_i);
            double y = yedges(y_i);

            vec posDist2 = arma::square(posX - x) + arma::square(posY - y);
            bool isNearTrack = arma::accu(arma::sqrt(posDist2) <= sigma) > 0;
            //std::cout << "isNearTrack: " << isNearTrack << std::endl;

            if (isNearTrack) {
                double normConst = trapz(arma::exp(-posDist2 / 2. / (sigma*sigma))) * pos_dt;
                //std::cout << "normConst: " << normConst << std::endl;
                vec neuronPosDist2 = arma::square(neuronPos_x - x) + 
                                     arma::square(neuronPos_y - y);
                double spikes = nansum(exp( -neuronPosDist2 / 2. / (sigma*sigma) ));
                //std::cout << "spikes: " << spikes << std::endl;
                (*rateMap)(x_i, y_i) = spikes / normConst;
            } else {
                (*rateMap)(x_i, y_i) = NAN;
            }

            nIter++;
        }
    }

    //rateMap->print();
    return rateMap;
}

} // namespace grids
