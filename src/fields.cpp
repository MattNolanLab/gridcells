
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
        if (i >= posData.n_elem) {
            std::cerr << i << " >= " << posData.n_elem << std::endl;
        }
        res(i) = posData(spikePosIdx(i));
    }
    return res;
}


double trapz(const vec& f)
{
    double partialSum = 0;
    uword sz = f.n_elem;
    
    if (sz > 2) {
        partialSum = arma::sum(f.subvec(1, sz-2));
    }

    return .5 * (f(0) + f(sz-1)) + partialSum;
}


mat*
spatialRateMap(const vec& spikeTimes,
               const Position2D& pos,
               const Arena& arena,
               double sigma)
{
    const VecPair& edges = arena.getDiscretisation().edges();
    mat* rateMap = new mat(edges.x.n_elem, edges.y.n_elem);
    rateMap->zeros();
    vec spikePosIdx = spikeTimes / pos.dt;
    vec neuronPos_x = extractSpikePos(spikePosIdx, pos.x);
    vec neuronPos_y = extractSpikePos(spikePosIdx, pos.y);


    int nIter = 0;
    for (int x_i = 0; x_i < edges.x.n_elem; x_i++) {
        for (int y_i =0; y_i < edges.y.n_elem; y_i++) {
            //std::cout << "nIter: " << nIter << std::endl;
            double x = edges.x(x_i);
            double y = edges.y(y_i);

            vec posDist2 = arma::square(pos.x - x) + arma::square(pos.y - y);
            bool isNearTrack = arma::accu(arma::sqrt(posDist2) <= sigma) > 0;

            if (isNearTrack) {
                double normConst = trapz(arma::exp(-posDist2 / 2. / (sigma*sigma))) * pos.dt;
                vec neuronPosDist2 = arma::square(neuronPos_x - x) + 
                                     arma::square(neuronPos_y - y);
                double spikes = arma::sum(exp( -neuronPosDist2 / 2. / (sigma*sigma)));
                (*rateMap)(x_i, y_i) = spikes / normConst;
            }

            nIter++;
        }
    }

    //rateMap->print();
    return rateMap;
}

} // namespace grids
