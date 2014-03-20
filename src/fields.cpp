#include "fields.hpp"


namespace grids
{

arma::vec
extractSpikePos(const arma::vec& spikePosIdx, const arma::vec& posData, double dt)
{
    arma::vec res(spikePosIdx.n_elem);
    for (int i = 0; i < spikePosIdx.n_elem; i++) {
        res(i) = posData(spikePosIdx(i));
    }
    return res;
}


arma::mat*
SNSpatialRate2D(const arma::vec& spikeTimes, const arma::vec& pos_x, const
        arma::vec& pos_y, double dt, double arenaDiam, double h)
{
    double precision = arenaDiam / h;
    arma::vec xedges = arma::linspace(-arenaDiam/2, arenaDiam/2, precision+1);
    arma::vec yedges = arma::linspace(-arenaDiam/2, arenaDiam/2, precision+1);

    arma::mat* rateMap = new arma::mat(xedges.n_elem, yedges.n_elem);
    rateMap->zeros();
    arma::vec spikePosIdx = spikeTimes / dt;
    arma::vec neuronPos_x = extractSpikePos(spikePosIdx, pos_x, dt);
    arma::vec neuronPos_y = extractSpikePos(spikePosIdx, pos_y, dt);


    int nIter = 0;
    for (int x_i = 0; x_i < xedges.n_elem; x_i++) {
        for (int y_i =0; y_i < yedges.n_elem; y_i++) {
            //std::cout << "nIter: " << nIter << std::endl;
            double x = xedges(x_i);
            double y = yedges(y_i);

            arma::vec posDist2 = arma::square(pos_x - x) +
                                 arma::square(pos_y - y);
            bool isNearTrack = arma::accu(arma::sqrt(posDist2) <= h) > 0;

            if (isNearTrack) {
                double normConst = arma::sum(arma::exp(-posDist2 / 2. / (h*h))) * dt;
                arma::vec neuronPosDist2 = arma::square(neuronPos_x - x) + 
                                     arma::square(neuronPos_y - y);
                double spikes = arma::sum(exp( -neuronPosDist2 / 2. / (h*h)));
                (*rateMap)(x_i, y_i) = spikes / normConst;
            }

            nIter++;
        }
    }

    return rateMap;
}

} // namespace grids
