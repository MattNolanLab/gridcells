
#include "fields.hpp"

const double dt = 20.;
const double arenaDiam = 180.;
const double h = 3.;


int main(void)
{
    vec spikeTimes;
    vec pos_x;
    vec pos_y;

    bool loadOK = true;
    loadOK &= spikeTimes.load("data/spikeTimes.txt", arma::raw_ascii);
    loadOK &= pos_x.load("data/pos_x.txt", arma::raw_ascii);
    loadOK &= pos_y.load("data/pos_y.txt", arma::raw_ascii);

    if (!loadOK){
        cerr << "Could not load data files!" << endl;
        return 1;
    }

    mat rateMap = SNSpatialRate2D(spikeTimes, pos_x, pos_y, dt, arenaDiam, h);
    (rateMap * 1e3).print();

    return 0;
}
