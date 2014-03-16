#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

const int NITER = 3720;
const int SZ = 60000;
const double sigma = 1.3;


int main(int argc, char** argv)
{


    vec d2 = randu(SZ);
    vec res(SZ);

    for (int it = 0; it < NITER; it++) {
        res = arma::exp(-d2 / 2.0 / (sigma*sigma));
        //cout << it << endl;
    }

    
    cout << "res[0]: " << res(0) << endl;

    return 0;

}
