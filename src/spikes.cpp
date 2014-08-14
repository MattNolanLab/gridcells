
#include <armadillo>
#include "spikes.hpp"

using arma::vec;

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

} // namespace grids
