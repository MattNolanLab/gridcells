#include <armadillo>
#include <cmath>

#include "signal.hpp"

using arma::vec;
using arma::dot;
using arma::span;

namespace grids {

vec*
correlation_function(const vec &v1, const vec &v2, int lag_start, int lag_end)
{
    int sz1 = v1.n_elem;
    int sz2 = v2.n_elem;
    int szRes = lag_end - lag_start + 1;
    if (lag_start <= -sz1 || lag_end >= sz2) {
        throw std::exception();
    }
    vec* res = new vec(szRes);

    int i = 0;
    for (int lag = lag_start; lag <= lag_end; lag++) {
        int s1 = std::max(0, -lag);
        int e1 = std::min(sz1 - 1, sz2 - lag - 1);
        (*res)(i) = dot(v1(span(s1, e1)),
                        v2(span(s1 + lag, e1 + lag)));
        i++;
    }

    return res;
}


} // namespace grids
