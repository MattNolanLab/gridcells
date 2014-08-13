#include <cassert>

#include "common.hpp"

using arma::vec;

#define SQ(x) ((x) * (x))
#define MIN(x1, x2) ((x1) < (x2) ? (x1) : (x2))

namespace grids
{

vec* twisted_torus_distance(double a_x, 
                            double a_y,
                            const vec& others_x,
                            const vec& others_y,
                            double dim_x,
                            double dim_y)
{
    assert(others_x.n_elem == others_y.n_elem);

    vec* distance = new vec(others_x.n_elem);
    a_x = divisor_mod(a_x, dim_x);
    a_y = divisor_mod(a_y, dim_y);

    for (unsigned i = 0; i < others_x.n_elem; i++)
    {
        double o_x = divisor_mod(others_x(i), dim_x);
        double o_y = divisor_mod(others_y(i), dim_y);

        double d1 = sqrt(SQ(a_x - o_x            ) + SQ(a_y - o_y        ));
        double d2 = sqrt(SQ(a_x - o_x - dim_x    ) + SQ(a_y - o_y        ));
        double d3 = sqrt(SQ(a_x - o_x + dim_x    ) + SQ(a_y - o_y        ));
        double d4 = sqrt(SQ(a_x - o_x + 0.5*dim_x) + SQ(a_y - o_y - dim_y));
        double d5 = sqrt(SQ(a_x - o_x - 0.5*dim_x) + SQ(a_y - o_y - dim_y));
        double d6 = sqrt(SQ(a_x - o_x + 0.5*dim_x) + SQ(a_y - o_y + dim_y));
        double d7 = sqrt(SQ(a_x - o_x - 0.5*dim_x) + SQ(a_y - o_y + dim_y));

        (*distance)(i) = MIN(d7, MIN(d6, MIN(d5, MIN(d4, MIN(d3, MIN(d2, d1))))));
    }

    return distance;
}


} // namespace grids
