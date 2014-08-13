#ifndef COMMON_HPP
#define COMMON_HPP

#include <armadillo>

namespace grids 
{


/**
 * Calculate distance from `a` to all elements in vectors `others`, on twisted
 * torus (TT) with dimensions `dim`.
 *
 * @param a_x Initial X coordinate of the point on the TT.
 * @param a_y Initial Y coordinate of the point on the TT.
 * @param others_x X coordinates of all other points on the TT.
 * @param others_y Y coordinates of all other points on the TT.
 * @param dim_x X dimension of the TT.
 * @param dim_y Y dimenison of the TT
 * @returns A vector of distances on the TT.
 */
arma::vec* twisted_torus_distance(double a_x, 
                                  double a_y,
                                  const arma::vec& others_x,
                                  const arma::vec& others_y,
                                  double dim_x,
                                  double dim_y);


/**
 * Modulo operation that has the same sign as divisor.
 */
inline double divisor_mod(double a, double b) {
    return a - floor(a/b)*b;
};


} // namespace grids

#endif // COMMON_HPP
