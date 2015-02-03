/*
 *   signal.hpp
 *
 *   Signal analysis python independent functions.
 */

#ifndef SIGNAL_HPP
#define SIGNAL_HPP

#include <armadillo>

namespace grids {

/**
 * Lag-restricted correlation function. 
 *
 * Compute the correlation function (CF) of two vectors (1D Arrays), with
 * lags that are user-defined.
 *
 * @param v1        First vector
 * @param v2        Second vector
 * @param lag_start Starting lag value
 * @param lag_en    End lag value (included in the CF)
 * @return A new 1D array with size lag_end - lag_start + 1
 *
 * @note It is the responsibility of the user to pass correct values of
 *       lag_start and lag_end as these are not tested for boundaries.
 * @note The result for both arrays being empty is undefined.
 */
arma::vec*
correlation_function(const arma::vec &v1,
                     const arma::vec &v2,
                     int lag_start,
                     int lag_end);


} // namespace grids

#endif // SIGNAL_HPP
