
#ifndef GRIDSCORE_HPP
#define GRIDSCORE_HPP

#include <armadillo>

namespace grids
{


/**
 * A class template specifying a pair of x and y data members.
 *
 * The members can only be specified in the constructor and remain constant.
 */
template <typename T>
struct XYPair
{
    const T x; ///< X data member
    const T y; ///< Y data member

    /**
     * Construct the object.
     */
    XYPair(const T& x, const T& y) : x(x), y(y) {};

    /** Copy constructor **/
    XYPair(const XYPair<T>& other) : x(other.x), y(other.y) {};
};


typedef XYPair<arma::vec> VecPair; ///< A pair of armadillo vectors.
typedef XYPair<unsigned> Size2D;   ///< A pair specifying a 2D size.



/* Necessary for correct SWIG processing. Do not remove. */
#ifdef SWIG
    %template(VecPair) XYPair<arma::vec>;
    %template(Size2D) XYPair<unsigned>;
#endif //SWIG



/**
 * Specifies positional information with a constant time step.
 */
struct Position2D : public VecPair
{
    const double dt; ///< Positional time step. Arbitrary units.

    /**
     * Construct the positional information.
     *
     * All data will be read-only.
     */
    Position2D(const arma::vec& x, const arma::vec& y, const double dt) :
        VecPair(x, y), dt(dt) {};

    /** Copy constructor. */
    Position2D(const Position2D& other) :
        VecPair(other.x, other.y), dt(other.dt) {};
};


} // namespace grids

#endif // GRIDSCORE_HPP
