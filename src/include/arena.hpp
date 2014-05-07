#ifndef arena_HPP
#define arena_HPP

#include <armadillo>

#include "gridsCore.hpp"

namespace grids {


/**
 * Discretisation/quantisation of a 2D rectangular object (e.g. an arena).
 *
 * Use this class to define how to discretise your arena.
 */
class Discretisation2D
{
    const Size2D size;        ///< Size of the discretised object
    const XYPair<double> q; ///< Discretisation interval, i.e. dx/dy
    unsigned _nx;             ///< Number of items in X direction
    unsigned _ny;             ///< Number of items in Y direction
    VecPair _edges;           ///< Edges, i.e. discretisation

  public:
    /**
     * Construct the object, given the size of the arena and quantisation
     * factor (dx, dy).
     *
     * @param sz Size of the arena.
     * @param q  Quantisation 
     */
    Discretisation2D(const Size2D& sz, const XYPair<double> q);

    const VecPair& edges() const { return _edges; };

    const Size2D& getSize() const { return size; };
    const XYPair<double> getQ() const { return q; };
    const size_t nX() const { return _nx; };
    const size_t nY() const { return _ny; };
};



/**
 * A 2D arena representing the space where an animal moves in the
 * simulation/experiment.
 *
 * This abstract class declares two pure virtual functions that the owner can
 * query in order to determine the parameters of the arena.
 */
class Arena
{
  public:
    /** Get the specified arena discretisation. **/
    virtual const Discretisation2D& getDiscretisation() const = 0;

    /**
     * Get the mask of the arena. Where the mask is \c true, data is invalid.
     */ 
    virtual const arma::umat& getMask() const = 0;


    virtual ~Arena() {};
};



/**
 * A rectangular arena.
 *
 * This arena class is specified by its size and quantisation parameters, i.e.
 * dx/dy.
 */
class RectangularArena : public Arena
{
  protected:
    const Size2D size;              ///< Size of the arena
    const Discretisation2D discret; ///< Arena's discretisation
    arma::umat mask;                ///< Arena's mask

  public:
    /**
     * Create the arena given size and quantisation step.
     *
     * @param sz Size of the arena.
     * @param q  Quantisation step, i.e. dx and dy.
     */
    RectangularArena(const Size2D& sz, const XYPair<double>& q) :
        size(sz), discret(Discretisation2D(sz, q)),
        mask(arma::umat(discret.nX(), discret.nY()).fill(false))
    {}

    //RectangularArena(const Size2D& sz, const Discretisation2D& d) :
    //    size(sz), discret(d);

    
    /** Obtain the arena's discretisation **/
    const Discretisation2D& getDiscretisation() const { return discret; };

    /** Obtain the mask for this arena **/
    const arma::umat& getMask() const { return mask; };

    const Size2D& getSize() const { return size; };
};


/**
 * A square arena. A special case of \c RectangularArena.
 */
class SquareArena : public RectangularArena
{
  public:
    SquareArena(double sz, const XYPair<double>& q) :
        RectangularArena(Size2D(sz, sz), q) {}
};



/** 
 * A circular arena.
 *
 * This is a special case of \c SquareArena in that the movement is restricted
 * to the arena's radius. However, the whole space is represented as a square,
 * in which all the values outside the circle are invalid.
 */
class CircularArena : public SquareArena
{
  private:
    double r; ///< Arena radius

  public:
    /**
     * Create the arena.
     *
     * @param r Radius
     * @param q Quantisation step, i.e. dx/dy
     */
    CircularArena(double r, const XYPair<double>& q) :
        SquareArena(r * 2., q), r(r) 
    {
        arma::mat X = arma::repmat(discret.edges().x.t(), discret.nY(),            1);
        arma::mat Y = arma::repmat(discret.edges().y,                1, discret.nX());
        mask = arma::sqrt(arma::square(X) + arma::square(Y)) > r;
    }

    double getR() const { return r; }; ///< Radius getter
};


} // namespace grids

#endif // arena_HPP
