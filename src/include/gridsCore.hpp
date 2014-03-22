
#ifndef GRIDCELLS_H
#define GRIDCELLS_H

#include <armadillo>

namespace grids
{


class Position2D
{
  public:
    const arma::vec x;
    const arma::vec y;
    const double dt;

    Position2D(const arma::vec x, const arma::vec y, const double dt) :
        x(x), y(y), dt(dt) {};
};


} // namespace grids

#endif // GRIDCELLS_H
