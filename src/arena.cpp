
#include "arena.hpp"

namespace grids 
{

Discretisation2D::Discretisation2D(const Size2D& sz, const XYPair<double> q) :
        size(sz), q(q),
        _nx(size.x / q.x + 1),
        _ny(size.y / q.y + 1),
        _edges(arma::linspace(-int(size.x) / 2., size.x / 2., _nx),
               arma::linspace(-int(size.y) / 2., size.y / 2., _ny))
{
}


} // namespace grids
