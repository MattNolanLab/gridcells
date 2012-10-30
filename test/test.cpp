
#include "test.hpp"

template class CubeTestClass< arma::Cube<double> >; 
template class CubeTestClass< arma::Cube<float> >; 

template class MatTestClass< arma::Mat<double> >; 
template class MatTestClass< arma::Mat<float> >; 

template class CxMatTestClass< arma::Mat< std::complex<double> > >; 
template class CxMatTestClass< arma::Mat< std::complex<float> > >; 

template class RowColTestClass< arma::Col<double> >;
template class RowColTestClass< arma::Col<float>  >;
template class RowColTestClass< arma::Col<arma::uword>  >;
template class RowColTestClass< arma::Col<int>  >;

template class RowColTestClass< arma::Row<double> >;
template class RowColTestClass< arma::Row<float>  >;
template class RowColTestClass< arma::Row<arma::uword>  >;
template class RowColTestClass< arma::Row<int>  >;
