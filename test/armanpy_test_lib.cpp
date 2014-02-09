// Copyright (C) 2012 thomas.natschlaeger@gmail.com
//
// This file is part of the ArmaNpy library.
// It is provided without any warranty of fitness
// for any purpose. You can redistribute this file
// and/or modify it under the terms of the GNU
// Lesser General Public License (LGPL) as published
// by the Free Software Foundation, either version 3
// of the License or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#include "armanpy_test_lib.hpp"

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
