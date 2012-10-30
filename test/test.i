
%module armanpytest
%{
#define SWIG_FILE_WITH_INIT

/* Includes the header in the wrapper code */
#include "test.hpp"
%}

/* We need thos for boost_shared::ptr support */
%include <boost_shared_ptr.i>

/* Now include ArmaNpy typemaps */
%include "armanpy.i"

/* Some minimal excpetion handling */
%exception {
    try {
        $action
    } catch( char * str ) {
        PyErr_SetString( PyExc_IndexError, str );
        SWIG_fail;
    } 
}

/* Parse the header file to generate wrappers */
%include "test.hpp"

/* Instantiate some of the classes */
%template() MatTestClass< arma::Mat< double > >;
%template() MatTestClass< arma::Mat< float > >;

%template() RowColTestClass< arma::Col<double> >;
%template() RowColTestClass< arma::Col<float> >;

%template() RowColTestClass< arma::Row<double> >;
%template() RowColTestClass< arma::Row<float> >;

%template(CubeTestDouble) CubeTestClass< arma::Cube<double> >;
%template(CubeTestFloat)  CubeTestClass< arma::Cube<float> >;

%template(MatTestDouble) MatTestClass< arma::Mat<double> >;
%template(MatTestFloat)  MatTestClass< arma::Mat<float> >;

%template(MatTestComplexDouble) CxMatTestClass< arma::Mat< std::complex< double > > >;
%template(MatTestComplexFloat)  CxMatTestClass< arma::Mat< std::complex< float > > >;

%template(ColTestDouble) RowColTestClass< arma::Col<double> >;
%template(ColTestFloat)  RowColTestClass< arma::Col<float>  >;
%template(ColTestInt)    RowColTestClass< arma::Col<int>  >;
%template(ColTestUWord)  RowColTestClass< arma::Col<arma::uword>  >;

%template(RowTestDouble) RowColTestClass< arma::Row<double> >;
%template(RowTestFloat)  RowColTestClass< arma::Row<float>  >;
%template(RowTestInt)    RowColTestClass< arma::Row<int>  >;
%template(RowTestUWord)  RowColTestClass< arma::Row<arma::uword>  >;



