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

#ifndef _TEST_HPP_
#define _TEST_HPP_

/* Cmake will define test_EXPORTS on Windows when it
configures to build a shared library. If you are going to use
another build system on windows or create the visual studio
projects by hand you need to define test_EXPORTS when
building a DLL on windows.
*/
// We are using the Visual Studio Compiler and building Shared libraries

#if defined (_WIN32)
	  #if defined(armanpy_test_lib_EXPORTS)
		#define DLLEXPORT __declspec(dllexport)
	  #else
		#define DLLEXPORT __declspec(dllimport)
	  #endif /* pcsim_EXPORTS */
#else /* defined (_WIN32) */
		#define DLLEXPORT
#endif

#ifndef SWIG
#undef ARMA_NO_DEBUG
#undef NDEBUG
#undef ARMA_EXTRA_DEBUG
#undef ARMA_DONT_PRINT_LOGIC_ERRORS
#define ARMA_DONT_PRINT_RUNTIME_ERRORS
#include <armadillo>
#include <boost/shared_ptr.hpp>
#else
namespace arma {
    template< typename T > class Mat { private: T *m; };
    template< typename T > class Row { private: T *m; };
    template< typename T > class Col { private: T *m; };
}
#endif

#pragma warning(disable:4251)

template< typename MatT >
class DLLEXPORT CubeTestClass {

public:
    CubeTestClass( int a ) : i(a) {
        m.randn(10,10,10);
    };

    MatT get_m(void) {
        return m;
    };

    void set_m( const MatT& m ) {
        this->m = m;
    };

    boost::shared_ptr< MatT > get_m_sptr(void) {
        boost::shared_ptr< MatT > p( new MatT( m ) );
        return p;
    };

    void rnd_m( unsigned s) {
        this->m.randn(s,s,s);
    };

    void mod_nothing( MatT& A ) {
    };

    void mod_content( MatT& A ) {
        A.randn( A.n_rows, A.n_cols, A.n_slices );
        A.fill(25);
        A(0,0,0)=0;
    };

    void mod_size( MatT& A, unsigned r, unsigned c, unsigned s ) {
        A.resize( r, c, s );
        A.randn( r, c, s );
        for( unsigned i=0; i<r; i++ ) {
            for( unsigned j=0; j<c; j++ ) {
                for( unsigned k=0; k<s; k++ ) {
                    A(i,j,k) = (typename MatT::elem_type)(i*100+j*10+k);
                }
            }
        }
    };

private:
    int i;
    MatT m;
};

template< typename MatT >
class DLLEXPORT MatTestClass {

public:
    MatTestClass( int a ) : i(a) {
        m.randn(10,10);
    };

    void set_i( int i ) {
        this->i = i;
    };

    int get_i(void) {
        return i;
    };

    MatT get_m(void) {
        return m;
    };

    boost::shared_ptr< MatT > get_m_sptr(void) {
        boost::shared_ptr< MatT > p( new MatT( m ) );
        return p;
    };

    void set_m( const MatT& m ) {
        this->m = m;
    };

    void rnd_m( unsigned s) {
        this->m.randn(s,s);
    };

    void mod_nothing( MatT& A ) {
    };

    void mod_content( MatT& A ) {
        A.randn( A.n_rows, A.n_cols );
        A.fill(25);
        A(0,0)=00;
    };

    void mod_size( MatT& A, unsigned r, unsigned c ) {
        A.resize( r, c );
        A.randn( r, c );
        for( unsigned i=0; i<r; i++ ) {
            for( unsigned j=0; j<c; j++ ) {
                A(i,j) = (typename MatT::elem_type)(i*10+j);
            }
        }
    };

private:
    int i;
    MatT m;
};

template< typename MatT >
class DLLEXPORT CxMatTestClass {

public:
    CxMatTestClass( int a ) : i(a) {
        m.randn(10,10);
    };

    void set_i( int i ) {
        this->i = i;
    };

    int get_i(void) {
        return i;
    };

    MatT get_m(void) {
        return m;
    };

    boost::shared_ptr< MatT > get_m_sptr(void) {
        boost::shared_ptr< MatT > p( new MatT( m ) );
        return p;
    };

    void set_m( const MatT& m ) {
        this->m = m;
    };

    void rnd_m( unsigned s) {
        this->m.randn(s,s);
    };

    void mod_nothing( MatT& A ) {
    };

    void mod_content( MatT& A ) {
        typedef typename MatT::elem_type eT;
        A.randn( A.n_rows, A.n_cols );
        A.fill( eT( 25, 35 ) );
        A(0,0)=00;
    };

    void mod_size( MatT& A, unsigned r, unsigned c )
    {
        typedef typename MatT::pod_type rT;
        typedef typename MatT::elem_type eT;
        A.resize( r, c );
        A.randn( r, c );
        for( unsigned i=0; i<r; i++ ) {
            for( unsigned j=0; j<c; j++ ) {
                rT re = rT(i*10+j);
                rT im = rT( 0.1 *( i*10+j ) );
                A(i,j) = eT( re, im );
            }
        }
    };

private:
    int i;
    MatT m;
};

template< typename MatT >
class DLLEXPORT RowColTestClass {

public:
    RowColTestClass( int a ) : i(a) {
        m.randn(10);
    };

    MatT get(void) {
        return m;
    };

    boost::shared_ptr< MatT > get_m_sptr(void) {
        boost::shared_ptr< MatT > p( new MatT( m ) );
        return p;
    };

    void set( const MatT& m ) {
        this->m = m;
    };

    void rnd( unsigned s) {
        this->m.randn(s);
    };

    void mod_nothing( MatT& A ) {
    };

    void mod_content( MatT& A ) {
        A.randn( A.n_rows, A.n_cols );
        A.fill(25);
        A(0,0)=00;
    };

    void mod_size( MatT& A, unsigned p ) {
        A.resize( p );
        A.randn( p );
        for( unsigned i=0; i<p; i++ ) {
            A(i) = (typename MatT::elem_type)i;
        }
    };

private:
    int i;
    MatT m;
};

boost::shared_ptr< arma::cx_vec > mul_scalar_ret_cx( arma::vec const& x, double opt = 1 )
{
    boost::shared_ptr< arma::cx_vec > r( new arma::cx_vec );
    *r = arma::conv_to< arma::cx_vec >::from( x * opt );
    return r;
};

boost::shared_ptr< arma::cx_fvec > mul_scalar_ret_cx( arma::fvec const& x, float opt = 1 )
{
    boost::shared_ptr< arma::cx_fvec > r( new arma::cx_fvec );
    *r = arma::conv_to< arma::cx_fvec >::from( x * opt );
    return r;
};


#endif
