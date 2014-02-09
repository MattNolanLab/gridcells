# Copyright (C) 2012 thomas.natschlaeger@gmail.com
#
# This file is part of the ArmaNpy library.
# It is provided without any warranty of fitness
# for any purpose. You can redistribute this file
# and/or modify it under the terms of the GNU
# Lesser General Public License (LGPL) as published
# by the Free Software Foundation, either version 3
# of the License or (at your option) any later version.
# (see http://www.opensource.org/licenses for more info)

import sys, unittest
import warnings

import sys
sys.path.append( "./build/bin" )

import numpy as N

from armanpytest import *

class MatUnitTests:

    def init_m( self, m, n, order='F' ):
        M=N.zeros( (m,n), order=order, dtype=self.dtype )
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M[i,j]=i*10+j;
        return M

    def test_i(self):
        self.c.set_i( 2 )
        self.assertTrue( self.c.get_i() == 2 )

    def test_set_get(self):
        a = self.init_m( 6, 7 )
        self.c.set_m( a )
        m=self.c.get_m()
        self.assertTrue( N.all( a == m ) )

    def test_get_sptr(self):
        a = self.init_m( 6, 7 )
        self.c.set_m( a )
        m=self.c.get_m_sptr()
        # print "test_get_sptr", type(m), m
        self.assertTrue( N.all( a == m ) )
        m1 = m
        del m
        self.assertTrue( N.all( a == m1 ) )

    def test_get_sptr_and_use_as_input(self):
        a = self.init_m( 6, 7 )
        self.c.set_m( a )
        m1 = self.c.get_m_sptr()
        self.c.set_m( m1 * 2 )
        m2 = self.c.get_m_sptr()
        self.assertTrue( N.all(     a == m1 ) )
        self.assertTrue( N.all( 2 * a == m2 ) )

    def test_rnd(self):
        self.c.rnd_m(3)
        m = self.c.get_m()
        self.assertTrue( N.all( m.shape == (3,3) ) )

    def test_mod_nothing(self):
        m = self.init_m( 5, 4 )
        a = self.init_m( 5, 4 )
        self.c.mod_nothing( m )
        self.assertTrue( N.all( a == m ) )

    def test_mod_content(self):
        m = self.init_m( 5, 4 )
        self.c.mod_content( m )
        a = [ [ 0.00, 25, 25, 25], [ 25, 25, 25, 25], [ 25, 25, 25, 25], [ 25, 25, 25, 25], [ 25, 25, 25, 25 ] ];
        self.assertTrue( N.all( a == m ) )

    def test_mod_small_size(self):
        m = self.init_m( 5, 4 )
        self.c.mod_size( m, 3, 4 )
        a = [ [ 00, 01, 02, 03 ], [ 10, 11, 12, 13  ], [ 20, 21, 22, 23 ] ];
        self.assertTrue( N.all( a == m ) )

    def test_mode_large_size(self):
        """Change size such that arma allocs new memory. This is the challange."""
        m = self.init_m( 5, 4 )
        a = self.init_m( 7, 9 )
        self.c.mod_size( m, a.shape[0], a.shape[1] )
        self.assertTrue( N.all( a == m ) )

    def test_mode_large_size_mod(self):
        m = self.init_m( 5, 4 )
        a = self.init_m( 7, 9 )
        self.c.mod_size( m, a.shape[0], a.shape[1] )
        # check wether we can change content
        a[1,1] = m[1,1] = 27
        a[2,2] = m[2,2] = 89
        self.assertTrue( N.all( a == m ) )

    def test_mode_large_size_copy_resize(self):
        m = self.init_m( 5, 4 )
        a = self.init_m( 7, 9 )
        self.c.mod_size( m, a.shape[0], a.shape[1] )
        # check wether we can copy and modify size
        mc = N.array( m );
        mc.resize( (80,80) )
        self.assertTrue( N.all( mc.shape == (80,80) ) )

    def test_check_no_conversion_allowed(self):
        self.assertRaisesRegexp( Exception, "Array of type '.*' required. A 'list' was given.", self.c.set_m, [ [1,2,3], [4,5,6] ] )


class MatDoubleUnitTests( MatUnitTests, unittest.TestCase ):
    def setUp(self):
        self.dtype='float64'
        self.c = MatTestDouble( 2 )


class MatFloatUnitTests( MatUnitTests, unittest.TestCase ):
    def setUp(self):
        self.dtype='float32'
        self.c = MatTestFloat( 2 )


class RowColUnitTests:

    def init_m( self, p, order='C' ):
        M=N.zeros( ( p, ), order=order, dtype=self.dtype )
        for i in range(M.shape[0]):
                M[i]=i;
        return M

    def set_non_numpy_arra(self):
        a = [ 1, 2, 3, 4, 5 ]
        self.c.set( a )
        m=self.c.get()
        self.assertTrue( N.all( a == m ) )

    def test_set_get(self):
        a = self.init_m( 6 )
        self.c.set( a )
        m=self.c.get()
        self.assertTrue( N.all( a == m ) )

    def test_set_get_both_orders(self):
        a = self.init_m( 8, order='C' )
        self.c.set( a )
        m=self.c.get()
        self.assertTrue( N.all( a == m ) )
        a = self.init_m( 7, order='F' )
        self.c.set( a )
        m=self.c.get()
        self.assertTrue( N.all( a == m ) )

    def test_get_sptr(self):
        a = self.init_m( 6 )
        self.c.set( a )
        m=self.c.get_m_sptr()
        if not N.all( a == m ):
            print "a", type(a), a
            print "m", type(m), m
        self.assertTrue( N.all( a == m ) )

    def test_rnd(self):
        self.c.rnd(3)
        m = self.c.get()
        self.assertTrue( N.all( m.shape == (3,) ) )

    def test_mod_nothing(self):
        m = self.init_m( 5 )
        a = self.init_m( 5 )
        self.c.mod_nothing( m )
        self.assertTrue( N.all( a == m ) )

    def test_mod_content(self):
        m = self.init_m( 5 )
        self.c.mod_content( m )
        a = [ [ 0.00, 25, 25, 25, 25 ] ]
        self.assertTrue( N.all( a == m ) )

    def test_mod_small_size(self):
        m = self.init_m( 5 )
        self.c.mod_size( m, 4 )
        a = [ [ 00, 01, 02, 03 ] ];
        self.assertTrue( N.all( a == m ) )

    def test_mode_large_size(self):
        """Change size such that arma allocs new memory. This is the challange."""
        m = self.init_m( 5 )
        a = self.init_m( 70 )
        self.c.mod_size( m, a.shape[0] )
        self.assertTrue( N.all( a == m ) )

    def test_mode_large_size_mod(self):
        m = self.init_m( 5  )
        a = self.init_m( 70 )
        self.c.mod_size( m, a.shape[0] )
        # check wether we can change content
        a[1] = m[1] = 27
        a[2] = m[2] = 89
        self.assertTrue( N.all( a == m ) )

    def test_mode_large_size_copy_resize(self):
        m = self.init_m( 5  )
        a = self.init_m( 70 )
        self.c.mod_size( m, a.shape[0] )
        # check wether we can copy and modify size
        mc = N.array( m );
        mc.resize( (80,) )
        self.assertTrue( N.all( mc.shape == (80,) ) )

    def test_check_dimensions(self):
        data = N.array( [ [1,2,3] ], dtype=self.dtype )
        self.assertRaisesRegexp( Exception, "Array with 1 dimension required. A 2-dimensional array was given.", self.c.set, data )

    def test_check_no_conversion_allowed(self):
        self.assertRaisesRegexp( Exception, "Array of type '.*' required. A 'list' was given.", self.c.set, [ [1,2,3], [4,5,6] ] )


class ColDoubleUnitTests( RowColUnitTests, unittest.TestCase ):
    def setUp(self):
        self.dtype='float64'
        self.c = ColTestDouble( 2 )

class ColFloatUnitTests( RowColUnitTests, unittest.TestCase ):
    def setUp(self):
        self.dtype='float32'
        self.c = ColTestFloat( 2 )

class ColInt32UnitTests( RowColUnitTests, unittest.TestCase ):
    def setUp(self):
        self.dtype='int32'
        self.c = ColTestInt( 2 )

class ColUint32UnitTests( RowColUnitTests, unittest.TestCase ):
    def setUp(self):
        self.dtype='uint32'
        self.c = ColTestUWord( 2 )

class RowDoubleUnitTests( RowColUnitTests, unittest.TestCase ):
    def setUp(self):
        self.dtype='float64'
        self.c = RowTestDouble( 2 )

class RowFloatUnitTests( RowColUnitTests, unittest.TestCase ):
    def setUp(self):
        self.dtype='float32'
        self.c = RowTestFloat( 2 )

class RowIntUnitTests( RowColUnitTests, unittest.TestCase ):
    def setUp(self):
        self.dtype='int32'
        self.c = RowTestInt( 2 )

class RowUintUnitTests( RowColUnitTests, unittest.TestCase ):
    def setUp(self):
        self.dtype='uint32'
        self.c = RowTestUWord( 2 )

class CubeUnitTests:

    def init_m( self, m, n, s, order='F' ):
        M=N.zeros( (m,n,s), order=order, dtype=self.dtype )
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                for k in range(M.shape[2]):
                    M[i,j,k]=i*100+j*10+k;
        return M

    def test_set_get(self):
        a = self.init_m( 3, 4, 3 )
        self.c.set_m( a )
        m=self.c.get_m()
        self.assertTrue( N.all( a == m ) )

    def test_set_get_sptr(self):
        a = self.init_m( 3, 4, 3 )
        self.c.set_m( a )
        m=self.c.get_m_sptr()
        self.assertTrue( N.all( a == m ) )

    def test_rnd(self):
        self.c.rnd_m(3)
        m = self.c.get_m()
        self.assertTrue( N.all( m.shape == (3,3,3) ) )

    def test_mod_nothing(self):
        m = self.init_m( 5, 4, 7 )
        a = self.init_m( 5, 4, 7 )
        self.c.mod_nothing( m )
        self.assertTrue( N.all( a == m ) )

    def test_mod_content(self):
        m = self.init_m( 5, 4, 2 )
        self.c.mod_content( m )
        a = [ 25., 25. ]
        a = [ a, a, a, a ]
        a = [ a, a, a, a, a ]
        a = N.array( a, order='F' )
        a[0,0,0]=0.0
        self.assertTrue( N.all( a == m ) )

    def test_mod_small_size(self):
        m = self.init_m( 4, 2, 2 )
        self.c.mod_size( m, 2, 2, 2 )
        a = N.array( [ [ [000., 001.], [010., 011.] ], [ [100., 101.], [110., 111.] ] ], order='F' )
        self.assertTrue( N.all( a == m ) )

    def test_mode_large_size(self):
        """Change size such that arma allocs new memory. This is the challange."""
        m = self.init_m( 5, 4, 9 )
        a = self.init_m( 7, 9, 3 )
        self.c.mod_size( m, a.shape[0], a.shape[1], a.shape[2] )
        self.assertTrue( N.all( a == m ) )

    def test_mode_large_size_mod(self):
        m = self.init_m( 5, 4, 9 )
        a = self.init_m( 7, 9, 3 )
        self.c.mod_size( m, a.shape[0], a.shape[1], a.shape[2] )
        #check wether we can change content
        a[1,1,2] = m[1,1,2] = 27
        a[2,2,1] = m[2,2,1] = 89
        self.assertTrue( N.all( a == m ) )

    def test_mode_large_size_copy_resize(self):
        m = self.init_m( 5, 4, 9 )
        a = self.init_m( 7, 9, 3 )
        self.c.mod_size( m, a.shape[0], a.shape[1], a.shape[2] )
        #check wether we can copy and modify size
        mc = N.array( m );
        mc.resize( (13,12,11) )
        self.assertTrue( N.all( mc.shape == (13,12,11) ) )

    def test_check_no_conversion_allowed(self):
        self.assertRaisesRegexp( Exception, "Array of type '.*' required. A 'list' was given.", self.c.set_m, [ [1,2,3], [4,5,6] ] )

    def test_check_dimension_assertion(self):
        data = N.array( [ [1,2,3], [4,5,6] ], dtype=self.dtype, order='C' )
        self.assertRaisesRegexp( Exception, "Array with 3 dimension required. A 2-dimensional array was given.", self.c.set_m, data )


class CubeDoubleTests( CubeUnitTests, unittest.TestCase ):
    def setUp(self):
        self.dtype='float64'
        self.c = CubeTestDouble( 2 )

class CubeFloatTests( CubeUnitTests, unittest.TestCase ):
    def setUp(self):
        self.dtype='float32'
        self.c = CubeTestFloat( 2 )

class MatComplexUnitTests:

    def init_m( self, m, n, order='F' ):
        M=N.zeros( (m,n), order=order, dtype=self.dtype )
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M[i,j]=N.complex(i*10+j,0.1*(i*10+j));
        return M

    def test_set_get(self):
        a = self.init_m( 6, 7 )
        self.c.set_m( a )
        m=self.c.get_m()
        self.assertTrue( N.all( a == m ) )

    def test_set_get_sptr(self):
        a = self.init_m( 6, 7 )
        self.c.set_m( a )
        m=self.c.get_m_sptr()
        self.assertTrue( N.all( a == m ) )

    def test_rnd(self):
        self.c.rnd_m(3)
        m = self.c.get_m()
        self.assertTrue( N.all( m.shape == (3,3) ) )

    def test_mod_nothing(self):
        m = self.init_m( 5, 4 )
        a = self.init_m( 5, 4 )
        self.c.mod_nothing( m )
        self.assertTrue( N.all( a == m ) )

    def test_mod_content(self):
        m = self.init_m( 5, 4 )
        self.c.mod_content( m )
        v = N.complex( 25, 35 )
        a = [ [ 0.00, v, v, v], [ v, v, v, v], [ v, v, v, v], [ v, v, v, v], [ v, v, v, v ] ];
        self.assertTrue( N.all( a == m ) )

    def test_mod_small_size(self):
        m = self.init_m( 5, 4 )
        self.c.mod_size( m, 3, 4 )
        c = N.complex
        a = [ [ c(0,0), c(1,0.1), c(2,0.2), c(3,0.3) ], [ c(10,1), c(11,1.1), c(12,1.2), c(13,1.3)  ], [ c(20,2.0), c(21,2.1), c(22,2.2), c(23,2.3) ] ];
        self.assertTrue( N.all( abs( a - m ) < 0.000001 ) )

    def test_mode_large_size(self):
        """Change size such that arma allocs new memory. This is the challange."""
        m = self.init_m( 5, 4 )
        a = self.init_m( 7, 9 )
        self.c.mod_size( m, a.shape[0], a.shape[1] )
        self.assertTrue( N.all( a == m ) )

    def test_mode_large_size_mod(self):
        m = self.init_m( 5, 4 )
        a = self.init_m( 7, 9 )
        self.c.mod_size( m, a.shape[0], a.shape[1] )
        #check wether we can change content
        a[1,1] = m[1,1] = N.complex(27,29)
        a[2,2] = m[2,2] = N.complex(89,34)
        self.assertTrue( N.all( a == m ) )

    def test_mode_large_size_copy_resize(self):
        m = self.init_m( 5, 4 )
        a = self.init_m( 7, 9 )
        self.c.mod_size( m, a.shape[0], a.shape[1] )
        #check wether we can copy and modify size
        mc = N.array( m );
        mc.resize( (80,80) )
        self.assertTrue( N.all( mc.shape == (80,80) ) )

    def test_check_no_conversion_allowed(self):
        self.assertRaisesRegexp( Exception, "Array of type '.*' required. A 'list' was given.", self.c.set_m, [ [1,2,3], [4,5,6] ] )


class MatComplexDoubleTests( MatComplexUnitTests, unittest.TestCase ):
    def setUp(self):
        self.dtype='complex128'
        self.c = MatTestComplexDouble( 2 )

class MatComplexFloatTests( MatComplexUnitTests, unittest.TestCase ):
    def setUp(self):
        self.dtype='complex64'
        self.c = MatTestComplexFloat( 2 )



class OverloadColTests( unittest.TestCase ):
    def setUp(self):
        pass


    def test_mul_scalar_ret_cx_default(self):
        x = N.array( [1.,2,3,4,5,4,3,2,1] )
        a = N.array( [1.,2,3,4,5,4,3,2,1] )

        y = mul_scalar_ret_cx( x, 2 )
        self.assertEqual( type(x), type(a) )
        self.assertEqual( x.dtype, a.dtype )

        self.assertEqual( type(y), type(a) )
        self.assertEqual( y.dtype, 'complex128' )
        self.assertTrue( N.all( y.real == x * 2 ) )

        self.assertEqual( type( dir( x ) ), type( ['a','b'] ) )

    def test_mul_scalar_ret_cx_32_2(self):
        x = N.array( [1.,2,3,4,5,4,3,2,1], dtype='float32' )
        a = N.array( [1.,2,3,4,5,4,3,2,1], dtype='float32' )

        y = mul_scalar_ret_cx( x, 2 )
        self.assertEqual( type(x), type(a) )
        self.assertEqual( x.dtype, a.dtype )

        self.assertEqual( type(y), type(a) )
        self.assertEqual( y.dtype, 'complex64' )
        self.assertTrue( N.all( y.real == x * 2 ) )

        self.assertEqual( type( dir( x ) ), type( ['a','b'] ) )

    def test_mul_scalar_ret_cx_64_2(self):
        x = N.array( [1.,2,3,4,5,4,3,2,1], dtype='float64' )
        a = N.array( [1.,2,3,4,5,4,3,2,1], dtype='float64' )

        y = mul_scalar_ret_cx( x, 2 )
        self.assertEqual( type(x), type(a) )
        self.assertEqual( x.dtype, a.dtype )

        self.assertEqual( type(y), type(a) )
        self.assertEqual( y.dtype, 'complex128' )
        self.assertTrue( N.all( y.real == x * 2 ) )

        self.assertEqual( type( dir( x ) ), type( ['a','b'] ) )

    def test_mul_scalar_ret_cx_32(self):
        x = N.array( [1.,2,3,4,5,4,3,2,1], dtype='float32' )
        a = N.array( [1.,2,3,4,5,4,3,2,1], dtype='float32' )

        y = mul_scalar_ret_cx( x )
        self.assertEqual( type(x), type(a) )
        self.assertEqual( x.dtype, a.dtype )

        self.assertEqual( type(y), type(a) )
        self.assertEqual( y.dtype, 'complex64' )
        self.assertTrue( N.all( y.real == x ) )

        self.assertEqual( type( dir( x ) ), type( ['a','b'] ) )

    def test_mul_scalar_ret_cx_64(self):
        x = N.array( [1.,2,3,4,5,4,3,2,1], dtype='float64' )
        a = N.array( [1.,2,3,4,5,4,3,2,1], dtype='float64' )

        y = mul_scalar_ret_cx( x )
        self.assertEqual( type(x), type(a) )
        self.assertEqual( x.dtype, a.dtype )

        self.assertEqual( type(y), type(a) )
        self.assertEqual( y.dtype, 'complex128' )
        self.assertTrue( N.all( y.real == x ) )

        self.assertEqual( type( dir( x ) ), type( ['a','b'] ) )


if __name__ == '__main__':
    unittest.main()
