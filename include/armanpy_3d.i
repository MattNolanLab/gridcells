
%fragment( "armanpy_cube_typemaps", "header", fragment="armanpy_typemaps" )
{

	template< typename MatT >
	bool armanpy_typecheck_cube_with_conversion( PyObject* input , int nd )
    {
		PyArrayObject* array=NULL;
		int is_new_object=0;
		if( armanpy_allow_conversion_flag ) {
			array = obj_to_array_fortran_allow_conversion( input, NumpyType< typename  MatT::elem_type >::val, &is_new_object );
			if ( !array || !require_dimensions( array, nd ) ) return false;
			return true;
		} else {
			array = obj_to_array_no_conversion( input, NumpyType< typename MatT::elem_type >::val );
			if( !array )                           return false;
			if( !require_dimensions( array, nd ) ) return false;
		    if( !array_is_fortran(array) )         return false;
			return true;
		}
	}
	
	template< typename MatT >
	PyArrayObject* armanpy_to_cube_with_conversion( PyObject* input , int nd )
    {
		PyArrayObject* array=NULL;
		int is_new_object=0;
		if( armanpy_allow_conversion_flag ) {
			array = obj_to_array_fortran_allow_conversion( input, NumpyType< typename MatT::elem_type >::val, &is_new_object );
			if ( !array || !require_dimensions( array, nd ) ) return NULL;
			if( armanpy_warn_on_conversion_flag && is_new_object ) {
				PyErr_WarnEx( PyExc_RuntimeWarning,
					"Argument converted (copied) to FORTRAN-contiguous array.", 1 );
			}
			return array;
		} else {
			array = obj_to_array_no_conversion( input, NumpyType< typename MatT::elem_type >::val );
			if ( !array || !require_dimensions( array, nd ) )return NULL;
				if( !array_is_fortran(array) ) {
					PyErr_SetString(PyExc_TypeError,
						"Array must be FORTRAN contiguous."\
						"  A non-FORTRAN-contiguous array was given");
					return NULL;
			   } 
			return array;
		}
	}

	template< typename MatT >
	void armanpy_cube_as_numpy_with_shared_memory( MatT *m, PyObject* input )
    {
		PyArrayObject* ary= (PyArrayObject*)input;
		ary->dimensions[0] = m->n_rows;
		ary->dimensions[1] = m->n_cols;
		ary->dimensions[2] = m->n_slices;
	    ary->strides[0]    = sizeof( typename MatT::elem_type  );
	    ary->strides[1]    = sizeof( typename MatT::elem_type  ) * m->n_rows;
	    ary->strides[2]    = sizeof( typename MatT::elem_type  ) * m->n_rows * m->n_cols;
	    if(  m->mem != ( typename MatT::elem_type *)array_data(ary) ) {
		    // if( ! m->uses_local_mem() ) {
				// 1. We do not need the memory at ary->data anymore
				//    This can be simply removed by PyArray_free( ary->data );
				PyArray_free( ary->data );

				// 2. We should "implant" the m->mem into ary->data
				//    Here we use the trick from http://blog.enthought.com/?p=62
				ary->flags = ary->flags & ~( NPY_OWNDATA ); 
				ArmaCapsule< MatT > *capsule;
				capsule      = PyObject_New( ArmaCapsule< MatT >, &ArmaCapsulePyType<MatT>::object );
				capsule->mat = m;
				ary->data = (char *)capsule->mat->mem;
				PyArray_BASE(ary) = (PyObject *)capsule;
			//} else {
			    // Here we just copy a few bytes, as local memory of arma is typically small
			//	memcpy ( ary->data, m->mem, sizeof( typename MatT::elem_type ) * m->n_elem );
			//	delete m;
			//}
		} else {
			// Memory was not changed at all; i.e. all modifications were done on the original
			// memory brought by the input numpy array. So we just delete the arma array
			// which does not free the memory as it was constructed with the aux memory constructor
			delete m;
		}
	}
	
	template< typename MatT >
	bool armanpy_numpy_as_cube_with_shared_memory( PyObject* input, MatT **m )
	{
		PyArrayObject* array = obj_to_array_no_conversion( input, NumpyType< typename MatT::elem_type >::val );
		if ( !array || !require_dimensions( array, 3) ) return false;
		if( ! ( PyArray_FLAGS(array) & NPY_OWNDATA ) ) {
            PyErr_SetString(PyExc_TypeError, "Array must own its data.");
            return false;
		}
		if ( !array_is_fortran(array) ) {
            PyErr_SetString(PyExc_TypeError,
                "Array must be FORTRAN contiguous.  A non-FORTRAN-contiguous array was given");
            return false;
        }
		unsigned r = array->dimensions[0];
		unsigned c = array->dimensions[1];
		unsigned s = array->dimensions[2];
		*m = new MatT( (typename MatT::elem_type *)array_data(array), r, c, s, false, false );
		return true;
	}
	
	template< typename MatT >
	PyObject* armanpy_cube_copy_to_numpy( MatT * m )
    {
		npy_intp dims[3] = { m->n_rows, m->n_cols, m->n_slices };
		PyObject* array = PyArray_EMPTY( ArmaTypeInfo<MatT>::numdim, dims, ArmaTypeInfo<MatT>::type, true);
		if ( !array || !array_is_fortran( array ) ) {
			PyErr_SetString( PyExc_TypeError, "Creation of 3-dimensional return array failed" );
			return NULL;
		}
		std::copy( m->begin(), m->end(), reinterpret_cast< typename MatT::elem_type *>(array_data(array)) );
		return array;
	 }
	 
#if defined(ARMANPY_SHARED_PTR)

    template< typename MatT >
	PyObject* armanpy_cube_bsptr_as_numpy_with_shared_memory( boost::shared_ptr< MatT > m )
    {
		npy_intp dims[3] = { 1, 1, 1 };
		PyArrayObject* ary = (PyArrayObject*)PyArray_EMPTY(3, dims, NumpyType< typename MatT::elem_type >::val, true);
		if ( !ary || !array_is_fortran(ary) ) { return NULL; }

		ary->dimensions[0] = m->n_rows;
		ary->dimensions[1] = m->n_cols;
		ary->dimensions[2] = m->n_slices;
	    ary->strides[0]    = sizeof( typename MatT::elem_type  );
	    ary->strides[1]    = sizeof( typename MatT::elem_type  ) * m->n_rows;
	    ary->strides[2]    = sizeof( typename MatT::elem_type  ) * m->n_rows * m->n_cols;

		// 1. We do not need the memory at ary->data anymore
		//    This can be simply removed by PyArray_free( ary->data );
		PyArray_free( ary->data );

		// 2. We should "implant" the m->mem into ary->data
		//    Here we use the trick from http://blog.enthought.com/?p=62
		ary->flags = ary->flags & ~( NPY_OWNDATA ); 
		ArmaBsptrCapsule< MatT > *capsule;
		capsule      = PyObject_New( ArmaBsptrCapsule< MatT >, &ArmaBsptrCapsulePyType<MatT>::object );
		capsule->mat = new boost::shared_ptr< MatT >();		
		ary->data = (char *)( m->mem );
		(*(capsule->mat)) = m;
		PyArray_BASE(ary) = (PyObject *)capsule;
		return (PyObject*)ary;
	}

#endif
	
}

//////////////////////////////////////////////////////////////////////////
// INPUT Arguments
//////////////////////////////////////////////////////////////////////////

%define %armanpy_cube_const_ref_typemaps( ARMA_MAT_TYPE )

	%typemap( typecheck, precedence=SWIG_TYPECHECK_FLOAT_ARRAY )
		( const ARMA_MAT_TYPE & ) ( PyArrayObject* array=NULL ),
		( const ARMA_MAT_TYPE   ) ( PyArrayObject* array=NULL ),
		(       ARMA_MAT_TYPE   ) ( PyArrayObject* array=NULL )
	{
		$1 = armanpy_basic_typecheck< ARMA_MAT_TYPE >( $input, false );
	}
	
	%typemap( in, fragment="armanpy_cube_typemaps" )
		( const ARMA_MAT_TYPE & ) ( PyArrayObject* array=NULL ),
		( const ARMA_MAT_TYPE   ) ( PyArrayObject* array=NULL ),
		(       ARMA_MAT_TYPE   ) ( PyArrayObject* array=NULL )
	{
		if( ! armanpy_basic_typecheck< ARMA_MAT_TYPE >( $input, true ) ) SWIG_fail;
		array = obj_to_array_no_conversion( $input, ArmaTypeInfo<ARMA_MAT_TYPE>::type );
		if( !array ) SWIG_fail;
		$1 = new ARMA_MAT_TYPE( ( ARMA_MAT_TYPE::elem_type *)array_data(array), 
					            array->dimensions[0], array->dimensions[1], array->dimensions[2], false );
	}
	
	%typemap( argout ) 
		( const ARMA_MAT_TYPE & ) ( PyArrayObject* array=NULL ),
		( const ARMA_MAT_TYPE   ) ( PyArrayObject* array=NULL ),
		(       ARMA_MAT_TYPE   ) ( PyArrayObject* array=NULL )
	{
	// NOOP
	}
	
	%typemap( freearg )
		( const ARMA_MAT_TYPE & ) ( PyArrayObject* array=NULL ),
		( const ARMA_MAT_TYPE   ) ( PyArrayObject* array=NULL ),
		(       ARMA_MAT_TYPE   ) ( PyArrayObject* array=NULL )
	{
	  delete $1;
	}

%enddef 

%armanpy_cube_const_ref_typemaps( arma::Cube< double > )
%armanpy_cube_const_ref_typemaps( arma::Cube< float >  )
%armanpy_cube_const_ref_typemaps( arma::Cube< int > )
%armanpy_cube_const_ref_typemaps( arma::Cube< unsigned >  )
%armanpy_cube_const_ref_typemaps( arma::Cube< arma::sword >  )
%armanpy_cube_const_ref_typemaps( arma::Cube< arma::uword >  )
%armanpy_cube_const_ref_typemaps( arma::Cube< arma::cx_double >  )
%armanpy_cube_const_ref_typemaps( arma::Cube< arma::cx_float >  )
%armanpy_cube_const_ref_typemaps( arma::Cube< std::complex< double > >  )
%armanpy_cube_const_ref_typemaps( arma::Cube< std::complex< float > >  )
%armanpy_cube_const_ref_typemaps( arma::cube )
%armanpy_cube_const_ref_typemaps( arma::fcube )
%armanpy_cube_const_ref_typemaps( arma::icube )
%armanpy_cube_const_ref_typemaps( arma::ucube )
%armanpy_cube_const_ref_typemaps( arma::uchar_cube )
%armanpy_cube_const_ref_typemaps( arma::u32_cube )
%armanpy_cube_const_ref_typemaps( arma::s32_cube )
%armanpy_cube_const_ref_typemaps( arma::cx_cube )
%armanpy_cube_const_ref_typemaps( arma::cx_fcube )

//////////////////////////////////////////////////////////////////////////
// Typemaps for input-output arguments. That is for arguments which are
// potentialliy modified in place.
//////////////////////////////////////////////////////////////////////////

// A macor for generating the typemaps for one matrix type
%define %armanpy_cube_ref_typemaps( ARMA_MAT_TYPE )

	%typemap( typecheck, precedence=SWIG_TYPECHECK_FLOAT_ARRAY )
		( ARMA_MAT_TYPE &)	
	{
		$1 = armanpy_basic_typecheck< ARMA_MAT_TYPE >( $input, false, true );
	}
	
	%typemap( in, fragment="armanpy_cube_typemaps" )
		( ARMA_MAT_TYPE &)
	{
		if( ! armanpy_basic_typecheck< ARMA_MAT_TYPE >( $input, true, true )             ) SWIG_fail;
		if( ! armanpy_numpy_as_cube_with_shared_memory< ARMA_MAT_TYPE >( $input, &($1) ) ) SWIG_fail;
	}

	%typemap( argout, fragment="armanpy_cube_typemaps" )
		( ARMA_MAT_TYPE & )
	{
		armanpy_cube_as_numpy_with_shared_memory( $1, $input );
	}
	
	%typemap( freearg )
		( ARMA_MAT_TYPE & )
	{
	   // NOOP
	}

%enddef 

%armanpy_cube_ref_typemaps( arma::Cube< double > )
%armanpy_cube_ref_typemaps( arma::Cube< float >  )
%armanpy_cube_ref_typemaps( arma::Cube< int > )
%armanpy_cube_ref_typemaps( arma::Cube< unsigned >  )
%armanpy_cube_ref_typemaps( arma::Cube< arma::sword >  )
%armanpy_cube_ref_typemaps( arma::Cube< arma::uword >  )
%armanpy_cube_ref_typemaps( arma::Cube< arma::cx_double >  )
%armanpy_cube_ref_typemaps( arma::Cube< arma::cx_float >  )
%armanpy_cube_ref_typemaps( arma::Cube< std::complex< double > >  )
%armanpy_cube_ref_typemaps( arma::Cube< std::complex< float > >  )
%armanpy_cube_ref_typemaps( arma::cube )
%armanpy_cube_ref_typemaps( arma::fcube )
%armanpy_cube_ref_typemaps( arma::icube )
%armanpy_cube_ref_typemaps( arma::ucube )
%armanpy_cube_ref_typemaps( arma::uchar_cube )
%armanpy_cube_ref_typemaps( arma::u32_cube )
%armanpy_cube_ref_typemaps( arma::s32_cube )
%armanpy_cube_ref_typemaps( arma::cx_cube )
%armanpy_cube_ref_typemaps( arma::cx_fcube )

//////////////////////////////////////////////////////////////////////////
// Typemaps for return by value functions/methods
//////////////////////////////////////////////////////////////////////////

%define %armanpy_cube_return_by_value_typemaps( ARMA_MAT_TYPE )
	%typemap( out ) 
		( ARMA_MAT_TYPE )
	{
	  PyObject* array = armanpy_cube_copy_to_numpy< ARMA_MAT_TYPE >( &$1 );
	  if ( !array ) SWIG_fail;
	  $result = SWIG_Python_AppendOutput($result, array);
	}
%enddef 

%armanpy_cube_return_by_value_typemaps( arma::Cube< double > )
%armanpy_cube_return_by_value_typemaps( arma::Cube< float >  )
%armanpy_cube_return_by_value_typemaps( arma::Cube< int > )
%armanpy_cube_return_by_value_typemaps( arma::Cube< unsigned >  )
%armanpy_cube_return_by_value_typemaps( arma::Cube< arma::sword >  )
%armanpy_cube_return_by_value_typemaps( arma::Cube< arma::uword >  )
%armanpy_cube_return_by_value_typemaps( arma::Cube< arma::cx_double >  )
%armanpy_cube_return_by_value_typemaps( arma::Cube< arma::cx_float >  )
%armanpy_cube_return_by_value_typemaps( arma::Cube< std::complex< double > >  )
%armanpy_cube_return_by_value_typemaps( arma::Cube< std::complex< float > >  )
%armanpy_cube_return_by_value_typemaps( arma::cube )
%armanpy_cube_return_by_value_typemaps( arma::fcube )
%armanpy_cube_return_by_value_typemaps( arma::icube )
%armanpy_cube_return_by_value_typemaps( arma::ucube )
%armanpy_cube_return_by_value_typemaps( arma::uchar_cube )
%armanpy_cube_return_by_value_typemaps( arma::u32_cube )
%armanpy_cube_return_by_value_typemaps( arma::s32_cube )
%armanpy_cube_return_by_value_typemaps( arma::cx_cube )
%armanpy_cube_return_by_value_typemaps( arma::cx_fcube )

//////////////////////////////////////////////////////////////////////////
// Typemaps for return by boost::shared_ptr< ... > functions/methods
//////////////////////////////////////////////////////////////////////////

#if defined(ARMANPY_SHARED_PTR)

%define %armanpy_cube_return_by_bsptr_typemaps( ARMA_MAT_TYPE )
	%typemap( out , fragment="armanpy_cube_typemaps" ) 
		( boost::shared_ptr< ARMA_MAT_TYPE > )
	{
	  PyObject* array = armanpy_cube_bsptr_as_numpy_with_shared_memory< ARMA_MAT_TYPE >( $1 );
	  if ( !array ) { SWIG_fail; }
	  $result = SWIG_Python_AppendOutput($result, array);
	}
%enddef 

%armanpy_cube_return_by_bsptr_typemaps( arma::Cube< double > )
%armanpy_cube_return_by_bsptr_typemaps( arma::Cube< float >  )
%armanpy_cube_return_by_bsptr_typemaps( arma::Cube< int > )
%armanpy_cube_return_by_bsptr_typemaps( arma::Cube< unsigned >  )
%armanpy_cube_return_by_bsptr_typemaps( arma::Cube< arma::sword >  )
%armanpy_cube_return_by_bsptr_typemaps( arma::Cube< arma::uword >  )
%armanpy_cube_return_by_bsptr_typemaps( arma::Cube< arma::cx_double >  )
%armanpy_cube_return_by_bsptr_typemaps( arma::Cube< arma::cx_float >  )
%armanpy_cube_return_by_bsptr_typemaps( arma::Cube< std::complex< double > >  )
%armanpy_cube_return_by_bsptr_typemaps( arma::Cube< std::complex< float > >  )
%armanpy_cube_return_by_bsptr_typemaps( arma::cube )
%armanpy_cube_return_by_bsptr_typemaps( arma::fcube )
%armanpy_cube_return_by_bsptr_typemaps( arma::icube )
%armanpy_cube_return_by_bsptr_typemaps( arma::ucube )
%armanpy_cube_return_by_bsptr_typemaps( arma::uchar_cube )
%armanpy_cube_return_by_bsptr_typemaps( arma::u32_cube )
%armanpy_cube_return_by_bsptr_typemaps( arma::s32_cube )
%armanpy_cube_return_by_bsptr_typemaps( arma::cx_cube )
%armanpy_cube_return_by_bsptr_typemaps( arma::cx_fcube )

#endif
