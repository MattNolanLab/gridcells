
%fragment( "armanpy_vec_typemaps", "header", fragment="armanpy_typemaps" )
{

	///////////////////////////////////////// numpy -> arma ////////////////////////////////////////
	
	template< typename VecT >
	bool armanpy_numpy_as_vec_with_shared_memory( PyObject* input, VecT **m )
	{
		PyArrayObject* array = obj_to_array_no_conversion( input, NumpyType< typename VecT::elem_type >::val );
		if ( !array || !require_dimensions(array, 1) ) return false;
		if( ! ( PyArray_FLAGS(array) & NPY_OWNDATA ) ) {
            PyErr_SetString(PyExc_TypeError, "Array must own its data.");
            return false;
		}
		if ( !array_is_contiguous(array) ) {
            PyErr_SetString(PyExc_TypeError,
                "Array must be FORTRAN contiguous.  A non-FORTRAN-contiguous array was given");
            return false;
        }
		unsigned p = array->dimensions[0];
		*m = new VecT( (typename VecT::elem_type *)array_data(array), p, false, false );
		return true;
	}
	
	/////////////////////////////////////// arma -> numpy ////////////////////////////////////////////
	
	template< typename VecT >
	void armanpy_vec_as_numpy_with_shared_memory( VecT *m, PyObject* input )
    {
		PyArrayObject* ary= (PyArrayObject*)input;
		ary->dimensions[0] = m->n_elem;
	    ary->strides[0]    = sizeof(  typename VecT::elem_type  );
	    if(  m->mem != ( typename VecT::elem_type *)array_data(ary) ) {
		    // if( ! m->uses_local_mem() ) {
				// 1. We do not need the memory at ary->data anymore
				//    This can be simply removed by PyArray_free( ary->data );
				PyArray_free( ary->data );

				// 2. We should "implant" the m->mem into ary->data
				//    Here we use the trick from http://blog.enthought.com/?p=62
				ary->flags = ary->flags & ~( NPY_OWNDATA ); 
				ArmaCapsule< VecT > *capsule;
				capsule      = PyObject_New( ArmaCapsule< VecT >, &ArmaCapsulePyType<VecT>::object );
				capsule->mat = m;
				ary->data = (char *)capsule->mat->mem;
				PyArray_BASE(ary) = (PyObject *)capsule;
			//} else {
			    // Here we just copy a few bytes, as local memory of arma is typically small
			//	memcpy ( ary->data, m->mem, sizeof( typename VecT::elem_type ) * m->n_elem );
			//	delete m;
			//}
		} else {
			// Memory was not changed at all; i.e. all modifications were done on the original
			// memory brought by the input numpy array. So we just delete the arma array
			// which does not free the memory as it was constructed with the aux memory constructor
			delete m;
		}
	}

	template< typename VecT >
	PyObject* armanpy_vec_copy_to_numpy( VecT * m )
    {
		npy_intp dims[1] = { m->n_elem };
		PyObject* array = PyArray_EMPTY( ArmaTypeInfo<VecT>::numdim, dims, ArmaTypeInfo<VecT>::type, true);
		if ( !array || !array_is_contiguous( array ) ) {
			PyErr_SetString( PyExc_TypeError, "Creation of 1-dimensional return array failed" );
			return NULL;
		}
		std::copy( m->begin(), m->end(), reinterpret_cast< typename VecT::elem_type *>(array_data(array)) );
		return array;
	 }
	  
#if defined(ARMANPY_SHARED_PTR)
	
	template< typename VecT >
	PyObject* armanpy_vec_bsptr_as_numpy_with_shared_memory( boost::shared_ptr< VecT > m )
    {
		npy_intp dims[1] = { 1 };
		PyArrayObject* ary = (PyArrayObject*)PyArray_EMPTY(1, dims, NumpyType< typename VecT::elem_type >::val, true);
		if ( !ary || !array_is_contiguous(ary) ) { return NULL; }

		ary->dimensions[0] = m->n_elem;
	    ary->strides[0]    = sizeof(  typename VecT::elem_type  );

		// 1. We do not need the memory at ary->data anymore
		//    This can be simply removed by PyArray_free( ary->data );
		PyArray_free( ary->data );

		// 2. We should "implant" the m->mem into ary->data
		//    Here we use the trick from http://blog.enthought.com/?p=62
		ary->flags = ary->flags & ~( NPY_OWNDATA ); 
		ArmaBsptrCapsule< VecT > *capsule;
		capsule      = PyObject_New( ArmaBsptrCapsule< VecT >, &ArmaBsptrCapsulePyType<VecT>::object );
		capsule->mat = new boost::shared_ptr< VecT >();		
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

%define %armanpy_vec_const_ref_typemaps( ARMA_MAT_TYPE )

	%typemap( typecheck, precedence=SWIG_TYPECHECK_FLOAT_ARRAY )
		( const ARMA_MAT_TYPE & ) ( PyArrayObject* array=NULL ),
		( const ARMA_MAT_TYPE   ) ( PyArrayObject* array=NULL ),
		(       ARMA_MAT_TYPE   ) ( PyArrayObject* array=NULL )
	{
		$1 = armanpy_basic_typecheck< ARMA_MAT_TYPE >( $input, false );
	}
	
	%typemap( in, fragment="armanpy_vec_typemaps" )
		( const ARMA_MAT_TYPE & ) ( PyArrayObject* array=NULL ),
		( const ARMA_MAT_TYPE   ) ( PyArrayObject* array=NULL ),
		(       ARMA_MAT_TYPE   ) ( PyArrayObject* array=NULL )
	{
		if( ! armanpy_basic_typecheck< ARMA_MAT_TYPE >( $input, true ) ) SWIG_fail;
		array = obj_to_array_no_conversion( $input, ArmaTypeInfo<ARMA_MAT_TYPE>::type );
		if( !array ) SWIG_fail;
		$1 = new ARMA_MAT_TYPE( ( ARMA_MAT_TYPE::elem_type *)array_data(array),
					            array->dimensions[0], false );
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

%armanpy_vec_const_ref_typemaps( arma::Col< double > )
%armanpy_vec_const_ref_typemaps( arma::Col< float >  )
%armanpy_vec_const_ref_typemaps( arma::Col< int > )
%armanpy_vec_const_ref_typemaps( arma::Col< unsigned >  )
%armanpy_vec_const_ref_typemaps( arma::Col< arma::sword >  )
%armanpy_vec_const_ref_typemaps( arma::Col< arma::uword >  )
%armanpy_vec_const_ref_typemaps( arma::Col< arma::cx_double >  )
%armanpy_vec_const_ref_typemaps( arma::Col< arma::cx_float >  )
%armanpy_vec_const_ref_typemaps( arma::Col< std::complex< double > > )
%armanpy_vec_const_ref_typemaps( arma::Col< std::complex< float > > )
%armanpy_vec_const_ref_typemaps( arma::vec )
%armanpy_vec_const_ref_typemaps( arma::fvec )
%armanpy_vec_const_ref_typemaps( arma::ivec )
%armanpy_vec_const_ref_typemaps( arma::uvec )
%armanpy_vec_const_ref_typemaps( arma::uchar_vec )
%armanpy_vec_const_ref_typemaps( arma::u32_vec )
%armanpy_vec_const_ref_typemaps( arma::s32_vec )
%armanpy_vec_const_ref_typemaps( arma::cx_vec )
%armanpy_vec_const_ref_typemaps( arma::cx_fvec )
%armanpy_vec_const_ref_typemaps( arma::colvec )
%armanpy_vec_const_ref_typemaps( arma::fcolvec )
%armanpy_vec_const_ref_typemaps( arma::icolvec )
%armanpy_vec_const_ref_typemaps( arma::ucolvec )
%armanpy_vec_const_ref_typemaps( arma::uchar_colvec )
%armanpy_vec_const_ref_typemaps( arma::u32_colvec )
%armanpy_vec_const_ref_typemaps( arma::s32_colvec )
%armanpy_vec_const_ref_typemaps( arma::cx_colvec )
%armanpy_vec_const_ref_typemaps( arma::cx_fcolvec )

%armanpy_vec_const_ref_typemaps( arma::Row< double > )
%armanpy_vec_const_ref_typemaps( arma::Row< float >  )
%armanpy_vec_const_ref_typemaps( arma::Row< int > )
%armanpy_vec_const_ref_typemaps( arma::Row< unsigned >  )
%armanpy_vec_const_ref_typemaps( arma::Row< arma::sword >  )
%armanpy_vec_const_ref_typemaps( arma::Row< arma::uword >  )
%armanpy_vec_const_ref_typemaps( arma::Row< std::complex< double > > )
%armanpy_vec_const_ref_typemaps( arma::Row< std::complex< float > > )
%armanpy_vec_const_ref_typemaps( arma::Row< arma::cx_double >  )
%armanpy_vec_const_ref_typemaps( arma::Row< arma::cx_float >  )
%armanpy_vec_const_ref_typemaps( arma::rowvec )
%armanpy_vec_const_ref_typemaps( arma::frowvec )
%armanpy_vec_const_ref_typemaps( arma::irowvec )
%armanpy_vec_const_ref_typemaps( arma::urowvec )
%armanpy_vec_const_ref_typemaps( arma::uchar_rowvec )
%armanpy_vec_const_ref_typemaps( arma::u32_rowvec )
%armanpy_vec_const_ref_typemaps( arma::s32_rowvec )
%armanpy_vec_const_ref_typemaps( arma::cx_rowvec )
%armanpy_vec_const_ref_typemaps( arma::cx_frowvec )

//////////////////////////////////////////////////////////////////////////
// Typemaps for input-output arguments. That is for arguments which are
// potentialliy modified in place.
//////////////////////////////////////////////////////////////////////////

// A macor for generating the typemaps for one matrix type
%define %armanpy_vec_ref_typemaps( ARMA_MAT_TYPE )

	%typemap( typecheck, precedence=SWIG_TYPECHECK_FLOAT_ARRAY )
		( ARMA_MAT_TYPE &)	
	{
		$1 = armanpy_basic_typecheck< ARMA_MAT_TYPE >( $input, false, true );
	}
	
	%typemap( in, fragment="armanpy_vec_typemaps" )
		( ARMA_MAT_TYPE &)
	{
		if( ! armanpy_basic_typecheck< ARMA_MAT_TYPE >( $input, true, true )            ) SWIG_fail;
		if( ! armanpy_numpy_as_vec_with_shared_memory< ARMA_MAT_TYPE >( $input, &($1) ) ) SWIG_fail;
	}

	%typemap( argout, fragment="armanpy_vec_typemaps" )
		( ARMA_MAT_TYPE & )
	{
		armanpy_vec_as_numpy_with_shared_memory( $1, $input );
	}
	
	%typemap( freearg )
		( ARMA_MAT_TYPE & )
	{
	   // NOOP
	}

%enddef 

%armanpy_vec_ref_typemaps( arma::Col< double > )
%armanpy_vec_ref_typemaps( arma::Col< float >  )
%armanpy_vec_ref_typemaps( arma::Col< int > )
%armanpy_vec_ref_typemaps( arma::Col< unsigned >  )
%armanpy_vec_ref_typemaps( arma::Col< arma::sword >  )
%armanpy_vec_ref_typemaps( arma::Col< arma::uword >  )
%armanpy_vec_ref_typemaps( arma::Col< arma::cx_double >  )
%armanpy_vec_ref_typemaps( arma::Col< arma::cx_float >  )
%armanpy_vec_ref_typemaps( arma::Col< std::complex< double > > )
%armanpy_vec_ref_typemaps( arma::Col< std::complex< float > > )
%armanpy_vec_ref_typemaps( arma::vec )
%armanpy_vec_ref_typemaps( arma::fvec )
%armanpy_vec_ref_typemaps( arma::ivec )
%armanpy_vec_ref_typemaps( arma::uvec )
%armanpy_vec_ref_typemaps( arma::uchar_vec )
%armanpy_vec_ref_typemaps( arma::u32_vec )
%armanpy_vec_ref_typemaps( arma::s32_vec )
%armanpy_vec_ref_typemaps( arma::cx_vec )
%armanpy_vec_ref_typemaps( arma::cx_fvec )
%armanpy_vec_ref_typemaps( arma::colvec )
%armanpy_vec_ref_typemaps( arma::fcolvec )
%armanpy_vec_ref_typemaps( arma::icolvec )
%armanpy_vec_ref_typemaps( arma::ucolvec )
%armanpy_vec_ref_typemaps( arma::uchar_colvec )
%armanpy_vec_ref_typemaps( arma::u32_colvec )
%armanpy_vec_ref_typemaps( arma::s32_colvec )
%armanpy_vec_ref_typemaps( arma::cx_colvec )
%armanpy_vec_ref_typemaps( arma::cx_fcolvec )


%armanpy_vec_ref_typemaps( arma::Row< double > )
%armanpy_vec_ref_typemaps( arma::Row< float >  )
%armanpy_vec_ref_typemaps( arma::Row< int > )
%armanpy_vec_ref_typemaps( arma::Row< unsigned >  )
%armanpy_vec_ref_typemaps( arma::Row< arma::sword >  )
%armanpy_vec_ref_typemaps( arma::Row< arma::uword >  )
%armanpy_vec_ref_typemaps( arma::Row< arma::cx_double >  )
%armanpy_vec_ref_typemaps( arma::Row< arma::cx_float >  )
%armanpy_vec_ref_typemaps( arma::Row< std::complex< double > > )
%armanpy_vec_ref_typemaps( arma::Row< std::complex< float > > )
%armanpy_vec_ref_typemaps( arma::rowvec )
%armanpy_vec_ref_typemaps( arma::frowvec )
%armanpy_vec_ref_typemaps( arma::irowvec )
%armanpy_vec_ref_typemaps( arma::urowvec )
%armanpy_vec_ref_typemaps( arma::uchar_rowvec )
%armanpy_vec_ref_typemaps( arma::u32_rowvec )
%armanpy_vec_ref_typemaps( arma::s32_rowvec )
%armanpy_vec_ref_typemaps( arma::cx_rowvec )
%armanpy_vec_ref_typemaps( arma::cx_frowvec )

//////////////////////////////////////////////////////////////////////////
// Typemaps for return by value functions/methods
//////////////////////////////////////////////////////////////////////////

%define %armanpy_vec_return_by_value_typemaps( ARMA_MAT_TYPE )
	%typemap( out ) 
		( ARMA_MAT_TYPE )
	{
	  PyObject* array = armanpy_vec_copy_to_numpy< ARMA_MAT_TYPE >( &$1 );
	  if ( !array ) SWIG_fail;
	  $result = SWIG_Python_AppendOutput($result, array);
	}
%enddef 

%armanpy_vec_return_by_value_typemaps( arma::Col< double > )
%armanpy_vec_return_by_value_typemaps( arma::Col< float >  )
%armanpy_vec_return_by_value_typemaps( arma::Col< int > )
%armanpy_vec_return_by_value_typemaps( arma::Col< unsigned >  )
%armanpy_vec_return_by_value_typemaps( arma::Col< arma::sword >  )
%armanpy_vec_return_by_value_typemaps( arma::Col< arma::uword >  )
%armanpy_vec_return_by_value_typemaps( arma::Col< arma::cx_double >  )
%armanpy_vec_return_by_value_typemaps( arma::Col< arma::cx_float >  )
%armanpy_vec_return_by_value_typemaps( arma::Col< std::complex< double > > )
%armanpy_vec_return_by_value_typemaps( arma::Col< std::complex< float > > )
%armanpy_vec_return_by_value_typemaps( arma::vec )
%armanpy_vec_return_by_value_typemaps( arma::fvec )
%armanpy_vec_return_by_value_typemaps( arma::ivec )
%armanpy_vec_return_by_value_typemaps( arma::uvec )
%armanpy_vec_return_by_value_typemaps( arma::uchar_vec )
%armanpy_vec_return_by_value_typemaps( arma::u32_vec )
%armanpy_vec_return_by_value_typemaps( arma::s32_vec )
%armanpy_vec_return_by_value_typemaps( arma::cx_vec )
%armanpy_vec_return_by_value_typemaps( arma::cx_fvec )
%armanpy_vec_return_by_value_typemaps( arma::colvec )
%armanpy_vec_return_by_value_typemaps( arma::fcolvec )
%armanpy_vec_return_by_value_typemaps( arma::icolvec )
%armanpy_vec_return_by_value_typemaps( arma::ucolvec )
%armanpy_vec_return_by_value_typemaps( arma::uchar_colvec )
%armanpy_vec_return_by_value_typemaps( arma::u32_colvec )
%armanpy_vec_return_by_value_typemaps( arma::s32_colvec )
%armanpy_vec_return_by_value_typemaps( arma::cx_colvec )
%armanpy_vec_return_by_value_typemaps( arma::cx_fcolvec )

%armanpy_vec_return_by_value_typemaps( arma::Row< double > )
%armanpy_vec_return_by_value_typemaps( arma::Row< float >  )
%armanpy_vec_return_by_value_typemaps( arma::Row< int > )
%armanpy_vec_return_by_value_typemaps( arma::Row< unsigned >  )
%armanpy_vec_return_by_value_typemaps( arma::Row< arma::sword >  )
%armanpy_vec_return_by_value_typemaps( arma::Row< arma::uword >  )
%armanpy_vec_return_by_value_typemaps( arma::Row< arma::cx_double >  )
%armanpy_vec_return_by_value_typemaps( arma::Row< arma::cx_float >  )
%armanpy_vec_return_by_value_typemaps( arma::Row< std::complex< double > > )
%armanpy_vec_return_by_value_typemaps( arma::Row< std::complex< float > > )
%armanpy_vec_return_by_value_typemaps( arma::rowvec )
%armanpy_vec_return_by_value_typemaps( arma::frowvec )
%armanpy_vec_return_by_value_typemaps( arma::irowvec )
%armanpy_vec_return_by_value_typemaps( arma::urowvec )
%armanpy_vec_return_by_value_typemaps( arma::uchar_rowvec )
%armanpy_vec_return_by_value_typemaps( arma::u32_rowvec )
%armanpy_vec_return_by_value_typemaps( arma::s32_rowvec )
%armanpy_vec_return_by_value_typemaps( arma::cx_rowvec )
%armanpy_vec_return_by_value_typemaps( arma::cx_frowvec )

//////////////////////////////////////////////////////////////////////////
// Typemaps for return by boost::shared_ptr< ... > functions/methods
//////////////////////////////////////////////////////////////////////////

#if defined(ARMANPY_SHARED_PTR)

%define %armanpy_vec_return_by_bsptr_typemaps( ARMA_MAT_TYPE )
	%typemap( out , fragment="armanpy_vec_typemaps" ) 
		( boost::shared_ptr< ARMA_MAT_TYPE > )
	{
	  PyObject* array = armanpy_vec_bsptr_as_numpy_with_shared_memory< ARMA_MAT_TYPE >( $1 );
	  if ( !array ) SWIG_fail;
	  $result = SWIG_Python_AppendOutput($result, array);
	}
%enddef 

%armanpy_vec_return_by_bsptr_typemaps( arma::Col< double > )
%armanpy_vec_return_by_bsptr_typemaps( arma::Col< float >  )
%armanpy_vec_return_by_bsptr_typemaps( arma::Col< int > )
%armanpy_vec_return_by_bsptr_typemaps( arma::Col< unsigned >  )
%armanpy_vec_return_by_bsptr_typemaps( arma::Col< arma::sword >  )
%armanpy_vec_return_by_bsptr_typemaps( arma::Col< arma::uword >  )
%armanpy_vec_return_by_bsptr_typemaps( arma::Col< arma::cx_double >  )
%armanpy_vec_return_by_bsptr_typemaps( arma::Col< arma::cx_float >  )
%armanpy_vec_return_by_bsptr_typemaps( arma::Col< std::complex< double > > )
%armanpy_vec_return_by_bsptr_typemaps( arma::Col< std::complex< float > > )
%armanpy_vec_return_by_bsptr_typemaps( arma::vec )
%armanpy_vec_return_by_bsptr_typemaps( arma::fvec )
%armanpy_vec_return_by_bsptr_typemaps( arma::ivec )
%armanpy_vec_return_by_bsptr_typemaps( arma::uvec )
%armanpy_vec_return_by_bsptr_typemaps( arma::uchar_vec )
%armanpy_vec_return_by_bsptr_typemaps( arma::u32_vec )
%armanpy_vec_return_by_bsptr_typemaps( arma::s32_vec )
%armanpy_vec_return_by_bsptr_typemaps( arma::cx_vec )
%armanpy_vec_return_by_bsptr_typemaps( arma::cx_fvec )
%armanpy_vec_return_by_bsptr_typemaps( arma::colvec )
%armanpy_vec_return_by_bsptr_typemaps( arma::fcolvec )
%armanpy_vec_return_by_bsptr_typemaps( arma::icolvec )
%armanpy_vec_return_by_bsptr_typemaps( arma::ucolvec )
%armanpy_vec_return_by_bsptr_typemaps( arma::uchar_colvec )
%armanpy_vec_return_by_bsptr_typemaps( arma::u32_colvec )
%armanpy_vec_return_by_bsptr_typemaps( arma::s32_colvec )
%armanpy_vec_return_by_bsptr_typemaps( arma::cx_colvec )
%armanpy_vec_return_by_bsptr_typemaps( arma::cx_fcolvec )

%armanpy_vec_return_by_bsptr_typemaps( arma::Row< double > )
%armanpy_vec_return_by_bsptr_typemaps( arma::Row< float >  )
%armanpy_vec_return_by_bsptr_typemaps( arma::Row< int > )
%armanpy_vec_return_by_bsptr_typemaps( arma::Row< unsigned >  )
%armanpy_vec_return_by_bsptr_typemaps( arma::Row< arma::sword >  )
%armanpy_vec_return_by_bsptr_typemaps( arma::Row< arma::uword >  )
%armanpy_vec_return_by_bsptr_typemaps( arma::Row< arma::cx_double >  )
%armanpy_vec_return_by_bsptr_typemaps( arma::Row< arma::cx_float >  )
%armanpy_vec_return_by_bsptr_typemaps( arma::Row< std::complex< double > > )
%armanpy_vec_return_by_bsptr_typemaps( arma::Row< std::complex< float > > )
%armanpy_vec_return_by_bsptr_typemaps( arma::rowvec )
%armanpy_vec_return_by_bsptr_typemaps( arma::frowvec )
%armanpy_vec_return_by_bsptr_typemaps( arma::irowvec )
%armanpy_vec_return_by_bsptr_typemaps( arma::urowvec )
%armanpy_vec_return_by_bsptr_typemaps( arma::uchar_rowvec )
%armanpy_vec_return_by_bsptr_typemaps( arma::u32_rowvec )
%armanpy_vec_return_by_bsptr_typemaps( arma::s32_rowvec )
%armanpy_vec_return_by_bsptr_typemaps( arma::cx_rowvec )
%armanpy_vec_return_by_bsptr_typemaps( arma::cx_frowvec )

#endif
