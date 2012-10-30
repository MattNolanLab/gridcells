cmake_minimum_required(VERSION 2.8.5 FATAL_ERROR)

project( armanpytest )

SET( CMAKE_VERBOSE_MAKEFILE ON )

### Armadillo
find_path( ARMADILLO_INCLUDE_DIR
		NAMES armadillo
		PATHS ../../../dependencies/armadillo-3.4.3 ../../../dependencies/armadillo
		PATH_SUFFIXES "include"
)
message( STATUS "Armadillo" )
message( STATUS "    IncDirs: ${ARMADILLO_INCLUDE_DIR}" )

### Swig
find_package(SWIG REQUIRED)
include( ${SWIG_USE_FILE} )
message( STATUS "SWIG ${SWIG_VERSION}" )
message( STATUS "  Executable: ${SWIG_EXECUTABLE}" )

### Python
find_package( PythonLibs REQUIRED )
find_package( PythonInterp REQUIRED )
message( STATUS "Python ${PYTHON_VERSION_STRING}" )
message( STATUS "  Executable: ${PYTHON_EXECUTABLE}" )

### Boost
find_package( Boost 1.42.0 REQUIRED )
message( STATUS "Boost C++" )
message( STATUS "    IncDirs: ${Boost_INCLUDE_DIRS}" )


### NumPy
if( NOT NUMPY_INCLUDE_DIRS ) 

	execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print numpy.get_include();"
					  RESULT_VARIABLE NUMPY_RESULT
					  OUTPUT_VARIABLE NUMPY_OUTPUT
					  ERROR_VARIABLE NUMPY_ERRUR
					  )

	if( ${NUMPY_RESULT} )
		message( FATAL "Failed to find numpy includes" )
	else()
		string(STRIP ${NUMPY_OUTPUT} NUMPY_INCLUDE_DIRS)
		set( NUMPY_INCLUDE_DIRS ${NUMPY_INCLUDE_DIRS} CACHE PATH "Include path for numpy headers" )
		
		execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print numpy.version.full_version;"
					  RESULT_VARIABLE NUMPY_RESULT
					  OUTPUT_VARIABLE NUMPY_OUTPUT
					  ERROR_VARIABLE NUMPY_ERRUR
					  )

		string(STRIP ${NUMPY_OUTPUT} NUMPY_VERSION_STRING)
		set( NUMPY_VERSION_STRING ${NUMPY_VERSION_STRING} CACHE STRING "Numpy version" )
	endif()
	
	
endif()
message( STATUS "Numpy ${NUMPY_VERSION_STRING}" )
message( STATUS "  Version: ${NUMPY_VERSION_STRING}" )
message( STATUS "  Include: ${NUMPY_INCLUDE_DIRS}" )

if( MSVC )
	# for multi-config build MSVC
	foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
		message( STATUS ${OUTPUTCONFIG}  )
		string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
		set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${PROJECT_BINARY_DIR}/bin" )
		set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${PROJECT_BINARY_DIR}/bin" )
		set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${PROJECT_BINARY_DIR}/bin" )
	endforeach()
else()
	# For the generic no-config case (e.g. with mingw)
	set( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin" )
	set( CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin" )
	set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin" )
endif()

set( ARMANPY_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../include )
include_directories(
	${PROJECT_SOURCE_DIR}
	${ARMANPY_INCLUDE_DIR}
	${NUMPY_INCLUDE_DIRS}
	${PYTHON_INCLUDE_DIRS}
	${CMAKE_CURRENT_SOURCE_DIR}
	${Boost_INCLUDE_DIRS}
	${ARMADILLO_INCLUDE_DIR} 
)

set( TEST_LIBRARY_NAME armanpytest )

configure_file ( dlldefines.h.in ${PROJECT_SOURCE_DIR}dlldefines.h )

add_library( ${TEST_LIBRARY_NAME} SHARED test.cpp test.hpp )

SET( CMAKE_SWIG_OUTDIR "${PROJECT_BINARY_DIR}/bin" )
set_source_files_properties( test.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties( test.i PROPERTIES SWIG_FLAGS "-includeall;-ignoremissing" ) # ;-debug-tmused;-debug-typedef;-debug-typemap" )
swig_add_module( armanpytest python test.i )
swig_link_libraries( armanpytest ${TEST_LIBRARY_NAME} ${PYTHON_LIBRARIES} )
message( STATUS "SWIG_MODULE_armanpytest_EXTRA_DEPS=${SWIG_MODULE_armanpytest_EXTRA_DEPS}" )
set( SWIG_MODULE_armanpytest_EXTRA_DEPS armanpy.i numpy.i test.cpp test.hpp )
message( STATUS "SWIG_MODULE_armanpytest_EXTRA_DEPS=${SWIG_MODULE_armanpytest_EXTRA_DEPS}" )
