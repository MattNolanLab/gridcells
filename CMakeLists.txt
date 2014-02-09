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

cmake_minimum_required(VERSION 2.8.5 FATAL_ERROR)

SET( CMAKE_VERBOSE_MAKEFILE OFF )

project( armanpy )

set( ARMANPY_VERSION_MAJOR 0 )
set( ARMANPY_VERSION_MINOR 1 )
set( ARMANPY_VERSION_PATCH 3 )

### Armadillo

if( NOT ARMADILLO_INCLUDE_DIR )
    message( FATAL_ERROR "You must specify the include_directory of your armadillo installation via -DARMADILLO_INCLUDE_DIR=/path/to/armadillio/include" )
endif()

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

set( ARMANPY_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include )

file( GLOB ARMANPY_FILES "${PROJECT_SOURCE_DIR}/include/*.i")

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/test
    ${PROJECT_SOURCE_DIR}/example
    ${NUMPY_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${ARMADILLO_INCLUDE_DIR} 
)

enable_testing()

add_subdirectory( test    )
add_subdirectory( example )

# installation (mainly for building the source package via cpack -G ZIP
install( DIRECTORY  "${CMAKE_CURRENT_SOURCE_DIR}/include/"  DESTINATION include )
install( DIRECTORY  "${CMAKE_CURRENT_SOURCE_DIR}/test/"     DESTINATION test PATTERN "build" EXCLUDE )
install( DIRECTORY  "${CMAKE_CURRENT_SOURCE_DIR}/example/"  DESTINATION example PATTERN "build" EXCLUDE )
install( FILES README LICENSE CHANGELOG DESTINATION . )

#build a CPack driven installer package

SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "ArmaNpy")
SET(CPACK_PACKAGE_VENDOR "Thomas Natschläger")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION_MAJOR "${ARMANPY_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${ARMANPY_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${ARMANPY_VERSION_PATCH}")
set( CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-${ARMANPY_VERSION_MAJOR}.${ARMANPY_VERSION_MINOR}.${ARMANPY_VERSION_PATCH}" )
set( CPACK_TOPLEVEL_TAG      "${PROJECT_NAME}-${ARMANPY_VERSION_MAJOR}.${ARMANPY_VERSION_MINOR}.${ARMANPY_VERSION_PATCH}" )
include (CPack)
