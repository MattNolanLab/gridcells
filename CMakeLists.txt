#
# Some of the code snippets were taken from the armanpy library
# Modifications by: Lukas Solanka <lsolanka@gmail.com>
#
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
cmake_minimum_required(VERSION 2.6)

set(CMAKE_VERBOSE_MAKEFILE OFF)

project(gridcells)

set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(GRIDCELLS_VERSION_MAJOR 0)
set(GRIDCELLS_VERSION_MINOR 1)


# Armadillo
if( NOT ARMADILLO_INCLUDE_DIR )
    message( FATAL_ERROR "You must specify the include_directory of your armadillo installation via -DARMADILLO_INCLUDE_DIR=/path/to/armadillio/include" )
endif()

message( STATUS "Armadillo" )
message( STATUS "    IncDirs: ${ARMADILLO_INCLUDE_DIR}" )


# Swig
find_package(SWIG REQUIRED)
include( ${SWIG_USE_FILE} )
message( STATUS "SWIG ${SWIG_VERSION}" )
message( STATUS "  Executable: ${SWIG_EXECUTABLE}" )


# Python stuff
find_package(PythonInterp REQUIRED)
find_package(PythonLibrary REQUIRED)
message (STATUS "Python ${PYTHON_VERSION_STRING}")
message (STATUS "  Include: ${PYTHON_INCLUDE_DIRS}")
message (STATUS "  Library: ${PYTHON_LIBRARIES}")
message (STATUS "  Executable: ${PYTHON_EXECUTABLE}")
find_package(Numpy REQUIRED)
message (STATUS "Numpy ${NUMPY_VERSION_STRING}")
message (STATUS "  Version: ${NUMPY_VERSION_STRING}")
message (STATUS "  Include: ${NUMPY_INCLUDE_DIRS}")


set (ARMANPY_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/armanpy/include")

include_directories(
    ${PROJECT_SOURCE_DIR}/src/include
    ${NUMPY_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    ${ARMADILLO_INCLUDE_DIR} 
    ${ARMANPY_INCLUDE_DIR}
)

add_subdirectory(src)
