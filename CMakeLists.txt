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

set(gridcells_VERSION_MAJOR 0)
set(gridcells_VERSION_MINOR 1)
set(gridcells_VERSION_PATCH 0)
set(gridcells_VERSION
    "${gridcells_VERSION_MAJOR}.${gridcells_VERSION_MINOR}.${gridcells_VERSION_PATCH}"
)

set (ARMANPY_DIR "${PROJECT_SOURCE_DIR}/external/armanpy")


# Armadillo
if( NOT ARMADILLO_INCLUDE_DIR )
    message( FATAL_ERROR "You must specify the include_directory of your armadillo installation via -DARMADILLO_INCLUDE_DIR=/path/to/armadillio/include" )
endif()

message( STATUS "Armadillo" )
message( STATUS "    IncDirs: ${ARMADILLO_INCLUDE_DIR}" )


# Swig
find_package(SWIG 2.0 REQUIRED)
if (${SWIG_VERSION} VERSION_LESS "2.0")
    message(FATAL_ERROR
        "SWIG version must be >= 2.0. Found version ${SWIG_VERSION}")
endif()
include( ${SWIG_USE_FILE} )
message( STATUS "SWIG ${SWIG_VERSION}" )
message( STATUS "  Executable: ${SWIG_EXECUTABLE}" )

# Python modules dependencies
find_package(PythonInterp REQUIRED)
find_package(PythonLibrary REQUIRED)
message(STATUS "Python ${PYTHON_VERSION_STRING}")
message(STATUS "  Include: ${PYTHON_INCLUDE_DIRS}")
message(STATUS "  Library: ${PYTHON_LIBRARIES}")
message(STATUS "  Executable: ${PYTHON_EXECUTABLE}")
find_package(Numpy REQUIRED)
message(STATUS "Numpy ${NUMPY_VERSION_STRING}")
message(STATUS "  Version: ${NUMPY_VERSION_STRING}")
message(STATUS "  Include: ${NUMPY_INCLUDE_DIRS}")

# Python module installation settings
message(STATUS "Python modules:")
if (NOT DEFINED PYTHON_MODS_INSTALL_PREFIX)
    set(PYTHON_MODS_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()
if ((${PYTHON_SETUP_ARGUMENTS} MATCHES "^--user$") OR
    (${PYTHON_SETUP_ARGUMENTS} MATCHES "[ ]+--user[ ]+") OR
    (${PYTHON_SETUP_ARGUMENTS} MATCHES "^--user[ ]+") OR
    (${PYTHON_SETUP_ARGUMENTS} MATCHES "[ ]+--user$"))
    set(PYTHON_MODS_INSTALL_PREFIX "")
    message(STATUS "  Installation directory: --user")
else()
    message(STATUS "  Installation directory: ${PYTHON_MODS_INSTALL_PREFIX}")
endif()


set (ARMANPY_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/armanpy/include")
file (GLOB COMMON_SWIG_DEPS
    "${ARMANPY_DIR}/include/*.i"
    "${ARMANPY_DIR}/include/armanpy.hpp")


include_directories(
    ${PROJECT_SOURCE_DIR}/src/include
    ${NUMPY_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    ${ARMADILLO_INCLUDE_DIR} 
    ${ARMANPY_INCLUDE_DIR}
)

configure_file("setup.py.in" ${CMAKE_SOURCE_DIR}/setup.py)
configure_file("setup.cfg.in" ${CMAKE_SOURCE_DIR}/setup.cfg)

add_subdirectory(src)

set(python_install_code
    "execute_process(COMMAND
        ${PYTHON_EXECUTABLE} setup.py install ${PYTHON_SETUP_ARGUMENTS}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})"
) 
install(CODE ${python_install_code})
