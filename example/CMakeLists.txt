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

# build the example library
add_library( examplelib SHARED example.cpp example.hpp )

##
## Set up the swig wrapper
##

# Swig shall but all stuff to the same directory where the *.so or *.pyd are placed
SET( CMAKE_SWIG_OUTDIR "${PROJECT_BINARY_DIR}/bin" )

# We need to say that we deal with C++ and do not care about missing include files and declarations
set_source_files_properties( example.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties( example.i PROPERTIES SWIG_FLAGS "-ignoremissing;-w509" )

swig_add_module    ( armanpyexample python example.i )
swig_link_libraries( armanpyexample examplelib ${PYTHON_LIBRARIES} )

configure_file( example.py "${CMAKE_SWIG_OUTDIR}/example.py" COPYONLY )

add_test( NAME example COMMAND ${PYTHON_EXECUTABLE} example.py WORKING_DIRECTORY ${CMAKE_SWIG_OUTDIR} )
