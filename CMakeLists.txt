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

project( armanpy )

set( ARMANPY_VERSION_MAJOR 0 )
set( ARMANPY_VERSION_MINOR 1 )
set( ARMANPY_VERSION_PATCH 2 )

install( DIRECTORY  "${CMAKE_CURRENT_SOURCE_DIR}/include/"  DESTINATION include )
install( DIRECTORY  "${CMAKE_CURRENT_SOURCE_DIR}/test/"     DESTINATION test PATTERN "build" EXCLUDE )
install( DIRECTORY  "${CMAKE_CURRENT_SOURCE_DIR}/example/"  DESTINATION example PATTERN "build" EXCLUDE )
install( FILES README LICENSE DESTINATION . )

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
