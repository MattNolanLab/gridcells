
SET(CMAKE_CXX_FLAGS "-Wall -O3 ${CMAKE_CXX_FLAGS}")

add_library (gridcells SHARED fields.cpp include/fields.hpp)

set_target_properties(gridcells
    PROPERTIES
        VERSION 0.1.0)

set (CMAKE_SWIG_OUTDIR "${PROJECT_SOURCE_DIR}/gridcells")

set_source_files_properties(cppfields.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(cppfields.i PROPERTIES SWIG_FLAGS "-ignoremissing" )

swig_add_module (cppfields python cppfields.i)
swig_link_libraries (cppfields gridcells ${PYTHON_LIBRARIES})


# Copy the generated module files into-source. This way, the library can be
# used without a need to install it.
set (suff ${CMAKE_SHARED_MODULE_SUFFIX})
set (cppfields_so ${SWIG_MODULE_cppfields_REAL_NAME}${suff})
add_custom_command(
    TARGET _cppfields
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${cppfields_so} ${PROJECT_SOURCE_DIR}/gridcells/${cppfields_so}
    COMMENT "Copying ${cppfields_so} shared object to source tree.")

set_directory_properties (PROPERTIES
    ADDITIONAL_MAKE_CLEAN_FILES "${PROJECT_SOURCE_DIR}/gridcells/${cppfields_so}")


install (TARGETS gridcells DESTINATION lib)
install (TARGETS _cppfields DESTINATION ${PROJECT_SOURCE_DIR}/gridcells)
