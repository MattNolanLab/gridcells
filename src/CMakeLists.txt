
SET(CMAKE_CXX_FLAGS "-Wall -O3 -g ${CMAKE_CXX_FLAGS}")

add_library (gridcells SHARED
    include/gridsCore.hpp
    include/arena.hpp arena.cpp
    fields.cpp include/fields.hpp
)

set_target_properties(gridcells
    PROPERTIES
        VERSION 0.1.0)


# Copy the generated module files into-source. This way, the library can be
# used without a need to install it.
set (suff ${CMAKE_SHARED_MODULE_SUFFIX})

macro (COPY_PYTHON_SO modname)
    set (mod_so ${SWIG_MODULE_${modname}_REAL_NAME}${suff})
    if (${mod_so} STREQUAL "")
        message (FATAL_ERROR
            "Cannot add python module copy command: '${modname}' does not exist!")
    endif (${mod_so} STREQUAL "")

    add_custom_command(
        TARGET _${modname}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${mod_so} ${PROJECT_SOURCE_DIR}/gridcells/${mod_so}
        COMMENT "Copying ${mod_so} shared object to source tree."
    )
    get_directory_property(extra_clean_files ADDITIONAL_MAKE_CLEAN_FILES )
    set_directory_properties (PROPERTIES
        ADDITIONAL_MAKE_CLEAN_FILES 
            "${extra_clean_files};${PROJECT_SOURCE_DIR}/gridcells/${mod_so}"
    )
endmacro(COPY_PYTHON_SO)


macro (COPY_PYTHON_WRAPPER_MOD modname)
    if (${modname} STREQUAL "")
        message (FATAL_ERROR
            "Cannot add python module copy command: modname is empty!")
    endif (${modname} STREQUAL "")
    set (filename ${modname}.py)
    add_custom_command(
        TARGET _${modname}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${filename} ${PROJECT_SOURCE_DIR}/gridcells/${filename}
        COMMENT "Copying ${filename} to source tree."
    )
    get_directory_property(extra_clean_files ADDITIONAL_MAKE_CLEAN_FILES )
    set_directory_properties (PROPERTIES
        ADDITIONAL_MAKE_CLEAN_FILES 
            "${extra_clean_files};${PROJECT_SOURCE_DIR}/gridcells/${filename}"
    )
endmacro (COPY_PYTHON_WRAPPER_MOD)

##############################################################################

set (PYMOD_SWIG_PROPS "-builtin;-O;-ignoremissing")
set (PYMOD_SWIG_SOURCES "fields.i;gridsCore.i;arena.i")

set_source_files_properties(${PYMOD_SWIG_SOURCES} PROPERTIES CPLUSPLUS ON)
set_source_files_properties(${PYMOD_SWIG_SOURCES} PROPERTIES SWIG_FLAGS ${PYMOD_SWIG_PROPS})

set (SWIG_MODULE_fields_EXTRA_DEPS
    "${COMMON_SWIG_DEPS}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/fields.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/gridsCore.hpp")

set (SWIG_MODULE_gridsCore_EXTRA_DEPS
    "${COMMON_SWIG_DEPS}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/gridsCore.hpp")

swig_add_module (fields python fields.i)
swig_add_module (gridsCore python gridsCore.i)
swig_add_module (arena python arena.i)
swig_link_libraries (fields gridcells ${PYTHON_LIBRARIES})
swig_link_libraries (gridsCore gridcells ${PYTHON_LIBRARIES})
swig_link_libraries (arena gridcells ${PYTHON_LIBRARIES})

copy_python_so(fields)
copy_python_so(gridsCore)
copy_python_so(arena)


install (TARGETS gridcells DESTINATION lib)
install (TARGETS _fields DESTINATION ${PROJECT_SOURCE_DIR}/gridcells)
