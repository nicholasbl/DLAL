cmake_minimum_required(VERSION 3.13)

project(libdlal
    VERSION 0.0.1
    DESCRIPTION "A header only SSE4.1 powered graphics-focused linear algebra library"
    LANGUAGES CXX)

option(DCTLIN_BUILD_TESTS "Build test executables"  OFF)
option(DCTLIN_BUILD_DOCS  "Build documentation"     OFF)

add_library(libdlal INTERFACE)

target_compile_features(libdlal INTERFACE cxx_std_17)

target_compile_options(libdlal INTERFACE -mavx)

target_sources(libdlal
    INTERFACE
    include/mat.h
    include/mat_detail.h
    include/mat_operations.h
    include/mat_transforms.h
    include/packed_mat.h
    include/packed_vec.h
    include/quat.h
    include/transformation.h
    include/vec.h
    )

target_include_directories(libdlal INTERFACE include/)


if (DCTLIN_BUILD_TESTS)
    find_package(GLM 0.9.2)

    add_executable(dctlin_tests
                    test/main.cpp
                    test/matrix.cpp
                    test/matrix_operations.cpp
                    test/matrix_transform.cpp
                    test/quat.cpp
                    test/tmatrix.cpp
                    test/vector.cpp
                    )
                    
    target_include_directories(dctlin_tests SYSTEM PUBLIC ${GLM_INCLUDE_DIRS})
    
    list(APPEND cppflags -fsanitize=address)
                    
    target_link_libraries(dctlin_tests libdlal)
    target_compile_options(dctlin_tests PRIVATE ${cppflags})
    target_link_options(dctlin_tests PRIVATE ${cppflags})
endif()

if (DCTLIN_BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    
    add_custom_target( documentation ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
        )
endif()
