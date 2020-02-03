cmake_minimum_required(VERSION 3.13)

project(libdctlin
    VERSION 0.0.1
    DESCRIPTION "A header only SSE4.1 powered graphics-focused linear algebra library"
    LANGUAGES CXX)

option(DCTLIN_BUILD_TESTS "Build test executables" OFF)

add_library(${PROJECT_NAME} INTERFACE)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

target_sources(${PROJECT_NAME}
    INTERFACE
    include/linear_algebra.h
    include/mat.h
    include/mat_detail.h
    include/mat_operations.h
    include/mat_transforms.h
    include/packed_mat.h
    include/packed_vec.h
    include/quat.h
    include/tmatrix.h
    include/vec.h
    include/vec_trig.h)

target_include_directories(${PROJECT_NAME} INTERFACE include/)


if (DCTLIN_BUILD_TESTS MATCHES "ON")
    find_package(GLM 0.9.2)

    add_executable(dctlin_tests
                    test/main.cpp
                    test/matrix.cpp
                    test/matrix_operations.cpp
                    test/matrix_transform.cpp
                    test/quat.cpp
                    test/tmatrix.cpp
                    test/vector.cpp)
                    
    target_include_directories(dctlin_tests SYSTEM PUBLIC ${GLM_INCLUDE_DIRS})
    
    list(APPEND cppflags -fsanitize=address)
                    
    target_link_libraries(dctlin_tests ${PROJECT_NAME})
    target_compile_options(dctlin_tests PRIVATE ${cppflags})
    target_link_options(dctlin_tests PRIVATE ${cppflags})
endif()
