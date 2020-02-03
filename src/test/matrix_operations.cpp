#include "doctest.h"
#include "mat_operations.h"

#include "test_common.h"

#include <glm/gtx/norm.hpp>

using namespace dct;


template <class T, size_t C, size_t R, class Function>
bool matrix_op_test(std::array<T, C * R> const& a, Function f) {
    auto our_side = f(mat<T, C, R>(a));

    glm::mat<C, R, T> ga;

    for (size_t i = 0; i < C * R; ++i) {
        glm::value_ptr(ga)[i] = a[i];
    }

    auto their_side = f(ga);

    /*
    for (auto f : our_side) {
        std::cerr << f << ", ";
    }
    std::cerr << "E" << std::endl;

    for (float* f = glm::value_ptr(their_side);
         f < glm::value_ptr(their_side) + (R * C);
         f++) {
        std::cerr << *f << ", ";
    }
    std::cerr << "E" << std::endl;
    */

    return is_same(our_side, their_side);
}


TEST_CASE("Matrix Operations Library") {
    SUBCASE("Determinant") {
        std::array<float, 9> a_src = { { 1, 2, 3, 0, 5, 0, 7, 8, 9 } };

        REQUIRE(matrix_op_test<float, 3, 3>(a_src, [](auto const& m) {
            using namespace glm;
            return determinant(m);
        }));

        {
            std::array<float, 16> b_src = {
                { 1, 2, 3, 4, 0, 1, 0, 2, 0, 0, 1, 1, 0, 0, 0, 1 }
            };

            REQUIRE(matrix_op_test<float, 4, 4>(b_src, [](auto const& m) {
                using namespace glm;
                return determinant(m);
            }));
        }

        {
            std::array<float, 16> b_src = {
                { 1, 2, 3, 4, 8, 1, 0, 2, 1, 1, 1, 1, 4, 0, -1, 1 }
            };

            REQUIRE(matrix_op_test<float, 4, 4>(b_src, [](auto const& m) {
                using namespace glm;
                return determinant(m);
            }));
        }
    }


    SUBCASE("Inverse") {
        std::array<float, 9> a_src = { { 1, 2, 3, 0, 5, 0, 7, 8, 9 } };

        REQUIRE(matrix_op_test<float, 3, 3>(a_src, [](auto const& m) {
            using namespace glm;
            return inverse(m);
        }));

        {
            std::array<float, 16> b_src = {
                { 1, 2, 3, 4, 0, 1, 0, 2, 0, 0, 1, 1, 0, 0, 0, 1 }
            };

            REQUIRE(matrix_op_test<float, 4, 4>(b_src, [](auto const& m) {
                using namespace glm;
                return inverse(m);
            }));
        }

        {
            std::array<float, 16> b_src = {
                { 1, 2, 3, 4, 8, 1, 0, 2, 1, 1, 1, 1, 4, 0, -1, 1 }
            };

            REQUIRE(matrix_op_test<float, 4, 4>(b_src, [](auto const& m) {
                using namespace glm;
                return inverse(m);
            }));
        }
    }
}
