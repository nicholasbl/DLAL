#include "doctest.h"

#include "test_common.h"

#include <glm/gtx/norm.hpp>

#include <iostream>

using namespace dct;

template <class T, size_t C, size_t R, class Function>
bool binary_test(std::array<T, C * R> const& a,
                 std::array<T, C * R> const& b,
                 Function                    f) {
    mat<T, C, R> our_side;
    {
        auto ma = mat<T, C, R>(a);
        auto mb = mat<T, C, R>(b);

        our_side = f(ma, mb);
    }

    glm::mat<C, R, T> their_side;

    {
        glm::mat<C, R, T> ga, gb;

        for (size_t i = 0; i < C * R; ++i) {
            glm::value_ptr(ga)[i] = a[i];
            glm::value_ptr(gb)[i] = b[i];
        }

        their_side = f(ga, gb);
    }

    return is_same(our_side, their_side);
}

template <class T, size_t C, size_t R, size_t C2, size_t R2, class Function>
bool binary_test_2(mat<T, C, R> const& a, mat<T, C2, R2> const& b, Function f) {

    auto our_side = f(a, b);

    glm::mat<C, R, T>   ga = to_glm(a);
    glm::mat<C2, R2, T> gb = to_glm(b);

    auto their_side = f(ga, gb);

    /*
    for (auto f : our_side) {
        std::cout << f << ", ";
    }
    std::cout << "-" << std::endl;

    for (float* f = glm::value_ptr(their_side);
         f < glm::value_ptr(their_side) + 16;
         f++) {
        std::cout << *f << ", ";
    }
    std::cout << "-" << std::endl;
    */

    return is_same(our_side, their_side);
}

template <class T, size_t C, size_t R, class Function>
bool binary_vector_2(mat<T, C, R> const& a, vec<T, R> const& b, Function f) {

    auto our_side = f(a, b);

    glm::mat<C, R, T> ga = to_glm(a);
    glm::vec<R, T>    gb = to_glm(b);

    auto their_side = f(ga, gb);


    //    for (auto f : our_side) {
    //        std::cout << f << ", ";
    //    }
    //    std::cout << "-" << std::endl;

    //    for (float* f = glm::value_ptr(their_side);
    //         f < glm::value_ptr(their_side) + R;
    //         f++) {
    //        std::cout << *f << ", ";
    //    }
    //    std::cout << "-" << std::endl;


    return is_same(our_side, their_side);
}

template <class T, size_t C, size_t R, class Function>
bool binary_scalar_test(std::array<T, C * R> const& a, T value, Function f) {
    mat<T, C, R> our_side;
    {
        auto ma = mat<T, C, R>(a);

        our_side = f(ma, value);
    }

    glm::mat<C, R, T> their_side;

    {
        glm::mat<C, R, T> ga;

        for (size_t i = 0; i < C * R; ++i) {
            glm::value_ptr(ga)[i] = a[i];
        }

        their_side = f(ga, value);
    }

    return is_same(our_side, their_side);
}

template <class T, size_t N>
std::array<T, N> make_filled(T value) {
    std::array<T, N> r;
    r.fill(value);
    return r;
}

TEST_CASE("Matrix Library") {

    SUBCASE("Constructors / Assignment") {
        mat4 m1(10);

        REQUIRE(is_same(m1, make_filled<float, 16>(10.0f)));

        mat4 m1a(mat3({ 1, 2, 3, 4, 5, 6, 7, 8, 9 }));

        REQUIRE(is_same(
            m1a, mat4({ 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 0, 0, 0, 0 })));


        std::array<float, 16> src = {
            { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }
        };

        mat4 m2(src);
        mat4 m2a(m2);

        REQUIRE(is_same(m2, src));
        REQUIRE(is_same(m2, m2a));

        std::array<float, 9> small_src = { { 1, 2, 3, 5, 6, 7, 9, 10, 11 } };

        mat3 m3(m2);

        REQUIRE(is_same(m3, small_src));
    }

    SUBCASE("Unary") {
        std::array<float, 16> src = {
            { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }
        };

        std::array<float, 16> nsrc = { { -1,
                                         -2,
                                         -3,
                                         -4,
                                         -5,
                                         -6,
                                         -7,
                                         -8,
                                         -9,
                                         -10,
                                         -11,
                                         -12,
                                         -13,
                                         -14,
                                         -15,
                                         -16 } };

        REQUIRE(is_same(-mat4(src), nsrc));
    }

    SUBCASE("Binary") {
        std::array<float, 16> a_src = {
            { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }
        };

        std::array<float, 16> b_src = {
            { 9, 2, 2, 4, 4, 5, 7, 8, 1, 9, 1, 2, 3, 4, 5, 6 }
        };

        REQUIRE(binary_test<float, 4, 4>(
            a_src, b_src, [](auto const& a, auto const& b) { return a + b; }));

        REQUIRE(binary_scalar_test<float, 4, 4>(
            a_src, 23.0f, [](auto const& a, auto const& b) { return a + b; }));

        REQUIRE(binary_scalar_test<float, 4, 4>(
            a_src, 23.0f, [](auto const& a, auto const& b) { return b + a; }));


        REQUIRE(binary_test<float, 4, 4>(
            a_src, b_src, [](auto const& a, auto const& b) { return a - b; }));

        REQUIRE(binary_scalar_test<float, 4, 4>(
            a_src, 23.0f, [](auto const& a, auto const& b) { return a - b; }));

        REQUIRE(binary_scalar_test<float, 4, 4>(
            a_src, 23.0f, [](auto const& a, auto const& b) { return b - a; }));


        REQUIRE(binary_test<float, 4, 4>(
            a_src, b_src, [](auto const& a, auto const& b) { return a * b; }));

        REQUIRE(binary_scalar_test<float, 4, 4>(
            a_src, 23.0f, [](auto const& a, auto const& b) { return a * b; }));

        REQUIRE(binary_scalar_test<float, 4, 4>(
            a_src, 23.0f, [](auto const& a, auto const& b) { return b * a; }));


        REQUIRE(binary_scalar_test<float, 4, 4>(
            a_src, 23.0f, [](auto const& a, auto const& b) { return a / b; }));

        REQUIRE(binary_scalar_test<float, 4, 4>(
            a_src, 23.0f, [](auto const& a, auto const& b) { return b / a; }));
    }

    SUBCASE("Binary - Mult") {
        {
            std::array<float, 12> a_src = {
                { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }
            };

            std::array<float, 12> b_src = {
                { 9, 2, 2, 4, 4, 5, 7, 8, 1, 9, 1, 2 }
            };

            REQUIRE(binary_test_2<float>(
                mat<float, 3, 4>(a_src),
                mat<float, 4, 3>(b_src),
                [](auto const& a, auto const& b) { return a * b; }));
        }

        {
            std::array<float, 9> a_src = { { 1, 2, 3, 4, 5, 6, 10, 11, 12 } };

            std::array<float, 9> b_src = { { 9, 2, 5, 7, 8, 1, 9, 1, 2 } };

            REQUIRE(binary_test_2<float>(
                mat3(a_src), mat3(b_src), [](auto const& a, auto const& b) {
                    return a * b;
                }));
        }

        {
            std::array<float, 16> a_src = {
                { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }
            };

            std::array<float, 4> b_src = { { 8, 1, 9, 1 } };

            REQUIRE(binary_vector_2<float>(
                mat4(a_src),
                dct::new_vec(b_src),
                [](auto const& a, auto const& b) { return a * b; }));
        }

        {
            std::array<float, 9> a_src = { { 1, 2, 3, 4, 5, 6, 7, 8, 9 } };

            std::array<float, 3> b_src = { { 8, 1, 9 } };

            REQUIRE(binary_vector_2<float>(
                mat3(a_src),
                dct::new_vec(b_src),
                [](auto const& a, auto const& b) { return a * b; }));
        }
    }
}
