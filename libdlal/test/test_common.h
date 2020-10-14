#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include "mat.h"
#include "packed_vec.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

template <class T, size_t C, size_t R>
glm::mat<int(C), int(R), T> to_glm(dlal::mat<T, C, R> const& a) {
    glm::mat<int(C), int(R), T> ret;

    for (size_t i = 0; i < C; ++i) {
        for (size_t j = 0; j < R; ++j) {
            ret[i][j] = a[i][j];
        }
    }
    return ret;
}

template <class T, size_t N>
glm::vec<N, T> to_glm(dlal::vec<T, N> const& a) {
    glm::vec<N, T> ret;

    for (size_t i = 0; i < N; ++i) {
        ret[i] = a[i];
    }
    return ret;
}

template <class T, size_t C, size_t R>
bool is_same(dlal::mat<T, C, R> const&   a,
             std::array<T, C * R> const& b,
             T limit = std::numeric_limits<T>::epsilon()) {

    for (size_t i = 0; i < C; ++i) {
        for (size_t j = 0; j < R; ++j) {
            T delta = std::abs(a[i][j] - b[i * R + j]);
            if (limit < delta) { return false; }
        }
    }
    return true;
}

template <class T, size_t R>
bool is_same(dlal::vec<T, R> const& a, glm::vec<int(R), T> const& b) {
    // weirdness due to how glm specifies their templates
    // static_assert(C == M);
    for (size_t i = 0; i < R; ++i) {
        T delta = std::abs(a[i] - b[i]);
        if (std::numeric_limits<T>::epsilon() < delta) { return false; }
    }
    return true;
}

template <class T, size_t C, size_t R>
bool is_same(dlal::mat<T, C, R> const&          a,
             glm::mat<int(C), int(R), T> const& b,
             T limit = std::numeric_limits<T>::epsilon()) {

    for (size_t i = 0; i < C; ++i) {
        for (size_t j = 0; j < R; ++j) {
            T delta = std::abs(a[i][j] - b[i][j]);
            if (limit < delta) { return false; }
        }
    }
    return true;
}

template <class T, size_t C, size_t R>
bool is_same(dlal::mat<T, C, R> const& a, dlal::mat<T, C, R> const& b) {
    return is_equal(a, b);
}

template <class T>
bool is_same(T const& a, T const& b) {
    return a == b;
}

template <class T, size_t C, size_t R>
bool operator==(dlal::mat<T, C, R> const&          a,
                glm::mat<int(C), int(R), T> const& b) {
    return is_same(a, b);
}

template <class T, size_t N>
bool operator==(dlal::vec<T, N> const& a, glm::vec<N, T> const& b) {
    return is_same(a, b);
}

#endif // TEST_COMMON_H
