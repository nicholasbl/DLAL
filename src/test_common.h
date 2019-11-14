#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include "include/mat.h"
#include "include/vec.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

template <class T, size_t C, size_t R>
bool is_same(dct::Matrix<T, C, R> const& a, std::array<T, C * R> const& b) {
    // weirdness due to how glm specifies their templates
    // static_assert(C == M);
    for (size_t i = 0; i < C * R; ++i) {
        T delta = std::abs(a.data()[i] - b[i]);
        if (std::numeric_limits<T>::epsilon() < delta) {
            return false;
        }
    }
    return true;
}

template <class T, size_t R>
bool is_same(dct::Vector<T, R> const& a, glm::vec<int(R), T> const& b) {
    // weirdness due to how glm specifies their templates
    // static_assert(C == M);
    for (size_t i = 0; i < R; ++i) {
        T delta = std::abs(a[i] - b[i]);
        if (std::numeric_limits<T>::epsilon() < delta) {
            return false;
        }
    }
    return true;
}

template <class T, size_t C, size_t R>
bool is_same(dct::Matrix<T, C, R> const&        a,
             glm::mat<int(C), int(R), T> const& b) {
    // weirdness due to how glm specifies their templates
    // static_assert(C == M);
    for (size_t i = 0; i < C * R; ++i) {
        T delta =
            std::abs(a.data()[i] - glm::value_ptr(b)[static_cast<int>(i)]);
        if (std::numeric_limits<T>::epsilon() < delta) {
            return false;
        }
    }
    return true;
}

template <class T, size_t C, size_t R>
bool is_same(dct::Matrix<T, C, R> const& a, dct::Matrix<T, C, R> const& b) {
    return is_equal(a, b);
}

template <class T>
bool is_same(T const& a, T const& b) {
    return a == b;
}


template <class T, size_t N>
bool operator==(dct::Vector<T, N> const& a, glm::vec<N, T> const& b) {
    return is_same(a, b);
}

#endif // TEST_COMMON_H
