#ifndef LINALG_MATRIX_DETAIL_H
#define LINALG_MATRIX_DETAIL_H

#include "vec.h"

#include <array>
#include <cstddef>

namespace dct {

namespace matrix_detail {

template <size_t N, class T, size_t M>
constexpr Vector<T, N> upgrade(Vector<T, M> const& v) {
    constexpr size_t C = vector_detail::cmin(N, M);
    static_assert(C <= 4);
    Vector<T, N> ret;
    if constexpr (C == 1) {
        ret.x = v.x;
    } else if constexpr (C == 2) {
        ret.x = v.x;
        ret.y = v.y;
    } else if constexpr (C == 3) {
        ret.x = v.x;
        ret.y = v.y;
        ret.z = v.z;
    } else if constexpr (C == 4) {
        ret.x = v.x;
        ret.y = v.y;
        ret.z = v.z;
        ret.w = v.w;
    }
    return ret;
}


#define MATRIX_UNARY(OP)                                                       \
    auto ret = m;                                                              \
    if constexpr (C == 1) {                                                    \
        ret[0] = OP m[0];                                                      \
    } else if constexpr (C == 2) {                                             \
        ret[0] = OP m[0];                                                      \
        ret[1] = OP m[1];                                                      \
    } else if constexpr (C == 3) {                                             \
        ret[0] = OP m[0];                                                      \
        ret[1] = OP m[1];                                                      \
        ret[2] = OP m[2];                                                      \
    } else if constexpr (C == 4) {                                             \
        ret[0] = OP m[0];                                                      \
        ret[1] = OP m[1];                                                      \
        ret[2] = OP m[2];                                                      \
        ret[3] = OP m[3];                                                      \
    }                                                                          \
    return ret;

#define MATRIX_BINARY_SCALAR_R(OP)                                             \
    Matrix<T, C, R> ret;                                                       \
    if constexpr (C == 1) {                                                    \
        ret[0] = m[0] OP scalar;                                               \
    } else if constexpr (C == 2) {                                             \
        ret[0] = m[0] OP scalar;                                               \
        ret[1] = m[1] OP scalar;                                               \
    } else if constexpr (C == 3) {                                             \
        ret[0] = m[0] OP scalar;                                               \
        ret[1] = m[1] OP scalar;                                               \
        ret[2] = m[2] OP scalar;                                               \
    } else if constexpr (C == 4) {                                             \
        ret[0] = m[0] OP scalar;                                               \
        ret[1] = m[1] OP scalar;                                               \
        ret[2] = m[2] OP scalar;                                               \
        ret[3] = m[3] OP scalar;                                               \
    }                                                                          \
    return ret;

#define MATRIX_BINARY_SCALAR_L(OP)                                             \
    Matrix<T, C, R> ret;                                                       \
    if constexpr (C == 1) {                                                    \
        ret[0] = scalar OP m[0];                                               \
    } else if constexpr (C == 2) {                                             \
        ret[0] = scalar OP m[0];                                               \
        ret[1] = scalar OP m[1];                                               \
    } else if constexpr (C == 3) {                                             \
        ret[0] = scalar OP m[0];                                               \
        ret[1] = scalar OP m[1];                                               \
        ret[2] = scalar OP m[2];                                               \
    } else if constexpr (C == 4) {                                             \
        ret[0] = scalar OP m[0];                                               \
        ret[1] = scalar OP m[1];                                               \
        ret[2] = scalar OP m[2];                                               \
        ret[3] = scalar OP m[3];                                               \
    }                                                                          \
    return ret;

#define MATRIX_BINARY(OP)                                                      \
    Matrix<T, C, R> ret;                                                       \
    if constexpr (C == 1) {                                                    \
        ret[0] = m[0] OP o[0];                                                 \
    } else if constexpr (C == 2) {                                             \
        ret[0] = m[0] OP o[0];                                                 \
        ret[1] = m[1] OP o[1];                                                 \
    } else if constexpr (C == 3) {                                             \
        ret[0] = m[0] OP o[0];                                                 \
        ret[1] = m[1] OP o[1];                                                 \
        ret[2] = m[2] OP o[2];                                                 \
    } else if constexpr (C == 4) {                                             \
        ret[0] = m[0] OP o[0];                                                 \
        ret[1] = m[1] OP o[1];                                                 \
        ret[2] = m[2] OP o[2];                                                 \
        ret[3] = m[3] OP o[3];                                                 \
    }                                                                          \
    return ret;

#define MATRIX_IN_PLACE(OP)                                                    \
    if constexpr (C == 1) {                                                    \
        m[0] OP o[0];                                                          \
    } else if constexpr (C == 2) {                                             \
        m[0] OP o[0];                                                          \
        m[1] OP o[1];                                                          \
    } else if constexpr (C == 3) {                                             \
        m[0] OP o[0];                                                          \
        m[1] OP o[1];                                                          \
        m[2] OP o[2];                                                          \
    } else if constexpr (C == 4) {                                             \
        m[0] OP o[0];                                                          \
        m[1] OP o[1];                                                          \
        m[2] OP o[2];                                                          \
        m[3] OP o[3];                                                          \
    }                                                                          \
    return m;

#define MATRIX_IN_PLACE_SCALAR_R(OP)                                           \
    if constexpr (C == 1) {                                                    \
        m[0] OP scalar;                                                        \
    } else if constexpr (C == 2) {                                             \
        m[0] OP scalar;                                                        \
        m[1] OP scalar;                                                        \
    } else if constexpr (C == 3) {                                             \
        m[0] OP scalar;                                                        \
        m[1] OP scalar;                                                        \
        m[2] OP scalar;                                                        \
    } else if constexpr (C == 4) {                                             \
        m[0] OP scalar;                                                        \
        m[1] OP scalar;                                                        \
        m[2] OP scalar;                                                        \
        m[3] OP scalar;                                                        \
    }                                                                          \
    return m;

#define MATRIX_BINARY_BOOL(OP)                                                 \
    Matrix<bool, C, R> ret;                                                    \
    if constexpr (C == 1) {                                                    \
        ret[0] = m[0] OP o[0];                                                 \
    } else if constexpr (C == 2) {                                             \
        ret[0] = m[0] OP o[0];                                                 \
        ret[1] = m[1] OP o[1];                                                 \
    } else if constexpr (C == 3) {                                             \
        ret[0] = m[0] OP o[0];                                                 \
        ret[1] = m[1] OP o[1];                                                 \
        ret[2] = m[2] OP o[2];                                                 \
    } else if constexpr (C == 4) {                                             \
        ret[0] = m[0] OP o[0];                                                 \
        ret[1] = m[1] OP o[1];                                                 \
        ret[2] = m[2] OP o[2];                                                 \
        ret[3] = m[3] OP o[3];                                                 \
    }                                                                          \
    return ret;

} // namespace matrix_detail

} // namespace dct

#endif // LINALG_MATRIX_DETAIL_H
