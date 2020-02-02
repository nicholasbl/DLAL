#ifndef LINALG_MATRIX_DETAIL_H
#define LINALG_MATRIX_DETAIL_H

#include "vec.h"

#include <array>
#include <cstddef>

namespace dct {

namespace matrix_detail {

template <int N, class T, int M>
constexpr vec<T, N> upgrade(vec<T, M> const& v) {
    constexpr size_t C = vector_detail::cmin(N, M);
    static_assert(C <= 4);
    vec<T, N> ret;
    if constexpr (C == 1) {
        return vec<T, N>{ v.x };
    } else if constexpr (C == 2) {
        return vec<T, N>{ v.x, v.y };
    } else if constexpr (C == 3) {
        return vec<T, N>{ v.x, v.y, v.z };
    } else if constexpr (C == 4) {
        return vec<T, N>{ v.x, v.y, v.z, v.w };
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
    mat<T, C, R> ret;                                                          \
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
    mat<T, C, R> ret;                                                          \
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
    mat<T, C, R> ret;                                                          \
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
    mat<int, C, R> ret;                                                        \
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

// =============================================================================

template <class T, size_t N, size_t AT>
vec<T, N> get_identity_vec() {
    if constexpr (N == 1) {
        return vec<T, N>(0);
    } else if constexpr (N == 1) {
        return vec<T, N>{ 1 };
    } else if constexpr (N == 2) {
        if constexpr (AT == 0) {
            return vec<T, N>{ 1, 0 };
        } else if constexpr (AT == 1) {
            return vec<T, N>{ 0, 1 };
        }
    } else if constexpr (N == 3) {
        if constexpr (AT == 0) {
            return vec<T, N>{ 1, 0, 0 };
        } else if constexpr (AT == 1) {
            return vec<T, N>{ 0, 1, 0 };
        } else if constexpr (AT == 2) {
            return vec<T, N>{ 0, 0, 1 };
        }
    } else if constexpr (N == 4) {
        if constexpr (AT == 0) {
            return vec<T, N>{ 1, 0, 0, 0 };
        } else if constexpr (AT == 1) {
            return vec<T, N>{ 0, 1, 0, 0 };
        } else if constexpr (AT == 2) {
            return vec<T, N>{ 0, 0, 1, 0 };
        } else if constexpr (AT == 3) {
            return vec<T, N>{ 0, 0, 0, 1 };
        }
    }
}

template <class T, size_t C, size_t R>
constexpr auto get_identity_storage() {
    using ColumnType = vec<T, R>;

    std::array<ColumnType, C> ret;

    if constexpr (C == 1) {
        ret[0] = get_identity_vec<T, R, 0>();
    } else if constexpr (C == 2) {
        ret[0] = get_identity_vec<T, R, 0>();
        ret[1] = get_identity_vec<T, R, 1>();
    } else if constexpr (C == 3) {
        ret[0] = get_identity_vec<T, R, 0>();
        ret[1] = get_identity_vec<T, R, 1>();
        ret[2] = get_identity_vec<T, R, 2>();
    } else if constexpr (C == 4) {
        ret[0] = get_identity_vec<T, R, 0>();
        ret[1] = get_identity_vec<T, R, 1>();
        ret[2] = get_identity_vec<T, R, 2>();
        ret[3] = get_identity_vec<T, R, 3>();
    }

    return ret;
}

} // namespace matrix_detail

} // namespace dct

#endif // LINALG_MATRIX_DETAIL_H
