#ifndef LINALG_VECTOR_DETAIL_H
#define LINALG_VECTOR_DETAIL_H

#include <cstddef>

#include <smmintrin.h>

namespace dct {

namespace vector_detail {

template <class T, unsigned N>
using vec __attribute__((vector_size(sizeof(T) * N))) = T;

using ivec4 = vec<int, 4>;
using vec4  = vec<float, 4>;

#ifdef __clang__
#define SWIZZLE(A, X, Y, Z, W)                                                 \
    __builtin_shufflevector(A, dct::vector_detail::ivec4{ X, Y, Z, W })
#define SHUFFLE(A, B, X, Y, Z, W) __builtin_shufflevector(A, B, X, Y, Z, W)
#else
#define SWIZZLE(A, X, Y, Z, W)                                                 \
    __builtin_shuffle(A, dct::vector_detail::ivec4{ X, Y, Z, W })
#define SHUFFLE(A, B, X, Y, Z, W)                                              \
    __builtin_shuffle(A, B, dct::vector_detail::ivec4{ X, Y, Z, W })
#endif

template <int x, int y = x, int z = x, int w = x>
__attribute__((always_inline)) auto swizzle(vec4 vec) {
    return SWIZZLE(vec, x, y, z, w);
}

template <>
__attribute__((always_inline)) inline auto swizzle<0, 0, 2, 2>(vec4 vec) {
    return _mm_moveldup_ps(vec);
}

template <>
__attribute__((always_inline)) inline auto swizzle<1, 1, 3, 3>(vec4 vec) {
    return _mm_movehdup_ps(vec);
}


template <int x, int y = x, int z = x, int w = x>
__attribute__((always_inline)) auto shuffle(vec4 a, vec4 b) {
    return SHUFFLE(a, b, x, y, z, w);
}

template <>
__attribute__((always_inline)) inline auto shuffle<0, 1, 0, 1>(vec4 a, vec4 b) {
    return _mm_movelh_ps(a, b);
}

template <>
__attribute__((always_inline)) inline auto shuffle<2, 3, 2, 3>(vec4 a, vec4 b) {
    return _mm_movehl_ps(b, a);
}

inline float dot(vec4 a, vec4 b) {
    // documentation and testing shows this is faster than dp or hadd
    __m128 mult, shuf, sums;
    mult = _mm_mul_ps(a, b);
    shuf = _mm_movehdup_ps(mult);
    sums = _mm_add_ps(mult, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

inline vec4 dot_vec(vec4 a, vec4 b) {
    // documentation and testing shows this is faster than dp or hadd
    __m128 mult, shuf, sums;
    mult = _mm_mul_ps(a, b);
    shuf = _mm_movehdup_ps(mult);
    sums = _mm_add_ps(mult, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    return _mm_add_ss(sums, shuf);
}


// ==================

// constexpr versions of these common functions

template <class T>
constexpr T cmin(T a, T b) {
    return (a < b) ? a : b;
}

template <class T>
constexpr T cmax(T a, T b) {
    return (a > b) ? a : b;
}

template <class T>
constexpr T cabs(T a) {
    return (a > T(0)) ? a : -a;
}

} // namespace vector_detail

// instead of using an array to access, we use the union. This reduces the
// number of instructions even in debug mode.

#define VECTOR_UNARY(OP)                                                       \
    Vector<T, N> ret;                                                          \
    if constexpr (N == 1) {                                                    \
        ret.x = OP v.x;                                                        \
    } else if constexpr (N == 2) {                                             \
        ret.x = OP v.x;                                                        \
        ret.y = OP v.y;                                                        \
    } else if constexpr (N == 3) {                                             \
        ret.x = OP v.x;                                                        \
        ret.y = OP v.y;                                                        \
        ret.z = OP v.z;                                                        \
    } else if constexpr (N == 4) {                                             \
        ret.x = OP v.x;                                                        \
        ret.y = OP v.y;                                                        \
        ret.z = OP v.z;                                                        \
        ret.w = OP v.w;                                                        \
    }                                                                          \
    return ret;

#define VECTOR_BINARY_SCALAR_R(OP)                                             \
    Vector<T, N> ret;                                                          \
    if constexpr (N == 1) {                                                    \
        ret.x = v.x OP scalar;                                                 \
    } else if constexpr (N == 2) {                                             \
        ret.x = v.x OP scalar;                                                 \
        ret.y = v.y OP scalar;                                                 \
    } else if constexpr (N == 3) {                                             \
        ret.x = v.x OP scalar;                                                 \
        ret.y = v.y OP scalar;                                                 \
        ret.z = v.z OP scalar;                                                 \
    } else if constexpr (N == 4) {                                             \
        ret.x = v.x OP scalar;                                                 \
        ret.y = v.y OP scalar;                                                 \
        ret.z = v.z OP scalar;                                                 \
        ret.w = v.w OP scalar;                                                 \
    }                                                                          \
    return ret;

#define VECTOR_BINARY_SCALAR_L(OP)                                             \
    Vector<T, N> ret;                                                          \
    if constexpr (N == 1) {                                                    \
        ret.x = scalar OP v.x;                                                 \
    } else if constexpr (N == 2) {                                             \
        ret.x = scalar OP v.x;                                                 \
        ret.y = scalar OP v.y;                                                 \
    } else if constexpr (N == 3) {                                             \
        ret.x = scalar OP v.x;                                                 \
        ret.y = scalar OP v.y;                                                 \
        ret.z = scalar OP v.z;                                                 \
    } else if constexpr (N == 4) {                                             \
        ret.x = scalar OP v.x;                                                 \
        ret.y = scalar OP v.y;                                                 \
        ret.z = scalar OP v.z;                                                 \
        ret.w = scalar OP v.w;                                                 \
    }                                                                          \
    return ret;

#define VECTOR_BINARY(OP)                                                      \
    Vector<T, N> ret;                                                          \
    if constexpr (N == 1) {                                                    \
        ret.x = v.x OP o.x;                                                    \
    } else if constexpr (N == 2) {                                             \
        ret.x = v.x OP o.x;                                                    \
        ret.y = v.y OP o.y;                                                    \
    } else if constexpr (N == 3) {                                             \
        ret.x = v.x OP o.x;                                                    \
        ret.y = v.y OP o.y;                                                    \
        ret.z = v.z OP o.z;                                                    \
    } else if constexpr (N == 4) {                                             \
        ret.x = v.x OP o.x;                                                    \
        ret.y = v.y OP o.y;                                                    \
        ret.z = v.z OP o.z;                                                    \
        ret.w = v.w OP o.w;                                                    \
    }                                                                          \
    return ret;


#define VECTOR_IN_PLACE(OP)                                                    \
    if constexpr (N == 1) {                                                    \
        v.x OP o.x;                                                            \
    } else if constexpr (N == 2) {                                             \
        v.x OP o.x;                                                            \
        v.y OP o.y;                                                            \
    } else if constexpr (N == 3) {                                             \
        v.x OP o.x;                                                            \
        v.y OP o.y;                                                            \
        v.z OP o.z;                                                            \
    } else if constexpr (N == 4) {                                             \
        v.x OP o.x;                                                            \
        v.y OP o.y;                                                            \
        v.z OP o.z;                                                            \
        v.w OP o.w;                                                            \
    }                                                                          \
    return v;

#define VECTOR_IN_PLACE_SCALAR_R(OP)                                           \
    if constexpr (N == 1) {                                                    \
        v.x OP scalar;                                                         \
    } else if constexpr (N == 2) {                                             \
        v.x OP scalar;                                                         \
        v.y OP scalar;                                                         \
    } else if constexpr (N == 3) {                                             \
        v.x OP scalar;                                                         \
        v.y OP scalar;                                                         \
        v.z OP scalar;                                                         \
    } else if constexpr (N == 4) {                                             \
        v.x OP scalar;                                                         \
        v.y OP scalar;                                                         \
        v.z OP scalar;                                                         \
        v.w OP scalar;                                                         \
    }                                                                          \
    return v;

#define VECTOR_BINARY_BOOL(OP)                                                 \
    Vector<bool, N> ret;                                                       \
    if constexpr (N == 1) {                                                    \
        ret.x = v.x OP o.x;                                                    \
    } else if constexpr (N == 2) {                                             \
        ret.x = v.x OP o.x;                                                    \
        ret.y = v.y OP o.y;                                                    \
    } else if constexpr (N == 3) {                                             \
        ret.x = v.x OP o.x;                                                    \
        ret.y = v.y OP o.y;                                                    \
        ret.z = v.z OP o.z;                                                    \
    } else if constexpr (N == 4) {                                             \
        ret.x = v.x OP o.x;                                                    \
        ret.y = v.y OP o.y;                                                    \
        ret.z = v.z OP o.z;                                                    \
        ret.w = v.w OP o.w;                                                    \
    }                                                                          \
    return ret;

} // namespace dct

#endif // LINALG_VECTOR_DETAIL_H
