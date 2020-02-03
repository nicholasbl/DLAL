#ifndef LINALG_VECTOR_TRIG_H
#define LINALG_VECTOR_TRIG_H

#include "packed_vec.h"

#include <cmath>

namespace dct {

#define VECTOR_OP(OP)                                                          \
    vec<T, N> ret;                                                             \
    if constexpr (N == 1) {                                                    \
        ret.x = OP(a.x);                                                       \
    } else if constexpr (N == 2) {                                             \
        ret.x = OP(a.x);                                                       \
        ret.y = OP(a.y);                                                       \
    } else if constexpr (N == 3) {                                             \
        ret.x = OP(a.x);                                                       \
        ret.y = OP(a.y);                                                       \
        ret.z = OP(a.z);                                                       \
    } else if constexpr (N == 4) {                                             \
        ret.x = OP(a.x);                                                       \
        ret.y = OP(a.y);                                                       \
        ret.z = OP(a.z);                                                       \
        ret.w = OP(a.w);                                                       \
    }                                                                          \
    return ret;

#define SPEC_VECTOR_OP(OP, SIMD)                                               \
    inline vec4 OP(vec4 const& a) { return SIMD(a); }                          \
    inline vec3 OP(vec3 const& a) {                                            \
        using namespace vector_detail;                                         \
        return v4to3(OP(v3to4(a)));                                            \
    }                                                                          \
    inline vec2 OP(vec2 const& a) {                                            \
        using namespace vector_detail;                                         \
        return v4to2(OP(v2to4(a)));                                            \
    }


template <class T, int N>
vec<T, N> sqrt(vec<T, N> const& a){ VECTOR_OP(std::sqrt) }

SPEC_VECTOR_OP(sqrt, _mm_sqrt_ps);

template <class T, int N>
vec<T, N> acos(vec<T, N> const& a) {
    VECTOR_OP(std::acos)
}

template <class T, int N>
vec<T, N> cos(vec<T, N> const& a) {
    VECTOR_OP(std::cos)
}

template <class T, int N>
vec<T, N> asin(vec<T, N> const& a) {
    VECTOR_OP(std::asin)
}

template <class T, int N>
vec<T, N> sin(vec<T, N> const& a) {
    VECTOR_OP(std::sin)
}

template <class T, int N>
vec<T, N> atan(vec<T, N> const& a) {
    VECTOR_OP(std::atan)
}

template <class T, int N>
vec<T, N> tan(vec<T, N> const& a) {
    VECTOR_OP(std::tan)
}

template <class T, int N>
vec<T, N> exp(vec<T, N> const& a) {
    VECTOR_OP(std::exp)
}

template <class T, int N>
vec<T, N> log(vec<T, N> const& a) {
    VECTOR_OP(std::log)
}

#undef VECTOR_OP
#undef SPEC_VECTOR_OP

} // namespace dct

#endif // LINALG_VECTOR_TRIG_H
