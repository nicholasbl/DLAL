#ifndef LINALG_VECTOR_TRIG_H
#define LINALG_VECTOR_TRIG_H

#include "vec.h"

#include <cmath>

namespace dct {

#define VECTOR_OP(OP)                                                          \
    Vector<T, N> ret;                                                          \
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

template <class T, size_t N>
Vector<T, N> acos(Vector<T, N> const& a) {
    VECTOR_OP(std::acos)
}
template <class T, size_t N>
Vector<T, N> cos(Vector<T, N> const& a) {
    VECTOR_OP(std::cos)
}
template <class T, size_t N>
Vector<T, N> asin(Vector<T, N> const& a) {
    VECTOR_OP(std::asin)
}
template <class T, size_t N>
Vector<T, N> sin(Vector<T, N> const& a) {
    VECTOR_OP(std::sin)
}
template <class T, size_t N>
Vector<T, N> atan(Vector<T, N> const& a) {
    VECTOR_OP(std::atan)
}
template <class T, size_t N>
Vector<T, N> tan(Vector<T, N> const& a) {
    VECTOR_OP(std::tan)
}
template <class T, size_t N>
Vector<T, N> exp(Vector<T, N> const& a) {
    VECTOR_OP(std::exp)
}
template <class T, size_t N>
Vector<T, N> log(Vector<T, N> const& a) {
    VECTOR_OP(std::log)
}

#undef VECTOR_OP

} // namespace dct

#endif // LINALG_VECTOR_TRIG_H
