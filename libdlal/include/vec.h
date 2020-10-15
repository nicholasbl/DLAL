#ifndef LINALG_VECTOR_DETAIL_H
#define LINALG_VECTOR_DETAIL_H

#include <array>
#include <cmath>
#include <cstddef>

#include <smmintrin.h>


namespace dlal {

/// Notes:
/// Vector equality uses -1 (all bits set)! If you want to use conditionals,
/// look at the is_all or is_any set of functions! The C++ conversion rules from
/// int to bool use b = i == 0; Any other value is true, which helps.
///

#ifndef __SSE4_1__
#    error SSE 4.1 Support is required
#endif

#ifndef __AVX__
#    error AVX Support is required
#endif

#ifndef __clang__
#    error Only the CLANG compiler is supported at this time
#endif

#ifndef __x86_64__
#    warning Only tested on 64 bit archs at this time
#endif

// Define how OpenCL views true and false
#define VEC_TRUE  -1
#define VEC_FALSE 0

// We require extended vectors
static_assert(__has_attribute(ext_vector_type),
              "We require extended vector types.");

/// @{
/// \brief A templated typedef using extended vector types. Storage size is
/// implementation defined.
///
/// Does NOT initialize on default construction:
/// \code
/// vec4 v;
/// \endcode
/// This does not ensure components are zero.
///
/// \code
/// vec4 v = {};
/// \endcode
/// This does.
///
/// Other rules:
/// \code
/// vec4 v(3); // fill all components with 3
/// vec4 v{1,2,3}; // fill components specified, and set w to zero.
/// \endcode
///
template <class T, int N>
using vec __attribute__((ext_vector_type(N))) = T;

// Typedefs ====================================================================
using vec1 = vec<float, 1>;
using vec2 = vec<float, 2>;
using vec3 = vec<float, 3>;
using vec4 = vec<float, 4>;

using dvec1 = vec<double, 1>;
using dvec2 = vec<double, 2>;
using dvec3 = vec<double, 3>;
using dvec4 = vec<double, 4>;

using ivec1 = vec<int, 1>;
using ivec2 = vec<int, 2>;
using ivec3 = vec<int, 3>;
using ivec4 = vec<int, 4>;

using i64vec1 = vec<int64_t, 1>;
using i64vec2 = vec<int64_t, 2>;
using i64vec3 = vec<int64_t, 3>;
using i64vec4 = vec<int64_t, 4>;
///@}

namespace vector_detail {

// Shuffle interface.
// TODO: clean up
#define SHUFFLE(A, B, X, Y, Z, W) __builtin_shufflevector(A, B, X, Y, Z, W)

// constexpr versions of these common functions

template <class T>
constexpr T cabs(T a) {
    return (a > T(0)) ? a : -a;
}

///
/// \brief Convert a bool to an OpenCL boolean-int.
///
inline int bool_to_vec_bool(bool b) {
    return b * -1;
}

template <int N>
inline vec4 vNto4(vec<float, N> a) {
    static_assert(N >= 1 and N <= 4);
    if constexpr (N == 1) {
        return __builtin_shufflevector(a, a, 0, -1, -1, -1);
    } else if constexpr (N == 2) {
        return __builtin_shufflevector(a, a, 0, 1, -1, -1);
    } else if constexpr (N == 3) {
        return __builtin_shufflevector(a, a, 0, 1, 2, -1);
    } else {
        return a;
    }
}

template <int N>
inline vec<float, N> v4toN(vec4 a) {
    static_assert(N >= 1 and N <= 4);
    if constexpr (N == 1) {
        return __builtin_shufflevector(a, a, 0);
    } else if constexpr (N == 2) {
        return __builtin_shufflevector(a, a, 0, 1);
    } else if constexpr (N == 3) {
        return __builtin_shufflevector(a, a, 0, 1, 2);
    } else {
        return a;
    }
}

} // namespace vector_detail


// Helper Defines ==============================================================

#define SLOW_VECTOR_OP(NAME, OP)                                               \
    template <class T, int N>                                                  \
    vec<T, N> NAME(vec<T, N> const& a) {                                       \
        vec<T, N> ret;                                                         \
        if constexpr (N == 1) {                                                \
            ret.x = OP(a.x);                                                   \
        } else if constexpr (N == 2) {                                         \
            ret.x = OP(a.x);                                                   \
            ret.y = OP(a.y);                                                   \
        } else if constexpr (N == 3) {                                         \
            ret.x = OP(a.x);                                                   \
            ret.y = OP(a.y);                                                   \
            ret.z = OP(a.z);                                                   \
        } else if constexpr (N == 4) {                                         \
            ret.x = OP(a.x);                                                   \
            ret.y = OP(a.y);                                                   \
            ret.z = OP(a.z);                                                   \
            ret.w = OP(a.w);                                                   \
        }                                                                      \
        return ret;                                                            \
    }

#define INTRINSIC_4F_OP(NAME, FUNC)                                            \
    template <size_t N>                                                        \
    vec<float, N> NAME(vec<float, N> a, vec<float, N> b) {                     \
        using namespace vector_detail;                                         \
        return v4toN<N>(FUNC(vNto4(a), vNto4(b)));                             \
    }


// Creation ====================================================================

template <class T, size_t N>
vec<T, N> new_vec(std::array<T, N> a) {
    vec<T, N> r;
    for (size_t i = 0; i < N; i++) {
        r[i] = a[i];
    }
    return r;
}

template <class T>
vec<T, 4> new_vec(vec<T, 3> a, T b) {
    return vec<T, 4> { a.x, a.y, a.z, b };
}

template <class T>
vec<T, 4> new_vec(vec<T, 2> a, vec<T, 2> b) {
    return vec<T, 4> { a.x, a.y, b.x, b.y };
}

template <class T>
vec<T, 4> new_vec(vec<T, 2> a, T b, T c) {
    return vec<T, 4> { a.x, a.y, b, c };
}

template <class T>
vec<T, 4> new_vec(T a, vec<T, 2> b, T c) {
    return vec<T, 4> { a, b.x, b.y, c };
}

template <class T>
vec<T, 4> new_vec(T a, T b, vec<T, 2> c) {
    return vec<T, 4> { a, b, c.x, c.y };
}

inline vec<int, 4> new_vec(int a, int b, int c, int d) {
    // clang, for some reason, doesnt use a single instruction, but does
    // insertelement for each. no idea why. does the same thing for the
    // initializer.
    return _mm_set_epi32(d, c, b, a);
}

inline vec<int, 4> new_vec(bool a, bool b, bool c, bool d) {
    using namespace vector_detail;
    return vec<int, 4> { bool_to_vec_bool(a),
                         bool_to_vec_bool(b),
                         bool_to_vec_bool(c),
                         bool_to_vec_bool(d) };
}

// Operations - Basic Mathematics ==============================================


SLOW_VECTOR_OP(sqrt, std::sqrt)
INTRINSIC_4F_OP(sqrt, _mm_sqrt_ps);

SLOW_VECTOR_OP(acos, std::acos)
INTRINSIC_4F_OP(acos, _mm_acos_ps);

SLOW_VECTOR_OP(cos, std::cos)
INTRINSIC_4F_OP(cos, _mm_cos_ps);

SLOW_VECTOR_OP(asin, std::asin)
INTRINSIC_4F_OP(asin, _mm_asin_ps);

SLOW_VECTOR_OP(sin, std::sin)
INTRINSIC_4F_OP(sin, _mm_sin_ps);

SLOW_VECTOR_OP(atan, std::atan)
INTRINSIC_4F_OP(atan, _mm_atan_ps);

SLOW_VECTOR_OP(tan, std::tan)
INTRINSIC_4F_OP(tan, _mm_tan_ps);

SLOW_VECTOR_OP(exp, std::exp)
INTRINSIC_4F_OP(exp, _mm_exp_ps);

SLOW_VECTOR_OP(log, std::log)
INTRINSIC_4F_OP(log, _mm_log_ps);


///
/// \brief Compute the sum of all components of a vector.
///
template <class T, size_t N>
T component_sum(vec<T, N> const& a) {
    using namespace vector_detail;
    if constexpr (N == 1) {
        return a.x;
    } else if constexpr (N == 2) {
        return a.x + a.y;
    } else if constexpr (N == 3) {
        return a.x + a.y + a.z;
    } else if constexpr (N == 4) {
        return a.x + a.y + a.z + a.w;
    }
}

// Operations - Dot Product ====================================================


///
/// \brief Dot product for arb vector
///
template <class T, size_t N>
float dot(vec<T, N> a, vec<T, N> b) {
    auto r = a * b;
    return component_sum(r);
}

///
/// \brief Dot product for vec4, returning the value as a vec4(d,d,d,d).
///
inline vec4 dot_vec(vec4 a, vec4 b) {
    // documentation and testing shows this is faster than dp or hadd
#ifndef __AVX__
    __m128 mult, shuf, sums;
    mult = _mm_mul_ps(a, b);
    shuf = _mm_movehdup_ps(mult);
    sums = _mm_add_ps(mult, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    return _mm_add_ss(sums, shuf);
#else
    return _mm_dp_ps(a, b, 0xff);
#endif
}

///
/// \brief Dot product specialization for vec4
///
inline float dot(vec4 a, vec4 b) {
    auto sums = dot_vec(a, b);
    return _mm_cvtss_f32(sums);
}

// Operations - Cross Product ==================================================

///
/// \brief Cross product.
///
template <class T>
vec<T, 3> cross(vec<T, 3> const& a, vec<T, 3> const& b) {
    // Improved version by http://threadlocalmutex.com/?p=8
    return (a * b.yzx - a.yzx * b).yzx;
}

// Operations - Length =========================================================

///
/// \brief Compute the length of a vector.
///
template <class T, int N>
T length(vec<T, N> const& a) {
    return std::sqrt(dot(a, a));
}


///
/// \brief Compute the length of a vector. Specialized for vec4
///
inline float length(vec4 const& a) {
    return _mm_cvtss_f32(sqrt(dot_vec(a, a)));
}

///
/// \brief Compute the length, squared, of a vector.
///
template <class T, int N>
T length_squared(vec<T, N> const& a) {
    return dot(a, a);
}

// Operations - Distance =======================================================

///
/// \brief Compute the distance between two points.
///
template <class T, int N>
T distance(vec<T, N> const& a, vec<T, N> const& b) {
    return length(b - a);
}

///
/// \brief Compute the distance, squared, between two points.
///
template <class T, int N>
T distance_squared(vec<T, N> const& a, vec<T, N> const& b) {
    return length_squared(b - a);
}

// Operations - Normalize ======================================================

///
/// \brief Normalize the length of a vector. This does NOT check if the length
/// of the vector is zero; if so, expect badness.
///
template <class T, int N>
vec<T, N> normalize(vec<T, N> const& a) {
    static_assert(std::is_floating_point_v<T>, "Floating point required");
    return a / length(a);
}

///
/// \brief Normalize the length of a vector in a quick approximate way. This
/// does NOT check if the length of the vector is zero; if so, expect badness.
/// Specialized for vec4
///
inline vec4 fast_normalize(vec4 const& a) {
    auto dp  = dot_vec(a, a);
    auto imm = _mm_rsqrt_ps(dp);
    return a * imm;
}


// Operations - Reflect ========================================================

///
/// \brief Reflect a vector
/// \param a The vector to reflect.
/// \param normal The surface normal to reflect off of.
///
template <class T, int N>
vec<T, N> reflect(vec<T, N> const& a, vec<T, N> const& normal) {
    static_assert(std::is_floating_point_v<T>, "Floating point required");
    return a - normal * dot(normal, a) * static_cast<T>(2);
}

// Boolean =====================================================================

///
/// \brief Check if all 'boolean ints' are true
///
template <int N>
bool is_all(vec<int, N> const& a) {
    // TODO: optimize
    if constexpr (N == 1) {
        return a.x == -1;
    } else if constexpr (N == 2) {
        return a.x == -1 and a.y == -1;
    } else if constexpr (N == 3) {
        return a.x == -1 and a.y == -1 and a.z == -1;
    } else if constexpr (N == 4) {
        return a.x == -1 and a.y == -1 and a.z == -1 and a.w == -1;
    }
}

///
/// \brief Check if any 'boolean ints' are true
///
template <int N>
bool is_any(vec<int, N> const& a) {
    if constexpr (N == 1) {
        return a.x == -1;
    } else if constexpr (N == 2) {
        return a.x == -1 or a.y == -1;
    } else if constexpr (N == 3) {
        return a.x == -1 or a.y == -1 or a.z == -1;
    } else if constexpr (N == 4) {
        return a.x == -1 or a.y == -1 or a.z == -1 or a.w == -1;
    }
}

///
/// \brief Check if two vectors are component-wise equal. Beware of using this
/// with floats.
///
template <class T, int N>
bool is_equal(vec<T, N> const& a, vec<T, N> const& b) {
    return is_all(a == b);
}

///
/// \brief Check if two vectors are component-wise close enough to each other,
/// with a given limit.
///
template <class T, size_t N>
bool is_equal(vec<T, N> const& a, vec<T, N> const& b, T limit) {
    static_assert(std::is_floating_point_v<T>);

    auto      delta = distance_squared(a, b);
    vec<T, N> c(limit * limit);

    return is_all(delta < c);
}

///
/// \brief Select components between two vectors.
/// \param s A boolean-int vector. O selects from a, -1 selects from b.
/// \param a First vector
/// \param b Second vector
///
/// Even though we are not using explicit vector instructions, this looks to be
/// optimal.
///
template <class T, int N>
vec<T, N> select(vec<int, N> s, vec<T, N> a, vec<T, N> b) {
    vec<T, N> ret;

    if constexpr (N == 1) {
        ret[0] = s[0] ? b[0] : a[0];
    } else if constexpr (N == 2) {
        ret[0] = s[0] ? b[0] : a[0];
        ret[1] = s[1] ? b[1] : a[1];
    } else if constexpr (N == 3) {
        ret[0] = s[0] ? b[0] : a[0];
        ret[1] = s[1] ? b[1] : a[1];
        ret[2] = s[2] ? b[2] : a[2];
    } else if constexpr (N == 4) {
        ret[0] = s[0] ? b[0] : a[0];
        ret[1] = s[1] ? b[1] : a[1];
        ret[2] = s[2] ? b[2] : a[2];
        ret[3] = s[3] ? b[3] : a[3];
    }

    return ret;
}

// Other =======================================================================

///
/// \brief Compute the component-wise min between two vectors.
///
template <class T, size_t N>
vec<T, N> min(vec<T, N> const& a, vec<T, N> const& b) {
    static_assert(N > 0 and N < 5);
    vec<T, N> ret;
    using namespace vector_detail;
    if constexpr (N == 1) {
        return vec<T, N> { std::min(a.x, b.x) };
    } else if constexpr (N == 2) {
        return vec<T, N> { std::min(a.x, b.x), std::min(a.y, b.y) };
    } else if constexpr (N == 3) {
        return vec<T, N> { std::min(a.x, b.x),
                           std::min(a.y, b.y),
                           std::min(a.z, b.z) };
    } else if constexpr (N == 4) {
        return vec<T, N> { std::min(a.x, b.x),
                           std::min(a.y, b.y),
                           std::min(a.z, b.z),
                           std::min(a.w, b.w) };
    }
    return ret;
}

template <size_t N>
vec<float, N> min(vec<float, N> a, vec<float, N> b) {
    using namespace vector_detail;
    auto tmp = _mm_min_ps(vNto4(a), vNto4(b));
    return v4toN<N>(tmp);
}

///
/// \brief Compute the component-wise max between two vectors.
///
template <class T, size_t N>
vec<T, N> max(vec<T, N> const& a, vec<T, N> const& b) {
    vec<T, N> ret;
    using namespace vector_detail;
    if constexpr (N == 1) {
        ret.x = std::max(a.x, b.x);
    } else if constexpr (N == 2) {
        ret.x = std::max(a.x, b.x);
        ret.y = std::max(a.y, b.y);
    } else if constexpr (N == 3) {
        ret.x = std::max(a.x, b.x);
        ret.y = std::max(a.y, b.y);
        ret.z = std::max(a.z, b.z);
    } else if constexpr (N == 4) {
        ret.x = std::max(a.x, b.x);
        ret.y = std::max(a.y, b.y);
        ret.z = std::max(a.z, b.z);
        ret.w = std::max(a.w, b.w);
    }
    return ret;
}

template <size_t N>
vec<float, N> max(vec<float, N> a, vec<float, N> b) {
    using namespace vector_detail;
    auto tmp = _mm_max_ps(vNto4(a), vNto4(b));
    return v4toN<N>(tmp);
}

///
/// \brief Compute the min between all components of a vector.
///
template <class T, size_t N>
T component_min(vec<T, N> const& a) {
    using namespace vector_detail;
    if constexpr (N == 1) {
        return a.x;
    } else if constexpr (N == 2) {
        return std::min(a.x, a.y);
    } else if constexpr (N == 3) {
        return std::min(std::min(a.x, a.y), a.z);
    } else if constexpr (N == 4) {
        return std::min(std::min(a.x, a.y), std::min(a.z, a.w));
    }
}

///
/// \brief Compute the max between all components of a vector.
///
template <class T, size_t N>
T component_max(vec<T, N> const& a) {
    using namespace vector_detail;
    if constexpr (N == 1) {
        return a.x;
    } else if constexpr (N == 2) {
        return std::max(a.x, a.y);
    } else if constexpr (N == 3) {
        return std::max(std::max(a.x, a.y), a.z);
    } else if constexpr (N == 4) {
        return std::max(std::max(a.x, a.y), std::max(a.z, a.w));
    }
}


///
/// \brief Round the vector component-wise.
///
template <class T, size_t N>
vec<T, N> round(vec<T, N> const& a) {
    vec<T, N> ret;
    using namespace vector_detail;
    using namespace std;
    if constexpr (N == 1) {
        ret.x = round(a.x);
    } else if constexpr (N == 2) {
        ret.x = round(a.x);
        ret.y = round(a.y);
    } else if constexpr (N == 3) {
        ret.x = round(a.x);
        ret.y = round(a.y);
        ret.z = round(a.z);
    } else if constexpr (N == 4) {
        ret.x = round(a.x);
        ret.y = round(a.y);
        ret.z = round(a.z);
        ret.w = round(a.w);
    }
    return ret;
}

template <size_t N>
vec<float, N> round(vec<float, N> a) {
    using namespace vector_detail;
    return v4toN<N>(_mm_round_ps(vNto4(a), _MM_FROUND_TO_NEAREST_INT));
}

///
/// \brief Compute the absolute value component-wise, of a vector.
///
template <class T, size_t N>
vec<T, N> abs(vec<T, N> const& a) {
    vec<T, N> ret;
    using namespace vector_detail;
    if constexpr (N == 1) {
        ret.x = cabs(a.x);
    } else if constexpr (N == 2) {
        ret.x = cabs(a.x);
        ret.y = cabs(a.y);
    } else if constexpr (N == 3) {
        ret.x = cabs(a.x);
        ret.y = cabs(a.y);
        ret.z = cabs(a.z);
    } else if constexpr (N == 4) {
        ret.x = cabs(a.x);
        ret.y = cabs(a.y);
        ret.z = cabs(a.z);
        ret.w = cabs(a.w);
    }
    return ret;
}


template <size_t N>
vec<float, N> abs(vec<float, N> a) {
    using namespace vector_detail;
    auto tmp =
        _mm_and_ps(vNto4(a), _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)));
    return v4toN<N>(tmp);
}

///
/// \brief Compute the floor for each component of a vector
///
template <class T, size_t N>
vec<T, N> floor(vec<T, N> const& a) {
    vec<T, N> ret;
    using namespace std;
    if constexpr (N == 1) {
        ret.x = ::floor(a.x);
    } else if constexpr (N == 2) {
        ret.x = ::floor(a.x);
        ret.y = ::floor(a.y);
    } else if constexpr (N == 3) {
        ret.x = ::floor(a.x);
        ret.y = ::floor(a.y);
        ret.z = ::floor(a.z);
    } else if constexpr (N == 4) {
        ret.x = ::floor(a.x);
        ret.y = ::floor(a.y);
        ret.z = ::floor(a.z);
        ret.w = ::floor(a.w);
    }
    return ret;
}

template <size_t N>
vec<float, N> floor(vec<float, N> const& a) {
    using namespace vector_detail;
    return v4toN<N>(_mm_floor_ps(vNto4(a)));
}

///
/// \brief Compute the ceil for each component of a vector
///
template <class T, size_t N>
vec<T, N> ceil(vec<T, N> const& a) {
    vec<T, N> ret;
    using namespace std;
    if constexpr (N == 1) {
        ret.x = ::ceil(a.x);
    } else if constexpr (N == 2) {
        ret.x = ::ceil(a.x);
        ret.y = ::ceil(a.y);
    } else if constexpr (N == 3) {
        ret.x = ::ceil(a.x);
        ret.y = ::ceil(a.y);
        ret.z = ::ceil(a.z);
    } else if constexpr (N == 4) {
        ret.x = ::ceil(a.x);
        ret.y = ::ceil(a.y);
        ret.z = ::ceil(a.z);
        ret.w = ::ceil(a.w);
    }
    return ret;
}

template <size_t N>
vec<float, N> ceil(vec<float, N> const& a) {
    using namespace vector_detail;
    return v4toN<N>(_mm_ceil_ps(vNto4(a)));
}

// Clamp =======================================================================

///
/// \brief Compute the floor for each component of a vector
///
template <class T, size_t N>
vec<T, N>
clamp(vec<T, N> const& x, vec<T, N> const& min_val, vec<T, N> const& max_val) {
    return max(min(x, max_val), min_val);
}

template <class T, size_t N>
vec<T, N> clamp(vec<T, N> const& x, T const& min_val, T const& max_val) {
    vec<T, N> lmin(min_val);
    vec<T, N> lmax(max_val);

    return clamp(x, lmin, lmax);
}

// Mix =========================================================================

template <class T, size_t N>
vec<T, N> mix(vec<T, N> const& a, vec<T, N> const& b, bool t) {
    return t ? b : a;
}

template <class T>
T mix(T const& a, T const& b, bool t) {
    return t ? b : a;
}

template <class T, size_t N>
vec<T, N> mix(vec<T, N> const& a, vec<T, N> const& b, vec<int, N> const& t) {
    return select(t, a, b);
}

template <class T, size_t N>
vec<T, N> mix(vec<T, N> const& a, vec<T, N> const& b, T const& t) {
    return a + ((b - a) * t);
}

template <class T>
T mix(T const& a, T const& b, T const& t) {
    return a + ((b - a) * t);
}

template <class T, size_t N>
vec<T, N> mix(vec<T, N> const& a, vec<T, N> const& b, vec<T, N> const& t) {
    return a + ((b - a) * t);
}

// Helper Defines ==============================================================

#undef SLOW_VECTOR_OP
#undef INTRINSIC_4F_OP

} // namespace dlal

#endif // LINALG_VECTOR_DETAIL_H
