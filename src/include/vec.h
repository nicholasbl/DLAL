#ifndef LINALG_VECTOR_DETAIL_H
#define LINALG_VECTOR_DETAIL_H

#include <array>
#include <cmath>
#include <cstddef>

#include <smmintrin.h>

namespace dct {

/// Notes:
/// Vector equality uses -1 (all bits set)! If you want to use conditionals,
/// look at the is_all or is_any set of functions! The C++ conversion rules from
/// int to bool use b = i == 0; Any other value is true, which helps.
///

#ifndef __SSE4_1__
#error> SSE 4.1 Support is required
#endif

// Define how OpenCL views true and false
#define VEC_TRUE -1
#define VEC_FALSE 0

// We require extended vectors
static_assert(__has_attribute(ext_vector_type));

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
#ifdef __clang__
#define SHUFFLE(A, B, X, Y, Z, W) __builtin_shufflevector(A, B, X, Y, Z, W)
#else
#define SHUFFLE(A, B, X, Y, Z, W)                                              \
    __builtin_shuffle(A, B, dct::vector_detail::ivec4{ X, Y, Z, W })
#endif

template <int x, int y = x, int z = x, int w = x>
__attribute__((always_inline)) auto shuffle(vec4 a, vec4 b) {
    return SHUFFLE(a, b, x, y, z, w);
}

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

///
/// \brief Convert a bool to an OpenCL boolean-int.
///
inline int bool_to_vec_bool(bool b) { return b * -1; }

inline vec4 v3to4(vec3 a) { return __builtin_shufflevector(a, a, 0, 1, 2, -1); }
inline vec4 v2to4(vec2 a) {
    return __builtin_shufflevector(a, a, 0, 1, -1, -1);
}

inline vec2 v4to2(vec4 a) { return __builtin_shufflevector(a, a, 0, 1); }
inline vec3 v4to3(vec4 a) { return __builtin_shufflevector(a, a, 0, 1, 2); }

} // namespace vector_detail

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
    return vec<T, 4>{ a.x, a.y, a.z, b };
}

template <class T>
vec<T, 4> new_vec(vec<T, 2> a, vec<T, 2> b) {
    return vec<T, 4>{ a.x, a.y, b.x, b.y };
}

template <class T>
vec<T, 4> new_vec(vec<T, 2> a, T b, T c) {
    return vec<T, 4>{ a.x, a.y, b, c };
}

template <class T>
vec<T, 4> new_vec(T a, vec<T, 2> b, T c) {
    return vec<T, 4>{ a, b.x, b.y, c };
}

template <class T>
vec<T, 4> new_vec(T a, T b, vec<T, 2> c) {
    return vec<T, 4>{ a, b, c.x, c.y };
}

inline vec<int, 4> new_vec(int a, int b, int c, int d) {
    // clang, for some reason, doesnt use a single instruction, but does
    // insertelement for each. no idea why. does the same thing for the
    // initializer.
    return _mm_set_epi32(d, c, b, a);
}

inline vec<int, 4> new_vec(bool a, bool b, bool c, bool d) {
    using namespace vector_detail;
    return vec<int, 4>{ bool_to_vec_bool(a),
                        bool_to_vec_bool(b),
                        bool_to_vec_bool(c),
                        bool_to_vec_bool(d) };
}

// Operations ==================================================================

///
/// \brief Generic dot product.
///
template <class T, int N>
T dot(vec<T, N> const& a, vec<T, N> const& b) {
    static_assert(N > 0 and N <= 4);
    T ret;
    if constexpr (N == 1) {
        ret = a.x * b.x;
    } else if constexpr (N == 2) {
        ret = a.x * b.x;
        ret += a.y * b.y;
    } else if constexpr (N == 3) {
        ret = a.x * b.x;
        ret += a.y * b.y;
        ret += a.z * b.z;
    } else if constexpr (N == 4) {
        ret = a.x * b.x;
        ret += a.y * b.y;
        ret += a.z * b.z;
        ret += a.w * b.w;
    }
    return ret;
}

///
/// \brief Dot product specialization for vec3
///
inline float dot(vec3 a, vec3 b) {
    auto r = a * b;
    return r.x + r.y + r.z;
}

///
/// \brief Dot product specialization for vec4
///
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

///
/// \brief Dot product for vec4, returning the value as a vec4(d,d,d,d).
///
inline vec4 dot_vec(vec4 a, vec4 b) {
    // documentation and testing shows this is faster than dp or hadd
    __m128 mult, shuf, sums;
    mult = _mm_mul_ps(a, b);
    shuf = _mm_movehdup_ps(mult);
    sums = _mm_add_ps(mult, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    return _mm_add_ss(sums, shuf);
}

///
/// \brief Cross product.
///
template <class T>
vec<T, 3> cross(vec<T, 3> const& a, vec<T, 3> const& b) {
    return (a.yzx * b.zxy) - (b.yzx * a.zxy);
}

///
/// \brief Compute the length of a vector.
///
template <class T, int N>
T length(vec<T, N> const& a) {
    return std::sqrt(dot(a, a));
}

///
/// \brief Compute the length, squared, of a vector.
///
template <class T, int N>
T length_squared(vec<T, N> const& a) {
    return dot(a, a);
}

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
        return vec<T, N>{ cmin(a.x, b.x) };
    } else if constexpr (N == 2) {
        return vec<T, N>{ cmin(a.x, b.x), cmin(a.y, b.y) };
    } else if constexpr (N == 3) {
        return vec<T, N>{ cmin(a.x, b.x), cmin(a.y, b.y), cmin(a.z, b.z) };
    } else if constexpr (N == 4) {
        return vec<T, N>{
            cmin(a.x, b.x), cmin(a.y, b.y), cmin(a.z, b.z), cmin(a.w, b.w)
        };
    }
    return ret;
}

inline vec2 min(vec2 const& a, vec2 const& b) {
    using namespace vector_detail;
    auto tmp = _mm_min_ps(v2to4(a), v2to4(b));
    return v4to2(tmp);
}

inline vec3 min(vec3 const& a, vec3 const& b) {
    using namespace vector_detail;
    auto tmp = _mm_min_ps(v3to4(a), v3to4(b));
    return v4to3(tmp);
}

inline vec4 min(vec4 const& a, vec4 const& b) { return _mm_min_ps(a, b); }

///
/// \brief Compute the component-wise max between two vectors.
///
template <class T, size_t N>
vec<T, N> max(vec<T, N> const& a, vec<T, N> const& b) {
    vec<T, N> ret;
    using namespace vector_detail;
    if constexpr (N == 1) {
        ret.x = cmax(a.x, b.x);
    } else if constexpr (N == 2) {
        ret.x = cmax(a.x, b.x);
        ret.y = cmax(a.y, b.y);
    } else if constexpr (N == 3) {
        ret.x = cmax(a.x, b.x);
        ret.y = cmax(a.y, b.y);
        ret.z = cmax(a.z, b.z);
    } else if constexpr (N == 4) {
        ret.x = cmax(a.x, b.x);
        ret.y = cmax(a.y, b.y);
        ret.z = cmax(a.z, b.z);
        ret.w = cmax(a.w, b.w);
    }
    return ret;
}

inline vec2 max(vec2 const& a, vec2 const& b) {
    using namespace vector_detail;
    auto tmp = _mm_max_ps(v2to4(a), v2to4(b));
    return v4to2(tmp);
}

inline vec3 max(vec3 const& a, vec3 const& b) {
    using namespace vector_detail;
    auto tmp = _mm_max_ps(v3to4(a), v3to4(b));
    return v4to3(tmp);
}

inline vec4 max(vec4 const& a, vec4 const& b) { return _mm_max_ps(a, b); }

///
/// \brief Compute the min between all components of a vector.
///
template <class T, size_t N>
T component_min(vec<T, N> const& a) {
    using namespace vector_detail;
    if constexpr (N == 1) {
        return a.x;
    } else if constexpr (N == 2) {
        return cmin(a.x, a.y);
    } else if constexpr (N == 3) {
        return cmin(cmin(a.x, a.y), a.z);
    } else if constexpr (N == 4) {
        return cmin(cmin(a.x, a.y), cmin(a.z, a.w));
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
        return cmax(a.x, a.y);
    } else if constexpr (N == 3) {
        return cmax(cmax(a.x, a.y), a.z);
    } else if constexpr (N == 4) {
        return cmax(cmax(a.x, a.y), cmax(a.z, a.w));
    }
}

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

inline vec2 round(vec2 const& a) {
    using namespace vector_detail;
    return v4to2(_mm_round_ps(v2to4(a), _MM_FROUND_TO_NEAREST_INT));
}

inline vec3 round(vec3 const& a) {
    using namespace vector_detail;
    return v4to3(_mm_round_ps(v3to4(a), _MM_FROUND_TO_NEAREST_INT));
}

inline vec4 round(vec4 const& a) {
    return _mm_round_ps(a, _MM_FROUND_TO_NEAREST_INT);
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

inline vec4 abs(vec4 const& a) {
    return _mm_and_ps(a, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)));
}

inline vec3 abs(vec3 const& a) {
    using namespace vector_detail;
    return v4to3(abs(v3to4(a)));
}

inline vec2 abs(vec2 const& a) {
    using namespace vector_detail;
    return v4to2(abs(v2to4(a)));
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

inline vec4 floor(vec4 const& a) { return _mm_floor_ps(a); }
inline vec3 floor(vec3 const& a) {
    using namespace vector_detail;
    return v4to3(floor(v3to4(a)));
}

inline vec2 floor(vec2 const& a) {
    using namespace vector_detail;
    return v4to2(floor(v2to4(a)));
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

inline vec4 ceil(vec4 const& a) { return _mm_ceil_ps(a); }
inline vec3 ceil(vec3 const& a) {
    using namespace vector_detail;
    return v4to3(ceil(v3to4(a)));
}

inline vec2 ceil(vec2 const& a) {
    using namespace vector_detail;
    return v4to2(ceil(v2to4(a)));
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

} // namespace dct

#endif // LINALG_VECTOR_DETAIL_H
