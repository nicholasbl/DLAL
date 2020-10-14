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

inline vec4 v3to4(vec3 a) {
    return __builtin_shufflevector(a, a, 0, 1, 2, -1);
}
inline vec4 v2to4(vec2 a) {
    return __builtin_shufflevector(a, a, 0, 1, -1, -1);
}

inline vec2 v4to2(vec4 a) {
    return __builtin_shufflevector(a, a, 0, 1);
}
inline vec3 v4to3(vec4 a) {
    return __builtin_shufflevector(a, a, 0, 1, 2);
}

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

// Operations ==================================================================


///
/// \brief Dot product specialization for arb vec2
///
template <class T>
float dot(vec<T, 2> a, vec<T, 2> b) {
    auto r = a * b;
    return r.x + r.y;
}

///
/// \brief Dot product specialization for arb vec3
///
template <class T>
float dot(vec<T, 3> a, vec<T, 3> b) {
    auto r = a * b;
    return r.x + r.y + r.z;
}

///
/// \brief Dot product specialization for arb vec4
///
template <class T>
float dot(vec<T, 4> a, vec<T, 4> b) {
    auto r = a * b;
    return r.x + r.y + r.z + r.w;
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
    // Improved version by http://threadlocalmutex.com/?p=8
    return (a * b.yzx - a.yzx * b).yzx;
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

inline vec4 min(vec4 const& a, vec4 const& b) {
    return _mm_min_ps(a, b);
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

inline vec4 max(vec4 const& a, vec4 const& b) {
    return _mm_max_ps(a, b);
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

inline vec4 floor(vec4 const& a) {
    return _mm_floor_ps(a);
}
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

inline vec4 ceil(vec4 const& a) {
    return _mm_ceil_ps(a);
}
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

} // namespace dlal

#endif // LINALG_VECTOR_DETAIL_H
#ifndef LINALG_VECTOR_H
#define LINALG_VECTOR_H


#include <algorithm>
#include <array>
#include <cmath>

namespace dlal {

/// \brief The basic packed vector class; specializations define the 1-4
/// component cases. These should be used when storage sizes are important.
template <class T, size_t N>
class packed_vector { };

// Vector 1 ====================================================================

/// \brief The basic packed vector class; specialized for 1 component.
template <class T>
class packed_vector<T, 1> {
    /// Storage is a simple array
    using StorageType = std::array<T, 1>;

    union {
        StorageType storage; ///< Array storage
        struct {
            T x; ///< Basic swizzle
        };
    };

public: // Basics
    /// \brief Count of vector components
    constexpr size_t size() { return storage.size(); }

    /// @{
    /// Access a component by index
    constexpr T&       operator[](size_t i) { return storage[i]; }
    constexpr T const& operator[](size_t i) const { return storage[i]; }
    /// @}

public:
    /// \brief Initialize all elements to zero
    constexpr packed_vector() : storage() { }

    /// \brief Default copy constructor
    constexpr packed_vector(packed_vector const&) = default;

    /// \brief Initialize all elements to the provided value
    constexpr packed_vector(T _) { storage.fill(_); }

    /// \brief Construct from a std::array
    constexpr packed_vector(StorageType st) : storage(st) { }

    /// \brief Construct from a non-packed vector
    constexpr packed_vector(vec<T, 1> simd) : x(simd.x) { }

    /// \brief Default copy assignment
    constexpr packed_vector& operator=(packed_vector const& v) = default;

    /// \brief Convert to non-packed vector
    operator vec<T, 1>() const { return vec<T, 1> { x }; }

public:
    /// @{
    /// \brief Access the underlying storage as a contiguous array.
    T*       data() { return storage.data(); }
    T const* data() const { return storage.data(); }
    /// @}

    /// @{
    /// \brief Iterator support
    constexpr auto begin() { return storage.begin(); }
    constexpr auto begin() const { return storage.begin(); }
    constexpr auto end() { return storage.end(); }
    constexpr auto end() const { return storage.end(); }
    /// @}
};

// Vector 2 ====================================================================

/// \brief The basic packed vector class; specialized for 2 components.
template <class T>
struct packed_vector<T, 2> {
    /// Storage is a simple array
    using StorageType = std::array<T, 2>;

    union {
        StorageType storage; ///< Array storage
        struct {
            T x, y; ///< Basic swizzle
        };
    };

public: // Basics
    /// \brief Count of vector components
    constexpr size_t size() { return storage.size(); }

    /// @{
    /// Access a component by index
    constexpr T&       operator[](size_t i) { return storage[i]; }
    constexpr T const& operator[](size_t i) const { return storage[i]; }
    /// @}

public:
    /// \brief Initialize all elements to zero
    constexpr packed_vector() : storage() { }

    /// \brief Default copy constructor
    constexpr packed_vector(packed_vector const&) = default;

    /// \brief Initialize all elements to the provided value
    constexpr packed_vector(T _xy) : packed_vector(_xy, _xy) { }

    /// \brief Construct from a std::array
    constexpr packed_vector(StorageType st) : storage(st) { }

    /// \brief Construct from loose values
    constexpr packed_vector(T _x, T _y) : storage { _x, _y } { }

    /// \brief Construct from a non-packed vector
    constexpr packed_vector(vec<T, 2> simd) : packed_vector(simd.x, simd.y) { }

    /// \brief Default copy assignment
    constexpr packed_vector& operator=(packed_vector const& v) = default;

    /// \brief Convert to non-packed vector
    operator vec<T, 2>() const { return vec<T, 2> { x, y }; }

public:
    /// @{
    /// \brief Access the underlying storage as a contiguous array.
    T*       data() { return storage.data(); }
    T const* data() const { return storage.data(); }
    /// @}

    /// @{
    /// \brief Iterator support
    constexpr auto begin() { return storage.begin(); }
    constexpr auto begin() const { return storage.begin(); }
    constexpr auto end() { return storage.end(); }
    constexpr auto end() const { return storage.end(); }
    /// @}
};

static_assert(sizeof(packed_vector<float, 2>) == sizeof(float) * 2);

// Vector 3 ====================================================================

/// \brief The basic packed vector class; specialized for 3 components.
template <class T>
struct packed_vector<T, 3> {
    /// Storage is a simple array
    using StorageType = std::array<T, 3>;

    union {
        StorageType storage; ///< Array storage
        struct {
            T x, y, z; ///< Basic swizzle
        };
    };

public: // Basics
    /// \brief Count of vector components
    constexpr size_t size() { return storage.size(); }

    /// @{
    /// Access a component by index
    constexpr T&       operator[](size_t i) { return storage[i]; }
    constexpr T const& operator[](size_t i) const { return storage[i]; }
    /// @}

public:
    /// \brief Initialize all elements to zero
    constexpr packed_vector() : storage() { }

    /// \brief Default copy constructor
    constexpr packed_vector(packed_vector const&) = default;

    /// \brief Initialize all elements to the provided value
    constexpr packed_vector(T _xyz) : packed_vector(_xyz, _xyz, _xyz) { }

    /// \brief Construct from a std::array
    constexpr packed_vector(StorageType st) : storage(st) { }

    /// \brief Construct from loose values
    constexpr packed_vector(T _x, T _y, T _z) : storage { _x, _y, _z } { }

    /// \brief Construct from a non-packed vector
    constexpr packed_vector(vec<T, 3> simd)
        : packed_vector(simd.x, simd.y, simd.z) { }

    /// \brief Default copy assignment
    constexpr packed_vector& operator=(packed_vector const& v) = default;

    /// \brief Convert to non-packed vector
    operator vec<T, 3>() const { return vec<T, 3> { x, y, z }; }

public:
    /// @{
    /// \brief Access the underlying storage as a contiguous array.
    T*       data() { return storage.data(); }
    T const* data() const { return storage.data(); }
    /// @}

    /// @{
    /// \brief Iterator support
    constexpr auto begin() { return storage.begin(); }
    constexpr auto begin() const { return storage.begin(); }
    constexpr auto end() { return storage.end(); }
    constexpr auto end() const { return storage.end(); }
    /// @}
};

static_assert(sizeof(packed_vector<float, 3>) == sizeof(float) * 3);

// Vector 4 ====================================================================

/// \brief The basic packed vector class; specialized for 4 components.
template <class T>
struct packed_vector<T, 4> {
    /// Storage is a simple array
    using StorageType = std::array<T, 4>;

    union {
        StorageType storage; ///< Array storage
        struct {
            T x, y, z, w; ///< Basic swizzle
        };
    };

public: // Basics
    /// \brief Count of vector components
    constexpr size_t size() { return storage.size(); }

    /// @{
    /// Access a component by index
    constexpr T&       operator[](size_t i) { return storage[i]; }
    constexpr T const& operator[](size_t i) const { return storage[i]; }
    /// @}

public:
    /// \brief Initialize all elements to zero
    constexpr packed_vector() : storage() { }

    /// \brief Default copy constructor
    constexpr packed_vector(packed_vector const&) = default;

    /// \brief Initialize all elements to the provided value
    constexpr packed_vector(T _xyzw)
        : packed_vector(_xyzw, _xyzw, _xyzw, _xyzw) { }

    /// \brief Construct from a std::array
    constexpr packed_vector(StorageType st) : storage(st) { }

    /// \brief Construct from loose values
    constexpr packed_vector(T _x, T _y, T _z, T _w)
        : storage { _x, _y, _z, _w } { }

    /// \brief Construct from a non-packed vector
    constexpr packed_vector(vec<T, 4> simd)
        : packed_vector(simd.x, simd.y, simd.z, simd.w) { }

    /// \brief Default copy assignment
    constexpr packed_vector& operator=(packed_vector const& v) = default;

    /// \brief Convert to non-packed vector
    operator vec<T, 4>() const { return vec<T, 4> { x, y, z, w }; }

public:
    /// @{
    /// \brief Access the underlying storage as a contiguous array.
    T*       data() { return storage.data(); }
    T const* data() const { return storage.data(); }
    /// @}

    /// @{
    /// \brief Iterator support
    constexpr auto begin() { return storage.begin(); }
    constexpr auto begin() const { return storage.begin(); }
    constexpr auto end() { return storage.end(); }
    constexpr auto end() const { return storage.end(); }
    /// @}
};

// Vector Typedefs =============================================================

using float16 = __fp16;

using packed_bvec1 = packed_vector<bool, 1>;
using packed_bvec2 = packed_vector<bool, 2>;
using packed_bvec3 = packed_vector<bool, 3>;
using packed_bvec4 = packed_vector<bool, 4>;

using packed_i8vec1 = packed_vector<int8_t, 1>;
using packed_i8vec2 = packed_vector<int8_t, 2>;
using packed_i8vec3 = packed_vector<int8_t, 3>;
using packed_i8vec4 = packed_vector<int8_t, 4>;

using packed_i16vec1 = packed_vector<int16_t, 1>;
using packed_i16vec2 = packed_vector<int16_t, 2>;
using packed_i16vec3 = packed_vector<int16_t, 3>;
using packed_i16vec4 = packed_vector<int16_t, 4>;

using packed_ivec1 = packed_vector<int32_t, 1>;
using packed_ivec2 = packed_vector<int32_t, 2>;
using packed_ivec3 = packed_vector<int32_t, 3>;
using packed_ivec4 = packed_vector<int32_t, 4>;

using packed_i64vec1 = packed_vector<int64_t, 1>;
using packed_i64vec2 = packed_vector<int64_t, 2>;
using packed_i64vec3 = packed_vector<int64_t, 3>;
using packed_i64vec4 = packed_vector<int64_t, 4>;

using packed_vec1 = packed_vector<float, 1>;
using packed_vec2 = packed_vector<float, 2>;
using packed_vec3 = packed_vector<float, 3>;
using packed_vec4 = packed_vector<float, 4>;

using packed_hvec1 = packed_vector<float16, 1>;
using packed_hvec2 = packed_vector<float16, 2>;
using packed_hvec3 = packed_vector<float16, 3>;
using packed_hvec4 = packed_vector<float16, 4>;

using packed_dvec1 = packed_vector<double, 1>;
using packed_dvec2 = packed_vector<double, 2>;
using packed_dvec3 = packed_vector<double, 3>;
using packed_dvec4 = packed_vector<double, 4>;

template <size_t N>
packed_vector<float16, N> half_vector(packed_vector<float, N> f) {
    static_assert(N > 0 and N <= 4);

    std::array<float16, N> ret;

    if constexpr (N == 1) {
        ret[0] = f[0];
    } else if constexpr (N == 2) {
        ret[0] = f[0];
        ret[1] = f[1];
    } else if constexpr (N == 3) {
        ret[0] = f[0];
        ret[1] = f[1];
        ret[2] = f[2];
    } else if constexpr (N == 4) {
        ret[0] = f[0];
        ret[1] = f[1];
        ret[2] = f[2];
        ret[3] = f[3];
    }

    return ret;
}


} // namespace dlal


#endif // LINALG_VECTOR_H
#ifndef LINALG_VECTOR_TRIG_H
#define LINALG_VECTOR_TRIG_H


#include <cmath>

namespace dlal {

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
vec<T, N> sqrt(vec<T, N> const& a) { VECTOR_OP(std::sqrt) }

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

} // namespace dlal

#endif // LINALG_VECTOR_TRIG_H
#ifndef LINALG_MATRIX_DETAIL_H
#define LINALG_MATRIX_DETAIL_H


#include <array>
#include <cstddef>

namespace dlal {

namespace matrix_detail {

///
/// \brief Helper function to clip or extend a vector, copying from another.
///
template <int N, class T, int M>
constexpr vec<T, N> upgrade(vec<T, M> const& v) {
    constexpr size_t C = std::min(N, M);
    static_assert(C <= 4);
    vec<T, N> ret;
    if constexpr (C == 1) {
        return vec<T, N> { v.x };
    } else if constexpr (C == 2) {
        return vec<T, N> { v.x, v.y };
    } else if constexpr (C == 3) {
        return vec<T, N> { v.x, v.y, v.z };
    } else if constexpr (C == 4) {
        return vec<T, N> { v.x, v.y, v.z, v.w };
    }
    return ret;
}


///
/// \brief Helper function to copy range of a vector onto another
///
template <int DEST_N, class T, int SRC_N>
constexpr void overlay(vec<T, SRC_N> const& src, vec<T, DEST_N>& dest) {
    constexpr size_t LOW_N = std::min(DEST_N, SRC_N);
    static_assert(LOW_N <= 4);

    if constexpr (SRC_N == DEST_N) {
        dest = src;
    } else if constexpr (LOW_N == 1) {
        dest.x = src.x;
    } else if constexpr (LOW_N == 2) {
        dest.x = src.x;
        dest.y = src.y;
    } else if constexpr (LOW_N == 3) {
        dest.x = src.x;
        dest.y = src.y;
        dest.z = src.z;
    } else if constexpr (LOW_N == 4) {
        dest.x = src.x;
        dest.y = src.y;
        dest.z = src.z;
        dest.w = src.w;
    }
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

///
/// \brief Construct a vector (branchless at runtime) with a 1 in the requested
/// spot.
///
template <class T, size_t N, size_t AT>
vec<T, N> get_identity_vec() {
    // If a 1 is wanted outside our range...
    if constexpr (N == 1 and AT >= N) {
        return vec<T, N>(0);
        // else construct a vector with a 1 in the right slot
    } else if constexpr (N == 1) {
        return vec<T, N> { 1 };
    } else if constexpr (N == 2) {
        if constexpr (AT == 0) {
            return vec<T, N> { 1, 0 };
        } else if constexpr (AT == 1) {
            return vec<T, N> { 0, 1 };
        }
    } else if constexpr (N == 3) {
        if constexpr (AT == 0) {
            return vec<T, N> { 1, 0, 0 };
        } else if constexpr (AT == 1) {
            return vec<T, N> { 0, 1, 0 };
        } else if constexpr (AT == 2) {
            return vec<T, N> { 0, 0, 1 };
        }
    } else if constexpr (N == 4) {
        if constexpr (AT == 0) {
            return vec<T, N> { 1, 0, 0, 0 };
        } else if constexpr (AT == 1) {
            return vec<T, N> { 0, 1, 0, 0 };
        } else if constexpr (AT == 2) {
            return vec<T, N> { 0, 0, 1, 0 };
        } else if constexpr (AT == 3) {
            return vec<T, N> { 0, 0, 0, 1 };
        }
    }
}

///
/// \brief Construct a (branchless at runtime) identity storage for mats.
///
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

} // namespace dlal

#endif // LINALG_MATRIX_DETAIL_H
#ifndef LINALG_MATRIX_H
#define LINALG_MATRIX_H


#include <xmmintrin.h>

namespace dlal {

///
/// \brief The identity_t struct allows the user to select the identity
/// constructor. Use dct::identity to initialize a new matrix with the identity.
///
struct identity_t {
} static const identity;

///
/// \brief The root Matrix template
///
/// \tparam T The cell value type of the matrix
/// \tparam C The number of columns
/// \tparam R The number of rows
///
template <class T, size_t C, size_t R>
struct mat {
    using ColumnType = vec<T, R>; ///< The column vector type
    using RowType    = vec<T, C>; ///< The row vector type

    /// The underlying storage type
    using StorageType = std::array<ColumnType, C>;

    /// Storage as columns
    StorageType storage;

    /// Indicates if the type uses contiguous storage, which can speed up
    /// copies/conversions.
    static constexpr bool is_contiguous =
        sizeof(storage) == (sizeof(T) * C * R);

public: // Basics
    ///
    /// \brief Get the total number of cells
    ///
    constexpr size_t size() { return C * R; }
    constexpr size_t row_count() { return R; }    ///< Count of rows (R)
    constexpr size_t column_count() { return C; } ///< Count of columns (C)

    /// @{
    /// \brief Access a column.
    constexpr ColumnType&       operator[](size_t i) { return storage[i]; }
    constexpr ColumnType const& operator[](size_t i) const {
        return storage[i];
    }
    /// @}

public:
    /// \brief Initialize all cells to zero
    constexpr mat() : storage() { }

    /// \brief Initialize the matrix to the identity.
    constexpr mat(identity_t)
        : storage(matrix_detail::get_identity_storage<T, C, R>()) { }

    /// \brief Default copy behavior.
    constexpr mat(mat const&) = default;

    /// \brief Construct a new matrix using the given value array. A single
    /// copy will be performed if possible. It may be faster to construct the
    /// matrix with either the storage constructor, or build a default matrix
    /// and then set columns.
    constexpr mat(std::array<float, C * R> const& a) {
        if constexpr (is_contiguous) {
            std::copy(a.data(),
                      a.data() + a.size(),
                      reinterpret_cast<T*>(&storage[0]));
        } else {
            for (size_t i = 0; i < C; ++i) {
                for (size_t j = 0; j < R; ++j) {
                    (*this)[i][j] = a[i * R + j];
                }
            }
        }
    }

    /// \brief Initialize all cells to the given value
    constexpr mat(T value) { storage.fill(value); }

    /// \brief Initialize the matrix with an array of columns.
    constexpr mat(StorageType pack) : storage(pack) { }

    /// \brief Initialize values from a differently sized matrix, zeros
    /// otherwise.
    template <size_t C2, size_t R2>
    constexpr explicit mat(mat<T, C2, R2> const& other) {
        using namespace matrix_detail;
        constexpr size_t bound = C2 < C ? C2 : C;
        // no loops for speed in debug mode

        if constexpr (bound == 1) {
            storage[0] = upgrade<R>(other.storage[0]);
        } else if constexpr (bound == 2) {
            storage[0] = upgrade<R>(other.storage[0]);
            storage[1] = upgrade<R>(other.storage[1]);
        } else if constexpr (bound == 3) {
            storage[0] = upgrade<R>(other.storage[0]);
            storage[1] = upgrade<R>(other.storage[1]);
            storage[2] = upgrade<R>(other.storage[2]);
        } else if constexpr (bound == 4) {
            storage[0] = upgrade<R>(other.storage[0]);
            storage[1] = upgrade<R>(other.storage[1]);
            storage[2] = upgrade<R>(other.storage[2]);
            storage[3] = upgrade<R>(other.storage[3]);
        }
    }

public:
    /// \brief Default copy assignment.
    mat& operator=(mat const& m) = default;


public:
    /// \brief Copy values from a differently sized matrix into this one.
    template <size_t C2, size_t R2>
    void inset(mat<T, C2, R2> const& other) {
        using namespace matrix_detail;
        constexpr size_t bound = C2 < C ? C2 : C;
        // no loops for speed in debug mode

        if constexpr (bound == 1) {
            overlay(other.storage[0], storage[0]);
        } else if constexpr (bound == 2) {
            overlay(other.storage[0], storage[0]);
            overlay(other.storage[1], storage[1]);
        } else if constexpr (bound == 3) {
            overlay(other.storage[0], storage[0]);
            overlay(other.storage[1], storage[1]);
            overlay(other.storage[2], storage[2]);
        } else if constexpr (bound == 4) {
            overlay(other.storage[0], storage[0]);
            overlay(other.storage[1], storage[1]);
            overlay(other.storage[2], storage[2]);
            overlay(other.storage[3], storage[3]);
        }
    }
};

// Typedefs ====================================================================

using mat2 = mat<float, 2, 2>;
using mat3 = mat<float, 3, 3>;
using mat4 = mat<float, 4, 4>;

using mat22 = mat<float, 2, 2>;
using mat23 = mat<float, 2, 3>;
using mat24 = mat<float, 2, 4>;

using mat32 = mat<float, 3, 2>;
using mat33 = mat<float, 3, 3>;
using mat34 = mat<float, 3, 4>;

using mat42 = mat<float, 4, 2>;
using mat43 = mat<float, 4, 3>;
using mat44 = mat<float, 4, 4>;

// Accessors ===================================================================

/// {
/// Get a pointer to the underlying value type array. This is only viable for
/// contiguous matrices. In general, use packed versions instead.
template <class T, size_t C, size_t R>
T* data(mat<T, C, R>& m) {
    static_assert(mat<T, C, R>::is_contiguous,
                  "Matrix must have contiguous storage.");
    return reinterpret_cast<T*>(&m.storage[0]);
}

template <class T, size_t C, size_t R>
T const* data(mat<T, C, R> const& m) {
    static_assert(mat<T, C, R>::is_contiguous,
                  "Matrix must have contiguous storage.");
    return reinterpret_cast<T const*>(&m.storage[0]);
}
/// }

// Unary =======================================================================
template <class T, size_t C, size_t R>
mat<T, C, R> operator-(mat<T, C, R> const& m) {
    MATRIX_UNARY(-)
}

template <class T, size_t C, size_t R>
mat<T, C, R> operator!(mat<T, C, R> const& m) {
    MATRIX_UNARY(!)
}

// Operators ===================================================================

// Addition ====================================================================
template <class T, size_t C, size_t R>
auto operator+(mat<T, C, R> const& m, mat<T, C, R> const& o) {
    MATRIX_BINARY(+)
}
template <class T, size_t C, size_t R>
auto operator+(mat<T, C, R> const& m, T scalar) {
    MATRIX_BINARY_SCALAR_R(+)
}
template <class T, size_t C, size_t R>
auto operator+(T scalar, mat<T, C, R> const& m) {
    MATRIX_BINARY_SCALAR_L(+)
}

template <class T, size_t C, size_t R>
mat<T, C, R>& operator+=(mat<T, C, R>& m, mat<T, C, R> const& o) {
    MATRIX_IN_PLACE(+=)
}
template <class T, size_t C, size_t R>
mat<T, C, R>& operator+=(mat<T, C, R>& m, T scalar) {
    MATRIX_IN_PLACE_SCALAR_R(+=)
}

// Subtraction =================================================================

template <class T, size_t C, size_t R>
auto operator-(mat<T, C, R> const& m, mat<T, C, R> const& o) {
    MATRIX_BINARY(-)
}
template <class T, size_t C, size_t R>
auto operator-(mat<T, C, R> const& m, T scalar) {
    MATRIX_BINARY_SCALAR_R(-)
}
template <class T, size_t C, size_t R>
auto operator-(T scalar, mat<T, C, R> const& m) {
    MATRIX_BINARY_SCALAR_L(-)
}

template <class T, size_t C, size_t R>
mat<T, C, R>& operator-=(mat<T, C, R>& m, mat<T, C, R> const& o) {
    MATRIX_IN_PLACE(-=)
}
template <class T, size_t C, size_t R>
mat<T, C, R>& operator-=(mat<T, C, R>& m, T scalar) {
    MATRIX_IN_PLACE_SCALAR_R(-=)
}

// Multiply ====================================================================

template <class T, size_t N, size_t R, size_t C>
auto operator*(mat<T, N, R> const& m, mat<T, C, N> const& o) {
    static_assert(std::is_same_v<typename mat<T, N, R>::RowType,
                                 typename mat<T, C, N>::ColumnType>);
    mat<T, C, R> ret(0);

    for (size_t i = 0; i < C; ++i) {
        for (size_t j = 0; j < R; ++j) {
            for (size_t k = 0; k < N; ++k) {
                ret[i][j] += m[k][j] * o[i][k];
            }
        }
    }

    return ret;
}

template <class T, size_t C, size_t R>
auto operator*(mat<T, C, R> const& m, vec<T, R> const& o) {
    vec<T, R> ret(0);

    for (size_t i = 0; i < C; ++i) {
        ret += m[i] * vec<T, R>(o[i]);
    }

    return ret;
}


/// Specialization for common case
inline auto operator*(mat3 const& m, mat3 const& o) {
    auto const m0 = m[0];
    auto const m1 = m[1];
    auto const m2 = m[2];

    auto const o0 = o[0];
    auto const o1 = o[1];
    auto const o2 = o[2];

    mat3 ret;
    ret[0] = m0 * o0[0] + m1 * o0[1] + m2 * o0[2];
    ret[1] = m0 * o1[0] + m1 * o1[1] + m2 * o1[2];
    ret[2] = m0 * o2[0] + m1 * o2[1] + m2 * o2[2];
    return ret;
}

/// Specialization for common case
inline auto operator*(mat4 const& m, mat4 const& o) {
    auto const m0 = m[0];
    auto const m1 = m[1];
    auto const m2 = m[2];
    auto const m3 = m[3];

    auto const o0 = o[0];
    auto const o1 = o[1];
    auto const o2 = o[2];
    auto const o3 = o[3];

    mat4 ret;
    ret[0] = m0 * o0[0] + m1 * o0[1] + m2 * o0[2] + m3 * o0[3];
    ret[1] = m0 * o1[0] + m1 * o1[1] + m2 * o1[2] + m3 * o1[3];
    ret[2] = m0 * o2[0] + m1 * o2[1] + m2 * o2[2] + m3 * o2[3];
    ret[3] = m0 * o3[0] + m1 * o3[1] + m2 * o3[2] + m3 * o3[3];
    return ret;
}

template <class T>
auto operator*(mat<T, 3, 3> const& m, vec<T, 3> const& o) {
    vec<T, 3> a0 = m[0] * vec<T, 3>(o[0]);
    vec<T, 3> a1 = m[1] * vec<T, 3>(o[1]);

    auto m1 = a0 + a1;

    vec<T, 3> a2 = m[2] * vec<T, 3>(o[2]);

    return m1 + a2;
}

template <class T>
auto operator*(mat<T, 4, 4> const& m, vec<T, 4> const& o) {
    vec<T, 4> a0 = m[0] * vec<T, 4>(o[0]);
    vec<T, 4> a1 = m[1] * vec<T, 4>(o[1]);

    auto m1 = a0 + a1;

    vec<T, 4> a2 = m[2] * vec<T, 4>(o[2]);
    vec<T, 4> a3 = m[3] * vec<T, 4>(o[3]);

    auto m2 = a2 + a3;

    return m1 + m2;
}


template <class T, size_t C, size_t R>
auto operator*(mat<T, C, R> const& m, T scalar) {
    MATRIX_BINARY_SCALAR_R(*)
}

template <class T, size_t C, size_t R>
auto operator*(T scalar, mat<T, C, R> const& m) {
    MATRIX_BINARY_SCALAR_L(*)
}


template <class T, size_t N, size_t R, size_t C>
mat<T, N, R> operator*=(mat<T, N, R>& m, mat<T, C, N> const& o) {
    return m = m * o;
}
template <class T, size_t C, size_t R>
mat<T, C, R>& operator*=(mat<T, C, R>& m, T scalar) {
    return m = m * scalar;
}
template <class T, size_t C, size_t R>
mat<T, C, R>& operator*=(T scalar, mat<T, C, R>& m) {
    return m = scalar * m;
}

// Division ====================================================================

template <class T, size_t C, size_t R>
auto operator/(mat<T, C, R> const& m, T scalar) {
    MATRIX_BINARY_SCALAR_R(/)
}

template <class T, size_t C, size_t R>
auto operator/(T scalar, mat<T, C, R> const& m) {
    MATRIX_BINARY_SCALAR_L(/)
}

// Boolean =====================================================================

template <class T, size_t C, size_t R>
auto operator==(mat<T, C, R> const& m, mat<T, C, R> const& o) {
    MATRIX_BINARY_BOOL(==)
}

template <class T, size_t C, size_t R>
auto operator!=(mat<T, C, R> const& m, mat<T, C, R> const& o) {
    MATRIX_BINARY_BOOL(!=)
}

template <class T, size_t C, size_t R>
auto operator&&(mat<bool, C, R> const& m, mat<bool, C, R> const& o) {
    MATRIX_BINARY_BOOL(&&)
}

template <class T, size_t C, size_t R>
auto operator||(mat<bool, C, R> const& m, mat<bool, C, R> const& o) {
    MATRIX_BINARY_BOOL(||)
}

#undef MATRIX_UNARY
#undef MATRIX_BINARY_SCALAR_R
#undef MATRIX_BINARY_SCALAR_L
#undef MATRIX_BINARY
#undef MATRIX_IN_PLACE
#undef MATRIX_IN_PLACE_SCALAR_R
#undef MATRIX_IN_PLACE_SCALAR_L
#undef MATRIX_BINARY_BOOL

// Other =======================================================================

///
/// \brief Ask if all cells are true. Note that 'true' is the OpenCL boolean
/// true of -1.
///
template <size_t C, size_t R>
bool is_all(mat<int, C, R> const& a) {
    if constexpr (C == 1) {
        return is_all(a[0]);
    } else if constexpr (C == 2) {
        return is_all(a[0]) and is_all(a[1]);
    } else if constexpr (C == 3) {
        return is_all(a[0]) and is_all(a[1]) and is_all(a[2]);
    } else if constexpr (C == 4) {
        return is_all(a[0]) and is_all(a[1]) and is_all(a[2]) and is_all(a[3]);
    }
}

///
/// \brief Ask if any cells are true. Note that 'true' is the OpenCL boolean
/// true of -1.
///
template <size_t C, size_t R>
bool is_any(mat<int, C, R> const& a) {
    if constexpr (C == 1) {
        return is_any(a[0]);
    } else if constexpr (C == 2) {
        return is_any(a[0]) or is_any(a[1]);
    } else if constexpr (C == 3) {
        return is_any(a[0]) or is_any(a[1]) or is_any(a[2]);
    } else if constexpr (C == 4) {
        return is_any(a[0]) or is_any(a[1]) or is_any(a[2]) or is_any(a[3]);
    }
}

///
/// \brief Ask two mats are equal. Note that this is not safe for use with
/// floats.
///
template <class T, size_t C, size_t R>
bool is_equal(mat<T, C, R> const& a, mat<T, C, R> const& b) {
    return is_all(a == b);
}

///
/// \brief Ask if two mats are equal with a per-cell error limit.
///
template <class T, size_t C, size_t R>
bool is_equal(mat<T, C, R> const& a, mat<T, C, R> const& b, T limit) {
    static_assert(std::is_floating_point_v<T>);

    auto         delta = abs(a - b);
    mat<T, C, R> c(limit);

    return is_all(delta < c);
}

} // namespace dlal

#endif // LINALG_MATRIX_H
#ifndef LINALG_QUAT_H
#define LINALG_QUAT_H



namespace dlal {

///
/// \brief The quaternion class models a rotation
///
template <class T>
struct quaternion {
    union {
        vec<T, 4> storage; ///< Storage is simd vector
        struct {
            T x, y, z, w; ///< Basic swizzle
        };
    };

public:
    /// \brief Initialize quaternion to zero
    quaternion() : storage { 0, 0, 0, 1 } { }

    /// \brief Initialize quaternion from loose values. W is the scalar.
    quaternion(T x, T y, T z, T w) : storage { x, y, z, w } { }

    /// \brief Initialize quaternion from vector and scalar.
    quaternion(T w, vec<T, 3> const& v) : storage { v.x, v.y, v.z, w } { }

    /// \brief Initialize quaternion from a vector; w should be the scalar.
    explicit quaternion(vec<T, 4> const& f) : storage(f) { }

public:
    /// \brief Convert to a vector
    explicit operator vec<T, 4>() const { return storage; }
};

using quat  = quaternion<float>;
using dquat = quaternion<double>;


// Operators ===================================================================

template <class T>
quaternion<T> operator+(quaternion<T> const& q, quaternion<T> const& r) {
    return quaternion<T>(q.storage + r.storage);
}

template <class T>
quaternion<T> operator-(quaternion<T> const& q, quaternion<T> const& r) {
    return quaternion<T>(q.storage - r.storage);
}

template <class T>
quaternion<T> operator*(quaternion<T> const& q, T scalar) {
    return quaternion<T>(q.storage * scalar);
}

// note that rotating a non-unit quaternion can do odd things

template <class T>
quaternion<T> operator*(quaternion<T> const& q, quaternion<T> const& r) {
    static constexpr dlal::vec<T, 4> mask1 { 1, 1, -1, -1 };
    static constexpr dlal::vec<T, 4> mask2 { -1, 1, 1, -1 };
    static constexpr dlal::vec<T, 4> mask3 { 1, -1, 1, -1 };

    dlal::vec<T, 4> p1 = r.storage.w * q.storage;
    dlal::vec<T, 4> p2 = mask1 * r.storage.x * q.storage.wzyx;
    dlal::vec<T, 4> p3 = mask2 * r.storage.y * q.storage.zwxy;
    dlal::vec<T, 4> p4 = mask3 * r.storage.z * q.storage.yxwz;

    return dlal::quat(p1 + p2 + p3 + p4);
}

template <class T>
vec<T, 3> operator*(quaternion<T> const& q, vec<T, 3> const& v) {
    vec<T, 3> const lqv = q.storage.xyz;
    vec<T, 3> const uv(cross(lqv, v));
    vec<T, 3> const uuv(cross(lqv, uv));

    return v + ((uv * q.w) + uuv) * static_cast<T>(2);
}


// Operations ==================================================================
template <class T>
T length(quaternion<T> const& q) {
    return length(vec<T, 4>(q));
}

template <class T>
quaternion<T> normalize(quaternion<T> const& q) {
    return quaternion<T>(normalize(q.storage));
}

template <class T>
quaternion<T> conjugate(quaternion<T> const& q) {
    return quaternion<T>(q.w, -vec<T, 3>(q.storage));
}

template <class T>
quaternion<T> inverse(quaternion<T> const& q) {
    return conjugate(q) / dot(q.storage, q.storage);
}

// Conversion ==================================================================


/// \brief Convert a UNIT quaternion to a mat3
template <class T>
mat<T, 3, 3> mat3_from_unit_quaternion(quaternion<T> const& q) {
    T const qxx(q.x * q.x);
    T const qyy(q.y * q.y);
    T const qzz(q.z * q.z);
    T const qxz(q.x * q.z);
    T const qxy(q.x * q.y);
    T const qyz(q.y * q.z);
    T const qwx(q.w * q.x);
    T const qwy(q.w * q.y);
    T const qwz(q.w * q.z);

    T const one(1);
    T const two(2);

    mat<T, 3, 3> ret;
    ret[0][0] = one - two * (qyy + qzz);
    ret[0][1] = two * (qxy + qwz);
    ret[0][2] = two * (qxz - qwy);

    ret[1][0] = two * (qxy - qwz);
    ret[1][1] = one - two * (qxx + qzz);
    ret[1][2] = two * (qyz + qwx);

    ret[2][0] = two * (qxz + qwy);
    ret[2][1] = two * (qyz - qwx);
    ret[2][2] = one - two * (qxx + qyy);
    return ret;
}

/// \brief Convert a UNIT quaternion to a mat3
template <class T>
mat<T, 4, 4> mat4_from_unit_quaternion(quaternion<T> const& q) {
    auto m3 = mat3_from_unit_quaternion(q);

    mat<T, 4, 4> ret;

    ret[0] = vector_detail::v3to4(m3[0]);
    ret[1] = vector_detail::v3to4(m3[1]);
    ret[2] = vector_detail::v3to4(m3[2]);
    ret[3] = vec4 { 0, 0, 0, 1 };

    ret[0].w = 0;
    ret[1].w = 0;
    ret[2].w = 0;

    return ret;
}

template <class T>
quaternion<T> quaternion_from_matrix(mat<T, 3, 3> const& m) {
    quaternion<T> q;

    float const trace = m[0][0] + m[1][1] + m[2][2];
    // printf("TRACE %f : %f %f %f\n", trace, m[0][0], m[1][1], m[2][2]);
    if (trace > 0) {
        float const s = 0.5f / sqrtf(trace + 1.0f);
        q.x           = (m[1][2] - m[2][1]) * s;
        q.y           = (m[2][0] - m[0][2]) * s;
        q.z           = (m[0][1] - m[1][0]) * s;
        q.w           = 0.25f / s;
    } else {
        if (m[0][0] > m[1][1] and m[0][0] > m[2][2]) {
            // printf("SW 1\n");
            float const s = 2.0f * sqrtf(1.0f + m[0][0] - m[1][1] - m[2][2]);
            q.x           = 0.25f * s;
            q.y           = (m[1][0] + m[0][1]) / s;
            q.z           = (m[2][0] + m[0][2]) / s;
            q.w           = (m[1][2] - m[2][1]) / s;
        } else if (m[1][1] > m[2][2]) {
            // printf("SW 2\n");
            float const s = 2.0f * sqrtf(1.0f + m[1][1] - m[0][0] - m[2][2]);
            q.x           = (m[1][0] + m[0][1]) / s;
            q.y           = 0.25f * s;
            q.z           = (m[2][1] + m[1][2]) / s;
            q.w           = (m[2][0] - m[0][2]) / s;
        } else {
            // printf("SW 3\n");
            float const s = 2.0f * sqrtf(1.0f + m[2][2] - m[0][0] - m[1][1]);
            q.x           = 0.25f * s;
            q.y           = (m[0][1] - m[1][0]) / s;
            q.z           = (m[2][0] + m[0][2]) / s;
            q.w           = (m[2][1] + m[1][2]) / s;
        }
    }

    // printf("L %f\n", length(q));

    return q;
}

template <class T>
quaternion<T> quaternion_from_matrix(mat<T, 4, 4> const& m) {
    return quaternion_from_matrix(mat<T, 3, 3>(m));
}

// Other =======================================================================

///
/// \brief Compute a rotation between two vectors
/// \param from A normalized source vector
/// \param to A normalized destination vector
///
template <class T>
quaternion<T> rotation_from_to(vec<T, 3> const& from, vec<T, 3> const& to) {
    vec<T, 3> const w = cross(from, to);

    vec<T, 4> lq { w.x, w.y, w.z, dot(from, to) };

    lq.w += dot(lq, lq);
    return normalize(quaternion<T>(lq));
}


///
/// \brief Compute a rotation given a direction and an 'up' vector
/// \param norm_direction The direction to look in, must be normalized
/// \param norm_up The 'up' direction, must be normalized
///
template <class T>
quaternion<T> look_at_lh(vec<T, 3> const& norm_direction,
                         vec<T, 3> const& norm_up) {
    if (std::abs(dot(norm_direction, norm_up)) >= 1) {
        return rotation_from_to(vec<T, 3> { 0, 0, -1 }, norm_direction);
    }

    mat<T, 3, 3> ret;
    ret[0] = normalize(cross(norm_up, norm_direction));
    ret[1] = cross(norm_direction, ret[0]);
    ret[2] = norm_direction;

    return quaternion<T>(quaternion_from_matrix(ret));
}

///
/// \brief Compute a quaternion from Euler angles, expressed in radians
///
template <class T>
quaternion<T> from_angles(vec<T, 3> angles) {
    vec<T, 3> const c = cos(angles * T(0.5));
    vec<T, 3> const s = sin(angles * T(0.5));

    quaternion<T> ret;
    ret.x = s.x * c.y * c.z - c.x * s.y * s.z;
    ret.y = c.x * s.y * c.z + s.x * c.y * s.z;
    ret.z = c.x * c.y * s.z - s.x * s.y * c.z;
    ret.w = c.x * c.y * c.z + s.x * s.y * s.z;
    return ret;
}

///
/// \brief Compute a quaternion from an axis and a rotation, expressed in
/// radians
///
template <class T>
quaternion<T> from_angle_axis(T angle, vec<T, 3> axis) {
    T const s = std::sin(angle * static_cast<T>(0.5));

    return quaternion<T>(std::cos(angle * static_cast<T>(0.5)), axis * s);
}


} // namespace dlal

#endif // QUAT_H
#ifndef LINALG_PACKED_MAT_H
#define LINALG_PACKED_MAT_H


namespace dlal {

///
/// \brief The PackedMatrix class defines a column-major, size restricted
/// matrix. Contents are packed to be sizeof(T) * C * R.
///
template <class T, size_t C, size_t R>
struct packed_mat {
    using ColumnType = packed_vector<T, R>; ///< The column vector type
    using RowType    = packed_vector<T, C>; ///< The row vector type

    /// The underlying storage type
    using StorageType = std::array<ColumnType, C>;

    /// Storage as columns
    StorageType storage;

    static_assert(sizeof(storage) == (sizeof(T) * C * R));

public: // Basics
    ///
    /// \brief Get the total number of cells
    ///
    constexpr size_t size() { return C * R; }
    constexpr size_t row_count() { return R; }    ///< Count of rows (R)
    constexpr size_t column_count() { return C; } ///< Count of columns (C)

    /// @{
    /// \brief Access a column.
    constexpr ColumnType&       operator[](size_t i) { return storage[i]; }
    constexpr ColumnType const& operator[](size_t i) const {
        return storage[i];
    }
    /// @}

public:
    /// \brief Initialize all cells to zero
    constexpr packed_mat() : storage({}) { }

    /// \brief Default copy constructor
    constexpr packed_mat(packed_mat const&) = default;

    /// \brief Convert from a non-packed matrix. This will try to use a single
    /// copy if possible.
    constexpr packed_mat(mat<T, C, R> const& other) {
        if constexpr (mat<T, C, R>::is_contiguous) {
            std::copy(data(other),
                      data(other) + other.size(),
                      reinterpret_cast<T*>(&storage[0]));
        } else {
            for (size_t c = 0; c < C; c++) {
                (*this)[c] = other[c];
            }
        }
    }

    /// \brief Create a matrix from an array of values.
    constexpr packed_mat(std::array<float, C * R> const& a) : storage(a) { }

    /// \brief Initialize all cells to the given value
    constexpr packed_mat(T value) { storage.fill(value); }

    /// \brief Initialize values from a differently sized matrix, zeros
    /// otherwise.
    template <size_t C2, size_t R2>
    constexpr explicit packed_mat(mat<T, C2, R2> const& other) : storage({}) {
        constexpr auto cbound = std::min(C2, C);
        constexpr auto rbound = std::min(R2, R);
        for (size_t c = 0; c < cbound; c++) {
            for (size_t r = 0; r < rbound; r++) {
                (*this)[c][r] = other[c][r];
            }
        }
    }

public:
    /// \brief Default copy assignment
    packed_mat& operator=(packed_mat const& m) = default;

public:
    /// @{
    /// \brief Access the underlying storage as a contiguous array.
    T*       data() { return storage.data(); }
    T const* data() const { return storage.data(); }
    /// @}
};

// Typedefs ====================================================================

using packed_mat2 = packed_mat<float, 2, 2>;
using packed_mat3 = packed_mat<float, 3, 3>;
using packed_mat4 = packed_mat<float, 4, 4>;

using packed_mat22 = packed_mat<float, 2, 2>;
using packed_mat23 = packed_mat<float, 2, 3>;
using packed_mat24 = packed_mat<float, 2, 4>;

using packed_mat32 = packed_mat<float, 3, 2>;
using packed_mat33 = packed_mat<float, 3, 3>;
using packed_mat34 = packed_mat<float, 3, 4>;

using packed_mat42 = packed_mat<float, 4, 2>;
using packed_mat43 = packed_mat<float, 4, 3>;
using packed_mat44 = packed_mat<float, 4, 4>;

} // namespace dlal

#endif // LINALG_PACKED_MAT_H
#ifndef LINALG_MAT_TRANSFORM_H
#define LINALG_MAT_TRANSFORM_H


#include <cmath>

namespace dlal {

///
/// \brief Add a translation to a matrix, in place
///
template <class T>
mat<T, 4, 4> translate(mat<T, 4, 4> const& m, vec<T, 3> const& v) {
    mat<T, 4, 4> ret(m);
    ret[3] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3];
    return ret;
}

///
/// \brief Add a translation to a matrix, in place
///
template <class T>
void translate_in_place(mat<T, 4, 4>& m, vec<T, 3> const& v) {
    m[3] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3];
}


///
/// \brief Add a rotation to a given matrix by an axis and an angle
/// \param m       Matrix to operate on
/// \param radians Rotation angle, in radians
/// \param axis    Rotate around this (possibly not-normalized) axis
///
template <class T>
mat<T, 4, 4> rotate(mat<T, 4, 4> const& m, T radians, vec<T, 3> const& axis) {
    T const a = radians;
    T const c = std::cos(a);
    T const s = std::sin(a);

    vec<T, 3> const naxis(normalize(axis));
    vec<T, 3> const cos_pack((T(1) - c) * naxis);
    vec<T, 3> const sin_pack = naxis * s;

    auto rotation_part  = mat<T, 4, 4>(identity);
    rotation_part[0][0] = c + cos_pack.x * naxis.x;
    rotation_part[0][1] = cos_pack.x * naxis.y + sin_pack.z;
    rotation_part[0][2] = cos_pack.x * naxis.z - sin_pack.y;

    rotation_part[1][0] = cos_pack.y * naxis.x - sin_pack.z;
    rotation_part[1][1] = c + cos_pack.y * naxis.y;
    rotation_part[1][2] = cos_pack.y * naxis.z + sin_pack.x;

    rotation_part[2][0] = cos_pack.z * naxis.x + sin_pack.y;
    rotation_part[2][1] = cos_pack.z * naxis.y - sin_pack.x;
    rotation_part[2][2] = c + cos_pack.z * naxis.z;

    mat<T, 4, 4> ret;
    ret[0] = m[0] * rotation_part[0][0] + m[1] * rotation_part[0][1] +
             m[2] * rotation_part[0][2];
    ret[1] = m[0] * rotation_part[1][0] + m[1] * rotation_part[1][1] +
             m[2] * rotation_part[1][2];
    ret[2] = m[0] * rotation_part[2][0] + m[1] * rotation_part[2][1] +
             m[2] * rotation_part[2][2];
    ret[3] = m[3];
    return ret;
}


///
/// \brief Add a scale to a given matrix
///
template <class T>
mat<T, 4, 4> scale(mat<T, 4, 4> const& m, vec<T, 3> const& v) {
    mat<T, 4, 4> ret;
    ret[0] = m[0] * v[0];
    ret[1] = m[1] * v[1];
    ret[2] = m[2] * v[2];
    ret[3] = m[3];
    return ret;
}

///
/// \brief Add a scale, in place, to a given matrix
///
template <class T>
void scale_in_place(mat<T, 4, 4>& m, vec<T, 3> const& v) {
    m[0] = m[0] * v[0];
    m[1] = m[1] * v[1];
    m[2] = m[2] * v[2];
    m[3] = m[3];
}

} // namespace dlal

#endif // LINALG_MAT_TRANSFORM_H
#ifndef LINALG_MATRIX_OPERATIONS_H
#define LINALG_MATRIX_OPERATIONS_H


namespace dlal {

template <class T, size_t C, size_t R>
mat<T, R, C> transpose(mat<T, C, R> const& m) {
    mat<T, R, C> ret;
    for (size_t c = 0; c < C; c++) {
        for (size_t r = 0; r < R; r++) {
            ret[r][c] = m[c][r];
        }
    }
    return ret;
}

template <class T>
mat<T, 2, 2> transpose(mat<T, 2, 2> const& m) {
    mat<T, 2, 2> ret;
    ret[0][0] = m[0][0];
    ret[0][1] = m[1][0];
    ret[1][0] = m[0][1];
    ret[1][1] = m[1][1];
    return ret;
}

template <class T>
mat<T, 3, 3> transpose(mat<T, 3, 3> const& m) {
    mat<T, 3, 3> ret;
    ret[0][0] = m[0][0];
    ret[0][1] = m[1][0];
    ret[0][2] = m[2][0];

    ret[1][0] = m[0][1];
    ret[1][1] = m[1][1];
    ret[1][2] = m[2][1];

    ret[2][0] = m[0][2];
    ret[2][1] = m[1][2];
    ret[2][2] = m[2][2];
    return ret;
}

template <class T>
mat<T, 4, 4> transpose(mat<T, 4, 4> const& m) {
    mat<T, 4, 4> ret;
    ret[0][0] = m[0][0];
    ret[0][1] = m[1][0];
    ret[0][2] = m[2][0];
    ret[0][3] = m[3][0];

    ret[1][0] = m[0][1];
    ret[1][1] = m[1][1];
    ret[1][2] = m[2][1];
    ret[1][3] = m[3][1];

    ret[2][0] = m[0][2];
    ret[2][1] = m[1][2];
    ret[2][2] = m[2][2];
    ret[2][3] = m[3][2];

    ret[3][0] = m[0][3];
    ret[3][1] = m[1][3];
    ret[3][2] = m[2][3];
    ret[3][3] = m[3][3];
    return ret;
}

template <class T>
inline T determinant(mat<T, 2, 2> const& m) {
    return m[0][0] * m[1][1] - m[1][0] * m[0][1];
}

template <class T>
inline T determinant(mat<T, 3, 3> const& m) {

    T const a = m[0][0];
    T const b = m[1][0];
    T const c = m[2][0];

    T const d = m[0][1];
    T const e = m[1][1];
    T const f = m[2][1];

    T const g = m[0][2];
    T const h = m[1][2];
    T const i = m[2][2];

    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

template <class T>
inline T determinant(mat<T, 4, 4> const& m) {
    T const a = m[2][2] * m[3][3] - m[3][2] * m[2][3];
    T const b = m[2][1] * m[3][3] - m[3][1] * m[2][3];
    T const c = m[2][1] * m[3][2] - m[3][1] * m[2][2];
    T const d = m[2][0] * m[3][3] - m[3][0] * m[2][3];
    T const e = m[2][0] * m[3][2] - m[3][0] * m[2][2];
    T const f = m[2][0] * m[3][1] - m[3][0] * m[2][1];

    vec<T, 4> const coeffs { +(m[1][1] * a - m[1][2] * b + m[1][3] * c),
                             -(m[1][0] * a - m[1][2] * d + m[1][3] * e),
                             +(m[1][0] * b - m[1][1] * d + m[1][3] * f),
                             -(m[1][0] * c - m[1][1] * e + m[1][2] * f) };

    return component_sum(m[0] * coeffs);
}


template <class T>
auto inverse(mat<T, 2, 2> const& m) {
    T const one_over_det = static_cast<T>(1) / determinant(m);

    return mat<T, 2, 2>(m[1][1] * one_over_det,
                        -m[0][1] * one_over_det,
                        -m[1][0] * one_over_det,
                        m[0][0] * one_over_det);
}

template <class T>
auto inverse(mat<T, 3, 3> const& m) {
    T const one_over_det = static_cast<T>(1) / determinant(m);

    mat<T, 3, 3> ret;
    ret[0][0] = +(m[1][1] * m[2][2] - m[2][1] * m[1][2]) * one_over_det;
    ret[1][0] = -(m[1][0] * m[2][2] - m[2][0] * m[1][2]) * one_over_det;
    ret[2][0] = +(m[1][0] * m[2][1] - m[2][0] * m[1][1]) * one_over_det;
    ret[0][1] = -(m[0][1] * m[2][2] - m[2][1] * m[0][2]) * one_over_det;
    ret[1][1] = +(m[0][0] * m[2][2] - m[2][0] * m[0][2]) * one_over_det;
    ret[2][1] = -(m[0][0] * m[2][1] - m[2][0] * m[0][1]) * one_over_det;
    ret[0][2] = +(m[0][1] * m[1][2] - m[1][1] * m[0][2]) * one_over_det;
    ret[1][2] = -(m[0][0] * m[1][2] - m[1][0] * m[0][2]) * one_over_det;
    ret[2][2] = +(m[0][0] * m[1][1] - m[1][0] * m[0][1]) * one_over_det;

    return ret;
}


inline vec4 _matrix2x2_multiply(vec4 vec1, vec4 vec2) {
    return vec1 * vec2.xwxw + (vec1.yxwz * vec2.zyzy);
}

inline vec4 _matrix2x2_adj_mult(vec4 vec1, vec4 vec2) {
    return (vec1.wwxx * vec2) - (vec1.yyzz * vec2.zwxy);
}


inline vec4 _matrix2x2_mult_adj(vec4 vec1, vec4 vec2) {
    return (vec1 * vec2.wxwx) - (vec1.yxwz * vec2.zyzy);
}

namespace matrix_detail {

inline auto extract_a(vec4 a, vec4 b) {
    return _mm_movelh_ps(a, b);
}
inline auto extract_b(vec4 a, vec4 b) {
    return _mm_movehl_ps(b, a);
}

} // namespace matrix_detail

inline auto transform_inverse(mat4 const& m) {
    using namespace vector_detail;
    using namespace matrix_detail;
    // implementation based off Eric Zhang's
    // https://lxjk.github.io/2017/09/03/Fast-4x4-Matrix-Inverse-with-SSE-SIMD-Explained.html
    constexpr float SMALL_VALUE = 1E-8F;

    mat4 ret;

    vec4 t0 = extract_a(m[0], m[1]);
    vec4 t1 = extract_b(m[0], m[1]);
    ret[0]  = SHUFFLE(t0, m[2], 0, 2, 4, 7);
    ret[1]  = SHUFFLE(t0, m[2], 1, 3, 5, 7);
    ret[2]  = SHUFFLE(t1, m[2], 0, 2, 6, 7);

    __m128 size_sqr = ret[0] * ret[0];
    size_sqr += ret[1] * ret[1];
    size_sqr += ret[2] * ret[2];

    __m128 one { 1.0f };
    __m128 rSizeSqr = _mm_blendv_ps(
        (one / size_sqr), one, _mm_cmplt_ps(size_sqr, __m128 { SMALL_VALUE }));

    ret[0] = (ret[0] * rSizeSqr);
    ret[1] = (ret[1] * rSizeSqr);
    ret[2] = (ret[2] * rSizeSqr);

    // last line
    ret[3] = (ret[0] * (m[3].xxxx));
    ret[3] = (ret[3] + (ret[1] * (m[3].yyyy)));
    ret[3] = (ret[3] + (ret[2] * (m[3].zzzz)));
    ret[3] = _mm_setr_ps(0.f, 0.f, 0.f, 1.f) - ret[3];

    return ret;
}

inline auto inverse(mat4 const& m) {
    using namespace vector_detail;
    using namespace matrix_detail;
    // implementation based off Eric Zhang's
    // https://lxjk.github.io/2017/09/03/Fast-4x4-Matrix-Inverse-with-SSE-SIMD-Explained.html

    vec4 const A = extract_a(m[0], m[1]);
    vec4 const B = extract_b(m[0], m[1]);
    vec4 const C = extract_a(m[2], m[3]);
    vec4 const D = extract_b(m[2], m[3]);

    vec4 const det_sub =
        (SHUFFLE(m[0], m[2], 0, 2, 4, 6) * SHUFFLE(m[1], m[3], 1, 3, 5, 7)) -
        (SHUFFLE(m[0], m[2], 1, 3, 5, 7) * SHUFFLE(m[1], m[3], 0, 2, 4, 6));
    vec4 const det_A = det_sub.xxxx;
    vec4 const det_B = det_sub.yyyy;
    vec4 const det_C = det_sub.zzzz;
    vec4 const det_D = det_sub.wwww;

    vec4 D_C = _matrix2x2_adj_mult(D, C);
    vec4 A_B = _matrix2x2_adj_mult(A, B);
    vec4 X   = (det_D * A) - _matrix2x2_multiply(B, D_C);
    vec4 Y   = (det_B * C) - _matrix2x2_mult_adj(D, A_B);
    vec4 Z   = (det_C * B) - _matrix2x2_mult_adj(A, D_C);
    vec4 W   = (det_A * D) - _matrix2x2_multiply(C, A_B);

    __m128 det_M = (det_A * det_D);
    det_M        = (det_M + (det_B * det_C));

    __m128 trace = (A_B * D_C.xzyw);
    trace        = _mm_hadd_ps(trace, trace);
    trace        = _mm_hadd_ps(trace, trace);
    det_M        = (det_M - trace);

    __m128 const adj_sign_mask = { 1.f, -1.f, -1.f, 1.f };
    __m128 const i_det_M       = (adj_sign_mask / det_M);

    X = (X * i_det_M);
    Y = (Y * i_det_M);
    Z = (Z * i_det_M);
    W = (W * i_det_M);

    mat4 ret;
    ret[0] = SHUFFLE(X, Y, 3, 1, 7, 5);
    ret[1] = SHUFFLE(X, Y, 2, 0, 6, 4);
    ret[2] = SHUFFLE(Z, W, 3, 1, 7, 5);
    ret[3] = SHUFFLE(Z, W, 2, 0, 6, 4);

    return ret;
}

} // namespace dlal

#endif // LINALG_MATRIX_OPERATIONS_H
#ifndef LINALG_TMATRIX_H
#define LINALG_TMATRIX_H


#include <cassert>

namespace dlal {


///
/// \brief The TMatrix class helps with transformations. It is the same size of
/// a Mat4, and also stores entries in a column-major form.
///
/// Note that everything is multiplied such that operations are done in reverse.
/// Thus if you want to create a transformation that first scales, rotates, then
/// translates, your code should add the translation, rotation, and finally
/// scale.
///
class Transformation {
    mat4 m_mat = mat4(identity);

public:
    /// \brief Create a default transformation matrix (the identity)
    Transformation() = default;

    /// \brief Create a matrix with the given rotation and scaling matrix
    explicit Transformation(mat3 const& f) : m_mat(f) { m_mat[3][3] = 1; }

    /// \brief Create a transformation matrix using a provided 4x4 mat.
    explicit Transformation(mat4 const& f) : m_mat(f) { }

    /// \brief Create a matrix with a rotation.
    explicit Transformation(quat const& q)
        : m_mat(mat4_from_unit_quaternion(q)) { }

    /// \brief Create a matrix with an array of values
    explicit Transformation(std::array<float, 16> const& f) : m_mat(f) { }

public: // setters
    /// \brief Add a translation by x,y,z
    void translate(float x, float y, float z);
    /// \brief Add a translation by a vector
    void translate(vec3 const&);
    /// \brief Clear the translation component
    void clear_translate();

    /// \brief Add a rotation by radians along the provided 3d axis
    void rotate(float radians, float x, float y, float z);
    /// \brief Add a rotation by a quaternion
    void rotate(quat const&);

    /// \brief Add a scale by factors in x,y,z
    void scale(float x, float y, float z);
    /// \brief Add a scale by factors in vector form
    void scale(vec3 const&);
    /// \brief Add an isotropic scale
    void scale(float);

    /// \brief Clear the rotation and scale portion of the transform
    void clear_rotation_scale();

    /// \brief Set a column to a vec3. The vector is zero-extended
    void set_column(size_t col, vec3 const&);
    /// \brief Set a column to a vec4.
    void set_column(size_t col, vec4 const&);

    /// \brief Get a column
    vec4 column(size_t col) const;

    /// \brief Set a row to a vec3. The vector is zero-extended
    void set_row(size_t row, vec3 const&);
    /// \brief Set a row to a vec4.
    void set_row(size_t row, vec4 const&);

    /// \brief Get a row
    vec4 row(size_t row) const;


    /// \brief Obtain an inverse transform
    Transformation inverted() const;
    /// \brief Invert this transform in-place
    void invert();

    /// \brief Obtain an transposed transform
    Transformation transposed() const;
    /// \brief Transpose this transform in-place
    void transpose();

public: // operation
    /// \brief Transform a vec3. The transform occurs using homogenous
    /// coordinates
    vec3 operator*(vec3 const&) const;
    /// \brief Transform a vec4.
    vec4 operator*(vec4 const&) const;

    /// \brief Add the transformations from another TMatrix
    Transformation operator*(Transformation const&) const;

    /// \brief Transform a vector, without translation
    vec3 rotate_scale_only(vec3 const&) const;

    /// \brief Obtain a transformation matrix with only the rotation and scale
    /// components of this transform.
    Transformation get_rotate_scale_only() const;


public: // access
    /// Convert to a rotation and scale only matrix
    explicit operator mat3() const;

    /// Convert to a mat4 (no-op)
    explicit operator mat4() const;

    /// Convert to a contiguous array
    explicit operator std::array<float, 16>() const;

    /// @{
    /// \brief Access the underlying storage as a contiguous array.
    float*       data();
    float const* data() const;
    /// @}
};

///
/// \brief Make a left handed perspective matrix.
///
/// Note that the depth transformation is to 0-1, thus unsuitable for OpenGL
///
/// \param fovy Field of view in y, radians
/// \param aspect Aspect ratio (x / y)
/// \param zNear Distance of near clipping plane. Must be positive and nonzero.
/// \param zFar Distance of far clipping plane. Must be positive.
///
Transformation
make_perspective_matrix_lh(float fovy, float aspect, float zNear, float zFar);

///
/// \brief Make a left handed frustrum matrix
///
/// Note that the depth transformation is to 0-1, thus unsuitable for OpenGL
///
/// \param left Left clipping plane
/// \param right Right clipping plane
/// \param bottom Bottom clipping plane
/// \param top Top clipping plane
/// \param near Distance of near clipping plane
/// \param far Distance of far clipping plane
///
Transformation make_frustum_matrix_lh(float left,
                                      float right,
                                      float bottom,
                                      float top,
                                      float near,
                                      float far);

///
/// \brief Make a left-handed orthogonal perspective matrix
/// Note that the depth transformation is to 0-1, thus unsuitable for OpenGL
///
/// \param left Left clipping plane
/// \param right Right clipping plane
/// \param bottom Bottom clipping plane
/// \param top Top clipping plane
/// \param near Distance of near clipping plane
/// \param far Distance of far clipping plane
///
Transformation make_ortho_matrix_lh(float left,
                                    float right,
                                    float bottom,
                                    float top,
                                    float zNear,
                                    float zFar);

///
/// \brief Make a left handed look-at transformation matrix
/// \param eye Coordinate of the eye
/// \param center Coordinate of the view target
/// \param up Vector that defines the 'up' direction. Must be normalized.
///
Transformation
make_look_at_lh(vec3 const& eye, vec3 const& center, vec3 const& up);

// Implementation ==============================================================


inline void Transformation::translate(vec3 const& v) {
    dlal::translate_in_place(m_mat, v);
}

inline void Transformation::translate(float x, float y, float z) {
    dlal::translate_in_place(m_mat, vec3 { x, y, z });
}

inline void Transformation::clear_translate() {
    auto const v = m_mat[3];
    m_mat[3]     = vec4 { 0, 0, 0, v.w };
}

inline void Transformation::rotate(float radians, float x, float y, float z) {
    vec3 v { x, y, z };
    m_mat = dlal::rotate(m_mat, radians, v);
}

inline void Transformation::rotate(quat const& q) {
    m_mat *= mat4_from_unit_quaternion(normalize(q));
}

inline void Transformation::scale(float x, float y, float z) {
    dlal::scale_in_place(m_mat, vec3 { x, y, z });
}

inline void Transformation::scale(vec3 const& v) {
    dlal::scale_in_place(m_mat, v);
}

inline void Transformation::scale(float f) {
    dlal::scale_in_place(m_mat, vec3 { f, f, f });
}

inline void Transformation::clear_rotation_scale() {
    m_mat.inset(mat3(identity));
}

inline void Transformation::set_column(size_t col, vec3 const& v) {
    m_mat[col] = vec4 { v.x, v.y, v.z, 0 };
}

inline void Transformation::set_column(size_t col, vec4 const& v) {
    m_mat[col] = vec4(v);
}


inline vec4 Transformation::column(size_t col) const {
    return m_mat[col];
}

inline void Transformation::set_row(size_t row, vec3 const& v) {
    set_row(row, vec4 { v.x, v.y, v.z, 0 });
}

inline void Transformation::set_row(size_t row, vec4 const& v) {
    m_mat[0][row] = v.x;
    m_mat[1][row] = v.y;
    m_mat[2][row] = v.z;
    m_mat[3][row] = v.w;
}

inline vec4 Transformation::row(size_t row) const {
    return vec4 { m_mat[0][row], m_mat[1][row], m_mat[2][row], m_mat[3][row] };
}

inline Transformation Transformation::inverted() const {
    return Transformation(dlal::inverse(m_mat));
}

inline void Transformation::invert() {
    m_mat = inverse(m_mat);
}

inline Transformation Transformation::transposed() const {
    return Transformation(dlal::transpose(m_mat));
}

inline void Transformation::transpose() {
    m_mat = dlal::transpose(m_mat);
}


inline vec3 Transformation::operator*(vec3 const& v) const {
    auto const r = operator*(vec4 { v.x, v.y, v.z, 1 });
    return vec3 { r.x, r.y, r.z } / r.w;
}

inline vec4 Transformation::operator*(vec4 const& v) const {
    return m_mat * v;
}

inline Transformation Transformation::operator*(Transformation const& m) const {
    return Transformation(m_mat * m.m_mat);
}

inline vec3 Transformation::rotate_scale_only(vec3 const& v) const {
    mat3 const lm = mat3(m_mat);
    return lm * v;
}

inline Transformation::operator mat3() const {
    return mat3(m_mat);
}

inline Transformation::operator mat4() const {
    return m_mat;
}

inline Transformation::operator std::array<float, 16>() const {
    static_assert(sizeof(mat4) == sizeof(std::array<float, 16>), "");
    return *reinterpret_cast<std::array<float, 16> const*>(&m_mat);
}

inline float* Transformation::data() {
    return dlal::data(m_mat);
}

inline float const* Transformation::data() const {
    return dlal::data(m_mat);
}

//

inline Transformation
make_perspective_matrix_lh(float fovy, float aspect, float zNear, float zFar) {
    assert(aspect > 0.0f);
    assert((zFar - zNear) > 0.0f);
    assert(zNear > 0.0f);
    assert(fovy > 0.0f);

    float const yscale = 1.0f / std::tan(fovy / 2.0f);
    float const xscale = yscale / aspect;

    auto mat  = mat4(0);
    mat[0][0] = xscale;
    mat[1][1] = -yscale;
    mat[2][3] = 1.0f;

    mat[2][2] = zFar / (zFar - zNear);
    mat[3][2] = -(zFar * zNear) / (zFar - zNear);

    return Transformation(mat);
}

inline Transformation make_frustum_matrix_lh(float left,
                                             float right,
                                             float bottom,
                                             float top,
                                             float near,
                                             float far) {
    auto mat  = mat4(0);
    mat[0][0] = 2.0f * near / (right - left);
    mat[1][1] = 2.0f * near / (top - bottom);
    mat[2][0] = (right + left) / (right - left);
    mat[2][1] = (top + bottom) / (top - bottom);
    mat[2][2] = far / (far - near);
    mat[2][3] = 1;
    mat[3][2] = -(far * near) / (far - near);
    return Transformation(mat);
}

inline Transformation make_ortho_matrix_lh(float left,
                                           float right,
                                           float bottom,
                                           float top,
                                           float zNear,
                                           float zFar) {
    auto mat  = mat4(0);
    mat[0][0] = 2.0f / (right - left);
    mat[1][1] = 2.0f / (top - bottom);
    mat[2][2] = 1.0f / (zFar - zNear);
    mat[3][3] = 1;
    mat[3][0] = -(right + left) / (right - left);
    mat[3][1] = -(top + bottom) / (top - bottom);
    mat[3][2] = -(zNear) / (zFar - zNear);
    return Transformation(mat);
}

inline Transformation
make_look_at_lh(vec3 const& eye, vec3 const& center, vec3 const& up) {
    vec3 const f(normalize(center - eye));
    vec3 const s(normalize(cross(up, f)));
    vec3 const u(cross(f, s));

    auto mat  = mat4(identity);
    mat[0][0] = s.x;
    mat[1][0] = s.y;
    mat[2][0] = s.z;
    mat[0][1] = u.x;
    mat[1][1] = u.y;
    mat[2][1] = u.z;
    mat[0][2] = f.x;
    mat[1][2] = f.y;
    mat[2][2] = f.z;
    mat[3][0] = -dot(s, eye);
    mat[3][1] = -dot(u, eye);
    mat[3][2] = -dot(f, eye);
    return Transformation(mat);
}

} // namespace dlal

#endif // LINALG_TMATRIX_H
