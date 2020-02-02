#ifndef LINALG_VECTOR_DETAIL_H
#define LINALG_VECTOR_DETAIL_H

#include <array>
#include <cmath>
#include <cstddef>

#include <smmintrin.h>

namespace dct {

///
/// Vector equality uses -1 (all bits set)! If you want to use conditionals,
/// look at the is_all or is_any set of functions!
///
/// Note that the C++ conversion rules from int to bool use b = i == 0; Any
/// other value is true.
///
///

#ifndef __SSE4_1__
#error> SSE 4.1 Support is required
#endif

#define VEC_TRUE -1
#define VEC_FALSE 0

static_assert(__has_attribute(ext_vector_type));

template <class T, int N>
using vec __attribute__((ext_vector_type(N))) = T;

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

namespace vector_detail {


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

inline int bool_to_vec_bool(bool b) { return b * -1; }

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
    // insertelement for each. no idea why.
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


template <class T>
vec<T, 3> cross(vec<T, 3> const& a, vec<T, 3> const& b) {
    return vec<T, 3>{ a.y * b.z - b.y * a.z,
                      a.z * b.x - b.z * a.x,
                      a.x * b.y - b.x * a.y };
}

template <class T, int N>
T length(vec<T, N> const& a) {
    return std::sqrt(dot(a, a));
}

template <class T, int N>
T length_squared(vec<T, N> const& a) {
    return dot(a, a);
}

template <class T, int N>
T distance(vec<T, N> const& a, vec<T, N> const& b) {
    return length(b - a);
}

template <class T, int N>
T distance_squared(vec<T, N> const& a, vec<T, N> const& b) {
    return length_squared(b - a);
}

template <class T, int N>
vec<T, N> normalize(vec<T, N> const& a) {
    static_assert(std::is_floating_point_v<T>, "Floating point required");
    return a / length(a);
}

template <class T, int N>
vec<T, N> reflect(vec<T, N> const& a, vec<T, N> const& normal) {
    static_assert(std::is_floating_point_v<T>, "Floating point required");
    return a - normal * dot(normal, a) * static_cast<T>(2);
}

// Boolean =====================================================================

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

template <class T, int N>
bool is_equal(vec<T, N> const& a, vec<T, N> const& b) {
    return is_all(a == b);
}

template <class T, size_t N>
bool is_equal(vec<T, N> const& a, vec<T, N> const& b, T limit) {
    static_assert(std::is_floating_point_v<T>);

    auto      delta = abs(a - b);
    vec<T, N> c(limit);

    return is_all(delta < c);
}

template <class T, int N>
vec<T, N> select(vec<int, N> s, vec<T, N> a, vec<T, N> b) {
    vec<T, N> ret;
    for (int i = 0; i < N; i++) {
        ret[i] = s[i] ? a[i] : b[i];
    }
    return ret;
}

// Other =======================================================================
template <class T, size_t N>
vec<T, N> min(vec<T, N> const& a, vec<T, N> const& b) {
    vec<T, N> ret;
    using namespace vector_detail;
    if constexpr (N == 1) {
        ret.x = cmin(a.x, b.x);
    } else if constexpr (N == 2) {
        ret.x = cmin(a.x, b.x);
        ret.y = cmin(a.y, b.y);
    } else if constexpr (N == 3) {
        ret.x = cmin(a.x, b.x);
        ret.y = cmin(a.y, b.y);
        ret.z = cmin(a.z, b.z);
    } else if constexpr (N == 4) {
        ret.x = cmin(a.x, b.x);
        ret.y = cmin(a.y, b.y);
        ret.z = cmin(a.z, b.z);
        ret.w = cmin(a.w, b.w);
    }
    return ret;
}

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

template <class T, size_t N>
vec<T, N>
clamp(vec<T, N> const& x, vec<T, N> const& min_val, vec<T, N> const& max_val) {
    vec<T, N> ret = select(x < min_val, min_val, x);
    return select(ret > max_val, max_val, ret);
}


template <class T, size_t N>
vec<T, N> clamp(vec<T, N> const& x, T const& min_val, T const& max_val) {
    vec<T, N> lmin(min_val);
    vec<T, N> lmax(max_val);

    return clamp(x, lmin, lmax);
}

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
