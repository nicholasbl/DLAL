#ifndef TEST_SINGLE_HEADER_H
#define TEST_SINGLE_HEADER_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include <smmintrin.h>
#include <xmmintrin.h>


namespace dct {

namespace vector_detail {

template <class T, unsigned N>
using vec __attribute__((vector_size(sizeof(T) * N))) = T;

using ivec4 = vec<int, 4>;
using vec4  = vec<float, 4>;

// vec4 should be equivalent to __m128

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

// =============================================================================

namespace dct {

/// \brief The basic vector class; specializations define the 1-4 component
/// cases.
template <class T, size_t N>
class Vector {};

// Vector 1 ====================================================================

template <class T>
class Vector<T, 1> {
    using StorageType = std::array<T, 1>;

    StorageType storage;

public: // Basics
    constexpr size_t size() { return storage.size(); }

    constexpr T&       operator[](size_t i) { return storage[i]; }
    constexpr T const& operator[](size_t i) const { return storage[i]; }

public:
    /// \brief Initialize all elements to zero
    constexpr Vector() : storage() {}
    constexpr Vector(Vector const&) = default;

    /// \brief Initialize all elements to the provided value
    constexpr Vector(T _) { storage.fill(_); }
    constexpr Vector(StorageType st) : storage(st) {}

    constexpr Vector& operator=(Vector const& v) = default;

public:
    T*       data() { return storage.data(); }
    T const* data() const { return storage.data(); }

    constexpr auto begin() { return storage.begin(); }
    constexpr auto begin() const { return storage.begin(); }
    constexpr auto end() { return storage.end(); }
    constexpr auto end() const { return storage.end(); }
};

// Vector 2 ====================================================================

template <class T>
struct Vector<T, 2> {
    using StorageType = std::array<T, 2>;

    union {
        StorageType storage;
        struct {
            T x, y;
        };
    };

public: // Basics
    constexpr size_t size() { return storage.size(); }

    constexpr T&       operator[](size_t i) { return storage[i]; }
    constexpr T const& operator[](size_t i) const { return storage[i]; }

public:
    /// \brief Initialize all elements to zero
    constexpr Vector() : storage() {}
    constexpr Vector(Vector const&) = default;

    /// \brief Initialize all elements to the provided value
    constexpr Vector(T _xy) : Vector(_xy, _xy) {}
    constexpr Vector(StorageType st) : storage(st) {}
    constexpr Vector(T _x, T _y) : storage{ _x, _y } {}

public: // conversion
    // downgrade
    constexpr Vector(Vector<T, 3> const& o) {
        x = o.x;
        y = o.y;
    }

    constexpr Vector(Vector<T, 4> const& o) {
        x = o.x;
        y = o.y;
    }

public:
    constexpr Vector& operator=(Vector const& v) = default;

public:
    T*       data() { return storage.data(); }
    T const* data() const { return storage.data(); }

    constexpr auto begin() { return storage.begin(); }
    constexpr auto begin() const { return storage.begin(); }
    constexpr auto end() { return storage.end(); }
    constexpr auto end() const { return storage.end(); }
};

static_assert(sizeof(Vector<float, 2>) == sizeof(float) * 2);

// Vector 3 ====================================================================

template <class T>
struct Vector<T, 3> {
    using StorageType = std::array<T, 3>;

    union {
        StorageType storage;
        struct {
            T x, y, z;
        };
    };

public: // Basics
    constexpr size_t size() { return storage.size(); }

    constexpr T&       operator[](size_t i) { return storage[i]; }
    constexpr T const& operator[](size_t i) const { return storage[i]; }

public:
    /// \brief Initialize all elements to zero
    constexpr Vector() : storage() {}
    constexpr Vector(Vector const&) = default;

    /// \brief Initialize all elements to the provided value
    constexpr Vector(T _xyz) : Vector(_xyz, _xyz, _xyz) {}
    constexpr Vector(StorageType st) : storage(st) {}
    constexpr Vector(T _x, T _y, T _z) : storage{ _x, _y, _z } {}

public: // conversion
    // upgrade
    constexpr explicit Vector(Vector<T, 2> const& o, T nz)
        : Vector(o.x, o.y, nz) {}

    constexpr explicit Vector(T nx, Vector<T, 2> const& o)
        : Vector(nx, o.x, o.y) {}

    // downgrade
    constexpr Vector(Vector<T, 4> const& o) : Vector(o.x, o.y, o.z) {}

public:
    constexpr Vector& operator=(Vector const& v) = default;

public:
    T*       data() { return storage.data(); }
    T const* data() const { return storage.data(); }

    constexpr auto begin() { return storage.begin(); }
    constexpr auto begin() const { return storage.begin(); }
    constexpr auto end() { return storage.end(); }
    constexpr auto end() const { return storage.end(); }
};

static_assert(sizeof(Vector<float, 3>) == sizeof(float) * 3);

// Vector 4 ====================================================================

template <class T>
struct Vector<T, 4> {
    using StorageType = std::array<T, 4>;

    union {
        StorageType storage;
        struct {
            T x, y, z, w;
        };
    };

public: // Basics
    constexpr size_t size() { return storage.size(); }

    constexpr T&       operator[](size_t i) { return storage[i]; }
    constexpr T const& operator[](size_t i) const { return storage[i]; }

public:
    /// \brief Initialize all elements to zero
    constexpr Vector() : storage() {}
    constexpr Vector(Vector const&) = default;

    /// \brief Initialize all elements to the provided value
    constexpr Vector(T _xyzw) : Vector(_xyzw, _xyzw, _xyzw, _xyzw) {}
    constexpr Vector(StorageType st) : storage(st) {}
    constexpr Vector(T _x, T _y, T _z, T _w) : storage{ _x, _y, _z, _w } {}

public: // conversion
    // upgrade

    // with a 2d
    constexpr explicit Vector(Vector<T, 2> const& o, T nz, T nw)
        : Vector(o.x, o.y, nz, nw) {}

    constexpr explicit Vector(T nx, Vector<T, 2> const& o, T nw)
        : Vector(nx, o.x, o.y, nw) {}

    constexpr explicit Vector(T nx, T ny, Vector<T, 2> const& o)
        : Vector(nx, ny, o.x, o.y) {}

    // with a 3d
    constexpr explicit Vector(Vector<T, 3> const& o, T nw)
        : Vector(o.x, o.y, o.z, nw) {}

    constexpr explicit Vector(T nx, Vector<T, 3> const& o)
        : Vector(nx, o.x, o.y, o.z) {}

public:
    constexpr Vector& operator=(Vector const& v) = default;

public:
    T*       data() { return storage.data(); }
    T const* data() const { return storage.data(); }

    constexpr auto begin() { return storage.begin(); }
    constexpr auto begin() const { return storage.begin(); }
    constexpr auto end() { return storage.end(); }
    constexpr auto end() const { return storage.end(); }
};

// Vector 4 Float ==============================================================

template <>
struct alignas(16) Vector<float, 4> {
    using StorageType = std::array<float, 4>;

    union {
        StorageType         storage;
        vector_detail::vec4 as_simd;
        struct {
            float x, y, z, w;
        };
    };

    static_assert(sizeof(storage) == sizeof(float) * 4);

public: // Basics
    constexpr size_t size() { return 4; }

    constexpr float&       operator[](size_t i) { return storage[i]; }
    constexpr float const& operator[](size_t i) const { return storage[i]; }

public:
    /// \brief Initialize all elements to zero
    constexpr Vector() : storage() {}
    constexpr Vector(Vector const&) = default;

    /// \brief Initialize all elements to the provided value
    constexpr Vector(float _xyzw) : Vector(_xyzw, _xyzw, _xyzw, _xyzw) {}
    constexpr Vector(vector_detail::vec4 const& st) : as_simd(st) {}
    constexpr Vector(StorageType st) : storage(st) {}
    constexpr Vector(float _x, float _y, float _z, float _w)
        : as_simd{ _x, _y, _z, _w } {}

public: // conversion
    // upgrade
    // with a 2d
    constexpr explicit Vector(Vector<float, 2> const& o, float nz, float nw)
        : Vector(o.x, o.y, nz, nw) {}

    constexpr explicit Vector(float nx, Vector<float, 2> const& o, float nw)
        : Vector(nx, o.x, o.y, nw) {}

    constexpr explicit Vector(float nx, float ny, Vector<float, 2> const& o)
        : Vector(nx, ny, o.x, o.y) {}

    // with a 3d
    constexpr explicit Vector(Vector<float, 3> const& o, float nw)
        : Vector(o.x, o.y, o.z, nw) {}

    constexpr explicit Vector(float nx, Vector<float, 3> const& o)
        : Vector(nx, o.x, o.y, o.z) {}

public:
    constexpr Vector& operator=(Vector const& v) = default;

public:
    float*       data() { return storage.data(); }
    float const* data() const { return storage.data(); }

    constexpr auto begin() { return storage.begin(); }
    constexpr auto begin() const { return storage.begin(); }
    constexpr auto end() { return storage.end(); }
    constexpr auto end() const { return storage.end(); }
};

// Vector Typedefs =============================================================

using BVec1 = Vector<bool, 2>;
using BVec2 = Vector<bool, 2>;
using BVec3 = Vector<bool, 3>;
using BVec4 = Vector<bool, 4>;

using I8Vec1 = Vector<int8_t, 2>;
using I8Vec2 = Vector<int8_t, 2>;
using I8Vec3 = Vector<int8_t, 3>;
using I8Vec4 = Vector<int8_t, 4>;

using I16Vec1 = Vector<int16_t, 2>;
using I16Vec2 = Vector<int16_t, 2>;
using I16Vec3 = Vector<int16_t, 3>;
using I16Vec4 = Vector<int16_t, 4>;

using IVec1 = Vector<int32_t, 2>;
using IVec2 = Vector<int32_t, 2>;
using IVec3 = Vector<int32_t, 3>;
using IVec4 = Vector<int32_t, 4>;

using I64Vec1 = Vector<int64_t, 2>;
using I64Vec2 = Vector<int64_t, 2>;
using I64Vec3 = Vector<int64_t, 3>;
using I64Vec4 = Vector<int64_t, 4>;

using Vec1 = Vector<float, 2>;
using Vec2 = Vector<float, 2>;
using Vec3 = Vector<float, 3>;
using Vec4 = Vector<float, 4>;

using DVec1 = Vector<double, 2>;
using DVec2 = Vector<double, 2>;
using DVec3 = Vector<double, 3>;
using DVec4 = Vector<double, 4>;

// Unary =======================================================================

template <class T, size_t N>
Vector<T, N> operator-(Vector<T, N> const& v) {
    VECTOR_UNARY(-)
}

inline Vec4 operator-(Vec4 const& v) { return Vec4(-v.as_simd); }

template <size_t N>
Vector<bool, N> operator!(Vector<bool, N> const& v) {
    using T = bool;
    VECTOR_UNARY(!)
}

// Operators ===================================================================

// Addition ====================================================================

template <class T, size_t N>
auto operator+(Vector<T, N> const& v, Vector<T, N> const& o) {
    VECTOR_BINARY(+)
}
template <class T, size_t N>
auto operator+(Vector<T, N> const& v, T scalar) {
    VECTOR_BINARY_SCALAR_R(+)
}
template <class T, size_t N>
auto operator+(T scalar, Vector<T, N> const& v) {
    VECTOR_BINARY_SCALAR_L(+)
}

template <class T, size_t N>
Vector<T, N>& operator+=(Vector<T, N>& v, Vector<T, N> const& o) {
    VECTOR_IN_PLACE(+=)
}
template <class T, size_t N>
Vector<T, N>& operator+=(Vector<T, N>& v, T scalar) {
    VECTOR_IN_PLACE_SCALAR_R(+=)
}

inline Vec4 operator+(Vec4 const& v, Vec4 const& o) {
    return v.as_simd + o.as_simd;
}
inline Vec4 operator+(Vec4 const& v, float scalar) {
    return v.as_simd + scalar;
}
inline Vec4 operator+(float scalar, Vec4 const& v) {
    return scalar + v.as_simd;
}

inline Vec4& operator+=(Vec4& v, Vec4 const& o) {
    v.as_simd += o.as_simd;
    return v;
}
inline Vec4& operator+=(float scalar, Vec4& v) {
    v.as_simd += scalar;
    return v;
}
inline Vec4& operator+=(Vec4& v, float scalar) {
    v.as_simd += scalar;
    return v;
}

// Subtraction =================================================================

template <class T, size_t N>
auto operator-(Vector<T, N> const& v, Vector<T, N> const& o) {
    VECTOR_BINARY(-)
}

template <class T, size_t N>
auto operator-(Vector<T, N> const& v, T scalar) {
    VECTOR_BINARY_SCALAR_R(-)
}

template <class T, size_t N>
auto operator-(T scalar, Vector<T, N> const& v) {
    VECTOR_BINARY_SCALAR_L(-)
}

template <class T, size_t N>
Vector<T, N>& operator-=(Vector<T, N>& v, Vector<T, N> const& o) {
    VECTOR_IN_PLACE(-=)
}

template <class T, size_t N>
Vector<T, N>& operator-=(Vector<T, N>& v, T scalar) {
    VECTOR_IN_PLACE_SCALAR_R(-=)
}

inline Vec4 operator-(Vec4 const& v, Vec4 const& o) {
    return v.as_simd - o.as_simd;
}
inline Vec4 operator-(Vec4 const& v, float scalar) {
    return v.as_simd - scalar;
}
inline Vec4 operator-(float scalar, Vec4 const& v) {
    return scalar - v.as_simd;
}

inline Vec4& operator-=(Vec4& v, Vec4 const& o) {
    v.as_simd -= o.as_simd;
    return v;
}
inline Vec4& operator-=(float scalar, Vec4& v) {
    v.as_simd -= scalar;
    return v;
}
inline Vec4& operator-=(Vec4& v, float scalar) {
    v.as_simd -= scalar;
    return v;
}

// Multiply ====================================================================

template <class T, size_t N>
auto operator*(Vector<T, N> const& v, Vector<T, N> const& o) {
    VECTOR_BINARY(*)
}
template <class T, size_t N>
auto operator*(Vector<T, N> const& v, T scalar) {
    VECTOR_BINARY_SCALAR_R(*)
}
template <class T, size_t N>
auto operator*(T scalar, Vector<T, N> const& v) {
    VECTOR_BINARY_SCALAR_L(*)
}

template <class T, size_t N>
Vector<T, N>& operator*=(Vector<T, N>& v, Vector<T, N> const& o) {
    VECTOR_IN_PLACE(*=)
}
template <class T, size_t N>
Vector<T, N>& operator*=(Vector<T, N>& v, T scalar) {
    VECTOR_IN_PLACE_SCALAR_R(*=)
}

inline Vec4 operator*(Vec4 const& v, Vec4 const& o) {
    return v.as_simd * o.as_simd;
}
inline Vec4 operator*(Vec4 const& v, float scalar) {
    return v.as_simd * scalar;
}
inline Vec4 operator*(float scalar, Vec4 const& v) {
    return scalar * v.as_simd;
}

inline Vec4& operator*=(Vec4& v, Vec4 const& o) {
    v.as_simd *= o.as_simd;
    return v;
}
inline Vec4& operator*=(float scalar, Vec4& v) {
    v.as_simd *= scalar;
    return v;
}
inline Vec4& operator*=(Vec4& v, float scalar) {
    v.as_simd *= scalar;
    return v;
}

// Division ====================================================================

template <class T, size_t N>
auto operator/(Vector<T, N> const& v, Vector<T, N> const& o) {
    VECTOR_BINARY(/)
}
template <class T, size_t N>
auto operator/(Vector<T, N> const& v, T scalar) {
    VECTOR_BINARY_SCALAR_R(/)
}
template <class T, size_t N>
auto operator/(T scalar, Vector<T, N> const& v) {
    VECTOR_BINARY_SCALAR_L(/)
}

template <class T, size_t N>
Vector<T, N>& operator/=(Vector<T, N>& v, Vector<T, N> const& o) {
    VECTOR_IN_PLACE(/=)
}
template <class T, size_t N>
Vector<T, N>& operator/=(Vector<T, N>& v, T scalar) {
    VECTOR_IN_PLACE_SCALAR_R(/=)
}

inline Vec4 operator/(Vec4 const& v, Vec4 const& o) {
    return v.as_simd / o.as_simd;
}
inline Vec4 operator/(Vec4 const& v, float scalar) {
    return v.as_simd / scalar;
}
inline Vec4 operator/(float scalar, Vec4 const& v) {
    return scalar / v.as_simd;
}

inline Vec4& operator/=(Vec4& v, Vec4 const& o) {
    v.as_simd /= o.as_simd;
    return v;
}
inline Vec4& operator/=(float scalar, Vec4& v) {
    v.as_simd /= scalar;
    return v;
}
inline Vec4& operator/=(Vec4& v, float scalar) {
    v.as_simd /= scalar;
    return v;
}

// Boolean =====================================================================

template <class T, size_t N>
Vector<bool, N> operator==(Vector<T, N> const& v, Vector<T, N> const& o) {
    VECTOR_BINARY_BOOL(==)
}

template <class T, size_t N>
Vector<bool, N> operator!=(Vector<T, N> const& v, Vector<T, N> const& o) {
    return !(v == o);
}

template <class T, size_t N>
Vector<bool, N> operator<(Vector<T, N> const& v, Vector<T, N> const& o) {
    VECTOR_BINARY_BOOL(<)
}

template <class T, size_t N>
Vector<bool, N> operator<=(Vector<T, N> const& v, Vector<T, N> const& o) {
    VECTOR_BINARY_BOOL(<=)
}

template <class T, size_t N>
Vector<bool, N> operator>(Vector<T, N> const& v, Vector<T, N> const& o) {
    VECTOR_BINARY_BOOL(>)
}

template <class T, size_t N>
Vector<bool, N> operator>=(Vector<T, N> const& v, Vector<T, N> const& o) {
    VECTOR_BINARY_BOOL(>=)
}

template <size_t N>
Vector<bool, N> operator&&(Vector<bool, N> const& v, Vector<bool, N> const& o) {
    VECTOR_BINARY_BOOL(&&)
}

template <size_t N>
Vector<bool, N> operator||(Vector<bool, N> const& v, Vector<bool, N> const& o) {
    VECTOR_BINARY_BOOL(||)
}

#undef VECTOR_UNARY
#undef VECTOR_BINARY_SCALAR_R
#undef VECTOR_BINARY_SCALAR_L
#undef VECTOR_BINARY
#undef VECTOR_IN_PLACE
#undef VECTOR_IN_PLACE_SCALAR_R
#undef VECTOR_IN_PLACE_SCALAR_L
#undef VECTOR_BINARY_BOOL

// Operations ==================================================================

template <class T, size_t N>
T dot(Vector<T, N> const& a, Vector<T, N> const& b) {
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
inline float dot(Vec4 const& a, Vec4 const& b) {
    return vector_detail::dot(a.as_simd, b.as_simd);
}

template <class T>
Vector<T, 3> cross(Vector<T, 3> const& a, Vector<T, 3> const& b) {
    return Vec3(
        a.y * b.z - b.y * a.z, a.z * b.x - b.z * a.x, a.x * b.y - b.x * a.y);
}


template <class T, size_t N>
T length(Vector<T, N> const& a) {
    return std::sqrt(dot(a, a));
}

template <class T, size_t N>
T length_squared(Vector<T, N> const& a) {
    return dot(a, a);
}

template <class T, size_t N>
T distance(Vector<T, N> const& a, Vector<T, N> const& b) {
    return length(b - a);
}

template <class T, size_t N>
T distance_squared(Vector<T, N> const& a, Vector<T, N> const& b) {
    return length_squared(b - a);
}

template <class T, size_t N>
Vector<T, N> normalize(Vector<T, N> const& a) {
    static_assert(std::is_floating_point_v<T>, "Floating point required");
    return a / length(a);
}

template <class T, size_t N>
Vector<T, N> reflect(Vector<T, N> const& a, Vector<T, N> const& normal) {
    static_assert(std::is_floating_point_v<T>, "Floating point required");
    return a - normal * dot(normal, a) * static_cast<T>(2);
}

// Additional Boolean ==========================================================
template <size_t N>
bool is_all(Vector<bool, N> const& a) {
    if constexpr (N == 1) {
        return a.x;
    } else if constexpr (N == 2) {
        return a.x and a.y;
    } else if constexpr (N == 3) {
        return a.x and a.y and a.z;
    } else if constexpr (N == 4) {
        return a.x and a.y and a.z and a.w;
    }
}

template <size_t N>
bool is_any(Vector<bool, N> const& a) {
    if constexpr (N == 1) {
        return a.x;
    } else if constexpr (N == 2) {
        return a.x or a.y;
    } else if constexpr (N == 3) {
        return a.x or a.y or a.z;
    } else if constexpr (N == 4) {
        return a.x or a.y or a.z or a.w;
    }
}

template <class T, size_t N>
bool is_equal(Vector<T, N> const& a, Vector<T, N> const& b) {
    return is_all(a == b);
}

template <class T, size_t N>
bool is_equal(Vector<T, N> const& a, Vector<T, N> const& b, T limit) {
    static_assert(std::is_floating_point_v<T>);

    auto         delta = abs(a - b);
    Vector<T, N> c(limit);

    return is_all(delta < c);
}

// Other =======================================================================
template <class T, size_t N>
Vector<T, N> min(Vector<T, N> const& a, Vector<T, N> const& b) {
    Vector<T, N> ret;
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
Vector<T, N> max(Vector<T, N> const& a, Vector<T, N> const& b) {
    Vector<T, N> ret;
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
T component_min(Vector<T, N> const& a) {
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
T component_max(Vector<T, N> const& a) {
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
T component_sum(Vector<T, N> const& a) {
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
Vector<T, N> abs(Vector<T, N> const& a) {
    Vector<T, N> ret;
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
Vector<T, N> floor(Vector<T, N> const& a) {
    Vector<T, N> ret;
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
Vector<T, N> ceil(Vector<T, N> const& a) {
    Vector<T, N> ret;
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
Vector<T, N> clamp(Vector<T, N> const& x, T const& min_val, T const& max_val) {
    Vector<T, N> ret;
    using namespace std;
    if constexpr (N == 1) {
        ret.x = clamp(x.x, min_val, max_val);
    } else if constexpr (N == 2) {
        ret.x = clamp(x.x, min_val, max_val);
        ret.y = clamp(x.y, min_val, max_val);
    } else if constexpr (N == 3) {
        ret.x = clamp(x.x, min_val, max_val);
        ret.y = clamp(x.y, min_val, max_val);
        ret.z = clamp(x.z, min_val, max_val);
    } else if constexpr (N == 4) {
        ret.x = clamp(x.x, min_val, max_val);
        ret.y = clamp(x.y, min_val, max_val);
        ret.z = clamp(x.z, min_val, max_val);
        ret.w = clamp(x.w, min_val, max_val);
    }
    return ret;
}

template <class T, size_t N>
Vector<T, N> clamp(Vector<T, N> const& x,
                   Vector<T, N> const& min_val,
                   Vector<T, N> const& max_val) {
    Vector<T, N> ret;
    using namespace std;
    if constexpr (N == 1) {
        ret.x = clamp(x.x, min_val.x, max_val.x);
    } else if constexpr (N == 2) {
        ret.x = clamp(x.x, min_val.x, max_val.x);
        ret.y = clamp(x.y, min_val.y, max_val.y);
    } else if constexpr (N == 3) {
        ret.x = clamp(x.x, min_val.x, max_val.x);
        ret.y = clamp(x.y, min_val.y, max_val.y);
        ret.z = clamp(x.z, min_val.z, max_val.z);
    } else if constexpr (N == 4) {
        ret.x = clamp(x.x, min_val.x, max_val.x);
        ret.y = clamp(x.y, min_val.y, max_val.y);
        ret.z = clamp(x.z, min_val.z, max_val.z);
        ret.w = clamp(x.w, min_val.w, max_val.w);
    }
    return ret;
}

template <class T, size_t N>
Vector<T, N> mix(Vector<T, N> const& a, Vector<T, N> const& b, bool t) {
    return t ? b : a;
}

template <class T>
T mix(T const& a, T const& b, bool t) {
    return t ? b : a;
}

template <class T, size_t N>
Vector<T, N>
mix(Vector<T, N> const& a, Vector<T, N> const& b, Vector<bool, N> const& t) {
    Vector<T, N> ret;
    for (size_t i = 0; i < N; ++i) {
        ret[i] = t[i] ? b[i] : a[i];
    }
    return ret;
}

template <class T, size_t N>
Vector<T, N> mix(Vector<T, N> const& a, Vector<T, N> const& b, T const& t) {
    return a + ((b - a) * t);
}

template <class T>
T mix(T const& a, T const& b, T const& t) {
    return a + ((b - a) * t);
}

template <class T, size_t N>
Vector<T, N>
mix(Vector<T, N> const& a, Vector<T, N> const& b, Vector<T, N> const& t) {
    return a + ((b - a) * t);
}


} // namespace dct

// =============================================================================

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

// =============================================================================

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

// =============================================================================

namespace dct {

///
/// \brief The MatrixStorage class defines the column-major storage of the
/// matrix
///
template <class T, size_t C, size_t R>
struct MatrixStorage {
    using ColumnType = Vector<T, R>;
    using RowType    = Vector<T, C>;

    using StorageType = std::array<ColumnType, C>;

    StorageType storage;

    static_assert(sizeof(StorageType) == sizeof(T) * C * R);

    constexpr MatrixStorage() : storage() {}
};

///
/// \brief This specialization attemps to force alignment to aid vector
/// instructions
///
template <>
struct alignas(16) MatrixStorage<float, 4, 4> {
    using ColumnType = Vector<float, 4>;
    using RowType    = Vector<float, 4>;

    using StorageType = std::array<ColumnType, 4>;

    static_assert(sizeof(__m128) == sizeof(vector_detail::vec4));

    union {
        StorageType         storage;
        vector_detail::vec4 as_vec[4];
    };

    static_assert(sizeof(StorageType) == sizeof(float) * 4 * 4);

    MatrixStorage() : storage() {}
};

///
/// \brief The root Matrix template
///
/// \tparam T The cell value type of the matrix
/// \tparam C The number of columns
/// \tparam R The number of rows
///
template <class T, size_t C, size_t R>
struct Matrix : public MatrixStorage<T, C, R> {

    using Parent = MatrixStorage<T, C, R>;

    using StorageType = typename MatrixStorage<T, C, R>::StorageType;
    using ColumnType  = typename MatrixStorage<T, C, R>::ColumnType;

    static_assert(sizeof(StorageType) == sizeof(T) * C * R);

public: // Basics
    constexpr size_t size() { return C * R; }
    constexpr size_t row_count() { return R; }
    constexpr size_t column_count() { return C; }

    // column major
    constexpr ColumnType& operator[](size_t i) { return Parent::storage[i]; }
    constexpr ColumnType const& operator[](size_t i) const {
        return Parent::storage[i];
    }

public:
    /// \brief Initialize all cells to zero
    constexpr Matrix() : Parent({}) {}

    constexpr Matrix(Matrix const&) = default;

    constexpr Matrix(std::array<float, C * R> const& a) {
        std::copy(a.data(), a.data() + a.size(), data());
    }

    /// \brief Initialize all cells to the given value
    constexpr Matrix(T value) { Parent::storage.fill(value); }

    constexpr Matrix(StorageType pack) : Parent::storage(pack) {}

    /// \brief Initialize values from a differently sized matrix, zeros
    /// otherwise.
    template <size_t C2, size_t R2>
    constexpr explicit Matrix(Matrix<T, C2, R2> const& other) {
        using namespace matrix_detail;
        constexpr size_t bound = C2 < C ? C2 : C;
        // no loops for speed in debug mode

        if constexpr (bound == 1) {
            Parent::storage[0] = upgrade<R>(other.storage[0]);
        } else if constexpr (bound == 2) {
            Parent::storage[0] = upgrade<R>(other.storage[0]);
            Parent::storage[1] = upgrade<R>(other.storage[1]);
        } else if constexpr (bound == 3) {
            Parent::storage[0] = upgrade<R>(other.storage[0]);
            Parent::storage[1] = upgrade<R>(other.storage[1]);
            Parent::storage[2] = upgrade<R>(other.storage[2]);
        } else if constexpr (bound == 4) {
            Parent::storage[0] = upgrade<R>(other.storage[0]);
            Parent::storage[1] = upgrade<R>(other.storage[1]);
            Parent::storage[2] = upgrade<R>(other.storage[2]);
            Parent::storage[3] = upgrade<R>(other.storage[3]);
        }
    }

public:
    /// \brief Copy values from a differently sized matrix.
    template <size_t C2, size_t R2>
    void assign(Matrix<T, C2, R2> const& other) {
        using namespace matrix_detail;
        constexpr size_t bound = C2 < C ? C2 : C;
        // no loops for speed in debug mode

        if constexpr (bound == 1) {
            Parent::storage[0] = upgrade<R>(other.storage[0]);
        } else if constexpr (bound == 2) {
            Parent::storage[0] = upgrade<R>(other.storage[0]);
            Parent::storage[1] = upgrade<R>(other.storage[1]);
        } else if constexpr (bound == 3) {
            Parent::storage[0] = upgrade<R>(other.storage[0]);
            Parent::storage[1] = upgrade<R>(other.storage[1]);
            Parent::storage[2] = upgrade<R>(other.storage[2]);
        } else if constexpr (bound == 4) {
            Parent::storage[0] = upgrade<R>(other.storage[0]);
            Parent::storage[1] = upgrade<R>(other.storage[1]);
            Parent::storage[2] = upgrade<R>(other.storage[2]);
            Parent::storage[3] = upgrade<R>(other.storage[3]);
        }
    }

public:
    /// \brief Obtain an identity matrix
    static Matrix const& identity() {
        static const Matrix ret = []() {
            Matrix l;

            for (size_t c = 0; c < C; c++) {
                for (size_t r = 0; r < R; r++) {
                    l[c][r] = (r == c) ? T(1) : T(0);
                }
            }

            return l;
        }();

        return ret;
    }

public:
    // storage is contiguous, and size is equivalent to a single array
    T*       data() { return reinterpret_cast<T*>(this); }
    T const* data() const { return reinterpret_cast<T const*>(this); }

    constexpr auto begin() { return data(); }
    constexpr auto begin() const { return data(); }
    constexpr auto end() { return data() + size(); }
    constexpr auto end() const { return data() + size(); }
};

// Typedefs ====================================================================

using Mat2 = Matrix<float, 2, 2>;
using Mat3 = Matrix<float, 3, 3>;
using Mat4 = Matrix<float, 4, 4>;

// Unary =======================================================================
template <class T, size_t C, size_t R>
Matrix<T, C, R> operator-(Matrix<T, C, R> const& m) {
    MATRIX_UNARY(-)
}

template <class T, size_t C, size_t R>
Matrix<T, C, R> operator!(Matrix<T, C, R> const& m) {
    MATRIX_UNARY(!)
}

// Operators ===================================================================

// Addition ====================================================================
template <class T, size_t C, size_t R>
auto operator+(Matrix<T, C, R> const& m, Matrix<T, C, R> const& o) {
    MATRIX_BINARY(+)
}
template <class T, size_t C, size_t R>
auto operator+(Matrix<T, C, R> const& m, T scalar) {
    MATRIX_BINARY_SCALAR_R(+)
}
template <class T, size_t C, size_t R>
auto operator+(T scalar, Matrix<T, C, R> const& m) {
    MATRIX_BINARY_SCALAR_L(+)
}

template <class T, size_t C, size_t R>
Matrix<T, C, R>& operator+=(Matrix<T, C, R>& m, Matrix<T, C, R> const& o) {
    MATRIX_IN_PLACE(+=)
}
template <class T, size_t C, size_t R>
Matrix<T, C, R>& operator+=(Matrix<T, C, R>& m, T scalar) {
    MATRIX_IN_PLACE_SCALAR_R(+=)
}

// Subtraction =================================================================

template <class T, size_t C, size_t R>
auto operator-(Matrix<T, C, R> const& m, Matrix<T, C, R> const& o) {
    MATRIX_BINARY(-)
}
template <class T, size_t C, size_t R>
auto operator-(Matrix<T, C, R> const& m, T scalar) {
    MATRIX_BINARY_SCALAR_R(-)
}
template <class T, size_t C, size_t R>
auto operator-(T scalar, Matrix<T, C, R> const& m) {
    MATRIX_BINARY_SCALAR_L(-)
}

template <class T, size_t C, size_t R>
Matrix<T, C, R>& operator-=(Matrix<T, C, R>& m, Matrix<T, C, R> const& o) {
    MATRIX_IN_PLACE(-=)
}
template <class T, size_t C, size_t R>
Matrix<T, C, R>& operator-=(Matrix<T, C, R>& m, T scalar) {
    MATRIX_IN_PLACE_SCALAR_R(-=)
}

// Multiply ====================================================================

template <class T, size_t N, size_t R, size_t C>
auto operator*(Matrix<T, N, R> const& m, Matrix<T, C, N> const& o) {
    static_assert(std::is_same_v<typename Matrix<T, N, R>::RowType,
                                 typename Matrix<T, C, N>::ColumnType>);
    Matrix<T, C, R> ret(0);

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
auto operator*(Matrix<T, C, R> const& m, Vector<T, R> const& o) {
    Vector<T, R> ret(0);

    for (size_t i = 0; i < C; ++i) {
        ret += m[i] * Vector<T, R>(o[i]);
    }

    return ret;
}


// loop free versions for the most common cases
inline auto operator*(Mat3 const& m, Mat3 const& o) {
    auto const m0 = m[0];
    auto const m1 = m[1];
    auto const m2 = m[2];

    auto const o0 = o[0];
    auto const o1 = o[1];
    auto const o2 = o[2];

    Mat3 ret;
    ret[0] = m0 * o0[0] + m1 * o0[1] + m2 * o0[2];
    ret[1] = m0 * o1[0] + m1 * o1[1] + m2 * o1[2];
    ret[2] = m0 * o2[0] + m1 * o2[1] + m2 * o2[2];
    return ret;
}

inline auto operator*(Mat4 const& m, Mat4 const& o) {
    auto const m0 = m[0];
    auto const m1 = m[1];
    auto const m2 = m[2];
    auto const m3 = m[3];

    auto const o0 = o[0];
    auto const o1 = o[1];
    auto const o2 = o[2];
    auto const o3 = o[3];

    Mat4 ret;
    ret[0] = m0 * o0[0] + m1 * o0[1] + m2 * o0[2] + m3 * o0[3];
    ret[1] = m0 * o1[0] + m1 * o1[1] + m2 * o1[2] + m3 * o1[3];
    ret[2] = m0 * o2[0] + m1 * o2[1] + m2 * o2[2] + m3 * o2[3];
    ret[3] = m0 * o3[0] + m1 * o3[1] + m2 * o3[2] + m3 * o3[3];
    return ret;
}

template <class T>
auto operator*(Matrix<T, 3, 3> const& m, Vector<T, 3> const& o) {
    Vector<T, 3> a0 = m[0] * Vector<T, 3>(o[0]);
    Vector<T, 3> a1 = m[1] * Vector<T, 3>(o[1]);

    auto m1 = a0 + a1;

    Vector<T, 3> a2 = m[2] * Vector<T, 3>(o[2]);

    return m1 + a2;
}

template <class T>
auto operator*(Matrix<T, 4, 4> const& m, Vector<T, 4> const& o) {
    Vector<T, 4> a0 = m[0] * Vector<T, 4>(o[0]);
    Vector<T, 4> a1 = m[1] * Vector<T, 4>(o[1]);

    auto m1 = a0 + a1;

    Vector<T, 4> a2 = m[2] * Vector<T, 4>(o[2]);
    Vector<T, 4> a3 = m[3] * Vector<T, 4>(o[3]);

    auto m2 = a2 + a3;

    return m1 + m2;
}


template <class T, size_t C, size_t R>
auto operator*(Matrix<T, C, R> const& m, T scalar) {
    MATRIX_BINARY_SCALAR_R(*)
}

template <class T, size_t C, size_t R>
auto operator*(T scalar, Matrix<T, C, R> const& m) {
    MATRIX_BINARY_SCALAR_L(*)
}


template <class T, size_t N, size_t R, size_t C>
Matrix<T, N, R> operator*=(Matrix<T, N, R>& m, Matrix<T, C, N> const& o) {
    return m = m * o;
}
template <class T, size_t C, size_t R>
Matrix<T, C, R>& operator*=(Matrix<T, C, R>& m, T scalar) {
    return m = m * scalar;
}
template <class T, size_t C, size_t R>
Matrix<T, C, R>& operator*=(T scalar, Matrix<T, C, R>& m) {
    return m = scalar * m;
}

// Division ====================================================================

template <class T, size_t C, size_t R>
auto operator/(Matrix<T, C, R> const& m, T scalar) {
    MATRIX_BINARY_SCALAR_R(/)
}

template <class T, size_t C, size_t R>
auto operator/(T scalar, Matrix<T, C, R> const& m) {
    MATRIX_BINARY_SCALAR_L(/)
}

// Boolean =====================================================================

template <class T, size_t C, size_t R>
auto operator==(Matrix<T, C, R> const& m, Matrix<T, C, R> const& o) {
    MATRIX_BINARY_BOOL(==)
}

template <class T, size_t C, size_t R>
auto operator!=(Matrix<T, C, R> const& m, Matrix<T, C, R> const& o) {
    MATRIX_BINARY_BOOL(!=)
}

template <class T, size_t C, size_t R>
auto operator&&(Matrix<bool, C, R> const& m, Matrix<bool, C, R> const& o) {
    MATRIX_BINARY_BOOL(&&)
}

template <class T, size_t C, size_t R>
auto operator||(Matrix<bool, C, R> const& m, Matrix<bool, C, R> const& o) {
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
template <size_t C, size_t R>
bool is_all(Matrix<bool, C, R> const& a) {
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

template <size_t C, size_t R>
bool is_any(Matrix<bool, C, R> const& a) {
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

template <class T, size_t C, size_t R>
bool is_equal(Matrix<T, C, R> const& a, Matrix<T, C, R> const& b) {
    return is_all(a == b);
}

template <class T, size_t C, size_t R>
bool is_equal(Matrix<T, C, R> const& a, Matrix<T, C, R> const& b, T limit) {
    static_assert(std::is_floating_point_v<T>);

    auto            delta = abs(a - b);
    Matrix<T, C, R> c(limit);

    return is_all(delta < c);
}

} // namespace dct

// =============================================================================

namespace dct {

template <class T>
struct Quaternion {
    union {
        Vector<T, 4> storage;
        struct {
            T x;
            T y;
            T z;
            T w;
        };
    };

public:
    Quaternion() : storage(0, 0, 0, 1) {}
    Quaternion(T w, Vector<T, 3> const& v) : storage(v, w) {}
    explicit Quaternion(Vector<T, 4> const& f) : storage(f) {}


public:
    explicit operator Vector<T, 4>() const { return storage; }
};

using Quat  = Quaternion<float>;
using DQuat = Quaternion<double>;


// Operators ===================================================================

template <class T>
Quaternion<T> operator+(Quaternion<T> const& q, Quaternion<T> const& r) {
    return Quaternion<T>(q.storage + r.storage);
}

template <class T>
Quaternion<T> operator-(Quaternion<T> const& q, Quaternion<T> const& r) {
    return Quaternion<T>(q.storage - r.storage);
}

template <class T>
Quaternion<T> operator*(Quaternion<T> const& q, T scalar) {
    return Quaternion<T>(q.storage * scalar);
}

// note that rotating a non-unit quaternion can do odd things

template <class T>
Quaternion<T> operator*(Quaternion<T> const& q, Quaternion<T> const& r) {

    Quaternion<T> ret;

    ret.x = q.w * r.x + q.x * r.w + q.y * r.z - q.z * r.y;
    ret.y = q.w * r.y + q.y * r.w + q.z * r.x - q.x * r.z;
    ret.z = q.w * r.z + q.z * r.w + q.x * r.y - q.y * r.x;
    ret.w = q.w * r.w - q.x * r.x - q.y * r.y - q.z * r.z;

    return ret;
}

template <class T>
Vector<T, 3> operator*(Quaternion<T> const& q, Vector<T, 3> const& v) {
    Vector<T, 3> const lqv(q.x, q.y, q.z);
    Vector<T, 3> const uv(cross(lqv, v));
    Vector<T, 3> const uuv(cross(lqv, uv));

    return v + ((uv * q.w) + uuv) * static_cast<T>(2);
}


// Operations ==================================================================
template <class T>
T length(Quaternion<T> const& q) {
    return length(Vector<T, 4>(q));
}

template <class T>
Quaternion<T> normalize(Quaternion<T> const& q) {
    return Quaternion<T>(normalize(q.storage));
}

template <class T>
Quaternion<T> conjugate(Quaternion<T> const& q) {
    return Quaternion<T>(q.w, -Vector<T, 3>(q.storage));
}

template <class T>
Quaternion<T> inverse(Quaternion<T> const& q) {
    return conjugate(q) / dot(q.storage, q.storage);
}

// Conversion ==================================================================


/// \brief Convert a UNIT quaternion to a mat3
template <class T>
Matrix<T, 3, 3> mat3_from_unit_quaternion(Quaternion<T> const& q) {
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

    Matrix<T, 3, 3> ret;
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
Matrix<T, 4, 4> mat4_from_unit_quaternion(Quaternion<T> const& q) {

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

    Matrix<T, 4, 4> ret(0);
    ret[0][0] = one - two * (qyy + qzz);
    ret[0][1] = two * (qxy + qwz);
    ret[0][2] = two * (qxz - qwy);

    ret[1][0] = two * (qxy - qwz);
    ret[1][1] = one - two * (qxx + qzz);
    ret[1][2] = two * (qyz + qwx);

    ret[2][0] = two * (qxz + qwy);
    ret[2][1] = two * (qyz - qwx);
    ret[2][2] = one - two * (qxx + qyy);

    ret[3][3] = one;
    return ret;
}

template <class T>
Quaternion<T> quaternion_from_matrix(Matrix<T, 3, 3> const& m) {
    Quaternion<T> q;

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
Quaternion<T> quaternion_from_matrix(Matrix<T, 4, 4> const& m) {
    return quaternion_from_matrix(Matrix<T, 3, 3>(m));
}

// Other =======================================================================

///
/// \brief Compute a rotation between two vectors
/// \param from A normalized source vector
/// \param to A normalized destination vector
///
template <class T>
Quaternion<T> rotation_from_to(Vector<T, 3> const& from,
                               Vector<T, 3> const& to) {
    Vector<T, 3> const w = cross(from, to);

    Vector<T, 4> lq(w.x, w.y, w.z, dot(from, to));

    lq.w += dot(lq, lq);
    return normalize(Quaternion<T>(lq));
}


///
/// \brief Compute a rotation given a direction and an 'up' vector
/// \param norm_direction The direction to look in, must be normalized
/// \param norm_up The 'up' direction, must be normalized
///
template <class T>
Quaternion<T> look_at_lh(Vector<T, 3> const& norm_direction,
                         Vector<T, 3> const& norm_up) {
    if (std::abs(dot(norm_direction, norm_up)) >= 1) {
        return rotation_from_to({ 0, 0, -1 }, norm_direction);
    }

    Matrix<T, 3, 3> ret;
    ret[0] = normalize(cross(norm_up, norm_direction));
    ret[1] = cross(norm_direction, ret[0]);
    ret[2] = norm_direction;

    return Quaternion<T>(quaternion_from_matrix(ret));
}

///
/// \brief Compute a quaternion from Euler angles, expressed in radians
///
template <class T>
Quaternion<T> from_angles(Vector<T, 3> angles) {
    Vector<T, 3> const c = cos(angles * T(0.5));
    Vector<T, 3> const s = sin(angles * T(0.5));

    Quaternion<T> ret;
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
Quaternion<T> from_angle_axis(T angle, Vector<T, 3> axis) {
    T const s = std::sin(angle * static_cast<T>(0.5));

    return Quaternion<T>(std::cos(angle * static_cast<T>(0.5)), axis * s);
}


} // namespace dct


#endif // TEST_SINGLE_HEADER_H
