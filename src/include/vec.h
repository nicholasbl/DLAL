#ifndef LINALG_VECTOR_H
#define LINALG_VECTOR_H

#include "vec_detail.h"

#include <algorithm>
#include <array>
#include <cmath>

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


#endif // LINALG_VECTOR_H
