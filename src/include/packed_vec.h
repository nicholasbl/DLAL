#ifndef LINALG_VECTOR_H
#define LINALG_VECTOR_H

#include "vec.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace dct {

/// \brief The basic packed vector class; specializations define the 1-4
/// component cases. These should be used when storage sizes are important.
template <class T, size_t N>
class PackedVector {};

// Vector 1 ====================================================================

template <class T>
class PackedVector<T, 1> {
    using StorageType = std::array<T, 1>;

    union {
        StorageType storage;
        struct {
            T x;
        };
    };

public: // Basics
    constexpr size_t size() { return storage.size(); }

    constexpr T&       operator[](size_t i) { return storage[i]; }
    constexpr T const& operator[](size_t i) const { return storage[i]; }

public:
    /// \brief Initialize all elements to zero
    constexpr PackedVector() : storage() {}
    constexpr PackedVector(PackedVector const&) = default;

    /// \brief Initialize all elements to the provided value
    constexpr PackedVector(T _) { storage.fill(_); }
    constexpr PackedVector(StorageType st) : storage(st) {}

    constexpr PackedVector(vec<T, 1> simd) : x(simd.x) {}

    constexpr PackedVector& operator=(PackedVector const& v) = default;

    operator vec<T, 1>() const { return vec<T, 1>{ x }; }

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
struct PackedVector<T, 2> {
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
    constexpr PackedVector() : storage() {}
    constexpr PackedVector(PackedVector const&) = default;

    /// \brief Initialize all elements to the provided value
    constexpr PackedVector(T _xy) : PackedVector(_xy, _xy) {}
    constexpr PackedVector(StorageType st) : storage(st) {}
    constexpr PackedVector(T _x, T _y) : storage{ _x, _y } {}

    constexpr PackedVector(vec<T, 2> simd) : PackedVector(simd.x, simd.y) {}

    constexpr PackedVector& operator=(PackedVector const& v) = default;

    operator vec<T, 2>() const { return vec<T, 2>{ x, y }; }

public:
    T*       data() { return storage.data(); }
    T const* data() const { return storage.data(); }

    constexpr auto begin() { return storage.begin(); }
    constexpr auto begin() const { return storage.begin(); }
    constexpr auto end() { return storage.end(); }
    constexpr auto end() const { return storage.end(); }
};

static_assert(sizeof(PackedVector<float, 2>) == sizeof(float) * 2);

// Vector 3 ====================================================================

template <class T>
struct PackedVector<T, 3> {
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
    constexpr PackedVector() : storage() {}
    constexpr PackedVector(PackedVector const&) = default;

    /// \brief Initialize all elements to the provided value
    constexpr PackedVector(T _xyz) : PackedVector(_xyz, _xyz, _xyz) {}
    constexpr PackedVector(StorageType st) : storage(st) {}
    constexpr PackedVector(T _x, T _y, T _z) : storage{ _x, _y, _z } {}

    constexpr PackedVector(vec<T, 3> simd)
        : PackedVector(simd.x, simd.y, simd.z) {}

    constexpr PackedVector& operator=(PackedVector const& v) = default;

    operator vec<T, 3>() const { return vec<T, 3>{ x, y, z }; }

public:
    T*       data() { return storage.data(); }
    T const* data() const { return storage.data(); }

    constexpr auto begin() { return storage.begin(); }
    constexpr auto begin() const { return storage.begin(); }
    constexpr auto end() { return storage.end(); }
    constexpr auto end() const { return storage.end(); }
};

static_assert(sizeof(PackedVector<float, 3>) == sizeof(float) * 3);

// Vector 4 ====================================================================

template <class T>
struct PackedVector<T, 4> {
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
    constexpr PackedVector() : storage() {}
    constexpr PackedVector(PackedVector const&) = default;

    /// \brief Initialize all elements to the provided value
    constexpr PackedVector(T _xyzw)
        : PackedVector(_xyzw, _xyzw, _xyzw, _xyzw) {}
    constexpr PackedVector(StorageType st) : storage(st) {}
    constexpr PackedVector(T _x, T _y, T _z, T _w)
        : storage{ _x, _y, _z, _w } {}

    constexpr PackedVector(vec<T, 4> simd)
        : PackedVector(simd.x, simd.y, simd.z, simd.w) {}

    constexpr PackedVector& operator=(PackedVector const& v) = default;

    operator vec<T, 4>() const { return vec<T, 4>{ x, y, z, w }; }

public:
    T*       data() { return storage.data(); }
    T const* data() const { return storage.data(); }

    constexpr auto begin() { return storage.begin(); }
    constexpr auto begin() const { return storage.begin(); }
    constexpr auto end() { return storage.end(); }
    constexpr auto end() const { return storage.end(); }
};

// Vector Typedefs =============================================================

using Packed_bvec1 = PackedVector<bool, 1>;
using Packed_bvec2 = PackedVector<bool, 2>;
using Packed_bvec3 = PackedVector<bool, 3>;
using Packed_bvec4 = PackedVector<bool, 4>;

using Packed_i8vec1 = PackedVector<int8_t, 1>;
using Packed_i8vec2 = PackedVector<int8_t, 2>;
using Packed_i8vec3 = PackedVector<int8_t, 3>;
using Packed_i8vec4 = PackedVector<int8_t, 4>;

using Packed_i16vec1 = PackedVector<int16_t, 1>;
using Packed_i16vec2 = PackedVector<int16_t, 2>;
using Packed_i16vec3 = PackedVector<int16_t, 3>;
using Packed_i16vec4 = PackedVector<int16_t, 4>;

using Packed_ivec1 = PackedVector<int32_t, 1>;
using Packed_ivec2 = PackedVector<int32_t, 2>;
using Packed_ivec3 = PackedVector<int32_t, 3>;
using Packed_ivec4 = PackedVector<int32_t, 4>;

using Packed_i64vec1 = PackedVector<int64_t, 1>;
using Packed_i64vec2 = PackedVector<int64_t, 2>;
using Packed_i64vec3 = PackedVector<int64_t, 3>;
using Packed_i64vec4 = PackedVector<int64_t, 4>;

using Packed_vec1 = PackedVector<float, 1>;
using Packed_vec2 = PackedVector<float, 2>;
using Packed_vec3 = PackedVector<float, 3>;
using Packed_vec4 = PackedVector<float, 4>;

using Packed_dvec1 = PackedVector<double, 1>;
using Packed_dvec2 = PackedVector<double, 2>;
using Packed_dvec3 = PackedVector<double, 3>;
using Packed_dvec4 = PackedVector<double, 4>;


} // namespace dct


#endif // LINALG_VECTOR_H
