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
class packed_vector {};

// Vector 1 ====================================================================

template <class T>
class packed_vector<T, 1> {
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
    constexpr packed_vector() : storage() {}
    constexpr packed_vector(packed_vector const&) = default;

    /// \brief Initialize all elements to the provided value
    constexpr packed_vector(T _) { storage.fill(_); }
    constexpr packed_vector(StorageType st) : storage(st) {}

    constexpr packed_vector(vec<T, 1> simd) : x(simd.x) {}

    constexpr packed_vector& operator=(packed_vector const& v) = default;

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
struct packed_vector<T, 2> {
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
    constexpr packed_vector() : storage() {}
    constexpr packed_vector(packed_vector const&) = default;

    /// \brief Initialize all elements to the provided value
    constexpr packed_vector(T _xy) : packed_vector(_xy, _xy) {}
    constexpr packed_vector(StorageType st) : storage(st) {}
    constexpr packed_vector(T _x, T _y) : storage{ _x, _y } {}

    constexpr packed_vector(vec<T, 2> simd) : packed_vector(simd.x, simd.y) {}

    constexpr packed_vector& operator=(packed_vector const& v) = default;

    operator vec<T, 2>() const { return vec<T, 2>{ x, y }; }

public:
    T*       data() { return storage.data(); }
    T const* data() const { return storage.data(); }

    constexpr auto begin() { return storage.begin(); }
    constexpr auto begin() const { return storage.begin(); }
    constexpr auto end() { return storage.end(); }
    constexpr auto end() const { return storage.end(); }
};

static_assert(sizeof(packed_vector<float, 2>) == sizeof(float) * 2);

// Vector 3 ====================================================================

template <class T>
struct packed_vector<T, 3> {
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
    constexpr packed_vector() : storage() {}
    constexpr packed_vector(packed_vector const&) = default;

    /// \brief Initialize all elements to the provided value
    constexpr packed_vector(T _xyz) : packed_vector(_xyz, _xyz, _xyz) {}
    constexpr packed_vector(StorageType st) : storage(st) {}
    constexpr packed_vector(T _x, T _y, T _z) : storage{ _x, _y, _z } {}

    constexpr packed_vector(vec<T, 3> simd)
        : packed_vector(simd.x, simd.y, simd.z) {}

    constexpr packed_vector& operator=(packed_vector const& v) = default;

    operator vec<T, 3>() const { return vec<T, 3>{ x, y, z }; }

public:
    T*       data() { return storage.data(); }
    T const* data() const { return storage.data(); }

    constexpr auto begin() { return storage.begin(); }
    constexpr auto begin() const { return storage.begin(); }
    constexpr auto end() { return storage.end(); }
    constexpr auto end() const { return storage.end(); }
};

static_assert(sizeof(packed_vector<float, 3>) == sizeof(float) * 3);

// Vector 4 ====================================================================

template <class T>
struct packed_vector<T, 4> {
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
    constexpr packed_vector() : storage() {}
    constexpr packed_vector(packed_vector const&) = default;

    /// \brief Initialize all elements to the provided value
    constexpr packed_vector(T _xyzw)
        : packed_vector(_xyzw, _xyzw, _xyzw, _xyzw) {}
    constexpr packed_vector(StorageType st) : storage(st) {}
    constexpr packed_vector(T _x, T _y, T _z, T _w)
        : storage{ _x, _y, _z, _w } {}

    constexpr packed_vector(vec<T, 4> simd)
        : packed_vector(simd.x, simd.y, simd.z, simd.w) {}

    constexpr packed_vector& operator=(packed_vector const& v) = default;

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

using packed_dvec1 = packed_vector<double, 1>;
using packed_dvec2 = packed_vector<double, 2>;
using packed_dvec3 = packed_vector<double, 3>;
using packed_dvec4 = packed_vector<double, 4>;


} // namespace dct


#endif // LINALG_VECTOR_H
