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
    constexpr packed_vector() : storage() {}

    /// \brief Default copy constructor
    constexpr packed_vector(packed_vector const&) = default;

    /// \brief Initialize all elements to the provided value
    constexpr packed_vector(T _) { storage.fill(_); }

    /// \brief Construct from a std::array
    constexpr packed_vector(StorageType st) : storage(st) {}

    /// \brief Construct from a non-packed vector
    constexpr packed_vector(vec<T, 1> simd) : x(simd.x) {}

    /// \brief Default copy assignment
    constexpr packed_vector& operator=(packed_vector const& v) = default;

    /// \brief Convert to non-packed vector
    operator vec<T, 1>() const { return vec<T, 1>{ x }; }

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
    constexpr packed_vector() : storage() {}

    /// \brief Default copy constructor
    constexpr packed_vector(packed_vector const&) = default;

    /// \brief Initialize all elements to the provided value
    constexpr packed_vector(T _xy) : packed_vector(_xy, _xy) {}

    /// \brief Construct from a std::array
    constexpr packed_vector(StorageType st) : storage(st) {}

    /// \brief Construct from loose values
    constexpr packed_vector(T _x, T _y) : storage{ _x, _y } {}

    /// \brief Construct from a non-packed vector
    constexpr packed_vector(vec<T, 2> simd) : packed_vector(simd.x, simd.y) {}

    /// \brief Default copy assignment
    constexpr packed_vector& operator=(packed_vector const& v) = default;

    /// \brief Convert to non-packed vector
    operator vec<T, 2>() const { return vec<T, 2>{ x, y }; }

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
    constexpr packed_vector() : storage() {}

    /// \brief Default copy constructor
    constexpr packed_vector(packed_vector const&) = default;

    /// \brief Initialize all elements to the provided value
    constexpr packed_vector(T _xyz) : packed_vector(_xyz, _xyz, _xyz) {}

    /// \brief Construct from a std::array
    constexpr packed_vector(StorageType st) : storage(st) {}

    /// \brief Construct from loose values
    constexpr packed_vector(T _x, T _y, T _z) : storage{ _x, _y, _z } {}

    /// \brief Construct from a non-packed vector
    constexpr packed_vector(vec<T, 3> simd)
        : packed_vector(simd.x, simd.y, simd.z) {}

    /// \brief Default copy assignment
    constexpr packed_vector& operator=(packed_vector const& v) = default;

    /// \brief Convert to non-packed vector
    operator vec<T, 3>() const { return vec<T, 3>{ x, y, z }; }

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
    constexpr packed_vector() : storage() {}

    /// \brief Default copy constructor
    constexpr packed_vector(packed_vector const&) = default;

    /// \brief Initialize all elements to the provided value
    constexpr packed_vector(T _xyzw)
        : packed_vector(_xyzw, _xyzw, _xyzw, _xyzw) {}

    /// \brief Construct from a std::array
    constexpr packed_vector(StorageType st) : storage(st) {}

    /// \brief Construct from loose values
    constexpr packed_vector(T _x, T _y, T _z, T _w)
        : storage{ _x, _y, _z, _w } {}

    /// \brief Construct from a non-packed vector
    constexpr packed_vector(vec<T, 4> simd)
        : packed_vector(simd.x, simd.y, simd.z, simd.w) {}

    /// \brief Default copy assignment
    constexpr packed_vector& operator=(packed_vector const& v) = default;

    /// \brief Convert to non-packed vector
    operator vec<T, 4>() const { return vec<T, 4>{ x, y, z, w }; }

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


} // namespace dct


#endif // LINALG_VECTOR_H
