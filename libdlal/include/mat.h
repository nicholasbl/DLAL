#ifndef LINALG_MATRIX_H
#define LINALG_MATRIX_H

#include "mat_detail.h"
#include "packed_vec.h"

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
