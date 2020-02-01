#ifndef LINALG_MATRIX_H
#define LINALG_MATRIX_H

#include "mat_detail.h"
#include "packed_vec.h"

#include <xmmintrin.h>

namespace dct {

///
/// \brief The MatrixStorage class defines the column-major storage of the
/// matrix. No size guarantees are made.
///
template <class T, size_t C, size_t R>
struct MatrixStorage {
    using ColumnType = vec<T, R>;
    using RowType    = vec<T, C>;

    using StorageType = std::array<ColumnType, C>;

    StorageType storage;

    static constexpr bool is_contiguous =
        sizeof(storage) == (sizeof(T) * C * R);

    constexpr MatrixStorage() : storage() {}
    constexpr MatrixStorage(MatrixStorage const& o) : storage(o.storage) {}
};

///
/// \brief The PackedMatrixStorage class defines the column-major storage of the
/// matrix. Contents are packed to be sizeof(T) * C * R.
///
template <class T, size_t C, size_t R>
struct PackedMatrixStorage {
    using ColumnType = PackedVector<T, R>;
    using RowType    = PackedVector<T, C>;

    using StorageType = std::array<ColumnType, C>;

    StorageType storage;

    static_assert(sizeof(StorageType) == sizeof(T) * C * R);

    constexpr PackedMatrixStorage() : storage() {}
    constexpr PackedMatrixStorage(PackedMatrixStorage const& o)
        : storage(o.storage) {}
};

template <class T, size_t C, size_t R>
struct mat;

template <class T, size_t C, size_t R>
struct packed_mat;

///
/// \brief The root Matrix template
///
/// \tparam T The cell value type of the matrix
/// \tparam C The number of columns
/// \tparam R The number of rows
///
template <class T, size_t C, size_t R>
struct mat : public MatrixStorage<T, C, R> {

    using Parent = MatrixStorage<T, C, R>;

    using StorageType = typename MatrixStorage<T, C, R>::StorageType;
    using ColumnType  = typename MatrixStorage<T, C, R>::ColumnType;

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
    constexpr mat() : Parent({}) {}

    constexpr mat(mat const&) = default;

    constexpr mat(std::array<float, C * R> const& a) {
        if constexpr (Parent::is_contiguous) {
            std::copy(a.data(),
                      a.data() + a.size(),
                      reinterpret_cast<T*>(&Parent::storage[0]));
        } else {
            for (size_t i = 0; i < C; ++i) {
                for (size_t j = 0; j < R; ++j) {
                    (*this)[i][j] = a[i * R + j];
                }
            }
        }
    }

    /// \brief Initialize all cells to the given value
    constexpr mat(T value) { Parent::storage.fill(value); }

    constexpr mat(StorageType pack) : Parent::storage(pack) {}

    /// \brief Initialize values from a differently sized matrix, zeros
    /// otherwise.
    template <size_t C2, size_t R2>
    constexpr explicit mat(mat<T, C2, R2> const& other) {
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
    void assign(mat<T, C2, R2> const& other) {
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
    static mat const& identity() {
        static const mat ret = []() {
            mat l;

            for (size_t c = 0; c < C; c++) {
                for (size_t r = 0; r < R; r++) {
                    l[c][r] = (r == c) ? T(1) : T(0);
                }
            }

            return l;
        }();

        return ret;
    }
};

// Typedefs ====================================================================

using mat2 = mat<float, 2, 2>;
using mat3 = mat<float, 3, 3>;
using mat4 = mat<float, 4, 4>;

// Accessors ===================================================================

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


// loop free versions for the most common cases
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

template <class T, size_t C, size_t R>
bool is_equal(mat<T, C, R> const& a, mat<T, C, R> const& b) {
    return is_all(a == b);
}

template <class T, size_t C, size_t R>
bool is_equal(mat<T, C, R> const& a, mat<T, C, R> const& b, T limit) {
    static_assert(std::is_floating_point_v<T>);

    auto         delta = abs(a - b);
    mat<T, C, R> c(limit);

    return is_all(delta < c);
}

} // namespace dct

#endif // LINALG_MATRIX_H
