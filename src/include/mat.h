#ifndef LINALG_MATRIX_H
#define LINALG_MATRIX_H

#include "mat_detail.h"
#include "vec.h"

#include <xmmintrin.h>

namespace dct {

template <class T, size_t C, size_t R>
struct MatrixStorage {
    using ColumnType = Vector<T, R>;
    using RowType    = Vector<T, C>;

    using StorageType = std::array<ColumnType, C>;

    StorageType storage;

    static_assert(sizeof(StorageType) == sizeof(T) * C * R);

    MatrixStorage() : storage() {}
};

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

template <class T, size_t C, size_t R>
struct MatrixCore : public MatrixStorage<T, C, R> {

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
    constexpr MatrixCore() : Parent({}) {}
    constexpr MatrixCore(MatrixCore const&) = default;

    constexpr MatrixCore(std::array<float, C * R> const& a) {
        std::copy(a.data(), a.data() + a.size(), data());
    }

    // Fills all cells with the same value
    constexpr MatrixCore(T value) { Parent::storage.fill(value); }

    constexpr MatrixCore(StorageType pack) : Parent::storage(pack) {}

    template <size_t C2, size_t R2>
    constexpr explicit MatrixCore(MatrixCore<T, C2, R2> const& other) {
        static_assert(R2 >= R and C2 >= C, "Cannot assign smaller to larger!");
        constexpr size_t bound = C2 < C ? C2 : C;
        // no loops for speed in debug mode
        if constexpr (bound == 1) {
            Parent::storage[0] = other.storage[0];
        } else if constexpr (bound == 2) {
            Parent::storage[0] = other.storage[0];
            Parent::storage[1] = other.storage[1];
        } else if constexpr (bound == 3) {
            Parent::storage[0] = other.storage[0];
            Parent::storage[1] = other.storage[1];
            Parent::storage[2] = other.storage[2];
        } else if constexpr (bound == 4) {
            Parent::storage[0] = other.storage[0];
            Parent::storage[1] = other.storage[1];
            Parent::storage[2] = other.storage[2];
            Parent::storage[3] = other.storage[3];
        }
    }

public:
    static MatrixCore const& identity() {
        static constexpr MatrixCore ret = []() {
            MatrixCore l;

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

using Mat2 = MatrixCore<float, 3, 3>;
using Mat3 = MatrixCore<float, 3, 3>;
using Mat4 = MatrixCore<float, 4, 4>;

// Common ======================================================================

inline void set_mat_to_identity(Mat4& mat) {
    mat[0] = { 1, 0, 0, 0 };
    mat[1] = { 0, 1, 0, 0 };
    mat[2] = { 0, 0, 1, 0 };
    mat[3] = { 0, 0, 0, 1 };
}

// Unary =======================================================================
template <class T, size_t C, size_t R>
MatrixCore<T, C, R> operator-(MatrixCore<T, C, R> const& m) {
    MATRIX_UNARY(-)
}

template <class T, size_t C, size_t R>
MatrixCore<T, C, R> operator!(MatrixCore<T, C, R> const& m) {
    MATRIX_UNARY(!)
}

// Operators ===================================================================

// Addition ====================================================================
template <class T, size_t C, size_t R>
auto operator+(MatrixCore<T, C, R> const& m, MatrixCore<T, C, R> const& o) {
    MATRIX_BINARY(+)
}
template <class T, size_t C, size_t R>
auto operator+(MatrixCore<T, C, R> const& m, T scalar) {
    MATRIX_BINARY_SCALAR_R(+)
}
template <class T, size_t C, size_t R>
auto operator+(T scalar, MatrixCore<T, C, R> const& m) {
    MATRIX_BINARY_SCALAR_L(+)
}

template <class T, size_t C, size_t R>
MatrixCore<T, C, R>& operator+=(MatrixCore<T, C, R>&       m,
                                MatrixCore<T, C, R> const& o) {
    MATRIX_IN_PLACE(+=)
}
template <class T, size_t C, size_t R>
MatrixCore<T, C, R>& operator+=(MatrixCore<T, C, R>& m, T scalar) {
    MATRIX_IN_PLACE_SCALAR_R(+=)
}


// Subtraction =================================================================

template <class T, size_t C, size_t R>
auto operator-(MatrixCore<T, C, R> const& m, MatrixCore<T, C, R> const& o) {
    MATRIX_BINARY(-)
}
template <class T, size_t C, size_t R>
auto operator-(MatrixCore<T, C, R> const& m, T scalar) {
    MATRIX_BINARY_SCALAR_R(-)
}
template <class T, size_t C, size_t R>
auto operator-(T scalar, MatrixCore<T, C, R> const& m) {
    MATRIX_BINARY_SCALAR_L(-)
}

template <class T, size_t C, size_t R>
MatrixCore<T, C, R>& operator-=(MatrixCore<T, C, R>&       m,
                                MatrixCore<T, C, R> const& o) {
    MATRIX_IN_PLACE(-=)
}
template <class T, size_t C, size_t R>
MatrixCore<T, C, R>& operator-=(MatrixCore<T, C, R>& m, T scalar) {
    MATRIX_IN_PLACE_SCALAR_R(-=)
}

// Multiply ====================================================================

template <class T, size_t N, size_t R, size_t C>
auto operator*(MatrixCore<T, N, R> const& m, MatrixCore<T, C, N> const& o) {
    static_assert(std::is_same_v<typename MatrixCore<T, N, R>::RowType,
                                 typename MatrixCore<T, C, N>::ColumnType>);
    MatrixCore<T, C, R> ret(0);

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
auto operator*(MatrixCore<T, C, R> const& m, Vector<T, R> const& o) {
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
auto operator*(MatrixCore<T, 3, 3> const& m, Vector<T, 3> const& o) {
    Vector<T, 3> a0 = m[0] * Vector<T, 3>(o[0]);
    Vector<T, 3> a1 = m[1] * Vector<T, 3>(o[1]);

    auto m1 = a0 + a1;

    Vector<T, 3> a2 = m[2] * Vector<T, 3>(o[2]);

    return m1 + a2;
}

template <class T>
auto operator*(MatrixCore<T, 4, 4> const& m, Vector<T, 4> const& o) {
    Vector<T, 4> a0 = m[0] * Vector<T, 4>(o[0]);
    Vector<T, 4> a1 = m[1] * Vector<T, 4>(o[1]);

    auto m1 = a0 + a1;

    Vector<T, 4> a2 = m[2] * Vector<T, 4>(o[2]);
    Vector<T, 4> a3 = m[3] * Vector<T, 4>(o[3]);

    auto m2 = a2 + a3;

    return m1 + m2;
}


template <class T, size_t C, size_t R>
auto operator*(MatrixCore<T, C, R> const& m, T scalar) {
    MATRIX_BINARY_SCALAR_R(*)
}

template <class T, size_t C, size_t R>
auto operator*(T scalar, MatrixCore<T, C, R> const& m) {
    MATRIX_BINARY_SCALAR_L(*)
}


template <class T, size_t N, size_t R, size_t C>
MatrixCore<T, N, R> operator*=(MatrixCore<T, N, R>&       m,
                               MatrixCore<T, C, N> const& o) {
    return m = m * o;
}
template <class T, size_t C, size_t R>
MatrixCore<T, C, R>& operator*=(MatrixCore<T, C, R>& m, T scalar) {
    return m = m * scalar;
}
template <class T, size_t C, size_t R>
MatrixCore<T, C, R>& operator*=(T scalar, MatrixCore<T, C, R>& m) {
    return m = scalar * m;
}

// Division ====================================================================

template <class T, size_t C, size_t R>
auto operator/(MatrixCore<T, C, R> const& m, T scalar) {
    MATRIX_BINARY_SCALAR_R(/)
}

template <class T, size_t C, size_t R>
auto operator/(T scalar, MatrixCore<T, C, R> const& m) {
    MATRIX_BINARY_SCALAR_L(/)
}

// Boolean =====================================================================

template <class T, size_t C, size_t R>
auto operator==(MatrixCore<T, C, R> const& m, MatrixCore<T, C, R> const& o) {
    MATRIX_BINARY_BOOL(==)
}

template <class T, size_t C, size_t R>
auto operator!=(MatrixCore<T, C, R> const& m, MatrixCore<T, C, R> const& o) {
    MATRIX_BINARY_BOOL(!=)
}

template <class T, size_t C, size_t R>
auto operator&&(MatrixCore<bool, C, R> const& m,
                MatrixCore<bool, C, R> const& o) {
    MATRIX_BINARY_BOOL(&&)
}

template <class T, size_t C, size_t R>
auto operator||(MatrixCore<bool, C, R> const& m,
                MatrixCore<bool, C, R> const& o) {
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
bool is_all(MatrixCore<bool, C, R> const& a) {
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
bool is_any(MatrixCore<bool, C, R> const& a) {
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
bool is_equal(MatrixCore<T, C, R> const& a, MatrixCore<T, C, R> const& b) {
    return is_all(a == b);
}

} // namespace dct

#endif // LINALG_MATRIX_H
