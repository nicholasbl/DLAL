#ifndef LINALG_PACKED_MAT_H
#define LINALG_PACKED_MAT_H

#include "mat.h"

namespace dct {

///
/// \brief The PackedMatrix class defines a column-major, size restricted
/// matrix. Contents are packed to be sizeof(T) * C * R.
///
template <class T, size_t C, size_t R>
struct packed_mat {
    using ColumnType  = packed_vector<T, R>;
    using RowType     = packed_vector<T, C>;
    using StorageType = std::array<ColumnType, C>;
    StorageType storage;

    static_assert(sizeof(storage) == (sizeof(T) * C * R));

public: // Basics
    ///
    /// \brief Get the total number of cells
    ///
    constexpr size_t size() { return C * R; }
    constexpr size_t row_count() { return R; }
    constexpr size_t column_count() { return C; }

    /// @{
    /// \brief Access a column.
    constexpr ColumnType&       operator[](size_t i) { return storage[i]; }
    constexpr ColumnType const& operator[](size_t i) const {
        return storage[i];
    }
    /// @}

public:
    /// \brief Initialize all cells to zero
    constexpr packed_mat() : storage({}) {}

    constexpr packed_mat(packed_mat const&) = default;

    /// \brief Convert from a non-packed matrix
    constexpr packed_mat(mat<T, C, R> const& other) {
        if constexpr (mat<T, C, R>::is_contiguous) {

        } else {
            for (size_t c = 0; c < C; c++) {
                for (size_t r = 0; r < R; r++) {
                    (*this)[c][r] = other[c][r];
                }
            }
        }
    }

    constexpr packed_mat(std::array<float, C * R> const& a) : storage(a) {}

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
    packed_mat& operator=(packed_mat const& m) = default;

public:
    T*       data() { return storage.data(); }
    T const* data() const { return storage.data(); }
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

} // namespace dct

#endif // LINALG_PACKED_MAT_H
