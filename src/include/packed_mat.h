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
    using ColumnType  = PackedVector<T, R>;
    using RowType     = PackedVector<T, C>;
    using StorageType = std::array<ColumnType, C>;
    StorageType storage;

public: // Basics
    constexpr size_t size() { return C * R; }
    constexpr size_t row_count() { return R; }
    constexpr size_t column_count() { return C; }

    // column major
    constexpr ColumnType&       operator[](size_t i) { return storage[i]; }
    constexpr ColumnType const& operator[](size_t i) const {
        return storage[i];
    }

public:
    /// \brief Initialize all cells to zero
    constexpr packed_mat() : storage({}) {}

    constexpr packed_mat(packed_mat const&) = default;

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
    /// \brief Copy values from a differently sized matrix.
    template <size_t C2, size_t R2>
    void assign(mat<T, C2, R2> const& other) {
        constexpr auto cbound = std::min(C2, C);
        constexpr auto rbound = std::min(R2, R);
        for (size_t c = 0; c < cbound; c++) {
            for (size_t r = 0; r < rbound; r++) {
                (*this)[c][r] = other[c][r];
            }
        }
    }
};

} // namespace dct

#endif // LINALG_PACKED_MAT_H
