#ifndef LINALG_TMATRIX_H
#define LINALG_TMATRIX_H

#include "mat.h"
#include "mat_operations.h"
#include "mat_transform.h"
#include "quat.h"

namespace dct {

// A transform matrix


class TMatrix {
    Mat4 m_mat = Mat4::identity();

public:
    TMatrix() = default;
    explicit TMatrix(Mat3 const& f) : TMatrix(Mat4(f)) {}
    explicit TMatrix(Mat4 const& f) : m_mat(f) {}
    explicit TMatrix(Quat const&);
    explicit TMatrix(std::array<float, 16> const& f) : m_mat(f) {}

public: // setters
    void translate(Vec3 const&);
    void translate(float x, float y, float z);
    void clear_translate();

    void rotate(float radians, float x, float y, float z);
    void rotate(Quat const&);

    void scale(float x, float y, float z);
    void scale(Vec3 const&);
    void scale(float);

    // note that underlying storage is column major
    void set_column(size_t col, Vec3 const&);
    void set_column(size_t col, Vec4 const&);

    Vec4 column(size_t col) const;

    void set_row(size_t row, Vec3 const&);
    void set_row(size_t row, Vec4 const&);

    Vec4 row(size_t row) const;


    TMatrix inverted() const;
    void    invert();

    TMatrix transposed() const;
    void    transpose();

public:
    void set_perspective_matrix_lh(float fovy,
                                   float aspect,
                                   float zNear,
                                   float zFar);

    void set_frustum_matrix_lh(float left,
                               float right,
                               float bottom,
                               float top,
                               float near,
                               float far);

    void set_ortho_matrix_lh(float left,
                             float right,
                             float bottom,
                             float top,
                             float zNear,
                             float zFar);

    void set_look_at_lh(Vec3 const& eye, Vec3 const& center, Vec3 const& up);

public: // operation
    Vec3 operator*(Vec3 const&)const;
    Vec4 operator*(Vec4 const&)const;

    TMatrix operator*(TMatrix const&)const;

    Vec3 rotate_scale_only(Vec3 const&) const;

    TMatrix get_rotate_scale_only() const;


public: // access
    explicit operator Mat3() const;
    explicit operator Mat4() const;
    explicit operator std::array<float, 16>() const;

    float*       data();
    float const* data() const;
};

// Implementation ==============================================================


inline void TMatrix::translate(Vec3 const& v) {
    dct::translate_in_place(m_mat, v);
}

inline void TMatrix::translate(float x, float y, float z) {
    dct::translate_in_place(m_mat, Vec3(x, y, z));
}

inline void TMatrix::clear_translate() {
    auto const v = m_mat[3];
    m_mat[3]     = Vec4(0, 0, 0, v.w);
}

inline void TMatrix::rotate(float radians, float x, float y, float z) {
    Vec3 v(x, y, z);
    m_mat = dct::rotate(m_mat, radians, v);
}

inline void TMatrix::rotate(Quat const& q) {
    m_mat *= mat4_from_unit_quaternion(normalize(q));
}

inline void TMatrix::scale(float x, float y, float z) {
    dct::scale_in_place(m_mat, { x, y, z });
}

inline void TMatrix::scale(Vec3 const& v) { dct::scale_in_place(m_mat, v); }

inline void TMatrix::scale(float f) { dct::scale_in_place(m_mat, { f, f, f }); }

inline void TMatrix::set_column(size_t col, Vec3 const& v) {
    m_mat[col] = Vec4(v, 0);
}

inline void TMatrix::set_column(size_t col, Vec4 const& v) {
    m_mat[col] = Vec4(v);
}


inline Vec4 TMatrix::column(size_t col) const { return m_mat[col]; }

inline void TMatrix::set_row(size_t row, Vec3 const& v) {
    set_row(row, Vec4(v, 0));
}

inline void TMatrix::set_row(size_t row, Vec4 const& v) {
    m_mat[0][row] = v.x;
    m_mat[1][row] = v.y;
    m_mat[2][row] = v.z;
    m_mat[3][row] = v.w;
}

inline Vec4 TMatrix::row(size_t row) const {
    return Vec4(m_mat[0][row], m_mat[1][row], m_mat[2][row], m_mat[3][row]);
}

inline TMatrix TMatrix::inverted() const {
    return TMatrix(dct::inverse(m_mat));
}

inline void TMatrix::invert() { m_mat = inverse(m_mat); }

inline TMatrix TMatrix::transposed() const {
    return TMatrix(dct::transpose(m_mat));
}

inline void TMatrix::transpose() { m_mat = dct::transpose(m_mat); }


inline Vec3 TMatrix::operator*(Vec3 const& v) const {
    auto const r = operator*(Vec4(v, 1));
    return Vec3(r) / r.w;
}

inline Vec4 TMatrix::operator*(Vec4 const& v) const { return m_mat * v; }

inline TMatrix TMatrix::operator*(TMatrix const& m) const {
    return TMatrix(m_mat * m.m_mat);
}

inline Vec3 TMatrix::rotate_scale_only(Vec3 const& v) const {
    Mat3 const lm = Mat3(m_mat);
    return lm * v;
}

inline TMatrix::operator Mat3() const { return Mat3(m_mat); }

inline TMatrix::operator Mat4() const { return m_mat; }

inline TMatrix::operator std::array<float, 16>() const {
    static_assert(sizeof(Mat4) == sizeof(std::array<float, 16>), "");
    return *reinterpret_cast<std::array<float, 16> const*>(&m_mat);
}

inline float* TMatrix::data() { return m_mat.data(); }

inline float const* TMatrix::data() const { return m_mat.data(); }

} // namespace dct

#endif // LINALG_TMATRIX_H
