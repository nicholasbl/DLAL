#ifndef LINALG_TMATRIX_H
#define LINALG_TMATRIX_H

#include "mat.h"
#include "mat_operations.h"
#include "mat_transforms.h"
#include "quat.h"

#include <cassert>

namespace dct {


///
/// \brief The TMatrix class helps with transformations. It is the same size of
/// a Mat4, and also stores entries in a column-major form.
///
/// Note that everything is multiplied such that operations are done in reverse.
/// Thus if you want to create a transformation that first scales, rotates, then
/// translates, your code should add the translation, rotation, and finally
/// scale.
///
class TMatrix {
    mat4 m_mat = mat4(identity);

public:
    /// \brief Create a default transformation matrix (the identity)
    TMatrix() = default;
    explicit TMatrix(mat3 const& f) : m_mat(f) { m_mat[3][3] = 1; }
    explicit TMatrix(mat4 const& f) : m_mat(f) {}
    explicit TMatrix(quat const& q) : m_mat(mat4_from_unit_quaternion(q)) {}
    explicit TMatrix(std::array<float, 16> const& f) : m_mat(f) {}

public: // setters
    /// \brief Add a translation by x,y,z
    void translate(float x, float y, float z);
    /// \brief Add a translation by a vector
    void translate(vec3 const&);
    /// \brief Clear the translation component
    void clear_translate();

    /// \brief Add a rotation by radians along the provided 3d axis
    void rotate(float radians, float x, float y, float z);
    /// \brief Add a rotation by a quaternion
    void rotate(quat const&);

    /// \brief Add a scale by factors in x,y,z
    void scale(float x, float y, float z);
    /// \brief Add a scale by factors in vector form
    void scale(vec3 const&);
    /// \brief Add an isotropic scale
    void scale(float);

    /// \brief Clear the rotation and scale portion of the transform
    void clear_rotation_scale();

    /// \brief Set a column to a vec3. The vector is zero-extended
    void set_column(size_t col, vec3 const&);
    /// \brief Set a column to a vec4.
    void set_column(size_t col, vec4 const&);

    /// \brief Get a column
    vec4 column(size_t col) const;

    /// \brief Set a row to a vec3. The vector is zero-extended
    void set_row(size_t row, vec3 const&);
    /// \brief Set a row to a vec4.
    void set_row(size_t row, vec4 const&);

    /// \brief Get a row
    vec4 row(size_t row) const;


    /// \brief Obtain an inverse transform
    TMatrix inverted() const;
    /// \brief Invert this transform in-place
    void invert();

    /// \brief Obtain an transposed transform
    TMatrix transposed() const;
    /// \brief Transpose this transform in-place
    void transpose();

public: // operation
    /// \brief Transform a vec3. The transform occurs using homogenous
    /// coordinates
    vec3 operator*(vec3 const&)const;
    /// \brief Transform a vec4.
    vec4 operator*(vec4 const&)const;

    TMatrix operator*(TMatrix const&)const;

    /// \brief Transform a vector, without translation
    vec3 rotate_scale_only(vec3 const&) const;

    /// \brief Obtain a transformation matrix with only the rotation and scale
    /// components of this transform.
    TMatrix get_rotate_scale_only() const;


public: // access
    explicit operator mat3() const;
    explicit operator mat4() const;
    explicit operator std::array<float, 16>() const;

    float*       data();
    float const* data() const;
};

///
/// \brief Make a left handed perspective matrix.
///
/// Note that the depth transformation is to 0-1, thus unsuitable for OpenGL
///
/// \param fovy Field of view in y, radians
/// \param aspect Aspect ratio (x / y)
/// \param zNear Distance of near clipping plane. Must be positive and nonzero.
/// \param zFar Distance of far clipping plane. Must be positive.
///
TMatrix
make_perspective_matrix_lh(float fovy, float aspect, float zNear, float zFar);

///
/// \brief Make a left handed frustrum matrix
///
/// Note that the depth transformation is to 0-1, thus unsuitable for OpenGL
///
/// \param left Left clipping plane
/// \param right Right clipping plane
/// \param bottom Bottom clipping plane
/// \param top Top clipping plane
/// \param near Distance of near clipping plane
/// \param far Distance of far clipping plane
///
TMatrix make_frustum_matrix_lh(float left,
                               float right,
                               float bottom,
                               float top,
                               float near,
                               float far);

///
/// \brief Make a left-handed orthogonal perspective matrix
/// Note that the depth transformation is to 0-1, thus unsuitable for OpenGL
///
/// \param left Left clipping plane
/// \param right Right clipping plane
/// \param bottom Bottom clipping plane
/// \param top Top clipping plane
/// \param near Distance of near clipping plane
/// \param far Distance of far clipping plane
///
TMatrix make_ortho_matrix_lh(float left,
                             float right,
                             float bottom,
                             float top,
                             float zNear,
                             float zFar);

///
/// \brief Make a left handed look-at transformation matrix
/// \param eye Coordinate of the eye
/// \param center Coordinate of the view target
/// \param up Vector that defines the 'up' direction. Must be normalized.
///
TMatrix make_look_at_lh(vec3 const& eye, vec3 const& center, vec3 const& up);

// Implementation ==============================================================


inline void TMatrix::translate(vec3 const& v) {
    dct::translate_in_place(m_mat, v);
}

inline void TMatrix::translate(float x, float y, float z) {
    dct::translate_in_place(m_mat, vec3{ x, y, z });
}

inline void TMatrix::clear_translate() {
    auto const v = m_mat[3];
    m_mat[3]     = vec4{ 0, 0, 0, v.w };
}

inline void TMatrix::rotate(float radians, float x, float y, float z) {
    vec3 v{ x, y, z };
    m_mat = dct::rotate(m_mat, radians, v);
}

inline void TMatrix::rotate(quat const& q) {
    m_mat *= mat4_from_unit_quaternion(normalize(q));
}

inline void TMatrix::scale(float x, float y, float z) {
    dct::scale_in_place(m_mat, vec3{ x, y, z });
}

inline void TMatrix::scale(vec3 const& v) { dct::scale_in_place(m_mat, v); }

inline void TMatrix::scale(float f) {
    dct::scale_in_place(m_mat, vec3{ f, f, f });
}

inline void TMatrix::clear_rotation_scale() { m_mat.inset(mat3(identity)); }

inline void TMatrix::set_column(size_t col, vec3 const& v) {
    m_mat[col] = vec4{ v.x, v.y, v.z, 0 };
}

inline void TMatrix::set_column(size_t col, vec4 const& v) {
    m_mat[col] = vec4(v);
}


inline vec4 TMatrix::column(size_t col) const { return m_mat[col]; }

inline void TMatrix::set_row(size_t row, vec3 const& v) {
    set_row(row, vec4{ v.x, v.y, v.z, 0 });
}

inline void TMatrix::set_row(size_t row, vec4 const& v) {
    m_mat[0][row] = v.x;
    m_mat[1][row] = v.y;
    m_mat[2][row] = v.z;
    m_mat[3][row] = v.w;
}

inline vec4 TMatrix::row(size_t row) const {
    return vec4{ m_mat[0][row], m_mat[1][row], m_mat[2][row], m_mat[3][row] };
}

inline TMatrix TMatrix::inverted() const {
    return TMatrix(dct::inverse(m_mat));
}

inline void TMatrix::invert() { m_mat = inverse(m_mat); }

inline TMatrix TMatrix::transposed() const {
    return TMatrix(dct::transpose(m_mat));
}

inline void TMatrix::transpose() { m_mat = dct::transpose(m_mat); }


inline vec3 TMatrix::operator*(vec3 const& v) const {
    auto const r = operator*(vec4{ v.x, v.y, v.z, 1 });
    return vec3{ r.x, r.y, r.z } / r.w;
}

inline vec4 TMatrix::operator*(vec4 const& v) const { return m_mat * v; }

inline TMatrix TMatrix::operator*(TMatrix const& m) const {
    return TMatrix(m_mat * m.m_mat);
}

inline vec3 TMatrix::rotate_scale_only(vec3 const& v) const {
    mat3 const lm = mat3(m_mat);
    return lm * v;
}

inline TMatrix::operator mat3() const { return mat3(m_mat); }

inline TMatrix::operator mat4() const { return m_mat; }

inline TMatrix::operator std::array<float, 16>() const {
    static_assert(sizeof(mat4) == sizeof(std::array<float, 16>), "");
    return *reinterpret_cast<std::array<float, 16> const*>(&m_mat);
}

inline float* TMatrix::data() { return dct::data(m_mat); }

inline float const* TMatrix::data() const { return dct::data(m_mat); }

//

inline TMatrix
make_perspective_matrix_lh(float fovy, float aspect, float zNear, float zFar) {
    assert(aspect > 0.0f);
    assert((zFar - zNear) > 0.0f);
    assert(zNear > 0.0f);
    assert(fovy > 0.0f);

    float const yscale = 1.0f / std::tan(fovy / 2.0f);
    float const xscale = yscale / aspect;

    auto mat  = mat4(0);
    mat[0][0] = xscale;
    mat[1][1] = -yscale;
    mat[2][3] = 1.0f;

    mat[2][2] = zFar / (zFar - zNear);
    mat[3][2] = -(zFar * zNear) / (zFar - zNear);

    return TMatrix(mat);
}

inline TMatrix make_frustum_matrix_lh(float left,
                                      float right,
                                      float bottom,
                                      float top,
                                      float near,
                                      float far) {
    auto mat  = mat4(0);
    mat[0][0] = 2.0f * near / (right - left);
    mat[1][1] = 2.0f * near / (top - bottom);
    mat[2][0] = (right + left) / (right - left);
    mat[2][1] = (top + bottom) / (top - bottom);
    mat[2][2] = far / (far - near);
    mat[2][3] = 1;
    mat[3][2] = -(far * near) / (far - near);
    return TMatrix(mat);
}

inline TMatrix make_ortho_matrix_lh(float left,
                                    float right,
                                    float bottom,
                                    float top,
                                    float zNear,
                                    float zFar) {
    auto mat  = mat4(0);
    mat[0][0] = 2.0f / (right - left);
    mat[1][1] = 2.0f / (top - bottom);
    mat[2][2] = 1.0f / (zFar - zNear);
    mat[3][3] = 1;
    mat[3][0] = -(right + left) / (right - left);
    mat[3][1] = -(top + bottom) / (top - bottom);
    mat[3][2] = -(zNear) / (zFar - zNear);
    return TMatrix(mat);
}

inline TMatrix
make_look_at_lh(vec3 const& eye, vec3 const& center, vec3 const& up) {
    vec3 const f(normalize(center - eye));
    vec3 const s(normalize(cross(up, f)));
    vec3 const u(cross(f, s));

    auto mat  = mat4(identity);
    mat[0][0] = s.x;
    mat[1][0] = s.y;
    mat[2][0] = s.z;
    mat[0][1] = u.x;
    mat[1][1] = u.y;
    mat[2][1] = u.z;
    mat[0][2] = f.x;
    mat[1][2] = f.y;
    mat[2][2] = f.z;
    mat[3][0] = -dot(s, eye);
    mat[3][1] = -dot(u, eye);
    mat[3][2] = -dot(f, eye);
    return TMatrix(mat);
}

} // namespace dct

#endif // LINALG_TMATRIX_H
