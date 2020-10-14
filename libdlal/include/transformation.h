#ifndef LINALG_TMATRIX_H
#define LINALG_TMATRIX_H

#include "mat.h"
#include "mat_operations.h"
#include "mat_transforms.h"
#include "quat.h"

#include <cassert>

namespace dlal {


///
/// \brief The TMatrix class helps with transformations. It is the same size of
/// a Mat4, and also stores entries in a column-major form.
///
/// Note that everything is multiplied such that operations are done in reverse.
/// Thus if you want to create a transformation that first scales, rotates, then
/// translates, your code should add the translation, rotation, and finally
/// scale.
///
class Transformation {
    mat4 m_mat = mat4(identity);

public:
    /// \brief Create a default transformation matrix (the identity)
    Transformation() = default;

    /// \brief Create a matrix with the given rotation and scaling matrix
    explicit Transformation(mat3 const& f) : m_mat(f) { m_mat[3][3] = 1; }

    /// \brief Create a transformation matrix using a provided 4x4 mat.
    explicit Transformation(mat4 const& f) : m_mat(f) { }

    /// \brief Create a matrix with a rotation.
    explicit Transformation(quat const& q)
        : m_mat(mat4_from_unit_quaternion(q)) { }

    /// \brief Create a matrix with an array of values
    explicit Transformation(std::array<float, 16> const& f) : m_mat(f) { }

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
    Transformation inverted() const;
    /// \brief Invert this transform in-place
    void invert();

    /// \brief Obtain an transposed transform
    Transformation transposed() const;
    /// \brief Transpose this transform in-place
    void transpose();

public: // operation
    /// \brief Transform a vec3. The transform occurs using homogenous
    /// coordinates
    vec3 operator*(vec3 const&) const;
    /// \brief Transform a vec4.
    vec4 operator*(vec4 const&) const;

    /// \brief Add the transformations from another TMatrix
    Transformation operator*(Transformation const&) const;

    /// \brief Transform a vector, without translation
    vec3 rotate_scale_only(vec3 const&) const;

    /// \brief Obtain a transformation matrix with only the rotation and scale
    /// components of this transform.
    Transformation get_rotate_scale_only() const;


public: // access
    /// Convert to a rotation and scale only matrix
    explicit operator mat3() const;

    /// Convert to a mat4 (no-op)
    explicit operator mat4() const;

    /// Convert to a contiguous array
    explicit operator std::array<float, 16>() const;

    /// @{
    /// \brief Access the underlying storage as a contiguous array.
    float*       data();
    float const* data() const;
    /// @}
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
Transformation
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
Transformation make_frustum_matrix_lh(float left,
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
Transformation make_ortho_matrix_lh(float left,
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
Transformation
make_look_at_lh(vec3 const& eye, vec3 const& center, vec3 const& up);

// Implementation ==============================================================


inline void Transformation::translate(vec3 const& v) {
    dlal::translate_in_place(m_mat, v);
}

inline void Transformation::translate(float x, float y, float z) {
    dlal::translate_in_place(m_mat, vec3 { x, y, z });
}

inline void Transformation::clear_translate() {
    auto const v = m_mat[3];
    m_mat[3]     = vec4 { 0, 0, 0, v.w };
}

inline void Transformation::rotate(float radians, float x, float y, float z) {
    vec3 v { x, y, z };
    m_mat = dlal::rotate(m_mat, radians, v);
}

inline void Transformation::rotate(quat const& q) {
    m_mat *= mat4_from_unit_quaternion(normalize(q));
}

inline void Transformation::scale(float x, float y, float z) {
    dlal::scale_in_place(m_mat, vec3 { x, y, z });
}

inline void Transformation::scale(vec3 const& v) {
    dlal::scale_in_place(m_mat, v);
}

inline void Transformation::scale(float f) {
    dlal::scale_in_place(m_mat, vec3 { f, f, f });
}

inline void Transformation::clear_rotation_scale() {
    m_mat.inset(mat3(identity));
}

inline void Transformation::set_column(size_t col, vec3 const& v) {
    m_mat[col] = vec4 { v.x, v.y, v.z, 0 };
}

inline void Transformation::set_column(size_t col, vec4 const& v) {
    m_mat[col] = vec4(v);
}


inline vec4 Transformation::column(size_t col) const {
    return m_mat[col];
}

inline void Transformation::set_row(size_t row, vec3 const& v) {
    set_row(row, vec4 { v.x, v.y, v.z, 0 });
}

inline void Transformation::set_row(size_t row, vec4 const& v) {
    m_mat[0][row] = v.x;
    m_mat[1][row] = v.y;
    m_mat[2][row] = v.z;
    m_mat[3][row] = v.w;
}

inline vec4 Transformation::row(size_t row) const {
    return vec4 { m_mat[0][row], m_mat[1][row], m_mat[2][row], m_mat[3][row] };
}

inline Transformation Transformation::inverted() const {
    return Transformation(dlal::inverse(m_mat));
}

inline void Transformation::invert() {
    m_mat = inverse(m_mat);
}

inline Transformation Transformation::transposed() const {
    return Transformation(dlal::transpose(m_mat));
}

inline void Transformation::transpose() {
    m_mat = dlal::transpose(m_mat);
}


inline vec3 Transformation::operator*(vec3 const& v) const {
    auto const r = operator*(vec4 { v.x, v.y, v.z, 1 });
    return vec3 { r.x, r.y, r.z } / r.w;
}

inline vec4 Transformation::operator*(vec4 const& v) const {
    return m_mat * v;
}

inline Transformation Transformation::operator*(Transformation const& m) const {
    return Transformation(m_mat * m.m_mat);
}

inline vec3 Transformation::rotate_scale_only(vec3 const& v) const {
    mat3 const lm = mat3(m_mat);
    return lm * v;
}

inline Transformation::operator mat3() const {
    return mat3(m_mat);
}

inline Transformation::operator mat4() const {
    return m_mat;
}

inline Transformation::operator std::array<float, 16>() const {
    static_assert(sizeof(mat4) == sizeof(std::array<float, 16>), "");
    return *reinterpret_cast<std::array<float, 16> const*>(&m_mat);
}

inline float* Transformation::data() {
    return dlal::data(m_mat);
}

inline float const* Transformation::data() const {
    return dlal::data(m_mat);
}

//

inline Transformation
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

    return Transformation(mat);
}

inline Transformation make_frustum_matrix_lh(float left,
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
    return Transformation(mat);
}

inline Transformation make_ortho_matrix_lh(float left,
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
    return Transformation(mat);
}

inline Transformation
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
    return Transformation(mat);
}

} // namespace dlal

#endif // LINALG_TMATRIX_H
