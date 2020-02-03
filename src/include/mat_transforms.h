#ifndef LINALG_MAT_TRANSFORM_H
#define LINALG_MAT_TRANSFORM_H

#include "mat.h"

#include <cmath>

namespace dct {

///
/// \brief Add a translation to a matrix, in place
///
template <class T>
mat<T, 4, 4> translate(mat<T, 4, 4> const& m, vec<T, 3> const& v) {
    mat<T, 4, 4> ret(m);
    ret[3] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3];
    return ret;
}

///
/// \brief Add a translation to a matrix, in place
///
template <class T>
void translate_in_place(mat<T, 4, 4>& m, vec<T, 3> const& v) {
    m[3] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3];
}


///
/// \brief Add a rotation to a given matrix by an axis and an angle
/// \param m       Matrix to operate on
/// \param radians Rotation angle, in radians
/// \param axis    Rotate around this (possibly not-normalized) axis
///
template <class T>
mat<T, 4, 4> rotate(mat<T, 4, 4> const& m, T radians, vec<T, 3> const& axis) {
    T const a = radians;
    T const c = std::cos(a);
    T const s = std::sin(a);

    vec<T, 3> const naxis(normalize(axis));
    vec<T, 3> const cos_pack((T(1) - c) * naxis);
    vec<T, 3> const sin_pack = naxis * s;

    auto rotation_part  = mat<T, 4, 4>(identity);
    rotation_part[0][0] = c + cos_pack.x * naxis.x;
    rotation_part[0][1] = cos_pack.x * naxis.y + sin_pack.z;
    rotation_part[0][2] = cos_pack.x * naxis.z - sin_pack.y;

    rotation_part[1][0] = cos_pack.y * naxis.x - sin_pack.z;
    rotation_part[1][1] = c + cos_pack.y * naxis.y;
    rotation_part[1][2] = cos_pack.y * naxis.z + sin_pack.x;

    rotation_part[2][0] = cos_pack.z * naxis.x + sin_pack.y;
    rotation_part[2][1] = cos_pack.z * naxis.y - sin_pack.x;
    rotation_part[2][2] = c + cos_pack.z * naxis.z;

    mat<T, 4, 4> ret;
    ret[0] = m[0] * rotation_part[0][0] + m[1] * rotation_part[0][1] +
             m[2] * rotation_part[0][2];
    ret[1] = m[0] * rotation_part[1][0] + m[1] * rotation_part[1][1] +
             m[2] * rotation_part[1][2];
    ret[2] = m[0] * rotation_part[2][0] + m[1] * rotation_part[2][1] +
             m[2] * rotation_part[2][2];
    ret[3] = m[3];
    return ret;
}


///
/// \brief Add a scale to a given matrix
///
template <class T>
mat<T, 4, 4> scale(mat<T, 4, 4> const& m, vec<T, 3> const& v) {
    mat<T, 4, 4> ret;
    ret[0] = m[0] * v[0];
    ret[1] = m[1] * v[1];
    ret[2] = m[2] * v[2];
    ret[3] = m[3];
    return ret;
}

///
/// \brief Add a scale, in place, to a given matrix
///
template <class T>
void scale_in_place(mat<T, 4, 4>& m, vec<T, 3> const& v) {
    m[0] = m[0] * v[0];
    m[1] = m[1] * v[1];
    m[2] = m[2] * v[2];
    m[3] = m[3];
}

} // namespace dct

#endif // LINALG_MAT_TRANSFORM_H
