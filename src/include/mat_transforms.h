#ifndef LINALG_MAT_TRANSFORM_H
#define LINALG_MAT_TRANSFORM_H

#include "mat.h"

#include <cmath>

namespace dct {

template <class T>
Matrix<T, 4, 4> translate(Matrix<T, 4, 4> const& m, Vector<T, 3> const& v) {
    Matrix<T, 4, 4> ret(m);
    ret[3] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3];
    return ret;
}

template <class T>
void translate_in_place(Matrix<T, 4, 4>& m, Vector<T, 3> const& v) {
    m[3] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3];
}


///
/// \brief Rotate a given vector by an axis and an angle
/// \param axis Rotate around this (possibly not-normalized) axis
///
template <class T>
Matrix<T, 4, 4>
rotate(Matrix<T, 4, 4> const& m, T angle, Vector<T, 3> const& axis) {
    T const a = angle;
    T const c = std::cos(a);
    T const s = std::sin(a);

    Vector<T, 3> const naxis(normalize(axis));
    Vector<T, 3> const cos_pack((T(1) - c) * naxis);
    Vector<T, 3> const sin_pack = naxis * s;

    auto rotation_part  = Matrix<T, 4, 4>::identity();
    rotation_part[0][0] = c + cos_pack.x * naxis.x;
    rotation_part[0][1] = cos_pack.x * naxis.y + sin_pack.z;
    rotation_part[0][2] = cos_pack.x * naxis.z - sin_pack.y;

    rotation_part[1][0] = cos_pack.y * naxis.x - sin_pack.z;
    rotation_part[1][1] = c + cos_pack.y * naxis.y;
    rotation_part[1][2] = cos_pack.y * naxis.z + sin_pack.x;

    rotation_part[2][0] = cos_pack.z * naxis.x + sin_pack.y;
    rotation_part[2][1] = cos_pack.z * naxis.y - sin_pack.x;
    rotation_part[2][2] = c + cos_pack.z * naxis.z;

    Matrix<T, 4, 4> ret;
    ret[0] = m[0] * rotation_part[0][0] + m[1] * rotation_part[0][1] +
             m[2] * rotation_part[0][2];
    ret[1] = m[0] * rotation_part[1][0] + m[1] * rotation_part[1][1] +
             m[2] * rotation_part[1][2];
    ret[2] = m[0] * rotation_part[2][0] + m[1] * rotation_part[2][1] +
             m[2] * rotation_part[2][2];
    ret[3] = m[3];
    return ret;
}

template <class T>
Matrix<T, 4, 4> scale(Matrix<T, 4, 4> const& m, Vector<T, 3> const& v) {
    Matrix<T, 4, 4> ret;
    ret[0] = m[0] * v[0];
    ret[1] = m[1] * v[1];
    ret[2] = m[2] * v[2];
    ret[3] = m[3];
    return ret;
}

template <class T>
void scale_in_place(Matrix<T, 4, 4>& m, Vector<T, 3> const& v) {
    m[0] = m[0] * v[0];
    m[1] = m[1] * v[1];
    m[2] = m[2] * v[2];
    m[3] = m[3];
}

} // namespace dct

#endif // LINALG_MAT_TRANSFORM_H
