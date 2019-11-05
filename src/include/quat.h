#ifndef LINALG_QUAT_H
#define LINALG_QUAT_H

#include "mat.h"
#include "vec.h"

#include "vec_trig.h"

namespace dct {

template <class T>
struct QuaternionCore {
    union {
        Vector<T, 4> storage;
        struct {
            T x;
            T y;
            T z;
            T w;
        };
    };

public:
    QuaternionCore() : storage(0, 0, 0, 1) {}
    QuaternionCore(T w, Vector<T, 3> const& v) : storage(v, w) {}
    explicit QuaternionCore(Vector<T, 4> const& f) : storage(f) {}


public:
    explicit operator Vector<T, 4>() const { return storage; }
};

using Quat  = QuaternionCore<float>;
using DQuat = QuaternionCore<double>;


// Operators ===================================================================

template <class T>
QuaternionCore<T> operator+(QuaternionCore<T> const& q,
                            QuaternionCore<T> const& r) {
    return QuaternionCore<T>(q.storage + r.storage);
}

template <class T>
QuaternionCore<T> operator-(QuaternionCore<T> const& q,
                            QuaternionCore<T> const& r) {
    return QuaternionCore<T>(q.storage - r.storage);
}

template <class T>
QuaternionCore<T> operator*(QuaternionCore<T> const& q, T scalar) {
    return QuaternionCore<T>(q.storage * scalar);
}

// note that rotating a non-unit quaternion can do odd things

template <class T>
QuaternionCore<T> operator*(QuaternionCore<T> const& q,
                            QuaternionCore<T> const& r) {

    QuaternionCore<T> ret;

    ret.x = q.w * r.x + q.x * r.w + q.y * r.z - q.z * r.y;
    ret.y = q.w * r.y + q.y * r.w + q.z * r.x - q.x * r.z;
    ret.z = q.w * r.z + q.z * r.w + q.x * r.y - q.y * r.x;
    ret.w = q.w * r.w - q.x * r.x - q.y * r.y - q.z * r.z;

    return ret;
}

template <class T>
Vector<T, 3> operator*(QuaternionCore<T> const& q, Vector<T, 3> const& v) {
    Vector<T, 3> const lqv(q.x, q.y, q.z);
    Vector<T, 3> const uv(cross(lqv, v));
    Vector<T, 3> const uuv(cross(lqv, uv));

    return v + ((uv * q.w) + uuv) * static_cast<T>(2);
}


// Operations ==================================================================
template <class T>
T length(QuaternionCore<T> const& q) {
    return length(Vector<T, 4>(q));
}

template <class T>
QuaternionCore<T> normalize(QuaternionCore<T> const& q) {
    return QuaternionCore<T>(normalize(q.storage));
}

template <class T>
QuaternionCore<T> conjugate(QuaternionCore<T> const& q) {
    return QuaternionCore<T>(q.w, -Vector<T, 3>(q.storage));
}

template <class T>
QuaternionCore<T> inverse(QuaternionCore<T> const& q) {
    return conjugate(q) / dot(q.storage, q.storage);
}

// Conversion ==================================================================


/// \brief Convert a UNIT quaternion to a mat3
template <class T>
MatrixCore<T, 3, 3> mat3_from_unit_quaternion(QuaternionCore<T> const& q) {
    T const qxx(q.x * q.x);
    T const qyy(q.y * q.y);
    T const qzz(q.z * q.z);
    T const qxz(q.x * q.z);
    T const qxy(q.x * q.y);
    T const qyz(q.y * q.z);
    T const qwx(q.w * q.x);
    T const qwy(q.w * q.y);
    T const qwz(q.w * q.z);

    T const one(1);
    T const two(2);

    MatrixCore<T, 3, 3> ret;
    ret[0][0] = one - two * (qyy + qzz);
    ret[0][1] = two * (qxy + qwz);
    ret[0][2] = two * (qxz - qwy);

    ret[1][0] = two * (qxy - qwz);
    ret[1][1] = one - two * (qxx + qzz);
    ret[1][2] = two * (qyz + qwx);

    ret[2][0] = two * (qxz + qwy);
    ret[2][1] = two * (qyz - qwx);
    ret[2][2] = one - two * (qxx + qyy);

    return ret;
}

template <class T>
MatrixCore<T, 4, 4> mat4_from_unit_quaternion(QuaternionCore<T> const& q) {

    T const qxx(q.x * q.x);
    T const qyy(q.y * q.y);
    T const qzz(q.z * q.z);
    T const qxz(q.x * q.z);
    T const qxy(q.x * q.y);
    T const qyz(q.y * q.z);
    T const qwx(q.w * q.x);
    T const qwy(q.w * q.y);
    T const qwz(q.w * q.z);

    T const one(1);
    T const two(2);

    MatrixCore<T, 4, 4> ret(0);
    ret[0][0] = one - two * (qyy + qzz);
    ret[0][1] = two * (qxy + qwz);
    ret[0][2] = two * (qxz - qwy);

    ret[1][0] = two * (qxy - qwz);
    ret[1][1] = one - two * (qxx + qzz);
    ret[1][2] = two * (qyz + qwx);

    ret[2][0] = two * (qxz + qwy);
    ret[2][1] = two * (qyz - qwx);
    ret[2][2] = one - two * (qxx + qyy);

    ret[3][3] = one;
    return ret;
}

template <class T>
QuaternionCore<T> quaternion_from_matrix(MatrixCore<T, 3, 3> const& m) {
    QuaternionCore<T> q;

    float const trace = m[0][0] + m[1][1] + m[2][2];
    // printf("TRACE %f : %f %f %f\n", trace, m[0][0], m[1][1], m[2][2]);
    if (trace > 0) {
        float const s = 0.5f / sqrtf(trace + 1.0f);
        q.x           = (m[1][2] - m[2][1]) * s;
        q.y           = (m[2][0] - m[0][2]) * s;
        q.z           = (m[0][1] - m[1][0]) * s;
        q.w           = 0.25f / s;
    } else {
        if (m[0][0] > m[1][1] and m[0][0] > m[2][2]) {
            // printf("SW 1\n");
            float const s = 2.0f * sqrtf(1.0f + m[0][0] - m[1][1] - m[2][2]);
            q.x           = 0.25f * s;
            q.y           = (m[1][0] + m[0][1]) / s;
            q.z           = (m[2][0] + m[0][2]) / s;
            q.w           = (m[1][2] - m[2][1]) / s;
        } else if (m[1][1] > m[2][2]) {
            // printf("SW 2\n");
            float const s = 2.0f * sqrtf(1.0f + m[1][1] - m[0][0] - m[2][2]);
            q.x           = (m[1][0] + m[0][1]) / s;
            q.y           = 0.25f * s;
            q.z           = (m[2][1] + m[1][2]) / s;
            q.w           = (m[2][0] - m[0][2]) / s;
        } else {
            // printf("SW 3\n");
            float const s = 2.0f * sqrtf(1.0f + m[2][2] - m[0][0] - m[1][1]);
            q.x           = 0.25f * s;
            q.y           = (m[0][1] - m[1][0]) / s;
            q.z           = (m[2][0] + m[0][2]) / s;
            q.w           = (m[2][1] + m[1][2]) / s;
        }
    }

    // printf("L %f\n", length(q));

    return q;
}

template <class T>
QuaternionCore<T> quaternion_from_matrix(MatrixCore<T, 4, 4> const& m) {
    return quaternion_from_matrix(MatrixCore<T, 3, 3>(m));
}

// Other =======================================================================

///
/// \brief Compute a rotation between two vectors
/// \param from A normalized source vector
/// \param to A normalized destination vector
///
template <class T>
QuaternionCore<T> rotation_from_to(Vector<T, 3> const& from,
                                   Vector<T, 3> const& to) {
    Vector<T, 3> const w = cross(from, to);

    Vector<T, 4> lq(w.x, w.y, w.z, dot(from, to));

    lq.w += dot(lq, lq);
    return normalize(QuaternionCore<T>(lq));
}


///
/// \brief Compute a rotation given a direction and an 'up' vector
/// \param norm_direction The direction to look in, must be normalized
/// \param norm_up The 'up' direction, must be normalized
///
template <class T>
QuaternionCore<T> look_at(Vector<T, 3> const& norm_direction,
                          Vector<T, 3> const& norm_up) {
    if (std::abs(dot(norm_direction, norm_up)) >= 1) {
        return rotation_from_to({ 0, 0, -1 }, norm_direction);
    }

    MatrixCore<T, 3, 3> ret;
    ret[0] = normalize(cross(norm_up, norm_direction));
    ret[1] = cross(norm_direction, ret[0]);
    ret[2] = norm_direction;

    return QuaternionCore<T>(quaternion_from_matrix(ret));
}

///
/// \brief Compute a quaternion from Euler angles, expressed in radians
///
template <class T>
QuaternionCore<T> from_angles(Vector<T, 3> angles) {
    Vector<T, 3> const c = cos(angles * T(0.5));
    Vector<T, 3> const s = sin(angles * T(0.5));

    QuaternionCore<T> ret;
    ret.x = s.x * c.y * c.z - c.x * s.y * s.z;
    ret.y = c.x * s.y * c.z + s.x * c.y * s.z;
    ret.z = c.x * c.y * s.z - s.x * s.y * c.z;
    ret.w = c.x * c.y * c.z + s.x * s.y * s.z;
    return ret;
}

///
/// \brief Compute a quaternion from an axis and a rotation, expressed in
/// radians
///
template <class T>
QuaternionCore<T> from_angle_axis(T angle, Vector<T, 3> axis) {
    T const s = std::sin(angle * static_cast<T>(0.5));

    return QuaternionCore<T>(std::cos(angle * static_cast<T>(0.5)), axis * s);
}


} // namespace dct

#endif // QUAT_H
