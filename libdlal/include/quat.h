#ifndef LINALG_QUAT_H
#define LINALG_QUAT_H

#include "mat.h"
#include "packed_vec.h"

#include "vec_trig.h"

namespace dlal {

///
/// \brief The quaternion class models a rotation
///
template <class T>
struct quaternion {
    union {
        vec<T, 4> storage; ///< Storage is simd vector
        struct {
            T x, y, z, w; ///< Basic swizzle
        };
    };

public:
    /// \brief Initialize quaternion to zero
    quaternion() : storage { 0, 0, 0, 1 } { }

    /// \brief Initialize quaternion from loose values. W is the scalar.
    quaternion(T x, T y, T z, T w) : storage { x, y, z, w } { }

    /// \brief Initialize quaternion from vector and scalar.
    quaternion(T w, vec<T, 3> const& v) : storage { v.x, v.y, v.z, w } { }

    /// \brief Initialize quaternion from a vector; w should be the scalar.
    explicit quaternion(vec<T, 4> const& f) : storage(f) { }

public:
    /// \brief Convert to a vector
    explicit operator vec<T, 4>() const { return storage; }
};

using quat  = quaternion<float>;
using dquat = quaternion<double>;


// Operators ===================================================================

template <class T>
quaternion<T> operator+(quaternion<T> const& q, quaternion<T> const& r) {
    return quaternion<T>(q.storage + r.storage);
}

template <class T>
quaternion<T> operator-(quaternion<T> const& q, quaternion<T> const& r) {
    return quaternion<T>(q.storage - r.storage);
}

template <class T>
quaternion<T> operator*(quaternion<T> const& q, T scalar) {
    return quaternion<T>(q.storage * scalar);
}

// note that rotating a non-unit quaternion can do odd things

template <class T>
quaternion<T> operator*(quaternion<T> const& q, quaternion<T> const& r) {
    static constexpr dlal::vec<T, 4> mask1 { 1, 1, -1, -1 };
    static constexpr dlal::vec<T, 4> mask2 { -1, 1, 1, -1 };
    static constexpr dlal::vec<T, 4> mask3 { 1, -1, 1, -1 };

    dlal::vec<T, 4> p1 = r.storage.w * q.storage;
    dlal::vec<T, 4> p2 = mask1 * r.storage.x * q.storage.wzyx;
    dlal::vec<T, 4> p3 = mask2 * r.storage.y * q.storage.zwxy;
    dlal::vec<T, 4> p4 = mask3 * r.storage.z * q.storage.yxwz;

    return dlal::quat(p1 + p2 + p3 + p4);
}

template <class T>
vec<T, 3> operator*(quaternion<T> const& q, vec<T, 3> const& v) {
    vec<T, 3> const lqv = q.storage.xyz;
    vec<T, 3> const uv(cross(lqv, v));
    vec<T, 3> const uuv(cross(lqv, uv));

    return v + ((uv * q.w) + uuv) * static_cast<T>(2);
}


// Operations ==================================================================
template <class T>
T length(quaternion<T> const& q) {
    return length(vec<T, 4>(q));
}

template <class T>
quaternion<T> normalize(quaternion<T> const& q) {
    return quaternion<T>(normalize(q.storage));
}

template <class T>
quaternion<T> conjugate(quaternion<T> const& q) {
    return quaternion<T>(q.w, -vec<T, 3>(q.storage));
}

template <class T>
quaternion<T> inverse(quaternion<T> const& q) {
    return conjugate(q) / dot(q.storage, q.storage);
}

// Conversion ==================================================================


/// \brief Convert a UNIT quaternion to a mat3
template <class T>
mat<T, 3, 3> mat3_from_unit_quaternion(quaternion<T> const& q) {
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

    mat<T, 3, 3> ret;
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

/// \brief Convert a UNIT quaternion to a mat3
template <class T>
mat<T, 4, 4> mat4_from_unit_quaternion(quaternion<T> const& q) {
    auto m3 = mat3_from_unit_quaternion(q);

    mat<T, 4, 4> ret;

    ret[0] = vector_detail::v3to4(m3[0]);
    ret[1] = vector_detail::v3to4(m3[1]);
    ret[2] = vector_detail::v3to4(m3[2]);
    ret[3] = vec4 { 0, 0, 0, 1 };

    ret[0].w = 0;
    ret[1].w = 0;
    ret[2].w = 0;

    return ret;
}

template <class T>
quaternion<T> quaternion_from_matrix(mat<T, 3, 3> const& m) {
    quaternion<T> q;

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
quaternion<T> quaternion_from_matrix(mat<T, 4, 4> const& m) {
    return quaternion_from_matrix(mat<T, 3, 3>(m));
}

// Other =======================================================================

///
/// \brief Compute a rotation between two vectors
/// \param from A normalized source vector
/// \param to A normalized destination vector
///
template <class T>
quaternion<T> rotation_from_to(vec<T, 3> const& from, vec<T, 3> const& to) {
    vec<T, 3> const w = cross(from, to);

    vec<T, 4> lq { w.x, w.y, w.z, dot(from, to) };

    lq.w += dot(lq, lq);
    return normalize(quaternion<T>(lq));
}


///
/// \brief Compute a rotation given a direction and an 'up' vector
/// \param norm_direction The direction to look in, must be normalized
/// \param norm_up The 'up' direction, must be normalized
///
template <class T>
quaternion<T> look_at_lh(vec<T, 3> const& norm_direction,
                         vec<T, 3> const& norm_up) {
    if (std::abs(dot(norm_direction, norm_up)) >= 1) {
        return rotation_from_to(vec<T, 3> { 0, 0, -1 }, norm_direction);
    }

    mat<T, 3, 3> ret;
    ret[0] = normalize(cross(norm_up, norm_direction));
    ret[1] = cross(norm_direction, ret[0]);
    ret[2] = norm_direction;

    return quaternion<T>(quaternion_from_matrix(ret));
}

///
/// \brief Compute a quaternion from Euler angles, expressed in radians
///
template <class T>
quaternion<T> from_angles(vec<T, 3> angles) {
    vec<T, 3> const c = cos(angles * T(0.5));
    vec<T, 3> const s = sin(angles * T(0.5));

    quaternion<T> ret;
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
quaternion<T> from_angle_axis(T angle, vec<T, 3> axis) {
    T const s = std::sin(angle * static_cast<T>(0.5));

    return quaternion<T>(std::cos(angle * static_cast<T>(0.5)), axis * s);
}


} // namespace dlal

#endif // QUAT_H
