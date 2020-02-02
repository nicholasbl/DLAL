#ifndef LINALG_MATRIX_OPERATIONS_H
#define LINALG_MATRIX_OPERATIONS_H

#include "mat.h"

namespace dct {

template <class T, size_t C, size_t R>
mat<T, R, C> transpose(mat<T, C, R> const& m) {
    mat<T, R, C> ret;
    for (size_t c = 0; c < C; c++) {
        for (size_t r = 0; r < R; r++) {
            ret[r][c] = m[c][r];
        }
    }
    return ret;
}

template <class T>
mat<T, 2, 2> transpose(mat<T, 2, 2> const& m) {
    mat<T, 2, 2> ret;
    ret[0][0] = m[0][0];
    ret[0][1] = m[1][0];
    ret[1][0] = m[0][1];
    ret[1][1] = m[1][1];
    return ret;
}

template <class T>
mat<T, 3, 3> transpose(mat<T, 3, 3> const& m) {
    mat<T, 3, 3> ret;
    ret[0][0] = m[0][0];
    ret[0][1] = m[1][0];
    ret[0][2] = m[2][0];

    ret[1][0] = m[0][1];
    ret[1][1] = m[1][1];
    ret[1][2] = m[2][1];

    ret[2][0] = m[0][2];
    ret[2][1] = m[1][2];
    ret[2][2] = m[2][2];
    return ret;
}

template <class T>
mat<T, 4, 4> transpose(mat<T, 4, 4> const& m) {
    mat<T, 4, 4> ret;
    ret[0][0] = m[0][0];
    ret[0][1] = m[1][0];
    ret[0][2] = m[2][0];
    ret[0][3] = m[3][0];

    ret[1][0] = m[0][1];
    ret[1][1] = m[1][1];
    ret[1][2] = m[2][1];
    ret[1][3] = m[3][1];

    ret[2][0] = m[0][2];
    ret[2][1] = m[1][2];
    ret[2][2] = m[2][2];
    ret[2][3] = m[3][2];

    ret[3][0] = m[0][3];
    ret[3][1] = m[1][3];
    ret[3][2] = m[2][3];
    ret[3][3] = m[3][3];
    return ret;
}

template <class T>
inline T determinant(mat<T, 2, 2> const& m) {
    return m[0][0] * m[1][1] - m[1][0] * m[0][1];
}

template <class T>
inline T determinant(mat<T, 3, 3> const& m) {

    T const a = m[0][0];
    T const b = m[1][0];
    T const c = m[2][0];

    T const d = m[0][1];
    T const e = m[1][1];
    T const f = m[2][1];

    T const g = m[0][2];
    T const h = m[1][2];
    T const i = m[2][2];

    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

template <class T>
inline T determinant(mat<T, 4, 4> const& m) {
    T const a = m[2][2] * m[3][3] - m[3][2] * m[2][3];
    T const b = m[2][1] * m[3][3] - m[3][1] * m[2][3];
    T const c = m[2][1] * m[3][2] - m[3][1] * m[2][2];
    T const d = m[2][0] * m[3][3] - m[3][0] * m[2][3];
    T const e = m[2][0] * m[3][2] - m[3][0] * m[2][2];
    T const f = m[2][0] * m[3][1] - m[3][0] * m[2][1];

    vec<T, 4> const coeffs{ +(m[1][1] * a - m[1][2] * b + m[1][3] * c),
                            -(m[1][0] * a - m[1][2] * d + m[1][3] * e),
                            +(m[1][0] * b - m[1][1] * d + m[1][3] * f),
                            -(m[1][0] * c - m[1][1] * e + m[1][2] * f) };

    return component_sum(m[0] * coeffs);
}


template <class T>
auto inverse(mat<T, 2, 2> const& m) {
    T const one_over_det = static_cast<T>(1) / determinant(m);

    return mat<T, 2, 2>(m[1][1] * one_over_det,
                        -m[0][1] * one_over_det,
                        -m[1][0] * one_over_det,
                        m[0][0] * one_over_det);
}

template <class T>
auto inverse(mat<T, 3, 3> const& m) {
    T const one_over_det = static_cast<T>(1) / determinant(m);

    mat<T, 3, 3> ret;
    ret[0][0] = +(m[1][1] * m[2][2] - m[2][1] * m[1][2]) * one_over_det;
    ret[1][0] = -(m[1][0] * m[2][2] - m[2][0] * m[1][2]) * one_over_det;
    ret[2][0] = +(m[1][0] * m[2][1] - m[2][0] * m[1][1]) * one_over_det;
    ret[0][1] = -(m[0][1] * m[2][2] - m[2][1] * m[0][2]) * one_over_det;
    ret[1][1] = +(m[0][0] * m[2][2] - m[2][0] * m[0][2]) * one_over_det;
    ret[2][1] = -(m[0][0] * m[2][1] - m[2][0] * m[0][1]) * one_over_det;
    ret[0][2] = +(m[0][1] * m[1][2] - m[1][1] * m[0][2]) * one_over_det;
    ret[1][2] = -(m[0][0] * m[1][2] - m[1][0] * m[0][2]) * one_over_det;
    ret[2][2] = +(m[0][0] * m[1][1] - m[1][0] * m[0][1]) * one_over_det;

    return ret;
}


inline vec4 _matrix2x2_multiply(vec4 vec1, vec4 vec2) {
    return vec1 * vec2.xwxw + (vec1.yxwz * vec2.zyzy);
}

inline vec4 _matrix2x2_adj_mult(vec4 vec1, vec4 vec2) {
    return (vec1.wwxx * vec2) - (vec1.yyzz * vec2.zwxy);
}


inline vec4 _matrix2x2_mult_adj(vec4 vec1, vec4 vec2) {
    return (vec1 * vec2.wxwx) - (vec1.yxwz * vec2.zyzy);
}

namespace matrix_detail {

inline auto extract_a(vec4 a, vec4 b) { return _mm_movelh_ps(a, b); }
inline auto extract_b(vec4 a, vec4 b) { return _mm_movehl_ps(b, a); }

} // namespace matrix_detail

inline auto transform_inverse(mat4 const& m) {
    using namespace vector_detail;
    using namespace matrix_detail;
    // implementation based off Eric Zhang's
    // https://lxjk.github.io/2017/09/03/Fast-4x4-Matrix-Inverse-with-SSE-SIMD-Explained.html
    constexpr float SMALL_VALUE = 1E-8F;

    mat4 ret;

    __m128 t0 = extract_a(m[0], m[1]);
    __m128 t1 = extract_b(m[0], m[1]);
    ret[0]    = shuffle<0, 2, 4, 7>(t0, m[2]);
    ret[1]    = shuffle<1, 3, 5, 7>(t0, m[2]);
    ret[2]    = shuffle<0, 2, 6, 7>(t1, m[2]);

    __m128 size_sqr = ret[0] * ret[0];
    size_sqr += ret[1] * ret[1];
    size_sqr += ret[2] * ret[2];

    __m128 one{ 1.0f };
    __m128 rSizeSqr = _mm_blendv_ps(
        (one / size_sqr), one, _mm_cmplt_ps(size_sqr, __m128{ SMALL_VALUE }));

    ret[0] = (ret[0] * rSizeSqr);
    ret[1] = (ret[1] * rSizeSqr);
    ret[2] = (ret[2] * rSizeSqr);

    // last line
    ret[3] = (ret[0] * (m[3].xxxx));
    ret[3] = (ret[3] + (ret[1] * (m[3].yyyy)));
    ret[3] = (ret[3] + (ret[2] * (m[3].zzzz)));
    ret[3] = _mm_setr_ps(0.f, 0.f, 0.f, 1.f) - ret[3];

    return ret;
}

inline auto inverse(mat4 const& m) {
    using namespace vector_detail;
    using namespace matrix_detail;
    // implementation based off Eric Zhang's
    // https://lxjk.github.io/2017/09/03/Fast-4x4-Matrix-Inverse-with-SSE-SIMD-Explained.html

    vec4 const A = extract_a(m[0], m[1]);
    vec4 const B = extract_b(m[0], m[1]);
    vec4 const C = extract_a(m[2], m[3]);
    vec4 const D = extract_b(m[2], m[3]);

    vec4 const det_sub =
        (shuffle<0, 2, 4, 6>(m[0], m[2]) * shuffle<1, 3, 5, 7>(m[1], m[3])) -
        (shuffle<1, 3, 5, 7>(m[0], m[2]) * shuffle<0, 2, 4, 6>(m[1], m[3]));
    vec4 const det_A = det_sub.xxxx;
    vec4 const det_B = det_sub.yyyy;
    vec4 const det_C = det_sub.zzzz;
    vec4 const det_D = det_sub.wwww;

    vec4 D_C = _matrix2x2_adj_mult(D, C);
    vec4 A_B = _matrix2x2_adj_mult(A, B);
    vec4 X   = (det_D * A) - _matrix2x2_multiply(B, D_C);
    vec4 Y   = (det_B * C) - _matrix2x2_mult_adj(D, A_B);
    vec4 Z   = (det_C * B) - _matrix2x2_mult_adj(A, D_C);
    vec4 W   = (det_A * D) - _matrix2x2_multiply(C, A_B);

    __m128 det_M = (det_A * det_D);
    det_M        = (det_M + (det_B * det_C));

    __m128 trace = (A_B * D_C.xzyw);
    trace        = _mm_hadd_ps(trace, trace);
    trace        = _mm_hadd_ps(trace, trace);
    det_M        = (det_M - trace);

    __m128 const adj_sign_mask = { 1.f, -1.f, -1.f, 1.f };
    __m128 const i_det_M       = (adj_sign_mask / det_M);

    X = (X * i_det_M);
    Y = (Y * i_det_M);
    Z = (Z * i_det_M);
    W = (W * i_det_M);

    mat4 ret;
    ret[0] = shuffle<3, 1, 7, 5>(X, Y);
    ret[1] = shuffle<2, 0, 6, 4>(X, Y);
    ret[2] = shuffle<3, 1, 7, 5>(Z, W);
    ret[3] = shuffle<2, 0, 6, 4>(Z, W);

    return ret;
}

} // namespace dct

#endif // LINALG_MATRIX_OPERATIONS_H
