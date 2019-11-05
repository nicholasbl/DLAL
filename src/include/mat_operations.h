#ifndef LINALG_MATRIX_OPERATIONS_H
#define LINALG_MATRIX_OPERATIONS_H

#include "mat.h"

namespace dct {

template <class T, size_t C, size_t R>
MatrixCore<T, R, C> transpose(MatrixCore<T, C, R> const& m) {
    MatrixCore<T, R, C> ret;
    for (size_t c = 0; c < C; c++) {
        for (size_t r = 0; r < R; r++) {
            ret[r][c] = m[c][r];
        }
    }
    return ret;
}

template <class T>
MatrixCore<T, 2, 2> transpose(MatrixCore<T, 2, 2> const& m) {
    MatrixCore<T, 2, 2> ret;
    ret[0][0] = m[0][0];
    ret[0][1] = m[1][0];
    ret[1][0] = m[0][1];
    ret[1][1] = m[1][1];
    return ret;
}

template <class T>
MatrixCore<T, 3, 3> transpose(MatrixCore<T, 3, 3> const& m) {
    MatrixCore<T, 3, 3> ret;
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
MatrixCore<T, 4, 4> transpose(MatrixCore<T, 4, 4> const& m) {
    MatrixCore<T, 4, 4> ret;
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
inline T determinant(MatrixCore<T, 2, 2> const& m) {
    return m[0][0] * m[1][1] - m[1][0] * m[0][1];
}

template <class T>
inline T determinant(MatrixCore<T, 3, 3> const& m) {

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
inline T determinant(MatrixCore<T, 4, 4> const& m) {
    T const a = m[2][2] * m[3][3] - m[3][2] * m[2][3];
    T const b = m[2][1] * m[3][3] - m[3][1] * m[2][3];
    T const c = m[2][1] * m[3][2] - m[3][1] * m[2][2];
    T const d = m[2][0] * m[3][3] - m[3][0] * m[2][3];
    T const e = m[2][0] * m[3][2] - m[3][0] * m[2][2];
    T const f = m[2][0] * m[3][1] - m[3][0] * m[2][1];

    Vector<T, 4> const coeffs(+(m[1][1] * a - m[1][2] * b + m[1][3] * c),
                              -(m[1][0] * a - m[1][2] * d + m[1][3] * e),
                              +(m[1][0] * b - m[1][1] * d + m[1][3] * f),
                              -(m[1][0] * c - m[1][1] * e + m[1][2] * f));

    return component_sum(m[0] * coeffs);
}


template <class T>
auto inverse(MatrixCore<T, 2, 2> const& m) {
    T const one_over_det = static_cast<T>(1) / determinant(m);

    return MatrixCore<T, 2, 2>(m[1][1] * one_over_det,
                               -m[0][1] * one_over_det,
                               -m[1][0] * one_over_det,
                               m[0][0] * one_over_det);
}

template <class T>
auto inverse(MatrixCore<T, 3, 3> const& m) {
    T const one_over_det = static_cast<T>(1) / determinant(m);

    MatrixCore<T, 3, 3> ret;
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


inline __m128 _matrix2x2_multiply(__m128 vec1, __m128 vec2) {
    return vec1 * SWIZZLE(vec2, 0, 3, 0, 3) +
           (SWIZZLE(vec1, 1, 0, 3, 2) * SWIZZLE(vec2, 2, 1, 2, 1));
}

inline __m128 _matrix2x2_adj_mult(__m128 vec1, __m128 vec2) {
    return (SWIZZLE(vec1, 3, 3, 0, 0) * vec2) -
           (SWIZZLE(vec1, 1, 1, 2, 2) * SWIZZLE(vec2, 2, 3, 0, 1));
}


inline __m128 _matrix2x2_mult_adj(__m128 vec1, __m128 vec2) {
    return (vec1 * SWIZZLE(vec2, 3, 0, 3, 0)) -
           (SWIZZLE(vec1, 1, 0, 3, 2) * SWIZZLE(vec2, 2, 1, 2, 1));
}

inline auto transform_inverse(Mat4 const& m) {
    using namespace vector_detail;
    // implementation based off Eric Zhang's
    // https://lxjk.github.io/2017/09/03/Fast-4x4-Matrix-Inverse-with-SSE-SIMD-Explained.html
    constexpr float SMALL_VALUE = 1E-8F;

    Mat4 ret;

    __m128 t0     = shuffle<0, 1, 0, 1>(m.as_vec[0], m.as_vec[1]);
    __m128 t1     = shuffle<2, 3, 2, 3>(m.as_vec[0], m.as_vec[1]);
    ret.as_vec[0] = shuffle<0, 2, 0, 3>(t0, m.as_vec[2]);
    ret.as_vec[1] = shuffle<1, 3, 1, 3>(t0, m.as_vec[2]);
    ret.as_vec[2] = shuffle<0, 2, 2, 3>(t1, m.as_vec[2]);

    __m128 size_sqr = ret.as_vec[0] * ret.as_vec[0];
    size_sqr += ret.as_vec[1] * ret.as_vec[1];
    size_sqr += ret.as_vec[2] * ret.as_vec[2];

    __m128 one{ 1.0f };
    __m128 rSizeSqr = _mm_blendv_ps(
        (one / size_sqr), one, _mm_cmplt_ps(size_sqr, __m128{ SMALL_VALUE }));

    ret.as_vec[0] = (ret.as_vec[0] * rSizeSqr);
    ret.as_vec[1] = (ret.as_vec[1] * rSizeSqr);
    ret.as_vec[2] = (ret.as_vec[2] * rSizeSqr);

    // last line
    ret.as_vec[3] = (ret.as_vec[0] * swizzle<0>(m.as_vec[3]));
    ret.as_vec[3] = (ret.as_vec[3] + (ret.as_vec[1] * swizzle<1>(m.as_vec[3])));
    ret.as_vec[3] = (ret.as_vec[3] + (ret.as_vec[2] * swizzle<2>(m.as_vec[3])));
    ret.as_vec[3] = _mm_setr_ps(0.f, 0.f, 0.f, 1.f) - ret.as_vec[3];

    return ret;
}

inline auto inverse(Mat4 const& m) {
    using namespace vector_detail;
    // implementation based off Eric Zhang's
    // https://lxjk.github.io/2017/09/03/Fast-4x4-Matrix-Inverse-with-SSE-SIMD-Explained.html

    __m128 const A = shuffle<0, 1, 0, 1>(m.as_vec[0], m.as_vec[1]);
    __m128 const B = shuffle<2, 3, 2, 3>(m.as_vec[0], m.as_vec[1]);
    __m128 const C = shuffle<0, 1, 0, 1>(m.as_vec[2], m.as_vec[3]);
    __m128 const D = shuffle<2, 3, 2, 3>(m.as_vec[2], m.as_vec[3]);

    __m128 const det_sub = (shuffle<0, 2, 4, 6>(m.as_vec[0], m.as_vec[2]) *
                            shuffle<1, 3, 5, 7>(m.as_vec[1], m.as_vec[3])) -
                           (shuffle<1, 3, 5, 7>(m.as_vec[0], m.as_vec[2]) *
                            shuffle<0, 2, 4, 6>(m.as_vec[1], m.as_vec[3]));
    __m128 const det_A = swizzle<0>(det_sub);
    __m128 const det_B = swizzle<1>(det_sub);
    __m128 const det_C = swizzle<2>(det_sub);
    __m128 const det_D = swizzle<3>(det_sub);

    __m128 D_C = _matrix2x2_adj_mult(D, C);
    __m128 A_B = _matrix2x2_adj_mult(A, B);
    __m128 X   = (det_D * A) - _matrix2x2_multiply(B, D_C);
    __m128 Y   = (det_B * C) - _matrix2x2_mult_adj(D, A_B);
    __m128 Z   = (det_C * B) - _matrix2x2_mult_adj(A, D_C);
    __m128 W   = (det_A * D) - _matrix2x2_multiply(C, A_B);

    __m128 det_M = (det_A * det_D);
    det_M        = (det_M + (det_B * det_C));

    __m128 trace = (A_B * swizzle<0, 2, 1, 3>(D_C));
    trace        = _mm_hadd_ps(trace, trace);
    trace        = _mm_hadd_ps(trace, trace);
    det_M        = (det_M - trace);

    __m128 const adj_sign_mask = { 1.f, -1.f, -1.f, 1.f };
    __m128 const i_det_M       = (adj_sign_mask / det_M);

    X = (X * i_det_M);
    Y = (Y * i_det_M);
    Z = (Z * i_det_M);
    W = (W * i_det_M);

    Mat4 ret;
    ret.as_vec[0] = shuffle<3, 1, 7, 5>(X, Y);
    ret.as_vec[1] = shuffle<2, 0, 6, 4>(X, Y);
    ret.as_vec[2] = shuffle<3, 1, 7, 5>(Z, W);
    ret.as_vec[3] = shuffle<2, 0, 6, 4>(Z, W);

    return ret;
}

} // namespace dct

#endif // LINALG_MATRIX_OPERATIONS_H
