#include "include/quat.h"
#include "doctest.h"

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/norm.hpp>

#include <iostream>

using namespace dct;

template <class T>
bool fuzzy_eq(T a, T b, long double bound) {
    static_assert(std::is_floating_point_v<T>);
    return std::abs(static_cast<long double>(a) - static_cast<long double>(b)) <
           bound;
}


template <class T>
bool is(QuaternionCore<T> const& a,
        T                        x,
        T                        y,
        T                        z,
        T                        w,
        long double              limit = std::numeric_limits<T>::epsilon()) {
    bool ret = fuzzy_eq(a.x, x, limit) and fuzzy_eq(a.y, y, limit) and
               fuzzy_eq(a.z, z, limit) and fuzzy_eq(a.w, w, limit);
    if (!ret) {
        MESSAGE(a.x << " " << a.y << " " << a.z << " " << a.w << " != " << x
                    << " " << y << " " << z << " " << w);
        MESSAGE("TESTS " << fuzzy_eq(a.x, x, limit) << fuzzy_eq(a.y, y, limit)
                         << fuzzy_eq(a.z, z, limit) << fuzzy_eq(a.w, w, limit));
    }
    return ret;
}

template <class T>
bool verify(QuaternionCore<T> const& a,
            glm::qua<T> const&       b,
            long double limit = std::numeric_limits<T>::epsilon()) {
    bool ret = fuzzy_eq(a.x, b.x, limit) and fuzzy_eq(a.y, b.y, limit) and
               fuzzy_eq(a.z, b.z, limit) and fuzzy_eq(a.w, b.w, limit);
    if (!ret) {
        MESSAGE(a.x << " " << a.y << " " << a.z << " " << a.w << " != " << b.x
                    << " " << b.y << " " << b.z << " " << b.w);
        MESSAGE(
            "TESTS " << fuzzy_eq(a.x, b.x, limit) << fuzzy_eq(a.y, b.y, limit)
                     << fuzzy_eq(a.z, b.z, limit) << fuzzy_eq(a.w, b.w, limit));
    }
    return ret;
}

template <class T, class Function>
bool quat_check(Vector<T, 4> a, Function f) {
    QuaternionCore<T> q1(Vec4(a.x, a.y, a.z, a.w));

    glm::qua<T> q3(a.w, a.x, a.y, a.z);

    return f(q1, q3);
}

template <class T, class Function>
bool binary_quat_check(Vector<T, 4> a, Vector<T, 4> b, Function f) {
    QuaternionCore<T> q1(Vec4(a.x, a.y, a.z, a.w));
    QuaternionCore<T> q2(Vec4(b.x, b.y, b.z, b.w));

    glm::qua<T> q3(a.w, a.x, a.y, a.z);
    glm::qua<T> q4(b.w, b.x, b.y, b.z);

    return f(q1, q2, q3, q4);
}

template <class T, size_t N, int M>
bool is_same(Vector<T, N> const& a, glm::vec<M, T> const& b) {
    // weirdness due to how glm specifies their templates
    static_assert(N == M);
    for (size_t i = 0; i < N; ++i) {
        T delta = std::abs(a[i] - b[static_cast<int>(i)]);
        if (std::numeric_limits<T>::epsilon() < delta) {
            return false;
        }
    }
    return true;
}

template <class T, size_t C, size_t R>
bool is_same(MatrixCore<T, C, R> const&         a,
             glm::mat<int(C), int(R), T> const& b) {
    // weirdness due to how glm specifies their templates
    // static_assert(C == M);
    for (size_t i = 0; i < C * R; ++i) {
        T delta =
            std::abs(a.data()[i] - glm::value_ptr(b)[static_cast<int>(i)]);
        if (std::numeric_limits<T>::epsilon() < delta) {
            return false;
        }
    }
    return true;
}

TEST_CASE("Quaternion") {

    SUBCASE("Constructors") {
        Quat q;

        REQUIRE(is(q, 0.0f, 0.0f, 0.0f, 1.0f));

        Quat q2(1.0, { 2.0f, 3.0f, 4.0f });

        REQUIRE(is(q2, 2.0f, 3.0f, 4.0f, 1.0f));

        Quat q3(Vec4(2.0f, 3.0f, 4.0f, 1.0f));

        REQUIRE(is(q3, 2.0f, 3.0f, 4.0f, 1.0f));
    }

    SUBCASE("Operators") {
        REQUIRE(
            binary_quat_check<float>({ 1, 2, 3, 4 },
                                     { 1, 5, 7, 9 },
                                     [](auto q1, auto q2, auto r1, auto r2) {
                                         return verify(q1 + q2, r1 + r2);
                                     }));

        REQUIRE(
            binary_quat_check<float>({ 1, 2, 3, 4 },
                                     { 1, 5, 7, 9 },
                                     [](auto q1, auto q2, auto r1, auto r2) {
                                         return verify(q1 - q2, r1 - r2);
                                     }));

        REQUIRE(
            binary_quat_check<float>({ 1, 2, 3, 4 },
                                     { 1, 5, 7, 9 },
                                     [](auto q1, auto q2, auto r1, auto r2) {
                                         return verify(q1 * q2, r1 * r2);
                                     }));

        REQUIRE(binary_quat_check<float>(
            { 1, 2, 3, 4 }, { 1, 5, 7, 9 }, [](auto q1, auto, auto r1, auto) {
                return verify(q1 * 2.0f, r1 * 2.0f);
            }));

        REQUIRE(binary_quat_check<float>(
            { 1, 2, 3, 4 }, { 1, 5, 7, 9 }, [](auto q1, auto, auto r1, auto) {
                return is_same(q1 * Vec3(1, 3, -1), r1 * glm::vec3(1, 3, -1));
            }));
    }

    SUBCASE("Operations") {
        REQUIRE(quat_check<float>({ 1, -1, 3, 4 }, [](auto q1, auto r1) {
            return length(q1) == glm::length(r1);
        }));

        REQUIRE(quat_check<float>({ 1, -1, 3, 4 }, [](auto q1, auto r1) {
            return verify(normalize(q1), glm::normalize(r1));
        }));
    }

    SUBCASE("Conversion") {
        REQUIRE(quat_check<float>({ 1, -1, 3, 3 }, [](auto q1, auto r1) {
            return is_same(mat3_from_unit_quaternion(q1), glm::mat3_cast(r1));
        }));

        {
            Mat3      m1({ 1, 2, 3, 0, 3, 1, 4, 0, 1 });
            glm::mat3 m2(1, 2, 3, 0, 3, 1, 4, 0, 1);

            REQUIRE(verify(quaternion_from_matrix(m1), glm::quat_cast(m2)));
        }

        {
            Mat3      m1({ 1, 2, 3, 0, -3, 1, 4, 0, 1 });
            glm::mat3 m2(1, 2, 3, 0, -3, 1, 4, 0, 1);

            REQUIRE(verify(quaternion_from_matrix(m1), glm::quat_cast(m2)));
        }

        {
            Mat3      m1({ 1, 2, 3, 0, -4, 1, 4, 0, 1 });
            glm::mat3 m2(1, 2, 3, 0, -4, 1, 4, 0, 1);

            REQUIRE(verify(quaternion_from_matrix(m1), glm::quat_cast(m2)));
        }

        {
            Mat3      m1({ 2, 2, 3, 0, -4, 1, 4, 0, 1 });
            glm::mat3 m2(2, 2, 3, 0, -4, 1, 4, 0, 1);

            REQUIRE(verify(quaternion_from_matrix(m1), glm::quat_cast(m2)));
        }

        {
            Mat3      m1({ -11, 2, 3, 0, 5, 1, 4, 0, 1 });
            glm::mat3 m2(-11, 2, 3, 0, 5, 1, 4, 0, 1);

            REQUIRE(verify(quaternion_from_matrix(m1), glm::quat_cast(m2)));
        }

        {
            Mat3      m1({ -10, 2, 3, 0, 5, 1, 4, 0, 1 });
            glm::mat3 m2(-10, 2, 3, 0, 5, 1, 4, 0, 1);

            REQUIRE(verify(quaternion_from_matrix(m1), glm::quat_cast(m2)));
        }
    }

    SUBCASE("Other") {
        // we use different approaches, thus we loosen the bounds just a bit
        {
            Vec3 a = normalize(Vec3(1, 2, -1));
            Vec3 b = normalize(Vec3(-1, -1, -1));

            auto quat = rotation_from_to(a, b);

            glm::vec3 ga = normalize(glm::vec3(1, 2, -1));
            glm::vec3 gb = normalize(glm::vec3(-1, -1, -1));

            auto gquat = glm::rotation(ga, gb);

            REQUIRE(
                verify(quat, gquat, 2 * std::numeric_limits<float>::epsilon()));
        }

        {
            Vec3 a = normalize(Vec3(-2, 2, -1));
            Vec3 b = normalize(Vec3(-1, -4, -1));

            auto quat = rotation_from_to(a, b);

            glm::vec3 ga = normalize(glm::vec3(-2, 2, -1));
            glm::vec3 gb = normalize(glm::vec3(-1, -4, -1));

            auto gquat = glm::rotation(ga, gb);

            REQUIRE(
                verify(quat, gquat, 3 * std::numeric_limits<float>::epsilon()));
        }

        // look at

        {
            Vec3 a = normalize(Vec3(-2, 2, -1));
            Vec3 b = normalize(Vec3(-1, -4, -1));

            auto quat = look_at(a, b);

            glm::vec3 ga = normalize(glm::vec3(-2, 2, -1));
            glm::vec3 gb = normalize(glm::vec3(-1, -4, -1));

            auto gquat = glm::quatLookAtLH(ga, gb);

            REQUIRE(verify(quat, gquat));
        }

        // from angles

        {
            Vec3 a(-2, 2, -1);

            auto quat = from_angles(a);

            glm::vec3 ga(-2, 2, -1);

            auto gquat = glm::quat(ga);

            REQUIRE(verify(quat, gquat));
        }

        // angle axis

        {
            Vec3 a(-2, 2, -1);

            auto quat = from_angle_axis(.25f, a);

            glm::vec3 ga(-2, 2, -1);

            auto gquat = glm::angleAxis(.25f, ga);

            REQUIRE(verify(quat, gquat));
        }
    }
}
