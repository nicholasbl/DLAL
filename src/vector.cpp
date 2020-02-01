#include "doctest.h"

#include "test_common.h"

#include <glm/gtx/norm.hpp>

#include <iostream>

using namespace dct;

template <class T>
bool is(Vector<T, 2> const& a, T x, T y) {
    return a.x == x and a.y == y;
}
template <class T>
bool is(Vector<T, 3> const& a, T x, T y, T z) {
    return a.x == x and a.y == y and a.z == z;
}
template <class T>
bool is(Vector<T, 4> const& a, T x, T y, T z, T w) {
    return a.x == x and a.y == y and a.z == z and a.w == w;
}

TEST_CASE("Vector Library") {

    SUBCASE("Constructors") {

        // vectors should init to zero

        {
            Vec2 zv2;
            Vec3 zv3;
            Vec4 zv4;

            glm::vec4 test;
            test.data = _mm_set1_ps(1);

            REQUIRE(is(zv2, 0.0f, 0.0f));
            REQUIRE(is(zv3, 0.0f, 0.0f, 0.0f));
            REQUIRE(is(zv4, 0.0f, 0.0f, 0.0f, 0.0f));
        }


        // init with value

        Vec4 v4_1(4.1f);
        REQUIRE(is(v4_1, 4.1f, 4.1f, 4.1f, 4.1f));

        Vec4      v4(1.0f, 2.3f, 3.2f, 4.1f);
        glm::vec4 lv4(1.0f, 2.3f, 3.2f, 4.1f);

        Vec3      v3(1.0, 2.3f, 3.2f);
        glm::vec3 lv3(1.0, 2.3f, 3.2f);

        Vec2      v2(1.0, 2.3f);
        glm::vec2 lv2(1.0, 2.3f);

        REQUIRE(is_same(v2, lv2));
        REQUIRE(is_same(v3, lv3));
        REQUIRE(is_same(v4, lv4));


        Vec4 w1(v2, 3.2f, 4.1f);
        Vec4 w2(3.2f, v2, 4.1f);
        Vec4 w3(3.2f, 4.1f, v2);

        REQUIRE(is_same(w1, glm::vec4(1.0f, 2.3f, 3.2f, 4.1f)));
        REQUIRE(is_same(w2, glm::vec4(3.2f, 1.0f, 2.3f, 4.1f)));
        REQUIRE(is_same(w3, glm::vec4(3.2f, 4.1f, 1.0f, 2.3f)));

        v2.x = 10;
        v2.y = 20;
        REQUIRE(is_same(v2, glm::vec2(10.0f, 20.0f)));

        v3.x = 10;
        v3.y = 20;
        v3.z = 30;
        REQUIRE(is_same(v3, glm::vec3(10.0f, 20.0f, 30.0f)));

        v4.x = 10;
        v4.y = 20;
        v4.z = 30;
        v4.w = 40;
        REQUIRE(is_same(v4, glm::vec4(10.0f, 20.0f, 30.0f, 40.0f)));
    }

    SUBCASE("Operators - Unary") {
        REQUIRE(is(-Vec4(1.0, 2.0, 3.0, 4.0), -1.0f, -2.0f, -3.0f, -4.0f));

        REQUIRE(is(!Vector<bool, 4>(true, false, false, true),
                   false,
                   true,
                   true,
                   false));
    }


    SUBCASE("Operators - Binary") {
        {
            REQUIRE(is(
                Vec4(1.0, 2.0, 3.0, 4.0) + Vec4(5.0), 6.0f, 7.0f, 8.0f, 9.0f));

            REQUIRE(
                is(Vec4(1.0, 2.0, 3.0, 4.0) + 5.0f, 6.0f, 7.0f, 8.0f, 9.0f));

            REQUIRE(
                is(5.0f + Vec4(1.0, 2.0, 3.0, 4.0), 6.0f, 7.0f, 8.0f, 9.0f));


            {
                Vec4 a(1.0, 2.0, 3.0, 4.0);
                a += Vec4(5.0);
                REQUIRE(is(a, 6.0f, 7.0f, 8.0f, 9.0f));
            }

            {
                Vec4 a(1.0, 2.0, 3.0, 4.0);
                a += 5.0f;
                REQUIRE(is(a, 6.0f, 7.0f, 8.0f, 9.0f));
            }
        }

        {
            REQUIRE(is(Vec4(1.0, 2.0, 3.0, 4.0) - Vec4(5.0),
                       -4.0f,
                       -3.0f,
                       -2.0f,
                       -1.0f));

            REQUIRE(is(
                Vec4(1.0, 2.0, 3.0, 4.0) - 5.0f, -4.0f, -3.0f, -2.0f, -1.0f));

            REQUIRE(
                is(5.0f - Vec4(1.0, 2.0, 3.0, 4.0), 4.0f, 3.0f, 2.0f, 1.0f));

            {
                Vec4 a(1.0, 2.0, 3.0, 4.0);
                a -= Vec4(5.0);
                REQUIRE(is(a, -4.0f, -3.0f, -2.0f, -1.0f));
            }

            {
                Vec4 a(1.0, 2.0, 3.0, 4.0);
                a -= 5.0f;
                REQUIRE(is(a, -4.0f, -3.0f, -2.0f, -1.0f));
            }
        }

        {
            REQUIRE(is(Vec4(1.0, 2.0, 3.0, 4.0) * Vec4(5.0),
                       5.0f,
                       10.0f,
                       15.0f,
                       20.0f));

            REQUIRE(
                is(Vec4(1.0, 2.0, 3.0, 4.0) * 5.0f, 5.0f, 10.0f, 15.0f, 20.0f));

            REQUIRE(
                is(5.0f * Vec4(1.0, 2.0, 3.0, 4.0), 5.0f, 10.0f, 15.0f, 20.0f));

            {
                Vec4 a(1.0, 2.0, 3.0, 4.0);
                a *= Vec4(5.0);
                REQUIRE(is(a, 5.0f, 10.0f, 15.0f, 20.0f));
            }

            {
                Vec4 a(1.0, 2.0, 3.0, 4.0);
                a *= 5.0f;
                REQUIRE(is(a, 5.0f, 10.0f, 15.0f, 20.0f));
            }
        }

        {
            REQUIRE(is(Vec4(1.0f, 2.0f, 3.0f, 4.0f) / Vec4(5.0f),
                       1.0f / 5.0f,
                       2.0f / 5.0f,
                       3.0f / 5.0f,
                       4.0f / 5.0f));

            REQUIRE(is(Vec4(1.0f, 2.0f, 3.0f, 4.0f) / 5.0f,
                       1.0f / 5.0f,
                       2.0f / 5.0f,
                       3.0f / 5.0f,
                       4.0f / 5.0f));

            REQUIRE(is(5.0f / Vec4(1.0f, 2.0f, 3.0f, 4.0f),
                       5.0f / 1.0f,
                       5.0f / 2.0f,
                       5.0f / 3.0f,
                       5.0f / 4.0f));

            {
                Vec4 a(1.0, 2.0, 3.0, 4.0);
                a /= Vec4(5.0);
                REQUIRE(
                    is(a, 1.0f / 5.0f, 2.0f / 5.0f, 3.0f / 5.0f, 4.0f / 5.0f));
            }

            {
                Vec4 a(1.0, 2.0, 3.0, 4.0);
                a /= 5.0f;
                REQUIRE(
                    is(a, 1.0f / 5.0f, 2.0f / 5.0f, 3.0f / 5.0f, 4.0f / 5.0f));
            }
        }


        REQUIRE(is(Vec4(1.0f, 2.0f, 3.0f, 4.0f) == Vec4(1.0f, 2.0f, 2.0f, 4.0f),
                   true,
                   true,
                   false,
                   true));

        REQUIRE(is(Vec4(1.0f, 2.0f, 3.0f, 4.0f) != Vec4(1.0f, 2.0f, 2.0f, 4.0f),
                   false,
                   false,
                   true,
                   false));


        REQUIRE(is(Vec4(0.0f, 2.0f, 3.0f, 4.0f) < Vec4(1.0f, 2.0f, 2.0f, 4.0f),
                   true,
                   false,
                   false,
                   false));

        REQUIRE(is(Vec4(1.0f, 2.0f, 3.0f, 4.0f) > Vec4(1.0f, 2.0f, 2.0f, 4.0f),
                   false,
                   false,
                   true,
                   false));

        REQUIRE(is(Vec4(0.0f, 2.0f, 3.0f, 4.0f) <= Vec4(1.0f, 2.0f, 2.0f, 4.0f),
                   true,
                   true,
                   false,
                   true));

        REQUIRE(is(Vec4(1.0f, 2.0f, 3.0f, 3.0f) >= Vec4(1.0f, 2.0f, 2.0f, 4.0f),
                   true,
                   true,
                   true,
                   false));

        auto and_test =
            BVec4(false, true, true, false) and BVec4(false, true, false, true);

        REQUIRE(is(and_test, false, true, false, false));

        auto or_test =
            BVec4(false, true, true, false) or BVec4(false, true, false, true);

        REQUIRE(is(or_test, false, true, true, true));
    }

    SUBCASE("Vector Math - Dot") {

        auto da =
            dot(Vec4(1.0f, 2.0f, 3.0f, 4.0f), Vec4(3.0f, 2.0f, 1.0f, 8.0f));
        auto db = glm::dot(glm::vec4(1.0f, 2.0f, 3.0f, 4.0f),
                           glm::vec4(3.0f, 2.0f, 1.0f, 8.0f));

        REQUIRE(da == db);
    }

    SUBCASE("Vector Math - Cross") {

        auto da = cross(Vec3(1.0f, 2.0f, 3.0f), Vec3(3.0f, 2.0f, 1.0f));
        auto db = glm::cross(glm::vec3(1.0f, 2.0f, 3.0f),
                             glm::vec3(3.0f, 2.0f, 1.0f));

        REQUIRE(is_same(da, db));
    }

    SUBCASE("Vector Math - Lengths") {
        auto la = length(Vec4(1.0f, 2.0f, 3.0f, 4.0f));
        auto lb = glm::length(glm::vec4(1.0f, 2.0f, 3.0f, 4.0f));
        REQUIRE(la == lb);

        auto sla = length_squared(Vec4(1.0f, 2.0f, 3.0f, 4.0f));
        auto slb = glm::length2(glm::vec4(1.0f, 2.0f, 3.0f, 4.0f));
        REQUIRE(sla == slb);


        auto da = distance(Vec4(1.0f, 2.0f, 3.0f, 4.0f),
                           Vec4(3.0f, 2.0f, 1.0f, 8.0f));
        auto db = glm::distance(glm::vec4(1.0f, 2.0f, 3.0f, 4.0f),
                                glm::vec4(3.0f, 2.0f, 1.0f, 8.0f));

        REQUIRE(da == db);

        auto da2 = distance_squared(Vec4(1.0f, 2.0f, 3.0f, 4.0f),
                                    Vec4(3.0f, 2.0f, 1.0f, 8.0f));
        auto db2 = glm::distance2(glm::vec4(1.0f, 2.0f, 3.0f, 4.0f),
                                  glm::vec4(3.0f, 2.0f, 1.0f, 8.0f));

        REQUIRE(da2 == db2);

        auto na = normalize(Vec4(3.0f, 2.0f, 1.0f, 8.0f));
        auto nb = glm::normalize(glm::vec4(3.0f, 2.0f, 1.0f, 8.0f));

        REQUIRE(is_same(na, nb));
        REQUIRE(length(na) == 1);
    }

    SUBCASE("Vector Math - Operations") {
        auto da2 = reflect(Vec3(1.0f, 2.0f, 3.0f), Vec3(3.0f, 2.0f, 1.0f));
        auto db2 = glm::reflect(glm::vec3(1.0f, 2.0f, 3.0f),
                                glm::vec3(3.0f, 2.0f, 1.0f));

        REQUIRE(is_same(da2, db2));
    }

    SUBCASE("Vector Math - Boolean") {
        REQUIRE(is_all(BVec3(true, true, true)));
        REQUIRE(!is_all(BVec3(true, false, true)));

        REQUIRE(is_any(BVec3(true, true, true)));
        REQUIRE(is_any(BVec3(true, false, true)));
        REQUIRE(!is_any(BVec3(false, false, false)));
    }

    SUBCASE("Vector Math - Other") {

        REQUIRE(is(
            min(Vec4(1.0f, 2.0f, 5.0f, -1.0f), Vec4(2.0f, 1.0f, 10.0f, -5.0f)),
            1.0f,
            1.0f,
            5.0f,
            -5.0f));

        REQUIRE(is(
            max(Vec4(1.0f, 2.0f, 5.0f, -1.0f), Vec4(2.0f, 1.0f, 10.0f, -5.0f)),
            2.0f,
            2.0f,
            10.0f,
            -1.0f));

        REQUIRE(component_min(Vec1(3.0f)) == 3.0f);
        REQUIRE(component_max(Vec1(3.0f)) == 3.0f);

        REQUIRE(component_min(Vec2(3.0f, 4.0f)) == 3.0f);
        REQUIRE(component_max(Vec2(3.0f, 4.0f)) == 4.0f);

        REQUIRE(component_min(Vec3(-1.0f, 1.0f, 4.0f)) == -1.0f);
        REQUIRE(component_max(Vec3(-1.0f, 1.0f, 4.0f)) == 4.0f);

        REQUIRE(component_min(Vec4(3.0f, -1.0f, 1.0f, 4.0f)) == -1.0f);
        REQUIRE(component_max(Vec4(3.0f, -1.0f, 1.0f, 4.0f)) == 4.0f);


        REQUIRE(component_sum(Vec4(1.0f, 2.0f, 5.0f, -1.0f)) == 7.0f);

        REQUIRE(
            is(abs(Vec4(1.0f, -2.0f, 5.0f, -1.0f)), 1.0f, 2.0f, 5.0f, 1.0f));

        REQUIRE(is(
            floor(Vec4(1.5f, -2.5f, 5.5f, -1.5f)), 1.0f, -3.0f, 5.0f, -2.0f));

        REQUIRE(
            is(ceil(Vec4(1.5f, -2.5f, 5.5f, -1.5f)), 2.0f, -2.0f, 6.0f, -1.0f));
    }

    SUBCASE("Vector Math - Mix") {
        REQUIRE(mix(1.0f, 2.0f, true) == 2.0f);
        REQUIRE(mix(1.0f, 2.0f, 0.5f) == 1.5f);

        REQUIRE(is(mix(Vec4(1.0f, -2.0f, 5.0f, -1.0f),
                       Vec4(2.0f, -4.0f, 10.0f, -1.0f),
                       0.5f),
                   1.5f,
                   -3.0f,
                   7.5f,
                   -1.0f));

        REQUIRE(is(mix(Vec4(1.0f, -2.0f, 5.0f, -1.0f),
                       Vec4(2.0f, -4.0f, 10.0f, -1.0f),
                       Vec4(0.5f, 0.0f, 1.0f, 0.5f)),
                   1.5f,
                   -2.0f,
                   10.0f,
                   -1.0f));
    }
}
