#include "doctest.h"

#include "test_common.h"

#include <glm/gtx/norm.hpp>

#include <iostream>

using namespace dlal;

template <class T, class U>
bool is(vec<T, 2> const& a, U x, U y) {
    return a.x == x and a.y == y;
}
template <class T, class U>
bool is(vec<T, 3> const& a, U x, U y, U z) {
    return a.x == x and a.y == y and a.z == z;
}
template <class T, class U>
bool is(vec<T, 4> const& a, U x, U y, U z, U w) {
    return a.x == x and a.y == y and a.z == z and a.w == w;
}

TEST_CASE("Vector Library - Constructors") {

    SUBCASE("Default") {
        vec2 zv2 = {};
        vec3 zv3 = {};
        vec4 zv4 = {};

        REQUIRE(is(zv2, 0.0f, 0.0f));
        REQUIRE(is(zv3, 0.0f, 0.0f, 0.0f));
        REQUIRE(is(zv4, 0.0f, 0.0f, 0.0f, 0.0f));
    }


    SUBCASE("With Value") {
        vec4 v4_1(4.1f);
        REQUIRE(is(v4_1, 4.1f, 4.1f, 4.1f, 4.1f));
    }

    SUBCASE("With Values") {
        vec4      v4 { 1.0f, 2.3f, 3.2f, 4.1f };
        glm::vec4 lv4(1.0f, 2.3f, 3.2f, 4.1f);

        vec3      v3 { 1.0, 2.3f, 3.2f };
        glm::vec3 lv3(1.0, 2.3f, 3.2f);

        vec2      v2 { 1.0, 2.3f };
        glm::vec2 lv2 { 1.0, 2.3f };

        REQUIRE(is_same(v2, lv2));
        REQUIRE(is_same(v3, lv3));
        REQUIRE(is_same(v4, lv4));

        {
            vec4 v4 { 1.0f };
            REQUIRE(is(v4, 1, 0, 0, 0));
        }
    }

    SUBCASE("With Value") {

        vec2 v2 { 1.0, 2.3f };
        vec3 v3 { 1.0, 2.3f, 3.2f };
        vec4 v4 { 1.0f, 2.3f, 3.2f, 4.1f };

        vec4 w1 = new_vec(v2, 3.2f, 4.1f);
        vec4 w2 = new_vec(3.2f, v2, 4.1f);
        vec4 w3 = new_vec(3.2f, 4.1f, v2);

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
}

TEST_CASE("Vector Library - Operators") {

    SUBCASE("Unary") {
        REQUIRE(is(-vec4 { 1.0, 2.0, 3.0, 4.0 }, -1.0f, -2.0f, -3.0f, -4.0f));

        REQUIRE(
            is_equal(!vec<int, 4> { true, false, false, true },
                     vec<int, 4> { VEC_FALSE, VEC_TRUE, VEC_TRUE, VEC_FALSE }));
    }


    SUBCASE("Addition") {

        REQUIRE(is(
            vec4 { 1.0, 2.0, 3.0, 4.0 } + vec4(5.0), 6.0f, 7.0f, 8.0f, 9.0f));

        REQUIRE(is(vec4 { 1.0, 2.0, 3.0, 4.0 } + 5.0f, 6.0f, 7.0f, 8.0f, 9.0f));

        REQUIRE(is(5.0f + vec4 { 1.0, 2.0, 3.0, 4.0 }, 6.0f, 7.0f, 8.0f, 9.0f));


        {
            vec4 a { 1.0, 2.0, 3.0, 4.0 };
            a += vec4(5.0);
            REQUIRE(is(a, 6.0f, 7.0f, 8.0f, 9.0f));
        }

        {
            vec4 a { 1.0, 2.0, 3.0, 4.0 };
            a += 5.0f;
            REQUIRE(is(a, 6.0f, 7.0f, 8.0f, 9.0f));
        }
    }


    SUBCASE("Subtraction") {

        REQUIRE(is(vec4 { 1.0, 2.0, 3.0, 4.0 } - vec4(5.0),
                   -4.0f,
                   -3.0f,
                   -2.0f,
                   -1.0f));

        REQUIRE(
            is(vec4 { 1.0, 2.0, 3.0, 4.0 } - 5.0f, -4.0f, -3.0f, -2.0f, -1.0f));

        REQUIRE(is(5.0f - vec4 { 1.0, 2.0, 3.0, 4.0 }, 4.0f, 3.0f, 2.0f, 1.0f));

        {
            vec4 a { 1.0, 2.0, 3.0, 4.0 };
            a -= vec4(5.0);
            REQUIRE(is(a, -4.0f, -3.0f, -2.0f, -1.0f));
        }

        {
            vec4 a { 1.0, 2.0, 3.0, 4.0 };
            a -= 5.0f;
            REQUIRE(is(a, -4.0f, -3.0f, -2.0f, -1.0f));
        }
    }


    SUBCASE("Mult") {

        REQUIRE(is(vec4 { 1.0, 2.0, 3.0, 4.0 } * vec4(5.0),
                   5.0f,
                   10.0f,
                   15.0f,
                   20.0f));

        REQUIRE(
            is(vec4 { 1.0, 2.0, 3.0, 4.0 } * 5.0f, 5.0f, 10.0f, 15.0f, 20.0f));

        REQUIRE(
            is(5.0f * vec4 { 1.0, 2.0, 3.0, 4.0 }, 5.0f, 10.0f, 15.0f, 20.0f));

        {
            vec4 a { 1.0, 2.0, 3.0, 4.0 };
            a *= vec4(5.0);
            REQUIRE(is(a, 5.0f, 10.0f, 15.0f, 20.0f));
        }

        {
            vec4 a { 1.0, 2.0, 3.0, 4.0 };
            a *= 5.0f;
            REQUIRE(is(a, 5.0f, 10.0f, 15.0f, 20.0f));
        }
    }


    SUBCASE("Div") {


        REQUIRE(is(vec4 { 1.0f, 2.0f, 3.0f, 4.0f } / vec4(5.0f),
                   1.0f / 5.0f,
                   2.0f / 5.0f,
                   3.0f / 5.0f,
                   4.0f / 5.0f));

        REQUIRE(is(vec4 { 1.0f, 2.0f, 3.0f, 4.0f } / 5.0f,
                   1.0f / 5.0f,
                   2.0f / 5.0f,
                   3.0f / 5.0f,
                   4.0f / 5.0f));

        REQUIRE(is(5.0f / vec4 { 1.0f, 2.0f, 3.0f, 4.0f },
                   5.0f / 1.0f,
                   5.0f / 2.0f,
                   5.0f / 3.0f,
                   5.0f / 4.0f));

        {
            vec4 a { 1.0, 2.0, 3.0, 4.0 };
            a /= vec4(5.0);
            REQUIRE(is(a, 1.0f / 5.0f, 2.0f / 5.0f, 3.0f / 5.0f, 4.0f / 5.0f));
        }

        {
            vec4 a { 1.0, 2.0, 3.0, 4.0 };
            a /= 5.0f;
            REQUIRE(is(a, 1.0f / 5.0f, 2.0f / 5.0f, 3.0f / 5.0f, 4.0f / 5.0f));
        }
    }


    SUBCASE("Equality") {

        REQUIRE(is(vec4 { 1.0f, 2.0f, 3.0f, 4.0f } ==
                       vec4 { 1.0f, 2.0f, 2.0f, 4.0f },
                   VEC_TRUE,
                   VEC_TRUE,
                   VEC_FALSE,
                   VEC_TRUE));

        REQUIRE(is(vec4 { 1.0f, 2.0f, 3.0f, 4.0f } !=
                       vec4 { 1.0f, 2.0f, 2.0f, 4.0f },
                   VEC_FALSE,
                   VEC_FALSE,
                   VEC_TRUE,
                   VEC_FALSE));


        REQUIRE(is(vec4 { 0.0f, 2.0f, 3.0f, 4.0f } <
                       vec4 { 1.0f, 2.0f, 2.0f, 4.0f },
                   VEC_TRUE,
                   VEC_FALSE,
                   VEC_FALSE,
                   VEC_FALSE));

        REQUIRE(is(vec4 { 1.0f, 2.0f, 3.0f, 4.0f } >
                       vec4 { 1.0f, 2.0f, 2.0f, 4.0f },
                   VEC_FALSE,
                   VEC_FALSE,
                   VEC_TRUE,
                   VEC_FALSE));

        REQUIRE(is(vec4 { 0.0f, 2.0f, 3.0f, 4.0f } <=
                       vec4 { 1.0f, 2.0f, 2.0f, 4.0f },
                   VEC_TRUE,
                   VEC_TRUE,
                   VEC_FALSE,
                   VEC_TRUE));

        REQUIRE(is(vec4 { 1.0f, 2.0f, 3.0f, 3.0f } >=
                       vec4 { 1.0f, 2.0f, 2.0f, 4.0f },
                   VEC_TRUE,
                   VEC_TRUE,
                   VEC_TRUE,
                   VEC_FALSE));

        auto and_test = ivec4 { false, true, true, false } and
                        ivec4 { false, true, false, true };

        REQUIRE(is(and_test, VEC_FALSE, VEC_TRUE, VEC_FALSE, VEC_FALSE));

        auto or_test = ivec4 { VEC_FALSE, VEC_TRUE, VEC_TRUE, VEC_FALSE } or
                       ivec4 { VEC_FALSE, VEC_TRUE, VEC_FALSE, VEC_TRUE };

        REQUIRE(is(or_test, VEC_FALSE, VEC_TRUE, VEC_TRUE, VEC_TRUE));
    }
}

TEST_CASE("Vector Library - Operations") {

    SUBCASE("Dot") {

        {
            auto da = dot(vec4 { 1.0f, 2.0f, 3.0f, 4.0f },
                          vec4 { 3.0f, 2.0f, 1.0f, 8.0f });
            auto db = glm::dot(glm::vec4 { 1.0f, 2.0f, 3.0f, 4.0f },
                               glm::vec4 { 3.0f, 2.0f, 1.0f, 8.0f });

            REQUIRE(da == db);
        }

        {
            auto da = dot(vec4 { -1.0f, 2.0f, -9.0f, 4.0f },
                          vec4 { 25.0f, -2.0f, 0.0f, 8.0f });
            auto db = glm::dot(glm::vec4 { -1.0f, 2.0f, -9.0f, 4.0f },
                               glm::vec4 { 25.0f, -2.0f, 0.0f, 8.0f });

            REQUIRE(da == db);
        }
    }

    SUBCASE("Cross") {

        {
            auto da =
                cross(vec3 { 1.0f, 2.0f, 3.0f }, vec3 { 3.0f, 2.0f, 1.0f });
            auto db = glm::cross(glm::vec3(1.0f, 2.0f, 3.0f),
                                 glm::vec3(3.0f, 2.0f, 1.0f));

            REQUIRE(is_same(da, db));
        }

        {
            auto da =
                cross(vec3 { -1.0f, 2.0f, 3.0f }, vec3 { 3.0f, -5.0f, 1.0f });
            auto db = glm::cross(glm::vec3(-1.0f, 2.0f, 3.0f),
                                 glm::vec3(3.0f, -5.0f, 1.0f));

            REQUIRE(is_same(da, db));
        }
    }

    SUBCASE("Length") {
        auto la = length(vec4 { 1.0f, 2.0f, 3.0f, 4.0f });
        auto lb = glm::length(glm::vec4(1.0f, 2.0f, 3.0f, 4.0f));
        REQUIRE(la == lb);

        auto sla = length_squared(vec4 { 1.0f, 2.0f, 3.0f, 4.0f });
        auto slb = glm::length2(glm::vec4(1.0f, 2.0f, 3.0f, 4.0f));
        REQUIRE(sla == slb);


        auto da = distance(vec4 { 1.0f, 2.0f, 3.0f, 4.0f },
                           vec4 { 3.0f, 2.0f, 1.0f, 8.0f });
        auto db = glm::distance(glm::vec4(1.0f, 2.0f, 3.0f, 4.0f),
                                glm::vec4(3.0f, 2.0f, 1.0f, 8.0f));

        REQUIRE(da == db);

        auto da2 = distance_squared(vec4 { 1.0f, 2.0f, 3.0f, 4.0f },
                                    vec4 { 3.0f, 2.0f, 1.0f, 8.0f });
        auto db2 = glm::distance2(glm::vec4(1.0f, 2.0f, 3.0f, 4.0f),
                                  glm::vec4(3.0f, 2.0f, 1.0f, 8.0f));

        REQUIRE(da2 == db2);

        auto na = normalize(vec4 { 3.0f, 2.0f, 1.0f, 8.0f });
        auto nb = glm::normalize(glm::vec4(3.0f, 2.0f, 1.0f, 8.0f));

        REQUIRE(is_same(na, nb));
        REQUIRE(length(na) == 1);
    }

    SUBCASE("Other Operations") {
        auto da2 =
            reflect(vec3 { 1.0f, 2.0f, 3.0f }, vec3 { 3.0f, 2.0f, 1.0f });
        auto db2 = glm::reflect(glm::vec3(1.0f, 2.0f, 3.0f),
                                glm::vec3(3.0f, 2.0f, 1.0f));

        REQUIRE(is_same(da2, db2));
    }

    SUBCASE("Boolean") {
        REQUIRE(is_all(ivec3 { VEC_TRUE, VEC_TRUE, VEC_TRUE }));
        REQUIRE(!is_all(ivec3 { VEC_TRUE, false, VEC_TRUE }));

        REQUIRE(is_any(ivec3 { VEC_TRUE, VEC_TRUE, VEC_TRUE }));
        REQUIRE(is_any(ivec3 { VEC_TRUE, false, VEC_TRUE }));
        REQUIRE(!is_any(ivec3 { false, false, false }));

        REQUIRE(is_equal(vec4(1), vec4(1)));
        REQUIRE(is_equal(vec4(1), vec4(.95), .2f));
        REQUIRE(!is_equal(vec4(1), vec4(.95), .1f));


        {
            auto a = vec4 { 1, 2, 3, 4 };
            auto b = vec4 { 11, 12, 13, 14 };

            auto c = select(new_vec(true, false, true, false), a, b);

            REQUIRE(is(c, 11, 2, 13, 4));
        }
    }

    SUBCASE("Other") {

        REQUIRE(is(min(vec4 { 1.0f, 2.0f, 5.0f, -1.0f },
                       vec4 { 2.0f, 1.0f, 10.0f, -5.0f }),
                   1.0f,
                   1.0f,
                   5.0f,
                   -5.0f));

        REQUIRE(is(max(vec4 { 1.0f, 2.0f, 5.0f, -1.0f },
                       vec4 { 2.0f, 1.0f, 10.0f, -5.0f }),
                   2.0f,
                   2.0f,
                   10.0f,
                   -1.0f));

        REQUIRE(component_min(vec1(3.0f)) == 3.0f);
        REQUIRE(component_max(vec1(3.0f)) == 3.0f);

        REQUIRE(component_min(vec2 { 3.0f, 4.0f }) == 3.0f);
        REQUIRE(component_max(vec2 { 3.0f, 4.0f }) == 4.0f);

        REQUIRE(component_min(vec3 { -1.0f, 1.0f, 4.0f }) == -1.0f);
        REQUIRE(component_max(vec3 { -1.0f, 1.0f, 4.0f }) == 4.0f);

        REQUIRE(component_min(vec4 { 3.0f, -1.0f, 1.0f, 4.0f }) == -1.0f);
        REQUIRE(component_max(vec4 { 3.0f, -1.0f, 1.0f, 4.0f }) == 4.0f);


        REQUIRE(component_sum(vec4 { 1.0f, 2.0f, 5.0f, -1.0f }) == 7.0f);

        REQUIRE(
            is(abs(vec4 { 1.0f, -2.0f, 5.0f, -1.0f }), 1.0f, 2.0f, 5.0f, 1.0f));

        REQUIRE(is(floor(vec4 { 1.5f, -2.5f, 5.5f, -1.5f }),
                   1.0f,
                   -3.0f,
                   5.0f,
                   -2.0f));

        REQUIRE(is(
            ceil(vec4 { 1.5f, -2.5f, 5.5f, -1.5f }), 2.0f, -2.0f, 6.0f, -1.0f));
    }

    SUBCASE("Mix") {
        REQUIRE(mix(1.0f, 2.0f, true) == 2.0f);
        REQUIRE(mix(1.0f, 2.0f, 0.5f) == 1.5f);

        REQUIRE(is(mix(vec4 { 1.0f, -2.0f, 5.0f, -1.0f },
                       vec4 { 2.0f, -4.0f, 10.0f, -1.0f },
                       0.5f),
                   1.5f,
                   -3.0f,
                   7.5f,
                   -1.0f));

        REQUIRE(is(mix(vec4 { 1.0f, -2.0f, 5.0f, -1.0f },
                       vec4 { 2.0f, -4.0f, 10.0f, -1.0f },
                       vec4 { 0.5f, 0.0f, 1.0f, 0.5f }),
                   1.5f,
                   -2.0f,
                   10.0f,
                   -1.0f));
    }
}
