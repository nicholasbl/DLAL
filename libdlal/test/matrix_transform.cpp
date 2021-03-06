#include "doctest.h"
#include "mat_transforms.h"

#include "test_common.h"

#include <glm/gtx/norm.hpp>

using namespace dlal;

TEST_CASE("Matrix Transform Library") {

    SUBCASE("Translate") {
        mat4      a({ 1, 2, 3, 4, 0, 1, 0, 2, 0, 0, 1, 1, 0, 0, 0, 1 });
        glm::mat4 b(1, 2, 3, 4, 0, 1, 0, 2, 0, 0, 1, 1, 0, 0, 0, 1);

        a = translate(a, vec3 { 1, 3, 4 });
        b = glm::translate(b, glm::vec3(1, 3, 4));

        REQUIRE(is_same(a, b));
    }

    SUBCASE("Translate In Place") {
        mat4      a({ 1, 2, 3, 4, 0, 1, 0, 2, 0, 0, 1, 1, 0, 0, 0, 1 });
        glm::mat4 b(1, 2, 3, 4, 0, 1, 0, 2, 0, 0, 1, 1, 0, 0, 0, 1);


        translate_in_place(a, vec3 { 1, 3, 4 });
        b = glm::translate(b, glm::vec3(1, 3, 4));

        REQUIRE(is_same(a, b));
    }

    SUBCASE("Rotate") {
        mat4      a({ 1, 2, 3, 4, 0, 1, 0, 2, 0, 0, 1, 1, 0, 0, 0, 1 });
        glm::mat4 b(1, 2, 3, 4, 0, 1, 0, 2, 0, 0, 1, 1, 0, 0, 0, 1);

        a = rotate(a, .5f, vec3 { 1, 3, 4 });
        b = glm::rotate(b, .5f, glm::vec3(1, 3, 4));

        REQUIRE(is_same(a, b));
    }

    SUBCASE("Scale") {
        mat4      a({ 1, 2, 3, 4, 0, 1, 0, 2, 0, 0, 1, 1, 0, 0, 0, 1 });
        glm::mat4 b(1, 2, 3, 4, 0, 1, 0, 2, 0, 0, 1, 1, 0, 0, 0, 1);

        a = scale(a, vec3 { 1, 3, 4 });
        b = glm::scale(b, glm::vec3(1, 3, 4));

        REQUIRE(is_same(a, b));
    }

    SUBCASE("Scale In Place") {
        mat4      a({ 1, 2, 3, 4, 0, 1, 0, 2, 0, 0, 1, 1, 0, 0, 0, 1 });
        glm::mat4 b(1, 2, 3, 4, 0, 1, 0, 2, 0, 0, 1, 1, 0, 0, 0, 1);


        scale_in_place(a, vec3 { 1, 3, 4 });
        b = glm::scale(b, glm::vec3(1, 3, 4));

        REQUIRE(is_same(a, b));
    }
}
