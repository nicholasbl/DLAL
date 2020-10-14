#include "doctest.h"
#include "transformation.h"

#include "test_common.h"

#include <glm/gtx/norm.hpp>

using namespace dlal;

TEST_CASE("TMatrix Library") {

    SUBCASE("Basics") {

        Transformation a;
        auto           ab = mat4(a);

        REQUIRE(is_same(
            ab, mat4({ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 })));
    }
}
