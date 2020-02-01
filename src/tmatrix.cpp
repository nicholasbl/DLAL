#include "include/tmatrix.h"
#include "doctest.h"

#include "test_common.h"

#include <glm/gtx/norm.hpp>

using namespace dct;

TEST_CASE("TMatrix Library") {

    SUBCASE("Basics") {

        TMatrix a;
        auto    ab = mat4(a);

        REQUIRE(is_same(
            ab, mat4({ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 })));
    }
}
