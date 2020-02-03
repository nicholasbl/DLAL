# DaciteLinalg
A header-only vector/matrix/quaternion library for Dacite.

Specifically intended for graphics related mathematics, transforms, etc.

## Requirements

Uses C++17. Only supports CLang, and uses SSE4.1 instructions.


## Installation

Copy the contents of `include` into your source tree, or git submodule it into
your own repo. If you wish a single header, there is a `single_include`
directory containing a single header that can be copied into your source
directory.

## Usage

All types are in the `dct` namespace.

Modules:

### Vector

`vec.h` provides the `vec<class T, int N>` core SIMD vector class (where `T` is the
component type, and `N` is the component count), and some handy
typedefs of the format.: *[i, d][M]vec(C)*.

- The optional prefix indicates type, where *i* for signed integer, *d* for double precision. No prefix is reserved for the common single precision floating point case.
- The optional second prefix *M* indicates component bit size.
- *C* indicates the number of components, 1 to 4.

This type does not provide size guarantees; it is intended to provide as much
performance as possible, thus some vectors may have odd sizes.

Most GLSL operations are supported. Trig functions are accessed with:
`vec_trig.h`

`packed_vec.h` provides size-stable versions of the same vectors, and supports
more types and sizes. It, however, does not have any mathematics support; it is
intended to be used only as a storage format.

### Matrix

`mat.h` provides the `mat<class T, size_t C, size_t R>` core matrix class
(where `T` is the cell type, `C` is the column count, `R` is the row count).

Typedefs include

- `mat2`, a floating point 2x2 matrix
- `mat3`, a floating point 3x3 matrix
- `mat4`, a floating point 4x4 matrix

Common GLSL-like matrix operations, including inversion, are accessed using:
`<mat_operations.h`

Common matrix transformation operations are accessed using:
`mat_transforms.h`

As with vectors, this type has no size guarantees. Use the `packed_mat` from
the `packed_mat.h` header for predictably sized matrices.


### Quaternion

`#include <quat.h>`

Provides the `quaternion<class T>` core quaternion class (where `T` is the
component type).

Typedefs include

- `quat`, a floating point quaternion
- `dquat`, a double precision floating point quaternion


### Transformation Matrix

`tmatrix.h` provides `TMatrix`, an abstraction for `mat4`, with the objective
to help construct and work with transformation matrices. For example, the
default constructor initializes to the identity. Also includes perspective
matrix builders.

## Todo

Finish accellerating the common typedefs in `vec` and `mat`.
