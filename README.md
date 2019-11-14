# DaciteLinalg
A header-only vector/matrix/quaternion library for Dacite.

Specifically intended for graphics related mathematics, transforms, etc.

## Requirements

Uses C++17. Only supports GCC or CLang. Also uses SSE4.1 instructions.


## Installation

Copy `include` into your source tree, or git submodule it into your own repo.

## Usage

All in the `dct` namespace.

Modules:

### Vector

`#include <vec.h>`

Provides the `Vector<class T, size_t N>` core vector class (where `T` is the
component type, and `N` is the component count), and some handy
typedefs of the format: *[B, I, D][M]Vec(C)*.

- The optional prefix indicates type, where *B* indicates bool, *I* for signed integer, *D* for double precision. No prefix is reserved for the common single precision floating point case.
- The optional second prefix *M* indicates component bit size.
- *C* indicates the number of components, 1 to 4.

`Vec4` is accellerated using CLang/GCC vector extensions.

Most GLSL operations are supported. Trig functions are accessed with:
`#include <vec_trig.h>`

### Matrix

`#include <mat.h>`

Provides the `Matrix<class T, size_t C, size_t R>` core matrix class
(where `T` is the cell type, `C` is the column count, `R` is the row count).

Typedefs include

- `Mat2`, a floating point 2x2 matrix
- `Mat3`, a floating point 3x3 matrix
- `Mat4`, a floating point 4x4 matrix

Common GLSL-like matrix operations, including inversion, are accessed using:
`#include <mat_operations.h>`

Common matrix transformation operations are accessed using:
`#include <mat_transforms.h>`


### Quaternion

`#include <quat.h>`

Provides the `Quaternion<class T>` core quaternion class (where `T` is the
component type).

Typedefs include

- `Quat`, a floating point quaternion
- `DQuat`, a double precision floating point quaternion


### Transformation Matrix

`#include <tmatrix.h>`

Provides `TMatrix`, an abstraction for `Mat4`, with the objective to help
construct and work with transformation matrices. For example, the default
constructor initializes to the identity. Also includes perspective matrix
builders.
