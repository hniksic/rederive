"""Vectors, Derive's one compound value, and their shapes.

In Derive a vector is a sequence of any values, including other vectors. A matrix is
nothing more than a vector whose elements are vectors of one length, its rows, and a
tensor of rank three is a vector of matrices. So `[1, 2]` and `[[1, 2]]` differ (a
vector of two numbers and a matrix of one row), `DIMENSION` counts elements at the top
level only, `APPEND` concatenates elements, whatever they are, and elements may be
equations, truth values, ragged vectors or calls still waiting for an argument.

All of these are one head, `Vector(*elements)`, nested for every level. Explicit vectors
and matrices are never sympy matrices, for three reasons:

* A sympy matrix has exactly two dimensions. A flat vector would have to be one row,
  making it the same object as a one-row matrix, and a rank-three tensor has no
  representation at all.
* Its elements must be expressions, so `[x = 1, y = 2]` can't be one.
* It must be rectangular, so ragged vectors would need a second head, and every
  function would have to handle both. Which one a value got would depend on what it
  happens to hold at the time: `[LIM([x, 1], x, 2), [3, 4]]` would become a matrix
  only once the limit was taken.

With one head, a value's shape is its nesting and nothing else. Linear algebra, which
needs a rectangular matrix of expressions, converts at the boundary: `matrix_of` gives
a sympy matrix for a vector of equal-length rows of expressions, and `from_matrix` turns
the answer back into a vector of rows. Elements are reached as `args`, and `mapped`
rebuilds a vector with a function applied to each.

`Vector` is a `Function`, hence an expression, so a vector can stand wherever sympy
expects one: a side of an equation, an argument of a call that stays unevaluated
(`ROW_REDUCE` of something not yet a matrix, or `SIN([1, 2])`, which Derive leaves
alone), or a term of a sum without a value, like `[1, 2] + 2`. Sympy never does vector
arithmetic, though: to it a vector is an opaque scalar, so `Mul(v, w)` of two vectors
means nothing (and may even reorder them). Arithmetic on vectors is the converter's
(`to_sympy.added`, `multiplied` and `dotted`), and a value put into an expression by
substitution must have that arithmetic redone (`to_sympy.settled`).

Declared nonscalars are a different thing: symbolic matrices of unknown size, which
sympy's matrix expressions (`MatrixSymbol`, `MatMul`, ...) represent. An explicit matrix
meets them as a sympy matrix (`matrix_of`), the one place such a matrix appears in a
converted value; the printer writes it as nested vectors.
"""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp

__all__ = [
    "Vector",
    "column",
    "from_matrix",
    "is_matrix",
    "is_vector",
    "mapped",
    "matrix_of",
    "transposed",
]


class Vector(sp.Function):
    """`[u, v, ...]`: a vector of any values, a matrix being a vector of rows.

    See the module docstring. A vector is not a number, even if all its elements are,
    so nothing that evaluates numbers (approximation, `evalf`) treats it as one; they
    reach its elements instead.
    """

    is_number = False

    def _eval_derivative(self, symbol: sp.Symbol) -> sp.Basic:
        return Vector(*(sp.diff(element, symbol) for element in self.args))


def is_vector(value: object) -> bool:
    """Whether `value` is an explicit vector, which includes matrices."""
    return isinstance(value, Vector)


def is_matrix(value: object) -> bool:
    """Whether `value` is a vector of one or more rows, all vectors of one length.

    The elements of the rows can be anything, including vectors, so a rank-three tensor
    is a matrix here too, whose elements are vectors.
    """
    if not isinstance(value, Vector) or not value.args:
        return False
    if not all(isinstance(row, Vector) for row in value.args):
        return False
    return len({len(row.args) for row in value.args}) == 1


def matrix_of(value: object) -> sp.ImmutableMatrix | None:
    """`value` as a sympy matrix for linear algebra, or None if it isn't one.

    That takes a vector of rows of one nonzero length whose elements are expressions,
    not vectors or relations. A flat vector is not a matrix: callers that treat one as
    a row or a column say which.
    """
    if not is_matrix(value) or not value.args[0].args:
        return None
    rows = [list(row.args) for row in value.args]
    if not all(_scalar(element) for row in rows for element in row):
        return None
    return sp.ImmutableMatrix(rows)


def _scalar(value: sp.Basic) -> bool:
    return isinstance(value, sp.Expr) and not isinstance(value, (Vector, sp.MatrixExpr))


def from_matrix(matrix: sp.MatrixBase) -> Vector:
    """A sympy matrix as a vector of rows, whatever its shape."""
    return Vector(*(Vector(*matrix.row(row)) for row in range(matrix.rows)))


def transposed(value: Vector) -> Vector | None:
    """The transpose of a vector, or None if it has none.

    A matrix's rows become its columns, whatever its elements are. A flat vector, whose
    elements are not vectors, is its own transpose, as in Derive: `[a, b]`` is
    `[a, b]`, while the one-row matrix `[[a, b]]` transposes to the column
    `[[a], [b]]`. A vector mixing vectors and other elements, or ragged rows, has no
    transpose.
    """
    if is_matrix(value):
        columns = zip(*(row.args for row in value.args), strict=True)
        return Vector(*(Vector(*column) for column in columns))
    if any(isinstance(element, Vector) for element in value.args):
        return None
    return value


def column(value: Vector) -> Vector:
    """A flat vector as a one-column matrix: `[a, b]` as `[[a], [b]]`."""
    return mapped(value, Vector)


def mapped(value: Vector, function: Callable[[sp.Basic], sp.Basic]) -> Vector:
    """`value` with `function` applied to each element."""
    return Vector(*(function(element) for element in value.args))
