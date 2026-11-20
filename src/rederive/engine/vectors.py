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

    def _get_args(self) -> tuple[sp.Basic, ...]:
        state = self.__dict__
        made = state.get("_made")
        if made is None:
            made = _laid_out(state["_run"], state["_low"], state["_high"])
            state["_made"] = made
        return made

    def _set_args(self, args: tuple[sp.Basic, ...]) -> None:
        self.__dict__["_made"] = args

    #: The elements, which a vector made by `adjoined`, `extended`, `prefixed` or
    #: `sliced` lays out from the run it shares only when they are asked for.
    _args = property(_get_args, _set_args)

    def __new__(cls, *args: sp.Basic, **options: object) -> Vector:
        """The vector of `args`, built directly when they are all expressions already:
        a vector has nothing to evaluate, and a program that builds a list element by
        element would otherwise spend most of its time in sympy's generic construction,
        which converts, caches and inspects every element anew at each step."""
        if options or not all(isinstance(arg, sp.Basic) for arg in args):
            return super().__new__(cls, *args, **options)
        return cls.of(args)

    @classmethod
    def of(cls, args: tuple[sp.Basic, ...]) -> Vector:
        """The vector of `args`, which must all be sympy objects, built without looking
        at them: list functions make long vectors from the elements of others."""
        vector = sp.Basic.__new__(cls, *args)
        vector.nargs = sp.S.Naturals0
        return vector

    def _eval_derivative(self, symbol: sp.Symbol) -> sp.Basic:
        return Vector(*(sp.diff(element, symbol) for element in self.args))

    # -- the list functions' vectors, which share their elements ---------------
    #
    # A program builds a list an element at a time, `v := ADJOIN(u, v)` or `v :=
    # APPEND(v, [u])`, and consumes one so, `v := REST(v)`; each step making a new
    # vector of all the elements would make the program quadratic in the length of the
    # list. So these vectors are views: positions `low` to `high` of a `_Run`, which
    # vectors made from one another share. A run only grows outward, at the ends no
    # view has claimed yet, so a view's elements never change; a vector whose end has
    # been claimed by another is copied into a run of its own. A view's elements are
    # laid out as a tuple the first time anything else asks for its `args`.

    def _placed(self) -> tuple[_Run, int, int]:
        """The run this vector's elements are in, and their positions there; a vector
        made otherwise gets a run of its own."""
        state = self.__dict__
        run = state.get("_run")
        if run is None:
            items = state["_made"]
            run = _Run([], items)
            state["_run"], state["_low"], state["_high"] = run, 0, len(items)
            return run, 0, len(items)
        return run, state["_low"], state["_high"]

    def size(self) -> int:
        """The number of elements, `len(self.args)`, without laying them out."""
        state = self.__dict__
        made = state.get("_made")
        if made is not None:
            return len(made)
        return state["_high"] - state["_low"]

    def item(self, place: int) -> sp.Basic:
        """The element at the zero-based `place`, which must be one of the vector's,
        `self.args[place]` without laying them out."""
        state = self.__dict__
        made = state.get("_made")
        if made is not None:
            return made[place]
        position = state["_low"] + place
        run = state["_run"]
        return run.front[-1 - position] if position < 0 else run.back[position]

    def adjoined(self, value: sp.Basic) -> Vector:
        """The vector of `value` followed by this vector's elements."""
        return self.prefixed((value,))

    def prefixed(self, values: tuple[sp.Basic, ...]) -> Vector:
        """The vector of `values` followed by this vector's elements."""
        run, low, high = self._placed()
        if low == -len(run.front):
            run.front.extend(reversed(values))
            return _view(type(self), run, low - len(values), high)
        items = self.args
        run = _Run(list(reversed(values)), list(items))
        return _view(type(self), run, -len(values), len(items))

    def extended(self, values: tuple[sp.Basic, ...]) -> Vector:
        """The vector of this vector's elements followed by `values`."""
        run, low, high = self._placed()
        back = run.back
        if high == len(back):
            if type(back) is not list:
                back = run.back = list(back)
            back.extend(values)
            return _view(type(self), run, low, high + len(values))
        items = [*self.args, *values]
        return _view(type(self), _Run([], items), 0, len(items))

    def sliced(self, start: int, stop: int) -> Vector:
        """The vector of the elements at zero-based places `start` to `stop`, which
        must be within the vector's, `self.args[start:stop]` without copying them."""
        run, low, _ = self._placed()
        return _view(type(self), run, low + start, low + max(stop, start))


class _Run:
    """The elements of vectors made from one another by the list functions (see
    `Vector.sliced`), at positions extending both ways from zero: `back[p]` is at
    position `p` and `front[-1 - p]` at a negative `p`. `back` may be a tuple, the
    elements of a vector made otherwise, until something extends it."""

    __slots__ = ("back", "front")

    def __init__(self, front: list, back: list | tuple):
        self.front = front
        self.back = back


def _view(cls: type[Vector], run: _Run, low: int, high: int) -> Vector:
    """The vector of the elements at positions `low` to `high` of `run`."""
    vector = object.__new__(cls)
    vector._assumptions = cls.default_assumptions
    vector._mhash = None
    vector.nargs = sp.S.Naturals0
    state = vector.__dict__
    state["_run"], state["_low"], state["_high"] = run, low, high
    return vector


def _laid_out(run: _Run, low: int, high: int) -> tuple[sp.Basic, ...]:
    """The elements at positions `low` to `high` of `run`, in order."""
    back = run.back
    if low >= 0:
        return tuple(back[low:high])
    front = run.front
    before = front[-min(high, 0) : -low]
    before.reverse()
    return (*before, *back[0 : max(high, 0)])


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
    `[[a], [b]]`. Ragged rows transpose as Derive does it: the j-th row of the result
    holds the j-th elements of the rows that have one, so `[[a, b], [c]]`` is
    `[[a, c], [b]]`. A vector mixing vectors and other elements has no transpose.
    """
    if is_matrix(value):
        columns = zip(*(row.args for row in value.args), strict=True)
        return Vector(*(Vector(*column) for column in columns))
    if value.args and all(isinstance(element, Vector) for element in value.args):
        rows = [row.args for row in value.args]
        return Vector(
            *(
                Vector(*(row[j] for row in rows if j < len(row)))
                for j in range(max(len(row) for row in rows))
            )
        )
    if any(isinstance(element, Vector) for element in value.args):
        return None
    return value


def column(value: Vector) -> Vector:
    """A flat vector as a one-column matrix: `[a, b]` as `[[a], [b]]`."""
    return mapped(value, Vector)


def mapped(value: Vector, function: Callable[[sp.Basic], sp.Basic]) -> Vector:
    """`value` with `function` applied to each element."""
    return Vector(*(function(element) for element in value.args))
