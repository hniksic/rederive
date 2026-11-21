"""Native values: the numbers, truth values and vectors a running program computes with,
held as Python objects rather than sympy ones.

A program that counts, sieves or searches spends its time on integer arithmetic,
comparisons and element reads and writes, and a sympy `Integer` costs many times what a
Python `int` does for each of them. So the evaluator computes such values natively
where a program runs (see "compiled nodes" in `evaluate`):

* an integer is an `int`, a rational number that isn't one a `Fraction`, and a truth
  value a `bool`;
* a vector a program writes elements of is a `NativeVector`, whose elements are any
  values, native or sympy, and which is written in place;
* the assignment of a global, whose value is the assignment rather than the value
  assigned, is a `NativeAssignment`, or the `Assigned` cell it gives the global, its
  `Assign` built only where it is read.

A variable's cell keeps its value in this form, a local's and a global's alike (see
`Cell` in `context`), and only the evaluator reads it so: everything else reads a
global's value through `given`, as a sympy value.

Each native value stands for exactly one sympy value, `sympy_of`, and a sympy value
that has a native form has exactly one, `native_of`; the two are inverse. Everything
that isn't native is a sympy value, so a value the evaluator holds is either.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import sympy as sp

from rederive.engine.to_sympy import SETTLED_ADDED, Assign, settled_since
from rederive.engine.vectors import Vector

from rederive.engine.context import Global

if TYPE_CHECKING:
    from rederive.engine.context import Store

__all__ = [
    "NUMBERS",
    "Assigned",
    "SCALARS",
    "NativeAssignment",
    "NativeVector",
    "assigned",
    "given",
    "native_of",
    "rational",
    "sympy_of",
]


class NativeVector:
    """A vector written in place: `v SUB i := u` replaces one element instead of
    building a new vector of all of them.

    A native vector has exactly one holder: the cell of the variable whose value it is
    (a local's or a global's), or the native vector it is an element of. Anything else
    that keeps the value (another variable, an element of another vector, the last value
    of a frame, a returned value, any sympy expression) keeps `vector()`, the sympy
    vector of its elements as they are now, so that a later write changes nothing it
    holds. `vector()` is remembered until the next write, which forgets it in every
    native vector the write goes through.

    No element is an assignment, a definition or a command's answer, which using a
    vector as a value would unwrap: so its elements are those of `vector()` as an
    operand. `symbolic` counts the elements that are not native numbers or truth
    values; while it is zero, no substitution changes the vector.

    `settled` is None, or for a vector made of one a program stored as it is under some
    settings (see `SETTLED` in to_sympy.py), those settings and the values written into
    it since, which `vector()` passes on (see `settled_since`).
    """

    __slots__ = ("items", "made", "settled", "symbolic")

    def __init__(self, items: list, settled: tuple[object, list] | None = None) -> None:
        self.items = items
        self.made: Vector | None = None
        self.symbolic = sum(1 for item in items if type(item) not in SCALARS)
        self.settled = settled

    def vector(self) -> Vector:
        made = self.made
        if made is None:
            made = self.made = Vector.of(tuple(sympy_of(item) for item in self.items))
            settled = self.settled
            if settled is not None:
                key, written = settled
                settled_since(made, key, [sympy_of(item) for item in written])
        return made

    def put(self, place: int, value: object) -> None:
        """Make `value` the element at the zero-based `place`, which it must have."""
        items = self.items
        self.symbolic += (type(value) not in SCALARS) - (
            type(items[place]) not in SCALARS
        )
        items[place] = value
        self.made = None
        self.wrote(value)

    def wrote(self, value: object) -> None:
        """Record that `value` is now an element, for `settled`."""
        settled = self.settled
        if settled is not None and type(value) not in SCALARS:
            if len(settled[1]) < SETTLED_ADDED:
                settled[1].append(value)
            else:
                self.settled = None


class NativeAssignment:
    """The assignment a program made of a global, `x := u` or `x SUB i := u`, whose
    value is the assignment rather than the value assigned: what is needed to build
    its `Assign`, which is built only where it is read.

    The value of a statement mostly goes unread, and building an `Assign` costs more
    than the rest of an assignment does. So this keeps the `evaluator` that made it
    and the assignment's tree `node`, the native indices `places` of an element
    assignment (None for a variable's), the `value` given (to the element, for one),
    and the declarations `domains` in force (`Store.domains`), which the target's
    symbol depends on and a later declaration replaces.
    """

    __slots__ = ("domains", "evaluator", "node", "places", "value")

    def __init__(
        self,
        evaluator: Any,
        node: Any,
        places: list | None,
        value: object,
        domains: Mapping[str, Any],
    ) -> None:
        self.evaluator = evaluator
        self.node = node
        self.places = places
        self.value = value
        self.domains = domains

    def assignment(self) -> Assign:
        return self.evaluator.assignment_of(self)


class Assigned(Global):
    """The cell a compiled program's assignment `x := u` gives the global `x`, which
    is also the assignment's value, as a `NativeAssignment` would be (with no
    `places`): a program assigning a global in a loop makes one object a pass, not
    two. So it is never changed: an element assignment changes a global's cell in
    place only where it holds a native vector, which none of these does."""

    __slots__ = ("domains", "evaluator", "node")

    places = None

    def __init__(
        self,
        raw: object,
        at: int,
        evaluator: Any,
        node: Any,
        domains: Mapping[str, Any],
    ) -> None:
        self.raw = raw
        self.at = at
        self.evaluator = evaluator
        self.node = node
        self.domains = domains

    @property
    def value(self) -> object:
        return self.raw

    def assignment(self) -> Assign:
        return self.evaluator.assignment_of(self)


#: The types of native numbers.
NUMBERS = frozenset({int, Fraction})

#: The types of native values other than vectors: numbers and truth values.
SCALARS = frozenset({int, Fraction, bool})


def rational(value: Fraction) -> int | Fraction:
    """`value` as a native number: an integer as an `int`."""
    return value.numerator if value.denominator == 1 else value


def sympy_of(value: object) -> object:
    """The sympy value a native value stands for; any other value as it is."""
    kind = type(value)
    if kind is int:
        return sp.Integer(value)
    if kind is bool:
        return sp.true if value else sp.false
    if kind is Fraction:
        return sp.Rational(value.numerator, value.denominator)
    if kind is NativeVector:
        return value.vector()
    if kind is NativeAssignment or kind is Assigned:
        return value.assignment()
    return value


def given(store: Store, name: str) -> object:
    """The value the command gave the global `name` so far, as a sympy value, `CLEARED`
    if it took its value away, or None if it gave it none (see `Store.cells`)."""
    cell = store.cells.get(name)
    if cell is None or not cell.at:
        return None
    return sympy_of(cell.raw)


def assigned(store: Store, name: str) -> bool:
    """Whether the command gave the global `name` a value so far, or took it away."""
    cell = store.cells.get(name)
    return cell is not None and cell.at > 0


def native_of(value: object) -> object:
    """The native form of a sympy number or truth value; any other value as it is.
    A vector stays a sympy one: it is made native only to be written into. So does an
    assignment: it is native only as a program makes it."""
    convert = _NATIVE.get(type(value))
    return value if convert is None else convert(value)


def _integer(value: sp.Integer) -> int:
    return value.p


def _fraction(value: sp.Rational) -> Fraction:
    return Fraction(value.p, value.q)


_NATIVE = {
    sp.Integer: _integer,
    type(sp.S.Zero): _integer,
    type(sp.S.One): _integer,
    type(sp.S.NegativeOne): _integer,
    sp.Rational: _fraction,
    type(sp.S.Half): _fraction,
    type(sp.true): lambda value: True,
    type(sp.false): lambda value: False,
}
