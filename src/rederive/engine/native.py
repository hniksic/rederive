"""Native values: the numbers, truth values and vectors a running program computes with,
held as Python objects rather than sympy ones.

A program that counts, sieves or searches spends its time on integer arithmetic,
comparisons and element reads and writes, and a sympy `Integer` costs many times what a
Python `int` does for each of them. So the evaluator computes such values natively
where a program runs (see "compiled nodes" in `evaluate`):

* an integer is an `int`, a rational number that isn't one a `Fraction`, and a truth
  value a `bool`;
* a vector a program writes elements of, or that its locals share, is a
  `NativeVector`, whose elements are any values, native or sympy, and which is written
  in place;
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

from rederive.engine.to_sympy import Assign
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


#: How many writes native vectors have taken, which a remembered `vector()` was made
#: after: in a list, since changing an attribute of the class would slow every read of
#: one of its instances' attributes.
_WRITES = [0]


class NativeVector:
    """A vector written in place: `v SUB i := u` replaces one element instead of
    building a new vector of all of them. It is also the vector that variables share.

    These are Derive 6's sharing rules, established from its documentation and
    worksheets; they are deliberate, not incidental. While a program runs:

    1. Locals hold references. A local is a parameter of the running function; `u :=
       t`, `[t, t]` and `A SUB i := t` with `t` a local make one vector that each holder
       sees written. The Derive 6.1 help documents this (ProceduralProgramming): "if a
       vector or matrix is assigned to different local variables, assignments to the
       elements of the vector or matrix affect the value of all the local variables."
    2. A global holds a value: reading one gives a copy, storing into one copies, and
       an element assignment to one changes its stored value. Variables of a program
       run at the prompt are globals. Derive User Group worksheets show this (DNL 52
       fft0_v6.dfw, DNL 48 matrixrepl.dfw).
    3. A vector written out is a new vector each time it is evaluated, so a loop that
       starts a pass with `c := [0, 0]` starts it from zeros (DNL 122 bolyai.dfw).

    Where the sources are silent, these are the simplest rules consistent with the
    ones above:

    4. A parameter given a local of the caller holds the caller's vector, and a
       function returns the vector its body ends with, as `IF`, `PROG`, `LOOP` and
       `RETURN` pass on theirs.
    5. An element of a local's vector, `A SUB i`, taken by a local is that vector's
       element itself: writing either changes both.
    6. A built-in's result is a new vector that shares nothing with its arguments.
       This departs from Derive on purpose. Derive is built on linked lists, so the
       results of `REST` and `ADJOIN` presumably share structure with their argument,
       and an element write through one can show through the other: that is inferred
       from its list representation, not observed. Rederive keeps a vector as an array,
       for indexing in constant time, with Python's and Lisp's reference sharing on
       top, and does not follow it: `REST` and `ADJOIN` copy.
    7. Operands are evaluated left to right on the vectors as they are: in
       `APPEND(t, PROG(t SUB 1 := 9, [0]))`, `APPEND` sees the 9.

    A local's vector, and each vector in it, is made native once read, so that what
    takes it shares it and an operand sees it written; until then the local holds the
    sympy vector, which nothing can change. A global's vector and the vectors in it
    are `kept`: the global holds them alone, and a read gives `vector()`. Wherever a
    value leaves the evaluator, it is the sympy vector of its elements as they are
    then.

    No element is an assignment, a definition or a command's answer, which using a
    vector as a value would unwrap: so its elements are those of `vector()` as an
    operand. `symbolic` counts the elements that are not native numbers or truth
    values, or is -1 until they are made native; while it is zero, no substitution
    changes the vector (see `fixed`). `vector()` is remembered until a native vector is
    next written, since this one may hold it; that of a vector made from a sympy one is
    that one until its elements are needed.
    """

    __slots__ = ("items", "kept", "made", "stamp", "symbolic")

    def __init__(
        self, items: list | None, made: Vector | None = None, kept: bool = False
    ) -> None:
        self.items = items
        self.kept = kept
        self.made = made
        self.stamp = _WRITES[0]
        self.symbolic = (
            -1 if items is None else sum(1 for item in items if type(item) not in SCALARS)
        )

    @classmethod
    def of(cls, vector: Vector, kept: bool = False) -> NativeVector:
        """The sympy vector `vector`, which holds no assignment, definition or command's
        answer, as a native one. Its elements are made native only once they are
        needed: until then `items` is None, and `symbolic` -1."""
        return cls(None, vector, kept)

    def unpacked(self) -> list:
        """The elements, `items`, made native where they still aren't."""
        items = self.items
        if items is None:
            assert self.made is not None
            items = self.items = [native_of(element) for element in self.made.args]
            self.symbolic = sum(1 for item in items if type(item) not in SCALARS)
        return items

    def fixed(self) -> bool:
        """Whether no substitution changes the vector: `symbolic` is zero."""
        if self.symbolic < 0:
            self.unpacked()
        return not self.symbolic

    def vector(self) -> Vector:
        made = self.made
        if self.items is None:
            return made
        if made is None or self.stamp != _WRITES[0]:
            made = self.made = Vector.of(tuple(sympy_of(item) for item in self.items))
            self.stamp = _WRITES[0]
        return made

    def put(self, place: int, value: object) -> None:
        """Make `value` the element at the zero-based `place`, which it must have."""
        items = self.items
        if items is None:
            items = self.unpacked()
        self.symbolic += (type(value) not in SCALARS) - (
            type(items[place]) not in SCALARS
        )
        items[place] = value
        _WRITES[0] += 1

    def fill(self, vector: Vector) -> None:
        """Make the elements those of the sympy vector `vector`, as `of` takes them."""
        self.items = None
        self.made = vector
        self.symbolic = -1
        _WRITES[0] += 1


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
