"""Finite sets, written with braces: `{1, 2, 3}`.

A set is one head, `SetOf(*elements)`, which like a `Vector` is opaque to sympy: the
arithmetic of sets (a scalar times a set, the Cartesian product `{a, b}*{c, d}`) is the
converter's. A set keeps each element once, in a canonical order, so two sets written
with the same elements are the same object: `{a, b, b, c}` and `{c, b, a}` are both
`{a, b, c}`. The order is numbers first, ascending, then everything else in sympy's
canonical order.

Elements are expressions, and two elements are one only when they are the same
expression once simplified: `{x + x, 2*x}` has one element, while `{a, b}` has two and
`{a, b} INTERSECTION {c}` is `{}`, as in Derive. A variable with a value is compared by
its value. An operation on an operand that isn't an explicit set, such as an undeclared
`s` or an ellipsis `{1, ..., n}` that can't be counted, stands as written.

The operations are heads that evaluate on construction (`SetUnion`, `SetIntersection`,
`SetDifference` and `SetComplement`), so substituting a value into one that stood
carries it out. The complement ``s` `` of a set has no universe to be taken in, so it
only stands, but it obeys the algebra: ``s`` `` is `s`, `s INTERSECTION s`` is `{}`,
`s UNION s`` is ``{}` ``, everything, and De Morgan's laws take the complement of a
union or intersection inside.

A variable declared `:epsilon Set` is a `SetSymbol`: a set with unknown elements, which
the operations act on as on any set, so `S UNION S` is `S`.

`ElidedVector` and `ElidedSet` are the ellipsis forms `[a, ..., b]` and `{a, b, ..., c}`
whose terms can't be counted yet, such as `[1, ..., n]`; they become the vector or set
once they can.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from fractions import Fraction

import sympy as sp

from rederive.engine.vectors import Vector

__all__ = [
    "ElidedSet",
    "ElidedVector",
    "SetComplement",
    "SetDifference",
    "SetIntersection",
    "SetOf",
    "SetSymbol",
    "SetUnion",
    "cartesian",
    "elided",
    "equal_sets",
    "is_elided",
    "is_set",
    "power_set",
    "size",
]

#: The most elements an ellipsis or a Cartesian power is expanded to, and the most
#: elements of a set whose power set is built. Beyond these the expression stands as
#: written.
MOST_TERMS = 100_000
MOST_SUBSETS_OF = 16


class SetOf(sp.Function):
    """`{u, v, ...}`: a finite set, its elements once each, in canonical order."""

    is_number = False

    @classmethod
    def eval(cls, *args: sp.Basic) -> sp.Basic | None:
        canonical = _canonical(args)
        if canonical == args:
            return None
        return cls(*canonical, evaluate=False)

    def _eval_derivative(self, symbol: sp.Symbol) -> sp.Basic:
        return SetOf(*(sp.diff(element, symbol) for element in self.args))


def is_set(value: object) -> bool:
    """Whether `value` is a set: a literal or an operation on sets that stands."""
    return isinstance(value, _SET_HEADS)


def _canonical(elements: Iterable[sp.Basic]) -> tuple[sp.Basic, ...]:
    return tuple(sorted(dict.fromkeys(elements), key=_order))


def _order(element: sp.Basic) -> tuple:
    if isinstance(element, sp.Rational):
        return (0, Fraction(int(element.p), int(element.q)))
    return (1, sp.default_sort_key(element))


# -- equality ----------------------------------------------------------------------


def equal_sets(left: sp.Basic, right: sp.Basic) -> bool | None:
    """Whether two sets are equal; None if that isn't known.

    Two set expressions are equal when they are the same: `(a INTERSECTION b)` = a`
    UNION b`` is `true` by De Morgan's law. Two set literals, their elements simplified,
    are equal exactly when they have the same elements, compared as expressions (see the
    module docstring): `{a, b} = {b, a}` is `true`, while `{a, b} = {a, c}` and `{x} =
    {1}` are `false`. A set that isn't a literal, such as a variable or a complement,
    is equal to another set only where it is the same one."""
    if not (is_set(left) and is_set(right)):
        return None
    if left == right:
        return True
    if not (isinstance(left, SetOf) and isinstance(right, SetOf)):
        return None
    return False


def size(collection: SetOf) -> sp.Basic:
    """The number of elements."""
    return sp.Integer(len(collection.args))


# -- the operations --------------------------------------------------------------


class SetSymbol(sp.Symbol):
    """A variable declared `:epsilon Set`."""

    is_number = False


class SetComplement(sp.Function):
    """``s` ``, the complement of a set: `(s UNION t)` `` is ``s` INTERSECTION t` `` for
    sets `s` and `t`, and for variables, which an operand of a set operation is (see
    `Converter._set_operation`)."""

    nargs = 1
    is_number = False

    @classmethod
    def eval(cls, operand: sp.Basic) -> sp.Basic | None:
        if isinstance(operand, SetComplement):
            return operand.args[0]
        if not all(
            is_set(each) or isinstance(each, sp.Symbol) for each in operand.args
        ):
            return None
        if isinstance(operand, SetUnion):
            return SetIntersection(*(SetComplement(each) for each in operand.args))
        if isinstance(operand, SetIntersection):
            return SetUnion(*(SetComplement(each) for each in operand.args))
        return None


class _Associative(sp.Function):
    """A union or intersection of any number of operands, flattened."""

    is_number = False

    @classmethod
    def eval(cls, *args: sp.Basic) -> sp.Basic | None:
        parts = (
            argument.args if isinstance(argument, cls) else (argument,)
            for argument in args
        )
        operands = list(dict.fromkeys(part for some in parts for part in some))
        result = cls._combined(operands)
        if result is None:
            if tuple(operands) == args:
                return None
            if len(operands) == 1:
                return operands[0]
            return cls(*operands, evaluate=False)
        return result

    @classmethod
    def _combined(cls, operands: list[sp.Basic]) -> sp.Basic | None:
        raise NotImplementedError


class SetUnion(_Associative):
    """`s UNION t`."""

    @classmethod
    def _combined(cls, operands: list[sp.Basic]) -> sp.Basic | None:
        if _EVERYTHING in operands or _complementary(operands):
            return _EVERYTHING
        literals = [operand for operand in operands if isinstance(operand, SetOf)]
        others = [operand for operand in operands if not isinstance(operand, SetOf)]
        if not literals:
            return None
        joined = SetOf(*(element for literal in literals for element in literal.args))
        if not others:
            return joined
        if len(literals) == 1 and joined.args:
            return None
        rest = others if not joined.args else [*others, joined]
        return rest[0] if len(rest) == 1 else SetUnion(*rest, evaluate=False)


class SetIntersection(_Associative):
    """`s INTERSECTION t`."""

    @classmethod
    def _combined(cls, operands: list[sp.Basic]) -> sp.Basic | None:
        empty = SetOf()
        if empty in operands or _complementary(operands):
            return empty
        operands = [operand for operand in operands if operand != _EVERYTHING]
        if not operands:
            return _EVERYTHING
        literals = [operand for operand in operands if isinstance(operand, SetOf)]
        others = [operand for operand in operands if not isinstance(operand, SetOf)]
        if len(literals) < 2:
            if len(operands) == 1:
                return operands[0]
            return None
        first, *rest = literals
        common = [e for e in first.args if all(e in other.args for other in rest)]
        if not common or not others:
            return SetOf(*common)
        return SetIntersection(*others, SetOf(*common), evaluate=False)


#: The complement of the empty set, everything.
_EVERYTHING = SetComplement(SetOf())


def _complementary(operands: list[sp.Basic]) -> bool:
    """Whether the operands include a set and its complement."""
    return any(SetComplement(operand) in operands for operand in operands)


class SetDifference(sp.Function):
    """`s \\ t`, the elements of `s` not in `t`."""

    nargs = 2
    is_number = False

    @classmethod
    def eval(cls, left: sp.Basic, right: sp.Basic) -> sp.Basic | None:
        empty = SetOf()
        if left == right or left == empty or right == _EVERYTHING:
            return empty
        if right == empty:
            return left
        if isinstance(right, SetComplement):
            return SetIntersection(left, right.args[0])
        if not (isinstance(left, SetOf) and isinstance(right, SetOf)):
            return None
        return SetOf(*(element for element in left.args if element not in right.args))


_SET_HEADS = (
    SetOf,
    SetUnion,
    SetIntersection,
    SetDifference,
    SetComplement,
    SetSymbol,
)


# -- products, power sets and sequences ---------------------------------------------


def cartesian(sets: Sequence[SetOf]) -> SetOf:
    """The Cartesian product: the set of vectors taking one element from each set.

    `{a, b}*{c, d}` is `{[a, c], [a, d], [b, c], [b, d]}`, and three sets give
    vectors of three. An element that is a vector is a tuple already, whose elements
    the product's vectors take in turn, so the product is associative:
    `({a}*{b})*{c}` and `{a}*({b}*{c})` are both `{[a, b, c]}`, and `{[1]}*{[2], [3]}`
    is `{[1, 2], [1, 3]}`.
    """
    chosen = itertools.product(*(each.args for each in sets))
    return SetOf(
        *(
            Vector(
                *(
                    part
                    for element in elements
                    for part in (
                        element.args if isinstance(element, Vector) else (element,)
                    )
                )
            )
            for elements in chosen
        )
    )


def power_set(collection: sp.Basic, count: sp.Basic | None = None) -> sp.Basic | None:
    """All the subsets of `collection`, or those of `count` elements; None if it isn't
    a set whose subsets can be listed.

    """
    if not isinstance(collection, SetOf) or len(collection.args) > MOST_SUBSETS_OF:
        return None
    elements = collection.args
    if count is None:
        sizes: Iterable[int] = range(len(elements) + 1)
    else:
        if not (isinstance(count, sp.Integer) and count >= 0):
            return None
        sizes = (int(count),)
    return SetOf(
        *(
            SetOf(*subset)
            for chosen in sizes
            for subset in itertools.combinations(elements, chosen)
        )
    )


class _Elided(sp.Function):
    """`[a, ..., c]` or `[a, b, ..., c]` whose terms can't be counted yet.

    The arguments are the terms written, two or three. The step is `b - a`, or 1 when
    only the ends are written, and the sequence stops at the last term not past `c`.
    """

    is_number = False
    #: What the sequence becomes once expanded.
    made: type[sp.Function]

    @classmethod
    def eval(cls, *args: sp.Basic) -> sp.Basic | None:
        terms = _terms(args)
        return None if terms is None else cls.made(*terms)


class ElidedVector(_Elided):
    made = Vector


class ElidedSet(_Elided):
    made = SetOf


def is_elided(value: object) -> bool:
    """Whether `value` is an ellipsis whose terms can't be counted yet."""
    return isinstance(value, _Elided)


def elided(args: Sequence[sp.Basic], braces: bool) -> sp.Basic:
    """The vector or set of an ellipsis' terms, standing if they can't be counted."""
    return (ElidedSet if braces else ElidedVector)(*args)


def _terms(args: Sequence[sp.Basic]) -> list[sp.Basic] | None:
    if len(args) not in (2, 3):
        return None
    first, last = args[0], args[-1]
    step = args[1] - first if len(args) == 3 else sp.S.One
    try:
        count = sp.cancel((last - first) / step) if step != 0 else None
    except Exception:
        return None
    if not isinstance(count, sp.Rational):
        return None
    terms = int(sp.floor(count)) + 1
    if terms > MOST_TERMS:
        return None
    return [first + index * step for index in range(max(terms, 0))]
