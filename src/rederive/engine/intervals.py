"""What a declared interval is worth beyond the sign it implies.

`x :epsilon Real (0, 1)` says two things about `x`, and a sympy symbol can
carry only one of them: there is a fact on a symbol for "above zero" and none
for "below one". So the upper bound is dropped where the expression is
converted, and `ABS(x - 1)` comes back as it went in, where Derive answers
`1 - x`.

The bounds travel beside the expression instead, as the relational predicates
`ask` takes as its second argument, and every question this module puts is one
question: what sign does this subexpression keep over the whole box the
declarations describe. That answer settles an absolute value, a `SIGN`, the
larger of two things, and the test of an `IF`.

Three things about the way the question has to be put:

* It is asked as `nonnegative` or `nonpositive` wherever that suffices, because
  a closed bound is worth an answer too: `ABS(x - 1)` is `1 - x` over `[0, 1]`
  as much as over `(0, 1)`, and the strict form of the question comes back
  undecided there. `SIGN` is the one that needs the strict form, having a third
  answer at zero.

* An integer variable is asked about through a real stand-in. Sympy answers
  nothing at all about a symbol declared integer and bounded by a relation, and
  every integer in an interval is a real in it, so the relaxed question has the
  same answer wherever it has one. Relaxing can only lose an answer, never
  change one - and it loses nothing that was already had, since what integrality
  alone settles is settled before this module is reached.

* Only where a bound says something a symbol could not. A zero bound is the
  sign the symbol already carries and an infinite one bounds nothing, so a
  declaration with neither kind of news never builds a box and never costs an
  `ask`.

The reasoning behind the answers is sympy's, and it reaches about as far as the
bounds are linear in: `x - 1` over `(0, 1)` is settled and `x^2 - 1` is not.
An unsettled question leaves the expression alone, which is the same answer an
undeclared variable gets.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import sympy as sp
from sympy import Q
from sympy.assumptions.assume import Predicate
from sympy.core.relational import Relational

from rederive.engine.context import Context, DomainKind
from rederive.engine.to_sympy import to_sympy
from rederive.model.expr import Node

Answer = TypeVar("Answer")

__all__ = ["decided", "settled"]

#: The heads a known sign settles. Every one of them is a question about the
#: sign of something it holds, which is why they are settled together.
_SETTLED = (sp.Abs, sp.sign, sp.Max, sp.Min)


def settled(expression: sp.Basic, context: Context) -> sp.Basic:
    """Every absolute value, sign and extremum the declared intervals settle."""
    if not expression.has(*_SETTLED):
        return expression
    box = _box(expression, context)
    if box is None:
        return expression
    try:
        return expression.replace(
            lambda e: isinstance(e, _SETTLED),
            lambda e: _resolve(e, box),
            simultaneous=False,
        )
    except Exception:
        return expression


def decided(test: sp.Basic, context: Context) -> bool | None:
    """Whether the declared intervals settle this relation, and which way.

    `IF(x < 1, a, b)` is `a` where `x` was declared below one, and `x < 1` on
    its own is `true` there: the test of an `IF` and a relation standing alone
    are put the same question and get the same answer.
    """
    if not isinstance(test, Relational):
        return None
    box = _box(test, context)
    if box is None:
        return None
    difference = _attempt(lambda: test.lhs - test.rhs)
    if difference is None:
        return None
    for holds, predicate in _READINGS.get(type(test), ()):
        if box.asks(predicate, difference):
            return holds
    return None


#: How each relation reads as a question about the sign of `lhs - rhs`, in the
#: order the answers are worth asking for: the first that comes back true
#: settles it, and the negative reading is asked second because a relation
#: written down is more often meant to hold than not.
_READINGS: dict[type, tuple[tuple[bool, Predicate], ...]] = {
    sp.StrictLessThan: ((True, Q.negative), (False, Q.nonnegative)),
    sp.LessThan: ((True, Q.nonpositive), (False, Q.positive)),
    sp.StrictGreaterThan: ((True, Q.positive), (False, Q.nonpositive)),
    sp.GreaterThan: ((True, Q.nonnegative), (False, Q.negative)),
    sp.Equality: ((True, Q.zero), (False, Q.nonzero)),
    sp.Unequality: ((True, Q.nonzero), (False, Q.zero)),
}


# -- the box -----------------------------------------------------------------


@dataclass(frozen=True)
class _Box:
    """The declared intervals, as something to ask questions against."""

    #: Every bound as a predicate, conjoined.
    facts: sp.Basic
    #: The real stand-ins the integer variables are asked about.
    standing_in: dict[sp.Symbol, sp.Symbol]

    def asks(self, predicate: Predicate, expression: sp.Basic) -> bool:
        """Whether `predicate` holds of `expression` everywhere in the box."""
        relaxed = expression.xreplace(self.standing_in)
        return _attempt(lambda: sp.ask(predicate(relaxed), self.facts)) is True

    def sign_of(self, expression: sp.Basic) -> int | None:
        """1 where the expression cannot be negative, -1 where it cannot be
        positive, and None where the box does not say."""
        if self.asks(Q.nonnegative, expression):
            return 1
        return -1 if self.asks(Q.nonpositive, expression) else None


def _box(expression: sp.Basic, context: Context) -> _Box | None:
    """The declared intervals over this expression's variables, or None.

    None where no declaration says anything the symbols do not already carry,
    which is the usual case and the one that has to stay free.
    """
    facts: list[sp.Basic] = []
    standing_in: dict[sp.Symbol, sp.Symbol] = {}
    worth_asking = False
    for symbol in expression.free_symbols:
        if not isinstance(symbol, sp.Symbol):
            continue
        domain = context.domain(symbol.name)
        if domain.kind in (DomainKind.COMPLEX, DomainKind.NONSCALAR):
            continue
        about = symbol
        if domain.kind is DomainKind.INTEGER:
            about = sp.Dummy(symbol.name, real=True)
            standing_in[symbol] = about
        for value, closed, above in (
            (_bound(domain.low), domain.closed_low, True),
            (_bound(domain.high), domain.closed_high, False),
        ):
            if value is None or not value.is_finite:
                continue
            facts.append(_predicate(about, value, closed, above))
            # A bound at zero is the sign the symbol carries anyway; anything
            # else is news, and news is what makes the asking worth its cost.
            worth_asking = worth_asking or not value.is_zero
    if not (facts and worth_asking):
        return None
    return _Box(sp.And(*facts), standing_in)


def _predicate(
    about: sp.Symbol, value: sp.Basic, closed: bool, above: bool
) -> sp.Basic:
    """One end of an interval, as the predicate that says it."""
    if above:
        return Q.ge(about, value) if closed else Q.gt(about, value)
    return Q.le(about, value) if closed else Q.lt(about, value)


def _bound(node: Node | None) -> sp.Basic | None:
    """What an interval bound is worth, or None where it is not a number.

    Read without the domains in hand, as the conversion reads them: a bound
    naming a declared variable would send the reading round in a circle.
    """
    if node is None:
        return None
    value = _attempt(lambda: to_sympy(node, Context()))
    return value if getattr(value, "is_number", False) else None


# -- what a sign settles -----------------------------------------------------


def _resolve(head: sp.Basic, box: _Box) -> sp.Basic:
    """One head, as what the box makes of it, or as it stands."""
    if isinstance(head, (sp.Max, sp.Min)):
        return _extremum(head, box)
    if isinstance(head, sp.sign):
        return _signum(head, box)
    argument = head.args[0]
    sign = box.sign_of(argument)
    return head if sign is None else sign * argument


def _signum(head: sp.Basic, box: _Box) -> sp.Basic:
    """`SIGN(u)`, which needs the strict question: zero is an answer of its own."""
    argument = head.args[0]
    if box.asks(Q.positive, argument):
        return sp.S.One
    if box.asks(Q.negative, argument):
        return sp.S.NegativeOne
    return head


def _extremum(head: sp.Basic, box: _Box) -> sp.Basic:
    """`MAX(a, b)` or `MIN(a, b)` where the box says which of the two it is.

    Two arguments, because that is the comparison a sign answers. More of them
    is a tournament, and one the box would have to be asked about a pair at a
    time; nothing in the corpus writes one.
    """
    if len(head.args) != 2:
        return head
    left, right = head.args
    sign = box.sign_of(left - right)
    if sign is None:
        return head
    larger = isinstance(head, sp.Max)
    return left if (sign > 0) == larger else right


def _attempt(question: Callable[[], Answer]) -> Answer | None:
    """The answer, or None where asking it raised."""
    try:
        return question()
    except Exception:
        return None
