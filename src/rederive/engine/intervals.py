"""Using declared intervals beyond the sign they imply.

A sympy symbol can carry "above zero" but not "below one", so for `x :epsilon Real (0,
1)` the upper bound is lost in conversion, and `ABS(x - 1)` stays as is where Derive
answers `1 - x`.

So the bounds are kept beside the expression, as predicates for `ask`'s second
argument, and this module asks one kind of question: what sign does a subexpression
keep over the whole box the declarations describe? That settles absolute values,
`SIGN`, `MAX`/`MIN`, and `IF` tests.

How the question is asked:

* As `nonnegative` or `nonpositive` where that suffices, so closed bounds work too:
  `ABS(x - 1)` is `1 - x` over `[0, 1]` as well as `(0, 1)`, where the strict question
  would be undecided. Only `SIGN` needs the strict form, since zero is a third answer.

* Integer variables are replaced by real stand-ins. Sympy answers nothing about a
  symbol declared integer and bounded by a relation, and every integer in an interval
  is a real in it, so the relaxed question can only lose answers, never change them.
  It loses none that matter, since what integrality alone settles is settled earlier.

* Only when a bound says something the symbol can't. A zero bound is already the
  symbol's sign and an infinite one bounds nothing, so such declarations never build
  a box or cost an `ask`.

Sympy's reasoning is tried first, which handles roughly linear bounds, then interval
arithmetic, which handles powers, absolute values and monotone functions: `x^2 - 1`
and `LN(x)` are both negative over `(0, 1)`. Interval arithmetic treats each
occurrence of a variable independently, so `x - x^2` over `(0, 1)` is put in `(-1,
1)` with unknown sign. An undecided question leaves the expression alone, as for an
undeclared variable.

Ranges also settle a `FLOOR` constant over them, an `ASIN(SIN(t))` whose angle stays on
one branch, and the cases of a conditional.
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

__all__ = ["covering", "decided", "never_negative", "settled"]

#: Heads settled by knowing the sign of an argument.
_SIGNED = (sp.Abs, sp.sign, sp.Max, sp.Min)

#: Heads settled by a known range: step functions constant over it, inverses undoing
#: their function on one branch, and conditionals whose cases it decides.
_RANGED = (sp.floor, sp.ceiling, sp.asin, sp.acos, sp.atan, sp.Piecewise)

_SETTLED = _SIGNED + _RANGED


def settled(expression: sp.Basic, context: Context) -> sp.Basic:
    """`expression` with every absolute value, sign, extremum, step, inverse and
    conditional that the declared intervals settle resolved."""
    if not expression.has(*_SETTLED):
        return expression
    box = _box(expression, context)
    if box is None:
        return expression
    try:
        return expression.replace(
            lambda e: isinstance(e, _SETTLED),
            lambda e: _resolve(e, box, context),
            simultaneous=False,
        )
    except Exception:
        return expression


def decided(test: sp.Basic, context: Context) -> bool | None:
    """Whether the declared intervals settle this relation, and which way.

    `IF(x < 1, a, b)` is `a` where `x` is declared below one, and `x < 1` alone is
    `true` there: both ask the same question.
    """
    if isinstance(test, (sp.And, sp.Or, sp.Not)):
        return _connected(test, context)
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


def never_negative(expression: sp.Basic) -> bool:
    """Whether `expression` is provably real and not below zero, whatever its variables.

    Goes by the symbols' own assumptions, with interval arithmetic for what sympy
    doesn't see: `1 + COS(x)`, `ABS(x)` and `x^2` all qualify for real `x`.
    """
    if expression.is_nonnegative:
        return True
    reach = _attempt(lambda: _range(expression, {}))
    return reach is not None and _holds(Q.nonnegative, reach)


def _connected(test: sp.Basic, context: Context) -> bool | None:
    """Decide a conjunction, disjunction or negation from its operands."""
    if isinstance(test, sp.Not):
        inner = decided(test.args[0], context)
        return None if inner is None else not inner
    conjunction = isinstance(test, sp.And)
    verdicts = [decided(operand, context) for operand in test.args]
    if (not conjunction) in verdicts:
        return not conjunction
    if all(verdict is conjunction for verdict in verdicts):
        return conjunction
    return None


def covering(symbol: sp.Symbol, context: Context) -> list[sp.Basic] | None:
    """Expressions in one positive variable that together cover every value the
    declaration allows `symbol`, or None if it declares no interval.

    This lets sympy, which only knows "positive", work over an interval: `(1, inf)` is
    `1 + p` for positive `p`, `(0, 1)` is `1/(1 + p)`, and each closed end is added as
    its own value. The variable is the positive symbol `p` in the first expression.
    """
    domain = context.domain(symbol.name)
    if domain.kind not in (DomainKind.REAL, DomainKind.UNKNOWN):
        return None
    low, high = _bound(domain.low), _bound(domain.high)
    low = None if low is None or not low.is_finite else low
    high = None if high is None or not high.is_finite else high
    if low is None and high is None:
        return None
    size = sp.Dummy("p", positive=True)
    if high is None:
        values = [low + size]
    elif low is None:
        values = [high - size]
    else:
        values = [low + (high - low) / (1 + size)]
    if low is not None and domain.closed_low:
        values.append(low)
    if high is not None and domain.closed_high:
        values.append(high)
    return values


#: Each relation as (verdict, predicate on `lhs - rhs`) pairs. The first predicate
#: that holds decides; the one for "holds" comes first since relations usually do.
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
    """The declared intervals, for asking questions against."""

    #: The conjunction of all bounds as predicates.
    facts: sp.Basic
    #: Real stand-ins for the integer variables.
    standing_in: dict[sp.Symbol, sp.Symbol]

    #: Each bounded variable's interval, including the real stand-ins.
    ranges: dict[sp.Symbol, _Range]

    def asks(self, predicate: Predicate, expression: sp.Basic) -> bool:
        """Whether `predicate` holds of `expression` everywhere in the box.

        Sympy is asked first, then the interval arithmetic of `_range`, which sees
        through absolute values and monotone functions that sympy doesn't:
        `ABS(x) < 1` over `(-1, 1)`, and `LN(x) < 0` over `(0, 1)`.
        """
        relaxed = expression.xreplace(self.standing_in)
        if _attempt(lambda: sp.ask(predicate(relaxed), self.facts)) is True:
            return True
        reach = _attempt(lambda: _range(relaxed, self.ranges))
        return reach is not None and _holds(predicate, reach)

    def range_of(self, expression: sp.Basic) -> _Range | None:
        """The values `expression` takes over the box, or None if unknown."""
        relaxed = expression.xreplace(self.standing_in)
        return _attempt(lambda: _range(relaxed, self.ranges))

    def sign_of(self, expression: sp.Basic) -> int | None:
        """1 if the expression can't be negative, -1 if it can't be positive, else
        None."""
        if self.asks(Q.nonnegative, expression):
            return 1
        return -1 if self.asks(Q.nonpositive, expression) else None


def _box(expression: sp.Basic, context: Context) -> _Box | None:
    """The declared intervals over this expression's variables, or None.

    None if no declaration adds anything to the symbols' assumptions, which is the
    common case and must stay cheap.
    """
    facts: list[sp.Basic] = []
    standing_in: dict[sp.Symbol, sp.Symbol] = {}
    ranges: dict[sp.Symbol, _Range] = {}
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
        low, high = _bound(domain.low), _bound(domain.high)
        ranges[about] = _declared(low, domain.closed_low, high, domain.closed_high)
        for value, closed, above in (
            (low, domain.closed_low, True),
            (high, domain.closed_high, False),
        ):
            if value is None or not value.is_finite:
                continue
            facts.append(_predicate(about, value, closed, above))
            # A zero bound is already the symbol's sign; only other bounds make
            # asking worthwhile.
            worth_asking = worth_asking or not value.is_zero
    if not (facts and worth_asking):
        return None
    return _Box(sp.And(*facts), standing_in, ranges)


def _predicate(
    about: sp.Symbol, value: sp.Basic, closed: bool, above: bool
) -> sp.Basic:
    """One end of an interval, as a predicate."""
    if above:
        return Q.ge(about, value) if closed else Q.gt(about, value)
    return Q.le(about, value) if closed else Q.lt(about, value)


def _bound(node: Node | None) -> sp.Basic | None:
    """The value of an interval bound, or None if it isn't a number.

    Converted without domains, as in the conversion layer, since a bound naming a
    declared variable would otherwise recurse.
    """
    if node is None:
        return None
    value = _attempt(lambda: to_sympy(node, Context()))
    return value if getattr(value, "is_number", False) else None


# -- the values an expression takes -------------------------------------------


@dataclass(frozen=True)
class _Range:
    """An interval of the extended reals, each end open or closed.

    An infinite end is always open. An end is closed only if the value is known to be
    reached. Wrongly marking a reached end open could decide `ABS(x) < 1` over `[-1,
    1]`, while wrongly marking one closed merely leaves a strict question undecided.
    """

    low: sp.Basic
    high: sp.Basic
    low_open: bool = True
    high_open: bool = True


_WHOLE_LINE = _Range(sp.S.NegativeInfinity, sp.S.Infinity)


def _declared(
    low: sp.Basic | None, closed_low: bool, high: sp.Basic | None, closed_high: bool
) -> _Range:
    """The range of a variable given its declared bounds."""
    low = sp.S.NegativeInfinity if low is None else low
    high = sp.S.Infinity if high is None else high
    return _Range(
        low,
        high,
        not (closed_low and low.is_finite),
        not (closed_high and high.is_finite),
    )


def _holds(predicate: Predicate, reach: _Range) -> bool:
    """Whether `predicate` holds for every value in `reach`."""
    above_zero = _true(reach.low > 0) or (reach.low_open and _true(reach.low >= 0))
    below_zero = _true(reach.high < 0) or (reach.high_open and _true(reach.high <= 0))
    if predicate is Q.positive:
        return above_zero
    if predicate is Q.negative:
        return below_zero
    if predicate is Q.nonnegative:
        return _true(reach.low >= 0)
    if predicate is Q.nonpositive:
        return _true(reach.high <= 0)
    if predicate is Q.nonzero:
        return above_zero or below_zero
    if predicate is Q.zero:
        return reach.low == 0 and reach.high == 0
    return False


def _true(relation: sp.Basic) -> bool:
    return relation is sp.true


def _range(expression: sp.Basic, ranges: dict[sp.Symbol, _Range]) -> _Range | None:
    """The values `expression` takes where its variables take theirs, or None.

    Interval arithmetic over sums, products, powers and monotone functions. The result
    may be wider than the true range (`x - x` over `(0, 1)` gives `(-1, 1)`) but never
    narrower, so sign conclusions hold. Unsupported expressions and variables not
    known real have no range.
    """
    if expression.is_number:
        if expression in (sp.oo, -sp.oo):
            return _Range(expression, expression)
        if expression.is_extended_real is not True or expression.is_finite is not True:
            return None
        return _Range(expression, expression, False, False)
    if isinstance(expression, sp.Symbol):
        return ranges.get(expression) or _assumed(expression)
    if isinstance(expression, sp.Add):
        return _fold(expression.args, ranges, _sum)
    if isinstance(expression, sp.Mul):
        return _fold(expression.args, ranges, _product)
    if isinstance(expression, sp.Pow):
        return _power(expression, ranges)
    if isinstance(expression, (sp.sin, sp.cos)):
        if _range(expression.args[0], ranges) is None:
            return None
        return _Range(sp.S.NegativeOne, sp.S.One, False, False)
    rule = _MONOTONE.get(type(expression))
    if rule is None or len(expression.args) != 1:
        return None
    inner = _range(expression.args[0], ranges)
    return None if inner is None else rule(inner)


def _assumed(symbol: sp.Symbol) -> _Range | None:
    """The range implied by an undeclared symbol's assumptions."""
    if symbol.is_extended_real is not True:
        return None
    if symbol.is_positive:
        return _Range(sp.S.Zero, sp.S.Infinity)
    if symbol.is_nonnegative:
        return _Range(sp.S.Zero, sp.S.Infinity, False)
    if symbol.is_negative:
        return _Range(sp.S.NegativeInfinity, sp.S.Zero)
    if symbol.is_nonpositive:
        return _Range(sp.S.NegativeInfinity, sp.S.Zero, True, False)
    return _WHOLE_LINE


def _fold(
    args: tuple[sp.Basic, ...],
    ranges: dict[sp.Symbol, _Range],
    combine: Callable[[_Range, _Range], _Range | None],
) -> _Range | None:
    total: _Range | None = None
    for arg in args:
        reach = _range(arg, ranges)
        if reach is None:
            return None
        total = reach if total is None else combine(total, reach)
        if total is None:
            return None
    return total


def _sum(a: _Range, b: _Range) -> _Range | None:
    low, high = a.low + b.low, a.high + b.high
    if low is sp.nan or high is sp.nan:
        return None
    return _Range(low, high, a.low_open or b.low_open, a.high_open or b.high_open)


def _product(a: _Range, b: _Range) -> _Range | None:
    """The hull of the endpoint products. An end is reached if both factors' ends are,
    or one is a reached zero. Infinity times zero is taken as zero, the needed limit."""
    candidates = []
    for x, x_open in ((a.low, a.low_open), (a.high, a.high_open)):
        for y, y_open in ((b.low, b.low_open), (b.high, b.high_open)):
            value = sp.S.Zero if x.is_zero or y.is_zero else x * y
            if value is sp.nan:
                return None
            reached = (not x_open and not y_open) or (
                (x.is_zero and not x_open) or (y.is_zero and not y_open)
            )
            candidates.append((value, reached and value.is_finite))
    return _hull(candidates)


def _hull(candidates: list[tuple[sp.Basic, bool]]) -> _Range | None:
    """The interval from the least to the greatest value, each end closed if some
    candidate at it is reached."""
    values = [value for value, _ in candidates]
    low = _extreme(values, least=True)
    high = _extreme(values, least=False)
    if low is None or high is None:
        return None
    return _Range(
        low,
        high,
        not any(reached for value, reached in candidates if value == low),
        not any(reached for value, reached in candidates if value == high),
    )


def _extreme(values: list[sp.Basic], least: bool) -> sp.Basic | None:
    best = values[0]
    for value in values[1:]:
        smaller = value < best
        if smaller not in (sp.true, sp.false):
            return None
        if (smaller is sp.true) == least and value != best:
            best = value
    return best


def _power(expression: sp.Pow, ranges: dict[sp.Symbol, _Range]) -> _Range | None:
    base, exponent = expression.args
    if exponent.is_number:
        reach = _range(base, ranges)
        if reach is None:
            return None
        return _to_power(reach, exponent)
    # A positive base to a variable power: rewrite as an exponential.
    if base.is_number and base.is_positive:
        return _range(sp.exp(exponent * sp.log(base), evaluate=False), ranges)
    return None


def _to_power(reach: _Range, exponent: sp.Basic) -> _Range | None:
    if exponent.is_zero:
        return _Range(sp.S.One, sp.S.One, False, False)
    if exponent.is_Integer and exponent.is_positive:
        return _monotone_power(reach, exponent)
    if exponent.is_Integer and exponent.is_negative:
        raised = _monotone_power(reach, -exponent)
        return None if raised is None else _reciprocal(raised)
    if exponent.is_extended_real and _true(reach.low >= 0):
        # Fractional powers only of a nonnegative base: increasing for a positive
        # exponent, decreasing for a negative one.
        if exponent.is_positive:
            return _image(reach, lambda v: v**exponent, increasing=True)
        if _true(reach.low > 0) or reach.low_open:
            return _image(reach, lambda v: _inverse(v**-exponent), increasing=False)
    return None


def _monotone_power(reach: _Range, n: sp.Integer) -> _Range | None:
    def raised(v: sp.Basic) -> sp.Basic:
        return v**n

    if n.is_odd or _true(reach.low >= 0):
        return _image(reach, raised, increasing=True)
    if _true(reach.high <= 0):
        return _image(reach, raised, increasing=False)
    low, high = raised(reach.low), raised(reach.high)
    return _hull(
        [(sp.S.Zero, True), (low, not reach.low_open), (high, not reach.high_open)]
    )


def _reciprocal(reach: _Range) -> _Range | None:
    """`1/u` over a range on one side of zero; an open zero end goes to infinity."""
    if _true(reach.low > 0) or (reach.low_open and _true(reach.low >= 0)):
        return _image(reach, _inverse, increasing=False)
    if _true(reach.high < 0) or (reach.high_open and _true(reach.high <= 0)):
        return _image(reach, lambda v: -_inverse(-v), increasing=False)
    return None


def _inverse(value: sp.Basic) -> sp.Basic:
    """`1/v` for nonnegative `v`, with zero approached from above."""
    if value.is_zero:
        return sp.S.Infinity
    return sp.S.Zero if value.is_infinite else 1 / value


def _image(
    reach: _Range, function: Callable[[sp.Basic], sp.Basic], increasing: bool
) -> _Range | None:
    """The image of the range under a function monotone over it, mapping ends to ends.
    Infinite image ends are open."""
    low, high = function(reach.low), function(reach.high)
    if not (low.is_extended_real and high.is_extended_real):
        return None
    low_open = reach.low_open or not low.is_finite
    high_open = reach.high_open or not high.is_finite
    if increasing:
        return _Range(low, high, low_open, high_open)
    return _Range(high, low, high_open, low_open)


def _over(lowest: sp.Basic, function: Callable[[sp.Basic], sp.Basic]):
    """A `_MONOTONE` rule for an increasing function defined from `lowest` up."""

    def rule(reach: _Range) -> _Range | None:
        if not _true(reach.low >= lowest):
            return None
        return _image(reach, function, increasing=True)

    return rule


def _within(low: sp.Basic, high: sp.Basic, function, increasing: bool):
    """A `_MONOTONE` rule for a monotone function defined between `low` and `high`."""

    def rule(reach: _Range) -> _Range | None:
        if not (_true(reach.low >= low) and _true(reach.high <= high)):
            return None
        return _image(reach, function, increasing)

    return rule


def _everywhere(function: Callable[[sp.Basic], sp.Basic]):
    """A `_MONOTONE` rule for an increasing function defined everywhere."""
    return lambda reach: _image(reach, function, increasing=True)


def _magnitude(reach: _Range) -> _Range | None:
    if _true(reach.low >= 0):
        return reach
    if _true(reach.high <= 0):
        return _Range(-reach.high, -reach.low, reach.high_open, reach.low_open)
    return _hull(
        [
            (sp.S.Zero, True),
            (-reach.low, not reach.low_open),
            (reach.high, not reach.high_open),
        ]
    )


#: Functions whose range is computed from their argument's range, with their rules.
_MONOTONE: dict[type, Callable[[_Range], _Range | None]] = {
    sp.Abs: _magnitude,
    sp.exp: _everywhere(sp.exp),
    sp.log: _over(sp.S.Zero, lambda v: -sp.oo if v.is_zero else sp.log(v)),
    sp.atan: _everywhere(sp.atan),
    sp.sinh: _everywhere(sp.sinh),
    sp.tanh: _everywhere(sp.tanh),
    sp.asinh: _everywhere(sp.asinh),
    sp.asin: _within(sp.S.NegativeOne, sp.S.One, sp.asin, increasing=True),
    sp.acos: _within(sp.S.NegativeOne, sp.S.One, sp.acos, increasing=False),
}


# -- what a sign settles -----------------------------------------------------


def _resolve(head: sp.Basic, box: _Box, context: Context) -> sp.Basic:
    """One head, resolved using the box if possible."""
    if isinstance(head, sp.Piecewise):
        return _cases(head, context)
    if isinstance(head, (sp.floor, sp.ceiling)):
        return _step(head, box)
    if isinstance(head, (sp.asin, sp.acos, sp.atan)):
        return _inverted(head, box)
    if isinstance(head, (sp.Max, sp.Min)):
        return _extremum(head, box)
    if isinstance(head, sp.sign):
        return _signum(head, box)
    argument = head.args[0]
    sign = box.sign_of(argument)
    return head if sign is None else sign * argument


def _cases(head: sp.Piecewise, context: Context) -> sp.Basic:
    """A conditional without the cases the box rules out, or the one case it picks.

    False cases are dropped, and a true case is the answer if all before it were
    dropped: `SUM(x^k, k, 0, inf)` over `(0, 1)` is `1/(1 - x)`, and over `(1, inf)` it
    is the sum itself (the else case).
    """
    kept = []
    for value, condition in head.args:
        verdict = True if condition is sp.true else decided(condition, context)
        if verdict is False:
            continue
        if verdict is True and not kept:
            return value
        kept.append((value, condition))
        if verdict is True:
            break
    if len(kept) == len(head.args):
        return head
    if not kept:
        return sp.nan
    return sp.Piecewise(*kept)


def _step(head: sp.Basic, box: _Box) -> sp.Basic:
    """`FLOOR(u)` as the lower integer when `u` stays in `[n, n + 1)`; `CEILING`
    likewise from above."""
    if len(head.args) != 1:
        return head
    reach = box.range_of(head.args[0])
    if reach is None or not (reach.low.is_finite and reach.high.is_finite):
        return head
    if isinstance(head, sp.floor):
        whole = sp.floor(reach.low)
        above = whole + 1
        if _true(reach.high < above) or (reach.high_open and reach.high == above):
            return whole
        return head
    whole = sp.ceiling(reach.high)
    below = whole - 1
    if _true(reach.low > below) or (reach.low_open and reach.low == below):
        return whole
    return head


def _inverted(head: sp.Basic, box: _Box) -> sp.Basic:
    """`ASIN(SIN(t))`, `ACOS(COS(t))` or `ATAN(TAN(t))`, when `t` stays on one branch,
    as `t` mapped to the principal branch.

    `ASIN(SIN(t))` is `t` over `[-pi/2, pi/2]` and `pi - t` over `[pi/2, 3*pi/2]`; on
    branch `k` it is `(-1)^k*(t - k*pi)`. Arc cosine branches start at multiples of
    pi, and arc tangent branches are the open intervals of width pi centered on them.
    """
    inner = head.args[0]
    function = {sp.asin: sp.sin, sp.acos: sp.cos, sp.atan: sp.tan}[type(head)]
    if not isinstance(inner, function):
        return head
    angle = inner.args[0]
    reach = box.range_of(angle)
    if reach is None or not (reach.low.is_finite and reach.high.is_finite):
        return head
    pi = sp.pi
    if isinstance(head, sp.acos):
        turns = sp.floor(reach.low / pi)
        start, end = turns * pi, (turns + 1) * pi
        if not (_true(reach.low >= start) and _true(reach.high <= end)):
            return head
        return angle - start if turns.is_even else end - angle
    turns = sp.floor(reach.low / pi + sp.S.Half)
    start, end = (turns - sp.S.Half) * pi, (turns + sp.S.Half) * pi
    if isinstance(head, sp.atan):
        after = _true(reach.low > start) or (reach.low_open and reach.low == start)
        before = _true(reach.high < end) or (reach.high_open and reach.high == end)
        inside = after and before
        return angle - turns * pi if inside else head
    if not (_true(reach.low >= start) and _true(reach.high <= end)):
        return head
    return (-1) ** turns * (angle - turns * pi)


def _signum(head: sp.Basic, box: _Box) -> sp.Basic:
    """`SIGN(u)`, using strict questions since zero is a separate answer."""
    argument = head.args[0]
    if box.asks(Q.positive, argument):
        return sp.S.One
    if box.asks(Q.negative, argument):
        return sp.S.NegativeOne
    return head


def _extremum(head: sp.Basic, box: _Box) -> sp.Basic:
    """`MAX(a, b)` or `MIN(a, b)`, if the box decides which argument it is.

    Only two arguments are handled; more would need pairwise comparisons, and
    nothing in the corpus uses them.
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
    """The answer, or None if the question raised."""
    try:
        return question()
    except Exception:
        return None
