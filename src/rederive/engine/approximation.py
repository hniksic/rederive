"""Approximate numbers: the simplest rational the precision allows.

Derive's approximate arithmetic is exact arithmetic on approximate numbers, each a
*rational*: the simplest one representing the value to the requested precision. π to
six digits is 355/113, which `Notation := Rational` shows; an approximate answer is an
exact value, just different from the one it approximates.

Consequences:

* Numbers that need no digits get none. 5 stays 5 and two thirds stays `2/3`, shown to
  six digits as `0.666666` (truncated, not rounded to `0.666667`).
* A value keeps its displayed digits. `SQRT(3)` is `1.73205`, not the `1.73204` a
  slightly smaller, simpler rational would show.
* Every digit shown is correct. A number is computed as if exactly, with whatever
  working precision that takes, and only then rounded, rather than rounding the pieces
  and letting arithmetic carry their errors. The thousandth power of a root of 12
  written in radicals is 12, not `11.9994`, and `SQRT(10^200 + 1) - 10^100` is
  `5.00000*10^-101`, not 0. Here we deliberately deviate from Derive, which rounds the
  pieces and answers `12 - 0.000430980*#i` and 0.
* That holds within one step of a computation. Between steps (iterates, the running
  total of a sum, the value a defined function returns) values are rounded to
  working precision, as Derive's floating point does, so iterations stay numeric;
  see `carried`.

Deviation from the manual: 3.8 p.46 relies on the two fractions in `SQRT(3422357/2313
- 1140443/771)` being rounded before subtraction, so Approximate mode falls short of
the exact 2/3 that Mixed mode reaches. Both are ratios of integers, so neither is
rounded here and both modes give 2/3. Rounding every step doesn't reproduce the
manual's digits either; its 0.666622 reflects binary floating point. `test_manual`
documents this.

"To the requested precision" is a tolerance one digit tighter than the digits shown.
The looser reading (the simplest rational with the same displayed digits) allows an
error of a whole unit in the last digit and picks the coarsest such value, which is
the worst choice for further exact arithmetic. With it, `SQRT(1000003) -
SQRT(1000002)` would give 0: both roots display `1000.00`, the simplest rational
displaying that is 1000, and a difference of about 0.0005 vanishes. The guard digit
keeps substituted values close enough for cancellations to stay meaningful.

`approximated` applies this rule to every number in an expression that needs digits.
Each number is the largest numeric subexpression: in `x*SQRT(2) + pi` these are
`SQRT(2)` and `pi`, and in `x + SQRT(3) - SQRT(2)` the difference of the roots.

The simplest rational within the tolerance is found by walking the continued fractions
of the interval's ends: each shared whole part is the next term, until an integer
falls between them. π gives 355/113 not because the tolerance is coarse but because
355/113 is an unusually good approximation for its denominator.

Derive's behavior can be checked directly: under Rational notation with NotationDigits
at 32 it prints approximations as ratios. That confirms this tolerance wherever its
value is accurate: π at three to nine digits gives 333/106, 355/113, 103993/33102,
208341/66317, each the first convergent within 10^-(digits + 1). Where it isn't, no
tolerance would match: its SQRT(10001) at six digits is 180109/1801, though 20001/200
is simpler and two hundred times nearer, and SQRT(2) at three digits is 379/268,
though 239/169 is simpler and three times nearer. That is the error of its own square
root, computed a digit or two past the requested precision. `test_manual`'s
`NOT_YET_HELD` records which manual pages this affects.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator, Sequence, Set
from contextlib import contextmanager
from contextvars import ContextVar
from decimal import Decimal
from fractions import Fraction
from typing import ParamSpec, TypeVar

import mpmath
import sympy as sp
from sympy.core.relational import Relational

from rederive.engine.context import Context, Precision
from rederive.engine.notation import exponent
from rederive.engine.vectors import Vector

__all__ = [
    "GUARD",
    "approximated",
    "carried",
    "drawn_fraction",
    "drew_fractions",
    "is_approximation",
    "noting_approximations",
    "reapproximated",
    "rescaled",
    "simplest",
    "simplest_rational",
]

#: Extra digits used when evaluating an irrational, so the requested digits are
#: correct rather than rounded.
GUARD = 5

#: Maximum working precision when evaluating a number. Enough for any cancellation
#: short of an exact zero (`SQRT(10^200 + 1) - 10^100` needs about 250), and a hidden
#: zero exhausts it in milliseconds.
MAX_DIGITS = 1000

_TEN = Fraction(10)


#: The magnitude, as a power of ten, from which a number is shown in scientific notation
#: rather than with all its integer digits.
_SHOWN_WHOLE = 4

_TENTH = Fraction(1, 10)


def simplest(value: Fraction, digits: int, cut: bool = True) -> Fraction:
    """The simplest rational representing `value` at `digits` of precision.

    The result must show the value's own digits, truncated as numbers are shown, or
    rounded. The tolerance alone would allow a simpler rational just below, so typing
    `3.14159` would show `3.14158`, a digit that is neither. So the search interval is
    the tolerance with its lower end raised to the value's truncation, which always
    contains the value. The upper end stays: the tolerance is below a tenth of the last
    digit, so it reaches the next digit only from a value that rounds up to it. There
    the simplest rational decides, as in Derive: `4.99999` to two digits is 5 and
    `APPROX(0.12399, 3)` is `0.124`, while π to twelve digits is 5419351/1725033,
    `3.14159265358`. Keeping below the next digit would make the first two `4.995` and
    `0.123989`, digits the value has at no precision.

    With `cut` false the whole tolerance is searched, for values computed from
    approximate numbers whose digits aren't meaningful; see `reapproximated`.
    """
    if not value:
        return Fraction(0)
    sign = -1 if value < 0 else 1
    value = abs(value)
    tolerance = value * Fraction(1, 10 ** (digits + 1))
    unit = _TEN ** (exponent(value) - digits + 1)
    if exponent(value) < _SHOWN_WHOLE and unit > _TENTH:
        # A number below 10^4 is shown with all its integer digits and one after the
        # point, however few digits are asked for, so that digit must be its own:
        # 365.92... to three digits shows as 365.9.
        unit = _TENTH
    shown = value // unit * unit
    low, high = value - tolerance, value + tolerance
    if cut:
        low = max(low, shown)
    simplest_here = _simplest_between(low, high)
    # Simplest fixes only the denominator: if the interval holds several rationals
    # over it, pick the nearest. For `10^7*pi` the interval holds six integers, and
    # 31415929 is the right one.
    denominator = simplest_here.denominator
    nearest = Fraction(round(value * denominator), denominator)
    inside = low <= nearest <= high
    return sign * (nearest if inside else simplest_here)


def _simplest_between(low: Fraction, high: Fraction) -> Fraction:
    """The simplest rational between `low` and `high`, inclusive, which must be
    positive.

    While both ends share a whole part, any rational between them has it too, so it is
    recorded and the walk continues with the reciprocals of the remainders (swapping
    the ends). Once an integer falls between them, the smallest one is the simplest
    choice.

    The walk runs over the numerators and denominators as integers: each end stays in
    lowest terms, and the continued fraction is folded back at the end.
    """
    low_top, low_bottom = low.numerator, low.denominator
    high_top, high_bottom = high.numerator, high.denominator
    wholes = []
    while True:
        whole = low_top // low_bottom
        if low_bottom == 1:
            top, bottom = low_top, 1
            break
        above = (whole + 1) * high_bottom
        if above <= high_top:
            top, bottom = whole + 1, 1
            break
        if low_top == whole * low_bottom:
            # The remainder reaches zero, so its reciprocal is unbounded above; take
            # the smallest admissible integer above 1/(high - whole).
            bound_top, bound_bottom = high_bottom, high_top - whole * high_bottom
            floor = bound_top // bound_bottom
            if floor * bound_bottom < bound_top:
                floor += 1
            top, bottom = whole * floor + 1, floor
            break
        wholes.append(whole)
        low_top, low_bottom, high_top, high_bottom = (
            high_bottom,
            high_top - whole * high_bottom,
            low_bottom,
            low_top - whole * low_bottom,
        )
    for whole in reversed(wholes):
        top, bottom = whole * top + bottom, top
    return Fraction(top, bottom)


def approximated(expression: sp.Basic, context: Context) -> sp.Basic:
    """`expression` with the precision mode applied.

    The last step of every command's pipeline. Factor simplifies exactly, factors,
    and only then rounds, so radical factoring reaches `SQRT(2)` before it is shown as
    `1.41421`.

    Under Approximate it is also applied, at working precision, to each value carried
    from one step of a computation to the next; see `carried`.
    """
    digits = context.precision_digits
    noted = _noted.get()
    match context.precision:
        case Precision.APPROXIMATE:
            # Approximate each number as a whole, then round any remaining rationals
            # (e.g. a coefficient multiplied into a rational).
            #
            # The second pass skips what the first rounded, since rounding is not
            # idempotent: the tolerance is measured from the new value, so it could
            # reach a simpler rational a whole tolerance away and lose a digit.
            approximate: set[sp.Rational] = set()
            expression = rescaled(expression, digits)
            result = _rounded(
                _approximated(expression, digits, approximate, rounding=True),
                digits,
                approximate,
            )
            if noted is not None:
                noted.update(_fraction(n) for n in _atoms(result, sp.Rational))
            return result
        case Precision.MIXED:
            approximate = set()
            result = _approximated(expression, digits, approximate)
            if noted is not None:
                noted.update(_fraction(n) for n in approximate if n.is_Rational)
            return result
    return expression


def carried(value: sp.Basic, context: Context) -> sp.Basic:
    """`value` as handed from one step of a computation to the next.

    Exact-first applies within a step, not across steps. Under Approximate, where
    Derive works in floating point, a value is rounded as it is carried on:

    * each iterate of `ITERATE` and `ITERATES`;
    * each running total of a `SUM` or `PRODUCT` written out term by term;
    * the value a defined function returns, which makes a recursion, and a function
      built from others (an AGM step, a Runge-Kutta stage, a Bessel recurrence),
      numeric step by step.

    The arithmetic inside one step (one function body, one summand, one update) is
    still exact and rounded only at its end, so `SQRT(10001) - 100` keeps every digit,
    and so does a defined function computing it. Left exact across steps, each one
    nests the previous one: an iterated mean builds a tower of radicals that doubles
    with every step and never reduces to the six digits it is printed with.

    Rounding is to working precision, `GUARD` digits past the requested ones, so the
    errors of many steps stay below the digits shown. Only numbers are rounded, and the
    numbers in a vector or matrix: a value with a variable in it stays exact, since it
    is still to be worked on symbolically. Nor is a number rounded whose exact form is
    no longer to write than its rounding: it can't grow into a tower, and a later step
    may need it exact, as a limit at `SQRT(2)`, a root of the denominator, does. Exact
    and Mixed modes carry every value exactly, however large it grows.
    """
    if context.precision is not Precision.APPROXIMATE:
        return value
    digits = context.precision_digits + GUARD
    if isinstance(value, sp.Integer) and abs(value.p) < 10**digits:
        # Its own rounding, and the commonest value a loop carries: a counter.
        return value
    return _numbers_carried(value, _working(context))


def carried_out(value: sp.Basic, context: Context) -> sp.Basic:
    """`value`, computed under `context`, as handed to a computation at another
    precision, as a function with a local `Precision` setting returns it: under
    Approximate, its numbers rounded to working precision (see `carried`), as the
    precision's last step, and as it is under any other precision."""
    if context.precision is not Precision.APPROXIMATE:
        return value
    return approximated(value, _working(context))


def _working(context: Context) -> Context:
    """`context` at working precision, `GUARD` digits past its own."""
    return context.with_precision(
        Precision.APPROXIMATE, context.precision_digits + GUARD
    )


def _numbers_carried(value: sp.Basic, working: Context) -> sp.Basic:
    contained = _elementwise(value, lambda e: _numbers_carried(e, working))
    if contained is not None:
        return contained
    if isinstance(value, Vector):
        return _vector_carried(value, working)
    if isinstance(value, sp.Rational):
        # What `approximated` makes of a rational, without walking it as an expression.
        result = simplest_rational(value, working.precision_digits)
        noted = _noted.get()
        if noted is not None:
            noted.add(_fraction(result))
        return result
    if not isinstance(value, sp.Expr) or not value.is_number:
        return value
    noted = _noted.get()
    token = _noted.set(None)
    try:
        rounded = approximated(value, working)
    finally:
        _noted.reset(token)
    if _written_size(value) <= _written_size(rounded):
        return value
    if noted is not None:
        noted.update(_fraction(n) for n in _atoms(rounded, sp.Rational))
    return rounded


def _written_size(value: sp.Basic) -> int:
    """How long `value` is to write: a digit for each digit of its integers, and one
    for each other part."""
    size = 0
    for part in sp.preorder_traversal(value):
        if isinstance(part, sp.Rational):
            size += len(str(abs(part.p))) + (len(str(part.q)) if part.q != 1 else 0)
        else:
            size += 1
    return size


def _vector_carried(vector: Vector, working: Context) -> Vector:
    """`_numbers_carried` of a vector: each element carried.

    A program carries the vector it builds at every step, so a vector that carrying
    leaves as it is, or makes, remembers that, with the digits and the record of
    approximations it was carried for (see `noting_approximations`): carried again
    under them, it is returned at once. Its numbers were recorded when it was first
    carried. A made vector is marked too: carried again, its numbers would only be
    rounded a second time, each by up to its tolerance.
    """
    digits = working.precision_digits
    noted = _noted.get()
    state = vector.__dict__
    known = state.get(_CARRIED_AS)
    if known is not None and known[0] == digits and known[1] is noted:
        return vector
    base = state.get(_CARRIED_FROM)
    if base is not None and base[0][0] == digits and base[0][1] is noted:
        # Made by a list function from a vector carried so, with `added` new.
        if all(_unchanged(element, working) for element in base[1]):
            state[_CARRIED_AS] = (digits, noted)
            return vector
    elements = [_numbers_carried(element, working) for element in vector.args]
    if all(
        new is old or (type(new) is type(old) and new == old)
        for new, old in zip(elements, vector.args, strict=True)
    ):
        state[_CARRIED_AS] = (digits, noted)
        return vector
    made = Vector(*elements)
    made.__dict__[_CARRIED_AS] = (digits, noted)
    return made


def _unchanged(element: sp.Basic, working: Context) -> bool:
    """Whether carrying `element` leaves it as it is (see `_vector_carried`)."""
    new = _numbers_carried(element, working)
    return new is element or (type(new) is type(element) and new == element)


#: The attribute under which a vector that carrying leaves as it is records the digits
#: and the record of approximations it was carried for (see `_vector_carried`).
_CARRIED_AS = "_carried_as"

#: The attribute under which a vector a list function made from another records what
#: that one was carried for, and the elements it added (see `carried_over`).
_CARRIED_FROM = "_carried_from"


def carried_over(made: Vector, source: Vector, added: Sequence[sp.Basic]) -> Vector:
    """`made`, the elements of `source` with `added` put among them, which carrying
    leaves as it is if it leaves `source` and `added` so: carrying it then looks only
    at `added`, so a program that builds or consumes a list element by element carries
    each element once."""
    known = source.__dict__.get(_CARRIED_AS)
    if known is not None:
        made.__dict__[_CARRIED_FROM] = (known, tuple(added))
    return made


def rescaled(expression: sp.Basic, digits: int) -> sp.Basic:
    """`expression` with each product written at the scale of its value, if it isn't.

    The normal form clears the denominators of a sum below the line and takes the
    content of a sum out (see `normal`), and Derive's display of an approximate answer
    is the same form: CURV_FIT.DOC's `3655.76/(9783*t - 11564)`. When the numbers are
    ratios of small integers, as there, that is what an approximate answer shows. When
    they are themselves approximations (a fit to `TAN` data, `PADE(TAN(x), x, 1, 2,
    2)`), each has a long denominator, and clearing them multiplies those together: the
    form is exact, but written to six digits it is `6.27810*10^-12*(-1.44677*10^42*x^2
    + ...)/(-5.83213*10^30*x^2 + ...)`, a scale that cancels between its parts and
    hides the coefficients.

    So a product is rescaled when at least two of its numeric coefficient, a sum above
    the line and a sum below have numbers longer than working precision (see `_long`),
    or a sum above the line does and there is a sum below. Rescaling is exact: each
    long sum above the line, and each sum below, is divided by the coefficient of its
    leading term (making a denominator monic), the coefficient takes the difference,
    and a lone long sum above the line takes the coefficient in. A fit of `(a*x^2 +
    b*x + c)/(x + d)` comes out in that shape, with its four numbers. Short numbers
    keep the normal form, CURV_FIT's included, and so does a single long number:
    `1/(123456789012345*x + 1)` keeps its denominator. Only the form changes, so the
    numbers are still approximated from the exact value.
    """

    def rescale(product: sp.Mul) -> sp.Basic:
        coefficient, rest = product.as_coeff_Mul()
        if not coefficient.is_Rational:
            return product
        factors = list(sp.Mul.make_args(rest))
        above, below = [], []
        for index, factor in enumerate(factors):
            base, exponent = factor.as_base_exp()
            if base.is_Add and exponent.is_Integer:
                (above if exponent > 0 else below).append(index)
        long_above = [i for i in above if _long_sum(factors[i], digits)]
        long_below = [i for i in below if _long_sum(factors[i], digits)]
        longs = _long(coefficient, digits) + bool(long_above) + bool(long_below)
        if longs < 2 and not (long_above and below):
            return product
        for index in long_above + below:
            base, exponent = factors[index].as_base_exp()
            lead = base.as_ordered_terms(order="grlex")[0].as_coeff_Mul()[0]
            if not lead.is_Rational or not lead:
                continue
            factors[index] = sp.Add(*(term / lead for term in base.args)) ** exponent
            coefficient *= lead**exponent
        if coefficient != 1 and len(long_above) == 1 == len(above):
            base = factors[above[0]]
            if base.is_Add:
                factors[above[0]] = sp.Add(*(coefficient * term for term in base.args))
                coefficient = sp.S.One
        return sp.Mul(coefficient, *factors)

    try:
        return expression.replace(lambda part: isinstance(part, sp.Mul), rescale)
    except Exception:
        return expression


def _long(number: sp.Basic, digits: int) -> bool:
    """Whether `number` is a ratio of integers with more digits than working precision.

    Trailing zeros don't count, so a power of ten is short.
    """
    if not isinstance(number, sp.Rational):
        return False

    def length(integer: int) -> int:
        return len(str(abs(integer)).rstrip("0"))

    return length(number.p) + length(number.q) > digits + GUARD


def _long_sum(power: sp.Expr, digits: int) -> bool:
    """Whether the sum `power` raises has a long number as a term's coefficient."""
    total = power.as_base_exp()[0]
    return any(_long(term.as_coeff_Mul()[0], digits) for term in total.args)


def reapproximated(number: sp.Basic, context: Context) -> sp.Basic:
    """`number`, computed from approximate numbers, approximated in turn.

    Such a number is accurate only to the tolerance, and its displayed digits aren't
    meaningful: an approximate `180/pi` is 57.2957 truncated, and times `pi/180` it
    gives 0.9999999 instead of 1. So the whole tolerance is searched (`cut` false).
    """
    digits = context.precision_digits
    value = _evaluated(number, digits + GUARD)
    if value is None or value[1]:
        return approximated(number, context)
    result = simplest_rational(value[0], digits, cut=False)
    noted = _noted.get()
    if noted is not None:
        noted.add(_fraction(result))
    return result


P = ParamSpec("P")
R = TypeVar("R")

#: The approximate numbers produced by the running command, if any.
_noted: ContextVar[set[Fraction] | None] = ContextVar("noted", default=None)
#: Whether the running command drew a random number in [0, 1) (see `drawn_fraction`).
_drawn: ContextVar[list[bool] | None] = ContextVar("drawn", default=None)


def drawn_fraction() -> None:
    """Record that the running command drew a random number in [0, 1), which Derive
    gives as an approximate number (see `drew_fractions`)."""
    drawn = _drawn.get()
    if drawn is not None:
        drawn[0] = True


def drew_fractions() -> bool:
    """Whether the running command drew a random number in [0, 1)."""
    drawn = _drawn.get()
    return bool(drawn and drawn[0])


def noting_approximations(command: Callable[P, R]) -> Callable[P, R]:
    """Decorate `command` to record which numbers in its answer are approximate.

    An approximate number is an ordinary rational, so nothing distinguishes an
    approximated `EXP(100)` (a 44-digit integer with 6 meaningful digits) from an
    exact integer. The notation writes them differently (see `is_approximation`), so
    approximations are recorded while the command runs, including printing. Nested
    commands record into the outermost one's set.
    """

    @functools.wraps(command)
    def noting(*args: P.args, **kwargs: P.kwargs) -> R:
        with _noting():
            return command(*args, **kwargs)

    return noting


@contextmanager
def _noting() -> Iterator[None]:
    if _noted.get() is not None:
        yield
        return
    token = _noted.set(set())
    drawn = _drawn.set([False])
    try:
        yield
    finally:
        _drawn.reset(drawn)
        _noted.reset(token)


def is_approximation(value: Fraction) -> bool:
    """Whether `value` is an approximation produced by the current command."""
    noted = _noted.get()
    return noted is not None and value in noted


def _atoms(expression: sp.Basic, kind: type) -> set:
    try:
        return expression.atoms(kind)
    except Exception:
        return set()


def _fraction(number: sp.Rational) -> Fraction:
    return Fraction(int(number.p), int(number.q))


def _evalf(expression: sp.Basic, digits: int) -> sp.Basic:
    try:
        return expression.evalf(digits, maxn=MAX_DIGITS)
    except Exception:
        return expression


def _rounded(
    expression: sp.Basic, digits: int, approximate: Set[sp.Rational] = frozenset()
) -> sp.Basic:
    """Every number in `expression` replaced by the simplest rational at this precision.

    So `2*y*3` gives `6*y` in every precision mode, and an approximate answer is an
    exact value that `Notation := Rational` can write in full.

    Numbers in `approximate` have already been rounded and are left alone.
    """
    contained = _elementwise(
        expression, lambda element: _rounded(element, digits, approximate)
    )
    if contained is not None:
        return contained

    def walk(part: sp.Basic) -> sp.Basic:
        if isinstance(part, sp.Rational | sp.Float):
            return part if part in approximate else simplest_rational(part, digits)
        if not isinstance(part, sp.Basic) or not part.args:
            return part
        args = tuple(walk(arg) for arg in part.args)
        if args == part.args:
            return part
        if isinstance(part, Relational):
            # Rebuilt unevaluated, or assumptions would decide it: `x = -0.900968 +
            # 0.433883*#i` for real `x` is a solution soLve found, not `false`.
            return part.func(*args, evaluate=False)
        return part.func(*args)

    try:
        return walk(expression)
    except Exception:
        return expression


def _elementwise(
    container: object, each: Callable[[sp.Basic], sp.Basic]
) -> sp.Basic | None:
    """`container` with `each` applied to its elements, or None if it isn't one.

    This is the one place that knows how containers are held. A sympy matrix, which
    appears only in linear algebra on the way to an answer, isn't a `sp.Basic` and has
    no `args` to walk, so its numbers are reached only from here; a `Vector`, which is
    how vectors and matrices are held otherwise, is an ordinary function head and needs
    no special case.
    """
    if isinstance(container, sp.MatrixBase):
        return container.applyfunc(each)
    return None


def simplest_rational(number: sp.Basic, digits: int, cut: bool = True) -> sp.Rational:
    """`simplest` for a sympy rational, or a float produced by evaluation.

    A float is read as its decimal digits, not its stored binary fraction, which
    differs past the float's precision: `SQRT(10^200 + 1)` evaluates to a float
    showing `1.0000000000e+100` but stored as `10^100 - 1.73*10^87`. Both are within
    tolerance, and choosing the nearest candidate to the stored value would keep the
    binary error in a number later used exactly.
    """
    if isinstance(number, sp.Float):
        value = Fraction(Decimal(str(number)))
        return sp.Rational(simplest(value, digits, cut))
    rational = sp.Rational(number)
    numerator, denominator = int(rational.p), int(rational.q)
    if numerator.bit_length() + denominator.bit_length() <= _REMEMBERED_BITS:
        return sp.Rational(*_simplest_ratio(numerator, denominator, digits, cut))
    value = _shortened(numerator, denominator, digits)
    return sp.Rational(simplest(value, digits, cut))


@functools.lru_cache(maxsize=1 << 16)
def _simplest_ratio(
    numerator: int, denominator: int, digits: int, cut: bool
) -> tuple[int, int]:
    """`simplest` of `numerator/denominator`, as a numerator and a denominator. A
    program carrying a vector rounds the same numbers at every step, so they are
    remembered."""
    found = simplest(_shortened(numerator, denominator, digits), digits, cut)
    return found.numerator, found.denominator


#: The longest rational, in bits of numerator and denominator, whose rounding is
#: remembered.
_REMEMBERED_BITS = 4096


#: Digits beyond the tolerance kept when shortening a long rational before
#: approximating it.
SHORTENED = 30


def _shortened(numerator: int, denominator: int, digits: int) -> Fraction:
    """`numerator/denominator`, or a nearby rational with fewer digits.

    A rational with hundreds of thousands of digits (`(1 + 10^-5)^10^5` computed
    exactly) makes each step of finding the simplest rational take seconds, though
    nearly all its digits are far below the tolerance. They are dropped, leaving a
    binary fraction within `10^-(digits + SHORTENED)` of the value (relatively, and
    absolutely for large values), indistinguishable at a tolerance of `10^-(digits +
    1)`. The whole part is kept exactly, so rounding a large value to an integer is
    unaffected.

    Only long denominators are shortened; short ones are cheap, and integers stay
    exact.
    """
    bits = int((digits + SHORTENED) * 3.33) + 64
    if denominator.bit_length() <= 2 * bits:
        return Fraction(numerator, denominator)
    magnitude = abs(numerator).bit_length() - denominator.bit_length()
    shift = bits + max(0, -magnitude)
    value = Fraction((abs(numerator) << shift) // denominator, 1 << shift)
    return value if numerator >= 0 else -value


def _approximated(
    expression: sp.Basic,
    digits: int,
    approximate: set[sp.Rational] | None = None,
    rounding: bool = False,
) -> sp.Basic:
    """Approximate each number in `expression` as a whole, outermost first.

    A number is computed as if exactly and only then rounded, so every digit is
    correct: `(12^(1/1000)*w)^1000`, with `w` a root of unity in radicals, is 12,
    whereas rounding the pieces first would multiply their error a thousandfold.

    Going outermost first approximates whole numbers rather than pieces, and the
    numeric terms of a partly numeric sum or product are combined first: `x +
    SQRT(1000003) - SQRT(1000002)` is `x + 0.000499999`, not two rounded roots
    subtracted.

    Each number is replaced by a rational. `approximate` collects them so the final
    rounding knows what is already rounded. With `rounding`, the final rounding will
    round every rational too (Approximate precision), so a number is approximated
    without first asking whether it is rational (see `_needs_digits`).
    """

    def approximation(part: sp.Basic) -> sp.Basic:
        value = _approximate(part, digits)
        if approximate is not None and value is not part:
            approximate.add(value)
        return value

    def walk(node: sp.Basic) -> sp.Basic:
        contained = _elementwise(node, walk)
        if contained is not None:
            return contained
        if _needs_digits(node, rounding):
            return approximation(node)
        if not isinstance(node, sp.Basic) or not node.args:
            return node
        if isinstance(node, sp.Add | sp.Mul):
            numeric = [arg for arg in node.args if arg.is_number]
            if 1 < len(numeric) < len(node.args):
                rest = [arg for arg in node.args if not arg.is_number]
                return node.func(walk(node.func(*numeric)), *map(walk, rest))
        args = tuple(map(walk, node.args))
        if args == node.args:
            return node
        if isinstance(node, Relational):
            # Keep the relation rather than letting assumptions decide it: `x =
            # 1.73205*#i` for real `x` is a solution soLve found, not `false`.
            return node.func(*args, evaluate=False)
        return node.func(*args)

    try:
        return walk(expression)
    except Exception:
        return expression


def _needs_digits(expression: sp.Basic, rounding: bool = False) -> bool:
    """Whether this number must be evaluated to be written down.

    Every number must, unless sympy proves it rational.

    Testing `is_rational is not True` rather than `is_irrational` matters: the latter
    is often unknown (a product of surds can be rational), so it would leave unproven
    cases exact, and a wholly numeric expression could keep radicals after
    approximation.

    Finiteness is tested the same way: only numbers known to be infinite are
    refused. `LN(0)` has no digits, but `SI(2)`, `EI(2)` and `ERFI(2)` are finite
    though sympy can't prove it. Unevaluated integrals and sums are the same case,
    and Derive approximates them numerically. This is safe because `_approximate`
    accepts only a float or rational result, so `nan` or a non-converging quadrature
    is left alone.

    With `rounding` (see `_approximated`) every number that isn't a rational or a
    decimal is evaluated, without asking whether it is rational or finite: the
    questions are answered by evaluating it, which on an exact radical expression may
    take far longer than the approximation itself, a rational number is rounded to the
    same digits either way, and an infinite one is left as it is by `_approximate`.
    """
    if rounding:
        return expression.is_number and not isinstance(
            expression, (sp.Rational, sp.Float)
        )
    return (
        expression.is_number
        and expression.is_rational is not True
        and expression.is_finite is not False
        and not isinstance(expression, sp.Float)
    )


def _approximate(part: sp.Basic, digits: int) -> sp.Basic:
    """The approximation of `part`, or `part` itself if it can't be evaluated.

    A non-real number such as `SQRT(-2)` or `(1 + #i)^10` is approximated as a
    rational real part plus `#i` times a rational imaginary part, as in Derive.
    """
    value = _evaluated(part, digits + GUARD)
    if value is None:
        return part
    real, imaginary = value
    if not imaginary:
        return simplest_rational(real, digits)
    return simplest_rational(real, digits) + simplest_rational(imaginary, digits) * sp.I


def _evaluated(part: sp.Basic, digits: int) -> tuple[sp.Basic, sp.Basic] | None:
    """The real and imaginary parts of `part`, each correct to `digits`.

    Sympy raises its working precision up to `MAX_DIGITS` until the digits are
    correct. A value still indistinguishable from zero comes back with no significant
    digits and is taken as zero.

    Sympy's precision is relative to the whole complex value, so a much smaller part
    may be noise. If it is below the precision it is taken as zero, making
    `(12^(1/1000)*w)^1000` 12 rather than `12 + 5.06*10^-17*#i`. Otherwise the value
    is re-evaluated with enough extra digits for that part to be correct as well.
    """
    part = _with_integrals_by_quadrature(part, digits)
    value = _evalf(part, digits)
    parts = _parts(value)
    if parts is None:
        return None
    if (
        _unresolved_imaginary(value)
        and part.has(sp.I)
        and sp.count_ops(part) <= _TINY_SIZE
    ):
        parts = _imaginary_resolved(part, digits, parts)
    composed = _composed(part)
    if composed:
        parts = _settled_parts(part, digits, parts)
    size = max(abs(parts[0]), abs(parts[1]))
    if not size:
        return parts
    smaller = min(abs(parts[0]), abs(parts[1]))
    if smaller and smaller < size * sp.Rational(1, 10**digits):
        parts = _resolved(part, digits, parts, size, smaller)
    elif smaller:
        extra = int(sp.ceiling(sp.log(size / smaller, 10)))
        if extra > 0:
            parts = _parts(_evalf(part, digits + extra)) or parts
            if composed:
                parts = _settled_parts(part, digits + extra, parts)
    return parts


def _unresolved_imaginary(value: sp.Basic) -> bool:
    """Whether `value`'s imaginary part came back with no significant digits: too small
    against the real part for the precision, whether or not it is zero."""
    try:
        imaginary = value.as_real_imag()[1]
    except Exception:
        return False
    return isinstance(imaginary, sp.Float) and imaginary._prec <= 1


#: How many digits beyond the precision an imaginary part may lie below the real part
#: and still be found.
_TINY_REACH = 60

#: The largest value, in operations, whose unresolved imaginary part is looked for on
#: its own: in a large one, such as a real value written with Cardano's radicals, the
#: search for a part that is zero costs seconds.
_TINY_SIZE = 80


def _imaginary_resolved(
    part: sp.Basic, digits: int, parts: tuple[sp.Basic, sp.Basic]
) -> tuple[sp.Basic, sp.Basic]:
    """The parts of `part` with its imaginary part evaluated on its own, to its own
    digits, where evaluating the whole value left it unresolved: `EXP(#i/10^30)` is `1
    + 10^(-30)*#i`, not 1. A part still unresolved is zero."""
    try:
        imaginary = sp.im(part).evalf(digits, maxn=digits + _TINY_REACH)
    except Exception:
        return parts
    if isinstance(imaginary, sp.Float) and imaginary._prec > 1 and imaginary != 0:
        return parts[0], imaginary
    return parts


def _resolved(
    part: sp.Basic,
    digits: int,
    parts: tuple[sp.Basic, sp.Basic],
    size: sp.Basic,
    smaller: sp.Basic,
) -> tuple[sp.Basic, sp.Basic]:
    """The parts of a value one of which is below the precision of the other: evaluated
    again with digits enough for the smaller one, which is kept if it comes out the
    same, as a part of the value does (`p(-0.846 - 6.2*10^(-53)*#i)` has an imaginary
    part of `-8.5*10^(-52)`), and taken as zero if it moves, as noise does."""
    extra = int(sp.ceiling(sp.log(size / smaller, 10))) + digits
    if extra <= digits + _TINY_REACH:
        again = _parts(_evalf(part, digits + extra))
        if again is not None:
            small = 0 if abs(parts[0]) < abs(parts[1]) else 1
            if again[small] and abs(again[small] - parts[small]) <= abs(
                again[small]
            ) * sp.Rational(1, 1000):
                return again
    return tuple(p if abs(p) == size else sp.Integer(0) for p in parts)


def _composed(part: sp.Basic) -> bool:
    """Whether `part` applies a function to a value of a function.

    Sympy tracks the error of arithmetic, but takes a function at a value only as
    precise as it was given: `ATAN(TAN(11 + 30*#i))` takes the arctangent next to its
    pole at `#i`, where the digits `TAN` gave are far too few.
    """
    return any(
        argument.has(sp.Function)
        for call in _atoms(part, sp.Function)
        for argument in call.args
    )


def _settled_parts(
    part: sp.Basic, digits: int, parts: tuple[sp.Basic, sp.Basic]
) -> tuple[sp.Basic, sp.Basic]:
    """The parts of `part`, evaluated again at more digits until two evaluations agree
    to `digits`. At `MAX_DIGITS` the last evaluation is kept.
    """
    working = digits
    while working < MAX_DIGITS:
        working = min(2 * working + GUARD, MAX_DIGITS)
        again = _parts(_evalf(part, working))
        if again is None:
            return parts
        if _agree(parts, again, digits):
            return parts
        parts = again
    return parts


def _agree(
    one: tuple[sp.Basic, sp.Basic], other: tuple[sp.Basic, sp.Basic], digits: int
) -> bool:
    """Whether two evaluations of a complex number agree to `digits` significant
    digits of its magnitude."""
    size = max(abs(part) for part in (*one, *other))
    tolerance = size * sp.Rational(1, 10**digits)
    return all(abs(a - b) <= tolerance for a, b in zip(one, other))


#: Below this, an exponent is taken to give 0 when integrating over an infinite range.
#: Far smaller than any digit an answer shows, and still cheap to compute: `#e^-x` for
#: `x` of `10^434` takes milliseconds, while `10^43429` takes minutes.
_VANISHING = -(10**6)


def _with_integrals_by_quadrature(part: sp.Basic, digits: int) -> sp.Basic:
    """`part` with each numeric definite integral replaced by its value by quadrature.

    Sympy's own evaluation raises the precision until the digits agree, which through a
    singularity never ends: ZETA.MTH's `DILOG(-1)`, `INT(LN(t)/(1 - t), t, 1, -1)`,
    passes the branch point of the logarithm at 0. Here the range is cut where the
    integrand may be singular (see `_breaks`), and the quadrature is done once at
    working precision. An integral whose error estimate is not within the digits asked
    for is left to sympy.

    Quadrature over an infinite range samples the integrand far out, and for
    BESSEL.MTH's `INT(#e^(-z*COSH(t))*COSH(n*t), t, 0, inf)` that means `#e^-x` for `x`
    so large that computing it (to a precision that grows with `x`) never ends, though
    the value is far below any digit shown. So an exponential gives 0 past
    `_VANISHING`.
    """

    def numeric(found: sp.Basic) -> bool:
        if not isinstance(found, sp.Integral) or len(found.limits) != 1:
            return False
        return len(found.limits[0]) == 3 and found.is_number

    try:
        if not part.has(sp.Integral):
            return part
        return part.replace(
            numeric, lambda found: _quadrature(found, digits), simultaneous=False
        )
    except Exception:
        return part


def by_quadrature(integral: sp.Integral, digits: int) -> sp.Basic:
    """The value of a numeric definite integral by quadrature, as a decimal of `digits`
    digits, or the integral itself if the error estimate is not within them (see
    `_with_integrals_by_quadrature`)."""
    return _quadrature(integral, digits)


def _quadrature(integral: sp.Integral, digits: int) -> sp.Basic:
    def end(value: sp.Basic) -> mpmath.mpf:
        if value.is_infinite:
            return mpmath.inf if value.is_extended_positive else -mpmath.inf
        return mpmath.mpf(str(value.evalf(digits + 2 * GUARD)))

    def exp(x):
        return mpmath.mpf(0) if mpmath.re(x) < _VANISHING else mpmath.exp(x)

    variable, low, high = integral.limits[0]
    try:
        integrand = sp.lambdify(
            variable, integral.function, modules=[{"exp": exp}, "mpmath"]
        )
        with mpmath.workdps(digits + GUARD):
            ends = [end(low), end(high)]
            inner = _breaks(integral.function, variable, *ends)
            if ends[0] > ends[1]:
                inner.reverse()
            value, error = mpmath.quad(integrand, [ends[0], *inner, ends[1]], error=True)
            if not mpmath.isfinite(value) or error > abs(value) * mpmath.mpf(10) ** (
                -digits
            ):
                return integral
            real, imaginary = mpmath.re(value), mpmath.im(value)
        number = sp.Float(real, digits)
        return number + sp.Float(imaginary, digits) * sp.I if imaginary else number
    except Exception:
        return integral


def _breaks(
    integrand: sp.Basic, variable: sp.Basic, low: mpmath.mpf, high: mpmath.mpf
) -> list[mpmath.mpf]:
    """The points strictly between the ends, in increasing order, where the integrand
    may be singular: real roots of a polynomial argument of a logarithm or absolute
    value, or of a base with a fractional or negative exponent. Other singularities
    aren't looked for."""
    bottom, top = sorted((low, high))
    found = set()
    arguments = [call.args[0] for call in integrand.atoms(sp.log, sp.Abs)]
    arguments += [
        power.base
        for power in integrand.atoms(sp.Pow)
        if not (power.exp.is_Integer and power.exp > 0)
    ]
    for argument in arguments:
        if not argument.has(variable) or argument.is_polynomial(variable) is not True:
            continue
        try:
            roots = sp.Poly(argument, variable).real_roots()
        except Exception:
            continue
        for root in roots:
            point = mpmath.mpf(str(sp.N(root, mpmath.mp.dps + GUARD)))
            if bottom < point < top:
                found.add(point)
    return sorted(found)


def _parts(value: sp.Basic) -> tuple[sp.Basic, sp.Basic] | None:
    """`value`'s real and imaginary parts, if both are floats or rationals."""
    real, imaginary = value.as_real_imag()
    parts = []
    for number in (real, imaginary):
        if isinstance(number, sp.Float):
            # No significant digits: sympy couldn't distinguish the value from zero.
            parts.append(sp.Integer(0) if number._prec <= 1 else number)
        elif number.is_Rational:
            parts.append(number)
        else:
            return None
    return parts[0], parts[1]
