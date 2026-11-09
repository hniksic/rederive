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
from collections.abc import Callable, Iterator, Set
from contextlib import contextmanager
from contextvars import ContextVar
from decimal import Decimal
from fractions import Fraction
from typing import ParamSpec, TypeVar

import sympy as sp
from sympy.core.relational import Relational

from rederive.engine.context import Context, Precision
from rederive.engine.notation import exponent

__all__ = [
    "GUARD",
    "approximated",
    "is_approximation",
    "noting_approximations",
    "reapproximated",
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


def simplest(value: Fraction, digits: int, cut: bool = True) -> Fraction:
    """The simplest rational representing `value` at `digits` of precision.

    The result must truncate to the value's own digits. The tolerance alone would
    allow a simpler rational just below, so typing `3.14159` would show `3.14158`, or
    just above, so `APPROX(0.12399, 3)` would show `0.124`. So the search interval is
    the tolerance narrowed to numbers truncating to the value's digits, which always
    contains the value.

    With `cut` false the whole tolerance is searched, for values computed from
    approximate numbers whose digits aren't meaningful; see `reapproximated`.
    """
    if not value:
        return Fraction(0)
    sign = -1 if value < 0 else 1
    value = abs(value)
    tolerance = value * Fraction(1, 10 ** (digits + 1))
    unit = _TEN ** (exponent(value) - digits + 1)
    shown = value // unit * unit
    low, high = value - tolerance, value + tolerance
    if cut:
        low, high = max(low, shown), min(high, shown + unit)
    # The upper end belongs to the next digit, so it is excluded.
    open_high = cut and high == shown + unit
    simplest_here = _simplest_between(low, high, high_open=open_high)
    # Simplest fixes only the denominator: if the interval holds several rationals
    # over it, pick the nearest. For `10^7*pi` the interval holds six integers, and
    # 31415929 is the right one.
    denominator = simplest_here.denominator
    nearest = Fraction(round(value * denominator), denominator)
    inside = low <= nearest and (nearest < high if open_high else nearest <= high)
    return sign * (nearest if inside else simplest_here)


def _simplest_between(
    low: Fraction, high: Fraction, low_open: bool = False, high_open: bool = False
) -> Fraction:
    """The simplest rational between `low` and `high`, which must be positive.

    Each end is included unless open. While both ends share a whole part, any
    rational between them has it too, so it is recorded and the walk continues with
    the reciprocals of the remainders (swapping the ends). Once an integer falls
    between them, the smallest one is the simplest choice.
    """
    whole = low.numerator // low.denominator
    if low.denominator == 1 and not low_open:
        return low
    if whole + 1 < high or (whole + 1 == high and not high_open):
        return Fraction(whole + 1)
    if low == whole:
        # The remainder reaches zero, so its reciprocal is unbounded above; take the
        # smallest admissible integer.
        bound = 1 / (high - whole)
        floor = bound.numerator // bound.denominator
        return whole + Fraction(1, floor + 1 if high_open or floor < bound else floor)
    return whole + 1 / _simplest_between(
        1 / (high - whole), 1 / (low - whole), high_open, low_open
    )


def approximated(expression: sp.Basic, context: Context) -> sp.Basic:
    """`expression` with the precision mode applied.

    The last step of every command's pipeline. Factor simplifies exactly, factors,
    and only then rounds, so radical factoring reaches `SQRT(2)` before it is shown as
    `1.41421`.

    Iterations also apply it to each iterate before the next step.
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
            result = _rounded(
                _approximated(expression, digits, approximate), digits, approximate
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
    try:
        yield
    finally:
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
    try:
        return expression.replace(
            lambda part: isinstance(part, sp.Rational | sp.Float)
            and part not in approximate,
            lambda part: simplest_rational(part, digits),
            simultaneous=False,
        )
    except Exception:
        return expression


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
    else:
        rational = sp.Rational(number)
        value = _shortened(rational.p, rational.q, digits)
    return sp.Rational(simplest(value, digits, cut))


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
    expression: sp.Basic, digits: int, approximate: set[sp.Rational] | None = None
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
    rounding knows what is already rounded.
    """

    def approximation(part: sp.Basic) -> sp.Basic:
        value = _approximate(part, digits)
        if approximate is not None and value is not part:
            approximate.add(value)
        return value

    def walk(node: sp.Basic) -> sp.Basic:
        if _needs_digits(node):
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


def _needs_digits(expression: sp.Basic) -> bool:
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
    """
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
    value = _evalf(part, digits)
    parts = _parts(value)
    if parts is None:
        return None
    size = max(abs(parts[0]), abs(parts[1]))
    if not size:
        return parts
    smaller = min(abs(parts[0]), abs(parts[1]))
    if smaller and smaller < size * sp.Rational(1, 10**digits):
        parts = tuple(p if abs(p) == size else sp.Integer(0) for p in parts)
    elif smaller:
        extra = int(sp.ceiling(sp.log(size / smaller, 10)))
        if extra > 0:
            parts = _parts(_evalf(part, digits + extra)) or parts
    return parts


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
