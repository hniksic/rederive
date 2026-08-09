"""What an approximate number is: the simplest rational the precision allows.

Derive's approximate arithmetic is exact arithmetic over approximate numbers,
and an approximate number is a *rational* - the simplest one that stands for
the value to the precision asked for. Approximating π to six digits gives
355/113, which is why `Notation := Rational` shows exactly that, and why an
approximate answer is as exact a value as any other: it is just a different
value from the one it approximates.

Three consequences:

* A number that needs no digits does not get any. The simplest rational
  standing in for 5 is 5, and the one standing in for two thirds is two
  thirds - so `2/3` approximated is `2/3`, and shown to six digits it is
  `0.666666`, cut rather than rounded up to `0.666667`.
* A value keeps the digits it is shown by. `SQRT(3)` is `1.73205` and not the
  `1.73204` that a simpler rational a little below it would show.
* Rounding reaches the numbers that need digits, and not the arithmetic
  between them. What gets a rational to stand for it is an irrational; a
  quotient of two whole numbers is an exact value already and stays the one it
  is. So `10^7*pi` is `10^7*355/113` worked out and then rounded, which is
  `31415929` and not the `31415900` that six digits of pi multiplied out
  would leave.

Where this parts from the manual: 3.8 p.46 turns on the two fractions of
`SQRT(3422357/2313 - 1140443/771)` being rounded before they are subtracted,
which is what would leave Approximate mode short of the exactly 2/3 that Mixed
mode reaches. Both are quotients of whole numbers, so neither is rounded here
and both modes answer 2/3. Rounding every step instead does not reach the
manual's digits either, and what its 0.666622 really records is binary
floating point; `test_manual` holds the case and says so.

"To the precision asked for" is a tolerance, and it is one digit tighter than
the digits shown. The looser reading - the simplest rational that displays the
same digits - admits an error of a whole unit in the last digit shown, and
picks deliberately the coarsest value that gets away with it. That is the
worst possible choice for a number that is about to be computed with, because
the arithmetic around an approximate number is exact and so carries the whole
of that error forward. `SQRT(1000003) - SQRT(1000002)` is what it costs: both
roots display `1000.00`, the coarsest rational displaying that is 1000, and a
difference of about 0.0005 comes out 0 with nothing to say it did. The guard
digit is what keeps the substituted value close enough that a cancellation
between two of them still means something.

`approximated` is the other half: what a precision mode does to a whole
expression, which is this rule applied to every number in it that needs digits
and to nothing else.

The answer is the simplest rational within the tolerance, found by walking the
continued fraction of the interval's ends: at each step the whole part they
share is the next term of the answer, and the walk stops at the first step
where an integer falls between them. That is what "simplest" means here, and
it is why π comes out 355/113 - not because a coarse tolerance let it, but
because π is approximated by 355/113 far better than a denominator of 113 has
any right to be.
"""

from __future__ import annotations

from collections.abc import Set
from fractions import Fraction

import sympy as sp

from rederive.engine.context import Context, Precision

__all__ = ["GUARD", "approximated", "simplest"]

#: Digits carried past the ones asked for while an irrational is evaluated, so
#: that the digits asked for are the value's own rather than a rounding of it.
GUARD = 5


def simplest(value: Fraction, digits: int) -> Fraction:
    """The simplest rational standing in for `value` at `digits` of precision."""
    if not value:
        return Fraction(0)
    sign = -1 if value < 0 else 1
    value = abs(value)
    tolerance = value * Fraction(1, 10 ** (digits + 1))
    low, high = value - tolerance, value + tolerance
    simplest_here = _simplest_between(low, high)
    # Simplest fixes the denominator, not the whole answer: where the interval
    # holds several rationals over that denominator, they are equally simple
    # and the one nearest the value is the one it stands for. `10^7*pi` turns
    # on it, the interval there holding six whole numbers of which 31415929 is
    # the value's own.
    denominator = simplest_here.denominator
    nearest = Fraction(round(value * denominator), denominator)
    return sign * (nearest if low <= nearest <= high else simplest_here)


def _simplest_between(low: Fraction, high: Fraction) -> Fraction:
    """The simplest rational in `[low, high]`, which must be positive.

    The whole parts of the two ends agree until the interval is wide enough to
    hold an integer, and everything they agree on is forced: any rational
    between them starts the same way. So each shared whole part is written
    down and the walk moves on to the reciprocals of what is left, where the
    ends trade places. The first integer to fall between them ends it, and the
    smallest such integer is the simplest thing the interval can hold.
    """
    whole = low.numerator // low.denominator
    if low.denominator == 1:
        return low
    if whole + 1 <= high:
        return Fraction(whole + 1)
    return whole + 1 / _simplest_between(1 / (high - whole), 1 / (low - whole))


def approximated(expression: sp.Basic, context: Context) -> sp.Basic:
    """What the precision mode does to an expression.

    The last step of every command's pipeline. Factor simplifies exactly, then
    factors, and only then rounds - so that radical factoring reaches `SQRT(2)`
    first and shows `1.41421` because of this, rather than factoring a number
    that has already been rounded.

    It is also what an iteration does to each iterate it hands on, an iterate
    being a value in its own right and the next step's number going in.
    """
    digits = context.precision_digits
    match context.precision:
        case Precision.APPROXIMATE:
            # The irrational parts are approximated, then what they make of
            # each other is exact, and the answer is approximated once more:
            # `10^7·π` is `10^7·355/113` worked out and then rounded, which is
            # `31415929` and not the `31415900` that six digits of the product
            # would leave.
            #
            # What the first pass already rounded is left alone by the second,
            # since rounding an approximate number again is not idempotent: the
            # tolerance is measured from wherever the value now stands, so a
            # second pass may reach a simpler rational a whole tolerance away
            # and cost the last digit shown. Only what the exact arithmetic
            # between the approximate numbers made of them is rounded here.
            approximate: set[sp.Rational] = set()
            return _rounded(
                _approximated(expression, digits, approximate), digits, approximate
            )
        case Precision.MIXED:
            return _approximated(expression, digits)
    return expression


def _evalf(expression: sp.Basic, digits: int) -> sp.Basic:
    try:
        return expression.evalf(digits)
    except Exception:
        return expression


def _rounded(
    expression: sp.Basic, digits: int, approximate: Set[sp.Rational] = frozenset()
) -> sp.Basic:
    """Every number in `expression` as the simplest rational at this precision.

    Which is what Derive's approximate numbers are, so this is what makes
    `2*y*3` come out `6*y` in every precision mode - only the numbers that
    actually need digits get them - and what leaves an approximate answer an
    exact value that `Notation := Rational` can write out in full.

    A number in `approximate` is one this precision has already rounded, and it
    stands as it is.
    """
    try:
        return expression.replace(
            lambda part: isinstance(part, sp.Rational | sp.Float)
            and part not in approximate,
            lambda part: _simplest(part, digits),
            simultaneous=False,
        )
    except Exception:
        return expression


def _simplest(number: sp.Basic, digits: int) -> sp.Rational:
    value = sp.Rational(number)
    return sp.Rational(simplest(Fraction(value.p, value.q), digits))


def _approximated(
    expression: sp.Basic, digits: int, approximate: set[sp.Rational] | None = None
) -> sp.Basic:
    """Approximate the irrational operations, keep the rational ones exact.

    Innermost first, so that a rational subexpression is computed before
    anything near it is rounded: `SQRT(3422357/2313 - 1140443/771)` is exactly
    `2/3` in Mixed mode, where Approximate rounds the two fractions on the way
    in and never reaches it.

    What each irrational is replaced by is a rational, since that is what an
    approximate number is; the arithmetic around it stays exact. `approximate`
    collects those rationals for the caller, which is how the rounding of the
    finished answer knows what it has already rounded.
    """

    def approximation(part: sp.Basic) -> sp.Basic:
        value = _approximate(part, digits)
        if approximate is not None and value is not part:
            approximate.add(value)
        return value

    try:
        return expression.replace(_needs_digits, approximation, simultaneous=False)
    except Exception:
        return expression


def _needs_digits(expression: sp.Basic) -> bool:
    """Whether this number has to be worked out before it can be written down.

    Every number does, unless sympy can show that it is rational: a rational
    is already what an approximate number is, and anything else stands for
    digits nothing has computed yet.

    The question is which way round to ask it, and the answer is not the
    obvious one. `is_irrational` has a third answer besides yes and no, and
    that third answer is the common one - sympy leaves a product of surds open,
    since irrationals can multiply to a rational - so asking for a proof of
    irrationality leaves everything unprovable exact, and approximating a
    wholly numeric expression can then answer with radicals still standing in
    it. Asking for a proof of rationality instead sends the unprovable cases
    the other way, and a number can only come out a number.

    Finiteness is asked the same way round and for the same reason. Only what
    is known to be infinite is refused: `LN(0)` has no digits and never will,
    while `SI(2)`, `EI(2)` and `LI(2)` are finite numbers sympy simply does not
    prove finite, and demanding the proof left every one of them written as
    itself. An unevaluated integral or sum is the same case, and approximating
    one numerically is what the original does with what it cannot do exactly.
    Nothing is risked by asking loosely here: `_approximate` takes only what
    comes back a float or a rational, so a value that will not evaluate - `nan`,
    or a quadrature that does not converge - is left standing anyway.
    """
    return (
        expression.is_number
        and expression.is_rational is not True
        and expression.is_finite is not False
        and not isinstance(expression, sp.Float)
    )


def _approximate(part: sp.Basic, digits: int) -> sp.Basic:
    """The approximate number `part` stands for, or `part` where there is none.

    A number whose value is not real - `SQRT(-2)` and everything else that
    evaluates with an `I` in it - has no rational standing in for it, and is
    left as it is rather than failing the approximation of everything around
    it.
    """
    value = _evalf(part, digits + GUARD)
    if not isinstance(value, sp.Float) and not value.is_Rational:
        return part
    return _simplest(value, digits)
