"""What an approximate number is: the simplest rational the precision allows.

Derive's approximate arithmetic is exact arithmetic over approximate numbers,
and an approximate number is a *rational* - the simplest one that stands for
the value to the precision asked for. Approximating π to six digits gives
355/113, which is why `Notation := Rational` shows exactly that, and why an
approximate answer is as exact a value as any other: it is just a different
value from the one it approximates.

Three consequences the original is recognisable by:

* A number that needs no digits does not get any. The simplest rational
  standing in for 5 is 5, and the one standing in for two thirds is two
  thirds - so `2/3` approximated is `2/3`, and shown to six digits it is
  `0.666666`, cut rather than rounded up to `0.666667`.
* A value keeps the digits it is shown by. `SQRT(3)` is `1.73205` and not the
  `1.73204` that a simpler rational a little below it would show.
* Rounding happens to the numbers going in, not to the arithmetic between
  them. The manual's own exercise turns on it: the two fractions of
  `SQRT(3422357/2313 - 1140443/771)` are rounded before they are subtracted,
  so approximate mode does not reach the exactly 2/3 that Mixed mode does.

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

The answer is the simplest rational within the tolerance, found by walking the
continued fraction of the interval's ends: at each step the whole part they
share is the next term of the answer, and the walk stops at the first step
where an integer falls between them. That is what "simplest" means here, and
it is why π comes out 355/113 - not because a coarse tolerance let it, but
because π is approximated by 355/113 far better than a denominator of 113 has
any right to be.
"""

from __future__ import annotations

from fractions import Fraction

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
