"""What an approximate number is: the simplest rational the precision allows.

Derive's approximate arithmetic is exact arithmetic over approximate numbers,
and an approximate number is a *rational* - the simplest one that shows the
same digits as the value it stands for. Approximating π to six digits gives
355/113, which is why `Notation := Rational` shows exactly that, and why an
approximate answer is as exact a value as any other: it is just a different
value from the one it approximates.

Three consequences the original is recognisable by:

* A number that needs no digits does not get any. The simplest rational
  showing the digits of 5 is 5, and the one showing the digits of two thirds
  is two thirds - so `2/3` approximated is `2/3`, and shown to six digits it is
  `0.666666`, cut rather than rounded up to `0.666667`.
* What is shown does not move. The approximation shows the digits the value
  showed, which is why `SQRT(3)` is `1.73205` and not the `1.73204` that a
  simpler rational a little below it would show.
* Rounding happens to the numbers going in, not to the arithmetic between
  them. The manual's own exercise turns on it: the two fractions of
  `SQRT(3422357/2313 - 1140443/771)` are rounded before they are subtracted,
  so approximate mode does not reach the exactly 2/3 that Mixed mode does.

The candidates are the value's continued fraction convergents, which are the
best rational approximations there are - each is closer to the value than any
rational with a smaller denominator - and the first of them showing the right
digits is the answer. The walk always ends: the last convergent of a rational
is the rational itself, which shows its own digits.
"""

from __future__ import annotations

from fractions import Fraction

from rederive.engine.notation import exponent

#: Digits carried past the ones asked for while an irrational is evaluated, so
#: that the digits asked for are the value's own rather than a rounding of it.
GUARD = 5


def simplest(value: Fraction, digits: int) -> Fraction:
    """The simplest rational showing the same `digits` significant digits."""
    if not value:
        return Fraction(0)
    sign = -1 if value < 0 else 1
    value = abs(value)
    step = Fraction(10) ** (exponent(value) - digits + 1)
    shown = value // step * step
    rest = value
    # The convergents, by the usual recurrence over the continued fraction:
    # each is the one before it times the next whole part, plus the one before
    # that. The pair standing in for the two that come before the first is
    # what makes the first convergent the value's whole part.
    numerator, denominator = 1, 0
    before = (0, 1)
    while True:
        whole = rest.numerator // rest.denominator
        numerator, before_numerator = whole * numerator + before[0], numerator
        denominator, before_denominator = whole * denominator + before[1], denominator
        before = (before_numerator, before_denominator)
        convergent = Fraction(numerator, denominator)
        if shown <= convergent < shown + step:
            return sign * convergent
        rest -= whole
        if not rest:
            return sign * value
        rest = 1 / rest
