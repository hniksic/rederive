"""Derive's four numerical output styles, as rules over an exact value.

`Options Notation` says how a number the engine works out is written: as the
ratio it is worth, as a decimal point numeral, in scientific notation, or in
whichever of the three suits the number. The style reaches a number wherever
one sits, so `x/3` simplified under Decimal is `0.333333·x`.

Three rules hold across the styles, and all three come from the original:

* A decimal numeral is *cut*, never rounded, and never into its whole part:
  two thirds is `0.666666` and `1234567.891` is `1234567.8`. What is cut to is
  significant digits, so the zeros a small number leads with are free -
  `1/300000` keeps six digits of its own in `0.00000333333`.
* The cut neither pads nor strips. A tenth to six digits is `0.1` because the
  expansion ends there, while `0.10000000000000001` is `0.100000` because it
  does not: the zeros stand for the digits that follow.
* Scientific notation is for extreme magnitudes only. A number worth at least
  a ten-thousandth and less than ten thousand is written as a plain decimal
  numeral whatever the digit count, which is why `1234.56` is not `1.23456·10³`.

What the styles do *not* reach is a numeral the user wrote. `12345.6789` typed
on the author line is shown as `12345.6` under every style, cut by
`NotationDigits` alone; the display layer does that, and the value behind it is
untouched either way.
"""

from __future__ import annotations

from fractions import Fraction

#: Neither half of a ratio may be bigger than this for Mixed to keep it a
#: ratio. The original's line falls exactly between 144 and 145: it writes
#: `1/144` and `144/5` as ratios, and `1/145` and `146/5` as decimals.
SIMPLE = 144

#: The magnitudes a plain decimal numeral covers, as powers of ten. Outside
#: them scientific notation takes over: `10^-5` and `10^4` are the first
#: exponents written that way.
PLAIN = (-4, 3)

_TEN = Fraction(10)


def simple(value: Fraction) -> bool:
    """Whether Mixed writes `value` as a ratio rather than as a decimal."""
    return abs(value.numerator) <= SIMPLE and value.denominator <= SIMPLE


def decimal(value: Fraction, digits: int) -> str:
    """`value` as a decimal numeral of `digits` significant digits.

    A whole number is written whole, however many digits that takes; the point
    is never moved to make a number fit. Anything else keeps at least one
    digit after the point, which is what is left for `1234567.891` once seven
    of its six digits are spent.
    """
    sign = "-" if value < 0 else ""
    value = abs(value)
    whole, rest = divmod(value.numerator, value.denominator)
    text = str(whole)
    remainder = Fraction(rest, value.denominator)
    if not remainder:
        return sign + text
    keep = max(digits - len(text), 1) if whole else digits
    return sign + text + "." + _expansion(remainder, keep, started=bool(whole))


def scientific(value: Fraction, digits: int) -> str:
    """`value` as `m*10^e`, or as a decimal numeral where its magnitude allows.

    The mantissa is dropped when it is one, so a power of ten is written as
    one: `10^-6`, not `1·10^-6`.
    """
    if not value:
        return "0"
    power = exponent(abs(value))
    if PLAIN[0] <= power <= PLAIN[1]:
        return decimal(value, digits)
    mantissa = decimal(value / _TEN**power, digits)
    if mantissa in ("1", "-1"):
        return mantissa[:-1] + f"10^{power}"
    return f"{mantissa}*10^{power}"


def exponent(value: Fraction) -> int:
    """The power of ten `value`, which must be positive, begins at."""
    # The digit counts put it within one, and the comparisons settle it.
    guess = len(str(value.numerator)) - len(str(value.denominator))
    while value < _TEN**guess:
        guess -= 1
    while value >= _TEN ** (guess + 1):
        guess += 1
    return guess


def _expansion(rest: Fraction, keep: int, started: bool) -> str:
    """The digits after the point, `keep` significant ones or as far as exact.

    `started` is False while the digits are still the leading zeros of a
    number below one, which are placeholders and cost nothing.
    """
    digits = []
    while rest and keep:
        rest *= 10
        digit = rest.numerator // rest.denominator
        rest -= digit
        digits.append(str(digit))
        if started or digit:
            started = True
            keep -= 1
    return "".join(digits)
