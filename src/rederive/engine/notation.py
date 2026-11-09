"""Derive's four numerical output styles, as rules over an exact value.

`Options Notation` chooses how computed numbers are written: as a ratio, a decimal, in
scientific notation, or whichever suits the number (Mixed). It applies to every number
in a result, so `x/3` simplified under Decimal is `0.333333·x`.

Three rules, all from Derive, hold across styles:

* Decimals are *truncated*, never rounded, and never into the whole part: two thirds is
  `0.666666` and `1234567.891` is `1234567.8`. Digits are significant digits, so leading
  zeros are free: `1/300000` is `0.00000333333`.
* Truncation neither pads nor strips. A tenth to six digits is `0.1` because the
  expansion ends there, while `0.10000000000000001` is `0.100000`: the zeros stand for
  digits that follow.
* Scientific notation is only for extreme magnitudes. Numbers from a ten-thousandth up
  to (not including) ten thousand are plain decimals, so `1234.56` is not `1.23456·10³`.

`decimal` is how Decimal notation writes an exact number. An approximate number has only
its precision's digits, and Decimal writes it as `scientific` does; the printer knows
which numbers are approximate.

Numerals the user typed are not affected by the style. `12345.6789` on the author line
shows as `12345.6` under every style, truncated by `NotationDigits` in the display
layer; the value itself is unchanged.
"""

from __future__ import annotations

from fractions import Fraction

#: Largest numerator and denominator Mixed writes as a ratio. Derive writes `1/144` and
#: `144/5` as ratios, and `1/145` and `146/5` as decimals.
SIMPLE = 144

#: Range of powers of ten written as plain decimals; `10^-5` and `10^4` are the first
#: written in scientific notation.
PLAIN = (-4, 3)

_TEN = Fraction(10)


def simple(value: Fraction) -> bool:
    """Whether Mixed writes `value` as a ratio rather than as a decimal."""
    return abs(value.numerator) <= SIMPLE and value.denominator <= SIMPLE


def decimal(value: Fraction, digits: int) -> str:
    """`value` as a decimal numeral of `digits` significant digits.

    A whole number is written in full regardless of `digits`. Anything else keeps at
    least one digit after the point, which is why `1234567.891` with six digits gives
    `1234567.8`.
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

    A mantissa of one is dropped: `10^-6`, not `1·10^-6`.
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
    """The exponent of the leading power of ten of `value`, which must be positive."""
    # The digit counts are within one of the answer; the loops correct it.
    guess = len(str(value.numerator)) - len(str(value.denominator))
    while value < _TEN**guess:
        guess -= 1
    while value >= _TEN ** (guess + 1):
        guess += 1
    return guess


def _expansion(rest: Fraction, keep: int, started: bool) -> str:
    """The digits after the point, `keep` significant ones or as far as exact.

    `started` is False while emitting the leading zeros of a number below one, which
    don't count as significant.
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
