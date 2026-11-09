"""Output spellings: one per name, one glyph per mark.

This is separate from `rederive.syntax.names` because input accepts many spellings of a
name while output produces exactly one. A test checks that every name the lexer knows
has a spelling here. Unicode is assumed, so there are no ASCII fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass

from rederive.display.options import ASTERISK, DOT, IMPLICIT

#: How the times operator prints, per `options.times`. Implicit is one space.
TIMES: dict[str, str] = {ASTERISK: "*", DOT: "·", IMPLICIT: " "}

#: The dot product, U+2219, distinct from the multiplication dot `·` U+00B7. It keeps a
#: space on each side in both display formats, since `[1, 2]∙[3, 4]` would read as one
#: bracketed expression.
DOT_PRODUCT = " ∙ "

#: The constants as Derive shows them. Those without a glyph print as their name.
CONSTANTS: dict[str, str] = {
    "pi": "π",
    "inf": "∞",
    "deg": "°",
    "#e": "ê",
    "#i": "î",
    "true": "true",
    "false": "false",
    "euler_gamma": "euler_gamma",
    "unit_circle": "unit_circle",
}

#: The pre-defined Greek variables, each as its own letter.
GREEK: dict[str, str] = {
    "alpha": "α",
    "beta": "β",
    "delta": "δ",
    "epsilon": "ε",
    "theta": "θ",
    "mu": "μ",
    "phi": "φ",
    "sigma": "σ",
    "tau": "τ",
    "omega": "ω",
}

NAMES: dict[str, str] = CONSTANTS | GREEK

#: Operators that print differently from their canonical spelling. `/=` has no glyph in
#: Derive's font and prints as written.
OPERATORS: dict[str, str] = {"<=": "≤", ">=": "≥", "+-": "±"}

#: The domain operator, which keeps its spaces in both formats.
DOMAIN = " :ε "

#: The fraction bar. Compressed format ends the bar with a half-width cell, to make it
#: look narrower without changing its column count.
BAR = "─"
HALF_BAR = "╴"

SQRT = "√"
SUM = "Σ"
PRODUCT = "Π"
ARROW = "→"
#: Marks a subscript too deep to lower, as `^` marks a superscript too deep to raise.
#: Derive's font has both arrows.
LOWERED = "↓"
DERIVATIVE = "d"
LIMIT = "lim"


@dataclass(frozen=True)
class Fence:
    """A pair of delimiters, in both a one-cell and a built-up spelling.

    `left` and `right` are the top, middle and bottom cells of a rail.
    """

    short: tuple[str, str]
    left: tuple[str, str, str]
    right: tuple[str, str, str]


PARENS = Fence(("(", ")"), ("┌", "│", "└"), ("┐", "│", "┘"))
BRACKETS = Fence(("[", "]"), ("┌", "│", "└"), ("┐", "│", "┘"))
#: Absolute value: the same rail whatever the height.
RAILS = Fence(("│", "│"), ("│", "│", "│"), ("│", "│", "│"))
#: The integral sign, drawn as a rail. It uses box-drawing arcs rather than `⌠` and `⌡`,
#: which terminals clip to the cell so the sign breaks where the pieces meet. Terminals
#: join box-drawing characters seamlessly, as with the matrix rails.
INTEGRAL = ("╭", "│", "╯")

_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def name(canonical: str) -> str:
    """How a name prints. Names not listed here print unchanged."""
    return NAMES.get(canonical, canonical)


def operator(canonical: str) -> str:
    return OPERATORS.get(canonical, canonical)


def numeral(value: str, base: int, digits: int) -> str:
    """`value`, decimal text from the parser, written in the output base.

    Above base ten the digits run A to Z, and a numeral starting with a letter gets a
    leading zero so it can't be read as a variable: fourteen is `0E` in hexadecimal, not
    `E`. A value with a radix point is cut to `digits` significant digits but not
    converted, whatever the base, since that would be arithmetic and this layer does
    none. A ratio (used for numerals with no finite decimal form) has each half
    converted.
    """
    if "." in value:
        return _fraction(value, digits)
    if base == 10:
        return value
    return "/".join(_whole(part, base) for part in value.split("/"))


def _fraction(value: str, digits: int) -> str:
    """A numeral with a radix point, cut to `digits` significant digits.

    Truncated, never rounded, and never into the whole part, with at least one
    fractional digit kept: `12345.67891` becomes `12345.6` and `1234567.891` becomes
    `1234567.8`. Leading zeros don't count, so `0.000123456789` keeps six
    digits after them. Trailing zeros left by the cut are kept because they stand for
    digits that follow: `0.10000000000000001` shows as `0.100000`, while a tenth shows as
    `0.1`.
    """
    integer, _, fraction = value.partition(".")
    whole = integer.lstrip("0")
    if whole:
        keep = max(digits - len(whole), 1)
    else:
        keep = digits + len(fraction) - len(fraction.lstrip("0"))
    return integer + "." + fraction[:keep]


def _whole(value: str, base: int) -> str:
    magnitude = int(value)
    text = ""
    while True:
        text = _DIGITS[magnitude % base] + text
        magnitude //= base
        if not magnitude:
            break
    return "0" + text if text[0].isalpha() else text
