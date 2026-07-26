"""The output inventory: one spelling per name, one glyph per mark.

Deliberately not `rederive.syntax.names`. Input accepts many spellings of a
name and output produces exactly one, so the two tables are separate even
where they overlap; a test guards the overlap by checking that every name the
lexer knows has a spelling here.

Unicode is always available, so there is no ASCII fallback anywhere below.
"""

from __future__ import annotations

from dataclasses import dataclass

from rederive.display.options import ASTERISK, DOT, IMPLICIT

#: How the times operator prints, per `options.times`. Implicit is one space.
TIMES: dict[str, str] = {ASTERISK: "*", DOT: "·", IMPLICIT: " "}

#: The dot product, U+2219. A different character from the multiplication dot
#: `·` U+00B7, and it keeps a space each side in both display formats: `[1,
#: 2]∙[3, 4]` would read as one bracketed expression.
DOT_PRODUCT = " ∙ "

#: The constants, in the spelling Derive shows them in. The four that have no
#: glyph print as their name.
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

#: Operators whose canonical spelling is not what is printed. `/=` has no
#: glyph in Derive's font and prints as written.
OPERATORS: dict[str, str] = {"<=": "≤", ">=": "≥", "+-": "±"}

#: The domain operator, which keeps its spaces in both formats.
DOMAIN = " :ε "

#: The fraction bar. Compressed format ends the bar with a half-width cell,
#: which narrows it without costing a column.
BAR = "─"
HALF_BAR = "╴"

SQRT = "√"
SUM = "Σ"
PRODUCT = "Π"
ARROW = "→"
#: A subscript too deep to lower, written the way `^` writes a superscript
#: too deep to raise. The original holds both arrows in its own font.
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
#: The integral sign, drawn as a rail of its own.
INTEGRAL = ("⌠", "│", "⌡")

_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def name(canonical: str) -> str:
    """How a name prints. Anything not in the inventory prints as it is."""
    return NAMES.get(canonical, canonical)


def operator(canonical: str) -> str:
    return OPERATORS.get(canonical, canonical)


def numeral(value: str, base: int) -> str:
    """`value`, decimal text from the parser, in the output base.

    Above base ten the digits run A to Z, and a numeral that would begin with
    a letter takes a leading zero so that it cannot be read as a variable
    name: fourteen is `0E` in hexadecimal, not `E`. A value carrying a radix
    point is shown as written whatever the base, since converting a fraction
    would be arithmetic and this layer does none.
    """
    if base == 10 or "." in value:
        return value
    magnitude = int(value)
    text = ""
    while True:
        text = _DIGITS[magnitude % base] + text
        magnitude //= base
        if not magnitude:
            break
    return "0" + text if text[0].isalpha() else text
