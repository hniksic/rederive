"""Arithmetic checked against an independent oracle rather than against expected text.

Other Simplify tests pin down the form of answers for hand-picked cases. This file checks
only that the number is right, over many generated expressions: exact arithmetic against
`fractions.Fraction` and `math`, and approximations against `mpmath` at far higher
precision. Sympy does the arithmetic; this checks that nothing between input and output
(parsing decimals and percentages, printing digits, rewrites) corrupts a value.

Cases come from a seeded generator, so runs are reproducible and failing cases are
identified by their text.
"""

from __future__ import annotations

import math
import random
import re
from fractions import Fraction

import mpmath
import pytest
from simplifying import parse, simp

from rederive.engine.computing import Context, approx
from rederive.engine.context import Notation, Precision

DECIMAL = Context(notation=Notation.DECIMAL)
APPROXIMATE = Context().with_precision(Precision.APPROXIMATE)


def written(value: Fraction) -> str:
    """A rational as exact Simplify writes it."""
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def neg(text: str, value) -> str:
    return f"({text})" if value < 0 else text


# -- exact arithmetic ---------------------------------------------------------


def literal(rng: random.Random) -> tuple[str, Fraction]:
    """An integer, a decimal or a percentage, as a user would type it."""
    kind = rng.choice(["integer", "integer", "decimal", "percent"])
    if kind == "integer":
        n = rng.randint(-999, 999)
        return neg(str(n), n), Fraction(n)
    places = rng.randint(1, 3)
    units = rng.randint(-99999, 99999)
    digits = str(abs(units)).rjust(places + 1, "0")
    text = ("-" if units < 0 else "") + digits[:-places] + "." + digits[-places:]
    value = Fraction(units, 10**places)
    if kind == "percent":
        return f"({text}%)", value / 100
    return neg(text, value), value


def expression(rng: random.Random, depth: int) -> tuple[str, Fraction]:
    """A random tree of the four operations and whole powers, with its exact value."""
    if depth == 0 or rng.random() < 0.25:
        return literal(rng)
    while True:
        op = rng.choice("+-*/^")
        a, x = expression(rng, depth - 1)
        if op == "^":
            k = rng.randint(-3, 3)
            if x == 0 and k <= 0:
                continue
            return f"({a})^{neg(str(k), k)}", x**k
        b, y = expression(rng, depth - 1)
        if op == "/" and y == 0:
            continue
        value = {"+": x + y, "-": x - y, "*": x * y, "/": x / y if y else None}[op]
        return f"({a}) {op} ({b})", value


def exact_cases(seed: int, count: int) -> list[tuple[str, Fraction]]:
    rng = random.Random(seed)
    return [expression(rng, rng.randint(1, 3)) for _ in range(count)]


EXACT = exact_cases(1, 300)


@pytest.mark.parametrize(("text", "value"), EXACT, ids=[t for t, _ in EXACT])
def test_exact_arithmetic_is_the_arithmetic_of_fractions(text, value):
    assert simp(text) == written(value)


NUMBER = re.compile(
    r"(?P<sign>-?)(?:(?P<mantissa>\d+(?:\.\d+)?)(?:\*10\^(?P<scale>-?\d+))?"
    r"|10\^(?P<exponent>-?\d+))"
)


def read_number(text: str) -> tuple[Fraction, Fraction, int]:
    """Parse a number in digits into its value, the unit of its last digit, and its
    count of significant digits."""
    match = NUMBER.fullmatch(text)
    assert match, text
    mantissa = match["mantissa"] or "1"
    exponent = int(match["scale"] or match["exponent"] or 0)
    whole, _, fraction = mantissa.partition(".")
    unit = Fraction(10) ** (exponent - len(fraction))
    value = int(whole + fraction) * unit
    significant = len((whole + fraction).lstrip("0"))
    return (-value if match["sign"] else value), unit, significant


@pytest.mark.parametrize(("text", "value"), EXACT, ids=[t for t, _ in EXACT])
def test_decimal_notation_cuts_an_exact_value_without_changing_its_digits(text, value):
    """In decimal notation an exact value is truncated to the notation's digits, never
    rounded."""
    shown, unit, significant = read_number(simp(text, DECIMAL))
    if shown == value:
        return
    assert (shown < 0) == (value < 0) or shown == 0
    assert abs(shown) < abs(value) < abs(shown) + unit
    assert significant >= DECIMAL.notation_digits


# -- integer functions --------------------------------------------------------


def integer_cases(seed: int) -> list[tuple[str, int]]:
    rng = random.Random(seed)

    def nonzero(lo, hi):
        while not (n := rng.randint(lo, hi)):
            pass
        return n

    cases = []
    for _ in range(40):
        a, b = rng.randint(-10**12, 10**12), nonzero(-10**6, 10**6)
        cases.append((f"MOD({a}, {b})", a % b))
        cases.append((f"FLOOR({a}, {b})", a // b))
        a, b, c = (rng.randint(1, 10**9) * rng.choice([1, 6, 210]) for _ in range(3))
        cases.append((f"GCD({a}, {b}, {c})", math.gcd(a, b, c)))
        a, b = rng.randint(1, 10**6), rng.randint(1, 10**6)
        cases.append((f"LCM({a}, {b})", math.lcm(a, b)))
        n = rng.randint(0, 80)
        cases.append((f"{n}!", math.factorial(n)))
        k = rng.randint(0, n)
        cases.append((f"COMB({n}, {k})", math.comb(n, k)))
        cases.append((f"PERM({n}, {k})", math.perm(n, k)))
        n = rng.randint(0, 10**9)
        cases.append((f"SQRT({n * n})", n))
        n, k = rng.randint(1, 10**4), rng.randint(3, 7)
        cases.append((f"({n**k})^(1/{k})", n))
        b, k = rng.randint(2, 50), rng.randint(-20, 40)
        power = written(Fraction(b) ** k)
        cases.append((f"LOG({power}, {b})", k))
        n = rng.randint(-10**15, 10**15)
        cases.append((f"ABS({neg(str(n), n)})", abs(n)))
    return cases


INTEGER = integer_cases(2)


@pytest.mark.parametrize(("text", "value"), INTEGER, ids=[t for t, _ in INTEGER])
def test_an_integer_function_agrees_with_python(text, value):
    assert simp(text) == str(value)


# -- approximation ------------------------------------------------------------

#: The digits the oracle works to, far beyond any precision asked for below.
ORACLE_DIGITS = 80
TINY = mpmath.mpf(10) ** -30


def positive_leaf(rng: random.Random) -> tuple[str, mpmath.mpf]:
    kind = rng.choice(["integer", "integer", "ratio", "decimal", "pi", "e"])
    if kind == "integer":
        n = rng.randint(1, 1000)
        return str(n), mpmath.mpf(n)
    if kind == "ratio":
        p, q = rng.randint(1, 100), rng.randint(2, 100)
        return f"{p}/{q}", mpmath.mpf(p) / q
    if kind == "decimal":
        units = rng.randint(1, 99999)
        return f"{units // 100}.{units % 100:02}", mpmath.mpf(units) / 100
    if kind == "pi":
        return "pi", +mpmath.pi
    return "#e", +mpmath.e


def transcendental(rng: random.Random, depth: int) -> tuple[str, mpmath.mpf]:
    """A random tree of roots, logarithms, exponentials, trigonometric functions and
    the four operations, built only where each function is real and finite.

    No subtree may be indistinguishable from zero to the oracle: SIN(pi) is zero only
    to the oracle's precision, so a logarithm or root of it would be noise."""
    while True:
        text, value = _node(rng, depth)
        if abs(value) > TINY:
            return text, value


def _node(rng: random.Random, depth: int) -> tuple[str, mpmath.mpf]:
    if depth == 0 or rng.random() < 0.2:
        return positive_leaf(rng)
    while True:
        a, x = transcendental(rng, depth - 1)
        op = rng.choice(["+", "-", "*", "/", "SQRT", "LN", "EXP", "SIN", "COS", "ATAN",
                         "root", "LOG"])
        if op in "+-*/":
            b, y = transcendental(rng, depth - 1)
            if op == "/" and y == 0:
                continue
            value = {"+": x + y, "-": x - y, "*": x * y, "/": x / y if y else None}[op]
            return f"({a}) {op} ({b})", value
        if op in ("SQRT", "LN", "root", "LOG") and x <= 0:
            continue
        if op == "EXP" and abs(x) > 50:
            continue
        if op == "root":
            p, q = rng.choice([(1, 3), (2, 3), (3, 2), (1, 5), (-1, 2), (5, 4)])
            return f"({a})^({p}/{q})", x ** (mpmath.mpf(p) / q)
        if op == "LOG":
            base = rng.randint(2, 20)
            return f"LOG({a}, {base})", mpmath.log(x, base)
        function = {"SQRT": mpmath.sqrt, "LN": mpmath.log, "EXP": mpmath.exp,
                    "SIN": mpmath.sin, "COS": mpmath.cos, "ATAN": mpmath.atan}[op]
        return f"{op}({a})", function(x)


def approximate_cases(seed: int, count: int) -> list[tuple[str, int, mpmath.mpf]]:
    rng = random.Random(seed)
    cases = []
    with mpmath.workdps(ORACLE_DIGITS):
        for _ in range(count):
            text, value = transcendental(rng, rng.randint(1, 3))
            cases.append((text, rng.randint(6, 30), value))
    return cases


APPROXIMATED = approximate_cases(3, 200)


def assert_close(shown_text: str, value: mpmath.mpf, digits: int):
    shown, _, _ = read_number(shown_text)
    with mpmath.workdps(ORACLE_DIGITS):
        exact = mpmath.mpf(shown.numerator) / shown.denominator
        error = abs(exact - value) / abs(value)
        assert error < mpmath.mpf(10) ** (1 - digits), (shown_text, value)


@pytest.mark.parametrize(
    ("text", "digits", "value"), APPROXIMATED, ids=[f"{t} @{d}" for t, d, _ in APPROXIMATED]
)
def test_an_approximation_is_right_to_the_digits_asked_for(text, digits, value):
    assert_close(approx(parse(text), Context(), digits).text, value, digits)


SESSION = APPROXIMATED[:60]


@pytest.mark.parametrize(
    ("text", "digits", "value"), SESSION, ids=[t for t, _, _ in SESSION]
)
def test_approximate_mode_is_right_to_its_six_digits(text, digits, value):
    assert_close(simp(text, APPROXIMATE), value, APPROXIMATE.precision_digits)
