"""The four numerical output styles, as the original writes them.

Every expectation was checked against the original, which is why some look arbitrary:
Mixed stops writing ratios between 144 and 145, and scientific notation takes over
outside the range from a ten-thousandth to ten thousand, regardless of digit count.

`notation` covers numbers the engine computes. Numerals the user typed are shown as
written, truncated only by `NotationDigits`; those are tested in `test_display`.
"""

from __future__ import annotations

from decimal import Decimal, getcontext
from fractions import Fraction

import pytest
import sympy as sp

from rederive.engine.approximation import simplest
from rederive.engine.computing import Context, author_text, to_sympy
from rederive.engine.context import Notation, Precision
from rederive.engine.notation import decimal, scientific, simple
from rederive.model.session import Session
from rederive.syntax import ParseState, parse_expression

getcontext().prec = 40

#: The irrationals used by the tests, with more digits than any digit count below.
PI = Decimal("3.141592653589793238462643383279502884197")
SQRT2 = Decimal("1.414213562373095048801688724209698078570")
SQRT3 = Decimal("1.732050807568877293527446341505872366943")

x = sp.Symbol("x", real=True)
y = sp.Symbol("y", real=True)


def written(expression, style, digits=6):
    context = Context(notation=Notation(style), notation_digits=digits)
    return author_text(expression, context)


# -- decimal point notation --------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # Truncated, never rounded: two thirds keeps its sixes.
        (Fraction(1, 3), "0.333333"),
        (Fraction(2, 3), "0.666666"),
        (Fraction(1, 6), "0.166666"),
        (Fraction(355, 113), "3.14159"),
        # The integer part is never truncated, and one fractional digit always stays.
        (Fraction(123456789, 10000), "12345.6"),
        (Fraction(1234567891, 1000), "1234567.8"),
        # Leading zeros don't count as digits.
        (Fraction(617, 500000000), "0.000001234"),
        (Fraction(1, 300000), "0.00000333333"),
        # Neither padded nor stripped of trailing zeros.
        (Fraction(1, 8), "0.125"),
        (Fraction(1, 2), "0.5"),
        (Fraction(1, 10), "0.1"),
        (Fraction(10**17 + 1, 10**18), "0.100000"),
        (Fraction(10000001, 10000000), "1.00000"),
        (Fraction(1, 2**100), "0." + "0" * 30 + "788860"),
        # An integer is written in full, however long.
        (Fraction(2**100), str(2**100)),
        (Fraction(0), "0"),
        (Fraction(-1, 3), "-0.333333"),
    ],
)
def test_decimal(value: Fraction, expected: str) -> None:
    assert decimal(value, 6) == expected


@pytest.mark.parametrize(
    ("digits", "expected"),
    [(1, "123.4"), (3, "123.4"), (6, "123.456"), (10, "123.456789")],
)
def test_the_whole_part_survives_any_digit_count(digits: int, expected: str) -> None:
    assert decimal(Fraction(123456789, 1000000), digits) == expected


# -- scientific notation -----------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Fraction(123456789, 10000), "1.23456*10^4"),
        (Fraction(1, 2**100), "7.88860*10^-31"),
        (Fraction(2**100), "1.26765*10^30"),
        (Fraction(123456789), "1.23456*10^8"),
        # A mantissa of one is dropped.
        (Fraction(1, 10**6), "10^-6"),
        (Fraction(10000), "10^4"),
        (Fraction(20000), "2*10^4"),
        (Fraction(12345), "1.2345*10^4"),
        (Fraction(200011, 20), "1.00005*10^4"),
        (Fraction(0), "0"),
        (Fraction(-15, 10) * 10**7, "-1.5*10^7"),
    ],
)
def test_scientific(value: Fraction, expected: str) -> None:
    assert scientific(value, 6) == expected


@pytest.mark.parametrize("digits", [3, 6, 10])
@pytest.mark.parametrize("power", range(-8, 9))
def test_where_scientific_takes_over(digits: int, power: int) -> None:
    """Plain numerals cover a ten-thousandth up to ten thousand.

    As in the original, the range doesn't depend on the digit count: at three digits
    `1234.5` is still written plainly, with one more integer digit than allowed.
    """
    value = Fraction(123456789, 10**8) * Fraction(10) ** power
    plain = "*" not in scientific(value, digits)
    assert plain == (-4 <= power <= 3)


# -- approximating with simple rationals -------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # The original's answers: approximate mode holds these rationals.
        (Fraction(PI), Fraction(355, 113)),
        (Fraction(3550000000, 113), Fraction(31415929)),
        # Numbers that are already simple are kept as they are.
        (Fraction(2, 3), Fraction(2, 3)),
        (Fraction(1, 3), Fraction(1, 3)),
        (Fraction(5), Fraction(5)),
        (Fraction(1, 10**7), Fraction(1, 10**7)),
        (Fraction(0), Fraction(0)),
        (Fraction(-2, 3), Fraction(-2, 3)),
        # A number with more digits than the precision becomes the simplest rational
        # within it; the original also holds `12345.6789` as 308642/25.
        (Fraction(123456789, 10000), Fraction(308642, 25)),
    ],
)
def test_simplest(value: Fraction, expected: Fraction) -> None:
    assert simplest(value, 6) == expected


@pytest.mark.parametrize(
    "value", [Fraction(PI), Fraction(SQRT2), Fraction(SQRT3), Fraction(2, 3)]
)
@pytest.mark.parametrize("digits", [3, 6, 12])
def test_an_approximation_shows_what_it_approximates(
    value: Fraction, digits: int
) -> None:
    """The approximation shows the same digits as the value.

    `SQRT(3)` must show as `1.73205`, never `1.73204`, so a simpler rational just below
    the value is not acceptable.
    """
    assert decimal(simplest(value, digits), digits) == decimal(value, digits)


def test_one_digit_of_an_irrational_is_a_coarse_ratio() -> None:
    """One digit still gets a guard digit, so the answer is not as coarse as 3.

    The original gives 22/7, which is closer; both show pi to one digit.
    """
    assert simplest(Fraction(PI), 1) == Fraction(19, 6)


# -- what Mixed calls simple -------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Fraction(1, 144), True),
        (Fraction(1, 145), False),
        (Fraction(144, 5), True),
        (Fraction(146, 5), False),
        (Fraction(128, 129), True),
        (Fraction(250, 251), False),
        (Fraction(355, 113), False),
        (Fraction(1, 3), True),
        (Fraction(-1, 3), True),
        (Fraction(1000), False),
    ],
)
def test_simple(value: Fraction, expected: bool) -> None:
    assert simple(value) is expected


# -- through the printer -----------------------------------------------------

CASES = [
    # expression, Rational, Decimal, Mixed, Scientific
    (sp.Rational(1, 3), "1/3", "0.333333", "1/3", "0.333333"),
    (sp.Rational(355, 113), "355/113", "3.14159", "3.14159", "3.14159"),
    (sp.Integer(7), "7", "7", "7", "7"),
    (sp.Integer(0), "0", "0", "0", "0"),
    (
        sp.Integer(123456789),
        "123456789",
        "123456789",
        "1.23456*10^8",
        "1.23456*10^8",
    ),
    # Numeric coefficients are written in the style too.
    (x / 3, "x/3", "0.333333*x", "x/3", "0.333333*x"),
    (sp.sin(x) / 7, "SIN(x)/7", "0.142857*SIN(x)", "SIN(x)/7", "0.142857*SIN(x)"),
    (x / 10**6, "x/1000000", "0.000001*x", "10^-6*x", "10^-6*x"),
    # Non-numeric quotients are left alone.
    (3 / (x + 1), "3/(x + 1)", "3/(x + 1)", "3/(x + 1)", "3/(x + 1)"),
    (x / sp.Symbol("y", real=True), "x/y", "x/y", "x/y", "x/y"),
]


@pytest.mark.parametrize(("expression", "rational", "dec", "mixed", "sci"), CASES)
def test_the_printer_writes_each_style(
    expression, rational: str, dec: str, mixed: str, sci: str
) -> None:
    assert written(expression, "Rational") == rational
    assert written(expression, "Decimal") == dec
    assert written(expression, "Mixed") == mixed
    assert written(expression, "Scientific") == sci


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (x / 3, "0.333333*x"),
        (x / (3 * y), "0.333333*x/y"),
        (sp.sqrt(2) / (3 * x * y), "0.333333*SQRT(2)/(x*y)"),
        (1 / (3 * x**2 * y), "0.333333/(x^2*y)"),
        (x / (3 * y**2), "0.333333*x/y^2"),
        (2 * x * y / 6, "0.333333*x*y"),
        (-x / 3, "-0.333333*x"),
    ],
)
def test_a_written_coefficient_keeps_the_rest_of_its_product(
    expression, expected: str
) -> None:
    """The rest of the product keeps its whole denominator.

    `SQRT(2)/(3·x·y)` is written over `x·y`, not as a fraction over x times y, nor with
    `x·y` moved into the numerator.
    """
    written = author_text(expression, Context(notation=Notation.DECIMAL))
    assert written == expected
    # Read back, it is the same expression with 1/3 replaced by its six digits.
    state = ParseState()
    read = to_sympy(parse_expression(written, state).node, Context())
    wanted = expression.subs(sp.Rational(1, 3), sp.Rational("0.333333")).subs(
        sp.Rational(1, 7), sp.Rational("0.142857")
    )
    assert sp.simplify(read - wanted) == 0


def test_a_power_of_ten_fences_itself_where_it_has_to() -> None:
    """A power-of-ten exponent is parenthesized, since `x^1.23456*10^8` means
    something else."""
    assert written(x ** sp.Integer(123456789), "Scientific") == "x^(1.23456*10^8)"
    # A product needs no parentheses.
    assert written(123456789 * x, "Scientific") == "1.23456*10^8*x"


# -- through the session -----------------------------------------------------


async def simplified(text: str, style: str, digits: int = 6):
    session = Session()
    session.author(f"Notation := {style}")
    session.author(f"NotationDigits := {digits}")
    entry = session.author(text)
    return await session.simplify(f"#{entry.number}")


@pytest.mark.parametrize(
    ("style", "expected"),
    [
        ("Rational", (" 1 ", "───", " 3 ")),
        ("Decimal", ("0.333333",)),
        ("Mixed", (" 1 ", "───", " 3 ")),
        ("Scientific", ("0.333333",)),
    ],
)
async def test_the_setting_reaches_the_answer(
    style: str, expected: tuple[str, ...]
) -> None:
    assert (await simplified("1/3", style)).layout.lines == expected


async def test_scientific_notation_is_drawn_as_a_power_of_ten() -> None:
    # The original draws the same three parts, and the arrow keys visit them: the
    # mantissa first, then the power.
    assert (await simplified("12345.6789", "Scientific")).layout.lines == (
        "          4",
        "1.23456·10 ",
    )


async def test_the_digits_reach_the_answer() -> None:
    assert (await simplified("1/3", "Decimal", digits=3)).text == "0.333"
    assert (await simplified("1/3", "Decimal", digits=12)).text == "0.333333333333"


def test_a_numeral_the_user_wrote_is_not_the_notation_s_business() -> None:
    """Every style shows an authored numeral as written, truncated to the digit count.

    As in the original, `12345.6789` typed on the author line shows as `12345.6` under
    any style.
    """
    for style in ("Rational", "Decimal", "Mixed", "Scientific"):
        session = Session()
        session.author(f"Notation := {style}")
        assert session.author("12345.6789").layout.lines == ("12345.6",)
        assert session.author("1.5").layout.lines == ("1.5",)


@pytest.mark.parametrize("style", ["Rational", "Decimal", "Mixed", "Scientific"])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("pi", "3.14159"),
        ("1/3", "0.333333"),
        ("10^7 * pi", "3.14159*10^7"),
        ("SQRT(2)", "1.41421"),
        ("1/10^7", "10^-7"),
    ],
)
async def test_approx_answers_in_scientific_whatever_the_style(
    style: str, text: str, expected: str
) -> None:
    """approX answers in Scientific notation regardless of the style.

    The original answers these the same way under all four styles, because
    approximating also switches notation to Scientific. The setting itself is left
    unchanged.
    """
    session = Session()
    session.author(f"Notation := {style}")
    entry = session.author(text)
    assert (await session.approx(f"#{entry.number}")).text == expected
    assert session.settings["Notation"] == style


async def test_the_value_behind_the_answer_is_the_exact_one() -> None:
    """A decimal style affects display only; the stored value stays exact.

    Matches the original: the third is still a third, whatever digits are shown.
    """
    session = Session()
    session.author("Notation := Decimal")
    session.author("1/3")
    answer = await session.simplify("#2")
    assert answer.text == "0.333333"
    session.author(f"3 * #{answer.number}")
    assert (await session.simplify(f"#{answer.number + 1}")).text == "1"
    session.author(f"#{answer.number} - 1/3")
    assert (await session.simplify(f"#{answer.number + 3}")).text == "0"
    # Under Rational notation it shows as the exact ratio.
    session.author("Notation := Rational")
    assert (await session.simplify(f"#{answer.number}")).text == "1/3"


async def test_an_answer_keeps_the_style_it_was_worked_out_in() -> None:
    """Changing the notation leaves what is already on screen alone."""
    session = Session()
    session.author("1/3")
    first = await session.simplify("#1")
    session.author("Notation := Decimal")
    assert first.layout.lines == (" 1 ", "───", " 3 ")
    assert (await session.simplify("#1")).layout.lines == ("0.333333",)


async def test_approximate_arithmetic_still_answers_in_digits() -> None:
    """Rational notation leaves a float alone, since it has no ratio to show."""
    session = Session()
    session.settings.apply({"Precision": Precision.APPROXIMATE.value})
    entry = session.author("pi")
    assert (await session.simplify(f"#{entry.number}")).text == "3.14159"
