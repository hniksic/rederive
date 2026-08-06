"""The four numerical output styles, as the original writes them.

Every expectation below was checked against the original, which is why some of
them look arbitrary: the line Mixed draws between a number simple enough for a
ratio and one that is not falls exactly between 144 and 145, and scientific
notation takes over exactly outside a ten-thousandth and ten thousand, whatever
the digit count.

`notation` decides how a number the engine works out is written. A numeral the
user wrote is not that; it is shown as written, cut by `NotationDigits` alone,
which is `test_display`'s business.
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

#: Enough digits of the three irrationals the tests use that cutting them to
#: any of the digit counts below cuts the value's own digits.
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
        # Cut, never rounded: two thirds keeps its sixes.
        (Fraction(1, 3), "0.333333"),
        (Fraction(2, 3), "0.666666"),
        (Fraction(1, 6), "0.166666"),
        (Fraction(355, 113), "3.14159"),
        # The whole part is never cut, and one fractional digit always stays.
        (Fraction(123456789, 10000), "12345.6"),
        (Fraction(1234567891, 1000), "1234567.8"),
        # Leading zeros hold a place rather than say a digit.
        (Fraction(617, 500000000), "0.000001234"),
        (Fraction(1, 300000), "0.00000333333"),
        # Neither padded nor stripped: the expansion ends, or it does not.
        (Fraction(1, 8), "0.125"),
        (Fraction(1, 2), "0.5"),
        (Fraction(1, 10), "0.1"),
        (Fraction(10**17 + 1, 10**18), "0.100000"),
        (Fraction(10000001, 10000000), "1.00000"),
        (Fraction(1, 2**100), "0." + "0" * 30 + "788860"),
        # A whole number is whole, however long.
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
        # The mantissa is dropped when it is one.
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
    """A plain numeral covers a ten-thousandth up to ten thousand, and no more.

    The window is the original's and does not move with the digit count: at
    three digits `1234.5` is still written plainly, one whole digit past the
    three it is allowed.
    """
    value = Fraction(123456789, 10**8) * Fraction(10) ** power
    plain = "*" not in scientific(value, digits)
    assert plain == (-4 <= power <= 3)


# -- what an approximate number is -------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # The original's own answers: approximate mode holds these rationals.
        (Fraction(PI), Fraction(355, 113)),
        (Fraction(3550000000, 113), Fraction(31415929)),
        # A number needing no digits gets none.
        (Fraction(2, 3), Fraction(2, 3)),
        (Fraction(1, 3), Fraction(1, 3)),
        (Fraction(5), Fraction(5)),
        (Fraction(1, 10**7), Fraction(1, 10**7)),
        (Fraction(0), Fraction(0)),
        (Fraction(-2, 3), Fraction(-2, 3)),
        # One needing more digits than there are is read as the simplest
        # rational the precision allows, which is the original's answer here
        # too: the original holds `12345.6789` as 308642/25.
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
    """The digits do not move. `SQRT(3)` is `1.73205`, never `1.73204`.

    A simpler rational a little below the value would show a last digit the
    value does not have, so it is not the one taken.
    """
    assert decimal(simplest(value, digits), digits) == decimal(value, digits)


def test_one_digit_of_an_irrational_is_a_coarse_ratio() -> None:
    """One digit still buys the guard digit, so it is not as coarse as 3.

    The original answers 22/7 here, which is nearer still; both stand for pi
    at one digit, and neither is the whole number that a tolerance of the
    digit shown alone would settle for.
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
    # A ratio reaches a number wherever one sits, coefficients included.
    (x / 3, "x/3", "0.333333*x", "x/3", "0.333333*x"),
    (sp.sin(x) / 7, "SIN(x)/7", "0.142857*SIN(x)", "SIN(x)/7", "0.142857*SIN(x)"),
    (x / 10**6, "x/1000000", "0.000001*x", "10^-6*x", "10^-6*x"),
    # A quotient that is not a number is left alone.
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
    """The rest of a product keeps the denominator it had, all of it.

    `SQRT(2)/(3·x·y)` is one fraction over `x·y`, not a fraction over x
    multiplied by y, and not one with the `x·y` cleared into the numerator.
    """
    written = author_text(expression, Context(notation=Notation.DECIMAL))
    assert written == expected
    # And it means what it says: read back, it is the same expression with
    # one third replaced by the six digits it was written to.
    state = ParseState()
    read = to_sympy(parse_expression(written, state).node, Context())
    wanted = expression.subs(sp.Rational(1, 3), sp.Rational("0.333333")).subs(
        sp.Rational(1, 7), sp.Rational("0.142857")
    )
    assert sp.simplify(read - wanted) == 0


def test_a_power_of_ten_fences_itself_where_it_has_to() -> None:
    """`x^123456789` is one expression; `x^1.23456*10^8` would be another."""
    assert written(x ** sp.Integer(123456789), "Scientific") == "x^(1.23456*10^8)"
    # A product needs no fence, associating the way it is meant to.
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
    # The original draws the same three parts, and lets the arrow keys walk
    # into them: the mantissa first, then the power.
    assert (await simplified("12345.6789", "Scientific")).layout.lines == (
        "          4",
        "1.23456·10 ",
    )


async def test_the_digits_reach_the_answer() -> None:
    assert (await simplified("1/3", "Decimal", digits=3)).text == "0.333"
    assert (await simplified("1/3", "Decimal", digits=12)).text == "0.333333333333"


def test_a_numeral_the_user_wrote_is_not_the_notation_s_business() -> None:
    """Every style shows an authored numeral as written, cut by the digits.

    Which is what the original does: `12345.6789` typed on the author line is
    `12345.6` under Rational and under Scientific alike.
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
    """approX approximates, and approximating carries the notation with it.

    The original answers all five of these the same way under all four styles,
    because taking the precision approximate takes the notation to Scientific
    with it. What it does not do is leave the setting there.
    """
    session = Session()
    session.author(f"Notation := {style}")
    entry = session.author(text)
    assert (await session.approx(f"#{entry.number}")).text == expected
    assert session.settings["Notation"] == style


async def test_the_value_behind_the_answer_is_the_exact_one() -> None:
    """What a decimal style shows is a view: the value behind it is untouched.

    Read off the original, which answers all three of these the same way: the
    third is still a third, however few of its digits are on the screen.
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
    # And asking for it in rational notation shows the ratio it always was.
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
    """Rational notation leaves a float alone: it has no ratio to be written as."""
    session = Session()
    session.settings.apply({"Precision": Precision.APPROXIMATE.value})
    entry = session.author("pi")
    assert (await session.simplify(f"#{entry.number}")).text == "3.14159"
