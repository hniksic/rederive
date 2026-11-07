"""What Simplify answers at a precision other than exact.

Approximate mode, the approX command and the `APPROX` function, Mixed mode, and the
decimal notation that writes an approximation to its digits. Each case is the author's
text in and the worksheet's text out, written inline as in `test_simplify.py` so that a
failure names the mathematics. `pi` is `3.14159` in Approximate mode, the same answer
approX gives from an exact line, because approX is Simplify at another precision;
`APPROX(pi)` in an exact line is `355/113`, the rational the digits stand for; a whole
number needs no digits anywhere; and an approximation shows only the digits that are
the value's, however many more the arithmetic carried.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
from simplifying import parse, simp

from rederive.engine.computing import Context, approx
from rederive.engine.context import Notation, Precision
from rederive.syntax import ParseState


# Built the way a session builds them: taking the precision carries the
# notation with it, so approximate arithmetic is read in the style that
# suits it rather than as the ratios it is made of.
APPROXIMATE = Context().with_precision(Precision.APPROXIMATE)
MIXED = Context().with_precision(Precision.MIXED)
# Exact arithmetic read as digits, which is where an approximation made inside
# an exact line shows what it is worth.
DECIMAL = Context(notation=Notation.DECIMAL)


def test_a_single_root_of_a_polynomial_approximates_to_that_root():
    """`ROOT_OF` is the same idea for one indexed root, and it is as live.

    Nothing found produces one today, so it is the companion of `ROOT_SUM`
    rather than an answer anything gives; what it must do is be a number when a
    number is asked for. The quintic's one real root is 1.1673039782...
    """
    assert approx(parse("ROOT_OF(t^5 - t - 1, t, 0)"), Context(), 10).text == (
        "1.167303978"
    )


def test_a_special_function_typed_in_lower_case_approximates():
    assert approx(parse("erfi(1)"), Context(), 6).text == "1.65042"


# -- precision ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    # Two thirds is cut to `0.666666`, not rounded to `0.666667`: the
    # approximation of it is two thirds, and six digits of that are sixes.
    [("pi", "3.14159"), ("SQRT(3)", "1.73205"), ("1/2 + 1/6", "0.666666")],
    ids=str,
)
def test_approximate_mode(text, expected):
    assert simp(text, APPROXIMATE) == expected


@pytest.mark.parametrize(
    ("text", "expected"), [("pi", "3.14159"), ("SQRT(3)", "1.73205")], ids=str
)
def test_approx_is_simplify_at_another_precision(text, expected):
    # The same answers from an exact context, which is what the manual says
    # approX is: Simplify with the precision mode set to Approximate.
    assert approx(parse(text), Context()).text == expected


def test_approx_takes_the_digits_it_is_given():
    assert approx(parse("pi"), Context(), 12).text == "3.14159265358"


APPROXIMATED = [
    # The function, not the command: it approximates what it holds and leaves
    # the line around it alone, and it does so in Exact mode like any other.
    # What it leaves is a rational, since that is what an approximate number
    # is, so a rational notation writes the ratio the digits stand for.
    ("APPROX(pi)", "355/113"),
    ("APPROX([1/3, pi])", "[1/3, 355/113]"),
    ("1/2 + APPROX(1/3)", "5/6"),
    # A whole number needs no digits, here as everywhere else.
    ("APPROX(2 + 3)", "5"),
    # Nothing to approximate is nothing done.
    ("APPROX(x + 1)", "x + 1"),
]


@pytest.mark.parametrize(("text", "expected"), APPROXIMATED, ids=str)
def test_the_approx_function_approximates_where_it_stands(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("APPROX(pi)", "3.14159"),
        ("APPROX([1/3, pi])", "[0.333333, 3.14159]"),
        ("1/2 + APPROX(1/3)", "0.833333"),
    ],
    ids=str,
)
def test_what_the_approx_function_leaves_shows_its_digits(text, expected):
    # The same values as above, under a notation that writes them as digits:
    # what an approximation is and how it is written are two questions, and
    # the function answers only the first.
    assert simp(text, DECIMAL) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # An approximate number has only its precision's digits to show, so Decimal
        # notation writes one far from 1 with a power of ten, as Scientific does,
        # rather than padding it out with zeros that are no digits of its own.
        ("APPROX(#e^(pi*SQRT(163)))", "2.62537*10^17"),
        ("APPROX(EXP(100))", "2.68811*10^43"),
        ("APPROX(EXP(-100))", "3.72007*10^-44"),
        ("APPROX(100!)", "9.33262*10^157"),
        ("APPROX(LN(1 + 10^(-15)))", "1.00000*10^-15"),
        ("APPROX(SQRT(2)*10^20)", "1.41421*10^20"),
        ("APPROX(10^(-7))", "10^-7"),
        ("APPROX(2^100)*x", "1.26765*10^30*x"),
        # Approximate, a whole number of more digits than the precision has is
        # written to the precision too.
        ("APPROX(123456789)", "1.23456*10^8"),
        # A moderate magnitude is written plainly.
        ("APPROX(1234)", "1234"),
        ("APPROX(0.000123)", "0.000123000"),
        # An exact number is written out in full, however long.
        ("100!", str(math.factorial(100))),
        ("1/2^100", "0.000000000000000000000000000000788860"),
        ("123456789", "123456789"),
    ],
    ids=str,
)
def test_decimal_notation_writes_an_approximate_number_to_its_precision(text, expected):
    assert simp(text, DECIMAL) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("EXP(100)", "2.68811*10^43"),
        ("2^100", "1.26765*10^30"),
        ("x/3", "0.333333*x"),
    ],
    ids=str,
)
def test_decimal_notation_in_approximate_mode_writes_every_number_to_its_precision(
    text, expected
):
    context = replace(APPROXIMATE, notation=Notation.DECIMAL)
    assert simp(text, context) == expected


def test_decimal_notation_in_mixed_mode_writes_only_the_approximations_shortened():
    context = replace(MIXED, notation=Notation.DECIMAL)
    assert simp("SQRT(2)*10^20", context) == "1.41421*10^20"
    assert simp("10^20/3", context) == "33333333333333333333.3"


def test_the_approx_function_takes_the_digits_it_is_given():
    # Twelve digits of pi, which is a nearer rational than six digits' 355/113.
    assert simp("APPROX(pi, 12)") == "5419351/1725033"


def test_a_digit_count_is_the_digits_of_it_that_are_whole():
    """A count need not be written as a whole number, and is worked to as one.

    NUMBER.MTH asks `PARTS` for `LOG(1/(4*n*SQRT(3))*EXP(pi*SQRT(2*n/3)), 10) +
    5` digits - an estimate of the answer's size, plus five to spare - which is
    5.78 where `n` is 4. What can be worked to is the digits that are whole.
    """
    assert simp("APPROX(pi, 12.5)") == simp("APPROX(pi, 12)")
    assert simp("APPROX(pi, LN(11)/LN(2) + 9)") == simp("APPROX(pi, 12)")
    # A count that is no number at all is one nothing can round to, and the
    # head waits for it: an index is a number once the sum is written out.
    assert simp("APPROX(pi, n)") == "APPROX(pi, n)"
    assert simp("SUM(APPROX(pi, k + 11), k, 0, 1)") == simp(
        "APPROX(pi, 11) + APPROX(pi, 12)"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Special functions sympy does not prove finite, which are finite.
        ("APPROX(EI(LN(2)), 15)", "1.04516"),
        ("APPROX(SI(2), 15)", "1.60541"),
        ("APPROX(EI(2), 15)", "4.95423"),
        ("APPROX(COSH_INT(2), 15)", "2.45266"),
        # What has no exact answer still has digits, which is the original's
        # own division of labour between Simplify and approX.
        ("APPROX(INT(SIN(x)/x, x, 1, 2), 15)", "0.659329"),
        ("APPROX(SUM(1/(k^3 + 1), k, 1, inf), 15)", "0.686503"),
        # An infinity is refused, being the one thing that has no digits.
        ("APPROX(LN(0), 15)", "-inf"),
        ("APPROX(1/0, 15)", "±inf"),
    ],
    ids=str,
)
def test_a_value_sympy_leaves_open_is_approximated_rather_than_refused(text, expected):
    """Finiteness is asked loosely, the way rationality is.

    Sympy answers `is_finite` with `None` for `SI` and `EI` and for any
    integral or sum it has not worked out, and demanding a proof of finiteness
    left all of them written as themselves. Only what is known to be infinite
    has no digits.

    Written under the notation that shows digits, so these are the six the
    notation asks for and not the fifteen the approximation was made to.
    """
    assert simp(text, DECIMAL) == expected


def test_an_approximation_holds_the_digits_it_was_asked_for():
    # The value behind the six shown above: asking for more digits gives a
    # nearer rational, which is what makes the approximation the one requested
    # rather than the one the notation happens to print.
    assert simp("APPROX(EI(LN(2)), 6)") == "2615/2502"
    assert simp("APPROX(EI(LN(2)), 15)") == "80302381/76832342"


def test_approx_waits_for_the_value_it_is_asked_to_approximate():
    """Which is why the head is held back rather than rounded on conversion.

    What has to be approximated is the answer and not the question: the
    integral is evaluated first, and `APPROX` sees the number it came out as -
    a logarithm here, so that an answer standing at the exact value would be
    visibly a different one.
    """
    assert simp("APPROX(INT(1/x, x, 1, 2))") == "2731/3940"
    assert simp("INT(1/x, x, 1, 2)") == "LN(2)"


def test_a_number_that_needs_no_digits_does_not_get_any():
    # Derive's approximate numbers are the simplest rationals accurate to the
    # precision, and an integer approximates itself.
    assert simp("2 + 3", APPROXIMATE) == "5"
    assert simp("2*y*3", APPROXIMATE) == "6*y"
    assert simp("SQRT(4)", APPROXIMATE) == "2"


def test_exact_mode_reads_a_decimal_as_the_fraction_it_is():
    assert simp("0.1") == "1/10"
    assert simp("2.5*2") == "5"


def test_mixed_mode_approximates_the_irrational_and_keeps_the_rational_exact():
    assert simp("SQRT(3)", MIXED) == "1.73205"
    # Cut, not rounded: the sum is 1178/339, whose six digits are 3.47492.
    assert simp("1/3 + pi", MIXED) == "3.47492"
    # The manual's own case: the difference of the two fractions is computed
    # exactly, so the root of it is exactly 2/3. Approximate mode rounds them
    # first and loses that.
    exercise = "SQRT(3422357/2313 - 1140443/771)"
    assert simp(exercise, MIXED) == "2/3"
    assert simp(exercise, APPROXIMATE) != "2/3"


#: A number that is not real is approximated as a real part and an imaginary
#: part, each the approximate number it would be on its own. Each answer is the
#: original's.
COMPLEX_APPROXIMATED = [
    ("SQRT(-2)", "1.41421*#i"),
    ("(-8)^(1/3)", "1 + 1.73205*#i"),
    ("LN(-1)", "3.14159*#i"),
    ("(SQRT(3) + #i)^5", "-27.7128 + 16*#i"),
    ("(SQRT(2) + #i)^3", "-1.41421 + 5*#i"),
]


@pytest.mark.parametrize(("text", "expected"), COMPLEX_APPROXIMATED, ids=str)
@pytest.mark.parametrize("context", [APPROXIMATE, MIXED], ids=["approximate", "mixed"])
def test_a_complex_number_is_approximated_part_by_part(text, expected, context):
    assert simp(text, context) == expected


#: A root of `x^1000 = 12`, written in radicals the way soLve writes it.
ROOT_OF_TWELVE = (
    "12^(1/1000)*(SQRT(2)*SQRT(5/8 - SQRT(5)/8)/2 + SQRT(10)/8 + SQRT(2)/8"
    " - SQRT(10)*#i/8 + SQRT(2)*#i*SQRT(5/8 - SQRT(5)/8)/2 - SQRT(2)*#i/8)"
)

#: Every digit of an approximation is the value's own: a number is worked out
#: as if exactly and rounded only then. Rounding its pieces first and computing
#: with them exactly would let the arithmetic carry their error into the digits
#: shown - the thousandth power of the root of twelve came out
#: `11.9994 + 0.000252822*#i` that way, and the original answers
#: `12 - 0.000430980*#i` - and a cancellation between rounded pieces would leave
#: nothing of the value at all.
EXACT_DIGITS = [
    (f"({ROOT_OF_TWELVE})^1000", "12"),
    ("SQRT(10^200 + 1) - 10^100", "5.00000*10^-101"),
    ("x + SQRT(1000003) - SQRT(1000002)", "x + 0.000499999"),
    # Zero, though nothing proves it: no nonzero value is that small next to
    # the numbers it is made of.
    ("SQRT(2) + SQRT(3) - SQRT(5 + 2*SQRT(6))", "0"),
    # An imaginary part far smaller than the real part keeps digits of its own.
    ("(SQRT(2) + #i/10^8)^2 - 2", "-1.00000*10^-16 + 2.82842*10^-8*#i"),
    # The rest go wrong where intermediate values are rounded to the working
    # precision; Rederive computes as if exactly and rounds only the result.
    # A huge argument reduced by a rounded pi leaves nothing of the sine.
    ("SIN(10^20)", "-0.645251"),
    ("SIN(10^100)", "-0.372376"),
    ("COS(10^10)", "0.873119"),
    # Ramanujan's near-integer, whose exponent must be known to many more digits
    # than the result shows.
    ("#e^(pi*SQRT(163))", "2.62537*10^17"),
    # The 1 would swallow the 10^-15 before the logarithm is taken.
    ("LN(1 + 10^(-15))", "1.00000*10^-15"),
    # Both powers would overflow before they are divided.
    ("2^(10^20)/2^(10^20)", "1"),
    # Powers too big to build exactly, of a base the working precision would round
    # to 1: the base is carried to as many more digits as the exponent has.
    ("(1 + 10^(-20))^(10^20)", "2.71828"),
    ("(1 + 10^(-10))^(10^10)", "2.71828"),
    ("(1 - 10^(-20))^(10^20)", "0.367879"),
    ("(1 + 10^(-20))^(2*10^20)", "7.38905"),
    ("(-1 - 10^(-20))^(10^20 + 1)", "-2.71828"),
    # Catastrophic cancellation.
    ("SQRT(10001) - 100", "0.00499987"),
    # The quadrature's own error must stay below the last digit shown.
    ("INT(SQRT(1 - x^4), x, 0, 1)", "0.874019"),
]


@pytest.mark.parametrize(("text", "expected"), EXACT_DIGITS, ids=str)
@pytest.mark.parametrize("context", [APPROXIMATE, MIXED], ids=["approximate", "mixed"])
def test_an_approximation_shows_only_digits_of_the_value(text, expected, context):
    assert simp(text, context) == expected


def test_a_power_near_one_to_a_high_exponent_keeps_its_digits():
    """Rounded to the working precision, `1 + 10^-5` is 1, and so is its power.
    Computed as if exactly, as in the table above, it is close to `#e`. Only the
    approximate mode is checked, mixed keeping the exact power of half a million
    digits as the rational it is before it is written."""
    assert simp("(1 + 10^(-5))^(10^5)", APPROXIMATE) == "2.71826"


def test_a_power_too_big_to_approximate_stands():
    # `10^10^10` approximated would be a rational of ten billion digits, which is
    # no more writable than the exact power.
    assert simp("10^10^10", APPROXIMATE) == "10^(10^10)"


def test_a_long_exact_answer_is_written_out_quickly():
    """Half a million digits of numerator and denominator, written in well under
    the time limit, where writing them one digit at a time takes minutes."""
    written = simp("(1 + 10^(-5))^(10^5)")
    numerator, denominator = written.split("/")
    assert denominator == "1" + "0" * 500000
    assert numerator.startswith("27182682371")


@pytest.mark.parametrize("context", [APPROXIMATE, MIXED], ids=["approximate", "mixed"])
def test_an_approximation_is_read_from_its_decimal_digits(context):
    """Not from the binary fraction the float evaluating it is stored as.

    `SQRT(10^200 + 1)` evaluates to a float that shows `1.0000000000e+100` and
    is stored as `10^100 - 1.73*10^87`. The digits are what it approximates to.
    """
    assert simp("SQRT(10^200 + 1)", context) == "10^100"


def test_numerals_come_back_in_the_input_base():
    state = ParseState(input_base=16)
    assert simp("0FF + 1", Context(input_base=16), state) == "100"
