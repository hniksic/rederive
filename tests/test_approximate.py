"""Simplify at non-exact precision.

Covers Approximate mode, the approX command and `APPROX` function, Mixed mode, and
Decimal notation. Cases are written inline, as in `test_simplify.py`, so a failure shows
the mathematics. Key rules: approX is Simplify at Approximate precision, so `pi` gives
`3.14159` either way; `1 + APPROX(pi)` in an exact line is the rational `468/113`, but a
line that is just `APPROX(pi)` is approX of `pi`; whole numbers never get decimal digits;
and an approximation shows only digits that are correct for the value.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import sympy as sp
from simplifying import declaring, parse, simp

from rederive.engine.approximation import approximated
from rederive.engine.computing import Context, approx
from rederive.engine.context import Notation, Precision
from rederive.syntax import ParseState


# Built as a session builds them: setting the precision also sets a matching notation,
# so approximate results are shown as digits rather than ratios.
APPROXIMATE = Context().with_precision(Precision.APPROXIMATE)
MIXED = Context().with_precision(Precision.MIXED)
# Exact arithmetic shown as digits, to see the value of an approximation made inside an
# exact line.
DECIMAL = Context(notation=Notation.DECIMAL)


def test_a_single_root_of_a_polynomial_approximates_to_that_root():
    """`ROOT_OF`, one indexed root of a polynomial, approximates to a number.

    Nothing produces `ROOT_OF` today; it exists as the companion of `ROOT_SUM`, but must
    still approximate. The quintic's one real root is 1.1673039782...
    """
    assert approx(parse("ROOT_OF(t^5 - t - 1, t, 0)"), Context(), 10).text == (
        "1.167303978"
    )


def test_a_special_function_typed_in_lower_case_approximates():
    assert approx(parse("erfi(1)"), Context(), 6).text == "1.65042"


# -- precision ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    # Two thirds is truncated to `0.666666`, not rounded to `0.666667`.
    [("pi", "3.14159"), ("SQRT(3)", "1.73205"), ("1/2 + 1/6", "0.666666")],
    ids=str,
)
def test_approximate_mode(text, expected):
    assert simp(text, APPROXIMATE) == expected


@pytest.mark.parametrize(
    ("text", "expected"), [("pi", "3.14159"), ("SQRT(3)", "1.73205")], ids=str
)
def test_approx_is_simplify_at_another_precision(text, expected):
    # The manual defines approX as Simplify with precision set to Approximate, so it
    # gives the same answers from an exact context.
    assert approx(parse(text), Context()).text == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    # A simpler rational lies within tolerance just below the first three and just above
    # the last, and choosing it would show a wrong digit.
    [
        ("3.14159", "3.14159"),
        ("1.73205", "1.73205"),
        ("0.333333", "0.333333"),
        ("APPROX(0.12399, 3)", "0.123"),
    ],
    ids=str,
)
def test_a_typed_decimal_keeps_its_digits(text, expected):
    assert simp(text, APPROXIMATE) == expected


def test_approx_takes_the_digits_it_is_given():
    assert approx(parse("pi"), Context(), 12).text == "3.14159265358"


APPROXIMATED = [
    # The function inside a line, not the command: it approximates its argument and
    # leaves the rest of the line alone, in Exact mode too. The result is a rational,
    # so rational notation shows the ratio the digits stand for.
    ("1 + APPROX(pi)", "468/113"),
    ("[APPROX(1/3), APPROX(pi)]", "[1/3, 355/113]"),
    ("1/2 + APPROX(1/3)", "5/6"),
    ("APPROX(1/2) + x", "x + 1/2"),
    ("2*APPROX(1/2)", "1"),
    # A whole number needs no decimal digits.
    ("APPROX(2 + 3)", "5"),
    # Nothing to approximate.
    ("APPROX(x + 1)", "x + 1"),
]


@pytest.mark.parametrize(("text", "expected"), APPROXIMATED, ids=str)
def test_the_approx_function_approximates_where_it_stands(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1 + APPROX(pi)", "4.14159"),
        ("[APPROX(1/3), APPROX(pi)]", "[0.333333, 3.14159]"),
        ("1/2 + APPROX(1/3)", "0.833333"),
    ],
    ids=str,
)
def test_what_the_approx_function_leaves_shows_its_digits(text, expected):
    # The same values as above under Decimal notation: the function decides the value,
    # the notation decides how it is written.
    assert simp(text, DECIMAL) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # An approximate number has only its precision's worth of digits, so Decimal
        # notation writes one far from 1 with a power of ten, as Scientific does,
        # rather than padding with meaningless zeros.
        ("APPROX(#e^(pi*SQRT(163)))", "2.62537*10^17"),
        ("APPROX(EXP(100))", "2.68811*10^43"),
        ("APPROX(EXP(-100))", "3.72007*10^-44"),
        ("APPROX(100!)", "9.33262*10^157"),
        ("APPROX(LN(1 + 10^(-15)))", "1.00000*10^-15"),
        ("APPROX(SQRT(2)*10^20)", "1.41421*10^20"),
        ("APPROX(10^(-7))", "10^-7"),
        ("APPROX(2^100)*x", "1.26765*10^30*x"),
        # An approximate whole number with more digits than the precision is
        # shortened too.
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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("APPROX(1/2)", "0.5"),
        ("APPROX(pi)", "3.14159"),
        ("APPROX([1/3, pi])", "[0.333333, 3.14159]"),
        ("APPROX(x/3)", "0.333333*x"),
        ("APPROX(1/3, 3)", "0.333"),
        ("APPROX(pi, 12)", "3.14159265358"),
        # Nothing to approximate.
        ("APPROX(2 + 3)", "5"),
        ("APPROX(x + 1)", "x + 1"),
    ],
    ids=str,
)
def test_a_line_that_is_an_approx_call_is_approx_of_its_operand(text, expected):
    # This is the approX command at the call's digit count, so the answer is written
    # as approximate even in an exact line.
    assert simp(text) == expected


def test_a_whole_line_approx_follows_a_reference():
    # `#1` stands for its line, so this line is an APPROX call too.
    context = Context(labels={1: parse("1/2")})
    assert simp("APPROX(#1)", context) == "0.5"


def test_the_approx_function_takes_the_digits_it_is_given():
    # Twelve digits of pi give a nearer rational than six digits' 355/113.
    assert simp("x + APPROX(pi, 12)") == "x + 5419351/1725033"


def test_a_digit_count_is_the_digits_of_it_that_are_whole():
    """A non-integer digit count is truncated to an integer.

    NUMBER.MTH asks `PARTS` for `LOG(1/(4*n*SQRT(3))*EXP(pi*SQRT(2*n/3)), 10) + 5`
    digits (an estimate of the answer's size, plus five), which is 5.78 for `n` = 4.
    """
    assert simp("APPROX(pi, 12.5)") == simp("APPROX(pi, 12)")
    assert simp("APPROX(pi, LN(11)/LN(2) + 9)") == simp("APPROX(pi, 12)")
    # A symbolic count leaves the call unevaluated until it becomes a number, as a sum
    # index does once the sum is expanded.
    assert simp("APPROX(pi, n)") == "APPROX(pi, n)"
    assert simp("SUM(APPROX(pi, k + 11), k, 0, 1)") == simp(
        "APPROX(pi, 11) + APPROX(pi, 12)"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Special functions sympy does not prove finite, which are finite.
        ("APPROX(EI(LN(2)), 15)", "1.04516378011749"),
        ("APPROX(SI(2), 15)", "1.60541297680269"),
        ("APPROX(EI(2), 15)", "4.95423435600189"),
        ("APPROX(COSH_INT(2), 15)", "2.45266692264691"),
        # What has no exact answer still has digits, matching the original's split
        # between Simplify and approX.
        ("APPROX(INT(SIN(x)/x, x, 1, 2), 15)", "0.659329906435511"),
        ("APPROX(SUM(1/(k^3 + 1), k, 1, inf), 15)", "0.686503342338623"),
        # Only an infinity has no digits.
        ("APPROX(LN(0), 15)", "-inf"),
        ("APPROX(1/0, 15)", "±inf"),
    ],
    ids=str,
)
def test_a_value_sympy_leaves_open_is_approximated_rather_than_refused(text, expected):
    """Only values known to be infinite are refused; unknown finiteness is not.

    Sympy's `is_finite` is `None` for `SI`, `EI`, and any integral or sum it can't
    evaluate, so requiring proven finiteness would leave them all unapproximated.

    Decimal notation shows six digits, not the fifteen the approximation was made to.
    """
    assert simp(text, DECIMAL) == expected


def test_an_approximation_holds_the_digits_it_was_asked_for():
    # The rational behind the digits above, kept as a rational inside a line: more
    # requested digits give a nearer rational, regardless of how many get printed.
    assert simp("y + APPROX(EI(LN(2)), 6)") == "y + 2615/2502"
    assert simp("y + APPROX(EI(LN(2)), 15)") == "y + 80302381/76832342"


def test_approx_waits_for_the_value_it_is_asked_to_approximate():
    """`APPROX` is held back until its argument is evaluated, not rounded on conversion.

    The integral is evaluated first and `APPROX` sees the result, here a logarithm, so
    an answer approximated from anything else would visibly differ.
    """
    assert simp("y + APPROX(INT(1/x, x, 1, 2))") == "y + 2731/3940"
    assert simp("INT(1/x, x, 1, 2)") == "LN(2)"


def test_a_number_that_needs_no_digits_does_not_get_any():
    # Derive's approximate numbers are the simplest rationals accurate to the
    # precision, so an integer approximates to itself.
    assert simp("2 + 3", APPROXIMATE) == "5"
    assert simp("2*y*3", APPROXIMATE) == "6*y"
    assert simp("SQRT(4)", APPROXIMATE) == "2"


def test_exact_mode_reads_a_decimal_as_the_fraction_it_is():
    assert simp("0.1") == "1/10"
    assert simp("2.5*2") == "5"


def test_mixed_mode_approximates_the_irrational_and_keeps_the_rational_exact():
    assert simp("SQRT(3)", MIXED) == "1.73205"
    # Truncated, not rounded: the sum is 1178/339, whose six digits are 3.47492.
    assert simp("1/3 + pi", MIXED) == "3.47492"
    # The manual's example: the difference of the fractions is exact, so its root is
    # exactly 2/3. Approximate mode rounds them first and loses that.
    exercise = "SQRT(3422357/2313 - 1140443/771)"
    assert simp(exercise, MIXED) == "2/3"
    assert simp(exercise, APPROXIMATE) != "2/3"


#: Non-real numbers are approximated part by part, real and imaginary. The answers are
#: the original's.
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

#: Every digit shown must be correct: a value is computed as if exactly and rounded only
#: at the end. Rounding the pieces first would carry their error into the result (the
#: thousandth power of the root of twelve came out `11.9994 + 0.000252822*#i` that way,
#: and the original gives `12 - 0.000430980*#i`), and cancellation between rounded
#: pieces could wipe out the value entirely.
EXACT_DIGITS = [
    (f"({ROOT_OF_TWELVE})^1000", "12"),
    ("SQRT(10^200 + 1) - 10^100", "5.00000*10^-101"),
    ("x + SQRT(1000003) - SQRT(1000002)", "x + 0.000499999"),
    # Zero, though unproven: no nonzero value is that small relative to its parts.
    ("SQRT(2) + SQRT(3) - SQRT(5 + 2*SQRT(6))", "0"),
    # An imaginary part much smaller than the real part keeps its own digits.
    ("(SQRT(2) + #i/10^8)^2 - 2", "-1.00000*10^-16 + 2.82842*10^-8*#i"),
    # The rest go wrong if intermediate values are rounded to working precision.
    # A huge argument reduced by a rounded pi loses the sine entirely.
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
    # Powers too big to compute exactly, of a base working precision would round to 1:
    # the base gets as many extra digits as the exponent has.
    ("(1 + 10^(-20))^(10^20)", "2.71828"),
    ("(1 + 10^(-10))^(10^10)", "2.71828"),
    ("(1 - 10^(-20))^(10^20)", "0.367879"),
    ("(1 + 10^(-20))^(2*10^20)", "7.38905"),
    ("(-1 - 10^(-20))^(10^20 + 1)", "-2.71828"),
    # Catastrophic cancellation.
    ("SQRT(10001) - 100", "0.00499987"),
    # The quadrature error must stay below the last digit shown.
    ("INT(SQRT(1 - x^4), x, 0, 1)", "0.874019"),
]


@pytest.mark.parametrize(("text", "expected"), EXACT_DIGITS, ids=str)
@pytest.mark.parametrize("context", [APPROXIMATE, MIXED], ids=["approximate", "mixed"])
def test_an_approximation_shows_only_digits_of_the_value(text, expected, context):
    assert simp(text, context) == expected


def test_a_power_near_one_to_a_high_exponent_keeps_its_digits():
    """Rounded to working precision, `1 + 10^-5` is 1, and so is its power; computed
    exactly, it is close to `#e`. Only Approximate mode is checked, since Mixed keeps
    the half-million-digit exact rational until it is written."""
    assert simp("(1 + 10^(-5))^(10^5)", APPROXIMATE) == "2.71826"


def test_a_power_too_big_to_approximate_stands():
    # Approximated, `10^10^10` would be a ten-billion-digit rational, no more writable
    # than the exact power.
    assert simp("10^10^10", APPROXIMATE) == "10^(10^10)"


def test_a_long_exact_answer_is_written_out_quickly():
    """Half a million digits each of numerator and denominator are written well within
    the time limit; writing them a digit at a time takes minutes."""
    written = simp("(1 + 10^(-5))^(10^5)")
    numerator, denominator = written.split("/")
    assert denominator == "1" + "0" * 500000
    assert numerator.startswith("27182682371")


@pytest.mark.parametrize("context", [APPROXIMATE, MIXED], ids=["approximate", "mixed"])
def test_an_approximation_is_read_from_its_decimal_digits(context):
    """An approximation comes from a float's decimal digits, not its binary value.

    `SQRT(10^200 + 1)` evaluates to a float that prints as `1.0000000000e+100` but is
    stored as `10^100 - 1.73*10^87`. The printed digits are what count.
    """
    assert simp("SQRT(10^200 + 1)", context) == "10^100"


def test_numerals_come_back_in_the_input_base():
    state = ParseState(input_base=16)
    assert simp("0FF + 1", Context(input_base=16), state) == "100"


# -- values carried from step to step -----------------------------------------


def defining(context, **definitions):
    """`context` with functions defined, each given as `(parameters, body)`."""
    state = declaring(*definitions)
    functions = {
        name: (tuple(parameters.split(",")), parse(body, state))
        for name, (parameters, body) in definitions.items()
    }
    return replace(context, functions={**context.functions, **functions})


def test_every_number_in_a_matrix_is_approximated():
    assert simp("[[SQRT(2), 1], [pi, 1/3]]", APPROXIMATE) == (
        "[[1.41421, 1], [3.14159, 0.333333]]"
    )


def test_a_system_iterates_numerically_in_approximate_mode():
    # Newton's method on the first component; left exact, each iterate nests the last
    # one inside its sine and cosine.
    text = "ITERATES([x - (x - COS(x))/(1 + SIN(x)), y], [x, y], [1, 0], 3)"
    assert simp(text, APPROXIMATE) == (
        "[[1, 0], [0.750363, 0], [0.739112, 0], [0.739085, 0]]"
    )


def test_a_recursion_goes_numeric_step_by_step():
    # The arithmetic-geometric mean of 1 and 2 is 1.4567910310...; computed exactly,
    # each level nests the square roots of the last.
    context = defining(
        APPROXIMATE, AGM=("a,b,n", "IF(n = 0, a, AGM((a + b)/2, SQRT(a*b), n - 1))")
    )
    assert simp("AGM(1, 2, 8)", context) == "1.45679"


def test_a_function_computes_its_own_value_exactly():
    # Only the value a call returns is rounded, so the cancellation inside the body
    # keeps its digits: the value is 0.0049998750...
    context = defining(APPROXIMATE, F=("x", "SQRT(x) - 100"))
    assert simp("F(10001)", context) == "0.00499987"


def test_a_value_with_a_variable_is_carried_exactly():
    context = defining(APPROXIMATE, F=("x", "x/3 + SQRT(2)"))
    assert simp("3*F(y)", context) == "y + 4.24264"


def test_a_long_sum_is_totalled_numerically():
    # The harmonic number H(100) is 5.1873775176...
    assert simp("SUM(1/k, k, 1, 100)", APPROXIMATE) == "5.18737"


def test_an_iteration_inside_approx_goes_numeric_in_an_exact_line():
    # The same mean by ITERATE on a pair. Inside APPROX it is carried numerically,
    # where an exact line would build a tower of radicals.
    text = "1 + APPROX(ITERATE([(a + b)/2, SQRT(a*b)], [a, b], [1, 2], 8) SUB 1)"
    assert simp(text, DECIMAL) == "2.45679"


def test_fitted_data_is_approximated_before_solving():
    # The least squares line through (0.1, TAN(0.1)) ... (0.4, TAN(0.4)).
    data = "VECTOR([x, TAN(x)], x, 0.1, 0.4, 0.1)"
    assert simp(f"FIT([x, a*x + b], {data})", APPROXIMATE) == "1.07400*x - 0.00970691"


def test_solve_searches_numerically_in_approximate_mode():
    # As in Mixed mode, an equation with no closed form is searched for a root.
    assert simp("SOLVE(ERF(x) + x = 1, x)", APPROXIMATE) == "[x = 0.489116]"


def test_solve_inside_approx_searches_numerically():
    text = "1 + APPROX(RHS(SOLVE(ERF(x) + x = 1, x) SUB 1))"
    assert simp(text, DECIMAL) == "1.48911"


def test_an_integral_over_an_infinite_range_of_a_vanishing_integrand():
    # BESSEL.MTH's integral for BESSEL_K(1, 1.3), which is 0.3725474956... Quadrature
    # samples the integrand so far out that `#e^-x` there can't be computed. The
    # simplest rational within tolerance is 4690/12589, shown as 0.372547.
    t = sp.Symbol("t")
    integral = sp.Integral(
        sp.exp(-sp.Rational(13, 10) * sp.cosh(t)) * sp.cosh(t), (t, 0, sp.oo)
    )
    assert approximated(integral, APPROXIMATE) == sp.Rational(4690, 12589)
