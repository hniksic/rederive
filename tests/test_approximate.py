"""Simplify at non-exact precision.

Covers Approximate mode, the approX command and `APPROX` function, Mixed mode, and
Decimal notation. Cases are written inline, as in `test_simplify.py`, so a failure shows
the mathematics. Key rules: approX is Simplify at Approximate precision, so `pi` gives
`3.141592653` either way; `1 + APPROX(pi)` in an exact line is the rational
`412221/99532`, but a line that is just `APPROX(pi)` is approX of `pi`; whole numbers
never get decimal digits; and an approximation shows only digits that are correct for the
value.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import sympy as sp
from simplifying import declaring, parse, simp

from rederive.engine.approximation import approximated, carried
from rederive.engine.computing import Context, approx
from rederive.engine.context import Definition, Notation, Precision
from rederive.engine.vectors import Vector
from rederive.model.session import Session
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
    [("pi", "3.141592653"), ("SQRT(3)", "1.732050807"), ("1/2 + 1/6", "0.6666666666")],
    ids=str,
)
def test_approximate_mode(text, expected):
    assert simp(text, APPROXIMATE) == expected


def test_an_ordering_with_an_undefined_side_is_false_in_numbers():
    """Tests are decided in numbers, where no ordering with an undefined value holds:
    a constant has no inverse, and the program goes on to its other branch."""
    assert simp("IF(ABS(ITERATE(-7, x, 0, -1)) >= 0, 1, 2)", APPROXIMATE) == "2"
    assert simp("IF(ABS(ITERATE(-7, x, 0, -1)) >= 0, 1, 2)") == "IF(? >= 0, 1, 2)"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A power of a negative number to a rational with a large denominator is too
        # costly to build exactly, and is approximated as its principal value.
        ("5*(-1)^(307249147/1500000000)", "4.000000000 + 2.999999999*#i"),
        ("(-3)^(1234567891/1000000000)", "-2.874679746 - 2.608632609*#i"),
    ],
    ids=str,
)
def test_a_held_power_of_a_negative_number_approximates_to_its_principal_value(
    text, expected
):
    assert simp(text, APPROXIMATE) == expected


@pytest.mark.parametrize(
    ("text", "expected"), [("pi", "3.141592653"), ("SQRT(3)", "1.732050807")], ids=str
)
def test_approx_is_simplify_at_another_precision(text, expected):
    # The manual defines approX as Simplify with precision set to Approximate, so it
    # gives the same answers from an exact context.
    assert approx(parse(text), Context()).text == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    # A simpler rational lies within tolerance just below each, and choosing it would
    # show a wrong digit.
    [
        ("3.141592653", "3.141592653"),
        ("1.732050807", "1.732050807"),
        ("0.333333", "0.333333"),
    ],
    ids=str,
)
def test_a_typed_decimal_keeps_its_digits(text, expected):
    assert simp(text, APPROXIMATE) == expected


@pytest.mark.parametrize(
    ("text", "precision", "notation", "expected"),
    [
        # Derive User Group Newsletter 22, WUNDER1.MTH.
        ("x^4 + 4.99999*x^3", 2, 6, "x^4 + 5*x^3"),
        ("x^4 + 4.99999*x^3", 2, 2, "x^4 + 5*x^3"),
        ("x^4 + 4.99999*x^3", 2, 10, "x^4 + 5*x^3"),
        ("4.99999", 3, 6, "5"),
        ("9.99999", 2, 6, "10"),
        ("365.99999", 2, 6, "366"),
        ("365.99999", 6, 6, "366"),
        ("0.12399", 3, 3, "0.124"),
        # Rounding down, the value keeps its truncated digits.
        ("4.94999", 3, 3, "4.95"),
        ("4.99999", 6, 6, "4.99999"),
    ],
    ids=str,
)
def test_a_decimal_that_rounds_up_becomes_its_rounding(
    text, precision, notation, expected
):
    """Near the next digit, the simplest rational within tolerance is that digit, which
    is the value's rounding: `4.99999` to two digits is 5, not `4.995`, which has digits
    the value has at no precision, shown when NotationDigits exceeds PrecisionDigits."""
    context = replace(
        APPROXIMATE, precision_digits=precision, notation_digits=notation
    )
    assert simp(text, context) == expected


def test_approx_takes_the_digits_it_is_given():
    assert approx(parse("pi"), Context(), 12).text == "3.14159265358"


APPROXIMATED = [
    # The function inside a line, not the command: it approximates its argument and
    # leaves the rest of the line alone, in Exact mode too. The result is a rational,
    # so rational notation shows the ratio the digits stand for.
    ("1 + APPROX(pi)", "412221/99532"),
    ("[APPROX(1/3), APPROX(pi)]", "[1/3, 312689/99532]"),
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
        ("1 + APPROX(pi)", "4.141592653"),
        ("[APPROX(1/3), APPROX(pi)]", "[0.3333333333, 3.141592653]"),
        ("1/2 + APPROX(1/3)", "0.8333333333"),
    ],
    ids=str,
)
def test_what_the_approx_function_leaves_shows_its_digits(text, expected):
    # The same values as above under Decimal notation: the function decides the value,
    # the notation decides how it is written.
    assert simp(text, DECIMAL) == expected


#: `ATAN(1/2) + ATAN(1/3)` is `pi/4`, which exact comparison can't establish, so exact
#: SORT leaves this list unsorted.
_UNDECIDED = "[pi/4, 1/2, ATAN(1/2) + ATAN(1/3)]"


def test_sort_compares_approximated_elements():
    """Under Approximate, SORT sorts the approximated elements, so the order agrees with
    the values shown and is always decided. Elements equal at working precision keep
    their order. Other expressions follow the numbers and variables."""
    line = "SORT([3, y, pi, #e, SQRT(2), x, 1/3, -3])"
    assert simp(line, APPROXIMATE) == (
        "[-3, 0.3333333333, 1.414213562, 2.718281828, 3, 3.141592653, x, y]"
    )
    assert simp(f"SORT({_UNDECIDED})") == f"SORT({_UNDECIDED})"
    sorted_ = "[0.5, 0.7853981633, 0.7853981633]"
    assert simp(f"SORT({_UNDECIDED})", APPROXIMATE) == sorted_
    assert simp(f"APPROX(SORT({_UNDECIDED}))") == sorted_
    assert simp("SORT([x + 1, 2])", APPROXIMATE) == "[2, x + 1]"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # An approximate number has only its precision's worth of digits, so Decimal
        # notation writes one far from 1 with a power of ten, as Scientific does,
        # rather than padding with meaningless zeros.
        ("APPROX(#e^(pi*SQRT(163)))", "2.625374126*10^17"),
        ("APPROX(EXP(100))", "2.688117141*10^43"),
        ("APPROX(EXP(-100))", "3.720075976*10^-44"),
        ("APPROX(100!)", "9.332621544*10^157"),
        ("APPROX(LN(1 + 10^(-15)))", "1.000000000*10^-15"),
        ("APPROX(SQRT(2)*10^20)", "1.414213562*10^20"),
        ("APPROX(10^(-7))", "10^-7"),
        ("APPROX(2^100)*x", "1.267650600*10^30*x"),
        # An approximate whole number with more digits than the precision is
        # shortened too.
        ("APPROX(123456789)", "1.23456789*10^8"),
        # A moderate magnitude is written plainly.
        ("APPROX(1234)", "1234"),
        ("APPROX(0.000123)", "0.000123"),
        # An exact number is written out in full, however long.
        ("100!", str(math.factorial(100))),
        ("1/2^100", "0.0000000000000000000000000000007888609052"),
        ("123456789", "123456789"),
    ],
    ids=str,
)
def test_decimal_notation_writes_an_approximate_number_to_its_precision(text, expected):
    assert simp(text, DECIMAL) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("EXP(100)", "2.688117141*10^43"),
        ("2^100", "1.267650600*10^30"),
        ("x/3", "0.3333333333*x"),
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
    assert simp("SQRT(2)*10^20", context) == "1.414213562*10^20"
    assert simp("10^20/3", context) == "33333333333333333333.3"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("APPROX(1/2)", "0.5"),
        ("APPROX(pi)", "3.141592653"),
        ("APPROX([1/3, pi])", "[0.3333333333, 3.141592653]"),
        ("APPROX(x/3)", "0.3333333333*x"),
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

    Decimal notation shows ten digits, not the fifteen the approximation was made to.
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
    assert simp("y + APPROX(INT(1/x, x, 1, 2))") == "y + 261740/377611"
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
    assert simp("SQRT(3)", MIXED) == "1.732050807"
    # Truncated, not rounded: the sum is 3.47492598692..., whose ten digits are
    # 3.474925986.
    assert simp("1/3 + pi", MIXED) == "3.474925986"
    # The manual's example: the difference of the fractions is exact, so its root is
    # exactly 2/3. Approximate mode rounds them first and loses that.
    exercise = "SQRT(3422357/2313 - 1140443/771)"
    assert simp(exercise, MIXED) == "2/3"
    assert simp(exercise, APPROXIMATE) != "2/3"


#: Non-real numbers are approximated part by part, real and imaginary. The answers are
#: the original's.
COMPLEX_APPROXIMATED = [
    ("SQRT(-2)", "1.414213562*#i"),
    ("(-8)^(1/3)", "1 + 1.732050807*#i"),
    ("LN(-1)", "3.141592653*#i"),
    ("(SQRT(3) + #i)^5", "-27.71281292 + 16*#i"),
    ("(SQRT(2) + #i)^3", "-1.414213562 + 5*#i"),
]


@pytest.mark.parametrize(("text", "expected"), COMPLEX_APPROXIMATED, ids=str)
@pytest.mark.parametrize("context", [APPROXIMATE, MIXED], ids=["approximate", "mixed"])
def test_a_complex_number_is_approximated_part_by_part(text, expected, context):
    assert simp(text, context) == expected


def test_a_function_of_a_function_value_gets_the_digits_it_needs():
    # `TAN(11 + 30*#i)` is within `10^-26` of the arctangent's pole at `#i`.
    assert simp("ATAN(TAN(11 + 30*#i))", APPROXIMATE) == "-1.566370614 + 30*#i"


def test_an_approximated_solution_stays_a_relation():
    # `x` is real, but a complex root is a solution soLve found, not `false`.
    answer = simp("SOLVE(x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 0, x)", APPROXIMATE)
    assert answer.count("x = ") == 6
    assert answer.count(" OR ") == 5


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
    ("SQRT(10^200 + 1) - 10^100", "5.000000000*10^-101"),
    ("x + SQRT(1000003) - SQRT(1000002)", "x + 0.0004999993750"),
    # Zero, though unproven: no nonzero value is that small relative to its parts.
    ("SQRT(2) + SQRT(3) - SQRT(5 + 2*SQRT(6))", "0"),
    # An imaginary part much smaller than the real part keeps its own digits.
    ("(SQRT(2) + #i/10^8)^2 - 2", "-1.000000000*10^-16 + 2.828427124*10^-8*#i"),
    # So does one below the precision of the real part, where it is a part of the
    # value rather than noise.
    ("EXP(#i/10^30)", "1 + 1.000000000*10^-30*#i"),
    # The rest go wrong if intermediate values are rounded to working precision.
    # A huge argument reduced by a rounded pi loses the sine entirely.
    ("SIN(10^20)", "-0.6452512852"),
    ("SIN(10^100)", "-0.3723761236"),
    ("COS(10^10)", "0.8731196226"),
    # Ramanujan's near-integer, whose exponent must be known to many more digits
    # than the result shows.
    ("#e^(pi*SQRT(163))", "2.625374126*10^17"),
    # The 1 would swallow the 10^-15 before the logarithm is taken. The value is
    # 9.9999999999999999950*10^-16, which rounds up.
    ("LN(1 + 10^(-15))", "1.000000000*10^-15"),
    # Both powers would overflow before they are divided.
    ("2^(10^20)/2^(10^20)", "1"),
    # Powers too big to compute exactly, of a base working precision would round to 1:
    # the base gets as many extra digits as the exponent has.
    ("(1 + 10^(-20))^(10^20)", "2.718281828"),
    ("(1 + 10^(-10))^(10^10)", "2.718281828"),
    ("(1 - 10^(-20))^(10^20)", "0.3678794411"),
    ("(1 + 10^(-20))^(2*10^20)", "7.389056098"),
    ("(-1 - 10^(-20))^(10^20 + 1)", "-2.718281828"),
    # Catastrophic cancellation.
    ("SQRT(10001) - 100", "0.004999875006"),
    # The quadrature error must stay below the last digit shown.
    ("INT(SQRT(1 - x^4), x, 0, 1)", "0.8740191847"),
]


@pytest.mark.parametrize(("text", "expected"), EXACT_DIGITS, ids=str)
@pytest.mark.parametrize("context", [APPROXIMATE, MIXED], ids=["approximate", "mixed"])
def test_an_approximation_shows_only_digits_of_the_value(text, expected, context):
    assert simp(text, context) == expected


def test_a_power_near_one_to_a_high_exponent_keeps_its_digits():
    """Rounded to working precision, `1 + 10^-5` is 1, and so is its power; computed
    exactly, it is close to `#e`. Only Approximate mode is checked, since Mixed keeps
    the half-million-digit exact rational until it is written."""
    assert simp("(1 + 10^(-5))^(10^5)", APPROXIMATE) == "2.718268237"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A root of a long rational is not factored, but its digits within the step
        # are all there: the difference cancels sixteen of them.
        (
            "SQRT(100000000000000000000000000000001) - 10000000000000000",
            "5.000000000*10^-17",
        ),
        ("SQRT(1234567890123456789/987654321987)", "1118.033983"),
        ("SQRT(1234567890123456789^2/987654321987^2)", "1.249999987*10^6"),
        ("(1234567890123456789/987654321987)^(3/2) - 1", "1.397542463*10^9"),
        (
            "ABS([1234567890123456789/987654321987, 2345678901234567/3456789])",
            "6.785727682*10^8",
        ),
        (
            "ABS(1234567890123456789/987654321987 + 2345678901234567/3456789*#i)",
            "6.785727682*10^8",
        ),
        ("ABS(3 + 4*#i)", "5"),
    ],
)
def test_a_root_of_a_long_rational_keeps_every_digit(text, expected):
    assert simp(text, APPROXIMATE) == expected


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
        name: Definition(tuple(parameters.split(",")), parse(body, state))
        for name, (parameters, body) in definitions.items()
    }
    return replace(context, functions={**context.functions, **functions})


def test_every_number_in_a_matrix_is_approximated():
    assert simp("[[SQRT(2), 1], [pi, 1/3]]", APPROXIMATE) == (
        "[[1.414213562, 1], [3.141592653, 0.3333333333]]"
    )


def test_the_numbers_of_a_carried_vector_are_rounded():
    # A program's vector of numbers is rounded as it is stored, as a number is: a
    # Gauss-Newton fit stores its parameters and the inverse of a matrix product each
    # step, and kept exact their digits grew into the thousands.
    huge, x = sp.Rational(3**500, 7**300), sp.Symbol("x")
    vector = Vector(Vector(huge, sp.Rational(1, 3)), Vector(x, sp.Integer(5)))
    rounded = carried(vector, Context().with_precision(Precision.APPROXIMATE))
    (first, third), (variable, five) = (row.args for row in rounded.args)
    assert first == carried(huge, Context().with_precision(Precision.APPROXIMATE))
    assert first.q < 10**20 and third.q < 10**20
    assert (variable, five) == (x, 5)


def test_a_system_iterates_numerically_in_approximate_mode():
    # Newton's method on the first component; left exact, each iterate nests the last
    # one inside its sine and cosine.
    text = "ITERATES([x - (x - COS(x))/(1 + SIN(x)), y], [x, y], [1, 0], 3)"
    assert simp(text, APPROXIMATE) == (
        "[[1, 0], [0.7503638678, 0], [0.7391128909, 0], [0.7390851333, 0]]"
    )


def test_an_iteration_with_a_symbolic_count_runs_until_it_repeats():
    # SOLVE.MTH's `NEWTON(u, x, x0, n)` called without `n`, which Derive then runs to
    # convergence; exactly, Newton's method for SQRT(3) never repeats.
    text = "ITERATES((x^2 + 3)/(2*x), x, 2, n)"
    assert simp(text, APPROXIMATE) == (
        "[2, 1.75, 1.732142857, 1.732050810, 1.732050807, 1.732050807]"
    )


def test_an_iteration_settling_on_a_rounding_boundary_has_converged():
    # At four digits the iterates end `0.7181, 0.7182, 0.7181`; the steps between them
    # shrink, so this is the limit seen through rounding, not a cycle.
    four = Context().with_precision(Precision.APPROXIMATE, 4)
    assert simp("ITERATE(-0.5*SIN(t) + pi/3, t, pi/3)", four) == "0.7181"
    assert simp("ITERATE(1/x, x, 1.0001)", four) == "?"


def test_a_test_between_numbers_is_decided_on_their_approximations():
    # Muller's method picks the root of larger magnitude: decided exactly, the complex
    # radicals took seconds a step.
    text = (
        "IF(ABS(27/2 + SQRT(2993/16 - 22*#i)) >= ABS(27/2 - SQRT(2993/16 - 22*#i)),"
        " 1, 2)"
    )
    assert simp(text, APPROXIMATE) == "1"


def test_a_recursion_goes_numeric_step_by_step():
    # The arithmetic-geometric mean of 1 and 2 is 1.4567910310...; computed exactly,
    # each level nests the square roots of the last.
    context = defining(
        APPROXIMATE, AGM=("a,b,n", "IF(n = 0, a, AGM((a + b)/2, SQRT(a*b), n - 1))")
    )
    assert simp("AGM(1, 2, 8)", context) == "1.456791031"


def test_a_function_computes_its_own_value_exactly():
    # Only the value a call returns is rounded, so the cancellation inside the body
    # keeps its digits: the value is 0.0049998750...
    context = defining(APPROXIMATE, F=("x", "SQRT(x) - 100"))
    assert simp("F(10001)", context) == "0.004999875006"


def test_a_value_with_a_variable_is_carried_exactly():
    context = defining(APPROXIMATE, F=("x", "x/3 + SQRT(2)"))
    assert simp("3*F(y)", context) == "y + 4.242640687"


@pytest.mark.parametrize("context", [APPROXIMATE, MIXED], ids=["approximate", "mixed"])
@pytest.mark.parametrize(
    "text",
    [
        "PROG(q := SOLUTIONS(s^2 - 2, s), LIM((s - q SUB 1)/(s^2 - 2), s, q SUB 1))",
        "PROG(q := SQRT(2), LIM((s - q)/(s^2 - 2), s, q))",
        "LIM((s - R(s^2 - 2) SUB 1)/(s^2 - 2), s, R(s^2 - 2) SUB 1)",
    ],
    ids=str,
)
def test_a_root_a_program_stores_stays_a_root(text, context):
    """A number whose exact form is short is carried exactly, so a limit at a root of
    the denominator, stored in a local or returned by a function, is the limit there,
    `SQRT(2)/4`. Rounded before the limit, the root would not make the denominator
    vanish, and the limit would be 0."""
    context = defining(context, R=("p", "SOLUTIONS(p, s)"))
    assert simp(text, context) == "0.3535533905"


def test_a_program_stores_an_exact_value_with_a_variable_exactly():
    # Simplified as a command at the same precision would be, the coefficient would be
    # rounded first and the limit taken where the denominator does not vanish.
    text = "PROG(q := SQRT(2)*x, LIM((s - q)/(s^2 - 2*x^2), s, q))"
    assert simp(text, APPROXIMATE) == "0.3535533905/x"


def test_a_long_sum_is_totalled_numerically():
    # The harmonic number H(100) is 5.1873775176...
    assert simp("SUM(1/k, k, 1, 100)", APPROXIMATE) == "5.187377517"


def test_an_iteration_inside_approx_goes_numeric_in_an_exact_line():
    # The same mean by ITERATE on a pair. Inside APPROX it is carried numerically,
    # where an exact line would build a tower of radicals.
    text = "1 + APPROX(ITERATE([(a + b)/2, SQRT(a*b)], [a, b], [1, 2], 8) SUB 1)"
    assert simp(text, DECIMAL) == "2.456791031"


def test_fitted_data_is_approximated_before_solving():
    # The least squares line through (0.1, TAN(0.1)) ... (0.4, TAN(0.4)).
    data = "VECTOR([x, TAN(x)], x, 0.1, 0.4, 0.1)"
    assert simp(f"FIT([x, a*x + b], {data})", APPROXIMATE) == (
        "1.074001854*x - 0.009706919529"
    )


def test_a_fit_is_approximated_as_its_model_is():
    # Derive User Group Newsletter 64, traffic.dfw: `SIN(pi/3*(t - 16.5))` is
    # `COS(pi*t/3)`, in the fit as in the model with symbolic coefficients.
    model = (
        "a*SIN(pi/12*(t - 16.5)) + b*SIN(pi/12*(t - 8.5)) + d*SIN(pi/6*(t - 16.5))"
        " + f*SIN(pi/3*(t - 8.5)) + g*SIN(pi/3*(t - 16.5)) + e*SIN(pi/6*(t - 8.5)) + c"
    )
    assert simp(f"FIT([t, {model}], VECTOR([i, i^2], i, 0, 24))", APPROXIMATE) == (
        "29.17481610*COS(1.047197551*t) + 209.0857058*SIN(0.2617993878*t - 4.319689898)"
        " + 168.0206041*SIN(0.2617993878*t - 2.225294796)"
        " + 95.08895904*SIN(0.5235987756*t - 8.639379797)"
        " + 86.28102160*SIN(0.5235987756*t - 4.450589592)"
        " + 48*SIN(1.047197551*t - 8.901179185) + 192.7540747"
    )


def test_a_fit_to_approximate_data_has_coefficients_at_their_own_scale():
    # FIT_.DMO's weighted linearization of a rational fit; numpy's least squares gives
    # 0.675509, -1.365447, -0.003606 and 1.444968. Solved exactly, the coefficients
    # are ratios of integers hundreds of digits long, shown as `1.42469*10^-230*w*(...)`.
    data = (
        "VECTOR([t, TAN(t), 1/ABS(t - 1.56725), t*TAN(t)/ABS(t - 1.56725)], t, 0, 1, 0.1)"
    )
    assert simp(f"FIT([x, y, w, w*(a*x^2 + b*x + c - d*y)], {data})", APPROXIMATE) == (
        "w*(0.6755088280*x^2 - 1.365447228*x + 1.444967922*y - 0.003605910238)"
    )


#: FIT_.MTH's rational fit: the numerator minus `y` times the denominator is fitted to
#: zero, and the fitted function is that solved for `y`.
_RATIONAL = ("f", "LIM(f, y, 0)/(-DIF(f, y))")
_LINEARIZED = "FIT([x, y, a*x^2 + b*x + c - d*y - x*y], VECTOR([t, TAN(t), 0], t, 0, {}, 0.1))"


def test_a_rational_fit_is_written_with_its_fitted_coefficients():
    # numpy: 0.907135, -0.897811, -0.0000577 and -0.906611 (the digits here are cut,
    # not rounded). The denominator is the template's `x + d`, as NUMERATOR and
    # DENOMINATOR, which FIT_ reweights with, also give it.
    context = defining(APPROXIMATE, G=_RATIONAL)
    fitted = f"G({_LINEARIZED.format(0.5)})"
    assert simp(fitted, context) == (
        "(0.9071346548*x^2 - 0.8978113610*x - 5.768141267*10^-5)/(x - 0.9066108799)"
    )
    assert simp(f"DENOMINATOR({fitted})", context) == "x - 0.9066108799"
    assert simp(f"NUMERATOR({fitted})", context) == (
        "0.9071346548*x^2 - 0.8978113610*x - 5.768141267*10^-5"
    )


def test_a_reweighted_rational_fit_stays_numeric():
    # FIT_'s one default iteration on FIT_.DMO's sixteen points, reweighted by the
    # reciprocal of the first fit's denominator. numpy: 0.452405, -1.313302, -0.029738
    # and -1.569618. With the fit solved exactly, this took minutes.
    weighted = (
        "G(FIT([x, y, w, w*(a*x^2 + b*x + c - d*y - x*y)], "
        "VECTOR([t, TAN(t), ABS(LIM(1/DENOMINATOR(u), x, t)), 0], t, 0, 1.5, 0.1)))"
    )
    context = defining(APPROXIMATE, G=_RATIONAL, H=("u", weighted))
    first = f"G({_LINEARIZED.format(1.5)})"
    assert simp(f"ITERATE(H(u), u, {first}, 1)", context) == (
        "(0.4524053719*x^2 - 1.313302296*x - 0.02973761600)/(x - 1.569618076)"
    )


#: APPROX.MTH's Pade approximant, which FIT_.DMO plots beside its fits.
_PADE = {
    "CF": ("a,m,n", "VECTOR(VECTOR(a SUB (m-n+i+j),j,1,n+1),i,1,n)"),
    "DV": ("a,m,n", "ELEMENT(ROW_REDUCE(CF(a,m,n))`,n+1)"),
    "NV": (
        "a,u,m,n",
        "VECTOR(a SUB (k+1)-SUM(u SUB (n+1-i)*a SUB (k+1-i),i,1,k),k,1,MIN(m,n))",
    ),
    "P3": (
        "a,v,u,x,c,m,n",
        "(a SUB 1+SUM(v SUB k*(x-c)^k,k,1,m))/(1-SUM(u SUB (n+1-k)*(x-c)^k,k,1,n))",
    ),
    "P2": ("a,u,x,c,m,n", "P3(a,NV(a,u,m,n),u,x,c,m,n)"),
    "P1": ("a,x,c,m,n", "P2(a,DV(a,m,n),x,c,m,n)"),
    "PADE": ("y,x,p,n,d", "P1(VECTOR(LIM(DIF(y,x,m_),x,p)/m_!,m_,0,n+d+1),x,p,n,d)"),
}


def test_an_approximate_rational_function_is_written_at_the_scale_of_its_value():
    # The coefficients carried between PADE's functions are long ratios. Cleared out
    # of the denominator, as the normal form does, they showed as `6.27810*10^-12*
    # (-1.44677*10^42*x^2 + ...)/(-5.83213*10^30*x^2 + ...)`. At x = 1 the value is
    # TAN(1), 1.55740.
    context = defining(APPROXIMATE, **_PADE)
    pade = "PADE(TAN(x), x, 1, 2, 2)"
    assert simp(pade, context) == (
        "(1.557407724*x^2 - 6.114815449*x - 0.1148154493)"
        "/(x^2 + 2.672223173*x - 6.672223173)"
    )
    # The coefficient of x is 3*TAN(1) - 2, 2.67222317396..., shown truncated above
    # and rounded here, each from the digits carried to its own computation.
    assert simp(f"DENOMINATOR({pade})", context) == "x^2 + 2.672223174*x - 6.672223173"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Ratios of short integers keep the normal form, as in CURV_FIT.DOC.
        ("3655.76/(9783*t - 11564)", "3655.76/(9783*t - 1.1564*10^4)"),
        # So does a single long number.
        ("1/(123456789012345*x + 1)", "1/(1.234567890*10^14*x + 1)"),
    ],
)
def test_an_approximate_quotient_of_short_numbers_keeps_the_normal_form(text, expected):
    assert simp(text, APPROXIMATE) == expected


def test_solve_searches_numerically_in_approximate_mode():
    # As in Mixed mode, an equation with no closed form is searched for a root.
    assert simp("SOLVE(ERF(x) + x = 1, x)", APPROXIMATE) == "x = 0.4891163449"


def test_solve_inside_approx_searches_numerically():
    text = "1 + APPROX(RHS(SOLVE(ERF(x) + x = 1, x)))"
    assert simp(text, DECIMAL) == "1.489116344"


def test_an_integral_over_an_infinite_range_of_a_vanishing_integrand():
    # BESSEL.MTH's integral for BESSEL_K(1, 1.3), which is 0.3725474956... Quadrature
    # samples the integrand so far out that `#e^-x` there can't be computed. The
    # simplest rational within tolerance is 186938/501783, shown as 0.3725474956.
    t = sp.Symbol("t")
    integral = sp.Integral(
        sp.exp(-sp.Rational(13, 10) * sp.cosh(t)) * sp.cosh(t), (t, 0, sp.oo)
    )
    assert approximated(integral, APPROXIMATE) == sp.Rational(186938, 501783)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Derive approximates the logarithms before the exponential is split into
        # powers of 2 and 5 (CURV_FIT's G_1).
        ("#e^(t*(LN(5) - LN(2))/3)", "#e^(0.3054302439*t)"),
        ("5^(t/3)", "5^(0.3333333333*t)"),
        ("2^x*3^y", "2^x*3^y"),
        # The constant part of an exponent is a factor: CURV_FIT.DOC's exponential fit
        # is `3.17338*#e^(-0.701797*t)`.
        ("#e^(x + 1)", "2.718281828*#e^x"),
        ("#e^2*#e^(-t)", "7.389056098*#e^(-t)"),
    ],
)
def test_an_approximated_exponential_has_a_numeric_exponent_and_coefficient(
    text, expected
):
    assert simp(text, APPROXIMATE) == expected


def test_curv_fit_s_exponential_and_power_fits_come_out_as_documented():
    """CURV_FIT.DOC fits `3.17338 #e^(-0.701797 t)` and `2.19899/t^(1.90553)` to the
    logarithms of its data; the exact fit of the logarithms is quick too."""
    data = "[[1, 1.5], [2, 0.8], [3, 0.4], [4, 0.2], [5, 0.09], [6, 0.047]]"
    logs = f"VECTOR([{data} SUB k SUB 1, LN({data} SUB k SUB 2)], k, 1, 6)"
    both = f"VECTOR([LN({data} SUB k SUB 1), LN({data} SUB k SUB 2)], k, 1, 6)"
    line = f"FIT([x, a*x + b], {logs})"
    slope = f"(LIM({line}, x, 1) - LIM({line}, x, 0))"
    fitted = f"#e^LIM({line}, x, 0)*#e^(t*{slope})"
    assert simp(fitted, APPROXIMATE) == "3.173383384*#e^(-0.7017976365*t)"
    line = f"FIT([x, a*x + b], {both})"
    slope = f"(LIM({line}, x, 1) - LIM({line}, x, 0))"
    fitted = f"#e^LIM({line}, x, 0)*t^{slope}"
    assert simp(fitted, APPROXIMATE) == "2.198996223/t^(1.905530878)"
    assert "x" in simp(f"FIT([x, a*x + b], {logs})")


@pytest.mark.parametrize(
    "text, expected",
    [
        # BESSEL.MTH's `BESSEL_I(1, 1.3)` times pi: I_1(1.3) is 0.7973293149... There is
        # no closed form, and heuristic Risch would search for a minute before
        # quadrature; it is skipped for an integral that is going to be approximated.
        ("INT(#e^(1.3*COS(t))*COS(t), t, 0, pi)", "2.504883918"),
        # A pole inside the range is still found.
        ("INT(1/x^2, x, -1, 1)", "inf"),
        # ZETA.MTH's `DILOG(-1)`, which is Li2(2) = pi^2/4 - pi*LN(2)*#i: the range is cut
        # at the logarithm's branch point, where sympy's quadrature never settles.
        ("INT(LN(t)/(1 - t), t, 1, -1)", "2.467401100 - 2.177586090*#i"),
        # `DILOG(2)`, Li2(-1) = -pi^2/12.
        ("INT(LN(t)/(1 - t), t, 1, 2)", "-0.8224670334"),
        # HYPERGEO.MTH's `KUMMER` integrand, on which sympy's definite integration
        # searches for minutes, and FUN.MTH's `JN(2/3, 2)`, with no closed form.
        ("INT(t^2*(#e^(2*t)*(1 - t)^(-2.5) - 1), t, 0, 1/2)", "0.2957644458"),
        ("INT(t^2*#e^(2*t)*(1 - t)^(-5/2), t, 0, 1/2)", "0.3374311124"),
        ("INT((1 - a^2)^(1/6)*COS(2*a), a, 0, 1)", "0.4579236772"),
    ],
)
def test_a_numeric_integral_is_approximated_by_quadrature(text, expected):
    assert simp(text, APPROXIMATE) == expected


def test_an_exact_closed_form_is_rounded_once():
    """A partial sum of the cosine series at 20 adds terms near 10^7 to about 0.4:
    rounded term by term, the digits that cancel are lost."""
    text = "SUBST(SUM((-1)^n*x^(2*n)/(2*n)!, n, 0, 30), x, 20)"
    assert simp(f"APPROX({text})") == "0.4080953868"
    assert simp(text, APPROXIMATE) == "0.4080953868"


def test_a_decimal_sum_keeps_its_coefficients():
    # A number goes in front of a sum only with the sum's whole content; otherwise into
    # the sum, which a decimal notation shows best.
    assert simp("(412.9563894*x + 412.9563894*y)*a", APPROXIMATE) == (
        "412.9563894*a*(x + y)"
    )
    assert simp("(3.1*x + 3.2*y)*a", APPROXIMATE) == "a*(3.1*x + 3.2*y)"


async def test_newtons_method_settling_on_an_integer_has_converged():
    # The carried iterates of x end a little above and a little below -1, shown as
    # -1.000000000 and -0.9999999999: both are -1 to the digits asked for, so the
    # iteration stops where Derive's does.
    session = Session()
    answer = await session.approx(
        "NEWTONS([3*x^2 + z - 10, x - 4*y^2 + 2*z + 15, x^2 + y^2 - 8], [x, y, z], "
        "[0, -2, -1])"
    )
    assert answer.text.endswith(
        "[-1.000000000, -2.645751311, 7], [-1, -2.645751311, 7]]"
    )
