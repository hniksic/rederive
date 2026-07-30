"""What Factor must produce.

Author text in, author text out, as the Simplify cases are, and for the same
reason: the text the worksheet shows is the whole of what the command
promises.

Almost every case here comes from the original rather than from the manual,
so what is recorded is what the original actually prints, including the
places the manual is wrong about it. Where the engine's form
differs from the original's - term order, an equivalent spelling - the
engine's form is recorded and the case says so; nothing is weakened to make a
test pass.
"""

from __future__ import annotations

import math

import pytest

from rederive.engine.computing import Amount, Context, factor, simplify
from rederive.engine.context import Precision
from rederive.syntax import ParseState, parse_expression

TRIVIAL = Amount.TRIVIAL
SQUAREFREE = Amount.SQUAREFREE
RATIONAL = Amount.RATIONAL
RADICAL = Amount.RADICAL
COMPLEX = Amount.COMPLEX

APPROXIMATE = Context().with_precision(Precision.APPROXIMATE)


def parse(text, state=None):
    return parse_expression(text, state or ParseState()).node


def fact(text, amount=RATIONAL, variables=(), context=None):
    """`text` factored, as the worksheet would show it."""
    return factor(parse(text), context or Context(), amount, variables).text


# -- the five amounts ---------------------------------------------------------

# Every expected value below is the original's own answer.
AMOUNTS = [
    # Trivial puts the expression over a common denominator and pulls out the
    # gcd of the coefficients and the least power of each variable.
    ("3 - 1/(x + 2)", TRIVIAL, "(3*x + 5)/(x + 2)"),
    ("6*x^2 + 10*x^3/y", TRIVIAL, "2*x^2*(5*x + 3*y)/y"),
    ("2*x^3 - 12*x^2 + 18*x", TRIVIAL, "2*x*(x^2 - 6*x + 9)"),
    # Squarefree also splits off perfect powers, and stops there.
    ("2*x^3 - 12*x^2 + 18*x", SQUAREFREE, "2*x*(x - 3)^2"),
    ("x^4 + 2*x^3 - 3*x^2 - 8*x - 4", SQUAREFREE, "(x + 1)^2*(x^2 - 4)"),
    # Rational also splits products of sums.
    ("x^4 + 2*x^3 - 3*x^2 - 8*x - 4", RATIONAL, "(x - 2)*(x + 1)^2*(x + 2)"),
    ("4*x^2 - 9", RATIONAL, "(2*x - 3)*(2*x + 3)"),
    ("x^2 - 5*x + 6", RATIONAL, "(x - 3)*(x - 2)"),
    ("x^2 - y^2", RATIONAL, "(x - y)*(x + y)"),
    # raDical also allows fractional powers of numbers.
    ("x^2 - 2", RADICAL, "(x - SQRT(2))*(x + SQRT(2))"),
    ("x^4 - 4", RADICAL, "(x - SQRT(2))*(x + SQRT(2))*(x^2 + 2)"),
    # Complex also allows complex numbers.
    ("x^2 + 2", COMPLEX, "(x - SQRT(2)*#i)*(x + SQRT(2)*#i)"),
]


@pytest.mark.parametrize(("text", "amount", "expected"), AMOUNTS, ids=str)
def test_the_amount_says_how_far_to_go(text, amount, expected):
    assert fact(text, amount) == expected


def test_squarefree_stops_at_equal_powers():
    """The manual's own example of squarefree deliberately stopping short.

    `x^2 - 4` is left whole because `x + 2` and `x - 2` are the same power of
    a sum; only Rational goes on to split it.
    """
    text = "x^4 + 2*x^3 - 3*x^2 - 8*x - 4"
    assert fact(text, SQUAREFREE) == "(x + 1)^2*(x^2 - 4)"
    assert fact(text, RATIONAL) == "(x - 2)*(x + 1)^2*(x + 2)"


def test_rational_is_the_default_amount():
    assert fact("x^2 - 5*x + 6") == fact("x^2 - 5*x + 6", RATIONAL)


def test_an_amount_never_undoes_the_one_before_it():
    """Each amount does everything the one before it does, and then more."""
    text = "2*x^3 - 12*x^2 + 18*x"
    assert fact(text, TRIVIAL) == "2*x*(x^2 - 6*x + 9)"
    for amount in (SQUAREFREE, RATIONAL, RADICAL, COMPLEX):
        assert fact(text, amount) == "2*x*(x - 3)^2"


# -- what only the roots can split --------------------------------------------


def test_a_radical_factor_keeps_a_real_quadratic_whole():
    """raDical means as far as the reals allow, not "split or give up".

    `x^3 - 2` has one real root and a conjugate pair. The pair has no real
    linear factors, but their product does, so it comes back as the quadratic
    it multiplies out to.
    """
    assert fact("x^3 - 2", RADICAL) == "(x - 2^(1/3))*(x^2 + 2^(1/3)*x + 2^(2/3))"


def test_a_complex_factor_splits_the_pair_a_radical_one_keeps():
    assert fact("x^2 + 2", RADICAL) == "x^2 + 2"
    assert fact("x^2 + 2", COMPLEX) == "(x - SQRT(2)*#i)*(x + SQRT(2)*#i)"


def test_the_reals_reach_further_than_one_quadratic():
    # Derive prints the same two quadratics, its `x` coefficients written
    # `1/2 - SQRT(5)/2` where the engine writes the one fraction.
    assert fact("x^5 - 1", RADICAL) == (
        "(x - 1)*(x^2 + x*(1 - SQRT(5))/2 + 1)*(x^2 + x*(1 + SQRT(5))/2 + 1)"
    )


def test_a_root_nobody_can_place_is_treated_as_real():
    """The quadratic formula, where nothing is known about a, b or c.

    Derive splits this, so a root that cannot be *shown* to be real has to be
    allowed rather than rejected. Its form is the original's with the sign of
    the second factor carried inside the numerator instead of before it.
    """
    assert fact("a*x^2 + b*x + c", RADICAL, ("x",)) == (
        "a*(x + (b - SQRT(b^2 - 4*a*c))/(2*a))*(x + (b + SQRT(b^2 - 4*a*c))/(2*a))"
    )


def test_a_factor_whose_roots_are_unknown_stays_whole():
    """A quintic with no radical solution is left alone, not approximated."""
    assert fact("x^5 - x + 1", RADICAL) == "x^5 - x + 1"


# -- the factorization variables ----------------------------------------------


def test_naming_fewer_variables_leaves_the_rest_alone():
    """The manual's rule that a subexpression in no factorization variable is
    neither expanded nor factored."""
    text = "x^2*y^2 - x^2 - y^4 + y^2"
    assert fact(text, RATIONAL, ("x", "y")) == "(x - y)*(x + y)*(y - 1)*(y + 1)"
    assert fact(text, RATIONAL, ("x",)) == "(x - y)*(x + y)*(y^2 - 1)"


def test_similar_powers_of_a_factoring_variable_are_collected():
    text = "x^2*y^4 - 2*x^2*y^2 + x^2 + 2*x*y^2 - 2*x + 1"
    assert fact(text, RATIONAL, ("x",)) == "(x*(y^2 - 1) + 1)^2"


def test_naming_every_variable_is_naming_none():
    text = "x^2*y^2 - x^2 - y^4 + y^2"
    assert fact(text, RATIONAL, ("x", "y")) == fact(text, RATIONAL)


def test_a_variable_the_expression_does_not_mention_selects_nothing():
    assert fact("x^2 - 4", RATIONAL, ("z",)) == fact("x^2 - 4", RATIONAL)


def test_a_polynomial_in_something_that_is_not_a_variable():
    """Nothing chosen lets sympy pick its own generator, which is what makes
    this factor in `SIN(x)` rather than in `x`, where it factors in nothing."""
    assert fact("SIN(x)^2 - 1", RATIONAL) == "(SIN(x) - 1)*(SIN(x) + 1)"


# -- numbers ------------------------------------------------------------------

NUMBERS = [
    ("1234567890", "2*3^2*5*3607*3803"),
    ("1234567890/49", "2*3^2*5*3607*3803/7^2"),
    ("10!", "2^8*3^4*5^2*7"),
    ("-12", "-2^2*3"),
    # Nothing to decompose: these are already products of primes.
    ("7", "7"),
    ("1/7", "1/7"),
    ("0", "0"),
    ("1", "1"),
    ("-1", "-1"),
]


@pytest.mark.parametrize(("text", "expected"), NUMBERS, ids=str)
def test_a_number_is_its_prime_decomposition(text, expected):
    assert fact(text) == expected


def test_the_amount_does_not_reach_a_number():
    """Which is why the original does not ask for one when the whole
    expression is a number."""
    for amount in Amount:
        assert fact("1234567890", amount) == "2*3^2*5*3607*3803"


def test_a_large_factorial_is_decomposed():
    assert fact("54!") == (
        "2^50*3^26*5^12*7^8*11^4*13^4*17^3*19^2*23^2*29*31*37*41*43*47*53"
    )


def test_the_decomposition_is_the_number_it_came_from():
    assert simplify(parse(fact("54!"))).text == str(math.factorial(54))


def test_a_numeral_longer_than_the_interpreter_converts_by_default():
    """A number the printer can write is one the parser can read back.

    CPython converts at most 4300 digits between an int and a numeral unless it
    is told otherwise, and `5000!` runs to 16326 of them. Factor is where the
    ceiling shows: Simplify reaches the number from a numeral short enough to
    convert, and factoring the answer it printed has to read the whole of it
    back in.
    """
    written = simplify(parse("5000!")).text
    assert written == str(math.factorial(5000))
    assert fact(written) == fact("5000!")


def test_a_number_that_is_no_rational_has_no_decomposition():
    assert fact("SQRT(2)") == "SQRT(2)"
    assert fact("pi") == "pi"


def test_a_polynomials_coefficients_are_not_decomposed():
    """The manual is explicit: factoring a polynomial leaves its numeric
    coefficients alone. The `12` stays a `12`."""
    assert fact("12*x^2 - 12") == "12*(x - 1)*(x + 1)"


# -- what an entry is built out of --------------------------------------------


def test_a_relation_is_factored_side_by_side():
    assert fact("x^2 - 5*x + 6 = 0") == "(x - 3)*(x - 2) = 0"


def test_a_vector_is_factored_element_by_element():
    assert fact("[x^2 - 1, 2*x + 2]") == "[(x - 1)*(x + 1), 2*(x + 1)]"


def test_a_matrix_is_factored_element_by_element():
    assert fact("[[x^2 - 1, 0], [0, x^2 - 4]]") == (
        "[[(x - 1)*(x + 1), 0], [0, (x - 2)*(x + 2)]]"
    )


def test_an_assignment_keeps_its_shape():
    assert fact("w := x^2 - 1") == "w := (x - 1)*(x + 1)"


# -- precision ----------------------------------------------------------------


def test_rounding_happens_after_factoring_not_before():
    """Approximate mode has to reach SQRT(2) before it can show 1.41421.

    Factoring a number that had already been rounded would find nothing to
    factor, so the precision mode is applied to the answer rather than to the
    input.
    """
    assert fact("x^2 - 2", RADICAL, context=APPROXIMATE) == (
        "(x - 1.41421)*(x + 1.41421)"
    )
    assert fact("x^2 + 2", COMPLEX, context=APPROXIMATE) == (
        "(x - 1.41421*#i)*(x + 1.41421*#i)"
    )


def test_an_exact_factorization_is_exact():
    assert fact("x^2 - 2", RADICAL) == "(x - SQRT(2))*(x + SQRT(2))"


@pytest.mark.parametrize("mode", list(Precision), ids=str)
def test_a_decomposition_is_never_rounded(mode):
    """A prime decomposition says what divides what, so no precision mode
    reaches it: the original prints it whole in Approximate mode too.

    Rounding it would not just lose the factors, it would lose the number -
    six significant digits of `1234567890` is a different integer.
    """
    context = Context(precision=mode)
    assert fact("1234567890", context=context) == "2*3^2*5*3607*3803"
    assert fact("1/3", context=context) == "1/3"


def test_rounding_reaches_the_factors_beside_a_decomposition():
    """The precision mode applies scalar by scalar, so one entry can hold both
    an exact decomposition and a rounded factorization."""
    assert fact("[1234567890, x^2 - 2]", RADICAL, context=APPROXIMATE) == (
        "[2*3^2*5*3607*3803, (x - 1.41421)*(x + 1.41421)]"
    )


# -- the promises -------------------------------------------------------------

# Every kind of thing the parser produces, factored. None of these may raise;
# what they come back as is not the point.
TOTAL = [
    "x",
    "?",
    "1/0",
    "INT(#e^(x^2), x)",
    "DIF(F(x)^3, x)",
    "LIM(SIN(x)/x, x, 0)",
    "SUM(k^2, k, 1, n)",
    "SOLVE(x^2 = 2, x)",
    "IF(x > 0, x, -x)",
    "x :epsilon Real [0, inf)",
    "F(y) := y^2 - 1",
    "[[1, 2], [3, 4]]",
    "x > 1 AND x < 3",
    "#e^(#i*pi)",
    "(x^2 - 1)^(1/3)",
    "RANDOM(5)",
    "x^2 - 1 = y^2 - 1",
]


@pytest.mark.parametrize("text", TOTAL, ids=str)
@pytest.mark.parametrize("amount", list(Amount), ids=str)
def test_factoring_never_raises(text, amount):
    factor(parse(text), Context(), amount)


@pytest.mark.parametrize(("text", "amount", "expected"), AMOUNTS, ids=str)
def test_factoring_an_answer_again_changes_nothing(text, amount, expected):
    once = factor(parse(text), Context(), amount)
    assert factor(once.node, Context(), amount).text == once.text


# -- FACTOR as an authored function -------------------------------------------

# `FACTOR(u, amount, x, y, ...)` is the author-line spelling of the command,
# so Simplify is what evaluates it. Every expected value below is the
# original's own answer for that very line.
HEADS = [
    ("FACTOR(4*x^3 - 8*x^2 - 11*x - 3, Rational, x)", "(x - 3)*(2*x + 1)^2"),
    ("FACTOR(x^2 - 1)", "(x - 1)*(x + 1)"),
    ("FACTOR(x^2 - 2, raDical)", "(x - SQRT(2))*(x + SQRT(2))"),
    ("FACTOR(10!)", "2^8*3^4*5^2*7"),
    ("FACTOR([x^2 - 1, 2*x + 2], Trivial)", "[x^2 - 1, 2*(x + 1)]"),
    ("FACTOR(2*x^3 - 12*x^2 + 18*x, Trivial)", "2*x*(x^2 - 6*x + 9)"),
    (
        "FACTOR(x^2*y^2 - x^2 - y^4 + y^2, Rational, x)",
        "(x - y)*(x + y)*(y^2 - 1)",
    ),
]


@pytest.mark.parametrize(("text", "expected"), HEADS, ids=str)
def test_simplify_evaluates_a_factor_head(text, expected):
    assert simplify(parse(text), Context()).text == expected


def test_the_amount_argument_may_be_left_out():
    assert simplify(parse("FACTOR(x^2 - 1)"), Context()).text == (
        simplify(parse("FACTOR(x^2 - 1, Rational)"), Context()).text
    )


def test_a_factor_head_is_factored_where_it_stands():
    """It is an expression like any other, so it may be part of a larger one."""
    assert simplify(parse("2*FACTOR(x^2 - 1)"), Context()).text == (
        "2*(x - 1)*(x + 1)"
    )


def test_nothing_later_undoes_a_factor_head():
    """Everything that reshapes a sum - the gated rewrites and the normal form
    alike - runs before this, so none of them is offered the chance to multiply
    it back out."""
    assert simplify(parse("FACTOR(x^2 - 4)"), Context()).text == "(x - 2)*(x + 2)"


def test_a_word_that_names_no_amount_is_not_an_amount():
    """`Turing` is a Derive 6 matrix factoring this engine has no mathematics
    for. It must not be read as an amount, and it must not raise."""
    simplify(parse("FACTOR(w, Turing)"), Context())


def test_only_factor_is_evaluated():
    # FACTORS is the syntactic decomposition and not this command under another
    # name: a number is one factor, however many primes it is a product of.
    assert simplify(parse("FACTORS(10!)"), Context()).text == "[3628800]"


def test_a_subtree_is_factored_on_its_own():
    """Factor works on any subtree, which is what lets the session factor what
    the user has highlighted."""
    node = parse("(x^2 - 1) + SIN(z)")
    part = node.children[0]
    assert factor(part, Context(), RATIONAL).text == "(x - 1)*(x + 1)"


# -- what the engine does not do yet ------------------------------------------


@pytest.mark.skip(reason="a Simplify gap, not a Factor one: see the test below")
def test_a_pythagorean_difference_becomes_a_single_square():
    assert fact("SIN(x)^2 - 1") == "-COS(x)^2"


def test_the_pythagorean_gap_belongs_to_simplify():
    """Derive answers `-COS(x)^2` for the *Simplify* of `SIN(x)^2 - 1`, so
    Factor prints it only because Simplify ran first.

    This engine's Simplify leaves the difference alone, which is why Factor
    answers with the factorization instead. Closing it is a `pipeline` change,
    and this case is here to say where it belongs; when Simplify learns the
    identity, this test is what will fail and point at the one above.
    """
    assert simplify(parse("SIN(x)^2 - 1"), Context()).text == "SIN(x)^2 - 1"
