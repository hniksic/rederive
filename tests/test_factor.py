"""Tests of the Factor command.

Like the Simplify cases, these compare text in to text out, since the displayed text is
what the command promises.

Nearly all cases come from what the original actually prints, even where the manual
disagrees. Where the engine's form differs (term order, an equivalent spelling), the
engine's form is recorded and the case says so; nothing is weakened to make a test pass.
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
    """Factor `text` and return it as the worksheet would show it."""
    return factor(parse(text), context or Context(), amount, variables).text


# -- the five amounts ---------------------------------------------------------

# Every expected value below is the original's answer, factor order included: of two
# factors differing only in a coefficient's sign, the positive one comes first, as in
# `x^2 - 4` -> `(x + 2)*(x - 2)`.
AMOUNTS = [
    # Trivial puts the expression over a common denominator and pulls out the gcd of
    # the coefficients and the lowest power of each variable.
    ("3 - 1/(x + 2)", TRIVIAL, "(3*x + 5)/(x + 2)"),
    ("6*x^2 + 10*x^3/y", TRIVIAL, "2*x^2*(5*x + 3*y)/y"),
    ("2*x^3 - 12*x^2 + 18*x", TRIVIAL, "2*x*(x^2 - 6*x + 9)"),
    # Exponentials are content as powers of variables are, with the least exponent in
    # each term of the exponents taken out. ODE.MTH's `BARE` relies on it.
    ("a*#e^(2*t) - #e^(2*t - b)", TRIVIAL, "(a*#e^b - 1)*#e^(2*t - b)"),
    ("a*#e^(x - y) + c*#e^(x + y)", TRIVIAL, "(c*#e^(2*y) + a)*#e^(x - y)"),
    ("#e^(x + 1) - 1", TRIVIAL, "#e^(x + 1) - 1"),
    # Squarefree also splits off perfect powers, and stops there.
    ("2*x^3 - 12*x^2 + 18*x", SQUAREFREE, "2*x*(x - 3)^2"),
    ("x^4 + 2*x^3 - 3*x^2 - 8*x - 4", SQUAREFREE, "(x + 1)^2*(x^2 - 4)"),
    # Rational also splits products of sums.
    ("x^4 + 2*x^3 - 3*x^2 - 8*x - 4", RATIONAL, "(x + 2)*(x - 2)*(x + 1)^2"),
    ("4*x^2 - 9", RATIONAL, "(2*x + 3)*(2*x - 3)"),
    ("x^2 - 5*x + 6", RATIONAL, "(x - 2)*(x - 3)"),
    ("x^2 - y^2", RATIONAL, "(x + y)*(x - y)"),
    # raDical also allows fractional powers of numbers.
    ("x^2 - 2", RADICAL, "(x + SQRT(2))*(x - SQRT(2))"),
    ("x^4 - 4", RADICAL, "(x + SQRT(2))*(x - SQRT(2))*(x^2 + 2)"),
    # Complex also allows complex numbers.
    ("x^2 + 2", COMPLEX, "(x + SQRT(2)*#i)*(x - SQRT(2)*#i)"),
]


@pytest.mark.parametrize(("text", "amount", "expected"), AMOUNTS, ids=str)
def test_the_amount_says_how_far_to_go(text, amount, expected):
    assert fact(text, amount) == expected


def test_squarefree_stops_at_equal_powers():
    """The manual's example of Squarefree deliberately stopping short.

    `x^2 - 4` stays whole because `x + 2` and `x - 2` appear to the same power; only
    Rational splits it.
    """
    text = "x^4 + 2*x^3 - 3*x^2 - 8*x - 4"
    assert fact(text, SQUAREFREE) == "(x + 1)^2*(x^2 - 4)"
    assert fact(text, RATIONAL) == "(x + 2)*(x - 2)*(x + 1)^2"


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
    """raDical factors as far as the reals allow, not "fully split or give up".

    `x^3 - 2` has one real root and a conjugate pair. The pair gives no real linear
    factors, but its product is a real quadratic, which is kept.
    """
    assert fact("x^3 - 2", RADICAL) == "(x - 2^(1/3))*(x^2 + 2^(1/3)*x + 2^(2/3))"


def test_a_complex_factor_splits_the_pair_a_radical_one_keeps():
    assert fact("x^2 + 2", RADICAL) == "x^2 + 2"
    assert fact("x^2 + 2", COMPLEX) == "(x + SQRT(2)*#i)*(x - SQRT(2)*#i)"


def test_the_reals_reach_further_than_one_quadratic():
    # Derive prints the same two quadratics, but writes the `x` coefficients as
    # `1/2 - SQRT(5)/2` where the engine writes a single fraction.
    assert fact("x^5 - 1", RADICAL) == (
        "(x - 1)*(x^2 + (SQRT(5) + 1)*x/2 + 1)*(x^2 + (1 - SQRT(5))*x/2 + 1)"
    )


def test_a_root_nobody_can_place_is_treated_as_real():
    """The quadratic formula, with nothing known about a, b or c.

    Derive splits this, so a root that can't be *shown* to be real must be allowed. The
    form is the original's (4.6, p.79): each factor puts the root's sign in the operator
    rather than on the numerator's leading term.
    """
    assert fact("a*x^2 + b*x + c", RADICAL, ("x",)) == (
        "a*(x + (SQRT(b^2 - 4*a*c) + b)/(2*a))*(x - (SQRT(b^2 - 4*a*c) - b)/(2*a))"
    )


def test_a_factor_whose_roots_are_unknown_stays_whole():
    """A quintic with no radical solution is left alone, not approximated."""
    assert fact("x^5 - x + 1", RADICAL) == "x^5 - x + 1"


def test_an_irreducible_cubic_with_three_real_roots_splits_over_the_reals():
    """The casus irreducibilis, which Cardano's formula can only write using `#i`.

    All three roots are real, but Cardano writes them as nested radicals around an
    imaginary unit that can't be simplified away, which is unreadable. Viete's
    trigonometric form is visibly real, and it is what Derive prints.
    """
    assert fact("x^3 - 3*x + 1", RADICAL) == (
        "(x + 2*COS(pi/9))*(x - 2*COS(2*pi/9))*(x - 2*COS(4*pi/9))"
    )


# -- the factorization variables ----------------------------------------------


def test_naming_fewer_variables_leaves_the_rest_alone():
    """Per the manual, a subexpression free of the factorization variables is neither
    expanded nor factored."""
    text = "x^2*y^2 - x^2 - y^4 + y^2"
    assert fact(text, RATIONAL, ("x", "y")) == "(x + y)*(x - y)*(y + 1)*(y - 1)"
    assert fact(text, RATIONAL, ("x",)) == "(x + y)*(x - y)*(y^2 - 1)"


def test_similar_powers_of_a_factoring_variable_are_collected():
    text = "x^2*y^4 - 2*x^2*y^2 + x^2 + 2*x*y^2 - 2*x + 1"
    assert fact(text, RATIONAL, ("x",)) == "(x*(y^2 - 1) + 1)^2"


def test_naming_every_variable_is_naming_none():
    text = "x^2*y^2 - x^2 - y^4 + y^2"
    assert fact(text, RATIONAL, ("x", "y")) == fact(text, RATIONAL)


def test_a_variable_the_expression_does_not_mention_selects_nothing():
    assert fact("x^2 - 4", RATIONAL, ("z",)) == fact("x^2 - 4", RATIONAL)


def test_a_polynomial_in_something_that_is_not_a_variable():
    """With no variables given, sympy picks its own generator, so this factors in
    `SIN(x)`; in `x` it wouldn't factor at all."""
    assert fact("SIN(x)^2 - 4", RATIONAL) == "(SIN(x) - 2)*(SIN(x) + 2)"


# -- numbers ------------------------------------------------------------------

NUMBERS = [
    ("1234567890", "2*3^2*5*3607*3803"),
    ("1234567890/49", "2*3^2*5*3607*3803/7^2"),
    ("10!", "2^8*3^4*5^2*7"),
    ("-12", "-2^2*3"),
    # Nothing to decompose.
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
    """Which is why the original doesn't ask for an amount when the expression is a
    number."""
    for amount in Amount:
        assert fact("1234567890", amount) == "2*3^2*5*3607*3803"


def test_a_number_no_trial_division_reaches_is_decomposed_anyway():
    """The search is unbounded, so semiprimes beyond trial division still split.

    These are out of reach of both trial division and Fermat's method (the factors are
    far apart); elliptic-curve factorization finds them. Sympy disables that whenever
    it is given a `limit`, so a bounded search would leave them unfactored.
    """
    assert fact("2^101 - 1") == "7432339208719*341117531003194129"
    assert fact("2^128 + 1") == "59649589127497217*5704689200685129054721"


def test_a_large_factorial_is_decomposed():
    assert fact("54!") == (
        "2^50*3^26*5^12*7^8*11^4*13^4*17^3*19^2*23^2*29*31*37*41*43*47*53"
    )


def test_the_decomposition_is_the_number_it_came_from():
    assert simplify(parse(fact("54!"))).text == str(math.factorial(54))


def test_a_numeral_longer_than_the_interpreter_converts_by_default():
    """Any number the printer can write, the parser can read back.

    By default CPython converts at most 4300 digits between int and str, and `5000!`
    has 16326. Factoring Simplify's printed answer has to parse all of it.
    """
    written = simplify(parse("5000!")).text
    assert written == str(math.factorial(5000))
    assert fact(written) == fact("5000!")


def test_a_number_that_is_no_rational_has_no_decomposition():
    assert fact("SQRT(2)") == "SQRT(2)"
    assert fact("pi") == "pi"


def test_a_polynomials_coefficients_are_not_decomposed():
    """Per the manual, factoring a polynomial leaves its numeric coefficients alone."""
    assert fact("12*x^2 - 12") == "12*(x + 1)*(x - 1)"


# -- what an entry is built out of --------------------------------------------


def test_a_relation_is_factored_side_by_side():
    assert fact("x^2 - 5*x + 6 = 0") == "(x - 2)*(x - 3) = 0"


def test_a_vector_is_factored_element_by_element():
    assert fact("[x^2 - 1, 2*x + 2]") == "[(x + 1)*(x - 1), 2*(x + 1)]"


def test_a_matrix_is_factored_element_by_element():
    assert fact("[[x^2 - 1, 0], [0, x^2 - 4]]") == (
        "[[(x + 1)*(x - 1), 0], [0, (x + 2)*(x - 2)]]"
    )


def test_an_assignment_keeps_its_shape():
    assert fact("w := x^2 - 1") == "w := (x + 1)*(x - 1)"


# -- precision ----------------------------------------------------------------


def test_rounding_happens_after_factoring_not_before():
    """Approximate mode has to find SQRT(2) before it can show 1.41421.

    Rounding first would leave nothing to factor, so precision applies to the answer,
    not the input.
    """
    assert fact("x^2 - 2", RADICAL, context=APPROXIMATE) == (
        "(x + 1.41421)*(x - 1.41421)"
    )
    assert fact("x^2 + 2", COMPLEX, context=APPROXIMATE) == (
        "(x + 1.41421*#i)*(x - 1.41421*#i)"
    )


def test_an_exact_factorization_is_exact():
    assert fact("x^2 - 2", RADICAL) == "(x + SQRT(2))*(x - SQRT(2))"


@pytest.mark.parametrize("mode", list(Precision), ids=str)
def test_a_decomposition_is_never_rounded(mode):
    """No precision mode rounds a prime decomposition; the original prints it whole in
    Approximate mode too.

    Rounding would lose not just the factors but the number: six significant digits of
    `1234567890` is a different integer.
    """
    context = Context(precision=mode)
    assert fact("1234567890", context=context) == "2*3^2*5*3607*3803"
    assert fact("1/3", context=context) == "1/3"


def test_rounding_reaches_the_factors_beside_a_decomposition():
    """Precision applies per scalar, so one entry can hold both an exact decomposition
    and a rounded factorization."""
    assert fact("[1234567890, x^2 - 2]", RADICAL, context=APPROXIMATE) == (
        "[2*3^2*5*3607*3803, (x + 1.41421)*(x - 1.41421)]"
    )


# -- the promises -------------------------------------------------------------

# One of every kind of expression the parser produces. None may raise; the result
# doesn't matter.
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

# `FACTOR(u, amount, x, y, ...)` is the command written as a function, evaluated by
# Simplify. Every expected value is the original's answer for that line.
HEADS = [
    ("FACTOR(4*x^3 - 8*x^2 - 11*x - 3, Rational, x)", "(x - 3)*(2*x + 1)^2"),
    ("FACTOR(x^2 - 1)", "(x + 1)*(x - 1)"),
    ("FACTOR(x^2 - 2, raDical)", "(x + SQRT(2))*(x - SQRT(2))"),
    ("FACTOR(x^4 + 1, raDical)", "(x^2 + SQRT(2)*x + 1)*(x^2 - SQRT(2)*x + 1)"),
    ("FACTOR(x^3 - 2, raDical)", "(x - 2^(1/3))*(x^2 + 2^(1/3)*x + 2^(2/3))"),
    ("FACTOR(x^2 + 1, Complex)", "(x + #i)*(x - #i)"),
    ("FACTOR(10!)", "2^8*3^4*5^2*7"),
    ("FACTOR([x^2 - 1, 2*x + 2], Trivial)", "[x^2 - 1, 2*(x + 1)]"),
    ("FACTOR(2*x^3 - 12*x^2 + 18*x, Trivial)", "2*x*(x^2 - 6*x + 9)"),
    (
        "FACTOR(x^2*y^2 - x^2 - y^4 + y^2, Rational, x)",
        "(x + y)*(x - y)*(y^2 - 1)",
    ),
]


@pytest.mark.parametrize(("text", "expected"), HEADS, ids=str)
def test_simplify_evaluates_a_factor_head(text, expected):
    assert simplify(parse(text), Context()).text == expected


def test_a_factor_head_splits_what_the_radicals_split():
    # Two rational quadratics, each split by its own square root.
    assert simplify(parse("FACTOR(x^4 - 5*x^2 + 6, raDical)"), Context()).text == (
        "(x + SQRT(2))*(x + SQRT(3))*(x - SQRT(2))*(x - SQRT(3))"
    )
    # A polynomial with radical coefficients is factored over their field, even at the
    # default amount.
    assert simplify(parse("FACTOR(x^2 - x*SQRT(2) - x + SQRT(2))"), Context()).text == (
        "(x - 1)*(x - SQRT(2))"
    )


def test_the_amount_argument_may_be_left_out():
    assert simplify(parse("FACTOR(x^2 - 1)"), Context()).text == (
        simplify(parse("FACTOR(x^2 - 1, Rational)"), Context()).text
    )


def test_a_factor_head_is_factored_where_it_stands():
    """`FACTOR` may be part of a larger expression."""
    assert simplify(parse("2*FACTOR(x^2 - 1)"), Context()).text == (
        "2*(x + 1)*(x - 1)"
    )


def test_a_factor_head_in_a_sum_is_the_sum_s_to_write():
    """A sum is written in normal form regardless of its terms, so a factored polynomial
    in it is multiplied out; a factored number is not a polynomial and stays. Checked
    against the original."""
    assert simplify(parse("FACTOR(x^2 - 1) + x^3"), Context()).text == "x^3 + x^2 - 1"
    assert simplify(parse("x*FACTOR(x^2 - 1) + 1"), Context()).text == "x^3 - x + 1"
    assert simplify(parse("FACTOR(12) + 1"), Context()).text == "2^2*3 + 1"


def test_nothing_later_undoes_a_factor_head():
    """Everything that reshapes a sum (gated rewrites and the normal form) runs before
    `FACTOR`, so nothing multiplies it back out."""
    assert simplify(parse("FACTOR(x^2 - 4)"), Context()).text == "(x + 2)*(x - 2)"


def test_a_word_that_names_no_amount_is_not_an_amount():
    """`Turing` is a Derive 6 matrix factorization this engine doesn't implement. It
    must not be read as an amount, and must not raise."""
    simplify(parse("FACTOR(w, Turing)"), Context())


def test_only_factor_is_evaluated():
    # FACTORS is a syntactic decomposition, not this command: a number is a single
    # factor, however many primes divide it.
    assert simplify(parse("FACTORS(10!)"), Context()).text == "[3628800]"


def test_the_factors_of_a_factored_number_are_its_prime_powers():
    """As in Derive, they are shown as Factor wrote them, so like Factor's own answer
    they are a form a second Simplify multiplies out."""
    assert simplify(parse("FACTORS(FACTOR(360))"), Context()).text == "[2^3, 3^2, 5]"
    assert simplify(parse("FACTORS(FACTOR(12/35))"), Context()).text == (
        "[2^2, 3, 1/5, 1/7]"
    )
    # A negative number's sign comes first (D).
    assert simplify(parse("FACTORS(FACTOR(-12))"), Context()).text == "[-1, 2^2, 3]"


def test_a_subtree_is_factored_on_its_own():
    """Factor works on any subtree, so the session can factor the user's selection."""
    node = parse("(x^2 - 1) + SIN(z)")
    part = node.children[0]
    assert factor(part, Context(), RATIONAL).text == "(x + 1)*(x - 1)"


# -- the Pythagorean identity -------------------------------------------------


def test_a_pythagorean_difference_becomes_a_single_square():
    assert fact("SIN(x)^2 - 1") == "-COS(x)^2"


def test_simplify_reduces_a_pythagorean_difference():
    """Derive's Simplify turns `SIN(x)^2 - 1` into `-COS(x)^2`, a cancellation of one
    term, so Factor shows it too."""
    assert simplify(parse("SIN(x)^2 - 1"), Context()).text == "-COS(x)^2"
