"""What Expand must produce.

Author text in, author text out, as the Factor and Simplify cases are, and for
the same reason: the text the worksheet shows is the whole of what the command
promises.

Almost every case here comes from the original rather than from the manual, so
what is recorded is what the original actually prints. Where the engine's form
differs from the original's - term order, an equivalent spelling - the engine's
form is recorded and the case says so; nothing is weakened to make a test pass.
"""

from __future__ import annotations

import pytest

from rederive.engine.computing import (
    Amount,
    Context,
    expand,
    simplify,
    written_as_ratio,
)
from rederive.engine.context import Precision
from rederive.syntax import ParseState, parse_expression

TRIVIAL = Amount.TRIVIAL
SQUAREFREE = Amount.SQUAREFREE
RATIONAL = Amount.RATIONAL
RADICAL = Amount.RADICAL

APPROXIMATE = Context().with_precision(Precision.APPROXIMATE)


def parse(text, state=None):
    return parse_expression(text, state or ParseState()).node


def exp(text, amount=RATIONAL, variables=(), context=None):
    """`text` expanded, as the worksheet would show it."""
    return expand(parse(text), context or Context(), amount, variables).text


# -- polynomial expansion -----------------------------------------------------

POLYNOMIALS = [
    ("2*x*(x - 3)^2", "2*x^3 - 12*x^2 + 18*x"),
    ("x*(x + 1)*(x + 2)", "x^3 + 3*x^2 + 2*x"),
    ("(x + 1)^3", "x^3 + 3*x^2 + 3*x + 1"),
    ("(a + b + c)^2", "a^2 + 2*a*b + 2*a*c + b^2 + 2*b*c + c^2"),
    ("SQRT(x)*(SQRT(x) + 1)", "x + SQRT(x)"),
    # Not a polynomial in anything, and left alone rather than refused.
    ("SIN(x + y)", "SIN(x + y)"),
    ("#e^(x + y)", "#e^(x + y)"),
    ("(x + 1)^(1/2)", "SQRT(x + 1)"),
    # An argument is expanded even where the call around it is not: Expand
    # does not do trigonometric expansion, but `(x + 1)^2` is still a square.
    ("SIN((x + 1)^2)", "SIN(x^2 + 2*x + 1)"),
    # An exponent is not a place Expand reaches.
    ("2^(x + 1)", "2^(x + 1)"),
]


@pytest.mark.parametrize(("text", "expected"), POLYNOMIALS, ids=str)
def test_a_polynomial_is_multiplied_out(text, expected):
    assert exp(text) == expected


def test_a_variable_the_expression_has_not_got_expands_nothing():
    assert exp("(x + 1)^2", variables=["y"]) == "(x + 1)^2"


# -- what the expansion variables change --------------------------------------
#
# The manual's own worked example, which is the clearest statement of the rule
# that everything free of the expansion variables is one opaque quantity.

CUBE = "(x + 2*y + 1)^3"


def test_one_variable_leaves_the_others_as_they_stand():
    assert exp(CUBE, variables=["x"]) == (
        "x^3 + 3*x^2*(2*y + 1) + 3*x*(2*y + 1)^2 + (2*y + 1)^3"
    )
    assert exp(CUBE, variables=["y"]) == (
        "8*y^3 + 12*y^2*(x + 1) + 6*y*(x + 1)^2 + (x + 1)^3"
    )


def test_every_variable_leaves_nothing_unexpanded():
    """The same terms the original prints, and in the order it prints them."""
    assert exp(CUBE) == (
        "x^3 + 6*x^2*y + 3*x^2 + 12*x*y^2 + 12*x*y + 3*x + 8*y^3 + 12*y^2 + 6*y + 1"
    )


def test_terms_of_equal_degree_are_collected():
    """The manual's example: `(b + 1)^2` is not multiplied out on the way,
    because `b` is not one of the expansion variables."""
    text = "3*a^2*x^2*y^3 + 7*(b + 1)^2*x^2*y^3"
    assert exp(text, variables=["x", "y"]) == "x^2*y^3*(3*a^2 + 7*(b + 1)^2)"


def test_a_collected_coefficient_keeps_its_common_factor_outside():
    text = "a*(x + 1)^2 + b*(x + 1)^2"
    assert exp(text, variables=["x"]) == "x^2*(a + b) + 2*x*(a + b) + a + b"


def test_a_factor_free_of_the_variables_is_not_reached_into():
    text = "(x + 1)^2*(y + 1)^2"
    assert exp(text, variables=["x"]) == "x^2*(y + 1)^2 + 2*x*(y + 1)^2 + (y + 1)^2"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Collecting these would be factoring, and Expand does not factor.
        ("2*SIN(x) + 2*COS(x)", "2*COS(x) + 2*SIN(x)"),
        # And this one factors inside the logarithm, which is worse.
        ("LN(x^2 - x) - LN(x)", "LN(x^2 - x) - LN(x)"),
    ],
    ids=str,
)
def test_nothing_is_collected_out_of_the_degree_zero_group(text, expected):
    """What the expansion variables did not reach is the whole expression when
    they reach nothing, and it is not a coefficient of anything."""
    assert exp(text) == expected


def test_a_head_that_binds_a_variable_comes_through_as_itself():
    """Not every argument of one is an operand: a derivative carries the
    variable it is taken over and how many times, a substitution the variables
    it binds and the points they take. Rewriting those reads them as something
    else, so what Expand has nothing to expand it hands back untouched - which
    is exactly what Simplify already answers."""
    text = "DIF(LIM(F(y), y, 2*x - y), y)"
    assert exp(text) == simplify(parse(text), Context()).text
    assert "SUBS(DIF(LIM(F(y), xi_2, 2*x - y), xi_2), [xi_2], [y])" in exp(text)


def test_an_integrand_stays_inside_its_integral():
    """The limits of an integral are not an operand, and standing in for them
    lets the integrand out: `t` is bound by the integral and means nothing
    outside it, so `t^(a - 1)*(...)*INT(1, t, 0, 1/2)` is not another way of
    writing this - it is a different expression."""
    text = "INT(t^(a-1)*((1-t)^(b-a-1)*#e^(t*z) - 1), t, 0, 1/2)"
    assert exp(text, variables=["z"]) == (
        "INT(t^(a - 1)*((1 - t)^(b - a - 1)*#e^(t*z) - 1), t, 0, 1/2)"
    )


# -- partial fraction expansion -----------------------------------------------

FRACTIONS = [
    ("1/(x^2 - 1)", RATIONAL, "1/(2*(x - 1)) - 1/(2*(x + 1))"),
    # Trivial factors nothing out of the denominator, so there is nothing to
    # split over and the ratio comes back as it went in.
    ("1/(x^2 - 1)", TRIVIAL, "1/(x^2 - 1)"),
    ("1/(x^2 - 1)", SQUAREFREE, "1/(x^2 - 1)"),
    # raDical reaches a denominator Rational cannot factor.
    ("1/(x^2 - 2)", RATIONAL, "1/(x^2 - 2)"),
    (
        "1/(x^2 - 2)",
        RADICAL,
        "SQRT(2)/(4*(x - SQRT(2))) - SQRT(2)/(4*(x + SQRT(2)))",
    ),
    # A repeated factor gets one term per power up to its own.
    ("1/(x*(x + 1)^2)", RATIONAL, "1/x - 1/(x + 1) - 1/(x + 1)^2"),
]


@pytest.mark.parametrize(("text", "amount", "expected"), FRACTIONS, ids=str)
def test_the_amount_says_how_far_the_denominator_is_factored(text, amount, expected):
    assert exp(text, amount) == expected


def test_an_improper_ratio_is_divided_out_first():
    """The quotient is expanded and the proper part decomposed; the order the
    two are printed in is the printer's."""
    assert exp("(x^4 + 1)/(x^2 - 1)", variables=["x"]) == (
        "x^2 - 1/(x + 1) + 1/(x - 1) + 1"
    )


def test_the_manual_s_worked_partial_fraction():
    """Section 4.7's example, term for term. `81*(a + 1)` keeps its form: `a`
    is not an expansion variable, so nothing about it is expanded, and the
    term is not split because its numerator holds no expansion variable."""
    text = "(25*x^4 + 81*a*x^2 + 324*a*x + 324*a)/(x^3 + x^2 - 8*x - 12)"
    assert exp(text, variables=["x"]) == (
        "25*x + 81*(a + 1)/(x - 3) + 144/(x + 2) - 80/(x + 2)^2 - 25"
    )


def test_a_numerator_holding_an_expansion_variable_is_split():
    assert exp("(x + 1)/(x^2 + 1)", variables=["x"]) == "x/(x^2 + 1) + 1/(x^2 + 1)"
    assert exp("(x^2 + x + 1)/(x^3 + 1)", variables=["x"]) == (
        "2*x/(3*(x^2 - x + 1)) + 2/(3*(x^2 - x + 1)) + 1/(3*(x + 1))"
    )


def test_the_primary_variable_is_the_first_one_the_denominator_holds():
    """`x/(a^2 - 1)` about `x` alone has no denominator to split, so it is a
    polynomial expansion; about both, `a` is what it splits over."""
    assert exp("x/(a^2 - 1)", variables=["x"]) == "x/(a^2 - 1)"
    assert exp("x/(a^2 - 1)") == "x/(2*(a - 1)) - x/(2*(a + 1))"


def test_a_denominator_free_of_the_variables_is_distributed():
    assert exp("(x + 1)^2/(a^2 - 1)", variables=["x"]) == (
        "x^2/(a^2 - 1) + 2*x/(a^2 - 1) + 1/(a^2 - 1)"
    )
    assert exp("(x + 1)^2/a", variables=["x"]) == "x^2/a + 2*x/a + 1/a"


def test_a_sum_of_a_ratio_and_a_polynomial_is_left_alone():
    """It is not written as a ratio, so there is nothing to decompose, and the
    polynomial expansion finds nothing to multiply out."""
    assert exp("x + 1/(x + 1)") == "x + 1/(x + 1)"


def test_a_multivariate_denominator_splits_over_the_primary_variable():
    assert exp("1/(x^2 - y^2)") == "1/(2*y*(x - y)) - 1/(2*y*(x + y))"


# -- Simplify runs first ------------------------------------------------------


def test_expand_is_simplify_and_then_expanding():
    """The manual says both commands reach a sufficiently simple form and that
    Expand goes further, so an answer may be Simplify's work alone."""
    assert exp("SIN(x)^2 + COS(x)^2") == "1"
    assert exp("2/x + 1/x") == "3/x"
    assert exp("(2 + 3)^2") == "25"


def test_an_expanded_number_is_the_number():
    assert exp("12") == "12"
    assert exp("7/12") == "7/12"


def test_rounding_happens_after_expanding_not_before():
    """A radical decomposition has to reach `SQRT(2)` before Approximate mode
    has anything to round: rounding first would leave a float with nothing to
    factor and so nothing to split over.

    The form is the engine's own. The original answers the same decomposition
    in rational approximations of the radicals rather than in decimals, which
    is not what it does for the same expression under Factor; the engine
    rounds the one way its own Approximate mode rounds everything.
    """
    assert exp("1/(x^2 - 2)", RADICAL, context=APPROXIMATE) == (
        "0.353553/(x - 1.41421) - 0.353553/(x + 1.41421)"
    )


# -- shapes -------------------------------------------------------------------


def test_a_vector_is_expanded_element_by_element():
    assert exp("[(x + 1)^2, (x + 2)^2]") == "[x^2 + 2*x + 1, x^2 + 4*x + 4]"


def test_a_relation_is_expanded_side_by_side():
    assert exp("(x + 1)^2 = (y + 1)^2") == "x^2 + 2*x + 1 = y^2 + 2*y + 1"


def test_a_subtree_is_expanded_on_its_own():
    """Expand works on any subtree, which is what lets the session expand what
    the user has highlighted."""
    node = parse("(x + 1)^2 + SIN(z)")
    part = node.children[0]
    assert expand(part, Context(), RATIONAL).text == "x^2 + 2*x + 1"


# -- what the command asks about ----------------------------------------------

RATIOS = [
    ("1/(x^2 - 1)", True),
    ("7/12", True),
    ("x^2/x", True),
    ("SIN(x)/2", True),
    # Written as a power, not a quotient.
    ("x^-1", False),
    # A quotient inside, but the expression is not one.
    ("-1/x", False),
    ("2*(1/x)", False),
    ("2/x + 1/x", False),
    ("(x + 1)^3", False),
]


@pytest.mark.parametrize(("text", "expected"), RATIOS, ids=str)
def test_only_something_written_as_a_ratio_needs_an_amount(text, expected):
    assert written_as_ratio(parse(text)) is expected


# -- the promises -------------------------------------------------------------

# Every kind of thing the parser produces, expanded. None of these may raise;
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
    "1/(x^2 - 1)",
]


@pytest.mark.parametrize("text", TOTAL, ids=str)
@pytest.mark.parametrize("amount", list(Amount), ids=str)
def test_expanding_never_raises(text, amount):
    expand(parse(text), Context(), amount)


@pytest.mark.parametrize(("text", "expected"), POLYNOMIALS, ids=str)
def test_expanding_an_answer_again_changes_nothing(text, expected):
    once = expand(parse(text), Context())
    assert expand(once.node, Context()).text == once.text


@pytest.mark.parametrize(("text", "amount", "expected"), FRACTIONS, ids=str)
def test_expanding_a_decomposition_again_changes_nothing(text, amount, expected):
    once = expand(parse(text), Context(), amount)
    assert expand(once.node, Context(), amount).text == once.text


# -- EXPAND as an authored function -------------------------------------------

# `EXPAND(u, amount, x, y, ...)` is the author-line spelling of the command,
# so Simplify is what evaluates it. Every expected value below is the
# original's own answer for that very line.
HEADS = [
    ("EXPAND((x + 1)^3)", "x^3 + 3*x^2 + 3*x + 1"),
    (
        "EXPAND((x + 2*y + 1)^3, x)",
        "x^3 + 3*x^2*(2*y + 1) + 3*x*(2*y + 1)^2 + (2*y + 1)^3",
    ),
    ("EXPAND(1/(x^2 - 1), Trivial)", "1/(x^2 - 1)"),
    (
        "EXPAND(1/(x^2 - 2), raDical, x)",
        "SQRT(2)/(4*(x - SQRT(2))) - SQRT(2)/(4*(x + SQRT(2)))",
    ),
    ("EXPAND([(x + 1)^2, (x + 2)^2])", "[x^2 + 2*x + 1, x^2 + 4*x + 4]"),
    # Complex is Factor's amount and not Expand's, so it names no amount here
    # and is read as a variable, which leaves the denominator unfactored.
    ("EXPAND(1/(x^2 + 1), Complex, x)", "1/(x^2 + 1)"),
]


@pytest.mark.parametrize(("text", "expected"), HEADS, ids=str)
def test_simplify_evaluates_an_expand_head(text, expected):
    assert simplify(parse(text), Context()).text == expected


def test_the_amount_argument_may_be_left_out():
    assert simplify(parse("EXPAND(1/(x^2 - 1))"), Context()).text == (
        simplify(parse("EXPAND(1/(x^2 - 1), Rational)"), Context()).text
    )


def test_an_expand_head_is_expanded_where_it_stands():
    """It is an expression like any other, so it may be part of a larger one.

    The engine multiplies the two together where the original writes
    `2*(x^2 + 2*x + 1)`, and that has nothing to do with Expand: authoring
    `2*(x^2 - 1)` and simplifying it divides the two the same way. A numeric
    coefficient distributes over a sum in the conversion layer, before any
    command runs, because the pipeline leans on sympy's automatic evaluation
    for Simplify's basic work - and because distributing is what makes an
    answer read back as the expression it prints. The same trade is recorded
    on the `INT(x^2, x, a, b)` case in `test_simplify`.

    Which makes it a form the engine cannot hold rather than one it declined
    to: Derive keeps whichever of the two was written, leaving `2*x^2 + 4*x + 2`
    split and `2*(x^2 + 2*x + 1)` grouped, so there is no normal form to reach
    for. Both spellings are the same expression.
    """
    assert simplify(parse("2*EXPAND((x + 1)^2)"), Context()).text == (
        "2*x^2 + 4*x + 2"
    )


def test_a_word_that_names_no_amount_is_not_an_amount():
    simplify(parse("EXPAND(w, Turing)"), Context())


def test_only_expand_is_evaluated():
    assert simplify(parse("EXPANDS((x + 1)^2)"), Context()).text == (
        "EXPANDS((x + 1)^2)"
    )


def test_either_head_may_be_written_inside_the_other():
    assert simplify(parse("EXPAND(FACTOR(x^2 - 1))"), Context()).text == "x^2 - 1"
    assert simplify(parse("FACTOR(EXPAND((x + 1)^2))"), Context()).text == "(x + 1)^2"
