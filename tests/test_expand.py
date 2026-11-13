"""What Expand must produce.

Like the Factor and Simplify cases, these go from author text to author text, since the
worksheet text is what the command promises.

Almost every case comes from running the original, not from the manual. Where the
engine's form differs (term order, an equivalent spelling), the engine's form is
recorded and the case says so; nothing is weakened to make a test pass.
"""

from __future__ import annotations

import pytest
import sympy as sp

from rederive.engine.computing import (
    Amount,
    Context,
    expand,
    simplify,
    written_as_ratio,
)
from rederive.engine.context import Precision
from rederive.engine import expanding
from rederive.engine.expanding import held_about
from rederive.syntax import FunctionDeclaration, ParseState, parse_expression

TRIVIAL = Amount.TRIVIAL
SQUAREFREE = Amount.SQUAREFREE
RATIONAL = Amount.RATIONAL
RADICAL = Amount.RADICAL

APPROXIMATE = Context().with_precision(Precision.APPROXIMATE)


def parse(text, state=None):
    return parse_expression(text, state or ParseState()).node


def exp(text, amount=RATIONAL, variables=(), context=None):
    """Expand `text` and return it as the worksheet would show it."""
    return expand(parse(text), context or Context(), amount, variables).text


# -- polynomial expansion -----------------------------------------------------

POLYNOMIALS = [
    ("2*x*(x - 3)^2", "2*x^3 - 12*x^2 + 18*x"),
    ("x*(x + 1)*(x + 2)", "x^3 + 3*x^2 + 2*x"),
    ("(x + 1)^3", "x^3 + 3*x^2 + 3*x + 1"),
    ("(a + b + c)^2", "a^2 + 2*a*b + 2*a*c + b^2 + 2*b*c + c^2"),
    ("SQRT(x)*(SQRT(x) + 1)", "x + SQRT(x)"),
    # Not polynomials; left alone rather than refused.
    ("SIN(x + y)", "SIN(x + y)"),
    ("#e^(x + y)", "#e^(x + y)"),
    ("(x + 1)^(1/2)", "SQRT(x + 1)"),
    # Arguments are expanded even when the call isn't: Expand does no trigonometric
    # expansion, but `(x + 1)^2` is still a square.
    ("SIN((x + 1)^2)", "SIN(x^2 + 2*x + 1)"),
    # Expand doesn't reach into exponents.
    ("2^(x + 1)", "2^(x + 1)"),
]


@pytest.mark.parametrize(("text", "expected"), POLYNOMIALS, ids=str)
def test_a_polynomial_is_multiplied_out(text, expected):
    assert exp(text) == expected


def test_a_variable_the_expression_has_not_got_expands_nothing():
    assert exp("(x + 1)^2", variables=["y"]) == "(x + 1)^2"


# -- what the expansion variables change --------------------------------------
#
# The manual's worked example, which shows clearly that anything free of the expansion
# variables is treated as an opaque quantity.

CUBE = "(x + 2*y + 1)^3"


def test_one_variable_leaves_the_others_as_they_stand():
    assert exp(CUBE, variables=["x"]) == (
        "x^3 + 3*x^2*(2*y + 1) + 3*x*(2*y + 1)^2 + (2*y + 1)^3"
    )
    assert exp(CUBE, variables=["y"]) == (
        "8*y^3 + 12*y^2*(x + 1) + 6*y*(x + 1)^2 + (x + 1)^3"
    )


def test_every_variable_leaves_nothing_unexpanded():
    """The same terms the original prints, in the same order."""
    assert exp(CUBE) == (
        "x^3 + 6*x^2*y + 3*x^2 + 12*x*y^2 + 12*x*y + 3*x + 8*y^3 + 12*y^2 + 6*y + 1"
    )


def test_terms_of_equal_degree_are_collected():
    """The manual's example: `(b + 1)^2` is not multiplied out, because `b` is not an
    expansion variable."""
    text = "3*a^2*x^2*y^3 + 7*(b + 1)^2*x^2*y^3"
    assert exp(text, variables=["x", "y"]) == "x^2*y^3*(3*a^2 + 7*(b + 1)^2)"


def test_a_collected_coefficient_keeps_its_common_factor_outside():
    # Simplify first factors out the common `(x + 1)^2`, so `(a + b)*(x + 1)^2` is
    # expanded, and collecting keeps the middle coefficient's `2` outside the bracket.
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
        # And here it would factor inside the logarithm, which is worse.
        ("LN(x^2 - x) - LN(x)", "LN(x^2 - x) - LN(x)"),
    ],
    ids=str,
)
def test_nothing_is_collected_out_of_the_degree_zero_group(text, expected):
    """When no term contains an expansion variable, the whole expression is the degree
    zero group, and its terms aren't collected into a coefficient."""
    assert exp(text) == expected


def test_a_head_that_binds_a_variable_comes_through_as_itself():
    """Not every argument of a binding head is an operand: a derivative carries its
    variable and order, a substitution its bound variables and their values. Rewriting
    those would change the meaning, so Expand returns such a head as Simplify does."""
    state = ParseState()
    state.declare(FunctionDeclaration("G", ("y", "z")))
    text = "DIF(LIM(G(y, z), y, 2*x - y), y)"
    answer = expand(parse(text, state), Context(), RATIONAL, (), state).text
    assert answer == simplify(parse(text, state), Context(), state).text
    assert "SUBS(DIF(G(x_, z), x_), [x_], [2*x - y])" in answer


def test_an_integrand_stays_inside_its_integral():
    """The integration variable `t` is bound by the integral and means nothing outside it,
    so pulling factors of the integrand out, as in `t^(a - 1)*(...)*INT(1, t, 0, 1/2)`,
    would give a different expression."""
    text = "INT(t^(a-1)*((1-t)^(b-a-1)*#e^(t*z) - 1), t, 0, 1/2)"
    assert exp(text, variables=["z"]) == (
        "INT(t^(a - 1)*((1 - t)^(-a + b - 1)*#e^(t*z) - 1), t, 0, 1/2)"
    )


# -- partial fraction expansion -----------------------------------------------

FRACTIONS = [
    ("1/(x^2 - 1)", RATIONAL, "1/(2*(x - 1)) - 1/(2*(x + 1))"),
    # Trivial factors nothing out of the denominator, so the ratio is unchanged.
    ("1/(x^2 - 1)", TRIVIAL, "1/(x^2 - 1)"),
    ("1/(x^2 - 1)", SQUAREFREE, "1/(x^2 - 1)"),
    # raDical factors a denominator that Rational can't.
    ("1/(x^2 - 2)", RATIONAL, "1/(x^2 - 2)"),
    (
        "1/(x^2 - 2)",
        RADICAL,
        "SQRT(2)/(4*(x - SQRT(2))) - SQRT(2)/(4*(x + SQRT(2)))",
    ),
    # A repeated factor gets one term per power up to its multiplicity.
    ("1/(x*(x + 1)^2)", RATIONAL, "1/x - 1/(x + 1) - 1/(x + 1)^2"),
]


@pytest.mark.parametrize(("text", "amount", "expected"), FRACTIONS, ids=str)
def test_the_amount_says_how_far_the_denominator_is_factored(text, amount, expected):
    assert exp(text, amount) == expected


def test_an_improper_ratio_is_divided_out_first():
    """The quotient is expanded and the proper part decomposed; the printer decides the
    term order."""
    assert exp("(x^4 + 1)/(x^2 - 1)", variables=["x"]) == (
        "x^2 - 1/(x + 1) + 1/(x - 1) + 1"
    )


def test_the_manual_s_worked_partial_fraction():
    """Section 4.7's example, term for term. `81*(a + 1)` is kept as is: `a` is not an
    expansion variable, and the term isn't split because its numerator contains no
    expansion variable."""
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
    """With only `x`, the denominator has no expansion variable, so this is a polynomial
    expansion; with both variables, it splits over `a`."""
    assert exp("x/(a^2 - 1)", variables=["x"]) == "x/(a^2 - 1)"
    assert exp("x/(a^2 - 1)") == "x/(2*(a - 1)) - x/(2*(a + 1))"


def test_a_denominator_free_of_the_variables_is_distributed():
    assert exp("(x + 1)^2/(a^2 - 1)", variables=["x"]) == (
        "x^2/(a^2 - 1) + 2*x/(a^2 - 1) + 1/(a^2 - 1)"
    )
    assert exp("(x + 1)^2/a", variables=["x"]) == "x^2/a + 2*x/a + 1/a"


def test_a_sum_of_a_ratio_and_a_polynomial_is_left_alone():
    """It isn't a ratio, so there is nothing to decompose, and nothing to multiply out."""
    assert exp("x + 1/(x + 1)") == "x + 1/(x + 1)"


def test_a_multivariate_denominator_splits_over_the_primary_variable():
    assert exp("1/(x^2 - y^2)") == "1/(2*y*(x - y)) - 1/(2*y*(x + y))"


# -- Simplify runs first ------------------------------------------------------


def test_expand_is_simplify_and_then_expanding():
    """Per the manual, Expand does what Simplify does and more, so an answer may come from
    Simplify alone."""
    assert exp("SIN(x)^2 + COS(x)^2") == "1"
    assert exp("2/x + 1/x") == "3/x"
    assert exp("(2 + 3)^2") == "25"


def test_an_expanded_number_is_the_number():
    assert exp("12") == "12"
    assert exp("7/12") == "7/12"


def test_rounding_happens_after_expanding_not_before():
    """The radical decomposition must find `SQRT(2)` before rounding; rounding first
    would leave a float with nothing to factor.

    The form is the engine's own. The original writes rational approximations of the
    radicals here (though not under Factor); the engine rounds to decimals as it does
    everywhere in Approximate mode.
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
    """Expand works on any subtree, so the session can expand a highlighted part."""
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
    # Contains a quotient, but isn't one.
    ("-1/x", False),
    ("2*(1/x)", False),
    ("2/x + 1/x", False),
    ("(x + 1)^3", False),
]


@pytest.mark.parametrize(("text", "expected"), RATIOS, ids=str)
def test_only_something_written_as_a_ratio_needs_an_amount(text, expected):
    assert written_as_ratio(parse(text)) is expected


# -- the promises -------------------------------------------------------------

# Every kind of node the parser produces. None may raise; the result doesn't matter.
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

# `EXPAND(u, amount, x, y, ...)` is the command written as a function, which Simplify
# evaluates. Every expected value is the original's answer for the same line.
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
    # Complex is an amount for Factor, not Expand, so here it's read as a variable and
    # the denominator stays unfactored.
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
    """An EXPAND head can be part of a larger expression.

    The original writes `2*(x^2 + 2*x + 1)`; the engine distributes the 2. That is not
    Expand's doing: sympy's automatic evaluation distributes a numeric coefficient over
    a sum during conversion, before any command runs (simplifying `2*(x^2 - 1)` does the
    same), and the pipeline relies on that evaluation. The same trade-off is noted on
    the `INT(x^2, x, a, b)` case in `test_simplify`.

    So the engine can't produce the grouped form at all. Derive keeps whichever form
    was written, so there is no normal form to aim for, and both are the same
    expression.
    """
    assert simplify(parse("2*EXPAND((x + 1)^2)"), Context()).text == (
        "2*x^2 + 4*x + 2"
    )


def test_a_word_that_names_no_amount_is_not_an_amount():
    simplify(parse("EXPAND(w, Turing)"), Context())


def test_only_expand_is_evaluated():
    # A user function whose name starts with EXPAND is its own function once declared.
    state = ParseState()
    state.declare(FunctionDeclaration("EXPANDS", ("u",)))
    assert simplify(parse("EXPANDS((x + 1)^2)", state), Context(), state).text == (
        "EXPANDS((x + 1)^2)"
    )


def test_either_head_may_be_written_inside_the_other():
    assert simplify(parse("EXPAND(FACTOR(x^2 - 1))"), Context()).text == "x^2 - 1"
    assert simplify(parse("FACTOR(EXPAND((x + 1)^2))"), Context()).text == "(x + 1)^2"


# -- what freezing keeps ------------------------------------------------------


def test_holding_a_sum_about_a_variable_keeps_every_term():
    """Freezing must not lose any term.

    A head not replaced whole is rebuilt from its rewritten arguments, which fails for
    heads whose arguments aren't all operands: a `RootSum` holds a polynomial and a
    dummy shared with its summand, and rebuilding it from the parts separately yields
    nothing. A lost term gives a confidently wrong answer, so the sum is checked term
    by term.
    """
    x, t = sp.Symbol("x", real=True), sp.Dummy("t")
    root_sum = sp.RootSum(sp.Poly(t**3 + t + 1, t), sp.Lambda(t, t * sp.log(x - t)))
    frozen, held = held_about(sp.log(x + 1) / 7 + root_sum, x)
    assert frozen.xreplace(held) == sp.log(x + 1) / 7 + root_sum


def test_a_head_that_will_not_survive_freezing_is_held_whole(monkeypatch):
    """A head missing from `_BOUND` costs only tidiness, not correctness.

    `_BOUND` lists heads that must be replaced whole. Freezing is verified by undoing
    it, and if that doesn't reproduce the original, the whole expression is replaced
    instead. The normal form then can't take it apart, which gives worse output but not
    a wrong answer. Removing `RootSum` from `_BOUND` reproduces a real case.
    """
    x, t = sp.Symbol("x", real=True), sp.Dummy("t")
    root_sum = sp.RootSum(sp.Poly(t**3 + t + 1, t), sp.Lambda(t, t * sp.log(x - t)))
    total = sp.log(x + 1) / 7 + root_sum
    monkeypatch.setattr(
        expanding, "_BOUND", tuple(h for h in expanding._BOUND if h is not sp.RootSum)
    )
    frozen, held = held_about(total, x)
    assert frozen.xreplace(held) == total
    # Replaced whole, rather than written about `x` with a term missing.
    assert frozen.is_Symbol


def test_an_approximate_spline_segment_expands_to_its_monomials():
    """SPLINE.DOC's #37 is the Expand of PARA under Approximate: `0.642857 t^3 -
    0.642857 t` for the segment PARA leaves as `t*(9*t^2 - 9)/14`."""
    assert exp("t*(9*t^2 - 9)/14", context=APPROXIMATE) == "0.642857*t^3 - 0.642857*t"
    assert (
        exp("-t*(27*t^2 - 81*t + 54)/28", context=APPROXIMATE)
        == "-0.964285*t^3 + 2.89285*t^2 - 1.92857*t"
    )
