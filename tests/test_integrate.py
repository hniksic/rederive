"""What Simplify makes of an integral.

This is where the integration mathematics is tested, in the order the package works
in: the methods that answer an integral, the gate an answer has to pass before it is
taken, the form the antiderivative taken is printed in, and what is peculiar to an
integral between limits. A failure names the step it is about.
`INT(1/(x^2 + x + a), x)` is one arctangent over `4*a - 1`, whatever its sign, because
the method completes the square rather than summing over the roots.
`INT(SIN(x)/x^2, x)` is `CI(x) - SIN(x)/x` because sympy's first answer is wrong for a
negative `x`, and the gate sends the integral back to be done again. `INT(x^n, x)` has
no case at `n = -1` because the form chooses the constant that makes `LN(x)` its limit
there. And `INT(1/x^2, x, -1, 1)` is `inf`, not the `-2` the antiderivative's
difference gives, because the singularity inside the interval is found before the ends
are put in.

As in `test_simplify.py`, a case is the author's text in and the worksheet's text out,
written inline so that a failure names the mathematics. The cases are the integrals
that show a rule: a case split that is no split, a logarithm that would be complex
where the integrand is real, a pole the integrand owns. `test_integrate_bronstein.py` is the net
underneath them, the integral tables of Bronstein & Semendyayev pinned in full, entry
by entry, so an entry of the tables is repeated here only where it is what shows the
rule.
"""

from __future__ import annotations

import math
import re

import pytest
import sympy as sp
from simplifying import declared, declaring, parse, simp

from rederive.engine.computing import Context, simplify, to_sympy
from rederive.engine.integration.indefinite import by_a_positive_substitution
from rederive.engine.integration.proof import unbroken
from rederive.model.expr import Kind
from rederive.syntax import ParseState, parse_expression


# -- the methods -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        # One arctangent over `4*a - 1`, whatever its sign, and none of the roots.
        ("INT(1/(x^2 + x + a), x)", "2*ATAN((2*x + 1)/SQRT(4*a - 1))/SQRT(4*a - 1)"),
        (
            "INT(1/((x^2 + 1)*(x^2 + x + a)), x)",
            "-(2*a - 3)*ATAN((2*x + 1)/SQRT(4*a - 1))/(SQRT(4*a - 1)*(a^2 - 2*a + 2)) "
            "+ (a - 1)*ATAN(x)/(a^2 - 2*a + 2) + (LN(x^2 + x + a) - LN(x^2 + 1))/(2*a^2 "
            "- 4*a + 4)",
        ),
        # The two logarithms with the same coefficient are one.
        ("INT(x^2/(a^3 + x^3), x)", "LN(x^3 + a^3)/3"),
        (
            "INT(1/(x^2 + x + a), x, 0, 1)",
            "IF(a = 0, inf, IF(a = -2, -inf, IF(-2 < a < 0, ?, (2*ATAN(3/SQRT(4*a - 1)) "
            "- 2*ATAN(1/SQRT(4*a - 1)))/SQRT(4*a - 1))))",
        ),
    ],
)
def test_a_quadratic_with_parametric_coefficients_is_integrated_by_its_square(
    text, expected
):
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "INT(1/((SIN(a*x) + 1)*SIN(a*x)), x)",
            "(LN(TAN(a*x/2)) + COS(a*x)/(SIN(a*x) + 1))/a",
        ),
        # No case for `a = 0`: the pole at `a^2` is only in the constant.
        (
            "INT(x/(SIN(a*x) + 1), x)",
            "-(LN(1/(COS(a*x) + 1)) - 2*LN(TAN(a*x/2) + 1) + a*x/TAN(a*x/2 + pi/4)"
            " + LN(2))/a^2",
        ),
    ],
)
def test_a_rational_function_of_a_parametric_sine_is_integrated(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # Odd in the sine: t = COS(x).
        ("INT(SIN(x)/(2 + COS(x))^2, x)", "1/(COS(x) + 2)"),
        # The half angle, with the logarithms of its forms in the whole angle.
        (
            "INT(1/(3 + 5*COS(x)), x)",
            "LN(3*COS(x) + 4*SIN(x) + 5)/8 - LN(3*COS(x) - 4*SIN(x) + 5)/8",
        ),
        ("INT(TAN(x)/(1 + TAN(x)), x)", "x/2 - LN(COS(x) + SIN(x))/2"),
        # A logarithm negative at zero is written over its negation.
        ("INT(TAN(x)/(TAN(x) - 1), x)", "LN(COS(x) - SIN(x))/2 + x/2"),
        # Over a linear form in both: one logarithm, real where the integrand is finite.
        (
            "INT(3/(2*SIN(x) - 5*COS(x)), x)",
            "-3*SQRT(29)*LN((2*COS(x) + 5*SIN(x) + SQRT(29))/(5*COS(x) - 2*SIN(x)))/29",
        ),
        # Multiples of one angle are written in its functions.
        (
            "INT(1/(SIN(2*x) + COS(x)), x)",
            "-LN(1 - SIN(x))/6 - LN(SIN(x) + 1)/2 + 2*LN(2*SIN(x) + 1)/3",
        ),
    ],
)
def test_a_rational_function_of_a_sine_and_cosine_by_the_tables_substitution(
    text, expected
):
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # Over the root of a linear function, the root itself is the variable.
        ("INT(1/((x + 2)*SQRT(x + 1)), x)", "2*ATAN(SQRT(x + 1))"),
        # Over the root of a quadratic, a multiple of the root and of `INT(1/root)`.
        ("INT(SQRT(x^2 + 1)/x^2, x)", "LN(SQRT(x^2 + 1) + x) - SQRT(x^2 + 1)/x"),
        (
            "INT(x/SQRT(x^2 + x + 1), x)",
            "SQRT(x^2 + x + 1) - LN(2*SQRT(x^2 + x + 1) + 2*x + 1)/2",
        ),
        # An arcsine where the leading coefficient is negative, and the arctangent it
        # equals where the arcsine would need an absolute value.
        ("INT(1/SQRT(1 + x - x^2), x)", "-ASIN(SQRT(5)*(1 - 2*x)/5)"),
        ("INT(1/SQRT(c - a^2*x^2), x)", "ATAN(a*x/SQRT(c - a^2*x^2))/a"),
        (
            "INT(SQRT(1 - x^2)^3/x^4, x)",
            "ASIN(x) + (2*x + 1)*(2*x - 1)*SQRT(1 - x^2)/(3*x^3)",
        ),
        # Over a linear factor, a logarithm, or an arctangent where the quadratic is
        # negative at the factor's root.
        (
            "INT(1/((x - 1)*SQRT(x^2 + 1)), x)",
            "SQRT(2)*LN((SQRT(2)*SQRT(x^2 + 1) - x - 1)/(x - 1))/2",
        ),
        (
            "INT(1/((x + 2)*SQRT(1 - x^2)), x)",
            "SQRT(3)*ATAN(SQRT(3)*(2*x + 1)/(3*SQRT(1 - x^2)))/3",
        ),
        ("INT(1/(x*SQRT(x^2 - 1)), x)", "ATAN(SQRT(x^2 - 1))"),
        # Over a quadratic factor, the root of a quadratic with no linear term is
        # substituted.
        (
            "INT(1/((x^2 + 1)*SQRT(x^2 + 4)), x)",
            "SQRT(3)*ATAN(SQRT(3)*x/SQRT(x^2 + 4))/3",
        ),
        (
            "INT(x/((x^2 + 2)*SQRT(x^2 - 1)), x)",
            "SQRT(3)*ATAN(SQRT(3)*SQRT(x^2 - 1)/3)/3",
        ),
    ],
)
def test_a_root_of_a_linear_or_quadratic_function_is_integrated_by_the_tables(
    text, expected
):
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # The root of a quotient of linear polynomials, and two roots of linear
        # polynomials, are substituted, the root of the quotient staying one.
        (
            "INT(SQRT((x + 1)/(x - 1)), x)",
            "-LN(SQRT((x + 1)/(x - 1)) - 1) + LN(SQRT((x + 1)/(x - 1)) + 1) + "
            "x*SQRT((x + 1)/(x - 1)) - SQRT((x + 1)/(x - 1))",
        ),
        (
            "INT(1/(SQRT(x + 1)*SQRT(x - 1)), x)",
            "LN(SQRT(x - 1)/SQRT(x + 1) + 1) - LN(SQRT(x - 1)/SQRT(x + 1) - 1)",
        ),
        ("SQRT((x + 1)/(x - 1)) + x", "SQRT((x + 1)/(x - 1)) + x"),
    ],
)
def test_roots_of_linear_quotients_are_integrated_by_substitution(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "INT(1/(b*#e^(a*x) + c*#e^(-a*x)), x)",
            "(ATAN(SQRT(b)*#e^(a*x)/SQRT(c)) - ATAN(SQRT(b)/SQRT(c)))/"
            "(a*SQRT(b)*SQRT(c))",
        ),
        ("INT(COTH(a*x)^2, x)", "x - 2/(a*#e^(2*a*x) - a)"),
    ],
)
def test_a_rational_function_of_exponentials_is_integrated_over_one(text, expected):
    """Not as a sum over the roots of a quadratic with parametric coefficients, and the
    same way whatever the order of a set of exponentials in the process."""
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("INT(x^(n - 1)/(1 + x^(2*n)), x)", "ATAN(x^n)/n"),
        ("INT(x^(2*n - 1)*#e^(x^n), x)", "(x^n - 1)*#e^(x^n)/n"),
        (
            "INT(TANH(a*x)^n/COSH(a*x)^2, x)",
            "(((#e^(a*x) - #e^(-a*x))/(#e^(a*x) + #e^(-a*x)))^(n + 1) - 1)/(a*(n + 1))",
        ),
    ],
)
def test_a_symbolic_power_is_integrated_where_its_derivative_divides(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("INT(ASIN(2*x + 1)/x^2, x)", "(2*SQRT(-x^2 - x) - ASIN(2*x + 1))/x"),
        (
            "INT(LN(x + SQRT(x^2 + 1))/x^2, x)",
            "LN((SQRT(x^2 + 1) - 1)/x) - LN(SQRT(x^2 + 1) + x)/x",
        ),
        (
            "INT(ACOSH(x), x)",
            "2*x*LN(SQRT(x - 1) + SQRT(x + 1)) - SQRT(x + 1)*SQRT(x - 1) - LN(2)*x",
        ),
        # An arc of a multiple of the variable over the variable has no elementary
        # antiderivative, and the integral stands at once.
        ("INT(ATAN(x/a)/x, x)", "INT(ATAN(x/a)/x, x)"),
        ("INT(3*ACOS(2*x)/x, x)", "INT(3*ACOS(2*x)/x, x)"),
    ],
)
def test_an_arc_or_logarithm_times_a_rational_function_is_integrated_by_parts(
    text, expected
):
    assert simp(text) == expected


def test_an_even_integrand_over_a_positive_variable_is_kept_where_proved_odd():
    """An antiderivative of an even integrand found for a positive variable holds for
    a negative one where it is odd. `3*x^(5/3)/5` for `(x^2)^(1/3)` is not: it is
    not even real left of zero, where the integrand is. Nor is one kept that is not
    proved to differentiate back for the positive variable, as the hypergeometric
    series sympy writes for `1/SQRT(1 - x^4)` is not."""
    x = sp.Symbol("x", real=True)
    found = by_a_positive_substitution(x**2 / sp.sqrt(x**2 + 1), x)
    assert found is not None
    assert sp.simplify(found + found.subs(x, -x)) == 0
    assert by_a_positive_substitution((x**2) ** sp.Rational(1, 3), x) is None
    assert by_a_positive_substitution(1 / sp.sqrt(1 - x**4), x) is None


def test_an_integral_a_rewrite_leaves_behind_is_evaluated_too():
    # Cancelling this integrand splits it in two, and the part that is an
    # integral of `x` is evaluated as well: an answer still holding
    # `INT(x, x)` would not be a simplified one.
    state = declaring("F", "G")
    assert simp("INT((x*G(x) + F(x))/G(x), x)", state=state) == (
        "INT(F(x)/G(x), x) + x^2/2"
    )


def test_partial_fractions_over_a_sine_leave_only_the_nonelementary_part_standing():
    # The standing integral is a term of its own, not taken under the denominator.
    assert simp("INT(x/((SIN(a*x) + 1)*SIN(a*x)), x)") == (
        "INT(x/SIN(a*x), x) + (LN(1/(COS(a*x) + 1)) - 2*LN(TAN(a*x/2) + 1)"
        " + a*x/TAN(a*x/2 + pi/4) + LN(2))/a^2"
    )


# -- the gate ----------------------------------------------------------------------------


def test_an_antiderivative_that_does_not_differentiate_back_is_not_taken():
    """Sympy's Risch implementation answers some of these wrongly.

    `INT(1/(x^2 + a), x)` comes back zero over a real `a` and
    `INT(x^2/(x^2 + a), x)` comes back `x`, and neither is refused. An answer
    that provably does not differentiate back to the integrand is no answer, so
    the integral is asked again by the rules a calculus course teaches, which do
    not reach that algorithm - and what those give does differentiate back, as
    the two lines below are here to say.

    Only a proven no counts, which is what keeps the answers that cannot be
    checked either way. Sympy's answer for `1/SQRT(1 - x^4)` is a hypergeometric
    series, a function the notation has no name for, so that integral stands as
    Derive leaves it.
    """
    assert simp("DIF(ATAN(x/SQRT(a))/SQRT(a), x)") == "1/(x^2 + a)"
    assert simp("DIF(x - SQRT(a)*ATAN(x/SQRT(a)), x)") == "1 - a/(x^2 + a)"
    assert simp("INT(1/SQRT(1 - x^4), x)") == "INT(1/SQRT(1 - x^4), x)"


@pytest.mark.parametrize(
    "text",
    [
        # A hypergeometric function to sympy.
        "INT(1/SQRT(1 - x^4), x)",
        # Incomplete gamma functions.
        "INT(x^a*#e^(-x), x)",
        # The harmonic number.
        "SUM(1/k, k, 1, n)",
    ],
    ids=str,
)
def test_an_operation_sympy_answers_only_in_a_function_without_a_name_stands(text):
    """What sympy can say only in a function the notation has no name for is not
    worked out, as Derive leaves what it cannot do: the head stands, as written."""
    assert simp(text) == text


@pytest.mark.parametrize(
    "text, expected",
    [
        ("INT(1/(x*(a*x + b)^n), x)", "INT((a*x + b)^(-n)/x, x)"),
        ("INT(1/(x^2*(a*x + b)^(n/2)), x)", "INT((a*x + b)^(-n/2)/x^2, x)"),
        ("INT(1/(a*x^2 + b*x + c)^n, x)", "INT((a*x^2 + b*x + c)^(-n), x)"),
        (
            "INT(1/(x^m*(a*x^2 + b*x + c)^n), x)",
            "INT(x^(-m)*(a*x^2 + b*x + c)^(-n), x)",
        ),
    ],
)
def test_a_symbolic_power_with_no_elementary_antiderivative_stands(text, expected):
    """These are reduction formulas in the tables, with another integral on the right,
    and no elementary antiderivative for a symbolic exponent. Sympy answers the first
    two in a Lerch function the notation has no name for, and searches on the last two
    for over a minute without finding anything. Each stands within a few seconds."""
    assert simp(text) == expected


def test_an_integral_that_will_not_evaluate_survives_as_itself():
    # Derive leaves the INT node standing, and so must we: an error here would
    # lose the entry the user authored.
    assert simp("INT(SIN(x)/LN(x), x)") == "INT(SIN(x)/LN(x), x)"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("INT(SIN(x)/x^2, x)", "CI(x) - SIN(x)/x"),
    ],
)
def test_an_antiderivative_wrong_for_negative_arguments_is_done_again(text, expected):
    """Not in the original, which has neither `CI` nor `COSH_INT`.

    Over a real variable sympy answers this, and `SINH(x)/x^2` the same way, with
    `SIGN(x)` terms that are wrong for negative `x`; over a plain one it answers
    correctly, with `LN(x^2)/2 - LN(x)` besides, which is a constant of integration
    and is not printed.
    """
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # Sympy's substitution answers these only where `-1 < x < 1`, which is
        # where the integrand is not real. The root substituted for gives an
        # antiderivative on both sides.
        ("INT(SQRT(x^2 - 1)/x, x)", "SQRT(x^2 - 1) - ATAN(SQRT(x^2 - 1))"),
        ("INT(1/(x*SQRT(x^2 - 1)), x)", "ATAN(SQRT(x^2 - 1))"),
        ("INT(1/(x*SQRT(x^2 - a^2)), x)", "ATAN(SQRT(x^2 - a^2)/a)/a"),
        # The case is right as it is, but for its condition.
        ("INT(1/(x^2*SQRT(x^2 - 1)), x)", "SQRT(x^2 - 1)/x"),
        ("INT(SQRT(x^2 - 1)/x, x, -2, -1)", "pi/3 - SQRT(3)"),
    ],
)
def test_an_antiderivative_holds_wherever_the_integrand_is_real(text, expected):
    """An antiderivative under a condition with no case for the other values of the
    variable is undefined there, and no answer."""
    assert simp(text) == expected


# -- the form ----------------------------------------------------------------------------


def test_an_antiderivative_is_written_with_no_constant_added():
    """An indefinite integral is one antiderivative out of many.

    Which one is printed is a choice, and the original's choice is the one with
    nothing constant added. The original answers this line - the quadratic
    formula's
    root integrated over its own middle coefficient - with

        b*(SQRT(b^2 - 4*a*c) - b)/(4*a) - c*LN(SQRT(b^2 - 4*a*c) + b)

    where sympy's antiderivative is over `LN(2*b + 2*SQRT(b^2 - 4*a*c))`. The
    two differ by `c*LN(2)`, which is a constant of integration and no part of
    the answer. The answer below is the original's, spelling and all.
    """
    answer = simp("INT((SQRT(b^2 - 4*a*c) - b)/(2*a), b)")
    assert answer == "b*(SQRT(b^2 - 4*a*c) - b)/(4*a) - c*LN(SQRT(b^2 - 4*a*c) + b)"


def test_a_case_split_over_one_antiderivative_is_no_split():
    """Sympy splits where its table has two forms for one function.

    Both of these come back as an `ATAN` where the parameter is positive and a
    pair of logarithms elsewhere, and the two cases are the same function up to
    a constant - so the split says nothing, and the original prints the `ATAN`
    for every parameter. Its answers: `ATAN(x/SQRT(a))/SQRT(a)` and
    `x - SQRT(a)*ATAN(x/SQRT(a))`.
    """
    assert simp("INT(1/(x^2 + a), x)") == "ATAN(x/SQRT(a))/SQRT(a)"
    assert simp("INT(x^2/(x^2 + a), x)") == "x - SQRT(a)*ATAN(x/SQRT(a))"


def test_an_inverse_hyperbolic_does_not_win_a_split_by_being_shorter():
    """Derive prints no inverse hyperbolic in an antiderivative.

    It writes each one as the logarithm it is, `ASINH(x)` authored on its own
    coming back `LN(SQRT(x^2 + 1) + x)`. The cases of this integral are an
    `ASINH` where `a` is nonzero and that same logarithm elsewhere, and the arc
    is the shorter of the two; the original answers with the logarithm, and so
    does this.
    """
    assert simp("INT(SQRT(x^2 + a^2), x)") == (
        "a^2*LN(SQRT(x^2 + a^2) + x)/2 + x*SQRT(x^2 + a^2)/2"
    )


@pytest.mark.parametrize(
    "text, expected",
    [
        ("INT(x^n, x)", "(x^(n + 1) - 1)/(n + 1)"),
        ("INT(x^n, x, 1, 2)", "(2^(n + 1) - 1)/(n + 1)"),
    ],
)
def test_an_antiderivative_is_continuous_in_its_parameters(text, expected):
    """Sympy splits `INT(x^n, x)` into `x^(n + 1)/(n + 1)` and `LN(x)` at `n = -1`.

    The original prints `(x^(n + 1) - 1)/(n + 1)` with no split: the constant of
    integration is chosen so that the formula has `LN(x)` as its limit, and then
    it is an antiderivative for every `n`. The integrand is analytic in `n`, so the
    integral from a fixed point is too, and the pole in sympy's formula was in its
    constant all along. Every answer here is the original's, spelling and all.
    """
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("INT(x^2*COS(a*x^3 + b), x)", "(SIN(a*x^3 + b) - SIN(b))/(3*a)"),
    ],
)
def test_the_constant_is_chosen_for_every_parameter_not_only_the_exponential(
    text, expected
):
    """The original chooses that constant where its tables happen to.

    Its `INT(SINH(a*x), x)` carries the `-1/a`, because it goes through
    `#e^(a*x)`, and its `INT(SIN(a*x), x)` is `-COS(a*x)/a`, undefined at `a = 0`
    for no reason the integral knows of. The same rule is applied to every
    parameter here, so these differ from the original's answers by a constant,
    and hold where the original's do not: `(1 - COS(a*x))/a` is `0` at `a = 0`,
    which is the integral of `SIN(0)`.
    """
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # Continuous at `n = -1`, where the limit is `LN(x)^2/2`: the pole at
        # `(n + 1)^2` is in the constant.
        (
            "INT(x^n*LN(x), x)",
            "(x^(n + 1)*(n + 1)*LN(x) - x^(n + 1) + 1)/(n^2 + 2*n + 1)",
        ),
        # An incomplete gamma function, which the notation has no name for.
        ("INT(x^n*#e^x, x)", "INT(x^n*#e^x, x)"),
    ],
)
def test_a_parametric_antiderivative_is_the_generic_case(text, expected):
    """Sympy splits these on parameters that make the generic formula fail: `a^2 +
    b^2 = 0`, `a = b`, `n = -1`. Such values are a choice of measure zero, and the
    generic case is the antiderivative printed, continuous where a constant makes
    it so."""
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # The integrand itself is singular at `a = 0`.
        ("INT(1/(a*x), x)", "LN(x)/a"),
        # A branch point rather than a pole; nothing is a Laurent series in it.
        ("INT(1/(x^2 + a^2), x)", "ATAN(x/a)/a"),
        ("INT(x/(a^2 + b^2*x^4), x)", "ATAN(b*x^2/a)/(2*a*b)"),
    ],
)
def test_a_pole_the_integrand_owns_is_left_in_the_antiderivative(text, expected):
    """Only a constant of integration may be taken out.

    The principal part of `LN(x)/a` at `a = 0` is `LN(x)/a` itself, which is no
    constant, and `ATAN(x/a)` has no Laurent expansion at all; the answer is what
    it was in both cases.
    """
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("INT(SIN(a*x)^2/x, x)", "LN(2*a*x)/2 - CI(2*a*x)/2"),
        ("INT((COS(a*x) - COS(b*x))/x, x)", "-LN(a*x) + LN(b*x) + CI(a*x) - CI(b*x)"),
        ("INT((#e^(a*x) - 1)/x, x)", "EI(a*x) - LN(a^2*x^2)/2"),
    ],
)
def test_an_antiderivative_is_continuous_where_the_integrand_is_regular(text, expected):
    """The logarithm beside a `CI` takes its imaginary part on the other side of zero
    from it where `a` is negative, and `LN(x)/2 - CI(2*a*x)/2` jumps there; over the
    argument of the `CI` the two take it on the same side."""
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, values",
    [
        ("INT(1/(2 + SIN(x)), x)", {}),
        ("INT(1/(b + c*COS(x)), x)", {"b": 2, "c": 1}),
        ("INT(1/(b + c*COS(x)), x)", {"b": -2, "c": 1}),
        ("INT(1/(b + c*SIN(x))^2, x)", {"b": -3, "c": 2}),
        ("INT(1/(b^2 - c^2*COS(x)^2), x)", {"b": 2, "c": 1}),
        ("INT(1/(SIN(x)^4 + COS(x)^4), x)", {}),
    ],
)
def test_a_trigonometric_antiderivative_is_continuous_where_the_tangent_jumps(
    text, values
):
    # TAN(x/2) jumps at pi and TAN(x) at pi/2, where these integrands are regular.
    answer = to_sympy(parse_expression(simp(text), ParseState()).node)
    symbols = {symbol.name: symbol for symbol in answer.free_symbols}
    answer = answer.subs({symbols[name]: value for name, value in values.items()})
    x = symbols["x"]
    for point in (sp.pi / 2, sp.pi, 3 * sp.pi):
        below = complex(answer.subs(x, point - sp.Rational(1, 10**6)).evalf(30))
        above = complex(answer.subs(x, point + sp.Rational(1, 10**6)).evalf(30))
        assert abs(above - below) < 1e-4


def test_an_antiderivative_is_written_in_radicals_where_the_roots_are_known():
    """The sum over the roots is written out wherever the roots can be.

    Each conjugate pair of them is a logarithm and an arctangent, and a real one
    a logarithm, which is how Derive writes all three of these. Sympy 1.14 loses
    its way on the first: the two pairs whose coordinates carry a cube root fall
    out of its real-form conversion without a word, and `1/(x^6 + 16)` comes back
    as the one arctangent in `2^(1/3)*x/2` alone - a third of the answer.
    `engine.rational` is what makes the answer whole, and what finds the roots of
    the quintic, which sympy's own conversion could not and so left as a
    `ROOT_SUM`. The seventh roots of unity are cosines and sines of `pi/7`,
    and those are what the answer to the last is written in.
    """
    assert simp("INT(1/(x^6 + 16), x)") == (
        "2^(2/3)*ATAN(2^(1/3)*x/2)/48 + 2^(2/3)*ATAN(2^(1/3)*x - SQRT(3))/96 "
        "+ 2^(2/3)*ATAN(2^(1/3)*x + SQRT(3))/96 "
        "- 2^(2/3)*SQRT(3)*LN(x^2 - 2^(2/3)*SQRT(3)*x + 2*2^(1/3))/192 "
        "+ 2^(2/3)*SQRT(3)*LN(x^2 + 2^(2/3)*SQRT(3)*x + 2*2^(1/3))/192"
    )
    assert simp("INT(1/(x^5 + 16), x)") == (
        "-2^(3/10)*SQRT(5 - SQRT(5))*ATAN(-2^(7/10)*x/SQRT(5 - SQRT(5)) "
        "+ SQRT(10)/(2*SQRT(5 - SQRT(5))) + SQRT(2)/(2*SQRT(5 - SQRT(5))))/80 "
        "+ 2^(3/10)*SQRT(SQRT(5) + 5)*ATAN(2^(7/10)*x/SQRT(SQRT(5) + 5) "
        "+ SQRT(10)/(2*SQRT(SQRT(5) + 5)) - SQRT(2)/(2*SQRT(SQRT(5) + 5)))/80 "
        "+ (2^(4/5)*SQRT(5) - 2^(4/5))*LN(x^2 + (2^(4/5)*SQRT(5) - 2^(4/5))*x/2 "
        "+ 2*2^(3/5))/320 + 2^(4/5)*LN(x + 2^(4/5))/80 "
        "- (2^(4/5)*SQRT(5) + 2^(4/5))*LN(x^2 - (2^(4/5)*SQRT(5) + 2^(4/5))*x/2 "
        "+ 2*2^(3/5))/320"
    )
    assert simp("INT(1/(x^7 + 1), x)") == (
        "2*COS(pi/14)*ATAN(x/COS(pi/14) - TAN(pi/14))/7 "
        "+ 4*SIN(pi/7)*COS(pi/7)*ATAN(x/COS(3*pi/14) + TAN(3*pi/14))/7 "
        "+ 2*SIN(pi/7)*ATAN(x/SIN(pi/7) - 1/TAN(pi/7))/7 "
        "- COS(pi/7)*LN(x^2 - 2*COS(pi/7)*x + 1)/7 "
        "+ SIN(3*pi/14)*LN(x^2 + 2*SIN(3*pi/14)*x + 1)/7 "
        "- SIN(pi/14)*LN(x^2 - 2*SIN(pi/14)*x + 1)/7 + LN(x + 1)/7"
    )


def test_an_antiderivative_pairs_roots_whose_radicals_are_written_apart():
    """The roots of `16777216*t^8 + 1` come back each in a radical form of its own,
    and no root's conjugate is written as the conjugate of its expression: the pairs
    are found by value, and each is written over the shorter root of the two. Left a
    `ROOT_SUM`, the integral between limits had no antiderivative to be taken at the
    ends, and fell to sympy, which loses the roots and answers 0 between 0 and 1.
    """
    assert simp("INT(1/(x^8 + 1), x)") == (
        "-SQRT(2 - SQRT(2))*ATAN(-2*x/SQRT(2 - SQRT(2)) + SQRT(2) + 1)/8 "
        "+ SQRT(2 - SQRT(2))*ATAN(2*x/SQRT(2 - SQRT(2)) + SQRT(2) + 1)/8 "
        "+ SQRT(SQRT(2) + 2)*ATAN(2*x/SQRT(SQRT(2) + 2) + SQRT(2) - 1)/8 "
        "+ SQRT(SQRT(2) + 2)*ATAN(2*x/SQRT(SQRT(2) + 2) - SQRT(2) + 1)/8 "
        "- SQRT(2 - SQRT(2))*LN(x^2 - SQRT(2 - SQRT(2))*x + 1)/16 "
        "+ SQRT(2 - SQRT(2))*LN(x^2 + SQRT(2 - SQRT(2))*x + 1)/16 "
        "- SQRT(SQRT(2) + 2)*LN(x^2 - SQRT(SQRT(2) + 2)*x + 1)/16 "
        "+ SQRT(SQRT(2) + 2)*LN(x^2 + SQRT(SQRT(2) + 2)*x + 1)/16"
    )
    between = to_sympy(parse_expression(simp("INT(1/(x^8 + 1), x, 0, 1)"), ParseState()).node)
    assert abs(float(between.evalf(30)) - 0.924651705775538) < 1e-12
    # Over the tenth roots of unity, of which sympy kept only the pair on the
    # imaginary axis, and answered `pi/10`.
    assert simp("INT(1/(x^10 + 1), x, 0, inf)") == "SQRT(5)*pi/10 + pi/10"


@pytest.mark.parametrize(
    "n, expected",
    [
        (7, "4*pi*(2*COS(pi/14) + 3*SIN(pi/7) + COS(3*pi/14))/49"),
        (9, "2*pi*(3*SQRT(3) + 4*COS(pi/18) + 8*SIN(pi/9) + 2*SIN(2*pi/9))/81"),
    ],
)
def test_an_integral_of_a_rational_function_converging_at_infinity_is_finite(
    n, expected
):
    """The logarithms in the antiderivative of `1/(x^n + 1)` grow as `LN(x)` with
    coefficients that add up to nothing only by identities of the cosines of multiples
    of `pi/n`, and the antiderivative's limit at `inf`, taken as it stands, is `inf`.
    The integral is `pi/(n*SIN(pi/n))`."""
    answer = simp(f"INT(1/(x^{n} + 1), x, 0, inf)")
    assert answer == expected
    value = to_sympy(parse_expression(answer, ParseState()).node)
    assert abs(float(value.evalf(30)) - math.pi / (n * math.sin(math.pi / n))) < 1e-12


@pytest.mark.parametrize(
    "text, expected",
    [
        ("INT(1/x, x, 1, x)", "LN(x)"),
        ("INT(1/(y^2 + 1), y, 0, y/x)", "ATAN(y/x)"),
        ("INT(1/t, t, 1, x)", "LN(x)"),
    ],
)
def test_an_integral_of_a_rational_function_up_to_its_own_variable_is_valued(
    text, expected
):
    """An end written in the variable of integration is the antiderivative taken
    there: the variable integrated over is another than the one the end is in."""
    assert simp(text) == expected


def _summed_over_numeric_roots(answer: str) -> complex:
    """The answer's value, each `ROOT_SUM` in it summed over its polynomial's roots
    found numerically. Sympy's own `RootSum.doit` writes the roots of a quartic by
    the formula and takes the logarithms of them on the wrong branch."""
    value = to_sympy(parse_expression(answer, ParseState()).node)
    value = value.replace(
        lambda e: isinstance(e, sp.RootSum),
        lambda e: sp.Add(*(e.fun(root) for root in e.poly.nroots(n=30))),
    )
    return complex(value.evalf(30))


@pytest.mark.parametrize(
    "text, expected, value",
    [
        (
            "INT(1/(x^5 + x + 1), x, 0, 1)",
            "-ROOT_SUM(161*t^3 - 23*t^2 - 12*t - 1, t, t*LN(-216384*t^4 + 54096*t^3 "
            "- 3284*t^2 + 5941*t + 256) - 4*LN(5)*t) + 5*SQRT(3)*pi/126 "
            "+ ROOT_SUM(161*t^3 - 23*t^2 - 12*t - 1, t, t*LN(-216384*t^4 + 54096*t^3 "
            "- 3284*t^2 + 5941*t + 881) - 4*LN(5)*t) - LN(3)/14",
            0.653892007815572,
        ),
        (
            "INT(1/(x^4 + x + 1), x, 0, inf)",
            "-ROOT_SUM(229*t^4 + 18*t^2 + 8*t + 1, t, "
            "t*LN(2061*t^3/64 - 687*t^2/64 + 391*t/64 + 27/64))",
            0.835877591843161,
        ),
        # At `-inf` each logarithm of the sum tends to one of `LN(-1)`, on one side of
        # the cut or the other, and the integral stands.
        ("INT(1/(x^4 + x + 1), x, -inf, inf)", "INT(1/(x^4 + x + 1), x, -inf, inf)", None),
    ],
)
def test_an_integral_of_a_rational_function_over_a_root_sum_is_taken_at_its_ends(
    text, expected, value
):
    """The antiderivative over roots that are not all had is a `ROOT_SUM`, of which
    sympy takes no limit. At a finite end it is continuous and is taken at its value,
    and at `inf` each of its logarithms, of the variable less a root, tends to
    `LN(1)`. Sympy is not asked instead: it searched for minutes for the real roots of
    the resolvent of `1/(x^5 + x + 1)`."""
    answer = simp(text)
    assert answer == expected
    if value is not None:
        assert abs(_summed_over_numeric_roots(answer) - value) < 1e-12


def test_an_antiderivative_over_a_denominator_that_does_not_factor_keeps_its_sum():
    """The logarithmic part of these is a sum over the roots of a polynomial.

    `ROOT_SUM(p, t, u)` is what the notation calls that, and the answer is the
    rational part plus one. What it must never be is the rational part alone:
    `1/(x*(x^4 + x + 1))` does not integrate to `LN(x)`, and an answer that
    quietly loses most of itself is worse than one that cannot be read.

    The polynomial summed over is the resolvent sympy builds and not the
    denominator it came from, which is why the coefficients are not the
    denominator's: the roots of one are the roots of the other transformed,
    and the summand is transformed to match.
    """
    assert simp("INT(1/(x*(x^4 + x + 1)), x)") == (
        "LN(x) + ROOT_SUM(229*t^4 + 229*t^3 + 96*t^2 + 16*t + 1, t, "
        "t*LN(x + 129614*t^4/1713 - 1145*t^3/1713 - 76423*t^2/1713 "
        "- 45760*t/1713 - 6286/1713))"
    )


def test_an_antiderivative_over_cardano_roots_keeps_its_sum_and_comes_back_at_once():
    """A cubic with one real root has its roots written by Cardano's formula, in cube
    roots of irrational numbers, and an antiderivative written over them took minutes
    of sympy's arithmetic in such numbers, for an answer pages long. The sum over the
    cubic's roots stands, and where the polynomial summed over factors, as it does for
    `1/(x^5 + x + 1)`, the quadratic factor's pair is written out all the same."""
    assert simp("INT(1/(x^3 - x^2 + 1), x)") == (
        "ROOT_SUM(23*t^3 + t - 1, t, t*LN(x + 46*t^2/25 + 69*t/25 - 7/25))"
    )
    assert simp("INT(1/(x^5 + x + 1), x)") == (
        "5*SQRT(3)*ATAN(2*SQRT(3)*x/3 + SQRT(3)/3)/21 - LN(x^2 + x + 1)/14 "
        "+ ROOT_SUM(161*t^3 - 23*t^2 - 12*t - 1, t, t*LN(x - 216384*t^4/625 "
        "+ 54096*t^3/625 - 3284*t^2/625 + 5941*t/625 + 256/625))"
    )


def test_a_sum_over_the_roots_of_a_polynomial_is_an_expression_and_not_a_string():
    """The head is live: it differentiates, and it settles.

    A quartic that does not factor has an antiderivative that is nothing but
    such a sum, so this is the head on its own rather than beside a rational
    part. Differentiating it back to the integrand is what says it is an
    expression and not a decorative one - a `Kind.STRING` would print the same
    and do none of it.
    """
    answer = simplify(parse("INT(1/(x^4 + x + 1), x)"), Context())
    assert answer.node.kind is not Kind.STRING
    assert answer.text == (
        "ROOT_SUM(229*t^4 + 18*t^2 + 8*t + 1, t, "
        "t*LN(x + 2061*t^3/64 - 687*t^2/64 + 391*t/64 + 27/64))"
    )
    assert simp(f"DIF({answer.text}, x)") == "1/(x^4 + x + 1)"
    # And it is a fixed point: read back, it is the answer it was written from.
    assert simp(answer.text) == answer.text


@pytest.mark.parametrize(
    "text, expected",
    [
        # `LN(SIN(x) - 1)` is complex for every real `x`; `LN(1 - SIN(x))` is the
        # same up to a constant, and real wherever the integral is.
        ("INT(SEC(x), x)", "LN(SIN(x) + 1)/2 - LN(1 - SIN(x))/2"),
        ("INT(CSC(x), x)", "LN(1 - COS(x))/2 - LN(COS(x) + 1)/2"),
        (
            "INT(SEC(x)^3, x)",
            "-LN(1 - SIN(x))/4 + LN(SIN(x) + 1)/4 + SIN(x)/(2*COS(x)^2)",
        ),
        ("INT(SEC(2*x), x)", "LN(SIN(2*x) + 1)/4 - LN(1 - SIN(2*x))/4"),
    ],
)
def test_an_antiderivative_takes_no_logarithm_of_a_negative_quantity(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("INT(EXP(x^2), x)", "SQRT(pi)*ERFI(x)/2"),
        ("DIF(erfi(x), x)", "2*#e^(x^2)/SQRT(pi)"),
        # `GAMMA(x)` is written as the factorial it is, `(x - 1)!`.
        ("DIF(GAMMA(x), x)", "(x - 1)!*DIGAMMA(x)"),
        ("DIF(LAMBERTW(x), x)", "LAMBERTW(x)/(x*(LAMBERTW(x) + 1))"),
        ("INT(SIN(x)/x, x)", "SI(x)"),
        # The product is written as the sum it is, the sine of the double angle.
        ("INT(SIN(a*x)*COS(a*x)/x, x)", "SI(2*a*x)/2"),
        ("INT(EXP(x)/x, x)", "EI(x)"),
        # Sympy's answer is the polylogarithm a full turn round the origin away, and
        # the principal one is an antiderivative too.
        ("INT(LN(1 - x)/x, x)", "-POLYLOG(2, x)"),
        ("DIF(-POLYLOG(2, x), x)", "LN(1 - x)/x"),
    ],
    ids=str,
)
def test_a_special_function_an_elementary_operation_produces_reads_back(text, expected):
    """The special functions rederive names beyond Derive's own are the ones its own
    operations reach from elementary input, and each answer is a fixed point."""
    assert simp(text) == expected
    assert simp(expected) == expected


def test_an_integral_answered_by_a_special_function_reads_back_as_that_function():
    """An answer is only as good as the name it is written under.

    A name something else answers to reads back as a different function from the
    one computed, with nothing to say so - which is what simplifying the answer
    again is here to catch. `CHI` is Derive's chi-square distribution long before
    it is anybody's cosh-integral, and the logarithmic integral sympy answers
    `1/LN(x)` with has no name of its own: it is the exponential integral of the
    logarithm, and is written as that.
    """
    for text, answer in (
        ("INT(1/LN(x), x)", "EI(LN(x))"),
        ("INT(COSH(x)/x, x)", "COSH_INT(x)"),
        ("INT(SINH(x)/x, x)", "SINH_INT(x)"),
    ):
        assert simp(text) == answer
        assert simp(answer) == answer
    assert simp("DIF(EI(LN(x)), x)") == "1/LN(x)"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("INT(COSH(x^2), x)", "SQRT(pi)*(ERF(x) + ERFI(x))/4"),
        ("INT(SINH(x^2), x)", "-SQRT(pi)*(ERF(x) - ERFI(x))/4"),
        (
            "INT(SIN(x^2), x)",
            "SQRT(2)*SQRT(pi)*(1 + #i)*(ERF(SQRT(2)*(1 + #i)*x/2)"
            " - #i*ERF(SQRT(2)*(1 - #i)*x/2))/8",
        ),
        (
            "INT(SIN(pi*t^2/2), t, 0, z)",
            "(1 + #i)*(ERF(SQRT(pi)*(1 + #i)*z/2) - #i*ERF(SQRT(pi)*(1 - #i)*z/2))/4",
        ),
    ],
    ids=str,
)
def test_a_sympy_head_without_a_name_is_written_in_functions_that_have_one(
    text, expected
):
    """Sympy's Fresnel integrals are error functions of complex arguments, so an
    answer in them is written in those."""
    assert simp(text) == expected


def test_an_answer_written_over_the_error_functions_differentiates_back():
    """The Fresnel integral written over `ERF` is the same antiderivative.

    Simplify does not take the derivative's complex exponentials back to a sine, so
    the derivative is compared with the integrand at a point.
    """
    derivative = to_sympy(parse(simp(f"DIF({simp('INT(SIN(x^2), x)')}, x)")))
    x = next(iter(derivative.free_symbols))
    assert abs(complex(derivative.subs(x, 0.7).evalf()) - math.sin(0.49)) < 1e-12


@pytest.mark.parametrize(
    "text, expected",
    [
        # `x*LN(a)` is a term of the antiderivative, not a constant.
        ("INT(LN(a*x), x)", "x*LN(a*x) - x"),
        ("INT(LN(a*x^2), x)", "x*LN(a*x^2) - 2*x"),
        ("INT(LN(x/a), x)", "x*LN(x/a) - x"),
        ("INT(x*LN(a*x), x)", "x^2*LN(a*x)/2 - x^2/4"),
        ("INT(LN(2*x), x)", "x*LN(x) + (LN(2) - 1)*x"),
        # A logarithm that is a term of its own still sheds its constant factor.
        ("INT(1/(2*x + 2), x)", "LN(x + 1)/2"),
    ],
)
def test_a_constant_inside_a_logarithm_is_shed_only_by_a_term(text, expected):
    assert simp(text) == expected


# -- definite integrals ------------------------------------------------------------------

SINGULAR_INTEGRALS = [
    # An integrand of one sign blowing up at least as fast as the reciprocal of the
    # distance makes the integral diverge to that sign's infinity, wherever the
    # singularity is. The antiderivative taken across it gives a finite number.
    ("INT(1/x^2, x, -1, 1)", "inf"),
    ("INT(1/x^4, x, -1, 1)", "inf"),
    ("INT(1/ABS(x), x, -1, 1)", "inf"),
    ("INT(1/x^2, x, -1, 2)", "inf"),
    ("INT(1/(x - 1)^2, x, 0, 2)", "inf"),
    ("INT(SEC(x)^2, x, 0, pi)", "inf"),
    ("INT(#e^(1/x), x, -1, 1)", "inf"),
    ("INT(1/x^2, x, -inf, inf)", "inf"),
    ("INT(1/x^2, x, 1, -1)", "-inf"),
    # At an end, too, and without the logarithm's imaginary part.
    ("INT(1/x, x, -1, 0)", "-inf"),
    ("INT(LN(x)/x, x, 0, 1)", "-inf"),
    # Diverging to both infinities, the integral has no value: the sum of its
    # parts is `inf - inf`. Not the principal value.
    ("INT(1/x, x, -1, 1)", "?"),
    ("INT(1/x, x, -1, 2)", "?"),
    ("INT(1/x^3, x, -1, 2)", "?"),
    ("INT(TAN(x), x, 0, pi)", "?"),
    ("INT(1/SIN(x), x, -1, 1)", "?"),
    ("INT(1/(x^2 - 1), x, 0, 2)", "?"),
    ("INT(x/(x^2 - 1)^2, x, -2, 2)", "?"),
    # Integrable singularities keep their values.
    ("INT(1/SQRT(ABS(x)), x, -1, 1)", "4"),
    ("INT(LN(ABS(x)), x, -1, 1)", "-2"),
    ("INT(1/SQRT(ABS(x - 1)), x, 0, 2)", "4"),
    ("INT(1/(x - 1)^(2/3), x, 0, 2)", "3/2 - 3*SQRT(3)*#i/2"),
    ("INT(LN(x), x, 0, 1)", "-1"),
    ("INT(SIN(x)/x, x, -1, 1)", "2*SI(1)"),
    # Integrated on each side of the singularity, where the antiderivative jumps.
    ("INT(#e^(-1/x^2)/x^2, x, -1, 1)", "SQRT(pi) - SQRT(pi)*ERF(1)"),
    # A divergence of unknown sign is worked out on each side of the parameter's.
    ("INT(a/x^2, x, -1, 1)", "IF(a > 0, inf, IF(a < 0, -inf, 0))"),
    ("INT(a/x, x, -1, 1)", "IF(a /= 0, ?, 0)"),
    # Undecided at zero, where the antiderivative's limit is not the integral's.
    ("INT(1/(x*LN(x)), x, 0, 1/2)", "INT(1/(x*LN(x)), x, 0, 1/2)"),
]


@pytest.mark.parametrize(("text", "expected"), SINGULAR_INTEGRALS, ids=str)
def test_an_integral_is_decided_at_its_singularities(text, expected):
    assert simp(text) == expected


PARAMETRIC_INTEGRALS = [
    # The sign of an infinity in the antiderivative depends on the parameter.
    ("INT(1/(x^2 + a^2), x, 0, inf)", "pi/(2*ABS(a))"),
    ("INT(a/(x^2 + a^2), x, 0, inf)", "pi*SIGN(a)/2"),
    ("INT(1/(x^2 + a), x, 0, inf)", "IF(a > 0, pi/(2*SQRT(a)), IF(a < 0, ?, inf))"),
    # A pole that moves with the parameter is in the interval for some values of
    # it, and there the integral diverges; the antiderivative's difference holds
    # only for the others.
    ("INT(1/(x - a)^2, x, 0, 2)", "IF(0 <= a <= 2, inf, -2/(a*(2 - a)))"),
    ("INT(1/(x + a)^2, x, 0, inf)", "IF(a <= 0, inf, 1/a)"),
    ("INT(1/(x^2 - a^2), x, 0, inf)", "IF(a = 0, inf, ?)"),
    # Poles real only where `SIN(a)` is zero, where the answer is not defined either.
    ("INT(1/(x^2 + 2*x*COS(a) + 1), x, -inf, inf)", "pi/ABS(SIN(a))"),
    (
        "INT(1/(x - a), x, 0, 1)",
        "IF(a = 0, inf, IF(a = 1, -inf, IF(0 < a < 1, ?, LN(1 - a) - LN(-a))))",
    ),
    # Where the pole cancels, the integral is a number.
    (
        "INT(x/(x - a), x, 0, 1)",
        "IF(a = 0, 1, IF(a = 1, -inf, IF(0 < a < 1, ?, -a*LN(-a) + a*LN(1 - a) + 1)))",
    ),
    # Diverging at the singular end to the same infinity whichever side of it the
    # other end is on, or to opposite ones, where the integral stands.
    ("INT(COS(a*x)/x, x, 0, alpha)", "inf"),
    ("INT(1/x^3, x, alpha, 0)", "-inf"),
    ("INT(1/x^2, x, 0, alpha)", "INT(1/x^2, x, 0, alpha)"),
    # A power of zero, which is 0 or infinite by its exponent's sign.
    ("INT(x^n, x, 0, 1)", "IF(n > -1, 1/(n + 1), inf)"),
    ("INT(1/x^n, x, 0, 1)", "IF(n < 1, 1/(1 - n), inf)"),
    # Convergent for one sign of the parameter, divergent to one infinity
    # otherwise.
    ("INT(#e^(-a*x), x, 0, inf)", "IF(a > 0, 1/a, inf)"),
    ("INT(#e^(a*x), x, 0, inf)", "IF(a < 0, -1/a, inf)"),
    ("INT(#e^(-a*x^2), x, -inf, inf)", "IF(a > 0, SQRT(pi)/SQRT(a), inf)"),
    ("INT(x^n*LN(x), x, 0, 1)", "IF(n > -1, -1/(n + 1)^2, -inf)"),
    ("INT(#e^(-x)*x^n, x, 0, inf)", "IF(n > -1, n!, inf)"),
    # One expression for every sign.
    ("INT(SIN(a*x)/x, x, 0, inf)", "pi*SIGN(a)/2"),
    ("INT(COS(a*x)*#e^(-x^2), x, -inf, inf)", "SQRT(pi)*#e^(-a^2/4)"),
    # Divergent by oscillation for a nonpositive `a`, to no sign, which stays
    # standing; the convergence condition is the sign of `a`.
    (
        "INT(#e^(-a*x)*SIN(x), x, 0, inf)",
        "IF(a > 0, 1/(a^2 + 1), INT(SIN(x)*#e^(-a*x), x, 0, inf))",
    ),
]


@pytest.mark.parametrize(("text", "expected"), PARAMETRIC_INTEGRALS, ids=str)
def test_a_parametric_integral_is_worked_out_on_each_side_of_a_sign(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("INT(LN(1 + x)/(x^2 + 1), x, 0, 1)", "INT(LN(x + 1)/(x^2 + 1), x, 0, 1)"),
        ("INT(LN(1 + x)/(x^2 + 1), x, 1, 2)", "INT(LN(x + 1)/(x^2 + 1), x, 1, 2)"),
        ("INT(LN(x)*LN(1 - x), x, 0, 1)", "INT(LN(x)*LN(1 - x), x, 0, 1)"),
        # The pieces of a sum need not converge where the sum does.
        ("INT((x^t - x^s)/LN(x), x, 0, 1)", "-INT((x^s - x^t)/LN(x), x, 0, 1)"),
        # What the methods do find is still found.
        ("INT(LN(1 + x)/x, x, 0, 1)", "pi^2/12"),
        ("INT(LN(x)/(x^2 + 1), x, 0, inf)", "0"),
    ],
)
def test_a_definite_integral_with_no_antiderivative_in_reach_stands(text, expected):
    """Sympy values these by an antiderivative over a positive variable, and over one
    its manual rules search for minutes for an antiderivative there is none of. Each
    stands within a few seconds, whole."""
    assert simp(text) == expected


_T, _A = sp.symbols("t a", real=True)


@pytest.mark.parametrize(
    ("expression", "low", "high", "proved"),
    [
        (sp.log(_T**2 + _A**2 + 1), 0, 1, True),
        (sp.log(_T - _A), 0, 1, False),
        (1 / (_T - _A), 0, 1, False),
        (sp.atan(_T / _A), -1, 1, True),
        (sp.atan(1 / _T), -1, 1, False),
        (sp.atan(1 / _T), 0, 1, True),
        (sp.sqrt(_T**2 - 1), 0, 2, True),
        (1 / sp.sqrt(_T**2 - 1), 0, 2, False),
        (sp.log(_T + sp.sqrt(_T**2 - 1)), 1, 2, True),
        (sp.log(_A * sp.cos(_T) + sp.sqrt(_A**2 * sp.cos(_T) ** 2 + 1)), 0, sp.pi, True),
        (sp.log(sp.tan(_T)) + sp.log(sp.cot(_T)), 0, sp.pi / 2, True),
        (sp.log(sp.tan(_T)), 0, sp.pi, False),
        (sp.log(sp.cos(_T)), 0, 1, True),
        (sp.log(sp.cos(_T)), 0, 2, False),
        (sp.log(_T), -sp.oo, 0, True),
        (sp.log(_T), -1, 1, False),
        (sp.asin(_T / 2), 0, 1, True),
        (sp.asin(_A * sp.sin(_T)), 0, sp.pi / 2, False),
        (sp.log(_T) * sp.log(_T + 1) + sp.polylog(2, -_T), 0, 1, True),
        (sp.I * sp.pi * sp.log(_T + 1) - sp.polylog(2, _T + 1), 0, 1, False),
        (sp.Ei((_A + 1) * sp.log(_T)) - sp.Ei(sp.log(_T)), 0, 1, False),
        (sp.exp(_T) * sp.sin(_A * _T), -sp.oo, sp.oo, True),
    ],
    ids=str,
)
def test_an_antiderivative_is_proved_continuous_on_an_interval(
    expression, low, high, proved
):
    """A logarithm whose argument may be zero, a pole, an arctangent of something that
    blows up and a polylogarithm on its cut are not proved continuous; an argument
    whose sign is settled for every value of the parameter is."""
    assert unbroken(expression, _T, sp.S(low), sp.S(high)) is proved


STEP_INTEGRALS = [
    # Split where the step jumps, and each stretch integrated with the step
    # constant.
    ("INT(FLOOR(x), x, 0, 3)", "3"),
    ("INT(FLOOR(x)^2, x, 0, 3)", "5"),
    ("INT(x*FLOOR(x), x, 0, 2)", "3/2"),
    ("INT(FLOOR(x), x, 3, 0)", "-3"),
    ("INT(FLOOR(x), x, -1/2, 2)", "1/2"),
    ("INT(FLOOR(x/2), x, 0, 5)", "4"),
    ("INT(FLOOR(x^2), x, 0, 2)", "-SQRT(2) - SQRT(3) + 5"),
    ("INT(MOD(x, 1), x, 0, 3)", "3/2"),
    ("INT(MOD(x, 2)*x, x, 0, 3)", "4"),
    ("INT(FLOOR(x)*SIGN(x - 3/2), x, 0, 3)", "2"),
    ("INT(FLOOR(x)/x, x, 1, 3)", "LN(9/2)"),
    ("INT(a*FLOOR(x), x, 0, 3)", "3*a"),
    # A sign of something other than a polynomial, whose antiderivative holds
    # only near the origin.
    ("INT(ABS(SIN(x)), x, 0, 10*pi)", "20"),
    ("INT(ABS(SIN(x)), x, -10, 10)", "2*COS(10) + 14"),
    ("INT(SIGN(SIN(x)), x, 0, 3*pi)", "pi"),
    ("INT(SIGN(COS(x)), x, 0, 5)", "5 - 2*pi"),
    ("INT(ABS(LN(x)), x, 1/2, 2)", "3*LN(2)/2 - 1/2"),
    # Jumps at no number to split at.
    ("INT(FLOOR(x), x, 0, a)", "INT(FLOOR(x), x, 0, a)"),
]


@pytest.mark.parametrize(("text", "expected"), STEP_INTEGRALS, ids=str)
def test_a_step_function_is_integrated_between_its_jumps(text, expected):
    assert simp(text) == expected


STEP_ANTIDERIVATIVES = [
    # A sign of a periodic function jumps forever, and the antiderivative counts the
    # jumps with a floor: each arch of `ABS(SIN(x))` adds 2.
    ("INT(ABS(SIN(x)), x)", "2*FLOOR(x/pi) - SIGN(SIN(x))*COS(x)"),
    ("INT(ABS(COS(x)), x)", "SIGN(COS(x))*SIN(x) + 2*FLOOR(x/pi + 1/2)"),
    ("INT(SIGN(SIN(x)), x)", "(-pi*FLOOR(x/pi) + x - pi/2)*SIGN(SIN(x))"),
    ("INT(ABS(SIN(2*x)), x)", "FLOOR(2*x/pi) - SIGN(SIN(2*x))*COS(2*x)/2"),
    (
        "INT(STEP(COS(x)), x)",
        "-pi*SIGN(COS(x))*FLOOR(x/pi + 1/2)/2 + x*SIGN(COS(x))/2 + x/2",
    ),
    (
        "INT(ABS(SIN(x))^3, x)",
        "SIGN(SIN(x))*COS(x)^3/3 - SIGN(SIN(x))*COS(x) + 4*FLOOR(x/pi)/3",
    ),
    # What each arch adds grows with it, by a polynomial in the arch's number.
    (
        "INT(x*ABS(SIN(x)), x)",
        "pi*FLOOR(x/pi)^2 + ABS(SIN(x)) - x*SIGN(SIN(x))*COS(x) + pi*FLOOR(x/pi)",
    ),
    # A floor is a constant between integers, and the constants add up.
    ("INT(FLOOR(x), x)", "-FLOOR(x)^2/2 + x*FLOOR(x) - FLOOR(x)/2"),
    ("INT(MOD(x, 1), x)", "FLOOR(x)^2/2 - x*FLOOR(x) + FLOOR(x)/2 + x^2/2"),
    ("INT(FLOOR(-x), x)", "FLOOR(-x)^2/2 + x*FLOOR(-x) + FLOOR(-x)/2"),
    (
        "INT(x*FLOOR(x), x)",
        "-FLOOR(x)^3/6 - FLOOR(x)^2/4 + x^2*FLOOR(x)/2 - FLOOR(x)/12",
    ),
    # Jumps not equally spaced, jumps not periodic, and an arch that grows
    # geometrically have no such answer here; a split near the origin would be
    # wrong away from it.
    ("INT(ABS(SIN(x) - 1/2), x)", "INT(ABS(SIN(x) - 1/2), x)"),
    ("INT(ABS(SIN(x^2)), x)", "INT(ABS(SIN(x^2)), x)"),
    ("INT(#e^x*ABS(SIN(x)), x)", "INT(ABS(SIN(x))*#e^x, x)"),
    ("INT(ABS(SIN(a*x)), x)", "INT(ABS(SIN(a*x)), x)"),
]


@pytest.mark.parametrize(("text", "expected"), STEP_ANTIDERIVATIVES, ids=str)
def test_a_step_function_is_integrated_across_all_its_jumps(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize(
    ("integrand", "low", "high"),
    [
        ("ABS(SIN(x))", "-20", "20"),
        ("ABS(COS(x))", "31/3", "113/3"),
        ("SIGN(SIN(x))", "-41/3", "-1/7"),
        ("x*ABS(SIN(x))", "-13", "17"),
        ("FLOOR(x)", "-37/3", "52/7"),
        ("x*FLOOR(x)", "-5/2", "100/3"),
    ],
)
def test_a_step_antiderivative_agrees_with_the_integral_far_from_the_origin(
    integrand, low, high
):
    """The antiderivative's difference is the definite integral, which is worked out
    stretch by stretch between the jumps."""
    antiderivative = simp(f"INT({integrand}, x)")
    at = {end: re.sub(r"\bx\b", f"({end})", antiderivative) for end in (low, high)}
    integral = f"INT({integrand}, x, {low}, {high})"
    assert simp(f"{at[high]} - ({at[low]}) - {integral}") == "0"


def test_a_real_integral_over_a_positive_parameter_is_real():
    """Over a `t` declared positive sympy's antiderivative stays off its branch cut
    and the value is real; undeclared it is `#i*pi` off, and the integral stands."""
    text = "INT((x^t - 1)/LN(x), x, 0, 1)"
    assert simp(text, declared("t :epsilon Real (0, inf)")) == "LN(t + 1)"
