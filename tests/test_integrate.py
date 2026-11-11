"""Tests of integration.

Sections follow the structure of the integration package: the methods, the check an
answer must pass to be accepted, the printed form of the antiderivative, and handling
specific to definite integrals. For example, `INT(SIN(x)/x^2, x)` is
`CI(x) - SIN(x)/x` because sympy's first answer is wrong for negative `x` and fails the
check, and `INT(1/x^2, x, -1, 1)` is `inf`, not `-2`, because the singularity inside the
interval is found before the limits are substituted.

As in `test_simplify.py`, cases are written inline as input and expected output, so a
failure shows the mathematics. Each case illustrates a rule. The full Bronstein &
Semendyayev integral tables are covered by `test_integrate_bronstein.py`, so table
entries appear here only when they illustrate a rule.
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
        # Completing the square gives one arctangent over `4*a - 1`, whatever its sign,
        # instead of a sum over the roots.
        ("INT(1/(x^2 + x + a), x)", "2*ATAN((2*x + 1)/SQRT(4*a - 1))/SQRT(4*a - 1)"),
        (
            "INT(1/((x^2 + 1)*(x^2 + x + a)), x)",
            "-(2*a - 3)*ATAN((2*x + 1)/SQRT(4*a - 1))/(SQRT(4*a - 1)*(a^2 - 2*a + 2)) "
            "+ (a - 1)*ATAN(x)/(a^2 - 2*a + 2) + (LN(x^2 + x + a) - LN(x^2 + 1))/(2*a^2 "
            "- 4*a + 4)",
        ),
        # Two logarithms with the same coefficient are combined.
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
        # The half-angle substitution, with the logarithms rewritten in the whole angle.
        (
            "INT(1/(3 + 5*COS(x)), x)",
            "LN(3*COS(x) + 4*SIN(x) + 5)/8 - LN(3*COS(x) - 4*SIN(x) + 5)/8",
        ),
        ("INT(TAN(x)/(1 + TAN(x)), x)", "x/2 - LN(COS(x) + SIN(x))/2"),
        # A logarithm negative at zero is written over its negation.
        ("INT(TAN(x)/(TAN(x) - 1), x)", "LN(COS(x) - SIN(x))/2 + x/2"),
        # Over a linear form in both: one logarithm, real wherever the integrand is
        # finite.
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
        # Over the root of a linear function, the root is substituted.
        ("INT(1/((x + 2)*SQRT(x + 1)), x)", "2*ATAN(SQRT(x + 1))"),
        # Over the root of a quadratic: a multiple of the root plus one of `INT(1/root)`.
        ("INT(SQRT(x^2 + 1)/x^2, x)", "LN(SQRT(x^2 + 1) + x) - SQRT(x^2 + 1)/x"),
        (
            "INT(x/SQRT(x^2 + x + 1), x)",
            "SQRT(x^2 + x + 1) - LN(2*SQRT(x^2 + x + 1) + 2*x + 1)/2",
        ),
        # An arcsine when the leading coefficient is negative, or the equivalent
        # arctangent when the arcsine would need an absolute value.
        ("INT(1/SQRT(1 + x - x^2), x)", "-ASIN(SQRT(5)*(1 - 2*x)/5)"),
        ("INT(1/SQRT(c - a^2*x^2), x)", "ATAN(a*x/SQRT(c - a^2*x^2))/a"),
        (
            "INT(SQRT(1 - x^2)^3/x^4, x)",
            "ASIN(x) + (2*x + 1)*(2*x - 1)*SQRT(1 - x^2)/(3*x^3)",
        ),
        # Over a linear factor: a logarithm, or an arctangent when the quadratic is
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
        # The root of a quotient of linear polynomials, or two roots of linear
        # polynomials, are substituted. The root of a quotient isn't split apart.
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
    """The result is not a sum over the roots of a quadratic with parametric
    coefficients, and doesn't depend on the order in which exponentials are processed."""
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
        # The power of `x` common to both terms is factored out, leaving the
        # derivative of the inside.
        ("INT((3*x^2 + 10*x^27)^12, x)", "(10*x^25 + 3)^13/3250"),
        ("INT(x^24*(3 + 10*x^25)^12, x)", "(10*x^25 + 3)^13/3250"),
        ("INT((a*x^2 + b*x^27)^12, x)", "(b*x^25 + a)^13/(325*b)"),
        ("INT((2*x + 3)^7, x)", "(2*x + 3)^8/16"),
        ("INT(x*(x^2 + a)^3, x)", "(x^2 + a)^4/8"),
        ("INT(x^(n - 1)*(a + b*x^n)^3, x)", "(b*x^n + a)^4/(4*b*n)"),
        ("INT(x^(1 - 3*n)*(x^n + 2*x^(n + 2))^3, x)", "(2*x^2 + 1)^4/16"),
        # Negative and fractional powers.
        ("INT(x^24/(3 + 10*x^25)^5, x)", "-1/(1000*(10*x^25 + 3)^4)"),
        ("INT(x^4*SQRT(3 + 10*x^5), x)", "(10*x^5 + 3)^(3/2)/75"),
        # A multiple of the derivative times a power of the inside leaves a short
        # polynomial factor.
        ("INT(x^49*(3 + 10*x^25)^12, x)", "(130*x^25 - 3)*(10*x^25 + 3)^13/455000"),
        ("INT(x^5*(a + b*x^3)^7, x)", "(8*b*x^3 - a)*(b*x^3 + a)^8/(216*b^2)"),
        ("INT(x^5/SQRT(1 + x^3), x)", "2*SQRT(x^3 + 1)*(x^3 - 2)/9"),
        # A logarithm is left to the rational integrator, as is a quotient whose
        # numerator can vanish at zero.
        ("INT(x^5/(1 + x^3)^3, x)", "x^6/(6*(x^3 + 1)^2)"),
        ("INT(x^24/(3 + 10*x^25), x)", "LN(10*x^25 + 3)/250"),
        # Exponents that don't line up are expanded, as is a binomial to a power lower
        # than the polynomial factor's degree would be.
        ("INT(x*(x^2 + x^4)^3, x)", "x^14/14 + x^12/4 + 3*x^10/10 + x^8/8"),
        ("INT((x^2 + x^5)^3, x)", "x^16/16 + 3*x^13/13 + 3*x^10/10 + x^7/7"),
        ("INT(x^5*(1 + x^3), x)", "x^9/9 + x^6/6"),
        ("INT(x^8*(1 + x^3)^2, x)", "x^15/15 + x^12/6 + x^9/9"),
        # A symbolic power is made continuous at -1.
        (
            "INT(x^(n - 1)*(a + b*x^n)^p, x)",
            "((b*x^n + a)^(p + 1) - (a + b)^(p + 1))/(b*n*(p + 1))",
        ),
    ],
)
def test_a_power_of_a_binomial_is_integrated_as_a_power_of_the_binomial(text, expected):
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
        # An arc function of a multiple of `x`, divided by `x`, has no elementary
        # antiderivative, and the integral is returned immediately.
        ("INT(ATAN(x/a)/x, x)", "INT(ATAN(x/a)/x, x)"),
        ("INT(3*ACOS(2*x)/x, x)", "INT(3*ACOS(2*x)/x, x)"),
    ],
)
def test_an_arc_or_logarithm_times_a_rational_function_is_integrated_by_parts(
    text, expected
):
    assert simp(text) == expected


def test_an_even_integrand_over_a_positive_variable_is_kept_where_proved_odd():
    """An antiderivative of an even integrand found for positive `x` also holds for
    negative `x` if it's odd.

    `3*x^(5/3)/5` for `(x^2)^(1/3)` isn't odd: it isn't even real left of zero, where
    the integrand is. An antiderivative is also rejected if it isn't proved to
    differentiate back for positive `x`, like sympy's hypergeometric series for
    `1/SQRT(1 - x^4)`."""
    x = sp.Symbol("x", real=True)
    found = by_a_positive_substitution(x**2 / sp.sqrt(x**2 + 1), x)
    assert found is not None
    assert sp.simplify(found + found.subs(x, -x)) == 0
    assert by_a_positive_substitution((x**2) ** sp.Rational(1, 3), x) is None
    assert by_a_positive_substitution(1 / sp.sqrt(1 - x**4), x) is None


def test_an_integral_a_rewrite_leaves_behind_is_evaluated_too():
    # Cancelling splits the integrand in two, and the `INT(x, x)` part must be
    # evaluated too.
    state = declaring("F", "G")
    assert simp("INT((x*G(x) + F(x))/G(x), x)", state=state) == (
        "INT(F(x)/G(x), x) + x^2/2"
    )


def test_partial_fractions_over_a_sine_leave_only_the_nonelementary_part_standing():
    # The unevaluated integral is a separate term, not put over the common denominator.
    assert simp("INT(x/((SIN(a*x) + 1)*SIN(a*x)), x)") == (
        "INT(x/SIN(a*x), x) + (LN(1/(COS(a*x) + 1)) - 2*LN(TAN(a*x/2) + 1)"
        " + a*x/TAN(a*x/2 + pi/4) + LN(2))/a^2"
    )


# -- the gate ----------------------------------------------------------------------------


def test_an_antiderivative_that_does_not_differentiate_back_is_not_taken():
    """Sympy's Risch implementation answers some integrals wrongly.

    For real `a`, it gives zero for `INT(1/(x^2 + a), x)` and `x` for
    `INT(x^2/(x^2 + a), x)`. An answer proved not to differentiate back is rejected,
    and the integral is retried with textbook rules, which avoid that algorithm. The
    first two lines check that those answers differentiate back.

    Only answers proved wrong are rejected, so those that can't be checked either way
    are kept. Sympy's answer for `1/SQRT(1 - x^4)` is a hypergeometric series the
    notation can't express, so the integral stays, as in Derive.
    """
    assert simp("DIF(ATAN(x/SQRT(a))/SQRT(a), x)") == "1/(x^2 + a)"
    assert simp("DIF(x - SQRT(a)*ATAN(x/SQRT(a)), x)") == "1 - a/(x^2 + a)"
    assert simp("INT(1/SQRT(1 - x^4), x)") == "INT(1/SQRT(1 - x^4), x)"


@pytest.mark.parametrize(
    "text",
    [
        # A hypergeometric function in sympy.
        "INT(1/SQRT(1 - x^4), x)",
        # Incomplete gamma functions.
        "INT(x^a*#e^(-x), x)",
        # The harmonic number.
        "SUM(1/k, k, 1, n)",
    ],
    ids=str,
)
def test_an_operation_sympy_answers_only_in_a_function_without_a_name_stands(text):
    """When sympy's answer uses a function the notation can't express, the call stays
    as written, as Derive leaves what it can't do."""
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
    """The tables give reduction formulas for these, with no elementary antiderivative
    for a symbolic exponent. Sympy answers the first two with a Lerch function the
    notation can't express, and searches for over a minute on the last two without
    result. Each must come back unevaluated within a few seconds."""
    assert simp(text) == expected


def test_an_integral_that_will_not_evaluate_survives_as_itself():
    # Derive leaves the INT unevaluated, and so must we: an error would lose the
    # user's entry.
    assert simp("INT(SIN(x)/LN(x), x)") == "INT(SIN(x)/LN(x), x)"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("INT(SIN(x)/x^2, x)", "CI(x) - SIN(x)/x"),
    ],
)
def test_an_antiderivative_wrong_for_negative_arguments_is_done_again(text, expected):
    """Not in Derive, which has neither `CI` nor `COSH_INT`.

    For a real variable, sympy's answer here (and for `SINH(x)/x^2`) has `SIGN(x)`
    terms that are wrong for negative `x`. For an unrestricted variable the answer is
    correct, plus `LN(x^2)/2 - LN(x)`, a constant of integration that isn't printed.
    """
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # Sympy's substitution answers these only for `-1 < x < 1`, where the
        # integrand isn't real. Substituting the root gives an antiderivative valid on
        # both sides.
        ("INT(SQRT(x^2 - 1)/x, x)", "SQRT(x^2 - 1) - ATAN(SQRT(x^2 - 1))"),
        ("INT(1/(x*SQRT(x^2 - 1)), x)", "ATAN(SQRT(x^2 - 1))"),
        ("INT(1/(x*SQRT(x^2 - a^2)), x)", "ATAN(SQRT(x^2 - a^2)/a)/a"),
        # Sympy's answer is right, apart from its condition.
        ("INT(1/(x^2*SQRT(x^2 - 1)), x)", "SQRT(x^2 - 1)/x"),
        ("INT(SQRT(x^2 - 1)/x, x, -2, -1)", "pi/3 - SQRT(3)"),
    ],
)
def test_an_antiderivative_holds_wherever_the_integrand_is_real(text, expected):
    """A conditional antiderivative with no case for other values of the variable is
    undefined there, so it's not an answer."""
    assert simp(text) == expected


# -- the form ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        # The rational part's denominator is written over the integrand's factors.
        ("INT(1/(1 + x)^3, x)", "-1/(2*(x + 1)^2)"),
        ("INT(1/(x^2 + 2*x + 1)^2, x)", "-1/(3*(x + 1)^3)"),
        ("INT(x/(x^2 + 1)^2, x)", "-1/(2*(x^2 + 1))"),
        ("INT(1/(x^2 + 1)^2, x)", "ATAN(x)/2 + x/(2*(x^2 + 1))"),
        ("INT(1/(1 + x)^2, x)", "-1/(x + 1)"),
        (
            "INT(x^4/(a + b*x)^6, x)",
            "-(5*b^4*x^4 + 10*a*b^3*x^3 + 10*a^2*b^2*x^2 + 5*a^3*b*x + "
            "a^4)/(5*b^5*(b*x + a)^5)",
        ),
        # Over the irreducible factors, keeping the variable and binomials whole,
        # however the denominator is written.
        (
            "INT(1/((x + 1)^2*(x + 2)^2), x)",
            "-2*LN(x + 1) + 2*LN(x + 2) - (2*x + 3)/((x + 1)*(x + 2))",
        ),
        (
            "INT(1/(x^2 + 3*x + 2)^2, x)",
            "-2*LN(x + 1) + 2*LN(x + 2) - (2*x + 3)/((x + 1)*(x + 2))",
        ),
        (
            "INT(1/(x^4 + 3*x^2 + 2)^2, x)",
            "9*SQRT(2)*ATAN(SQRT(2)*x/2)/8 - 3*ATAN(x)/2 + x*(3*x^2 + 5)/(4*(x^2 + "
            "1)*(x^2 + 2))",
        ),
        (
            "INT(1/(x^3 + 1)^2, x)",
            "2*SQRT(3)*ATAN(2*SQRT(3)*x/3 - SQRT(3)/3)/9 + 2*LN(x + 1)/9 - LN(x^2 - x + "
            "1)/9 + x/(3*(x^3 + 1))",
        ),
        ("INT(x^2/(x^3 + 1)^2, x)", "-1/(3*(x^3 + 1))"),
        ("INT(x^2/(x^3 + 1)^3, x)", "-1/(6*(x^3 + 1)^2)"),
        ("INT(1/(x^2 - 1)^2, x)", "-LN(x - 1)/4 + LN(x + 1)/4 - x/(2*(x^2 - 1))"),
        (
            "INT(1/((x - 1)^2*(x + 1)^2), x)",
            "-LN(x - 1)/4 + LN(x + 1)/4 - x/(2*(x^2 - 1))",
        ),
        (
            "INT(1/(x^4 - 1)^2, x)",
            "3*ATAN(x)/8 - 3*LN(x - 1)/16 + 3*LN(x + 1)/16 - x/(4*(x^4 - 1))",
        ),
        (
            "INT(1/(x^3 - x)^2, x)",
            "-3*LN(x - 1)/4 + 3*LN(x + 1)/4 + (2 - 3*x^2)/(2*x*(x^2 - 1))",
        ),
        # A numeric denominator stays beside the factors, so the numerator keeps integer
        # coefficients.
        ("INT(x^2/(1 + x)^3, x)", "LN(x + 1) + (4*x + 3)/(2*(x + 1)^2)"),
        (
            "INT(x^5/(1 + x)^6, x)",
            "LN(x + 1) + (300*x^4 + 900*x^3 + 1100*x^2 + 625*x + 137)/(60*(x + 1)^5)",
        ),
        # A number times a single factor is multiplied out, which is how it reads back.
        (
            "INT((2*x^2 + 2*x + 13)/((x - 2)*(x^2 + 1)^2), x)",
            "-4*ATAN(x) + LN(x - 2) - LN(x^2 + 1)/2 + (3 - 4*x)/(2*x^2 + 2)",
        ),
        # The constant is chosen to make the rational part vanish at zero, when that
        # shortens its numerator.
        ("INT(x^4/(1 + x)^6, x)", "x^5/(5*(x + 1)^5)"),
        ("INT(x^2/(1 + x)^4, x)", "x^3/(3*(x + 1)^3)"),
        ("INT(x/(1 + x)^3, x)", "x^2/(2*(x + 1)^2)"),
        ("INT(x^3/(1 + x^2)^3, x)", "x^4/(4*(x^2 + 1)^2)"),
        ("INT(x^5/(2 + 3*x^2)^4, x)", "x^6/(12*(3*x^2 + 2)^3)"),
        ("INT(x^4/(2*x + 2)^6, x)", "x^5/(320*(x + 1)^5)"),
        ("INT(x^3/(1 + x)^6, x)", "x^4*(x + 5)/(20*(x + 1)^5)"),
        ("INT(x^4/(1 + b*x)^6, x)", "x^5/(5*(b*x + 1)^5)"),
        ("INT(x^4/(a^2 + 1 + b*x)^6, x)", "x^5/(5*(a^2 + 1)*(b*x + a^2 + 1)^5)"),
        # The result depends on the function, not on how it's written.
        (
            "INT(x^4/(x^6 + 6*x^5 + 15*x^4 + 20*x^3 + 15*x^2 + 6*x + 1), x)",
            "x^5/(5*(x + 1)^5)",
        ),
        ("INT(3*x^4*(x + 2)/((x + 2)*(x + 1)^6), x)", "3*x^5/(5*(x + 1)^5)"),
        # The constant of integration doesn't affect definite integrals.
        ("INT(x^4/(1 + x)^6, x, 0, t)", "t^5/(5*(t + 1)^5)"),
        ("INT(x^4/(1 + x)^6, x, 0, 1)", "1/160"),
        ("INT(x^4/(1 + x)^6, x, 0, inf)", "1/5"),
        ("INT(x^4/(1 + x)^6, x, -2, 1)", "inf"),
        ("INT(x^4/(1 + x)^6, x, -1, 0)", "inf"),
        ("INT(x^3/(1 + x^2)^3, x, 0, inf)", "1/4"),
        ("INT(x^4/(1 + b*x)^6, x, 0, 1)", "IF(b <= -1, inf, 1/(5*(b + 1)^5))"),
    ],
)
def test_the_rational_part_is_over_factors_and_vanishes_at_zero_where_shorter(
    text, expected
):
    """The rational part of a rational antiderivative is written over powers of the
    denominator's factors, with the constant that makes it zero at zero when that
    shortens its numerator.

    This isn't done when the denominator may vanish at zero for some parameter value,
    as `a + b*x` does at `a = 0`, since the constant would add a pole there."""
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, rational, longest",
    [
        ("INT(1/(x^6 + 16)^2, x)", " + x/(96*x^6 + 1536)", 270),
        (
            "INT(1/(x^6 + 16)^3, x)",
            " + x*(11*x^6 + 272)/(18432*x^12 + 589824*x^6 + 4718592)",
            322,
        ),
        ("INT(1/(x^6 + 2)^2, x)", " + x/(12*x^6 + 24)", 228),
    ],
)
def test_a_factored_rational_part_does_not_put_the_answer_over_one_denominator(
    text, rational, longest
):
    """If the rational part were factored, Simplify would put the whole answer over one
    denominator, multiplying the logarithms and arctangents by it and doubling the
    length. So the rational part stays expanded, as a separate term."""
    answer = simp(text)
    assert answer.endswith(rational)
    assert len(answer) <= longest


def test_an_antiderivative_is_written_with_no_constant_added():
    """Derive prints the antiderivative with no constant added.

    For the root of the quadratic formula integrated over its middle coefficient,
    Derive answers

        b*(SQRT(b^2 - 4*a*c) - b)/(4*a) - c*LN(SQRT(b^2 - 4*a*c) + b)

    while sympy's answer has `LN(2*b + 2*SQRT(b^2 - 4*a*c))`, differing by the
    constant `c*LN(2)`. We match Derive exactly.
    """
    answer = simp("INT((SQRT(b^2 - 4*a*c) - b)/(2*a), b)")
    assert answer == "b*(SQRT(b^2 - 4*a*c) - b)/(4*a) - c*LN(SQRT(b^2 - 4*a*c) + b)"


def test_a_case_split_over_one_antiderivative_is_no_split():
    """Sympy splits cases where its table has two forms of one function.

    Both come back as an `ATAN` for positive parameter and a pair of logarithms
    otherwise. The cases differ only by a constant, so the split is pointless, and
    Derive prints the `ATAN` for every parameter.
    """
    assert simp("INT(1/(x^2 + a), x)") == "ATAN(x/SQRT(a))/SQRT(a)"
    assert simp("INT(x^2/(x^2 + a), x)") == "x - SQRT(a)*ATAN(x/SQRT(a))"


def test_an_inverse_hyperbolic_does_not_win_a_split_by_being_shorter():
    """Derive prints no inverse hyperbolics in an antiderivative.

    It writes them as logarithms, e.g. `ASINH(x)` simplifies to
    `LN(SQRT(x^2 + 1) + x)`. Sympy's cases here are an `ASINH` for nonzero `a` and the
    logarithm otherwise. The `ASINH` is shorter, but we answer with the logarithm, as
    Derive does.
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

    Derive prints `(x^(n + 1) - 1)/(n + 1)`, choosing the constant so that the limit
    at `n = -1` is `LN(x)`, which makes it valid for every `n`. The integrand is
    analytic in `n`, so the integral from a fixed point is too; the pole in sympy's
    formula was only in the constant. The answers match Derive exactly.
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
    """Derive chooses such a constant only where its tables happen to.

    Its `INT(SINH(a*x), x)` has the `-1/a` because it goes through `#e^(a*x)`, but
    its `INT(SIN(a*x), x)` is `-COS(a*x)/a`, needlessly undefined at `a = 0`. We apply
    the rule to every parameter, so our answers differ from Derive's by a constant and
    also hold where Derive's don't: `(1 - COS(a*x))/a` is `0` at `a = 0`, the integral
    of `SIN(0)`.
    """
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # Continuous at `n = -1`, where the limit is `LN(x)^2/2`; the pole at
        # `(n + 1)^2` is only in the constant.
        (
            "INT(x^n*LN(x), x)",
            "(x^(n + 1)*(n + 1)*LN(x) - x^(n + 1) + 1)/(n^2 + 2*n + 1)",
        ),
        # An incomplete gamma function, which the notation can't express.
        ("INT(x^n*#e^x, x)", "INT(x^n*#e^x, x)"),
    ],
)
def test_a_parametric_antiderivative_is_the_generic_case(text, expected):
    """Sympy splits these on parameter values where the generic formula fails, such
    as `a^2 + b^2 = 0`, `a = b`, or `n = -1`. Those are measure-zero cases, so we
    print the generic case, made continuous by the choice of constant where possible."""
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # The integrand itself is singular at `a = 0`.
        ("INT(1/(a*x), x)", "LN(x)/a"),
        # A branch point rather than a pole, with no Laurent series.
        ("INT(1/(x^2 + a^2), x)", "ATAN(x/a)/a"),
        ("INT(x/(a^2 + b^2*x^4), x)", "ATAN(b*x^2/a)/(2*a*b)"),
    ],
)
def test_a_pole_the_integrand_owns_is_left_in_the_antiderivative(text, expected):
    """Only a constant of integration may be removed.

    The principal part of `LN(x)/a` at `a = 0` is `LN(x)/a` itself, which isn't a
    constant, and `ATAN(x/a)` has no Laurent expansion, so both answers are unchanged.
    """
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("INT(SIN(a*x)^2/x, x)", "LN(2*a*x)/2 - CI(2*a*x)/2"),
        ("INT((COS(a*x) - COS(b*x))/x, x)", "-LN(a*x) + LN(b*x) + CI(a*x) - CI(b*x)"),
        ("INT((#e^(a*x) - 1)/x, x)", "EI(a*x) - LN(ABS(a*x))"),
    ],
)
def test_an_antiderivative_is_continuous_where_the_integrand_is_regular(text, expected):
    """For negative `a`, `LN(x)` and `CI(2*a*x)` pick up their imaginary parts on
    opposite sides of zero, so `LN(x)/2 - CI(2*a*x)/2` jumps there. Taking the
    logarithm of the `CI`'s argument keeps them on the same side."""
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
    """The sum over the roots is written out whenever the roots can be found.

    Each conjugate pair gives a logarithm and an arctangent, and each real root a
    logarithm, as Derive writes all three. Sympy 1.14 silently drops the two pairs
    with cube roots in their coordinates from its real-form conversion, returning
    only the arctangent in `2^(1/3)*x/2`, a third of the answer, for `1/(x^6 + 16)`.
    `engine.rational` produces the full answer, and also finds the quintic's roots,
    which sympy left as a `ROOT_SUM`. The last answer is written in cosines and sines
    of `pi/7`, the seventh roots of unity.
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
    """The roots of `16777216*t^8 + 1` each come back in a different radical form, so
    no root's conjugate is written as the conjugate of its expression. Pairs are found
    by numeric value, and each is written using the shorter of its two roots. Left as
    a `ROOT_SUM`, the definite integral would fall to sympy, which loses roots and
    answers 0 from 0 to 1.
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
    # Over the tenth roots of unity, sympy kept only the pair on the imaginary axis and
    # answered `pi/10`.
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
    coefficients that sum to zero only via identities of cosines of multiples of
    `pi/n`, so the naive limit at `inf` is `inf`. The integral is `pi/(n*SIN(pi/n))`."""
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
    """A limit containing the integration variable evaluates the antiderivative there;
    the bound variable is distinct from the one in the limit."""
    assert simp(text) == expected


def _summed_over_numeric_roots(answer: str) -> complex:
    """Evaluate the answer, summing each `ROOT_SUM` over numerically found roots.

    Sympy's `RootSum.doit` uses the quartic formula and takes the logarithms on the
    wrong branch."""
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
        # At `-inf` each logarithm tends to `LN(-1)` from one side of the cut or the
        # other, so the integral stays.
        ("INT(1/(x^4 + x + 1), x, -inf, inf)", "INT(1/(x^4 + x + 1), x, -inf, inf)", None),
    ],
)
def test_an_integral_of_a_rational_function_over_a_root_sum_is_taken_at_its_ends(
    text, expected, value
):
    """When not all roots are found, the antiderivative is a `ROOT_SUM`, which sympy
    can't take limits of. It's continuous, so at a finite limit it's simply evaluated,
    and at `inf` each logarithm (of the variable minus a root) tends to `LN(1)`. Sympy
    isn't used instead because it searched for minutes for the real roots of the
    resolvent of `1/(x^5 + x + 1)`."""
    answer = simp(text)
    assert answer == expected
    if value is not None:
        assert abs(_summed_over_numeric_roots(answer) - value) < 1e-12


def test_an_antiderivative_over_a_denominator_that_does_not_factor_keeps_its_sum():
    """The logarithmic part is a sum over the roots of a polynomial, `ROOT_SUM(p, t, u)`.

    The answer must never drop it and keep only the rest: `1/(x*(x^4 + x + 1))`
    doesn't integrate to `LN(x)`, and a silently incomplete answer is worse than an
    unreadable one.

    The polynomial is sympy's resolvent rather than the denominator, so its
    coefficients differ; its roots are transformed roots of the denominator, and the
    summand is transformed to match.
    """
    assert simp("INT(1/(x*(x^4 + x + 1)), x)") == (
        "LN(x) + ROOT_SUM(229*t^4 + 229*t^3 + 96*t^2 + 16*t + 1, t, "
        "t*LN(x + 129614*t^4/1713 - 1145*t^3/1713 - 76423*t^2/1713 "
        "- 45760*t/1713 - 6286/1713))"
    )


def test_an_antiderivative_over_cardano_roots_keeps_its_sum_and_comes_back_at_once():
    """The roots of a cubic with one real root are written by Cardano's formula, in
    cube roots of irrational numbers. Writing the antiderivative over them took
    minutes of sympy arithmetic and gave a pages-long answer, so the sum over the
    cubic's roots stays. When the polynomial factors, as for `1/(x^5 + x + 1)`, the
    quadratic factor's pair is still written out."""
    assert simp("INT(1/(x^3 - x^2 + 1), x)") == (
        "ROOT_SUM(23*t^3 + t - 1, t, t*LN(x + 46*t^2/25 + 69*t/25 - 7/25))"
    )
    assert simp("INT(1/(x^5 + x + 1), x)") == (
        "5*SQRT(3)*ATAN(2*SQRT(3)*x/3 + SQRT(3)/3)/21 - LN(x^2 + x + 1)/14 "
        "+ ROOT_SUM(161*t^3 - 23*t^2 - 12*t - 1, t, t*LN(x - 216384*t^4/625 "
        "+ 54096*t^3/625 - 3284*t^2/625 + 5941*t/625 + 256/625))"
    )


def test_a_sum_over_the_roots_of_a_polynomial_is_an_expression_and_not_a_string():
    """`ROOT_SUM` is a real expression that can be differentiated and simplified.

    A quartic that doesn't factor has an antiderivative that is just such a sum.
    Differentiating it back to the integrand shows it's a genuine expression; a
    `Kind.STRING` would print the same but couldn't be computed with.
    """
    answer = simplify(parse("INT(1/(x^4 + x + 1), x)"), Context())
    assert answer.node.kind is not Kind.STRING
    assert answer.text == (
        "ROOT_SUM(229*t^4 + 18*t^2 + 8*t + 1, t, "
        "t*LN(x + 2061*t^3/64 - 687*t^2/64 + 391*t/64 + 27/64))"
    )
    assert simp(f"DIF({answer.text}, x)") == "1/(x^4 + x + 1)"
    # It's also a fixed point of simplification.
    assert simp(answer.text) == answer.text


@pytest.mark.parametrize(
    "text, expected",
    [
        # `LN(SIN(x) - 1)` is complex for every real `x`; `LN(1 - SIN(x))` differs by a
        # constant and is real wherever the integral is.
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
        # `GAMMA(x)` is written as the factorial `(x - 1)!`.
        ("DIF(GAMMA(x), x)", "(x - 1)!*DIGAMMA(x)"),
        ("DIF(LAMBERTW(x), x)", "LAMBERTW(x)/(x*(LAMBERTW(x) + 1))"),
        ("INT(SIN(x)/x, x)", "SI(x)"),
        # The product is rewritten as the sine of the double angle.
        ("INT(SIN(a*x)*COS(a*x)/x, x)", "SI(2*a*x)/2"),
        ("INT(EXP(x)/x, x)", "EI(x)"),
        # Sympy's answer is on another branch, a full turn around the origin; the
        # principal polylogarithm is also an antiderivative.
        ("INT(LN(1 - x)/x, x)", "-POLYLOG(2, x)"),
        ("DIF(-POLYLOG(2, x), x)", "LN(1 - x)/x"),
    ],
    ids=str,
)
def test_a_special_function_an_elementary_operation_produces_reads_back(text, expected):
    """The special functions we support beyond Derive's are those our operations
    produce from elementary input, and each answer is a fixed point."""
    assert simp(text) == expected
    assert simp(expected) == expected


def test_an_integral_answered_by_a_special_function_reads_back_as_that_function():
    """An answer must be printed under a name that reads back as the same function.

    A name that means something else would silently read back as a different
    function, which simplifying the answer again catches. `CHI` is Derive's
    chi-square distribution, not the cosh-integral, and sympy's logarithmic integral
    for `1/LN(x)` has no name of its own, so it's written as `EI(LN(x))`.
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
    """Sympy's Fresnel integrals are written as error functions of complex
    arguments."""
    assert simp(text) == expected


def test_an_answer_written_over_the_error_functions_differentiates_back():
    """The Fresnel integral written with `ERF` is still an antiderivative.

    Simplify doesn't turn the derivative's complex exponentials back into a sine, so
    the derivative is compared with the integrand numerically at a point.
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
        # A standalone logarithm term still drops its constant factor.
        ("INT(1/(2*x + 2), x)", "LN(x + 1)/2"),
    ],
)
def test_a_constant_inside_a_logarithm_is_shed_only_by_a_term(text, expected):
    assert simp(text) == expected


# -- definite integrals ------------------------------------------------------------------

SINGULAR_INTEGRALS = [
    # An integrand of one sign that blows up at least as fast as 1/distance diverges to
    # that sign's infinity, wherever the singularity is. Naively evaluating the
    # antiderivative across it would give a finite number.
    ("INT(1/x^2, x, -1, 1)", "inf"),
    ("INT(1/x^4, x, -1, 1)", "inf"),
    ("INT(1/ABS(x), x, -1, 1)", "inf"),
    ("INT(1/x^2, x, -1, 2)", "inf"),
    ("INT(1/(x - 1)^2, x, 0, 2)", "inf"),
    ("INT(SEC(x)^2, x, 0, pi)", "inf"),
    ("INT(#e^(1/x), x, -1, 1)", "inf"),
    ("INT(1/x^2, x, -inf, inf)", "inf"),
    ("INT(1/x^2, x, 1, -1)", "-inf"),
    # Likewise at an endpoint, without the logarithm's imaginary part.
    ("INT(1/x, x, -1, 0)", "-inf"),
    ("INT(LN(x)/x, x, 0, 1)", "-inf"),
    # Diverging to both infinities gives `inf - inf`, which has no value. The principal
    # value isn't used.
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
    # Integrated separately on each side of the singularity, where the antiderivative
    # jumps.
    ("INT(#e^(-1/x^2)/x^2, x, -1, 1)", "SQRT(pi) - SQRT(pi)*ERF(1)"),
    # A divergence of unknown sign is split on the parameter's sign.
    ("INT(a/x^2, x, -1, 1)", "IF(a > 0, inf, IF(a < 0, -inf, 0))"),
    ("INT(a/x, x, -1, 1)", "IF(a /= 0, ?, 0)"),
    # Undecided at zero, where the antiderivative's limit isn't the integral's.
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
    # A pole that moves with the parameter lies in the interval for some values, where
    # the integral diverges; the antiderivative's difference holds only elsewhere.
    ("INT(1/(x - a)^2, x, 0, 2)", "IF(0 <= a <= 2, inf, -2/(a*(2 - a)))"),
    ("INT(1/(x + a)^2, x, 0, inf)", "IF(a <= 0, inf, 1/a)"),
    ("INT(1/(x^2 - a^2), x, 0, inf)", "IF(a = 0, inf, ?)"),
    # The poles are real only where `SIN(a)` is zero, where the answer is undefined too.
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
    # Diverging at the singular end to the same infinity whichever side the other end
    # is on gives that infinity; opposite infinities leave the integral unevaluated.
    ("INT(COS(a*x)/x, x, 0, alpha)", "inf"),
    ("INT(1/x^3, x, alpha, 0)", "-inf"),
    ("INT(1/x^2, x, 0, alpha)", "INT(1/x^2, x, 0, alpha)"),
    # A power of zero, which is 0 or infinite depending on the exponent's sign.
    ("INT(x^n, x, 0, 1)", "IF(n > -1, 1/(n + 1), inf)"),
    ("INT(1/x^n, x, 0, 1)", "IF(n < 1, 1/(1 - n), inf)"),
    # Convergent for one sign of the parameter, divergent to one infinity otherwise.
    ("INT(#e^(-a*x), x, 0, inf)", "IF(a > 0, 1/a, inf)"),
    ("INT(#e^(a*x), x, 0, inf)", "IF(a < 0, -1/a, inf)"),
    ("INT(#e^(-a*x^2), x, -inf, inf)", "IF(a > 0, SQRT(pi)/SQRT(a), inf)"),
    ("INT(x^n*LN(x), x, 0, 1)", "IF(n > -1, -1/(n + 1)^2, -inf)"),
    ("INT(#e^(-x)*x^n, x, 0, inf)", "IF(n > -1, n!, inf)"),
    # One expression covering every sign.
    ("INT(SIN(a*x)/x, x, 0, inf)", "pi*SIGN(a)/2"),
    ("INT(COS(a*x)*#e^(-x^2), x, -inf, inf)", "SQRT(pi)*#e^(-a^2/4)"),
    # For nonpositive `a` it diverges by oscillation and stays unevaluated.
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
        # The terms of a sum needn't converge where the sum does.
        ("INT((x^t - x^s)/LN(x), x, 0, 1)", "-INT((x^s - x^t)/LN(x), x, 0, 1)"),
        # Integrals the methods can do are still found.
        ("INT(LN(1 + x)/x, x, 0, 1)", "pi^2/12"),
        ("INT(LN(x)/(x^2 + 1), x, 0, inf)", "0"),
    ],
)
def test_a_definite_integral_with_no_antiderivative_in_reach_stands(text, expected):
    """Sympy evaluates these via an antiderivative in a positive variable, for which
    its manual rules search for minutes without finding one. Each must come back
    unevaluated within a few seconds."""
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
    blows up, and a polylogarithm on its cut are not proved continuous. An argument
    whose sign is fixed for every parameter value is."""
    assert unbroken(expression, _T, sp.S(low), sp.S(high)) is proved


STEP_INTEGRALS = [
    # Split at the jumps, integrating each piece with the step held constant.
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
    # A sign of a non-polynomial, whose antiderivative holds only near the origin.
    ("INT(ABS(SIN(x)), x, 0, 10*pi)", "20"),
    ("INT(ABS(SIN(x)), x, -10, 10)", "2*COS(10) + 14"),
    ("INT(SIGN(SIN(x)), x, 0, 3*pi)", "pi"),
    ("INT(SIGN(COS(x)), x, 0, 5)", "5 - 2*pi"),
    ("INT(ABS(LN(x)), x, 1/2, 2)", "3*LN(2)/2 - 1/2"),
    # The jumps can't be located numerically.
    ("INT(FLOOR(x), x, 0, a)", "INT(FLOOR(x), x, 0, a)"),
]


@pytest.mark.parametrize(("text", "expected"), STEP_INTEGRALS, ids=str)
def test_a_step_function_is_integrated_between_its_jumps(text, expected):
    assert simp(text) == expected


STEP_ANTIDERIVATIVES = [
    # A sign of a periodic function jumps forever, so the antiderivative counts jumps
    # with a floor: each arch of `ABS(SIN(x))` adds 2.
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
    # Each arch's contribution grows polynomially with the arch's number.
    (
        "INT(x*ABS(SIN(x)), x)",
        "pi*FLOOR(x/pi)^2 + ABS(SIN(x)) - x*SIGN(SIN(x))*COS(x) + pi*FLOOR(x/pi)",
    ),
    # A floor is constant between integers, and the constants add up.
    ("INT(FLOOR(x), x)", "-FLOOR(x)^2/2 + x*FLOOR(x) - FLOOR(x)/2"),
    ("INT(MOD(x, 1), x)", "FLOOR(x)^2/2 - x*FLOOR(x) + FLOOR(x)/2 + x^2/2"),
    ("INT(FLOOR(-x), x)", "FLOOR(-x)^2/2 + x*FLOOR(-x) + FLOOR(-x)/2"),
    (
        "INT(x*FLOOR(x), x)",
        "-FLOOR(x)^3/6 - FLOOR(x)^2/4 + x^2*FLOOR(x)/2 - FLOOR(x)/12",
    ),
    # Unequally spaced jumps, non-periodic jumps, and geometrically growing arches get
    # no answer, since a split near the origin would be wrong away from it.
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
    """The antiderivative's difference equals the definite integral, which is computed
    piece by piece between the jumps."""
    antiderivative = simp(f"INT({integrand}, x)")
    at = {end: re.sub(r"\bx\b", f"({end})", antiderivative) for end in (low, high)}
    integral = f"INT({integrand}, x, {low}, {high})"
    assert simp(f"{at[high]} - ({at[low]}) - {integral}") == "0"


def test_a_real_integral_over_a_positive_parameter_is_real():
    """With `t` declared positive, sympy's antiderivative stays off its branch cut and
    the value is real. Undeclared, it's off by `#i*pi` and the integral stays."""
    text = "INT((x^t - 1)/LN(x), x, 0, 1)"
    assert simp(text, declared("t :epsilon Real (0, inf)")) == "LN(t + 1)"
