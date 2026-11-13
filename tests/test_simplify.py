"""What Simplify must produce: the normal form, and the command's contract.

Each case is written as a user would type it and checked against the text the
worksheet would show. Cases are inline rather than in a corpus file so that a failure
names the mathematics involved.

This file covers what Simplify itself does: numbers, polynomials and rational
functions, the normal form of sums and quotients, the domains that license a rewrite,
absolute values, trigonometry, logarithms and the branch setting, substitution, and the
general contract: every node kind and builtin survives simplification, and every answer
is a fixed point of printing and of a second Simplify. Other areas have their own files:
`test_integrate.py`, `test_calculus.py`, `test_matrices.py`, `test_logic.py` (relations,
logic, conditionals), `test_approximate.py` (precision modes) and `test_functions.py`
(special values and named functions). Their case lists are imported here only for the
fixed-point check.

Where the engine's form differs from the manual's (term order, an equivalent spelling),
the engine's form is recorded and the case says so. No expectation is weakened to make
a test pass: each is either what Derive prints or a hand-checked equivalent of
comparable simplicity.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace

import pytest
from sexpr import to_sexpr
from simplifying import COMPLEX_X, POSITIVE_X, declared, declaring, parse, simp
from test_approximate import APPROXIMATED
from test_calculus import CALCULUS, INFINITE_SUMS
from test_functions import INDICATOR_AND_NORMAL, SPECIAL
from test_logic import CONDITIONALS, LOGIC, TRUTH_TABLES
from test_matrices import (
    DECOMPOSITION,
    GENERATED,
    ITERATION,
    LINEAR_ALGEBRA,
    NUMBER_THEORY,
    SELECTED,
    STATISTICS,
    STRUCTURE,
    VECTOR_CALCULUS,
    VECTORS,
)

from rederive.engine.computing import Context, from_sympy, simplify, to_sympy
from rederive.engine.context import Angle, Branch, Direction, Precision, TrigPower
from rederive.model.expr import Kind
from rederive.syntax import ParseState, VariableDeclaration, parse_expression
from rederive.syntax.names import BUILTIN_FUNCTIONS


# -- numbers, polynomials and rational functions ------------------------------

ARITHMETIC = [
    ("1/2 + 1/6", "2/3"),
    ("22% + 36%", "29/50"),
    ("(7/8)^2", "49/64"),
    ("5^99", str(5**99)),
    ("54!", str(math.factorial(54))),
    ("2*y*3", "6*y"),
    ("x^2*y*x", "x^3*y"),
    ("3*x + 7 + x", "4*x + 7"),
    ("(3*x*y^3)^2", "9*x^2*y^6"),
    ("x - x", "0"),
    ("x/x", "1"),
    ("x^0", "1"),
    ("0^0", "1"),
    ("MOD(2^100, 7)", "2"),
    ("GCD(12, 18, 27)", "3"),
    # GCD of non-numbers stays unevaluated. Checked against the original, which writes
    # the arguments least main first.
    ("GCD(m, n)", "GCD(n, m)"),
    ("GCD(6, 4, m)", "GCD(2, m)"),
    ("GCD(m, 0)", "ABS(m)"),
    ("LCM(m, n)", "ABS(m*n)/GCD(n, m)"),
    ("LCM(m, 2)", "2*ABS(m)/GCD(2, m)"),
    ("LCM(x, x^2)", "x^2*ABS(x)/GCD(x, x^2)"),
]

RATIONAL = [
    ("(x^2 + 2*x*y + y^2)/(x^2 - y^2)", "(x + y)/(x - y)"),
    ("(x + 1)^2 - x^2", "2*x + 1"),
    ("2*x/(x^2 - 1) - 1/(x - 1)", "1/(x + 1)"),
    # Difference of squares, found by multiplying out the product. Expanding the
    # power as well would produce a hundred terms, slowly.
    ("x^2 - (x + (y+1)^50)*(x - (y+1)^50)", "(y + 1)^100"),
    (
        "a^3/((a-b)*(a-c)) + b^3/((b-c)*(b-a)) + c^3/((c-a)*(c-b))",
        "a + b + c",
    ),
]

RADICALS = [
    ("SQRT(8)", "2*SQRT(2)"),
    ("SQRT(1/12)", "SQRT(3)/6"),
    ("SQRT(4 - 2*SQRT(3))", "SQRT(3) - 1"),
    ("SQRT(2)*SQRT(3)", "SQRT(6)"),
    # A cube root denests when the radicand is the cube of a sum of surds. Sympy
    # only denests square roots, so this factors the minimal polynomial over the
    # field the surds generate.
    ("(243*SQRT(5) - 294*SQRT(3))^(1/3)", "3*SQRT(5) - 2*SQRT(3)"),
    ("(10 + 6*SQRT(3))^(1/3)", "SQRT(3) + 1"),
    # Denesting counts as simpler even when the result is no shorter: the cube of the
    # golden ratio is `2 + SQRT(5)`, and `SQRT(2 + SQRT(3))` is two surds over two. A
    # root that does not denest stays.
    ("(2 + SQRT(5))^(1/3)", "SQRT(5)/2 + 1/2"),
    ("SQRT(2 + SQRT(3))", "SQRT(6)/2 + SQRT(2)/2"),
    ("SQRT(4 + SQRT(15))", "SQRT(10)/2 + SQRT(6)/2"),
    ("x*SQRT(2 + SQRT(3))", "(SQRT(6) + SQRT(2))*x/2"),
    ("SQRT(1 + SQRT(2))", "SQRT(SQRT(2) + 1)"),
    # An integer power of a sum of surds, or its reciprocal, is written as an element
    # of the field they span, even if that is longer. Expand gives the same result.
    ("(1 + SQRT(2))^2", "2*SQRT(2) + 3"),
    ("(1 + SQRT(2))^5", "29*SQRT(2) + 41"),
    ("(1 + SQRT(2))^(-3)", "5*SQRT(2) - 7"),
    ("1/(1 + SQRT(2))^2", "3 - 2*SQRT(2)"),
    ("(1 + 2^(1/3))^3", "3*2^(2/3) + 3*2^(1/3) + 3"),
    ("(SQRT(2) + SQRT(3) + SQRT(5))^4", "56*SQRT(15) + 64*SQRT(10) + 80*SQRT(6) + 224"),
    ("(2 + #i)^3", "2 + 11*#i"),
    ("(2 - #i)^2", "3 - 4*#i"),
    ("(1 + SQRT(2))^5*x", "(29*SQRT(2) + 41)*x"),
    ("EXPAND((SQRT(2) + 1)^3)", "5*SQRT(2) + 7"),
    ("EXPAND((1 + 2^(1/3))^3)", "3*2^(2/3) + 3*2^(1/3) + 3"),
    # A power whose value would run to hundreds of digits keeps its exponent, and a
    # power involving a transcendental or a variable is not expanded.
    ("(1 + SQRT(2))^1000", "(SQRT(2) + 1)^1000"),
    ("(1 + pi)^2", "(pi + 1)^2"),
    ("(x + SQRT(2))^2", "(x + SQRT(2))^2"),
    ("(x + 1)^5", "(x + 1)^5"),
    # The quotient from 3.8 p.47. None of the four roots simplifies on its own, but
    # the numerator is `(SQRT(3) + 1)^(3/2)` and the denominator twice that, so the
    # whole quotient lies in the field of `SQRT(3)`.
    ("(SQRT(SQRT(3) + 1) + SQRT(3*SQRT(3) + 3))/SQRT(24*SQRT(3) + 40)", "1/2"),
    # The value need not be rational, only in that field.
    (
        "SQRT(3)*(SQRT(SQRT(3) + 1) + SQRT(3*SQRT(3) + 3))/SQRT(24*SQRT(3) + 40)",
        "SQRT(3)/2",
    ),
    # A variable factor doesn't prevent it.
    ("x*(SQRT(SQRT(3) + 1) + SQRT(3*SQRT(3) + 3))/SQRT(24*SQRT(3) + 40)", "x/2"),
    # Combined differently, the same roots stay: the value `(SQRT(3) + 1)^(3/2)/2`
    # is one root deeper than the field, so there is no shorter spelling.
    (
        "(SQRT(SQRT(3) + 1) + SQRT(3*SQRT(3) + 3))^2/SQRT(24*SQRT(3) + 40)",
        "(SQRT(3*SQRT(3) + 3) + SQRT(SQRT(3) + 1))^2/SQRT(24*SQRT(3) + 40)",
    ),
]

COMBINATORIAL = [
    ("(3/2)!", "3*SQRT(pi)/4"),
    ("GAMMA(1/2)", "SQRT(pi)"),
    ("(n + 2)!/(n - 1)!", "n*(n + 1)*(n + 2)"),
    ("COMB(5, 2)", "10"),
    ("PERM(5, 2)", "20"),
    # Section 6.9 defines both as ratios of factorials, which are reduced:
    # `COMB(x, 2)` is `x!/(2*(x - 2)!)`, which cancels to `x*(x - 1)/2`.
    ("COMB(x, 2)", "x*(x - 1)/2"),
    ("PERM(x, 2)", "x*(x - 1)"),
    # Inside an unevaluated head, `COMB` is kept. Rewritten as factorials of the
    # index, sympy's search for a closed form of the sum never returns.
    (
        "SUM(COMB(n, k)*LN(k + 1), k, 0, n)",
        "SUM(COMB(n, k)*LN(k + 1), k, 0, n)",
    ),
]


@pytest.mark.parametrize(
    ("text", "expected"), ARITHMETIC + RATIONAL + RADICALS + COMBINATORIAL, ids=str
)
def test_numbers_and_algebra(text, expected):
    assert simp(text) == expected


def test_a_polynomial_is_not_factored():
    # Factoring is the Factor command's job. Simplify leaves the sum alone even
    # though the product is shorter.
    assert simp("x^2 + 2*x") == "x^2 + 2*x"


def test_a_high_power_of_a_sum_is_not_multiplied_out_to_be_thrown_away():
    # Derive returns both unchanged, instantly. Expanded, the sixtieth power has
    # six hundred thousand terms that no rewrite keeps, so the expansion's size is
    # estimated first and the expansion is declined. The ratio covers `cancel`,
    # which is tried on quotients but must not be at this size.
    assert simp("(v + w + x + y + z)^60") == "(x + y + z + v + w)^60"
    assert simp("(v + w + x + y + z)^60/q") == "(x + y + z + v + w)^60/q"


# -- the normal form ----------------------------------------------------------

#: All checked against the original. A sum is written as a rational function of its
#: most main variable; a product or power that is not itself a sum is left as written.
#:
#: Terms and term order match the original. The order list decides both which sums
#: get multiplied out and which term leads; these cases are about the former, i.e.
#: what stays folded because the primary variable is not in it.
NORMAL_FORM = [
    # A sum in the primary variable is multiplied out however long it gets; a term
    # free of it is not touched.
    ("(x + 1)^9 + y", "x^9 + 9*x^8 + 36*x^7 + 84*x^6 + 126*x^5 + 126*x^4 "
                      "+ 84*x^3 + 36*x^2 + 9*x + y + 1"),
    ("(y + 1)^9 + x", "x + (y + 1)^9"),
    ("(x + 1)^2 + y", "x^2 + 2*x + y + 1"),
    # Not a sum, so nothing happens - this is not a wholesale expansion.
    ("(x + 1)^9", "(x + 1)^9"),
    ("2*x*(x - 3)^2", "2*x*(x - 3)^2"),
    ("(x + 1)*(y + 1)", "(x + 1)*(y + 1)"),
    ("(x + 1)*(x + 2)", "(x + 1)*(x + 2)"),
    ("(x + y)^2", "(x + y)^2"),
    ("(x + 1)^2/y", "(x + 1)^2/y"),
    # Calls whose arguments are the same polynomial written two ways cancel.
    ("#e^(2*x*(y + 1) + y) - #e^(2*x*y + 2*x + y)", "0"),
    ("SIN(2*x*(y + 1)) - SIN(2*x*y + 2*x)", "0"),
    # The rule applies wherever the sum appears: in a product, a power, a function
    # argument, a denominator, a vector element.
    ("z*((x + 1)^2 + y)", "z*(x^2 + 2*x + y + 1)"),
    ("((x + 1)^2 + y)^2", "(x^2 + 2*x + y + 1)^2"),
    ("SIN((x + 1)^2 + y)", "SIN(x^2 + 2*x + y + 1)"),
    ("1/((x + 1)^2 + y)", "1/(x^2 + 2*x + y + 1)"),
    ("1/((y + 1)^2 + x)", "1/(x + (y + 1)^2)"),
    ("[(x + 1)^2 + y, (y + 1)^2 + x]", "[x^2 + 2*x + y + 1, x + (y + 1)^2]"),
    # A non-integer exponent can't be expanded, but the power still involves `x`,
    # so it leads the `y`.
    ("(x + 1)^n + y", "(x + 1)^n + y"),
    # Coefficients are opaque: the part free of `x` is one quantity, so the answer
    # is in powers of `2*y + 1`, not `y`. The term free of `x` is then written about
    # `y`, its own primary variable.
    ("(x + 2*y + 1)^3 + z", "x^3 + 3*x^2*(2*y + 1) + 3*x*(2*y + 1)^2 "
                            "+ 8*y^3 + 12*y^2 + 6*y + z + 1"),
    # The `x^2` coefficient keeps its numeric content, where the original writes
    # `2*x^2*(y + 3)`. The original is inconsistent here; this follows the
    # manual's LAPLACIAN example.
    ("((x + 1)^2 + y)^2 + z", "x^4 + 4*x^3 + x^2*(2*y + 6) + 4*x*(y + 1) "
                              "+ y^2 + 2*y + z + 1"),
    ("x*((y + 1)^2 + z) + w", "x*(y^2 + 2*y + z + 1) + w"),
    # Off the order list, variables order among themselves alphabetically.
    ("(a + 1)^2 + b", "a^2 + 2*a + b + 1"),
    ("(b + 1)^2 + a", "a + (b + 1)^2"),
    # The manual's 4.2 exercise.
    ("(5x - 3x + 1)^7 - x", "128*x^7 + 448*x^6 + 672*x^5 + 560*x^4 "
                            "+ 280*x^3 + 84*x^2 + 13*x + 1"),
]


@pytest.mark.parametrize(("text", "expected"), NORMAL_FORM, ids=str)
def test_a_sum_is_written_about_its_primary_variable(text, expected):
    assert simp(text) == expected


#: The rational half of the same rule, also checked against the original. Terms with
#: the primary variable go over a common denominator, and long division extracts the
#: polynomial part, leaving a proper fraction.
RATIONAL_FORM = [
    ("1/(x + 1) + 1/(x + 2) + y", "y + (2*x + 3)/((x + 1)*(x + 2))"),
    ("1/(x + 1)^2 + 1/(x + 1) + y", "y + (x + 2)/(x + 1)^2"),
    ("y/(x + 1) + z/(x + 1)", "(y + z)/(x + 1)"),
    ("(x^3 + 1)/(x + 2) + y", "x^2 - 2*x + y - 7/(x + 2) + 4"),
    ("x/(x + 1) + y", "y - 1/(x + 1) + 1"),
    ("x^2 + 1/(x + 1) + y", "x^2 + y + 1/(x + 1)"),
    # The denominator keeps its written shape: Derive leaves the two factors rather
    # than multiplying out a quartic. Derive writes the numerator as
    # `2*(x^2 + y^2 + 18)`, a form we can't represent - see
    # `test_a_numeric_factor_stays_among_the_terms_it_divides`.
    ("1/(9 + x^2 + (y - 3)^2) + 1/(9 + x^2 + (y + 3)^2)",
     "(2*x^2 + 2*y^2 + 36)/((x^2 + y^2 + 6*y + 18)*(x^2 + y^2 - 6*y + 18))"),
    # A denominator free of the primary variable is part of a coefficient, so it
    # is combined only when several terms share it.
    ("x/y + z", "x/y + z"),
    ("1/y + z", "1/y + z"),
    ("(x + 1)^2/y + z", "(x^2 + 2*x + y*z + 1)/y"),
    ("x/y + x/z", "x*(1/y + 1/z)"),
    ("x/(y + 1) + x/(y + 2)", "x*(2*y + 3)/((y + 1)*(y + 2))"),
]


@pytest.mark.parametrize(("text", "expected"), RATIONAL_FORM, ids=str)
def test_a_sum_of_ratios_is_written_over_one_denominator(text, expected):
    assert simp(text) == expected


#: A quotient's numerator has its common factor pulled out, as Derive's gcd leaves
#: it. Checked against the original.
NUMERATOR_CONTENT = [
    ("(x^2 + x)/n", "x*(x + 1)/n"),
    ("(a x + a y)/z", "a*(x + y)/z"),
    ("(x y + x)/z", "x*(y + 1)/z"),
    ("(n^2 + n)/(x + 1)", "n*(n + 1)/(x + 1)"),
    ("(x^2 + x)/(2 y)", "x*(x + 1)/(2*y)"),
    # A common factor, not a factorization: these terms share nothing, and Derive
    # prints them as they are.
    ("(x^2 - 1)/(y + 1)", "(x^2 - 1)/(y + 1)"),
    ("(x^2 + 2 x + 1)/y", "(x^2 + 2*x + 1)/y"),
    # Without a quotient, a polynomial keeps its terms.
    ("x^2 + x", "x^2 + x"),
]


@pytest.mark.parametrize(("text", "expected"), NUMERATOR_CONTENT, ids=str)
def test_a_quotient_carries_its_numerator_s_common_factor_outside(text, expected):
    assert simp(text) == expected


#: A number common to a numerator and a denominator sum cancels, as the gcd over the
#: rationals leaves it (Derive), and so does a constant such as `pi`. A denominator sum
#: keeps whatever content doesn't cancel.
CANCELLED_CONTENT = [
    ("(2*x + 2)/(2*x^2 + 4)", "(x + 1)/(x^2 + 2)"),
    ("(2*x + 2)/(4*x^2 + 8)", "(x + 1)/(2*x^2 + 4)"),
    ("2/(2*x^2 + 4)", "1/(x^2 + 2)"),
    ("4*x/(2*x + 2)^2", "x/(x + 1)^2"),
    ("(3 - 4*x)/(2*x^2 + 2)", "(3 - 4*x)/(2*x^2 + 2)"),
    ("x/(2*x + 2)", "x/(2*x + 2)"),
    ("(pi - pi*ERF(z))/pi", "1 - ERF(z)"),
    ("(#e*x + #e)/#e", "x + 1"),
    ("(x + 1)/pi", "(x + 1)/pi"),
]


@pytest.mark.parametrize(("text", "expected"), CANCELLED_CONTENT, ids=str)
def test_a_number_common_to_numerator_and_denominator_cancels(text, expected):
    assert simp(text) == expected


def test_a_numeric_factor_stays_among_the_terms_it_divides():
    """`(2*x + 2)/y` rather than Derive's `2*(x + 1)/y`.

    Sympy distributes a number over a sum when building it, so the second form can't
    survive a print/parse round trip, and `test_simplify_corpus` fails when text and
    tree disagree. A factor containing a variable does survive, and is taken out.
    """
    assert simp("(2 x + 2)/y") == "(2*x + 2)/y"
    assert simp("(2 x y + 2 y)/z") == "2*y*(x + 1)/z"


#: The same rule from the other side: a polynomial's coefficients are rational, so a
#: number that divides into them goes among them rather than in front. One that
#: doesn't divide in is a denominator the whole sum shares, and stays in front.
COEFFICIENTS = [
    ("(m^2 + m + 2)*(m - 1)!/2", "(m^2/2 + m/2 + 1)*(m - 1)!"),
    ("(n^2 + n)/2", "n^2/2 + n/2"),
    ("x*(x - 1)/2", "x*(x - 1)/2"),
    ("z*(x + y)/2", "z*(x + y)/2"),
    # An integer factor in front is not a denominator and stays.
    ("6*a*b*(a*x + b*y)", "6*a*b*(a*x + b*y)"),
    # The reverse doesn't happen: terms with their own denominators are never
    # gathered over a common one. All three come from the original.
    ("x^2/6 + x/2", "x^2/6 + x/2"),
    ("x^10/2 - x^9/3 + x/6", "x^10/2 - x^9/3 + x/6"),
    ("1/2 + SIN(x)/2", "SIN(x)/2 + 1/2"),
]


@pytest.mark.parametrize(("text", "expected"), COEFFICIENTS, ids=str)
def test_a_number_that_divides_a_sum_is_written_into_it(text, expected):
    assert simp(text) == expected


def test_a_sum_over_two_powers_of_one_thing_is_written_over_the_higher():
    """The lower power divides the higher, so the sum becomes one ratio.

    All four match the original. This matters for derivatives: sympy differentiates
    a quotient into a sum over two powers of the denominator, where the original
    writes one ratio, whose numerator (`1` in the first case) is often the point.

    It applies only where it shortens the sum. `1/SQRT(x) + 1/x` over `x^(3/2)` is
    longer, and the original leaves it alone. The original also leaves the first
    case's sum alone if typed directly - the expression has two fixed points, and
    this test follows the original's answer for the question asked.
    """
    assert simp("DIF(x/SQRT(1 - x^2), x)") == "1/(1 - x^2)^(3/2)"
    assert simp("DIF(x^3/SQRT(x^2 + 1), x)") == "x^2*(2*x^2 + 3)/(x^2 + 1)^(3/2)"
    assert simp("1/(x + 1)^(3/2) + 1/SQRT(x + 1)") == "(x + 2)/(x + 1)^(3/2)"
    assert simp("1/SQRT(x) + 1/x") == "1/SQRT(x) + 1/x"


def test_no_denominator_carries_a_denominator_of_its_own():
    """A compound fraction is cleared however long that makes it.

    The original gives `a/(x^2 + a^2)` for `DIF(ATAN(x/a), x)`, sympy gives
    `1/(a + x^2/a)`. Both have the same operation count, so the rule can't be
    a count; the original just never writes nested denominators.
    """
    assert simp("DIF(ATAN(x/a), x)") == "a/(x^2 + a^2)"
    assert simp("1/(1 + 1/x)") == "x/(x + 1)"


def test_the_order_list_decides_what_gets_multiplied_out():
    """The manual's 4.3 exercise: the same entry expands under one order list and
    stays as written under another, because the list picks the primary variable."""
    assert simp("(x + 1)^9 + y", Context(order=("x", "y", "z"))).startswith("x^9")
    assert simp("(x + 1)^9 + y", Context(order=("y", "x", "z"))) == "y + (x + 1)^9"


# -- domains: the whole of "don't guess" --------------------------------------

DOMAINS = [
    # An undeclared variable is real, and a real square has a real root.
    ("SQRT(x^2)", "ABS(x)", None),
    ("SQRT(x^2)", "SQRT(x^2)", COMPLEX_X),
    ("SQRT(x^2)", "x", POSITIVE_X),
    ("SQRT(x^2)", "-x", declared("x :epsilon Real (-inf, 0)")),
    # A real `x` could be negative, and then `LN(x^2 - x)` is not
    # `LN(x) + LN(x - 1)`. Term order is the engine's.
    ("LN(x^2 - x) - LN(x)", "LN(x^2 - x) - LN(x)", None),
    ("LN(x^2 - x) - LN(x)", "LN(x - 1)", POSITIVE_X),
    ("LN(x^2)", "2*LN(ABS(x))", None),
    ("LN(x^2)", "2*LN(x)", POSITIVE_X),
    ("LN(x^3)", "LN(x^3)", None),
    ("LN(x^3)", "3*LN(x)", POSITIVE_X),
    ("SQRT(x)*SQRT(y)", "SQRT(x)*SQRT(y)", None),
    ("(-1)^(2*n)", "1", declared("n :epsilon Integer")),
    ("(-1)^(2*n)", "(-1)^(2*n)", None),
    # `pi` alone would be wrong where `m` is 0.
    (
        "INT(SIN(m*x)^2, x, 0, 2*pi)",
        "IF(m /= 0, pi, 0)",
        declared("m :epsilon Integer"),
    ),
    # 0 would be wrong where `m` is 2 or -2 (the integral is then `pi/2` or
    # `-pi/2`). Without a case split, the integral stays.
    (
        "INT(SIN(m*x)*SIN(2*x), x, 0, pi)",
        "INT(SIN(2*x)*SIN(m*x), x, 0, pi)",
        declared("m :epsilon Integer"),
    ),
    # The tangent pair sums to a right angle with the sign of its argument, which
    # covers both cases and so is not a guess. For complex arguments the identity
    # doesn't hold (`SIGN` is no longer `±1`), so the pair stays.
    ("ACOT(t) + ATAN(t)", "pi*SIGN(t)/2", None),
    ("ACOT(x) + ATAN(x)", "ATAN(x) + ACOT(x)", COMPLEX_X),
]

#: What a declared interval tells beyond its sign. A sympy symbol can only carry
#: assumptions like "positive", so these are questions only the bounds can answer.
INTERVALS = [
    ("ABS(x - 1)", "1 - x", "x :epsilon Real (0, 1)"),
    ("ABS(x - 1)", "1 - x", "x :epsilon Real [0, 1]"),
    ("ABS(x - 1)", "ABS(x - 1)", "x :epsilon Real (0, 2)"),
    ("ABS(x - 1)", "ABS(x - 1)", "x :epsilon Real (0, inf)"),
    ("SQRT((x - 1)^2)", "1 - x", "x :epsilon Real (0, 1)"),
    # Both bars come off, and then the sum cancels.
    ("ABS(x) + ABS(x - 1)", "1", "x :epsilon Real (0, 1)"),
    # `SIGN` needs a strict bound, since it has a third value at zero; an absolute
    # value is the same either way.
    ("SIGN(x - 1)", "-1", "x :epsilon Real (0, 1)"),
    ("SIGN(x - 1)", "SIGN(x - 1)", "x :epsilon Real (0, 1]"),
    ("ABS(x - 1)", "1 - x", "x :epsilon Real (0, 1]"),
    ("MAX(x, 1)", "1", "x :epsilon Real (0, 1)"),
    ("MIN(x, 1)", "x", "x :epsilon Real (0, 1)"),
    # Bounds are checked before `MAX` is rewritten as the closed form of 6.7, so a
    # bound that decides the larger argument returns it. Otherwise the formula is
    # left, its absolute value undecidable for the same reason.
    ("MAX(x, 1)", "ABS(x - 1)/2 + x/2 + 1/2", "x :epsilon Real (0, 2)"),
    # An integer variable is queried through a real stand-in, since sympy can't
    # reason about an integer bounded by a relation.
    ("ABS(n - 7)", "7 - n", "n :epsilon Integer [2, 5]"),
    ("ABS(n - 7)", "ABS(n - 7)", "n :epsilon Integer [2, 9]"),
    # An `IF` test and a bare relation are decided the same way.
    ("IF(x < 1, a, b)", "a", "x :epsilon Real (0, 1)"),
    ("IF(x > 1, a, b)", "b", "x :epsilon Real (0, 1)"),
    ("IF(x <= 1, a, b)", "a", "x :epsilon Real (0, 1]"),
    ("IF(x = 1, a, b)", "b", "x :epsilon Real (0, 1)"),
    ("IF(x /= 1, a, b)", "a", "x :epsilon Real (0, 1)"),
    ("IF(x < 1, a, b)", "IF(x < 1, a, b)", "x :epsilon Real (0, 2)"),
    ("x < 1", "true", "x :epsilon Real (0, 1)"),
    ("x < 1", "x < 1", "x :epsilon Real (0, 2)"),
    # Interval arithmetic bounds powers, absolute values and monotone functions of a
    # bounded quantity.
    ("ABS(x^2 - 1)", "1 - x^2", "x :epsilon Real (0, 1)"),
    ("ABS(x) < 1", "true", "x :epsilon Real (-1, 1)"),
    ("ABS(x) < 1", "ABS(x) < 1", "x :epsilon Real [-1, 1)"),
    ("ABS(x) <= 1", "true", "x :epsilon Real [-1, 1]"),
    ("LN(x) < 0", "true", "x :epsilon Real (0, 1)"),
    ("LN(x) < 0", "LN(x) < 0", "x :epsilon Real (0, 1]"),
    ("LN(x) <= 0", "true", "x :epsilon Real (0, 1]"),
    ("SQRT(x) < 1", "true", "x :epsilon Real (0, 1)"),
    ("#e^x < 3", "true", "x :epsilon Real (-1, 1)"),
    ("#e^x < 2", "#e^x < 2", "x :epsilon Real (-1, 1)"),
    ("1/x > 1", "true", "x :epsilon Real (0, 1)"),
    ("ATAN(x) > pi/4", "true", "x :epsilon Real (1, inf)"),
    # Interval arithmetic treats each occurrence of a variable independently, so it
    # misses bounds that depend on correlation: `x - x^2` is positive over `(0, 1)`,
    # but stays unresolved.
    ("ABS(x - x^2)", "ABS(x^2 - x)", "x :epsilon Real (0, 1)"),
    # A step function is constant between consecutive integers.
    ("FLOOR(x)", "0", "x :epsilon Real (0, 1)"),
    ("FLOOR(x)", "0", "x :epsilon Real [0, 1)"),
    ("FLOOR(x)", "FLOOR(x)", "x :epsilon Real (0, 1]"),
    ("FLOOR(x)", "FLOOR(x)", "x :epsilon Real (-1, 1)"),
    ("FLOOR(x + 3/2)", "1", "x :epsilon Real (-1/2, 1/2)"),
    ("FLOOR(-x)", "-1", "x :epsilon Real (0, 1)"),
    ("FLOOR(LN(x))", "1", "x :epsilon Real (3, 7)"),
    # An inverse function undoes its function when the angle stays within one
    # branch, shifted onto the principal branch.
    ("ASIN(SIN(x))", "x", "x :epsilon Real (-1, 1)"),
    ("ASIN(SIN(x))", "x", "x :epsilon Real [-pi/2, pi/2]"),
    ("ASIN(SIN(x))", "pi - x", "x :epsilon Real (2, 4)"),
    ("ASIN(SIN(x))", "ASIN(SIN(x))", "x :epsilon Real (1, 2)"),
    ("ACOS(COS(x))", "x", "x :epsilon Real (0, 3)"),
    ("ACOS(COS(x))", "-x", "x :epsilon Real (-3, 0)"),
    ("ACOS(COS(x))", "2*pi - x", "x :epsilon Real (4, 6)"),
    ("ACOS(COS(x))", "ACOS(COS(x))", "x :epsilon Real (-1, 1)"),
    ("ATAN(TAN(x))", "x", "x :epsilon Real (-1, 1)"),
    ("ATAN(TAN(x))", "x - pi", "x :epsilon Real (2, 4)"),
    ("ATAN(TAN(x))", "ATAN(TAN(x))", "x :epsilon Real (1, 2)"),
]


@pytest.mark.parametrize(("text", "expected", "declaration"), INTERVALS, ids=str)
def test_an_interval_settles_what_a_sign_alone_cannot(text, expected, declaration):
    assert simp(text, declared(declaration)) == expected


def test_two_declared_intervals_are_asked_together():
    # Both bounds are used together, answering questions about the pair that
    # neither answers alone.
    box = declared("x :epsilon Real (0, 1)", "y :epsilon Real (2, 3)")
    assert simp("ABS(x - y)", box) == "y - x"
    assert simp("MAX(x, y)", box) == "y"
    assert simp("IF(x < y, a, b)", box) == "a"
    assert simp("x*y < 3", box) == "true"
    assert simp("x*y < 2", box) == "x*y < 2"


def test_nothing_is_settled_where_nothing_is_declared_or_bounded():
    # Without a declaration there is only the whole real line to go by, which
    # settles nothing about an unbounded argument.
    assert simp("ABS(x - 1)") == "ABS(x - 1)"
    assert simp("MAX(x, 1)") == "ABS(x - 1)/2 + x/2 + 1/2"
    assert simp("IF(x < 1, a, b)") == "IF(x < 1, a, b)"


BOUNDED = [
    # A sum bounded by nature keeps its sign for any real variable.
    ("ABS(1 + COS(x))", "COS(x) + 1"),
    ("ABS(COS(x) - 1)", "1 - COS(x)"),
    ("ABS(COS(x) - 2)", "2 - COS(x)"),
    ("ABS(SIN(x) + COS(x) - 2)", "-COS(x) - SIN(x) + 2"),
    ("ABS(1 + SIN(x)*COS(y))", "SIN(x)*COS(y) + 1"),
    ("ABS(ATAN(x) - 2)", "2 - ATAN(x)"),
    ("ABS(pi/2 - ATAN(x))", "pi/2 - ATAN(x)"),
    ("ABS(COSH(x) - 1)", "#e^x/2 + #e^(-x)/2 - 1"),
    ("SQRT((1 + COS(x))^2)", "COS(x) + 1"),
    ("MAX(COS(x), 2)", "2"),
    ("MIN(SIN(x), -1)", "-1"),
    ("SIGN(2 + COS(x))", "1"),
    # `SIGN` needs zero excluded, and `1 + COS(x)` reaches it.
    ("SIGN(1 + COS(x))", "SIGN(COS(x) + 1)"),
    # Not bounded on one side of zero.
    ("ABS(COS(x) + 1/2)", "ABS(COS(x) + 1/2)"),
    ("ABS(x*ATAN(x) + 1)", "ABS(x*ATAN(x) + 1)"),
    # A polynomial keeps its sign if its factors of odd multiplicity have no real
    # root. A square can be zero, so it settles `ABS` but not `SIGN`.
    ("ABS(x^2 + x + 1)", "x^2 + x + 1"),
    ("ABS(-x^2 + 2*x - 3)", "x^2 - 2*x + 3"),
    ("ABS(x^2 - 2*x + 1)", "x^2 - 2*x + 1"),
    ("ABS(x^2 - 2*x*y + y^2)", "x^2 - 2*x*y + y^2"),
    ("SIGN(x^2 + x + 1)", "1"),
    ("SIGN(x^2 - 2*x + 1)", "SIGN(x^2 - 2*x + 1)"),
    ("MAX(x^2 + 1, x)", "x^2 + 1"),
    ("ABS(x^2 + x - 1)", "ABS(x^2 + x - 1)"),
    ("ABS(x^3 + 1)", "ABS(x^3 + 1)"),
]


@pytest.mark.parametrize(("text", "expected"), BOUNDED, ids=str)
def test_a_sign_kept_over_the_whole_real_line_needs_no_declaration(text, expected):
    assert simp(text) == expected


def test_a_complex_variable_has_no_range():
    context = declared("z :epsilon Complex")
    assert simp("ABS(1 + COS(z))", context) == "ABS(COS(z) + 1)"
    assert simp("ABS(z^2 + z + 1)", context) == "ABS(z^2 + z + 1)"


@pytest.mark.parametrize(("text", "expected", "context"), DOMAINS, ids=str)
def test_a_rewrite_needs_a_domain_that_justifies_it(text, expected, context):
    assert simp(text, context) == expected


def test_a_one_point_interval_is_a_value():
    assert simp("x + 1", declared("x :epsilon Real [7, 7]")) == "8"


#: Conditional answers whose condition a declaration decides. The chosen branch is
#: returned; if it is the else branch, that is worked out.
DECIDED_BY_A_DECLARATION = [
    ("SUM(x^k, k, 0, inf)", "1/(1 - x)", "x :epsilon Real (-1, 1)"),
    ("SUM(x^k, k, 0, inf)", "1/(1 - x)", "x :epsilon Real (0, 1)"),
    ("SUM(k*x^k, k, 0, inf)", "x/(1 - x)^2", "x :epsilon Real (0, 1)"),
    ("SUM(x^k/k, k, 1, inf)", "-LN(1 - x)", "x :epsilon Real (-1, 1)"),
    # Past one the series takes the else branch and diverges: to `inf` for
    # positive terms and `-inf` for negative.
    ("SUM(x^k, k, 0, inf)", "inf", "x :epsilon Real (1, inf)"),
    ("SUM(x^k, k, 0, inf)", "inf", "x :epsilon Real [2, inf)"),
    ("SUM(k*x^k, k, 0, inf)", "inf", "x :epsilon Real (1, inf)"),
    ("SUM(-x^k, k, 0, inf)", "-inf", "x :epsilon Real (1, inf)"),
    # With alternating signs the sum has no limit to report.
    (
        "SUM((-x)^k, k, 0, inf)",
        "SUM(x^k*(-1)^k, k, 0, inf)",
        "x :epsilon Real (1, inf)",
    ),
    # A declaration that doesn't decide the condition leaves it.
    (
        "SUM(x^k, k, 0, inf)",
        "IF(x < 1, 1/(1 - x), SUM(x^k, k, 0, inf))",
        "x :epsilon Real (0, 2)",
    ),
    # A power of a declared base tending to zero or infinity.
    ("LIM(x^t, t, inf)", "0", "x :epsilon Real (0, 1)"),
    ("LIM(x^t, t, inf)", "inf", "x :epsilon Real (1, inf)"),
    ("LIM(x^t, t, -inf)", "0", "x :epsilon Real (1, inf)"),
    ("LIM(x^t, t, -inf)", "inf", "x :epsilon Real (0, 1)"),
    # A user-written test is decided the same way.
    ("IF(ABS(x) < 1, a, b)", "a", "x :epsilon Real (-1, 1)"),
    ("IF(LN(x) < 0, a, b)", "a", "x :epsilon Real (0, 1)"),
    ("IF(x > 1 OR x < -1, a, b)", "b", "x :epsilon Real (-1, 1)"),
    ("IF(x < 1 AND x > 0, a, b)", "a", "x :epsilon Real (0, 1)"),
]


@pytest.mark.parametrize(
    ("text", "expected", "declaration"), DECIDED_BY_A_DECLARATION, ids=str
)
def test_a_declaration_decides_the_conditional_an_answer_comes_with(
    text, expected, declaration
):
    assert simp(text, declared(declaration)) == expected


def test_a_conditional_on_a_condition_already_asked_asks_it_once():
    # The else branch for `y*x^k` is the sum of `x^k`, whose own conditional has
    # the same condition; it fails wherever the first does, so it isn't repeated.
    assert simp("SUM(y*x^k, k, 0, inf)") == (
        "y*IF(ABS(x) < 1, 1/(1 - x), SUM(x^k, k, 0, inf))"
    )


def test_a_recursion_guarded_by_a_declared_variable_stands():
    """A recursion whose test the declaration leaves open is not unfolded."""
    body = parse("IF(n = 0, 1, n*F(n - 1))", declaring("F"))
    functions = {"F": (("n",), body)}
    assert simp("F(5)", Context(functions=functions)) == "120"
    context = replace(declared("m :epsilon Integer [0, 3]"), functions=functions)
    assert simp("F(m)", context) == "IF(m = 0, 1, m*F(m - 1))"
    # Above two, the first tests are false, so the recursion unfolds up to the
    # first undecided one.
    context = replace(declared("m :epsilon Integer (2, inf)"), functions=functions)
    assert simp("F(m)", context) == "m*(m - 1)*(m - 2)*IF(m = 3, 1, (m - 3)*F(m - 4))"


#: A complex quantity times its conjugate is its squared modulus.
CONJUGATE_PAIRS = [
    ("CONJ(z)*z", "ABS(z)^2"),
    ("z*CONJ(z) + 1", "ABS(z)^2 + 1"),
    ("CONJ(z)^2*z", "CONJ(z)*ABS(z)^2"),
    ("CONJ(z)*z^3", "z^2*ABS(z)^2"),
    ("1/(z*CONJ(z))", "1/ABS(z)^2"),
    # A quotient doesn't pair.
    ("CONJ(z)/z", "CONJ(z)/z"),
]


@pytest.mark.parametrize(("text", "expected"), CONJUGATE_PAIRS, ids=str)
def test_a_conjugate_pair_is_a_squared_modulus(text, expected):
    assert simp(text, declared("z :epsilon Complex")) == expected


# -- absolute values and signs ------------------------------------------------

#: `ABS(u)`, `SIGN(u)` and `u` are related, so a product of two of them reduces to
#: whichever single form is shortest.
SIGNED = [
    ("ABS(x)*SIGN(x)", "x", None),
    ("ABS(x)/SIGN(x)", "x", None),
    ("x*SIGN(x)", "ABS(x)", None),
    ("x/ABS(x)", "SIGN(x)", None),
    ("ABS(x)/x", "SIGN(x)", None),
    ("SIGN(x)^2", "1", None),
    ("SIGN(x)^3", "SIGN(x)", None),
    ("1/SIGN(x)", "SIGN(x)", None),
    ("ABS(x)^2", "x^2", None),
    # The argument can be any expression, and other factors are left alone.
    ("ABS(SIN(x))*SIGN(SIN(x))", "SIN(x)", None),
    ("ABS(x - 1)/SIGN(x - 1)", "x - 1", None),
    ("y*ABS(x)*SIGN(x)", "x*y", None),
    ("ABS(x)*SIGN(y)", "ABS(x)*SIGN(y)", None),
    # A single one of the three, or a bare power, stays.
    ("ABS(x)", "ABS(x)", None),
    ("SIGN(x)", "SIGN(x)", None),
    ("x^3", "x^3", None),
    # A square is never negative but can be zero, which the sign distinguishes;
    # `SIGN(ABS(x))` would be no simpler.
    ("SIGN(x^2)", "SIGN(x^2)", None),
    ("SIGN(x^2 + 1)", "1", None),
    # A reciprocal has the sign of its denominator where defined. It is undefined
    # at zero, so `SIGN(1/x^2)` is 1 where `SIGN(x^2)` is not.
    ("SIGN(1/x)", "SIGN(x)", None),
    ("SIGN(x^(-3))", "SIGN(x)", None),
    ("SIGN(1/x^2)", "1", None),
    ("SIGN(1/ABS(x))", "1", None),
    ("SIGN(1/(x + y))", "SIGN(x + y)", None),
    ("SIGN(1/z)", "SIGN(1/z)", declared("z :epsilon Complex")),
    # At zero the identities above are only conventions (`SIGN(0)` is zero, so one
    # side is undefined there); a declaration excluding zero makes them exact.
    ("ABS(x)*SIGN(x)", "x", POSITIVE_X),
    ("SIGN(x)^2", "1", POSITIVE_X),
    ("x/ABS(x)", "1", POSITIVE_X),
    # For complex `z` nothing folds: `SIGN(z)` is `z/|z|`, whose square is not one,
    # and sympy won't collect its product with `ABS(z)` either.
    ("ABS(z)*SIGN(z)", "SIGN(z)*ABS(z)", declared("z :epsilon Complex")),
    ("ABS(z)/SIGN(z)", "ABS(z)/SIGN(z)", declared("z :epsilon Complex")),
    ("SIGN(z)^2", "SIGN(z)^2", declared("z :epsilon Complex")),
    ("z/ABS(z)", "z/ABS(z)", declared("z :epsilon Complex")),
    # An odd increasing function keeps the sign of a real argument, and an even
    # function ignores its argument's absolute value. Neither holds for complex ones.
    ("SIGN(ATAN(x))", "SIGN(x)", None),
    ("SIGN(SINH(x))", "SIGN(x)", None),
    ("SIGN(x^3)", "SIGN(x)", None),
    ("SIGN(ATAN(x)^3)", "SIGN(x)", None),
    ("SIGN(x^2)", "SIGN(x^2)", None),
    # A logarithm is increasing and crosses zero at 1, which gives its sign where its
    # argument can't be negative. Elsewhere the logarithm may be complex.
    ("SIGN(LN(1 + COS(x)))", "SIGN(COS(x))", None),
    ("SIGN(LN(1 + SIN(x)))", "SIGN(SIN(x))", None),
    ("SIGN(LN(1 - COS(x)))", "-SIGN(COS(x))", None),
    ("SIGN(LN(2 + COS(x)))", "SIGN(COS(x) + 1)", None),
    ("SIGN(LN(ABS(x)))", "SIGN(ABS(x) - 1)", None),
    ("SIGN(LN(x^2))", "SIGN(x^2 - 1)", None),
    ("SIGN(LN(x^2 + 2))", "1", None),
    ("SIGN(LN(x))", "SIGN(x - 1)", POSITIVE_X),
    ("SIGN(LN(x))", "SIGN(LN(x))", None),
    ("SIGN(LN(x + 1))", "SIGN(LN(x + 1))", None),
    ("SIGN(LN(COS(x)))", "SIGN(LN(COS(x)))", None),
    ("SIGN(LN(COS(x) + 1/2))", "SIGN(LN(COS(x) + 1/2))", None),
    # An exponential crosses 1 at zero.
    ("SIGN(#e^x - 1)", "SIGN(x)", None),
    ("SIGN(1 - #e^x)", "-SIGN(x)", None),
    ("SIGN(#e^z - 1)", "SIGN(#e^z - 1)", declared("z :epsilon Complex")),
    # Factors with the same sign need no bars between them, in one `ABS` or several.
    # The odd one out and unrelated factors keep theirs.
    ("ABS(x*ATAN(x))", "x*ATAN(x)", None),
    ("ABS(x^3*ATAN(x))", "x^3*ATAN(x)", None),
    ("ABS(x*ATAN(x^3))", "x*ATAN(x^3)", None),
    ("ABS(ATAN(x)/x)", "ATAN(x)/x", None),
    ("ABS(x)*ABS(ATAN(x))", "x*ATAN(x)", None),
    ("SQRT(x^2*ATAN(x)^2)", "x*ATAN(x)", None),
    ("ABS(x*ABS(x))", "x^2", None),
    ("ABS(x*ERF(x))", "x*ERF(x)", None),
    ("ABS(x*SINH(x))", "x*(#e^x - #e^(-x))/2", None),
    ("ABS(x*(#e^x - 1))", "x*(#e^x - 1)", None),
    ("ABS(x*(1 - #e^x))", "x*(#e^x - 1)", None),
    ("ABS(x*y*ATAN(x))", "x*ATAN(x)*ABS(y)", None),
    ("ABS(x*SIN(x))*ABS(SIN(x))", "SIN(x)^2*ABS(x)", None),
    ("ABS(x*ATAN(x)*ERF(x))", "ERF(x)*ATAN(x)*ABS(x)", None),
    # A factor that can't be negative comes out too.
    ("ABS(x*(1 + COS(x)))", "(COS(x) + 1)*ABS(x)", None),
    ("ABS(x*ATAN(x)*(1 + COS(x)))", "x*(COS(x) + 1)*ATAN(x)", None),
    # Unrelated signs stay together, and `ASIN` is complex off `[-1, 1]`.
    ("ABS(x*ATAN(y))", "ABS(x*ATAN(y))", None),
    ("ABS(x*SIN(x))", "ABS(x*SIN(x))", None),
    ("ABS(x*ASIN(x))", "ABS(x*ASIN(x))", None),
    ("ABS(ATAN(x))", "ABS(ATAN(x))", None),
    ("ABS(z*ATAN(z))", "ABS(z*ATAN(z))", declared("z :epsilon Complex")),
    ("COS(ABS(x))", "COS(x)", None),
    ("SIGN(ATAN(z))", "SIGN(ATAN(z))", declared("z :epsilon Complex")),
    ("COS(ABS(z))", "COS(ABS(z))", declared("z :epsilon Complex")),
]


@pytest.mark.parametrize(("text", "expected", "context"), SIGNED, ids=str)
def test_an_absolute_value_and_a_sign_are_one_quantity(text, expected, context):
    assert simp(text, context) == expected


# -- trigonometry -------------------------------------------------------------

TRIGONOMETRY = [
    ("SIN(pi/4)", "SQRT(2)/2", None),
    ("SIN(x)^2 + COS(x)^2", "1", None),
    ("TAN(x)*COT(x)", "1", None),
    ("SIN(x)^3 + SIN(x) + COS(x)^2*SIN(x)", "2*SIN(x)", None),
    ("SIN(6*x)/SIN(3*x)", "2*COS(3*x)", None),
    ("ATAN(2 + SQRT(3))", "5*pi/12", None),
    ("ATAN(1, 1)", "pi/4", None),
    # Complementary arcs of one argument sum to a right angle. For the tangent pair
    # the sign of the argument matters, so the answer has `SIGN` unless a
    # declaration or a nonnegative argument decides it.
    ("ASIN(x) + ACOS(x)", "pi/2", None),
    ("ASEC(x) + ACSC(x)", "pi/2", None),
    ("2*ASIN(x) + 2*ACOS(x)", "pi", None),
    ("ASIN(x) + ACOS(x) + y", "y + pi/2", None),
    ("ASIN(x) + ACOS(x)", "90", Context(angle=Angle.DEGREE)),
    ("ACOT(t) + ATAN(t)", "pi*SIGN(t)/2", None),
    ("ATAN(t) + ACOT(t) + y", "pi*SIGN(t)/2 + y", None),
    ("2*ACOT(t) + 2*ATAN(t)", "pi*SIGN(t)", None),
    ("ACOT(t) + ATAN(t)", "90*SIGN(t)", Context(angle=Angle.DEGREE)),
    ("ACOT(t) + ATAN(t)", "pi/2", declared("t :epsilon Real (0, inf)")),
    ("ACOT(t) + ATAN(t)", "-pi/2", declared("t :epsilon Real (-inf, 0)")),
    # `ACOT(0)` is `pi/2`, so for a square argument the sum is `pi/2` everywhere.
    ("ACOT(x^2) + ATAN(x^2)", "pi/2", None),
    ("ACOT(x^2 + 1) + ATAN(x^2 + 1)", "pi/2", None),
    ("ACOT(3) + ATAN(3)", "pi/2", None),
    ("ASIN(x) + ACOS(y)", "ASIN(x) + ACOS(y)", None),
    # An arc of a side ratio of a right triangle becomes a different arc, however
    # the hypotenuse is written. The root has a branch cut, so a complex argument
    # stays as it is.
    ("ASIN(x/SQRT(x^2 + 1))", "ATAN(x)", None),
    ("ASIN(-x/SQRT(x^2 + 1))", "-ATAN(x)", None),
    ("ASIN((x + 1)/SQRT(x^2 + 2*x + 2))", "ATAN(x + 1)", None),
    ("ATAN(x/SQRT(1 - x^2))", "ASIN(x)", None),
    ("ASIN(x/SQRT(x^2 + 1))", "ASIN(x/SQRT(x^2 + 1))", COMPLEX_X),
    ("ATAN(x/SQRT(1 - x^2))", "ATAN(x/SQRT(1 - x^2))", COMPLEX_X),
    # The reciprocal arcs are defined (6.4) as `ASEC(z) = ACOS(1/z)` and
    # `ACSC(z) = ASIN(1/z)` for any `z`, so need no domain. Neither appears in
    # answers.
    ("ASEC(1/x)", "ACOS(x)", None),
    ("ACSC(1/x)", "ASIN(x)", None),
    ("ASEC(1/x)", "ACOS(x)", COMPLEX_X),
    ("ASEC(x)", "ACOS(1/x)", None),
    # Angles related by a fractional factor cancel once written in terms of the
    # common angle. (An integer factor is `SIN(6*x)/SIN(3*x)` above.)
    ("1/(1 + TAN(a)*TAN(a/2))", "COS(a)", None),
    ("SIN(45)", "SQRT(2)/2", Context(angle=Angle.DEGREE)),
    ("ASIN(1/2)", "30", Context(angle=Angle.DEGREE)),
    # In degree mode answers measure angles in degrees, like the input, so they read
    # back as themselves. The derivative's `pi/180` comes from the chain rule, and
    # the cosine's argument is still `x` degrees.
    ("SIN(2*x)", "SIN(2*x)", Context(angle=Angle.DEGREE)),
    ("SIN(x + y)", "SIN(x + y)", Context(angle=Angle.DEGREE)),
    ("ASIN(x)", "ASIN(x)", Context(angle=Angle.DEGREE)),
    ("ATAN(y, x)", "ATAN(y, x)", Context(angle=Angle.DEGREE)),
    ("DIF(SIN(x), x)", "pi*COS(x)/180", Context(angle=Angle.DEGREE)),
    ("pi*COS(x)/180", "pi*COS(x)/180", Context(angle=Angle.DEGREE)),
    ("INT(SIN(x), x)", "-180*COS(x)/pi", Context(angle=Angle.DEGREE)),
    ("INT(1/(1 + x^2), x)", "pi*ATAN(x)/180", Context(angle=Angle.DEGREE)),
    ("ASIN(SIN(x))", "ASIN(SIN(x))", Context(angle=Angle.DEGREE)),
    ("APPROX(SIN(x) + ASIN(y))", "SIN(x) + ASIN(y)", Context(angle=Angle.DEGREE)),
    ("APPROX(2*ASIN(y))", "2*ASIN(y)", Context(angle=Angle.DEGREE)),
    # A phase is an angle, in the same unit as the inverse trig functions.
    ("PHASE(#i)", "90", Context(angle=Angle.DEGREE)),
    ("PHASE(-1 - #i)", "-135", Context(angle=Angle.DEGREE)),
    # `deg` is pi/180, so degrees work in radian mode too. In degree mode it is 1.
    ("SIN(45 deg)", "SQRT(2)/2", None),
    ("SIN(45 deg)", "SQRT(2)/2", Context(angle=Angle.DEGREE)),
    ("45 deg", "45", Context(angle=Angle.DEGREE)),
    ("SIN(2*x)", "2*SIN(x)*COS(x)", Context(trigonometry=Direction.EXPAND)),
    ("SIN(x)*COS(x)", "SIN(2*x)/2", Context(trigonometry=Direction.COLLECT)),
    ("COS(x)^2", "1 - SIN(x)^2", Context(trigpower=TrigPower.SINES)),
    ("SIN(x)^2", "1 - COS(x)^2", Context(trigpower=TrigPower.COSINES)),
    # The original has no secant, and leaves a reciprocal cosine power as typed.
    ("1/COS(x)^2", "1/COS(x)^2", None),
    ("1/COS(x)^4", "1/COS(x)^4", None),
    ("1/(4*COS(x)^4)", "1/(4*COS(x)^4)", None),
    ("COS(x)^-6", "1/COS(x)^6", None),
    ("SEC(x)^2", "1/COS(x)^2", None),
    ("1/COS(x)^3", "1/COS(x)^3", None),
    ("SIN(x)/COS(x)^2", "SIN(x)/COS(x)^2", None),
]


@pytest.mark.parametrize(("text", "expected", "context"), TRIGONOMETRY, ids=str)
def test_trigonometry(text, expected, context):
    assert simp(text, context) == expected


def test_a_sum_that_is_zero_by_the_pythagorean_identity_is_zero_however_large():
    """TENSOR1.DMO: a vector's length is invariant under a change of coordinates.

    `V` is transformed to spherical coordinates both as a covariant vector (by the
    transposed Jacobian) and as a contravariant one (by its inverse), and the product of
    the two is the length of `V` again. `trigsimp` leaves
    `c^2*(COS(t)^2 + SIN(t)^2 - 1)` and other terms; Derive gives 0.
    """
    jacobian = (
        "[[SIN(t)*COS(p), r*COS(t)*COS(p), -r*SIN(t)*SIN(p)],"
        " [SIN(t)*SIN(p), r*COS(t)*SIN(p), r*SIN(t)*COS(p)], [COS(t), -r*SIN(t), 0]]"
    )
    inverse = (
        "[[SIN(t)*COS(p), SIN(t)*SIN(p), COS(t)],"
        " [COS(t)*COS(p)/r, COS(t)*SIN(p)/r, -SIN(t)/r],"
        " [-SIN(p)/(r*SIN(t)), COS(p)/(r*SIN(t)), 0]]"
    )
    v = "[a*r*SIN(t)*COS(p), b, c]"
    assert simp(f"({jacobian}` . {v}) . ({inverse} . {v}) - {v} . {v}") == "0"


def test_a_hyperbolic_function_of_a_trigonometric_one_keeps_the_latter():
    """Only the hyperbolic call is written over exponentials, not its argument."""
    assert simp("COSH(1/COS(x))") == "(#e^(2/COS(x)) + 1)*#e^(-1/COS(x))/2"
    assert simp("COSH(TAN(x))") == "#e^TAN(x)/2 + #e^(-TAN(x))/2"
    assert simp("3*SECH(SEC(x))") == "6*#e^(1/COS(x))/(#e^(2/COS(x)) + 1)"
    assert simp("STEP(x)") == "SIGN(x)/2 + 1/2"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SIN(x) + COS(x)", "SQRT(2)*SIN(x + pi/4)"),
        ("3*SIN(x) + 4*COS(x)", "5*SIN(x + ATAN(4/3))"),
    ],
    ids=str,
)
def test_collect_writes_a_sine_and_a_cosine_as_one_phase_shifted_sine(text, expected):
    # Our own rewrite, since fu's `TR10i` only collects equal coefficients.
    assert simp(text, Context(trigonometry=Direction.COLLECT)) == expected


#: Arctangents of numbers are summed via Gaussian integers: `ATAN(p/q)` is the
#: argument of `q + p*#i`, and a sum is a rational multiple of pi exactly when the
#: product lies on an axis or a diagonal.
ARC_TANGENT_SUMS = [
    # `ATAN(1)` is `pi/4`, and the other two make up the rest.
    ("ATAN(1) + ATAN(2) + ATAN(3)", "pi", None),
    ("ATAN(2) + ATAN(3)", "3*pi/4", None),
    ("ATAN(1/2) + ATAN(1/3)", "pi/4", None),
    ("ATAN(-2) + ATAN(-3)", "-3*pi/4", None),
    ("ATAN(2) - ATAN(1/3)", "pi/4", None),
    # Machin's formula and its relatives, multiples and all.
    ("4*ATAN(1/5) - ATAN(1/239)", "pi/4", None),
    ("ATAN(1/2) + ATAN(1/5) + ATAN(1/8)", "pi/4", None),
    ("2*ATAN(1/3) + ATAN(1/7)", "pi/4", None),
    ("5*ATAN(1/7) + 2*ATAN(3/79)", "pi/4", None),
    ("2*ATAN(2) + ATAN(4/3)", "pi", None),
    # The part that sums to a multiple of pi is reduced, the rest stays.
    ("ATAN(2) + ATAN(3) + ATAN(1/7)", "3*pi/4 + ATAN(1/7)", None),
    ("ATAN(1/2) + ATAN(1/3) + x", "x + pi/4", None),
    ("y*ATAN(1/2) + y*ATAN(1/3)", "pi*y/4", None),
    ("ATAN(1/2) + ATAN(1/3)", "45", Context(angle=Angle.DEGREE)),
    # Otherwise a sum becomes one arctangent only if its argument is no larger than
    # the ones it replaces.
    ("ATAN(1/3) + ATAN(1/7)", "ATAN(1/2)", None),
    ("ATAN(1/2) + ATAN(1/4)", "ATAN(1/2) + ATAN(1/4)", None),
    ("ATAN(2) + ATAN(4)", "ATAN(4) + ATAN(2)", None),
    ("3*ATAN(1/5)", "3*ATAN(1/5)", None),
    # `ATAN(1/u)` is `ACOT(u)` for real `u`, and pairs with `ATAN(u)` the same way.
    ("ATAN(x) + ATAN(1/x)", "pi*SIGN(x)/2", None),
    ("ATAN(x/y) + ATAN(y/x)", "pi*SIGN(x/y)/2", None),
    ("2*ATAN(x) + 2*ATAN(1/x)", "pi*SIGN(x)", None),
    ("ATAN(x^2) + ATAN(1/x^2)", "pi/2", None),
    ("ATAN(z) + ATAN(1/z)", "ATAN(1/z) + ATAN(z)", declared("z :epsilon Complex")),
]


@pytest.mark.parametrize(("text", "expected", "context"), ARC_TANGENT_SUMS, ids=str)
def test_arctangents_of_numbers_add_up(text, expected, context):
    assert simp(text, context) == expected


#: Sines and cosines of integer multiples of one angle are reduced together and
#: written in the shorter of two canonical forms: a sum of functions of multiples, or
#: a polynomial in one of the two functions.
MULTIPLE_ANGLES = [
    # `trigsimp` rewrites only the last term and leaves the sum half done.
    ("SIN(x)^6 + COS(x)^6 + 3*SIN(x)^2*COS(x)^2", "1"),
    ("SIN(x)^4 - COS(x)^4", "-COS(2*x)"),
    ("COS(2*x) + 2*SIN(x)^2", "1"),
    ("SIN(3*x) + 4*SIN(x)^3", "3*SIN(x)"),
    ("SIN(2*x)*COS(x) - COS(2*x)*SIN(x)", "SIN(x)"),
    # A quotient that cancels to a polynomial is cancelled even if that is no
    # shorter, and becomes a sum of multiples.
    ("SIN(3*x)/SIN(x)", "2*COS(2*x) + 1"),
    ("COS(3*x)/COS(x)", "2*COS(2*x) - 1"),
    ("SIN(5*x)/SIN(x)", "2*COS(2*x) + 2*COS(4*x) + 1"),
    ("(COS(x) + COS(3*x))/COS(2*x)", "2*COS(x)"),
    ("SIN(3*x)/SIN(x) + 1", "4*COS(x)^2"),
    # Nothing cancels in these, and there is no shorter form.
    ("SIN(x)^4 + COS(x)^4", "COS(x)^4 + SIN(x)^4"),
    ("SIN(x)*COS(2*x)", "SIN(x)*COS(2*x)"),
    ("SIN(3*x)/SIN(y)", "SIN(3*x)/SIN(y)"),
]


@pytest.mark.parametrize(("text", "expected"), MULTIPLE_ANGLES, ids=str)
def test_calls_of_multiples_of_one_angle_reduce_together(text, expected):
    assert simp(text) == expected


#: A trig function of a small multiple of an arc becomes algebraic via the
#: multiple-angle formulas.
ARC_MULTIPLES = [
    ("SIN(2*ACOS(x))", "2*x*SQRT(1 - x^2)"),
    ("SIN(2*ASIN(x))", "2*x*SQRT(1 - x^2)"),
    ("COS(2*ASIN(x))", "1 - 2*x^2"),
    ("COS(3*ACOS(x))", "4*x^3 - 3*x"),
    ("SIN(2*ATAN(x))", "2*x/(x^2 + 1)"),
    ("TAN(2*ATAN(x))", "2*x/(1 - x^2)"),
    ("SIN(-2*ACOS(x))", "-2*x*SQRT(1 - x^2)"),
    # Above the multiple limit the call stays.
    ("SIN(5*ACOS(x))", "SIN(5*ACOS(x))"),
]


@pytest.mark.parametrize(("text", "expected"), ARC_MULTIPLES, ids=str)
def test_a_call_of_a_multiple_of_an_arc_is_algebraic(text, expected):
    assert simp(text) == expected


#: Sums and products of trig values at rational multiples of pi that are rational or
#: quadratic, as their minimal polynomial proves.
TRIGONOMETRIC_CONSTANTS = [
    ("COS(pi/7) - COS(2*pi/7) + COS(3*pi/7)", "1/2"),
    ("COS(2*pi/7) + COS(4*pi/7) + COS(6*pi/7)", "-1/2"),
    ("COS(pi/9)*COS(2*pi/9)*COS(4*pi/9)", "1/8"),
    ("COS(pi/7)*COS(2*pi/7)*COS(3*pi/7)", "1/8"),
    ("SIN(pi/7)*SIN(2*pi/7)*SIN(3*pi/7)", "SQRT(7)/8"),
    ("TAN(pi/7)*TAN(2*pi/7)*TAN(3*pi/7)", "SQRT(7)"),
    ("TAN(pi/9)*TAN(2*pi/9)*TAN(4*pi/9)", "SQRT(3)"),
    ("COS(pi/11) + COS(3*pi/11) + COS(5*pi/11) + COS(7*pi/11) + COS(9*pi/11)", "1/2"),
    # Alongside a variable, in a sum or a product.
    ("x + COS(pi/9)*COS(2*pi/9)*COS(4*pi/9)", "x + 1/8"),
    ("x*COS(pi/9)*COS(2*pi/9)*COS(4*pi/9)", "x/8"),
    # A value of higher degree is no simpler, and stays.
    ("SIN(pi/7) + SIN(2*pi/7)", "COS(3*pi/14) + SIN(pi/7)"),
    ("2*COS(pi/7)", "2*COS(pi/7)"),
    ("SIN(1)*COS(2)", "SIN(1)*COS(2)"),
]


@pytest.mark.parametrize(("text", "expected"), TRIGONOMETRIC_CONSTANTS, ids=str)
def test_a_trigonometric_constant_is_written_as_the_number_it_is(text, expected):
    assert simp(text) == expected


#: A sine or cosine at a rational multiple of pi past the first octant is the cofunction
#: of the complement, and a denominator with many factors of two isn't expanded into
#: nested roots. RANDOM_NORMAL in MISC.MTH takes such cosines of `2*pi*RANDOM(1)`.
OCTANT_ANGLES = [
    ("COS(pi*204285811/536870912)", "SIN(64149645*pi/536870912)"),
    ("SIN(pi*204285811/536870912)", "COS(64149645*pi/536870912)"),
    ("TAN(pi*204285811/536870912)", "TAN(204285811*pi/536870912)"),
    ("-COS(5*pi/14)", "-SIN(pi/7)"),
    ("COS(pi/128)", "COS(pi/128)"),
    ("COS(pi/8)", "SQRT(SQRT(2) + 2)/2"),
]


@pytest.mark.parametrize(("text", "expected"), OCTANT_ANGLES, ids=str)
def test_a_trigonometric_value_is_written_within_the_first_octant(text, expected):
    assert simp(text) == expected


#: An arcsine or arccosine of a real number with magnitude above one is complex, and
#: is written in rectangular form, as are `ACOSH(1/2)` and `ATANH(2)`.
ARCS_OFF_THE_LINE = [
    ("ASIN(2)", "pi/2 - #i*LN(SQRT(3) + 2)"),
    ("ASIN(-2)", "-pi/2 + #i*LN(SQRT(3) + 2)"),
    ("ACOS(2)", "#i*LN(SQRT(3) + 2)"),
    ("ACOS(-2)", "pi - #i*LN(SQRT(3) + 2)"),
    ("ASEC(1/2)", "#i*LN(SQRT(3) + 2)"),
    ("ACSC(-1/3)", "-pi/2 + #i*LN(2*SQRT(2) + 3)"),
    ("ASIN(SQRT(2))", "pi/2 - #i*LN(SQRT(2) + 1)"),
    ("ACOSH(1/2)", "#i*pi/3"),
    ("ATANH(2)", "LN(3)/2 - #i*pi/2"),
    ("ASIN(1/3)", "ASIN(1/3)"),
    ("ASIN(1)", "pi/2"),
]


@pytest.mark.parametrize(("text", "expected"), ARCS_OFF_THE_LINE, ids=str)
def test_an_arc_off_the_real_line_is_written_in_rectangular_form(text, expected):
    assert simp(text) == expected


# -- logarithms and exponentials ----------------------------------------------

LOGARITHMS = [
    ("EXP(LN(x))", "x", None),
    ("LN(#e^3)", "3", None),
    ("#e^(#i*pi)", "-1", None),
    ("LOG(8, 2)", "3", None),
    ("#e^(2*LN(x))", "x^2", None),
    # The logarithm at zero is `-inf`, not an unsigned pole infinity; a base below
    # one flips the sign.
    ("LN(0)", "-inf", None),
    ("LN(0) + 1", "-inf", None),
    ("LN(1 - 1)", "-inf", None),
    ("EXP(LN(0))", "0", None),
    ("LOG(0)", "-inf", None),
    ("LOG(0, 10)", "-inf", None),
    ("LOG(0, 1/2)", "inf", None),
    ("VECTOR(LN(k), k, 0, 1)", "[-inf, 0]", None),
    # `e` and `i` are ordinary variables; only `#e` and `#i` are the constants.
    ("LN(e)", "LN(e)", None),
    ("i^2", "i^2", None),
    # Auto is 6.2's mixed direction: a sum of logarithms collects into one, and a
    # logarithm of a power comes apart. The latter is done even though it is
    # longer, just as `SQRT(12)` becomes `2*SQRT(3)`. Collect keeps the power.
    ("LN(256)", "8*LN(2)", None),
    ("8*LN(2)", "8*LN(2)", None),
    ("LN(256)", "LN(256)", Context(logarithm=Direction.COLLECT)),
    ("LN(4/9)", "2*LN(2/3)", None),
    # An argument that is not a perfect power stays, and a sum still collects into
    # one: Derive answers `LN(40)`.
    ("LN(12)", "LN(12)", None),
    ("3*LN(2) + LN(5)", "LN(40)", None),
    # The power rule `k*LN(a) = LN(a^k)` is the manual's for rational `k` only, so a
    # variable coefficient stays outside, in Auto as in Collect (Derive). Terms with
    # the same coefficient still add up.
    ("LN(3)*x + LN(5)", "LN(3)*x + LN(5)", None),
    ("LN(3)*x + LN(5)", "LN(3)*x + LN(5)", Context(logarithm=Direction.COLLECT)),
    ("pi*LN(2) + LN(3)", "pi*LN(2) + LN(3)", None),
    ("LN(2)*LN(3) + LN(5)", "LN(5) + LN(2)*LN(3)", None),
    ("x*LN(2) + x*LN(3)", "LN(6)*x", None),
    # The rule requires a nonnegative base, or an exponent with magnitude below one.
    ("LN(x^3)", "LN(x^3)", None),
    ("LN(x^8)", "8*LN(ABS(x))", None),
    ("LN(x^8)", "8*LN(x)", declared("x :epsilon Real [0, inf)")),
    ("LN(SQRT(z))", "LN(z)/2", None),
    # An even power of a real argument is that power of its absolute value, so it
    # splits as well; an odd power and a complex argument stay.
    ("LN(x^(-2))", "-2*LN(ABS(x))", None),
    ("LN(x^(-3))", "LN(1/x^3)", None),
    ("LN(x^2) + LN(x^4)", "6*LN(ABS(x))", None),
    ("LN((x^2 + 1)^2)", "2*LN(x^2 + 1)", None),
    ("LN(x^2)/LN(x^(-2))", "-1", None),
    # A product that is a power as a whole splits the same way, and so does a
    # polynomial that factors into one. `2*x^2` and `x^2*y` aren't powers.
    ("LN(4*x^2)", "2*LN(2*ABS(x))", None),
    ("LN(x^2/4)", "2*LN(ABS(x)/2)", None),
    ("LN(x^2*y^2)", "2*LN(ABS(x*y))", None),
    ("LN((x/y)^2)", "2*LN(ABS(x/y))", None),
    ("LN((x/y)^2)", "2*LN(x/ABS(y))", POSITIVE_X),
    ("LN(1/(x^2*y^2))", "-2*LN(ABS(x*y))", None),
    ("LN(x^4*y^2)", "2*LN(x^2*ABS(y))", None),
    ("LN(8*x^3)", "LN(8*x^3)", None),
    ("LN(8*x^3)", "3*LN(2*x)", POSITIVE_X),
    ("LN(2*x^2)", "LN(2*x^2)", None),
    ("LN(x^2*y)", "LN(x^2*y)", None),
    ("LN(x^2 + 2*x + 1)", "2*LN(ABS(x + 1))", None),
    ("LN(x^4 + 2*x^2 + 1)", "2*LN(x^2 + 1)", None),
    ("LN(x^2 - 1)", "LN(x^2 - 1)", None),
    ("LN(4*x^2)", "LN(4*x^2)", COMPLEX_X),
    ("LN(x^2 + 2*x + 1)", "LN(x^2 + 2*x + 1)", COMPLEX_X),
    ("LN(1/x)", "LN(1/x)", None),
    ("LN(x^(-2))", "LN(1/x^2)", COMPLEX_X),
    # A real power of `#e` is a positive factor, so it leaves the logarithm, following
    # the original. The rest of the argument stays whole, also beside a term in its
    # variable.
    ("LN(2*x^3*EXP(x))", "LN(2*x^3) + x", None),
    ("LN(x/EXP(y))", "LN(x) - y", None),
    ("LN(2*EXP(x))", "x + LN(2)", None),
    ("LN(4*x^2*EXP(x))", "2*LN(2*ABS(x)) + x", None),
    ("LN(x*EXP(x))", "LN(x*#e^x)", COMPLEX_X),
    ("LN(x*EXP(#i*y))", "LN(x*#e^(#i*y))", None),
    ("LN(2*x) + x", "LN(2*x) + x", None),
    ("LN(x*y)", "LN(x*y)", Context(logarithm=Direction.EXPAND)),
    (
        "LN(x*y)",
        "LN(x) + LN(y)",
        declared(
            "x :epsilon Real (0, inf)",
            "y :epsilon Real (0, inf)",
            logarithm=Direction.EXPAND,
        ),
    ),
    (
        "LN(x) + LN(y)",
        "LN(x*y)",
        declared(
            "x :epsilon Real (0, inf)",
            "y :epsilon Real (0, inf)",
            logarithm=Direction.COLLECT,
        ),
    ),
    ("LN(x) + LN(y)", "LN(x) + LN(y)", Context(logarithm=Direction.COLLECT)),
    ("#e^(x + y)", "#e^x*#e^y", Context(exponential=Direction.EXPAND)),
    ("#e^x*#e^y", "#e^(x + y)", Context(exponential=Direction.COLLECT)),
    # Powers of one base also collect in Auto, not because it's shorter: sympy's
    # `n*x^n/x` for `DIF(x^n, x)` is the same length, but the original never writes
    # that form.
    ("x^n/x", "x^(n - 1)", None),
    ("DIF(x^n, x)", "n*x^(n - 1)", None),
    # `a^LN(b)` equals `b^LN(a)` (both are `#e^(LN(a)*LN(b))`), and is written with
    # the smaller base.
    ("2^LN(3) - 3^LN(2)", "0", None),
    ("2^LN(5) - 5^LN(2)", "0", None),
    ("3^LN(2)", "2^LN(3)", None),
    ("#e^(LN(2)*LN(3))", "2^LN(3)", None),
    ("5^LN(2)/2^LN(5)", "1", None),
    ("x^LN(2)", "x^LN(2)", None),
    # Expand splits an even power of a real argument too, also inside a product. An
    # odd power and a complex argument stay.
    ("LN(x^2)", "2*LN(ABS(x))", Context(logarithm=Direction.EXPAND)),
    ("LN(4*x^2)", "2*LN(ABS(x)) + 2*LN(2)", Context(logarithm=Direction.EXPAND)),
    ("LN(x^3)", "LN(x^3)", Context(logarithm=Direction.EXPAND)),
    (
        "LN(z^2)",
        "LN(z^2)",
        declared("z :epsilon Complex", logarithm=Direction.EXPAND),
    ),
]


@pytest.mark.parametrize(("text", "expected", "context"), LOGARITHMS, ids=str)
def test_logarithms_and_exponentials(text, expected, context):
    assert simp(text, context) == expected


def test_a_line_through_logarithms_stays_a_line_when_approximated():
    """CURV_FIT.MTH fits a line to the logarithms of data. Combined by the power rule,
    the line was `LN(2.93615*16^(0.142857*x)*...)`, and on its six data points the fit
    ran away."""
    approximate = Context().with_precision(Precision.APPROXIMATE)
    assert simp("LN(3)*x + LN(5)", approximate) == "1.09861*x + 1.60943"
    fitted = "FIT([x, a*x + b], [[1, LN(1.5)], [2, LN(0.8)], [3, LN(0.4)]])"
    assert simp(fitted, approximate) == "1.07709 - 0.660877*x"


# -- the branch setting -------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected", "branch"),
    [
        # The principal cube root, in 4.5's rectangular form: Derive answers
        # `1 + SQRT(3)*#i` and approximates it as `1 + 1.73205*#i`.
        ("(-8)^(1/3)", "1 + SQRT(3)*#i", Branch.PRINCIPAL),
        ("(-8)^(1/3)", "-2", Branch.REAL),
        ("SQRT(x^2)", "ABS(x)", Branch.PRINCIPAL),
        # Any permits the rewrite without a domain proof.
        ("SQRT(x^2)", "x", Branch.ANY),
    ],
    ids=str,
)
def test_which_root_an_expression_may_mean(text, expected, branch):
    assert simp(text, Context(branch=branch)) == expected


#: `#e^(u*LN(b))` is `b^u`, following the original: term by term through a sum in the
#: exponent, with the terms that have no logarithm left over `#e`. Of two logarithms in a
#: term, the one whose argument has the most main variable gives the base. The
#: exponential stays under Exponential Collect.
POWERS_OF_LOGARITHMS = [
    ("#e^(3*x*LN(y))", "y^(3*x)", None),
    ("#e^(x*LN(2))", "2^x", None),
    ("#e^(x*LN(y + 1))", "(y + 1)^x", None),
    ("#e^(-x*LN(y)/2)", "y^(-x/2)", None),
    ("#e^(#i*x*LN(y))", "y^(#i*x)", None),
    ("#e^(3*x*LN(y) + 2)", "#e^2*y^(3*x)", None),
    ("#e^(x + x*LN(y))", "y^x*#e^x", None),
    ("#e^(x*LN(y) + x*LN(z))", "y^x*z^x", None),
    ("#e^(x*(LN(y) + LN(z)))", "y^x*z^x", None),
    ("#e^(x*y*LN(z) + 3*LN(w))", "w^3*z^(x*y)", None),
    ("#e^(LN(x)*LN(y))", "x^LN(y)", None),
    ("#e^(LN(y)*LN(x))", "x^LN(y)", None),
    ("#e^(LN(x)*LN(y))", "y^LN(x)", Context(order=("y", "x", "z"))),
    ("#e^(2*LN(x)*LN(y))", "x^(2*LN(y))", None),
    ("#e^(x*LN(y)*LN(z))", "y^(x*LN(z))", None),
    ("#e^(x*LN(y)*LN(2))", "y^(LN(2)*x)", None),
    ("#e^(LN(x + 1)*LN(y))", "(x + 1)^LN(y)", None),
    ("#e^(LN(z)*LN(y + 1))", "(y + 1)^LN(z)", None),
    ("#e^(LN(x*y)*LN(z))", "(x*y)^LN(z)", None),
    ("#e^(LN(y)^2)", "#e^(LN(y)^2)", None),
    # A complex number in the exponent stays there, following the original.
    ("#e^(#i*LN(2))", "#e^(#i*LN(2))", None),
    ("#e^(-pi/2 + #i*LN(2))", "#e^(-pi/2 + #i*LN(2))", None),
    ("#e^(pi*LN(2))", "2^pi", None),
    ("SINH(x*LN(y))", "y^x/2 - y^(-x)/2", None),
    ("#e^(3*x*LN(y))", "y^(3*x)", Context(exponential=Direction.EXPAND)),
    ("#e^(3*x*LN(y))", "#e^(3*x*LN(y))", Context(exponential=Direction.COLLECT)),
    (
        "#e^(3*x*LN(y) + 2)",
        "#e^(3*x*LN(y) + 2)",
        Context(exponential=Direction.COLLECT),
    ),
]


@pytest.mark.parametrize(
    ("text", "expected", "context"), POWERS_OF_LOGARITHMS, ids=str
)
def test_an_exponential_of_a_logarithm_is_a_power(text, expected, context):
    assert simp(text, context) == expected


#: Section 4.5's rectangular form, and three cases it doesn't cover: a symbolic base;
#: a fifth root, which would need nested radicals; and a seventh, which would need a
#: cosine, i.e. the polar form the rule avoids.
RECTANGULAR = [
    ("(1 + #i)^3", "-2 + 2*#i"),
    ("(-8)^(2/3)", "-2 + 2*SQRT(3)*#i"),
    ("(-1)^(1/4)", "SQRT(2)/2 + SQRT(2)*#i/2"),
    ("(-x)^(1/3)", "(-x)^(1/3)"),
    ("(-32)^(1/5)", "2*(-1)^(1/5)"),
    ("(-1)^(1/7)", "(-1)^(1/7)"),
]


@pytest.mark.parametrize(("text", "expected"), RECTANGULAR, ids=str)
def test_a_numeric_power_that_is_complex_is_written_in_rectangular_form(text, expected):
    assert simp(text) == expected


#: A power with a complex exponent is its principal value, `#e^(z*LN(b))`. An
#: exponential of a rational multiple of `pi*#i` is written in rectangular form when
#: the cosine and sine have simple values, like a root of -1 above.
COMPLEX_CONSTANTS = [
    ("#i^#i", "#e^(-pi/2)"),
    ("(-1)^#i", "#e^(-pi)"),
    ("#i^(2*#i)", "#e^(-pi)"),
    ("#i^(1 + #i)", "#i*#e^(-pi/2)"),
    # The logarithm of the value recovers the exponent.
    ("LN(#i^#i)", "-pi/2"),
    ("#i^#i^#i", "#i^#e^(-pi/2)"),
    ("(2*#i)^#i", "#e^(-pi/2 + #i*LN(2))"),
    ("(1 + #i)^#i", "#e^(-pi/4 + #i*LN(2)/2)"),
    # A positive base keeps its power, the shorter spelling.
    ("2^#i", "2^#i"),
    ("pi^#i", "pi^#i"),
    ("#e^(#i*pi/3)", "1/2 + SQRT(3)*#i/2"),
    ("#e^(#i*pi/4)", "SQRT(2)/2 + SQRT(2)*#i/2"),
    ("#e^(2*pi*#i/3)", "-1/2 + SQRT(3)*#i/2"),
    ("#e^(-#i*pi/4)", "SQRT(2)/2 - SQRT(2)*#i/2"),
    ("#e^(1 + #i*pi/3)", "#e*(1 + SQRT(3)*#i)/2"),
    ("#e^(pi*#i/3 + LN(2))", "1 + SQRT(3)*#i"),
    ("#e^(#i*pi/3)*x", "(1 + SQRT(3)*#i)*x/2"),
    # Fifths and sevenths of a turn have no simple rectangular form, and one radian
    # is not a rational part of a turn.
    ("#e^(#i*pi/5)", "#e^(#i*pi/5)"),
    ("#e^(#i*pi/7)", "#e^(#i*pi/7)"),
    ("#e^#i", "#e^#i"),
    # A cosine or sine of a complex number is split around its real angle.
    ("COS(pi + #i)", "-(#e^2 + 1)*#e^(-1)/2"),
    (
        "COS(pi/3 + #i)",
        "#e/4 + #e^(-1)/4 - SQRT(3)*#i*(#e/2 - #e^(-1)/2)/2",
    ),
    ("COS(1 + #i)", "COS(1 + #i)"),
]


@pytest.mark.parametrize(("text", "expected"), COMPLEX_CONSTANTS, ids=str)
def test_a_complex_constant_is_written_as_its_principal_value(text, expected):
    assert simp(text) == expected


#: A root of a polynomial that is a perfect power is taken, with only the absolute
#: values a real variable needs: intervals can settle them, a positive base needs
#: none, and a complex variable is left alone.
PERFECT_POWERS = [
    ("SQRT(x^2 + 2*x + 1)", "ABS(x + 1)", None),
    ("SQRT(x^2 - 2*x + 1)", "ABS(x - 1)", None),
    ("SQRT(4*x^2 + 4*x + 1)", "ABS(2*x + 1)", None),
    ("SQRT(x^2/4 + x + 1)", "ABS(x + 2)/2", None),
    ("SQRT(2*x^2 + 4*x + 2)", "SQRT(2)*ABS(x + 1)", None),
    ("SQRT(x^2 + 2*x*y + y^2)", "ABS(x + y)", None),
    ("SQRT(x^4 + 2*x^2 + 1)", "x^2 + 1", None),
    ("(x^2 + 2*x + 1)^(3/2)", "(x + 1)^2*ABS(x + 1)", None),
    ("(x^4 + 4*x^3 + 6*x^2 + 4*x + 1)^(1/4)", "ABS(x + 1)", None),
    ("SQRT(x^2 + 2*x + 1) + SQRT(x^2 - 2*x + 1)", "ABS(x - 1) + ABS(x + 1)", None),
    ("SQRT(x^2 + 2*x + 1)", "x + 1", POSITIVE_X),
    ("SQRT(x^2 + 2*x + 1)", "-x - 1", declared("x :epsilon Real (-inf, -1)")),
    ("SQRT(x^2 + 2*x + 1)", "SQRT(x^2 + 2*x + 1)", COMPLEX_X),
    # The principal cube root of a negative cube is not real, and a negative radicand
    # or one that is not a perfect power has no root to take.
    ("(x^3 + 3*x^2 + 3*x + 1)^(1/3)", "(x^3 + 3*x^2 + 3*x + 1)^(1/3)", None),
    ("SQRT(-x^2 - 2*x - 1)", "SQRT(-x^2 - 2*x - 1)", None),
    ("SQRT(x^2 + 1)", "SQRT(x^2 + 1)", None),
]


@pytest.mark.parametrize(("text", "expected", "context"), PERFECT_POWERS, ids=str)
def test_a_root_of_a_perfect_power_is_taken(text, expected, context):
    assert simp(text, context) == expected


def test_an_interval_goes_on_being_computed_with():
    """`INTERVAL` takes part in arithmetic, because sympy's does.

    The result is the range of the operation over every value in the interval, which
    is why the square is `INTERVAL(0, 1)`. Differentiation and integration pass it
    through too, so a limit like this one can be used further.
    """
    assert simp("INTERVAL(-1, 1) + 1") == "INTERVAL(0, 2)"
    assert simp("2*INTERVAL(-1, 1)") == "INTERVAL(-2, 2)"
    assert simp("INTERVAL(-1, 1)^2") == "INTERVAL(0, 1)"
    assert simp("SIN(LIM(SIN(1/x), x, 0))") == "INTERVAL(-SIN(1), SIN(1))"
    assert simp("DIF(y*LIM(SIN(1/x), x, 0), y)") == "INTERVAL(-1, 1)"
    assert simp("INT(LIM(SIN(1/x), x, 0), y)") == "INTERVAL(-1, 1)*y"


#: An undecidable four-argument `IF` yields its unknown clause, which can be a
#: non-expression. Both shapes occur in Derive's utility files: the truth value in
#: `SYMMETRIC_TEST_1` and `B_TYPE`, the relation in `CLAIRAUT`, all in ODE.MTH.
NOT_EXPRESSIONS = [
    ("IF(v = [] OR v = [a], true, false, false)", "false"),
    ("IF(x > 0, y = 1, y = 2, y = 3)", "y = 3"),
]


@pytest.mark.parametrize(("text", "expected"), NOT_EXPRESSIONS, ids=str)
def test_a_rewrite_is_not_asked_about_something_that_is_no_expression(text, expected):
    """A truth value has no numerator and no denominator.

    Gated rewrites are offered whatever the conditional produced, which is not always
    an expression. Splitting a truth value into a ratio makes sympy build a `Mul` of a
    `BooleanFalse` and emit a deprecation warning, which the `try` around each
    rewrite would not catch. The test requires no warnings.
    """
    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        answer = simp(text)
    assert answer == expected
    assert [str(w.message) for w in raised] == []


# -- definitions and declarations are inert -----------------------------------

INERT = [
    ("x :epsilon Real", "x :epsilon Real"),
    ("x :epsilon Integer [0, inf)", "x :epsilon Integer [0, inf)"),
    ("u := 2 + 3", "u := 5"),
    ("F(x, y) := x + x", "F(x, y) := 2*x"),
    ("F(x) :=", "F(x) :="),
    ("u :==", "u :=="),
    ('"note"', '"note"'),
    ("?", "?"),
    ("2 + 3 =", "5"),
]


@pytest.mark.parametrize(("text", "expected"), INERT, ids=str)
def test_a_definition_keeps_its_shape_and_simplifies_its_value(text, expected):
    assert simp(text) == expected


#: Calls to functions declared without a body. Each must come back unchanged, so
#: simplifying a worksheet containing one doesn't damage it.
OPAQUE = [
    "MY_FUNCTION(x, 2)",
    "ANOTHER(1)",
]
OPAQUE_FUNCTIONS = ("MY_FUNCTION", "ANOTHER")


@pytest.mark.parametrize("text", OPAQUE, ids=str)
def test_what_the_engine_has_no_mathematics_for_passes_through(text):
    assert simp(text, state=declaring(*OPAQUE_FUNCTIONS)) == text


#: `FIT` calls that can't be answered: no parameter to solve for, data whose width
#: doesn't match the label vector, and a nonlinear dependence on the parameter. Each
#: comes back unchanged rather than being guessed at.
NO_FIT = [
    "FIT([x, 1], [[1, 2], [3, 4]])",
    "FIT([x, a*x + b], [[1, 2, 3], [3, 4, 5]])",
    "FIT([x, a^2*x], [[1, 2], [3, 4]])",
]


@pytest.mark.parametrize("text", NO_FIT, ids=str)
def test_a_least_squares_fit_of_what_is_no_fit_passes_through(text):
    assert simp(text) == text


# -- substitution -------------------------------------------------------------


def test_an_assigned_variable_is_replaced_by_its_value():
    context = Context(assignments={"u": parse("2*w")})
    assert simp("u + u", context) == "4*w"


def test_a_bound_name_survives_an_assignment_wherever_the_head_binds_it():
    """A root sum binds its variable in both the polynomial and the summand.

    Elsewhere the binding covers only the first argument: a sum's limits and a
    root's index are substituted. But a root sum's polynomial and summand share the
    variable, so an assigned `t` must reach neither; otherwise an integral's answer
    would change when simplified again.
    """
    context = Context(assignments={"t": parse("5")})
    assert simp("ROOT_SUM(t^3 + t + 1, t, t*LN(x - t))", context) == (
        "ROOT_SUM(t^3 + t + 1, t, t*LN(x - t))"
    )
    # Cases that must still be substituted: an index, a limit, a free name.
    assert simp("ROOT_OF(z^5 - z - 1, z, t - 5)", context) == "ROOT_OF(z^5 - z - 1, z, 0)"
    assert simp("SUM(k, k, 1, t)", context) == "15"
    assert simp("SUM(t^2, t, 1, 3)", context) == "14"
    assert simp("t + 1", context) == "6"


def test_a_defined_function_is_replaced_by_its_body():
    context = Context(functions={"ACCELERATION": (("f", "m"), parse("f/m"))})
    assert simp("ACCELERATION(6, 2)", context) == "3"
    # Partial application: an unsupplied parameter stays as its name.
    assert simp("ACCELERATION(6)", context) == "6/m"


def test_a_counted_sum_is_written_out_for_the_recursion_under_it():
    """A sum with a known number of terms is expanded before its summand is evaluated.

    `NONZERO_ROWS` of VECTOR.MTH counts a matrix's rows by summing a recursion over
    `m SUB n_`. With a symbolic index the recursion never terminates. Expanded first,
    each term recurses over one concrete row and terminates.
    """
    state = declaring("NONZERO")
    nonzero = parse(
        "IF(i > DIMENSION(v), 0, IF(v SUB i = 0, NONZERO(v, i + 1), 1, 1))", state
    )
    rows = parse("SUM(NONZERO(m SUB k, 1), k, 1, DIMENSION(m))", state)
    context = Context(
        functions={"NONZERO": (("v", "i"), nonzero), "ROWS": (("m",), rows)}
    )
    assert simp("ROWS([[1, 0], [0, 0], [0, 3]])", context) == "2"


def defining(**definitions):
    """A context with these functions, each given as `F=("u, v", "body")`."""
    state = declaring(*definitions)
    return Context(
        functions={
            name: (tuple(p.strip() for p in params.split(",")), parse(body, state))
            for name, (params, body) in definitions.items()
        }
    )


#: NUMBER.MTH's RECURRENCE.
RECURRENCE = (
    "u, v, w, m",
    "REVERSE_VECTOR(ITERATE(APPEND([u], v), v, w, m - DIMENSION(w)))",
)

#: Calls whose argument mentions a variable the body gives values to. Definitions are
#: textual substitution, so the argument is evaluated again at each value.
REWRITTEN_ARGUMENTS = [
    ({"T": ("u, v, m", "ITERATE([u], v, [1], m)")}, "T(w SUB 1 + 1, w, 2)", "[3]"),
    ({"F": ("p, v, d", "VECTOR(p, v, d)")}, "F([a, b, c] SUB k, k, 3)", "[a, b, c]"),
    ({"T": ("u, x", "ITERATE(u, x, [3, 4], 1)")}, "T(DIMENSION(x), x)", "2"),
    ({"R": RECURRENCE}, "R(v SUB 1 + v SUB 2, v, [1, 1], 6)", "[1, 1, 2, 3, 5, 8]"),
    # ORTH_POL.MTH's CHEBYCHEV_T_LIST.
    (
        {"R": RECURRENCE},
        "R(2*x*v SUB 1 - v SUB 2, v, [x, 1], 4)",
        "[1, x, 2*x^2 - 1, 4*x^3 - 3*x]",
    ),
    # TENSOR.MTH's vNestx: the vector's index is the `n`th name of a parameter.
    (
        {"NEST": ("p, w, d, n", "IF(n < 1, p, NEST(VECTOR(p, w SUB n, d), w, d, n-1))")},
        "NEST(a + b, [a, b], 2, 2)",
        "[[2, 3], [3, 4]]",
    ),
    # TENSOR.MTH's OD_T, by way of vNestx: a derivative in an element of a vector of
    # variables, taken once its index is known.
    (
        {"G": ("u, v", "DIF(u, v)"), "H": ("u, v", "VECTOR(u, v, 2)")},
        "H(G(a*b, [a, b] SUB k), k)",
        "[b, a]",
    ),
]


@pytest.mark.parametrize(("functions", "text", "expected"), REWRITTEN_ARGUMENTS, ids=str)
def test_an_argument_is_evaluated_again_where_the_body_binds_its_variable(
    functions, text, expected
):
    assert simp(text, defining(**functions)) == expected


def test_a_derivative_in_an_element_not_yet_known_stands():
    # Taken in the unevaluated element it would be 0, the derivative of a constant.
    assert simp("DIF(a*b, [a, b] SUB k)") == "DIF(a*b, ELEMENT([a, b], k))"
    assert simp("DIF(a*b, [a, b] SUB 2)") == "a"
    context = defining(G=("u, v", "DIF(u, v)"))
    assert simp("G(a*b, [a, b] SUB k)", context) == "DIF(a*b, ELEMENT([a, b], k))"


@pytest.mark.parametrize(
    "text", ["SB(u, v, a) := ITERATE(u, v, a, 1)", "G(u, v) := DIF(u, v)"], ids=str
)
def test_a_definition_shows_a_binding_over_a_parameter_as_written(text):
    # Its variable is whatever a call supplies, which the other arguments may mention.
    assert simp(text) == text


def test_two_spellings_of_one_variable_are_one_variable():
    """Case-insensitively, `x` and `X` name the same thing.

    The spelling in a tree depends on what was declared when it was parsed:
    `X := ELEMENT(x, 1)` declares `X` but mentions `x` first, so the tree holds both.
    The session's symbol table unifies them, and the answer uses its spelling.
    """
    state = ParseState()
    parsed = parse_expression("[a:=, b:=, c:=, X := ELEMENT(x, 1)]", state)
    for declaration in parsed.declarations:
        state.declare(declaration)
    answer = simplify(parsed.node, Context(), state)
    assert answer.text == "[a :=, b :=, c :=, X := ELEMENT(X, 1)]"
    # Without a symbol table the tree stays as parsed.
    assert simplify(parsed.node, Context()).text.endswith("X := ELEMENT(x, 1)]")


def test_a_label_is_replaced_by_what_it_names():
    context = Context(labels={3: parse("x + 1")})
    assert simp("#3*2", context) == "2*x + 2"


def test_an_unresolved_label_survives():
    assert simp("#3 + 1") == "#3 + 1"


def test_an_assignment_that_reaches_itself_stops():
    state = ParseState()
    state.declare(VariableDeclaration("pn", True))
    context = Context(assignments={"pn": parse("pn + 1", state)})
    assert simp("pn", context, state) == "pn + 1"


# -- Simplify works on a subexpression ----------------------------------------


def test_a_subtree_simplifies_on_its_own():
    # The session simplifies the highlighted subexpression, so the engine must not
    # assume it has a whole entry.
    node = parse("1 + (2*x + 3*x)")
    assert simplify(node.children[1], Context()).text == "5*x"


def test_a_result_carries_a_tree_whose_spans_index_its_text():
    result = simplify(parse("(x + 1)*(x - 1) + 1"), Context())
    assert result.text[result.node.start : result.node.end] == result.text
    assert to_sexpr(result.node) == to_sexpr(parse(result.text))


# -- from Soft Warehouse's own demo scripts -----------------------------------
#
# `artifacts/sessions/derive-3.14-dos/*.DMO` are the annotated demos shipped with the
# original, which together make a good capability test for Simplify. These are their
# cases, in author notation, with this engine's answers.

DEMO_NUMBERS = [
    ("5 (9 - 3) / 2", "15"),
    ("(2/27)^(2/3)", "2^(2/3)/9"),
    ("SQRT(-1)", "#i"),
    ("(1 + 2 #i) (3 + #i)", "1 + 7*#i"),
    # A complex denominator is rationalized.
    ("(2 + #i) / (#i + 1)", "3/2 - #i/2"),
    ("SQRT(55/36 + 4/3 #i)", "4/3 + #i/2"),
    ("inf + inf + 5", "inf"),
    # Two nested radicals that denest only separately.
    ("SQRT(5 + SQRT 24) + SQRT(5 - SQRT 24)", "2*SQRT(3)"),
]

DEMO_ALGEBRA = [
    ("(x+a)^2-2*a*x", "x^2 + a^2"),
    # The tenth power stays folded while the square around it is expanded.
    ("(x + (a + 1)^10)^2 - (a + 1)^20", "x^2 + 2*x*(a + 1)^10"),
    (
        "((a*n + b*m)^2 + (a*m - b*n)^2) / ((a*p + b*q)^2 + (a*q - b*p)^2)",
        "(m^2 + n^2)/(p^2 + q^2)",
    ),
    (
        "((p*x^2+(k-s)*x+r)^2-(p*x^2+(k+s)*x+r)^2)"
        "/((p*x^2+(k+t)*x+r)^2-(p*x^2+(k-t)*x+r)^2)",
        "-s/t",
    ),
    # The common fifth power is factored out; the point is the cofactor, not the
    # factor order, which is sympy's.
    (
        "2 (x^2 - y^2)^6 - (x^2 - y^2)^5(2 x^2 - 3)",
        "(x^2 - y^2)^5*(3 - 2*y^2)",
    ),
]

DEMO_FUNCTIONS = [
    ("LN 6 - LN 2", "LN(3)"),
    ("LN 16 / LN 8", "4/3"),
    ("LN (1 + #i)", "LN(2)/2 + #i*pi/4"),
    ("LOG (10^3, 10)", "3"),
    ("LOG x", "LN(x)"),
    ("LN EXP x", "x"),
    ("4 (2^z)^2 - 4^(z+1)", "0"),
    ("a^(x+1) - a a^x", "0"),
    ("SIGN 5", "1"),
    ("ABS (x^2)", "x^2"),
    ("(ABS x)^2", "x^2"),
    # The demo's closing statistics. A numeric coefficient distributes, so the
    # average is written termwise where the manual writes one quotient. The average
    # of 1/1 through 1/10 is the tenth harmonic number over 10.
    ("AVERAGE (x, y, z)", "x/3 + y/3 + z/3"),
    ("AVERAGE (VECTOR (1/k, k, 1, 10))", "7381/25200"),
    ("RMS ([2, 3, 5])", "SQRT(114)/3"),
    # The unbiased sample variance of two values, and its square root, which comes
    # out of the radical because the variance is a square.
    ("VAR (x, y)", "(x - y)^2/2"),
    ("STDEV (x, y)", "SQRT(2)*ABS(x - y)/2"),
]

DEMO_TRIGONOMETRY = [
    ("COS (17/6 pi)", "-SQRT(3)/2"),
    ("SIN (-30 deg)", "-1/2"),
    (
        "(1 - (COS x)^2)^4 (1 - (SIN x)^2)^3 ((SIN x)^2 + (COS x)^2)^5",
        "SIN(x)^8*COS(x)^6",
    ),
    ("(COS (a/2) + SIN (a/2))^2", "SIN(a) + 1"),
    ("1/(1+TAN(a)*TAN(a/2))", "COS(a)"),
    ("CSC (2x) (2 (COS (x/2))^2 - 1)", "1/(2*SIN(x))"),
    ("(1 + SIN a) * (1 + SEC a) / ((1 + COS a) * (1 + CSC a))", "TAN(a)"),
    ("(TAN a)^2 / (1 + (TAN a)^2) * (1 + (COT a)^2) / (COT a)^2", "TAN(a)^2"),
    ("TAN ATAN x", "x"),
    ("ATAN (-1, -1)", "-3*pi/4"),
    ("ASIN (-1/2)", "-pi/6"),
    ("ACOS (- 1 / SQRT 2)", "3*pi/4"),
    ("ASIN (x/SQRT(x^2 + 1))", "ATAN(x)"),
    ("ATAN (x/SQRT(1-x^2))", "ASIN(x)"),
    ("ASEC (1/x)", "ACOS(x)"),
]

DEMO_CALCULUS = [
    ("LIM (((x + h)^3 - x^3) / h, h, 0)", "3*x^2"),
    ("LIM (x^(1/x), x, inf)", "1"),
    ("LIM ((a^x - b^x) / x, x, 0)", "LN(a) - LN(b)"),
    ("LIM (1/x, x, 0, -1)", "-inf"),
    # A numeric coefficient distributes over the sum, so the text reads back as
    # the same expression.
    ("INT (x^2, x, a, b)", "b^3/3 - a^3/3"),
    ("INT (1/SQRT x, x, 0, b^2)", "2*ABS(b)"),
    ("INT (INT (x y, y, 0, SQRT (r^2 - x^2)), x, 0, r)", "r^4/8"),
    # The constant of integration makes the answer continuous at `a = 0`, where
    # Derive's `SIN(a*x^3 + b)/(3*a)` is undefined.
    ("INT (x^2 COS (a x^3 + b), x)", "(SIN(a*x^3 + b) - SIN(b))/(3*a)"),
    ("TAYLOR (#e^x, x, 0, 5)", "x^5/120 + x^4/24 + x^3/6 + x^2/2 + x + 1"),
    ("TAYLOR (LN COS (a x), x, 0, 7)", "-a^6*x^6/45 - a^4*x^4/12 - a^2*x^2/2"),
    # The closed form keeps the summation formula's factored shape, as the demo's
    # screen shows.
    ("SUM (k, k, 0, n)", "n*(n + 1)/2"),
    ("SUM (k^3, k, 0, n)", "n^2*(n + 1)^2/4"),
    ("SUM (2^-k, k, 0, inf)", "2"),
    ("PRODUCT (2 k, k, 1, n)", "2^n*n!"),
]

DEMO_MATRICES = [
    # The table of squares and cubes the demo opens with.
    (
        "VECTOR([x,x^2,x^3],x,1,8)",
        "[[1, 1, 1], [2, 4, 8], [3, 9, 27], [4, 16, 64], [5, 25, 125], "
        "[6, 36, 216], [7, 49, 343], [8, 64, 512]]",
    ),
    ("IDENTITY_MATRIX(3)", "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"),
    ("[[a,b,c],[1,2,3]] SUB 1 SUB 2", "b"),
    ("[[a,b,c],[1,2,3]] SUB 2", "[1, 2, 3]"),
    ("2*[[a,2],[3,b]]+[[1,3],[a,-b]]", "[[2*a + 1, 7], [a + 6, b]]"),
    ("[2,a,5] . [2*a,3,-1]", "7*a - 5"),
    # `.` is the matrix product wherever the shapes conform, the dot product being
    # a special case.
    ("[[a,b],[c,d]] . [[x],[y]]", "[[a*x + b*y], [c*x + d*y]]"),
    ("[[a,b],[c,d]] . [[a,b],[c,d]]^(-1)", "[[1, 0], [0, 1]]"),
    (
        "[[a,b],[c,d]]^(-1) . [[e],[f]]",
        "[[(d*e - b*f)/(a*d - b*c)], [(a*f - c*e)/(a*d - b*c)]]",
    ),
    # The outer product, built with the same operator.
    (
        "[[a],[b],[c]] . [[2,3,4]]",
        "[[2*a, 3*a, 4*a], [2*b, 3*b, 4*b], [2*c, 3*c, 4*c]]",
    ),
    ("CROSS([1,2,3],[a,b,c])", "[2*c - 3*b, 3*a - c, b - 2*a]"),
    ("DET([[2,3],[a,b]])", "2*b - 3*a"),
    ("TRACE([[a,b],[1,2]])", "a + 2"),
    ("[[a,b,c],[1,2,3]]`", "[[a, 1], [b, 2], [c, 3]]"),
    (
        "[[a,b],[2,3]]^(-1)",
        "[[3/(3*a - 2*b), -b/(3*a - 2*b)], [-2/(3*a - 2*b), a/(3*a - 2*b)]]",
    ),
    # A singular but consistent system: the zero row leaves the second unknown
    # arbitrary, and the first row reads `x + 2*y = 3`.
    ("ROW_REDUCE([[2,4],[3,6]],[[6],[9]])", "[[1, 2, 3], [0, 0, 0]]"),
    ("CHARPOLY([[a,b],[b,a]],z)", "z^2 - 2*a*z + a^2 - b^2"),
    # With no variable given, the answer uses `w`, the manual's default. Derive
    # lists the two in the opposite order.
    ("EIGENVALUES([[a,b],[b,a]])", "[w = a - b, w = a + b]"),
    # The demo's vector calculus, in default Cartesian x, y, z. It ends with a
    # cross-check: the vector potential of the curl computed two lines above gives
    # back the original vector.
    ("GRAD(x+y^2+z^3)", "[1, 2*y, 3*z^2]"),
    ("DIV([1,2*y,3*z^2])", "6*z + 2"),
    ("LAPLACIAN(x+y^2+z^3)", "6*z + 2"),
    ("CURL([y^2,2*x*z,0])", "[-2*x, 0, 2*z - 2*y]"),
    ("POTENTIAL([1,2*y,3*z^2])", "x + y^2 + z^3"),
    ("VECTOR_POTENTIAL([-2*x,0,2*z-2*y])", "[y^2, 2*x*z, 0]"),
]

DEMO = (
    DEMO_NUMBERS
    + DEMO_ALGEBRA
    + DEMO_FUNCTIONS
    + DEMO_TRIGONOMETRY
    + DEMO_CALCULUS
    + DEMO_MATRICES
)


@pytest.mark.parametrize(("text", "expected"), DEMO, ids=str)
def test_a_case_from_the_demo_scripts(text, expected):
    assert simp(text) == expected


# -- totality -----------------------------------------------------------------

#: One expression per node kind the parser can build. `PARAMS` and `INTERVAL` are
#: covered inside the definition and the declaration that contain them.
KINDS = [
    "2/3",
    "x",
    '"note"',
    "#3",
    "?",
    "x - y + 2",
    "2*x*y",
    "x/y",
    "x^2",
    "a . b",
    "-x",
    "±x",
    "x!",
    "a`",
    "x SUB 1",
    "|x|",
    "SIN(x)",
    "SIN x",
    "SIN^2 x",
    "[1, 2, 3]",
    "x = 1",
    "NOT p",
    "p AND q",
    "p OR q",
    "p XOR q",
    "p IMP q",
    "u := 5",
    "F(x, y) := x + y",
    "x :epsilon Real",
    "x :epsilon Real [0, inf)",
    "x + 1 =",
]


def test_every_node_kind_reaches_simplify():
    covered = set()
    for text in KINDS:
        stack = [parse(text)]
        while stack:
            node = stack.pop()
            covered.add(node.kind)
            stack.extend(node.children)
    assert covered == set(Kind)


@pytest.mark.parametrize("text", KINDS, ids=str)
def test_a_construct_survives_being_simplified(text):
    once = simplify(parse(text), Context())
    assert simplify(once.node, Context()).text == once.text


@pytest.mark.parametrize("name", sorted(BUILTIN_FUNCTIONS), ids=str)
def test_every_builtin_survives_being_simplified(name):
    """Simplify is total over builtins: none raises or loses its arguments.

    This doesn't claim they compute anything; many don't yet.
    """
    text = f"{name}(x, 2)"
    once = simplify(parse(text), Context())
    assert simplify(once.node, Context()).text == once.text


# -- invariants over every case above -----------------------------------------

#: Inputs that take seconds to come back unchanged. Unevaluated heads are covered by
#: many cheaper cases.
SLOW_TO_STAND = {"INT(SIN(m*x)*SIN(2*x), x, 0, pi)"}

#: Every input this module and its siblings simplify, with its context, except those
#: above.
EVERY_CASE = [
    *((text, None) for text, _ in ARITHMETIC + RATIONAL + RADICALS + COMBINATORIAL),
    *((text, context) for text, _, context in DOMAINS if text not in SLOW_TO_STAND),
    *((text, context) for text, _, context in TRIGONOMETRY),
    *((text, context) for text, _, context in LOGARITHMS),
    *((text, context) for text, _, context in POWERS_OF_LOGARITHMS),
    *(
        (text, None)
        for text, _ in CALCULUS
        + INFINITE_SUMS
        + SPECIAL
        + VECTORS
        + LINEAR_ALGEBRA
        + VECTOR_CALCULUS
        + GENERATED
        + SELECTED
        + ITERATION
        + STATISTICS
        + STRUCTURE
        + DECOMPOSITION
        + NUMBER_THEORY
        + INDICATOR_AND_NORMAL
        + LOGIC
        + TRUTH_TABLES
        + INERT
        + APPROXIMATED
    ),
    *((text, context) for text, _, context in CONDITIONALS),
    *((text, None) for text in OPAQUE),
    *((text, None) for text, _ in DEMO),
]


#: The only inputs whose answer a second Simplify may change; see
#: `test_a_heading_heads_its_column_only_while_it_is_a_heading`.
HEADED = {text for text, _ in TRUTH_TABLES}


@pytest.mark.parametrize(("text", "context"), EVERY_CASE, ids=str)
def test_a_result_is_a_fixed_point_of_printing_and_of_simplify(text, context):
    """An answer survives print/parse unchanged, and a second Simplify doesn't change it.

    The first is what the worksheet relies on: results are stored as text and read
    back, so a change would make a label refer to a different expression than the
    one shown. The second is what "fully simplified" means. Both are checked on one
    answer since computing it is most of the cost.
    """
    context = context or Context()
    state = declaring(*OPAQUE_FUNCTIONS)
    once = simplify(parse(text, state), context, state)
    twice = from_sympy(to_sympy(once.node, context), context)
    assert twice.text == once.text
    assert to_sexpr(twice.node) == to_sexpr(once.node)
    if text not in HEADED:
        assert simplify(once.node, context, state).text == once.text


def test_a_heading_heads_its_column_only_while_it_is_a_heading():
    """A truth table's column headings stay unsimplified inside the table. Once on
    the worksheet the table is an ordinary matrix, and a second Simplify simplifies
    `p XOR q` like anywhere else."""
    once = simplify(parse("TRUTH_TABLE(p, q, p XOR q)"), Context())
    assert once.text.startswith("[[p, q, p XOR q], ")
    again = simplify(once.node, Context())
    assert again.text.startswith("[[p, q, NOT p AND q OR p AND NOT q], ")


#: Corpus expressions that settle only on a second pass are listed in
#: `test_simplify_corpus.py`, next to the sweep that finds them.
