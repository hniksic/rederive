"""What Simplify must produce: the normal form, and the command's contract.

Author text in, author text out: each case is written the way a user would type
it and asserted against the text the worksheet would show, because that is the
whole of what the command promises. The cases are inline rather than in a corpus
file so that a failure names the mathematics it is about.

This file holds what Simplify itself answers for: numbers, polynomials and rational
functions, the normal form a sum or a quotient is written in, the domains that decide
what a rewrite may assume, absolute values, trigonometry, logarithms and the branch
setting, substitution, and the contract over every case - that each node kind and each
builtin survives being simplified, and that an answer is a fixed point of printing and
of a second Simplify. What Simplify is merely the door to has files of its own:
`test_integrate.py` for integrals, `test_calculus.py` for the other calculus heads,
`test_matrices.py` for vectors and matrices, `test_logic.py` for relations, logic and
conditionals, `test_approximate.py` for the precision modes and `test_functions.py` for
special values and the named functions. Their case lists are imported below only to be
asked the fixed-point question here.

Where the engine's form differs from the manual's printed one - term order, an
equivalent spelling - the engine's form is what is recorded, and the case says
so. Nothing here is weakened to make a test pass: an expected value is either
what Derive prints or a hand-checked equal of comparable simplicity.
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
from rederive.engine.context import Angle, Branch, Direction, TrigPower
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
    # A divisor of integers, standing over anything else: checked against the
    # original, which writes the arguments least main first.
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
    # Difference of squares. Multiplying out the product is what shows it; also
    # expanding the power would produce a hundred terms, and be slow doing it.
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
    # A cube root denests too, where the radicand is a sum of surds that some
    # cube of one is. Sympy denests square roots and no others, so this is the
    # minimal polynomial factored over the field the surds generate.
    ("(243*SQRT(5) - 294*SQRT(3))^(1/3)", "3*SQRT(5) - 2*SQRT(3)"),
    ("(10 + 6*SQRT(3))^(1/3)", "SQRT(3) + 1"),
    # A root taken out of a root is a simplification even where the spelling without
    # it is no shorter: the cube of the golden ratio is `2 + SQRT(5)`, and the square
    # root of `2 + SQRT(3)` is two surds over two. One that does not denest stands.
    ("(2 + SQRT(5))^(1/3)", "SQRT(5)/2 + 1/2"),
    ("SQRT(2 + SQRT(3))", "SQRT(6)/2 + SQRT(2)/2"),
    ("SQRT(4 + SQRT(15))", "SQRT(10)/2 + SQRT(6)/2"),
    ("x*SQRT(2 + SQRT(3))", "(SQRT(6) + SQRT(2))*x/2"),
    ("SQRT(1 + SQRT(2))", "SQRT(SQRT(2) + 1)"),
    # A whole power of a sum of surds is a number of the field they span, and is
    # written as one there whether or not that is shorter; so is its reciprocal. Under
    # Expand too, which reaches the same number.
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
    # A power whose value runs to hundreds of digits keeps its exponent, and one of a
    # transcendental or of a variable is no number of such a field.
    ("(1 + SQRT(2))^1000", "(SQRT(2) + 1)^1000"),
    ("(1 + pi)^2", "(pi + 1)^2"),
    ("(x + SQRT(2))^2", "(x + SQRT(2))^2"),
    ("(x + 1)^5", "(x + 1)^5"),
    # 3.8 p.47's quotient, where no one root of the four is anything simpler on
    # its own: the numerator is `(SQRT(3) + 1)^(3/2)` and the denominator twice
    # it, so the value is a number of the field `SQRT(3)` generates and the
    # whole quotient at once is what shows it.
    ("(SQRT(SQRT(3) + 1) + SQRT(3*SQRT(3) + 3))/SQRT(24*SQRT(3) + 40)", "1/2"),
    # The value need not be rational, only a number of that field.
    (
        "SQRT(3)*(SQRT(SQRT(3) + 1) + SQRT(3*SQRT(3) + 3))/SQRT(24*SQRT(3) + 40)",
        "SQRT(3)/2",
    ),
    # And a variable standing beside it is no reason to leave the number alone.
    ("x*(SQRT(SQRT(3) + 1) + SQRT(3*SQRT(3) + 3))/SQRT(24*SQRT(3) + 40)", "x/2"),
    # The same three roots put together another way stand, that value being
    # `(SQRT(3) + 1)^(3/2)/2` - one root deeper than the field, so there is no
    # shorter spelling of it to find.
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
    # Section 6.9 writes both as ratios of factorials, and a ratio that reduces
    # is reduced rather than left standing: `COMB(x, 2)` is `x!/(2*(x - 2)!)`
    # written out and `x*(x - 1)/2` once the two factorials have cancelled.
    ("COMB(x, 2)", "x*(x - 1)/2"),
    ("PERM(x, 2)", "x*(x - 1)"),
    # A head that declined to evaluate keeps its `COMB`. Written out it would
    # be a summand nobody has offered to sympy yet, and the search for a closed
    # form over a ratio of factorials of the index does not come back.
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
    # Factoring is the Factor command. Simplify leaves the sum a sum even
    # though the product is shorter to write.
    assert simp("x^2 + 2*x") == "x^2 + 2*x"


def test_a_high_power_of_a_sum_is_not_multiplied_out_to_be_thrown_away():
    # Derive answers both of these unchanged, and answers at once. The
    # sixtieth power is six hundred thousand terms multiplied out, which no
    # rewrite here keeps and every rewrite that builds it pays for, so the
    # expansion is counted before it is built and declined. The ratio is the
    # case a polynomial test does not cover: `cancel` is worth offering where
    # something really is divided, and not at this size.
    assert simp("(v + w + x + y + z)^60") == "(x + y + z + v + w)^60"
    assert simp("(v + w + x + y + z)^60/q") == "(x + y + z + v + w)^60/q"


# -- the normal form ----------------------------------------------------------

#: Every one checked against the original. A sum is written as a rational
#: function of the most main variable it holds; a product or a power that is
#: not itself a sum is left exactly as it was written.
#:
#: The terms are the original's and so is the order they are written in, the
#: order list deciding both which sums get multiplied out and which term of a
#: sum leads. The point of each case is the first of those: what was left
#: folded because the primary variable is not in it.
NORMAL_FORM = [
    # A sum in the primary variable is multiplied out however long it gets,
    # and a term free of the primary variable is not touched.
    ("(x + 1)^9 + y", "x^9 + 9*x^8 + 36*x^7 + 84*x^6 + 126*x^5 + 126*x^4 "
                      "+ 84*x^3 + 36*x^2 + 9*x + y + 1"),
    ("(y + 1)^9 + x", "x + (y + 1)^9"),
    ("(x + 1)^2 + y", "x^2 + 2*x + y + 1"),
    # No sum to write, so nothing happens - this is not a wholesale expansion.
    ("(x + 1)^9", "(x + 1)^9"),
    ("2*x*(x - 3)^2", "2*x*(x - 3)^2"),
    ("(x + 1)*(y + 1)", "(x + 1)*(y + 1)"),
    ("(x + 1)*(x + 2)", "(x + 1)*(x + 2)"),
    ("(x + y)^2", "(x + y)^2"),
    ("(x + 1)^2/y", "(x + 1)^2/y"),
    # Calls whose arguments are one polynomial written two ways are one kernel,
    # and cancel as one.
    ("#e^(2*x*(y + 1) + y) - #e^(2*x*y + 2*x + y)", "0"),
    ("SIN(2*x*(y + 1)) - SIN(2*x*y + 2*x)", "0"),
    # Wherever the sum stands: inside a product, under a power, under a
    # function, in a denominator, in an exponent, one element of a vector.
    ("z*((x + 1)^2 + y)", "z*(x^2 + 2*x + y + 1)"),
    ("((x + 1)^2 + y)^2", "(x^2 + 2*x + y + 1)^2"),
    ("SIN((x + 1)^2 + y)", "SIN(x^2 + 2*x + y + 1)"),
    ("1/((x + 1)^2 + y)", "1/(x^2 + 2*x + y + 1)"),
    ("1/((y + 1)^2 + x)", "1/(x + (y + 1)^2)"),
    ("[(x + 1)^2 + y, (y + 1)^2 + x]", "[x^2 + 2*x + y + 1, x + (y + 1)^2]"),
    # An exponent that is not a whole number is no polynomial degree. The
    # power is still a kernel of `x`, so it leads the `y` beside it.
    ("(x + 1)^n + y", "(x + 1)^n + y"),
    # The coefficients are opaque: what is free of the primary variable is one
    # quantity, so the answer is in powers of `2*y + 1` and not of `y`. The
    # part that is free of `x` is then written about `y`, its own primary.
    ("(x + 2*y + 1)^3 + z", "x^3 + 3*x^2*(2*y + 1) + 3*x*(2*y + 1)^2 "
                            "+ 8*y^3 + 12*y^2 + 6*y + z + 1"),
    # The `x^2` coefficient keeps its numeric content, where the original
    # writes `2*x^2*(y + 3)`; the original writes it the other way round too,
    # and the manual's LAPLACIAN is the case this follows.
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


#: The rational half of the same rule, also checked against the original.
#: Terms holding the primary variable go over a common denominator and long
#: division takes the polynomial part out, so what is left over the
#: denominator is proper.
RATIONAL_FORM = [
    ("1/(x + 1) + 1/(x + 2) + y", "y + (2*x + 3)/((x + 1)*(x + 2))"),
    ("1/(x + 1)^2 + 1/(x + 1) + y", "y + (x + 2)/(x + 1)^2"),
    ("y/(x + 1) + z/(x + 1)", "(y + z)/(x + 1)"),
    ("(x^3 + 1)/(x + 2) + y", "x^2 - 2*x + y - 7/(x + 2) + 4"),
    ("x/(x + 1) + y", "y - 1/(x + 1) + 1"),
    ("x^2 + 1/(x + 1) + y", "x^2 + y + 1/(x + 1)"),
    # The denominator keeps whatever shape it was written in: Derive answers
    # this with the two factors standing, not with a quartic underneath. It
    # writes the numerator `2*(x^2 + y^2 + 18)`, which is the same number of
    # terms with the 2 taken outside and is not a form this can hold - see
    # `test_a_numeric_factor_stays_among_the_terms_it_divides`.
    ("1/(9 + x^2 + (y - 3)^2) + 1/(9 + x^2 + (y + 3)^2)",
     "(2*x^2 + 2*y^2 + 36)/((x^2 + y^2 + 6*y + 18)*(x^2 + y^2 - 6*y + 18))"),
    # A denominator free of the primary variable belongs to a coefficient, so
    # it is combined only where more than one term carries it.
    ("x/y + z", "x/y + z"),
    ("1/y + z", "1/y + z"),
    ("(x + 1)^2/y + z", "(x^2 + 2*x + y*z + 1)/y"),
    ("x/y + x/z", "x*(1/y + 1/z)"),
    ("x/(y + 1) + x/(y + 2)", "x*(2*y + 3)/((y + 1)*(y + 2))"),
]


@pytest.mark.parametrize(("text", "expected"), RATIONAL_FORM, ids=str)
def test_a_sum_of_ratios_is_written_over_one_denominator(text, expected):
    assert simp(text) == expected


#: What a quotient's numerator has in common, written outside it, which is the
#: form Derive's own gcd leaves behind and prints. Checked against the original.
NUMERATOR_CONTENT = [
    ("(x^2 + x)/n", "x*(x + 1)/n"),
    ("(a x + a y)/z", "a*(x + y)/z"),
    ("(x y + x)/z", "x*(y + 1)/z"),
    ("(n^2 + n)/(x + 1)", "n*(n + 1)/(x + 1)"),
    ("(x^2 + x)/(2 y)", "x*(x + 1)/(2*y)"),
    # A common factor and not a factorization: these two have nothing in
    # common term by term, and Derive prints them as they stand as well.
    ("(x^2 - 1)/(y + 1)", "(x^2 - 1)/(y + 1)"),
    ("(x^2 + 2 x + 1)/y", "(x^2 + 2*x + 1)/y"),
    # Nothing divided is no numerator to read: a polynomial keeps its terms.
    ("x^2 + x", "x^2 + x"),
]


@pytest.mark.parametrize(("text", "expected"), NUMERATOR_CONTENT, ids=str)
def test_a_quotient_carries_its_numerator_s_common_factor_outside(text, expected):
    assert simp(text) == expected


def test_a_numeric_factor_stays_among_the_terms_it_divides():
    """`(2*x + 2)/y` and not `2*(x + 1)/y`, where Derive writes the second.

    Not a judgement about which reads better. Sympy multiplies a number back
    over a sum as it builds one, so the second form is one no answer can be
    written in: printing it and reading the print back gives the first, and an
    entry whose text and tree disagree is what `test_simplify_corpus` fails on.
    A factor with a variable in it survives that round trip and is taken out.
    """
    assert simp("(2 x + 2)/y") == "(2*x + 2)/y"
    assert simp("(2 x y + 2 y)/z") == "2*y*(x + 1)/z"


#: A number in front of a sum, which is the same rule seen from the other side:
#: a polynomial's coefficients are rational, term by term, so a number that
#: divides into them belongs among them and not in front. The one that does not
#: divide in is a denominator the whole sum shares, and that one stays in front.
COEFFICIENTS = [
    ("(m^2 + m + 2)*(m - 1)!/2", "(m^2/2 + m/2 + 1)*(m - 1)!"),
    ("(n^2 + n)/2", "n^2/2 + n/2"),
    ("x*(x - 1)/2", "x*(x - 1)/2"),
    ("z*(x + y)/2", "z*(x + y)/2"),
    # A whole number in front is not a denominator and is left where it is.
    ("6*a*b*(a*x + b*y)", "6*a*b*(a*x + b*y)"),
    # And the way back is not taken: a sum of terms with denominators of their
    # own is never gathered over the one they have in common, however many
    # carry it. All three come from the original.
    ("x^2/6 + x/2", "x^2/6 + x/2"),
    ("x^10/2 - x^9/3 + x/6", "x^10/2 - x^9/3 + x/6"),
    ("1/2 + SIN(x)/2", "SIN(x)/2 + 1/2"),
]


@pytest.mark.parametrize(("text", "expected"), COEFFICIENTS, ids=str)
def test_a_number_that_divides_a_sum_is_written_into_it(text, expected):
    assert simp(text) == expected


def test_a_sum_over_two_powers_of_one_thing_is_written_over_the_higher():
    """The lower power divides the higher, so the sum is one ratio.

    All four match the original. The rule is what its derivatives need: sympy
    differentiates a quotient into a sum over two powers of the denominator,
    where the original writes one ratio - and the numerator of that ratio is
    often what the whole answer turns on, `1` in the first case here.

    Only where it shortens the sum. `1/SQRT(x) + 1/x` over `x^(3/2)` is longer
    than it was and the original leaves it alone, as it does the first case's
    sum authored as it stands - two fixed points for the one expression, and
    this takes the answer it gives to the question that was asked.
    """
    assert simp("DIF(x/SQRT(1 - x^2), x)") == "1/(1 - x^2)^(3/2)"
    assert simp("DIF(x^3/SQRT(x^2 + 1), x)") == "x^2*(2*x^2 + 3)/(x^2 + 1)^(3/2)"
    assert simp("1/(x + 1)^(3/2) + 1/SQRT(x + 1)") == "(x + 2)/(x + 1)^(3/2)"
    assert simp("1/SQRT(x) + 1/x") == "1/SQRT(x) + 1/x"


def test_no_denominator_carries_a_denominator_of_its_own():
    """A compound fraction is cleared however long that makes it.

    Against the original: `DIF(ATAN(x/a), x)` is `a/(x^2 + a^2)` there and
    `1/(a + x^2/a)` to sympy, and the two are the same number of operations - so
    the rule cannot be a count. It is a shape the original never writes.
    """
    assert simp("DIF(ATAN(x/a), x)") == "a/(x^2 + a^2)"
    assert simp("1/(1 + 1/x)") == "x/(x + 1)"


def test_the_order_list_decides_what_gets_multiplied_out():
    """The manual's 4.3 exercise, as the original answers it: the same entry
    expands under one order list and comes back as it was written under
    another, because the primary variable is what the list says it is."""
    assert simp("(x + 1)^9 + y", Context(order=("x", "y", "z"))).startswith("x^9")
    assert simp("(x + 1)^9 + y", Context(order=("y", "x", "z"))) == "y + (x + 1)^9"


# -- domains: the whole of "don't guess" --------------------------------------

DOMAINS = [
    # An undeclared variable is real, and a real square has a real root.
    ("SQRT(x^2)", "ABS(x)", None),
    ("SQRT(x^2)", "SQRT(x^2)", COMPLEX_X),
    ("SQRT(x^2)", "x", POSITIVE_X),
    ("SQRT(x^2)", "-x", declared("x :epsilon Real (-inf, 0)")),
    # `x` real could be negative, and then `LN(x^2 - x)` is not `LN(x) +
    # LN(x - 1)`. Term order is the engine's.
    ("LN(x^2 - x) - LN(x)", "LN(x^2 - x) - LN(x)", None),
    ("LN(x^2 - x) - LN(x)", "LN(x - 1)", POSITIVE_X),
    ("LN(x^2)", "LN(x^2)", None),
    (
        "LN(x^2)",
        "2*LN(x)",
        declared("x :epsilon Real (0, inf)", logarithm=Direction.EXPAND),
    ),
    ("SQRT(x)*SQRT(y)", "SQRT(x)*SQRT(y)", None),
    ("(-1)^(2*n)", "1", declared("n :epsilon Integer")),
    ("(-1)^(2*n)", "(-1)^(2*n)", None),
    # `pi` alone would be wrong where `m` is 0.
    (
        "INT(SIN(m*x)^2, x, 0, 2*pi)",
        "IF(m /= 0, pi, 0)",
        declared("m :epsilon Integer"),
    ),
    # 0 would be wrong where `m` is 2 or -2: the integral is then `pi/2` or
    # `-pi/2`. Without a case split to say so, the integral stands.
    (
        "INT(SIN(m*x)*SIN(2*x), x, 0, pi)",
        "INT(SIN(2*x)*SIN(m*x), x, 0, pi)",
        declared("m :epsilon Integer"),
    ),
    # `#e^(k*LN(y))` is `y^k` for positive `y` and not otherwise. Derive
    # answers it whatever `y` is; where nothing declares it, the expression is
    # the answer.
    ("#e^(3*x*LN(y))", "#e^(3*x*LN(y))", None),
    ("#e^(3*x*LN(y))", "y^(3*x)", declared("y :epsilon Real (0, inf)")),
    # The tangent pair adds to a right angle turned the way its argument is
    # signed, which is an answer and not a guess: both halves are written down.
    # Off the real line it is neither - `SIGN` is no longer `±1` there, and the
    # identity does not hold - so the pair stands as it was written.
    ("ACOT(t) + ATAN(t)", "pi*SIGN(t)/2", None),
    ("ACOT(x) + ATAN(x)", "ATAN(x) + ACOT(x)", COMPLEX_X),
]

#: What a declaration's interval is worth past the sign of it. A symbol can
#: carry "above zero" and nothing else, so everything here is a question the
#: bounds answer and the converted expression cannot.
INTERVALS = [
    ("ABS(x - 1)", "1 - x", "x :epsilon Real (0, 1)"),
    ("ABS(x - 1)", "1 - x", "x :epsilon Real [0, 1]"),
    ("ABS(x - 1)", "ABS(x - 1)", "x :epsilon Real (0, 2)"),
    ("ABS(x - 1)", "ABS(x - 1)", "x :epsilon Real (0, inf)"),
    ("SQRT((x - 1)^2)", "1 - x", "x :epsilon Real (0, 1)"),
    # Both bars come off, and then the sum cancels.
    ("ABS(x) + ABS(x - 1)", "1", "x :epsilon Real (0, 1)"),
    # `SIGN` is the one that needs the strict question: it has a third answer
    # at zero, where an absolute value has the same answer either way.
    ("SIGN(x - 1)", "-1", "x :epsilon Real (0, 1)"),
    ("SIGN(x - 1)", "SIGN(x - 1)", "x :epsilon Real (0, 1]"),
    ("ABS(x - 1)", "1 - x", "x :epsilon Real (0, 1]"),
    ("MAX(x, 1)", "1", "x :epsilon Real (0, 1)"),
    ("MIN(x, 1)", "x", "x :epsilon Real (0, 1)"),
    # The bounds are asked before `MAX` is written out as the closed form of
    # 6.7, so a bound that decides which argument is the greater answers with
    # that argument; one that does not leaves the formula, whose bars are
    # undecidable over the same interval for the same reason.
    ("MAX(x, 1)", "ABS(x - 1)/2 + x/2 + 1/2", "x :epsilon Real (0, 2)"),
    # An integer variable is asked about through a real stand-in, sympy having
    # no answer at all about one bounded by a relation.
    ("ABS(n - 7)", "7 - n", "n :epsilon Integer [2, 5]"),
    ("ABS(n - 7)", "ABS(n - 7)", "n :epsilon Integer [2, 9]"),
    # The test of an `IF`, and the bare relation, which are asked the same
    # question and get the same answer: a bound that decides one decides both.
    ("IF(x < 1, a, b)", "a", "x :epsilon Real (0, 1)"),
    ("IF(x > 1, a, b)", "b", "x :epsilon Real (0, 1)"),
    ("IF(x <= 1, a, b)", "a", "x :epsilon Real (0, 1]"),
    ("IF(x = 1, a, b)", "b", "x :epsilon Real (0, 1)"),
    ("IF(x /= 1, a, b)", "a", "x :epsilon Real (0, 1)"),
    ("IF(x < 1, a, b)", "IF(x < 1, a, b)", "x :epsilon Real (0, 2)"),
    ("x < 1", "true", "x :epsilon Real (0, 1)"),
    ("x < 1", "x < 1", "x :epsilon Real (0, 2)"),
    # A power, an absolute value or a monotone function of a bounded quantity is
    # bounded too, by the interval arithmetic over its parts.
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
    # The arithmetic takes each occurrence of a variable on its own, so a bound
    # that holds only because two of them move together is not seen: `x - x^2` is
    # positive over `(0, 1)`, and stands. The answer to a question nobody can
    # settle is the expression.
    ("ABS(x - x^2)", "ABS(x^2 - x)", "x :epsilon Real (0, 1)"),
    # A step function is constant between consecutive integers.
    ("FLOOR(x)", "0", "x :epsilon Real (0, 1)"),
    ("FLOOR(x)", "0", "x :epsilon Real [0, 1)"),
    ("FLOOR(x)", "FLOOR(x)", "x :epsilon Real (0, 1]"),
    ("FLOOR(x)", "FLOOR(x)", "x :epsilon Real (-1, 1)"),
    ("FLOOR(x + 3/2)", "1", "x :epsilon Real (-1/2, 1/2)"),
    ("FLOOR(-x)", "-1", "x :epsilon Real (0, 1)"),
    ("FLOOR(LN(x))", "1", "x :epsilon Real (3, 7)"),
    # An inverse function undoes its function where the angle stays on one of its
    # branches, moved onto the principal branch.
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
    # One box over both variables, so a question about the pair is answerable
    # where neither bound answers it alone.
    box = declared("x :epsilon Real (0, 1)", "y :epsilon Real (2, 3)")
    assert simp("ABS(x - y)", box) == "y - x"
    assert simp("MAX(x, y)", box) == "y"
    assert simp("IF(x < y, a, b)", box) == "a"
    assert simp("x*y < 3", box) == "true"
    assert simp("x*y < 2", box) == "x*y < 2"


def test_an_interval_costs_nothing_where_nothing_is_declared():
    # The box is built only where a bound says something the symbol could not,
    # so an undeclared variable reaches none of this.
    assert simp("ABS(x - 1)") == "ABS(x - 1)"
    assert simp("MAX(x, 1)") == "ABS(x - 1)/2 + x/2 + 1/2"
    assert simp("IF(x < 1, a, b)") == "IF(x < 1, a, b)"


@pytest.mark.parametrize(("text", "expected", "context"), DOMAINS, ids=str)
def test_a_rewrite_needs_a_domain_that_justifies_it(text, expected, context):
    assert simp(text, context) == expected


def test_a_one_point_interval_is_a_value():
    assert simp("x + 1", declared("x :epsilon Real [7, 7]")) == "8"


#: A conditional an answer comes with, where the declaration decides its condition:
#: the case it picks, and the else case worked out where that is the one.
DECIDED_BY_A_DECLARATION = [
    ("SUM(x^k, k, 0, inf)", "1/(1 - x)", "x :epsilon Real (-1, 1)"),
    ("SUM(x^k, k, 0, inf)", "1/(1 - x)", "x :epsilon Real (0, 1)"),
    ("SUM(k*x^k, k, 0, inf)", "x/(1 - x)^2", "x :epsilon Real (0, 1)"),
    ("SUM(x^k/k, k, 1, inf)", "-LN(1 - x)", "x :epsilon Real (-1, 1)"),
    # Past one the series is the else case, and its terms grow without bound, so
    # it diverges: to `inf` where they are positive and `-inf` where negative.
    ("SUM(x^k, k, 0, inf)", "inf", "x :epsilon Real (1, inf)"),
    ("SUM(x^k, k, 0, inf)", "inf", "x :epsilon Real [2, inf)"),
    ("SUM(k*x^k, k, 0, inf)", "inf", "x :epsilon Real (1, inf)"),
    ("SUM(-x^k, k, 0, inf)", "-inf", "x :epsilon Real (1, inf)"),
    # Terms that alternate in sign have no limit to say which way the sum goes.
    (
        "SUM((-x)^k, k, 0, inf)",
        "SUM(x^k*(-1)^k, k, 0, inf)",
        "x :epsilon Real (1, inf)",
    ),
    # A declaration that does not decide the condition leaves it.
    (
        "SUM(x^k, k, 0, inf)",
        "IF(x < 1, 1/(1 - x), SUM(x^k, k, 0, inf))",
        "x :epsilon Real (0, 2)",
    ),
    # A power of a declared base that tends to zero or grows without bound.
    ("LIM(x^t, t, inf)", "0", "x :epsilon Real (0, 1)"),
    ("LIM(x^t, t, inf)", "inf", "x :epsilon Real (1, inf)"),
    ("LIM(x^t, t, -inf)", "0", "x :epsilon Real (1, inf)"),
    ("LIM(x^t, t, -inf)", "inf", "x :epsilon Real (0, 1)"),
    # An authored test is asked the same question.
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
    # The else case of a sum of `y*x^k` is the sum of `x^k`, which comes with the
    # same condition once more, and where the first fails so does the second.
    assert simp("SUM(y*x^k, k, 0, inf)") == (
        "y*IF(ABS(x) < 1, 1/(1 - x), SUM(x^k, k, 0, inf))"
    )


def test_a_recursion_guarded_by_a_declared_variable_stands():
    """A test the declaration leaves open is suspended as any other is, so a
    recursion over a declared variable does not unfold its untaken branch."""
    body = parse("IF(n = 0, 1, n*F(n - 1))", declaring("F"))
    functions = {"F": (("n",), body)}
    assert simp("F(5)", Context(functions=functions)) == "120"
    context = replace(declared("m :epsilon Integer [0, 3]"), functions=functions)
    assert simp("F(m)", context) == "IF(m = 0, 1, m*F(m - 1))"
    # Above two, the first tests are decided false and the recursion unfolds to the
    # first one that is not.
    context = replace(declared("m :epsilon Integer (2, inf)"), functions=functions)
    assert simp("F(m)", context) == "m*(m - 1)*(m - 2)*IF(m = 3, 1, (m - 3)*F(m - 4))"


#: A complex quantity times its conjugate is its squared modulus.
CONJUGATE_PAIRS = [
    ("CONJ(z)*z", "ABS(z)^2"),
    ("z*CONJ(z) + 1", "ABS(z)^2 + 1"),
    ("CONJ(z)^2*z", "CONJ(z)*ABS(z)^2"),
    ("CONJ(z)*z^3", "z^2*ABS(z)^2"),
    ("1/(z*CONJ(z))", "1/ABS(z)^2"),
    # A quotient pairs nothing.
    ("CONJ(z)/z", "CONJ(z)/z"),
]


@pytest.mark.parametrize(("text", "expected"), CONJUGATE_PAIRS, ids=str)
def test_a_conjugate_pair_is_a_squared_modulus(text, expected):
    assert simp(text, declared("z :epsilon Complex")) == expected


# -- absolute values and signs ------------------------------------------------

#: One real quantity written three ways. A product holding two of `ABS(u)`,
#: `SIGN(u)` and `u` says the same thing twice and comes back saying it once,
#: whichever of the three is the shorter way to say it.
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
    # The argument is whatever it is written about, and the rest of the product
    # is no part of the group it makes.
    ("ABS(SIN(x))*SIGN(SIN(x))", "SIN(x)", None),
    ("ABS(x - 1)/SIGN(x - 1)", "x - 1", None),
    ("y*ABS(x)*SIGN(x)", "x*y", None),
    ("ABS(x)*SIGN(y)", "ABS(x)*SIGN(y)", None),
    # One of the three by itself is what it is, and so is a bare power.
    ("ABS(x)", "ABS(x)", None),
    ("SIGN(x)", "SIGN(x)", None),
    ("x^3", "x^3", None),
    # A square is never negative but is zero at zero, which a sign tells apart, and
    # `SIGN(ABS(x))` is no simpler a way to say so.
    ("SIGN(x^2)", "SIGN(x^2)", None),
    ("SIGN(x^2 + 1)", "1", None),
    # A reciprocal has the sign of what it divides by, wherever it is defined; at
    # zero it is not, so `SIGN(1/x^2)` is 1 where `SIGN(x^2)` is not.
    ("SIGN(1/x)", "SIGN(x)", None),
    ("SIGN(x^(-3))", "SIGN(x)", None),
    ("SIGN(1/x^2)", "1", None),
    ("SIGN(1/ABS(x))", "1", None),
    ("SIGN(1/(x + y))", "SIGN(x + y)", None),
    ("SIGN(1/z)", "SIGN(1/z)", declared("z :epsilon Complex")),
    # Zero is the point the identities are a licence at rather than a fact -
    # `SIGN(0)` is zero, so the left side is undefined there and the right side
    # is not - and a declaration away from zero makes them facts.
    ("ABS(x)*SIGN(x)", "x", POSITIVE_X),
    ("SIGN(x)^2", "1", POSITIVE_X),
    ("x/ABS(x)", "1", POSITIVE_X),
    # Off the real line there is nothing to fold: `SIGN(z)` is `z/|z|` there,
    # whose square is not one and whose product with `ABS(z)` sympy will not
    # collect either.
    ("ABS(z)*SIGN(z)", "SIGN(z)*ABS(z)", declared("z :epsilon Complex")),
    ("ABS(z)/SIGN(z)", "ABS(z)/SIGN(z)", declared("z :epsilon Complex")),
    ("SIGN(z)^2", "SIGN(z)^2", declared("z :epsilon Complex")),
    ("z/ABS(z)", "z/ABS(z)", declared("z :epsilon Complex")),
    # An odd increasing function keeps the sign of a real argument, and an even
    # function does not see its absolute value. Neither holds off the real line.
    ("SIGN(ATAN(x))", "SIGN(x)", None),
    ("SIGN(SINH(x))", "SIGN(x)", None),
    ("SIGN(x^3)", "SIGN(x)", None),
    ("SIGN(ATAN(x)^3)", "SIGN(x)", None),
    ("SIGN(x^2)", "SIGN(x^2)", None),
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
    # Complementary arcs of one argument are a right angle. The tangent pair is
    # the one that turns with the sign of its argument, and answers the turn
    # itself where nothing decides the sign; a declaration that decides it, or
    # an argument that cannot be negative, gets the constant outright.
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
    # `ACOT(0)` is `pi/2`, so a square - which is a right angle's worth above
    # zero and `pi/2` at it - is the constant and not the turn.
    ("ACOT(x^2) + ATAN(x^2)", "pi/2", None),
    ("ACOT(x^2 + 1) + ATAN(x^2 + 1)", "pi/2", None),
    ("ACOT(3) + ATAN(3)", "pi/2", None),
    ("ASIN(x) + ACOS(y)", "ASIN(x) + ACOS(y)", None),
    # An arc of a ratio is the arc of a side of the right triangle the ratio
    # describes, however the hypotenuse was written. The root has a branch cut,
    # so a complex argument keeps the ratio it came in as.
    ("ASIN(x/SQRT(x^2 + 1))", "ATAN(x)", None),
    ("ASIN(-x/SQRT(x^2 + 1))", "-ATAN(x)", None),
    ("ASIN((x + 1)/SQRT(x^2 + 2*x + 2))", "ATAN(x + 1)", None),
    ("ATAN(x/SQRT(1 - x^2))", "ASIN(x)", None),
    ("ASIN(x/SQRT(x^2 + 1))", "ASIN(x/SQRT(x^2 + 1))", COMPLEX_X),
    ("ATAN(x/SQRT(1 - x^2))", "ATAN(x/SQRT(1 - x^2))", COMPLEX_X),
    # The reciprocal arcs are 6.4's definitions rather than identities over the
    # reals, so they need no domain: `ASEC(z)` is `ACOS(1/z)` whatever `z` is,
    # and `ACSC(z)` is `ASIN(1/z)`. Neither head is kept in an answer.
    ("ASEC(1/x)", "ACOS(x)", None),
    ("ACSC(1/x)", "ASIN(x)", None),
    ("ASEC(1/x)", "ACOS(x)", COMPLEX_X),
    ("ASEC(x)", "ACOS(1/x)", None),
    # Two angles a factor apart cancel once both are written about the angle
    # they are multiples of; a whole factor apart is `SIN(6*x)/SIN(3*x)` above,
    # which needs none of that.
    ("1/(1 + TAN(a)*TAN(a/2))", "COS(a)", None),
    ("SIN(45)", "SQRT(2)/2", Context(angle=Angle.DEGREE)),
    ("ASIN(1/2)", "30", Context(angle=Angle.DEGREE)),
    # `deg` is the constant pi/180, so degrees work in radian mode too.
    ("SIN(45 deg)", "SQRT(2)/2", None),
    ("SIN(2*x)", "2*SIN(x)*COS(x)", Context(trigonometry=Direction.EXPAND)),
    ("SIN(x)*COS(x)", "SIN(2*x)/2", Context(trigonometry=Direction.COLLECT)),
    ("COS(x)^2", "1 - SIN(x)^2", Context(trigpower=TrigPower.SINES)),
    ("SIN(x)^2", "1 - COS(x)^2", Context(trigpower=TrigPower.COSINES)),
    # The original has no secant, and writes an even reciprocal power of a
    # cosine over the tangent: `DIF(LN(COS(x)), x, 2)` is `- TAN(x)^2 - 1`
    # there. Only where the cosine is the whole of the term, and only for an
    # even power.
    ("1/COS(x)^2", "TAN(x)^2 + 1", None),
    ("1/COS(x)^4", "(TAN(x)^2 + 1)^2", None),
    ("SEC(x)^2", "TAN(x)^2 + 1", None),
    ("1/COS(x)^3", "1/COS(x)^3", None),
    ("SIN(x)/COS(x)^2", "SIN(x)/COS(x)^2", None),
]


@pytest.mark.parametrize(("text", "expected", "context"), TRIGONOMETRY, ids=str)
def test_trigonometry(text, expected, context):
    assert simp(text, context) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SIN(x) + COS(x)", "SQRT(2)*SIN(x + pi/4)"),
        ("3*SIN(x) + 4*COS(x)", "5*SIN(x + ATAN(4/3))"),
    ],
    ids=str,
)
def test_collect_writes_a_sine_and_a_cosine_as_one_phase_shifted_sine(text, expected):
    # Ours rather than fu's: `TR10i` only collects equal coefficients.
    assert simp(text, Context(trigonometry=Direction.COLLECT)) == expected


#: Arctangents of numbers add up through the Gaussian integers whose arguments they
#: are: `ATAN(p/q)` is the argument of `q + p*#i`, and a sum of them is a rational
#: multiple of pi exactly when the product lies on an axis or a diagonal.
ARC_TANGENT_SUMS = [
    # `ATAN(1)` is `pi/4` as it is written, and the other two make up the rest.
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
    # The part that adds to a multiple of pi does, and the rest stands beside it.
    ("ATAN(2) + ATAN(3) + ATAN(1/7)", "3*pi/4 + ATAN(1/7)", None),
    ("ATAN(1/2) + ATAN(1/3) + x", "x + pi/4", None),
    ("y*ATAN(1/2) + y*ATAN(1/3)", "pi*y/4", None),
    ("ATAN(1/2) + ATAN(1/3)", "45", Context(angle=Angle.DEGREE)),
    # Otherwise a plain sum is one arctangent only where its argument is no larger
    # than the ones it replaces.
    ("ATAN(1/3) + ATAN(1/7)", "ATAN(1/2)", None),
    ("ATAN(1/2) + ATAN(1/4)", "ATAN(1/2) + ATAN(1/4)", None),
    ("ATAN(2) + ATAN(4)", "ATAN(4) + ATAN(2)", None),
    ("3*ATAN(1/5)", "3*ATAN(1/5)", None),
    # `ATAN(1/u)` is `ACOT(u)` for a real `u`, and pairs with `ATAN(u)` the same way.
    ("ATAN(x) + ATAN(1/x)", "pi*SIGN(x)/2", None),
    ("ATAN(x/y) + ATAN(y/x)", "pi*SIGN(x/y)/2", None),
    ("2*ATAN(x) + 2*ATAN(1/x)", "pi*SIGN(x)", None),
    ("ATAN(x^2) + ATAN(1/x^2)", "pi/2", None),
    ("ATAN(z) + ATAN(1/z)", "ATAN(1/z) + ATAN(z)", declared("z :epsilon Complex")),
]


@pytest.mark.parametrize(("text", "expected", "context"), ARC_TANGENT_SUMS, ids=str)
def test_arctangents_of_numbers_add_up(text, expected, context):
    assert simp(text, context) == expected


#: Sines and cosines of whole multiples of one angle are reduced together, over the
#: single angle, and written the shortest of the ways that are unique: as a sum of
#: calls of multiples, or as a polynomial in one of the two functions.
MULTIPLE_ANGLES = [
    # `trigsimp` rewrites the last term alone and leaves the sum half done.
    ("SIN(x)^6 + COS(x)^6 + 3*SIN(x)^2*COS(x)^2", "1"),
    ("SIN(x)^4 - COS(x)^4", "-COS(2*x)"),
    ("COS(2*x) + 2*SIN(x)^2", "1"),
    ("SIN(3*x) + 4*SIN(x)^3", "3*SIN(x)"),
    ("SIN(2*x)*COS(x) - COS(2*x)*SIN(x)", "SIN(x)"),
    # A quotient that cancels to a polynomial is cancelled even where that is no
    # shorter, and is then a sum of multiples.
    ("SIN(3*x)/SIN(x)", "2*COS(2*x) + 1"),
    ("COS(3*x)/COS(x)", "2*COS(2*x) - 1"),
    ("SIN(5*x)/SIN(x)", "2*COS(2*x) + 2*COS(4*x) + 1"),
    ("(COS(x) + COS(3*x))/COS(2*x)", "2*COS(x)"),
    ("SIN(3*x)/SIN(x) + 1", "4*COS(x)^2"),
    # Nothing cancels in these, and nothing shorter is there to write.
    ("SIN(x)^4 + COS(x)^4", "COS(x)^4 + SIN(x)^4"),
    ("SIN(x)*COS(2*x)", "SIN(x)*COS(2*x)"),
    ("SIN(3*x)/SIN(y)", "SIN(3*x)/SIN(y)"),
]


@pytest.mark.parametrize(("text", "expected"), MULTIPLE_ANGLES, ids=str)
def test_calls_of_multiples_of_one_angle_reduce_together(text, expected):
    assert simp(text) == expected


#: A call of a small multiple of an arc is the algebraic function the
#: multiple-angle formulas make of it.
ARC_MULTIPLES = [
    ("SIN(2*ACOS(x))", "2*x*SQRT(1 - x^2)"),
    ("SIN(2*ASIN(x))", "2*x*SQRT(1 - x^2)"),
    ("COS(2*ASIN(x))", "1 - 2*x^2"),
    ("COS(3*ACOS(x))", "4*x^3 - 3*x"),
    ("SIN(2*ATAN(x))", "2*x/(x^2 + 1)"),
    ("TAN(2*ATAN(x))", "2*x/(1 - x^2)"),
    ("SIN(-2*ACOS(x))", "-2*x*SQRT(1 - x^2)"),
    # Past the bound the call stands.
    ("SIN(5*ACOS(x))", "SIN(5*ACOS(x))"),
]


@pytest.mark.parametrize(("text", "expected"), ARC_MULTIPLES, ids=str)
def test_a_call_of_a_multiple_of_an_arc_is_algebraic(text, expected):
    assert simp(text) == expected


#: Sums and products of trigonometric values of rational multiples of pi that are
#: rational or quadratic, as the minimal polynomial of the whole proves them to be.
TRIGONOMETRIC_CONSTANTS = [
    ("COS(pi/7) - COS(2*pi/7) + COS(3*pi/7)", "1/2"),
    ("COS(2*pi/7) + COS(4*pi/7) + COS(6*pi/7)", "-1/2"),
    ("COS(pi/9)*COS(2*pi/9)*COS(4*pi/9)", "1/8"),
    ("COS(pi/7)*COS(2*pi/7)*COS(3*pi/7)", "1/8"),
    ("SIN(pi/7)*SIN(2*pi/7)*SIN(3*pi/7)", "SQRT(7)/8"),
    ("TAN(pi/7)*TAN(2*pi/7)*TAN(3*pi/7)", "SQRT(7)"),
    ("TAN(pi/9)*TAN(2*pi/9)*TAN(4*pi/9)", "SQRT(3)"),
    ("COS(pi/11) + COS(3*pi/11) + COS(5*pi/11) + COS(7*pi/11) + COS(9*pi/11)", "1/2"),
    # Beside a variable, in a sum or a product.
    ("x + COS(pi/9)*COS(2*pi/9)*COS(4*pi/9)", "x + 1/8"),
    ("x*COS(pi/9)*COS(2*pi/9)*COS(4*pi/9)", "x/8"),
    # A value of higher degree is no simpler number, and stands.
    ("SIN(pi/7) + SIN(2*pi/7)", "SIN(2*pi/7) + SIN(pi/7)"),
    ("2*COS(pi/7)", "2*COS(pi/7)"),
    ("SIN(1)*COS(2)", "SIN(1)*COS(2)"),
]


@pytest.mark.parametrize(("text", "expected"), TRIGONOMETRIC_CONSTANTS, ids=str)
def test_a_trigonometric_constant_is_written_as_the_number_it_is(text, expected):
    assert simp(text) == expected


#: An arcsine or arccosine of a real number past one in size is complex, and is
#: written in rectangular form, as `ACOSH(1/2)` and `ATANH(2)` are.
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
    # The logarithm falls without bound as its argument falls to zero, so at zero
    # it is minus infinity and not the unsigned infinity of a pole; a base below
    # one turns the sign round.
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
    # Auto is 6.2's mixed direction: a sum of logarithms collects into one, and
    # a logarithm of a power comes apart. The second is the direction the
    # longer answer lies in, and it is taken anyway - what `LN(256)` is to
    # `8*LN(2)` is what `SQRT(12)` is to `2*SQRT(3)`. Collect is the setting
    # that keeps the power where it stands.
    ("LN(256)", "8*LN(2)", None),
    ("8*LN(2)", "8*LN(2)", None),
    ("LN(256)", "LN(256)", Context(logarithm=Direction.COLLECT)),
    ("LN(4/9)", "2*LN(2/3)", None),
    # Nothing comes out of an argument that is no whole power of anything, and
    # the sum still collects over one that is not: Derive answers `LN(40)`.
    ("LN(12)", "LN(12)", None),
    ("3*LN(2) + LN(5)", "LN(40)", None),
    # The rule wants a nonnegative base, or an exponent within one of zero.
    ("LN(x^8)", "LN(x^8)", None),
    ("LN(x^8)", "8*LN(x)", declared("x :epsilon Real [0, inf)")),
    ("LN(SQRT(z))", "LN(z)/2", None),
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
    # Powers of one base collect in Auto too, and not because they are shorter:
    # `DIF(x^n, x)` is `n*x^n/x` to sympy, which counts the same and is a form
    # the original does not write.
    ("x^n/x", "x^(n - 1)", None),
    ("DIF(x^n, x)", "n*x^(n - 1)", None),
    # A power to a logarithm is the exponential of the product of two, whichever
    # base is written, and is written over the smaller.
    ("2^LN(3) - 3^LN(2)", "0", None),
    ("2^LN(5) - 5^LN(2)", "0", None),
    ("3^LN(2)", "2^LN(3)", None),
    ("#e^(LN(2)*LN(3))", "2^LN(3)", None),
    ("5^LN(2)/2^LN(5)", "1", None),
    ("x^LN(2)", "x^LN(2)", None),
    # An even power of a real argument is that power of its absolute value. Expand
    # takes the logarithm apart over it; Auto keeps the shorter `LN(x^2)`, as the
    # rule of 6.2 does not reach it, and so does a complex argument.
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


# -- the branch setting -------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected", "branch"),
    [
        # The principal cube root, in the rectangular form 4.5 promises: Derive
        # answers `1 + SQRT(3)*#i` and approximates it `1 + 1.73205*#i`.
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


#: Section 4.5's rectangular form, and the three things it does not reach. A
#: base that is no number has no rectangle until something says what it is; a
#: fifth root buys one with a radical inside a radical, and a seventh with a
#: cosine, which is the polar form the rule is about avoiding.
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


#: A power to a complex exponent is its principal value, `#e^(z*LN(b))`, and an
#: exponential turning through a rational multiple of pi is written in rectangular
#: form where the cosine and the sine are plainly written, as a root of -1 is above.
COMPLEX_CONSTANTS = [
    ("#i^#i", "#e^(-pi/2)"),
    ("(-1)^#i", "#e^(-pi)"),
    ("#i^(2*#i)", "#e^(-pi)"),
    ("#i^(1 + #i)", "#i*#e^(-pi/2)"),
    # Built again around its value, the logarithm of it is its exponent.
    ("LN(#i^#i)", "-pi/2"),
    ("#i^#i^#i", "#i^#e^(-pi/2)"),
    ("(2*#i)^#i", "#e^(-pi/2 + #i*LN(2))"),
    ("(1 + #i)^#i", "#e^(-pi/4 + #i*LN(2)/2)"),
    # A positive base keeps its power, the shorter spelling of the same exponential.
    ("2^#i", "2^#i"),
    ("pi^#i", "pi^#i"),
    ("#e^(#i*pi/3)", "1/2 + SQRT(3)*#i/2"),
    ("#e^(#i*pi/4)", "SQRT(2)/2 + SQRT(2)*#i/2"),
    ("#e^(2*pi*#i/3)", "-1/2 + SQRT(3)*#i/2"),
    ("#e^(-#i*pi/4)", "SQRT(2)/2 - SQRT(2)*#i/2"),
    ("#e^(1 + #i*pi/3)", "#e*(1 + SQRT(3)*#i)/2"),
    ("#e^(pi*#i/3 + LN(2))", "1 + SQRT(3)*#i"),
    ("#e^(#i*pi/3)*x", "(1 + SQRT(3)*#i)*x/2"),
    # The fifth and seventh parts of a turn are no rectangle plainly written, and a
    # turn through one radian is no rational part of one.
    ("#e^(#i*pi/5)", "#e^(#i*pi/5)"),
    ("#e^(#i*pi/7)", "#e^(#i*pi/7)"),
    ("#e^#i", "#e^#i"),
    # A cosine or a sine of a complex number is taken apart over its real angle.
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


#: A root of a polynomial that is a perfect power is taken, with the bars a real
#: variable needs and no more: the intervals settle them, a positive base needs none,
#: and a complex variable has no real absolute value to write.
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
    # The principal cube root of a negative cube is no real number, and a negative
    # radicand and one that is no power have no root to take.
    ("(x^3 + 3*x^2 + 3*x + 1)^(1/3)", "(x^3 + 3*x^2 + 3*x + 1)^(1/3)", None),
    ("SQRT(-x^2 - 2*x - 1)", "SQRT(-x^2 - 2*x - 1)", None),
    ("SQRT(x^2 + 1)", "SQRT(x^2 + 1)", None),
]


@pytest.mark.parametrize(("text", "expected", "context"), PERFECT_POWERS, ids=str)
def test_a_root_of_a_perfect_power_is_taken(text, expected, context):
    assert simp(text, context) == expected


def test_an_interval_goes_on_being_computed_with():
    """The head is live, because sympy's arithmetic over it is.

    Every value in the range is carried through the operation and the answer is
    the range of the results, which is why the square is `INTERVAL(0, 1)`: the
    square of the range is not the range of the squares. Differentiation and
    integration pass it through the same way, so a limit like this can be used
    for something rather than only looked at.
    """
    assert simp("INTERVAL(-1, 1) + 1") == "INTERVAL(0, 2)"
    assert simp("2*INTERVAL(-1, 1)") == "INTERVAL(-2, 2)"
    assert simp("INTERVAL(-1, 1)^2") == "INTERVAL(0, 1)"
    assert simp("SIN(LIM(SIN(1/x), x, 0))") == "INTERVAL(-SIN(1), SIN(1))"
    assert simp("DIF(y*LIM(SIN(1/x), x, 0), y)") == "INTERVAL(-1, 1)"
    assert simp("INT(LIM(SIN(1/x), x, 0), y)") == "INTERVAL(-1, 1)*y"


#: What an undecidable four-argument `IF` leaves standing where the rest of the
#: pipeline expects an expression: the unknown clause is whatever was written.
#: Both shapes reach the gated rewrites, and both are in Derive's own utility
#: files - the truth value in `SYMMETRIC_TEST_1` and `B_TYPE`, the relation in
#: `CLAIRAUT`, all three in ODE.MTH.
NOT_EXPRESSIONS = [
    ("IF(v = [] OR v = [a], true, false, false)", "false"),
    ("IF(x > 0, y = 1, y = 2, y = 3)", "y = 3"),
]


@pytest.mark.parametrize(("text", "expected"), NOT_EXPRESSIONS, ids=str)
def test_a_rewrite_is_not_asked_about_something_that_is_no_expression(text, expected):
    """A truth value has no numerator and no denominator.

    Every gated rewrite is offered whatever the conditional left behind, and
    that is not always an expression. Asking sympy to split a truth value into
    a ratio builds a `Mul` out of a `BooleanFalse`, which it warns about and
    says it will one day refuse - so the answer must arrive without a word from
    sympy. A warning is not an exception, and the `try` around every rewrite
    would not have caught it.
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


#: Commands the engine has no mathematics for: calls to functions declared without
#: a body, the functions below. Each must come back exactly as it went in, so that a
#: worksheet holding one is not damaged by simplifying it.
OPAQUE = [
    "MY_FUNCTION(x, 2)",
    "ANOTHER(1)",
]
OPAQUE_FUNCTIONS = ("MY_FUNCTION", "ANOTHER")


@pytest.mark.parametrize("text", OPAQUE, ids=str)
def test_what_the_engine_has_no_mathematics_for_passes_through(text):
    assert simp(text, state=declaring(*OPAQUE_FUNCTIONS)) == text


#: Calls to a function the engine does know, written so that they name nothing
#: it can answer: a fit with no parametric variable to solve for, data of a
#: width the label vector does not match, and a dependence on the parameter
#: that is not linear. Each comes back as it went in rather than being answered
#: from a reading nobody asked for.
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
    """A root sum names its variable in both the polynomial and the summand.

    Everywhere else the binding stops after the first argument: a sum's limits
    are outer expressions and are substituted, and so is the index a root is
    counted by. But a root sum runs over the roots of a polynomial written in
    the very variable its summand names, so an assigned `t` must reach neither
    half - otherwise the answer to an integral becomes a different expression
    the second time it is simplified.
    """
    context = Context(assignments={"t": parse("5")})
    assert simp("ROOT_SUM(t^3 + t + 1, t, t*LN(x - t))", context) == (
        "ROOT_SUM(t^3 + t + 1, t, t*LN(x - t))"
    )
    # The cases the exception must not swallow: an index, a limit, a free name.
    assert simp("ROOT_OF(z^5 - z - 1, z, t - 5)", context) == "ROOT_OF(z^5 - z - 1, z, 0)"
    assert simp("SUM(k, k, 1, t)", context) == "15"
    assert simp("SUM(t^2, t, 1, 3)", context) == "14"
    assert simp("t + 1", context) == "6"


def test_a_defined_function_is_replaced_by_its_body():
    context = Context(functions={"ACCELERATION": (("f", "m"), parse("f/m"))})
    assert simp("ACCELERATION(6, 2)", context) == "3"
    # Partial application: the parameter nobody supplied stays as its name.
    assert simp("ACCELERATION(6)", context) == "6/m"


def test_a_counted_sum_is_written_out_for_the_recursion_under_it():
    """A range of known length is written out even where a body is still owed.

    `NONZERO_ROWS` of VECTOR.MTH counts the rows of a matrix by summing a
    recursion over `m SUB n_`, and with the index still a symbol that recursion
    has nothing to stop at: it runs off the end of a vector whose length is
    unknown, a level per pass, and never arrives. Written out first, each term
    recurses over one row and every one of them ends.
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


def test_two_spellings_of_one_variable_are_one_variable():
    """Case-insensitively, `x` and `X` name the same thing.

    Which spelling a line carries depends on what had been declared when it was
    read: `X := ELEMENT(x, 1)` declares `X` and mentions `x` before that
    declaration exists, so the tree holds both. The session's symbol table is
    what says they are one, and the answer is written under the spelling it
    records.
    """
    state = ParseState()
    parsed = parse_expression("[a:=, b:=, c:=, X := ELEMENT(x, 1)]", state)
    for declaration in parsed.declarations:
        state.declare(declaration)
    answer = simplify(parsed.node, Context(), state)
    assert answer.text == "[a :=, b :=, c :=, X := ELEMENT(X, 1)]"
    # And with no symbol table there is nothing to resolve them against, so
    # the tree stands as it was read.
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
    # The session simplifies what the user has highlighted, not only whole
    # lines, so nothing in the engine may assume it has a whole entry.
    node = parse("1 + (2*x + 3*x)")
    assert simplify(node.children[1], Context()).text == "5*x"


def test_a_result_carries_a_tree_whose_spans_index_its_text():
    result = simplify(parse("(x + 1)*(x - 1) + 1"), Context())
    assert result.text[result.node.start : result.node.end] == result.text
    assert to_sexpr(result.node) == to_sexpr(parse(result.text))


# -- from Soft Warehouse's own demo scripts -----------------------------------
#
# `artifacts/sessions/derive-3.14-dos/*.DMO` are the annotated demonstrations
# that shipped with the original, and between them they are a capability test
# for Simplify. These are their cases, in author notation, with what this
# engine answers.

DEMO_NUMBERS = [
    ("5 (9 - 3) / 2", "15"),
    ("(2/27)^(2/3)", "2^(2/3)/9"),
    ("SQRT(-1)", "#i"),
    ("(1 + 2 #i) (3 + #i)", "1 + 7*#i"),
    # A complex denominator is rationalized.
    ("(2 + #i) / (#i + 1)", "3/2 - #i/2"),
    ("SQRT(55/36 + 4/3 #i)", "4/3 + #i/2"),
    ("inf + inf + 5", "inf"),
    # Two nested radicals that only denest one at a time.
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
    # The fifth power common to both terms is collected back out, and the
    # cofactor - not the factors' order, which is sympy's - is the answer.
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
    # The statistics the demo closes on. A numeric coefficient distributes, so
    # the average of three names is written termwise where the manual writes it
    # as one quotient, and the demo's exact average of 1/1 through 1/10 is the
    # tenth harmonic number over 10.
    ("AVERAGE (x, y, z)", "x/3 + y/3 + z/3"),
    ("AVERAGE (VECTOR (1/k, k, 1, 10))", "7381/25200"),
    ("RMS ([2, 3, 5])", "SQRT(114)/3"),
    # The unbiased sample variance of two things, and its square root, which
    # comes out of the radical because the variance is a square.
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
    # A numeric coefficient distributes rather than standing outside the sum,
    # so that the text reads back as the expression it prints.
    ("INT (x^2, x, a, b)", "b^3/3 - a^3/3"),
    ("INT (1/SQRT x, x, 0, b^2)", "2*ABS(b)"),
    ("INT (INT (x y, y, 0, SQRT (r^2 - x^2)), x, 0, r)", "r^4/8"),
    # The constant of integration is the one that makes the answer continuous
    # at `a = 0`, where Derive's `SIN(a*x^3 + b)/(3*a)` is undefined.
    ("INT (x^2 COS (a x^3 + b), x)", "(SIN(a*x^3 + b) - SIN(b))/(3*a)"),
    ("TAYLOR (#e^x, x, 0, 5)", "x^5/120 + x^4/24 + x^3/6 + x^2/2 + x + 1"),
    ("TAYLOR (LN COS (a x), x, 0, 7)", "-a^6*x^6/45 - a^4*x^4/12 - a^2*x^2/2"),
    # The closed form keeps the shape the summation formula has, which is the
    # factored one the demo's own screen shows.
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
    # `.` is the matrix product wherever the shapes conform; the dot product is
    # the case of it the operator is named after.
    ("[[a,b],[c,d]] . [[x],[y]]", "[[a*x + b*y], [c*x + d*y]]"),
    ("[[a,b],[c,d]] . [[a,b],[c,d]]^(-1)", "[[1, 0], [0, 1]]"),
    (
        "[[a,b],[c,d]]^(-1) . [[e],[f]]",
        "[[(d*e - b*f)/(a*d - b*c)], [(a*f - c*e)/(a*d - b*c)]]",
    ),
    # The outer product, which the demo builds out of the same operator.
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
    # A singular system, and a consistent one: the zero row leaves the second
    # unknown arbitrary, and the row above it reads `x + 2*y = 3`.
    ("ROW_REDUCE([[2,4],[3,6]],[[6],[9]])", "[[1, 2, 3], [0, 0, 0]]"),
    ("CHARPOLY([[a,b],[b,a]],z)", "z^2 - 2*a*z + a^2 - b^2"),
    # No variable given, so the answer is written in `w`, the manual's default.
    # Derive lists these the other way round; the pair is the same pair.
    ("EIGENVALUES([[a,b],[b,a]])", "[w = a - b, w = a + b]"),
    # The demo's vector calculus, in the default Cartesian x, y, z. It closes by
    # cross-checking itself: the vector it takes the potential of is the curl
    # computed two lines above, and the answer is the vector that curl was of.
    # The potential comes out in the engine's term order, leading term first,
    # where Derive writes the same sum as `x + y^2 + z^3`.
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

#: One expression per node kind the parser can build. `PARAMS` and `INTERVAL`
#: ride along inside the definition and the declaration that own them.
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
    """Whatever the engine has no mathematics for comes back intact.

    Not a claim that any of these compute anything - most do not yet. The
    claim is that Simplify is total over them: no name in the inventory can
    make it raise, and none of them loses its arguments on the way through.
    """
    text = f"{name}(x, 2)"
    once = simplify(parse(text), Context())
    assert simplify(once.node, Context()).text == once.text


# -- invariants over every case above -----------------------------------------

#: Inputs that take seconds to give up on and come back as written. That a standing
#: head prints back as itself is asked of many cheaper cases.
SLOW_TO_STAND = {"INT(SIN(m*x)*SIN(2*x), x, 0, pi)"}

#: Every input this module and its siblings simplify, with the context each uses, but
#: for those above.
EVERY_CASE = [
    *((text, None) for text, _ in ARITHMETIC + RATIONAL + RADICALS + COMBINATORIAL),
    *((text, context) for text, _, context in DOMAINS if text not in SLOW_TO_STAND),
    *((text, context) for text, _, context in TRIGONOMETRY),
    *((text, context) for text, _, context in LOGARITHMS),
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


#: The one shape a second Simplify is allowed to change, and the test after the
#: next says what it changes it to.
HEADED = {text for text, _ in TRUTH_TABLES}


@pytest.mark.parametrize(("text", "context"), EVERY_CASE, ids=str)
def test_a_result_is_a_fixed_point_of_printing_and_of_simplify(text, context):
    """Print, parse, print: the second text is the first. And simplify, simplify:
    the second answer is the first.

    The first is the invariant the worksheet rests on. A result is stored as text
    and read back later, and an answer that changed on the way through would make
    the entry the user is looking at a different expression from the one the label
    refers to. The second is what "sufficiently simple" means: a second pass that
    found more to do would mean the first stopped early.

    The two are asked of one answer, since the answer is most of what either costs
    to ask.
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
    """A truth table names its columns with the expressions they are about, and
    a name is not a claim: the heading stands unsimplified over the values it
    was worked out into. That holds while the heading is inside the table. Once
    the table is on the worksheet it is a matrix of expressions like any other,
    and a second Simplify answers `p XOR q` there as it would anywhere."""
    once = simplify(parse("TRUTH_TABLE(p, q, p XOR q)"), Context())
    assert once.text.startswith("[[p, q, p XOR q], ")
    again = simplify(once.node, Context())
    assert again.text.startswith("[[p, q, NOT p AND q OR p AND NOT q], ")


#: The expressions the corpus holds that settle only on a second pass are
#: recorded in `test_simplify_corpus.py`, beside the sweep that finds them.
