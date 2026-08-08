"""What Simplify must produce.

Author text in, author text out: each case is written the way a user would type
it and asserted against the text the worksheet would show, because that is the
whole of what the command promises. The cases are inline rather than in a corpus
file so that a failure names the mathematics it is about.

Where the engine's form differs from the manual's printed one - term order, an
equivalent spelling - the engine's form is what is recorded, and the case says
so. Nothing here is weakened to make a test pass: an expected value is either
what Derive prints or a hand-checked equal of comparable simplicity.
"""

from __future__ import annotations

import math
import warnings
from fractions import Fraction

import pytest
from sexpr import to_sexpr

from rederive.engine.computing import (
    Context,
    Domain,
    DomainKind,
    approx,
    domain_of_node,
    from_sympy,
    simplify,
    to_sympy,
)
from rederive.engine.context import (
    Angle,
    Branch,
    Direction,
    Notation,
    Precision,
    TrigPower,
)
from rederive.model.expr import Kind
from rederive.syntax import ParseState, VariableDeclaration, parse_expression
from rederive.syntax.names import BUILTIN_FUNCTIONS


def parse(text, state=None):
    return parse_expression(text, state or ParseState()).node


def simp(text, context=None, state=None):
    """`text` simplified, as the worksheet would show it."""
    return simplify(parse(text, state), context or Context(), state).text


def declared(*declarations, **settings):
    """A context whose domains are what these declarations say."""
    domains = {}
    for declaration in declarations:
        name, domain = domain_of_node(parse(declaration))
        domains[name] = domain
    return Context(domains=domains, **settings)


# Built the way a session builds them: taking the precision carries the
# notation with it, so approximate arithmetic is read in the style that
# suits it rather than as the ratios it is made of.
APPROXIMATE = Context().with_precision(Precision.APPROXIMATE)
MIXED = Context().with_precision(Precision.MIXED)
# Exact arithmetic read as digits, which is where an approximation made inside
# an exact line shows what it is worth.
DECIMAL = Context(notation=Notation.DECIMAL)
COMPLEX_X = Context(domains={"x": Domain(DomainKind.COMPLEX)})
POSITIVE_X = declared("x :epsilon Real (0, inf)")


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
        "SUM(COMB(n, k), k, 0, n)",
        "IF(n > 0 OR -1 < n <= 0, 2^n, SUM(COMB(n, k), k, 0, n))",
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
    # As far as the reasoning reaches, which is about as far as the bounds are
    # linear in. A square of a bounded variable is bounded and sympy does not
    # see it; the answer to a question nobody can settle is the expression.
    ("ABS(x^2 - 1)", "ABS(x^2 - 1)", "x :epsilon Real (0, 1)"),
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


# -- logarithms and exponentials ----------------------------------------------

LOGARITHMS = [
    ("EXP(LN(x))", "x", None),
    ("LN(#e^3)", "3", None),
    ("#e^(#i*pi)", "-1", None),
    ("LOG(8, 2)", "3", None),
    ("#e^(2*LN(x))", "x^2", None),
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
]


@pytest.mark.parametrize(("text", "expected", "context"), LOGARITHMS, ids=str)
def test_logarithms_and_exponentials(text, expected, context):
    assert simp(text, context) == expected


# -- the functions that are spellings for something else ----------------------

BY_DEFINITION = [
    # 6.4: the two reciprocal circular functions.
    ("SEC(z)", "1/COS(z)"),
    ("CSC(z)", "1/SIN(z)"),
    # 6.5: every hyperbolic is an exponential.
    ("SINH(z)", "#e^z/2 - #e^(-z)/2"),
    ("COSH(z)", "#e^z/2 + #e^(-z)/2"),
    ("TANH(z)", "(#e^(2*z) - 1)/(#e^(2*z) + 1)"),
    ("COTH(z)", "(#e^(2*z) + 1)/(#e^(2*z) - 1)"),
    ("SECH(z)", "2*#e^z/(#e^(2*z) + 1)"),
    ("CSCH(z)", "2*#e^z/(#e^(2*z) - 1)"),
    # 6.6: every inverse hyperbolic is the logarithm that inverts it.
    ("ATANH(z)", "LN(z + 1)/2 - LN(1 - z)/2"),
    ("ASINH(z)", "LN(SQRT(z^2 + 1) + z)"),
    # A circular function beside a hyperbolic one keeps the form it was written
    # in: only the head that has a definition is written out by it.
    ("SIN(z) + SINH(z)", "#e^z/2 - #e^(-z)/2 + SIN(z)"),
    ("SIN(z)", "SIN(z)"),
]


@pytest.mark.parametrize(("text", "expected"), BY_DEFINITION, ids=str)
def test_the_functions_that_are_spellings_for_something_else(text, expected):
    assert simp(text) == expected


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


# -- the calculus heads -------------------------------------------------------

CALCULUS = [
    ("DIF(SIN(a*x^2), x)", "2*a*x*COS(a*x^2)"),
    ("DIF(x^2, x, 2)", "2"),
    # `F` was never given a body, so it differentiates symbolically.
    ("DIF(F(x)^3, x)", "3*F(x)^2*DIF(F(x), x)"),
    ("INT(x*SIN(x), x)", "SIN(x) - x*COS(x)"),
    ("INT(1/x, x, 1, 2)", "LN(2)"),
    ("INT(1/x^2, x, 1, inf)", "1"),
    ("SUM(1/k^2, k, 1, 5)", "5269/3600"),
    ("SUM(k^2, k, 1, m)", "m*(m + 1)*(2*m + 1)/6"),
    ("PRODUCT(n^2, n, 1, m)", "m!^2"),
    # A third argument that is a vector names the values the index takes
    # instead of the ends of a range, for both heads.
    ("SUM(k^2, k, [2, 3, 5, 7, 11])", "208"),
    ("PRODUCT(k^2, k, [2, 3, 5, 7, 11])", "5336100"),
    ("LIM(SIN(x)/x, x, 0)", "1"),
    ("LIM(SIGN(x), x, 0, 1)", "1"),
    ("LIM(SIGN(x), x, 0, -1)", "-1"),
    # A vector of variables against a vector of points is the limits taken one
    # after another, which is the substitution `x = 0` alone cannot make.
    ("LIM(x^2 + y^2, [x, y], [2, 3])", "13"),
    ("LIM(SIN(x)/x + y, [x, y], [0, 1])", "2"),
    # The order is the maximum degree, and a derivative that simplifies to zero
    # at the expansion point costs a term: this one has no even powers.
    ("TAYLOR(c*SIN(x), x, 0, 6)", "c*x^5/120 - c*x^3/6 + c*x"),
    ("TAYLOR(1/(1 - x), x, 0, 3)", "x^3 + x^2 + x + 1"),
]


@pytest.mark.parametrize(("text", "expected"), CALCULUS, ids=str)
def test_a_calculus_head_is_evaluated(text, expected):
    assert simp(text) == expected


SUMMATION_FORMULAS = [
    ("SUM(k, k, 0, n)", "n*(n + 1)/2"),
    ("SUM(k^2, k, 1, n)", "n*(n + 1)*(2*n + 1)/6"),
    ("SUM(k^3, k, 0, n)", "n^2*(n + 1)^2/4"),
    ("SUM(k(k + 1), k, 1, n)", "n*(n + 1)*(n + 2)/3"),
    ("SUM(a k + b, k, 0, n)", "(a*n + 2*b)*(n + 1)/2"),
    # Longer factored than expanded, and factored all the same: the shape is
    # the formula's and not an economy. Derive prints this one as
    # `n*(n + 1)*(6*n^3 + 9*n^2 + n - 1)/30`, which is the same product one
    # factor short.
    ("SUM(k^4, k, 1, n)", "n*(n + 1)*(2*n + 1)*(3*n^2 + 3*n - 1)/30"),
    # A vector is summed element by element, and each element is a formula.
    ("SUM([k, k^2], k, 0, n)", "[n*(n + 1)/2, n*(n + 1)*(2*n + 1)/6]"),
]


@pytest.mark.parametrize(("text", "expected"), SUMMATION_FORMULAS, ids=str)
def test_a_summation_formula_keeps_the_shape_it_comes_in(text, expected):
    """Derive's closed forms for a sum are factored, and Simplify leaves them so.

    Sympy hands back Faulhaber's polynomial multiplied out, and writing it as
    the original writes it is what this is. Not a factoring rule: the same
    polynomial authored by hand is left exactly as it was authored, which is
    what `test_a_polynomial_is_not_factored_for_having_a_common_factor` holds.
    """
    assert simp(text) == expected


def test_a_polynomial_is_not_factored_for_having_a_common_factor():
    # Factoring is the Factor command's business. The formula above keeps its
    # own shape; nothing gives this one a shape it did not have.
    assert simp("n^2/2 + n/2") == "n^2/2 + n/2"
    assert simp("n^2 + n") == "n^2 + n"


def test_a_closed_form_inside_a_larger_answer_is_the_larger_answer_s_to_shape():
    # The formula is kept where it stands whole, and nowhere else: what it is
    # part of is simplified as anything else is.
    assert simp("SIN(SUM(k, k, 0, n))") == "SIN(n*(n + 1)/2)"
    assert simp("SUM(k, k, 0, n)/n") == "n/2 + 1/2"
    assert simp("SUM(k, k, 0, n) - n^2/2") == "n/2"


def test_an_integral_a_rewrite_leaves_behind_is_evaluated_too():
    # Cancelling this integrand splits it in two, and the part that is an
    # integral of `x` is evaluated as well: an answer still holding
    # `INT(x, x)` would not be a simplified one.
    assert simp("INT((x*G(x) + F(x))/G(x), x)") == "INT(F(x)/G(x), x) + x^2/2"


def test_an_integral_that_will_not_evaluate_survives_as_itself():
    # Derive leaves the INT node standing, and so must we: an error here would
    # lose the entry the user authored.
    assert simp("INT(SIN(x)/LN(x), x)") == "INT(SIN(x)/LN(x), x)"


def test_a_taylor_polynomial_that_does_not_exist_survives_as_itself():
    """No polynomial, no answer.

    The derivatives of `SQRT(x)` at zero are not finite, so there is no first
    order Taylor polynomial; Derive answers such a case with complex infinity
    or `?`, and the expansion the head stands for comes back instead. What
    sympy has for it - the series `SQRT(x)`, which is no polynomial - would be
    an answer to a different question.
    """
    assert simp("TAYLOR(SQRT(x), x, 0, 1)") == "TAYLOR(SQRT(x), x, 0, 1)"
    assert simp("TAYLOR(LN(x), x, 0, 2)") == "TAYLOR(LN(x), x, 0, 2)"
    # An order that is no order to expand to leaves the head alone as well.
    assert simp("TAYLOR(SIN(x), x, 0, n)") == "TAYLOR(SIN(x), x, 0, n)"


def test_a_derivative_a_substitution_binds_is_not_evaluated():
    """What sympy holds under a `SUBS` is held on purpose.

    Differentiating a limit whose endpoint moves gives the slope of `F` at a
    point, written as a derivative over a bound variable and a substitution
    into it. That derivative looks like the derivative of a constant and is
    not one, so nothing inside the substitution is evaluated - and the answer
    is the same on the second pass as on the first.
    """
    once = simplify(parse("DIF(LIM(F(y), y, 2*x - y), y)"), Context())
    assert "SUBS(DIF(LIM(F(y), xi_2, 2*x - y), xi_2), [xi_2], [y])" in once.text
    assert simplify(once.node, Context()).text == once.text


def test_a_conditional_whose_branches_are_vectors_simplifies_elementwise():
    # The clause taken is a vector, and a vector is simplified element by
    # element whether it was written or arrived at.
    taken = "IF(a > 0, [1, 2], [3, 4], [SQRT(b)*SQRT(b), 2*b/b])"
    assert simp(taken) == simp("[SQRT(b)*SQRT(b), 2*b/b]") == "[b, 2]"


def test_an_antiderivative_is_written_with_no_constant_added():
    """An indefinite integral is one antiderivative out of many.

    Which one is printed is a choice, and the original's choice is the one with
    nothing constant added. The original answers this line - the quadratic
    formula's
    root integrated over its own middle coefficient - with

        b*(SQRT(b^2 - 4*a*c) - b)/(4*a) - c*LN(SQRT(b^2 - 4*a*c) + b)

    where sympy's antiderivative is over `LN(2*b + 2*SQRT(b^2 - 4*a*c))`. The
    two differ by `c*LN(2)`, which is a constant of integration and no part of
    the answer. The terms below are the original's; only the order and the `b`
    left inside the numerator are the engine's - the original's two terms are
    three here, and a sum of three keeps the leading minus the original's pair
    turns round.
    """
    answer = simp("INT((SQRT(b^2 - 4*a*c) - b)/(2*a), b)")
    assert answer == (
        "-c*LN(SQRT(b^2 - 4*a*c) + b) + b*SQRT(b^2 - 4*a*c)/(4*a) - b^2/(4*a)"
    )


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


def test_a_case_split_between_two_antiderivatives_is_kept():
    """Where the cases are different functions the split is doing real work.

    `SIN(a*x^3 + b)/(3*a)` is an antiderivative for every `a` it is defined for,
    which is every `a` but zero, and at zero the integrand is `x^2*COS(b)`. So
    the case that holds unconditionally is the one the split has to be checked
    against, and here it is `x^3*COS(b)/3`, which is no antiderivative anywhere
    else. The original prints the first case alone; the conditional answer is this
    engine's, and it is the more informative of the two.
    """
    assert simp("INT(x^2*COS(a*x^3 + b), x)") == (
        "IF(a /= 0, SIN(a*x^3 + b)/(3*a), x^3*COS(b)/3)"
    )


def test_an_antiderivative_that_does_not_differentiate_back_is_not_taken():
    """Sympy's Risch implementation answers some of these wrongly.

    `INT(1/(x^2 + a), x)` comes back zero over a real `a` and
    `INT(x^2/(x^2 + a), x)` comes back `x`, and neither is refused. An answer
    that provably does not differentiate back to the integrand is no answer, so
    the integral is asked again by the rules a calculus course teaches, which do
    not reach that algorithm - and what those give does differentiate back, as
    the two lines below are here to say.

    Only a proven no counts, which is what keeps the answers that cannot be
    checked either way: nothing differentiates a hypergeometric series back to
    `1/SQRT(1 - x^4)`, and refusing what cannot be checked would throw away the
    only answer there is.
    """
    assert simp("DIF(ATAN(x/SQRT(a))/SQRT(a), x)") == "1/(x^2 + a)"
    assert simp("DIF(x - SQRT(a)*ATAN(x/SQRT(a)), x)") == "x^2/(x^2 + a)"
    assert simp("INT(1/SQRT(1 - x^4), x)").startswith("x*HYPER(")


INFINITE_PRODUCTS = [
    # The manual's own example, and Wallis'.
    ("PRODUCT(1 - 1/k^2, k, 2, inf)", "1/2"),
    ("PRODUCT(4*k^2/(4*k^2 - 1), k, 1, inf)", "pi/2"),
    # These two are `SINH(pi)/pi` and `COSH(SQRT(3)*pi/2)/(3*pi)`, written the
    # way Simplify writes any hyperbolic of a constant.
    ("PRODUCT(1 + 1/k^2, k, 1, inf)", "(#e^pi/2 - #e^(-pi)/2)/pi"),
    (
        "PRODUCT(1 - 1/k^3, k, 2, inf)",
        "(#e^(SQRT(3)*pi) + 1)*#e^(-SQRT(3)*pi/2)/(6*pi)",
    ),
    # A product need not converge to a number to have a value.
    ("PRODUCT(k/(k + 1), k, 1, inf)", "0"),
    ("PRODUCT((k + 1)/k, k, 1, inf)", "inf"),
    # A lower end that is a variable telescopes one step short of the whole.
    ("PRODUCT(1 - 1/k^2, k, m, inf)", "(m - 1)/m"),
    # And where either half of the definition declines, the product stands: the
    # first has no closed form to take a limit of, and the second's limit
    # depends on the sign of an `x` nothing has been told anything about.
    ("PRODUCT(COS(pi/2^k), k, 1, inf)", "PRODUCT(COS(pi*2^(-k)), k, 1, inf)"),
    ("PRODUCT(1 - x/k, k, 1, inf)", "PRODUCT(1 - x/k, k, 1, inf)"),
    # An oscillating product has no value, and `nan` is not one: it stands, the
    # way the sum that oscillates the same way does.
    ("PRODUCT(-1, k, 1, inf)", "PRODUCT(-1, k, 1, inf)"),
    ("SUM((-1)^k, k, 1, inf)", "SUM((-1)^k, k, 1, inf)"),
]


@pytest.mark.parametrize(("text", "expected"), INFINITE_PRODUCTS, ids=str)
def test_an_infinite_product_is_the_limit_of_its_partial_products(text, expected):
    """Sympy has no routine for one, so the definition is used instead.

    The product up to `n` is a closed form in `n` wherever the body is
    hypergeometric, and the limit of that closed form is what the infinite
    product means. Both halves are sympy's, and putting them together is what
    reaches the answer Derive prints.
    """
    assert simp(text) == expected


def test_an_antiderivative_over_a_denominator_that_does_not_factor_keeps_its_sum():
    """The logarithmic part of these is a sum over the roots of a polynomial.

    `ROOT_SUM(p, t, u)` is what the notation calls that, and the answer is the
    rational part plus one. What it must never be is the rational part alone:
    `1/(x^7 + 1)` does not integrate to `LN(x + 1)/7`, and an answer that
    quietly loses two thirds of itself is worse than one that cannot be read.

    The polynomial summed over is the resolvent sympy builds and not the
    denominator it came from, which is why the coefficients are powers of seven:
    the roots of one are the roots of the other scaled, and the summand is
    scaled to match.
    """
    for text, answer in (
        (
            "INT(1/(x^7 + 1), x)",
            "LN(x + 1)/7 + ROOT_SUM(117649*t^6 + 16807*t^5 + 2401*t^4 + 343*t^3 "
            "+ 49*t^2 + 7*t + 1, t, t*LN(x + 7*t))",
        ),
        (
            "INT(1/(x^7 - 1), x)",
            "LN(x - 1)/7 + ROOT_SUM(117649*t^6 + 16807*t^5 + 2401*t^4 + 343*t^3 "
            "+ 49*t^2 + 7*t + 1, t, t*LN(x - 7*t))",
        ),
    ):
        assert simp(text) == answer


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


def test_a_single_root_of_a_polynomial_approximates_to_that_root():
    """`ROOT_OF` is the same idea for one indexed root, and it is as live.

    Nothing found produces one today, so it is the companion of `ROOT_SUM`
    rather than an answer anything gives; what it must do is be a number when a
    number is asked for. The quintic's one real root is 1.1673039782...
    """
    assert approx(parse("ROOT_OF(t^5 - t - 1, t, 0)"), Context(), 10).text == (
        "1.167303978"
    )


def test_an_integral_answered_by_a_special_function_reads_back_as_that_function():
    """An answer is only as good as the name it is written under.

    Each of these was written under a name something else answers to. `LI` is
    sympy's offset logarithmic integral, `li(x) - li(2)`, as readily as its
    plain one; `CHI` is Derive's chi-square distribution long before it is
    anybody's cosh-integral. So each answer read back was a different function
    from the one computed, with nothing to say so - which is what simplifying
    the answer again is here to catch.
    """
    for text, answer in (
        ("INT(1/LN(x), x)", "LI(x)"),
        ("INT(COSH(x)/x, x)", "COSH_INT(x)"),
        ("INT(SINH(x)/x, x)", "SINH_INT(x)"),
    ):
        assert simp(text) == answer
        assert simp(answer) == answer
    # The offset integral is exactly zero at 2, which is what kept the wrong
    # reading out of sight. The logarithmic integral there is 1.0451637801,
    # and it waits for digits nothing can give it while sympy leaves the
    # finiteness of `li(2)` open.
    assert simp("APPROX(LI(2))") != "0"


def test_an_integral_is_taken_straight_through_an_interior_singularity():
    """A definite integral is the difference of the antiderivative's endpoints.

    Nothing looks in between, which the manual is explicit about (7.4, p.199):
    finding the singularities inside an interval and splitting the integral
    there is the reader's job. So a pole inside is integrated straight through.
    Over an odd power that lands on the Cauchy principal value, the positive
    and negative infinite areas cancelling exactly; over an even power it lands
    on a number the manual itself calls obviously wrong, the integrand being
    positive throughout. Split at the pole and the divergence is there to see.
    """
    assert simp("INT(1/x^3, x, -1, 2)") == "3/8"
    assert simp("INT(1/x^2, x, -1, 1)") == "-2"
    assert simp("INT(1/x^2, x, -1, 0) + INT(1/x^2, x, 0, 1)") == "inf"


def test_a_two_sided_limit_whose_sides_disagree_is_plus_or_minus():
    assert simp("LIM(SIGN(x), x, 0)") == "±1"


def test_a_bounded_limit_that_never_settles_is_the_range_it_keeps_to():
    """`INTERVAL(a, b)`, where there is no one value to answer with.

    `SIN(1/x)` oscillates the whole way to zero and `SIN(x)` never stops
    oscillating at all, so none of these limits exists - but each stays inside
    a range it keeps returning to, and the range is what is known. Answering
    `?` would throw that away.

    The last of them arrives inside a product rather than as the whole answer,
    which is the case a head that could only be written on its own would leave
    unreadable. None of them may come back as a `Kind.STRING`: that is what an
    answer the notation has no name for looks like, and it is dead on arrival.
    """
    for text, answer in (
        ("LIM(SIN(1/x), x, 0)", "INTERVAL(-1, 1)"),
        ("LIM(COS(1/x), x, 0)", "INTERVAL(-1, 1)"),
        ("LIM(SIN(x), x, inf)", "INTERVAL(-1, 1)"),
        ("LIM(x*SIN(x), x, inf)", "inf*SIGN(INTERVAL(-1, 1))"),
    ):
        answered = simplify(parse(text), Context())
        assert answered.text == answer
        assert answered.node.kind is not Kind.STRING


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


def test_a_limit_that_settles_or_runs_away_is_no_interval():
    """The range is the answer to the bounded case alone.

    One that exists is its value; one that grows without bound is `±inf`, which
    is not a range of values but a single one the notation writes both signs of;
    and one whose two sides are different numbers is `?`, the floor's being 0
    from the left and 1 from the right.
    """
    assert simp("LIM(SIN(x)/x, x, 0)") == "1"
    assert simp("LIM(1/x, x, 0)") == "±inf"
    assert simp("LIM(FLOOR(x), x, 1)") == "?"


# -- special values -----------------------------------------------------------

SPECIAL = [
    ("0/0", "?"),
    ("inf - inf", "?"),
    ("0*inf", "?"),
    ("inf/inf", "?"),
    ("? + 1", "?"),
    ("1/0", "±inf"),
    ("TAN(pi/2)", "±inf"),
    ("SIGN(0)", "±1"),
    ("inf + inf", "inf"),
    ("-inf", "-inf"),
]


@pytest.mark.parametrize(("text", "expected"), SPECIAL, ids=str)
def test_a_special_value(text, expected):
    assert simp(text) == expected


#: Section 6.8's sign of a complex number: the point of that number on the unit
#: circle, `z/|z|`. The original answers the first of these `3/5 + 4*#i/5`. A
#: name is not a number and waits, whatever is written beside it.
COMPLEX_SIGNS = [
    ("SIGN(3 + 4*#i)", "3/5 + 4*#i/5"),
    ("SIGN(2 + #i)", "SQRT(5)*(2 + #i)/5"),
    ("SIGN(#i)", "#i"),
    ("SIGN(-3)", "-1"),
    ("SIGN(x + #i)", "SIGN(x + #i)"),
]


@pytest.mark.parametrize(("text", "expected"), COMPLEX_SIGNS, ids=str)
def test_the_sign_of_a_complex_number(text, expected):
    assert simp(text) == expected


# -- the indicator function and the normal distribution ------------------------

#: Two of Section 6's functions that are a closed form rather than a rule:
#: whatever else is known about the arguments, the answer is the formula.
INDICATOR_AND_NORMAL = [
    # One difference of signs: 1 between the bounds and 0 outside them.
    ("CHI(0, 1/2, 1)", "1"),
    ("CHI(0, 2, 1)", "0"),
    ("CHI(0, -1, 1)", "0"),
    # Bounds the wrong way round give -1, which is the manual's
    # `CHI(a, x, b) = -CHI(b, x, a)`.
    ("CHI(1, 1/2, 0)", "-1"),
    # `a` defaults to 0 and `b` to 1, so both of these indicate the unit
    # interval, and the formula is the answer where nothing can be compared.
    ("CHI(x)", "SIGN(x)/2 - SIGN(x - 1)/2"),
    ("CHI(0, x)", "SIGN(x)/2 - SIGN(x - 1)/2"),
    ("CHI(2, x, 1)", "SIGN(x - 2)/2 - SIGN(x - 1)/2"),
    # On the edge the indicator is undecided, `SIGN(0)` being `±1`. Written
    # `(±1)/2` where Derive writes `±1/2`: the same number, this printer's
    # spelling of it.
    ("CHI(0, 0, 1)", "(±1)/2 + 1/2"),
    # The cumulative normal distribution, over the error function. Derive
    # writes it `(ERF(√2*z/2) + 1)/2`; sympy distributes the half.
    ("NORMAL(z)", "ERF(SQRT(2)*z/2)/2 + 1/2"),
    ("NORMAL(0)", "1/2"),
    # The normal form reaches inside the error function's argument and writes it
    # about `z`, which divides the `2*s` into each term: Derive answers
    # `(ERF(√2*z/(2*s) - √2*m/(2*s)) + 1)/2`, not the folded `(z - m)/(2*s)`.
    # `ERF` is odd, so sympy carries the negation outside the call rather than
    # inside the argument; the two are the same number.
    (
        "NORMAL(z, m, s)",
        "1/2 - ERF(SQRT(2)*m/(2*s) - SQRT(2)*z/(2*s))/2",
    ),
]


@pytest.mark.parametrize(("text", "expected"), INDICATOR_AND_NORMAL, ids=str)
def test_the_indicator_and_the_normal_distribution(text, expected):
    assert simp(text) == expected


# -- the financial functions --------------------------------------------------

#: Section 6.12's annuity, solved for one of its own each time. The original
#: answers every one of these.
ANNUITIES = [
    # The four closed forms, in the arrangement the original writes them in.
    ("PMT(i, n, v)", "i*v*(i + 1)^n/(1 - (i + 1)^n)"),
    ("PMT(i, n, v, f, t)", "i*(f + v*(i + 1)^n)/((1 - (i + 1)^n)*(i*t + 1))"),
    ("NPER(i, p, v, f, t)", "LN((p*(i*t + 1) - f*i)/(i*(p*t + v) + p))/LN(i + 1)"),
    # Undefined where there is no interest to compound, the payment term
    # dividing by the rate. The original answers `?` and takes no limit to
    # rescue it.
    ("PMT(0, 10, 1000)", "?"),
    # Too few arguments to say what the contract is, or more than the function
    # takes, and the call is worth no more than itself.
    ("PMT(3)", "PMT(3)"),
    ("PMT(1, 2, 3, 4, 5, 6)", "PMT(1, 2, 3, 4, 5, 6)"),
    # The rate is the one of the five that is no closed form. It is searched
    # for between bounds instead, and the bounds are part of the answer: the
    # original writes in the [0, 1] it looks between when it is given none.
    ("RATE(n, p, v)", "RATE(n, p, v, 0, 0, 0, 1)"),
    ("RATE(n, p, v, f, t)", "RATE(n, p, v, f, t, 0, 1)"),
    ("RATE(36, -300, 9000)", "402/39383"),
    ("RATE(36, -300, 9000, 0, 0, 0, 1)", "402/39383"),
    # An interval holding no rate answers nothing, and a search that finds
    # nothing leaves the call it was asked as. Bounds are wanted in pairs: six
    # arguments name a lower bound and no upper one, which is no contract.
    ("RATE(36, -300, 9000, 0, 0, 0.5, 1)", "RATE(36, -300, 9000, 0, 0, 1/2, 1)"),
    ("RATE(3)", "RATE(3)"),
    ("RATE(1, 2, 3, 4, 5, 6)", "RATE(1, 2, 3, 4, 5, 6)"),
]


@pytest.mark.parametrize(("text", "expected"), ANNUITIES, ids=str)
def test_the_financial_functions(text, expected):
    assert simp(text) == expected


#: 6.12's annuity with one of its own left to be filled in, which is what each
#: financial function computes. Whichever one goes back in, the equation holds.
ANNUITY = "{0}*(1 + i)^n + {1}*(1 + i*t)*((1 + i)^n - 1)/i + {2}"


@pytest.mark.parametrize(
    "equation",
    [
        ANNUITY.format("PVAL(i, n, p, f, t)", "p", "f"),
        ANNUITY.format("v", "p", "FVAL(i, n, p, v, t)"),
        ANNUITY.format("v", "PMT(i, n, v, f, t)", "f"),
    ],
    ids=("PVAL", "FVAL", "PMT"),
)
def test_a_contract_is_worth_what_the_annuity_equation_says(equation):
    assert simp(equation) == "0"


def test_a_contract_approximates_to_the_digits_the_original_gives():
    """Four contracts the original answers, which the manual gives no example of."""
    ten = Context().with_precision(Precision.APPROXIMATE, 10)
    assert simp("PVAL(5%/12, 24, -200, 1000, 1)", ten) == "3672.749170"
    assert simp("FVAL(5%/12, 24, -200, 1000, 1)", ten) == "3953.231038"
    assert simp("NPER(5%/12, -200, 4000, 1000, 1)", ten) == "25.77406754"
    assert simp("RATE(10, 100, -500)", ten) == "0.1509841447"


def test_a_random_number_is_a_number():
    """Section 6.13's four cases, asked for what kind of answer each gives.

    What is drawn cannot be pinned and is not tried: the seed is the one thing
    here with an answer of its own, and what the rest have to be is numbers of
    the right kind in the right place.
    """
    assert simp("RANDOM(6)") in {"0", "1", "2", "3", "4", "5"}
    assert 0 <= Fraction(simp("RANDOM(1)")) < 1
    assert simp("RANDOM(-5)") == "5"
    assert Fraction(simp("RANDOM(0)")) >= 0
    # A whole number of outcomes or nothing, and nothing is the call itself.
    assert simp("RANDOM(6.5)") == "RANDOM(13/2)"
    assert simp("RANDOM(x)") == "RANDOM(x)"


def test_a_seeded_generator_draws_what_it_drew_before():
    seeded = [simp("RANDOM(-20)")] + [simp("RANDOM(1000)") for _ in range(5)]
    again = [simp("RANDOM(-20)")] + [simp("RANDOM(1000)") for _ in range(5)]
    assert seeded == again


def test_a_searched_rate_is_as_close_as_the_search_it_came_from():
    """A rate is bisected for, so the last digits belong to the search.

    They are not the original's: it answers `0.01020744896` where this answers
    `0.01020744900`, and moving the bounds moves its answer too - between
    `0.005` and `0.02` it says `0.01020744901`. What both agree on is the rate,
    to every digit the equation itself decides.
    """
    ten = Context().with_precision(Precision.APPROXIMATE, 10)
    assert simp("RATE(36, -300, 9000)", ten).startswith("0.010207449")


def test_plus_or_minus_simplifies_its_argument_and_seals_it_in():
    assert simp("±(2 + 3)") == "±5"
    # Arithmetic does not propagate the ambiguity outward; that is out of scope
    # and the head stays inert instead of guessing a sign.
    assert simp("±1 + 1") == "±1 + 1"


def test_a_singular_matrix_inverse_comes_back_unsimplified():
    assert simp("[[1, 2], [2, 4]]^-1") == "[[1, 2], [2, 4]]^(-1)"
    assert simp("[[1, 2], [3, 4]]^-1") == "[[-2, 1], [3/2, -1/2]]"


# -- vectors and matrices -----------------------------------------------------

VECTORS = [
    ("[2*x, -5, x^2] + [-x, 8, 2*x]", "[x, 3, x^2 + 2*x]"),
    ("[2, a, 5] . [2*a, 3, -1]", "7*a - 5"),
    ("[2, 3]*[4, 5]", "23"),
    ("DET([[1, 2], [3, 4]])", "-2"),
    ("TRACE([[1, 2], [3, 4]])", "5"),
    ("DIMENSION([1, 2, 3])", "3"),
    ("[[1, 2], [3, 4]]`", "[[1, 3], [2, 4]]"),
    ("SUM([1, 2, 3])", "6"),
    ("[[1, 2], [3, 4]] . [[a], [b]]", "[[a + 2*b], [3*a + 4*b]]"),
    # A flat vector to the right of a matrix is the column it stands for, and
    # the answer is flat again; written as a column it stays one.
    ("[[a, b], [c, d]] . [2, 3]", "[2*a + 3*b, 2*c + 3*d]"),
    ("[2, 3] . [[a, b], [c, d]]", "[2*a + 3*c, 2*b + 3*d]"),
    # The cross product of two vectors of three elements, and of two of two:
    # the second is the third component of the first, the one the other two
    # come to zero in, and it is a number rather than a vector.
    ("CROSS([1, 2, 3], [a, b, c])", "[2*c - 3*b, 3*a - c, b - 2*a]"),
    ("CROSS([1, 2], [a, b])", "b - 2*a"),
    # An index that is a vector reaches through as many dimensions as it holds.
    ("[[2, 3, 5], [7, 1, 4]] SUB [2, 3]", "4"),
    ("ELEMENT([[2, 3, 5], [7, 1, 4]], [2, 3])", "4"),
    ("[[2, 3, 5], [7, 1, 4]] SUB [2]", "[7, 1, 4]"),
    # Shapes that will not multiply keep the operator, unevaluated: better the
    # expression back than a guess at which product was meant.
    ("[[1, 2], [3, 4]] . [1, 2, 3]", "[[1, 2], [3, 4]] . [1, 2, 3]"),
    ("a . b", "a . b"),
    # Ragged: no matrix to be, and nothing to do to it either.
    ("[[1, 2], [3]]", "[[1, 2], [3]]"),
]


@pytest.mark.parametrize(("text", "expected"), VECTORS, ids=str)
def test_vectors_and_matrices(text, expected):
    assert simp(text) == expected


# -- declared nonscalars ------------------------------------------------------

NONSCALAR = declared(*(f"{name} :epsilon Nonscalar" for name in "abcd"))

#: What a declared nonscalar is worth past the nine rules section 8.8 states.
#: Every one of these comes from the original and none of them is in the manual.
NONSCALARS = [
    # 8.4: multiplication between two matrices is the matrix product, however
    # it was written.
    ("a*b", "a . b"),
    # A matrix beside its own inverse is left exactly as it was written. There
    # is no notation for an identity matrix of unknown dimension, so an answer
    # holding one could not be shown at all.
    ("a . a^-1", "a . a^-1"),
    ("a^-1 . a", "a^-1 . a"),
    # A longer product nests to the right, and the parentheses are printed.
    # `a . (b . c)` is the fixed point; `(a . b) . c` is not.
    ("((a . b) . c) . d", "a . (b . (c . d))"),
    ("a . b . c . d", "a . (b . (c . d))"),
    ("a . (b . c)", "a . (b . c)"),
    # Which is the associativity rule of 8.8, seen from the other side.
    ("a . (b . c) - (a . b) . c", "0"),
    # The determinant of a product, which the manual states only for an
    # inverse.
    ("DET(a . b)", "DET(a)*DET(b)"),
    ("TRACE(a + b)", "TRACE(a) + TRACE(b)"),
    # Distribution reaches inside a longer product too.
    ("a . (b + c) . a", "a . (b . a) + a . (c . a)"),
    # A scalar commutes with everything, so a negated product needs no
    # parentheses and a difference of two of them reads like any other sum.
    ("a . b - b . a", "a . b - b . a"),
    ("-(a . b)", "-a . b"),
    # Dividing by a matrix is multiplying by its inverse, that being the only
    # reading the notation has for it.
    ("1/a", "a^-1"),
    ("x/a", "x*a^-1"),
]


@pytest.mark.parametrize(("text", "expected"), NONSCALARS, ids=str)
def test_the_algebra_of_a_declared_nonscalar(text, expected):
    assert simp(text, NONSCALAR) == expected


def test_a_scalars_transpose_collapses_at_the_first_backquote():
    # 8.5: the transpose of a scalar is the scalar, and everything no
    # declaration calls nonscalar is one.
    assert simp("x`") == "x"
    assert simp("x``") == "x"
    assert simp("(x + y)`") == "x + y"
    # A nonscalar's does not, and takes two to come back.
    assert simp("a`", NONSCALAR) == "a`"
    assert simp("a``", NONSCALAR) == "a"


def test_a_default_domain_of_nonscalar_does_not_make_every_variable_a_matrix():
    # `default :epsilon Nonscalar` widens the domain of everything at once,
    # and everything includes the argument of every function: a matrix there
    # would convert and mean nothing. So the default is worth its assumptions
    # and not the shape a declared name gets.
    default = declared("default :epsilon Nonscalar")
    assert simp("SIN(x)", default) == "SIN(x)"
    assert simp("x . y", default) == "x . y"


#: Row echelon form, characteristic polynomial and eigenvalues: the manual's own
#: examples, and the argument forms each of the three accepts.
LINEAR_ALGEBRA = [
    # Both of the manual's augmented examples. The second is singular: its right
    # column is inconsistent and the one before it is consistent, which is what
    # the reduced form says by putting the pivot in the last column.
    ("ROW_REDUCE([[1,2],[5,6]],[[3,4],[7,8]])", "[[1, 0, -1, -2], [0, 1, 2, 3]]"),
    ("ROW_REDUCE([[1,1],[2,2]],[[1,1],[2,1]])", "[[1, 1, 1, 0], [0, 0, 0, 1]]"),
    # The manual's exercise, exactly rather than approximately: the augmented
    # column holds the solution of the 3 x 3 system.
    (
        "ROW_REDUCE([[5,3,-7],[2,-8,1],[-1,9,4]],[[4],[6],[5]])",
        "[[1, 0, 0, 879/302], [0, 1, 0, 53/302], [0, 0, 1, 239/151]]",
    ),
    # A vector is adjoined as the column it stands for, so it says the same
    # thing as a one-column matrix does.
    (
        "ROW_REDUCE([[5,3,-7],[2,-8,1],[-1,9,4]],[4,6,5])",
        "[[1, 0, 0, 879/302], [0, 1, 0, 53/302], [0, 0, 1, 239/151]]",
    ),
    # One argument reduces the matrix alone: a nonsingular one reduces to the
    # identity, and a rank-deficient one keeps a zero row.
    ("ROW_REDUCE([[1,2],[3,4]])", "[[1, 0], [0, 1]]"),
    ("ROW_REDUCE([[1,2,3],[4,5,6],[7,8,9]])", "[[1, 0, -1], [0, 1, 2], [0, 0, 0]]"),
    # Symbolic entries reduce the generic case, the one where no pivot is zero -
    # the answer the inverse gives for the same system.
    (
        "ROW_REDUCE([[a,b],[c,d]],[e,f])",
        "[[1, 0, (d*e - b*f)/(a*d - b*c)], [0, 1, (a*f - c*e)/(a*d - b*c)]]",
    ),
    # The manual's characteristic polynomial, term for term.
    ("CHARPOLY([[2,3],[a,b]],z)", "z^2 - z*(b + 2) - 3*a + 2*b"),
    # The variable defaults to `w` when none is written, and `w` is not on the
    # order list, so the most main variable of the answer is `a` and the normal
    # form takes `-3*a` off and writes what is left about `b`. Derive answers
    # `(2 - w)*(b - w) - 3*a`: its determinant of `[[2 - w, 3], [a, b - w]]`
    # stays folded, and a product is not a sum for the normal form to touch.
    # Sympy computes the characteristic polynomial by recurrence and hands back
    # a multiplied-out one, which no longer folds. Given that multiplied-out sum
    # Derive answers exactly as this does.
    ("CHARPOLY([[2,3],[a,b]])", "-3*a + b*(2 - w) + w^2 - 2*w"),
    # `DET(A - z*IDENTITY_MATRIX(3))` is the definition, so the cube is negated
    # in an odd dimension; a sum of more than two terms keeps the leading minus.
    ("CHARPOLY([[1,2,3],[4,5,6],[7,8,10]],z)", "-z^3 + 16*z^2 + 12*z - 3"),
    # Eigenvalues are answered as solved equations. The manual's example, and a
    # 2 x 2 whose eigenvalues the quadratic formula has to be used on. Where a
    # pair differs only in the sign of a radical Derive lists the sum first;
    # this is sympy's canonical order, which agrees with Derive on every other
    # case and reverses that one.
    ("EIGENVALUES([[2,3],[0,b]],z)", "[z = 2, z = b]"),
    ("EIGENVALUES([[5,0],[0,2]],z)", "[z = 2, z = 5]"),
    ("EIGENVALUES([[1,2],[3,4]],x)", "[x = 5/2 - SQRT(33)/2, x = SQRT(33)/2 + 5/2]"),
    # A repeated eigenvalue is one eigenvalue: the multiplicity is the number of
    # parameters its eigenvector carries, not a second solution.
    ("EIGENVALUES([[1,0,0],[0,1,0],[0,0,2]],z)", "[z = 1, z = 2]"),
    # A symmetric matrix has real eigenvalues, and this one's characteristic
    # polynomial is the casus irreducibilis: three real roots that Cardano's
    # formula can only write through `#i`. Viete's cubic writes them as the
    # cosines they are, so nothing here is spelled with an imaginary unit.
    (
        "EIGENVALUES([[0,1,1],[1,1,1],[1,1,-1]],z)",
        "[z = -4*SQRT(3)*SIN(pi/6 - ACOS(3*SQRT(3)/8)/3)/3, "
        "z = -4*SQRT(3)*COS(pi/3 - ACOS(3*SQRT(3)/8)/3)/3, "
        "z = 4*SQRT(3)*COS(ACOS(3*SQRT(3)/8)/3)/3]",
    ),
    # No square matrix, no characteristic polynomial - and no radicals for the
    # quintic the manual's 5 x 5 exercise leads to, which is the manual's own
    # account of why exact eigenvalues stop at 4 x 4. Both come back unchanged.
    ("CHARPOLY([[1,2],[3,4],[5,6]],z)", "CHARPOLY([[1, 2], [3, 4], [5, 6]], z)"),
    (
        "EIGENVALUES([[-2,1,1,1,1],[1,-3,-1,0,-1],[1,-1,1,0,-3],[1,0,0,3,0],"
        "[1,-1,-3,0,2]],z)",
        "EIGENVALUES([[-2, 1, 1, 1, 1], [1, -3, -1, 0, -1], [1, -1, 1, 0, -3], "
        "[1, 0, 0, 3, 0], [1, -1, -3, 0, 2]], z)",
    ),
]


@pytest.mark.parametrize(("text", "expected"), LINEAR_ALGEBRA, ids=str)
def test_linear_algebra(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize(
    "matrix",
    [
        "[[2,3],[a,b]]",
        "[[1,2,3],[4,5,6],[7,8,10]]",
        # The symmetric matrix the manual's last eigenvalue exercise poses.
        "[[-2,1,1,1,1],[1,-3,-1,0,-1],[1,-1,1,0,-3],[1,0,0,3,0],[1,-1,-3,0,2]]",
    ],
    ids=str,
)
def test_the_characteristic_polynomial_is_the_determinant_defining_it(matrix):
    # The manual defines CHARPOLY as the determinant of the difference of the
    # matrix and a variable times the identity matrix, so the two must agree -
    # including in odd dimensions, where that determinant is not monic. The
    # difference is what is asserted rather than the two texts: the polynomial
    # comes out with its terms collected in the variable and the determinant does
    # not, which is a spelling and not a disagreement.
    written_out = f"DET({matrix} - z*IDENTITY_MATRIX(DIMENSION({matrix})))"
    assert simp(f"CHARPOLY({matrix},z) - {written_out}") == "0"


#: Differential and integral vector calculus: the manual's own examples for each
#: of the six operators, and the argument forms they accept.
VECTOR_CALCULUS = [
    # The default coordinate system is three-dimensional Cartesian in x, y, z,
    # so a gradient written without one has three elements however few variables
    # the expression mentions.
    ("GRAD(x*y^2*z^3)", "[y^2*z^3, 2*x*y*z^3, 3*x*y^2*z^2]"),
    ("GRAD(x^2 + y^2)", "[2*x, 2*y, 0]"),
    # An arbitrary function has an arbitrary gradient, and it is written the way
    # any other underived derivative is.
    (
        "GRAD(F(x, y, z))",
        "[DIF(F(x, y, z), x), DIF(F(x, y, z), y), DIF(F(x, y, z), z)]",
    ),
    # A vector of variables names Cartesian coordinates of the caller's
    # choosing, and there may be any number of them.
    ("GRAD(c*w + x^2 + y^3 + z^4, [w, x, y, z])", "[c, 2*x, 3*y^2, 4*z^3]"),
    ("GRAD(x*y, [x, y])", "[y, x]"),
    ("DIV([y^2*z^3, 2*x*y*z^3, 3*x*y^2*z^2])", "x*(6*y^2*z + 2*z^3)"),
    # Degree one in the primary variable, so the normal form writes it as `x`
    # times its coefficient, and the coefficient keeps its numeric content -
    # which is the manual's own answer. Derive is not consistent about that:
    # the same coefficient `2*y + 6` comes back `x*(2*y + 6)` authored as
    # `2*x*y + 6*x` and `2*x^2*(y + 3)` when it falls out of expanding
    # `((x + 1)^2 + y)^2`, both taken from the original, so there is no one form
    # that matches everything. LAPLACIAN is DIV of GRAD, so the two agree.
    ("LAPLACIAN(x*y^2*z^3)", "x*(6*y^2*z + 2*z^3)"),
    ("CURL([y^2, 2*x*z, 0])", "[-2*x, 0, 2*z - 2*y]"),
    # The curl of a plane field is one number, not the space vector `[0, 0, w]`
    # that number is the last element of.
    ("CURL([v^2, u], [[u, v], [1, 1]])", "1 - 2*v"),
    ("CURL([-y, x], [u, v])", "0"),
    # A two-row second argument is a coordinate geometry matrix: variables
    # above, scale factors below. The manual's spherical example, written out
    # rather than loaded from VECTOR.MTH, which is where the name lives. The
    # manual prints the middle element as `COT(phi)*COS(theta)`; a cotangent
    # comes out of the engine's canonical form as one over a tangent.
    (
        "GRAD(r*SIN(theta)*COS(phi), [[r, theta, phi], [1, r*SIN(phi), r]])",
        "[SIN(theta)*COS(phi), COS(theta)/TAN(phi), -SIN(theta)*SIN(phi)]",
    ),
    # Cylindrical coordinates, where the scale factors are what make the answers
    # differ from the Cartesian ones: `DIV` of a purely radial field grows with
    # the circumference the flux crosses.
    ("DIV([r^2, 0, 0], [[r, theta, z], [1, r, 1]])", "3*r"),
    ("LAPLACIAN(r^2, [[r, theta, z], [1, r, 1]])", "4"),
    ("CURL([0, r, 0], [[r, theta, z], [1, r, 1]])", "[0, 0, 2]"),
    ("POTENTIAL([y^2*z^3, 2*x*y*z^3, 3*x*y^2*z^2])", "x*y^2*z^3"),
    ("VECTOR_POTENTIAL([x, 0, y - z])", "[-y^2/2, -x*z, 0]"),
    # The starting coordinates of the line integrals, which default to the
    # origin. Starting at [1, 1, 1] rather than the origin subtracts the value
    # the potential would have had there, an additive constant and no more.
    ("POTENTIAL([2*x, 2*y, 2*z])", "x^2 + y^2 + z^2"),
    ("POTENTIAL([2*x, 2*y, 2*z], [1, 1, 1])", "x^2 + y^2 + z^2 - 3"),
    # The manual's warning about a bad starting point, exactly: this field is
    # infinite at the origin, so the potential from the origin is infinite too,
    # and 1 is the alternative the manual recommends for a logarithm.
    ("POTENTIAL([1/x, 1/y, 1/z], [1, 1, 1])", "LN(x) + LN(y) + LN(z)"),
    ("POTENTIAL([1/x, 1/y, 1/z])", "LN(x) + LN(y) + LN(z) + inf"),
    # A field of fewer elements than the system has coordinates is no field on
    # it, and the default system has three. Two elements are a planar curl only
    # when a planar system is given to work in.
    ("CURL([-y, x])", "CURL([-y, x])"),
    ("DIV([x, y])", "DIV([x, y])"),
    ("CURL([-y, x], [[x, y], [1, 1]])", "2"),
    # Not every field has a potential, and Derive does not check. POTENTIAL
    # computes one line integral and hands back its value; `[-y, x, 0]`
    # circulates, so the gradient of that value is not the field it came from.
    # The manual leaves that comparison to the caller, and CURL is the easier
    # test.
    ("CURL([-y, x, 0])", "[0, 0, 2]"),
    ("POTENTIAL([-y, x, 0])", "x*y"),
    ("GRAD(POTENTIAL([-y, x, 0]))", "[y, x, 0]"),
    # The same for a vector potential, whose obstruction is a nonzero
    # divergence: `[x, 0, 0]` spreads out, and the curl of the answer is not it.
    ("DIV([x, 0, 0])", "1"),
    ("VECTOR_POTENTIAL([x, 0, 0])", "[0, -x*z, 0]"),
    ("CURL(VECTOR_POTENTIAL([x, 0, 0]))", "[x, 0, -z]"),
    # Arguments the operators can make nothing of: a vector has no gradient, a
    # single variable is no coordinate system, curl and vector potential are
    # defined in the plane and in space and nowhere else. Each comes back the
    # call it was written as.
    ("GRAD([1, 2])", "GRAD([1, 2])"),
    ("GRAD(u, x)", "GRAD(u, x)"),
    ("CURL([a, b, c, d])", "CURL([a, b, c, d])"),
    ("VECTOR_POTENTIAL([a, b])", "VECTOR_POTENTIAL([a, b])"),
    ("DIV([1, 2, 3, 4])", "DIV([1, 2, 3, 4])"),
]


@pytest.mark.parametrize(("text", "expected"), VECTOR_CALCULUS, ids=str)
def test_vector_calculus(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize(
    "field",
    ["[y^2*z^3, 2*x*y*z^3, 3*x*y^2*z^2]", "[1, 2*y, 3*z^2]", "[y*z, x*z, x*y]"],
    ids=str,
)
def test_a_conservative_field_is_the_gradient_of_its_potential(field):
    # What the manual says to check, on fields that pass it: POTENTIAL is only
    # a line integral, and it answers the question asked of it exactly when the
    # curl of the field is zero.
    assert simp(f"CURL({field})") == "[0, 0, 0]"
    assert simp(f"GRAD(POTENTIAL({field})) - {field}") == "[0, 0, 0]"


@pytest.mark.parametrize(
    "field",
    ["[-2*x, 0, 2*z - 2*y]", "[x, 0, y - z]", "[y, z, x]"],
    ids=str,
)
def test_a_solenoidal_field_is_the_curl_of_its_vector_potential(field):
    # The same check for VECTOR_POTENTIAL, whose condition is a vanishing
    # divergence. Two equally valid vector potentials differ by a gradient, so
    # it is the curl that has to be compared and not the vector itself.
    assert simp(f"DIV({field})") == "0"
    assert simp(f"CURL(VECTOR_POTENTIAL({field})) - {field}") == "[0, 0, 0]"


#: The four ways of saying which values a generated vector's variable takes.
#: The first three are the manual's own examples.
GENERATED = [
    ("VECTOR(x^2, x, 5)", "[1, 4, 9, 16, 25]"),
    ("VECTOR(j!, j, 0, 4)", "[1, 1, 2, 6, 24]"),
    ("VECTOR(k^2, k, [2, 3, 5, 7, 11])", "[4, 9, 25, 49, 121]"),
    # A step, and the count rounded down: four elements, the last one short of
    # pi/4. Exact, the manual's decimals being what it approximates to.
    ("VECTOR(SIN(z), z, 0, pi/4, 1/5)", "[0, SIN(1/5), SIN(2/5), SIN(3/5)]"),
    # A vector of vectors is a matrix, so a nested call generates one.
    (
        "VECTOR(VECTOR(j + k, k, 1, 4), j, 1, 3)",
        "[[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]",
    ),
    # The variable ranges over a matrix's rows, as every other element access
    # does.
    ("VECTOR(k, k, [[1, 2], [3, 4]])", "[[1, 2], [3, 4]]"),
    # A range the step cannot cross once has one element; one going the wrong
    # way has none.
    ("VECTOR(x^2, x, 3, 3)", "[9]"),
    ("VECTOR(x, x, 5, 1)", "[]"),
    # Bounds that are no sequence yet, and a second argument that is no
    # variable: the call itself is the answer.
    ("VECTOR(x^2, x, n)", "VECTOR(x^2, x, n)"),
    ("VECTOR(x^2, 5, 1, 3)", "VECTOR(x^2, 5, 1, 3)"),
    ("VECTOR(x^2, x, 1, 3, 0)", "VECTOR(x^2, x, 1, 3, 0)"),
]


@pytest.mark.parametrize(("text", "expected"), GENERATED, ids=str)
def test_generated_vectors(text, expected):
    assert simp(text) == expected


#: `SELECT` says which values in the same four ways, and answers with the
#: values themselves rather than with anything computed from them.
SELECTED = [
    # The manual's own two examples.
    ("SELECT(PRIME(k), k, [3, 5, 7, 9, 11])", "[3, 5, 7, 11]"),
    ("SELECT(PRIME(k), k, 1, 20)", "[2, 3, 5, 7, 11, 13, 17, 19]"),
    # A count, and a range with a step.
    ("SELECT(k > 1, k, 5)", "[2, 3, 4, 5]"),
    ("SELECT(k > 1, k, 1, 10, 3)", "[4, 7, 10]"),
    ("SELECT(k^2 > 5, k, 1, 5)", "[3, 4, 5]"),
    ("SELECT(k > 5, k, 1, 3)", "[]"),
    # A second argument that is no variable binds nothing, and a test nobody
    # can decide is no answer: both come back the call they were written as.
    # Derive answers the second with the empty vector, having dropped the
    # element it could not judge; this engine would rather say so.
    ("SELECT(k > 1, 5, [1, 2])", "SELECT(k > 1, 5, [1, 2])"),
    ("SELECT(PRIME(k), k, x, x + 2)", "SELECT(PRIME(k), k, x, x + 2)"),
]


@pytest.mark.parametrize(("text", "expected"), SELECTED, ids=str)
def test_selected_elements(text, expected):
    assert simp(text) == expected


#: The two iteration functions, which differ only in how much of the sequence
#: they answer with.
ITERATION = [
    # `n` updates make `n + 1` iterates; ITERATE keeps the last of them.
    ("ITERATES(x^2, x, 2, 3)", "[2, 4, 16, 256]"),
    ("ITERATE(x^2, x, 2, 3)", "256"),
    ("ITERATES(x^2, x, 2, 0)", "[2]"),
    # The manual's POWER: `n` copies of `x` multiplied into an accumulator.
    ("ITERATE(a*x, a, 1, 3)", "x^3"),
    # Uncounted, the sequence runs until a value comes round, and the repeat is
    # the last element. ITERATE wants a cycle of length one and answers `?`
    # where it finds a longer one.
    ("ITERATES(1/x, x, 2)", "[2, 1/2, 2]"),
    ("ITERATE(1/x, x, 2)", "?"),
    # A vector of variables and a vector of their starting values, which is how
    # the manual writes Fibonacci without subscripts.
    ("ITERATE([k, j + k], [j, k], [0, 1], 10)", "[55, 89]"),
    ("ITERATES([k, j + k], [j, k], [0, 1], 3)", "[[0, 1], [1, 1], [1, 2], [2, 3]]"),
    # A negative count iterates the inverse, which is what MISC.MTH defines
    # `INVERSE(u, x) := ITERATE(u, x, x, -1)` on.
    ("ITERATES(TAN(x), x, x, -1)", "[x, ATAN(x)]"),
    ("ITERATES(x + 1, x, 0, -2)", "[0, -1, -2]"),
    # Where there is a choice of inverses there is no inverse function, and the
    # call comes back as written. Derive takes the principal one and answers
    # `[2, SQRT(2)]`; choosing among roots is soLve's business, and an
    # iteration is not going to make that choice on its own.
    ("ITERATES(x^2, x, 2, -1)", "ITERATES(x^2, x, 2, -1)"),
    # An iteration that neither comes round nor was counted comes back as
    # written: Derive runs one until memory is gone, which is no answer to
    # give. Neither is a count that is still a name.
    ("ITERATES(x^2, x, 2)", "ITERATES(x^2, x, 2)"),
    ("ITERATE(a*x, a, 1, n)", "ITERATE(a*x, a, 1, n)"),
]


@pytest.mark.parametrize(("text", "expected"), ITERATION, ids=str)
def test_iteration(text, expected):
    assert simp(text) == expected


def test_an_element_access_that_could_not_be_taken_is_taken_once_generated():
    # An access into a vector converts before the index is a number, and becomes
    # an inert one. Each generated element has a number for it, so the access is
    # worth making again - which is how the demo's outer product comes out.
    assert simp("VECTOR([[a, b, c] SUB i], i, 3)") == "[[a], [b], [c]]"
    assert simp("VECTOR([a, b, c] SUB i + 1, i, 2)") == "[a + 1, b + 1]"


# -- statistics ---------------------------------------------------------------

#: The three forms the statistical functions take - written-out arguments, one
#: vector, one matrix whose rows are each a sample - and the convention the
#: manual fixes: the variance is the unbiased sample one, over `n - 1`, and the
#: standard deviation is its square root.
STATISTICS = [
    # Both spellings of one sample, which is the manual's own pair.
    ("AVERAGE(2, 3, 4)", "3"),
    ("AVERAGE([2, 3, 4])", "3"),
    ("RMS([2, 3, 5])", "SQRT(114)/3"),
    # The manual's matrix example: the statistic of each row, as a vector.
    ("RMS([[3, 4, 5], [5, 12, 13]])", "[5*SQRT(6)/3, 13*SQRT(6)/3]"),
    ("AVERAGE([[1, 2], [3, 4]])", "[3/2, 7/2]"),
    # One sample answered by all four, which is what pins the convention down:
    # over `n - 1` the variance is 20/3, and over `n` it would be 5.
    ("AVERAGE(2, 4, 6, 8)", "5"),
    ("RMS(2, 4, 6, 8)", "SQRT(30)"),
    ("VAR(2, 4, 6, 8)", "20/3"),
    ("STDEV(2, 4, 6, 8)", "2*SQRT(15)/3"),
    # The formulas are written over `n` arguments for any `n`, so a sample of
    # one is a sample: its mean is itself, and its root mean square is its
    # magnitude.
    ("AVERAGE(x)", "x"),
    ("RMS(x)", "ABS(x)"),
    # A call nobody could make yet is no sample at all. Reading one as a sample
    # of one would answer `ABS` of a vector that has not been built, which is
    # what a definition holding `RMS(VECTOR(...))` would become.
    ("RMS(VECTOR(u, k, n))", "RMS(VECTOR(u, k, n))"),
    # A sample of one deviates from its own average by nothing, and Derive
    # answers the zero that sums to rather than the `0/0` the formula reads as.
    ("VAR([5])", "0"),
    ("VAR(x)", "0"),
    ("STDEV([5])", "0"),
]


@pytest.mark.parametrize(("text", "expected"), STATISTICS, ids=str)
def test_statistics(text, expected):
    assert simp(text) == expected


# -- what an expression is made of, and a vector rebuilt ----------------------

STRUCTURE = [
    # Syntactic terms, so the power is not multiplied out: the manual's own
    # example, and the reason it tells the caller to compose with EXPAND.
    ("TERMS(x*(a + b)^2 + c)", "[x*(a + b)^2, c]"),
    ("TERMS(x^2 + x + 1)", "[x^2, x, 1]"),
    # What is no sum has one term, and a vector distributes.
    ("TERMS(x*y)", "[x*y]"),
    ("TERMS([x + 1, y + 2])", "[[x, 1], [y, 2]]"),
    # Most main to least: the order list `x`, `y`, `z` first and the rest
    # alphabetically, which is the order every other command offers them in.
    ("VARIABLES(x^2 + a*y)", "[x, y, a]"),
    ("VARIABLES(a*b*z)", "[z, a, b]"),
    ("VARIABLES(5)", "[]"),
    # NUMBER is a predicate and answers a truth-value whichever way it comes
    # out. The manual's example is the second pair: true exactly where the sum
    # of two squares is a perfect square.
    ("NUMBER(2/3)", "true"),
    ("NUMBER(x)", "false"),
    ("NUMBER(SQRT(9))", "true"),
    ("NUMBER(SQRT(2))", "false"),
    # Which is what the utility files use it for: an argument still standing as
    # the variable it was written as is an argument nobody supplied.
    ("IF(NUMBER(d), d, 1)", "1"),
    # Counting from 1, and a matrix's elements are its rows.
    ("DELETE_ELEMENT([a, b, c], 2)", "[a, c]"),
    ("DELETE_ELEMENT([[1, 2], [3, 4], [5, 6]], 2)", "[[1, 2], [5, 6]]"),
    ("DELETE_ELEMENT([a], 1)", "[]"),
    # The value to write in comes first and the vector second; the index
    # defaults to 1.
    ("REPLACE_ELEMENT(d, [a, b, c], 2)", "[a, d, c]"),
    ("REPLACE_ELEMENT(d, [a, b, c])", "[d, b, c]"),
    ("REPLACE_ELEMENT([7, 8], [[1, 2], [3, 4]], 1)", "[[7, 8], [3, 4]]"),
    # An index that is no index yet, and one that is past the end: both come
    # back the call they were written as.
    ("DELETE_ELEMENT([a, b, c], n)", "DELETE_ELEMENT([a, b, c], n)"),
    ("DELETE_ELEMENT([a, b, c], 4)", "DELETE_ELEMENT([a, b, c], 4)"),
    # Inserted before the nth element, and the index defaults to 1. One past
    # the end is an index here where it is none above, because that is how an
    # element is added to the end.
    ("INSERT_ELEMENT(d, [a, b, c], 2)", "[a, d, b, c]"),
    ("INSERT_ELEMENT(d, [a, b, c])", "[d, a, b, c]"),
    ("INSERT_ELEMENT(d, [a, b, c], 4)", "[a, b, c, d]"),
    # Two past it is no index at all, and the call comes back as written -
    # DELETE_ELEMENT's rule above, and this engine's throughout. Derive answers
    # `[a, b, c]`, quietly dropping the element it was asked to insert.
    ("INSERT_ELEMENT(d, [a, b, c], 5)", "INSERT_ELEMENT(d, [a, b, c], 5)"),
    # Reversed, and a matrix's elements are its rows there as everywhere.
    ("REVERSE_VECTOR([a, b, c])", "[c, b, a]"),
    ("REVERSE_VECTOR([[1, 2], [3, 4]])", "[[3, 4], [1, 2]]"),
    # Concatenation: the elements of each vector, run together. Matrices stack,
    # their elements being their rows - except for a single matrix, which the
    # manual makes the exception and flattens.
    ("APPEND([a, b], [c, d], [e, f])", "[a, b, c, d, e, f]"),
    ("APPEND([[a, b], [c, d], [e, f]])", "[a, b, c, d, e, f]"),
    ("APPEND([[a, b], [c, d]], [[e, f], [g, h]])", "[[a, b], [c, d], [e, f], [g, h]]"),
    ("APPEND([a, b])", "[a, b]"),
]


@pytest.mark.parametrize(("text", "expected"), STRUCTURE, ids=str)
def test_structure(text, expected):
    assert simp(text) == expected


#: Section 6.14's other decomposition functions: what an expression is written
#: as the product of, which side of a relation is which, and the polynomial
#: division and gcd that sit beside `TERMS` in the manual.
DECOMPOSITION = [
    # Syntactic factors, `TERMS`'s counterpart: nothing is factored first, so a
    # sum is one factor and so is a number - `FACTOR` is what factors.
    ("FACTORS(3*x*(x + 1)^2)", "[(x + 1)^2, x, 3]"),
    ("FACTORS(x^2 - 1)", "[x^2 - 1]"),
    ("FACTORS(12)", "[12]"),
    ("FACTORS(2/3)", "[2/3]"),
    ("FACTORS(x/y)", "[x, 1/y]"),
    # Most main first, and the compound before the simple where two are about
    # the same variable.
    ("FACTORS(3*x*y)", "[x, y, 3]"),
    ("FACTORS(x^2*y^3)", "[x^2, y^3]"),
    ("FACTORS(x*SIN(x))", "[SIN(x), x]"),
    ("FACTORS((x + 1)*(x + 2))", "[x + 2, x + 1]"),
    # A vector distributes, as it does for TERMS.
    ("FACTORS([x*y, 2*z])", "[[x, y], [z, 2]]"),
    # The sides of a relation, and a vector of them - which is what makes
    # `RHS(SOLVE(u, x))` the vector of roots.
    ("LHS(2*x + 3 = 5)", "2*x + 3"),
    ("RHS(2*x + 3 = 5)", "5"),
    ("LHS([x = 1, y = 2])", "[x, y]"),
    ("RHS([x = 1, y = 2])", "[1, 2]"),
    # What is no relation has no sides to take and answers with itself.
    ("LHS(2*x + 3)", "2*x + 3"),
    ("RHS(3)", "3"),
    # Division in the main variable: the manual's own pair.
    ("QUOTIENT(x^4 + 3*x^3 + 5*x + 6, x^2 - 5)", "x^2 + 3*x + 5"),
    ("REMAINDER(x^4 + 3*x^3 + 5*x + 6, x^2 - 5)", "20*x + 31"),
    ("QUOTIENT(x, x^2)", "0"),
    ("REMAINDER(x, x^2)", "x"),
    # The other variables ride along in the coefficients as a field would: in
    # `x` the divisor is a constant, and a constant divides exactly.
    ("QUOTIENT(x*y + 1, y)", "x + 1/y"),
    ("REMAINDER(x*y + 1, y)", "0"),
    # Two numbers have no variable to divide in and are a field on their own,
    # so the quotient is the fraction and nothing is left over.
    ("QUOTIENT(7, 2)", "7/2"),
    ("REMAINDER(7, 2)", "0"),
    ("QUOTIENT(-7, 2)", "-7/2"),
    # And what is a polynomial in a kernel is divided in the kernel.
    ("QUOTIENT(SIN(x), SIN(x)^2)", "0"),
    ("REMAINDER(SIN(x), SIN(x)^2)", "SIN(x)"),
    ("QUOTIENT(SIN(x), x)", "SIN(x)/x"),
    # The manual's gcd example, and the same function over numbers.
    ("POLY_GCD(x^3 + 3*x^2 + 5*x + 6, x^3 + 2*x - 3)", "x^2 + x + 3"),
    ("POLY_GCD(12, 18)", "6"),
]


@pytest.mark.parametrize(("text", "expected"), DECOMPOSITION, ids=str)
def test_what_an_expression_decomposes_into(text, expected):
    assert simp(text) == expected


#: The two number-theoretic functions, both of them predicates over the
#: integers in the sense that they answer rather than compute.
NUMBER_THEORY = [
    ("PRIME(7)", "true"),
    ("PRIME(8)", "false"),
    ("PRIME(1)", "false"),
    # The manual's second argument is how many rounds of a probabilistic test
    # to run. This test is not probabilistic, so the count changes nothing.
    ("PRIME(7, 20)", "true"),
    # What is no integer is not false but undecided, and comes back as written.
    ("PRIME(x)", "PRIME(x)"),
    # Strictly larger, and the argument need be no integer.
    ("NEXT_PRIME(1000)", "1009"),
    ("NEXT_PRIME(7)", "11"),
    ("NEXT_PRIME(7/2)", "5"),
]


@pytest.mark.parametrize(("text", "expected"), NUMBER_THEORY, ids=str)
def test_number_theory(text, expected):
    assert simp(text) == expected


def test_the_utility_files_appended_columns():
    # ``APPEND_COLUMNS(A, B) := APPEND(A`, B`)` ``, VECTOR.MTH's definition and
    # the exercise the manual sets. It works only because appending two
    # matrices stacks their rows rather than flattening them.
    columns = "APPEND([[a, b], [c, d]]`, [[e, f], [g, h]]`)`"
    assert simp(columns) == "[[a, b, e, f], [c, d, g, h]]"


def test_the_utility_files_matrix_minor():
    # ``MINOR(a,i,j) := DELETE_ELEMENT(DELETE_ELEMENT(a,i)`,j)` ``, VECTOR.MTH's
    # definition and the exercise the manual sets: delete row `i`, transpose,
    # delete what was column `j`, transpose back.
    body = parse("DELETE_ELEMENT(DELETE_ELEMENT(a,i)`,j)`")
    context = Context(functions={"MINOR": (("a", "i", "j"), body)})
    assert simp("MINOR([[1,2,3],[4,5,6],[7,8,9]],2,3)", context) == "[[1, 2], [7, 8]]"


# -- relations and logic ------------------------------------------------------


def test_a_relation_that_decides_is_decided_and_one_that_does_not_stands():
    """A relation is a statement, and a statement that says something definite
    is simplified by saying it. Every row here was checked against Derive 4.11
    under DOSBox, which answers all six `true` or `false`.

    The judgement is three-valued: `3*x = x^2 + c` settles nothing and so comes
    back as it was authored, sides simplified and the relation standing. That
    is what leaves a conditional's test alone, and what leaves soLve something
    to solve."""
    assert simp("2 < 3") == "true"
    assert simp("2 = 2") == "true"
    assert simp("1 = 2") == "false"
    assert simp("x = x") == "true"
    assert simp("2 <= 2") == "true"
    assert simp("x < x + 1") == "true"
    assert simp("x + 2*x = c + x*x") == "3*x = x^2 + c"


def test_only_the_reals_are_ordered_so_a_complex_variable_decides_nothing():
    """`x < x + 1` is a fact about the reals and says nothing about the complex
    plane, where `<` has no meaning to be true of. Equality is a question every
    domain answers, so it is decided there too."""
    assert simp("x < x + 1", COMPLEX_X) == "x < x + 1"
    assert simp("x = x", COMPLEX_X) == "true"


def test_relations_joined_over_one_variable_are_solved():
    assert simp("6 >= -2*x AND 3*x /= -9") == "x > -3"
    assert simp("x < 1 AND x > 3") == "false"


def test_a_range_is_written_as_the_chain_the_original_writes():
    """Two bounds on one variable are a range, and a range has a spelling of
    its own that the grammar reads back. The two strictnesses are independent,
    so a half-open range is a chain too."""
    assert simp("x > -2 AND x < 2") == "-2 < x < 2"
    assert simp("x >= -2 AND x < 2") == "-2 <= x < 2"


def test_a_conjunction_it_cannot_solve_keeps_its_shape():
    assert simp("x >= 1 OR x <= -1") == "x >= 1 OR x <= -1"
    # Nothing is solved here - two variables are two unknowns - and the pair is
    # still a range around `y`, which is written as the chain it is.
    assert simp("x < y AND y < 1") == "x < y < 1"


LOGIC = [
    ("NOT NOT p", "p"),
    ("NOT p", "NOT p"),
    # The algebra: absorption, the operators that are spellings for the other
    # three, a variable and its negation, and a conjunct common to every
    # disjunct. A product of sums is left folded, which is where the algebra
    # stops - Derive does not multiply one out into a sum of products.
    ("p AND (q OR p)", "p"),
    ("p OR NOT p", "true"),
    ("p AND NOT p", "false"),
    ("p IMP q", "NOT p OR q"),
    ("p XOR q", "NOT p AND q OR p AND NOT q"),
    ("NOT (p AND q)", "NOT p OR NOT q"),
    ("p AND (q OR r)", "p AND (q OR r)"),
    ("p AND r OR p AND q", "p AND (q OR r)"),
    # Operands are put in the order list's order, a term sorting by the
    # variable it leads with, and a negated literal coming first on a tie.
    ("s AND r OR q AND p", "p AND q OR r AND s"),
    ("c OR b AND a", "a AND b OR c"),
    ("p AND q OR NOT p AND r", "NOT p AND r OR p AND q"),
    # Six variables is past where the algebra pays: both normal forms are read
    # off a truth table, and this one comes back longer than it went in.
    ("a XOR b XOR c XOR d XOR e XOR f", "a XOR b XOR c XOR d XOR e XOR f"),
    # Boolean on booleans, bitwise on integers.
    ("3 OR 5", "7"),
    ("NOT 5", "-6"),
    ("12 AND 10", "8"),
    ("3 XOR 5", "6"),
    ("3 AND p", "3 AND p"),
]


@pytest.mark.parametrize(("text", "expected"), LOGIC, ids=str)
def test_logic(text, expected):
    assert simp(text) == expected


#: A truth table: the columns as they were written, then one row per
#: assignment, the last variable changing fastest and `true` before `false`.
TRUTH_TABLES = [
    (
        "TRUTH_TABLE(p, q, p AND q, p OR q, p XOR q, p IMP q)",
        "[[p, q, p AND q, p OR q, p XOR q, p IMP q], "
        "[true, true, true, true, false, true], "
        "[true, false, false, true, true, false], "
        "[false, true, false, true, true, true], "
        "[false, false, false, false, false, true]]",
    ),
    # Nothing but variables is the table of the assignments themselves.
    ("TRUTH_TABLE(p)", "[[p], [true], [false]]"),
    (
        "TRUTH_TABLE(p, q)",
        "[[p, q], [true, true], [true, false], [false, true], [false, false]]",
    ),
    ("TRUTH_TABLE(p, NOT p)", "[[p, NOT p], [true, false], [false, true]]"),
    # A leading argument that is no variable leaves nothing to vary.
    ("TRUTH_TABLE(3, p)", "TRUTH_TABLE(3, p)"),
]


@pytest.mark.parametrize(("text", "expected"), TRUTH_TABLES, ids=str)
def test_truth_tables(text, expected):
    assert simp(text) == expected


# -- IF, and the test it is asked whether it holds -----------------------------

CONDITIONALS = [
    ("IF(2 = 2, 1, 2)", "1", None),
    ("IF(2 = 3, 1, 2)", "2", None),
    # No else clause and a false test: the value is unknown.
    ("IF(2 = 3, 1)", "?", None),
    ("IF(x > 0, 1, -1)", "1", POSITIVE_X),
    # Undecidable, with an unknown clause to fall back on.
    ("IF(x > 0, 1, -1, 0)", "0", None),
    # Undecidable, with none: the whole expression comes back.
    ("IF(x > 0, 1, -1)", "IF(x > 0, 1, -1)", None),
]


@pytest.mark.parametrize(("text", "expected", "context"), CONDITIONALS, ids=str)
def test_a_conditional_takes_the_branch_its_test_decides(text, expected, context):
    assert simp(text, context) == expected


def test_an_undecidable_conditional_keeps_its_branches_unsimplified():
    # Derive's own behaviour, and the safe one: a branch that is never taken
    # may be nonsense the guard exists to avoid.
    assert simp("IF(x > 0, (x + 1)^2 - x^2, -1)") == "IF(x > 0, (x + 1)^2 - x^2, -1)"


UNTOUCHED_CONDITIONALS = [
    # Not even the arithmetic: an arm is shown as it was written.
    ("IF(x > 0, 2 + 3, 4 + 5)", "IF(x > 0, 2 + 3, 4 + 5)"),
    # Nor the order the operands were written in, nor a product a coefficient
    # would be multiplied through.
    ("IF(x > 0, b*a, 400 + 15*(x - 40))", "IF(x > 0, b*a, 400 + 15*(x - 40))"),
    # Nor the side of the test each half was written on.
    ("IF(40 >= h, 1, 2)", "IF(40 >= h, 1, 2)"),
    # An arm of the conditional that was decided is simplified as any other
    # expression is; only the one left standing is shown as written.
    ("IF(2 = 2, 2 + 3, 4 + 5)", "5"),
    ("IF(x > 0, 2 + 3, 4 + 5, 6 + 7)", "13"),
    # A conditional inside a larger expression keeps its place in it.
    ("2*IF(x > 0, 2 + 3, 4 + 5)", "2*IF(x > 0, 2 + 3, 4 + 5)"),
    ("IF(x > 0, IF(y > 0, 1 + 1, 2), 3)", "IF(x > 0, IF(y > 0, 1 + 1, 2), 3)"),
]


@pytest.mark.parametrize(("text", "expected"), UNTOUCHED_CONDITIONALS, ids=str)
def test_an_undecidable_conditional_comes_back_as_it_was_written(text, expected):
    """10.3: Derive answers an undecidable `IF` with the `IF` the author typed.

    Simplifying an arm is what the conditional exists to prevent, and doing the
    arithmetic in one is simplifying it. What comes back is therefore the text
    that went in - which is also what makes the answer stand still, a second
    Simplify having nothing left to change.
    """
    assert simp(text) == expected
    assert simp(expected) == expected


def test_an_undecidable_conditional_keeps_the_variables_it_depends_on():
    """A held-back `IF` must not look constant to what surrounds it.

    Standing it aside under a plain symbol would hide `k` from the sum, and
    sympy would answer `n*IF(k > 0, k, 0)` - having been shown a summand that
    no longer depends on what it is summed over.
    """
    assert simp("SUM(IF(k > 0, k, 0), k, 1, n)") == "SUM(IF(k > 0, k, 0), k, 1, n)"
    assert simp("DIF(IF(x > 0, x^2, 0), x)") == "DIF(IF(x > 0, x^2, 0), x)"


def test_a_decided_conditional_simplifies_the_branch_it_took():
    assert simp("IF(2 = 2, (x + 1)^2 - x^2, -1)") == "2*x + 1"
    assert simp("1 + IF(2 = 2, 3, 4)") == "4"


def test_a_conditional_answer_keeps_the_test_that_qualifies_it():
    """A sum that has a closed form only where it converges answers with that
    condition attached, and the condition is exactly the kind nothing can
    decide: `ABS(x) < 1` is true for some `x` and false for others. So it
    stands, and the answer goes on saying what it is worth.
    """
    assert simp("SUM(k^2*x^k, k, 1, inf)") == (
        "IF(ABS(x) < 1, -x*(x + 1)/(x - 1)^3, SUM(k^2*x^k, k, 1, inf))"
    )


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


#: Commands the engine has no mathematics for yet. Each must come back exactly
#: as it went in, so that a worksheet holding one is not damaged by simplifying
#: it.
OPAQUE = [
    "MY_FUNCTION(x, 2)",
    "ANOTHER(1)",
]


@pytest.mark.parametrize("text", OPAQUE, ids=str)
def test_what_the_engine_has_no_mathematics_for_passes_through(text):
    assert simp(text) == text


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


# -- precision ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    # Two thirds is cut to `0.666666`, not rounded to `0.666667`: the
    # approximation of it is two thirds, and six digits of that are sixes.
    [("pi", "3.14159"), ("SQRT(3)", "1.73205"), ("1/2 + 1/6", "0.666666")],
    ids=str,
)
def test_approximate_mode(text, expected):
    assert simp(text, APPROXIMATE) == expected


@pytest.mark.parametrize(
    ("text", "expected"), [("pi", "3.14159"), ("SQRT(3)", "1.73205")], ids=str
)
def test_approx_is_simplify_at_another_precision(text, expected):
    # The same answers from an exact context, which is what the manual says
    # approX is: Simplify with the precision mode set to Approximate.
    assert approx(parse(text), Context()).text == expected


def test_approx_takes_the_digits_it_is_given():
    assert approx(parse("pi"), Context(), 12).text == "3.14159265358"


APPROXIMATED = [
    # The function, not the command: it approximates what it holds and leaves
    # the line around it alone, and it does so in Exact mode like any other.
    # What it leaves is a rational, since that is what an approximate number
    # is, so a rational notation writes the ratio the digits stand for.
    ("APPROX(pi)", "355/113"),
    ("APPROX([1/3, pi])", "[1/3, 355/113]"),
    ("1/2 + APPROX(1/3)", "5/6"),
    # A whole number needs no digits, here as everywhere else.
    ("APPROX(2 + 3)", "5"),
    # Nothing to approximate is nothing done.
    ("APPROX(x + 1)", "x + 1"),
]


@pytest.mark.parametrize(("text", "expected"), APPROXIMATED, ids=str)
def test_the_approx_function_approximates_where_it_stands(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("APPROX(pi)", "3.14159"),
        ("APPROX([1/3, pi])", "[0.333333, 3.14159]"),
        ("1/2 + APPROX(1/3)", "0.833333"),
    ],
    ids=str,
)
def test_what_the_approx_function_leaves_shows_its_digits(text, expected):
    # The same values as above, under a notation that writes them as digits:
    # what an approximation is and how it is written are two questions, and
    # the function answers only the first.
    assert simp(text, DECIMAL) == expected


def test_the_approx_function_takes_the_digits_it_is_given():
    # Twelve digits of pi, which is a nearer rational than six digits' 355/113.
    assert simp("APPROX(pi, 12)") == "5419351/1725033"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Special functions sympy does not prove finite, which are finite.
        ("APPROX(LI(2), 15)", "1.04516"),
        ("APPROX(SI(2), 15)", "1.60541"),
        ("APPROX(EI(2), 15)", "4.95423"),
        ("APPROX(COSH_INT(2), 15)", "2.45266"),
        # What has no exact answer still has digits, which is the original's
        # own division of labour between Simplify and approX.
        ("APPROX(INT(SIN(x)/x, x, 1, 2), 15)", "0.659329"),
        ("APPROX(SUM(1/(k^3 + 1), k, 1, inf), 15)", "0.686503"),
        # An infinity is refused, being the one thing that has no digits.
        ("APPROX(LN(0), 15)", "±inf"),
        ("APPROX(1/0, 15)", "±inf"),
    ],
    ids=str,
)
def test_a_value_sympy_leaves_open_is_approximated_rather_than_refused(text, expected):
    """Finiteness is asked loosely, the way rationality is.

    Sympy answers `is_finite` with `None` for `SI`, `EI`, `LI` and for any
    integral or sum it has not worked out, and demanding a proof of finiteness
    left all of them written as themselves. Only what is known to be infinite
    has no digits.

    Written under the notation that shows digits, so these are the six the
    notation asks for and not the fifteen the approximation was made to.
    """
    assert simp(text, DECIMAL) == expected


def test_an_approximation_holds_the_digits_it_was_asked_for():
    # The value behind the six shown above: asking for more digits gives a
    # nearer rational, which is what makes the approximation the one requested
    # rather than the one the notation happens to print.
    assert simp("APPROX(LI(2), 6)") == "2615/2502"
    assert simp("APPROX(LI(2), 15)") == "80302381/76832342"


def test_approx_waits_for_the_value_it_is_asked_to_approximate():
    """Which is why the head is held back rather than rounded on conversion.

    What has to be approximated is the answer and not the question: the
    integral is evaluated first, and `APPROX` sees the number it came out as -
    a logarithm here, so that an answer standing at the exact value would be
    visibly a different one.
    """
    assert simp("APPROX(INT(1/x, x, 1, 2))") == "2731/3940"
    assert simp("INT(1/x, x, 1, 2)") == "LN(2)"


def test_a_number_that_needs_no_digits_does_not_get_any():
    # Derive's approximate numbers are the simplest rationals accurate to the
    # precision, and an integer approximates itself.
    assert simp("2 + 3", APPROXIMATE) == "5"
    assert simp("2*y*3", APPROXIMATE) == "6*y"
    assert simp("SQRT(4)", APPROXIMATE) == "2"


def test_exact_mode_reads_a_decimal_as_the_fraction_it_is():
    assert simp("0.1") == "1/10"
    assert simp("2.5*2") == "5"


def test_mixed_mode_approximates_the_irrational_and_keeps_the_rational_exact():
    assert simp("SQRT(3)", MIXED) == "1.73205"
    # Cut, not rounded: the sum is 1178/339, whose six digits are 3.47492.
    assert simp("1/3 + pi", MIXED) == "3.47492"
    # The manual's own case: the difference of the two fractions is computed
    # exactly, so the root of it is exactly 2/3. Approximate mode rounds them
    # first and loses that.
    exercise = "SQRT(3422357/2313 - 1140443/771)"
    assert simp(exercise, MIXED) == "2/3"
    assert simp(exercise, APPROXIMATE) != "2/3"


def test_numerals_come_back_in_the_input_base():
    state = ParseState(input_base=16)
    assert simp("0FF + 1", Context(input_base=16), state) == "100"


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
    # A conditional answer is written as the `IF` the notation has for one.
    (
        "INT (x^2 COS (a x^3 + b), x)",
        "IF(a /= 0, SIN(a*x^3 + b)/(3*a), x^3*COS(b)/3)",
    ),
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


def test_the_demos_matrix_generator():
    # `MATRIX(z,i,m,j,n) := VECTOR(VECTOR(z,j,1,n),i,1,m)`, the definition the
    # demo builds out of two nested generated vectors.
    body = parse("VECTOR(VECTOR(z,j,1,n),i,1,m)")
    context = Context(functions={"MATRIX": (("z", "i", "m", "j", "n"), body)})
    assert simp("MATRIX(i-j,i,2,j,3)", context) == "[[0, -1, -2], [1, 0, -1]]"


def test_the_demos_outer_product():
    # `OUTER(v,w) := VECTOR([v SUB i],i,DIMENSION(v)) . [w]`. The generator's
    # length is a call on its own argument, so nothing generates until the
    # definition is applied.
    body = parse("VECTOR([v SUB i],i,DIMENSION(v)) . [w]")
    context = Context(functions={"OUTER": (("v", "w"), body)})
    assert simp("OUTER([a,b,c],[2,3,4])", context) == (
        "[[2*a, 3*a, 4*a], [2*b, 3*b, 4*b], [2*c, 3*c, 4*c]]"
    )


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

#: Every input this module simplifies, with the context it uses.
EVERY_CASE = [
    *((text, None) for text, _ in ARITHMETIC + RATIONAL + RADICALS + COMBINATORIAL),
    *((text, context) for text, _, context in DOMAINS),
    *((text, context) for text, _, context in TRIGONOMETRY),
    *((text, context) for text, _, context in LOGARITHMS),
    *(
        (text, None)
        for text, _ in CALCULUS
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


@pytest.mark.parametrize(("text", "context"), EVERY_CASE, ids=str)
def test_printing_a_result_is_a_fixed_point(text, context):
    """Print, parse, print: the second text is the first.

    The invariant the worksheet rests on. A result is stored as text and read
    back later, and an answer that changed on the way through would make the
    entry the user is looking at a different expression from the one the label
    refers to.
    """
    context = context or Context()
    once = simplify(parse(text), context)
    twice = from_sympy(to_sympy(once.node, context), context)
    assert twice.text == once.text
    assert to_sexpr(twice.node) == to_sexpr(once.node)


#: The one shape a second Simplify is allowed to change, and the test below
#: this one says what it changes it to.
HEADED = {text for text, _ in TRUTH_TABLES}


@pytest.mark.parametrize(
    ("text", "context"),
    [case for case in EVERY_CASE if case[0] not in HEADED],
    ids=str,
)
def test_simplifying_a_result_again_changes_nothing(text, context):
    """Simplify is idempotent, which is what "sufficiently simple" means.

    A second pass that found more to do would mean the first stopped early.
    """
    context = context or Context()
    once = simplify(parse(text), context)
    assert simplify(once.node, context).text == once.text


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


# -- what the engine does not do yet ------------------------------------------


@pytest.mark.skip(reason="no cube-root denesting: sqrtdenest handles squares only")
def test_a_nested_cube_root_is_denested():
    # Derive gets this one; sympy has no cube-root denesting to borrow.
    assert simp("(243*SQRT(5) - 294*SQRT(3))^(1/3)") == "3*SQRT(5) - 2*SQRT(3)"


#: The expressions the corpus holds that settle only on a second pass are
#: recorded in `test_simplify_corpus.py`, beside the sweep that finds them.
