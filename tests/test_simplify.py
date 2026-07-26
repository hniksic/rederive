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
import time

import pytest
from sexpr import to_sexpr

from rederive.engine import (
    Context,
    Domain,
    DomainKind,
    approx,
    domain_of_node,
    from_sympy,
    simplify,
    to_sympy,
)
from rederive.engine.context import Angle, Branch, Direction, Precision, TrigPower
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
    # `pi/2` only for positive `t`, and Derive answers it regardless.
    ("ACOT(t) + ATAN(t)", "ACOT(t) + ATAN(t)", None),
]


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
    # Complementary arcs of one argument are a right angle. The tangent pair
    # is the one that needs a domain, being `-pi/2` below zero.
    ("ASIN(x) + ACOS(x)", "pi/2", None),
    ("ASEC(x) + ACSC(x)", "pi/2", None),
    ("2*ASIN(x) + 2*ACOS(x)", "pi", None),
    ("ASIN(x) + ACOS(x) + y", "y + pi/2", None),
    ("ASIN(x) + ACOS(x)", "90", Context(angle=Angle.DEGREE)),
    ("ACOT(t) + ATAN(t)", "pi/2", declared("t :epsilon Real (0, inf)")),
    ("ACOT(t) + ATAN(t)", "-pi/2", declared("t :epsilon Real (-inf, 0)")),
    ("ASIN(x) + ACOS(y)", "ACOS(y) + ASIN(x)", None),
    ("SIN(45)", "SQRT(2)/2", Context(angle=Angle.DEGREE)),
    ("ASIN(1/2)", "30", Context(angle=Angle.DEGREE)),
    # `deg` is the constant pi/180, so degrees work in radian mode too.
    ("SIN(45 deg)", "SQRT(2)/2", None),
    ("SIN(2*x)", "2*SIN(x)*COS(x)", Context(trigonometry=Direction.EXPAND)),
    ("SIN(x)*COS(x)", "SIN(2*x)/2", Context(trigonometry=Direction.COLLECT)),
    ("COS(x)^2", "1 - SIN(x)^2", Context(trigpower=TrigPower.SINES)),
    ("SIN(x)^2", "1 - COS(x)^2", Context(trigpower=TrigPower.COSINES)),
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
]


@pytest.mark.parametrize(("text", "expected", "context"), LOGARITHMS, ids=str)
def test_logarithms_and_exponentials(text, expected, context):
    assert simp(text, context) == expected


# -- the branch setting -------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected", "branch"),
    [
        ("(-8)^(1/3)", "2*(-1)^(1/3)", Branch.PRINCIPAL),
        ("(-8)^(1/3)", "-2", Branch.REAL),
        ("SQRT(x^2)", "ABS(x)", Branch.PRINCIPAL),
        # Any permits the rewrite without a domain proof.
        ("SQRT(x^2)", "x", Branch.ANY),
    ],
    ids=str,
)
def test_which_root_an_expression_may_mean(text, expected, branch):
    assert simp(text, Context(branch=branch)) == expected


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
    ("SUM(k^2, k, 1, m)", "m^3/3 + m^2/2 + m/6"),
    ("PRODUCT(n^2, n, 1, m)", "m!^2"),
    ("LIM(SIN(x)/x, x, 0)", "1"),
    ("LIM(SIGN(x), x, 0, 1)", "1"),
    ("LIM(SIGN(x), x, 0, -1)", "-1"),
    # The order is the maximum degree, and a derivative that simplifies to zero
    # at the expansion point costs a term: this one has no even powers.
    ("TAYLOR(c*SIN(x), x, 0, 6)", "c*x^5/120 - c*x^3/6 + c*x"),
    ("TAYLOR(1/(1 - x), x, 0, 3)", "x^3 + x^2 + x + 1"),
]


@pytest.mark.parametrize(("text", "expected"), CALCULUS, ids=str)
def test_a_calculus_head_is_evaluated(text, expected):
    assert simp(text) == expected


def test_an_integral_a_rewrite_leaves_behind_is_evaluated_too():
    # Cancelling this integrand splits it in two, and the part that is an
    # integral of `x` is evaluated as well: an answer still holding
    # `INT(x, x)` would not be a simplified one.
    assert simp("INT((x*G(x) + F(x))/G(x), x)") == "x^2/2 + INT(F(x)/G(x), x)"


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


def test_a_parametrised_integral_of_an_affordable_shape_is_answered():
    """The heuristic method, where its ansatz is measured to be small.

    A product of powers of polynomials is one generator per base and nothing
    composed on top, so `heurisch` answers this in under a second where the
    bounded methods have nothing for it. Derive answers it as the incomplete
    beta function; sympy's spelling is the hypergeometric it is defined by.
    """
    start = time.monotonic()
    answer = simp("INT(x^(a-1)*(1-x)^(b-1), x)")
    assert time.monotonic() - start < 5
    assert answer == "x^a*HYPER([a, 1 - b], [a + 1], x*EXP_POLAR(2*#i*pi))/a"


def test_a_parametrised_integral_is_answered_or_left_alone_promptly():
    """The heuristic method is asked only where its cost is bounded.

    This line is out of Derive's own utility library, and the answer to it is
    that there is no answer. Reaching that answer through sympy's heuristic
    Risch algorithm takes the better part of a minute and gigabytes of memory,
    because the algorithm's ansatz grows with the three symbolic parameters in
    the integrand. The generous bound below is not a benchmark; it is there so
    that a change which puts the minute back fails loudly.
    """
    text = "INT(t^(a - 1)*(#e^(z*t)*(1 - t)^(b - a - 1) - 1), t, 0, 1/2)"
    start = time.monotonic()
    answer = simp(text)
    assert time.monotonic() - start < 10
    assert answer.startswith("INT(")


def test_the_interior_singularity_derive_missed():
    # Derive answers -2, having integrated straight through the pole at zero.
    # Sympy is right and we keep its answer; this is not a bug to reproduce.
    assert simp("INT(1/x^2, x, -1, 1)") == "inf"


def test_a_two_sided_limit_whose_sides_disagree_is_plus_or_minus():
    assert simp("LIM(SIGN(x), x, 0)") == "±1"


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


# -- relations and logic ------------------------------------------------------


def test_the_sides_of_a_relation_are_simplified_and_the_relation_is_not_decided():
    assert simp("x + 2*x = c + x*x") == "3*x = x^2 + c"
    assert simp("2 = 2") == "2 = 2"
    assert simp("2 < 3") == "2 < 3"


def test_relations_joined_over_one_variable_are_solved():
    assert simp("6 >= -2*x AND 3*x /= -9") == "x > -3"
    assert simp("x < 1 AND x > 3") == "false"


def test_a_conjunction_it_cannot_solve_keeps_its_shape():
    assert simp("x >= 1 OR x <= -1") == "x >= 1 OR x <= -1"
    assert simp("x < y AND y < 1") == "x < y AND y < 1"


LOGIC = [
    ("NOT NOT p", "p"),
    ("NOT p", "NOT p"),
    ("p AND (q OR p)", "p AND (p OR q)"),
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


# -- IF, the one place a test is asked whether it holds ------------------------

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
    "SOLVE(x^2 = 1, x)",
    "RANDOM(10)",
    "ITERATE(x^2, x, 2, 3)",
    "FIT([x, 1], [[1, 2], [3, 4]])",
    "TRUTH_TABLE(p, q)",
    "PMT(1/100, 12, 1000)",
    "MY_FUNCTION(x, 2)",
]


@pytest.mark.parametrize("text", OPAQUE, ids=str)
def test_what_the_engine_has_no_mathematics_for_passes_through(text):
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
    ("(x+a)^2-2*a*x", "a^2 + x^2"),
    # The tenth power stays folded while the square around it is expanded.
    ("(x + (a + 1)^10)^2 - (a + 1)^20", "2*x*(a + 1)^10 + x^2"),
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
        "(3 - 2*y^2)*(x^2 - y^2)^5",
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
]

DEMO_TRIGONOMETRY = [
    ("COS (17/6 pi)", "-SQRT(3)/2"),
    ("SIN (-30 deg)", "-1/2"),
    (
        "(1 - (COS x)^2)^4 (1 - (SIN x)^2)^3 ((SIN x)^2 + (COS x)^2)^5",
        "SIN(x)^8*COS(x)^6",
    ),
    ("(COS (a/2) + SIN (a/2))^2", "SIN(a) + 1"),
    ("CSC (2x) (2 (COS (x/2))^2 - 1)", "1/(2*SIN(x))"),
    ("(1 + SIN a) * (1 + SEC a) / ((1 + COS a) * (1 + CSC a))", "TAN(a)"),
    ("(TAN a)^2 / (1 + (TAN a)^2) * (1 + (COT a)^2) / (COT a)^2", "TAN(a)^2"),
    ("TAN ATAN x", "x"),
    ("ATAN (-1, -1)", "-3*pi/4"),
    ("ASIN (-1/2)", "-pi/6"),
    ("ACOS (- 1 / SQRT 2)", "3*pi/4"),
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
    ("SUM (k, k, 0, n)", "n^2/2 + n/2"),
    ("SUM (k^3, k, 0, n)", "n^4/4 + n^3/2 + n^2/4"),
    ("SUM (2^-k, k, 0, inf)", "2"),
    ("PRODUCT (2 k, k, 1, n)", "2^n*n!"),
]

DEMO_MATRICES = [
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

#: Every input this module simplifies, with the context it uses.
EVERY_CASE = [
    *((text, None) for text, _ in ARITHMETIC + RATIONAL + RADICALS + COMBINATORIAL),
    *((text, context) for text, _, context in DOMAINS),
    *((text, context) for text, _, context in TRIGONOMETRY),
    *((text, context) for text, _, context in LOGARITHMS),
    *((text, None) for text, _ in CALCULUS + SPECIAL + VECTORS + LOGIC + INERT),
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


@pytest.mark.parametrize(("text", "context"), EVERY_CASE, ids=str)
def test_simplifying_a_result_again_changes_nothing(text, context):
    """Simplify is idempotent, which is what "sufficiently simple" means.

    A second pass that found more to do would mean the first stopped early.
    """
    context = context or Context()
    once = simplify(parse(text), context)
    assert simplify(once.node, context).text == once.text


# -- what the engine does not do yet ------------------------------------------


@pytest.mark.skip(reason="no cube-root denesting: sqrtdenest handles squares only")
def test_a_nested_cube_root_is_denested():
    # Derive gets this one; sympy has no cube-root denesting to borrow.
    assert simp("(243*SQRT(5) - 294*SQRT(3))^(1/3)") == "3*SQRT(5) - 2*SQRT(3)"


#: The expressions the corpus holds that settle only on a second pass are
#: recorded in `test_simplify_corpus.py`, beside the sweep that finds them.
