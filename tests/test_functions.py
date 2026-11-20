"""What Simplify makes of special values and named functions.

Covers non-numbers (`0/0` is `?`, `1/0` is `±inf`), the sign of a complex number,
functions defined in terms of others, sympy heads with no Derive name (written by their
definitions), the indicator function and normal distribution, the financial functions,
random numbers, and plus-or-minus. Cases are inline, as in `test_simplify.py`, so that a
failure shows the mathematics.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
import sympy as sp
from simplifying import declared, declaring, parse, simp

from rederive.engine.computing import Context, approx
from rederive.engine.context import Angle, Definition, Precision
from rederive.engine.vocabulary import in_vocabulary
from rederive.model.session import Session
from rederive.syntax import DeriveSyntaxError


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
    # 6.6: every inverse hyperbolic is a logarithm.
    ("ATANH(z)", "LN(z + 1)/2 - LN(1 - z)/2"),
    ("ASINH(z)", "LN(SQRT(z^2 + 1) + z)"),
    # Only the head with a definition is rewritten; a circular function stays as is.
    ("SIN(z) + SINH(z)", "#e^z/2 - #e^(-z)/2 + SIN(z)"),
    ("SIN(z)", "SIN(z)"),
]


@pytest.mark.parametrize(("text", "expected"), BY_DEFINITION, ids=str)
def test_the_functions_that_are_spellings_for_something_else(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("PRODUCT(k + a, k, 1, n)", "(a + n)!/a!"),
        ("PRODUCT(k/(k + a), k, 1, n)", "a!*n!/(a + n)!"),
    ],
    ids=str,
)
def test_a_rising_factorial_is_a_ratio_of_factorials(text, expected):
    """Sympy answers these with the rising factorial of `a + 1`, which equals the ratio of
    gammas except at the gamma poles, where it equals the ratio's limit. Derive 4.11
    gives these answers with `a` undeclared."""
    assert simp(text) == expected
    assert simp(text, declared("a :epsilon Real [0, inf)")) == expected


# Positive, so that the order-one exponential integral is `-EI(-x)`.
X, Y = sp.symbols("x y", positive=True)
# Of unknown sign.
Z = sp.Symbol("z")


@pytest.mark.parametrize(
    ("head", "written"),
    [
        (sp.li(X), sp.Ei(sp.log(X))),
        (sp.Li(X), sp.Ei(sp.log(X)) - sp.Ei(sp.log(2))),
        (sp.expint(1, X), -sp.Ei(-X)),
        (sp.expint(2, X), X * sp.Ei(-X) + sp.exp(-X)),
        (sp.ceiling(X), -sp.floor(-X)),
        (sp.beta(X, Y), sp.gamma(X) * sp.gamma(Y) / sp.gamma(X + Y)),
        (sp.RisingFactorial(X, Y), sp.gamma(X + Y) / sp.gamma(X)),
        (sp.loggamma(X), sp.log(sp.gamma(X))),
        (sp.sinc(X), sp.sin(X) / X),
        (sp.exp_polar(sp.I * sp.pi) * X + sp.polar_lift(Y), Y - X),
    ],
    ids=str,
)
def test_a_sympy_head_without_a_name_is_spelled_by_its_definition(head, written):
    """Each rewrite is checked by form and numerically at a point."""
    assert in_vocabulary(head) == written
    point = {X: sp.Rational(7, 3), Y: sp.Rational(5, 4)}
    assert abs(sp.N(head.subs(point) - written.subs(point))) < 1e-12


@pytest.mark.parametrize(
    ("head", "spelled"),
    [
        (sp.expint(1, Z), -sp.Ei(-Z)),
        (sp.loggamma(Z), sp.log(sp.gamma(Z))),
    ],
    ids=str,
)
def test_a_spelling_that_holds_only_for_a_positive_argument_waits_for_one(head, spelled):
    """`expint(1, z) = -EI(-z)` and `loggamma(z) = LN(GAMMA(z))` hold only for positive
    `z`: at `-1/2` they are off by `pi*#i` and `2*pi*#i`. With the sign of `z` unknown,
    the head is kept."""
    assert in_vocabulary(head) == head
    point = {Z: -sp.Rational(1, 2)}
    assert abs(sp.N(head.subs(point) - spelled.subs(point))) > 1


def test_a_polar_number_inside_a_branch_cut_keeps_its_sheet():
    # `Ei(x*exp_polar(I*pi))` is `EI(-x) + pi*#i`; writing it as `EI(-x)` would drop the
    # `pi*#i`, so the polar marker is kept.
    head = sp.Ei(X * sp.exp_polar(sp.I * sp.pi))
    assert in_vocabulary(head) == head


# -- a built-in the user has redefined ----------------------------------------------
#
# Old utility files define functions that are built in now. A few names yield to such a
# definition, `ZETA` among them; the definition owns the name, so no answer may spell
# the built-in: read back, it would call the user's function. The added built-ins don't
# yield: `EI(x, m)` and `POLYLOG(n, z, m)`, which the utility files define as partial
# sums, are argument forms of the built-ins.


def _defining(**bodies):
    """A context defining each function `NAME="params: body"`."""
    functions = {}
    for name, definition in bodies.items():
        parameters, body = definition.split(":")
        functions[name] = Definition(tuple(parameters.split(",")), parse(body))
    return Context(functions=functions)


#: SPECIAL.MTH of Derive 1.62 defines `ZETA`, which yields to a definition.
SPECIAL_MTH = _defining(ZETA="s: s + 1")


@pytest.mark.parametrize(
    ("text", "context", "built_in", "redefined"),
    [
        ("SUM(1/k^3, k, 1, inf)", SPECIAL_MTH, "ZETA(3)", "SUM(1/k^3, k, 1, inf)"),
        (
            "SUM(1/k^s, k, 1, inf)",
            SPECIAL_MTH,
            "IF(s > 1, ZETA(s), SUM(k^(-s), k, 1, inf))",
            "SUM(k^(-s), k, 1, inf)",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_a_redefined_built_in_is_never_an_answer(text, context, built_in, redefined):
    """Simplified once without the definition and once with it, in either order, since
    sums are cached."""
    assert simp(text) == built_in
    assert simp(text, context) == redefined
    assert simp(text) == built_in


def test_a_value_the_redefined_built_in_would_spell_is_approximated_as_it_stands():
    state = declaring(*SPECIAL_MTH.functions)
    node = parse("SUM(1/k^3, k, 1, inf)", state)
    assert approx(node, SPECIAL_MTH, state=state).text == "1.202056903"


def test_an_added_built_in_is_not_redefined_by_a_utility_file():
    """Derive 6.1's utility files define `EI`, `SI` and `POLYGAMMA`, which are built in
    here; their definitions are refused, so the built-ins keep answering."""
    for text in ("EI(x, m) := x + m", "SI(z) := z", "POLYGAMMA(n, z) := n + z"):
        with pytest.raises(DeriveSyntaxError):
            parse(text)
    assert simp("INT(#e^x/x, x)") == "EI(x)"
    assert simp("DIF(DIGAMMA(x), x)") == "POLYGAMMA(1, x)"


#: The utility files' argument forms of the added built-ins, Derive's partial sums of
#: their series: `EI(x, m)` is `euler_gamma + LN(x) + SUM(x^n/(n*n!), n, 1, m)` and
#: `POLYLOG(n, z, m)` is `SUM(z^k/k^n, k, 1, m)`. Infinitely many terms give the function
#: where the series is known to converge to it.
PARTIAL_SUMS = [
    ("POLYLOG(2, 1/2, 2)", "9/16"),
    ("POLYLOG(2, z, 3)", "z^3/9 + z^2/4 + z"),
    ("POLYLOG(2, 1/2, inf)", "pi^2/12 - LN(2)^2/2"),
    ("EI(1, 2)", "euler_gamma + 5/4"),
    ("EI(2, 1)", "LN(2) + euler_gamma + 2"),
    ("EI(x, 0)", "LN(x) + euler_gamma"),
    ("EI(x, m)", "LN(x) + SUM(x^n/(n*n!), n, 1, m) + euler_gamma"),
    ("EI(2, inf)", "EI(2)"),
    ("EI(-1, inf)", "EI(-1) + #i*pi"),
]


@pytest.mark.parametrize(("text", "expected"), PARTIAL_SUMS, ids=str)
def test_a_partial_sum_form_of_an_added_built_in(text, expected):
    assert simp(text) == expected


def test_a_partial_sum_is_approximated_as_the_sum_it_is():
    assert approx(parse("EI(1, 2)"), Context()).text == "1.827215664"
    assert approx(parse("EI(1)"), Context()).text == "1.895117816"


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


# -- Derive 6's mathematical functions ---------------------------------------------

#: The values the Derive 6.1 help documents for its functions, and results recorded in
#: Derive 5 and 6 worksheets. Where the original writes another of two equal forms
#: (`-ACOT(9/46)` for `-ATAN(46/9)`), the value is what is tested.
DERIVE_6_FUNCTIONS = [
    # Factorials and the gamma function; no double factorial, so `3!!` is `(3!)!`.
    ("VECTOR(k!, k, 0, 9)", "[1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]"),
    ("(3/2)!", "3*SQRT(pi)/4"),
    ("(-1)!", "±inf"),
    ("3!!", "720"),
    ("VECTOR(GAMMA(k), k, 1, 10)", "[1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]"),
    ("GAMMA(z)", "(z - 1)!"),
    (
        "VECTOR(DIGAMMA(k + 1) + euler_gamma, k, 0, 7)",
        "[0, 1, 3/2, 11/6, 25/12, 137/60, 49/20, 363/140]",
    ),
    ("DIGAMMA(1/2)", "-euler_gamma - 2*LN(2)"),
    ("POLYGAMMA(-1, 2)", "0"),
    ("POLYGAMMA(-1, z)", "LN((z - 1)!)"),
    ("POLYGAMMA(0, z)", "DIGAMMA(z)"),
    ("POLYGAMMA(1, 1)", "pi^2/6"),
    ("DIF(DIGAMMA(x), x)", "POLYGAMMA(1, x)"),
    ("PERM(n, n)", "n!"),
    ("COMB(12, 0)*COMB(38, 4)/COMB(50, 4)", "2109/6580"),
    ("VECTOR(VECTOR(COMB(j, k), k, 0, 2), j, 0, 2)", "[[1, 0, 0], [1, 1, 0], [1, 2, 1]]"),
    ("SUM(COMB(n, k), k, 0, n)", "2^n"),
    # Statistics of arguments, of a vector, and of each row of a matrix.
    ("AVERAGE(2, 4, 6, 8)", "5"),
    ("AVERAGE([2, 4, 6, 8])", "5"),
    ("RMS(2, 4, 6, 8)", "SQRT(30)"),
    ("VARIANCE(2, 4, 6, 8)", "20/3"),
    ("VAR(2, 4, 6, 8)", "20/3"),
    ("STDEV(2, 4, 6, 8)", "2*SQRT(15)/3"),
    ("RMS([3, 4, 5; 5, 12, 13])", "[5*SQRT(6)/3, 13*SQRT(6)/3]"),
    ("FIT([x, a*x^2 + b*x + c], [[4, 1], [0, 3], [2, 4]])", "-x^2/2 + 3*x/2 + 3"),
    ("FIT([y, a*y^2 + b*y + c], [[0, 3], [6, 6], [2, 1]])", "3*y^2/8 - 7*y/4 + 3"),
    ("ERF(z, w)", "ERF(w) - ERF(z)"),
    ("NORMAL(109.8, 105, 7.5)", "ERF(8*SQRT(2)/25)/2 + 1/2"),
    # Elementary functions; `e` is a variable, `#e` the constant.
    ("LN(-3)", "LN(3) + #i*pi"),
    ("LN(#e^3)", "3"),
    ("LOG(8, 2)", "3"),
    ("LOG(x)", "LN(x)"),
    ("EXP(z)", "#e^z"),
    ("SIN(45 deg)", "SQRT(2)/2"),
    ("4^3^2", "262144"),
    ("LN(e)", "LN(e)"),
    # Inverse circular functions. The arctangent of the tangent of a rational multiple
    # of pi, written in radicals, is that angle.
    ("ATAN(SQRT(6) - SQRT(3) + SQRT(2) - 2)", "pi/24"),
    ("ATAN(SQRT(3) - 2)", "-pi/12"),
    ("ATAN(SQRT(2))", "ATAN(SQRT(2))"),
    ("ATAN(1, -1)", "3*pi/4"),
    ("ATAN(-1, -1)", "-3*pi/4"),
    ("ACOT(-1)", "3*pi/4"),
    ("ACOT(x, y)", "ATAN(y, x)"),
    ("ATAN(-46/9)", "-ATAN(46/9)"),
    # Infinity of either sign has an angle of either sign.
    ("ATAN(SQRT(3)/0)/2", "±pi/4"),
    ("ASIN(SQRT(3)/2)", "pi/3"),
    # The same inverses in degrees, whatever the angle setting: a number of degrees
    # where the radian inverse has a value, else the call.
    ("ARCTAN(1)", "45"),
    ("ARCSIN(1/2)", "30"),
    ("ARCCOS(-1)", "180"),
    ("ARCTAN(-1, -1)", "-135"),
    ("ARCCOT(-1)", "135"),
    ("ARCCOT(1, -1)", "-45"),
    ("ARCSEC(2)", "60"),
    ("ARCCSC(2)", "30"),
    ("ARCCOS(-SQRT(6)/3)", "ARCCOS(-SQRT(6)/3)"),
    (
        "ARCTAN(SQRT((-4*d^2 - 4*d + 3)/(1 - 2*d)^2))",
        "ARCTAN(SQRT((2*d + 3)/(1 - 2*d)))",
    ),
    ("ARCCOS(COS(2*pi/9))", "40"),
    ("ARCCOS(COS(7*pi/9))", "140"),
    ("DIF(ARCSIN(x), x)", "180/(pi*SQRT(1 - x^2))"),
    ("ARCSIN([0, 1/2])", "[0, 30]"),
    # Complex variable functions, over vectors too.
    ("SIGN(3 + 4*#i)", "3/5 + 4*#i/5"),
    ("SIGN([3, 4])", "[3/5, 4/5]"),
    # The zero vector has no direction.
    ("SIGN([0, 0, 0])", "SIGN([0, 0, 0])"),
    ("ABS(x + #i*y)", "SQRT(x^2 + y^2)"),
    ("ABS([[1, 2], [3, 4]])", "SQRT(30)"),
    ("ABS([1, 2])", "SQRT(5)"),
    ("CONJ(x + #i*y)", "x - #i*y"),
    ("CONJ([1 + #i, 2 - #i])", "[1 - #i, 2 + #i]"),
    ("IM([#i, 2 + 3*#i])", "[1, 3]"),
    ("PHASE(-1)", "pi"),
    (
        "RE((x + #i*y)*#e^(x + #i*y) - (u + #i*v))",
        "x*#e^x*COS(y) - y*#e^x*SIN(y) - u",
    ),
    ("MAX(x, y)", "ABS(x - y)/2 + x/2 + y/2"),
    ("MAX([[1, 5], [7, 2]])", "[5, 7]"),
    # Riemann's and Hurwitz's zeta functions: a pole at 1, closed forms at even
    # arguments, and a shift by a half or a whole number reduces to Riemann's.
    (
        "VECTOR(ZETA(s), s, -4, 4)",
        "[0, 1/120, 0, -1/12, -1/2, ±inf, pi^2/6, ZETA(3), pi^4/90]",
    ),
    (
        "VECTOR(ZETA(s, 1/2), s, -4, 4)",
        "[0, -7/960, 0, 1/24, 0, ±inf, pi^2/2, 7*ZETA(3), pi^4/6]",
    ),
    ("ZETA(s, 1)", "ZETA(s)"),
    ("ZETA(2, 3)", "pi^2/6 - 5/4"),
    ("ZETA(2, 5/2)", "pi^2/2 - 40/9"),
    ("ZETA(2, 1/3)", "ZETA(2, 1/3)"),
    # The dilogarithm is POLYLOG(2, 1 - z), with its exact values.
    ("DILOG(1)", "0"),
    ("DILOG(0)", "pi^2/6"),
    ("DILOG(2)", "-pi^2/12"),
    ("DILOG(1/2)", "pi^2/12 - LN(2)^2/2"),
    ("DILOG(x)", "DILOG(x)"),
    ("DIF(DILOG(x), x)", "LN(x)/(1 - x)"),
    # Constants.
    ("60 deg", "pi/3"),
    ("#i^2", "-1"),
    ("ABS(unit_circle)", "1"),
    ("ABS(3*unit_circle)", "3"),
    ("3*unit_circle", "3*unit_circle"),
]

#: Approximations, which read back as the exact rationals they print as, so their
#: answers are not fixed points of Simplify in exact precision.
DERIVE_6_APPROXIMATED_FUNCTIONS = [
    ("APPROX(STDEV(2, 4, 6, 8), 10)", "2.581988897"),
    (
        "APPROX(VARIANCE([75, 20, 70, 70, 85, 90, 100, 40, 35, 65, 90, 35])"
        "/VARIANCE([20, 35, 55, 50, 65, 40]), 10)",
        "2.672876304",
    ),
    ("APPROX(ERF(1), 15)", "0.842700792949714"),
    ("APPROX(ATAN(1), 10)", "0.7853981633"),
    ("APPROX(ARCCOS(-SQRT(2)/8 - 1/4), 12)", "115.263174354"),
    ("APPROX(ARCCOS(4*SQRT(2)/17 + 7/17), 10)", "41.88204094"),
    ("APPROX(DILOG(1/2), 10)", "0.5822405264"),
    ("APPROX(euler_gamma, 25)", "0.5772156649015328606065120"),
    ("APPROX(pi, 50)", "3.1415926535897932384626433832795028841971693993751"),
]


@pytest.mark.parametrize(("text", "expected"), DERIVE_6_FUNCTIONS, ids=str)
def test_derive_6_functions(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize(("text", "expected"), DERIVE_6_APPROXIMATED_FUNCTIONS, ids=str)
def test_derive_6_approximated_functions(text, expected):
    assert simp(text, Context()) == expected


async def test_an_indicator_with_a_value_at_its_bound_in_a_definition():
    assert await answers(
        "> PRIMEPI(x, d, a) := IF(d = 0, IF(x >= 3, 2, CHI(2, x, 3, 1)), 0)",
        "PRIMEPI(2, 0, 0)",
    ) == ["1"]


def test_the_degree_valued_inverses_ignore_the_angle_setting():
    degrees = Context(angle=Angle.DEGREE)
    assert simp("ARCTAN(1)", degrees) == "45"
    assert simp("ARCSIN(x)", degrees) == "ARCSIN(x)"


def test_an_approximated_degree_valued_inverse_is_in_degrees():
    approximate = Context().with_precision(Precision.APPROXIMATE, 10)
    assert simp("ARCCOS(-SQRT(6)/3)", approximate) == "144.7356103"


#: Section 6.8's sign of a complex number: `z/|z|`, its point on the unit circle. The
#: original answers the first `3/5 + 4*#i/5`. Expressions with a variable stay as is.
COMPLEX_SIGNS = [
    ("SIGN(3 + 4*#i)", "3/5 + 4*#i/5"),
    ("SIGN(2 + #i)", "SQRT(5)*(2 + #i)/5"),
    ("SIGN(#i)", "#i"),
    ("SIGN(-3)", "-1"),
    ("SIGN(x + #i)", "SIGN(x + #i)"),
    # An imaginary multiple of a real quantity lies on the imaginary axis.
    ("PHASE(#i*x)", "pi*SIGN(x)/2"),
    ("PHASE(-3*#i*x)", "-pi*SIGN(x)/2"),
    ("PHASE(#i*(x + 1))", "pi*SIGN(x + 1)/2"),
    ("PHASE(x + #i)", "PHASE(x + #i)"),
]


@pytest.mark.parametrize(("text", "expected"), COMPLEX_SIGNS, ids=str)
def test_the_sign_of_a_complex_number(text, expected):
    assert simp(text) == expected


#: The magnitude of a power is the power of the magnitude, `ABS(w^r) = ABS(w)^r` for
#: real `r`, and of a product the product of the magnitudes.
MAGNITUDES_OF_POWERS = [
    ("ABS((x + y*#i)^2)", "x^2 + y^2"),
    ("ABS((x + y*#i)^3)", "(x^2 + y^2)^(3/2)"),
    ("ABS((x + y*#i)^(-2))", "1/(x^2 + y^2)"),
    ("ABS(SQRT(x + y*#i))", "(x^2 + y^2)^(1/4)"),
    ("ABS(x*(x + y*#i)^2)", "(x^2 + y^2)*ABS(x)"),
    (
        "ABS((x + y*#i - 3/2)^2*(x + y*#i + 3/2)^3)",
        "(x^2 - 3*x + y^2 + 9/4)*(x^2 + 3*x + y^2 + 9/4)^(3/2)",
    ),
    ("SUBST(ABS(a), a, (x + y*#i)^2)", "x^2 + y^2"),
    ("ABS((1 + #i)^5)", "4*SQRT(2)"),
    # A sum is split into its real and imaginary parts.
    ("ABS((x + y*#i)^2 + 1)", "SQRT(x^4 + x^2*(2*y^2 + 2) + y^4 - 2*y^2 + 1)"),
]


@pytest.mark.parametrize(("text", "expected"), MAGNITUDES_OF_POWERS, ids=str)
def test_the_magnitude_of_a_power_of_a_complex_expression(text, expected):
    assert simp(text) == expected


def test_the_magnitude_of_a_power_of_a_complex_variable():
    context = declared("z :epsilon Complex")
    assert simp("ABS(z^2)", context) == "ABS(z)^2"
    assert simp("ABS((1 + #i)/z^3)", context) == "SQRT(2)/ABS(z)^3"


#: Real and imaginary parts of a power of a positive base, split as the exponential
#: `#e^(z*LN(a))`. A real exponent gives no imaginary part.
PARTS_OF_POWERS = [
    ("IM(2^(#i*y))", "SIN(LN(2)*y)"),
    ("RE(2^(#i*y))", "COS(LN(2)*y)"),
    ("RE(2^(x + #i*y))", "2^x*COS(LN(2)*y)"),
    ("IM(2^(x + #i*y))", "2^x*SIN(LN(2)*y)"),
    ("IM(3^(2*#i*y))", "SIN(2*LN(3)*y)"),
    ("RE(2^(1 + #i))", "2*COS(LN(2))"),
    ("IM(2^x)", "0"),
    ("RE(#e^(#i*y))", "COS(y)"),
    (
        "IM(2^(x + #i*y) = (x + #i*y)^6)",
        "2^x*SIN(LN(2)*y) = 6*x^5*y - 20*x^3*y^3 + 6*x*y^5",
    ),
]


@pytest.mark.parametrize(("text", "expected"), PARTS_OF_POWERS, ids=str)
def test_the_parts_of_a_power_of_a_positive_base(text, expected):
    assert simp(text) == expected


def test_the_parts_of_a_power_of_a_declared_positive_base():
    assert simp("IM(a^(#i*y))", declared("a :epsilon Real (0, inf)")) == "SIN(y*LN(a))"


# -- the indicator function and the normal distribution ------------------------

#: Two Section 6 functions defined by a closed form: the answer is the formula unless
#: the arguments are numbers.
INDICATOR_AND_NORMAL = [
    # A difference of signs: 1 between the bounds, 0 outside.
    ("CHI(0, 1/2, 1)", "1"),
    ("CHI(0, 2, 1)", "0"),
    ("CHI(0, -1, 1)", "0"),
    # Reversed bounds give -1, per the manual's `CHI(a, x, b) = -CHI(b, x, a)`.
    ("CHI(1, 1/2, 0)", "-1"),
    # `a` defaults to 0 and `b` to 1, so both indicate the unit interval; with symbolic
    # arguments the formula is the answer.
    ("CHI(x)", "SIGN(x)/2 - SIGN(x - 1)/2"),
    ("CHI(0, x)", "SIGN(x)/2 - SIGN(x - 1)/2"),
    ("CHI(2, x, 1)", "SIGN(x - 2)/2 - SIGN(x - 1)/2"),
    # At a bound the indicator is undecided, since `SIGN(0)` is `±1`. Derive writes
    # `±1/2` for the same number this printer writes `(±1)/2`.
    ("CHI(0, 0, 1)", "±1/2 + 1/2"),
    # A fourth argument is the value at the lower bound, one minus it at the upper; a
    # fifth is the value at the upper bound. Elsewhere it is the indicator, but only
    # where `x` is known to be no bound; otherwise it stands, with both values written
    # out as Derive writes them.
    ("CHI(2, 2, 3, 1)", "1"),
    ("CHI(2, 3, 3, 1)", "0"),
    ("CHI(2, 5/2, 3, 1)", "1"),
    ("CHI(0, 1, 1, 1/2, 1/3)", "1/3"),
    ("CHI(0, 5, 1, 1/2, 1/3)", "0"),
    ("CHI(0, x, 1, 1/2)", "CHI(0, x, 1, 1/2, 1/2)"),
    ("CHI(4, t, 5, 1)", "CHI(4, t, 5, 1, 0)"),
    ("STEP(-3)", "0"),
    ("MAX(#i, 1)", "MAX(#i, 1)"),
    # The cumulative normal distribution, via the error function. Derive writes
    # `(ERF(√2*z/2) + 1)/2`; sympy distributes the half.
    ("NORMAL(z)", "ERF(SQRT(2)*z/2)/2 + 1/2"),
    ("NORMAL(0)", "1/2"),
    # Derive answers `(ERF(√2*z/(2*s) - √2*m/(2*s)) + 1)/2`, dividing `2*s` into each
    # term. The engine puts the argument over the shared `2*s`, following the
    # semi-fraction rule Derive uses elsewhere, so this case is an expected failure
    # below. `ERF` is odd, so sympy moves the negation outside the call; the values
    # agree.
    (
        "NORMAL(z, m, s)",
        "1/2 - ERF(SQRT(2)*m/(2*s) - SQRT(2)*z/(2*s))/2",
    ),
]


#: Cases above whose expected form the engine doesn't produce, mapped to the reason.
INDICATOR_AND_NORMAL_NOT_HELD = {
    "NORMAL(z, m, s)": (
        "Derive's (ERF(...) + 1)/2 contradicts its semi-fraction rule elsewhere, "
        "(x + 1)^2/y + z -> (x^2 + 2*x + y*z + 1)/y"
    ),
}


@pytest.mark.parametrize(("text", "expected"), INDICATOR_AND_NORMAL, ids=str)
def test_the_indicator_and_the_normal_distribution(text, expected, request):
    if text in INDICATOR_AND_NORMAL_NOT_HELD:
        reason = INDICATOR_AND_NORMAL_NOT_HELD[text]
        request.node.add_marker(pytest.mark.xfail(reason=reason, strict=True))
    assert simp(text) == expected


# -- the financial functions --------------------------------------------------

#: Section 6.12's annuity, solved for each of its quantities. The original answers all of
#: these.
ANNUITIES = [
    # The closed forms, arranged as the original writes them.
    ("PMT(i, n, v)", "i*v*(i + 1)^n/(1 - (i + 1)^n)"),
    ("PMT(i, n, v, f, t)", "i*(f + v*(i + 1)^n)/((1 - (i + 1)^n)*(i*t + 1))"),
    ("NPER(i, p, v, f, t)", "LN((p*(i*t + 1) - f*i)/(i*(p*t + v) + p))/LN(i + 1)"),
    # At zero interest the annuity equation is its limit `v + n*p + f = 0`.
    ("PMT(0, 10, 1000)", "-100"),
    ("PVAL(0, 12, 100)", "-1200"),
    ("FVAL(0, 12, 100)", "-1200"),
    ("NPER(0, -100, 1000)", "10"),
    ("PMT(0, 4, 1000, -200, 1)", "-200"),
    # Neither interest nor payments: no number of periods reaches the future value.
    ("NPER(0, 0, 1000)", "?"),
    # Too few or too many arguments: the call stays unevaluated.
    ("PMT(3)", "PMT(3)"),
    ("PMT(1, 2, 3, 4, 5, 6)", "PMT(1, 2, 3, 4, 5, 6)"),
    # The rate has no closed form, so it is searched for between bounds, which are part
    # of the answer: without bounds, the original fills in its default [0, 1].
    ("RATE(n, p, v)", "RATE(n, p, v, 0, 0, 0, 1)"),
    ("RATE(n, p, v, f, t)", "RATE(n, p, v, f, t, 0, 1)"),
    ("RATE(36, -300, 9000)", "7671/751510"),
    ("RATE(36, -300, 9000, 0, 0, 0, 1)", "7671/751510"),
    # A search that finds no rate in the interval leaves the call unevaluated. Bounds
    # come in pairs, so six arguments (a lower bound without an upper) are invalid.
    ("RATE(36, -300, 9000, 0, 0, 0.5, 1)", "RATE(36, -300, 9000, 0, 0, 1/2, 1)"),
    ("RATE(3)", "RATE(3)"),
    ("RATE(1, 2, 3, 4, 5, 6)", "RATE(1, 2, 3, 4, 5, 6)"),
]


@pytest.mark.parametrize(("text", "expected"), ANNUITIES, ids=str)
def test_the_financial_functions(text, expected):
    assert simp(text) == expected


#: 6.12's annuity equation with placeholders. Substituting any financial function for
#: the quantity it computes must make the equation hold.
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
    """Values from the original for four contracts the manual has no example of."""
    ten = Context().with_precision(Precision.APPROXIMATE, 10)
    assert simp("PVAL(5%/12, 24, -200, 1000, 1)", ten) == "3672.749170"
    assert simp("FVAL(5%/12, 24, -200, 1000, 1)", ten) == "3953.231038"
    assert simp("NPER(5%/12, -200, 4000, 1000, 1)", ten) == "25.77406754"
    assert simp("RATE(10, 100, -500)", ten) == "0.1509841447"


def test_a_random_number_is_a_number():
    """Section 6.13's four cases, checked for the kind of answer each gives."""
    assert simp("RANDOM(6)") in {"0", "1", "2", "3", "4", "5"}
    assert 0 <= Fraction(simp("RANDOM(1)")) < 1
    assert simp("RANDOM(-5)") == "5"
    assert Fraction(simp("RANDOM(0)")) >= 0
    # A positive fraction scales a number in [0, 1); anything else stays unevaluated.
    assert 0 <= Fraction(simp("RANDOM(6.5)")) < Fraction(13, 2)
    assert simp("RANDOM(x)") == "RANDOM(x)"
    assert simp("RANDOM(-3/2)") == "RANDOM(-3/2)"


async def answers(*lines):
    """What Simplify answers for each of `lines` in turn, in one fresh session; the
    answer of a line starting with `>` is left out."""
    session = Session(step_limit=-1)
    answers = []
    for line in lines:
        answer = (await session.simplify(line.removeprefix("> "))).text
        if not line.startswith("> "):
            answers.append(answer)
    return answers


async def test_a_session_draws_from_derives_generator():
    """The generator is the original's, so a seeded session draws the numbers its
    worksheets recorded: a seed is a negative argument, and each draw of `RANDOM(n)`
    steps a 32-bit state once (`RANDOM(1)` twice)."""
    assert await answers(
        "RANDOM(-10)",
        "VECTOR(RANDOM(2^32), i, 1, 5)",
        "RANDOM(-10)",
        "VECTOR(FLOOR(10^6*RANDOM(1)), i, 1, 5)",
        "RANDOM(-10)",
        "VECTOR(RANDOM(6), k, 1, 10)",
    ) == [
        "10",
        "[774553435, 803170996, 1017851477, 3401906302, 3287189871]",
        "10",
        "[180339, 236987, 765358, 641051, 468350]",
        "10",
        "[1, 1, 1, 4, 4, 3, 3, 1, 2, 2]",
    ]
    assert await answers(
        "RANDOM(-1292520975)",
        "VECTOR(RANDOM(2^32), i, 10)",
        "> RANDOM(-1292520975)",
        "VECTOR(FLOOR(10^6*RANDOM(1)), i, 10)",
    ) == [
        "1292520975",
        "[547478792, 1691357513, 271827218, 3100759203, 1590744636, 2179531293, "
        "32298630, 3287163831, 128162800, 198370161]",
        "[127469, 63289, 370374, 7520, 29840, 611864, 473472, 310225, 954647, 120034]",
    ]


async def test_a_number_drawn_from_zero_to_one_is_approximate():
    # Derive draws `RANDOM(1)` as an approximate number, so what is computed from it is
    # shown in decimals; the draws themselves are the same.
    logarithm, floors, fraction = await answers(
        "> RANDOM(-10)",
        "-LN(RANDOM(1))/2",
        "> RANDOM(-10)",
        "VECTOR(FLOOR(10^6*RANDOM(1)), i, 1, 5)",
        "> RANDOM(-10)",
        "[RANDOM(1), 1/3]",
    )
    assert logarithm == "0.8564562846"
    assert floors == "[180339, 236987, 765358, 641051, 468350]"
    assert fraction == "[0.1803397748, 0.3333333333]"


async def test_a_new_session_draws_from_the_state_zero():
    assert await answers("VECTOR(RANDOM(100), i, 5)", "RANDOM([2^32, 2^32])") == [
        "[0, 61, 17, 58, 80]",
        "[3859210750, 2886860527]",
    ]


async def test_a_large_number_of_outcomes_draws_all_its_digits():
    # Two consecutive draws recorded in a Derive worksheet; the state before them is the
    # one that gives the first its leading digits.
    assert await answers(
        "> RANDOM(-2336595714)",
        "RANDOM(10^40)",
        "RANDOM(10^40)",
        "[MOD(RANDOM(10^60), 10^20) = 0]",
    ) == [
        "8353277490109198729488714160267139225088",
        "40861891978474490715430917101371268241",
        "[false]",
    ]


async def test_random_permutations_drawn_by_a_program():
    assert await answers(
        "> rperm(n, a, u := [], v, h) := PROG(v := VECTOR(i, i, 1, n), "
        "LOOP(IF(DIM(u) < a, PROG(IF(n > 1, h := RANDOM(n) + 1, h := 1), "
        "u := APPEND(u, [v SUB h]), v := DELETE(v, h), n := n - 1), RETURN(u))))",
        "> RANDOM(-1285230889)",
        "rperm(10, 10)",
        "rperm(10, 10)",
        "rperm(10, 10)",
        "rperm(45, 6)",
        "rperm(45, 6)",
        "SORT(rperm(45, 6))",
    ) == [
        "[3, 2, 4, 8, 1, 7, 6, 5, 10, 9]",
        "[3, 7, 9, 10, 8, 6, 5, 1, 4, 2]",
        "[7, 5, 10, 3, 1, 6, 9, 8, 4, 2]",
        "[24, 20, 15, 42, 43, 1]",
        "[1, 28, 21, 22, 26, 3]",
        "[2, 4, 10, 23, 27, 33]",
    ]


async def test_a_session_split_in_two_draws_the_same_in_both():
    session = Session(step_limit=-1)
    await session.simplify("RANDOM(-7)")
    other = session.copy()
    assert (await session.simplify("RANDOM(1000)")).text == (
        await other.simplify("RANDOM(1000)")
    ).text


def test_a_searched_rate_is_as_close_as_the_search_it_came_from():
    """The rate is found by bisection, so the last digits depend on the search.

    The original answers `0.01020744896` where this answers `0.01020744900`, and its own
    answer shifts with the bounds (`0.01020744901` between `0.005` and `0.02`). Both
    agree on the digits the equation determines.
    """
    ten = Context().with_precision(Precision.APPROXIMATE, 10)
    assert simp("RATE(36, -300, 9000)", ten).startswith("0.010207449")


def test_plus_or_minus_simplifies_its_argument_and_seals_it_in():
    assert simp("±(2 + 3)") == "±5"
    # Arithmetic doesn't propagate the ambiguity (out of scope); the head stays inert
    # instead of guessing a sign.
    assert simp("±1 + 1") == "±1 + 1"


def test_each_plus_or_minus_has_a_sign_of_its_own():
    # Two of them may add up or cancel; a square is one sign twice.
    assert simp("SIGN(0) + SIGN(0)") == "?"
    assert simp("SIGN(0)*SIGN(0)") == "±1"
    assert simp("SIGN(0)^2") == "1"


# -- numerator and denominator -------------------------------------------------

#: Derive reads both off the simplified argument's product structure: factors with
#: negative exponents and the denominator of the numeric coefficient are below the
#: line, and a sum, not put over a common denominator, is its own numerator (D).
#: ODE.MTH's PARTS splits a canonic form so; FIT_.MTH reweights a rational fit by its
#: denominator (see `test_approximate.py`).
PARTS = [
    ("DENOMINATOR(1/x + 1/y)", "1"),
    ("NUMERATOR(1/x + 1/y)", "1/x + 1/y"),
    ("DENOMINATOR(x^(-2) + 1/4)", "1"),
    ("DENOMINATOR(x/4 + 1/3)", "1"),
    ("NUMERATOR(x/4 + 1/3)", "x/4 + 1/3"),
    ("DENOMINATOR(1/x + 1)", "1"),
    ("DENOMINATOR((x + 1)/(x*y))", "x*y"),
    ("NUMERATOR((x + 1)/(x*y))", "x + 1"),
    ("DENOMINATOR(3/4)", "4"),
    ("DENOMINATOR(x/4)", "4"),
    ("DENOMINATOR(x^(-2))", "x^2"),
    ("DENOMINATOR(x^(-k))", "x^k"),
    # The argument is simplified first.
    ("DENOMINATOR((x^2 - 1)/(x - 1))", "1"),
]


@pytest.mark.parametrize(("text", "expected"), PARTS, ids=str)
def test_numerator_and_denominator_read_the_simplified_product(text, expected):
    assert simp(text) == expected


def test_a_factored_argument_is_simplified_before_its_denominator_is_read():
    # ODE.MTH's PARTS for the Bessel equation in `(x - r)^(2*k + 2)*y''` (D): the
    # factored form simplifies to a sum, whose denominator is 1.
    context = declared("k :epsilon Real (0, inf)")
    factored = "FACTOR(z^2*x^(-2*k) + 1/4 - v^2, Trivial)"
    assert simp(f"DENOMINATOR({factored})", context) == "1"
    assert simp(f"NUMERATOR({factored})", context) == "x^(-2*k)*z^2 - v^2 + 1/4"


def test_the_denominator_of_a_factored_quotient_by_a_number_is_that_number():
    # Polynomial arithmetic mod p clears a remainder's denominator so.
    assert simp("DENOMINATOR(FACTOR(x^2/4 + 3, Trivial, x))") == "4"
    assert simp("NUMERATOR(FACTOR(x^2/4 + 3))") == "x^2 + 12"


def test_a_reciprocal_symbolic_power_is_multiplied_into_a_sum_it_cancels_in():
    context = declared("k :epsilon Real (0, inf)")
    assert simp("x^(-k)*(x^k + z)", context) == "x^(-k)*z + 1"
    assert simp("x^(-2)*(x^2 + z)") == "(x^2 + z)/x^2"


async def test_a_long_sum_of_draws_draws_for_each_term():
    """A sum whose body draws is written out term by term even beyond the length that is
    otherwise left to a closed form: summed as one expression over a symbolic index it
    would draw once."""
    (total,) = await answers("SUM(RANDOM(2), i, 1, 2000)")
    assert 800 < int(total) < 1200


async def test_a_sequence_from_sines_of_multiples_of_a_phase_is_integral():
    """`TT(a, b, n)` is `(a^n - b^n)/(a - b)` for conjugate `a` and `b`, an integer."""
    assert await answers(
        "> TT(a, b, n) := (a*b)^((n - 1)/2)*SIN(n*PHASE(a))/SIN(PHASE(a))",
        "VECTOR(TT(2 + 3*#i, 2 - 3*#i, k), k, 0, 10)",
    ) == ["[0, 1, 4, 3, -40, -199, -276, 1483, 9520, 18801, -48556]"]
