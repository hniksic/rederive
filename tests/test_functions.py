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
from rederive.engine.context import Precision
from rederive.engine.vocabulary import in_vocabulary


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


# -- an added built-in the user has redefined ------------------------------------
#
# Derive has no `EI`, `SI`, `POLYLOG` or `POLYGAMMA`, so its utility files define them:
# EXP_INT.MTH's `EI(x, m)` is a series of `m` terms and ZETA.MTH's `POLYLOG(n, z, m)` a
# partial sum. The definition owns the name, so no answer may spell the built-in: read
# back, it would call the user's function, and `-EI(-1)` approximated through `EI(x, m)`
# is complex.


def _defining(**bodies):
    """A context defining each function `NAME="params: body"`."""
    functions = {}
    for name, definition in bodies.items():
        parameters, body = definition.split(":")
        functions[name] = (tuple(parameters.split(",")), parse(body))
    return Context(functions=functions)


EXP_INT = _defining(EI="x,m: x + m", SI="z: z", CI="z: z")


@pytest.mark.parametrize(
    ("text", "context", "built_in", "redefined"),
    [
        ("INT(#e^(-t)/t, t, 1, inf)", EXP_INT, "-EI(-1)", "INT(#e^(-t)/t, t, 1, inf)"),
        ("INT(#e^x/x, x)", EXP_INT, "EI(x)", "INT(#e^x/x, x)"),
        ("INT(1/LN(x), x)", EXP_INT, "EI(LN(x))", "INT(1/LN(x), x)"),
        # The same arity and meaning, but still the user's name.
        ("INT(SIN(t)/t, t, 0, x)", EXP_INT, "SI(x)", "INT(SIN(t)/t, t, 0, x)"),
        (
            "SUM(x^k/k^2, k, 1, inf)",
            _defining(POLYLOG="n,z,m: n + z + m"),
            "IF(ABS(x) <= 1, POLYLOG(2, x), SUM(x^k/k^2, k, 1, inf))",
            "SUM(x^k/k^2, k, 1, inf)",
        ),
        # PROBABIL.MTH's `POLYGAMMA(n, z)` takes the name but not `DIGAMMA`'s.
        (
            "DIF(DIGAMMA(x), x)",
            _defining(POLYGAMMA="n,z: n + z"),
            "POLYGAMMA(1, x)",
            "DIF(DIGAMMA(x), x)",
        ),
        (
            "DIF(GAMMA(x), x)",
            _defining(POLYGAMMA="n,z: n + z"),
            "(x - 1)!*DIGAMMA(x)",
            "(x - 1)!*DIGAMMA(x)",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_a_redefined_built_in_is_never_an_answer(text, context, built_in, redefined):
    """Simplified once without the definition and once with it, in either order, since
    integrals are cached."""
    assert simp(text) == built_in
    assert simp(text, context) == redefined
    assert simp(text) == built_in


def test_a_value_the_redefined_built_in_would_spell_is_approximated_as_it_stands():
    """EXP_INT.MTH's `EN(1, 1)`, approximated from its unevaluated integral."""
    state = declaring(*EXP_INT.functions)
    node = parse("INT(#e^(-t)/t, t, 1, inf)", state)
    assert approx(node, EXP_INT, state=state).text == "0.219383"


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
    ("CHI(0, 0, 1)", "(±1)/2 + 1/2"),
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
    # Undefined at zero interest, since the payment term divides by the rate. The original
    # answers `?` rather than taking a limit.
    ("PMT(0, 10, 1000)", "?"),
    # Too few or too many arguments: the call stays unevaluated.
    ("PMT(3)", "PMT(3)"),
    ("PMT(1, 2, 3, 4, 5, 6)", "PMT(1, 2, 3, 4, 5, 6)"),
    # The rate has no closed form, so it is searched for between bounds, which are part
    # of the answer: without bounds, the original fills in its default [0, 1].
    ("RATE(n, p, v)", "RATE(n, p, v, 0, 0, 0, 1)"),
    ("RATE(n, p, v, f, t)", "RATE(n, p, v, f, t, 0, 1)"),
    ("RATE(36, -300, 9000)", "402/39383"),
    ("RATE(36, -300, 9000, 0, 0, 0, 1)", "402/39383"),
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
    """Section 6.13's four cases, checked for the kind of answer each gives.

    Random values can't be pinned, so only the seed's answer is exact; the rest are
    checked for being numbers of the right kind and range.
    """
    assert simp("RANDOM(6)") in {"0", "1", "2", "3", "4", "5"}
    assert 0 <= Fraction(simp("RANDOM(1)")) < 1
    assert simp("RANDOM(-5)") == "5"
    assert Fraction(simp("RANDOM(0)")) >= 0
    # The argument must be an integer; otherwise the call stays unevaluated.
    assert simp("RANDOM(6.5)") == "RANDOM(13/2)"
    assert simp("RANDOM(x)") == "RANDOM(x)"


def test_a_seeded_generator_draws_what_it_drew_before():
    seeded = [simp("RANDOM(-20)")] + [simp("RANDOM(1000)") for _ in range(5)]
    again = [simp("RANDOM(-20)")] + [simp("RANDOM(1000)") for _ in range(5)]
    assert seeded == again


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
