"""Simplification of calculus heads other than integrals.

Covers derivatives, Taylor polynomials, sums and products, and limits. Cases are written
inline as input and expected output, as in `test_simplify.py`, so a failure shows the
mathematics. Integrals are tested in `test_integrate.py`.
"""

from __future__ import annotations

import pytest
from simplifying import declaring, parse, simp

from rederive.engine.computing import Context, simplify
from rederive.model.expr import Kind


# -- the calculus heads -------------------------------------------------------

CALCULUS = [
    ("DIF(SIN(a*x^2), x)", "2*a*x*COS(a*x^2)"),
    ("DIF(x^2, x, 2)", "2"),
    ("INT(x*SIN(x), x)", "SIN(x) - x*COS(x)"),
    ("INT(1/x, x, 1, 2)", "LN(2)"),
    ("INT(1/x^2, x, 1, inf)", "1"),
    # Sympy leaves the divergent case unevaluated; Derive takes the antiderivative to
    # its limit and prints the infinity it diverges to.
    ("INT(x^n, x, 1, inf)", "IF(n < -1, -1/(n + 1), inf)"),
    ("INT(1/x^n, x, 1, inf)", "IF(n > 1, 1/(n - 1), inf)"),
    # The halves either side of 1 cancel, which `x -> 1/x` shows.
    ("INT(LN(x)/(1 + x^2), x, 0, inf)", "0"),
    # The imaginary error function writes this without an imaginary unit.
    ("INT(#e^(x^2), x, 0, 1)", "SQRT(pi)*ERFI(1)/2"),
    # Real integrands over real intervals, for which sympy answers with spurious
    # imaginary parts (`-pi^2/12 + 2*#i*pi*LN(2)`, `LN(t + 1) + #i*pi`). With no other
    # value proved, the integrals stay unevaluated.
    ("INT(LN(x)/(x + 1), x, 0, 1)", "INT(LN(x)/(x + 1), x, 0, 1)"),
    ("INT(LN(x)/(x + 1), x, 1, 2)", "INT(LN(x)/(x + 1), x, 1, 2)"),
    ("INT((x^t - 1)/LN(x), x, 0, 1)", "INT((x^t - 1)/LN(x), x, 0, 1)"),
    # Sympy gives `-EI(EXP_POLAR(#i*pi))`, i.e. `-EI(-1) - pi*#i`. Our own
    # antiderivative `EI(-x)` is continuous on the interval, so it gives the integral.
    ("INT(#e^(-x)/x, x, 1, inf)", "-EI(-1)"),
    # LN is not real left of zero, so the imaginary part is genuine.
    ("INT(LN(x), x, -1, 1)", "-2 + #i*pi"),
    # Sympy has no antiderivative here. Ours is continuous for every `q` (the argument
    # of its logarithm times its conjugate is -1, so never zero), so it gives the
    # integral.
    (
        "INT(SIN(x)/SQRT(1 + q^2*COS(x)^2), x, 0, pi)",
        "(LN(SQRT(q^2 + 1) + q) - LN(SQRT(q^2 + 1) - q))/q",
    ),
    # For `q` > 1 the root in the antiderivative is imaginary on part of the interval
    # and its arctangent isn't proved continuous, so the integral stays.
    (
        "INT(COS(x)/SQRT(1 - q^2*SIN(x)^2), x, -1, 2)",
        "INT(COS(x)/SQRT(1 - q^2*SIN(x)^2), x, -1, 2)",
    ),
    # The integrand changes sign when reflected about the interval's midpoint. This is
    # proved by its sum with the reflection having zero derivative, being continuous,
    # and being zero at the midpoint.
    ("INT(LN(SIN(x)) - LN(COS(x)), x, 0, pi/2)", "0"),
    ("SUM(1/k^2, k, 1, 5)", "5269/3600"),
    ("SUM(k^2, k, 1, m)", "m*(m + 1)*(2*m + 1)/6"),
    ("PRODUCT(n^2, n, 1, m)", "m!^2"),
    # A vector as the third argument lists the index values instead of a range.
    ("SUM(k^2, k, [2, 3, 5, 7, 11])", "208"),
    ("PRODUCT(k^2, k, [2, 3, 5, 7, 11])", "5336100"),
    # A non-vector third argument is the upper end of a range from 1. VECTOR.MTH's
    # EXACT_EIGENVECTOR sums over `n_` up to `DIMENSION(a)` this way.
    ("SUM(k, k, 3)", "6"),
    ("SUM(k, k, 3.5)", "6"),
    ("PRODUCT(k, k, 4)", "24"),
    ("SUM([a, b, c] SUB k, k, DIMENSION([a, b, c]))", "a + b + c"),
    ("SUM(k^2, k, m)", "m*(m + 1)*(2*m + 1)/6"),
    ("PRODUCT(n^2, n, m)", "m!^2"),
    # A vector as one end of a range gives one answer per element, as a vector of
    # points does for LIM. Vectors at both ends aren't paired up.
    ("INT(x^2, x, 0, [1, 2])", "[1/3, 8/3]"),
    ("INT(x, x, [0, 1], 3)", "[9/2, 4]"),
    ("SUM(k, k, [1, 2], 3)", "[6, 5]"),
    ("PRODUCT(k, k, 1, [2, 3])", "[2, 6]"),
    ("SUM(k, k, [1, 2], [3, 4])", "SUM(k, k, [1, 2], [3, 4])"),
    # A vector is not a valid variable.
    ("INT(x, [x, y])", "INT(x, [x, y])"),
    ("DIF(x, [x, y])", "DIF(x, [x, y])"),
    ("SUM(k, [k, j], 1, 2)", "SUM(k, [k, j], 1, 2)"),
    # Vectors and matrices are handled element by element, and keep their shape even
    # when all elements are zero.
    ("DIF([1, 1], x)", "[0, 0]"),
    ("INT([0, 0], x)", "[0, 0]"),
    ("INT([[0, 0], [0, 0]], x, 0, 1)", "[[0, 0], [0, 0]]"),
    ("INT([x, x^2], x)", "[x^2/2, x^3/3]"),
    ("DIF([x^2, x], x, 2)", "[2, 0]"),
    # A definite integral of an equation integrates both sides. Indefinite integrals
    # stay, since each side's antiderivative has its own arbitrary constant.
    ("INT(y = x, x, 0, 1)", "y = 1/2"),
    ("INT(y = x, x)", "INT(y = x, x)"),
    ("DIF(y = x, x, -1)", "DIF(y = x, x, -1)"),
    ("LIM(SIN(x)/x, x, 0)", "1"),
    # An omitted point is zero, for every variable of a vector.
    ("LIM(SIN(x)/x, x)", "1"),
    ("LIM(x + 1, x)", "1"),
    ("LIM(SIN(x)/x + y + 2, [x, y])", "3"),
    ("LIM(SIGN(x), x, 0, 1)", "1"),
    ("LIM(SIGN(x), x, 0, -1)", "-1"),
    # Toward `-inf`, an exponential with base above one vanishes, below one it grows.
    ("LIM(2^x, x, -inf)", "0"),
    ("LIM((1/2)^x, x, -inf)", "inf"),
    ("LIM(x^3, x, -inf)", "-inf"),
    # A vector of variables and points takes the limits one after another, which plain
    # substitution of `x = 0` can't do.
    ("LIM(x^2 + y^2, [x, y], [2, 3])", "13"),
    ("LIM(SIN(x)/x + y, [x, y], [0, 1])", "2"),
    # A matrix limit is taken element by element; SOLVE.MTH relies on this to write an
    # iterate into a whole augmented matrix.
    ("LIM([[SIN(x)/x, x + y], [x^2, 3]], x, 0)", "[[1, y], [0, 3]]"),
    ("LIM([[x*y, x + y], [x^2, y]], [x, y], [1, 2])", "[[2, 3], [1, 2]]"),
    # The order is the maximum degree, so derivatives that vanish at the expansion point
    # drop terms: this one has no even powers.
    ("TAYLOR(c*SIN(x), x, 0, 6)", "c*x^5/120 - c*x^3/6 + c*x"),
    ("TAYLOR(1/(1 - x), x, 0, 3)", "x^3 + x^2 + x + 1"),
]


@pytest.mark.parametrize(("text", "expected"), CALCULUS, ids=str)
def test_a_calculus_head_is_evaluated(text, expected):
    assert simp(text) == expected


def test_a_function_without_a_body_is_differentiated_symbolically():
    # `F(x) :=` gives `F` no body, so the chain rule stops at `F'`, and its gradient is
    # written as unevaluated derivatives.
    state = declaring("F")
    assert simp("DIF(F(x)^3, x)", state=state) == "3*F(x)^2*F'(x)"
    assert simp("GRAD(F(x, y, z))", state=state) == (
        "[DIF(F(x, y, z), x), DIF(F(x, y, z), y), DIF(F(x, y, z), z)]"
    )


SUMMATION_FORMULAS = [
    ("SUM(k, k, 0, n)", "n*(n + 1)/2"),
    ("SUM(k^2, k, 1, n)", "n*(n + 1)*(2*n + 1)/6"),
    ("SUM(k^3, k, 0, n)", "n^2*(n + 1)^2/4"),
    ("SUM(k(k + 1), k, 1, n)", "n*(n + 1)*(n + 2)/3"),
    ("SUM(a k + b, k, 0, n)", "(a*n + 2*b)*(n + 1)/2"),
    # Factored even though that's longer than expanded, to keep the formula's shape.
    # Derive prints `n*(n + 1)*(6*n^3 + 9*n^2 + n - 1)/30`, the same product with one
    # factor fewer.
    ("SUM(k^4, k, 1, n)", "n*(n + 1)*(2*n + 1)*(3*n^2 + 3*n - 1)/30"),
    # A vector is summed element by element, each giving a formula.
    ("SUM([k, k^2], k, 0, n)", "[n*(n + 1)/2, n*(n + 1)*(2*n + 1)/6]"),
]


#: Sums to infinity, answered only where they converge.
INFINITE_SUMS = [
    # A geometric series and its derivative converge only inside the unit circle.
    ("SUM(x^k, k, 0, inf)", "IF(ABS(x) < 1, 1/(1 - x), SUM(x^k, k, 0, inf))"),
    (
        "SUM(k*x^(k-1), k, 1, inf)",
        "IF(ABS(x) < 1, 1/(1 - x)^2, SUM(k*x^(k - 1), k, 1, inf))",
    ),
    # An oscillating sum has no value and stays unevaluated.
    ("SUM((-1)^k, k, 0, inf)", "SUM((-1)^k, k, 0, inf)"),
    # Leibniz's series.
    ("SUM((-1)^(k+1)/(2*k-1), k, 1, inf)", "pi/4"),
    # `k^2 = k*(k - 1) + k`, and each part sums to `#e`.
    ("SUM(k^2/k!, k, 0, inf)", "2*#e"),
    # Zeta where the series converges, the sum where it may not.
    ("SUM(1/k^p, k, 1, inf)", "IF(p > 1, ZETA(p), SUM(k^(-p), k, 1, inf))"),
]


@pytest.mark.parametrize(("text", "expected"), INFINITE_SUMS, ids=str)
def test_an_infinite_sum_is_answered_where_it_converges(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize(("text", "expected"), SUMMATION_FORMULAS, ids=str)
def test_a_summation_formula_keeps_the_shape_it_comes_in(text, expected):
    """Closed forms for sums are factored, as Derive writes them.

    Sympy returns Faulhaber's polynomial expanded. This isn't a general factoring rule:
    the same polynomial typed by hand is left as written (see
    `test_a_polynomial_is_not_factored_for_having_a_common_factor`).
    """
    assert simp(text) == expected


def test_a_range_that_stops_between_two_terms_is_written_out():
    """The index runs up as long as it doesn't pass the end, as in Derive.

    Sympy plugs a non-integer end straight into Faulhaber's polynomial, giving a
    polynomial in the bound instead of a sum of terms. NUMBER.MTH's `PARTS` bounds
    Rademacher's series this way, with `SQRT(n)/LOG(n, 11)` (3.459 for `n` = 4).
    """
    assert simp("SUM(k, k, 1, 3.5)") == "6"
    assert simp("SUM(k^2, k, 1, LN(11)/LN(2))") == "14"
    assert simp("PRODUCT(k, k, 1, 3.5)") == "6"
    # An end just below the start is an empty range.
    assert simp("SUM(k, k, 1, 1/2)") == "0"
    # An end further below gives the negated sum of the terms stepped over, as Derive
    # does.
    assert simp("SUM(k, k, 5, 3.5)") == "-4"
    assert simp("SUM(k, k, 5.5, 3)") == "-8"
    assert simp("PRODUCT(k, k, 5, 3.5)") == "1/4"
    # An infinite range is answered by the closed form.
    assert simp("SUM(2^-k, k, 0, inf)") == "2"


def test_a_polynomial_is_not_factored_for_having_a_common_factor():
    # Factoring is the Factor command's job. Summation formulas keep their shape, but
    # Simplify doesn't factor other polynomials.
    assert simp("n^2/2 + n/2") == "n^2/2 + n/2"
    assert simp("n^2 + n") == "n^2 + n"


def test_a_closed_form_inside_a_larger_answer_is_the_larger_answer_s_to_shape():
    # The formula keeps its shape only as a whole answer; as part of a larger
    # expression it is simplified normally.
    assert simp("SIN(SUM(k, k, 0, n))") == "SIN(n*(n + 1)/2)"
    assert simp("SUM(k, k, 0, n)/n") == "n/2 + 1/2"
    assert simp("SUM(k, k, 0, n) - n^2/2") == "n/2"


def test_a_taylor_polynomial_that_does_not_exist_is_undefined():
    """No derivatives at the point means no polynomial.

    The derivatives of `SQRT(x)` at zero aren't finite, so the answer is `?`, as in
    Derive. Sympy's series `SQRT(x)` is not a polynomial.
    """
    assert simp("TAYLOR(SQRT(x), x, 0, 1)") == "?"
    assert simp("TAYLOR(LN(x), x, 0, 2)") == "?"
    # Order zero needs no derivatives: the manual's `TAYLOR(SQRT(x), x, 0, 0)` is 0.
    assert simp("TAYLOR(SQRT(x), x, 0, 0)") == "0"
    # A symbolic order leaves the call unevaluated.
    assert simp("TAYLOR(SIN(x), x, 0, n)") == "TAYLOR(SIN(x), x, 0, n)"
    # Likewise for other singularities at the point.
    assert simp("TAYLOR(SQRT(x), x, 0, 3)") == "?"
    assert simp("TAYLOR(LN(x), x, 0, 3)") == "?"
    assert simp("TAYLOR(1/x, x, 0, 3)") == "?"


TAYLOR_AT_A_KINK_OR_A_JUMP = [
    # The one-sided expansions differ, and a Taylor polynomial needs derivatives at the
    # point itself.
    ("TAYLOR(ABS(x), x, 0, 2)", "TAYLOR(ABS(x), x, 0, 2)"),
    ("TAYLOR(SQRT(x^2), x, 0, 2)", "TAYLOR(ABS(x), x, 0, 2)"),
    ("TAYLOR(SIGN(x), x, 0, 2)", "TAYLOR(SIGN(x), x, 0, 2)"),
    ("TAYLOR(x*ABS(x), x, 0, 3)", "TAYLOR(x*ABS(x), x, 0, 3)"),
    ("TAYLOR(ABS(SIN(x)), x, 0, 3)", "TAYLOR(ABS(SIN(x)), x, 0, 3)"),
    ("TAYLOR(FLOOR(x), x, 0, 2)", "TAYLOR(FLOOR(x), x, 0, 2)"),
    ("TAYLOR(STEP(x), x, 0, 2)", "TAYLOR(SIGN(x)/2 + 1/2, x, 0, 2)"),
    ("TAYLOR(MAX(x, 0), x, 0, 2)", "TAYLOR(ABS(x)/2 + x/2, x, 0, 2)"),
    # Differentiable as often as the order requires, so the polynomial exists:
    # `x*ABS(x)` once and `ABS(x)^3` twice, both 0.
    ("TAYLOR(x*ABS(x), x, 0, 1)", "0"),
    ("TAYLOR(ABS(x)^3, x, 0, 2)", "0"),
    # Away from the kink or jump, the function is analytic.
    ("TAYLOR(ABS(x - 1), x, 0, 3)", "1 - x"),
    ("TAYLOR(ABS(x), x, 1, 2)", "x"),
    ("TAYLOR(ABS(x), x, -1, 2)", "-x"),
    ("TAYLOR(ABS(SIN(x)), x, pi/2, 3)", "-x^2/2 + pi*x/2 - pi^2/8 + 1"),
    ("TAYLOR(FLOOR(x), x, 1/2, 2)", "0"),
    ("TAYLOR(MOD(x, 1), x, 1/2, 2)", "x"),
    # Smooth but not analytic, with every derivative 0 from both sides.
    ("TAYLOR(#e^(-1/x^2), x, 0, 4)", "0"),
]


@pytest.mark.parametrize(("text", "expected"), TAYLOR_AT_A_KINK_OR_A_JUMP, ids=str)
def test_a_taylor_polynomial_at_a_kink_or_a_jump_stands(text, expected):
    """A function not differentiable at the point has no Taylor polynomial there, so
    the call stays instead of using a one-sided expansion."""
    assert simp(text) == expected


DERIVATIVES_OF_STEPS = [
    # A step function is constant between jumps and not differentiable at them, so its
    # derivative is 0 wherever it exists.
    ("DIF(FLOOR(x), x)", "0"),
    ("DIF(SIGN(x), x)", "0"),
    ("DIF(STEP(x), x)", "0"),
    ("DIF(FLOOR(x), x, 3)", "0"),
    ("DIF(SIGN(SIN(x)), x)", "0"),
    ("DIF(FLOOR(x, 2), x)", "0"),
    ("DIF(FLOOR(x)^2, x)", "0"),
    ("DIF(MOD(x, 1), x)", "1"),
    ("DIF(MOD(x^2, 3), x)", "2*x"),
    # A step times a smooth function is differentiated as if the step were constant.
    ("DIF(SIGN(x)*x, x)", "SIGN(x)"),
    ("DIF(STEP(x)*x^2, x)", "x*(SIGN(x) + 1)"),
    ("DIF(FLOOR(x)*x, x)", "FLOOR(x)"),
    ("DIF(FLOOR(x^2)*SIN(x), x)", "FLOOR(x^2)*COS(x)"),
    ("DIF(SIGN(x^2 - 1)*x^3, x)", "3*x^2*SIGN(x^2 - 1)"),
    # The derivative of an absolute value is a sign, whose derivative is 0.
    ("DIF(ABS(x), x)", "SIGN(x)"),
    ("DIF(ABS(x), x, 2)", "0"),
    ("DIF(x*ABS(x), x, 2)", "2*SIGN(x)"),
    ("DIF(ABS(x^2 - 1), x, 2)", "2*SIGN(x^2 - 1)"),
    ("DIF(ABS(SIN(x)), x, 2)", "-ABS(SIN(x))"),
    ("DIF(MAX(x^2, 1), x, 2)", "SIGN(x^2 - 1) + 1"),
    ("DIF(x*STEP(x), x, 2)", "0"),
    # The antiderivatives of steps differentiate back to them.
    ("DIF(INT(FLOOR(x), x), x)", "FLOOR(x)"),
    ("DIF(INT(SIGN(SIN(x)), x), x)", "SIGN(SIN(x))"),
    ("DIF(INT(MOD(x, 1), x), x)", "x - FLOOR(x)"),
]


@pytest.mark.parametrize(("text", "expected"), DERIVATIVES_OF_STEPS, ids=str)
def test_a_step_function_has_derivative_zero_between_its_jumps(text, expected):
    """Sympy gives a `DiracDelta` at jumps, which the notation can't express, and
    nothing for floor. The answer is 0, the derivative wherever it exists."""
    assert simp(text) == expected


def test_a_derivative_a_substitution_binds_is_not_evaluated():
    """A derivative held under `SUBS` is not evaluated.

    Differentiating an unknown function of a non-variable argument gives the slope of
    `F` there, written as a derivative in a bound variable with a substitution. That
    derivative looks like one of a constant but isn't, so nothing inside the
    substitution is evaluated, and a second pass gives the same answer.

    The argument is written as `LIM(F(y), y, 2*x - y)`, so the chain rule contributes
    the minus. A one-argument function prints as a prime, a two-argument one as the
    explicit substitution; both are stable.
    """
    state = declaring("F", "G")
    once = simplify(parse("DIF(LIM(F(y), y, 2*x - y), y)", state), Context(), state)
    assert once.text == "-F'(2*x - y)"
    assert simplify(once.node, Context(), state).text == once.text
    two = simplify(parse("DIF(LIM(G(y, z), y, 2*x - y), y)", state), Context(), state)
    assert two.text == "-SUBS(DIF(G(x_, z), x_), [x_], [2*x - y])"
    assert simplify(two.node, Context(), state).text == two.text


def test_the_derivative_of_an_undefined_function_is_written_with_primes():
    """`BB'(r)`, as Derive prints and reads it.

    `DIF(BB(r), r)` is twice as long, which makes e.g. a 4x4 matrix of Ricci
    components unreadable. There's one mark per order, on the function rather than
    the call, so the point is whatever the chain rule left there.
    """
    state = declaring("BB")
    assert simp("DIF(BB(r), r)", state=state) == "BB'(r)"
    assert simp("DIF(BB(r), r, 3)", state=state) == "BB'''(r)"
    assert simp("DIF(BB(r)^2, r)", state=state) == "2*BB'(r)*BB(r)"
    assert simp("DIF(BB(r^2), r)", state=state) == "2*r*BB'(r^2)"
    # Primes are read back, in either direction.
    assert simp("BB''(r)", state=state) == "BB''(r)"
    assert simp("DIF(BB'(r), r)", state=state) == "BB''(r)"
    assert simp("INT(BB'(r), r)", state=state) == "BB(r)"


def test_a_prime_at_a_moving_point_is_differentiated_again():
    """`DIF(BB'(r^2), r)` is `2*r*BB''(r^2)`: one more mark, at the same point.

    A prime at a non-variable point is a derivative held there, deliberately opaque.
    Differentiating it doesn't need to look inside: the chain rule raises the order,
    keeps the point, and multiplies by the point's derivative. This matches Derive at
    every order; how higher-order terms are gathered is our own choice.
    """
    state = declaring("BB", "G")
    assert simp("DIF(BB'(r^2), r)", state=state) == "2*r*BB''(r^2)"
    assert simp("DIF(BB''(r^2), r)", state=state) == "2*r*BB'''(r^2)"
    assert simp("DIF(BB'(SIN(r)), r)", state=state) == "BB''(SIN(r))*COS(r)"
    assert simp("DIF(BB'(x*y), x)", state=state) == "y*BB''(x*y)"
    assert simp("DIF(BB'(r^2), r, 2)", state=state) == "4*r^2*BB'''(r^2) + 2*BB''(r^2)"
    assert simp("DIF(BB'(r^2), r, 3)", state=state) == (
        "4*r*(2*r^2*BB''''(r^2) + 3*BB'''(r^2))"
    )
    assert simp("DIF(r*BB'(r^2), r)", state=state) == "2*r^2*BB''(r^2) + BB'(r^2)"
    # Related cases: the same for a plain call, and simpler prime cases.
    assert simp("DIF(BB(r^2), r)", state=state) == "2*r*BB'(r^2)"
    assert simp("DIF(BB(r^2), r, 2)", state=state) == "4*r^2*BB''(r^2) + 2*BB'(r^2)"
    assert simp("DIF(BB'(r), r)", state=state) == "BB''(r)"
    assert simp("DIF(BB'(r)^2, r)", state=state) == "2*BB'(r)*BB''(r)"
    assert simp("DIF(BB'(r^2), r, 0)", state=state) == "BB'(r^2)"
    # A slope at a number is a constant.
    assert simp("DIF(BB'(2), r)", state=state) == "0"
    # A two-argument function can't use primes, so the answer is a `DIF` or a
    # substitution, as Derive writes it.
    assert simp("DIF(G'(x, y), x)", state=state) == "DIF(G(x, y), x, 2)"
    assert simp("DIF(G'(x^2, y), x)", state=state) == (
        "2*x*SUBS(DIF(G(x_, y), x_, 2), [x_], [x^2])"
    )


def test_a_substitution_binding_no_derivative_is_carried_out():
    """`SUBS` is carried out unless it holds a derivative.

    Only the slope of an unknown function at a point must stay opaque, since
    evaluating it would wrongly give the zero slope of a constant. Any other `SUBS` is
    just a pending substitution and gets performed.
    """
    state = declaring("BB")
    assert simp("SUBS(x^2, [x], [3])", state=state) == "9"
    assert simp("SUBS(x^2 + y, [x, y], [3, 4])", state=state) == "13"
    assert simp("SUBS(BB(u), [u], [r^2])", state=state) == "BB(r^2)"
    assert simp("DIF(SUBS(BB(u), [u], [r^2]), r)", state=state) == "2*r*BB'(r^2)"
    # One that holds a derivative keeps it at its point, even when differentiated
    # again.
    assert simp("SUBS(DIF(BB(u), u), [u], [r^2])", state=state) == "BB'(r^2)"
    assert simp("DIF(SUBS(DIF(BB(u), u), [u], [r^2]), r)", state=state) == (
        "2*r*BB''(r^2)"
    )


def test_a_function_of_several_arguments_keeps_the_derivative_it_is_written_as():
    """A prime gives the order but not the argument.

    Like Derive, we print `DIF(G(x, y), x)`, since `G'(x, y)` wouldn't say which
    argument it differentiates by. `G'(x, y)` is still accepted as input, meaning the
    slope in the first argument, which becomes a substitution when that argument isn't
    a plain variable.
    """
    state = declaring("G")
    assert simp("DIF(G(x, y), x)", state=state) == "DIF(G(x, y), x)"
    assert simp("G'(x, y)", state=state) == "DIF(G(x, y), x)"
    assert simp("G'(x, x)", state=state) == "SUBS(DIF(G(x_, x), x_), [x_], [x])"


def test_a_prime_differentiates_a_function_that_has_a_body():
    """`F(r) := r^2` makes `F'` the doubling function, at any point.

    The function is differentiated first and then evaluated at the point, so
    `F'(r^2)` is `2*r^2`. Substituting first would wrongly give `2*r`.
    """
    context = Context(functions={"F": (("r",), parse("r^2"))})
    assert simp("F'(r)", context) == "2*r"
    assert simp("F'(r^2)", context) == "2*r^2"
    assert simp("F''(r)", context) == "2"
    # So is a built-in function.
    assert simp("SIN'(x)") == "COS(x)"


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "SUM(SIN(k)*COS(2*k) - SIN(3*k)*COS(5*k), k, 1, n)",
            "SUM(SIN(k)*COS(2*k) - SIN(3*k)*COS(5*k), k, 1, n)",
        ),
        (
            "SUM(SIN(a*k)*COS(k*x) - SIN(b*k)*COS(k*x), k, 1, n)",
            "SUM((SIN(a*k) - SIN(b*k))*COS(k*x), k, 1, n)",
        ),
        (
            "SUM(SIN(k)*COS(2*k) - SIN(3*k)*COS(5*k) + k, k, 1, n)",
            "SUM(SIN(k)*COS(2*k) - SIN(3*k)*COS(5*k) + k, k, 1, n)",
        ),
    ],
)
def test_trigonometric_terms_are_not_searched_for_a_telescope(text, expected):
    """Sympy checks whether a two-term summand telescopes by solving for the shift,
    which is slow or never returns for trigonometric functions.

    All three cases ran for an hour; they must return the sum unevaluated at once. A
    sum of single waves, such as the Fourier series in Derive's `INTEGRAL.MTH` after
    its coefficients are integrated, has a closed form and never reaches the search
    (see `test_a_sum_of_waves_has_a_closed_form`).
    """
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("SUM(SIN(k + 1)*COS(k + 1) - SIN(k)*COS(k), k, 1, n)", "SIN(n)*COS(n + 2)"),
        ("SUM(SIN(k + 1) - SIN(k) + k, k, 1, n)", "SIN(n + 1) + n^2/2 + n/2 - SIN(1)"),
    ],
)
def test_a_shifted_pair_of_trigonometric_terms_still_telescopes(text, expected):
    """A pair recognized by pattern matching still telescopes, without the solve."""
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("SUM(SIN(k), k, 1, n)", "SIN(n/2)*SIN(n/2 + 1/2)/SIN(1/2)"),
        ("SUM(COS(k), k, 0, n)", "SIN(n/2 + 1/2)*COS(n/2)/SIN(1/2)"),
        # If the step is a multiple of `2*pi`, every term is the same and the formula's
        # denominator is zero.
        (
            "SUM(SIN(k*x), k, 1, n)",
            "IF(SIN(x/2) = 0, 0, SIN(n*x/2)*SIN(x*(n + 1)/2)/SIN(x/2))",
        ),
        (
            "SUM(COS(k*x), k, 0, n)",
            "IF(SIN(x/2) = 0, n + 1, SIN(x*(n + 1)/2)*COS(n*x/2)/SIN(x/2))",
        ),
        (
            "SUM(COS(k*x + 1), k, 2, n + 3)",
            "IF(SIN(x/2) = 0, COS(1)*(n + 2), "
            "SIN(x*(n/2 + 1))*COS(x*(n + 5)/2 + 1)/SIN(x/2))",
        ),
        # A step known to be such a multiple is decided.
        ("SUM(COS(2*pi*k), k, 1, n)", "n"),
        ("SUM(SIN(k*pi/2), k, 1, n)", "SQRT(2)*SIN(pi*n/4)*SIN(pi*(n/4 + 1/4))"),
        # Each wave is summed separately, and non-wave terms by sympy.
        (
            "SUM(SIN(2*k + 1) + 3*COS(k), k, 1, n)",
            "3*SIN(n/2)*COS(n/2 + 1/2)/SIN(1/2) + SIN(n)*SIN(n + 2)/SIN(1)",
        ),
        ("SUM(SIN(k) + k, k, 1, n)", "SIN(n/2)*SIN(n/2 + 1/2)/SIN(1/2) + n^2/2 + n/2"),
        # The shape of a Fourier series after integrating its coefficients; the
        # telescope search took ninety seconds. Waves with the same step are gathered.
        (
            "SUM(a*SIN(k*x) - a*SIN(k*y) - b*SIN(k*x) + b*SIN(k*y), k, 1, n)",
            "(a - b)*IF(SIN(x/2) = 0, 0, SIN(n*x/2)*SIN(x*(n + 1)/2)/SIN(x/2)) + "
            "(b - a)*IF(SIN(y/2) = 0, 0, SIN(n*y/2)*SIN(y*(n + 1)/2)/SIN(y/2))",
        ),
        # A wave whose amplitude depends on the index isn't covered.
        ("SUM(k*SIN(k), k, 1, n)", "SUM(k*SIN(k), k, 1, n)"),
    ],
)
def test_a_sum_of_waves_has_a_closed_form(text, expected):
    """`SIN(a*k + b)` summed over `N` terms is the imaginary part of a geometric sum:
    `SIN(N*a/2)/SIN(a/2)` times the wave at the middle of the range.

    The cosine sum is often written `SIN((n + 1/2)*x)/(2*SIN(x/2)) + 1/2`, the same
    function. When the step `x` is a multiple of `2*pi`, both forms divide by zero and
    the sum is `N` times the single term, just as `SUM(x^k, k, 0, n)` is `n + 1` at
    `x = 1`.
    """
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("PRODUCT(#e^k, k, 1, n)", "#e^(n*(n + 1)/2)"),
        ("PRODUCT(#e^(k*x), k, 0, n)", "#e^(n*x*(n + 1)/2)"),
        ("PRODUCT(3^(k^2), k, 1, n)", "3^(n*(n + 1)*(2*n + 1)/6)"),
        ("PRODUCT(#e^(1/k^2), k, 1, inf)", "#e^(pi^2/6)"),
        # The exponential is continuous, so its limits follow the exponent's.
        ("PRODUCT(#e^k, k, 1, inf)", "inf"),
        ("PRODUCT(#e^(-k), k, 1, inf)", "0"),
        # The exponent sums to a harmonic number, which the notation can't express.
        ("PRODUCT(#e^(1/k), k, 1, n)", "PRODUCT(#e^(1/k), k, 1, n)"),
    ],
)
def test_a_product_of_exponentials_is_the_exponential_of_the_sum(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("SUM(LN(k), k, 1, n)", "LN(n!)"),
        ("SUM(LN(k^2), k, 1, n)", "LN(n!^2)"),
        # The product telescopes.
        ("SUM(LN(1 + 1/k), k, 1, n)", "LN(n + 1)"),
        ("SUM(LN((k + 1)/k), k, 1, n)", "LN(n + 1)"),
        ("SUM(LN(1 - 1/k^2), k, 2, inf)", "-LN(2)"),
        # Not sums of logarithms of positive numbers: `LN(0)` is a term, and `x` may be
        # negative.
        ("SUM(LN(k), k, 0, n)", "SUM(LN(k), k, 0, n)"),
        ("SUM(LN(x*k), k, 1, n)", "SUM(LN(k*x), k, 1, n)"),
    ],
)
def test_a_sum_of_logarithms_is_the_logarithm_of_the_product(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("SUM(COMB(n, k), k, 0, n)", "2^n"),
        ("SUM(COMB(n, k), k, 1, n)", "2^n - 1"),
        ("SUM(k*COMB(n, k), k, 0, n)", "n*2^(n - 1)"),
        ("SUM(k^2*COMB(n, k), k, 0, n)", "n*(n + 1)*2^(n - 2)"),
        ("SUM(COMB(2*n, k), k, 0, 2*n)", "2^(2*n)"),
        # Sympy adds a condition on `ABS(x)`, but the binomial theorem needs none.
        ("SUM(COMB(n, k)*x^k, k, 0, n)", "(x + 1)^n"),
        ("SUM(COMB(n, k)*COMB(m, k), k, 0, n)", "(m + n)!/(m!*n!)"),
        # Sympy answers 0, which is wrong at `n = 0`.
        ("SUM((-1)^k*COMB(n, k), k, 0, n)", "0^n"),
        # Sympy answers 0 here too, wrong at `n = 1`, and nothing else it has holds at
        # `n = 0`.
        ("SUM((-1)^k*k*COMB(n, k), k, 0, n)", "SUM(k*(-1)^k*COMB(n, k), k, 0, n)"),
    ],
)
def test_a_sum_up_to_its_own_parameter_holds_for_every_count(text, expected):
    """A sum to `n` whose terms involve `n` only makes sense for integer `n`, so
    sympy's convergence condition for the infinite series is irrelevant.

    Instead, the closed form is checked against the explicit sums for the first few
    values of `n`, and is valid for ranges of at least one term.
    """
    assert simp(text) == expected


INFINITE_PRODUCTS = [
    # The manual's own example, and Wallis'.
    ("PRODUCT(1 - 1/k^2, k, 2, inf)", "1/2"),
    ("PRODUCT(4*k^2/(4*k^2 - 1), k, 1, inf)", "pi/2"),
    # `SINH(pi)/pi`, written as Simplify writes any hyperbolic of a constant.
    ("PRODUCT(1 + 1/k^2, k, 1, inf)", "(#e^pi/2 - #e^(-pi)/2)/pi"),
    # A divergent product can still have a value.
    ("PRODUCT((k + 1)/k, k, 1, inf)", "inf"),
    # A variable lower end also telescopes.
    ("PRODUCT(1 - 1/k^2, k, m, inf)", "(m - 1)/m"),
    # The product stays when either step fails: the first has no closed form, and the
    # second's limit depends on the unknown sign of `x`.
    ("PRODUCT(COS(pi/2^k), k, 1, inf)", "PRODUCT(COS(pi*2^(-k)), k, 1, inf)"),
    ("PRODUCT(1 - x/k, k, 1, inf)", "PRODUCT(1 - x/k, k, 1, inf)"),
    # An oscillating product has no value (`nan` isn't one) and stays, like the
    # corresponding sum.
    ("PRODUCT(-1, k, 1, inf)", "PRODUCT(-1, k, 1, inf)"),
    ("SUM((-1)^k, k, 1, inf)", "SUM((-1)^k, k, 1, inf)"),
]


@pytest.mark.parametrize(("text", "expected"), INFINITE_PRODUCTS, ids=str)
def test_an_infinite_product_is_the_limit_of_its_partial_products(text, expected):
    """Sympy can't do infinite products, so we use the definition.

    For a hypergeometric body, sympy finds a closed form for the product up to `n`
    and then its limit, which gives the answer Derive prints.
    """
    assert simp(text) == expected


def test_a_two_sided_limit_whose_sides_disagree_is_plus_or_minus():
    assert simp("LIM(SIGN(x), x, 0)") == "±1"


def test_a_bounded_limit_that_never_settles_is_the_range_it_keeps_to():
    """The answer is `INTERVAL(a, b)` when there's no single value.

    None of these limits exists, as the functions oscillate forever, but each stays in
    a known range. Answering `?` would lose that.

    The last case puts the interval inside a product, to check it isn't only
    printable on its own. None may come back as `Kind.STRING`, which is how an answer
    the notation can't express looks.
    """
    for text, answer in (
        ("LIM(SIN(1/x), x, 0)", "INTERVAL(-1, 1)"),
        ("LIM(COS(1/x), x, 0)", "INTERVAL(-1, 1)"),
        ("LIM(SIN(x), x, inf)", "INTERVAL(-1, 1)"),
        ("y*LIM(SIN(x), x, inf)", "INTERVAL(-1, 1)*y"),
    ):
        answered = simplify(parse(text), Context())
        assert answered.text == answer
        assert answered.node.kind is not Kind.STRING


UNBOUNDED_OSCILLATION = [
    # Unbounded oscillation reaches every value.
    ("LIM(x*SIN(x), x, inf)", "INTERVAL(-inf, inf)"),
    ("LIM(x*COS(x), x, inf)", "INTERVAL(-inf, inf)"),
    ("LIM(x^2*SIN(x), x, inf)", "INTERVAL(-inf, inf)"),
    ("LIM(#e^x*SIN(x), x, inf)", "INTERVAL(-inf, inf)"),
    ("LIM(x*SIN(x), x, -inf)", "INTERVAL(-inf, inf)"),
    ("LIM(SIN(1/x)/x, x, 0)", "INTERVAL(-inf, inf)"),
    # Growth times an oscillation of one sign reaches only that half of the line, and
    # if the oscillation stays away from zero, there's no oscillation in the limit.
    ("LIM(x*(SIN(x) + 1), x, inf)", "INTERVAL(0, inf)"),
    ("LIM(-x*SIN(x)^2, x, inf)", "INTERVAL(-inf, 0)"),
    ("LIM(#e^x*(SIN(x) - 1), x, inf)", "INTERVAL(-inf, 0)"),
    ("LIM(x*(SIN(x) + 2), x, inf)", "inf"),
    # The tangent too, though it never was a product with the sign of a range.
    ("LIM(TAN(x), x, inf)", "INTERVAL(-inf, inf)"),
    # The sign and absolute value of a range are ranges.
    ("LIM(SIGN(SIN(x)), x, inf)", "INTERVAL(-1, 1)"),
    ("LIM(ABS(SIN(x)), x, inf)", "INTERVAL(0, 1)"),
    ("SIGN(INTERVAL(1, 2))", "1"),
    ("ABS(INTERVAL(-2, 1))", "INTERVAL(0, 2)"),
    ("ABS(INTERVAL(-3, -1))", "INTERVAL(1, 3)"),
]


@pytest.mark.parametrize(("text", "expected"), UNBOUNDED_OSCILLATION, ids=str)
def test_an_unbounded_limit_that_never_settles_is_the_range_it_reaches(text, expected):
    """`INTERVAL(-inf, inf)` or one of its halves, instead of sympy's unreadable
    infinity times the sign of a range.

    This matches the bounded case, and the interval can be computed with, whereas `?`
    would lose which half of the line is reached.
    """
    assert simp(text) == expected


LIMITS_OVER_A_PARAMETER_S_SIGN = [
    # The arc tangent tends to a right angle with the parameter's sign, for every
    # nonzero parameter (zero gives a different function).
    ("LIM(ATAN(x/a), x, inf)", "pi*SIGN(a)/2"),
    ("LIM(ATAN(a*x), x, inf)", "pi*SIGN(a)/2"),
    ("LIM(ATAN(a*x), x, -inf)", "-pi*SIGN(a)/2"),
    ("LIM(TANH(a*x), x, inf)", "SIGN(a)"),
    # No single expression covers both 0 and inf, so the limit stays.
    ("LIM(#e^(a*x), x, inf)", "LIM(#e^(a*x), x, inf)"),
    # A limit depending on whether the parameter is above or below one becomes a
    # conditional. Elsewhere (one, or a base with a non-real logarithm) the limit
    # stays.
    ("LIM(a^x, x, inf)", "IF(a > 1, inf, IF(0 < a < 1, 0, LIM(a^x, x, inf)))"),
    ("LIM(a^x, x, -inf)", "IF(a > 1, 0, IF(0 < a < 1, inf, LIM(a^x, x, -inf)))"),
    (
        "LIM(x*LN(a), x, inf)",
        "IF(a > 1, inf, IF(0 < a < 1, -inf, LIM(x*LN(a), x, inf)))",
    ),
    # Infinity with the parameter's sign.
    ("LIM(x^2/a, x, inf)", "inf*SIGN(a)"),
    ("LIM(x/ABS(a), x, inf)", "inf"),
]


@pytest.mark.parametrize(("text", "expected"), LIMITS_OVER_A_PARAMETER_S_SIGN, ids=str)
def test_a_limit_that_turns_on_a_parameter_s_sign_is_written_with_it(text, expected):
    """The limits for positive and negative parameters, combined, instead of a
    meaningless unsigned infinity inside an expression like `ATAN(±inf/a)`."""
    assert simp(text) == expected


LIMITS_OF_SUMS_AND_PRODUCTS = [
    # The closed forms are generalized harmonic numbers, which can't be expressed, but
    # their limits can.
    ("LIM(SUM(1/k^2, k, 1, n), n, inf)", "pi^2/6"),
    ("LIM(SUM(1/k^3, k, 1, n), n, inf)", "ZETA(3)"),
    ("LIM(SUM(1/k, k, 1, n) - LN(n), n, inf)", "euler_gamma"),
    ("LIM(SUM(1/k, k, 1, n) - LN(n + 1), n, inf)", "euler_gamma"),
    ("LIM(SUM(1/k, k, 1, n)/LN(n), n, inf)", "1"),
    ("LIM(SUM(1/k, k, 1, n), n, inf)", "inf"),
    # A side given for an infinite point changes nothing.
    ("LIM(SUM(1/k^2, k, 1, n), n, inf, -1)", "pi^2/6"),
    # The partial products are ratios of factorials of complex arguments, and their
    # limit is `SINH(pi)/pi`.
    ("LIM(PRODUCT(1 + 1/k^2, k, 1, n), n, inf)", "(#e^pi/2 - #e^(-pi)/2)/pi"),
    # `INT` and `LIM` write this constant the same way.
    ("INT(#e^(-x)*LN(x), x, 0, inf)", "-euler_gamma"),
    # An unevaluated limit is written as given; the side sympy records for an infinite
    # point doesn't make it one-sided.
    ("LIM(SUM((-1)^k/k, k, 1, n), n, inf)", "LIM(SUM((-1)^k/k, k, 1, n), n, inf)"),
    ("LIM(ZETA(s) - 1/(s - 1), s, 1)", "LIM(ZETA(s) - 1/(s - 1), s, 1)"),
]


@pytest.mark.parametrize(("text", "expected"), LIMITS_OF_SUMS_AND_PRODUCTS, ids=str)
def test_a_limit_of_a_sum_is_taken_of_its_closed_form(text, expected):
    """Only the final answer must be expressible in the notation, not the intermediate
    closed form."""
    assert simp(text) == expected


def test_a_limit_that_settles_or_runs_away_is_no_interval():
    """Only bounded oscillation gives an interval.

    An existing limit is its value. Unbounded growth is `±inf`, a single value with
    both signs rather than a range. Sides that disagree give `?`, e.g. floor at 1 is 0
    from the left and 1 from the right.
    """
    assert simp("LIM(SIN(x)/x, x, 0)") == "1"
    assert simp("LIM(1/x, x, 0)") == "±inf"
    assert simp("LIM(FLOOR(x), x, 1)") == "?"
    assert simp("LIM(SIN(x)/ABS(x), x, 0)") == "±1"
    assert simp("LIM(ZETA(s), s, 1)") == "±inf"
    assert simp("LIM(LN(x), x, 0)") == "-inf"


def test_a_limit_towards_a_moving_point_is_the_substitution_it_means():
    """A limit toward a point that depends on the variable is a substitution.

    In `LIM(u, x, a)` where `a` mentions `x`, every `x` is replaced at once, so
    `LIM((x*v - y)^2, y, x*v - y)` is `y^2`. `ODE1.MTH` relies on this (`CLAIRAUT`
    changes variables with it twice in one line). A side argument is ignored.
    """
    assert simp("LIM((x*v - y)^2, y, x*v - y)") == "y^2"
    assert simp("LIM(x^2, x, x + 1)") == "(x + 1)^2"
    assert simp("LIM(x^2, x, x + 1, 1)") == "(x + 1)^2"
    # The body is simplified before substituting, so a derivative is taken before the
    # point is inserted.
    assert simp("LIM(DIF(LIM((x*v - y)^2, y, x*v - y), y), y, x*v - y)") == (
        "2*v*x - 2*y"
    )


def test_a_calculus_head_waits_for_the_body_of_the_call_under_it():
    """A call whose definition isn't expanded yet is not a constant.

    Each pass expands one level of recursion, so `F(1)` is still unresolved while
    `F(2)` is being unfolded. It mentions no `x`, but differentiating it as a constant
    would wrongly give zero for what is a derivative of `x^2`.
    """
    body = parse("IF(n = 0, x^2, DIF(F(n - 1), x))", declaring("F"))
    context = Context(functions={"F": (("n",), body)})
    assert simp("F(0)", context) == "x^2"
    assert simp("F(1)", context) == "2*x"
    assert simp("F(2)", context) == "2"
