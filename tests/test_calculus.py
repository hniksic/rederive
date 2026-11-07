"""What Simplify makes of the calculus heads other than an integral.

Derivatives, Taylor polynomials, sums and products, and limits, each case the author's
text in and the worksheet's text out, written inline as in `test_simplify.py` so that a
failure names the mathematics. The cases are the ones that show a rule: a derivative of
a function without a body is written with primes rather than refused, a geometric series
answers `IF(ABS(x) < 1, 1/(1 - x), SUM(x^k, k, 0, inf))` because it converges only
inside the unit circle, a summation formula keeps the shape it comes in, and
`LIM(SIGN(x), x, 0)` is `±1` because its two sides disagree. A limit that never settles
is the range it keeps to, `LIM(SIN(x), x, inf)` being `INTERVAL(-1, 1)`. Integrals have
`test_integrate.py`.
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
    # Sympy proves where an improper integral converges and leaves the rest
    # standing; Derive works the antiderivative to its limit and prints the
    # infinity it diverges to.
    ("INT(x^n, x, 1, inf)", "IF(n < -1, -1/(n + 1), inf)"),
    ("INT(1/x^n, x, 1, inf)", "IF(n > 1, 1/(n - 1), inf)"),
    # The halves either side of 1 cancel, which `x -> 1/x` shows.
    ("INT(LN(x)/(1 + x^2), x, 0, inf)", "0"),
    # The imaginary error function writes this without an imaginary unit.
    ("INT(#e^(x^2), x, 0, 1)", "SQRT(pi)*ERFI(1)/2"),
    # A real integrand over a real interval: sympy's values, `-pi^2/12 +
    # 2*#i*pi*LN(2)` and `LN(t + 1) + #i*pi`, carry an imaginary part that is not
    # there, and with no other value proved the integrals stand.
    ("INT(LN(x)/(x + 1), x, 0, 1)", "INT(LN(x)/(x + 1), x, 0, 1)"),
    ("INT(LN(x)/(x + 1), x, 1, 2)", "INT(LN(x)/(x + 1), x, 1, 2)"),
    ("INT((x^t - 1)/LN(x), x, 0, 1)", "INT((x^t - 1)/LN(x), x, 0, 1)"),
    # Sympy's value is `-EI(EXP_POLAR(#i*pi))`, which is `-EI(-1) - pi*#i`. Simplify's
    # own antiderivative, `EI(-x)`, is continuous on the interval, and its difference
    # at the ends is the integral.
    ("INT(#e^(-x)/x, x, 1, inf)", "-EI(-1)"),
    # Not real left of zero, so the imaginary part is the integral's own.
    ("INT(LN(x), x, -1, 1)", "-2 + #i*pi"),
    # Sympy has no antiderivative for this, and Simplify's own is continuous on the
    # interval for every `q`: the argument of its logarithm times its conjugate is -1,
    # so it is never zero. The integral is the difference at the ends.
    (
        "INT(SIN(x)/SQRT(1 + q^2*COS(x)^2), x, 0, pi)",
        "(LN(SQRT(q^2 + 1) + q) - LN(SQRT(q^2 + 1) - q))/q",
    ),
    # For `q` past 1 the root in the antiderivative is imaginary on part of the
    # interval, and its arctangent is not proved continuous: the integral stands.
    (
        "INT(COS(x)/SQRT(1 - q^2*SIN(x)^2), x, -1, 2)",
        "INT(COS(x)/SQRT(1 - q^2*SIN(x)^2), x, -1, 2)",
    ),
    # An integrand that turns its sign when reflected about the middle of the interval,
    # which its sum with the reflection having zero derivative, being continuous and
    # zero at the middle proves.
    ("INT(LN(SIN(x)) - LN(COS(x)), x, 0, pi/2)", "0"),
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
    # Toward `-inf` an exponential of a base above one dies away and one of a base
    # below one grows.
    ("LIM(2^x, x, -inf)", "0"),
    ("LIM((1/2)^x, x, -inf)", "inf"),
    ("LIM(x^3, x, -inf)", "-inf"),
    # A vector of variables against a vector of points is the limits taken one
    # after another, which is the substitution `x = 0` alone cannot make.
    ("LIM(x^2 + y^2, [x, y], [2, 3])", "13"),
    ("LIM(SIN(x)/x + y, [x, y], [0, 1])", "2"),
    # A matrix is approached element by element, which is how SOLVE.MTH writes
    # an iterate into a whole augmented matrix at once.
    ("LIM([[SIN(x)/x, x + y], [x^2, 3]], x, 0)", "[[1, y], [0, 3]]"),
    ("LIM([[x*y, x + y], [x^2, y]], [x, y], [1, 2])", "[[2, 3], [1, 2]]"),
    # The order is the maximum degree, and a derivative that simplifies to zero
    # at the expansion point costs a term: this one has no even powers.
    ("TAYLOR(c*SIN(x), x, 0, 6)", "c*x^5/120 - c*x^3/6 + c*x"),
    ("TAYLOR(1/(1 - x), x, 0, 3)", "x^3 + x^2 + x + 1"),
]


@pytest.mark.parametrize(("text", "expected"), CALCULUS, ids=str)
def test_a_calculus_head_is_evaluated(text, expected):
    assert simp(text) == expected


def test_a_function_without_a_body_is_differentiated_symbolically():
    # `F(x) :=` gives `F` no body, so it differentiates symbolically and the chain
    # rule stops at the prime. An arbitrary function has an arbitrary gradient, and
    # it is written the way any other underived derivative is.
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
    # Longer factored than expanded, and factored all the same: the shape is
    # the formula's and not an economy. Derive prints this one as
    # `n*(n + 1)*(6*n^3 + 9*n^2 + n - 1)/30`, which is the same product one
    # factor short.
    ("SUM(k^4, k, 1, n)", "n*(n + 1)*(2*n + 1)*(3*n^2 + 3*n - 1)/30"),
    # A vector is summed element by element, and each element is a formula.
    ("SUM([k, k^2], k, 0, n)", "[n*(n + 1)/2, n*(n + 1)*(2*n + 1)/6]"),
]


#: Sums to infinity, answered only where they converge.
INFINITE_SUMS = [
    # A geometric series converges only inside the unit circle, and the answer says
    # so, for the series and for its derivative.
    ("SUM(x^k, k, 0, inf)", "IF(ABS(x) < 1, 1/(1 - x), SUM(x^k, k, 0, inf))"),
    (
        "SUM(k*x^(k-1), k, 1, inf)",
        "IF(ABS(x) < 1, 1/(1 - x)^2, SUM(k*x^(k - 1), k, 1, inf))",
    ),
    # An oscillating sum has no value and stands.
    ("SUM((-1)^k, k, 0, inf)", "SUM((-1)^k, k, 0, inf)"),
    # Leibniz's series.
    ("SUM((-1)^(k+1)/(2*k-1), k, 1, inf)", "pi/4"),
    # `k^2 = k*(k - 1) + k`, and each part sums to `#e`.
    ("SUM(k^2/k!, k, 0, inf)", "2*#e"),
    # Zeta where the series converges, and the sum where it may not.
    ("SUM(1/k^p, k, 1, inf)", "IF(p > 1, ZETA(p), SUM(k^(-p), k, 1, inf))"),
]


@pytest.mark.parametrize(("text", "expected"), INFINITE_SUMS, ids=str)
def test_an_infinite_sum_is_answered_where_it_converges(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize(("text", "expected"), SUMMATION_FORMULAS, ids=str)
def test_a_summation_formula_keeps_the_shape_it_comes_in(text, expected):
    """Derive's closed forms for a sum are factored, and Simplify leaves them so.

    Sympy hands back Faulhaber's polynomial multiplied out, and writing it as
    the original writes it is what this is. Not a factoring rule: the same
    polynomial authored by hand is left exactly as it was authored, which is
    what `test_a_polynomial_is_not_factored_for_having_a_common_factor` holds.
    """
    assert simp(text) == expected


def test_a_range_that_stops_between_two_terms_is_written_out():
    """The index runs up while it does not pass the end, which is Derive's rule.

    Sympy reads an end that is no whole number of steps away as the `n` of a
    closed form and puts it into Faulhaber's polynomial, which answers a
    polynomial in the bound where a row of terms was meant. NUMBER.MTH's
    `PARTS` bounds Rademacher's series that way - `SQRT(n)/LOG(n, 11)`, which
    is 3.459 where `n` is 4 - and the terms are what its answer is made of.
    """
    assert simp("SUM(k, k, 1, 3.5)") == "6"
    assert simp("SUM(k^2, k, 1, LN(11)/LN(2))") == "14"
    assert simp("PRODUCT(k, k, 1, 3.5)") == "6"
    # An end below the start is a range of no terms rather than no range.
    assert simp("SUM(k, k, 1, 1/2)") == "0"
    # One further below steps back from the start, and the sum is the one stepped
    # over, negated: checked against the original.
    assert simp("SUM(k, k, 5, 3.5)") == "-4"
    assert simp("SUM(k, k, 5.5, 3)") == "-8"
    assert simp("PRODUCT(k, k, 5, 3.5)") == "1/4"
    # An endless range ends nowhere, and what answers it is the closed form.
    assert simp("SUM(2^-k, k, 0, inf)") == "2"


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


def test_a_taylor_polynomial_that_does_not_exist_is_undefined():
    """No derivatives at the point, no polynomial.

    The derivatives of `SQRT(x)` at zero are not finite, so no term past the
    zeroth can be summed and the answer is `?`, which is what Derive answers
    such a case with. What sympy has for it - the series `SQRT(x)`, which is no
    polynomial - would be an answer to a different question.
    """
    assert simp("TAYLOR(SQRT(x), x, 0, 1)") == "?"
    assert simp("TAYLOR(LN(x), x, 0, 2)") == "?"
    # The order the derivatives do reach is the polynomial of that order, and
    # `SQRT(0)` is a value like any other: the manual's `TAYLOR(SQRT(x), x, 0,
    # 0)` is 0.
    assert simp("TAYLOR(SQRT(x), x, 0, 0)") == "0"
    # An order that is no order to expand to leaves the head alone.
    assert simp("TAYLOR(SIN(x), x, 0, n)") == "TAYLOR(SIN(x), x, 0, n)"
    # The same holds for the other singularities at the point.
    assert simp("TAYLOR(SQRT(x), x, 0, 3)") == "?"
    assert simp("TAYLOR(LN(x), x, 0, 3)") == "?"
    assert simp("TAYLOR(1/x, x, 0, 3)") == "?"


TAYLOR_AT_A_KINK_OR_A_JUMP = [
    # The expansion from the right is not the one from the left, and neither is the
    # Taylor polynomial, which needs the derivatives at the point itself.
    ("TAYLOR(ABS(x), x, 0, 2)", "TAYLOR(ABS(x), x, 0, 2)"),
    ("TAYLOR(SQRT(x^2), x, 0, 2)", "TAYLOR(ABS(x), x, 0, 2)"),
    ("TAYLOR(SIGN(x), x, 0, 2)", "TAYLOR(SIGN(x), x, 0, 2)"),
    ("TAYLOR(x*ABS(x), x, 0, 3)", "TAYLOR(x*ABS(x), x, 0, 3)"),
    ("TAYLOR(ABS(SIN(x)), x, 0, 3)", "TAYLOR(ABS(SIN(x)), x, 0, 3)"),
    ("TAYLOR(FLOOR(x), x, 0, 2)", "TAYLOR(FLOOR(x), x, 0, 2)"),
    ("TAYLOR(STEP(x), x, 0, 2)", "TAYLOR(SIGN(x)/2 + 1/2, x, 0, 2)"),
    ("TAYLOR(MAX(x, 0), x, 0, 2)", "TAYLOR(ABS(x)/2 + x/2, x, 0, 2)"),
    # Differentiable as often as the order asks, the function has the polynomial:
    # `x*ABS(x)` once and `ABS(x)^3` twice, both 0.
    ("TAYLOR(x*ABS(x), x, 0, 1)", "0"),
    ("TAYLOR(ABS(x)^3, x, 0, 2)", "0"),
    # Away from the kink or the jump, the function is analytic.
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
    """A function that is not differentiable at the point has no Taylor polynomial
    there, and the call stands rather than answer with the expansion from one side."""
    assert simp(text) == expected


DERIVATIVES_OF_STEPS = [
    # A step function is constant between its jumps and has no derivative at them, so
    # its derivative is 0 wherever there is one.
    ("DIF(FLOOR(x), x)", "0"),
    ("DIF(SIGN(x), x)", "0"),
    ("DIF(STEP(x), x)", "0"),
    ("DIF(FLOOR(x), x, 3)", "0"),
    ("DIF(SIGN(SIN(x)), x)", "0"),
    ("DIF(FLOOR(x, 2), x)", "0"),
    ("DIF(FLOOR(x)^2, x)", "0"),
    ("DIF(MOD(x, 1), x)", "1"),
    ("DIF(MOD(x^2, 3), x)", "2*x"),
    # A step times a smooth function is differentiated as a constant times it.
    ("DIF(SIGN(x)*x, x)", "SIGN(x)"),
    ("DIF(STEP(x)*x^2, x)", "x*(SIGN(x) + 1)"),
    ("DIF(FLOOR(x)*x, x)", "FLOOR(x)"),
    ("DIF(FLOOR(x^2)*SIN(x), x)", "FLOOR(x^2)*COS(x)"),
    ("DIF(SIGN(x^2 - 1)*x^3, x)", "3*x^2*SIGN(x^2 - 1)"),
    # The derivative of an absolute value is a sign, and the one after it is the
    # sign's.
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
    """What sympy writes at a jump is a `DiracDelta`, which the notation has no name
    for, and of a floor it writes nothing; the derivative wherever it exists is 0,
    and that is the answer."""
    assert simp(text) == expected


def test_a_derivative_a_substitution_binds_is_not_evaluated():
    """What sympy holds under a `SUBS` is held on purpose.

    Differentiating an unknown function of a moving argument gives the slope of
    `F` at that argument, written as a derivative over a bound variable and a
    substitution into it. That derivative looks like the derivative of a
    constant and is not one, so nothing inside the substitution is evaluated -
    and the answer is the same on the second pass as on the first.

    The argument here is written as the substitution `LIM(F(y), y, 2*x - y)`
    is, so the chain rule contributes the minus in front. A function of one
    argument is a prime and a function of two is the substitution spelled out;
    both hold the same thing and both come back as themselves.
    """
    state = declaring("F", "G")
    once = simplify(parse("DIF(LIM(F(y), y, 2*x - y), y)", state), Context(), state)
    assert once.text == "-F'(2*x - y)"
    assert simplify(once.node, Context(), state).text == once.text
    two = simplify(parse("DIF(LIM(G(y, z), y, 2*x - y), y)", state), Context(), state)
    assert two.text == "-SUBS(DIF(G(x_, z), x_), [x_], [2*x - y])"
    assert simplify(two.node, Context(), state).text == two.text


def test_the_derivative_of_an_undefined_function_is_written_with_primes():
    """`BB'(r)`, which is what the original prints and what it reads back.

    `DIF(BB(r), r)` says the same thing at twice the length, and that is the
    whole of the case: a four by four matrix of Ricci components is readable in
    primes and is not readable in `DIF`. One mark per order, and the mark is on
    the function rather than on the call, so the point it is taken at is
    whatever the chain rule left there.
    """
    state = declaring("BB")
    assert simp("DIF(BB(r), r)", state=state) == "BB'(r)"
    assert simp("DIF(BB(r), r, 3)", state=state) == "BB'''(r)"
    assert simp("DIF(BB(r)^2, r)", state=state) == "2*BB'(r)*BB(r)"
    assert simp("DIF(BB(r^2), r)", state=state) == "2*r*BB'(r^2)"
    # And the notation reads back what it wrote, in either direction.
    assert simp("BB''(r)", state=state) == "BB''(r)"
    assert simp("DIF(BB'(r), r)", state=state) == "BB''(r)"
    assert simp("INT(BB'(r), r)", state=state) == "BB(r)"


def test_a_prime_at_a_moving_point_is_differentiated_again():
    """`DIF(BB'(r^2), r)` is `2*r*BB''(r^2)`: one more mark, the same point.

    A prime taken somewhere other than at a variable of its own is a derivative
    held at a point, and the derivative it holds is unevaluated on purpose -
    nothing may look inside. Differentiating it needs nothing from inside
    either: the chain rule raises the order by one, leaves the point where it
    was, and puts the slope of the point in front. Which is what the original
    answers, at every order and through whatever the point is written as; how the
    terms of a higher order are then gathered is the engine's spelling.
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
    # The neighbours, which say the same thing about a call rather than about a
    # prime, and a prime at a point that moves nowhere.
    assert simp("DIF(BB(r^2), r)", state=state) == "2*r*BB'(r^2)"
    assert simp("DIF(BB(r^2), r, 2)", state=state) == "4*r^2*BB''(r^2) + 2*BB'(r^2)"
    assert simp("DIF(BB'(r), r)", state=state) == "BB''(r)"
    assert simp("DIF(BB'(r)^2, r)", state=state) == "2*BB'(r)*BB''(r)"
    assert simp("DIF(BB'(r^2), r, 0)", state=state) == "BB'(r^2)"
    # A slope taken at a number is a number, and a number does not depend on r.
    assert simp("DIF(BB'(2), r)", state=state) == "0"
    # A function of two arguments has no prime to print, so the answer is the
    # `DIF` or the substitution the original writes the same thing as.
    assert simp("DIF(G'(x, y), x)", state=state) == "DIF(G(x, y), x, 2)"
    assert simp("DIF(G'(x^2, y), x)", state=state) == (
        "2*x*SUBS(DIF(G(x_, y), x_, 2), [x_], [x^2])"
    )


def test_a_substitution_binding_no_derivative_is_carried_out():
    """`SUBS` is a value where it holds a derivative and a substitution where not.

    What may not be looked inside is the slope of an unknown function at a
    point, which is unevaluated because evaluating it would answer the zero a
    constant's slope is. A `SUBS` over anything else is holding no such thing:
    it is a substitution nobody has made yet, and making it is the whole of
    what it says.
    """
    state = declaring("BB")
    assert simp("SUBS(x^2, [x], [3])", state=state) == "9"
    assert simp("SUBS(x^2 + y, [x, y], [3, 4])", state=state) == "13"
    assert simp("SUBS(BB(u), [u], [r^2])", state=state) == "BB(r^2)"
    assert simp("DIF(SUBS(BB(u), [u], [r^2]), r)", state=state) == "2*r*BB'(r^2)"
    # And the one that does hold a derivative keeps it, at the point it was
    # taken at, whether or not something outside differentiates it again.
    assert simp("SUBS(DIF(BB(u), u), [u], [r^2])", state=state) == "BB'(r^2)"
    assert simp("DIF(SUBS(DIF(BB(u), u), [u], [r^2]), r)", state=state) == (
        "2*r*BB''(r^2)"
    )


def test_a_function_of_several_arguments_keeps_the_derivative_it_is_written_as():
    """A prime says which order and not which argument.

    The original prints `DIF(G(x, y), x)` for the slope in the first of two,
    and so does this: `G'(x, y)` would be an answer that did not say what it
    was the derivative with respect to. It is read on the way in, though, since
    the original reads it - as the slope in the first argument, which is a
    substitution wherever the first argument is not a variable of its own.
    """
    state = declaring("G")
    assert simp("DIF(G(x, y), x)", state=state) == "DIF(G(x, y), x)"
    assert simp("G'(x, y)", state=state) == "DIF(G(x, y), x)"
    assert simp("G'(x, x)", state=state) == "SUBS(DIF(G(x_, x), x_), [x_], [x])"


def test_a_prime_differentiates_a_function_that_has_a_body():
    """`F(r) := r^2` makes `F'` the doubling function, at whatever point.

    Differentiated and then taken at the point, never the other way round:
    `F'(r^2)` is twice `r^2` and not twice `r`, and substituting first would
    give the second.
    """
    context = Context(functions={"F": (("r",), parse("r^2"))})
    assert simp("F'(r)", context) == "2*r"
    assert simp("F'(r^2)", context) == "2*r^2"
    assert simp("F''(r)", context) == "2"
    # A head the tables know is differentiated the same way.
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
    """Sympy asks of a two-term summand whether it telescopes, by solving for the
    shift, and over trigonometric functions that solve is slow or never returns.

    All three ran for an hour. The sum written out is the answer either way, and it
    has to be the answer at once. A sum of single waves, the shape the Fourier
    series of the original's `INTEGRAL.MTH` comes to once its coefficients are
    integrated, has a closed form and does not reach the search at all: see
    `test_a_sum_of_waves_has_a_closed_form`.
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
    """A pair the match settles never reaches the solve, and is not refused."""
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("SUM(SIN(k), k, 1, n)", "SIN(n/2)*SIN(n/2 + 1/2)/SIN(1/2)"),
        ("SUM(COS(k), k, 0, n)", "SIN(n/2 + 1/2)*COS(n/2)/SIN(1/2)"),
        # A step that may be a multiple of `2*pi` makes every term the same, and the
        # formula's denominator zero.
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
        # A step that is such a multiple is decided.
        ("SUM(COS(2*pi*k), k, 1, n)", "n"),
        ("SUM(SIN(k*pi/2), k, 1, n)", "SQRT(2)*SIN(pi*n/4)*SIN(pi*(n/4 + 1/4))"),
        # Each wave is summed for itself, and what is no wave by sympy.
        (
            "SUM(SIN(2*k + 1) + 3*COS(k), k, 1, n)",
            "3*SIN(n/2)*COS(n/2 + 1/2)/SIN(1/2) + SIN(n)*SIN(n + 2)/SIN(1)",
        ),
        ("SUM(SIN(k) + k, k, 1, n)", "SIN(n/2)*SIN(n/2 + 1/2)/SIN(1/2) + n^2/2 + n/2"),
        # The shape of a Fourier series once its coefficients are integrated, whose
        # search for a telescope took ninety seconds: waves of the same step are
        # gathered.
        (
            "SUM(a*SIN(k*x) - a*SIN(k*y) - b*SIN(k*x) + b*SIN(k*y), k, 1, n)",
            "(a - b)*IF(SIN(x/2) = 0, 0, SIN(n*x/2)*SIN(x*(n + 1)/2)/SIN(x/2)) + "
            "(b - a)*IF(SIN(y/2) = 0, 0, SIN(n*y/2)*SIN(y*(n + 1)/2)/SIN(y/2))",
        ),
        # A wave whose amplitude depends on the index is no such sum.
        ("SUM(k*SIN(k), k, 1, n)", "SUM(k*SIN(k), k, 1, n)"),
    ],
)
def test_a_sum_of_waves_has_a_closed_form(text, expected):
    """`SIN(a*k + b)` summed over a range is the imaginary part of a geometric sum,
    `SIN(N*a/2)/SIN(a/2)` times the wave at the middle of the range, for `N` terms.

    The cosine sum is often written `SIN((n + 1/2)*x)/(2*SIN(x/2)) + 1/2`, the same
    function. Where the step `x` is a multiple of `2*pi`, at which either form divides
    by zero, the sum is `N` times the one term, as the geometric sum
    `SUM(x^k, k, 0, n)` is `n + 1` at `x = 1`.
    """
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("PRODUCT(#e^k, k, 1, n)", "#e^(n*(n + 1)/2)"),
        ("PRODUCT(#e^(k*x), k, 0, n)", "#e^(n*x*(n + 1)/2)"),
        ("PRODUCT(3^(k^2), k, 1, n)", "3^(n*(n + 1)*(2*n + 1)/6)"),
        ("PRODUCT(#e^(1/k^2), k, 1, inf)", "#e^(pi^2/6)"),
        # The exponential is continuous, and its limits are those of the exponent.
        ("PRODUCT(#e^k, k, 1, inf)", "inf"),
        ("PRODUCT(#e^(-k), k, 1, inf)", "0"),
        # The exponent sums to a harmonic number, which has no name here.
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
        # `LN(0)` is a term, and `x` may be negative: neither is a sum of logarithms
        # of positive numbers.
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
        # Sympy's condition is on `ABS(x)`, where the binomial theorem has none.
        ("SUM(COMB(n, k)*x^k, k, 0, n)", "(x + 1)^n"),
        ("SUM(COMB(n, k)*COMB(m, k), k, 0, n)", "(m + n)!/(m!*n!)"),
        # Sympy answers 0, which the sum is not at `n = 0`.
        ("SUM((-1)^k*COMB(n, k), k, 0, n)", "0^n"),
        # Sympy answers 0 here too, which the sum is not at `n = 1`, and nothing else
        # it has holds at `n = 0`.
        ("SUM((-1)^k*k*COMB(n, k), k, 0, n)", "SUM(k*(-1)^k*COMB(n, k), k, 0, n)"),
    ],
)
def test_a_sum_up_to_its_own_parameter_holds_for_every_count(text, expected):
    """A sum to `n` whose terms involve `n` has terms only for an integer `n`, and
    the condition sympy attaches, which is where the series it sums over every `k`
    converges, says nothing about it.

    The closed form is checked against the terms added up for the first several
    counts instead, and answers for a range of at least one term.
    """
    assert simp(text) == expected


INFINITE_PRODUCTS = [
    # The manual's own example, and Wallis'.
    ("PRODUCT(1 - 1/k^2, k, 2, inf)", "1/2"),
    ("PRODUCT(4*k^2/(4*k^2 - 1), k, 1, inf)", "pi/2"),
    # This is `SINH(pi)/pi`, written the way Simplify writes any hyperbolic of a
    # constant.
    ("PRODUCT(1 + 1/k^2, k, 1, inf)", "(#e^pi/2 - #e^(-pi)/2)/pi"),
    # A product need not converge to a number to have a value.
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
        ("y*LIM(SIN(x), x, inf)", "INTERVAL(-1, 1)*y"),
    ):
        answered = simplify(parse(text), Context())
        assert answered.text == answer
        assert answered.node.kind is not Kind.STRING


UNBOUNDED_OSCILLATION = [
    # Oscillation that grows without bound keeps coming back to every value, and
    # its range is the whole line.
    ("LIM(x*SIN(x), x, inf)", "INTERVAL(-inf, inf)"),
    ("LIM(x*COS(x), x, inf)", "INTERVAL(-inf, inf)"),
    ("LIM(x^2*SIN(x), x, inf)", "INTERVAL(-inf, inf)"),
    ("LIM(#e^x*SIN(x), x, inf)", "INTERVAL(-inf, inf)"),
    ("LIM(x*SIN(x), x, -inf)", "INTERVAL(-inf, inf)"),
    ("LIM(SIN(1/x)/x, x, 0)", "INTERVAL(-inf, inf)"),
    # Growth over an oscillation that keeps to one sign of zero reaches only that
    # half of the line, and over one that never comes near zero it is no
    # oscillation at all.
    ("LIM(x*(SIN(x) + 1), x, inf)", "INTERVAL(0, inf)"),
    ("LIM(-x*SIN(x)^2, x, inf)", "INTERVAL(-inf, 0)"),
    ("LIM(#e^x*(SIN(x) - 1), x, inf)", "INTERVAL(-inf, 0)"),
    ("LIM(x*(SIN(x) + 2), x, inf)", "inf"),
    # The same as the tangent's, which never was a product with the sign of a range.
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
    """`INTERVAL(-inf, inf)` and its halves, rather than infinity times the sign of a
    range, which is what sympy leaves of these and is no value anyone can read.

    The range is what the bounded case answers with too, and it is a value the rest
    of the notation computes with, where `?` would throw away which half of the line
    is reached.
    """
    assert simp(text) == expected


LIMITS_OVER_A_PARAMETER_S_SIGN = [
    # The arc tangent tends to a right angle of the parameter's sign, for every
    # parameter but zero, where the function is another one altogether.
    ("LIM(ATAN(x/a), x, inf)", "pi*SIGN(a)/2"),
    ("LIM(ATAN(a*x), x, inf)", "pi*SIGN(a)/2"),
    ("LIM(ATAN(a*x), x, -inf)", "-pi*SIGN(a)/2"),
    ("LIM(TANH(a*x), x, inf)", "SIGN(a)"),
    # No one expression is both 0 and inf, and the limit stands.
    ("LIM(#e^(a*x), x, inf)", "LIM(#e^(a*x), x, inf)"),
    # A limit that turns on which side of one the parameter is on is a conditional
    # on it, and where it is on neither side - one, or a base whose logarithm is
    # not real - the limit stands.
    ("LIM(a^x, x, inf)", "IF(a > 1, inf, IF(0 < a < 1, 0, LIM(a^x, x, inf)))"),
    ("LIM(a^x, x, -inf)", "IF(a > 1, 0, IF(0 < a < 1, inf, LIM(a^x, x, -inf)))"),
    (
        "LIM(x*LN(a), x, inf)",
        "IF(a > 1, inf, IF(0 < a < 1, -inf, LIM(x*LN(a), x, inf)))",
    ),
    # Infinity times a sign that is the sign of the parameter is written so.
    ("LIM(x^2/a, x, inf)", "inf*SIGN(a)"),
    ("LIM(x/ABS(a), x, inf)", "inf"),
]


@pytest.mark.parametrize(("text", "expected"), LIMITS_OVER_A_PARAMETER_S_SIGN, ids=str)
def test_a_limit_that_turns_on_a_parameter_s_sign_is_written_with_it(text, expected):
    """Not `ATAN(±inf/a)`, an unsigned infinity inside an expression, which is no
    value: the limit for a positive and for a negative parameter, put together."""
    assert simp(text) == expected


LIMITS_OF_SUMS_AND_PRODUCTS = [
    # The closed forms of these sums are generalized harmonic numbers, which are no
    # answer, but their limits are.
    ("LIM(SUM(1/k^2, k, 1, n), n, inf)", "pi^2/6"),
    ("LIM(SUM(1/k^3, k, 1, n), n, inf)", "ZETA(3)"),
    ("LIM(SUM(1/k, k, 1, n) - LN(n), n, inf)", "euler_gamma"),
    ("LIM(SUM(1/k, k, 1, n) - LN(n + 1), n, inf)", "euler_gamma"),
    ("LIM(SUM(1/k, k, 1, n)/LN(n), n, inf)", "1"),
    ("LIM(SUM(1/k, k, 1, n), n, inf)", "inf"),
    # A side written on an infinite point changes nothing.
    ("LIM(SUM(1/k^2, k, 1, n), n, inf, -1)", "pi^2/6"),
    # The partial products are ratios of factorials of complex arguments over a
    # constant, `SINH(pi)/pi`, and the constant is the limit.
    ("LIM(PRODUCT(1 + 1/k^2, k, 1, n), n, inf)", "(#e^pi/2 - #e^(-pi)/2)/pi"),
    # The constant `INT` answers in is the one a limit answers in.
    ("INT(#e^(-x)*LN(x), x, 0, inf)", "-euler_gamma"),
    # A limit that stays is written back as it was written: the side sympy records
    # on an infinite point is not a one-sided limit.
    ("LIM(SUM((-1)^k/k, k, 1, n), n, inf)", "LIM(SUM((-1)^k/k, k, 1, n), n, inf)"),
    ("LIM(ZETA(s) - 1/(s - 1), s, 1)", "LIM(ZETA(s) - 1/(s - 1), s, 1)"),
]


@pytest.mark.parametrize(("text", "expected"), LIMITS_OF_SUMS_AND_PRODUCTS, ids=str)
def test_a_limit_of_a_sum_is_taken_of_its_closed_form(text, expected):
    """Only the answer is held to the functions the notation has names for, not the
    closed form the limit is taken of."""
    assert simp(text) == expected


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
    assert simp("LIM(SIN(x)/ABS(x), x, 0)") == "±1"
    assert simp("LIM(ZETA(s), s, 1)") == "±inf"
    assert simp("LIM(LN(x), x, 0)") == "-inf"


def test_a_limit_towards_a_moving_point_is_the_substitution_it_means():
    """Nothing approaches a point that moves with the variable.

    `LIM(u, x, a)` where `a` mentions `x` is no limit, and what it means is the
    substitution: every `x` goes at once and none of them again, so
    `LIM((x*v - y)^2, y, x*v - y)` is `y^2`. `ODE1.MTH` is written around this -
    `CLAIRAUT` changes variables with it twice in one line - and a side written
    on such a call says nothing, there being nothing to come at from a side.
    """
    assert simp("LIM((x*v - y)^2, y, x*v - y)") == "y^2"
    assert simp("LIM(x^2, x, x + 1)") == "(x + 1)^2"
    assert simp("LIM(x^2, x, x + 1, 1)") == "(x + 1)^2"
    # And the substitution happens after what stands under it is worked out, so
    # that a derivative is differentiated before the point is written into it.
    assert simp("LIM(DIF(LIM((x*v - y)^2, y, x*v - y), y), y, x*v - y)") == (
        "2*v*x - 2*y"
    )


def test_a_calculus_head_waits_for_the_body_of_the_call_under_it():
    """A call whose definition is not written in yet is no constant.

    One pass writes a body in and the next carries the recursion on, so `F(1)`
    stands unresolved while the pass that unfolded `F(2)` is running. It
    mentions no `x`, and differentiating it as it stands would answer zero for
    something that is a derivative of `x^2` throughout.
    """
    body = parse("IF(n = 0, x^2, DIF(F(n - 1), x))", declaring("F"))
    context = Context(functions={"F": (("n",), body)})
    assert simp("F(0)", context) == "x^2"
    assert simp("F(1)", context) == "2*x"
    assert simp("F(2)", context) == "2"
