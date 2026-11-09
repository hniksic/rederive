"""Antiderivatives of rational functions of the variable and a root, by the methods of
integral tables.

The root may be of a linear or quadratic polynomial, of a quotient of linear
polynomials, or the square roots of two linear polynomials.

For a root of a linear `a*x + b`, `t = (a*x + b)^(1/q)` makes the integrand rational
in `t`, which `engine.rational` integrates; likewise `t = ((f*x + g)/(a*x + b))^(1/q)`
for a root of the quotient. `t` is positive by construction, so the answer holds
wherever the integrand is real. For square roots of linear `L1` and `L2`, `t =
SQRT(L1)/SQRT(L2)` works when each term has an even total number of them: `t^2 =
L1/L2`, and `t` is positive where the integrand is real, since `L1/L2` is positive
there.

For the square root `y` of a quadratic `Q = a*x^2 + b*x + c`, the integrand becomes
`R1(x) + R2(x)*y` after writing `y^2` as `Q` and rationalizing the denominator. Only
the second term, `R(x)/y` with `R = R2*Q`, is handled. Its integral is

    A(x)*y + sum of k_i*INT(1/((x - r_i)*y), x) + k*INT(1/y, x),

with `A` rational. A factor of `R`'s denominator that divides `Q` appears in `A`'s
denominator to the same power, and any other factor to one less; the logarithmic
integrals take the simple poles off the roots of `Q`. The unknown coefficients of
`A`'s numerator and the `k`s come from a linear system obtained by differentiating
and comparing coefficients. The identity is algebraic in `y`, so the answer holds on
every interval where `Q` is positive. Sympy takes minutes over `(a*x^2 + b*x +
c)^(7/2)` and splits the answer by cases on the coefficients; this takes a second.

`INT(1/y, x)` is an arcsine when `a` is a negative number. Where the arcsine would
need the absolute value of a parameter (`ASIN(x/ABS(a))`), or `a` is a negated square
as in `c - a^2*x^2`, it is the equal arctangent of `(2*a*x + b)/(2*SQRT(-a)*y)`, e.g.
`ATAN(x/SQRT(a^2 - x^2))`. Otherwise it is `LN(2*SQRT(a)*y + 2*a*x + b)/SQRT(a)`. The
arctangent differentiates back for either sign of the root of `-a`, and the logarithm
for any square root of `a`; for negative `a` the logarithm's argument stays on a
circle in the upper half-plane, never crossing the cut. `INT(1/((x - r)*y), x)` is a
logarithm or arctangent likewise. A denominator factor the ansatz can't handle (a
quadratic not dividing `Q`) is handled when `Q` has no linear term, by the
substitutions `t = x/y` and `w = y`, which make it rational.

If the quadratic is a square, `L^2/(4*a)` with `L` linear, its root is `L*k` with `k`
constant on each side of the root of `L`. The integrand is then rational in `x`, and
the answer is written with `y/L` for `k`.
"""

from __future__ import annotations

import sympy as sp

from .rational import formal_root, rational_antiderivative

__all__ = ["radical_antiderivative"]


def radical_antiderivative(integrand: sp.Basic, variable: sp.Basic) -> sp.Basic | None:
    """An antiderivative of a rational function of the variable and the roots listed in
    the module docstring, or None if the integrand isn't one or the method fails."""
    try:
        return _antiderivative(integrand, variable)
    except Exception:
        return None


def _antiderivative(integrand: sp.Basic, x: sp.Basic) -> sp.Basic | None:
    roots = {
        power
        for power in integrand.atoms(sp.Pow)
        if power.base.has(x) and power.exp.is_Rational and not power.exp.is_Integer
    }
    radicands = {power.base for power in roots}
    if len(radicands) == 2:
        return _by_two_linear_roots(integrand, x, roots)
    if len(radicands) != 1:
        return None
    (radicand,) = radicands
    order = sp.ilcm(1, *(power.exp.q for power in roots))
    if radicand.is_polynomial(x) is not True or sp.degree(radicand, x) == 1:
        return _by_a_fractional_root(integrand, x, radicand, order)
    if sp.degree(radicand, x) != 2 or order != 2:
        return None
    a, b, c = sp.Poly(radicand, x).all_coeffs()
    if sp.cancel(b**2 - 4 * a * c) == 0:
        return _over_a_square(integrand, x, radicand, 2 * a * x + b)
    return _by_reduction(integrand, x, radicand)


def _over_a_square(
    integrand: sp.Basic, x: sp.Basic, radicand: sp.Basic, linear: sp.Basic
) -> sp.Basic | None:
    """The integral under the square root `y` of a quadratic that is `linear^2/(4*a)`.

    `k = y/linear` is constant on each side of the root of `linear`, so with `y =
    k*linear` the integrand is rational in `x` with parameter `k`. Its antiderivative,
    `G0(x) + k*G1(x)` after writing `k^2` as `1/(4*a)`, holds on each side. `G1` is
    shifted to vanish at the root where it is finite there, so the answer is
    continuous across it where the integrand is integrable."""
    k = sp.Dummy("k")
    a = sp.Poly(radicand, x).LC()
    written = integrand.replace(
        lambda e: isinstance(e, sp.Pow) and e.base == radicand and e.exp.is_Rational,
        lambda e: (k * linear) ** int(2 * e.exp) if e.exp.q == 2 else e,
    )
    if written.has(radicand) or written.is_rational_function(x) is not True:
        return None
    found = rational_antiderivative(sp.cancel(written), x, formal=True)
    if found is None or found.has(sp.RootSum, sp.I):
        return None
    mirrored = found.subs(k, -k)
    m = sp.Dummy("m")
    square = 1 / (4 * a)
    even = sp.cancel((found + mirrored) / 2).subs(k, sp.sqrt(m)).subs(m, square)
    odd = sp.cancel((found - mirrored) / (2 * k)).subs(k, sp.sqrt(m)).subs(m, square)
    if even.has(k, m) or odd.has(k, m):
        return None
    if odd != 0:
        odd = _vanishing_at(odd, x, sp.solve(linear, x)[0])
    return sp.factor(even) + odd * sp.sqrt(radicand) / linear


def _by_a_fractional_root(
    integrand: sp.Basic, x: sp.Basic, radicand: sp.Basic, order: int
) -> sp.Basic | None:
    """The integral via `t = radicand^(1/order)`, for a radicand that is linear or a
    quotient of linear polynomials `(f*x + g)/(a*x + b)`, so `x` is rational in `t`.
    Each power `radicand^(k/order)` becomes `t^k`."""
    numerator, denominator = sp.fraction(sp.cancel(sp.together(radicand)))
    if not all(
        part.is_polynomial(x) and sp.degree(part, x) <= 1
        for part in (numerator, denominator)
    ):
        return None
    f, g = _linear_coefficients(numerator, x)
    a, b = _linear_coefficients(denominator, x)
    t = sp.Dummy("t", positive=True)
    written = integrand.replace(
        lambda e: isinstance(e, sp.Pow) and e.base == radicand and e.exp.is_Rational,
        lambda e: t ** (e.exp * order) if not e.exp.is_Integer else e,
    )
    in_t = (g - b * t**order) / (a * t**order - f)
    if sp.cancel(sp.diff(in_t, t)) == 0:
        return None
    root = radicand ** sp.Rational(1, order)
    return _over_the_substitution(written, x, t, in_t, order, sp.cancel(radicand), root)


def _by_two_linear_roots(
    integrand: sp.Basic, x: sp.Basic, roots: set[sp.Basic]
) -> sp.Basic | None:
    """The integral of a rational function of `x` and the square roots `r1`, `r2` of
    linear `L1`, `L2`, via `t = r1/r2`, when it is rational in `x`, `t` and `r2^2`.

    `t^2 = L1/L2`, so `x` is rational in `t`, and `r1^k` is `t^k*r2^k`. Where the
    integrand is real, `L1/L2` is positive and so is `t`: either both roots are real,
    or both imaginary with `t = SQRT(-L1)/SQRT(-L2)`."""
    if any(power.exp.q != 2 for power in roots):
        return None
    first, second = sorted({power.base for power in roots}, key=sp.default_sort_key)
    if not all(
        root.is_polynomial(x) and sp.degree(root, x) == 1 for root in (first, second)
    ):
        return None
    t = sp.Dummy("t", positive=True)
    s = sp.Dummy("s")
    written = integrand.replace(
        lambda e: isinstance(e, sp.Pow) and e.base in (first, second) and e.exp.q == 2,
        lambda e: (t * s if e.base == first else s) ** int(2 * e.exp),
    )
    m = sp.Dummy("m")
    written = sp.cancel(sp.cancel(written).subs(s, sp.sqrt(m)))
    if written.has(s) or written.is_rational_function(x, t, m) is not True:
        return None
    written = written.subs(m, second)
    a, b = _linear_coefficients(first, x)
    f, g = _linear_coefficients(second, x)
    in_t = (b - g * t**2) / (f * t**2 - a)
    if sp.cancel(sp.diff(in_t, t)) == 0:
        return None
    root = sp.sqrt(first) / sp.sqrt(second)
    square = sp.cancel(first / second)
    return _over_the_substitution(written, x, t, in_t, 2, square, root)


def _linear_coefficients(linear: sp.Basic, x: sp.Basic) -> tuple[sp.Basic, sp.Basic]:
    """The coefficients of `x` and 1 in a polynomial of degree at most one."""
    poly = sp.Poly(linear, x)
    return poly.coeff_monomial(x), poly.coeff_monomial(1)


def _over_the_substitution(
    written: sp.Basic,
    x: sp.Basic,
    t: sp.Basic,
    in_t: sp.Basic,
    order: int,
    power: sp.Basic,
    root: sp.Basic,
) -> sp.Basic | None:
    """The integral of `written`, rational in `x` and `t`, via the substitution `x =
    in_t`, where `t` stands for `root` and `t^order` for `power`."""
    transformed = sp.cancel(written.subs(x, in_t) * sp.diff(in_t, t))
    if transformed.has(x) or transformed.is_rational_function(t) is not True:
        return None
    found = rational_antiderivative(transformed, t, binomials=True, formal=True)
    if found is None or found.has(sp.RootSum, sp.I):
        return None
    transcendental, algebraic = sp.S.Zero, sp.S.Zero
    for term in sp.Add.make_args(found):
        if term.has(sp.log, sp.atan):
            transcendental += sp.factor(term)
        else:
            algebraic += term
    return transcendental.subs(t, root) + _in_the_root(algebraic, t, power, order, root)


def _in_the_root(
    algebraic: sp.Basic, t: sp.Basic, power: sp.Basic, order: int, root: sp.Basic
) -> sp.Basic:
    """The algebraic part of an antiderivative in `t` (the root), written back in `x`,
    where `power` is `t^order` in `x`. For a square root it is written as `E(power) +
    O(power)*t` with each part factored: `2*(a*x + b)^(3/2)*(3*a*x - 2*b)/(15*a^2)`
    rather than a sum of powers of the root."""
    if order != 2:
        return sp.Add(
            *(sp.factor(term.subs(t, root)) for term in sp.Add.make_args(algebraic))
        )
    mirrored = algebraic.subs(t, -t)
    s = sp.Dummy("s")
    even = sp.cancel((algebraic + mirrored) / 2).subs(t, sp.sqrt(s))
    odd = sp.cancel((algebraic - mirrored) / (2 * t)).subs(t, sp.sqrt(s))
    return sp.factor(even.subs(s, power)) + sp.factor(odd.subs(s, power)) * root


def _vanishing_at(value: sp.Basic, x: sp.Basic, point: sp.Basic) -> sp.Basic:
    """The value minus its limit at the point, if finite, with each logarithm `LN(A)`
    finite there rewritten as `LN(A/A(point))`."""
    infinite = (sp.oo, -sp.oo, sp.zoo, sp.nan, sp.Limit)
    at_the_point = sp.limit(value, x, point)
    if at_the_point.is_finite is False or at_the_point.has(*infinite):
        return value
    for logarithm in value.atoms(sp.log):
        argument = logarithm.args[0]
        there = sp.cancel(argument.subs(x, point))
        if there != 0 and not there.has(*infinite):
            value = value.xreplace({logarithm: sp.log(sp.factor(argument / there))})
    return sp.factor(value - sp.limit(value, x, point))


def _by_reduction(
    integrand: sp.Basic, x: sp.Basic, radicand: sp.Basic
) -> sp.Basic | None:
    """The integral of `R1(x) + R2(x)*y`, `y` being the square root of the quadratic,
    by the ansatz in the module docstring, when `R1` is zero."""
    y = sp.Dummy("y")
    quadratic = sp.expand(radicand)
    written = integrand.replace(
        lambda e: isinstance(e, sp.Pow) and e.base == radicand and e.exp.is_Rational,
        lambda e: y ** int(2 * e.exp) if e.exp.q == 2 else e,
    )
    if written.is_rational_function(x, y) is not True:
        return None
    numerator, denominator = sp.fraction(sp.cancel(sp.together(written)))
    modulus = sp.Poly(y**2 - quadratic, y)
    numerator = sp.Poly(numerator, y).rem(modulus)
    denominator = sp.Poly(denominator, y).rem(modulus)
    p0, p1 = _halves(numerator)
    q0, q1 = _halves(denominator)
    norm = sp.expand(q0**2 - q1**2 * quadratic)
    if sp.cancel(p0 * q0 - p1 * q1 * quadratic) != 0:
        return None
    reduced = sp.cancel((p1 * q0 - p0 * q1) * quadratic / norm)
    return _reduced(reduced, x, quadratic, sp.sqrt(radicand))


def _halves(poly: sp.Poly) -> tuple[sp.Basic, sp.Basic]:
    """The coefficients of 1 and `y` in a polynomial of degree at most 1 in `y`."""
    coefficients = poly.all_coeffs()
    if len(coefficients) == 1:
        return coefficients[0], sp.S.Zero
    return coefficients[1], coefficients[0]


def _reduced(
    rational: sp.Basic, x: sp.Basic, quadratic: sp.Basic, root: sp.Basic
) -> sp.Basic | None:
    """`INT(rational/root, x)`, where root is the square root of the quadratic, as
    `A*root` plus multiples of the integrals over linear factors and of `1/root`."""
    numerator, denominator = sp.fraction(rational)
    numerator, denominator = sp.Poly(numerator, x), sp.Poly(denominator, x)
    q = sp.Poly(quadratic, x)
    _, factors = sp.factor_list(denominator.as_expr(), x)
    below = sp.S.One
    simple = []
    for factor, multiplicity in factors:
        factor_poly = sp.Poly(factor, x)
        if factor_poly.degree() == 0:
            continue
        if sp.cancel(q.rem(factor_poly).as_expr()) == 0:
            below *= factor**multiplicity
        elif factor_poly.degree() == 1:
            below *= factor ** (multiplicity - 1)
            simple.append(factor)
        else:
            return _by_substitution(rational, x, quadratic, root)
    below_degree = sp.degree(below, x)
    excess = numerator.degree() - denominator.degree()
    size = below_degree + max(excess - 1, 0)
    unknowns = sp.symbols(f"n0:{size + 1}", cls=sp.Dummy)
    weights = sp.symbols(f"k0:{len(simple) + 1}", cls=sp.Dummy)
    top = sum(n * x**i for i, n in enumerate(unknowns))
    algebraic = top / below
    derivative = (
        sp.diff(algebraic, x) * quadratic + algebraic * sp.diff(quadratic, x) / 2
    )
    model = derivative + weights[0] + sum(k / f for k, f in zip(weights[1:], simple))
    difference = sp.together(model - rational)
    equations = sp.Poly(sp.expand(sp.numer(difference)), x).coeffs()
    solutions = sp.linsolve(equations, [*unknowns, *weights])
    if not solutions:
        return None
    (solution,) = solutions
    values = dict(zip([*unknowns, *weights], solution))
    if any(value.free_symbols & set(values) for value in values.values()):
        values = {k: v.subs({u: 0 for u in values}) for k, v in values.items()}
    result = _times_the_root(sp.cancel(algebraic.subs(values)), x, quadratic, root)
    if values[weights[0]] != 0:
        logarithmic = _over_the_root(quadratic, x, root)
        if logarithmic is None:
            return None
        result += sp.factor(values[weights[0]]) * logarithmic
    for weight, factor in zip(weights[1:], simple):
        coefficient = sp.cancel(values[weight])
        if coefficient != 0:
            logarithmic = _over_a_factor_and_the_root(factor, x, quadratic, root)
            if logarithmic is None:
                return None
            result += coefficient * logarithmic
    return result


def _over_a_factor_and_the_root(
    factor: sp.Basic, x: sp.Basic, quadratic: sp.Basic, root: sp.Basic
) -> sp.Basic | None:
    """`INT(1/(factor*root), x)` for a linear factor `l*(x - r)` not dividing the
    quadratic, root being its square root, or None if the quadratic is a square.

    With `u = x - r`, the quadratic is `a*u^2 + p*u + s`, `s = Q(r)` nonzero and `p =
    Q'(r)`. The integral is `LN((2*w*root - 2*s - p*u)/u)/(l*w)` for any `w` with
    `w^2 = s`, or, where `s` is `-w^2`, the real `ATAN((2*s + p*u)/(2*w*root))/(l*w)`,
    which is `ATAN(root/w)/(l*w)` plus a constant when `p` is zero. Each differentiates
    back for either sign of `w`. Where the quadratic is positive, `u` keeps its sign
    and the logarithm is continuous: for real `w` its argument is real and nonzero
    (its product with `2*w*root + 2*s + p*u` is `(4*a*s - p^2)*u^2`), and for imaginary
    `w` the argument's imaginary part `2*w*root/u` keeps its sign, so the cut is never
    crossed."""
    slope, offset = sp.Poly(factor, x).all_coeffs()
    r = -offset / slope
    q = sp.Poly(quadratic, x)
    s = sp.factor(q.eval(r))
    p = sp.factor(q.diff(x).eval(r))
    if sp.cancel(p**2 - 4 * q.LC() * s) == 0:
        return None
    opposite = formal_root(-s)
    if s.could_extract_minus_sign() and opposite.is_extended_real:
        if p == 0:
            return sp.atan(root / opposite) / (slope * opposite)
        argument = sp.factor((2 * s * slope + p * factor) / (2 * opposite * slope))
        return sp.atan(argument / root) / (slope * opposite)
    scale = formal_root(s)
    numerator = sp.expand(slope * (2 * scale * root - 2 * s) - p * factor)
    _, numerator = numerator.as_content_primitive()
    if numerator.could_extract_minus_sign() and factor.could_extract_minus_sign():
        numerator, factor = -numerator, -factor
    return sp.log(numerator / factor) / (slope * scale)


def _by_substitution(
    rational: sp.Basic, x: sp.Basic, quadratic: sp.Basic, root: sp.Basic
) -> sp.Basic | None:
    """`INT(rational/root, x)` for a quadratic without a linear term, `f*x^2 + g`, as
    rational integrals, or None.

    The rational function is split into an even part `E(x^2)` and odd part
    `x*O(x^2)`. With `t = x/root`, `x^2 = g*t^2/(1 - f*t^2)` and `dx/root = dt/(1 -
    f*t^2)`, so the even part is rational in `t`; with `w = root`, `x*dx/root = dw/f`,
    so the odd part is rational in `w`. Where the quadratic is positive, `t` is
    monotonic (its derivative is `g/root^3`) and `w` positive, so the results hold on
    each such interval. This handles denominators the ansatz can't, like
    `INT(1/((a*x^2 + b)*SQRT(f*x^2 + g)), x)`.
    """
    f, middle, g = sp.Poly(quadratic, x).all_coeffs()
    if middle != 0:
        return None
    mirrored = rational.subs(x, -x)
    even = sp.cancel((rational + mirrored) / 2)
    odd = sp.cancel((rational - mirrored) / (2 * x))
    z = sp.Dummy("z")
    t, w = sp.Dummy("t"), sp.Dummy("w", positive=True)
    result = sp.S.Zero
    for part, square, measure, variable, back in (
        (even, g * t**2 / (1 - f * t**2), 1 / (1 - f * t**2), t, x / root),
        (odd, (w**2 - g) / f, 1 / f, w, root),
    ):
        if part == 0:
            continue
        in_square = sp.cancel(part.subs(x, sp.sqrt(z)))
        if in_square.has(x) or in_square.is_rational_function(z) is not True:
            return None
        transformed = sp.cancel(in_square.subs(z, square) * measure)
        found = rational_antiderivative(
            transformed, variable, binomials=True, formal=True
        )
        if found is None or found.has(sp.RootSum, sp.I):
            return None
        result += found.subs(variable, back)
    return result


def _times_the_root(
    rational: sp.Basic, x: sp.Basic, quadratic: sp.Basic, root: sp.Basic
) -> sp.Basic:
    """`rational*root`, with powers of the quadratic in the numerator and denominator
    folded into the root's exponent: `x/SQRT(x^2 - a^2)` rather than `x*SQRT(x^2 -
    a^2)/(x^2 - a^2)`."""
    numerator, denominator = sp.fraction(rational)
    q = sp.Poly(quadratic, x)
    numerator, raised = _without_powers_of(sp.Poly(numerator, x), q)
    denominator, lowered = _without_powers_of(sp.Poly(denominator, x), q)
    exponent = sp.S.Half + raised - lowered
    return sp.factor(numerator.as_expr() / denominator.as_expr()) * root.base**exponent


def _without_powers_of(poly: sp.Poly, q: sp.Poly) -> tuple[sp.Poly, int]:
    """The polynomial divided by the highest power of `q` that divides it, and that
    power."""
    power = 0
    while not poly.is_zero and poly.degree() >= q.degree():
        quotient, remainder = poly.div(q)
        if not remainder.is_zero:
            break
        poly, power = quotient, power + 1
    return poly, power


def _over_the_root(
    quadratic: sp.Basic, x: sp.Basic, root: sp.Basic
) -> sp.Basic | None:
    """`INT(1/root, x)`, root being the square root of the quadratic, or None if the
    quadratic is a square (making the root an absolute value)."""
    a, b, c = sp.Poly(quadratic, x).all_coeffs()
    if sp.cancel(b**2 - 4 * a * c) == 0:
        return None
    if a.is_number and a.is_negative:
        discriminant = sp.sqrt(sp.factor(b**2 - 4 * a * c))
        if not discriminant.has(sp.Abs):
            return -sp.asin((2 * a * x + b) / discriminant) / sp.sqrt(-a)
    opposite = formal_root(-a)
    if a.could_extract_minus_sign() and opposite.is_extended_real:
        # d/dx ATAN((2*a*x + b)/(2*r*root)) = -r/root for any r with r^2 = -a.
        return -sp.atan((2 * a * x + b) / (2 * opposite * root)) / opposite
    scale = formal_root(a)
    argument = 2 * scale * root + 2 * a * x + b
    _, argument = argument.as_content_primitive()
    return sp.log(argument) / scale
