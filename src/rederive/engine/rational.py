"""The antiderivative of a rational function, in the real form Derive writes.

Sympy's `ratint` splits a rational function into rational and logarithmic parts
(Horowitz-Ostrogradsky), writes the logarithmic part as sums of `a*LN(h(a, x))` over
the roots `a` of a polynomial `q` (Lazard-Rioboo-Trager), and turns each conjugate pair
into a real logarithm and arctangent, as Derive prints it. In sympy 1.14 that last step
loses roots: where the roots' coordinates involve a cube root, as for `1/(x^6 + 16)`,
counting real roots raises a `DomainError` that is taken to mean "no roots". Two pairs
are silently dropped, and `INT(1/(x^6 + 16), x)` returns a third of the answer. For a
quintic such as `1/(x^5 + 16)` it can't find the roots at all and leaves a `ROOT_SUM`
despite an answer in radicals.

So rational integrands bypass sympy's `integrate`. They are split with `ratint`'s
public helpers `ratint_ratpart` and `ratint_logpart`, and the logarithmic part is made
real here from the roots of `q`: each conjugate pair gives a logarithm and an
arctangent via `log_to_atan`, and each known-real root a logarithm. If not all roots
are found, or a root's realness is undecidable, the sum stays a `RootSum`, as with
`ratint`. Only documented sympy API is used.

Upstream commit 9bef76b40d39842c063599e98b919c0b3e09fee7 ("fix(integrals): don't lose
roots in ratint", PR 28189, merged 2025-07-09, after 1.14.0 and unreleased at the time
of writing) makes `log_to_real` do what `_real_form` does. Once sympy includes it, this
module and `_of_a_rational_function` in `integration.indefinite` can go, and rational
integrands can use sympy's `doit`. Two differences need keeping or upstreaming first:
`_real_form` gives up on a root with no conjugate that isn't known real, whereas
upstream assumes it real and puts `#i` into the answer for `1/(x^4 + x + 1)`; and
`ratint` doesn't avoid the roots of parametric quadratics, as
`_over_parametric_quadratics` does.
"""

from __future__ import annotations

import sympy as sp
from sympy.core.sorting import ordered
from sympy.integrals.rationaltools import log_to_atan, ratint_logpart, ratint_ratpart
from sympy.polys.polyroots import roots

__all__ = ["formal_root", "over_a_parametric_quadratic", "rational_antiderivative"]


def rational_antiderivative(
    integrand: sp.Basic,
    variable: sp.Basic,
    binomials: bool = False,
    formal: bool = False,
) -> sp.Basic | None:
    """An antiderivative of a rational function of the variable, or None.

    Follows `ratint`'s steps via its public helpers; only the real form of the
    logarithmic part is done here. None if the integrand isn't a rational function of
    the variable or sympy's polynomial arithmetic fails on it.

    With `binomials`, a parametric quadratic without a linear term, `b^2*x^2 + c^2`,
    is handled like one with a linear term rather than over its roots.

    With `formal`, the square root of `4*a*c - b^2` in such a quadratic's arctangent
    takes each squared factor's base out unchanged, sign and all (`a*b` for
    `a^2*b^2`). The formula differentiates back for any square root, and this one
    prints without `ABS`.
    """
    if integrand.is_rational_function(variable) is not True:
        return None
    try:
        return _antiderivative(integrand, variable, binomials, formal)
    except Exception:
        return None


def over_a_parametric_quadratic(
    integrand: sp.Basic, variable: sp.Basic, binomials: bool = False
) -> bool:
    """Whether the integrand is a rational function of the variable whose denominator
    has a `_parametric_quadratic` factor (with `binomials`, including ones without a
    linear term).

    Its antiderivative is written here as in integral tables, with a logarithm and an
    arctangent of the quadratic, and nothing elsewhere gives a shorter one.
    """
    if integrand.is_rational_function(variable) is not True:
        return False
    try:
        _, factors = sp.factor_list(sp.denom(sp.cancel(integrand)), variable)
        return any(
            _parametric_quadratic(sp.Poly(factor, variable), binomials)
            for factor, _ in factors
        )
    except Exception:
        return False


def _antiderivative(
    integrand: sp.Basic,
    variable: sp.Basic,
    binomials: bool = False,
    formal: bool = False,
) -> sp.Basic:
    p, q = integrand.as_numer_denom()
    p = sp.Poly(p, variable, composite=False, field=True)
    q = sp.Poly(q, variable, composite=False, field=True)
    coefficient, p, q = p.cancel(q)
    polynomial, p = p.div(q)
    result = polynomial.integrate(variable).as_expr()
    if p.is_zero:
        return coefficient * result
    rational, remainder = ratint_ratpart(p, q, variable)
    numerator, denominator = remainder.as_numer_denom()
    numerator, denominator = sp.Poly(numerator, variable), sp.Poly(denominator, variable)
    polynomial, numerator = numerator.div(denominator)
    result += rational + polynomial.integrate(variable).as_expr()
    if not numerator.is_zero:
        pieces = _over_parametric_quadratics(numerator, denominator, variable, binomials)
        if pieces is None:
            result += _logarithmic(numerator, denominator, variable)
        else:
            result += _merged(
                sum(_of_a_factor(n, factor, variable, formal) for n, factor in pieces),
                variable,
            )
    return coefficient * result


def _logarithmic(numerator: sp.Poly, denominator: sp.Poly, x: sp.Basic) -> sp.Basic:
    """The integral of a proper fraction over a squarefree denominator, via
    `ratint_logpart`, made real by `_real_form` where possible."""
    t = sp.Dummy("t")
    result = sp.S.Zero
    for h, q in ratint_logpart(numerator, denominator, x, t):
        _, h = h.primitive()
        real = _real_form(h, q, x, t)
        if real is None:
            factors = _factors(q)
            if len(factors) > 1:
                real = sum(_over_a_factor(h, factor, x, t) for factor in factors)
            else:
                real = sp.RootSum(q, sp.Lambda(t, t * sp.log(h.as_expr())), quadratic=True)
        result += real
    return result


def _factors(q: sp.Poly) -> list[sp.Poly]:
    """The irreducible factors of `q`, or just `q` if factoring fails."""
    try:
        _, factors = q.factor_list()
    except Exception:
        return [q]
    return [factor for factor, _ in factors] or [q]


def _over_a_factor(h: sp.Poly, factor: sp.Poly, x: sp.Basic, t: sp.Basic) -> sp.Basic:
    """The sum of `a*LN(h(a, x))` over the roots `a` of one factor of `q`, made real by
    `_real_form` where possible, else a `RootSum`.

    Summing per factor lets one factor's roots be used when another's can't be found:
    the logarithmic part of `1/(x^5 + x + 1)` is over the roots of a quadratic and a
    cubic, and the quadratic still gives a logarithm and arctangent in `x^2 + x + 1`.
    """
    real = _real_form(h, factor, x, t)
    if real is not None:
        return real
    return sp.RootSum(factor, sp.Lambda(t, t * sp.log(h.as_expr())), quadratic=True)


def _over_parametric_quadratics(
    numerator: sp.Poly, denominator: sp.Poly, x: sp.Basic, binomials: bool = False
) -> list[tuple[sp.Poly, sp.Poly]] | None:
    """The proper fraction split into partial fractions over the irreducible factors of
    its squarefree denominator, as (numerator, factor) pairs, or None if no factor is a
    `_parametric_quadratic`.

    The roots of `a*x^2 + b*x + c` are radicals in its coefficients, and a logarithmic
    part over them is a page of nested square roots that takes later steps many
    seconds to process. Partial fractions over the irreducible factors, as in integral
    tables, need no quadratic roots.
    """
    _, factors = sp.factor_list(denominator.as_expr(), x)
    factors = [sp.Poly(factor, x, field=True) for factor, _ in factors]
    if not any(_parametric_quadratic(factor, binomials) for factor in factors):
        return None
    numerator = sp.Poly(numerator.as_expr(), x, field=True)
    denominator = sp.Poly(denominator.as_expr(), x, field=True)
    product = sp.Poly(sp.Mul(*(f.as_expr() for f in factors)), x, field=True)
    # The denominator is the product of the factors times this constant.
    scale = sp.cancel(denominator.LC() / product.LC())
    pieces = []
    for factor in factors:
        cofactor = product.exquo(factor)
        piece = (numerator * cofactor.invert(factor)).rem(factor)
        piece = sp.Poly(sp.cancel(piece.as_expr() / scale), x, field=True)
        if not piece.is_zero:
            pieces.append((piece, factor))
    return pieces


def _parametric_quadratic(factor: sp.Poly, binomials: bool = False) -> bool:
    """Whether the factor is a quadratic with a non-numeric coefficient and a linear
    term (or any such quadratic with `binomials`)."""
    if factor.degree() != 2:
        return False
    coefficients = factor.all_coeffs()
    linear = binomials or coefficients[1] != 0
    return linear and not all(c.is_number for c in coefficients)


def _merged(value: sp.Basic, x: sp.Basic) -> sp.Basic:
    """The sum with logarithms of polynomials sharing a coefficient merged:
    `k*LN(u) + k*LN(v)` becomes `k*LN(u*v)`.

    The two differ by a multiple of `2*pi*#i`, constant wherever `u` and `v` are
    nonzero, so both are antiderivatives. This rejoins factors that
    `_over_parametric_quadratics` split apart, as in `-LN(x^3 + a^3)/3`.
    """
    groups: dict[sp.Basic, list[sp.Basic]] = {}
    others = []
    for term in sp.Add.make_args(value):
        k, rest = term.as_independent(x, as_Add=False)
        if isinstance(rest, sp.log) and rest.args[0].is_polynomial(x):
            for known in groups:
                if sp.cancel(known - k) == 0:
                    groups[known].append(rest.args[0])
                    break
            else:
                groups[k] = [rest.args[0]]
        else:
            others.append(term)
    merged = [k * sp.log(sp.expand(sp.Mul(*args))) for k, args in groups.items()]
    return sp.Add(*others, *merged)


def _of_a_factor(
    numerator: sp.Poly, factor: sp.Poly, x: sp.Basic, formal: bool = False
) -> sp.Basic:
    """The integral of a partial fraction over one irreducible factor.

    Over a quadratic `a*x^2 + b*x + c`, with `d = 4*a*c - b^2`, the integral of `(m*x
    + n)/(a*x^2 + b*x + c)` is `m/(2*a)*LN(a*x^2 + b*x + c) + (2*a*n - b*m)/a*ATAN((2*a*x
    + b)/SQRT(d))/SQRT(d)`. For negative `d` this is the `ATANH` form, via the
    arctangent of an imaginary argument, and still an antiderivative between the
    factor's real roots. Where `a` or `d` is zero it is undefined rather than wrong.
    """
    if factor.degree() == 1:
        return numerator.LC() / factor.LC() * sp.log(factor.as_expr())
    if factor.degree() != 2:
        return _logarithmic(numerator, factor, x)
    a, b, c = factor.all_coeffs()
    m, n = numerator.all_coeffs() if numerator.degree() == 1 else (0, numerator.LC())
    d = sp.expand(4 * a * c - b**2)
    result = sp.factor(m / (2 * a)) * sp.log(factor.as_expr())
    if formal and b == 0:
        # d/dx ATAN(r_a*x/r_c)/(r_a*r_c) = 1/(r_c^2 + r_a^2*x^2), for any roots r_a, r_c
        # of a and c.
        root_a, root_c = formal_root(a), formal_root(c)
        return result + sp.factor(n / (root_a * root_c)) * sp.atan(root_a * x / root_c)
    root = formal_root(d) if formal else sp.sqrt(d)
    result += sp.factor((2 * a * n - b * m) / a) * sp.atan((2 * a * x + b) / root) / root
    return result


def formal_root(value: sp.Basic) -> sp.Basic:
    """A square root of the value, with each squared factor's base taken out as is,
    sign and all: `a` for `a^2`. Useful where only the square matters."""
    outside, inside = sp.S.One, sp.S.One
    for factor in sp.Mul.make_args(sp.factor(value)):
        base, exponent = factor.as_base_exp()
        if exponent.is_Integer and exponent % 2 == 0:
            outside *= base ** (exponent // 2)
        else:
            inside *= factor
    return outside * sp.sqrt(inside)


def _real_form(h: sp.Poly, q: sp.Poly, x: sp.Basic, t: sp.Basic) -> sp.Basic | None:
    """The sum of `a*LN(h(a, x))` over the roots `a` of `q`, in real form, or None.

    For a conjugate pair `u +- i*v`, with `h(u + i*v, x)` split into `A + i*B`, the
    pair's two terms sum to `u*LN(A^2 + B^2)` plus `v` times the arctangent
    `log_to_atan` makes of `A` and `B`; a real root `r` contributes `r*LN(h(r, x))`.
    """
    separated = _roots_real_complex(q)
    if separated is None:
        return None
    reals, complexes = separated
    u, v = sp.Dummy("u"), sp.Dummy("v")
    split = sp.collect(h.as_expr().xreplace({t: u + sp.I * v}).expand(), sp.I, evaluate=False)
    a, b = split.get(sp.S.One, sp.S.Zero), split.get(sp.I, sp.S.Zero)
    result = sp.S.Zero
    for r_u, r_v in complexes:
        A = sp.Poly(a.xreplace({u: r_u, v: r_v}), x)
        B = sp.Poly(b.xreplace({u: r_u, v: r_v}), x)
        result += r_u * sp.log((A**2 + B**2).as_expr()) + r_v * log_to_atan(A, B)
    for r in reals:
        result += r * sp.log(h.as_expr().subs(t, r))
    return result


def _roots_real_complex(q: sp.Poly) -> tuple[dict, dict] | None:
    """The real roots of `q` and one root of each conjugate pair, or None.

    None if `roots` finds fewer roots than the degree, or a root has no conjugate and
    isn't known real: the quartic formula writes the roots of `x^4 + x + 1` via the
    cube root of a complex number, their conjugates are other expressions, and summing
    them individually would put complex coefficients into a real answer.

    Conjugates are matched symbolically via `conjugate`, and for numeric real `q` also
    by value: `roots` writes each root of `16777216*t^8 + 1` differently, so none is
    the symbolic conjugate of another. For real `q` a value match is a true pair; it
    is written via the real and imaginary parts of the shorter root, which must be
    known real.

    Also None if a root contains a cube or higher root of an irrational number, as
    Cardano's formula gives for every cubic with one real root. For `INT(1/(x^3 - x^2
    + 1), x)`, later steps (`log_to_atan` first) test coefficients for zero using
    minimal polynomials that take minutes to build, for a pages-long answer. Roots of
    rationals, as in `1/(x^6 + 16)`, and square roots, nested or not, are what Derive's
    answers use.
    """
    found = roots(q)
    if sum(found.values()) != q.degree():
        return None
    if any(_past_square_roots(r) for r in found):
        return None
    numeric = all(c.is_number and c.is_extended_real for c in q.all_coeffs())
    reals: dict = {}
    complexes: dict = {}
    remaining = list(found)
    while remaining:
        r = remaining.pop()
        conjugate = r.conjugate()
        if r != conjugate and conjugate in remaining:
            remaining.remove(conjugate)
            r_re, r_im = r.as_real_imag()
        elif r.is_extended_real:
            reals[r] = found[r]
            continue
        elif not numeric or (conjugate := _conjugate_by_value(r, remaining)) is None:
            return None
        else:
            remaining.remove(conjugate)
            r = min(r, conjugate, key=sp.count_ops)
            r_re, r_im = r.as_real_imag()
            if not (r_re.is_extended_real and r_im.is_extended_real):
                return None
        _, r_im = ordered([r_im, -r_im])
        complexes[(r_re, r_im)] = found[r]
    return reals, complexes


def _conjugate_by_value(r: sp.Basic, others: list) -> sp.Basic | None:
    """The unique element of `others` numerically equal to the conjugate of `r`, or None
    (also if `r` is real)."""
    value = sp.N(r, 30)
    if abs(sp.im(value)) <= sp.Float("1e-20", 30) * (1 + abs(value)):
        return None
    close = [
        other
        for other in others
        if abs(sp.N(other, 30) - sp.conjugate(value)) <= sp.Float("1e-20", 30) * abs(value)
    ]
    return close[0] if len(close) == 1 else None


def _past_square_roots(root: sp.Basic) -> bool:
    """Whether the root contains a cube or higher root of an irrational number (see
    `_roots_real_complex`)."""
    return any(
        power.exp.is_Rational and power.exp.q > 2 and not power.base.is_Rational
        for power in root.atoms(sp.Pow)
    )
