"""The antiderivative of a rational function, in the real form Derive writes.

Sympy integrates a rational function by `ratint`: the Horowitz-Ostrogradsky split
into a rational part and a logarithmic part, then the Lazard-Rioboo-Trager method,
which writes the logarithmic part as sums of `a*LN(h(a, x))` over the roots `a` of
a polynomial `q`. Each conjugate pair of roots is then written as a real logarithm
and an arctangent, which is the form Derive prints. In sympy 1.14 that last step
loses roots: it finds the pairs through a resultant and a count of real roots, and
where the roots' coordinates carry a cube root, as for `1/(x^6 + 16)`, the count
raises a `DomainError` that is taken for "no roots here". The two pairs that live
there are dropped without a word, and `INT(1/(x^6 + 16), x)` comes back as the one
arctangent in `2^(1/3)*x/2`, a third of the answer, with nothing to say it is wrong.
Over a quintic such as `1/(x^5 + 16)` the same step cannot find the roots at all,
and leaves a `ROOT_SUM` where there is an answer in radicals.

So a rational integrand is not handed to sympy's `integrate` at all. It is split
with the same two public functions `ratint` uses, `ratint_ratpart` and
`ratint_logpart`, and the logarithmic part is written in real form here, from the
roots of `q` themselves: each root is paired with its conjugate, each pair gives a
logarithm and an arctangent through `log_to_atan`, and each root known to be real
gives a logarithm. Where the roots cannot all be found, or one is of a realness
nothing can decide, the sum stays a `RootSum`, which is what `ratint` leaves in that
case as well. Everything here is built on what sympy documents; nothing in sympy is
altered.

The right way, once the sympy depended on carries upstream commit
9bef76b40d39842c063599e98b919c0b3e09fee7 ("fix(integrals): don't lose roots in
ratint", merged in pull request 28189 on 2025-07-09, after the 1.14.0 release and
in no release at the time of writing): that commit rewrites `log_to_real` to do
what `_real_form` below does, and `ratint` then answers every case here the same
way. This module then goes, `_of_a_rational_function` in `pipeline` goes with it,
and sympy's own `doit` is what a rational integrand gets, as any other integrand
does. One difference is worth keeping or offering upstream first: `_real_form`
gives up on a root that has no conjugate among the others and is not known to be
real, where the upstream rewrite takes it for real and writes `#i` into the answer
to `1/(x^4 + x + 1)`. Dropping this module without that check prints that `#i`.
Nor does `ratint` write a quadratic with parametric coefficients without its roots,
which `_over_parametric_quadratics` does.
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

    The steps are `ratint`'s own, over its public halves; only the writing of the
    logarithmic part in real form is this module's. None where the integrand is no
    rational function of the variable, or where sympy's polynomial arithmetic
    refuses it.

    With `binomials`, a parametric quadratic with no linear term, `b^2*x^2 + c^2`, is
    written as one with a linear term is, rather than over its roots.

    With `formal`, the square root of `4*a*c - b^2` in the arctangent of such a
    quadratic is taken with each squared factor's base brought out as it stands, sign
    and all, `a*b` for `a^2*b^2`: the formula differentiates back for any square root,
    and this one prints without `ABS`.
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
    has a `_parametric_quadratic` factor, one with no linear term among them where
    `binomials` is given.

    The antiderivative of one is written here the way the integral tables write it,
    with a logarithm and an arctangent of the quadratic, and nothing shorter is to
    be had for it from elsewhere.
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
    """The integral of a proper fraction over a squarefree denominator, by
    `ratint_logpart` and written real by `_real_form` where it can be."""
    t = sp.Dummy("t")
    result = sp.S.Zero
    for h, q in ratint_logpart(numerator, denominator, x, t):
        _, h = h.primitive()
        real = _real_form(h, q, x, t)
        if real is None:
            real = sp.RootSum(q, sp.Lambda(t, t * sp.log(h.as_expr())), quadratic=True)
        result += real
    return result


def _over_parametric_quadratics(
    numerator: sp.Poly, denominator: sp.Poly, x: sp.Basic, binomials: bool = False
) -> list[tuple[sp.Poly, sp.Poly]] | None:
    """The proper fraction over a squarefree denominator as partial fractions over its
    irreducible factors, each a numerator and its factor, or None where no factor is a
    `_parametric_quadratic`.

    The roots of `a*x^2 + b*x + c` are radicals in its coefficients, and a
    logarithmic part written over them is a page of nested square roots that
    everything after this takes many seconds to work through. Over the irreducible
    factors of the denominator, which the integral tables use, the fraction needs
    no roots of a quadratic at all.
    """
    _, factors = sp.factor_list(denominator.as_expr(), x)
    factors = [sp.Poly(factor, x, field=True) for factor, _ in factors]
    if not any(_parametric_quadratic(factor, binomials) for factor in factors):
        return None
    numerator = sp.Poly(numerator.as_expr(), x, field=True)
    denominator = sp.Poly(denominator.as_expr(), x, field=True)
    product = sp.Poly(sp.Mul(*(f.as_expr() for f in factors)), x, field=True)
    # The denominator is the product of its factors times a constant.
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
    """Whether the factor is a quadratic with a linear term, or any quadratic with
    `binomials`, and a coefficient that is not a number."""
    if factor.degree() != 2:
        return False
    coefficients = factor.all_coeffs()
    linear = binomials or coefficients[1] != 0
    return linear and not all(c.is_number for c in coefficients)


def _merged(value: sp.Basic, x: sp.Basic) -> sp.Basic:
    """The sum with its logarithms of polynomials that have the same coefficient
    taken together, `k*LN(u) + k*LN(v)` as `k*LN(u*v)`.

    The two differ by a multiple of `2*pi*#i` that is constant wherever `u` and `v`
    are nonzero, so either is an antiderivative where the other is. The fractions
    `_over_parametric_quadratics` makes keep apart factors that the logarithmic part
    of their common fraction would have taken together, as in `-LN(x^3 + a^3)/3`.
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
    + b)/SQRT(d))/SQRT(d)`. Where `d` is negative that is the `ATANH` form, through the
    arctangent of an imaginary argument, and an antiderivative on each interval
    between the two real roots of the factor just as well; where `a` or `d` is zero
    it is undefined rather than wrong.
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
    """A square root of the value, with each squared factor's base taken out as it
    stands, sign and all: `a` for `a^2`. Any square root serves where only its square
    matters."""
    outside, inside = sp.S.One, sp.S.One
    for factor in sp.Mul.make_args(sp.factor(value)):
        base, exponent = factor.as_base_exp()
        if exponent.is_Integer and exponent % 2 == 0:
            outside *= base ** (exponent // 2)
        else:
            inside *= factor
    return outside * sp.sqrt(inside)


def _real_form(h: sp.Poly, q: sp.Poly, x: sp.Basic, t: sp.Basic) -> sp.Basic | None:
    """The sum of `a*LN(h(a, x))` over the roots `a` of `q`, written real, or None.

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
    """The roots of `q` as the real ones and one of each conjugate pair, or None.

    None where `roots` finds fewer than the degree, and where a root has no
    conjugate among the others and is not known to be real: the quartic formula
    writes the roots of `x^4 + x + 1` through the cube root of a complex number,
    each root's conjugate is some other expression, and a sum over them one by one
    would carry complex coefficients into a real answer.
    """
    found = roots(q)
    if sum(found.values()) != q.degree():
        return None
    reals: dict = {}
    complexes: dict = {}
    remaining = list(found)
    while remaining:
        r = remaining.pop()
        conjugate = r.conjugate()
        if r != conjugate and conjugate in found:
            remaining.remove(conjugate)
            r_re, r_im = r.as_real_imag()
            _, r_im = ordered([r_im, -r_im])
            complexes[(r_re, r_im)] = found[r]
        elif r.is_extended_real:
            reals[r] = found[r]
        else:
            return None
    return reals, complexes
