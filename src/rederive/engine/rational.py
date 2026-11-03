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
"""

from __future__ import annotations

import sympy as sp
from sympy.core.sorting import ordered
from sympy.integrals.rationaltools import log_to_atan, ratint_logpart, ratint_ratpart
from sympy.polys.polyroots import roots

__all__ = ["rational_antiderivative"]


def rational_antiderivative(integrand: sp.Basic, variable: sp.Basic) -> sp.Basic | None:
    """An antiderivative of a rational function of the variable, or None.

    The steps are `ratint`'s own, over its public halves; only the writing of the
    logarithmic part in real form is this module's. None where the integrand is no
    rational function of the variable, or where sympy's polynomial arithmetic
    refuses it.
    """
    if integrand.is_rational_function(variable) is not True:
        return None
    try:
        return _antiderivative(integrand, variable)
    except Exception:
        return None


def _antiderivative(integrand: sp.Basic, variable: sp.Basic) -> sp.Basic:
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
        t = sp.Dummy("t")
        for h, q in ratint_logpart(numerator, denominator, variable, t):
            _, h = h.primitive()
            real = _real_form(h, q, variable, t)
            if real is None:
                real = sp.RootSum(q, sp.Lambda(t, t * sp.log(h.as_expr())), quadratic=True)
            result += real
    return coefficient * result


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
