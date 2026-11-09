"""Rewriting sympy functions Derive lacks in terms of ones it has.

Sympy answers with functions Derive has no name for (the logarithmic integral, Fresnel
integrals, numbers marked with a logarithm's sheet). `in_vocabulary` rewrites each in
Derive's functions where that is exact.
"""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp

__all__ = ["INVERSE_HYPERBOLIC", "POLAR", "in_vocabulary"]


def _as_exponential_integral(order: sp.Basic, argument: sp.Basic) -> sp.Basic:
    """`expint(n, z)` in terms of `EI`, where possible.

    Sympy expands integer or half-integer orders to `expint(1, z)` and elementary
    functions. `expint(1, z)` equals `-EI(-z)` only for positive `z` (for negative `z`
    they differ by `pi*#i`), so it is rewritten only where sympy proves `z` positive,
    which is where the integrals producing it converge. Otherwise it is left alone, and
    the answer can't be written in Derive notation.
    """
    expanded = sp.expand_func(sp.expint(order, argument))
    return expanded.replace(
        lambda e: isinstance(e, sp.expint) and e.args[0] == 1 and e.args[1].is_positive,
        lambda e: -sp.Ei(-e.args[1]),
    )


def _as_log_gamma(argument: sp.Basic) -> sp.Basic:
    """`loggamma(x)` as `LN(GAMMA(x))`, where `x` is positive.

    Sympy's `loggamma` is analytic off the negative real axis and equals the principal
    `LN(GAMMA(x))` only for positive `x`: at `-1/2` one is `LN(2*SQRT(pi)) - pi*#i` and
    the other has `+ pi*#i`. If sympy can't prove `x` positive, `loggamma` stays, and
    the answer can't be written in Derive notation.
    """
    if argument.is_positive:
        return sp.log(sp.gamma(argument))
    return sp.loggamma(argument)


def _as_gamma_ratio(base: sp.Basic, count: sp.Basic) -> sp.Basic:
    """`RisingFactorial(x, k)` as `GAMMA(x + k)/GAMMA(x)`, whatever the sign of `x`.

    They agree except at the gamma function's poles, where the quotient is undefined
    and its limit is the rising factorial's value (a removable point, as for `sinc`).
    Derive uses this form too, answering `PRODUCT(k + a, k, 1, n)` as `(a + n)!/a!`
    with nothing declared about `a`. Sympy's own rewrite differs for negative `x` and
    becomes a `Piecewise` when the sign is unknown, so it isn't used.
    """
    return sp.gamma(base + count) / sp.gamma(base)


#: Sympy functions Derive lacks, mapped to rewrites in functions it has. For example
#: `INT(1/LN(x), x)` gives `li(x)`, which is `EI(LN(x))`. `Li` is rewritten via `li`,
#: so it comes first. `ceiling(x)` is `-floor(-x)`, `beta` and the rising factorial
#: are gamma ratios, and `loggamma` is the log of gamma. The exponential integral and
#: `loggamma` are exact only on part of their domain, so their rewrites ask sympy
#: whether the argument lies there. `sinc` becomes `SIN(x)/x`, as Derive writes it;
#: they differ only at the removable point 0. The gamma ratio likewise differs from the
#: rising factorial only at removable points (the gamma poles). Fresnel integrals are
#: handled by `_as_error_functions`. `POLYLOG(1, z)` is `-LN(1 - z)` everywhere, with
#: the same branch cut, and is the one elementary polylogarithm sympy doesn't expand.
_SPELLED_OVER: dict[type, Callable[..., sp.Basic]] = {
    sp.Li: lambda x: sp.Ei(sp.log(x)) - sp.Ei(sp.log(2)),
    sp.li: lambda x: sp.Ei(sp.log(x)),
    sp.expint: _as_exponential_integral,
    sp.ceiling: lambda x: -sp.floor(-x),
    sp.beta: lambda *args: sp.beta(*args).rewrite(sp.gamma),
    sp.RisingFactorial: _as_gamma_ratio,
    sp.loggamma: _as_log_gamma,
    sp.sinc: lambda x: sp.sin(x) / x,
    sp.polylog: lambda s, z: -sp.log(1 - z) if s == 1 else sp.polylog(s, z),
}

#: The Fresnel integrals, which are `ERF`s of complex arguments.
_FRESNEL = (sp.fresnels, sp.fresnelc)


def _as_error_functions(expression: sp.Basic) -> sp.Basic:
    """Every Fresnel integral in `expression` rewritten as error functions.

    Sympy answers `INT(COSH(x^2), x)` with a Fresnel cosine integral of
    `x*#e^(#i*pi/4)`, whose `erf` rewrite has argument `x*(1 - #i)*#e^(#i*pi/4)`. That
    is plain `x` once the exponential is written as a complex number, so numeric
    exponentials are expanded and the arguments multiplied out. An `ERF` of an
    imaginary argument then becomes `ERFI` by `ERF(#i*u) = #i*ERFI(u)`, giving
    `SQRT(pi)*(ERF(x) + ERFI(x))/4`.
    """
    for head in _FRESNEL:
        expression = expression.replace(head, lambda z, h=head: h(z).rewrite(sp.erf))
    expression = expression.replace(
        lambda e: isinstance(e, sp.exp) and e.args[0].is_number,
        sp.expand_complex,
    )
    return expression.replace(sp.erf, _erf_expanded)


def _erf_expanded(argument: sp.Basic) -> sp.Basic:
    """`ERF(u)` with `u` expanded if that makes it real, or `#i*ERFI(u/#i)` if `u/#i`
    is real. A complex argument is left alone, keeping a factor like `1 + #i` whole."""
    expanded = sp.expand(argument)
    turned = sp.expand(argument / sp.I)
    if turned.is_extended_real and not turned.is_zero:
        return sp.I * sp.erfi(turned)
    return sp.erf(expanded if expanded.is_extended_real else argument)


#: Inverse hyperbolic functions, which Derive writes as logarithms: `ASINH(x)` simplifies
#: to `LN(SQRT(x^2 + 1) + x)`. Antiderivatives are never printed with them, so
#: `INT(SQRT(x^2 + a^2), x)` uses the logarithm even where `ASINH` would be shorter.
INVERSE_HYPERBOLIC = (sp.asinh, sp.acosh, sp.atanh, sp.acoth, sp.asech, sp.acsch)


#: Markers sympy's G-function integration leaves to say which sheet of a logarithm a
#: number lies on.
POLAR = (
    sp.exp_polar,
    sp.polar_lift,
    sp.periodic_argument,
    sp.principal_branch,
)


def in_vocabulary(expression: sp.Basic) -> sp.Basic:
    """`expression` with sympy functions rewritten in Derive's, where exact.

    `unpolarify` removes polar markers where the value doesn't depend on the sheet
    (`exp_polar(I*pi)` is -1, `polar_lift(x)` is `x`) and keeps them where it does:
    `Ei(x*exp_polar(I*pi))` is `EI(-x) + pi*#i`, and `EI(-x)` would lose the `pi*#i`.
    `is_written` then rejects any remaining marker.
    """
    if expression.has(*POLAR):
        expression = sp.unpolarify(expression)
    if expression.has(*_FRESNEL):
        expression = _as_error_functions(expression)
    for head, written in _SPELLED_OVER.items():
        if expression.has(head):
            expression = expression.replace(head, written)
    return expression
