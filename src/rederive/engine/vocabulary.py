"""The sympy heads the notation has no name for, written in the ones it has.

Sympy answers in functions Derive has no word for - a logarithmic integral, a
Fresnel integral, a number marked with the sheet of a logarithm it lies on - and each
is written here in the notation's own functions wherever it is exactly that.
`in_vocabulary` is the whole of it.
"""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp

__all__ = ["INVERSE_HYPERBOLIC", "POLAR", "in_vocabulary"]


def _as_exponential_integral(order: sp.Basic, argument: sp.Basic) -> sp.Basic:
    """`expint(n, z)` over `EI`, where its order lets it be.

    Sympy's expansion takes an integer or half-integer order down to `expint(1, z)`
    and elementary functions; that one is `-EI(-z)` where `z` is positive, and only
    there: for a negative `z` the two differ by `pi*#i`. So it is written over `EI`
    only where sympy proves `z` positive, which is where the integrals that produce
    it converge. Anywhere else, and at any other order, it stays as it is, and an
    answer holding it is one the notation cannot write.
    """
    expanded = sp.expand_func(sp.expint(order, argument))
    return expanded.replace(
        lambda e: isinstance(e, sp.expint) and e.args[0] == 1 and e.args[1].is_positive,
        lambda e: -sp.Ei(-e.args[1]),
    )


def _as_log_gamma(argument: sp.Basic) -> sp.Basic:
    """`loggamma(x)` as `LN(GAMMA(x))`, where `x` is positive.

    Sympy's `loggamma` is the branch of the logarithm that is analytic off the
    negative real axis, which is the principal logarithm of the gamma function only
    for a positive argument: at `-1/2` the one is `LN(2*SQRT(pi)) - pi*#i` and the
    other has `+ pi*#i`. Where sympy cannot prove the argument positive the head
    stays, and an answer holding it is one the notation cannot write.
    """
    if argument.is_positive:
        return sp.log(sp.gamma(argument))
    return sp.loggamma(argument)


def _as_gamma_ratio(base: sp.Basic, count: sp.Basic) -> sp.Basic:
    """The rising factorial `RisingFactorial(x, k)` as the gammas it is made of,
    `GAMMA(x + k)/GAMMA(x)`, whatever the sign of `x`.

    The two agree everywhere but at the poles of the gamma function, where the
    quotient is undefined and its limit is the rising factorial's value: at `x = -2`
    the quotient `GAMMA(x + 1)/GAMMA(x)` is `x` wherever it is defined, and `-2` in
    the limit, which is what `RisingFactorial(-2, 1)` is. That is the removable point
    of `sinc` again, and the quotient is the spelling Derive gives it: the original
    answers `PRODUCT(k + a, k, 1, n)` as `(a + n)!/a!` with nothing declared about
    `a`. Sympy's own rewrite is a different expression for a negative `x`, and a
    `Piecewise` of the two where the sign is open, which is why it is not used.
    """
    return sp.gamma(base + count) / sp.gamma(base)


#: The sympy heads the notation has no name for that are spellings of functions it
#: does have, each written in those. Sympy answers in them freely: `INT(1/LN(x), x)`
#: is its logarithmic integral `li(x)`, which is `EI(LN(x))`. The offset logarithmic
#: integral `Li` is written over `li` and goes on to `EI` from there, which is why it
#: comes first. `ceiling` is the floor of the negation, `beta` and the rising
#: factorial are the ratios of gammas they abbreviate, and `loggamma` is the logarithm
#: of the gamma function. Each is written so only where it is exactly that, and the
#: exponential integral and `loggamma` are that only on part of their domain, so those
#: are written by functions that ask sympy whether the argument lies in it. `sinc` is
#: the quotient it abbreviates, `SIN(x)/x`, which is the spelling Derive would give it:
#: the two differ only at the removable point 0, where the quotient is undefined as any
#: quotient is at a zero denominator. The ratio of gammas differs from the rising
#: factorial only at removable points in the same way, the poles of the gamma
#: function, and is written everywhere by `_as_gamma_ratio`. The Fresnel integrals are
#: written by `_as_error_functions`. `POLYLOG(1, z)` is `-LN(1 - z)` everywhere, cut
#: along the same ray from 1, and is the one elementary polylogarithm sympy does not
#: write out itself.
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
    """Every Fresnel integral in `expression` as the error functions it is made of.

    Sympy answers `INT(COSH(x^2), x)` as a cosine integral of `x*#e^(#i*pi/4)`, so
    the error functions its rewrite reaches are of `x*(1 - #i)*#e^(#i*pi/4)`, which
    is plain `x` once the exponential is written as the complex number it is. So
    every exponential of an imaginary number in the expression is written that way,
    and the arguments multiplied out. An `ERF` of an imaginary argument is then the
    `ERFI` of a real one, by `ERF(#i*u) = #i*ERFI(u)`: `INT(COSH(x^2), x)` is
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
    """`ERF(u)` with `u` multiplied out where that leaves it real, and as
    `#i*ERFI(u/#i)` where `u/#i` is real. A complex argument is left as it is, which
    keeps a factor such as `1 + #i` whole."""
    expanded = sp.expand(argument)
    turned = sp.expand(argument / sp.I)
    if turned.is_extended_real and not turned.is_zero:
        return sp.I * sp.erfi(turned)
    return sp.erf(expanded if expanded.is_extended_real else argument)


#: The inverse hyperbolic functions, which Derive writes as the logarithms they are:
#: `ASINH(x)` authored on its own comes back `LN(SQRT(x^2 + 1) + x)`. No antiderivative
#: is printed over one, and `INT(SQRT(x^2 + a^2), x)`, whose cases are an `ASINH` where
#: `a` is nonzero and that same logarithm elsewhere, is printed over the logarithm
#: however much shorter the arc is.
INVERSE_HYPERBOLIC = (sp.asinh, sp.acosh, sp.atanh, sp.acoth, sp.asech, sp.acsch)


#: The marks sympy's G-function integration leaves on a number that says which sheet
#: of a logarithm it lies on.
POLAR = (
    sp.exp_polar,
    sp.polar_lift,
    sp.periodic_argument,
    sp.principal_branch,
)


def in_vocabulary(expression: sp.Basic) -> sp.Basic:
    """`expression` with the sympy heads that spell the notation's functions written
    in them.

    A polar number is taken off its sheet by `unpolarify`, which does it wherever
    the value does not depend on the sheet - `exp_polar(I*pi)` is -1 and
    `polar_lift(x)` is `x` - and leaves the mark where it does: `Ei(x*exp_polar(I*pi))`
    is `EI(-x) + pi*#i`, and writing it as `EI(-x)` would lose the `pi*#i`. Such a
    mark is left for `is_written` to decline.
    """
    if expression.has(*POLAR):
        expression = sp.unpolarify(expression)
    if expression.has(*_FRESNEL):
        expression = _as_error_functions(expression)
    for head, written in _SPELLED_OVER.items():
        if expression.has(head):
            expression = expression.replace(head, written)
    return expression
