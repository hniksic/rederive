"""Definite integrals by rule, where no antiderivative gives them.

Each rule recognizes an integrand and an interval, proves the conditions under which its
formula holds, and gives the value; otherwise it gives None and the integral goes on to
the general methods. `by_a_definite_rule` tries them in turn.
"""

from __future__ import annotations

import sympy as sp

# Import the module rather than its names, since `indefinite` imports from this one.
from rederive.engine.integration import indefinite
from rederive.engine.rewriting import attempt

__all__ = ["by_a_definite_rule"]


def by_a_definite_rule(head: sp.Basic) -> sp.Basic | None:
    """The value of a definite integral by one of the rules of this module, or None."""
    if len(head.limits) != 1 or len(head.limits[0]) != 3:
        return None
    variable, low, high = head.limits[0]
    if not head.function.has(variable):
        return None
    for rule in (_gaussian, _bose, _reciprocal_binomial, _log_sine, _by_scaling):
        value = rule(head.function, variable, low, high)
        if value is not None:
            return value
    return None


def _exponential_split(
    integrand: sp.Basic, variable: sp.Basic
) -> tuple[sp.Basic, sp.Basic]:
    """`integrand` as `P*#e^Q`: the factor outside the exponentials and the sum of their
    exponents."""
    outside, exponent = [], sp.S.Zero
    for factor in sp.Mul.make_args(integrand):
        if isinstance(factor, sp.exp):
            exponent += factor.args[0]
        elif factor.is_Pow and factor.base is sp.E:
            exponent += factor.exp
        else:
            outside.append(factor)
    return sp.Mul(*outside), exponent


def _positive_where_defined(value: sp.Basic, integrand: sp.Basic) -> bool:
    """Whether `value` is positive for every value of its variables at which the
    integrand is defined.

    A real variable at which zero leaves the integrand undefined, as `s` in
    `#e^(-x^2/(2*s^2))`, is taken to be nonzero, so `1/(2*s^2)` is positive.
    """
    if value.is_positive:
        return True
    stand_ins = {}
    for symbol in value.free_symbols:
        if not symbol.is_real:
            continue
        at_zero = attempt(integrand, lambda f: f.subs(symbol, 0))
        if at_zero is not None and at_zero.has(sp.zoo, sp.nan):
            stand_ins[symbol] = sp.Dummy(symbol.name, real=True, nonzero=True)
    if not stand_ins:
        return False
    written = attempt(value, lambda v: v.xreplace(stand_ins))
    return written is not None and written.is_positive is True


def _gaussian(
    integrand: sp.Basic, variable: sp.Basic, low: sp.Basic, high: sp.Basic
) -> sp.Basic | None:
    """`INT(p(x)*#e^(-a*x^2 + b*x + c), x, -inf, inf)` for a polynomial `p` and `a`
    positive: with `x = y + b/(2*a)` the exponential is even in `y`, and its moments are
    `GAMMA((k + 1)/2)/a^((k + 1)/2)` for even `k`, 0 for odd."""
    if (low, high) != (-sp.oo, sp.oo):
        return None
    outside, exponent = _exponential_split(integrand, variable)
    try:
        quadratic = sp.Poly(sp.expand(exponent), variable)
        polynomial = sp.Poly(sp.expand(outside), variable)
    except sp.PolynomialError:
        return None
    if quadratic.degree() != 2:
        return None
    a, b, c = [-quadratic.coeff_monomial(variable**2)] + [
        quadratic.coeff_monomial(m) for m in (variable, 1)
    ]
    if not _positive_where_defined(a, integrand):
        return None
    shift = b / (2 * a)
    total = sp.S.Zero
    for (power,), coefficient in polynomial.terms():
        for k in range(0, power + 1, 2):
            moment = sp.gamma(sp.Rational(k + 1, 2)) / a ** sp.Rational(k + 1, 2)
            total += coefficient * sp.binomial(power, k) * shift ** (power - k) * moment
    return sp.exp(c + b**2 / (4 * a)) * total


def _bose(
    integrand: sp.Basic, variable: sp.Basic, low: sp.Basic, high: sp.Basic
) -> sp.Basic | None:
    """`INT(x^p/(c*#e^(a*x) - c), x, 0, inf)` = `p!*ZETA(p + 1)/(a^(p + 1)*c)` for
    `a > 0` and `p > 0`, from the geometric series of `1/(#e^(a*x) - 1)`."""
    if (low, high) != (0, sp.oo):
        return None
    together = attempt(integrand, sp.together)
    if together is None:
        return None
    numerator, denominator = sp.fraction(together)
    terms = sp.Add.make_args(sp.expand(denominator))
    if len(terms) != 2:
        return None
    constant = [term for term in terms if not term.has(variable)]
    if len(constant) != 1:
        return None
    scale = -constant[0]
    (rest,) = [term for term in terms if term.has(variable)]
    exponential = attempt(rest, lambda r: sp.powsimp(r / scale))
    if exponential is None:
        return None
    if exponential.is_Pow and exponential.base is sp.E:
        rate = exponential.exp / variable
    elif isinstance(exponential, sp.exp):
        rate = exponential.args[0] / variable
    else:
        return None
    rate = sp.cancel(rate)
    if rate.has(variable) or rate.is_positive is not True:
        return None
    factor, power = sp.S.One, sp.S.Zero
    for part in sp.Mul.make_args(numerator):
        if part == variable:
            power += 1
        elif part.is_Pow and part.base == variable and not part.exp.has(variable):
            power += part.exp
        elif not part.has(variable):
            factor *= part
        else:
            return None
    if power.is_positive is not True:
        return None
    moment = sp.gamma(power + 1) * sp.zeta(power + 1)
    return factor * moment / (rate ** (power + 1) * scale)


def _reciprocal_binomial(
    integrand: sp.Basic, variable: sp.Basic, low: sp.Basic, high: sp.Basic
) -> sp.Basic | None:
    """`INT(x^m/(a*x^n + b), x, 0, inf)` =
    `pi*(b/a)^((m + 1)/n)/(n*b*SIN((m + 1)*pi/n))` for `a` and `b` of one sign and
    `-1 < m < n - 1`, by the beta integral."""
    if (low, high) != (0, sp.oo):
        return None
    together = attempt(integrand, sp.together)
    if together is None:
        return None
    top, denominator = sp.fraction(together)
    numerator, raised = sp.S.One, sp.S.Zero
    for part in sp.Mul.make_args(top):
        if part == variable:
            raised += 1
        elif part.is_Pow and part.base == variable and part.exp.is_Rational:
            raised += part.exp
        elif not part.has(variable):
            numerator *= part
        else:
            return None
    terms = sp.Add.make_args(sp.expand(denominator))
    if len(terms) != 2:
        return None
    constant = [term for term in terms if not term.has(variable)]
    if len(constant) != 1:
        return None
    b = constant[0]
    (rest,) = [term for term in terms if term.has(variable)]
    a, power = sp.S.One, None
    for part in sp.Mul.make_args(rest):
        if part.is_Pow and part.base == variable and part.exp.is_number:
            power = part.exp
        elif not part.has(variable):
            a *= part
        else:
            return None
    if power is None or not power.is_rational or not -1 < raised < power - 1:
        return None
    if a.is_negative and b.is_negative:
        a, b, numerator = -a, -b, -numerator
    if not (a.is_positive and b.is_positive):
        return None
    share = (raised + 1) / power
    return numerator * sp.pi * (b / a) ** share / (power * b * sp.sin(sp.pi * share))


def _log_sine(
    integrand: sp.Basic, variable: sp.Basic, low: sp.Basic, high: sp.Basic
) -> sp.Basic | None:
    """The log-sine integrals: `INT(LN(SIN(x)), x, 0, pi/2)` and the same of `COS(x)` are
    `-pi*LN(2)/2`, and `INT(LN(SIN(x)), x, 0, pi)` is `-pi*LN(2)`."""
    if low != 0 or not isinstance(integrand, sp.log):
        return None
    inner = integrand.args[0]
    if inner == sp.sin(variable) and high in (sp.pi / 2, sp.pi):
        return -high * sp.log(2)
    if inner == sp.cos(variable) and high == sp.pi / 2:
        return -high * sp.log(2)
    return None


def _by_scaling(
    integrand: sp.Basic, variable: sp.Basic, low: sp.Basic, high: sp.Basic
) -> sp.Basic | None:
    """An integral between multiples of a parameter `r`, by `x = r*u`, if that makes
    the integrand a power of `r` times a function of `u` alone.

    `INT(x*SQRT((r - x)/r)/SQRT((x + r)/r), x, 0, r)` is `r^2` times the same integral
    of `u*SQRT(1 - u)/SQRT(u + 1)` from 0 to 1, whatever the sign of `r`. The radicands
    are cancelled, which holds wherever the integrand is defined.
    """
    ends = [end for end in (low, high) if end != 0]
    if not ends:
        return None
    scale = None
    for end in ends:
        multiple, rest = end.as_coeff_Mul()
        if not (multiple.is_Rational and rest.is_Symbol and rest != variable):
            return None
        if scale is not None and rest != scale:
            return None
        scale = rest
    if not (integrand.free_symbols <= {variable, scale}) or scale not in (
        integrand.free_symbols
    ):
        return None
    unit = sp.Dummy("u", real=True)
    written = attempt(integrand, lambda f: f.subs(variable, scale * unit) * scale)
    if written is None:
        return None
    written = attempt(
        written,
        lambda w: w.replace(
            lambda e: e.is_Pow and e.base.has(scale) and e.base.has(unit),
            lambda e: sp.cancel(e.base) ** e.exp,
        ),
    )
    if written is None:
        return None
    shape = attempt(written, lambda w: w.subs(scale, 1))
    if shape is None or shape == 0 or shape.has(sp.zoo, sp.nan):
        return None
    ratio = attempt(written, lambda w: sp.cancel(sp.together(w / shape)))
    if ratio is None or ratio.has(unit):
        return None
    inner = indefinite.integral(sp.Integral(shape, (unit, low / scale, high / scale)))
    if inner.has(sp.Integral) or inner.is_finite is not True:
        return None
    return ratio * inner
