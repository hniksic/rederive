"""Antiderivatives of rational functions of the sine and cosine of one linear argument,
by the substitutions of integral tables.

`R(SIN(u), COS(u))`, with `u = a*x + b`, becomes rational in `t` under one of four
substitutions, tried in order:

- `t = COS(u)` where `R` is odd in the sine, as in `SIN(u)/(b + c*COS(u))^2`;
- `t = SIN(u)` where `R` is odd in the cosine, as in `1/(COS(u)*(1 + SIN(u)))`;
- `t = TAN(u)` where `R` is unchanged by negating both, as in `1/(b^2 + c^2*SIN(u)^2)`,
  giving a quadratic in `t` where the half angle would give a quartic;
- `t = TAN(u/2)` otherwise.

The rational function is integrated by `engine.rational`, which writes a parametric
quadratic as one logarithm and one arctangent without splitting into cases on the
parameters. That is where the speed comes from: sympy integrates `1/(b + c*SIN(a*x))^2`
in a minute with a page of cases, this in a fraction of a second.

`t = TAN(w)` jumps from `+inf` to `-inf` where `w` crosses an odd multiple of `pi/2`,
and so does an antiderivative in `t`, even where the integrand is regular. Two fixes:

- The logarithm of a degree-`n` polynomial in `t` is written as the logarithm of the
  homogeneous polynomial in `SIN(w)` and `COS(w)` equal to it times `COS(w)^n`, minus
  `n*LN(COS(w))`. Where the integrand is regular at `t = inf` the integral converges
  there and the `LN(COS(w))` terms cancel. Over the half angle, a degree-2 form is
  written in the whole angle (`LN(b*TAN(u/2)^2 + 2*c*TAN(u/2) + b)` is `LN(b +
  c*SIN(u))`), and a degree-1 form as half the logarithm of its square. Each differs
  from the original by a constant between the zeros of its argument, which are poles
  of the integrand.
- An arctangent of an odd-degree polynomial in `t` jumps by `pi` there, so a multiple
  of `pi*FLOOR(w/pi + 1/2)` is added to jump back, as sympy does.

If the integrand itself has a pole at those points, the logarithms don't cancel and
are left in `TAN(w)`. The rational part, if finite at `t = inf`, is written in the
sine and cosine of `u`: `-COS(u)/(1 + SIN(u))` for `-2/(1 + TAN(u/2))`, which is
defined at `u = pi` where the tangent isn't.

Only denominators that are sums are handled; products of powers of sine and cosine,
like `1/(SIN(u)*COS(u)^2)`, are integrated equally well elsewhere.

Functions of small multiples of the argument are first expanded, `SIN(2*u)` as
`2*SIN(u)*COS(u)`. A logarithm negative at zero is written over its negation, and two
with opposite coefficients and parametric arguments as the logarithm of their quotient,
so the integral from zero (towards which the answer is regularized) needs no constant
subtracted unnecessarily.
"""

from __future__ import annotations

from functools import reduce

import sympy as sp

from .rational import rational_antiderivative

__all__ = ["trigonometric_antiderivative"]

_TRIGONOMETRIC = (sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc)

# The largest multiple of the argument that is expanded into functions of the argument.
_MULTIPLES = 4


def trigonometric_antiderivative(
    integrand: sp.Basic, variable: sp.Basic
) -> sp.Basic | None:
    """An antiderivative of a rational function of the sine and cosine of one argument
    linear in the variable, or None if the integrand isn't one or the rational
    function in `t` has no closed form."""
    try:
        return _antiderivative(integrand, variable)
    except Exception:
        return None


def _antiderivative(integrand: sp.Basic, variable: sp.Basic) -> sp.Basic | None:
    found = _argument(integrand, variable)
    if found is None:
        return None
    u, slope, integrand = found
    s, c, t = sp.Dummy("s"), sp.Dummy("c"), sp.Dummy("t")
    written = integrand.xreplace(
        {
            atom: {
                sp.sin: s,
                sp.cos: c,
                sp.tan: s / c,
                sp.cot: c / s,
                sp.sec: 1 / c,
                sp.csc: 1 / s,
            }[atom.func]
            for atom in integrand.atoms(*_TRIGONOMETRIC)
        }
    )
    if written.has(variable) or written.is_rational_function(s, c) is not True:
        return None
    written = sp.cancel(written)
    if not _summed(sp.denom(written), s, c):
        return None
    linear = _over_a_linear_form(written, s, c, u)
    if linear is not None:
        return _positive_at_zero(linear / slope, variable)
    by_sine = written.xreplace({s: -s})
    by_cosine = written.xreplace({c: -c})
    if sp.cancel(by_sine + written) == 0:
        # dt = -SIN(u)*du
        rational = _in_the_square(-written / s, s, 1 - t**2)
        back, half = sp.cos(u), None
        rational = rational and rational.xreplace({c: t})
    elif sp.cancel(by_cosine + written) == 0:
        # dt = COS(u)*du
        rational = _in_the_square(written / c, c, 1 - t**2)
        back, half = sp.sin(u), None
        rational = rational and rational.xreplace({s: t})
    elif sp.cancel(by_sine.xreplace({c: -c}) - written) == 0:
        # SIN(u) = t*COS(u), COS(u)^2 = 1/(1 + t^2), du = dt/(1 + t^2)
        rational = _in_the_square(written.xreplace({s: t * c}), c, 1 / (1 + t**2))
        rational = rational and rational / (1 + t**2)
        back, half = sp.tan(u), False
    else:
        # SIN(u) = 2*t/(1 + t^2), COS(u) = (1 - t^2)/(1 + t^2), du = 2*dt/(1 + t^2)
        rational = written.xreplace({s: 2 * t / (1 + t**2), c: (1 - t**2) / (1 + t**2)})
        rational = 2 * rational / (1 + t**2)
        back, half = sp.tan(u / 2), True
    if rational is None:
        return None
    antiderivative = rational_antiderivative(sp.cancel(rational), t, binomials=True)
    if antiderivative is None or antiderivative.has(sp.RootSum, sp.I):
        return None
    antiderivative = antiderivative.xreplace(
        {
            root: _without_squares(root.base, root.exp)
            for root in antiderivative.atoms(sp.Pow)
            if root.exp.is_Rational and root.exp.q == 2
        }
    )
    if half is None:
        written = antiderivative.replace(
            sp.log, lambda argument: _logarithm(_positive_on_the_circle(argument, t))
        ).xreplace({t: back})
    else:
        written = _through_the_poles(antiderivative, t, back.args[0], u, half)
    return _positive_at_zero(_gathered(written / slope), variable)


def _positive_at_zero(value: sp.Basic, variable: sp.Basic) -> sp.Basic:
    """The value with its logarithms `_paired`, and each logarithm of a function of the
    variable negative at zero negated inside (a constant `#i*pi` away):
    `LN(COS(x) - SIN(x))` for `LN(SIN(x) - COS(x))`, real near zero, where the
    integral from zero is taken."""
    value = _paired(value, variable)

    def positive(argument: sp.Basic) -> sp.Basic:
        at_zero = argument.subs(variable, 0)
        if at_zero.is_number and at_zero.is_negative is True:
            return sp.log(-argument)
        return sp.log(argument)

    return value.replace(sp.log, positive)


def _over_a_linear_form(
    written: sp.Basic, s: sp.Basic, c: sp.Basic, u: sp.Basic
) -> sp.Basic | None:
    """The antiderivative in `u` of `k/(b*SIN(u) + c*COS(u))`, or None if the rational
    function isn't of that form.

    With `r = SQRT(b^2 + c^2)` the denominator is `r*SIN(u + p)`, where `COS(p) = b/r`
    and `SIN(p) = c/r`, and the integral is `LN(TAN((u + p)/2))/r`, which is
    `LN((1 - COS(u + p))/SIN(u + p))/r`, or `LN((r - b*COS(u) + c*SIN(u))/(b*SIN(u) +
    c*COS(u)))/r`. The half angle would give an arctangent of an imaginary argument
    (its quadratic's discriminant is `-4*r^2`), jumping by an imaginary multiple of
    `pi` where the tangent jumps.
    """
    numerator, denominator = sp.fraction(written)
    if numerator.has(s, c):
        return None
    poly = sp.Poly(denominator, s, c)
    b, k = poly.coeff_monomial(s), poly.coeff_monomial(c)
    if poly.total_degree() != 1 or len(poly.terms()) != 2 or b == 0 or k == 0:
        return None
    root = _without_squares(sp.expand(b**2 + k**2), sp.S.Half)
    sine, cosine = sp.sin(u), sp.cos(u)
    quotient = (root - b * cosine + k * sine) / (b * sine + k * cosine)
    return numerator * sp.log(quotient) / root


def _paired(value: sp.Basic, variable: sp.Basic) -> sp.Basic:
    """The sum with each pair of logarithms that have opposite coefficients and
    arguments parametric at zero merged: `k*LN(u) - k*LN(v)` becomes `k*LN(u/v)`. They
    differ by a multiple of `2*pi*#i`, constant wherever `u` and `v` are finite and
    nonzero.

    `LN(b*COS(x) + c*SIN(x)) - LN(b*COS(x) - c*SIN(x))` is then `LN(1)` at zero rather
    than two `LN(b)` terms for the integral from zero to subtract. A sum with other
    terms in the variable is left alone, since there the quotient is no shorter."""
    terms = list(sp.Add.make_args(sp.expand_mul(value, deep=False)))
    logarithms = []
    for i, term in enumerate(terms):
        k, rest = term.as_independent(variable, as_Add=False)
        if isinstance(rest, sp.log):
            logarithms.append((i, k, rest.args[0]))
        elif rest.has(variable):
            return value
    used = set()
    for i, k, u in logarithms:
        for j, m, v in logarithms:
            if i in used or j in used or i == j or sp.cancel(k + m) != 0:
                continue
            if all(w.subs(variable, 0).is_number for w in (u, v)):
                continue
            terms[i], terms[j] = k * sp.log(sp.cancel(u / v)), sp.S.Zero
            used.update((i, j))
    return sp.Add(*terms) if used else value


def _positive_on_the_circle(argument: sp.Basic, t: sp.Basic) -> sp.Basic:
    """The argument of a logarithm in `t = SIN(u)` or `COS(u)`, negated if it is
    negative for all `t` between -1 and 1, like `t - 1`: `LN(1 - SIN(u))` is real
    where `LN(SIN(u) - 1)` is `#i*pi` off."""
    if argument.free_symbols != {t} or argument.is_polynomial(t) is not True:
        return argument
    poly = sp.Poly(argument, t)
    inside = any(-1 < root < 1 for root in poly.real_roots())
    if not inside and poly.eval(0) < 0:
        return -argument
    return argument


def _gathered(value: sp.Basic) -> sp.Basic:
    """The value with each product of a radicand's factors written as the radicand:
    `(b - c)*(b + c)/SQRT(b^2 - c^2)` as `SQRT(b^2 - c^2)`."""
    radicands = {
        power.base
        for power in value.atoms(sp.Pow)
        if power.exp.is_Rational and power.exp.q == 2 and isinstance(power.base, sp.Add)
    }
    for radicand in radicands:
        _, factors = sp.factor_list(radicand)
        if len(factors) < 2 or any(multiplicity > 1 for _, multiplicity in factors):
            continue
        product = sp.Mul(*(factor for factor, _ in factors))
        scale = sp.cancel(radicand / product)
        bases = [factor for factor, _ in factors]

        def gather(m: sp.Mul) -> sp.Basic:
            powers = m.as_powers_dict()
            exponents = {powers.get(base, 0) for base in bases}
            if len(exponents) != 1 or 0 in exponents:
                return m
            (exponent,) = exponents
            rest = m / product**exponent
            return rest * (radicand / scale) ** exponent

        value = value.replace(lambda e: isinstance(e, sp.Mul), gather)
    return value


def _without_squares(base: sp.Basic, exponent: sp.Basic) -> sp.Basic:
    """The power of the radicand, factored if that exposes a square (`4*b^2*(b^2 +
    c^2)`), otherwise with only its content taken out: `2*SQRT(b^2 - c^2)` for
    `SQRT(4*b^2 - 4*c^2)`."""
    _, factors = sp.factor_list(base)
    if any(multiplicity > 1 for _, multiplicity in factors):
        return sp.factor(base) ** exponent
    content, primitive = sp.factor_terms(base).as_content_primitive()
    return content**exponent * primitive**exponent


def _summed(denominator: sp.Basic, s: sp.Basic, c: sp.Basic) -> bool:
    """Whether the denominator is more than a product of powers of the sine and cosine,
    which other methods integrate equally well."""
    return len(sp.Poly(denominator, s, c).terms()) > 1


def _argument(
    integrand: sp.Basic, variable: sp.Basic
) -> tuple[sp.Basic, sp.Basic, sp.Basic] | None:
    """The argument linear in the variable of which every trigonometric function's
    argument is a small integer multiple, its slope, and the integrand rewritten in
    functions of that argument: `SIN(2*a*x)/(1 + COS(a*x))` becomes
    `2*SIN(a*x)*COS(a*x)/(1 + COS(a*x))`. None if there is no such argument."""
    arguments = sorted(
        {atom.args[0] for atom in integrand.atoms(*_TRIGONOMETRIC)},
        key=sp.default_sort_key,
    )
    if not arguments:
        return None
    ratios = [sp.cancel(argument / arguments[0]) for argument in arguments]
    if not all(ratio.is_Rational for ratio in ratios):
        return None
    u = arguments[0] * reduce(sp.gcd, ratios)
    if not u.has(variable):
        return None
    slope = sp.diff(u, variable)
    if slope.has(variable) or slope == 0:
        return None
    if len(arguments) > 1:
        w = sp.Dummy("w")
        multiples = {}
        for atom in integrand.atoms(*_TRIGONOMETRIC):
            multiple = sp.cancel(atom.args[0] / u)
            if abs(multiple) > _MULTIPLES:
                return None
            multiples[atom] = sp.expand_trig(atom.func(multiple * w)).xreplace({w: u})
        integrand = integrand.xreplace(multiples)
    return u, slope, integrand


def _in_the_square(
    expression: sp.Basic, symbol: sp.Basic, square: sp.Basic
) -> sp.Basic | None:
    """The rational function, even in `symbol`, with `symbol^2` replaced by `square`,
    or None if it isn't even."""
    numerator, denominator = sp.fraction(sp.cancel(expression))
    numerator = sp.Poly(numerator, symbol)
    denominator = sp.Poly(denominator, symbol)
    parities = {
        monomial[0] % 2 for p in (numerator, denominator) for monomial in p.monoms()
    }
    if len(parities) != 1:
        return None
    (odd,) = parities

    def evened(p: sp.Poly) -> sp.Basic:
        return sp.Add(
            *(
                coefficient * square ** ((monomial[0] - odd) // 2)
                for monomial, coefficient in p.terms()
            )
        )

    return evened(numerator) / evened(denominator)


def _through_the_poles(
    antiderivative: sp.Basic, t: sp.Basic, w: sp.Basic, u: sp.Basic, half: bool
) -> sp.Basic:
    """The antiderivative in `t = TAN(w)` written in `w`, continuous across odd
    multiples of `pi/2` unless the integrand has a pole there."""
    cosine = sp.Dummy("cosine")
    steps = sp.floor(w / sp.pi + sp.Rational(1, 2))
    angle = _Angle(t, w, u, half)

    def logarithm(argument: sp.Basic) -> sp.Basic:
        if not argument.has(t) or argument.is_polynomial(t) is not True:
            return sp.log(argument)
        poly = sp.Poly(argument, t)
        n = poly.degree()
        if len(poly.terms()) == 1:
            # The integrand has a pole at t = 0, so LN(TAN(w)) is fine.
            return sp.log(argument.xreplace({t: sp.tan(w)}))
        form = angle.form(poly, n)
        if half and n % 2:
            return angle.whole(sp.expand(form**2)) / 2 - n * cosine
        if half:
            return angle.whole(form) - n * cosine
        return _logarithm(_on_the_circle(angle.written(form), w)) - n * cosine

    def arctangent(argument: sp.Basic) -> sp.Basic:
        if argument == t:
            # ATAN(TAN(w)) + pi*FLOOR(w/pi + 1/2) is w.
            return w
        result = sp.atan(argument.xreplace({t: sp.tan(w)}))
        if not argument.has(t):
            return result
        if argument.is_polynomial(t) is not True:
            raise ValueError("an arctangent with poles of its own")
        poly = sp.Poly(argument, t)
        if poly.degree() % 2 == 0:
            return result
        direction = _direction(poly.LC())
        if direction is None:
            raise ValueError("no direction")
        return result + direction * sp.pi * steps

    written = antiderivative.replace(sp.log, logarithm).replace(sp.atan, arctangent)
    weight = sp.cancel(sp.diff(written, cosine))
    if weight.has(t) or weight.has(cosine):
        raise ValueError("not linear in the logarithms")
    if weight == 0:
        written = written.xreplace({cosine: 0})
    else:
        # The integrand has a pole where the tangent jumps, so the logarithms and
        # arctangents are left in terms of it.
        written = antiderivative.replace(
            sp.atan, lambda argument: w if argument == t else sp.atan(argument)
        )
    written = sp.expand_mul(written, deep=False)
    rational = sp.Add(
        *(
            term
            for term in sp.Add.make_args(written)
            if term.has(t) and not term.has(sp.log, sp.atan)
        )
    )
    rest = (written - rational).xreplace({t: sp.tan(w)})
    return rest + angle.fraction(rational)


class _Angle:
    """Polynomials in `t = TAN(w)` written as forms in the sine and cosine of `w`, or,
    for the half angle `w = u/2`, of `u`."""

    def __init__(self, t: sp.Basic, w: sp.Basic, u: sp.Basic, half: bool):
        self.t, self.w, self.u, self.half = t, w, u, half
        self.s, self.c = sp.Dummy("s"), sp.Dummy("c")

    def form(self, poly: sp.Poly, degree: int) -> sp.Basic:
        """`COS(w)^degree` times the polynomial, as a form in the dummies."""
        return sp.Add(
            *(
                coefficient * self.s**k * self.c ** (degree - k)
                for (k,), coefficient in poly.terms()
            )
        )

    def written(self, form: sp.Basic) -> sp.Basic:
        return form.xreplace({self.s: sp.sin(self.w), self.c: sp.cos(self.w)})

    def whole(self, form: sp.Basic) -> sp.Basic:
        """The logarithm of an even-degree form, written in the whole angle."""
        return _logarithm(self.in_the_whole_angle(form))

    def in_the_whole_angle(self, form: sp.Basic) -> sp.Basic:
        """An even-degree form in the sine and cosine of `u/2` as a polynomial in those
        of `u`: `SIN(u/2)^2 = (1 - COS(u))/2`, `COS(u/2)^2 = (1 + COS(u))/2` and
        `SIN(u/2)*COS(u/2) = SIN(u)/2`."""
        sine, cosine = sp.sin(self.u), sp.cos(self.u)
        result = sp.S.Zero
        for (i, j), coefficient in sp.Poly(form, self.s, self.c).terms():
            term = coefficient
            if i % 2:
                term *= sine / 2
                i, j = i - 1, j - 1
            term *= ((1 - cosine) / 2) ** (i // 2) * ((1 + cosine) / 2) ** (j // 2)
            result += term
        return sp.expand(result)

    def fraction(self, rational: sp.Basic) -> sp.Basic:
        """A rational function of `t` finite at `t = inf`, written in the sine and
        cosine of `w` (or of `u` for the half angle), which unlike the tangent are
        defined at odd multiples of `pi/2`. Anything else, including a function of `1/t`
        alone (`1/TAN(w)` is 0 there), is written in `TAN(w)`."""
        if rational == 0:
            return rational
        tangent = rational.xreplace({self.t: sp.tan(self.w)})
        numerator, denominator = sp.fraction(sp.cancel(sp.together(rational)))
        numerator = sp.Poly(numerator, self.t)
        denominator = sp.Poly(denominator, self.t)
        if numerator.degree() > denominator.degree() or len(denominator.terms()) == 1:
            return tangent
        constant, factors = sp.factor_list(denominator.as_expr(), self.t)
        top = self.form(numerator, denominator.degree())
        bottom = sp.S.One
        for factor, multiplicity in factors:
            poly = sp.Poly(factor, self.t)
            form = self.form(poly, poly.degree())
            if not self.half:
                bottom *= _on_the_circle(self.written(form), self.w) ** multiplicity
            elif poly.degree() % 2 == 0:
                bottom *= sp.factor_terms(self.in_the_whole_angle(form)) ** multiplicity
            else:
                if multiplicity % 2:
                    top *= form
                    multiplicity += 1
                square = sp.factor_terms(self.in_the_whole_angle(sp.expand(form**2)))
                bottom *= square ** (multiplicity // 2)
        if self.half:
            top = self.in_the_whole_angle(sp.expand(top))
        else:
            top = sp.expand(self.written(top))
        top = _less_a_constant(top, bottom, self.u)
        return sp.factor(top) / (constant * bottom)


def _logarithm(argument: sp.Basic) -> sp.Basic:
    """The logarithm of the argument minus that of its rational content, a constant."""
    _, primitive = sp.factor_terms(argument).as_content_primitive()
    return sp.log(primitive)


def _on_the_circle(form: sp.Basic, w: sp.Basic) -> sp.Basic:
    """The form in the sine and cosine of `w`, with even powers of the cosine written
    in the sine where that is shorter, using `SIN(w)^2 + COS(w)^2 = 1`."""
    sine, cosine = sp.sin(w), sp.cos(w)
    reduced = sp.factor(sp.rem(sp.expand(form), sine**2 + cosine**2 - 1, cosine))
    return reduced if sp.count_ops(reduced) < sp.count_ops(form) else form


def _less_a_constant(top: sp.Basic, bottom: sp.Basic, u: sp.Basic) -> sp.Basic:
    """The shortest numerator for `top/bottom` obtained by subtracting a multiple of
    `bottom` and optionally reducing cosine powers above one to sines. The fractions
    differ by a constant: `(b*COS(u) + b + c*SIN(u))/(b + c*SIN(u))` is `b*COS(u)/(b +
    c*SIN(u))` plus one."""
    sine, cosine = sp.sin(u), sp.cos(u)
    circle = sine**2 + cosine**2 - 1
    shortest, length = top, sp.count_ops(top)
    denominator = sp.Poly(sp.rem(sp.expand(bottom), circle, cosine), sine, cosine)
    for written in (top, sp.rem(sp.expand(top), circle, cosine)):
        numerator = sp.Poly(written, sine, cosine)
        multiples = [sp.S.Zero]
        for monomial, coefficient in denominator.terms():
            mine = numerator.coeff_monomial(monomial)
            if mine != 0:
                multiples.append(sp.cancel(mine / coefficient))
        for multiple in multiples:
            candidate = (numerator - denominator * multiple).as_expr()
            if sp.count_ops(candidate) < length:
                shortest, length = candidate, sp.count_ops(candidate)
    return shortest


def _direction(k: sp.Basic) -> sp.Basic | None:
    """The sign of `ATAN(k*t)` as `t` goes to `+inf`, or None if unknown.

    For real `k` this is its sign. A parametric `k` can be imaginary: `k = 2*A/SQRT(d)`
    for a quadratic with negative discriminant `d`. For real `q` and `r`, `r*SQRT(q)`
    has the sign of `r` and `r/SQRT(q)` that of `r*q`, which for `q < 0` is the sign
    of its imaginary part, the side the arctangent approaches.
    """
    odd = []
    roots = 0
    for factor in sp.Mul.make_args(sp.factor(k)):
        base, exponent = factor.as_base_exp()
        if base.is_extended_real is not True or not exponent.is_Rational:
            return None
        if exponent.q == 1:
            if exponent.p % 2:
                odd.append(base)
        elif exponent.q == 2:
            roots += 1
            if roots > 1:
                return None
            # base^(p/2) = base^((p - 1)/2)*SQRT(base) or base^((p + 1)/2)/SQRT(base)
            whole = (exponent.p - 1) // 2 if exponent > 0 else (exponent.p + 1) // 2
            if whole % 2:
                odd.append(base)
            if exponent < 0:
                odd.append(base)
        else:
            return None
    constant, factors = sp.factor_list(sp.Mul(*odd))
    signs = [
        factor
        for factor, multiplicity in factors
        if multiplicity % 2 and factor.is_nonnegative is not True
    ]
    return sp.sign(constant) * sp.sign(sp.Mul(*signs))
