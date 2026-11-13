"""Integration by trying each method in turn.

Every integral enters through `integral`. `_METHODS`, at the end of the module, lists
the methods in order, and `_accepted` decides whether a method's answer is taken.
Definite-integral fixes live in `definite`, and the printed form of answers in `form`.

Methods correct by construction come first: a power times a multiple of its base's
derivative, our own rational, trigonometric, radical and exponential integrators, then
textbook rules (substitution, products as sums, integration by parts, a positive
variable, substituting a square root). They write answers as the integral tables do,
without parameter splits. Sympy's general integrators come next, mainly for answers in
special functions; they are slower, split more and are sometimes wrong, so their
answers are taken only if proved to differentiate back and expressible in the
notation.

The manual rules alone and the integral over an unassumed variable are retries, tried
only after a sympy answer is refused, since where sympy found nothing they find nothing
either.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import NamedTuple

import sympy as sp
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.integrals.manualintegrate import manualintegrate
from sympy.integrals.risch import NonElementaryIntegral, risch_integrate
from sympy.simplify.fu import TR8

from rederive.engine.integration.definite import (
    CONVERGENT,
    UNDECIDED,
    VERDICTS,
    across_the_steps,
    at_the_singularities,
    between_the_ends,
    between_the_steps,
    beyond_convergence,
    by_the_antiderivative,
    by_the_sign_of_a_parameter,
    definite,
    ends,
    odd_about_the_middle,
    past_a_singular_end,
    real_where_real,
    unresolved,
    up_to_the_steps,
    with_moving_poles,
    within_each_case,
)
from rederive.engine.integration.form import (
    anywhere,
    antiderivative,
    continuous,
    defined_cases,
    on_the_principal_sheet,
    parametrised,
    single_case,
    somewhere_only,
    unwritable,
)
from rederive.engine.integration.proof import differentiates_back, holds
from rederive.engine.printer import shadowed_names
from rederive.engine.radical import radical_antiderivative
from rederive.engine.rational import rational_antiderivative
from rederive.engine.rewriting import attempt, forced
from rederive.engine.trigonometric import trigonometric_antiderivative

__all__ = [
    "by_a_positive_substitution",
    "by_sympy",
    "integral",
    "numerically",
    "within_reach",
]


def integral(head: sp.Basic) -> sp.Basic:
    """Evaluate an integral, correcting what sympy gets wrong about definite integrals.

    Definite integrals are first split at step-function jumps and examined at their
    singularities; an answer that depends on a parameter's sign is recomputed for each
    sign. Results are cached, since an unevaluated integral is requested again by
    simplification and by later commands. The answers that can be written depend on the
    built-ins the user has redefined, so those are part of the key.

    Products of exponentials are combined first, as Simplify combines them, and an
    integral left unevaluated is tried once more with its radicals resolved (see
    `_with_radicals_resolved`).

    Within `numerically`, heuristic Risch isn't run; see there.
    """
    return _integral(head, shadowed_names(), _numerically.get())


@lru_cache(maxsize=256)
def _integral(head: sp.Basic, shadowed: frozenset[str], numeric: bool) -> sp.Basic:
    """`integral`, with `shadowed` the built-ins the answer may not use, and `numeric`
    whether it is within `numerically`."""
    tidied = _with_integrand_tidied(head)
    if tidied is not None:
        found = integral(tidied)
        return head if found.has(sp.Integral) else found
    combined = _with_exponentials_combined(head)
    if combined != head:
        return integral(combined)
    over_one = _with_radicands_combined(head)
    if over_one is not None:
        found = integral(over_one)
        if not found.has(sp.Integral):
            return found
    stepped = between_the_steps(head)
    if stepped is not None:
        return stepped
    across = across_the_steps(head)
    if across is not None:
        return across
    up_to = up_to_the_steps(head)
    if up_to is not None:
        return up_to
    value = _integral_through(head)
    moving = with_moving_poles(head, value)
    value = within_each_case(real_where_real(head, value if moving is None else moving))
    if value.has(sp.Integral):
        resolved = _with_radicals_resolved(head)
        if resolved is not None:
            again = integral(resolved)
            if not again.has(sp.Integral):
                return again
    return value


#: How much smaller a tidied integrand must be, in operations, to replace the integrand,
#: and the largest integrand that is tidied.
_TIDIER = sp.Rational(2, 3)
_TIDIED_SIZE = 300


def _with_integrand_tidied(head: sp.Basic) -> sp.Basic | None:
    """`head` with its integrand much smaller over one denominator, cancelled, or with
    the square roots of its denominator sums moved to the numerator, or None.

    An integrand as evaluated may be a quotient of quotients that Derive would have
    simplified first, and on such a form sympy's manual rules can search for half a
    minute and fail, where the smaller form integrates at once. ODE.MTH's `LAGRANGE_B`
    integrates `DIF(g, d)/(d + g)` for `g = (SQRT(d^2 + 1) - 1)/(-d - SQRT(d^2 +
    1))`, which is `(-d^2 + d*SQRT(d^2 + 1) - 1)/((d^2 + 1)*SQRT(d^2 + 1))`, and its
    `CANONIC_8` divides by an unsimplified Wronskian. Only the smaller form is tried.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 1:
        return None
    variable, integrand = head.limits[0][0], head.function
    try:
        size = sp.count_ops(integrand)
        if size > _TIDIED_SIZE or not integrand.has(sp.Pow):
            return None
        candidates = [sp.cancel(integrand)]
        denominator = sp.fraction(sp.together(integrand))[1]
        if any(
            isinstance(part, sp.Add)
            and any(_is_square_root(power, variable) for power in part.atoms(sp.Pow))
            for part in sp.Mul.make_args(denominator)
        ):
            top, bottom = sp.fraction(sp.radsimp(sp.together(integrand)))
            candidates.append(sp.cancel(sp.expand(top) / sp.expand(bottom)))
        best = min(candidates, key=sp.count_ops)
        if sp.count_ops(best) > _TIDIER * size:
            return None
    except Exception:
        return None
    return head.func(best, *head.limits)


def _is_square_root(power: sp.Pow, variable: sp.Basic) -> bool:
    return (
        power.exp.is_Rational
        and power.exp.q == 2
        and power.base.has(variable)
        and bool(power.base.is_polynomial(variable))
    )


#: Whether the integral being computed is a number that is going to be approximated.
_numerically: ContextVar[bool] = ContextVar("_numerically", default=False)


@contextmanager
def numerically() -> Iterator[None]:
    """Compute integrals that are going to be approximated: without heuristic Risch.

    Under Approximate, a definite integral with numeric limits and no parameters is
    computed exactly first and approximated after, but where no closed form comes
    quickly, quadrature gives the digits at once, as it does in Derive. Heuristic Risch
    is the search that doesn't come quickly: on `INT(#e^(1.3*COS(t))*COS(t), t, 0, pi)`
    it takes a minute to fail, and BESSEL.MTH's BESSEL_I is that integral. The other
    methods still run, and so does the examination of singularities, so that
    `INT(1/x^2, x, -1, 1)` is still `inf` rather than a number.
    """
    token = _numerically.set(True)
    try:
        yield
    finally:
        _numerically.reset(token)


def _with_exponentials_combined(head: sp.Basic) -> sp.Basic:
    """The integral with each product of exponentials in its integrand written as one
    exponential, as `#e^(-a*t)*#e^(-s*t)` is `#e^(-a*t - s*t)`. LAPLACE in INT_APPS.MTH
    multiplies its integrand by such a factor."""

    def several(e: sp.Basic) -> bool:
        return e.is_Mul and sum(isinstance(f, sp.exp) for f in e.args) > 1

    def combined(product: sp.Basic) -> sp.Basic:
        exponents = [f.args[0] for f in product.args if isinstance(f, sp.exp)]
        rest = [f for f in product.args if not isinstance(f, sp.exp)]
        return sp.Mul(*rest) * sp.exp(sp.Add(*exponents))

    if not head.function.has(sp.exp):
        return head
    function = attempt(head.function, lambda f: f.replace(several, combined))
    if function is None or function == head.function:
        return head
    return head.func(function, *head.limits)


def _with_radicands_combined(head: sp.Basic) -> sp.Basic | None:
    """The integral with each radicand that is a sum of quotients and comes to a
    nonnegative constant over a function of the variables written as that quotient,
    with the numerator taken out of the root, or None if there is none.

    The surface of a sphere, SURFACE_AREA in INT_APPS.MTH, integrates `SQRT(x^2/(1 -
    x^2 - y^2) + y^2/(1 - x^2 - y^2) + 1)`, which is `1/SQRT(1 - x^2 - y^2)`; as
    written, the full methods search for seconds before giving up. Taking the
    numerator out changes the root only where the denominator is negative and the
    integrand isn't real, as in `_with_radicals_resolved`. The integral is used only
    if this form answers it.
    """
    variables = [limit[0] for limit in head.limits]

    def compound(e: sp.Basic) -> bool:
        return (
            e.is_Pow
            and e.exp.is_Rational
            and not e.exp.is_Integer
            and e.base.is_Add
            and any(sp.denom(term).has(*variables) for term in e.base.args)
        )

    def combined(power: sp.Basic) -> sp.Basic:
        base, exponent = power.args
        numerator, denominator = sp.fraction(sp.cancel(sp.together(base)))
        if numerator.is_negative:
            numerator, denominator = -numerator, sp.expand(-denominator)
        if numerator.has(*variables) or numerator.is_nonnegative is not True:
            return power
        if not denominator.has(*variables):
            return power
        return numerator**exponent / denominator**exponent

    if not head.function.has(*variables):
        return None
    function = attempt(head.function, lambda f: f.replace(compound, combined))
    if function is None or function == head.function:
        return None
    return head.func(function, *head.limits)


def _with_radicals_resolved(head: sp.Basic) -> sp.Basic | None:
    """The integral with radicals the integrators don't see through rewritten, or None
    if it has none.

    - A root of a reciprocal is the reciprocal of the root: `SQRT(1/(1 - y^2))` is
      `1/SQRT(1 - y^2)`, which differs only where the radicand is negative and the
      integrand isn't real, so `INT(SQRT(1/(1 - y^2)), y)` is `ASIN(y)`.
    - Over a real variable, `a*(1 + COS(u))` is `2*a*COS(u/2)^2` and `a*(1 - COS(u))`
      is `2*a*SIN(u/2)^2`, so for a positive `a` a root of one is a power of an
      absolute value: `SQRT(1 + COS(x))` is `SQRT(2)*ABS(COS(x/2))`, which
      integrates between jumps.
    """
    variables = [limit[0] for limit in head.limits]

    def radical(e: sp.Basic) -> bool:
        return (
            e.is_Pow
            and e.exp.is_Rational
            and not e.exp.is_Integer
            and e.base.has(*variables)
        )

    def resolved(power: sp.Basic) -> sp.Basic:
        base, exponent = power.args
        if base.is_Pow and base.exp.is_Integer and base.exp < 0:
            return base.base ** (base.exp * exponent)
        if not exponent.is_Rational or exponent.q != 2:
            return power
        if not all(variable.is_real for variable in variables):
            return power
        half = _half_angle_square(base)
        return power if half is None else half**exponent

    function = attempt(head.function, lambda f: f.replace(radical, resolved))
    if function is None or function == head.function:
        return None
    return head.func(function, *head.limits)


def _half_angle_square(base: sp.Basic) -> sp.Basic | None:
    """`base` as `c*COS(u/2)^2` or `c*SIN(u/2)^2` with a positive `c`, where it is
    `a + b*COS(u)` with `b` equal to `a` or `-a`, or None."""
    expanded = sp.expand(base)
    cosines = [c for c in expanded.atoms(sp.cos) if c.free_symbols]
    if len(cosines) != 1:
        return None
    (cosine,) = cosines
    b = expanded.coeff(cosine)
    a = sp.expand(expanded - b * cosine)
    if a.has(cosine) or b.has(cosine) or a.free_symbols & cosine.free_symbols:
        return None
    if b.free_symbols & cosine.free_symbols or not (2 * a).is_positive:
        return None
    half = cosine.args[0] / 2
    if sp.expand(a - b) == 0:
        return 2 * a * sp.cos(half) ** 2
    if sp.expand(a + b) == 0:
        return 2 * a * sp.sin(half) ** 2
    return None


def _integral_through(head: sp.Basic) -> sp.Basic:
    """An integral evaluated by the methods, after step functions are handled; see
    `integral`.

    A definite integral's answer is checked against the verdicts of `definite`. Only
    then is an unevaluated integral split by `_by_linearity`, so that one that diverges
    or depends on a sign is answered as a whole first.
    """
    verdict = at_the_singularities(head)
    if verdict is not None and verdict not in VERDICTS:
        return verdict
    if verdict is UNDECIDED:
        value = head
    elif verdict is CONVERGENT and odd_about_the_middle(head):
        return sp.S.Zero
    else:
        value = _answered(head, _METHODS)
        if value is None:
            value = head
        if isinstance(value, sp.Integral) and definite(head):
            value = by_the_antiderivative(head) or value
        infinite = value.has(sp.nan, sp.zoo) or value.is_finite is False
        if verdict is not None and infinite:
            value = head
    past = past_a_singular_end(head, value)
    if past is not None:
        return past
    beyond = beyond_convergence(head, value)
    if beyond is not None:
        return beyond
    if value != head or verdict is UNDECIDED:
        signed = by_the_sign_of_a_parameter(head, value)
        if signed is not None:
            return signed
    if definite(head) and value.has(sp.zoo) and value is not sp.zoo:
        return head
    return _by_linearity(head) if value == head else value


class Candidate(NamedTuple):
    """A method's answer, with flags for `_accepted` and the methods to retry if it is
    refused."""

    value: sp.Basic
    #: Correct by construction, so not proved.
    trusted: bool = False
    #: Already in the integral tables' form, or produced by the manual rules, so the
    #: manual rules aren't tried for a shorter one.
    tabled: bool = False
    #: Built from already accepted and formatted answers, so taken as is.
    formed: bool = False
    #: Methods to retry, in order, if this answer is refused. They're only worth trying
    #: when sympy answered wrongly, not when it found nothing.
    retries: tuple[Method, ...] = ()


#: A method takes a head and returns a candidate answer or None.
Method = Callable[[sp.Basic], Candidate | None]


def _answered(head: sp.Basic, methods: tuple[Method, ...]) -> sp.Basic | None:
    """The first answer `_accepted` takes, in printed form, or None.

    When an answer is refused, its `retries` are tried through the same check. Each
    retry runs at most once, since it works from the head and would repeat the same
    slow search.
    """
    retried: set[Method] = set()
    for method in methods:
        candidate = method(head)
        if candidate is None:
            continue
        value = _accepted(head, candidate)
        if value is None:
            for retry in candidate.retries:
                if retry in retried:
                    continue
                retried.add(retry)
                fallback = retry(head)
                if fallback is None:
                    continue
                value = _accepted(head, fallback)
                if value is not None:
                    candidate = fallback
                    break
        if value is None:
            continue
        if candidate.formed:
            return value
        return antiderivative(head, value, tabled=candidate.tabled)
    return None


def _accepted(head: sp.Basic, candidate: Candidate) -> sp.Basic | None:
    """The method's answer as accepted, or None if it is refused.

    A sympy answer is accepted only if `differentiates_back` proves it, or after
    `single_case` narrows it to a proved case; an unevaluated integral is better than
    an unproved antiderivative. `trusted` answers skip the proof, which can take longer
    than the integration did.

    An answer in a function the notation can't express is refused first, trusted or
    not, since proving it can take seconds to minutes. Before that, the answer is moved
    to the principal sheet, and one undefined for some values of the variable is
    narrowed to a case valid everywhere. Answers with no antiderivative to prove
    (definite, multiple, unevaluated, or `formed`) are taken as is.
    """
    value = defined_cases(candidate.value, head.function, head.limits[0][0])
    if (
        candidate.formed
        or len(head.limits) != 1
        or len(head.limits[0]) != 1
        or value.has(sp.Integral)
    ):
        return value
    variable, integrand = head.limits[0][0], head.function
    value = on_the_principal_sheet(value, integrand, variable)
    if somewhere_only(value):
        value = anywhere(value, integrand, variable)
        if value is None:
            return None
    if unwritable(value, variable):
        return None
    if candidate.trusted or differentiates_back(value, integrand, variable):
        return value
    narrowed = single_case(value, integrand, variable)
    if narrowed == value or unwritable(narrowed, variable):
        return None
    return narrowed


def by_sympy(head: sp.Basic) -> sp.Basic:
    """An integral evaluated by sympy's methods alone, as `integral` runs them. `definite`
    uses this for the auxiliary integrals it builds.
    """
    value = _answered(head, _SYMPY)
    return head if value is None else value


def _of_a_binomial_power(head: sp.Basic) -> Candidate | None:
    """An antiderivative of `c*x^m*(a*x^p + b*x^q)^n` as a power of the binomial, where
    the power of `x` outside is a multiple of the derivative of the binomial's inside,
    or None.

    With `k = q - p`, the integrand is `c*x^(m + n*p)*(a + b*x^k)^n` (Rubi's rule 1607).
    If `m + n*p = k - 1`, its antiderivative is `c*(a + b*x^k)^(n + 1)/(b*k*(n + 1))`
    (rule 267), so `INT((3*x^2 + 10*x^27)^12, x)` is `(10*x^25 + 3)^13/3250`, where
    sympy and the rational integrator expand thirteen terms. More generally, if `m + n*p
    = k*j - 1` for an integer `j` up to `_SPAN`, it is the binomial to the `n + 1` times
    a polynomial of degree `j - 1` in `x^k`, as `INT(x^5*(a + b*x^3)^7, x)` is `(8*b*x^3
    - a)*(b*x^3 + a)^8/(216*b^2)`. Either term of the binomial may be the one factored
    out, unless one is constant: factoring `x` out of `a + b*x` would give `1/a` in the
    answer where the integrand has no pole at `a = 0`.

    `n` must be a number other than -1, 0 and 1, and an integer unless `p = 0`, since
    otherwise `x^(n*p)` can't be factored out. A symbolic `n` is left to
    `_by_derivative_divides`, which makes the split at `n = -1` continuous, and a
    logarithm, where `n + i + 1 = 0` for some `i < j`, to the other methods. For `j >
    1`, a positive integer `n` must be at least `2*j - 1`, else the expanded polynomial
    is shorter, and a negative integer `n` is left to the rational integrator, whose
    numerator vanishes at zero, as in `x^2/(2*(x + 1)^2)`.

    `a`, `b` and the exponents of `x` may be symbolic, and are taken as generic, as the
    tables take them. The answer is `formed`: choosing the constant so that it is
    continuous at `b = 0` or `k = 0`, as `antiderivative` would, gives back the expanded
    polynomial for an integer `n`.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 1:
        return None
    variable, integrand = head.limits[0][0], head.function
    coefficient, rest = integrand.as_independent(variable, as_Add=False)
    outside, binomial = sp.S.Zero, None
    for factor in sp.Mul.make_args(rest):
        base, exponent = factor.as_base_exp()
        if exponent.has(variable):
            return None
        if base == variable:
            outside += exponent
        elif binomial is None and isinstance(base, sp.Add) and len(base.args) == 2:
            binomial = base, exponent
        else:
            return None
    if binomial is None:
        return None
    base, n = binomial
    if not n.is_Rational or n in (-1, 0, 1):
        return None
    monomials = []
    for term in base.args:
        scale, power = term.as_independent(variable, as_Add=False)
        if power == 1:
            monomials.append((scale, sp.S.Zero))
            continue
        bare, exponent = power.as_base_exp()
        if bare != variable or exponent.has(variable):
            return None
        monomials.append((scale, exponent))
    constant = any(exponent == 0 for _, exponent in monomials)
    for (a, p), (b, q) in (monomials, monomials[::-1]):
        k = q - p
        if (p != 0 and (constant or not n.is_Integer)) or k == 0:
            continue
        # `x^(m + n*p)` is `x^(k*j - 1)`, which is `u^(j - 1)*x^(k - 1)` for `u = x^k`.
        j = attempt(outside + n * p + 1, lambda e: sp.cancel(e / k))
        if j is None or not j.is_Integer or not 1 <= j <= _SPAN:
            continue
        if j > 1 and n.is_Integer and (n < 0 or 2 * j > n + 1):
            continue
        if any(n + i == 0 for i in range(1, j + 1)):
            continue
        inner = a + b * variable**k
        # `INT(v^n*(v - a)^(j - 1), v)/v^(n + 1)`, at `v = a + b*u`.
        v = sp.Dummy("v")
        factor = sum(
            (
                sp.binomial(j - 1, i) * (-a) ** (j - 1 - i) * v**i / (n + i + 1)
                for i in range(j)
            ),
            sp.S.Zero,
        )
        factor = sp.together(sp.expand(factor.subs(v, inner)))
        value = coefficient * inner ** (n + 1) * factor / (k * b**j)
        return Candidate(value, formed=True)
    return None


#: The largest `j = (m + n*p + 1)/k` for which `_of_a_binomial_power` writes the
#: antiderivative as a power of the binomial times a polynomial of degree `j - 1`.
_SPAN = 4


def _of_a_power_times_its_derivative(head: sp.Basic) -> Candidate | None:
    """An antiderivative of `c*g(x)*f(x)^n` as `k*f(x)^(n + 1)/(n + 1)`, where `g/f'` is
    a constant `k`, or None.

    `INT((2*x + 1)*(x^2 + x + 1)^12, x)` is `(x^2 + x + 1)^13/13`, and
    `INT(COS(x)*(SIN(x) + 2)^12, x)` is `(SIN(x) + 2)^13/13`, where sympy and the
    rational and trigonometric integrators expand the power or take minutes. Candidates
    for `f` are the bases of factors that are powers with a rational exponent `n` other
    than -1, so an integrand without one is rejected without work, and each costs one
    cancellation of `g/f'`. A power of the variable itself is left to the other
    methods, as is a logarithm, and a base whose derivative vanishes, such as
    `SIN(x)^2 + COS(x)^2`, is a constant in disguise.

    A parameter in the denominator of `k` makes `f'` vanish for some value of it, where
    the answer is undefined. For `n > -1` it is taken as generic, as the tables and
    `_of_a_binomial_power` take it, since an answer continuous there is the expanded
    power for an integer `n`. A negative `n` is left to the other integrators, which
    write such answers as short and continuous, as `(1/COS(a*x) - 1)/a` for
    `INT(SIN(a*x)/COS(a*x)^2, x)`. The answer is `formed`, since `antiderivative` would
    expand it for continuity.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 1:
        return None
    variable, integrand = head.limits[0][0], head.function
    coefficient, rest = integrand.as_independent(variable, as_Add=False)
    factors = sp.Mul.make_args(rest)
    for index, factor in enumerate(factors):
        if not factor.is_Pow or factor.base == variable:
            continue
        f, n = factor.args
        if not n.is_Rational or n == -1 or not f.has(variable):
            continue
        outside = sp.Mul(*factors[:index], *factors[index + 1 :])
        derivative = attempt(sp.diff(f, variable), sp.cancel)
        if derivative is None or derivative == 0:
            continue
        k = attempt(outside / derivative, sp.cancel)
        if k is None or k.has(variable, sp.zoo, sp.nan):
            continue
        if n < 0 and sp.denom(k).free_symbols:
            continue
        return Candidate(coefficient * k * f ** (n + 1) / (n + 1), formed=True)
    return None


def _between_the_ends_of_a_power(head: sp.Basic) -> Candidate | None:
    """A definite integral as the endpoint difference of the antiderivative
    `_of_a_binomial_power` or `_of_a_power_times_its_derivative` finds, or None.

    `INT(COS(x)*(SIN(x) + 2)^12, x, 0, pi)` is 0 at once, where sympy's definite
    integrators expand the power and take minutes. `between_the_ends` checks that the
    antiderivative is continuous on the interval, and the checks in `_integral_through`
    still apply to the value.
    """
    if not definite(head):
        return None
    variable = head.limits[0][0]
    indefinite = sp.Integral(head.function, variable)
    for method in (_of_a_binomial_power, _of_a_power_times_its_derivative):
        candidate = method(indefinite)
        if candidate is not None:
            value = between_the_ends(head, candidate.value)
            return None if value is None else Candidate(value, formed=True)
    return None


def _of_a_rational_function(head: sp.Basic) -> Candidate | None:
    """An integral of a rational function of its variable, done here, or None.

    Sympy's rational integrator loses terms (see `engine.rational`), so that module
    supplies the antiderivative. A definite integral is the difference of its limits at
    the ends, as Derive does (7.4, p.199). If that fails, sympy is tried, except for an
    integral with no parameters: sympy would give 0 for `INT(1/(x^8 + 1), x, 0, 1)` or
    spend minutes on `1/(x^5 + x + 1)`, so it is left unevaluated.

    Only integrands rational as a whole qualify; a rational term in a larger sum goes
    to sympy with the rest.
    """
    if len(head.limits) != 1 or len(head.limits[0]) not in (1, 3):
        return None
    variable, integrand = head.limits[0][0], head.function
    if integrand.is_rational_function(variable) is not True:
        return None
    if len(head.limits[0]) == 1:
        primitive = rational_antiderivative(integrand, variable)
        if primitive is None:
            return None
        return Candidate(primitive, trusted=True)
    _, low, high = head.limits[0]
    if low.has(variable) or high.has(variable):
        # `INT(1/x, x, 1, x)` is the antiderivative from 1 to `x`; rename the
        # integration variable so it differs from the one in the limit.
        bound = sp.Dummy(variable.name, **variable.assumptions0)
        return _of_a_rational_function(
            sp.Integral(integrand.xreplace({variable: bound}), (bound, low, high))
        )
    primitive = integral(sp.Integral(integrand, variable))
    if primitive.has(sp.Integral):
        return None
    at_ends = ends(
        primitive,
        variable,
        low,
        high,
        lambda point: _toward_an_end(primitive, integrand, variable, point),
    )
    value = None if at_ends is None else at_ends[0] - at_ends[1]
    if value is None or value.has(sp.nan):
        if head.free_symbols or integrand.free_symbols - {variable}:
            return None
        return Candidate(head, formed=True)
    return Candidate(value, formed=True)


def _toward_an_end(
    primitive: sp.Basic, integrand: sp.Basic, variable: sp.Basic, end: sp.Basic
) -> sp.Basic:
    """The antiderivative of a rational function, rewritten so its limit at an end can be
    taken.

    Sympy can't take limits of a `RootSum` of `a*LN(h(a, x))`. If each `h` is monic and
    linear in `x`, the sum is continuous and is evaluated at a finite end, and tends to
    zero at `inf`; at `-inf` it is left alone.

    At an infinite end where the integrand decays like `1/x^2` or faster, each `LN(p)`
    is rewritten as `LN(p/x^d)`; the dropped `d*LN(x)` terms must cancel since the
    integral converges. Sympy can't always see that: for `INT(1/(x^7 + 1), x)` the
    cancellation needs identities of cosines of multiples of `pi/7`, and it gets `inf`
    at `inf`.
    """
    sums = [rs for rs in primitive.atoms(sp.RootSum) if rs.has(variable)]
    if not all(_of_monic_linear_logarithms(rs, variable) for rs in sums):
        return primitive
    if not end.is_infinite:
        return primitive.xreplace({rs: rs.subs(variable, end) for rs in sums})
    numerator, denominator = sp.fraction(sp.cancel(integrand))
    if sp.degree(denominator, variable) - sp.degree(numerator, variable) < 2:
        return primitive
    if sums and end is not sp.oo:
        return primitive
    # Only logarithms outside the sums; those inside are over the sums' own variables.
    outside = primitive.xreplace({rs: sp.S.Zero for rs in sums})
    logarithms = [log for log in outside.atoms(sp.log) if log.has(variable)]
    if not all(log.args[0].is_polynomial(variable) for log in logarithms):
        return primitive
    toward = variable if end is sp.oo else -variable
    return primitive.xreplace(
        {
            **{rs: sp.S.Zero for rs in sums},
            **{
                log: sp.log(
                    sp.expand(log.args[0] / toward ** sp.degree(log.args[0], variable))
                )
                for log in logarithms
            },
        }
    )


def _of_monic_linear_logarithms(rs: sp.RootSum, variable: sp.Basic) -> bool:
    """Whether the `RootSum` is of `a*LN(h(a, x))`, with each `h` linear and monic in
    the variable."""
    (t,), body = rs.fun.args
    coefficient, logarithm = body.as_independent(variable, as_Add=False)
    if coefficient != t or not isinstance(logarithm, sp.log):
        return False
    if not logarithm.args[0].is_polynomial(variable):
        return False
    argument = sp.Poly(logarithm.args[0], variable)
    return argument.degree() == 1 and argument.LC() == 1


def _of_a_trigonometric_function(head: sp.Basic) -> Candidate | None:
    """An antiderivative of a rational function of the sine and cosine of one linear
    argument, by `engine.trigonometric`, or None. Sympy can spend minutes on these with
    parametric coefficients, and splits over parameter values.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 1:
        return None
    variable = head.limits[0][0]
    primitive = trigonometric_antiderivative(head.function, variable)
    if primitive is None:
        return None
    return Candidate(primitive, trusted=True, tabled=True)


def _of_a_radical(head: sp.Basic) -> Candidate | None:
    """An antiderivative of a rational function of the variable and roots of linear or
    quadratic polynomials, by `engine.radical`, or None.

    Sympy can spend minutes on these with parametric coefficients. Remaining integrals
    in the answer are evaluated separately; if one fails or is the head itself, this
    returns None.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 1:
        return None
    variable = head.limits[0][0]
    primitive = radical_antiderivative(head.function, variable)
    if primitive is None:
        return None
    for left in primitive.atoms(sp.Integral):
        if sp.cancel(left.function - head.function) == 0:
            return None
        done = integral(left)
        if done.has(sp.Integral):
            return None
        primitive = primitive.xreplace({left: done})
    return Candidate(primitive, trusted=True, tabled=True)


def _over_an_exponential(head: sp.Basic) -> Candidate | None:
    """An antiderivative of a rational function of exponentials of multiples of one linear
    argument, by the substitution `u = #e^(g*x)`, or None.

    Risch leaves such integrals as a `ROOT_SUM` when it can't tell real roots from
    complex ones, where the substitution gives the table's `ATAN` form. Risch is also
    nondeterministic here, its generator depending on the string hash seed.

    `g` is the largest value of which every exponent's slope is an integer multiple.
    Returns None if anything else depends on the variable, or if the rational function
    of `u` has an irreducible denominator factor above degree 2, whose roots would give
    pages of nested radicals.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 1:
        return None
    variable, integrand = head.limits[0][0], head.function
    exponentials = [e for e in integrand.atoms(sp.exp) if e.has(variable)]
    if not exponentials:
        return None
    linear = []
    for exponential in exponentials:
        exponent = sp.expand(exponential.args[0])
        slope = exponent.coeff(variable)
        offset = sp.expand(exponent - slope * variable)
        if slope == 0 or slope.has(variable) or offset.has(variable):
            return None
        linear.append((exponential, slope, offset))
    _, first, _ = min(linear, key=lambda entry: sp.default_sort_key(entry[1]))
    ratios = [sp.cancel(slope / first) for _, slope, _ in linear]
    if not all(ratio.is_Rational for ratio in ratios):
        return None
    unit = sp.gcd([sp.Rational(ratio) for ratio in ratios])
    numeric, _ = first.as_coeff_Mul()
    if numeric < 0:
        unit = -unit
    rate = first * unit
    u = sp.Dummy("u", positive=True)
    written = integrand.xreplace(
        {
            exponential: sp.exp(offset) * u ** (ratio / unit)
            for (exponential, _, offset), ratio in zip(linear, ratios)
        }
    )
    if written.has(variable):
        return None
    rational = sp.cancel(written / (rate * u))
    try:
        _, factors = sp.factor_list(sp.denom(rational), u)
    except Exception:
        return None
    if any(sp.degree(factor, u) > 2 for factor, _ in factors):
        return None
    found = rational_antiderivative(rational, u, binomials=True, formal=True)
    if found is None or found.has(sp.RootSum, sp.I):
        return None
    found = found.xreplace({u: sp.exp(rate * variable)})
    return Candidate(found, trusted=True, tabled=True)


def _of_an_exponential_times_a_wave(head: sp.Basic) -> Candidate | None:
    """An antiderivative of `c*#e^(k*x + d)*COS(b*x + e)` or the same with a sine, by
    the integral tables, or None.

    The tables give `#e^(k*x + d)*(k*COS(b*x + e) + b*SIN(b*x + e))/(k^2 + b^2)`, and
    `#e^(k*x + d)*(k*SIN(b*x + e) - b*COS(b*x + e))/(k^2 + b^2)` for the sine. Sympy
    finds these for rational slopes but not for `INT(COS(x)*#e^((1 - SQRT(13))*x/2),
    x)`, so only numeric slopes are taken here, real and nonzero; symbolic ones are
    left to sympy, which splits off `k^2 + b^2 = 0`.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 1:
        return None
    variable, integrand = head.limits[0][0], head.function
    coefficient, rest = integrand.as_independent(variable, as_Add=False)
    exponent, wave = sp.S.Zero, None
    for factor in sp.Mul.make_args(rest):
        if isinstance(factor, sp.exp):
            exponent += factor.args[0]
        elif wave is None and isinstance(factor, (sp.sin, sp.cos)):
            wave = factor
        else:
            return None
    if wave is None or exponent == 0:
        return None
    slopes = []
    for argument in (exponent, wave.args[0]):
        polynomial = attempt(argument, lambda a: sp.Poly(a, variable))
        if polynomial is None or polynomial.degree() != 1:
            return None
        slope = polynomial.LC()
        if not slope.is_number or not slope.is_extended_real or slope == 0:
            return None
        slopes.append(slope)
    k, b = slopes
    argument = wave.args[0]
    if isinstance(wave, sp.cos):
        shape = k * sp.cos(argument) + b * sp.sin(argument)
    else:
        shape = k * sp.sin(argument) - b * sp.cos(argument)
    scale = forced(1 / (k**2 + b**2), sp.radsimp)
    value = coefficient * scale * sp.exp(exponent) * shape
    return Candidate(value, trusted=True, tabled=True)


def _by_derivative_divides(head: sp.Basic) -> Candidate | None:
    """An antiderivative by `u = g(x)`, where the integrand is `h(g(x))*g'(x)` and
    contains a power with a symbolic exponent, or None.

    Sympy misses `u = x^n` in `1/(x*SQRT(x^n + a^2))` and `u = TAN(a*x)` in
    `TAN(a*x)^n/COS(a*x)^2`. Candidates for `g` are such powers with a numeric part of
    the exponent removed, and their bases if they are functions. If the integrand over
    `g'` is free of `x` after substitution, `INT(h(u), u)` is integrated separately.

    A parameter split in the result is made continuous; a pole without a split is left,
    since its principal part may be imaginary. Only one level of substitution is done,
    and the result is correct by construction.
    """
    if _substituting.get() or len(head.limits) != 1 or len(head.limits[0]) != 1:
        return None
    variable, integrand = head.limits[0][0], head.function
    powers = [
        power
        for power in integrand.atoms(sp.Pow)
        if power.base.has(variable)
        and not power.exp.has(variable)
        and power.exp.free_symbols
    ]
    candidates = []
    for power in powers:
        base = power.base
        if isinstance(base, sp.Function) and len(base.args) == 1:
            candidates.extend((base, slope) for slope in _derivatives(base, variable))
        _, exponent = sp.Add(*power.exp.as_coeff_add()[1]).as_coeff_Mul()
        core = base**exponent
        candidates.append((core, exponent * core * sp.diff(base, variable) / base))
    candidates.extend(_sine_and_cosine(integrand, variable))
    nested = _sine_and_cosine(integrand, variable, nested=True)
    candidates = list(dict.fromkeys(candidates + nested))
    u = sp.Dummy("u", real=True)
    token = _substituting.set(True)
    try:
        for inner, slope in candidates:
            substituted = attempt(
                integrand / slope,
                lambda q: _squares_as_complements(
                    sp.cancel(q).subs(inner, u), inner, u
                ),
            )
            if substituted is None or substituted.has(variable, sp.zoo, sp.nan):
                continue
            if isinstance(inner, (sp.sin, sp.cos)) and substituted.atoms(sp.Function):
                if (inner, slope) not in nested or not all(
                    call.has(u) for call in substituted.atoms(sp.Function)
                ):
                    continue
            done = integral(sp.Integral(substituted, u))
            if done.has(sp.Integral):
                continue
            answer = done.subs(u, inner)
            if answer.has(sp.Piecewise):
                answer = continuous(answer, variable)
            return Candidate(answer, formed=True)
    finally:
        _substituting.reset(token)
    return None


def _sine_and_cosine(
    integrand: sp.Basic, variable: sp.Basic, nested: bool = False
) -> list[tuple]:
    """Candidates for `u` in `_by_derivative_divides` from a sine or cosine of a linear
    argument under a radical, as `u = COS(x)` for `SIN(x)/SQRT(1 - q^2*SIN(x)^2)`, or,
    if `nested`, inside another function, as `u = COS(t)` for
    `SIN(t)^3*COS(2*COS(t))`. Rational functions of the two are left to
    `engine.trigonometric`.
    """
    calls = integrand.atoms(sp.sin, sp.cos)
    if nested:
        inside = [
            outer
            for outer in integrand.atoms(sp.Function)
            if outer.args and any(argument.has(*calls) for argument in outer.args)
        ]
    else:
        inside = [
            power
            for power in integrand.atoms(sp.Pow)
            if power.exp.is_Rational
            and not power.exp.is_integer
            and power.base.has(*calls)
        ]
    if not calls or not inside:
        return []
    candidates = []
    for call in calls:
        (argument,) = call.args
        slope = sp.diff(argument, variable)
        if slope == 0 or slope.has(variable):
            continue
        sine, cosine = sp.sin(argument), sp.cos(argument)
        candidates += [(cosine, -slope * sine), (sine, slope * cosine)]
    return candidates


def _squares_as_complements(
    expression: sp.Basic, inner: sp.Basic, u: sp.Basic
) -> sp.Basic:
    """The expression with even powers of the sine written in terms of `u`, where `u`
    stands for the cosine of the same argument, or vice versa."""
    if not isinstance(inner, (sp.sin, sp.cos)):
        return expression
    other = (sp.cos if isinstance(inner, sp.sin) else sp.sin)(*inner.args)
    return expression.replace(
        lambda e: e.is_Pow and e.base == other and e.exp.is_even,
        lambda e: (1 - u**2) ** (e.exp / 2),
    )


#: Whether `_by_derivative_divides` is already integrating its substituted integral.
_substituting: ContextVar[bool] = ContextVar("_substituting", default=False)


def _by_products_as_sums(head: sp.Basic) -> Candidate | None:
    """An antiderivative of the integrand with products of sines and cosines rewritten as
    sums, or None.

    Sympy leaves `INT(SIN(a*x)*COS(a*x)/x, x)` unevaluated but answers the equivalent
    `INT(SIN(2*a*x)/x, x)`. Rational functions of sines and cosines alone are skipped,
    since sympy's power rules give the table's form there.
    """
    if _summing.get() or len(head.limits) != 1 or len(head.limits[0]) != 1:
        return None
    variable, integrand = head.limits[0][0], head.function
    if _of_sines_and_cosines(integrand, variable):
        return None
    products = [
        product
        for product in integrand.atoms(sp.Mul, sp.Pow)
        if len(
            [
                factor
                for factor in sp.Mul.make_args(product)
                for _ in range(_multiplicity(factor))
                if isinstance(factor.as_base_exp()[0], (sp.sin, sp.cos))
                and factor.as_base_exp()[0].has(variable)
            ]
        )
        > 1
    ]
    if not products:
        return None
    written = attempt(integrand, lambda f: sp.expand(TR8(f)))
    if written is None or written == integrand:
        return None
    token = _summing.set(True)
    try:
        done = integral(sp.Integral(written, variable))
    finally:
        _summing.reset(token)
    return None if done.has(sp.Integral) else Candidate(done, formed=True)


def _of_sines_and_cosines(integrand: sp.Basic, variable: sp.Basic) -> bool:
    """Whether the integrand is a rational function of sines and cosines, with the
    variable appearing nowhere else."""
    calls = [call for call in integrand.atoms(sp.sin, sp.cos) if call.has(variable)]
    stand_ins = [sp.Dummy() for _ in calls]
    written = integrand.xreplace(dict(zip(calls, stand_ins)))
    return not written.has(variable) and written.is_rational_function(*stand_ins)


def _multiplicity(factor: sp.Basic) -> int:
    """How many calls a factor counts as, e.g. two for `SIN(x)^2`."""
    exponent = factor.as_base_exp()[1]
    return int(exponent) if exponent.is_Integer and exponent > 0 else 1


#: Whether `_by_products_as_sums` is already integrating the rewritten sums.
_summing: ContextVar[bool] = ContextVar("_summing", default=False)


#: The functions integrated by parts. Their derivatives are algebraic, and so is what
#: is left to integrate.
_PARTED = (sp.asin, sp.acos, sp.atan, sp.acot, sp.log)


def _by_parts_over_a_polynomial(head: sp.Basic) -> Candidate | None:
    """An antiderivative by parts, as in `_by_parts`, where `P` is a polynomial and the
    answer has no `#i`, `ABS` or `SIGN`, or None.

    Sympy answers `INT(x*ATAN(x/a), x)` with `ABS` terms where parts gives
    `a^2*ATAN(x/a)/2`. For quotients, and where the remaining integral involves `#i` or
    a sign, sympy does better, so those are left to it.
    """
    candidate = _by_parts_with(head, over_a_polynomial=True)
    if candidate is None or candidate.value.has(sp.I, sp.Abs, sp.sign):
        return None
    return candidate


def _by_parts(head: sp.Basic) -> Candidate | None:
    """An antiderivative of `P(x)*f(x) + Q(x)` by parts, where `f` is an inverse
    trigonometric function or a logarithm, `P` and `Q` are rational, and `P` has a
    rational antiderivative `V`, or None.

    The answer is `V*f - INT(V*f') + INT(Q)`, with `f'` algebraic; sympy leaves
    `INT(ASIN(x/a)/x^2, x)` unevaluated. Only one level deep. It runs after sympy, but
    before the retries only if sympy answered, since on `INT(LN(x + SQRT(x^2 + 1))/x^2,
    x)` the retries would search for seconds in vain.
    """
    return _by_parts_with(head, over_a_polynomial=False)


def _by_parts_with(head: sp.Basic, over_a_polynomial: bool) -> Candidate | None:
    """An antiderivative by parts, as in `_by_parts`, requiring `P` to be a polynomial
    if `over_a_polynomial` is set, or None."""
    if _parting.get() or len(head.limits) != 1 or len(head.limits[0]) != 1:
        return None
    variable, integrand = head.limits[0][0], head.function
    kernels = {call for call in integrand.atoms(*_PARTED) if call.has(variable)}
    if len(kernels) != 1:
        return None
    (kernel,) = kernels
    marker = sp.Dummy("f")
    written = attempt(integrand, lambda f: sp.expand(f.xreplace({kernel: marker})))
    if written is None or written.has(kernel):
        return None
    factor, rest = sp.diff(written, marker), written.subs(marker, 0)
    if (
        factor == 0
        or factor.has(marker)
        or factor.is_rational_function(variable) is not True
        or rest.is_rational_function(variable) is not True
        or (over_a_polynomial and factor.is_polynomial(variable) is not True)
    ):
        return None
    primitive = rational_antiderivative(factor, variable)
    if primitive is None or primitive.is_rational_function(variable) is not True:
        return None
    slope = attempt(kernel, lambda f: sp.cancel(sp.together(sp.diff(f, variable))))
    if slope is None or any(call.has(variable) for call in slope.atoms(sp.Function)):
        return None
    token = _parting.set(True)
    try:
        left = [sp.Integral(forced(primitive * slope, sp.cancel), variable)]
        if rest != 0:
            left.append(sp.Integral(rest, variable))
        done = [integral(leftover) for leftover in left]
    finally:
        _parting.reset(token)
    if any(value.has(sp.Integral) for value in done):
        return None
    answer = primitive * kernel - done[0] + sum(done[1:], sp.S.Zero)
    return Candidate(answer, trusted=True, tabled=True)


#: Whether `_by_parts` is already integrating its remaining integrals.
_parting: ContextVar[bool] = ContextVar("_parting", default=False)


#: Inverse trigonometric functions `f` for which `f(b*x)/x` has no elementary
#: antiderivative; see `_an_arc_over_its_variable`.
_ARCS_OVER_THE_VARIABLE = (sp.asin, sp.acos, sp.atan, sp.acot)


def _an_arc_over_its_variable(head: sp.Basic) -> bool:
    """Whether the head is an indefinite integral of `c*f(b*x)/x` with `f` an inverse
    trigonometric function, whose antiderivative is non-elementary and on which sympy
    spends twenty seconds.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 1:
        return False
    variable, integrand = head.limits[0][0], head.function
    arcs = [
        factor
        for factor in sp.Mul.make_args(integrand)
        if isinstance(factor, _ARCS_OVER_THE_VARIABLE) and factor.has(variable)
    ]
    if len(arcs) != 1:
        return False
    (arc,) = arcs
    scale = sp.cancel(arc.args[0] / variable)
    return (
        not scale.has(variable)
        and scale.is_zero is not True
        and not (integrand * variable / arc).has(variable)
    )


def _derivatives(function: sp.Basic, variable: sp.Basic) -> list[sp.Basic]:
    """The derivative of a function of the variable as sympy writes it, plus, for
    tangent, cotangent and their hyperbolic forms, the tables' form over a square:
    `a/COS(a*x)^2` besides `a*(TAN(a*x)^2 + 1)`."""
    (argument,) = function.args
    inner = sp.diff(argument, variable)
    squares = {
        sp.tan: inner / sp.cos(argument) ** 2,
        sp.cot: -inner / sp.sin(argument) ** 2,
        sp.tanh: inner / sp.cosh(argument) ** 2,
        sp.coth: -inner / sp.sinh(argument) ** 2,
    }
    written = [squares[function.func]] if function.func in squares else []
    return [*written, sp.diff(function, variable)]


def _by_linearity(head: sp.Basic) -> sp.Basic:
    """An integral sympy left unevaluated, as a sum of integrals where its integrand is a
    disguised sum.

    Sympy integrates a literal sum term by term, but one non-integrable term sinks a
    quotient like `(x*G(x) + F(x))/G(x)`, which Derive splits. So a quotient with a sum
    numerator becomes one integral per term. A denominator polynomial in one sine or
    cosine is split into partial fractions over it, but only for indefinite integrals.

    The split is kept only if some piece is evaluated, so an integral that no method
    answers stays whole. A definite integral is split only if moreover none is
    infinite, since the pieces of a convergent integral need not converge: `INT((x^t -
    x^s)/LN(x), x, 0, 1)` converges, while its two pieces don't.
    """
    integrand = head.function
    if isinstance(integrand, sp.Add):
        return head
    numerator, denominator = integrand.as_numer_denom()
    if isinstance(numerator, sp.Add):
        terms = [term / denominator for term in numerator.args]
    else:
        terms = _partial_fractions(head, denominator)
    if len(terms) < 2:
        return head
    pieces = [head.func(term, *head.limits) for term in terms]
    values = [integral(piece) for piece in pieces]
    if all(value.has(sp.Integral) for value in values):
        return head
    if definite(head) and any(
        value.has(sp.oo, -sp.oo, sp.zoo, sp.nan)
        for value in values
        if not value.has(sp.Integral)
    ):
        return head
    return sp.Add(*pieces)


def _partial_fractions(head: sp.Basic, denominator: sp.Basic) -> list[sp.Basic]:
    """The integrand of an antiderivative as partial fractions over the single periodic
    function its denominator is a polynomial in, or [] if there is none."""
    if any(len(limit) != 1 for limit in head.limits):
        return []
    variables = [limit[0] for limit in head.limits]
    generator = in_one_periodic(denominator, variables)
    if generator is None:
        return []
    stand_in = sp.Dummy()
    written = head.function.subs(generator, stand_in)
    split = attempt(written, lambda w: sp.apart(w, stand_in))
    if split is None:
        return []
    return [term.subs(stand_in, generator) for term in sp.Add.make_args(split)]


def _by_a_radical(head: sp.Basic) -> Candidate | None:
    """The integral by substituting a square root of a polynomial that it contains, or
    None.

    `INT(SQRT(x^2 - 1)/x, x)` becomes `u - ATAN(u)` with `u = SQRT(x^2 - 1)`, valid on
    both sides of the interval where the root isn't real, where sympy's own methods
    split. The result must pass `holds`.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 1:
        return None
    variable, integrand = head.limits[0][0], head.function
    roots = {
        sp.sqrt(power.base)
        for power in integrand.atoms(sp.Pow)
        if power.exp.is_Rational
        and power.exp.q == 2
        and power.base.has(variable)
        and power.base.is_polynomial(variable)
    }
    for root in sorted(roots, key=sp.default_sort_key):
        substitute = sp.Dummy("u", positive=True)
        transformed = attempt(
            sp.Integral(integrand, variable), lambda h: h.transform(root, substitute)
        )
        if transformed is None:
            continue
        found = attempt(transformed, lambda h: h.doit(deep=False, **BOUNDED))
        if found is None or found.has(sp.Integral, sp.Piecewise):
            continue
        value = found.subs(substitute, root)
        if holds(value, integrand, variable):
            return Candidate(value, trusted=True)
    return None


def _by_the_bounded_methods(head: sp.Basic) -> Candidate | None:
    """An integral by sympy's bounded-cost methods, `BOUNDED`, or None.

    An answer still containing an unevaluated integral is offered only if the full
    methods are unaffordable. For a definite integral left unevaluated or depending on
    an unknown sign, the Meijer G route is also tried.
    """
    if _an_arc_over_its_variable(head) or _beside_a_composition(head):
        return None
    hints = within_reach(head)
    found = _standing(
        head, attempt(head, lambda h: h.doit(deep=False, **BOUNDED, **hints))
    )
    if found == head:
        found = None
    standing = found is None or found.has(sp.Integral)
    if standing and not _unaffordable(head):
        return None
    if definite(head) and (standing or unresolved(head, found)):
        tabled = _by_meijer_g(head)
        if tabled is not None:
            return Candidate(tabled, retries=_RETRIES)
    return None if found is None else Candidate(found, retries=_RETRIES)


def _by_meijer_g(head: sp.Basic) -> sp.Basic | None:
    """A definite integral with at most one parameter, by Meijer G functions, or None.

    This answers integrals like `INT(COS(a*x)*#e^(-x^2), x, -inf, inf)` that the
    bounded methods leave. With more parameters the search grows too much. An answer
    that still depends on an unknown sign gives None.
    """
    variables = {limit[0] for limit in head.limits}
    if len(head.function.free_symbols - variables) > 1:
        return None
    hints = {"heurisch": False, **within_reach(head)}
    tabled = _standing(head, attempt(head, lambda h: h.doit(deep=False, **hints)))
    if tabled is None or tabled == head:
        return None
    if not tabled.has(sp.Integral) and unresolved(head, tabled):
        return None
    return tabled


def _by_the_full_methods(head: sp.Basic) -> Candidate | None:
    """An integral by all of sympy's methods, including heuristic Risch, or None.

    Heuristic Risch builds an ansatz over all generators, and with parameters its cost
    explodes: an integral from the Kummer function in Derive's utility library takes
    most of a minute and gigabytes of memory before giving up. So for a parametrized
    integrand it runs only if `affordable` says so.
    """
    if (
        _an_arc_over_its_variable(head)
        or _beside_a_composition(head)
        or _unaffordable(head)
        or _elliptic(head)
    ):
        return None
    hints = within_reach(head)
    if _numerically.get():
        hints = {**hints, "heurisch": False}
    found = _standing(head, attempt(head, lambda h: h.doit(deep=False, **hints)))
    if found is None or found == head:
        return None
    return Candidate(found, retries=_RETRIES)


def _beside_a_composition(head: sp.Basic) -> bool:
    """Whether the integrand's denominator holds a sum in which a transcendental call of
    a non-polynomial argument stands beside a term that varies without it, as in
    `1/(SIN(1/y) + y^3)`, while the numerator holds no such call.

    Such an integral has no elementary antiderivative, since the derivative of the
    denominator, which a logarithm or a power of it would need, brings in the
    derivative of the call; sympy's methods search for most of a minute before giving
    up. With such a call in the numerator, `(3*y^2 - COS(1/y)/y^2)/(SIN(1/y) + y^3)`
    is `LN(SIN(1/y) + y^3)`, and the methods run. Only indefinite integrals of one
    variable are considered.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 1:
        return False
    variable = head.limits[0][0]
    try:
        numerator, denominator = head.function.as_numer_denom()
    except Exception:
        return False

    def composed(term: sp.Basic) -> bool:
        return any(
            call.args[0].has(variable)
            and call.args[0].is_polynomial(variable) is False
            for call in term.atoms(TrigonometricFunction, HyperbolicFunction, sp.exp)
        )

    if composed(numerator):
        return False
    for factor in sp.Mul.make_args(denominator):
        base = factor.as_base_exp()[0]
        if not base.is_Add:
            continue
        terms = sp.Add.make_args(base)
        if any(map(composed, terms)) and any(
            term.has(variable) and not composed(term) for term in terms
        ):
            return True
    return False


def _elliptic(head: sp.Basic) -> bool:
    """Whether the integrand is a rational function of `w = SIN(u)^2` times an odd power
    of the square root of `a + b*w`, for a linear `u` and with neither `a`, `b` nor `a +
    b` zero, so that the integral is elliptic and has no elementary antiderivative.

    ELLIPTIC.MTH defines its integrals this way, and on `INT(1/((1 - 3*SIN(t)^2/10)*
    SQRT(1 - SIN(t)^2/2)), t, 0, 1)` heuristic Risch searches for minutes.
    """
    variables = [limit[0] for limit in head.limits]
    if len(variables) != 1:
        return False
    (variable,) = variables
    integrand = head.function
    calls = {call for call in integrand.atoms(sp.sin, sp.cos) if call.has(variable)}
    arguments = {call.args[0] for call in calls}
    if len(arguments) != 1:
        return False
    (argument,) = arguments
    polynomial = attempt(argument, lambda a: sp.Poly(a, variable))
    if polynomial is None or polynomial.degree() != 1:
        return False
    w = sp.Dummy("w", positive=True)

    def squared(e: sp.Basic) -> bool:
        return e.is_Pow and e.base in calls and e.exp.is_even

    def in_w(e: sp.Basic) -> sp.Basic:
        square = w if isinstance(e.base, sp.sin) else 1 - w
        return square ** (e.exp / 2)

    written = attempt(integrand, lambda f: f.replace(squared, in_w))
    if written is None or written.has(variable):
        return False
    radicands = {
        power.base
        for power in written.atoms(sp.Pow)
        if power.base.has(w) and power.exp.is_Rational and power.exp.q == 2
    }
    if len(radicands) != 1:
        return False
    (radicand,) = radicands
    linear = attempt(radicand, lambda r: sp.Poly(r, w))
    if linear is None or linear.degree() != 1:
        return False
    b, a = linear.all_coeffs()
    if a == 0 or sp.expand(a + b) == 0:
        return False
    rooted = sp.Dummy("r", positive=True)
    rest = written.replace(
        lambda e: e.is_Pow and e.base == radicand and e.exp.is_Rational,
        lambda e: rooted ** (2 * e.exp),
    )
    exponents = {
        power.exp
        for power in rest.atoms(sp.Pow)
        if power.base == rooted and power.exp.is_Integer
    }
    return bool(exponents) and all(e.is_odd for e in exponents) and (
        rest.is_rational_function(w, rooted) is True
    )


def _unaffordable(head: sp.Basic) -> bool:
    """Whether the integrand has too many parameters for the full methods to afford."""
    return parametrised(head) and not affordable(head)


#: Hints for the bounded-cost methods: sympy's tables, the trigonometric rules, Risch,
#: and the manual rules. Excludes `meijerg`, which on its own answers
#: `INT(#e^x*COS(x), x)` with a page of Bessel functions, and `heurisch`, whose cost
#: is the problem described below.
BOUNDED = {"meijerg": False, "heurisch": False}


#: The maximum number of bases with symbolic exponents before the ansatz is too big.
#: With two, `INT(x^(a-1)*(1-x)^(b-1), x)`, sympy answers in under a second; with a
#: third, `INT(x^(a-1)*(1-x)^(c-a-1)*(1-x*z)^(-b), x)`, it doesn't finish. Measured
#: over every parametrized integral in Derive's utility library: under this rule each
#: either answers within a second or is refused, and no refusal loses an answer.
_ANSATZ = 2


def affordable(head: sp.Basic) -> bool:
    """Whether the heuristic method can be run on a parametrized integrand.

    It is cheap for products of powers of polynomials, where each base is one
    generator. Compositions like `#e^(z*COS(t))` don't finish, nor do three powers with
    symbolic exponents, so their count is bounded by `_ANSATZ`. A symbolic exponent is
    allowed only on a base with at most two terms; on a full quadratic the search takes
    minutes and fails.

    For indefinite integrals, an exponential, sine or cosine of a linear argument, a
    logarithm of a power of the variable, or a polynomial in one sine or cosine counts
    as one more generator against the same bound; a logarithm is allowed only alongside
    powers. For definite integrals these are excluded, since the cases sympy splits off
    would each have to be computed.
    """
    variables = [limit[0] for limit in head.limits]
    indefinite = all(len(limit) == 1 for limit in head.limits)
    symbolic = logarithms = 0
    for factor in sp.Mul.make_args(head.function):
        base, exponent = factor.as_base_exp()
        if indefinite and (
            _linear_generator(factor, variables)
            or (_linear_generator(base, variables) and exponent.is_Integer)
            or (in_one_periodic(base, variables) is not None and exponent.is_Integer)
        ):
            symbolic += 1
            logarithms += isinstance(base, sp.log)
            continue
        if exponent.has(*variables) or base.is_polynomial(*variables) is not True:
            return False
        if exponent.free_symbols:
            if len(sp.Poly(base, *variables).terms()) > 2:
                return False
            symbolic += 1
    transcendental = [
        factor
        for factor in sp.Mul.make_args(head.function)
        if factor.has(sp.exp, *_PERIODIC)
    ]
    if logarithms and transcendental:
        return False
    return symbolic <= _ANSATZ


#: Functions of a linear argument that the heuristic method can afford as generators.
_PERIODIC = (sp.sin, sp.cos, sp.sinh, sp.cosh)


def _linear_generator(factor: sp.Basic, variables: list) -> bool:
    """Whether the factor is an exponential, sine or cosine of an argument linear in the
    variable, or a logarithm of a power of the variable."""
    if isinstance(factor, sp.log):
        argument = factor.args[0]
        _, rest = argument.as_independent(*variables, as_Add=False)
        return len(variables) == 1 and rest.as_base_exp()[0] == variables[0]
    if not isinstance(factor, (sp.exp, *_PERIODIC)):
        return False
    argument = factor.args[0]
    try:
        return sp.Poly(argument, *variables).total_degree() == 1
    except sp.PolynomialError:
        return False


def in_one_periodic(base: sp.Basic, variables: list) -> sp.Basic | None:
    """The single sine or cosine (circular or hyperbolic) of a linear argument that `base`
    is a polynomial in, with the variable appearing nowhere else, or None.
    """
    generators = {
        atom for atom in base.atoms(*_PERIODIC) if _linear_generator(atom, variables)
    }
    if len(generators) != 1:
        return None
    (generator,) = generators
    stand_in = sp.Dummy()
    written = base.subs(generator, stand_in)
    if written.has(*variables) or written.is_polynomial(stand_in) is not True:
        return None
    return generator


@lru_cache(maxsize=256)
def within_reach(head: sp.Basic) -> dict[str, bool]:
    """Sympy hints for a definite integral: Risch and the manual rules disabled if they
    can't find an antiderivative, otherwise no hints.

    Over a positive variable the manual rules can search for minutes (`INT(LN(1 +
    x)/(x^2 + 1), x, 1, 2)`), while over an unassumed variable they give up at once, so
    that is checked first. Risch is disabled too, since sympy passes its non-elementary
    part to the manual rules. Rational functions are skipped, as Risch would use the
    slow `ratint`.
    """
    if not definite(head):
        return {}
    if head.function.is_rational_function(head.limits[0][0]) is True:
        return {}
    plain = sp.Dummy("x")
    integrand = head.function.subs(head.limits[0][0], plain)
    try:
        elementary, nonelementary = risch_integrate(
            integrand, plain, separate_integral=True
        )
    except Exception:
        return {}
    if elementary != 0 or nonelementary == 0:
        return {}
    manual = attempt(integrand, lambda f: manualintegrate(f, plain))
    if manual is None or manual.has(sp.Integral):
        return {"risch": False, "manual": False}
    return {}


def _standing(head: sp.Basic, value: sp.Basic | None) -> sp.Basic | None:
    """`value`, or the head itself if `value` is the same integral in another guise.

    Sympy returns a proved non-elementary integral as a `NonElementaryIntegral`, or an
    integral over a merely factored integrand. Neither compares equal to the head, so
    it would be mistaken for an answer.
    """
    if not isinstance(value, sp.Integral):
        return value
    if value.args == head.args or isinstance(value, NonElementaryIntegral):
        return head
    if value.limits == head.limits:
        if value.function == forced(head.function, sp.factor):
            return head
    return value


def _over_a_positive_variable(head: sp.Basic) -> Candidate | None:
    """An antiderivative by `by_a_positive_substitution`, or None. An `_elliptic`
    integrand is skipped, since sympy's search for it doesn't end."""
    if len(head.limits) != 1 or len(head.limits[0]) != 1 or _elliptic(head):
        return None
    value = by_a_positive_substitution(head.function, head.limits[0][0])
    return None if value is None else Candidate(value, trusted=True, tabled=True)


def by_a_positive_substitution(
    integrand: sp.Basic, variable: sp.Basic
) -> sp.Basic | None:
    """The integral over a new variable that is positive wherever the integrand is real,
    or None.

    Sympy pulls a real variable out of a square root as if it were positive, so
    `INT(1/(x*SQRT(x^2 + a^2)), x)` comes back valid only for positive `x`. A variable
    positive by construction avoids this:

    - if every root is of one linear `a*x + b`, the root `t = (a*x + b)^(1/q)` itself;
    - if the integrand is odd, `u = x^2`, then a root of a linear function of `u` as
      above;
    - if it is even, `x` assumed positive, keeping the result if proved odd, so it
      holds for negative `x` too.

    If the result is split over parameters, the last case is used alone if it is
    proved.
    """
    value = _positive_antiderivative(integrand, variable)
    if value is None or value.has(sp.Integral):
        return None
    folded = attempt(value, sp.piecewise_fold)
    if isinstance(folded, sp.Piecewise) and folded.args[-1].cond is sp.true:
        if holds(folded.args[-1].expr, integrand, variable):
            return folded.args[-1].expr
    return value


def _positive_antiderivative(
    integrand: sp.Basic, variable: sp.Basic
) -> sp.Basic | None:
    """An unchecked antiderivative by the substitutions in `by_a_positive_substitution`,
    or None if none applies."""
    radicands = {
        power.base
        for power in integrand.atoms(sp.Pow)
        if power.base.has(variable)
        and power.exp.is_Rational
        and not power.exp.is_Integer
    }
    if not radicands:
        return None
    substitute = sp.Dummy("u", positive=True)
    if len(radicands) == 1:
        (radicand,) = radicands
        polynomial = attempt(radicand, lambda r: sp.Poly(r, variable))
        if polynomial is not None and polynomial.degree() == 1:
            return _by_a_linear_root(integrand, variable, radicand, polynomial)
    mirrored = integrand.subs(variable, -variable)
    if attempt(mirrored + integrand, sp.cancel) == 0:
        halved = (integrand / variable).subs(variable, sp.sqrt(substitute)) / 2
        halved = forced(halved, sp.powsimp)
        if halved.has(variable):
            return None
        found = _positive_antiderivative(halved, substitute)
        return None if found is None else found.subs(substitute, variable**2)
    if attempt(mirrored - integrand, sp.cancel) == 0:
        positive = sp.Integral(integrand.subs(variable, substitute), substitute)
        if _unaffordable(positive):
            return None
        found = attempt(positive, lambda h: h.doit(deep=False))
        if found is None or not _odd(found, substitute):
            return None
        if not differentiates_back(found, positive.function, substitute):
            return None
        return found.subs(substitute, variable)
    return None


def _odd(expression: sp.Basic, variable: sp.Basic) -> bool:
    """Whether `f(x) + f(-x)` simplifies to something free of the variable."""
    balance = attempt(
        expression, lambda e: sp.simplify(e + e.subs(variable, -variable))
    )
    return balance is not None and not balance.has(variable)


def _by_a_linear_root(
    integrand: sp.Basic, variable: sp.Basic, radicand: sp.Basic, polynomial: sp.Poly
) -> sp.Basic | None:
    """The integral over `t = radicand^(1/q)`, where the radicand is linear in the
    variable and `q` is the common denominator of its exponents, or None."""
    order = sp.ilcm(
        1,
        *(
            power.exp.q
            for power in integrand.atoms(sp.Pow)
            if power.base == radicand and power.exp.is_Rational
        ),
    )
    root = sp.Dummy("t", positive=True)
    slope, offset = polynomial.all_coeffs()
    # radicand^(k/order) is root^k for the principal root, for any radicand.
    rooted = attempt(
        integrand,
        lambda f: f.replace(
            lambda e: e.is_Pow and e.base == radicand and e.exp.is_Rational,
            lambda e: root ** (e.exp * order),
        ),
    )
    transformed = attempt(
        rooted,
        lambda f: f.subs(variable, (root**order - offset) / slope)
        * order
        * root ** (order - 1)
        / slope,
    )
    if transformed is None or transformed.has(variable):
        return None
    found = rational_antiderivative(transformed, root)
    if found is None:
        return None
    return found.subs(root, radicand ** sp.Rational(1, order))


def _by_the_manual_rules(head: sp.Basic) -> Candidate | None:
    """An antiderivative by sympy's manual rules alone, or None. It avoids the algorithm
    the wrong answers come from; one of `_RETRIES`.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 1:
        return None
    manual = attempt(head, lambda h: h.doit(deep=False, manual=True))
    if manual is None or manual.has(sp.Integral):
        return None
    return Candidate(manual, tabled=True)


def _over_any_variable(head: sp.Basic) -> Candidate | None:
    """The integral redone over a variable with no assumptions, or None.

    For real `x`, sympy's Meijer G route gets `INT(SIN(x)/x^2, x)` wrong for negative
    `x`, but over a plain symbol it is right. One of `_RETRIES`, skipped when the full
    methods are unaffordable.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 1:
        return None
    if _unaffordable(head):
        return None
    variable, integrand = head.limits[0][0], head.function
    plain = sp.Dummy("x")
    again = attempt(
        sp.Integral(integrand.subs(variable, plain), plain),
        lambda h: h.doit(deep=False),
    )
    if again is None or again.has(sp.Integral):
        return None
    return Candidate(again.subs(plain, variable), tabled=True)


#: Methods tried, in order, after an answer from sympy's general integrators is
#: refused.
_RETRIES: tuple[Method, ...] = (_by_the_manual_rules, _over_any_variable)

#: The integration methods, in the order tried. Each takes the head and returns a
#: `Candidate` or None, and sets its `trusted`, `formed` and `retries` fields.
_METHODS: tuple[Method, ...] = (
    _between_the_ends_of_a_power,
    _of_a_binomial_power,
    _of_a_power_times_its_derivative,
    _of_a_rational_function,
    _of_a_trigonometric_function,
    _of_a_radical,
    _over_an_exponential,
    _of_an_exponential_times_a_wave,
    _by_derivative_divides,
    _by_products_as_sums,
    _by_parts_over_a_polynomial,
    _over_a_positive_variable,
    _by_a_radical,
    _by_the_bounded_methods,
    _by_the_full_methods,
    _by_parts,
)

#: Sympy's methods, in `_METHODS` order, none of them trusted.
_SYMPY: tuple[Method, ...] = (_by_the_bounded_methods, _by_the_full_methods)
