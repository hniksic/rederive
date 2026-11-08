"""An integral, asked of the methods that may answer it, in order.

`integral` is where every integral comes in. `_METHODS`, at the foot of the module, is
the whole order the methods are asked in, and `_accepted` is the one place where what a
method answers is taken or refused. What is peculiar to an integral between limits is
`definite`'s, and the form an answer taken here is printed in is `form`'s.

A method answers with a `Candidate`, or with None where it does not apply, and the first
answer `_accepted` takes is the integral's. The methods that are right by construction
come first. They are the integrands done here by a construction of their own -
rational, trigonometric, radical, exponential - and then the rules a calculus course
teaches: the substitution `u = g(x)` where the integrand is `h(g(x))*g'(x)`, products of
sines and cosines written as sums, parts over an arc or a logarithm, a variable that is
positive by construction, the substitution of a square root. Each writes its answer the
way the integral tables do, and over the parameters with no split.

Sympy's general integrators come after them, for what they do not reach, which is mostly
the integrals whose answers are special functions, `SI(x)`, `EI(x)`, `ERF(x)` and
`POLYLOG(2, x)`, and the definite integrals the tables of Meijer G functions answer.
Asked first, they would answer much of the rest too, but more slowly, split over the
values of the variable or the parameters, and now and then wrongly: so an answer of
sympy's is taken only where it is proved to differentiate back to the integrand, and no
answer is taken in a function the notation has no word for. Where a rule of the course
writes a longer answer than sympy's for some kind of integrand, the rule's own guard
says so: products as sums leave the powers of sines and cosines to sympy, and parts
over a quotient is asked after it.

Two more of sympy's methods are no entries of the list but its retries: the manual
rules alone, and the whole integral again over a variable nothing is assumed of. They
answer what sympy's general integrators answered wrongly or in a function the notation
has no word for, `INT(SIN(x)/x^2, x)` among them, and are asked only once one of those
answers has been refused. Where sympy found nothing they would find nothing either, and
an integral that stands, `INT(LN(x)^2*ATAN(x), x)`, would spend seconds on each before
standing anyway.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from functools import lru_cache
from typing import NamedTuple

import sympy as sp
from sympy.integrals.manualintegrate import manualintegrate
from sympy.integrals.risch import NonElementaryIntegral, risch_integrate
from sympy.simplify.fu import TR8

from rederive.engine.integration.definite import (
    CONVERGENT,
    UNDECIDED,
    VERDICTS,
    across_the_steps,
    at_the_singularities,
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
    with_moving_poles,
)
from rederive.engine.integration.form import (
    anywhere,
    antiderivative,
    continuous,
    on_the_principal_sheet,
    parametrised,
    single_case,
    somewhere_only,
    unwritable,
)
from rederive.engine.integration.proof import differentiates_back, holds
from rederive.engine.radical import radical_antiderivative
from rederive.engine.rational import rational_antiderivative
from rederive.engine.rewriting import attempt, forced
from rederive.engine.trigonometric import trigonometric_antiderivative

__all__ = ["by_a_positive_substitution", "by_sympy", "integral", "within_reach"]


@lru_cache(maxsize=256)
def integral(head: sp.Basic) -> sp.Basic:
    """An integral, by sympy, with what sympy gets wrong about definite ones put right.

    A definite integral is split where a step function in it jumps, and looked at
    where its integrand blows up, before sympy is asked: a step is something sympy
    integrates over only in part, and a singularity inside the interval is one it
    reads straight through, as an antiderivative taken across it would. What sympy
    then answers under a condition on a parameter, or with a parameter whose sign it
    did not know, is asked again on each side of that sign.

    The answer is a function of the head alone, and is kept: an integral that stands
    is asked again by the pass that simplifies the answer, and by the next command
    over the line, and asking sympy costs seconds where a lookup costs nothing.
    """
    stepped = between_the_steps(head)
    if stepped is not None:
        return stepped
    across = across_the_steps(head)
    if across is not None:
        return across
    value = _integral_through(head)
    moving = with_moving_poles(head, value)
    return real_where_real(head, value if moving is None else moving)


def _integral_through(head: sp.Basic) -> sp.Basic:
    """An integral, by its methods, once its steps have been looked at: see `integral`.

    What the methods make of a definite integral is put to the verdicts of `definite`
    around them. What still stands after those is split by `_by_linearity` into
    integrals of its terms, outside the methods and after the verdicts: an integral
    that diverges, or turns on the sign of a parameter, is answered as a whole before
    it is taken apart.
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
    """An answer a method offers, what `_accepted` is to make of it, and what is asked
    where `_accepted` refuses it."""

    value: sp.Basic
    #: Right by construction, and not put to the proof.
    trusted: bool = False
    #: Written the way the integral tables write it already, or by the manual rules
    #: themselves: the manual rules are not asked for a shorter one.
    tabled: bool = False
    #: Made of answers taken already, and in their form: taken as it stands.
    formed: bool = False
    #: The methods asked, in order, where this answer is refused: what answers an
    #: integral that sympy answered wrongly need not be asked of one it did not answer.
    retries: tuple[Method, ...] = ()


#: What a method is: a head in, an answer offered or None out.
Method = Callable[[sp.Basic], Candidate | None]


def _answered(head: sp.Basic, methods: tuple[Method, ...]) -> sp.Basic | None:
    """The first answer `_accepted` takes, in its form, or None where none is taken.

    An answer refused is as good as none, but for its `retries`: those are asked in
    turn, each answer of theirs put to the same gate, and the first taken is the
    integral's. With none taken the methods after it are asked. A retry is asked at
    most once, though the bounded and the full methods may both have an answer
    refused: it is asked of the head and not of the answer, and would search the same
    seconds again for what it did not find the first time. An answer that is `formed`
    already is taken as it stands, and any other in the form `antiderivative` chooses.
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
    """The answer a method offers, as it is taken, or None where it is refused.

    Nothing of sympy's is trusted: an answer of sympy's is taken only where its
    derivative is proved to be the integrand, each case of a split where it applies as
    `differentiates_back` proves it, or else once the split has been narrowed to the
    case `single_case` proves and keeps. One that is not is refused, and with nothing
    after it to answer the integral stands: unevaluated is an answer, an
    antiderivative nobody proved is not. A `trusted` answer is right by construction
    and is not put to the proof, which on the derivative of `ATAN(SQRT(b^2 +
    c^2)*TAN(a*x)/ABS(b))` takes longer than the integral did.

    An answer in a function of the variable the notation has no word for is refused,
    trusted or not, and before it is proved: it would be declined in the end anyway,
    proving it runs `sympy.simplify` for seconds over a hypergeometric series, making
    `INT(1/(x*(a*x + b)^n), x)` continuous in `n` takes minutes over the Lerch function
    sympy writes it in.

    Before either, an answer is taken back to the principal sheet where it strayed off
    it, and one that says nothing for some values of the variable is narrowed to its one
    case that holds for every value, and refused where there is no such case.

    A definite integral's answer, a multiple integral's, one with an integral left
    standing in it and one `formed` already are taken as they are, there being no
    antiderivative in hand to prove.
    """
    value = candidate.value
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
    """An integral by sympy's methods alone, `_SYMPY`, asked as `integral` asks them.

    `definite` asks this of the integrals it makes up on the way to a verdict, which it
    wants as sympy answers them.
    """
    value = _answered(head, _SYMPY)
    return head if value is None else value


def _of_a_rational_function(head: sp.Basic) -> Candidate | None:
    """An integral of a rational function of its variable, done here, or None.

    Sympy's rational integrator loses terms, as `engine.rational` explains, so a
    rational integrand is not offered to it: the antiderivative is that module's,
    and goes through the same choices as any other. A definite integral of one is
    the difference of that antiderivative's limits at the ends, approached from
    inside, which is Derive's way with every definite integral (7.4, p.199) and the
    same difference sympy would take with an antiderivative that was whole. A pole
    between the ends or at one of them, where the limits are numbers, has been
    looked at by `at_the_singularities` before this is asked; where the ends come
    to nothing, or the antiderivative was not had, sympy is asked after all - but
    not of an integral with no parameter in it, its integrand or its ends, whose
    antiderivative is had always: sympy would answer from the same `ratint` that loses the roots, and
    answers `INT(1/(x^8 + 1), x, 0, 1)` with 0, or search for minutes for the real
    roots of a resolvent, as it does over `1/(x^5 + x + 1)`. That integral stands.

    Only an integrand rational as a whole is taken: a rational term in a sum with a
    transcendental one goes to sympy with the rest, and what sympy makes of it has to
    be proved like any other answer of sympy's. The antiderivative made here is right
    by construction and is not put to that proof.
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
        # `INT(1/x, x, 1, x)` is the antiderivative from 1 to `x`: the variable of
        # integration is another than the one the end is written in.
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
    """The antiderivative of a rational function, written for its limit at an end.

    A `RootSum` is a sum of `a*LN(h(a, x))` over the roots `a` of a polynomial, and
    sympy takes the limit of no expression with one in it. Where each `h` is linear
    in `x` with a leading coefficient of 1, each logarithm is of `x` less a root of
    the integrand's denominator, which is either not real or is a pole, and so the
    sum is continuous: at a finite end it is put in at its value there. At `inf`, see
    below, each logarithm tends to `LN(1)`, and the sum to nothing. At `-inf` each
    tends to `LN(-1)`, which is `#i*pi` or `-#i*pi` as the root's imaginary part has
    one sign or the other, and the sum is left to stand, as it is at any end where
    an `h` is of another form.

    At an infinite end where the integrand falls off as the square of the reciprocal
    of the variable or faster, so that the integral converges there, each `LN(p)` of a
    polynomial of degree `d` is written as `LN(p/x^d)`, which tends to a number. The
    two differ by `d*LN(x)`, `LN(ABS(x))` at `-inf`, and those add up to nothing:
    the antiderivative has a limit there, and nothing else in it grows. The limit
    of the sum as it stands is not always had: the coefficients of `LN(x)` in the
    logarithms of `INT(1/(x^7 + 1), x)` cancel only by identities of the cosines of
    multiples of `pi/7`, which sympy does not see, and it takes the limit at `inf`
    for `inf`.

    Anywhere else, and where a logarithm in it is of no polynomial, the
    antiderivative is as it stands.
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
    # The logarithms outside the sums: those inside are over the sums' own variables.
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
    """Whether the `RootSum` is of `a*LN(h(a, x))`, each `h` linear in the variable
    with a leading coefficient of 1."""
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
    argument, done here by `engine.trigonometric`, or None.

    Sympy's own ways with these spend up to minutes over parametric coefficients,
    `INT(1/(b + c*SIN(a*x))^2, x)` among them, and answer with a split over the values
    of the parameters for which the generic formula fails. The substitution the
    integral tables use makes a rational function of these, and the rational
    integrator writes its parametric quadratics without roots or cases.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 1:
        return None
    variable = head.limits[0][0]
    primitive = trigonometric_antiderivative(head.function, variable)
    if primitive is None:
        return None
    return Candidate(primitive, trusted=True, tabled=True)


def _of_a_radical(head: sp.Basic) -> Candidate | None:
    """An antiderivative of a rational function of the variable and roots of linear
    or quadratic polynomials in it, done here by `engine.radical`, or None.

    Sympy's own ways with these spend minutes over parametric coefficients, as on
    `(a*x^2 + b*x + c)^(7/2)` or `1/(SQRT(a*x + b)*(f*x + g)^3)`, and answer with a
    split over the values of the parameters, or not at all. An integral left in the
    answer is integrated as a head of its own; where one of them stands, or is the
    head itself, this is None, and the head goes the general way.
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
    """An antiderivative of a rational function of exponentials of multiples of one
    linear argument, by the substitution `u = #e^(g*x)`, or None.

    The Risch algorithm writes the logarithmic part of such an integral over the roots
    of a polynomial whose coefficients carry the parameters, and where it cannot tell
    real roots from complex ones leaves the sum over them standing: `INT(1/(b*#e^(a*x)
    + c*#e^(-a*x)), x)` comes back as `ROOT_SUM(4*b*c*z^2 + 1, z, z*LN(#e^(a*x) +
    2*c*z))/a`. Under `u = #e^(a*x)` that is the integral of `1/(a*(b*u^2 + c))`, which
    the rational integrator writes as the table does, `ATAN(SQRT(b)*u/SQRT(c))/(a*
    SQRT(b)*SQRT(c))`. Nor is what the Risch algorithm makes of these always the same
    from one run to the next: it takes `#e^(a*x)` or `#e^(-a*x)` for its generator by
    the order of a set, and `INT(COTH(a*x)^2, x)` comes back as `x - 2/(a*#e^(2*a*x) -
    a)` or as `x + 2/(a*#e^(-2*a*x) - a)` by the seed of the process's string hashes.
    The substitution here is chosen by the exponents alone.

    Each exponent is `k*x + m` with the ratios of the `k` rational numbers; `g` is the
    largest of them that each `k` is a whole multiple of, and `#e^(k*x + m)` is
    `#e^m*u^(k/g)`. None where something else in the integrand depends on the variable,
    or the denominator of the rational function of `u` has an irreducible factor of
    degree above 2: the logarithmic part over the roots of a cubic is a page of nested
    radicals that the steps after this take minutes over.
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


def _by_derivative_divides(head: sp.Basic) -> Candidate | None:
    """An antiderivative over `u = g(x)`, where the integrand is `h(g(x))*g'(x)` and
    holds a power whose exponent is a symbol, or None.

    Sympy finds `u = SIN(a*x)` in `SIN(a*x)^n*COS(a*x)`, but not `u = x^n` in
    `1/(x*SQRT(x^n + a^2))`, nor `u = TAN(a*x)` in `TAN(a*x)^n/COS(a*x)^2`, as it
    writes the derivative `a*(TAN(a*x)^2 + 1)`. The candidates for `g` are each such
    power with a number added to its exponent or multiplying it taken off, `x^n` for
    `x^(n - 1)` or `x^(2*n)`, and its base where that is a function, the derivative
    of a tangent or cotangent also written over the square of the cosine or sine. Where
    the integrand over `g'` is free of `x` once `g` is `u`, `INT(h(u), u)` is
    integrated as a head of its own, and its answer `H(u)` gives `H(g(x))`.

    That answer keeps the form the integral in `u` has, but for a split over a
    parameter, `IF(n /= -1, u^(n + 1)/(n + 1), LN(u))`, which is made continuous as
    `continuous` makes it. A pole not split over is left: the `1/n` the derivative
    of `x^n` puts into `2*ATAN(SQRT(x^n - a^2)/a)/(a*n)` is the table's, and its
    principal part, `2*ATAN(SQRT(1 - a^2)/a)/(a*n)`, is imaginary for `a` past one.
    One substitution deep only: the integral in `u` is not taken apart again.

    The answer is an antiderivative by construction, the integral in `u` having been
    answered under the same rules as any other head, and is not checked again.
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
    candidates = list(dict.fromkeys(candidates))
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
            if substituted is None or substituted.has(variable):
                continue
            if isinstance(inner, (sp.sin, sp.cos)) and substituted.atoms(sp.Function):
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


def _sine_and_cosine(integrand: sp.Basic, variable: sp.Basic) -> list[tuple]:
    """The candidates for `u` in `_by_derivative_divides` that a sine or cosine of a
    linear argument gives, over a radical in it: `u = COS(x)` for `SIN(x)/SQRT(1 -
    q^2*SIN(x)^2)`, the even powers of the sine being written in the cosine. A
    rational function of the two is `engine.trigonometric`'s, and is not looked at."""
    calls = integrand.atoms(sp.sin, sp.cos)
    if not calls or not any(
        power.exp.is_Rational and not power.exp.is_integer and power.base.has(*calls)
        for power in integrand.atoms(sp.Pow)
    ):
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
    """An expression with the even powers of the sine written in `u` where `u` stands
    for the cosine of the same argument, and the other way round."""
    if not isinstance(inner, (sp.sin, sp.cos)):
        return expression
    other = (sp.cos if isinstance(inner, sp.sin) else sp.sin)(*inner.args)
    return expression.replace(
        lambda e: e.is_Pow and e.base == other and e.exp.is_even,
        lambda e: (1 - u**2) ** (e.exp / 2),
    )


#: Whether `_by_derivative_divides` is integrating over its substitution already.
_substituting: ContextVar[bool] = ContextVar("_substituting", default=False)


def _by_products_as_sums(head: sp.Basic) -> Candidate | None:
    """An antiderivative of the integrand with its products of sines and cosines
    written as sums, or None.

    Sympy leaves `INT(SIN(a*x)*COS(a*x)/x, x)` standing, and answers `INT(SIN(2*a*x)/x,
    x)`, the same integral with the product written as a sine of the double angle,
    with `SI(2*a*x)`. The integral with the products written as sums is integrated as
    a head of its own, once: the sums are not taken apart again. The two integrands
    are the same function, so its answer is this one's.

    Not over a rational function of sines and cosines alone, which sympy's rules for
    their powers integrate the table's way: `INT(SIN(a*x)^3, x)` is `(COS(a*x) +
    2)*(COS(a*x) - 1)^2/(3*a)` by `u = COS(a*x)`, where the sum `3*SIN(a*x)/4 -
    SIN(3*a*x)/4` gives a cosine of the triple angle.
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
    """Whether the integrand is a rational function of sines and cosines of the
    variable, the variable appearing nowhere else."""
    calls = [call for call in integrand.atoms(sp.sin, sp.cos) if call.has(variable)]
    stand_ins = [sp.Dummy() for _ in calls]
    written = integrand.xreplace(dict(zip(calls, stand_ins)))
    return not written.has(variable) and written.is_rational_function(*stand_ins)


def _multiplicity(factor: sp.Basic) -> int:
    """How many calls a factor of a product is, two for `SIN(x)^2`."""
    exponent = factor.as_base_exp()[1]
    return int(exponent) if exponent.is_Integer and exponent > 0 else 1


#: Whether `_by_products_as_sums` is integrating the sums already.
_summing: ContextVar[bool] = ContextVar("_summing", default=False)


#: The functions an integrand is integrated by parts over: their derivatives are
#: algebraic, and so is what is left to integrate once they are differentiated.
_PARTED = (sp.asin, sp.acos, sp.atan, sp.acot, sp.log)


def _by_parts_over_a_polynomial(head: sp.Basic) -> Candidate | None:
    """An antiderivative by parts, as `_by_parts` makes it, where `P` is a polynomial
    and the answer is written without `#i`, `ABS` or `SIGN`; or None.

    Sympy answers `INT(x*ATAN(x/a), x)` with `a^3*ATAN(x/ABS(a))/(2*ABS(a))` among its
    terms, where by parts it is `a^2*ATAN(x/a)/2`. Over a quotient sympy does better:
    `V*f'` has a pole of its own where `V` has, and for `INT(ATAN(x/a)/x^2, x)` the
    integral left over is written `LN(x)/a - LN(x^2 + a^2)/(2*a)`, which sympy writes
    together as the table does, `-LN(a^2/x^2 + 1)/(2*a)`. So it does where the integral
    left over is written over `#i` or a sign, as that of `INT(x*ASIN(x/a), x)` is. Those
    are left to sympy, and to `_by_parts` after it.
    """
    candidate = _by_parts_with(head, over_a_polynomial=True)
    if candidate is None or candidate.value.has(sp.I, sp.Abs, sp.sign):
        return None
    return candidate


def _by_parts(head: sp.Basic) -> Candidate | None:
    """An antiderivative of `P(x)*f(x) + Q(x)` by parts, where `f` is an arc or a
    logarithm, `P` and `Q` are rational and `P` has a rational antiderivative `V`, or
    None.

    The answer is `V*f - INT(V*f') + INT(Q)`, `f'` being algebraic: sympy leaves
    `INT(ASIN(x/a)/x^2, x)` standing, and does not get past the two roots of the
    derivative of `ACOSH(x/a)` written as the logarithm of 6.6, where the integral of
    `x/(a*SQRT(x/a - 1)*SQRT(x/a + 1))` is the product of the roots again. Where `V`
    has a logarithm in it, `V*f'` is not algebraic and the head stands. So does it
    where an integral left over stands. One step deep only.

    Asked after sympy's bounded and full methods, as `_by_parts_over_a_polynomial` says
    why. Their retries, the manual rules and the unassumed variable, come before it
    only where sympy answered and was refused: where it found nothing they are not
    asked, and over `INT(LN(x + SQRT(x^2 + 1))/x^2, x)` they would search for seconds
    and find nothing, where this answers at once.
    """
    return _by_parts_with(head, over_a_polynomial=False)


def _by_parts_with(head: sp.Basic, over_a_polynomial: bool) -> Candidate | None:
    """An antiderivative by parts, as `_by_parts` says, `P` a polynomial where
    `over_a_polynomial` is set; or None."""
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


#: Whether `_by_parts` is integrating what it left over already.
_parting: ContextVar[bool] = ContextVar("_parting", default=False)


#: The arcs that have no elementary antiderivative over their variable: see
#: `_an_arc_over_its_variable`.
_ARCS_OVER_THE_VARIABLE = (sp.asin, sp.acos, sp.atan, sp.acot)


def _an_arc_over_its_variable(head: sp.Basic) -> bool:
    """Whether the head is an indefinite integral of `c*f(b*x)/x`, `f` an arc.

    Its antiderivative is a dilogarithm or a Clausen function, which is not
    elementary, and sympy spends twenty seconds on `INT(ATAN(x/a)/x, x)` before it
    says so.
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
    """The derivative of a function of the variable, written as sympy writes it and,
    for a tangent or cotangent and their hyperbolic kin, over a square as the tables
    write it: `a/COS(a*x)^2` beside `a*(TAN(a*x)^2 + 1)`."""
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
    """An integral sympy left standing, as a sum of integrals where its integrand is a
    sum in disguise.

    Sympy integrates a sum term by term, keeping as an integral only the terms it
    cannot do - but only a sum it is handed. `(x*G(x) + F(x))/G(x)` is a quotient to
    it, and the whole integral is given up for the one term that will not integrate.
    Derive divides the numerator into its terms and answers `INT(F(x)/G(x), x) +
    x^2/2`. So a quotient whose numerator is a sum is written as one integral per
    term, and those are new heads: each is integrated, or left standing, on its own,
    and none is the integral already tried whole.

    A quotient whose denominator is a polynomial in one sine or cosine of a linear
    argument is split into partial fractions over that function, so that one
    nonelementary piece does not take the rest down with it: of `INT(x/((SIN(a*x) +
    1)*SIN(a*x)), x)`, `INT(x/(SIN(a*x) + 1), x)` is integrated and only
    `INT(x/SIN(a*x), x)` is left standing. Only for an antiderivative, as the pieces of a definite integral that
    converges need not converge.

    For the same reason a definite integral's numerator is split only where some of
    the pieces have a value and none of those is infinite: neither `INT(x^t/LN(x), x,
    0, 1)` nor `INT(x^s/LN(x), x, 0, 1)` converges, while `INT((x^t - x^s)/LN(x), x,
    0, 1)` does. Where one piece has a finite value, the rest converge if the whole
    does.

    An integrand that was a sum to begin with has had its terms tried by sympy
    already, and is not offered again.
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
    if definite(head):
        values = [integral(piece) for piece in pieces]
        if all(value.has(sp.Integral) for value in values) or any(
            value.has(sp.oo, -sp.oo, sp.zoo, sp.nan)
            for value in values
            if not value.has(sp.Integral)
        ):
            return head
    return sp.Add(*pieces)


def _partial_fractions(head: sp.Basic, denominator: sp.Basic) -> list[sp.Basic]:
    """The integrand of an antiderivative as its partial fractions over the one
    periodic function its denominator is a polynomial in, or [] where it is not."""
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
    """The integral by the substitution of a square root of a polynomial in it, or None.

    `INT(SQRT(x^2 - 1)/x, x)` is `INT(u^2/(u^2 + 1), u)` over `u = SQRT(x^2 - 1)`,
    which is `u - ATAN(u)`; written back, `SQRT(x^2 - 1) - ATAN(SQRT(x^2 - 1))` is an
    antiderivative for `x` on either side of the interval where the root is not real.
    Sympy's own methods answer it over a split at that interval, or under the condition
    that `x` is outside it, and Derive prints no split where one antiderivative serves
    everywhere.

    Sympy makes the substitution only where the integrand is a function of the root and
    of the root's derivative, and the result is put to the proof `holds` makes, so it is
    taken only where it differentiates back for every `x`.
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
    """An integral by sympy's methods whose cost is bounded, `BOUNDED`, or None.

    An answer with an integral of what they could not do left standing in it is
    offered only where the full methods are not affordable: where they are, they are
    asked for the whole of it instead. Not over an arc over its variable, which no
    method of sympy's has an elementary antiderivative for.

    Between limits, an integral they leave standing, or answer only under a sign they
    do not know, is asked of the Meijer G route as well, and its answer is offered
    where it is better, as `_by_meijer_g` says.

    An answer refused is asked again of `_RETRIES`, as of any of sympy's methods.
    """
    if _an_arc_over_its_variable(head):
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
    """A definite integral over at most one parameter by Meijer G functions, or None.

    The bounded methods answer `INT(x^n*#e^(-x), x, 0, inf)` with `uppergamma(n + 1,
    0)`, which is `GAMMA(n + 1)` where `n > -1` and nothing elsewhere, and leave
    `INT(COS(a*x)*#e^(-x^2), x, -inf, inf)` standing. Sympy's Meijer G route knows
    the tables of definite integrals these come out of, writes the condition it
    needs, and over a single parameter answers quickly; over more of them it is
    not asked, its search growing with each. An answer that is no better - that
    still turns on a sign it does not know, or is the integral itself - is None.
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
    """An integral by every method of sympy's, the heuristic Risch algorithm among
    them, or None.

    The heuristic Risch algorithm is what finds the integrals nothing else does, and
    its cost grows with the generators of the integrand: it builds an ansatz over every
    one it can see and solves for the coefficients. With nothing symbolic to carry it
    is quick. `INT(t^(a-1)*(#e^(z*t)*(1-t)^(b-a-1) - 1), t, 0, 1/2)` - a real line out
    of the Kummer function in Derive's own utility library - carries three parameters
    over four generators, and sympy spends most of a minute and gigabytes of memory
    before answering, correctly, that it does not know. Simplify must not do that: a
    minute is not an answer, and the same integral comes back unevaluated either way.

    So over a parametrised integrand the expensive method is asked only where the
    ansatz is small enough to be affordable, which `affordable` is the measured rule
    for. Everything else comes back unevaluated, which is a documented answer where a
    minute of thrashing is not.

    An answer refused is asked again of `_RETRIES`, as of any of sympy's methods.
    """
    if _an_arc_over_its_variable(head) or _unaffordable(head):
        return None
    hints = within_reach(head)
    found = _standing(head, attempt(head, lambda h: h.doit(deep=False, **hints)))
    if found is None or found == head:
        return None
    return Candidate(found, retries=_RETRIES)


def _unaffordable(head: sp.Basic) -> bool:
    """Whether the integrand is parametrised past what the full methods can afford."""
    return parametrised(head) and not affordable(head)


#: Integration by the methods whose cost is bounded: sympy's tables, the
#: trigonometric rules, Risch, and the manual rules. Not `meijerg`, which
#: answers `INT(#e^x*COS(x), x)` with a page of Bessel functions when it is
#: reached on its own, and not `heurisch`, whose cost is the problem below.
BOUNDED = {"meijerg": False, "heurisch": False}


#: How many bases may carry a symbolic exponent before the ansatz is too big.
#: Two is `INT(x^(a-1)*(1-x)^(b-1), x)`, which sympy answers in under a second;
#: `INT(x^(a-1)*(1-x)^(c-a-1)*(1-x*z)^(-b), x)` is the same shape with a third,
#: and does not finish. Measured over every parametrised integral in Derive's
#: utility library: under this rule each one either answers within a second or
#: is refused, and no refusal loses an answer the rule would have found.
_ANSATZ = 2


def affordable(head: sp.Basic) -> bool:
    """Whether the heuristic method can be asked about a parametrised integrand.

    Cheap where the integrand is algebraic - a product of powers of polynomials
    in the variable - because each base is one generator and nothing composes
    to make more. `#e^(z*COS(t))*COS(n*t)` is where that stops being true: the
    exponential of a cosine is a generator built on another, and sympy does not
    finish. Neither does a product of three powers with symbolic exponents, so
    the count is bounded as well.

    A symbolic exponent is taken only over a base of at most two terms in the
    variable, `a*x + b` or `a^2 + x^2`. Over a full quadratic the search comes back
    empty-handed, and slowly: `INT(1/(a*x^2 + b*x + c)^n, x)` takes over a minute to
    stand, and even `INT(1/(x^2 + x + 1)^n, x)` takes seconds.

    An exponential, sine or cosine of a linear argument is one generator more, with
    nothing built on it, and so is a logarithm of a power of the variable: `INT(#e^(a*x)
    *COS(b*x), x)` and `INT(x^n*LN(x), x)` are answered in a fraction of a second. They
    count against the same bound as the symbolic exponents, a third of either kind
    being where the ansatz grows past a few seconds, and a logarithm is taken only
    beside powers, a logarithm times a cosine being past that as well. A polynomial
    in one sine or cosine of a linear argument, `SIN(a*x) + 1`, is that same generator
    under a rational function, and counts as it does: `INT(x/(SIN(a*x) + 1), x)` is
    answered in a fraction of a second. Only for an
    antiderivative: between limits, the cases sympy splits these into for the
    values of the parameters where the generic formula fails are no constants of
    integration, and each has to be had and printed.
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


#: The functions of a linear argument the heuristic method affords a generator for.
_PERIODIC = (sp.sin, sp.cos, sp.sinh, sp.cosh)


def _linear_generator(factor: sp.Basic, variables: list) -> bool:
    """Whether the factor is an exponential, sine or cosine of an argument linear in
    the variable, or a logarithm of a power of it."""
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
    """The one sine, cosine or hyperbolic one of an argument linear in the variable
    that `base` is a polynomial in, with the variable appearing nowhere else, or None."""
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
    """The hints for sympy on a definite integral: the Risch algorithm and the manual
    rules switched off where they cannot find an antiderivative, and none otherwise.

    Sympy values a definite integral over an interval right of zero by an
    antiderivative over a positive variable, and its manual rules, which the Risch
    algorithm hands what it proves nonelementary, search on over such a variable for
    minutes: `INT(LN(1 + x)/(x^2 + 1), x, 1, 2)` does not come back. Over a variable
    nothing is assumed of, the same two methods give up on it at once. An integrand
    that both give up on there has no antiderivative they would find over a positive
    variable either, the function being the same, and the definite integral is left
    to the methods that do without one: the tables, and the Meijer G route for an
    infinite end. The Risch algorithm is switched off along with the manual rules, as
    sympy hands its nonelementary part to them with every other hint dropped.

    A rational function has an elementary antiderivative always, and is not asked:
    the Risch algorithm would integrate it by sympy's `ratint`, which searches for
    minutes for the real roots of the resolvent of `1/(x^5 + x + 1)`.
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
    """`value`, as the head itself where it is the same integral under another class.

    Sympy hands back an integral its Risch algorithm proved nonelementary as a
    `NonElementaryIntegral`, over the integrand as the algorithm rewrote it, which is
    not equal to the `Integral` it was asked and so is taken for an answer: written
    out, read back, asked on each side of a parameter's sign and asked all over
    again as a head of its own. What it says is that the head stands, and that is
    what it is called. So does the same integral over an integrand sympy only
    factored, as it gives back `INT(t/((t^2 + z^2)*(#e^(2*pi*t) - 1)), t, 0, inf)` with
    the exponential split into two factors.
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
    """An antiderivative by `by_a_positive_substitution`, or None."""
    if len(head.limits) != 1 or len(head.limits[0]) != 1:
        return None
    value = by_a_positive_substitution(head.function, head.limits[0][0])
    return None if value is None else Candidate(value, trusted=True, tabled=True)


def by_a_positive_substitution(
    integrand: sp.Basic, variable: sp.Basic
) -> sp.Basic | None:
    """The integral over a new variable that is positive wherever the integrand is
    real, or None.

    Sympy takes a real variable out of a square root as if it were positive:
    `INT(1/(x*SQRT(x^2 + a^2)), x)` comes back as `-ASINH(ABS(a)/x)/ABS(a)`, and
    `INT(1/(x^2*SQRT(a*x + b)), x)` over `SQRT(b/(a*x) + 1)/SQRT(x)`. Both
    differentiate back for positive `x` only. A variable that is positive by
    construction gives sympy nothing to get wrong:

    - where every root in the integrand is of one linear `a*x + b`, it is the root
      `t = (a*x + b)^(1/q)` itself, over which the integrand is rational where it
      was rational in `x` and the root;
    - where the integrand is odd, it is `u = x^2`, and the antiderivative, a function
      of `x^2`, is even as it must be; a root of a linear function of `u` is then
      taken out as above, which for `SQRT(x^2 + a^2)` is the substitution of the
      root itself;
    - where it is even, it is `x` itself taken positive, and the antiderivative is
      kept where it is proved odd, `F(x) + F(-x)` simplifying to a constant: its
      derivative is then even, as the integrand is, and it holds for negative `x` as
      well.

    Each of these is an antiderivative by construction, sympy's answer over the
    positive variable once it is proved to differentiate back there. Where it is split
    over the parameters, the last case is taken alone where it is proved to
    differentiate back to the integrand.
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
    """An antiderivative by the substitutions `by_a_positive_substitution` lists, not
    yet checked, or None where none applies."""
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
    """Whether the expression plus its value at the negated variable simplifies to
    something free of the variable."""
    balance = attempt(
        expression, lambda e: sp.simplify(e + e.subs(variable, -variable))
    )
    return balance is not None and not balance.has(variable)


def _by_a_linear_root(
    integrand: sp.Basic, variable: sp.Basic, radicand: sp.Basic, polynomial: sp.Poly
) -> sp.Basic | None:
    """The integral over `t = radicand^(1/q)`, the radicand linear in the variable and
    `q` the common denominator of the powers it is raised to, or None."""
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
    # radicand^(k/order) is root^k for the principal root, for every radicand.
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
    """An antiderivative by the rules a calculus course teaches alone, or None.

    They are the one method of sympy's that does not reach the algorithm the wrong
    answers come out of, and what the methods before them answer wrongly they may
    answer right; theirs is put to the same proof. Nor are they asked again for a
    shorter answer than their own. One of `_RETRIES`, asked only where an answer of
    sympy's was refused.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 1:
        return None
    manual = attempt(head, lambda h: h.doit(deep=False, manual=True))
    if manual is None or manual.has(sp.Integral):
        return None
    return Candidate(manual, tabled=True)


def _over_any_variable(head: sp.Basic) -> Candidate | None:
    """The integral again over a variable nothing is assumed of, or None.

    What sympy's Meijer G route makes of a real variable is not always what it makes
    of an unassumed one: over a real `x` it writes `ABS(x)/x` into the answer to
    `INT(SIN(x)/x^2, x)` and gets the negative half wrong, while over a plain symbol
    it answers `CI(x) - SIN(x)/x` plus logarithms that cancel. That answer is put to
    the same proof as any other. One of `_RETRIES`, asked only where an answer of
    sympy's was refused, and not where the full methods are not affordable: they are
    what is asked again here, and the bounded methods offer an answer where they are
    not.
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


#: What is asked of an integral where an answer of sympy's general integrators is
#: refused, in order.
_RETRIES: tuple[Method, ...] = (_by_the_manual_rules, _over_any_variable)

#: The methods an integral is asked, in order. Each takes the head and offers a
#: `Candidate`, or None where it does not apply; which candidates are `trusted` or
#: `formed`, and which have `retries`, is the method's to say.
_METHODS: tuple[Method, ...] = (
    _of_a_rational_function,
    _of_a_trigonometric_function,
    _of_a_radical,
    _over_an_exponential,
    _by_derivative_divides,
    _by_products_as_sums,
    _by_parts_over_a_polynomial,
    _over_a_positive_variable,
    _by_a_radical,
    _by_the_bounded_methods,
    _by_the_full_methods,
    _by_parts,
)

#: Sympy's methods, in the order `_METHODS` asks them, none of them trusted.
_SYMPY: tuple[Method, ...] = (_by_the_bounded_methods, _by_the_full_methods)
