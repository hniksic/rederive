"""Definite integrals: what sympy gets wrong between limits, put right.

Sympy reads straight through a singularity inside the interval, integrates over a
step function only in part, and answers under a condition on a parameter it could
have split on. What is here looks at the steps and the singularities of an integrand
before sympy is asked, and at what sympy answers after: an end it read past, a
condition of convergence, a parameter whose sign decides the answer, a pole that
moves with one. The dispatch in `indefinite` is what asks each of them, and what
each integrates on the way is integrated by it again.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import sympy as sp
from sympy.core.relational import Relational
from sympy.functions.elementary.trigonometric import TrigonometricFunction

# The module rather than its names, since `indefinite` imports from this one: an
# integral met on the way here is handed back to its dispatch.
from rederive.engine.integration import indefinite
from rederive.engine.integration.form import parametric_divisors
from rederive.engine.integration.proof import (
    Stretch,
    agrees,
    agrees_at,
    proved_zero,
    unbroken,
)
from rederive.engine.rewriting import attempt
from rederive.engine.signs import folded_signs

__all__ = [
    "CONVERGENT",
    "UNDECIDED",
    "VERDICTS",
    "across_the_steps",
    "at_the_singularities",
    "between_the_steps",
    "beyond_convergence",
    "by_the_antiderivative",
    "by_the_sign_of_a_parameter",
    "definite",
    "ends",
    "odd_about_the_middle",
    "past_a_singular_end",
    "real_where_real",
    "unresolved",
    "with_moving_poles",
]


def real_where_real(head: sp.Basic, value: sp.Basic) -> sp.Basic:
    """A definite integral's value, refused where the integral is real and the value
    is not proved to be.

    Sympy answers `INT(LN(x)/(x + 1), x, 0, 1)` with `-pi^2/12 + 2*#i*pi*LN(2)` and
    `INT((x^t - 1)/LN(x), x, 0, 1)` with `LN(t + 1) + #i*pi`: its antiderivatives are
    taken at an end on the other side of a branch cut. A real integrand over a real
    interval has a real integral, so where the integrand is proved real on the open
    interval for every real value of the parameters, as `Stretch` proves it, a value
    with `#i` in it is kept only where its imaginary part is proved zero. Otherwise it
    is replaced by the difference of Simplify's own antiderivative at the ends where
    that is proved the integral, as `by_the_antiderivative` proves it, and the
    integral stands where it is not.

    An integrand not proved real on the interval, as `x^(1/3)` is not left of zero,
    has a value that need not be real either, and is not looked at. Where it is real,
    a value with an imaginary part that is infinite or still an integral is no answer
    either: `INT(LN(x)*LN(1 - x), x, 0, 1)` comes back as `2 - #i*inf` plus an
    integral over a split, which would take minutes to evaluate.
    """
    if not definite(head) or value == head or not value.has(sp.I):
        return value
    variable, low, high = head.limits[0]
    orientation = _orientation(low, high)
    if orientation is None or not (low.is_extended_real and high.is_extended_real):
        return value
    bounds = (low, high) if orientation > 0 else (high, low)
    if not Stretch(variable, *bounds).real(head.function):
        return value
    if value.has(sp.Integral, sp.oo, -sp.oo, sp.zoo, sp.nan):
        return head
    imaginary = attempt(value, lambda v: sp.im(sp.expand_complex(v)))
    if imaginary is not None and proved_zero(imaginary):
        return value
    return by_the_antiderivative(head) or head


def odd_about_the_middle(head: sp.Basic) -> bool:
    """Whether a definite integral that converges is zero by symmetry: over a finite
    interval, with an integrand free of parameters that turns its sign when reflected
    about the middle of it.

    `INT(LN(TAN(x)), x, 0, pi/2)` has no antiderivative sympy finds, and `LN(TAN(pi/2
    - x))` is `-LN(TAN(x))` on the interval. The integral is the integral of the
    reflected integrand, so twice it is the integral of the sum of the two, and that
    sum is proved zero on the interval: its derivative simplifies to zero, it is
    continuous there as `unbroken` proves, and it is zero at the middle, where it is
    twice the integrand.
    """
    variable, low, high = head.limits[0]
    integrand = head.function
    if integrand.free_symbols != {variable} or not (low.is_finite and high.is_finite):
        return False
    orientation = _orientation(low, high)
    if orientation is None:
        return False
    balance = integrand + integrand.subs(variable, low + high - variable)
    at_middle = attempt(integrand.subs(variable, (low + high) / 2), sp.simplify)
    if at_middle != 0:
        return False
    slope = attempt(balance, lambda b: sp.simplify(sp.diff(b, variable)))
    if slope != 0:
        return False
    bounds = (low, high) if orientation > 0 else (high, low)
    return unbroken(balance, variable, *bounds)


def by_the_antiderivative(head: sp.Basic) -> sp.Basic | None:
    """A definite integral sympy left standing, as the difference of the limits at the
    ends of the antiderivative Simplify finds, where that is proved the integral, or
    None.

    Sympy values a definite integral by an antiderivative of its own, and has none for
    `INT(SIN(x)/SQRT(1 + q^2*COS(x)^2), x, 0, pi)`, which `u = COS(x)` makes
    elementary. The antiderivative found here is the one an indefinite integral is
    answered with. Its difference at the ends is the integral where it is continuous on
    the open interval and differentiates to the integrand there, which `unbroken`
    proves for every value of the parameters: an antiderivative over a logarithm whose
    argument crosses zero inside the interval, or a polylogarithm on its cut, has a
    difference that is not the integral. An integrand that both the Risch algorithm
    and the manual rules give up on, as `within_reach` finds, is not looked at, its
    antiderivative being out of reach.

    That proof says nothing where the parameters make the antiderivative undefined,
    as `1/(m^2 - 4)` in the antiderivative of `SIN(m*x)*SIN(2*x)` does at `m = 2`,
    where the integral is `pi/2`. A value that has a value there itself, as the `0`
    the integral over `0` to `pi` comes to for an integer `m` does, claims one the
    proof does not cover, and is refused; one that is undefined there as well is the
    generic answer it is.
    """
    if head.function.has(sp.Piecewise) or indefinite.within_reach(head):
        return None
    variable, low, high = head.limits[0]
    orientation = _orientation(low, high)
    if orientation is None:
        return None
    antiderivative = indefinite.integral(sp.Integral(head.function, variable))
    if antiderivative.has(sp.Integral, sp.Piecewise):
        return None
    bounds = (low, high) if orientation > 0 else (high, low)
    if not unbroken(antiderivative, variable, *bounds):
        return None
    at_ends = ends(antiderivative, variable, low, high)
    if at_ends is None:
        return None
    value = at_ends[0] - at_ends[1]
    if value.has(sp.Limit, sp.nan, sp.zoo, sp.oo, -sp.oo):
        return None
    if _defined_past(value, antiderivative, variable):
        return None
    return value


def _defined_past(value: sp.Basic, antiderivative: sp.Basic, variable: sp.Basic) -> bool:
    """Whether the value is defined at some value of the parameters where the
    antiderivative divides by zero, or that is not known: see `by_the_antiderivative`.
    """
    for divisor in parametric_divisors(antiderivative, variable):
        for parameter in sorted(divisor.free_symbols, key=sp.default_sort_key):
            try:
                roots = sp.solve(divisor, parameter)
            except Exception:
                return True
            for root in roots:
                there = attempt(value, lambda v: v.subs(parameter, root))
                if there is None or not there.has(sp.nan, sp.zoo):
                    return True
    return False


def definite(head: sp.Basic) -> bool:
    """Whether this is an integral over one variable between two limits."""
    return len(head.limits) == 1 and len(head.limits[0]) == 3


def ends(
    antiderivative: sp.Basic,
    variable: sp.Basic,
    low: sp.Basic,
    high: sp.Basic,
    near: Callable[[sp.Basic], sp.Basic] | None = None,
) -> list[sp.Basic] | None:
    """The antiderivative's limits at the upper end and then the lower, each
    approached from inside the interval, or None where either is not had.

    Sympy leaves the limit of `ATAN((COS(a) + x)/ABS(SIN(a)))` at infinity standing,
    not knowing that `ABS(SIN(a))` is not zero. Where it is zero the antiderivative is
    not defined at any `x`, as it divides by it, so the limit is taken for the
    absolute value positive, and kept where it divides by that value too.

    With `near`, the limit at each end is taken of what `near` makes of that end, the
    antiderivative written for the limit there.
    """
    ends = []
    for point, side in ((high, "-"), (low, "+")):
        written = antiderivative if near is None else near(point)
        end = attempt(written, lambda a: sp.limit(a, variable, point, side))
        if end is not None and end.has(sp.Limit):
            end = _positively(written, variable, point, side) or end
        if end is None:
            return None
        ends.append(end)
    return ends


def _positively(
    antiderivative: sp.Basic, variable: sp.Basic, point: sp.Basic, side: str
) -> sp.Basic | None:
    """The limit of an antiderivative with its absolute values and square roots of
    parameters taken for positive, where it is undefined with each of them zero and so
    is the limit, or None: see `ends`. A square root is taken only where its radicand
    is never negative."""
    kernels = {
        sp.sqrt(call.base) if isinstance(call, sp.Pow) else call
        for call in antiderivative.atoms(sp.Abs, sp.Pow)
        if not call.has(variable) and call.free_symbols
        if isinstance(call, sp.Abs) or call.exp.is_Rational and call.exp.q == 2
    }
    stand_ins = {
        kernel: sp.Dummy(positive=True) for kernel in kernels if _never_negative(kernel)
    }
    if not stand_ins:
        return None
    written = attempt(antiderivative, lambda a: a.subs(stand_ins))
    if written is None or not written.has(variable):
        return None
    end = attempt(written, lambda a: sp.limit(a, variable, point, side))
    if end is None or end.has(sp.Limit):
        return None
    for stand_in in stand_ins.values():
        if not written.has(stand_in):
            continue
        for expression in (written, end):
            vanished = attempt(expression, lambda e: e.subs(stand_in, 0))
            if vanished is None or not vanished.has(sp.zoo, sp.nan):
                return None
    return end.xreplace({stand_in: call for call, stand_in in stand_ins.items()})


def past_a_singular_end(head: sp.Basic, value: sp.Basic) -> sp.Basic | None:
    """A definite integral from a singular end to a symbolic one, valued by how it
    diverges there, or None.

    The antiderivative's limit at the singular end is taken from inside the interval,
    which is the side the other end is on, and that side is not known: `INT(1/x^2, x,
    0, alpha)` comes back as `inf - 1/alpha`, the integral for a positive `alpha`
    only, as for a negative one it is `-inf`. Where the integrand blows up there no
    slower than the reciprocal of the distance, the integral diverges whatever the
    other end, which is a regular point of the integrand for all but isolated values,
    adds to it. So where the answer has an infinity in it, the singular end is the
    only real singularity of the integrand, and the integrand is real on both sides of
    it, the integral is the infinity
    `_end_verdict` finds on each side: `INT(COS(a*x)/x, x, 0, alpha)` is `inf` for
    either sign of `alpha`. Where the two sides diverge to different infinities, or
    to one not known, the integral stands.
    """
    if not definite(head) or value in (sp.oo, -sp.oo):
        return None
    if not value.has(sp.oo, -sp.oo) or value.has(sp.Integral):
        return None
    variable, low, high = head.limits[0]
    numbers = [end for end in (low, high) if not end.free_symbols]
    if len(numbers) != 1 or not (numbers[0].is_extended_real and numbers[0].is_finite):
        return None
    (point,) = numbers
    integrand = head.function
    poles = attempt(integrand, lambda f: sp.singularities(f, variable, sp.Reals))
    if poles != sp.FiniteSet(point):
        return None
    distance = sp.Dummy("d", positive=True)
    for side in (1, -1):
        near = attempt(integrand, lambda f: f.subs(variable, point + side * distance))
        if near is None or near.is_extended_real is not True:
            return None
    above = _end_verdict(integrand, variable, point, 1)
    below = _end_verdict(integrand, variable, point, -1)
    if above not in (sp.oo, -sp.oo) or below not in (sp.oo, -sp.oo):
        return None
    # Up from the singular end on the side above it, or down to it from there.
    beyond, before = (above, -below) if point == low else (-above, below)
    return beyond if beyond == before else head


def beyond_convergence(head: sp.Basic, value: sp.Basic) -> sp.Basic | None:
    """A definite integral's divergent case valued, where it diverges to a sign.

    Sympy answers an improper integral with a parameter in it over the condition
    its convergence test proves: `INT(x^n, x, 1, inf)` is `-1/(n + 1)` where `n <
    -1`, and elsewhere the integral itself, unevaluated. That is where the integral
    diverges, and Derive says so: it works the antiderivative to its limit at each
    end and prints the infinity it finds, so the second case is `inf` to it. Sympy
    will find the same infinity where it knows the sign of the exponent, which the
    parameter it was handed does not say and a dummy of known sign does.

    So the complement of the condition is written as substitutions: `n < -1` fails
    where `n` is `-1 - s` for a positive `s`, and where `n` is `-1` exactly. The
    integral is asked again under each, and where every answer is one infinity,
    that is the case's value. A finite answer, an unevaluated one, or two that
    disagree keeps the integral standing: a convergence test proves convergence
    and not divergence, and an oscillating integrand diverges to no sign at all.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 3:
        return None
    folded = attempt(value, sp.piecewise_fold)
    if not isinstance(folded, sp.Piecewise) or len(folded.args) != 2:
        return None
    (converged, condition), (standing, otherwise) = folded.args
    if otherwise is not sp.true or standing != head:
        return None
    outcomes = set()
    for parameter, point in _complement(condition):
        found = attempt(head, lambda h: indefinite.by_sympy(h.subs(parameter, point)))
        if found is None or not found.is_infinite or found.is_extended_real is not True:
            return None
        outcomes.add(found)
    if len(outcomes) != 1:
        return None
    return sp.Piecewise((converged, condition), (outcomes.pop(), sp.true))


#: The inequalities a convergence condition is written in, and the only ones
#: whose complement is a sign the parameter can be given.
_INEQUALITIES = (sp.StrictLessThan, sp.StrictGreaterThan, sp.LessThan, sp.GreaterThan)


def _complement(condition: sp.Basic) -> list[tuple[sp.Basic, sp.Basic]]:
    """Where a one-parameter inequality fails, as values of its parameter.

    Each is the parameter written in terms of a dummy whose sign sympy knows: the
    complement of `g > 0` is `g = -s` for a positive `s`, and where the inequality
    was strict, `g = 0` as well. A condition that is no such inequality - on more
    than one parameter, or on the real part of one, or that does not determine its
    parameter - has no complement here, and the empty list leaves nothing valued.
    """
    if not isinstance(condition, _INEQUALITIES) or len(condition.free_symbols) != 1:
        return []
    (parameter,) = condition.free_symbols
    if isinstance(condition, (sp.StrictGreaterThan, sp.GreaterThan)):
        excess = condition.lhs - condition.rhs
    else:
        excess = condition.rhs - condition.lhs
    if excess.is_polynomial(parameter) is not True:
        return []
    shortfall = sp.Dummy("s", positive=True)
    targets = [-shortfall]
    if isinstance(condition, (sp.StrictLessThan, sp.StrictGreaterThan)):
        targets.append(sp.S.Zero)
    points = []
    for target in targets:
        try:
            solutions = sp.solve(sp.Eq(excess, target), parameter)
        except Exception:
            return []
        if len(solutions) != 1:
            return []
        points.append((parameter, solutions[0]))
    return points


#: What `at_the_singularities` says of an integral it finds no infinity in: that
#: every end it looked at converges, or that some end could not be decided.
CONVERGENT = sp.Symbol("convergent")
_UNSETTLED = sp.Symbol("unsettled")
UNDECIDED = sp.Symbol("undecided")
VERDICTS = (CONVERGENT, _UNSETTLED, UNDECIDED)

#: How many singularities an interval is looked at across.
_POLES = 40


def at_the_singularities(head: sp.Basic) -> sp.Basic | None:
    """A definite integral decided by how its integrand behaves where it blows up.

    The integral over an interval with a singularity in it is the sum of the
    integrals up to the singularity and on from it, each an improper integral of
    its own, and it exists only where each of those does. Taken straight through by
    an antiderivative it comes out as a number anyway: `INT(1/x^2, x, -1, 1)` is -2
    that way, for an integrand positive throughout, and `INT(1/x, x, -1, 1)` is the
    principal value 0, or `-#i*pi` with the logarithm's branch in it. Neither is the
    integral, which is `inf` for the first and has no value at all for the second.

    So the interval is cut at each singularity sympy can place in it, and the
    integrand is looked at on each side of each cut, and at an infinite end, by
    `_end_verdict`. An end where it blows up at least as fast as the reciprocal of
    the distance makes that part of the integral diverge to the infinity of the
    integrand's sign there. One such infinity, with every other end converging, is
    the integral; two of opposite signs leave nothing to add up, and the answer is
    `?`, which is what the sum of the parts - `inf - inf` - would be. An integrable
    singularity, such as `1/SQRT(ABS(x))` or `LN(x)` at zero, is the integral sympy
    finds, and `CONVERGENT` says to ask it. Where the singularity is inside the
    interval the integral is still the sum of the parts: an antiderivative taken
    across it can jump there, as `SQRT(pi)*ERF(1/x)/2` does for `#e^(-1/x^2)/x^2`,
    so each part is integrated on its own.

    `UNDECIDED` says an end diverges to an infinity whose sign is not known -
    `a/x^2` at zero, with the sign of `a` unknown - and then sympy's answer, which
    is the antiderivative's across it, is not had at all: the integral stands, or
    is worked out on each side of the parameter's sign. `_UNSETTLED` says an end
    could not be told either way, as `1/(x*LN(x))` at zero or the oscillating tail
    of `SIN(x)/x` cannot, and sympy's answer is printed unless it is infinite.

    None where there is nothing to look at: no singularity in the interval, an end
    that is not a number, or a singularity that moves with a parameter.
    """
    if not definite(head):
        return None
    variable, low, high = head.limits[0]
    integrand = head.function
    if low.free_symbols or high.free_symbols:
        return None
    unsigned = {
        symbol
        for symbol in integrand.free_symbols - {variable}
        if symbol.is_positive is None and symbol.is_negative is None
    }
    orientation = _orientation(low, high)
    if orientation is None:
        return None
    if orientation < 0:
        low, high = high, low
    interval = attempt(low, lambda bound: sp.Interval(bound, high))
    if not isinstance(interval, sp.Interval):
        return None
    poles = attempt(integrand, lambda f: sp.singularities(f, variable, interval))
    if not isinstance(poles, sp.FiniteSet) or not poles or len(poles) > _POLES:
        return None
    if not all(
        pole.is_extended_real and pole.is_finite and not pole.has(*unsigned)
        for pole in poles
    ):
        return None
    ends = _ordered({low, high, *poles})
    if ends is None:
        return None
    verdicts = []
    for left, right in zip(ends, ends[1:]):
        for end, side in ((left, 1), (right, -1)):
            if end in poles or end.is_infinite:
                verdicts.append(_end_verdict(integrand, variable, end, side))
    infinities = {verdict for verdict in verdicts if verdict.is_infinite}
    if len(infinities) > 1:
        return sp.nan
    for verdict in (UNDECIDED, _UNSETTLED):
        if verdict in verdicts:
            return verdict
    if infinities:
        return orientation * infinities.pop()
    if len(ends) == 2:
        return CONVERGENT
    total = sp.S.Zero
    for left, right in zip(ends, ends[1:]):
        part = indefinite.integral(sp.Integral(integrand, (variable, left, right)))
        if part.has(sp.Integral) or part.is_finite is not True:
            return CONVERGENT
        total += part
    return orientation * total


def with_moving_poles(head: sp.Basic, value: sp.Basic) -> sp.Basic | None:
    """A definite integral of a rational function whose poles move with a parameter,
    split on where the poles are, or None where none can be in the interval.

    `INT(1/(x - a)^2, x, 0, 2)` taken through its antiderivative is `-2/(a*(2 - a))`,
    which is the integral only where `a` is outside `[0, 2]`: inside it the pole is
    in the interval, and a pole of a rational function is never integrable. So the
    values of the parameter that put a real pole between the ends, or on one, are
    found. They are cut where what decides the integral can change: where a pole
    reaches an end, where two poles meet, where one cancels against the numerator
    or goes off to infinity. Between two such values every pole stays where it is
    relative to the ends and keeps its order and the sign it blows up with, so the
    integral is had by asking at one value inside, as at any number, and at each
    cut by asking there. The answer already had is the integral for the other
    values, and for these too where it agrees with every answer asked for - as one
    split on the parameter's sign by `by_the_sign_of_a_parameter` can. An answer at
    a value inside a stretch that is not an infinity, or `?`, means something was
    missed, and the integral is left standing.

    Poles that are real only for isolated values of the parameter, as `#i*a` is
    real only at `a = 0`, are left to the formula, which blows up there as well.
    """
    if not definite(head):
        return None
    variable, low, high = head.limits[0]
    integrand = head.function
    if low.free_symbols or high.free_symbols:
        return None
    parameters = integrand.free_symbols - {variable}
    if len(parameters) != 1:
        return None
    (parameter,) = parameters
    if parameter.is_extended_real is not True or parameter.is_positive is not None:
        return None
    if parameter.is_negative is not None or parameter.is_zero is not None:
        return None
    if integrand.is_rational_function(variable) is not True:
        return None
    orientation = _orientation(low, high)
    if orientation is None:
        return None
    if orientation < 0:
        low, high = high, low
    together = attempt(integrand, lambda f: sp.cancel(sp.together(f)))
    if together is None:
        return None
    numerator, denominator = sp.fraction(together)
    try:
        polynomial = sp.Poly(denominator, variable)
        poles = sp.roots(polynomial, multiple=False)
    except Exception:
        return None
    if not poles or sum(poles.values()) != polynomial.degree():
        return None
    region = sp.S.EmptySet
    critical = []
    for pole in poles:
        if not pole.has(parameter):
            continue
        found = _real_pole(pole, parameter)
        if found is None:
            return head
        if isinstance(found, sp.FiniteSet):
            continue
        for end, relation, at in ((low, sp.Ge, pole), (high, sp.Le, pole)):
            if not end.is_finite:
                continue
            part = attempt(
                at, lambda p: sp.solveset(relation(p, end), parameter, sp.S.Reals)
            )
            if part is None or part.has(sp.ConditionSet):
                return head
            found = sp.Intersection(found, part)
            critical.append(pole - end)
        region = sp.Union(region, found)
    if region is sp.S.EmptySet or isinstance(region, sp.FiniteSet):
        return None
    try:
        critical += [
            sp.resultant(numerator, denominator, variable),
            sp.discriminant(polynomial),
            polynomial.LC(),
        ]
    except Exception:
        return head
    cuts = set()
    for equation in critical:
        if equation == 0:
            continue
        found = attempt(equation, lambda e: sp.solveset(e, parameter, sp.S.Reals))
        if found is None or not isinstance(found, sp.FiniteSet):
            if found is not None and found.is_empty:
                continue
            return head
        cuts |= {cut for cut in found if region.contains(cut) is sp.true}
    stretches = attempt(region, lambda r: r - sp.FiniteSet(*cuts))
    if stretches is None or len(cuts) > _POLES:
        return head
    pieces = stretches.args if isinstance(stretches, sp.Union) else (stretches,)
    if not all(isinstance(piece, sp.Interval) for piece in pieces):
        return head
    if value.has(sp.Integral, sp.Limit):
        return None
    answers: dict[sp.Basic, sp.Basic] = {}
    agreeing = True
    for where, sample in [
        *((sp.FiniteSet(cut), cut) for cut in cuts),
        *((piece, _inside(piece)) for piece in pieces),
    ]:
        found = attempt(head, lambda h: indefinite.integral(h.subs(parameter, sample)))
        divergent = found is sp.nan or found in (sp.oo, -sp.oo)
        if found is None or found.has(sp.Integral) or not divergent and where.is_open:
            return head
        answers[found] = sp.Union(answers.get(found, sp.S.EmptySet), where)
        there = attempt(value, lambda v: v.subs(parameter, sample))
        agreeing = (
            agreeing
            and there is not None
            and (there == found if divergent else agrees(there, found))
        )
    if agreeing:
        return None
    cases = []
    for answer, where in answers.items():
        condition = attempt(where, lambda w: w.as_relational(parameter))
        if condition is None:
            return head
        cases.append((answer, condition))
    if region != sp.S.Reals:
        cases.append((value, sp.true))
    else:
        cases[-1] = (cases[-1][0], sp.true)
    return sp.Piecewise(*cases)


def _real_pole(pole: sp.Basic, parameter: sp.Basic) -> sp.Basic | None:
    """The values of the parameter for which a pole is real, or None where they are not
    had.

    A pole written without the imaginary unit is real where every radicand in it is at
    least zero; one written with it, where its square is. Sympy takes the square of a
    pole such as `SQRT(COS(a)^2 - 1) - COS(a)` for a number at least zero for every
    `a`, which is why the radicands are asked first. A periodic function of the
    parameter is solved over one period only, so the values are had there only where
    they are isolated points, which repeat as isolated points.
    """
    if pole.has(sp.I):
        conditions = [sp.Ge(pole**2, 0)]
    else:
        conditions = [
            sp.Ge(power.base, 0)
            for power in pole.atoms(sp.Pow)
            if power.exp.is_Rational and not power.exp.is_integer
        ]
    found = sp.S.Reals
    for condition in conditions:
        part = attempt(condition, lambda c: sp.solveset(c, parameter, sp.S.Reals))
        if part is None or part.has(sp.ConditionSet):
            return None
        found = sp.Intersection(found, part)
    periodic = any(call.has(parameter) for call in pole.atoms(TrigonometricFunction))
    if periodic and not isinstance(found, sp.FiniteSet):
        return None
    return found


def _never_negative(call: sp.Basic) -> bool:
    """Whether an absolute value, or a square root whose radicand is a function of one
    real parameter that is at least zero for every value of it."""
    if isinstance(call, sp.Abs):
        return True
    if not isinstance(call, sp.Pow) or call.exp != sp.S.Half:
        return False
    if len(call.free_symbols) != 1:
        return False
    (parameter,) = call.free_symbols
    if parameter.is_extended_real is not True:
        return False
    below = attempt(
        call.base, lambda r: sp.solveset(sp.Lt(r, 0), parameter, sp.S.Reals)
    )
    return below is not None and below.is_empty is True


def _inside(stretch: sp.Interval) -> sp.Basic:
    """A number inside an interval."""
    start, end = stretch.start, stretch.end
    if start.is_finite and end.is_finite:
        return (start + end) / 2
    if start.is_finite:
        return start + 1
    if end.is_finite:
        return end - 1
    return sp.S.Zero


def _orientation(low: sp.Basic, high: sp.Basic) -> int | None:
    """1 for limits in increasing order, -1 for decreasing, None where not known."""
    span = attempt(high, lambda h: h - low)
    if span is None or high == low:
        return None
    if span.is_extended_positive:
        return 1
    if span.is_extended_negative:
        return -1
    return None


def _ordered(points: set) -> list[sp.Basic] | None:
    """Real numbers in increasing order, or None where they cannot be placed.

    A point may carry a parameter of known sign, as `p` and `2*p` do for a positive
    `p`, so the points are placed by deciding their differences rather than by
    their values.
    """
    ordered: list[sp.Basic] = []
    for point in points:
        place = 0
        for other in ordered:
            difference = attempt(point, lambda p: p - other)
            if difference is None or point == other:
                return None
            if difference.is_extended_positive:
                place += 1
            elif difference.is_extended_negative is not True:
                return None
        ordered.insert(place, point)
    return ordered


#: The powers of the distance an end is weighted with to show that it converges:
#: an integrand bounded by such a power of the reciprocal distance to a point, or by
#: such a power of the reciprocal of the variable towards an infinite end, is
#: integrable there.
_NEAR = (sp.Rational(1, 2), sp.Rational(9, 10))
_FAR = (sp.Integer(2), sp.Rational(11, 10))


def _end_verdict(
    integrand: sp.Basic, variable: sp.Basic, end: sp.Basic, side: int
) -> sp.Basic:
    """How the integral fares at one end of a part of the interval.

    The end is approached from the side `side` points to, `1` being from above,
    and `distance` is the positive distance to it, or the size of the variable
    where the end is infinite. If the integrand times that distance tends to a
    real number other than zero, or to an infinity, the integrand is of one sign
    there and no smaller than a multiple of the reciprocal distance, whose
    integral diverges; the part diverges to the infinity of that sign, or to one
    `UNDECIDED` where the sign is not known. If it times a power of the distance
    short of 1 (beyond 1 at an infinite end) stays bounded, the part converges by
    comparison. Anything else is `_UNSETTLED`.
    """
    if end.is_infinite:
        distance = variable if end is sp.oo else -variable
        powers = _FAR
        direction = "-" if end is sp.oo else "+"
    else:
        distance = side * (variable - end)
        powers = _NEAR
        direction = "+" if side > 0 else "-"

    def approached(weight: sp.Basic) -> sp.Basic | None:
        found = attempt(
            integrand, lambda f: sp.limit(f * weight, variable, end, direction)
        )
        if found is None or found.has(sp.AccumBounds, sp.nan, sp.zoo, sp.Limit):
            return None
        return found

    leading = approached(distance)
    if leading is not None and leading != 0:
        if leading.is_extended_positive:
            return sp.oo
        if leading.is_extended_negative:
            return -sp.oo
        return UNDECIDED
    for power in powers:
        bounded = approached(distance**power)
        if bounded is not None and bounded.is_finite:
            return CONVERGENT
    return _UNSETTLED


#: The functions an integrand is split at the jumps of: each is constant, or a
#: polynomial times a constant sign, between the points where its argument crosses
#: a jump.
_STEPS = (sp.floor, sp.ceiling, sp.frac, sp.sign, sp.Abs, sp.Heaviside)
_ROUNDING = (sp.floor, sp.ceiling, sp.frac)

#: How many parts an interval is split into at the jumps of its steps.
_STEP_PIECES = 64


def between_the_steps(head: sp.Basic) -> sp.Basic | None:
    """A definite integral of a step function, as the sum over the steps.

    `INT(FLOOR(x), x, 0, 3)` is `0 + 1 + 2`: between consecutive integers the floor
    is a constant, and the integral is the sum of the integrals over each stretch.
    Sympy has no antiderivative for it, and so leaves the integral standing. Over
    limits that are numbers the jumps are finitely many, and the integrand is
    integrated stretch by stretch with each step replaced by the value it holds
    there - a constant for `FLOOR`, a sign times its argument for `ABS`.

    Only an integrand sympy cannot do whole is split: one with a rounding function
    in it, or a sign or absolute value of something other than a polynomial, whose
    antiderivative sympy makes up of pieces that hold only near the origin -
    `INT(ABS(SIN(x)), x, 0, 10*pi)` is 20, ten arches of 2, where the
    antiderivative taken across it gives 4. Each step's argument has to be free of
    parameters, so that the jumps are numbers: a polynomial of degree at most two
    for the rounding functions, whose crossings of each integer are then radicals,
    and for the signs one sympy can solve for its zeros between the limits.
    """
    if not definite(head):
        return None
    variable, low, high = head.limits[0]
    if low.free_symbols or high.free_symbols or not (low.is_finite and high.is_finite):
        return None
    steps = [step for step in head.function.atoms(*_STEPS) if step.has(variable)]
    if not any(
        isinstance(step, _ROUNDING) or not step.args[0].is_polynomial(variable)
        for step in steps
    ):
        return None
    orientation = _orientation(low, high)
    if orientation is None:
        return None
    if orientation < 0:
        low, high = high, low
    jumps: set = set()
    for step in steps:
        try:
            found = _jumps(step, variable, low, high)
        except (TypeError, ValueError, NotImplementedError):
            return None
        if found is None:
            return None
        jumps |= found
        if len(jumps) >= _STEP_PIECES:
            return None
    ends = _ordered(jumps | {low, high})
    if ends is None:
        return None
    total = sp.S.Zero
    for left, right in zip(ends, ends[1:]):
        middle = (left + right) / 2
        held = {step: _held(step, variable, middle) for step in steps}
        if any(value is None for value in held.values()):
            return None
        piece = attempt(head.function, lambda f: f.xreplace(held))
        if piece is None:
            return None
        value = indefinite.integral(sp.Integral(piece, (variable, left, right)))
        if value.has(sp.Integral):
            return None
        total += value
    return orientation * total


def _jumps(
    step: sp.Basic, variable: sp.Basic, low: sp.Basic, high: sp.Basic
) -> set | None:
    """The points strictly between the limits where a step function jumps, or None
    where they cannot be had as exact numbers."""
    argument = step.args[0]
    if argument.free_symbols - {variable}:
        return None
    if isinstance(step, _ROUNDING):
        try:
            polynomial = sp.Poly(argument, variable)
        except sp.PolynomialError:
            return None
        coefficients = polynomial.all_coeffs()
        if polynomial.degree() > 2 or not all(c.is_Rational for c in coefficients):
            return None
        turns = [
            point
            for point in sp.solve(polynomial.diff(variable).as_expr(), variable)
            if low < point < high
        ]
        values = [argument.subs(variable, point) for point in (low, high, *turns)]
        levels = range(int(sp.floor(min(values))), int(sp.ceiling(max(values))) + 1)
        if len(levels) > _STEP_PIECES:
            return None
        crossings = set()
        for level in levels:
            roots = sp.solve(argument - level, variable)
            crossings |= {root for root in roots if root.is_real and low < root < high}
        return crossings
    zeros = attempt(
        argument, lambda a: sp.solveset(a, variable, sp.Interval.open(low, high))
    )
    if not isinstance(zeros, sp.FiniteSet) or len(zeros) > _STEP_PIECES:
        return None
    return set(zeros)


#: How many jumps of a step a period of its argument may hold.
_PERIOD_JUMPS = 16


class _Lattice(NamedTuple):
    """The jumps of a step function at `start + k*spacing` for every integer `k`:
    `values(k)` is what the step holds between the `k`-th jump and the next, `count`
    is `k` as a function of the variable there, and `alternating` says the step is a
    sign that changes at each jump rather than a floor that counts them."""

    start: sp.Basic
    spacing: sp.Basic
    values: Callable[[sp.Basic], sp.Basic]
    count: sp.Basic
    alternating: bool


def across_the_steps(head: sp.Basic) -> sp.Basic | None:
    """An indefinite integral of a step function, valid across all of its jumps.

    Sympy integrates `ABS(SIN(x))` over the pieces it finds near the origin and
    writes the last one out to infinity: `IF(x <= 0, COS(x), IF(x <= pi, 2 - COS(x),
    COS(x) + 4))` is an antiderivative between `-pi` and `2*pi` and nowhere else.
    A step whose argument crosses its jumps infinitely often - a sign of a periodic
    function, or a rounding function at all - has no such answer, and whatever is
    returned here is either an antiderivative on the whole line or the integral
    left standing.

    The jumps are where the step's argument crosses zero (or an integer, for a
    rounding function), at `start + k*spacing` for every integer `k`, and between
    two of them the step is a constant: a sign that alternates, or the integer `k`
    counted from the start. With that constant written as `w`, the integrand is a
    function `G` of `x` and `w` to integrate as any other, and on the `k`-th stretch
    `G(x, w_k) + C(k)` is an antiderivative for any constant `C(k)`. It is one
    continuous across every jump where `C(k + 1) - C(k)` is what `G` drops there,
    and that sum is had in closed form for a polynomial in `k` or one times `(-1)^k`.
    Writing `k` as the `FLOOR` it is and `(-1)^k` as the sign it is gives the answer:
    `2*FLOOR(x/pi) - COS(x)*SIGN(SIN(x))` for `ABS(SIN(x))`, and `x*FLOOR(x) -
    FLOOR(x)*(FLOOR(x) + 1)/2` for `FLOOR(x)`. The answer is continuous wherever the
    integrand is integrable, apart from the jumps themselves, where a sign is zero.

    None where there is no step, or none whose jumps go on forever: a sign or
    absolute value of a polynomial has finitely many, and sympy's split is right.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 1:
        return None
    variable = head.limits[0][0]
    if variable.is_real is not True or not _endless_steps(head.function, variable):
        return None
    total = sp.S.Zero
    for term in sp.Add.make_args(head.function):
        if not _endless_steps(term, variable):
            part = indefinite.integral(sp.Integral(term, variable))
        else:
            part = _over_the_lattice(term, variable)
        if part is None or part.has(sp.Integral):
            return head
        total += part
    constant, _ = sp.expand(total).as_independent(variable, as_Add=True)
    return folded_signs(sp.expand(total - constant))


def _endless_steps(integrand: sp.Basic, variable: sp.Basic) -> bool:
    """Whether a step in the integrand may jump infinitely often: a rounding
    function, or a sign or absolute value of something other than a polynomial."""
    return any(
        step.has(variable)
        and (isinstance(step, _ROUNDING) or not step.args[0].is_polynomial(variable))
        for step in integrand.atoms(*_STEPS)
    )


def _over_the_lattice(term: sp.Basic, variable: sp.Basic) -> sp.Basic | None:
    """One term of `across_the_steps`, with one step in it, or None."""
    level = sp.Dummy("w", real=True)
    index = sp.Dummy("k", integer=True)
    written = attempt(term, _in_signs_and_floors)
    if written is None:
        return None
    steps = {step for step in written.atoms(sp.sign, sp.floor) if step.has(variable)}
    if len(steps) != 1:
        return None
    (step,) = steps
    lattice = _lattice(step, variable)
    if lattice is None:
        return None
    smooth = written.xreplace({step: level})
    if any(inner.has(variable) for inner in smooth.atoms(*_STEPS)):
        return None
    antiderivative = indefinite.integral(sp.Integral(smooth, variable))
    if antiderivative.has(sp.Integral, sp.Piecewise, sp.nan, sp.zoo):
        return None
    jump = lattice.start + (index + 1) * lattice.spacing
    drop = attempt(
        antiderivative,
        lambda g: sp.expand(
            g.subs({variable: jump, level: lattice.values(index)})
            - g.subs({variable: jump, level: lattice.values(index + 1)})
        ),
    )
    if drop is None or drop.has(sp.nan, sp.zoo):
        return None
    cumulative = _cumulative(drop, index, lattice.alternating)
    if cumulative is None:
        return None
    plain, alternate = cumulative
    value = antiderivative.subs(level, step) + plain.subs(index, lattice.count)
    if lattice.alternating:
        # Between two jumps `(-1)^k` is the sign times the sign it starts with.
        value += lattice.values(0) * step * alternate.subs(index, lattice.count)
    return value


def _in_signs_and_floors(term: sp.Basic) -> sp.Basic:
    """The term with every step written as a sign or a floor, as it is between its
    jumps: `ABS(u)` is `u*SIGN(u)`, `STEP(u)` is `(1 + SIGN(u))/2`, `CEILING(u)` is
    `FLOOR(u) + 1`, and the fractional part is `u - FLOOR(u)`."""
    return (
        term.replace(sp.Abs, lambda u: u * sp.sign(u))
        .replace(sp.Heaviside, lambda u, *_: (1 + sp.sign(u)) / 2)
        .replace(sp.ceiling, lambda u: sp.floor(u) + 1)
        .replace(sp.frac, lambda u: u - sp.floor(u))
    )


def _lattice(step: sp.Basic, variable: sp.Basic) -> _Lattice | None:
    """Where a sign or floor of a function of the variable alone jumps, as a
    lattice, or None where its jumps are not equally spaced.

    A floor jumps where a linear argument crosses an integer. A sign jumps where a
    continuous periodic argument crosses zero, and is taken where those crossings
    are equally spaced and the sign alternates across each, as it does for a sine
    or cosine of a linear argument.
    """
    argument = step.args[0]
    if argument.free_symbols != {variable}:
        return None
    if isinstance(step, sp.floor):
        try:
            polynomial = sp.Poly(argument, variable)
        except sp.PolynomialError:
            return None
        if polynomial.degree() != 1:
            return None
        slope, offset = polynomial.all_coeffs()
        if slope.is_extended_positive:
            return _Lattice(-offset / slope, 1 / slope, lambda k: k, step, False)
        if slope.is_extended_negative:
            return _Lattice(
                -offset / slope, -1 / slope, lambda k: -k - 1, -step - 1, False
            )
        return None
    period = attempt(argument, lambda a: sp.periodicity(a, variable))
    if period is None or not period.is_extended_positive or period.free_symbols:
        return None
    window = sp.Interval.Ropen(0, period)
    poles = attempt(argument, lambda a: sp.singularities(a, variable, window))
    if poles is not sp.S.EmptySet:
        return None
    zeros = attempt(argument, lambda a: sp.solveset(a, variable, window))
    if not isinstance(zeros, sp.FiniteSet) or not 0 < len(zeros) <= _PERIOD_JUMPS:
        return None
    ordered = _ordered(set(zeros))
    if ordered is None or len(ordered) % 2:
        return None
    spacing = period / len(ordered)
    if any(
        sp.simplify(right - left - spacing) != 0
        for left, right in zip(ordered, [*ordered[1:], ordered[0] + period])
    ):
        return None
    start = ordered[0] if ordered[0] == 0 else ordered[0] - spacing
    middles = [start + (2 * i + 1) * spacing / 2 for i in range(len(ordered))]
    signs = [
        attempt(argument, lambda a: sp.sign(a.subs(variable, middle)))
        for middle in middles
    ]
    first = signs[0]
    if first not in (1, -1) or any(s != first * (-1) ** i for i, s in enumerate(signs)):
        return None
    count = sp.floor(sp.expand((variable - start) / spacing))
    return _Lattice(start, spacing, lambda k: first * (-1) ** k, count, True)


def _cumulative(
    difference: sp.Basic, index: sp.Basic, alternating: bool
) -> tuple[sp.Basic, sp.Basic] | None:
    """`C(k)` with `C(0) = 0` and `C(k + 1) - C(k)` the difference, for every
    integer `k`, as polynomials `P` and `Q` in `k` with `C(k) = P(k) + (-1)^k*Q(k)`,
    or None where the difference is not a polynomial in `k` or one times `(-1)^k`.

    Each is had by undetermined coefficients: `P(k + 1) - P(k)` is the plain part
    of the difference, and `-Q(k + 1) - Q(k)` its alternating part. The result is
    checked against the difference it was solved for.
    """
    parity = sp.Dummy("s")
    signed = difference.replace(
        lambda e: isinstance(e, sp.Pow) and e.base == -1 and e.exp.has(index),
        lambda e: _parity(e.exp, index, parity),
    )
    if signed.has(sp.Pow) and any(
        power.base == -1 and power.exp.has(index) for power in signed.atoms(sp.Pow)
    ):
        return None
    reduced = attempt(signed, lambda d: sp.rem(sp.expand(d), parity**2 - 1, parity))
    if reduced is None:
        return None
    plain_part = reduced.coeff(parity, 0)
    signed_part = reduced.coeff(parity, 1)
    if signed_part != 0 and not alternating:
        return None
    parts = []
    for part, shift in ((plain_part, 1), (signed_part, -1)):
        if part.is_polynomial(index) is not True:
            return None
        degree = sp.degree(part, index) if part != 0 else 0
        degree = max(degree, 0) + (1 if shift == 1 else 0)
        coefficients = sp.symbols(f"c0:{degree + 1}", cls=sp.Dummy)
        ansatz = sum(c * index**i for i, c in enumerate(coefficients))
        equation = sp.expand(shift * ansatz.subs(index, index + 1) - ansatz - part)
        try:
            solved = sp.solve(
                sp.Poly(equation, index).all_coeffs(), coefficients, dict=True
            )
        except Exception:
            return None
        if not solved:
            return None
        found = ansatz.subs(solved[0]).subs({c: 0 for c in coefficients})
        parts.append(found)
    plain, alternate = parts
    total = (
        plain
        - plain.subs(index, 0)
        + (-1) ** index * alternate
        - alternate.subs(index, 0)
    )
    check = attempt(
        total,
        lambda t: sp.simplify(t.subs(index, index + 1) - t - difference),
    )
    if check != 0:
        return None
    return plain - plain.subs(index, 0) - alternate.subs(index, 0), alternate


def _parity(exponent: sp.Basic, index: sp.Basic, parity: sp.Basic) -> sp.Basic:
    """`(-1)^exponent` for an exponent linear in the index, over `parity` standing
    for `(-1)^index`; unchanged where the exponent is anything else."""
    try:
        polynomial = sp.Poly(exponent, index)
    except sp.PolynomialError:
        return sp.Pow(-1, exponent)
    if polynomial.degree() != 1:
        return sp.Pow(-1, exponent)
    slope, offset = polynomial.all_coeffs()
    if not slope.is_Integer:
        return sp.Pow(-1, exponent)
    return parity**slope * sp.Pow(-1, offset)


def _held(step: sp.Basic, variable: sp.Basic, point: sp.Basic) -> sp.Basic | None:
    """What a step function is between two of its jumps, from its value at a point
    between them, or None where that value is not decided."""
    argument = step.args[0]
    if isinstance(step, sp.Abs):
        sign = attempt(argument, lambda a: sp.sign(a.subs(variable, point)))
        return sign * argument if sign in (1, -1) else None
    if isinstance(step, sp.frac):
        level = attempt(argument, lambda a: sp.floor(a.subs(variable, point)))
        return argument - level if level is not None and level.is_Integer else None
    value = attempt(step, lambda s: s.subs(variable, point))
    return value if value is not None and value.is_Number else None


#: How many signs of a parameter an integral is asked again over.
_THRESHOLDS = 3


def by_the_sign_of_a_parameter(head: sp.Basic, value: sp.Basic) -> sp.Basic | None:
    """A definite integral over a parameter, worked out on each side of its sign.

    What sympy makes of `INT(#e^(-a*x), x, 0, inf)` depends on what it knows of
    `a`. Told nothing it answers `1/a` where the argument of `a` is less than a
    right angle and leaves the integral standing elsewhere; through the
    antiderivative it takes `INT(1/(x^2 + a^2), x, 0, inf)` to `ATAN(x/a)/a` at an
    infinite `x`, and that is `ATAN(±inf/a)`, with the sign of the infinity
    undecided; at the lower end of `INT(x^n, x, 0, 1)` it is left holding `0^(n +
    1)`, which is 0 or infinite by the sign of `n + 1`. Told the sign, it answers:
    `1/a` and `inf`, `pi/(2*a)` for a positive `a`, `1/(n + 1)` and `inf`.

    So such an answer is asked again with the parameter on each side of a
    threshold - above, below, and at it - and the three answers are put together.
    The thresholds are the quantities the answer turned on: the exponent of a
    power of zero, a condition sympy wrote, the exponent of a power of the
    variable plus one (which decides whether it is integrable at zero), and the
    parameter itself. Where one expression is all three answers - itself, of the
    absolute value of the threshold, or that times its sign - it is the answer
    (`pi/(2*ABS(a))`, `pi*SIGN(a)/2`); otherwise the answer is a case split (`IF(a >
    0, 1/a, inf)`). A side that sympy cannot do, or does only with another
    condition, leaves the answer as it was.

    An integral sympy left standing over the unsigned parameter is not asked again
    this way: what it could not do knowing nothing of the sign it cannot do knowing
    it, since a sign it was not told leaves it more to answer with, under a
    condition, and not less. `_integral_through` asks only over an answer that turned
    on the sign, or over a head sympy was not asked about because the integrand's
    behaviour at an end was undecided.
    """
    if not definite(head) or not unresolved(head, value):
        return None
    variable = head.limits[0][0]
    parameters = head.free_symbols
    if len(parameters) != 1:
        return None
    (parameter,) = parameters
    if not (parameter.is_extended_real and parameter.is_positive is None):
        return None
    if parameter.is_negative is not None or parameter.is_zero is not None:
        return None
    for threshold in _thresholds(head, value, parameter, variable)[:_THRESHOLDS]:
        found = _on_each_side(head, parameter, threshold)
        if found is not None:
            return found
    signed = attempt(value, lambda v: _with_signed_conditions(v, parameter))
    return None if signed is None or signed == value else signed


def _with_signed_conditions(value: sp.Basic, parameter: sp.Basic) -> sp.Basic:
    """A case split over a real parameter with each condition written as its sign.

    Sympy writes the convergence of `INT(#e^(-a*x)*SIN(x), x, 0, inf)` as `ABS(PHASE(a))
    < pi/2`, which over a real `a` is `a > 0`. A condition in the parameter alone
    that is decided on each side of zero and at it is written as the comparison with
    zero that holds on the same sides; any other is kept. One undecided at zero
    alone, as a phase is, is taken not to hold there where every case after it is
    the integral left standing, which is no wrong answer for any value.
    """
    size = sp.Dummy("p", positive=True)

    def signed(condition: sp.Basic, standing: bool) -> sp.Basic:
        if condition.free_symbols != {parameter}:
            return condition
        truths = [
            attempt(condition, lambda c: c.subs(parameter, point))
            for point in (size, -size, sp.S.Zero)
        ]
        if truths[2] not in (sp.true, sp.false) and standing:
            truths[2] = sp.false
        if not all(truth in (sp.true, sp.false) for truth in truths):
            return condition
        return _SIGNED[tuple(bool(truth) for truth in truths)](parameter)

    def cases(piecewise: sp.Piecewise) -> sp.Piecewise:
        args = piecewise.args
        rewritten = []
        for i, (case, condition) in enumerate(args):
            standing = all(later.has(sp.Integral) for later, _ in args[i + 1 :])
            rewritten.append((case, signed(condition, standing)))
        return sp.Piecewise(*rewritten)

    return value.replace(lambda e: isinstance(e, sp.Piecewise), cases)


#: The comparison with zero that holds where a parameter is positive, negative,
#: and zero, as given.
_SIGNED = {
    (True, False, False): lambda a: sp.StrictGreaterThan(a, 0),
    (False, True, False): lambda a: sp.StrictLessThan(a, 0),
    (True, True, False): lambda a: sp.Ne(a, 0),
    (True, False, True): lambda a: sp.GreaterThan(a, 0),
    (False, True, True): lambda a: sp.LessThan(a, 0),
    (False, False, True): lambda a: sp.Eq(a, 0),
    (True, True, True): lambda a: sp.true,
    (False, False, False): lambda a: sp.false,
}


def unresolved(head: sp.Basic, value: sp.Basic) -> bool:
    """Whether an integral's answer is one that turns on a sign it does not know."""
    if value == head or value.has(sp.Integral, sp.Limit, sp.zoo, sp.nan):
        return True
    return any(_zero_power(power) for power in value.atoms(sp.Pow)) or any(
        gamma.args[1] == 0 for gamma in value.atoms(sp.uppergamma)
    )


def _zero_power(power: sp.Pow) -> bool:
    """Whether this is a power of zero, which is 0 or infinite by its exponent's sign."""
    return power.base == 0 and bool(power.exp.free_symbols)


def _thresholds(
    head: sp.Basic, value: sp.Basic, parameter: sp.Basic, variable: sp.Basic
) -> list[sp.Basic]:
    """The quantities an integral's answer may turn on, each the parameter plus a
    number, most telling first."""
    candidates = [power.exp for power in value.atoms(sp.Pow) if _zero_power(power)]
    candidates += [gamma.args[0] for gamma in value.atoms(sp.uppergamma)]
    candidates += [
        relation.lhs - relation.rhs
        for relation in value.atoms(Relational)
        if relation.free_symbols == {parameter}
    ]
    candidates += [
        power.exp + 1
        for power in head.function.atoms(sp.Pow)
        if power.base == variable and power.exp.has(parameter)
    ]
    candidates.append(parameter)
    thresholds: list[sp.Basic] = []
    for candidate in candidates:
        try:
            polynomial = sp.Poly(candidate, parameter)
        except sp.PolynomialError:
            continue
        if polynomial.degree() != 1 or not all(
            c.is_Rational for c in polynomial.all_coeffs()
        ):
            continue
        threshold = sp.expand(candidate / polynomial.LC())
        if threshold not in thresholds:
            thresholds.append(threshold)
    return thresholds


def _on_each_side(
    head: sp.Basic, parameter: sp.Basic, threshold: sp.Basic
) -> sp.Basic | None:
    """The integral as one answer for every value of the parameter, from its answers
    where `threshold`, the parameter plus a number, is positive, negative and zero."""
    shift = threshold - parameter
    size = sp.Dummy("p", positive=True)
    answers = []
    for point in (size - shift, -size - shift, -shift):
        answer = attempt(head, lambda h: indefinite.integral(h.subs(parameter, point)))
        if answer is None or not _clean(answer):
            return None
        answers.append(answer)
    above, below, at = answers
    candidates = [
        above.subs(size, threshold),
        above.subs(size, sp.Abs(threshold)),
        sp.sign(threshold) * above.subs(size, sp.Abs(threshold)),
    ]
    for candidate in candidates:
        if agrees(candidate.subs(parameter, -size - shift), below) and agrees_at(
            candidate, parameter, -shift, at
        ):
            return candidate
    upper = above.subs(size, threshold)
    lower = below.subs(size, -threshold)
    over = sp.StrictGreaterThan(parameter, -shift)
    under = sp.StrictLessThan(parameter, -shift)
    if upper == lower:
        return sp.Piecewise((upper, sp.Ne(parameter, -shift)), (at, True))
    if lower == at:
        return sp.Piecewise((upper, over), (at, True))
    if upper == at:
        return sp.Piecewise((lower, under), (at, True))
    return sp.Piecewise((upper, over), (lower, under), (at, True))


def _clean(answer: sp.Basic) -> bool:
    """Whether an answer is a value: worked out, and an infinity, or no value at all,
    only on its own."""
    if answer is sp.nan:
        return True
    if answer.has(sp.Integral, sp.Piecewise, sp.zoo, sp.nan, sp.uppergamma):
        return False
    if any(_zero_power(power) for power in answer.atoms(sp.Pow)):
        return False
    return answer in (sp.oo, -sp.oo) or not answer.has(sp.oo, -sp.oo)
