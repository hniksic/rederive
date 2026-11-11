"""Definite integrals: fixes for what sympy gets wrong between limits.

Sympy integrates through interior singularities, handles step functions only partially,
and answers under parameter conditions it could have split on. These functions examine
the integrand before sympy runs and fix its answer afterwards. The dispatch in
`indefinite` calls them, and any integral they need goes back through it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import sympy as sp
from sympy.core.relational import Relational
from sympy.functions.elementary.trigonometric import TrigonometricFunction

# Import the module rather than its names, since `indefinite` imports from this one.
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
    "between_the_ends",
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
    """A definite integral's value, rejected if the integral is real but the value is not
    proved real.

    Sympy answers `INT(LN(x)/(x + 1), x, 0, 1)` with an imaginary part, from evaluating
    an antiderivative across a branch cut. If `Stretch` proves the integrand real on
    the interval, a value with `#i` is kept only if its imaginary part is proved zero,
    and is otherwise replaced by `by_the_antiderivative` or left unevaluated. A value
    with an infinite or unevaluated imaginary part is rejected too, since evaluating it
    can take minutes.
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
    """Whether a convergent integral over a finite interval is zero because its
    parameter-free integrand changes sign when reflected about the middle.

    This handles `INT(LN(TAN(x)), x, 0, pi/2)`, which has no antiderivative sympy
    finds. The integrand plus its reflection is proved zero: its derivative is zero, it
    is continuous, and it is zero at the middle.
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
    """A definite integral sympy left unevaluated, as the endpoint difference of
    Simplify's antiderivative if `unbroken` proves that correct, or None.

    Sympy has no antiderivative for `INT(SIN(x)/SQRT(1 + q^2*COS(x)^2), x, 0, pi)`, but
    `u = COS(x)` makes it elementary. Integrands `within_reach` rules out are skipped.

    The proof says nothing where the parameters make the antiderivative undefined, so a
    value that is defined there is rejected: for `SIN(m*x)*SIN(2*x)` from `0` to `pi`,
    the generic answer `0` is wrong at `m = 2`.
    """
    if head.function.has(sp.Piecewise) or indefinite.within_reach(head):
        return None
    variable, low, high = head.limits[0]
    if _orientation(low, high) is None:
        return None
    antiderivative = indefinite.integral(sp.Integral(head.function, variable))
    return between_the_ends(head, antiderivative)


def between_the_ends(head: sp.Basic, antiderivative: sp.Basic) -> sp.Basic | None:
    """A definite integral as the endpoint difference of an antiderivative of its
    integrand if `unbroken` proves that correct, or None; see `by_the_antiderivative`.

    Limits whose order isn't known, as in `INT(COS(x)*(SIN(x) + 2)^12, x, 0, a)`, are
    taken only for an antiderivative continuous on the whole real line and without a
    parameter in a divisor, and the ends are then substituted.
    """
    if antiderivative.has(sp.Integral, sp.Piecewise):
        return None
    variable, low, high = head.limits[0]
    orientation = _orientation(low, high)
    if orientation is None:
        if parametric_divisors(antiderivative, variable) or not unbroken(
            antiderivative, variable, -sp.oo, sp.oo
        ):
            return None
        at_ends = [
            attempt(antiderivative, lambda a: a.subs(variable, end))
            for end in (high, low)
        ]
        if None in at_ends:
            return None
        value = at_ends[0] - at_ends[1]
        return None if value.has(sp.nan, sp.zoo, sp.oo, -sp.oo) else value
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
    """Whether the value is (or may be) defined at a parameter value where the
    antiderivative divides by zero; see `by_the_antiderivative`.
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
    """The antiderivative's limits at the upper and then the lower end, approached from
    inside, or None if either can't be computed.

    Sympy leaves the limit of `ATAN((COS(a) + x)/ABS(SIN(a)))` at infinity unevaluated,
    not knowing `ABS(SIN(a))` is nonzero, so `_positively` retries with it assumed
    positive. If given, `near(end)` supplies the form to take the limit of at each end.
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
    """The limit of an antiderivative with absolute values and square roots of parameters
    assumed positive, or None; see `ends`.

    Valid only if the antiderivative and the limit are both undefined when each of them
    is zero. Square roots qualify only if their radicand is never negative.
    """
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
    """A definite integral from a singular endpoint to a symbolic one, valued by how it
    diverges there, or None.

    `INT(1/x^2, x, 0, alpha)` comes back as `inf - 1/alpha`, correct only for positive
    `alpha`. If the endpoint is the integrand's only real singularity and the integrand
    is real on both sides, the integral diverges to the infinity `_end_verdict` finds,
    whatever the other end: `INT(COS(a*x)/x, x, 0, alpha)` is `inf`. If the two sides
    disagree, the integral is left unevaluated.
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
    # Integrating up from the singular end on its upper side, or down to it from there.
    beyond, before = (above, -below) if point == low else (-above, below)
    return beyond if beyond == before else head


def beyond_convergence(head: sp.Basic, value: sp.Basic) -> sp.Basic | None:
    """A definite integral with its divergent case evaluated, if it diverges to a signed
    infinity.

    Sympy answers `INT(x^n, x, 1, inf)` with `-1/(n + 1)` for `n < -1` and the
    unevaluated integral otherwise, where Derive prints `inf`. The complement of the
    condition is expressed as substitutions (`n = -1 - s` with positive `s`, and `n =
    -1`), and if the integral is the same infinity under each, that is the value.
    Otherwise it is left alone: a convergence test proves nothing about divergence, and
    an oscillating integrand diverges to no signed infinity.
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


#: The inequalities used in convergence conditions, and the only ones whose complement
#: can be expressed by giving the parameter a sign.
_INEQUALITIES = (sp.StrictLessThan, sp.StrictGreaterThan, sp.LessThan, sp.GreaterThan)


def _complement(condition: sp.Basic) -> list[tuple[sp.Basic, sp.Basic]]:
    """Where a one-parameter inequality fails, as values of its parameter.

    The complement of `g > 0` is `g = -s` for positive `s`, plus `g = 0` if the
    inequality is strict. Any other condition gives an empty list.
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


#: What `at_the_singularities` returns for an integral with no infinity found: every
#: end examined converges, or some end could not be decided.
CONVERGENT = sp.Symbol("convergent")
_UNSETTLED = sp.Symbol("unsettled")
UNDECIDED = sp.Symbol("undecided")
VERDICTS = (CONVERGENT, _UNSETTLED, UNDECIDED)

#: The maximum number of singularities in an interval to examine.
_POLES = 40


def at_the_singularities(head: sp.Basic) -> sp.Basic | None:
    """A definite integral decided by the integrand's behavior at its singularities.

    An integral across a singularity exists only if both improper halves do, but an
    antiderivative taken straight through gives a number anyway: -2 for `INT(1/x^2, x,
    -1, 1)` instead of `inf`. So the interval is cut at every singularity sympy can
    locate, and `_end_verdict` examines each side and any infinite end. One infinity
    with all other ends converging is the integral; two of opposite signs give `?`. If
    everything converges, the parts are integrated separately, since an antiderivative
    can jump at an interior singularity; with no interior cut the result is
    `CONVERGENT`, meaning sympy should be asked.

    `UNDECIDED` means an end diverges with unknown sign (`a/x^2` at zero), so sympy's
    answer is unusable. `_UNSETTLED` means an end couldn't be decided (`1/(x*LN(x))` at
    zero), and sympy's answer is used unless infinite. Returns None if there is no
    singularity in the interval, an end isn't numeric, or a singularity moves with a
    parameter.
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
    """A definite integral of a rational function whose poles move with a parameter, split
    on where the poles are, or None if no pole can be in the interval.

    `INT(1/(x - a)^2, x, 0, 2)` via its antiderivative is `-2/(a*(2 - a))`, wrong for
    `a` in `[0, 2]`, where the non-integrable pole is inside. The parameter values that
    put a real pole in the interval are cut wherever the integral's behavior can change
    (a pole reaching an end, poles meeting, cancelling or going to infinity), and the
    integral is computed at each cut and one sample per stretch. The existing answer
    covers the remaining values. Poles real only at isolated parameter values are left
    to the formula, which blows up there too.
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
    """The parameter values for which a pole is real, or None if they can't be found.

    Radicands are checked rather than the square of the pole where possible, since
    sympy considers the square of `SQRT(COS(a)^2 - 1) - COS(a)` nonnegative for every
    `a`. A periodic pole is solved over one period, so only isolated points are
    accepted.
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
    """Whether the call is an absolute value, or a square root whose radicand is a
    function of one real parameter that is nonnegative for every value of it."""
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
    """Real numbers in increasing order, or None if they can't be ordered. Points are
    compared by the sign of their difference, so `p` and `2*p` work for positive `p`.
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


#: Powers of the distance used to show convergence at an end: an integrand bounded by
#: such a power of the reciprocal distance to a point, or of the reciprocal variable
#: toward an infinite end, is integrable there.
_NEAR = (sp.Rational(1, 2), sp.Rational(9, 10))
_FAR = (sp.Integer(2), sp.Rational(11, 10))


def _end_verdict(
    integrand: sp.Basic, variable: sp.Basic, end: sp.Basic, side: int
) -> sp.Basic:
    """How the integral behaves at one end of a part of the interval.

    `side` is the direction of approach (`1` from above), and `distance` is the
    distance to the end, or the variable's magnitude at an infinite end. If the
    integrand times the distance tends to a nonzero value, the part diverges to the
    infinity of its sign (`UNDECIDED` if unknown). If it stays bounded times a power of
    the distance below 1 (above 1 at infinity), the part converges by comparison.
    Otherwise it is `_UNSETTLED`.
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


#: Step functions at whose jumps an integrand is split. Between jumps, each is a
#: constant or its argument times a constant sign.
_STEPS = (sp.floor, sp.ceiling, sp.frac, sp.sign, sp.Abs, sp.Heaviside)
_ROUNDING = (sp.floor, sp.ceiling, sp.frac)

#: The maximum number of parts an interval is split into at step jumps.
_STEP_PIECES = 64


def between_the_steps(head: sp.Basic) -> sp.Basic | None:
    """A definite integral of a step function, as a sum over the steps.

    Sympy has no antiderivative for `INT(FLOOR(x), x, 0, 3)`. With numeric limits there
    are finitely many jumps, so each piece is integrated with the steps replaced by
    their values there. Only integrands sympy mishandles are split: those with a
    rounding function, or a sign or absolute value of a non-polynomial, where sympy's
    antiderivative is valid only near the origin (`INT(ABS(SIN(x)), x, 0, 10*pi)` gives
    4 instead of 20). Step arguments must be parameter-free, and for rounding functions
    at most quadratic, so the jumps are exact numbers.
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
    """The points strictly between the limits where a step function jumps, or None if
    they can't be found as exact numbers."""
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


#: The maximum number of jumps of a step within one period of its argument.
_PERIOD_JUMPS = 16


class _Lattice(NamedTuple):
    """The jumps of a step function at `start + k*spacing` for every integer `k`.

    `values(k)` is the step's value after the `k`-th jump, `count` is `k` as a function
    of the variable, and `alternating` is true for a sign rather than a floor.
    """

    start: sp.Basic
    spacing: sp.Basic
    values: Callable[[sp.Basic], sp.Basic]
    count: sp.Basic
    alternating: bool


def across_the_steps(head: sp.Basic) -> sp.Basic | None:
    """An indefinite integral of a step function, valid across all of its jumps.

    Sympy's antiderivative of `ABS(SIN(x))` is valid only between `-pi` and `2*pi`. For
    a step with infinitely many jumps, this returns an antiderivative on the whole line
    or the unevaluated integral.

    Between jumps at `start + k*spacing`, the step is a constant `w` (an alternating
    sign or the integer `k`), so the integrand is an ordinary function `G(x, w)`.
    Adding a constant `C(k)` to each stretch with `C(k + 1) - C(k)` equal to the drop
    in `G` makes it continuous, and that sum has a closed form when the drop is
    polynomial in `k`, possibly times `(-1)^k`. Writing `k` as a `FLOOR` and `(-1)^k`
    as a sign gives `2*FLOOR(x/pi) - COS(x)*SIGN(SIN(x))` for `ABS(SIN(x))`.

    Returns None if no step has infinitely many jumps.
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
    """The integral of one term of `across_the_steps`, containing one step, or None."""
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
        # Between two jumps, `(-1)^k` is the step's sign times its initial sign.
        value += lattice.values(0) * step * alternate.subs(index, lattice.count)
    return value


def _in_signs_and_floors(term: sp.Basic) -> sp.Basic:
    """The term with every step written as a sign or floor, valid between jumps:
    `ABS(u)` is `u*SIGN(u)`, `STEP(u)` is `(1 + SIGN(u))/2`, `CEILING(u)` is
    `FLOOR(u) + 1`, and the fractional part is `u - FLOOR(u)`."""
    return (
        term.replace(sp.Abs, lambda u: u * sp.sign(u))
        .replace(sp.Heaviside, lambda u, *_: (1 + sp.sign(u)) / 2)
        .replace(sp.ceiling, lambda u: sp.floor(u) + 1)
        .replace(sp.frac, lambda u: u - sp.floor(u))
    )


def _lattice(step: sp.Basic, variable: sp.Basic) -> _Lattice | None:
    """The jumps of a sign or floor of a function of the variable alone, as a lattice, or
    None if they aren't equally spaced.

    A floor needs a linear argument. A sign needs a continuous periodic argument whose
    zeros are equally spaced with the sign alternating, as for a sine of a linear
    argument.
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
    """`C(k)` with `C(0) = 0` and `C(k + 1) - C(k)` equal to the difference, as
    polynomials `P` and `Q` with `C(k) = P(k) + (-1)^k*Q(k)`, or None if the difference
    isn't of that form. Found by undetermined coefficients and verified.
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
    """`(-1)^exponent` for an exponent linear in the index, written in terms of `parity`
    standing for `(-1)^index`; unchanged for any other exponent."""
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
    """What a step function equals between two jumps, judged by its value at a point
    between them, or None if that value is undecided."""
    argument = step.args[0]
    if isinstance(step, sp.Abs):
        sign = attempt(argument, lambda a: sp.sign(a.subs(variable, point)))
        return sign * argument if sign in (1, -1) else None
    if isinstance(step, sp.frac):
        level = attempt(argument, lambda a: sp.floor(a.subs(variable, point)))
        return argument - level if level is not None and level.is_Integer else None
    value = attempt(step, lambda s: s.subs(variable, point))
    return value if value is not None and value.is_Number else None


#: The maximum number of thresholds to split an integral's parameter on.
_THRESHOLDS = 3


def by_the_sign_of_a_parameter(head: sp.Basic, value: sp.Basic) -> sp.Basic | None:
    """A definite integral with a parameter, computed separately for each sign of the
    parameter.

    Without knowing a sign, sympy answers under conditions or leaves expressions like
    `ATAN(+/-inf/a)` or `0^(n + 1)`; given the sign, it answers. So the integral is
    recomputed with the parameter above, below and at a threshold: the exponent of a
    power of zero, a condition sympy wrote, the exponent plus one of a power of the
    variable, or the parameter itself. If one expression matches all three answers (in
    the threshold, its absolute value, or that times its sign), it is the answer, as
    `pi/(2*ABS(a))`; otherwise it is a case split like `IF(a > 0, 1/a, inf)`.

    An integral sympy couldn't do at all isn't retried, since without the sign it could
    still have answered under a condition.
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
    """A case split over a real parameter with each condition rewritten as a sign test.

    Sympy writes `ABS(PHASE(a)) < pi/2`, which for real `a` is `a > 0`. Conditions
    decided for positive, negative and zero values are replaced; a condition undecided
    only at zero is taken as false there if every later case is the unevaluated
    integral, which is never wrong.
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


#: The comparison with zero that holds exactly when the parameter is positive,
#: negative, and zero, as the key flags say.
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
    """Whether an integral's answer depends on an unknown sign."""
    if value == head or value.has(sp.Integral, sp.Limit, sp.zoo, sp.nan):
        return True
    return any(_zero_power(power) for power in value.atoms(sp.Pow)) or any(
        gamma.args[1] == 0 for gamma in value.atoms(sp.uppergamma)
    )


def _zero_power(power: sp.Pow) -> bool:
    """Whether this is a power of zero, which is 0 or infinite depending on its
    exponent's sign."""
    return power.base == 0 and bool(power.exp.free_symbols)


def _thresholds(
    head: sp.Basic, value: sp.Basic, parameter: sp.Basic, variable: sp.Basic
) -> list[sp.Basic]:
    """The quantities an integral's answer may depend on, each the parameter plus a
    number, most likely first."""
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
    """The integral as one answer for every parameter value, combined from its answers
    where `threshold` (the parameter plus a number) is positive, negative and zero."""
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
    """Whether an answer is fully evaluated, with an infinity or undefined value only
    as the whole answer."""
    if answer is sp.nan:
        return True
    if answer.has(sp.Integral, sp.Piecewise, sp.zoo, sp.nan, sp.uppergamma):
        return False
    if any(_zero_power(power) for power in answer.atoms(sp.Pow)):
        return False
    return answer in (sp.oo, -sp.oo) or not answer.has(sp.oo, -sp.oo)
