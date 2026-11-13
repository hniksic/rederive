"""Choosing the printed form of an antiderivative.

`antiderivative` picks which antiderivative of an accepted answer to print. The form
checks `indefinite` applies before accepting an answer also live here.
"""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp

from rederive.engine.integration.proof import differentiates_back, holds, proved_zero
from rederive.engine.printer import is_written
from rederive.engine.rational import over_a_parametric_quadratic
from rederive.engine.rewriting import attempt, forced, pays
from rederive.engine.vocabulary import INVERSE_HYPERBOLIC, POLAR, in_vocabulary

__all__ = [
    "antiderivative",
    "anywhere",
    "continuous",
    "on_the_principal_sheet",
    "parametric_divisors",
    "parametrised",
    "single_case",
    "somewhere_only",
    "unwritable",
]


def antiderivative(head: sp.Basic, value: sp.Basic, tabled: bool = False) -> sp.Basic:
    """An accepted answer to an integral, in the form Derive writes it.

    Derive prints the antiderivative with no added constant, and avoids a case split
    where one antiderivative works everywhere; sympy's tables do neither. Definite and
    unevaluated integrals are left alone, apart from dropping a parameter split that
    distinguishes nothing.

    Unless the answer is `tabled`, a parametrized answer with a logarithm is offered to
    the manual rules for a shorter form. The answer is assumed already proved.
    """
    if len(head.limits) != 1 or value.has(sp.Integral):
        return value
    if len(head.limits[0]) != 1:
        return continuous(value, None)
    variable, integrand = head.limits[0][0], head.function
    if (
        not tabled
        and parametrised(head)
        and value.has(sp.log)
        and not over_a_parametric_quadratic(integrand, variable)
        and not _plainly_rational(value, integrand, variable)
    ):
        value = _shorter_by_the_manual_rules(head, value, integrand, variable)
    value = _without_constants(single_case(value, integrand, variable), variable)
    value = _with_matching_logarithms(continuous(value, variable), variable)
    return forced(value, lambda v: _with_a_factored_rational_part(v, integrand, variable))


#: The 6.11 integrals with a logarithmic singularity at zero, mapped to the logarithm of
#: the same argument that cancels it: `CI(z) - LN(z)` and `COSH_INT(z) - LN(z)` are
#: entire (the two share a branch cut), and `EI(z) - LN(z^2)/2` is continuous and real
#: on the real line, where `EI` is real on both sides of zero.
_WITH_A_LOGARITHM: dict[type, Callable[[sp.Basic], sp.Basic]] = {
    sp.Ci: sp.log,
    sp.Chi: sp.log,
    sp.Ei: lambda z: sp.log(z**2) / 2,
}


def _with_matching_logarithms(value: sp.Basic, variable: sp.Basic) -> sp.Basic:
    """An antiderivative whose logarithms are rewritten over the arguments of the 6.11
    integrals whose singularities they cancel.

    `INT(SIN(a*x)^2/x, x)` is `LN(x)/2 - CI(2*a*x)/2`, but for negative `a` the two
    take their imaginary parts on opposite sides of zero, so the answer jumps by
    `#i*pi/2` where the integrand is regular. When logarithm and 6.11 terms in
    multiples of one linear argument have coefficients summing to zero, the logarithms
    are replaced by `-d_k*LN(u_k)`, which differs by a constant on each side and is
    continuous.
    """
    if not isinstance(value, sp.Basic) or not value.has(*_WITH_A_LOGARITHM):
        return value
    integrals = [
        integral
        for integral in sorted(value.atoms(*_WITH_A_LOGARITHM), key=sp.default_sort_key)
        if integral.args[0].is_polynomial(variable) is True
        and sp.degree(integral.args[0], variable) == 1
    ]
    logarithms = sorted(value.atoms(sp.log), key=sp.default_sort_key)
    while integrals:
        reference = integrals[0].args[0]

        def proportional(call: sp.Basic) -> bool:
            ratio = sp.cancel(call.args[0] / reference)
            return not ratio.has(variable) and ratio.is_zero is not True

        matching = [integral for integral in integrals if proportional(integral)]
        integrals = [integral for integral in integrals if integral not in matching]
        logs = [logarithm for logarithm in logarithms if proportional(logarithm)]
        calls = [*logs, *matching]
        stand_ins = [sp.Dummy() for _ in calls]
        written = value.xreplace(dict(zip(calls, stand_ins)))
        if written.has(*calls):
            continue
        coefficients = [sp.diff(written, stand_in) for stand_in in stand_ins]
        if any(c.has(variable, *stand_ins) for c in coefficients):
            continue
        if sp.cancel(sum(coefficients)) != 0:
            continue
        matched = [
            -d * _WITH_A_LOGARITHM[type(integral)](integral.args[0])
            for d, integral in zip(coefficients[len(logs) :], matching)
        ]
        value = value.xreplace({logarithm: sp.S.Zero for logarithm in logs})
        value = sp.Add(value, *matched)
    return value


def _with_a_factored_rational_part(
    value: sp.Basic, integrand: sp.Basic, variable: sp.Basic
) -> sp.Basic:
    """An antiderivative of a rational function with its rational part written over the
    factors of its denominator, shifted to vanish at zero when that is shorter.

    `INT(1/(1 + x)^3, x)` should be `-1/(2*(x + 1)^2)`, not `-1/(2*x^2 + 4*x + 2)`. The
    denominator is written over `_denominator_factors`, unless the factored form would
    read back multiplied out anyway (`_multiplied_when_read`) or would make the whole
    answer merge over one denominator (`_merged_only_when_factored`).

    `R - R(0)` is used if its numerator is shorter (`INT(x^4/(1 + x)^6, x)` is
    `x^5/(5*(x + 1)^5)`), but only if `q(0)` is nonzero for every parameter value;
    otherwise `R(0)` adds a pole, as at `a = 0` in `INT(x^4/(a + b*x)^6, x)`.
    """
    if integrand.is_rational_function(variable) is not True:
        return value
    coefficient, rest = _split(value, variable)
    rational, others = [], []
    for term in sp.Add.make_args(rest):
        if term.is_rational_function(variable) and not term.is_polynomial(variable):
            rational.append(term)
        else:
            others.append(term)
    if not rational:
        return value
    q = sp.Poly(sp.denom(sp.cancel(integrand)), variable)
    part = sp.cancel(coefficient * sp.Add(*rational))
    shifted = None
    if q.eval(0).is_zero is False:
        shifted = _shorter_at_zero(part, variable)
        if shifted is not None:
            part = shifted
    written = _over_the_factors(part, _denominator_factors(q))
    if written is None or _multiplied_when_read(written):
        if shifted is None:
            return value
        written = part
    beside = coefficient * sp.Add(*others)
    if others and _merged_only_when_factored(beside + written, beside + part):
        return value if shifted is None else beside + part
    return beside + written


def _shorter_at_zero(part: sp.Basic, variable: sp.Basic) -> sp.Basic | None:
    """`R - R(0)` for the cancelled rational part `R`, if `R(0)` is nonzero and the
    difference has a numerator with fewer terms than `R`'s, otherwise None."""
    numerator, denominator = sp.fraction(part)
    below = sp.cancel(denominator.subs(variable, 0))
    if below == 0:
        return None
    constant = sp.cancel(numerator.subs(variable, 0) / below)
    if constant == 0 or constant.has(sp.zoo, sp.oo, sp.nan):
        return None
    shifted = sp.cancel(part - constant)
    terms = len(sp.Add.make_args(sp.expand(sp.fraction(shifted)[0])))
    if terms >= len(sp.Add.make_args(sp.expand(numerator))):
        return None
    return shifted


def _denominator_factors(q: sp.Poly) -> list[sp.Poly]:
    """The factors to write the rational part's denominator over, from the integrand's
    cancelled denominator `q`.

    Each square-free factor of `q` is split into a power of the variable and the rest,
    which is factored over the rationals unless it is a binomial: `x^4 + 3*x^2 + 2`
    gives `x^2 + 1` and `x^2 + 2`, but `x^3 + 1` stays whole.
    """
    variable = q.gens[0]
    factors: list[sp.Poly] = []
    _, parts = q.sqf_list()
    for part, _ in parts:
        (lowest,), _ = part.terms()[-1]
        if lowest:
            factors.append(sp.Poly(variable, variable))
            part = part.exquo(sp.Poly(variable**lowest, variable))
        if part.degree() <= 0:
            continue
        if len(part.terms()) == 2:
            factors.append(part)
            continue
        try:
            _, irreducible = part.factor_list()
        except Exception:
            irreducible = [(part, 1)]
        factors.extend(factor for factor, _ in irreducible if factor.degree() > 0)
    return factors


def _over_the_factors(part: sp.Basic, bases: list[sp.Poly]) -> sp.Basic | None:
    """The cancelled rational part over a constant times powers of `bases`, or None if its
    denominator isn't such a product. The constant is written square-free in the
    parameters.
    """
    if not bases:
        return None
    numerator, denominator = sp.fraction(part)
    remaining = sp.Poly(denominator, bases[0].gens[0], field=True)
    powers = []
    for base in bases:
        exponent = 0
        while remaining.degree() >= base.degree():
            quotient, remainder = remaining.div(base)
            if not remainder.is_zero:
                break
            remaining, exponent = quotient, exponent + 1
        if exponent:
            powers.append(base.as_expr() ** exponent)
    if remaining.degree() != 0 or not powers:
        return None
    content = sp.factor_terms(sp.sqf(remaining.as_expr()))
    under = [factor**-1 for factor in sp.Mul.make_args(content)]
    return sp.Mul(numerator, *under, *(power**-1 for power in powers))


def _merged_only_when_factored(factored: sp.Basic, multiplied: sp.Basic) -> bool:
    """Whether the simplifier would merge the whole answer over one denominator with its
    rational part factored, but not with it multiplied out.

    A factored denominator gives `radsimp` a common factor to merge everything over:
    for `INT(1/(x^6 + 16)^2, x)` the result is twice as long as the sum.
    """
    return pays(sp.radsimp(factored), factored) and not pays(
        sp.radsimp(multiplied), multiplied
    )


def _multiplied_when_read(written: sp.Basic) -> bool:
    """Whether the quotient is a sum over a number times a sum, which sympy reads back
    with the denominator multiplied out: `(3 - 4*x)/(2*(x^2 + 1))` becomes `(3 -
    4*x)/(2*x^2 + 2)`.
    """
    numerator, denominator = sp.fraction(written)
    return numerator.is_Add and denominator.is_Add


def somewhere_only(value: sp.Basic) -> bool:
    """Whether the answer is a split that leaves some values of the variable without a
    case.

    Sympy's trigonometric substitution answers `INT(SQRT(x^2 - 1)/x, x)` only for `-1 <
    x < 1`, exactly where the integrand isn't real. Such an answer is rejected.
    """
    return any(
        split.args[-1].cond is not sp.true for split in value.atoms(sp.Piecewise)
    )


def anywhere(
    value: sp.Basic, integrand: sp.Basic, variable: sp.Basic
) -> sp.Basic | None:
    """The single case of an answer refused by `somewhere_only`, if it differentiates back
    unconditionally, otherwise None. For `INT(1/(x^2*SQRT(x^2 - 1)), x)` the condition
    is the only thing wrong.
    """
    folded = attempt(value, sp.piecewise_fold)
    if not isinstance(folded, sp.Piecewise) or len(folded.args) != 1:
        return None
    case = folded.args[0].expr
    return case if holds(case, integrand, variable) else None


def _shorter_by_the_manual_rules(
    head: sp.Basic, value: sp.Basic, integrand: sp.Basic, variable: sp.Basic
) -> sp.Basic:
    """The manual rules' answer, if it is shorter than sympy's and holds.

    With a parameter of unknown sign, sympy's rational integrator can't tell real roots
    from complex ones and leaves `INT(1/(x^2 + a), x)` as a pair of logarithms. The
    manual rules give `ATAN(x/SQRT(a))/SQRT(a)`, which Derive prints for every `a`.

    Skipped for a quadratic with parametric coefficients, which `engine.rational`
    already writes the table's way, and where the manual rules can spend a minute
    finding nothing.
    """
    manual = _by_the_manual_rules(head, integrand, variable)
    if manual is None:
        return value
    manual = single_case(manual, integrand, variable)
    if manual.has(sp.Piecewise) or not pays(manual, value):
        return value
    return manual if holds(manual, integrand, variable) else value


def _plainly_rational(value: sp.Basic, integrand: sp.Basic, variable: sp.Basic) -> bool:
    """Whether the integrand is rational over a parametric quadratic with no linear term,
    and the answer's logarithms are all of plain polynomials.

    The manual rules only improve on logarithms of roots the rational integrator
    couldn't tell were complex (`x^2 + a`). For a real quadratic like `x^2 + a^2` they
    find nothing shorter and can spend ten seconds doing it.
    """
    if not over_a_parametric_quadratic(integrand, variable, binomials=True):
        return False
    for call in value.atoms(sp.log):
        argument = call.args[0]
        if argument.has(sp.I) or argument.is_polynomial(variable) is not True:
            return False
        for coefficient in sp.Poly(argument, variable).all_coeffs():
            if not coefficient.is_polynomial() or any(
                not power.exp.is_Integer or power.exp < 0
                for power in coefficient.atoms(sp.Pow)
            ):
                return False
    return True


def continuous(value: sp.Basic, variable: sp.Basic | None) -> sp.Basic:
    """An antiderivative made continuous in its parameters, where subtracting a constant
    achieves that.

    Sympy answers `INT(x^n, x)` with a split at `n = -1`, but the pole is only in the
    constant of integration: `(x^(n + 1) - 1)/(n + 1)` works for every `n`, with
    `LN(x)` as its limit, and is what Derive prints. So for each pole a parameter
    introduces, its principal part is subtracted if it is free of the variable.
    Otherwise the integrand itself is singular and the split stays, as it does for a
    branch point.

    Derive does this only for powers and exponentials (it gives `INT(SIN(a*x), x)` as
    `-COS(a*x)/a`); here it is done for every parameter. With no variable (a definite
    integral) nothing is subtracted, but a split is still dropped where the generic
    case has no pole. If no pole is found as written, common factors are pulled out
    first, since a pole can hide in denominators like `a^2*TAN(a*x/2) + a^2`.
    """
    folded = attempt(value, sp.piecewise_fold)
    if not isinstance(folded, sp.Piecewise):
        if variable is None:
            return value
        regular, _, _ = _regularized(value, variable)
        return regular
    if len(folded.args) != 2 or folded.args[1][1] is not sp.true:
        return value
    generic, condition = folded.args[0]
    if not isinstance(condition, sp.Ne):
        return value
    regular, found, kept = _regularized(generic, variable)
    if not found:
        exposed = attempt(generic, sp.factor_terms)
        if exposed is not None:
            regular, found, kept = _regularized(exposed, variable)
            regular = generic - (exposed - regular)
    return value if kept or not found else regular


def _regularized(
    expression: sp.Basic, variable: sp.Basic | None
) -> tuple[sp.Basic, bool, bool]:
    """The expression minus the constant principal parts of its parameter poles.

    Returns the result, whether any pole was examined, and whether any was kept. The
    expansion is in the divisor itself, so `INT(a^x, x)` subtracts Derive's `1/LN(a)`
    rather than `1/(a - 1)`.
    """
    found = kept = False
    for divisor in parametric_divisors(expression, variable):
        principal = _principal_part(expression, divisor)
        if principal is None:
            continue
        found = True
        if principal.has(sp.I):
            flipped = _with_logarithms_flipped(expression, divisor)
            again = None if flipped is None else _principal_part(flipped, divisor)
            if again is not None and not again.has(sp.I):
                expression, principal = flipped, again
        if principal == 0:
            continue
        if variable is None or principal.has(variable):
            kept = True
            continue
        expression = expression - principal
    return expression, found, kept


def _with_logarithms_flipped(expression: sp.Basic, divisor: sp.Basic) -> sp.Basic | None:
    """The expression with each logarithm whose argument is negative where the divisor
    vanishes, or goes to minus infinity there, written over the negated argument, or
    None if there is none.

    The two differ by a constant multiple of `pi*#i`, so the antiderivative stays one.
    Sympy's `INT(1/COS(a*x), x)` has `LN(SIN(a*x) - 1)`, which at `a = 0` is `LN(-1)`:
    made continuous there as written, it carries a `-#i*pi/(2*a)` term, while over `LN(1
    - SIN(a*x))` it needs none.
    """
    if len(divisor.free_symbols) != 1:
        return None
    (parameter,) = divisor.free_symbols
    try:
        roots = sp.solve(divisor, parameter)
    except Exception:
        return None
    if len(roots) != 1:
        return None
    flips = {}
    for call in expression.atoms(sp.log):
        there = attempt(call.args[0], lambda a: sp.cancel(a.subs(parameter, roots[0])))
        if there is not None and there.has(sp.zoo, sp.nan):
            there = attempt(
                call.args[0], lambda a: sp.limit(a, parameter, roots[0], "+-")
            )
        if there is not None and there.is_number and there.is_extended_negative:
            flips[call] = sp.log(sp.expand(-call.args[0]))
    return expression.xreplace(flips) if flips else None


def parametric_divisors(expression: sp.Basic, variable: sp.Basic | None) -> list:
    """The divisors in the expression that contain symbols but not the variable; each
    is where a parameter puts a pole into the answer."""
    divisors = {
        power.base
        for power in expression.atoms(sp.Pow)
        if power.exp.is_negative
        and power.base.free_symbols
        and not (variable is not None and power.base.has(variable))
    }
    return sorted(divisors, key=sp.default_sort_key)


def _principal_part(expression: sp.Basic, divisor: sp.Basic) -> sp.Basic | None:
    """The diverging terms of the expression's Laurent expansion in the divisor, or None
    if there is none.

    A divisor in several parameters, or one that can't be solved for its parameter, is
    skipped; a power such as `(n + 1)^2` is expanded in its base.
    """
    if len(divisor.free_symbols) != 1:
        return None
    _, factored = (attempt(divisor, sp.factor) or divisor).as_coeff_Mul()
    if isinstance(factored, sp.Pow) and factored.exp.is_Integer and factored.exp > 1:
        divisor = factored.base
    (parameter,) = divisor.free_symbols
    coordinate = sp.Dummy("u")
    try:
        inverted = sp.solve(sp.Eq(divisor, coordinate), parameter)
    except Exception:
        return None
    if len(inverted) != 1:
        return None
    shifted = expression.subs(parameter, inverted[0])
    if _logarithmic_branch_point(shifted, coordinate):
        return None
    expansion = attempt(
        shifted, lambda s: sp.expand(sp.series(s, coordinate, 0, 1).removeO())
    )
    if expansion is None:
        return None
    principal = sp.S.Zero
    for term in sp.Add.make_args(expansion):
        coefficient, exponent = term.as_coeff_exponent(coordinate)
        if not exponent.is_integer or coefficient.has(coordinate):
            return None
        if exponent < 0:
            principal += term
    return principal.subs(coordinate, divisor)


#: Functions with a logarithmic branch point where their argument is plus or minus the
#: given value.
_BRANCH_POINTS = {sp.log: 0, sp.atan: sp.I, sp.acot: sp.I, sp.atanh: 1, sp.acoth: 1}


def _logarithmic_branch_point(expression: sp.Basic, coordinate: sp.Basic) -> bool:
    """Whether the expression has a logarithm or inverse tangent at a branch point when
    the coordinate is zero, so there is no Laurent expansion.

    Checked directly because sympy's `series` takes seconds to a minute to find out,
    for example for `INT(x^2/(a*x^2 + b*x + c), x)` at `a = 0`. An arctangent whose
    argument goes to complex infinity also counts.
    """
    for call in expression.atoms(*_BRANCH_POINTS):
        if not call.has(coordinate):
            continue
        at_zero = attempt(call.args[0], lambda a: a.subs(coordinate, 0))
        if at_zero is None:
            continue
        if at_zero.has(sp.zoo, sp.oo, sp.nan):
            if call.func in (sp.atan, sp.acot) and at_zero.has(sp.zoo):
                return True
            continue
        point = _BRANCH_POINTS[call.func]
        if attempt(at_zero, lambda a: sp.cancel(sp.expand(a**2 - point**2))) == 0:
            return True
    return False


def unwritable(value: sp.Basic, variable: sp.Basic) -> bool:
    """Whether an antiderivative contains a function of the variable the notation can't
    express.

    `is_written` would reject it later anyway, so refusing it now saves
    `differentiates_back` seconds of simplifying a hypergeometric series. Constants and
    case splits don't count, since `_without_constants` and `single_case` may remove
    them.
    """
    if value.has(sp.Piecewise):
        return False
    spelled = forced(value, in_vocabulary)
    try:
        calls = spelled.atoms(sp.Function)
    except Exception:
        return False
    return any(
        variable in call.free_symbols and not is_written(call) for call in calls
    )


def on_the_principal_sheet(
    value: sp.Basic, integrand: sp.Basic, variable: sp.Basic
) -> sp.Basic:
    """An antiderivative with full turns about the origin removed, if it still
    differentiates back.

    Sympy's Meijer G route answers `INT(LN(1 - x)/x, x)` with `-POLYLOG(2,
    x*EXP_POLAR(2*pi*#i))`. A full turn is the number 1, and `-POLYLOG(2, x)` is also
    an antiderivative. Other angles actually move the argument and are kept.
    """
    if not value.has(sp.exp_polar):
        return value
    stripped = attempt(
        value,
        lambda v: sp.unpolarify(v).replace(
            lambda e: isinstance(e, sp.exp_polar) and _full_turns(e.exp),
            lambda e: sp.S.One,
        ),
    )
    if stripped is None or stripped.has(*POLAR):
        return value
    difference = attempt(
        stripped, lambda v: sp.expand_func(sp.diff(v, variable) - integrand)
    )
    return stripped if difference is not None and proved_zero(difference) else value


def _full_turns(angle: sp.Basic) -> bool:
    """Whether a polar angle is a whole number of turns, i.e. the factor is 1."""
    turns = attempt(angle, lambda a: a / (2 * sp.pi * sp.I))
    return bool(turns is not None and turns.is_integer)


def _by_the_manual_rules(
    head: sp.Basic, integrand: sp.Basic, variable: sp.Basic
) -> sp.Basic | None:
    """The integral computed by sympy's manual (textbook) rules, if the result is
    expressible and proved to differentiate back, otherwise None."""
    manual = attempt(head, lambda h: h.doit(deep=False, manual=True))
    if manual is None or manual.has(sp.Integral):
        return None
    if unwritable(manual, variable):
        return None
    return manual if differentiates_back(manual, integrand, variable) else None


def single_case(value: sp.Basic, integrand: sp.Basic, variable: sp.Basic) -> sp.Basic:
    """A split over antiderivatives narrowed to the one case worth printing.

    Sympy splits where its table has several forms that differ only by a constant. The
    unconditional last case decides: if it differentiates back, it is valid everywhere.
    A shorter conditional case that also differentiates back unconditionally can
    replace it: `INT(1/(x^2 + a), x)` gives the `ATAN` Derive prints for every `a`.

    If the unconditional case fails, the split stays; whether its pole belongs to the
    integral or the constant is for `continuous` to decide.
    """
    folded = attempt(value, sp.piecewise_fold)
    if not isinstance(folded, sp.Piecewise):
        return value
    best, condition = folded.args[-1]
    if condition is not sp.true or not holds(best, integrand, variable):
        return value
    for case, _ in folded.args[:-1]:
        if case.has(*INVERSE_HYPERBOLIC) and not best.has(*INVERSE_HYPERBOLIC):
            continue
        if pays(case, best) and holds(case, integrand, variable):
            best = case
    return best


def _without_constants(value: sp.Basic, variable: sp.Basic) -> sp.Basic:
    """The antiderivative with all constants of integration removed.

    These are terms free of the variable, and constant factors inside a top-level
    logarithm, `LN(k*u)` being `LN(u)` plus a constant. Deeper constants are part of
    their term: in `x*LN(a*x) - x`, the `LN(a)` contributes `x*LN(a)`.
    """
    coefficient, terms = _split(value, variable)
    bare = [_bare_term(term, variable) for term in sp.Add.make_args(terms)]
    if not isinstance(terms, sp.Add):
        return coefficient * bare[0]
    kept = [term for term in bare if term.has(variable)]
    return _carried(coefficient, _without_flat_logarithms(kept, variable))


def _bare_term(term: sp.Basic, variable: sp.Basic) -> sp.Basic:
    """A term with the constant factor removed from its logarithm, if it is a logarithm
    times a constant.

    A logarithm of something never positive is rewritten over its negation, a constant
    `pi*#i` away: sympy's `LN(SIN(x) - 1)` for `INT(SEC(x), x)` becomes `LN(1 -
    SIN(x))`.
    """
    factor, rest = _split(term, variable)
    if not isinstance(rest, sp.log) or len(rest.args) != 1:
        return term
    bare = attempt(rest, lambda call: _bare_logarithm(call, variable))
    if bare is None:
        return term
    if isinstance(bare, sp.log) and _never_positive(bare.args[0], variable):
        bare = sp.log(-bare.args[0])
    return factor * bare


def _never_positive(argument: sp.Basic, variable: sp.Basic) -> bool:
    """Whether a function of a real variable alone is nonpositive everywhere, judged by
    its bounds over the whole real line."""
    if variable.is_real is not True or argument.free_symbols != {variable}:
        return False
    bounds = attempt(
        argument, lambda a: a.subs(variable, sp.AccumBounds(-sp.oo, sp.oo))
    )
    return bool(
        isinstance(bounds, sp.AccumBounds) and bounds.max.is_nonpositive is True
    )


def _without_flat_logarithms(terms: list[sp.Basic], variable: sp.Basic) -> list:
    """The terms without their logarithms, if those sum to a constant, as `LN(x^2)/2 -
    LN(x)` does in sympy's `INT(SIN(x)/x^2, x)`.
    """
    logarithms = [term for term in terms if _is_logarithm_term(term, variable)]
    if len(logarithms) < 2:
        return terms
    derivative = attempt(
        sp.Add(*logarithms), lambda s: sp.simplify(sp.diff(s, variable))
    )
    if derivative != 0:
        return terms
    return [term for term in terms if not _is_logarithm_term(term, variable)]


def _is_logarithm_term(term: sp.Basic, variable: sp.Basic) -> bool:
    """Whether the term is a logarithm times a factor free of the variable."""
    _, rest = _split(term, variable)
    return isinstance(rest, sp.log)


def _carried(coefficient: sp.Basic, terms: list[sp.Basic]) -> sp.Basic:
    """A fractional coefficient distributed into the terms where its denominator cancels
    and kept in front of the rest.

    Derive writes `b*(SQRT(b^2 - 4*a*c) - b)/(4*a) - c*LN(SQRT(b^2 - 4*a*c) + b)` where
    sympy puts `1/(2*a)` in front of everything, which the normal form would merge into
    one quotient. A purely numeric coefficient stays in front of the whole sum.
    """
    if not sp.fraction(coefficient)[1].free_symbols:
        return coefficient * sp.Add(*terms)
    carried, absorbed = [], []
    for term in terms:
        product = coefficient * term
        (absorbed if not sp.fraction(product)[1].free_symbols else carried).append(term)
    return coefficient * sp.Add(*carried) + sp.Add(*(coefficient * t for t in absorbed))


def _bare_logarithm(call: sp.Basic, variable: sp.Basic) -> sp.Basic:
    """`LN(k*u)` as `LN(u)`, where `k` is free of the variable. The factor must be
    extracted, since sympy writes the argument expanded.
    """
    argument = attempt(call.args[0], sp.factor_terms)
    if argument is None:
        return call
    coefficient, rest = _split(argument, variable)
    if coefficient == 1 or not rest.has(variable):
        return call
    return sp.log(rest)


def _split(expression: sp.Basic, variable: sp.Basic) -> tuple[sp.Basic, sp.Basic]:
    """`expression` split into a factor free of the variable times the rest.

    Only products go to `as_independent`, which treats the terms of a sum as factors.
    """
    if not isinstance(expression, sp.Mul):
        return sp.S.One, expression
    return expression.as_independent(variable, as_Mul=True)


def parametrised(head: sp.Basic) -> bool:
    """Whether the integrand contains symbols other than the integration variables. Limits
    don't count, since they are substituted after the costly search.
    """
    variables = {limit[0] for limit in head.limits}
    return bool(head.function.free_symbols - variables)


