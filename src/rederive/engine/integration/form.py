"""What an antiderivative is printed as.

An indefinite integral is one antiderivative out of many, and which one is printed is
a choice rather than a result. `antiderivative` is where an answer `indefinite` has
taken is put through that choice: the case of a split worth printing, the constants
taken out, the logarithms matched, the form continuous wherever the integrand is.
What its gate asks of an answer before it is taken is here too, being a question of
its form: whether it is on the principal sheet, whether it says something for every
value of the variable, which case of a split holds, and whether the notation can say
it.
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
    """An answer taken for an integral, in the form Derive writes it in.

    An indefinite integral is one antiderivative out of many: any two of them
    differ by a constant, and which one is printed is a choice rather than a
    result. Derive prints the one with no constant added and never a case split
    where a single antiderivative serves everywhere. Sympy's tables make the
    other choice both times, so the choice is made here.

    Nothing is taken out of a definite integral, where a constant does not
    cancel but is part of the answer, nor out of an integral that was not done:
    neither has an antiderivative in hand to choose among. A definite integral's
    split over a parameter is still dropped where it distinguishes nothing.

    Where a parametrised answer carries a logarithm, the manual rules are asked for a
    shorter one, as `_shorter_by_the_manual_rules` explains: not where the answer is
    `tabled`, written the way the integral tables write it already or by the manual
    rules themselves.

    The rational part of an antiderivative of a rational function is written over the
    factors of its denominator, with the constant that makes it vanish at zero where
    that is shorter, as `_with_a_factored_rational_part` explains.

    Whether the answer is one to take at all is not asked here: `indefinite` has
    proved it, or knows it right by construction, before it is handed over.
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


#: The integrals of 6.11 with a logarithmic singularity at zero, and the logarithm of
#: the same argument that takes it off: `CI(z) - LN(z)` and `COSH_INT(z) - LN(z)` are
#: entire, the two sharing a branch cut, and `EI(z) - LN(z^2)/2` is continuous and
#: real along the real line, where `EI` is real on both sides of zero.
_WITH_A_LOGARITHM: dict[type, Callable[[sp.Basic], sp.Basic]] = {
    sp.Ci: sp.log,
    sp.Chi: sp.log,
    sp.Ei: lambda z: sp.log(z**2) / 2,
}


def _with_matching_logarithms(value: sp.Basic, variable: sp.Basic) -> sp.Basic:
    """An antiderivative with the logarithms that cancel the singularities of integrals
    of 6.11 written over those integrals' arguments.

    `INT(SIN(a*x)^2/x, x)` is `LN(x)/2 - CI(2*a*x)/2`, whose derivative is the
    integrand on each side of zero; but for a negative `a` the logarithm and the
    cosine integral take their imaginary parts on opposite sides, and the answer jumps
    by `#i*pi/2` at zero, where the integrand is regular. So does `CI(a*x) - CI(b*x)`
    where `a` and `b` differ in sign. Where terms `c_j*LN(L_j)` and `d_k*CI(u_k)`, every
    `L_j` and `u_k` a multiple of one polynomial of degree 1 in the variable, have
    coefficients that add up to zero, the logarithms are written as the sum of
    `-d_k*LN(u_k)`. That differs from them by a constant on each side of zero and
    makes the whole continuous through it. Logarithms that do not cancel the
    singularities are left as they are: the integrand is singular there itself.
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
    """An antiderivative of a rational function with its rational part written over
    the factors of its denominator, and with the constant that makes that part vanish
    at zero where that writes it shorter.

    The rational part is the sum of the terms that are rational functions of the
    variable and not polynomials in it: what Hermite's reduction splits off, over
    `gcd(q, q')` for the integrand's denominator `q`. That denominator is a product of
    powers of the square-free factors of `q`, and multiplied out it hides them:
    `INT(1/(1 + x)^3, x)` is `-1/(2*(x + 1)^2)`, which says where its pole is, and not
    `-1/(2*x^2 + 4*x + 2)`. So the denominator is written as a factor free of the
    variable times a power of each of the factors `_denominator_factors` finds, by
    dividing them out: nothing is factored that the square-free decomposition or the
    integrand as written did not already split, and `INT(x^2/(x^3 + 1)^2, x)` is over
    `x^3 + 1`. Where the normal form would take a number in that denominator into the
    numerator, as `_spread_by_the_normal_form` says, it is left multiplied out, and so
    it is where the factored part would have the whole answer put over one
    denominator, as `_merged_only_when_factored` says.

    The rational part `R` and `R - R(0)` are antiderivatives alike, and the second is
    taken where its numerator has fewer terms: `INT(x^4/(1 + x)^6, x)` is
    `x^5/(5*(x + 1)^5)`, where `R` is a quartic over the same denominator. Only where
    `q(0)` is known not to be zero for any value of the parameters: where it may be,
    `R(0)` has a pole there that `R` does not, as at `a = 0` in `INT(x^4/(a + b*x)^6,
    x)`, and `R - R(0)` would be undefined for a value of the parameters the integral
    is defined for. The rest of the answer, its polynomial and its logarithms and
    arctangents, is left as it is.
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
    written = _over_the_factors(part, _denominator_factors(integrand, q, variable))
    if written is None or _spread_by_the_normal_form(written):
        if shifted is None:
            return value
        written = part
    beside = coefficient * sp.Add(*others)
    if others and _merged_only_when_factored(beside + written, beside + part):
        return value if shifted is None else beside + part
    return beside + written


def _shorter_at_zero(part: sp.Basic, variable: sp.Basic) -> sp.Basic | None:
    """`R - R(0)` for the cancelled rational part `R`, where `R(0)` is other than zero
    and the numerator of the difference has fewer terms than that of `R`, or None."""
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


def _denominator_factors(
    integrand: sp.Basic, q: sp.Poly, variable: sp.Basic
) -> list[sp.Poly]:
    """The factors the rational part's denominator is written over: the square-free
    factors of the integrand's cancelled denominator `q`, split where the denominator
    was written with factors of its own, by gcds and never by a factorization.

    `1/((x + 1)^2*(x + 2)^2)` has the one square-free factor `x^2 + 3*x + 2`, and the
    rational part of its integral is over `(x + 1)*(x + 2)`, the factors it was written
    with, as the normal form keeps the factors a denominator arrives with. Each is
    taken square-free, and the whole is made pairwise coprime: a factor that shares a
    divisor with another is replaced by the divisor and what is left of the two.
    """
    written = []
    for factor in sp.Mul.make_args(sp.denom(integrand)):
        base, _ = factor.as_base_exp()
        if base.has(variable) and base.is_polynomial(variable):
            _, parts = sp.Poly(base, variable).sqf_list()
            written.extend(part for part, _ in parts)
    _, parts = q.sqf_list()
    basis: list[sp.Poly] = []
    for candidate in [*written, *(part for part, _ in parts)]:
        pending = [candidate]
        while pending:
            factor = pending.pop()
            if factor.degree() <= 0:
                continue
            for index, known in enumerate(basis):
                common = sp.gcd(factor, known)
                if common.degree() <= 0:
                    continue
                if common.degree() == known.degree():
                    common = known
                else:
                    del basis[index]
                    pending.extend([common, sp.quo(known, common)])
                pending.append(sp.quo(factor, common))
                break
            else:
                basis.append(factor)
    return basis


def _over_the_factors(part: sp.Basic, bases: list[sp.Poly]) -> sp.Basic | None:
    """The cancelled rational part over a factor free of the variable times a power of
    each of `bases`, or None where its denominator is no such product.

    The factor free of the variable is written square-free in the parameters,
    `2*(4*a*c - b^2)^2` and not a polynomial of degree four in them.
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
    """Whether the simplifier would put the whole answer over one denominator with its
    rational part factored, and not with it multiplied out.

    Simplify keeps `radsimp`'s answer over one denominator wherever that counts fewer
    operations, and a factored denominator is a common factor `radsimp` can merge
    every other term over: `INT(1/(x^6 + 16)^2, x)`, its logarithms and arctangents
    carrying `2^(2/3)`, would come back as one quotient twice the length of the sum.
    Such an answer keeps its rational part multiplied out.
    """
    return pays(sp.radsimp(factored), factored) and not pays(
        sp.radsimp(multiplied), multiplied
    )


def _spread_by_the_normal_form(written: sp.Basic) -> bool:
    """Whether the normal form would write the quotient with fractions in its
    numerator.

    The normal form takes a number under the line into the one sum over it wherever
    that leaves some term of the sum a whole coefficient, `(4*x + 3)/(2*(x + 1)^2)` as
    `(2*x + 3/2)/(x + 1)^2`. A rational part it would write that way is kept over its
    denominator multiplied out, where the number stays inside a sum.
    """
    coefficient, rest = written.as_coeff_Mul()
    if not coefficient.is_Rational or coefficient.q == 1:
        return False
    sums = [factor for factor in sp.Mul.make_args(rest) if factor.is_Add]
    if len(sums) != 1:
        return False
    return any(
        (coefficient * term.as_coeff_Mul()[0]).is_Integer
        for term in sp.Add.make_args(sp.expand(sums[0]))
    )


def somewhere_only(value: sp.Basic) -> bool:
    """Whether the answer is a split that leaves some values of the variable without
    a case, and so says nothing there.

    Sympy's rule for a trigonometric substitution writes the condition it was made
    under into its answer: `INT(SQRT(x^2 - 1)/x, x)` comes back as `SQRT(x^2 - 1) -
    ACOS(1/x)` where `-1 < x < 1`, and no case otherwise. That is the interval where
    the integrand is not real, and the answer is undefined on the whole of the real
    line where the integral is something. An antiderivative has to be one wherever
    the integrand is defined, so an answer like this is no answer.
    """
    return any(
        split.args[-1].cond is not sp.true for split in value.atoms(sp.Piecewise)
    )


def anywhere(
    value: sp.Basic, integrand: sp.Basic, variable: sp.Basic
) -> sp.Basic | None:
    """The one case of an answer `somewhere_only` refuses, where it differentiates
    back to the integrand for every value of the variable, or None.

    `INT(1/(x^2*SQRT(x^2 - 1)), x)` comes back as `SQRT(x^2 - 1)/x` under the same
    condition, and the condition is all that is wrong with it.
    """
    folded = attempt(value, sp.piecewise_fold)
    if not isinstance(folded, sp.Piecewise) or len(folded.args) != 1:
        return None
    case = folded.args[0].expr
    return case if holds(case, integrand, variable) else None


def _shorter_by_the_manual_rules(
    head: sp.Basic, value: sp.Basic, integrand: sp.Basic, variable: sp.Basic
) -> sp.Basic:
    """The answer the manual rules give, where it pays over sympy's and holds.

    Over a parameter whose sign it does not know, sympy's rational integrator can
    tell the real roots of a denominator from the complex ones no better than the
    reader can, and writes the logarithms it would otherwise have paired into
    arctangents: `INT(1/(x^2 + a), x)` is `SQRT(-1/a)*LN(x + a*SQRT(-1/a))/2 -
    SQRT(-1/a)*LN(x - a*SQRT(-1/a))/2` to it. That is an antiderivative, and the
    manual rules have a shorter one - `ATAN(x/SQRT(a))/SQRT(a)`, under the condition
    that `a` is positive, which `single_case` finds says nothing - and the shorter
    one is what Derive prints for every `a`. So where a parametrised answer carries a
    logarithm, the manual rules are asked as well, and theirs is printed where it is
    shorter and differentiates back unconditionally.

    Not over a quadratic with parametric coefficients, which `engine.rational`
    already writes the table's way, as one logarithm and one arctangent of it: the
    manual rules spend most of a minute on `INT(1/(x*(a*x^2 + b*x + c)^3), x)` to find
    nothing shorter.
    """
    manual = _by_the_manual_rules(head, integrand, variable)
    if manual is None:
        return value
    manual = single_case(manual, integrand, variable)
    if manual.has(sp.Piecewise) or not pays(manual, value):
        return value
    return manual if holds(manual, integrand, variable) else value


def _plainly_rational(value: sp.Basic, integrand: sp.Basic, variable: sp.Basic) -> bool:
    """Whether the integrand is a rational function of the variable over a quadratic
    with parametric coefficients and no linear term, and the answer's logarithms are
    all of polynomials in it with polynomial coefficients, free of roots, `#i` and
    fractions.

    The manual rules improve on the logarithms of a pair of roots the rational
    integrator could not tell were complex, as those of `x^2 + a`; where the quadratic
    came out real, as `x^2 + a^2` does, they find nothing shorter, and spend ten
    seconds on `INT(1/(x*(a^2 + x^2)^3), x)`, whose answer is in `LN(x)` and `LN(x^2 +
    a^2)`, to find it.
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
    """An antiderivative made continuous in its parameters, where a constant does it.

    `INT(x^n, x)` is `x^(n + 1)/(n + 1)` for every `n` but `-1`, and `LN(x)` there,
    and sympy answers with that case split. But the integrand is analytic in `n` at
    `-1`, so the integral from any fixed point is too, and the pole in the formula
    is nowhere but in its constant of integration: `(x^(n + 1) - 1)/(n + 1)` is an
    antiderivative for every `n`, with `LN(x)` as its limit. That is the one Derive
    prints, and the split was never about the integral.

    So every pole a parameter puts into the answer is looked at. Its principal part
    - the terms of the Laurent expansion that blow up - is a constant of
    integration wherever it is free of the variable, and is taken out. Where it
    depends on the variable the integrand itself is singular there, the split is
    doing real work, and it stays; so does one whose expansion is no Laurent series
    at all, a fractional power or a logarithm of the parameter being a branch point
    rather than a pole.

    Derive does this where its tables happen to, which is for powers and
    exponentials: `INT(SIN(a*x), x)` is `-COS(a*x)/a` to it, undefined at `a = 0`
    for no reason the integral knows of. Here it is done for every parameter, and
    the answers Derive subtracts from come out identical.

    With no variable to be free of, as in a definite integral, nothing may be
    subtracted; a split still goes where the generic case has no pole to begin
    with, `INT(x^n, x, 1, 2)` being `(2^(n + 1) - 1)/(n + 1)` with `LN(2)` as its
    limit.

    A parameter's pole can be spread over denominators that also carry the
    variable - `a^2*TAN(a*x/2) + a^2` - where it is no divisor of its own. Where
    none is found as written, the common factors are taken out and it is looked
    for again.
    """
    folded = attempt(value, sp.piecewise_fold)
    if not isinstance(folded, sp.Piecewise):
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
    """The expression less the constant principal parts of its parameter poles.

    Returns the result, whether any pole was examined, and whether any was left in
    place. The expansion is taken in the divisor itself rather than in the
    parameter: `INT(a^x, x)` has its pole where `LN(a)` vanishes, and expanding in
    `LN(a)` subtracts `1/LN(a)`, which is Derive's constant, where expanding in
    `a - 1` would subtract `1/(a - 1)`, an uglier constant for the same answer.
    """
    found = kept = False
    for divisor in parametric_divisors(expression, variable):
        principal = _principal_part(expression, divisor)
        if principal is None:
            continue
        found = True
        if principal == 0:
            continue
        if variable is None or principal.has(variable):
            kept = True
            continue
        expression = expression - principal
    return expression, found, kept


def parametric_divisors(expression: sp.Basic, variable: sp.Basic | None) -> list:
    """The bases divided by in the expression that are free of the variable but not
    of symbols: each is somewhere a parameter puts a pole into the answer."""
    divisors = {
        power.base
        for power in expression.atoms(sp.Pow)
        if power.exp.is_negative
        and power.base.free_symbols
        and not (variable is not None and power.base.has(variable))
    }
    return sorted(divisors, key=sp.default_sort_key)


def _principal_part(expression: sp.Basic, divisor: sp.Basic) -> sp.Basic | None:
    """The terms of the expression's Laurent expansion in the divisor that blow up
    where it vanishes, or None where there is no such expansion to speak of.

    A divisor in more than one parameter, or one that does not determine its
    parameter, is passed over. A power of one that does - `n^2 + 2*n + 1`, which is
    `(n + 1)^2` - is expanded in what it is a power of.
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


#: Functions with a logarithmic branch point where their argument is plus or minus
#: this.
_BRANCH_POINTS = {sp.log: 0, sp.atan: sp.I, sp.acot: sp.I, sp.atanh: 1, sp.acoth: 1}


def _logarithmic_branch_point(expression: sp.Basic, coordinate: sp.Basic) -> bool:
    """Whether the expression holds a logarithm or an inverse tangent whose argument
    is at a branch point where the coordinate is zero, so that it has no Laurent
    expansion there.

    Sympy's `series` takes many seconds to find that out about an arctangent: in
    the coefficient `a`, `INT(x^2/(a*x^2 + b*x + c), x)` is over `ATAN((2*a*x +
    b)/SQRT(4*a*c - b^2))`, which is `ATAN(-#i*b/ABS(b))` where `a` is zero.

    An arctangent whose argument goes to complex infinity is taken as no expansion
    either: it tends to plus or minus `pi/2` by the side and the sign of what is
    divided, and sympy spends most of a minute on `ATAN((c - 2*a^2*x)/(2*a*SQRT(b +
    c*x - a^2*x^2)))` in `a` to find that out.
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
    """Whether an antiderivative holds, in the variable, a function the notation has
    no word for.

    Such an answer is declined whatever it proves to be: `is_written` turns it down
    before it reaches the worksheet, and nothing between here and there takes a
    function of the variable out of it. So it is refused, as an unproved answer is,
    before it is put to `differentiates_back`, which would run `sympy.simplify` over a
    hypergeometric series to find out - for seconds, and for nothing. A constant the
    notation cannot say is not what decides, since `_without_constants` may yet drop
    it, and a case split is left to `single_case` to narrow before anything is said
    about it.
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
    """An antiderivative with its full turns about the origin taken out, where that
    leaves one.

    Sympy's Meijer G route answers `INT(LN(1 - x)/x, x)` with
    `-POLYLOG(2, x*EXP_POLAR(2*pi*#i))`: the polylogarithm on the sheet a full turn
    away, which differs from the principal one past the cut at 1 and so is a mark
    `unpolarify` keeps. A full turn is still the number 1, though, and dropping it
    gives `-POLYLOG(2, x)`, which is an antiderivative in its own right; any
    antiderivative is a right answer to an indefinite integral. So the turns are
    dropped and the result kept where it differentiates back to the integrand, which
    is decided rather than merely not refuted. A mark of any other angle, such as the
    half turn in `EI(x*EXP_POLAR(pi*#i))`, moves the argument and is not touched.
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
    """Whether a polar angle is a whole number of turns, and so leaves 1 where it is."""
    turns = attempt(angle, lambda a: a / (2 * sp.pi * sp.I))
    return bool(turns is not None and turns.is_integer)


def _by_the_manual_rules(
    head: sp.Basic, integrand: sp.Basic, variable: sp.Basic
) -> sp.Basic | None:
    """The integral again by the rules a calculus course teaches, where it is proved
    to differentiate back and the notation can say it, or None: what is not is no
    improvement on the answer already taken."""
    manual = attempt(head, lambda h: h.doit(deep=False, manual=True))
    if manual is None or manual.has(sp.Integral):
        return None
    if unwritable(manual, variable):
        return None
    return manual if differentiates_back(manual, integrand, variable) else None


def single_case(value: sp.Basic, integrand: sp.Basic, variable: sp.Basic) -> sp.Basic:
    """A split over antiderivatives narrowed to the one worth printing.

    Sympy splits where its table has more than one form to offer: the integral
    of `(SQRT(b^2 - 4*a*c) - b)/(2*a)` comes back an `ASINH` where `a*c` is
    negative and a logarithm elsewhere, and the two are the same function up to
    a constant, so the split distinguishes nothing. What settles that is the
    unconditional case, the one sympy writes last: where it differentiates back
    to the integrand with no condition attached it is an antiderivative wherever
    it is defined, and so are the cases that agree with it.

    That case has to be the one that holds, but it need not be the one printed.
    A conditional case is worth the whole domain where it differentiates back
    just as unconditionally and says it shorter - `INT(1/(x^2 + a), x)` splits
    into an `ATAN` where `a` is positive and a pair of logarithms elsewhere, and
    the `ATAN` is the answer Derive gives for every `a`.

    Where the unconditional case does not hold the split is left alone here:
    `INT(x^2*COS(a*x^3 + b), x)` splits on whether `a` is zero, and
    `SIN(a*x^3 + b)/(3*a)` is the answer for every `a` it is defined for - which
    is every `a` but the one the other case is there for. Whether that pole is
    the integral's or only its constant's is `continuous`'s question.
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
    """The antiderivative with every constant of integration taken out of it.

    A term free of the variable is such a constant, and so is a factor free of
    the variable inside a logarithm, `LN(k*u)` being `LN(u)` plus a constant.
    Sympy leaves both: the integral above comes back over `LN(2*b + 2*SQRT(b^2 -
    4*a*c))`, where Derive's answer is over `LN(b + SQRT(b^2 - 4*a*c))`.

    A constant deeper in than that is no constant of integration - it is part of
    whatever holds it - so only the terms of the answer itself are looked at,
    through a factor common to all of them where there is one. That goes for the
    logarithms too: only one that is a term, times a factor free of the variable,
    sheds a constant. In `x*LN(a*x) - x` the `LN(a)` it holds is `x*LN(a)`, and
    that is no constant.
    """
    coefficient, terms = _split(value, variable)
    bare = [_bare_term(term, variable) for term in sp.Add.make_args(terms)]
    if not isinstance(terms, sp.Add):
        return coefficient * bare[0]
    kept = [term for term in bare if term.has(variable)]
    return _carried(coefficient, _without_flat_logarithms(kept, variable))


def _bare_term(term: sp.Basic, variable: sp.Basic) -> sp.Basic:
    """A term of an antiderivative with the constant factor taken out of its
    logarithm, where it is a logarithm times a factor free of the variable.

    A logarithm of something never positive for a real variable is written over its
    negation, which is `pi*#i` less, a constant as well: sympy answers `INT(SEC(x),
    x)` over `LN(SIN(x) - 1)`, which is complex for every real `x`, and `LN(1 -
    SIN(x))` is real wherever the integral is.
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
    """Whether a function of a real variable alone is at most zero for every value of
    it, by the bounds of its value over the whole line."""
    if variable.is_real is not True or argument.free_symbols != {variable}:
        return False
    bounds = attempt(
        argument, lambda a: a.subs(variable, sp.AccumBounds(-sp.oo, sp.oo))
    )
    return bool(
        isinstance(bounds, sp.AccumBounds) and bounds.max.is_nonpositive is True
    )


def _without_flat_logarithms(terms: list[sp.Basic], variable: sp.Basic) -> list:
    """The terms less their logarithms, where those add up to a constant.

    `LN(x^2)/2 - LN(x)` mentions the variable and is still a constant of
    integration: its derivative is zero, and the sum is zero for positive `x` and
    `-pi*#i` for negative. Sympy leaves such a pair in `INT(SIN(x)/x^2, x)`, and
    `a` times it in `INT(SIN(a*x)/x^2, x)`.
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
    """A coefficient with a denominator times the terms of an antiderivative, kept in
    front of the terms it still divides.

    Sympy integrates `(SQRT(b^2 - 4*a*c) - b)/(2*a)` over `b` with the `1/(2*a)` in
    front of everything, one of the terms being `-2*a*c*LN(SQRT(b^2 - 4*a*c) + b)`.
    Left that way, the `a` inside makes the whole a quotient over `a`, and the normal
    form writes it as one. Derive's answer is `b*(SQRT(b^2 - 4*a*c) - b)/(4*a) -
    c*LN(SQRT(b^2 - 4*a*c) + b)`: the term the coefficient cancels into is a term of
    its own, and the coefficient stays a single factor in front of the rest.

    A coefficient with nothing symbolic underneath divides nothing and is left in
    front of the whole sum, for the normal form to write as it would any product.
    """
    if not sp.fraction(coefficient)[1].free_symbols:
        return coefficient * sp.Add(*terms)
    carried, absorbed = [], []
    for term in terms:
        product = coefficient * term
        (absorbed if not sp.fraction(product)[1].free_symbols else carried).append(term)
    return coefficient * sp.Add(*carried) + sp.Add(*(coefficient * t for t in absorbed))


def _bare_logarithm(call: sp.Basic, variable: sp.Basic) -> sp.Basic:
    """`LN(k*u)` as `LN(u)`, where `k` is free of the variable.

    The factor has to be looked for: sympy writes the argument out, so what is
    there to find is `2*b + 2*SQRT(b^2 - 4*a*c)` rather than the product it
    collects to.
    """
    argument = attempt(call.args[0], sp.factor_terms)
    if argument is None:
        return call
    coefficient, rest = _split(argument, variable)
    if coefficient == 1 or not rest.has(variable):
        return call
    return sp.log(rest)


def _split(expression: sp.Basic, variable: sp.Basic) -> tuple[sp.Basic, sp.Basic]:
    """`expression` as a factor free of the variable times the rest of it.

    `as_independent` asked for a product takes the terms of a sum for factors -
    `1 - x` comes back as `1` times `-x` - so only a product is put to it.
    """
    if not isinstance(expression, sp.Mul):
        return sp.S.One, expression
    return expression.as_independent(variable, as_Mul=True)


def parametrised(head: sp.Basic) -> bool:
    """Whether the integrand mentions more than the variables integrated over.

    The limits do not count: they are substituted into an antiderivative that
    has already been found, so they are no part of the search that costs.
    """
    variables = {limit[0] for limit in head.limits}
    return bool(head.function.free_symbols - variables)


