"""The mathematics of soLve: finding the solutions of an equation.

This sits below the pipeline, like `factoring`, so the `solve` command and an authored
`SOLVE` head share it. The result is a tuple of entries, one worksheet line each: a
relation with the variable on the left, a system's solution vector, a chained range `-2
< x < 2`, or one entry per piece of a union.

Three non-solutions are distinguished. The empty tuple means "no solutions found". An
identity gives `x = @1`, with the counter from `Context.arbitrary_index` so constants
are unique per session. An unsolvable equation gives its residual equation (zero on the
right), as the original does; this module never raises on parser output.

Two deliberate differences from the original: periodic equations get one representative
per family near the origin, without Derive's redundant ones, and solutions come in a
stable order, reals ascending and then complex ones by angle.

Domains don't filter solutions (manual 5.4); solutions are found over the complexes,
except that an equation in the `ABS` of a real variable is solved over the reals.
`SOLVE(u, x, Real)` drops solutions known to have an imaginary component (manual 6.1);
`x = SQRT(a)`, ranges and residual equations stay.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar

import sympy as sp
from sympy.core.relational import Relational

from rederive.engine.approximation import GUARD, simplest_rational
from rederive.engine.context import Branch, Context, Precision
from rederive.engine.normal import normal_form
from rederive.engine.ordering import main_order
from rederive.engine.printer import is_written, spelling
from rederive.engine.signs import polar_logarithms
from rederive.engine.to_sympy import Vector

__all__ = ["equations_of", "oriented", "solutions", "solution_variables"]

#: The number of pieces a bracketing scan cuts an interval into before bisecting:
#: fine enough to separate roots in ordinary worksheets, coarse enough to be cheap.
_SCAN = 64

#: How many times a piece of the scan next to a point where the equation is undefined is
#: scanned again, more finely, for a root beside that point.
_RESCANS = 2

#: The starting half-width of Mixed mode's search, and the limit on doubling before
#: it gives up and returns the residual equation. The manual documents the doubling
#: but not a cap; the cap is ours, since an endless search is not an answer.
_REACH = 10
_REACH_CAP = 1 << 16


class _Constants:
    """The `@n` counter, handed out in order and never reused."""

    def __init__(self, first: int = 1) -> None:
        self.next = max(1, first)

    def fresh(self) -> sp.Symbol:
        constant = sp.Symbol(f"@{self.next}")
        self.next += 1
        return constant


#: The `@n` counter of the command being computed, or None outside `minting`.
_MINTING: ContextVar[_Constants | None] = ContextVar("minting", default=None)


@contextmanager
def minting(context: Context) -> Iterator[None]:
    """Run a command whose solves share one `@n` counter, starting at the context's.

    Every solve during the command starts past the values the earlier ones handed out,
    so `VECTOR(SOLVE(w = w, w), i, 1, 3)` gives `@1`, `@2` and `@3` rather than one
    constant three times: a utility file makes fresh unknowns this way. A command run
    inside another, as Simplify runs inside soLve, shares the outer one's counter.
    """
    if _MINTING.get() is not None:
        yield
        return
    token = _MINTING.set(_Constants(context.arbitrary_index))
    try:
        yield
    finally:
        _MINTING.reset(token)


def solutions(
    expression: sp.Basic,
    context: Context | None = None,
    variables: Sequence[str] = (),
    bounds: tuple[sp.Basic, sp.Basic] | None = None,
    divisors: Sequence[sp.Basic] = (),
    real: bool = False,
) -> tuple[sp.Basic, ...]:
    """The entries `expression` solves to, one per worksheet line.

    `variables` lists the variables to solve for; if empty, the most main ones are
    used, one per equation. `bounds` confines a numeric search (Approximate precision);
    None leaves it to the precision mode. `divisors` are what the equation divided by
    before Simplify cancelled anything: a value making one zero is not a solution.
    `real` drops solutions known to have an imaginary component.

    A solution written with a function the notation has no name for, such as sympy's
    inverse error function, is no answer, since it couldn't be typed back: the
    equation is then left unsolved.
    """
    context = context or Context()
    minted = _MINTING.get()
    constants = _Constants(
        context.arbitrary_index
        if minted is None
        else max(minted.next, context.arbitrary_index)
    )
    if not variables and not _named_in(expression):
        # An equation whose variable cancelled out, like `x/x = 1`, is still in that
        # variable via its divisors, and is solved by any value of it.
        divided = {name for divisor in divisors for name in _named_in(divisor)}
        variables = main_order(divided, context.order)[:1]
    try:
        found = _solutions(
            expression,
            context,
            tuple(variables),
            bounds,
            tuple(divisors),
            constants,
            real,
        )
        if any(_unnamed(answer) for answer in found):
            return _unsolvable(expression, context)
        if minted is not None:
            minted.next = constants.next
        return found
    except Exception:
        # The engine never raises on parser output, and "can't solve this" is an
        # answer, not a failure: the residual equation says exactly what is known.
        return _unsolvable(expression, context)


def _unnamed(answer: sp.Basic) -> bool:
    """Whether `answer` holds a call the notation has no name for."""
    return any(spelling(call) is None for call in answer.atoms(sp.Function))


def solution_variables(
    expression: sp.Basic, context: Context, count: int = 1
) -> tuple[sp.Symbol, ...]:
    """The `count` most main variables, for when the user names none."""
    names = main_order(_named_in(expression), context.order)
    symbols = _symbols_of(expression)
    return tuple(symbols.get(name, sp.Symbol(name)) for name in names[:count])


def equations_of(expression: sp.Basic) -> tuple[Relational, ...] | None:
    """The equations of a system, or None if this isn't a system.

    A system is a vector of relations or expressions, an expression meaning `u = 0`, as
    in `EXACT_EIGENVECTOR` in VECTOR.MTH. A vector containing vectors or truth values
    isn't a system.
    """
    elements = _elements(expression)
    if not elements or not all(_equatable(e) for e in elements):
        return None
    return tuple(
        e if isinstance(e, Relational) else _relation(sp.Eq, e, sp.Integer(0))
        for e in elements
    )


def _equatable(element: sp.Basic) -> bool:
    """Whether a vector element can be one equation of a system."""
    if isinstance(element, Relational):
        return True
    return isinstance(element, sp.Expr) and not isinstance(element, Vector)


def oriented(solved: sp.Basic) -> sp.Basic:
    """A solved relation with the variable on the left: sympy's `-3 < x` becomes `x > -3`.
    Simplify uses this for ranges too.
    """
    if isinstance(solved, sp.And | sp.Or):
        return solved.func(*(oriented(part) for part in solved.args))
    if isinstance(solved, Relational) and solved.lhs.is_number:
        return solved.reversed
    return solved


# -- dispatch on the kind of problem ------------------------------------------


def _solutions(
    expression: sp.Basic,
    context: Context,
    variables: tuple[str, ...],
    bounds: tuple[sp.Basic, sp.Basic] | None,
    divisors: tuple[sp.Basic, ...],
    constants: _Constants,
    real: bool,
) -> tuple[sp.Basic, ...]:
    equations = equations_of(expression)
    if equations is not None:
        return _system(equations, context, variables, constants, real)
    return _scalar(expression, context, variables, bounds, divisors, constants, real)


def _scalar(
    expression: sp.Basic,
    context: Context,
    variables: tuple[str, ...],
    bounds: tuple[sp.Basic, sp.Basic] | None,
    divisors: tuple[sp.Basic, ...],
    constants: _Constants,
    real: bool,
) -> tuple[sp.Basic, ...]:
    """One equation or inequality, solved for one variable.

    Inequalities are already solved over the reals, so `real` has no effect on them.
    """
    chosen = _chosen(expression, context, variables, 1)
    if not chosen:
        return _unsolvable(expression, context)
    variable = chosen[0]
    if isinstance(expression, Relational) and not isinstance(expression, sp.Equality):
        return _inequality(expression, variable, context, constants)
    return _equation(
        _residual(expression), variable, context, bounds, divisors, constants, real
    )


def _chosen(
    expression: sp.Basic,
    context: Context,
    variables: tuple[str, ...],
    count: int,
) -> tuple[sp.Symbol, ...]:
    """The variables to solve for: the user's choices, or the most main ones."""
    if not variables:
        return solution_variables(expression, context, count)
    symbols = _symbols_of(expression)
    return tuple(symbols.get(name, sp.Symbol(name)) for name in variables)


def _residual(expression: sp.Basic) -> sp.Basic:
    """What must be zero: a relation's difference, or a bare expression itself."""
    if isinstance(expression, Relational):
        return expression.lhs - expression.rhs
    return expression


# -- equations ---------------------------------------------------------------


def _equation(
    residual: sp.Basic,
    variable: sp.Symbol,
    context: Context,
    bounds: tuple[sp.Basic, sp.Basic] | None,
    divisors: tuple[sp.Basic, ...],
    constants: _Constants,
    real: bool,
) -> tuple[sp.Basic, ...]:
    """`residual = 0` solved for `variable`, by whatever the precision allows.

    Bounds mean a numeric search (Approximate). Otherwise an exact solution is tried
    first, and Mixed and Approximate then search for a real root over a widening
    interval: under Approximate, `SOLVE(ERF(x) + x = 1, x)` is `[x = 0.489116]`.

    A solution written with a function the notation has no name for is no answer (see
    `solutions`), so under Approximate `SOLVE(ERF(x) = 0.3, x)` is searched for too.

    A candidate is dropped only if provably not a solution: it zeros a divisor of the
    original equation (`_divides_by_zero`), substituting it back fails (`_fails`), or,
    with `real`, it is known to be non-real (`_imaginary`). If nothing remains, the
    answer is "no solutions found".
    """
    if bounds is not None:
        return _numeric(residual, variable, bounds[0], bounds[1], context)
    if residual.is_zero or _cancels(residual, variable):
        return (_equated(variable, constants.fresh(), context),)
    found = _values(residual, variable)
    if (found is None or found.has(sp.ConditionSet)) and context.branch is Branch.ANY:
        found = _inverted_whatever(residual, variable) or found
    if found is not None:
        polynomial = _polynomial(residual, variable) is not None

        def admissible(value: sp.Basic) -> bool:
            if _divides_by_zero(divisors, variable, value):
                return False
            if real and _imaginary(value):
                return False
            return polynomial or not _fails(residual, variable, value)

        entries = _entries(found, variable, context, constants, admissible)
        if entries is not None and not any(_unnamed(entry) for entry in entries):
            return entries
    if context.precision is not Precision.EXACT:
        searched = _searched(residual, variable, context)
        if searched is not None:
            return searched
    return _unsolved(residual, context)


def _values(residual: sp.Basic, variable: sp.Symbol) -> sp.Set | None:
    """Every value of `variable` that makes `residual` vanish, or None if unknown.

    Polynomials go to `roots` rather than `solveset`, as in the original: it gives
    radicals rather than naming roots by their polynomial, and collapses
    multiplicities. `trig` gives Viete's form in the casus irreducibilis, so `x^3 - 3*x
    + 1` yields the original's `2*COS(2*pi/9)` and companions; a rational cubic with
    one real root gets its radicals from `_cardano_roots`.
    """
    polynomial = _polynomial(residual, variable)
    if polynomial is not None:
        if polynomial.degree() < 1:
            return sp.S.EmptySet if not polynomial.is_zero else sp.S.Complexes
        found = (
            _binomial_roots(polynomial)
            or _cardano_factored(polynomial)
            or sp.roots(polynomial, trig=True)
        )
        if not found or sum(found.values()) != polynomial.degree():
            return None
        return sp.FiniteSet(*found)
    if variable.is_real and _has_absolute(residual, variable):
        return _over_reals(residual, variable)
    try:
        found = sp.solveset(residual, variable, sp.S.Complexes)
    except Exception:
        return None
    if found.has(sp.ConditionSet):
        rewritten = _exponential(residual, variable)
        if rewritten is not None:
            found = rewritten
    if isinstance(found, sp.ConditionSet):
        return _inverted(residual, variable)
    return found


def _has_absolute(residual: sp.Basic, variable: sp.Symbol) -> bool:
    return any(part.has(variable) for part in residual.atoms(sp.Abs))


def _over_reals(residual: sp.Basic, variable: sp.Symbol) -> sp.Set | None:
    """The real solutions of an equation in `ABS`, or None.

    `solveset` over the reals splits the equation by the signs of the `ABS` arguments.
    Over the complexes `ABS(x) = 3` is a circle, so this is only for real variables.
    The answer can be a range: `ABS(x) = x` holds for `x >= 0`.
    """
    try:
        found = sp.solveset(residual, variable, sp.S.Reals)
    except Exception:
        return None
    return None if found.has(sp.ConditionSet) else found


def _exponential(residual: sp.Basic, variable: sp.Symbol) -> sp.Set | None:
    """The solutions of `residual` with powers of positive numbers rewritten as powers of
    `#e`, or None if that doesn't help, since `solveset` inverts `#e^x = c` but not
    `3^x = c`.
    """

    def rewritable(part: sp.Basic) -> bool:
        return (
            part.is_Pow
            and part.base is not sp.E
            and part.base.is_number
            and part.base.is_positive is True
            and part.exp.has(variable)
        )

    rewritten = residual.replace(
        rewritable, lambda part: sp.exp(part.exp * sp.log(part.base))
    )
    if rewritten == residual:
        return None
    try:
        found = sp.solveset(rewritten, variable, sp.S.Complexes)
    except Exception:
        return None
    return None if found.has(sp.ConditionSet) else found


def _binomial_roots(polynomial: sp.Poly) -> dict[sp.Basic, int] | None:
    """The roots of `a*x^n + b` for real `-b/a`, as one real root times each root of
    unity, or None.

    `roots` distributes the real root into every term, so `x^100 = 12` would repeat the
    hundredth root of 12 three times per solution.
    """
    degree = polynomial.degree()
    terms = polynomial.terms()
    if degree < 2 or len(terms) != 2 or terms[1][0] != (0,):
        return None
    value = -terms[1][1] / terms[0][1]
    if not (value.is_number and value.is_extended_real and value.is_nonzero):
        return None
    unit = sp.Dummy("unit")
    sign = 1 if value.is_positive else -1
    unity = sp.roots(sp.Poly(unit**degree - sign, unit), trig=True)
    if sum(unity.values()) != degree:
        return None
    modulus = sp.Abs(value) ** sp.Rational(1, degree)
    return {modulus * root: 1 for root in unity}


def _cardano_factored(polynomial: sp.Poly) -> dict[sp.Basic, int] | None:
    """The roots of a rational polynomial with a cubic factor handled by `_cardano_roots`,
    each factor solved separately, or None. Since `roots` also factors over the
    rationals, this loses nothing.
    """
    try:
        if polynomial.degree() < 3 or polynomial.domain not in (sp.ZZ, sp.QQ):
            return None
        _, factors = polynomial.factor_list()
        written = [_cardano_roots(factor) for factor, _ in factors]
        if not any(written):
            return None
        found: dict[sp.Basic, int] = {}
        for (factor, multiplicity), roots in zip(factors, written):
            for root, count in (roots or sp.roots(factor, trig=True)).items():
                found[root] = found.get(root, 0) + count * multiplicity
        return found
    except Exception:
        return None


def _cardano_roots(polynomial: sp.Poly) -> dict[sp.Basic, int] | None:
    """The roots of an irreducible rational cubic with one real root, in Cardano's
    radicals as the original writes them, or None.

    For the depressed cubic `t^3 + p*t + q`, the real root is `u + v`, the real cube
    roots of `-q/2 +- SQRT(D)` with `D = q^2/4 + p^3/27` positive, and the complex pair
    is `-(u + v)/2 +- SQRT(3)*#i*(u - v)/2`. `roots` would write `v` as `-p/(3*u)`.
    `_real_cube_root` keeps negative radicands from being taken on the complex
    principal branch.
    """
    if polynomial.degree() != 3 or polynomial.domain not in (sp.ZZ, sp.QQ):
        return None
    if not polynomial.is_irreducible:
        return None
    leading, b, c, d = (sp.Rational(k) for k in polynomial.all_coeffs())
    b, c, d = b / leading, c / leading, d / leading
    shift = -b / 3
    p = c - b**2 / 3
    q = 2 * b**3 / 27 - b * c / 3 + d
    discriminant = q**2 / 4 + p**3 / 27
    if not discriminant.is_positive:
        return None
    surd = sp.sqrt(discriminant)
    u = _real_cube_root(-q / 2 + surd)
    v = _real_cube_root(-q / 2 - surd)
    if u is None or v is None:
        return None
    mean = shift - (u + v) / 2
    # Pull the coefficients of `u - v` out of the imaginary part, as simplification
    # would: `SQRT(3)*#i*(U - V)/4` rather than `SQRT(3)*#i*(U/2 - V/2)/2`.
    terms = sp.Add.make_args(u - v)
    coefficients = [term.as_coeff_Mul()[0] for term in terms]
    numerator = math.gcd(0, *(k.p for k in coefficients))
    content = sp.Rational(numerator, sp.ilcm(1, *(k.q for k in coefficients)))
    difference = sp.Add(*(term / content for term in terms))
    spread = content / 2 * sp.sqrt(3) * sp.I * difference
    return {shift + u + v: 1, mean + spread: 1, mean - spread: 1}


def _real_cube_root(value: sp.Basic) -> sp.Basic | None:
    """The real cube root of `alpha + beta*SQRT(m)` for rational `alpha` and `beta`, in
    natural form, or None.

    The sign and denominator are pulled out, as are cube factors: `(-SQRT(5) - 1)/2`
    has the root `-(4*SQRT(5) + 4)^(1/3)/2`.
    """
    alpha, surd = value.as_coeff_Add()
    beta, radical = surd.as_coeff_Mul()
    if not (alpha.is_Rational and beta.is_Rational):
        return None
    if radical != 1 and not (
        radical.is_Pow and radical.exp == sp.S.Half and radical.base.is_Integer
    ):
        return None
    if value.is_zero:
        return sp.S.Zero
    sign = 1 if value.is_positive else -1 if value.is_negative else None
    if sign is None:
        return None
    denominator = sp.ilcm(alpha.q, beta.q)
    scale = sign * denominator**3
    alpha, beta = int(alpha * scale), int(beta * scale)
    content = math.gcd(alpha, beta)
    cube = 1
    for prime, power in sp.factorint(content).items():
        cube *= prime ** (power // 3)
    inside = sp.Integer(alpha // cube**3) + sp.Integer(beta // cube**3) * radical
    return sign * sp.Integer(cube) * sp.Pow(inside, sp.Rational(1, 3)) / denominator


def _inverted(residual: sp.Basic, variable: sp.Symbol) -> sp.Set | None:
    """What `solve` finds where `solveset` gives up, or None.

    `solve` can invert Lambert W (`x*#e^x = 1`), and answers on both real branches. It
    respects the variable's real assumption, so an empty answer means only "no real
    solution" and isn't accepted; nor are inexpressible solutions or ones that don't
    check out.
    """
    try:
        found = sp.solve(residual, variable)
    except Exception:
        return None
    if not found or not all(
        is_written(value) and _satisfies(residual, variable, value) for value in found
    ):
        return None
    return sp.FiniteSet(*found)


def _inverted_whatever(residual: sp.Basic, variable: sp.Symbol) -> sp.Set | None:
    """What `solve` finds, as `_inverted`, checked as Branch Any takes powers, where a
    power of a power is one power whatever the base: `z*(x - r)^k = 1` is solved by `x =
    r + (1/z)^(1/k)`, which ODE.MTH's `POLES` needs for the hypergeometric equation.

    A product is solved factor by factor, and a power to an exponent not known to be
    positive is no factor that vanishes: `(1 - z*(x - r)^k)*(x - r)^(1 - k)` is zero
    only where its first factor is."""
    if residual.is_Mul:
        found = []
        for factor in residual.args:
            if not factor.has(variable):
                continue
            base, exponent = factor.as_base_exp()
            if not exponent.is_Number and exponent.is_positive is not True:
                continue
            values = _inverted_whatever(factor, variable)
            if values is None:
                return None
            found.extend(values)
        return sp.FiniteSet(*found) if found else None
    try:
        found = sp.solve(residual, variable)
    except Exception:
        return None

    def checks(value: sp.Basic) -> bool:
        try:
            left = sp.powdenest(residual.subs(variable, value), force=True)
            return sp.simplify(left).is_zero is True
        except Exception:
            return False

    if not found or not all(is_written(value) and checks(value) for value in found):
        return None
    return sp.FiniteSet(*found)


def _satisfies(residual: sp.Basic, variable: sp.Symbol, value: sp.Basic) -> bool:
    """Whether substituting `value` is proved to make `residual` vanish, applying Lambert
    W identities that simplification misses.
    """
    try:
        left = residual.subs(variable, value)
        if sp.simplify(left).is_zero is True:
            return True
        if left.free_symbols or not left.has(sp.LambertW):
            return False
        return sp.simplify(_unwound(left)).is_zero is True
    except Exception:
        return False


def _unwound(expression: sp.Basic) -> sp.Basic:
    """The expression with Lambert W identities applied, for `W` a branch of
    `LAMBERTW(z)`.

    `#e^(k*W)` becomes `(z/W)^k`, and `LN(c*W)` becomes `LN(c*z) - W` where `c*W` is
    real and positive or `c` is 1 and `W` real, since `W*#e^W` is `z` on every branch.
    Two rounds, since rewriting a logarithm can expose an exponential of it.
    """

    def exponential(call: sp.exp) -> sp.Basic | None:
        multiple, branch = sp.expand(call.args[0]).as_coeff_Mul()
        if not (isinstance(branch, sp.LambertW) and multiple.is_Integer):
            return None
        return (branch.args[0] / branch) ** multiple

    def logarithm(call: sp.log) -> sp.Basic | None:
        argument = call.args[0]
        for sign in (1, -1):
            branch = sign * argument
            if not isinstance(branch, sp.LambertW):
                continue
            real = branch.is_extended_real is True
            if (sign == 1 and real) or (real and (sign * branch).is_positive is True):
                return sp.log(sign * branch.args[0]) - branch
        return None

    expression = expression.rewrite(sp.exp)
    for _ in range(2):
        expression = expression.replace(
            lambda e: isinstance(e, sp.exp) and exponential(e) is not None, exponential
        )
        expression = expression.replace(
            lambda e: isinstance(e, sp.log) and logarithm(e) is not None, logarithm
        )
        expression = sp.expand_log(sp.powsimp(sp.expand_power_exp(expression)))
    return expression


def _fails(residual: sp.Basic, variable: sp.Symbol, value: sp.Basic) -> bool:
    """Whether substituting `value` into `residual` provably leaves it nonzero.

    Combining logarithms turns `LN(x) + LN(x - 1) = LN(6)` into `x*(x - 1) = 6`,
    offering the false solution -2. Only decided results drop a candidate.
    """
    try:
        left = residual.xreplace({variable: value})
        if left.free_symbols:
            return False
        return sp.simplify(left).is_zero is False
    except Exception:
        return False


def _divides_by_zero(
    divisors: tuple[sp.Basic, ...], variable: sp.Symbol, value: sp.Basic
) -> bool:
    """Whether `value` provably zeros one of the equation's divisors. The variable is
    matched by name, which identifies it across Simplify.
    """
    for divisor in divisors:
        try:
            named = {
                symbol: value
                for symbol in divisor.free_symbols
                if getattr(symbol, "name", None) == variable.name
            }
            if not named:
                continue
            if sp.simplify(divisor.xreplace(named)).is_zero is True:
                return True
        except Exception:
            continue
    return False


def _cancels(residual: sp.Basic, variable: sp.Symbol) -> bool:
    """Whether `residual` is a rational function that cancels to zero, making the equation
    an identity, like `1/(x - 1) + 1 = x/(x - 1)`, which solveset says has no solution.
    """
    try:
        return bool(residual.is_rational_function(variable)) and sp.cancel(residual) == 0
    except Exception:
        return False


def _polynomial(residual: sp.Basic, variable: sp.Symbol) -> sp.Poly | None:
    """`residual` as a polynomial in `variable`, or None if it isn't one."""
    try:
        return sp.Poly(residual, variable)
    except Exception:
        return None


def _entries(
    found: sp.Set,
    variable: sp.Symbol,
    context: Context,
    constants: _Constants,
    admissible: Callable[[sp.Basic], bool] = lambda value: True,
) -> tuple[sp.Basic, ...] | None:
    """A solution set as relations in a stable order, omitting values `admissible`
    rejects, or None if no finite list of relations describes it.
    """
    if _is_everything(found):
        return (_equated(variable, constants.fresh(), context),)
    values = _members(found)
    if values is None:
        return _ranges(found, variable, context, constants)
    return _answers(variable, [value for value in values if admissible(value)], context)


def _members(found: sp.Set) -> list[sp.Basic] | None:
    """The members of a solution set as a finite list, or None.

    Solution families (images of the integers) have no notation, so each contributes
    one representative near the origin.
    """
    if found is sp.S.EmptySet:
        return []
    if isinstance(found, sp.FiniteSet):
        return list(found.args)
    if isinstance(found, sp.Union):
        gathered: list[sp.Basic] = []
        for piece in found.args:
            part = _members(piece)
            if part is None:
                return None
            gathered.extend(part)
        return gathered
    if isinstance(found, sp.ImageSet) and found.base_sets == (sp.S.Integers,):
        representative = _representative(found)
        return None if representative is None else [representative]
    if isinstance(found, sp.Complement):
        return _remaining(*found.args)
    return None


def _remaining(inside: sp.Set, excluded: sp.Set) -> list[sp.Basic] | None:
    """The members of `inside` that aren't provably members of `excluded`.

    For `SOLVE(1 - 2*m/r = 0, r)` sympy answers `Complement({2*m}, {0})`. Undecidable
    isn't excluded, so the answer is the original's `r = 2*m`. Only finite sets are
    handled; a family minus a member has no notation.
    """
    if not isinstance(inside, sp.FiniteSet) or not isinstance(excluded, sp.FiniteSet):
        return None
    return [value for value in inside.args if not _provably_holds(excluded, value)]


def _provably_holds(excluded: sp.Set, value: sp.Basic) -> bool:
    """Whether `excluded` provably contains `value`; only sympy's boolean true counts."""
    try:
        return excluded.contains(value) is sp.true
    except Exception:
        return False


#: The indices at which a family is sampled, nearest the origin first. Two periods each
#: way suffice to find the member nearest zero: no family's step exceeds its period, so
#: a third would only be further out.
_NEARBY = (0, -1, 1, -2, 2)


def _representative(family: sp.ImageSet) -> sp.Basic | None:
    """The member of one solution family nearest the origin, as the original prints it:
    for `COS(x) = 1/2` that is `-pi/3`, not the index-zero `5*pi/3`. A family that
    can't be compared numerically is sampled at zero.
    """
    members = []
    for index in _NEARBY:
        try:
            members.append(family.lamda(index))
        except Exception:
            continue
    if not members:
        return None
    return min(members, key=_distance)


def _distance(value: sp.Basic) -> tuple[int, float, int]:
    """The distance of a family member from the origin, with ties going to the positive
    one."""
    number = _complex(value)
    if number is None:
        return (1, 0.0, 0)
    return (0, abs(number), 0 if number.real >= 0 else 1)


def _is_everything(found: sp.Set) -> bool:
    """Whether the set contains every value, i.e. the equation is an identity."""
    return found in (sp.S.Complexes, sp.S.Reals, sp.S.UniversalSet)


def _unsolved(expression: sp.Basic, context: Context) -> tuple[sp.Basic, ...]:
    """The residual relation, the answer for anything unsolved: `COS(x) = x` comes back as
    `COS(x) - x = 0`. The operator is kept, and a relation whose sides can't be
    subtracted is returned as is.
    """
    func = expression.func if isinstance(expression, Relational) else sp.Eq
    try:
        residual = normal_form(_residual(expression), context.order)
    except Exception:
        return (expression,)
    return (_relation(func, residual, sp.Integer(0)),)


def _unsolvable(expression: sp.Basic, context: Context) -> tuple[sp.Basic, ...]:
    """The residual form of anything, for code paths that give up. A system gives its
    vector of residuals, since "no solutions found" would be false.
    """
    equations = equations_of(expression)
    if equations is None:
        return _unsolved(expression, context)
    residuals = [_unsolved(equation, context)[0] for equation in equations]
    return (Vector(*residuals),)


# -- inequalities ------------------------------------------------------------


def _inequality(
    relation: Relational,
    variable: sp.Symbol,
    context: Context,
    constants: _Constants,
) -> tuple[sp.Basic, ...]:
    """`<`, `<=`, `>`, `>=` and `/=`, solved into ranges of the variable.

    A one-variable problem is solved as a set, giving one chained entry per interval.
    Anything else, such as a symbolic bound, goes to `reduce_inequalities`.
    """
    # Relations arrive unevaluated, as Simplify builds them; the solvers need the
    # evaluated form.
    evaluated = _evaluated(relation)
    if evaluated is sp.true:
        return (_equated(variable, constants.fresh(), context),)
    if evaluated is sp.false:
        return ()
    if evaluated.free_symbols == {variable}:
        try:
            found = sp.solve_univariate_inequality(
                evaluated, variable, relational=False
            )
        except Exception:
            found = None
        if found is not None:
            entries = _ranges(found, variable, context, constants)
            if entries is not None:
                return entries
    try:
        reduced = sp.reduce_inequalities([evaluated], [variable])
    except Exception:
        return _unsolved(relation, context)
    entries = _reduced(reduced, variable, context, constants)
    if entries is None:
        return _unsolved(relation, context)
    return entries


def _evaluated(relation: Relational) -> sp.Basic:
    try:
        return relation.func(relation.lhs, relation.rhs)
    except Exception:
        return relation


def _ranges(
    found: sp.Set,
    variable: sp.Symbol,
    context: Context,
    constants: _Constants,
) -> tuple[sp.Basic, ...] | None:
    """One entry per piece of a solution set, or None if it isn't a range."""
    if found is sp.S.EmptySet:
        return ()
    if _is_everything(found):
        return (_equated(variable, constants.fresh(), context),)
    if isinstance(found, sp.Union):
        gathered: list[sp.Basic] = []
        for piece in found.args:
            part = _ranges(piece, variable, context, constants)
            if part is None:
                return None
            gathered.extend(part)
        return tuple(gathered)
    if isinstance(found, sp.FiniteSet):
        return _answers(variable, found.args, context)
    if isinstance(found, sp.Interval):
        return (_span(found, variable, context, constants),)
    return None


def _span(
    interval: sp.Interval,
    variable: sp.Symbol,
    context: Context,
    constants: _Constants,
) -> sp.Basic:
    """One interval as a relation: a chain for two finite ends, a single relation for one,
    and an identity for none.
    """
    low, high = interval.start, interval.end
    below = sp.Lt if interval.right_open else sp.Le
    above = sp.Lt if interval.left_open else sp.Le
    if low is sp.S.NegativeInfinity and high is sp.S.Infinity:
        return _equated(variable, constants.fresh(), context)
    if low is sp.S.NegativeInfinity:
        return _relation(below, variable, normal_form(high, context.order))
    if high is sp.S.Infinity:
        return oriented(_relation(above, normal_form(low, context.order), variable))
    return sp.And(
        _relation(above, normal_form(low, context.order), variable),
        _relation(below, variable, normal_form(high, context.order)),
    )


def _reduced(
    reduced: sp.Basic,
    variable: sp.Symbol,
    context: Context,
    constants: _Constants,
) -> tuple[sp.Basic, ...] | None:
    """The answer of `reduce_inequalities`, as entries with the variable on the left."""
    if reduced is sp.true:
        return (_equated(variable, constants.fresh(), context),)
    if reduced is sp.false:
        return ()
    if isinstance(reduced, sp.Or):
        gathered: list[sp.Basic] = []
        for part in reduced.args:
            entries = _reduced(part, variable, context, constants)
            if entries is None:
                return None
            gathered.extend(entries)
        return tuple(gathered)
    if isinstance(reduced, sp.And):
        parts = [oriented(_written(part, context)) for part in reduced.args]
        return (sp.And(*parts),)
    if isinstance(reduced, Relational):
        return (oriented(_written(reduced, context)),)
    return None


def _written(relation: Relational, context: Context) -> Relational:
    """A relation with both sides in normal form, so `x >= 3*y/2 - 7/2` becomes `x >= (3*y
    - 7)/2`.
    """
    return _relation(
        relation.func,
        normal_form(relation.lhs, context.order),
        normal_form(relation.rhs, context.order),
    )


# -- systems -----------------------------------------------------------------


def _system(
    equations: tuple[Relational, ...],
    context: Context,
    variables: tuple[str, ...],
    constants: _Constants,
    real: bool,
) -> tuple[sp.Basic, ...]:
    """A vector of equations, solved for as many variables as there are equations.

    Each solution is one entry, the solution vector. Unchosen variables stay on the
    right, so `[x + y + z = 1, x - y = 0]` can be solved for `x` and `y`. With `real`,
    a solution with any known non-real value is dropped.
    """
    joined = Vector(*equations)
    chosen = _chosen(joined, context, variables, len(equations))
    if not chosen:
        return _unsolvable(joined, context)
    residuals = [_residual(equation) for equation in equations]
    solved = _linear(residuals, chosen, constants)
    if solved is None:
        solved = _nonlinear(residuals, chosen)
    if solved is None:
        return _unsolvable(joined, context)
    if real:
        solved = [
            mapping
            for mapping in solved
            if not any(_imaginary(value) for value in mapping.values())
        ]
    return tuple(
        Vector(
            *(
                _equated(variable, mapping.get(variable, variable), context)
                for variable in chosen
            )
        )
        for mapping in solved
    )


def _linear(
    residuals: list[sp.Basic],
    chosen: tuple[sp.Symbol, ...],
    constants: _Constants,
) -> list[dict[sp.Symbol, sp.Basic]] | None:
    """The solution of a system linear in the chosen variables, or None if `linsolve`
    rejects it. An empty list means the system is inconsistent.

    The variables are passed in reverse because `linsolve` leaves the last one free and
    the original parametrizes the first: `[x + 3*y = 1, 2*x + 6*y = 2]` is `[x = @1, y
    = (1 - @1)/3]`. Free variables become fresh `@n` constants.
    """
    try:
        found = sp.linsolve(residuals, list(reversed(chosen)))
    except Exception:
        return None
    if not isinstance(found, sp.Set):
        return None
    if found is sp.S.EmptySet:
        return []
    if not isinstance(found, sp.FiniteSet) or len(found.args) != 1:
        return None
    values = dict(zip(reversed(chosen), found.args[0]))
    arbitrary = {
        variable: constants.fresh()
        for variable in chosen
        if values.get(variable) == variable
    }
    return [
        {variable: value.xreplace(arbitrary) for variable, value in values.items()}
    ]


def _nonlinear(
    residuals: list[sp.Basic], chosen: tuple[sp.Symbol, ...]
) -> list[dict[sp.Symbol, sp.Basic]] | None:
    """A system `linsolve` refused, solved as far as `solve` can, beyond what the original
    handles. An empty answer means "nothing found", not "no solutions".
    """
    try:
        found = sp.solve(residuals, list(chosen), dict=True)
    except Exception:
        return None
    if not found:
        return None
    return [dict(mapping) for mapping in found]


# -- numbers -----------------------------------------------------------------


def _numeric(
    residual: sp.Basic,
    variable: sp.Symbol,
    low: sp.Basic,
    high: sp.Basic,
    context: Context,
) -> tuple[sp.Basic, ...]:
    """Every real root in `[low, high]`, for Approximate precision.

    Unlike Derive, all of a polynomial's roots are returned. Anything else is bracketed
    by a sign-change scan and bisected, finding one root; an infinite interval goes to
    `_unbounded`, and if that finds nothing the equation is returned unsolved.
    """
    if _polynomial(residual, variable) is None and not _finite(low, high):
        found = _unbounded(residual, variable, low, high, context)
        return found if found is not None else _unsolved(residual, context)
    return tuple(
        _equated(variable, root, context)
        for root in _real_roots(residual, variable, low, high, context)
    )


def _real_roots(
    residual: sp.Basic,
    variable: sp.Symbol,
    low: sp.Basic,
    high: sp.Basic,
    context: Context,
) -> list[sp.Basic]:
    polynomial = _polynomial(residual, variable)
    if polynomial is not None:
        # Each distinct root once, collapsing multiplicity as the exact path does:
        # `(x - 1)^2` has the single solution 1.
        try:
            found = [root for root, _ in polynomial.real_roots(multiple=False)]
        except Exception:
            found = []
        return [
            _approximate(root, context) for root in found if _within(root, low, high)
        ]
    root = _bracketed(residual, variable, low, high, context)
    return [] if root is None else [root]


def _finite(low: sp.Basic, high: sp.Basic) -> bool:
    """Whether neither end of a search interval is infinite."""
    ends = (_number(low), _number(high))
    return not any(end is not None and math.isinf(end) for end in ends)


def _unbounded(
    residual: sp.Basic,
    variable: sp.Symbol,
    low: sp.Basic,
    high: sp.Basic,
    context: Context,
) -> tuple[sp.Basic, ...] | None:
    """A root in an interval with an infinite end, or None.

    This searches a doubling window like Mixed mode's, from `_REACH` to `_REACH_CAP`.
    Finding nothing proves nothing, hence None rather than an empty tuple.
    """
    first, last = _number(low), _number(high)
    if first is None or last is None or not first < last:
        return None
    reach = _REACH
    while reach <= _REACH_CAP:
        if math.isinf(first) and math.isinf(last):
            window = (-reach, reach)
        elif math.isinf(last):
            window = (first, first + reach)
        else:
            window = (max(first, last - reach), last)
        start, end = (sp.Rational(bound) for bound in window)
        found = _real_roots(residual, variable, start, end, context)
        if found:
            return tuple(_equated(variable, root, context) for root in found)
        reach *= 2
    return None


def _approximate(root: sp.Basic, context: Context) -> sp.Basic:
    """A root as the simplest rational with its digits, like every approximate number, so
    it prints like one: `-1.16730` at six digits, truncated.
    """
    digits = context.precision_digits
    return simplest_rational(root.evalf(digits + GUARD), digits)


def _within(root: sp.Basic, low: sp.Basic, high: sp.Basic) -> bool:
    value, first, last = _number(root), _number(low), _number(high)
    if value is None or first is None or last is None:
        return False
    return first <= value <= last


def _number(value: sp.Basic) -> float | None:
    """`value` as a real float, or None if it isn't real."""
    try:
        found = complex(sp.sympify(value).evalf())
    except Exception:
        return None
    if found.imag:
        return None
    return found.real


def _bracketed(
    residual: sp.Basic,
    variable: sp.Symbol,
    low: sp.Basic,
    high: sp.Basic,
    context: Context,
) -> sp.Basic | None:
    """One root of `residual` in `[low, high]`, found by scanning and bisection. The scan
    finds a piece whose ends straddle a single root.
    """
    first, last = _number(low), _number(high)
    if first is None or last is None or not first < last:
        return None
    return _scanned(residual, variable, first, last, context, _RESCANS)


def _scanned(
    residual: sp.Basic,
    variable: sp.Symbol,
    first: float,
    last: float,
    context: Context,
    depth: int,
) -> sp.Basic | None:
    """One root in `[first, last]` by a scan of `_SCAN` pieces, or None.

    A point where the residual is undefined tells nothing about the sign beside it, so
    a root next to it is missed: `ATAN(z - 0.1/z) + ATAN(z + 0.1/z) = 0.5` is undefined
    at 0 and has its root at 0.28, short of the next point. Once the whole interval is
    scanned, a piece with one end undefined is scanned again, `depth` levels deep. A
    piece undefined at both ends is not, nor one whose sign change bisects to no root:
    rescanning every pole of `1/SIN(x)` makes the widening search far too slow.
    """
    step = (last - first) / _SCAN
    suspect: list[tuple[float, float]] = []
    previous: float | None = None
    # Whether the previous point was defined; a piece undefined at both ends, as `LN(x)`
    # is to the left of 0 over the reals, is not rescanned.
    defined = False
    for index in range(_SCAN + 1):
        point = first + index * step
        evaluated = residual.subs(variable, sp.Float(point))
        value = _number(evaluated)
        if value is None:
            if defined:
                suspect.append((point - step, point))
            previous, defined = None, False
            continue
        # A value too small for a float is 0.0 without being zero: `#e^x` far to
        # the left is not a root.
        if value == 0.0 and _vanishes(evaluated):
            return _approximate(sp.Float(point), context)
        if index and not defined:
            suspect.append((point - step, point))
        elif previous is not None and (previous < 0) != (value < 0):
            found = _bisect(residual, variable, point - step, point, context)
            if found is not None:
                return found
        previous, defined = value, True
    if depth:
        for start, end in suspect:
            found = _scanned(residual, variable, start, end, context, depth - 1)
            if found is not None:
                return found
    return None


def _vanishes(value: sp.Basic) -> bool:
    try:
        return sp.sympify(value).evalf().is_zero is True
    except Exception:
        return False


def _bisect(
    residual: sp.Basic, variable: sp.Symbol, low: float, high: float, context: Context
) -> sp.Basic | None:
    digits = context.precision_digits + GUARD
    try:
        found = sp.nsolve(residual, variable, (low, high), solver="bisect", prec=digits)
    except Exception:
        return None
    return _approximate(found, context)


def _searched(
    residual: sp.Basic, variable: sp.Symbol, context: Context
) -> tuple[sp.Basic, ...] | None:
    """The last resort of Mixed and Approximate modes: search `[-10, 10]` and double until
    a real root turns up, as the manual describes for Mixed. The cap is ours.
    """
    reach = _REACH
    while reach <= _REACH_CAP:
        found = _real_roots(
            residual, variable, sp.Integer(-reach), sp.Integer(reach), context
        )
        if found:
            return tuple(_equated(variable, root, context) for root in found)
        reach *= 2
    return None


# -- formatting answers -------------------------------------------------------


def _equated(variable: sp.Symbol, value: sp.Basic, context: Context) -> sp.Basic:
    """`x = value`, with the value in normal form like every answer."""
    return _relation(sp.Eq, variable, normal_form(value, context.order))


def _relation(func, left: sp.Basic, right: sp.Basic) -> sp.Basic:
    """A relation built unevaluated, as the engine builds every relation."""
    try:
        return func(left, right, evaluate=False)
    except Exception:
        return func(left, right)


def _answers(
    variable: sp.Symbol, values: Iterable[sp.Basic], context: Context
) -> tuple[sp.Basic, ...]:
    """`x = value` for each distinct value, in `_ordered` order. Duplicates are removed
    again after normalization, since different-looking values can be equal.
    """
    written: list[sp.Basic] = []
    for value in _ordered(values):
        entry = _equated(variable, _tidied(value), context)
        if entry not in written:
            written.append(entry)
    return tuple(written)


def _tidied(value: sp.Basic) -> sp.Basic:
    """`value` in the plainer of the forms a solver may produce: `SQRT(r^2 - x^2)` for
    `SQRT(-(x + r)*(x - r))`, `LN(3)/(LN(3) - LN(2))` for `-LN(3)/(LN(2) - LN(3))`, and
    `LN(x)` for `LN(ABS(x)) + #i*PHASE(x)`.
    """
    try:
        value = polar_logarithms(value)
        value = value.replace(_factored_radicand, _expanded_radicand)
        return _unnegated(value)
    except Exception:
        return value


def _factored_radicand(part: sp.Basic) -> bool:
    return (
        part.is_Pow
        and part.exp in (sp.S.Half, -sp.S.Half)
        and part.base.is_Mul
        and any(factor.is_Add for factor in part.base.args)
        and bool(part.base.free_symbols)
    )


def _expanded_radicand(part: sp.Basic) -> sp.Basic:
    expanded = sp.expand(part.base)
    if sp.count_ops(expanded) > sp.count_ops(part.base):
        return part
    return sp.Pow(expanded, part.exp)


def _unnegated(value: sp.Basic) -> sp.Basic:
    if not value.is_Mul:
        return value
    coefficient, rest = value.as_coeff_Mul()
    if not coefficient.is_negative:
        return value
    factors = sp.Mul.make_args(rest)
    for index, factor in enumerate(factors):
        base = factor.base if factor.is_Pow and factor.exp == -1 else None
        if base is None or not base.is_Add or not base.is_number:
            continue
        if not base.is_extended_negative:
            continue
        others = factors[:index] + factors[index + 1 :]
        flipped = sp.Add(*(-term for term in base.args))
        return sp.Mul(-coefficient, *others, sp.Pow(flipped, -1))
    return value


def _ordered(values: Iterable[sp.Basic]) -> list[sp.Basic]:
    """The solutions in output order: reals ascending, then the rest. The original's order
    is accidental, so predictability wins.
    """
    seen: list[sp.Basic] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return sorted(seen, key=_rank)


def _rank(value: sp.Basic) -> tuple[int, float, float, object]:
    number = _complex(value)
    if number is None:
        return (2, 0.0, 0.0, sp.default_sort_key(value))
    if not number.imag:
        return (0, number.real, 0.0, ())
    # Sort by angle from the positive real axis, with the positive-imaginary conjugate
    # first, as the original prints a pair: `x = #i`, then `x = -#i`. The angle is
    # rounded so a pair differing in the last bit still sorts together.
    angle = round(abs(cmath.phase(number)), 9)
    return (1, angle, 0.0 if number.imag > 0 else 1.0, (abs(number),))


#: The imaginary part, relative to the number's magnitude, below which it is taken as
#: evaluation noise. A real root written with cancelling `#i` terms evaluates with an
#: imaginary residue some twenty digits down; a genuinely complex one is nowhere near
#: this small.
_NOISE = 1e-9


def _imaginary(value: sp.Basic) -> bool:
    """Whether `value` is known to have an imaginary component: for a number, an evaluated
    imaginary part above `_NOISE` of its magnitude; for a parametric value, proved
    non-real by sympy.
    """
    try:
        if not value.free_symbols:
            number = _complex(value)
            if number is None or not cmath.isfinite(number):
                return False
            return abs(number.imag) > _NOISE * abs(number)
        return value.is_extended_real is False
    except Exception:
        return False


def _complex(value: sp.Basic) -> complex | None:
    try:
        return complex(value.evalf())
    except Exception:
        return None


# -- inspecting expressions ---------------------------------------------------


def _elements(expression: sp.Basic) -> list[sp.Basic]:
    """The elements of a vector, or an empty list if this isn't a vector."""
    if isinstance(expression, Vector):
        return list(expression.args)
    return []


def _symbols_of(expression: sp.Basic) -> dict[str, sp.Symbol]:
    """The expression's symbols by name, so their domain declarations are preserved when
    the user names a variable.
    """
    return {
        symbol.name: symbol
        for symbol in expression.free_symbols
        if type(symbol) is sp.Symbol
    }


def _named_in(expression: sp.Basic) -> list[str]:
    return list(_symbols_of(expression))
