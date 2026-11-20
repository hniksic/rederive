"""The mathematics of soLve: finding the solutions of an equation.

This sits below the pipeline, like `factoring`, so the `solve` command and the solving
functions share it (`solution_forms` writes their answers). The result is a tuple of
entries, one worksheet line each: a relation with the variable on the left, a system's
solution vector, a chained range `-2 < x < 2`, one entry per piece of a union, or a
condition on the parameters under which every value is a solution (`R = 0`).

Three non-solutions are distinguished. The empty tuple means "no solutions found". An
identity gives `x = @1`, with the counter from `Context.arbitrary_index` so constants
are unique per session. An unsolvable equation gives itself, rearranged as the original
does (`_rearranged`); this module never raises on parser output.

`numeric_roots` is `NSOLVE`'s search, also the last resort of Mixed and Approximate
precision.

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
import itertools
import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from fractions import Fraction

import mpmath
import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.simplify.fu import TR1, TR2

from rederive.engine import budget
from rederive.engine.approximation import GUARD, simplest_rational
from rederive.engine.context import Branch, Context, DomainKind, Precision
from rederive.engine.normal import normal_form
from rederive.engine.ordering import main_order
from rederive.engine.printer import is_written, spelling, term_order
from rederive.engine.signs import polar_logarithms
from rederive.engine.to_sympy import (
    UNIT_CIRCLE,
    Assign,
    FunDef,
    Modulo,
    PlusMinus,
    Vector,
)

__all__ = [
    "Outcome",
    "arbitrary",
    "equations_of",
    "listed_in_order",
    "numeric_roots",
    "oriented",
    "outcome",
    "parameter_condition",
    "solutions",
    "solution_variables",
    "unsolved_equation",
]

#: The number of pieces a bracketing scan cuts an interval into before bisecting:
#: fine enough to separate roots in ordinary worksheets, coarse enough to be cheap.
_SCAN = 64

#: How many times a piece of the scan next to a point where the equation is undefined is
#: scanned again, more finely, for a root beside that point.
_RESCANS = 2

#: How many times the piece at an end of the interval where the equation is undefined is
#: scanned again: a root can lie next to an end that is a pole or a removable
#: singularity, as a bond's yield next to a rate of zero, and scanning one piece costs
#: little.
_EDGE_RESCANS = 8

#: The starting half-width of Mixed mode's search, and the limit on doubling before
#: it gives up and returns the residual equation. The manual documents the doubling
#: but not a cap; the cap is ours, since an endless search is not an answer.
_REACH = 10
_REACH_CAP = 1 << 16


class _Constants:
    """The `@n` counter, handed out in order and never reused."""

    def __init__(self, first: int = 1) -> None:
        self.next = max(1, first)
        self.minted: list[sp.Symbol] = []

    def fresh(self) -> sp.Symbol:
        constant = sp.Symbol(f"@{self.next}")
        self.next += 1
        self.minted.append(constant)
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


def arbitrary(context: Context) -> sp.Symbol:
    """A new arbitrary constant `@n` from the command's counter."""
    minted = _MINTING.get()
    constants = _Constants(
        context.arbitrary_index
        if minted is None
        else max(minted.next, context.arbitrary_index)
    )
    constant = constants.fresh()
    if minted is not None:
        minted.next = constants.next
    return constant


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

    An assignment or a definition is solved for what it defines: `W(k) := k^2 - 4`
    solves `k^2 - 4 = 0`.
    """
    return outcome(expression, context, variables, bounds, divisors, real).entries


@dataclass(frozen=True)
class Outcome:
    """What solving found: the entries `solutions` returns, and the arbitrary constants
    minted for them. An entry `x = @n` with `@n` minted says that every value of `x`
    is a solution; `@n` in a system's solution is a free value."""

    entries: tuple[sp.Basic, ...]
    minted: frozenset[sp.Symbol]

    def is_arbitrary(self, value: sp.Basic) -> bool:
        return value in self.minted


def outcome(
    expression: sp.Basic,
    context: Context | None = None,
    variables: Sequence[str] = (),
    bounds: tuple[sp.Basic, sp.Basic] | None = None,
    divisors: Sequence[sp.Basic] = (),
    real: bool = False,
    commit: bool = True,
) -> Outcome:
    """`solutions` with the constants it minted. With `commit` off, the constants are not
    taken from the command's counter, for a caller that shows none of them."""
    if isinstance(expression, (Assign, FunDef)):
        expression = expression.args[-1]
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
            return Outcome(_unsolvable(expression, context), frozenset())
        if minted is not None and commit:
            minted.next = constants.next
        return Outcome(found, frozenset(constants.minted))
    except Exception:
        # The engine never raises on parser output, and "can't solve this" is an
        # answer, not a failure: the residual equation says exactly what is known.
        return Outcome(_unsolvable(expression, context), frozenset())


def _unnamed(answer: sp.Basic) -> bool:
    """Whether `answer` holds a call the notation has no name for. A `Piecewise` is
    written as an `IF`."""
    return any(
        spelling(call) is None and not isinstance(call, sp.Piecewise)
        for call in answer.atoms(sp.Function)
    )


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
    in `EXACT_EIGENVECTOR` in VECTOR.MTH. A matrix is a system of all its elements, so an
    equation between matrices, such as `[[1, 2], [3, 4]] . [[x], [y]] = [[5], [6]]` with
    column vectors, is the system of the equations between their elements. A vector
    mixing vectors and other elements, or containing truth values, isn't a system.
    """
    elements = _elements(expression)
    if elements and all(isinstance(e, Vector) for e in elements):
        elements = [leaf for e in elements for leaf in _elements(e)]
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
    if equations is None and isinstance(expression, sp.And):
        equations = equations_of(Vector(*expression.args))
    if bounds is None:
        parted = _in_parts(expression, equations, context, variables, constants)
        if parted is not None:
            return parted
    if equations is not None:
        return _system(equations, context, variables, constants, real)
    return _scalar(expression, context, variables, bounds, divisors, constants, real)


def _in_parts(
    expression: sp.Basic,
    equations: tuple[Relational, ...] | None,
    context: Context,
    variables: tuple[str, ...],
    constants: _Constants,
) -> tuple[sp.Basic, ...] | None:
    """Complex equations in variables declared real, solved as the system of their real
    and imaginary parts, or None if that doesn't apply.

    With `x` and `y` declared Real, `3*x - (2 + #i)*y = -3*#i` is the two equations
    `3*x - 2*y = 0` and `-y = -3`. A variable declared Complex that the equation takes
    the conjugate, real or imaginary part of is two real unknowns: `(1 + #i)*z + (2 -
    #i)*CONJ(z) = -3*#i` is solved by `z = 2 + 3*#i`. This applies only where each
    equation splits into two that aren't trivial, so `ABS(z - 1) = 2`, a circle, is
    not split.
    """
    if equations is None:
        if not isinstance(expression, sp.Equality):
            return None
        equations = (expression,)
    if not all(isinstance(equation, sp.Equality) for equation in equations):
        return None
    joined = Vector(*equations)
    if not variables:
        return None
    symbols = _symbols_of(joined)
    chosen = [symbols.get(name) for name in variables]
    if any(symbol is None for symbol in chosen):
        return None
    residuals = [_residual(equation) for equation in equations]
    declared = [context.domains.get(symbol.name) for symbol in chosen]
    kinds = {None if domain is None else domain.kind for domain in declared}
    replaced: dict[sp.Symbol, sp.Basic] = {}
    unknowns: list[sp.Symbol] = []
    if kinds == {DomainKind.REAL}:
        if len(chosen) <= len(equations) or not any(r.has(sp.I) for r in residuals):
            return None
        unknowns = list(chosen)
    elif kinds == {DomainKind.COMPLEX}:
        if len(chosen) != len(equations):
            return None
        if not any(r.has(sp.conjugate, sp.re, sp.im) for r in residuals):
            return None
        for symbol in chosen:
            real_part = sp.Dummy(f"{symbol.name}_re", real=True)
            imaginary_part = sp.Dummy(f"{symbol.name}_im", real=True)
            replaced[symbol] = real_part + sp.I * imaginary_part
            unknowns += [real_part, imaginary_part]
    else:
        return None
    parts = []
    try:
        for residual in residuals:
            written = sp.expand_complex(residual.xreplace(replaced))
            for part in written.as_real_imag():
                if part.has(sp.re, sp.im, sp.conjugate):
                    return None
                parts.append(sp.expand(part))
    except Exception:
        return None
    if any(part.is_zero for part in parts):
        return None
    solved = _linear(parts, tuple(unknowns), constants)
    if solved is None:
        split = _nonlinear(parts, tuple(unknowns))
        if split is None or split[1]:
            return None
        solved = split[0]
    if not solved:
        return None
    answers = []
    for mapping in solved:
        values = [
            replaced[symbol].xreplace(mapping)
            if replaced
            else mapping.get(symbol, symbol)
            for symbol in chosen
        ]
        if any(value.has(*unknowns) for value in values):
            return None
        relations = [
            _equated(symbol, sp.expand(value), context)
            for symbol, value in zip(chosen, values)
        ]
        single = len(relations) == 1 and equations == (expression,)
        answers.append(relations[0] if single else Vector(*relations))
    return tuple(answers)


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
    sides = (
        (expression.lhs, expression.rhs)
        if isinstance(expression, Relational)
        else (expression,)
    )
    if any(_infinite(side) for side in sides):
        found = _at_poles(sides, variable, context, constants, real)
        return (expression,) if found is None else found
    residual = _residual(expression)
    parted = None if bounds is not None else _parameter_factors(residual, variable)
    if parted is None:
        return _equation(residual, variable, context, bounds, divisors, constants, real)
    reduced, factors = parted
    found = _equation(reduced, variable, context, None, divisors, constants, real)
    if any(_is_identity(entry, variable, constants) for entry in found):
        return found
    conditions = [parameter_condition(factor, context) for factor in factors]
    return found + tuple(
        condition
        for condition in dict.fromkeys(conditions)
        if condition is not None and condition not in found
    )


def _infinite(side: sp.Basic) -> bool:
    """Whether an equation's side is infinite or undefined, wherever it is evaluated."""
    return side in (sp.oo, -sp.oo, sp.zoo, sp.nan) or isinstance(side, PlusMinus)


def _at_poles(
    sides: tuple[sp.Basic, ...],
    variable: sp.Symbol,
    context: Context,
    constants: _Constants,
    real: bool,
) -> tuple[sp.Basic, ...] | None:
    """An equation with the side `±inf` solved where the other side is `±inf`: at the
    zeros of its denominator, with tangents and secants written in sines and cosines,
    each checked to give the other side that value: `TAN(2*a) = 4*SQRT(3)/0` holds
    where `TAN(2*a)` is `±inf`, at `a = pi/4`. A polynomial is never `±inf`. None if
    this doesn't apply or a value doesn't check.
    """
    infinite = [side for side in sides if _infinite(side)]
    others = [side for side in sides if not _infinite(side)]
    if len(infinite) != 1 or len(others) != 1:
        return None
    if infinite[0] is not sp.zoo and not (
        isinstance(infinite[0], PlusMinus) and infinite[0].args[0] == sp.oo
    ):
        return None
    (other,) = others
    if not other.has(variable):
        return None
    try:
        denominator = sp.fraction(sp.together(TR2(TR1(other))))[1]
    except Exception:
        return None
    if not denominator.has(variable):
        # A polynomial has no poles; a logarithm is infinite at 0 without one.
        return () if other.is_polynomial(variable) else None
    found = _equation(denominator, variable, context, None, (), constants, real)
    if not found:
        return None
    for entry in found:
        if not (isinstance(entry, sp.Equality) and entry.lhs == variable):
            return None
        try:
            value = sp.simplify(other.xreplace({variable: entry.rhs}))
        except Exception:
            return None
        if value is not sp.zoo:
            return None
    return found


def _parameter_factors(
    residual: sp.Basic, variable: sp.Symbol
) -> tuple[sp.Basic, list[sp.Basic]] | None:
    """`residual` without the factors of its numerator free of `variable`, and those
    factors, or None if it has none but numbers.

    Each such factor is a condition on the parameters under which every value solves
    the equation: `Ieff^2*R*T = Imax^2*R*T/2` holds for any `Ieff` where `R = 0` or `T =
    0`. A power counts once, and the factors are those of the numerator over the
    rationals, so a product of parameters is split.
    """
    if not residual.has(variable) or not residual.free_symbols - {variable}:
        return None
    try:
        numerator, denominator = sp.fraction(sp.together(residual))
        coefficient, factors = sp.factor_list(numerator)
    except Exception:
        return None
    free = [
        factor
        for factor, _ in factors
        if not factor.has(variable) and not factor.is_number
    ]
    if not free:
        return None
    kept = sp.Mul(
        *(
            factor**power
            for factor, power in factors
            if factor.has(variable) or factor.is_number
        )
    )
    return coefficient * kept / denominator, free


#: Functions without zeros.
_NEVER_ZERO = (sp.factorial, sp.gamma, sp.exp)


def _never_zero(value: sp.Basic) -> bool:
    """Whether `value` is a product of powers of functions without zeros and nonzero
    numbers: `n!/(k!*(n - k)!)`."""
    if isinstance(value, _NEVER_ZERO):
        return True
    if value.is_number:
        return value.is_zero is False
    if isinstance(value, sp.Mul):
        return all(_never_zero(factor) for factor in value.args)
    if isinstance(value, sp.Pow):
        return _never_zero(value.base)
    return False


def parameter_condition(factor: sp.Basic, context: Context) -> sp.Basic | None:
    """`factor = 0` as a condition on the parameters, solved for the parameter where it
    is the only one and the factor is linear in it, or None where it provably fails: a
    variable declared positive is never 0, nor is a factorial, a gamma function or an
    exponential."""
    if factor.is_zero is False or _never_zero(factor):
        return None
    parameters = list(factor.free_symbols)
    if len(parameters) == 1:
        polynomial = _polynomial(factor, parameters[0])
        if polynomial is not None and polynomial.degree() == 1:
            slope, offset = polynomial.all_coeffs()
            return _equated(parameters[0], -offset / slope, context)
    return _relation(sp.Eq, normal_form(factor, context.order), sp.Integer(0))


def _is_identity(entry: sp.Basic, variable: sp.Symbol, constants: _Constants) -> bool:
    """Whether `entry` says every value of `variable` is a solution."""
    return (
        isinstance(entry, sp.Equality)
        and entry.lhs == variable
        and entry.rhs in constants.minted
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
    if residual.free_symbols and not residual.has(variable):
        # Whether it holds depends on the other variables only, so it is the condition
        # on them: `SOLVE(v = 2*u + 1, y)` isn't "no solutions".
        return _unsolved(residual, context)
    if _stepped(residual, variable):
        congruent = _congruence(residual, variable)
        if congruent is None:
            return _unsolved(residual, context)
        return _answers(variable, congruent, context)
    if not real and variable.is_extended_real is not True:
        circle = _on_circle(residual, variable)
        if circle is not None:
            radius = circle
            if radius.is_extended_negative:
                return ()
            if radius.is_extended_nonnegative:
                return (_equated(variable, radius * UNIT_CIRCLE, context),)
    found = None
    if context.branch is Branch.REAL:
        found = _on_real_branch(residual, variable)
    # Solutions on the real branch hold by construction; substituting them would take
    # the principal root.
    checked = found is None
    if found is None:
        found = _values(residual, variable)
    if found is None:
        split = _by_factors(residual, variable, context, divisors, constants, real)
        if split is not None:
            return split
    if (found is None or found.has(sp.ConditionSet)) and context.branch is Branch.ANY:
        found = _inverted_whatever(residual, variable) or found
    if found is None or found.has(sp.ConditionSet):
        split = _by_product(residual, variable, context, divisors, constants, real)
        if split is not None:
            return split
    if found is not None:
        polynomial = _polynomial(residual, variable) is not None

        def admissible(value: sp.Basic) -> bool:
            if _divides_by_zero(divisors, variable, value):
                return False
            if real and _imaginary(value):
                return False
            return polynomial or not checked or not _fails(residual, variable, value)

        if not polynomial and not isinstance(found, sp.FiniteSet) and residual.has(
            *_ARC_RANGES
        ):
            # The representatives of a family are guarded like a finite set: an
            # inverse trigonometric function takes back only angles in its range.
            members = None if _is_everything(found) else _members(found)
            if members is not None:
                found = sp.FiniteSet(*members)
        if (
            not polynomial
            and not isinstance(found, sp.FiniteSet)
            and not found.has(sp.ImageSet, sp.ConditionSet)
        ):
            # Finitely many candidates, some taken out where the equation is
            # undefined, are guarded each.
            members = _members(found)
            if members is not None:
                found = sp.FiniteSet(*members)
        if not polynomial and isinstance(found, sp.FiniteSet):
            guarded = [
                # A condition is already its guard.
                value
                if isinstance(value, sp.Piecewise)
                else _guarded(residual, variable, value)
                for value in found.args
            ]
            found = sp.FiniteSet(
                *_points_merged([value for value in guarded if value is not None])
            )
        entries = _entries(found, variable, context, constants, admissible)
        if entries is not None and not any(_unnamed(entry) for entry in entries):
            return entries
    if context.precision is not Precision.EXACT:
        searched = _searched(residual, variable, context, real)
        if searched is not None:
            return searched
    return _unsolved(residual, context)



def _points_merged(values: list[sp.Basic]) -> list[sp.Basic]:
    """`values` without each candidate that holds only at isolated values of a parameter
    where another candidate holds with the same value, and with the value of each other
    such candidate written at its points.

    `x = v^(3/4)` offers `x^(4/3)` where `x >= 0`, and the cube roots of unity times it
    only where `x = 0`, where all are 0: the answer is the first alone. Both that the
    other candidate holds at the point and that the values there are equal are proved
    exactly; a candidate with a value different from all the others' is a solution of
    its own there, and stays.
    """
    kept = list(values)
    for value in values:
        points = _isolated_points(value)
        if points is None:
            continue
        (expression, condition), = value.args
        others = [other for other in kept if other is not value]
        if all(_given_by(others, expression, point) for point in points):
            kept.remove(value)
            continue
        if len(points) == 1:
            (point,) = points
            at = expression.xreplace(point)
            if not at.has(sp.nan, sp.zoo):
                kept[kept.index(value)] = sp.Piecewise((sp.simplify(at), condition))
    return kept


def _isolated_points(value: sp.Basic) -> list[dict[sp.Symbol, sp.Basic]] | None:
    """The values of the one real parameter where the candidate `value` holds, if it is
    `IF(p(x) = 0, ...)` for a polynomial `p` with exactly known roots; None otherwise."""
    if not isinstance(value, sp.Piecewise) or len(value.args) != 1:
        return None
    (_, condition), = value.args
    if not isinstance(condition, sp.Eq):
        return None
    difference = condition.lhs - condition.rhs
    if len(difference.free_symbols) != 1:
        return None
    (parameter,) = difference.free_symbols
    if parameter.is_real is not True:
        return None
    polynomial = _polynomial(difference, parameter)
    if polynomial is None or polynomial.degree() < 1:
        return None
    try:
        roots = sp.roots(polynomial)
    except Exception:
        return None
    if sum(roots.values()) != polynomial.degree():
        return None
    if any(root.is_real is None for root in roots):
        return None
    return [{parameter: root} for root in roots if root.is_real]


def _given_by(
    others: list[sp.Basic], expression: sp.Basic, point: dict[sp.Symbol, sp.Basic]
) -> bool:
    """Whether one of the candidates `others` is proved to hold at `point` with the value
    `expression` has there."""
    try:
        at = expression.xreplace(point)
        if at.has(sp.nan, sp.zoo):
            return False
        for other in others:
            if isinstance(other, sp.Piecewise):
                if len(other.args) != 1:
                    continue
                (other, condition), = other.args
                if condition.xreplace(point) is not sp.true:
                    continue
            there = other.xreplace(point)
            if there.has(sp.nan, sp.zoo):
                continue
            if sp.simplify(at - there) == 0:
                return True
    except Exception:
        return False
    return False

def _on_circle(residual: sp.Basic, variable: sp.Symbol) -> sp.Basic | None:
    """The radius `c` if `residual = 0` says `ABS(z) = c` of the complex `variable` z,
    with `c` free of it; None otherwise. Its solutions are the circle
    `z = c*unit_circle`."""
    modulus = sp.Abs(variable)
    if not isinstance(modulus, sp.Abs):
        return None
    marker = sp.Dummy()
    linear = residual.subs(modulus, marker)
    if linear.has(variable):
        return None
    try:
        polynomial = sp.Poly(linear, marker)
    except sp.PolynomialError:
        return None
    if polynomial.degree() != 1:
        return None
    slope, offset = polynomial.all_coeffs()
    return sp.simplify(-offset / slope)


def _on_real_branch(residual: sp.Basic, variable: sp.Symbol) -> sp.Set | None:
    """The real solutions of an equation in odd roots of the variable, where Branch Real
    takes the real root of a negative number, or None if it isn't one.

    With `L` the least common multiple of the roots' orders, `x = u^L` for real `u`
    makes every root of `x` a power of `u`, so the equation is a polynomial in `u`:
    `x^(2/3) - 2*x^(1/3) - 35` is `u^2 - 2*u - 35`, whose roots 7 and -5 give 343 and
    -125.
    """
    powers = [
        part
        for part in residual.atoms(sp.Pow)
        if part.base == variable and part.exp.is_Rational and not part.exp.is_Integer
    ]
    if not powers or any(part.exp.q % 2 == 0 for part in powers):
        return None
    order = math.lcm(*(int(part.exp.q) for part in powers))
    root = sp.Dummy("u", real=True)
    written = residual.xreplace(
        {part: root ** int(part.exp * order) for part in powers}
    ).xreplace({variable: root**order})
    if written.has(variable):
        return None
    polynomial = _polynomial(sp.fraction(sp.together(written))[0], root)
    if polynomial is None or polynomial.degree() < 1:
        return None
    found = _values(polynomial.as_expr(), root)
    if not isinstance(found, sp.FiniteSet):
        return None
    values = [value for value in found.args if value.is_extended_real]
    if len(values) != sum(1 for value in found.args if not _imaginary(value)):
        return None
    return sp.FiniteSet(*(value**order for value in values))


#: Functions constant on intervals of their argument.
_STEPS = (sp.floor, sp.ceiling, Modulo)


def _stepped(residual: sp.Basic, variable: sp.Symbol) -> bool:
    """Whether the variable is inside `FLOOR`, `CEILING`, `ROUND`, `MOD` or `MODS`.

    Such an equation holds on infinitely many intervals or points, of which a solver
    finds one at best (`MOD(2*x + 3, 5) = 4` at `x = 1/2`), so it is left unsolved.
    """
    return any(part.has(variable) for part in residual.atoms(*_STEPS))


def _congruence(residual: sp.Basic, variable: sp.Symbol) -> list[sp.Basic] | None:
    """The solutions Derive gives of `MOD(a*x + b, m) = r` (and of what reduces to it,
    `c*MOD(...) + d = 0`), with `m` a positive number and `a`, `b`, `r` free of `x`:
    `a*x + b` is `r`, `r + m` or `r - m`, after `b` is reduced modulo `m` when it is a
    number. `SOLUTIONS(MOD(8*77 + z, 31) + 65 = 65, z)` is `[4, -27, -58]`. The
    equation holds on infinitely many points; these are the representatives Derive
    lists, each verified. An empty list if `r` is outside `[0, m)`; None for any other
    equation in a step function, which is left unsolved."""
    calls = [call for call in residual.atoms(*_STEPS) if call.has(variable)]
    if len(calls) != 1 or not isinstance(calls[0], Modulo):
        return None
    (call,) = calls
    inside, modulus = call.args
    if modulus.has(variable) or not (modulus.is_number and modulus.is_positive):
        return None
    marker = sp.Dummy("m")
    replaced = residual.xreplace({call: marker})
    if replaced.has(variable):
        return None
    try:
        linear = sp.Poly(replaced, marker)
        polynomial = sp.Poly(inside, variable)
    except sp.PolynomialError:
        return None
    if linear.degree() != 1 or polynomial.degree() != 1:
        return None
    slope, intercept = polynomial.all_coeffs()
    scale, rest = linear.all_coeffs()
    remainder = -rest / scale
    if not remainder.is_number or slope.has(variable) or intercept.has(variable):
        return None
    if remainder.is_negative or (remainder - modulus).is_nonnegative:
        return []
    if not (remainder.is_nonnegative and (remainder - modulus).is_negative):
        return None
    if intercept.is_Rational and modulus.is_Rational:
        intercept = intercept - modulus * sp.floor(intercept / modulus)
    values = [
        sp.simplify((remainder + shift - intercept) / slope)
        for shift in (sp.S.Zero, modulus, -modulus)
    ]
    return [value for value in values if _satisfies(residual, variable, value)]


def _by_factors(
    residual: sp.Basic,
    variable: sp.Symbol,
    context: Context,
    divisors: tuple[sp.Basic, ...],
    constants: _Constants,
    real: bool,
) -> tuple[sp.Basic, ...] | None:
    """A polynomial whose roots aren't all found, solved factor by factor, or None if
    it doesn't factor.

    A factor without found roots stays as its equation among the solutions of the
    others: `SOLVE((z - 1)*(z^5 + z + 3) = 0, z)` is `[z = 1, z^5 + z + 3 = 0]`.
    """
    if _polynomial(residual, variable) is None:
        return None
    try:
        _, factors = sp.factor_list(residual)
    except Exception:
        return None
    parts = [factor for factor, _ in factors if factor.has(variable)]
    if len(parts) < 2:
        return None
    answers: list[sp.Basic] = []
    for part in parts:
        for answer in _equation(part, variable, context, None, divisors, constants, real):
            if answer not in answers:
                answers.append(answer)
    return tuple(answers)


def _by_product(
    residual: sp.Basic,
    variable: sp.Symbol,
    context: Context,
    divisors: tuple[sp.Basic, ...],
    constants: _Constants,
    real: bool,
) -> tuple[sp.Basic, ...] | None:
    """A product with a power of the variable to a symbolic exponent, solved factor by
    factor, or None if it isn't one.

    Whether such a power vanishes depends on the sign of its exponent, so it stays as
    its equation among the solutions of the other factors: the derivative of
    `t^k*(1 - t)^(n - k)` is zero where `k = n*t`, and where `t^(k - 1) = 0` or `(1 -
    t)^(n - k - 1) = 0`, solved for k as well as for t.
    """
    try:
        numerator = sp.fraction(sp.together(residual))[0]
    except Exception:
        return None
    factors = [factor for factor in sp.Mul.make_args(numerator) if factor.has(variable)]
    symbolic = [
        factor
        for factor in factors
        if factor.is_Pow
        and (
            not factor.exp.has(variable)
            and not factor.exp.is_number
            and factor.base.has(variable)
            # A power of a parameter to an exponent in the variable, `t^(k - 1)` solved
            # for k, vanishes only where its base does, and only for some exponents.
            or not factor.base.has(variable)
            and not factor.base.is_number
        )
    ]
    if len(factors) < 2 or not symbolic:
        return None
    answers: list[sp.Basic] = []
    for factor in factors:
        if factor in symbolic:
            found = _unsolved(factor, context)
        else:
            found = _equation(factor, variable, context, None, divisors, constants, real)
        if any(_is_identity(entry, variable, constants) for entry in found):
            return found
        for answer in found:
            if answer not in answers:
                answers.append(answer)
    return tuple(answers)


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
    if _roots_unwritten_in_root(residual, variable):
        return None
    standing = {
        part: sp.Dummy(nonnegative=True)
        for part in residual.atoms(sp.Abs)
        if not part.has(variable)
    }
    if standing:
        # Solveset refuses any `ABS` over the complexes, even of a parameter, which is
        # just a nonnegative constant to it.
        found = _values(residual.xreplace(standing), variable)
        if found is None:
            return None
        return found.xreplace({dummy: part for part, dummy in standing.items()})
    try:
        found = sp.solveset(residual, variable, sp.S.Complexes)
    except Exception:
        return None
    if found.has(sp.ConditionSet):
        found = _without_impossible_poles(found)
    if found.has(sp.ConditionSet):
        found = _candidates_checked(found, residual, variable)
    if found.has(sp.ConditionSet):
        found = _through_kernels(found, variable)
    if found.has(sp.ConditionSet):
        rewritten = _exponential(residual, variable)
        if rewritten is not None:
            found = rewritten
    if isinstance(found, sp.ConditionSet):
        return _between_arcs(residual, variable) or _inverted(residual, variable)
    return found


def _without_impossible_poles(found: sp.Set) -> sp.Set:
    """`found` with each set of the zeros of a function that has none taken as empty:
    solveset takes out of `(k - n*t)/k!` the values where `k! = 0`, which none is."""

    def empty(conditions: sp.ConditionSet) -> sp.Basic:
        condition = conditions.condition
        if (
            isinstance(condition, sp.Eq)
            and condition.rhs == 0
            and _never_zero(condition.lhs)
        ):
            return sp.S.EmptySet
        return conditions

    try:
        return found.replace(lambda e: isinstance(e, sp.ConditionSet), empty)
    except Exception:
        return found


def _roots_unwritten_in_root(residual: sp.Basic, variable: sp.Symbol) -> bool:
    """Whether `residual` is a polynomial in a root of `variable` with roots that have
    no radical form.

    Solveset would name them by their polynomial and then refine each numerically,
    which for `7*p^(17/10) - 225000*p^(2/5) - 2625000` takes minutes; the equation is
    left unsolved instead, as a polynomial with such roots is.
    """
    exponents = {
        part.exp
        for part in residual.atoms(sp.Pow)
        if part.base == variable and part.exp.is_Rational
    }
    if not exponents or all(exponent.is_Integer for exponent in exponents):
        return False
    root = sp.Dummy()
    order = math.lcm(*(int(exponent.q) for exponent in exponents))
    expanded = sp.expand(sp.fraction(sp.together(residual))[0])
    powers = {
        part: root ** (part.exp * order)
        for part in expanded.atoms(sp.Pow)
        if part.base == variable
    }
    written = expanded.xreplace(powers).xreplace({variable: root**order})
    try:
        polynomial = sp.Poly(written, root)
        if polynomial.degree() < 5 or polynomial.free_symbols - {root}:
            return False
        found = sp.roots(polynomial)
    except Exception:
        return False
    return sum(found.values()) != polynomial.degree()


def _candidates_checked(
    found: sp.Set, residual: sp.Basic, variable: sp.Symbol
) -> sp.Set:
    """`found` with each condition set over finitely many candidates decided, where
    each candidate is proved a solution or not one by substituting it.

    For `x^(2/3) - 2*x^(1/3) - 35 = 0` solveset offers -125 under a condition, from
    the quadratic in `x^(1/3)`, which the principal cube root of -125 doesn't meet.
    A candidate undecided either way leaves the condition set as it is.
    """

    def decided(part: sp.Set) -> sp.Set:
        if isinstance(part, sp.Complement):
            # The values where the equation is undefined, taken out of the candidates.
            return sp.Complement(decided(part.args[0]), part.args[1])
        if not isinstance(part, sp.ConditionSet):
            return part
        if not isinstance(part.base_set, sp.FiniteSet):
            return part
        kept = []
        for value in part.base_set.args:
            vanishes = _vanishes_at(residual, variable, value)
            if vanishes is None:
                signed = _sign_guard(residual, variable, sp.factor(value))
                if signed is sp.S.EmptySet:
                    continue
                if signed is not None:
                    kept.append(signed)
                    continue
                # A candidate that is a solution for some values of a single parameter
                # is kept under that condition: `SQRT(v) + t = 24` gives `(t - 24)^2`
                # where `t <= 24`.
                guarded = _guarded(residual, variable, sp.factor(value), signs=False)
                if isinstance(guarded, sp.Piecewise):
                    kept.append(guarded)
                    continue
                return part
            if vanishes:
                kept.append(value)
        return sp.FiniteSet(*kept)

    if isinstance(found, sp.Union):
        return sp.Union(*(decided(part) for part in found.args))
    return decided(found)


def _through_kernels(found: sp.Set, variable: sp.Symbol) -> sp.Set:
    """`found` with each condition set over the complexes solved through a kernel, where
    its equation is a polynomial in one function of the variable, as `(SIN(x)^2 + 1)^2`
    is in `SIN(x)`: the polynomial's roots `r` are found, then `SIN(x) = r` is solved
    for each. A condition set not solved that way is left as it is.
    """

    def solved(part: sp.Set) -> sp.Set:
        if not isinstance(part, sp.ConditionSet) or part.base_set != sp.S.Complexes:
            return part
        condition = part.condition
        if not isinstance(condition, sp.Equality):
            return part
        residual = (condition.lhs - condition.rhs).xreplace({part.sym: variable})
        return _kernel_solved(residual, variable) or part

    if isinstance(found, sp.Union):
        return sp.Union(*(solved(part) for part in found.args))
    return solved(found)


def _kernel_solved(residual: sp.Basic, variable: sp.Symbol) -> sp.Set | None:
    """The solutions of `residual = 0` as a polynomial in a function of `variable`, as in
    `_through_kernels`, or None."""
    kernels = {part for part in residual.atoms(sp.Function) if part.has(variable)}
    unknown = sp.Dummy("u")
    for kernel in sorted(kernels, key=sp.count_ops):
        written = residual.xreplace({kernel: unknown})
        if written.has(variable):
            continue
        polynomial = _polynomial(written, unknown)
        if polynomial is None or polynomial.degree() < 1:
            continue
        values = _values(written, unknown)
        if not isinstance(values, sp.FiniteSet):
            continue
        pieces = []
        for value in values.args:
            try:
                piece = sp.solveset(kernel - value, variable, sp.S.Complexes)
            except Exception:
                return None
            if piece.has(sp.ConditionSet):
                return None
            pieces.append(piece)
        return sp.Union(*pieces)
    return None


#: The inverse trigonometric functions an equation between two of them may use, with
#: the range of each over nonnegative and over negative real arguments.
_ARCS = {
    sp.asin: ((0, sp.pi / 2), (-sp.pi / 2, 0)),
    sp.atan: ((0, sp.pi / 2), (-sp.pi / 2, 0)),
    sp.acot: ((0, sp.pi / 2), (-sp.pi / 2, 0)),
    sp.acos: ((0, sp.pi / 2), (sp.pi / 2, sp.pi)),
}


def _between_arcs(residual: sp.Basic, variable: sp.Symbol) -> sp.Set | None:
    """The real solutions of an equation between two inverse trigonometric functions of
    the variable, like `ASIN(x) = ATAN(x)`, or None.

    Taking the tangent of both sides leaves an algebraic equation, whose real roots
    are the candidates. Equal tangents make the two sides differ by a multiple of
    `pi`, and a candidate is a solution if that multiple is zero: the ranges of the two
    functions, read off the signs of their arguments, are within `pi` of each other.
    A candidate that can't be decided either way leaves the equation unsolved.
    """
    terms = sp.Add.make_args(residual)
    if len(terms) != 2:
        return None
    sides = []
    for term in terms:
        coefficient, arc = term.as_coeff_Mul()
        if coefficient not in (1, -1) or type(arc) not in _ARCS:
            return None
        sides.append(arc)
    if terms[0].as_coeff_Mul()[0] == terms[1].as_coeff_Mul()[0]:
        return None
    left, right = sides
    if left.args[0].has(*_ARCS) or right.args[0].has(*_ARCS):
        return None
    try:
        tangents = sp.fraction(sp.together(sp.tan(left) - sp.tan(right)))[0]
        candidates = sp.solveset(tangents, variable, sp.S.Reals)
    except Exception:
        return None
    if not isinstance(candidates, sp.FiniteSet):
        return None
    kept = []
    for value in candidates.args:
        holds = _arcs_agree(left, right, variable, value)
        if holds is None:
            return None
        if holds:
            kept.append(value)
    return sp.FiniteSet(*kept)


def _arcs_agree(
    left: sp.Basic, right: sp.Basic, variable: sp.Symbol, value: sp.Basic
) -> bool | None:
    """Whether two inverse trigonometric functions are equal at `value`, a real root of
    the equation between their tangents, as in `_between_arcs`, or None if undecided."""
    if _satisfies(left - right, variable, value):
        return True
    if _fails(left - right, variable, value):
        return False
    ranges = []
    for arc in (left, right):
        argument = sp.simplify(arc.args[0].subs(variable, value))
        if argument.is_extended_real is not True:
            return None
        if type(arc) in (sp.asin, sp.acos) and (sp.Abs(argument) - 1).is_positive:
            return None
        sign = argument.is_nonnegative
        if sign is None:
            return None
        ranges.append(_ARCS[type(arc)][0 if sign else 1])
        tangent = sp.simplify(sp.tan(arc.func(argument)))
        if tangent.is_finite is not True:
            return None
    tangents = sp.tan(left) - sp.tan(right)
    if sp.simplify(tangents.subs(variable, value)).is_zero is not True:
        return None
    (low, high), (other_low, other_high) = ranges
    # The difference lies in `[low - other_high, high - other_low]`; a multiple of `pi`
    # in there other than zero would need a side at an end where its tangent is
    # infinite.
    if high - other_low <= sp.pi and other_high - low <= sp.pi:
        return True
    return None


def _vanishes_at(
    residual: sp.Basic, variable: sp.Symbol, value: sp.Basic
) -> bool | None:
    """Whether `residual` is proved zero (True) or nonzero (False) at `value`, or None.

    A number is also split into its real and imaginary parts, which decides sums of
    roots of unity: at -125, `x^(2/3) - 2*x^(1/3) - 35` is `-105/2 + 15*SQRT(3)*#i/2`.
    """
    if _satisfies(residual, variable, value):
        return True
    if _fails(residual, variable, value):
        return False
    try:
        left = residual.xreplace({variable: value})
        if left.free_symbols:
            return None
        return sp.expand_complex(left).is_zero
    except Exception:
        return None


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
        found = None
    if found is None or found.has(sp.ConditionSet, sp.Intersection):
        return _sign_by_sign(residual, variable) or None
    return found


def _sign_by_sign(residual: sp.Basic, variable: sp.Symbol) -> sp.Set | None:
    """The real solutions of an equation in the `ABS` of rational polynomials, found
    between consecutive real roots of the arguments, or None.

    `solveset` resolves one `ABS` at a time and gives up on `MAX(2 - x^2, x) =
    MAX(-x, x^3/9)`, which has three. Between two roots every argument has a fixed sign,
    read at a rational point, so the equation there is free of `ABS`; a solution of it
    counts if it is real and provably inside. The roots themselves are substituted.
    """
    absolutes = [part for part in residual.atoms(sp.Abs) if part.has(variable)]
    product = sp.S.One
    for part in absolutes:
        polynomial = _polynomial(part.args[0], variable)
        if polynomial is None or polynomial.domain not in (sp.ZZ, sp.QQ):
            return None
        product *= part.args[0]
    try:
        polynomial = sp.Poly(product, variable).sqf_part()
        roots = polynomial.real_roots()
        isolating = [interval for interval, _ in polynomial.intervals()]
    except Exception:
        return None
    if roots:
        samples = [sp.Rational(isolating[0][0]) - 1]
        samples += [
            (sp.Rational(isolating[i][1]) + sp.Rational(isolating[i + 1][0])) / 2
            for i in range(len(roots) - 1)
        ]
        samples.append(sp.Rational(isolating[-1][1]) + 1)
    else:
        samples = [sp.S.Zero]
    ends = [sp.S.NegativeInfinity, *roots, sp.S.Infinity]
    found: list[sp.Set] = []
    for index, sample in enumerate(samples):
        low, high = ends[index], ends[index + 1]
        signs = {}
        for part in absolutes:
            sign = sp.sign(part.args[0].subs(variable, sample))
            if sign == 0:
                return None
            signs[part] = sign * part.args[0]
        piece = sp.expand(residual.xreplace(signs))
        if piece.is_zero:
            found.append(sp.Interval.open(low, high))
            continue
        values = _values(piece, variable)
        if not isinstance(values, sp.FiniteSet):
            return None
        for value in values.args:
            if _imaginary(value):
                continue
            inside = [
                value.is_extended_real,
                low.is_infinite or (value - low).is_positive,
                high.is_infinite or (high - value).is_positive,
            ]
            if False in inside:
                continue
            if None in inside:
                return None
            found.append(sp.FiniteSet(value))
    for root in roots:
        vanishes = _vanishes_at(residual, variable, root)
        if vanishes is None:
            return None
        if vanishes:
            found.append(sp.FiniteSet(root))
    return sp.Union(*found)


def _exponential(residual: sp.Basic, variable: sp.Symbol) -> sp.Set | None:
    """The solutions of `residual` with powers of positive numbers or of parameters
    rewritten as powers of `#e`, or None if that doesn't help, since `solveset` inverts
    `#e^x = c` but not `3^x = c`. A power of a parameter `b^x` is `#e^(x*LN(b))` on the
    principal branch, which is what makes `a*b^x = c*d^x` solvable.
    """

    def rewritable(part: sp.Basic) -> bool:
        return (
            part.is_Pow
            and part.base is not sp.E
            and not part.base.has(variable)
            and (part.base.is_positive is True or not part.base.is_number)
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
    found = _candidates_checked(found, rewritten, variable)
    return None if found.has(sp.ConditionSet) else found


def _binomial_roots(polynomial: sp.Poly) -> dict[sp.Basic, int] | None:
    """The roots of `a*x^n + b` for real `-b/a`, as one real root times each root of
    unity, or None; also of `a*(x + c)^n + b` for rational `c`.

    `roots` distributes the real root into every term, so `x^100 = 12` would repeat the
    hundredth root of 12 three times per solution.
    """
    degree = polynomial.degree()
    terms = polynomial.terms()
    if degree < 2:
        return None
    if len(terms) > 2 and degree > 4:
        # `(x + c)^n = d` multiplied out: shifted by `c`, it is binomial again, and
        # `roots` would take long over its many terms. Up to quartics `roots` has
        # formulas, which write the roots more simply.
        coefficients = polynomial.all_coeffs()
        shift = coefficients[1] / (degree * coefficients[0])
        if not shift.is_Rational:
            return None
        try:
            shifted = polynomial.to_field().shift(-shift)
        except Exception:
            return None
        if len(shifted.terms()) != 2:
            return None
        found = _binomial_roots(shifted)
        if found is None:
            return None
        return {root - shift: count for root, count in found.items()}
    if len(terms) != 2 or terms[1][0] != (0,):
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
        if left.has(sp.exp) and _exponentials_cancel(left):
            return True
        if left.free_symbols or not left.has(sp.LambertW):
            return False
        return sp.simplify(_unwound(left)).is_zero is True
    except Exception:
        return False


def _exponentials_cancel(left: sp.Basic) -> bool:
    """Whether a sum of exponentials is proved zero by dividing it by its first term and
    combining each quotient of exponentials into one, with its exponent cancelled: at
    `x = (LN(c) - LN(a))/(LN(b) - LN(d))`, `a*#e^(x*LN(b)) - c*#e^(x*LN(d))` is `a`
    times `1 - c/a*#e^(LN(c) - LN(a))`.
    """
    terms = sp.Add.make_args(sp.expand(left))
    if len(terms) < 2:
        return False
    ratio = sp.powsimp(sp.expand(left / terms[0]), combine="exp")
    ratio = ratio.replace(
        lambda e: isinstance(e, sp.exp),
        lambda e: sp.exp(sp.cancel(sp.together(e.args[0]))),
    )
    return sp.simplify(ratio).is_zero is True


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


def _guarded(
    residual: sp.Basic, variable: sp.Symbol, value: sp.Basic, signs: bool = True
) -> sp.Basic | None:
    """A candidate of an equation with a root of the variable, under the condition on
    its parameter for which it is a solution, or None if it is a solution for none.

    Squaring away a root can bring in a value that solves the equation only for some
    values of a parameter: `SQRT(x) = a` gives `a^2`, which is a solution where `a >=
    0`, so the answer is `IF(a >= 0, a^2)`. The condition is found by splitting on the
    signs the roots of the substituted equation depend on (`_sign_guard`, unless
    `signs` is off because the caller tried it), or by solving the substituted equation
    for a single real parameter, exactly; anything else leaves the candidate as it is.
    """
    arc = _arc_guard(residual, variable, value)
    if arc is not None:
        return None if arc is sp.S.EmptySet else arc
    radical = any(
        power.has(variable) and not power.exp.is_integer
        for power in residual.atoms(sp.Pow)
    )
    parameters = value.free_symbols
    if radical and len(parameters) > 1:
        isolated = _isolated_root_guard(residual, variable, value)
        if isolated is not None:
            return isolated
    if radical and signs:
        signed = _sign_guard(residual, variable, value)
        if signed is not None:
            return None if signed is sp.S.EmptySet else signed
    if not radical or len(parameters) != 1:
        return value
    (parameter,) = parameters
    if parameter.is_real is not True:
        return value
    try:
        check = sp.simplify(residual.xreplace({variable: value}))
        if check.is_zero:
            return value
        holds = sp.solveset(check, parameter, sp.S.Reals)
        if holds.has(sp.ConditionSet):
            return value
        if holds is sp.S.EmptySet:
            return None
        if holds == sp.S.Reals:
            return value
        condition = holds.as_relational(parameter)
    except Exception:
        return value
    return sp.Piecewise((value, condition))


#: The inverse trigonometric functions whose inversion `_arc_guard` guards: the forward
#: function, and the range as the bounds of a relation `low < w < high` (`<=` where the
#: flag says the end is included).
_ARC_RANGES = {
    sp.atan: (sp.tan, -sp.pi / 2, sp.pi / 2, False),
    sp.asin: (sp.sin, -sp.pi / 2, sp.pi / 2, True),
    sp.acos: (sp.cos, sp.S.Zero, sp.pi, True),
}


def _arc_guard(
    residual: sp.Basic, variable: sp.Symbol, value: sp.Basic
) -> sp.Basic | None:
    """A candidate of an equation with one inverse trigonometric function of the
    variable, under the condition that the inverse takes the angle back, the empty set
    if it does for no value, or None if the equation is not of that form.

    Inverting `2*ATAN(c) = x0` gives `c = TAN(x0/2)`, but `ATAN(TAN(x0/2))` is `x0/2`
    only where `-pi/2 < x0/2 < pi/2`, so the answer is `IF(-pi < x0 < pi, TAN(x0/2))`.
    The candidate must solve the equation once the inverse is replaced by that angle,
    which is proved by substitution; the condition is stated for real parameters only.
    """
    arcs = [arc for arc in residual.atoms(*_ARC_RANGES) if arc.has(variable)]
    if len(arcs) != 1 or not value.free_symbols:
        return None
    if any(symbol.is_real is not True for symbol in value.free_symbols):
        return None
    (arc,) = arcs
    forward, low, high, closed = _ARC_RANGES[arc.func]
    try:
        # A family's representative may come with its logarithms in parts.
        value = polar_logarithms(value)
        inner = sp.simplify(arc.args[0].xreplace({variable: value}))
        if not isinstance(inner, forward):
            return None
        (angle,) = inner.args
        if angle.has(variable):
            return None
        unknown = sp.Dummy("arc")
        rest = residual.xreplace({arc: unknown})
        if rest.has(variable) and sp.simplify(
            rest.xreplace({variable: value, unknown: angle})
        ) != 0:
            return None
        if not rest.has(variable) and sp.simplify(rest.xreplace({unknown: angle})) != 0:
            return None
        below = sp.Le if closed else sp.Lt
        condition = sp.And(below(low, angle), below(angle, high))
        if condition is sp.true:
            return value
        if condition is sp.false:
            return sp.S.EmptySet
        parameters = angle.free_symbols
        if len(parameters) == 1:
            (parameter,) = parameters
            holds = sp.Intersection(
                *(
                    sp.solveset(relation, parameter, sp.S.Reals)
                    for relation in condition.args
                )
            )
            if holds.has(sp.ConditionSet):
                return None
            if holds == sp.S.Reals:
                return value
            if holds is sp.S.EmptySet:
                return sp.S.EmptySet
            condition = holds.as_relational(parameter)
    except Exception:
        return None
    return sp.Piecewise((value, condition))


def _isolated_root_guard(
    residual: sp.Basic, variable: sp.Symbol, value: sp.Basic
) -> sp.Basic | None:
    """A candidate of `c*SQRT(P) + R = 0`, one square root of the variable alone in
    it, under the condition that makes it a solution, or None if the equation is not
    of that form or the candidate doesn't solve its square.

    The candidate solves `P = Q^2` for `Q = -R/c`, proved by substitution; for real
    parameters `SQRT(Q^2)` is `Q` exactly where `Q >= 0`, so the candidate is a solution
    there: `SQRT(a^2*k^2 + b^2) + k*x = y` gives each root of the square where `y - k*x
    >= 0`. Any parameter declared other than real leaves it to the caller.
    """
    if any(symbol.is_real is not True for symbol in value.free_symbols):
        return None
    terms = sp.Add.make_args(sp.expand(residual))
    rooted = [
        term
        for term in terms
        if any(
            power.exp == sp.S.Half and power.base.has(variable)
            for power in sp.Mul.make_args(term)
            if isinstance(power, sp.Pow)
        )
    ]
    roots = {
        power
        for term in rooted
        for power in sp.Mul.make_args(term)
        if isinstance(power, sp.Pow)
        and power.exp == sp.S.Half
        and power.base.has(variable)
    }
    if len(roots) != 1:
        return None
    (root,) = roots
    coefficient = sp.Add(*rooted) / root
    rest = sp.Add(*(term for term in terms if term not in rooted))
    if coefficient.has(variable) or rest.has(root) or coefficient.is_zero is not False:
        return None
    other = -rest / coefficient
    try:
        squared = sp.simplify(
            (root.base - other**2).xreplace({variable: value})
        )
        if squared != 0:
            return None
        side = sp.factor(sp.together(other.xreplace({variable: value})))
        if side.is_nonnegative is True:
            return value
        if side.is_negative is True:
            return None
        return sp.Piecewise((value, sp.Ge(side, 0, evaluate=False)))
    except Exception:
        return None


#: How many `ABS` deep `_sign_guard` splits before giving up.
_SIGN_SPLITS = 3


def _sign_guard(
    residual: sp.Basic, variable: sp.Symbol, value: sp.Basic
) -> sp.Basic | None:
    """A candidate decided by substitution, split on the signs of `q` where that
    leaves the absolute value of a parameter expression `q`: the candidate itself if
    it is a solution for every sign, the empty set if for none, the candidate under
    the condition on the sign of `q` otherwise, or None if undecided.

    A root of a perfect power of real parameters is an absolute value, so it is the
    sign of the expression under the root that decides whether a candidate of an
    equation in a root of the variable is a solution: `LN(SQRT(v) - x) = k` offers
    `(#e^k + x)^2`, which leaves `LN(ABS(#e^k + x) - x) - k`; that is 0 where `#e^k + x
    >= 0` and nowhere else, so the answer is `IF(#e^k + x >= 0, (#e^k + x)^2)`. Each
    sign is decided exactly, by substituting for a parameter in which `q` is linear;
    an `ABS(q)` in the candidate is written without it under the condition.
    """
    symbols = value.free_symbols | (residual.free_symbols - {variable})
    if value.has(sp.Piecewise) or not value.free_symbols:
        return None
    if any(symbol.is_real is not True for symbol in symbols):
        return None
    try:
        left = residual.xreplace({variable: value})
        left = left.replace(
            lambda e: isinstance(e, sp.Pow)
            and not e.exp.is_integer
            and bool(e.base.free_symbols),
            lambda e: sp.factor(e.base) ** e.exp,
        )
        holds = _zero_by_signs(left, value, _SIGN_SPLITS, None)
    except Exception:
        return None
    if holds is None:
        return None
    if holds is True:
        return value
    if holds is False:
        return sp.S.EmptySet
    absolute, signs = holds
    (inside,) = absolute.args
    condition = {
        frozenset({1}): sp.Gt(inside, 0),
        frozenset({1, 0}): sp.Ge(inside, 0),
        frozenset({0}): sp.Eq(inside, 0),
        frozenset({0, -1}): sp.Le(inside, 0),
        frozenset({-1}): sp.Lt(inside, 0),
        frozenset({1, -1}): sp.Ne(inside, 0),
    }[signs]
    if signs <= {1, 0}:
        value = value.xreplace({absolute: inside})
    elif signs <= {0, -1}:
        value = value.xreplace({absolute: -inside})
    return sp.Piecewise((value, condition))


def _zero_by_signs(
    left: sp.Basic, value: sp.Basic, depth: int, region: sp.Dummy | None
) -> bool | tuple[sp.Abs, frozenset[int]] | None:
    """Whether `left`, the residual at the candidate `value`, is zero for all real
    values of its parameters (True) or for none (False), or the `ABS(q)` in it and the
    signs of `q` (1, 0, -1) for which it is, or None if undecided. `region` is the
    positive dummy standing for the size of an `ABS` split before; it is not split on.
    Where `q` is zero, a part of either that is undefined makes no solution, though the
    whole may evaluate to a number.
    """
    absolutes = sorted(left.atoms(sp.Abs), key=sp.default_sort_key)
    if not absolutes:
        return _zero_decided(left, region)
    if depth == 0:
        return None
    for absolute in absolutes:
        (inside,) = absolute.args
        for parameter in sorted(
            (s for s in inside.free_symbols if not isinstance(s, sp.Dummy)),
            key=sp.default_sort_key,
        ):
            polynomial = _polynomial(sp.expand(inside), parameter)
            if polynomial is None or polynomial.degree() != 1:
                continue
            slope, offset = polynomial.all_coeffs()
            if not slope.is_number or slope.is_zero is not False:
                continue
            break
        else:
            continue
        break
    else:
        return _zero_by_sides(left, absolutes[0]) if len(absolutes) == 1 else None
    found: dict[int, bool] = {}
    for sign in (1, 0, -1):
        size = sp.Dummy("size", positive=True) if sign else None
        point = {parameter: ((sign * size if size else 0) - offset) / slope}
        if not sign and _undefined_at(sp.Tuple(left, value), point):
            found[sign] = False
            continue
        replaced, placed = (
            part.xreplace(point).replace(
                lambda e: isinstance(e, sp.Abs),
                lambda e: sp.Abs(sp.expand(e.args[0])),
            )
            for part in (left, value)
        )
        decided = _zero_by_signs(replaced, placed, depth - 1, size)
        if not isinstance(decided, bool):
            return None
        found[sign] = decided
    signs = frozenset(sign for sign, holds in found.items() if holds)
    if len(signs) == 3:
        return True
    if not signs:
        return False
    return absolute, signs


def _undefined_at(expression: sp.Basic, point: dict[sp.Symbol, sp.Basic]) -> bool:
    """Whether a part of `expression` is undefined at `point`: `x - 2/v^(3/2)` with
    `v` infinite is `x` to sympy."""
    return any(
        part.xreplace(point).has(sp.nan, sp.zoo)
        for part in sp.preorder_traversal(expression)
        if part.free_symbols & point.keys()
    )


def _zero_by_sides(
    left: sp.Basic, absolute: sp.Abs
) -> bool | tuple[sp.Abs, frozenset[int]] | None:
    """`_zero_by_signs` for the one `ABS(q)` in `left`, with no parameter to substitute
    for `q`: each side of zero is decided with `ABS(q)` written as `q` or `-q`, holding
    on the whole side where that is identically zero, and nowhere where it is never
    zero or its zeros are proved to lie on the other side."""
    (inside,) = absolute.args
    found: dict[int, str] = {}
    for sign in (1, -1):
        side = left.xreplace({absolute: sign * inside})
        decided = _zero_decided(side, None)
        if decided is True:
            found[sign] = "all"
        elif decided is False:
            found[sign] = "never"
        elif _zeros_outside(side, sign * inside):
            found[sign] = "outside"
        else:
            return None
    if found[1] == found[-1] == "all":
        return True
    if found[1] == "all":
        return absolute, frozenset({1, 0})
    if found[-1] == "all":
        return absolute, frozenset({0, -1})
    if found[1] == found[-1] == "never":
        return False
    return None


def _zeros_outside(side: sp.Basic, positive: sp.Basic) -> bool:
    """Whether every zero of `side` is proved to make `positive` not positive, by solving
    `side = 0` exactly for one of its parameters."""
    for parameter in sorted(side.free_symbols, key=sp.default_sort_key):
        zeros = sp.solveset(side, parameter, sp.S.Reals)
        if zeros is sp.S.EmptySet:
            return True
        if isinstance(zeros, sp.FiniteSet) and all(
            positive.xreplace({parameter: zero}).is_positive is False
            for zero in zeros.args
        ):
            return True
    return False


def _zero_decided(left: sp.Basic, region: sp.Dummy | None) -> bool | None:
    """Whether `left`, free of the variable and of `ABS`, is zero for all real values
    of its parameters (True), or for none (False), or None if undecided. A value that
    is undefined is not zero. With `region`, a positive dummy, a solution of `left = 0`
    for it is sought exactly; there being none decides False.
    """
    if left.has(sp.nan, sp.zoo):
        return False
    simplified = sp.simplify(left)
    if simplified.has(sp.nan):
        return None
    if simplified.has(sp.zoo):
        return False
    zero = simplified.is_zero
    if zero is None:
        zero = sp.expand_complex(simplified).is_zero
    if zero is None:
        polar = _polar_powers(simplified)
        if polar != simplified:
            polar = sp.simplify(polar)
            zero = polar.is_zero
            if zero is None:
                zero = sp.expand_complex(polar).is_zero
    if zero is None and sp.im(simplified).is_zero is False:
        zero = False
    if zero is None and region is not None and _no_positive_zero(left, region):
        zero = False
    return zero



#: The largest denominator of `r` that `_polar_powers` tries in an argument `r*pi`.
_ARGUMENT_DENOMINATOR = 60


def _polar_powers(expression: sp.Basic) -> sp.Basic:
    """`expression` with each non-integer power of a complex constant whose argument is
    a rational multiple of pi written in polar form, `m^e*#e^(#i*e*r*pi)`, the
    principal value, so that the power of the unit evaluates exactly: `(-1 - SQRT(5) +
    #i*SQRT(2)*SQRT(5 - SQRT(5)))^(5/2)` is `4^(5/2)*#e^(2*#i*pi)`, which is 32.

    The candidate `r` is read off the numeric argument, and is used only if the base
    is proved equal to `m*(COS(r*pi) + #i*SIN(r*pi))` exactly: the minimal polynomial
    of the difference is `x`.
    """

    def candidate(power: sp.Basic) -> bool:
        return (
            isinstance(power, sp.Pow)
            and power.exp.is_Rational
            and not power.exp.is_Integer
            and power.base.is_number
            and power.base.is_positive is not True
            and sp.count_ops(power.base) <= 40
        )

    def polar(power: sp.Pow) -> sp.Basic:
        base = power.base
        try:
            real, imaginary = base.as_real_imag()
            modulus = sp.sqrt(sp.expand(real**2 + imaginary**2))
            turns = sp.N(sp.arg(base) / sp.pi, 40)
            ratio = Fraction(str(turns)).limit_denominator(_ARGUMENT_DENOMINATOR)
            turn = sp.Rational(ratio.numerator, ratio.denominator)
            if abs(turns - turn) > 1e-30 or not -1 < turn <= 1:
                return power
            angle = turn * sp.pi
            unknown = sp.Dummy()
            difference = base - modulus * (sp.cos(angle) + sp.I * sp.sin(angle))
            if sp.minimal_polynomial(difference, unknown) != unknown:
                return power
        except Exception:
            return power
        return modulus**power.exp * sp.exp(sp.I * angle * power.exp)

    try:
        return expression.replace(candidate, polar)
    except Exception:
        return expression

def _no_positive_zero(left: sp.Basic, region: sp.Dummy) -> bool:
    """Whether `left = 0` is proved to have no solution for the positive `region`, by
    solving each of its factors exactly. Solveset takes a coefficient it can't decide
    for nonzero, so a factor free of `region` must be proved nonzero, and a polynomial
    factor must have its leading coefficient proved nonzero: `size*(1 - c)` with `c` a
    power that is 1, though not visibly, has every positive `size` for a zero.
    """
    size = sp.Symbol(region.name, real=True)
    positive = sp.Interval.open(0, sp.oo)
    try:
        factors = sp.Mul.make_args(sp.factor_terms(left.xreplace({region: size})))
        for factor in factors:
            if isinstance(factor, sp.Pow) and factor.exp.is_positive:
                factor = factor.base
            if not factor.has(size):
                if _zero_decided(factor, None) is not False:
                    return False
                continue
            polynomial = _polynomial(factor, size)
            if (
                polynomial is not None
                and _zero_decided(polynomial.LC(), None) is not False
            ):
                return False
            if sp.solveset(factor, size, positive) is not sp.S.EmptySet:
                return False
    except Exception:
        return False
    return True


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
        cores = [_core(piece) for piece in found.args]
        for piece in found.args:
            core = _core(piece)
            if isinstance(core, sp.Intersection) and any(
                part in cores for part in core.args
            ):
                # Part of another piece, whose members stand for it.
                continue
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


def _core(piece: sp.Set) -> sp.Set:
    """A piece of a solution set without what it excludes."""
    return piece.args[0] if isinstance(piece, sp.Complement) else piece


def _remaining(inside: sp.Set, excluded: sp.Set) -> list[sp.Basic] | None:
    """The members of `inside` that aren't provably members of `excluded`.

    For `SOLVE(1 - 2*m/r = 0, r)` sympy answers `Complement({2*m}, {0})`. Undecidable
    isn't excluded, so the answer is the original's `r = 2*m`.

    A family minus another, as where the equation's divisor vanishes at `2*n*pi`, is
    represented by the member nearest the origin, unless that member is excluded:
    a family minus some of its members has no notation.
    """
    if isinstance(inside, sp.ImageSet) and not isinstance(excluded, sp.FiniteSet):
        members = _members(inside)
        if members is None or any(_provably_holds(excluded, value) for value in members):
            return None
        return members
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
    """The relation, the answer for anything unsolved.

    An equation is written as `_rearranged` says: `SIN(x) = 1 - x/6` comes back as
    `6*SIN(x) + x = 6`. Another relation keeps its operator, with its residual on the
    left: `x^2 < y` is `x^2 - y < 0`. A relation whose sides can't be subtracted is
    returned as is.
    """
    func = expression.func if isinstance(expression, Relational) else sp.Eq
    try:
        if func is sp.Eq:
            return (_rearranged(_residual(expression), context),)
        residual = normal_form(_residual(expression), context.order)
    except Exception:
        return (expression,)
    return (_relation(func, residual, sp.Integer(0)),)


def unsolved_equation(residual: sp.Basic, context: Context) -> sp.Basic:
    """`residual = 0` written as an unsolved equation (see `_rearranged`)."""
    try:
        return _rearranged(residual, context)
    except Exception:
        return _relation(sp.Eq, residual, sp.Integer(0))


def _rearranged(residual: sp.Basic, context: Context) -> sp.Basic:
    """`residual = 0` as an unsolved equation is written: the numerator of its common
    fraction, with integer coefficients that share no factor and its leading term
    positive, with the number term moved to the right side.

    So `x^5 + x + 3` is `x^5 + x = -3`, `1.5*x*COS(x) - SIN(x)/2` is `3*x*COS(x) -
    SIN(x) = 0`, and terms in parameters stay on the left: `v - 2*u - 1` is `v - 2*u =
    1`. The denominator's zeros are no solutions, so leaving it out loses nothing.
    """
    numerator = sp.fraction(sp.together(residual))[0]
    numerator = normal_form(numerator, context.order)
    _, primitive = numerator.as_content_primitive()
    if _leads_negative(primitive, context):
        primitive = -primitive
    number = sp.Add(*(term for term in sp.Add.make_args(primitive) if term.is_number))
    left = normal_form(primitive - number, context.order)
    if left == 0:
        return _relation(sp.Eq, primitive, sp.Integer(0))
    return _relation(sp.Eq, left, -number)


def _leads_negative(expression: sp.Basic, context: Context) -> bool:
    """Whether the term of `expression` written first has a negative coefficient."""
    try:
        terms = term_order(expression.as_ordered_terms(), context.order)
    except Exception:
        return expression.could_extract_minus_sign()
    return terms[0].could_extract_minus_sign()


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
    if isinstance(relation, sp.Unequality):
        return _unequal(relation, variable, context, constants)
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


def _unequal(
    relation: Relational,
    variable: sp.Symbol,
    context: Context,
    constants: _Constants,
) -> tuple[sp.Basic, ...]:
    """`u /= v`, solved as every real value but the real solutions of `u = v`:
    `x^2 /= 4` is `x /= -2 AND x /= 2`. An equation without solutions makes it hold
    everywhere, an identity nowhere, and an unsolved one leaves it unsolved.
    """
    found = _equation(
        _residual(relation), variable, context, None, (), constants, True
    )
    values = []
    for entry in found:
        if _is_identity(entry, variable, constants):
            return ()
        if not (
            isinstance(entry, sp.Equality)
            and entry.lhs == variable
            and not entry.rhs.has(variable, sp.Piecewise)
        ):
            return _unsolved(relation, context)
        values.append(entry.rhs)
    if not values:
        return (_equated(variable, constants.fresh(), context),)
    return (sp.And(*(_relation(sp.Ne, variable, value) for value in values)),)


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
    a solution with any known non-real value is dropped. Solutions that can't be written
    follow as the vector of their reduced equations, one per factor of the eliminating
    polynomial without radical roots, as a single equation keeps such a factor (see
    `_split`).
    """
    joined = Vector(*equations)
    chosen = _chosen(joined, context, variables, len(equations))
    if not chosen:
        return _unsolvable(joined, context)
    if not all(isinstance(equation, sp.Equality) for equation in equations):
        return _inequalities(equations, chosen, context, constants)
    residuals = [_residual(equation) for equation in equations]
    conditions: list[sp.Basic] = []
    if any(residual.has(*chosen) for residual in residuals):
        # An equation in none of the variables is a condition on the parameters that
        # every solution carries: `[2*y*COS(t) = 3*SIN(2*t), SIN(2*t) = 0]` solved for
        # `y` is `y = 3*SIN(t)/COS(t) AND SIN(2*t) = 0`. One that is false has none.
        kept = []
        for residual in residuals:
            if residual.has(*chosen):
                kept.append(residual)
                continue
            zero = _vanishes_identically(residual)
            if zero is False:
                return ()
            if zero is not True and not _cancels(residual, chosen[0]):
                conditions.append(unsolved_equation(residual, context))
        if conditions and _inconsistent(
            [condition.lhs - condition.rhs for condition in conditions], ()
        ):
            return ()
        if conditions:
            # A condition that gives a parameter its value holds in every solution, so
            # the value replaces the parameter in the other equations.
            fixed = {
                condition.lhs: condition.rhs
                for condition in conditions
                if isinstance(condition, sp.Equality)
                and isinstance(condition.lhs, sp.Symbol)
                and not condition.rhs.has(condition.lhs, *chosen)
            }
            substituted = [residual.xreplace(fixed) for residual in kept]
            if all(residual.has(*chosen) for residual in substituted):
                kept = substituted
            found = _system(
                tuple(_relation(sp.Eq, residual, 0) for residual in kept),
                context,
                tuple(variable.name for variable in chosen),
                constants,
                real,
            )
            return tuple(
                Vector(*_conjuncts(entry), *dict.fromkeys(conditions)) for entry in found
            )
        residuals = kept
    solved = _linear(residuals, chosen, constants)
    if solved == []:
        # Inconsistent only where the parameters take no special values: `[a = x, a =
        # y]` holds where `x = y`, and `[a = COS(t)^2 + 1, a = 2 - SIN(t)^2]` everywhere.
        # The equations that depend on the others are conditions on the parameters.
        dependent = _dependent_conditions(residuals, chosen)
        if dependent is None:
            return _unsolvable(joined, context)
        independent, left = dependent
        if not any(residual.has(*chosen) for residual in independent):
            # No equation is in the variables: the system is its conditions alone.
            if any(
                _vanishes_identically(residual) is False for residual in left
            ) or _inconsistent(left, ()):
                return ()
            return _unsolvable(joined, context)
        if not left:
            return _unsolvable(joined, context)
        return _system(
            tuple(_relation(sp.Eq, residual, 0) for residual in independent + left),
            context,
            tuple(variable.name for variable in chosen),
            constants,
            real,
        )
    remaining: list[_Unwritten] = []
    if solved is None:
        split = _nonlinear(
            residuals, chosen, numeric=context.precision is not Precision.EXACT
        )
        if split is None:
            if _inconsistent(residuals, chosen):
                return ()
            return _unsolvable(joined, context)
        solved, remaining = split
    if remaining and context.precision is not Precision.EXACT:
        completed = [
            _numerically(part, residuals, chosen, context, real) for part in remaining
        ]
        if all(part is not None for part in completed):
            solved = solved + [mapping for part in completed for mapping in part]
            remaining = []
    if real:
        solved = [
            mapping
            for mapping in solved
            if not any(_imaginary(value) for value in mapping.values())
        ]
    solved = [_with_arbitrary_values(mapping, chosen, constants) for mapping in solved]
    answers = [
        Vector(
            *(
                _equated(variable, mapping.get(variable, variable), context)
                for variable in chosen
            )
        )
        for mapping in solved
    ]
    for part in remaining:
        answers.append(
            Vector(
                *(
                    unsolved_equation(value, context)
                    if variable is None
                    else _equated(variable, value, context)
                    for variable, value in part
                )
            )
        )
    return tuple(answers)


def _conjuncts(entry: sp.Basic) -> tuple[sp.Basic, ...]:
    """The relations of an entry of a system's solution."""
    return entry.args if isinstance(entry, Vector) else (entry,)


def _numerically(
    part: _Unwritten,
    residuals: list[sp.Basic],
    chosen: tuple[sp.Symbol, ...],
    context: Context,
    real: bool,
) -> list[dict[sp.Symbol, sp.Basic]] | None:
    """The solutions of a reduced system whose eliminating polynomial has no radical
    roots, in numbers: its roots are found numerically and substituted back, equation
    by equation, a polynomial in one remaining unknown branching on its roots. None if
    an unknown can't be reached so.

    A branch is kept only where every equation of the reduced system and every one of
    the system itself (`residuals`) vanishes at it to the working precision: a root of
    one equation of the reduced system need not solve the others in the same unknown.

    The work is done at twice the digits shown, so the values substituted back keep
    the digits of the answer.
    """
    working = context.with_precision(
        Precision.APPROXIMATE, 2 * context.precision_digits + GUARD
    )
    branches: list[dict[sp.Symbol, sp.Basic]] = [{}]
    pending = list(part)
    while pending:
        progressed = False
        for relation in list(pending):
            variable, value = relation
            grown: list[dict[sp.Symbol, sp.Basic]] = []
            for known in branches:
                found = _completed(variable, value, known, chosen, working, real)
                if found is None:
                    break
                grown.extend(found)
            else:
                branches = grown
                pending.remove(relation)
                progressed = True
        if not progressed:
            return None
    if any(set(chosen) - set(known) for known in branches):
        return None
    digits = context.precision_digits + GUARD
    return [
        {
            variable: _approximate_number(value, context)
            for variable, value in known.items()
        }
        for known in branches
        if all(_negligible(residual, known, digits) for residual in residuals)
    ]


def _completed(
    variable: sp.Symbol | None,
    value: sp.Basic,
    known: dict[sp.Symbol, sp.Basic],
    chosen: tuple[sp.Symbol, ...],
    working: Context,
    real: bool,
) -> list[dict[sp.Symbol, sp.Basic]] | None:
    """`known` extended by one relation of a reduced system, `v = value` or `value = 0`,
    as the branches it splits into, or None if the relation has more than one unknown
    left."""
    written = value
    value = value.xreplace(known)
    unknown = [symbol for symbol in chosen if symbol not in known and value.has(symbol)]
    if variable is not None:
        if unknown:
            return None
        return [{**known, variable: _approximate_number(value, working)}]
    if not unknown:
        digits = working.precision_digits // 2
        return [known] if _negligible(written, known, digits) else []
    if len(unknown) != 1:
        return None
    polynomial = _polynomial(sp.expand(value), unknown[0])
    if polynomial is None or polynomial.degree() < 1:
        return None
    roots = _polynomial_roots(polynomial, working, real)
    return [{**known, unknown[0]: root} for root in roots]


def _negligible(
    expression: sp.Basic, known: dict[sp.Symbol, sp.Basic], digits: int
) -> bool:
    """Whether `expression` is zero at the approximate values `known` to `digits` digits
    of the largest of its terms."""
    try:
        terms = sp.Add.make_args(sp.expand(expression))
        sizes = [
            abs(complex(term.xreplace(known).evalf(digits + GUARD))) for term in terms
        ]
        total = abs(complex(expression.xreplace(known).evalf(digits + GUARD)))
    except Exception:
        return False
    return total <= max(sizes, default=0.0) * 10.0 ** -digits


def _dependent_conditions(
    residuals: list[sp.Basic], chosen: tuple[sp.Symbol, ...]
) -> tuple[list[sp.Basic], list[sp.Basic]] | None:
    """A linear system that `linsolve` finds inconsistent, split into the equations that
    are consistent and the residuals of the others at their solution, which are free of
    the variables: `[a - x, a - y]` is `[a - x]` and `[x - y]`. None if the split fails.

    `linsolve` decides consistency for values of the parameters and calls in them that
    are unrelated, so an equation that follows from the others only for some values, or
    only through an identity it doesn't know, is inconsistent to it. Each residual left
    is a condition on the parameters: the system holds exactly where they vanish.
    """
    unknowns = list(reversed(chosen))
    independent: list[sp.Basic] = []
    values: dict[sp.Symbol, sp.Basic] = {}
    left: list[sp.Basic] = []
    try:
        for residual in residuals:
            found = sp.linsolve([*independent, residual], unknowns)
            if isinstance(found, sp.FiniteSet) and len(found.args) == 1:
                independent.append(residual)
                values = dict(zip(unknowns, found.args[0], strict=True))
                continue
            if found is not sp.S.EmptySet:
                return None
            condition = sp.cancel(sp.together(residual.xreplace(values)))
            if condition.has(*chosen):
                return None
            left.append(condition)
    except Exception:
        return None
    return independent, left


def _vanishes_identically(residual: sp.Basic) -> bool | None:
    """Whether a residual free of the variables is zero for every value of the
    parameters, False if it is for none, None if that is undecided. `COS(t)^2 +
    SIN(t)^2 - 1` is zero, `x - y` undecided."""
    known = residual.is_zero
    if known is None:
        try:
            simplified = budget.within(budget.DECIDING, lambda: sp.simplify(residual))
        except Exception:
            return None
        known = None if simplified is None else simplified.is_zero
    return known


def _inconsistent(residuals: list[sp.Basic], chosen: tuple[sp.Symbol, ...]) -> bool:
    """Whether a polynomial system has no solution for any value of the parameters: its
    numerators generate the unit ideal in the variables and parameters together, so
    their Groebner basis is `[1]`. `[x^2 + y^2 = 1, x^2 + y^2 = 2]` has none, while
    `[x^2 = a, x^2 = b]` has some where `a = b`.

    Only a system rational in all its symbols counts: functions of the parameters are
    related in ways the basis doesn't know, as `TAN(a)*COS(2*a)` is to `COS(a)`.
    """
    if not all(
        residual.is_rational_function(*residual.free_symbols) for residual in residuals
    ):
        return False
    parameters = sorted(
        set().union(*(residual.free_symbols for residual in residuals)) - set(chosen),
        key=sp.default_sort_key,
    )
    try:
        basis = budget.within(
            budget.DECIDING,
            lambda: lex_basis(residuals, (*chosen, *parameters), rational=True),
        )
    except Exception:
        return False
    return basis is not None and basis.exprs == [1]


def lex_basis(
    residuals: Sequence[sp.Basic], chosen: Sequence[sp.Symbol], rational: bool = False
) -> sp.GroebnerBasis:
    """The reduced Groebner basis of the numerators of `residuals`, lexicographic with
    `chosen[0] > chosen[1] > ...`; other symbols are coefficients.

    Raises when a numerator is not a polynomial in `chosen`, and with `rational` also
    when a coefficient is not a rational function of the other symbols with rational
    coefficients: over constants like `COT(161*pi/900)` the basis can't tell zero from
    not.
    """
    polynomials = [
        sp.Poly(sp.fraction(sp.together(residual))[0], *chosen)
        for residual in residuals
    ]
    if rational and not all(
        _rational_domain(polynomial.domain) for polynomial in polynomials
    ):
        raise ValueError("coefficients not rational")
    return sp.groebner(polynomials, *chosen, order="lex")


def _rational_domain(domain) -> bool:
    """Whether polynomial coefficients in `domain` are rational, or polynomials or
    quotients of them in parameters with rational coefficients."""
    if domain.is_ZZ or domain.is_QQ:
        return True
    if domain.is_PolynomialRing or domain.is_FractionField:
        return domain.dom.is_ZZ or domain.dom.is_QQ
    return False


def _with_arbitrary_values(
    mapping: dict[sp.Symbol, sp.Basic],
    chosen: tuple[sp.Symbol, ...],
    constants: _Constants,
) -> dict[sp.Symbol, sp.Basic]:
    """A solution with each chosen variable it leaves free given a fresh `@n` constant,
    in its own entry and in the values written in it: `[x^2 + y^2 = t^2]` solved for
    `x`, `y` and `t` has `x = @1, y = @2, t = SQRT(@1^2 + @2^2)`.
    """
    free = [variable for variable in chosen if variable not in mapping]
    if not free:
        return mapping
    arbitrary = {variable: constants.fresh() for variable in free}
    return {
        variable: arbitrary[variable]
        if variable in arbitrary
        else mapping[variable].xreplace(arbitrary)
        for variable in chosen
    }


def _inequalities(
    relations: tuple[Relational, ...],
    chosen: tuple[sp.Symbol, ...],
    context: Context,
    constants: _Constants,
) -> tuple[sp.Basic, ...]:
    """A system with an inequality in it, solved into ranges of a single variable.

    `[x > 1, x < 3]` is `1 < x < 3`. A system in several variables is a region, which no
    list of solutions describes: it is left unsolved.
    """
    joined = Vector(*relations)
    if len(chosen) != 1:
        return _unsolvable(joined, context)
    (variable,) = chosen
    try:
        reduced = sp.reduce_inequalities(
            [_evaluated(relation) for relation in relations], [variable]
        )
    except Exception:
        return _unsolvable(joined, context)
    entries = _reduced(reduced, variable, context, constants)
    return _unsolvable(joined, context) if entries is None else entries


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
        if found is sp.S.EmptySet and any(
            residual.has(TrigonometricFunction) for residual in residuals
        ):
            # Elimination sees `TAN(phi)*COS(2*phi)` and `2*COS(phi)^2 - 1` as unrelated
            # and finds no solution; in sines and cosines of single angles it does.
            written = [sp.expand_trig(TR2(residual)) for residual in residuals]
            found = sp.linsolve(written, list(reversed(chosen)))
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


_Unwritten = list[tuple[sp.Symbol | None, sp.Basic]]
"""What remains of a system once its written solutions are listed: `(v, value)` for `v =
value`, `(None, polynomial)` for `polynomial = 0`."""


def _nonlinear(
    residuals: list[sp.Basic], chosen: tuple[sp.Symbol, ...], numeric: bool = False
) -> tuple[list[dict[sp.Symbol, sp.Basic]], list[_Unwritten]] | None:
    """A system `linsolve` refused, solved as far as `solve` can, beyond what the original
    handles: the solutions, and the equations of those that can't be written (see
    `_split`). None means "nothing found", not "no solutions".

    A rational system is solved over the complexes, as a single equation is: `solve`
    would otherwise drop the roots it can prove non-real for real variables, but keep
    those it can't, so `[x^5 + x + y = 0, y = 1]` lost the roots of `x^2 + x + 1` while
    keeping the complex ones of the cubic factor.

    A constant like `COT(161*pi/900)` is solved for as a parameter: `solve` would try
    to find the number field it lies in, and fail.
    """
    plain = {variable: sp.Dummy(variable.name) for variable in chosen}
    if not all(residual.is_rational_function(*chosen) for residual in residuals):
        plain = {variable: variable for variable in chosen}
    calls = {
        call
        for residual in residuals
        for call in residual.atoms(sp.Function)
        if call.is_number
    }
    constants = {call: sp.Dummy("constant") for call in sp.ordered(calls)}
    plain.update(constants)
    back = {stand_in: variable for variable, stand_in in plain.items()}
    residuals = [residual.xreplace(plain) for residual in residuals]
    stand_ins = tuple(plain[variable] for variable in chosen)
    # The split first: where it applies, `solve` would try to write the roots that
    # have no radicals, which for a system of six Lagrange equations takes minutes.
    split = _split(residuals, stand_ins, numeric)
    if split is None:
        try:
            found = sp.solve(residuals, list(stand_ins), dict=True)
        except Exception:
            return None
        split = ([dict(mapping) for mapping in found], [])
    solutions, remaining = split
    if not (solutions or remaining):
        return None
    return (
        [
            {back[key]: value.xreplace(back) for key, value in mapping.items()}
            for mapping in solutions
            if all(key in back for key in mapping)
        ],
        [
            [
                (None if key is None else back[key], value.xreplace(back))
                for key, value in part
            ]
            for part in remaining
        ],
    )


#: The size, in operations, of a radical root of an eliminating polynomial of degree
#: above 4 beyond which it is treated as a root without radicals.
_WRITTEN_ROOT = 60


def _split(
    residuals: list[sp.Basic], chosen: tuple[sp.Symbol, ...], numeric: bool = False
) -> tuple[list[dict[sp.Symbol, sp.Basic]], list[_Unwritten]] | None:
    """A polynomial system split by the factors of its eliminating polynomial, when some
    have no radicals for their roots. When none lacks them, the solutions from
    `_from_eliminants` where a factor is above quadratic, or else None.

    `solve` drops the solutions whose last variable is a root of such a factor without a
    word. For `[x^2*y + 3*y*z = 4, 2*y^2 - 3*x^2*z = -1, 2*y*z^2 - z^2 = 1]` the lex
    basis eliminates to `(z - 1)*(3*z^2 - 2*z + 1)*q(z)`, `q` a quintic. The solutions
    on the other factors are solved from the basis joined with their product; those on
    each factor without radicals are the reduced basis joined with it, each element
    written as `v = value` where linear in its leading variable `v`: `[3*x^2 + r(z) = 0,
    y = s(z), q(z) = 0]`. Each solution of the system is in exactly one of the parts.

    With `numeric`, for an answer in numbers, every factor is such a part once one is.

    A system with a variable in a denominator, or whose parts can't be solved, gives
    nothing at all: it is left unsolved rather than answered in part.
    """
    try:
        basis = lex_basis(residuals, chosen)
    except Exception:
        return None
    last = chosen[-1]
    eliminants = [
        polynomial
        for polynomial in basis.exprs
        if polynomial.free_symbols & set(chosen) == {last}
    ]
    if not eliminants:
        return None
    try:
        _, factors = sp.factor_list(eliminants[0], last)
    except Exception:
        return None
    written: list[sp.Basic] = []
    unwritten: list[sp.Basic] = []
    for factor, _ in factors:
        degree = sp.degree(factor, last)
        if degree < 1:
            continue
        found = sp.roots(factor, last)
        roots = sum(found.values())
        if degree > 4 and any(sp.count_ops(root) > _WRITTEN_ROOT for root in found):
            # Radicals this large, from a polynomial that decomposes, make the
            # substitution into the rest of the basis take minutes.
            roots = 0
        (written if roots == degree else unwritten).append(factor)
    if not unwritten:
        if not any(sp.degree(factor, last) > 2 for factor in written):
            return None
        found = _from_eliminants(basis, chosen, residuals)
        return None if found is None else (found, [])
    if numeric:
        # The solutions will be numbers, which the reduced bases give every one of:
        # `solve` on the factors with radicals may miss some, or take minutes.
        unwritten, written = written + unwritten, []
    unsolved: tuple[list[dict[sp.Symbol, sp.Basic]], list[_Unwritten]] = ([], [])
    if any(
        sp.fraction(sp.together(residual))[1].has(*chosen) for residual in residuals
    ):
        return unsolved
    solutions: list[dict[sp.Symbol, sp.Basic]] = []
    if written:
        try:
            found = sp.solve(
                [*basis.exprs, sp.Mul(*written)], list(chosen), dict=True
            )
        except Exception:
            return unsolved
        if not found:
            return unsolved
        solutions = [dict(mapping) for mapping in found]
    remaining: list[_Unwritten] = []
    for factor in unwritten:
        try:
            part = lex_basis([*basis.exprs, factor], chosen)
        except Exception:
            return unsolved
        remaining.append([_solved_for_leading(e, chosen) for e in part.exprs])
    return solutions, remaining


def _from_eliminants(
    basis: sp.GroebnerBasis, chosen: tuple[sp.Symbol, ...], residuals: list[sp.Basic]
) -> list[dict[sp.Symbol, sp.Basic]] | None:
    """The solutions of a rational system with finitely many, each variable's value
    written as a root of its own eliminating polynomial, or None where that doesn't
    apply or the solutions aren't told apart.

    Solving the basis by substitution writes a variable through the roots of the
    variables after it: in `[a + b + c = 3, a^2 + b^2 + c^2 = 9, a^3 + b^3 + c^3 = 24]`
    the cubic for `c` is solved by Cardano with complex cube roots, and `b` from a
    quadratic in that. Each of `a`, `b` and `c` is a root of `t^3 - 3*t^2 + 1`, which a
    single equation writes in cosines; the solutions are the combinations of such
    roots at which every element of the basis vanishes.

    A combination at which a numeric value of an element is clearly nonzero is
    rejected. The rest are the solutions when there are as many as the system has,
    which is counted exactly (see `_solution_count`); otherwise some combination
    numbers can't reject is not a solution, and None is returned.
    """
    if any(
        sp.fraction(sp.together(residual))[1].has(*chosen) for residual in residuals
    ):
        return None
    elements = list(basis.exprs)
    if any(element.free_symbols - set(chosen) for element in elements):
        return None
    if not all(_rational_domain(sp.Poly(e, *chosen).domain) for e in elements):
        return None
    eliminants: list[sp.Basic] = []
    candidates: dict[sp.Symbol, list[sp.Basic]] = {}
    for variable in chosen:
        order = (*(other for other in chosen if other != variable), variable)
        try:
            ordered = basis if order == chosen else lex_basis(elements, order)
        except Exception:
            return None
        found = [e for e in ordered.exprs if e.free_symbols == {variable}]
        if not found:
            return None
        eliminants.append(found[0])
        roots = _values(found[0], variable)
        if not isinstance(roots, sp.FiniteSet):
            return None
        candidates[variable] = list(roots.args)
    count = _solution_count(elements, eliminants, chosen)
    if count is None:
        return None
    # A univariate element vanishes at every root of the eliminant, which divides it.
    checked = [element for element in elements if len(element.free_symbols) > 1]
    points: list[dict[sp.Symbol, sp.Basic]] = [{}]
    for variable in reversed(chosen):
        known = {variable, *points[0]}
        now = [
            element
            for element in checked
            if variable in element.free_symbols and element.free_symbols <= known
        ]
        extended = []
        for assigned in points:
            for value in candidates[variable]:
                point = {**assigned, variable: value}
                try:
                    if any(abs(sp.N(e.xreplace(point), 30)) > 1e-15 for e in now):
                        continue
                except Exception:
                    return None
                extended.append(point)
                if len(extended) > count:
                    return None
        points = extended
    if len(points) != count:
        return None
    return [{variable: point[variable] for variable in chosen} for point in points]


#: The most monomials `_solution_count` enumerates.
_COUNTED_MONOMIALS = 10000


def _solution_count(
    elements: list[sp.Basic], eliminants: list[sp.Basic], chosen: tuple[sp.Symbol, ...]
) -> int | None:
    """The number of distinct solutions of the system with the basis `elements`, given
    an eliminating polynomial in each variable, or None if too many monomials to count.

    The ideal joined with the squarefree part of each eliminant is radical, and has
    the same solutions; their number is the count of monomials that its basis
    doesn't reduce, those divisible by no leading monomial.
    """
    try:
        squarefree = [sp.sqf_part(eliminant) for eliminant in eliminants]
        radical = lex_basis([*elements, *squarefree], chosen)
        leading = [sp.Poly(e, *chosen).monoms(order="lex")[0] for e in radical.exprs]
    except Exception:
        return None
    bounds = []
    for index in range(len(chosen)):
        powers = [
            monomial[index]
            for monomial in leading
            if all(k == 0 for j, k in enumerate(monomial) if j != index)
        ]
        if not powers:
            return None
        bounds.append(min(powers))
    if math.prod(bounds) > _COUNTED_MONOMIALS:
        return None
    return sum(
        1
        for monomial in itertools.product(*(range(bound) for bound in bounds))
        if not any(all(m >= k for m, k in zip(monomial, lead)) for lead in leading)
    )


def _solved_for_leading(
    polynomial: sp.Basic, chosen: tuple[sp.Symbol, ...]
) -> tuple[sp.Symbol | None, sp.Basic]:
    """A basis element as `(v, value)` when linear in its leading variable `v` with a
    coefficient free of the chosen variables, else as `(None, polynomial)`."""
    leading = next(variable for variable in chosen if polynomial.has(variable))
    poly = sp.Poly(polynomial, leading)
    if poly.degree() == 1:
        coefficient, rest = poly.all_coeffs()
        if not coefficient.has(*chosen):
            return leading, sp.expand(-rest / coefficient)
    return None, polynomial


# -- numbers -----------------------------------------------------------------


def _numeric(
    residual: sp.Basic,
    variable: sp.Symbol,
    low: sp.Basic,
    high: sp.Basic,
    context: Context,
) -> tuple[sp.Basic, ...]:
    """Every real root in `[low, high]`, for Approximate precision.

    Unlike Derive, all of a polynomial's roots are returned. Anything else gives one
    root, as `numeric_roots` finds it; an equation whose search neither finds a root
    nor proves there is none is returned unsolved.
    """
    roots = numeric_roots(residual, variable, context, True, (low, high))
    if roots is None:
        return _unsolved(residual, context)
    return tuple(_equated(variable, root, context) for root in roots)


def numeric_roots(
    residual: sp.Basic,
    variable: sp.Symbol,
    context: Context,
    real: bool = False,
    bounds: tuple[sp.Basic, sp.Basic] | None = None,
) -> list[sp.Basic] | None:
    """The roots `NSOLVE` finds for `residual = 0`, a function of `variable` alone, or
    None if the search found none without proving that there is none.

    A polynomial gives every distinct root (`_polynomial_roots`), only the real ones
    with `real` and only those in `bounds` with bounds; that list is complete. Anything
    else gives one real root: the leftmost one a scan of `bounds` detects, or without
    bounds that of the first window of `[-10, 10]`, `[-20, 20]`, ... that holds one.
    Where the scan detects none, interval arithmetic searches the whole range (see
    `_explored`): it finds roots too close together for the scan to see a sign change
    between them, and an empty list is its proof that the range holds no root.
    """
    polynomial = _number_polynomial(residual, variable)
    if polynomial is not None:
        low, high = bounds if bounds is not None else (-sp.oo, sp.oo)
        roots = _polynomial_roots(polynomial, context, real or bounds is not None)
        return [root for root in roots if bounds is None or _within(root, low, high)]
    if _calls_defined(residual, context):
        return _pointwise_roots(residual, variable, context, bounds)
    low, high = bounds if bounds is not None else (-sp.oo, sp.oo)
    first, last = _number(low), _number(high)
    if first is None or last is None or not first < last:
        return None
    if bounds is None:
        found = _searched(residual, variable, context)
    elif _finite(*bounds):
        root = _bracketed(residual, variable, low, high, context)
        found = None if root is None else (_equated(variable, root, context),)
    else:
        found = _unbounded(residual, variable, low, high, context)
    if found:
        return [entry.rhs for entry in found]
    search = _explored(residual, variable, first, last, context)
    if search.root is not None:
        return [search.root]
    return [] if search.rootless else None


def _calls_defined(residual: sp.Basic, context: Context) -> bool:
    """Whether `residual` holds a call of a defined function that stands, as a program
    over a variable does: substituting a number into it computes nothing."""
    return any(
        context.function(type(call).__name__) is not None
        for call in residual.atoms(AppliedUndef)
    )


#: How many points `_pointwise_roots` samples a range at, looking for a sign change,
#: and the program steps one point may take.
_POINTWISE_SAMPLES = 200
_POINTWISE_STEPS = 2_000


def _pointwise_roots(
    residual: sp.Basic,
    variable: sp.Symbol,
    context: Context,
    bounds: tuple[sp.Basic, sp.Basic] | None,
) -> list[sp.Basic] | None:
    """One real root of a residual holding calls of defined functions, found by
    evaluating it anew at each point with its programs running, as a plot does (see
    `plot_form.Pointwise`), or None if none is found.

    The range, `bounds` or else `[-10, 10]` doubled until one turns up, is sampled for
    a sign change between two values, which is bisected; the root is taken only if
    the value there is small next to those at the bracket's ends, which a pole is
    not. Finding none proves nothing, so the answer is then None, never empty.
    """
    from rederive.engine.from_sympy import from_sympy
    from rederive.engine.plot_form import Pointwise

    try:
        node = from_sympy(residual, context).node
    except Exception:
        return None
    # Past the digits asked for, so that the bisection settles the digits shown.
    working = context.with_precision(Precision.APPROXIMATE, context.precision_digits + 5)
    pointwise = Pointwise([node], working, [variable.name], _POINTWISE_STEPS)

    def value(point: float) -> float | None:
        (found,) = pointwise.at(point)
        if found is None or abs(found.imag) > 1e-9 * max(1.0, abs(found.real)):
            return None
        return found.real

    if bounds is not None:
        first, last = _number(bounds[0]), _number(bounds[1])
        if first is None or last is None or not first < last:
            return None
        if math.isinf(first) or math.isinf(last):
            return None
        windows = [(first, last)]
    else:
        windows = []
        reach = _REACH
        while reach <= _REACH_CAP:
            windows.append((-reach, reach))
            reach *= 4
    for low, high in windows:
        root = _sign_change(value, low, high)
        if root is not None:
            return [_approximate(sp.Float(root, 17), context)]
    return None


def _sign_change(
    value: Callable[[float], float | None], low: float, high: float
) -> float | None:
    """The leftmost root of `value` a scan of `[low, high]` brackets, bisected, or
    None (see `_pointwise_roots`)."""
    step = (high - low) / _POINTWISE_SAMPLES
    before_point, before = None, None
    for index in range(_POINTWISE_SAMPLES + 1):
        point = low + index * step
        here = value(point)
        if here == 0:
            return point
        if here is not None and before is not None and (here < 0) != (before < 0):
            root = _bisected_points(value, before_point, point, before)
            if root is not None:
                return root
        if here is not None:
            before_point, before = point, here
    return None


def _bisected_points(
    value: Callable[[float], float | None], start: float, end: float, before: float
) -> float | None:
    """The root of `value` in `[start, end]`, where its sign changes, or None."""
    scale = abs(before)
    for _ in range(80):
        middle = (start + end) / 2
        if middle in (start, end):
            break
        here = value(middle)
        if here is None:
            return None
        if here == 0:
            return middle
        if (here < 0) == (before < 0):
            start, before = middle, here
        else:
            end = middle
        scale = max(scale, abs(here))
    middle = (start + end) / 2
    here = value(middle)
    if here is None or abs(here) > scale * 1e-6:
        return None
    return middle


def _number_polynomial(residual: sp.Basic, variable: sp.Symbol) -> sp.Poly | None:
    """`residual` as a polynomial in `variable` of degree at least 1 with number
    coefficients, or None."""
    polynomial = _polynomial(residual, variable)
    if polynomial is None or polynomial.degree() < 1:
        return None
    if any(not coefficient.is_number for coefficient in polynomial.all_coeffs()):
        return None
    return polynomial


def _polynomial_roots(
    polynomial: sp.Poly, context: Context, real: bool
) -> list[sp.Basic]:
    """Every distinct root of a polynomial with number coefficients, approximated, or
    its real roots with `real`.

    The polynomial is made square-free and factored exactly first, so a multiple root
    is one root and a linear factor gives its root exactly: `3*x - 4` gives 4/3. Real
    roots of a factor with real coefficients are isolated exactly before they are
    refined, so they come out real however close together they lie, and complex ones
    come in conjugate pairs.
    """
    try:
        square_free = polynomial.sqf_part()
        _, factors = square_free.factor_list()
    except Exception:
        factors = [(polynomial, 1)]
    found: list[sp.Basic] = []
    for factor, _ in factors:
        if factor.degree() < 1:
            continue
        if factor.degree() == 1:
            slope, offset = factor.all_coeffs()
            root = -offset / slope
            if real and root.is_extended_real is not True:
                continue
            exact = root.is_Rational or root in (sp.I, -sp.I)
            found.append(root if exact else _approximate_number(root, context))
            continue
        found.extend(_approximate_roots(factor, context, real))
    return _ordered(found)


def _approximate_roots(
    factor: sp.Poly, context: Context, real: bool
) -> list[sp.Basic]:
    """The roots of a square-free polynomial of degree 2 or more, approximated.

    The roots are refined together at a working precision well past the digits shown.
    With real coefficients the number of real roots is counted exactly first (Sturm),
    and that many roots nearest the real axis are the real ones, so a real root never
    gains an imaginary part however close the roots lie; the others are written as
    conjugate pairs.
    """
    digits = context.precision_digits + GUARD
    working = 2 * digits + 20
    coefficients = factor.all_coeffs()
    with mpmath.workdps(working):
        try:
            values = [
                mpmath.mpc(*(mpmath.mpf(str(part.evalf(working))) for part in parts))
                for parts in (c.as_real_imag() for c in coefficients)
            ]
            roots = mpmath.polyroots(values, maxsteps=200, extraprec=4 * working)
        except Exception:
            return []
        found: list[sp.Basic] = []
        if all(coefficient.is_extended_real for coefficient in coefficients):
            try:
                count = _real_count(factor, coefficients, working)
            except Exception:
                return []
            ordered = sorted(roots, key=lambda root: abs(mpmath.im(root)))
            for root in ordered[:count]:
                found.append(_approximate_number(_mp(mpmath.re(root)), context))
            if not real:
                for root in ordered[count:]:
                    if mpmath.im(root) > 0:
                        value = _approximate_number(_mp(root), context)
                        found += [value, sp.conjugate(value)]
            return found
        for root in roots:
            number = _approximate_number(_mp(root), context)
            if real and sp.im(number) != 0:
                continue
            found.append(number)
        return found


def _real_count(factor: sp.Poly, coefficients: list[sp.Basic], working: int) -> int:
    """The number of real roots of a square-free polynomial with real coefficients,
    counted exactly; irrational coefficients are first replaced by rationals agreeing
    to `working` digits."""
    if factor.domain not in (sp.ZZ, sp.QQ):
        factor = sp.Poly(
            [sp.Rational(str(c.evalf(working))) for c in coefficients], factor.gen
        )
    return factor.count_roots()


def _mp(value) -> sp.Basic:
    """An mpmath number as a sympy one, keeping its digits."""
    if isinstance(value, mpmath.mpc):
        return sp.Float(str(value.real), mpmath.mp.dps) + sp.I * sp.Float(
            str(value.imag), mpmath.mp.dps
        )
    return sp.Float(str(value), mpmath.mp.dps)


def _approximate_number(value: sp.Basic, context: Context) -> sp.Basic:
    """A number, real or complex, approximated part by part; a part far below the
    other's digits, the noise of evaluating a real or an imaginary value, is 0."""
    digits = context.precision_digits
    number = sp.sympify(value).evalf(digits + GUARD)
    re, im = sp.re(number), sp.im(number)
    size = max(abs(re), abs(im))
    if size == 0:
        return sp.S.Zero
    small = size * sp.Float(10) ** -(digits + GUARD)
    parts = [
        sp.S.Zero if abs(part) <= small else simplest_rational(part, digits)
        for part in (re, im)
    ]
    return parts[0] + parts[1] * sp.I


def _real_roots(
    residual: sp.Basic,
    variable: sp.Symbol,
    low: sp.Basic,
    high: sp.Basic,
    context: Context,
) -> list[sp.Basic]:
    polynomial = _number_polynomial(residual, variable)
    if polynomial is not None:
        # Each distinct root once, collapsing multiplicity as the exact path does:
        # `(x - 1)^2` has the single solution 1.
        return [
            root
            for root in _polynomial_roots(polynomial, context, True)
            if _within(root, low, high)
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
    return _scanned(residual, variable, first, last, context, _RESCANS, _EDGE_RESCANS)


def _scanned(
    residual: sp.Basic,
    variable: sp.Symbol,
    first: float,
    last: float,
    context: Context,
    depth: int,
    edges: int = 0,
) -> sp.Basic | None:
    """One root in `[first, last]` by a scan of `_SCAN` pieces, or None.

    A point where the residual is undefined tells nothing about the sign beside it, so
    a root next to it is missed: `ATAN(z - 0.1/z) + ATAN(z + 0.1/z) = 0.5` is undefined
    at 0 and has its root at 0.28, short of the next point. Once the whole interval is
    scanned, a piece with one end undefined is scanned again, `depth` levels deep. A
    piece undefined at both ends is not, nor one whose sign change bisects to no root:
    rescanning every pole of `1/SIN(x)` makes the widening search far too slow.

    A piece at an end of the interval where the residual is undefined is scanned again
    up to `edges` levels deep: a single piece per level, the one next to the end.
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
    for start, end in suspect:
        at_edge = start == first or end == last
        if not (depth or (edges and at_edge)):
            continue
        found = _scanned(
            residual,
            variable,
            start,
            end,
            context,
            max(depth - 1, 0),
            edges - 1 if at_edge and edges else 0,
        )
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
    """The root of `residual` between `low` and `high`, where its sign changes, or None
    if it changes sign without vanishing there, as at a pole.

    A root is where the residual is zero to the working precision, measured against
    the size of its terms there: `PHI^n/SQRT(5) - 10^100` is zero at its root to a
    hundred digits less than its terms. A residual that can't be evaluated in floating
    point, such as a conditional in a user's function, is bisected by evaluating it
    at each point (`_bisected_by_value`).
    """
    digits = context.precision_digits + GUARD
    # Each step halves the bracket: enough steps to narrow it to the digits wanted.
    width = max(abs(low), abs(high), 1.0)
    steps = int((digits + 5) * math.log2(10) + math.log2(width)) + 20
    try:
        found = sp.nsolve(
            residual,
            variable,
            (low, high),
            solver="bisect",
            prec=digits,
            verify=False,
            maxsteps=steps,
        )
    except Exception:
        return _bisected_by_value(residual, variable, low, high, context)
    if not _vanishing(residual, variable, found, digits) and not _dips(
        residual, variable, found, digits
    ):
        return None
    return _approximate(found, context)


def _dips(residual: sp.Basic, variable: sp.Symbol, point, digits: int) -> bool:
    """Whether `residual` is smaller at `point` than a little to either side, as at a
    root and unlike at a pole. This accepts a root where the residual is ill
    conditioned, `SIN(x^2) = #e^x` far from 0, whose value at the root is far above
    the precision of its terms.

    Nor where its denominator dips too: a residual divided by `x - r`, `r` a root found
    before and rounded, has a root within the rounding of `r`, the same root again.
    """
    try:
        step = sp.Float(10, digits) ** (-(digits // 2)) * max(1, abs(sp.Float(point)))
        denominator = sp.fraction(sp.together(residual))[1]

        def dips(expression: sp.Basic) -> bool:
            left, middle, right = (
                abs(complex(expression.evalf(digits, subs={variable: point + offset})))
                for offset in (-step, 0, step)
            )
            return middle * 100 < min(left, right)

        if denominator.has(variable) and dips(denominator):
            return False
        return dips(residual)
    except Exception:
        return False


def _bisected_by_value(
    residual: sp.Basic, variable: sp.Symbol, low: float, high: float, context: Context
) -> sp.Basic | None:
    """The root between `low` and `high` of a residual evaluated by substitution, or
    None. The bracket is halved until it is narrower than the digits wanted; it holds a
    root rather than a pole if the residual there is small next to its values at the
    ends."""
    digits = context.precision_digits + GUARD
    with mpmath.workdps(digits + 10):
        start, end = mpmath.mpf(low), mpmath.mpf(high)
        values = [_value_at(residual, variable, point, digits) for point in (start, end)]
        if None in values or (values[0] < 0) == (values[1] < 0):
            return None
        scale = max(abs(values[0]), abs(values[1]))
        before = values[0]
        tolerance = mpmath.mpf(10) ** (-digits) * max(1, abs(start), abs(end))
        middle = start
        for _ in range(int((digits + 5) * math.log2(10)) + 60):
            if end - start <= tolerance:
                break
            middle = (start + end) / 2
            value = _value_at(residual, variable, middle, digits)
            if value is None:
                return None
            if value == 0:
                break
            if (value < 0) == (before < 0):
                start, before = middle, value
            else:
                end = middle
        middle = (start + end) / 2
        value = _value_at(residual, variable, middle, digits)
        if value is None or abs(value) > scale * mpmath.mpf(10) ** (2 - digits // 2):
            return None
        return _approximate(sp.Float(middle, digits + 10), context)


def _value_at(
    residual: sp.Basic, variable: sp.Symbol, point, digits: int
) -> mpmath.mpf | None:
    """The real value of `residual` at `point`, or None.

    The point is substituted as the exact rational it is, so a conditional evaluated
    at it computes exactly and only the result is rounded, to `digits` digits.
    """
    try:
        sign, mantissa, exponent, _ = mpmath.mpf(point)._mpf_
        exact = (-1) ** sign * sp.Integer(mantissa) * sp.Integer(2) ** exponent
        value = residual.subs(variable, exact).evalf(digits)
        number = complex(value)
    except Exception:
        return None
    if number.imag and abs(number.imag) > 1e-12 * max(1.0, abs(number.real)):
        return None
    try:
        return mpmath.mpf(sp.re(value).evalf(digits))
    except Exception:
        return None


def _vanishing(residual: sp.Basic, variable: sp.Symbol, point, digits: int) -> bool:
    """Whether `residual` is zero at `point` to `digits` digits of its largest term."""
    try:
        terms = sp.Add.make_args(residual)
        values = [
            abs(complex(term.evalf(digits, subs={variable: point}))) for term in terms
        ]
        total = abs(complex(residual.evalf(digits, subs={variable: point})))
    except Exception:
        return False
    scale = max(values) if len(values) > 1 else 1.0
    return total <= scale * 10.0 ** (2 - digits)


def _searched(
    residual: sp.Basic, variable: sp.Symbol, context: Context, real: bool = True
) -> tuple[sp.Basic, ...] | None:
    """The last resort of Mixed and Approximate modes: every root of a polynomial with
    number coefficients (the real ones with `real`), and of anything else the first
    real root of `[-10, 10]`, doubling the interval until one turns up, as the manual
    describes for Mixed. The cap is ours.
    """
    polynomial = _number_polynomial(residual, variable)
    if polynomial is not None:
        roots = _polynomial_roots(polynomial, context, real)
        if not roots:
            return None
        return tuple(_equated(variable, root, context) for root in roots)
    reach = _REACH
    while reach <= _REACH_CAP:
        found = _real_roots(
            residual, variable, sp.Integer(-reach), sp.Integer(reach), context
        )
        if found:
            return tuple(_equated(variable, root, context) for root in found)
        reach *= 2
    return None


#: The most interval enclosures one search may compute.
_EXPLORATION = 1000


@dataclass(frozen=True)
class _Search:
    """What `_explored` found: a root, or whether it proved that there is none."""

    root: sp.Basic | None
    rootless: bool


def _explored(
    residual: sp.Basic,
    variable: sp.Symbol,
    first: float,
    last: float,
    context: Context,
) -> _Search:
    """A root of `residual` in `[first, last]` found by interval arithmetic, or the proof
    that it has none there.

    The interval is cut in halves, leftmost first. A piece on which the enclosure of
    the residual's values excludes zero holds no root and is dropped; the sign of the
    residual is known at each cut, and a piece whose ends differ in sign is bisected.
    So two roots closer together than a scan's step, `SIN(x^2) = #e^x` near -6.86, are
    told apart once the pieces are narrow enough to put a cut between them. If every
    piece is dropped, there is no root; a piece that gets too narrow, a residual that
    has no enclosure, or running out of `_EXPLORATION` proves nothing. An infinite end
    is reached by pieces that double in length.
    """
    enclose = _enclosing(residual, variable)
    if enclose is None:
        return _Search(None, False)
    budget = _EXPLORATION
    complete = True
    pieces = [
        (first, last, _sign_at(enclose, first), _sign_at(enclose, last))
    ]
    while pieces:
        low, high, below, above = pieces.pop()
        # An end proved no root is left out, so a jump there, as of `SIGN(z)` at 0,
        # doesn't keep the piece from being dropped.
        inner = (
            math.nextafter(low, math.inf) if below not in (0, None) else low,
            math.nextafter(high, -math.inf) if above not in (0, None) else high,
        )
        values = _enclosed(enclose, *inner) if inner[0] <= inner[1] else None
        budget -= 1
        if budget < 0:
            return _Search(None, False)
        if values is not None and _excludes_zero(values):
            continue
        if below in (1, -1) and above in (1, -1) and below != above:
            root = _bisect(residual, variable, low, high, context)
            if root is not None:
                return _Search(root, False)
        middle = _cut(low, high)
        if middle is None:
            complete = False
            continue
        sign = _sign_at(enclose, middle)
        if sign == 0:
            root = _polished(residual, variable, middle, low, high, context)
            if root is not None:
                return _Search(root, False)
        pieces.append((middle, high, sign, above))
        pieces.append((low, middle, below, sign))
    return _Search(None, complete)


def _cut(low: float, high: float) -> float | None:
    """Where to cut `[low, high]` in two, or None if it is too narrow to cut. A piece
    with an infinite end is cut so that its finite part doubles."""
    if math.isinf(low) and math.isinf(high):
        return 0.0
    if math.isinf(high):
        return low + max(1.0, abs(low))
    if math.isinf(low):
        return high - max(1.0, abs(high))
    middle = (low + high) / 2
    if not low < middle < high or high - low <= 1e-13 * max(1.0, abs(middle)):
        return None
    return middle


def _sign_at(enclose: Callable, point: float) -> int | None:
    """The sign of the residual at `point` as its enclosure there proves it: 0 if the
    enclosure holds zero, 2 if the value is proved not real, None if there is none."""
    if math.isinf(point):
        return None
    values = _enclosed(enclose, point, point)
    if values is None:
        return None
    real, imaginary = values
    if imaginary is not None and not _holds_zero(imaginary):
        return 2
    if real.a > 0:
        return 1
    if real.b < 0:
        return -1
    return 0


def _polished(
    residual: sp.Basic,
    variable: sp.Symbol,
    point: float,
    low: float,
    high: float,
    context: Context,
) -> sp.Basic | None:
    """The root near `point`, where the residual's enclosure holds zero, refined by
    Newton's method, or None if that leaves `[low, high]` or finds no root."""
    digits = context.precision_digits + GUARD
    try:
        found = sp.nsolve(residual, variable, point, prec=digits, verify=False)
    except Exception:
        return None
    value = _number(found)
    if value is None or not low <= value <= high:
        return None
    if not _vanishing(residual, variable, found, digits):
        return None
    return _approximate(found, context)


def _excludes_zero(values) -> bool:
    """Whether an enclosure `(real, imaginary)` proves the value nonzero."""
    real, imaginary = values
    return not _holds_zero(real) or (imaginary is not None and not _holds_zero(imaginary))


def _holds_zero(interval) -> bool:
    return not (interval.a > 0 or interval.b < 0)


def _enclosed(enclose: Callable, low: float, high: float):
    """The enclosure of the residual's values over `[low, high]` as a pair of intervals,
    its real part and its imaginary part (None for a real value), or None if the
    residual is undefined somewhere in it or the arithmetic fails."""
    try:
        real, imaginary = enclose(mpmath.iv.mpf([low, high]))
    except Exception:
        return None
    for part in (real, imaginary):
        if part is None:
            continue
        if not isinstance(part, mpmath.ctx_iv.ivmpf):
            return None
        if mpmath.isnan(part.a) or mpmath.isnan(part.b):
            return None
    return real, imaginary


class _NoEnclosure(Exception):
    """A residual with a part interval arithmetic has no rule for."""


def _enclosing(residual: sp.Basic, variable: sp.Symbol) -> Callable | None:
    """A function from an interval of `variable` to an enclosure of every value of
    `residual` on it, its real and imaginary parts, or None if a part of it has no
    such rule.

    The rules are those of `mpmath.iv`, which rounds outwards, so the result is a
    proof. A constant is enclosed by evaluating it to more digits than the interval
    arithmetic carries. A power of a negative base is complex, `x^(1/3)` being
    `ABS(x)^(1/3)*#e^(#i*pi/3)` there, so `x^(2/3) - 2*x^(1/3) = 35` is proved to
    have no root below 0 by its imaginary part. A function of a complex value, a power
    of a base whose values have both signs and `LN` of one not positive have no rule;
    they raise, and the piece they are on is then not dropped.
    """
    try:
        return _enclosure(residual, variable)
    except Exception:
        return None


def _enclosure(expression: sp.Basic, variable: sp.Symbol) -> Callable:
    iv = mpmath.iv
    if not expression.has(variable):
        constant = _constant_enclosure(expression)
        return lambda box: constant
    if expression == variable:
        return lambda box: (box, None)
    if isinstance(expression, (sp.Add, sp.Mul)):
        parts = [_enclosure(part, variable) for part in expression.args]
        operation = _sum if isinstance(expression, sp.Add) else _product

        def combined(box):
            total = parts[0](box)
            for part in parts[1:]:
                total = operation(total, part(box))
            return total

        return combined
    if isinstance(expression, sp.Pow):
        base, exponent = expression.args
        lower = _enclosure(base, variable)
        if exponent.is_Integer:
            power = int(exponent)
            return lambda box: _integer_power(lower(box), power)
        if exponent.is_Rational:
            turn = _constant_enclosure(exponent)[0] * iv.pi
            return lambda box: _rational_power(_real(lower(box)), exponent, turn)
        upper = _enclosure(exponent, variable)
        return lambda box: (_real_power(_real(lower(box)), _real(upper(box))), None)
    if isinstance(expression, sp.Abs):
        inner = _enclosure(expression.args[0], variable)
        return lambda box: (_absolute(inner(box)), None)
    if isinstance(expression, sp.sign):
        inner = _enclosure(expression.args[0], variable)
        return lambda box: (_signs(_real(inner(box))), None)
    calls = {
        sp.exp: iv.exp,
        sp.sin: iv.sin,
        sp.cos: iv.cos,
        sp.tan: iv.tan,
        sp.atan: lambda values: iv.atan2(values, iv.mpf(1)),
        sp.log: _logarithm,
    }
    call = calls.get(type(expression))
    if call is None or len(expression.args) != 1:
        raise _NoEnclosure(type(expression).__name__)
    inner = _enclosure(expression.args[0], variable)
    return lambda box: (call(_real(inner(box))), None)


def _constant_enclosure(constant: sp.Basic):
    """An enclosure of a constant, real or complex."""
    iv = mpmath.iv
    if constant.is_Rational:
        return iv.mpf(int(constant.p)) / iv.mpf(int(constant.q)), None
    value = sp.sympify(constant).evalf(30, strict=True)
    real, imaginary = value.as_real_imag()
    if not (real.is_Number and imaginary.is_Number):
        raise _NoEnclosure("not a number")

    def interval(part: sp.Basic):
        number = mpmath.mpf(str(part))
        radius = abs(number) * mpmath.mpf(10) ** -25 + mpmath.mpf(10) ** -300
        return iv.mpf([number - radius, number + radius])

    return interval(real), None if imaginary == 0 else interval(imaginary)


def _real(values):
    """The real interval of an enclosure that is real."""
    real, imaginary = values
    if imaginary is not None:
        raise ValueError("a function of a complex value")
    return real


def _sum(left, right):
    (a, b), (c, d) = left, right
    imaginary = b if d is None else d if b is None else b + d
    return a + c, imaginary


def _product(left, right):
    (a, b), (c, d) = left, right
    if b is None and d is None:
        return a * c, None
    if b is None:
        return a * c, a * d
    if d is None:
        return a * c, b * c
    return a * c - b * d, a * d + b * c


def _integer_power(values, power: int):
    real, imaginary = values
    if imaginary is None:
        return real**power, None
    if power < 0:
        size = real**2 + imaginary**2
        values = (real / size, -imaginary / size)
        power = -power
    result = (mpmath.iv.mpf(1), None)
    for _ in range(power):
        result = _product(result, values)
    return result


def _rational_power(base, exponent: sp.Rational, turn):
    """`base^exponent` on the principal branch, for a real base of one sign: the turn
    of a negative base is `exponent*pi`."""
    iv = mpmath.iv
    if base.a >= 0:
        return _real_power(base, iv.mpf(int(exponent.p)) / iv.mpf(int(exponent.q))), None
    if base.b > 0:
        raise ValueError("a power of a value of either sign")
    size = _real_power(-base, iv.mpf(int(exponent.p)) / iv.mpf(int(exponent.q)))
    return size * iv.cos(turn), size * iv.sin(turn)


def _real_power(base, exponent):
    """`base^exponent` over intervals, for a base whose values are nonnegative."""
    iv = mpmath.iv
    if base.a > 0:
        return iv.exp(exponent * iv.log(base))
    if base.a == 0 and exponent.a > 0:
        top = iv.exp(exponent * iv.log(iv.mpf(base.b))) if base.b > 0 else iv.mpf(0)
        return iv.mpf([0, top.b])
    raise ValueError("a power of a value that may be negative")


def _signs(values):
    """The signs of the values in an interval."""
    low = 1 if values.a > 0 else 0 if values.a == 0 else -1
    high = -1 if values.b < 0 else 0 if values.b == 0 else 1
    return mpmath.iv.mpf([low, high])


def _logarithm(argument):
    if argument.a > 0:
        return mpmath.iv.log(argument)
    raise ValueError("a logarithm of a value that may not be positive")


def _absolute(values):
    real, imaginary = values
    if imaginary is not None:
        return mpmath.iv.sqrt(real**2 + imaginary**2)
    if real.a >= 0:
        return real
    if real.b <= 0:
        return -real
    return mpmath.iv.mpf([0, max(-real.a, real.b)])


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


def _imaginary(value: sp.Basic) -> bool:
    """Whether `value` is proved to have an imaginary component.

    For a number, its imaginary part is evaluated with the error bounded, and found
    nonzero beyond the bound: that is a proof. Ferrari's real roots, written with `#i`
    terms that cancel, evaluate to an imaginary part no larger than its error, so they
    are not imaginary; a value whose imaginary part can't be decided either way is
    kept. A parametric value must be proved non-real by sympy.
    """
    try:
        if value.free_symbols:
            return value.is_extended_real is False
        if value.is_extended_real:
            return False
        part = sp.im(value)
        if part.is_zero is not None:
            return part.is_zero is False
        return part.evalf(_PROVING, strict=True) != 0
    except Exception:
        return False


#: The digits to which an imaginary part is evaluated to prove it nonzero.
_PROVING = 30


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


# -- the order of solutions ------------------------------------------------------------


def listed_in_order(rows: Sequence[tuple[sp.Basic, ...]]) -> list[tuple[sp.Basic, ...]]:
    """The solutions of `SOLUTIONS` (one value per variable in each row) in the order
    Derive lists them: programs take the k-th solution, so the order is part of the
    answer. The rows are ordered lexicographically by their values, each variable's
    values in `_value_order`. `SOLVE` writes its disjunction in the reverse order."""
    rows = list(dict.fromkeys(rows))
    if len(rows) < 2:
        return rows
    width = len(rows[0])
    places = []
    for column in range(width):
        values = [row[column] if column < len(row) else sp.S.Zero for row in rows]
        ordered = _value_order(values)
        places.append({value: place for place, value in enumerate(ordered)})
    return sorted(
        rows,
        key=lambda row: tuple(
            places[column].get(value, 0) for column, value in enumerate(row[:width])
        ),
    )


def _value_order(values: Iterable[sp.Basic]) -> list[sp.Basic]:
    """Values in Derive's order: rational numbers first, by denominator, then by
    absolute value, the positive before the negative (`[0, 1, -1, 1/2, -2/3]`); then
    the others, a value next to its partner (the conjugate of a complex number, the
    other sign of a square root: `SQRT(5) - 1` and `-SQRT(5) - 1`), real ones before
    complex ones and those before symbolic ones, groups by size, and in a pair the one
    with the radical or the imaginary part added first.

    Two kinds of roots have an order of their own. Pairs of opposite numbers, the roots
    of a polynomial in `x^2`, come as `_opposites_by_square` says. The three real roots
    of an irreducible cubic, written with the arc cosine of one number, come as Derive's
    trigonometric formula lists them (see `_cubic_order`)."""
    distinct = list(dict.fromkeys(values))
    rational = sorted(
        (value for value in distinct if value.is_Rational),
        key=lambda value: (value.q, abs(value.p), value.p < 0),
    )
    rest = [value for value in distinct if not value.is_Rational]
    groups: list[list[sp.Basic]] = []
    paired: set[int] = set()
    for index, value in enumerate(rest):
        if index in paired:
            continue
        group = [value]
        for other in range(index + 1, len(rest)):
            if other not in paired and _partners(value, rest[other]):
                paired.add(other)
                group = _plus_first(value, rest[other])
                break
        groups.append(group)
    groups.sort(key=_group_rank)
    groups = _opposites_by_square(groups)
    ordered = [value for group in groups for value in group]
    return rational + _cubic_order(ordered)


def _opposites_by_square(groups: list[list[sp.Basic]]) -> list[list[sp.Basic]]:
    """`groups` with the pairs of opposite numbers among them, in the places they hold,
    in Derive's order: real before imaginary; pairs whose square is rational by the
    square-free part of the square, the smallest first (`3*SQRT(5)` before `SQRT(10)`,
    `SQRT(3)` before `SQRT(30)`), then the others by their square, the largest first."""
    keys: dict[int, tuple] = {}
    for index, group in enumerate(groups):
        if len(group) != 2:
            continue
        one, other = (_approximately(value) for value in group)
        if one is None or other is None:
            continue
        if abs(one + other) > 1e-9 * max(abs(one), 1e-300):
            continue
        square = (one * one).real
        exact = sp.expand(group[0] ** 2)
        if exact.is_Rational:
            free = abs(sp.Rational(exact.p * exact.q))
            for prime, power in sp.factorint(free).items():
                free /= prime ** (power - power % 2)
            keys[index] = (square < 0, 0, float(free))
        else:
            keys[index] = (square < 0, 1, -round(square, 9))
    if len(keys) < 2:
        return groups
    places = list(keys)
    pairs = sorted(places, key=lambda index: keys[index])
    reordered = list(groups)
    for place, index in zip(places, pairs):
        reordered[place] = groups[index]
    return reordered


def _cubic_order(values: list[sp.Basic]) -> list[sp.Basic]:
    """`values` with the three real roots of an irreducible cubic among them in the order
    of Derive's trigonometric formula, or unchanged.

    The roots are recognized by their form, the arc cosine of one number in each, as the
    trigonometric formula writes them. With `t` the roots less their mean, the roots of
    the depressed cubic `t^3 + p*t + q`, Derive lists the largest, the smallest and the
    middle one where `q < 0`, and the middle, the smallest and the largest one where
    `q > 0`: the eigenvalues of `[[1, 0.7, 0.5], [0.7, 1, 0.9], [0.5, 0.9, 1]]` and the
    solutions of `2*z^3 - z^2 - z + 1/4 = 0`.
    """
    arcs = [value for value in values if value.has(sp.acos)]
    if len(arcs) != 3:
        return values
    arguments = {call.args[0] for value in arcs for call in value.atoms(sp.acos)}
    numbers = [_approximately(value) for value in arcs]
    if len(arguments) != 1 or any(
        number is None or _nonreal(number) for number in numbers
    ):
        return values
    reals = [number.real for number in numbers]
    mean = sum(reals) / 3
    q = -math.prod(real - mean for real in reals)
    smallest, middle, largest = sorted(range(3), key=lambda index: reals[index])
    order = [largest, smallest, middle] if q < 0 else [middle, smallest, largest]
    cubic = iter(arcs[index] for index in order)
    return [next(cubic) if value in arcs else value for value in values]


def _approximately(value: sp.Basic) -> complex | None:
    """`value` as a float, to pick an order (never to decide a fact about it)."""
    if not value.is_number:
        return None
    try:
        return complex(sp.N(value, 15))
    except Exception:
        return None


def _nonreal(number: complex) -> bool:
    """Whether an approximate value has an imaginary part beyond rounding noise."""
    return abs(number.imag) > 1e-12 * abs(number)


def _radicals(expression: sp.Basic) -> int:
    return sum(
        1
        for power in expression.atoms(sp.Pow)
        if power.exp.is_Rational and not power.exp.is_Integer
    )


def _partners(one: sp.Basic, other: sp.Basic) -> bool:
    """Whether two solutions are the two signs of one root: complex conjugates or
    negatives of each other, or values whose sum has fewer radicals than either."""
    first, second = _approximately(one), _approximately(other)
    if first is not None and second is not None and (
        _nonreal(first) or _nonreal(second)
    ):
        scale = max(abs(first), abs(second), 1e-300)
        return (
            abs(first - second.conjugate()) <= 1e-9 * scale
            or abs(first + second) <= 1e-9 * scale
        )
    if sp.count_ops(one) + sp.count_ops(other) > _PARTNER_SIZE:
        return False
    try:
        together = sp.expand(one + other)
    except Exception:
        return False
    return _radicals(together) < min(_radicals(one), _radicals(other))


#: The largest pair of solutions, in operations, compared as partners by expanding
#: their sum; larger ones (a quartic's roots) are left unpaired.
_PARTNER_SIZE = 400


def _plus_first(one: sp.Basic, other: sp.Basic) -> list[sp.Basic]:
    """A pair of partners with the added radical or imaginary part first: the larger
    real part, then the larger imaginary part; for symbolic values the one from which
    the other is subtracted without a leading minus. Where that difference divides by a
    variable, whose sign is unknown, Derive's recorded results (the inflection points
    of a general quartic) follow neither sign, and the pair is in canonical order."""
    first, second = _approximately(one), _approximately(other)
    if first is not None and second is not None:
        if _plus_rank(first) <= _plus_rank(second):
            return [one, other]
        return [other, one]
    try:
        difference = sp.expand(one - other)
        if sp.fraction(sp.together(difference))[1].free_symbols:
            return sorted((one, other), key=sp.default_sort_key)
        negative = difference.could_extract_minus_sign()
    except Exception:
        negative = False
    return [other, one] if negative else [one, other]


def _plus_rank(number: complex) -> tuple[float, float]:
    return (-round(number.real, 12), -round(number.imag, 12))


def _group_rank(group: list[sp.Basic]) -> tuple:
    numbers = [_approximately(value) for value in group]
    if any(number is None for number in numbers):
        return (3, 0.0, sp.default_sort_key(group[0]))
    tier = 2 if any(_nonreal(number) for number in numbers) else 1
    size = max(abs(number) for number in numbers)
    return (tier, round(size, 9), ())
