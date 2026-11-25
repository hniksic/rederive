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
    InOrder,
    Modulo,
    PlusMinus,
    Vector,
)

__all__ = [
    "Outcome",
    "arbitrary",
    "between",
    "confined",
    "equations_of",
    "listed_in_order",
    "numeric_roots",
    "oriented",
    "outcome",
    "parameter_condition",
    "solved_range",
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
    used, one per equation. `bounds` confines the solutions of one equation or
    inequality: an equation is searched for numerically, and an identity or a range is
    cut to them. None leaves the search to the precision mode. `divisors` are what the
    equation divided by before Simplify cancelled anything: a value making one zero is
    not a solution. `real` drops solutions known to have an imaginary component.

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
        return _system(equations, context, variables, constants, real, bounds)
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
    With `bounds` their ranges are cut to them.
    """
    chosen = _chosen(expression, context, variables, 1)
    if not chosen:
        return _unsolvable(expression, context)
    variable = chosen[0]
    if isinstance(expression, Relational) and not isinstance(expression, sp.Equality):
        found = _inequality(expression, variable, context, constants)
        if bounds is None:
            return found
        return _bounded_ranges(found, variable, bounds, context, constants)
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

    With bounds an identity is every value in them, and anything else is searched for
    numerically (Approximate). Otherwise an exact solution is tried first, and Mixed
    and Approximate then search for a real root over a widening interval: under
    Approximate, `SOLVE(ERF(x) + x = 1, x)` is `[x = 0.489116]`.

    A solution written with a function the notation has no name for is no answer (see
    `solutions`), so under Approximate `SOLVE(ERF(x) = 0.3, x)` is searched for too.

    A candidate is dropped only if provably not a solution: it zeros a divisor of the
    original equation (`_divides_by_zero`), substituting it back fails (`_fails`), or,
    with `real`, it is known to be non-real (`_imaginary`). If nothing remains, the
    answer is "no solutions found".
    """
    if residual.is_zero or _cancels(residual, variable):
        if bounds is not None and bounds != (-sp.oo, sp.oo):
            return _everything_within(variable, bounds, context, constants)
        return (_equated(variable, constants.fresh(), context),)
    if bounds is not None:
        return _numeric(residual, variable, bounds[0], bounds[1], context)
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


def confined(
    entry: sp.Basic | None,
    variable: sp.Symbol,
    bounds: tuple[sp.Basic, sp.Basic],
    context: Context,
    constants: _Constants | None = None,
) -> tuple[sp.Basic, ...] | None:
    """The entries of a range or value of `variable` cut to `[low, high]`, None for
    `entry` standing for every value: `x > 2` in `[0, 5]` is `2 < x <= 5`, and every
    value in `[1, 3]` is `1 <= x <= 3`. A value or a range that ends at a parameter, or
    an end that can't be compared with a bound exactly, gives None. Every value between
    infinite bounds is `x = @n`, from `constants`."""
    low, high = bounds
    try:
        found = sp.Interval(low, high)
        if entry is not None:
            region = _region(entry, variable)
            if region is None:
                return None
            found = region.intersect(found)
    except Exception:
        return None
    if found.has(sp.Min, sp.Max):
        return None
    constants = constants or _Constants(context.arbitrary_index)
    entries = _ranges(found, variable, context, constants)
    if entries is None:
        return None
    # A range between two ends is written from its lower end, beside other relations
    # too: `2 < x AND x <= 5 AND a = b`.
    return tuple(
        InOrder(*sorted(entry.args, key=lambda relation: relation.lhs == variable))
        if isinstance(entry, sp.And)
        else entry
        for entry in entries
    )


def between(value: sp.Basic, bounds: tuple[sp.Basic, sp.Basic]) -> list[sp.Basic]:
    """The relations that put `value` in `[low, high]`, `low <= value` and `value <=
    high`, without one at an infinite bound."""
    low, high = bounds
    return [
        relation
        for relation in (
            sp.Le(low, value, evaluate=False),
            sp.Le(value, high, evaluate=False),
        )
        if not relation.has(sp.oo, -sp.oo)
    ]


def _bounded_ranges(
    entries: tuple[sp.Basic, ...],
    variable: sp.Symbol,
    bounds: tuple[sp.Basic, sp.Basic],
    context: Context,
    constants: _Constants,
) -> tuple[sp.Basic, ...]:
    """The entries solving an inequality cut to the bounds. A range that can't be cut
    exactly holds where it is inside them; a relation left unsolved stays as it is."""
    bounded: list[sp.Basic] = []
    for entry in entries:
        if _is_identity(entry, variable, constants):
            cut = (
                (entry,)
                if bounds == (-sp.oo, sp.oo)
                else _everything_within(variable, bounds, context, constants)
            )
        elif solved_range(entry, variable):
            cut = confined(entry, variable, bounds, context, constants)
            if cut is None:
                cut = (InOrder(*sp.And.make_args(entry), *between(variable, bounds)),)
        else:
            cut = (entry,)
        bounded.extend(part for part in cut or () if part not in bounded)
    return tuple(bounded)


def _everything_within(
    variable: sp.Symbol,
    bounds: tuple[sp.Basic, sp.Basic],
    context: Context,
    constants: _Constants,
) -> tuple[sp.Basic, ...]:
    """Every value of `variable` in `[low, high]`, as the range of the bounds: `1 <= x
    <= 3`."""
    cut = confined(None, variable, bounds, context, constants)
    return (InOrder(*between(variable, bounds)),) if cut is None else cut


def solved_range(entry: sp.Basic, variable: sp.Symbol) -> bool:
    """Whether `entry` is a value or a range of `variable` as solving writes one: each
    relation has the variable alone on one side, `x > a`, `-2 < x < 2`, `x /= 2`."""
    return all(
        isinstance(relation, Relational)
        and (
            relation.lhs == variable
            and not relation.rhs.has(variable)
            or relation.rhs == variable
            and not relation.lhs.has(variable)
        )
        for relation in sp.And.make_args(entry)
    )


def _region(entry: sp.Basic, variable: sp.Symbol) -> sp.Set | None:
    """The real values of `variable` a value or a range of it with number ends holds
    at, or None."""
    region: sp.Set = sp.S.Reals
    for relation in sp.And.make_args(entry):
        if not isinstance(relation, Relational):
            return None
        if relation.rhs == variable:
            relation = relation.reversed
        end = relation.rhs
        if relation.lhs != variable or end.free_symbols or not end.is_extended_real:
            return None
        if isinstance(relation, sp.Equality):
            part: sp.Set = sp.FiniteSet(end)
        elif isinstance(relation, sp.Unequality):
            part = sp.Union(sp.Interval.open(-sp.oo, end), sp.Interval.open(end, sp.oo))
        elif isinstance(relation, sp.StrictLessThan):
            part = sp.Interval.open(-sp.oo, end)
        elif isinstance(relation, sp.LessThan):
            part = sp.Interval(-sp.oo, end)
        elif isinstance(relation, sp.StrictGreaterThan):
            part = sp.Interval.open(end, sp.oo)
        elif isinstance(relation, sp.GreaterThan):
            part = sp.Interval(end, sp.oo)
        else:
            return None
        region = region.intersect(part)
    return region


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
    bounds: tuple[sp.Basic, sp.Basic] | None = None,
) -> tuple[sp.Basic, ...]:
    """A vector of equations, solved for as many variables as there are equations.

    Each solution is one entry, the solution vector. Unchosen variables stay on the
    right, so `[x + y + z = 1, x - y = 0]` can be solved for `x` and `y`. With `real`,
    a solution with any known non-real value is dropped. Solutions that can't be written
    follow as the vector of their reduced equations, one per factor of the eliminating
    polynomial without radical roots, as a single equation keeps such a factor (see
    `_split`).

    A relation in none of the variables is a condition on the parameters, which each
    solution of the others carries. `bounds` confine the search for a single equation
    left beside such conditions, as they do for an equation on its own.
    """
    joined = Vector(*equations)
    chosen = _chosen(joined, context, variables, len(equations))
    if not chosen:
        return _unsolvable(joined, context)
    names = tuple(variable.name for variable in chosen)
    if bounds is not None and len(equations) == 1:
        return _scalar(equations[0], context, names, bounds, (), constants, real)
    if not all(isinstance(equation, sp.Equality) for equation in equations):
        sides = [relation for relation in equations if not relation.has(*chosen)]
        # The inequalities are taken apart first, so that equations left in the
        # parameters are conditions as in a system of equations.
        split = [
            relation for relation in sides if not isinstance(relation, sp.Equality)
        ] or sides
        if not split or len(split) == len(equations):
            return _inequalities(equations, chosen, context, constants)
        conditions = []
        for relation in split:
            truth = _side_truth(relation)
            if truth is False:
                return ()
            if truth is None:
                conditions.append(relation)
        rest = tuple(relation for relation in equations if relation not in split)
        found = _system(rest, context, names, constants, real, bounds)
        return tuple(Vector(*_conjuncts(entry), *conditions) for entry in found)
    residuals = [_residual(equation) for equation in equations]
    conditions: list[sp.Basic] = []
    if any(residual.has(*chosen) for residual in residuals):
        # An equation in none of the variables is a condition on the parameters that
        # every solution carries, written as given: `[2*y*COS(t) = 3*SIN(2*t), SIN(2*t)
        # = 0]` solved for `y` is `y = 3*SIN(t)/COS(t) AND SIN(2*t) = 0`. One that is
        # false has none.
        kept = []
        rearranged: list[sp.Basic] = []
        for equation, residual in zip(equations, residuals, strict=True):
            if residual.has(*chosen):
                kept.append(residual)
                continue
            zero = _vanishes_identically(residual)
            if zero is False:
                return ()
            if zero is not True and not _cancels(residual, chosen[0]):
                conditions.append(equation)
                rearranged.append(unsolved_equation(residual, context))
        if conditions and _inconsistent(
            [condition.lhs - condition.rhs for condition in rearranged], ()
        ):
            return ()
        if conditions:
            # A condition that gives a parameter its value holds in every solution, so
            # the value replaces the parameter in the other equations.
            fixed = {
                condition.lhs: condition.rhs
                for condition in rearranged
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
                names,
                constants,
                real,
                bounds,
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
            (
                *(_relation(sp.Eq, residual, 0) for residual in independent),
                *(unsolved_equation(residual, context) for residual in left),
            ),
            context,
            names,
            constants,
            real,
            bounds,
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


def _side_truth(relation: Relational) -> bool | None:
    """Whether a relation in none of the variables holds for every value of the
    parameters (True), for none (False), or is a condition on them (None)."""
    if isinstance(relation, sp.Equality):
        return _vanishes_identically(_residual(relation))
    evaluated = _evaluated(relation)
    if evaluated is sp.true:
        return True
    if evaluated is sp.false:
        return False
    return None


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
    if roots is None:
        return None
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
    with `real` and only those in `bounds` with bounds. Its real roots are isolated
    exactly and the nonreal ones make up its degree, so that list is complete; where
    they can't be found that way, the answer is None. Anything
    else gives one real root: the leftmost one a scan of `bounds` detects, or without
    bounds that of the first window of `[-10, 10]`, `[-20, 20]`, ... that holds one.
    Where the scan detects none, interval arithmetic searches the whole range (see
    `_explored`): it finds roots too close together for the scan to see a sign change
    between them, and an empty list is its proof that the range holds no root.

    One root for anything but a polynomial is Derive's contract: "given a polynomial
    equation with numeric coefficients, NSOLVE and NSOLUTIONS return all the real and
    complex solutions to the equation. However, for any other type of equation, NSOLVE
    and NSOLUTIONS return only one solution" (Derive 6 help, Solving Equations
    Numerically), and another root is found by giving bounds that exclude the known
    ones. Derive programs depend on it: `RHS(NSOLVE(u, x))` is their way to the number,
    and `RHS` of a disjunction of several roots is not a number. The scan could report
    every sign change it passes, but that list would hold whichever roots its step
    happens to separate, written like the complete list a polynomial gives. So the rule
    is that several roots are given only where the list is known to be complete.

    Which root is ours. Derive doesn't document its choice, and its recorded answers
    follow no rule: `3*x*COS(x) - SIN(x) = 0` between 1 and 10 gave it 7.81, the third
    root of three. The leftmost one is at least the same every time. Derive's help
    gives one root for a polynomial with bounds as well; here it gives all of those in
    the bounds, that list being complete. The doubling window is the one the Derive 3
    manual describes for Mixed mode.
    """
    polynomial = _number_polynomial(residual, variable)
    if polynomial is not None:
        return _polynomial_roots(polynomial, context, real, bounds)
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
    search = _explored(residual, variable, low, high, context)
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

    The range, `bounds` or else `[-10, 10]` quadrupled until one turns up, is sampled
    for a sign change between two values, which `_root_between` narrows to a root or
    rejects. Finding none proves nothing, so the answer is then None, never empty.
    """
    from rederive.engine.from_sympy import from_sympy
    from rederive.engine.plot_form import Pointwise

    try:
        node = from_sympy(residual, context).node
    except Exception:
        return None
    digits = context.precision_digits + GUARD
    # Past the digits the root is narrowed to, so that the signs there are right.
    working = context.with_precision(Precision.APPROXIMATE, digits + 5)
    pointwise = Pointwise([node], working, [variable.name], _POINTWISE_STEPS)

    def value(point) -> _Value | None:
        (found,) = pointwise.at(_exact(point))
        if found is None or not cmath.isfinite(found):
            return None
        if abs(found.imag) > 1e-9 * max(1.0, abs(found.real)):
            return None
        # A program computes in floats: its 0 has no sign.
        if found.real == 0:
            return _UNSIGNED
        return _Value(1 if found.real > 0 else -1, mpmath.mpf(abs(found.real)))

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
        root = _sign_change(value, low, high, digits)
        if root is not None:
            return [_approximate(_exact(root), context)]
    return None


def _sign_change(
    value: Callable[[float], _Value | None], low: float, high: float, digits: int
) -> mpmath.mpf | None:
    """The leftmost root of `value` that a scan of `[low, high]` brackets, or None (see
    `_pointwise_roots`). A point whose sign can't be told is passed over."""
    step = (high - low) / _POINTWISE_SAMPLES
    before: tuple[float, int] | None = None
    for index in range(_POINTWISE_SAMPLES + 1):
        point = high if index == _POINTWISE_SAMPLES else low + index * step
        here = value(point)
        if here is None or here.sign is None:
            continue
        if before is not None and before[1] != here.sign:
            root = _root_between(value, before[0], point, digits)
            if root is not None:
                return root
        before = (point, here.sign)
    return None


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
    polynomial: sp.Poly,
    context: Context,
    real: bool,
    bounds: tuple[sp.Basic, sp.Basic] | None = None,
) -> list[sp.Basic] | None:
    """Every distinct root of a polynomial with number coefficients, approximated; with
    `real` only its real roots, and with `bounds` only the real roots in them. None if
    the roots couldn't be found or told apart, which proves nothing.

    The polynomial is made square-free and factored exactly first, so a multiple root
    is one root and a linear factor gives its root exactly: `3*x - 4` gives 4/3. A
    decimal coefficient is taken as the rational it stands for. Which roots are real,
    and which real ones lie in the bounds, is decided exactly (see `_real_intervals`
    and `_side`), so a real root is never written with an imaginary part, nor a
    nonreal one without one, however close the roots lie, and a root at a bound is in.
    """
    approximate = any(
        coefficient.has(sp.Float) for coefficient in polynomial.all_coeffs()
    )
    try:
        square_free = _exact_polynomial(polynomial).sqf_part()
        _, factors = square_free.factor_list()
    except Exception:
        return None
    real = real or bounds is not None
    found: list[sp.Basic] = []
    for factor, _ in factors:
        if factor.degree() < 1:
            continue
        if factor.degree() == 1:
            roots = _linear_root(factor, context, real, bounds, approximate)
        else:
            roots = _approximate_roots(factor, context, real, bounds)
        if roots is None:
            return None
        found.extend(roots)
    return _ordered(found)


def _exact_polynomial(polynomial: sp.Poly) -> sp.Poly:
    """`polynomial` with each float coefficient replaced by the rational it stores,
    over the smallest domain holding its coefficients exactly: `SQRT(2)` makes an
    algebraic field, where arithmetic decides zero exactly, not an expression domain.
    """
    coefficients = [
        sp.nsimplify(coefficient, rational=True)
        if coefficient.has(sp.Float)
        else coefficient
        for coefficient in polynomial.all_coeffs()
    ]
    try:
        return sp.Poly(coefficients, polynomial.gen, extension=True)
    except Exception:
        return sp.Poly(coefficients, polynomial.gen)


def _linear_root(
    factor: sp.Poly,
    context: Context,
    real: bool,
    bounds: tuple[sp.Basic, sp.Basic] | None,
    approximate: bool,
) -> list[sp.Basic] | None:
    """The root of a linear factor, exact where it is rational or `#i` and the
    polynomial had no decimal coefficient, as a list of the one root, or none if it is
    not real and `real` or outside `bounds`; None if that can't be decided."""
    slope, offset = factor.all_coeffs()
    root = sp.sympify(-offset / slope)
    if real:
        imaginary = _number_sign(sp.im(root))
        if imaginary is None:
            return None
        if imaginary:
            return []
    if bounds is not None:
        low, high = bounds
        above = low == -sp.oo or _number_sign(root - low)
        below = high == sp.oo or _number_sign(high - root)
        if above is None or below is None:
            return None
        if above < 0 or below < 0:
            return []
    exact = not approximate and (root.is_Rational or root in (sp.I, -sp.I))
    return [root if exact else _approximate_number(root, context)]


def _approximate_roots(
    factor: sp.Poly,
    context: Context,
    real: bool,
    bounds: tuple[sp.Basic, sp.Basic] | None,
) -> list[sp.Basic] | None:
    """The roots of a square-free polynomial of degree 2 or more, approximated (the
    real ones with `real`, those in `bounds` with bounds), or None if they couldn't be
    found.

    The real roots come from exact isolating intervals (`_real_intervals`); with a
    coefficient that isn't real, they are those of the greatest common divisor of the
    real and the imaginary part, the polynomials of the coefficients' real and
    imaginary parts. The other roots are computed numerically, as many as the degree
    leaves (`_nonreal_roots`), so every root is there once and each nonreal one is
    known to be nonreal. Where only real roots are asked for, nonreal ones aren't
    computed at all.
    """
    parts = [sp.sympify(c).as_real_imag() for c in factor.all_coeffs()]
    imaginary = [_number_sign(part) for _, part in parts]
    if None in imaginary:
        return None
    symmetric = not any(imaginary)
    if symmetric:
        divisor = factor
    else:
        try:
            divisor = _exact_polynomial(
                sp.Poly([re for re, _ in parts], factor.gen)
            ).gcd(_exact_polynomial(sp.Poly([im for _, im in parts], factor.gen)))
        except Exception:
            return None
    intervals = _real_intervals(divisor, context)
    if intervals is None:
        return None
    if bounds is not None:
        kept = []
        for interval in intervals:
            sides = [_side(divisor, interval, bound) for bound in bounds]
            if None in sides:
                return None
            if sides[0] >= 0 and sides[1] <= 0:
                kept.append(interval)
        intervals = kept
    found = [_approximate_number((low + high) / 2, context) for low, high, _ in intervals]
    if real:
        return found
    nonreal = _nonreal_roots(
        factor, [(low + high) / 2 for low, high, _ in intervals], symmetric, context
    )
    return None if nonreal is None else found + nonreal


#: An isolating interval `(low, high, sign)` of a real root of a square-free
#: polynomial: the root is its only one in `[low, high]`, and `sign` is the sign of
#: the polynomial at `low`, nonzero unless `low == high` is the root itself.
_Isolated = tuple[sp.Rational, sp.Rational, int]


def _real_intervals(polynomial: sp.Poly, context: Context) -> list[_Isolated] | None:
    """Isolating intervals of every real root of a square-free polynomial with real
    coefficients, each narrower than the root's size by the digits shown and the
    guard, ascending; None if they couldn't be found.

    With rational coefficients the intervals are SymPy's, found and refined exactly.
    Otherwise the polynomial's real roots are counted with a Sturm sequence over its
    own coefficient domain, and the intervals are those of a rational approximation of
    the polynomial, each widened a little and checked to have a sign change of the
    polynomial itself across it: disjoint intervals with a sign change, as many as the
    count, hold one root each and are all of them. A sign is decided exactly where it
    is zero and by evaluation with SymPy's error bound otherwise (`_number_sign`).
    """
    if polynomial.degree() < 1:
        return []
    tolerance = sp.Rational(1, 10 ** (context.precision_digits + GUARD))
    if polynomial.domain in (sp.ZZ, sp.QQ):
        found = []
        for (low, high), _ in polynomial.intervals():
            refined = _refined(polynomial, low, high, tolerance)
            if refined is None:
                return None
            low, high = refined
            sign = _number_sign(polynomial.eval(low))
            if sign is None or (sign == 0) != (low == high):
                return None
            found.append((low, high, sign))
        return found
    count = _sturm_count(polynomial)
    if count is None:
        return None
    if count == 0:
        return []
    working = 2 * (context.precision_digits + GUARD) + 20
    for _ in range(3):
        found = _checked_intervals(polynomial, count, working, tolerance)
        if found is not None:
            return found
        working *= 2
    return None


def _checked_intervals(
    polynomial: sp.Poly, count: int, working: int, tolerance: sp.Rational
) -> list[_Isolated] | None:
    """The isolating intervals of `_real_intervals` from a rational approximation of
    `polynomial` to `working` digits, if they check out against the polynomial."""
    try:
        approximation = sp.Poly(
            [
                sp.Rational(str(sp.sympify(coefficient).evalf(working)))
                for coefficient in polynomial.all_coeffs()
            ],
            polynomial.gen,
        )
        intervals = approximation.intervals()
    except Exception:
        return None
    if len(intervals) != count:
        return None
    found: list[_Isolated] = []
    for (low, high), _ in intervals:
        refined = _refined(approximation, low, high, tolerance)
        if refined is None:
            return None
        low, high = refined
        width = max(high - low, abs(low) * tolerance, tolerance**2)
        low, high = low - width, high + width
        signs = [_number_sign(polynomial.eval(end)) for end in (low, high)]
        if None in signs:
            return None
        if signs[0] * signs[1] >= 0 or (found and found[-1][1] >= low):
            return None
        found.append((low, high, signs[0]))
    return found


def _refined(
    polynomial: sp.Poly, low: sp.Rational, high: sp.Rational, tolerance: sp.Rational
) -> tuple[sp.Rational, sp.Rational] | None:
    """An isolating interval of a rational polynomial's root narrowed by bisection
    until its width is `tolerance` of its distance from 0, or None if that takes too
    long. Signs are computed exactly, in integers."""
    numerators = [sp.Rational(coefficient) for coefficient in polynomial.all_coeffs()]
    scale = math.lcm(*(int(coefficient.q) for coefficient in numerators))
    coefficients = [int(coefficient * scale) for coefficient in numerators]

    def sign(point: Fraction) -> int:
        top, bottom = point.numerator, point.denominator
        value = coefficients[0]
        for power, coefficient in enumerate(coefficients[1:], 1):
            value = value * top + coefficient * bottom**power
        return (value > 0) - (value < 0)

    first, last = Fraction(int(low.p), int(low.q)), Fraction(int(high.p), int(high.q))
    at_first = sign(first)
    if at_first == 0:
        return low, low
    narrow = Fraction(int(tolerance.p), int(tolerance.q))
    for _ in range(_REFINEMENT):
        if first == last or (
            first * last > 0 and last - first <= min(abs(first), abs(last)) * narrow
        ):
            return sp.Rational(first), sp.Rational(last)
        middle = Fraction(0) if first < 0 < last else (first + last) / 2
        at_middle = sign(middle)
        if at_middle == 0:
            return sp.Rational(middle), sp.Rational(middle)
        if at_middle == at_first:
            first = middle
        else:
            last = middle
    return None


#: The most bisections `_refined` makes of one interval.
_REFINEMENT = 20000


def _sturm_count(polynomial: sp.Poly) -> int | None:
    """The number of real roots of a square-free polynomial with real coefficients,
    by its Sturm sequence over its own coefficient domain, or None if a sign in it
    can't be decided."""
    try:
        field = polynomial if polynomial.domain.is_Field else polynomial.to_field()
        sequence = field.sturm()
    except Exception:
        return None
    signs = [_number_sign(member.LC()) for member in sequence]
    if None in signs or 0 in signs:
        return None
    at_minus = [sign * (-1) ** member.degree() for sign, member in zip(signs, sequence)]
    return _variations(at_minus) - _variations(signs)


def _variations(signs: list[int]) -> int:
    return sum(1 for first, second in itertools.pairwise(signs) if first != second)


def _side(polynomial: sp.Poly, interval: _Isolated, bound: sp.Basic) -> int | None:
    """The sign of `root - bound` for the root of `polynomial` in `interval`, or None
    if it can't be decided. That the root is the bound is decided exactly."""
    if bound == -sp.oo:
        return 1
    if bound == sp.oo:
        return -1
    low, high, sign = interval
    if low == high:
        return _number_sign(low - bound)
    below = _number_sign(bound - low)
    if below is None or below <= 0:
        return None if below is None else 1
    above = _number_sign(high - bound)
    if above is None or above <= 0:
        return None if above is None else -1
    value = _number_sign(polynomial.as_expr().subs(polynomial.gen, bound))
    if value is None or value == 0:
        return value
    # The polynomial keeps its sign at `low` from `low` up to the root.
    return 1 if value == sign else -1


def _number_sign(value: sp.Basic) -> int | None:
    """The sign of a real number, or None if it can't be decided.

    Zero is decided exactly: by SymPy's arithmetic, or for an algebraic number by its
    minimal polynomial. A nonzero number's sign is that of its evaluation, which
    SymPy gives only once its error bound excludes 0.
    """
    value = sp.sympify(value)
    if value.is_Rational:
        return int(sp.sign(value))
    try:
        if value.is_extended_positive:
            return 1
        if value.is_extended_negative:
            return -1
        if sp.expand(value) == 0:
            return 0
        if value.is_algebraic and sp.minimal_polynomial(value).is_Symbol:
            return 0
    except Exception:
        return None
    return None


def _nonreal_roots(
    factor: sp.Poly, real: list[sp.Rational], symmetric: bool, context: Context
) -> list[sp.Basic] | None:
    """The nonreal roots of a square-free polynomial whose real roots are `real`, or
    None if they couldn't be computed to the digits shown.

    All roots are computed together (mpmath's `polyroots`), the one nearest each real
    root is set aside, and the others are the nonreal ones. Each must be accurate to
    the digits shown and the guard in its imaginary part, which is known not to be
    zero, and in its real part unless that is negligible beside the root's size (then
    it is 0, as `#i` is). With real coefficients (`symmetric`) the nonreal roots come
    in conjugate pairs, written as such. Where that isn't reached, the working
    precision is doubled.
    """
    count = factor.degree() - len(real)
    if count == 0:
        return []
    digits = context.precision_digits + GUARD
    working = 2 * digits + 20
    parts = [sp.sympify(c).as_real_imag() for c in factor.all_coeffs()]
    for attempt in range(4):
        with mpmath.workdps(working):
            try:
                values = [
                    mpmath.mpc(*(mpmath.mpf(str(part.evalf(working))) for part in pair))
                    for pair in parts
                ]
                roots, error = mpmath.polyroots(
                    values,
                    maxsteps=200 << attempt,
                    cleanup=False,
                    extraprec=4 * working,
                    error=True,
                )
            except Exception:
                working *= 2
                continue
            remaining = list(roots)
            for root in real:
                value = mpmath.mpf(root.p) / root.q
                remaining.remove(min(remaining, key=lambda found: abs(found - value)))
            found = _nonreal_found(remaining, error, symmetric, digits, context)
            if found is not None:
                return found
        working *= 2
    return None


def _nonreal_found(
    roots: list, error, symmetric: bool, digits: int, context: Context
) -> list[sp.Basic] | None:
    """`roots`, nonreal roots computed with `error`, as numbers, if each is accurate to
    `digits` (see `_nonreal_roots`); with `symmetric` they must pair up as conjugates.
    """
    tolerance = mpmath.mpf(10) ** -digits
    if any(
        error > min(abs(root.imag), abs(root) * tolerance) * tolerance for root in roots
    ):
        return None
    if symmetric:
        upper = sorted((root for root in roots if root.imag > 0), key=_key)
        lower = sorted((mpmath.conj(root) for root in roots if root.imag < 0), key=_key)
        if len(upper) != len(lower) or any(
            abs(first - second) > 2 * error for first, second in zip(upper, lower)
        ):
            return None
        roots = upper
    found: list[sp.Basic] = []
    for root in roots:
        size = abs(root) * tolerance
        real = sp.S.Zero if abs(root.real) <= size else _rounded(root.real, context)
        value = real + _rounded(root.imag, context) * sp.I
        found += [value, sp.conjugate(value)] if symmetric else [value]
    return found


def _key(root) -> tuple:
    return (root.real, root.imag)


def _rounded(value, context: Context) -> sp.Rational:
    """An mpmath real as the simplest rational with the digits shown."""
    number = sp.Float(str(value), mpmath.mp.dps)
    return simplest_rational(number, context.precision_digits)


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

    This searches a doubling window like Mixed mode's, from `_REACH` to `_REACH_CAP`,
    within one `_Budget`. Finding nothing proves nothing, hence None rather than an
    empty tuple.
    """
    first, last = _number(low), _number(high)
    if first is None or last is None or not first < last:
        return None
    budget = _Budget()
    reach = _REACH
    while reach <= _REACH_CAP:
        if math.isinf(first) and math.isinf(last):
            window = (-reach, reach)
        elif math.isinf(last):
            window = (first, first + reach)
        else:
            window = (max(first, last - reach), last)
        start, end = (sp.Rational(bound) for bound in window)
        root = _bracketed(residual, variable, start, end, context, budget)
        if root is not None:
            return (_equated(variable, root, context),)
        reach *= 2
    return None


def _approximate(root: sp.Basic, context: Context) -> sp.Basic:
    """A root as the simplest rational with its digits, like every approximate number, so
    it prints like one: `-1.16730` at six digits, truncated.
    """
    digits = context.precision_digits
    return simplest_rational(root.evalf(digits + GUARD), digits)


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
    budget: _Budget | None = None,
) -> sp.Basic | None:
    """One root of `residual` in `[low, high]`, found by scanning and bisection. The scan
    finds a piece whose ends straddle a single root. The evaluations are counted
    against `budget`, a search's, or else a budget of their own.
    """
    first, last = _number(low), _number(high)
    if first is None or last is None or not first < last:
        return None
    budget = budget or _Budget()
    return _scanned(
        residual, variable, first, last, context, budget, _RESCANS, _EDGE_RESCANS
    )


#: The most times one search may evaluate its residual, scanning and bisecting. A
#: residual undefined on many short intervals has every piece next to one scanned again,
#: and one with many poles has each bisected: `SQRT(SIN(x)) = 2`, which has no root,
#: would take minutes over all the windows.
_SEARCH_BUDGET = 5000


class _Budget:
    """The evaluations left to a search (see `_SEARCH_BUDGET`)."""

    def __init__(self) -> None:
        self.left = _SEARCH_BUDGET

    def spend(self) -> bool:
        """Whether one more evaluation is allowed, counting it."""
        self.left -= 1
        return self.left >= 0


def _scanned(
    residual: sp.Basic,
    variable: sp.Symbol,
    first: float,
    last: float,
    context: Context,
    budget: _Budget,
    depth: int,
    edges: int = 0,
) -> sp.Basic | None:
    """One root in `[first, last]` by a scan of `_SCAN` pieces, or None, also once
    `budget` runs out.

    The scan evaluates in floats, which only picks what to try: a sign change is
    bisected (`_bisect`), and a point where the value comes to 0 is evaluated again
    exactly (`_value_at`), a root only if that proves it zero. A point where the sign
    can't be told is passed over, the points beside it compared instead.

    A point where the residual is undefined tells nothing about the sign beside it, so
    a root next to it is missed: `ATAN(z - 0.1/z) + ATAN(z + 0.1/z) = 0.5` is undefined
    at 0 and has its root at 0.28, short of the next point. Once the whole interval is
    scanned, a piece with one end undefined is scanned again, `depth` levels deep. A
    piece undefined at both ends is not, nor one whose sign change bisects to no root:
    rescanning every pole of `1/SIN(x)` makes the widening search far too slow.

    A piece at an end of the interval where the residual is undefined is scanned again
    up to `edges` levels deep: a single piece per level, the one next to the end.
    """
    digits = context.precision_digits + GUARD
    step = (last - first) / _SCAN
    # The pieces to scan again, and whether each is at an end of the interval.
    suspect: list[tuple[float, float, bool]] = []
    # The last point whose sign is known, and that sign.
    previous: tuple[float, int] | None = None
    # Whether the last point told was defined; a piece undefined at both ends, as
    # `LN(x)` is to the left of 0 over the reals, is not rescanned.
    defined = False
    start = first
    for index in range(_SCAN + 1):
        point = last if index == _SCAN else first + index * step
        piece = (start, point, index in (1, _SCAN))
        start = point
        if not budget.spend():
            return None
        sign = _scan_sign(residual, variable, point)
        if sign == 0:
            exact = _value_at(residual, variable, point, digits)
            if exact is not None and exact.sign == 0:
                return _approximate(_exact(point), context)
            if exact is not None and exact.sign is None:
                continue
            sign = None if exact is None else exact.sign
        if sign is None:
            if defined:
                suspect.append(piece)
            previous, defined = None, False
            continue
        if index and not defined:
            suspect.append(piece)
        elif previous is not None and previous[1] != sign:
            found = _bisect(residual, variable, previous[0], point, context, budget)
            if found is not None:
                return found
        previous, defined = (point, sign), True
    for low, high, at_edge in suspect:
        if not (depth or (edges and at_edge)):
            continue
        found = _scanned(
            residual,
            variable,
            low,
            high,
            context,
            budget,
            max(depth - 1, 0),
            edges - 1 if at_edge and edges else 0,
        )
        if found is not None:
            return found
    return None


def _scan_sign(residual: sp.Basic, variable: sp.Symbol, point: float) -> int | None:
    """The sign of `residual` at `point` as evaluating it in floats gives it, 0 where
    that comes to zero, or None where it is undefined or not real.

    The sign is that of the sympy float, whose exponent has no limit: a value too small
    for a Python float keeps its sign.
    """
    evaluated = _substituted(residual, variable, point)
    if evaluated is None:
        return None
    try:
        real, imaginary = evaluated.evalf().as_real_imag()
    except Exception:
        return None
    if imaginary != 0 or not (real.is_Float or real.is_Rational):
        return None
    return 1 if real.is_positive else -1 if real.is_negative else 0


def _substituted(
    residual: sp.Basic, variable: sp.Symbol, point: float
) -> sp.Basic | None:
    """`residual` with `point` substituted as a float, or None where that fails, as
    `GAMMA` does at a pole: the residual is undefined there."""
    try:
        return residual.subs(variable, sp.Float(point))
    except Exception:
        return None


def _bisect(
    residual: sp.Basic,
    variable: sp.Symbol,
    low: float,
    high: float,
    context: Context,
    budget: _Budget | None = None,
) -> sp.Basic | None:
    """The root of `residual` between `low` and `high`, where its sign changes, or None
    if it changes sign without vanishing there, as at a pole or a jump.

    The residual is evaluated at exact points (`_value_at`), so its signs are right
    however much its terms cancel: `PHI^n/SQRT(5) - 10^100` is zero at its root to a
    hundred digits less than its terms. `_root_between` narrows the bracket and judges
    what it holds. Each evaluation is counted against `budget` if there is one.
    """
    digits = context.precision_digits + GUARD

    def value(point) -> _Value | None:
        if budget is not None and not budget.spend():
            return None
        return _value_at(residual, variable, point, digits)

    root = _root_between(value, low, high, digits)
    return None if root is None else _approximate(_exact(root), context)


#: How far, as a power of 2, the bracket of `_root_between` narrows between two brackets
#: whose values it compares, and the least power of that narrowing by which a root's
#: values shrink: a simple root's shrink by all of it, and over a narrowing of 65536
#: they must shrink by 16 at least.
_NARROWING = 16
_ORDER = mpmath.mpf(1) / 4

#: How far, as a power of 2, the bracket of `_root_between` narrows before a value
#: larger than at the wider bracket marks a pole.
_POLE_NARROWING = 10

#: The most points `_root_between` evaluates, enough to narrow a bracket of 1 to every
#: digit of a root near 10^-300 by halving alone.
_NARROWING_STEPS = 1200


def _root_between(
    value: Callable[[mpmath.mpf], _Value | None], low, high, digits: int
) -> mpmath.mpf | None:
    """The root between `low` and `high` of the residual that `value` evaluates, to
    `digits` digits, or None if the bracket holds none it can show.

    The residual must have opposite signs at the ends, or be proved zero at one. The
    bracket is narrowed by the secant method, as the Illinois variant keeps both ends
    moving, with a halving wherever that is slow, until it is `digits` digits of the
    root wide and `_NARROWING` halvings narrower than it was. A bracket holding 0 is cut
    there first, so that a root at 0 or next to it has its digits too. A point where
    the residual is undefined ends the search; one where its sign can't be told is
    stepped around (`_stepped_around`).

    What the bracket holds is judged by how the larger of the values at its ends
    changes as it narrows: at a root it shrinks with the bracket, at a pole it grows,
    and at a jump, as of `FLOOR(x) - 1/2` at 1, it comes to a side of the jump and
    stays. The search gives up where the value is larger than `_POLE_NARROWING`
    halvings before, or has not shrunk by the power `_ORDER` of the narrowing over
    three times `_NARROWING`, and takes the root only if it shrank by that power over
    the last `_NARROWING`. No size of a value could tell these apart: a jump can be
    small, and a root's residual large next to its digits.
    """
    with mpmath.workprec(int((digits + 10) * 3.33) + 10):
        a, b = mpmath.mpf(low), mpmath.mpf(high)
        if not a < b:
            return None
        fa, fb = value(a), value(b)
        if fa is None or fb is None:
            return None
        if fa.sign == 0:
            return a
        if fb.sign == 0:
            return b
        if fa.sign is None or fb.sign is None or fa.sign == fb.sign:
            return None
        if a < 0 < b:
            zero = mpmath.mpf(0)
            middle = value(zero)
            if middle is not None and middle.sign == 0:
                return zero
            if middle is not None and middle.sign == fa.sign:
                a, fa = zero, middle
            elif middle is not None and middle.sign == fb.sign:
                b, fb = zero, middle
        # The width of the bracket and the larger value at its ends, step by step.
        history = [(b - a, max(fa.size, fb.size))]
        tolerance = mpmath.mpf(2) ** -int(digits * 3.33)
        enough = history[0][0] * mpmath.mpf(2) ** -_NARROWING
        # The secant's weights on the values at the ends: an end kept twice running
        # has the other's halved, so that it moves too.
        weights = [mpmath.mpf(1), mpmath.mpf(1)]
        kept = None
        for _ in range(_NARROWING_STEPS):
            width = b - a
            if width <= enough and width <= tolerance * _magnitude(a, b):
                break
            middle = (a + b) / 2
            # Halved where the last two steps didn't halve the bracket.
            if len(history) < 3 or width <= history[-3][0] / 2:
                left, right = fa.size * weights[0], fb.size * weights[1]
                if left + right > 0:
                    middle = a + width * left / (left + right)
            if not a < middle < b:
                middle = (a + b) / 2
                if not a < middle < b:
                    break
            here = value(middle)
            if here is None:
                return None
            if here.sign == 0:
                return middle
            if here.sign is None:
                stepped = _stepped_around(value, (a, fa), (b, fb), middle)
                if stepped is None or isinstance(stepped, mpmath.mpf):
                    return stepped
                (a, fa), (b, fb) = stepped
                weights, kept = [mpmath.mpf(1), mpmath.mpf(1)], None
            elif here.sign == fa.sign:
                a, fa = middle, here
                weights = [mpmath.mpf(1), weights[1] / 2 if kept == 1 else weights[1]]
                kept = 1
            else:
                b, fb = middle, here
                weights = [weights[0] / 2 if kept == 0 else weights[0], mpmath.mpf(1)]
                kept = 0
            width, size = b - a, max(fa.size, fb.size)
            # A pole, and a jump: neither can be narrowed to a root.
            wider = _wider(history, width, _POLE_NARROWING)
            if wider is not None and size > wider[1]:
                return None
            wider = _wider(history, width, 3 * _NARROWING)
            if wider is not None and size > wider[1] * (width / wider[0]) ** _ORDER:
                return None
            history.append((width, size))
        else:
            return None
        width, size = b - a, max(fa.size, fb.size)
        wider = _wider(history, width, _NARROWING) or history[0]
        if size > wider[1] * (width / wider[0]) ** _ORDER:
            return None
        return a + (b - a) * fa.size / (fa.size + fb.size)


def _magnitude(low: mpmath.mpf, high: mpmath.mpf) -> mpmath.mpf:
    """The least magnitude in `[low, high]`, the scale of a root there."""
    if low <= 0 <= high:
        return mpmath.mpf(0)
    return min(abs(low), abs(high))


def _wider(
    history: list[tuple[mpmath.mpf, mpmath.mpf]], width: mpmath.mpf, halvings: int
) -> tuple[mpmath.mpf, mpmath.mpf] | None:
    """The last bracket of `history` at least `halvings` halvings wider than `width`."""
    for entry in reversed(history):
        if entry[0] >= width * 2**halvings:
            return entry
    return None


def _stepped_around(
    value: Callable[[mpmath.mpf], _Value | None],
    low: tuple[mpmath.mpf, _Value],
    high: tuple[mpmath.mpf, _Value],
    middle: mpmath.mpf,
):
    """The bracket `[low, high]` narrowed around `middle`, a point where the residual's
    sign can't be told, by its values a little to either side: a root there if one is
    proved zero, None if a sign there can't be told either."""
    offset = (high[0] - low[0]) * mpmath.mpf(2) ** -20
    beside = []
    for point in (middle - offset, middle + offset):
        here = value(point)
        if here is None or here.sign is None:
            return None
        if here.sign == 0:
            return point
        beside.append((point, here))
    left, right = beside
    if left[1].sign != right[1].sign:
        return left, right
    if left[1].sign == low[1].sign:
        return right, high
    return low, left


@dataclass(frozen=True)
class _Value:
    """The residual's value at a point as far as its sign is known: `sign` is 1 or -1,
    0 where the value is proved zero, or None where it is too small for its sign to be
    told; `size` is its absolute value."""

    sign: int | None
    size: mpmath.mpf


#: A value whose sign can't be told.
_UNSIGNED = _Value(None, mpmath.mpf(0))


def _value_at(
    residual: sp.Basic, variable: sp.Symbol, point, digits: int
) -> _Value | None:
    """The value of `residual` at `point`, or None where it is undefined or not real.

    The point is substituted as the exact rational it is, so a conditional evaluated
    at it computes exactly, and the value is evaluated to `digits` correct digits,
    with the working precision that takes; a value that needs more than evaluation
    allows has no sign. It is proved zero only where substitution computes an exact 0:
    a residual with float coefficients computes in floats, and its 0 has no sign.
    """
    try:
        exact = residual.subs(variable, _exact(point))
        if exact == 0:
            return _Value(0, mpmath.mpf(0)) if exact is sp.S.Zero else _UNSIGNED
        real, imaginary = exact.evalf(digits, strict=True).as_real_imag()
    except sp.core.evalf.PrecisionExhausted:
        return _UNSIGNED
    except Exception:
        return None
    if imaginary != 0 or not real.is_Float:
        return None
    if real == 0:
        return _UNSIGNED
    return _Value(1 if real > 0 else -1, abs(mpmath.mpf(real)))


def _exact(point) -> sp.Rational:
    """A float, sympy's or mpmath's or Python's, as the rational it is."""
    if isinstance(point, sp.Rational):
        return point
    if isinstance(point, float):
        return sp.Rational(point)
    # Read without rounding to the working precision, as `mpmath.mpf` would.
    sign, mantissa, exponent, _ = point._mpf_
    return (-1) ** sign * sp.Integer(mantissa) * sp.Integer(2) ** exponent


def _vanishing(residual: sp.Basic, variable: sp.Symbol, point, digits: int) -> bool:
    """Whether `residual` has a root at `point` to `digits - 2` digits: whether it is
    proved zero there, or `_root_between` finds a root in a bracket that wide around
    it."""

    def value(at) -> _Value | None:
        return _value_at(residual, variable, at, digits)

    with mpmath.workprec(int((digits + 10) * 3.33) + 10):
        try:
            center = mpmath.mpf(point)
        except Exception:
            return False
        here = value(center)
        if here is None:
            return False
        if here.sign == 0:
            return True
        offset = (abs(center) or mpmath.mpf(1)) * mpmath.mpf(10) ** (2 - digits)
        return _root_between(value, center - offset, center + offset, digits) is not None


def _searched(
    residual: sp.Basic, variable: sp.Symbol, context: Context, real: bool = True
) -> tuple[sp.Basic, ...] | None:
    """The last resort of Mixed and Approximate modes: every root of a polynomial with
    number coefficients (the real ones with `real`), and of anything else the first
    real root of `[-10, 10]`, doubling the interval until one turns up, as the manual
    describes for Mixed. The cap is ours, and so is the `_Budget` of the search.
    """
    polynomial = _number_polynomial(residual, variable)
    if polynomial is not None:
        roots = _polynomial_roots(polynomial, context, real)
        if not roots:
            return None
        return tuple(_equated(variable, root, context) for root in roots)
    budget = _Budget()
    reach = _REACH
    while reach <= _REACH_CAP:
        root = _bracketed(
            residual, variable, sp.Integer(-reach), sp.Integer(reach), context, budget
        )
        if root is not None:
            return (_equated(variable, root, context),)
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
    low: sp.Basic,
    high: sp.Basic,
    context: Context,
) -> _Search:
    """A root of `residual` in `[low, high]` found by interval arithmetic, or the finding
    that it has none there.

    The interval is cut in halves, leftmost first. A piece on which the enclosure of
    the residual's values excludes zero holds no root and is dropped; the sign of the
    residual is known at each cut, and a piece whose ends differ in sign is bisected.
    So two roots closer together than a scan's step, `SIN(x^2) = #e^x` near -6.86, are
    told apart once the pieces are narrow enough to put a cut between them. If every
    piece is dropped, there is no root; a piece that gets too narrow, a residual that
    has no enclosure, or running out of `_EXPLORATION` proves nothing. An infinite end
    is reached by pieces that double in length. The pieces cover the doubles from the
    one at or below `low` to the one at or above `high`, so the bounds themselves are
    covered whatever their value. A finding of no root is as sound as the enclosures of
    `_enclosing`, short of the reals strictly between an end of a piece whose sign is
    proved and the next double, which no enclosure looks at (see below).

    On the real branch an odd root of a negative value is real (`_on_real_branch_form`),
    and the residual is enclosed, bisected and refined in that form.
    """
    principal = context.branch is Branch.PRINCIPAL
    if context.branch is Branch.REAL:
        residual = _on_real_branch_form(residual)
        principal = True
    enclose = _enclosing(residual, variable, principal)
    first, last = _outward(low, -math.inf), _outward(high, math.inf)
    if enclose is None or first is None or last is None:
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


def _outward(bound: sp.Basic, toward: float) -> float | None:
    """The double nearest `bound` on its side `toward` (minus or plus infinity) or at
    it, as SymPy decides the sign of their difference, or None if it can't."""
    if bound.is_infinite:
        return toward if (bound > 0) == (toward > 0) else -toward
    if bound.is_Float:
        bound = sp.Rational(bound)
    try:
        point = float(bound.evalf(20))
    except Exception:
        return None
    for _ in range(4):
        if math.isinf(point):
            return None
        difference = sp.Rational(point) - bound
        if (difference.is_nonpositive if toward < 0 else difference.is_nonnegative):
            return point
        point = math.nextafter(point, toward)
    return None


def _on_real_branch_form(expression: sp.Basic) -> sp.Basic:
    """`expression` with each odd root of a value taken as Branch Real takes it, real
    for a negative value: `u^(p/q)` with `q` odd is `SIGN(u)^p*ABS(u)^(p/q)`."""
    if not expression.args:
        return expression
    args = [_on_real_branch_form(arg) for arg in expression.args]
    changed = args != list(expression.args)
    if (
        isinstance(expression, sp.Pow)
        and expression.exp.is_Rational
        and not expression.exp.is_Integer
        and expression.exp.q % 2 == 1
    ):
        base = args[0]
        return sp.sign(base) ** int(expression.exp.p) * sp.Abs(base) ** expression.exp
    return expression.func(*args) if changed else expression


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


def _enclosing(
    residual: sp.Basic, variable: sp.Symbol, principal: bool
) -> Callable | None:
    """A function from an interval of `variable` to an enclosure of every value of
    `residual` on it, its real and imaginary parts, or None if a part of it has no
    such rule.

    The rules are those of `mpmath.iv`, which rounds outwards. A constant other than a
    rational is enclosed by SymPy's 30-digit evaluation of it, widened by 10^-25 of its
    size (`_constant_enclosure`): that evaluation's accuracy is what SymPy's error
    tracking estimates, not a bound it proves, so an enclosure holding such a constant
    is only as good as that estimate. A power of a negative base is complex on the
    principal branch, `x^(1/3)` being `ABS(x)^(1/3)*#e^(#i*pi/3)` there, so `x^(2/3) -
    2*x^(1/3) = 35` is proved to have no root below 0 by its imaginary part. Off it
    (`principal` false) a fractional power of a base that may be negative has no rule.
    A function of a complex value, a fractional power of a base whose values have both
    signs and `LN` of one not positive have no rule either; they raise, and the piece
    they are on is then not dropped.
    """
    try:
        return _enclosure(residual, variable, principal)
    except Exception:
        return None


def _enclosure(expression: sp.Basic, variable: sp.Symbol, principal: bool) -> Callable:
    iv = mpmath.iv
    if not expression.has(variable):
        constant = _constant_enclosure(expression)
        return lambda box: constant
    if expression == variable:
        return lambda box: (box, None)
    if isinstance(expression, (sp.Add, sp.Mul)):
        parts = [_enclosure(part, variable, principal) for part in expression.args]
        operation = _sum if isinstance(expression, sp.Add) else _product

        def combined(box):
            total = parts[0](box)
            for part in parts[1:]:
                total = operation(total, part(box))
            return total

        return combined
    if isinstance(expression, sp.Pow):
        base, exponent = expression.args
        lower = _enclosure(base, variable, principal)
        if exponent.is_Integer:
            power = int(exponent)
            return lambda box: _integer_power(lower(box), power)
        if exponent.is_Rational:
            turn = _constant_enclosure(exponent)[0] * iv.pi if principal else None
            return lambda box: _rational_power(_real(lower(box)), exponent, turn)
        upper = _enclosure(exponent, variable, principal)
        return lambda box: (_real_power(_real(lower(box)), _real(upper(box))), None)
    if isinstance(expression, sp.Abs):
        inner = _enclosure(expression.args[0], variable, principal)
        return lambda box: (_absolute(inner(box)), None)
    if isinstance(expression, sp.sign):
        inner = _enclosure(expression.args[0], variable, principal)
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
    inner = _enclosure(expression.args[0], variable, principal)
    return lambda box: (call(_real(inner(box))), None)


def _constant_enclosure(constant: sp.Basic):
    """An enclosure of a constant, real or complex: exact for a rational, otherwise
    SymPy's 30-digit value widened by 10^-25 of its size and rounded outwards."""
    iv = mpmath.iv
    if constant.is_Rational:
        return iv.mpf(int(constant.p)) / iv.mpf(int(constant.q)), None
    value = sp.sympify(constant).evalf(30, strict=True)
    real, imaginary = value.as_real_imag()
    if not (real.is_Number and imaginary.is_Number):
        raise _NoEnclosure("not a number")
    with mpmath.workprec(256):
        # A 30-digit value converts to an mpf exactly at this precision.
        parts = [mpmath.mpf(sp.Float(part, 30)._mpf_) for part in (real, imaginary)]
        size = max(abs(part) for part in parts)
        if not size or not mpmath.isfinite(size):
            raise _NoEnclosure("no size to widen by")
        radius = size * mpmath.mpf(10) ** -25
    # `iv` keeps the exact endpoints it is given and rounds the sum outwards.
    widening = iv.mpf([-radius, radius])

    def interval(part):
        return iv.mpf(part) + widening

    return interval(parts[0]), None if imaginary == 0 else interval(parts[1])


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
    """`base^exponent` for a real base of one sign: on the principal branch the turn of
    a negative base is `exponent*pi`; with no `turn`, off it, a base that may be
    negative has no rule."""
    iv = mpmath.iv
    if base.a >= 0:
        return _real_power(base, iv.mpf(int(exponent.p)) / iv.mpf(int(exponent.q))), None
    if base.b > 0 or turn is None:
        raise ValueError("a power of a value that may be negative")
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
