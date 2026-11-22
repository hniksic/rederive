"""The answers of `SOLVE`, `SOLUTIONS`, `NSOLVE` and `NSOLUTIONS`.

`solving` finds the solutions; this module reads the arguments of the four functions and
writes what they find in Derive 6's forms.

`SOLVE(u, x)` is a Boolean expression equivalent to `u`, in disjunctive normal form: `x
= -1 OR x = 1`, one disjunct per solution, `x = 2 AND y = 1` for a solution of a system,
`true` when every value solves `u` and `false` when none does. A disjunct may also be an
interval of an inequality, an equation left unsolved, or a condition on the parameters
under which every value is a solution (`R = 0`). Solutions that hold under the same
conditions on the parameters state them once: `(x = -2 OR x = 2) AND a = b`. With a
vector of relations as its first argument the answer is the vector of the solutions,
each with its conditions, instead.

`SOLUTIONS(u, x)` is the vector of the values `SOLVE` finds for `x`, or with a vector of
variables the matrix of the solutions, one row each; a free value is a new arbitrary
constant `@n`. It holds the solutions found: an equation left unsolved and a condition
on the parameters found beside the solutions of an equation are left out, so `[]` says
that none was found, not that there is none. A value that holds under a condition on
the parameters is `IF(condition, value)`, and a formula in none of the variables is
solved by every value where it holds: `SOLUTIONS(a = b, x)` is `[IF(a - b = 0, @1)]`.

`NSOLVE` and `NSOLUTIONS` solve one equation in one variable numerically, in the forms
of `SOLVE` and `SOLUTIONS`: every root of a polynomial, or one real root of anything
else. `NSOLVE` answers `false` only where a polynomial's roots or interval arithmetic
prove there is no root, and a search that finds none without proving it leaves the call
as written; `NSOLUTIONS` lists the roots found, `[]` when there are none.

A call of none of these forms stays as it is written. The answers are built from the
first argument as simplified, with equations not decided (see `pipeline._commanded`).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import sympy as sp
from sympy.core.relational import Relational
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.logic.boolalg import Boolean, BooleanAtom

from rederive.engine.context import Angle, Context, Precision
from rederive.engine.normal import normal_form
from rederive.engine.ordering import main_order
from rederive.engine.sets import is_set
from rederive.engine.solving import (
    Outcome,
    arbitrary,
    between,
    confined,
    listed_in_order,
    numeric_roots,
    outcome,
    parameter_condition,
    solved_range,
    unsolved_equation,
)
from rederive.engine.to_sympy import (
    DEGREE,
    InOrder,
    InOrderOr,
    Logical,
    StringLiteral,
    Vector,
    in_degrees,
)

__all__ = ["nsolve_head", "solutions_head", "solve_head"]

#: The solution domains the third argument may name.
_DOMAIN_WORDS = ("Real", "Complex")

#: Decides a relation free of variables: the pipeline's simplification with deciding on.
Decider = Callable[[sp.Basic], sp.Basic]


@dataclass(frozen=True)
class _Call:
    """The arguments of a solving function, read."""

    target: sp.Basic
    names: tuple[str, ...]
    #: Whether the variables were given as a vector, which makes SOLUTIONS a matrix.
    listed: bool
    real: bool
    bounds: tuple[sp.Basic, sp.Basic] | None


def _read(args: Sequence[sp.Basic], one: bool) -> _Call | None:
    """The arguments of `SOLVE`, `SOLUTIONS`, `NSOLVE` or `NSOLUTIONS`, or None for a call
    of no form they accept. `one` allows the target alone, whose variables are then
    found by the caller."""
    if not args or len(args) > 4 or (len(args) == 1 and not one):
        return None
    target, *rest = args
    if is_set(target) or isinstance(target, StringLiteral):
        return None
    names: tuple[str, ...] = ()
    listed = False
    if rest:
        names = _variables(rest[0])
        if not names or len(set(names)) != len(names):
            return None
        listed = isinstance(rest[0], Vector)
    real = False
    bounds = None
    if len(rest) == 2:
        word = _domain_word(rest[1])
        if word is None:
            return None
        real = word == "Real"
    elif len(rest) == 3:
        low, high = rest[1], rest[2]
        if not (_real_bound(low) and _real_bound(high)):
            return None
        bounds = (low, high)
    return _Call(target, names, listed, real, bounds)


def _variables(argument: sp.Basic) -> tuple[str, ...]:
    """The variables a solving function names: one, or a vector of them, nested vectors
    read in order. Anything else names none."""
    if isinstance(argument, sp.Symbol) and not isinstance(argument, StringLiteral):
        return () if argument.name in _DOMAIN_WORDS else (argument.name,)
    if not isinstance(argument, Vector) or not argument.args:
        return ()
    names: list[str] = []
    for element in argument.args:
        inner = _variables(element)
        if not inner:
            return ()
        names.extend(inner)
    return tuple(names)


def _domain_word(argument: sp.Basic) -> str | None:
    if isinstance(argument, sp.Symbol) and argument.name in _DOMAIN_WORDS:
        return argument.name
    return None


def _real_bound(value: sp.Basic) -> bool:
    """Whether `value` can bound a search: a real number or an infinity."""
    if value in (sp.oo, -sp.oo):
        return True
    return (
        isinstance(value, sp.Expr)
        and value.is_number is True
        and value.is_extended_real is True
    )


# -- the first argument ---------------------------------------------------------


#: The connectives of a formula with an expression for an operand, `u AND v = w`, which
#: sympy declines (see `Logical`).
_CONNECTIVES = {"AND": sp.And, "OR": sp.Or, "NOT": sp.Not}


#: The heads whose limits may be infinite while their value is finite, and the
#: conditional, infinite in some cases only.
_OVER_LIMITS = (sp.Integral, sp.Sum, sp.Product, sp.Limit, sp.Piecewise)


def _unbounded(side: sp.Basic) -> bool:
    """Whether a side of an equation holds an infinity or `?` outside the limits of an
    integral, sum, product or limit and outside a conditional's cases:
    `INT(#e^(-s*t), t, 0, inf)` is no infinity, nor is its value `IF(s > 0, 1/s,
    inf)`."""
    if side.has(*_OVER_LIMITS):
        side = side.replace(
            lambda found: isinstance(found, _OVER_LIMITS), lambda found: sp.Dummy()
        )
    return side.has(sp.zoo, sp.oo, -sp.oo, sp.nan)


def _formula(target: sp.Basic) -> sp.Basic | None:
    """The first argument as one Boolean formula over relations, or None if it isn't
    one: a vector or matrix is the conjunction of its elements, an expression `u` the
    equation `u = 0`, also as an operand of `AND`, `OR` or `NOT`."""
    if isinstance(target, Logical):
        operator, *operands = target.args
        connective = _CONNECTIVES.get(str(operator))
        parts = [_formula(operand) for operand in operands]
        if connective is None or any(part is None for part in parts):
            return None
        return connective(*parts)
    if isinstance(target, Vector):
        parts = []
        for element in _leaves(target):
            part = _formula(element)
            if part is None:
                return None
            parts.append(part)
        return sp.And(*parts)
    if isinstance(target, Relational) and isinstance(target.lhs, Relational):
        # A chain the operand computed, as `DIF(v = h^2, h) = 0` does: `0 = 2*h = 0`
        # is the conjunction of its links.
        inner = _formula(target.lhs)
        if inner is None:
            return None
        link = target.func(target.lhs.rhs, target.rhs, evaluate=False)
        return sp.And(inner, link)
    if isinstance(target, sp.Equality) and any(
        _unbounded(side) for side in target.args
    ):
        # An equation with an infinite or undefined side has no solution:
        # `TAN(2*alpha) = 4*SQRT(3)/0` is `TAN(2*alpha) = +-inf`.
        return sp.false
    if isinstance(target, (BooleanAtom, Relational)):
        return target
    if isinstance(target, (sp.And, sp.Or, sp.Not, sp.Xor, sp.Implies, sp.Equivalent)):
        return target
    if type(target) is sp.Symbol:
        # A variable is a `Boolean` to sympy, but it is the equation `x = 0` here.
        return sp.Eq(target, 0, evaluate=False)
    if isinstance(target, Boolean) or isinstance(target, StringLiteral):
        return None
    if isinstance(target, sp.Expr) and not is_set(target):
        return sp.Eq(target, 0, evaluate=False)
    return None


def _leaves(vector: Vector) -> list[sp.Basic]:
    """The elements of a vector, a matrix read row by row."""
    found = []
    for element in vector.args:
        if isinstance(element, Vector):
            found.extend(_leaves(element))
        else:
            found.append(element)
    return found


def _conjunctions(formula: sp.Basic) -> list[list[sp.Basic]] | None:
    """The formula in disjunctive normal form, as the relations of each conjunction, or
    None if one of them is no relation (a Boolean variable)."""
    if formula is sp.true:
        return [[]]
    if formula is sp.false:
        return []
    if isinstance(formula, Relational):
        return [[formula]]
    # A solution's equations in their order are a conjunction like any other: `v AND
    # SOLVE(...)` joins them with more.
    formula = formula.replace(
        lambda found: isinstance(found, InOrder), lambda found: sp.And(*found.args)
    )
    try:
        normal = sp.to_dnf(formula)
    except Exception:
        return None
    found = []
    for disjunct in sp.Or.make_args(normal):
        if disjunct is sp.true:
            return [[]]
        if disjunct is sp.false:
            continue
        literals = []
        # A conjunction's operands come in no fixed order; solved in a fixed one, the
        # elimination is the same every time.
        for literal in sorted(sp.And.make_args(disjunct), key=sp.default_sort_key):
            if isinstance(literal, sp.Not) and isinstance(literal.args[0], Relational):
                literal = literal.args[0].negated
            if not isinstance(literal, Relational):
                return None
            literals.append(literal)
        found.append(literals)
    return found


def _named(formula: sp.Basic, context: Context) -> tuple[str, ...]:
    """The variables of `formula` in the variable order, for the one-argument form."""
    names = {
        symbol.name
        for symbol in formula.free_symbols
        if type(symbol) is sp.Symbol
    }
    return main_order(names, context.order)


# -- what was found ---------------------------------------------------------------


@dataclass
class _Found:
    """The solutions of a problem, in a form each function writes its own way."""

    #: Every value of the variables is a solution.
    everything: bool = False
    #: The explicit solutions, one value per variable; a free value is a minted
    #: constant.
    rows: list[tuple[sp.Basic, ...]] = field(default_factory=list)
    #: Disjuncts that are no explicit solution: intervals, unsolved relations,
    #: conditions on parameters, reduced systems, and solutions that hold under
    #: conditions on the parameters.
    others: list[sp.Basic | _Branch] = field(default_factory=list)
    #: Whether a disjunct holds an interval of values, which no vector of values lists.
    intervals: bool = False
    #: The constants minted for this answer.
    minted: frozenset[sp.Symbol] = frozenset()
    #: The constants that stand for every value, when the solver minted them.
    arbitrary: tuple[sp.Symbol, ...] = ()
    #: The system each solution with free values was found from, as the conjunction of
    #: its relations, for `SOLVE`'s reduced form.
    reduced: dict[tuple[sp.Basic, ...], sp.Basic] = field(default_factory=dict)

    def merge(self, other: _Found) -> None:
        self.everything = self.everything or other.everything
        for row in other.rows:
            if row not in self.rows:
                self.rows.append(row)
        for part in other.others:
            if isinstance(part, _Branch):
                same = next(
                    (
                        mine
                        for mine in self.others
                        if isinstance(mine, _Branch)
                        and mine.conditions == part.conditions
                    ),
                    None,
                )
                if same is not None:
                    # Solutions under the same conditions are one branch, in the order
                    # of solutions, whichever conjunction they come from.
                    same.found.merge(part.found)
                    same.found.rows = listed_in_order(same.found.rows)
                    continue
            if part not in self.others:
                self.others.append(part)
        self.intervals = self.intervals or other.intervals
        self.minted = self.minted | other.minted
        self.arbitrary = self.arbitrary or other.arbitrary
        self.reduced.update(other.reduced)


@dataclass
class _Branch:
    """Solutions that hold where conditions on the parameters do: `x^2 = 4 AND a = b`
    solved for `x` is the solutions of `x^2 = 4` under `a = b`. They are kept apart from
    their conditions so that they are bounded, polished and written as the same
    solutions without the conditions are, and so that the answer states the conditions
    once."""

    conditions: tuple[sp.Basic, ...]
    found: _Found


def _solve_all(
    conjunctions: list[list[sp.Basic]],
    symbols: tuple[sp.Symbol, ...],
    context: Context,
    divisors: Sequence[sp.Basic],
    real: bool,
    listing: bool,
) -> _Found:
    """Each conjunction solved for `symbols`, the answers joined. `listing` is for
    `SOLUTIONS`, which solves a single equation in several variables as a system rather
    than factor by factor, and keeps the constants it mints."""
    found = _Found()
    names = tuple(symbol.name for symbol in symbols)
    for relations in conjunctions:
        conditions = [
            relation for relation in relations if not relation.has(*symbols)
        ]
        truths = [_truth(condition) for condition in conditions]
        if sp.false in truths:
            continue
        conditions = [
            condition
            for condition, truth in zip(conditions, truths, strict=True)
            if truth is not sp.true
        ]
        if conditions and len(conditions) < len(relations):
            found.merge(
                _conditioned(
                    [relation for relation in relations if relation not in conditions],
                    conditions,
                    symbols,
                    context,
                    divisors,
                    real,
                    listing,
                )
            )
            continue
        if not relations:
            found.everything = True
            continue
        if len(symbols) > 1 and all(
            isinstance(relation, sp.Unequality) for relation in relations
        ):
            # Inequations in several variables are no system to solve: they stay as
            # they are (see `_is_interval`).
            found.others.append(sp.And(*relations))
            continue
        if len(relations) == 1 and len(symbols) > 1 and not listing:
            found.merge(_single(relations[0], symbols, context, divisors, real))
            continue
        single = len(relations) == 1 and len(symbols) == 1
        problem = relations[0] if single else Vector(*relations)
        result = outcome(
            problem,
            context,
            names,
            None,
            divisors if single else (),
            real,
            commit=listing,
        )
        found.merge(
            _classified(result, symbols, relations, context, bool(conditions))
        )
    found.rows = listed_in_order(found.rows)
    # The conjunctions come in no fixed order, so neither would the branches.
    places = [
        index for index, part in enumerate(found.others) if isinstance(part, _Branch)
    ]
    branches = sorted(
        (found.others[index] for index in places),
        key=lambda branch: [sp.default_sort_key(c) for c in branch.conditions],
    )
    for index, branch in zip(places, branches, strict=True):
        found.others[index] = branch
    return found


def _truth(relation: sp.Basic) -> sp.Basic | None:
    """The truth value of a relation in none of the variables where it is known, else
    None."""
    try:
        if isinstance(relation, sp.Equality):
            zero = (relation.lhs - relation.rhs).is_zero
            return None if zero is None else sp.true if zero else sp.false
        if isinstance(relation, Relational):
            value = relation.func(relation.lhs, relation.rhs)
            return value if isinstance(value, BooleanAtom) else None
    except Exception:
        return None
    return None


def _conditioned(
    relations: list[sp.Basic],
    conditions: list[sp.Basic],
    symbols: tuple[sp.Symbol, ...],
    context: Context,
    divisors: Sequence[sp.Basic],
    real: bool,
    listing: bool,
) -> _Found:
    """A conjunction with relations in none of the variables, `conditions`, solved: the
    other relations are solved, and each of their solutions holds where the conditions
    do. `2*y*COS(t) = 3*SIN(2*t) AND SIN(2*t) = 0` solved for `y` is `y = ... AND SIN(2*t)
    = 0`. The conditions are written as given; `SOLUTIONS` lists such a solution as
    `IF(conditions, value)`."""
    inner = _solve_all([relations], symbols, context, divisors, real, listing)
    found = _Found(minted=inner.minted)
    if inner.everything or inner.rows or inner.others:
        found.others.append(_Branch(tuple(conditions), inner))
    found.intervals = inner.intervals
    return found


def _single(
    relation: Relational,
    symbols: tuple[sp.Symbol, ...],
    context: Context,
    divisors: Sequence[sp.Basic],
    real: bool,
) -> _Found:
    """One relation solved for several variables, as `SOLVE` does it.

    An equation is solved factor by factor, each factor for the one listed variable in
    it: `2*x*(y^2 - b^2)` is `x = 0 OR y = -b OR y = b`. A factor in two of them is left
    as an equation, and one in none is a condition on the parameters. An inequality in
    several variables describes a region, so it is left as it is.
    """
    found = _Found()
    if not isinstance(relation, sp.Equality):
        found.others.append(relation)
        found.intervals = True
        return found
    residual = relation.lhs - relation.rhs
    try:
        numerator = sp.fraction(sp.together(residual))[0]
        _, factors = sp.factor_list(numerator)
    except Exception:
        factors = [(residual, 1)]
    for factor, _ in factors:
        if factor.is_number:
            if factor.is_zero:
                found.everything = True
            continue
        present = [symbol for symbol in symbols if factor.has(symbol)]
        if not present:
            condition = parameter_condition(factor, context)
            if condition is not None:
                found.others.append(condition)
            continue
        if len(present) > 1:
            found.others.append(unsolved_equation(factor, context))
            continue
        (symbol,) = present
        result = outcome(
            sp.Eq(factor, 0, evaluate=False),
            context,
            (symbol.name,),
            None,
            divisors,
            real,
            commit=False,
        )
        part = _classified(result, (symbol,), [relation], context)
        if part.everything:
            found.everything = True
            continue
        found.others.extend(_equated(symbol, row[0], context) for row in part.rows)
        found.others.extend(part.others)
        found.intervals = found.intervals or part.intervals
    return found


def _classified(
    result: Outcome,
    symbols: tuple[sp.Symbol, ...],
    relations: list[sp.Basic],
    context: Context,
    free: bool = False,
) -> _Found:
    """The entries of a solve sorted into what each function writes. `free` says that
    the relations are in none of the variables, so that each entry in none of them is a
    condition under which every value is a solution: `a = b` solved for `x`."""
    found = _Found(minted=result.minted)
    names = [symbol.name for symbol in symbols]
    for entry in result.entries:
        if free and not isinstance(entry, BooleanAtom) and not entry.has(*symbols):
            parts = entry.args if isinstance(entry, Vector) else (entry,)
            found.others.append(_Branch(tuple(parts), _Found(everything=True)))
            continue
        row = _row(entry, names)
        if row is not None:
            if all(result.is_arbitrary(value) for value in row):
                found.everything = True
                found.arbitrary = row
            else:
                if any(value.has(*result.minted) for value in row):
                    found.reduced[row] = sp.And(*relations)
                found.rows.append(row)
            continue
        branch = _conditional_entry(entry, symbols, result)
        if branch is not None:
            found.others.append(branch)
            continue
        if isinstance(entry, Vector):
            entry = InOrder(*entry.args)
        found.others.append(entry)
        if _is_interval(entry, names):
            found.intervals = True
    return found


def _row(entry: sp.Basic, names: list[str]) -> tuple[sp.Basic, ...] | None:
    """The values of an entry that solves for every variable in order, or None."""
    relations = list(entry.args) if isinstance(entry, Vector) else [entry]
    if len(relations) != len(names):
        return None
    values = []
    for relation, name in zip(relations, names, strict=True):
        if not isinstance(relation, sp.Equality):
            return None
        side = relation.lhs
        if not (isinstance(side, sp.Symbol) and side.name == name):
            return None
        if any(
            getattr(symbol, "name", None) in names
            for symbol in relation.rhs.free_symbols
        ):
            return None
        values.append(relation.rhs)
    return tuple(values)


def _conditional_entry(
    entry: sp.Basic, symbols: tuple[sp.Symbol, ...], result: Outcome
) -> _Branch | None:
    """An entry of a system's solution that the solver found under conditions on the
    parameters, its values followed by the conditions, as `[x = a, a - b = 0]` for `[x =
    a, x = b]`, or None for any other entry. A solution with free values is left to
    `_conditioned_underdetermined`."""
    if not isinstance(entry, Vector) or len(entry.args) <= len(symbols):
        return None
    names = [symbol.name for symbol in symbols]
    row = _row(Vector(*entry.args[: len(symbols)]), names)
    conditions = entry.args[len(symbols) :]
    if row is None or any(value.has(*result.minted) for value in row):
        return None
    if any(
        not isinstance(condition, Relational) or condition.has(*symbols)
        for condition in conditions
    ):
        return None
    return _Branch(tuple(conditions), _Found(rows=[row]))


def _is_interval(entry: sp.Basic, names: list[str]) -> bool:
    """Whether an entry that solves nothing explicitly states a range of a variable: an
    inequality or a chain of them in a solve variable, or an inequation in the one
    variable (all values but some). An inequation in several variables that the solver
    left as it is is unsolved, and like any unsolved relation it is dropped."""
    excluded = sp.Equality if len(names) == 1 else (sp.Equality, sp.Unequality)
    for part in sp.And.make_args(entry):
        if isinstance(part, Relational) and not isinstance(part, excluded):
            if any(
                getattr(symbol, "name", None) in names for symbol in part.free_symbols
            ):
                return True
    return False


def _equated(symbol: sp.Symbol, value: sp.Basic, context: Context) -> sp.Basic:
    """`symbol = value`, in normal form. Under `Angle := Degree`, the solution of a
    trigonometric equation may be an angle in degrees, which stays as it is."""
    if value.has(DEGREE):
        return sp.Eq(symbol, value, evaluate=False)
    value = normal_form(value, context.order)
    coefficient, rest = value.as_coeff_Mul()
    if context.precision is not Precision.EXACT and coefficient != 1 and rest.is_Add:
        # Shown in digits, a number is distributed over a sum rather than kept as a
        # quotient of long integers: `t = -31.80084241*COS(s) - 96.14099856`.
        value = sp.Add(*(coefficient * term for term in rest.args))
    return sp.Eq(symbol, value, evaluate=False)


def _string_variables(
    args: Sequence[sp.Basic],
) -> tuple[Sequence[sp.Basic], dict[sp.Symbol, StringLiteral]]:
    """The arguments with each string the variables argument names, `"a"` in
    `SOLVE("a" + 1 = 0, "a")`, made a variable, and the strings by those variables.

    A string is a variable Derive solves for: RationalApproximation's `PADE` names the
    unknown coefficients by strings. Each becomes a symbol no name can spell, and
    `_restored` puts the strings back into the answer.
    """
    if len(args) < 2:
        return args, {}
    strings = {
        found
        for found in sp.preorder_traversal(args[1])
        if isinstance(found, StringLiteral)
    }
    if not strings:
        return args, {}
    symbols = {string: sp.Symbol(f'"{string.name}"') for string in strings}
    replaced = [
        argument.xreplace(symbols) if isinstance(argument, sp.Basic) else argument
        for argument in args
    ]
    return replaced, {symbol: string for string, symbol in symbols.items()}


def _restored(
    answer: sp.Basic | None, strings: dict[sp.Symbol, StringLiteral]
) -> sp.Basic | None:
    """`answer` with the strings `_string_variables` made variables put back."""
    if answer is None or not strings:
        return answer
    return answer.xreplace(strings)


# -- SOLVE ----------------------------------------------------------------------


def solve_head(
    args: Sequence[sp.Basic],
    divisors: Sequence[sp.Basic],
    context: Context,
    decide: Decider,
    vector: bool,
) -> sp.Basic | None:
    """`SOLVE(u)`, `SOLVE(u, x)`, `SOLVE(u, x, domain)` and `SOLVE(u, x, a, b)`, or None
    for a call of no such form, which stays as written.

    `vector` says the first argument was written as a vector, which makes the answer a
    vector of the disjuncts. `decide` gives the truth value of a relation without
    variables.
    """
    args, strings = _string_variables(args)
    return _restored(_solved(args, divisors, context, decide, vector), strings)


def _solved(
    args: Sequence[sp.Basic],
    divisors: Sequence[sp.Basic],
    context: Context,
    decide: Decider,
    vector: bool,
) -> sp.Basic | None:
    """`solve_head` with the variables symbols."""
    call = _read(args, one=True)
    if call is None:
        return None
    formula = _formula(call.target)
    if formula is None:
        return None
    names = call.names or _named(formula, context)
    if not names:
        decided = decide(formula)
        if not vector:
            return decided
        return Vector() if decided is sp.false else Vector(decided)
    conjunctions = _conjunctions(formula)
    if conjunctions is None:
        return None
    symbols = _symbols(formula, names)
    found = _solve_all(conjunctions, symbols, context, divisors, call.real, False)
    _polish(found, decide)
    if call.bounds is not None:
        found = _bounded(found, symbols, call.bounds, context, call.real)
    _possible(found, symbols, decide)
    if context.angle is Angle.DEGREE and formula.has(TrigonometricFunction):
        # The angles of a trigonometric equation are shown in degrees.
        _in_degrees(found)
    disjuncts = _disjuncts(found, symbols, context)
    if vector:
        return Vector(*(_conjoined(*disjunct) for disjunct in disjuncts))
    if disjuncts == [(sp.true, ())]:
        return sp.true
    # Derive builds the disjunction back to front: SOLVE lists the solutions in the
    # reverse of the order of SOLUTIONS.
    return _either(list(reversed(disjuncts)))


def _conjoined(solution: sp.Basic, conditions: tuple[sp.Basic, ...]) -> sp.Basic:
    """A solution and the conditions under which it holds, as one conjunction, the
    solution's relations in their order."""
    relations = solution.args if isinstance(solution, sp.And) else (solution,)
    return InOrder(*relations, *conditions)


def _either(disjuncts: list[tuple[sp.Basic, tuple[sp.Basic, ...]]]) -> sp.Basic:
    """The disjunction of the solutions, in their order, each run of solutions under the
    same conditions written with the conditions once: `(x = -2 OR x = 2) AND a = b`."""
    written = []
    for conditions, run in itertools.groupby(disjuncts, key=lambda pair: pair[1]):
        solutions = [solution for solution, _ in run]
        if conditions and len(solutions) > 1:
            written.append(InOrder(InOrderOr(*solutions), *conditions))
        else:
            written.extend(_conjoined(solution, conditions) for solution in solutions)
    return InOrderOr(*written)


def _polish(found: _Found, simplified: Decider) -> None:
    """Simplify the values of the solutions of a system that hold a root of a quotient
    with a variable denominator, or a quotient with a root in its variable denominator,
    where that makes them no larger.

    Substituting into the other equations of a system leaves such roots as one radical
    of an expanded quotient, `SQRT(P/(r1 - r2)^2)`, which Simplify writes with the
    denominator taken out and the radicand factored: `SQRT(Q)/ABS(r1 - r2)`. Solving
    for one variable and substituting leaves quotients of sums of roots, which Simplify
    may rationalize and cancel.
    """
    for index, row in enumerate(found.rows):
        if len(row) < 2 or row in found.reduced:
            continue
        polished = tuple(
            _simpler(value, simplified)
            if _root_of_quotient(value) or _quotient_over_root(value)
            else value
            for value in row
        )
        if polished != row and polished not in found.rows:
            found.rows[index] = polished
    for part in found.others:
        if isinstance(part, _Branch):
            _polish(part.found, simplified)


def _in_degrees(found: _Found) -> None:
    """Write the angles of the solutions in degrees, as `Angle := Degree` asks."""
    found.rows = [tuple(in_degrees(value) for value in row) for row in found.rows]
    for part in found.others:
        if isinstance(part, _Branch):
            _in_degrees(part.found)


def _root_of_quotient(value: sp.Basic) -> bool:
    return any(
        power.exp.is_Rational
        and not power.exp.is_Integer
        and sp.fraction(sp.together(power.base))[1].free_symbols
        for power in value.atoms(sp.Pow)
    )


def _quotient_over_root(value: sp.Basic) -> bool:
    if not isinstance(value, sp.Expr):
        return False
    denominator = sp.fraction(sp.together(value))[1]
    return bool(denominator.free_symbols) and any(
        power.exp.is_Rational and not power.exp.is_Integer and not power.base.is_number
        for power in denominator.atoms(sp.Pow)
    )


def _simpler(value: sp.Basic, simplified: Decider) -> sp.Basic:
    try:
        candidate = simplified(value)
    except Exception:
        return value
    if not isinstance(candidate, sp.Expr) or candidate.has(sp.nan, sp.zoo):
        return value
    return candidate if sp.count_ops(candidate) <= sp.count_ops(value) else value


def _symbols(formula: sp.Basic, names: Sequence[str]) -> tuple[sp.Symbol, ...]:
    """The variables named, as the symbols the formula holds, so their declarations
    count."""
    held = {
        symbol.name: symbol
        for symbol in formula.free_symbols
        if type(symbol) is sp.Symbol
    }
    return tuple(held.get(name, sp.Symbol(name)) for name in names)


def _disjuncts(
    found: _Found, symbols: tuple[sp.Symbol, ...], context: Context
) -> list[tuple[sp.Basic, tuple[sp.Basic, ...]]]:
    """What `SOLVE` writes for `found`, one disjunct each, as the solution and the
    conditions on the parameters it holds under: `[(true, ())]` for everything."""
    if found.everything:
        return [(sp.true, ())]
    disjuncts: list[tuple[sp.Basic, tuple[sp.Basic, ...]]] = []
    reduced_done: list[sp.Basic] = []
    for row in found.rows:
        if any(value.has(*found.minted) for value in row):
            written = _underdetermined(row, symbols, found, context, reduced_done)
            if written is None:
                continue
            if written is sp.true:
                return [(sp.true, ())]
        elif len(row) == 1:
            written = _equated(symbols[0], row[0], context)
        else:
            written = InOrder(
                *(
                    _equated(symbol, value, context)
                    for symbol, value in zip(symbols, row, strict=True)
                )
            )
        if (written, ()) not in disjuncts:
            disjuncts.append((written, ()))
    for part in found.others:
        if isinstance(part, _Branch):
            for solution, conditions in _disjuncts(part.found, symbols, context):
                disjunct = (solution, (*conditions, *part.conditions))
                if disjunct not in disjuncts:
                    disjuncts.append(disjunct)
            continue
        if part is sp.true:
            return [(sp.true, ())]
        part = _conditioned_underdetermined(part, symbols, found, context)
        if part is not sp.false and (part, ()) not in disjuncts:
            disjuncts.append((part, ()))
    return disjuncts


def _underdetermined(
    row: tuple[sp.Basic, ...],
    symbols: tuple[sp.Symbol, ...],
    found: _Found,
    context: Context,
    done: list[sp.Basic],
) -> sp.Basic | None:
    """A solution with free values, as `SOLVE` writes it: a linear system reduced to
    row echelon form, otherwise the solution with each free value named by its
    variable again. None if its system is already written."""
    relations = found.reduced.get(row)
    if relations is not None:
        if relations in done:
            return None
        echelon = _echelon(list(sp.And.make_args(relations)), symbols, context)
        if echelon is not None:
            done.append(relations)
            return echelon
    named = {
        value: symbol
        for symbol, value in zip(symbols, row, strict=True)
        if value in found.minted
    }
    equations = [
        _equated(symbol, value.xreplace(named), context)
        for symbol, value in zip(symbols, row, strict=True)
        if value not in found.minted
    ]
    return InOrder(*equations)


def _conditioned_underdetermined(
    part: sp.Basic,
    symbols: tuple[sp.Symbol, ...],
    found: _Found,
    context: Context,
) -> sp.Basic:
    """A solution with free values under conditions on the parameters, as `SOLVE`
    writes it: the solution as `_underdetermined` writes one, then the conditions. `x =
    @1 AND y = 1 - @1 AND k = 2` is `x + y = 1 AND k = 2`. Any other disjunct is
    returned as it is."""
    if not isinstance(part, sp.And) or not found.minted:
        return part
    values: dict[sp.Basic, sp.Basic] = {}
    conditions = []
    for relation in part.args:
        if not relation.has(*symbols):
            conditions.append(relation)
        elif (
            isinstance(relation, sp.Equality)
            and relation.lhs in symbols
            and relation.lhs not in values
            and not relation.rhs.has(*symbols)
        ):
            values[relation.lhs] = relation.rhs
        else:
            return part
    if len(values) != len(symbols) or not conditions:
        return part
    row = tuple(values[symbol] for symbol in symbols)
    if not any(value.has(*found.minted) for value in row):
        return part
    named = {
        value: symbol
        for symbol, value in zip(symbols, row, strict=True)
        if value in found.minted
    }
    equations = [
        _equated(symbol, value.xreplace(named), context)
        for symbol, value in zip(symbols, row, strict=True)
        if value not in found.minted
    ]
    if any(equation.has(*found.minted) for equation in equations):
        return part
    written = _echelon(equations, symbols, context) if equations else None
    if written is None:
        return InOrder(*equations, *conditions)
    rows = written.args if isinstance(written, sp.And) else (written,)
    return InOrder(*rows, *conditions)


def _echelon(
    relations: list[sp.Basic], symbols: tuple[sp.Symbol, ...], context: Context
) -> sp.Basic | None:
    """A consistent underdetermined linear system as its reduced row echelon form, with
    the columns in the order of the variables, or None if it isn't linear.

    Each row is written with its pivot variable first, then the terms in the free
    variables, and the rest on the right: `2*x + 6*y = 2 AND 3*x = 3 - 9*y` is `x + 3*y
    = 1`.
    """
    try:
        residuals = [relation.lhs - relation.rhs for relation in relations]
        matrix, right = sp.linear_eq_to_matrix(residuals, list(symbols))
        augmented = matrix.row_join(right)
        echelon, pivots = augmented.rref(
            iszerofunc=lambda value: sp.cancel(sp.together(value)).is_zero,
            simplify=lambda value: sp.cancel(sp.together(value)),
        )
    except Exception:
        return None
    if len(symbols) in pivots:
        return None
    rows = []
    for index, pivot in enumerate(pivots):
        left = sum(
            (
                echelon[index, column] * symbol
                for column, symbol in enumerate(symbols)
                if column not in pivots or column == pivot
            ),
            sp.S.Zero,
        )
        value = echelon[index, len(symbols)]
        left = normal_form(sp.cancel(left), context.order)
        value = normal_form(value, context.order)
        rows.append(sp.Eq(left, value, evaluate=False))
    return InOrder(*rows)


def _bounded(
    found: _Found,
    symbols: tuple[sp.Symbol, ...],
    bounds: tuple[sp.Basic, sp.Basic],
    context: Context,
    real: bool,
) -> _Found:
    """`found` cut to the real solutions in `[a, b]`, each proved inside or outside
    exactly. A solution that is neither holds where it is inside: `x = SQRT(a) AND
    SQRT(a) <= 1`. Where every value is a solution, the solutions are every value in
    the bounds, `a <= x <= b`, and a range is cut to them: `x > 2` in `[0, 5]` is `2 <
    x <= 5`. A range that can't be cut exactly holds where it is inside them. Solutions
    under conditions on the parameters are cut as the same solutions without them
    are.

    In Approximate and Mixed precision an equation left unsolved is searched in the
    interval numerically, as `NSOLVE` would.
    """
    low, high = bounds
    kept = _Found(minted=found.minted)
    if len(symbols) != 1:
        return found
    (symbol,) = symbols
    if found.everything:
        if bounds == (-sp.oo, sp.oo):
            return found
        # Every value in the bounds, of which each other solution found is one.
        everything = confined(None, symbol, bounds, context)
        if everything is None:
            everything = (InOrder(*between(symbol, bounds)),)
        _cut(kept, everything, symbol)
        return kept
    for row in found.rows:
        inside = _inside(row[0], low, high)
        if inside:
            kept.rows.append(row)
        elif inside is None:
            (value,) = row
            kept.others.append(
                InOrder(_equated(symbol, value, context), *between(value, bounds))
            )
    for part in found.others:
        if isinstance(part, _Branch):
            inner = _bounded(part.found, symbols, bounds, context, real)
            if inner.rows or inner.others:
                kept.others.append(_Branch(part.conditions, inner))
                kept.intervals = kept.intervals or inner.intervals
            continue
        if context.precision is not Precision.EXACT and _searchable(part, symbols[0]):
            roots = numeric_roots(
                part.lhs - part.rhs, symbols[0], context, True, (low, high)
            )
            if roots is not None:
                kept.rows.extend((root,) for root in roots)
                continue
        if _is_interval(part, [symbol.name]) and solved_range(part, symbol):
            cut = confined(part, symbol, bounds, context)
            if cut is None:
                cut = (InOrder(*sp.And.make_args(part), *between(symbol, bounds)),)
            _cut(kept, cut, symbol)
            continue
        kept.others.append(part)
        kept.intervals = kept.intervals or found.intervals
    return kept


def _possible(
    found: _Found,
    symbols: tuple[sp.Symbol, ...],
    decide: Decider,
    outer: tuple[sp.Basic, ...] = (),
) -> None:
    """Drop the solutions whose conditions on the parameters hold for no value of them,
    as Simplify decides them: `x = -SQRT(a)` inside `[1, 2]` needs `1 <= -SQRT(a)`,
    which no real `a` satisfies. `outer` are the conditions of the enclosing branches,
    which hold too."""
    kept: list[sp.Basic | _Branch] = []
    for part in found.others:
        if isinstance(part, _Branch):
            conditions = (*outer, *part.conditions)
            if _never(conditions, decide):
                continue
            _possible(part.found, symbols, decide, conditions)
            if not (part.found.everything or part.found.rows or part.found.others):
                continue
        else:
            conditions = (
                *outer,
                *(
                    relation
                    for relation in sp.And.make_args(part)
                    if not relation.has(*symbols)
                ),
            )
            if len(conditions) > len(outer) and _never(conditions, decide):
                continue
        kept.append(part)
    found.others = kept


def _never(conditions: Sequence[sp.Basic], decide: Decider) -> bool:
    """Whether Simplify decides the conjunction of `conditions` false."""
    try:
        return decide(sp.And(*conditions)) is sp.false
    except Exception:
        return False


def _cut(kept: _Found, entries: Sequence[sp.Basic], symbol: sp.Symbol) -> None:
    """Add the entries of a range cut to bounds to `kept`: a value as a solution, a
    range as one that holds an interval."""
    for entry in entries:
        if isinstance(entry, sp.Equality) and entry.lhs == symbol:
            if (entry.rhs,) not in kept.rows:
                kept.rows.append((entry.rhs,))
        elif entry not in kept.others:
            kept.others.append(entry)
            kept.intervals = True


def _inside(value: sp.Basic, low: sp.Basic, high: sp.Basic) -> bool | None:
    """Whether `value` is real and in `[low, high]`, or None if that can't be decided
    exactly."""
    try:
        if value.is_extended_real is False:
            return False
        above = True if low is -sp.oo else (value - low).is_nonnegative
        below = True if high is sp.oo else (high - value).is_nonnegative
        if above is False or below is False:
            return False
        if above and below and value.is_extended_real:
            return True
        return None
    except Exception:
        return None


def _searchable(part: sp.Basic, symbol: sp.Symbol) -> bool:
    """Whether `part` is an equation in `symbol` alone, which a numeric search solves."""
    return isinstance(part, sp.Equality) and part.free_symbols == {symbol}


# -- SOLUTIONS ------------------------------------------------------------------


def solutions_head(
    args: Sequence[sp.Basic],
    divisors: Sequence[sp.Basic],
    context: Context,
    decide: Decider,
) -> sp.Basic | None:
    """`SOLUTIONS(u, x)` and its longer forms, or None for a call of no such form.

    With one variable the vector of its values, with a vector of variables the matrix
    of the solutions. A solution set holding an interval has no vector of values, so
    such a call stays as written.
    """
    args, strings = _string_variables(args)
    return _restored(_solutions(args, divisors, context, decide), strings)


def _solutions(
    args: Sequence[sp.Basic],
    divisors: Sequence[sp.Basic],
    context: Context,
    decide: Decider,
) -> sp.Basic | None:
    """`solutions_head` with the variables symbols."""
    call = _read(args, one=False)
    if call is None:
        return None
    formula = _formula(call.target)
    if formula is None:
        return None
    conjunctions = _conjunctions(formula)
    if conjunctions is None:
        return None
    symbols = _symbols(formula, call.names)
    if not (formula.free_symbols & set(symbols)) and not isinstance(
        formula, BooleanAtom
    ):
        # A formula in none of the variables is solved by every value of them where
        # it holds, so only one decided false has no solution.
        decided = decide(formula)
        if decided is sp.true:
            conjunctions = [[]]
        elif decided is sp.false:
            conjunctions = []
    found = _solve_all(conjunctions, symbols, context, divisors, call.real, True)
    _polish(found, decide)
    if call.bounds is not None:
        found = _bounded(found, symbols, call.bounds, context, call.real)
    _possible(found, symbols, decide)
    if found.intervals:
        return None
    if found.everything:
        constants = found.arbitrary or tuple(arbitrary(context) for _ in symbols)
        row = Vector(*constants)
        return Vector(row) if call.listed else Vector(constants[0])
    found_rows = list(found.rows)
    for part in found.others:
        for row in _conditional_rows(part, symbols, context):
            if row not in found_rows:
                found_rows.append(row)
    rows = [Vector(*row) for row in found_rows]
    if call.listed:
        return Vector(*rows)
    return Vector(*(row[0] for row in found_rows))


def _conditional_rows(
    part: sp.Basic | _Branch, symbols: tuple[sp.Basic, ...], context: Context
) -> list[tuple[sp.Basic, ...]]:
    """The values of a disjunct that is no explicit solution but holds explicit ones
    under conditions on the parameters, each value `IF(condition, value)`, in the order
    `SOLUTIONS` lists them. Every value under conditions is arbitrary constants under
    them: `[IF(a = b, @1)]`."""
    if not isinstance(part, _Branch):
        row = _conditional_row(part, symbols)
        return [] if row is None else [row]
    condition = sp.And(*part.conditions)
    if part.found.everything:
        constants = part.found.arbitrary or tuple(arbitrary(context) for _ in symbols)
        return [tuple(sp.Piecewise((value, condition)) for value in constants)]
    rows = [
        tuple(sp.Piecewise((value, condition)) for value in row)
        for row in part.found.rows
    ]
    for inner in part.found.others:
        if isinstance(inner, _Branch):
            inner = _Branch((*inner.conditions, *part.conditions), inner.found)
        else:
            inner = _conjoined(inner, part.conditions)
        rows.extend(_conditional_rows(inner, symbols, context))
    return rows


def _conditional_row(
    part: sp.Basic, symbols: tuple[sp.Basic, ...]
) -> tuple[sp.Basic, ...] | None:
    """The values of a disjunct that solves every variable explicitly under conditions on
    the parameters, `x = 2 AND a = b`, as `(IF(a = b, 2),)`, or None for any other
    disjunct."""
    if not isinstance(part, sp.And):
        return None
    values: dict[sp.Basic, sp.Basic] = {}
    conditions = []
    for relation in part.args:
        if not relation.has(*symbols):
            conditions.append(relation)
        elif (
            isinstance(relation, sp.Equality)
            and relation.lhs in symbols
            and relation.lhs not in values
            and not relation.rhs.has(*symbols)
        ):
            values[relation.lhs] = relation.rhs
        else:
            return None
    if len(values) != len(symbols) or not conditions:
        return None
    condition = sp.And(*conditions)
    return tuple(sp.Piecewise((values[symbol], condition)) for symbol in symbols)


# -- NSOLVE and NSOLUTIONS ------------------------------------------------------


def nsolve_head(
    args: Sequence[sp.Basic], context: Context, listing: bool
) -> sp.Basic | None:
    """`NSOLVE(u, x)` (`listing` off) or `NSOLUTIONS(u, x)` (on), with an optional
    domain or bounds, or None if the call takes no numeric solving: a system, a vector
    of variables, an inequality, or an equation with a parameter in it. For `NSOLVE`
    None also if the search neither finds a root nor proves there is none; `NSOLUTIONS`
    lists the roots found.
    """
    call = _read(args, one=False)
    if call is None or call.listed or len(call.names) != 1:
        return None
    target = call.target
    if isinstance(target, Relational) and not isinstance(target, sp.Equality):
        return None
    if isinstance(target, (Vector, Boolean)) and not isinstance(target, sp.Equality):
        return None
    residual = target.lhs - target.rhs if isinstance(target, sp.Equality) else target
    if not isinstance(residual, sp.Expr):
        return None
    symbols = _symbols(residual, call.names)
    (symbol,) = symbols
    if residual.free_symbols - {symbol}:
        return None
    if call.bounds is not None:
        low, high = call.bounds
        if not (low is -sp.oo or high is sp.oo) and (high - low).is_positive is not True:
            return None
        if low is sp.oo or high is -sp.oo:
            return None
    roots = numeric_roots(residual, symbol, context, call.real, call.bounds)
    if listing:
        return Vector(*(roots or ()))
    if roots is None:
        return None
    disjuncts = [sp.Eq(symbol, root, evaluate=False) for root in roots]
    return sp.Or(*disjuncts) if disjuncts else sp.false
