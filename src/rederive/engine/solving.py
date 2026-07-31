"""The mathematics of soLve: what the solutions of an equation are.

Below the pipeline rather than above it, for the reason `factoring` is: an
authored `SOLVE(u, x)` asks for exactly what the command asks for, and Simplify
is what evaluates an authored line. So the command in `solve` is the pipeline
and then this, and the `SOLVE` head is this reached from inside the pipeline.

What comes back is a tuple of *entries*, not a set of values, because that is
the shape the original answers in: one worksheet line per solution, each of
them a relation with the variable alone on the left. A system answers with one
entry holding the whole solution vector, an interval with one chained relation
`-2 < x < 2`, and a union of intervals with one entry per piece.

Three answers are not solutions and have to be told apart, because the command
does different things with them:

* The empty tuple is "no solutions found". Nothing is appended and the message
  line says so.
* An equation that holds for every value answers `x = @1`, an arbitrary-value
  constant. `Context.arbitrary_index` is where the counter that mints those
  comes in, and it is the session's rather than this module's so that no two
  solves in one worksheet ever mint the same one.
* An equation nothing here can solve answers with the residual equation -
  everything moved to the left, zero on the right - which is what the original
  appends for `3^x = x^2`. It is a normal one-entry answer and never an
  exception: this module promises what the rest of the engine promises, that
  nothing the parser produced makes it raise.

Where the original's answers and ours differ they differ in spelling, and
deliberately so in two places. A periodic equation gets a finite sample of
representatives near the origin, one per solution family, where Derive emits
redundant ones - `SIN(x) = 1/2` is `pi/6` and `5*pi/6` here and `pi/6`,
`5*pi/6`, `-7*pi/6` there. And multiple solutions come out in a stable order,
reals ascending and then the complex ones, where the original's order is
whatever its solver happened to produce.

Domains do not filter solutions (manual 5.4): a declaration says what a
variable *is*, which is what decides how an answer simplifies, and never which
of the answers are wanted. Solutions are sought over the complexes whatever a
variable was declared.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from fractions import Fraction

import sympy as sp
from sympy.core.relational import Relational

from rederive.engine.approximation import GUARD, simplest
from rederive.engine.context import Context, Precision
from rederive.engine.normal import normal_form
from rederive.engine.ordering import main_order
from rederive.engine.to_sympy import InertVector

__all__ = ["equations_of", "oriented", "solutions", "solution_variables"]

#: How many pieces a bracketing scan cuts an interval into before it starts
#: bisecting. Fine enough to separate the roots of anything an ordinary
#: worksheet holds, coarse enough that the scan itself costs nothing.
_SCAN = 64

#: Where Mixed mode's search starts, and how far the doubling goes before it
#: gives up and answers with the residual equation. The manual documents the
#: doubling and not the cap; the cap is ours, because a search that never ends
#: is not an answer.
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


def solutions(
    expression: sp.Basic,
    context: Context | None = None,
    variables: Sequence[str] = (),
    bounds: tuple[sp.Basic, sp.Basic] | None = None,
) -> tuple[sp.Basic, ...]:
    """The entries `expression` solves to, one per worksheet line.

    `variables` names what to solve for, in the order chosen; empty means work
    it out - the one variable of a scalar, or as many of a system's most main
    as it has equations. `bounds` is the interval a numeric search is confined
    to, and is what Approximate precision asks the user for; None leaves the
    precision mode to decide, which is exact everywhere but Mixed.
    """
    context = context or Context()
    constants = _Constants(context.arbitrary_index)
    try:
        return _solutions(expression, context, tuple(variables), bounds, constants)
    except Exception:
        # The engine never raises on parser output, and "I could not solve
        # this" is an answer rather than a failure: the residual equation says
        # exactly as much as is known.
        return _unsolvable(expression, context)


def solution_variables(
    expression: sp.Basic, context: Context, count: int = 1
) -> tuple[sp.Symbol, ...]:
    """The variables soLve works on where the user named none.

    Most main first, and `count` of them: one for a scalar equation, one per
    equation for a system. Fewer where the expression holds fewer.
    """
    names = main_order(_named_in(expression), context.order)
    symbols = _symbols_of(expression)
    return tuple(symbols.get(name, sp.Symbol(name)) for name in names[:count])


def equations_of(expression: sp.Basic) -> tuple[Relational, ...] | None:
    """The equations of a system, or None where this is no system.

    A system is a vector of relations, which is what the original solves and
    what the UI counts to decide how many variables to ask about. A vector of
    anything else is not one.
    """
    elements = _elements(expression)
    if not elements or not all(isinstance(e, Relational) for e in elements):
        return None
    return tuple(elements)


def oriented(solved: sp.Basic) -> sp.Basic:
    """A solved relation with the variable on the left.

    Sympy answers `-3 < x`; the answer to a question about `x` is written
    `x > -3`. Shared with Simplify, which reduces a conjunction of relations
    over one variable to the range it describes and writes it the same way.
    """
    if isinstance(solved, sp.And | sp.Or):
        return solved.func(*(oriented(part) for part in solved.args))
    if isinstance(solved, Relational) and solved.lhs.is_number:
        return solved.reversed
    return solved


# -- which of the three shapes this is ---------------------------------------


def _solutions(
    expression: sp.Basic,
    context: Context,
    variables: tuple[str, ...],
    bounds: tuple[sp.Basic, sp.Basic] | None,
    constants: _Constants,
) -> tuple[sp.Basic, ...]:
    equations = equations_of(expression)
    if equations is not None:
        return _system(equations, context, variables, constants)
    return _scalar(expression, context, variables, bounds, constants)


def _scalar(
    expression: sp.Basic,
    context: Context,
    variables: tuple[str, ...],
    bounds: tuple[sp.Basic, sp.Basic] | None,
    constants: _Constants,
) -> tuple[sp.Basic, ...]:
    """One equation or inequality, solved for one variable."""
    chosen = _chosen(expression, context, variables, 1)
    if not chosen:
        return _unsolvable(expression, context)
    variable = chosen[0]
    if isinstance(expression, Relational) and not isinstance(expression, sp.Equality):
        return _inequality(expression, variable, context, constants)
    return _equation(_residual(expression), variable, context, bounds, constants)


def _chosen(
    expression: sp.Basic,
    context: Context,
    variables: tuple[str, ...],
    count: int,
) -> tuple[sp.Symbol, ...]:
    """The variables to solve for: the user's answers, or the most main ones."""
    if not variables:
        return solution_variables(expression, context, count)
    symbols = _symbols_of(expression)
    return tuple(symbols.get(name, sp.Symbol(name)) for name in variables)


def _residual(expression: sp.Basic) -> sp.Basic:
    """What has to be zero: a relation's difference, or the expression itself.

    A bare expression is solved as `u = 0`, which is the manual's own rule and
    what makes `SOLVE(x^2 - 4, x)` mean what `SOLVE(x^2 - 4 = 0, x)` means.
    """
    if isinstance(expression, Relational):
        return expression.lhs - expression.rhs
    return expression


# -- equations ---------------------------------------------------------------


def _equation(
    residual: sp.Basic,
    variable: sp.Symbol,
    context: Context,
    bounds: tuple[sp.Basic, sp.Basic] | None,
    constants: _Constants,
) -> tuple[sp.Basic, ...]:
    """`residual = 0` solved for `variable`, by whatever the precision allows.

    Bounds mean a numeric search and nothing else, which is Approximate mode.
    Exact and Mixed both solve exactly first; where that answers nothing, Mixed
    goes looking for a real root over a widening interval and Exact stops.
    """
    if bounds is not None:
        return _numeric(residual, variable, bounds[0], bounds[1], context)
    if residual.is_zero:
        return (_equated(variable, constants.fresh(), context),)
    found = _values(residual, variable)
    if found is not None:
        entries = _entries(found, variable, context, constants)
        if entries is not None:
            return entries
    if context.precision is Precision.MIXED:
        searched = _searched(residual, variable, context)
        if searched is not None:
            return searched
    return _unsolved(residual, context)


def _values(residual: sp.Basic, variable: sp.Symbol) -> sp.Set | None:
    """Every value of `variable` that makes `residual` vanish, or None.

    None is "nothing here knows", which is a different answer from the empty
    set and is what the residual equation gets appended for.

    A polynomial goes to `roots` rather than to `solveset`, for two reasons the
    original shares: `roots` gives the radicals - Cardano's and Ferrari's
    included - where `solveset` is content to name a root by the polynomial it
    is a root of, and it collapses multiplicities, so `x^2*(8*x - 9)` has two
    solutions and not three. A polynomial it cannot split is one nothing can
    split in radicals, and that is exactly when the original gives up too.
    """
    polynomial = _polynomial(residual, variable)
    if polynomial is not None:
        if polynomial.degree() < 1:
            return sp.S.EmptySet if not polynomial.is_zero else sp.S.Complexes
        found = sp.roots(polynomial)
        if not found or sum(found.values()) != polynomial.degree():
            return None
        return sp.FiniteSet(*found)
    try:
        return sp.solveset(residual, variable, sp.S.Complexes)
    except Exception:
        return None


def _polynomial(residual: sp.Basic, variable: sp.Symbol) -> sp.Poly | None:
    """`residual` as a polynomial in `variable`, or None where it is not one."""
    try:
        return sp.Poly(residual, variable)
    except Exception:
        return None


def _entries(
    found: sp.Set,
    variable: sp.Symbol,
    context: Context,
    constants: _Constants,
) -> tuple[sp.Basic, ...] | None:
    """The relations one solution set is written as, in a stable order.

    None where the set is one no finite list of relations describes, which is
    the residual equation's case.
    """
    if _is_everything(found):
        return (_equated(variable, constants.fresh(), context),)
    values = _members(found)
    if values is None:
        return None
    return tuple(_equated(variable, value, context) for value in _ordered(values))


def _members(found: sp.Set) -> list[sp.Basic] | None:
    """What a solution set holds, as a finite list of values, or None.

    A periodic equation's answer is an image of the integers, and there is no
    notation for one: Derive has no `x = 2*pi*n` and neither have we. What is
    written instead is one representative per solution family, taken at the
    index nearest the origin, which is the finite sample the original prints -
    minus its redundancies, since each family contributes exactly once.
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
    return None


#: The indices a family is sampled at, nearest the origin first. Two periods
#: each way is enough to find the member nearest zero: no family's own step is
#: bigger than its period, so a third would only ever be further out.
_NEARBY = (0, -1, 1, -2, 2)


def _representative(family: sp.ImageSet) -> sp.Basic | None:
    """The member of one solution family nearest the origin.

    Which is what "a finite sample near the origin" comes to, and it is worth
    the few evaluations: sympy writes the second family of `COS(x) = 1/2` as
    `2*n*pi + 5*pi/3`, whose index-zero member is `5*pi/3` and whose nearest
    member is `-pi/3` - the one the original prints. A family whose members are
    no numbers to compare - `2*n*pi*#i + LN(2)` is one - is sampled at zero.
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
    """How far from the origin a family member is, ties going to the positive."""
    number = _complex(value)
    if number is None:
        return (1, 0.0, 0)
    return (0, abs(number), 0 if number.real >= 0 else 1)


def _is_everything(found: sp.Set) -> bool:
    """Whether the set is every value there is, which is an identity."""
    return found in (sp.S.Complexes, sp.S.Reals, sp.S.UniversalSet)


def _unsolved(expression: sp.Basic, context: Context) -> tuple[sp.Basic, ...]:
    """The residual relation, which is what anything unsolved is worth.

    `3^x = x^2` comes back `3^x - x^2 = 0`: no warning, no guess, and an entry
    that says exactly what was asked without pretending to have answered it.
    The operator is kept, so an inequality nobody could solve comes back an
    inequality - moving everything to the left of a `<` does not make it an `=`.

    Where the two sides will not even subtract - a matrix set equal to a
    number - there is no residual to write, and the relation as it stands says
    as much as is known about it.
    """
    func = expression.func if isinstance(expression, Relational) else sp.Eq
    try:
        residual = normal_form(_residual(expression), context.order)
    except Exception:
        return (expression,)
    return (_relation(func, residual, sp.Integer(0)),)


def _unsolvable(expression: sp.Basic, context: Context) -> tuple[sp.Basic, ...]:
    """The residual form of anything at all, for the paths that give up.

    A system falls back to the vector of its own residual equations rather
    than to "no solutions found", which would be a lie: not knowing and there
    being nothing to know are different answers.
    """
    equations = equations_of(expression)
    if equations is None:
        return _unsolved(expression, context)
    residuals = [_unsolved(equation, context)[0] for equation in equations]
    return (InertVector(*residuals),)


# -- inequalities ------------------------------------------------------------


def _inequality(
    relation: Relational,
    variable: sp.Symbol,
    context: Context,
    constants: _Constants,
) -> tuple[sp.Basic, ...]:
    """`<`, `<=`, `>`, `>=` and `/=`, solved into ranges of the variable.

    A range over one variable is asked for as a set, because a set says
    plainly what one interval and a union of two look like: one interval is
    one chained entry `-2 < x < 2`, a union is one entry per piece, and a
    half-line is a single relation. Anything else - a symbolic bound above
    all - is reduced instead, and comes back as relations already.
    """
    # The relations reach here undecided, which is how Simplify assembles one;
    # the solvers need the evaluated form they build their own answers from.
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
    """One entry per piece of a solution set, or None where it is no range."""
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
        return tuple(_equated(variable, value, context) for value in _ordered(found.args))
    if isinstance(found, sp.Interval):
        return (_span(found, variable, context, constants),)
    return None


def _span(
    interval: sp.Interval,
    variable: sp.Symbol,
    context: Context,
    constants: _Constants,
) -> sp.Basic:
    """One interval as the relation the original writes it as.

    Two finite ends make the chain `-2 < x < 2`, one end makes a single
    relation `x > 1`, and neither makes an identity - the inequality holds
    wherever the variable does.
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
    """What `reduce_inequalities` answered, as entries with the variable left."""
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
    """A relation with both sides in the normal form every answer is written in.

    `reduce_inequalities` answers `x >= 3*y/2 - 7/2`, and a sum is written as a
    rational function of its own primary variable wherever it stands, which is
    what makes that `x >= (3*y - 7)/2`.
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
) -> tuple[sp.Basic, ...]:
    """A vector of equations, solved for as many variables as there are of them.

    One entry comes of it whatever the count, and that entry is the solution
    vector: `[x + y = 3, x - y = 1]` answers `[x = 2, y = 1]`. Variables the
    user did not choose stay as themselves on the right, which is what makes
    `[x + y + z = 1, x - y = 0]` solvable for `x` and `y` alone.
    """
    joined = InertVector(*equations)
    chosen = _chosen(joined, context, variables, len(equations))
    if not chosen:
        return _unsolvable(joined, context)
    residuals = [_residual(equation) for equation in equations]
    solved = _linear(residuals, chosen, constants)
    if solved is None:
        solved = _nonlinear(residuals, chosen)
    if solved is None:
        return _unsolvable(joined, context)
    return tuple(
        InertVector(
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
    """The solution of a system linear in the chosen variables, or None.

    An empty list is a consistent answer - the system is inconsistent and has
    no solutions - and None means `linsolve` would not take it, which is what
    sends a system on to the nonlinear route.

    The variables are handed over in reverse, and that is not cosmetic.
    A singular consistent system has a free variable, `linsolve` leaves the
    *last* of the variables free, and the original parametrises the *first*:
    `[x + 3*y = 1, 2*x + 6*y = 2]` is `[x = @1, y = (1 - @1)/3]` and not the
    other way about. Reversing the list is what puts the freedom where the
    original puts it.

    Whatever is left free is replaced by a fresh `@n` rather than left as
    itself, so that an answer says which quantity is arbitrary instead of
    reading as an equation the variable still appears in.
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
    """A system `linsolve` refused, as far as `solve` will take it.

    Beyond the original, which handles only systems linear in the solution
    variables and refers the rest to a manual five-step recipe. An empty answer
    is read as "nothing found" rather than as "no solutions": `solve` says the
    same thing either way, and claiming there are none would be a guess.
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
    """Every real root inside `[low, high]`, as Approximate precision wants them.

    Better than the original where it costs nothing: Derive returns one root
    even when the interval holds several, and a polynomial's roots can all be
    had at once. Anything else is bracketed by a scan for a sign change and
    then bisected, which finds one root and says so by finding only one.
    """
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
        try:
            found = polynomial.real_roots()
        except Exception:
            found = []
        return [
            _approximate(root, context) for root in found if _within(root, low, high)
        ]
    root = _bracketed(residual, variable, low, high, context)
    return [] if root is None else [root]


def _approximate(root: sp.Basic, context: Context) -> sp.Basic:
    """A root as the approximate number it is: the simplest rational showing it.

    Which is what every other approximate number in the engine is, so a root
    found numerically prints the way an approximated constant prints -
    `-1.16730` at six digits, cut rather than rounded up, and written in
    whatever style the precision mode selected.
    """
    digits = context.precision_digits
    exact = sp.Rational(root.evalf(digits + GUARD))
    return sp.Rational(simplest(Fraction(exact.p, exact.q), digits))


def _within(root: sp.Basic, low: sp.Basic, high: sp.Basic) -> bool:
    value, first, last = _number(root), _number(low), _number(high)
    if value is None or first is None or last is None:
        return False
    return first <= value <= last


def _number(value: sp.Basic) -> float | None:
    """`value` as a real float, or None where it is not one."""
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
    """One root of `residual` in `[low, high]`, found by a scan and a bisection.

    The scan is what makes the bisection honest: bisecting an interval whose
    ends do not straddle a root finds nothing, and one that holds several roots
    has to be cut small enough for one of them to be alone in a piece.
    """
    first, last = _number(low), _number(high)
    if first is None or last is None or not first < last:
        return None
    step = (last - first) / _SCAN
    previous = None
    for index in range(_SCAN + 1):
        point = first + index * step
        value = _number(residual.subs(variable, sp.Float(point)))
        if value is None:
            previous = None
            continue
        if value == 0.0:
            return _approximate(sp.Float(point), context)
        if previous is not None and (previous < 0) != (value < 0):
            found = _bisect(residual, variable, point - step, point, context)
            if found is not None:
                return found
        previous = value
    return None


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
    """Mixed mode's own last resort: look for a real root, reaching further.

    The manual has Mixed searching `[-10, 10]` and doubling the interval until
    something turns up. The cap is ours: a search that has reached sixty-five
    thousand and found nothing is not going to, and the residual equation is a
    better answer than a computation that will not stop.
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


# -- writing an answer down --------------------------------------------------


def _equated(variable: sp.Symbol, value: sp.Basic, context: Context) -> sp.Basic:
    """`x = value`, with the value written the way every answer is written."""
    return _relation(sp.Eq, variable, normal_form(value, context.order))


def _relation(func, left: sp.Basic, right: sp.Basic) -> sp.Basic:
    """A relation assembled undecided, as the engine assembles every relation."""
    try:
        return func(left, right, evaluate=False)
    except Exception:
        return func(left, right)


def _ordered(values: Iterable[sp.Basic]) -> list[sp.Basic]:
    """The solutions in the order they are appended: reals up, then the rest.

    The original's order is its solver's accident - `x^2 - 5*x + 6` comes out
    ascending and `x^2 - 4` comes out `2, -2` - so there is nothing to be
    faithful to and something to gain from being predictable.
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
    # The conjugate with the positive part first, which is how the original
    # prints a pair: `x = #i` and then `x = -#i`.
    return (1, number.real, -number.imag, ())


def _complex(value: sp.Basic) -> complex | None:
    try:
        return complex(value.evalf())
    except Exception:
        return None


# -- reading an expression ---------------------------------------------------


def _elements(expression: sp.Basic) -> list[sp.Basic]:
    """What a vector holds, or nothing at all where this is no vector."""
    if isinstance(expression, InertVector):
        return list(expression.args)
    if isinstance(expression, sp.MatrixBase) and expression.rows == 1:
        return list(expression)
    return []


def _symbols_of(expression: sp.Basic) -> dict[str, sp.Symbol]:
    """The expression's own symbols by name, so their declarations survive.

    A variable named on a prompt is a name; the symbol the conversion built
    for it carries what its domain declaration said, and rebuilding it from
    the name alone would throw that away.
    """
    return {
        symbol.name: symbol
        for symbol in expression.free_symbols
        if type(symbol) is sp.Symbol
    }


def _named_in(expression: sp.Basic) -> list[str]:
    return list(_symbols_of(expression))
