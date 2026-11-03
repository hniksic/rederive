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
  appends for an equation it cannot solve. It is a normal one-entry answer and never an
  exception: this module promises what the rest of the engine promises, that
  nothing the parser produced makes it raise. The original appends one for
  `3^x = x^2` too, which is solved here in Lambert W.

Where the original's answers and ours differ they differ in spelling, and
deliberately so in two places. A periodic equation gets a finite sample of
representatives near the origin, one per solution family, where Derive emits
redundant ones - `SIN(x) = 1/2` is `pi/6` and `5*pi/6` here and `pi/6`,
`5*pi/6`, `-7*pi/6` there. And multiple solutions come out in a stable order,
reals ascending and then the complex ones by their angle, where the original's order is
whatever its solver happened to produce.

Domains do not filter solutions (manual 5.4): a declaration says what a
variable *is*, which is what decides how an answer simplifies, and never which
of the answers are wanted. Solutions are sought over the complexes whatever a
variable was declared, with one exception: an equation in the `ABS` of a real
variable, undeclared ones included, is solved over the reals, since over the
complexes its solutions are curves no list of relations describes.
"""

from __future__ import annotations

import cmath
from collections.abc import Callable, Iterable, Sequence

import sympy as sp
from sympy.core.relational import Relational

from rederive.engine.approximation import GUARD, simplest_rational
from rederive.engine.context import Context, Precision
from rederive.engine.normal import normal_form
from rederive.engine.ordering import main_order
from rederive.engine.printer import is_written
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
    divisors: Sequence[sp.Basic] = (),
) -> tuple[sp.Basic, ...]:
    """The entries `expression` solves to, one per worksheet line.

    `variables` names what to solve for, in the order chosen; empty means work
    it out - the one variable of a scalar, or as many of a system's most main
    as it has equations. `bounds` is the interval a numeric search is confined
    to, and is what Approximate precision asks the user for; None leaves the
    precision mode to decide, which is exact everywhere but Mixed. `divisors` are
    what the equation divides by as it was written, before Simplify cancelled
    anything: a value that provably makes one of them zero is no solution.
    """
    context = context or Context()
    constants = _Constants(context.arbitrary_index)
    try:
        return _solutions(
            expression, context, tuple(variables), bounds, tuple(divisors), constants
        )
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
    divisors: tuple[sp.Basic, ...],
    constants: _Constants,
) -> tuple[sp.Basic, ...]:
    equations = equations_of(expression)
    if equations is not None:
        return _system(equations, context, variables, constants)
    return _scalar(expression, context, variables, bounds, divisors, constants)


def _scalar(
    expression: sp.Basic,
    context: Context,
    variables: tuple[str, ...],
    bounds: tuple[sp.Basic, sp.Basic] | None,
    divisors: tuple[sp.Basic, ...],
    constants: _Constants,
) -> tuple[sp.Basic, ...]:
    """One equation or inequality, solved for one variable."""
    chosen = _chosen(expression, context, variables, 1)
    if not chosen:
        return _unsolvable(expression, context)
    variable = chosen[0]
    if isinstance(expression, Relational) and not isinstance(expression, sp.Equality):
        return _inequality(expression, variable, context, constants)
    return _equation(
        _residual(expression), variable, context, bounds, divisors, constants
    )


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
    divisors: tuple[sp.Basic, ...],
    constants: _Constants,
) -> tuple[sp.Basic, ...]:
    """`residual = 0` solved for `variable`, by whatever the precision allows.

    Bounds mean a numeric search and nothing else, which is Approximate mode.
    Exact and Mixed both solve exactly first; where that answers nothing, Mixed
    goes looking for a real root over a widening interval and Exact stops.

    An exact candidate is kept unless it is provably no solution: one that makes a
    divisor of the equation as written zero is not (`_divides_by_zero`), and nor is
    one a solver found for an equation it rewrote first, where putting it back
    shows it does not hold (`_fails`).
    """
    if bounds is not None:
        return _numeric(residual, variable, bounds[0], bounds[1], context)
    if residual.is_zero or _cancels(residual, variable):
        return (_equated(variable, constants.fresh(), context),)
    found = _values(residual, variable)
    if found is not None:
        polynomial = _polynomial(residual, variable) is not None

        def admissible(value: sp.Basic) -> bool:
            if _divides_by_zero(divisors, variable, value):
                return False
            return polynomial or not _fails(residual, variable, value)

        entries = _entries(found, variable, context, constants, admissible)
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

    `trig` asks for Viete's cubic rather than Cardano's, which differs only for
    the casus irreducibilis: three real roots that the radicals can only reach
    through `#i`. `SOLVE(x^3 - 3*x + 1 = 0, x)` is then the original's own
    `2*COS(2*pi/9)` and its two companions, and not three nested radicals over
    an imaginary unit that cancels.
    """
    polynomial = _polynomial(residual, variable)
    if polynomial is not None:
        if polynomial.degree() < 1:
            return sp.S.EmptySet if not polynomial.is_zero else sp.S.Complexes
        found = _binomial_roots(polynomial) or sp.roots(polynomial, trig=True)
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

    For a real variable an `ABS` is its argument or the negation of it, depending on
    the argument's sign, so the equation is finitely many equations, each over the
    part of the line where its signs hold, and what each solves to is kept where it
    lies inside its own part. `solveset` over the reals does that splitting. Over
    the complexes `ABS(x) = 3` is a whole circle, which no list of relations
    describes, so a variable not known to be real is not solved this way. The
    answer can be a range: `ABS(x) = x` holds for `x >= 0`.
    """
    try:
        found = sp.solveset(residual, variable, sp.S.Reals)
    except Exception:
        return None
    return None if found.has(sp.ConditionSet) else found


def _exponential(residual: sp.Basic, variable: sp.Symbol) -> sp.Set | None:
    """The solutions of `residual` with every power of a positive number in the
    variable written as a power of `#e`, or None where that solves nothing more.

    `solveset` inverts `#e^x = c` over the complexes into its whole family of
    logarithms, and leaves `3^x = c` as a condition, so `3^(2*x) - 4*3^x + 3 = 0`
    stood where `#e^(2*x) - 3*#e^x + 2 = 0` was solved. `3^x` is `#e^(x*LN(3))`,
    and solved in that spelling it has the same families, `x = 1` and `x = 0` being
    their real members.
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
    """The roots of `a*x^n + b` for a real `-b/a`, as one real root times each root
    of unity, or None for any other polynomial.

    `roots` multiplies the real root into every term of the root of unity, so
    `x^100 = 12` is answered with the hundredth root of 12 written three times
    over in every solution. Written once, a solution reads as the point on a
    circle it is: `12^(1/100)*(COS(pi/50) + #i*SIN(pi/50))`.
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


def _inverted(residual: sp.Basic, variable: sp.Symbol) -> sp.Set | None:
    """What `solve` finds where `solveset` gives up, or None.

    `solveset` has no Lambert W inversion and `solve` has, so `x*#e^x = 1` is
    `LAMBERTW(1)` from the one and a condition set from the other. `solve` keeps to the
    variable's assumptions, which for an authored variable means real, and it then
    answers on both real branches where there are two: `x = LN(x) + 2` has both its
    roots. That makes an empty answer "no real solution" rather than "no solution", so
    it is not taken for one; nor is a solution with a head the notation has no word for,
    or one that does not check out when put back into the equation.
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


def _satisfies(residual: sp.Basic, variable: sp.Symbol, value: sp.Basic) -> bool:
    """Whether `value` makes `residual` vanish, exactly or else to working precision.

    The numeric check is for values like `LAMBERTW(-#e^(-2), -1)`, where `simplify`
    cannot see that the logarithm of the branch undoes it.
    """
    try:
        left = residual.subs(variable, value)
        if sp.simplify(left).is_zero:
            return True
        if left.free_symbols:
            return False
        return abs(complex(left.evalf(30))) < 1e-20
    except Exception:
        return False


def _fails(residual: sp.Basic, variable: sp.Symbol, value: sp.Basic) -> bool:
    """Whether putting `value` back into `residual` provably leaves it nonzero.

    A solver that combines logarithms solves `LN(x) + LN(x - 1) = LN(6)` as
    `x*(x - 1) = 6`, and so offers -2 as well as 3. But `LN(-2) + LN(-3)` is
    `LN(6) + 2*pi*#i`, the principal logarithms of the two negative numbers each
    carrying `pi*#i`, and -2 is no solution. Each candidate is therefore put back
    and evaluated, the principal branch being the one every function is evaluated
    on. Only a definite answer drops one: a value with a parameter left in it is
    kept, and so is one whose evaluation fails or is not a finite number. Nonzero
    means nonzero beyond any rounding, measured against the size of the terms.
    """
    try:
        left = residual.xreplace({variable: value})
        if left.free_symbols:
            return False
        found = complex(left.evalf(30))
        sizes = [
            abs(complex(term.xreplace({variable: value}).evalf(15)))
            for term in sp.Add.make_args(residual)
        ]
        scale = max([1.0, *sizes])
    except Exception:
        return False
    if not (cmath.isfinite(found) and cmath.isfinite(scale)):
        return False
    return abs(found) > 1e-12 * scale


def _divides_by_zero(
    divisors: tuple[sp.Basic, ...], variable: sp.Symbol, value: sp.Basic
) -> bool:
    """Whether `value` provably makes one of the equation's divisors zero.

    The divisors are those of the equation as written, and the symbol in them is
    found by name, which is how a variable is known on either side of Simplify.
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
    """Whether `residual` is a rational function of `variable` that is zero once its
    terms are brought over one denominator, which makes the equation an identity.

    `1/(x - 1) + 1 = x/(x - 1)` holds wherever both sides are defined, and solveset
    answers it with the empty set; like `(x^2 - 1)/(x - 1) = x + 1`, it solves to an
    arbitrary value.
    """
    try:
        return bool(residual.is_rational_function(variable)) and sp.cancel(residual) == 0
    except Exception:
        return False


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
    admissible: Callable[[sp.Basic], bool] = lambda value: True,
) -> tuple[sp.Basic, ...] | None:
    """The relations one solution set is written as, in a stable order.

    The values `admissible` turns down are left out. A set that is a range, which
    an equation in `ABS` can solve to, is written as the inequalities saying so.
    None where the set is one no finite list of relations describes, which is
    the residual equation's case.
    """
    if _is_everything(found):
        return (_equated(variable, constants.fresh(), context),)
    values = _members(found)
    if values is None:
        return _ranges(found, variable, context, constants)
    return _answers(variable, [value for value in values if admissible(value)], context)


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
    if isinstance(found, sp.Complement):
        return _remaining(*found.args)
    return None


def _remaining(inside: sp.Set, excluded: sp.Set) -> list[sp.Basic] | None:
    """The members of `inside` that are not provably members of `excluded`.

    `SOLVE(1 - 2*m/r = 0, r)` is what this is here for: sympy answers
    `Complement({2*m}, {0})`, because dividing by `r` is only allowed where `r`
    is not zero, and `2*m` is zero for one value of `m`. Dropping a candidate
    needs proof that it is excluded, and there is none here - `m` is an
    undeclared parameter, and an undecidable case is not an exclusion - so
    `2*m` stands and the answer is the original's own `r = 2*m`. Nothing is
    papered over by that: with `m` literally zero the residual is the constant
    1, sympy answers with the empty set, and no solution is offered at all.

    Only a finite `inside` is read this way. The complement of a solution
    family is a set no finite list of relations describes - removing one member
    of `2*pi*n` leaves the rest, and there is no notation for the rest - so
    None is the answer there, and the residual equation is what gets appended.
    """
    if not isinstance(inside, sp.FiniteSet) or not isinstance(excluded, sp.FiniteSet):
        return None
    return [value for value in inside.args if not _provably_holds(excluded, value)]


def _provably_holds(excluded: sp.Set, value: sp.Basic) -> bool:
    """Whether `excluded` provably holds `value`, undecided counting as no.

    Sympy's own membership test answers with a boolean where it knows and with
    an undecided `Contains` or equation where it does not, and only the boolean
    true is proof.
    """
    try:
        return excluded.contains(value) is sp.true
    except Exception:
        return False


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

    `COS(x) = x` comes back `COS(x) - x = 0`: no warning, no guess, and an entry
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
    return simplest_rational(root.evalf(digits + GUARD), digits)


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


def _answers(
    variable: sp.Symbol, values: Iterable[sp.Basic], context: Context
) -> tuple[sp.Basic, ...]:
    """`x = value` for each distinct value, in the order `_ordered` gives.

    Values that differ as written can be the same number once written down:
    `solveset` answers `x^(1/3)/3 + 3/x^(1/3) = 1` with the cubes of a conjugate
    pair, and both are -27. So repeats are dropped again after the writing.
    """
    written: list[sp.Basic] = []
    for value in _ordered(values):
        entry = _equated(variable, _tidied(value), context)
        if entry not in written:
            written.append(entry)
    return tuple(written)


def _tidied(value: sp.Basic) -> sp.Basic:
    """`value` in the plainer of two spellings a solver can leave it in.

    A square root of a product of sums is the root of their expanded product
    where that is no longer: `roots` writes `SQRT(-(x + r)*(x - r))` for what is
    `SQRT(r^2 - x^2)`. And a negated quotient over a negative number is the
    quotient of the two negations, `LN(3)/(LN(3) - LN(2))` rather than
    `-LN(3)/(LN(2) - LN(3))`. Each is the same number either way.
    """
    try:
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
    # Round the circle from the positive real axis, the conjugate with the
    # positive part first, which is how the original prints a pair: `x = #i`
    # and then `x = -#i`. The angle is rounded so that a pair whose angles
    # differ in the last bit of their evaluation still sorts as a pair.
    angle = round(abs(cmath.phase(number)), 9)
    return (1, angle, 0.0 if number.imag > 0 else 1.0, (abs(number),))


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
