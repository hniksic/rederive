"""Simplify: the first command built on the engine.

Derive's Simplify aims for a *sufficiently simple* form (no superfluous variables,
roots, functions or reducible degrees) in a normal form, so that two ways of writing the
same thing come back the same. It changes as little as possible, so this is a sequence
of named rewrites rather than a call to `sympy.simplify`, which would pick its own idea
of a nicer expression.

The steps:

1. Evaluate the tree innermost first (`evaluate`): substitute names, apply defined
   functions, and decide each `IF` test before choosing a branch. Sympy's automatic
   evaluation does most of Simplify's basic work along the way.
2. During evaluation, `_evaluate` works out calculus heads, and `_commanded` runs
   `FACTOR`, `EXPAND`, `SOLVE` and `RATE`. A calculus head that won't evaluate is kept,
   so `INT(#e^(x^2), x)` comes back as an integral rather than an error.
3. Reduce the kernels: try each identity and keep it only if it pays (fewer operations
   or one variable fewer), and apply the rules a Collect or Expand mode requires. The
   tables at the end of this module list them all.
4. Put every sum and quotient in normal form (`normal_form`): a sum as a rational
   function of its main generator, a product cancelled but not distributed. This
   always applies; it is why `(x + 1)^9 + y` expands to a polynomial while
   `(y + 1)^9 + x` does not, and why `(x^2 + 2*x*y + y^2)/(x^2 - y^2)` becomes
   `(x + y)/(x - y)`.
5. Follow Derive's spelling conventions: `1/COS(x)^2` as `TAN(x)^2 + 1`, `LN(256)` as
   `8*LN(2)`.
6. Approximate, if the precision mode asks for it.

Every rewrite runs inside a `try`, and one that raises is discarded, keeping the
previous form. Nothing sympy does can turn a valid entry into an error.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import reduce
from itertools import combinations
from math import gcd
from typing import NamedTuple

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.logic.boolalg import Boolean
from sympy.simplify.fu import TR2, TR5, TR6, TR7, TR8, TR11

from rederive.engine.approximation import approximated, noting_approximations
from rederive.engine.boundary import DEFAULT_AMOUNT, Amount, Result
from rederive.engine.context import (
    Branch,
    Context,
    Direction,
    Precision,
    TrigPower,
)
from rederive.engine.evaluate import (
    Commanded,
    Environment,
    evaluate,
    named_as_declared,
    released,
)
from rederive.engine.expanding import EXPAND_AMOUNTS, expanded_expression
from rederive.engine.factoring import amount_named, factored_expression
from rederive.engine.from_sympy import from_sympy
from rederive.engine.integration import agrees, integral, proved_zero
from rederive.engine.intervals import covering, decided, never_negative, settled
from rederive.engine.normal import normal_form
from rederive.engine.ordering import main_order
from rederive.engine.printer import is_written, shadowing_built_ins
from rederive.engine.rewriting import Rewrite, attempt, forced, pays
from rederive.engine.signs import folded_signs
from rederive.engine.solving import oriented, solutions
from rederive.engine.to_sympy import (
    CALCULUS_HEADS,
    Approx,
    Assign,
    annuity,
    Declare,
    FunDef,
    InertVector,
    Logical,
    PlusMinus,
    as_condition,
    authored_conditionals,
    digit_count,
)
from rederive.engine.vocabulary import INVERSE_HYPERBOLIC, in_vocabulary
from rederive.model.expr import Kind, Node
from rederive.syntax.state import ParseState

__all__ = ["approx", "approximated", "simplified", "simplify"]


#: `combsimp` is tried only where one of these appears, because on an ordinary
#: polynomial it factors (`x^2 + 2*x` becomes `x*(x + 2)`), which is Factor's job.
_COMBINATORIAL = (sp.factorial, sp.factorial2, sp.binomial, sp.gamma, sp.ff, sp.rf)

#: Maximum expression size for `combsimp`, which compares every pair of factorials.
#: On Derive's `F_0M` (102 operations, half of them symbolic factorials) it takes 23
#: seconds; at this size and below, no corpus expression takes over 0.2 seconds.
_COMBSIMP = 60

#: Maximum number of variables for boolean simplification. Both normal forms come from
#: a truth table, so work and result size double with each variable: six chained
#: `XOR`s give 129 operations, longer than the input. Sympy's own limit of eight is
#: already too high to be useful.
_BOOLEAN_VARIABLES = 5


@noting_approximations
@shadowing_built_ins
def simplify(
    node: Node, context: Context | None = None, state: ParseState | None = None
) -> Result:
    """Derive's Simplify: `node` in its sufficiently simple form.

    Works on any subtree, so the session can simplify a highlighted subexpression and
    splice the answer back. `state` is the symbol table used to reparse the answer; a
    session in a non-default input or case mode must pass its own.

    The author's conditionals are collected up front and passed to the printer, so an
    undecidable one is shown as written (see `authored_conditionals`).
    """
    context = context or Context()
    node = named_as_declared(node, state)
    whole = _whole_approximation(node, context)
    if whole is not None:
        value, digits = whole
        return approx(value, context, digits, state)
    return from_sympy(
        simplified(node, context),
        context,
        state,
        authored_conditionals(node, context),
    )


def approx(
    node: Node,
    context: Context | None = None,
    digits: int | None = None,
    state: ParseState | None = None,
) -> Result:
    """Simplify with the precision mode set to Approximate.

    This is the same pipeline as `simplify` at a different precision, as the manual
    describes approX. It is also what an `APPROX` call simplifies to.
    """
    context = context or Context()
    approximate = context.with_precision(Precision.APPROXIMATE, digits)
    return simplify(node, approximate, state)


def _whole_approximation(
    node: Node, context: Context
) -> tuple[Node, int | None] | None:
    """The operand and digit count of an expression that is an `APPROX` call.

    Such an expression simplifies to approX of the operand at the given digit count:
    `APPROX(1/2)` gives 0.5 and `APPROX(1/3, 3)` gives 0.333. Nested deeper, the call is
    just the rational it rounds to, so `APPROX(1/2) + 1/3` is 5/6.

    Entry references like `#1` are followed. Returns None if the expression isn't such a
    call or the digit count is unusable, leaving it to ordinary Simplify.
    """
    while node.kind is Kind.LABEL and str(node.value).isdigit():
        labelled = context.labels.get(int(node.value))
        if labelled is None:
            return None
        node = labelled
    if node.kind is not Kind.CALL:
        return None
    name, *arguments = node.children
    if name.value != "APPROX" or len(arguments) not in (1, 2):
        return None
    if len(arguments) == 1:
        return arguments[0], None
    count = digit_count(simplified(arguments[1], context))
    return None if count is None else (arguments[0], count)


def simplified(
    node: Node, context: Context | None = None, decide: bool = True
) -> sp.Basic:
    """The simplified expression as sympy, before conversion back to a node.

    `simplify` is this plus `from_sympy`; commands that keep computing can use this to
    avoid printing and reparsing.

    `decide` makes a relation that holds come back as `true`. soLve turns it off: it
    only prepares the equation, and `x = x` decided to `true` would leave nothing to
    solve, where the answer should be `x = @1`.

    An undecidable conditional stays suspended through all steps, keeping its branches
    as written, and is released as a conditional only at the end.
    """
    context = context or Context()
    environment = Environment(
        context,
        decide=lambda test: _decide_simplified(test, context),
        work_out=lambda head: _evaluate(head, context),
        command=lambda name, values, divisors: _commanded(
            name, values, divisors, context
        ),
    )
    expression = evaluate(node, environment)
    try:
        result = _transform(expression, context, decide)
    except Exception:
        # The evaluated input is still a correct answer.
        result = expression
    return released(result)


# -- by shape ----------------------------------------------------------------


def _transform(expression: sp.Basic, context: Context, decide: bool) -> sp.Basic:
    """Simplify one expression according to what kind of thing it is."""
    if isinstance(expression, Relational):
        return _relation(expression, context, decide)
    if isinstance(expression, Declare):
        return expression
    if isinstance(expression, (Assign, FunDef)):
        return _definition(expression, context, decide)
    if isinstance(expression, sp.MatrixBase):
        return expression.applyfunc(
            lambda element: _transform(element, context, decide)
        )
    if isinstance(expression, (InertVector, Logical)):
        return expression.func(
            *(_transform(a, context, decide) for a in expression.args)
        )
    if isinstance(expression, Boolean):
        return _boolean(expression, context, decide)
    return _expression(expression, context, decide)


def _relation(expression: Relational, context: Context, decide: bool) -> sp.Basic:
    """Simplify both sides, and decide the relation if possible.

    As in the original, `2 = 2` gives `true` and `1 = 2` gives `false`. The decision is
    three-valued, so a relation that settles nothing (`x = 2`, or `ABS(x) < 1` with no
    declaration) is returned as authored.

    An ordering is decided only for real sides, since `x < x + 1` means nothing for
    complex `x`. It is decided like an `IF` test, using declared domains. An equation
    is decided by arithmetic only (`_equated`).

    With `decide` off (soLve's pipeline), the sides are simplified and the relation left
    for solving.

    Two matrices of the same shape are related element by element, as section 8.5 uses
    to read off a solved system: `[[x], [y], [z]] = A^-1 . b` gives three equations. If
    the elements decide a matrix (in)equation, the answer is one truth value; see
    `_whole`.
    """
    left = _transform(expression.lhs, context, decide)
    right = _transform(expression.rhs, context, decide)
    spread = _spread(expression.func, left, right, context, decide)
    if spread is not None:
        if decide and expression.func in (sp.Eq, sp.Ne):
            return _whole(expression.func, spread, left, right)
        return spread
    return _stated(expression.func, left, right, context, decide)


def _whole(func, spread: sp.Basic, left: sp.Basic, right: sp.Basic) -> sp.Basic:
    """Decide a matrix equation or inequation from its element-wise relations.

    Matrices are equal when all element pairs are and different when any pair is, so
    `[1, 2] = [1, 3]` is `false` and `[1, 2] /= [1, 3]` is `true`. An undecided
    equation becomes the system of element equations; an undecided inequation stays as
    written, since no single element relation expresses it.
    """
    verdicts = _elements_of_spread(spread)
    equal = func is sp.Eq
    if any(verdict == (sp.false if equal else sp.true) for verdict in verdicts):
        return sp.false if equal else sp.true
    if all(verdict == (sp.true if equal else sp.false) for verdict in verdicts):
        return sp.true if equal else sp.false
    return spread if equal else sp.Ne(left, right, evaluate=False)


def _elements_of_spread(spread: sp.Basic) -> list[sp.Basic]:
    if isinstance(spread, InertVector):
        return [leaf for row in spread.args for leaf in _elements_of_spread(row)]
    return [spread]


def _spread(
    func, left: sp.Basic, right: sp.Basic, context: Context, decide: bool
) -> sp.Basic | None:
    """A relation between two matrices, applied element by element.

    Returns None unless both sides are matrices of the same shape. The result keeps
    that shape, with a single row written flat like any one-row matrix.
    """
    if not isinstance(left, sp.MatrixBase) or not isinstance(right, sp.MatrixBase):
        return None
    if left.shape != right.shape or not left.rows or not left.cols:
        return None
    rows = [
        InertVector(
            *(
                _stated(func, left[row, col], right[row, col], context, decide)
                for col in range(left.cols)
            )
        )
        for row in range(left.rows)
    ]
    return rows[0] if left.rows == 1 else InertVector(*rows)


def _stated(
    func, left: sp.Basic, right: sp.Basic, context: Context, decide: bool
) -> sp.Basic:
    """Build a relation of two simplified sides, decided where possible."""
    try:
        relation = func(left, right, evaluate=False)
    except Exception:
        relation = func(left, right)
    if not (decide and isinstance(relation, Relational) and _comparable(relation)):
        return relation
    held = _answered(relation, context)
    if held is None:
        return relation
    return sp.true if held else sp.false


def _answered(relation: Relational, context: Context) -> bool | None:
    """Whether a relation is definitely true or false, or None if undecided.

    Equations are decided by arithmetic alone (`_equated`). Orderings are decided like
    an `IF` test, including declared domains, since domains are about where on the line
    a value lies.
    """
    if isinstance(relation, (sp.Equality, sp.Unequality)):
        return _equated(relation, context)
    return _decide(relation, context)


def _equated(relation: Relational, context: Context) -> bool | None:
    """Whether the two sides of an equation are equal, by arithmetic alone.

    Like the original, this ignores declared domains. `x = #i` (from a quadratic with a
    negative discriminant) stands whatever `x` is declared as, and so does
    `ABS(x) + 1 = 0`, which no real satisfies.

    The equation is decided when the difference of the sides is a number: zero means
    equal, a nonzero real means not. A non-real difference decides nothing, as in the
    original: `#i = 1` and `2*#i = #i` both stand.

    Matrices of mismatched shapes have no difference, so the written values are
    compared: `[1, 2] = [1, 2, 3]` is false, while `[x, y] = [1, 2, 3]` stands.
    """
    if any(isinstance(side, sp.MatrixBase) for side in relation.args):
        if relation.free_symbols:
            return None
        return _decide(relation, context)
    same = _one_value(relation, context)
    if same is None:
        return None
    return same is isinstance(relation, sp.Equality)


def _one_value(relation: Relational, context: Context) -> bool | None:
    """Whether the two sides of an equation are the same value.

    Identical sides are equal even without a difference to take (`inf = inf`,
    `true = true`). Otherwise the difference is simplified with the pipeline's rewrites,
    so that `SIN(x)^2 + COS(x)^2 = 1` is decided too.
    """
    if relation.lhs == relation.rhs:
        return True
    difference = attempt(relation, lambda r: _rewritten(r.lhs - r.rhs, context))
    if difference is None or not difference.is_number:
        return None
    if difference.is_zero:
        return True
    if difference.is_zero is False and difference.is_extended_real is True:
        return False
    return None


def _comparable(relation: Relational) -> bool:
    """Whether this relation can be decided at all.

    A side containing `?` is unknown, so nothing can be decided about it, even though
    sympy would find two `?` to be the same object.

    Orderings need both sides known to be real, since only the reals are ordered.
    Equality has no such condition.
    """
    if relation.has(sp.nan):
        return False
    if isinstance(relation, (sp.Equality, sp.Unequality)):
        return True
    return all(side.is_extended_real is True for side in relation.args)


def _definition(expression: sp.Basic, context: Context, decide: bool) -> sp.Basic:
    """Simplify the value of an assignment or function definition.

    The first two arguments, the name and the defining operator, are left untouched so
    the line still reads as a definition.
    """
    head, operator, *value = expression.args
    return expression.func(
        head, operator, *(_transform(part, context, decide) for part in value)
    )


def _boolean(expression: Boolean, context: Context, decide: bool) -> sp.Basic:
    """Simplify relations joined by boolean operators.

    `6 >= -2*x AND 3*x /= -9` involves one variable, so its simplest form is the range
    it describes. Otherwise (several variables, or non-relation operands) boolean
    algebra runs: `p XOR q` is rewritten in the other operators, `p OR NOT p` becomes
    `true`, and a conjunct common to every disjunct is factored out. The result keeps
    its shape, with its operands simplified.
    """
    solved = _reduced(expression)
    if solved is not None:
        return solved
    expression = _algebra(expression)
    if not isinstance(expression, Boolean) or not expression.args:
        # An atom (`p OR NOT p` -> `true`, `p AND (p OR q)` -> `p`) has nothing left
        # to simplify, and passing it to `_transform` would just loop back here.
        return expression
    try:
        return expression.func(
            *(_transform(a, context, decide) for a in expression.args)
        )
    except Exception:
        return expression


def _algebra(expression: Boolean) -> sp.Basic:
    """`expression` in the shorter of its two boolean normal forms.

    Both the sum of products and the product of sums are computed and the shorter wins,
    as the manual prefers; this leaves `p AND (q OR r)` alone instead of multiplying it
    out. Along the way `NOT` is pushed inward, `XOR` and `IMP` are rewritten, and a
    conjunct common to every disjunct is factored out. Ties go to the earlier form, so
    `p XOR q` comes back as a sum of products.

    Relations are treated as opaque variables; their insides are simplified separately.

    Three forms are requested, not two, because sympy returns an expression already in
    the named form unchanged (`p OR NOT p` as DNF comes back as is). The unnamed request
    always does the algebra, and it comes first so it also wins ties.
    """
    if _predicates(expression) > _BOOLEAN_VARIABLES:
        return expression
    forms = []
    for form in (None, "dnf", "cnf"):
        try:
            forms.append(sp.logic.boolalg.simplify_logic(expression, form, deep=False))
        except Exception:
            pass
    if not forms:
        return expression
    return min(forms, key=sp.count_ops)


def _predicates(expression: Boolean) -> int:
    """The number of independent truth values the expression depends on.

    Each relation counts as one, however many variables it names, so `x > 1 AND x < 2`
    counts as two.
    """
    relations = expression.atoms(Relational)
    inside = {symbol for relation in relations for symbol in relation.free_symbols}
    return len(relations) + len(expression.free_symbols - inside)


def _reduced(expression: Boolean) -> sp.Basic | None:
    """The range a conjunction of relations in one variable describes."""
    if not isinstance(expression, sp.And):
        return None
    parts = expression.args
    if not all(isinstance(part, Relational) for part in parts):
        return None
    variables = expression.free_symbols
    if len(variables) != 1:
        return None
    # The relations were built unevaluated; `reduce_inequalities` needs them evaluated.
    try:
        evaluated = [part.func(part.lhs, part.rhs) for part in parts]
        solved = sp.reduce_inequalities(evaluated, *variables)
    except Exception:
        return None
    if not isinstance(solved, sp.Basic):
        return None
    return oriented(solved)


# -- the pipeline proper -----------------------------------------------------


def _expression(expression: sp.Basic, context: Context, decide: bool) -> sp.Basic:
    """Run an ordinary expression through the whole sequence.

    The answer of a `FACTOR`, `EXPAND` or `SOLVE` head is the form the author asked for,
    and the steps here would undo it: gated rewrites prefer shorter forms (a
    factorization is often longer), and the normal form would multiply a prime
    decomposition back out. So each such answer is replaced by a placeholder symbol
    during the steps and restored before approximation.

    Inside a sum this applies only to numeric answers, since a sum is put in normal
    form regardless: `FACTOR(x^2 - 1) + x^3` is `x^3 + x^2 - 1`, while
    `FACTOR(12) + 1` keeps its `2^2*3`.
    """
    expression = _summed(expression)
    held = {found: sp.Dummy("answer") for found in _commands(expression)}
    if held:
        expression = expression.xreplace(held)
    expression = _numeric(expression)
    expression = forced(expression, _spliced_cases)
    # Roots of perfect powers go before the intervals, which then settle the absolute
    # value left behind: `SQRT(x^2 + 2*x + 1)` is `-x - 1` below -1.
    expression = forced(expression, _perfect_powers)
    # Before the rewrites, so they see what the intervals settle: `ABS(x) + ABS(x - 1)`
    # over `(0, 1)` cancels only once both bars are removed.
    expression = settled(expression, context)
    expression = _divergent(expression, context)
    expression = _rewritten_beside_standing(expression, context)
    expression = _factorials(expression)
    # The normal form is not gated on getting shorter.
    #
    # `over_numbers` is off so numeric denominators stay with their terms instead of
    # becoming a common denominator: `x^2/6 + x/2` comes back as authored.
    expression = normal_form(expression, context.order, over_numbers=False)
    expression = _spelled(expression, context)
    if held:
        expression = expression.xreplace(
            {dummy: found.args[0] for found, dummy in held.items()}
        )
    return approximated(expression, context)


def _spliced_cases(expression: sp.Basic) -> sp.Basic:
    """Merge each conditional whose else case is a conditional into one.

    A conditional closed form has the head itself as its else case, and evaluating that
    head again yields the same condition: the sum of `y*x^k` is `y` times a conditional
    on `ABS(x) < 1` whose else case is the sum of `x^k`, which is again conditional on
    `ABS(x) < 1`. The inner case is unreachable, so only its else case is kept.
    """

    def spliced(cases: sp.Piecewise) -> sp.Basic:
        last, condition = cases.args[-1]
        if condition is not sp.true or not isinstance(last, sp.Piecewise):
            return cases
        return sp.Piecewise(*cases.args[:-1], *last.args)

    if not expression.has(sp.Piecewise):
        return expression
    return expression.replace(lambda e: isinstance(e, sp.Piecewise), spliced)


def _divergent(expression: sp.Basic, context: Context) -> sp.Basic:
    """Replace each unevaluated divergent series over a declared variable with its
    infinity.

    A series whose terms tend to a nonzero limit diverges to `inf` or `-inf` by the
    limit's sign. Sympy finds the limit of `h^k` only when it knows which side of one
    `h` lies, so the limit is taken over the declared range of `h`, expressed through a
    positive variable (see `covering`). `SUM(h^k, k, 0, inf)` is thus `inf` for `h`
    declared above one, the else case of the closed form's conditional.
    """

    def diverged(series: sp.Sum) -> sp.Basic:
        if len(series.limits) != 1:
            return series
        term, (index, low, high) = series.function, series.limits[0]
        if high is not sp.oo or not low.is_finite:
            return series
        parameters = term.free_symbols - {index}
        if len(parameters) != 1:
            return series
        (parameter,) = parameters
        values = covering(parameter, context)
        if values is None:
            return series
        ends = set()
        for value in values:
            limit = attempt(
                term, lambda t: sp.limit(t.subs(parameter, value), index, sp.oo)
            )
            if limit is None:
                return series
            ends.add(_infinity_of(limit))
        if len(ends) != 1 or None in ends:
            return series
        return ends.pop()

    if not expression.has(sp.Sum):
        return expression
    return forced(
        expression, lambda e: e.replace(lambda h: isinstance(h, sp.Sum), diverged)
    )


def _infinity_of(limit: sp.Basic) -> sp.Basic | None:
    """The infinity a series diverges to if its terms tend to `limit`, or None if that
    doesn't show divergence."""
    if limit is sp.oo or (limit.is_finite and limit.is_positive):
        return sp.oo
    if limit is sp.S.NegativeInfinity or (limit.is_finite and limit.is_negative):
        return sp.S.NegativeInfinity
    return None


def _summed(expression: sp.Basic) -> sp.Basic:
    """Unwrap non-numeric command answers inside sums, so they are treated as values."""

    def valued(total: sp.Basic) -> sp.Basic:
        return total.xreplace(
            {
                found: found.args[0]
                for found in _commands(total)
                if not found.args[0].is_number
            }
        )

    try:
        return expression.replace(
            lambda found: isinstance(found, sp.Add) and found.has(Commanded), valued
        )
    except Exception:
        return expression


def _commands(expression: sp.Basic) -> set[sp.Basic]:
    try:
        return expression.atoms(Commanded)
    except Exception:
        return set()


# -- IF, and asking a test whether it holds -----------------------------------


def _decide(test: sp.Basic, context: Context) -> bool | None:
    """Whether `test` holds, or None if it can't be settled.

    Used for every `IF` test before choosing a branch. A relation arrives unevaluated,
    so it is tried evaluated (where declared domains apply), then as the difference of
    its sides under the pipeline's rewrites. Failing those, the declared intervals are
    consulted, since symbols can't carry them into evaluation.

    A test that isn't a relation is read as a comparison with zero, as in conversion.
    """
    test = as_condition(test)
    for candidate in _truths(test, context):
        if candidate is sp.true:
            return True
        if candidate is sp.false:
            return False
    return decided(test, context)


def _decide_simplified(test: sp.Basic, context: Context) -> bool | None:
    """Whether an `IF` test holds, using full simplification if necessary.

    The quicker `_decide` is tried first. If it is undecided, the test is simplified as
    if it stood alone, which decides e.g. `x < 1 AND x > 3` (an empty range) and
    `(x^2 + x^2*y^2)/(x^2*y) = (1 + y^2)/y`. ODE1.MTH's `HOMOGENEOUS` needs this to see
    that a derivative is zero.
    """
    verdict = _decide(test, context)
    if verdict is not None:
        return verdict
    try:
        simplified = _transform(as_condition(test), context, True)
    except Exception:
        return None
    if simplified is sp.true:
        return True
    if simplified is sp.false:
        return False
    return None


def _truths(test: sp.Basic, context: Context) -> list[sp.Basic]:
    """Forms of `test` worth checking, easiest first."""
    if not isinstance(test, Relational):
        return [test]
    readings = [test, attempt(test, lambda t: t.func(t.lhs, t.rhs))]
    difference = attempt(test, lambda t: _rewritten(t.lhs - t.rhs, context))
    if difference is not None:
        readings.append(attempt(test, lambda t: t.func(difference, 0)))
    return [reading for reading in readings if reading is not None]


# -- the calculus heads ------------------------------------------------------


def _evaluate(head: sp.Basic, context: Context) -> sp.Basic:
    """Evaluate one calculus head, or return it unchanged if it won't evaluate.

    Called on each head as it is built, with operands already evaluated. This lets an
    `IF` test contain one: ODE1.MTH uses `DIF(u, x)` to ask whether `u` depends on `x`,
    and needs the derivative, not a `Derivative`.

    A head that doesn't evaluate is retried with its operands spelled as an answer would
    spell them. A `COSH` never appears in an answer, and Derive integrates over the
    exponentials it stands for: BESSEL.MTH's
    `INT(#e^(-z*COSH(t))*COSH(n*t), t, 0, inf)` becomes two integrals of exponentials,
    as Derive prints. Left as `COSH`, the integral would stand and later rewrites would
    change its integrand.
    """
    value = _answer_in_words(head, context)
    if value is not None:
        return value
    spelled = attempt(head, _by_definition)
    if spelled is None or spelled == head:
        return head
    again = _answer_in_words(spelled, context)
    return head if again is None else again


def _answer_in_words(head: sp.Basic, context: Context) -> sp.Basic | None:
    """The value of `head` in Derive's own functions, or None if it doesn't evaluate.

    An answer that needs a function Derive has no name for is rejected:
    `INT(1/SQRT(1 - x^4), x)` is hypergeometric to sympy, and Derive can't do it, so the
    integral stands, just as if sympy had declined. Sympy functions expressible in
    Derive's are first rewritten by `in_vocabulary`.
    """
    value = _worked_out(head, context)
    if value == head:
        return None
    value = forced(value, in_vocabulary)
    return value if is_written(value) else None


def _worked_out(head: sp.Basic, context: Context) -> sp.Basic:
    """Evaluate one head by the rule for its kind."""
    if _unfinished(head, context):
        return head
    if isinstance(head, Approx):
        return _approximation(head, context)
    if isinstance(head, sp.Integral):
        return integral(head)
    if isinstance(head, (sp.Sum, sp.Product)):
        closed = _closed_by_rule(head, context)
        if closed is not None:
            return closed
    if isinstance(head, (sp.Sum, sp.Product)) and not _searchable(head):
        return head
    if isinstance(head, sp.Product):
        return _product(head)
    if isinstance(head, sp.Sum):
        return _summation(head)
    if isinstance(head, sp.Limit):
        return _limit(head)
    try:
        value = head.doit(deep=False)
    except Exception:
        return _undecided(head)
    if value is None:
        return head
    if isinstance(head, sp.Derivative):
        return _flat_between_jumps(value)
    return value


#: Functions that are constant between jumps, so their derivative is zero wherever it
#: exists.
_PIECEWISE_CONSTANT = (sp.floor, sp.ceiling, sp.sign, sp.Heaviside)


def _flat_between_jumps(derivative: sp.Basic) -> sp.Basic:
    """A derivative with the derivative of every step function taken as zero.

    Floor, sign and unit step are constant between jumps, so by convention their
    derivative is 0: `DIF(FLOOR(x)^2, x)` is 0 and `DIF(x*SIGN(x), x)` is `SIGN(x)`.
    Sympy instead produces `DiracDelta` or leaves an unevaluated derivative of a floor
    (possibly under a `Subs`); all of these become zero.
    """

    def is_flat(expression: sp.Basic) -> bool:
        if isinstance(expression, sp.Subs):
            return expression.expr.is_zero is True
        if isinstance(expression, sp.Derivative):
            return isinstance(expression.expr, _PIECEWISE_CONSTANT)
        return isinstance(expression, sp.DiracDelta)

    if not derivative.has(sp.DiracDelta, sp.Derivative):
        return derivative
    flattened = attempt(derivative, lambda d: d.replace(is_flat, lambda _: sp.S.Zero))
    return derivative if flattened is None else flattened


def _unfinished(head: sp.Basic, context: Context) -> bool:
    """Whether `head` contains an unexpanded call of a defined function.

    Such a call remains only when recursion hit the nesting limit, as with
    `F(n) := DIF(F(n + 1), x)`. It isn't a constant; it lacks `x` only because its body
    was never substituted. Differentiating would wrongly give zero, so the head waits.
    """
    if not context.functions:
        return False
    try:
        calls = head.atoms(AppliedUndef)
    except Exception:
        return False
    return any(type(call).__name__ in context.functions for call in calls)


def _approximation(head: sp.Basic, context: Context) -> sp.Basic:
    """`APPROX(u, n)`: `u` approximated to `n` digits.

    This is what approX does to a whole entry, but at the call's digit count rather
    than the session's. The same rules apply as outside a call, so `APPROX(2 + 3)` is 5.

    If the count isn't a number yet, the head waits: `PARTS(i_)` gives a count in terms
    of an enclosing sum's index, which becomes a number once the sum is expanded.
    """
    value, digits = head.args
    count = digit_count(digits)
    if count is None:
        return head
    approximate = context.with_precision(Precision.APPROXIMATE, count)
    return approximated(value, approximate)


def _summation(head: sp.Sum) -> sp.Basic:
    """A sum, with a polynomial closed form written factored.

    Derive's summation formulas are factored and stay that way: `SUM(k, k, 0, n)` is
    `n*(n + 1)/2` and `SUM(k^2, k, 1, n)` is `n*(n + 1)*(2*n + 1)/6`. Sympy gives
    Faulhaber's polynomial expanded, so it is factored here.

    This is done here rather than in the pipeline, which never factors: authored
    `n^2/2 + n/2` prints back unchanged in Derive. It isn't gated on length either:
    Derive prints `SUM(k^4, k, 1, n)` as `n*(n + 1)*(6*n^3 + 9*n^2 + n - 1)/30` even
    though that is longer. The normal form never distributes products, so nothing later
    undoes it.

    Only ranges of unknown length get a formula. A range of known length is added up
    term by term like any other sum: `SOLVE.MTH`'s three-term Lagrange expansion is
    `3*y^3/2 - y^2 + y` in the manual (9.1, p.230), not factored.
    """
    value = attempt(head, lambda s: s.doit(deep=False))
    if value is None:
        return _undecided(head)
    if _counted(head):
        return value
    value = _over_the_counts(head, value)
    if isinstance(value, sp.MatrixBase):
        # Each element of a vector sum is its own closed form: `SUM([k, k^2], k, 0, n)`
        # is a vector of the two factored formulas.
        entries = attempt(value, lambda m: m.applyfunc(_factored))
        return value if entries is None else entries
    return _factored(value)


def _counted(head: sp.Sum | sp.Product) -> bool:
    """Whether every range of `head` has a known number of terms.

    This distinguishes a sum added up term by term from one given by a formula. A
    symbolic or infinite bound makes the count unknown, as in `SUM(k, k, 1, n)` or
    `SUM(2^-k, k, 0, inf)`.
    """
    return all(
        len(limit) == 3 and (limit[2] - limit[1]).is_Integer for limit in head.limits
    )


#: How many term counts a closed form is checked against before it is trusted.
_COUNTS_CHECKED = 6


def _over_the_counts(head: sp.Sum, value: sp.Basic) -> sp.Basic:
    """The closed form of a sum whose summand involves its upper limit, valid for
    integer term counts.

    Sympy treats `SUM(COMB(n, k), k, 0, n)` as a hypergeometric series over all `k` and
    attaches a convergence condition for real `n`: `2^n` where `n > -1`, or
    `(x + 1)^n` with a condition on `ABS(x)`. But the sum only makes sense for integer
    `n`, where the series terminates at `n` on its own, so the condition is irrelevant:
    `(x + 1)^n` holds for every `x` (the binomial theorem). The condition is dropped if
    the closed form matches the explicit sum for the first few counts. Each checked
    range has at least one term; at `n = -1` the range is empty.

    Unconditional closed forms are checked too, since sympy's shortcuts can be wrong at
    small counts: it gives 0 for `SUM((-1)^k*COMB(n, k), k, 0, n)`, which is 1 at
    `n = 0`. Such an answer is replaced by the hypergeometric one (`0^n`) if that
    checks out, and otherwise the sum stands.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 3:
        return value
    index, low, high = head.limits[0]
    bounds = high.free_symbols
    if not low.is_Integer or len(bounds) != 1:
        return value
    (bound,) = bounds
    if bound == index or bound not in head.function.free_symbols:
        return value
    conditional = (
        isinstance(value, sp.Piecewise)
        and len(value.args) == 2
        and value.args[1].cond is sp.true
        and value.args[1].expr.has(sp.Sum)
    )
    if conditional:
        candidate = value.args[0].expr
    elif value.has(sp.Sum) or not isinstance(value, sp.Expr):
        return value
    else:
        candidate = value
    if _holds_at_the_counts(candidate, head, bound):
        return candidate
    hypergeometric = attempt(
        head.function,
        lambda f: sp.concrete.summations.eval_sum_hyper(f, (index, low, high)),
    )
    if isinstance(hypergeometric, sp.Piecewise):
        hypergeometric = hypergeometric.args[0].expr
    if hypergeometric is not None and not hypergeometric.has(sp.Sum):
        if _holds_at_the_counts(hypergeometric, head, bound):
            return hypergeometric
    return value if conditional else head


def _holds_at_the_counts(candidate: sp.Basic, head: sp.Sum, bound: sp.Symbol) -> bool:
    """Whether `candidate` equals the sum `head` for the first `_COUNTS_CHECKED` values
    of `bound` that give a nonempty range, for any values of the other variables."""
    index, low, high = head.limits[0]
    reduced = attempt(candidate, sp.gammasimp)
    if reduced is None:
        return False
    checked = 0
    for count in range(-4, 4 * _COUNTS_CHECKED):
        top = high.subs(bound, count)
        if not top.is_Integer or top < low:
            continue
        terms = head.function.subs(bound, count)
        added = attempt(terms, lambda t: sp.Sum(t, (index, low, top)).doit())
        closed = attempt(reduced, lambda c: c.subs(bound, count))
        if added is None or closed is None or closed.has(sp.nan, sp.zoo, sp.oo):
            return False
        difference = attempt(closed - added, sp.expand)
        if difference is None:
            return False
        if difference != 0:
            difference = attempt(difference, sp.simplify)
            if difference != 0:
                return False
        checked += 1
        if checked == _COUNTS_CHECKED:
            return True
    return False


def _closed_by_rule(head: sp.Sum | sp.Product, context: Context) -> sp.Basic | None:
    """A closed form for a sum or product sympy can't do, by a rule for its summand's
    shape, or None.

    Sums of sines or cosines of the index, products of exponentials and sums of
    logarithms each follow from one identity, while sympy either leaves them or never
    returns: see `_sum_of_waves`, `_product_of_powers` and `_sum_of_logarithms`. Ranges
    of known length are added up term by term instead.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 3 or _counted(head):
        return None
    if isinstance(head, sp.Product):
        return _product_of_powers(head, context)
    waves = _sum_of_waves(head, context)
    if waves is not None:
        return waves
    return _sum_of_logarithms(head, context)


def _sum_of_waves(head: sp.Sum, context: Context) -> sp.Basic | None:
    """Closed form of a sum of sines and cosines of a linear function of the index.

    The terms `SIN(a*k + b)` for `k` from `p` to `q` are `N = q - p + 1` equally spaced
    points on a circle, and their sum is the imaginary part of a geometric sum of
    `#e^(#i*(a*k + b))`:

        SIN(N*a/2)/SIN(a/2) * SIN(a*(p + q)/2 + b),

    and likewise for cosine: `SUM(SIN(k), k, 1, n)` is
    `SIN(n/2)*SIN((n + 1)/2)/SIN(1/2)`. If the step `a` is a multiple of `2*pi`, every
    term is equal and the sum is `N*SIN(b)`; if it may be one, the answer is an `IF`,
    just as `SUM(x^k, k, 0, n)` has one on `x = 1`.

    A telescoping pair is left to sympy: `SUM(SIN(k + 1) - SIN(k), k, 1, n)` gives
    `SIN(n + 1) - SIN(1)`, shorter than this formula.

    Terms with equal step and phase are combined, and non-trigonometric terms are summed
    separately. Products of waves, or waves with index-dependent amplitude, are left to
    `_searchable`, since sympy can't sum them and may never return.
    """
    index, low, high = head.limits[0]
    if low.has(sp.oo, -sp.oo) or high.has(sp.oo, -sp.oo):
        return None
    expanded = attempt(head.function, sp.expand)
    if expanded is None:
        return None
    waves: dict[tuple[type, sp.Basic, sp.Basic], sp.Basic] = {}
    rest = []
    for term in sp.Add.make_args(expanded):
        amplitude, wave = term.as_independent(index, as_Add=False)
        is_wave = isinstance(wave, (sp.sin, sp.cos))
        line = _linear(wave.args[0], index) if is_wave else None
        if line is not None:
            key = (type(wave), *line)
            waves[key] = waves.get(key, sp.S.Zero) + amplitude
        elif any(f.has(index) for f in term.atoms(TrigonometricFunction)):
            return None
        else:
            rest.append(term)
    if not waves or _telescoping(expanded, index):
        return None
    total = sp.S.Zero
    if rest:
        total = _worked_out(sp.Sum(sp.Add(*rest), head.limits[0]), context)
        if total.has(sp.Sum):
            return None
    count = high - low + 1
    for (wave, step, phase), amplitude in waves.items():
        half = step / 2
        spread = sp.sin(count * half) / sp.sin(half)
        generic = spread * wave(step * (low + high) / 2 + phase)
        same = count * wave(phase)
        if sp.sin(half).is_zero:
            closed = same
        elif sp.sin(half).is_zero is False:
            closed = generic
        else:
            closed = sp.Piecewise((same, sp.Eq(sp.sin(half), 0)), (generic, True))
        total += amplitude * closed
    return total


def _linear(argument: sp.Basic, index: sp.Basic) -> tuple[sp.Basic, sp.Basic] | None:
    """The slope and intercept of `argument` as a line in `index`, or None."""
    try:
        line = sp.Poly(argument, index)
    except sp.PolynomialError:
        return None
    if line.degree() != 1:
        return None
    step, phase = line.all_coeffs()
    if step.has(index) or phase.has(index):
        return None
    return step, phase


def _product_of_powers(head: sp.Product, context: Context) -> sp.Basic | None:
    """A product of powers of one positive base, as the base to the sum of exponents:
    `PRODUCT(#e^k, k, 1, n)` is `#e^(n*(n + 1)/2)` and `PRODUCT(#e^(1/k^2), k, 1, inf)`
    is `#e^(pi^2/6)`.

    By continuity this holds for infinite products too, giving `inf` or 0 for a
    divergent sum. If the sum has no closed form, the product stands:
    `PRODUCT(#e^(1/k), k, 1, n)` would need a harmonic number.
    """
    index = head.limits[0][0]
    function = head.function
    if isinstance(function, sp.exp):
        base, exponent = sp.E, function.args[0]
    elif isinstance(function, sp.Pow) and not function.base.has(index):
        base, exponent = function.base, function.exp
    else:
        return None
    if not exponent.has(index) or base.is_positive is not True:
        return None
    total = _worked_out(sp.Sum(exponent, head.limits[0]), context)
    if total.has(sp.Sum, sp.Product):
        return None
    return sp.exp(total) if base == sp.E else base**total


def _sum_of_logarithms(head: sp.Sum, context: Context) -> sp.Basic | None:
    """A sum of logarithms, as the logarithm of the product of their arguments if that
    has a closed form: `SUM(LN(k), k, 1, n)` is `LN(n!)` and `SUM(LN(1 + 1/k), k, 1, n)`
    telescopes to `LN(n + 1)`.

    This needs every argument in the range to be known positive. By continuity it also
    holds for infinite sums.
    """
    index, low, high = head.limits[0]
    if not isinstance(head.function, sp.log):
        return None
    argument = head.function.args[0]
    step = sp.Dummy("j", integer=True, nonnegative=True)
    reached = attempt(argument, lambda a: a.subs(index, low + step))
    if reached is None or reached.is_positive is not True:
        return None
    product = _worked_out(sp.Product(argument, head.limits[0]), context)
    if product.has(sp.Sum, sp.Product):
        return None
    return sp.log(product)


def _factored(value: sp.Basic) -> sp.Basic:
    """Factor a closed form if it is a polynomial; unevaluated sums are left alone."""
    if value.has(sp.Sum) or not _polynomial(value):
        return value
    factored = attempt(value, sp.factor)
    if factored is None:
        return value
    return factored


def _polynomial(value: sp.Basic) -> bool:
    """Whether `value` is a polynomial with at least one variable.

    Numbers are excluded because `factor` would write 12 as `2^2*3`; `SUM(k, k, 1, 5)`
    should stay 15. `value` need not be an expression (a sum of truth values is a truth
    value), so failures count as no.
    """
    try:
        return bool(value.free_symbols) and value.is_polynomial() is True
    except Exception:
        return False


def _product(head: sp.Product) -> sp.Basic:
    """A product, by sympy first and then by definition.

    Sympy can't do infinite products like `PRODUCT(1 - 1/k^2, k, 2, inf)` directly, but
    it can take the limit of the partial products, which is the definition. So an
    infinite product sympy leaves standing goes to `_partial_products`.
    """
    value = attempt(head, lambda p: p.doit(deep=False))
    if value is None:
        return head
    if not isinstance(value, sp.Product):
        return value
    partial = _partial_products(value)
    return value if partial is None else partial


def _partial_products(head: sp.Product) -> sp.Basic | None:
    """An infinite product as the limit of the products up to `n`, or None.

    This is sympy's documented workaround and simply the definition: for a
    hypergeometric body the partial product has a closed form in `n`, and its limit is
    the infinite product. `PRODUCT(1 - 1/k^2, k, 2, inf)` telescopes to a ratio of
    rising factorials with limit 1/2, as in Derive; Wallis' product gives `pi/2`, and
    `PRODUCT(k/(k + 1), k, 1, inf)` gives 0.

    Returns None if either step fails (`PRODUCT(COS(pi/2^k), k, 1, inf)` has no closed
    form; `PRODUCT(1 - x/k, k, 1, inf)` depends on the unknown sign of `x`), leaving the
    product unevaluated, as Derive does.

    The lower bound must be known finite, though a declared variable is fine:
    `PRODUCT(1 - 1/k^2, k, m, inf)` is `(m - 1)/m`.

    Oscillating products are left standing: sympy's limits of `(-1)^n` and `(-2)^n` are
    `nan` and complex infinity, and calling `PRODUCT(-1, k, 1, inf)` undefined would
    claim more than the head does. This matches `SUM((-1)^k, k, 1, inf)` standing
    unevaluated. Divergence to infinity is a value, so `PRODUCT((k + 1)/k, k, 1, inf)`
    is `inf`.

    The limit may come out in gammas of complex argument, which the reflection formula
    turns into a real hyperbolic: `PRODUCT(1 + 1/k^2, k, 1, inf)` is `SINH(pi)/pi`, not
    `1/(GAMMA(1 - #i)*GAMMA(1 + #i))`. `simplify` is used rather than `gammasimp`
    because only it also cancels the leftovers, and the expression is a constant anyway.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 3:
        return None
    index, low, high = head.limits[0]
    if high != sp.oo or not low.is_finite:
        return None
    count = sp.Dummy("n", integer=True, positive=True)
    partial = attempt(
        head.function,
        lambda body: sp.Product(body, (index, low, count)).doit(deep=False),
    )
    if partial is None or partial.has(sp.Product):
        return None
    value = attempt(partial, lambda closed: sp.limit(closed, count, sp.oo))
    if value is None or value.has(count, sp.Product, sp.Limit, sp.nan, sp.zoo):
        return None
    reduced = attempt(value, sp.simplify) if value.has(sp.gamma) else None
    return value if reduced is None else reduced


#: Heads that trigger sympy's costly hypergeometric search, which recognizes such
#: terms by their ratios.
_TERMS = (sp.factorial, sp.factorial2, sp.gamma, sp.rf, sp.ff)


def _searchable(head: sp.Basic) -> bool:
    """Whether searching for a closed form of this sum is known to terminate.

    A symbolic limit sends sympy into Gosper's algorithm, which never returns for a
    summand with a factorial of the index scaled by more than one, like `(n - 2*k)!`
    (how the classical polynomial families are written). Chebyshev's
    `SUM((-1)^k*(n - k)!/(k!*(n - 2*k)!)*(2*x)^(n - 2*k), k, 0, n/2)`, from Derive's
    utility library, was still running after forty minutes. There's no closed form
    anyway, so the sum stands.

    This is deliberately narrow: `SUM(COMB(n, k), k, 0, n)` (`2^n`) and Legendre's
    `SUM((-1)^k*COMB(n, k)*COMB(2*n - 2*k, n)*x^(n - 2*k), k, 0, n/2)` take half a
    second and are allowed.

    The other non-terminating search, for telescoping trigonometric pairs, is handled by
    `_trigonometric_terms`.
    """
    for limit in head.limits:
        if len(limit) < 3 or all(bound.is_number for bound in limit[1:]):
            continue
        if _scaled_term(head.function, limit[0]):
            return False
        if _trigonometric_terms(head.function, limit[0]):
            return False
    return True


def _trigonometric_terms(function: sp.Basic, index: sp.Basic) -> bool:
    """Whether two summand terms have trigonometric functions of the index without
    being shifts of each other, which hangs sympy's telescoping check.

    For a two-term summand, sympy first checks whether one term is the other's negation
    shifted along the index, by matching and then by solving `L(k + m) + R(k) = 0` for
    `m`. With trigonometric functions, the solve rewrites everything as exponentials
    and builds polynomials in them. `SUM(SIN(a*k) - SIN(b*k), k, 1, n)` takes four
    seconds, the four sines of the Fourier series in Derive's `INTEGRAL.MTH` take
    ninety, and `SUM(SIN(k)*COS(2*k) - SIN(3*k)*COS(5*k), k, 1, n)` never finishes. A
    third term doesn't help, since pairs are tried in turn.

    Nothing is lost by refusing: if that search fails, sympy can't sum the terms anyway.
    The exception is a pair the match alone settles, such as
    `SIN(k + 1)*COS(k + 1) - SIN(k)*COS(k)`, which telescopes to `SIN(n)*COS(n + 2)`, so
    the same match is tried here first. Hyperbolic terms are excluded; sympy sums them
    as exponentials quickly.
    """
    expanded = attempt(function, sp.expand)
    if expanded is None:
        return False
    waves = [
        term
        for term in sp.Add.make_args(expanded)
        if any(f.has(index) for f in term.atoms(TrigonometricFunction))
    ]
    return len(waves) >= 2 and not _telescoping(expanded, index)


def _telescoping(expanded: sp.Basic, index: sp.Basic) -> bool:
    """Whether the trigonometric terms of `expanded` are a pair sympy telescopes; see
    `_trigonometric_terms`."""
    waves = [
        term
        for term in sp.Add.make_args(expanded)
        if any(f.has(index) for f in term.atoms(TrigonometricFunction))
    ]
    return len(waves) == 2 and _shifted(waves[0], waves[1], index)


def _shifted(left: sp.Basic, right: sp.Basic, index: sp.Basic) -> bool:
    """Whether `right` is `-left` shifted by an integer, matched the way sympy's
    telescoping check does it, with `left` the shifted term."""
    shift = sp.Wild("shift")
    match = (-right).match(left.subs(index, index + shift))
    if match is None or shift not in match or not match[shift].is_Integer:
        return False
    return (left.subs(index, index + match[shift]) + right) == 0


def _scaled_term(function: sp.Basic, index: sp.Basic) -> bool:
    """Whether a factorial in `function` scales `index` by more than one."""
    for call in function.atoms(*_TERMS):
        for argument in call.args:
            coefficient = argument.coeff(index)
            if coefficient.is_number and abs(coefficient) > 1:
                return True
    return False


def _limit(head: sp.Limit) -> sp.Basic:
    """A limit, taken over the closed forms of the sums and products inside it.

    Sums and products under a limit stand when their closed form needs a function
    Derive lacks, and the limit can't be taken of the unevaluated head.
    `SUM(1/k^2, k, 1, n)` is a generalized harmonic number, not an acceptable answer,
    but its limit `pi^2/6` is. So the limit is taken of the closed forms, and only the
    result must be in Derive's vocabulary.

    A constant factor is pulled out first, since sympy's series code can lose a product
    of gammas of complex arguments and would give 1 for the limit of the partial
    products of `1 + 1/k^2`, instead of `SINH(pi)/pi`.
    """
    expression, variable, point, direction = head.args
    closed = _closed_forms(expression)
    constant, varying = sp.S.One, closed
    try:
        factor, rest = closed.as_independent(variable, as_Add=False)
    except Exception:
        factor = sp.S.One
    if factor != 1 and _finite_nonzero(factor):
        constant, varying = factor, rest
    if point is sp.S.NegativeInfinity:
        # Take the limit of the reflection at `inf` instead; sympy's limit at `-inf`
        # wrongly sends `2^x` to `inf`.
        varying, point, direction = varying.subs(variable, -variable), sp.oo, "-"
    asked = sp.Limit(varying, variable, point, dir=direction)
    try:
        value = asked.doit(deep=False)
    except Exception:
        value = _undecided(asked)
    if value is None or value == asked or (value.has(sp.zoo) and value != sp.zoo):
        signed = _limit_by_the_sign_of_a_parameter(asked)
        value = value if signed is None else signed
    if value is None or value == asked:
        return head
    if value.has(sp.Limit) and asked != head:
        return head
    value = _bounds_settled(constant * value)
    if value.has(sp.gamma) and value.is_number:
        value = forced(value, sp.simplify)
    split = attempt(value, lambda v: _split_on_a_sign(v, head))
    return value if split is None else split


def _split_on_a_sign(value: sp.Basic, head: sp.Limit) -> sp.Basic | None:
    """Rewrite a limit that depends on the sign of a function of its one parameter as a
    conditional on that sign.

    Sympy gives `LIM(a^x, x, inf)` as `#e^(inf*SIGN(LN(a)))`, meaning `inf` for `a > 1`
    and 0 for `0 < a < 1`. Each case gets the sign substituted, and the remaining cases
    (`a = 1`, or where the logarithm isn't real) keep the unevaluated limit. The sign of
    a bare parameter is left alone, since `inf*SIGN(a)` is as clear as a conditional.
    """
    value = _signs_of_reciprocals(value)
    signs = [
        found
        for found in value.atoms(sp.sign)
        if not isinstance(found.args[0], sp.Symbol)
    ]
    if len(signs) != 1:
        return None
    (signed,) = signs
    argument = signed.args[0]
    parameters = argument.free_symbols
    if len(parameters) != 1 or parameters != value.free_symbols:
        return None
    (parameter,) = parameters
    if not parameter.is_extended_real:
        return None
    cases = []
    for sign, relation in ((1, sp.StrictGreaterThan), (-1, sp.StrictLessThan)):
        where = sp.solve_univariate_inequality(
            relation(argument, 0), parameter, relational=True
        )
        if where in (sp.true, sp.false):
            return None
        written = value.xreplace({signed: sp.Integer(sign)})
        if written.has(sp.sign, sp.nan, sp.zoo):
            return None
        cases.append((written, where))
    return sp.Piecewise(*cases, (head, True))


def _limit_by_the_sign_of_a_parameter(asked: sp.Limit) -> sp.Basic | None:
    """A limit with a real parameter, combined from the cases where the parameter is
    positive and negative.

    Without the sign of `a`, sympy gives `ATAN(±inf/a)` for `LIM(ATAN(x/a), x, inf)` and
    fails on `LIM(TANH(a*x), x, inf)`. With it, sympy answers `pi/2` and `-pi/2`, or 1
    and -1. If one expression in `ABS(a)`, or that times `SIGN(a)`, matches both, it is
    the limit for all nonzero `a` (at zero the expression is a different function).
    """
    expression, variable, point = asked.args[:3]
    parameters = asked.free_symbols
    if len(parameters) != 1:
        return None
    (parameter,) = parameters
    if not (parameter.is_extended_real and parameter.is_positive is None):
        return None
    size = sp.Dummy("p", positive=True)
    answers = []
    for value in (size, -size):
        answer = attempt(asked, lambda a: a.subs(parameter, value).doit(deep=False))
        if answer is None or answer.has(sp.Limit, sp.zoo, sp.nan):
            return None
        answers.append(answer)
    above, below = answers
    magnitude = above.subs(size, sp.Abs(parameter))
    for candidate in (magnitude, sp.sign(parameter) * magnitude):
        if agrees(candidate.subs(parameter, -size), below):
            return candidate
    return None


def _finite_nonzero(constant: sp.Basic) -> bool:
    """Whether `constant` is a number proved finite and nonzero, by the assumptions or,
    where those fail (as for `GAMMA(1 + #i)`), by structure: products of such numbers,
    their powers to finite exponents, exponentials of finite numbers, and gamma off its
    poles (it has no zeros)."""
    if not constant.is_number:
        return False
    if constant.is_finite and constant.is_zero is False:
        return True
    if isinstance(constant, sp.Mul):
        return all(_finite_nonzero(factor) for factor in constant.args)
    if isinstance(constant, sp.Pow):
        return _finite_nonzero(constant.base) and constant.exp.is_finite is True
    if isinstance(constant, sp.exp):
        return constant.args[0].is_finite is True
    if isinstance(constant, sp.gamma):
        (argument,) = constant.args
        return argument.is_finite is True and (
            argument.is_integer is False or argument.is_positive is True
        )
    return False


def _closed_forms(expression: sp.Basic) -> sp.Basic:
    """`expression` with each sum and product replaced by sympy's closed form, in
    whatever functions, where it has one."""

    def closed(head: sp.Basic) -> sp.Basic:
        if not _searchable(head):
            return head
        value = attempt(head, lambda h: h.doit(deep=False))
        return head if value is None else value

    if not expression.has(sp.Sum, sp.Product):
        return expression
    return forced(
        expression,
        lambda e: e.replace(lambda h: isinstance(h, (sp.Sum, sp.Product)), closed),
    )


def _bounds_settled(expression: sp.Basic) -> sp.Basic:
    """`expression` with sign and absolute value of an `INTERVAL` evaluated.

    Both are monotone on each side of zero, so they map bounds to bounds:
    `SIGN(INTERVAL(-1, 1))` is `INTERVAL(-1, 1)` and `ABS(INTERVAL(-2, 1))` is
    `INTERVAL(0, 2)`. Sympy leaves them unevaluated, and its limits produce them:
    `LIM(x*SIN(x), x, inf)` comes back as `inf` times the sign of `SIN(x)`'s bounds,
    which works out to `INTERVAL(-inf, inf)`. Non-numeric bounds are left alone.
    """

    def numeric(bounds: sp.Basic) -> bool:
        return isinstance(bounds, sp.AccumBounds) and all(
            end.is_extended_real for end in bounds.args
        )

    def sign(bounds: sp.Basic) -> sp.Basic:
        low, high = bounds.args
        return sp.AccumBounds(sp.sign(low), sp.sign(high))

    def size(bounds: sp.Basic) -> sp.Basic:
        low, high = bounds.args
        if low.is_extended_nonnegative:
            return bounds
        if high.is_extended_nonpositive:
            return sp.AccumBounds(-high, -low)
        return sp.AccumBounds(0, sp.Max(-low, high))

    if not expression.has(sp.AccumBounds):
        return expression
    for head, rule in ((sp.sign, sign), (sp.Abs, size)):
        expression = forced(
            expression,
            lambda e, h=head, r=rule: e.replace(
                lambda c: isinstance(c, h) and numeric(c.args[0]),
                lambda c: r(c.args[0]),
            ),
        )
    return expression


def _undecided(head: sp.Basic) -> sp.Basic:
    """The value of a head whose evaluation raised.

    Only two-sided limits get an answer: sympy refuses one whose sides disagree, and
    sides that are negations of each other are exactly what `±` expresses. Sides that
    agree are the limit (`SIN(1/x)/x` gives `INTERVAL(-inf, inf)` from both). Anything
    else is returned as written.
    """
    if not (isinstance(head, sp.Limit) and str(head.args[3]) == "+-"):
        return head
    expression, variable, point = head.args[:3]
    sides = []
    for direction in ("+", "-"):
        try:
            side = sp.Limit(expression, variable, point, dir=direction).doit()
        except Exception:
            return head
        sides.append(_bounds_settled(side))
    right, left = sides
    if right.has(sp.Limit) or left.has(sp.Limit):
        return head
    if right == left:
        return right
    if right != -left:
        return sp.nan
    # Opposite infinite sides make unsigned infinity, which Derive writes `±inf`.
    return sp.zoo if right in (sp.oo, -sp.oo) else PlusMinus(right)


# -- the FACTOR, EXPAND and SOLVE heads ---------------------------------------


def _commanded(
    name: str, values: list, divisors: list, context: Context
) -> sp.Basic:
    """Run `FACTOR`, `EXPAND`, `SOLVE` or `RATE` on its evaluated arguments.

    `FACTOR(u, amount, x, y, ...)`, `EXPAND(u, amount, x, y, ...)` and `SOLVE(u, x)`
    are the in-expression forms of those commands, so simplifying them runs the
    command, as in the original.

    The operand is simplified at exact precision, as the command itself would do;
    rounding comes only at the end. The result is wrapped in `Commanded` so later steps
    don't rewrite it, while an enclosing call like `RHS(SOLVE(x^2 - 5*x + 6 = 0, x))`
    sees its value.

    `divisors` are what the `SOLVE` equation divides by as written, which the simplified
    operand may no longer show.
    """
    exact = context.with_precision(Precision.EXACT)
    args = [
        _transform(value, exact, decide=False) if not place or name == "RATE" else value
        for place, value in enumerate(values)
    ]
    match name:
        case "SOLVE":
            solved = _solve_head(args, divisors, context)
            return Commanded(_command_head(name, args) if solved is None else solved)
        case "RATE":
            return _rate_head(args, context)
    return Commanded(_factor_head(name, args, context))


def _factor_head(name: str, args: list, context: Context) -> sp.Basic:
    """Run a `FACTOR` or `EXPAND` head, parsing its target, amount and variables.

    After the target come an optional amount word and optional variables, in that order.
    A word that names no valid amount is read as a variable, so a meaningless call like
    Derive 6's `FACTOR(u, Turing)` stays an expression rather than an error. The same
    applies to `EXPAND(u, Complex)`, since Complex is an amount only Factor offers.
    """
    expanding = name == "EXPAND"
    amounts = EXPAND_AMOUNTS if expanding else tuple(Amount)
    target, *rest = args
    amount = DEFAULT_AMOUNT
    variables = []
    for argument in rest:
        named = amount_named(str(argument)) if isinstance(argument, sp.Symbol) else None
        if named is not None and named in amounts and not variables:
            amount = named
        elif isinstance(argument, sp.Symbol):
            variables.append(argument.name)
    if expanding:
        return expanded_expression(target, amount, variables, order=context.order)
    return factored_expression(target, amount, variables)


def _solve_head(args: list, divisors: list, context: Context) -> sp.Basic | None:
    """`SOLVE(u, x)` and its longer forms, as a vector of solutions.

    This follows Derive 3.x and 4.x: the answer is a *vector* of relations, which
    `DIMENSION` counts, `RHS` distributes over, and the shipped libraries rely on.
    Derive 5's disjunction of equations means the same but breaks Derive 4 worksheets.

    Elements are the entries the command would append, in order: a system gives one
    inner vector per solution, and no solutions give the empty vector.

    `SOLVE(u = v, x, a, b)` searches `[a, b]` numerically in every precision mode. The
    original's help says "if in approximate mode", but the shipped files assume it
    always applies, which is also more useful and less ambiguous.

    `SOLVE(u, x, Real)` keeps only solutions without an imaginary part (Derive 6's
    documented third argument). `SOLVE(u, x, Complex)` equals `SOLVE(u, x)`, Complex
    being the solve dialog's default domain.

    Returns None for any other form, leaving the call as written: an unknown third
    argument, more than four arguments, or a domain word in the variable's place. A
    misspelled `Reals` shouldn't be silently ignored.
    """
    target, *rest = args
    if len(rest) > 3 or (rest and _domain_word(rest[0]) is not None):
        return None
    variables = _solved_for(rest[0]) if rest else ()
    bounds = (rest[1], rest[2]) if len(rest) == 3 else None
    real = False
    if len(rest) == 2:
        domain = _domain_word(rest[1])
        if domain is None:
            return None
        real = domain == "Real"
    answers = solutions(target, context, variables, bounds, divisors, real)
    if not answers:
        return sp.Matrix(0, 0, [])
    return InertVector(*answers)


#: The variable the rate is solved for. It can't collide with anything, since the
#: search only runs when every argument is a number.
_RATE = "i"


def _rate_head(args: list, context: Context) -> sp.Basic:
    """`RATE(n, p, v, f, t, a, b)`: the contract's rate between `a` and `b`.

    The only financial function without a closed form, so it is done here rather than
    in conversion, as a bounded `SOLVE` on the annuity equation behind `PVAL` and the
    others. The interval picks which rate is found, and symbolic arguments can't be
    searched.

    If the interval holds no rate, the call stays as is, like in the original:
    `RATE(36, -300, 9000, 0, 0, 0.5, 1)`.
    """
    if len(args) != 7 or not all(argument.is_number for argument in args):
        return _command_head("RATE", args)
    periods, payment, present, future, when, low, high = args
    rate = sp.Symbol(_RATE)
    equation = sp.Eq(annuity(rate, periods, payment, present, future, when), 0)
    answers = solutions(equation, context, (_RATE,), (low, high))
    if len(answers) != 1 or not isinstance(answers[0], Relational):
        return _command_head("RATE", args)
    return answers[0].rhs


def _command_head(name: str, args: list) -> sp.Basic:
    """The unevaluated call, for when a command has no answer."""
    return sp.Function(name)(*args)


#: The solution domains a `SOLVE` head's third argument may name.
_DOMAIN_WORDS = ("Real", "Complex")


def _domain_word(argument: sp.Basic) -> str | None:
    """The solution domain `argument` names, or None."""
    if isinstance(argument, sp.Symbol) and argument.name in _DOMAIN_WORDS:
        return argument.name
    return None


def _solved_for(argument: sp.Basic) -> tuple[str, ...]:
    """The variables a `SOLVE` head names: one, or a vector of them."""
    if isinstance(argument, sp.Symbol):
        return (argument.name,)
    if isinstance(argument, sp.MatrixBase):
        elements = list(argument)
    elif isinstance(argument, InertVector):
        elements = list(argument.args)
    else:
        return ()
    return tuple(e.name for e in elements if isinstance(e, sp.Symbol))


# -- evaluating what is only a number ----------------------------------------


def _numeric(expression: sp.Basic) -> sp.Basic:
    """Evaluate constants sympy leaves unevaluated.

    `(3/2)!` is a number, which sympy evaluates when written as a gamma, so it is
    rewritten that way. The sign or absolute value of an `INTERVAL` is evaluated too.
    Neither is gated on length, since evaluating a constant always counts as
    simplifying.
    """
    expression = _bounds_settled(expression)
    try:
        return expression.replace(_is_fractional_factorial, _as_gamma, simultaneous=False)
    except Exception:
        return expression


def _is_fractional_factorial(expression: sp.Basic) -> bool:
    if not isinstance(expression, sp.factorial):
        return False
    argument = expression.args[0]
    return bool(argument.is_Rational) and not argument.is_integer


def _as_gamma(expression: sp.Basic) -> sp.Basic:
    return sp.gamma(expression.args[0] + 1)


def _as_binomial(whole: sp.Basic, part: sp.Basic) -> sp.Basic:
    return sp.factorial(whole) / (sp.factorial(part) * sp.factorial(whole - part))


def _as_falling(whole: sp.Basic, part: sp.Basic) -> sp.Basic:
    return sp.factorial(whole) / sp.factorial(whole - part)


#: Heads section 6.9 defines as factorial ratios: `COMB(z, w)` is `z!/(w!*(z - w)!)`
#: and `PERM(z, w)` is `z!/(z - w)!`.
_AS_FACTORIALS: dict[type, Callable[..., sp.Basic]] = {
    sp.binomial: _as_binomial,
    sp.ff: _as_falling,
}


def _ratios(expression: sp.Basic) -> sp.Basic:
    """Expand `_AS_FACTORIALS`, except under a calculus head.

    A `COMB` in an unevaluated sum is left alone: written out, it would give a new
    summand that the next Simplify would hand to sympy, and that search over a ratio of
    factorials of the index never returns. For example, `SUM(COMB(n, k), k, 0, n)` is
    `2^n` above zero and the unevaluated sum elsewhere. Compare `_searchable`.
    """
    written = _AS_FACTORIALS.get(type(expression))
    if written is not None:
        return written(*expression.args)
    if isinstance(expression, CALCULUS_HEADS) or not expression.args:
        return expression
    arguments = [_ratios(argument) for argument in expression.args]
    if all(written is argument for written, argument in zip(arguments, expression.args)):
        # Return the original rather than a rebuild: rebuilding normalizes (a matrix
        # inverse becomes a matrix power), so unaffected subtrees are not rebuilt.
        return expression
    return expression.func(*arguments)


def _factorials(expression: sp.Basic) -> sp.Basic:
    """Write the heads section 6.9 defines by factorials as factorials.

    `GAMMA(z)` becomes `(z - 1)!`, and the `_AS_FACTORIALS` heads become ratios.
    Factorials are Derive's preferred spelling, so this isn't gated on length even
    though all three get longer.

    This runs after the rewrites because `combsimp` produces most gammas and turns
    ratios of three factorials back into binomials. The ratios are also written out
    before `combsimp`, in `_by_definition`, so it can shorten them: `COMB(x, 2)` becomes
    `x*(x - 1)/2` rather than a ratio only the next Simplify would reduce.
    """
    try:
        if expression.has(*_AS_FACTORIALS):
            expression = _ratios(expression)
        return expression.replace(sp.gamma, _as_factorial, simultaneous=False)
    except Exception:
        return expression


def _as_factorial(argument: sp.Basic) -> sp.Basic:
    return sp.factorial(argument - 1)


# -- the kernel rules --------------------------------------------------------
#
# Each rewrite below is one kernel rule. The three tables at the end of the module
# decide which are tried, in which mode, and whether they must pay.


def _distributed(expression: sp.Basic) -> sp.Basic:
    """Distribute every matrix product over the sums inside it.

    `a . (b + c)` becomes `a . b + a . c`, and `(b + c) . a` becomes `b . a + c . a`.
    Section 8.8 states these as rules, so they are forced even though they add an
    operation (like `_complementary_arcs`).

    This is deliberately the only rewrite applied to expressions containing a matrix.
    The others are written for scalars and mangle matrices (`radsimp` returns a plain
    product of two matrices sympy can't work with), and sympy's construction already
    applies the other rules of section 8.8.
    """
    return expression.replace(
        _is_matrix_product, lambda product: product.expand(), simultaneous=False
    )


def _is_matrix_product(expression: sp.Basic) -> bool:
    return isinstance(expression, sp.MatMul) and any(
        isinstance(factor, sp.MatAdd) for factor in expression.args
    )


#: The hyperbolic functions, defined by exponentials in 6.5 (their inverses by
#: logarithms in 6.6).
_HYPERBOLIC = (sp.sinh, sp.cosh, sp.tanh, sp.coth, sp.sech, sp.csch)

def _as_arc_cosh(argument: sp.Basic) -> sp.Basic:
    """`ACOSH(z)` as `2*LN(SQRT(z + 1) + SQRT(z - 1)) - LN(2)`, per section 6.6.

    Sympy's rewrite gives `LN(SQRT(z - 1)*SQRT(z + 1) + z)`, the same value in a form
    nothing downstream collects. They agree on the whole plane:
    `(SQRT(z + 1) + SQRT(z - 1))^2` is `2*(z + SQRT(z - 1)*SQRT(z + 1))`, and both roots
    have arguments in `(-pi/2, pi/2]`, so doubling the logarithm crosses no cut.

    Numbers are left to sympy, which gets a shorter answer: `ACOSH(2)` is
    `LN(SQRT(3) + 2)` rather than `LN((SQRT(3) + 1)^2/2)`.
    """
    if argument.is_number:
        return sp.acosh(argument)
    return 2 * sp.log(sp.sqrt(argument + 1) + sp.sqrt(argument - 1)) - sp.log(2)


def _as_arc_coth(argument: sp.Basic) -> sp.Basic:
    """`ACOTH(z)` as `LN((z + 1)/(z - 1))/2`, per section 6.6.

    Sympy's rewrite gives `LN(1 + 1/z)/2 - LN(1 - 1/z)/2`, which nothing downstream
    collects into one logarithm. The two agree wherever either is defined.
    """
    return sp.log((argument + 1) / (argument - 1)) / 2


#: Heads rewritten here rather than by sympy's `rewrite`, which gives the right value
#: in a form the manual doesn't use.
_WRITTEN_AS: dict[type, Callable[..., sp.Basic]] = {
    sp.acosh: _as_arc_cosh,
    sp.acoth: _as_arc_coth,
}

#: For each function Derive never leaves in an answer, the function it is rewritten
#: in. `SEC`, `CSC`, `ASEC` and `ACSC` are reciprocals and their arcs (6.4), the
#: hyperbolics and their inverses are exponentials and logarithms (6.5, 6.6), and
#: `ERFC` is the complement of `ERF` (6.11). `STEP`, `MIN` and `MAX` use the closed
#: forms of 6.7: a step is a half plus half a sign, and min and max are the midpoint
#: minus or plus half the distance.
_DEFINED_OVER: dict[type, type] = {
    sp.sec: sp.cos,
    sp.csc: sp.sin,
    sp.asec: sp.acos,
    sp.acsc: sp.asin,
    sp.erfc: sp.erf,
    sp.Heaviside: sp.sign,
    sp.Min: sp.Abs,
    sp.Max: sp.Abs,
    **{head: sp.exp for head in _HYPERBOLIC},
    **{head: sp.log for head in INVERSE_HYPERBOLIC},
}


def _by_definition(expression: sp.Basic) -> sp.Basic:
    """Expand functions that are just alternative spellings by their definitions.

    These are not optional rewrites, so none is gated or offered as a setting: Derive
    never writes an answer with `SECH`, `ERFC` or `MAX`. Several make the expression
    longer (`MIN(x, y)` goes from two operations to seven), which the gate would reject.

    Sympy's rewrite is used where its form matches the manual's (`TANH(z)` is
    `(#e^(2*z) - 1)/(#e^(2*z) + 1)` in both), and `_WRITTEN_AS` otherwise. The factorial
    ratios are also written out here, before `combsimp`, so it can shorten them.

    Sympy heads Derive has no name for are first rewritten by `in_vocabulary`, so their
    replacements get expanded with the rest.
    """
    expression = forced(expression, in_vocabulary)
    for head, written in _WRITTEN_AS.items():
        if not expression.has(head):
            continue
        expression = forced(expression, lambda e, h=head, w=written: e.replace(h, w))
    if expression.has(*_AS_FACTORIALS):
        expression = forced(expression, _ratios)
    for head, over in _DEFINED_OVER.items():
        if not expression.has(head):
            continue
        expression = forced(expression, lambda e, h=head, o=over: _defined(e, h, o))
    return expression


def _defined(expression: sp.Basic, head: type, over: type) -> sp.Basic:
    """Rewrite every `head` in `expression` in terms of `over`, and nothing else.

    `rewrite` on the whole expression would also convert other functions: a `SIN` next
    to a `SINH` would become exponentials too.
    """
    return expression.replace(head, lambda *args: head(*args).rewrite(over))


def _gated(expression: sp.Basic, rewrite: Rewrite) -> sp.Basic:
    """`rewrite`, if the result is simpler than what went in."""
    candidate = attempt(expression, rewrite)
    if candidate is None:
        return expression
    return candidate if pays(candidate, expression) else expression


def _unnested(expression: sp.Basic, rewrite: Rewrite) -> sp.Basic:
    """`rewrite`, if the result is simpler or has less deeply nested radicals.

    Denesting counts as simplification even if it adds terms: `SQRT(2 + SQRT(3))` is
    `SQRT(6)/2 + SQRT(2)/2` and `(2 + SQRT(5))^(1/3)` is `SQRT(5)/2 + 1/2`. But not if
    the result more than doubles in length.
    """
    candidate = attempt(expression, rewrite)
    if candidate is None:
        return expression
    if pays(candidate, expression):
        return candidate
    try:
        flatter = _nesting(candidate) < _nesting(expression)
        short = sp.count_ops(candidate) <= 2 * sp.count_ops(expression)
    except Exception:
        return expression
    return candidate if flatter and short else expression


def _cancelled(expression: sp.Basic, rewrite: Rewrite) -> sp.Basic:
    """`rewrite`, if the result is simpler or clears calls from the denominator.

    A quotient of calls that cancels to a polynomial in them counts as simpler even if
    it isn't shorter, just like `(x^3 - 1)/(x - 1)`: `SIN(3*x)/SIN(x)` becomes
    `2*COS(2*x) + 1`. As with `_unnested`, not if the result more than doubles.
    """
    candidate = attempt(expression, rewrite)
    if candidate is None:
        return expression
    if pays(candidate, expression):
        return candidate
    try:
        cancelled = _called_denominator(expression) and not _called_denominator(
            candidate
        )
        short = sp.count_ops(candidate) <= 2 * sp.count_ops(expression)
    except Exception:
        return expression
    return candidate if cancelled and short else expression


def _called_denominator(expression: sp.Basic) -> bool:
    """Whether a trigonometric call stands in a denominator of `expression`."""
    return any(
        power.exp.is_negative and power.base.has(TrigonometricFunction)
        for power in expression.atoms(sp.Pow)
    )


def _nesting(expression: sp.Basic) -> int:
    """How deep the radicals of `expression` are nested in one another."""
    if not isinstance(expression, sp.Basic) or not expression.args:
        return 0
    if isinstance(expression, sp.Pow) and not expression.exp.is_Integer:
        return 1 + _nesting(expression.base)
    return max(_nesting(argument) for argument in expression.args)


def _signs_of_reciprocals(expression: sp.Basic) -> sp.Basic:
    """Remove the division from every `SIGN` of a negative power.

    For real `u`, `SIGN(1/u)` is `SIGN(u)` and `SIGN(1/u^2)` is 1; `SIGN(1/ABS(u))` is 1
    for any `u` (at zero the reciprocal is undefined anyway). A limit growing like
    `x^2/a` comes back as `inf*SIGN(1/a)`, which becomes `inf*SIGN(a)`.
    """

    def unsigned(head: sp.sign) -> sp.Basic:
        power = head.args[0]
        base, exponent = power.args
        if base.is_nonnegative:
            return sp.S.One
        if not (base.is_extended_real and exponent.is_integer):
            return head
        return sp.S.One if exponent.is_even else sp.sign(base)

    return expression.replace(
        lambda e: isinstance(e, sp.sign)
        and isinstance(e.args[0], sp.Pow)
        and e.args[0].exp.is_extended_negative,
        unsigned,
    )


#: Odd functions increasing on the whole real line, so `SIGN(f(u))` is `SIGN(u)` for
#: real `u`.
_SIGN_KEEPING = (sp.atan, sp.asinh, sp.sinh, sp.tanh, sp.erf)

#: Even functions, so `f(ABS(u))` is `f(u)` for real `u`.
_EVEN = (sp.cos, sp.sec, sp.cosh, sp.sech)


def _symmetries(expression: sp.Basic) -> sp.Basic:
    """Simplify `SIGN` of odd functions and logarithms, even functions of `ABS`, and
    `ABS` of products whose signs cancel.

    For real `u`, `SIGN(ATAN(u))` and `SIGN(u^3)` are `SIGN(u)`, and `COS(ABS(u))` is
    `COS(u)`. This must run before the definitions turn hyperbolics into exponentials,
    hiding the symmetry. Arguments not known to be real are left alone.
    """

    def signed(head: sp.sign) -> sp.Basic:
        simpler = _same_sign(head.args[0])
        return head if simpler is None else sp.sign(simpler)

    def even(call: sp.Basic) -> sp.Basic:
        inner = call.args[0]
        if isinstance(inner, sp.Abs) and inner.args[0].is_extended_real:
            return call.func(inner.args[0])
        return call

    # Repeat until stable, since one unwrapping can expose another:
    # `SIGN(ATAN(x)^3)` -> `SIGN(ATAN(x))` -> `SIGN(x)`.
    for _ in range(8):
        rewritten = expression.replace(lambda e: isinstance(e, sp.sign), signed)
        rewritten = rewritten.replace(lambda e: isinstance(e, _EVEN), even)
        if rewritten == expression:
            break
        expression = rewritten
    return _paired_signs(expression)


def _same_sign(inner: sp.Basic) -> sp.Basic | None:
    """A simpler expression that has the sign of `inner` everywhere, or None.

    An odd increasing function has the sign of its argument, as does an odd positive
    power of its base. A logarithm is increasing and crosses zero at 1, so `LN(u)` has
    the sign of `u - 1` where `u` can't be negative: `SIGN(LN(1 + COS(x)))` is
    `SIGN(COS(x))`. A logarithm of a negative number is complex, so `SIGN(LN(x))`
    stays. Likewise an exponential crosses 1 at zero, so `#e^u - 1` has the sign of
    `u` and `1 - #e^u` that of `-u`.
    """
    if isinstance(inner, _SIGN_KEEPING) and inner.args[0].is_extended_real:
        return inner.args[0]
    if isinstance(inner, sp.log) and never_negative(inner.args[0]):
        return inner.args[0] - 1
    # Negative exponents are handled by `_signs_of_reciprocals`.
    if isinstance(inner, sp.Pow) and inner.exp.is_positive:
        base, exponent = inner.args
        if base.is_extended_real and exponent.is_integer and exponent.is_odd:
            return base
    if isinstance(inner, sp.Add) and len(inner.args) == 2:
        for constant, other, sign in (
            (sp.S.NegativeOne, inner - sp.S.NegativeOne, 1),
            (sp.S.One, sp.S.One - inner, -1),
        ):
            if constant not in inner.args:
                continue
            if isinstance(other, sp.exp) and other.args[0].is_extended_real:
                return sign * other.args[0]
    return None


def _sign_kernel(factor: sp.Basic) -> tuple[sp.Basic, int]:
    """The simplest expression known to have the sign of `factor` or the opposite one,
    and 1 or -1 to say which.

    `ATAN(x)`, `SINH(2*x^3)` and `1/x` all reduce to `x`, and `1 - #e^x` to `x` with
    the opposite sign, so factors with the same kernel differ in sign by a constant.
    """
    sign = 1
    for _ in range(8):
        coefficient, factor = factor.as_coeff_Mul()
        if coefficient.is_negative:
            sign = -sign
        if isinstance(factor, sp.Add) and factor.could_extract_minus_sign():
            factor, sign = -factor, -sign
        if isinstance(factor, sp.Pow) and factor.exp.is_Integer and factor.exp.is_odd:
            simpler = factor.base
        else:
            simpler = _same_sign(factor)
        if simpler is None:
            break
        factor = simpler
    return factor, sign


def _paired_signs(expression: sp.Basic) -> sp.Basic:
    """Take out of each `ABS` the factors whose signs cancel or can't be negative.

    `ABS(x*ATAN(x))` is `x*ATAN(x)`, since both factors have the sign of `x`, and
    likewise `ABS(x)*ABS(SINH(x))` and `ABS(ATAN(x)/x)`. Factors are paired by their
    sign kernel; the odd one out of a group stays inside, as does anything unpaired:
    `ABS(x*y*ATAN(x))` is `x*ATAN(x)*ABS(y)`. A factor that can't be negative comes
    out too: `ABS(x*(COS(x) + 1))` is `(COS(x) + 1)*ABS(x)`.
    """

    def paired(product: sp.Basic) -> sp.Basic:
        inside: list[sp.Basic] = []
        outside: list[sp.Basic] = []
        for factor in sp.Mul.make_args(product):
            base, exponent = factor.as_base_exp()
            if isinstance(base, sp.Abs) and exponent in (1, -1):
                inside.extend(
                    part**exponent for part in sp.Mul.make_args(base.args[0])
                )
            else:
                outside.append(factor)
        # A reciprocal isn't known real, being undefined at zero, but its base is.
        if not all(part.as_base_exp()[0].is_extended_real for part in inside):
            return product
        if not all(
            part.is_extended_real or part.as_base_exp()[1].is_Integer
            for part in inside
        ):
            return product
        if any(part.has(sp.AccumBounds) for part in inside):
            return product
        groups: dict[sp.Basic, list[tuple[sp.Basic, int]]] = {}
        freed: list[sp.Basic] = []
        for part in inside:
            if never_negative(part):
                freed.append(part)
                continue
            kernel, sign = _sign_kernel(part)
            groups.setdefault(kernel, []).append((part, sign))
        staying: list[sp.Basic] = []
        for members in groups.values():
            if len(members) % 2:
                # The odd one out; the simplest, so the bars enclose the least.
                odd = min(members, key=lambda member: sp.count_ops(member[0]))
                members.remove(odd)
                staying.append(odd[0])
            freed.extend(sign * part for part, sign in members)
        if not freed:
            return product
        return sp.Mul(*outside, *freed, sp.Abs(sp.Mul(*staying)))

    if not expression.has(sp.Abs):
        return expression
    return expression.replace(
        lambda e: isinstance(e, (sp.Abs, sp.Mul)) and e.has(sp.Abs), paired
    )


def _conjugate_pairs(expression: sp.Basic) -> sp.Basic:
    """Replace each product of a quantity and its conjugate with its squared modulus.

    `CONJ(z)*z` is `ABS(z)^2`, and likewise for each pair in integer powers of the same
    sign: `z^2*CONJ(z)` is `z*ABS(z)^2`, and `1/(z*CONJ(z))` is `1/ABS(z)^2`.
    """

    def paired(product: sp.Mul) -> sp.Basic:
        powers: dict[sp.Basic, sp.Basic] = {}
        for factor in product.args:
            base, exponent = factor.as_base_exp()
            powers[base] = powers.get(base, 0) + exponent
        changed = False
        for base in list(powers):
            if not isinstance(base, sp.conjugate):
                continue
            plain = base.args[0]
            ours, theirs = powers.get(plain), powers[base]
            if ours is None or not (ours.is_integer and theirs.is_integer):
                continue
            if (ours > 0) != (theirs > 0):
                continue
            common = min(ours, theirs) if ours > 0 else max(ours, theirs)
            powers[plain] = ours - common
            powers[base] = theirs - common
            modulus = sp.Abs(plain)
            powers[modulus] = powers.get(modulus, 0) + 2 * common
            changed = True
        if not changed:
            return product
        return sp.Mul(*(base**exponent for base, exponent in powers.items()))

    if not expression.has(sp.conjugate):
        return expression
    return expression.replace(lambda e: isinstance(e, sp.Mul), paired)


def _denested(expression: sp.Basic) -> sp.Basic:
    """Denest each radical separately.

    `sqrtdenest` gives up on a sum of nested roots but handles each one alone:
    `SQRT(5 + SQRT(24)) + SQRT(5 - SQRT(24))` becomes `2*SQRT(3)` only if the terms are
    denested separately.

    `sqrtdenest` only handles square roots. Section 3.8's other example, a cube root of
    a sum of surds, is handled by `_flat_root`.
    """
    return expression.replace(
        lambda e: isinstance(e, sp.Pow) and e.exp.is_Rational and e.exp.q > 1,
        lambda e: sp.sqrtdenest(e) if e.exp is sp.S.Half else _flat_root(e),
        simultaneous=False,
    )


#: Maximum polynomial degree for denesting a root of surds. An nth root of a number
#: over k square roots satisfies a polynomial of degree at most n*2^k: twelve for
#: 3.8's `(243*SQRT(5) - 294*SQRT(3))^(1/3)` (four in practice). The bound is computed
#: before building any polynomial, so taller towers are refused cheaply instead of
#: spending minutes on a minimal polynomial of hundreds of terms.
_DENESTING_DEGREE = 12

#: The variable the minimal polynomial of a radical is written in.
_ROOT = sp.Dummy("root")


def _flat_root(power: sp.Pow) -> sp.Basic:
    """Remove the outer root from a root of a sum of surds, if possible.

    `(243*SQRT(5) - 294*SQRT(3))^(1/3)` is `3*SQRT(5) - 2*SQRT(3)`, since the cube root
    lies in the field of the radicand's surds. It denests exactly when its minimal
    polynomial has a linear factor over that field, which factoring over the extension
    reveals.

    Each linear factor is only a candidate: a quartic gives four, one per branch, and
    only one is the principal root. A candidate is kept only if raising it back gives
    the radicand and `proved_zero` proves it equal to the radical.
    """
    exponent = power.exp
    if exponent.p != 1 or not power.base.is_number or power.base.is_Rational:
        return power
    field = _surd_field(power.base)
    if field is None or exponent.q * 2 ** len(field) > _DENESTING_DEGREE:
        return power
    try:
        minimal = sp.minimal_polynomial(power, _ROOT)
        factored = sp.factor(minimal, extension=list(field))
    except Exception:
        return power
    for factor in sp.Mul.make_args(factored):
        candidate = _linear_root(factor)
        if candidate is None or candidate.count_ops() > 2 * power.count_ops():
            continue
        if sp.expand(sp.radsimp(candidate**exponent.q - power.base)) != 0:
            continue
        if proved_zero(candidate - power):
            return candidate
    return power


def _surd_field(radicand: sp.Basic) -> tuple[sp.Basic, ...] | None:
    """The square roots `radicand` is built from, or None if it isn't a sum of
    rational multiples of square roots of rationals.

    The field's degree sets the bound above. Anything else (a transcendental, a nested
    root, a cube root) is refused rather than handed to an expensive factoring.
    """
    surds = set()
    for term in sp.Add.make_args(radicand):
        for factor in sp.Mul.make_args(term):
            if factor.is_Rational:
                continue
            root = _rational_square_root(factor)
            if root is None:
                return None
            surds.add(root)
    if not surds:
        return None
    return tuple(sorted(surds, key=sp.default_sort_key))


def _rational_square_root(factor: sp.Basic) -> sp.Basic | None:
    """`SQRT(5)` for `SQRT(5)` and for `1/SQRT(5)`, and None for anything else."""
    if not isinstance(factor, sp.Pow) or abs(factor.exp) != sp.S.Half:
        return None
    if not (factor.base.is_Rational and factor.base.is_positive):
        return None
    return sp.sqrt(factor.base)


def _linear_root(factor: sp.Basic) -> sp.Basic | None:
    """The root of `factor`, if it is linear in `_ROOT`."""
    try:
        polynomial = sp.Poly(factor, _ROOT)
    except sp.PolynomialError:
        return None
    if polynomial.degree() != 1:
        return None
    above, below = polynomial.all_coeffs()
    return sp.radsimp(-below / above)


def _flattened(expression: sp.Basic) -> sp.Basic:
    """Reduce every constant containing nested radicals to the field of its surds.

    `_denested` works one radical at a time, which misses 3.8 p.47's
    `(SQRT(SQRT(3) + 1) + SQRT(3*SQRT(3) + 3))/SQRT(24*SQRT(3) + 40)`, equal to 1/2
    although none of its four roots simplifies alone (the numerator is
    `(SQRT(3) + 1)^(3/2)` and the denominator twice that). So the `_flat_root` question
    is asked of whole sums and products: does the value lie in the field of its surds?

    Only constants with nested roots are tried. Sympy already combines roots of
    rationals, so the minimal polynomial would only confirm that at a cost.
    """
    return expression.replace(
        lambda part: isinstance(part, sp.Add | sp.Mul), _flat_part, simultaneous=False
    )


def _flat_part(part: sp.Add | sp.Mul) -> sp.Basic:
    """`part` with its constant flattened.

    The tree walk reaches constant terms of a sum directly, but not the constant part of
    a product: `x` times the quotient from 3.8 is one product of five factors, so the
    numeric factors must be gathered first.
    """
    if part.is_number:
        return _flat_value(part)
    if not isinstance(part, sp.Mul):
        return part
    constants = [factor for factor in part.args if factor.is_number]
    if len(constants) < 2:
        return part
    value = sp.Mul(*constants)
    flat = _flat_value(value)
    if flat is value:
        return part
    return flat * sp.Mul(*[factor for factor in part.args if not factor.is_number])


#: Maximum field degree for flattening a constant, computed like `_DENESTING_DEGREE`
#: as the product of its root orders: sixteen for the quotient of 3.8 (four in
#: practice). Two more roots would mean a degree-64 minimal polynomial and minutes of
#: work, usually for nothing.
_FLATTENING_DEGREE = 16


def _flat_value(constant: sp.Basic) -> sp.Basic:
    """`constant` written in the field of its surds, if its value lies there.

    The same steps as `_flat_root`: take the minimal polynomial, factor it over the
    field to get a linear factor per branch, and keep only a candidate `proved_zero`
    proves equal to the constant, since a shorter but different number is the one
    mistake to avoid.
    """
    field, degree = _radical_field(constant)
    if field is None or degree > _FLATTENING_DEGREE:
        return constant
    try:
        minimal = sp.minimal_polynomial(constant, _ROOT)
        factored = sp.factor(minimal, extension=list(field))
    except Exception:
        return constant
    for factor in sp.Mul.make_args(factored):
        candidate = _linear_root(factor)
        if candidate is None or candidate.count_ops() >= constant.count_ops():
            continue
        if proved_zero(candidate - constant):
            return candidate
    return constant


def _radical_field(constant: sp.Basic) -> tuple[tuple[sp.Basic, ...] | None, int]:
    """The roots of rationals in `constant` and a degree bound, or None if it isn't a
    nested radical over them.

    The generators are the roots with rational radicands. The degree is the product of
    the orders of all roots, an upper bound on the field's degree. Anything other than
    rational arithmetic over such roots (a transcendental, a variable, a root of a
    negative) is refused, as is a constant with no nesting, which is already flat.
    """
    generators: set[sp.Basic] = set()
    orders: dict[sp.Pow, int] = {}
    if not _over_radicals(constant, generators, orders):
        return None, 0
    if not generators or all(root.base.is_Rational for root in orders):
        return None, 0
    degree = 1
    for order in orders.values():
        degree *= order
    return tuple(sorted(generators, key=sp.default_sort_key)), degree


def _over_radicals(
    part: sp.Basic, generators: set[sp.Basic], orders: dict[sp.Pow, int]
) -> bool:
    """Whether `part` is rational arithmetic over roots of positive rationals,
    collecting the generators and the order of every root."""
    if part.is_Rational:
        return True
    if isinstance(part, sp.Add | sp.Mul):
        return all(_over_radicals(argument, generators, orders) for argument in part.args)
    if not (isinstance(part, sp.Pow) and part.exp.is_Rational):
        return False
    if not _over_radicals(part.base, generators, orders):
        return False
    if part.exp.q > 1:
        orders[part] = part.exp.q
        if part.base.is_Rational:
            if not part.base.is_positive:
                return False
            # Use the plain qth root; it generates the same field as its powers.
            generators.add(sp.Pow(part.base, sp.Rational(1, part.exp.q)))
    return True


#: Maximum field degree for evaluating trigonometric constants: the degree of the
#: cyclotomic field of their angles times the orders of accompanying roots, which
#: bounds the minimal polynomial's cost. Sympy writes a cosine past a quarter turn as a
#: sine with twice the denominator, so pi/11 becomes pi/22, needing degree 20.
_TRIGONOMETRIC_DEGREE = 24
_TRIGONOMETRIC_SIZE = 10**12


def _trigonometric_constants(expression: sp.Basic) -> sp.Basic:
    """Write sums and products of trigonometric values at rational multiples of pi as
    rationals or quadratic surds, where they are.

    `COS(pi/7) - COS(2*pi/7) + COS(3*pi/7)` is 1/2 and
    `SIN(pi/7)*SIN(2*pi/7)*SIN(3*pi/7)` is `SQRT(7)/8`, though no single call
    simplifies. Each value is algebraic; if its minimal polynomial has degree one or
    two, it is solved. `_cyclotomic_root` proves the value is a root exactly, and when
    there are two roots of different sign, the one matching `_known_sign` is kept.

    Single calls are skipped: sympy already evaluates those it can, and the rest have no
    degree-two form.
    """
    return expression.replace(
        lambda part: isinstance(part, sp.Add | sp.Mul),
        _trigonometric_part,
        simultaneous=False,
    )


def _trigonometric_part(part: sp.Add | sp.Mul) -> sp.Basic:
    """`part` with its trigonometric constant evaluated, gathering it from a product
    as `_flat_part` does."""
    if part.is_number:
        return _trigonometric_value(part)
    if not isinstance(part, sp.Mul):
        return part
    constants = [factor for factor in part.args if factor.is_number]
    if len(constants) < 2:
        return part
    value = sp.Mul(*constants)
    written = _trigonometric_value(value)
    if written is value:
        return part
    return written * sp.Mul(*[factor for factor in part.args if not factor.is_number])


def _trigonometric_value(part: sp.Basic) -> sp.Basic:
    calls = part.atoms(TrigonometricFunction)
    if not calls or (len(calls) == 1 and not part.has(sp.Pow)):
        return part
    degree = _cyclotomic_degree(part)
    if degree is None or degree > _TRIGONOMETRIC_DEGREE:
        return part
    try:
        minimal = sp.Poly(sp.minimal_polynomial(part, _ROOT), _ROOT)
    except Exception:
        return part
    # An answer with coefficients this large is no simpler.
    if minimal.degree() > 2 or max(map(abs, minimal.coeffs())) > _TRIGONOMETRIC_SIZE:
        return part
    if not _cyclotomic_root(part, minimal):
        return part
    candidates = list(dict.fromkeys(sp.roots(minimal, multiple=True)))
    if len(candidates) > 1:
        sign = _known_sign(part)
        signs = [_surd_sign(candidate) for candidate in candidates]
        if sign is None or None in signs or signs.count(sign) != 1:
            return part
        candidates = [candidates[signs.index(sign)]]
    (candidate,) = candidates
    return candidate if candidate.count_ops() < part.count_ops() else part


def _cyclotomic_root(part: sp.Basic, polynomial: sp.Poly) -> bool:
    """Whether a constant of trigonometric values at rational multiples of pi is proved
    a root of the polynomial.

    With `z` a primitive `M`-th root of unity, `#e^(#i*pi*p/q)` is a power of `z` when
    `M` is a multiple of `2*q`, and so is `#i` when `M` is a multiple of 4. Each
    trigonometric value is then a rational function of `z`, and so is the polynomial
    evaluated at the constant. It is zero exactly when its numerator is divisible by
    the order-`M` cyclotomic polynomial, the minimal polynomial of `z`. Constants with
    anything else in them, including roots of rationals, are not proved.
    """
    calls = part.atoms(TrigonometricFunction)
    turns = [call.args[0] / sp.pi for call in calls]
    if not all(turn.is_Rational for turn in turns):
        return False
    order = sp.ilcm(4, *(2 * turn.q for turn in turns))
    z = sp.Dummy("z")

    def power(turn: sp.Rational) -> sp.Basic:
        # `#e^(#i*pi*turn)` as a power of `z`, which is `#e^(2*#i*pi/order)`.
        return z ** int(turn * order / 2)

    written = {}
    for call, turn in zip(calls, turns):
        rising, falling = power(turn), power(-turn)
        sine = (rising - falling) / (2 * power(sp.Rational(1, 2)))
        cosine = (rising + falling) / 2
        quotients = {
            sp.sin: sine, sp.cos: cosine, sp.tan: sine / cosine, sp.cot: cosine / sine,
            sp.sec: 1 / cosine, sp.csc: 1 / sine,
        }
        if call.func not in quotients:
            return False
        written[call] = quotients[call.func]
    value = part.xreplace(written)
    if value.free_symbols != {z} or any(
        not power_.exp.is_Integer for power_ in value.atoms(sp.Pow) if power_.base != z
    ) or value.has(sp.I):
        return False
    try:
        numerator, _ = sp.fraction(sp.together(polynomial.as_expr().subs(_ROOT, value)))
        remainder = sp.rem(sp.expand(numerator), sp.cyclotomic_poly(order, z), z)
    except Exception:
        return False
    return remainder == 0


def _known_sign(constant: sp.Basic) -> int | None:
    """The sign of a constant built from rationals, roots of positive rationals and
    trigonometric values at rational multiples of pi, determined from structure, or
    None.

    A trigonometric value's sign comes from where its angle falls in the period;
    products multiply signs, and a sum of terms with one sign has that sign."""
    if constant.is_Rational:
        return _sign(constant.p)
    if isinstance(constant, sp.Mul):
        signs = [_known_sign(factor) for factor in constant.args]
        return None if None in signs else reduce(lambda a, b: a * b, signs, 1)
    if isinstance(constant, sp.Add):
        signs = {_known_sign(term) for term in constant.args}
        if len(signs) == 1:
            return signs.pop()
        return _surd_sign(constant) or _bounded_sign(constant)
    if isinstance(constant, sp.Pow):
        if constant.base.is_Rational and constant.base > 0:
            return 1
        if not constant.exp.is_Integer:
            return None
        base = _known_sign(constant.base)
        return None if not base else base ** abs(int(constant.exp))
    if isinstance(constant, TrigonometricFunction):
        turn = constant.args[0] / sp.pi
        if not turn.is_Rational:
            return None
        sine = _half_turn_sign(turn)
        cosine = _half_turn_sign(turn + sp.Rational(1, 2))
        signs = {
            sp.sin: sine, sp.csc: sine, sp.cos: cosine, sp.sec: cosine,
            sp.tan: sine * cosine, sp.cot: sine * cosine,
        }
        return signs.get(constant.func)
    return None


def _bounded_sign(constant: sp.Add) -> int | None:
    """The sign of `r + c*t`, for rational `r` and `c` and a trigonometric value `t` at
    a rational multiple of pi, if the range of `t` decides it. Sine and cosine lie in
    [-1, 1] and secant and cosecant outside it (sympy has already evaluated values at
    the endpoints)."""
    rational, rest = constant.as_coeff_Add()
    coefficient, call = rest.as_coeff_Mul()
    if not (rational.is_Rational and coefficient.is_Rational and coefficient != 0):
        return None
    func = call.func
    if isinstance(call, sp.Pow) and call.exp == -1:
        # `1/COS(x)` is the secant and `1/SIN(x)` the cosecant.
        func = {sp.cos: sp.sec, sp.sin: sp.csc}.get(call.base.func)
        call = call.base
    if not isinstance(call, TrigonometricFunction) or _known_sign(call) is None:
        return None
    # The sign of `c*(t + r/c)`.
    offset = rational / coefficient
    if func in (sp.sin, sp.cos):
        inner = 1 if offset >= 1 else -1 if offset <= -1 else None
    elif func in (sp.sec, sp.csc):
        inner = _known_sign(call) if abs(offset) <= 1 else None
    else:
        inner = None
    return None if inner is None else _sign(coefficient.p) * inner


def _half_turn_sign(turn: sp.Rational) -> int:
    """The sign of the sine of `pi*turn`."""
    place = turn % 2
    return 0 if place in (0, 1) else 1 if place < 1 else -1


def _surd_sign(number: sp.Basic) -> int | None:
    """The sign of a rational, or of a rational plus a rational multiple of the square
    root of a positive rational (by comparing squares), or None."""
    rational, rest = number.as_coeff_Add()
    if not rest:
        return _sign(rational.p) if rational.is_Rational else None
    coefficient, root = rest.as_coeff_Mul()
    if not (
        rational.is_Rational
        and coefficient.is_Rational
        and isinstance(root, sp.Pow)
        and root.exp == sp.S.Half
        and root.base.is_Rational
        and root.base > 0
    ):
        return None
    first, second = _sign(rational.p), _sign(coefficient.p)
    if first == second or first == 0:
        return second
    # Opposite signs: the larger square wins.
    return first * _sign((rational**2 - coefficient**2 * root.base).p)


def _cyclotomic_degree(constant: sp.Basic) -> int | None:
    """A bound on the degree of the field containing `constant`, or None if it isn't
    rational arithmetic over trigonometric values at rational multiples of pi and roots
    of positive rationals."""
    denominator, orders = 1, 1
    for part in sp.preorder_traversal(constant):
        if isinstance(part, TrigonometricFunction):
            turn = part.args[0] / sp.pi
            if not turn.is_Rational:
                return None
            denominator = sp.ilcm(denominator, turn.q)
        elif isinstance(part, sp.Pow):
            if not part.exp.is_Rational:
                return None
            if part.exp.q > 1:
                if not (part.base.is_Rational and part.base.is_positive):
                    return None
                orders *= part.exp.q
        elif isinstance(part, sp.Function) or (
            part.is_Atom and not part.is_Rational and part is not sp.pi
        ):
            return None
    return int(sp.totient(2 * denominator)) * orders


# -- the trigonometric, logarithmic and power rules --------------------------


#: Each inverse function and its complement. `ASIN(u) + ACOS(u)` is a right angle
#: wherever defined; the tangent pair is a right angle with the sign of its argument.
#: `ASEC` and `ACSC` are covered because they are rewritten as `ACOS` and `ASIN`
#: first.
_COMPLEMENT = {
    sp.asin: sp.acos,
    sp.acos: sp.asin,
    sp.atan: sp.acot,
    sp.acot: sp.atan,
}


def _complementary_arcs(expression: sp.Basic) -> sp.Basic:
    """Combine pairs of complementary arcs of the same argument."""
    return expression.replace(
        lambda e: _right_angle(e) is not None,
        lambda e: _right_angle(e),
        simultaneous=False,
    )


def _right_angle(expression: sp.Basic) -> sp.Basic | None:
    """This sum with a complementary pair of arcs replaced by its value, or None.

    The pair must share both argument and coefficient, which makes the rule work in
    degree mode too, where each arc is multiplied by `180/pi` and the pair adds to `90`.
    """
    if not isinstance(expression, sp.Add):
        return None
    arcs: dict[tuple, int] = {}
    for index, term in enumerate(expression.args):
        arc, coefficient = _arc(term)
        if arc is None:
            continue
        arcs[(type(arc), coefficient, arc.args[0])] = index
        # `ATAN(1/u)` is `ACOT(u)` for real nonzero `u`, so it pairs with `ATAN(u)`.
        if isinstance(arc, sp.atan) and not arc.args[0].is_number:
            arcs.setdefault((sp.acot, coefficient, 1 / arc.args[0]), index)
    for (func, coefficient, argument), index in arcs.items():
        other = arcs.get((_COMPLEMENT[func], coefficient, argument))
        if other is None or other == index:
            continue
        pair = {type(_arc(expression.args[i])[0]) for i in (index, other)}
        if pair == {sp.atan}:
            # For `ATAN(u) + ATAN(1/u)`, either argument gives the sign; use the
            # shorter.
            func, argument = sp.atan, min(argument, 1 / argument, key=sp.count_ops)
        angle = _quarter_turn(func, argument)
        if angle is None:
            continue
        rest = [t for i, t in enumerate(expression.args) if i not in (index, other)]
        return sp.Add(coefficient * angle, *rest)
    return None


def _arc(term: sp.Basic) -> tuple[sp.Basic | None, sp.Basic]:
    """The complementable arc in `term`, and its coefficient."""
    arc, others = None, []
    for factor in sp.Mul.make_args(term):
        if arc is None and type(factor) in _COMPLEMENT:
            arc = factor
        else:
            others.append(factor)
    return arc, sp.Mul(*others)


def _quarter_turn(func: type, argument: sp.Basic) -> sp.Basic | None:
    """What the pair adds up to, or None.

    `ATAN(t) + ACOT(t)` is `pi/2` for positive `t` and `-pi/2` for negative, i.e.
    `pi*SIGN(t)/2`. Derive answers `pi/2` regardless, which is wrong for negative `t`.

    At zero, `ACOT(0)` is `pi/2` by the convention sympy and Derive share, so the sum is
    `pi/2` while `SIGN` gives 0. A nonnegative argument therefore gets `pi/2`, exact
    everywhere; the general answer is off only at the single point where `ACOT` jumps.

    The identity fails for complex arguments (at `#i/2` the sum is `-pi/2`), so a pair
    not known to be real is left as written.
    """
    if func in (sp.atan, sp.acot):
        if not _real_where_defined(argument):
            return None
        if argument.is_nonnegative:
            return sp.pi / 2
        if argument.is_negative:
            return -sp.pi / 2
        return sp.pi * sp.sign(argument) / 2
    return sp.pi / 2


#: Limits for combining arctangents of numbers: the number of terms, the integer
#: multiple each may carry, and the bit size the Gaussian integer products may reach.
_ARCTANGENTS = 8
_ARCTANGENT_MULTIPLE = 12
_ARCTANGENT_BITS = 4096


def _arc_tangent_sums(expression: sp.Basic) -> sp.Basic:
    """Combine sums of arctangents of rationals where the result is simpler."""
    return expression.replace(
        lambda e: isinstance(e, sp.Add) and e.has(sp.atan),
        _arc_tangent_sum,
        simultaneous=False,
    )


def _arc_tangent_sum(expression: sp.Add) -> sp.Basic:
    """This sum with its arctangents of rationals combined.

    `ATAN(p/q)` with `q > 0` is the argument of the Gaussian integer `q + p*#i`, so an
    integer combination of them is the argument of a product of such integers, up to
    whole turns. That argument is a rational multiple of pi exactly when the product
    lies on an axis or diagonal, giving Machin's `4*ATAN(1/5) - ATAN(1/239) = pi/4` and
    `ATAN(2) + ATAN(3) = 3*pi/4`. Whole turns are counted while building the product,
    since each factor turns it by less than a right angle.

    Otherwise, arctangents are merged into one only if they form a plain sum and the new
    argument is no larger than the old ones: `ATAN(1/3) + ATAN(1/7)` becomes
    `ATAN(1/2)`, but `ATAN(1/2) + ATAN(1/4)` doesn't become `ATAN(6/7)`.

    Terms are grouped by their non-rational coefficient, so this works in degree mode
    (where each arc has a factor of `180/pi`) and for `y*ATAN(1/2) + y*ATAN(1/3)`.
    """
    groups: dict[sp.Basic, list[tuple[sp.Rational, sp.Rational]]] = {}
    rest = []
    for term in expression.args:
        arc = _rational_arc(term)
        if arc is None:
            rest.append(term)
            continue
        scale, coefficient, argument = arc
        groups.setdefault(scale, []).append((coefficient, argument))
    if not any(len(arcs) > 1 for arcs in groups.values()):
        return expression
    terms = list(rest)
    for scale, arcs in groups.items():
        # Divide out the common measure to get the integer multiples.
        measure = reduce(sp.gcd, [coefficient for coefficient, _ in arcs])
        multiples = [(int(coefficient / measure), r) for coefficient, r in arcs]
        if any(abs(multiple) > _ARCTANGENT_MULTIPLE for multiple, _ in multiples):
            terms.extend(coefficient * scale * sp.atan(r) for coefficient, r in arcs)
            continue
        terms.extend(measure * scale * term for term in _combined_arcs(multiples))
    return sp.Add(*terms)


def _rational_arc(
    term: sp.Basic,
) -> tuple[sp.Basic, sp.Rational, sp.Rational] | None:
    """`(scale, c, r)` where `term` is `c*scale*ATAN(r)` for rationals `c` and `r`."""
    arcs = [factor for factor in sp.Mul.make_args(term) if isinstance(factor, sp.atan)]
    if len(arcs) != 1 or not arcs[0].args[0].is_Rational:
        return None
    coefficient, scale = (term / arcs[0]).as_coeff_Mul()
    if not coefficient.is_Rational or scale.has(sp.atan):
        return None
    return scale, coefficient, arcs[0].args[0]


def _combined_arcs(arcs: list[tuple[int, sp.Rational]]) -> list[sp.Basic]:
    """Combine terms `k*ATAN(r)`, merging each subset that lands on an axis or
    diagonal."""
    bits = sum(
        abs(multiple) * (max(abs(r.p), r.q).bit_length() + 1) for multiple, r in arcs
    )
    if len(arcs) > _ARCTANGENTS or bits > _ARCTANGENT_BITS:
        return [multiple * sp.atan(r) for multiple, r in arcs]
    terms = []
    left = list(arcs)
    for size in range(len(left), 1, -1):
        found = True
        while found and len(left) >= size:
            found = False
            for chosen in combinations(range(len(left)), size):
                part = [left[index] for index in chosen]
                angle = _straight_angle(part)
                if angle is not None:
                    terms.append(angle)
                    left = [arc for index, arc in enumerate(left) if index not in chosen]
                    found = True
                    break
    if len(left) > 1 and all(abs(multiple) == 1 for multiple, _ in left):
        single = _single_arc(left)
        if single is not None:
            return [*terms, single]
    return [*terms, *(multiple * sp.atan(r) for multiple, r in left)]


def _gaussian(arcs: list[tuple[int, sp.Rational]]) -> tuple[int, int, int]:
    """The Gaussian integer whose argument is `sum(k*ATAN(r))` modulo whole turns, and
    the number of turns: the sum equals the principal argument in `(-pi, pi]` plus
    `2*pi` times the turns.

    A negative multiple uses the conjugate, which has the opposite argument. Each factor
    turns the product by less than a right angle, so a full turn is detected when the
    principal argument moves against the factor's direction, using only integer signs.
    """
    real, imaginary, turns = 1, 0, 0
    for multiple, r in arcs:
        a, b = r.q, r.p if multiple > 0 else -r.p
        for _ in range(abs(multiple)):
            before = (real, imaginary)
            real, imaginary = real * a - imaginary * b, real * b + imaginary * a
            if b > 0 and _argument_below((real, imaginary), before):
                turns += 1
            elif b < 0 and _argument_below(before, (real, imaginary)):
                turns -= 1
    return real, imaginary, turns


def _argument_below(first: tuple[int, int], second: tuple[int, int]) -> bool:
    """Whether the first nonzero Gaussian integer has a smaller principal argument than
    the second: compare half planes (`(-pi, 0]` below `(0, pi]`), then the sign of the
    cross product."""

    def upper(z: tuple[int, int]) -> bool:
        return z[1] > 0 or (z[1] == 0 and z[0] < 0)

    if upper(first) != upper(second):
        return upper(second)
    return first[0] * second[1] - first[1] * second[0] > 0


#: Principal argument, in eighths of a turn, of a Gaussian integer on an axis or
#: diagonal, keyed by the signs of its real and imaginary parts.
_DIRECTIONS = {
    (1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3,
    (-1, 0): 4, (-1, -1): -3, (0, -1): -2, (1, -1): -1,
}


def _sign(number: int) -> int:
    return (number > 0) - (number < 0)


def _straight_angle(arcs: list[tuple[int, sp.Rational]]) -> sp.Basic | None:
    """The rational multiple of pi these arcs add up to, or None."""
    real, imaginary, turns = _gaussian(arcs)
    if real != 0 and imaginary != 0 and abs(real) != abs(imaginary):
        return None
    direction = _DIRECTIONS[_sign(real), _sign(imaginary)]
    return sp.pi * sp.Rational(direction + 8 * turns, 4)


def _single_arc(arcs: list[tuple[int, sp.Rational]]) -> sp.Basic | None:
    """These arcs as one arctangent, if its argument is no larger than theirs and the
    sum lies within the arctangent's range (the product is right of the imaginary
    axis)."""
    real, imaginary, turns = _gaussian(arcs)
    if real <= 0 or turns != 0:
        return None
    argument = sp.Rational(imaginary, real)
    if max(abs(argument.p), argument.q) > max(max(abs(r.p), r.q) for _, r in arcs):
        return None
    return sp.atan(argument)


def _real_where_defined(argument: sp.Basic) -> bool:
    """Whether `argument` is real wherever it is defined.

    Sympy won't say so of `1/x` or `x/y`, which are infinite where the denominator
    vanishes, but the arcs are undefined there anyway, so any rational function of real
    variables with real coefficients counts.
    """
    if argument.is_extended_real:
        return True
    return (
        bool(argument.is_rational_function())
        and all(symbol.is_real for symbol in argument.free_symbols)
        and not argument.has(sp.I)
    )


def _simpler_arcs(expression: sp.Basic) -> sp.Basic:
    """Rewrite arcs in terms of a shorter argument where possible."""
    return expression.replace(
        lambda e: _shorter_arc(e) is not None,
        lambda e: _shorter_arc(e),
        simultaneous=False,
    )


def _shorter_arc(expression: sp.Basic) -> sp.Basic | None:
    """This arc rewritten using a simpler side of its triangle, or None.

    A right triangle with legs `u` and `1` has hypotenuse `SQRT(u^2 + 1)`, so
    `ASIN(u/SQRT(u^2 + 1))` is `ATAN(u)`, and likewise `ATAN(u/SQRT(1 - u^2))` is
    `ASIN(u)`.

    Both hold only for real `u` (for complex `u` the root's branch cut makes them
    differ by a period), so arguments not known to be real are left alone.
    """
    if not isinstance(expression, (sp.asin, sp.atan)):
        return None
    sides = _over_a_root(expression.args[0])
    if sides is None:
        return None
    leg, square = sides
    if not leg.is_real:
        return None
    if isinstance(expression, sp.asin) and _vanishes(square - leg**2 - 1):
        return sp.atan(leg)
    if isinstance(expression, sp.atan) and _vanishes(square + leg**2 - 1):
        return sp.asin(leg)
    return None


def _over_a_root(argument: sp.Basic) -> tuple[sp.Basic, sp.Basic] | None:
    """`u` and `v` where `argument` is `u/SQRT(v)`."""
    numerator, denominator = sp.fraction(sp.together(argument))
    if not (isinstance(denominator, sp.Pow) and denominator.exp is sp.S.Half):
        return None
    return numerator, denominator.base


def _vanishes(difference: sp.Basic) -> bool:
    """Whether this difference expands to zero.

    Recognizing the triangle in `(x + 1)/SQRT(x^2 + 2*x + 2)` requires expanding rather
    than comparing trees.
    """
    return bool(sp.expand(difference).is_zero)


#: Arcs whose small multiples under a sine or cosine are expanded, and the largest
#: multiple. `TAN` is expanded only over the arctangent, the one arc whose multiple
#: tangent is a rational function.
_ARCS = (sp.asin, sp.acos, sp.atan)
_ARC_MULTIPLE = 4


def _arc_multiples(expression: sp.Basic) -> sp.Basic:
    """Expand a sine, cosine or tangent of a small multiple of an arc algebraically.

    Sympy writes `SIN(ACOS(x))` as `SQRT(1 - x^2)`, and the multiple-angle formulas
    extend that to `SIN(2*ACOS(x))` = `2*x*SQRT(1 - x^2)`. This holds on the whole
    plane, since the formulas are identities.
    """
    return expression.replace(_is_arc_multiple, sp.expand_trig, simultaneous=False)


def _is_arc_multiple(expression: sp.Basic) -> bool:
    if not isinstance(expression, (sp.sin, sp.cos, sp.tan)):
        return False
    multiple, arc = expression.args[0].as_coeff_Mul()
    if not (multiple.is_Integer and 1 < multiple <= _ARC_MULTIPLE):
        return False
    if isinstance(expression, sp.tan):
        return isinstance(arc, sp.atan)
    return isinstance(arc, _ARCS)


def _half_angles(expression: sp.Basic) -> sp.Basic:
    """Rewrite every trigonometric call in terms of the angle they all share.

    `1/(1 + TAN(a)*TAN(a/2))` is `COS(a)`, which no rule working on one call at a time
    can see. Written over the half angle, the tangents cancel.

    The angle used is the greatest common measure of the arguments, so nothing is
    expanded more than necessary.
    """
    measure = _common_angle(expression)
    if measure is None:
        return expression
    return sp.trigsimp(TR11(TR2(expression), base=measure))


def _common_angle(expression: sp.Basic) -> sp.Basic | None:
    """The greatest angle every trigonometric argument is a multiple of.

    None if there is no such angle, or if every argument is an integer multiple of it;
    that case is the multiple-angle expansion `trigsimp` already does. Only fractional
    multiples are handled here.
    """
    calls = expression.atoms(TrigonometricFunction)
    if not calls:
        return None
    measure, angles = None, set()
    for call in calls:
        coefficient, angle = call.args[0].as_coeff_Mul()
        if not coefficient.is_Rational:
            return None
        angles.add(angle)
        measure = coefficient if measure is None else sp.gcd(measure, coefficient)
    if len(angles) != 1 or measure.is_Integer:
        return None
    return measure * angles.pop()


def _phase_angles(expression: sp.Basic) -> sp.Basic:
    """Rewrite `a*SIN(z) + b*COS(z)` as a single sine.

    fu's `TR10i` does this only when the coefficients match, but the rule holds for any.
    """
    return expression.replace(
        lambda e: _phase_angle(e) is not None,
        lambda e: _phase_angle(e),
        simultaneous=False,
    )


def _phase_angle(expression: sp.Basic) -> sp.Basic | None:
    """`SQRT(a^2 + b^2)*SIN(z + ATAN(b, a))`, if that is what this is."""
    if not isinstance(expression, sp.Add) or len(expression.args) != 2:
        return None
    parts: dict[type, tuple[sp.Basic, sp.Basic]] = {}
    for term in expression.args:
        coefficient, rest = term.as_independent(sp.sin, sp.cos)
        if not isinstance(rest, (sp.sin, sp.cos)):
            return None
        parts[rest.func] = (coefficient, rest.args[0])
    if set(parts) != {sp.sin, sp.cos}:
        return None
    (a, angle), (b, other) = parts[sp.sin], parts[sp.cos]
    if angle != other:
        return None
    return sp.sqrt(a**2 + b**2) * sp.sin(angle + sp.atan2(b, a))


#: Limits for `_multiple_angles`: the largest degree in the single angle (a call's
#: multiple times its power) and the expression's operation count. Degree twelve covers
#: any identity a worksheet states.
_MULTIPLE_ANGLE = 12
_MULTIPLE_OPERATIONS = 60
_DEGREE = 4 * _MULTIPLE_ANGLE


def _multiple_angles(expression: sp.Basic) -> sp.Basic:
    """Reduce sines and cosines of integer multiples of one angle together.

    `trigsimp` rewrites each term of a sum separately, so terms that cancel only
    together are left half done: `SIN(x)^6 + COS(x)^6 + 3*SIN(x)^2*COS(x)^2` is 1, but
    `trigsimp` stops at rewriting the last term as `3/8 - 3*COS(4*x)/8`. Nor does it
    cancel `SIN(3*x)/SIN(x)`, which needs both calls over the same angle.

    Over the single angle, a polynomial in sine and cosine has unique forms: a sum of
    sines and cosines of multiples (as `trigsimp` writes `SIN(x)^2*COS(x)^2`), or a
    polynomial in just one of the two functions. The shortest is chosen, preferring the
    sum of multiples on ties, so `SIN(3*x)/SIN(x)` is `2*COS(2*x) + 1`.
    """
    angle = _whole_angle(expression)
    if angle is None:
        return expression
    single = sp.cancel(sp.expand_trig(expression))
    sine, cosine = sp.sin(angle), sp.cos(angle)
    candidates = []
    # A quotient may cancel further after rewriting squares of one function in the
    # other: `SIN(x)^3/(COS(x)^2 - 1)` is `-SIN(x)`.
    for form in (
        single,
        sp.cancel(sp.expand(TR5(single, max=_DEGREE))),
        sp.cancel(sp.expand(TR6(single, max=_DEGREE))),
    ):
        if not _trigonometric_polynomial(form, sine, cosine):
            candidates.append(form)
            continue
        squares = sp.expand(form)
        candidates += [
            _sum_of_multiples(squares),
            sp.expand(TR5(squares, max=_DEGREE)),
            sp.expand(TR6(squares, max=_DEGREE)),
        ]
    return min(candidates, key=sp.count_ops)


def _whole_angle(expression: sp.Basic) -> sp.Basic | None:
    """The angle every sine and cosine in `expression` is an integer multiple of.

    None unless at least two different multiples occur (otherwise `trigsimp` suffices),
    if other trigonometric functions appear, or if the expression is too large.
    """
    calls = expression.atoms(TrigonometricFunction)
    if not calls or any(not isinstance(call, (sp.sin, sp.cos)) for call in calls):
        return None
    angles, multiples = set(), set()
    for call in calls:
        multiple, angle = call.args[0].as_coeff_Mul()
        if not multiple.is_Integer or abs(multiple) > _MULTIPLE_ANGLE:
            return None
        angles.add(angle)
        multiples.add(abs(multiple))
    # A power of a call multiplies its degree in the single angle.
    for power in expression.atoms(sp.Pow):
        if isinstance(power.base, (sp.sin, sp.cos)):
            multiple = power.base.args[0].as_coeff_Mul()[0]
            if abs(power.exp * multiple) > _MULTIPLE_ANGLE:
                return None
    if len(angles) != 1 or len(multiples) < 2:
        return None
    if sp.count_ops(expression) > _MULTIPLE_OPERATIONS:
        return None
    return angles.pop()


def _trigonometric_polynomial(
    expression: sp.Basic, sine: sp.Basic, cosine: sp.Basic
) -> bool:
    """Whether `expression` is a polynomial in this sine and cosine."""
    s, c = sp.Dummy(), sp.Dummy()
    bare = expression.xreplace({sine: s, cosine: c})
    return bool(bare.is_polynomial(s, c)) and not bare.has(TrigonometricFunction)


def _sum_of_multiples(polynomial: sp.Basic) -> sp.Basic:
    """A polynomial in sines and cosines as a sum of calls of multiple angles.

    Each `TR8` round turns products into sums but may leave new squares, so it repeats
    until stable; the degree bounds the number of rounds.
    """
    for _ in range(2 * _MULTIPLE_ANGLE):
        written = sp.expand(TR8(polynomial))
        if written == polynomial:
            break
        polynomial = written
    return polynomial


def _nonnegatively(rewrite: Rewrite) -> Rewrite:
    """Run `rewrite` with every nonnegative variable treated as positive.

    Section 6.2 states the logarithm rules for nonnegative arguments, but sympy's
    `logcombine` and `expand_log` require `is_positive`, skipping a variable declared
    `[0, inf)`. The only difference is zero, where both sides of every rule are the same
    infinity, so the manual's condition is safe.

    Only declared variables are affected; for an undeclared `x`, `LN(x^2 - x) - LN(x)`
    stays as is.
    """

    def run(expression: sp.Basic) -> sp.Basic:
        standing = {
            symbol: sp.Dummy(symbol.name, **{**symbol.assumptions0, "positive": True})
            for symbol in expression.free_symbols
            if isinstance(symbol, sp.Symbol) and symbol.is_nonnegative
            if not symbol.is_positive
        }
        if not standing:
            return rewrite(expression)
        back = {dummy: symbol for symbol, dummy in standing.items()}
        return rewrite(expression.xreplace(standing)).xreplace(back)

    return run


def _logarithmic_powers(expression: sp.Basic) -> sp.Basic:
    """Rewrite `b^LN(a)` as `a^LN(b)` for positive rationals `a < b`.

    Both equal `#e^(LN(a)*LN(b))`, so `2^LN(3) - 3^LN(2)` is 0, but that is only visible
    with a common base. The smaller base is used, also for an exponential of a product
    of two logarithms: `#e^(LN(2)*LN(3))` becomes `2^LN(3)`.
    """
    return expression.replace(
        lambda e: _over_the_smaller_base(e) is not None,
        lambda e: _over_the_smaller_base(e),
        simultaneous=False,
    )


def _over_the_smaller_base(expression: sp.Basic) -> sp.Basic | None:
    if isinstance(expression, sp.exp):
        base, exponent = sp.E, expression.args[0]
    elif isinstance(expression, sp.Pow):
        base, exponent = expression.args
    else:
        return None
    logarithms = [
        factor
        for factor in sp.Mul.make_args(exponent)
        if isinstance(factor, sp.log) and _positive_rational(factor.args[0])
    ]
    if base is sp.E:
        for factor in sp.Mul.make_args(exponent):
            inner, power = factor.as_base_exp()
            if isinstance(inner, sp.log) and _positive_rational(inner.args[0]):
                if power.is_Integer and power > 1:
                    logarithms.extend([inner, inner])
        if len(logarithms) < 2:
            return None
        logarithms = sorted(logarithms, key=lambda log: log.args[0])[:2]
        smaller, larger = logarithms[0].args[0], logarithms[1].args[0]
        rest = exponent / (logarithms[0] * logarithms[1])
        return sp.Pow(smaller, rest * sp.log(larger))
    if not (_positive_rational(base) and logarithms):
        return None
    argument = min(log.args[0] for log in logarithms)
    if argument >= base:
        return None
    rest = exponent / sp.log(argument)
    return sp.Pow(argument, rest * sp.log(base))


def _positive_rational(number: sp.Basic) -> bool:
    return bool(number.is_Rational and number.is_positive and number != 1)


def _powers_of_logarithms(expression: sp.Basic, order: Sequence[str]) -> sp.Basic:
    """Rewrite `#e^(u*LN(b))` as `b^u`, which is what the power means.

    Each term of the exponent with a logarithm among its factors becomes a power, and
    the terms without one stay over `#e`: `#e^(3*x*LN(y) + 2)` is `#e^2*y^(3*x)` and
    `#e^(x*LN(y) + x*LN(z))` is `y^x*z^x`. A power of a logarithm is not such a factor,
    so `#e^(LN(y)^2)` stands, and so does a complex number: `#e^(#i*LN(2))` is how a
    principal value is written.

    Of several logarithms in one term, the one whose argument has the most main variable
    gives the base, and the others stay in the exponent: `#e^(LN(x)*LN(y))` is
    `x^LN(y)`, and `#e^(x*LN(2)*LN(y))` is `y^(x*LN(2))`.
    """
    return expression.replace(
        lambda e: isinstance(e, sp.exp) and e.args[0].has(sp.log),
        lambda e: _power_of_logarithms(e, order),
        simultaneous=False,
    )


def _power_of_logarithms(exponential: sp.Basic, order: Sequence[str]) -> sp.Basic:
    powers, standing = [], []
    for term in sp.Add.make_args(sp.expand_mul(exponential.args[0])):
        logarithms = [
            factor for factor in sp.Mul.make_args(term) if isinstance(factor, sp.log)
        ]
        if not logarithms or (term.is_number and term.is_real is not True):
            standing.append(term)
            continue
        base = min(logarithms, key=lambda log: _mainness(log.args[0], order))
        powers.append(sp.Pow(base.args[0], term / base))
    if not powers:
        return exponential
    return sp.Mul(*powers) * sp.exp(sp.Add(*standing))


def _mainness(argument: sp.Basic, order: Sequence[str]) -> tuple[int, int]:
    """A sort key that puts the argument with the most main variable first, and numbers
    last."""
    names = {symbol.name for symbol in argument.free_symbols}
    if not names:
        return (1, 0)
    return (0, main_order({*names, *order}, order).index(main_order(names, order)[0]))


def _extracted_powers(expression: sp.Basic) -> sp.Basic:
    """Rewrite `LN(z^k)` as `k*LN(z)` wherever the rule holds."""
    return expression.replace(
        lambda e: _extracted_power(e) is not None,
        lambda e: _extracted_power(e),
        simultaneous=False,
    )


def _extracted_power(expression: sp.Basic) -> sp.Basic | None:
    """`k*LN(z)`, if `expression` is a logarithm of `z^k` that may be split.

    Section 6.2 gives `k*LN(x) <-> LN(x^k)` for rational `k` when `x` is nonnegative,
    and for any `x` when `-1 < k <= 1`. An even power of a real `x` equals that power of
    its absolute value, so it is split too: `LN(x^2)` is `2*LN(ABS(x))` and `LN(x^(-2))`
    is `-2*LN(ABS(x))`, following the original.

    A product or a polynomial that is a power as a whole counts as one: `LN(4*x^2)` is
    `2*LN(2*ABS(x))` and `LN(x^2 + 2*x + 1)` is `2*LN(ABS(x + 1))`.

    A factor `#e^u` with real `u` is positive, so its logarithm leaves whatever it
    multiplies: `LN(2*x^3*#e^x)` is `LN(2*x^3) + x`.
    """
    if not isinstance(expression, sp.log) or len(expression.args) != 1:
        return None
    argument = expression.args[0]
    if argument.is_Rational and argument.is_positive:
        power = _perfect_power(argument)
        return None if power is None else power[1] * sp.log(power[0])
    if argument.is_Mul:
        exponentials, rest = sp.sift(
            argument.args,
            lambda factor: isinstance(factor, sp.exp) and factor.args[0].is_real is True,
            binary=True,
        )
        if exponentials:
            remaining = sp.log(sp.Mul(*rest))
            split = _extracted_power(remaining)
            exponents = sp.Add(*(factor.args[0] for factor in exponentials))
            return (remaining if split is None else split) + exponents
    if argument.is_Add and argument.is_polynomial():
        argument = sp.factor(argument)
    if argument.is_Mul:
        power = _whole_power(argument)
        if power is None:
            return None
        base, exponent = power
    elif argument.is_Pow:
        base, exponent = argument.args
    else:
        return None
    if not exponent.is_Rational:
        return None
    if base.is_nonnegative or bool(-1 < exponent <= 1):
        return exponent * sp.log(base)
    if exponent.is_Integer and exponent.is_even and _real_where_defined(base):
        return exponent * sp.log(_magnitude(base))
    return None


def _magnitude(expression: sp.Basic) -> sp.Basic:
    """`ABS(expression)`, with the reciprocal outside: `x/ABS(y)`, where sympy answers
    `x*ABS(1/y)` for a positive `x`."""
    return sp.Abs(expression).replace(
        lambda e: isinstance(e, sp.Abs) and e.args[0].is_Pow and e.args[0].exp.is_Integer,
        lambda e: sp.Abs(e.args[0].base) ** e.args[0].exp,
    )


def _whole_power(product: sp.Mul) -> tuple[sp.Basic, sp.Integer] | None:
    """`(z, k)` where `product` is `z^k`, for the largest integer `k` above one.

    Every factor must be an integer power and a numeric coefficient a perfect power:
    `4*x^2` is `(2*x)^2` and `x^4*y^2` is `(x^2*y)^2`, while `2*x^2` and `x^2*y` aren't
    powers. `k` is negative if every exponent is: `1/(x^2*y^2)` is `(x*y)^(-2)`.
    """
    coefficient, rest = product.as_coeff_Mul()
    powers = [factor.as_base_exp() for factor in sp.Mul.make_args(rest)]
    if not all(exponent.is_Integer for _, exponent in powers):
        return None
    exponents = [exponent for _, exponent in powers]
    root = sp.S.One
    if coefficient != 1:
        if not (coefficient.is_Rational and coefficient.is_positive):
            return None
        perfect = _perfect_power(coefficient)
        if perfect is None:
            return None
        root, exponent = perfect
        exponents.append(exponent)
    k = sp.Integer(sp.igcd(*exponents))
    if k < 2:
        return None
    if all(exponent < 0 for _, exponent in powers):
        k = -k
    if coefficient != 1:
        root = root ** (exponents[-1] / abs(k))
        if k < 0:
            root = 1 / root
    return root * sp.Mul(*(base ** (exponent / k) for base, exponent in powers)), k


def _perfect_power(number: sp.Rational) -> tuple[sp.Rational, sp.Integer] | None:
    """`(z, k)` where `number` is `z^k`, for the largest integer `k` above one.

    The numerator and denominator must be powers with a common exponent: 256 is 2^8 and
    4/9 is (2/3)^2, while 12 and 4/3 aren't perfect powers.
    """
    parts = [sp.perfect_power(whole) or (whole, 1) for whole in (number.p, number.q)]
    exponents = [exponent for base, exponent in parts if base > 1]
    if not exponents:
        return None
    exponent = gcd(*exponents)
    if exponent < 2:
        return None
    (top, above), (bottom, below) = parts
    root = sp.Rational(top ** (above // exponent), bottom ** (below // exponent))
    return root, sp.Integer(exponent)


def _collected_powers(expression: sp.Basic) -> sp.Basic:
    """Combine powers of the same base in a product, and nothing else.

    This follows Derive's spelling rather than saving length: sympy gives `DIF(x^n, x)`
    as `n*x^n/x` and the original as `n*x^(n - 1)`, the same length, so a gate would
    keep sympy's. Under Exponential Expand, Derive does answer `n*x^n/x`.

    `combine="exp"` applies only the exponent rule, leaving different bases under one
    exponent alone, since that part of `powsimp` is a rewriting direction rather than a
    normal form.
    """
    return sp.powsimp(expression, combine="exp")


#: Maximum terms and total degree of a radicand for trying square-free factorization.
#: Perfect squares in worksheets are small, and factoring larger radicands is wasted
#: effort.
_PERFECT_TERMS = 24
_PERFECT_DEGREE = 12


def _perfect_powers(expression: sp.Basic) -> sp.Basic:
    """Take roots of polynomials that are perfect powers.

    For real `x`, `SQRT(x^2 + 2*x + 1)` is `ABS(x + 1)`, just like `SQRT((x + 1)^2)`, and
    `(x^2 + 2*x + 1)^(3/2)` is `(x + 1)^2*ABS(x + 1)`. The radicand must be a positive
    rational times a power of a polynomial, and sympy takes the root using declared
    domains: no bars when the base is provably nonnegative (`SQRT(x^4 + 2*x^2 + 1)` is
    `x^2 + 1`), and a complex variable keeps its root.
    """
    return expression.replace(_is_polynomial_root, _perfect_root, simultaneous=False)


def _is_polynomial_root(expression: sp.Basic) -> bool:
    return (
        isinstance(expression, sp.Pow)
        and bool(expression.exp.is_Rational)
        and not expression.exp.is_Integer
        and isinstance(expression.base, sp.Add)
        and len(expression.base.args) <= _PERFECT_TERMS
        and bool(expression.base.free_symbols)
    )


def _perfect_root(power: sp.Pow) -> sp.Basic:
    base, exponent = power.args
    try:
        polynomial = sp.Poly(base, *sorted(base.free_symbols, key=sp.default_sort_key))
    except sp.PolynomialError:
        return power
    if not polynomial.domain.is_QQ and not polynomial.domain.is_ZZ:
        return power
    if polynomial.total_degree() > _PERFECT_DEGREE:
        return power
    content, factors = sp.sqf_list(base)
    multiple = reduce(gcd, (int(times) for _, times in factors), 0)
    if multiple < 2 or not content.is_positive:
        return power
    root = sp.Mul(*(factor ** (times // multiple) for factor, times in factors))
    whole = root**multiple
    taken = content**exponent * whole**exponent
    if any(found.base == whole for found in taken.atoms(sp.Pow)):
        return power
    return taken


def _real_roots(expression: sp.Basic) -> sp.Basic:
    """Take odd roots of negative numbers as real.

    Sympy has already pulled out the sign (`(-8)^(1/3)` is `2*(-1)^(1/3)`), so this acts
    on the leftover `(-1)^(p/q)`, which has a real root exactly when `q` is odd.
    """
    return expression.replace(_is_odd_root_of_minus_one, _real_sign, simultaneous=False)


def _is_odd_root_of_minus_one(expression: sp.Basic) -> bool:
    return (
        isinstance(expression, sp.Pow)
        and expression.base is sp.S.NegativeOne
        and bool(expression.exp.is_Rational)
        and expression.exp.q % 2 == 1
    )


def _real_sign(expression: sp.Basic) -> sp.Basic:
    return sp.Integer(-1) ** expression.exp.p


def _rectangular(expression: sp.Basic) -> sp.Basic:
    """Write a non-real numeric power in rectangular form.

    Section 4.5 says complex results come out in rectangular form: `(1 + #i)^3` is
    `-2 + 2*#i` and the principal `(-8)^(1/3)` is `1 + SQRT(3)*#i`. Sympy reaches
    neither on its own.

    Only powers with numeric base and exponent qualify, since `(-x)^(1/3)` depends on
    `x`. And only when both parts come out plainly written: the seventh root of -1
    would be `COS(pi/7) + #i*SIN(pi/7)`, polar form in disguise, and the fifth root
    needs nested radicals, so both are left alone.
    """
    return expression.replace(_is_complex_power, _in_rectangular, simultaneous=False)


def _is_complex_power(expression: sp.Basic) -> bool:
    return (
        isinstance(expression, sp.Pow)
        and bool(expression.base.is_number)
        and bool(expression.exp.is_Rational)
        and expression.is_real is False
    )


def _in_rectangular(expression: sp.Basic) -> sp.Basic:
    parts = sp.expand_complex(expression)
    if parts == expression or not _plainly_written(parts):
        return expression
    return parts


def _plainly_written(value: sp.Basic) -> bool:
    """Whether `value` uses only arithmetic and roots of rationals."""
    return not value.atoms(sp.Function) and all(
        bool(power.base.is_Rational)
        for power in value.atoms(sp.Pow)
        if not power.exp.is_Integer
    )


def _parts_of_powers(expression: sp.Basic) -> sp.Basic:
    """Evaluate `RE(a^z)` and `IM(a^z)` for positive `a` and `z` not known to be real.

    Sympy splits an exponential (`IM(#e^(#i*y))` is `SIN(y)`) but not a power, giving
    `IM(2^(x + #i*y))` as `2^x*IM(2^(#i*y))`. So under `RE` and `IM`, a positive-base
    power is rewritten as `#e^(z*LN(a))`, which sympy then splits into
    `2^x*SIN(y*LN(2))`. The spelling rules turn real exponentials back into powers.
    """
    return expression.replace(
        lambda e: isinstance(e, (sp.re, sp.im)) and _has_complex_power(e.args[0]),
        lambda e: type(e)(e.args[0].replace(_is_power_of_positive, _as_exponential)),
        simultaneous=False,
    )


def _has_complex_power(expression: sp.Basic) -> bool:
    return any(_is_power_of_positive(power) for power in expression.atoms(sp.Pow))


def _is_power_of_positive(expression: sp.Basic) -> bool:
    return (
        isinstance(expression, sp.Pow)
        and expression.base.is_positive is True
        and expression.exp.is_extended_real is not True
    )


def _as_exponential(power: sp.Pow) -> sp.Basic:
    return sp.exp(sp.expand(power.exp * sp.log(power.base)))


def _imaginary_phases(expression: sp.Basic) -> sp.Basic:
    """Rewrite `PHASE(#i*u)` as `pi*SIGN(u)/2` for real `u`.

    At zero, where `PHASE` is undefined, this gives 0.
    """
    return expression.replace(
        lambda e: _imaginary_phase(e) is not None,
        lambda e: _imaginary_phase(e),
        simultaneous=False,
    )


def _imaginary_phase(expression: sp.Basic) -> sp.Basic | None:
    if not isinstance(expression, sp.arg):
        return None
    coefficient, rest = expression.args[0].as_coeff_Mul()
    imaginary = [factor for factor in sp.Mul.make_args(rest) if factor is sp.I]
    if len(imaginary) != 1:
        return None
    real = coefficient * rest / sp.I
    if not _real_where_defined(real):
        return None
    return sp.pi * sp.sign(real) / 2


def _arcs_off_the_line(expression: sp.Basic) -> sp.Basic:
    """Write the arcsine or arccosine of a real number beyond [-1, 1] in rectangular
    form.

    The values are complex, so like `ACOSH(1/2)` (`#i*pi/3`) and `ATANH(2)`
    (`LN(3)/2 - #i*pi/2`), `ASIN(2)` becomes `pi/2 - #i*LN(SQRT(3) + 2)`. Principal
    values follow `-#i*LN(#i*z + SQRT(1 - z^2))` and `pi/2 - ASIN(z)`.
    """
    return expression.replace(_is_arc_off_the_line, _arc_rectangle, simultaneous=False)


def _is_arc_off_the_line(expression: sp.Basic) -> bool:
    if not isinstance(expression, (sp.asin, sp.acos)):
        return False
    argument = expression.args[0]
    return bool(
        argument.is_number
        and argument.is_extended_real
        and (sp.Abs(argument) - 1).is_positive
    )


def _arc_rectangle(expression: sp.asin | sp.acos) -> sp.Basic:
    argument = expression.args[0]
    size = sp.Abs(argument)
    logarithm = sp.log(size + sp.sqrt(size**2 - 1))
    if isinstance(expression, sp.asin):
        return sp.sign(argument) * (sp.pi / 2 - sp.I * logarithm)
    if argument.is_positive:
        return sp.I * logarithm
    return sp.pi - sp.I * logarithm


def _complex_constants(expression: sp.Basic) -> sp.Basic:
    """Write numeric powers with complex exponents as their principal values.

    `b^z` is `#e^(z*LN(b))` with the principal logarithm, so `#i^#i` is `#e^(-pi/2)`,
    `(-1)^#i` is `#e^(-pi)` and `(2*#i)^#i` is `#e^(#i*LN(2) - pi/2)`. A positive base
    keeps its shorter power form: `2^#i` stands. An exponential whose imaginary part is a
    rational multiple of pi becomes rectangular when cosine and sine are plainly written,
    as in `_rectangular`: `#e^(#i*pi/3)` is `1/2 + SQRT(3)*#i/2`, while `#e^(#i*pi/5)`
    stands. Sine and cosine of complex numbers are split likewise: `COS(pi/3 + #i)` is
    `COS(pi/3)*COSH(1) - #i*SIN(pi/3)*SINH(1)`.

    This works bottom up, so enclosing expressions are rebuilt around the value:
    `LN(#i^#i)` is `-pi/2`.
    """
    return expression.replace(_is_complex_constant, _principal_value, simultaneous=False)


def _is_complex_constant(expression: sp.Basic) -> bool:
    if isinstance(expression, sp.Pow):
        return bool(
            expression.base.is_number
            and expression.exp.is_number
            and expression.exp.is_real is False
        )
    if isinstance(expression, (sp.exp, sp.sin, sp.cos)):
        argument = expression.args[0]
        return bool(argument.is_number and argument.is_real is False)
    return False


def _principal_value(expression: sp.Basic) -> sp.Basic:
    if isinstance(expression, sp.Pow):
        base, exponent = expression.args
        if base.is_positive:
            return expression
        real, imaginary = sp.log(base).as_real_imag()
        logarithm = real + sp.I * imaginary
        return _rectangular_exponential(sp.exp(sp.expand(exponent * logarithm)))
    if isinstance(expression, sp.exp):
        return _rectangular_exponential(expression)
    real, imaginary = expression.args[0].as_real_imag()
    if real == 0:
        return expression
    angle = _plain_angle(real)
    if angle is None:
        return expression
    cosine, sine = angle
    if isinstance(expression, sp.cos):
        return cosine * sp.cosh(imaginary) - sp.I * sine * sp.sinh(imaginary)
    return sine * sp.cosh(imaginary) + sp.I * cosine * sp.sinh(imaginary)


def _rectangular_exponential(power: sp.Basic) -> sp.Basic:
    """`#e^(a + b*#i)` as `#e^a*(COS(b) + #i*SIN(b))`, if those are plainly written."""
    if not isinstance(power, sp.exp):
        return power
    real, imaginary = power.args[0].as_real_imag()
    angle = _plain_angle(imaginary)
    if angle is None:
        return power
    cosine, sine = angle
    return sp.exp(real) * (cosine + sp.I * sine)


def _plain_angle(angle: sp.Basic) -> tuple[sp.Basic, sp.Basic] | None:
    """The cosine and sine of a rational multiple of pi if both are plainly written,
    else None."""
    if not (angle / sp.pi).is_Rational:
        return None
    cosine, sine = sp.cos(angle), sp.sin(angle)
    if not _plainly_written(cosine + sine):
        return None
    return cosine, sine


# -- the kernel layer --------------------------------------------------------
#
# Every rewrite above is a rule about kernels: the trigonometric, logarithmic,
# exponential, sign, arc and radical calls an expression is a rational function of.
# Three tables, applied by one driver, decide where each is used: identities tried in
# every mode and kept if they pay; directions requested by mode settings; and spelling
# conventions applied after the normal form. The output vocabulary is the one spelling
# applied first, because the identities are stated in it: `TANH(z)` becomes
# `(#e^(2*z) - 1)/(#e^(2*z) + 1)` only if written as exponentials before denominators
# are cleared.


class _Rule(NamedTuple):
    """One kernel-layer rewrite, with its name and the reason for it.

    `kept` decides whether to take the result: `_gated` keeps it if it pays, `forced`
    always. `yields_to` names a mode setting for which this rule is the Auto behavior,
    so it is skipped when that setting isn't Auto: `1/COS(x)^2` becomes `TAN(x)^2 + 1`
    only while Trigpower is Auto, not under Cosines.
    """

    name: str
    rewrite: Rewrite
    kept: Callable[[sp.Basic, Rewrite], sp.Basic]
    reason: str
    yields_to: str | None = None


def _rewritten(expression: sp.Basic, context: Context) -> sp.Basic:
    """Reduce the kernels: apply every identity that pays, then every requested
    direction.

    Works on the vocabulary `_by_definition` produces, and runs before the normal form,
    which tidies up the sums the rules leave.
    """
    if expression.has(sp.MatrixExpr):
        return forced(expression, _distributed)
    # Before the definitions, which turn its hyperbolic leftovers into exponentials.
    expression = forced(expression, _complex_constants)
    expression = forced(expression, _symmetries)
    expression = _by_definition(expression)
    expression = _applied(expression, _IDENTITIES, context)
    for setting, directions in _DIRECTED.items():
        rules = directions.get(getattr(context, setting), ())
        expression = _applied(expression, rules, context)
    # Not in the tables, whose rules don't see the order list. Following the original,
    # the exponential stays under Collect.
    if context.exponential != Direction.COLLECT:
        expression = forced(
            expression, lambda e: _powers_of_logarithms(e, context.order)
        )
    return expression


#: Heads that remain in an answer as unevaluated integrals, sums and products.
_STANDING = (sp.Integral, sp.Sum, sp.Product)


def _rewritten_beside_standing(expression: sp.Basic, context: Context) -> sp.Basic:
    """`_rewritten`, but rewriting unevaluated integrals, sums and products in a sum
    separately from the other terms.

    An unevaluated integral is a separate term, as in `INT(x/SIN(a*x), x)` plus an
    antiderivative. Rewritten together, the sum may be put over a common denominator,
    pulling the integral under it and printing it twice.
    """
    if not isinstance(expression, sp.Add):
        return _rewritten(expression, context)
    standing = [term for term in expression.args if term.has(*_STANDING)]
    if not standing or len(standing) == len(expression.args):
        return _rewritten(expression, context)
    rest = sp.Add(*(term for term in expression.args if not term.has(*_STANDING)))
    return _rewritten(rest, context) + sp.Add(
        *(_rewritten(term, context) for term in standing)
    )


def _spelled(expression: sp.Basic, context: Context) -> sp.Basic:
    """Apply Derive's spelling conventions after the normal form.

    Running last means nothing rewrites them afterwards. Each is the form Derive prints,
    shorter or not: `1/COS(x)^2` is `TAN(x)^2 + 1`, and `LN(256)` is `8*LN(2)`.
    """
    if expression.has(sp.MatrixExpr):
        return expression
    return _applied(expression, _SPELLINGS, context)


def _applied(
    expression: sp.Basic, rules: tuple[_Rule, ...], context: Context
) -> sp.Basic:
    for rule in rules:
        if rule.yields_to is not None and getattr(context, rule.yields_to) != "Auto":
            continue
        expression = rule.kept(expression, rule.rewrite)
    return expression


def _combinatorial(expression: sp.Basic) -> sp.Basic:
    """`combsimp`, if the expression is combinatorial and small enough."""
    if expression.has(*_COMBINATORIAL) and sp.count_ops(expression) <= _COMBSIMP:
        return sp.combsimp(expression)
    return expression


def _on_powers(rewrite: Rewrite) -> Rewrite:
    """`rewrite`, but only if it changed some power.

    The power rules rebuild every product, and sympy distributes numbers over sums on
    construction: `powsimp` turns `3*x^2*(2*y + 1)` into `x^2*(6*y + 3)` without
    collecting anything. Derive keeps the original, so if the powers are unchanged, so
    is the expression.
    """

    def run(expression: sp.Basic) -> sp.Basic:
        candidate = rewrite(expression)
        if candidate.atoms(sp.Pow, sp.exp) == expression.atoms(sp.Pow, sp.exp):
            return expression
        return candidate

    return run


#: Identities tried in every mode. Gated ones are kept when `pays` says the result is
#: simpler; forced ones always apply, since gating would keep a side that is merely no
#: longer.
_IDENTITIES = (
    _Rule("arcs off the line", _arcs_off_the_line, forced,
          "ASIN(2) is pi/2 - #i*LN(SQRT(3) + 2): a complex value in rectangular form"),
    _Rule("parts of powers", _parts_of_powers, forced,
          "IM(2^(x + #i*y)) is 2^x*SIN(y*LN(2)): a power of a positive base splits"),
    _Rule("imaginary phases", _imaginary_phases, forced,
          "PHASE(#i*x) is pi*SIGN(x)/2 for a real x"),
    _Rule("signs", folded_signs, forced,
          "ABS(u)*SIGN(u) is u and u/ABS(u) is SIGN(u), for a real u"),
    _Rule("signs of reciprocals", _signs_of_reciprocals, forced,
          "SIGN(1/u) is SIGN(u), and SIGN(1/ABS(u)) is 1 wherever it is defined"),
    _Rule("paired signs", _paired_signs, forced,
          "ABS(x*ATAN(x)) is x*ATAN(x): factors whose signs cancel need no bars"),
    _Rule("conjugate pairs", _conjugate_pairs, forced,
          "CONJ(z)*z is ABS(z)^2"),
    _Rule("combinatorial", _combinatorial, _gated,
          "a ratio of factorials is a binomial or a polynomial when it is shorter"),
    # The arc and angle rules are our own: sympy doesn't collect these arcs, shorten
    # their arguments, or have rules spanning two angles.
    _Rule("simpler arcs", _simpler_arcs, _gated,
          "ASIN(u/SQRT(u^2 + 1)) is ATAN(u): an arc about the shorter argument"),
    _Rule("complementary arcs", _complementary_arcs, forced,
          "ASIN(u) + ACOS(u) is pi/2 and ATAN(u) + ATAN(1/u) is pi*SIGN(u)/2"),
    _Rule("arctangent sums", _arc_tangent_sums, forced,
          "ATAN(2) + ATAN(3) is 3*pi/4: arctangents of numbers added up"),
    _Rule("arc multiples", _arc_multiples, forced,
          "SIN(2*ACOS(x)) is 2*x*SQRT(1 - x^2): a call of a multiple of an arc"),
    _Rule("logarithmic powers", _logarithmic_powers, forced,
          "3^LN(2) is 2^LN(3): a power to a logarithm written over the smaller base"),
    _Rule("half angles", _half_angles, _gated,
          "calls of fractional multiples of one angle cancel written about it"),
    _Rule("denested", _denested, _unnested,
          "SQRT(5 + SQRT(24)) is SQRT(2) + SQRT(3): each radical denested"),
    _Rule("flattened", _flattened, _gated,
          "a constant over nested radicals reduced in the field its surds span"),
    _Rule("trigonometric constants", _trigonometric_constants, _gated,
          "COS(pi/9)*COS(2*pi/9)*COS(4*pi/9) is 1/8: values of rational angles"),
    _Rule("one denominator", sp.radsimp, _gated,
          "a denominator under a denominator is cleared, #e^(-z) put over #e^z"),
)

#: Directions, keyed by mode setting and value. Auto rules mean "only where it
#: cancels" and are gated; Collect and Expand are instructions and forced. No rule
#: ignores domains: the logarithm rules are sympy's without `force`, run under
#: `_nonnegatively`, and the power rules keep sympy's validity check for
#: `(#e^z)^k -> #e^(k*z)`. Only the direction is chosen here.
_DIRECTED: dict[str, dict[str, tuple[_Rule, ...]]] = {
    # Principal is sympy's convention; Real makes `(-8)^(1/3)` -2, and Any drops the
    # domain checks from the power rules, so `SQRT(x^2)` is `x`.
    "branch": {
        Branch.REAL: (
            _Rule("real roots", _real_roots, forced,
                  "an odd root of a negative number is its real root"),
        ),
        Branch.ANY: (
            _Rule("any powdenest", lambda e: sp.powdenest(e, force=True), forced,
                  "a power of a power is one power, whatever the base"),
            _Rule("any powsimp", lambda e: sp.powsimp(e, force=True), forced,
                  "powers over one exponent are one power, whatever the bases"),
        ),
    },
    "trigonometry": {
        Direction.AUTO: (
            _Rule("trigsimp", sp.trigsimp, _gated,
                  "trigonometric identities, where they cancel something"),
            _Rule("multiple angles", _multiple_angles, _cancelled,
                  "SIN(3*x)/SIN(x) is 2*COS(2*x) + 1: multiples of one angle reduced"),
        ),
        Direction.COLLECT: (
            _Rule("phase angles", _phase_angles, forced,
                  "a*SIN(z) + b*COS(z) is one sine, which fu's TR10i has only for a = b"),
            _Rule("TR8", TR8, forced, "products of calls written as sums"),
            _Rule("TR7", TR7, forced, "powers of cosines reduced to multiple angles"),
        ),
        Direction.EXPAND: (
            _Rule("expand_trig", sp.expand_trig, forced,
                  "calls of sums and multiples written over single angles"),
            _Rule("special angles", lambda e: e.rewrite(sp.sqrt), _gated,
                  "the algebraic values of the special angles that come with it"),
        ),
    },
    "trigpower": {
        TrigPower.SINES: (
            _Rule("TR6", TR6, forced, "even powers of cosines written over sines"),
        ),
        TrigPower.COSINES: (
            _Rule("TR5", TR5, forced, "even powers of sines written over cosines"),
        ),
    },
    # Auto is the mixed direction of 6.2: the sum rule combines here, so
    # `3*LN(2) + LN(5)` is `LN(40)`, and the power rule extracts in `_SPELLINGS`, so
    # `LN(256)` is `8*LN(2)`, like `SQRT(12)` is `2*SQRT(3)`. `LN(x^2 - x) - LN(x)`
    # combines only if `x` is declared to exclude values where that is false.
    "logarithm": {
        Direction.AUTO: (
            _Rule("logcombine", _nonnegatively(sp.logcombine), _gated,
                  "logarithms added into one, where it is shorter"),
        ),
        Direction.COLLECT: (
            _Rule("logcombine", _nonnegatively(sp.logcombine), forced,
                  "logarithms added into one"),
        ),
        Direction.EXPAND: (
            _Rule("expand_log", _nonnegatively(sp.expand_log), forced,
                  "a logarithm of a product taken apart"),
            _Rule("extracted powers", _extracted_powers, forced,
                  "LN(x^2) is 2*LN(ABS(x)): a logarithm of a power taken apart"),
        ),
    },
    "exponential": {
        Direction.AUTO: (
            _Rule("powsimp", _on_powers(sp.powsimp), _gated,
                  "powers collected, where it is shorter"),
            _Rule("expand_power_exp", _on_powers(sp.expand_power_exp), _gated,
                  "a power of a sum split, where it is shorter"),
            _Rule("expand_power_base", _on_powers(sp.expand_power_base), _gated,
                  "a power of a product split, where it is shorter"),
        ),
        Direction.COLLECT: (
            _Rule("powsimp", _on_powers(sp.powsimp), forced, "powers collected"),
        ),
        Direction.EXPAND: (
            _Rule("expand_power_exp", _on_powers(sp.expand_power_exp), forced,
                  "a power of a sum split"),
            _Rule("expand_power_base", _on_powers(sp.expand_power_base), forced,
                  "a power of a product split"),
        ),
    },
}

#: Spelling conventions, applied after the normal form. None is gated: several make the
#: expression longer, but they are how Derive writes the answer.
_SPELLINGS = (
    # After the branch rules, which decide a root's value: under Real, `(-8)^(1/3)` is
    # already `-2`.
    _Rule("rectangular", _rectangular, forced,
          "a numeric complex power is real part plus imaginary part"),
    _Rule("extracted powers", _extracted_powers, forced,
          "LN(b^k) is k*LN(b) where 6.2 allows it", yields_to="logarithm"),
    _Rule("collected powers", _collected_powers, forced,
          "powers of one base are one power: n*x^n/x is n*x^(n - 1)",
          yields_to="exponential"),
)
