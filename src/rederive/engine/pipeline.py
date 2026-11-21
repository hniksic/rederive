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
from contextvars import ContextVar
from dataclasses import replace
from functools import reduce
from itertools import combinations
from math import gcd, lcm, prod
from typing import NamedTuple

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.core.mul import _keep_coeff
from sympy.core.relational import Relational
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.logic.boolalg import Boolean, BooleanAtom
from sympy.simplify.fu import TR2, TR5, TR6, TR7, TR8, TR11

from rederive.engine.approximation import (
    approximated,
    drew_fractions,
    noting_approximations,
)
from rederive.engine.boundary import DEFAULT_AMOUNT, Amount, Result
from rederive.engine.context import (
    Branch,
    Context,
    Direction,
    DomainKind,
    FileNeeded,
    Notation,
    Precision,
    TrigPower,
)
from rederive.engine.evaluate import (
    Commanded,
    Environment,
    Suspended,
    branched,
    cancelled_over_roots,
    evaluate,
    exact_closed_form,
    exact_number,
    limit_by_substitution,
    moving_integrals,
    named_as_declared,
    released,
    released_held,
)
from rederive.engine.expanding import EXPAND_AMOUNTS, expanded_expression
from rederive.engine.factoring import amount_named, factored_expression
from rederive.engine.fresh import afresh
from rederive.engine.matrix_factoring import MATRIX_AMOUNTS, matrix_product
from rederive.engine.from_sympy import from_sympy
from rederive.engine.integration import agrees, integral, numerically, proved_zero
from rederive.engine.intervals import (
    always_positive,
    covering,
    decided,
    never_negative,
    settled,
)
from rederive.engine.normal import normal_form, over_one_denominator
from rederive.engine.ordering import main_order
from rederive.engine.printer import is_written, shadowing_built_ins
from rederive.engine import budget
from rederive.engine.budget import within
from rederive.engine.rewriting import Rewrite, attempt, forced, larger_than, pays
from rederive.engine.sets import SetOf, equal_sets, is_elided, is_set
from rederive.engine.signs import folded_signs, polar_logarithms
from rederive.engine.solution_forms import nsolve_head, solutions_head, solve_head
from rederive.engine.solving import minting, oriented, solutions
from rederive.engine.to_sympy import (
    CALCULUS_HEADS,
    DEGREE,
    Antidifference,
    Approx,
    Assign,
    annuity,
    Declare,
    FunDef,
    InOrder,
    Logical,
    MAPPED_OVER_EQUATIONS,
    MovingLimit,
    PlusMinus,
    StringLiteral,
    Subscript,
    Truth,
    ambiguous_products,
    Vector,
    as_condition,
    authored_conditionals,
    connected,
    digit_count,
    is_statement,
    floors_as_residues,
    is_simple,
    simple_constants,
    residues_as_floors,
)
from rederive.engine.vocabulary import INVERSE_HYPERBOLIC, in_vocabulary
from rederive.model.expr import Kind, Node
from rederive.syntax import DeriveSyntaxError, parse_expression, write_expression
from rederive.syntax.names import SETTING_VALUES
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


@afresh
@noting_approximations
@shadowing_built_ins
def simplify(
    node: Node,
    context: Context | None = None,
    state: ParseState | None = None,
    top: bool | None = None,
) -> Result:
    """Derive's Simplify: `node` in its sufficiently simple form.

    Works on any subtree, so the session can simplify a highlighted subexpression and
    splice the answer back. `state` is the symbol table `node` was read with, which
    spells each of its variables one way (see `named_as_declared`); a session must pass
    its own.

    The author's conditionals are collected up front and passed to the printer, so an
    undecidable one is shown as written (see `authored_conditionals`).

    `top` says whether an equation that is the answer is a top-level one, kept with its
    sides simplified rather than decided (see `top_level_equation`); None reads it from
    `node`.
    """
    context = (context or Context()).fresh()
    node = named_as_declared(node, state)
    whole = _whole_approximation(node, context)
    if whole is not None:
        value, digits = whole
        return approx(value, context, digits, state, top=False)
    assigned = _assigned_approximation(node, context)
    if assigned is not None:
        value, digits = assigned
        return approx(value, context, digits, state, top=False)
    if top is None:
        top = top_level_equation(node, context)
    if context.precision is Precision.APPROXIMATE and exact_closed_form(node, context):
        # A number simplified exactly, then approximated: rounded once, it keeps the
        # digits that cancel, which rounding its parts would lose (see
        # `exact_closed_form`).
        exact = context.with_precision(Precision.EXACT)
        with minting(exact):
            value = simplified(node, exact, top=top)
        if exact_number(value):
            node = from_sympy(value, exact).node
    if _numerically_solved(node) and context.precision is not Precision.APPROXIMATE:
        return approx(node, context, None, state, top=top)
    with minting(context):
        try:
            value = simplified(node, context, top=top)
        except FileNeeded as needed:
            return _needing(node, state, needed.name)
        # The answer is written under the settings in force when the command ends,
        # which a program may have changed.
        written = context.store.context or context
        if drew_fractions() and written.precision is Precision.EXACT:
            # A random number in [0, 1) is approximate, as Derive's is: what is
            # computed from it is shown in decimals, `-LN(RANDOM(1))/2` rather than
            # the logarithm of a 64-bit fraction, while integers drawn or computed
            # exactly from it (`FLOOR(10^6*RANDOM(1))`) stay integers.
            written = replace(
                written.with_precision(Precision.MIXED), notation=Notation.DECIMAL
            )
            value = approximated(value, written)
        return from_sympy(
            value,
            written,
            authored_conditionals(
                node,
                context,
                lambda test: _transform(test, context, True),
                context.store.retested,
            ),
        )


def _needing(node: Node, state: ParseState | None, name: str) -> Result:
    """The answer of a command that ended at a `LOAD` of file `name`, for the session
    to look for (see `FileNeeded`): to a caller with no files to offer, the expression
    as it stands."""
    text = write_expression(node)
    try:
        node = parse_expression(text, state or ParseState()).node
    except DeriveSyntaxError:
        pass
    return Result(node, text, needs=name)


def approx(
    node: Node,
    context: Context | None = None,
    digits: int | None = None,
    state: ParseState | None = None,
    top: bool | None = None,
) -> Result:
    """Simplify with the precision mode set to Approximate.

    This is the same pipeline as `simplify` at a different precision, as the manual
    describes approX. It is also what an `APPROX` call simplifies to.

    """
    context = context or Context()
    approximate = context.with_precision(Precision.APPROXIMATE, digits)
    return simplify(node, approximate, state, top)


def top_level_equation(node: Node, context: Context) -> bool:
    """Whether an equation `node` comes to is a top-level one, which Simplify keeps
    with its sides simplified, so that it can still be worked on by hand: `2 = 2` stays
    `2 = 2`, where `[2 = 2]` is `[true]`.

    An equation is top-level when it is the whole expression, the result of arithmetic
    on equations or of a function applied to both sides of one (`(x = x + 1) - x` is
    `0 = 1`), the value of a variable or entry, or the right side of an assignment or
    definition. An equation anywhere else is decided where it can be: an element, an
    argument of any other function, a user function's result (`SUBST(a*x + b = 0, x,
    -b/a)` is `true`), an operand of a logical operator, an `IF` test.
    """
    match node.kind:
        case Kind.REL:
            return node.value == "="
        case Kind.NAME:
            return True
        case Kind.LABEL:
            labelled = (
                context.labels.get(int(node.value))
                if str(node.value).isdigit()
                else None
            )
            return labelled is None or top_level_equation(labelled, context)
        case Kind.SUM | Kind.PRODUCT | Kind.BINOP | Kind.UNOP:
            return any(top_level_equation(child, context) for child in node.children)
        case Kind.CALL:
            name, *arguments = node.children
            return (
                str(name.value) in _MAPPING_CALLS
                and bool(arguments)
                and top_level_equation(arguments[0], context)
            )
        case Kind.ASSIGN:
            return (
                node.value == ":="
                and len(node.children) == 2
                and node.children[0].kind is Kind.NAME
                and top_level_equation(node.children[1], context)
            )
        case Kind.FUNDEF:
            return len(node.children) == 2 and top_level_equation(
                node.children[1], context
            )
    return False


#: The calls whose value for an equation argument is the equation of their values for
#: its sides.
_MAPPING_CALLS = MAPPED_OVER_EQUATIONS | {"DIF", "INT"}


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


def _assigned_approximation(
    node: Node, context: Context
) -> tuple[Node, int | None] | None:
    """An assignment whose right side is an `APPROX` call, as `b := APPROX(a, 6)`, with
    the call replaced by its operand, and the digit count: it simplifies as approX of
    the assignment, so that the value is shown in digits as the call alone would be."""
    if node.kind is not Kind.ASSIGN or node.value != ":=" or len(node.children) != 2:
        return None
    target, value = node.children
    if target.kind is not Kind.NAME:
        return None
    whole = _whole_approximation(value, context)
    if whole is None:
        return None
    operand, digits = whole
    return replace(node, children=(target, operand)), digits


def _numerically_solved(node: Node) -> bool:
    """Whether the expression calls `NSOLVE` or `NSOLUTIONS`, whose numbers are shown as
    approximations are, whatever the precision: such an entry is simplified as approX
    would. A definition is left alone: its body is computed where it is used."""
    if node.kind in (Kind.ASSIGN, Kind.FUNDEF):
        return False
    if node.kind is Kind.CALL and node.children[0].value in ("NSOLVE", "NSOLUTIONS"):
        return True
    return any(_numerically_solved(child) for child in node.children)


def simplified(
    node: Node, context: Context | None = None, decide: bool = True, top: bool = False
) -> sp.Basic:
    """The simplified expression as sympy, before conversion back to a node.

    `simplify` is this plus `from_sympy`; commands that keep computing can use this to
    avoid printing and reparsing.

    `decide` makes a relation that holds come back as `true`. soLve turns it off: it
    only prepares the equation, and `x = x` decided to `true` would leave nothing to
    solve, where the answer should be `x = @1`.

    `top` keeps an equation that is the answer, or the value of an assignment that is
    the answer, as an equation with its sides simplified (see `top_level_equation`).

    An undecidable conditional stays suspended through all steps, keeping its branches
    as written, and is released as a conditional only at the end. Terms that make up a
    `MOD`, `u - v*FLOOR(u/v)`, are collected into one.
    """
    context = context or Context()
    environment = Environment(
        context,
        decide=lambda test: _decide_simplified(test, context),
        work_out=lambda head: _evaluate(head, context),
        command=_commanded,
        simplify=_argument,
        running=True,
        top=node if top else None,
    )
    token = _simplified_already.set({})
    try:
        expression = evaluate(node, environment)
        if context.precision is Precision.APPROXIMATE:
            # Approximating computes the value a held expression stands for.
            expression = released_held(expression, environment)
        if _is_content_form(expression):
            return expression.args[0]
        try:
            result = _transform(
                expression, context.store.context or context, decide, top
            )
            result = floors_as_residues(result)
        except Exception:
            # The evaluated input is still a correct answer.
            result = expression
    finally:
        _simplified_already.reset(token)
    return released(result)


def _is_content_form(expression: sp.Basic) -> bool:
    """Whether `expression` is a `FACTOR` answer that is a vector's content times the
    vector of cofactors, `10*[1, 10, 100]`, which is the answer as written: the
    product is not carried out."""
    if not isinstance(expression, Commanded):
        return False
    form = expression.args[0]
    return isinstance(form, sp.Mul) and any(
        isinstance(factor, Vector) for factor in form.args
    )


#: The expressions `_expression` has simplified during one `simplified` call, keyed by
#: the expression, `decide` and whether it is an argument, with the context it was
#: simplified in. ODE.MTH's tests ask about the same long parameters again and again:
#: `F21_SOL` tests `MOD` of each of three in two conditions.
_simplified_already: ContextVar[dict | None] = ContextVar(
    "_simplified_already", default=None
)


def _argument(value: sp.Basic, context: Context, top: bool = False) -> sp.Basic:
    """`value` simplified as the argument of a built-in that reads its form, as
    Derive simplifies it first: `VARIABLES((a^2*y^2 + a^2*y*z)/(y^2 + y*z))` is `[a]`.

    A calculus head still standing in `value` was tried when it was built, and stays
    as it is: it is held out while the rest is simplified, since simplifying would try
    each integral again and prove each answer by differentiating it back. ODE.MTH
    passes large equations with integrals in them to `FREE`, whose `VARIABLES` call
    took minutes over them.

    `top` keeps an equation `value` is with its sides simplified, as the value of a
    top-level assignment (see `top_level_equation`).
    """
    try:
        heads = value.atoms(*_HELD)
    except Exception:
        heads = set()
    held = {head: sp.Dummy("held") for head in sp.ordered(heads)}
    if held:
        value = value.xreplace(held)
    token = _in_argument.set(True)
    try:
        simplified = _transform(value, context, True, top)
    finally:
        _in_argument.reset(token)
    if held:
        simplified = simplified.xreplace({dummy: head for head, dummy in held.items()})
    return simplified


def _degrees_beside_pi(expression: sp.Basic) -> sp.Basic:
    """`expression` with `deg` written as `pi/180` in every sum that also holds `pi`,
    so that the two kinds of angle meet: `90*deg - pi/2` is 0. An angle in degrees
    elsewhere stays one, `45*deg + x`."""
    if not isinstance(expression, sp.Basic) or not expression.has(DEGREE):
        return expression
    radians = {DEGREE: sp.pi / 180}
    return expression.replace(
        lambda e: e.is_Add and e.has(DEGREE) and e.xreplace({DEGREE: 1}).has(sp.pi),
        lambda e: e.xreplace(radians),
    )


#: Whether an argument is being simplified by `_argument`.
_in_argument: ContextVar[bool] = ContextVar("_in_argument", default=False)

#: Calculus heads `_argument` leaves standing.
_HELD = (sp.Integral, sp.Sum, sp.Product, sp.Limit, sp.Derivative, MovingLimit)


# -- by shape ----------------------------------------------------------------


def _transform(
    expression: sp.Basic, context: Context, decide: bool, top: bool = False
) -> sp.Basic:
    """Simplify one expression according to what kind of thing it is.

    `top` says that an equation `expression` is, or that an assignment assigns, is a
    top-level one, which keeps its form (see `top_level_equation`).
    """
    expression = _degrees_beside_pi(expression)
    if is_simple(expression):
        # Plain data: numbers, strings, truth values, variables, and vectors and sets of
        # them. No step of Simplify changes it; only Approximate rounds its numbers,
        # and Mixed its `pi` and `#e`, which the steps for Mixed approximate.
        if context.precision is Precision.APPROXIMATE:
            return approximated(expression, context)
        if context.precision is not Precision.MIXED or not (
            simple_constants(expression) & _MIXED_APPROXIMATES
        ):
            return expression
    return _transformed(expression, context, decide, top)


#: The constants `is_simple` takes as simple that Mixed precision approximates.
_MIXED_APPROXIMATES = frozenset({sp.pi, sp.E})


def _transformed(
    expression: sp.Basic, context: Context, decide: bool, top: bool = False
) -> sp.Basic:
    """`_transform` of anything but plain data."""
    if isinstance(expression, Relational):
        return _relation(expression, context, decide, top)
    if isinstance(expression, Declare):
        return expression
    if isinstance(expression, (Assign, FunDef)):
        return _definition(expression, context, decide, top)
    if isinstance(expression, Logical):
        return _logical(expression, context, decide)
    if isinstance(expression, sp.Symbol):
        return expression
    if isinstance(expression, Vector) or is_set(expression) or is_elided(expression):
        # A set or ellipsis is rebuilt, so its elements are merged or counted anew.
        return expression.func(
            *(_transform(a, context, decide) for a in expression.args)
        )
    if isinstance(expression, Boolean):
        return _boolean(expression, context, decide)
    if isinstance(expression, sp.Basic) and expression.has(Truth, AppliedUndef):
        expression = _truths_decided(expression, context, decide)
    return _expression(expression, context, decide)


def _truths_decided(expression: sp.Basic, context: Context, decide: bool) -> sp.Basic:
    """`expression` with each relation or logical expression used in its arithmetic, or
    as an argument of a call that stands, simplified, so decided where it can be: `(2 <
    3) - 2` is `true - 2` and `MOD(0 > 4, 0)` is `MOD(false, 0)`. The truth value stays
    in the arithmetic or the call as it is."""

    def statement_argument(e: sp.Basic) -> bool:
        return isinstance(e, AppliedUndef) and any(is_statement(a) for a in e.args)

    def arguments_decided(e: sp.Basic) -> sp.Basic:
        return e.func(
            *(_transform(a, context, decide) if is_statement(a) else a for a in e.args)
        )

    return expression.replace(
        lambda e: isinstance(e, Truth) or statement_argument(e),
        lambda e: (
            Truth(_transform(e.args[0], context, decide))
            if isinstance(e, Truth)
            else arguments_decided(e)
        ),
    )


def _logical(expression: Logical, context: Context, decide: bool) -> sp.Basic:
    """An operator sympy declined, with its operands simplified and joined again.

    An operand may be decided only once simplified: at `c = 3/2`, `MOD(c) AND c >= 2`
    has the comparison false, which decides the conjunction, so `MOD(c - 1/2) = 0 OR
    MOD(c) AND c >= 2` is true there, as in Derive. Operands that are still variables,
    as in a truth table's headings, leave the operator as written.
    """
    word, *operands = expression.args
    simplified = [_transform(operand, context, decide) for operand in operands]
    if any(isinstance(operand, Commanded) for operand in operands) and all(
        isinstance(operand, Boolean) and not isinstance(operand, sp.Expr)
        for operand in simplified
    ):
        # A solution combined with other statements is simplified with them, as the
        # Boolean rules simplify any conjunction: `SOLVE(x^2 = 4, x) AND x > 0` is
        # `x = 2`.
        joined = connected(word.name, simplified)
        if not isinstance(joined, Logical):
            return _transform(joined, context, decide)
    if any(isinstance(operand, BooleanAtom) for operand in simplified) or all(
        isinstance(operand, sp.Integer) for operand in simplified
    ):
        joined = connected(word.name, simplified)
        if not isinstance(joined, Logical):
            return _transform(joined, context, decide)
    return Logical(word, *simplified)


def _relation(
    expression: Relational, context: Context, decide: bool, top: bool = False
) -> sp.Basic:
    """Simplify both sides, and decide the relation if possible.

    `[2 = 2]` gives `[true]` and `[1 = 2]` gives `[false]`. The decision is three-valued,
    so a relation that settles nothing (`x = 2`, or `ABS(x) < 1` with no declaration) is
    returned as authored. A top-level equation (`top`) is not decided by arithmetic:
    `2 = 2` stays, so that it can still be worked on by hand, and so does one whose
    simplified sides are the same, whatever their kind (`[[-3], [-3]] = [[-3], [-3]]`).
    Otherwise values whose kinds decide an equation (`compared`, `equal_sets`) decide it
    there too, so an equation of two vectors is still the conjunction of the element
    equations.

    An ordering is decided only for real sides, since `x < x + 1` means nothing for
    complex `x`. It is decided like an `IF` test, using declared domains. An equation
    is decided by arithmetic only (`_equated`).

    With `decide` off (soLve's pipeline), the sides are simplified and the relation left
    for solving.

    Two sets are equal when they have the same elements, compared as simplified
    expressions. Two vectors, two strings, and a vector or string and a value of another
    kind are compared as `compared` says.

    For soLve, two vectors of the same length are equated element by element, as section
    8.5 uses to read off a solved system: `[[x], [y], [z]] = A^-1 . b` gives three
    equations (see `_spread`). An ordering of vectors stands: vectors are not ordered.
    """
    left = _transform(expression.lhs, context, decide)
    right = _transform(expression.rhs, context, decide)
    if top and expression.func is sp.Eq and left == right:
        return sp.Eq(left, right, evaluate=False)
    if decide and expression.func in (sp.Eq, sp.Ne) and _declared_vector_string(
        (left, right), context
    ):
        # A string declared a vector is a vector variable: `"IRR" = [10, 20]` stands.
        return expression.func(left, right, evaluate=False)
    if decide and expression.func in (sp.Eq, sp.Ne):
        verdict = equal_sets(left, right)
        if verdict is not None:
            return sp.true if verdict == (expression.func is sp.Eq) else sp.false

        def element(func, one: sp.Basic, other: sp.Basic) -> sp.Basic:
            return _stated(func, one, other, context, decide)

        found = compared(expression.func, left, right, element)
        if found is not None:
            return found
    spread = _spread(expression.func, left, right, context, decide)
    if spread is not None:
        return spread
    if isinstance(left, Vector) or isinstance(right, Vector):
        return expression.func(left, right, evaluate=False)
    keep = top and expression.func is sp.Eq
    return _stated(expression.func, left, right, context, decide and not keep)


def _declared_vector_string(sides: tuple[sp.Basic, ...], context: Context) -> bool:
    """Whether one of `sides` is a string declared a vector variable, beside a vector."""
    if not any(isinstance(side, Vector) for side in sides):
        return False
    return any(
        isinstance(side, StringLiteral)
        and context.domain(f'"{side.name}"').kind is DomainKind.VECTOR
        for side in sides
    )


def compared(
    func,
    left: sp.Basic,
    right: sp.Basic,
    element: Callable[..., sp.Basic],
) -> sp.Basic | None:
    """An equation or inequation of values whose kinds decide it, or None.

    Two vectors are equal when their elements are pairwise: `[a, b] = [c, d]` is `a = c
    AND b = d`, each element equation made by `element` (decided where it can be), so a
    matrix is compared element by element; `/=` gives the disjunction of the element
    inequations. Vectors of different lengths, at any depth, are different. A vector is
    never a number, a string, a set or a truth value. Two strings are equal exactly when
    their texts are, case counting, and a string is never a vector, a set or a truth
    value (see `_another_kind`). Two truth values are equal when they are the same, and
    a statement equals `true` where it holds and `false` where it fails: `(x > 1) =
    false` is `x <= 1`. Two value words of settings are equal exactly when
    they are the same word: `Precision = Exact` is decided. The unknown `?` equals
    itself and no number, string, vector or set. A vector or string beside a variable
    stands, as does `?` beside one.
    """
    if func not in (sp.Eq, sp.Ne):
        return None
    equal = func is sp.Eq

    def verdict(same: bool) -> sp.Basic:
        return sp.true if same == equal else sp.false

    if left is sp.nan or right is sp.nan:
        other = right if left is sp.nan else left
        if other is sp.nan:
            return verdict(True)
        if isinstance(other, (Vector, StringLiteral)) or is_set(other) or (
            isinstance(other, sp.Expr) and other.is_number
        ):
            return verdict(False)
        return None
    if isinstance(left, Vector) and isinstance(right, Vector):
        if left.size() != right.size():
            return verdict(False)
        links = []
        for one, other in zip(left.args, right.args, strict=True):
            link = compared(func, one, other, element)
            if link is not None:
                links.append(link)
            elif isinstance(one, BooleanAtom) and isinstance(other, BooleanAtom):
                links.append(verdict(one is other))
            elif isinstance(one, sp.Expr) and isinstance(other, sp.Expr):
                links.append(element(func, one, other))
            else:
                # A relation is compared with nothing: `[x = 1] = [1]` stands.
                return None
        return sp.And(*links) if equal else sp.Or(*links)
    if isinstance(left, StringLiteral) and isinstance(right, StringLiteral):
        return verdict(left.name == right.name)
    if isinstance(left, BooleanAtom) and isinstance(right, BooleanAtom):
        return verdict(left is right)
    for one, other in ((left, right), (right, left)):
        if isinstance(one, BooleanAtom) and is_statement(other):
            return other if (one is sp.true) == equal else connected("NOT", [other])
    if _is_value_word(left) and _is_value_word(right):
        return verdict(left.name == right.name)
    for one, other in ((left, right), (right, left)):
        if isinstance(one, (Vector, StringLiteral)) and _another_kind(one, other):
            return verdict(False)
    return None


def _is_value_word(value: sp.Basic) -> bool:
    """Whether `value` is a value word of a setting, such as `Exact`: a reserved name
    that stands for itself, so that two are equal exactly when they are the same."""
    return type(value) is sp.Symbol and value.name in SETTING_VALUES


def _another_kind(value: sp.Basic, other: sp.Basic) -> bool:
    """Whether `other` is a value of another kind than the vector or string `value`: a
    truth value, a set, a string or vector unlike it, or for a vector a number. A
    string beside a number is an equation in a string variable, as strings are in
    arithmetic: `APPROX(SUBST("dz" = x*"dx", [x, "dx"], [2, 0.004]))` is `"dz" =
    0.008`."""
    if isinstance(other, BooleanAtom) or isinstance(other, SetOf):
        return True
    if isinstance(other, (Vector, StringLiteral)):
        return type(other) is not type(value)
    return (
        isinstance(value, Vector)
        and isinstance(other, sp.Expr)
        and other.is_number is True
    )


def _spread(
    func, left: sp.Basic, right: sp.Basic, context: Context, decide: bool
) -> sp.Basic | None:
    """An equation between two vectors, for soLve, applied element by element.

    Returns None unless both sides are nonempty vectors of the same length, of
    expressions rather than relations or truth values. Elements that are vectors on
    both sides are related the same way, if they can be, so the result keeps the sides'
    shape.
    """
    if func is not sp.Eq:
        return None
    if not isinstance(left, Vector) or not isinstance(right, Vector):
        return None
    if len(left.args) != len(right.args) or not left.args:
        return None
    if not all(isinstance(e, sp.Expr) for e in (*left.args, *right.args)):
        return None
    related = []
    for one, other in zip(left.args, right.args, strict=True):
        spread = _spread(func, one, other, context, decide)
        related.append(
            _stated(func, one, other, context, decide) if spread is None else spread
        )
    return Vector(*related)


def _stated(
    func, left: sp.Basic, right: sp.Basic, context: Context, decide: bool
) -> sp.Basic:
    """Build a relation of two simplified sides, decided where possible.

    Under Approximate, where tests are decided in numbers, an ordering with `?` on a
    side is false, as a comparison with an undefined floating-point value is and as
    Derive decides `ABS(?) >= 0` there. Exactly, it is undecided.
    """
    try:
        relation = func(left, right, evaluate=False)
    except Exception:
        relation = func(left, right)
    if decide and context.precision is Precision.APPROXIMATE and _unordered(relation):
        return sp.false
    if not (decide and isinstance(relation, Relational) and _comparable(relation)):
        return relation
    held = _answered(relation, context)
    if held is None:
        return _against_zero(relation)
    return sp.true if held else sp.false


def _unordered(relation: sp.Basic) -> bool:
    """Whether `relation` is an ordering with `?` on a side."""
    return (
        isinstance(relation, Relational)
        and not isinstance(relation, (sp.Equality, sp.Unequality))
        and sp.nan in relation.args
    )


def _against_zero(relation: Relational) -> Relational:
    """An ordering of two rational multiples of one real expression as that expression's
    comparison with 0, or the relation as it stands.

    `c*u > d*u` holds exactly where `(c - d)*u > 0`, which is `u > 0` for `c > d` and
    `u < 0` for `c < d`: an integral between `-p` and `p` splits on `p > -p`, which is
    `p > 0`.
    """
    if isinstance(relation, (sp.Equality, sp.Unequality)):
        return relation
    left, right = relation.args
    if left.is_number or right.is_number:
        return relation
    ratio = attempt(left / right, sp.cancel)
    if ratio is None or not ratio.is_Rational or ratio == 1:
        return relation
    content, primitive = right.as_content_primitive()
    sign, u = primitive.as_coeff_Mul()
    if u.could_extract_minus_sign():
        sign, u = -sign, -u
    d = content * sign
    c = ratio * d
    if c > d:
        return relation.func(u, 0, evaluate=False)
    return relation.func(0, u, evaluate=False).reversed


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

    Vectors of mismatched shapes, or a vector and a scalar, have no difference. Two
    vectors are different values once they hold no variables: `[1, 2] = [1, 2, 3]` is
    false, while `[x, y] = [1, 2, 3]` stands. A vector and a scalar stand, as in Derive:
    `[1, 2, 2] = 7` stays as written, which TENSOR.MTH's `NZEL_T` relies on to build
    equations of indices and elements.
    """
    if any(isinstance(side, Vector) for side in relation.args):
        if relation.free_symbols or not all(
            isinstance(side, Vector) for side in relation.args
        ):
            return None
        same = relation.lhs == relation.rhs
        return same is isinstance(relation, sp.Equality)
    same = _one_value(relation, context)
    if same is None:
        return None
    return same is isinstance(relation, sp.Equality)


def _one_value(relation: Relational, context: Context) -> bool | None:
    """Whether the two sides of an equation are the same value.

    Identical sides are equal even without a difference to take (`inf = inf`,
    `true = true`). Otherwise the difference is simplified with the pipeline's rewrites,
    so that `SIN(x)^2 + COS(x)^2 = 1` is decided too, and then put in normal form, which
    multiplies out and cancels: `(x + 1)^2 = x^2 + 2*x + 1` is decided, as is any
    equation whose sides simplify to the same value, and the sides are equal where
    Simplify would make their difference 0. A difference still in a variable decides
    nothing: the equation may hold for some values and not others.
    """
    if relation.lhs == relation.rhs:
        return True
    difference = attempt(relation, lambda r: _rewritten(r.lhs - r.rhs, context))
    if difference is not None and not difference.is_number:
        difference = normal_form(difference, context.order, over_numbers=False)
    if difference is None or not difference.is_number:
        return None
    # A variable beside an infinity is absorbed by it, so `x = inf` leaves `x` free.
    if difference.is_finite is not True and relation.free_symbols:
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


def _definition(
    expression: sp.Basic, context: Context, decide: bool, top: bool = False
) -> sp.Basic:
    """Simplify the value of an assignment or function definition.

    The first two arguments, the name and the defining operator, are left untouched so
    the line still reads as a definition. `top` keeps an equation that is the value.
    """
    head, operator, *value = expression.args
    return expression.func(
        head, operator, *(_transform(part, context, decide, top) for part in value)
    )


def _boolean(expression: Boolean, context: Context, decide: bool) -> sp.Basic:
    """Simplify relations joined by boolean operators.

    `6 >= -2*x AND 3*x /= -9` involves one variable, so its simplest form is the range
    it describes. Otherwise (several variables, or non-relation operands) boolean
    algebra runs: `p XOR q` is rewritten in the other operators, `p OR NOT p` becomes
    `true`, and a conjunct common to every disjunct is factored out. The result keeps
    its shape, with its operands simplified.

    The equations of a system's solution keep their order, that of its variables.
    """
    if getattr(expression, "in_order", False):
        return expression.func(*(_transform(a, context, decide) for a in expression.args))
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
    """The range a conjunction of relations in one variable describes.

    A conjunct may be a disjunction of such relations, as a solution is: the conjunction
    is then taken apart into its disjunctive normal form and each disjunct reduced, so
    that `(x = -2 OR x = 2) AND x > 0` is `x = 2`.
    """
    if not isinstance(expression, sp.And):
        return None
    parts = expression.args
    if len(expression.free_symbols) != 1:
        return None
    if all(isinstance(part, Relational) for part in parts):
        return _range_of(expression)
    if not all(
        isinstance(part, Relational)
        or (
            isinstance(part, sp.Or)
            and all(isinstance(link, Relational) for link in part.args)
        )
        for part in parts
    ):
        return None
    if prod(len(part.args) for part in parts if isinstance(part, sp.Or)) > 16:
        return None
    try:
        disjuncts = sp.to_dnf(expression).args
    except Exception:
        return None
    reduced = []
    for disjunct in disjuncts:
        found = (
            _range_of(disjunct)
            if isinstance(disjunct, sp.And)
            else disjunct if isinstance(disjunct, Relational) else None
        )
        if found is None:
            return None
        reduced.append(found)
    return sp.Or(*reduced)


def _range_of(expression: sp.And) -> sp.Basic | None:
    """The range a conjunction of relations in one variable describes, or None."""
    parts = expression.args
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
    """`_simplified_expression`, remembered for the rest of the command (see
    `_simplified_already`)."""
    memo = _simplified_already.get()
    if memo is None:
        return _simplified_expression(expression, context, decide)
    try:
        key = (expression, decide, _in_argument.get())
        found = memo.get(key)
    except TypeError:
        return _simplified_expression(expression, context, decide)
    if found is not None and found[0] is context:
        return found[1]
    result = _simplified_expression(expression, context, decide)
    memo[key] = (context, result)
    return result


def _simplified_expression(expression: sp.Basic, context: Context, decide: bool) -> sp.Basic:
    """Run an ordinary expression through the whole sequence, or, for a number that
    Approximate precision approximates as it is, straight to its last steps (see
    `_approximated_at_once`), and for a polynomial with integer coefficients, through
    the normal form only (see `_plain_polynomial_form`)."""
    expression = _summed(expression)
    if _approximated_at_once(expression, context):
        return _approximated_finally(expression, context)
    normal = _plain_polynomial_form(expression, context)
    if normal is not None:
        return _approximated_finally(normal, context)
    return _simplified_fully(expression, context, decide)


def _plain_polynomial_form(expression: sp.Basic, context: Context) -> sp.Basic | None:
    """What Simplify makes of a polynomial with integer coefficients (see
    `_plain_polynomial`), or None for any other expression: its normal form, which no
    other step changes. Values a program computes are often such polynomials, and
    the other steps of Simplify cost many times what the normal form does.

    Coefficients that are fractions are left to the whole sequence, where a rewrite
    may put the polynomial over one denominator before the normal form, which then
    writes it otherwise: `(1 - x^2)*(105*x^2/2 - 15/2)` is `(1 - x^2)*(105*x^2 -
    15)/2`."""
    if not _plain_polynomial(expression):
        return None
    return normal_form(expression, context.order, over_numbers=False)


def _plain_polynomial(expression: sp.Basic) -> bool:
    """Whether `expression` is a polynomial with integer coefficients in variables,
    declared or not, other than a lone number or variable: a sum, product or positive
    integer power of those."""
    if not isinstance(expression, (sp.Add, sp.Mul, sp.Pow)):
        return False
    pending = [expression]
    while pending:
        node = pending.pop()
        kind = type(node)
        if kind is sp.Add or kind is sp.Mul:
            pending.extend(node.args)
        elif kind is sp.Pow:
            base, exponent = node.args
            if not (exponent.is_Integer and exponent > 0):
                return False
            pending.append(base)
        elif kind is not sp.Symbol and not isinstance(node, sp.Integer):
            return False
    return True


def _approximated_at_once(expression: sp.Basic, context: Context) -> bool:
    """Whether `expression` is a number Approximate precision approximates as it is:
    an arithmetic of numbers, constants and the elementary functions, which evaluate
    to digits, under the principal branch. Simplifying it first changes only its form,
    never its value, and it is then a number again; simplifying the parts of a
    constant through every rewrite can take seconds."""
    if context.precision is not Precision.APPROXIMATE:
        return False
    if context.branch is not Branch.PRINCIPAL:
        # Under Real, an odd root of a negative number is real, which Simplify makes
        # it: the number itself is the principal root.
        return False
    if not isinstance(expression, sp.Expr) or isinstance(expression, sp.Rational):
        return False
    return all(
        type(part) in _AT_ONCE or isinstance(part, sp.Rational)
        for part in sp.preorder_traversal(expression)
    )


#: The parts of a number `_approximated_at_once` approximates besides rationals:
#: arithmetic, the constants, and the functions that evaluate to digits wherever they
#: are defined.
_AT_ONCE = frozenset(
    {
        sp.Add,
        sp.Mul,
        sp.Pow,
        sp.exp,
        sp.log,
        sp.Abs,
        sp.sin,
        sp.cos,
        sp.tan,
        sp.cot,
        sp.sec,
        sp.csc,
        sp.asin,
        sp.acos,
        sp.atan,
        sp.sinh,
        sp.cosh,
        sp.tanh,
        type(sp.I),
        type(sp.pi),
        type(sp.E),
    }
)


def _simplified_fully(expression: sp.Basic, context: Context, decide: bool) -> sp.Basic:
    """`_simplified_expression` through every step.

    The answer of a `FACTOR`, `EXPAND` or `SOLVE` head is the form the author asked for,
    and the steps here would undo it: gated rewrites prefer shorter forms (a
    factorization is often longer), and the normal form would multiply a prime
    decomposition back out. So each such answer is replaced by a placeholder symbol
    during the steps and restored before approximation.

    Inside a sum this applies only to numeric answers, since a sum is put in normal
    form regardless: `FACTOR(x^2 - 1) + x^3` is `x^3 + x^2 - 1`, while
    `FACTOR(12) + 1` keeps its `2^2*3`.
    """
    held = {found: sp.Dummy("answer") for found in sp.ordered(_commands(expression))}
    if held:
        expression = expression.xreplace(held)
    expression = _numeric(expression)
    expression = _bounded(expression, _spliced_cases)
    # Roots of perfect powers go before the intervals, which then settle the absolute
    # value left behind: `SQRT(x^2 + 2*x + 1)` is `-x - 1` below -1. Under Branch Any
    # the root is taken whatever the sign, by the branch rules.
    if context.branch is not Branch.ANY:
        expression = _bounded(expression, _perfect_powers)
    # Before the rewrites, so they see what the intervals settle: `ABS(x) + ABS(x - 1)`
    # over `(0, 1)` cancels only once both bars are removed.
    expression = _bounded(expression, lambda e: settled(e, context))
    expression = _bounded(expression, _cases_under_their_tests)
    expression = _divergent(expression, context)
    expression = _bounded(expression, _absorbed_by_infinity)
    expression = _rewritten_beside_standing(expression, context)
    expression = _factorials(expression)
    expression = forced(expression, ambiguous_products)
    # The normal form is not gated on getting shorter.
    #
    # `over_numbers` is off so numeric denominators stay with their terms instead of
    # becoming a common denominator: `x^2/6 + x/2` comes back as authored.
    expression = normal_form(expression, context.order, over_numbers=False)
    expression = _spelled(expression, context)
    expression = forced(expression, ambiguous_products)
    if held:
        expression = expression.xreplace(
            {dummy: found.args[0] for found, dummy in held.items()}
        )
    return _approximated_finally(expression, context)


def _approximated_finally(expression: sp.Basic, context: Context) -> sp.Basic:
    """The last steps of `_simplified_expression`: the precision mode applied, and
    decimal coefficients spread as a decimal notation shows them."""
    if context.precision == Precision.APPROXIMATE:
        expression = forced(expression, _one_exponential)
    expression = approximated(expression, context)
    if context.precision is Precision.APPROXIMATE or context.notation in (
        Notation.DECIMAL,
        Notation.SCIENTIFIC,
    ):
        # After the numbers are rounded, so that spreading a coefficient changes only
        # the form, never how a number is approximated.
        expression = _within_budget(expression, forced, _decimal_coefficients)
    return expression


def _one_exponential(expression: sp.Basic) -> sp.Basic:
    """Powers of numbers over multiples of one exponent as one exponential, and the
    constant part of an exponential's exponent as a factor.

    Under Approximate, Derive approximates the logarithms in `#e^(t*(LN(5) - LN(2))/3)`
    as it goes, so the answer is `#e^(0.305430*t)`. The exact rewrites turn it into
    `2^(-t/3)*5^(t/3)` first, which this puts back before approximation. A single power
    stays: `5^(t/3)` is `5^(0.333333*t)`.

    Likewise `#e^1.15479*#e^(-0.701797*t)` stays two factors, the first a number:
    CURV_FIT.DOC's exponential fit is `3.17338*#e^(-0.701797*t)`. Sympy joins the two,
    so the constant is split off again here, unevaluated for approximation to round.
    """

    def split(power: sp.Basic) -> sp.Basic:
        constant, rest = power.args[0].as_independent(*power.free_symbols, as_Add=True)
        if constant == 0 or not rest.free_symbols:
            return power
        return sp.Mul(sp.exp(constant), sp.exp(rest), evaluate=False)

    def merged(product: sp.Basic) -> sp.Basic:
        groups: dict[sp.Basic, list[tuple[sp.Basic, sp.Basic]]] = {}
        for factor in product.args:
            base, exponent = factor.as_base_exp()
            if base.is_Rational and base > 0 and base != 1 and exponent.free_symbols:
                coefficient, rest = exponent.as_coeff_Mul()
                groups.setdefault(rest, []).append((factor, coefficient * sp.log(base)))
        merging = {rest: group for rest, group in groups.items() if len(group) > 1}
        if not merging:
            return product
        dropped = {factor for group in merging.values() for factor, _ in group}
        kept = [factor for factor in product.args if factor not in dropped]
        powers = [
            sp.exp(sp.Mul(sp.Add(*(log for _, log in group)), rest, evaluate=False))
            for rest, group in merging.items()
        ]
        return sp.Mul(*kept, *powers)

    if expression.has(sp.Pow):
        expression = expression.replace(lambda e: e.is_Mul, merged)
    if not expression.has(sp.exp):
        return expression
    return expression.replace(
        lambda e: isinstance(e, sp.exp) and e.args[0].is_Add and bool(e.free_symbols),
        split,
    )


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


def _cases_under_their_tests(expression: sp.Basic) -> sp.Basic:
    """Each case of a conditional simplified knowing that its test holds and the tests
    before it fail, where that makes it simpler; each test in the shorter of its boolean
    normal forms, with its orderings `_against_zero`, and simplified knowing that the
    tests before it fail.

    A case is only ever evaluated where it applies: `SUM(SIN((2*k - 1)*x), k, 1, n)` is
    0 where `SIN(x) = 0`, not `-n*SIN(x)`, and where `p > -p` holds `ABS(p)` is `p`. A
    test that holds once the ones before it fail leaves the cases after it out. See
    `_with_fact` for what a test tells. Derive's four-argument `IF` is not a
    `Piecewise`, so its unknown case assumes nothing.
    """

    def known(cases: sp.Piecewise) -> sp.Basic:
        failed: list[Relational] = []
        rebuilt = []
        for value, test in cases.args:
            if isinstance(test, Boolean) and not isinstance(test, BooleanAtom):
                test = _algebra(test)
                against = attempt(
                    test,
                    lambda t: t.replace(
                        lambda e: isinstance(e, Relational) and _comparable(e),
                        _against_zero,
                    ),
                )
                if against is not None:
                    test = against
                asked = test
                for fact in failed:
                    asked = _with_fact(asked, fact)
                if isinstance(asked, BooleanAtom) or pays(asked, test):
                    test = asked
            candidate = value
            for fact in _facts(test) + failed:
                candidate = _with_fact(candidate, fact)
            rebuilt.append((candidate if pays(candidate, value) else value, test))
            if test is not sp.true:
                failed += _facts(sp.to_nnf(sp.Not(test)))
        return sp.Piecewise(*rebuilt)

    if not expression.has(sp.Piecewise):
        return expression
    return expression.replace(lambda e: isinstance(e, sp.Piecewise), known)


def _facts(test: sp.Basic) -> list[Relational]:
    """The relations a test implies on its own: itself, or each relation of a
    conjunction."""
    if isinstance(test, Relational):
        return [test]
    if isinstance(test, sp.And):
        return [part for part in test.args if isinstance(part, Relational)]
    return []


def _with_fact(value: sp.Basic, fact: Relational) -> sp.Basic:
    """`value` as it is where `fact` holds, or `value` itself if it can't tell.

    A relation of a polynomial in one real variable is solved for its range. A single
    point is substituted; a ray or the line without a point puts the variable as its
    end plus or minus a positive (or nonzero) unknown, which settles signs, and back. An
    equation in anything else substitutes one side for the other. A substitution that
    meets an undefined value anywhere in `value` (`1/x` at 0) is not made.
    """
    variables = fact.free_symbols
    if len(variables) == 1:
        (variable,) = variables
        difference = fact.lhs - fact.rhs
        if variable.is_real and difference.is_polynomial(variable):
            values = sp.solveset(fact, variable, sp.Reals)
            restricted = _restricted(value, variable, values)
            if restricted != value:
                return restricted
    if not isinstance(fact, sp.Eq):
        return value
    candidates = [
        _at(value, old, new)
        for old, new in ((fact.lhs, fact.rhs), (fact.rhs, fact.lhs))
        if not old.is_number and not new.has(old)
    ]
    candidates = [found for found in candidates if found is not None]
    return min(candidates, key=sp.count_ops, default=value)


def _restricted(value: sp.Basic, variable: sp.Symbol, values: sp.Set) -> sp.Basic:
    """`value` where `variable` is restricted to the real `values` (see `_with_fact`)."""
    if isinstance(values, sp.FiniteSet) and len(values) == 1:
        found = _at(value, variable, values.args[0])
        return value if found is None else found
    if isinstance(values, sp.Union) and len(values.args) == 2:
        left, right = sorted(values.args, key=lambda part: part.inf)
        if (
            left.inf is sp.S.NegativeInfinity
            and right.sup is sp.oo
            and left.sup == right.inf
            and left.right_open
            and right.left_open
        ):
            return _from_an_end(value, variable, left.sup, 1, "nonzero")
        return value
    if not isinstance(values, sp.Interval):
        return value
    if values.sup is sp.oo and values.inf.is_finite:
        sign = "positive" if values.left_open else "nonnegative"
        return _from_an_end(value, variable, values.inf, 1, sign)
    if values.inf is sp.S.NegativeInfinity and values.sup.is_finite:
        sign = "positive" if values.right_open else "nonnegative"
        return _from_an_end(value, variable, values.sup, -1, sign)
    return value


def _from_an_end(
    value: sp.Basic, variable: sp.Symbol, end: sp.Basic, side: int, sign: str
) -> sp.Basic:
    """`value` with `variable` written as `end + side*t`, for a real unknown `t` that is
    `sign` ("positive", "nonnegative" or "nonzero"), and back."""
    unknown = sp.Dummy("t", real=True, **{sign: True})
    return value.subs(variable, end + side * unknown).subs(
        unknown, side * (variable - end)
    )


#: Values that show a substitution met something undefined.
_UNDEFINED = (sp.nan, sp.zoo, sp.oo, sp.S.NegativeInfinity, sp.AccumBounds)


def _at(value: sp.Basic, old: sp.Basic, new: sp.Basic) -> sp.Basic | None:
    """`value` with `old` replaced by `new`, or None if a part of it becomes undefined
    or infinite there or binds a variable of `old`."""
    bound = old.free_symbols

    def put(part: sp.Basic) -> sp.Basic:
        if part == old:
            return new
        if part.is_Atom or not part.has(old):
            return part
        if bound & set(getattr(part, "variables", ())):
            raise ValueError("bound")
        rebuilt = part.func(*(put(argument) for argument in part.args))
        if rebuilt.has(*_UNDEFINED) and not part.has(*_UNDEFINED):
            raise ValueError("undefined")
        return rebuilt

    try:
        return put(value)
    except Exception:
        return None


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


def _absorbed_by_infinity(expression: sp.Basic) -> sp.Basic:
    """Drop the terms with variables from each sum with a real infinite term.

    Derive takes a variable for a finite value, so `inf - 1/x` is `inf`, like
    `inf - x`, where sympy keeps `1/x` as possibly infinite. Terms holding an infinity
    themselves stay, so `inf - inf*x` is not decided.
    """
    infinities = (sp.oo, sp.S.NegativeInfinity, sp.zoo, sp.nan)

    def absorbed(total: sp.Basic) -> sp.Basic:
        kept = [
            term
            for term in total.args
            if not term.free_symbols or term.has(*infinities) or not term.is_Atom
            and term.has(sp.Integral, sp.Sum, sp.Product, sp.Limit, sp.Derivative)
        ]
        return total if len(kept) == len(total.args) else sp.Add(*kept)

    if not expression.has(sp.oo, sp.S.NegativeInfinity):
        return expression
    return expression.replace(
        lambda e: e.is_Add and (sp.oo in e.args or sp.S.NegativeInfinity in e.args),
        absorbed,
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

    As a test, vectors of different lengths are unequal, whatever their elements:
    ODE.MTH's `IF(u SUB 1 = [], ...)` takes the second branch for `[a] = []`. Shown as
    an answer, such an equation of unknowns stands (see `_relation`).
    """
    test = as_condition(test)
    if _formula_against_truth(test):
        return None
    if isinstance(test, (sp.Eq, sp.Ne)) and sp.nan in test.args:
        found = compared(test.func, test.lhs, test.rhs, _link)
        return bool(found) if isinstance(found, BooleanAtom) else None
    if (
        isinstance(test, (sp.Eq, sp.Ne))
        and isinstance(test.lhs, Vector)
        and isinstance(test.rhs, Vector)
        and len(test.lhs.args) != len(test.rhs.args)
    ):
        return isinstance(test, sp.Ne)
    for candidate in _truths(test, context):
        if candidate is sp.true:
            return True
        if candidate is sp.false:
            return False
    return decided(test, context)


def _formula_against_truth(test: sp.Basic) -> bool:
    """Whether `test` equates a logical formula kept as written (a truth table's
    heading, `Logical`) with a truth value: the formula holds for some values of its
    variables and not others, so the test is undecided, as `(x AND y) = true` is."""
    if not isinstance(test, (sp.Eq, sp.Ne)):
        return False
    left, right = test.args
    return any(
        isinstance(one, Logical) and isinstance(other, BooleanAtom)
        for one, other in ((left, right), (right, left))
    )


def _decide_simplified(test: sp.Basic, context: Context) -> bool | None:
    """Whether an `IF` test holds, using full simplification if necessary.

    The quicker `_decide` is tried first. If it is undecided, the test is simplified as
    if it stood alone, which decides e.g. `x < 1 AND x > 3` (an empty range) and
    `(x^2 + x^2*y^2)/(x^2*y) = (1 + y^2)/y`. ODE1.MTH's `HOMOGENEOUS` needs this to see
    that a derivative is zero.
    """
    if _formula_against_truth(test):
        return None
    if isinstance(test, (sp.Eq, sp.Ne)):
        found = compared(test.func, test.lhs, test.rhs, _link)
        if isinstance(found, BooleanAtom):
            return bool(found)
        if sp.nan in test.args:
            # `u = ?` with u not a value of its own (a variable, an expression in
            # one) is undecided: the unknown may be anything.
            return None
    elif _vectors_ordered(test):
        return _elementwise_ordering(test, context)
    if (
        _against_a_relation(test)
        or _ordering_a_lone_variable(test, context)
        or _lone_variable_against_a_number(test, context)
    ):
        return None
    verdict = _vector_against_a_number(test)
    if verdict is not None:
        return verdict
    verdict = _decide(test, context)
    if verdict is not None:
        return verdict
    if _against_a_lone_variable(test, context):
        return None
    try:
        simplified = _transform(as_condition(test), context, True)
    except Exception:
        return None
    if simplified is sp.true:
        return True
    if simplified is sp.false:
        return False
    return None


def _vectors_ordered(test: sp.Basic) -> bool:
    """Whether `test` is an ordering between two vectors."""
    return (
        isinstance(test, Relational)
        and isinstance(test.lhs, Vector)
        and isinstance(test.rhs, Vector)
    )


def _elementwise_ordering(test: Relational, context: Context) -> bool | None:
    """Whether an ordering between two vectors holds, as a test: when it holds between
    each pair of elements, so `[1, 2] <= [2, 2]` holds and `[1, 3] <= [2, 2]` does not.
    Groebner basis programs test whether one monomial divides another so, by comparing
    their exponent vectors. Vectors of different lengths are not ordered (None)."""
    if len(test.lhs.args) != len(test.rhs.args):
        return None
    verdicts = set()
    for one, other in zip(test.lhs.args, test.rhs.args, strict=True):
        try:
            link = test.func(one, other, evaluate=False)
        except Exception:
            return None
        verdict = _decide_simplified(link, context)
        if verdict is False:
            return False
        verdicts.add(verdict)
    return True if verdicts <= {True} else None


def _link(func, one: sp.Basic, other: sp.Basic) -> sp.Basic:
    """An element relation of a test comparing vectors, decided where sympy decides it
    on its own, as it does for numbers."""
    try:
        return func(one, other)
    except Exception:
        return func(one, other, evaluate=False)


def _vector_against_a_number(test: sp.Basic) -> bool | None:
    """Whether a test equating a vector of numbers with a number holds: it never does.

    Only a test is decided so: a vector of numbers is not a number, so `SELECT(k /= 0,
    k, [0, [1, 1], 0])` keeps `[1, 1]`. The equation simplified on its own stands (see
    `_equated`). None for any other test.
    """
    if not isinstance(test, (sp.Eq, sp.Ne)):
        return None
    vectors = [isinstance(side, Vector) for side in test.args]
    if vectors.count(True) != 1:
        return None
    vector, number = test.args if vectors[0] else reversed(test.args)
    if vector.free_symbols or not getattr(number, "is_number", False):
        return None
    return isinstance(test, sp.Ne)


def _against_a_lone_variable(test: sp.Basic, context: Context) -> bool:
    """Whether `test` equates an undeclared variable with something not mentioning it,
    which holds for one value of the variable and not for others, so that simplifying
    cannot decide it: ODE.MTH's `IF(fail = (local :== ...), ...)` compares the word
    `fail` with each method's answer, often long, and gives `local` when undecided."""
    if not isinstance(test, (sp.Eq, sp.Ne)):
        return False
    for side, other in (test.args, reversed(test.args)):
        if (
            type(side) is sp.Symbol
            and side.name not in context.domains
            and isinstance(other, sp.Basic)
            and not isinstance(other, Relational)
            and not other.has(side)
            and other.free_symbols
        ):
            return True
    return False


def _lone_variable_against_a_number(test: sp.Basic, context: Context) -> bool:
    """Whether `test` equates an undeclared variable with a real number, which holds
    for that value of the variable only: a program testing `IF(q = k, ...)` with `k`
    counting through numbers asks it at every pass, and nothing can decide it."""
    if not isinstance(test, (sp.Eq, sp.Ne)):
        return False
    for side, other in (test.args, reversed(test.args)):
        if (
            type(side) is sp.Symbol
            and side.name not in context.domains
            and isinstance(other, sp.Number)
            and other.is_finite
        ):
            return True
    return False


def _ordering_a_lone_variable(test: sp.Basic, context: Context) -> bool:
    """Whether `test` orders a variable that may be any real number against a finite
    expression not mentioning it, which holds for some values of the variable and not
    for others: a loop that runs until `k > w`, with `w` a variable, tests that at
    every pass, and simplifying it cannot decide it."""
    if not isinstance(test, Relational) or isinstance(test, (sp.Eq, sp.Ne)):
        return False
    for side, other in (test.args, reversed(test.args)):
        if (
            type(side) is sp.Symbol
            and _unrestricted(side.name, context)
            and isinstance(other, sp.Expr)
            and not other.has(side, sp.oo, -sp.oo, sp.zoo, sp.nan)
        ):
            return True
    return False


def _unrestricted(name: str, context: Context) -> bool:
    """Whether the variable `name` may be any real number."""
    domain = context.domain(name)
    real = domain.kind in (DomainKind.REAL, DomainKind.UNKNOWN)
    return real and not domain.has_interval


def _against_a_relation(test: sp.Basic) -> bool:
    """Whether `test` equates an expression with a relation, which simplifying the
    relation cannot decide: ODE.MTH's `IF(fail = (local :== ...), ...)` compares the
    word `fail` with an equation whose sides may be integrals that take minutes to
    simplify, and gives `local` in both of the other cases."""
    if not isinstance(test, (sp.Eq, sp.Ne)):
        return False
    sides = [isinstance(side, Relational) for side in test.args]
    return sides[0] != sides[1] and all(
        isinstance(side, (sp.Expr, Relational)) for side in test.args
    )


def _truths(test: sp.Basic, context: Context) -> list[sp.Basic]:
    """Forms of `test` worth checking, easiest first.

    A difference that is a number is also multiplied out, which a complex one needs:
    `(2 + 3*#i)^2 = 1` is decided by `-6 + 12*#i` being nonzero.
    """
    if not isinstance(test, Relational):
        return [test]
    readings = [test, attempt(test, lambda t: t.func(t.lhs, t.rhs))]
    if isinstance(test.lhs, sp.MatrixExpr) or isinstance(test.rhs, sp.MatrixExpr):
        # A difference of vector variables is a matrix expression, which sympy takes
        # for unequal to zero.
        return [reading for reading in readings if reading is not None]
    difference = attempt(test, lambda t: _rewritten(t.lhs - t.rhs, context))
    if difference is not None:
        readings.append(attempt(test, lambda t: t.func(difference, 0)))
        if difference.is_number and not difference.is_Number:
            readings.append(attempt(test, lambda t: t.func(sp.expand(difference), 0)))
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

    Exponents that are constant are made so first; see `_constant_exponents`. A `MOD`
    is worked on as its definition, a floor, and an answer's floors that make up a
    `MOD` are collected into one again.
    """
    written = forced(head, _constant_exponents)
    head = residues_as_floors(written)
    value = _answer_in_words(head, context)
    if value is not None:
        return floors_as_residues(value)
    spelled = attempt(head, _by_definition)
    if spelled is None or spelled == head:
        return written
    # A second try, which may not end where the first ended at once: within its budget
    # (see `budget.RESPELLED`), else the head stands.
    again = within(budget.RESPELLED, lambda: _answer_in_words(spelled, context))
    return written if again is None else floors_as_residues(again)


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


def _at_undefined(head: sp.Basic) -> bool:
    """Whether an integral has an undefined bound or a limit is taken at an undefined
    point: then it is undefined itself, as a sum to `?` is."""
    if isinstance(head, sp.Integral):
        return any(sp.nan in limit[1:] for limit in head.limits)
    if isinstance(head, sp.Limit):
        return head.args[2] is sp.nan
    return False


def _worked_out(head: sp.Basic, context: Context) -> sp.Basic:
    """Evaluate one head by the rule for its kind. A command's answer in its operand,
    as in `LIM(EXPAND(u)/EXPAND(v), n, inf)`, is worked on as the value it is."""
    if _unfinished(head, context):
        return head
    if head.has(Commanded):
        head = head.replace(
            lambda found: isinstance(found, Commanded), lambda found: found.args[0]
        )
    if isinstance(head, Approx):
        return _approximation(head, context)
    if _at_undefined(head):
        return sp.nan
    if isinstance(head, sp.Integral):
        pieces = _integral_by_branches(head, context)
        if pieces is None:
            pieces = _integral_by_pieces(head)
        if pieces is not None:
            return pieces
        stepped = _integral_by_steps(head)
        if stepped is not None:
            value = _worked_out(stepped, context)
            if not value.has(sp.Integral):
                return value
        if context.precision is Precision.APPROXIMATE and head.is_number:
            with numerically():
                return integral(head)
        if context.branch is Branch.ANY:
            # Under Branch Any a logarithm of a power is split with no absolute value:
            # `INT((TAN(t)^2 + 1)/TAN(t), t)`, found as `LN(TAN(t)^2)/2`, is
            # `LN(TAN(t))`, and its exponential is not `ABS(TAN(t))`.
            return forced(integral(head), _any_extracted_powers)
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
        geometric = _geometric_terms_limit(*head.args[:3], context)
        return _limit(head) if geometric is None else geometric
    if isinstance(head, Antidifference):
        return _antidifference(head, context)
    if isinstance(head, sp.Derivative):
        cases = _by_cases(head.expr, set(head.variables))
        if cases is not None:
            head = head.func(cases, *head.variable_count, evaluate=False)
        rounded = _rounding_differentiated(head)
        if rounded is not None:
            return rounded
        polynomial = _polynomial_derivative(head)
        if polynomial is not None:
            return polynomial
    try:
        value = head.doit(deep=False)
    except Exception:
        return _undecided(head)
    if value is None:
        return head
    if isinstance(head, sp.Derivative):
        # Over one denominator, since a derivative may be differentiated again, or
        # have a point substituted into it, before it is simplified: sympy's raw
        # derivative is a sum of fractions that the product rule multiplies out with
        # every order. FUN.MTH's F21 nests three derivatives of up to fifth order,
        # which left raw grow to hundreds of thousands of characters.
        return over_one_denominator(_flat_between_jumps(value))
    return value


class _Rounded(sp.Function):
    """A standing `APPROX` call while its derivative is taken (see
    `_rounding_differentiated`)."""

    def _eval_derivative(self, variable: sp.Basic) -> sp.Basic:
        value, *digits = self.args
        return self.func(sp.diff(value, variable), *digits)


def _rounding_differentiated(head: sp.Derivative) -> sp.Basic | None:
    """The derivative of an expression holding a standing `APPROX` call, as one in a
    definition's body that acts on a parameter, or None if it holds none. Rounding is
    no function of the variable: the derivative of a rounded value is the derivative
    rounded, so `DIF(APPROX(b*LN(x)), x)` is `APPROX(b/x)`."""

    def standing(found: sp.Basic) -> bool:
        return isinstance(found, AppliedUndef) and found.func.__name__ == "APPROX"

    if not head.expr.has(AppliedUndef) or not head.expr.has(sp.Function("APPROX")):
        return None
    rounded = attempt(
        head.expr, lambda e: e.replace(standing, lambda f: _Rounded(*f.args))
    )
    if rounded is None or rounded == head.expr:
        return None
    value = attempt(rounded, lambda e: sp.diff(e, *head.variable_count))
    if value is None:
        return None
    return value.replace(lambda found: isinstance(found, _Rounded), _rounding_of)


def _rounding_of(rounded: _Rounded) -> sp.Basic:
    """The standing `APPROX` call of a `_Rounded`, with the factors that hold no number
    to round (integer powers of variables) taken out: `APPROX(u/x)` is `APPROX(u)/x`."""
    value, *digits = rounded.args
    plain = [
        factor
        for factor in sp.Mul.make_args(value)
        if factor.is_Symbol
        or (factor.is_Pow and factor.base.is_Symbol and factor.exp.is_Integer)
    ]
    rest = sp.Mul(*(f for f in sp.Mul.make_args(value) if f not in plain))
    return sp.Mul(*plain) * sp.Function("APPROX")(rest, *digits)


def _polynomial_derivative(head: sp.Derivative) -> sp.Basic | None:
    """A higher derivative of a product of polynomials in one variable, worked out on
    the expanded polynomial, or None for any other derivative.

    Sympy differentiates a product `n` times by the general Leibniz rule, a term for
    every way of sharing out the `n` differentiations, so the tenth derivative of the
    product of `1 + x^i` for `i` from 1 to 10 has tens of thousands of terms and takes
    half a minute; expanded first, it takes milliseconds."""
    if len(head.variable_count) != 1 or not isinstance(head.expr, sp.Mul):
        return None
    ((variable, count),) = head.variable_count
    if not (isinstance(count, sp.Integer) and count > 1):
        return None
    try:
        polynomial = sp.Poly(head.expr, variable)
    except sp.PolynomialError:
        return None
    if not all(coefficient.is_number for coefficient in polynomial.coeffs()):
        return None
    return polynomial.diff((variable, int(count))).as_expr()


def _by_cases(expression: sp.Basic, variables: set) -> sp.Basic | None:
    """`expression` with each suspended conditional in `variables` written as the
    `Piecewise` of its branches (see `branched`), or None if it has none.

    Calculus in a variable works on each case where its test picks it, so a function
    defined by cases has a derivative and one-sided limits. A conditional in a branch
    is written out too, to a few levels.
    """

    def in_variables(found: sp.Basic) -> bool:
        return isinstance(found, Suspended) and bool(found.free_symbols & variables)

    changed = False
    for _ in range(_CASE_LEVELS):
        found = [part for part in expression.atoms(Suspended) if in_variables(part)]
        cases = {part: branched(part) for part in found}
        cases = {part: value for part, value in cases.items() if value is not None}
        if not cases:
            break
        expression = expression.xreplace(cases)
        changed = True
    return expression if changed else None


#: How deep `_by_cases` writes out conditionals in the branches of conditionals.
_CASE_LEVELS = 4


def _integral_by_steps(head: sp.Integral) -> sp.Integral | None:
    """An integral over one variable of a function defined by cases, with each case
    multiplied by the step functions of its test, or None.

    `INT(IF(1 <= t <= 2, COS(t), 0), t, 0, u)` is `INT(COS(t)*(SIGN(t - 1) - SIGN(t -
    2))/2, t, 0, u)`, as `CHI(1, t, 2)` writes the same indicator. The two integrands
    differ only where a test is on its boundary, which no integral sees. Only tests
    built from comparisons, by AND, OR and NOT, are written so, and a conditional must
    have a value in every case.
    """
    if len(head.limits) != 1:
        return None
    variable = head.limits[0][0]
    cases = _by_cases(head.function, {variable})
    if cases is None:
        return None

    def indicator(test: sp.Basic) -> sp.Basic | None:
        if isinstance(test, (sp.StrictLessThan, sp.LessThan)):
            return (1 - sp.sign(test.lhs - test.rhs)) / 2
        if isinstance(test, (sp.StrictGreaterThan, sp.GreaterThan)):
            return (1 + sp.sign(test.lhs - test.rhs)) / 2
        parts = [indicator(arg) for arg in test.args]
        if any(part is None for part in parts):
            return None
        if isinstance(test, sp.And):
            return sp.Mul(*parts)
        if isinstance(test, sp.Or):
            return 1 - sp.Mul(*(1 - part for part in parts))
        if isinstance(test, sp.Not):
            return 1 - parts[0]
        return None

    def stepped(piecewise: sp.Piecewise) -> sp.Basic | None:
        total, outside = sp.S.Zero, sp.S.One
        earlier: list[sp.Basic] = []
        for value, test in piecewise.args:
            if value.has(sp.nan) and _nowhere_between(
                sp.And(test, *(sp.Not(other) for other in earlier)), head.limits[0]
            ):
                value = sp.S.Zero
            if test is sp.true:
                return total + outside * value
            weight = indicator(test)
            if weight is None:
                return None
            total += outside * weight * value
            outside *= 1 - weight
            earlier.append(test)
        return None

    if any(
        branch.has(sp.Piecewise)
        for part in cases.atoms(sp.Piecewise)
        for branch in part.args
    ):
        # Nested conditionals as one, so that a `?` for the values none of them picks is
        # seen as such.
        cases = attempt(cases, sp.piecewise_fold) or cases
    pieces = {part: stepped(part) for part in cases.atoms(sp.Piecewise)}
    if not pieces or any(value is None for value in pieces.values()):
        return None
    function = attempt(cases, lambda c: sp.expand(c.xreplace(pieces)))
    if function is None or function.has(sp.Piecewise, sp.nan):
        return None
    return head.func(function, *head.limits)


def _integral_by_branches(head: sp.Integral, context: Context) -> sp.Basic | None:
    """The integral of a conditional whose tests don't involve the variable of
    integration, as the conditional of its branches' integrals, or None:
    `INT(IF(p > 0, x, x^2), x)` is `IF(p > 0, x^2/2, x^3/3)`."""
    if len(head.limits) != 1:
        return None
    variable = head.limits[0][0]
    function = head.function
    if not isinstance(function, Suspended) or not function.has(variable):
        return None
    cases = branched(function)
    if not isinstance(cases, sp.Piecewise) or any(
        test.has(variable) for _, test in cases.args
    ):
        return None
    branches = []
    for case, test in cases.args:
        part = _worked_out(sp.Integral(case, *head.limits), context)
        branches.append((part, test))
    return sp.Piecewise(*branches)


def _integral_by_pieces(head: sp.Integral) -> sp.Basic | None:
    """A definite integral of a function defined by cases, with ends in known order, as
    the sum of the integrals between the points where a test changes, or None.

    Between two consecutive such points every test keeps its truth value, so one case
    applies throughout, and which one is decided at a point inside. A `?` case applying
    there makes the integral `?`; one applying only at the points doesn't count:
    `INT(IF(0 < x < 1, x, IF(1 < x < 2, 1, ?)), x, 0, 2)` is 3/2.

    Conditionals (`IF`) are taken so wherever the tests are on the variable alone; the
    absolute value, sign, step, maximum and minimum of a polynomial in it only between
    numbers or on an infinite interval; with a symbolic end, the step functions' own
    rules give a closed form.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 3:
        return None
    variable, low, high = head.limits[0]
    if not (low.is_extended_real and high.is_extended_real) or low == high:
        return None
    orientation = (high - low).is_extended_positive
    if orientation is None:
        return None
    if not orientation:
        low, high = high, low
    cases = _by_cases(head.function, {variable})
    if cases is None:
        if not head.function.has(*_KINKS):
            return None
        if low.is_finite and high.is_finite and not (low.is_number and high.is_number):
            return None
        kinks = [
            kink
            for kink in head.function.atoms(*_KINKS)
            if kink.has(variable)
        ]
        if not kinks or not all(
            arg.is_polynomial(variable) and arg.free_symbols <= {variable}
            for kink in kinks
            for arg in kink.args
        ):
            return None
        cases = attempt(head.function, lambda f: f.rewrite(sp.Piecewise))
        if cases is None:
            return None
    folded = attempt(cases, sp.piecewise_fold)
    if not isinstance(folded, sp.Piecewise):
        return None
    tests = [test for _, test in folded.args]
    if any(test.free_symbols - {variable} for test in tests):
        return None
    points = {low, high}
    for test in tests:
        if test is sp.true or test is sp.false:
            continue
        edges = attempt(test, lambda t: t.as_set().boundary)
        if not isinstance(edges, (sp.FiniteSet, sp.sets.sets.EmptySet)):
            return None
        points |= {
            point for point in edges if point.is_real and (low < point) == (point < high)
        }
    if len(points) > _PIECES:
        return None
    ordered = sorted(points, key=lambda point: float(point) if point.is_finite else (
        float("-inf") if point == -sp.oo else float("inf")
    ))
    if any(
        (right - left).is_extended_positive is not True
        for left, right in zip(ordered, ordered[1:])
    ):
        return None
    total = sp.S.Zero
    for left, right in zip(ordered, ordered[1:]):
        if left.is_finite and right.is_finite:
            inside = (left + right) / 2
        else:
            inside = left + 1 if left.is_finite else right - 1
        value = None
        for case, test in folded.args:
            held = attempt(test, lambda t: t.subs(variable, inside))
            if held is sp.true:
                value = case
                break
            if held is not sp.false:
                return None
        if value is None or value.has(sp.nan):
            return None if value is None else sp.nan
        part = integral(sp.Integral(value, (variable, left, right)))
        if part.has(sp.Integral) or part.has(sp.nan):
            return None
        total += part
    return total if orientation else -total


#: The functions whose argument's sign picks a case, which `_integral_by_pieces` splits
#: at on an infinite interval.
_KINKS = (sp.Abs, sp.sign, sp.Heaviside, sp.Max, sp.Min)

#: The most points an integral is split at by `_integral_by_pieces`.
_PIECES = 40


def _nowhere_between(test: sp.Basic, limit: tuple) -> bool:
    """Whether `test`, on the variable of a definite integral's `limit`, holds at no more
    than finitely many points between its ends, so that the value of a case it picks
    doesn't count: the `?` that `IF(0 < x < 1, x, IF(1 < x < 2, 1, ?))` gives at 1
    leaves the integral from 0 to 2 defined."""
    if len(limit) != 3:
        return False
    variable, low, high = limit
    if not (low.is_extended_real and high.is_extended_real) or test.free_symbols != {
        variable
    }:
        return False
    if (high - low).is_extended_negative:
        low, high = high, low
    elif not (high - low).is_extended_nonnegative:
        return False
    where = attempt(test, lambda t: t.as_set() & sp.Interval(low, high))
    return isinstance(where, (sp.FiniteSet, sp.sets.sets.EmptySet))


def _constant_exponents(expression: sp.Basic) -> sp.Basic:
    """`expression` with every exponent that is a constant rational function replaced by
    its value.

    Operands reach a calculus head unsimplified, and a function argument can bring an
    exponent like `4*y/(y - x^8/y) - 4*x^4/(y^2/x^4 - x^4)`, which is 4. Taken as it
    stands, the operand isn't a rational function of `y`, and the head is computed over
    a power it can't see through: ODE1.MTH's GEN_HOM_GEN integrates `1/(y^2*x^k - 4*y +
    1)` with such a `k`.
    """

    def constant(power: sp.Basic) -> bool:
        exponent = power.exp
        return (
            bool(exponent.free_symbols)
            and not exponent.is_polynomial()
            and bool(exponent.is_rational_function())
            and sp.count_ops(exponent) <= _CONSTANT_EXPONENT
        )

    def valued(power: sp.Basic) -> sp.Basic:
        value = sp.cancel(power.exp)
        return power if value.free_symbols else power.func(power.base, value)

    def exponent_of(power: sp.Basic) -> bool:
        return isinstance(power, (sp.Pow, sp.exp)) and constant(power)

    if not any(map(exponent_of, expression.atoms(sp.Pow, sp.exp))):
        return expression
    return expression.replace(exponent_of, valued)


#: The largest exponent `_constant_exponents` tries to reduce, in operations.
_CONSTANT_EXPONENT = 60


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
    return any(
        (definition := context.function(type(call).__name__)) is not None
        and definition.body is not None
        for call in calls
    )


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


def _antidifference(head: Antidifference, context: Context) -> sp.Basic:
    """`SUM(u, n)`: by Gosper's algorithm, or else as the definite sum from 0 to
    `n - 1`.

    The definite sum `F(n)` satisfies `F(n + 1) - F(n) = u(n)` by construction, so
    whatever closed forms definite sums have serve here too: sines and cosines of a
    linear function of the index (see `_sum_of_waves`), which Gosper's algorithm can't
    do. `SUM(COS(2*pi*x/3), x)` is `-SQRT(3)*COS(2*pi*x/3 + pi/6)/3`.

    Products of sines and cosines are written as sums, and terms free of the index are
    dropped as the constant of summation. A summand undefined at 0 isn't summed from
    there; Gosper's algorithm covers the rational ones.
    """
    value = attempt(head, lambda h: h.doit(deep=False))
    if value is None:
        return _undecided(head)
    if value != head:
        return value
    expression, index = head.args
    if type(index) is not sp.Symbol:
        return head
    start = attempt(expression, lambda e: e.subs(index, 0))
    if start is None or start.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        return head
    running = sp.Dummy(index.name, integer=True)
    total = _worked_out(
        sp.Sum(expression.subs(index, running), (running, 0, index - 1)), context
    )
    if total.has(sp.Sum, sp.Product, Antidifference) or not is_written(total):
        return head
    return _without_constant(total, index)


def _without_constant(total: sp.Basic, index: sp.Symbol) -> sp.Basic:
    """An antidifference with products of sines and cosines written as sums, and the
    terms free of `index` dropped, in each branch of a conditional."""
    if isinstance(total, sp.Piecewise):
        return sp.Piecewise(
            *((_without_constant(expr, index), cond) for expr, cond in total.args)
        )
    spread = attempt(total, lambda t: sp.expand(TR8(t)))
    if spread is None:
        return total
    terms = sp.Add.make_args(spread)
    kept = [term for term in terms if term.has(index)]
    return sp.Add(*kept) if kept else total


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
    if not _counted(head):
        split = _split_over_sequence(head)
        if split is not None:
            return split
        rational = _rational_series(head)
        if rational is not None:
            return rational
    value = attempt(head, lambda s: s.doit(deep=False))
    if value is None:
        return _undecided(head)
    if _counted(head):
        return value
    if isinstance(value, sp.Sum):
        value = _reflected(head) or _geometric_series(head) or value
    value = _exponential_integral_series(head, value)
    value = _over_the_counts(head, value)
    if isinstance(value, Vector):
        # Each element of a vector sum is its own closed form: `SUM([k, k^2], k, 0, n)`
        # is a vector of the two factored formulas.
        entries = attempt(value, lambda v: v.func(*map(_factored, v.args)))
        return value if entries is None else entries
    return _factored(value)


def _geometric_series(head: sp.Sum) -> sp.Basic | None:
    """An infinite sum whose terms have a constant ratio `r`, which sympy misses where
    the powers have different bases: `SUM((3/4)^k*#e^(k*t), k, 0, inf)` has `r =
    3*#e^t/4`. It is `IF(ABS(r) < 1, u0/(1 - r))`, `u0` the first term, and diverges
    otherwise, where the sum stands. None if the terms aren't geometric."""
    if len(head.limits) != 1:
        return None
    term = head.function
    index, low, high = head.limits[0]
    if high is not sp.oo or not low.is_finite or not term.has(index):
        return None
    try:
        following = term.subs(index, index + 1)
        ratio = sp.powsimp(sp.expand_power_exp(following / term), force=True)
        if ratio.has(index) or ratio.is_zero is not False:
            return None
        # The forced combination of powers is checked exactly.
        if sp.simplify(following - ratio * term) != 0:
            return None
        first = term.subs(index, low)
        return sp.Piecewise((first / (1 - ratio), sp.Abs(ratio) < 1), (head, True))
    except Exception:
        return None


def _reflected(head: sp.Sum) -> sp.Basic | None:
    """The sum with its terms in reverse order, `i` going to `a + b - i`, when that has
    a closed form the sum as written doesn't get, or None.

    `SUM(COMB(n - i - 1, n - i), i, 0, n)` is `SUM(COMB(k - 1, k), k, 0, n)`, which is
    1. The range must be known not to be empty, `b - a` a nonnegative integer, for the
    closed form to hold for every value.
    """
    if len(head.limits) != 1:
        return None
    index, low, high = head.limits[0]
    length = high - low
    if not (length.is_integer and length.is_nonnegative):
        return None
    reflected = sp.Sum(head.function.subs(index, low + high - index), head.limits[0])
    value = attempt(reflected, lambda s: s.doit(deep=False))
    if value is None or value.has(sp.Sum):
        return None
    return value


def _exponential_integral_series(head: sp.Sum, value: sp.Basic) -> sp.Basic:
    """`value`, sympy's closed form of the sum `head`, with its exponential integrals
    made right for every value of their argument, or `head` if they can't be.

    Sympy sums `SUM(x^k/(k*k!), k, 1, inf)` to `EI(x) - LN(x) - euler_gamma`, which holds
    for positive `x` only: for negative `x` it is off by `pi*#i`, as the series is real,
    and the right form is `EI(x) - LN(-x) - euler_gamma`; at 0 it is undefined, while the
    series is 0. So a logarithm of an `EI` argument known negative is taken of its
    negation, and an argument of unknown sign leaves the sum standing. A number argument
    comes on the polar plane, `EI(exp_polar(pi*#i))` for -1, which is `EI(-1) + pi*#i`
    on the real line.
    """
    if not isinstance(value, sp.Basic) or not value.has(sp.Ei):
        return value
    half_turn = sp.exp_polar(sp.I * sp.pi)

    def off_the_plane(e: sp.Basic) -> sp.Basic:
        size, turn = e.args[0].as_independent(sp.exp_polar, as_Add=False)
        if turn != half_turn or not size.is_positive:
            return e
        if isinstance(e, sp.Ei):
            return sp.Ei(-size) + sp.I * sp.pi
        return sp.log(size) + sp.I * sp.pi

    def polar(e: sp.Basic) -> bool:
        return isinstance(e, (sp.Ei, sp.log)) and e.args[0].has(sp.exp_polar)

    value = attempt(value, lambda v: v.replace(polar, off_the_plane))
    if value is None or value.has(sp.exp_polar):
        return head
    negated = {}
    for call in value.atoms(sp.Ei):
        argument = call.args[0]
        if argument.is_positive or argument.is_number and argument.is_negative:
            continue
        if not argument.is_extended_negative:
            return head
        negated[sp.log(argument)] = sp.log(-argument)
    return value.xreplace(negated) if negated else value


def _rational_series(head: sp.Sum) -> sp.Basic | None:
    """An infinite sum of a rational function with rational coefficients and simple
    rational poles, by the digamma function `psi`, or None if the summand isn't one.

    In partial fractions such a summand is a sum of `c_i/(k - p_i)`, and if its
    denominator has degree at least two more than its numerator, the series converges
    absolutely and the `c_i` add up to zero. As `SUM(1/(k - p) - 1/(k - q), k, a, inf)`
    is `psi(a - q) - psi(a - p)`, the sum is `-SUM(c_i*psi(a - p_i))`, and Gauss's
    theorem writes `psi` at a rational point in logarithms and `pi`:
    `SUM(1/((3*k + 1)*(3*k + 2)*(3*k + 3)), k, 0, inf)` is `SQRT(3)*pi/12 - LN(3)/4`.

    Sympy splits such a sum into divergent parts, and for `SUM(1/(k*(2*k + 1)), k, 1,
    inf)` answers `nan` from `inf - inf`. A pole in the range makes the sum undefined:
    `SUM(1/(k^2 - 1), k, 0, inf)` is `?`.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 3:
        return None
    index, low, high = head.limits[0]
    if high != sp.oo or not low.is_Integer or head.function.free_symbols != {index}:
        return None
    if not head.function.is_rational_function(index):
        return None
    try:
        numerator, denominator = sp.fraction(sp.cancel(head.function))
        top, bottom = sp.Poly(numerator, index), sp.Poly(denominator, index)
    except sp.PolynomialError:
        return None
    if not (top.domain.is_QQ or top.domain.is_ZZ) or bottom.degree() < top.degree() + 2:
        return None
    if not (bottom.domain.is_QQ or bottom.domain.is_ZZ):
        return None
    poles = sp.roots(bottom)
    if sum(poles.values()) != bottom.degree() or any(
        times != 1 or not pole.is_Rational for pole, times in poles.items()
    ):
        return None
    if any(pole.is_Integer and pole >= low for pole in poles):
        return sp.nan
    total = sp.S.Zero
    for pole in poles:
        residue = sp.cancel(numerator / sp.diff(denominator, index)).subs(index, pole)
        total -= residue * sp.digamma(low - pole)
    closed = attempt(
        total, lambda t: sp.expand(sp.expand_log(sp.expand_func(t), force=True))
    )
    if closed is None or closed.has(sp.polygamma, sp.EulerGamma):
        return None
    return closed


def _split_over_sequence(head: sp.Sum) -> sp.Basic | None:
    """A sum over the elements of an unknown sequence, split into a sum per term.

    `SUM((y SUB i - b)^2, i, 1, n)` is `SUM(y SUB i^2, i, 1, n) - 2*b*SUM(y SUB i, i, 1,
    n) + b^2*n`: the summand is expanded, each term's factors free of the index are
    taken out, and the terms without a subscript get their closed forms. None if the
    summand has no subscript by the index, or splitting leaves it as it was.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 3:
        return None
    index = head.limits[0][0]
    if not any(index in part.args for part in head.function.atoms(Subscript)):
        return None
    expanded = attempt(head.function, sp.expand)
    if expanded is None:
        return None
    total = []
    for term in sp.Add.make_args(expanded):
        factor, varying = term.as_independent(index, as_Add=False)
        part = sp.Sum(varying, head.limits[0])
        if not varying.atoms(Subscript):
            part = _summation(part)
        total.append(factor * part)
    if len(total) == 1 and total[0] == head:
        return None
    return sp.Add(*total)


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


def _limit_at_both_infinities(head: sp.Limit) -> sp.Basic:
    """`LIM(u, x, +-inf)`: the limit where `x` grows without bound in either direction,
    which exists where the limits at `inf` and at `-inf` are the same. `LIM(-c*(y -
    b)/b, b, +-inf)` is `c`. Otherwise the limit stands."""
    expression, variable = head.args[:2]
    sides = []
    for point, direction in ((sp.oo, "-"), (sp.S.NegativeInfinity, "+")):
        side = _limit(sp.Limit(expression, variable, point, dir=direction))
        if side.has(sp.Limit, sp.nan, sp.AccumBounds):
            return head
        sides.append(side)
    first, second = sides
    if first == second or sp.expand(first - second) == 0:
        return first
    return head


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

    Where the expression is continuous at the point, its value there is the limit, and
    is found by substituting (see `limit_by_substitution`). Evaluation does this for a
    written `LIM`; this covers the rest, such as a prime at a point and limits sympy
    leaves in its answers, and spares sympy's series code, which may spend minutes on
    an integral it can't do.
    """
    expression, variable, point, direction = head.args
    if _nowhere_defined(expression):
        return sp.nan
    if not expression.has(variable):
        # Also covers what is not a number, such as a truth value.
        return expression
    if point is sp.zoo:
        return _limit_at_both_infinities(head)
    by_cases = _limit_by_cases(expression, variable, point, str(direction))
    if by_cases is not None:
        return by_cases
    vanishing = _power_of_a_vanishing_wave(expression, variable, point)
    if vanishing is not None:
        return vanishing
    substituted = limit_by_substitution(expression, variable, point)
    if substituted is not None:
        return substituted
    if moving_integrals(expression, {variable: point}):
        # Sympy would substitute into the integrand.
        return head
    geometric = _geometric_terms_limit(expression, variable, point)
    if geometric is not None:
        return geometric
    closed = _closed_forms(expression)
    constant, varying = sp.S.One, closed
    try:
        factor, rest = closed.as_independent(variable, as_Add=False)
    except Exception:
        factor = sp.S.One
    if factor != 1 and _finite_nonzero(factor):
        constant, varying = factor, rest
    varying = _symbolic_powers_combined(varying, variable)
    varying = _constant_powers_combined(varying, variable)
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
    if value is sp.zoo and str(direction) == "+-":
        value = _unsigned_infinity(asked)
    if (value is None or value == asked) and point is sp.oo:
        value = _limit_of_error_functions(varying, variable)
    if value is None or value == asked:
        return head
    if value.has(sp.Limit) and asked != head:
        return head
    value = _bounds_settled(constant * value)
    if value.has(sp.gamma) and value.is_number:
        value = forced(value, sp.simplify)
    split = attempt(value, lambda v: _split_on_a_sign(v, head))
    if split is not None:
        return split
    if any(_complex_sign(found) for found in value.atoms(sp.sign)):
        # Sympy's limit of `b^n` at `inf` is `#e^(inf*SIGN(LN(b)))`, which means
        # something only for `b > 0`: the limit stands where nothing decides it.
        return head
    return value


def _complex_sign(found: sp.sign) -> bool:
    """Whether the sign `found` may be of a complex number: of one sympy knows not to be
    real, or of an expression holding a logarithm of a number not proved positive."""
    (argument,) = found.args
    return argument.is_extended_real is False or any(
        logarithm.args[0].is_extended_positive is not True
        for logarithm in argument.atoms(sp.log)
    )


def _geometric_terms_limit(
    expression: sp.Basic,
    variable: sp.Basic,
    point: sp.Basic,
    context: Context | None = None,
) -> sp.Basic | None:
    """The limit at `inf` of a sum whose terms in `variable` are each
    `c*n^p*r^(k*n + d)`, with `c`, `r` and `d` free of `n`, `p` and `k` rationals, `k`
    nonzero, and `ABS(r^k)` proved below or above 1, by sympy or by the declared
    intervals of `context`: `r = -(SQRT(1 - 4*m) + 1)/2` has `ABS(r) < 1` for `m` in
    `(0, 1/4]`, not in `[0, 1/4]`. At `-inf` it is the limit at `inf` with `n` negated.

    Terms with `ABS(r^k) < 1` vanish, whatever `c` is, and the limit is the sum of the
    other terms: `LIM(a + b*(-3/4)^n, n, inf)` is `a`. Otherwise a single term with
    `c` proved finite and nonzero, beside finite terms free of `n`, decides the limit:
    with `r^k > 1` it is `inf` with the sign of `c`, and with `r^k < -1` the term swings
    between ever larger values of both signs, so the limit is `±inf`, as
    `LIM((-2)^n, n, inf)` is. None where a term is of another form or the comparison is
    not proved."""
    if point is sp.S.NegativeInfinity:
        expression, point = expression.subs(variable, -variable), sp.oo
    if point is not sp.oo or not expression.has(variable):
        return None
    try:
        terms = sp.Add.make_args(sp.expand(expression, deep=False, mul=True))
        kept, growing = [], []
        for term in terms:
            if not term.has(variable):
                kept.append(term)
                continue
            factor, varying = term.as_independent(variable, as_Add=False)
            ratio = _geometric_ratio(varying, variable)
            if factor.has(sp.oo, sp.zoo, sp.nan) or ratio is None:
                return None
            if _proved(ratio, context, "vanishes"):
                continue
            if not (
                _proved(ratio, context, "swings") or _proved(ratio, context, "grows")
            ):
                return None
            growing.append((factor, ratio))
        if not growing:
            return sp.Add(*kept)
        if len(growing) > 1 or not all(term.is_finite for term in kept):
            return None
        ((factor, ratio),) = growing
        if not (factor.is_finite and factor.is_zero is False):
            return None
        if _proved(ratio, context, "swings"):
            return sp.zoo
        return sp.oo * sp.sign(factor) if factor.is_extended_real else None
    except Exception:
        return None


def _geometric_ratio(varying: sp.Basic, variable: sp.Basic) -> sp.Basic | None:
    """The ratio per step of `varying`, a product of powers `r^(k*n + d)` and at most
    one power `n^p`, as the product of the `r^k`; None if it is of another form or has
    no `r^k` other than 1."""
    ratio, polynomial = sp.S.One, False
    for factor in sp.Mul.make_args(varying):
        base, exponent = factor.as_base_exp()
        if base == variable and exponent.is_rational and not polynomial:
            polynomial = True
            continue
        if base.has(variable):
            return None
        slope = exponent.as_independent(variable, as_Add=True)[1]
        step = sp.expand(slope / variable)
        if step.has(variable) or not step.is_rational or step == 0:
            return None
        ratio *= base**step
    return None if ratio == 1 else ratio


def _proved(ratio: sp.Basic, context: Context | None, how: str) -> bool:
    """Whether a power with `ratio` per step is proved to vanish (`ABS(ratio) < 1`),
    to swing (`ratio < -1`) or to grow (`ratio > 1`), by the assumptions or by the
    declared intervals of `context`."""
    if how == "vanishes" and (sp.Abs(ratio) - 1).is_negative:
        return True
    if how == "swings" and (ratio + 1).is_negative:
        return True
    if how == "grows" and (ratio - 1).is_positive:
        return True
    if context is None or ratio.is_extended_real is False:
        return False

    def holds(test: sp.Basic) -> bool:
        # The assumptions may settle a side on their own: `-q < 1` is true for `q`
        # declared positive.
        return test is sp.true or bool(decided(test, context))

    if how == "vanishes":
        return holds(sp.Lt(ratio, 1)) and holds(sp.Gt(ratio, -1))
    if how == "swings":
        return holds(sp.Lt(ratio, -1))
    return holds(sp.Gt(ratio, 1))


def _limit_of_error_functions(
    expression: sp.Basic, variable: sp.Basic
) -> sp.Basic | None:
    """The limit at `inf` of a polynomial in error functions `ERF(c*x + d)`, with `x`
    the variable and the constant `c` in the sector `ABS(ARG(c)) <= pi/4`, or None if
    `expression` is no such polynomial.

    `ERF(w)` tends to 1 as `w` goes to infinity within that sector, also on its edges,
    where the Fresnel integrals converge to 1/2, so the polynomial tends to its value
    with each error function 1. Sympy takes no such limit for a complex `c`.
    """
    functions = [
        found
        for found in expression.atoms(sp.erf)
        if found.has(variable)
    ]
    if not functions:
        return None
    for function in functions:
        try:
            slope, rest = sp.Poly(function.args[0], variable).all_coeffs()
        except Exception:
            return None
        if rest.has(variable) or slope.has(variable):
            return None
        real, imaginary = slope.as_real_imag()
        if not (real.is_positive and (real - sp.Abs(imaginary)).is_nonnegative):
            return None
    placeholders = [sp.Dummy() for _ in functions]
    polynomial = expression.xreplace(dict(zip(functions, placeholders, strict=True)))
    if polynomial.has(variable):
        return None
    try:
        sp.Poly(polynomial, *placeholders)
    except sp.PolynomialError:
        return None
    value = polynomial.xreplace(dict.fromkeys(placeholders, sp.S.One))
    return value if value.is_finite else None


def _power_of_a_vanishing_wave(
    expression: sp.Basic, variable: sp.Basic, point: sp.Basic
) -> sp.Basic | None:
    """The limit at an infinite point of `w^e`, with `w` an even power or the absolute
    value of a sine or cosine of a linear argument and `e` tending to 0 from above:
    `INTERVAL(0, 1)`, or None if it isn't such a power.

    Sympy answers 1 for `(SIN(x)^2)^(1/x)` at `inf`, but the power is 0 at every zero of
    the sine, and takes every value between there and 1 near each of them.
    """
    if not point.is_infinite or not expression.is_Pow:
        return None
    base, exponent = expression.args
    if isinstance(base, sp.Abs):
        wave = base.args[0]
    elif base.is_Pow and base.exp.is_Integer and base.exp.is_even and base.exp > 0:
        wave = base.base
    else:
        return None
    if not isinstance(wave, (sp.sin, sp.cos)):
        return None
    argument = wave.args[0]
    if not (argument.is_polynomial(variable) and sp.degree(argument, variable) == 1):
        return None
    if argument.free_symbols != {variable}:
        return None
    tends = attempt(exponent, lambda e: sp.limit(e, variable, point))
    if tends != 0:
        return None
    side = attempt(exponent, lambda e: sp.limit(sp.sign(e), variable, point))
    if side != 1:
        return None
    return sp.AccumBounds(0, 1)


def _nowhere_defined(expression: sp.Basic) -> bool:
    """Whether `expression` is a quotient whose denominator is identically zero, like
    `(TAN(x)^2 + 1 - SEC(x)^2)/(SIN(x)^2 + COS(x)^2 - 1)`, so it has no limit anywhere.

    Sympy's limit can spend minutes on such a quotient, which Simplify writes as `?`.
    Other denominators that vanish identically sympy cancels by itself; trigonometric
    ones are checked with the Pythagorean identity.
    """
    if not isinstance(expression, (sp.Mul, sp.Pow)):
        return False
    denominator = attempt(expression, lambda e: sp.fraction(sp.together(e))[1])
    if denominator is None or not denominator.has(TrigonometricFunction):
        return False
    vanished = attempt(denominator, lambda d: _pythagorean(sp.expand(d)))
    return vanished is not None and vanished.is_zero is True


def _limit_by_cases(
    expression: sp.Basic, variable: sp.Basic, point: sp.Basic, direction: str
) -> sp.Basic | None:
    """The limit of a function defined by cases, or None if it isn't one or the limit
    can't be found this way.

    From each side, the case that holds all along that side of the point decides the
    limit: for `IF(x < 1, 2*x + 1, x^2)` at 1 that is 1 from above and 3 from below,
    and a two-sided limit whose sides differ is `?`, or `+-u` for sides `u` and `-u`.
    A case holding only at the point, as in `IF(x = 0, 5, x*SIN(1/x))`, plays no part.
    Which case holds is decided exactly, from the set of values each test holds for.
    """
    cases = _by_cases(expression, {variable})
    if cases is None:
        if not expression.has(sp.Piecewise):
            return None
        cases = expression
    folded = attempt(cases, sp.piecewise_fold)
    if not isinstance(folded, sp.Piecewise):
        return None
    if point is sp.S.Infinity:
        sides = ["-"]
    elif point is sp.S.NegativeInfinity:
        sides = ["+"]
    else:
        sides = ["+", "-"] if direction == "+-" else [direction]
    values = []
    for side in sides:
        value = _one_sided_by_cases(folded, variable, point, side)
        if value is None:
            return None
        values.append(value)
    if len(values) == 2 and values[0] != values[1]:
        difference = attempt(values[0] - values[1], sp.expand)
        if difference is None or not difference.is_number or difference.is_zero is None:
            return None
        if not difference.is_zero:
            opposite = attempt(values[0] + values[1], sp.expand)
            return PlusMinus(values[0]) if opposite == 0 else sp.nan
    return values[0]


def _one_sided_by_cases(
    cases: sp.Piecewise, variable: sp.Basic, point: sp.Basic, side: str
) -> sp.Basic | None:
    """The limit of `cases` from one side of `point`, or None (see `_limit_by_cases`)."""
    for arm, test in cases.args:
        holds = _holds_beside(test, variable, point, side)
        if holds is None:
            return None
        if not holds:
            continue
        if arm is sp.nan:
            return sp.nan
        asked = sp.Limit(arm, variable, point, dir=side)
        value = _limit(asked)
        return None if value == asked or value.has(sp.Limit) else value
    return sp.nan


def _holds_beside(
    test: sp.Basic, variable: sp.Basic, point: sp.Basic, side: str
) -> bool | None:
    """Whether `test` holds all along one side of `point` near it (True), nowhere near
    it on that side (False), or neither or undecided (None)."""
    if test is sp.true:
        return True
    if test.free_symbols != {variable} or not variable.is_extended_real:
        return None
    try:
        holding = test.as_set().intersect(sp.S.Reals)
        failing = sp.S.Reals - holding
        if _absent_beside(failing, point, side):
            return True
        if _absent_beside(holding, point, side):
            return False
    except Exception:
        return None
    return None


def _absent_beside(values: sp.Set, point: sp.Basic, side: str) -> bool:
    """Whether `values` has no points on one side of `point` near it, as decided exactly."""
    if point is sp.S.Infinity:
        return values is sp.S.EmptySet or (values.sup < sp.oo) is sp.true
    if point is sp.S.NegativeInfinity:
        return values is sp.S.EmptySet or (values.inf > -sp.oo) is sp.true
    if side == "+":
        near = values.intersect(sp.Interval.open(point, sp.oo))
        return near is sp.S.EmptySet or (near.inf > point) is sp.true
    near = values.intersect(sp.Interval.open(-sp.oo, point))
    return near is sp.S.EmptySet or (near.sup < point) is sp.true


def _symbolic_powers_combined(expression: sp.Basic, variable: sp.Basic) -> sp.Basic:
    """`expression` multiplied out, if it has powers of `variable` to symbolic
    exponents, so that those powers are combined term by term.

    Sympy's limit compares growth factor by factor and does not see powers of the
    variable cancel across a product: ODE.MTH's KUMMER takes `LIM(x^(-2*k - 2)*(...),
    x, inf)` of a sum with terms in `x^(2*k + 2)`, which is finite, and sympy finds it
    infinite.
    """
    try:
        powers = [
            power
            for power in expression.atoms(sp.Pow)
            if power.base == variable and power.exp.free_symbols - {variable}
        ]
        if not powers or sp.count_ops(expression) > _COMBINED_SIZE:
            return expression
        return sp.expand(expression, log=False)
    except Exception:
        return expression


#: The largest expression, in operations, that `_symbolic_powers_combined` multiplies out.
_COMBINED_SIZE = 200


def _constant_powers_combined(expression: sp.Basic, variable: sp.Basic) -> sp.Basic:
    """`expression` with the powers of positive constants in each product combined
    where their exponents vary alike, and those of reciprocal constants cancelled:
    `(SQRT(2) - 1)^(4*n + 4)*(SQRT(2) + 1)^(4*n)` is `(SQRT(2) - 1)^4`, as `a^e*b^e =
    (a*b)^e` for positive `a` and `b`. Sympy's limit compares the growth of each power
    on its own and does not see such powers cancel. Bases are kept as they are rather
    than multiplied out, so the powers left are still of the bases the other terms
    have."""

    def combined(product: sp.Mul) -> sp.Basic:
        others = []
        varying: dict[sp.Basic, dict[sp.Basic, int]] = {}
        for factor in product.args:
            if (
                factor.is_Pow
                and factor.base.is_number
                and factor.base.is_positive
                and factor.exp.has(variable)
            ):
                fixed, moving = factor.exp.as_independent(variable, as_Add=True)
                counts = varying.setdefault(moving, {})
                counts[factor.base] = counts.get(factor.base, 0) + 1
                others.append(factor.base**fixed)
            else:
                others.append(factor)
        if not any(sum(counts.values()) > 1 for counts in varying.values()):
            return product
        for moving, counts in varying.items():
            bases = list(counts)
            for at, base in enumerate(bases):
                for other in bases[at + 1 :]:
                    if counts[base] and counts[other] and _reciprocal(base, other):
                        cancelled = min(counts[base], counts[other])
                        counts[base] -= cancelled
                        counts[other] -= cancelled
            others.extend(base ** (count * moving) for base, count in counts.items())
        return sp.Mul(*others)

    try:
        if not any(
            power.base.is_number and power.exp.has(variable)
            for power in expression.atoms(sp.Pow)
        ):
            return expression
        return expression.replace(lambda found: found.is_Mul, combined)
    except Exception:
        return expression


def _reciprocal(one: sp.Basic, other: sp.Basic) -> bool:
    """Whether the constants `one` and `other` multiply to 1, exactly."""
    return sp.radsimp(sp.expand(one * other)) == 1


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


def _unsigned_infinity(asked: sp.Limit) -> sp.Basic:
    """A two-sided limit sympy answers with unsigned infinity, as Derive answers it.

    Sympy gives unsigned infinity whenever both sides are infinite, but Derive's `±inf`
    is a real infinity of unknown sign, `LIM(1/x, x, 0)`. At a branch point one side
    is imaginary: `1/SQRT(x)` tends to `inf` from the right and to `-#i*inf` from the
    left. That is no value Rederive has, so it is `?`, which is why ODE_APPR.MTH's
    `TAYLOR_ODE1(y^2 + SQRT(x), x, y, 0, 1, 2)` is `?`, as the manual says.
    """
    expression, variable, point = asked.args[:3]
    for direction in ("+", "-"):
        side = attempt(
            asked, lambda a: sp.Limit(expression, variable, point, dir=direction).doit()
        )
        if side is None or side.has(sp.Limit):
            return sp.zoo
        if not _real_infinity(side):
            return sp.nan
    return sp.zoo


def _real_infinity(value: sp.Basic) -> bool:
    """Whether `value` is an infinity of real but possibly unknown sign, such as `inf`
    times the sign of a parameter: `LIM(1/(a*x), x, 0)` from the right is
    `inf*SIGN(1/a)`, and with the other side that is `±inf`, since Derive's `±inf`
    takes in a factor of unknown sign."""
    if value in (sp.oo, sp.S.NegativeInfinity, sp.zoo):
        return True
    if not value.is_Mul:
        return False
    infinite = [factor for factor in value.args if factor in (sp.oo, -sp.oo)]
    others = [factor for factor in value.args if factor not in infinite]
    return len(infinite) == 1 and all(
        isinstance(factor, sp.sign) and _real_where_defined(factor.args[0])
        or factor.is_extended_real and factor.is_zero is False
        for factor in others
    )


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
    return sp.zoo if _real_infinity(right) else PlusMinus(right)


# -- the FACTOR, EXPAND and SOLVE heads ---------------------------------------


def _commanded(
    name: str, values: list, divisors: list, context: Context
) -> sp.Basic:
    """Run `FACTOR`, `EXPAND`, `RATE` or one of the solving functions on its evaluated
    arguments.

    `FACTOR(u, amount, x, y, ...)`, `EXPAND(u, amount, x, y, ...)` and `SOLVE(u, x)`
    are the in-expression forms of those commands, so simplifying them runs the
    command, as in the original.

    The operand is simplified at exact precision, as the command itself would do;
    rounding comes only at the end. An equation in it is not decided, so that it is
    left to solve. The result is wrapped in `Commanded` so later steps don't rewrite
    it, while an enclosing call like `RHS(SOLVE(x^2 = 4, x))` sees its value.

    `divisors` are what the `SOLVE` equation divides by as written, which the simplified
    operand may no longer show.
    """
    exact = context.with_precision(Precision.EXACT)
    args = [
        _transform(value, exact, decide=False) if not place or name == "RATE" else value
        for place, value in enumerate(values)
    ]

    def decide(value: sp.Basic) -> sp.Basic:
        return _transform(value, exact, decide=True)

    match name:
        case "SOLVE":
            vector = isinstance(values[0], Vector) if values else False
            solved = solve_head(args, divisors, context, decide, vector)
        case "SOLUTIONS":
            solved = solutions_head(args, divisors, context, decide)
        case "NSOLVE" | "NSOLUTIONS":
            solved = nsolve_head(args, context, name == "NSOLUTIONS")
        case "RATE":
            return _rate_head(args, context)
        case _:
            return Commanded(_decided_form(_factor_head(name, args, context), exact))
    return Commanded(_command_head(name, args) if solved is None else solved)


def _decided_form(form: sp.Basic, context: Context) -> sp.Basic:
    """An expanded or factored relation decided where its sides decide it, as a relation
    in an argument is: `EXPAND((x + 1)^2 = x^2 + 2*x + 1)` is `true`. One left undecided
    keeps its form."""
    if not isinstance(form, Relational):
        return form
    decided = _stated(form.func, form.lhs, form.rhs, context, True)
    return decided if isinstance(decided, BooleanAtom) else form


def _factor_head(name: str, args: list, context: Context) -> sp.Basic:
    """Run a `FACTOR` or `EXPAND` head, parsing its target, amount and variables.

    After the target come an optional amount word and optional variables, in that order.
    A word that names no valid amount is read as a variable, so `EXPAND(u, Complex)`
    stays an expression rather than an error, Complex being an amount only Factor
    offers. A matrix amount on anything but a matrix leaves the call as written:
    `FACTOR(x^2 - 1, Turing)`.
    """
    target, *rest = args
    strings = {
        found
        for argument in args
        if isinstance(argument, sp.Basic)
        for found in sp.preorder_traversal(argument)
        if isinstance(found, StringLiteral)
    }
    if strings:
        # A string in arithmetic is a variable, as for `SOLVE`: a Groebner basis
        # program expands in the variables "x1", "x2", ... it builds as strings.
        symbols = {string: sp.Symbol(f'"{string.name}"') for string in strings}
        made = _factor_head(
            name, [argument.xreplace(symbols) for argument in args], context
        )
        return made.xreplace({symbol: string for string, symbol in symbols.items()})
    expanding = name == "EXPAND"
    amounts = EXPAND_AMOUNTS if expanding else tuple(Amount)
    amount = None
    variables = []
    for argument in rest:
        named = amount_named(str(argument)) if isinstance(argument, sp.Symbol) else None
        if named is not None and named in amounts and not variables and amount is None:
            amount = named
        elif isinstance(argument, sp.Symbol):
            variables.append(argument.name)
    if expanding:
        return expanded_expression(
            target, amount or DEFAULT_AMOUNT, variables, order=context.order
        )
    if amount in MATRIX_AMOUNTS:
        product = matrix_product(target, amount)
        return _command_head(name, args) if product is None else product
    # A vector's content is pulled out by `FACTOR(v)` and `FACTOR(v, Trivial)`; any
    # other amount factors each element.
    content = amount in (None, Amount.TRIVIAL)
    return factored_expression(
        target, amount or DEFAULT_AMOUNT, variables, content=content
    )


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
    """`COMB(z, w)` as `z!/(w!*(z - w)!)`, except where `z - w` is a negative integer
    and `w` may be 0: the ratio is 0 there, but `COMB(w - 1, w)` is 1 at `w = 0`."""
    below = whole - part
    if below.is_integer and below.is_negative and part.is_zero is not False:
        return sp.binomial(whole, part)
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

    A sum that gains a sum as a term is rebuilt with sympy's evaluation, which
    flattens it and collects like terms, so `p . (q + r) - p . q` is `p . r`. That
    doesn't evaluate the products inside, so `a . a^-1` stays.
    """
    distributed = expression.replace(
        _is_matrix_product, lambda product: product.expand(), simultaneous=False
    )
    return distributed.replace(
        lambda e: isinstance(e, sp.MatAdd)
        and any(isinstance(term, sp.MatAdd) for term in e.args),
        lambda total: sp.MatAdd(*total.args, evaluate=True),
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
#: minus or plus half the distance. `COT` is worked with as the reciprocal of `TAN`,
#: and spelled back at the end (`_cotangents`).
_DEFINED_OVER: dict[type, type] = {
    sp.cot: sp.tan,
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
    to a `SINH` would become exponentials too. Nor are the arguments rewritten, which
    are written already: `COSH(1/COS(x))` keeps its cosine.
    """

    def defined(*args: sp.Basic) -> sp.Basic:
        standing = [sp.Dummy(**_known(arg)) for arg in args]
        written = head(*standing).rewrite(over)
        return written.xreplace(dict(zip(standing, args)))

    return expression.replace(head, defined)


def _over_roots(expression: sp.Basic) -> sp.Basic:
    """A quotient with roots of its variables cancelled (see `cancelled_over_roots`),
    which the normal form, taking `SQRT(y)` and `y` for unrelated, does not see. A sum
    stays a sum: `1/SQRT(x) + 1/x` is not put over `x`."""
    if not isinstance(expression, sp.Expr) or expression.is_Add or not any(
        power.base.is_Symbol and power.exp.is_Rational and not power.exp.is_Integer
        for power in expression.atoms(sp.Pow)
    ):
        return expression
    if sp.count_ops(expression) > _OVER_ROOTS_SIZE or not sp.fraction(
        sp.together(expression)
    )[1].free_symbols:
        return expression
    cancelled = cancelled_over_roots(expression)
    return expression if cancelled is None else cancelled


#: The largest expression, in operations, `_over_roots` cancels.
_OVER_ROOTS_SIZE = 200


def _over_conjugates(expression: sp.Basic) -> sp.Basic:
    """Each quotient whose denominator is `p + q*SQRT(u)` for one radicand `u`, `p` and
    `q` free of its roots, multiplied above and below by the conjugate `p - q*SQRT(u)`
    and cancelled, where that is shorter.

    The denominator is then `p^2 - q^2*u`, without the root, and may cancel against the
    numerator: `(x + 1)/(SQRT(x + 2) + 1)` is `SQRT(x + 2) - 1`, as is a quotient of
    sums of `SQRT(-x^2 + 2*a*x + 3*a^2)` that is a solution of a trigonometric system.
    The roots of `u` are written as powers of one root `r`, reduced by `r^2 = u`, which
    holds for every `u`. The cancelled quotient `N'/D'` then equals `N/D` as a rational
    function of the variables and `r` modulo `r^2 - u`, so `N*D'` is `D*N'` at every
    point, and the two have one value wherever both denominators are nonzero. Where the
    conjugate vanishes but `D` doesn't, `D'` may vanish too: `1/(SQRT(x) + 1)` is
    `(1 - SQRT(x))/(1 - x)`, undefined at `x = 1`. So `D'` must divide `g^2`, `g` the
    greatest common divisor of `p` and `q`, which vanishes only where `D` does; the
    result then only adds points where `D` vanished, as `x/x` is 1. Where the
    rationalized denominator doesn't cancel, the result is longer and is not kept. A
    sum stays a sum, as in `_over_roots`.
    """
    if not isinstance(expression, sp.Expr) or not _radicals(expression):
        return expression
    if sp.count_ops(expression) > _OVER_ROOTS_SIZE:
        return expression

    def rationalized(quotient: sp.Mul) -> sp.Basic:
        found = _over_conjugate(quotient)
        return quotient if found is None or not pays(found, quotient) else found

    return expression.replace(
        lambda e: e.is_Mul and any(_square_roots(f) for f in _denominators(e)),
        rationalized,
    )


def _denominators(product: sp.Mul) -> list[sp.Basic]:
    """The bases of the factors of `product` to a negative integer power."""
    return [
        factor.base
        for factor in product.args
        if factor.is_Pow and factor.exp.is_Integer and factor.exp.is_negative
    ]


def _square_roots(expression: sp.Basic) -> set[sp.Basic]:
    """The radicands of the square roots in `expression` that aren't numbers."""
    return {
        power.base
        for power in _radicals(expression)
        if power.exp.q == 2 and not power.base.is_number
    }


def _over_conjugate(quotient: sp.Mul) -> sp.Basic | None:
    """`quotient` multiplied by the conjugate of its denominator and cancelled (see
    `_over_conjugates`), or None if the denominator isn't linear in one root."""
    numerator, denominator = sp.fraction(sp.together(quotient))
    radicands = _square_roots(denominator)
    if len(radicands) != 1:
        return None
    (radicand,) = radicands
    root = sp.Dummy("root")

    def written(part: sp.Basic) -> sp.Poly | None:
        part = part.replace(
            lambda e: e.is_Pow
            and e.base == radicand
            and e.exp.is_Rational
            and e.exp.q == 2,
            lambda e: root ** int(2 * e.exp),
        )
        if any(
            power.base == radicand and not power.exp.is_Integer
            for power in part.atoms(sp.Pow)
        ):
            return None
        return sp.Poly(part, root).rem(sp.Poly(root**2 - radicand, root))

    above, below = written(numerator), written(denominator)
    if above is None or below is None or below.degree() != 1:
        return None
    p, q = below.coeff_monomial(1), below.coeff_monomial(root)
    conjugate = sp.Poly(p - q * root, root)
    rationalized = sp.expand(p**2 - q**2 * radicand)
    if rationalized == 0:
        return None
    raised = (above * conjugate).rem(sp.Poly(root**2 - radicand, root)).as_expr()
    cancelled = sp.cancel(raised / rationalized, extension=True)
    remaining = sp.fraction(cancelled)[1]
    if remaining.has(root):
        return None
    common = sp.gcd(p, q, extension=True)
    if sp.fraction(sp.cancel(common**2 / remaining, extension=True))[1].free_symbols:
        return None
    return cancelled.xreplace({root: sp.sqrt(radicand)})


def _radicals(expression: sp.Basic) -> set[sp.Basic]:
    """The roots in `expression`: powers to a fraction that isn't an integer."""
    return {
        power
        for power in expression.atoms(sp.Pow)
        if power.exp.is_Rational and not power.exp.is_Integer
    }


def _conjugate_products(expression: sp.Basic) -> sp.Basic:
    """Two sums in a product multiplied out where that leaves fewer roots and is shorter.

    `(SQRT(x + 1) - SQRT(x))*(SQRT(x + 1) + SQRT(x))` is `1` and `(SQRT(a) + 1)*(SQRT(a)
    - 1)` is `a - 1`: factors conjugate over a root multiply to a difference of squares,
    which the normal form, keeping a product of sums as it is, doesn't form. Multiplying
    out is ring arithmetic and `SQRT(u)^2` is `u` for every `u`, so this holds wherever
    the factors are defined. Factors are paired in equal integer powers, so a
    denominator `1/((SQRT(a) + 1)*(SQRT(a) - 1))` pairs too. A product of sums without
    roots, such as `(x - 1)*(x + 1)`, stays factored.
    """

    def paired(product: sp.Mul) -> sp.Basic:
        factors = [factor.as_base_exp() for factor in product.args]
        for i, (first, power) in enumerate(factors):
            if not first.is_Add or not power.is_Integer or not _radicals(first):
                continue
            for j in range(i + 1, len(factors)):
                second, other = factors[j]
                if not second.is_Add or other != power or not _radicals(second):
                    continue
                written = sp.Mul(first, second)
                multiplied = sp.expand(written)
                roots = _radicals(first) | _radicals(second)
                if len(_radicals(multiplied)) >= len(roots) or sp.count_ops(
                    multiplied
                ) >= sp.count_ops(written):
                    continue
                rest = [b**e for k, (b, e) in enumerate(factors) if k not in (i, j)]
                return paired_or_not(sp.Mul(multiplied**power, *rest))
        return product

    def paired_or_not(node: sp.Basic) -> sp.Basic:
        return paired(node) if node.is_Mul else node

    if not isinstance(expression, sp.Expr) or not _radicals(expression):
        return expression
    if sp.count_ops(expression) > _OVER_ROOTS_SIZE:
        return expression
    return expression.replace(lambda e: isinstance(e, sp.Mul), paired)


def _over_surds(expression: sp.Basic) -> sp.Basic:
    """A quotient cancelled over the field its numeric roots span.

    `(x^3 + (SQRT(2) - 2)*x^2 - (2*SQRT(2) + 3)*x - 3*SQRT(2))/(x^2 - 2)` has the
    factor `x + SQRT(2)` above and below, which cancelling over the rationals can't
    see; over `Q(SQRT(2))` it is `(x^2 - 2*x - 3)/(x - SQRT(2))`. Only a cancelled
    factor is taken: a denominator written as a product, `(x + 2^(1/3))*(x^2 - 2^(1/3)*x
    + 2^(2/3))`, isn't multiplied out. A sum stays a sum, as in `_over_roots`.
    """
    if not isinstance(expression, sp.Expr) or expression.is_Add:
        return expression
    surds = {
        power
        for power in expression.atoms(sp.Pow)
        if power.base.is_Rational and power.exp.is_Rational and not power.exp.is_Integer
    }
    if not surds or len(surds) > _SURDS or sp.count_ops(expression) > _OVER_ROOTS_SIZE:
        return expression
    denominator = sp.fraction(sp.together(expression))[1]
    if not denominator.free_symbols:
        return expression
    cancelled = sp.cancel(expression, extension=True)
    return cancelled if _lowered(denominator, sp.fraction(cancelled)[1]) else expression


def _lowered(before: sp.Basic, after: sp.Basic) -> bool:
    """Whether denominator `after` has lower degree than `before`: whether cancelling
    removed a factor, rather than only multiplying the factors out."""
    symbols = sorted(before.free_symbols, key=str)
    return sp.Poly(after, *symbols).total_degree() < sp.Poly(before, *symbols).total_degree()


#: The most numeric roots `_over_surds` cancels over.
_SURDS = 2


def _over_exponentials(expression: sp.Basic) -> sp.Basic:
    """A quotient with exponentials of fractional multiples of one exponent cancelled.

    The normal form takes `#e^x` and `#e^(x/2)` for unrelated, so `(#e^x - 1)/(#e^(x/2)
    + 1)` stays. Writing every `#e^(c*u)` as `t^(c/g)`, with `t = #e^(g*u)` and `g` the
    greatest common measure of the `c`, makes the quotient rational in `t`, and
    cancelled it is `#e^(x/2) - 1`. Likewise for powers `b^(c*u)` of a positive number
    `b`, so `(4^x - 1)/(2^x + 1)` is `2^x - 1`. As `c/g` is an integer, `#e^(c*u)` is
    `t^(c/g)` whatever `u` is. A sum stays a sum, as in `_over_roots`.
    """
    if not isinstance(expression, sp.Expr) or expression.is_Add:
        return expression
    groups: dict[tuple[sp.Basic, sp.Basic], list[sp.Rational]] = {}
    for found in expression.atoms(sp.exp, sp.Pow):
        key = _exponential_key(found)
        if key is not None:
            groups.setdefault(key[:2], []).append(key[2])
    measures = {}
    for (base, exponent), coefficients in groups.items():
        measure = sp.Rational(
            gcd(*(c.p for c in coefficients)), lcm(*(c.q for c in coefficients))
        )
        if all(c == measure for c in coefficients):
            continue
        measures[base, exponent] = measure
    # Exponentials to cancel, which are rare, before the costlier tests.
    if not measures:
        return expression
    if sp.count_ops(expression) > _OVER_ROOTS_SIZE or not sp.fraction(
        sp.together(expression)
    )[1].free_symbols:
        return expression
    standing = {key: (sp.Dummy("t"), measure) for key, measure in measures.items()}

    def written(found: sp.Basic) -> sp.Basic:
        key = _exponential_key(found)
        if key is None or key[:2] not in standing:
            return found
        dummy, measure = standing[key[:2]]
        return dummy ** int(key[2] / measure)

    rational = expression.replace(
        lambda e: _exponential_key(e) is not None, written, simultaneous=False
    )
    if any(
        _exponential_key(found) is not None and _exponential_key(found)[:2] in standing
        for found in rational.atoms(sp.exp, sp.Pow)
    ):
        return expression
    cancelled = sp.cancel(rational)
    if not _lowered(sp.fraction(sp.together(rational))[1], sp.fraction(cancelled)[1]):
        return expression
    return cancelled.xreplace(
        {
            dummy: sp.Pow(base, measure * exponent)
            for (base, exponent), (dummy, measure) in standing.items()
        }
    )


def _exponential_key(found: sp.Basic) -> tuple[sp.Basic, sp.Basic, sp.Rational] | None:
    """`(b, u, c)` for an exponential `b^(c*u)` with `c` rational, `u` not a number and
    `b` `#e` or a positive number, or None. A perfect power `b` is taken over its root:
    `4^x` is `(2, x, 2)`."""
    if isinstance(found, sp.exp):
        base, exponent = sp.E, found.args[0]
    elif (
        isinstance(found, sp.Pow)
        and found.base.is_number
        and found.base.is_positive is True
        and not found.exp.is_number
    ):
        base, exponent = found.base, found.exp
    else:
        return None
    coefficient, rest = exponent.as_coeff_Mul()
    if not coefficient.is_Rational or rest.is_number:
        return None
    if base.is_Integer:
        perfect = sp.perfect_power(int(base))
        if perfect:
            base, coefficient = sp.Integer(perfect[0]), coefficient * perfect[1]
    return base, rest, coefficient


def _over_sines(expression: sp.Basic) -> sp.Basic:
    """Exponentials of imaginary multiples of real arguments written as cosines and
    sines, by Euler's formula `#e^(#i*t) = COS(t) + #i*SIN(t)`.

    The numerator and the denominator are each first written about the mean of their
    terms' imaginary exponents, so conjugate exponentials meet as `#e^(#i*t)` and
    `#e^(-#i*t)`: `(#e^(2*#i*x) + 1)/#e^(#i*x)` is `2*COS(x)`, and a quotient whose
    imaginary units cancel is real, as in `(#e^(#i*x) - #e^(-#i*x))/(2*#i)`, which is
    `SIN(x)`. Kept by `_realized`.
    """
    if not isinstance(expression, sp.Expr) or not expression.has(sp.I):
        return expression
    if not any(_imaginary_angle(found) is not None for found in expression.atoms(sp.exp)):
        return expression
    if sp.count_ops(expression) > _OVER_ROOTS_SIZE:
        return expression
    numerator, denominator = sp.fraction(sp.together(expression))
    numerator, above = _about_mean_angle(numerator)
    denominator, below = _about_mean_angle(denominator)
    return numerator / denominator * sp.exp(sp.I * (above - below))


def _imaginary_angle(found: sp.Basic) -> tuple[sp.Basic, sp.Basic] | None:
    """`(a, t)` for an exponential `#e^(a + #i*t)` with `a` and `t` real and `t` not
    zero, or None."""
    if not isinstance(found, sp.exp):
        return None
    real, imaginary = [], []
    for term in sp.Add.make_args(sp.expand(found.args[0])):
        if term.is_real:
            real.append(term)
            continue
        angle = term.as_coefficient(sp.I)
        if angle is None or angle.is_real is not True:
            return None
        imaginary.append(angle)
    if not imaginary:
        return None
    return sp.Add(*real), sp.Add(*imaginary)


def _about_mean_angle(total: sp.Basic) -> tuple[sp.Basic, sp.Basic]:
    """`(s, m)` with `total = s*#e^(#i*m)`, `m` the mean imaginary exponent of the terms
    of `total` and `s` written over cosines and sines."""

    def angle(term: sp.Basic) -> sp.Basic:
        found = [_imaginary_angle(factor) for factor in sp.Mul.make_args(term)]
        return sp.Add(*(parts[1] for parts in found if parts is not None))

    def euler(found: sp.Basic) -> sp.Basic:
        real, imaginary = _imaginary_angle(found)
        return sp.exp(real) * (sp.cos(imaginary) + sp.I * sp.sin(imaginary))

    terms = sp.Add.make_args(sp.expand(total))
    mean = sp.Add(*(angle(term) for term in terms)) / len(terms)
    shifted = sp.expand(total * sp.exp(-sp.I * mean))
    written = shifted.replace(lambda e: _imaginary_angle(e) is not None, euler)
    return sp.expand(written), mean


def _unchanged(candidate: sp.Basic, expression: sp.Basic) -> bool:
    """Whether a rewrite left `expression` as it was, which no gate keeps differently
    (and the gates below would measure at length)."""
    try:
        return candidate is expression or candidate == expression
    except Exception:
        return False


def _realized(expression: sp.Basic, rewrite: Rewrite) -> sp.Basic:
    """`rewrite`, if the result is simpler, or free of the imaginary unit and at most
    twice as long: a real quantity written as one."""
    candidate = attempt(expression, rewrite)
    if candidate is None or candidate is expression:
        return expression
    if pays(candidate, expression):
        return candidate
    if not expression.has(sp.I) or candidate.has(sp.I):
        return expression
    try:
        short = sp.count_ops(candidate) <= 2 * sp.count_ops(expression)
    except Exception:
        return expression
    return candidate if short else expression


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
    if candidate is None or _unchanged(candidate, expression):
        return expression
    if pays(candidate, expression):
        return candidate
    try:
        flatter = _nesting(candidate) < _nesting(expression)
        short = sp.count_ops(candidate) <= 2 * sp.count_ops(expression)
    except Exception:
        return expression
    return candidate if flatter and short else expression


def _vanished(expression: sp.Basic, rewrite: Rewrite) -> sp.Basic:
    """`rewrite`, if the result is 0: a test applied only for that outcome."""
    if not isinstance(expression, sp.Add):
        return expression
    candidate = attempt(expression, rewrite)
    return candidate if candidate is not None and candidate == 0 else expression


def _cancelled(expression: sp.Basic, rewrite: Rewrite) -> sp.Basic:
    """`rewrite`, if the result is simpler or clears calls from the denominator.

    A quotient of calls that cancels to a polynomial in them counts as simpler even if
    it isn't shorter, just like `(x^3 - 1)/(x - 1)`: `SIN(3*x)/SIN(x)` becomes
    `2*COS(2*x) + 1`. As with `_unnested`, not if the result more than doubles.
    """
    candidate = attempt(expression, rewrite)
    if candidate is None or _unchanged(candidate, expression):
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


def _fewer_terms(expression: sp.Basic, rewrite: Rewrite) -> sp.Basic:
    """`rewrite`, if the result is simpler, or no longer and with fewer terms once
    multiplied out: a cancellation, which Section 6.3's Auto applies, as in
    `r*(COS(t)^2 - 1)` becoming `-r*SIN(t)^2`. Powers of sums beyond squares aren't
    multiplied out to count."""
    candidate = attempt(expression, rewrite)
    if candidate is None or _unchanged(candidate, expression):
        return expression
    if pays(candidate, expression):
        return candidate
    try:
        if any(
            power.base.is_Add and power.exp.is_Integer and abs(power.exp) > 2
            for power in (*expression.atoms(sp.Pow), *candidate.atoms(sp.Pow))
        ):
            return expression
        fewer = len(sp.Add.make_args(sp.expand(candidate))) < len(
            sp.Add.make_args(sp.expand(expression))
        )
        short = sp.count_ops(candidate) <= sp.count_ops(expression)
    except Exception:
        return expression
    return candidate if fewer and short else expression


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
    hiding the symmetry. Arguments not known to be real are left alone, except that a
    rational function of real variables is real wherever it is defined, so
    `ABS(x/y)^2` is `x^2/y^2`.
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
        rewritten = _even_magnitudes(rewritten)
        if rewritten == expression:
            break
        expression = rewritten
    return _paired_signs(expression)


def _even_magnitudes(expression: sp.Basic) -> sp.Basic:
    """Even powers of `ABS(u)` as powers of `u`, for `u` real wherever it is defined:
    `ABS(x/y)^2` is `x^2/y^2`."""

    def is_even_power(e: sp.Basic) -> bool:
        return (
            isinstance(e, sp.Pow)
            and isinstance(e.base, sp.Abs)
            and e.exp.is_integer is True
            and e.exp.is_even is True
            and _real_where_defined(e.base.args[0])
        )

    return expression.replace(is_even_power, lambda e: e.base.args[0] ** e.exp)


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
    `ABS(x*y*ATAN(x))` is `x*ATAN(x)*ABS(y)`. A factor that can't be negative where it
    is defined comes out too, whatever the other factors are: `ABS(x*(COS(x) + 1))` is
    `(COS(x) + 1)*ABS(x)`, and `ABS(z/ABS(w))` is `ABS(z)/ABS(w)` for complex `z`.
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
        if any(part.has(sp.AccumBounds) for part in inside):
            return product
        # A factor that can't be negative wherever it is defined comes out whatever the
        # others are: `ABS(z/ABS(w)^3)` is `ABS(z)/ABS(w)^3`.
        freed, inside = sp.sift(inside, _never_negative_where_defined, binary=True)
        # A reciprocal isn't known real, being undefined at zero, but its base is.
        if not all(
            part.as_base_exp()[0].is_extended_real
            and (part.is_extended_real or part.as_base_exp()[1].is_Integer)
            for part in inside
        ):
            if not freed:
                return product
            return sp.Mul(*outside, *freed, sp.Abs(sp.Mul(*inside)))
        groups: dict[sp.Basic, list[tuple[sp.Basic, int]]] = {}
        for part in inside:
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


def _never_negative_where_defined(factor: sp.Basic) -> bool:
    """Whether `factor` is never negative where it is defined: an integer power of
    something that can't be negative, or an even one of something real, as `1/ABS(x)`
    and `1/x^2` are for real `x`."""
    if never_negative(factor):
        return True
    base, exponent = factor.as_base_exp()
    if not exponent.is_Integer:
        return False
    return never_negative(base) or (exponent.is_even and base.is_extended_real is True)


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


def _one_spelling_of_radicands(expression: sp.Basic) -> sp.Basic:
    """Write the roots of one radicand spelled two ways with one spelling, the shorter,
    so that they cancel or combine.

    `SQRT((2*d - m + r)*(m + r))/SQRT(2*d*(m + r) - m^2 + r^2)` is 1 where defined: the
    radicands are one polynomial, factored and multiplied out. Only roots of compound
    radicands are compared, and only when two of them have the same variables.

    Radicands that are positive rational multiples of one polynomial are spelled as
    that polynomial, the multiple taken out of the root: `SQRT(3*x^2 - 3)` beside
    `SQRT(x^2 - 1)` is `SQRT(3)*SQRT(x^2 - 1)`. That holds for every value, as a
    positive number has no phase to add to the radicand's. The polynomial is the
    shortest of the radicands that are their own primitive part, or else that part.
    """
    roots = [
        power
        for power in expression.atoms(sp.Pow)
        if power.exp.is_Rational
        and not power.exp.is_Integer
        and power.base.has(sp.Add)
        and not power.base.is_number
    ]
    bases: dict[frozenset, set[sp.Basic]] = {}
    for power in roots:
        bases.setdefault(frozenset(power.base.free_symbols), set()).add(power.base)
    spellings: dict[sp.Basic, list[tuple[sp.Basic, sp.Basic]]] = {}
    for group in bases.values():
        if len(group) < 2:
            continue
        for base in group:
            content, primitive = _positive_content(sp.expand(base))
            spellings.setdefault(primitive, []).append((base, content))
    chosen: dict[sp.Basic, tuple[sp.Basic, sp.Basic]] = {}
    for primitive, same in spellings.items():
        if len(same) < 2:
            continue
        contents = {content for _, content in same}
        if len(contents) == 1:
            spelling, content = min(same, key=lambda pair: _spelling_key(pair[0]))
        else:
            plain = [base for base, content in same if content == 1]
            spelling = min(plain, key=_spelling_key) if plain else primitive
            content = sp.S.One
        chosen.update(
            {
                base: (spelling, multiple / content)
                for base, multiple in same
                if base != spelling
            }
        )
    if not chosen:
        return expression
    return expression.xreplace(
        {
            power: chosen[power.base][1] ** power.exp
            * sp.Pow(chosen[power.base][0], power.exp)
            for power in roots
            if power.base in chosen
        }
    )


def _positive_content(polynomial: sp.Basic) -> tuple[sp.Basic, sp.Basic]:
    """`(c, p)` with `polynomial` equal to `c*p`, `c` a positive rational and `p`
    having integer coefficients with no common factor, or `(1, polynomial)`."""
    content, primitive = polynomial.as_content_primitive()
    if not content.is_Rational or not content.is_positive:
        return sp.S.One, polynomial
    return content, primitive


def _spelling_key(base: sp.Basic) -> tuple:
    return sp.count_ops(base), sp.default_sort_key(base)


def _joined_roots(expression: sp.Basic) -> sp.Basic:
    """Write two square roots in a product as one where their radicands cancel, if
    that holds for every real value of the variables.

    `SQRT(s)*SQRT(1 - 1/s)` is `SQRT(s - 1)` and `SQRT(s - 1)/SQRT((s - 1)/s)` is
    `SQRT(s)`. For real nonzero `x` and `y`, `SQRT(x)*SQRT(y)` is `SQRT(x*y)` unless
    both are negative (then it is `-SQRT(x*y)`), and `SQRT(x)/SQRT(y)` is `SQRT(x/y)`
    unless `x` is positive and `y` negative (then it is `-SQRT(x/y)`). At zero both
    sides vanish or are undefined. So the roots are joined only where that failing
    case is proved impossible: `SQRT(s)*SQRT(1 - 1/s)` qualifies since `1 - 1/s` is
    positive wherever `s` is negative, but `SQRT(a)/SQRT(a/b)` stays, being
    `-SQRT(b)` for positive `a` and negative `b`. A power `x^(k/2)` is `x^((k - 1)/2)`
    times `SQRT(x)` for any `x`, so the root is taken from it. The joined radicand is
    cancelled and must be shorter than the two it replaces; a point where it is
    defined but the roots were not is a removable singularity, as `x/x` is 1.
    """

    def root(factor: sp.Basic) -> bool:
        return (
            factor.is_Pow
            and factor.exp.is_Rational
            and factor.exp.q == 2
            and not factor.base.is_number
            and all(symbol.is_extended_real for symbol in factor.base.free_symbols)
        )

    def joined(product: sp.Mul) -> sp.Basic:
        factors = list(product.args)
        changed = True
        while changed:
            changed = False
            roots = [index for index, factor in enumerate(factors) if root(factor)]
            for i, j in combinations(roots, 2):
                found = _joined_pair(factors[i], factors[j])
                if found is not None:
                    factors[i], factors[j] = found, sp.S.One
                    changed = True
                    break
        return sp.Mul(*factors)

    return expression.replace(
        lambda e: e.is_Mul and sum(1 for factor in e.args if root(factor)) > 1,
        joined,
    )


def _joined_pair(first: sp.Pow, second: sp.Pow) -> sp.Basic | None:
    """`first*second`, two powers to halves, with their square roots joined into one,
    or None where that isn't shorter or isn't proved to hold (see `_joined_roots`)."""
    (x, a), (y, b) = first.args, second.args
    if a < 0 and b > 0:
        (x, a), (y, b) = (y, b), (x, a)
    half = sp.S.Half if a > 0 else -sp.S.Half
    if (a > 0) == (b > 0):
        # SQRT(x)*SQRT(y), or its reciprocal: wrong where both are negative.
        radicand = sp.cancel(x * y)
        held = _never_both_negative(x, y)
    else:
        # SQRT(x)/SQRT(y): wrong where x is positive and y negative.
        radicand = sp.cancel(x / y)
        held = _never_both_negative(-x, y)
    if sp.count_ops(radicand) >= max(sp.count_ops(x), sp.count_ops(y)):
        return None
    if not held:
        return None
    rest = x ** (a - half) * y ** (b - (half if (a > 0) == (b > 0) else -half))
    return rest * radicand**half


def _never_both_negative(p: sp.Basic, q: sp.Basic) -> bool:
    """Whether no real value of the variables makes `p` and `q` both negative.

    Either one never being negative settles it. Otherwise, for rational functions of
    one real variable, the sets where each is negative are found exactly and must not
    meet.
    """
    if never_negative(p) or never_negative(q):
        return True
    symbols = (p + q).free_symbols
    if len(symbols) != 1:
        return False
    (symbol,) = symbols
    if not symbol.is_extended_real:
        return False
    for part in (p, q):
        if not part.is_rational_function(symbol):
            return False
        for side in sp.fraction(sp.cancel(part)):
            polynomial = sp.Poly(side, symbol)
            if not (polynomial.domain.is_QQ or polynomial.domain.is_ZZ):
                return False
            if polynomial.degree() > _NEVER_BOTH_DEGREE:
                return False
    plain = sp.Dummy(real=True)
    below = [
        sp.solve_univariate_inequality(
            sp.StrictLessThan(part.xreplace({symbol: plain}), 0),
            plain,
            relational=False,
            domain=sp.S.Reals,
        )
        for part in (p, q)
    ]
    return sp.Intersection(*below) is sp.S.EmptySet


#: The highest degree of a numerator or denominator whose sign `_never_both_negative`
#: works out.
_NEVER_BOTH_DEGREE = 6


def _radicand_contents(expression: sp.Basic) -> sp.Basic:
    """Take the perfect powers in the numerator of the numeric content of a sum under
    a root out of it, as sympy does for a product: `SQRT(768*x + 1280)` is `16*SQRT(3*x
    + 5)`, like `SQRT(768*x)` is `16*SQRT(3)*SQRT(x)`. The content is positive, so this
    holds for every value of the sum. A denominator stays: `(x^2 + 3*x + 9/4)^(3/2)` is
    no simpler as `(4*x^2 + 12*x + 9)^(3/2)/8`."""

    def root_of_sum(power: sp.Basic) -> bool:
        return (
            isinstance(power, sp.Pow)
            and power.base.is_Add
            and power.exp.is_Rational
            and not power.exp.is_Integer
            and not power.base.is_number
        )

    def taken(power: sp.Pow) -> sp.Basic:
        content, primitive = power.base.as_content_primitive()
        if not content.is_Rational or content <= 0:
            return power
        degree = power.exp.q
        outside = _perfect_part(content.p, degree)
        if outside == 1:
            return power
        rest = content / sp.Integer(outside) ** degree * primitive
        return sp.Integer(outside) ** power.exp.p * sp.Pow(rest, power.exp)

    return expression.replace(root_of_sum, taken)


def _perfect_part(number: int, degree: int) -> int:
    """The largest `k` whose `degree`-th power divides `number`, among the factors found
    by trial division up to `_CONTENT_FACTORS`."""
    part = 1
    for prime, times in sp.factorint(number, limit=_CONTENT_FACTORS).items():
        part *= prime ** (times // degree)
    return part


#: The trial division bound `_perfect_part` factors a root's content with.
_CONTENT_FACTORS = 10**5


def _combined_radicands(expression: sp.Basic) -> sp.Basic:
    """Write the radicand of each square root that holds a denominator as one factored
    quotient, and take the square factors out of the root, where that is shorter.

    `SQRT(100*pi + 100*(9*pi - 110)^2/(11*(220 - 7*pi)))` is `20*SQRT(11)*(pi +
    55)/(11*SQRT(220 - 7*pi))`: over one denominator the numerator is `400*(pi + 55)^2`.
    A square factor `b^2` of a real `b` leaves the root as `ABS(b)`, which holds for
    every value, since the radicand is `b^2` times the rest: `SQRT(5*(5 - 8*w)/(16*(4*w
    - 5)^2) + 15/16)` is `SQRT(15*w^2 - 40*w + 25)/ABS(4*w - 5)`. A factor of both
    numerator and denominator cancels only as far as some of it stays below the line,
    so the root is undefined where it was.
    """

    def candidate(power: sp.Basic) -> bool:
        if not (
            isinstance(power, sp.Pow) and power.exp.is_Rational and power.exp.q == 2
        ):
            return False
        base = power.base
        if base.is_Add:
            return any(not sp.fraction(term)[1].is_Number for term in base.args)
        # A quotient of sums: `SQRT(P/(r1 - r2)^2)`.
        return (
            base.is_Mul
            and base.has(sp.Add)
            and not sp.fraction(base)[1].is_Number
        )

    def combined(power: sp.Pow) -> sp.Basic:
        numerator, denominator = sp.fraction(sp.together(power.base))
        if denominator == 1:
            return power
        outside = sp.S.One
        parts = []
        above = [
            factor.as_base_exp() for factor in sp.Mul.make_args(sp.factor(numerator))
        ]
        below = [
            factor.as_base_exp() for factor in sp.Mul.make_args(sp.factor(denominator))
        ]
        # A factor of both is cancelled as long as some of it stays below the line, so
        # that the root is still undefined where it was: `(1 - 2*d)*(2*d + 3)/(1 -
        # 2*d)^2` is `(2*d + 3)/(1 - 2*d)` under it.
        powers_below = {}
        for base, exponent in below:
            if exponent.is_Integer:
                powers_below[base] = powers_below.get(base, 0) + int(exponent)
        cancelled = {}
        for base, exponent in above:
            if exponent.is_Integer and powers_below.get(base, 0) > 1:
                taken = min(int(exponent), powers_below[base] - 1)
                cancelled[base] = taken
                powers_below[base] -= taken
        numerator = sp.Mul(
            *(base ** (exponent - cancelled.get(base, 0)) for base, exponent in above)
        )
        denominator = sp.Mul(
            *(
                base ** (exponent - cancelled.get(base, 0))
                for base, exponent in below
            )
        )
        for whole, sign in ((numerator, 1), (denominator, -1)):
            rest = sp.S.One
            for factor in sp.Mul.make_args(sp.factor(whole)):
                base, exponent = factor.as_base_exp()
                if exponent.is_Integer and exponent % 2 == 0 and base.is_real is True:
                    outside *= sp.Abs(base) ** (sign * exponent / 2)
                else:
                    rest *= factor
            # The rest multiplied out, or as factored where that is shorter.
            content, primitive = sp.expand(rest).as_content_primitive()
            factored = sp.factor_terms(rest)
            if len(str(factored)) < len(str(primitive)):
                content, primitive = factored.as_content_primitive()
            parts.append((content, primitive))
        (top_number, top), (bottom_number, bottom) = parts
        if bottom.could_extract_minus_sign() or (
            bottom != 1 and top.could_extract_minus_sign()
        ):
            top, bottom = -top, -bottom
        number = top_number / bottom_number
        if top == 1 and outside == 1:
            # Nothing comes out of the root: `SQRT(1/D)` is no better than the
            # reciprocal root the other rules write.
            return power
        if bottom != 1 and bottom.is_positive and bottom_number > 0:
            # The root of the denominator's number apart, so that it can meet the same
            # surd in other terms: `20*SQRT(11)*(pi + 55)/(11*SQRT(220 - 7*pi))`.
            root = sp.sqrt(top_number * top) / (sp.sqrt(bottom_number) * sp.sqrt(bottom))
        else:
            root = sp.sqrt(number * top / bottom)
        found = (outside * root) ** (2 * power.exp)
        # Measured as written, since counting operations misses the fractions nested in
        # a sum under the root: shorter, or no longer by more than a fifth and with no
        # such fraction left.
        before, after = (len(str(form).replace(" ", "")) for form in (power, found))
        if after < before or (
            outside != 1 and 5 * after <= 6 * before and not found.has(*nested(found))
        ):
            return found
        return power

    def nested(expression: sp.Basic) -> list[sp.Basic]:
        return [part for part in expression.atoms(sp.Pow) if candidate(part)]

    return expression.replace(candidate, combined)


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


def _rationalized(expression: sp.Basic) -> sp.Basic:
    """Each constant over square roots of rationals written over a rational denominator,
    where that is shorter, including the constant part of a product.

    Rationalizing the whole expression puts a sum's terms over one denominator, which
    is longer, so the coefficients of a sum's terms are taken one by one: the term
    `(SQRT(5) - 5)*(1/2 - SQRT(5)/2)^n/(5*(SQRT(5) - 1))` is
    `-SQRT(5)*(1/2 - SQRT(5)/2)^n/5` beside other terms just as it is alone.
    """
    return expression.replace(
        lambda part: isinstance(part, sp.Add | sp.Mul), _rationalized_part
    )


def _rationalized_part(part: sp.Add | sp.Mul) -> sp.Basic:
    if part.is_number:
        return _rationalized_value(part)
    if not isinstance(part, sp.Mul):
        return part
    constants = [factor for factor in part.args if factor.is_number]
    if len(constants) < 2:
        return part
    value = sp.Mul(*constants)
    rational = _rationalized_value(value)
    if rational is value:
        return part
    return rational * sp.Mul(*[factor for factor in part.args if not factor.is_number])


def _rationalized_value(constant: sp.Basic) -> sp.Basic:
    """`constant` with a rational denominator if it is rational arithmetic over square
    roots of positive rationals with a root in a denominator, and that is shorter."""
    generators: set[sp.Basic] = set()
    orders: dict[sp.Pow, int] = {}
    if not _over_radicals(constant, generators, orders) or not generators:
        return constant
    if any(order != 2 or not root.base.is_Rational for root, order in orders.items()):
        return constant
    _, denominator = sp.fraction(sp.together(constant))
    if denominator.is_Rational or len(generators) > _SURDS:
        return constant
    candidate = sp.radsimp(constant)
    if sp.fraction(candidate)[1].is_Rational and sp.count_ops(
        candidate
    ) < sp.count_ops(constant):
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
    two, it is solved. `_quadratic_minimal` finds that polynomial exactly, and when
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
        minimal = _quadratic_minimal(part)
    except Exception:
        return part
    # An answer with coefficients this large is no simpler.
    if minimal is None or max(map(abs, minimal.coeffs())) > _TRIGONOMETRIC_SIZE:
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


def _quadratic_minimal(part: sp.Basic) -> sp.Poly | None:
    """The minimal polynomial of a constant of trigonometric values at rational
    multiples of pi, in `_ROOT`, if it has degree one or two, else None.

    With `z` a primitive `M`-th root of unity, `#e^(#i*pi*p/q)` is a power of `z` when
    `M` is a multiple of `2*q`, and so is `#i` when `M` is a multiple of 4. Each
    trigonometric value is then a rational function of `z`, and so is the constant: an
    element of the cyclotomic field, written exactly as a polynomial in `z` reduced
    modulo the order-`M` cyclotomic polynomial. The constant is rational when that
    polynomial is, and satisfies `x^2 + b*x + c` when its square minus `b` times it is
    the rational `-c`, which is linear algebra on the coefficients. Constants with
    anything else in them, including roots of rationals, give None.

    Sympy's `minimal_polynomial` finds the same answer by composing the minimal
    polynomials of the terms, whose product degree can run into the hundreds, and
    factoring that, which takes seconds for `(SIN(4*pi/25) + COS(4*pi/25))/2`.
    """
    calls = part.atoms(TrigonometricFunction)
    turns = [call.args[0] / sp.pi for call in calls]
    if not turns or not all(turn.is_Rational for turn in turns):
        return None
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
            return None
        written[call] = quotients[call.func]
    value = part.xreplace(written)
    if value.free_symbols != {z} or any(
        not power_.exp.is_Integer for power_ in value.atoms(sp.Pow) if power_.base != z
    ) or value.has(sp.I):
        return None
    cyclotomic = sp.Poly(sp.cyclotomic_poly(order, z), z, domain=sp.QQ)
    numerator, denominator = (
        sp.Poly(side, z, domain=sp.QQ).rem(cyclotomic)
        for side in sp.fraction(sp.together(value))
    )
    if denominator.is_zero:
        return None
    element = (numerator * denominator.invert(cyclotomic)).rem(cyclotomic)
    if element.degree() <= 0:
        return sp.Poly(_ROOT - element.as_expr(), _ROOT)
    square = (element * element).rem(cyclotomic)
    size = cyclotomic.degree()
    first = [element.coeff_monomial(z**k) for k in range(size)]
    second = [square.coeff_monomial(z**k) for k in range(size)]
    leading = next(k for k in range(1, size) if first[k])
    linear = -second[leading] / first[leading]
    if any(second[k] + linear * first[k] for k in range(1, size)):
        return None
    constant = -(second[0] + linear * first[0])
    return sp.Poly(_ROOT**2 + linear * _ROOT + constant, _ROOT)


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
#: wherever defined. `ASEC` and `ACSC` are covered because they are rewritten as `ACOS`
#: and `ASIN` first, and `ACOT(u)` is `pi/2 - ATAN(u)` from the start. The complement of
#: an arctangent is the arctangent of the reciprocal, `_RECIPROCAL` in the table: the
#: pair is a right angle with the sign of its argument.
_RECIPROCAL = "arctangent of the reciprocal"
_COMPLEMENT: dict[type, type | str] = {
    sp.asin: sp.acos,
    sp.acos: sp.asin,
    sp.atan: _RECIPROCAL,
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

    The pair must share both argument and coefficient.
    """
    if not isinstance(expression, sp.Add):
        return None
    arcs: dict[tuple, int] = {}
    for index, term in enumerate(expression.args):
        arc, coefficient = _arc(term)
        if arc is None:
            continue
        arcs[(type(arc), coefficient, arc.args[0])] = index
        if isinstance(arc, sp.atan) and not arc.args[0].is_number:
            arcs.setdefault((_RECIPROCAL, coefficient, 1 / arc.args[0]), index)
    for (func, coefficient, argument), index in arcs.items():
        if func is _RECIPROCAL:
            continue
        other = arcs.get((_COMPLEMENT[func], coefficient, argument))
        if other is None or other == index:
            continue
        if func is sp.atan:
            # For `ATAN(u) + ATAN(1/u)`, either argument gives the sign; use the
            # shorter.
            argument = min(argument, 1 / argument, key=sp.count_ops)
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

    `ATAN(t) + ATAN(1/t)` is `pi/2` for positive `t` and `-pi/2` for negative, i.e.
    `pi*SIGN(t)/2`; at zero the pair is undefined, so a nonnegative argument gets
    `pi/2`. The identity fails for complex arguments (at `#i/2` the sum is `-pi/2`), so
    a pair not known to be real is left as written.
    """
    if func is sp.atan:
        if not _real_where_defined(argument):
            return None
        if argument.is_nonnegative:
            return sp.pi / 2
        if argument.is_negative:
            return -sp.pi / 2
        return sp.pi * sp.sign(argument) / 2
    return sp.pi / 2


#: Limits for combining arctangents of numbers: the number of terms searched for
#: subsets that add up to a multiple of pi (the whole sum is always tried), the integer
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
    diagonal.

    The number of subsets grows exponentially, so beyond `_ARCTANGENTS` terms only the
    whole sum is tried.
    """
    bits = sum(
        abs(multiple) * (max(abs(r.p), r.q).bit_length() + 1) for multiple, r in arcs
    )
    if bits > _ARCTANGENT_BITS:
        return [multiple * sp.atan(r) for multiple, r in arcs]
    terms = []
    left = list(arcs)
    smallest = 2 if len(left) <= _ARCTANGENTS else len(left)
    for size in range(len(left), smallest - 1, -1):
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

#: The largest multiple of an arc of a real rational number that is expanded. Its sine
#: and cosine are a power of a number in a quadratic field, so the value stays small
#: for multiples far above `_ARC_MULTIPLE`.
_RATIONAL_ARC_MULTIPLE = 64


def _arc_multiples(expression: sp.Basic) -> sp.Basic:
    """Expand a sine, cosine or tangent of a multiple of an arc algebraically.

    Sympy writes `SIN(ACOS(x))` as `SQRT(1 - x^2)`, and the multiple-angle formulas
    extend that to `SIN(2*ACOS(x))` = `2*x*SQRT(1 - x^2)`. This holds on the whole
    plane, since the formulas are identities. An arc of a real rational number is
    expanded up to a larger multiple: `SIN(5*ATAN(3/2))` is `-597*SQRT(13)/2197`.
    """
    return expression.replace(
        _is_arc_multiple, _expanded_arc_multiple, simultaneous=False
    )


def _is_arc_multiple(expression: sp.Basic) -> bool:
    if not isinstance(expression, (sp.sin, sp.cos, sp.tan)):
        return False
    multiple, arc = expression.args[0].as_coeff_Mul()
    if not (multiple.is_Integer and multiple > 1):
        return False
    if isinstance(expression, sp.tan):
        if not isinstance(arc, sp.atan):
            return False
    elif not isinstance(arc, _ARCS):
        return False
    if multiple <= _ARC_MULTIPLE:
        return True
    return multiple <= _RATIONAL_ARC_MULTIPLE and _arc_in_a_field(arc) is not None


def _expanded_arc_multiple(expression: sp.Basic) -> sp.Basic:
    multiple, arc = expression.args[0].as_coeff_Mul()
    field = _arc_in_a_field(arc)
    if field is None:
        return sp.expand_trig(expression)
    cosine, sine = _power_of_angle(*field, int(multiple))
    if isinstance(expression, sp.cos):
        return cosine
    if isinstance(expression, sp.sin):
        return sine
    # The cosine isn't zero: a multiple of `ATAN(r)` is an odd multiple of `pi/2` only
    # for `r` of 0 or +/-1, whose arctangents are evaluated.
    return sp.radsimp(sine / cosine)


#: A number `a + b*SQRT(d)` of a quadratic field, as the rationals `a` and `b`.
_Surd = tuple[sp.Rational, sp.Rational]


def _arc_in_a_field(arc: sp.Basic) -> tuple[sp.Rational, _Surd, _Surd] | None:
    """The cosine and sine of an arc of a real rational number, in `Q(SQRT(d))`.

    `ATAN(r)` is the angle of `1 + r*#i`, so its cosine and sine are `1/SQRT(d)` and
    `r/SQRT(d)` with `d = 1 + r^2`. `ACOS(r)` has the cosine `r` and the sine
    `SQRT(1 - r^2)`, and `ASIN(r)` the other way round; both are real only for `r` in
    `[-1, 1]`. Returns `d` and the cosine and sine, or None for any other arc.
    """
    argument = arc.args[0]
    if not isinstance(arc, _ARCS) or not argument.is_Rational:
        return None
    zero, one = sp.S.Zero, sp.S.One
    if isinstance(arc, sp.atan):
        square = 1 + argument**2
        return square, (zero, 1 / square), (zero, argument / square)
    if abs(argument) > 1:
        return None
    square = 1 - argument**2
    if isinstance(arc, sp.acos):
        return square, (argument, zero), (zero, one)
    return square, (zero, one), (argument, zero)


def _power_of_angle(
    square: sp.Rational, cosine: _Surd, sine: _Surd, power: int
) -> tuple[sp.Basic, sp.Basic]:
    """The cosine and sine of `power` times an angle, given the angle's cosine and sine
    in `Q(SQRT(square))`.

    They are the real and imaginary parts of `(cosine + #i*sine)^power`, computed exactly
    in the field by repeated squaring.
    """

    def times(u: _Surd, v: _Surd) -> _Surd:
        return (u[0] * v[0] + u[1] * v[1] * square, u[0] * v[1] + u[1] * v[0])

    def plus(u: _Surd, v: _Surd, sign: int = 1) -> _Surd:
        return (u[0] + sign * v[0], u[1] + sign * v[1])

    def product(z, w):
        return (
            plus(times(z[0], w[0]), times(z[1], w[1]), -1),
            plus(times(z[0], w[1]), times(z[1], w[0])),
        )

    zero, one = sp.S.Zero, sp.S.One
    result = ((one, zero), (zero, zero))
    base = (cosine, sine)
    while power:
        if power & 1:
            result = product(result, base)
        base = product(base, base)
        power >>= 1
    root = sp.sqrt(square)
    return tuple(part[0] + part[1] * root for part in result)


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


def _half_angle_roots(expression: sp.Basic) -> sp.Basic:
    """Write a root of `a*(1 + COS(u))` or `a*(1 - COS(u))` over the half angle.

    The radicand is `2*a*COS(u/2)^2` or `2*a*SIN(u/2)^2`, whose root sympy then takes
    as far as the sign of the half-angle call allows: `SQRT((1 + COS(6))/2)` is
    `-COS(3)`, and `SQRT((1 + COS(y))/2)` is `ABS(COS(y/2))` for a real `y`.
    """
    return expression.replace(_is_half_angle_root, _half_angle_root, simultaneous=False)


def _is_half_angle_root(expression: sp.Basic) -> bool:
    return (
        isinstance(expression, sp.Pow)
        and bool(expression.exp.is_Rational)
        and expression.exp.q == 2
        and isinstance(expression.base, sp.Add)
        and _half_angle_square(expression.base) is not None
    )


def _half_angle_root(power: sp.Pow) -> sp.Basic:
    return _half_angle_square(power.base) ** power.exp


def _half_angle_square(radicand: sp.Add) -> sp.Basic | None:
    """`radicand`, `a + b*COS(u)` with `b = a` or `b = -a`, as a square over `u/2`."""
    cosines = radicand.atoms(sp.cos)
    if len(cosines) != 1:
        return None
    (cosine,) = cosines
    constant, slope = sp.S.Zero, sp.S.Zero
    for term in radicand.args:
        if not term.has(cosine):
            constant += term
            continue
        coefficient = sp.cancel(term / cosine)
        if coefficient.has(cosine):
            return None
        slope += coefficient
    half = cosine.args[0] / 2
    if constant.is_zero is not False or slope.is_zero is not False:
        return None
    if sp.expand(constant - slope) == 0:
        return 2 * constant * sp.cos(half) ** 2
    if sp.expand(constant + slope) == 0:
        return 2 * constant * sp.sin(half) ** 2
    return None


def _phases_of_positive_values(expression: sp.Basic) -> sp.Basic:
    """`ATAN(0, u)`, the angle of `u`, as 0 where `u` is positive and as pi where it is
    negative, for every real value of its variables (see `intervals.always_positive`).
    The angle of a square root's radicand, as `IM` of a product writes it, is such."""
    if not expression.has(sp.atan2):
        return expression

    def angle(call: sp.atan2) -> sp.Basic:
        ordinate, abscissa = call.args
        if ordinate != 0:
            return call
        if always_positive(abscissa):
            return sp.S.Zero
        if always_positive(-abscissa):
            return sp.pi
        return call

    return expression.replace(lambda e: isinstance(e, sp.atan2), angle)


#: The largest power `_powers_of_one_unit` looks for between two bases.
_MOST_UNIT_POWER = 64


def _powers_of_one_unit(expression: sp.Basic) -> sp.Basic:
    """Powers with a variable in the exponent whose bases are integer powers of one
    positive real algebraic number, written as powers of it: `(12*SQRT(2) + 17)^n` is
    `(SQRT(2) + 1)^(4*n)` and `(SQRT(2) - 1)^n` is `(SQRT(2) + 1)^-n`. For a positive
    base, `(u^k)^n = u^(k*n)` holds for every n. The base chosen is the smallest one
    above 1 among those in the expression and their reciprocals; each relation `b =
    u^k` is proved exactly, the size of `k` only guessed from the values."""
    powers = [
        power
        for power in expression.atoms(sp.Pow)
        if not power.exp.is_number
        and power.base.is_number
        and not power.base.is_Rational
        and power.base.is_positive is True
        and power.base.is_algebraic is True
    ]
    bases = list(dict.fromkeys(power.base for power in powers))
    if len(bases) < 2:
        return expression
    try:
        sizes = {base: float(sp.log(base).evalf(30)) for base in bases}
    except (TypeError, ValueError):
        return expression
    candidates = [
        (abs(size), base if size > 0 else sp.radsimp(1 / base))
        for base, size in sizes.items()
        if size != 0
    ]
    if not candidates:
        return expression
    smallest, unit = min(candidates, key=lambda pair: pair[0])
    exponents: dict[sp.Basic, int] = {}
    for base, size in sizes.items():
        guess = round(size / smallest)
        if guess == 0 or abs(guess) > _MOST_UNIT_POWER:
            continue
        if abs(size / smallest - guess) > 1e-9:
            continue
        power = unit ** abs(guess)
        proved = sp.expand(power - base) if guess > 0 else sp.expand(power * base - 1)
        if proved == 0:
            exponents[base] = guess
    if len(exponents) < 2:
        return expression
    written = {
        power: unit ** (exponents[power.base] * power.exp)
        for power in powers
        if power.base in exponents
    }
    unified = list(dict.fromkeys(written.values()))
    # Terms over the same power now combine.
    return expression.xreplace(written).replace(
        lambda e: e.is_Add and e.has(*unified),
        lambda e: sp.collect(sp.expand(e), unified),
    )


def _exponentials_as_waves(expression: sp.Basic) -> sp.Basic:
    """Each sum with `#e^(#i*u)` or `#e^(-#i*u)` and a sine or cosine of `u` in it, the
    exponential written as `COS(u) +- #i*SIN(u)` (Euler's formula, for every complex
    `u`), so that the waves may combine: `#e^(-#i*u) + #i*SIN(u)` is `COS(u)`."""

    def angles(total: sp.Add) -> set[sp.Basic]:
        return {call.args[0] for call in total.atoms(sp.sin, sp.cos)}

    def written(total: sp.Add) -> sp.Basic:
        known = angles(total)

        def wave(power: sp.exp) -> sp.Basic:
            angle = power.args[0] / sp.I
            if angle in known:
                return sp.cos(angle) + sp.I * sp.sin(angle)
            if -angle in known:
                return sp.cos(-angle) - sp.I * sp.sin(-angle)
            return power

        terms = [
            sp.expand(term.replace(lambda e: isinstance(e, sp.exp), wave), mul=True)
            if term.has(sp.exp)
            else term
            for term in total.args
        ]
        return sp.Add(*terms)

    return expression.replace(
        lambda e: e.is_Add and e.has(sp.exp) and e.has(sp.sin, sp.cos), written
    )


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


def _trigsimp(expression: sp.Basic) -> sp.Basic:
    """`trigsimp`, unless the expression is too costly for it.

    Sympy's `futrig` tries its identities on every trigonometric call and on every call
    inside the angles, and the cost grows steeply with calls nested in angles: the
    third Newton step for `COS(x) - x` from 1, a tower of 150 operations, takes
    minutes, while its second step, of 35, takes a third of a second. Such a tower is
    left as it is beyond `_NESTED_TRIGSIMP` operations.

    Nor is it tried on the argument of a built-in that reads its form (see
    `_argument`): which variables such an argument has, or its numerator, are told by
    cancelling, and ODE.MTH's `FREE` asks that of equations in `TAN(x - y + 1)` dozens
    of times.
    """
    if _in_argument.get():
        return expression
    if isinstance(expression, sp.Expr) and not expression.has(
        TrigonometricFunction, HyperbolicFunction
    ):
        # `trigsimp` rewrites only these calls (hyperbolic ones as trigonometric), and
        # rebuilds nothing else, but it would cancel over the whole expression first.
        return expression
    calls = expression.atoms(TrigonometricFunction)
    nested = any(call.args[0].has(TrigonometricFunction) for call in calls)
    if nested and sp.count_ops(expression) > _NESTED_TRIGSIMP:
        return expression
    try:
        angle = _whole_angle(expression)
        unchanged = angle is not None and _without_cancellation(
            expression, sp.cancel(sp.expand_trig(expression)), angle
        )
    except Exception:
        unchanged = False
    return expression if unchanged else sp.trigsimp(expression)


#: The largest expression with trigonometric calls nested in angles that `_trigsimp`
#: takes on, in operations.
_NESTED_TRIGSIMP = 60


def _pythagorean(expression: sp.Basic) -> sp.Basic:
    """A rational function of sines and cosines reduced by `SIN(z)^2 + COS(z)^2 = 1`.

    The numerator and denominator are multiplied out, and every power of a sine beyond
    the first is written over the cosine, or every power of a cosine over the sine,
    whichever is shorter. That is a canonical form for polynomials in the sines and
    cosines of distinct angles, so a sum that is zero by the identity becomes zero
    however large, which `trigsimp` doesn't promise: the length of a vector, which
    TENSOR1.DMO transforms to spherical coordinates and back, keeps `c^2*(COS(theta)^2 +
    SIN(theta)^2 - 1)`.

    Calls of other functions and roots are opaque coefficients, each with its own
    arguments reduced where that pays (see `_fewer_terms`): a radicand is reduced as it
    would be on its own, so `SQRT(4*COS(t)^6 + 4*SIN(t)^6 - 12*SIN(t)^4 + 9*SIN(t)^2 +
    4)` is `SQRT(3*COS(t)^2 + 5)`.
    An unevaluated integral, sum or limit is opaque too, but keeps its operand as
    written, as the rest of Simplify does.
    """
    if not expression.has(sp.sin, sp.cos) or sp.count_ops(expression) > _PYTHAGOREAN:
        return expression
    opaque = {part: sp.Dummy() for part in sp.ordered(_opaque_parts(expression))}
    if not opaque:
        return _pythagorean_reduced(expression)
    inside = {
        part: part
        if isinstance(part, (*_HELD, Antidifference))
        else part.func(*(_fewer_terms(argument, _pythagorean) for argument in part.args))
        for part in opaque
    }
    hidden = expression.xreplace(opaque)
    reduced = _pythagorean_reduced(hidden)
    return reduced.xreplace({dummy: inside[part] for part, dummy in opaque.items()})


def _opaque_parts(expression: sp.Basic) -> set[sp.Basic]:
    """The outermost parts of `expression` with a sine or cosine in them that aren't
    sums, products, integer powers or sines and cosines: calls of other functions,
    roots, symbolic powers."""
    if isinstance(expression, (sp.sin, sp.cos)) or not expression.has(sp.sin, sp.cos):
        return set()
    if isinstance(expression, (sp.Add, sp.Mul)) or (
        isinstance(expression, sp.Pow) and expression.exp.is_Integer
    ):
        return set().union(*(_opaque_parts(argument) for argument in expression.args))
    return {expression}


def _pythagorean_reduced(expression: sp.Basic) -> sp.Basic:
    """`_pythagorean` of an expression whose sines and cosines all stand in its
    arithmetic: its calls of other functions hold none."""
    calls = expression.atoms(sp.sin, sp.cos)
    angles = {call.args[0] for call in calls}
    if not angles:
        return expression
    sines = {angle: sp.Dummy() for angle in sp.ordered(angles)}
    cosines = {angle: sp.Dummy() for angle in sp.ordered(angles)}
    standing = {sp.sin(a): s for a, s in sines.items()}
    standing.update({sp.cos(a): c for a, c in cosines.items()})
    bare = expression.xreplace(standing)
    generators = [*sines.values(), *cosines.values()]
    if bare.has(*calls) or not bare.is_rational_function(*generators):
        return expression
    back = {symbol: call for call, symbol in standing.items()}
    numerator, denominator = sp.fraction(sp.together(bare))
    candidates = []
    for squared, other in ((sines, cosines), (cosines, sines)):
        rules = [(squared[a], 1 - other[a] ** 2) for a in angles]
        parts = [_squares_reduced(part, rules) for part in (numerator, denominator)]
        if None in parts:
            return expression
        if parts != [sp.expand(numerator), sp.expand(denominator)]:
            candidates.append((parts[0] / parts[1]).xreplace(back))
    # Without a square to reduce, the result would only be multiplied out. Of two
    # equally short, the one with fewer terms: `-r*SIN(t)^2`, not `r*COS(t)^2 - r`.
    return (
        min(candidates, key=lambda c: (sp.count_ops(c), len(sp.Add.make_args(c))))
        if candidates
        else expression
    )


def _squares_reduced(
    polynomial: sp.Basic, rules: list[tuple[sp.Basic, sp.Basic]]
) -> sp.Basic | None:
    """`polynomial` multiplied out, with each `symbol^2` replaced by its value until
    every symbol is at most linear in each term; None if too large."""
    polynomial = sp.expand(polynomial)
    if len(sp.Add.make_args(polynomial)) > _PYTHAGOREAN_TERMS:
        return None
    for symbol, square in rules:
        if not polynomial.has(symbol):
            continue
        polynomial = sp.expand(
            polynomial.replace(
                lambda e, s=symbol: e.is_Pow and e.base == s and e.exp.is_Integer
                and e.exp > 1,
                lambda e, s=symbol, q=square: s ** (e.exp % 2) * q ** (e.exp // 2),
            )
        )
        if len(sp.Add.make_args(polynomial)) > _PYTHAGOREAN_TERMS:
            return None
    return polynomial


#: The largest expression `_pythagorean` multiplies out, in operations, and the most
#: terms it lets the product reach.
_PYTHAGOREAN = 400
_PYTHAGOREAN_TERMS = 2000


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
    if _without_cancellation(expression, single, angle):
        return expression
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


def _without_cancellation(
    expression: sp.Basic, single: sp.Basic, angle: sp.Basic
) -> bool:
    """Whether `expression`, a sum of sines and cosines of multiples of `angle` with
    `single` its form over the single angle, has nothing that cancels.

    Section 6.3's Auto applies only the transformations that clearly yield a
    cancellation. A polynomial in the calls is taken to cancel only if its sum of
    multiples, which is unique, has fewer terms than it: `COS(2*x) + COS(4*x)/4` and
    `SIN(2*x) + SIN(4*x)/2` stay as written, rather than becoming `2*COS(x)^4 - 3/4`
    and `4*SIN(x)*COS(x)^3`, while `SIN(x)^6 + COS(x)^6 + 3*SIN(x)^2*COS(x)^2` is 1.
    A quotient with calls in its denominator is left to `_cancelled`.
    """
    if _called_denominator(expression):
        return False
    if not _trigonometric_polynomial(single, sp.sin(angle), sp.cos(angle)):
        return False
    multiples = _sum_of_multiples(sp.expand(single))
    terms = sp.Add.make_args(sp.expand(expression))
    return len(sp.Add.make_args(multiples)) >= len(terms)


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


def _known(expression: sp.Basic) -> dict[str, bool]:
    """What is known of `expression`'s value, as assumptions for a symbol standing for
    it: definitions such as `STEP(x)` as `SIGN(x)/2 + 1/2` hold only for real `x`."""
    facts = ("positive", "negative", "zero", "nonnegative", "nonpositive", "nonzero")
    known = {"integer": True} if expression.is_integer else {}
    if expression.is_extended_real:
        known["real" if expression.is_finite else "extended_real"] = True
    for fact in facts:
        if getattr(expression, f"is_{fact}"):
            known[fact] = True
            break
    return known


def _logs_combined(expression: sp.Basic) -> sp.Basic:
    """`logcombine` of logarithms added to each other, with the power rule
    `k*LN(a) = LN(a^k)` only for rational `k`.

    Only a sum holding two or more logarithmic terms combines; a lone `k*LN(a)` stays
    as written beside other terms, as in Derive: `13*LN(2) - 1` is unchanged while
    `3*LN(2) + LN(5)` is `LN(40)`. The power rule takes only rational `k`, the
    manual's condition, while sympy takes any real `k`: `LN(3)*x + LN(5)` stays as
    written instead of becoming `LN(5*3^x)`. In each term holding one logarithm, every
    factor but the logarithm and a rational is hidden behind a symbol of unknown sign,
    which sympy keeps outside; a product of several logarithms is hidden whole. The sum
    rule is unaffected: `x*LN(2) + x*LN(3)` is `x*LN(6)`.

    The rational must be small too, numerator and denominator at most `_LOG_POWER`,
    and the power of a numeric argument it would build at most `_LOG_BITS` bits: a
    coefficient like `12345678901/98765432109`, as an approximation leaves it, is not
    a power anyone writes, and sympy would spend minutes on its root.

    Each sum is combined on its own: the arguments of its logarithms and its terms
    without one are hidden too, so that `logcombine`, which works on the whole tree,
    does not reach the sums inside them, which were already handled.
    """
    if not expression.has(sp.log):
        return expression

    def combined(node: sp.Basic) -> sp.Basic:
        if node.is_Atom or not node.has(sp.log):
            return node
        node = node.func(*map(combined, node.args))
        if not node.is_Add:
            return node
        terms = sp.Add.make_args(node)
        if sum(1 for term in terms if _log_term(term)) < 2:
            return node
        hidden: dict[sp.Basic, sp.Dummy] = {}

        def hide(part: sp.Basic, **known: bool) -> sp.Dummy:
            return hidden.setdefault(part, sp.Dummy(**known))

        def logarithm(log: sp.log) -> sp.Basic:
            (argument,) = log.args
            return sp.log(hide(argument, **_known(argument)), evaluate=False)

        def prepared(term: sp.Basic) -> sp.Basic:
            if isinstance(term, sp.log):
                return logarithm(term)
            logs = _log_term(term)
            if not logs:
                return hide(term)
            if len(logs) > 1:
                return hide(term)
            (log,) = logs
            factors = []
            for factor in term.args:
                if factor == log:
                    factors.append(logarithm(log))
                elif _small_rational(factor) and _small_power(log.args[0], factor):
                    factors.append(factor)
                else:
                    factors.append(hide(factor))
            return sp.Mul(*factors)

        written = sp.Add(*map(prepared, terms))
        back = {dummy: part for part, dummy in hidden.items()}
        return sp.logcombine(written).xreplace(back)

    return combined(expression)


def _log_arguments_factored(expression: sp.Basic) -> sp.Basic:
    """Each logarithm of a polynomial with a repeated factor, with its argument
    factored: `LN(x*y^4 + 2*x*y^2 + x)` is `LN(x*(y^2 + 1)^2)`. The form shows the
    power the logarithm rules can take out, where they may. A sum of quotients, the
    argument or a factor or base in it, is written as one factored quotient where that
    is not longer but for a sign: `LN((y - 2)/(x + 1) + 1)` is `LN((x + y - 1)/(x +
    1))`, whose parts the rules can take apart."""

    def over_one_denominator(total: sp.Basic) -> sp.Basic:
        if (
            not total.is_Add
            or sp.count_ops(total) > _LOG_FACTORED
            or not total.is_rational_function()
            or all(sp.fraction(term)[1].is_number for term in total.args)
        ):
            return total
        quotient = sp.factor(sp.together(total))
        # A sign taken out in front is no longer form: `(y - 2)/(x + 1) - 1` is
        # `-(x - y + 3)/(x + 1)`.
        sign = 1 if quotient.could_extract_minus_sign() else 0
        allowed = sp.count_ops(total) + sign
        return total if sp.count_ops(quotient) > allowed else quotient

    def factored(log: sp.log) -> sp.Basic:
        (argument,) = log.args
        if not argument.free_symbols:
            return log
        try:
            if not argument.is_Add:
                parts = argument.args if argument.is_Mul else [argument]
                bases = {
                    part.base if part.is_Pow else part
                    for part in parts
                    if (part.base if part.is_Pow else part).is_Add
                }
                combined = {base: over_one_denominator(base) for base in bases}
                return sp.log(argument.xreplace(combined)) if combined else log
            if sp.count_ops(argument) > _LOG_FACTORED:
                return log
            if not argument.is_polynomial():
                return sp.log(over_one_denominator(argument))
            _, factors = sp.factor_list(argument)
        except Exception:
            return log
        if not any(power > 1 for _, power in factors):
            return log
        return sp.log(sp.factor(argument))

    return expression.replace(
        lambda e: isinstance(e, sp.log) and len(e.args) == 1, factored
    )


#: The largest logarithm argument, by operation count, `_log_arguments_factored`
#: factors.
_LOG_FACTORED = 60


def _logs_collected(expression: sp.Basic) -> sp.Basic:
    """`Logarithm := Collect`: the logarithms of a sum gathered into one where the
    rules allow. `k*LN(u)` is `LN(u^k)` for a rational `k` in (-1, 1], whatever `u`
    is, and for any rational `k` where `u` is nonnegative; `LN(u) + LN(v)` is
    `LN(u*v)` where one of them is nonnegative: `LN(x)/2 + LN(y^2 + 1)` is
    `LN(SQRT(x)*(y^2 + 1))`. `2*LN(x)` stays, since `x` may be negative."""

    def powered(term: sp.Basic) -> sp.Basic:
        coefficient, rest = term.as_coeff_Mul()
        if not (isinstance(rest, sp.log) and len(rest.args) == 1):
            return term
        (argument,) = rest.args
        if (
            coefficient.is_Rational
            and _small_rational(coefficient)
            and (-1 < coefficient <= 1 or argument.is_nonnegative)
            and _small_power(argument, coefficient)
        ):
            return sp.log(argument**coefficient)
        return term

    def collected(total: sp.Basic) -> sp.Basic:
        logs, others = [], []
        for term in sp.Add.make_args(total):
            if isinstance(term, sp.log) and len(term.args) == 1:
                logs.append(term.args[0])
            else:
                others.append(term)
        if len(logs) < 2:
            return total
        pending = list(logs)
        merged = [pending.pop(0)]
        while pending:
            argument = pending.pop(0)
            for place, held in enumerate(merged):
                if held.is_nonnegative or argument.is_nonnegative:
                    merged[place] = held * argument
                    break
            else:
                merged.append(argument)
        if len(merged) == len(logs):
            return total
        return sp.Add(*others, *(sp.log(argument) for argument in merged))

    expression = expression.replace(
        lambda e: e.is_Mul and isinstance(e.as_coeff_Mul()[1], sp.log), powered
    )
    return expression.replace(
        lambda e: e.is_Add and sum(1 for t in e.args if t.has(sp.log)) >= 2, collected
    )


def _log_term(term: sp.Basic) -> list[sp.log]:
    """The logarithms that are `term` or its factors."""
    if isinstance(term, sp.log):
        return [term]
    if term.is_Mul:
        return [factor for factor in term.args if isinstance(factor, sp.log)]
    return []


def _small_rational(number: sp.Basic) -> bool:
    return bool(number.is_Rational) and max(abs(number.p), number.q) <= _LOG_POWER


def _small_power(argument: sp.Basic, power: sp.Rational) -> bool:
    """Whether `argument^power` is a number of at most `_LOG_BITS` bits, when the
    argument is a rational."""
    if not argument.is_Rational:
        return True
    bits = max(abs(argument.p), argument.q).bit_length()
    return bits * abs(power.p) <= _LOG_BITS


#: The largest numerator or denominator of a rational the logarithm's power rule takes.
_LOG_POWER = 1000

#: The largest power of a number, in bits, the logarithm's power rule builds.
_LOG_BITS = 4096


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


def _any_extracted_powers(expression: sp.Basic) -> sp.Basic:
    """`_extracted_powers` under Branch Any, which takes `LN(x^2)` to be `2*LN(x)`, with
    no absolute value."""
    return expression.replace(
        lambda e: _extracted_power(e, whatever=True) is not None,
        lambda e: _extracted_power(e, whatever=True),
        simultaneous=False,
    )


def _extracted_power(expression: sp.Basic, whatever: bool = False) -> sp.Basic | None:
    """`k*LN(z)`, if `expression` is a logarithm of `z^k` that may be split.

    Section 6.2 gives `k*LN(x) <-> LN(x^k)` for rational `k` when `x` is nonnegative,
    and for any `x` when `-1 < k <= 1`. An even power of a real `x` equals that power of
    its absolute value, so it is split too: `LN(x^2)` is `2*LN(ABS(x))` and `LN(x^(-2))`
    is `-2*LN(ABS(x))`, following the original.

    A product or a polynomial that is a power as a whole counts as one: `LN(4*x^2)` is
    `2*LN(2*ABS(x))` and `LN(x^2 + 2*x + 1)` is `2*LN(ABS(x + 1))`.

    A factor `#e^u` with real `u` is positive, so its logarithm leaves whatever it
    multiplies: `LN(2*x^3*#e^x)` is `LN(2*x^3) + x`.

    `whatever` splits a power of any base and exponent, as Branch Any does.
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
            split = _extracted_power(remaining, whatever)
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
    if base.is_positive and exponent.is_extended_real and not exponent.is_Rational:
        # A real power of a positive base: `LN(2^(2 - x))` is `(2 - x)*LN(2)`.
        return exponent * sp.log(base)
    if not exponent.is_Rational:
        return None
    if whatever or base.is_nonnegative or bool(-1 < exponent <= 1):
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
    `(x^2 + 2*x + 1)^(3/2)` is `(x + 1)^2*ABS(x + 1)`. The radicand is a positive
    rational times a power of a polynomial, or has such powers among its factors,
    which come out of the root (see `_extracted_root`), and sympy takes the root using
    declared domains: no bars when the base is provably nonnegative (`SQRT(x^4 + 2*x^2 +
    1)` is `x^2 + 1`), and a complex variable keeps its root. A call is a variable of the
    polynomial too: ODE.MTH's `SPECIAL_A_1` takes the root of `4*PHI(x)^2 -
    8*k*PHI(x) + 4*k^2`.

    A quotient of such powers is taken over its denominator when that is an even power,
    and so positive where it is defined: `SQRT(1/x^2 + 2/x + 1)` is `ABS(x + 1)/ABS(x)`.
    Even powers of `ABS` are written out first, since a magnitude of a vector is a root
    of a sum of them.
    """
    expression = _even_magnitudes(expression)
    return expression.replace(_is_polynomial_root, _perfect_root, simultaneous=False)


def _any_powsimp(expression: sp.Basic) -> sp.Basic:
    """Powers over one exponent as one power, whatever the bases, as Branch Any takes
    them: `SQRT(x)*SQRT(y)` is `SQRT(x*y)`.

    Sympy writes a power with a multiple exponent as a power of a power, `(x - r)^(2*k)`
    as `((x - r)^2)^k`, which for real `x` is `ABS(x - r)^(2*k)`. The integer part of
    such an exponent goes back into it.
    """

    def joined(power: sp.Pow) -> sp.Basic:
        base, exponent = power.args
        if base.is_Pow and base.exp.is_Integer:
            return base.base ** (base.exp * exponent)
        powers = [factor.as_base_exp() for factor in base.args]
        if not all(power.is_Integer for _, power in powers):
            return power
        common = gcd(*(int(power) for _, power in powers))
        if common < 2:
            return power
        inner = sp.Mul(*(factor ** (power // common) for factor, power in powers))
        return inner ** (common * exponent)

    # Without assumptions, so that sympy keeps a power of a power for this to join.
    plain = {
        symbol: sp.Dummy(symbol.name) for symbol in sp.ordered(expression.free_symbols)
    }
    combined = sp.powsimp(expression.xreplace(plain), force=True)
    back = {dummy: symbol for symbol, dummy in plain.items()}
    if combined.xreplace(back) == expression:
        return expression
    combined = combined.replace(
        lambda e: (
            e.is_Pow
            and not e.exp.is_Number
            and (
                (e.base.is_Pow and e.base.exp.is_Integer)
                or e.base.is_Mul
            )
        ),
        joined,
    )
    return combined.xreplace(back)


def _any_perfect_powers(expression: sp.Basic) -> sp.Basic:
    """Take roots of polynomials that are perfect powers, as `_perfect_powers` does,
    but whatever the sign of the base, as Branch Any takes `SQRT(x^2)` to be `x`."""
    return expression.replace(
        _is_polynomial_root,
        lambda power: _perfect_root(power, whatever=True),
        simultaneous=False,
    )


def _is_polynomial_root(expression: sp.Basic) -> bool:
    return (
        isinstance(expression, sp.Pow)
        and (
            (bool(expression.exp.is_Rational) and not expression.exp.is_Integer)
            or not expression.exp.is_number
        )
        and (
            isinstance(expression.base, sp.Add)
            or (
                isinstance(expression.base, sp.Mul)
                and any(isinstance(factor, sp.Add) for factor in expression.base.args)
            )
        )
        and len(expression.base.args) <= _PERFECT_TERMS
        and bool(expression.base.free_symbols)
    )


def _perfect_root(power: sp.Pow, whatever: bool = False) -> sp.Basic:
    taken = _whole_root(power, whatever)
    if taken == power:
        taken = _extracted_root(power, whatever)
    return taken


def _whole_root(power: sp.Pow, whatever: bool) -> sp.Basic:
    """The root of `power` whose radicand is a rational times a power of a polynomial
    with rational coefficients, or a quotient of such, or `power` unchanged."""
    roots = _variable_roots(power.base)
    if roots:
        written = power.base.replace(
            lambda e: e.is_Pow and e.base in roots and e.exp.is_Rational,
            lambda e: roots[e.base][0] ** int(e.exp * roots[e.base][1]),
        )
        written = written.xreplace({s: root**q for s, (root, q) in roots.items()})
        taken = _perfect_root(written**power.exp, whatever)
        if taken == written**power.exp:
            return power
        back = {root: s ** sp.Rational(1, q) for s, (root, q) in roots.items()}
        return taken.xreplace(back)
    base, exponent = power.args
    numerator, denominator = sp.fraction(sp.together(base))
    if denominator.is_number:
        numerator, denominator = base, sp.S.One
    top = _as_power(numerator)
    bottom = _as_power(denominator)
    if top is None or bottom is None:
        return power
    multiple = gcd(top[2], bottom[2])
    content = top[0] / bottom[0]
    # Whatever the sign, a negative content's root is taken too: `SQRT(-a^2 + 2*a*b -
    # b^2)` is `#i*(a - b)`.
    if multiple < 2 or not (content.is_positive or whatever and content.is_nonzero):
        return power
    upper = top[1] ** (top[2] // multiple)
    lower = bottom[1] ** (bottom[2] // multiple)
    if denominator != 1:
        # The root of a quotient is the quotient of the roots where the denominator
        # is positive, which an even power with a positive content is wherever it is
        # nonzero: `SQRT((x + y)^2/z^2)` is `ABS(x + y)/ABS(z)`.
        if multiple % 2 or not bottom[0].is_positive or not exponent.is_Rational:
            return power
        wholes = (upper**multiple, lower**multiple)
        taken = content**exponent * wholes[0] ** exponent / wholes[1] ** exponent
        if any(found.base in wholes for found in taken.atoms(sp.Pow)):
            return power
        return taken
    root = upper
    whole = root**multiple
    if whatever:
        # Raised in one step, since sympy writes `((x - 1)^2)^(1/2)` as `ABS(x - 1)`
        # for real `x`.
        taken = content**exponent * root ** (multiple * exponent)
    else:
        taken = content**exponent * whole**exponent
    if any(found.base == whole for found in taken.atoms(sp.Pow)):
        return power
    return taken


def _as_power(polynomial: sp.Basic) -> tuple[sp.Basic, sp.Basic, int] | None:
    """`(c, r, m)` where the polynomial `polynomial` is `c*r^m` for its square-free
    factorization, `m` the largest such multiplicity, or None if it isn't a polynomial
    with rational coefficients of a size worth factoring. A number is `(c, 1, 0)`."""
    if polynomial.is_number:
        return (polynomial, sp.S.One, 0) if polynomial.is_Rational else None
    try:
        terms = sp.Poly(polynomial)
    except sp.PolynomialError:
        return None
    if not terms.domain.is_QQ and not terms.domain.is_ZZ:
        return None
    if terms.total_degree() > _PERFECT_DEGREE or len(terms.terms()) > _PERFECT_TERMS:
        return None
    content, factors = sp.sqf_list(polynomial)
    multiple = reduce(gcd, (int(times) for _, times in factors), 0)
    if multiple < 1:
        return None
    root = sp.Mul(*(factor ** (times // multiple) for factor, times in factors))
    return content, root, multiple


#: The largest degree of the field spanned by a radicand's numeric roots over which
#: its powers are found: two square roots, or one fourth root.
_ROOTS_DEGREE = 4


def _extracted_root(power: sp.Pow, whatever: bool) -> sp.Basic:
    """The `q`-th root of a polynomial with the `q`-th powers of its factors taken out,
    or `power` unchanged.

    `SQRT(x^3 + x^2)` is `ABS(x)*SQRT(x + 1)`, as `SQRT(x^2*(x + 1))` is, since the root
    of a product is the product of the roots where one of the factors is nonnegative,
    as an even power of a real polynomial is. The coefficients may be roots of
    rationals, the factors then found over the field they span: `SQRT(3*x^2 +
    2*SQRT(3)*x*y + y^2)` is `ABS(SQRT(3)*x + y)`. What is left under the root is
    multiplied out. Under Branch Any (`whatever`) the power comes out whatever its sign.
    """
    base, exponent = power.args
    if not exponent.is_Rational or exponent.is_Integer:
        return power
    split = _power_part(base, exponent.q)
    if split is None:
        return power
    root, rest = split
    if rest.is_number and (rest.is_zero or not whatever and _known_sign(rest) != 1):
        return power
    whole = root**exponent.q
    if whatever:
        outside = root ** (exponent.q * exponent)
    elif whole.is_nonnegative:
        outside = whole**exponent
        if any(found.base == whole for found in outside.atoms(sp.Pow)):
            return power
    else:
        return power
    return outside * rest**exponent


def _power_part(polynomial: sp.Basic, q: int) -> tuple[sp.Basic, sp.Basic] | None:
    """`(r, s)` with `polynomial` equal to `r^q*s`, `r` the product of the factors of
    the square-free factorization of `polynomial` to the power of their multiplicity
    divided by `q`, and `s` multiplied out, or None if `r` is a number or `polynomial`
    isn't a polynomial whose coefficients are rational or in the field of a few
    roots of positive rationals, of a size worth factoring."""
    try:
        terms = sp.Poly(polynomial)
    except sp.PolynomialError:
        return None
    variables = [gen for gen in terms.gens if gen.free_symbols]
    numbers = [gen for gen in terms.gens if not gen.free_symbols]
    if not variables:
        return None
    if numbers:
        if not all(_is_surd(number) for number in numbers):
            return None
        if _roots_degree(numbers) > _ROOTS_DEGREE:
            return None
    elif not terms.domain.is_QQ and not terms.domain.is_ZZ:
        return None
    if terms.total_degree() > _PERFECT_DEGREE or len(terms.terms()) > _PERFECT_TERMS:
        return None
    if numbers:
        return within(
            budget.EXTENDED, lambda: _power_part_over_roots(polynomial, variables, q)
        )
    _, factors = terms.sqf_list()
    root = sp.Mul(*(factor.as_expr() ** (times // q) for factor, times in factors))
    if root.is_number:
        return None
    return root, terms.exquo(sp.Poly(root**q, *terms.gens)).as_expr()


def _power_part_over_roots(
    polynomial: sp.Basic, variables: list[sp.Basic], q: int
) -> tuple[sp.Basic, sp.Basic] | None:
    """`_power_part` for a polynomial whose coefficients have roots of rationals in
    them, factored over the field the roots span."""
    terms = sp.Poly(polynomial, *variables, extension=True)
    if not terms.domain.is_Algebraic:
        return None
    _, factors = terms.sqf_list()
    root = sp.Poly(1, *variables, domain=terms.domain)
    for factor, times in factors:
        root *= factor ** (times // q)
    if root.is_ground:
        return None
    return _plainest_split(terms, root, q)


def _plainest_split(terms: sp.Poly, root: sp.Poly, q: int) -> tuple[sp.Basic, sp.Basic]:
    """`(r, s)` with `terms` equal to `r^q*s` and `r` a multiple of `root`, the plainest
    of the multiples that have a coefficient 1, or that take in the `q`-th root of a
    positive rational `s`: `3*x^2 + 2*SQRT(3)*x*y + y^2` is `(SQRT(3)*x + y)^2`, not
    `3*(x + SQRT(3)*y/3)^2`, and `2*x^2 + 2*SQRT(2)*x + 1` is `(SQRT(2)*x + 1)^2`."""
    domain = root.domain
    splits = []
    for coefficient in root.rep.coeffs():
        multiple = root.mul_ground(domain.quo(domain.one, coefficient))
        rest = terms.exquo(multiple**q)
        splits.append((multiple, rest))
        if rest.is_ground and rest.as_expr().is_Rational and rest.as_expr() > 0:
            try:
                taken = domain.from_sympy(sp.root(rest.as_expr(), q))
            except sp.polys.polyerrors.CoercionFailed:
                continue
            splits.append((multiple.mul_ground(taken), rest.quo_ground(taken**q)))
    found = [(multiple.as_expr(), sp.expand(rest.as_expr())) for multiple, rest in splits]
    return min(found, key=_plainness)


def _plainness(split: tuple[sp.Basic, sp.Basic]) -> tuple[int, int]:
    """How plainly a root and the rest of a radicand are written: by the denominators
    of their rational coefficients, then by their operations."""
    denominators = sum(
        term.as_coeff_Mul()[0].q
        for part in split
        for term in sp.Add.make_args(part)
        if term.as_coeff_Mul()[0].is_Rational
    )
    return denominators, sum(sp.count_ops(part) for part in split)


def _roots_degree(numbers: list[sp.Pow]) -> int:
    """A bound on the degree of the field the roots of positive rationals `numbers`
    span: each prime of their bases contributes the least common multiple of the orders
    of its roots, so that `SQRT(2)`, `SQRT(3)` and `SQRT(6)` span a field of degree 4."""
    orders: dict[int, int] = {}
    for number in numbers:
        primes = sp.factorrat(number.base, limit=1000)
        for prime in primes:
            orders[prime] = lcm(orders.get(prime, 1), number.exp.q)
    return prod(orders.values())


def _is_surd(number: sp.Basic) -> bool:
    """Whether `number` is a root of a positive rational."""
    return (
        isinstance(number, sp.Pow)
        and number.base.is_Rational
        and number.base.is_positive
        and number.exp.is_Rational
        and not number.exp.is_Integer
    )


def _variable_roots(base: sp.Basic) -> dict[sp.Symbol, tuple[sp.Dummy, int]]:
    """For each variable of `base` raised to fractions, a new variable `t` standing for
    its principal `q`-th root, `q` the least common denominator of the fractions, so
    that `4*r + 4*SQRT(r) + 1` is the polynomial `4*t^2 + 4*t + 1`. `t` is nonnegative
    where the variable is."""
    denominators: dict[sp.Symbol, int] = {}
    for found in base.atoms(sp.Pow):
        if found.base.is_Symbol and found.exp.is_Rational and not found.exp.is_Integer:
            denominators[found.base] = lcm(denominators.get(found.base, 1), found.exp.q)
    return {
        symbol: (
            sp.Dummy(symbol.name, nonnegative=True)
            if symbol.is_nonnegative
            else sp.Dummy(symbol.name),
            q,
        )
        for symbol, q in denominators.items()
    }


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


def _negative_radicands(expression: sp.Basic) -> sp.Basic:
    """Write a square root of a negative radicand as `#i` times a real root.

    For `u` known to be negative, `u^(p/2)` is `#i^p*(-u)^(p/2)` on the principal
    branch, so `SQRT(z)` is `#i*SQRT(-z)` and `1/SQRT(z)` is `-#i/SQRT(-z)` where `z <
    0`. Sympy already does this for numbers and for some radicands (`SQRT(1/z)` is
    `#i/SQRT(-z)`), and writing every such root the same way lets them cancel.
    """
    return expression.replace(_is_negative_radicand, _imaginary_root, simultaneous=False)


def _is_negative_radicand(expression: sp.Basic) -> bool:
    return (
        isinstance(expression, sp.Pow)
        and bool(expression.exp.is_Rational)
        and expression.exp.q == 2
        and not expression.base.is_number
        and expression.base.is_negative is True
    )


def _imaginary_root(power: sp.Pow) -> sp.Basic:
    return sp.I**power.exp.p * (-power.base) ** power.exp


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
    #: A rule for the form of exact answers only: under Approximate the numbers it
    #: would rewrite are approximations, which another form would round again.
    exact_only: bool = False
    #: The work the rule may do (see `budget.REWRITING`).
    work: int = budget.REWRITING


def _rewritten(expression: sp.Basic, context: Context) -> sp.Basic:
    """Reduce the kernels: apply every identity that pays, then every requested
    direction.

    Works on the vocabulary `_by_definition` produces, and runs before the normal form,
    which tidies up the sums the rules leave.
    """
    if expression.has(sp.MatrixExpr, Vector):
        return forced(expression, _distributed)
    # Before the definitions, which turn its hyperbolic leftovers into exponentials.
    expression = forced(expression, _complex_constants)
    expression = forced(expression, _symmetries)
    expression = _by_definition(expression)
    expression = _applied(expression, _IDENTITIES, context)
    for setting, directions in _DIRECTED.items():
        rules = directions.get(getattr(context, setting), ())
        if setting == "logarithm" and context.precision == Precision.APPROXIMATE:
            rules = tuple(
                rule._replace(rewrite=_numbers_held(rule.rewrite)) for rule in rules
            )
        expression = _applied(expression, rules, context)
    # Not in the tables, whose rules don't see the order list. Following the original,
    # the exponential stays under Collect.
    if context.exponential != Direction.COLLECT:
        expression = forced(
            expression, lambda e: _powers_of_logarithms(e, context.order)
        )
    return expression


def _numbers_held(rewrite: Rewrite) -> Rewrite:
    """`rewrite` with the logarithms of numbers held as they are.

    Under Approximate, Derive approximates the logarithms of numbers as it goes, so
    they are never combined: `#e^(t*(LN(5) - LN(2))/3)` is `#e^(0.305430*t)`, not a
    power of 5/2.
    """

    def run(expression: sp.Basic) -> sp.Basic:
        held = {
            log: sp.Dummy(real=True)
            for log in sp.ordered(expression.atoms(sp.log))
            if log.args[0].is_number
        }
        if not held:
            return rewrite(expression)
        back = {dummy: log for log, dummy in held.items()}
        return rewrite(expression.xreplace(held)).xreplace(back)

    return run


#: Heads that remain in an answer as unevaluated integrals, sums and products.
_STANDING = (sp.Integral, sp.Sum, sp.Product)


def _rewritten_beside_standing(expression: sp.Basic, context: Context) -> sp.Basic:
    """`_rewritten`, but rewriting unevaluated integrals, sums and products in a sum
    separately from the other terms.

    An unevaluated integral is a separate term, as in `INT(x/SIN(a*x), x)` plus an
    antiderivative. Rewritten together, the sum may be put over a common denominator,
    pulling the integral under it and printing it twice.

    An expression larger than `_REWRITTEN` nodes is left to the normal form: on it,
    each of the dozens of rules walks it all, and Derive too does bounded work on a
    large value.
    """
    if larger_than(expression, _REWRITTEN):
        return expression
    if not isinstance(expression, sp.Add):
        return _retried(_rewritten(expression, context), context)
    standing = [term for term in expression.args if term.has(*_STANDING)]
    if not standing or len(standing) == len(expression.args):
        return _retried(_rewritten(expression, context), context)
    rest = sp.Add(*(term for term in expression.args if not term.has(*_STANDING)))
    return _rewritten(rest, context) + sp.Add(
        *(_retried(_rewritten(term, context), context) for term in standing)
    )


#: The size, in nodes (see `rewriting.larger_than`), above which Simplify leaves out
#: the kernel rewrites. In the test suite no expression larger than 2,000 nodes is
#: changed by them, and in the recorded worksheets that compute none larger than
#: 25,497; those that ran out of time rewrote expressions of 50,000 to 200,000 nodes
#: again and again, in programs that grow a value at each step.
_REWRITTEN = 30_000

#: Heads `_retried` tries again after the rewrites.
_RETRIED = (sp.Integral, Antidifference)


def _retried(expression: sp.Basic, context: Context) -> sp.Basic:
    """The rewritten expression with each unevaluated integral or antidifference tried
    again, since the rewrites may have changed its integrand into one that integrates.
    RECUREQN.MTH's `LIN2_CCF` sums a quotient over `SIN(t)^2 + COS(t)^2`, which the
    rewrites reduce to a sum of cosines.

    `INT(SQRT(1 - SIN(t)^2), t, 0, pi/2)` is 1 once its integrand is `ABS(COS(t))`, and
    `INT(SQRT(SIN(x)^2 + COS(x)^2), x)` is `x` once it is 1. The normal form doesn't
    reach into an integral, so an integral still left is tried with its integrand in
    normal form, but kept as it was unless that answers it. An integral the rewrites
    left alone is answered from the cache, and one whose answer contains it again, as
    the else case of a conditional, stays. An answer is rewritten in turn, but its own
    unevaluated integrals are not tried again.

    A limit around an integral is taken again too, innermost first, if the integral
    changed: SURFACE_AREA in INT_APPS.MTH integrates a `LIM` of an inner integral.
    """

    def retried(e: sp.Basic) -> sp.Basic:
        if not e.has(*_RETRIED):
            return e
        arguments = [retried(argument) for argument in e.args]
        changed = any(new is not old for new, old in zip(arguments, e.args))
        rebuilt = e.func(*arguments) if changed else e
        if isinstance(e, _RETRIED) or (isinstance(e, sp.Limit) and changed):
            return tried(rebuilt)
        return rebuilt

    def tried(head: sp.Basic) -> sp.Basic:
        answer = _evaluate(head, context)
        if answer == head and isinstance(head, _RETRIED):
            answer = normalized(head)
        return head if answer.has(head) else answer

    def normalized(head: sp.Basic) -> sp.Basic:
        function, *rest = head.args
        normal = normal_form(function, context.order, over_numbers=False)
        if normal == function:
            return head
        answer = _evaluate(head.func(normal, *rest), context)
        return head if answer.has(type(head)) else answer

    if not expression.has(*_RETRIED):
        return expression
    again = forced(expression, retried)
    return expression if again == expression else _rewritten(again, context)


def _spelled(expression: sp.Basic, context: Context) -> sp.Basic:
    """Apply Derive's spelling conventions after the normal form.

    Running last means nothing rewrites them afterwards. Each is the form Derive prints,
    shorter or not: `1/COS(x)^2` is `TAN(x)^2 + 1`, and `LN(256)` is `8*LN(2)`.
    """
    if expression.has(sp.MatrixExpr, Vector):
        return expression
    return _applied(expression, _SPELLINGS, context)


def _decimal_coefficients(expression: sp.Basic) -> sp.Basic:
    """A product of a number and a sum, as decimals show it best.

    The normal form takes the terms' common denominator out of a sum, so `a*(3.1*x +
    3.2*y)` is `a*(31*x + 32*y)/10`, which a decimal notation would write `0.1*a*(31*x
    + 32*y)`, and `412.9563894*a*(x + y)` comes out as `2*10^-7*a*(2.064781947*10^9*x
    + ...)`. In a decimal notation the number goes in front only with the whole content
    of the sum, leaving it with coefficients 1 and -1; otherwise it goes into the sum.
    Either only where the numbers it makes are decimals that terminate, such as typed
    ones: other rationals are approximations, which a new form would round again.
    """

    def spelled(product: sp.Basic) -> sp.Basic:
        if not product.free_symbols:
            # A number is approximated as a whole, however it is written.
            return product
        coefficient, rest = product.as_coeff_Mul()
        factors = sp.Mul.make_args(rest)
        sums = [factor for factor in factors if factor.is_Add]
        if not coefficient.is_Rational or coefficient.q == 1 or len(sums) != 1:
            return product
        numbers = [term.as_coeff_Mul()[0] for term in sums[0].args]
        if not all(number.is_Rational for number in numbers):
            return product
        others = [factor for factor in factors if factor is not sums[0]]
        content, primitive = sums[0].as_content_primitive()
        if all(abs(term.as_coeff_Mul()[0]) == 1 for term in primitive.args):
            if not _terminating(coefficient * content):
                return product
            return sp.Mul(coefficient * content, *others, primitive, evaluate=False)
        spread = [coefficient * term for term in sums[0].args]
        if not all(_terminating(term.as_coeff_Mul()[0]) for term in spread):
            # Spread, the coefficients would be decimals cut short: `-0.00857*t^2*(38*t^8
            # - ...)` keeps its form at three digits.
            return product
        return sp.Mul(*others, sp.Add(*spread)) if others else sp.Add(*spread)

    def _terminating(number: sp.Basic) -> bool:
        if not number.is_Rational:
            return False
        denominator = int(number.q)
        for prime in (2, 5):
            while denominator % prime == 0:
                denominator //= prime
        return denominator == 1

    return expression.replace(lambda e: isinstance(e, sp.Mul), spelled)


def _applied(
    expression: sp.Basic, rules: tuple[_Rule, ...], context: Context
) -> sp.Basic:
    for rule in rules:
        if rule.yields_to is not None and getattr(context, rule.yields_to) != "Auto":
            continue
        if rule.exact_only and context.precision is Precision.APPROXIMATE:
            continue
        rewrite = rule.rewrite
        if rewrite is _extracted_powers and context.branch is Branch.ANY:
            rewrite = _any_extracted_powers
        expression = _within_budget(expression, rule.kept, rewrite, rule.work)
    return expression


def _bounded(expression: sp.Basic, rewrite: Rewrite) -> sp.Basic:
    """`rewrite` applied regardless of cost, but within the work one rewrite may do
    (see `budget.REWRITING`): one of Simplify's own steps, each an identity, so leaving
    it out leaves the expression as correct as it was."""
    return _within_budget(expression, forced, rewrite)


def _within_budget(
    expression: sp.Basic,
    kept: Callable[[sp.Basic, Rewrite], sp.Basic],
    rewrite: Rewrite,
    work: int = budget.REWRITING,
) -> sp.Basic:
    """`kept(expression, rewrite)`, or `expression` if the rule runs out of `work`, the
    work one rewrite may do (see `budget.REWRITING`)."""
    found = within(work, lambda: kept(expression, rewrite))
    return expression if found is None else found


def _one_denominator(expression: sp.Basic) -> sp.Basic:
    """`radsimp`, with a reciprocal exponential multiplied back into its sum.

    Sympy takes `#e^(-z)` as a denominator, so `radsimp` puts `#e^z/2 + #e^(-z)/2` over
    `#e^z` as `(#e^(2*z) + 1)*#e^(-z)/2`, which is shorter. Derive keeps the
    exponentials as separate terms, as COSH's definition gives them: `COSH(1/COS(x))`
    is `#e^(1/COS(x))/2 + #e^(-1/COS(x))/2`. A denominator under a denominator stays
    cleared: `1/(#e^z + #e^(-z))` is `#e^z/(#e^(2*z) + 1)`.

    A root of a power or product that may be negative is opaque to `radsimp`, which
    takes `SQRT(v^3)` for `v^(3/2)`: for a negative `v` they are conjugates, so
    `(SQRT(v^3) + 1)/(v^(3/2) + 1)` is not 1. So is a power like `v^(3/2)` itself,
    which `radsimp` writes as `SQRT(v^3)`.

    Numeric denominators alone are not gathered: `radsimp` would put `a/2 + b/3` over
    6, which is shorter only by the divisions it saves, and sympy spreads the 6 over the
    terms again whenever it builds a sum from it. The rules after this one would then
    count their results longer for that alone: in a sum of several sines with
    fractional coefficients, `trigsimp` writes `SIN(pi*(t/3 - 11/2))` as `COS(pi*t/3)`,
    but with its coefficients spread over the terms again the result is no shorter than
    the quotient, and the cosine would not be taken.
    """
    found = _radsimp_opened(expression)
    return expression if _numbers_distributed(found) == expression else found


def _numbers_distributed(expression: sp.Basic) -> sp.Basic:
    """`expression` with each product of a number and a sum multiplied out, as sympy
    builds it: `(3*a + 2*b)/6` is `a/2 + b/3`."""
    return expression.replace(
        lambda e: e.is_Mul and e.args[0].is_Rational and any(f.is_Add for f in e.args),
        lambda e: sp.Mul(*e.args),
    )


def _radsimp_opened(expression: sp.Basic) -> sp.Basic:
    """`radsimp`, with reciprocal exponentials multiplied back and roots that may be
    negative hidden from it (see `_one_denominator`)."""
    hidden = {
        power: sp.Dummy("root")
        for power in sp.ordered(expression.atoms(sp.Pow))
        if _root_of_compound(power) or _power_of_root(power)
    }
    if not hidden:
        return _exponentials_distributed(sp.radsimp(expression))
    shown = {dummy: power for power, dummy in hidden.items()}
    rationalized = sp.radsimp(expression.xreplace(hidden))
    return _exponentials_distributed(rationalized.xreplace(shown))


def _root_of_compound(power: sp.Pow) -> bool:
    """Whether `power` is a root of a power or product not known to be nonnegative,
    which `radsimp` may split or merge with other roots as if it were."""
    base, exponent = power.args
    return (
        not exponent.is_integer
        and (base.is_Pow or base.is_Mul)
        and base.is_nonnegative is not True
    )


def _power_of_root(power: sp.Pow) -> bool:
    """Whether `power` is a power of a root, like `v^(3/2)`, of a base not known to be
    nonnegative."""
    base, exponent = power.args
    return (
        exponent.is_Rational
        and not exponent.is_Integer
        and abs(exponent.p) != 1
        and base.is_nonnegative is not True
    )


def _polynomials_distributed(expression: sp.Basic) -> sp.Basic:
    """`_exponentials_distributed` for products of nothing else but numbers, of sums
    without symbolic denominators, as the normal form leaves `SINH(2*x)/COSH(x)`:
    `(#e^(2*x) - 1)*#e^(-x)` is `#e^x - #e^(-x)`."""
    return _exponentials_distributed(expression, polynomial=True)


def _exponentials_distributed(
    expression: sp.Basic, polynomial: bool = False
) -> sp.Basic:
    """Every product of a reciprocal exponential and a sum of exponentials multiplied
    out, each term's powers collected; with `polynomial`, only products of nothing else
    but numbers, of sums whose terms have numeric denominators."""

    def exponential(node: sp.Basic) -> bool:
        return isinstance(node, sp.exp) or (
            isinstance(node, sp.Pow) and node.base.is_number and not node.exp.is_Number
        )

    def reciprocal(node: sp.Basic) -> bool:
        exponent = exponential(node) and node.as_base_exp()[1]
        return bool(exponent) and bool(exponent.free_symbols) and (
            exponent.could_extract_minus_sign()
        )

    def of_exponentials(factor: sp.Basic) -> bool:
        base, exponent = factor.as_base_exp()
        return (
            base.is_Add
            and exponent.is_Integer
            and exponent > 0
            and any(map(exponential, sp.preorder_traversal(base)))
            and not (polynomial and any(map(divided, base.args)))
        )

    def divided(term: sp.Basic) -> bool:
        return any(
            factor.is_Pow
            and factor.exp.is_negative
            and not factor.base.is_number
            and not exponential(factor)
            for factor in sp.Mul.make_args(term)
        )

    def spread(product: sp.Basic) -> bool:
        return (
            product.is_Mul
            and any(map(reciprocal, product.args))
            and any(map(of_exponentials, product.args))
            and not (polynomial and any(map(other, product.args)))
        )

    def other(factor: sp.Basic) -> bool:
        return not (factor.is_number or reciprocal(factor) or of_exponentials(factor))

    def spread_out(product: sp.Basic) -> sp.Basic:
        terms = sp.Add.make_args(
            sp.expand(product, power_base=False, power_exp=False, log=False)
        )
        return sp.Add(*(sp.powsimp(term) for term in terms))

    if not expression.has(sp.exp, sp.Pow):
        return expression
    return expression.replace(spread, spread_out)


def _powers_of_e(expression: sp.Basic) -> sp.Basic:
    """Each sum of rational multiples of integer powers of `#e`, one of them negative,
    written as the lowest power times a polynomial in `#e`, unless it is part of a
    larger sum, which the normal form would multiply out again.

    Exponentials of variables stay separate terms (`_exponentials_distributed`), but
    `#e` to a number is a constant that Derive collects like a variable over its
    denominator: `COS(pi + #i)` is `-#e^(-1)*(#e^2 + 1)/2`, and `INT(x*#e^(-x), x, 0,
    1)` is `#e^(-1)*(#e - 2)`. In a logarithm's argument the power comes out as a
    term: `LN(1 - #e^(-1))` is `LN(#e - 1) - 1`.
    """

    def power(term: sp.Basic) -> tuple[sp.Rational, int] | None:
        coefficient, rest = term.as_coeff_Mul()
        if not coefficient.is_Rational:
            return None
        if rest == 1:
            return coefficient, 0
        base, exponent = rest.as_base_exp()
        if base is not sp.E or not exponent.is_Integer:
            return None
        return coefficient, int(exponent)

    def laurent(total: sp.Basic) -> bool:
        if not total.is_Add:
            return False
        powers = [power(term) for term in total.args]
        return None not in powers and min(k for _, k in powers) < 0

    def collected(total: sp.Basic) -> sp.Basic:
        powers = [power(term) for term in total.args]
        lowest = min(k for _, k in powers)
        inner = sp.Add(*(c * sp.E ** (k - lowest) for c, k in powers))
        content, primitive = inner.primitive()
        if max(powers, key=lambda p: p[1])[0] < 0:
            content, primitive = -content, -primitive
        factors = (content, sp.exp(lowest), primitive) if content != 1 else (
            sp.exp(lowest), primitive
        )
        return sp.Mul(*factors, evaluate=False)

    def walked(node: sp.Basic, in_sum: bool) -> sp.Basic:
        if node.is_Atom:
            return node
        if node.is_Add and not in_sum and laurent(node):
            return collected(node)
        if isinstance(node, sp.log) and len(node.args) == 1 and laurent(node.args[0]):
            # The power of `#e` is positive, so it leaves the logarithm.
            powers = [power(term) for term in node.args[0].args]
            lowest = min(k for _, k in powers)
            rest = sp.Add(*(c * sp.E ** (k - lowest) for c, k in powers))
            return sp.log(rest) + lowest
        inner = node.is_Add or (in_sum and node.is_Mul)
        args = [walked(arg, inner) for arg in node.args]
        return node.func(*args) if args != list(node.args) else node

    if not expression.has(sp.E, sp.exp):
        return expression
    return walked(expression, False)


def _collected_by_pieces(expression: sp.Basic) -> sp.Basic:
    """Collect the terms of a sum by their piecewise factors: one polynomial beside each
    `SIGN(u)`, `ABS(u)` or standing `CHI` call, as a spline is written.

    `5*x^3*SIGN(x)/76 + 33*ABS(x)/76 + SIGN(x)/2` is `(5*x^3 + 33*x + 38)*SIGN(x)/76`:
    `ABS(u)` is `u*SIGN(u)` where `SIGN(u)` stands beside it. A sum without two terms
    over the same piecewise factor is left alone, and so is one where such a factor's
    terms are no polynomial, as an antiderivative counting arches with `FLOOR` is.
    """
    if not expression.has(sp.sign, sp.Abs, AppliedUndef):
        return expression

    def piece(factor: sp.Basic) -> bool:
        base = factor.base if factor.is_Pow and factor.exp.is_Integer else factor
        if isinstance(base, (sp.sign, sp.Abs)):
            return not base.args[0].is_number
        return isinstance(base, AppliedUndef) and base.func.__name__ == "CHI"

    def collected(total: sp.Add) -> sp.Basic:
        signs = {
            factor.args[0]
            for term in total.args
            for factor in sp.Mul.make_args(term)
            if isinstance(factor, sp.sign)
        }
        groups: dict[sp.Basic, list[sp.Basic]] = {}
        for term in total.args:
            if signs:
                term = term.xreplace(
                    {
                        bars: bars.args[0] * sp.sign(bars.args[0])
                        for bars in sp.Mul.make_args(term)
                        if isinstance(bars, sp.Abs) and bars.args[0] in signs
                    }
                )
            kernel, rest = [], []
            for factor in sp.Mul.make_args(term):
                (kernel if piece(factor) else rest).append(factor)
            groups.setdefault(sp.Mul(*kernel), []).append(sp.Mul(*rest))
        collected = []
        for kernel, parts in groups.items():
            together = sp.Add(*parts)
            if kernel == 1 or len(parts) < 2:
                collected.extend(part * kernel for part in parts)
            elif not together.is_polynomial(*together.free_symbols):
                return total
            else:
                content, rest = together.as_content_primitive()
                collected.append(_keep_coeff(content, rest) * kernel)
        if all(len(parts) < 2 for kernel, parts in groups.items() if kernel != 1):
            return total
        return sp.Add(*collected)

    return expression.replace(
        lambda e: e.is_Add
        and any(piece(f) for t in e.args for f in sp.Mul.make_args(t)),
        collected,
    )


def _cotangents(expression: sp.Basic) -> sp.Basic:
    """Each reciprocal power of a tangent written as a power of the cotangent, which
    Derive prints: `1/TAN(x)` is `COT(x)`. Not where the same tangent stands elsewhere
    in the product, as in `(TAN(x)^2 - 1)/TAN(x)`, which would mix the two."""

    def reciprocal(factor: sp.Basic) -> bool:
        return (
            factor.is_Pow
            and isinstance(factor.base, sp.tan)
            and factor.exp.is_Integer
            and factor.exp < 0
        )

    def written(node: sp.Basic) -> sp.Basic:
        if node.is_Atom:
            return node
        if reciprocal(node):
            return sp.cot(node.base.args[0]) ** -node.exp
        if not node.is_Mul:
            args = [written(arg) for arg in node.args]
            return node.func(*args) if args != list(node.args) else node
        factors = []
        for factor in node.args:
            others = [other for other in node.args if other is not factor]
            if reciprocal(factor) and not any(o.has(factor.base) for o in others):
                factors.append(sp.cot(factor.base.args[0]) ** -factor.exp)
            elif reciprocal(factor):
                factors.append(factor)
            else:
                factors.append(written(factor))
        return sp.Mul(*factors)

    if not expression.has(sp.tan):
        return expression
    return written(expression)


def _combinatorial(expression: sp.Basic) -> sp.Basic:
    """`combsimp`, if the expression is combinatorial and small enough.

    Not where a binomial `COMB(n, k)` has `n - k` a negative integer and `k` may be 0:
    `combsimp` makes it 0, but `COMB(k - 1, k)` is 1 at `k = 0`.
    """
    if not expression.has(*_COMBINATORIAL) or sp.count_ops(expression) > _COMBSIMP:
        return expression
    for binomial in expression.atoms(sp.binomial):
        top, bottom = binomial.args
        below = top - bottom
        if below.is_integer and below.is_negative and bottom.is_zero is not False:
            return expression
    return sp.combsimp(expression)


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


def _powers_expanded(expression: sp.Basic, head: type) -> sp.Basic:
    """`expression` with each integer power of a sum of calls of `head` expanded, as
    `Trigpower` writes its result as a polynomial: `(1 - COS(x)^2)^2` is
    `COS(x)^4 - 2*COS(x)^2 + 1`."""
    return expression.replace(
        lambda e: e.is_Pow
        and e.exp.is_Integer
        and e.exp > 1
        and e.base.is_Add
        and e.base.has(head),
        sp.expand,
    )


def _powers_into_exponentials(expression: sp.Basic) -> sp.Basic:
    """Under `Exponential := Collect`, a power `u^a` with a symbolic exponent standing
    in a product with an exponential is drawn into it as `#e^(a*LN(u))`, where `u` is
    known to be positive: `x^a*#e^(b + c)` is `#e^(a*LN(x) + b + c)` for positive `x`.
    For other `u` the identity fails (at 0, or for a negative `u`), so it stays."""

    def drawn(product: sp.Basic) -> sp.Basic:
        exponents = [
            factor.args[0] for factor in product.args if isinstance(factor, sp.exp)
        ]
        drawn_in = [
            factor
            for factor in product.args
            if factor.is_Pow
            and not factor.exp.is_number
            and factor.base.is_positive
            and factor.base is not sp.E
        ]
        if not exponents or not drawn_in:
            return product
        exponents += [power.exp * sp.log(power.base) for power in drawn_in]
        rest = [
            factor
            for factor in product.args
            if not isinstance(factor, sp.exp) and factor not in drawn_in
        ]
        return sp.Mul(*rest) * sp.exp(sp.Add(*exponents))

    return expression.replace(lambda e: e.is_Mul and e.has(sp.exp), drawn)


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
    _Rule("negative radicands", _negative_radicands, forced,
          "SQRT(z) is #i*SQRT(-z) for a negative z"),
    _Rule("signs", folded_signs, forced,
          "ABS(u)*SIGN(u) is u and u/ABS(u) is SIGN(u), for a real u"),
    _Rule("signs of reciprocals", _signs_of_reciprocals, forced,
          "SIGN(1/u) is SIGN(u), and SIGN(1/ABS(u)) is 1 wherever it is defined"),
    _Rule("paired signs", _paired_signs, forced,
          "ABS(x*ATAN(x)) is x*ATAN(x): factors whose signs cancel need no bars"),
    _Rule("conjugate pairs", _conjugate_pairs, forced,
          "CONJ(z)*z is ABS(z)^2"),
    _Rule("phases of positive values", _phases_of_positive_values, forced,
          "ATAN(0, 4 - 3*COS(t)^2) is 0: the angle of a value positive for every "
          "value of its variables"),
    _Rule("powers of one unit", _powers_of_one_unit, _gated,
          "(12*SQRT(2) + 17)^n is (SQRT(2) + 1)^(4*n): powers of one positive real "
          "number written over it"),
    _Rule("exponentials beside waves", _exponentials_as_waves, _gated,
          "#e^(-#i*u) + #i*SIN(u) is COS(u): an exponential of an imaginary angle "
          "written as waves where a sine or cosine of the angle stands beside it"),
    _Rule("polar logarithms", polar_logarithms, forced,
          "LN(ABS(z)) + #i*PHASE(z) is LN(z): a logarithm's parts joined"),
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
    _Rule("half-angle roots", _half_angle_roots, _gated,
          "SQRT((1 + COS(6))/2) is -COS(3): a root of 1 + COS(u) over the half angle"),
    _Rule("half angles", _half_angles, _gated,
          "calls of fractional multiples of one angle cancel written about it",
          work=budget.TRIGONOMETRIC),
    _Rule("one spelling of radicands", _one_spelling_of_radicands, forced,
          "SQRT((d + 1)*(d - 1))/SQRT(d^2 - 1) is 1: one radicand spelled one way"),
    _Rule("joined roots", _joined_roots, forced,
          "SQRT(s)*SQRT(1 - 1/s) is SQRT(s - 1): roots whose radicands cancel joined, "
          "where that holds for every real value"),
    _Rule("combined radicands", _combined_radicands, forced,
          "SQRT(a + b^2/c) is ABS(b')/SQRT(c') when its radicand over one denominator "
          "factors with squares", exact_only=True),
    _Rule("denested", _denested, _unnested,
          "SQRT(5 + SQRT(24)) is SQRT(2) + SQRT(3): each radical denested"),
    _Rule("flattened", _flattened, _gated,
          "a constant over nested radicals reduced in the field its surds span"),
    _Rule("rationalized", _rationalized, _gated,
          "the coefficient 1/(SQRT(2) + 1) of a term is SQRT(2) - 1"),
    _Rule("trigonometric constants", _trigonometric_constants, _gated,
          "COS(pi/9)*COS(2*pi/9)*COS(4*pi/9) is 1/8: values of rational angles"),
    _Rule("conjugate products", _conjugate_products, _gated,
          "(SQRT(a) + 1)*(SQRT(a) - 1) is a - 1: conjugates over a root multiplied out"),
    _Rule("over roots", _over_roots, _gated,
          "a quotient in x and SQRT(y) cancels: (x*SQRT(y) + y)/(x + SQRT(y)) is SQRT(y)"),
    _Rule("over conjugates", _over_conjugates, _gated,
          "(x + 1)/(SQRT(x + 2) + 1) is SQRT(x + 2) - 1: a denominator linear in one "
          "root multiplied by its conjugate, where that cancels"),
    _Rule("over surds", _over_surds, _gated,
          "a quotient cancels over the numeric roots in it: (x^2 - 2)/(x - SQRT(2))"),
    _Rule("over sines", _over_sines, _realized,
          "(#e^(#i*x) + #e^(-#i*x))/2 is COS(x): imaginary exponentials of real angles"),
    _Rule("over exponentials", _over_exponentials, _gated,
          "(#e^x - 1)/(#e^(x/2) + 1) is #e^(x/2) - 1: exponentials over one measure"),
    _Rule("one denominator", _one_denominator, _gated,
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
            _Rule("any powsimp", _any_powsimp, forced,
                  "powers over one exponent are one power, whatever the bases"),
            _Rule("any perfect powers", _any_perfect_powers, forced,
                  "SQRT(PHI(x)^2 - 2*PHI(x) + 1) is PHI(x) - 1, whatever PHI(x) is"),
        ),
    },
    "trigonometry": {
        Direction.AUTO: (
            _Rule("pythagorean", _pythagorean, _fewer_terms,
                  "SIN(z)^2 + COS(z)^2 is 1, in sums of any size"),
            _Rule("trigsimp", _trigsimp, _gated,
                  "trigonometric identities, where they cancel something",
                  work=budget.TRIGONOMETRIC),
            _Rule("multiple angles", _multiple_angles, _cancelled,
                  "SIN(3*x)/SIN(x) is 2*COS(2*x) + 1: multiples of one angle reduced"),
        ),
        Direction.COLLECT: (
            _Rule("vanishing sums", _trigsimp, _vanished,
                  "a sum that identities make 0 is 0, before it is written over "
                  "multiple angles", work=budget.TRIGONOMETRIC),
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
            _Rule("TR6", lambda e: _powers_expanded(TR6(e), sp.sin), forced,
                  "even powers of cosines written over sines, expanded"),
        ),
        TrigPower.COSINES: (
            _Rule("TR5", lambda e: _powers_expanded(TR5(e), sp.cos), forced,
                  "even powers of sines written over cosines, expanded"),
        ),
    },
    # Auto is the mixed direction of 6.2: the sum rule combines here, so
    # `3*LN(2) + LN(5)` is `LN(40)`, and the power rule extracts in `_SPELLINGS`, so
    # `LN(256)` is `8*LN(2)`, like `SQRT(12)` is `2*SQRT(3)`. `LN(x^2 - x) - LN(x)`
    # combines only if `x` is declared to exclude values where that is false.
    "logarithm": {
        Direction.AUTO: (
            _Rule("factored arguments", _log_arguments_factored, forced,
                  "LN(x*y^4 + 2*x*y^2 + x) is LN(x*(y^2 + 1)^2)"),
            _Rule("logcombine", _nonnegatively(_logs_combined), _gated,
                  "logarithms added into one, where it is shorter"),
        ),
        Direction.COLLECT: (
            _Rule("factored arguments", _log_arguments_factored, forced,
                  "LN(x*y^4 + 2*x*y^2 + x) is LN(x*(y^2 + 1)^2)"),
            _Rule("collected logarithms", _logs_collected, forced,
                  "LN(x)/2 + LN(y^2 + 1) is LN(SQRT(x)*(y^2 + 1)) where 6.2 allows it"),
            _Rule("logcombine", _nonnegatively(_logs_combined), forced,
                  "logarithms added into one"),
        ),
        Direction.EXPAND: (
            _Rule("factored arguments", _log_arguments_factored, forced,
                  "LN(x*y^4 + 2*x*y^2 + x) is LN(x*(y^2 + 1)^2)"),
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
            _Rule("powers into exponentials", _powers_into_exponentials, forced,
                  "x^a*#e^b is #e^(a*LN(x) + b) for positive x"),
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
    _Rule("distributed exponentials", _polynomials_distributed, forced,
          "#e^x - #e^(-x), not (#e^(2*x) - 1)*#e^(-x), as the normal form leaves it"),
    _Rule("powers of #e", _powers_of_e, forced,
          "#e/2 + #e^(-1)/2 is #e^(-1)*(#e^2 + 1)/2: #e collected like a variable"),
    _Rule("cotangents", _cotangents, forced, "1/TAN(x) is COT(x)"),
    _Rule("collected by pieces", _collected_by_pieces, forced,
          "x*SIGN(x) + SIGN(x) is (x + 1)*SIGN(x): one polynomial beside each "
          "piecewise factor, as a spline is written", exact_only=True),
    _Rule("combined radicands", _combined_radicands, forced,
          "a radicand the normal form left with fractions in a sum, over one "
          "denominator", exact_only=True),
    _Rule("signs", folded_signs, forced,
          "u/ABS(u) is SIGN(u), for the bars a combined radicand leaves",
          exact_only=True),
    # After a radicand with fractions is combined, which takes its own square factors
    # out: a content taken first would leave it with fractions still in a sum.
    _Rule("radicand contents", _radicand_contents, forced,
          "SQRT(768*x + 1280) is 16*SQRT(3*x + 5): perfect powers out of a sum's "
          "content", exact_only=True),
)
