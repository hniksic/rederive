"""Simplify: the first command built on the engine's two doors.

Derive's Simplify has two goals at once. Reach a form that is *sufficiently
simple* - no superfluous variables, roots, functions or reducible degrees - and
put it in a normal form, so that two ways of writing one thing come back the
same way. Between the two it transforms as little as it can, which is why this
is a deliberate sequence of named rewrites and not a call to `sympy.simplify`:
a wholesale simplifier decides for itself what a nicer expression is, and
Derive does not want a nicer expression, it wants the same expression with the
slack taken out.

The sequence is:

1. Evaluate the tree, innermost first, which is `evaluate`'s business: names
   replaced by what they stand for, calls of defined functions worked out over
   their arguments, and the test of every `IF` decided before a branch is
   chosen. Sympy's automatic evaluation does most of what the manual lists as
   Simplify's basic work on the way: combining numbers, collecting like
   factors and terms, the identities for zero and one, distributing integer
   powers over products.
2. On the way, as each is made, the calculus heads are worked out by
   `_evaluate`, and the `FACTOR`, `EXPAND`, `SOLVE` and `RATE` heads are run by
   `_commanded`. A calculus head that will not evaluate survives to the output
   as itself, which is how `INT(#e^(x^2), x)` comes back an integral rather than
   an error.
3. Reduce the kernels, over the vocabulary the answer is written in: offer
   each identity in turn and keep it only if it pays - fewer operations, or
   one variable fewer - and apply the rules a mode set to Collect or Expand
   asks for, that being an instruction rather than an option. The tables at
   the end of this module are the whole of it.
4. Write every sum and quotient in the normal form `normal_form` is about: a
   sum as a rational function of the most main generator it holds, a product
   cancelled and never distributed. This one is not gated on anything - it is
   what a normal form means - and it is where `(x + 1)^9 + y` becomes a
   ninth-degree polynomial while `(y + 1)^9 + x` does not, and where
   `(x^2 + 2*x*y + y^2)/(x^2 - y^2)` comes back `(x + y)/(x - y)`.
5. Spell the answer the way Derive does where that is a convention and not an
   economy: `1/COS(x)^2` as `TAN(x)^2 + 1`, `LN(256)` as `8*LN(2)`.
6. Approximate, if the precision mode asks for it.

Every rewrite is offered inside a `try`, and a rewrite that raises is a rewrite
that was not worth having: the previous form stands. That is what makes the
command total. Nothing sympy does can turn a valid entry into an error.
"""

from __future__ import annotations

from collections.abc import Callable
from math import gcd
from typing import NamedTuple

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.logic.boolalg import Boolean
from sympy.simplify.fu import TR2, TR5, TR6, TR7, TR8, TR11

from rederive.engine.approximation import approximated
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
from rederive.engine.intervals import decided, settled
from rederive.engine.normal import normal_form
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
from rederive.model.expr import Node
from rederive.syntax.state import ParseState

__all__ = ["approx", "approximated", "simplified", "simplify"]

Rewrite = Callable[[sp.Basic], sp.Basic]

#: What `combsimp` is for. It is offered only where one of these appears,
#: because on an ordinary polynomial it factors - `x^2 + 2*x` becomes
#: `x*(x + 2)` - and factoring is the Factor command's business, not Simplify's.
_COMBINATORIAL = (sp.factorial, sp.factorial2, sp.binomial, sp.gamma, sp.ff, sp.rf)

#: And only on an expression this size or under. `combsimp` compares every pair
#: of factorials it can see, and over Derive's own `F_0M` - a hundred and two
#: operations, half of them factorials of symbolic arguments - it takes
#: twenty-three seconds. Measured over every combinatorial expression in the
#: corpus: at this size and below, none costs more than a fifth of a second.
_COMBSIMP = 60

#: How many truths a boolean expression may be about before the algebra is
#: left undone. Both normal forms are read off a truth table, so the work and
#: the answer alike double with every variable added: a chain of six `XOR`s
#: comes back as a hundred and twenty-nine operations, which is a longer
#: expression than the one asked about and no simplification of it. Sympy has
#: a limit of its own at eight, and eight is already past the point where the
#: answer stops being worth having.
_BOOLEAN_VARIABLES = 5


def simplify(
    node: Node, context: Context | None = None, state: ParseState | None = None
) -> Result:
    """Derive's Simplify: `node` in its sufficiently simple form.

    Works on any subtree, not only on a whole authored entry, so that the
    session can simplify a highlighted subexpression and splice the answer
    back. `state` is the symbol table the answer is reparsed with; a session
    working in a non-default input or case mode must pass its own.

    The conditionals the author wrote are collected before anything is done to
    them and given to the printer, an undecidable one being shown as written
    and not as converted. `authored_conditionals` is what that means.
    """
    context = context or Context()
    node = named_as_declared(node, state)
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
    """Derive's approX: Simplify with the precision mode set to Approximate.

    Not a second evaluator. The only difference from `simplify` is the
    precision the same pipeline runs at, which is what the manual says approX
    is.
    """
    context = context or Context()
    approximate = context.with_precision(Precision.APPROXIMATE, digits)
    return simplify(node, approximate, state)


def simplified(
    node: Node, context: Context | None = None, decide: bool = True
) -> sp.Basic:
    """The simplified expression, before it is written back out.

    The sympy-level entry point. `simplify` is this plus `from_sympy`; a later
    command that wants to keep computing has no reason to print and reparse in
    between.

    `decide` is what makes a relation that holds come back as `true`, and soLve
    is the one caller that turns it off. Its pipeline runs to prepare an
    equation for solving, not to answer it, and `x = x` decided to `true` on
    the way in is an equation with nothing left to solve for - where the answer
    it has to reach is `x = @1`, every value there is.

    A conditional nothing could decide is suspended by the evaluation and stays
    so through every step here, which is what leaves its branches as they were
    written; it is shown as the conditional it is only once all of them are
    done.
    """
    context = context or Context()
    environment = Environment(
        context,
        decide=lambda test: _decide(test, context),
        work_out=lambda head: _evaluate(head, context),
        command=lambda name, values: _commanded(name, values, context),
    )
    expression = evaluate(node, environment)
    try:
        result = _transform(expression, context, decide)
    except Exception:
        # Whatever went wrong, the evaluated input is still a correct answer to
        # "simplify this".
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
    """The two sides simplified apart, and the relation answered if it decides.

    `2 = 2` is `true` and `1 = 2` is `false`, which is what the original
    answers: a relation is a statement, and simplifying a statement that says
    something definite means saying it. The judgement is three-valued, so a
    relation that settles nothing - `x = 2`, or `ABS(x) < 1` under no
    declaration - comes back exactly as it was authored rather than as a guess.

    An ordering is asked only of sides that are real, because that is the only
    domain it means anything over: `x < x + 1` holds for a real `x` and says
    nothing about a complex one, where sympy's own comparison declines. It is
    asked the way the test of an `IF` is, declared domains and all. An equation
    is asked of the arithmetic only, which is `_equated`.

    `decide` off is soLve's pipeline, which wants the sides simplified and the
    relation left standing to be solved.

    Two matrices of one shape are related element by element, which is what
    section 8.5 reads a solved system off: `[[x], [y], [z]] = A^-1 . b` comes
    back as the three equations it stands for.
    """
    left = _transform(expression.lhs, context, decide)
    right = _transform(expression.rhs, context, decide)
    spread = _spread(expression.func, left, right, context, decide)
    if spread is not None:
        return spread
    return _stated(expression.func, left, right, context, decide)


def _spread(
    func, left: sp.Basic, right: sp.Basic, context: Context, decide: bool
) -> sp.Basic | None:
    """A relation between two matrices, stated of each pair of elements.

    None where the sides are not two matrices of one shape, which is a
    statement about the matrices themselves and is decided whole. The answer
    keeps the shape it was made from, a row of them being written flat as any
    one-row matrix is.
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
    """One relation over two sides already simplified, answered where it decides."""
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
    """Whether a relation standing on its own says something definite.

    An equation is answered by arithmetic alone, which is `_equated`. An
    ordering is put the question the test of an `IF` is put, the declared
    domains included, since what an ordering says is about where on the line a
    value falls and that is what a domain is about.
    """
    if isinstance(relation, (sp.Equality, sp.Unequality)):
        return _equated(relation, context)
    return _decide(relation, context)


def _equated(relation: Relational, context: Context) -> bool | None:
    """Whether the two sides of an equation are one value, by arithmetic alone.

    The original does not put an equation to the domains. `x = #i` is where a
    quadratic with a negative discriminant lands, and it stands as written
    whether `x` was declared real, declared complex or declared nothing at all;
    so does `ABS(x) + 1 = 0`, which no real satisfies and which is still an
    equation somebody may have meant to write. An equation is a question about
    a value, not a claim to be refuted from what the value is known to be.

    What answers one is the difference of the two sides coming out a number:
    zero and the sides are the same value, a nonzero real and they are not.

    A difference that is a number and not real answers nothing, which is the
    answer the original gives: `#i = 1` and `2*#i = #i` both stand. Only the
    reals are ordered, and off the line the one thing a difference settles is
    being zero.

    Matrices have no difference to take where the shapes do not fit, and the
    two written values are then all there is to compare: `[1, 2] = [1, 2, 3]`
    is false, while `[x, y] = [1, 2, 3]` says something about `x` and `y` and
    stands.
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

    Two sides that came out written the same are, whatever they are written
    out of: `inf = inf` and `true = true` hold, and neither has a difference to
    take. Everything else is the difference, taken over the rewrites the rest
    of the pipeline uses so that `SIN(x)^2 + COS(x)^2 = 1` is answered too.
    """
    if relation.lhs == relation.rhs:
        return True
    difference = _attempt(relation, lambda r: _rewritten(r.lhs - r.rhs, context))
    if difference is None or not difference.is_number:
        return None
    if difference.is_zero:
        return True
    if difference.is_zero is False and difference.is_extended_real is True:
        return False
    return None


def _comparable(relation: Relational) -> bool:
    """Whether this relation is one that can be given an answer at all.

    A side holding `?` is a side whose value nobody knows, and nothing is
    decidably equal to an unknown - sympy compares the two `?` it can see and
    finds them the same object, which is a fact about the notation and not
    about the values it stands for.

    `<` and its three companions order their sides, and only the reals are
    ordered, so a side not known to be real is one an ordering has no answer
    about however the arithmetic on the two happens to come out. Equality needs
    no such condition: every value is equal to itself or is not.
    """
    if relation.has(sp.nan):
        return False
    if isinstance(relation, (sp.Equality, sp.Unequality)):
        return True
    return all(side.is_extended_real is True for side in relation.args)


def _definition(expression: sp.Basic, context: Context, decide: bool) -> sp.Basic:
    """An assignment or a function definition: its value, and its own shape.

    The first two arguments are the name being defined and the operator that
    defines it, which is exactly what must survive untouched so that the line
    still reads as a definition.
    """
    head, operator, *value = expression.args
    return expression.func(
        head, operator, *(_transform(part, context, decide) for part in value)
    )


def _boolean(expression: Boolean, context: Context, decide: bool) -> sp.Basic:
    """Relations joined by boolean operators, solved and simplified.

    `6 >= -2*x AND 3*x /= -9` is a statement about one variable, and its
    simplest form is the range it describes. Where that does not work - more
    than one variable, an operand that is no relation - the boolean algebra
    runs instead: `p XOR q` is written out of the three operators that are
    left, `p OR NOT p` is `true`, and a conjunct standing in every disjunct is
    taken out. Whatever survives that keeps its shape, with its operands
    simplified.
    """
    solved = _reduced(expression)
    if solved is not None:
        return solved
    expression = _algebra(expression)
    if not isinstance(expression, Boolean) or not expression.args:
        # `p OR NOT p` is `true` and `p AND (p OR q)` is `p`: an atom is the
        # whole answer, and there is nothing left to simplify inside it.
        # Handing it back to `_transform` would only arrive here again.
        return expression
    try:
        return expression.func(
            *(_transform(a, context, decide) for a in expression.args)
        )
    except Exception:
        return expression


def _algebra(expression: Boolean) -> sp.Basic:
    """`expression` in the shorter of the two normal forms boolean algebra has.

    A sum of products and a product of sums are computed both, and the shorter
    of the two is the answer - which is the preference the manual states, and
    what leaves `p AND (q OR r)` alone where a simplifier committed to sums of
    products would multiply it out. `NOT` is driven inward on the way, `XOR`
    and `IMP` are written out of the operators that remain, and a conjunct
    every disjunct holds is taken outside. A tie goes to the form asked for
    first, which is how `p XOR q` comes back as a sum of products.

    Relations are opaque to it: each one stands for a variable of its own, and
    what is inside is the business of the operand's own simplification.

    Three forms are asked for and not two. Sympy answers a request for a named
    form that the expression is already written in with the expression itself,
    so `p OR NOT p` asked for as a sum of products comes back as it went in;
    the unnamed request is the one that always does the algebra, and it leads
    the list so that it also settles a tie.
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
    """How many independent truths the expression is a statement about.

    A relation is one of them however many variables it names, `x > 1 AND
    x < 2` being two statements and not one.
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
    # The relations were assembled undecided; `reduce_inequalities` needs them
    # in the evaluated form it builds its own answers from.
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
    """An ordinary expression, through the whole sequence.

    What a `FACTOR`, `EXPAND` or `SOLVE` head answered is the form the author
    asked for, and every step here would take it apart - the gated rewrites ask
    whether a candidate is shorter, and a factorization is routinely longer, and
    the normal form would multiply a prime decomposition back into the integer
    it decomposes. So each such answer is stood in for by a symbol of its own
    while the steps run, and put back for the approximation alone.

    Not inside a sum, though, unless it is a number. A sum is written in the
    normal form whatever its terms were asked to look like, so
    `FACTOR(x^2 - 1) + x^3` is `x^3 + x^2 - 1` and `x*FACTOR(x^2 - 1) + 1` is
    `x^3 - x + 1`, while `FACTOR(12) + 1` keeps its `2^2*3`, a prime
    decomposition being no polynomial to expand.
    """
    expression = _summed(expression)
    held = {found: sp.Dummy("answer") for found in _commands(expression)}
    if held:
        expression = expression.xreplace(held)
    expression = _numeric(expression)
    # Before the rewrites rather than after, so that what the declared
    # intervals settle is settled for them too: `ABS(x) + ABS(x - 1)` over
    # `(0, 1)` is a sum that cancels, and only once both bars are off.
    expression = settled(expression, context)
    expression = _rewritten(expression, context)
    expression = _factorials(expression)
    # The normal form, and the one step that is not gated on getting shorter:
    # a sum is written about its primary variable whether or not that pays.
    #
    # `over_numbers` off: nothing here divided, so a number several terms carry
    # underneath is their own coefficient and not a denominator the sum is
    # written over. `x^2/6 + x/2` is what was authored and what comes back.
    expression = normal_form(expression, context.order, over_numbers=False)
    expression = _spelled(expression, context)
    if held:
        expression = expression.xreplace(
            {dummy: found.args[0] for found, dummy in held.items()}
        )
    return approximated(expression, context)


def _summed(expression: sp.Basic) -> sp.Basic:
    """`expression` with each command's answer within a sum taken for its value,
    where the answer is not a number."""

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
    """Whether `test` holds, or None when nothing available settles it.

    What the evaluation asks of the test of every `IF` it meets, before it
    chooses a branch. A relation reaches here undecided, so it is offered again
    evaluated - which is where the declared domains do their work - and then with
    its two sides brought together by the same rewrites the rest of the pipeline
    uses. What none of those settles is put to the declared intervals, which are
    the one thing a symbol could not carry into the evaluation itself.

    A test that is no relation at all is a comparison with zero written short,
    the same reading the conversion gives one.
    """
    test = as_condition(test)
    for candidate in _truths(test, context):
        if candidate is sp.true:
            return True
        if candidate is sp.false:
            return False
    return decided(test, context)


def _truths(test: sp.Basic, context: Context) -> list[sp.Basic]:
    """Every reading of `test` worth asking, easiest first."""
    if not isinstance(test, Relational):
        return [test]
    readings = [test, _attempt(test, lambda t: t.func(t.lhs, t.rhs))]
    difference = _attempt(test, lambda t: _rewritten(t.lhs - t.rhs, context))
    if difference is not None:
        readings.append(_attempt(test, lambda t: t.func(difference, 0)))
    return [reading for reading in readings if reading is not None]


# -- the calculus heads ------------------------------------------------------


def _evaluate(head: sp.Basic, context: Context) -> sp.Basic:
    """One calculus head evaluated, or the head itself if it will not evaluate.

    What the evaluation asks of each head as it is made, its operands being
    values already. A head standing unevaluated decides nothing, and this is
    what lets a test hold one: `DIF(u, x)` is how ODE1.MTH asks whether `u`
    depends on `x`, and the answer is the derivative and not the `Derivative`.

    A head that stands is asked once more over its operands spelled the way an
    answer spells them, where that is another spelling: a `COSH` is never
    written in an answer, and Derive works an integral out over the
    exponentials the hyperbolic function stands for. BESSEL.MTH's
    `INT(#e^(-z*COSH(t))*COSH(n*t), t, 0, inf)` is two integrals that way, each
    of an exponential, which is what Derive prints; left over the `COSH` the
    integral would stand, and the rewrites after the evaluation would change its
    integrand under it.
    """
    value = _worked_out(head, context)
    if value != head:
        return value
    spelled = _attempt(head, _by_definition)
    if spelled is None or spelled == head:
        return head
    again = _worked_out(spelled, context)
    return head if again == spelled else again


def _worked_out(head: sp.Basic, context: Context) -> sp.Basic:
    """One head by the rule its kind is worked out by."""
    if _unfinished(head, context):
        return head
    if isinstance(head, Approx):
        return _approximation(head, context)
    if isinstance(head, sp.Integral):
        return _integral(head)
    if isinstance(head, (sp.Sum, sp.Product)) and not _searchable(head):
        return head
    if isinstance(head, sp.Product):
        return _product(head)
    if isinstance(head, sp.Sum):
        return _summation(head)
    try:
        value = head.doit(deep=False)
    except Exception:
        return _undecided(head)
    return head if value is None else value


def _unfinished(head: sp.Basic, context: Context) -> bool:
    """Whether a call of a defined function stands unworked under `head`.

    The evaluation enters every such call, so one is left only where a recursion
    reached the nesting bound without arriving - `F(n) := DIF(F(n + 1), x)` - and
    what stands there is no constant: it mentions no `x` only because its body was
    never written in. Differentiating it would answer zero, so the head waits.
    """
    if not context.functions:
        return False
    try:
        calls = head.atoms(AppliedUndef)
    except Exception:
        return False
    return any(type(call).__name__ in context.functions for call in calls)


def _approximation(head: sp.Basic, context: Context) -> sp.Basic:
    """`APPROX(u, n)`: `u` as approximate arithmetic at `n` digits leaves it.

    Exactly what the approX command does to a whole entry, at the digit count
    the call asked for rather than the one the session is set to, and whatever
    precision that is. So the same rules hold inside a call as outside one: a
    number that needs no digits gets none, and `APPROX(2 + 3)` is 5.

    A count that is still no number is one nothing can round to, and the head
    waits for it the way it waits for the value: `PARTS(i_)` asks for a count
    written in the index of the sum it stands in, which is a number once that
    sum has been written out.
    """
    value, digits = head.args
    count = digit_count(digits)
    if count is None:
        return head
    approximate = context.with_precision(Precision.APPROXIMATE, count)
    return approximated(value, approximate)


def _summation(head: sp.Sum) -> sp.Basic:
    """A sum, with a closed form that is a polynomial written factored.

    Derive's summation formulas are factored and stay that way: `SUM(k, k, 0,
    n)` is `n*(n + 1)/2`, `SUM(k^2, k, 1, n)` is `n*(n + 1)*(2*n + 1)/6` and
    `SUM(k^3, k, 0, n)` is `n^2*(n + 1)^2/4`, none of which Simplify then
    multiplies out. Sympy answers with Faulhaber's polynomial expanded, so the
    factoring is done here, where the closed form arrives.

    Here and not in the pipeline, because the pipeline factors nothing and the
    original does not either: `n^2/2 + n/2` authored as it stands is what
    Derive prints back, and turning that into a product is the Factor command's
    business. What is kept is the shape the formula came in, which is a
    property of this answer rather than of every expression equal to it.

    Not gated on getting shorter, because the original does not gate it
    either: `SUM(k^4, k, 1, n)` is longer factored than expanded and Derive
    prints `n*(n + 1)*(6*n^3 + 9*n^2 + n - 1)/30` all the same. Nothing
    after this takes it back apart, because the normal form never distributes
    a product.

    A formula is what a range of unknown length is answered by, and only that.
    A range of known length is added up term by term, and the total is a sum
    like any other: `SOLVE.MTH`'s Lagrange expansion over three terms is
    `3*y^3/2 - y^2 + y` in the manual (9.1, p.230), not that polynomial's
    factorization.
    """
    value = _attempt(head, lambda s: s.doit(deep=False))
    if value is None:
        return _undecided(head)
    if _counted(head):
        return value
    if isinstance(value, sp.MatrixBase):
        # A sum of vectors is summed element by element, and each element is a
        # closed form of its own: `SUM([k, k^2], k, 0, n)` is a vector of the
        # two formulas and not of Faulhaber's two polynomials.
        entries = _attempt(value, lambda m: m.applyfunc(_factored))
        return value if entries is None else entries
    return _factored(value)


def _counted(head: sp.Sum | sp.Product) -> bool:
    """Whether every range of `head` holds a known number of terms.

    Which is what tells a sum that was added up from one a formula answered.
    A bound that is a symbol makes the count unknown, and so does an infinite
    one: neither `SUM(k, k, 1, n)` nor `SUM(2^-k, k, 0, inf)` is a row of terms.
    """
    return all(
        len(limit) == 3 and (limit[2] - limit[1]).is_Integer for limit in head.limits
    )


def _factored(value: sp.Basic) -> sp.Basic:
    """One closed form, factored where there is a polynomial to factor.

    A sum sympy could not do is not a closed form and has nothing to factor.
    """
    if value.has(sp.Sum) or not _polynomial(value):
        return value
    factored = _attempt(value, sp.factor)
    if factored is None:
        return value
    return factored


def _polynomial(value: sp.Basic) -> bool:
    """Whether `value` is a polynomial, and one with a variable in it.

    A number has nothing to factor and `factor` would rewrite it anyway - it
    writes 12 as `2^2*3` - so `SUM(k, k, 1, 5)` stays 15.

    Asked of whatever a head evaluated to, which need not be an expression at
    all: a sum of truth values is a truth value, and that is not a polynomial
    however the question is answered.
    """
    try:
        return bool(value.free_symbols) and value.is_polynomial() is True
    except Exception:
        return False


def _product(head: sp.Product) -> sp.Basic:
    """A product, by sympy first and then by its own definition.

    Sympy has no routine for an infinite product, so `PRODUCT(1 - 1/k^2, k, 2,
    inf)` comes back as it went in. What it does have is the definition: the
    infinite product is the limit of the partial products, and both halves of
    that it can do. So a product over an endless range that sympy leaves
    standing is asked again as `_partial_products`.
    """
    value = _attempt(head, lambda p: p.doit(deep=False))
    if value is None:
        return head
    if not isinstance(value, sp.Product):
        return value
    partial = _partial_products(value)
    return value if partial is None else partial


def _partial_products(head: sp.Product) -> sp.Basic | None:
    """An infinite product as the limit of the products up to `n`, or None.

    This is sympy's own documented way around the gap, and it is the
    definition rather than a trick: the product to `n` is a closed form in `n`
    wherever the body is hypergeometric, and where the limit of that closed
    form exists it is what the infinite product means. `PRODUCT(1 - 1/k^2, k,
    2, inf)` telescopes to a ratio of rising factorials and the limit of that
    is 1/2, which is Derive's answer; Wallis' product comes out `pi/2` the same
    way, and `PRODUCT(k/(k + 1), k, 1, inf)` comes out 0.

    None wherever either half declines - `PRODUCT(COS(pi/2^k), k, 1, inf)` has
    no closed form to take a limit of, and `PRODUCT(1 - x/k, k, 1, inf)`
    depends on the sign of an `x` nothing knows - and the product then stands
    unevaluated, which is the honest answer and the one Derive gives.

    A lower end that is not known finite is refused, which leaves the range
    running from somewhere to infinity and the limit meaning nothing. A
    declared variable is known finite and is allowed: `PRODUCT(1 - 1/k^2, k,
    m, inf)` is `(m - 1)/m`, the same telescoping one step short of the whole.

    An oscillating product is left standing rather than answered. Sympy's
    limit of `(-1)^n` is `nan` and of `(-2)^n` is complex infinity, and
    neither is a value: `PRODUCT(-1, k, 1, inf)` has none, and saying it is
    undefined is a stronger claim than the head makes. That is what the
    matching sums do - `SUM((-1)^k, k, 1, inf)` stands unevaluated - and
    divergence to infinity is a value in the same house style, so
    `PRODUCT((k + 1)/k, k, 1, inf)` is `inf`.

    The closed form is written in gammas, and the limit of it in gammas of
    complex argument, which the reflection formula turns back into a real
    hyperbolic: `PRODUCT(1 + 1/k^2, k, 1, inf)` is `SINH(pi)/pi` and not
    `1/(GAMMA(1 - #i)*GAMMA(1 + #i))`. `simplify` rather than `gammasimp`
    because only the former also cancels what the formula leaves behind, and
    the expression it is asked about is a constant.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 3:
        return None
    index, low, high = head.limits[0]
    if high != sp.oo or not low.is_finite:
        return None
    count = sp.Dummy("n", integer=True, positive=True)
    partial = _attempt(
        head.function,
        lambda body: sp.Product(body, (index, low, count)).doit(deep=False),
    )
    if partial is None or partial.has(sp.Product):
        return None
    value = _attempt(partial, lambda closed: sp.limit(closed, count, sp.oo))
    if value is None or value.has(count, sp.Product, sp.Limit, sp.nan, sp.zoo):
        return None
    reduced = _attempt(value, sp.simplify) if value.has(sp.gamma) else None
    return value if reduced is None else reduced


#: Integration by the methods whose cost is bounded: sympy's tables, the
#: trigonometric rules, Risch, and the manual rules. Not `meijerg`, which
#: answers `INT(#e^x*COS(x), x)` with a page of Bessel functions when it is
#: reached on its own, and not `heurisch`, whose cost is the problem below.
_BOUNDED = {"meijerg": False, "heurisch": False}


def _integral(head: sp.Basic) -> sp.Basic:
    """An integral, by sympy first and then by Derive's rule for a pole inside.

    Sympy's answer is the one to print wherever the integrand is integrable.
    Where it is not, because the integrand blows up strictly inside the
    interval, Derive prints something else, and `_across_a_singularity` is that
    something else.
    """
    value = _by_sympy(head)
    across = _across_a_singularity(head, value)
    if across is not None:
        return across
    beyond = _beyond_convergence(head, value)
    if beyond is not None:
        return beyond
    return _by_linearity(head) if value == head else value


def _by_linearity(head: sp.Basic) -> sp.Basic:
    """An integral sympy left standing, as a sum of integrals where its integrand is a
    sum in disguise.

    Sympy integrates a sum term by term, keeping as an integral only the terms it
    cannot do - but only a sum it is handed. `(x*G(x) + F(x))/G(x)` is a quotient to
    it, and the whole integral is given up for the one term that will not integrate.
    Derive divides the numerator into its terms and answers `INT(F(x)/G(x), x) +
    x^2/2`. So a quotient whose numerator is a sum is written as one integral per
    term, and those are new heads: each is integrated, or left standing, on its own,
    and none is the integral already tried whole.

    An integrand that was a sum to begin with has had its terms tried by sympy
    already, and is not offered again.
    """
    integrand = head.function
    if isinstance(integrand, sp.Add):
        return head
    numerator, denominator = integrand.as_numer_denom()
    if not isinstance(numerator, sp.Add):
        return head
    return sp.Add(
        *(head.func(term / denominator, *head.limits) for term in numerator.args)
    )


def _by_sympy(head: sp.Basic) -> sp.Basic:
    """An integral, by the bounded methods first.

    The heuristic Risch algorithm is what finds the integrals nothing else
    does, and its cost grows with the generators of the integrand: it builds an
    ansatz over every one it can see and solves for the coefficients. With
    nothing symbolic to carry it is quick. `INT(t^(a-1)*(#e^(z*t)*(1-t)^(b-a-1)
    - 1), t, 0, 1/2)` - a real line out of the Kummer function in Derive's own
    utility library - carries three parameters over four generators, and sympy
    spends most of a minute and gigabytes of memory before answering, correctly,
    that it does not know. Simplify must not do that: a minute is not an answer,
    and the same integral comes back unevaluated either way.

    So over a parametrised integrand the expensive method is asked only where
    the ansatz is small enough to be affordable, which `_affordable` is the
    measured rule for. Everything else comes back unevaluated, which is a
    documented answer where a minute of thrashing is not.
    """
    bounded = _attempt(head, lambda h: h.doit(deep=False, **_BOUNDED))
    if bounded is not None and not bounded.has(sp.Integral):
        return _antiderivative(head, bounded)
    if _parametrised(head) and not _affordable(head):
        return head if bounded is None else bounded
    full = _attempt(head, lambda h: h.doit(deep=False))
    return head if full is None else _antiderivative(head, full)


def _across_a_singularity(head: sp.Basic, value: sp.Basic) -> sp.Basic | None:
    """A definite integral read across an interior pole the way Derive reads one.

    Derive simplifies a definite integral by finding a closed-form
    antiderivative and subtracting its limit at the lower endpoint from its
    limit at the upper one, both approached from inside the interval (7.4,
    p.199). It looks for no singularity in between - the manual says finding
    those is the reader's job - so where one is there the answer is the
    difference of the antiderivative anyway. `INT(1/x^3, x, -1, 2)` comes out
    3/8, which is the Cauchy principal value, and `INT(1/x^2, x, -1, 1)` comes
    out -2, which the manual itself calls obviously wrong for an integrand
    positive throughout. Both are what Derive prints, so both are what this
    prints.

    Only where the integral has no value otherwise, which is where sympy has
    split at the pole and found the halves do not cancel. An integrable
    singularity - a removable one, or `1/SQRT(|x|)` - keeps sympy's answer,
    which is the Riemann integral and the same number this rule would reach.
    And only where the pole is strictly inside: at an endpoint the technique is
    valid, the manual keeps it, and `INT(1/x^2, x, -1, 0)` stays infinite.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 3:
        return None
    if not isinstance(value, sp.Expr) or value.is_finite:
        return None
    variable, low, high = head.limits[0]
    if not _poles_inside(head.function, variable, low, high):
        return None
    antiderivative = _integral(sp.Integral(head.function, variable))
    if antiderivative.has(sp.Integral):
        return None
    ends = []
    for point, side in ((high, "-"), (low, "+")):
        end = _attempt(antiderivative, lambda a: sp.limit(a, variable, point, side))
        if end is None or not end.is_finite:
            return None
        ends.append(end)
    return ends[0] - ends[1]


def _beyond_convergence(head: sp.Basic, value: sp.Basic) -> sp.Basic | None:
    """A definite integral's divergent case valued, where it diverges to a sign.

    Sympy answers an improper integral with a parameter in it over the condition
    its convergence test proves: `INT(x^n, x, 1, inf)` is `-1/(n + 1)` where `n <
    -1`, and elsewhere the integral itself, unevaluated. That is where the integral
    diverges, and Derive says so: it works the antiderivative to its limit at each
    end and prints the infinity it finds, so the second case is `inf` to it. Sympy
    will find the same infinity where it knows the sign of the exponent, which the
    parameter it was handed does not say and a dummy of known sign does.

    So the complement of the condition is written as substitutions: `n < -1` fails
    where `n` is `-1 - s` for a positive `s`, and where `n` is `-1` exactly. The
    integral is asked again under each, and where every answer is one infinity,
    that is the case's value. A finite answer, an unevaluated one, or two that
    disagree keeps the integral standing: a convergence test proves convergence
    and not divergence, and an oscillating integrand diverges to no sign at all.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 3:
        return None
    folded = _attempt(value, sp.piecewise_fold)
    if not isinstance(folded, sp.Piecewise) or len(folded.args) != 2:
        return None
    (converged, condition), (standing, otherwise) = folded.args
    if otherwise is not sp.true or standing != head:
        return None
    outcomes = set()
    for parameter, point in _complement(condition):
        found = _attempt(head, lambda h: _by_sympy(h.subs(parameter, point)))
        if found is None or not found.is_infinite or found.is_extended_real is not True:
            return None
        outcomes.add(found)
    if len(outcomes) != 1:
        return None
    return sp.Piecewise((converged, condition), (outcomes.pop(), sp.true))


#: The inequalities a convergence condition is written in, and the only ones
#: whose complement is a sign the parameter can be given.
_INEQUALITIES = (sp.StrictLessThan, sp.StrictGreaterThan, sp.LessThan, sp.GreaterThan)


def _complement(condition: sp.Basic) -> list[tuple[sp.Basic, sp.Basic]]:
    """Where a one-parameter inequality fails, as values of its parameter.

    Each is the parameter written in terms of a dummy whose sign sympy knows: the
    complement of `g > 0` is `g = -s` for a positive `s`, and where the inequality
    was strict, `g = 0` as well. A condition that is no such inequality - on more
    than one parameter, or on the real part of one, or that does not determine its
    parameter - has no complement here, and the empty list leaves nothing valued.
    """
    if not isinstance(condition, _INEQUALITIES) or len(condition.free_symbols) != 1:
        return []
    (parameter,) = condition.free_symbols
    if isinstance(condition, (sp.StrictGreaterThan, sp.GreaterThan)):
        excess = condition.lhs - condition.rhs
    else:
        excess = condition.rhs - condition.lhs
    if excess.is_polynomial(parameter) is not True:
        return []
    shortfall = sp.Dummy("s", positive=True)
    targets = [-shortfall]
    if isinstance(condition, (sp.StrictLessThan, sp.StrictGreaterThan)):
        targets.append(sp.S.Zero)
    points = []
    for target in targets:
        try:
            solutions = sp.solve(sp.Eq(excess, target), parameter)
        except Exception:
            return []
        if len(solutions) != 1:
            return []
        points.append((parameter, solutions[0]))
    return points


def _poles_inside(
    integrand: sp.Basic, variable: sp.Basic, low: sp.Basic, high: sp.Basic
) -> bool:
    """Whether the integrand is known to blow up strictly between the limits.

    Known is the word: a singularity sympy cannot locate, or locates at a point
    it cannot place against the limits, is one Derive's rule is not applied
    over. Nothing is lost by that - the answer without the rule is the answer
    with it wherever the rule does not fire.
    """
    interval = _attempt(low, lambda bound: sp.Interval.open(bound, high))
    if not isinstance(interval, sp.Interval):
        return False
    try:
        poles = sp.singularities(integrand, variable, interval)
    except Exception:
        return False
    return isinstance(poles, sp.FiniteSet) and bool(poles)


def _antiderivative(head: sp.Basic, value: sp.Basic) -> sp.Basic:
    """An indefinite integral's answer in the form Derive writes it in.

    An indefinite integral is one antiderivative out of many: any two of them
    differ by a constant, and which one is printed is a choice rather than a
    result. Derive prints the one with no constant added and never a case split
    where a single antiderivative serves everywhere. Sympy's tables make the
    other choice both times, so the choice is made here.

    Nothing is taken out of a definite integral, where a constant does not
    cancel but is part of the answer, nor out of an integral that was not done:
    neither has an antiderivative in hand to choose among. A definite integral's
    split over a parameter is still dropped where it distinguishes nothing.
    """
    if len(head.limits) != 1 or value.has(sp.Integral):
        return value
    if len(head.limits[0]) != 1:
        return _continuous(value, None)
    variable, integrand = head.limits[0][0], head.function
    if _refuted(value, integrand, variable):
        value = _by_the_manual_rules(head, integrand, variable) or value
    value = _without_constants(_single_case(value, integrand, variable), variable)
    return _continuous(value, variable)


def _continuous(value: sp.Basic, variable: sp.Basic | None) -> sp.Basic:
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
    """
    folded = _attempt(value, sp.piecewise_fold)
    if not isinstance(folded, sp.Piecewise):
        regular, _, _ = _regularized(value, variable)
        return regular
    if len(folded.args) != 2 or folded.args[1][1] is not sp.true:
        return value
    generic, condition = folded.args[0]
    if not isinstance(condition, sp.Ne):
        return value
    regular, found, kept = _regularized(generic, variable)
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
    for divisor in _parametric_divisors(expression, variable):
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


def _parametric_divisors(expression: sp.Basic, variable: sp.Basic | None) -> list:
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
    parameter, is passed over.
    """
    if len(divisor.free_symbols) != 1:
        return None
    (parameter,) = divisor.free_symbols
    coordinate = sp.Dummy("u")
    try:
        inverted = sp.solve(sp.Eq(divisor, coordinate), parameter)
    except Exception:
        return None
    if len(inverted) != 1:
        return None
    shifted = expression.subs(parameter, inverted[0])
    expansion = _attempt(
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


def _refuted(value: sp.Basic, integrand: sp.Basic, variable: sp.Basic) -> bool:
    """Whether this is provably not an antiderivative of the integrand.

    Sympy's Risch implementation answers `INT(1/(x^2 + a), x)` with zero over a
    real `a`, and `INT(x^2/(x^2 + a), x)` with `x`. Both are wrong and neither is
    refused, so an answer is checked rather than trusted.

    Only a decided no counts. `equals` settles what it can and says nothing
    about the rest, and the rest is where the answers this cannot check live: a
    nonelementary integral comes back as the hypergeometric series it is, and
    nothing differentiates that back to the integrand.
    """
    difference = _attempt(value, lambda v: sp.diff(v, variable) - integrand)
    if difference is None:
        return False
    try:
        return difference.equals(0) is False
    except Exception:
        return False


def _by_the_manual_rules(
    head: sp.Basic, integrand: sp.Basic, variable: sp.Basic
) -> sp.Basic | None:
    """The integral again by the rules a calculus course teaches, or None.

    They are the one method that does not reach the algorithm the wrong answers
    come out of, and their answer is put to the same check: what cannot be
    differentiated back is no improvement on what was already in hand.
    """
    manual = _attempt(head, lambda h: h.doit(deep=False, manual=True))
    if manual is None or manual.has(sp.Integral):
        return None
    if _refuted(manual, integrand, variable):
        return None
    return manual


def _single_case(value: sp.Basic, integrand: sp.Basic, variable: sp.Basic) -> sp.Basic:
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
    the integral's or only its constant's is `_continuous`'s question.
    """
    folded = _attempt(value, sp.piecewise_fold)
    if not isinstance(folded, sp.Piecewise):
        return value
    best, condition = folded.args[-1]
    if condition is not sp.true or not _holds(best, integrand, variable):
        return value
    for case, _ in folded.args[:-1]:
        if case.has(*_INVERSE_HYPERBOLIC) and not best.has(*_INVERSE_HYPERBOLIC):
            continue
        if _pays(case, best) and _holds(case, integrand, variable):
            best = case
    return best


#: Not what an antiderivative is printed over. Derive writes every one of these
#: as the logarithm it is - `ASINH(x)` authored on its own comes back
#: `LN(SQRT(x^2 + 1) + x)` - so `INT(SQRT(x^2 + a^2), x)`, whose cases are an
#: `ASINH` where `a` is nonzero and that same logarithm elsewhere, is printed
#: over the logarithm however much shorter the arc is.
_INVERSE_HYPERBOLIC = (sp.asinh, sp.acosh, sp.atanh, sp.acoth, sp.asech, sp.acsch)


def _holds(case: sp.Basic, integrand: sp.Basic, variable: sp.Basic) -> bool:
    """Whether this case is an antiderivative with no condition attached.

    Proof, not evidence: a case is about to be given a domain it was not handed,
    so the derivative has to come back to the integrand symbolically rather than
    at sampled points.
    """
    if case.has(sp.Piecewise):
        return False
    difference = _attempt(case, lambda c: sp.simplify(sp.diff(c, variable) - integrand))
    return difference == 0


def _without_constants(value: sp.Basic, variable: sp.Basic) -> sp.Basic:
    """The antiderivative with every constant of integration taken out of it.

    A term free of the variable is such a constant, and so is a factor free of
    the variable inside a logarithm, `LN(k*u)` being `LN(u)` plus a constant.
    Sympy leaves both: the integral above comes back over `LN(2*b + 2*SQRT(b^2 -
    4*a*c))`, where Derive's answer is over `LN(b + SQRT(b^2 - 4*a*c))`.

    A constant deeper in than that is no constant of integration - it is part of
    whatever holds it - so only the terms of the answer itself are looked at,
    through a factor common to all of them where there is one.
    """
    bare = _attempt(
        value,
        lambda v: v.replace(
            lambda e: isinstance(e, sp.log) and len(e.args) == 1,
            lambda e: _bare_logarithm(e, variable),
            simultaneous=False,
        ),
    )
    if bare is None:
        bare = value
    coefficient, terms = _split(bare, variable)
    if not isinstance(terms, sp.Add):
        return bare
    return _carried(coefficient, [term for term in terms.args if term.has(variable)])


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
    argument = _attempt(call.args[0], sp.factor_terms)
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


def _parametrised(head: sp.Basic) -> bool:
    """Whether the integrand mentions more than the variables integrated over.

    The limits do not count: they are substituted into an antiderivative that
    has already been found, so they are no part of the search that costs.
    """
    variables = {limit[0] for limit in head.limits}
    return bool(head.function.free_symbols - variables)


#: How many bases may carry a symbolic exponent before the ansatz is too big.
#: Two is `INT(x^(a-1)*(1-x)^(b-1), x)`, which sympy answers in under a second;
#: `INT(x^(a-1)*(1-x)^(c-a-1)*(1-x*z)^(-b), x)` is the same shape with a third,
#: and does not finish. Measured over every parametrised integral in Derive's
#: utility library: under this rule each one either answers within a second or
#: is refused, and no refusal loses an answer the rule would have found.
_ANSATZ = 2


def _affordable(head: sp.Basic) -> bool:
    """Whether the heuristic method can be asked about a parametrised integrand.

    Cheap where the integrand is algebraic - a product of powers of polynomials
    in the variable - because each base is one generator and nothing composes
    to make more. `#e^(z*COS(t))*COS(n*t)` is where that stops being true: the
    exponential of a cosine is a generator built on another, and sympy does not
    finish. Neither does a product of three powers with symbolic exponents, so
    the count is bounded as well.
    """
    variables = [limit[0] for limit in head.limits]
    symbolic = 0
    for factor in sp.Mul.make_args(head.function):
        base, exponent = factor.as_base_exp()
        if exponent.has(*variables) or base.is_polynomial(*variables) is not True:
            return False
        if exponent.free_symbols:
            symbolic += 1
    return symbolic <= _ANSATZ


#: The heads a closed form is searched for through: sympy recognises a
#: hypergeometric term by their ratios, and that is the search that costs.
_TERMS = (sp.factorial, sp.factorial2, sp.gamma, sp.rf, sp.ff)


def _searchable(head: sp.Basic) -> bool:
    """Whether looking for a closed form of this sum is known to come back.

    A limit that is not a number sends sympy looking for one by Gosper's
    algorithm. Over a summand carrying a factorial of the index scaled by more
    than one - `(n - 2*k)!`, which is how every classical polynomial family is
    written - that search does not come back at all: `SUM((-1)^k*(n - k)!/(k!*
    (n - 2*k)!)*(2*x)^(n - 2*k), k, 0, n/2)` is Chebyshev's, out of Derive's
    own utility library, and it was still running after forty minutes. There is
    no closed form to be had there, and the sum written out is the answer.

    Narrow on purpose. `SUM(COMB(n, k), k, 0, n)` is `2^n` and `SUM((-1)^k*
    COMB(n, k)*COMB(2*n - 2*k, n)*x^(n - 2*k), k, 0, n/2)` is Legendre's, which
    sympy answers in half a second; neither is refused.

    The other search that does not come back is for a telescoping pair of
    trigonometric terms, which `_trigonometric_terms` is the measured rule for.
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
    """Whether two terms of the summand carry trigonometric functions of the index
    without being each other's shift, which is what sympy's telescoping check
    does not come back from.

    Before anything else, sympy asks of a two-term summand whether one term is the
    other's negation shifted along the index, first by matching and then, where the
    match fails, by solving `L(k + m) + R(k) = 0` for `m`. Over trigonometric
    functions that solve rewrites everything as exponentials and builds
    polynomials in them. `SUM(SIN(a*k) - SIN(b*k), k, 1, n)` takes four seconds
    that way, the four sines the Fourier series of Derive's own `INTEGRAL.MTH`
    comes to once its coefficients are integrated take ninety, and a product on
    each side - `SUM(SIN(k)*COS(2*k) - SIN(3*k)*COS(5*k), k, 1, n)` - never
    finishes at all, whatever the frequencies. A third term is no protection,
    since the pairs are tried in turn.

    Nothing is given up: once that search fails, sympy has no way of summing a
    trigonometric term and the sum comes back as written anyway. Except the one
    pair the match settles - `SIN(k + 1)*COS(k + 1) - SIN(k)*COS(k)` telescopes
    to `SIN(n)*COS(n + 2)` without the solve ever being reached - so the same
    match is made here first, and that pair is let through. Hyperbolic terms
    are not trigonometric to this rule: sympy sums them as the exponentials they
    are, and comes back.
    """
    expanded = _attempt(function, sp.expand)
    if expanded is None:
        return False
    waves = [
        term
        for term in sp.Add.make_args(expanded)
        if any(f.has(index) for f in term.atoms(TrigonometricFunction))
    ]
    if len(waves) < 2:
        return False
    return len(waves) != 2 or not _shifted(waves[0], waves[1], index)


def _shifted(left: sp.Basic, right: sp.Basic, index: sp.Basic) -> bool:
    """Whether `right` is `-left` with the index moved by an integer, the way
    sympy's telescoping asks it: by matching, with `left` the term it shifts."""
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


def _undecided(head: sp.Basic) -> sp.Basic:
    """What a head that raised is worth.

    Only the two-sided limit has an answer here: sympy refuses one whose sides
    disagree, and when they are each other's negation the ambiguity is exactly
    what `±` says. Anything else stays as it was written.
    """
    if not (isinstance(head, sp.Limit) and str(head.args[3]) == "+-"):
        return head
    expression, variable, point = head.args[:3]
    sides = []
    for direction in ("+", "-"):
        try:
            sides.append(sp.Limit(expression, variable, point, dir=direction).doit())
        except Exception:
            return head
    right, left = sides
    if right != -left:
        return sp.nan
    # Two infinite sides that are each other's negation are unsigned infinity,
    # which sympy has an object of its own for and the notation writes `±inf`.
    return sp.zoo if right in (sp.oo, -sp.oo) else PlusMinus(right)


# -- the FACTOR, EXPAND and SOLVE heads ---------------------------------------


def _commanded(name: str, values: list, context: Context) -> sp.Basic:
    """`FACTOR`, `EXPAND`, `SOLVE` or `RATE` run over its evaluated arguments.

    `FACTOR(u, amount, x, y, ...)`, `EXPAND(u, amount, x, y, ...)` and
    `SOLVE(u, x)` are the author-line spellings of three of the commands, so
    simplifying a line that carries one runs it - which is what the original
    does, and what makes the functions worth having rather than inert heads
    that print back as themselves.

    What a command works on is a simplified operand, exactly as the command
    itself would have it, and rounding is the last thing done to its answer and
    not the first thing done to its question: the operand is simplified at the
    exact precision. A factorization, an expansion or a solution is then held
    as `Commanded`, which is what keeps everything after the evaluation from
    rewriting it; a call over one - `RHS(SOLVE(x^2 - 5*x + 6 = 0, x))` - takes it
    as the value it is.
    """
    exact = context.with_precision(Precision.EXACT)
    args = [
        _transform(value, exact, decide=False) if not place or name == "RATE" else value
        for place, value in enumerate(values)
    ]
    match name:
        case "SOLVE":
            return Commanded(_solve_head(args, context))
        case "RATE":
            return _rate_head(args, context)
    return Commanded(_factor_head(name, args, context))


def _factor_head(name: str, args: list, context: Context) -> sp.Basic:
    """One head, read as its target, its amount and its variables.

    The arguments after the first are a word naming the amount, a list of
    variables, or both in that order; the manual allows any of them to be left
    out. A word that names no amount is read as a variable, which is what
    keeps a call nobody can make sense of - a Derive 6 `FACTOR(u, Turing)` -
    an expression rather than an error. `EXPAND(u, Complex)` is such a call:
    Complex is an amount Factor offers and Expand does not.
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


def _solve_head(args: list, context: Context) -> sp.Basic:
    """`SOLVE(u, x)` and its longer forms, as the vector of what they solve to.

    This is the 3.x and 4.x shape and it is the one worth having: the answer is
    a *vector* of relations, which `DIMENSION` counts and `RHS` distributes
    over, and which the shipped libraries are written against. Derive 5 turned
    it into a disjunction of equations; that form says the same thing and no
    Derive 4 worksheet can read it.

    Each element is one of the entries the command would have appended, in the
    same order - so a system contributes one inner vector per solution, and an
    equation with no solutions gives the empty vector.

    `SOLVE(u = v, x, a, b)` searches `[a, b]` numerically. It does so in every
    precision mode: the original's help says "if in approximate mode", the shipped
    files use it as though it always applied, and always-numeric is both the
    more useful reading and the one that makes the call mean one thing.
    """
    target, *rest = args
    variables = _solved_for(rest[0]) if rest else ()
    bounds = (rest[1], rest[2]) if len(rest) >= 3 else None
    answers = solutions(target, context, variables, bounds)
    if not answers:
        return sp.Matrix(0, 0, [])
    return InertVector(*answers)


#: The name the rate is searched for under. Nothing can collide with it: the
#: search is only run over a contract whose every argument is a number, so the
#: equation holds no other name at all.
_RATE = "i"


def _rate_head(args: list, context: Context) -> sp.Basic:
    """`RATE(n, p, v, f, t, a, b)`: the rate between `a` and `b` the contract implies.

    The one financial function that is no closed form, and so the one done
    here rather than in the conversion: it is the bounded `SOLVE` above,
    applied to the annuity `PVAL` and the rest are read off, and it inherits
    that search whole - the interval decides which rate is found, and a
    contract of symbols is one no search can start on.

    Where the interval holds no rate the call stays as it stands, which is
    what the original leaves behind: `RATE(36, -300, 9000, 0, 0, 0.5, 1)` is
    worth nothing better than itself.
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
    """The call itself, standing, where a command has no answer to give."""
    return sp.Function(name)(*args)


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
    """Finish off the constants sympy leaves standing.

    A factorial of a fraction is one: `(3/2)!` is a number, and a number that
    is still written as a call has not been simplified. Sympy will evaluate it
    as a gamma, so that is what it becomes. This is not gated on getting
    shorter - evaluating a constant is a simplification whatever it costs to
    write down.
    """
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


#: The two heads section 6.9 writes as a ratio of factorials: `COMB(z, w)` is
#: `z!/(w!*(z - w)!)` and `PERM(z, w)` is `z!/(z - w)!`.
_AS_FACTORIALS: dict[type, Callable[..., sp.Basic]] = {
    sp.binomial: _as_binomial,
    sp.ff: _as_falling,
}


def _ratios(expression: sp.Basic) -> sp.Basic:
    """`_AS_FACTORIALS` written out, everywhere but under a calculus head.

    A `COMB` inside a sum that has already declined to evaluate is left as it
    is. Written out it makes a summand nobody has offered to sympy, and the
    answer's own Simplify would offer it - and over a ratio of factorials of the
    index that search does not come back: `SUM(COMB(n, k), k, 0, n)` is `2^n`
    above zero and the sum itself elsewhere, and the sum it is left with would
    be searched forever. That is `_searchable`'s point, one head further out.
    """
    written = _AS_FACTORIALS.get(type(expression))
    if written is not None:
        return written(*expression.args)
    if isinstance(expression, CALCULUS_HEADS) or not expression.args:
        return expression
    arguments = [_ratios(argument) for argument in expression.args]
    if all(written is argument for written, argument in zip(arguments, expression.args)):
        # The expression itself, and not a rebuild of it: rebuilding is what
        # `replace` does too, and it is a normalization of everything it walks
        # through - a matrix inverse comes back a power of a matrix - so
        # nothing that holds none of these heads is put through it.
        return expression
    return expression.func(*arguments)


def _factorials(expression: sp.Basic) -> sp.Basic:
    """The heads section 6.9 writes as factorials, written as factorials.

    `GAMMA(z)` is `(z - 1)!`, and the two of `_AS_FACTORIALS` are ratios. The
    factorial is the notation's own spelling and these are the ones it falls
    back on, so an answer that reached one of them has not been written yet.
    Not gated on getting shorter - every one of the three is longer written
    out, and is still the form asked for.

    After the rewrites, because `combsimp` is where most of the gammas come
    from and where the other two come back: a ratio of three factorials is a
    binomial to it, and a shorter expression than the ratio. The ratios are
    written out before it as well, in `_by_definition`, so that one it can
    shorten is shortened - `COMB(x, 2)` is `x*(x - 1)/2` and not a ratio of
    factorials that only the next Simplify would reduce.
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
# Each rewrite from here on is one rule about kernels. Which of them is offered, in
# which mode, and whether it has to pay, is the business of the three tables at the
# end of the module.


def _distributed(expression: sp.Basic) -> sp.Basic:
    """Every matrix product multiplied over the sums inside it.

    `a . (b + c)` is `a . b + a . c` and `(b + c) . a` is `b . a + c . a`,
    which section 8.8 states as rules and not as economies: distributing adds
    an operation, so the gate would decline both, and the manual asks for them
    anyway. `_complementary_arcs` is forced for the same reason.

    This is the whole of the rewrite stage for an expression holding a matrix,
    and deliberately so. The rest of it is written for scalars and mangles a
    matrix rather than simplifying it - `radsimp` hands back a plain product of
    two matrices, which is no expression sympy can go on working with - and
    what the rules of section 8.8 do is done by sympy's own construction long
    before anything here is offered.
    """
    return expression.replace(
        _is_matrix_product, lambda product: product.expand(), simultaneous=False
    )


def _is_matrix_product(expression: sp.Basic) -> bool:
    return isinstance(expression, sp.MatMul) and any(
        isinstance(factor, sp.MatAdd) for factor in expression.args
    )


#: The hyperbolics, which are exponentials by 6.5 the way their inverses are
#: logarithms by 6.6.
_HYPERBOLIC = (sp.sinh, sp.cosh, sp.tanh, sp.coth, sp.sech, sp.csch)

def _as_arc_cosh(argument: sp.Basic) -> sp.Basic:
    """`ACOSH(z)` as `2*LN(SQRT(z + 1) + SQRT(z - 1)) - LN(2)`, which 6.6 says.

    Sympy's own rewrite reaches `LN(SQRT(z - 1)*SQRT(z + 1) + z)` - the same
    value in an uncollected form, and nothing downstream collects it. The two
    agree over the whole plane: `(SQRT(z + 1) + SQRT(z - 1))^2` is
    `2*(z + SQRT(z - 1)*SQRT(z + 1))`, and each root has an argument in
    `(-pi/2, pi/2]`, so their sum does too and doubling its logarithm crosses
    no cut.

    A number is left to sympy, which works the roots out and reaches the
    shorter answer for it: `ACOSH(2)` is `LN(SQRT(3) + 2)` there and
    `LN((SQRT(3) + 1)^2/2)` here, the same number written longer.
    """
    if argument.is_number:
        return sp.acosh(argument)
    return 2 * sp.log(sp.sqrt(argument + 1) + sp.sqrt(argument - 1)) - sp.log(2)


def _as_arc_coth(argument: sp.Basic) -> sp.Basic:
    """`ACOTH(z)` as `LN((z + 1)/(z - 1))/2`, which section 6.6 says it is.

    Sympy's own rewrite reaches `LN(1 + 1/z)/2 - LN(1 - 1/z)/2`, which is one
    logarithm short of it and nothing downstream collects either. The quotient
    `(1 + 1/z)/(1 - 1/z)` is `(z + 1)/(z - 1)` wherever either side is written.
    """
    return sp.log((argument + 1) / (argument - 1)) / 2


#: The heads written out here rather than left to sympy's own `rewrite`, which
#: reaches the right value in a form the manual does not print.
_WRITTEN_AS: dict[type, Callable[..., sp.Basic]] = {
    sp.acosh: _as_arc_cosh,
    sp.acoth: _as_arc_coth,
}

#: What each function that Derive keeps no answer in is written over. `SEC` and
#: `CSC` are the reciprocals 6.4 names them for and `ASEC` and `ACSC` the arcs
#: of those reciprocals, the twelve hyperbolics are the exponentials and
#: logarithms of 6.5 and 6.6, and `ERFC` is the complement 6.11 names it for.
#: `STEP`, `MIN` and `MAX` are the closed forms of 6.7, where a step is half a
#: sign away from a half and the two extremes are the midpoint of a pair give
#: or take half the distance between them.
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
    **{head: sp.log for head in _INVERSE_HYPERBOLIC},
}


def _by_definition(expression: sp.Basic) -> sp.Basic:
    """The functions that are spellings for something else, spelled out.

    None of these is a direction to rewrite in, so none is gated or offered as
    a setting: an answer holding a `SECH` is one Derive never writes, and
    neither is one holding an `ERFC` or a `MAX`. Several of them lengthen what
    they replace - `MIN(x, y)` is two operations and its answer is seven - and
    the gate would decline every one of those.

    What a head turns into is left to sympy wherever sympy's form is the
    manual's - `TANH(z)` is `(#e^(2*z) - 1)/(#e^(2*z) + 1)` in both - and
    written out by `_WRITTEN_AS` where it is not. The factorial ratios are
    written out here as well as in `_factorials`, before `combsimp` rather than
    after, so that one it can shorten is shortened.
    """
    for head, written in _WRITTEN_AS.items():
        if not expression.has(head):
            continue
        expression = _forced(expression, lambda e, h=head, w=written: e.replace(h, w))
    if expression.has(*_AS_FACTORIALS):
        expression = _forced(expression, _ratios)
    for head, over in _DEFINED_OVER.items():
        if not expression.has(head):
            continue
        expression = _forced(expression, lambda e, h=head, o=over: _defined(e, h, o))
    return expression


def _defined(expression: sp.Basic, head: type, over: type) -> sp.Basic:
    """Every `head` in `expression` written over `over`, and nothing else.

    Head by head rather than over the whole expression, because `rewrite` takes
    everything it can reach with it: a `SIN` beside a `SINH` would go to
    exponentials too, and one asked for in `#e` is not one asked for in `#i`.
    """
    return expression.replace(head, lambda *args: head(*args).rewrite(over))


def _gated(expression: sp.Basic, rewrite: Rewrite) -> sp.Basic:
    """`rewrite`, if the result is simpler than what went in."""
    candidate = _attempt(expression, rewrite)
    if candidate is None:
        return expression
    return candidate if _pays(candidate, expression) else expression


def _forced(expression: sp.Basic, rewrite: Rewrite) -> sp.Basic:
    """`rewrite`, whatever it costs: a direction asked for, a fact, or a spelling."""
    candidate = _attempt(expression, rewrite)
    return expression if candidate is None else candidate


def _attempt(expression: sp.Basic, rewrite: Rewrite) -> sp.Basic | None:
    try:
        candidate = rewrite(expression)
    except Exception:
        return None
    return candidate if isinstance(candidate, sp.Basic) else None


def _pays(candidate: sp.Basic, previous: sp.Basic) -> bool:
    """Whether a rewrite earned its keep.

    One variable fewer always counts, however long the answer: a superfluous
    variable is the first thing the manual says a simplified expression does
    not have. One denominator fewer under another counts the same way, being
    the other thing a simplified expression does not have. Failing both, the
    rewrite has to be shorter.
    """
    try:
        if candidate == previous:
            return False
        if len(candidate.free_symbols) < len(previous.free_symbols):
            return True
        if _compound(candidate) != _compound(previous):
            return _compound(previous)
        return sp.count_ops(candidate) < sp.count_ops(previous)
    except Exception:
        return False


def _compound(expression: sp.Basic) -> bool:
    """Whether a denominator in `expression` has a denominator of its own.

    Which is the one thing Derive never leaves standing whatever it costs:
    `DIF(ATAN(x/a), x)` is `1/(a + x^2/a)` to sympy and `a/(a^2 + x^2)` to the
    original, and the two are the same length, so nothing that counts
    operations can tell them apart.

    A number underneath is not a denominator of this kind. `x + 1/2` is a sum
    of a variable and a number wherever it stands, and putting it over two is
    the longer way to write it.
    """
    for power in expression.atoms(sp.Pow):
        if not power.exp.is_negative:
            continue
        for inner in power.base.atoms(sp.Pow):
            if inner.exp.is_negative and not inner.base.is_number:
                return True
    return False


#: Where the three ways of writing one real quantity sit in a product. Each is
#: a power of the other two - `ABS(u) = u*SIGN(u)` and `SIGN(u)^2 = 1` - so a
#: group is no more than the exponent every one of them carries.
_ABSOLUTE, _SIGNUM, _BARE = 0, 1, 2


def _folded_signs(expression: sp.Basic) -> sp.Basic:
    """Every product of a real argument with its absolute value and its sign."""
    return expression.replace(
        lambda e: _folded_product(e) is not None,
        lambda e: _folded_product(e),
        simultaneous=False,
    )


def _folded_product(expression: sp.Basic) -> sp.Basic | None:
    """This product with `ABS(u)`, `SIGN(u)` and `u` collected into one another.

    For a real `u` the three are one quantity written three ways, so a product
    holding more than one of them says the same thing twice: `ABS(u)*SIGN(u)`
    and `ABS(u)/SIGN(u)` are both `u`, `u/ABS(u)` is `SIGN(u)`, `u*SIGN(u)` is
    `ABS(u)` and `SIGN(u)^2` is `1`.

    Zero is where the identities are a licence rather than a fact: `SIGN(0)` is
    zero, so `ABS(u)/SIGN(u)` is undefined there and `u` is not. Derive removes
    the singularity and so does this, on the same terms that make `u/u` come
    back `1`. Off the real line there is nothing to remove and nothing to fold:
    `SIGN(z)` is `z/|z|`, whose square is not one, so an argument nothing
    declares real is left as it was written.
    """
    groups, rest = _signed_powers(expression)
    if not any(_worth_folding(*exponents) for exponents in groups.values()):
        return None
    folded = [
        _shortest_spelling(argument, *exponents)
        if _worth_folding(*exponents)
        else _written(argument, *exponents)
        for argument, exponents in groups.items()
    ]
    candidate = sp.Mul(*rest, *folded)
    return None if candidate == expression else candidate


def _worth_folding(absolute: int, signum: int, bare: int) -> bool:
    """Whether this group says the same thing twice.

    One `ABS` or one `SIGN` by itself is what it is, and a bare power with
    neither is no business of this rule. What is left over is a group that
    holds two of the three, or a sign the argument carries more than once, or
    a sign under a division: all three collapse.
    """
    if not (absolute or signum):
        return False
    present = sum(1 for exponent in (absolute, signum, bare) if exponent)
    return present > 1 or abs(absolute) > 1 or signum not in (0, 1)


def _signed_powers(
    expression: sp.Basic,
) -> tuple[dict[sp.Basic, list[int]], list[sp.Basic]]:
    """This product read as groups of one real argument, and the factors that are not.

    A factor joins the group of the argument it is about - `ABS(u)^2` and
    `1/SIGN(u)` and `u` are all the group of `u` - and anything else, an
    argument sympy cannot call real or a power the exponents cannot be added
    up over, stays a factor of its own.

    A range is not one of these arguments. `SIGN(INTERVAL(-1, 1))` is the
    three-point set the interval's sign can be and not a number the identities
    hold for, so a bound stays the factor it was written as.
    """
    groups: dict[sp.Basic, list[int]] = {}
    rest: list[sp.Basic] = []
    for factor in sp.Mul.make_args(expression):
        base, exponent = factor.as_base_exp()
        if isinstance(base, sp.Abs):
            slot, argument = _ABSOLUTE, base.args[0]
        elif isinstance(base, sp.sign):
            slot, argument = _SIGNUM, base.args[0]
        else:
            slot, argument = _BARE, base
        if (
            exponent.is_Integer
            and argument.is_extended_real
            and not argument.has(sp.AccumBounds)
        ):
            groups.setdefault(argument, [0, 0, 0])[slot] += int(exponent)
        else:
            rest.append(factor)
    return groups, rest


def _written(argument: sp.Basic, absolute: int, signum: int, bare: int) -> sp.Basic:
    """This group as the author wrote it."""
    return sp.Abs(argument) ** absolute * sp.sign(argument) ** signum * argument**bare


def _shortest_spelling(
    argument: sp.Basic, absolute: int, signum: int, bare: int
) -> sp.Basic:
    """The shortest of the ways this group can be written.

    Trading one absolute value for the argument leaves a sign behind, and two
    signs are one, so the total power and the parity of the sign are all the
    group is: every `k` of `ABS(u)^k * u^(total - k) * SIGN(u)^(parity - k)`
    says it, with the last exponent taken to zero or one. The rule is therefore
    no direction to write `ABS` or to write `SIGN` but a choice of whichever
    `k` comes out shorter, and ties - the spelling the author wrote among them
    - go to what was already there.
    """
    total = absolute + bare
    parity = (absolute + signum) % 2
    spellings = [_written(argument, absolute, signum, bare)] + [
        sp.Abs(argument) ** count
        * argument ** (total - count)
        * sp.sign(argument) ** ((parity - count) % 2)
        for count in sorted({0, 1, absolute, total})
    ]
    return min(spellings, key=sp.count_ops)


def _denested(expression: sp.Basic) -> sp.Basic:
    """Every radical denested on its own.

    One radical at a time rather than the whole expression at once, because
    `sqrtdenest` gives up on a sum of nested roots and gets each of them
    separately: `SQRT(5 + SQRT(24)) + SQRT(5 - SQRT(24))` is `2*SQRT(3)` only
    if the two terms are denested apart.

    `sqrtdenest` is a square root's rule and the whole of what sympy offers.
    Section 3.8's other example is a cube root of a sum of surds, which is
    `_flat_root`'s.
    """
    return expression.replace(
        lambda e: isinstance(e, sp.Pow) and e.exp.is_Rational and e.exp.q > 1,
        lambda e: sp.sqrtdenest(e) if e.exp is sp.S.Half else _flat_root(e),
        simultaneous=False,
    )


#: How big a polynomial denesting a root of surds may work in. An nth root of
#: a number written over k square roots satisfies one of degree n*2^k at worst,
#: which is twelve for 3.8's own `(243*SQRT(5) - 294*SQRT(3))^(1/3)` and four
#: as it turns out. The bound is read off the radical before any polynomial is
#: built, so a taller tower of roots costs nothing to decline: what it would
#: cost is the point, a minimal polynomial of a few hundred terms being
#: minutes of work to prove a root nobody expected does not exist.
_DENESTING_DEGREE = 12

#: The variable the minimal polynomial of a radical is written in.
_ROOT = sp.Dummy("root")


def _flat_root(power: sp.Pow) -> sp.Basic:
    """A root of a sum of surds written without the outer root, where it can be.

    `(243*SQRT(5) - 294*SQRT(3))^(1/3)` is `3*SQRT(5) - 2*SQRT(3)`: the cube
    root is a number of the very field the radicand is written in, so it has a
    spelling with one root fewer. Whether it does is a question about the
    minimal polynomial - the root denests exactly when that polynomial has a
    linear factor over the field the surds generate - and factoring over an
    extension is what answers it.

    The linear factor is the candidate and not the answer: a quartic over the
    field offers four of them, one per branch, and only one is the value of the
    principal root the radicand was written with. So a candidate is kept only
    once raising it back is the radicand again and its value is the radical's
    own.
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
        if candidate is None or candidate.count_ops() >= power.count_ops():
            continue
        if sp.expand(sp.radsimp(candidate**exponent.q - power.base)) != 0:
            continue
        if _same_number(candidate, power):
            return candidate
    return power


def _surd_field(radicand: sp.Basic) -> tuple[sp.Basic, ...] | None:
    """The square roots `radicand` is written over, or None if it is not written
    over square roots.

    A sum of rational multiples of square roots of rationals is what denesting
    has a field for, and the degree of that field is what the bound above is
    read from. Anything else - a transcendental, a nested root, a cube root
    among the terms - is left alone rather than handed to a factoring that
    would have to build the field first.
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
    """The root of `factor`, if it is linear in the variable it is written in."""
    try:
        polynomial = sp.Poly(factor, _ROOT)
    except sp.PolynomialError:
        return None
    if polynomial.degree() != 1:
        return None
    above, below = polynomial.all_coeffs()
    return sp.radsimp(-below / above)


def _same_number(candidate: sp.Basic, value: sp.Basic) -> bool:
    """Whether `candidate` is the very number `value` stands for.

    The algebra has already said the two are roots of one polynomial; this is
    what tells the roots of it apart, and thirty digits are enough for that
    because two branches of a root are a whole angle apart and not a rounding
    apart. The comparison is against the size of the value for the same
    reason: what would fail a fixed tolerance is a large root, where the two
    branches are further apart still.
    """
    try:
        difference = complex(sp.N(candidate - value, 30))
        size = abs(complex(sp.N(value, 30)))
    except (TypeError, ValueError):
        return False
    return abs(difference) < 1e-20 * (1 + size)


def _flattened(expression: sp.Basic) -> sp.Basic:
    """Every constant written over a nested radical reduced to the field its own
    surds generate.

    `_denested` works one radical at a time, which is as far as a radical's own
    rule reaches: 3.8 p.47's
    `(SQRT(SQRT(3) + 1) + SQRT(3*SQRT(3) + 3))/SQRT(24*SQRT(3) + 40)` is 1/2,
    and no one of its four roots is anything simpler on its own. What makes it
    a half is the whole quotient at once - the numerator is `(SQRT(3) + 1)^(3/2)`
    and the denominator twice that - so the question `_flat_root` asks of a root
    is asked here of a sum or a product: whether its value has a spelling in the
    field the surds it is written over generate.

    Only a constant holding a root of something irrational is asked. Roots of
    rationals are combined by sympy's own arithmetic as they are built, so a
    constant with none of the nesting has already had every reduction of this
    kind that there is to have, and the minimal polynomial would only confirm it
    at a price.
    """
    return expression.replace(
        lambda part: isinstance(part, sp.Add | sp.Mul), _flat_part, simultaneous=False
    )


def _flat_part(part: sp.Add | sp.Mul) -> sp.Basic:
    """`part` with the number in it flattened.

    A sum keeps every constant term of it as a term of its own, so a sum is
    reached by the walk that finds it. A product does not: `x` times the
    quotient of 3.8 is one product of five factors, and the number the walk
    should be asking about has to be gathered back out of it first.
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


#: How big a field the value of a constant is looked for in, read off the
#: radicals written in it the way `_DENESTING_DEGREE` is read off one radical:
#: the product of their orders, which is sixteen for the quotient of 3.8 and
#: four as it turns out. The cost of being generous is what sets it - a tower
#: two roots taller is a minimal polynomial of degree sixty-four, minutes of
#: work to prove that a value nobody expected to be simpler is not.
_FLATTENING_DEGREE = 16


def _flat_value(constant: sp.Basic) -> sp.Basic:
    """`constant` written over one field fewer, where its value lies in one.

    The same three steps as `_flat_root`: the minimal polynomial says what the
    value is a root of, factoring that over the field the surds generate offers
    a linear factor per branch, and only the branch whose value is this one is
    kept - thirty digits telling the branches apart, since a shorter spelling
    that is a different number is the one mistake this can make.
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
        if _same_number(candidate, constant):
            return candidate
    return constant


def _radical_field(constant: sp.Basic) -> tuple[tuple[sp.Basic, ...] | None, int]:
    """The roots of rationals `constant` is written over and the degree they
    bound, or None where it is not a nested radical of them.

    The generators are the roots whose radicand is a rational, which are the
    ones a field can be built from; the degree is the product of the orders of
    every root in the constant, an upper bound on the field it all lives in.
    Anything but rational arithmetic over such roots - a transcendental, a
    variable, a root of a negative - is nothing to ask a minimal polynomial
    about, and neither is a constant whose roots are all roots of rationals,
    which is already as flat as it gets.
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
    collecting those roots and the order of every root on the way."""
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
            # The qth root itself and not the power of it that is written: the
            # two generate one field, and the root is the shorter way to say so.
            generators.add(sp.Pow(part.base, sp.Rational(1, part.exp.q)))
    return True


# -- the trigonometric, logarithmic and power rules --------------------------


def _tangent_squares(expression: sp.Basic) -> sp.Basic:
    """An even reciprocal power of a cosine, written over the tangent instead.

    Derive has no secant of its own to write `1/COS(x)^2` as, and does not leave
    it standing either: it answers `DIF(LN(COS(x)), x, 2)` with `-TAN(x)^2 - 1`,
    and writes `TAN(x)^2 + 1` wherever else that power stands. Only an even
    power, and only a negative one - `1/COS(x)^3` and `COS(x)^2` are both left
    as they are, neither being a power of the identity.

    The cosine has to be the whole of what the number in front multiplies, so
    that only a quotient of a number by a squared cosine is rewritten:
    `SIN(x)/COS(x)^2` keeps its cosine, there being no tangent identity for a
    sine over one. That is why this walks the terms rather than the powers -
    a power knows nothing about what stands beside it.

    Forced rather than gated: `TAN(x)^2 + 1` is the longer of the two, and the
    original writes it anyway. Left out of both Trigpower directions, each of
    which is asking for the powers in one named function.
    """
    if isinstance(expression, sp.Add):
        return sp.Add(*(_tangent_squares(term) for term in expression.args))
    coefficient, rest = expression.as_coeff_Mul()
    base, exponent = rest.as_base_exp()
    if not (isinstance(base, sp.cos) and exponent.is_Integer and exponent < 0):
        return expression
    half = -exponent / 2
    if not half.is_Integer:
        return expression
    return coefficient * (sp.tan(base.args[0]) ** 2 + 1) ** half


#: The inverse functions that pair off, each with the one it complements.
#: `ASIN(u) + ACOS(u)` is a right angle wherever both are defined; the tangent
#: pair is a right angle turned the way its argument is signed. The secant pair
#: is here by way of the first: `ASEC` and `ACSC` are written over `ACOS` and
#: `ASIN` before any of this is offered, and add up through them.
_COMPLEMENT = {
    sp.asin: sp.acos,
    sp.acos: sp.asin,
    sp.atan: sp.acot,
    sp.acot: sp.atan,
}


def _complementary_arcs(expression: sp.Basic) -> sp.Basic:
    """Two complementary arcs of one argument, added up, wherever they occur."""
    return expression.replace(
        lambda e: _right_angle(e) is not None,
        lambda e: _right_angle(e),
        simultaneous=False,
    )


def _right_angle(expression: sp.Basic) -> sp.Basic | None:
    """This sum with a complementary pair of arcs replaced by what it is.

    The pair has to carry the same coefficient as well as the same argument,
    which is what lets the rule hold in degree measure: there each arc comes
    out multiplied by `180/pi`, and the two of them add to `90`.
    """
    if not isinstance(expression, sp.Add):
        return None
    arcs: dict[tuple, int] = {}
    for index, term in enumerate(expression.args):
        arc, coefficient = _arc(term)
        if arc is not None:
            arcs[(type(arc), coefficient, arc.args[0])] = index
    for (func, coefficient, argument), index in arcs.items():
        other = arcs.get((_COMPLEMENT[func], coefficient, argument))
        if other is None or other == index:
            continue
        angle = _quarter_turn(func, argument)
        if angle is None:
            continue
        rest = [t for i, t in enumerate(expression.args) if i not in (index, other)]
        return sp.Add(coefficient * angle, *rest)
    return None


def _arc(term: sp.Basic) -> tuple[sp.Basic | None, sp.Basic]:
    """The one complementable arc in `term`, and what multiplies it."""
    arc, others = None, []
    for factor in sp.Mul.make_args(term):
        if arc is None and type(factor) in _COMPLEMENT:
            arc = factor
        else:
            others.append(factor)
    return arc, sp.Mul(*others)


def _quarter_turn(func: type, argument: sp.Basic) -> sp.Basic | None:
    """What the pair adds to, or None where it is no right angle at all.

    The tangent pair is the one that turns: `ATAN(t) + ACOT(t)` is `pi/2` above
    zero and `-pi/2` below, which is `pi*SIGN(t)/2`. That is an answer rather
    than a guess - both halves are written down, and neither is assumed - where
    Derive answers `pi/2` regardless and is wrong for the negative half.

    Zero is the one point the turn cannot speak for. `ACOT(0)` is `pi/2` by the
    convention sympy and Derive share, so the sum there is `pi/2` where `SIGN`
    would say nothing at all. An argument that cannot be negative is therefore
    answered `pi/2` outright, which is exact over the whole of it; what is left
    over disagrees at the single point its `ACOT` jumps over, and agrees
    everywhere else.

    Complex arguments are no part of this: `SIGN(z)` is `z/|z|` off the real
    line, and the identity does not hold there anyway - at `#i/2` the sum is
    `-pi/2`. A pair nothing declares real stays as it was written.
    """
    if func in (sp.atan, sp.acot):
        if not argument.is_real:
            return None
        if argument.is_nonnegative:
            return sp.pi / 2
        if argument.is_negative:
            return -sp.pi / 2
        return sp.pi * sp.sign(argument) / 2
    return sp.pi / 2


def _simpler_arcs(expression: sp.Basic) -> sp.Basic:
    """Every arc rewritten about a shorter argument, wherever one is there."""
    return expression.replace(
        lambda e: _shorter_arc(e) is not None,
        lambda e: _shorter_arc(e),
        simultaneous=False,
    )


def _shorter_arc(expression: sp.Basic) -> sp.Basic | None:
    """This arc as the arc of one side of the triangle it describes.

    A right triangle with legs `u` and `1` has hypotenuse `SQRT(u^2 + 1)`, so
    the angle whose sine is `u/SQRT(u^2 + 1)` is the angle whose tangent is
    `u`; the same triangle read off the other way makes the angle whose tangent
    is `u/SQRT(1 - u^2)` the angle whose sine is `u`. Either way the ratio is a
    longer way of writing an argument the triangle already holds.

    Both hold for a real `u` alone - the root has a branch cut a complex
    argument crosses, and the two sides then disagree by a period - so an
    argument nothing declares real is left as it stands.
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
    """Whether this difference is zero once it has been multiplied out.

    The hypotenuse is written however the author wrote it, so recognising the
    triangle behind `(x + 1)/SQRT(x^2 + 2*x + 2)` means expanding the square
    rather than comparing two trees.
    """
    return bool(sp.expand(difference).is_zero)


def _half_angles(expression: sp.Basic) -> sp.Basic:
    """Every trigonometric call rewritten about the one angle they all share.

    `1/(1 + TAN(a)*TAN(a/2))` is `COS(a)`, and no rule that looks at one call
    at a time can see it: what cancels here is the relation between the two
    arguments, and sympy's rules each rewrite a call by itself. Written about
    the half angle both are multiples of, the tangents cancel and what is left
    is a cosine.

    The angle is the greatest common measure of the arguments, so nothing is
    expanded further than it has to be.
    """
    measure = _common_angle(expression)
    if measure is None:
        return expression
    return sp.trigsimp(TR11(TR2(expression), base=measure))


def _common_angle(expression: sp.Basic) -> sp.Basic | None:
    """The greatest angle every trigonometric argument is a multiple of.

    None where there is no such angle - arguments built on different things, or
    one that is no rational multiple of anything - and none where every
    argument is a whole multiple of it. That last case is the multiple-angle
    expansion `trigsimp` already offers, and this rule has nothing to add to
    it; a fractional multiple is what nothing else handles.
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
    """`a*SIN(z) + b*COS(z)` as a single sine, wherever it occurs.

    Ours rather than fu's: `TR10i` collects a sine and a cosine of the same
    angle only when their coefficients match, and the rule holds for any two.
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


def _nonnegatively(rewrite: Rewrite) -> Rewrite:
    """`rewrite` run with every nonnegative variable standing up as a positive one.

    6.2 states the logarithm rules for a nonnegative argument, and sympy's
    carry a strictly positive one: `logcombine` and `expand_log` ask whether an
    argument `is_positive`, so a variable declared `[0, inf)` is one they pass
    over. Between the two conditions lies the single point zero, where both
    sides of every rule are the same infinity, so the manual's condition is the
    one to work under.

    A declared variable only. An undeclared one is neither, and `LN(x^2 - x) -
    LN(x)` stays apart under it, which is the whole of what the rules are gated
    on.
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


def _extracted_powers(expression: sp.Basic) -> sp.Basic:
    """`LN(z^k)` written as `k*LN(z)`, everywhere the rule holds."""
    return expression.replace(
        lambda e: _extracted_power(e) is not None,
        lambda e: _extracted_power(e),
        simultaneous=False,
    )


def _extracted_power(expression: sp.Basic) -> sp.Basic | None:
    """`k*LN(z)`, if `expression` is a logarithm of `z^k` that may be taken apart.

    6.2 gives the rule as `k*LN(x) <-> LN(x^k)` for a rational `k` where `x` is
    nonnegative, and for any `x` where `-1 < k <= 1`.
    """
    if not isinstance(expression, sp.log) or len(expression.args) != 1:
        return None
    argument = expression.args[0]
    if argument.is_Rational and argument.is_positive:
        power = _perfect_power(argument)
        return None if power is None else power[1] * sp.log(power[0])
    if not argument.is_Pow:
        return None
    base, exponent = argument.args
    if not exponent.is_Rational:
        return None
    if not (base.is_nonnegative or bool(-1 < exponent <= 1)):
        return None
    return exponent * sp.log(base)


def _perfect_power(number: sp.Rational) -> tuple[sp.Rational, sp.Integer] | None:
    """`(z, k)` where `number` is `z^k`, for the largest whole `k` above one.

    A positive rational is a power of itself whenever its numerator and its
    denominator are powers to a common exponent: 256 is 2^8 and 4/9 is (2/3)^2,
    while 12 and 4/3 are powers of nothing but themselves.
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
    """Powers of one base in a product added, and nothing else.

    Not an economy but Derive's spelling: `DIF(x^n, x)` is `n*x^n/x` to sympy and
    `n*x^(n - 1)` to the original, and the two are the same length, so a gate that
    counts operations would keep the first. Exponential Expand is the setting that
    asks for powers apart, and under it Derive answers `n*x^n/x`.

    `combine="exp"` is the exponent rule on its own: it leaves two bases under
    one exponent alone, which is the part of `powsimp` that is a direction to
    rewrite in rather than a normal form.
    """
    return sp.powsimp(expression, combine="exp")


def _real_roots(expression: sp.Basic) -> sp.Basic:
    """An odd root of a negative number, taken real.

    Sympy has already pulled the sign out - `(-8)^(1/3)` is `2*(-1)^(1/3)` -
    so the rule to apply is about that leftover factor, and an odd denominator
    is exactly when the real root exists.
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
    """A numeric power that is not real, written as real part plus imaginary.

    Section 4.5's promise: a complex result comes out in rectangular form.
    `(1 + #i)^3` is `-2 + 2*#i` and the principal `(-8)^(1/3)` is
    `1 + SQRT(3)*#i`, neither of which sympy reaches on its own - it leaves a
    power of a sum folded and pulls the sign of a negative base out as
    `2*(-1)^(1/3)`.

    Only a power, and only one whose base and exponent are both numbers: what
    `(-x)^(1/3)` is depends on `x`, and there is no rectangle to write until
    something says. And only where the two parts come out plainly written -
    the seventh root of -1 is `COS(pi/7) + #i*SIN(pi/7)`, which is the polar
    form the promise was about avoiding, and the fifth root buys its rectangle
    with a radical inside a radical. Both are left as they stand.
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
    """Whether `value` is arithmetic and roots of whole numbers, and no more."""
    return not value.atoms(sp.Function) and all(
        bool(power.base.is_Rational)
        for power in value.atoms(sp.Pow)
        if not power.exp.is_Integer
    )


# -- the kernel layer --------------------------------------------------------
#
# Every rewrite above is a rule about kernels - the trigonometric, logarithmic,
# exponential, sign, arc and radical calls an expression is a rational function of -
# and what follows is the whole of where each is offered. Three tables, applied by one
# driver: the identities offered in every mode, kept where they pay; the directions
# the mode settings ask for; and the spelling conventions an answer is written in,
# applied once the normal form has written its sums. The output vocabulary is the one
# spelling applied first rather than last: it is the language the identities are
# stated in, and `TANH(z)` reaches `(#e^(2*z) - 1)/(#e^(2*z) + 1)` only by being
# written over exponentials before the denominators are cleared.


class _Rule(NamedTuple):
    """One rewrite of the kernel layer, with what it is called and why it is there.

    `kept` is how an answer is taken: `_gated` keeps one that pays, `_forced` keeps
    any. `yields_to` names a mode setting the rule is the Auto behavior of: a spelling
    the user has asked away from is not one to impose, so `TAN(x)^2 + 1` is written
    for `1/COS(x)^2` only while Trigpower is Auto, Cosines asking for the cosine.
    """

    name: str
    rewrite: Rewrite
    kept: Callable[[sp.Basic, Rewrite], sp.Basic]
    reason: str
    yields_to: str | None = None


def _rewritten(expression: sp.Basic, context: Context) -> sp.Basic:
    """The kernels reduced: every identity that pays, then every direction asked for.

    Over the vocabulary `_by_definition` writes, and before the normal form, which is
    what writes the sums the rules leave behind.
    """
    if expression.has(sp.MatrixExpr):
        return _forced(expression, _distributed)
    expression = _by_definition(expression)
    expression = _applied(expression, _IDENTITIES, context)
    for setting, directions in _DIRECTED.items():
        rules = directions.get(getattr(context, setting), ())
        expression = _applied(expression, rules, context)
    return expression


def _spelled(expression: sp.Basic, context: Context) -> sp.Basic:
    """The answer with Derive's spelling conventions, once its sums are written.

    After the normal form, so that nothing after them rewrites what they write. Each
    is the form Derive prints whether or not it is the shorter one: `1/COS(x)^2` is
    `TAN(x)^2 + 1`, and `LN(256)` is `8*LN(2)`.
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
    """`combsimp`, where there is something combinatorial and it is affordable."""
    if expression.has(*_COMBINATORIAL) and sp.count_ops(expression) <= _COMBSIMP:
        return sp.combsimp(expression)
    return expression


def _on_powers(rewrite: Rewrite) -> Rewrite:
    """`rewrite`, where what it changed is a power.

    The power rules rebuild every product they pass through, and sympy's product
    distributes a number over a sum as it is built: `powsimp` hands
    `3*x^2*(2*y + 1)` back as `x^2*(6*y + 3)` having collected nothing. Derive keeps
    the first, and an answer that holds the same powers it was given is the same
    answer, so that is the one kept.
    """

    def run(expression: sp.Basic) -> sp.Basic:
        candidate = rewrite(expression)
        if candidate.atoms(sp.Pow, sp.exp) == expression.atoms(sp.Pow, sp.exp):
            return expression
        return candidate

    return run


#: The identities, offered in every mode. Gated ones are economies, kept where
#: `_pays` says the answer is simpler; forced ones are facts about the kernels they
#: relate, where gating would keep the longer side for being no longer.
_IDENTITIES = (
    _Rule("signs", _folded_signs, _forced,
          "ABS(u)*SIGN(u) is u and u/ABS(u) is SIGN(u), for a real u"),
    _Rule("combinatorial", _combinatorial, _gated,
          "a ratio of factorials is a binomial or a polynomial when it is shorter"),
    # The arcs and angles are ours rather than sympy's, which collects neither these
    # arcs nor their tangents, writes none of them about a shorter argument, and has
    # no rule that spans two angles at once.
    _Rule("simpler arcs", _simpler_arcs, _gated,
          "ASIN(u/SQRT(u^2 + 1)) is ATAN(u): an arc about the shorter argument"),
    _Rule("complementary arcs", _complementary_arcs, _forced,
          "ASIN(u) + ACOS(u) is pi/2 and ATAN(u) + ACOT(u) is pi*SIGN(u)/2"),
    _Rule("half angles", _half_angles, _gated,
          "calls of fractional multiples of one angle cancel written about it"),
    _Rule("denested", _denested, _gated,
          "SQRT(5 + SQRT(24)) is SQRT(2) + SQRT(3): each radical denested"),
    _Rule("flattened", _flattened, _gated,
          "a constant over nested radicals reduced in the field its surds span"),
    _Rule("one denominator", sp.radsimp, _gated,
          "a denominator under a denominator is cleared, #e^(-z) put over #e^z"),
)

#: The directions, keyed by the setting that asks for them and its value. Auto, where
#: it has rules here, means "only where it cancels", and those are gated; Collect and
#: Expand are instructions rather than options, and forced. No rule is ever forced
#: past a domain: the logarithm rules are sympy's without `force`, run over
#: `_nonnegatively`'s reading of the declarations, and the power rules carry sympy's
#: own validity gate for `(#e^z)^k -> #e^(k*z)`, so the direction is all that is
#: chosen here.
_DIRECTED: dict[str, dict[str, tuple[_Rule, ...]]] = {
    # Principal is sympy's own convention; Real takes `(-8)^(1/3)` to be -2, and Any
    # lifts the domain proof off the power rules, so `SQRT(x^2)` is `x`.
    "branch": {
        Branch.REAL: (
            _Rule("real roots", _real_roots, _forced,
                  "an odd root of a negative number is its real root"),
        ),
        Branch.ANY: (
            _Rule("any powdenest", lambda e: sp.powdenest(e, force=True), _forced,
                  "a power of a power is one power, whatever the base"),
            _Rule("any powsimp", lambda e: sp.powsimp(e, force=True), _forced,
                  "powers over one exponent are one power, whatever the bases"),
        ),
    },
    "trigonometry": {
        Direction.AUTO: (
            _Rule("trigsimp", sp.trigsimp, _gated,
                  "trigonometric identities, where they cancel something"),
        ),
        Direction.COLLECT: (
            _Rule("phase angles", _phase_angles, _forced,
                  "a*SIN(z) + b*COS(z) is one sine, which fu's TR10i has only for a = b"),
            _Rule("TR8", TR8, _forced, "products of calls written as sums"),
            _Rule("TR7", TR7, _forced, "powers of cosines reduced to multiple angles"),
        ),
        Direction.EXPAND: (
            _Rule("expand_trig", sp.expand_trig, _forced,
                  "calls of sums and multiples written over single angles"),
            _Rule("special angles", lambda e: e.rewrite(sp.sqrt), _gated,
                  "the algebraic values of the special angles that come with it"),
        ),
    },
    "trigpower": {
        TrigPower.SINES: (
            _Rule("TR6", TR6, _forced, "even powers of cosines written over sines"),
        ),
        TrigPower.COSINES: (
            _Rule("TR5", TR5, _forced, "even powers of sines written over cosines"),
        ),
    },
    # Auto is the mixed direction 6.2 describes: the sum rule rightward here, so
    # `3*LN(2) + LN(5)` is `LN(40)`, and the power rule leftward in `_SPELLINGS`, so
    # `LN(256)` is `8*LN(2)` the way `SQRT(12)` is `2*SQRT(3)`, shorter or not.
    # `LN(x^2 - x) - LN(x)` collects only once `x` is declared away from the values
    # that would make it false.
    "logarithm": {
        Direction.AUTO: (
            _Rule("logcombine", _nonnegatively(sp.logcombine), _gated,
                  "logarithms added into one, where it is shorter"),
        ),
        Direction.COLLECT: (
            _Rule("logcombine", _nonnegatively(sp.logcombine), _forced,
                  "logarithms added into one"),
        ),
        Direction.EXPAND: (
            _Rule("expand_log", _nonnegatively(sp.expand_log), _forced,
                  "a logarithm of a product taken apart"),
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
            _Rule("powsimp", _on_powers(sp.powsimp), _forced, "powers collected"),
        ),
        Direction.EXPAND: (
            _Rule("expand_power_exp", _on_powers(sp.expand_power_exp), _forced,
                  "a power of a sum split"),
            _Rule("expand_power_base", _on_powers(sp.expand_power_base), _forced,
                  "a power of a product split"),
        ),
    },
}

#: The spelling conventions, applied after the normal form. None is an economy and
#: none is gated: several lengthen what they replace, and are still the only way
#: Derive writes the answer.
_SPELLINGS = (
    # After the branch, which is what settles how many values a root has: with Real
    # asked for, `(-8)^(1/3)` is `-2` and there is no rectangle to write.
    _Rule("rectangular", _rectangular, _forced,
          "a numeric complex power is real part plus imaginary part"),
    _Rule("tangent squares", _tangent_squares, _forced,
          "1/COS(u)^(2*k) is (TAN(u)^2 + 1)^k", yields_to="trigpower"),
    _Rule("extracted powers", _extracted_powers, _forced,
          "LN(b^k) is k*LN(b) where 6.2 allows it", yields_to="logarithm"),
    _Rule("collected powers", _collected_powers, _forced,
          "powers of one base are one power: n*x^n/x is n*x^(n - 1)",
          yields_to="exponential"),
)
