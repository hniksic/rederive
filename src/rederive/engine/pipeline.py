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
from rederive.engine.intervals import covering, decided, settled
from rederive.engine.normal import normal_form
from rederive.engine.printer import is_written
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


@noting_approximations
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

    Not a second evaluator. The only difference from `simplify` is the precision
    the same pipeline runs at, which is what the manual says approX is. It is what
    an expression that is an `APPROX` call simplifies to, and so what the approX
    command, which simplifies `APPROX` of what it is given, comes to.
    """
    context = context or Context()
    approximate = context.with_precision(Precision.APPROXIMATE, digits)
    return simplify(node, approximate, state)


def _whole_approximation(
    node: Node, context: Context
) -> tuple[Node, int | None] | None:
    """The operand and digit count of an expression that is an `APPROX` call.

    Simplifying such an expression is approX of the operand at the call's digit
    count, its answer written as approximate: the original simplifies
    `APPROX(1/2)` to 0.5 and `APPROX(1/3, 3)` to 0.333. Anywhere further in, the
    call is worth the rational it rounds to and nothing marks it approximate:
    `APPROX(1/2) + 1/3` is 5/6 there.

    A reference to an entry is followed, since the original reads `#1` as the
    expression it names. None where the expression is not such a call, or where
    its count is no workable number, which is ordinary Simplify's to handle.
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
    back as the three equations it stands for. An equation or an inequation of two
    matrices is still one statement about them, and where the elements decide it
    the answer is that one truth value; see `_whole`.
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
    """An equation or inequation of two matrices, from what each pair of elements says.

    Two matrices are equal when every pair of elements is and different when any
    pair is, so `[1, 2] = [1, 3]` is `false` and `[1, 2] /= [1, 3]` is `true`. An
    equation not yet decided is the equations of its elements, to be solved as a
    system; an inequation not yet decided says that some pair differs, which no
    one of its elements says, and it stands as written.
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
    difference = attempt(relation, lambda r: _rewritten(r.lhs - r.rhs, context))
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
    expression = forced(expression, _spliced_cases)
    # A root of a perfect power is taken ahead of the intervals, which settle the
    # absolute value it leaves: `SQRT(x^2 + 2*x + 1)` is `-x - 1` below -1.
    expression = forced(expression, _perfect_powers)
    # Before the rewrites rather than after, so that what the declared
    # intervals settle is settled for them too: `ABS(x) + ABS(x - 1)` over
    # `(0, 1)` is a sum that cancels, and only once both bars are off.
    expression = settled(expression, context)
    expression = _divergent(expression, context)
    expression = _rewritten_beside_standing(expression, context)
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


def _spliced_cases(expression: sp.Basic) -> sp.Basic:
    """Every conditional whose else case is a conditional, as one with the cases of
    both, each condition asked once.

    A closed form that holds under a condition comes with the head itself as the
    else case, and that head worked out again comes with the same condition: a
    sum of `y*x^k` is `y` times a conditional whose else case is the sum of `x^k`,
    which is the conditional on `ABS(x) < 1` once more. Where the first condition
    fails the second does too, so the inner case is dead and only its own else case
    is left.
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
    """Each standing series over a declared variable that diverges, as the infinity
    it goes to.

    A series whose terms tend to a limit other than zero diverges, to `inf` where
    that limit is positive or `inf` and to `-inf` where it is negative or `-inf`.
    Sympy finds the limit of `h^k` only knowing which side of one `h` is on, so the
    limit is taken over what the declaration allows `h`, written in a positive
    variable that sympy can be told about (see `covering`): `SUM(h^k, k, 0, inf)`
    is `inf` where `h` was declared above one, being the else case of the
    conditional the closed form comes with.
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
    """The infinity a series diverges to whose terms tend to `limit`, or None where
    `limit` does not say that it diverges."""
    if limit is sp.oo or (limit.is_finite and limit.is_positive):
        return sp.oo
    if limit is sp.S.NegativeInfinity or (limit.is_finite and limit.is_negative):
        return sp.S.NegativeInfinity
    return None


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


def _decide_simplified(test: sp.Basic, context: Context) -> bool | None:
    """Whether the test of an `IF` holds, by everything Simplify would make of it.

    What `_decide` settles is asked first, being quicker. A test it leaves open is
    simplified whole, as the same test standing on a line of its own would be, and
    decides where that comes out `true` or `false`: `x < 1 AND x > 3` is a range
    that is empty, and `(x^2 + x^2*y^2)/(x^2*y) = (1 + y^2)/y` an equation whose
    sides cancel to one value. ODE1.MTH's `HOMOGENEOUS` asks whether a derivative
    is zero, and the derivative is zero only once it is simplified.
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
    """Every reading of `test` worth asking, easiest first."""
    if not isinstance(test, Relational):
        return [test]
    readings = [test, attempt(test, lambda t: t.func(t.lhs, t.rhs))]
    difference = attempt(test, lambda t: _rewritten(t.lhs - t.rhs, context))
    if difference is not None:
        readings.append(attempt(test, lambda t: t.func(difference, 0)))
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
    value = _answer_in_words(head, context)
    if value is not None:
        return value
    spelled = attempt(head, _by_definition)
    if spelled is None or spelled == head:
        return head
    again = _answer_in_words(spelled, context)
    return head if again is None else again


def _answer_in_words(head: sp.Basic, context: Context) -> sp.Basic | None:
    """What `head` works out to, in the notation's own functions, or None if it stands.

    An answer sympy can give only in a function the notation has no name for is no
    answer here. `INT(1/SQRT(1 - x^4), x)` is a hypergeometric function to sympy and
    an integral Derive cannot do, and the integral is what is shown: the same
    head left standing as for an integral sympy itself declines, which the next
    command can still work on. A sympy head that is a spelling of functions the
    notation has is written in them first, by `in_vocabulary`, and only what is
    left after that decides.
    """
    value = _worked_out(head, context)
    if value == head:
        return None
    value = forced(value, in_vocabulary)
    return value if is_written(value) else None


def _worked_out(head: sp.Basic, context: Context) -> sp.Basic:
    """One head by the rule its kind is worked out by."""
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


#: The functions that are constant between their jumps, whose derivative is zero
#: wherever there is one.
_PIECEWISE_CONSTANT = (sp.floor, sp.ceiling, sp.sign, sp.Heaviside)


def _flat_between_jumps(derivative: sp.Basic) -> sp.Basic:
    """A derivative with the derivative of every step function taken as zero.

    A floor, sign or unit step is constant between its jumps and has no derivative
    at them, so its derivative is 0 wherever it exists, the convention a derivative
    of a step function is written in: `DIF(FLOOR(x)^2, x)` is 0 and `DIF(x*SIGN(x),
    x)` is `SIGN(x)`. Sympy writes the derivative at a jump as a `DiracDelta`, which
    is no function, and leaves one of a floor standing, perhaps under a `Subs` of
    the floor's argument; all of them are that zero.
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
    value = attempt(head, lambda s: s.doit(deep=False))
    if value is None:
        return _undecided(head)
    if _counted(head):
        return value
    value = _over_the_counts(head, value)
    if isinstance(value, sp.MatrixBase):
        # A sum of vectors is summed element by element, and each element is a
        # closed form of its own: `SUM([k, k^2], k, 0, n)` is a vector of the
        # two formulas and not of Faulhaber's two polynomials.
        entries = attempt(value, lambda m: m.applyfunc(_factored))
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


#: How many counts of terms a closed form is checked at before it answers for all.
_COUNTS_CHECKED = 6


def _over_the_counts(head: sp.Sum, value: sp.Basic) -> sp.Basic:
    """The closed form of a sum whose summand involves its own upper limit, held
    to the integer counts of terms the sum is written for.

    Sympy sums `SUM(COMB(n, k), k, 0, n)` as the hypergeometric series it would be
    over every `k`, and states where that series converges, over a real `n` and over
    any other variable: `2^n` where `n > -1`, `(x + 1)^n` under a condition on
    `ABS(x)` as well. But the sum has a number of terms only for an integer `n`, and
    there the series stops at `n` by itself, so the convergence condition does not
    describe the sum at all: `(x + 1)^n` is the sum for every `x`, which is the
    binomial theorem. Its condition is dropped where the closed form agrees with the
    terms added up for the first several counts. Each of those ranges holds at least
    one term, which is what a formula is written for: `SUM(COMB(n, k), k, 0, n)` at
    `n = -1` is no sum of binomial coefficients but an empty range.

    The same check is made of a closed form sympy gives with no condition at all,
    because the symbolic shortcuts it tries first can be wrong at the smallest
    counts: it answers `SUM((-1)^k*COMB(n, k), k, 0, n)` with 0, where the sum is
    1 at `n = 0`. Such an answer is replaced with the hypergeometric one, `0^n`,
    where that one holds, and the sum stands where nothing does.
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
    """Whether `candidate` is the sum `head` for the first `_COUNTS_CHECKED` values
    of `bound` that give its range at least one term, whatever the other variables."""
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
    """A sum or product sympy does not do, by the rule its summand's shape answers
    to, or None where none does.

    A sum of sines or cosines of the index, a product of exponentials and a sum of
    logarithms each have a closed form that follows from one identity, where sympy
    either leaves them standing or never comes back from the search: see
    `_sum_of_waves`, `_product_of_powers` and `_sum_of_logarithms`. A range of known
    length is added up term by term instead.
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
    """A sum of sines and cosines of a linear function of the index, in closed form.

    The terms `SIN(a*k + b)` for `k` from `p` to `q` are `N = q - p + 1` points
    equally spaced around a circle, and their sum is the imaginary part of a
    geometric sum of `#e^(#i*(a*k + b))`, which is

        SIN(N*a/2)/SIN(a/2) * SIN(a*(p + q)/2 + b),

    and the cosine likewise with `COS` of the middle angle: `SUM(SIN(k), k, 1, n)` is
    `SIN(n/2)*SIN((n + 1)/2)/SIN(1/2)`. Where the step `a` is a multiple of `2*pi`
    every term is the same, and the sum is `N*SIN(b)`, the value the formula tends to
    there; where the step is a variable that may be such a multiple the answer is an
    `IF` on it, as the geometric sum `SUM(x^k, k, 0, n)` is one on `x = 1`.

    A pair of waves that are each other's negation shifted along the index is left
    to sympy, which telescopes it to the two ends: `SUM(SIN(k + 1) - SIN(k), k, 1,
    n)` is `SIN(n + 1) - SIN(1)`, shorter than the difference of two such products.

    Terms with the same step and phase are gathered first, and a term that holds no
    trigonometric function of the index is summed as a sum of its own. A product of
    waves, or a wave whose amplitude depends on the index, is no such sum, and is
    left to `_searchable`: sympy has no way of summing it and may not come back from
    looking.
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
    """A product of powers of one positive base, as that base to the sum of the
    exponents: `PRODUCT(#e^k, k, 1, n)` is `#e^(n*(n + 1)/2)` and `PRODUCT(#e^(1/k^2),
    k, 1, inf)` is `#e^(pi^2/6)`.

    The exponential is continuous, so an infinite product is the base to the
    infinite sum, and to a sum that diverges it is `inf` or 0. Where the sum has no
    closed form the product has none either and stands: `PRODUCT(#e^(1/k), k, 1, n)`
    is the base to a harmonic number.
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
    """A sum of logarithms, as the logarithm of the product of their arguments where
    that has a closed form: `SUM(LN(k), k, 1, n)` is `LN(n!)` and `SUM(LN(1 + 1/k), k,
    1, n)`, the product telescoping, is `LN(n + 1)`.

    The logarithm of a product is the sum of the logarithms only of positive
    factors, so every term the range reaches has to be known positive; and the
    logarithm is continuous on them, so an infinite sum is the logarithm of the
    infinite product.
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
    """One closed form, factored where there is a polynomial to factor.

    A sum sympy could not do is not a closed form and has nothing to factor.
    """
    if value.has(sp.Sum) or not _polynomial(value):
        return value
    factored = attempt(value, sp.factor)
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
    value = attempt(head, lambda p: p.doit(deep=False))
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
    """Whether the terms of `expanded` holding trigonometric functions of the index
    are a pair sympy telescopes, see `_trigonometric_terms`."""
    waves = [
        term
        for term in sp.Add.make_args(expanded)
        if any(f.has(index) for f in term.atoms(TrigonometricFunction))
    ]
    return len(waves) == 2 and _shifted(waves[0], waves[1], index)


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


def _limit(head: sp.Limit) -> sp.Basic:
    """A limit, taken of the closed forms of the sums and products under it.

    A sum or product under the limit that stands unevaluated stands because its
    closed form is written in a function the notation has no name for, and the
    limit cannot be taken of the head itself. `SUM(1/k^2, k, 1, n)` is a
    generalized harmonic number, which is no answer, but its limit `pi^2/6` is.
    So the limit is taken of the closed forms, and only what it comes to is held
    to the vocabulary.

    A constant factor is taken outside the limit before it is asked for, as the
    limit laws allow. Sympy's series machinery can lose such a factor when it is a
    product of gammas of complex arguments, and would answer the limit of the
    partial products of `1 + 1/k^2`, which is `SINH(pi)/pi`, with 1.
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
        # Taken as the limit at `inf` of the reflection, which is the same limit:
        # sympy's own way to `-inf` has `2^x` go to `inf` there.
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
    """A limit that turns on the sign of a function of its one parameter, as the
    conditional on where that function is positive and where negative.

    `LIM(a^x, x, inf)` is `#e^(inf*SIGN(LN(a)))` to sympy: `inf` where `a > 1`, 0
    where `0 < a < 1`, and nothing that says so. Each side is the answer with the
    sign written in, and what is on neither - `a = 1`, and an `a` the logarithm is
    not real at - is the limit itself, still to be taken. A sign of the bare
    parameter is left as it is, `inf*SIGN(a)` saying it as plainly as a conditional
    would.
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
    """A limit over a real parameter, from its values where the parameter is positive
    and where it is negative.

    Not told the sign of `a`, sympy takes `LIM(ATAN(x/a), x, inf)` to `ATAN(±inf/a)`,
    an infinity of no sign inside the arc tangent, and refuses `LIM(TANH(a*x), x,
    inf)` altogether. Told it, sympy answers `pi/2` and `-pi/2`, or 1 and -1. Where
    one expression in the absolute value of the parameter, or that times its sign,
    is both answers, it is the limit for every value but zero, the one value at
    which the expression is a different function of the variable altogether.
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
    """Whether `constant` is a number proved finite and not zero: by the assumptions,
    or where they do not settle it, as for `GAMMA(1 + #i)`, by its structure. A product
    of such numbers is one, and so is a power of one to a finite exponent, an
    exponential of a finite number, and a gamma function off its poles, which has no
    zeros."""
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
    """`expression` with each sum and product in it replaced by its closed form,
    whatever functions that is written in, where sympy has one."""

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
    """`expression` with the sign and absolute value of an `INTERVAL` worked out.

    Both are monotone on either side of zero, so each maps bounds to bounds:
    `SIGN(INTERVAL(-1, 1))` is `INTERVAL(-1, 1)` and `ABS(INTERVAL(-2, 1))` is
    `INTERVAL(0, 2)`. Sympy leaves both standing, and its limits produce the
    first: `LIM(x*SIN(x), x, inf)` comes back as `inf` times the sign of the
    bounds of `SIN(x)`, which once worked out is `INTERVAL(-inf, inf)`, the values
    `x*SIN(x)` keeps coming back to. Bounds that are not numbers are left alone,
    there being no deciding which side of zero they lie on.
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
    """What a head that raised is worth.

    Only the two-sided limit has an answer here: sympy refuses one whose sides
    disagree, and when they are each other's negation the ambiguity is exactly
    what `±` says. Sides that agree are the limit: `SIN(1/x)/x` reaches
    `INTERVAL(-inf, inf)` from both. A side sympy cannot take leaves the limit
    as it was written, and so does anything else.
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
    # Two infinite sides that are each other's negation are unsigned infinity,
    # which sympy has an object of its own for and the notation writes `±inf`.
    return sp.zoo if right in (sp.oo, -sp.oo) else PlusMinus(right)


# -- the FACTOR, EXPAND and SOLVE heads ---------------------------------------


def _commanded(
    name: str, values: list, divisors: list, context: Context
) -> sp.Basic:
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

    `divisors` are what a `SOLVE`'s equation divides by as written, which its
    simplified operand may no longer show.
    """
    exact = context.with_precision(Precision.EXACT)
    args = [
        _transform(value, exact, decide=False) if not place or name == "RATE" else value
        for place, value in enumerate(values)
    ]
    match name:
        case "SOLVE":
            return Commanded(_solve_head(args, divisors, context))
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


def _solve_head(args: list, divisors: list, context: Context) -> sp.Basic:
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
    answers = solutions(target, context, variables, bounds, divisors)
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
    write down. So is the sign or absolute value of an `INTERVAL`, which is an
    `INTERVAL` again.
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
    **{head: sp.log for head in INVERSE_HYPERBOLIC},
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

    The sympy heads the notation has no name for are spelled first, by
    `in_vocabulary`, so that what they are written in is spelled with the rest.
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
    """Every `head` in `expression` written over `over`, and nothing else.

    Head by head rather than over the whole expression, because `rewrite` takes
    everything it can reach with it: a `SIN` beside a `SINH` would go to
    exponentials too, and one asked for in `#e` is not one asked for in `#i`.
    """
    return expression.replace(head, lambda *args: head(*args).rewrite(over))


def _gated(expression: sp.Basic, rewrite: Rewrite) -> sp.Basic:
    """`rewrite`, if the result is simpler than what went in."""
    candidate = attempt(expression, rewrite)
    if candidate is None:
        return expression
    return candidate if pays(candidate, expression) else expression


def _unnested(expression: sp.Basic, rewrite: Rewrite) -> sp.Basic:
    """`rewrite`, if the result is simpler or has radicals nested less deep.

    A root taken out of a root is a simplification on its own, even where the
    terms it leaves are more: `SQRT(2 + SQRT(3))` is `SQRT(6)/2 + SQRT(2)/2` and
    `(2 + SQRT(5))^(1/3)` is `SQRT(5)/2 + 1/2`. Not past twice the length, though,
    which is no longer a spelling anyone would prefer.
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
    """`rewrite`, if the result is simpler or has no call left in a denominator.

    A quotient of calls that cancels to a polynomial in them is simplified whether
    or not it is shorter, the way `(x^3 - 1)/(x - 1)` is: `SIN(3*x)/SIN(x)` is
    `2*COS(2*x) + 1`, which is no fewer operations. Not past twice the length,
    though, as with `_unnested`.
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
    """Every `SIGN` of a power to a negative exponent, written without the division.

    `SIGN(1/u)` is `SIGN(u)` and `SIGN(1/u^2)` is 1 for a real `u`, and `SIGN(1/ABS(u))`
    is 1 for any: where `u` is zero the reciprocal is undefined, and everywhere else
    the two have the same sign. A limit that grows like `x^2/a` comes back as
    `inf*SIGN(1/a)`, which is `inf*SIGN(a)`.
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


#: Functions that are odd and increasing over the whole real line, so that for a real
#: `u` the sign of `f(u)` is the sign of `u`.
_SIGN_KEEPING = (sp.atan, sp.asinh, sp.sinh, sp.tanh, sp.erf)

#: Functions that are even, so that for a real `u` they are the same of `ABS(u)` as
#: of `u`.
_EVEN = (sp.cos, sp.sec, sp.cosh, sp.sech)


def _symmetries(expression: sp.Basic) -> sp.Basic:
    """Every `SIGN` of an odd function and every even function of an `ABS`, undone.

    For a real `u`, `SIGN(ATAN(u))` is `SIGN(u)` and so is `SIGN(u^3)`, and
    `COS(ABS(u))` is `COS(u)`. Ahead of the definitions, which write the hyperbolic
    functions over exponentials where neither symmetry can be seen any more. Off the
    real line neither holds, so an argument nothing declares real is left alone.
    """

    def signed(head: sp.sign) -> sp.Basic:
        inner = head.args[0]
        if isinstance(inner, _SIGN_KEEPING) and inner.args[0].is_extended_real:
            return sp.sign(inner.args[0])
        # A negative exponent is `_signs_of_reciprocals`' business, which knows that
        # `SIGN(1/ABS(u))` is 1.
        if isinstance(inner, sp.Pow) and inner.exp.is_positive:
            base, exponent = inner.args
            if base.is_extended_real and exponent.is_integer and exponent.is_odd:
                return sp.sign(base)
        return head

    def even(call: sp.Basic) -> sp.Basic:
        inner = call.args[0]
        if isinstance(inner, sp.Abs) and inner.args[0].is_extended_real:
            return call.func(inner.args[0])
        return call

    # Until nothing changes, since what one unwrapping exposes can be another:
    # `SIGN(ATAN(x)^3)` is `SIGN(ATAN(x))` before it is `SIGN(x)`.
    for _ in range(8):
        rewritten = expression.replace(lambda e: isinstance(e, sp.sign), signed)
        rewritten = rewritten.replace(lambda e: isinstance(e, _EVEN), even)
        if rewritten == expression:
            break
        expression = rewritten
    return expression


def _conjugate_pairs(expression: sp.Basic) -> sp.Basic:
    """Every product of a complex quantity and its conjugate, as its squared modulus.

    `CONJ(z)*z` is `ABS(z)^2`, and so is each pair of factors in a product of
    powers of the two to whole exponents of one sign: `z^2*CONJ(z)` is
    `z*ABS(z)^2`, and `1/(z*CONJ(z))` is `1/ABS(z)^2`.
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
    once raising it back is the radicand again and it is proved the radical itself,
    as `proved_zero` proves the difference zero.
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
    a linear factor per branch, and only the branch proved to be this very number,
    as `proved_zero` proves their difference zero, is kept, since a shorter
    spelling that is a different number is the one mistake this can make.
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


#: How large a field the values of the trigonometric constants are sought in: the
#: degree of the cyclotomic field their angles lie in, times the orders of the roots
#: beside them, a bound on how long the minimal polynomial takes. Sympy writes a
#: cosine past a quarter turn as the sine of an angle with twice the denominator, so
#: an eleventh of pi becomes a twenty-second, and the degree that covers it is 20.
_TRIGONOMETRIC_DEGREE = 24
_TRIGONOMETRIC_SIZE = 10**12


def _trigonometric_constants(expression: sp.Basic) -> sp.Basic:
    """Every sum or product of trigonometric values of rational multiples of pi
    that is a rational number or a quadratic surd, written as one.

    `COS(pi/7) - COS(2*pi/7) + COS(3*pi/7)` is 1/2 and
    `SIN(pi/7)*SIN(2*pi/7)*SIN(3*pi/7)` is `SQRT(7)/8`, though no one of their
    calls is anything simpler. Each such value is an algebraic number, the minimal
    polynomial of the whole says which, and a polynomial of degree one or two is
    solved for its roots. That the value is a root of it is then proved exactly, as
    `_cyclotomic_root` proves it, and of two roots the one of the value's sign is
    kept, where they differ in sign and the value's is known, as `_known_sign` knows
    it.

    A call alone is not asked about: sympy writes the values it has a spelling for
    as it builds the call, and the rest have none of degree two.
    """
    return expression.replace(
        lambda part: isinstance(part, sp.Add | sp.Mul),
        _trigonometric_part,
        simultaneous=False,
    )


def _trigonometric_part(part: sp.Add | sp.Mul) -> sp.Basic:
    """`part` with its trigonometric constant written as a number, gathered out of
    a product the way `_flat_part` gathers its constant."""
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
    """Whether a constant of trigonometric values of rational multiples of pi is proved
    a root of the polynomial.

    With `z` a primitive `M`-th root of unity, `#e^(#i*pi*p/q)` is a power of `z`
    wherever `M` is a multiple of `2*q`, and so is `#i` where it is a multiple of 4.
    Every sine, cosine and tangent of the constant is then a rational function of `z`
    with rational coefficients, and so is the polynomial at the constant; it is zero
    exactly where its numerator is a multiple of the cyclotomic polynomial of order
    `M`, the minimal polynomial of `z`. A constant with anything else in it, roots of
    rationals among them, is not proved anything.
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
    """The sign of a constant built of rationals, roots of positive rationals and
    trigonometric values of rational multiples of pi, where its structure says it:
    each value's sign is read off where its angle falls in the period, and a product
    or a sum of terms of one sign has the sign they give it. None where it does not
    say."""
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
    """The sign of `r + c*t`, `r` and `c` rational and `t` a trigonometric value of a
    rational multiple of pi, where the bounds of `t` decide it: a sine or cosine is
    inside -1 to 1 and a secant or cosecant outside it, sympy having written the
    values on the bounds as the numbers they are."""
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
    root of a positive rational, decided by comparing squares, or None for anything
    else."""
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
    """A bound on the degree of the field `constant` lies in, or None where it is
    not rational arithmetic over trigonometric values of rational multiples of pi
    and roots of positive rationals."""
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
        if arc is None:
            continue
        arcs[(type(arc), coefficient, arc.args[0])] = index
        # `ATAN(1/u)` is `ACOT(u)` for every real `u` but zero, where the first is
        # not defined, so it pairs with `ATAN(u)` the way `ACOT(u)` does.
        if isinstance(arc, sp.atan) and not arc.args[0].is_number:
            arcs.setdefault((sp.acot, coefficient, 1 / arc.args[0]), index)
    for (func, coefficient, argument), index in arcs.items():
        other = arcs.get((_COMPLEMENT[func], coefficient, argument))
        if other is None or other == index:
            continue
        pair = {type(_arc(expression.args[i])[0]) for i in (index, other)}
        if pair == {sp.atan}:
            # The pair is `ATAN(u) + ATAN(1/u)`, whose turn either argument tells:
            # the shorter is the one to write it about.
            func, argument = sp.atan, min(argument, 1 / argument, key=sp.count_ops)
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
        if not _real_where_defined(argument):
            return None
        if argument.is_nonnegative:
            return sp.pi / 2
        if argument.is_negative:
            return -sp.pi / 2
        return sp.pi * sp.sign(argument) / 2
    return sp.pi / 2


#: How large a sum of arctangents of numbers is combined: the terms it may hold, the
#: whole multiple each may carry, and how many bits the Gaussian integers multiplied
#: for it may grow to.
_ARCTANGENTS = 8
_ARCTANGENT_MULTIPLE = 12
_ARCTANGENT_BITS = 4096


def _arc_tangent_sums(expression: sp.Basic) -> sp.Basic:
    """Arctangents of rational numbers added up, wherever they add to something
    simpler."""
    return expression.replace(
        lambda e: isinstance(e, sp.Add) and e.has(sp.atan),
        _arc_tangent_sum,
        simultaneous=False,
    )


def _arc_tangent_sum(expression: sp.Add) -> sp.Basic:
    """This sum with its arctangents of rationals combined.

    `ATAN(p/q)` with `q > 0` is the argument of the Gaussian integer `q + p*#i`, so a
    whole combination of them is the argument of a product of such integers, each
    raised to its multiple, up to a whole number of turns. The argument of a
    Gaussian integer is a rational multiple of pi exactly when the integer lies on
    an axis or a diagonal, and then the combination is that multiple: which is
    Machin's `4*ATAN(1/5) - ATAN(1/239) = pi/4`, and `ATAN(2) + ATAN(3) = 3*pi/4`.
    The whole turns are counted as the product is built, each factor turning it by
    less than a right angle.

    Where no part of the sum lies on such a line, its arctangents are combined into
    one only where they are a plain sum and the one argument is no larger than
    those it replaces: `ATAN(1/3) + ATAN(1/7)` is `ATAN(1/2)`, while
    `ATAN(1/2) + ATAN(1/4)` stays rather than become `ATAN(6/7)`.

    The terms are grouped by what multiplies their whole coefficient, so the rule
    holds in degree measure too, where each arc comes multiplied by `180/pi`, and
    for `y*ATAN(1/2) + y*ATAN(1/3)`.
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
        # The coefficients over their common measure, which leaves the whole
        # multiples the Gaussian integers are raised to.
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
    """The terms `k*ATAN(r)` add up to, each part on an axis or diagonal combined."""
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
    """The Gaussian integer whose argument is `sum(k*ATAN(r))` up to whole turns, and
    the number of those turns: the sum is the principal argument of the integer, in
    `(-pi, pi]`, plus that many times `2*pi`.

    A negative multiple takes the conjugate, which has the opposite argument and is
    the reciprocal over a positive real. Each factor turns the product by less than a
    right angle, so the product has gone once round where its principal argument
    moved against the factor's turn, which is read off the signs of integers alone.
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
    """Whether the principal argument of the first nonzero Gaussian integer is below
    that of the second: by the half plane each is in, `(-pi, 0]` below `(0, pi]`, and
    within one half plane by the sign of their cross product."""

    def upper(z: tuple[int, int]) -> bool:
        return z[1] > 0 or (z[1] == 0 and z[0] < 0)

    if upper(first) != upper(second):
        return upper(second)
    return first[0] * second[1] - first[1] * second[0] > 0


#: The principal argument of a Gaussian integer on an axis or a diagonal, in eighths
#: of a turn, by the signs of its real and imaginary parts.
_DIRECTIONS = {
    (1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3,
    (-1, 0): 4, (-1, -1): -3, (0, -1): -2, (1, -1): -1,
}


def _sign(number: int) -> int:
    return (number > 0) - (number < 0)


def _straight_angle(arcs: list[tuple[int, sp.Rational]]) -> sp.Basic | None:
    """The rational multiple of pi these arcs add to, or None where it is none."""
    real, imaginary, turns = _gaussian(arcs)
    if real != 0 and imaginary != 0 and abs(real) != abs(imaginary):
        return None
    direction = _DIRECTIONS[_sign(real), _sign(imaginary)]
    return sp.pi * sp.Rational(direction + 8 * turns, 4)


def _single_arc(arcs: list[tuple[int, sp.Rational]]) -> sp.Basic | None:
    """These arcs as one arctangent, if it is that and no larger than they are: where
    their sum is the argument of a Gaussian integer right of the imaginary axis, and
    so inside the range of the arctangent."""
    real, imaginary, turns = _gaussian(arcs)
    if real <= 0 or turns != 0:
        return None
    argument = sp.Rational(imaginary, real)
    if max(abs(argument.p), argument.q) > max(max(abs(r.p), r.q) for _, r in arcs):
        return None
    return sp.atan(argument)


def _real_where_defined(argument: sp.Basic) -> bool:
    """Whether `argument` is real wherever it is defined.

    Sympy will not say so of `1/x` or `x/y`, which are infinite where a denominator
    vanishes, but neither arc of such an argument is defined there either, so a
    rational function of real variables over real numbers counts.
    """
    if argument.is_extended_real:
        return True
    return (
        bool(argument.is_rational_function())
        and all(symbol.is_real for symbol in argument.free_symbols)
        and not argument.has(sp.I)
    )


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


#: The arcs a sine or cosine of a multiple of one is written out about, and the
#: largest multiple. `TAN` is written out about the arctangent alone, the one arc
#: whose double tangent is a rational function.
_ARCS = (sp.asin, sp.acos, sp.atan)
_ARC_MULTIPLE = 4


def _arc_multiples(expression: sp.Basic) -> sp.Basic:
    """A sine, cosine or tangent of a small multiple of an arc, as the algebraic
    function it is.

    Sympy writes `SIN(ACOS(x))` as `SQRT(1 - x^2)` as it is built, and the
    multiple-angle formulas carry that to `SIN(2*ACOS(x))`, which is
    `2*x*SQRT(1 - x^2)`. Over the whole plane, as the single angle is: the
    formulas are identities, and the calls of the arc are the principal values
    they are on either side.
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


#: How far `_multiple_angles` writes calls out over their single angle: the largest
#: degree over it, a call's multiple times the power it is raised to, and how many
#: operations the expression may hold. A sine of twelve times an angle is a
#: polynomial of degree twelve, which is as far as any identity a worksheet states
#: will go.
_MULTIPLE_ANGLE = 12
_MULTIPLE_OPERATIONS = 60
_DEGREE = 4 * _MULTIPLE_ANGLE


def _multiple_angles(expression: sp.Basic) -> sp.Basic:
    """Sines and cosines of whole multiples of one angle reduced together.

    `trigsimp` rewrites a sum from the bottom up, each term in its own way, and a
    sum whose terms cancel only together is left half rewritten:
    `SIN(x)^6 + COS(x)^6 + 3*SIN(x)^2*COS(x)^2` is 1, and `trigsimp` turns its last
    term into `3/8 - 3*COS(4*x)/8` and stops there. Neither does it cancel
    `SIN(3*x)/SIN(x)`, whose quotient is a polynomial only once both calls are
    written over the one angle.

    Written over the single angle and cancelled, a polynomial in its sine and
    cosine has two spellings that are unique: the sum of sines and cosines of its
    multiples, which is how `trigsimp` writes `SIN(x)^2*COS(x)^2`, and the
    polynomial in one of the two functions alone, the other's square being one
    less the first's. The shortest of these is what is offered, the sum of
    multiples where they tie, so `SIN(3*x)/SIN(x)` is `2*COS(2*x) + 1`.
    """
    angle = _whole_angle(expression)
    if angle is None:
        return expression
    single = sp.cancel(sp.expand_trig(expression))
    sine, cosine = sp.sin(angle), sp.cos(angle)
    candidates = []
    # A quotient left over may cancel further once a square of one function is
    # written over the other: `SIN(x)^3/(COS(x)^2 - 1)` is `-SIN(x)`.
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
    """The angle every sine and cosine in `expression` is a whole multiple of.

    None unless at least two different multiples of it are there, which is what
    makes writing them over one angle a step `trigsimp` does not take; and none
    where another trigonometric call stands beside them, or the expression is too
    large to expand.
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
    # The degree over the single angle: a power of a call multiplies its multiple.
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
    """A polynomial in sines and cosines as a sum of the calls of multiple angles.

    Each round of `TR8` writes the products it finds as sums, which may leave
    squares of the calls it wrote, so the rounds go on until nothing changes; the
    degree bounds how many there can be.
    """
    for _ in range(2 * _MULTIPLE_ANGLE):
        written = sp.expand(TR8(polynomial))
        if written == polynomial:
            break
        polynomial = written
    return polynomial


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


def _logarithmic_powers(expression: sp.Basic) -> sp.Basic:
    """`b^LN(a)` written as `a^LN(b)` for positive rationals `a < b`.

    Both are `#e^(LN(a)*LN(b))`, so `2^LN(3) - 3^LN(2)` is 0, and nothing that
    compares the two terms can see it until they are written over one base. The
    smaller base is the one kept, which is how the exponential of a product of two
    logarithms is written too: `#e^(LN(2)*LN(3))` is `2^LN(3)`.
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


def _extracted_powers(expression: sp.Basic, even: bool = False) -> sp.Basic:
    """`LN(z^k)` written as `k*LN(z)`, everywhere the rule holds."""
    return expression.replace(
        lambda e: _extracted_power(e, even) is not None,
        lambda e: _extracted_power(e, even),
        simultaneous=False,
    )


def _expanded_powers(expression: sp.Basic) -> sp.Basic:
    """`LN(z^k)` written as `k*LN(z)`, and an even power of a real `z` as
    `k*LN(ABS(z))`."""
    return _extracted_powers(expression, even=True)


def _extracted_power(expression: sp.Basic, even: bool = False) -> sp.Basic | None:
    """`k*LN(z)`, if `expression` is a logarithm of `z^k` that may be taken apart.

    6.2 gives the rule as `k*LN(x) <-> LN(x^k)` for a rational `k` where `x` is
    nonnegative, and for any `x` where `-1 < k <= 1`. With `even`, an even power of
    a real `x` is taken apart as well, being the same power of `ABS(x)`: `LN(x^2)`
    is `2*LN(ABS(x))`. That is a direction Logarithm Expand asks for, and longer
    than what it replaces, so Auto keeps `LN(x^2)`.
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
    if base.is_nonnegative or bool(-1 < exponent <= 1):
        return exponent * sp.log(base)
    if even and exponent.is_Integer and exponent.is_even and _real_where_defined(base):
        return exponent * sp.log(sp.Abs(base))
    return None


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


#: How large a radicand may be for its square-free factorization to be asked for: the
#: terms it has and its total degree. Squares a worksheet writes are a few terms of
#: low degree, and the factorization of anything larger is the cost of a question
#: nobody asked.
_PERFECT_TERMS = 24
_PERFECT_DEGREE = 12


def _perfect_powers(expression: sp.Basic) -> sp.Basic:
    """A root of a polynomial that is a perfect power taken.

    `SQRT(x^2 + 2*x + 1)` is `ABS(x + 1)` for a real `x`, the way `SQRT((x + 1)^2)`
    is, and `(x^2 + 2*x + 1)^(3/2)` is `(x + 1)^2*ABS(x + 1)`. The radicand is a
    positive rational times a power of a polynomial, and the root of that power is
    sympy's, which knows what the variables are declared to be: there are no bars
    where the base is provably nonnegative, `SQRT(x^4 + 2*x^2 + 1)` being `x^2 + 1`, and a
    complex variable keeps its root, there being no real absolute value to write.
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


def _parts_of_powers(expression: sp.Basic) -> sp.Basic:
    """`RE(a^z)` and `IM(a^z)` worked out, for a positive `a` and a `z` not known real.

    Sympy splits an exponential into its parts, `IM(#e^(#i*y))` being `SIN(y)`, and
    leaves a power standing, so that `IM(2^(x + #i*y))` is `2^x*IM(2^(#i*y))`. For a
    positive base the power is the exponential `#e^(z*LN(a))`, and is written so
    under `RE` and `IM`, which then split it: `2^x*SIN(y*LN(2))`. The exponential is
    written back as a power by the spelling rules, where the part it ends in is
    real.
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
    """`PHASE(#i*u)` as `pi*SIGN(u)/2`, for a real `u`.

    An imaginary number points up or down the imaginary axis as its coefficient is
    signed. At zero, where `PHASE` has no value, the answer is 0.
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
    """An arcsine or arccosine of a real number past 1 in size, in rectangular form.

    Their values there are complex, and a complex constant is written as real part
    plus imaginary, the way `ACOSH(1/2)` is `#i*pi/3` and `ATANH(2)` is
    `LN(3)/2 - #i*pi/2`: `ASIN(2)` is `pi/2 - #i*LN(SQRT(3) + 2)`. The principal
    values are those of `-#i*LN(#i*z + SQRT(1 - z^2))` and `pi/2 - ASIN(z)`.
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
    """Every numeric power with a complex exponent written over its principal value.

    `b^z` is `#e^(z*LN(b))` with the principal logarithm, so `#i^#i` is `#e^(-pi/2)`,
    `(-1)^#i` is `#e^(-pi)` and `(2*#i)^#i` is `#e^(#i*LN(2) - pi/2)`. A positive base
    keeps its power, which is the shorter way to write the same exponential: `2^#i`
    stands. An exponential whose imaginary part is a rational multiple of pi is then
    real part plus imaginary where the cosine and the sine are plainly written, as
    `_rectangular` has it for roots of -1: `#e^(#i*pi/3)` is `1/2 + SQRT(3)*#i/2`, and
    `#e^(#i*pi/5)` stands. A cosine or a sine of a complex number is taken apart the
    same way, `COS(pi/3 + #i)` being `COS(pi/3)*COSH(1) - #i*SIN(pi/3)*SINH(1)`.

    Bottom up, so that what holds such a constant is built again around its value:
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
    """`#e^(a + b*#i)` as `#e^a*(COS(b) + #i*SIN(b))`, where those are plainly written."""
    if not isinstance(power, sp.exp):
        return power
    real, imaginary = power.args[0].as_real_imag()
    angle = _plain_angle(imaginary)
    if angle is None:
        return power
    cosine, sine = angle
    return sp.exp(real) * (cosine + sp.I * sine)


def _plain_angle(angle: sp.Basic) -> tuple[sp.Basic, sp.Basic] | None:
    """The cosine and the sine of a rational multiple of pi, where both are plainly
    written, and None otherwise."""
    if not (angle / sp.pi).is_Rational:
        return None
    cosine, sine = sp.cos(angle), sp.sin(angle)
    if not _plainly_written(cosine + sine):
        return None
    return cosine, sine


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

    `kept` is how an answer is taken: `_gated` keeps one that pays, `forced` keeps
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
        return forced(expression, _distributed)
    # Ahead of the definitions, which write the hyperbolic parts it leaves over
    # exponentials.
    expression = forced(expression, _complex_constants)
    expression = forced(expression, _symmetries)
    expression = _by_definition(expression)
    expression = _applied(expression, _IDENTITIES, context)
    for setting, directions in _DIRECTED.items():
        rules = directions.get(getattr(context, setting), ())
        expression = _applied(expression, rules, context)
    return expression


#: The heads that stand in an answer as integrals, sums and products not done.
_STANDING = (sp.Integral, sp.Sum, sp.Product)


def _rewritten_beside_standing(expression: sp.Basic, context: Context) -> sp.Basic:
    """`_rewritten`, with the terms of a sum that are integrals, sums or products left
    standing rewritten apart from the rest.

    A standing integral is a term of its own, as in `INT(x/SIN(a*x), x)` plus an
    antiderivative. Rewritten together, the rest of the sum is put over a common
    denominator wherever that is shorter, and the integral is taken under it,
    multiplied out by the denominator and printed twice.
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
#: `pays` says the answer is simpler; forced ones are facts about the kernels they
#: relate, where gating would keep the longer side for being no longer.
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
    _Rule("conjugate pairs", _conjugate_pairs, forced,
          "CONJ(z)*z is ABS(z)^2"),
    _Rule("combinatorial", _combinatorial, _gated,
          "a ratio of factorials is a binomial or a polynomial when it is shorter"),
    # The arcs and angles are ours rather than sympy's, which collects neither these
    # arcs nor their tangents, writes none of them about a shorter argument, and has
    # no rule that spans two angles at once.
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
            _Rule("logcombine", _nonnegatively(sp.logcombine), forced,
                  "logarithms added into one"),
        ),
        Direction.EXPAND: (
            _Rule("expand_log", _nonnegatively(sp.expand_log), forced,
                  "a logarithm of a product taken apart"),
            _Rule("expanded powers", _expanded_powers, forced,
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

#: The spelling conventions, applied after the normal form. None is an economy and
#: none is gated: several lengthen what they replace, and are still the only way
#: Derive writes the answer.
_SPELLINGS = (
    # After the branch, which is what settles how many values a root has: with Real
    # asked for, `(-8)^(1/3)` is `-2` and there is no rectangle to write.
    _Rule("rectangular", _rectangular, forced,
          "a numeric complex power is real part plus imaginary part"),
    _Rule("tangent squares", _tangent_squares, forced,
          "1/COS(u)^(2*k) is (TAN(u)^2 + 1)^k", yields_to="trigpower"),
    _Rule("extracted powers", _extracted_powers, forced,
          "LN(b^k) is k*LN(b) where 6.2 allows it", yields_to="logarithm"),
    _Rule("collected powers", _collected_powers, forced,
          "powers of one base are one power: n*x^n/x is n*x^(n - 1)",
          yields_to="exponential"),
)
