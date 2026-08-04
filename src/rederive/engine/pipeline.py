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

1. Substitute labels, assignments and function definitions, on the tree.
2. Convert to sympy. Sympy's automatic evaluation already does most of what the
   manual lists as Simplify's basic work: combining numbers, collecting like
   factors and terms, the identities for zero and one, distributing integer
   powers over products.
3. Evaluate the calculus heads innermost first. One that will not evaluate
   survives to the output as itself, which is how `INT(#e^(x^2), x)` comes back
   an integral rather than an error.
4. Offer each remaining rewrite in turn and keep it only if it pays: fewer
   operations, or one variable fewer. That gate is what makes
   `(x^2 + 2*x*y + y^2)/(x^2 - y^2)` come back a ratio rather than a
   polynomial. A mode set to Collect or Expand is an instruction rather than an
   option, so those rewrites are applied whether or not they shorten anything.
5. Write every sum in the normal form `normal_form` is about: a rational
   function of the most main variable that sum holds. This one is not gated on
   anything - it is what a normal form means - and it is where `(x + 1)^9 + y`
   becomes a ninth-degree polynomial while `(y + 1)^9 + x` does not.
6. Approximate, if the precision mode asks for it.

The sequence runs again while a user function is still standing as a call,
because one pass unfolds a definition once and a recursion needs the pass
between two unfoldings: it is where the test of the `IF` is decided and where
the arithmetic on the counted-down argument is done. A size bound ends the
passes for a recursion that never arrives.

Every rewrite is offered inside a `try`, and a rewrite that raises is a rewrite
that was not worth having: the previous form stands. That is what makes the
command total. Nothing sympy does can turn a valid entry into an error.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from math import gcd

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.logic.boolalg import Boolean
from sympy.simplify.fu import TR2, TR5, TR6, TR7, TR8, TR11

from rederive.engine.approximation import GUARD, simplest
from rederive.engine.boundary import DEFAULT_AMOUNT, Amount, Result
from rederive.engine.context import (
    Branch,
    Context,
    Direction,
    Precision,
    TrigPower,
)
from rederive.engine.expanding import EXPAND_AMOUNTS, expanded_expression
from rederive.engine.factoring import amount_named, factored_expression
from rederive.engine.from_sympy import from_sympy
from rederive.engine.intervals import decided, settled
from rederive.engine.normal import normal_form
from rederive.engine.solving import oriented, solutions
from rederive.engine.substitute import named_as_declared, substitute
from rederive.engine.to_sympy import (
    COMMAND_HEADS,
    Antidifference,
    Antiquotient,
    Approx,
    Assign,
    annuity,
    Declare,
    FunDef,
    InertVector,
    Logical,
    PlusMinus,
    Taylor,
    as_condition,
    authored_conditionals,
    is_conditional,
    outsized,
    reread,
    to_sympy,
)
from rederive.model.expr import Node
from rederive.syntax.state import ParseState

__all__ = ["approx", "approximated", "simplified", "simplify"]

Rewrite = Callable[[sp.Basic], sp.Basic]

#: The heads that stand for a computation nobody has asked for yet: the
#: calculus ones, and `APPROX`, which waits for the same reason and in the same
#: place. It has to round the value an integral came out as rather than the
#: integral, and evaluating innermost first is exactly what gives it that.
_CALCULUS = (
    sp.Derivative,
    sp.Integral,
    sp.Sum,
    sp.Product,
    sp.Limit,
    Antidifference,
    Antiquotient,
    Taylor,
    Approx,
)

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

    A pass that leaves a user function still called is run again, because that
    is what makes a recursive definition arrive anywhere. `substitute` writes a
    body in once and then declines, so `FACT(5)` comes out of one pass as
    `5*FACT(4)`; only a whole further pass can decide the next `IF` and do the
    next multiplication, since the pre-pass has no evaluator and `n - 1` is a
    tree to it rather than a number. So the recursion lives here, around the
    pass, where deciding the test and doing the arithmetic have already
    happened.

    `decide` is what makes a relation that holds come back as `true`, and soLve
    is the one caller that turns it off. Its pipeline runs to prepare an
    equation for solving, not to answer it, and `x = x` decided to `true` on
    the way in is an equation with nothing left to solve for - where the answer
    it has to reach is `x = @1`, every value there is.
    """
    context = context or Context()
    result = _once(node, context, decide)
    for _ in range(_UNFOLDS - 1):
        if not _unfolds_further(result, context):
            break
        try:
            following = _once(from_sympy(result, context).exact, context, decide)
        except Exception:
            break
        if following == result:
            # The call that is left is one nothing can write in: a definition
            # line, or a name applied to arguments it has no body for.
            break
        result = following
    return result


def _once(node: Node, context: Context, decide: bool) -> sp.Basic:
    """One pass: substitute, convert, and transform by shape."""
    expression = to_sympy(substitute(node, context), context)
    try:
        return _transform(expression, context, decide)
    except Exception:
        # Whatever went wrong, the faithful translation of the input is still
        # a correct answer to "simplify this".
        return expression


def _unfolds_further(result: sp.Basic, context: Context) -> bool:
    """Whether another pass over `result` is worth running.

    A pass is worth running while a function the session defined is still
    standing as a call, and while what has been worked out so far is small
    enough to keep working out. The size bound is `ITERATE`'s, and for the same
    reason: a recursion that does not come round runs away instead, and it runs
    away faster than any count of passes can catch.

    What happens when a bound trips is that the unfinished expression stands as
    the answer. `FACT(-1)` comes back as a product of everything counted down
    so far times a `FACT` of a number heading away from zero - which is a true
    rewriting of what was asked, and one that shows why it will not finish.
    Derive itself exhausts memory there and answers nothing at all; an engine
    that is total everywhere else has no reason to have one hole in it.
    """
    if not context.functions:
        return False
    try:
        calls = result.atoms(AppliedUndef)
    except Exception:
        return False
    if not any(type(call).__name__ in context.functions for call in calls):
        return False
    return not outsized(result)


#: How many passes a recursion is given. Each is a whole Simplify, so the count
#: is a real cost - but a small one buys nothing: the manual's own
#: `RAISE(b, 256)` takes two hundred and fifty-seven of them, and Derive's
#: recursions are written to count down one at a time.
_UNFOLDS = 300


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
    something definite means saying it. The judgement is three-valued and is
    `_decide`'s, the same reading the test of an `IF` gets, so a relation that
    settles nothing - `x = 2`, or `ABS(x) < 1` under no declaration - comes back
    exactly as it was authored rather than as a guess.

    An ordering is asked only of sides that are real, because that is the only
    domain it means anything over: `x < x + 1` holds for a real `x` and says
    nothing about a complex one, where sympy's own comparison declines. An
    equation is asked whatever the domain, `=` being a question every value
    answers.

    `decide` off is soLve's pipeline, which wants the sides simplified and the
    relation left standing to be solved.
    """
    left = _transform(expression.lhs, context, decide)
    right = _transform(expression.rhs, context, decide)
    try:
        relation = expression.func(left, right, evaluate=False)
    except Exception:
        relation = expression.func(left, right)
    if not (decide and isinstance(relation, Relational) and _comparable(relation)):
        return relation
    held = _decide(relation, context)
    if held is None:
        return relation
    return sp.true if held else sp.false


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
    """An ordinary expression, through the whole sequence."""
    expression, frozen = _conditionals(expression, context)
    if not frozen and isinstance(expression, sp.MatrixBase):
        # A conditional whose branches are vectors is a vector once it has been
        # taken, and a vector is simplified element by element - the same
        # answer whether the matrix was written or arrived at.
        return _transform(expression, context, decide)
    expression, held = _held(expression)
    tried: set[sp.Basic] = set()
    expression = _calculus(expression, context, tried)
    expression = _numeric(expression)
    # Before the rewrites rather than after, so that what the declared
    # intervals settle is settled for them too: `ABS(x) + ABS(x - 1)` over
    # `(0, 1)` is a sum that cancels, and only once both bars are off.
    expression = settled(expression, context)
    expression = _rewritten(expression, context)
    expression = _factorials(expression)
    # The rewrites can leave a head where none stood: `cancel` splits a single
    # integrand into a sum, and each part is an integral of its own. Those have
    # been offered to nothing yet, and an answer still holding `INT(x, x)` is
    # not a simplified one. Asked before walking, since most expressions have
    # no such head at all. Not cheap where it does run - a rewrite can reshape
    # a derivative without changing what it means, and expanding it again is
    # most of what Derive's `F_1A` costs - but the alternative is an answer
    # that reaches its form only on a second Simplify.
    if expression.has(*_CALCULUS):
        expression = _calculus(expression, context, tried)
    # The normal form, and the one step that is not gated on getting shorter:
    # a sum is written about its primary variable whether or not that pays.
    # Before the placeholders go back, so that a frozen `IF`'s branches - which
    # nothing else in the pipeline touches either - keep the form they were
    # written in; and before `_commanded`, so that a factorization a `FACTOR`
    # head produces is not multiplied straight back out.
    #
    # `over_numbers` off: nothing here divided, so a number several terms carry
    # underneath is their own coefficient and not a denominator the sum is
    # written over. `x^2/6 + x/2` is what was authored and what comes back.
    expression = normal_form(expression, context.order, over_numbers=False)
    expression = _thawed(expression, held)
    if frozen:
        # A conditional comes back where its placeholder was applied, which
        # need not be where it was frozen: an index written in is a test that
        # may be decidable now, so the conditionals are offered once more.
        thawed = _reconditioned(_thawed(expression, frozen), context)
        expression = _resolved(thawed, context, _kept)
    return approximated(_commanded(_canonical(expression), context), context)


def _held(expression: sp.Basic) -> tuple[sp.Basic, dict[sp.Basic, sp.Basic]]:
    """Every `Subs` set aside, with what to put back in its place.

    A `Subs` is sympy's way of writing "this derivative, at this point", and
    the derivative in one is unevaluated on purpose: the slope of `F` at `y` is
    not the slope of a constant, though it looks like one once the variable it
    is taken over has been bound. Evaluating inside would answer zero, so
    nothing inside is touched at all.
    """
    if not expression.has(sp.Subs):
        return expression, {}
    held: dict[sp.Basic, sp.Basic] = {}

    def hold(head: sp.Basic) -> sp.Basic:
        placeholder = _placeholder(head, len(held) + _SUBSTITUTIONS)
        held[placeholder] = head
        return placeholder

    try:
        return expression.replace(_is_substitution, hold, simultaneous=False), held
    except Exception:
        return expression, {}


#: Where the placeholders for held substitutions start, so that they cannot
#: collide with the ones a frozen conditional uses.
_SUBSTITUTIONS = 1000


def _is_substitution(expression: sp.Basic) -> bool:
    return isinstance(expression, sp.Subs)


def _canonical(expression: sp.Basic) -> sp.Basic:
    """Rebuild every sum and product, so that what is written is what is meant.

    Two things need this. `cancel` and its neighbours keep a numeric
    coefficient outside a sum - `(n^2 + n)/2` rather than `n^2/2 + n/2` - by
    building a product with evaluation switched off; it is the nicer thing to
    look at and the notation cannot hold it, since reading `(n^2 + n)/2` back
    gives the distributed form. And a frozen `IF` put back by `xreplace` lands
    wherever its placeholder stood, which is not where sympy sorts the `IF`
    itself, so `(c + IF(...))*#e^(-p*x)` and `#e^(-p*x)*(c + IF(...))` would be
    the same answer printed two ways depending on which pass produced it.

    Rebuilding sorts both back into the order sympy would have put them in, so
    the text and the expression it reads back as are the same thing. A product
    is rebuilt by `_coefficient_in`, which finishes the first of the two where
    sympy's own construction leaves it half done.
    """
    try:
        return expression.replace(_is_run, _rebuilt, simultaneous=False)
    except Exception:
        return expression


def _is_run(expression: sp.Basic) -> bool:
    return bool(expression.is_Add or expression.is_Mul)


def _rebuilt(run: sp.Basic) -> sp.Basic:
    try:
        return _coefficient_in(run) if run.is_Mul else run.func(*run.args)
    except Exception:
        return run


def _coefficient_in(run: sp.Basic) -> sp.Basic:
    """A product with its fractional coefficient written into the sum beside it.

    Sympy writes one in where the sum is all there is beside the number -
    `(n^2 + n)/2` is `n^2/2 + n/2` the moment it is built - and stops the
    moment anything else stands in the product. Derive does not stop. Its
    answer to the manual's difference equation is `(m^2/2 + m/2 + 1)*(m - 1)!`,
    where the half is written in although a factorial stands beside the sum,
    because a polynomial's coefficients are rational there, term by term, the
    way `SUM(n^2, n)` is `n^3/3 - n^2/2 + n/6`.

    Not where the number would land under every term. Then it is a denominator
    the whole sum shares, and a shared denominator is written outside the sum
    and not into it - `_shared` in `normal_form`, read the other way round.
    `x*(x - 1)/2` is that one: written in, it would be `x*(x/2 - 1/2)`, where
    the two halves are the one half the product already carried.
    """
    rebuilt = run.func(*run.args)
    if not rebuilt.is_Mul:
        return rebuilt
    coefficient, rest = rebuilt.as_coeff_Mul()
    if not (coefficient.is_Rational and coefficient.q != 1):
        return rebuilt
    factors = sp.Mul.make_args(rest)
    sums = [factor for factor in factors if factor.is_Add]
    if len(factors) < 2 or len(sums) != 1:
        return rebuilt
    terms = [coefficient * term for term in sums[0].args]
    if not any(term.as_coeff_Mul()[0].is_integer for term in terms):
        return rebuilt
    return sp.Mul(*(factor for factor in factors if not factor.is_Add), sp.Add(*terms))


# -- IF, and asking a test whether it holds -----------------------------------


def _conditionals(
    expression: sp.Basic, context: Context
) -> tuple[sp.Basic, dict[sp.Symbol, sp.Basic]]:
    """Resolve every conditional whose test can be decided, innermost first.

    Deciding the test is the whole of it. A true test leaves only the then
    clause to simplify, a false one only the else clause - `?` when there is
    none - and an undecidable test takes the unknown clause if the author wrote
    one.

    A conditional is an authored `IF` or the `Piecewise` it converts to, and
    the two are resolved alike: a `Piecewise` is what sympy answers a
    conditional integral with, and what an `IF` reads back as, so a rule that
    held for only one of them would make an answer depend on which pass it came
    from. Only the four-argument `IF`, whose fourth argument is the value where
    the test cannot be decided, has no `Piecewise` to be.

    With no unknown clause an undecidable `IF` comes back untouched, branches
    and all, which is Derive's behaviour and a deliberate one: simplifying a
    branch that may never be taken can turn a guarded division by zero into an
    error. Such an `IF` is set aside under a placeholder for the rest of the
    pipeline and put back at the end.

    The placeholder is applied to the variables the `IF` mentions, and that is
    not decoration. A bare symbol would hide them, and then `SUM(IF(k > 0, k,
    0), k, 1, n)` would come back `n*IF(k > 0, k, 0)` - sympy summing what it
    had been shown no longer depends on `k`. Standing in as a function of `k`
    keeps the sum unevaluated, which is the true answer.

    It is a function of `k` in earnest: what is remembered is the conditional
    as a `Lambda` over those variables, and `_thawed` puts it back by applying
    that rather than by matching what was frozen. A finite `SUM` is written out
    by applying what it sums to each index in turn, so the placeholder reaches
    the end as `IF0(1) + ... + IF0(100)`, and each of those terms is the
    conditional with an index written in.
    """
    frozen: dict[sp.Basic, sp.Basic] = {}

    def freeze(head: sp.Basic) -> sp.Basic:
        placeholder = _placeholder(head, len(frozen))
        frozen[placeholder] = head
        return placeholder

    return _resolved(expression, context, freeze), frozen


def _kept(head: sp.Basic) -> sp.Basic:
    """An undecidable conditional left standing as itself, frozen for nothing."""
    return head


def _resolved(
    expression: sp.Basic, context: Context, freeze: Callable[[sp.Basic], sp.Basic]
) -> sp.Basic:
    """Every conditional decided that can be, the rest handed to `freeze`."""

    def resolve(head: sp.Basic) -> sp.Basic:
        if isinstance(head, sp.Piecewise):
            settled = _cases(head, context)
            return freeze(settled) if isinstance(settled, sp.Piecewise) else settled
        test, *branches = head.args
        decided = _decide(test, context)
        if not branches:
            # A test and nothing else is that test as a number, one or zero.
            if decided is None:
                return freeze(head)
            return sp.Integer(1) if decided else sp.Integer(0)
        if decided is True:
            return branches[0]
        if decided is False:
            return branches[1] if len(branches) > 1 else sp.nan
        if len(branches) > 2:
            return branches[2]
        return freeze(head)

    try:
        return expression.replace(is_conditional, resolve, simultaneous=False)
    except Exception:
        return expression


def _placeholder(head: sp.Basic, index: int) -> sp.Basic:
    """Something inert that depends on exactly what `head` depends on."""
    variables = sorted(head.free_symbols, key=sp.default_sort_key)
    if not variables:
        return sp.Dummy(f"IF{index}")
    return sp.Function(f"IF{index}", nargs=len(variables))(*variables)


def _thawed(expression: sp.Basic, standing_in: dict[sp.Basic, sp.Basic]) -> sp.Basic:
    """Every placeholder put back, wherever and however it was applied.

    A placeholder that was applied to the variables its head mentions stands
    for that head as a function of them, so putting it back is applying that
    function to the arguments it is found under. Matching the placeholder as it
    was frozen would miss `IF0(1)`, which is what a finite `SUM` leaves behind,
    and the standing-in name would reach the reader.

    One with no variables at all is a name and not a function, and there is
    nothing to apply: it goes back where it stands.

    Repeated because a head can hold a placeholder of its own - a conditional
    frozen inside a conditional - and one pass puts back only the outer one.
    """
    if not standing_in:
        return expression
    applied: dict[type, sp.Lambda] = {}
    named: dict[sp.Basic, sp.Basic] = {}
    for placeholder, head in standing_in.items():
        if isinstance(placeholder, AppliedUndef):
            applied[type(placeholder)] = sp.Lambda(tuple(placeholder.args), head)
        else:
            named[placeholder] = head
    for _ in range(len(standing_in)):
        thawed = _thaw(expression, applied, named)
        if thawed == expression:
            return thawed
        expression = thawed
    return expression


def _thaw(expression: sp.Basic, applied: dict[type, sp.Lambda], named: dict) -> sp.Basic:
    """One pass of putting placeholders back: the applied ones and the named."""
    if named:
        expression = expression.xreplace(named)
    if not applied:
        return expression
    try:
        return expression.replace(
            lambda found: type(found) in applied,
            lambda found: _written_in(applied[type(found)], found.args),
        )
    except Exception:
        return expression


def _written_in(recipe: sp.Lambda, arguments: tuple) -> sp.Basic:
    """The frozen head with these arguments written in for its variables.

    By `subs` rather than by applying the `Lambda`, which writes in by
    `xreplace` and so reaches into a sum whose index happens to be one of those
    variables. `NUMBER.MTH` has one: a conditional that is a function of `i_`
    and holds a sum over `i_` of its own, where a limit rewritten to `(0, 1, k_
    - 1)` is no limit at all.
    """
    written = dict(zip(recipe.variables, arguments, strict=True))
    return recipe.expr.subs(written, simultaneous=True)


def _cases(head: sp.Piecewise, context: Context) -> sp.Basic:
    """A case split with every case its condition decides taken out.

    A case whose condition is decidably false drops out, and where a decidably
    true one is reached the split ends there - which is the then and the else
    clause of an `IF` under another name. What is left is the split from the
    first undecidable case on, or `?` where every case was ruled out, since
    outside all of its conditions a `Piecewise` has no value.
    """
    remaining: list[tuple[sp.Basic, sp.Basic]] = []
    for value, condition in head.args:
        decided = _decide(condition, context)
        if decided is False:
            continue
        remaining.append((value, sp.true if decided else condition))
        if decided:
            break
    if not remaining:
        return sp.nan
    try:
        # One case left over that always holds is no longer a split, and sympy
        # collapses it to its value on the way through.
        return sp.Piecewise(*remaining)
    except Exception:
        return head


def _decide(test: sp.Basic, context: Context) -> bool | None:
    """Whether `test` holds, or None when nothing available settles it.

    A relation reaches here undecided, so it is offered again evaluated - which
    is where the declared domains do their work - and then with its two sides
    brought together by the same rewrites the rest of the pipeline uses. What
    none of those settles is put to the declared intervals, which are the one
    thing a symbol could not carry into the evaluation itself.

    A test that is no relation at all is a comparison with zero written short,
    the same reading the conversion gives one, so that the inert four-argument
    `IF` decides its test by the rule the rest of them do.
    """
    test = as_condition(test)
    for candidate in _truths(test, context):
        if candidate is sp.true:
            return True
        if candidate is sp.false:
            return False
    return decided(test, context)


def _reconditioned(expression: sp.Basic, context: Context) -> sp.Basic:
    """Every test read again over the heads that have since answered.

    A test the conversion could make nothing of is read as a comparison with
    zero, which is how Derive reads a number. `IF(PRIME(n))` is one of those:
    with `n` still a variable, what `PRIME(n)` is has not been asked, so the
    conversion had a call and no truth-value to work from. An index written in
    by a sum asks it, and a truth-value read as `test = 0` is read backwards -
    so a comparison whose subject has turned into a statement becomes that
    statement.

    A subject that answers nothing new is left alone, comparison and all: `x =
    0` is a comparison somebody wrote, and nothing here makes it a claim.
    """

    def restated(found: sp.Basic) -> sp.Basic:
        subject = reread(found.lhs, context)
        if not isinstance(subject, (Boolean, Logical)):
            return found
        return as_condition(subject)

    try:
        return expression.replace(_is_zero_test, restated, simultaneous=False)
    except Exception:
        return expression


def _is_zero_test(expression: sp.Basic) -> bool:
    return (
        isinstance(expression, sp.Equality)
        and expression.rhs == 0
        and expression.lhs.has(AppliedUndef)
    )


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


def _calculus(
    expression: sp.Basic, context: Context, tried: set[sp.Basic] | None = None
) -> sp.Basic:
    """`.doit()` on every calculus head, innermost first, until none is new.

    Evaluating one head can hand back others: sympy answers an integral of a
    sum with a sum of integrals, and those have been offered to nothing yet.
    So the rounds go on while heads keep appearing, and a head that has already
    been tried and stood is not tried again - which is what ends them, and what
    keeps a hard integral from being attempted twice.

    `tried` is that record, and a caller running this more than once over one
    expression passes its own so that the second run inherits it.
    """
    if tried is None:
        tried = set()

    def pending(head: sp.Basic) -> bool:
        return isinstance(head, _CALCULUS) and head not in tried

    for _ in range(_ROUNDS):
        # What this round offers is remembered only once the round is over, so
        # that a head standing in two places is evaluated in both - and so that
        # one sympy hands back inside its own answer, as it does for a sum it
        # can only do under a condition, is not offered again and again.
        offered: set[sp.Basic] = set()

        def evaluate(head: sp.Basic) -> sp.Basic:
            offered.add(head)
            return _evaluate(head, context)

        try:
            rewritten = expression.replace(pending, evaluate, simultaneous=False)
        except Exception:
            return expression
        tried |= offered
        if rewritten == expression:
            return rewritten
        expression = rewritten
    return expression


#: How many times the heads are offered. Two rounds are what the cases in hand
#: need; the third is there so that a rewrite feeding itself cannot spin.
_ROUNDS = 3


def _evaluate(head: sp.Basic, context: Context) -> sp.Basic:
    """One head evaluated, or the head itself if it will not evaluate."""
    if isinstance(head, Approx):
        return _approximation(head, context)
    if isinstance(head, sp.Integral):
        return _integral(head)
    if isinstance(head, (sp.Sum, sp.Product)) and not _searchable(head):
        return head
    try:
        value = head.doit(deep=False)
    except Exception:
        return _undecided(head)
    return head if value is None else value


def _approximation(head: sp.Basic, context: Context) -> sp.Basic:
    """`APPROX(u, n)`: `u` as approximate arithmetic at `n` digits leaves it.

    Exactly what the approX command does to a whole entry, at the digit count
    the call asked for rather than the one the session is set to, and whatever
    precision that is. So the same rules hold inside a call as outside one: a
    number that needs no digits gets none, and `APPROX(2 + 3)` is 5.
    """
    value, digits = head.args
    approximate = context.with_precision(Precision.APPROXIMATE, int(digits))
    return approximated(value, approximate)


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
    return value if across is None else across


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
    neither has an antiderivative in hand to choose among.
    """
    if len(head.limits) != 1 or len(head.limits[0]) != 1 or value.has(sp.Integral):
        return value
    variable, integrand = head.limits[0][0], head.function
    if _refuted(value, integrand, variable):
        value = _by_the_manual_rules(head, integrand, variable) or value
    return _without_constants(_single_case(value, integrand, variable), variable)


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

    Where the unconditional case does not hold the split is doing real work and
    is left alone: `INT(x^2*COS(a*x^3 + b), x)` splits on whether `a` is zero,
    and `SIN(a*x^3 + b)/(3*a)` is the answer for every `a` it is defined for -
    which is every `a` but the one the other case is there for.
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
    return coefficient * sp.Add(*(term for term in terms.args if term.has(variable)))


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
    """
    for limit in head.limits:
        if len(limit) < 3 or all(bound.is_number for bound in limit[1:]):
            continue
        if _scaled_term(head.function, limit[0]):
            return False
    return True


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


def _commanded(expression: sp.Basic, context: Context) -> sp.Basic:
    """Evaluate every `FACTOR`, `EXPAND` and `SOLVE` head, innermost first.

    `FACTOR(u, amount, x, y, ...)`, `EXPAND(u, amount, x, y, ...)` and
    `SOLVE(u, x)` are the author-line spellings of three of the commands, so
    simplifying a line that carries one runs it - which is what the original
    does, and what makes the functions worth having rather than inert heads
    that print back as themselves. All three are matched in one pass so that
    any of them may be written inside another.

    Last of the rewrites on purpose, and after `normal_form` and `_canonical`
    as well, because everything above this one would undo a factorization.
    The gated rewrites ask whether a candidate is shorter than what it
    replaces, and a factorization is routinely longer: `_multiplied_out`
    offered `(x - 3)*(2*x + 1)^2` would happily multiply it back out. The
    normal form does not ask, and would multiply out any factor that is a sum
    in the primary variable. And `_canonical` rebuilds every product it can
    reach, which would multiply a prime decomposition straight back into the
    integer it decomposes.

    Anything applied *to* one of these heads was left inert by the conversion,
    there being nothing yet to apply it to, so what has changed is offered to
    the function tables again: `RHS(SOLVE(x^2 - 5*x + 6 = 0, x))` is a vector
    of roots only once the `SOLVE` inside it has become a vector.
    """
    try:
        commanded = expression.replace(
            _is_command, lambda head: _command(head, context), simultaneous=False
        )
    except Exception:
        return expression
    if commanded == expression:
        return commanded
    try:
        return reread(commanded, context)
    except Exception:
        return commanded


def _is_command(expression: sp.Basic) -> bool:
    return isinstance(expression, AppliedUndef) and type(expression).__name__ in (
        COMMAND_HEADS
    )


def _command(head: sp.Basic, context: Context) -> sp.Basic:
    match type(head).__name__:
        case "SOLVE":
            return _solve_head(head, context)
        case "RATE":
            return _rate_head(head, context)
    return _factor_head(head, context)


def _factor_head(head: sp.Basic, context: Context) -> sp.Basic:
    """One head, read as its target, its amount and its variables.

    The arguments after the first are a word naming the amount, a list of
    variables, or both in that order; the manual allows any of them to be left
    out. A word that names no amount is read as a variable, which is what
    keeps a call nobody can make sense of - a Derive 6 `FACTOR(u, Turing)` -
    an expression rather than an error. `EXPAND(u, Complex)` is such a call:
    Complex is an amount Factor offers and Expand does not.
    """
    expanding = type(head).__name__ == "EXPAND"
    amounts = EXPAND_AMOUNTS if expanding else tuple(Amount)
    target, *rest = head.args
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


def _solve_head(head: sp.Basic, context: Context) -> sp.Basic:
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
    target, *rest = head.args
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


def _rate_head(head: sp.Basic, context: Context) -> sp.Basic:
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
    if not all(argument.is_number for argument in head.args):
        return head
    periods, payment, present, future, when, low, high = head.args
    rate = sp.Symbol(_RATE)
    equation = sp.Eq(annuity(rate, periods, payment, present, future, when), 0)
    answers = solutions(equation, context, (_RATE,), (low, high))
    if len(answers) != 1 or not isinstance(answers[0], Relational):
        return head
    return answers[0].rhs


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
    is. Written out it makes a summand nobody has offered to sympy yet, so the
    search for a closed form runs again - and over a ratio of factorials of the
    index it does not come back: `SUM(COMB(n, k), k, 0, n)` is `2^n` above zero
    and the sum itself elsewhere, and the sum it is left with would be searched
    forever. That is `_searchable`'s point, one head further out.
    """
    written = _AS_FACTORIALS.get(type(expression))
    if written is not None:
        return written(*expression.args)
    if isinstance(expression, _CALCULUS) or not expression.args:
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


# -- the gated rewrites ------------------------------------------------------


def _rewritten(expression: sp.Basic, context: Context) -> sp.Basic:
    """Every rewrite the settings select, each kept only if it pays."""
    if expression.has(sp.MatrixExpr):
        return _forced(expression, _distributed)
    expression = _by_definition(expression)
    expression = _branch(expression, context)
    # After the branch, which is what settles how many values a root has: with
    # Real asked for, `(-8)^(1/3)` is `-2` and there is no rectangle to write.
    expression = _forced(expression, _rectangular)
    expression = _gated(expression, _multiplied_out)
    expression = _gated(expression, _cancelled)
    if expression.has(*_COMBINATORIAL) and sp.count_ops(expression) <= _COMBSIMP:
        expression = _gated(expression, sp.combsimp)
    # Ours rather than sympy's, which collects neither these arcs nor their
    # tangents, writes none of them about a shorter argument, and has no rule
    # that spans two angles at once. All three are offered in every mode: an
    # identity is not a direction to rewrite in. The complementary pair is the
    # one taken whatever it costs, because what a pair of arcs adds up to is a
    # fact about them and not an economy - `ATAN(t) + ACOT(t)` and
    # `pi*SIGN(t)/2` are three operations either way, and gating that would
    # keep the pair and throw the answer away.
    expression = _gated(expression, _simpler_arcs)
    expression = _forced(expression, _complementary_arcs)
    expression = _gated(expression, _half_angles)
    expression = _trigonometry(expression, context)
    expression = _logarithms(expression, context)
    expression = _exponentials(expression, context)
    expression = _gated(expression, _denested)
    return _gated(expression, sp.radsimp)


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
    """`rewrite`, because a mode set away from Auto asked for it."""
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


def _cancelled(expression: sp.Basic) -> sp.Basic:
    """`cancel`, except where it would multiply something out for nothing.

    Cancelling a common factor is what this is for, and it is worth having.
    Rebuilding the denominator is not: `cancel` writes every ratio over one
    expanded denominator, and where the normal form has already put a sum over
    a product of sums that turns the original's own answer into a longer one.
    `1/(9 + x^2 + (y - 3)^2) + 1/(9 + x^2 + (y + 3)^2)` is
    `2*(x^2 + y^2 + 18)/((x^2 + y^2 - 6*y + 18)*(x^2 + y^2 + 6*y + 18))` to
    Derive, and a quartic denominator to `cancel`.

    A denominator of one factor has nothing to lose, which is every sum of
    ratios that has not been put over a common denominator yet.

    Nothing divided is nothing to cancel. Over a polynomial `cancel` can only
    multiply out, which is what `_multiplied_out` has just declined to do, and
    over a high power of a sum it builds every term of an expansion `_pays`
    then throws away: `(v + w + x + y + z)^60` is six hundred thousand of them,
    a minute and half a gigabyte to reach the answer it was written as. Asked by
    `is_polynomial` rather than by the denominator above, since `fraction`
    reads a sum of ratios as having no denominator at all. Derive returns that
    power unchanged and returns it at once, which is the behaviour being kept.

    What arrives here is not always an expression. An undecidable four-argument
    `IF` simplifies to its unknown clause, and `IF(u, true, false, false)` has a
    truth value for one; asking that for a numerator and a denominator is asking
    the wrong question, and sympy answers it by building a `Mul` out of a
    `BooleanFalse`, which it warns is on its way to being an error.
    """
    if not isinstance(expression, sp.Expr):
        return expression
    if expression.is_polynomial():
        return expression
    if len(sp.Mul.make_args(sp.fraction(expression)[1])) > 1:
        return expression
    if _expanded_terms(expression) > _EXPANSION:
        return expression
    return sp.cancel(expression)


def _multiplied_out(expression: sp.Basic) -> sp.Basic:
    """Multiply out the structure, holding a high power of a sum together.

    `x^2 - (x + (y + 1)^50)*(x - (y + 1)^50)` is a difference of squares whose
    answer is `(y + 1)^100`, and seeing that means distributing the product
    without also expanding the fiftieth power - which would produce a hundred
    terms and hide the answer inside them. So each sum that already appears
    raised to a power stands in as a single symbol, the rest is multiplied out
    around it, and the sums come back afterwards. `(x + (a + 1)^10)^2 -
    (a + 1)^20` is the same trick a level up: the square is expanded, the tenth
    power is not, and what is left is `x^2 + 2*x*(a + 1)^10`.

    Below `_FOLDED` the expansion is small enough to be worth having outright,
    one power of one sum at a time. Enough of them multiplied together is not:
    a dozen squared binomials come to half a million terms with no power held
    at all, so what the standing-in leaves is counted before it is built, and
    an expansion past `_EXPANSION` terms is declined the way a folded power is.

    What is multiplied out gets collected back up while the power is still
    standing in, so that `2*(x^2 - y^2)^6 - (x^2 - y^2)^5*(2*x^2 - 3)` comes
    back as the fifth power times what is left of it - and that is why this
    still earns its place beside the normal form, which writes sums and leaves
    products alone. What this hands back is a product, and the normal form
    leaves it exactly as it is.
    """
    stood_in: dict[sp.Basic, sp.Symbol] = {}

    def hide(power: sp.Basic) -> sp.Basic:
        symbol = stood_in.get(power.base)
        if symbol is None:
            symbol = stood_in.setdefault(power.base, sp.Dummy())
        return symbol**power.exp

    hidden = expression.replace(_is_folded_power, hide, simultaneous=False)
    if _expanded_terms(hidden) > _EXPANSION:
        return expression
    expanded = _collected(sp.expand(hidden), set(stood_in.values()))
    return expanded.xreplace({symbol: base for base, symbol in stood_in.items()})


def _collected(expression: sp.Basic, standing_in: set[sp.Symbol]) -> sp.Basic:
    """A sum with a standing-in power common to its terms taken back out.

    `factor_terms` is the collecting, and it is offered here only where a power
    that was stood in for comes out with it. On anything else it factors -
    `x^2 + 2*x` becomes `x*(x + 2)` - and factoring is the Factor command's
    business, not Simplify's.
    """
    if not (standing_in and isinstance(expression, sp.Add)):
        return expression
    candidate = _attempt(expression, sp.factor_terms)
    if not isinstance(candidate, sp.Mul):
        return expression
    if not any(_stands_in(factor, standing_in) for factor in candidate.args):
        return expression
    return candidate


def _stands_in(factor: sp.Basic, standing_in: set[sp.Symbol]) -> bool:
    """Whether `factor` is one of the stood-in sums, or a power of one."""
    base = factor.base if isinstance(factor, sp.Pow) else factor
    return base in standing_in


#: The exponent from which a power of a sum is left folded up.
_FOLDED = 3


def _is_folded_power(expression: sp.Basic) -> bool:
    return (
        isinstance(expression, sp.Pow)
        and isinstance(expression.base, sp.Add)
        # A symbol stands in for the base, so it had better commute like one.
        and bool(expression.base.is_commutative)
        and bool(expression.exp.is_Integer)
        and abs(int(expression.exp)) >= _FOLDED
    )


#: How many terms an expansion may come to before the rewrite that would build
#: it is not offered at all. A rewrite is kept only where it pays, and one that
#: does not pay still costs everything it took to build and to measure - which
#: for a high power of a sum is minutes and gigabytes for an answer thrown
#: away. Measured over every expression in the corpus: the largest expansion
#: any rewrite is offered there is a hundred and twenty terms, kept or not, so
#: this stands two orders of magnitude above anything a worksheet asks for.
_EXPANSION = 20_000


def _expanded_terms(expression: sp.Basic) -> int:
    """How many terms multiplying `expression` out would come to, at most.

    A bound and not a count: terms that would collect are counted apart, and
    over a limit whose purpose is to decline the expansions nobody asked for
    that is the safe direction to be wrong in. Counted no higher than the limit
    it is measured against, so that a power of a power cannot make the estimate
    the expensive thing.
    """
    if isinstance(expression, sp.Add):
        return min(sum(_expanded_terms(term) for term in expression.args), _CEILING)
    if isinstance(expression, sp.Mul):
        total = 1
        for factor in expression.args:
            total = min(total * _expanded_terms(factor), _CEILING)
        return total
    if isinstance(expression, sp.Pow):
        return _power_terms(expression)
    return 1


#: What `_expanded_terms` counts up to, one term past what it is asked about.
_CEILING = _EXPANSION + 1


def _power_terms(power: sp.Pow) -> int:
    """The terms a power of a sum comes to, one where it is not one.

    `(u_1 + ... + u_k)^n` has `C(n + k - 1, k - 1)` terms in it, built here as
    the product it is rather than as a binomial, so that a count already past
    the ceiling can stop where it gets there: the number itself is not wanted,
    only which side of the limit it falls.
    """
    terms = _expanded_terms(power.base)
    if terms <= 1 or not power.exp.is_Integer or power.exp <= 0:
        return 1
    exponent = int(power.exp)
    count = 1.0
    for index in range(1, terms):
        count *= (exponent + index) / index
        if count > _EXPANSION:
            return _CEILING
    return round(count)


def _denested(expression: sp.Basic) -> sp.Basic:
    """Every square root denested on its own.

    One radical at a time rather than the whole expression at once, because
    `sqrtdenest` gives up on a sum of nested roots and gets each of them
    separately: `SQRT(5 + SQRT(24)) + SQRT(5 - SQRT(24))` is `2*SQRT(3)` only
    if the two terms are denested apart.
    """
    return expression.replace(
        lambda e: isinstance(e, sp.Pow) and e.exp is sp.S.Half,
        sp.sqrtdenest,
        simultaneous=False,
    )


# -- what each mode selects --------------------------------------------------


def _trigonometry(expression: sp.Basic, context: Context) -> sp.Basic:
    """The Trigonometry and Trigpower settings.

    Auto means "only where it cancels", which is what `trigsimp` does. Collect
    is the product-to-sum and power-reduction direction, plus the phase-angle
    rule; Expand is the angle-sum direction, and the algebraic values of the
    special angles that come with it.
    """
    match context.trigonometry:
        case Direction.EXPAND:
            expression = _forced(expression, sp.expand_trig)
            expression = _gated(expression, lambda e: e.rewrite(sp.sqrt))
        case Direction.COLLECT:
            expression = _forced(expression, _phase_angles)
            expression = _forced(expression, TR8)
            expression = _forced(expression, TR7)
        case _:
            expression = _gated(expression, sp.trigsimp)
    # Auto adds nothing here: either `trigsimp` above has already taken the
    # powers that cancel, or a direction was asked for and reversing part of it
    # would be perverse.
    match context.trigpower:
        case TrigPower.SINES:
            return _forced(expression, TR6)
        case TrigPower.COSINES:
            return _forced(expression, TR5)
    return _forced(expression, _tangent_squares)


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


def _logarithms(expression: sp.Basic, context: Context) -> sp.Basic:
    """The Logarithm setting, never forced past what the domains allow.

    `logcombine` and `expand_log` both take a `force` argument that assumes
    every argument positive. It is not used, in either direction: `LN(x^2 - x)
    - LN(x)` collects only once `x` has been declared away from the values that
    would make it false.

    Auto is the mixed direction 6.2 describes, the sum rule rightward and the
    power rule leftward: `3*LN(2) + LN(5)` collects to `LN(40)` and `LN(256)`
    comes apart into `8*LN(2)`. The second half is not gated on getting
    shorter, because the shorter form is the one it gives up: a power is taken
    out of a logarithm the way it is taken out from under a radical, where
    `SQRT(12)` is `2*SQRT(3)` and counts no less.
    """
    match context.logarithm:
        case Direction.COLLECT:
            return _forced(expression, sp.logcombine)
        case Direction.EXPAND:
            return _forced(expression, sp.expand_log)
    expression = _gated(expression, sp.logcombine)
    return _forced(expression, _extracted_powers)


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


def _exponentials(expression: sp.Basic, context: Context) -> sp.Basic:
    """The Exponential setting, over the power rules sympy gates for us.

    `(#e^z)^k -> #e^(k*z)` needs an integer `k` or a strip constraint on `z`;
    `powsimp` and `expand_power_base` know that, so the validity gate is
    theirs and the direction is ours.

    One base written twice collects whatever it costs, in every mode. It is not
    an economy but a normal form: `DIF(x^n, x)` is `n*x^n/x` to sympy and
    `n*x^(n - 1)` to the original, and the two are the same length, so a gate
    that counts operations would keep the first.
    """
    expression = _forced(expression, _collected_powers)
    match context.exponential:
        case Direction.COLLECT:
            return _forced(expression, sp.powsimp)
        case Direction.EXPAND:
            expression = _forced(expression, sp.expand_power_exp)
            return _forced(expression, sp.expand_power_base)
    expression = _gated(expression, sp.powsimp)
    expression = _gated(expression, sp.expand_power_exp)
    return _gated(expression, sp.expand_power_base)


def _collected_powers(expression: sp.Basic) -> sp.Basic:
    """Powers of one base in a product added, and nothing else.

    `combine="exp"` is the exponent rule on its own: it leaves two bases under
    one exponent alone, which is the part of `powsimp` that is a direction to
    rewrite in rather than a normal form.
    """
    return sp.powsimp(expression, combine="exp")


def _branch(expression: sp.Basic, context: Context) -> sp.Basic:
    """The Branch setting: which root of many an expression is allowed to mean.

    Principal is sympy's own convention and needs nothing. Real asks for the
    real root of a negative number, so `(-8)^(1/3)` is `-2` rather than the
    principal complex value. Any lifts the domain proof off the power rules,
    which is what lets `SQRT(x^2)` become `x` for an undeclared `x`.
    """
    match context.branch:
        case Branch.REAL:
            return _forced(expression, _real_roots)
        case Branch.ANY:
            expression = _forced(expression, lambda e: sp.powdenest(e, force=True))
            return _forced(expression, lambda e: sp.powsimp(e, force=True))
    return expression


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


# -- precision ---------------------------------------------------------------


def approximated(expression: sp.Basic, context: Context) -> sp.Basic:
    """What the precision mode does to a finished answer.

    The last step of the pipeline, and public because it is the last step of
    every command's pipeline. Factor runs the rest of this file exactly, then
    factors, and only then rounds - so that radical factoring reaches `SQRT(2)`
    first and shows `1.41421` because of this, rather than factoring a number
    that has already been rounded.
    """
    digits = context.precision_digits
    match context.precision:
        case Precision.APPROXIMATE:
            # The irrational parts are approximated, then what they make of
            # each other is exact, and the answer is approximated once more:
            # `10^7·π` is `10^7·355/113` worked out and then rounded, which is
            # `31415929` and not the `31415900` that six digits of the product
            # would leave.
            return _rounded(_approximated(expression, digits), digits)
        case Precision.MIXED:
            return _approximated(expression, digits)
    return expression


def _evalf(expression: sp.Basic, digits: int) -> sp.Basic:
    try:
        return expression.evalf(digits)
    except Exception:
        return expression


def _rounded(expression: sp.Basic, digits: int) -> sp.Basic:
    """Every number in `expression` as the simplest rational at this precision.

    Which is what Derive's approximate numbers are, so this is what makes
    `2*y*3` come out `6*y` in every precision mode - only the numbers that
    actually need digits get them - and what leaves an approximate answer an
    exact value that `Notation := Rational` can write out in full.
    """
    try:
        return expression.replace(
            lambda part: isinstance(part, sp.Rational | sp.Float),
            lambda part: _simplest(part, digits),
            simultaneous=False,
        )
    except Exception:
        return expression


def _simplest(number: sp.Basic, digits: int) -> sp.Rational:
    value = sp.Rational(number)
    return sp.Rational(simplest(Fraction(value.p, value.q), digits))


def _approximated(expression: sp.Basic, digits: int) -> sp.Basic:
    """Approximate the irrational operations, keep the rational ones exact.

    Innermost first, so that a rational subexpression is computed before
    anything near it is rounded: `SQRT(3422357/2313 - 1140443/771)` is exactly
    `2/3` in Mixed mode, where Approximate rounds the two fractions on the way
    in and never reaches it.

    What each irrational is replaced by is a rational, since that is what an
    approximate number is; the arithmetic around it stays exact.
    """
    try:
        return expression.replace(
            _needs_digits,
            lambda part: _approximate(part, digits),
            simultaneous=False,
        )
    except Exception:
        return expression


def _needs_digits(expression: sp.Basic) -> bool:
    """Whether this number has to be worked out before it can be written down.

    Every number does, unless sympy can show that it is rational: a rational
    is already what an approximate number is, and anything else stands for
    digits nothing has computed yet.

    The question is which way round to ask it, and the answer is not the
    obvious one. `is_irrational` has a third answer besides yes and no, and
    that third answer is the common one - sympy leaves a product of surds open,
    since irrationals can multiply to a rational - so asking for a proof of
    irrationality leaves everything unprovable exact, and approximating a
    wholly numeric expression can then answer with radicals still standing in
    it. Asking for a proof of rationality instead sends the unprovable cases
    the other way, and a number can only come out a number.

    Finiteness is asked the same way round and for the same reason. Only what
    is known to be infinite is refused: `LN(0)` has no digits and never will,
    while `SI(2)`, `EI(2)` and `LI(2)` are finite numbers sympy simply does not
    prove finite, and demanding the proof left every one of them written as
    itself. An unevaluated integral or sum is the same case, and approximating
    one numerically is what the original does with what it cannot do exactly.
    Nothing is risked by asking loosely here: `_approximate` takes only what
    comes back a float or a rational, so a value that will not evaluate - `nan`,
    or a quadrature that does not converge - is left standing anyway.
    """
    return (
        expression.is_number
        and expression.is_rational is not True
        and expression.is_finite is not False
        and not isinstance(expression, sp.Float)
    )


def _approximate(part: sp.Basic, digits: int) -> sp.Basic:
    """The approximate number `part` stands for, or `part` where there is none.

    A number whose value is not real - `SQRT(-2)` and everything else that
    evaluates with an `I` in it - has no rational standing in for it, and is
    left as it is rather than failing the approximation of everything around
    it.
    """
    value = _evalf(part, digits + GUARD)
    if not isinstance(value, sp.Float) and not value.is_Rational:
        return part
    return _simplest(value, digits)
