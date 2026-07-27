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

Every rewrite is offered inside a `try`, and a rewrite that raises is a rewrite
that was not worth having: the previous form stands. That is what makes the
command total. Nothing sympy does can turn a valid entry into an error.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean
from sympy.simplify.fu import TR5, TR6, TR7, TR8

from rederive.engine.approximation import GUARD, simplest
from rederive.engine.context import (
    Branch,
    Context,
    Direction,
    Precision,
    TrigPower,
)
from rederive.engine.expanding import EXPAND_AMOUNTS, expanded_expression
from rederive.engine.factoring import (
    DEFAULT_AMOUNT,
    Amount,
    amount_named,
    factored_expression,
)
from rederive.engine.from_sympy import Result, from_sympy
from rederive.engine.normal import normal_form
from rederive.engine.substitute import named_as_declared, substitute
from rederive.engine.to_sympy import (
    Approx,
    Assign,
    Declare,
    FunDef,
    InertVector,
    Logical,
    PlusMinus,
    Taylor,
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
_CALCULUS = (sp.Derivative, sp.Integral, sp.Sum, sp.Product, sp.Limit, Taylor, Approx)

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


def simplify(
    node: Node, context: Context | None = None, state: ParseState | None = None
) -> Result:
    """Derive's Simplify: `node` in its sufficiently simple form.

    Works on any subtree, not only on a whole authored entry, so that the
    session can simplify a highlighted subexpression and splice the answer
    back. `state` is the symbol table the answer is reparsed with; a session
    working in a non-default input or case mode must pass its own.
    """
    context = context or Context()
    node = named_as_declared(node, state)
    return from_sympy(simplified(node, context), context, state)


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


def simplified(node: Node, context: Context | None = None) -> sp.Basic:
    """The simplified expression, before it is written back out.

    The sympy-level entry point. `simplify` is this plus `from_sympy`; a later
    command that wants to keep computing has no reason to print and reparse in
    between.
    """
    context = context or Context()
    expression = to_sympy(substitute(node, context), context)
    try:
        return _transform(expression, context)
    except Exception:
        # Whatever went wrong, the faithful translation of the input is still
        # a correct answer to "simplify this".
        return expression


# -- by shape ----------------------------------------------------------------


def _transform(expression: sp.Basic, context: Context) -> sp.Basic:
    """Simplify one expression according to what kind of thing it is."""
    if isinstance(expression, Relational):
        return _relation(expression, context)
    if isinstance(expression, Declare):
        return expression
    if isinstance(expression, (Assign, FunDef)):
        return _definition(expression, context)
    if isinstance(expression, sp.MatrixBase):
        return expression.applyfunc(lambda element: _transform(element, context))
    if isinstance(expression, (InertVector, Logical)):
        return expression.func(*(_transform(a, context) for a in expression.args))
    if isinstance(expression, Boolean):
        return _boolean(expression, context)
    return _expression(expression, context)


def _relation(expression: Relational, context: Context) -> sp.Basic:
    """The two sides, simplified apart and put back together undecided.

    `2 = 2` stays `2 = 2`: whether a relation holds is a question for soLve, or
    for the test of an `IF`, and never something Simplify answers on its own.
    """
    left = _transform(expression.lhs, context)
    right = _transform(expression.rhs, context)
    try:
        return expression.func(left, right, evaluate=False)
    except Exception:
        return expression.func(left, right)


def _definition(expression: sp.Basic, context: Context) -> sp.Basic:
    """An assignment or a function definition: its value, and its own shape.

    The first two arguments are the name being defined and the operator that
    defines it, which is exactly what must survive untouched so that the line
    still reads as a definition.
    """
    head, operator, *value = expression.args
    return expression.func(
        head, operator, *(_transform(part, context) for part in value)
    )


def _boolean(expression: Boolean, context: Context) -> sp.Basic:
    """Relations joined by boolean operators, solved where they can be.

    `6 >= -2*x AND 3*x /= -9` is a statement about one variable, and its
    simplest form is the range it describes. Where that does not work - more
    than one variable, an operand that is no relation - the operands are
    simplified and the shape is kept.
    """
    solved = _reduced(expression)
    if solved is not None:
        return solved
    try:
        return expression.func(*(_transform(a, context) for a in expression.args))
    except Exception:
        return expression


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
    return _oriented(solved)


def _oriented(solved: sp.Basic) -> sp.Basic:
    """A solved relation with the variable on the left.

    `reduce_inequalities` answers `-3 < x`; the answer to a question about `x`
    is written `x > -3`.
    """
    if isinstance(solved, sp.And | sp.Or):
        return solved.func(*(_oriented(part) for part in solved.args))
    if isinstance(solved, Relational) and solved.lhs.is_number:
        return solved.reversed
    return solved


# -- the pipeline proper -----------------------------------------------------


def _expression(expression: sp.Basic, context: Context) -> sp.Basic:
    """An ordinary expression, through the whole sequence."""
    expression, frozen = _conditionals(expression, context)
    if not frozen and isinstance(expression, sp.MatrixBase):
        # A conditional whose branches are vectors is a vector once it has been
        # taken, and a vector is simplified element by element - the same
        # answer whether the matrix was written or arrived at.
        return _transform(expression, context)
    expression, held = _held(expression)
    tried: set[sp.Basic] = set()
    expression = _calculus(expression, context, tried)
    expression = _numeric(expression)
    expression = _rewritten(expression, context)
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
    expression = normal_form(expression, context.order)
    for standing_in in (held, frozen):
        if standing_in:
            expression = expression.xreplace(standing_in)
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
    the text and the expression it reads back as are the same thing.
    """
    try:
        return expression.replace(_is_run, _rebuilt, simultaneous=False)
    except Exception:
        return expression


def _is_run(expression: sp.Basic) -> bool:
    return bool(expression.is_Add or expression.is_Mul)


def _rebuilt(run: sp.Basic) -> sp.Basic:
    try:
        return run.func(*run.args)
    except Exception:
        return run


# -- IF, the one place a relation is asked whether it holds -------------------


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
    """
    frozen: dict[sp.Basic, sp.Basic] = {}

    def freeze(head: sp.Basic) -> sp.Basic:
        placeholder = _placeholder(head, len(frozen))
        frozen[placeholder] = head
        return placeholder

    def resolve(head: sp.Basic) -> sp.Basic:
        if isinstance(head, sp.Piecewise):
            settled = _cases(head, context)
            return freeze(settled) if isinstance(settled, sp.Piecewise) else settled
        test, *branches = head.args
        decided = _decide(test, context)
        if decided is True:
            return branches[0]
        if decided is False:
            return branches[1] if len(branches) > 1 else sp.nan
        if len(branches) > 2:
            return branches[2]
        return freeze(head)

    try:
        resolved = expression.replace(_is_conditional, resolve, simultaneous=False)
    except Exception:
        return expression, {}
    return resolved, frozen


def _placeholder(head: sp.Basic, index: int) -> sp.Basic:
    """Something inert that depends on exactly what `head` depends on."""
    variables = sorted(head.free_symbols, key=sp.default_sort_key)
    if not variables:
        return sp.Dummy(f"IF{index}")
    return sp.Function(f"IF{index}", nargs=len(variables))(*variables)


def _is_conditional(expression: sp.Basic) -> bool:
    if isinstance(expression, sp.Piecewise):
        return True
    return (
        isinstance(expression, AppliedUndef)
        and type(expression).__name__ == "IF"
        and 2 <= len(expression.args) <= 4
    )


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
    brought together by the same rewrites the rest of the pipeline uses.
    """
    for candidate in _truths(test, context):
        if candidate is sp.true:
            return True
        if candidate is sp.false:
            return False
    return None


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
        return bounded
    if _parametrised(head) and not _affordable(head):
        return head if bounded is None else bounded
    full = _attempt(head, lambda h: h.doit(deep=False))
    return head if full is None else full


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


# -- the FACTOR and EXPAND heads ---------------------------------------------


def _commanded(expression: sp.Basic, context: Context) -> sp.Basic:
    """Evaluate every `FACTOR` and `EXPAND` head, innermost first.

    `FACTOR(u, amount, x, y, ...)` and `EXPAND(u, amount, x, y, ...)` are the
    author-line spellings of the two commands, so simplifying a line that
    carries one runs it - which is what the original does, and what makes the
    functions worth having rather than inert heads that print back as
    themselves. Both are matched in one pass so that either may be written
    inside the other.

    Last of the rewrites on purpose, and after `normal_form` and `_canonical`
    as well, because everything above this one would undo a factorization.
    The gated rewrites ask whether a candidate is shorter than what it
    replaces, and a factorization is routinely longer: `_multiplied_out`
    offered `(x - 3)*(2*x + 1)^2` would happily multiply it back out. The
    normal form does not ask, and would multiply out any factor that is a sum
    in the primary variable. And `_canonical` rebuilds every product it can
    reach, which would multiply a prime decomposition straight back into the
    integer it decomposes.
    """
    try:
        return expression.replace(
            _is_command, lambda head: _command(head, context), simultaneous=False
        )
    except Exception:
        return expression


def _is_command(expression: sp.Basic) -> bool:
    return isinstance(expression, AppliedUndef) and type(expression).__name__ in (
        "FACTOR",
        "EXPAND",
    )


def _command(head: sp.Basic, context: Context) -> sp.Basic:
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


# -- the gated rewrites ------------------------------------------------------


def _rewritten(expression: sp.Basic, context: Context) -> sp.Basic:
    """Every rewrite the settings select, each kept only if it pays."""
    expression = _branch(expression, context)
    expression = _gated(expression, _multiplied_out)
    expression = _gated(expression, _cancelled)
    if expression.has(*_COMBINATORIAL) and sp.count_ops(expression) <= _COMBSIMP:
        expression = _gated(expression, sp.combsimp)
    # Ours rather than sympy's, which collects neither these arcs nor their
    # tangents. Offered in every mode: a pair of them adding to a right angle
    # is an identity, not a direction to rewrite in.
    expression = _gated(expression, _complementary_arcs)
    expression = _trigonometry(expression, context)
    expression = _logarithms(expression, context)
    expression = _exponentials(expression, context)
    expression = _gated(expression, _denested)
    return _gated(expression, sp.radsimp)


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
    not have. Failing that, it has to be shorter.
    """
    try:
        if candidate == previous:
            return False
        if len(candidate.free_symbols) < len(previous.free_symbols):
            return True
        return sp.count_ops(candidate) < sp.count_ops(previous)
    except Exception:
        return False


def _cancelled(expression: sp.Basic) -> sp.Basic:
    """`cancel`, except where it would multiply a factored denominator out.

    Cancelling a common factor is what this is for, and it is worth having.
    Rebuilding the denominator is not: `cancel` writes every ratio over one
    expanded denominator, and where the normal form has already put a sum over
    a product of sums that turns the original's own answer into a longer one.
    `1/(9 + x^2 + (y - 3)^2) + 1/(9 + x^2 + (y + 3)^2)` is
    `2*(x^2 + y^2 + 18)/((x^2 + y^2 - 6*y + 18)*(x^2 + y^2 + 6*y + 18))` to
    Derive, and a quartic denominator to `cancel`.

    A denominator of one factor has nothing to lose, which is every sum of
    ratios that has not been put over a common denominator yet.

    What arrives here is not always an expression. An undecidable four-argument
    `IF` simplifies to its unknown clause, and `IF(u, true, false, false)` has a
    truth value for one; asking that for a numerator and a denominator is asking
    the wrong question, and sympy answers it by building a `Mul` out of a
    `BooleanFalse`, which it warns is on its way to being an error.
    """
    if not isinstance(expression, sp.Expr):
        return expression
    if len(sp.Mul.make_args(sp.fraction(expression)[1])) > 1:
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

    Below `_FOLDED` the expansion is small enough to be worth having outright.

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
    return expression


#: The inverse functions that pair off, each with the one it complements.
#: `ASIN(u) + ACOS(u)` is a right angle wherever both are defined, and so are
#: the secant pair and - for a positive argument - the tangent pair.
_COMPLEMENT = {
    sp.asin: sp.acos,
    sp.acos: sp.asin,
    sp.asec: sp.acsc,
    sp.acsc: sp.asec,
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
    """What the pair adds to, or None where nothing says which right angle.

    The tangent pair is the one that needs a domain: `ATAN(t) + ACOT(t)` is
    `pi/2` for positive `t` and `-pi/2` for negative, so an undeclared `t` -
    which is real and could be either - leaves the sum as it stands. Derive
    answers `pi/2` regardless, and is wrong for the negative half.
    """
    if func in (sp.atan, sp.acot):
        if argument.is_positive:
            return sp.pi / 2
        return -sp.pi / 2 if argument.is_negative else None
    return sp.pi / 2


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
    """
    match context.logarithm:
        case Direction.COLLECT:
            return _forced(expression, sp.logcombine)
        case Direction.EXPAND:
            return _forced(expression, sp.expand_log)
    expression = _gated(expression, sp.logcombine)
    return _gated(expression, sp.expand_log)


def _exponentials(expression: sp.Basic, context: Context) -> sp.Basic:
    """The Exponential setting, over the power rules sympy gates for us.

    `(#e^z)^k -> #e^(k*z)` needs an integer `k` or a strip constraint on `z`;
    `powsimp` and `expand_power_base` know that, so the validity gate is
    theirs and the direction is ours.
    """
    match context.exponential:
        case Direction.COLLECT:
            return _forced(expression, sp.powsimp)
        case Direction.EXPAND:
            expression = _forced(expression, sp.expand_power_exp)
            return _forced(expression, sp.expand_power_base)
    expression = _gated(expression, sp.powsimp)
    expression = _gated(expression, sp.expand_power_exp)
    return _gated(expression, sp.expand_power_base)


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

    Best effort, and it says so: whether a number is irrational is a question
    sympy sometimes leaves open, and a number it cannot classify is left
    exact.
    """
    try:
        return expression.replace(
            _is_irrational,
            lambda part: _simplest(_evalf(part, digits + GUARD), digits),
            simultaneous=False,
        )
    except Exception:
        return expression


def _is_irrational(expression: sp.Basic) -> bool:
    return (
        expression.is_number
        and expression.is_irrational is True
        and expression.is_finite is True
        and not isinstance(expression, sp.Float)
    )
