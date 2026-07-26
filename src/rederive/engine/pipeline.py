"""Simplify: the first command built on the engine's two doors.

Derive's Simplify has two goals at once. Reach a form that is *sufficiently
simple* - no superfluous variables, roots, functions or reducible degrees - and
get there by transforming as little as possible. The second goal is why this is
a deliberate sequence of named rewrites and not a call to `sympy.simplify`: a
wholesale simplifier decides for itself what a nicer expression is, and Derive
does not want a nicer expression, it wants the same expression with the slack
taken out.

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
   operations, or one variable fewer. This is the "expand only when it helps"
   rule, and it is what makes `(x + 1)^2 - x^2` become `2*x + 1` while
   `(x^2 + 2*x*y + y^2)/(x^2 - y^2)` becomes a ratio rather than a polynomial.
   A mode set to Collect or Expand is an instruction rather than an option, so
   those rewrites are applied whether or not they shorten anything.
5. Approximate, if the precision mode asks for it.

Every rewrite is offered inside a `try`, and a rewrite that raises is a rewrite
that was not worth having: the previous form stands. That is what makes the
command total. Nothing sympy does can turn a valid entry into an error.
"""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean
from sympy.simplify.fu import TR5, TR6, TR7, TR8

from rederive.engine.context import (
    Branch,
    Context,
    Direction,
    Precision,
    TrigPower,
)
from rederive.engine.from_sympy import Result, from_sympy
from rederive.engine.substitute import substitute
from rederive.engine.to_sympy import (
    Assign,
    Declare,
    FunDef,
    InertVector,
    Logical,
    PlusMinus,
    to_sympy,
)
from rederive.model.expr import Node
from rederive.syntax.state import ParseState

__all__ = ["approx", "simplified", "simplify"]

Rewrite = Callable[[sp.Basic], sp.Basic]

#: The heads that stand for a computation nobody has asked for yet.
_CALCULUS = (sp.Derivative, sp.Integral, sp.Sum, sp.Product, sp.Limit)

#: What `combsimp` is for. It is offered only where one of these appears,
#: because on an ordinary polynomial it factors - `x^2 + 2*x` becomes
#: `x*(x + 2)` - and factoring is the Factor command's business, not Simplify's.
_COMBINATORIAL = (sp.factorial, sp.factorial2, sp.binomial, sp.gamma, sp.ff, sp.rf)


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
    expression = _calculus(expression)
    expression = _numeric(expression)
    expression = _rewritten(expression, context)
    if frozen:
        expression = expression.xreplace(frozen)
    return _approximated(_canonical(expression), context)


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
    """Resolve every `IF` whose test can be decided, innermost first.

    Deciding the test is the whole of it. A true test leaves only the then
    clause to simplify, a false one only the else clause - `?` when there is
    none - and an undecidable test takes the unknown clause if the author wrote
    one.

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

    def resolve(head: sp.Basic) -> sp.Basic:
        test, *branches = head.args
        decided = _decide(test, context)
        if decided is True:
            return branches[0]
        if decided is False:
            return branches[1] if len(branches) > 1 else sp.nan
        if len(branches) > 2:
            return branches[2]
        placeholder = _placeholder(head, len(frozen))
        frozen[placeholder] = head
        return placeholder

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
    return (
        isinstance(expression, AppliedUndef)
        and type(expression).__name__ == "IF"
        and 2 <= len(expression.args) <= 4
    )


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


def _calculus(expression: sp.Basic) -> sp.Basic:
    """`.doit()` on every calculus head, innermost first."""
    try:
        return expression.replace(
            lambda e: isinstance(e, _CALCULUS), _evaluate, simultaneous=False
        )
    except Exception:
        return expression


def _evaluate(head: sp.Basic) -> sp.Basic:
    """One head evaluated, or the head itself if it will not evaluate."""
    if isinstance(head, sp.Integral):
        return _integral(head)
    try:
        value = head.doit(deep=False)
    except Exception:
        return _undecided(head)
    return head if value is None else value


#: Integration by the methods whose cost is bounded: sympy's tables, the
#: trigonometric rules, Risch, and the manual rules. Not `meijerg`, which
#: answers `INT(#e^x*COS(x), x)` with a page of Bessel functions when it is
#: reached on its own, and not `heurisch`, whose cost is the problem below.
_BOUNDED = {"meijerg": False, "heurisch": False}


def _integral(head: sp.Basic) -> sp.Basic:
    """An integral, by the bounded methods first.

    The heuristic Risch algorithm is what finds the integrals nothing else
    does, and its cost grows with the number of symbolic parameters in the
    integrand: it builds an ansatz over every generator it can see and solves
    for the coefficients. With no parameters to carry it is quick.
    `INT(t^(a-1)*(#e^(z*t)*(1-t)^(b-a-1) - 1), t, 0, 1/2)` - a real line out of
    the Kummer function in Derive's own utility library - carries three, and
    sympy spends most of a minute and gigabytes of memory on it before
    answering, correctly, that it does not know. Simplify must not do that: a
    minute is not an answer, and the same integral comes back unevaluated
    either way.

    So the expensive method is asked only where it is known to be affordable,
    and an integral over a parametrised integrand that the bounded methods
    cannot do comes back unevaluated. That is a real gap - sympy given all day
    does solve some of them - and it is deliberate: an unevaluated `INT` is a
    documented answer, and a minute of thrashing is not.
    """
    bounded = _attempt(head, lambda h: h.doit(deep=False, **_BOUNDED))
    if bounded is not None and not bounded.has(sp.Integral):
        return bounded
    if _parametrised(head):
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
    return PlusMinus(right) if right == -left else sp.nan


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
    expression = _gated(expression, sp.cancel)
    if expression.has(*_COMBINATORIAL):
        expression = _gated(expression, sp.combsimp)
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
    which is what makes `(x + 1)^2 - x^2` reach `2*x + 1`.
    """
    stood_in: dict[sp.Basic, sp.Symbol] = {}

    def hide(power: sp.Basic) -> sp.Basic:
        symbol = stood_in.get(power.base)
        if symbol is None:
            symbol = stood_in.setdefault(power.base, sp.Dummy())
        return symbol**power.exp

    hidden = expression.replace(_is_folded_power, hide, simultaneous=False)
    expanded = sp.expand(hidden)
    return expanded.xreplace({symbol: base for base, symbol in stood_in.items()})


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


def _approximated(expression: sp.Basic, context: Context) -> sp.Basic:
    digits = context.precision_digits
    match context.precision:
        case Precision.APPROXIMATE:
            return _whole(_evalf(expression, digits))
        case Precision.MIXED:
            return _mixed(expression, digits)
    return expression


def _evalf(expression: sp.Basic, digits: int) -> sp.Basic:
    try:
        return expression.evalf(digits)
    except Exception:
        return expression


def _whole(expression: sp.Basic) -> sp.Basic:
    """A float that is exactly an integer is written as that integer.

    Derive's approximate numbers are the simplest rationals accurate to the
    current precision, and the simplest rational that approximates an integer
    is the integer. So `2*y*3` is `6*y` in every precision mode, and only the
    numbers that actually need digits get them.
    """
    try:
        return expression.replace(_is_whole_float, _as_integer, simultaneous=False)
    except Exception:
        return expression


def _is_whole_float(expression: sp.Basic) -> bool:
    # Compared as the exact binary rational the float is: sympy does not hold
    # `Float(5.0)` and `Integer(5)` to be the same number.
    return isinstance(expression, sp.Float) and sp.Rational(expression) == int(expression)


def _as_integer(expression: sp.Basic) -> sp.Basic:
    return sp.Integer(int(expression))


def _mixed(expression: sp.Basic, digits: int) -> sp.Basic:
    """Approximate the irrational operations, keep the rational ones exact.

    Innermost first, so that a rational subexpression is computed before
    anything near it is rounded: `SQRT(3422357/2313 - 1140443/771)` is exactly
    `2/3` in Mixed mode, where Approximate rounds the two fractions first and
    gets `0.666622`.

    Best effort, and it says so: whether a number is irrational is a question
    sympy sometimes leaves open, and a number it cannot classify is left
    exact.
    """
    try:
        return expression.replace(
            _is_irrational, lambda e: _evalf(e, digits), simultaneous=False
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
