"""Evaluate: work out the tree innermost first, in an environment.

This is what Derive does before any rewriting. Names are replaced by their values,
calls to session-defined functions evaluate the body over the arguments, and an `IF`
test is decided before either branch is evaluated, so only the chosen branch is. The
result is a sympy expression that the rest of Simplify then simplifies.

The evaluator is the `to_sympy` conversion with an `Environment` threaded through,
overriding the node kinds whose meaning depends on context. `_Evaluator.convert`
dispatches them:

* A name: a parameter's argument, a bound variable (itself), or an assigned value,
  followed at most once per branch so `pn := pn + 1` simplifies to `pn + 1` rather
  than looping.
* A label, followed the same way.
* `IF(test, a, b, c)`: the test is decided and only the chosen branch evaluated. An
  undecidable test takes the fourth argument if present, otherwise it is suspended
  (see `Suspended`).
* A call of a defined function: arguments are evaluated, bound to the leading
  parameters (partial application leaves the rest as names), and the body evaluated.
  Recursion nests calls until the terminating test decides; past `_NESTING` levels
  the pending call is left in the answer.
* `F'(a)` of a defined `F`: the body is differentiated in its first parameter, then
  evaluated at `a`.
* A variable-binding function (`SUM`, `INT`, `VECTOR` and the rest of
  `BINDING_FUNCTIONS`), with its variable standing for itself. Over a range of known
  values, a counted `SUM` or `PRODUCT`, `VECTOR`, `SELECT` and `ITERATE` instead
  evaluate the body once per value, which decides tests on the index and enters
  recursions on it.
* Calculus heads and `APPROX`: computed on the spot via `Environment.work_out`, since
  their operands are already values.
* `FACTOR`, `EXPAND`, `SOLVE` and `RATE`: run on their evaluated operands via
  `Environment.command`, so they behave like ordinary calls.
* `SUBS(u, [x], [a])`: the substitution is made, unless it binds into a derivative,
  which is kept as sympy's `Subs`.
* An assignment keeps its target unevaluated, and a definition's parameters stand for
  themselves in its body. Inside a called function's body, an assignment or definition
  is also performed (see `_Evaluator.performs`).

Everything else is handled by the converter.
"""

from __future__ import annotations

import itertools
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import cached_property

import sympy as sp
from sympy.core.function import AppliedUndef, UndefinedFunction

from rederive.engine.approximation import carried
from rederive.engine.context import Context, Precision
from rederive.engine.from_sympy import from_sympy
from rederive.engine.intervals import decided
from rederive.engine.to_sympy import (
    CALCULUS_HEADS,
    COMMAND_HEADS,
    Assign,
    Converter,
    FunDef,
    Vector,
    added,
    as_condition,
    chosen_from,
    digit_count,
    divided,
    is_conditional,
    iterated,
    multiplied,
    negated,
    settled,
    steps,
)
from rederive.engine.vectors import mapped
from rederive.model.expr import Kind, Node
from rederive.syntax import ParseState
from rederive.syntax.names import (
    BINDING_FUNCTIONS,
    BINDING_THROUGHOUT,
    PRIME_MARK,
    SETTINGS,
)

__all__ = [
    "Commanded",
    "Environment",
    "Suspended",
    "divisors",
    "evaluate",
    "evaluated",
    "named_as_declared",
    "released",
]

Decider = Callable[[sp.Basic], bool | None]
WorkOut = Callable[[sp.Basic], sp.Basic]
Command = Callable[[str, list, list, Context], sp.Basic]


@dataclass(frozen=True)
class Environment:
    """What names mean at a point of the evaluation, and how tests are decided.

    `bindings` maps call parameters to their arguments, and names assigned in a test
    to their values for later branches. `supplied` records, for each call parameter,
    the argument as written and where it was written, to evaluate it again once the
    body gives a value to a variable it mentions (see `Argument`). `bound` holds names
    that stand for themselves: a definition's parameters, and the variable of a sum,
    integral or generated vector. `parameters` holds just the parameters of the
    definition being shown. `expanding` and `labels` are the assigned names and labels
    being followed on this branch, to stop self-reference. `depth` is the nesting
    depth of defined function calls.

    `written` holds the values a suspended conditional was resumed with, substituted
    wherever those variables occur.

    `decide` answers an `IF` test with True, False, or None if undecided. Simplify
    supplies one that uses rewrites and declared domains; otherwise
    `_plainly_decided` is used.

    `work_out` computes calculus heads, and `command` runs `FACTOR`, `EXPAND`, `SOLVE`
    or `RATE` on evaluated arguments (and, for `SOLVE`, the divisors of the equation
    as written; see `divisors`), at the precision of the point of call, which inside
    `APPROX` is Approximate. Simplify supplies both. Without them these heads are
    left as is, which suits callers that only want an expression's variables.

    `writing` means only substitute names, used to display an undecidable
    conditional: no call is entered and no test decided, so branches stay as written.
    `defining` means a definition is being shown, so the calls its body enters perform
    no assignment.
    """

    context: Context
    decide: Decider | None = None
    work_out: WorkOut | None = None
    command: Command | None = None
    bindings: Mapping[str, sp.Basic] = field(default_factory=dict)
    supplied: Mapping[str, Argument] = field(default_factory=dict, compare=False)
    bound: frozenset[str] = frozenset()
    parameters: frozenset[str] = frozenset()
    expanding: frozenset[str] = frozenset()
    labels: frozenset[int] = frozenset()
    depth: int = 0
    written: Mapping[sp.Basic, sp.Basic] = field(default_factory=dict)
    writing: bool = False
    defining: bool = False
    suspended: dict[sp.Basic, type] = field(default_factory=dict, compare=False)

    def decides(self, test: sp.Basic) -> bool | None:
        """Whether `test` holds, or None if undecided (including if deciding raises)."""
        try:
            if self.decide is not None:
                return self.decide(test)
            return _plainly_decided(test, self.context)
        except Exception:
            return None


@dataclass(frozen=True, eq=False)
class Argument:
    """A call's argument: as written, the environment it was written in, and its value.

    Definitions are textual substitution, so an argument mentioning a variable the body
    gives values to means something different at each value. `T(u, v, m) :=
    ITERATE([u], v, [1], m)` called with `w SUB 1 + 1` for `u` and `w` for `v` adds one
    to the first element at each step, which substituting into the value computed once
    can't do, since `w SUB 1` of a symbol `w` is a name of its own. Likewise
    `ELEMENT([a, b, c], k)` substituted for the index of a `VECTOR` only selects
    elements once evaluated at each `k`. So the argument is evaluated again where the
    body has given values to a variable in `names`: one written in it, or one its value
    mentions. Elsewhere `value` is used as is.
    """

    node: Node
    env: Environment
    value: sp.Basic

    @cached_property
    def names(self) -> frozenset[str]:
        return _written_names(self.node) | _symbol_names(self.value)

    def rewritten(self, written: Mapping[sp.Basic, sp.Basic]) -> dict | None:
        """The values `written` gives the variables of the argument since it was
        written, or None if it gives none.
        """
        if written is self.env.written:
            return None
        new = {
            symbol: value
            for symbol, value in written.items()
            if symbol not in self.env.written
        }
        if not any(getattr(symbol, "name", None) in self.names for symbol in new):
            return None
        return new


def _written_names(node: Node) -> frozenset[str]:
    """Every name written in `node`."""
    names = {str(node.value)} if node.kind is Kind.NAME else set()
    for child in node.children:
        names |= _written_names(child)
    return frozenset(names)


def _symbol_names(value: sp.Basic) -> frozenset[str]:
    """The names of the variables in `value`, and of the bases of subscripted ones."""
    try:
        symbols = value.free_symbols
    except Exception:
        return frozenset()
    names = set()
    for symbol in symbols:
        name = getattr(symbol, "name", None)
        if name is not None:
            names.add(name)
            names.add(name.split(" SUB ")[0])
    return frozenset(names)


def evaluate(node: Node, env: Environment) -> sp.Basic:
    """`node` evaluated in `env`, leaving undecidable conditionals suspended."""
    with _deep():
        return _Evaluator(env).convert(node)


def evaluated(node: Node, context: Context, decide: Decider | None = None) -> sp.Basic:
    """`node` evaluated with nothing bound, with conditionals released for display.

    For callers that just want a value, e.g. to list variables or to plot.
    """
    return released(evaluate(node, Environment(context, decide)))


def _plainly_decided(test: sp.Basic, context: Context) -> bool | None:
    """Decide a test with sympy, then with the declared intervals."""
    condition = as_condition(test)
    if condition is sp.true:
        return True
    if condition is sp.false:
        return False
    return decided(condition, context)


#: Maximum nesting depth of defined function calls. Typical Derive recursions
#: terminate long before this (the manual's `RAISE(b, 256)` reaches 257). One that
#: doesn't, like `F(n) := DIF(F(n + 1), x)` or `FACT(-1)`, returns what it built so
#: far with the pending call inside: a correct rewriting that shows why it won't
#: finish. Derive itself runs out of memory there.
_NESTING = 300

#: The Python recursion limit while evaluating. Each recursion level takes a few dozen
#: frames, so the default thousand would stop well before `_NESTING`.
_STACK = 50_000


@contextmanager
def _deep() -> Iterator[None]:
    previous = sys.getrecursionlimit()
    if previous < _STACK:
        sys.setrecursionlimit(_STACK)
    try:
        yield
    finally:
        sys.setrecursionlimit(previous)


class _Evaluator(Converter):
    """The converter with an environment; `convert` dispatches by node kind."""

    def __init__(self, env: Environment, performed: dict[str, sp.Basic] | None = None):
        super().__init__(env.context)
        self.env = env
        self.parameters = env.parameters
        #: Names assigned by the test being evaluated, with their values. None outside
        #: a test, where an assignment is a definition rather than an action.
        self.performed = performed

    def within(self, env: Environment) -> _Evaluator:
        return _Evaluator(env, self.performed)

    def convert(self, node: Node) -> sp.Basic:
        match node.kind:
            case Kind.CALL | Kind.APPLY:
                return self._call(node)
            case Kind.DOMAIN:
                # A declaration's bounds are read without the current scope.
                return Converter(self.context).convert(node)
        return super().convert(node)

    # -- names --------------------------------------------------------------

    def _name(self, node: Node) -> sp.Basic:
        name = str(node.value)
        env = self.env
        value = env.bindings.get(name)
        if value is not None:
            argument = env.supplied.get(name)
            if argument is not None and argument.value is value:
                rewritten = argument.rewritten(env.written)
                if rewritten is not None:
                    again = replace(_at(argument.env, rewritten), writing=env.writing)
                    return _value_of(_Evaluator(again).convert(argument.node))
            return _over(value, env.written)
        if name not in env.bound and name not in env.expanding:
            assigned = env.context.assigned(name)
            if assigned is not None:
                deeper = replace(env, bindings={}, expanding=env.expanding | {name})
                return self.within(deeper).convert(assigned)
        if not env.writing:
            digits = _digits_setting(name, env.context)
            if digits is not None:
                return digits
        return self._written(super()._name(node))

    def _written(self, symbol: sp.Basic) -> sp.Basic:
        return self.env.written.get(symbol, symbol)

    def _subscript(self, base_node: Node, base: sp.Basic, index: sp.Basic) -> sp.Basic:
        return self._written(super()._subscript(base_node, base, index))

    def operand(self, value: sp.Basic) -> sp.Basic:
        return _value_of(value)

    def _label(self, node: Node) -> sp.Basic:
        """The value of label `#3`, or the label itself if unknown or self-referential."""
        try:
            number = int(str(node.value))
        except ValueError:
            number = -1
        target = self.context.labels.get(number)
        if target is None or number in self.env.labels:
            return sp.Symbol(f"#{node.value}")
        deeper = replace(self.env, bindings={}, labels=self.env.labels | {number})
        return self.within(deeper).convert(target)

    # -- definitions --------------------------------------------------------

    def performs(self) -> bool:
        """Whether an assignment or definition here is performed.

        It is in the body of a called function, as in Derive: with `A1(u, x) := (A2(x)
        := u)`, `A1(y^2, y)` defines `A2(y) := y^2` and is `y^2`. It isn't in what the
        user wrote, which only states a definition (the session makes it), nor in a
        definition being shown or a conditional being written out.
        """
        env = self.env
        return env.depth > 0 and not env.writing and not env.defining

    def _assign(self, node: Node) -> sp.Basic:
        """An assignment: the value is evaluated, the target isn't.

        Inside a conditional's test it is also performed, Derive's idiom for
        computing something once and using it in the branches. ODE1.MTH's `DSOLVE1`
        relies on it: `IF("inapplicable" = a_ := SEP(...), ..., a_, a_)` tries a
        method, checks whether it declined, and returns its result. The assignment
        evaluates to the value, and `_conditional` binds the name in the branches.

        In a called function's body it is performed for good, and evaluates to the
        value; taking a value away evaluates to the assignment. FIT_.MTH's `[y_ :=,
        FIT_AUX5(...)] SUB 2` makes `y_` a variable again before using it.
        """
        target, *values = node.children
        written = [self.convert(child) for child in values]
        if target.kind is Kind.NAME and self.performs():
            name = self._target(str(target.value))
            if name is not None and name not in SETTINGS:
                assignment = Assign(
                    sp.Symbol(name), sp.Symbol(str(node.value)), *written[-1:]
                )
                if self._perform(assignment, Kind.ASSIGN) and self.performed is None:
                    return written[-1] if written else assignment
        if self.performed is not None and written and target.kind is Kind.NAME:
            self.performed[str(target.value)] = written[-1]
            return written[-1]
        plain = Converter(self.context)
        return Assign(plain.convert(target), sp.Symbol(str(node.value)), *written)

    def _target(self, name: str) -> str | None:
        """The name a definition here assigns or uses for `name`, or None if it has none.

        Definitions are textual substitution, so a parameter of the call stands for its
        argument, which must then be a variable.
        """
        value = self.env.bindings.get(name)
        if value is None:
            return name
        if type(value) is sp.Symbol and value.name.isidentifier():
            return value.name
        return None

    def _perform(self, definition: sp.Basic, kind: Kind) -> bool:
        """Perform `definition`, an `Assign` or `FunDef`, if it reads back as one."""
        try:
            node = from_sympy(definition, self.context).exact
        except Exception:
            return False
        if node.kind is not kind:
            return False
        self.context.performed.perform(node)
        return True

    def _fundef(self, node: Node) -> sp.Basic:
        """A definition: the body is evaluated, its parameters standing for themselves.

        A variable-binding call over a parameter is left as a call, since its variable
        is whatever a call of the function supplies (see `_binding`).

        In a called function's body it is performed instead (see `_performed`).
        """
        if self.performs():
            performed = self._performed(node)
            if performed is not None:
                return performed
        parameters, *body = node.children
        names = frozenset(str(child.value) for child in parameters.children)
        inner = replace(
            self.env,
            bindings={},
            bound=self.env.bound | names,
            parameters=names,
            written=_unwritten(self.env.written, names),
            defining=True,
        )
        plain = Converter(self.context)
        written = [self.within(inner).convert(part) for part in body]
        return FunDef(
            sp.Symbol(str(node.value)),
            sp.Tuple(*(plain.convert(child) for child in parameters.children)),
            *written,
        )

    def _performed(self, node: Node) -> sp.Basic | None:
        """A definition in a called function's body, performed: the body's value.

        The call's arguments are written into the definition, its parameters included:
        `A1(u, x) := (A2(x) := u)` called with `y^2` and `y` defines `A2(y) := y^2`.
        The body is stored as written, with only names substituted, and the new
        definition holds while the body's value is worked out, so it may recurse.

        None, leaving the definition as written, if a parameter is supplied with
        something other than a variable, or the definition doesn't read back.
        """
        parameters, *body = node.children
        name = self._target(str(node.value))
        names = [self._target(str(child.value)) for child in parameters.children]
        if name is None or None in names:
            return None
        head = sp.Symbol(name)
        symbols = sp.Tuple(*(sp.Symbol(parameter) for parameter in names))
        if not body:
            definition = FunDef(head, symbols)
            return definition if self._perform(definition, Kind.FUNDEF) else None
        written = replace(self.env, writing=True, bound=self.env.bound | set(names))
        stored = released(self.within(written).convert(body[0]))
        if not self._perform(FunDef(head, symbols, stored), Kind.FUNDEF):
            return None
        return self.convert(body[0])

    # -- calls --------------------------------------------------------------

    def _call(self, node: Node) -> sp.Basic:
        head, *arguments = node.children
        name = str(head.value)
        if name in BINDING_FUNCTIONS:
            return self._binding(name, arguments)
        if name == "IF" and 1 <= len(arguments) <= 4 and not self.env.writing:
            return self._conditional(node, arguments)
        if name == "APPROX" and 1 <= len(arguments) <= 2 and not self.env.writing:
            values = self._approximating(arguments)
        else:
            values = [_value_of(self.convert(argument)) for argument in arguments]
        if name == "SUBS" and not self.env.writing:
            return self._substitution(values)
        derived = self._derived(name, values)
        if derived is not None:
            return derived
        definition = self.context.function(name)
        if definition is not None:
            if self.env.writing:
                return self.opaque(name, values)
            return self._entered(name, definition, values, arguments)
        made = self.call(name, values)
        if name == "FACTORS" and _decomposed(values, made) and not self.env.writing:
            return Commanded(made)
        if name in COMMAND_HEADS and _is_call_of(made, name):
            if self.env.command is not None and not self.env.writing:
                return self.env.command(
                    name, list(made.args), self._divisors(name, arguments), self.context
                )
        return self._worked(made, values)

    def _approximating(self, arguments: Sequence[Node]) -> list[sp.Basic]:
        """The arguments of `APPROX(u, n)`, with `u` evaluated under Approximate.

        So `u` is computed as approX computes a whole entry, carrying each step's value
        numerically (see `carried`), rather than exactly and rounded only at the end.
        The digit count is evaluated first, in the enclosing scope; if it isn't a count
        yet, `u` is evaluated at the session's digits.
        """
        operand, *rest = arguments
        digits = [_value_of(self.convert(argument)) for argument in rest]
        count = digit_count(digits[0]) if digits else None
        approximate = self.context.with_precision(Precision.APPROXIMATE, count)
        inner = self.within(replace(self.env, context=approximate))
        return [_value_of(inner.convert(operand)), *digits]

    def _divisors(self, name: str, arguments: Sequence[Node]) -> list[sp.Basic]:
        """The divisors of a `SOLVE` equation as written, evaluated here."""
        if name != "SOLVE" or not arguments:
            return []
        found = []
        for divisor in divisors(arguments[0]):
            try:
                found.append(_value_of(self.convert(divisor)))
            except Exception:
                continue
        return found

    def _worked(self, made: sp.Basic, values: Sequence[sp.Basic]) -> sp.Basic:
        """`made` with every calculus head this call created worked out.

        Heads already in the arguments were tried when those were made, so they are
        skipped: an unsolvable integral is attempted only once. A result may contain
        new heads (sympy may answer an integral with a sum of integrals), and each is
        worked out once by the same rule.
        """
        work_out = self.env.work_out
        if work_out is None or self.env.writing:
            return made
        try:
            if not made.has(*CALCULUS_HEADS):
                return made
            standing = set()
            for value in values:
                if isinstance(value, sp.Basic):
                    standing |= value.atoms(*CALCULUS_HEADS)
        except Exception:
            return made
        # Sympy computes with a vector as if it were a scalar, so its answer may need
        # the vector arithmetic done: an antidifference of a vector is `x` times it.
        return settled(_worked_out(made, standing, work_out))

    def _entered(
        self,
        name: str,
        definition: tuple[tuple[str, ...], Node],
        values: list,
        arguments: Sequence[Node],
    ) -> sp.Basic:
        """A call of a defined function: its body evaluated over its arguments.

        Partial application binds only the leading parameters: `ACCELERATION(f, m) :=
        f/m` applied to 6 alone is `6/m`.

        Each parameter stands for its argument's value, and the argument as written is
        kept to be evaluated again where the body gives values to its variables (see
        `Argument`).
        """
        if self.env.depth >= _NESTING:
            return self.opaque(name, values)
        parameters, body = definition
        inner = replace(
            self.env,
            bindings=dict(zip(parameters, values, strict=False)),
            supplied={
                parameter: Argument(argument, self.env, value)
                for parameter, argument, value in zip(
                    parameters, arguments, values, strict=False
                )
            },
            bound=frozenset(),
            parameters=frozenset(),
            depth=self.env.depth + 1,
        )
        return carried(_Evaluator(inner).convert(body), self.context)

    def _derived(self, name: str, values: list) -> sp.Basic | None:
        """`F'(a)` for a defined `F`: its body differentiated and evaluated at `a`.

        A prime differentiates the function, not the call, so the body is
        differentiated in its first parameter before substituting: with `F(r) :=
        r^2`, `F'(r^2)` is `2*r^2`, not `2*r`. The point may mention the variable of
        differentiation, so it is taken as a limit, as Derive does: it prints
        `lim d/d@ F(@)` at the point for a derivative it can't take.

        None if there are no primes or `F` isn't defined; an undefined function stays
        a call for the converter to treat as a derivative.
        """
        base = name.rstrip(PRIME_MARK)
        order = len(name) - len(base)
        definition = self.context.function(base)
        if not order or definition is None or not values or self.env.writing:
            return None
        parameters, body = definition
        if not parameters or self.env.depth >= _NESTING:
            return None
        variable = self.symbol(parameters[0])
        inner = replace(
            self.env,
            bindings=dict(zip(parameters[1:], values[1:], strict=False)),
            bound=frozenset({parameters[0]}),
            depth=self.env.depth + 1,
        )
        derived = [_value_of(_Evaluator(inner).convert(body)), variable]
        if order > 1:
            derived.append(sp.Integer(order))
        at = [self._worked(self.call("DIF", derived), derived), variable, values[0]]
        return self._worked(self.call("LIM", at), at)

    def _substitution(self, values: list) -> sp.Basic:
        """`SUBS(u, [x, ...], [a, ...])`, performed where possible.

        A `Subs` of an ordinary expression is performed: `SUBS(BB(u), [u], [r^2])`
        is `BB(r^2)`. Only one containing a derivative is kept, since that derivative
        is deliberately unevaluated: the slope of `F` at `y` would otherwise become
        the slope of a constant once its variable is substituted.
        """
        held = self.call("SUBS", values)
        if not isinstance(held, sp.Subs) or held.expr.has(sp.Derivative):
            return held
        try:
            return held.doit(deep=False)
        except Exception:
            return held

    # -- the variable a function binds --------------------------------------

    def _binding(self, name: str, arguments: list[Node]) -> sp.Basic:
        """`SUM(u, k, m, n)` and similar: `k` stands for itself in `u` only.

        An assigned index isn't substituted (`x := 5` would leave nothing to range
        over), and the arguments after the index are evaluated in the enclosing
        scope, since limits aren't part of what they bound.

        A parameter index, however, is substituted with the name the caller supplied:
        `LIN1_DIFFERENCE(p, q, x, x0, y0)` sums over `x`, so called with `m` it sums
        over `m`. The body keeps that parameter bound, so the index can capture what
        another parameter supplied: `PRODUCT(p, x, x0, x - 1)` called with `m` for
        both `p` and `x` is the product of `m` over `m`, `(m - 1)!` as in Derive.
        Definitions are textual substitution, capture included.

        A vector of names binds all of them, as in iterated limits and system
        iterations: `LIM(u, [x, y], [a, b])` and `ITERATE([k, j + k], [j, k], [0, 1],
        n)`. A parameter supplied with such a vector works the same way, which
        SOLVE.MTH depends on: `NEWTON_AUX`'s `LIM(a, x, xk)` is called with `[x, y]`
        for `x`.

        A parameter supplied with an element whose index isn't known yet binds nothing
        either, so `DIF(u, v)` called with `[x1, x2] SUB k` for `v` stays a call until
        `k` is a number, as written out it would.

        A second argument written otherwise binds what it evaluates to, if that names
        variables: TENSOR.MTH's `vNestx(P, ivP, d, n)` generates `VECTOR(P, ivP SUB n,
        d)`, over the `n`th name in `ivP`. One that evaluates to anything else binds
        nothing, nor does `SUM(v)` over a vector; both are evaluated like ordinary
        calls.

        In a definition being shown, a binding over one of its parameters is left as a
        call with its arguments evaluated: `SB(u, v, a) := ITERATE(u, v, a, 1)` iterates
        over whatever variable a call supplies for `v`, which `u` may then mention.

        For `BINDING_THROUGHOUT` heads the binding covers the later arguments too: a
        root sum ranges over the roots of a polynomial in the same variable, so an
        assigned `t` reaches no part of `ROOT_SUM(p, t, u)`.
        """
        if len(arguments) < 2:
            return self._unbound(name, arguments)
        body, written_index, *rest = arguments
        if _variables_of(written_index) is not None and not self._pending(written_index):
            index, names, shadowed = self._index(written_index)
        else:
            index = _value_of(self.convert(written_index))
            names, shadowed = _names_in(index), frozenset()
            if names is None:
                return self._unbound(name, arguments, index)
        inner = replace(
            self.env,
            bindings={n: v for n, v in self.env.bindings.items() if n not in shadowed},
            bound=self.env.bound | names,
            written=_unwritten(self.env.written, names),
        )
        evaluator = self.within(inner)
        outer = evaluator if name in BINDING_THROUGHOUT else self
        limits = [_value_of(outer.convert(argument)) for argument in rest]
        taken = _taken(self.env.written, names)
        if self.binds_parameter(index):
            return self.opaque(name, [_value_of(evaluator.convert(body)), index, *limits])
        if name in _COUNTED and not self.env.writing:
            counted = self._counted(name, body, index, inner, limits)
            if counted is not None:
                return _over(counted, taken)
        values = [_value_of(evaluator.convert(body)), index, *limits]
        return _over(self._worked(self.call(name, values), values), taken)

    def _pending(self, index: Node) -> bool:
        """Whether `index` is a parameter supplied with an element not yet known."""
        if index.kind is not Kind.NAME:
            return False
        supplied = self.env.bindings.get(str(index.value))
        return supplied is not None and _is_call_of(supplied, "ELEMENT")

    def _unbound(
        self, name: str, arguments: list[Node], index: sp.Basic | None = None
    ) -> sp.Basic:
        """A binding function's call that binds nothing, evaluated like any other.

        `index` is the second argument's value, if already evaluated.
        """
        values = [
            index if i == 1 and index is not None else _value_of(self.convert(argument))
            for i, argument in enumerate(arguments)
        ]
        return self._worked(self.call(name, values), values)

    def _counted(
        self,
        name: str,
        body: Node,
        index: sp.Basic,
        inner: Environment,
        limits: list[sp.Basic],
    ) -> sp.Basic | None:
        """A binding call over known values, evaluating the body at each.

        The body is evaluated per value with the variable substituted, rather than
        converted once and substituted into, so tests on the index decide,
        recursions on it are entered, and vector elements it indexes are available.
        NUMBER.MTH's `DISTINCT_PARTS` sums a `PARTS` guarded by a test on the index,
        and VECTOR.MTH's `NONZERO_ROWS` sums a test over `m SUB n_`, which is a row
        only once `n_` is a number.

        None if the values aren't all known (a symbolic end, or more than
        `_WRITTEN_OUT`); the call is then made over the body as is, for sympy to
        handle. Also None if a `SELECT` test is undecidable or an iteration can't
        continue.
        """

        def at(*values: sp.Basic) -> sp.Basic:
            variables = list(index.args) if isinstance(index, Vector) else [index]
            written = dict(zip(variables, values, strict=True))
            return self.within(_at(inner, written)).convert(body)

        try:
            if name in ("ITERATE", "ITERATES"):
                if not limits or len(limits) > 2:
                    return None
                start, *count = limits
                return iterated(
                    name,
                    self,
                    index,
                    start,
                    count,
                    lambda _, values: _value_of(at(*values)),
                )
            if type(index) is not sp.Symbol:
                return None
            if name in ("SUM", "PRODUCT"):
                return _total(name, limits, at, self.context)
            if name == "VECTOR":
                return Vector(*(at(value) for value in steps(limits)))
            return chosen_from(steps(limits), lambda value: _value_of(at(value)))
        except Exception:
            return None

    def _index(self, index: Node) -> tuple[sp.Basic, frozenset[str], frozenset[str]]:
        """The variable a binding call ranges over, the names it binds, and the
        parameters it shadows.

        A parameter here is replaced by the caller's argument, if that names
        variables. Otherwise (a number or expression), the parameter itself is the
        index and shadows its argument in the body.
        """
        if index.kind is Kind.NAME:
            name = str(index.value)
            supplied = self.env.bindings.get(name)
            names = _names_in(supplied) if supplied is not None else None
            if names is None:
                return self.symbol(name), frozenset({name}), frozenset({name})
            return supplied, names, frozenset()
        values, names, shadowed = [], frozenset(), frozenset()
        for child in index.children:
            value, named, hidden = self._index(child)
            values.append(value)
            names |= named
            shadowed |= hidden
        return Vector(*values), names, shadowed

    # -- IF -----------------------------------------------------------------

    def _conditional(self, node: Node, arguments: list[Node]) -> sp.Basic:
        """`IF(test, a, b, c)`: evaluate only the branch the test picks.

        True picks the then clause, false the else clause (`?` if absent), and
        undecidable the unknown clause if present. A bare test is 1 or 0, so
        `SUM(IF(PRIME(n)), n, 1, 100)` counts primes.

        Without an unknown clause an undecidable `IF` is suspended whole, as in
        Derive: evaluating a branch that may never be taken could turn a guarded
        division by zero into an error, and recursion guards are such tests.
        """
        test_node, *branches = arguments
        performed: dict[str, sp.Basic] = {}
        test = _value_of(_Evaluator(self.env, performed).convert(test_node))
        verdict = self.env.decides(test)
        env = self.env
        if performed:
            env = replace(
                env,
                bindings={**env.bindings, **performed},
                bound=env.bound - frozenset(performed),
            )
        if not branches:
            if verdict is None:
                return self._suspended(node)
            return sp.Integer(1) if verdict else sp.Integer(0)
        if verdict is True:
            return self.within(env).convert(branches[0])
        if verdict is False:
            if len(branches) < 2:
                return sp.nan
            return self.within(env).convert(branches[1])
        if len(branches) > 2:
            return self.within(env).convert(branches[2])
        return self._suspended(node)

    def _suspended(self, node: Node) -> sp.Basic:
        """An undecidable conditional, suspended as a function of its variables."""
        display = _Evaluator(replace(self.env, writing=True)).convert(node)
        if not is_conditional(display):
            return display
        cls = self.env.suspended.get(display)
        if cls is None:
            variables = tuple(sorted(display.free_symbols, key=sp.default_sort_key))
            cls = UndefinedFunction(
                f"IF_{next(_SUSPENSIONS)}",
                bases=(Suspended,),
                __dict__={
                    "node": node,
                    "environment": self.env,
                    "display": display,
                    "variables": variables,
                },
            )
            self.env.suspended[display] = cls
        try:
            return cls(*cls.variables)
        except Exception:
            return display


class Suspended(AppliedUndef):
    """A conditional with an undecided test, as a function of its variables.

    Derive shows an undecidable `IF` as written, so later processing must not look
    inside it, nor treat it as a constant. A function of exactly its variables does
    both: `SUM(IF(k > 0, k, 0), k, 1, n)` stays a sum, where a bare name would let
    sympy answer `n*IF(k > 0, k, 0)`, and `DIF(IF(x > 0, x^2, 0), x)` stays a
    derivative.

    Substituting values for its variables (a sum written out term by term, an
    iteration step) resumes the suspended evaluation with those values, so the test
    may now decide. NUMBER.MTH's `DISTINCT_PARTS` sums terms that each guard a
    `PARTS` with a test on the index, decidable only once the index is substituted.

    Like any undefined function, sympy doesn't assume continuity, so a limit isn't
    taken by substituting the point.

    Each is its own class, made by `_Evaluator._suspended`, carrying the node, its
    environment, its `display` (the conditional with names substituted and nothing
    evaluated), and the `variables` it is applied to. `released` replaces whatever
    remains at the end with its display.
    """

    node: Node
    environment: Environment
    display: sp.Basic
    variables: tuple[sp.Basic, ...] = ()

    @classmethod
    def eval(cls, *args):
        if cls is Suspended or args == cls.variables:
            return None
        written = {
            variable: value
            for variable, value in zip(cls.variables, args, strict=True)
            if variable != value
        }
        if not all(_resumes(value) for value in written.values()):
            return None
        try:
            return _resumed(cls, written)
        except Exception:
            return None


def _resumes(value: sp.Basic) -> bool:
    """Whether substituting `value` for a variable should resume evaluation.

    Yes for finite numbers and truth values, as substituted by sums, iterations and
    truth tables. Sympy also substitutes for its own purposes (a limit may use an
    infinity or the reciprocal of its own variable), and testing there would answer a
    question nobody asked; those leave the conditional suspended, with the value
    substituted into its display.
    """
    if isinstance(value, sp.logic.boolalg.BooleanAtom):
        return True
    return bool(getattr(value, "is_number", False)) and value.is_finite is not False


#: Counter for naming suspended conditional classes uniquely.
_SUSPENSIONS = itertools.count()


def _resumed(cls: type[Suspended], written: dict[sp.Basic, sp.Basic]) -> sp.Basic:
    """The suspended conditional re-evaluated with its variables given these values."""
    with _deep():
        return _Evaluator(_at(cls.environment, written)).convert(cls.node)


def _at(env: Environment, written: Mapping[sp.Basic, sp.Basic]) -> Environment:
    """`env` with these values substituted for these variables.

    This also applies to parameter values, since definitions are textual
    substitution: `PRODUCT(p, x, x0, x - 1)` called with `m` for both `p` and `x`
    multiplies `m` at every value of `m`. Parameter values are substituted when used
    (see `_Evaluator._name`) rather than here, so a nested binding call that rebinds
    the variable sees the parameter as supplied.
    """
    return replace(
        env,
        written={
            **{symbol: _over(value, written) for symbol, value in env.written.items()},
            **written,
        },
    )


def _over(value: sp.Basic, written: Mapping[sp.Basic, sp.Basic]) -> sp.Basic:
    """`value` with `written` substituted, where possible."""
    try:
        if not any(symbol in value.free_symbols for symbol in written):
            return value
        return settled(value.subs(written, simultaneous=True))
    except Exception:
        return value


#: Binding calls whose body is evaluated once per value when the range is known.
_COUNTED = frozenset({"SUM", "PRODUCT", "VECTOR", "SELECT", "ITERATE", "ITERATES"})

#: Maximum number of terms to write out a range as. Longer ranges are left to sympy.
#: Derive's own utility files use at most four.
_WRITTEN_OUT = 1000


def _total(
    name: str,
    limits: list[sp.Basic],
    at: Callable[[sp.Basic], sp.Basic],
    context: Context,
) -> sp.Basic | None:
    """A counted `SUM` or `PRODUCT`: the terms at each value, added or multiplied.

    Each running total is `carried`, so under Approximate a long sum stays numeric.

    The values are a vector, or a range (from 1 if only one end is given). None if
    neither, or if a term isn't an expression, leaving the sum to sympy. A backwards
    range gives the negated sum or reciprocal product over the values stepped back
    over.
    """
    backwards = False
    if len(limits) == 1 and isinstance(limits[0], Vector):
        values = steps(limits)
    elif len(limits) in (1, 2):
        run = _run_of(*limits) if len(limits) == 2 else _run_of(sp.Integer(1), *limits)
        if run is None:
            return None
        values, backwards = run
    else:
        return None
    terms = [at(value) for value in values]
    if not all(isinstance(term, sp.Expr) for term in terms):
        return None
    def combine(*parts: sp.Basic) -> sp.Basic:
        if name == "SUM":
            return added(list(parts))
        return multiplied(list(parts)) if parts else sp.Integer(1)

    if context.precision is Precision.APPROXIMATE:
        total = combine()
        for term in terms:
            total = carried(combine(total, term), context)
    else:
        total = combine(*terms)
    if name == "SUM":
        return negated(total) if backwards else total
    return divided(sp.Integer(1), total) if backwards else total


def _run_of(low: sp.Basic, high: sp.Basic) -> tuple[list[sp.Basic], bool] | None:
    """The values an index takes from `low` to `high`, and whether backwards.

    As in Derive, the index starts at `low` and steps by one while not past `high`,
    so a non-integer end is meaningful. `PARTS` bounds Rademacher's series by
    `SQRT(n)/LOG(n, 11)`, which for `n` of 4 is 3.459: the index runs 1, 2, 3, where
    sympy would plug the bound into Faulhaber's polynomial.

    If `high` is below `low`, it counts that many steps back from `low - 1`, which
    keeps `SUM(u, k, a, b) + SUM(u, k, b + 1, c)` equal to the sum from `a` to `c` in
    any order: `SUM(k, k, 5, 3.5)` is -4, and `SUM(k, k, 5.5, 3)` is -8 (from 4.5 and
    3.5).

    Both ends must be numbers and the count at most `_WRITTEN_OUT`.
    """
    if not low.is_number:
        return None
    count = sp.floor(high - low) + 1
    if not count.is_Integer or abs(count) > _WRITTEN_OUT:
        return None
    if count >= 0:
        return [low + step for step in range(int(count))], False
    return [low - 1 - step for step in range(int(-count))], True


def _worked_out(made: sp.Basic, standing: set[sp.Basic], work_out: WorkOut) -> sp.Basic:
    """Every calculus head in `made` except those in `standing`, worked out.

    Each distinct head is worked out once, and its result is searched for new heads
    the same way. If a result contains the head being worked out, it is left as is.
    """
    answers: dict[sp.Basic, sp.Basic] = {}

    def fresh(found: sp.Basic) -> bool:
        return isinstance(found, CALCULUS_HEADS) and found not in standing

    def work(head: sp.Basic) -> sp.Basic:
        if head not in answers:
            answers[head] = head
            try:
                answer = work_out(head)
                if answer != head and answer.has(*CALCULUS_HEADS):
                    answer = answer.replace(fresh, work, simultaneous=False)
            except Exception:
                answer = head
            answers[head] = answer
        return answers[head]

    try:
        return made.replace(fresh, work, simultaneous=False)
    except Exception:
        return made


class Commanded(sp.Function):
    """The answer of a `FACTOR`, `EXPAND` or `SOLVE` head, in the command's form.

    A factorization is a form, not just a value: `FACTOR(12)` is `2^2*3`, which
    Simplify's normal form would multiply back out. A solution is written as soLve
    writes it, `x >= (3*y - 7)/2`, not as Simplify would. So the answer is wrapped
    through the rest of the pipeline and unwrapped only at the end. A call taking it
    as an argument gets the unwrapped value: `RHS(SOLVE(x^2 - 5*x + 6 = 0, x))` is
    `[2, 3]`.
    """

    nargs = 1


def _is_call_of(value: sp.Basic, name: str) -> bool:
    return isinstance(value, AppliedUndef) and type(value).__name__ == name


#: Node kinds `divisors` looks inside: arithmetic, relations and calls. Definitions,
#: assignments and vectors say nothing about where an equation is undefined.
_ARITHMETIC = frozenset(
    {Kind.SUM, Kind.PRODUCT, Kind.BINOP, Kind.UNOP, Kind.ABS, Kind.CALL, Kind.REL}
)


def divisors(node: Node) -> list[Node]:
    """The subtrees `node` divides by as written, including nested ones.

    An equation is undefined where one is zero, which only the written form shows:
    `x^2/x = 0` and `(x^2 - 1)/(x - 1) = 2` simplify to `x = 0` and `x + 1 = 2`, whose
    only solutions are where the original is undefined. So soLve is given the
    divisors: denominators, and bases raised to negative numbers. The contents of
    `SUM`, `INT` and other binding functions are skipped, as their divisions are in
    their own variable.
    """
    found: list[Node] = []

    def walk(part: Node) -> None:
        if part.kind not in _ARITHMETIC:
            return
        if part.kind is Kind.CALL and str(part.children[0].value) in BINDING_FUNCTIONS:
            return
        if part.kind is Kind.BINOP and part.value == "/":
            found.append(part.children[1])
        elif part.kind is Kind.BINOP and part.value == "^":
            if _negative(part.children[1]):
                found.append(part.children[0])
        for child in part.children:
            walk(child)

    walk(node)
    return found


def _negative(exponent: Node) -> bool:
    """Whether an exponent is a negated number, as in `x^(-2)`."""
    return (
        exponent.kind is Kind.UNOP
        and exponent.value == "-"
        and exponent.children[0].kind is Kind.NUMBER
    )


def _digits_setting(name: str, context: Context) -> sp.Integer | None:
    """The value of `PrecisionDigits` or `NotationDigits`, or None for another name.

    The two numeric settings read as variables, 6 by default, as in Derive: BESSEL.MTH's
    `BESSEL_J` sums `FLOOR(PrecisionDigits/2 + 4*ABS(z))` terms. The value is the one in
    force, so inside `APPROX(u, 20)` it is 20. The other settings have keyword values
    and stay names.
    """
    if name == "PrecisionDigits":
        return sp.Integer(context.precision_digits)
    if name == "NotationDigits":
        return sp.Integer(context.notation_digits)
    return None


def _decomposed(values: Sequence[sp.Basic], made: sp.Basic) -> bool:
    """Whether `made` is `FACTORS` of a prime decomposition, `FACTORS(FACTOR(360))`.

    Its elements are the prime powers as `FACTOR` wrote them, `[2^3, 3^2, 5]`, which is
    a form Simplify would multiply out, so it is shown as a command's answer is.
    """
    return (
        len(values) == 1
        and isinstance(values[0], sp.UnevaluatedExpr)
        and isinstance(made, Vector)
    )


def _value_of(value: sp.Basic) -> sp.Basic:
    """`value`, unwrapped from `Commanded` if wrapped.

    The prime powers of `FACTORS(FACTOR(360))` are values like any other once used:
    `PRIME` of its first element is false, that element being 8.
    """
    if not isinstance(value, Commanded):
        return value
    inner = value.args[0]
    if isinstance(inner, Vector) and any(
        isinstance(element, sp.Pow) and element.is_number for element in inner.args
    ):
        return mapped(inner, lambda element: element.doit())
    return inner


def released(value: sp.Basic) -> sp.Basic:
    """`value` with every suspended conditional and `Commanded` unwrapped for display.

    A suspended conditional is shown as originally written, with its variables'
    current values substituted, so an untouched one prints as the author typed it. A
    command's answer is shown in the command's form. One that is a vector takes part
    in the arithmetic around it only now: `2*SOLVE(x = 1, x)` is `[2*x = 2]`.
    """
    try:
        if not value.atoms(Suspended, Commanded):
            return settled(value)
    except Exception:
        return value
    try:
        return settled(_released(value))
    except Exception:
        pass
    # Rebuilding the parents of what was released can fail where the display is
    # rejected as an operand (`1 > DIMENSION(...)` around a conditional), so they are
    # rebuilt as they stand, without evaluation.
    try:
        with sp.evaluate(False):
            return _released(value)
    except Exception:
        return value


def _released(value: sp.Basic) -> sp.Basic:
    return value.replace(lambda found: isinstance(found, Suspended | Commanded), _shown)


def _shown(found: Suspended | Commanded) -> sp.Basic:
    if isinstance(found, Commanded):
        return released(found.args[0])
    cls = type(found)
    shown = cls.display
    if found.args != cls.variables:
        shown = _over(shown, dict(zip(cls.variables, found.args, strict=True)))
    return released(shown)


def _variables_of(index: Node) -> tuple[str, ...] | None:
    """The variables `index` names (one name or a vector of names), or None."""
    if index.kind is Kind.NAME:
        return (str(index.value),)
    if index.kind is not Kind.VECTOR or not index.children:
        return None
    if any(child.kind is not Kind.NAME for child in index.children):
        return None
    return tuple(str(child.value) for child in index.children)


def _taken(
    written: Mapping[sp.Basic, sp.Basic], names: frozenset[str]
) -> Mapping[sp.Basic, sp.Basic]:
    """The substitutions for the variables a binding rebinds.

    The binding works with its variable symbolic, and these values are substituted
    into its result afterwards, in Derive's order: `VECTOR(DIF(x^2, x), x, 1, 3)`
    differentiates in `x` and then evaluates `2*x` at each value. SOLVE.MTH's
    `NEWTON` is an `ITERATES` over a `DIF` in the variable it iterates.
    """
    return {
        symbol: value
        for symbol, value in written.items()
        if getattr(symbol, "name", None) in names
    }


def _unwritten(
    written: Mapping[sp.Basic, sp.Basic], names: frozenset[str]
) -> Mapping[sp.Basic, sp.Basic]:
    """The substitutions, minus the variables a binding rebinds.

    A sum over `k` inside a conditional resumed at `k = 2` has its own `k`.
    """
    if not any(getattr(symbol, "name", None) in names for symbol in written):
        return written
    return {
        symbol: value
        for symbol, value in written.items()
        if getattr(symbol, "name", None) not in names
    }


def _names_in(value: sp.Basic) -> frozenset[str] | None:
    """The variables a supplied argument names, or None if it isn't names."""
    if type(value) is sp.Symbol:
        return frozenset({value.name})
    elements = list(value.args) if isinstance(value, Vector) else []
    if not elements or any(type(element) is not sp.Symbol for element in elements):
        return None
    return frozenset(element.name for element in elements)


# -- spelling -------------------------------------------------------------------


def named_as_declared(node: Node, state: ParseState | None) -> Node:
    """`node` with every name spelled as the symbol table records it.

    In the default case-insensitive mode `x` and `X` are one variable, and the
    spelling depends on what was declared when the line was read. `X := ELEMENT(x,
    1)` declares `X` but mentions `x` before that, so the line contains both; the
    engine needs one spelling per variable.

    Without a symbol table the tree is returned unchanged.
    """
    if state is None:
        return node
    return _named(node, state)


def _named(node: Node, state: ParseState) -> Node:
    """The walk for `named_as_declared`. Call heads are left alone.

    The table's spelling is only meaningful for variables; for other names it may be
    something else entirely (the setting `Exact` for `EXACT(...)`, the letter `δ`
    for `DELTA(...)`), which wouldn't read back as the same call.
    """
    head = node.children[0] if node.kind in (Kind.CALL, Kind.APPLY) else None
    children = tuple(
        child if child is head else _named(child, state) for child in node.children
    )
    if node.kind is Kind.NAME:
        known = state.resolve(str(node.value))
        if known is not None and not known.is_function and known.canonical != node.value:
            return replace(node, value=known.canonical, children=children)
    return replace(node, children=children) if children != node.children else node
