"""Evaluate: the tree worked out innermost first, in an environment.

What Derive does before anything is rewritten. A name is replaced by what it
stands for, a call to a function the session defined is the body worked out over
the arguments, and the test of an `IF` is decided before either branch is looked
at, so that only the branch it chooses is ever worked out. The answer is a sympy
expression, which the rest of Simplify then writes in its sufficiently simple form.

The walk is the converter's: `to_sympy` is the arithmetic of every node kind, and
the evaluator is that conversion with an `Environment` threaded through it, taking
over the kinds where what a node means depends on where it stands. Those are
these, and `_Evaluator.convert` is where each is dispatched:

* A name: a parameter's argument, a bound variable standing for itself, or an
  assigned value, followed at most once along a branch so that `pn := pn + 1`
  simplifies to `pn + 1` rather than spinning.
* A label, followed the same way.
* `IF(test, a, b, c)`: the test decided, and only the branch it picks evaluated.
  An undecidable test takes the fourth argument where there is one and is
  otherwise suspended - `Suspended` is what that is.
* A call of a defined function: the arguments evaluated, the leading parameters
  bound to them (partial application leaves the rest standing as names), and the
  body evaluated. A recursion is this, call inside call, until the test that ends
  it decides; `_NESTING` is how deep it may go before the call that has not
  arrived is left standing as the answer.
* `F'(a)` of a defined `F`: the body differentiated in its first parameter and
  only then taken at `a`.
* A function that binds a variable - `SUM`, `INT`, `VECTOR` and the rest of
  `BINDING_FUNCTIONS` - with its variable standing for itself in what it binds.
  Over a range of known values the variable takes each of them in turn instead:
  a counted `SUM` or `PRODUCT`, a `VECTOR`, a `SELECT` and an `ITERATE` are
  their body evaluated once for every value, which is what decides a test in
  the index and enters a recursion on it.
* A calculus head, and `APPROX`: worked out where it is made, its operands
  being values already, by what `Environment.work_out` supplies.
* `FACTOR`, `EXPAND`, `SOLVE` and `RATE`: run on their evaluated operands by
  `Environment.command`, so that a call over one is an ordinary call.
* `SUBS(u, [x], [a])`: the substitution made, unless what it binds is a
  derivative, which is held as sympy's `Subs`.
* An assignment, which keeps its target, and a definition, whose parameters
  stand for themselves in its body.

Everything else is the converter's own.
"""

from __future__ import annotations

import itertools
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

import sympy as sp
from sympy.core.function import AppliedUndef, UndefinedFunction

from rederive.engine.context import Context
from rederive.engine.intervals import decided
from rederive.engine.to_sympy import (
    CALCULUS_HEADS,
    COMMAND_HEADS,
    Assign,
    Converter,
    FunDef,
    InertVector,
    as_condition,
    chosen_from,
    is_conditional,
    iterated,
    steps,
    vector_of,
)
from rederive.model.expr import Kind, Node
from rederive.syntax import ParseState
from rederive.syntax.names import BINDING_FUNCTIONS, BINDING_THROUGHOUT, PRIME_MARK

__all__ = [
    "Commanded",
    "Environment",
    "Suspended",
    "evaluate",
    "evaluated",
    "named_as_declared",
    "released",
]

Decider = Callable[[sp.Basic], bool | None]
WorkOut = Callable[[sp.Basic], sp.Basic]
Command = Callable[[str, list], sp.Basic]


@dataclass(frozen=True)
class Environment:
    """What a name means at one point of the evaluation, and how a test is decided.

    `bindings` are the values a call's parameters stand for, and the values an
    assignment in a test gave its name for the branches after it. `bound` are the
    names that stand for themselves: a definition's own parameters, and the
    variable a sum, an integral or a generated vector ranges over. `expanding` and
    `labels` are the assigned names and the labels being followed along this
    branch, which is what stops one that reaches itself. `depth` is how many calls
    of defined functions deep the evaluation stands.

    `written` is what a suspended conditional was resumed with: the values its
    variables have taken since it was set aside, written in wherever one of those
    variables is met.

    `decide` answers the test of an `IF`: True, False, or None where nothing
    settles it. Simplify supplies its own, which reads a test with the rewrites
    and the declared domains; anything else gets `_plainly_decided`.

    `work_out` is what a calculus head is worth, and `command` runs a `FACTOR`,
    `EXPAND`, `SOLVE` or `RATE` over its evaluated arguments. Simplify supplies
    both. Without them those heads stand as they were made, which is what a
    caller that only wants the variables of an expression needs.

    `writing` is the evaluation that only writes names in, which is what an
    undecidable conditional is shown as: no call is entered and no test decided,
    so the branches stay as they were written.
    """

    context: Context
    decide: Decider | None = None
    work_out: WorkOut | None = None
    command: Command | None = None
    bindings: Mapping[str, sp.Basic] = field(default_factory=dict)
    bound: frozenset[str] = frozenset()
    expanding: frozenset[str] = frozenset()
    labels: frozenset[int] = frozenset()
    depth: int = 0
    written: Mapping[sp.Basic, sp.Basic] = field(default_factory=dict)
    writing: bool = False
    suspended: dict[sp.Basic, type] = field(default_factory=dict, compare=False)

    def decides(self, test: sp.Basic) -> bool | None:
        """Whether `test` holds. A test that raises is a test nothing settles."""
        try:
            if self.decide is not None:
                return self.decide(test)
            return _plainly_decided(test, self.context)
        except Exception:
            return None


def evaluate(node: Node, env: Environment) -> sp.Basic:
    """`node` worked out in `env`. Undecidable conditionals in it stay suspended."""
    with _deep():
        return _Evaluator(env).convert(node)


def evaluated(node: Node, context: Context, decide: Decider | None = None) -> sp.Basic:
    """`node` worked out with nothing bound, every conditional shown as it stands.

    What a caller that wants a value and no more uses: the variables an expression
    holds, and what a plot draws.
    """
    return released(evaluate(node, Environment(context, decide)))


def _plainly_decided(test: sp.Basic, context: Context) -> bool | None:
    """A test read as sympy reads it, and then put to the declared intervals."""
    condition = as_condition(test)
    if condition is sp.true:
        return True
    if condition is sp.false:
        return False
    return decided(condition, context)


#: How many calls of defined functions deep an evaluation may stand. A recursion
#: Derive's way counts down one at a time and arrives long before this - the
#: manual's own `RAISE(b, 256)` stands two hundred and fifty-seven deep at the
#: bottom - and one that does not arrive, `F(n) := DIF(F(n + 1), x)` or `FACT(-1)`,
#: answers with what it has built when it gets here: the call that is left over,
#: inside everything worked out on the way down. That is a true rewriting of what
#: was asked, and one that shows why it will not finish. Derive itself exhausts
#: memory there and answers nothing at all.
_NESTING = 300

#: The Python recursion limit while evaluating. Each level of a recursion is a
#: few dozen frames of walk, conversion and sympy construction, and the default
#: thousand would end a recursion well before `_NESTING` does.
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
    """The converter with an environment: the walk is `convert`, kind by kind."""

    def __init__(self, env: Environment, performed: dict[str, sp.Basic] | None = None):
        super().__init__(env.context)
        self.env = env
        #: The names a test being evaluated has assigned, and their values. None
        #: outside a test, where an assignment is a definition and not an act.
        self.performed = performed

    def within(self, env: Environment) -> _Evaluator:
        return _Evaluator(env, self.performed)

    def convert(self, node: Node) -> sp.Basic:
        match node.kind:
            case Kind.CALL | Kind.APPLY:
                return self._call(node)
            case Kind.DOMAIN:
                # A declaration describes a variable, and its bounds are read
                # where it was written, not in whatever scope it stands in.
                return Converter(self.context).convert(node)
        return super().convert(node)

    # -- names --------------------------------------------------------------

    def _name(self, node: Node) -> sp.Basic:
        name = str(node.value)
        env = self.env
        value = env.bindings.get(name)
        if value is not None:
            return value
        if name not in env.bound and name not in env.expanding:
            assigned = env.context.assignments.get(name)
            if assigned is not None:
                deeper = replace(env, bindings={}, expanding=env.expanding | {name})
                return self.within(deeper).convert(assigned)
        return self._written(super()._name(node))

    def _written(self, symbol: sp.Basic) -> sp.Basic:
        return self.env.written.get(symbol, symbol)

    def _subscript(self, base_node: Node, base: sp.Basic, index: sp.Basic) -> sp.Basic:
        return self._written(super()._subscript(base_node, base, index))

    def operand(self, value: sp.Basic) -> sp.Basic:
        return _value_of(value)

    def _label(self, node: Node) -> sp.Basic:
        """`#3`, or itself where nobody knows it or it reaches itself."""
        try:
            number = int(str(node.value))
        except ValueError:
            number = -1
        target = self.context.labels.get(number)
        if target is None or number in self.env.labels:
            return sp.Symbol(f"#{node.value}")
        deeper = replace(self.env, bindings={}, labels=self.env.labels | {number})
        return self.within(deeper).convert(target)

    def _vector(self, node: Node) -> sp.Basic:
        """A vector, whose elements may be vectors that a name stands for.

        `[w]` is a matrix of one row where `w` is a vector, exactly as `[[2, 3,
        4]]` is: a name is replaced by its value, and a value that is a vector is
        a row like one written out. VECTOR.MTH's `OUTER(v, w) := VECTOR([v SUB
        n_], n_, DIMENSION(v)) . [w]` multiplies by that row.
        """
        if not node.children:
            return sp.Matrix(0, 0, [])
        elements = self._children(node)
        if any(
            child.kind in (Kind.NAME, Kind.LABEL)
            and isinstance(element, (sp.MatrixBase, InertVector))
            for child, element in zip(node.children, elements, strict=True)
        ):
            return vector_of(elements)
        return self.vector_over(node.children, elements)

    # -- definitions --------------------------------------------------------

    def _assign(self, node: Node) -> sp.Basic:
        """An assignment: its value is evaluated, its target is not.

        Inside the test of a conditional it is also carried out, which is
        Derive's way of computing something once and reading it back where the
        answer belongs. ODE1.MTH's `DSOLVE1` is built out of them: `IF(
        "inapplicable" = a_ := SEP(...), ..., a_, a_)` tries a method, asks
        whether it declined, and answers with what it found. The value stands
        where the assignment did, and `_conditional` binds the name to it in the
        branches.
        """
        target, *values = node.children
        written = [self.convert(child) for child in values]
        if self.performed is not None and written and target.kind is Kind.NAME:
            self.performed[str(target.value)] = written[-1]
            return written[-1]
        plain = Converter(self.context)
        return Assign(plain.convert(target), sp.Symbol(str(node.value)), *written)

    def _fundef(self, node: Node) -> sp.Basic:
        """A definition: the body is evaluated, its parameters stand for themselves."""
        parameters, *body = node.children
        names = frozenset(str(child.value) for child in parameters.children)
        inner = replace(
            self.env,
            bindings={},
            bound=self.env.bound | names,
            written=_unwritten(self.env.written, names),
        )
        plain = Converter(self.context)
        written = [self.within(inner).convert(part) for part in body]
        return FunDef(
            sp.Symbol(str(node.value)),
            sp.Tuple(*(plain.convert(child) for child in parameters.children)),
            *written,
        )

    # -- calls --------------------------------------------------------------

    def _call(self, node: Node) -> sp.Basic:
        head, *arguments = node.children
        name = str(head.value)
        if name in BINDING_FUNCTIONS:
            return self._binding(name, arguments)
        if name == "IF" and 1 <= len(arguments) <= 4 and not self.env.writing:
            return self._conditional(node, arguments)
        values = [_value_of(self.convert(argument)) for argument in arguments]
        if name == "SUBS" and not self.env.writing:
            return self._substitution(values)
        derived = self._derived(name, values)
        if derived is not None:
            return derived
        definition = self.context.functions.get(name)
        if definition is not None:
            if self.env.writing:
                return self.opaque(name, values)
            return self._entered(name, definition, values)
        made = self.call(name, values)
        if name in COMMAND_HEADS and _is_call_of(made, name):
            if self.env.command is not None and not self.env.writing:
                return self.env.command(name, list(made.args))
        return self._worked(made, values)

    def _worked(self, made: sp.Basic, values: Sequence[sp.Basic]) -> sp.Basic:
        """`made` with every calculus head the call made in it worked out.

        The heads standing in the arguments were offered when those were made
        and stood, so they are not offered again: an integral nothing can do is
        tried once, wherever it ends up. What a head is worked out to may hold
        heads of its own - sympy answers an integral it cannot do whole with a
        sum of integrals - and each of those is worked out where it stands, by
        the same rule, and once.
        """
        work_out = self.env.work_out
        if work_out is None or self.env.writing:
            return made
        try:
            if not made.has(*CALCULUS_HEADS):
                return made
            standing = set()
            for value in values:
                if isinstance(value, sp.Basic | sp.MatrixBase):
                    standing |= value.atoms(*CALCULUS_HEADS)
        except Exception:
            return made
        return _worked_out(made, standing, work_out)

    def _entered(
        self, name: str, definition: tuple[tuple[str, ...], Node], values: list
    ) -> sp.Basic:
        """A call of a defined function: its body over its arguments.

        Partial application binds the leading parameters only: `ACCELERATION(f,
        m) := f/m` applied to 6 alone is `6/m`.
        """
        if self.env.depth >= _NESTING:
            return self.opaque(name, values)
        parameters, body = definition
        inner = replace(
            self.env,
            bindings=dict(zip(parameters, values, strict=False)),
            bound=frozenset(),
            depth=self.env.depth + 1,
        )
        return _Evaluator(inner).convert(body)

    def _derived(self, name: str, values: list) -> sp.Basic | None:
        """`F'(a)` where `F` is a definition: its body, derived and taken at `a`.

        A prime is a derivative of the function and not of the call, so the body
        is differentiated in its own first parameter and only then is the
        argument written in - `F(r) := r^2` makes `F'(r^2)` twice `r^2` and not
        twice `r`. The point may mention the very name being differentiated over,
        so it arrives as the point of a limit, which is the original's own
        reading: what it prints for a derivative it cannot take as a whole is
        `lim d/d@ F(@)` at the point.

        None where there are no marks, or nothing defined under them. A function
        nobody defined has no body to differentiate and stays the call it is, for
        the converter to read as a derivative of a function of itself.
        """
        base = name.rstrip(PRIME_MARK)
        order = len(name) - len(base)
        definition = self.context.functions.get(base)
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
        """`SUBS(u, [x, ...], [a, ...])`: the substitution, made where it can be.

        A `Subs` binding nothing but an ordinary expression is a substitution
        waiting rather than a value: `SUBS(BB(u), [u], [r^2])` says `BB` at `r^2`
        and comes to `BB(r^2)`. Only one holding a derivative over what it binds
        stands, since that derivative is unevaluated on purpose - the slope of
        `F` at `y` is not the slope of a constant, which is what it would look
        like once the variable it is taken over had been bound.
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
        """`SUM(u, k, m, n)`: `k` stands for itself in `u`, and nowhere else.

        An assigned name is left as written - substituting `x := 5` into the
        index would leave the call with no variable to range over - and the
        arguments after it are read in the enclosing scope, because a limit is
        not part of what it bounds.

        A parameter is the other case, and there the name a call supplies is the
        variable summed over: `LIN1_DIFFERENCE(p, q, x, x0, y0)` sums over `x`
        and is called with `m` for it, so what it sums over is `m`. The body is
        evaluated with that parameter still bound, which lets the index catch
        what another parameter wrote in - `PRODUCT(p, x, x0, x - 1)` called with
        `m` for both `p` and `x` is the product of `m` over `m`, and `(m - 1)!`
        is what Derive answers. A definition is textual substitution, and
        capture is part of it.

        A vector of names binds every one of them, which is what an iterated
        limit and a system's iteration are written with: `LIM(u, [x, y], [a,
        b])` and `ITERATE([k, j + k], [j, k], [0, 1], n)`. A parameter supplied
        with such a vector is one of these too, and SOLVE.MTH turns on its being
        read as one - `NEWTON_AUX`'s `LIM(a, x, xk)` is called with `[x, y]` for
        `x`, and a parameter left standing there is a limit in a variable the
        caller never named.

        A second argument that is no name and no vector of names binds nothing.
        `SUM(v)` over a vector's elements has none either, and both are
        evaluated like any other call.

        `BINDING_THROUGHOUT` is the exception to where the binding stops. A root
        sum runs its summand over the roots of a polynomial written in the same
        variable, so the arguments after the name are inside the binding rather
        than outside it, and an assigned `t` reaches none of `ROOT_SUM(p, t, u)`.
        """
        if len(arguments) < 2 or _variables_of(arguments[1]) is None:
            values = [_value_of(self.convert(argument)) for argument in arguments]
            return self._worked(self.call(name, values), values)
        body, written_index, *rest = arguments
        index, names, shadowed = self._index(written_index)
        inner = replace(
            self.env,
            bindings={n: v for n, v in self.env.bindings.items() if n not in shadowed},
            bound=self.env.bound | names,
            written=_unwritten(self.env.written, names),
        )
        evaluator = self.within(inner)
        outer = evaluator if name in BINDING_THROUGHOUT else self
        limits = [_value_of(outer.convert(argument)) for argument in rest]
        if name in _COUNTED and not self.env.writing:
            counted = self._counted(name, body, index, inner, limits)
            if counted is not None:
                return counted
        values = [_value_of(evaluator.convert(body)), index, *limits]
        return self._worked(self.call(name, values), values)

    def _counted(
        self,
        name: str,
        body: Node,
        index: sp.Basic,
        inner: Environment,
        limits: list[sp.Basic],
    ) -> sp.Basic | None:
        """A binding call over values all known, as its body at each of them.

        The body is evaluated once a value, with the variable written in, and not
        converted once and substituted into: a test in the index decides then, a
        recursion on it is entered then, and an element of a vector it indexes
        is there to take. NUMBER.MTH's `DISTINCT_PARTS` sums a `PARTS` guarded by
        a test on the index, and VECTOR.MTH's `NONZERO_ROWS` sums a test over
        `m SUB n_`, which is a row to test only once `n_` is a number.

        None where the values are not all known - a symbolic end, or more of
        them than `_WRITTEN_OUT` - and the call is then made over the body as
        it stands, for sympy to sum or for the call to wait. None too where a
        `SELECT` meets a test nothing decides, or an iteration cannot go on.
        """

        def at(*values: sp.Basic) -> sp.Basic:
            variables = list(index) if isinstance(index, sp.MatrixBase) else [index]
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
                return _total(name, limits, at)
            if name == "VECTOR":
                return vector_of([at(value) for value in steps(limits)])
            return chosen_from(steps(limits), lambda value: _value_of(at(value)))
        except Exception:
            return None

    def _index(self, index: Node) -> tuple[sp.Basic, frozenset[str], frozenset[str]]:
        """The variable a binding call ranges over, the names it binds, and the
        parameters it hides.

        A parameter in this position is written in, because the name a call
        supplies is the name the body ranges over. What it is written in with has
        to name variables itself, and where it does not - a parameter supplied
        with a number, or with an expression - the parameter is left standing
        and shadows the argument in the body, there being no other name to range
        over.
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
        return vector_of(values), names, shadowed

    # -- IF -----------------------------------------------------------------

    def _conditional(self, node: Node, arguments: list[Node]) -> sp.Basic:
        """`IF(test, a, b, c)`: the branch the test decides, and only that branch.

        A true test leaves the then clause, a false one the else clause - `?`
        when there is none - and an undecidable test takes the unknown clause if
        the author wrote one. A test and nothing else is that test as a number,
        one or zero, which is what makes `SUM(IF(PRIME(n)), n, 1, 100)` count the
        primes.

        With no unknown clause an undecidable `IF` is suspended, branches and
        all, which is Derive's behavior and a deliberate one: working out a
        branch that may never be taken can turn a guarded division by zero into
        an error, and a recursion's guard is exactly such a test.
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
        """An undecidable conditional, set aside as a function of its variables."""
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
    """A conditional whose test nothing could decide yet, as a function of its variables.

    An undecidable `IF` is Derive's to show as it was written, branches and all,
    so nothing that comes after the evaluation may look inside it - and nothing
    may mistake it for a constant either. A function of exactly the variables it
    mentions is both: `SUM(IF(k > 0, k, 0), k, 1, n)` stays a sum, where a bare
    name would let sympy answer `n*IF(k > 0, k, 0)`, and `DIF(IF(x > 0, x^2, 0),
    x)` stays a derivative.

    It is a function in earnest. Writing values in for its variables - a sum
    written out term by term, an iteration taking a step - resumes the
    evaluation the test was suspended in, with those values in place, and a test
    that decides now chooses its branch then. NUMBER.MTH's `DISTINCT_PARTS` is a
    sum whose terms each guard a `PARTS` behind a test in the index, and no test
    decides until the sum has written the index in.

    Unknown to sympy, as a function nobody defined is: a limit of one is not
    taken by writing the point in, since nothing says it is continuous there.

    Each one is a class of its own, made by `_Evaluator._suspended`, which
    carries the node, the environment it was met in, the `display` it is shown
    as - the conditional with names written in and nothing evaluated - and the
    `variables` it is applied to. `released` shows what is left of them at the
    end.
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
    """Whether a value written in for a variable is one to resume the evaluation at.

    A finite number or a truth value: what a sum writing out its terms, an
    iteration taking its steps or a truth table filling its rows writes in. Sympy
    also writes things in for its own purposes - a limit puts in an infinity or the
    reciprocal of a variable of its own to look at the far end - and a test asked
    there is a question nobody put. Those leave the conditional standing, with
    the value written into its display.
    """
    if isinstance(value, sp.logic.boolalg.BooleanAtom):
        return True
    return bool(getattr(value, "is_number", False)) and value.is_finite is not False


#: What tells one suspended conditional's class from another's.
_SUSPENSIONS = itertools.count()


def _resumed(cls: type[Suspended], written: dict[sp.Basic, sp.Basic]) -> sp.Basic:
    """The suspended conditional evaluated again, its variables taking these values."""
    with _deep():
        return _Evaluator(_at(cls.environment, written)).convert(cls.node)


def _at(env: Environment, written: Mapping[sp.Basic, sp.Basic]) -> Environment:
    """`env` with these values written in for these variables.

    Wherever the variables are met, and in what the parameters were bound to as
    well, since a definition is textual substitution: `PRODUCT(p, x, x0, x -
    1)` called with `m` for both `p` and `x` multiplies `m` at every value `m`
    takes.
    """
    return replace(
        env,
        bindings={name: _over(value, written) for name, value in env.bindings.items()},
        written={
            **{symbol: _over(value, written) for symbol, value in env.written.items()},
            **written,
        },
    )


def _over(value: sp.Basic, written: Mapping[sp.Basic, sp.Basic]) -> sp.Basic:
    """`value` with what was written in, wherever it can be."""
    try:
        if not any(symbol in value.free_symbols for symbol in written):
            return value
        return value.subs(written, simultaneous=True)
    except Exception:
        return value


#: The binding calls whose body is evaluated once for each value of a range,
#: where the range is known.
_COUNTED = frozenset({"SUM", "PRODUCT", "VECTOR", "SELECT", "ITERATE", "ITERATES"})

#: How many terms a range is written out as. A range longer than this is one
#: whose terms nobody wants to read, and sympy sums it as it was written.
#: Derive's own utility files reach four.
_WRITTEN_OUT = 1000


def _total(
    name: str, limits: list[sp.Basic], at: Callable[[sp.Basic], sp.Basic]
) -> sp.Basic | None:
    """A counted `SUM` or `PRODUCT`: its terms at each value, added or multiplied.

    The values are a vector of them, or a range. None where they are neither,
    or where a term is no expression to add, which leaves the sum to sympy. A
    range that runs backwards is the negated sum, or the reciprocal product, of
    the values it steps back over.
    """
    backwards = False
    if len(limits) == 1 and isinstance(limits[0], sp.MatrixBase):
        values = steps(limits)
    elif len(limits) == 2:
        run = _run_of(*limits)
        if run is None:
            return None
        values, backwards = run
    else:
        return None
    terms = [at(value) for value in values]
    if not all(isinstance(term, sp.Expr) for term in terms):
        return None
    if name == "SUM":
        total = sp.Add(*terms)
        return -total if backwards else total
    product = sp.Mul(*terms)
    return 1 / product if backwards else product


def _run_of(low: sp.Basic, high: sp.Basic) -> tuple[list[sp.Basic], bool] | None:
    """The values an index runs through from `low` to `high`, and whether backwards.

    The index starts at the low end and steps by one for as long as it does not
    pass the high one, which is Derive's reading of a range and is what makes an
    end between two terms mean anything. `PARTS` bounds Rademacher's series by
    `SQRT(n)/LOG(n, 11)`, which for `n` of 4 is 3.459: the index runs 1, 2 and 3,
    where sympy would read the bound as the `n` of a closed form and put it into
    Faulhaber's polynomial.

    A high end below the low one counts as many steps back from the value before
    the low end, the convention that keeps `SUM(u, k, a, b) + SUM(u, k, b + 1, c)`
    the sum from `a` to `c` whatever order the three come in: `SUM(k, k, 5, 3.5)`
    is -4, the 4 stepped back over negated, and `SUM(k, k, 5.5, 3)` is -8, from 4.5
    and 3.5.

    Both ends have to be numbers - a symbolic one is a range of unknown length -
    and there have to be few enough values to be worth writing down.
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
    """Every calculus head in `made` but those `standing` worked out, innermost first.

    A head comes up once however often it stands, and what it is worked out to
    is searched for heads of its own the same way; a head an answer hands back
    that is the one being worked out stands where it is.
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
    """What a `FACTOR`, `EXPAND` or `SOLVE` head answered, in the form the command chose.

    A factorization is a form and not only a value: `FACTOR(12)` is `2^2*3`, and
    the normal form Simplify writes a value in would multiply it straight back
    out. A solution is written the way soLve writes it, `x >= (3*y - 7)/2`, and
    not the way Simplify would. So the answer is carried in this wrapper through
    everything that comes after the evaluation, and only shown as itself at the
    very end. A call that takes the answer as an argument takes the value and
    not the wrapper: `RHS(SOLVE(x^2 - 5*x + 6 = 0, x))` is `[2, 3]`.
    """

    nargs = 1


def _is_call_of(value: sp.Basic, name: str) -> bool:
    return isinstance(value, AppliedUndef) and type(value).__name__ == name


def _value_of(value: sp.Basic) -> sp.Basic:
    """`value`, out of the `Commanded` holding it if one does."""
    return value.args[0] if isinstance(value, Commanded) else value


def released(value: sp.Basic) -> sp.Basic:
    """`value` with every suspended conditional shown as the conditional it is.

    What is shown is the conditional as it was met, with what its variables have
    become since written in, so that an untouched one is exactly what the author
    typed and the printer can write it that way. A command's answer is shown as
    the command wrote it.
    """
    try:
        if not value.atoms(Suspended, Commanded):
            return value
        return value.replace(
            lambda found: isinstance(found, Suspended | Commanded), _shown
        )
    except Exception:
        return value


def _shown(found: Suspended | Commanded) -> sp.Basic:
    if isinstance(found, Commanded):
        return released(found.args[0])
    cls = type(found)
    shown = cls.display
    if found.args != cls.variables:
        shown = _over(shown, dict(zip(cls.variables, found.args, strict=True)))
    return released(shown)


def _variables_of(index: Node) -> tuple[str, ...] | None:
    """The variables `index` names, or None where it names none.

    One name, or a vector of them. An empty vector and a vector holding anything
    but names are no binding position at all.
    """
    if index.kind is Kind.NAME:
        return (str(index.value),)
    if index.kind is not Kind.VECTOR or not index.children:
        return None
    if any(child.kind is not Kind.NAME for child in index.children):
        return None
    return tuple(str(child.value) for child in index.children)


def _unwritten(
    written: Mapping[sp.Basic, sp.Basic], names: frozenset[str]
) -> Mapping[sp.Basic, sp.Basic]:
    """What is written in, less the variables a binding takes over.

    A sum over `k` inside a conditional that was resumed at `k = 2` ranges over a
    `k` of its own, and the value is no business of what it binds.
    """
    if not any(getattr(symbol, "name", None) in names for symbol in written):
        return written
    return {
        symbol: value
        for symbol, value in written.items()
        if getattr(symbol, "name", None) not in names
    }


def _names_in(value: sp.Basic) -> frozenset[str] | None:
    """The variables a supplied argument names, or None where it names none."""
    if type(value) is sp.Symbol:
        return frozenset({value.name})
    elements: list = []
    if isinstance(value, sp.MatrixBase) and value.rows == 1 and value.cols:
        elements = list(value)
    elif isinstance(value, InertVector) and value.args:
        elements = list(value.args)
    if not elements or any(type(element) is not sp.Symbol for element in elements):
        return None
    return frozenset(element.name for element in elements)


# -- spelling -------------------------------------------------------------------


def named_as_declared(node: Node, state: ParseState | None) -> Node:
    """`node` with every name spelled the way the symbol table records it.

    Case-insensitively - which is the default - `x` and `X` are one variable,
    and which of the two a line carries depends on what had been declared when
    it was read. `X := ELEMENT(x, 1)` declares `X` and mentions `x` before that
    declaration exists, so the line comes back holding both spellings; the
    engine has to see one name for one variable, or the answer holds two and
    reading it back turns them into one.

    Without a symbol table nothing can be resolved and the tree stands as it
    is, which is what a caller with no session of its own gets.
    """
    if state is None:
        return node
    return _named(node, state)


def _named(node: Node, state: ParseState) -> Node:
    """The walk. A call's head is a name of another kind and is left alone.

    What the table records for a variable is the spelling to write it under,
    and for anything else it is the spelling of something else entirely: the
    setting value `Exact` for `EXACT(...)`, the letter `δ` for `DELTA(...)`.
    Neither reads back as the call it was written as.
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
