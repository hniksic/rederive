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
* A call of a defined function: arguments are evaluated and bound to the parameters,
  an omitted one taking its default or else standing for the variable of its name,
  and the body evaluated. Recursion nests calls until the terminating test decides;
  past `_NESTING` levels the pending call is left in the answer.
* `F'(a)` of a defined `F`: the body is differentiated in its first parameter, then
  evaluated at `a`.
* A variable-binding function (`SUM`, `INT`, `VECTOR` and the rest of
  `BINDING_FUNCTIONS`), with its variable standing for itself. Over a range of known
  values, a counted `SUM` or `PRODUCT`, `VECTOR`, `SELECT` and `ITERATE` instead
  evaluate the body once per value, which decides tests on the index and enters
  recursions on it; `MAP_LIST`, `MAP`, `EVERY` and `SOME` always do.
* Calculus heads and `APPROX`: computed on the spot via `Environment.work_out`, since
  their operands are already values.
* `FACTOR`, `EXPAND`, `SOLVE` and `RATE`: run on their evaluated operands via
  `Environment.command`, so they behave like ordinary calls.
* `SUBS(u, [x], [a])`: the substitution is made, unless it binds into a derivative,
  which is kept as sympy's `Subs`.
* An assignment keeps its target unevaluated, and a definition's parameters stand for
  themselves in its body. Where a command runs programs, and inside a called
  function's body, an assignment or definition is also made (see
  `_Evaluator.performs`).
* `PROG`, `LOOP`, `RETURN` and `ASSIGN` are run rather than converted, and `exit` ends
  what it is in (see `_Evaluator.statement`).

Programs run in activations. Each call of a defined function gets a `Frame` holding its
parameters, its locals; an assignment to any other name changes the global, kept for
the rest of the command in the context's `Store` and handed back to the session with
the answer. `RETURN` and `exit` are the exceptions `_Return` and `_Exit`, caught by the
construct or activation they end. A setting assigned takes effect for the statements
after it (see `Store.context`), and each statement, pass of an iteration and call counts
against the command's step limit (see `Context.step_limit`). Locals and globals are
kept in the same kind of `Cell`.

Everything else is handled by the converter.
"""

from __future__ import annotations

import contextvars
import copy
import dataclasses
import itertools
import math
import operator
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cached_property
from types import MappingProxyType
from typing import Any

import sympy as sp
from sympy.core.function import AppliedUndef, UndefinedFunction
from sympy.core.mul import _keep_coeff
from sympy.core.relational import Relational
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.logic.boolalg import BooleanAtom

from rederive import library
from rederive.engine import budget
from rederive.engine.approximation import approximated, carried, carried_out
from rederive.engine.budget import within
from rederive.engine.context import (
    CLEARED,
    CLEARED_SETTING,
    Cell,
    Context,
    Definition,
    Domain,
    DomainKind,
    FileNeeded,
    Global,
    PassedOverTooFar,
    Precision,
    SettingValue,
    StepLimitExceeded,
    library_definition,
)
from rederive.engine.fresh import forgotten
from rederive.engine.from_sympy import (
    deep_recursion,
    from_sympy,
    setting_value,
    written_tree,
)
from rederive.engine.intervals import decided
from rederive.engine.native import (
    NUMBERS,
    SCALARS,
    Assigned,
    NativeAssignment,
    NativeVector,
    assigned,
    given,
    native_of,
    rational,
    sympy_of,
)
from rederive.engine.printer import HeldBase, author_text, has_held
from rederive.engine.to_sympy import (
    CALCULUS_HEADS,
    COMMAND_HEADS,
    COMPARING_TRUTHS,
    CONSTANTS,
    DECIDING,
    DECOMPOSING,
    LOGICAL_WORDS,
    ON_FORM,
    SIMPLE,
    Assign,
    ColumnsOf,
    Converter,
    Deeper,
    FunDef,
    Logical,
    MovingLimit,
    PlusMinus,
    RELATIONS,
    RowsOf,
    STATEMENT_KINDS,
    StringLiteral,
    Subscript,
    Vector,
    absorbed_by_infinity,
    added,
    as_condition,
    chosen_from,
    collection_of,
    connected,
    digit_count,
    divided,
    formals,
    held,
    held_value,
    is_conditional,
    is_plain,
    is_simple,
    iterated,
    multiplied,
    negated,
    numeral,
    as_written,
    over_values,
    programmed,
    place_of,
    settled,
    sorted_by,
    steps,
    table_row,
    terms_by_expansion,
    vector_from,
    with_worked_test,
)
from rederive.engine.sets import SetOf, is_set
from rederive.engine.vectors import mapped
from rederive.model.expr import (
    Kind,
    Node,
    depth,
    is_chain,
    parameter_names,
    parameters_of,
    variable_key,
)
from rederive.syntax import ParseState
from rederive.syntax.names import CONSTANTS as CONSTANT_NAMES
from rederive.syntax.names import (
    BINDING_FUNCTIONS,
    BINDING_THROUGHOUT,
    BUILTIN_FUNCTIONS,
    EXIT,
    PRIME_MARK,
    PROGRAM_FUNCTIONS,
    SETTINGS,
)

__all__ = [
    "Commanded",
    "Environment",
    "Suspended",
    "branched",
    "divisors",
    "evaluate",
    "evaluated",
    "exact_closed_form",
    "exact_number",
    "limit_by_substitution",
    "moving_integrals",
    "named_as_declared",
    "released",
    "released_held",
    "value_of",
]

Decider = Callable[[sp.Basic], bool | None]
WorkOut = Callable[[sp.Basic], sp.Basic]
Command = Callable[[str, list, list, Context], sp.Basic]
Simplifier = Callable[..., sp.Basic]

#: Built-ins that a definition's body keeps as calls where they act on its parameters:
#: they are evaluated when the function is called, on the argument values, so
#: `EX(u) := EXPAND(u)` expands what it is given.
_AT_CALL = frozenset({"EXPAND", "FACTOR", "APPROX"})

#: Built-ins whose answer depends on the form of their argument, which Derive
#: simplifies first: `VARIABLES((a^2*y^2 + a^2*y*z)/(y^2 + y*z))` is `[a]`, since the
#: argument cancels to `a^2`. ODE.MTH's `SYMMETRIC_TEST` relies on this, and its `FREE`
#: tests, which decide which method solves an equation, on the full simplification.
#: `TERMS` and `FACTORS` take apart their first argument as Simplify gives it, so the
#: terms of `x*(2*y + z) + x*(3*y + z)` are those of `x*(5*y + 2*z)`, one, and a
#: command's answer keeps the form it asked for, as Simplify keeps it:
#: `TERMS(EXPAND(x*(y + z)))` is `[x*y, x*z]`. `TERMS` with an amount or variables
#: expands its argument instead. The others read a command's answer simplified too:
#: ODE.MTH's `PARTS` takes the denominator of a factored sum as 1.
_SIMPLIFIED_ARGUMENTS = frozenset(
    {"VARIABLES", "NUMERATOR", "DENOMINATOR", "TERMS", "FACTORS"}
)

#: The solving functions, whose second argument names the variables to solve for.
_SOLVING = frozenset({"SOLVE", "SOLUTIONS", "NSOLVE", "NSOLUTIONS"})


class _Return(BaseException):
    """`RETURN u` ending the activation it is evaluated in, with the value of `u`.

    Like `_Exit`, this is a `BaseException`, so that it passes through the handlers of
    built-ins, which turn any `Exception` into a call left standing, on its way out of
    a binder body to the activation it ends.
    """

    def __init__(self, value: sp.Basic):
        super().__init__()
        self.value = value


class _Exit(BaseException):
    """`exit`, ending the innermost `LOOP` or `PROG` of the activation, or else its
    body; the value of what it ends is the last value `Frame.last`."""


@dataclass(eq=False)
class Frame:
    """One activation of a user function, or one top-level evaluation, as it runs.

    `formals` are the function's parameters, its locals; an assignment to any other
    name assigns the global. `locals` holds the cells of the locals that have been
    assigned (one that is only bound to its argument is in `Environment.bindings`); a
    cell holding `CLEARED` is a local without a value, which reads as the global of its
    name. `last` is the value of the statement completed last, which `exit` leaves a
    construct with.
    `loops` and `progs` count the `LOOP`s and `PROG`s running, so that `exit` knows
    what it ends. `activation` marks a function call, whose body `exit` ends if
    nothing else; at top level an `exit` outside every construct is just the name.
    `last` may be a native value, as a local's may (see `Cell`).
    `through` holds the values given to variables by assignments to the formals
    standing for them (see `_Evaluator._variable_of_formal`), which the arguments
    mentioning them see.
    """

    formals: frozenset[str] = frozenset()
    activation: bool = False
    locals: dict[str, Cell] = field(default_factory=dict)
    last: object = sp.nan
    loops: int = 0
    progs: int = 0
    through: dict[sp.Basic, sp.Basic] = field(default_factory=dict)

    def has(self, name: str) -> bool:
        """Whether `name` is a local of this activation with a cell of its own."""
        return name in self.locals


@dataclass(frozen=True)
class Environment:
    """What names mean at a point of the evaluation, and how tests are decided.

    `bindings` maps call parameters to their arguments. `supplied` records, for each
    call parameter, its argument and where it was evaluated, to evaluate it again once
    the body gives a value to a variable it mentions (see `Argument`). `frame` is the
    running activation (see `Frame`), whose assigned locals take precedence over
    `bindings`. `bound` holds names that stand for themselves: a definition's
    parameters, and the variable of a sum, integral or generated vector; `computing`
    those of them that a calculus head binds (a limit, derivative or integral), which
    stays a variable rather than taking values. `parameters`
    holds just the parameters of the definition being shown. `expanding` and `labels`
    are the assigned names and labels being followed on this branch, to stop
    self-reference. `depth` is the nesting depth of defined function calls.

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
    `simplify` simplifies the argument of a built-in in `_SIMPLIFIED_ARGUMENTS`, and
    each value a program stores; without it values are taken as evaluated.

    `values` holds the values of assigned variables worked out so far, shared by the
    whole evaluation (see `_assigned`).

    `top` is the assignment that is the whole expression being simplified, when its
    value is a top-level equation, which is stored with its sides simplified rather than
    decided: `e := 2 = 2` assigns `2 = 2`.

    `running` means programs run: a command is computing, so `PROG`, `LOOP`, `RETURN`,
    `exit`, updates and element assignments do what they say, and assignments are
    made. Otherwise (listing an expression's variables, plotting) they stand as
    written, and only the assignments in called bodies are made, for that evaluation.
    `writing` means only substitute names, used to display an undecidable
    conditional: no call is entered and no test decided, so branches stay as written.
    `defining` means a definition is being shown, so the calls its body enters perform
    no assignment. `in_call` means a call of a defined function is running, which stands
    if its program can't decide which way to go (see `_Evaluator._conditional`).
    """

    context: Context
    decide: Decider | None = None
    work_out: WorkOut | None = None
    command: Command | None = None
    simplify: Simplifier | None = None
    bindings: Mapping[str, sp.Basic] = field(default_factory=dict)
    supplied: Mapping[str, Argument] = field(default_factory=dict, compare=False)
    frame: Frame | None = field(default=None, compare=False)
    bound: frozenset[str] = frozenset()
    computing: frozenset[str] = frozenset()
    parameters: frozenset[str] = frozenset()
    expanding: frozenset[str] = frozenset()
    labels: frozenset[int] = frozenset()
    depth: int = 0
    written: Mapping[sp.Basic, sp.Basic] = field(default_factory=dict)
    running: bool = False
    writing: bool = False
    defining: bool = False
    suspended: dict[sp.Basic, type] = field(default_factory=dict, compare=False)
    values: dict[tuple, sp.Basic] = field(default_factory=dict, compare=False)
    top: Node | None = field(default=None, compare=False)
    in_call: bool = False

    def decides(self, test: sp.Basic) -> bool | None:
        """Whether `test` holds, or None if undecided (including if deciding raises)."""
        try:
            if self.decide is not None:
                return self.decide(test)
            return _plainly_decided(test, self.context)
        except Exception:
            return None

    @property
    def runs(self) -> bool:
        """Whether a program construct here runs (see `running`)."""
        return self.running and not self.writing and not self.defining


def replace(value: Any, /, **changes: Any) -> Any:
    """`dataclasses.replace`, made directly for an `Environment`, which has no
    validation or derived fields: evaluation makes one per call, element and step."""
    if type(value) is Environment and changes.keys() <= _ENVIRONMENT_FIELDS:
        made = object.__new__(Environment)
        state = made.__dict__
        state.update(value.__dict__)
        state.update(changes)
        return made
    return dataclasses.replace(value, **changes)


#: The fields of an `Environment`.
_ENVIRONMENT_FIELDS = frozenset(each.name for each in dataclasses.fields(Environment))


@dataclass(frozen=True, eq=False)
class Argument:
    """A call's argument: its value, and the environment it was evaluated in.

    As in Derive, an argument is evaluated once, before the body, and the parameter
    stands for that value. Where the body gives values to variables the value mentions,
    they are substituted into it and it is evaluated again, so that what couldn't be
    evaluated with the variable symbolic now is. `T(u, v, m) := ITERATE([u], v, [1],
    m)` called with `w SUB 1 + 1` for `u` and `w` for `v` adds one to the first element
    at each step: the value keeps `w SUB 1`, which selects an element once `w` is a
    vector. Likewise `ELEMENT([a, b, c], k)` substituted for the index of a `VECTOR`
    selects an element at each `k`, and `DIMENSION(x)` counts what `x` is given. The
    argument as written is never evaluated again: `Q(z, x) := ITERATES(x + z SUB 2, x,
    1, 1)` called with `ITERATES(DIF(w, x), w, x^2, 1)` for `z` iterates over `[x^2,
    2*x]`, not over a derivative taken with `x` a number.
    """

    env: Environment
    value: sp.Basic

    @cached_property
    def names(self) -> frozenset[str]:
        return _symbol_names(self.value)

    def rewritten(self, written: Mapping[sp.Basic, sp.Basic]) -> dict | None:
        """The values `written` gives the variables of the value since it was
        evaluated, or None if it gives none.
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

    def at(self, written: Mapping[sp.Basic, sp.Basic], writing: bool) -> sp.Basic:
        """The value with `written` substituted, evaluated again.

        Its variables stand for themselves, as they did when it was evaluated. A value
        that doesn't read back as an expression is substituted into as it stands, and
        so is one with a held value in it, which reading back would release.
        """
        context = self.env.context
        if context.store.held and has_held(self.value):
            return _over(self.value, written)
        try:
            skeleton, held = _held_apart(released(self.value), written)
            node = from_sympy(skeleton, context).exact
        except Exception:
            node = None
        if node is None or node.kind is Kind.STRING:
            return _over(self.value, written)
        plain = replace(
            self.env,
            bindings={},
            supplied={},
            bound=self.env.bound | self.names | frozenset(held),
            writing=writing,
        )
        value = _value_of(_Evaluator(_at(plain, written)).convert(node))
        return _put_back(value, held)


def _held_apart(
    value: sp.Basic, written: Mapping[sp.Basic, sp.Basic]
) -> tuple[sp.Basic, dict[str, sp.Basic]]:
    """`value` with each plain part that `written` doesn't reach replaced by a name, and
    the parts by name.

    `Argument.at` evaluates its value again by writing it out and reading it back, and
    a part free of the substituted variables evaluates to itself. Held apart, it isn't
    written out: TENSOR.MTH's `TF_T` substitutes each index into an element of a 3x3
    matrix of long trigonometric entries, which would otherwise be written and read
    three times per index. Only parts of sums, products, powers, vectors and element
    accesses are held, and only parts built from numbers, variables and elementary
    functions, so no head that binds or tests a variable is reached into, and no call
    that may evaluate differently is hidden.
    """
    variables = set(written)
    taken = {getattr(symbol, "name", None) for symbol in value.free_symbols} | {
        getattr(symbol, "name", None) for symbol in variables
    }
    held: dict[str, sp.Basic] = {}
    names: dict[sp.Basic, sp.Symbol] = {}

    def plain(part: sp.Basic) -> bool:
        return all(
            node.is_Atom
            or isinstance(node, (sp.Add, sp.Mul, sp.Pow))
            or (
                isinstance(node, sp.Function)
                and not isinstance(node, AppliedUndef)
                and type(node).__module__.startswith("sympy.functions.elementary")
            )
            for node in sp.preorder_traversal(part)
        )

    def hold(part: sp.Basic) -> sp.Basic:
        name = names.get(part)
        if name is None:
            index = len(names) + 1
            while f"hELd{index}_" in taken:
                index += 1
            name = names[part] = sp.Symbol(f"hELd{index}_")
            held[name.name] = part
        return name

    def walk(part: sp.Basic) -> sp.Basic:
        if part.is_Atom:
            return part
        if (
            isinstance(part, sp.Expr)
            and part.is_commutative
            and not part.free_symbols & variables
            and plain(part)
        ):
            return hold(part)
        if isinstance(part, (Vector, sp.Add, sp.Mul, sp.Pow, Subscript)) or _is_call_of(
            part, "ELEMENT"
        ):
            arguments = [walk(argument) for argument in part.args]
            if all(new is old for new, old in zip(arguments, part.args)):
                return part
            return part.func(*arguments)
        return part

    if not isinstance(value, sp.Basic) or _size(value) < _HELD_APART:
        return value, {}
    skeleton = walk(value)
    return (value, {}) if not held else (skeleton, held)


#: Below this many nodes a value is written out whole: holding its parts apart costs
#: more than it saves.
_HELD_APART = 40


def _size(value: sp.Basic) -> int:
    return sum(1 for _ in sp.preorder_traversal(value))


def _put_back(value: sp.Basic, held: Mapping[str, sp.Basic]) -> sp.Basic:
    """`value` with the parts `_held_apart` named put back."""
    if not held:
        return value
    try:
        back = {
            symbol: held[symbol.name]
            for symbol in value.free_symbols
            if getattr(symbol, "name", None) in held
        }
        return settled(value.xreplace(back)) if back else value
    except Exception:
        return value


def _with_elements(value: sp.Basic, written: Mapping[sp.Basic, sp.Basic]) -> sp.Basic:
    """`value` with the values `written` for its variables, a vector's elements for the
    subscripted variable `m SUB k` and for `ELEMENT(m, k)` (k a number) where the
    variable `m` takes a vector."""
    replaced: dict[sp.Basic, sp.Basic] = dict(written)
    for call in value.atoms(sp.Function):
        if type(call).__name__ != "ELEMENT" or len(call.args) != 2:
            continue
        vector, place = written.get(call.args[0]), call.args[1]
        if isinstance(vector, Vector) and place.is_Integer and place > 0:
            replaced[call] = _element_or_undefined(vector, int(place))
    for symbol in value.free_symbols:
        base, _, index = getattr(symbol, "name", "").partition(" SUB ")
        if not index.isdigit() or int(index) < 1:
            continue
        for variable, vector in written.items():
            if isinstance(vector, Vector) and getattr(variable, "name", None) == base:
                replaced[symbol] = _element_or_undefined(vector, int(index))
    return value.xreplace(replaced)


def _fills(value: sp.Basic, filled: sp.Basic) -> bool:
    """Whether `filled` is `value` with only its undefined parts given values: equal
    wherever `value` is defined, element by element in vectors."""
    if not _undefined_in(value):
        return value == filled
    if isinstance(value, Vector):
        return (
            isinstance(filled, Vector)
            and len(value.args) == len(filled.args)
            and all(_fills(one, other) for one, other in zip(value.args, filled.args))
        )
    return True


def _element_or_undefined(vector: Vector, place: int) -> sp.Basic:
    return vector.args[place - 1] if place <= len(vector.args) else sp.nan


def _undefined_in(value: sp.Basic) -> bool:
    """Whether `value` is undefined or holds an undefined part."""
    try:
        return isinstance(value, sp.Basic) and value.has(sp.nan, sp.zoo)
    except Exception:
        return False


def _assigned_names(node: Node) -> set[str]:
    """The names the assignments in `node` assign, wherever they are in it."""
    found = set()
    if node.kind is Kind.ASSIGN and node.children:
        target = node.children[0]
        if target.kind in (Kind.NAME, Kind.STRING):
            found.add(variable_key(target))
    for child in node.children:
        found |= _assigned_names(child)
    return found


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
    """`node` evaluated in `env`, leaving undecidable conditionals suspended.

    Where programs run, this is a top-level evaluation with a frame of its own, which a
    `RETURN` ends with its value.
    """
    with _deep():
        if not env.running:
            return _Evaluator(env).convert(node)
        return _unit(_Evaluator(replace(env, frame=Frame())), node)


def _unit(evaluator: _Evaluator, node: Node) -> sp.Basic:
    """`node` evaluated as an evaluation of its own: a top-level command, or the value
    of a variable a program is stored in. A `RETURN` ends it with its value, and an
    `exit` that nothing else ends ends it with the last value."""
    frame = evaluator.env.frame
    assert frame is not None
    try:
        return evaluator.statement(node)
    except _Return as returned:
        return returned.value
    except _Exit:
        return sympy_of(frame.last)


def released_held(value: sp.Basic, env: Environment) -> sp.Basic:
    """`value` with its held values released (see `_Evaluator.released_held`), in a
    top-level evaluation in `env`."""
    if not env.context.store.held or not has_held(value):
        return value
    with _deep():
        return _Evaluator(env).released_held(value)


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
#: terminate long before this; a program recursing over a list of thousands of elements
#: doesn't. One that never terminates, like `F(n) := DIF(F(n + 1), x)` or `FACT(-1)`,
#: returns what it built so far with the pending call inside: a correct rewriting that
#: shows why it won't finish. Derive itself runs out of memory there.
_NESTING = 10_000

#: Program steps a call of a program may take in a definition being shown (see
#: `_Evaluator._run_in_definition`): a program over symbolic values that ends does so
#: in few steps, while one that doesn't may make each step slower.
_SHOWN_PROGRAM_STEPS = 5_000

#: How many levels deeper the calls of a call may nest once it has gone on past an
#: undecided equation (see `_Evaluator._conditional`): enough for a polynomial's degree
#: past `IF(u = 0, RETURN -1)`, or a loop that starts by comparing with a list it has
#: not built yet, while `fact(m)` recursing past `IF(m = 0, RETURN 1)` for a symbolic m
#: stands at once rather than nesting to the limit. A loop may pass such a test any
#: number of times.
_PASSED_OVER_DEPTH = 50

class _Undecided(BaseException):
    """A running program met a test it can't decide that steers it (see
    `_Evaluator._conditional`): the call of the defined function running it stands.

    A `BaseException`, so that no handler in the engine that turns a failure into an
    unevaluated call takes it for one before the call's own `_entered` does.
    """


def _open_question(node: Node, test: sp.Basic | None, omitted: frozenset[str]) -> bool:
    """Whether the undecided test `node`, evaluated to `test`, is a question about the
    values of variables, so that a program can't go past it (see
    `_Evaluator._conditional`): not an equation, naming a variable other than the
    `omitted` locals, and over no element that doesn't exist."""
    if test is None or _equations_only(node) or not _variables_named(test) - omitted:
        return False
    return not test.has(Subscript) and not any(
        isinstance(found.args[0], Vector)
        for found in test.atoms(AppliedUndef)
        if _is_call_of(found, "ELEMENT")
    )


def _equations_only(node: Node) -> bool:
    """Whether test `node` is an equation, or a conjunction or disjunction of them."""
    if node.kind is Kind.REL:
        return str(node.value) == "="
    if node.kind in (Kind.AND, Kind.OR):
        return all(_equations_only(child) for child in node.children)
    return False


def _variables_named(value: sp.Basic) -> set[str]:
    """The names of the variables `value` mentions, the operator a `Logical` names
    aside."""
    if isinstance(value, Logical):
        return set().union(*(_variables_named(operand) for operand in value.args[1:]))
    if isinstance(value, sp.Symbol):
        return {value.name}
    if not isinstance(value, sp.Basic):
        return set()
    return set().union(*(_variables_named(argument) for argument in value.args))


def _literal(node: Node) -> bool:
    """Whether the tree `node` is a literal: a numeral, a negated numeral, or a vector
    written out of literals. Its value is the same in every precision mode and
    environment, as a numeral's is (see `numeral`)."""
    kind = node.kind
    if kind is Kind.NUMBER:
        return True
    if kind is Kind.UNOP:
        return node.value == "-" and node.children[0].kind is Kind.NUMBER
    if kind is Kind.VECTOR:
        return all(
            child.kind is not Kind.ELLIPSIS and _literal(child)
            for child in node.children
        )
    return False


def _holds_program(body: Node) -> bool:
    """Whether the body `body` of a defined function holds a program construct, which
    may meet a test it can't decide (see `_Evaluator._conditional`)."""
    return _remembered(_PROGRAMS, body, programmed)


def _remembered(memo: dict[int, tuple[Node, Any]], node: Node, find: Callable) -> Any:
    """`find(node)`, remembered in `memo` by the tree's identity, with the tree, which
    keeps the identity from being reused; at most `_TREES_KEPT` trees are kept."""
    found = memo.get(id(node))
    if found is None or found[0] is not node:
        if len(memo) >= _TREES_KEPT:
            memo.clear()
        found = memo[id(node)] = (node, find(node))
    return found[1]


#: The answers `_remembered` keeps, one memo for each question asked of trees.
_PROGRAMS: dict[int, tuple[Node, bool]] = {}
_MENTIONED: dict[int, tuple[Node, tuple[frozenset[str], frozenset[str]] | None]] = {}
_DEFINING: dict[int, tuple[Node, bool]] = {}
_LITERALS: dict[int, tuple[Node, bool]] = {}
_TREES_KEPT = 200_000


#: The Python recursion limit while evaluating. Each recursion level takes a dozen or so
#: frames, so the default thousand would stop well before `_NESTING`.
_STACK = 200_000


@contextmanager
def _deep() -> Iterator[None]:
    with deep_recursion(_STACK):
        yield


#: The operator of an assignment, as an `Assign` holds it.
_ASSIGNED = sp.Symbol(":=")

#: The cells of an evaluator without variables of its own to look up first.
_NO_CELLS: Mapping[str, Cell] = MappingProxyType({})


class _Evaluator(Converter):
    """The converter with an environment; `convert` dispatches by node kind."""

    def __init__(self, env: Environment):
        super().__init__(env.context)
        self.env = env
        self.parameters = env.parameters
        #: The value of the last `IF` test this evaluator left undecided.
        self._undecided_test: sp.Basic | None = None
        #: Whether the statements this evaluator runs compute with native values (see
        #: "compiled nodes").
        self.native = (
            COMPILED
            and env.running
            and env.frame is not None
            and not env.writing
            and not env.defining
            # Approximate rounds the values a program carries (see `carried`).
            and env.context.precision is not Precision.APPROXIMATE
        )
        #: The native values of variables a native `EVERY` or `SOME` binds here, by
        #: name, and what it found of the bindings it makes (see `_mapped_closure`).
        self.indices: dict[str, object] | None = None
        self.scopes: dict[int, tuple] | None = None
        #: The cells a name is looked up in first where values are native: a call's
        #: locals, or the globals for any other evaluation, where no name is a local.
        #: An evaluation that follows a variable's value or has arguments bound looks
        #: up its globals only after checking for them (see `_name_closure`).
        frame = env.frame
        if frame is None:
            self.scope: Mapping[str, Cell] = _NO_CELLS
        elif frame.formals:
            self.scope = frame.locals
        elif env.expanding or env.bindings:
            self.scope = _NO_CELLS
        else:
            self.scope = env.context.store.cells

    def at(self, variable: sp.Basic, value: sp.Basic) -> _Evaluator:
        """`self.within(_at(self.env, {variable: value}))`, for an evaluator that has
        evaluated nothing, made by copying it."""
        env = self.env
        written = {variable: value}
        if env.written:
            written = {
                **{symbol: _over(each, written) for symbol, each in env.written.items()},
                **written,
            }
        made = object.__new__(_Evaluator)
        made.__dict__.update(self.__dict__)
        made.env = replace(env, written=written)
        return made

    def within(self, env: Environment) -> _Evaluator:
        return _Evaluator(env)

    def capturing(self) -> frozenset[str]:
        return self.parameters | self.env.bound

    def convert(self, node: Node) -> sp.Basic:
        compiled = _NATIVES.get(id(node))
        if compiled is not None and COMPILED:
            value = compiled[1](self)
            # Only a native evaluator has native values (see "compiled nodes").
            return sympy_of(value) if self.native else value
        return self._converted(node)

    def _converted(self, node: Node) -> sp.Basic:
        """`node` converted, dispatched by its kind (see `convert`)."""
        kind = node.kind
        if kind is Kind.NAME:
            return self._name(node)
        if kind is Kind.NUMBER:
            return numeral(node)
        match kind:
            case Kind.CALL | Kind.APPLY:
                return self._call(node)
            case Kind.DOMAIN:
                if self.env.runs and self._in_program():
                    self._declared(node)
                # A declaration's bounds are read without the current scope.
                return Converter(self.context).convert(node)
            case Kind.STRING:
                return self._string(node)
            case Kind.AND | Kind.OR:
                return self._sequenced(node)
            case Kind.QUOTE:
                return self._quoted(node)
            case Kind.SUB if self.env.depth == 0:
                selected = _written_element(node)
                if selected is not None:
                    return self.convert(selected)
        return super().convert(node)

    def _transpose(self, operand: sp.Basic) -> sp.Basic:
        """``operand` ``, seeing through the `Commanded` wrapper of a vector.

        A `SOLVE` answers with a vector, still wrapped when it is an element of the
        operand, so `[SOLVE(w = 1, w), SOLVE(w = 2, w)]`` would otherwise look like a
        flat vector, which is its own transpose, rather than the column it is.
        """
        operand = _vector_value(operand)
        if isinstance(operand, Vector):
            operand = mapped(operand, _vector_value)
        return super()._transpose(operand)

    def over_equations(
        self, build: Callable[[list[sp.Basic]], sp.Basic], operands: list[sp.Basic]
    ) -> sp.Basic | None:
        """`Converter.over_equations`, seeing through the `Commanded` wrapper of one.

        Arithmetic on a command's answer that is an equation applies to both sides like
        arithmetic on any equation: with `q := FACTOR(x^2 - 1 = 0)`, `2*q` is an
        equation, not a product of 2 and the wrapped one.
        """
        return super().over_equations(build, [_equation_value(o) for o in operands])

    # -- names --------------------------------------------------------------

    def _name(self, node: Node) -> sp.Basic:
        """A name's value, looked up in this order: a variable bound here stands for
        itself; a local of the running activation has its value; a parameter its
        argument; a global the value this command gave it, or else the session's.

        `exit` ends what is running (see `Frame`), where a program runs; elsewhere it
        is just the name.
        """
        name = str(node.value)
        env = self.env
        if name == EXIT and env.runs:
            frame = env.frame
            if frame is not None and (frame.loops or frame.progs or frame.activation):
                raise _Exit
        frame = env.frame
        unset = False
        if frame is not None and name not in env.bound:
            cell = frame.locals.get(name)
            if cell is not None:
                if cell.raw is not CLEARED:
                    return _over(sympy_of(cell.raw), env.written)
                unset = True
        value = None if unset else env.bindings.get(name)
        if value is not None:
            argument = env.supplied.get(name)
            if argument is not None and argument.value is value:
                # The value was evaluated with the substitutions made then, so only
                # those made since apply to it.
                written = env.written
                if frame is not None and frame.through:
                    written = {**written, **frame.through}
                rewritten = argument.rewritten(written)
                if rewritten is None:
                    return value
                return argument.at(rewritten, env.writing)
            return _over(value, env.written)
        if name not in env.bound and name not in env.expanding:
            stored = given(self.context.store, name)
            if stored is CLEARED:
                return self._written(super()._name(node))
            if stored is not None:
                if self.context.store.held and has_held(stored):
                    stored = self._settled(self.released_held(stored))
                return _over(stored, env.written)
            assigned = env.context.assigned(name)
            if assigned is not None:
                return self._assigned(name, assigned)
        if not env.writing and name in SETTINGS:
            # A setting reads as its value in force, so inside `APPROX(u, 20)`,
            # `PrecisionDigits` is 20: BESSEL.MTH's `BESSEL_J` sums
            # `FLOOR(PrecisionDigits/2 + 4*ABS(z))` terms. Its keywords stand for
            # themselves, so `trig_ := Trigonometry` saves a setting that
            # `Trigonometry := trig_` restores.
            setting = self.context.setting(name)
            if setting is not None:
                return setting_value(setting)
        return self._written(super()._name(node))

    def _string(self, node: Node) -> sp.Basic:
        """A string: the value of the string variable it names, if it has one, else
        itself. A program's `"$start1" := true` gives one a value, and a comment such
        as `"DISPLAY(cds)"` has none."""
        name = variable_key(node)
        env = self.env
        frame = env.frame
        if frame is not None:
            cell = frame.locals.get(name)
            if cell is not None and cell.raw is not CLEARED:
                return sympy_of(cell.raw)
            if cell is None and name in env.bindings:
                return env.bindings[name]
        stored = given(self.context.store, name)
        if stored is not None and stored is not CLEARED:
            return stored
        if stored is None:
            assigned = self.context.assigned(name)
            if assigned is not None and name not in env.expanding:
                return self._assigned(name, assigned)
        # A binding over the string's variable gives it the value bound.
        return self._written(StringLiteral(str(node.value)))

    def _local(self, name: str) -> sp.Basic | None:
        """The value of `name` as a local of the running activation: assigned, or bound
        to its argument; None if it has none here."""
        env = self.env
        if env.frame is not None and name not in env.bound:
            cell = env.frame.locals.get(name)
            if cell is not None:
                return None if cell.raw is CLEARED else sympy_of(cell.raw)
        return env.bindings.get(name)

    def _assigned(self, name: str, assigned: Node) -> sp.Basic:
        """The value of assigned variable `name`, `assigned` evaluated.

        The value is evaluated in the scope of the session, whichever function refers
        to the variable; one that is a program runs when the variable is read, with a
        frame of its own, so a `RETURN` in it gives the variable's value.

        A value is worked out once for a given environment and reused: an iteration
        step may refer to a chain of assigned variables, each several times, so
        evaluating each reference anew is exponential in the length of the chain. An
        assignment made meanwhile, or a different environment, means evaluating it
        again, and a value that itself assigns, defines or draws a random number is
        never reused.

        A value written out as a literal (see `_literal`) is the same wherever it is
        evaluated, so it is worked out once in a command and kept in an unstamped cell
        of the store, which the compiled programs read as they read a global's.
        """
        env = self.env
        store = self.context.store
        literal = _remembered(_LITERALS, assigned, _literal)
        if literal:
            cell = store.cells.get(name)
            if cell is not None:
                return sympy_of(cell.raw)
        version = store.version
        try:
            if _defines(assigned):
                raise TypeError("performs when evaluated")
            key = (
                name,
                id(assigned),
                id(self.context),
                version,
                env.bound,
                env.parameters,
                env.expanding,
                env.labels,
                env.depth,
                env.running,
                env.writing,
                env.defining,
                self._written_read(name, assigned),
                frozenset(env.supplied),
                id(env.decide),
                id(env.work_out),
                id(env.command),
                id(env.simplify),
            )
            hash(key)
        except TypeError:
            key = None
        if key is not None and key in env.values:
            return env.values[key]
        deeper = replace(
            env, bindings={}, frame=Frame(), expanding=env.expanding | {name}
        )
        if env.runs:
            value = _unit(self.within(deeper), assigned)
        else:
            value = self.within(deeper).convert(assigned)
        if literal and name not in store.cells:
            # Unstamped: no value the command gave it (see `given`).
            store.cells[name] = Global(native_of(value), 0)
        if key is not None and store.version == version:
            env.values[key] = value
        return value

    def _written_read(self, name: str, assigned: Node) -> tuple:
        """The values binders give that the value of assigned variable `name` may
        read: those of the variables its tree mentions, and those of the variables
        they hold in turn. Inside `SUM(... v SUB k ..., k, 1, n)`, a vector `v` stored
        with numbers doesn't depend on `k`, so it is worked out once, not at each `k`.
        A value that calls a defined function, which may read any global, depends on
        all of them."""
        written = self.env.written
        if not written:
            return ()
        names = self._read_by(assigned, {name})
        if names is None:
            return tuple(written.items())
        return tuple(
            (symbol, value)
            for symbol, value in written.items()
            if getattr(symbol, "name", None) in names or not hasattr(symbol, "name")
        )

    def _read_by(self, node: Node, seen: set[str]) -> set[str] | None:
        """The variable names the value of `node` may read, following the variables
        it mentions to their values; None if it calls a defined function."""
        mentioned = _mentioned(node)
        if mentioned is None:
            return None
        names, calls = mentioned
        context = self.context
        if any(context.function(call) is not None for call in calls):
            return None
        found = set(names)
        for each in names - seen:
            seen.add(each)
            stored = given(context.store, each)
            if stored is not None:
                if isinstance(stored, sp.Basic):
                    found |= {symbol.name for symbol in _free_symbols(stored)}
                continue
            value = context.assigned(each)
            if value is not None:
                deeper = self._read_by(value, seen)
                if deeper is None:
                    return None
                found |= deeper
        return found

    def _written(self, symbol: sp.Basic) -> sp.Basic:
        return self.env.written.get(symbol, symbol)

    def _subscript(self, base_node: Node, base: sp.Basic, index: sp.Basic) -> sp.Basic:
        return self._written(super()._subscript(base_node, base, index))

    def operand(self, value: sp.Basic) -> sp.Basic:
        return _assigned_value(_value_of(value))

    def operands_of(self, values: list[sp.Basic]) -> list[sp.Basic]:
        return [_assigned_value(operand) for operand in super().operands_of(values)]

    def sided(self, value: sp.Basic) -> sp.Basic:
        """A side of a relation: an assignment there counts as its value, so `2 > n
        := DIM(x)` compares with the dimension."""
        return _assigned_value(value)

    def _label(self, node: Node) -> sp.Basic:
        """The value of label `#3`, or the label itself if unknown or self-referential."""
        try:
            number = int(str(node.value))
        except ValueError:
            number = -1
        target = self.context.labels.get(number)
        if target is None or number in self.env.labels:
            return sp.Symbol(f"#{node.value}")
        deeper = replace(
            self.env, bindings={}, frame=Frame(), labels=self.env.labels | {number}
        )
        return self.within(deeper).convert(target)

    # -- the quote ------------------------------------------------------------

    def _quoted(self, node: Node) -> sp.Basic:
        """`'u`: `u` as written, held (see `held_value`), with each name in it that is
        a local in scope here replaced by its value, in quotes nested in it too. A
        quote shown or written out stays the quote, as written."""
        if self.env.writing or self.env.defining:
            return as_written(node)
        self.context.store.held = True
        tree = _filled(node.children[0], self._quote_scope(), self.context)
        return held_value(tree, Converter(self.context))

    def _quote_scope(self) -> Callable[[str], sp.Basic | None]:
        """The value a name has as a local in scope here, None if it has none: the
        variable of a binder whose body this is, at its current value, or a parameter
        of the running activation with a value."""
        env = self.env
        by_name = {
            getattr(symbol, "name", None): value for symbol, value in env.written.items()
        }

        def value_of(name: str) -> sp.Basic | None:
            if name in env.bound:
                return by_name.get(name)
            local = self._local(name)
            if local is not None:
                return _assigned_value(_value_of(local))
            return by_name.get(name)

        return value_of

    def released_held(self, value: sp.Basic) -> sp.Basic:
        """`value` with each held value in it evaluated as an expression at top level:
        its locals were filled in when it was made, and the globals in it are read now.
        A quote in it gives a new held value, and an assignment is made."""
        if not self.context.store.held or not has_held(value):
            return value
        env = replace(
            self.env,
            bindings={},
            supplied={},
            frame=Frame() if self.env.running else None,
            bound=frozenset(),
            parameters=frozenset(),
            written={},
            defining=False,
            writing=False,
        )
        evaluator = self.within(env)
        found = {}
        for atom in value.atoms(HeldBase):
            tree = type(atom).node
            if env.frame is not None:
                found[atom] = _value_of(_unit(evaluator, tree))
            else:
                found[atom] = _value_of(evaluator.convert(tree))
        try:
            return settled(value.xreplace(found))
        except Exception:
            return value.xreplace(found)

    # -- logic ----------------------------------------------------------------

    def _logical(self, node: Node) -> sp.Basic:
        """A logical operator, its operands evaluated left to right; an assignment
        among them counts as its value."""
        operands = [_assigned_value(self.convert(child)) for child in node.children]
        return connected(LOGICAL_WORDS[node.kind], operands)

    def _sequenced(self, node: Node) -> sp.Basic:
        """`p AND q`, `p OR q`: the right operand is evaluated only if the left one
        leaves the answer open, so programs sequence statements with them. `false AND
        q` is false and `true OR q` true without `q`; otherwise `q` is evaluated with
        its assignments and control statements."""
        word = LOGICAL_WORDS[node.kind]
        left = self._truth(_assigned_value(self.convert(node.children[0])))
        if left is DECIDING[word]:
            return left
        right = self._truth(_assigned_value(self.convert(node.children[1])))
        return connected(word, [left, right])

    def _truth(self, value: sp.Basic) -> sp.Basic:
        """`value`, a comparison of two rational numbers or an equation of two strings
        decided, as a loop's tests mostly are: deciding it at once spares building and
        deciding a conjunction of relations. Shown, a comparison stays as written."""
        if self.env.writing or not isinstance(value, Relational):
            return value
        left, right = value.args
        if (
            sp.nan in (left, right)
            and not isinstance(value, (sp.Eq, sp.Ne))
            and self.context.precision is Precision.APPROXIMATE
        ):
            # Decided in numbers, no ordering with `?` holds (see `pipeline._stated`).
            return sp.false
        if isinstance(left, sp.Rational) and isinstance(right, sp.Rational):
            holds = _ORDERED_BY_SIGN.get(type(value))
            if holds is None:
                return sp.true if value.func(left, right) else sp.false
            # The sign of the difference, from the integers: denominators are positive.
            difference = left.p * right.q - right.p * left.q
            return sp.true if holds(difference) else sp.false
        if (
            isinstance(value, (sp.Eq, sp.Ne))
            and isinstance(left, StringLiteral)
            and isinstance(right, StringLiteral)
        ):
            same = left.name == right.name
            return sp.true if same == isinstance(value, sp.Eq) else sp.false
        return value

    # -- programs -------------------------------------------------------------

    def statement(self, node: Node) -> sp.Basic:
        """`node` evaluated as a statement, its value made the frame's last value.

        An `IF` sets the last value to its test's truth value once decided, and a vector
        to each element's value as it completes, so an `exit` in a clause or an element
        leaves its construct with that; the chosen clause and the elements are
        statements in turn.
        """
        frame = self.env.frame
        if frame is None or not self.env.runs:
            return self.convert(node)
        current = self.context.store.context
        if current is not None and current is not self.context:
            # A statement before this one assigned a setting.
            return self.within(replace(self.env, context=current)).statement(node)
        if self.native:
            return sympy_of(_statement(node)(self))
        self.context.step()
        if COMPILED and id(node) not in _STATEMENTS:
            _statement(node)
        if node.kind is Kind.CALL:
            head, *arguments = node.children
            if str(head.value) == "IF" and 1 <= len(arguments) <= 4:
                value = self._conditional(node, arguments, statement=True)
                frame.last = value
                return value
        if node.kind is Kind.VECTOR and not any(
            child.kind is Kind.ELLIPSIS for child in node.children
        ):
            value = Vector(*(self.statement(child) for child in node.children))
        else:
            value = self.convert(node)
        frame.last = value
        return value

    def _program(self, statements: Sequence[Node]) -> sp.Basic:
        """`PROG(s1, ..., sn)`: the statements in order, the value of the last one.

        An `exit` in it ends it with the last value, also inside a `LOOP`: the
        innermost `PROG` or `LOOP` is what an `exit` ends. LaplaceTransforms' loop over
        the terms skips the rest of a term's `PROG` so, going on with the next term.
        """
        frame = self.env.frame
        assert frame is not None
        frame.progs += 1
        try:
            value: sp.Basic = sp.nan
            for statement in statements:
                value = self.statement(statement)
            return value
        except _Exit:
            return sympy_of(frame.last)
        finally:
            frame.progs -= 1

    def _loop(self, statements: Sequence[Node]) -> sp.Basic:
        """`LOOP(s1, ..., sn)`: the statements in order, over and over, until an `exit`
        ends it with the last value, or a `RETURN` the activation."""
        frame = self.env.frame
        assert frame is not None
        frame.loops += 1
        try:
            while True:
                for statement in statements:
                    self.statement(statement)
        except _Exit:
            return sympy_of(frame.last)
        finally:
            frame.loops -= 1

    def _shown(self, name: str, arguments: Sequence[Node]) -> sp.Basic:
        """`DISPLAY(u, n)` and `WRITE(u)`: the value of `u`, simplified, which is shown
        as text as well: a string as its characters, anything else as an answer would
        be written under the settings in force. `DISPLAY` adds the text to the
        worksheet before the command's answer, `WRITE` writes it on the status line
        (see `Store.show`). `n`, the lines `DISPLAY` may take, has no use in a terminal
        worksheet, whose text lines wrap; a call with an `n` that is no positive integer,
        or with more arguments, stands, showing nothing."""
        values = [
            _assigned_value(_value_of(self.convert(argument))) for argument in arguments
        ]
        value = approximated(self._settled(values[0]), self.context)
        rest = values[1:]
        if len(arguments) > (2 if name == "DISPLAY" else 1) or (
            rest and not (isinstance(rest[0], sp.Integer) and rest[0] > 0)
        ):
            return self.opaque(name, [value, *rest])
        store = self.context.store
        if isinstance(value, StringLiteral):
            text = value.name
        else:
            text = author_text(released(value), store.context or self.context)
        store.show(text, name == "DISPLAY")
        return value

    def _load(self, argument: Node) -> sp.Basic:
        """`LOAD(s)`: true once the session has loaded the file the string `s` names,
        and the call standing if it found none. The engine reads no files: a file the
        session hasn't looked for yet ends the command (see `FileNeeded`). Anything but
        a string leaves the call standing."""
        value = _assigned_value(_value_of(self.convert(argument)))
        if not isinstance(value, StringLiteral):
            return self.opaque("LOAD", [value])
        found = self.context.loads.get(value.name)
        if found is None:
            raise FileNeeded(value.name)
        return sp.true if found else self.opaque("LOAD", [value])

    def _returned(self, operand: Node) -> sp.Basic:
        """`RETURN u`: the activation ends with the value of `u`."""
        raise _Return(self._settled(self.convert(operand)))

    def _settled(self, value: sp.Basic, top: bool = False) -> sp.Basic:
        """`value` as a program stores it: simplified as Simplify would simplify it
        alone, so that a loop works on simple values rather than ever longer
        expressions. Numbers, strings, truth values and vectors and sets of those are
        already as simple as they get. `top` keeps an equation that is the value of the
        assignment being simplified (see `Environment.top`).

        The value is simplified exactly, and under Approximate rounded only as any value
        carried from one step to the next is (see `carried`): the precision's
        approximation is the last step of the command, and a statement after this one
        may need the exact value, as a limit at a root of the denominator does."""
        value = _value_of(value)
        if is_simple(value) or self._approximated(value):
            return carried(value, self.context)
        simplify = self.env.simplify
        if simplify is not None and not self.env.writing:
            try:
                exact = self.context.with_precision(Precision.EXACT)
                value = simplify(value, exact, top=top)
            except Exception:
                pass
        return carried(value, self.context)

    def _approximated(self, value: sp.Basic) -> bool:
        """Whether `value` is a number that Approximate precision approximates as it
        is: simplifying it first would only cost time."""
        if self.context.precision is not Precision.APPROXIMATE:
            return False
        try:
            return isinstance(value, sp.Expr) and not value.free_symbols and (
                value.is_number is True
            )
        except Exception:
            return False

    # -- definitions --------------------------------------------------------

    def performs(self) -> bool:
        """Whether an assignment or definition here is made.

        It is where a command runs programs, and in the body of a called function, as
        in Derive: with `A1(u, x) := (A2(x) := u)`, `A1(y^2, y)` defines `A2(y) :=
        y^2`. It isn't in a line that is only entered, which only states a definition
        (the session makes it), nor in a definition being shown or a conditional
        being written out.
        """
        env = self.env
        return (env.running or env.depth > 0) and not env.writing and not env.defining

    def _assign(self, node: Node) -> sp.Basic:
        """An assignment: the value is evaluated, the target isn't.

        Where it is made (see `performs`), the target is a local if it is a parameter
        of the running activation and a global otherwise, and the stored value is the
        value simplified (see `_settled`). The assignment of a local counts as its
        value; that of a global stays the assignment, with the value, which an operand
        or argument takes the value of (see `_assigned_value`). `x :=` takes the value
        away. An update `x :+ u` assigns `x + u`, and a target with subscripts an
        element of the variable's vector.

        An assignment that is only stated doesn't follow a value of its target that
        refers to the target itself: that value is what the session stored for this
        very assignment, so `a := a + 7` simplifies to `a := a + 7`, not to `a := a +
        14`. The same holds for one made at top level.
        """
        target, *values = node.children
        operator = str(node.value)
        plain = operator == ":=" and target.kind in (Kind.NAME, Kind.STRING)
        if not self.performs() or not (plain or self.env.runs):
            return self._stated(node)
        if target.kind not in (Kind.NAME, Kind.STRING):
            return self._element_assigned(node)
        name = variable_key(target)
        if target.kind is Kind.NAME and name in SETTINGS:
            return self._setting_assigned(
                name, self._target_symbol(target), operator, values
            )
        variable = self._variable_of_formal(target)
        # The target as the assignment is written back, which a local's needs not be.
        symbol: sp.Basic | None = variable
        if variable is not None:
            name = variable.name
        local = variable is None and self._is_local(name)
        if symbol is None and not local:
            symbol = self._target_symbol(target)
        if not values or (
            operator == ":=" and not local and _names_target(values[0], target)
        ):
            if variable is not None:
                self.env.frame.through.pop(variable, None)
            if symbol is None:
                symbol = self._target_symbol(target)
            return self._cleared(name, symbol, local)
        value = _assigned_value(_value_of(self._right_side(target, values[0])))
        if operator != ":=":
            value = _updated(operator, self.convert(target), value)
        settled = self._settled(value, top=node is self.env.top)
        # An update's operand is no value the variable takes over (see `_given`).
        shared = values[0] if operator == ":=" else None
        if variable is not None:
            # Written as an assignment to the formal, it counts as its value, as an
            # assignment to a local does.
            self.env.frame.through[variable] = settled
            self._given(name, symbol, settled, shared, local)
            return settled
        return self._given(name, symbol, settled, shared, local)

    def _variable_of_formal(self, target: Node) -> sp.Symbol | None:
        """The variable a formal stands for, where an assignment to the formal assigns
        that variable: the formal was supplied with a variable's bare name and holds it
        still. So `SIMPLE_SUBST(expr_, var_, svar_) := PART([var_ := svar_, expr_, var_
        := var_], 2)`, called with `x` for `var_`, gives `x` the value `svar_` while
        `expr_` is evaluated, as the formal's use as a binder variable is that variable
        (see `_binding`). In a `PROG` or `LOOP` a formal is a local like any other, so
        this is for an assignment in a body that is no program. None for any other
        target."""
        frame = self.env.frame
        name = str(target.value)
        if (
            target.kind is not Kind.NAME
            or frame is None
            or frame.progs
            or frame.loops
            or name not in frame.formals
            or name in frame.locals
            or name in self.env.bound
        ):
            return None
        value = self.env.bindings.get(name)
        if type(value) is not sp.Symbol or value.name in frame.formals:
            return None
        return value

    def _setting_assigned(
        self, name: str, symbol: sp.Basic, operator: str, values: Sequence[Node]
    ) -> sp.Basic:
        """`Setting := u`: the setting takes the value of `u` at once, so that the
        statements after this one are evaluated under it, and keeps it after the
        command, unless it is a parameter of the running activation, which `_entered`
        restores on return. The value is the assignment, or for a parameter its value;
        an invalid value changes nothing and leaves the assignment standing."""
        written = [_assigned_value(_value_of(self.convert(child))) for child in values]
        assignment = Assign(symbol, sp.Symbol(operator), *written)
        if operator != ":=" or len(written) > 1:
            return assignment
        if written:
            (value,) = written
            setting = _setting_written(value)
        else:
            # `Name :=` restores the factory default.
            setting = CLEARED_SETTING
        changed = None if setting is None else self.context.with_setting(name, setting)
        if changed is None:
            return assignment
        if not written:
            value = setting_value(changed.setting(name))
        store = self.context.store
        store.context = changed
        if self._is_local(name):
            assert self.env.frame is not None
            self.env.frame.locals[name] = Cell(value)
            return value
        store.set(name, setting)
        return assignment

    def _right_side(self, target: Node, value: Node) -> sp.Basic:
        """The value of the right side of an assignment to `target`, not following the
        target's value at top level where that refers to the target (see `_assign`)."""
        name = str(target.value)
        if (
            target.kind is Kind.NAME
            and self.env.depth == 0
            and (self.env.frame is None or not self.env.frame.activation)
            and not assigned(self.context.store, name)
            and _refers_to(self.context.assigned(name), name)
        ):
            expanding = self.env.expanding | {name}
            return self.within(replace(self.env, expanding=expanding)).convert(value)
        return self.convert(value)

    def _stated(self, node: Node) -> sp.Basic:
        """An assignment that is not made: the value evaluated, the target as written.

        In the value of a variable or label followed here, it is a value like any
        other: with `home := [x := 0, y := 0]`, `home` is `[0, 0]` where nothing runs.
        """
        target, *values = node.children
        operator = str(node.value)
        plain = Converter(self.context)
        if operator != ":=":
            written = [self.convert(child) for child in values]
            return Assign(plain.convert(target), sp.Symbol(operator), *written)
        written = [
            self._right_side(target, child)
            if target.kind is Kind.NAME
            else self.convert(child)
            for child in values
        ]
        if written and (self.env.expanding or self.env.labels) and not self.env.writing:
            return written[-1]
        return Assign(plain.convert(target), sp.Symbol(operator), *written)

    def _target_symbol(self, target: Node) -> sp.Basic:
        """The target of an assignment as the assignment is written back."""
        if target.kind is Kind.STRING:
            return StringLiteral(str(target.value))
        return Converter(self.context).symbol(str(target.value))

    def assignment_of(self, made: NativeAssignment | Assigned) -> Assign:
        """The `Assign` that the assignment of a global a compiled program made stands
        for (see `NativeAssignment`), its target as the declarations in force then
        write it."""
        context = self.context
        store = context.store
        if store.domains is not made.domains:
            then = copy.copy(store)
            then.domains = made.domains
            context = dataclasses.replace(context, store=then)
        target = made.node.children[0]
        value = sympy_of(made.value)
        if made.places is None:
            symbol = Converter(context).symbol(str(target.value))
            return Assign(symbol, _ASSIGNED, value)
        links, base = _element_links(target)
        written = Converter(context).symbol(str(base.value))
        for (link, _), place in zip(links, made.places, strict=True):
            written = _linked(link, written, self.operand(sympy_of(place)))
        return Assign(written, _ASSIGNED, value)

    def _is_local(self, name: str) -> bool:
        frame = self.env.frame
        return frame is not None and name in frame.formals

    def _given(
        self,
        name: str,
        symbol: sp.Basic | None,
        value: sp.Basic,
        written: Node | None,
        local: bool | None = None,
    ) -> sp.Basic:
        """`value` given to the variable `name`: a local, whose value the assignment
        counts as, or a global, whose assignment it stays, to `symbol`. Which, `local`
        says, or if None, whether `name` is a parameter of the running activation.

        A local given another local by its bare name shares its vector (see `Cell`).
        """
        frame = self.env.frame
        if local is None:
            local = self._is_local(name)
        if local:
            assert frame is not None
            cell = None
            if written is not None and isinstance(value, Vector):
                cell = self._cell_of(written)
            frame.locals[name] = cell if cell is not None else Cell(value)
            return value
        self.context.store.assign(name, value)
        return Assign(symbol, sp.Symbol(":="), value)

    def _cell_of(self, written: Node) -> Cell | None:
        """The cell of the local `written` names, made for it if it has only its
        argument, or None if it names no local with a value."""
        frame = self.env.frame
        if frame is None or written.kind is not Kind.NAME:
            return None
        name = str(written.value)
        if name in self.env.bound or name not in frame.formals:
            return None
        cell = frame.locals.get(name)
        if cell is not None:
            return None if cell.raw is CLEARED else cell
        if name not in self.env.bindings:
            return None
        cell = frame.locals[name] = Cell(self.convert(written))
        return cell

    def _cleared(
        self, name: str, symbol: sp.Basic, local: bool | None = None
    ) -> sp.Basic:
        """`x :=`: the local or global loses its value (`local` as for `_given`). So
        does a global assigned its own name, `x := x`, as in Derive: `[x := y, y^2 - x,
        x := x]` is `[y, y^2 - y, x]`, and leaves `x` a variable."""
        frame = self.env.frame
        if local is None:
            local = self._is_local(name)
        if local:
            assert frame is not None
            frame.locals[name] = Cell(CLEARED)
        else:
            self.context.store.assign(name, CLEARED)
        return Assign(symbol, sp.Symbol(":="))

    def _element_assigned(self, node: Node) -> sp.Basic:
        """`A SUB i SUB j := u`: element `j` of element `i` of the vector `A` holds
        replaced by `u`, or with a vector index, the listed elements by those of `u`.

        The indices are evaluated first, left to right, then `u`, then the vector is
        updated, so `h SUB i := h SUB j` with `i` and `j` vectors swaps rows. Nothing is
        assigned where an index isn't one of the vector's positions counted from 1
        (reading counts a negative one from the end, but assigning at it leaves the
        vector as it is, as in Derive), where the variable doesn't hold a vector there,
        or where the target is a column (`A SUB SUB j`, `A COL j`); the assignment then
        stands, its parts evaluated. The value is as for `_given`: the element's for a
        local, the assignment, with its indices evaluated, for a global.
        """
        target, *values = node.children
        links, base = _element_links(target)
        indices = [self.operand(self.convert(index)) for _, index in links]
        value = (
            _assigned_value(_value_of(self.convert(values[0]))) if values else None
        )
        return self._element_stored(node, links, base, indices, value)

    def _element_stored(
        self,
        node: Node,
        links: list[tuple[Node, Node]],
        base: Node,
        indices: list[sp.Basic],
        value: sp.Basic | None,
    ) -> sp.Basic:
        """The rest of `_element_assigned`, its indices and value evaluated."""
        operator = str(node.value)
        assignable = base.kind in (Kind.NAME, Kind.STRING) and all(
            link.kind is Kind.SUB or link.value == "ROW" for link, _ in links
        )
        if base.kind in (Kind.NAME, Kind.STRING):
            written = self._target_symbol(base)
        else:
            written = self.convert(base)
        for (link, _), index in zip(links, indices, strict=True):
            written = _linked(link, written, index)
        if value is None or not assignable:
            parts = [] if value is None else [value]
            return Assign(written, sp.Symbol(":="), *parts)
        name = variable_key(base)
        current = self.convert(base)
        if operator != ":=":
            element = _element_at_path(current, indices)
            if element is None:
                return Assign(written, sp.Symbol(":="), value)
            value = _updated(operator, element, value)
        value = self._settled(value)
        replaced = _replaced_at(current, indices, value)
        if replaced is None:
            return Assign(written, sp.Symbol(":="), value)
        frame = self.env.frame
        if frame is not None and name in frame.formals:
            cell = frame.locals.get(name)
            if cell is None or cell.raw is CLEARED:
                frame.locals[name] = Cell(replaced)
            else:
                cell.raw = replaced
            return value
        self.context.store.assign(name, replaced)
        return Assign(written, sp.Symbol(":="), value)

    def _assigned_by_name(self, arguments: Sequence[Node]) -> sp.Basic:
        """`ASSIGN(v, u)`: `u` assigned to the variable `v` evaluates to, as `v := u`
        would assign it; a computed target. If `v` isn't a variable (a name, an
        @-name or a string), nothing is assigned and the call stands."""
        target, value = (_assigned_value(self.convert(a)) for a in arguments)
        name = None
        if isinstance(target, StringLiteral):
            name = f'"{target.name}"'
        elif type(target) is sp.Symbol and " " not in target.name:
            name = target.name
        if name is None or name in SETTINGS:
            return self.opaque("ASSIGN", [target, value])
        return self._given(name, target, self._settled(value), None)

    def _in_program(self) -> bool:
        """Whether this is inside a called body or a running `PROG` or `LOOP`, rather
        than a line as the user stated it."""
        frame = self.env.frame
        return self.env.depth > 0 or (
            frame is not None and bool(frame.progs or frame.loops)
        )

    def _declared(self, node: Node) -> None:
        """A declaration a program makes: of the global, whatever the name is here."""
        self.context.store.declare(node)

    def _target(self, name: str) -> str | None:
        """The name a definition here defines or uses for `name`, or None if it has
        none.

        A definition in a called body is textual substitution, so a parameter of the
        call stands for its argument, which must then be a variable.
        """
        value = self._local(name)
        if value is None:
            return name
        if type(value) is sp.Symbol and value.name.isidentifier():
            return value.name
        return None

    def _perform(self, definition: sp.Basic) -> bool:
        """Make the definition `definition`, a `FunDef`, if it reads back as one."""
        try:
            node = from_sympy(definition, self.context).exact
        except Exception:
            return False
        if node.kind is not Kind.FUNDEF:
            return False
        self.context.store.define(node)
        return True

    def _fundef(self, node: Node) -> sp.Basic:
        """A definition: the body is evaluated, its parameters standing for themselves.

        A variable-binding call over a parameter is left as a call, since its variable
        is whatever a call of the function supplies (see `_binding`).

        In a called function's body, or in a program a command runs, it is made instead
        (see `_performed`).
        """
        if self.performs() and self._in_program():
            performed = self._performed(node)
            if performed is not None:
                return performed
        parameters, *body = node.children
        plain = Converter(self.context)
        if body and programmed(body[0]):
            # Running a program over symbolic parameters need not end, so the body is
            # kept as written, and so is one that assigns or defines.
            converted = [plain.convert(child) for child in parameters.children]
            return FunDef(
                sp.Symbol(str(node.value)),
                formals(parameters, converted),
                as_written(body[0]),
            )
        names = frozenset(parameter_names(parameters))
        inner = replace(
            self.env,
            bindings={},
            bound=self.env.bound | names,
            parameters=names,
            written=_unwritten(self.env.written, names),
            defining=True,
        )
        written = [self.within(inner).convert(part) for part in body]
        converted = [plain.convert(child) for child in parameters.children]
        return FunDef(
            sp.Symbol(str(node.value)), formals(parameters, converted), *written
        )

    def _performed(self, node: Node) -> sp.Basic | None:
        """A definition made by a program: the definition.

        The call's arguments are written into the definition, its parameters included:
        `A1(u, x) := (A2(x) := u)` called with `y^2` and `y` defines `A2(y) := y^2`.
        The body is stored as written, with only names substituted.

        None, leaving the definition as written, if a parameter is supplied with
        something other than a variable, or the definition doesn't read back.
        """
        parameters, *body = node.children
        name = self._target(str(node.value))
        found = parameters_of(parameters)
        names = [self._target(parameter) for parameter, _ in found]
        if name is None or None in names:
            return None
        head = sp.Symbol(name)
        written = replace(self.env, writing=True, bound=self.env.bound | set(names))
        writer = self.within(written)
        symbols = formals(
            parameters,
            [
                sp.Symbol(parameter)
                if default is None
                else Assign(
                    sp.Symbol(parameter),
                    sp.Symbol(":="),
                    released(writer.convert(default)),
                )
                for parameter, (_, default) in zip(names, found, strict=True)
            ],
        )
        definition = FunDef(head, symbols)
        if body:
            definition = FunDef(head, symbols, released(writer.convert(body[0])))
        return definition if self._perform(definition) else None

    # -- calls --------------------------------------------------------------

    def _call(self, node: Node) -> sp.Basic:
        head, *arguments = node.children
        name = str(head.value)
        if name in BINDING_FUNCTIONS:
            return self._binding(name, arguments)
        if name == "IF" and 1 <= len(arguments) <= 4 and not self.env.writing:
            return self._conditional(node, arguments)
        if name in PROGRAM_FUNCTIONS and not (
            self.env.runs and self.env.frame is not None
        ):
            # The program stands, and the names it assigns are its own.
            self.context.store.inert.update(_assigned_names(node))
        elif name in PROGRAM_FUNCTIONS:
            if name == "PROG" and arguments:
                return self._program(arguments)
            if name == "LOOP" and arguments:
                return self._loop(arguments)
            if name == "RETURN" and len(arguments) == 1:
                return self._returned(arguments[0])
            if name == "ASSIGN" and len(arguments) == 2:
                return self._assigned_by_name(arguments)
            if name in ("DISPLAY", "WRITE") and arguments:
                return self._shown(name, arguments)
            if name == "LOAD" and len(arguments) == 1:
                return self._load(arguments[0])
        if name == "SORT" and len(arguments) == 4:
            ordered = self._sorted(arguments)
            if ordered is not None:
                return ordered
        if name in _SOLVING and len(arguments) >= 2 and not self.env.writing:
            unknowns = self._unknowns(arguments[1])
            if unknowns:
                inner = replace(self.env, bound=self.env.bound | unknowns)
                return self.within(inner)._call(node)
        if name == "TRUTH_TABLE" and not self.env.writing:
            # The leading names are the table's variables, which take the values true
            # and false whatever value they have: with `a1 := "11110000"`, the table of
            # `a1 OR a2` is over `a1`.
            variables = frozenset(
                variable_key(argument)
                for argument in arguments
                if argument.kind is Kind.NAME
            ) - self.env.bound
            if variables:
                inner = replace(
                    self.env,
                    bound=self.env.bound | variables,
                    written=_unwritten(self.env.written, variables),
                )
                return self.within(inner)._call(node)
        if name == "APPROX" and 1 <= len(arguments) <= 2 and not self.env.writing:
            values = self._approximating(arguments)
        elif name in ("NSOLVE", "NSOLUTIONS") and len(arguments) == 4:
            values = self._searched_in(arguments)
        elif name in ("MOD", "MODS") and _powered(arguments) and not self.env.writing:
            values = self._power_residue(arguments)
        elif name == "VARIABLES":
            # The variables of terms an infinity absorbs in the argument are listed
            # too, those in the bodies of the functions it calls aside.
            with absorbed_by_infinity(True):
                values = [self.convert(argument) for argument in arguments]
                return self.called(node, values)
        else:
            values = [self.convert(argument) for argument in arguments]
        return self.called(node, values)

    def called(self, node: Node, values: list[sp.Basic]) -> sp.Basic:
        """The call `node` on `values`, its arguments evaluated as `_call` evaluates
        them, a command's answer still wrapped in `Commanded`."""
        head, *arguments = node.children
        name = str(head.value)
        current = self.context.store.context
        if (
            current is not None
            and current is not self.context
            and not self.env.writing
            and current.setting_values != self.context.setting_values
        ):
            # Evaluating an argument assigned a setting (a program called in it), which
            # is in force from then on: `STRING(f(pi))` writes in the digits f set.
            return self.within(replace(self.env, context=current)).called(node, values)
        written = [_assigned_value(value) for value in values]
        values = [_assigned_value(_value_of(value)) for value in values]
        if name == "SUBS" and not self.env.writing:
            return self._substitution(values)
        derived = self._derived(name, values)
        if derived is not None:
            return derived
        definition = self.context.function(name)
        if definition is not None:
            if self.env.writing:
                return self.opaque(name, values)
            with absorbed_by_infinity(False):
                value = self._entered(name, definition, values, arguments)
            if name == "LAPLACE" and definition is library_definition(name):
                return _transform_where_it_converges(value, values)
            return value
        if (
            self.context.store.held
            and not self.env.writing
            and any(has_held(value) for value in values)
        ):
            if name in _ON_HELD_FORM:
                read = self._held_form(name, values)
                if read is not None:
                    return read
            if name not in _HOLDING and name not in _ON_HELD_FORM and (
                name in BUILTIN_FUNCTIONS or name in COMMAND_HEADS
            ):
                values = [self.released_held(value) for value in values]
        if name in _SIMPLIFIED_ARGUMENTS:
            values = self._read_in_form(name, values, written)
        if (
            name in _AT_CALL
            and self.env.defining
            and self._parametric(values)
            and not self._approximating_again(name, values)
        ):
            return self.opaque(name, values)
        if name in ON_FORM and self._unsettled(
            values, defining=name not in DECOMPOSING
        ):
            return self.opaque(name, values)
        made = self.call(name, values)
        if name == "TERMS" and len(arguments) == 1 and _expansion_in(arguments[0]):
            # The terms of an expansion in `x` come by the powers of `x`, as those of
            # `TERMS(u, x)` do.
            variables = [
                _value_of(self.convert(child)) for child in arguments[0].children[2:]
            ]
            made = terms_by_expansion(made, variables, self.context.order)
        if name in COMMAND_HEADS and _is_call_of(made, name):
            if self.env.command is not None and not self.env.writing:
                return self.env.command(
                    name, list(made.args), self._divisors(name, arguments), self.context
                )
        if name in _STRUCTURAL:
            return made
        return self._worked(made, values)

    def _approximating_again(self, name: str, values: list[sp.Basic]) -> bool:
        """Whether `APPROX(u)` changes nothing, being computed under Approximate: the
        definition's body is approximated already, so it needn't stay a call."""
        return (
            name == "APPROX"
            and len(values) == 1
            and self.context.precision is Precision.APPROXIMATE
        )

    def _unknowns(self, variables: Node) -> frozenset[str]:
        """The names a solving function's variables argument lists that have a value
        which is not a variable, so the call solves for the name itself: with `d1 := (a1
        - b1)/2`, `SOLVE(2*d1 = a1 - b1, d1)` is `d1 = (a1 - b1)/2`. A name whose value
        is a variable or a vector of them names those, as `vars1 := [x, y]` does."""
        names = (
            [variables]
            if variables.kind is Kind.NAME
            else list(variables.children)
            if variables.kind is Kind.VECTOR
            else []
        )
        found = set()
        for child in names:
            if child.kind is not Kind.NAME:
                continue
            name = variable_key(child)
            if name in self.env.bound:
                continue
            stored = given(self.context.store, name)
            if stored is None and self.context.assigned(name) is None:
                continue
            if stored is CLEARED:
                continue
            try:
                value = _value_of(self.convert(child))
            except Exception:
                continue
            if type(value) is sp.Symbol or isinstance(value, (Vector, StringLiteral)):
                continue
            found.add(name)
        return frozenset(found)

    def _searched_in(self, arguments: Sequence[Node]) -> list[sp.Basic]:
        """The arguments of `NSOLVE(u, x, a, b)` or `NSOLUTIONS(u, x, a, b)`: `u` is
        evaluated with `x` taken to lie in `[a, b]`, as the search does, so that a test on
        `x` the interval decides chooses its branch. Predicted_Values solves
        `STUDENT(t, v) = ci` over `[0, inf)`, where `STUDENT`'s `IF(t < 0, ...)` would
        otherwise stand, leaving nothing to evaluate at a point."""
        rest = [
            _assigned_value(_value_of(self.convert(argument)))
            for argument in arguments[1:]
        ]
        variable, low, high = rest
        bounded = (
            not self.env.writing
            and type(variable) is sp.Symbol
            and all(_real_end(end) for end in (low, high))
        )
        if not bounded:
            first = _assigned_value(_value_of(self.convert(arguments[0])))
            return [first, *rest]
        store = self.context.store
        name = variable.name
        before = store.domains.get(name)
        store.domains = {
            **store.domains,
            name: Domain(
                DomainKind.REAL,
                written_tree(low, self.context),
                written_tree(high, self.context),
                closed_low=low.is_finite is True,
                closed_high=high.is_finite is True,
            ),
        }
        store.version += 1
        try:
            first = _assigned_value(_value_of(self.convert(arguments[0])))
        finally:
            domains = dict(store.domains)
            if before is None:
                domains.pop(name, None)
            else:
                domains[name] = before
            store.domains = domains
            store.version += 1
        # The answer is in the variable as the caller has it.
        declared = {
            symbol: variable
            for symbol in first.free_symbols
            if type(symbol) is sp.Symbol and symbol.name == name and symbol != variable
        }
        if declared:
            first = first.xreplace(declared)
        return [first, *rest]

    def _simplified(self, values: list[sp.Basic]) -> list[sp.Basic]:
        """`values` simplified, as Derive simplifies a built-in's arguments, where
        `Environment.simplify` is supplied. A value that fails to simplify is kept.

        A number dividing a sum, as `FACTOR` writes `(x^2 + 12)/4`, is already in its
        simplified form; only the sum is simplified, so `DENOMINATOR` still sees the 4.
        """
        simplify = self.env.simplify
        if simplify is None or self.env.writing:
            return values
        simplified = []
        for value in values:
            try:
                if not value.is_Mul:
                    simplified.append(simplify(value, self.context))
                    continue
                coefficient, rest = value.as_coeff_Mul()
                if coefficient.is_Rational and rest.is_Add:
                    simplified.append(
                        _keep_coeff(coefficient, simplify(rest, self.context))
                    )
                else:
                    simplified.append(simplify(value, self.context))
            except Exception:
                simplified.append(value)
        return simplified

    def _read_in_form(
        self, name: str, values: list[sp.Basic], written: list[sp.Basic]
    ) -> list[sp.Basic]:
        """The arguments `values` of `name`, a built-in in `_SIMPLIFIED_ARGUMENTS`, as
        it reads them (see there). `written` are the arguments with a command's answer
        still wrapped, which simplifying keeps as it is."""
        if name not in DECOMPOSING:
            return self._simplified(values)
        if name == "TERMS" and len(values) > 1 or is_plain(values[0]):
            return values
        (first,) = self._simplified(written[:1])
        return [_value_of(first), *values[1:]]

    def simplified_values(self, values: list[sp.Basic]) -> list[sp.Basic]:
        return self._simplified(values)

    def _parametric(self, values: Sequence[sp.Basic]) -> bool:
        """Whether `values` mention a parameter of the definition being shown."""
        return any(
            getattr(symbol, "name", "").split(" SUB ")[0] in self.env.parameters
            for value in values
            if isinstance(value, sp.Basic)
            for symbol in _free_symbols(value)
            if not isinstance(symbol, StringLiteral)
        )

    def _unsettled(self, values: Sequence[sp.Basic], defining: bool = True) -> bool:
        """Whether a built-in answering from the form of its arguments (see `ON_FORM`)
        must stand on `values`, since what they are depends on what is still to come:
        they mention a variable bound here, whose values the binding supplies
        (`VECTOR(STRING(k), k, 1, n)` over a symbolic `n`), or any variable in a
        definition being shown, whose calls supply its parameters. Written out, no
        test is decided either. The variable of a calculus head takes no values: its
        body is worked out with it a variable, so `LIM(TERMS(u), x, 1)` takes the limit
        of each term of `u`."""
        names = {
            symbol.name.split(" SUB ")[0]
            for value in values
            if isinstance(value, sp.Basic)
            for symbol in _free_symbols(value)
            if not isinstance(symbol, StringLiteral) and hasattr(symbol, "name")
        }
        if not names:
            return False
        env = self.env
        return (
            (defining and env.defining)
            or env.writing
            or not names.isdisjoint(env.bound - env.computing)
        )

    def sets_simplified(self, values: list[sp.Basic]) -> list[sp.Basic]:
        """`values` with each set among them simplified (see `_simplified`)."""
        return [
            self._simplified([value])[0] if is_set(value) else value for value in values
        ]

    def _approximating(self, arguments: Sequence[Node]) -> list[sp.Basic]:
        """The arguments of `APPROX(u, n)`, with `u` evaluated under Approximate.

        So `u` is computed as approX computes a whole entry, carrying each step's value
        numerically (see `carried`), rather than exactly and rounded only at the end.
        The digit count is evaluated first, in the enclosing scope; if it isn't a count
        yet, `u` is evaluated at the session's digits.

        An exact closed form is the exception (see `exact_closed_form`): it is
        computed exactly and rounded once, since rounding its parts loses the digits
        that cancel: `APPROX(SUBST(SUM((-1)^n*x^(2*n)/(2*n)!, n, 0, 30), x, 20))` adds
        terms near `10^7` to `0.408095`.
        """
        operand, *rest = arguments
        digits = [_value_of(self.convert(argument)) for argument in rest]
        count = digit_count(digits[0]) if digits else None
        approximate = self.context.with_precision(Precision.APPROXIMATE, count)
        if exact_closed_form(operand, self.context):
            exact = approximate.with_precision(Precision.EXACT)
            inner = self.within(replace(self.env, context=exact))
            value = _value_of(inner.convert(operand))
            if exact_number(value):
                return [value, *digits]
        inner = self.within(replace(self.env, context=approximate))
        return [_value_of(inner.convert(operand)), *digits]

    def _power_residue(self, arguments: Sequence[Node]) -> list[sp.Basic]:
        """The arguments of `MOD(a^p, n)` or `MODS(a^p, n)`, the power reduced modulo `n`
        by modular powering if `a`, `p` and `n` are integers, `p` nonnegative: a
        primality test takes a 50-digit number to a 50-digit power."""
        power, *rest = arguments
        base, exponent = self._operands(power)
        others = [_assigned_value(_value_of(self.convert(other))) for other in rest]
        modulus = others[0] if others else sp.Integer(1)
        if (
            all(isinstance(part, sp.Integer) for part in (base, exponent, modulus))
            and exponent >= 0
            and modulus != 0
            and self.context.function("MOD") is None
            and self.context.function("MODS") is None
        ):
            reduced = sp.Integer(pow(int(base), int(exponent), int(modulus)))
        else:
            reduced = _assigned_value(_value_of(self.binary("^", [base, exponent])))
        return [reduced, *others]

    def _divisors(self, name: str, arguments: Sequence[Node]) -> list[sp.Basic]:
        """The divisors of a `SOLVE` or `SOLUTIONS` equation as written, evaluated
        here."""
        if name not in ("SOLVE", "SOLUTIONS") or not arguments:
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
            if is_simple(made) or not made.has(*CALCULUS_HEADS):
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
        definition: Definition,
        values: list,
        arguments: Sequence[Node] = (),
    ) -> sp.Basic:
        """A call of a defined function: its body evaluated in a new activation.

        Each parameter stands for its argument's value, which is evaluated again where
        the body gives values to its variables (see `Argument`). A quotient that cancels
        to fewer variables is passed cancelled (see `_cancelled`). A parameter without
        an argument takes its default, evaluated in the new activation, so it sees the
        parameters before it; one without either is a local without a value, which reads
        as the global of its name: `ACCELERATION(f, m) := f/m` applied to 6 alone is
        `6/m`. Arguments past the last parameter are evaluated and ignored, and a
        variadic function's one parameter takes them all as a vector.

        An argument that is a bare local of the caller holding a vector is shared with
        it, so the callee's element assignments change the caller's vector (see `Cell`).

        The value is the body's, or what a `RETURN` gives; an `exit` that ends nothing
        in the body ends the body. A function without a body stands, with its omitted
        arguments filled in, defaults evaluated and the rest named by the parameters:
        with `FOO(x, y := 3, z) :=`, `FOO(5)` is `FOO(5, 3, z)`.
        """
        if self.env.depth >= _NESTING:
            return self.opaque(name, values)
        if self.env.running:
            self.context.step()
            store = self.context.store
            if (
                store.passed_over is not None
                and self.env.depth - store.passed_over >= _PASSED_OVER_DEPTH
            ):
                raise PassedOverTooFar
        values = [_cancelled(value) for value in values]
        names = definition.names
        if definition.variadic and names:
            values = [Vector(*values)]
            arguments = ()
        frame = Frame(frozenset(names), activation=True)
        bindings: dict[str, sp.Basic] = {}
        supplied: dict[str, Argument] = {}
        inner = replace(
            self.env,
            bindings=bindings,
            supplied=supplied,
            frame=frame,
            bound=frozenset(),
            parameters=frozenset(),
            depth=self.env.depth + 1,
        )
        evaluator = _Evaluator(inner)
        for index, parameter in enumerate(names):
            if index < len(values):
                value = values[index]
                bindings[parameter] = value
                supplied[parameter] = Argument(self.env, value)
                frame.locals.pop(parameter, None)
                if index < len(arguments) and isinstance(value, Vector):
                    cell = self._cell_of(arguments[index])
                    if cell is not None:
                        frame.locals[parameter] = cell
                continue
            default = definition.default(index)
            if default is not None:
                value = evaluator._settled(evaluator.convert(default))
                bindings[parameter] = value
                supplied.pop(parameter, None)
                frame.locals.pop(parameter, None)
        if definition.body is None:
            completed = list(values)
            for index in range(len(values), len(names)):
                completed.append(
                    bindings.get(names[index], evaluator.symbol(names[index]))
                )
            return self.opaque(name, completed)
        settings = [parameter for parameter in names if parameter in SETTINGS]
        if not (self.env.running and _holds_program(definition.body)):
            return self._dispatched(evaluator, definition, name, values, settings)
        # A program whose flow can't be decided leaves the call standing, with nothing
        # it did kept (see `_conditional`).
        evaluator = evaluator.within(replace(evaluator.env, in_call=True))
        store = self.context.store
        kept = store.save()
        depth = self.env.depth + 1
        try:
            return self._dispatched(evaluator, definition, name, values, settings)
        except _Undecided:
            store.restore(kept)
            return self.opaque(name, values)
        except PassedOverTooFar:
            if store.passed_over is None or store.passed_over < depth:
                raise
            # This call, or one it made, went on past an undecided test and then
            # nested too deep: it stands.
            store.restore(kept)
            return self.opaque(name, values)
        finally:
            store.release(kept)
            if store.passed_over is not None and store.passed_over >= depth:
                store.passed_over = None

    def _dispatched(
        self,
        evaluator: _Evaluator,
        definition: Definition,
        name: str,
        values: list,
        settings: list[str],
    ) -> sp.Basic:
        """The value of a call of `name`, its activation made: the body evaluated in
        the way the call needs (see `_entered`)."""
        assert definition.body is not None
        if settings and self.env.running:
            return self._with_local_settings(
                settings, evaluator, definition, name, values
            )
        if self.env.defining and self.env.running and programmed(definition.body):
            return self._run_in_definition(evaluator, definition.body, name, values)
        return self._body(evaluator, definition.body, name, values)

    def _run_in_definition(
        self, evaluator: _Evaluator, body: Node, name: str, values: list
    ) -> sp.Basic:
        """A call of a program in a definition being shown: the program runs, over the
        definition's parameters standing for themselves, as the body's other parts are
        worked out. `h1(t) := INVLaplace(1/(s^2 + 3*s + 2), s, t)` is `h1(t) := #e^(-t)
        - #e^(-2*t)`.

        A program over symbolic values need not end, so it runs within a budget of
        steps (see `_SHOWN_PROGRAM_STEPS`); past it, the call stands. As anywhere in a
        definition being shown, nothing is performed: what the program assigns,
        defines or shows is undone once its value is known.
        """
        context = self.context
        store = context.store
        limit = context.step_limit
        budget = store.steps + _SHOWN_PROGRAM_STEPS
        if limit is not None and limit < budget:
            budget = limit
        running = dataclasses.replace(context, step_limit=budget)
        inner = replace(evaluator.env, defining=False, context=running)
        kept = store.save()
        try:
            return self._body(_Evaluator(inner), body, name, values)
        except StepLimitExceeded:
            if limit is not None and store.steps > limit:
                raise
            return self.opaque(name, values)
        finally:
            store.restore(kept)
            store.release(kept)

    def _body(
        self, evaluator: _Evaluator, body: Node, name: str, values: list
    ) -> sp.Basic:
        """The value of the body of a call of `name`, evaluated by `evaluator`."""
        frame = evaluator.env.frame
        assert frame is not None
        try:
            value = evaluator.statement(body)
        except _Return as returned:
            value = returned.value
        except _Exit:
            value = sympy_of(frame.last)
        except RecursionError:
            return self.opaque(name, values)
        # Rounded under the settings in force at its return, which the body may have
        # changed: a value computed at a raised PrecisionDigits keeps its digits. A
        # value computed under Approximate that returns to another precision is
        # approximated, as the end of a command at that precision would approximate it.
        returning = evaluator.context.store.context or evaluator.context
        if returning.precision is not self.context.precision:
            return carried_out(value, returning)
        return carried(value, returning)

    def _with_local_settings(
        self,
        settings: list[str],
        evaluator: _Evaluator,
        definition: Definition,
        name: str,
        values: list,
    ) -> sp.Basic:
        """The body of a call whose parameters `settings` are named after settings,
        which are local settings: each has the parameter's value while the body runs
        (a parameter without one keeps the setting's value until the body assigns
        it), and its value from before the call afterwards."""
        store = self.context.store
        before, assigned = store.context, dict(store.settings)
        previous = {setting: self.context.setting(setting) for setting in settings}
        current = self.context
        for setting in settings:
            local = evaluator._local(setting)
            written = None if local is None else _setting_written(local)
            if written is not None:
                current = current.with_setting(setting, written) or current
        store.context = current
        inner = evaluator.within(replace(evaluator.env, context=current))
        assert definition.body is not None
        try:
            return self._body(inner, definition.body, name, values)
        finally:
            if store.settings == assigned:
                store.context = before
            else:
                restored = store.context or current
                for setting, value in previous.items():
                    if value is not None:
                        restored = restored.with_setting(setting, value) or restored
                store.context = restored

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
        if not order:
            return None
        definition = self.context.function(base)
        if definition is None or not values or self.env.writing:
            return None
        parameters, body = definition.names, definition.body
        if not parameters or body is None or self.env.depth >= _NESTING:
            return None
        # A variable of its own, since a binding around the call may give the first
        # parameter's name a value: `TABLE(F'(t), t, [7, 21])` with `F(t) := t^3`.
        named = self.symbol(parameters[0])
        variable = sp.Dummy(named.name, **named.assumptions0)
        inner = replace(
            self.env,
            bindings={
                parameters[0]: variable,
                **dict(zip(parameters[1:], values[1:], strict=False)),
            },
            supplied={},
            frame=Frame(frozenset(parameters), activation=True),
            bound=frozenset({parameters[0]}),
            depth=self.env.depth + 1,
        )
        try:
            derived = [_value_of(_Evaluator(inner).statement(body)), variable]
        except _Return as returned:
            derived = [_value_of(returned.value), variable]
        except _Exit:
            return None
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

        A parameter supplied with a number or another expression that names no variable
        binds the parameter itself, and its argument is substituted into the result,
        as `_taken` does for a resumed value: with `F(x) := DIF(x^3, x)`, `F(2)`
        differentiates in `x` and is 12 at 2, the derivative of `F` there, rather than
        `3*x^2` with a name the caller never wrote. `F(x) := INT(x^2, x)` is likewise an
        antiderivative evaluated at the argument, while a definite integral or a sum
        over the parameter has no `x` left to substitute.

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
            if name in ("VECTOR", "SELECT") and arguments:
                # Nothing to range over: the call stands as written, its body not
                # evaluated, as Derive leaves it.
                return self.opaque(name, [as_written(each) for each in arguments])
            return self._unbound(name, arguments)
        body, written_index, *rest = arguments
        if name == "VECTOR" and written_index.kind is Kind.VECTOR and len(rest) > 1:
            grid = self._grid(body, written_index, rest)
            if grid is not None:
                return grid
        if _variables_of(written_index) is not None and not self._pending(written_index):
            index, names, shadowed = self._index(written_index)
        else:
            index = _value_of(self.convert(written_index))
            names, shadowed = _names_in(index), frozenset()
            if names is None:
                return self._unbound(name, arguments, index)
        inner = self._scope(name, names, shadowed)
        evaluator = self.within(inner)
        outer = evaluator if name in BINDING_THROUGHOUT else self
        limits = [_value_of(outer.convert(argument)) for argument in rest]
        taken = self._taken_by(names, shadowed)
        converted = None
        if self.env.parameters and not names.isdisjoint(self.env.parameters):
            converted = _value_of(evaluator.convert(body))
            if self.binds_parameter(index, [converted, *limits]):
                return self.opaque(name, [converted, index, *limits])
        if name in _MAPPING and not self.env.writing:
            return _over(self._mapped(name, body, index, inner, limits), taken)
        if name in _COUNTED and not self.env.writing:
            counted = self._counted(name, body, index, inner, limits)
            if counted is not None:
                return _over(counted, taken)
        if name == "LIM" and not self.env.writing:
            value = self._substituted_limit(body, index, limits, inner)
            if value is not None:
                return _over(value, taken)
        if converted is None and self._over_unknown_values(name, index, limits):
            return _over(self._standing(name, body, index, inner, limits), taken)
        if converted is None:
            converted = _value_of(evaluator.convert(body))
        values = [_assigned_value(converted), index, *limits]
        if name in _COMPUTING_BINDERS and not self.env.writing:
            values = [self.released_held(value) for value in values]
        return _over(self._worked(self.call(name, values), values), taken)

    def _scope(
        self, name: str, names: frozenset[str], shadowed: frozenset[str]
    ) -> Environment:
        """The environment of the body of a call of the binding function `name`, which
        binds `names` and shadows the parameters `shadowed` (see `_binding`)."""
        computing = self.env.computing - names
        if name in _COMPUTING_BINDERS:
            computing |= names
        return replace(
            self.env,
            bindings={n: v for n, v in self.env.bindings.items() if n not in shadowed},
            bound=self.env.bound | names,
            computing=computing,
            written=_unwritten(self.env.written, names),
        )

    def _taken_by(
        self, names: frozenset[str], shadowed: frozenset[str]
    ) -> dict[sp.Basic, sp.Basic]:
        """The substitutions into the result of a binding call that binds `names` and
        shadows `shadowed`: the values those variables have outside it."""
        return {
            **self._assigned_to(names),
            **_taken(self.env.written, names),
            **{
                self.symbol(name): self.env.bindings[name]
                for name in shadowed
                if name in self.env.bindings
            },
        }

    def _grid(self, body: Node, index: Node, bounds: list[Node]) -> sp.Basic | None:
        """`VECTOR(u, [x, y], [x0, y0], [x1, y1], [dx, dy])`: `u` over the grid of the
        points, the first variable running fastest, as nested `VECTOR`s with the last
        variable outermost. None if the bounds are not vectors of one value per
        variable."""
        if not all(name.kind is Kind.NAME for name in index.children):
            return None
        values = [_value_of(self.convert(bound)) for bound in bounds]
        count = len(index.children)
        if not all(
            isinstance(value, Vector) and len(value.args) == count for value in values
        ):
            return None
        nested = body
        call = Node(Kind.NAME, body.start, body.start, (), "VECTOR")
        for place, name in enumerate(index.children):
            limits = [
                from_sympy(value.args[place], self.context).exact for value in values
            ]
            nested = Node(Kind.CALL, body.start, body.end, (call, nested, name, *limits))
        return self.convert(nested)

    def _assigned_to(self, names: frozenset[str]) -> dict[sp.Basic, sp.Basic]:
        """The values of the session variables among the names a binding binds.

        The binding works with its variable symbolic, and the variable's value is
        substituted into the result, as for `_taken`: with `x := z*SQRT(t)`, `DIF(x^2,
        x)` is `2*z*SQRT(t)`.
        """
        env = self.env
        values = {}
        for name in names:
            if name in env.bound or name in env.bindings or name in env.expanding:
                continue
            if env.context.assigned(name) is None:
                continue
            value = _value_of(self.convert(Node(Kind.NAME, 0, 0, (), name)))
            symbol = self.symbol(name)
            if value != symbol:
                values[symbol] = value
        return values

    def _substituted_limit(
        self, body: Node, index: sp.Basic, limits: list[sp.Basic], inner: Environment
    ) -> sp.Basic | None:
        """`LIM(u, x, a)` as the value of `u` at `x = a`, where that is the limit.

        Derive's own files use `LIM` to substitute and evaluate: TENSOR.MTH's `EPS`
        takes a `LIM(VECTOR(...), dIm, n)`, FUN.MTH's `KN` a limit of an integral at a
        number, NUMBER.MTH's `BERNOULLI_POLY` a limit of a sum up to `x_ - 1` at `x`.
        As in Derive, `u` is simplified first, with `x` a variable, and the point is
        substituted into that value, which is then evaluated again: so the sum is
        summed in closed form before `x` is substituted (summed up to a number, it
        would be counted term by term), while a `VECTOR` over a symbolic count is
        generated, an `IF` decided and an element selected once the point is in.

        That is the limit wherever the value is continuous at the point (see
        `substitutable`); elsewhere, and where the value there is undefined (see
        `defined`), None is returned and the limit is taken.

        An integral in `u` is not worked out with `x` a variable, only at the point:
        in general it has no closed form, which sympy may take minutes to find out.
        INT_APPS.MTH's `SURFACE_AREA` integrates the limit of an integral. Only
        Simplify takes limits, so without `Environment.work_out` this isn't tried.

        A vector of variables takes a vector of points, substituted together where no
        point mentions a variable, when the order makes no difference. The name of a
        constant stands for the constant, so `LIM(2*true + false, [true, false], [1,
        0])` is 2. A point at infinity, or one mentioning a variable (see `MovingLimit`
        and `_iterated_limit`), is left to the limit.
        """
        written = _limit_point(index, limits)
        work_out = self.env.work_out
        if written is None or work_out is None:
            return None

        def all_but_integrals(head: sp.Basic) -> sp.Basic:
            return head if isinstance(head, sp.Integral) else work_out(head)

        try:
            simplified = replace(inner, work_out=all_but_integrals, suspended={})
            form = _factored_integrals(
                _renamed_integrals(
                    _value_of(self.within(simplified).convert(body)), written
                ),
                written,
            )
            if not substitutable(form, written):
                return None
            node = from_sympy(released(form), self.context).exact
            names = frozenset(_symbol_names(form))
            plain = replace(
                self.env, bindings={}, supplied={}, bound=self.env.bound | names
            )
            at = _value_of(self.within(_at(plain, written)).convert(node))
        except Exception:
            return None
        return at if defined(at) else None

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

        `index` is the second argument's value, if already evaluated. An assignment
        among the arguments counts as its value: `SUM(o := [])` is 0.
        """
        values = [
            index
            if i == 1 and index is not None
            else _assigned_value(_value_of(self.convert(argument)))
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

        None if the values aren't all known (a symbolic end, or a range too long to
        write out, see `_total`); the call is then made over the body as is, for sympy
        to handle. Also None if an iteration can't continue. A `SELECT` test that is
        still undecided drops its element, as in Derive.

        The terms of a `SUM` or `PRODUCT` may be truth values, which add up like
        unknowns: `SUM(PRIME(k), k, 1, 3)` is `false + 2*true`.
        """

        def at(*values: sp.Basic) -> sp.Basic:
            if self.env.running:
                self.context.step()
            variables = list(index.args) if isinstance(index, Vector) else [index]
            written = dict(zip(variables, values, strict=True))
            return _assigned_value(self.within(_at(inner, written)).convert(body))

        try:
            if name in ("ITERATE", "ITERATES"):
                if not limits or len(limits) > 2:
                    return None
                start, *count = limits
                if count and self._uncounted(count[0]):
                    count = []
                symbolic: list[sp.Basic | None] = []

                def following(values: Sequence[sp.Basic]) -> sp.Basic:
                    value = at(*values)
                    if _undefined_in(value):
                        filled = self._through_symbolic(
                            body, index, inner, values, symbolic
                        )
                        if filled is not None and _fills(self._settled(value), filled):
                            value = filled
                    return value

                return iterated(
                    name,
                    self,
                    index,
                    start,
                    count,
                    # Each iterate is simplified, as Derive simplifies it, so that an
                    # iterate equal to an earlier one is recognized as such.
                    lambda _, values: self._settled(self._iterate(following(values))),
                )
            if type(index) is not sp.Symbol:
                return None
            if name in ("SUM", "PRODUCT"):
                return _total(
                    name,
                    limits,
                    at,
                    self.context,
                    lambda: _draws(body) or _opaque(self.within(inner).convert(body)),
                )
            if name == "VECTOR":
                return Vector(*(at(value) for value in steps(limits)))
            if name == "TABLE":
                spread = body.kind in (Kind.VECTOR, Kind.NAME)
                return Vector(
                    *(table_row(value, at(value), spread) for value in steps(limits))
                )
            # A definition being shown keeps a call whose test is undecided: the test
            # may be on a parameter, which a call of the function decides.
            decide = None if self.env.defining else self._decides_released
            chosen = chosen_from(
                steps(limits), lambda value: _uncommanded(_value_of(at(value))), decide
            )
            return over_values(chosen, limits)
        except Exception:
            return None

    def _through_symbolic(
        self,
        body: Node,
        index: sp.Basic,
        inner: Environment,
        values: Sequence[sp.Basic],
        symbolic: list[sp.Basic | None],
    ) -> sp.Basic | None:
        """The iterate at `values` from the body simplified with its variable symbolic,
        as Derive simplifies it before iterating, for where the body evaluated at
        `values` is undefined: a divided difference `(F(x2) - F(x1))/(x2 - x1)` is a
        polynomial in `x1` and `x2`, defined where the iterates have converged and `x1 =
        x2`. None if that gives no defined value.

        `symbolic` caches the simplified body, computed once within a budget of work; a
        body that assigns or draws is not evaluated so.
        """
        if not symbolic:
            general = None
            if not (_assigned_names(body) or _draws(body)):
                try:
                    general = within(
                        budget.REWRITING,
                        lambda: self._settled(self.within(inner).convert(body)),
                    )
                except Exception:
                    general = None
            symbolic.append(general)
        general = symbolic[0]
        if general is None:
            return None
        variables = list(index.args) if isinstance(index, Vector) else [index]
        try:
            written = _with_elements(
                general, dict(zip(variables, values, strict=True))
            )
            value = self._settled(written)
        except Exception:
            return None
        if _undefined_in(value) or written.free_symbols & set(variables):
            return None
        return value

    def _iterate(self, value: sp.Basic) -> sp.Basic:
        """`value`, the next iterate of an `ITERATE` or `ITERATES`.

        An iterate holding an undecided conditional is carried into the next pass,
        whose conditional nests it in turn, so that the conditionals double with every
        pass: a strict command (see `Context.strict`) ends there.
        """
        if (
            self.context.strict
            and isinstance(value, sp.Basic)
            and value.atoms(Suspended)
        ):
            self.context.unsettled("an iteration carries an undecided test")
        return value

    def _mapped(
        self,
        name: str,
        body: Node,
        index: sp.Basic,
        inner: Environment,
        limits: list[sp.Basic],
    ) -> sp.Basic:
        """`MAP_LIST`, `MAP`, `EVERY` or `SOME` over the values `limits` give the
        variable `index` (see `collection_of`), evaluating the body anew at each, in
        order, with its effects.

        `MAP_LIST` gives the vector of the values, or the set over a set; `MAP` is done
        for the body's effects and gives true. `EVERY` and `SOME` decide the body as an
        `IF` test at each value and stop at the first that decides the answer, false
        for `EVERY` and true for `SOME`; past the last value the answer is the other
        truth value, unless a test was undecided. Where the values aren't known, or a
        test stays undecided, the call stands with the body as written: nothing in it
        is done.
        """
        try:
            if type(index) is not sp.Symbol:
                raise TypeError("not a variable")
            values, as_set = collection_of(limits)
        except Exception:
            return self._standing(name, body, index, inner, limits)

        def at(value: sp.Basic) -> sp.Basic:
            if self.env.running:
                self.context.step()
            evaluator = self.within(_at(inner, {index: value}))
            return _assigned_value(_value_of(evaluator.convert(body)))

        if name in ("EVERY", "SOME"):
            deciding = name == "SOME"
            undecided = False
            for value in values:
                verdict = self._decided(at(value))
                if verdict is deciding:
                    return sp.true if deciding else sp.false
                undecided = undecided or verdict is None
            if undecided:
                return self._standing(name, body, index, inner, limits)
            return sp.false if deciding else sp.true
        results = [at(value) for value in values]
        if name == "MAP":
            return sp.true
        return SetOf(*results) if as_set else Vector(*results)

    def _over_unknown_values(
        self, name: str, index: sp.Basic, limits: list[sp.Basic]
    ) -> bool:
        """Whether a binding call ranges its variable over the elements of a collection
        that is not known yet and mentions variables: `VECTOR(DIF(u_, x), u_,
        TERMS(y*x^2))` in a definition of `H(y)`, where `TERMS` waits for the call.

        The elements may then be expressions in those variables, so nothing in the body
        may treat the bound variable as independent of them (the derivative is not 0):
        the call stands with its body as written.
        """
        if name not in _OVER_ELEMENTS or len(limits) != 1:
            return False
        (collection,) = limits
        # A count, `SUM(k^2, k, m)` over 1 to `m`, has values free of any variable; the
        # parts of an expression that `TERMS(u)` or `FACTORS(u)` waits to take are
        # expressions in its variables.
        if not (
            isinstance(collection, AppliedUndef)
            and str(collection.func) in DECOMPOSING
        ):
            return False
        variables = {
            symbol
            for symbol in _free_symbols(collection)
            if not isinstance(symbol, StringLiteral)
        }
        own = set(index.args) if isinstance(index, Vector) else {index}
        return bool(variables - own)

    def _standing(
        self,
        name: str,
        body: Node,
        index: sp.Basic,
        inner: Environment,
        limits: list[sp.Basic],
    ) -> sp.Basic:
        """A call of a binding built-in left standing, its body as written: its names
        substituted, nothing in it entered, decided or done."""
        written = self.within(replace(inner, writing=True)).convert(body)
        return self.opaque(name, [_value_of(written), index, *limits])

    def _sorted(self, arguments: list[Node]) -> sp.Basic | None:
        """`SORT(v, p, x, y)`: the vector or set `v` sorted by the test `p`, in which
        `x` and `y` are bound to the two elements compared (see `sorted_by`).

        The test is evaluated for each pair with the variables substituted, as a
        `SELECT` test is for each value, so a test on elements of the elements decides.
        If a test is undecided, the call stands with `p` evaluated with the variables
        free. None if `x` and `y` aren't names; the call is then made like any other.
        """
        collection, test, *variables = arguments
        if any(variable.kind is not Kind.NAME for variable in variables):
            return None
        indexed = [self._index(variable) for variable in variables]
        symbols = [index for index, _, _ in indexed]
        if any(type(symbol) is not sp.Symbol for symbol in symbols):
            return None
        names = frozenset().union(*(named for _, named, _ in indexed))
        shadowed = frozenset().union(*(hidden for _, _, hidden in indexed))
        inner = replace(
            self.env,
            bindings={n: v for n, v in self.env.bindings.items() if n not in shadowed},
            bound=self.env.bound | names,
            written=_unwritten(self.env.written, names),
        )
        (values,) = self.sets_simplified([_value_of(self.convert(collection))])
        first, second = symbols

        def at(u: sp.Basic, v: sp.Basic) -> sp.Basic:
            written = {first: u, second: v}
            return _value_of(self.within(_at(inner, written)).convert(test))

        if isinstance(values, (Vector, SetOf)) and not self.env.writing:
            try:
                return sorted_by(list(values.args), at)
            except Exception:
                pass
        written_test = _value_of(self.within(inner).convert(test))
        return self.opaque("SORT", [values, written_test, *symbols])

    def _uncounted(self, count: sp.Basic) -> bool:
        """Whether an `ITERATE` count runs the iteration as if it were omitted.

        The manual: "If n is a positive integer, the iteration is repeated n times;
        otherwise, it is repeated until x is equal to one of its earlier values."
        SOLVE.MTH's `NEWTON(u, x, x0, n)` called without `n` leaves it a variable, and
        Derive then iterates to convergence; `ITERATES(u, n, n0, inf)` asks for the
        same. Any other count that isn't a number leaves the call standing. In a
        definition being shown, or where the count is a parameter or bound variable,
        it may yet become a number, so the call is left as written.
        """
        if count is sp.oo:
            return True
        if self.env.defining or type(count) is not sp.Symbol:
            return False
        return count.name not in self.env.bound | self.env.parameters

    def _index(self, index: Node) -> tuple[sp.Basic, frozenset[str], frozenset[str]]:
        """The variable a binding call ranges over, the names it binds, and the
        parameters it shadows.

        A parameter here is replaced by the caller's argument, if that names
        variables. Otherwise (a number or expression), the parameter itself is the
        index and shadows its argument in the body. A variable a running program
        assigned names the variables of its value the same way: TaylorODE's
        `lastvar := vars SUB o` and then `DIF(u, lastvar)`.
        """
        if index.kind is Kind.NAME:
            name = str(index.value)
            supplied = self._local(name)
            if supplied is None and self.env.running and name not in self.env.bound:
                stored = given(self.context.store, name)
                if stored is not None and stored is not CLEARED:
                    supplied = stored
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

    def _conditional(
        self, node: Node, arguments: list[Node], statement: bool = False
    ) -> sp.Basic:
        """`IF(test, a, b, c)`: evaluate only the branch the test picks.

        True picks the then clause, false the else clause (`?` if absent), and
        undecidable the unknown clause if present. A bare test is 1 or 0, so
        `SUM(IF(PRIME(n)), n, 1, 100)` counts primes.

        Without an unknown clause an undecidable `IF` is suspended whole, as in
        Derive: evaluating a branch that may never be taken could turn a guarded
        division by zero into an error, and recursion guards are such tests.

        An assignment in the test is made like any other, so ODE1.MTH's `DSOLVE1` tries
        a method once with `IF("inapplicable" = a_ := SEP(...), ..., a_, a_)` and reads
        its answer in the branches. As a `statement`, the decided test is the frame's
        last value and the chosen clause is a statement too.

        A program can't go on past a test it can't decide whose clauses steer it (a
        `RETURN`, an `exit`, an assignment): whatever it went on to compute would hold
        only if the test were false, so `stp(x) := PROG(IF(x < 0, RETURN 0), IF(x < 1,
        RETURN x), 1)` is not 1 for a symbolic `x`. The call of the defined function
        running the program then stands, with nothing the call did kept (see
        `_entered`). Some tests are passed over instead, as in Derive, which programs
        rely on: an equation, which a program asks of symbolic values meaning
        "identically" (POLY_DEGREE's `IF(u = 0, RETURN -1)` of a polynomial); a test
        that names no variable but the function's own locals without a value, as a
        parameter the call leaves out is (a tolerance `eps` not given, so the loop
        runs to its end); and one over an element that doesn't exist, which no value
        of a variable makes exist. A test whose clauses only give a value doesn't
        steer: the conditional is that value. A call that has gone on past an
        undecided equation may nest only a little deeper (see `_PASSED_OVER_DEPTH`);
        past that it stands, with nothing it did kept, as `fact(m)` with `IF(m = 0,
        RETURN 1)` does for a symbolic m rather than recurse without end.
        """
        return self._chosen(node, arguments, self._tested(arguments[0]), statement)

    def _chosen(
        self, node: Node, arguments: list[Node], verdict: bool | None, statement: bool
    ) -> sp.Basic:
        """The rest of `_conditional`, its test decided as `verdict`."""
        test_node, *branches = arguments
        test = self._undecided_test if verdict is None else None
        frame = self.env.frame
        if statement and verdict is not None and frame is not None:
            frame.last = sp.true if verdict else sp.false
        run = self.statement if statement else self.convert
        if not branches:
            if verdict is None:
                return self._suspended(node, test)
            return sp.Integer(1) if verdict else sp.Integer(0)
        if verdict is True:
            return run(branches[0])
        if verdict is False:
            if len(branches) < 2:
                return sp.nan
            return run(branches[1])
        if len(branches) > 2:
            return run(branches[2])
        steering = any(programmed(branch) for branch in branches)
        if self.context.strict and (
            steering or (statement and frame is not None and (frame.progs or frame.loops))
        ):
            self.context.unsettled("a program's test is undecided")
        if (
            steering
            and self.env.in_call
            and self.env.runs
            and _open_question(test_node, self._undecided_test, self._omitted())
        ):
            raise _Undecided
        if (
            steering
            and self.env.in_call
            and self.env.runs
            and _equations_only(test_node)
        ):
            store = self.context.store
            if store.passed_over is None:
                store.passed_over = self.env.depth
        return self._suspended(node, test)

    def _omitted(self) -> frozenset[str]:
        """The locals of the running activation that have no value: parameters the
        call left out and locals not assigned yet, which read as variables."""
        frame = self.env.frame
        if frame is None:
            return frozenset()
        return frozenset(
            name
            for name in frame.formals
            if name not in self.env.bindings
            and (name not in frame.locals or frame.locals[name].raw is CLEARED)
        )

    def _tested(self, node: Node) -> bool | None:
        """Whether the `IF` test `node` holds, or None if undecided (see `_decided`).

        A relation between two rational numbers, as a loop's tests mostly are, is
        decided from the numbers, without building the relation `_decided` would decide
        the same way (see `_truth`)."""
        if node.kind is Kind.REL and not self.env.writing:
            left_node, right_node = node.children
            word = str(node.value)
            if left_node.kind is not Kind.REL and not (
                word in COMPARING_TRUTHS
                and left_node.kind in STATEMENT_KINDS
                and right_node.kind in STATEMENT_KINDS
            ):
                left = self.side(left_node)
                return self._tested_sides(word, left, self.side(right_node))
        return self._decided(_value_of(self.convert(node)))

    def _tested_sides(self, word: str, left: sp.Basic, right: sp.Basic) -> bool | None:
        """Whether the `IF` test that is the relation `word` holds between the sides
        `left` and `right` (see `_tested`)."""
        # A command's answer compares as its value: `SOLVE(2 = 2) = true`.
        if isinstance(left, Commanded):
            left = left.args[0]
        if isinstance(right, Commanded):
            right = right.args[0]
        if isinstance(left, sp.Rational) and isinstance(right, sp.Rational):
            holds = _ORDERED_BY_SIGN[RELATIONS[word]]
            return holds(left.p * right.q - right.p * left.q)
        return self._decided(_value_of(self.related(word, left, right)))

    def _decides_released(self, test: sp.Basic) -> bool | None:
        """`Environment.decides` on `test` with its held values released, and a
        command's answer on a side of a relation taken as its value."""
        return self.env.decides(_uncommanded(self.released_held(test)))

    def _held_form(self, name: str, values: list[sp.Basic]) -> sp.Basic | None:
        """The built-in `name`, which reads the form of its argument, on a held value:
        the expression as written. A part it gives is held, unless it is a name, a
        number or a string. None if the argument is no held value, for the built-in to
        take as it is; one that has no such part, such as `LHS` of a sum, is released
        instead."""
        if len(values) != 1 or not isinstance(values[0], HeldBase):
            return None
        node = type(values[0]).node
        plain = Converter(self.context)

        def part(child: Node) -> sp.Basic:
            return _held_part(child, plain)

        match name:
            case "TERMS":
                return Vector(*(part(term) for term in _signed_terms(node)))
            case "NUMERATOR" | "DENOMINATOR":
                if node.kind is Kind.BINOP and node.value == "/":
                    return part(node.children[0 if name == "NUMERATOR" else 1])
                return values[0] if name == "NUMERATOR" else sp.Integer(1)
            case "LHS" | "RHS":
                if node.kind is Kind.REL:
                    return part(node.children[0 if name == "LHS" else 1])
            case "VARIABLES":
                names = sorted(_names_in_tree(node))
                symbols = [plain.symbol(found) for found in names]
                return self.call(name, [Vector(*symbols)])
            case "SUM?":
                return sp.true if node.kind is Kind.SUM else sp.false
            case "PRODUCT?":
                return sp.true if _is_held_product(node) else sp.false
            case "POWER?":
                powered = node.kind is Kind.BINOP and node.value == "^"
                numeric = powered and _is_number_node(node.children[1])
                return sp.true if numeric else sp.false
            case "VARIABLE?":
                return sp.true if node.kind is Kind.NAME else sp.false
            case "LOGICAL?":
                return sp.true if node.kind in _LOGICAL_NODES else sp.false
            case "STRING":
                return None
        return self.call(name, [self.released_held(values[0])])

    def _decided(self, test: sp.Basic) -> bool | None:
        """Whether an `IF` test holds, or None if undecided.

        An equation with a plus-or-minus in it holds if it does for either sign, and its
        negation if it holds for neither: JACOBI tests `MOD(t_, 2) = 1 AND MODS(b, 8) =
        +-3`. Such an equation is decided on its own wherever it is in the test. The
        infinity `±inf`, which `1/0` is, counts as such: `1/0 = inf` holds.
        """

        def signed(test: sp.Basic) -> sp.Basic:
            return test.replace(
                lambda e: isinstance(e, (sp.Eq, sp.Ne)) and e.has(PlusMinus, sp.zoo),
                self._either_sign,
            )

        test = self.released_held(test)
        self._undecided_test = test
        # Two numbers or strings compared hold no infinity or plus-or-minus.
        quick = self._truth(test)
        if quick is sp.true or quick is sp.false:
            return quick is sp.true
        if _mentions(test, sp.zoo):
            try:
                test = signed(test)
            except Exception:
                return None
        test = self._truth(test)
        if test is sp.true or test is sp.false:
            return test is sp.true
        if _mentions(test, PlusMinus):
            try:
                test = signed(test)
            except Exception:
                return None
        return self.env.decides(_numeric_test(test, self.context))

    def _either_sign(self, relation: sp.Basic) -> sp.Basic:
        """The equation or inequation `relation` decided for both signs of the
        plus-or-minus in it, or itself if undecided."""
        verdicts = set()
        for sign in (1, -1):
            signed = relation.func(
                *(
                    side.replace(
                        lambda e: isinstance(e, PlusMinus), lambda e: sign * e.args[0]
                    ).replace(sp.zoo, sign * sp.oo)
                    for side in relation.args
                )
            )
            verdicts.add(self.env.decides(_numeric_test(signed, self.context)))
        holds = True if isinstance(relation, sp.Eq) else False
        if holds in verdicts:
            return sp.true if holds else sp.false
        if verdicts == {not holds}:
            return sp.false if holds else sp.true
        return relation

    def _suspended(self, node: Node, test: sp.Basic | None = None) -> sp.Basic:
        """An undecidable conditional, suspended as a function of its variables.

        Its display shows the test as `test`, the value it was computed to, and the
        branches as written, with names substituted and nothing in them entered,
        decided or performed, as Derive shows them: with `f(x) := x^2`, `IF(f(a) = 4, 1
        + 1, 2 + 2)` is `IF(a^2 = 4, 1 + 1, 2 + 2)`, `IF(DIF(x^2, x) = 0, 1, 2)` is
        `IF(2*x = 0, 1, 2)`, and `IF(x = 0, 0, INT(t*x, t, 0, 1))` keeps its integral.
        Where that changes the test, the display as written is recorded with the one
        shown, for the answer to keep the spelling of the branches (see
        `Store.retested`). Without `test`, the test is as written with the calculus in
        it worked out.
        """
        display = _Evaluator(replace(self.env, writing=True)).convert(node)
        if test is not None and is_conditional(display):
            written = display
            display = _with_test(display, test)
            if display != written:
                self.context.store.retested[written] = display
        elif self.env.work_out is not None and display.has(*CALCULUS_HEADS):
            display = self._worked_test(display)
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

    def _worked_test(self, display: sp.Basic) -> sp.Basic:
        """The conditional `display` with the calculus heads in its test worked out
        (see `with_worked_test`)."""
        work_out = self.env.work_out
        assert work_out is not None
        return with_worked_test(display, lambda head: _worked_out(head, set(), work_out))


def _with_test(conditional: sp.Basic, test: sp.Basic) -> sp.Basic:
    """The conditional `conditional` with its test replaced by `test`, read as the
    conditional reads it (see `as_condition`) but kept as computed, and its arms as
    they are. A test that only reads as the one the conditional has leaves it as it
    is: `IF(n = 0, ...)` with `m - 3` for `n` stays `IF(m = 3, ...)`."""
    try:
        if isinstance(conditional, sp.Piecewise):
            (arm, _), *rest = conditional.args
            condition = as_condition(test)
            if sp.Piecewise((arm, condition), *rest) == conditional:
                return conditional
            with sp.evaluate(False):
                return sp.Piecewise(ExprCondPair(arm, condition), *rest)
        with sp.evaluate(False):
            return conditional.func(test, *conditional.args[1:])
    except Exception:
        return conditional


# -- compiled nodes ---------------------------------------------------------
#
# A program evaluates the same nodes over and over, and walking a node through
# `_Evaluator.convert` and the converter's dispatch costs more than the arithmetic it
# leads to. So the statements a program runs are compiled, once, into closures: one per
# node, each doing exactly what converting the node does, with the dispatch done ahead
# and its children's closures called directly. A closure takes the evaluator, so it
# reads the environment as the conversion would. `_Evaluator.statement` runs a
# statement's closures, and `_Evaluator.convert` a node's closure where it has one.
#
# Where a program runs (see `_Evaluator.native`), the closures compute with native
# values (see `rederive.engine.native`): Python integers, fractions and truth values,
# vectors written in place, and assignments of globals not yet built. A closure's value
# stands for the value the conversion gives (`sympy_of` of it is that value). Each one
# evaluates its children with their closures, in the order the conversion evaluates
# them, and where their values are native numbers, truth values or vectors it computes
# the result itself, as the integers, fractions and lists of Python do, in the cases
# where that is the result the conversion gives. Otherwise it hands the children's
# values, as sympy values, to what the conversion does with them: since a child is
# evaluated once either way, a value that turns out not to be native halfway through a
# statement is no reason to evaluate anything again. A node of a kind with no closure of
# its own is converted, its value made native if it has a native form.
#
# Anywhere else, no value is native: each closure that could make one (a number, a
# name, a value handed back by the conversion) gives the sympy value instead, and the
# closures of the constructs that are of use only natively (the program constructs,
# `IF`, `AND` and `OR`, `EVERY` and `SOME`, `|u|`, a vector written out, an element
# assignment) convert their nodes. What remains of a closure there is the dispatch
# done ahead.
#
# A variable's cell keeps a native value as it is, a local's and a global's alike (see
# `Cell` and `Store`), and so does the last value of a frame; everything else that
# reads one reads it as a sympy value, and so does a closure where values are not
# native. Each evaluator looks a name up first in the cells of its own variables
# (`_Evaluator.scope`): a call's locals, or the globals of a program run at top level.
# A native vector is kept only by the cell of the variable whose value it is, or the
# native vector it is an element of, so that writing it in place changes what
# replacing it would change: wherever else its value goes (another variable, an
# element, a returned value, any expression), it goes as the sympy vector of its
# elements at the time.

#: Whether programs run compiled; without, every node is converted as it is. The two
#: give the same results, which tests compare.
COMPILED = True

#: A compiled closure: what evaluating its node in the evaluator gives, as a native
#: value where it has one there.
Native = Callable[[_Evaluator], Any]

#: How many compiled nodes are kept in each table.
_COMPILED_KEPT = 200_000

#: The calls `_Evaluator._call` makes otherwise than by evaluating each argument and
#: handing the values to `_Evaluator.called`.
_SPECIAL_CALLS = (
    BINDING_FUNCTIONS
    | PROGRAM_FUNCTIONS
    | _SOLVING
    | {"IF", "SORT", "TRUTH_TABLE", "APPROX", "NSOLVE", "NSOLUTIONS", "MOD", "MODS"}
    | {"VARIABLES"}
)

#: A missing native value of a bound variable.
_NO_VALUE = object()

#: The largest power `^` computes natively, in bits of the base times the exponent: a
#: bound well below `POWER_BITS`, past which the conversion keeps a power standing.
_NATIVE_POWER_BITS = 4096

#: The relations, decided between native numbers.
_HOLDS: dict[str, Callable[[Any, Any], bool]] = {
    "=": operator.eq,
    "/=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def _as_native(evaluator: _Evaluator, value: Any) -> Any:
    """`value`, a sympy value, in its native form where `evaluator` computes natively."""
    return native_of(value) if evaluator.native else value


def _statement(node: Node) -> Native:
    """The closure running `node` as a statement, natively, as `_Evaluator.statement`
    does: counted as a step, its value made the frame's last value. Making it compiles
    the statement."""
    found = _STATEMENTS.get(id(node))
    if found is not None:
        return found[1]
    if len(_STATEMENTS) >= _COMPILED_KEPT:
        _STATEMENTS.clear()
    inner: Native | None = None
    if node.kind is Kind.CALL:
        head, *arguments = node.children
        if str(head.value) == "IF" and 1 <= len(arguments) <= 4:
            inner = _conditional_closure(node, statement=True)
    if (
        inner is None
        and node.kind is Kind.VECTOR
        and not any(child.kind is Kind.ELLIPSIS for child in node.children)
    ):
        elements = [_statement(child) for child in node.children]

        def listed(evaluator: _Evaluator) -> sp.Basic:
            return Vector(*(sympy_of(element(evaluator)) for element in elements))

        inner = listed
    if inner is None:
        inner = _native(node)
    run = inner

    def ran(evaluator: _Evaluator) -> Any:
        context = evaluator.context
        store = context.store
        current = store.context
        if current is not None and current is not context:
            # A statement before this one assigned a setting.
            return evaluator.statement(node)
        limit = context.step_limit
        if context.deadline is None and (limit is None or store.steps < limit):
            # A step `Context.step` counts, and that doesn't end the command.
            store.steps += 1
        else:
            context.step()
        value = run(evaluator)
        if type(value) is NativeVector:
            value = value.vector()
        evaluator.env.frame.last = value
        return value

    _STATEMENTS[id(node)] = (node, ran)
    return ran


def _native(node: Node) -> Native:
    """The closure of `node`, compiled with what it holds."""
    found = _NATIVES.get(id(node))
    if found is None:
        found = _OTHERS.get(id(node))
        if found is None:
            closure = _native_closure(node)
            table = _NATIVES
            if closure is None:
                closure = _converting(node)
                table = _OTHERS
            elif _natively_only(node):
                table = _OTHERS
            if len(table) >= _COMPILED_KEPT:
                table.clear()
            found = table[id(node)] = (node, closure)
    return found[1]


def _natively_only(node: Node) -> bool:
    """Whether the closure of `node` is of use only where values are native, and
    converts the node elsewhere: one of a program construct, `IF`, `AND` or `OR`,
    `EVERY` or `SOME`, `|u|`, a vector written out, or an element assignment."""
    kind = node.kind
    if kind in (Kind.CALL, Kind.APPLY):
        return str(node.children[0].value) in _NATIVE_CONSTRUCTS
    if kind is Kind.ASSIGN:
        return node.children[0].kind is not Kind.NAME
    return kind in (Kind.AND, Kind.OR, Kind.ABS, Kind.VECTOR)


#: The calls whose closures `_natively_only` tells apart.
_NATIVE_CONSTRUCTS = frozenset({"IF", "PROG", "LOOP", "RETURN", "EVERY", "SOME"})


def _converting(node: Node) -> Native:
    """The closure of a node converted as it is."""
    for child in node.children:
        _native(child)

    def converted(evaluator: _Evaluator) -> Any:
        return _as_native(evaluator, evaluator._converted(node))

    return converted


def _native_closure(node: Node) -> Native | None:
    """The closure of `node`, or None if it is of no kind compiled."""
    kind = node.kind
    if kind is Kind.NUMBER:
        value = numeral(node)
        native = native_of(value)

        def number(evaluator: _Evaluator) -> Any:
            return native if evaluator.native else value

        return number
    if kind is Kind.NAME:
        return _name_closure(node)
    if kind is Kind.SUM:
        return _sum_closure(node)
    if kind is Kind.PRODUCT:
        return _product_closure(node)
    if kind is Kind.BINOP:
        return _binary_closure(node)
    if kind is Kind.UNOP:
        return _prefix_closure(node)
    if kind is Kind.SUB:
        return _element_closure(node)
    if kind is Kind.REL:
        return _relation_closure(node)
    if kind in (Kind.AND, Kind.OR):
        return _sequence_closure(node)
    if kind is Kind.ABS:
        return _magnitude_closure(node)
    if kind is Kind.VECTOR and not any(
        child.kind is Kind.ELLIPSIS for child in node.children
    ):
        elements = [_native(child) for child in node.children]

        def listed(evaluator: _Evaluator) -> sp.Basic:
            if not evaluator.native:
                return evaluator._converted(node)
            return Vector(*(sympy_of(element(evaluator)) for element in elements))

        return listed
    if kind is Kind.ASSIGN:
        target = node.children[0]
        if target.kind is Kind.NAME:
            return _assignment_closure(node)
        return _element_assignment_closure(node)
    if kind in (Kind.CALL, Kind.APPLY):
        return _call_closure(node)
    return None


def _name_closure(node: Node) -> Native | None:
    """A name, read as `_Evaluator._name` reads it: a local's or a global's value as
    its cell keeps it, an argument made native, and a variable a native `EVERY` or
    `SOME` binds its value. A global's value is read where its own rules for a name
    don't apply: it has one (not `CLEARED`) and no held value to release."""
    name = str(node.value)
    if name == EXIT:
        return None

    def named(evaluator: _Evaluator) -> Any:
        env = evaluator.env
        if not evaluator.native:
            # A local's value, or a parameter's argument, as `_name` reads them where
            # nothing is substituted.
            frame = env.frame
            if frame is not None and not env.written and name not in env.bound:
                cell = frame.locals.get(name)
                if cell is None:
                    if not frame.through:
                        value = env.bindings.get(name)
                        if value is not None:
                            return value
                else:
                    raw = cell.raw
                    if raw is not CLEARED:
                        return sympy_of(raw)
            return evaluator._name(node)
        if name not in env.bound:
            # The evaluator's own variables: a call's locals, or the globals.
            cell = evaluator.scope.get(name)
            if cell is None:
                value = env.bindings.get(name)
                if value is not None:
                    if not env.written and not env.frame.through:
                        if type(value) is sp.Integer:
                            return value.p
                        return native_of(value)
                elif name not in env.expanding:
                    # A global, read from a function's body.
                    cell = evaluator.context.store.cells.get(name)
            if cell is not None:
                raw = cell.raw
                if type(raw) is int:
                    return raw
                kind = type(raw)
                if kind not in SCALARS and kind is not NativeVector:
                    # A value the general operations stored, as `_given` does.
                    raw = native_of(raw)
                    kind = type(raw)
                # A global's held value is released when read (see `_name`).
                if raw is not CLEARED and (
                    kind in SCALARS
                    or type(cell) is Cell
                    or not evaluator.context.store.held
                ):
                    if not env.written or kind in SCALARS:
                        return raw
                    if kind is NativeVector:
                        if not raw.symbolic:
                            return raw
                        raw = raw.vector()
                    return native_of(_over(raw, env.written))
        else:
            indices = evaluator.indices
            if indices is not None:
                value = indices.get(name, _NO_VALUE)
                if value is not _NO_VALUE:
                    return value
        return native_of(evaluator._name(node))

    return named


def _sum_closure(node: Node) -> Native:
    """A sum: of native numbers, their sum."""
    operators = str(node.value)
    terms = [_native(child) for child in node.children]
    negative = [False, *(sign == "-" for sign in operators)]

    def slow(evaluator: _Evaluator, values: list) -> Any:
        if evaluator.native:
            if all(type(value) in NUMBERS for value in values):
                total: Any = 0
                for value, minus in zip(values, negative, strict=True):
                    total = total - value if minus else total + value
                return rational(total) if type(total) is Fraction else total
            operands = evaluator.operands_of([sympy_of(value) for value in values])
            return native_of(evaluator.summed(operators, operands))
        for value in values:
            if not isinstance(value, sp.Rational):
                return evaluator.summed(operators, evaluator.operands_of(values))
        # Rational numbers are their own operands, and no equation.
        signs = [values[0]]
        for value, minus in zip(values[1:], negative[1:], strict=True):
            signs.append(negated(value) if minus else value)
        return added(signs)

    if len(terms) == 2:
        first, second = terms
        minus = operators == "-"

        def two(evaluator: _Evaluator) -> Any:
            left = first(evaluator)
            right = second(evaluator)
            if type(left) is int and type(right) is int:
                return left - right if minus else left + right
            return slow(evaluator, [left, right])

        return two

    def summed(evaluator: _Evaluator) -> Any:
        return slow(evaluator, [term(evaluator) for term in terms])

    return summed


def _product_closure(node: Node) -> Native:
    """A product: of native numbers, their product."""
    factors = [_native(child) for child in node.children]

    def slow(evaluator: _Evaluator, values: list) -> Any:
        if not evaluator.native:
            return evaluator.multiplied_out(evaluator.operands_of(values))
        if all(type(value) in NUMBERS for value in values):
            product: Any = 1
            for value in values:
                product = product * value
            return rational(product) if type(product) is Fraction else product
        operands = evaluator.operands_of([sympy_of(value) for value in values])
        return native_of(evaluator.multiplied_out(operands))

    if len(factors) == 2:
        first, second = factors

        def two(evaluator: _Evaluator) -> Any:
            left = first(evaluator)
            right = second(evaluator)
            if type(left) is int and type(right) is int:
                return left * right
            return slow(evaluator, [left, right])

        return two

    def multiplied(evaluator: _Evaluator) -> Any:
        return slow(evaluator, [factor(evaluator) for factor in factors])

    return multiplied


def _binary_closure(node: Node) -> Native:
    """`/`, `^` or `.`: a quotient of native numbers by a nonzero one, and a native
    number to an integer power, unless the power is zero to a power that isn't
    positive or bigger than `_NATIVE_POWER_BITS`."""
    operator_ = str(node.value)
    first, second = (_native(child) for child in node.children)

    def slow(evaluator: _Evaluator, left: Any, right: Any) -> Any:
        if not evaluator.native:
            return evaluator.binary(operator_, evaluator.operands_of([left, right]))
        operands = evaluator.operands_of([sympy_of(left), sympy_of(right)])
        return native_of(evaluator.binary(operator_, operands))

    if operator_ == "/":

        def quotient(evaluator: _Evaluator) -> Any:
            left = first(evaluator)
            right = second(evaluator)
            if type(left) in NUMBERS and type(right) in NUMBERS and right != 0:
                if type(left) is int and type(right) is int:
                    if left % right == 0:
                        return left // right
                    return Fraction(left, right)
                return rational(Fraction(left) / right)
            return slow(evaluator, left, right)

        return quotient
    if operator_ == "^":

        def power(evaluator: _Evaluator) -> Any:
            base = first(evaluator)
            exponent = second(evaluator)
            if type(exponent) is int and type(base) in NUMBERS:
                if base == 0:
                    if exponent > 0:
                        return 0
                else:
                    bits = (
                        abs(base).bit_length()
                        if type(base) is int
                        else max(abs(base.numerator), base.denominator).bit_length()
                    )
                    if bits * abs(exponent) <= _NATIVE_POWER_BITS:
                        if exponent >= 0:
                            raised = base**exponent
                        else:
                            raised = Fraction(1) / base ** (-exponent)
                        return rational(raised) if type(raised) is Fraction else raised
            return slow(evaluator, base, exponent)

        return power

    def binary(evaluator: _Evaluator) -> Any:
        left = first(evaluator)
        return slow(evaluator, left, second(evaluator))

    return binary


def _prefix_closure(node: Node) -> Native:
    """A prefix operator: the negation of a native number."""
    operator_ = str(node.value)
    (operand,) = (_native(child) for child in node.children)

    def prefixed(evaluator: _Evaluator) -> Any:
        value = operand(evaluator)
        if not evaluator.native:
            return evaluator.prefixed(operator_, evaluator.operands_of([value]))
        if operator_ == "-" and type(value) in NUMBERS:
            return -value
        operands = evaluator.operands_of([sympy_of(value)])
        return native_of(evaluator.prefixed(operator_, operands))

    return prefixed


def _element_closure(node: Node) -> Native:
    """`v SUB i`: the element of a vector at one of its places, counted from the end
    if negative (see `place_of`). At top level, a subscript that selects an element of a
    vector written out (see `_written_element`) is converted, as is any subscript
    there where values are not native."""
    base, index = (_native(child) for child in node.children)
    selected = _written_element(node)

    def element(evaluator: _Evaluator) -> Any:
        if selected is not None and evaluator.env.depth == 0:
            return _as_native(evaluator, evaluator._converted(node))
        if not evaluator.native:
            if evaluator.env.depth == 0:
                return evaluator._converted(node)
            # The index first, as `Converter._sub` evaluates it.
            at = evaluator.operand(index(evaluator))
            whole = evaluator.operand(base(evaluator))
            if isinstance(whole, Vector) and isinstance(at, sp.Integer):
                # An element of a vector at one of its places, as `element` takes it.
                place = place_of(at, whole.size())
                if place is not None:
                    return whole.item(place)
            return evaluator.element(node, whole, at)
        at = index(evaluator)
        whole = base(evaluator)
        if type(whole) is NativeVector:
            if type(at) is int:
                items = whole.items
                count = len(items)
                if 0 < at <= count:
                    return items[at - 1]
                if -count <= at < 0:
                    return items[count + at]
        whole = evaluator.operand(sympy_of(whole))
        if type(at) is int and isinstance(whole, Vector):
            count = whole.size()
            if 0 < at <= count:
                return native_of(whole.item(at - 1))
            if -count <= at < 0:
                return native_of(whole.item(count + at))
        at = evaluator.operand(sympy_of(at))
        return native_of(evaluator.element(node, whole, at))

    return element


def _decided_relation(node: Node) -> bool:
    """Whether the relation `node` is one that `_Evaluator._tested` decides from its
    sides, and that its closure relates from them."""
    left, right = node.children
    return left.kind is not Kind.REL and not (
        str(node.value) in COMPARING_TRUTHS
        and left.kind in STATEMENT_KINDS
        and right.kind in STATEMENT_KINDS
    )


def _relation_closure(node: Node) -> Native | None:
    """A relation, as a value: built from its sides (see `Converter.related`)."""
    if not _decided_relation(node):
        return None
    word = str(node.value)
    left, right = (_native(child) for child in node.children)

    def related(evaluator: _Evaluator) -> Any:
        first = left(evaluator)
        second = right(evaluator)
        return evaluator.related(
            word,
            evaluator.sided(sympy_of(first)),
            evaluator.sided(sympy_of(second)),
        )

    return related


def _test_closure(node: Node) -> Callable[[_Evaluator], bool | None]:
    """The `IF` test `node`, decided natively as `_Evaluator._tested` decides it: a
    relation between native numbers from the numbers."""
    if node.kind is Kind.REL and _decided_relation(node):
        word = str(node.value)
        holds = _HOLDS[word]
        left, right = (_native(child) for child in node.children)
        # The relation as a value, where the test is converted.
        _native(node)

        def related(evaluator: _Evaluator) -> bool | None:
            first = left(evaluator)
            second = right(evaluator)
            if type(first) is int and type(second) is int:
                return holds(first, second)
            if type(first) in NUMBERS and type(second) in NUMBERS:
                return holds(first, second)
            return evaluator._tested_sides(
                word,
                evaluator.sided(sympy_of(first)),
                evaluator.sided(sympy_of(second)),
            )

        return related
    value_of = _native(node)

    def tested(evaluator: _Evaluator) -> bool | None:
        value = value_of(evaluator)
        if type(value) is bool:
            return value
        return evaluator._decided(_value_of(sympy_of(value)))

    return tested


def _truth_closure(node: Node) -> Native:
    """An operand of `AND` or `OR`, made a truth value natively as
    `_Evaluator._sequenced` makes it (see `_Evaluator._truth`): a relation between
    native numbers decided."""
    if node.kind is Kind.REL and _decided_relation(node):
        word = str(node.value)
        holds = _HOLDS[word]
        left, right = (_native(child) for child in node.children)
        # The relation as a value, where the test is converted.
        _native(node)

        def related(evaluator: _Evaluator) -> Any:
            first = left(evaluator)
            second = right(evaluator)
            if type(first) in NUMBERS and type(second) in NUMBERS:
                return holds(first, second)
            relation = evaluator.related(
                word,
                evaluator.sided(sympy_of(first)),
                evaluator.sided(sympy_of(second)),
            )
            return native_of(evaluator._truth(_assigned_value(relation)))

        return related
    value_of = _native(node)

    def truth(evaluator: _Evaluator) -> Any:
        value = value_of(evaluator)
        if type(value) is bool:
            return value
        return native_of(evaluator._truth(_assigned_value(sympy_of(value))))

    return truth


def _sequence_closure(node: Node) -> Native:
    """`p AND q`, `p OR q`, the right operand evaluated only if the left one leaves the
    answer open (see `_Evaluator._sequenced`): of truth values, a truth value."""
    word = LOGICAL_WORDS[node.kind]
    deciding = word == "OR"
    left, right = (_truth_closure(child) for child in node.children[:2])

    def sequenced(evaluator: _Evaluator) -> Any:
        if not evaluator.native:
            return evaluator._converted(node)
        first = left(evaluator)
        if first is deciding:
            return first
        second = right(evaluator)
        if type(first) is bool and type(second) is bool:
            # The first doesn't decide, so the second does.
            return second
        return native_of(connected(word, [sympy_of(first), sympy_of(second)]))

    return sequenced


def _magnitude_closure(node: Node) -> Native:
    """`|u|`: of a native number, its absolute value."""
    (operand,) = (_native(child) for child in node.children)

    def magnitude(evaluator: _Evaluator) -> Any:
        if not evaluator.native:
            return evaluator._converted(node)
        value = operand(evaluator)
        if type(value) in NUMBERS:
            return abs(value)
        return native_of(evaluator.call("ABS", [sympy_of(value)]))

    return magnitude


def _conditional_closure(node: Node, statement: bool) -> Native:
    """`IF(test, a, b, c)`, as `_Evaluator._conditional` evaluates it: a test decided
    natively picks its branch here, and an undecided one is left to `_chosen`."""
    _, *arguments = node.children
    test_node, *branch_nodes = arguments
    tested = _test_closure(test_node)
    make = _statement if statement else _native
    branches = [make(branch) for branch in branch_nodes]
    count = len(branches)

    def conditional(evaluator: _Evaluator) -> Any:
        if not statement and not evaluator.native:
            return evaluator._converted(node)
        verdict = tested(evaluator)
        if verdict is True:
            if statement:
                evaluator.env.frame.last = True
            return branches[0](evaluator) if count else 1
        if verdict is False:
            if statement:
                evaluator.env.frame.last = False
            if not count:
                return 0
            if count < 2:
                return sp.nan
            return branches[1](evaluator)
        return native_of(evaluator._chosen(node, arguments, None, statement))

    return conditional


def _call_closure(node: Node) -> Native | None:
    """A call: the program constructs run here, `EVERY` and `SOME` over native values,
    a built-in of `_NATIVE_CALLS` computed on native values, and any other call
    without a special evaluation made on its arguments' values (see
    `_Evaluator.called`)."""
    head, *arguments = node.children
    name = str(head.value)
    if name == "IF":
        if 1 <= len(arguments) <= 4:
            return _conditional_closure(node, statement=False)
        return None
    if name == "PROG" and arguments:
        return _program_closure(node, arguments)
    if name == "LOOP" and arguments:
        return _loop_closure(node, arguments)
    if name == "RETURN" and len(arguments) == 1:
        return _return_closure(node, arguments[0])
    if name in ("EVERY", "SOME"):
        return _mapped_closure(node)
    if name in ("MOD", "MODS") and _powered(arguments):
        return None
    if name in _SPECIAL_CALLS - {"MOD", "MODS"}:
        return None
    parts = [_native(argument) for argument in arguments]
    computed = _NATIVE_CALLS.get(name)
    if name in library.FUNCTIONS:
        computed = None

    def made(evaluator: _Evaluator, values: list) -> Any:
        if not evaluator.native:
            return evaluator.called(node, values)
        if computed is not None:
            value = computed(values)
            if value is not None:
                # As `_Evaluator.called` would compute it: with no setting changed by
                # an argument, no held value about, and the built-in not redefined.
                context = evaluator.context
                store = context.store
                if (
                    (store.context is None or store.context is context)
                    and not store.held
                    and name not in store.functions
                    and name not in context.functions
                ):
                    return value
        written = [sympy_of(value) for value in values]
        return native_of(evaluator.called(node, written))

    if name == "MOD" and len(parts) == 2:
        first, second = parts

        def modulo(evaluator: _Evaluator) -> Any:
            number = first(evaluator)
            divisor = second(evaluator)
            if type(number) is int and type(divisor) is int and divisor:
                context = evaluator.context
                store = context.store
                if (
                    (store.context is None or store.context is context)
                    and not store.held
                    and name not in store.functions
                    and name not in context.functions
                ):
                    return number % divisor
            return made(evaluator, [number, divisor])

        return modulo

    def called(evaluator: _Evaluator) -> Any:
        return made(evaluator, [part(evaluator) for part in parts])

    return called


def _program_closure(node: Node, statements: Sequence[Node]) -> Native:
    """`PROG(s1, ..., sn)`, as `_Evaluator._program` runs it."""
    run = [_statement(each) for each in statements]

    def program(evaluator: _Evaluator) -> Any:
        if not evaluator.native:
            return evaluator._converted(node)
        frame = evaluator.env.frame
        frame.progs += 1
        try:
            value: Any = sp.nan
            for statement in run:
                value = statement(evaluator)
            return value
        except _Exit:
            return frame.last
        finally:
            frame.progs -= 1

    return program


def _loop_closure(node: Node, statements: Sequence[Node]) -> Native:
    """`LOOP(s1, ..., sn)`, as `_Evaluator._loop` runs it."""
    run = [_statement(each) for each in statements]

    def loop(evaluator: _Evaluator) -> Any:
        if not evaluator.native:
            return evaluator._converted(node)
        frame = evaluator.env.frame
        frame.loops += 1
        try:
            while True:
                for statement in run:
                    statement(evaluator)
        except _Exit:
            return frame.last
        finally:
            frame.loops -= 1

    return loop


def _return_closure(node: Node, operand: Node) -> Native:
    """`RETURN u`, as `_Evaluator._returned` evaluates it."""
    value_of = _native(operand)

    def returned(evaluator: _Evaluator) -> Any:
        if not evaluator.native:
            return evaluator._converted(node)
        value = value_of(evaluator)
        if type(value) in SCALARS:
            # Exact, `_settled` keeps a number or truth value as it is.
            raise _Return(sympy_of(value))
        raise _Return(evaluator._settled(sympy_of(value)))

    return returned


def _updated_natively(operator_: str, current: Any, value: Any) -> Any:
    """`_updated` of native numbers, or None if they are not both."""
    if type(current) is int and type(value) is int:
        if operator_ == ":+":
            return current + value
        if operator_ == ":-":
            return current - value
        if operator_ == ":*":
            return current * value
    if type(current) not in NUMBERS or type(value) not in NUMBERS:
        return None
    match operator_:
        case ":+":
            result = current + value
        case ":-":
            result = current - value
        case ":*":
            result = current * value
        case _:
            if value == 0:
                return None
            result = Fraction(current) / value
    return rational(result) if type(result) is Fraction else result


def _assignment_closure(node: Node) -> Native | None:
    """`name := u`, `name :+ u` and the other updates, as `_Evaluator._assign` makes
    them where a program runs: of a local in a running `PROG` or `LOOP` of a called
    function, and of a global where values are native and its right side is evaluated
    as written (see `_Evaluator._right_side`); any other assignment to a name
    converted."""
    target, *values = node.children
    name = str(target.value)
    if len(values) != 1 or name in SETTINGS:
        return None
    operator_ = str(node.value)
    current = _native(target)
    right = _native(values[0])
    # An update's operand is no value the variable takes over (see `_given`).
    shared = values[0] if operator_ == ":=" else None
    # `x := x` takes the global's value away (see `_Evaluator._cleared`).
    clears = shared is not None and _names_target(shared, target)
    add = {":+": operator.add, ":-": operator.sub, ":*": operator.mul}.get(operator_)

    def worked_out(evaluator: _Evaluator, value: Any) -> Any:
        """The value assigned, from the value `value` of the right side."""
        if type(value) not in SCALARS:
            value = _as_native(evaluator, _assigned_value(_value_of(sympy_of(value))))
        if shared is None:
            now = current(evaluator)
            updated = _updated_natively(operator_, now, value)
            if updated is None:
                updated = _as_native(
                    evaluator, _updated(operator_, sympy_of(now), sympy_of(value))
                )
            value = updated
        if type(value) not in SCALARS:
            top = node is evaluator.env.top
            value = _as_native(evaluator, evaluator._settled(value, top=top))
        return value

    def assignment(evaluator: _Evaluator) -> Any:
        env = evaluator.env
        frame = env.frame
        if not evaluator.native and (
            frame is None
            or not env.depth
            or not env.running
            or env.writing
            or env.defining
        ):
            return evaluator._converted(node)
        if name in env.bound:
            return _as_native(evaluator, evaluator._converted(node))
        if name in frame.formals:
            store = None
            cells = frame.locals
            if not (frame.progs or frame.loops or name in cells):
                return _as_native(evaluator, evaluator._converted(node))
        else:
            store = evaluator.context.store
            cells = store.cells
            if not evaluator.native or clears:
                return _as_native(evaluator, evaluator._converted(node))
            if not (env.depth or frame.activation):
                # At top level, only once the command has assigned it (see
                # `_Evaluator._right_side`).
                cell = cells.get(name)
                if cell is None or not cell.at:
                    return _as_native(evaluator, evaluator._converted(node))
        value = right(evaluator)
        cell = None
        if type(value) is int and add is not None:
            # The commonest update, of a counter or a total, where the name reads as
            # the cell's value.
            cell = cells.get(name)
            if (
                cell is not None
                and type(cell.raw) is int
                and (store is None or name not in env.expanding)
            ):
                value = add(cell.raw, value)
            else:
                cell = None
        if cell is None:
            value = worked_out(evaluator, value)
        if store is not None:
            store.version = at = store.version + 1
            cell = cells[name] = Assigned(value, at, evaluator, node, store.domains)
            return cell
        if shared is not None and isinstance(value, Vector):
            cell = evaluator._cell_of(shared)
            if cell is not None:
                cells[name] = cell
                return value
        cells[name] = Cell(value)
        return value

    return assignment


def _element_assignment_closure(node: Node) -> Native | None:
    """`v SUB i SUB j := u` and its updates, as `_Evaluator._element_assigned` makes
    them: where `v` is a local or a global holding a vector with these places, the
    element written in place (see `rederive.engine.native.NativeVector`)."""
    target, *values = node.children
    links, base = _element_links(target)
    if (
        len(values) != 1
        or base.kind is not Kind.NAME
        or any(link.kind is not Kind.SUB for link, _ in links)
    ):
        return None
    name = str(base.value)
    operator_ = str(node.value)
    indices = [_native(index) for _, index in links]
    right = _native(values[0])
    _native(target)

    def element_assigned(evaluator: _Evaluator) -> Any:
        if not evaluator.native:
            return evaluator._converted(node)
        places = [index(evaluator) for index in indices]
        value = right(evaluator)
        if type(value) not in SCALARS:
            value = native_of(_assigned_value(_value_of(sympy_of(value))))
        written = _written_in_place(evaluator, node, name, operator_, places, value)
        if written is not _NO_VALUE:
            return written
        indices_ = [evaluator.operand(sympy_of(place)) for place in places]
        stored = evaluator._element_stored(
            node, links, base, indices_, sympy_of(value)
        )
        return native_of(stored)

    return element_assigned


def _written_in_place(
    evaluator: _Evaluator,
    node: Node,
    name: str,
    operator_: str,
    places: list,
    value: Any,
) -> Any:
    """The element assignment `node` of `value` (or the update by it) at `places` in
    the local or global `name`, made in place, and its value; `_NO_VALUE` where it isn't
    so made, with nothing changed.

    A global's value is the assignment (see `_Evaluator._given`), and its cell is
    written in place only where no save point holds it, which one stamped after the
    innermost save point was taken can't be (see `Store.save`); otherwise the write
    starts a new cell."""
    env = evaluator.env
    frame = env.frame
    if env.written or name in env.bound:
        return _NO_VALUE
    store = None
    if name in frame.formals:
        cell = frame.locals.get(name)
        if cell is not None:
            whole = cell.raw
            if whole is CLEARED:
                return _NO_VALUE
        elif frame.through:
            return _NO_VALUE
        else:
            whole = env.bindings.get(name)
    else:
        store = evaluator.context.store
        cell = store.cells.get(name)
        # A held value in it would be released where the global is read.
        if cell is None or store.held:
            return _NO_VALUE
        whole = cell.raw
        if whole is CLEARED:
            return _NO_VALUE
    # The path is checked whole before anything is written.
    container = whole
    for place in places:
        if type(place) is not int or place <= 0:
            return _NO_VALUE
        if type(container) is NativeVector:
            items = container.items
        elif isinstance(container, Vector) and _unwrapped(container):
            items = container.args
        else:
            return _NO_VALUE
        if place > len(items):
            return _NO_VALUE
        element = items[place - 1]
        container = element if type(element) is NativeVector else native_of(element)
    if operator_ != ":=":
        updated = _updated_natively(operator_, container, value)
        if updated is None:
            updated = native_of(
                _updated(operator_, sympy_of(container), sympy_of(value))
            )
        value = updated
    if type(value) not in SCALARS:
        value = native_of(evaluator._settled(sympy_of(value)))
        if type(value) is NativeVector or isinstance(
            value, (Assign, FunDef, Commanded)
        ):
            return _NO_VALUE
    if store is not None:
        if cell.at <= store.saved or type(whole) is not NativeVector:
            whole = _native_vector(sympy_of(whole))
            cell = store.cells[name] = Global(whole, 0)
    elif type(whole) is not NativeVector:
        whole = _native_vector(whole)
        if cell is None:
            frame.locals[name] = Cell(whole)
        else:
            cell.raw = whole
    container = whole
    for place in places[:-1]:
        container.made = None
        element = container.items[place - 1]
        if type(element) is not NativeVector:
            element = _native_vector(element)
            container.items[place - 1] = element
        container = element
    container.put(places[-1] - 1, value)
    if store is None:
        return value
    store.version += 1
    cell.at = store.version
    return NativeAssignment(evaluator, node, places, value, store.domains)


def _native_vector(vector: Vector) -> NativeVector:
    """The vector `vector`, which `_unwrapped` holds of, as a native one."""
    return NativeVector([native_of(element) for element in vector.args])


def _mapped_closure(node: Node) -> Native | None:
    """`EVERY(u, k, ...)` and `SOME(u, k, ...)` over the values of a variable `k`, as
    `_Evaluator._binding` and `_Evaluator._mapped` evaluate them: the body decided
    natively at each value, with `k` the value as a native one."""
    head, *arguments = node.children
    name = str(head.value)
    if not 3 <= len(arguments) <= 5:
        return None
    body, written_index, *rest = arguments
    if written_index.kind is not Kind.NAME:
        return None
    variable = str(written_index.value)
    if variable in CONSTANTS or variable in ("deg", EXIT):
        return None
    for argument in arguments:
        _native(argument)
    limits = [_native(each) for each in rest]
    decide = _body_test_closure(body)
    deciding = name == "SOME"
    names = frozenset({variable})

    def scope(evaluator: _Evaluator) -> tuple | None:
        """The variable, the names bound and shadowed, the environment of the body
        and an evaluator in it, as `_Evaluator._binding` finds them; None where the
        binding is not over the variable `variable`. They depend on the environment,
        the globals (which `Store.version` tells apart) and the local of the
        variable's name, so they are remembered for an evaluator while the globals
        stay as they are and the variable is no local with a value of its own."""
        store = evaluator.context.store
        scopes = evaluator.scopes
        if scopes is None:
            scopes = evaluator.scopes = {}
        found = scopes.get(id(node))
        if (
            found is not None
            and found[0] is node
            and found[1] == store.version
            and variable not in evaluator.env.frame.locals
        ):
            return found[2]
        env = evaluator.env
        if env.parameters or evaluator._pending(written_index):
            made = None
        else:
            index, bound, shadowed = evaluator._index(written_index)
            if type(index) is not sp.Symbol or bound != names:
                made = None
            else:
                inner = evaluator._scope(name, bound, shadowed)
                made = (index, bound, shadowed, inner, evaluator.within(inner))
        scopes[id(node)] = (node, store.version, made)
        return made

    def mapped(evaluator: _Evaluator) -> Any:
        if not evaluator.native:
            return evaluator._converted(node)
        found = scope(evaluator)
        if found is None:
            return native_of(evaluator._converted(node))
        index, bound, shadowed, inner, template = found
        ends = [limit(evaluator) for limit in limits]
        taken = evaluator._taken_by(bound, shadowed)
        if len(ends) == 2 and type(ends[0]) is int and type(ends[1]) is int:
            values: Any = range(ends[0], ends[1] + 1)
        else:
            try:
                written = [_value_of(sympy_of(end)) for end in ends]
                values = (native_of(each) for each in collection_of(written)[0])
            except Exception:
                standing = evaluator._standing(name, body, index, inner, written)
                return native_of(_over(standing, taken))
        outside = evaluator.indices or {}
        undecided = False
        for value in values:
            evaluator.context.step()
            at = template.at(index, sympy_of(value))
            at.indices = {**outside, variable: value}
            verdict = decide(at, evaluator)
            if verdict is deciding:
                return deciding
            undecided = undecided or verdict is None
        if undecided:
            written = [_value_of(sympy_of(end)) for end in ends]
            standing = evaluator._standing(name, body, index, inner, written)
            return native_of(_over(standing, taken))
        return not deciding

    return mapped


def _body_test_closure(body: Node) -> Callable[[_Evaluator, _Evaluator], bool | None]:
    """The body of `EVERY` or `SOME`, evaluated by the first evaluator and decided by
    the second as `_Evaluator._mapped` decides it at each value."""
    if body.kind is Kind.REL and _decided_relation(body):
        word = str(body.value)
        holds = _HOLDS[word]
        left, right = (_native(child) for child in body.children)
        # The relation as a value, where the body is converted.
        _native(body)

        def related(at: _Evaluator, evaluator: _Evaluator) -> bool | None:
            first = left(at)
            second = right(at)
            if type(first) in NUMBERS and type(second) in NUMBERS:
                return holds(first, second)
            relation = at.related(
                word, at.sided(sympy_of(first)), at.sided(sympy_of(second))
            )
            return evaluator._decided(_assigned_value(_value_of(relation)))

        return related
    value_of = _native(body)

    def tested(at: _Evaluator, evaluator: _Evaluator) -> bool | None:
        value = value_of(at)
        if type(value) is bool:
            return value
        return evaluator._decided(_assigned_value(_value_of(sympy_of(value))))

    return tested


def _modulo(values: list) -> Any:
    """`MOD(m, n)` of integers (see `Modulo`)."""
    if len(values) == 1:
        (number,) = values
        return 0 if type(number) is int else None
    if len(values) != 2:
        return None
    number, divisor = values
    if type(number) is not int or type(divisor) is not int:
        return None
    return number % divisor if divisor else number


def _symmetric_modulo(values: list) -> Any:
    """`MODS(m, n)` of integers (see `SymmetricModulo`)."""
    if len(values) == 1:
        values = [values[0], 1]
    if len(values) != 2:
        return None
    number, divisor = values
    if type(number) is not int or type(divisor) is not int:
        return None
    if not divisor:
        return number
    return number - divisor * ((2 * number + divisor) // (2 * divisor))


def _quotient_of(values: list) -> Any:
    """The quotient `m/n` of the native numbers of `FLOOR(m, n)` or `CEILING(m, n)`,
    `n` defaulting to 1; None if they are no such numbers, or `n` is zero."""
    if len(values) == 1:
        values = [values[0], 1]
    if len(values) != 2:
        return None
    number, divisor = values
    if type(number) not in NUMBERS or type(divisor) not in NUMBERS or divisor == 0:
        return None
    return Fraction(number) / divisor


def _floored(values: list) -> Any:
    """`FLOOR(m, n)` of native numbers."""
    quotient = _quotient_of(values)
    return None if quotient is None else math.floor(quotient)


def _ceiled(values: list) -> Any:
    """`CEILING(m, n)` of native numbers."""
    quotient = _quotient_of(values)
    return None if quotient is None else math.ceil(quotient)


def _magnitude(values: list) -> Any:
    """`ABS(u)` of a native number."""
    if len(values) != 1 or type(values[0]) not in NUMBERS:
        return None
    return abs(values[0])


def _count(values: list) -> Any:
    """`DIM(v)` of a native vector."""
    if len(values) != 1 or type(values[0]) is not NativeVector:
        return None
    return len(values[0].items)


def _first(values: list) -> Any:
    """`FIRST(v)` of a native vector: `?` for an empty one."""
    if len(values) != 1 or type(values[0]) is not NativeVector:
        return None
    items = values[0].items
    return items[0] if items else sp.nan


#: The built-ins computed on native values, by name: each gives what the built-in gives
#: on the values, or None where it takes them otherwise.
_NATIVE_CALLS: dict[str, Callable[[list], Any]] = {
    "MOD": _modulo,
    "MODS": _symmetric_modulo,
    "FLOOR": _floored,
    "CEILING": _ceiled,
    "ABS": _magnitude,
    "DIM": _count,
    "FIRST": _first,
}

#: The compiled nodes, by `id`, each with its node, which keeps the `id` from being
#: reused while the entry is kept, and its closure: those `_Evaluator.convert` runs;
#: those it doesn't, of the nodes converted as they are and of those `_natively_only`
#: tells apart, which it converts itself; and those of statements.
_NATIVES: dict[int, tuple[Node, Native]] = {}
_OTHERS: dict[int, tuple[Node, Native]] = {}
_STATEMENTS: dict[int, tuple[Node, Native]] = {}


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


def branched(found: Suspended) -> sp.Piecewise | None:
    """The suspended conditional `found` as a `Piecewise` of its evaluated branches, or
    None if it can't be one.

    This is for calculus on a function defined by cases, `DIF(IF(x < 0, -x, x), x)`,
    which works on each branch where its test picks it. A conditional in a branch is
    suspended in turn. A test that assigns, or isn't a condition, gives None, as does
    a branch that fails to evaluate.
    """
    cls = type(found)
    test_node, *branches = cls.node.children[1:]
    if not 1 <= len(branches) <= 2:
        return None
    written = {
        variable: value
        for variable, value in zip(cls.variables, found.args, strict=True)
        if variable != value
    }
    env = _at(cls.environment, written) if written else cls.environment
    try:
        with _deep():
            if _defines(test_node):
                return None
            test = _value_of(_Evaluator(env).convert(test_node))
            arms = [_Evaluator(env).convert(branch) for branch in branches]
        condition = as_condition(test)
        if not isinstance(condition, sp.logic.boolalg.Boolean) or isinstance(
            condition, sp.Expr
        ):
            return None
        if len(arms) == 1:
            return sp.Piecewise((arms[0], condition))
        return sp.Piecewise((arms[0], condition), (arms[1], True))
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
    """The suspended conditional re-evaluated with its variables given these values.

    Its activation is over, so a `RETURN` or `exit` in it ends nothing: the conditional
    stays suspended.
    """
    with _deep():
        try:
            return _Evaluator(_at(cls.environment, written)).convert(cls.node)
        except (_Return, _Exit):
            raise ValueError("a control statement outside its activation") from None


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
    """`value` with `written` substituted, where possible.

    Where every key is a symbol and every part of `value` substitutes by replacing its
    operands (see `_replaceable`), `xreplace` does what `subs` would at a fraction of the
    cost: a value read under bound variables in a loop is substituted into at every
    step."""
    if not written:
        return value
    try:
        symbols = _free_symbols(value)
        if not any(symbol in symbols for symbol in written):
            return value
        if all(type(key) in _SYMBOL_TYPES for key in written) and _replaceable(value):
            return settled(value.xreplace(dict(written)))
        return settled(value.subs(written, simultaneous=True))
    except Exception:
        return value


#: The keys `_over` replaces by `xreplace`.
_SYMBOL_TYPES = frozenset({sp.Symbol, sp.Dummy})


def _replaceable(value: sp.Basic) -> bool:
    """Whether substituting a symbol into `value` by `subs` only replaces its
    occurrences and rebuilds the parts above them: no part binds a variable (an
    integral, a sum, a derivative, a lambda) or substitutes in its own way (a
    piecewise function skips its false branches, a deferred limit substitutes into its
    point only). Remembered for the values seen last, as `_free_symbols`."""
    found = _REPLACEABLE.get(value)
    if found is None:
        found = all(
            _replaces_plainly(type(part)) for part in sp.preorder_traversal(value)
        )
        if len(_REPLACEABLE) >= _FREE_SYMBOLS_KEPT:
            _REPLACEABLE.clear()
        _REPLACEABLE[value] = found
    return found


def _replaces_plainly(kind: type) -> bool:
    """Whether `subs` substitutes a symbol into an object of class `kind` by
    substituting into its operands and building it again from them."""
    found = _REPLACES_PLAINLY.get(kind)
    if found is None:
        method = getattr(kind, "_eval_subs", None)
        found = method in _PLAIN_SUBSTITUTION and not any(
            "bound_symbols" in vars(base) for base in kind.__mro__ if base is not sp.Basic
        )
        _REPLACES_PLAINLY[kind] = found
    return found


#: The `_eval_subs` methods that, for a symbol, leave the substitution to the operands
#: (`Pow` and `exp` replace a base that is the symbol, which is the same).
_PLAIN_SUBSTITUTION = frozenset(
    {
        sp.Basic._eval_subs,
        sp.Symbol._eval_subs,
        sp.Number._eval_subs,
        sp.Add._eval_subs,
        sp.Mul._eval_subs,
        sp.Pow._eval_subs,
        sp.exp._eval_subs,
        sp.core.function.Application._eval_subs,
    }
)

_REPLACES_PLAINLY: dict[type, bool] = {}
_REPLACEABLE: dict[sp.Basic, bool] = {}
forgotten(_REPLACEABLE.clear)


def _free_symbols(value: sp.Basic) -> frozenset[sp.Basic]:
    """The free symbols of `value`, remembered for the values seen last: a loop or a
    binder body reads the same long vector at every step, and finding its symbols
    walks all of it. A vector or set remembers its own, so that a new list made of
    elements of older ones, as a loop makes it, needs only their remembered symbols."""
    if isinstance(value, (Vector, SetOf)):
        state = value.__dict__
        found = state.get(_FREE_SYMBOLS_OF)
        if found is None:
            found = _symbols_of_elements(value)
            state[_FREE_SYMBOLS_OF] = found
        return found
    found = _FREE_SYMBOLS.get(value)
    if found is None:
        found = frozenset(value.free_symbols)
        if len(_FREE_SYMBOLS) >= _FREE_SYMBOLS_KEPT:
            _FREE_SYMBOLS.clear()
        _FREE_SYMBOLS[value] = found
    return found


def _symbols_of_elements(value: Vector | SetOf) -> frozenset[sp.Basic]:
    """The free symbols of a vector or set, from those of its elements."""
    found: set[sp.Basic] = set()
    try:
        for element in value.args:
            if type(element) in _SYMBOL_FREE:
                continue
            if isinstance(element, (Vector, SetOf)):
                found |= _free_symbols(element)
            else:
                found |= element.free_symbols
    except RecursionError:
        return frozenset(value.free_symbols)
    return frozenset(found)


#: The commonest elements, which have no free symbols.
_SYMBOL_FREE = frozenset(
    {
        sp.Integer,
        sp.Rational,
        type(sp.S.Zero),
        type(sp.S.One),
        type(sp.S.NegativeOne),
        type(sp.S.Half),
        StringLiteral,
    }
)


#: The memory of `_free_symbols`, and how many values it keeps.
_FREE_SYMBOLS: dict[sp.Basic, frozenset[sp.Basic]] = {}
_FREE_SYMBOLS_KEPT = 256
forgotten(_FREE_SYMBOLS.clear)

#: The attribute under which a vector or set records its free symbols (see
#: `_free_symbols`).
_FREE_SYMBOLS_OF = "_free_symbols"


#: Built-ins that rearrange the parts of their arguments, so they make no calculus head
#: the arguments don't hold already.
_STRUCTURAL = frozenset(
    "ADJOIN APPEND DELETE DIM ELEMENT FIRST INSERT REPLACE REST REVERSE".split()
)

#: Binding calls whose body is evaluated once per value when the range is known.
_COUNTED = frozenset(
    {"SUM", "PRODUCT", "VECTOR", "TABLE", "SELECT", "ITERATE", "ITERATES"}
)

#: Binding calls over a collection or range, whose body is evaluated once per value,
#: and never where the values aren't known (see `_Evaluator._mapped`).
_MAPPING = frozenset({"MAP", "MAP_LIST", "EVERY", "SOME"})

#: Binding calls whose form with one argument after the variable ranges it over the
#: elements of a collection.
_OVER_ELEMENTS = frozenset({"SUM", "PRODUCT", "VECTOR", "SELECT"})

#: Number of terms up to which a sum or product over a range is written out. Longer
#: ones are left to sympy, which has closed forms (`SUM(k^2, k, 1, 10^9)`), unless the
#: summand is opaque to it. Derive's own utility files write out at most four.
_WRITTEN_OUT = 1000

#: Number of terms up to which a sum or product whose summand sympy can't see into is
#: written out. NUMBER.MTH's `PRIMEPI(10000, 6, 1)` adds 1667 values of `PRIME`. This
#: bounds the work, as each term is evaluated on its own.
_WRITTEN_OUT_OPAQUE = 20000


def _total(
    name: str,
    limits: list[sp.Basic],
    at: Callable[[sp.Basic], sp.Basic],
    context: Context,
    opaque: Callable[[], bool],
) -> sp.Basic | None:
    """A counted `SUM` or `PRODUCT`: the terms at each value, added or multiplied.

    Each running total is `carried`, so under Approximate a long sum stays numeric.

    The values are a vector, or a range (from 1 if only one end is given). None if
    neither, or if a term is neither an expression nor a truth value (which adds like
    an unknown, see `added`), leaving the sum to sympy. A backwards
    range gives the negated sum or reciprocal product over the values stepped back
    over.

    A range of more than `_WRITTEN_OUT` values is written out only if the summand is
    `opaque`, up to `_WRITTEN_OUT_OPAQUE` values: sympy could only write it out too,
    without evaluating the terms.
    """
    backwards = False
    if len(limits) == 1 and isinstance(limits[0], (Vector, SetOf)):
        values = steps(limits)
    elif len(limits) == 3:
        low, high, step = limits
        if not (low.is_number and high.is_number and step.is_number):
            return None
        if step.is_zero is not False:
            return None
        count = sp.floor((high - low) / step) + 1
        if not count.is_Integer or count > _WRITTEN_OUT_OPAQUE:
            return None
        values = steps(limits)
    elif len(limits) in (1, 2):
        low, high = limits if len(limits) == 2 else (sp.Integer(1), *limits)
        run = _run_of(low, high, _WRITTEN_OUT)
        if run is None and _run_of(low, high, _WRITTEN_OUT_OPAQUE) and opaque():
            run = _run_of(low, high, _WRITTEN_OUT_OPAQUE)
        if run is None:
            return None
        values, backwards = run
    else:
        return None
    terms = [at(value) for value in values]
    if not all(isinstance(term, (sp.Expr, sp.logic.boolalg.Boolean)) for term in terms):
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


def _transform_where_it_converges(value: sp.Basic, arguments: list[sp.Basic]) -> sp.Basic:
    """The library's `LAPLACE(y, t, s)`, whose body is the integral of `#e^(-s*t)*y`
    over `t` from 0 to `inf`, as the function of `s` it defines.

    The Laplace transform is defined on the half-plane of `s` where its integral
    converges, so where the integral converges beyond a bound on `s`, the transform is
    its value there: `LAPLACE(SIN(t), t, s)` is `1/(s^2 + 1)`, which the integral is
    for `s > 0`, rather than the conditional the integral alone gives for an `s` of any
    sign. A condition of another shape, or none, leaves the integral's value as it is.
    """
    if len(arguments) != 3 or type(arguments[2]) is not sp.Symbol:
        return value
    if not isinstance(value, sp.Piecewise) or len(value.args) != 2:
        return value
    (transform, condition), (_, otherwise) = value.args
    if otherwise is not sp.true or not _beyond_a_bound(condition, arguments[2]):
        return value
    return transform


def _beyond_a_bound(condition: sp.Basic, variable: sp.Symbol) -> bool:
    """Whether `condition` says that `variable`, or its real part, exceeds a real bound
    free of it: a half-line, or a half-plane of the complex numbers."""
    if isinstance(condition, (sp.StrictGreaterThan, sp.GreaterThan)):
        above, bound = condition.args
    elif isinstance(condition, (sp.StrictLessThan, sp.LessThan)):
        bound, above = condition.args
    else:
        return False
    if above != variable and above != sp.re(variable):
        return False
    return not bound.has(variable) and bound.is_extended_real is True


def _draws(node: Node) -> bool:
    """Whether `node` calls `RANDOM`, so that a sum of it must draw for each term: as
    one expression over a symbolic index it would draw once."""
    if node.kind is Kind.CALL and node.children and node.children[0].value == "RANDOM":
        return True
    return any(_draws(child) for child in node.children if isinstance(child, Node))


def _opaque(summand: sp.Basic) -> bool:
    """Whether sympy can't sum `summand`, the body with its index symbolic, other than
    by writing out its terms.

    True if it contains a call or conditional left standing for a symbolic index, as
    `PRIME(n)` and `IF(n > 500, 1, 0)` are, or is a truth value.
    """
    return isinstance(summand, sp.logic.boolalg.Boolean) or bool(
        summand.atoms(AppliedUndef)
    )


def _run_of(
    low: sp.Basic, high: sp.Basic, most: int
) -> tuple[list[sp.Basic], bool] | None:
    """The values an index takes from `low` to `high`, and whether backwards.

    As in Derive, the index starts at `low` and steps by one while not past `high`,
    so a non-integer end is meaningful. `PARTS` bounds Rademacher's series by
    `SQRT(n)/LOG(n, 11)`, which for `n` of 4 is 3.459: the index runs 1, 2, 3, where
    sympy would plug the bound into Faulhaber's polynomial.

    If `high` is below `low`, it counts that many steps back from `low - 1`, which
    keeps `SUM(u, k, a, b) + SUM(u, k, b + 1, c)` equal to the sum from `a` to `c` in
    any order: `SUM(k, k, 5, 3.5)` is -4, and `SUM(k, k, 5.5, 3)` is -8 (from 4.5 and
    3.5).

    Both ends must be numbers and the count at most `most`.
    """
    if not low.is_number:
        return None
    count = sp.floor(high - low) + 1
    if not count.is_Integer or abs(count) > most:
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

    def _eval_derivative(self, symbol: sp.Basic) -> sp.Basic:
        # The answer is its value: `DIF(EXPAND(u)/2, x)` is the derivative of `u/2`.
        return self.args[0].diff(symbol)


def _uncommanded(test: sp.Basic) -> sp.Basic:
    """A test with a command's answer on a side of its relation taken as its value, so
    that `SOLVE(2 = 2) /= true` is decided as `true /= true`."""
    if not isinstance(test, Relational) or not any(
        isinstance(side, Commanded) for side in test.args
    ):
        return test
    return test.func(
        *(side.args[0] if isinstance(side, Commanded) else side for side in test.args),
        evaluate=False,
    )


def _expansion_in(node: Node) -> bool:
    """Whether `node` is a call of `EXPAND` with an amount or variables."""
    return (
        node.kind is Kind.CALL
        and node.children[0].value == "EXPAND"
        and len(node.children) > 2
    )


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


def _setting_written(value: sp.Basic) -> SettingValue | None:
    """A value assigned to a setting, as the setting takes it: a keyword, a number, or
    a vector of variables; None if it is none of those."""
    if isinstance(value, sp.Integer):
        return int(value)
    if type(value) is sp.Symbol:
        return value.name
    if isinstance(value, BooleanAtom):
        return "true" if value else "false"
    if isinstance(value, Vector) and all(
        type(e) is sp.Symbol or isinstance(e, StringLiteral) for e in value.args
    ):
        return tuple(element.name for element in value.args)
    return None


def _cancelled(value: sp.Basic) -> sp.Basic:
    """`value` as a function's argument: cancelled if it is a quotient of polynomials
    in its variables and their roots that cancels to fewer variables, and otherwise as
    it is.

    Derive simplifies an argument before the body sees it. Rederive leaves that to the
    end, except here, where the form as evaluated would mislead the body about what
    the value depends on: ODE.MTH's `SYMMETRIC_N_4` computes a scaling exponent as a
    ratio of two dot products that cancels to 1, then transforms the equation by
    `#e^(n*x)`, which with the ratio uncancelled runs away. Its `HOMOGENEOUS_4`
    computes one for `y' - SQRT(y) - x` that is a quotient in `x` and `SQRT(y)` and
    cancels to -2.
    """
    if is_simple(value) or isinstance(value, Vector) or not isinstance(value, sp.Expr):
        return value
    try:
        denominator = sp.fraction(value)[1]
        if not denominator.free_symbols or sp.count_ops(value) > _CANCELLED_SIZE:
            return value
        cancelled = cancelled_over_roots(value)
        if cancelled is not None and cancelled.free_symbols < value.free_symbols:
            return cancelled
    except Exception:
        pass
    return value


#: The largest argument, in operations, that `_cancelled` cancels.
_CANCELLED_SIZE = 200


def cancelled_over_roots(value: sp.Expr) -> sp.Expr | None:
    """`value` cancelled as a quotient of polynomials in its variables, their
    rational powers and the calls in it, each call taken for a variable of its own.

    Each variable raised to fractions with least common denominator `q` is written as
    the `q`-th power of a new variable `t` standing for its principal `q`-th root, of
    which each of those powers is a whole power: `y^(3/2)` is `t^3` for `t =
    SQRT(y)`. That holds whatever `y` is, so the cancelled quotient, with `t` put
    back, is equal to `value`: `x*(-(x + SQRT(y))/y - x/y)/(y*(x*(x + SQRT(y))/y^2 -
    x/(2*y^(3/2))))` is -2.
    """
    roots: dict[sp.Basic, int] = {}
    for found in value.atoms(sp.Pow):
        base, exponent = found.args
        if base.is_Symbol and exponent.is_Rational and not exponent.is_Integer:
            roots[base] = math.lcm(roots.get(base, 1), int(exponent.q))
    if not roots:
        return sp.cancel(value)
    standing = {base: sp.Dummy(base.name) for base in sp.ordered(roots)}
    written = value.replace(
        lambda e: e.is_Pow and e.base in roots and e.exp.is_Rational,
        lambda e: standing[e.base] ** int(e.exp * roots[e.base]),
    )
    written = written.xreplace(
        {base: standing[base] ** roots[base] for base in roots}
    )
    cancelled = sp.cancel(written)
    return cancelled.xreplace(
        {root: base ** sp.Rational(1, roots[base]) for base, root in standing.items()}
    )


def _powered(arguments: Sequence[Node]) -> bool:
    """Whether the first of `arguments` is written as a power."""
    return bool(arguments) and (
        arguments[0].kind is Kind.BINOP and str(arguments[0].value) == "^"
    )


def _limit_point(
    index: sp.Basic, limits: list[sp.Basic]
) -> dict[sp.Basic, sp.Basic] | None:
    """The values `LIM(u, index, *limits)` substitutes for its variables, or None if
    they don't match. An omitted point is zero."""
    if len(limits) > 2:
        return None
    variables = [
        CONSTANTS.get(name.name, name) if type(name) is sp.Symbol else name
        for name in (index.args if isinstance(index, Vector) else [index])
    ]
    if not limits:
        points = [sp.Integer(0)] * len(variables)
    elif not isinstance(index, Vector):
        points = [limits[0]]
    elif isinstance(limits[0], Vector):
        points = list(limits[0].args)
    else:
        return None
    if len(points) != len(variables):
        return None
    return dict(zip(variables, points, strict=True))


def limit_by_substitution(
    expression: sp.Basic, variable: sp.Basic, point: sp.Basic
) -> sp.Basic | None:
    """The limit of `expression` as `variable` approaches `point`, by substituting the
    point, or None if substitution can't be trusted to give it.

    See `substitutable` and `defined`.
    """
    written = {variable: point}
    expression = _factored_integrals(_renamed_integrals(expression, written), written)
    if not substitutable(expression, written):
        return None
    # A conditional's `?` case is a value it gives, not one the substitution makes.
    undefined = sp.Dummy("undefined")
    held = {}
    if expression.has(sp.Piecewise, sp.nan):
        try:
            held = {sp.nan: undefined}
            expression = expression.replace(
                lambda e: isinstance(e, sp.Piecewise),
                lambda e: e.xreplace(held),
            )
        except Exception:
            return None
    try:
        value = settled(expression.subs(written))
    except Exception:
        return None
    if not defined(value):
        return None
    return value.xreplace({undefined: sp.nan}) if held else value


#: The built-ins an exact closed form may call (see `exact_closed_form`).
_CLOSED_FORM_CALLS = frozenset(
    "SUBST SUM PRODUCT SQRT ABS SIGN COMB PERM EXP LN LOG SIN COS TAN ATAN ASIN ACOS "
    "SINH COSH TANH".split()
)


def exact_closed_form(node: Node, context: Context) -> bool:
    """Whether `node` is written with exact numbers, variables without a value and the
    arithmetic and elementary built-ins of `_CLOSED_FORM_CALLS` only: its exact value is
    quickly had, and where it is a number, approximating it once is right to the digits
    shown (see `exact_number`)."""
    if node.kind is Kind.NUMBER:
        return "." not in str(node.value)
    if node.kind is Kind.NAME:
        name = str(node.value)
        return name not in SETTINGS and (
            name in CONSTANT_NAMES
            or (
                context.assigned(name) is None
                and not assigned(context.store, name)
                and context.function(name) is None
            )
        )
    if node.kind is Kind.CALL:
        head, *arguments = node.children
        return str(head.value) in _CLOSED_FORM_CALLS and all(
            exact_closed_form(argument, context) for argument in arguments
        )
    if node.kind in (Kind.SUM, Kind.PRODUCT, Kind.BINOP, Kind.UNOP):
        return all(exact_closed_form(child, context) for child in node.children)
    if node.kind is Kind.POSTOP and str(node.value) == "!":
        return exact_closed_form(node.children[0], context)
    return False


def exact_number(value: sp.Basic) -> bool:
    """Whether an exact value is a number: approximated, it gives the same form."""
    return isinstance(value, sp.Expr) and not value.free_symbols


def _written_element(node: Node) -> Node | None:
    """The element a `SUB` of a vector written out selects by a number written out, or
    None. In a line as entered it is selected before anything is evaluated, so the
    other elements are not, as in Derive: `[x := w^2, y^2 - x, x := x] SUB 2` is `y^2 -
    x`. In a function's body, and passed as an argument, a vector is evaluated whole,
    its elements in order, as statements (see `_variable_of_formal`)."""
    base, index = node.children
    if base.kind is not Kind.VECTOR or index.kind is not Kind.NUMBER:
        return None
    if any(child.kind is Kind.ELLIPSIS for child in base.children):
        return None
    text = str(index.value)
    if not text.isdigit():
        return None
    place = int(text)
    if not 1 <= place <= len(base.children):
        return None
    return base.children[place - 1]


def _names_target(value: Node, target: Node) -> bool:
    """Whether the right side of an assignment is its target's bare name."""
    return value.kind is Kind.NAME and str(value.value) == str(target.value)


def substitutable(value: sp.Basic, written: Mapping[sp.Basic, sp.Basic]) -> bool:
    """Whether substituting `written` into `value` gives its limit there, as far as
    the form of `value` shows.

    The points must be finite. A point mentioning its own variable is left to
    `MovingLimit`, and one mentioning another variable of a vector to the limits taken
    one after another, as Derive takes them: `LIM(x + y, [x, y], [y, 2])` is 4. And
    `value` must be continuous at the point: every function with a jump (`SIGN`, `FLOOR`, a
    conditional's test, ...) must have its argument away from the jump there, and a
    power of a base going to zero must have a constant exponent, since `x^x` is 1 at
    zero only by convention. A symbolic point is taken to be away from every jump, as
    Derive takes `LIM(F(x), x, a)` to be `F(a)`. A numeric point must be shown away,
    so a jump is assumed where that can't be decided. A vector substituted for a
    variable selects or combines elements and is taken as it is.

    A relation is substituted into on both sides, as Derive 3.14 does: `LIM(x + 1 =
    y, x, 2)` is `3 = y`.

    An indefinite integral in a variable has no value at a point until it is worked
    out: substituted into, it would integrate in the point (but see
    `_renamed_integrals`). One in a substitution still waiting for the variable is
    bound there, and the point goes into the substitution's own point. Nor is a
    variable substituted inside an integral over `t` by a point mentioning `t`: in
    `INT(y*F(t), t)` the `y` is a constant, which `y = G(t)` is not (but see
    `_factored_integrals`).
    """
    variables = set(written)
    try:
        if any(point.has(*variables) for point in written.values()):
            return False
        for integral in _unbound_integrals(value, variables):
            if any(len(limit) == 1 and limit[0] in variables for limit in integral.limits):
                return False
            moving = {
                limit[0]
                for limit in integral.limits
                if len(limit) == 1
                and any(point.has(limit[0]) for point in written.values())
            }
            if moving and integral.function.has(*variables):
                return False
        if any(point.has(sp.oo, -sp.oo, sp.zoo, sp.nan) for point in written.values()):
            return False
        points = list(written.values())
        if any(isinstance(point, Vector) for point in points):
            return True
        if not _divisors_nonzero(value, written):
            return False
        return all(
            _away(jump.subs(written), integer)
            for jump, integer in _jumps(value, variables)
        )
    except Exception:
        return False


def _divisors_nonzero(value: sp.Basic, written: Mapping[sp.Basic, sp.Basic]) -> bool:
    """Whether every divisor in `value` that the point makes a number is shown not to be
    zero there. A quotient whose form does not cancel a common root, such as `(s -
    p)/(10*s^2 + s + 90)` at a root `p`, is 0/0 there, while its limit is not: the
    limit must then be taken. A divisor that stays symbolic is taken to be away from
    zero, as for the jumps (see `substitutable`)."""
    variables = set(written)
    for power in value.atoms(sp.Pow):
        base, exponent = power.args
        if not (exponent.is_negative and base.has(*variables)):
            continue
        at = base.subs(written)
        if at.free_symbols:
            continue
        at = sp.expand(at)
        if at.is_zero is not False or (at.has(sp.Float) and _cancelling(at)):
            return False
    return True


def _cancelling(value: sp.Basic) -> bool:
    """Whether an approximate sum is zero but for rounding: its terms cancel to within
    a millionth of their sizes."""
    try:
        terms = sp.Add.make_args(value)
        size = sum(abs(complex(sp.N(term))) for term in terms)
        return abs(complex(sp.N(value))) <= size * 1e-6
    except (TypeError, ValueError):
        return True


def _unbound_integrals(value: sp.Basic, variables: set[sp.Basic]) -> list[sp.Basic]:
    """The integrals in `value`, except those in the operand of a substitution still
    waiting for one of `variables`, where that variable is bound."""
    if isinstance(value, sp.Integral):
        return [value, *_unbound_integrals(value.function, variables)]
    if isinstance(value, MovingLimit) and value.args[1] in variables:
        return _unbound_integrals(value.args[2], variables)
    return [
        found for argument in value.args for found in _unbound_integrals(argument, variables)
    ]


def _renamed_integrals(value: sp.Basic, written: Mapping[sp.Basic, sp.Basic]) -> sp.Basic:
    """`value` with each indefinite integral over a variable of `written` whose point is
    another variable, absent from `value`, taken over that variable instead.

    Such a point only renames the integral's variable: ODE.MTH's `MISSING_Y_1` takes
    `LIM(t + INT(1/(u^2 - u - PHI(u)), u), u, y1)`, which is `t + INT(1/(y1^2 - y1 -
    PHI(y1)), y1)`.
    """
    try:
        # A substitution still waiting binds its variable, in an integral too.
        waiting = {head.args[1] for head in value.atoms(MovingLimit)}
        renaming = {
            variable: point
            for variable, point in written.items()
            if type(point) is sp.Symbol
            and point not in value.free_symbols
            and variable not in waiting
        }
        if not renaming or not value.has(sp.Integral):
            return value
        return value.replace(
            lambda e: (
                isinstance(e, sp.Integral)
                and len(e.limits) == 1
                and len(e.limits[0]) == 1
                and e.limits[0][0] in renaming
            ),
            lambda e: e.xreplace({e.limits[0][0]: renaming[e.limits[0][0]]}),
        )
    except Exception:
        return value


def moving_integrals(
    value: sp.Basic, written: Mapping[sp.Basic, sp.Basic]
) -> bool:
    """Whether `value` has an indefinite integral over `t` whose integrand mentions a
    variable of `written` with a point mentioning `t`, where substituting would change
    what is integrated (see `substitutable`)."""
    try:
        variables = set(written)
        for integral in value.atoms(sp.Integral):
            for limit in integral.limits:
                if (
                    len(limit) == 1
                    and any(point.has(limit[0]) for point in written.values())
                    and integral.function.has(*variables)
                ):
                    return True
    except Exception:
        return False
    return False


def _factored_integrals(
    value: sp.Basic, written: Mapping[sp.Basic, sp.Basic]
) -> sp.Basic:
    """`value` with the factors free of `t` taken out of each indefinite integral over
    `t` whose integrand mentions a variable of `written` with a point mentioning `t`.

    The variable is a constant of the integral, so `INT(y*F(t), t)` is `y*INT(F(t), t)`,
    into which `y = G(t)` may be substituted; a sum is integrated term by term. ODE.MTH's
    reduction of `y1^2 - PHI(x)*y*y1 - ... = 0` integrates in `y` and then puts `F(x)`
    for it.
    """
    try:
        variables = set(written)
        if not value.has(sp.Integral):
            return value

        def moving(e: sp.Basic) -> bool:
            if not (
                isinstance(e, sp.Integral)
                and len(e.limits) == 1
                and len(e.limits[0]) == 1
            ):
                return False
            (bound,) = e.limits[0]
            return e.function.has(*variables) and any(
                point.has(bound) for point in written.values()
            )

        def factored(e: sp.Basic) -> sp.Basic:
            (bound,) = e.limits[0]
            terms = []
            for term in sp.Add.make_args(e.function):
                constant, rest = term.as_independent(bound, as_Add=False)
                inner = bound if rest == 1 else sp.Integral(rest, bound)
                terms.append(constant * inner)
            return sp.Add(*terms)

        return value.replace(moving, factored)
    except Exception:
        return value


def defined(value: sp.Basic) -> bool:
    """Whether `value`, found by substituting a point, is a value there: nothing in it
    undefined (as `0/0` and `1/0` are), and no deferred substitution into a derivative,
    which stays a limit, as Derive writes a derivative it can't take at a point.

    A signed infinity is a value: an integral may run to one, and a conditional may
    give one. So is the derivative of an undefined function at a point, `V'(a*b)`.
    """
    try:
        if value.has(sp.nan, sp.zoo, sp.AccumBounds):
            return False
        return all(_primed_at(each) for each in value.atoms(sp.Subs))
    except Exception:
        return False


def _primed_at(value: sp.Subs) -> bool:
    """Whether `value` is the derivative of an undefined function at a point."""
    derivative = value.expr
    return isinstance(derivative, sp.Derivative) and isinstance(
        derivative.expr, AppliedUndef
    )


def _jumps(
    value: sp.Basic, variables: set[sp.Basic]
) -> Iterator[tuple[sp.Basic, bool]]:
    """Where `value` may jump, as expressions in `variables`: each is a jump where it
    is zero, or, with True, where it is an integer."""
    for found in sp.preorder_traversal(value):
        if not (isinstance(found, sp.Basic) and found.free_symbols & variables):
            continue
        if isinstance(found, (sp.sign, sp.Heaviside)):
            yield found.args[0], False
        elif isinstance(found, (sp.floor, sp.ceiling, sp.frac)):
            yield found.args[0], True
        elif isinstance(found, sp.Mod):
            yield found.args[0] / found.args[1], True
        elif isinstance(found, sp.Pow) and found.exp.free_symbols & variables:
            yield found.base, False
        elif isinstance(found, (sp.Piecewise, Suspended)):
            tests = (
                type(found).display if isinstance(found, Suspended) else found
            ).atoms(Relational)
            for test in tests:
                yield test.lhs - test.rhs, False


def _away(jump: sp.Basic, integer: bool) -> bool:
    """Whether `jump`, at the point, is shown away from a jump: nonzero, or not an
    integer. A symbolic one is taken to be. A number is decided exactly, by its value
    as a rational or by what is known of it, and taken to be at a jump where that
    can't be decided."""
    if not jump.is_number:
        return True
    decided = jump.is_integer if integer else jump.is_zero
    if decided is None:
        # Written another way, a sum of roots may cancel: `SQRT(2)*SQRT(8) - 4`.
        try:
            other = sp.radsimp(sp.expand(jump))
        except Exception:
            return False
        decided = other.is_integer if integer else other.is_zero
    return decided is False


def _vector_value(value: sp.Basic) -> sp.Basic:
    """`value`, unwrapped from `Commanded` if what it wraps is a vector."""
    if isinstance(value, Commanded) and isinstance(value.args[0], Vector):
        return _value_of(value)
    return value


def _equation_value(value: sp.Basic) -> sp.Basic:
    """`value`, unwrapped from `Commanded` if what it wraps is an equation."""
    if isinstance(value, Commanded) and isinstance(value.args[0], sp.Equality):
        return value.args[0]
    return value


def _refers_to(value: Node | None, name: str) -> bool:
    """Whether the tree `value` mentions the variable `name`."""
    if value is None:
        return False
    if value.kind is Kind.NAME and str(value.value) == name:
        return True
    return any(_refers_to(child, name) for child in value.children)


#: Built-ins that take an expression apart, which see a held value as written (see
#: `_Evaluator._held_form`).
_ON_HELD_FORM = frozenset(
    "TERMS NUMERATOR DENOMINATOR LHS RHS STRING VARIABLES SUM? PRODUCT? POWER? "
    "VARIABLE? LOGICAL?".split()
)

#: Built-ins that pass a held value on as it is: those that rearrange elements without
#: computing with them, and the predicates of a kind of value, which a held value is
#: not (a held sum is no number).
_HOLDING = frozenset(
    "ADJOIN APPEND INSERT REPLACE DELETE REVERSE FIRST REST ELEMENT SORT SELECT "
    "POSITION MEMBER? IDENTICAL? DIM DIMENSION DISPLAY WRITE NUMBER? INTEGER? "
    "RATIONAL? EVEN? ODD? STRING? VECTOR? MATRIX? SET? NUMBER INTEGER".split()
)

#: Binding built-ins that compute with their body's value, which releases a held value
#: in it; the others (`VECTOR`, `SUM`, ...) keep it held at each value of the variable.
_COMPUTING_BINDERS = frozenset(
    "DIF INT LIM TAYLOR ROOT_OF ROOT_SUM SUBST_DIFF".split()
)

_LOGICAL_NODES = frozenset(
    (Kind.REL, Kind.NOT, Kind.AND, Kind.OR, Kind.XOR, Kind.IMP, Kind.IFF)
)


def _filled(
    node: Node, value_of: Callable[[str], sp.Basic | None], context: Context
) -> Node:
    """The quoted tree `node` with each name `value_of` gives a value replaced by the
    tree of that value (see `_value_tree`). A call's name, an assignment's target and a
    definition are left alone."""
    return _fill(node, value_of, context)[0]


def _fill(
    node: Node, value_of: Callable[[str], sp.Basic | None], context: Context
) -> tuple[Node, bool | int]:
    """`_filled`, and whether the tree is a value put in for a name, or a product or
    quotient whose sign was taken out of such a value (see `_signed_out`)."""
    if node.kind is Kind.NAME:
        value = value_of(str(node.value))
        if value is None:
            return node, False
        return _value_tree(value, context), True
    if not node.children or node.kind in (Kind.FUNDEF, Kind.DOMAIN):
        return node, False
    kept = 1 if node.kind in (Kind.CALL, Kind.APPLY, Kind.FUNCPOW, Kind.ASSIGN) else 0
    children = list(node.children[:kept])
    inserted: list[bool | int] = [False] * kept
    for child in node.children[kept:]:
        filled, put = _fill(child, value_of, context)
        children.append(filled)
        inserted.append(put)
    if all(new is old for new, old in zip(children, node.children, strict=True)):
        return node, False
    node = replace(node, children=tuple(children))
    if node.kind in (Kind.SUM, Kind.PRODUCT) and any(inserted):
        node = _spliced(node, inserted)
    if node.kind in (Kind.PRODUCT, Kind.BINOP) and inserted[0]:
        return _signed_out(node, inserted[0] is _SIGNED_OUT)
    return node, False


#: What `_fill` says of a product or quotient whose sign it took out (see `_signed_out`).
_SIGNED_OUT = 2


def _signed_out(node: Node, signed: bool) -> tuple[Node, bool | int]:
    """The product or quotient `node`, whose first operand was put in for a name (or
    `signed`, had its sign taken out), with the sign of a negative number put in taken
    out of it, as Derive writes it: `c*x/11` with -4 for `c` is `-4*x/11`, so that `x^2
    + c*x/11` is `x^2 - 4*x/11`. Whether the sign was taken out."""
    first = node.children[0]
    if node.kind is Kind.BINOP and node.value != "/":
        return node, False
    if not (first.kind is Kind.UNOP and first.value == "-") or not (
        signed or _is_ratio_node(first.children[0])
    ):
        return node, False
    unsigned = replace(node, children=(first.children[0], *node.children[1:]))
    return Node(Kind.UNOP, node.start, node.end, (unsigned,), "-"), _SIGNED_OUT


def _is_ratio_node(node: Node) -> bool:
    """Whether `node` is a numeral or a quotient of numerals, as a rational number is
    written."""
    if node.kind is Kind.BINOP and node.value == "/":
        return all(child.kind is Kind.NUMBER for child in node.children)
    return node.kind is Kind.NUMBER


def _value_tree(value: sp.Basic, context: Context) -> Node:
    """The tree a value goes into a held expression as: a held value's own, any other
    value's as it is written."""
    if isinstance(value, HeldBase):
        return type(value).node
    return written_tree(value, context)


def _spliced(node: Node, inserted: Sequence[bool | int]) -> Node:
    """The sum or product `node` with a sum or product put in for a name as a term or
    factor joined into it, as Derive writes it: `res_ + s1_` with `1 + 2` for `res_` is
    `1 + 2 + 3`, not `(1 + 2) + 3`. A term after a minus stays parenthesized, and a
    negative term added is subtracted: `x + c` with `-2/11` for `c` is `x - 2/11`."""
    kind = node.kind
    if kind is Kind.PRODUCT and inserted[-1]:
        last = node.children[-1]
        if last.kind is Kind.BINOP and last.value == "/":
            # `2*c` with 121/16 for `c` is `2*121/16`.
            numerator, denominator = last.children
            factors = replace(
                node, children=(*node.children[:-1], numerator), surface=None
            )
            factors = _spliced(factors, [*inserted[:-1], True])
            return Node(Kind.BINOP, node.start, node.end, (factors, denominator), "/")
    signs = str(node.value) if kind is Kind.SUM else "*" * (len(node.children) - 1)
    children: list[Node] = []
    gaps: list[str] = []
    for index, (child, put) in enumerate(zip(node.children, inserted, strict=True)):
        sign = "+" if index == 0 else signs[index - 1]
        joined = put and child.kind is kind and sign in "+*"
        parts = list(child.children) if joined else [child]
        inner = (
            (str(child.value) if kind is Kind.SUM else "*" * (len(parts) - 1))
            if joined
            else ""
        )
        if index:
            gaps.append(sign)
        children.append(parts[0])
        for gap, part in zip(inner, parts[1:], strict=True):
            gaps.append(gap)
            children.append(part)
    if kind is Kind.SUM:
        for index in range(1, len(children)):
            child = children[index]
            if (
                gaps[index - 1] == "+"
                and child.kind is Kind.UNOP
                and child.value == "-"
            ):
                gaps[index - 1] = "-"
                children[index] = child.children[0]
    return replace(node, children=tuple(children), value="".join(gaps), surface=None)


def _held_part(node: Node, plain: Converter) -> sp.Basic:
    """A part of a held value: held, unless it is a name, a number or a string."""
    if node.kind in (Kind.NAME, Kind.NUMBER, Kind.STRING) or _is_number_node(node):
        return plain.convert(node)
    return held(node)


def _is_number_node(node: Node) -> bool:
    """Whether `node` is a numeral, signed or not."""
    if node.kind is Kind.UNOP and node.value in ("-", "+"):
        node = node.children[0]
    return node.kind is Kind.NUMBER


def _signed_terms(node: Node) -> list[Node]:
    """The terms of a held sum, each subtracted one negated; any other tree is its one
    term."""
    if node.kind is not Kind.SUM:
        return [node]
    terms = [node.children[0]]
    for sign, child in zip(str(node.value), node.children[1:], strict=True):
        if sign == "-":
            child = Node(Kind.UNOP, child.start, child.end, (child,), "-")
        terms.append(child)
    return terms


def _is_held_product(node: Node) -> bool:
    """Whether a held tree is written as a product: `a*b`, `a/b`, `-a`."""
    if node.kind is Kind.PRODUCT or (node.kind is Kind.BINOP and node.value == "/"):
        return True
    return node.kind is Kind.UNOP and node.value == "-" and not _is_number_node(node)


def _mentioned(node: Node) -> tuple[frozenset[str], frozenset[str]] | None:
    """The variable names and the names of the functions called in `node`; None for a
    tree that holds an assignment, a definition or a quote, whose effects may reach
    anything."""
    return _remembered(_MENTIONED, node, _mentioned_in)


def _mentioned_in(node: Node) -> tuple[frozenset[str], frozenset[str]] | None:
    names: set[str] = set()
    calls: set[str] = set()
    pending = [node]
    while pending:
        each = pending.pop()
        if each.kind is Kind.NAME:
            name = str(each.value)
            if name not in CONSTANTS:
                names.add(name)
            continue
        if each.kind in (Kind.CALL, Kind.APPLY):
            calls.add(str(each.children[0].value))
            pending.extend(each.children[1:])
            continue
        if each.kind in (Kind.ASSIGN, Kind.FUNDEF, Kind.QUOTE):
            return None
        pending.extend(each.children)
    return frozenset(names), frozenset(calls)


def _names_in_tree(node: Node) -> set[str]:
    """The variable names a tree mentions, call names left out."""
    if node.kind is Kind.NAME:
        name = str(node.value)
        return set() if name in CONSTANTS else {name}
    children = node.children
    if node.kind in (Kind.CALL, Kind.APPLY):
        children = children[1:]
    found: set[str] = set()
    for child in children:
        found |= _names_in_tree(child)
    return found


def _numeric_test(test: sp.Basic, context: Context) -> sp.Basic:
    """Under Approximate, a relation between numbers with each side approximated.

    The user asked for numbers, so a test between them is decided by them, as Derive
    does in floating point: an exact decision of `ABS(a + SQRT(d)) >= ABS(a -
    SQRT(d))` for complex radicals takes seconds.
    """
    if (
        context.precision is not Precision.APPROXIMATE
        or not isinstance(test, Relational)
        or not all(isinstance(side, sp.Expr) and side.is_number for side in test.args)
    ):
        return test
    try:
        return test.func(*(carried(side, context) for side in test.args))
    except Exception:
        return test


def _defines(node: Node) -> bool:
    """Whether `node` contains an assignment, a definition, a program construct or a
    random number, which evaluating it anew may give a new value or effect; remembered
    per tree, since a stored value is asked each time it is read."""
    return _remembered(_DEFINING, node, _defining)


def _defining(node: Node) -> bool:
    if node.kind in (Kind.ASSIGN, Kind.FUNDEF):
        return True
    if node.kind is Kind.CALL and str(node.children[0].value) in _ACTIVE:
        return True
    return any(_defining(child) for child in node.children)


#: Calls whose evaluation does something, so that their value must not be reused.
_ACTIVE = PROGRAM_FUNCTIONS | {"RANDOM"}


def _mentions(value: sp.Basic, kind: object) -> bool:
    """`value.has(kind)`, without walking the sides of a relation that `is_simple`
    finds simple, which hold numbers, strings, truth values and variables only: a loop
    tests a long list it builds at every pass."""
    if isinstance(value, Relational):
        return any(_mentions(side, kind) for side in value.args)
    if is_simple(value):
        return False
    return value.has(kind)


#: Whether a relation holds between two numbers, from the sign of their difference.
_ORDERED_BY_SIGN: dict[type, Callable[[int], bool]] = {
    sp.Eq: lambda d: d == 0,
    sp.Ne: lambda d: d != 0,
    sp.StrictLessThan: lambda d: d < 0,
    sp.LessThan: lambda d: d <= 0,
    sp.StrictGreaterThan: lambda d: d > 0,
    sp.GreaterThan: lambda d: d >= 0,
}


def _updated(operator: str, current: sp.Basic, value: sp.Basic) -> sp.Basic:
    """The new value an update `x :+ u` (or `:-`, `:*`, `:/`) gives `x`, `u` taken
    whole: `x :* 1 + 0*#i` multiplies by `1 + 0*#i`."""
    match operator:
        case ":+":
            return added([current, value])
        case ":-":
            return added([current, negated(value)])
        case ":*":
            return multiplied([current, value])
    return divided(current, value)


def _element_links(target: Node) -> tuple[list[tuple[Node, Node]], Node]:
    """The subscript links of the target of an element assignment, outermost first,
    each with its index, and the variable they subscript."""
    links: list[tuple[Node, Node]] = []
    base = target
    while base.kind is Kind.SUB or (
        base.kind is Kind.INFIX and (base.value == "ROW" or is_chain(base))
    ):
        links.append((base, base.children[1]))
        base = base.children[0]
    links.reverse()
    return links, base


def _linked(link: Node, base: sp.Basic, index: sp.Basic) -> sp.Basic:
    """`base` with the subscript link `link` to `index`, as the target of an element
    assignment is written back."""
    if link.kind is Kind.SUB:
        return Subscript(base, index)
    match str(link.value):
        case "ROW":
            return RowsOf(base, index)
        case "COL":
            return ColumnsOf(base, index)
    return Deeper(base, index, sp.Integer(depth(link)))


def _element_at_path(value: sp.Basic, indices: Sequence[sp.Basic]) -> sp.Basic | None:
    """The element of `value` at the path of `indices`, the last of which may be a
    vector of indices (giving the vector of those elements); None if there is none."""
    for position, index in enumerate(indices):
        if not isinstance(value, Vector):
            return None
        count = len(value.args)
        if isinstance(index, Vector) and position == len(indices) - 1:
            places = [place_of(each, count) for each in index.args]
            if None in places:
                return None
            return Vector(*(value.args[place] for place in places))
        place = place_of(index, count)
        if place is None:
            return None
        value = value.args[place]
    return value


def _replaced_at(
    value: sp.Basic, indices: Sequence[sp.Basic], new: sp.Basic
) -> sp.Basic | None:
    """`value` with its element at the path of `indices` replaced by `new`, or with a
    vector of indices last, the listed elements by those of the vector `new`; None if
    the path isn't in `value`, or `new` doesn't match. A vector is never extended.

    An index is a position counted from 1: Derive's `v SUB (-1) := u` leaves `v` as it
    is, and a program that steps an index down past the first row, as a Gauss-Jordan
    elimination does, relies on that."""
    if not indices:
        return new
    if not isinstance(value, Vector):
        return None
    items = list(value.args)
    index, *rest = indices
    if isinstance(index, Vector) and not rest:
        if not isinstance(new, Vector) or len(new.args) != len(index.args):
            return None
        for each, element in zip(index.args, new.args, strict=True):
            place = _assigned_place(each, len(items))
            if place is None:
                return None
            items[place] = element
        return vector_from(items, [value], new.args)
    place = _assigned_place(index, len(items))
    if place is None:
        return None
    inner = _replaced_at(items[place], rest, new)
    if inner is None:
        return None
    items[place] = inner
    return vector_from(items, [value], [inner])


def _assigned_place(index: sp.Basic, count: int) -> int | None:
    """The zero-based place an element assignment at `index` replaces among `count`
    elements; None if `index` isn't a positive integer up to `count`."""
    if isinstance(index, sp.Integer) and index > 0:
        return place_of(index, count)
    return None


def _assigned_value(value: sp.Basic) -> sp.Basic:
    """The value of an assignment used as an operand or argument, else `value`.

    An assignment within an expression stands for the value it assigns: `DIF(c := x^2,
    x)` is `2*x`, and `2*(c := 3)` is 6. A definition stands for its body likewise:
    `DIF(F(u) := p/u, u)` is `-p/u^2`. So does an assignment that is an element of a
    vector used so: with `v_ := [vr := DIF(r, t), vp := r*DIF(phi, t)]`, `v_*a_` is the
    product of the values.
    """
    if not isinstance(value, (Assign, FunDef, Vector)):
        return value
    if isinstance(value, Assign) and len(value.args) == 3:
        return value.args[2]
    if isinstance(value, FunDef) and len(value.args) == 3:
        return value.args[2]
    if isinstance(value, Vector) and not _unwrapped(value):
        return Vector(*(_assigned_value(element) for element in value.args))
    return value


def _unwrapped(value: Vector | SetOf) -> bool:
    """Whether no element of the vector or set `value` is an assignment, a definition
    or a command's answer, which its use as a value unwraps; remembered on the value,
    which a program may read element by element as often as it has elements."""
    found = value.__dict__.get(_UNWRAPPED)
    if found is None:
        found = value.__dict__.get(SIMPLE) or not any(
            isinstance(element, (Assign, FunDef, Commanded)) for element in value.args
        )
        value.__dict__[_UNWRAPPED] = found
    return found


#: The attribute `_unwrapped` remembers its answer in.
_UNWRAPPED = "_unwrapped"


def value_of(value: sp.Basic) -> sp.Basic:
    """`value`, unwrapped from `Commanded` if wrapped (see `_value_of`), for callers
    that take an evaluated value as it stands."""
    return _value_of(value)


def _value_of(value: sp.Basic) -> sp.Basic:
    """`value`, unwrapped from `Commanded` if wrapped.

    The prime powers of `FACTORS(FACTOR(360))` are values like any other once used:
    `PRIME` of its first element is false, that element being 8. The elements of a
    vector or set are unwrapped too, so `APPEND([SOLUTIONS(x^2 = 4, x), [7]])` sees two
    vectors.
    """
    if isinstance(value, (Vector, SetOf)):
        if _unwrapped(value) or not any(
            isinstance(element, Commanded) for element in value.args
        ):
            return value
        return type(value)(*(_value_of(element) for element in value.args))
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

    Each distinct conditional is worked out once in a display, however often it
    appears in it or in the displays of others: a program that accumulates
    conditionals over undecided tests builds displays that contain the earlier ones
    many times over.
    """
    try:
        if not value.atoms(Suspended, Commanded):
            return settled(value)
    except Exception:
        return value
    if _RELEASING.get() is not None:
        return _releasing(value)
    token = _RELEASING.set(_Releasing())
    try:
        return _releasing(value)
    finally:
        _RELEASING.reset(token)


def _releasing(value: sp.Basic) -> sp.Basic:
    try:
        found = within(budget.REWRITING, lambda: settled(_released(value)))
        if found is not None:
            return found
    except Exception:
        pass
    # Rebuilding the parents of what was released can fail where the display is
    # rejected as an operand (`1 > DIMENSION(...)` around a conditional), so they are
    # rebuilt as they stand, without evaluation. So are they where evaluating them runs
    # out of the work of an optional rewrite: sympy folds nested conditionals into one,
    # simplifying their conditions, which on large ones may not end.
    try:
        return _written(value)
    except Exception:
        return value


@dataclass
class _Releasing:
    """What the outermost `released` that is running has worked out, by the
    conditional or command's answer it was worked out for."""

    #: The displays worked out with their parents evaluated.
    shown: dict[sp.Basic, sp.Basic] = field(default_factory=dict)
    #: The expressions rebuilt as they stand, by `_written`.
    written: dict[sp.Basic, sp.Basic] = field(default_factory=dict)


#: The work of the outermost `released` that is running, None outside one.
_RELEASING: contextvars.ContextVar[_Releasing | None] = contextvars.ContextVar(
    "_RELEASING", default=None
)


def _released(value: sp.Basic) -> sp.Basic:
    return value.replace(lambda found: isinstance(found, Suspended | Commanded), _shown)


def _shown(found: Suspended | Commanded) -> sp.Basic:
    shown = _RELEASING.get().shown
    display = shown.get(found)
    if display is None:
        display = shown[found] = released(_display(found))
    return display


def _display(found: Suspended | Commanded) -> sp.Basic:
    """The display of `found` with its arguments substituted, before release."""
    if isinstance(found, Commanded):
        return found.args[0]
    cls = type(found)
    if found.args == cls.variables:
        return cls.display
    return _over(cls.display, dict(zip(cls.variables, found.args, strict=True)))


def _written(value: sp.Basic) -> sp.Basic:
    """`value` released with nothing rebuilt by evaluation.

    A conditional already worked out with its parents evaluated keeps that display;
    any other is replaced by its display as written. Sympy folds a conditional in the
    test of another into it (rewriting it to an if-then-else over the sets each test
    holds on) even with evaluation off, so a conditional is put together from its
    branches directly. Under evaluation off, the solving of inequalities that folding
    does builds ever larger unsimplified expressions, which sympy then simplifies for
    seconds before it runs out of stack.
    """
    releasing = _RELEASING.get()
    written = releasing.written

    def rebuilt(node: sp.Basic) -> sp.Basic:
        found = written.get(node)
        if found is not None:
            return found
        if isinstance(node, Suspended | Commanded):
            found = releasing.shown.get(node)
            if found is None:
                found = rebuilt(_display(node))
        elif not node.args:
            return node
        else:
            args = tuple(rebuilt(arg) for arg in node.args)
            if args == node.args:
                found = node
            elif isinstance(node, ExprCondPair):
                found = sp.Tuple.__new__(ExprCondPair, *args)
            elif isinstance(node, sp.Piecewise):
                found = sp.Basic.__new__(sp.Piecewise, *args)
            else:
                with sp.evaluate(False):
                    found = node.func(*args)
        written[node] = found
        return found

    return rebuilt(value)


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


def _real_end(value: sp.Basic) -> bool:
    """Whether `value` can end a search interval: a real number or an infinity."""
    if value in (sp.oo, -sp.oo):
        return True
    return isinstance(value, sp.Expr) and value.is_number and value.is_real is True


def _names_in(value: sp.Basic) -> frozenset[str] | None:
    """The variables a supplied argument names, or None if it isn't names. A string
    names the variable it spells, as in a program whose variables are `"y0"`, `"y1"`,
    ..."""
    if type(value) in (sp.Symbol, StringLiteral):
        return frozenset({value.name})
    elements = list(value.args) if isinstance(value, Vector) else []
    if not elements or any(
        type(element) not in (sp.Symbol, StringLiteral) for element in elements
    ):
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
