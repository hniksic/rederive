"""The evaluation state every engine command reads.

`Context` is plain data (settings, variable domains, assignments, function definitions
and labels) independent of the session, so tests and commands can build one directly.
`Context.from_settings` builds one from the session's `Settings` and the parser's
declarations.

Undeclared variables are real, not complex; see `Context.domain`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import cache
from time import monotonic
from typing import Any, TypeVar

from rederive import library
from rederive.engine.ordering import ORDER_LIST
from rederive.model.expr import VARIADIC, Kind, Node, parameters
from rederive.model.settings import FIELDS, Settings
from rederive.syntax.state import Declaration, DomainDeclaration, InputMode, ParseState

_Choice = TypeVar("_Choice", bound=StrEnum)


class Precision(StrEnum):
    EXACT = "Exact"
    APPROXIMATE = "Approximate"
    MIXED = "Mixed"


class Notation(StrEnum):
    """How computed numbers are written. See `engine.notation`."""

    DECIMAL = "Decimal"
    MIXED = "Mixed"
    RATIONAL = "Rational"
    SCIENTIFIC = "Scientific"


class Branch(StrEnum):
    PRINCIPAL = "Principal"
    REAL = "Real"
    ANY = "Any"


class Direction(StrEnum):
    """The choices for the Exponential, Logarithm and Trigonometry settings."""

    AUTO = "Auto"
    COLLECT = "Collect"
    EXPAND = "Expand"


class TrigPower(StrEnum):
    AUTO = "Auto"
    SINES = "Sines"
    COSINES = "Cosines"


class Angle(StrEnum):
    RADIAN = "Radian"
    DEGREE = "Degree"


class DomainKind(StrEnum):
    """The domains a `:epsilon` declaration names.

    `VECTOR` is a vector or matrix, whose products do not commute; `NONSCALAR` is the
    older word for it, accepted as typed. `SET` is a set, which only the set operators
    act on, and `LOGICAL` a truth value, which only the logical operators act on.
    `UNKNOWN` is the parser's fallback for an unrecognized domain; it is treated as
    Real, like an undeclared variable.
    """

    INTEGER = "Integer"
    REAL = "Real"
    COMPLEX = "Complex"
    VECTOR = "Vector"
    SET = "Set"
    LOGICAL = "Logical"
    NONSCALAR = "Nonscalar"
    UNKNOWN = "?"

    @property
    def is_scalar(self) -> bool:
        """Whether a variable of this kind is a number."""
        return self in (
            DomainKind.INTEGER,
            DomainKind.REAL,
            DomainKind.COMPLEX,
            DomainKind.UNKNOWN,
        )

    @property
    def is_matrix(self) -> bool:
        """Whether a variable of this kind is a vector or matrix."""
        return self in (DomainKind.VECTOR, DomainKind.NONSCALAR)


@dataclass(frozen=True)
class Domain:
    """What `x :epsilon Real [0, inf)` says about `x`.

    The bounds are expression trees, since they may be symbolic; the conversion layer
    decides what facts they imply.
    """

    kind: DomainKind = DomainKind.REAL
    low: Node | None = None
    high: Node | None = None
    closed_low: bool = False
    closed_high: bool = False

    @property
    def has_interval(self) -> bool:
        return self.low is not None or self.high is not None


#: The notation each precision mode selects. Setting Options Precision also changes the
#: notation, as Derive does silently (recording only the precision).
NOTATION_FOR: dict[Precision, Notation] = {
    Precision.EXACT: Notation.RATIONAL,
    Precision.APPROXIMATE: Notation.SCIENTIFIC,
    Precision.MIXED: Notation.MIXED,
}

#: The domain of an undeclared variable, unless `default` is declared.
DEFAULT_DOMAIN = Domain()

#: Name standing for every undeclared variable. `Declare Variable` accepts it, and the
#: manual uses `default :epsilon Complex` to widen the domain of all variables.
DEFAULT_NAME = "default"


def domain_of_node(node: Node) -> tuple[str, Domain] | None:
    """Read a `DOMAIN` node as a name and the domain it declares.

    The parser's `DomainDeclaration` for the node carries only the kind; the interval
    is available only from the tree.
    """
    if node.kind is not Kind.DOMAIN or not node.children:
        return None
    declared = node.children[0]
    name = str(declared.value)
    if declared.kind is Kind.STRING:
        # A string is a variable in arithmetic, named by its text in quotes.
        name = f'"{name}"'
    kind = _choice(DomainKind, str(node.value), DomainKind.UNKNOWN)
    interval = next((c for c in node.children[1:] if c.kind is Kind.INTERVAL), None)
    if interval is None or len(interval.children) != 2:
        return name, Domain(kind)
    brackets = str(interval.value)
    return name, Domain(
        kind,
        low=interval.children[0],
        high=interval.children[1],
        closed_low=brackets.startswith("["),
        closed_high=brackets.endswith("]"),
    )


@dataclass(frozen=True)
class Definition:
    """A user function: its parameters, their defaults, and its body.

    `names` are the parameter names in order (a string parameter is named by its text in
    quotes), and `defaults` the default of each, an unevaluated tree or None; a call that
    omits the argument evaluates the default. `variadic` marks the head `F v`, whose one
    parameter receives all the arguments of a call as a vector. `body` is None for a
    function declared without one, `F(x, y) :=`, whose calls stand.
    """

    names: tuple[str, ...]
    body: Node | None
    defaults: tuple[Node | None, ...] = ()
    variadic: bool = False

    def default(self, index: int) -> Node | None:
        """The default of the parameter at `index`, or None if it has none."""
        return self.defaults[index] if index < len(self.defaults) else None

    @classmethod
    def of(cls, node: Node) -> Definition:
        """The definition a `FUNDEF` tree makes."""
        head = node.children[0]
        found = parameters(head)
        return cls(
            tuple(name for name, _ in found),
            node.children[1] if len(node.children) > 1 else None,
            tuple(default for _, default in found),
            head.value == VARIADIC,
        )


@cache
def library_definition(name: str) -> Definition:
    """The definition of the library name `name`, read the first time it is asked for
    in this process; a variable's is one without parameters, its body the value.

    Each text is read as a file's line, with only the library's names known, so that
    its parameters and binders are not variables of the session.
    """
    from rederive.syntax import parse_expression

    state = ParseState(input_mode=InputMode.WORD, reading_file=True)
    node = parse_expression(library.text(name), state).node
    if node.kind is Kind.ASSIGN:
        return Definition((), node.children[1])
    return Definition.of(node)


class Cell:
    """The value of one variable while a command runs: a local of a running function
    (see `Frame` in evaluate.py), or as a `Global`, a global. `raw` is the value as
    the evaluator keeps it, `CLEARED` for a variable that has none.

    Two locals share a cell where Derive shares a vector: `w := v` with `v` a local,
    and a parameter given a bare local of the caller. An element assignment through
    either name changes the cell's value, so both see it. Globals never share one.
    """

    __slots__ = ("raw",)

    def __init__(self, raw: object) -> None:
        self.raw = raw


class Global(Cell):
    """The cell of a global: `at` is the store's clock (`Store.version`) when the
    command last gave it a value, which orders the hand-back to the session, or 0 for
    an unstamped cell, which keeps the session's value of a variable written as a
    literal once it is worked out (see `_Evaluator._assigned` in evaluate.py)."""

    __slots__ = ("at",)

    def __init__(self, raw: object, at: int) -> None:
        self.raw = raw
        self.at = at


@dataclass(eq=False)
class Store:
    """What a command does to the session while it computes, as it does it.

    A program's assignment to a global, `x := u` with `x` no parameter of the function
    whose body assigns it, takes effect at once for the rest of the command, as does a
    definition `F(x) := u` made in a called body: FIT_.MTH's `SUBST(u, x, x0) :=
    [SUBST_AUX1(u, x), SUBST_AUX2(x0)] * [0, 1]` defines `SUBST_AUX2` in the first
    element and calls it in the second. When the command ends, the answer carries the
    final state back for the session to perform too (see `Result.performed`).

    `cells` holds the globals assigned so far, each in a `Global` cell with its value,
    `CLEARED` for one whose value was taken away; reading a global looks here before the
    session's assignments. Its unstamped cells are no value the command gave: they are
    never handed back, and only the evaluator reads them. `functions` holds the
    definitions made, None for one whose body was taken away, `domains` the declarations
    made, `settings` the settings assigned, and `effects` the trees of the definitions
    and declarations, each key with the clock when it was last made. `version` is the
    clock: it counts every change, so a value worked out from the globals can tell
    whether it is still current, and the stamps it gives order what the session
    performs. `steps` counts the program steps taken (see `Context.step`). `displayed`
    holds the texts `DISPLAY` showed, in order, and `written` the last text `WRITE`
    wrote (see `show`).

    A cell keeps its value as the evaluator does a local's, which may be native (a
    Python number, or a vector written in place; see `rederive.engine.native`).
    Everything outside the evaluator reads it as a sympy value (`native.given`), and it
    is written out once at the end; so a loop of a million assignments to a global
    costs what one to a local does.
    """

    cells: dict[str, Global] = field(default_factory=dict)
    """This dict is never replaced, only changed: evaluators hold it (see `restore`)."""
    functions: dict[str, Definition | None] = field(default_factory=dict)
    domains: dict[str, Domain] = field(default_factory=dict)
    """Replaced, not changed, by each declaration, so that a reference to it is the
    declarations in force when it was taken."""
    settings: dict[str, SettingValue] = field(default_factory=dict)
    effects: dict[tuple[str, str], tuple[int, Node | None]] = field(
        default_factory=dict
    )
    version: int = 0
    saved: int = 0
    """The clock when the innermost save point still active was taken (see `save`)."""
    steps: int = 0
    displayed: list[str] = field(default_factory=list)
    written: str | None = None
    random: int | None = None
    """The random state after the draws the command made, None while it made none (see
    `Context.random_state`)."""
    context: Context | None = None
    """The context with the settings in force now, where the command changed any (see
    `Context.with_setting`); None while it hasn't."""
    held: bool = False
    """Whether the command made a held value, the value of a quote: only then can a
    value have one, which spares looking for them everywhere else."""
    inert: set[str] = field(default_factory=set)
    """The names that a program left standing assigns (one met where programs don't
    run): they are the program's own, not variables of the expression."""
    passed_over: int | None = None
    """The nesting depth of the activation that went on past an undecided test, None
    while no running call has (see `_Evaluator._conditional` in evaluate.py)."""
    retested: dict[Any, Any] = field(default_factory=dict)
    """The undecided conditionals with their tests as written (names substituted), each
    mapped to the one shown, with its test as computed (see `_Evaluator._suspended` in
    evaluate.py)."""

    def show(self, text: str, displayed: bool) -> None:
        """Record a text a program shows: one `DISPLAY` adds to the worksheet when the
        command ends, if `displayed`, or else one `WRITE` writes on the status line.
        Either is passed on at once to whoever listens to the command's progress (see
        `reporting`)."""
        if displayed:
            self.displayed.append(text)
        else:
            self.written = text
        if _listener is not None:
            _listener(text, displayed)

    def assign(self, name: str, value: Any) -> None:
        """Give the global `name` the computed `value`, or take its value away with
        `CLEARED`."""
        self.version += 1
        self.cells[name] = Global(value, self.version)

    def define(self, node: Node) -> None:
        """Perform the `FUNDEF` tree `node`."""
        name = str(node.value)
        definition = Definition.of(node)
        self.functions[name] = None if definition.body is None else definition
        self._effect(("define", name), node)

    def set(self, name: str, value: SettingValue) -> None:
        """Record the setting `name` assigned `value` for the session."""
        self.settings[name] = value
        self._effect(("set", name), None)

    def declare(self, node: Node) -> None:
        """Perform the `DOMAIN` tree `node`, a declaration of a global variable."""
        declared = domain_of_node(node)
        if declared is None:
            return
        name, domain = declared
        self.domains = {**self.domains, name: domain}
        self._effect(("declare", name), node)

    def _effect(self, key: tuple[str, str], node: Node | None) -> None:
        self.version += 1
        self.effects[key] = (self.version, node)

    def save(self) -> Saved:
        """A save point: what the command has done to the session so far, for `restore`
        to put back where what it does next is to be undone (see `_entered` in
        evaluate.py). It stays active until `release`.

        The copies are shallow. That suffices for the cells, since giving a global a
        value makes it a new cell, provided no cell from before the save point is
        changed in place while it is active: a cell is changed in place only where
        its stamp is later than `saved`."""
        kept = Saved(
            dict(self.cells),
            dict(self.functions),
            self.domains,
            dict(self.settings),
            dict(self.effects),
            self.version,
            list(self.displayed),
            self.written,
            self.random,
            self.context,
            self.held,
            self.saved,
        )
        self.saved = self.version
        return kept

    def restore(self, kept: Saved) -> None:
        """Undo what the command did since the save point `kept` was taken."""
        self.cells.clear()
        self.cells.update(kept.cells)
        self.functions = dict(kept.functions)
        self.domains = kept.domains
        self.settings = dict(kept.settings)
        self.effects = dict(kept.effects)
        self.version = kept.version
        self.displayed = list(kept.displayed)
        self.written = kept.written
        self.random = kept.random
        self.context = kept.context
        self.held = kept.held

    def release(self, kept: Saved) -> None:
        """End the save point `kept`, restored or not."""
        self.saved = kept.saved


@dataclass(frozen=True, eq=False)
class Saved:
    """What `Store.save` keeps of a store; `saved` is the store's own `saved` from
    before, which `release` puts back."""

    cells: dict[str, Global]
    functions: dict[str, Definition | None]
    domains: dict[str, Domain]
    settings: dict[str, SettingValue]
    effects: dict[tuple[str, str], tuple[int, Node | None]]
    version: int
    displayed: list[str]
    written: str | None
    random: int | None
    context: Context | None
    held: bool
    saved: int


#: Where the texts a running command shows go as it shows them; see `reporting`.
_listener: Callable[[str, bool], None] | None = None


@contextmanager
def reporting(listener: Callable[[str, bool], None] | None) -> Iterator[None]:
    """Pass each text `DISPLAY` or `WRITE` shows to `listener` while the block runs,
    with whether it is a `DISPLAY`'s, so that the user sees a long program's progress
    before it ends. A worker passes them over its pipe, an in-process engine straight to
    the session."""
    global _listener
    previous, _listener = _listener, listener
    try:
        yield
    finally:
        _listener = previous


class FileNeeded(BaseException):
    """A program evaluated `LOAD` of a file the command hasn't looked for yet.

    The engine reads no files, so the command ends here; the session looks for the file,
    loads it if it is found, and runs the command again with the outcome in
    `Context.loads`. It is a `BaseException`, so that no handler in the engine that
    turns a failure into an unevaluated call can take it for one.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


class StepLimitExceeded(BaseException):
    """A command ran more program steps than its `Context.step_limit` allows.

    It abandons the command as an abort does: nothing the command did reaches the
    session. It is a `BaseException`, so that no handler in the engine that turns a
    failure into an unevaluated call can take it for one.
    """


class PassedOverTooFar(BaseException):
    """A call that went on past an undecided test nested deeper than such a call may
    (see `Store.passed_over`): the call stands.

    A `BaseException`, so that no handler in the engine that turns a failure into an
    unevaluated call takes it for one before the call's own `_entered` does.
    """


class Unsettled(BaseException):
    """A strict command (see `Context.strict`) met a program it can't settle over
    symbolic values, or ran out of its time.

    The simplification a plot starts from is strict: a program whose flow depends on
    the plot's variable has no single symbolic value, and is run at each point instead.
    It is a `BaseException`, so that no handler in the engine that turns a failure into
    an unevaluated call can take it for one; the command is abandoned without effect.
    """


#: The value of a setting as a program reads and assigns it: a word such as
#: `Approximate`, a number, or for `VariableOrder` the names in order.
SettingValue = str | int | tuple[str, ...]

#: The setting that is the variable order list rather than a field of the settings.
VARIABLE_ORDER = "VariableOrder"

#: The value `Name :=` assigns a setting: its factory default.
CLEARED_SETTING = ""


class _Cleared:
    """The type of `CLEARED`."""

    def __repr__(self) -> str:
        return "CLEARED"


#: The value of a global whose value a command took away, `x :=`.
CLEARED = _Cleared()


@dataclass(frozen=True)
class Context:
    """Everything an engine command's answer may depend on.

    Defaults are exact arithmetic to ten digits, principal branch, every
    transformation mode on Auto, radians, base ten, and Derive's initial order list.
    """

    precision: Precision = Precision.EXACT
    precision_digits: int = 10
    notation: Notation = Notation.RATIONAL
    notation_digits: int = 10
    branch: Branch = Branch.PRINCIPAL
    exponential: Direction = Direction.AUTO
    logarithm: Direction = Direction.AUTO
    trigonometry: Direction = Direction.AUTO
    trigpower: TrigPower = TrigPower.AUTO
    angle: Angle = Angle.RADIAN
    input_base: int = 10
    order: tuple[str, ...] = ORDER_LIST
    """The variable order list, most main first, as set by `Manage Ordering`.

    This is session state rather than a setting (it isn't a dialog field value or a
    `Name := Value`), but command answers depend on it, so it belongs here.
    """
    arbitrary_index: int = 1
    """The next free `@n` for soLve's arbitrary values.

    Session state, like the order list: `x = x` solves to `x = @1` and the counter
    only increases, so no two solves in a worksheet produce the same constant. It
    lives here so that answers depend only on the tree and the context.
    """
    random_state: int = field(default=0, compare=False)
    """The state of `RANDOM`'s generator when the command starts, a 32-bit number.

    Session state, like the arbitrary index: each draw steps it, and the command's
    answer carries the final state back (see `Store.random`), so that a session draws
    reproducibly after a seed.
    """
    domains: Mapping[str, Domain] = field(default_factory=dict)
    assignments: Mapping[str, Node] = field(default_factory=dict)
    functions: Mapping[str, Definition] = field(default_factory=dict)
    labels: Mapping[int, Node] = field(default_factory=dict)
    store: Store = field(default_factory=Store, compare=False)
    step_limit: int | None = None
    """How many program steps a command may take (see `step`), or None for no
    limit.

    Where a command can be aborted (the app's worker is killed by Esc), there is none:
    a program runs until it ends or the user stops it, as in Derive. A session that
    computes in its own process cannot be interrupted, so it sets one.
    """

    setting_values: Mapping[str, str | int] = field(default_factory=dict)
    """Every setting's value by name, as the session's `Settings` holds them, for the
    settings a command reads but doesn't compute with (see `setting`)."""

    loads: Mapping[str, bool] = field(default_factory=dict)
    """The files the session looked for at a `LOAD` of this command, by the name as
    written, and whether each was found and loaded (see `FileNeeded`)."""

    withdrawn: frozenset[str] = frozenset()
    """The names of the pre-defined library the session took away by clearing them.

    The others are in force wherever `functions` and `assignments` don't define the
    name (see `rederive.library`). Only these few names travel with a command: each
    process reads a library definition itself, the first time it is called.
    """

    strict: bool = field(default=False, compare=False)
    """Whether the command must settle every program it runs, as the simplification a
    plot starts from must (see `Unsettled`): a test of a program's flow that stays
    undecided, or an iteration carrying an undecided conditional, ends it at once, and
    so does a step taken past `deadline`."""

    deadline: float | None = field(default=None, compare=False)
    """The `time.monotonic` time past which a strict command ends at its next program
    step, or None for none."""

    def step(self) -> None:
        """Count one step of a program (a statement, a pass of an iteration, or a call
        of a user function) in `Store.steps`, raise `StepLimitExceeded` once there are
        more than `step_limit`, and end a strict command that is out of time."""
        store = self.store
        store.steps += 1
        if self.step_limit is not None and store.steps > self.step_limit:
            raise StepLimitExceeded("Step limit exceeded")
        if self.deadline is not None and monotonic() > self.deadline:
            raise Unsettled("out of time")

    def unsettled(self, why: str) -> None:
        """End a strict command because a program met something it can't settle,
        which `why` names; nothing happens in any other."""
        if self.strict:
            raise Unsettled(why)

    def setting(self, name: str) -> SettingValue | None:
        """The value of the setting `name` in this context, or None if there is none.

        The settings the engine computes with are read from its own fields, so the
        value is the one in force: inside `APPROX(u, 20)`, `PrecisionDigits` is 20.
        """
        if name == VARIABLE_ORDER:
            return self.order
        if name not in FIELDS:
            return None
        return _settings_of(self)[name]

    def with_setting(self, name: str, value: SettingValue) -> Context | None:
        """The same context with the setting `name` given `value`, or None if `value`
        isn't one of its values.

        A word matches a choice whatever its case and a number is accepted where the
        setting takes one, as when a setting is assigned at top level; precision
        brings its notation along, as in the Options dialog (see `Settings.apply`).
        `CLEARED_SETTING` restores the factory default.
        """
        if name == VARIABLE_ORDER:
            if value == CLEARED_SETTING:
                return replace(self, order=ORDER_LIST)
            if not isinstance(value, tuple):
                return None
            # A repeated name counts once, at its first place.
            return replace(self, order=tuple(dict.fromkeys(value)))
        if isinstance(value, tuple) or name not in FIELDS:
            return None
        settings = _settings_of(self)
        if not settings.assign(name, str(value)):
            return None
        return replace(self, **_fields_of(settings), setting_values=settings.values())

    def function(self, name: str) -> Definition | None:
        """The definition of function `name`, including one this command made, and
        the pre-defined library's where the session has none."""
        if name in self.store.functions:
            return self.store.functions[name]
        found = self.functions.get(name)
        if found is None and name in library.FUNCTIONS and name not in self.withdrawn:
            return library_definition(name)
        return found

    def assigned(self, name: str) -> Node | None:
        """The value the session gives variable `name`, as a tree to evaluate, or
        the pre-defined library's where the session gives none.

        A value this command assigned is in `store` instead, already computed.
        """
        found = self.assignments.get(name)
        if found is None and name in library.VARIABLES and name not in self.withdrawn:
            return library_definition(name).body
        return found

    def fresh(self) -> Context:
        """The same context with nothing done yet, for a command to start from."""
        return replace(self, store=Store())

    def domain(self, name: str) -> Domain:
        """The domain of variable `name`.

        Undeclared variables get the domain declared for `default`, or Real.
        """
        declared = self.store.domains.get(name) or self.domains.get(name)
        if declared is not None:
            return declared
        return self.domains.get(DEFAULT_NAME, DEFAULT_DOMAIN)

    def with_precision(
        self, precision: Precision, digits: int | None = None
    ) -> Context:
        """The same context at another precision, as approX uses.

        The notation style and digits change with it, as in Derive. So approX answers
        in scientific notation regardless of the notation setting (`approX 10^7·π` is
        `3.141592653·10⁷` under Rational too), and approX to twelve digits shows twelve.
        An answer keeps the style it was written in, since renders are never redone.
        """
        return replace(
            self,
            precision=precision,
            precision_digits=self.precision_digits if digits is None else digits,
            notation=NOTATION_FOR[precision],
            notation_digits=self.notation_digits if digits is None else digits,
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        declarations: Iterable[Declaration] = (),
        *,
        order: Sequence[str] | None = None,
        arbitrary_index: int = 1,
        random_state: int = 0,
        domains: Mapping[str, Domain] | None = None,
        assignments: Mapping[str, Node] | None = None,
        functions: Mapping[str, Definition] | None = None,
        labels: Mapping[int, Node] | None = None,
        step_limit: int | None = None,
        withdrawn: frozenset[str] = frozenset(),
    ) -> Context:
        """Build a context from the session's settings and declarations.

        `declarations` supplies the recorded domain *kinds*; `domains` overrides them
        with full domains, including intervals, which only the `DOMAIN` node has.
        """
        declared = {
            declaration.name: Domain(
                _choice(DomainKind, declaration.domain, DomainKind.UNKNOWN)
            )
            for declaration in declarations
            if isinstance(declaration, DomainDeclaration)
        }
        declared.update(domains or {})
        return cls(
            **_fields_of(settings),
            setting_values=settings.values(),
            order=ORDER_LIST if order is None else tuple(order),
            arbitrary_index=arbitrary_index,
            random_state=random_state,
            domains=declared,
            assignments=dict(assignments or {}),
            functions=dict(functions or {}),
            labels=dict(labels or {}),
            step_limit=step_limit,
            withdrawn=withdrawn,
        )


def _fields_of(settings: Settings) -> dict[str, Any]:
    """The fields of a context that `settings` give."""
    return dict(
        precision=_setting(settings, "Precision", Precision, Precision.EXACT),
        precision_digits=_number(settings, "PrecisionDigits", 10),
        notation=_setting(settings, "Notation", Notation, Notation.RATIONAL),
        notation_digits=_number(settings, "NotationDigits", 10),
        branch=_setting(settings, "Branch", Branch, Branch.PRINCIPAL),
        exponential=_setting(settings, "Exponential", Direction, Direction.AUTO),
        logarithm=_setting(settings, "Logarithm", Direction, Direction.AUTO),
        trigonometry=_setting(settings, "Trigonometry", Direction, Direction.AUTO),
        trigpower=_setting(settings, "Trigpower", TrigPower, TrigPower.AUTO),
        angle=_setting(settings, "Angle", Angle, Angle.RADIAN),
        input_base=settings.base("InputBase"),
    )


def _settings_of(context: Context) -> Settings:
    """The settings `context` computes under: the session's, with the fields of the
    context, which may differ from them (inside `APPROX`, for one), in force."""
    settings = Settings()
    settings.apply(
        {
            **context.setting_values,
            "Precision": context.precision.value,
            "PrecisionDigits": context.precision_digits,
            "Notation": context.notation.value,
            "NotationDigits": context.notation_digits,
            "Branch": context.branch.value,
            "Exponential": context.exponential.value,
            "Logarithm": context.logarithm.value,
            "Trigonometry": context.trigonometry.value,
            "Trigpower": context.trigpower.value,
            "Angle": context.angle.value,
        }
    )
    return settings


def _choice(enum: type[_Choice], value: str, fallback: _Choice) -> _Choice:
    """The member named by `value`, case-insensitively."""
    for member in enum:
        if member.value.lower() == value.lower():
            return member
    return fallback


def _setting(
    settings: Settings, name: str, enum: type[_Choice], fallback: _Choice
) -> _Choice:
    """Read a setting as a member of `enum`.

    Every setting read here belongs to a dialog field, so it is always present; the
    fallback covers values outside the field's choices.
    """
    return _choice(enum, str(settings[name]), fallback)


def _number(settings: Settings, name: str, fallback: int) -> int:
    value = settings[name]
    return value if isinstance(value, int) else fallback
