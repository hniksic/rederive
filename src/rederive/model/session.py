"""The Algebra session: the numbered history and the selection within it.

Pure Python, with no Textual and no key names. The UI translates key presses into calls
on `Session` and paints the result; it never calls the math engine itself.

The engine is the one thing a session takes from outside. The five calls that can be
expensive go through `runner` and are awaited, so the program can keep drawing while one
runs; what does the waiting differs by platform (a thread on the desktop, an event on a
web page). Everything else is cheap and goes to `engine.client`, which holds no
mathematics, so the app process can draw a worksheet without importing sympy. Commands
that compute are coroutines; all other commands are ordinary calls.

An answer is appended only after its call returns, so a call that fails or is aborted
leaves the worksheet unchanged.

The session owns the parse state (so `InputMode`, `CaseMode` and definitions carry to
later lines), the settings (which an authored `Name := Value` writes to, like an Options
dialog does), and each entry's render, made once when the entry is authored. It also
owns the values, function bodies and variable domains the lines define, since every
engine command needs them. `Session.context` bundles those with the settings for the
engine.

The Declare commands are thin: each writes the expression the user could have typed and
authors it, as the original does, so there is only one path into the symbol table.

Renders are never remade. Changing the times operator or display format affects only
later expressions; entries already on screen keep the look they had when entered, as in
the original.

Entries leave the history only through `remove`, which keeps them in `removed` for
`unremove`. Any engine command empties that buffer; `move_block` only rearranges, so it
leaves the buffer alone.

Manage commands that need no engine live here too: `renumber` restores label sequence
after removing and unremoving, `annotate` edits an entry's origin, and `order_list`
reads the variable order list, which the engine receives through the context.

`substitute` does need the engine. It always works on the whole entry, however little is
highlighted, because the substitution happens inside the entry and there is no part to
splice an answer back into.

`solve` bypasses `_command`, because `_command` appends exactly one entry while soLve
appends one per solution (possibly none). Like Substitute it works on the whole entry.
It also owns `arbitrary`, the counter for `@n` values, which advances past every one
that reaches the worksheet and never goes back.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from rederive import platform
from rederive.display import DisplayOptions, Layout, Region, render

# Only the client half of the engine is imported, under the name `engine`, so that an
# accidental `engine.simplify` fails to resolve. The one place that needs the computing
# half imports it separately, in `_computing_here`.
from rederive.engine import client as engine
from rederive.engine.replacing import same_expression
from rederive.model import data, state, worksheet
from rederive.model.expr import Kind, Node
from rederive.model.settings import Settings
from rederive.syntax import (
    PARSING_SETTINGS,
    ClearWhat,
    Declaration,
    DeriveSyntaxError,
    InputMode,
    Language,
    ParseResult,
    ParseState,
    SettingDeclaration,
    Source,
    is_name,
    parse_expression,
    source_lines,
    write_expression,
    write_source,
)

#: A route into an entry's selection tree: indices into `Region.children`,
#: empty for the whole expression.
Route = tuple[int, ...]
Rect = tuple[int, int, int, int]

#: What the status line says about a line the user wrote.
AUTHORED = "User"

#: How a declaration spells the domain operator and an infinite bound. The parser reads
#: either spelling; these are the ones the original writes.
DOMAIN_OPERATOR = ":ε"
INFINITY = "∞"

#: The name behind that glyph, as a tree carries it.
_INFINITE = "inf"

#: The Declare Variable menu's word for a variable declared without a domain, and the
#: interval menu's two words for intervals without a name of their own.
VALUE = "Value"
ALL = "All"
INTERVAL = "Interval"

#: A reference to a numbered entry, in an expression or an annotation like `Simp(#3)`.
LABEL = re.compile(r"#(\d+)")

#: An arbitrary value, as written by soLve or by Simplify of an authored `SOLVE`. The
#: session scans every entry for these so its counter never reuses one.
ARBITRARY = re.compile(r"@(\d+)")

#: Maps a label number to its replacement, or to None to leave it unchanged.
Rename = Callable[[int], int | None]

#: One engine command. All take the same three arguments and return an awaitable
#: answer, so `_command` can run any of them without knowing which.
Command = Callable[[Node, "engine.Context", ParseState], Awaitable["engine.Result"]]


class Runner(Protocol):
    """The five engine calls that can be expensive.

    Everything else the session asks the engine (is a tree a quotient, what is its main
    variable, how is it written) is a cheap tree walk and stays a direct call. These
    five convert to sympy, and conversion alone can hang: `10^10^10` never finishes
    being built.

    They are awaited so the session needn't care where the waiting happens.
    `remote.RemoteEngine` runs each in a thread blocking on a pipe to a child process,
    which is what makes Esc work in the app; `_Here` computes in-process, for tests and
    direct callers; and an implementation without threads or processes can resolve
    them from an event.
    """

    async def simplify(
        self, node: Node, context: engine.Context, state: ParseState | None = ...
    ) -> engine.Result: ...

    async def factor(
        self,
        node: Node,
        context: engine.Context,
        amount: engine.Amount = ...,
        variables: Sequence[str] = ...,
        state: ParseState | None = ...,
    ) -> engine.Result: ...

    async def expand(
        self,
        node: Node,
        context: engine.Context,
        amount: engine.Amount = ...,
        variables: Sequence[str] = ...,
        state: ParseState | None = ...,
    ) -> engine.Result: ...

    async def solve(
        self,
        node: Node,
        context: engine.Context,
        variables: Sequence[str] = ...,
        bounds: tuple[Node, Node] | None = ...,
        state: ParseState | None = ...,
    ) -> tuple[engine.Result, ...]: ...

    async def expression_variables(
        self, node: Node, context: engine.Context | None = ..., written: bool = ...
    ) -> tuple[str, ...]: ...

    async def relation_of(
        self, node: Node, context: engine.Context | None = ...
    ) -> str | None: ...


class _Here:
    """The engine computing in this process, behind the five awaitable calls.

    These coroutines never suspend, so a session computing this way cannot be aborted:
    there is no moment for an Esc to arrive. That is why the app uses a proxy instead.
    """

    def __init__(self, computing: Any) -> None:
        self._engine = computing

    async def simplify(
        self, node: Node, context: engine.Context, state: ParseState | None = None
    ) -> engine.Result:
        return self._engine.simplify(node, context, state)

    async def factor(
        self,
        node: Node,
        context: engine.Context,
        amount: engine.Amount = engine.Amount.RATIONAL,
        variables: Sequence[str] = (),
        state: ParseState | None = None,
    ) -> engine.Result:
        return self._engine.factor(node, context, amount, variables, state)

    async def expand(
        self,
        node: Node,
        context: engine.Context,
        amount: engine.Amount = engine.Amount.RATIONAL,
        variables: Sequence[str] = (),
        state: ParseState | None = None,
    ) -> engine.Result:
        return self._engine.expand(node, context, amount, variables, state)

    async def solve(
        self,
        node: Node,
        context: engine.Context,
        variables: Sequence[str] = (),
        bounds: tuple[Node, Node] | None = None,
        state: ParseState | None = None,
    ) -> tuple[engine.Result, ...]:
        return self._engine.solve(node, context, variables, bounds, state)

    async def expression_variables(
        self, node: Node, context: engine.Context | None = None, written: bool = False
    ) -> tuple[str, ...]:
        return self._engine.expression_variables(node, context, written)

    async def relation_of(
        self, node: Node, context: engine.Context | None = None
    ) -> str | None:
        return self._engine.relation_of(node, context)


def _computing_here() -> Runner:
    """The in-process engine, for a session given no proxy.

    This is the only place the session layer imports the computing half, and it does
    so lazily, when a session is made rather than when this module is imported. That
    import pulls in sympy (some four hundred modules), which is why the app passes a
    `RemoteEngine` instead and its UI process never loads sympy. Tests and direct
    callers use this.

    `_Here` only wraps the module in awaitables; the module's own functions stay
    ordinary, since the worker process calls them directly.
    """
    from rederive.engine import computing

    return _Here(computing)


@dataclass(frozen=True)
class Bounds:
    """An interval as a declaration writes it: two bounds, each open or closed.

    The bounds are text, as held by a data entry field. The default is the whole real
    line, which is what the original's bounds screen starts with. An infinite bound is
    always open, since a variable cannot be declared infinite (manual section 4.10).
    """

    low: str = f"-{INFINITY}"
    high: str = INFINITY
    closed_low: bool = False
    closed_high: bool = False

    @property
    def text(self) -> str:
        """The interval in standard notation, as `(0, ∞)`."""
        opening = "[" if self.closed_low and not _is_infinite(self.low) else "("
        closing = "]" if self.closed_high and not _is_infinite(self.high) else ")"
        return f"{opening}{self.low}, {self.high}{closing}"


#: The bounds each interval menu word stands for. `All` is absent because the whole
#: line is written as no interval at all.
NAMED_INTERVALS: dict[str, Bounds] = {
    "Positive": Bounds("0", INFINITY),
    "Negative": Bounds(f"-{INFINITY}", "0"),
    "nonpoSitive": Bounds(f"-{INFINITY}", "0", closed_high=True),
    "nonneGative": Bounds("0", INFINITY, closed_low=True),
}


@dataclass(frozen=True)
class Entry:
    """One numbered line of the history: what was authored, and how it looks.

    `text` is the expression in author notation, written back from its tree rather than
    as typed: `x (x + 1)` is held as `x*(x+1)` and `(NOT p) OR (q AND r)` as
    `NOT p OR q AND r`. `node`'s spans index this text, so a subexpression can be
    extracted and spliced back. What the user sees is `layout`, drawn from the same tree.

    `annotation` is the entry's origin, as shown in the status line: `User` for an
    authored line, `Simp(#3)` for a Simplify of entry 3, and `Simp(#3')` when only the
    highlighted part of entry 3 was simplified.
    """

    number: int
    text: str
    node: Node
    layout: Layout
    annotation: str = AUTHORED
    exact: Node | None = None
    """The exact value behind the line, when the notation shows less than all of it.

    A decimal style shows and saves one third as `0.333333`, which reads back as a
    different number. The original keeps computing with the exact third, so `3·#n` is
    1, not 0.999999. `node` is what is drawn and saved; this is what later commands
    get. None when the two are the same, as for every authored line and every answer
    in Rational notation.
    """

    @property
    def value(self) -> Node:
        """The tree a later command computes with."""
        return self.node if self.exact is None else self.exact

    @property
    def height(self) -> int:
        return self.layout.height


class Session:
    """The expression history plus the selection cursor over it.

    The selection is an entry index plus a route into that entry's render; an empty
    route selects the whole entry. Every navigation method returns whether the
    selection moved, so the UI can beep if not.

    Navigation follows the render rather than the parse, because in the original you
    select what you see: `a + b - c` offers three terms, and `SIN(x + 1)` offers its
    argument but never its name.

    Stepping into a node returns to the operand last selected under it rather than the
    first one. Only one expression keeps that memory at a time; see `_preferred`.
    """

    def __init__(
        self, settings: Settings | None = None, runner: Runner | None = None
    ) -> None:
        self.settings = settings if settings is not None else Settings()
        #: Runs the five expensive engine calls. Defaults to in-process computation,
        #: which can't be interrupted; the app passes a proxy to a child process so
        #: computations can be aborted and their resource use capped.
        self.runner: Runner = runner if runner is not None else _computing_here()
        self.state = ParseState()
        self.entries: list[Entry] = []
        self.selected: int | None = None
        self.route: Route = ()
        #: The operand last selected under each node of the explored expression, keyed
        #: by route; stepping in returns there. It belongs to `_preferred_owner` only,
        #: and stepping into another expression discards it.
        self._preferred: dict[Route, int] = {}
        self._preferred_owner: Entry | None = None
        #: The file this session was last saved to or read from, if any.
        self.file: Path | None = None
        #: What the last Remove took out, for an Unremove to put back.
        self.removed: list[Entry] = []
        #: What the lines so far have defined, for the engine to substitute.
        self.assignments: dict[str, Node] = {}
        self.functions: dict[str, engine.Definition] = {}
        self.domains: dict[str, engine.Domain] = {}
        #: The variable order list `Manage Ordering` writes, most main first.
        self.order: tuple[str, ...] = engine.ORDER_LIST
        #: The next free `@n` for soLve's arbitrary values. It only increases: solving
        #: `x = x` three times gives `@1`, `@2` and `@3`.
        self.arbitrary = 1
        self._next_number = 1
        self.settings.watch(self._settings_changed)
        self._settings_changed(PARSING_SETTINGS)

    def copy(self) -> "Session":
        """A copy of this session, for the new half of a split window.

        The copies share the settings, which are global rather than per-window, and the
        runner. Everything else is copied, so the histories diverge from here on, down
        to label numbers.

        Entries themselves are shared, which is safe because they are never mutated
        (annotating replaces the entry in the list).
        """
        other = Session(self.settings, self.runner)
        other.state = replace(
            self.state,
            functions=dict(self.state.functions),
            variables=dict(self.state.variables),
            _index={},
            _index_stamp=(-1,),
        )
        other.entries = list(self.entries)
        other.selected = self.selected
        other.route = self.route
        other.file = self.file
        other.removed = list(self.removed)
        other.assignments = dict(self.assignments)
        other.functions = dict(self.functions)
        other.domains = dict(self.domains)
        other.order = self.order
        other.arbitrary = self.arbitrary
        other._next_number = self._next_number
        return other

    def discard(self) -> None:
        """Unsubscribe from the settings; called when the session's window closes.

        A session mirrors three settings into its parse state and stays subscribed until
        then.
        """
        self.settings.unwatch(self._settings_changed)

    # -- authoring ---------------------------------------------------------

    def author(self, text: str) -> Entry:
        """Parse `text`, append it as a new entry, and select it as a whole.

        Raises `DeriveSyntaxError` and appends nothing if the line doesn't parse. Nothing
        is simplified: an expression is inert until the user acts on it (R-HIST1), and
        label numbers only increase (R-HIST2).

        The entry shows the parsed expression rather than the typed spelling; see
        `_redrawn`.

        Declarations take effect before the line is drawn, so a typed
        `DisplayFormat := Compressed` prints itself compressed, as in the original.

        A line with a trailing `=` is entered as its `showvalue` parse, uncomputed. To
        compute the value, use the `show_value` command.
        """
        result = self._authored(text)
        return self._append(result.source.text, result.node)

    def _authored(self, text: str) -> ParseResult:
        """Parse `text` as an author line, and apply what the line declares.

        This is all of authoring except the append, shared so that a line with a trailing
        `=` declares and defines exactly what it would without it.
        """
        result = self._redrawn(parse_expression(text, self.state))
        for declaration in result.declarations:
            self.declare(declaration)
        self._define(result.node, result.declarations)
        return result

    def shows_value(self, text: str) -> bool:
        """Whether `text` is an author line typed with a trailing `=`.

        Parsing is cheap, so the app asks this to decide whether entering the line is a
        plain append or a computation to run as a separate task.

        Raises `DeriveSyntaxError` if `text` doesn't parse, as entering it would.
        """
        return parse_expression(text, self.state).node.kind is Kind.SHOWVALUE

    async def show_value(self, text: str) -> Entry:
        """Enter `text`, whose trailing `=` asks for its value beside it.

        `2^3=` enters `2^3 = 8` (manual section 3.1): the typed expression equated to its
        simplified form. The parser returns the trailing `=` as a `showvalue` node, which
        this turns into the equation.

        This is separate from `author` because the value must be computed, and all
        computation goes through the runner. `author` never computes. A line without a
        trailing `=` is simply authored, so this handles any author line.

        The entry is annotated as authored, since the user typed the `=`.

        Raises `DeriveSyntaxError` and appends nothing if the line doesn't parse.
        """
        result = self._authored(text)
        if result.node.kind is not Kind.SHOWVALUE:
            return self._append(result.source.text, result.node)
        shown = result.node.children[0]
        answer = await self.runner.simplify(shown, self.context, self.state)
        line, node = self._equated(shown, answer.node)
        exact = None if answer.value is None else self._equated(shown, answer.value)[1]
        return self._append(line, node, AUTHORED, exact)

    def _equated(self, shown: Node, value: Node) -> tuple[str, Node]:
        """`shown = value` written out and read back, as one expression.

        Written spaced, as answers are, since half of the line is an answer. It is read
        back so the entry's spans index its text, which highlighting requires.
        """
        text = write_expression(Node(Kind.REL, 0, 0, (shown, value), "="), spaced=True)
        return text, parse_expression(text, self.state).node

    def _redrawn(self, result: ParseResult) -> ParseResult:
        """`result` written back out from its tree and read again.

        As in the original, the worksheet shows the expression rather than the typed line:
        `(NOT p) OR (q AND r)` becomes `NOT p OR q AND r`, without unneeded parentheses.
        Nothing is reordered (`q OR p` stays `q OR p`); writing is the inverse of parsing,
        not a normal form.

        It is reparsed so the tree's spans index the entry's text, which lets a
        subexpression be copied out and spliced back.

        Reparsing uses the state from before the line's declarations are applied, or
        `InputBase := 16` would come back as twenty-two. If the round trip changes the
        tree, the typed line is kept, since faithfulness beats tidiness.
        """
        try:
            written = parse_expression(write_expression(result.node), self.state)
        except DeriveSyntaxError:
            return result
        return written if _same_tree(written.node, result.node) else result

    def record(self, setting: str) -> Entry:
        """Append the `Name := Value` a settings change records.

        The expression is built rather than parsed, because the settings have already
        changed: `InputBase := 7` would not lex in base seven.
        """
        return self._append(
            self.settings.assignment(setting), self.settings.assignment_node(setting)
        )

    def _append(
        self,
        text: str,
        node: Node,
        annotation: str = AUTHORED,
        exact: Node | None = None,
    ) -> Entry:
        entry = Entry(
            self._next_number,
            text,
            node,
            render(node, self.options),
            annotation,
            exact,
        )
        self._next_number += 1
        self._minted(text)
        self.entries.append(entry)
        self.selected = len(self.entries) - 1
        self.route = ()
        return entry

    def _minted(self, text: str) -> None:
        """Put the arbitrary-value counter past every `@n` this entry carries.

        This applies to every entry, not just soLve's, because a typed `SOLVE` mints them
        through Simplify, and a loaded worksheet carries whatever its session minted. The
        counter never decreases, so no two arbitrary values in a worksheet collide.
        """
        for found in ARBITRARY.finditer(text):
            self.arbitrary = max(self.arbitrary, int(found.group(1)) + 1)

    def _define(self, node: Node, declarations: Iterable[Declaration]) -> None:
        """Record what a line defines: a value, a function body, or a domain.

        The parse state knows which names are taken, but their meaning lives only in the
        tree, so this is what carries `x := 5` to later commands. Settings are not
        variables: `Notation := Mixed` defines nothing, and `declarations` identifies
        them.

        Assignments inside a function body are not definitions. `DSOLVE1(p, q, x, y, x0,
        y0, a_) := IF("inapplicable" = a_ := ..., ...)` assigns `a_` when called, not when
        defined, so the walk stops at function bodies.
        """
        settings = {
            declaration.setting
            for declaration in declarations
            if isinstance(declaration, SettingDeclaration)
        }
        for found in _defining(node):
            match found.kind:
                case Kind.ASSIGN:
                    self._assigned(found, settings)
                case Kind.FUNDEF:
                    self._defined(found)
                case Kind.DOMAIN:
                    declared = engine.domain_of_node(found)
                    if declared is not None:
                        self.domains[declared[0]] = declared[1]

    def _assigned(self, node: Node, settings: set[str]) -> None:
        """`x := 5` gives x a value; `x :=` with nothing after it takes it away."""
        target = node.children[0]
        if target.kind is not Kind.NAME:
            return
        name = str(target.value)
        if name in settings:
            return
        if len(node.children) > 1:
            self.assignments[name] = node.children[1]
        else:
            self.assignments.pop(name, None)

    def _defined(self, node: Node) -> None:
        """`F(x) := x^2` gives F a body; `F(x) :=` leaves F an arbitrary function."""
        name = str(node.value)
        parameters = tuple(str(child.value) for child in node.children[0].children)
        if len(node.children) > 1:
            self.functions[name] = (parameters, node.children[1])
        else:
            self.functions.pop(name, None)

    def declare(self, declaration: Declaration) -> None:
        """Apply one declaration: a setting to the settings, the rest to the state.

        A setting goes to the settings, which mirror it back to the parse state, so an
        invalid value can't leave the two disagreeing about, say, the input base. The
        mirroring is repeated here because the settings report only changes: a file read
        in Word mode that assigns `InputMode := Character` must switch the parse state
        even though the setting already says Character.
        """
        if isinstance(declaration, SettingDeclaration):
            self.settings.assign(declaration.setting, declaration.value)
            self._settings_changed(frozenset({declaration.setting}))
            return
        self.state.declare(declaration)

    @property
    def options(self) -> DisplayOptions:
        """The display options the settings currently call for."""
        return DisplayOptions(
            times=str(self.settings["TimesOperator"]),
            compressed=self.settings["DisplayFormat"] == "Compressed",
            output_base=self.settings.base("OutputBase"),
            notation_digits=int(self.settings["NotationDigits"]),
        )

    def _settings_changed(self, changed: frozenset[str]) -> None:
        """Mirror the three settings that affect lexing into the parse state."""
        for setting in changed & PARSING_SETTINGS:
            value = str(self.settings[setting])
            self.state.declare(SettingDeclaration(setting, value))

    # -- commands ----------------------------------------------------------

    @property
    def context(self) -> engine.Context:
        """Everything an engine command's answer may depend on.

        Every entry is offered as a label, so `#3` means entry 3 wherever it appears.
        """
        return engine.Context.from_settings(
            self.settings,
            order=self.order,
            arbitrary_index=self.arbitrary,
            domains=self.domains,
            assignments=self.assignments,
            functions=self.functions,
            labels={entry.number: entry.value for entry in self.entries},
        )

    async def simplify(self, request: str) -> Entry:
        """Append the simplified form of the expression `request` names."""
        return await self._command(request, "Simp", self.runner.simplify)

    async def approx(self, request: str) -> Entry:
        """Append the approximated form of the expression `request` names.

        This simplifies `APPROX` applied to it, at `PrecisionDigits` precision; otherwise
        it behaves exactly like Simplify.
        """

        def run(
            node: Node, context: engine.Context, state: ParseState
        ) -> Awaitable[engine.Result]:
            name = Node(Kind.NAME, node.start, node.start, (), "APPROX")
            call = Node(Kind.CALL, node.start, node.end, (name, node))
            return self.runner.simplify(call, context, state)

        return await self._command(request, "Approx", run)

    async def factor(
        self,
        request: str,
        amount: engine.Amount = engine.Amount.RATIONAL,
        variables: Sequence[str] = (),
    ) -> Entry:
        """Append the factored form of the expression `request` names.

        `amount` is how far to factor and `variables` lists the factorization variables in
        the order chosen; otherwise it behaves exactly like Simplify.
        """

        def run(
            node: Node, context: engine.Context, state: ParseState
        ) -> Awaitable[engine.Result]:
            return self.runner.factor(node, context, amount, variables, state)

        return await self._command(request, "Fctr", run)

    async def expand(
        self,
        request: str,
        amount: engine.Amount = engine.Amount.RATIONAL,
        variables: Sequence[str] = (),
    ) -> Entry:
        """Append the expanded form of the expression `request` names.

        `variables` lists the expansion variables in the order chosen, and `amount` is how
        far a denominator is factored for partial fractions; otherwise it behaves exactly
        like Simplify.
        """

        def run(
            node: Node, context: engine.Context, state: ParseState
        ) -> Awaitable[engine.Result]:
            return self.runner.expand(node, context, amount, variables, state)

        return await self._command(request, "Expd", run)

    async def solve(
        self,
        request: str,
        variables: Sequence[str] = (),
        bounds: tuple[str, str] | None = None,
    ) -> list[Entry]:
        """Append one entry per solution of the expression `request` names.

        Two roots give two entries, a system gives one holding the solution vector, and no
        solutions give none, which the UI reports with a message. Entries are returned in
        order, and the last one is selected.

        Unlike Simplify, soLve acts on the whole entry even if part is highlighted, since
        a solution can't be spliced back into the rest. `variables` lists what to solve
        for, in the order chosen, and `bounds` is the pair of expressions Approximate
        precision asks for.

        Raises `DeriveSyntaxError` and appends nothing if `request` or a bound doesn't
        parse.
        """
        entry, target = self._requested(request)
        interval = (
            None
            if bounds is None
            else (
                parse_expression(bounds[0], self.state).node,
                parse_expression(bounds[1], self.state).node,
            )
        )
        # Like every engine command, this empties the unremove buffer.
        self.removed = []
        source = AUTHORED if entry is None else f"#{entry.number}"
        results = await self.runner.solve(
            target if entry is None else entry.value,
            self.context,
            variables,
            interval,
            self.state,
        )
        return [
            self._append(result.text, result.node, f"Solve({source})", result.value)
            for result in results
        ]

    async def solve_variables(self, request: str) -> tuple[str, ...]:
        """The variables soLve would offer for what `request` names.

        `variables` won't do, because soLve acts on the whole entry rather than the
        highlighted part. And when the expression simplifies to one with no variables, as
        `x/x = 1` does, the variables it was written in are listed, since there is still
        a question to answer about each.

        Raises `DeriveSyntaxError` when `request` does not parse.
        """
        _, target = self._requested(request)
        return await self.runner.expression_variables(target, self.context, written=True)

    def equations(self, request: str) -> int:
        """The number of equations in the system `request` names, or zero if not a system.

        This decides how many variables soLve asks for: E times for a system of E
        equations in more than E variables, and at most once for a scalar.

        Raises `DeriveSyntaxError` when `request` does not parse.
        """
        _, target = self._requested(request)
        return engine.equations_in(target)

    def substitute(self, request: str, values: Mapping[str, str]) -> Entry:
        """Append what `request` names with a value written in for each variable.

        `values` maps variable names to replacements; a blank one (Enter on the offered
        name) leaves that variable alone. Substitution is simultaneous, so `y` for `x`
        and `x` for `y` swaps them.

        Nothing is simplified, by design: `a*x^2 + b*x + c` with 2, 3 and 5 written in
        becomes `3*2^2 + 5*2 + c`, for the user to inspect before simplifying.

        Raises `DeriveSyntaxError` and appends nothing if `request` or a value doesn't
        parse.
        """
        entry, target = self._requested(request)
        replacements = [
            (Node(Kind.NAME, 0, 0, (), name), parse_expression(text, self.state).node)
            for name, text in values.items()
            if text.strip()
        ]
        return self._substituted(target, replacements, entry)

    def substitute_part(self, request: str, value: str) -> Entry:
        """Append the entry `request` names with the highlighted part replaced.

        Every structural match of the part in the entry is replaced: substituting for
        `t^3` leaves `t^6` alone. A blank value replaces nothing.

        The result derives from the whole entry, so unlike Simplify of a part, the
        annotation has no quote.

        Raises `DeriveSyntaxError` and appends nothing if `request` or `value` doesn't
        parse.
        """
        entry, target = self._requested(request)
        part = self._highlighted_in(entry)
        replacements = (
            []
            if part is None or not value.strip()
            else [(part, parse_expression(value, self.state).node)]
        )
        return self._substituted(target, replacements, entry)

    def substitutes_part(self, request: str) -> bool:
        """Whether a Substitute for what `request` names would replace a part.

        This decides the prompts: one question with no default for a highlighted part,
        or one per variable for a whole expression.

        Raises `DeriveSyntaxError` when `request` does not parse.
        """
        entry, _ = self._requested(request)
        return self._highlighted_in(entry) is not None

    def target(self, request: str) -> Node:
        """The expression a command for `request` would act on.

        Factor and Expand need it before prompting, to decide which variables to offer
        and whether to ask at all.

        Raises `DeriveSyntaxError` when `request` does not parse.
        """
        return self.named_target(request)[0]

    def named_target(self, request: str) -> tuple[Node, str]:
        """That expression, and what an annotation calls it.

        The name is `#3` for a whole entry, `#3'` for its highlighted part, and `User` for
        typed text. Build and the Calculus commands use it in their annotations.

        Raises `DeriveSyntaxError` when `request` does not parse.
        """
        node = parse_expression(request, self.state).node
        entry = self._labelled(node)
        if entry is None:
            return node, AUTHORED
        part = self._highlighted_in(entry)
        if part is None:
            return entry.node, f"#{entry.number}"
        return part, f"#{entry.number}'"

    def is_variable(self, text: str) -> bool:
        """Whether `text` is a single variable name and nothing else.

        This validates the Calculus variable fields (differentiation, integration or sum
        index variable). It needn't occur in the expression, since differentiating by an
        absent name legitimately gives zero, but it must be a name not already standing
        for something else.
        """
        try:
            node = parse_expression(text, self.state).node
        except DeriveSyntaxError:
            return False
        return node.kind is Kind.NAME and self._is_variable(str(node.value))

    async def build(self, node: Node, annotation: str, simplified: bool = False) -> Entry:
        """Append a built expression, unsimplified, as Build leaves it.

        `node` is assembled from resolved operands, so its spans index nothing; it is
        written out and read back to give the entry proper spans.

        `simplified` means Ctrl-Enter on the operator menu's `Done`. As in the original,
        only the simplified expression is appended, with a wrapped annotation like
        `Simp(#1+#1)`.
        """
        return await self._derived(node, annotation, simplified)

    async def calculus(
        self,
        head: str,
        prefix: str,
        request: str,
        arguments: Sequence[str],
        simplified: bool = False,
    ) -> Entry:
        """Append `HEAD(u, ...)` over what `request` names, unsimplified.

        The Calculus commands compute nothing: `Calculus Differentiate` leaves a `DIF`
        for a later Simplify to evaluate. So one method serves all seven, given the head
        to write and the annotation prefix.

        `arguments` follow the expression, in the head's order, with omitted ones left
        out. As in the original, the annotation names only the expression and the
        variable: `Dif(#1,x)` says nothing about the order.

        Raises `DeriveSyntaxError` and appends nothing if `request` or an argument doesn't
        parse.
        """
        target, source = self.named_target(request)
        parsed = [parse_expression(text, self.state).node for text in arguments]
        name = Node(Kind.NAME, 0, 0, (), head)
        call = Node(Kind.CALL, 0, 0, (name, target, *parsed))
        variable = arguments[0] if arguments else ""
        return await self._derived(call, f"{prefix}({source},{variable})", simplified)

    async def _derived(
        self, node: Node, annotation: str, simplified: bool = False
    ) -> Entry:
        """Append a tree built rather than computed, written out and read back.

        Like every command that appends an answer, this empties the unremove buffer.

        With Ctrl-Enter (`simplified`), the built tree is simplified and only the result
        appended, with an annotation recording both steps: `Simp(Dif(#1,x))` is the
        evaluated derivative of entry 1.
        """
        self.removed = []
        result = engine.replace(node, (), self.state)
        if not simplified:
            return self._append(result.text, result.node, annotation)
        return await self._answered(
            result.node, self.runner.simplify, f"Simp({annotation})"
        )

    async def variables(self, request: str) -> tuple[str, ...]:
        """The variables Factor or Expand would offer for what `request` names.

        Most main first, as in the original, which asks nothing when there are fewer
        than two.

        Raises `DeriveSyntaxError` when `request` does not parse.
        """
        return await self.runner.expression_variables(self.target(request), self.context)

    async def relation(self, request: str) -> str | None:
        """The relation operator of what `request` names once evaluated, or None.

        Plot asks this alongside `variables`: `IM(u = v)` is a call as written but an
        equation once evaluated, and the equation is what gets drawn.

        Raises `DeriveSyntaxError` when `request` does not parse.
        """
        return await self.runner.relation_of(self.target(request), self.context)

    def decomposes(self, request: str) -> bool:
        """Whether what `request` names is a number, which Factor just decomposes.

        Raises `DeriveSyntaxError` when `request` does not parse.
        """
        return engine.decomposes(self.target(request))

    def written_as_ratio(self, request: str) -> bool:
        """Whether what `request` names is a quotient, which Expand asks about.

        Raises `DeriveSyntaxError` when `request` does not parse.
        """
        return engine.written_as_ratio(self.target(request))

    async def _command(self, request: str, prefix: str, run: Command) -> Entry:
        """Run one engine command on what `request` names, and append the answer.

        A bare `#3` is entry 3. If that entry is selected with only part highlighted, the
        part alone is transformed and the rest copied around it, and the annotation gets
        a quote. Any other text is transformed as an expression in its own right and
        annotated as the user's.

        Raises `DeriveSyntaxError` and appends nothing if `request` doesn't parse.
        """
        node = parse_expression(request, self.state).node
        # Deriving an expression empties the unremove buffer, as in the original.
        self.removed = []
        entry = self._labelled(node)
        if entry is None:
            return await self._answered(node, run, f"{prefix}({AUTHORED})")
        if self._highlighted_in(entry) is not None:
            return await self._answered_part(entry, run, prefix)
        return await self._answered(entry.value, run, f"{prefix}(#{entry.number})")

    def _labelled(self, node: Node) -> Entry | None:
        """The entry a bare `#3` names, or None when that is not what this is."""
        if node.kind is not Kind.LABEL:
            return None
        try:
            number = int(str(node.value))
        except ValueError:
            return None
        return self.numbered(number)

    def _requested(self, request: str) -> tuple[Entry | None, Node]:
        """The entry `request` labels (or None), and the whole expression to act on.

        Substitution works on the whole entry even when part is highlighted: the part
        says what to replace, not where.
        """
        node = parse_expression(request, self.state).node
        entry = self._labelled(node)
        return entry, node if entry is None else entry.node

    def _highlighted_in(self, entry: Entry | None) -> Node | None:
        """The part of `entry` the selection has highlighted, if it has one.

        Only the selected entry can have a highlighted part; any other entry is named
        whole.
        """
        if entry is None or entry is not self.selected_entry or not self.route:
            return None
        return self.selected_node

    def _substituted(
        self,
        node: Node,
        replacements: Sequence[engine.Replacement],
        entry: Entry | None,
    ) -> Entry:
        """Append `node` with `replacements` written into it, unsimplified.

        `entry` is the source, for the annotation; None means text typed at the prompt.
        """
        # Like every engine command, this empties the unremove buffer.
        self.removed = []
        result = engine.replace(node, replacements, self.state)
        source = AUTHORED if entry is None else f"#{entry.number}"
        return self._append(result.text, result.node, f"Sub({source})")

    async def _answered(self, node: Node, run: Command, annotation: str) -> Entry:
        result = await run(node, self.context, self.state)
        return self._append(result.text, result.node, annotation, result.value)

    async def _answered_part(self, entry: Entry, run: Command, prefix: str) -> Entry:
        """`entry` again, with the highlighted part of it transformed.

        The answer is spliced into the entry's text and the whole line reparsed, so the
        new entry's spans index its text.
        """
        part = self.selected_node
        assert part is not None
        result = await run(part, self.context, self.state)
        text, node = self._spliced(entry, part, result.text)
        return self._append(text, node, f"{prefix}(#{entry.number}')")

    def _spliced(self, entry: Entry, part: Node, fragment: str) -> tuple[str, Node]:
        """`entry`'s text with `fragment` written where `part` stands, and read back.

        Missing parentheses would change the meaning, so the parenthesized splice is the
        fallback. Extra ones are harmless but end up in saved files, and the slot often
        has parentheses already: the highlighted sum in `(5*x - 3*x + 1)^7` sits inside a
        pair the line keeps. So the bare splice is used whenever it parses to the same
        expression, checked by parsing rather than guessed from precedence.
        """
        fenced = entry.text[: part.start] + f"({fragment})" + entry.text[part.end :]
        node = parse_expression(fenced, self.state).node
        bare = entry.text[: part.start] + fragment + entry.text[part.end :]
        try:
            plain = parse_expression(bare, self.state).node
        except DeriveSyntaxError:
            return fenced, node
        return (bare, plain) if same_expression(plain, node) else (fenced, node)

    # -- declaring ---------------------------------------------------------
    #
    # Every Declare command writes the expression the user could have typed and authors
    # it, as the original does (the manual gives the equivalent line for each). So
    # nothing here touches the symbol table, domains or definitions directly.

    def declare_domain(self, name: str, kind: str, bounds: Bounds | None = None) -> Entry:
        """Author `x :ε Real (0, ∞)`: the domain of one variable.

        `kind` is Integer, Real, Complex or Nonscalar, and `bounds` the interval, allowed
        only for the first two. The name `default` sets the fallback domain for
        undeclared variables.
        """
        interval = "" if bounds is None else f" {bounds.text}"
        return self.author(f"{name} {DOMAIN_OPERATOR} {kind}{interval}")

    def declare_value(self, name: str, value: str = "") -> Entry:
        """Author `area := π r^2`, or `area :=` to take the value away."""
        return self.author(f"{name} :={_after(value)}")

    def declare_function(self, name: str, definition: str) -> Entry:
        """Author `HYP(a, b) := √(a^2 + b^2)`, deriving the parameters.

        The definition's variables become the parameters, most main first; the manual
        recommends an authored definition when their order or number matters. A
        definition with no variables yields a plain value assignment like `q := 5`, as
        in the original.
        """
        variables = self.definition_variables(definition)
        return self.declare_arbitrary(name, variables, definition)

    def declare_arbitrary(
        self, name: str, variables: Sequence[str], definition: str = ""
    ) -> Entry:
        """Author `F(x, y) :=`: a function with parameters and no body.

        The original calls this an arbitrary function. With no parameters, the line is
        just `g :=`.
        """
        head = f"{name}({', '.join(variables)})" if variables else name
        return self.author(f"{head} :={_after(definition)}")

    def declare_vector(self, elements: Sequence[str]) -> Entry:
        """Author `[1, 2, x]`, one element per answer collected."""
        return self.author(_vector(elements))

    def declare_matrix(self, rows: Sequence[Sequence[str]]) -> Entry:
        """Author `[[1, 2], [3, 4]]`: a vector of rows, each of the same width."""
        return self.author(_vector([_vector(row) for row in rows]))

    # -- what the Declare screens open on ----------------------------------

    def declared_as(self, name: str) -> str:
        """The Declare Variable menu word matching `name`'s current declaration.

        That is its declared domain, `Value` if declared without one, or `Real` (the
        implicit domain) if undeclared.
        """
        known = self.state.variables.get(self._canonical(name))
        if known is None:
            return str(engine.DomainKind.REAL)
        return known.domain or VALUE

    def declared_interval(self, name: str) -> str:
        """The interval menu word matching `name`'s domain.

        `All` if there is no interval, the interval's name if it has one, or `Interval`
        for unnamed bounds.
        """
        domain = self.domains.get(self._canonical(name))
        if domain is None or not domain.has_interval:
            return ALL
        bounds = self.bounds_of(name)
        for word, named in NAMED_INTERVALS.items():
            if named == bounds:
                return word
        return INTERVAL

    def bounds_of(self, name: str) -> Bounds:
        """The bounds the Declare Variable Interval screen opens on.

        The variable's own interval if it has one, else the whole real line.
        """
        domain = self.domains.get(self._canonical(name))
        if domain is None or not domain.has_interval:
            return Bounds()
        low, high = domain.low, domain.high
        return Bounds(
            f"-{INFINITY}" if low is None else _bound_text(low),
            INFINITY if high is None else _bound_text(high),
            domain.closed_low,
            domain.closed_high,
        )

    # -- what the Declare screens will take --------------------------------

    def declarable(self, name: str) -> bool:
        """Whether `name` may be declared.

        It must be a single name, multi-character even in Character input mode (so this is
        a lexical check, not a parse), and not a pre-defined function or constant, which
        can't be redeclared (manual sections 4.10 and 4.12).

        Names the session itself defined may be redeclared, which is how `Declare
        Variable` turns a user function back into a variable and vice versa.
        """
        if not is_name(name):
            return False
        known = self.state.resolve(name)
        if known is None or self._is_the_users(known.canonical):
            return True
        return not (known.is_function or known.is_constant or known.is_keyword_operator)

    def _is_the_users(self, name: str) -> bool:
        """Whether `name` is one the session's own lines declared."""
        return name in self.state.functions or name in self.state.variables

    def _canonical(self, name: str) -> str:
        """The recorded form of a name typed on a Declare line, regardless of case."""
        return self.state.lookup(name) or name

    def reads(self, text: str) -> None:
        """Raise `DeriveSyntaxError` unless `text` is an expression.

        Used by commands that collect expressions one at a time, so each vector element
        is checked as it is entered.
        """
        parse_expression(text, self.state)

    def is_bound(self, text: str) -> bool:
        """Whether `text` is something an interval bound may say.

        Like the original, this accepts a number or an infinity: `-5`, `1/2` and `2.5`,
        but not `a + 1`.
        """
        try:
            node = parse_expression(text, self.state).node
        except DeriveSyntaxError:
            return False
        return _is_bound(node)

    def definition_variables(self, definition: str) -> tuple[str, ...]:
        """The variables in `definition`, most main first.

        `Declare Function` uses these as the parameters. Variables with assigned values
        count, since they are still written as variables; constants and function names
        don't.

        Raises `DeriveSyntaxError` when `definition` does not parse.
        """
        if not definition.strip():
            return ()
        node = parse_expression(definition, self.state).node
        return engine.main_order(
            (
                str(found.value)
                for found in _subtrees(node)
                if found.kind is Kind.NAME and self._is_variable(str(found.value))
            ),
            self.order,
        )

    def _is_variable(self, name: str) -> bool:
        """Whether `name` is free to be a variable, or is a pre-defined name."""
        known = self.state.resolve(name)
        return known is None or not (
            known.is_function or known.is_constant or known.is_keyword_operator
        )

    # -- removing ----------------------------------------------------------

    def numbered(self, number: int) -> Entry | None:
        """The entry labelled `number`, or None when the history holds no such.

        Labels don't correspond to positions after removing and unremoving, so they are
        looked up.
        """
        return next((entry for entry in self.entries if entry.number == number), None)

    def remove(self, first: int, last: int) -> int:
        """Take out the block `first` and `last` delimit, and say how many went.

        The labels delimit a physically contiguous block, which needn't be numerically
        contiguous, since unremove can put entries out of order. Either end may come
        first; the same label twice removes one entry.

        The removed entries replace the unremove buffer. Labels are not reused and
        nothing is renumbered, except that emptying the history restarts labels at 1, as
        in the original (a restored entry whose label was taken since gets a fresh one).
        Definitions made by the removed lines remain in effect.

        The selection moves to the entry now in the block's place, or to the last entry
        if the block was at the end.

        Raises `KeyError` when either label names no entry.
        """
        start = self._index_of(first)
        stop = self._index_of(last)
        if start > stop:
            start, stop = stop, start
        self.removed = self.entries[start : stop + 1]
        del self.entries[start : stop + 1]
        if not self.entries:
            self._next_number = 1
        self.selected = min(start, len(self.entries) - 1) if self.entries else None
        self.route = ()
        return len(self.removed)

    def unremove(self, before: int | None = None) -> int:
        """Put the removed expressions back, and say how many came.

        They go before the entry labelled `before`; None (typing `end` in the field)
        appends them at the end.

        The buffer is kept, so a block can be restored repeatedly. A restored entry keeps
        its label if free and gets a fresh one otherwise, so labels stay unique. The next
        authored entry is numbered past every label in the history.

        The selection moves to the entry the block went before, or to the last restored
        entry if they went at the end.

        Raises `KeyError` when `before` names no entry.
        """
        at = len(self.entries) if before is None else self._index_of(before)
        taken = {entry.number for entry in self.entries}
        restored = [self._relabelled(entry, taken) for entry in self.removed]
        self.entries[at:at] = restored
        self._next_number = max(self._next_number, max(taken, default=0) + 1)
        following = at + len(restored)
        self.selected = following if before is not None else following - 1
        self.route = ()
        return len(restored)

    def _index_of(self, number: int) -> int:
        """The index of the entry labelled `number`."""
        for index, entry in enumerate(self.entries):
            if entry.number == number:
                return index
        raise KeyError(number)

    def _relabelled(self, entry: Entry, taken: set[int]) -> Entry:
        """`entry` under its own label, or under a fresh one when that is taken.

        `taken` holds labels in use, including ones restored so far, and gains the label
        assigned. After an emptied history restarts from 1, a fresh label can itself be
        taken, so taken ones are skipped.
        """
        if entry.number in taken:
            while self._next_number in taken:
                self._next_number += 1
            entry = replace(entry, number=self._next_number)
            self._next_number += 1
        taken.add(entry.number)
        return entry

    # -- moving ------------------------------------------------------------

    def move_block(self, before: int | None, first: int, last: int) -> int:
        """Put the block `first` and `last` delimit in front of `before`.

        None (typing `end` in the field) moves it to the end. The block is delimited as
        in `remove`.

        Entries keep their labels, leaving them out of sequence for `renumber`. The
        unremove buffer is untouched.

        A destination inside the block does nothing, silently, as in the original.
        Otherwise the highlight moves to the entry the block went before, or to the last
        moved entry if they went at the end.

        Returns how many entries moved (zero in the no-op case).

        Raises `KeyError` when any of the three labels names no entry.
        """
        start = self._index_of(first)
        stop = self._index_of(last)
        if start > stop:
            start, stop = stop, start
        at = len(self.entries) if before is None else self._index_of(before)
        if start <= at <= stop:
            return 0
        block = self.entries[start : stop + 1]
        del self.entries[start : stop + 1]
        if at > stop:
            at -= len(block)
        self.entries[at:at] = block
        self.selected = len(self.entries) - 1 if before is None else at + len(block)
        self.route = ()
        return len(block)

    # -- managing ----------------------------------------------------------

    def renumber(self) -> None:
        """Label the entries 1, 2, 3 ... in the order they physically sit.

        This fixes the sequence after `moVe`, `Remove` and `Unremove`. References to moved
        labels are updated, both in annotations (as the manual promises "provided the
        numbers are preceded by the # character") and inside expressions.

        Expressions must be rewritten because Rederive keeps `#3` as a reference, whereas
        the original resolved it when reading the line. Otherwise it would silently point
        at whichever entry gets the number; `merge` makes the same argument.

        Entries whose text changed are re-rendered; the rest keep their renders. The
        selection stays put, since relabeling changes no expression's shape.

        The next authored entry is numbered one past the new last label.
        """
        if not self.entries:
            return
        moved = {
            entry.number: number
            for number, entry in enumerate(self.entries, start=1)
            if entry.number != number
        }
        self._next_number = len(self.entries) + 1
        if not moved:
            return
        self.entries = [
            self._renumbered(entry, number, moved)
            for number, entry in enumerate(self.entries, start=1)
        ]

    def _renumbered(self, entry: Entry, number: int, moved: dict[int, int]) -> Entry:
        """`entry` under its new label, its references renamed by `moved`."""
        annotation = _rewrite_annotation(entry.annotation, moved.get)
        text = _rewrite_labels(entry.text, entry.node, moved.get)
        if text == entry.text:
            return replace(entry, number=number, annotation=annotation)
        node = parse_expression(text, self.state).node
        return Entry(number, text, node, render(node, self.options), annotation)

    def annotate(self, number: int, annotation: str) -> None:
        """Say where the entry labelled `number` came from.

        This edits the same field as the automatic `User` and `Simp(#3)`, shown in the
        status line and written by `Transfer Save Derive`. Nothing is appended and the
        selection does not move.

        Raises `KeyError` when `number` names no entry.
        """
        index = self._index_of(number)
        self.entries[index] = replace(
            self.entries[index], annotation=annotation.strip()
        )

    def order_list(self, text: str) -> tuple[str, ...] | None:
        """The variable order list `text` spells, or None when it is not one.

        The words are variable names separated by spaces or commas, canonicalized as in
        an expression, so `X` becomes `x` unless `CaseMode := Sensitive`. A word that is
        not a variable name, or a repeated variable, rejects the whole line. An empty
        line gives an empty list, ordering all variables alphabetically.

        The original stores whatever is typed (`q 2 + w` verbatim), but reproducing that
        would be a bug, not fidelity.
        """
        names: list[str] = []
        for word in text.replace(",", " ").split():
            if not self.declarable(word) or self.state.is_function(word):
                return None
            name = self.state.canonical_variable(word)
            if name in names:
                return None
            names.append(name)
        return tuple(names)

    # -- files -------------------------------------------------------------

    def save(self, path: Path, first: int | None = None, last: int | None = None) -> int:
        """Write the history to `path`, and say how many entries went.

        `first` and `last` bound the label numbers written (the Range option); by
        default the whole history is written. Annotations and line length follow the
        `Transfer Save Options` settings.

        Entries are written from their trees, so `x (x + 1)` is saved as `x*(x+1)`, as in
        the original.
        """
        records = self._records(first, last)
        platform.current().storage().write(path, self._file_text(records))
        self.file = path
        return len(records)

    def written(self, first: int | None = None, last: int | None = None) -> str:
        """The file `save` would write, without writing it anywhere.

        A browser puts the same text in a shared link, so a worksheet reads identically
        from either.
        """
        return self._file_text(self._records(first, last))

    def _records(self, first: int | None, last: int | None) -> list[worksheet.Record]:
        """The history as a file's records: an expression each, with its note."""
        return [
            worksheet.Record(write_expression(entry.node), self._noted(entry))
            for entry in self._block(first, last)
        ]

    def _file_text(self, records: Iterable[worksheet.Record]) -> str:
        """Records as a math file, laid out the way the save settings ask for."""
        return worksheet.write(
            records,
            length=int(self.settings["SaveLength"]),
            annotations=self.settings["SaveAnnotation"] == "Save",
        )

    def _block(self, first: int | None, last: int | None) -> Iterator[Entry]:
        """The entries a save writes: the label numbers `first` to `last`."""
        return (
            entry
            for entry in self.entries
            if (first is None or entry.number >= first)
            and (last is None or entry.number <= last)
        )

    @staticmethod
    def _noted(entry: Entry) -> str:
        """What to write above `entry`; a line the user wrote needs nothing."""
        return "" if entry.annotation == AUTHORED else entry.annotation

    def save_source(
        self,
        path: Path,
        language: Language,
        first: int | None = None,
        last: int | None = None,
    ) -> int:
        """Write the history to `path` as `language` source, and say how many went.

        The same records as `save`, in another notation. Range and the annotation option
        apply, but line length doesn't, since no target language minds long lines.
        Annotations go after the language's comment marker.
        """
        entries = list(self._block(first, last))
        text = worksheet.write(
            [
                worksheet.Record(write_source(entry.node, language), self._noted(entry))
                for entry in entries
            ],
            length=None,
            annotations=self.settings["SaveAnnotation"] == "Save",
            comment=f"{language.comment} ",
        )
        platform.current().storage().write(path, text)
        self.file = path
        return len(entries)

    def save_state(self, path: Path) -> None:
        """Write every system control setting to `path`, and the order list.

        `file` is not updated, since a state file is not the worksheet.
        """
        platform.current().storage().write(path, self.state_written())

    def state_written(self) -> str:
        """The state file `save_state` would write, without writing it anywhere.

        The settings counterpart of `written`: a browser tab stores its settings as the
        text a file would hold.
        """
        return state.write(self.settings, self.order)

    def load_state(self, path: Path) -> int:
        """Apply the settings in `path`, and say how many lines would not take.

        The order list is restored too. It is validated here rather than in the file
        reader, because what counts as a variable name depends on the symbol table, and
        an invalid one counts as a refused line.

        The history is unchanged: existing entries keep their renders, and only new
        ones reflect the new settings.
        """
        refused, ordering = state.read(worksheet.text_of(path), self.settings)
        if ordering is None:
            return refused
        names = self.order_list(ordering)
        if names is None:
            return refused + 1
        self.order = names
        return refused

    def load(self, path: Path) -> int:
        """Replace the history with the expressions in `path`.

        Numbering restarts at one, as in the original. Existing definitions remain
        (clearing them is a separate command), and the file's definitions are applied as
        it is read.

        Returns how many of the file's expressions failed to parse and were skipped.

        The file is read before the history is cleared, so a bad path loses nothing.
        """
        text = worksheet.text_of(path)
        self.entries = []
        self.selected = None
        self.route = ()
        self._next_number = 1
        return self._read(path, text)

    def merge(self, path: Path) -> int:
        """Append the expressions in `path` to the history.

        Numbering continues, and label references are shifted to match: a file whose
        third expression is `#1 + 2` still refers to its own first expression when merged
        after five others. The original resolves references while reading; Rederive
        keeps them, so they must be shifted.
        """
        return self._read(path, worksheet.text_of(path))

    def load_utility(self, path: Path) -> int:
        """Take the definitions in `path` without showing them.

        A utility file is a library whose definitions should become available, not
        appear on screen. Nothing is appended, numbering is unchanged, and `file` keeps
        naming the current worksheet.

        The file is read in Word input mode; see `reading_file`.

        Returns how many of the file's expressions did not parse.
        """
        source = Source.from_file(worksheet.text_of(path), str(path))
        skipped = 0
        with self.reading_file():
            for start, stop in source_lines(source):
                try:
                    line = source.text[start:stop].strip()
                    result = parse_expression(line, self.state)
                except DeriveSyntaxError:
                    skipped += 1
                    continue
                for declaration in result.declarations:
                    self.declare(declaration)
                self._define(result.node, result.declarations)
        return skipped

    @contextmanager
    def reading_file(self) -> Iterator[None]:
        """Read what happens inside in Word input mode, as the original reads files.

        Every file the original reads (a utility file, a worksheet, a demo's steps) is
        read in Word mode whatever the session's mode: SOLVE.MTH uses `xk` as one
        variable, not `x*k`. Character mode applies only to the Author line. Afterwards
        the mode is restored from the settings, so a file that assigns `InputMode`
        changes it for the rest of the file and keeps it changed, like any setting
        assignment.
        """
        self.state.declare(SettingDeclaration("InputMode", str(InputMode.WORD)))
        try:
            yield
        finally:
            self.state.declare(
                SettingDeclaration("InputMode", str(self.settings["InputMode"]))
            )

    def load_data(self, path: Path) -> int:
        """Append the matrices in `path`, and say how many blocks would not read.

        As in the original, data is appended rather than replacing the history. A block
        containing a non-number is skipped and counted.
        """
        skipped = 0
        for block in data.blocks(worksheet.text_of(path)):
            try:
                self.author(data.matrix(block))
            except (ValueError, DeriveSyntaxError):
                skipped += 1
        self.file = path
        return skipped

    def _read(self, path: Path, text: str) -> int:
        """Append what `path` holds, and say how many lines would not parse.

        As in the original, a line that doesn't parse is skipped and the rest is read, so
        one bad line doesn't lose the whole file. The file is read in Word input mode;
        see `reading_file`.
        """
        source = Source.from_file(text, str(path))
        annotations = worksheet.annotations_of(text)
        offset = self._next_number - 1
        skipped = 0
        with self.reading_file():
            for start, stop in source_lines(source):
                annotation = annotations.get(source.locate(start)[0], AUTHORED)
                try:
                    self._entered(source.text[start:stop].strip(), annotation, offset)
                except DeriveSyntaxError:
                    skipped += 1
        self.file = path
        return skipped

    def _entered(self, text: str, annotation: str, offset: int) -> None:
        """Append one line of a file as an entry, with its label references shifted.

        The line is redrawn like an authored line, normalizing hand-written files; files
        this program wrote come back unchanged.
        """
        result = parse_expression(text, self.state)
        if offset:
            text = _shift_labels(text, result.node, offset)
            result = parse_expression(text, self.state)
        result = self._redrawn(result)
        for declaration in result.declarations:
            self.declare(declaration)
        self._define(result.node, result.declarations)
        self._append(
            result.source.text, result.node, _shift_annotation(annotation, offset)
        )

    # -- clearing ----------------------------------------------------------

    def clear_expressions(self) -> None:
        """Empty the history, and start the numbering again at one.

        Definitions remain; the manual treats the four clear commands as separate. The
        unremove buffer is emptied too.
        """
        self.entries = []
        self.removed = []
        self.selected = None
        self.route = ()
        self._next_number = 1

    def clear_variables(self) -> None:
        """Forget every value and domain a line has assigned.

        The assigning expressions stay on screen, but a `v` that stood for 7 becomes a
        free variable again.
        """
        self.assignments.clear()
        self.domains.clear()
        self.state.clear(ClearWhat.VARIABLES)

    def clear_functions(self) -> None:
        """Forget every function a line has defined.

        Names are removed from the symbol table too, so a later `F(3)` reads as `f*3`, as
        in the original.
        """
        self.functions.clear()
        self.state.clear(ClearWhat.FUNCTIONS)

    def clear_all(self) -> None:
        """Clear expressions, variables and functions."""
        self.clear_expressions()
        self.clear_variables()
        self.clear_functions()

    # -- selection ---------------------------------------------------------

    @property
    def selected_entry(self) -> Entry | None:
        if self.selected is None:
            return None
        return self.entries[self.selected]

    @property
    def selected_region(self) -> Region | None:
        """The region the route points at, or None when nothing is selected."""
        entry = self.selected_entry
        if entry is None:
            return None
        return entry.layout.at(self.route)

    @property
    def selected_node(self) -> Node | None:
        """The selected subexpression, which an operation would act on."""
        region = self.selected_region
        return region.node if region is not None else None

    @property
    def highlighted_text(self) -> str | None:
        """The highlighted text, sliced from the entry's own text.

        That is the whole entry or the selected subexpression's span. Slicing rather
        than reprinting the node preserves the exact spelling. None when nothing is
        selected.
        """
        entry = self.selected_entry
        if entry is None:
            return None
        node = self.selected_node
        if node is None or not self.route:
            return entry.text
        return entry.text[node.start : node.end]

    def selection_rect(self) -> Rect | None:
        """The rectangle to invert, in the selected entry's own render.

        Coordinates index the entry's `layout.lines`; the work area offsets them.
        """
        region = self.selected_region
        return region.rect if region is not None else None

    def _siblings(self) -> tuple[Region, ...]:
        """The regions the selection sits among, empty at the top level."""
        entry = self.selected_entry
        if entry is None or not self.route:
            return ()
        parent = entry.layout.at(self.route[:-1])
        return parent.children if parent is not None else ()

    def _descend(self, route: Route) -> bool:
        """Step from `route` into the operand last selected there, else the first one.

        Fails on a node without operands.

        Like the original, there is one memory rather than one per entry, so stepping
        into another expression forgets the first. Moving past an expression as a whole
        is not stepping in and keeps the memory.
        """
        entry = self.selected_entry
        if entry is None:
            return False
        region = entry.layout.at(route)
        if region is None or not region.children:
            return False
        self._owns(entry)
        self.route = route + (self._preferred.get(route, 0),)
        return True

    def _owns(self, entry: Entry) -> None:
        """Point the operand memory at `entry`, forgetting another entry's."""
        if self._preferred_owner is not entry:
            self._preferred = {}
            self._preferred_owner = entry

    def _remember(self) -> None:
        """Remember the current operand as the one to return to in its parent."""
        self._preferred[self.route[:-1]] = self.route[-1]

    # -- navigation --------------------------------------------------------

    def select_entry(self, index: int) -> bool:
        """Select entry `index` as a whole, clamped to the history."""
        if not self.entries:
            return False
        index = max(0, min(index, len(self.entries) - 1))
        moved = index != self.selected or bool(self.route)
        self.selected = index
        self.route = ()
        return moved

    def select_at(self, index: int, row: int, column: int) -> bool:
        """Select what entry `index` draws at that cell of its render.

        A click on another entry selects it whole; further clicks and arrows pick parts.

        Within the selected entry, a click moves one level toward its target, as Right
        and Down do. So clicking the `x` of `(x + 1)/y` selects the numerator, and the
        next click the `x`, making terms of a long sum easy to hit. A click in a different
        branch selects that branch where the two diverge, and a click on an enclosing node
        (the `+` belongs to the sum, not to either term) goes back out to it.

        A cell no subexpression covers, including the label, selects the whole entry.
        """
        if not 0 <= index < len(self.entries):
            return False
        entry = self.entries[index]
        if index != self.selected:
            return self.select_entry(index)
        pointed = entry.layout.route_at(row, column)
        if pointed is None:
            return self.select_entry(index)
        # The common prefix decides the step: down if the cell is inside the selection,
        # across if the routes diverge.
        shared = 0
        for mine, theirs in zip(self.route, pointed):
            if mine != theirs:
                break
            shared += 1
        route = pointed if shared == len(pointed) else pointed[: shared + 1]
        if route == self.route:
            return False
        self._owns(entry)
        self.route = route
        if route:
            self._remember()
        return True

    def jump(self, number: int) -> bool:
        """Select the entry labelled `number`, and say whether one was found.

        A missing label (removed, or below where numbering now starts) selects the entry
        with the smallest label above it. So `0` selects the first entry, and a label
        past the last finds nothing, which the original refuses too.

        Lookup is by label, not position, which matters when unremove has left labels
        out of order.

        Jumping to the already selected entry keeps the highlighted subexpression; any
        other entry is selected whole.
        """
        above = [
            index for index, entry in enumerate(self.entries) if entry.number >= number
        ]
        if not above:
            return False
        index = min(above, key=lambda index: self.entries[index].number)
        if index != self.selected:
            self.select_entry(index)
        return True

    def move_up(self) -> bool:
        """Previous entry, or one level up towards the whole expression."""
        if self.selected is None:
            return False
        if self.route:
            self.route = self.route[:-1]
            return True
        return self.select_entry(self.selected - 1)

    def move_down(self) -> bool:
        """Next entry, or one level down into the operands.

        Only from a whole expression does this move to the next entry; inside one it goes
        deeper and stops at a leaf.
        """
        if self.selected is None:
            return False
        if self.route:
            return self._descend(self.route)
        return self.select_entry(self.selected + 1)

    def move_right(self) -> bool:
        """Into the expression, or on to the next operand.

        On a whole expression either horizontal arrow steps in. Inside, this stops at
        the last operand, without wrapping or climbing out.
        """
        if self.selected_entry is None:
            return False
        if not self.route:
            return self._descend(())
        if self.route[-1] + 1 >= len(self._siblings()):
            return False
        self.route = self.route[:-1] + (self.route[-1] + 1,)
        self._remember()
        return True

    def move_left(self) -> bool:
        """Into the expression, or back to the previous operand.

        On a whole expression this steps in, like `move_right`. Inside, it stops at the
        first operand.
        """
        if self.selected_entry is None:
            return False
        if not self.route:
            return self._descend(())
        if self.route[-1] == 0:
            return False
        self.route = self.route[:-1] + (self.route[-1] - 1,)
        self._remember()
        return True

    def move_first_sibling(self) -> bool:
        if self.selected_entry is None or not self.route:
            return False
        moved = self.route[-1] != 0
        self.route = self.route[:-1] + (0,)
        self._remember()
        return moved

    def move_last_sibling(self) -> bool:
        if self.selected_entry is None or not self.route:
            return False
        last = len(self._siblings()) - 1
        moved = self.route[-1] != last
        self.route = self.route[:-1] + (last,)
        self._remember()
        return moved

    def move_first_entry(self) -> bool:
        return self.select_entry(0)

    def move_last_entry(self) -> bool:
        return self.select_entry(len(self.entries) - 1)

    def move_page_up(self, rows: int) -> bool:
        """Select the expression at the top of a pane of `rows`.

        A page is measured in rows, not entries, since tall expressions fill a pane
        faster. An expression too tall to share the pane is a page of its own, so the
        highlight still moves by one entry.
        """
        if self.selected is None:
            return False
        first, _ = self._pane(rows)
        return self._page_to(first if first < self.selected else self.selected - 1)

    def move_page_down(self, rows: int) -> bool:
        """Select the expression at the bottom of a pane of `rows`.

        If the highlight is already there, the pane scrolls to the farthest entry that
        still fits on screen together with the current one.
        """
        if self.selected is None:
            return False
        _, last = self._pane(rows)
        if last > self.selected:
            return self._page_to(last)
        return self._page_to(self._paneful_below(rows))

    def _page_to(self, index: int) -> bool:
        """Select entry `index` whole, unless the page had nowhere to go.

        At the end of the history nothing moves, including the route, so a highlighted
        subexpression stays highlighted.
        """
        assert self.selected is not None
        index = max(0, min(index, len(self.entries) - 1))
        if index == self.selected:
            return False
        return self.select_entry(index)

    def _pane(self, rows: int) -> tuple[int, int]:
        """The first and last entries a pane of `rows` holds.

        As in the original, the selected expression is at the bottom with as much
        history above it as fits. When the top of the history is reached, the rest of the
        pane is filled downward, leaving the highlight above the bottom edge.
        """
        assert self.selected is not None
        first = last = self.selected
        used = self.entries[first].height
        while first > 0 and used + 1 + self.entries[first - 1].height <= rows:
            first -= 1
            used += 1 + self.entries[first].height
        if first == 0:
            while last + 1 < len(self.entries) and (
                used + 1 + self.entries[last + 1].height <= rows
            ):
                last += 1
                used += 1 + self.entries[last].height
        return first, last

    def _paneful_below(self, rows: int) -> int:
        """The farthest entry below the selected one that fits on the pane with it.

        At least the next entry, whatever its height, so paging never gets stuck.
        """
        assert self.selected is not None
        index = self.selected
        used = self.entries[index].height
        while index + 1 < len(self.entries) and (
            index == self.selected or used + 1 + self.entries[index + 1].height <= rows
        ):
            index += 1
            used += 1 + self.entries[index].height
        return index


def _is_infinite(bound: str) -> bool:
    """Whether a bound is an infinity, however it was spelled."""
    return bound.lstrip("+-").strip().lower() in (INFINITY, _INFINITE)


def _after(text: str) -> str:
    """What follows `:=` on a definition line; empty text clears the definition."""
    return f" {text.strip()}" if text.strip() else ""


def _vector(elements: Sequence[str]) -> str:
    return "[" + ", ".join(elements) + "]"


def _is_bound(node: Node) -> bool:
    """Whether `node` is something an interval bound may say."""
    if node.kind is Kind.NAME:
        return str(node.value) == _INFINITE
    if node.kind is Kind.UNOP and node.value in ("-", "+"):
        return all(_is_bound(child) for child in node.children)
    return engine.decomposes(node)


def _bound_text(node: Node) -> str:
    """A bound as a data entry field shows it: on one line, infinity as a glyph.

    A bound is only a number or an infinity, so one line suffices: a fraction is
    written `1/2`.
    """
    if node.kind is Kind.UNOP and node.value in ("-", "+"):
        return str(node.value) + _bound_text(node.children[0])
    if node.kind is Kind.NAME and str(node.value) == _INFINITE:
        return INFINITY
    return write_expression(node)


def _rewrite_labels(text: str, node: Node, rename: Rename) -> str:
    """`text` with every label it refers to renamed by `rename`.

    Rewritten right to left so earlier spans stay valid. Walking the tree rather than
    the characters leaves a `#3` inside a string literal alone.
    """
    for found in sorted(_subtrees(node), key=lambda found: found.start, reverse=True):
        if found.kind is not Kind.LABEL:
            continue
        try:
            written = int(str(found.value))
        except ValueError:
            continue
        number = rename(written)
        if number is None:
            continue
        text = f"{text[: found.start]}#{number}{text[found.end :]}"
    return text


def _rewrite_annotation(annotation: str, rename: Rename) -> str:
    """`_rewrite_labels` for an annotation, which is text rather than a tree."""

    def renamed(found: re.Match[str]) -> str:
        number = rename(int(found.group(1)))
        return found.group(0) if number is None else f"#{number}"

    return LABEL.sub(renamed, annotation)


def _shift_labels(text: str, node: Node, offset: int) -> str:
    """`text` with every label it refers to shifted by `offset`."""
    return _rewrite_labels(text, node, lambda number: number + offset)


def _shift_annotation(annotation: str, offset: int) -> str:
    """`_shift_labels` for an annotation, which is text rather than a tree."""
    if not offset:
        return annotation
    return _rewrite_annotation(annotation, lambda number: number + offset)


def _same_tree(one: Node, other: Node) -> bool:
    """Whether two trees mean the same, ignoring spans and spelling.

    Nodes compare unequal when they cover different text, as every rewritten line
    does, and `x y` and `x*y` are the same expression. A numeral's digits do count,
    since `0FF` and `255` are equal only in one base.
    """
    return _shape(one) == _shape(other)


def _shape(node: Node) -> tuple:
    """A comparable form of `node`, normalized as the writer normalizes it.

    A bare application becomes a call and `|u|` becomes `ABS(u)`, as the writer
    writes them. A numeral keeps its digits, the one surface spelling that carries
    meaning beyond the value.
    """
    kind, children = node.kind, node.children
    if kind is Kind.APPLY:
        kind = Kind.CALL
    elif kind is Kind.ABS:
        kind = Kind.CALL
        children = (Node(Kind.NAME, 0, 0, (), "ABS"), *children)
    surface = node.surface if kind is Kind.NUMBER else None
    return (kind, node.value, surface, tuple(_shape(child) for child in children))


def _subtrees(node: Node) -> Iterator[Node]:
    """`node` and everything under it.

    A definition need not be the whole line: `[x := 1, y := 2]` defines both.
    """
    yield node
    for child in node.children:
        yield from _subtrees(child)


def _defining(node: Node) -> Iterator[Node]:
    """The subtrees of `node` that can define something.

    Like `_subtrees`, but not inside function bodies, whose assignments happen only
    when the function is called.
    """
    yield node
    if node.kind is Kind.FUNDEF:
        return
    for child in node.children:
        yield from _defining(child)
