"""The Algebra session: the numbered history and what is selected in it.

Pure Python - no Textual, no key names. The UI layer translates key presses
into calls on `Session` and paints the result. This is the session layer, and
it is where the math engine is reached from: the UI never calls a command
itself, it asks the session for one.

Which engine answers is the one thing a session takes from outside. The six
calls that can cost anything go through `runner`, and they are awaited: the
program has to be able to go on drawing while one is in flight, and the caller
that can wait for it is not the same everywhere - a thread on the desktop, an
event on a page. Everything else this module asks the engine for costs nothing
and is asked of `engine.client`, the half that holds no mathematics: that is
what lets the app process draw a worksheet without sympy in it. A command that
computes is a coroutine and every other one is an ordinary call, which is the
line between what may wait and what may not.

Awaiting changes nothing about when the worksheet moves. The session appends an
answer only once the call has returned, so a call that dies - or is aborted
mid-await - leaves the worksheet exactly as the command found it.

The session owns the three things an authored line needs: the parse state, so
that `InputMode`, `CaseMode` and every definition reach the lines that follow;
the settings, which an authored `Name := Value` writes to exactly as an Options
dialog does; and the render of each entry, made once when the entry is authored.

It also owns what the lines have defined - values, function bodies and variable
domains - because only the tree holds those, and every engine command reads
them. `Session.context` is that plus the settings, in the form the engine takes.

The Declare commands are how those definitions are made from the menu rather
than from the author line, and they are thin for that reason: each writes the
expression the user could have written and authors it, which is what the
original does and what leaves one path into the symbol table instead of two.

A render is not remade. Switching the times operator or the display format
changes how later expressions are drawn and leaves the ones already on screen
alone, which is what the original does: what you see is what it looked like
when it was entered.

Entries leave the history only through `remove`, which keeps what it took in
`removed` for `unremove` to put back. That buffer is the whole of what the two
commands share, and any engine command empties it. `move_block` rearranges the
history without taking anything out of it, so it leaves that buffer alone.

The Manage commands that need no engine command of their own are here for the
same reason the Declare ones are: `renumber` puts the labels back in sequence
that removing and unremoving leave out of it, `annotate` edits where an entry
says it came from, and `order_list` reads the variable order list, which is
session state the engine hears about through the context.

`substitute` is the one that does need an engine command. It is beside the
other three rather than among them because what it appends is derived from the
whole entry however little of it was highlighted: the substitution happens
inside the entry, so there is no part to splice an answer back into.

`solve` is the one command that does not go through `_command` at all, because
`_command` appends exactly one entry and soLve appends none, one, or several -
one per solution. It names the whole entry for the same reason Substitute does,
and it owns one piece of state besides the history: `arbitrary`, the counter
the `@n` values come out of, which climbs past every one that reaches the
worksheet and never goes back.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from rederive import platform
from rederive.display import DisplayOptions, Layout, Region, render

# The client half of the engine, under the name the mathematics goes by, because a
# session names the engine constantly and holds none of it: what it may reach for is
# the proxy, the vocabulary and the questions a tree answers by itself. Naming the
# half is what makes that a rule rather than a habit - an `engine.simplify` written
# below does not resolve, and the one place that does need the other half says so in
# an import of its own, which is `_computing_here`.
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

#: How a declaration spells the domain operator and an infinite bound. The
#: parser reads either spelling; these are the ones the original writes.
DOMAIN_OPERATOR = ":ε"
INFINITY = "∞"

#: The name behind that glyph, as a tree carries it.
_INFINITE = "inf"

#: The word the Declare Variable menu offers for a variable that has been
#: declared without a domain, and the two words its interval menu offers for
#: an interval that has no name of its own.
VALUE = "Value"
ALL = "All"
INTERVAL = "Interval"

#: A reference to a numbered entry, wherever one is written: in an expression,
#: and in an annotation such as `Simp(#3)`.
LABEL = re.compile(r"#(\d+)")

#: An arbitrary value, as soLve writes one and as an authored `SOLVE` can mint
#: one through a plain Simplify. The session watches every entry for these so
#: that its counter never hands the same one out twice.
ARBITRARY = re.compile(r"@(\d+)")

#: How a label rewriting names one label: the number to put in its place, or
#: None to leave it as it is.
Rename = Callable[[int], int | None]

#: One engine command, as the session calls it. Every command takes the same
#: three things and gives back an answer to wait for, which is what lets
#: `_command` serve all of them and know nothing about which one it is running.
Command = Callable[[Node, "engine.Context", ParseState], Awaitable["engine.Result"]]


class Runner(Protocol):
    """The six engine calls that can cost anything, as the session makes them.

    Everything else the session asks the engine for - whether a tree is a
    quotient, what its main variable is, how to write it - is a walk over a
    tree and costs nothing, so it stays a direct call on the module. These six
    convert to sympy, and converting alone can hang: `10^10^10` never finishes
    being built.

    They are awaited because the six are the only calls that can take long
    enough to have to be waited for, and because where the waiting happens is
    not the session's business. `remote.RemoteEngine` hands each one to a thread
    that blocks on a pipe to a child process, which is how the app makes Esc
    mean something; `_Here` computes it in this process, which is what a test
    and a direct caller want; and an implementation with neither a thread nor a
    process to spare can resolve one from an event and satisfy the same
    signatures.
    """

    async def simplify(
        self, node: Node, context: engine.Context, state: ParseState | None = ...
    ) -> engine.Result: ...

    async def approx(
        self,
        node: Node,
        context: engine.Context,
        digits: int | None = ...,
        state: ParseState | None = ...,
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
        self, node: Node, context: engine.Context | None = ...
    ) -> tuple[str, ...]: ...


class _Here:
    """The engine computing in this process, wearing the six awaitable calls.

    An answer worked out here is in hand the moment it is asked for, so nothing
    below ever waits: these coroutines suspend at no point, and awaiting one
    costs what the call costs and nothing more. That is exactly why a session
    computing this way cannot be aborted - there is no moment between the ask
    and the answer for an Esc to arrive in - and why the app hands in a proxy
    instead.
    """

    def __init__(self, computing: Any) -> None:
        self._engine = computing

    async def simplify(
        self, node: Node, context: engine.Context, state: ParseState | None = None
    ) -> engine.Result:
        return self._engine.simplify(node, context, state)

    async def approx(
        self,
        node: Node,
        context: engine.Context,
        digits: int | None = None,
        state: ParseState | None = None,
    ) -> engine.Result:
        return self._engine.approx(node, context, digits, state)

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
        self, node: Node, context: engine.Context | None = None
    ) -> tuple[str, ...]:
        return self._engine.expression_variables(node, context)


def _computing_here() -> Runner:
    """The engine itself, for a session that was handed no proxy to compute through.

    The one place the session layer reaches past the client half, and a function
    rather than an import so that reaching happens when a session is made and not
    when this module is read. What it costs is sympy, some four hundred modules of
    it, which is why the app never takes this door: it hands in a `RemoteEngine`, and
    the process that draws the screen goes on knowing no mathematics at all. Tests
    and direct callers take it, and want exactly what it does.

    `_Here` is the awaitable face over it and nothing more. The module's own calls
    stay ordinary functions, since the worker calls them in the process where the
    computing happens and has nothing to wait for either.
    """
    from rederive.engine import computing

    return _Here(computing)


@dataclass(frozen=True)
class Bounds:
    """An interval as a declaration writes it: two bounds, each open or closed.

    The bounds are text rather than trees, because this is what a data entry
    field holds and what a declaration is written from. The default is the
    whole real line, which is what the original opens its bounds screen on.

    An infinite bound is always open however it was set, since a variable
    cannot be declared infinite (manual section 4.10).
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


#: The bounds each word of the interval menu stands for. `All` is not among
#: them: it is the whole line, which is written as no interval at all.
NAMED_INTERVALS: dict[str, Bounds] = {
    "Positive": Bounds("0", INFINITY),
    "Negative": Bounds(f"-{INFINITY}", "0"),
    "nonpoSitive": Bounds(f"-{INFINITY}", "0", closed_high=True),
    "nonneGative": Bounds("0", INFINITY, closed_low=True),
}


@dataclass(frozen=True)
class Entry:
    """One numbered line of the history: what was authored, and how it looks.

    `text` is the expression in author notation and not the line that was
    typed: an authored line is written back out from its tree, so `x (x + 1)`
    is held as `x*(x+1)` and `(NOT p) OR (q AND r)` as `NOT p OR q AND r`.
    `node`'s spans index that text, which is what lets a subexpression be taken
    out of the line and put back into it. What the user sees is `layout`, drawn
    from the same tree in the glyphs the screen has.

    `annotation` is where the entry came from, as the status line reports it:
    `User` for a line that was written, `Simp(#3)` for one a Simplify derived
    from entry 3, and `Simp(#3')` when only the highlighted part of entry 3
    was simplified.
    """

    number: int
    text: str
    node: Node
    layout: Layout
    annotation: str = AUTHORED
    exact: Node | None = None
    """The value behind the line, where the notation shows less than all of it.

    A decimal style shows one third as `0.333333` and saves it that way, and
    that text read back is a different number. The original goes on computing
    with the third, so `3·#n` is 1 and not 0.999999, and this is what makes
    that so: what is drawn and saved is `node`, what a later command is given
    is this. None where the two are the same, which is every authored line and
    every answer under Rational notation.
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

    The selection is an entry index together with a route into that entry's
    render; an empty route means the entry is selected as a whole. Every
    navigation method returns whether it moved the selection, so the UI can
    beep or stay quiet accordingly.

    Navigation goes through the render rather than through the parse, because
    the original's rule is that you select what you see: `a + b - c` offers
    three terms, and `SIN(x + 1)` offers its argument and never its name.

    Stepping into an expression is not a walk from the left: each node
    remembers the operand the selection last stood on under it, and stepping
    back in returns there rather than to the first. One expression carries that
    memory at a time - see `_preferred`.
    """

    def __init__(
        self, settings: Settings | None = None, runner: Runner | None = None
    ) -> None:
        self.settings = settings if settings is not None else Settings()
        #: Who answers the six engine calls that can cost something. The
        #: module by default, which computes here and cannot be interrupted;
        #: the app hands in a proxy to a child process instead, so that a
        #: computation can be aborted and its appetite capped.
        self.runner: Runner = runner if runner is not None else _computing_here()
        self.state = ParseState()
        self.entries: list[Entry] = []
        self.selected: int | None = None
        self.route: Route = ()
        #: Which operand each node of the expression being explored was last
        #: left on, keyed by the route to the node. What stepping in returns
        #: to. Only one expression has it at a time: `_preferred_owner` says
        #: which, and stepping into any other throws it away.
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
        #: The next free `@n`, which soLve mints its arbitrary values from.
        #: Session-global and monotone: solving `x = x` three times gives `@1`,
        #: `@2` and `@3`, and nothing is ever reused.
        self.arbitrary = 1
        self._next_number = 1
        self.settings.watch(self._settings_changed)
        self._settings_changed(PARSING_SETTINGS)

    def copy(self) -> "Session":
        """This session again, as splitting a window gives the new half one.

        The two share their settings, which are the system's and not a
        window's, and the engine that answers for them. Everything else is
        copied, so the histories fork: what is authored in one window from
        here on is nothing to the other, down to the label numbers.

        Entries are shared rather than copied because nothing ever edits one -
        annotating replaces the entry in the list - so two histories may hold
        the same entry and neither can change it under the other.
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
        """Stop listening to the settings, which a closed window's session must.

        A session mirrors three of the settings into its parse state and stays
        subscribed for as long as it lives. Closing the window it belonged to
        is where that ends.
        """
        self.settings.unwatch(self._settings_changed)

    # -- authoring ---------------------------------------------------------

    def author(self, text: str) -> Entry:
        """Parse `text`, append it as a new entry, and select it as a whole.

        Raises `DeriveSyntaxError` and appends nothing when the line does not
        parse. Nothing is simplified: an accepted expression is inert until the
        user asks for something to be done to it (R-HIST1), and label numbers
        only ever increase (R-HIST2).

        What goes up is the expression rather than the line - see `_redrawn` -
        so what the entry says is what was parsed and not how it was spelled.

        What the line declares is applied before it is drawn, so a hand-written
        `DisplayFormat := Compressed` prints itself compressed, as it does in
        the original.
        """
        result = self._redrawn(parse_expression(text, self.state))
        for declaration in result.declarations:
            self.declare(declaration)
        self._define(result.node, result.declarations)
        return self._append(result.source.text, result.node)

    def _redrawn(self, result: ParseResult) -> ParseResult:
        """`result` written back out from its tree and read again.

        What lands on the worksheet is the expression rather than the line that
        was typed, which is what the original puts up: `(NOT p) OR (q AND r)`
        goes up as `NOT p OR q AND r`, the fences the grammar does not need
        being gone. Nothing is reordered - `q OR p` stays `q OR p` - because
        writing is the parser run backwards and not a normal form.

        It is read again rather than only written, so that the tree's spans
        index the text the entry carries: that is what lets a subexpression be
        copied out of the line and spliced back into it.

        The reading is under the state the writing was read under, before what
        the line declares is applied, or `InputBase := 16` would come back as
        twenty-two. A tree the round trip does not return unchanged keeps the
        line as it was typed, since an entry has to say what the user wrote
        before it says it tidily.
        """
        try:
            written = parse_expression(write_expression(result.node), self.state)
        except DeriveSyntaxError:
            return result
        return written if _same_tree(written.node, result.node) else result

    def record(self, setting: str) -> Entry:
        """Append the `Name := Value` a settings change records.

        The expression is built rather than parsed, because the settings have
        already changed by the time it is written: `InputBase := 7` does not
        lex in base seven, and Derive records the expression it constructs
        rather than reading its own record back.
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

        Every entry and not only a soLve's, because a `SOLVE` written on the
        author line mints them too and reaches the history through a plain
        Simplify - and because a worksheet read from a file carries whatever
        the session that wrote it had minted. The counter only ever climbs, so
        no two arbitrary values in one worksheet can stand for one quantity.
        """
        for found in ARBITRARY.finditer(text):
            self.arbitrary = max(self.arbitrary, int(found.group(1)) + 1)

    def _define(self, node: Node, declarations: Iterable[Declaration]) -> None:
        """Record what a line defines: a value, a function body, or a domain.

        The parse state knows which names are taken; what they stand for lives
        in the tree and nowhere else, so this is what carries `x := 5` to the
        next command. A setting is not a variable: `Notation := Mixed` defines
        nothing, and the declarations say which names those are.

        A function body defines nothing by being written down. `DSOLVE1(p, q,
        x, y, x0, y0, a_) := IF("inapplicable" = a_ := ..., ...)` assigns `a_`
        every time it is called and never when it is defined, so the walk stops
        at a body: taking the assignment here would leave `a_` standing for the
        body's own call with the parameters unwritten.
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

        A setting never goes straight to the parse state. It is applied to the
        settings, which mirror it back, so that a value no field takes cannot
        leave the two disagreeing about what base a numeral is written in.
        """
        if isinstance(declaration, SettingDeclaration):
            self.settings.assign(declaration.setting, declaration.value)
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
        """Keep the parse state's modes level with the settings.

        The settings are the one store; the parse state mirrors the three of
        them that decide how a line lexes.
        """
        for setting in changed & PARSING_SETTINGS:
            value = str(self.settings[setting])
            self.state.declare(SettingDeclaration(setting, value))

    # -- commands ----------------------------------------------------------

    @property
    def context(self) -> engine.Context:
        """Everything an engine command's answer may depend on.

        The whole history is offered as labels, so `#3` stands for entry 3
        wherever it is written, and every entry is one whether or not anything
        refers to it.
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

        Simplify with the precision temporarily approximate, at whatever
        `PrecisionDigits` says; the rest is Simplify's story exactly.
        """

        def run(
            node: Node, context: engine.Context, state: ParseState
        ) -> Awaitable[engine.Result]:
            return self.runner.approx(node, context, None, state)

        return await self._command(request, "Approx", run)

    async def factor(
        self,
        request: str,
        amount: engine.Amount = engine.Amount.RATIONAL,
        variables: Sequence[str] = (),
    ) -> Entry:
        """Append the factored form of the expression `request` names.

        `amount` is how far to go and `variables` names the factorization
        variables, in the order they were chosen; the rest is Simplify's story
        exactly.
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

        `variables` names the expansion variables, in the order they were
        chosen, and `amount` is how far the denominator of a ratio is factored
        on the way to partial fractions; the rest is Simplify's story exactly.
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

        The one command that appends any number of entries rather than exactly
        one: two roots make two, a system makes one holding the solution
        vector, and no solutions make none at all - which is what the UI turns
        into its message, there being nothing on the worksheet to say it with.
        The entries come back in the order they were appended, so the selection
        is left on the last of them.

        Unlike Simplify, soLve acts on the whole entry even where part of it is
        highlighted: a solution of a subexpression is not something there is
        any way to splice back around the rest, so `#3` names entry 3 and
        nothing else. `variables` is what to solve for, in the order the user
        chose, and `bounds` the pair of expressions Approximate precision asks
        for.

        Raises `DeriveSyntaxError` and appends nothing when `request` or one of
        the bounds does not parse.
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
        # Deriving an expression empties the unremove buffer, as every other
        # engine command does.
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

        `variables` is not enough on its own: soLve names the whole entry where
        the other commands name a highlighted part of it, so the list has to be
        read off the same expression the command will act on.

        Raises `DeriveSyntaxError` when `request` does not parse.
        """
        _, target = self._requested(request)
        return await self.runner.expression_variables(target, self.context)

    def equations(self, request: str) -> int:
        """How many equations what `request` names is a system of, or zero.

        Which is what decides how many variables soLve asks about: a system of
        E equations in more than E variables is asked about E times, and a
        scalar at most once.

        Raises `DeriveSyntaxError` when `request` does not parse.
        """
        _, target = self._requested(request)
        return engine.equations_in(target)

    def substitute(self, request: str, values: Mapping[str, str]) -> Entry:
        """Append what `request` names with a value written in for each variable.

        `values` holds one replacement per variable name. A blank one leaves
        that variable alone, which is what Enter on the name the prompt offers
        comes to. They all go in at once, so answering `y` for `x` and `x` for
        `y` interchanges the two rather than writing one of them twice.

        Nothing is simplified, which is the point of the command: `a*x^2 + b*x
        + c` with 2, 3 and 5 written in is appended as `3*2^2 + 5*2 + c`, for
        the user to look at before asking for a Simplify.

        Raises `DeriveSyntaxError` and appends nothing when `request` or one of
        the values does not parse.
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

        Every exact match of that part in the whole entry goes, matching being
        structural: substituting for `t^3` leaves the `t^6` beside it alone. A
        blank value replaces nothing.

        The answer is derived from the whole entry, since the substitution
        happens inside it, so the annotation carries no quote - unlike a
        Simplify of the same part, which transforms the part alone.

        Raises `DeriveSyntaxError` and appends nothing when `request` or
        `value` does not parse.
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

        Which is what decides what the command asks for: a highlighted part is
        one question offering nothing, and a whole expression is one question
        per variable.

        Raises `DeriveSyntaxError` when `request` does not parse.
        """
        entry, _ = self._requested(request)
        return self._highlighted_in(entry) is not None

    def target(self, request: str) -> Node:
        """The expression a command for `request` would act on.

        What the command itself resolves, offered on its own because Factor and
        Expand have to look at it before they can ask their questions: which
        variables to offer, and whether to ask anything at all.

        Raises `DeriveSyntaxError` when `request` does not parse.
        """
        return self.named_target(request)[0]

    def named_target(self, request: str) -> tuple[Node, str]:
        """That expression, and what an annotation calls it.

        `#3` for a whole entry, `#3'` for the highlighted part of one, and
        `User` for a line the user wrote out instead - which is what Build and
        the Calculus commands write their annotations from, both of them
        naming their operands rather than describing what was done to them.

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

        All the Calculus variable fields take: the variable to differentiate
        by, to integrate over, or to run a sum's index over. It need not be a
        variable of the expression - differentiating by a name that is not in
        it is a legitimate way to get zero - but it does have to be a name, and
        one that stands for nothing else already.
        """
        try:
            node = parse_expression(text, self.state).node
        except DeriveSyntaxError:
            return False
        return node.kind is Kind.NAME and self._is_variable(str(node.value))

    async def build(self, node: Node, annotation: str, simplified: bool = False) -> Entry:
        """Append a built expression, unsimplified, as Build leaves it.

        `node` is hung together from operands the command resolved as it
        collected them, so its spans index nothing; it is written out and read
        back here, which is what gives the entry spans into the text it shows.

        `simplified` is Ctrl-Enter on the operator menu's `Done`, and it
        appends one entry rather than two: the original enters the simplified
        expression alone, the built form it came from never reaching the
        history, and says so by wrapping the annotation - `Simp(#1+#1)`.
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

        The Calculus commands compute nothing: `Calculus Differentiate` leaves
        a `DIF` standing for the derivative, and taking it is what a Simplify
        after it is for. That is the whole of the command - which is why one
        method serves all seven, told the head to write and the word the
        annotation is spelled with.

        `arguments` are the arguments after the expression, already in the
        order the head takes them and with the ones the command leaves off
        left off. The annotation names only the expression and the variable,
        as the original's does: `Dif(#1,x)` says nothing about the order.

        Raises `DeriveSyntaxError` and appends nothing when `request` or one of
        the arguments does not parse.
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

        Deriving an expression empties the unremove buffer, which every other
        command that appends an answer does with it.

        A command taken with Ctrl-Enter simplifies what it built instead of
        appending it, so that one entry comes of it and not two, and the
        annotation records both steps: `Simp(Dif(#1,x))` is the derivative of
        entry 3, taken.
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

        Most main first, as the original lists them. Fewer than two is no
        choice at all, and the original asks nothing then.

        Raises `DeriveSyntaxError` when `request` does not parse.
        """
        return await self.runner.expression_variables(self.target(request), self.context)

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

        `#3` on its own is entry 3. When that is the entry the selection is in
        and only part of it is highlighted, that part alone is transformed and
        the rest of the entry is copied around it; the annotation marks such an
        entry with a quote. Any other text is an expression in its own right,
        transformed as it stands, and is annotated as the user's.

        Raises `DeriveSyntaxError` and appends nothing when `request` does not
        parse.
        """
        node = parse_expression(request, self.state).node
        # Deriving an expression empties the unremove buffer, which is what the
        # original does with the memory it was holding.
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
        """The entry `request` labels, if it labels one, and the whole of it.

        What a substitution works on, which is the whole entry even where part
        of it is highlighted: the part says what to replace, not what to
        replace it in.
        """
        node = parse_expression(request, self.state).node
        entry = self._labelled(node)
        return entry, node if entry is None else entry.node

    def _highlighted_in(self, entry: Entry | None) -> Node | None:
        """The part of `entry` the selection has highlighted, if it has one.

        Naming `#3` while a subexpression of entry 3 is highlighted names that
        subexpression; any other entry is named whole, wherever the highlight
        happens to be.
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

        `entry` is where it came from, for the annotation to say so; None is a
        line the user typed on the prompt rather than a label.
        """
        # Deriving an expression empties the unremove buffer, as every other
        # engine command does.
        self.removed = []
        result = engine.replace(node, replacements, self.state)
        source = AUTHORED if entry is None else f"#{entry.number}"
        return self._append(result.text, result.node, f"Sub({source})")

    async def _answered(self, node: Node, run: Command, annotation: str) -> Entry:
        result = await run(node, self.context, self.state)
        return self._append(result.text, result.node, annotation, result.value)

    async def _answered_part(self, entry: Entry, run: Command, prefix: str) -> Entry:
        """`entry` again, with the highlighted part of it transformed.

        The answer is spliced into the entry's own text and the whole line read
        back, so that the new entry's spans index its text exactly as an
        authored line's do.
        """
        part = self.selected_node
        assert part is not None
        result = await run(part, self.context, self.state)
        text, node = self._spliced(entry, part, result.text)
        return self._append(text, node, f"{prefix}(#{entry.number}')")

    def _spliced(self, entry: Entry, part: Node, fragment: str) -> tuple[str, Node]:
        """`entry`'s text with `fragment` written where `part` stands, and read back.

        A fence too few would change what the line says, so the fenced splice is
        what the answer falls back on. One too many says the same thing, but the
        text is what a saved file holds, and the slot a part came out of often
        carries fences already: the highlighted sum of `(5*x - 3*x + 1)^7` stood
        between a pair the line still has. So the bare splice is what is taken
        wherever the line reads back as the same expression either way, which is
        the question asked here rather than guessed at from precedence.
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
    # Every Declare command comes to the same thing: it writes the expression
    # the user could have written on the author line, and authors it. That is
    # what the original does - the manual gives the equivalent authored line
    # for each of them - and it is why nothing here has to touch the symbol
    # table, the domains or the definitions: authoring already does all three.

    def declare_domain(self, name: str, kind: str, bounds: Bounds | None = None) -> Entry:
        """Author `x :ε Real (0, ∞)`: the domain of one variable.

        `kind` is Integer, Real, Complex or Nonscalar, and `bounds` the
        interval, which only the first two can carry. `default` as the name
        declares the domain every unnamed variable falls back on.
        """
        interval = "" if bounds is None else f" {bounds.text}"
        return self.author(f"{name} {DOMAIN_OPERATOR} {kind}{interval}")

    def declare_value(self, name: str, value: str = "") -> Entry:
        """Author `area := π r^2`, or `area :=` to take the value away."""
        return self.author(f"{name} :={_after(value)}")

    def declare_function(self, name: str, definition: str) -> Entry:
        """Author `HYP(a, b) := √(a^2 + b^2)`, deriving the parameters.

        The definition's own variables become the parameters, most main first
        - unlike an authored definition, which is where the manual says to go
        when the order or the number of them matters. A definition with no
        variables in it defines no function: `q := 5` is what the original
        writes for one, and that is a variable with a value.
        """
        variables = self.definition_variables(definition)
        return self.declare_arbitrary(name, variables, definition)

    def declare_arbitrary(
        self, name: str, variables: Sequence[str], definition: str = ""
    ) -> Entry:
        """Author `F(x, y) :=`: a function with parameters and no body.

        Which is what the original calls an arbitrary function. With no
        parameters there is no function either, and the line is `g :=`.
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
        """Which word of the Declare Variable menu `name` is already answered by.

        The domain it was declared with, `Value` for a variable declared
        without one, and `Real` for a name nothing has declared - which is the
        domain such a variable has anyway.
        """
        known = self.state.variables.get(self._canonical(name))
        if known is None:
            return str(engine.DomainKind.REAL)
        return known.domain or VALUE

    def declared_interval(self, name: str) -> str:
        """Which word of the interval menu `name`'s domain is already answered by.

        `All` for a domain with no interval, the name of the interval where it
        has one, and `Interval` for bounds that have no name of their own.
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

        The variable's own, where it has an interval; the whole real line
        where it has not.
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

        It has to be one name - multi-character even in Character input mode,
        which is what makes this a lexical question rather than a parse - and
        it may not be one of the pre-defined functions or constants, which
        cannot be redeclared (manual sections 4.10 and 4.12).

        A name the session's own lines defined may be declared again, which is
        how `Declare Variable` turns a user-defined function back into a
        variable and `Declare Function` turns a variable into a function.
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
        """How a name typed on a Declare line is recorded, however it was cased."""
        return self.state.lookup(name) or name

    def reads(self, text: str) -> None:
        """Raise `DeriveSyntaxError` unless `text` is an expression.

        What a command that collects expressions one at a time needs: an
        element of a vector is judged as it is entered rather than when the
        whole vector has been.
        """
        parse_expression(text, self.state)

    def is_bound(self, text: str) -> bool:
        """Whether `text` is something an interval bound may say.

        A number or an infinity, which is what the original takes: it refuses
        `a + 1` and accepts `-5`, `1/2` and `2.5`.
        """
        try:
            node = parse_expression(text, self.state).node
        except DeriveSyntaxError:
            return False
        return _is_bound(node)

    def definition_variables(self, definition: str) -> tuple[str, ...]:
        """The variables in `definition`, most main first.

        What `Declare Function` makes the parameters of the function it is
        defining. A variable with a value assigned is one of them, since it is
        still written as a variable; a constant and a function name are not.

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

        Removing an entry does not free its label, and unremoving one may have
        to find it taken, so a label is worth looking up rather than indexing.
        """
        return next((entry for entry in self.entries if entry.number == number), None)

    def remove(self, first: int, last: int) -> int:
        """Take out the block `first` and `last` delimit, and say how many went.

        The two labels name the ends of a physically contiguous block, which is
        not necessarily a numerically contiguous run of labels: an earlier
        unremove can leave the history in an order its numbers do not follow.
        Either end may be given first, and naming one label twice removes that
        entry alone.

        The removed entries become the unremove buffer, replacing whatever was
        in it. Their labels are not reused and the entries that stay keep
        theirs, so removing renumbers nothing. What the removed lines defined is
        left standing, since a value outlives the line that gave it.

        The selection lands on the entry that takes the block's place, or on the
        last entry when the block was at the end.

        Raises `KeyError` when either label names no entry.
        """
        start = self._index_of(first)
        stop = self._index_of(last)
        if start > stop:
            start, stop = stop, start
        self.removed = self.entries[start : stop + 1]
        del self.entries[start : stop + 1]
        self.selected = min(start, len(self.entries) - 1) if self.entries else None
        self.route = ()
        return len(self.removed)

    def unremove(self, before: int | None = None) -> int:
        """Put the removed expressions back, and say how many came.

        They go in front of the entry labelled `before`; None puts them after
        the last entry, which is what typing `end` in the field asks for.

        The buffer is not emptied, so the same block can be put back more than
        once. A restored entry keeps its own label where the history has left it
        free and takes a fresh one where it has not, which is what keeps two
        entries from answering to the same number.

        The selection lands on the entry the block went in front of, or on the
        last of the restored ones when they went at the end.

        Raises `KeyError` when `before` names no entry.
        """
        at = len(self.entries) if before is None else self._index_of(before)
        restored = [self._relabelled(entry) for entry in self.removed]
        self.entries[at:at] = restored
        following = at + len(restored)
        self.selected = following if before is not None else following - 1
        self.route = ()
        return len(restored)

    def _index_of(self, number: int) -> int:
        """Where the entry labelled `number` sits, which is what a block is cut by."""
        for index, entry in enumerate(self.entries):
            if entry.number == number:
                return index
        raise KeyError(number)

    def _relabelled(self, entry: Entry) -> Entry:
        """`entry` under its own label, or under a fresh one when that is taken."""
        if self.numbered(entry.number) is None:
            return entry
        entry = replace(entry, number=self._next_number)
        self._next_number += 1
        return entry

    # -- moving ------------------------------------------------------------

    def move_block(self, before: int | None, first: int, last: int) -> int:
        """Put the block `first` and `last` delimit in front of `before`.

        None puts it after the last entry, which is what typing `end` in the
        field asks for. The block is read the way `remove` reads one: a
        physically contiguous run, either end first, one label twice for one
        entry.

        The entries keep their labels, so a move leaves them out of sequence
        for `renumber` to put back. Nothing is removed, so the unremove buffer
        is left alone.

        A destination inside the block itself is nothing to do, and the
        original does nothing about it: no beep, no message, and the highlight
        left where it was. Everywhere else the highlight lands on the entry the
        block went in front of, or on the last of the moved ones when they went
        at the end.

        Says how many entries moved, which is none in that one case.

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

        Which is what `moVe`, `Remove` and `Unremove` leave out of sequence.
        Every reference to a label that moved moves with it: in an annotation,
        which is what the manual promises "provided the numbers are preceded by
        the # character", and inside an expression, which it says nothing
        about.

        A `#3` in an expression is rewritten because Rederive keeps such a
        reference as a reference where the original resolved it as it read the
        line. Leaving it alone would silently repoint it at whichever entry
        ends up wearing the number - which is the argument `merge` already
        makes about a file whose expressions refer to each other.

        An entry whose text changed is drawn again, since what it says has
        changed; the rest keep the render they were authored with. The
        selection stays where it is, subexpression and all, relabelling
        changing no expression's shape.

        The next authored entry comes one past the new last label, so that a
        renumbered history goes on counting from where it now ends. A history
        already in sequence is left alone, and an empty one is nothing to
        renumber.
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

        The automatic `User` and `Simp(#3)` are the same field, so this edits
        what the status line already shows and what `Transfer Save Derive`
        already writes above the expression. Nothing is appended and the
        selection does not move: annotating is a note about an expression and
        not an expression.

        Raises `KeyError` when `number` names no entry.
        """
        index = self._index_of(number)
        self.entries[index] = replace(
            self.entries[index], annotation=annotation.strip()
        )

    def order_list(self, text: str) -> tuple[str, ...] | None:
        """The variable order list `text` spells, or None when it is not one.

        The words are variable names separated by spaces or commas, and each is
        recorded as an expression would record it, so `X` goes on the list as
        `x` unless `CaseMode := Sensitive` says the two are different
        variables. A word that is not a variable name refuses the whole line,
        and so does a variable named twice: one variable cannot be in two
        places in one order. An empty line is an empty list, which leaves every
        variable to be ordered alphabetically.

        The original takes whatever is typed - `q 2 + w` was stored verbatim -
        and
        reproducing that would be a bug rather than fidelity.
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

        `first` and `last` bound the block of label numbers written, which is
        what the Range option asks for; by default the whole history goes. The
        settings decide whether annotations go with it and how long a line may
        be, so a file is written the way `Transfer Save Options` was left.

        An entry is written from its tree, not from the text it was typed as,
        so the file says what the expression is rather than how it was reached:
        `x (x + 1)` goes out as `x*(x+1)`, as it does in the original.
        """
        records = self._records(first, last)
        platform.current().storage().write(path, self._file_text(records))
        self.file = path
        return len(records)

    def written(self, first: int | None = None, last: int | None = None) -> str:
        """The file `save` would write, without writing it anywhere.

        Split out because a browser has somewhere else to put it: the same text
        is what a shared link carries, and a worksheet that read one way out of
        a file and another out of a link would be two worksheets.
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

        The same records `save` writes, spelled in another notation: the Range
        block and the annotation option apply exactly as they do to a math
        file, and the line length does not, none of the four targets minding a
        long line. An annotation goes behind the target's own comment marker.
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

        The session's own file is not touched: a state file is not the
        worksheet, and the status line goes on naming the worksheet.
        """
        platform.current().storage().write(path, self.state_written())

    def state_written(self) -> str:
        """The state file `save_state` would write, without writing it anywhere.

        The settings half of `written`, and there for the same reason: a tab
        keeps its settings in a store rather than in a file, and it keeps them
        by asking the session what a file would have said.
        """
        return state.write(self.settings, self.order)

    def load_state(self, path: Path) -> int:
        """Apply the settings in `path`, and say how many lines would not take.

        The order list comes back with them, judged here rather than in the
        file's own reader because what a variable name is depends on this
        session's symbol table. A line of names it will not take is counted
        like any other line that would not take.

        The history is left alone. What a setting change does to the history is
        the same here as anywhere: the entries already drawn keep the render
        they were drawn with, and the next one is drawn the new way.
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

        Numbering starts again at one, as it does in the original. What earlier
        lines defined is left alone: clearing variables and functions is its
        own command, and the file's own definitions are applied as it is read.

        Returns how many of the file's expressions did not parse and were
        therefore left out.

        The file is read before the history goes, so that a name that turns out
        to be nothing costs nothing: what is on screen is still there to try
        again from.
        """
        text = worksheet.text_of(path)
        self.entries = []
        self.selected = None
        self.route = ()
        self._next_number = 1
        return self._read(path, text)

    def merge(self, path: Path) -> int:
        """Append the expressions in `path` to the history.

        Numbering carries on, and every reference to a numbered entry moves
        with it: a file whose third expression is `#1 + 2` still means its own
        first expression once merged behind five others. The original resolves
        such a reference as it reads, which comes to the same thing; Rederive
        keeps the reference, so it is the reference that has to move.
        """
        return self._read(path, worksheet.text_of(path))

    def load_utility(self, path: Path) -> int:
        """Take the definitions in `path` without showing them.

        A utility file is a library: the point of loading one is that its
        functions and variables are available afterwards, not that two hundred
        definitions land on screen. So nothing is appended, no numbering moves,
        and the session goes on naming the worksheet it already had.

        Returns how many of the file's expressions did not parse.
        """
        source = Source.from_file(worksheet.text_of(path), str(path))
        skipped = 0
        for start, stop in source_lines(source):
            try:
                result = parse_expression(source.text[start:stop].strip(), self.state)
            except DeriveSyntaxError:
                skipped += 1
                continue
            for declaration in result.declarations:
                self.declare(declaration)
            self._define(result.node, result.declarations)
        return skipped

    def load_data(self, path: Path) -> int:
        """Append the matrices in `path`, and say how many blocks would not read.

        A data file appends rather than replaces, as the original's does: the
        numbers are something to work on beside what is already there, not a
        worksheet of their own. A block holding something that is not a number
        is left out and counted, as an unreadable line of a math file is.
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

        A line that does not parse is left out and the rest of the file is
        read, which is what the original does with a file an editor has damaged
        - one bad line does not cost you the other two hundred.
        """
        source = Source.from_file(text, str(path))
        annotations = worksheet.annotations_of(text)
        offset = self._next_number - 1
        skipped = 0
        for start, stop in source_lines(source):
            annotation = annotations.get(source.locate(start)[0], AUTHORED)
            try:
                self._entered(source.text[start:stop].strip(), annotation, offset)
            except DeriveSyntaxError:
                skipped += 1
        self.file = path
        return skipped

    def _entered(self, text: str, annotation: str, offset: int) -> None:
        """Append one line of a file as an entry, its references moved on.

        Redrawn as an authored line is, so that a file written by hand comes up
        as the expressions it holds rather than as the spellings it holds them
        in. A file this program wrote is already in that form and comes back
        unchanged.
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

        What the expressions defined is left standing: clearing a value is its
        own command, and the manual is explicit that these are four commands
        rather than degrees of one. The unremove buffer goes with the history -
        there is nothing left for a restored entry to go back among.
        """
        self.entries = []
        self.removed = []
        self.selected = None
        self.route = ()
        self._next_number = 1

    def clear_variables(self) -> None:
        """Forget every value and domain a line has assigned.

        The expressions that assigned them stay on screen; what goes is what
        they mean to the next command, so a `v` that stood for 7 is a free
        variable again.
        """
        self.assignments.clear()
        self.domains.clear()
        self.state.clear(ClearWhat.VARIABLES)

    def clear_functions(self) -> None:
        """Forget every function a line has defined.

        The name goes from the symbol table as well as the body, so a later
        `F(3)` is a call on a function nothing has defined - stuck, rather than
        nine. The original goes one step further and reads it as `f*3`, its
        Character-mode lexer making a product of a name it has never heard of;
        Rederive reads an undefined call as a call whether or not anything was
        ever cleared, and this command does not change that.
        """
        self.functions.clear()
        self.state.clear(ClearWhat.FUNCTIONS)

    def clear_all(self) -> None:
        """The other three at once, which is what the fourth command is."""
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
        """The subexpression that is selected. What an operation would act on."""
        region = self.selected_region
        return region.node if region is not None else None

    @property
    def highlighted_text(self) -> str | None:
        """What is highlighted, written as it stands in the entry's own text.

        The whole entry where it is selected whole, and just the span of the
        subexpression where the route points inside one. Taking the text rather
        than reprinting the node is what makes this exact: the spans index the
        text the entry was built from, so what comes back is what is on the
        screen, down to how it was spelled. None when nothing is selected.
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

        Rows and columns index that entry's `layout.lines`; the work area
        offsets them by where it drew the entry.
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
        """Step from `route` into the operand it was last left on.

        The first operand where it has not been stepped into before. A node
        with no operands to offer is where this refuses, which is what makes an
        atom the bottom.

        The original keeps one such memory rather than one per entry, so
        stepping into a second expression forgets the first: that one starts
        from the left, and so does the expression left behind when the
        highlight comes back to it. Passing over an expression as a whole is
        not stepping in and leaves the memory alone, which is what lets the
        history be looked down and returned from.
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
        """Record where the selection stands as its parent's operand to return to."""
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
        """Select what entry `index` draws at that cell of its own render.

        Where the mouse points at another entry, that entry is selected whole:
        the pointer has said which expression, and the arrows and the further
        clicks are what go on to say which part of it.

        Inside the entry already selected, the click goes one level towards
        what it points at rather than all the way down to it, the way Right and
        Down step in. So a click on the `x` of `(x + 1)/y` takes the numerator
        and the next click takes the `x`, which is what makes a term of a long
        sum reachable without hitting the one character it is drawn as. A click
        pointing into a different branch takes that branch at the level the two
        part company, and one pointing at something the selection is inside of -
        the `+` between two terms belongs to the sum, not to either term - goes
        back out to it.

        A cell no subexpression covers, the label field included, selects the
        whole expression: the number in front of a render names all of it.
        """
        if not 0 <= index < len(self.entries):
            return False
        entry = self.entries[index]
        if index != self.selected:
            return self.select_entry(index)
        pointed = entry.layout.route_at(row, column)
        if pointed is None:
            return self.select_entry(index)
        # How much of the way there the selection already is, which is what says
        # what one level further means: a step down where the cell is inside the
        # selection, and a step across where the two have parted.
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

        A label the history no longer holds - one that was removed, or one from
        below where the numbering now starts - lands on the entry above it: the
        one with the smallest label of those left over. So `0` always lands on
        the first entry, and a label past the last one names nothing at all,
        which is what the original refuses to jump to.

        It is the label that is looked up and not the position, so a history an
        unremove has left out of numerical order is jumped around the way its
        labels read rather than the way it is stacked.

        Landing on the entry already selected leaves the route alone, so
        jumping to the expression you are inside of keeps the subexpression
        highlighted. Any other entry is selected as a whole.
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

        Only a whole expression counts as being on the history: from inside one
        this goes deeper, and a leaf is where it stops rather than carrying on
        to the expression below.
        """
        if self.selected is None:
            return False
        if self.route:
            return self._descend(self.route)
        return self.select_entry(self.selected + 1)

    def move_right(self) -> bool:
        """Into the expression, or on to the next operand.

        Whole, either horizontal arrow steps in; the difference between them
        only shows once inside, where this is the one that goes forward. The
        last operand is where it stops: there is no wrapping round to the
        first, and no climbing back out.
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

        A whole expression has no operand to the left of it, so the arrow does
        there what its opposite does: it steps in. Inside, the first operand is
        where it stops.
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

        A page is measured in rows and not in entries, since a paneful of
        built-up fractions is fewer expressions than a paneful of one-line
        ones. An expression too tall to share the pane with anything is a page
        of its own, and the highlight moves one entry rather than none.
        """
        if self.selected is None:
            return False
        first, _ = self._pane(rows)
        return self._page_to(first if first < self.selected else self.selected - 1)

    def move_page_down(self, rows: int) -> bool:
        """Select the expression at the bottom of a pane of `rows`.

        Once the highlight is on that expression the pane has to scroll to go
        any further, and what comes up is the paneful below: the farthest entry
        that still fits on screen together with the one selected now.
        """
        if self.selected is None:
            return False
        _, last = self._pane(rows)
        if last > self.selected:
            return self._page_to(last)
        return self._page_to(self._paneful_below(rows))

    def _page_to(self, index: int) -> bool:
        """Select entry `index` whole, unless the page had nowhere to go.

        A page that runs into the end of the history moves nothing at all, and
        that includes the route: an expression the highlight is inside of keeps
        the subexpression highlighted rather than closing up to the whole.
        """
        assert self.selected is not None
        index = max(0, min(index, len(self.entries) - 1))
        if index == self.selected:
            return False
        return self.select_entry(index)

    def _pane(self, rows: int) -> tuple[int, int]:
        """The entries a pane of `rows` holds, the first and the last.

        The pane is where the original puts it: the selected expression at the
        bottom, with as much history above it as fits. What it cannot do is
        scroll past the first expression, and when the top of the history is on
        screen the rest of the pane is filled downward instead - which is what
        leaves the highlight somewhere other than the bottom edge.
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
        """The farthest entry below the selected one that fits on the pane too.

        At least the next entry, whatever its height, so that a page key is
        never a dead end.
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
    """What follows `:=` on a definition line; nothing at all clears it."""
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

    A bound is a number or an infinity and nothing else, which is what makes
    one line enough: a fraction is written `1/2` here and built up only where
    an expression is drawn.
    """
    if node.kind is Kind.UNOP and node.value in ("-", "+"):
        return str(node.value) + _bound_text(node.children[0])
    if node.kind is Kind.NAME and str(node.value) == _INFINITE:
        return INFINITY
    return write_expression(node)


def _rewrite_labels(text: str, node: Node, rename: Rename) -> str:
    """`text` with every label it refers to renamed by `rename`.

    Rewritten from the right, so that a span still indexes `text` when its turn
    comes. Going through the tree rather than over the characters is what keeps
    a `#3` inside a string literal the data it is.
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
    """The same renaming over an annotation, which is text and not a tree."""

    def renamed(found: re.Match[str]) -> str:
        number = rename(int(found.group(1)))
        return found.group(0) if number is None else f"#{number}"

    return LABEL.sub(renamed, annotation)


def _shift_labels(text: str, node: Node, offset: int) -> str:
    """`text` with every label it refers to moved on by `offset`."""
    return _rewrite_labels(text, node, lambda number: number + offset)


def _shift_annotation(annotation: str, offset: int) -> str:
    """The same shift over an annotation, which is text and not a tree."""
    if not offset:
        return annotation
    return _rewrite_annotation(annotation, lambda number: number + offset)


def _same_tree(one: Node, other: Node) -> bool:
    """Whether two trees say the same thing, spans and spelling aside.

    Two nodes compare unequal when they cover different text, which every
    rewritten line does, and how a product's gaps were written is no part of
    what it means: `x y` and `x*y` are one expression. A numeral's surface is
    its digits and does count, since `0FF` and `255` are the same value under
    only one base.
    """
    return _shape(one) == _shape(other)


def _shape(node: Node) -> tuple:
    """`node` as what it means, under the three normalizations writing makes.

    A bare application is a call and `|u|` is `ABS(u)` - the forms the writer
    puts them in, which the display draws the same way either way - and a
    numeral keeps its digits, those being the one surface spelling that says
    something the value does not.
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
    """`node` and everything under it that a line defines something by holding.

    Everywhere `_subtrees` goes except inside a function body, which is a
    recipe and not a line: what it assigns is assigned when the function is
    called.
    """
    yield node
    if node.kind is Kind.FUNDEF:
        return
    for child in node.children:
        yield from _defining(child)
