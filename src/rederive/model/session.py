"""The Algebra session: the numbered history and what is selected in it.

Pure Python - no Textual, no key names. The UI layer translates key presses
into calls on `Session` and paints the result. This is the session layer, and
it is where the math engine is reached from: the UI never calls a command
itself, it asks the session for one.

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
commands share, and any engine command empties it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from rederive import engine
from rederive.display import DisplayOptions, Layout, Region, render
from rederive.model import worksheet
from rederive.model.expr import Kind, Node
from rederive.model.settings import Settings
from rederive.syntax import (
    PARSING_SETTINGS,
    Declaration,
    DeriveSyntaxError,
    ParseState,
    SettingDeclaration,
    Source,
    is_name,
    parse_expression,
    source_lines,
    write_expression,
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

#: One engine command, as the session calls it. Every command takes the same
#: three things and gives back an answer, which is what lets `_command` serve
#: all of them and know nothing about which one it is running.
Command = Callable[[Node, "engine.Context", ParseState], "engine.Result"]


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
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else Settings()
        self.state = ParseState()
        self.entries: list[Entry] = []
        self.selected: int | None = None
        self.route: Route = ()
        #: The file this session was last saved to or read from, if any.
        self.file: Path | None = None
        #: What the last Remove took out, for an Unremove to put back.
        self.removed: list[Entry] = []
        #: What the lines so far have defined, for the engine to substitute.
        self.assignments: dict[str, Node] = {}
        self.functions: dict[str, engine.Definition] = {}
        self.domains: dict[str, engine.Domain] = {}
        self._next_number = 1
        self.settings.watch(self._settings_changed)
        self._settings_changed(PARSING_SETTINGS)

    # -- authoring ---------------------------------------------------------

    def author(self, text: str) -> Entry:
        """Parse `text`, append it as a new entry, and select it as a whole.

        Raises `DeriveSyntaxError` and appends nothing when the line does not
        parse. Nothing is simplified: an accepted expression is inert until the
        user asks for something to be done to it (R-HIST1), and label numbers
        only ever increase (R-HIST2).

        What the line declares is applied before it is drawn, so a hand-written
        `DisplayFormat := Compressed` prints itself compressed, as it does in
        the original.
        """
        result = parse_expression(text, self.state)
        for declaration in result.declarations:
            self.declare(declaration)
        self._define(result.node, result.declarations)
        return self._append(text, result.node)

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

    def _append(self, text: str, node: Node, annotation: str = AUTHORED) -> Entry:
        entry = Entry(
            self._next_number, text, node, render(node, self.options), annotation
        )
        self._next_number += 1
        self.entries.append(entry)
        self.selected = len(self.entries) - 1
        self.route = ()
        return entry

    def _define(self, node: Node, declarations: Iterable[Declaration]) -> None:
        """Record what a line defines: a value, a function body, or a domain.

        The parse state knows which names are taken; what they stand for lives
        in the tree and nowhere else, so this is what carries `x := 5` to the
        next command. A setting is not a variable: `Notation := Mixed` defines
        nothing, and the declarations say which names those are.
        """
        settings = {
            declaration.setting
            for declaration in declarations
            if isinstance(declaration, SettingDeclaration)
        }
        for found in _subtrees(node):
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
            domains=self.domains,
            assignments=self.assignments,
            functions=self.functions,
            labels={entry.number: entry.node for entry in self.entries},
        )

    def simplify(self, request: str) -> Entry:
        """Append the simplified form of the expression `request` names."""
        return self._command(request, "Simp", engine.simplify)

    def factor(
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

        def run(node: Node, context: engine.Context, state: ParseState) -> engine.Result:
            return engine.factor(node, context, amount, variables, state)

        return self._command(request, "Fctr", run)

    def target(self, request: str) -> Node:
        """The expression a command for `request` would act on.

        What the command itself resolves, offered on its own because Factor has
        to look at it before it can ask its questions: which variables it may
        offer, and whether to ask anything at all.

        Raises `DeriveSyntaxError` when `request` does not parse.
        """
        node = parse_expression(request, self.state).node
        entry = self._labelled(node)
        if entry is None:
            return node
        if entry is self.selected_entry and self.route:
            part = self.selected_node
            if part is not None:
                return part
        return entry.node

    def factor_variables(self, request: str) -> tuple[str, ...]:
        """The variables Factor would offer for what `request` names.

        Alphabetical, as the original lists them. Fewer than two is no choice
        at all, and the original asks nothing then.

        Raises `DeriveSyntaxError` when `request` does not parse.
        """
        return engine.factor_variables(self.target(request), self.context)

    def decomposes(self, request: str) -> bool:
        """Whether what `request` names is a number, which Factor just decomposes.

        Raises `DeriveSyntaxError` when `request` does not parse.
        """
        return engine.decomposes(self.target(request))

    def _command(self, request: str, prefix: str, run: Command) -> Entry:
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
            return self._answered(node, run, f"{prefix}({AUTHORED})")
        if entry is self.selected_entry and self.route:
            return self._answered_part(entry, run, prefix)
        return self._answered(entry.node, run, f"{prefix}(#{entry.number})")

    def _labelled(self, node: Node) -> Entry | None:
        """The entry a bare `#3` names, or None when that is not what this is."""
        if node.kind is not Kind.LABEL:
            return None
        try:
            number = int(str(node.value))
        except ValueError:
            return None
        return self.numbered(number)

    def _answered(self, node: Node, run: Command, annotation: str) -> Entry:
        result = run(node, self.context, self.state)
        return self._append(result.text, result.node, annotation)

    def _answered_part(self, entry: Entry, run: Command, prefix: str) -> Entry:
        """`entry` again, with the highlighted part of it transformed.

        The answer is spliced into the entry's own text and the whole line read
        back, so that the new entry's spans index its text exactly as an
        authored line's do. The splice is fenced unless it is a single atom,
        and fences the line already carries are left where they are: a pair too
        many changes nothing about how the line reads or is drawn, since what
        is drawn comes from the tree, where a fence is a matter of precedence
        rather than of text. A pair too few would change what the line says.
        """
        part = self.selected_node
        assert part is not None
        result = run(part, self.context, self.state)
        fragment = result.text if result.node.is_atom else f"({result.text})"
        text = entry.text[: part.start] + fragment + entry.text[part.end :]
        node = parse_expression(text, self.state).node
        return self._append(text, node, f"{prefix}(#{entry.number}')")

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
            str(found.value)
            for found in _subtrees(node)
            if found.kind is Kind.NAME and self._is_variable(str(found.value))
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
        records = [
            worksheet.Record(write_expression(entry.node), self._noted(entry))
            for entry in self.entries
            if (first is None or entry.number >= first)
            and (last is None or entry.number <= last)
        ]
        text = worksheet.write(
            records,
            length=int(self.settings["SaveLength"]),
            annotations=self.settings["SaveAnnotation"] == "Save",
        )
        path.write_text(text, encoding="utf-8")
        self.file = path
        return len(records)

    @staticmethod
    def _noted(entry: Entry) -> str:
        """What to write above `entry`; a line the user wrote needs nothing."""
        return "" if entry.annotation == AUTHORED else entry.annotation

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
        text = _text_of(path)
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
        return self._read(path, _text_of(path))

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
        """Append one line of a file as an entry, its references moved on."""
        result = parse_expression(text, self.state)
        if offset:
            text = _shift_labels(text, result.node, offset)
            result = parse_expression(text, self.state)
        for declaration in result.declarations:
            self.declare(declaration)
        self._define(result.node, result.declarations)
        self._append(text, result.node, _shift_annotation(annotation, offset))

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
        """Next entry, or one level down into the first operand."""
        if self.selected is None:
            return False
        if self.route:
            region = self.selected_region
            if region is None or not region.children:
                return False
            self.route += (0,)
            return True
        return self.select_entry(self.selected + 1)

    def move_right(self) -> bool:
        """First operand of the whole expression, or the next sibling."""
        entry = self.selected_entry
        if entry is None:
            return False
        if not self.route:
            if not entry.layout.root.children:
                return False
            self.route = (0,)
            return True
        if self.route[-1] + 1 >= len(self._siblings()):
            return False
        self.route = self.route[:-1] + (self.route[-1] + 1,)
        return True

    def move_left(self) -> bool:
        """Previous sibling; a whole expression has none."""
        if self.selected_entry is None or not self.route or self.route[-1] == 0:
            return False
        self.route = self.route[:-1] + (self.route[-1] - 1,)
        return True

    def move_first_sibling(self) -> bool:
        if self.selected_entry is None or not self.route:
            return False
        moved = self.route[-1] != 0
        self.route = self.route[:-1] + (0,)
        return moved

    def move_last_sibling(self) -> bool:
        if self.selected_entry is None or not self.route:
            return False
        last = len(self._siblings()) - 1
        moved = self.route[-1] != last
        self.route = self.route[:-1] + (last,)
        return moved

    def move_first_entry(self) -> bool:
        return self.select_entry(0)

    def move_last_entry(self) -> bool:
        return self.select_entry(len(self.entries) - 1)


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


def _text_of(path: Path) -> str:
    """A file's text. Raises, as reading a file does, before anything changes.

    UTF-8 is what Rederive writes. A file the original wrote is code page 437,
    which only shows in one where a glyph left ASCII - a Greek variable name,
    almost always - and that is what the fallback reads. Code page 437 decodes
    any byte at all, so a file is never refused for what is in it.
    """
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp437")


def _shift_labels(text: str, node: Node, offset: int) -> str:
    """`text` with every label it refers to moved on by `offset`.

    Rewritten from the right, so that a span still indexes `text` when its turn
    comes. Going through the tree rather than over the characters is what keeps
    a `#3` inside a string literal the data it is.
    """
    for found in sorted(_subtrees(node), key=lambda found: found.start, reverse=True):
        if found.kind is not Kind.LABEL:
            continue
        try:
            number = int(str(found.value)) + offset
        except ValueError:
            continue
        text = f"{text[: found.start]}#{number}{text[found.end :]}"
    return text


def _shift_annotation(annotation: str, offset: int) -> str:
    """The same shift over an annotation, which is text and not a tree."""
    if not offset:
        return annotation
    return LABEL.sub(lambda found: f"#{int(found.group(1)) + offset}", annotation)


def _subtrees(node: Node) -> Iterator[Node]:
    """`node` and everything under it.

    A definition need not be the whole line: `[x := 1, y := 2]` defines both.
    """
    yield node
    for child in node.children:
        yield from _subtrees(child)
