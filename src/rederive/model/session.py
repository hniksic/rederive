"""The Algebra session: the numbered history and what is selected in it.

Pure Python - no Textual, no key names. The UI layer translates key presses
into calls on `Session` and paints the result. This is the seed of the session
layer that will later sit between the TUI and the math engine.

The session owns the three things an authored line needs: the parse state, so
that `InputMode`, `CaseMode` and every definition reach the lines that follow;
the settings, which an authored `Name := Value` writes to exactly as an Options
dialog does; and the render of each entry, made once when the entry is authored.

A render is not remade. Switching the times operator or the display format
changes how later expressions are drawn and leaves the ones already on screen
alone, which is what the original does: what you see is what it looked like
when it was entered.
"""

from __future__ import annotations

from dataclasses import dataclass

from rederive.display import DisplayOptions, Layout, Region, render
from rederive.model.expr import Node
from rederive.model.settings import Settings
from rederive.syntax import (
    PARSING_SETTINGS,
    Declaration,
    ParseState,
    SettingDeclaration,
    parse_expression,
)

#: A route into an entry's selection tree: indices into `Region.children`,
#: empty for the whole expression.
Route = tuple[int, ...]
Rect = tuple[int, int, int, int]


@dataclass(frozen=True)
class Entry:
    """One numbered line of the history: what was authored, and how it looks."""

    number: int
    text: str
    node: Node
    layout: Layout

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

    def _append(self, text: str, node: Node) -> Entry:
        entry = Entry(self._next_number, text, node, render(node, self.options))
        self._next_number += 1
        self.entries.append(entry)
        self.selected = len(self.entries) - 1
        self.route = ()
        return entry

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
