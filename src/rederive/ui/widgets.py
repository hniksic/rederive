"""Widgets for the bands of an Algebra pane.

These render model state and nothing else: no navigation rules, no parsing.
Colors are read from the app's palette at render time, so an `Options Color`
command repaints simply by asking for another render.
"""

from __future__ import annotations

from rich.text import Text
from textual.containers import VerticalScroll
from textual.geometry import Region
from textual.widgets import Static

from rederive.model.session import Entry
from rederive.model.settings import (
    ChoiceField,
    DialogEditor,
    Field,
    Item,
    NumberField,
    TextField,
)
from rederive.ui.menu import Menu

#: Columns the label field takes, and so the column every render starts in.
#: `#1:` and `#10:` both leave the expression at column 5, as they do in the
#: original; a number too long for the field pushes it right.
_LABEL_WIDTH = 5

#: Blanks between two fields of an Options dialog.
_FIELD_GAP = "  "

#: Columns a number field's value takes, so that the field beside it does not
#: shift as digits are typed. The original reserves the same width.
_NUMBER_WIDTH = 5

#: Where the status line names the file the session last read or wrote.
_FILE_COLUMN = 22


def _width(field: Field) -> int:
    """Columns a field's value takes; a text field says for itself."""
    return field.width if isinstance(field, TextField) else _NUMBER_WIDTH


def _captioned(item: Item) -> bool:
    """Whether a dialog line's item prints something of its own in front.

    A plain word does, and so does any field with a label. A field without one
    starts with the space that goes before its value.
    """
    return isinstance(item, str) or bool(item.label)


def _label(entry: Entry) -> str:
    return f"#{entry.number}:"


def _indent(entry: Entry) -> int:
    return max(_LABEL_WIDTH, len(_label(entry)) + 1)


def _label_row(entry: Entry) -> int:
    """The row of the entry's render the label sits on.

    Vertically centred, biased downward when the height is even: a four-row
    fraction carries its label on the third row.
    """
    return entry.height // 2


def _painted(entry: Entry) -> list[str]:
    """The entry's render, each line prefixed with the label field."""
    indent = _indent(entry)
    label = _label(entry).ljust(indent)
    blank = " " * indent
    on = _label_row(entry)
    return [
        (label if row == on else blank) + line
        for row, line in enumerate(entry.layout.lines)
    ]


def _offsets(lines: list[str]) -> list[int]:
    """Where each line begins once the lines are joined with newlines."""
    offsets: list[int] = []
    position = 0
    for line in lines:
        offsets.append(position)
        position += len(line) + 1
    return offsets


def _invert(
    text: Text,
    offsets: list[int],
    first: int,
    indent: int,
    rect: tuple[int, int, int, int],
    style: str,
) -> None:
    """Highlight the selection, one row of its rectangle at a time.

    The rectangle covers the blanks inside it, so the highlight over a built-up
    fraction is a solid block rather than a ragged outline.
    """
    top, left, height, width = rect
    for row in range(top, top + height):
        begin = offsets[first + row] + indent + left
        text.stylize(style, begin, begin + width)


class Band(Static):
    """A widget that colors itself from the palette the app is showing."""

    @property
    def colors(self) -> dict[str, str]:
        return self.app.palette.styles


class WorkArea(VerticalScroll):
    """The numbered expression history, scrolled to keep the selection visible.

    Each entry is painted as the lines of its own render, labelled down the
    left edge and separated by one blank line, and the entries stack upward
    from the bottom of the pane.

    Nothing here wraps or reflows: a render is as wide as it is, and one wider
    than the pane is clipped at the right edge until Ctrl-Right scrolls it into
    view. That is what keeps the guarantee the selection depends on - one
    subexpression, one rectangle.
    """

    can_focus = False

    def compose(self):
        yield Static(id="work-content")

    def show(
        self,
        entries: list[Entry],
        selected: int | None,
        rect: tuple[int, int, int, int] | None,
    ) -> None:
        styles = self.app.palette.styles
        lines: list[str] = []
        # For each entry, the line it starts on and the column its render does.
        starts: list[int] = []
        indents: list[int] = []
        for index, entry in enumerate(entries):
            if index:
                lines.append("")
            starts.append(len(lines))
            indents.append(_indent(entry))
            lines.extend(_painted(entry))
        text = Text("\n".join(lines), style=styles["work"], no_wrap=True)
        if selected is not None and rect is not None:
            _invert(
                text,
                _offsets(lines),
                starts[selected],
                indents[selected],
                rect,
                styles["selection"],
            )
        self.query_one("#work-content", Static).update(text)
        if selected is not None:
            # After the refresh, so the new content has been laid out and the
            # scrollable region knows how tall it is.
            region = Region(0, starts[selected], 1, entries[selected].height)
            self.call_after_refresh(self.scroll_to_region, region, animate=False)

    def scroll_half_pane(self, direction: int) -> None:
        """Scroll sideways, which moves no selection and reveals no new entry."""
        half = max(1, self.size.width // 2)
        self.scroll_relative(x=direction * half, animate=False)


class MessageLine(Band):
    """One line saying what is happening or what input is expected."""

    def show(self, message: str) -> None:
        self.update(Text(f" {message}", style=self.colors["prompt"]))


class MenuBand(Band):
    """A menu of words, one of them highlighted in inverse video."""

    def say(self, text: str) -> None:
        """Put one line of plain text on the band in place of a menu.

        What a demonstration's script is shown on: the comment above the
        expression it has just run stands where the menu words go.
        """
        self.update(Text(f" {text}", style=self.colors["option"], no_wrap=True))

    def show(self, menu: Menu, highlighted: int | None) -> None:
        """Render `menu`; `None` highlights nothing, as while Quit asks."""
        break_at = min(menu.first_line, len(menu.words))
        first = self._line(menu, f" {menu.title} ", 0, break_at, highlighted)
        if break_at == len(menu.words):
            self.update(first)
            return
        indent = " " * (len(menu.title) + 2)
        rest = self._line(menu, indent, break_at, len(menu.words), highlighted)
        self.update(Text("\n").join([first, rest]))

    def _line(
        self,
        menu: Menu,
        prefix: str,
        start: int,
        stop: int,
        highlighted: int | None,
    ) -> Text:
        colors = self.colors
        text = Text(prefix, style=colors["option"], no_wrap=True)
        for index in range(start, stop):
            if index > start:
                text.append(" ")
            highlight = index == highlighted
            style = colors["option-highlight"] if highlight else colors["option"]
            text.append(menu.words[index], style=style)
        return text


class FieldBand(Band):
    """An Options dialog: its title and its fields, the active one live.

    A field that is not the active one shows its value in parentheses, the way
    the original distinguishes "this is what it is set to" from "this is what
    you are setting".

    A line may also carry a plain word, which is printed where it stands and is
    no field: the variable whose bounds `Declare Variable Interval` is asking
    for stands between the two bounds.
    """

    def show(self, editor: DialogEditor) -> None:
        title = editor.dialog.title
        lines, index = [], 0
        for number, items in enumerate(editor.lines):
            prefix = f" {title} " if number == 0 else " " * (len(title) + 2)
            if not _captioned(items[0]):
                # A field with no caption prints a space in front of its
                # value, and that space is the one the title would print, so
                # the title goes without: `MANAGE BRANCH: Principal Real Any`.
                # The value still begins in the column the next line indents
                # to, which is what keeps the two lines square.
                prefix = prefix[:-1]
            line = Text(prefix, style=self.colors["option"], no_wrap=True)
            for position, item in enumerate(items):
                if position:
                    line.append(_FIELD_GAP)
                if isinstance(item, str):
                    line.append(item, style=self.colors["option"])
                    continue
                line.append(self._field(editor, item, active=index == editor.active))
                index += 1
            lines.append(line)
        self.update(Text("\n").join(lines))

    def _field(self, editor: DialogEditor, field: Field, active: bool) -> Text:
        colors = self.colors
        # A field with no label of its own is printed as its value alone, which
        # is how the strictness of an interval bound is shown.
        label = f"{field.label}:" if field.label else ""
        text = Text(label, style=colors["option"], no_wrap=True)
        current = editor.value(field)
        if isinstance(field, NumberField | TextField):
            self._typed(text, editor, current, active, _width(field))
        elif not field.inline:
            # Too many choices to print, so only the current one is shown.
            style = colors["option-highlight"] if active else colors["option"]
            text.append(" ")
            text.append(str(current), style=style)
        elif active:
            self._choices(text, field, current)
        else:
            self._parenthesized(text, field, current)
        return text

    def _typed(
        self,
        text: Text,
        editor: DialogEditor,
        current: str | int,
        active: bool,
        width: int,
    ) -> None:
        """A field that is typed into: its value, and the cursor when it is live."""
        if not active:
            text.append(f" {str(current).ljust(width)}")
            return
        cursor = self.colors["option-highlight"]
        typed = editor.text or ""
        text.append(" ")
        for position, character in enumerate(typed):
            text.append(character, style=cursor if position == editor.cursor else None)
        written = len(typed)
        if editor.cursor >= len(typed):
            # Past the last character, the cursor is a cell of its own.
            text.append(" ", style=cursor)
            written += 1
        text.append(" " * max(0, width - written))

    def _choices(self, text: Text, field: ChoiceField, current: str | int) -> None:
        colors = self.colors
        for choice in field.choices:
            text.append(" ")
            style = colors["option-highlight"] if choice == current else colors["option"]
            text.append(choice, style=style)

    @staticmethod
    def _parenthesized(text: Text, field: ChoiceField, current: str | int) -> None:
        """`Mode:(Character)Word` - the parentheses eat the spaces beside them."""
        for index, choice in enumerate(field.choices):
            if choice == current:
                text.append(f"({choice})")
            elif index and field.choices[index - 1] == current:
                text.append(choice)
            else:
                text.append(f" {choice}")


class MenuRule(Static):
    """The horizontal rule separating the work area from the command bands."""

    def render(self) -> Text:
        return Text("─" * self.size.width, no_wrap=True)


class StatusLine(Band):
    """Bottom line: selection annotation, current file, memory field, pane type."""

    annotation = ""
    file = ""
    #: The original's muLISP heap gauge lived here. The slot is kept for
    #: something useful today, such as a busy indicator during long
    #: computations, and stays empty until there is one.
    center = ""
    pane = "Rederive Algebra"

    def show(self, annotation: str, file: str = "") -> None:
        self.annotation = annotation
        self.file = file
        self.refresh()

    def render(self) -> Text:
        style = self.colors["status"]
        width = max(self.size.width, len(self.pane) + 2)
        text = Text(" ", no_wrap=True)
        text.append(self.annotation, style=style)
        used = 1 + len(self.annotation)
        if self.file:
            file_at = max(used + 1, _FILE_COLUMN)
            text.append(" " * (file_at - used))
            text.append(self.file, style=style)
            used = file_at + len(self.file)
        center_at = max(used + 1, (width - len(self.center)) // 2)
        text.append(" " * (center_at - used))
        text.append(self.center, style=style)
        used = center_at + len(self.center)
        pane_at = max(used + 1, width - 1 - len(self.pane))
        text.append(" " * (pane_at - used))
        text.append(self.pane, style=style)
        return text
