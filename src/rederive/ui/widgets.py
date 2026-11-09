"""Widgets for the bands of an Algebra pane.

These only render model state, with no navigation rules or parsing. Colors are read
from the app's palette at render time, so `Options Color` takes effect on the next
render.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from rich.cells import cell_len
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.geometry import Offset, Region, Size
from textual.message import Message
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import Input, Static

from rederive import memory
from rederive.model.session import Entry
from rederive.model.settings import (
    ChoiceField,
    DialogEditor,
    Field,
    Item,
    NumberField,
    TextField,
)
from rederive.model.windows import Drawing
from rederive.ui.menu import Menu

#: Width of the label field, i.e. the column where renders start. `#1:` and `#10:` both
#: start the expression at column 5, as in Derive; longer labels push it right.
_LABEL_WIDTH = 5

#: Ctrl-Right and Ctrl-Left scroll by a third of the visible width, as in Derive.
#: Keeping two thirds in place makes it read as a slide rather than a new screen.
_SCROLL_SHARE = 3

#: Columns kept visible on either side of a highlight the pane scrolls to follow. Derive
#: keeps two, which shows the expression continues past the edge.
_SCROLL_MARGIN = 2

#: Blanks between two fields of an Options dialog.
_FIELD_GAP = "  "

#: Blanks between two words of a menu.
_WORD_GAP = 1

#: Minimum height of a command band; Derive reserves two rows even for a short menu.
_BAND_ROWS = 2

#: Rows a band leaves for the rest of the screen when it grows on a narrow terminal: the
#: rule above, the message and status lines below, and two for expressions. If the
#: screen is too short, the band is cut off rather than pushing lines off the bottom.
_BAND_RESERVE = 5

#: Width of a number field's value, so the next field doesn't shift while typing. Derive
#: uses the same width.
_NUMBER_WIDTH = 5

#: Status line column showing the file the session last read or wrote.
_FILE_COLUMN = 22

#: Number of kills the prompt line keeps for Ctrl-Y, as in Bash. Plenty for a line of
#: algebra, and bounded so a long session doesn't keep every deleted scrap.
_KILLS_KEPT = 10

#: Seconds between updates of the memory field, which shows the app plus the engine
#: worker. A runaway computation must show up while there's still time to press Esc.
#: Two `psutil` calls a second are cheap, and it repaints only when the figure changes.
_MEMORY_PERIOD = 0.5


@dataclass(frozen=True)
class _Place:
    """Where a dialog field was drawn, and its clickable choices.

    `start` and `end` are band columns, against which clicks on that row are checked.
    `choices` are the field's values with their columns, empty if it offers none.
    """

    row: int
    start: int
    end: int
    index: int
    choices: tuple[tuple[str, int, int], ...]


def _width(field: Field) -> int:
    """Width of a field's value; text fields specify their own."""
    return field.width if isinstance(field, TextField) else _NUMBER_WIDTH


def _captioned(item: Item) -> bool:
    """Whether a dialog line item prints a caption before its value.

    True for plain words and labeled fields. An unlabeled field starts with the space
    before its value.
    """
    return isinstance(item, str) or bool(item.label)


def _label(entry: Entry) -> str:
    return f"#{entry.number}:"


def _indent(entry: Entry) -> int:
    return max(_LABEL_WIDTH, len(_label(entry)) + 1)


def _label_row(entry: Entry) -> int:
    """The row of the entry's render that holds the label.

    Vertically centered, biased downward for even heights: a four-row fraction has its
    label on the third row.
    """
    return entry.height // 2


def _painted(entry: Entry, offset: int = 0) -> list[str]:
    """The entry's render, each line prefixed with the label field.

    `offset` is the horizontal scroll. The label doesn't scroll, so it stays readable.
    """
    indent = _indent(entry)
    label = _label(entry).ljust(indent)
    blank = " " * indent
    on = _label_row(entry)
    return [
        (label if row == on else blank) + line[offset:]
        for row, line in enumerate(entry.layout.lines)
    ]


def _marks(
    first: int,
    indent: int,
    rect: tuple[int, int, int, int],
    offset: int = 0,
) -> dict[int, tuple[int, int]]:
    """The columns the selection inverts on each row it covers.

    The rectangle includes blanks, so a built-up fraction highlights as a solid block.
    `offset` is the horizontal scroll; parts of the selection scrolled off the left edge
    are not painted.
    """
    top, left, height, width = rect
    marks: dict[int, tuple[int, int]] = {}
    for row in range(top, top + height):
        begin, end = max(indent + left - offset, indent), indent + left + width - offset
        if end > begin:
            marks[first + row] = (begin, end)
    return marks


class _Rows(Widget):
    """Rows of plain text, each drawn only when the screen requests it.

    A worksheet can have thousands of rows while a pane shows a few dozen, and moving
    the highlight should cost only what's visible. So rows are plain strings with at
    most one inverted run each, and the height is simply their count.
    """

    def __init__(self, **arguments: Any) -> None:
        super().__init__(**arguments)
        self._lines: list[str] = []
        self._plain = ""
        self._marked = ""
        self._marks: dict[int, tuple[int, int]] = {}

    def paint(
        self,
        lines: list[str],
        plain: str,
        marked: str = "",
        marks: dict[int, tuple[int, int]] | None = None,
    ) -> None:
        """Show `lines` in style `plain`, with the `marks` runs in `marked`."""
        relaid = len(lines) != len(self._lines)
        self._lines, self._plain, self._marked = lines, plain, marked
        self._marks = marks or {}
        self.refresh(layout=relaid)

    @property
    def content(self) -> Text:
        """The rows as a single text, with marked runs styled."""
        text = Text("\n".join(self._lines), style=self._plain, no_wrap=True)
        start = 0
        for row, line in enumerate(self._lines):
            mark = self._marks.get(row)
            if mark is not None:
                begin, end = mark
                text.stylize(self._marked, start + begin, start + min(end, len(line)))
            start += len(line) + 1
        return text

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        return len(self._lines)

    def render_line(self, y: int) -> Strip:
        base = self.rich_style
        if y >= len(self._lines):
            return Strip.blank(self.size.width, base)
        line = self._lines[y]
        plain = base + Style.parse(self._plain)
        mark = self._marks.get(y)
        if mark is None:
            return Strip([Segment(line, plain)])
        begin, end = mark
        marked = base + Style.parse(self._marked)
        pieces = [(line[:begin], plain), (line[begin:end], marked), (line[end:], plain)]
        return Strip([Segment(piece, style) for piece, style in pieces if piece])


def _packed(
    widths: list[int],
    items: range,
    room: int,
    gap: int,
    tails: list[int] | None = None,
) -> list[list[int]]:
    """Split `items` into rows of `room` columns, with `gap` between items.

    An item that doesn't fit on the current row starts a new one; an item too wide for
    any row is left to be clipped.

    `tails` gives each item's width when it ends a row, which can be smaller because a
    field's padding is unnecessary when nothing follows it.
    """
    tails = widths if tails is None else tails
    rows: list[list[int]] = []
    for index in items:
        taken = sum(widths[at] + gap for at in rows[-1]) if rows else 0
        if not rows or taken + tails[index] > room:
            rows.append([])
        rows[-1].append(index)
    return rows


class Band(Static):
    """A widget that takes its colors from the app's palette.

    The command bands (the menu, and Options dialogs shown in its place) lay out for the
    actual width rather than Derive's eighty columns. A line too long for the terminal
    wraps and the band grows a row, taken from the work area above, so that no command
    becomes unreachable.
    """

    #: The width the band was last laid out for, or 0. A resize updates `size` before
    #: `on_resize` redraws the band, so this tells whether the drawing has caught up.
    drawn_width: int = 0

    @property
    def colors(self) -> dict[str, str]:
        return self.app.palette.styles

    @property
    def width(self) -> int:
        """The band's width, or the screen's before the first layout."""
        return self.size.width or self.app.size.width

    def room(self, prefix: int) -> int:
        """Columns available on a row after a prefix of width `prefix`."""
        return max(1, self.width - prefix)

    def put(self, rows: list[Text]) -> int:
        """Show `rows`, growing the band to fit them.

        Returns how many rows fit, since rows cut off at the bottom can't be clicked.
        """
        limit = max(_BAND_ROWS, self.app.size.height - _BAND_RESERVE)
        rows = rows[:limit]
        for row in rows:
            # Truncate rather than let the widget wrap a row, which would push the last
            # row off the band.
            row.truncate(self.width)
        self.styles.height = max(_BAND_ROWS, len(rows))
        self.update(Text("\n").join(rows))
        self.drawn_width = self.width
        return len(rows)

    def within(self, offset: Offset) -> Offset:
        """Convert screen cell `offset` to a position within this band's contents."""
        return offset - self.content_region.offset


class WorkArea(VerticalScroll):
    """The numbered expression history, scrolled to keep the selection visible.

    Each entry is painted as its rendered lines with a label on the left, separated by
    a blank line, stacking upward from the bottom of the pane.

    Nothing wraps: a render wider than the pane is clipped at the right edge until
    Ctrl-Right scrolls it into view. This preserves the guarantee the selection relies
    on, one rectangle per subexpression.

    As in Derive, horizontal scrolling moves only the selected entry, so the others stay
    aligned with their labels. Ctrl-Right and Ctrl-Left scroll manually, and moving the
    highlight past either edge scrolls along with it.

    Vertical scrolling is Textual's own. The pane does have to map clicks back to
    entries, since only it knows which row each entry was painted on.
    """

    can_focus = False

    def __init__(self, *args, **kwargs) -> None:
        # Built here rather than composed, because a pane is painted as soon as its
        # window is created, before Textual composes it. Identified by class rather
        # than id since there is one per window.
        self.content = _Rows(classes="work-content")
        super().__init__(self.content, *args, **kwargs)
        #: How many columns the selected entry is scrolled left.
        self.shift = 0
        #: The selection last scrolled into view, so that an unchanged highlight isn't
        #: chased and a manually scrolled pane stays put.
        self._followed: object = None
        #: The selected entry's width and label width at the last render, which bound
        #: the shift.
        self._widest = 0
        self._indent = _LABEL_WIDTH
        #: Where the last render put each entry: start row, start column, and height.
        #: Used to map clicks to entries.
        self._drawn: list[tuple[int, int, int]] = []
        #: Which entry was drawn shifted, so clicks on it can account for the shift.
        self._shifted: int | None = None
        #: The selected entry's rows while waiting to be scrolled into view; see
        #: `_reveal`.
        self._revealing: Region | None = None

    def reset(self) -> None:
        """Reset scroll and layout state, for reuse in a new window."""
        self.shift = 0
        self._followed = None
        self._widest = 0
        self._indent = _LABEL_WIDTH
        self._drawn = []
        self._shifted = None
        self._revealing = None

    def show(
        self,
        entries: list[Entry],
        selected: int | None,
        rect: tuple[int, int, int, int] | None,
    ) -> None:
        styles = self.app.palette.styles
        self._follow(entries, selected, rect)
        lines: list[str] = []
        # Each entry's start line and start column.
        starts: list[int] = []
        indents: list[int] = []
        for index, entry in enumerate(entries):
            if index:
                lines.append("")
            starts.append(len(lines))
            indents.append(_indent(entry))
            lines.extend(_painted(entry, self.shift if index == selected else 0))
        self._drawn = [
            (start, indent, entries[index].height)
            for index, (start, indent) in enumerate(zip(starts, indents, strict=True))
        ]
        self._shifted = selected
        marks = None
        if selected is not None and rect is not None:
            marks = _marks(starts[selected], indents[selected], rect, self.shift)
        self.content.paint(lines, styles["work"], styles["selection"], marks)
        self._revealing = None
        if selected is not None:
            self._revealing = Region(0, starts[selected], 1, entries[selected].height)
            self.call_after_refresh(self._reveal)

    def _reveal(self) -> None:
        """Scroll the selected entry into view, once the layout has room for it.

        A post-refresh callback can run before the layout grows the scrollable region,
        and scrolling then would be clamped to the old height, leaving the newest entry
        out of view. So this waits until the virtual height includes the region, and is
        retried when that height changes.

        The scroll is forced because, when this content is the first to overflow, the
        scrollbar doesn't exist yet and Textual won't scroll a pane without one.
        """
        region = self._revealing
        if region is None or region.bottom > self.virtual_size.height:
            return
        self._revealing = None
        self.scroll_to_region(region, animate=False, force=True)

    def watch_virtual_size(self) -> None:
        self._reveal()

    def show_help(self, lines: list[str], rows: int, width: int, titled: bool) -> None:
        """Paint a help page over this window's expressions.

        As in Derive, help is not an overlay: it replaces the expressions in this window
        only, inside the window's border, with no border, scrollbar or shadow of its own.

        The page is padded to the pane's full height so it reads from the top rather
        than sitting at the bottom like the history. A titled page shows its title on
        the first row in inverse video, covering only the title's columns.
        """
        styles = self.app.palette.styles
        self._drawn, self._shifted, self._revealing = [], None, None
        page = [line[:width] for line in lines[:rows]]
        page += [""] * (rows - len(page))
        marks = None
        if titled and page[0].strip():
            marks = {0: (len(page[0]) - len(page[0].lstrip()), len(page[0]))}
        self.content.paint(page, styles["work"], styles["selection"], marks)
        self.call_after_refresh(self.scroll_to, 0, 0, animate=False)

    def show_greeting(
        self, lines: Sequence[str], footer: str, rows: int, width: int
    ) -> None:
        """Paint the opening notice in place of this window's expressions.

        Centered horizontally, with the name block in the upper third and the footer a
        couple of rows from the bottom, spaced like a title page as in Derive. It is
        padded to the pane's full height. If the pane is too short, the end of the block
        is dropped rather than the footer, which names a key.
        """
        styles = self.app.palette.styles
        self._drawn, self._shifted, self._revealing = [], None, None
        centred = [line.center(width).rstrip()[:width] for line in lines]
        above = max(1, (rows - len(centred) - 3) // 3)
        page = ([""] * above + centred)[: max(0, rows - 2)]
        at = max(len(page), rows - 3)
        page += [""] * (at - len(page))
        if len(page) < rows:
            page.append(footer.center(width).rstrip()[:width])
        page += [""] * (rows - len(page))
        self.content.paint(page, styles["work"])

    def pointed(self, offset: Offset) -> tuple[int, int, int] | None:
        """The entry under screen cell `offset`, and the row and column within its render.

        The row and column index the entry's layout lines. A column left of the render,
        in the label field, is negative; it refers to the whole expression.

        Returns None if the cell is on no entry, such as the blank row between entries
        or the space above a short history.
        """
        column, row = offset - self.content.region.offset
        for index, (start, indent, height) in enumerate(self._drawn):
            if start <= row < start + height:
                across = column - indent
                if across >= 0 and index == self._shifted:
                    across += self.shift
                return index, row - start, across
        return None

    @property
    def showable(self) -> int:
        """Columns available for a render, excluding the label field."""
        return max(1, self.size.width - self._indent)

    def _inside(self, shift: int) -> int:
        """`shift` clamped so the pane doesn't scroll past either end of the render."""
        return max(0, min(shift, self._widest - self.showable))

    def _follow(
        self,
        entries: list[Entry],
        selected: int | None,
        rect: tuple[int, int, int, int] | None,
    ) -> None:
        """Update the scroll state and bring a moved highlight into view.

        The render scrolls minimally, and only when the highlight has moved, so a
        manually scrolled pane stays put until the selection leaves it. A couple of
        columns of margin are kept on either side, since a subexpression flush against
        the edge looks cut off.

        If the selection is wider than the pane, its left edge is kept in view.
        """
        if selected is None:
            self.shift, self._widest, self._indent = 0, 0, _LABEL_WIDTH
            return
        entry = entries[selected]
        self._widest, self._indent = entry.layout.width, _indent(entry)
        # Until the pane has a size, leave the selection unfollowed rather than follow
        # it wrongly.
        if rect is None or not self.size.width:
            return
        at = (selected, rect)
        moved, self._followed = at != self._followed, at
        if moved:
            _, left, _, width = rect
            most = left - _SCROLL_MARGIN
            least = left + width + _SCROLL_MARGIN - self.showable
            self.shift = min(max(self.shift, least), most)
        self.shift = self._inside(self.shift)

    def scroll_across(self, direction: int) -> None:
        """Scroll the selected entry sideways by a third of the visible width."""
        self.shift = self._inside(
            self.shift + direction * max(1, self.showable // _SCROLL_SHARE)
        )


class MessageLine(Band):
    """One line saying what is happening or what input is expected."""

    def show(self, message: str) -> None:
        self.update(Text(f" {message}", style=self.colors["prompt"]))


class PromptLine(Input):
    """The input line used by every command that reads a line.

    Textual's input only inserts, so this adds Derive's overwrite mode: with
    `overwrite` set, a typed character replaces the one at the cursor. The key is
    intercepted before the base class would insert it. Typing over a selection replaces
    it in either mode, as the base class already does.

    It also adds an Emacs-style kill ring. Ctrl-U, Ctrl-K, Ctrl-W and Alt-Backspace save
    what they delete, Ctrl-Y yanks the last kill, and Alt-Y right after Ctrl-Y cycles
    back through earlier kills. Consecutive kills join into one, so a line deleted word
    by word comes back whole. The ring belongs to the line, which is a single widget for
    the whole session, so text deleted from one Author line can be yanked into the next
    one or into a Substitute value.
    """

    BINDINGS = [
        Binding("ctrl+y", "yank", "Restore deleted text", show=False),
        Binding("alt+y", "yank_pop", "Restore an earlier deletion", show=False),
    ]

    overwrite = False

    def __init__(self, **arguments: Any) -> None:
        super().__init__(**arguments)
        #: Killed text, oldest first.
        self.kills: list[str] = []
        #: Direction of the kill in progress: -1 for text before the cursor, 1 for text
        #: after it, or None if the current deletion isn't a kill.
        self._killing: int | None = None
        #: The line and cursor after the last kill, to tell whether the next kill
        #: continues it.
        self._killed_at: tuple[str, int] | None = None
        #: The last yank: kill index, start, end, and resulting line. Alt-Y uses these to
        #: check that the yanked text hasn't been edited since.
        self._yanked: tuple[int, int, int, str] | None = None

    class Pasted(Message):
        """Text pasted from the terminal or browser, for the app to insert."""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def _on_paste(self, event: events.Paste) -> None:
        """Forward a paste from the terminal or browser to the app.

        The base class would replace the selection, which on a line offering a label
        leaves the `#` in front of the pasted text. The app handles it like Ctrl-V,
        replacing the `#` too.
        """
        event.prevent_default()
        event.stop()
        self.post_message(self.Pasted(event.text))

    async def _on_key(self, event: events.Key) -> None:
        if not (self.overwrite and event.is_printable and self.selection.is_empty):
            await super()._on_key(event)
            return
        assert event.character is not None
        event.stop()
        event.prevent_default()
        self._restart_blink()
        at = self.cursor_position
        self.replace(event.character, at, at + 1)

    # -- kill and yank -------------------------------------------------------

    def action_delete_left_all(self) -> None:
        """Ctrl-U: kill the text before the cursor."""
        self._kill(-1, super().action_delete_left_all)

    def action_delete_left_word(self) -> None:
        """Ctrl-W: kill the word before the cursor."""
        self._kill(-1, super().action_delete_left_word)

    def action_delete_right_all(self) -> None:
        """Ctrl-K: kill the text from the cursor to the end."""
        self._kill(1, super().action_delete_right_all)

    def action_delete_right_word(self) -> None:
        """Alt-Backspace: kill the word after the cursor."""
        self._kill(1, super().action_delete_right_word)

    def _kill(self, direction: int, delete: Callable[[], None]) -> None:
        """Run `delete`, saving whatever it removes as a kill in `direction`.

        The base class decides how much each action deletes (and some delegate to
        others), so this only sets a flag that `delete` checks.
        """
        was, self._killing = self._killing, direction
        try:
            delete()
        finally:
            self._killing = was

    def delete(self, start: int, end: int) -> None:
        """Delete the text between `start` and `end`, saving it if it's a kill.

        All deletions pass through here, making it the one place to capture killed
        text. Backspace and Delete pass through without saving anything.
        """
        if self._killing is None:
            super().delete(start, end)
            return
        start, end = sorted((max(0, start), min(len(self.value), end)))
        taken = self.value[start:end]
        before = (self.value, self.cursor_position)
        super().delete(start, end)
        if not taken or self.value == before[0]:
            # An empty kill, such as Ctrl-K at the end of the line, doesn't break the
            # run, so kills on either side still join.
            return
        self._keep(taken, joins=before == self._killed_at)
        self._killed_at = (self.value, self.cursor_position)

    def _keep(self, taken: str, joins: bool) -> None:
        """Add `taken` to the ring, or append it to the previous kill if `joins`.

        A kill that starts where the last one ended continues it. Text killed leftwards
        is prepended, and text killed rightwards appended.
        """
        if joins and self.kills:
            last = self.kills[-1]
            self.kills[-1] = last + taken if self._killing == 1 else taken + last
            return
        self.kills.append(taken)
        del self.kills[:-_KILLS_KEPT]

    def action_yank(self) -> None:
        """Ctrl-Y: insert the last kill at the cursor, replacing any selection.

        The cursor ends up after the inserted text.
        """
        if self.kills:
            self._put(len(self.kills) - 1, *sorted(self.selection))

    def action_yank_pop(self) -> None:
        """Alt-Y: replace the text just yanked with the previous kill.

        Works only right after a yank, while the yanked text is unchanged and the cursor
        is at its end. Repeating it cycles back through the ring, wrapping from oldest
        to newest.
        """
        if self._yanked is None:
            return
        index, start, end, line = self._yanked
        if line != self.value or self.cursor_position != end:
            self._yanked = None
            return
        self._put((index - 1) % len(self.kills), start, end)

    def _put(self, index: int, start: int, end: int) -> None:
        """Write kill `index` over the text between `start` and `end`."""
        text = self.kills[index]
        self.replace(text, start, end)
        self._yanked = (index, start, start + len(text), self.value)


class CompletionList(Band):
    """The possible completions of a partly typed filename.

    It opens above the input line with one name per row, the one currently shown on the
    line highlighted in inverse video. Directories keep their trailing separator.

    At most `MAX_ROWS` names are shown at once, scrolled to keep the highlight visible,
    with a count in the border title showing how many are hidden. That makes it usable
    for browsing a directory tree.
    """

    #: Maximum rows taken from the work area. Enough to take in a directory at a glance
    #: while leaving the newest expressions visible on a typical terminal.
    MAX_ROWS = 10

    #: Screen rows needed for everything but the names: the list's two border rows, the
    #: rule, the two-row prompt band, the message and status lines, and one row of work
    #: area. On a short terminal the list shrinks rather than hide the input line.
    ROWS_ELSEWHERE = 8

    def __init__(self, *arguments: object, **named: object) -> None:
        super().__init__(*arguments, **named)
        #: What the list currently shows, kept for redrawing on resize, since the
        #: highlight bar and title depend on the width, unknown until layout.
        self._showing: tuple[list[str], int | None, str] = ([], None, "")
        #: Index of the name on the top row, for mapping clicked rows to names.
        self._first = 0

    def visible_rows(self, count: int) -> int:
        """How many of `count` names fit on screen at once."""
        room = self.app.size.height - self.ROWS_ELSEWHERE
        return max(1, min(count, self.MAX_ROWS, room))

    def rows_for(self, count: int) -> int:
        """The list's height for `count` names, including borders."""
        return self.visible_rows(count) + 2

    def on_resize(self) -> None:
        """Redraw for the new width and height.

        The height is recomputed here too, since a shrinking terminal reduces the rows
        the list may take.
        """
        names, at, where = self._showing
        if names:
            self.styles.height = self.rows_for(len(names))
        self.show(names, at, where)

    def show(self, names: list[str], at: int | None, where: str) -> None:
        """Draw the names, `at` highlighted, under a title naming `where`."""
        self._showing = (names, at, where)
        if not names:
            return
        rows = self.visible_rows(len(names))
        first = self._first = self._first_shown(len(names), at, rows)
        shown = names[first : first + rows]
        text = Text(style=self.colors["work"])
        for row, name in enumerate(shown):
            highlight = at is not None and first + row == at
            style = self.colors["selection"] if highlight else self.colors["work"]
            # Pad rows to the full width so the highlight is a bar across the list.
            width = max(self.size.width - 2, len(name) + 1)
            text.append(f" {name}".ljust(width) + "\n", style=style)
        self.border_title = self._title(where, len(names), first, len(shown))
        self.update(text)

    def name_at(self, offset: Offset) -> int | None:
        """The index of the name at screen cell `offset`, or None.

        Each name spans the full row, so only the row matters. Returns None on the
        border or past the last name.
        """
        names, _, _ = self._showing
        if not names:
            return None
        point = self.within(offset)
        if not 0 <= point.x < self.content_region.width:
            return None
        at = self._first + point.y
        return at if 0 <= point.y < self.visible_rows(len(names)) else None

    def _first_shown(self, count: int, at: int | None, rows: int) -> int:
        """The index of the first visible name, chosen to keep `at` visible."""
        if at is None or count <= rows:
            return 0
        # Center the highlighted name when possible.
        return max(0, min(at - rows // 2, count - rows))

    def _title(self, where: str, count: int, first: int, shown: int) -> str:
        """The border title: the directory, and how many names are showing.

        The count is always shown in full, and the directory is truncated from the left
        to fit, since the end of a path is the informative part.
        """
        seen = f"{first + 1}-{first + shown}"
        tally = f"{count}" if shown >= count else f"{seen} of {count}"
        # Leave room for the corners, padding and the hyphen between the two parts.
        room = self.size.width - len(tally) - 10
        if room > 0 and len(where) > room:
            where = "..." + where[-(room - 3) :]
        return f" {where} - {tally} "


class MenuBand(Band):
    """A menu of words, one of them highlighted in inverse video.

    Word positions are recorded while drawing, so clicks can be mapped to words. The
    title isn't clickable.

    The word under the mouse pointer is underlined, to show that the keyboard-driven
    menu is also clickable.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #: The menu shown, kept for relayout on resize.
        self._showing: tuple[Menu, int | None] | None = None
        #: Each word's row, start column, end column, and index in the menu.
        self._words: list[tuple[int, int, int, int]] = []
        #: The screen cell under the pointer, or None. Stored as a cell rather than a
        #: word, so that after a redraw (a submenu opening, a resize) the word now at
        #: that cell is marked.
        self._pointer: Offset | None = None

    def say(self, text: str) -> None:
        """Show one line of plain text instead of a menu.

        Used by demo scripts to show the comment for the expression just run.
        """
        self._showing = None
        self._words = []
        self.put([Text(f" {text}", style=self.colors["option"], no_wrap=True)])

    def show(self, menu: Menu, highlighted: int | None) -> None:
        """Render `menu`. `None` highlights nothing, as while Quit asks to confirm."""
        self._showing = (menu, highlighted)
        prefix = f" {menu.title} "
        indent = " " * len(prefix)
        rows: list[Text] = []
        words: list[tuple[int, int, int, int]] = []
        for group in self._rows(menu, len(prefix)):
            row = len(rows)
            line, placed = self._line(
                menu, prefix if row == 0 else indent, group, highlighted
            )
            words += [(row, start, end, index) for start, end, index in placed]
            rows.append(line)
        self._mark(rows, words)
        shown = self.put(rows)
        self._words = [word for word in words if word[0] < shown]

    def point_at(self, offset: Offset | None) -> None:
        """Record the pointer at screen cell `offset`, or nowhere if None.

        Redraws only when the word under the pointer changes.
        """
        moved = self.word_at(offset) != self.word_at(self._pointer)
        self._pointer = offset
        if moved and self._showing is not None:
            self.show(*self._showing)

    def word_at(self, offset: Offset | None) -> int | None:
        """The index of the menu word at screen cell `offset`, or None."""
        if offset is None:
            return None
        point = self.within(offset)
        for row, start, end, index in self._words:
            if row == point.y and start <= point.x < end:
                return index
        return None

    def on_resize(self) -> None:
        """Relayout the menu for the new terminal width."""
        if self._showing is not None:
            self.show(*self._showing)

    def _mark(self, rows: list[Text], words: list[tuple[int, int, int, int]]) -> None:
        """Underline the word under the pointer, if any.

        Uses the new layout rather than the previous one, so a word that moved under a
        stationary pointer is the one marked.
        """
        if self._pointer is None:
            return
        point = self.within(self._pointer)
        for row, start, end, _ in words:
            if row == point.y and start <= point.x < end:
                rows[row].stylize(self.colors["option-pointed"], start, end)
                return

    def _rows(self, menu: Menu, prefix: int) -> list[list[int]]:
        """The words on each row, as indices into the menu.

        Every row has a prefix of the same width (the title, or an indent under it).

        The menu's own line break is kept while both lines fit, so eighty columns look
        like Derive. Otherwise all words are packed across rows, which takes fewer rows
        than wrapping each line separately.
        """
        room, widths = self.room(prefix), [len(word) for word in menu.words]
        break_at = min(menu.first_line, len(menu.words))
        named = [
            _packed(widths, range(0, break_at), room, _WORD_GAP),
            _packed(widths, range(break_at, len(widths)), room, _WORD_GAP),
        ]
        if all(len(rows) <= 1 for rows in named):
            return [row for rows in named for row in rows]
        return _packed(widths, range(len(widths)), room, _WORD_GAP)

    def _line(
        self,
        menu: Menu,
        prefix: str,
        indices: list[int],
        highlighted: int | None,
    ) -> tuple[Text, list[tuple[int, int, int]]]:
        """One row of the menu, and the columns of each word on it."""
        colors = self.colors
        text = Text(prefix, style=colors["option"], no_wrap=True)
        placed = []
        for position, index in enumerate(indices):
            if position:
                text.append(" " * _WORD_GAP)
            highlight = index == highlighted
            style = colors["option-highlight"] if highlight else colors["option"]
            at = len(text.plain)
            text.append(menu.words[index], style=style)
            placed.append((at, len(text.plain), index))
        return text, placed


class FieldBand(Band):
    """An Options dialog: its title and fields, with the active one editable.

    As in Derive, inactive fields show their value in parentheses, distinguishing the
    current setting from the one being edited.

    A line may also contain a plain word that isn't a field, such as the variable name
    between the two bounds in `Declare Variable Interval`.

    Positions of fields and their choices are recorded while drawing, so clicks can be
    mapped to a field or value. The layout depends on the width, so they can't be
    computed afterwards.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #: The dialog shown, kept for relayout on resize.
        self._showing: DialogEditor | None = None
        #: Where each field and its choices were drawn.
        self._places: list[_Place] = []

    def show(self, editor: DialogEditor) -> None:
        self._showing = editor
        title = editor.dialog.title
        indent = " " * (len(title) + 2)
        rows: list[Text] = []
        places: list[_Place] = []
        index = 0
        for number, items in enumerate(editor.lines):
            prefix = f" {title} " if number == 0 else indent
            if not _captioned(items[0]):
                # An uncaptioned field starts with a space, so drop the title's trailing
                # space: `MANAGE BRANCH: Principal Real Any`. The value still starts at
                # the column the next line indents to.
                prefix = prefix[:-1]
            drawn: list[Text] = []
            # The field index of each item and its choices; None and () for plain words.
            owners: list[int | None] = []
            choices: list[tuple[tuple[str, int, int], ...]] = []
            for item in items:
                if isinstance(item, str):
                    drawn.append(Text(item, style=self.colors["option"], no_wrap=True))
                    owners.append(None)
                    choices.append(())
                    continue
                text, offered = self._field(editor, item, active=index == editor.active)
                drawn.append(text)
                owners.append(index)
                choices.append(offered)
                index += 1
            # A line too wide for the terminal wraps between fields and continues under
            # the title, like a menu. Fields themselves are never split.
            widths = [item.cell_len for item in drawn]
            tails = [cell_len(item.plain.rstrip()) for item in drawn]
            room = self.room(len(indent))
            for row, group in enumerate(
                _packed(widths, range(len(drawn)), room, len(_FIELD_GAP), tails)
            ):
                line = Text(
                    prefix if row == 0 else indent,
                    style=self.colors["option"],
                    no_wrap=True,
                )
                for position, at in enumerate(group):
                    if position:
                        line.append(_FIELD_GAP)
                    column = len(line.plain)
                    line.append(drawn[at])
                    owner = owners[at]
                    if owner is not None:
                        places.append(
                            _Place(
                                len(rows),
                                column,
                                len(line.plain),
                                owner,
                                tuple(
                                    (value, column + start, column + end)
                                    for value, start, end in choices[at]
                                ),
                            )
                        )
                rows.append(line)
        shown = self.put(rows)
        self._places = [place for place in places if place.row < shown]

    def field_at(self, offset: Offset) -> tuple[int, str | None] | None:
        """The field and choice at screen cell `offset`.

        The choice is None if the cell is on the field but not on a value, such as its
        label or a number field's padding. Returns None if the cell is on no field, such
        as a plain word or the title.
        """
        point = self.within(offset)
        for place in self._places:
            if place.row != point.y or not place.start <= point.x < place.end:
                continue
            for value, start, end in place.choices:
                if start <= point.x < end:
                    return place.index, value
            return place.index, None
        return None

    def on_resize(self) -> None:
        """Relayout the dialog for the new terminal width."""
        if self._showing is not None:
            self.show(self._showing)

    def _field(
        self, editor: DialogEditor, field: Field, active: bool
    ) -> tuple[Text, tuple[tuple[str, int, int], ...]]:
        """A field as drawn, and the columns of its choices.

        The columns are relative to the field; the caller offsets them. Typed fields,
        and fields with too many choices to print, return no choices.
        """
        colors = self.colors
        # A field without a label prints only its value, as with interval strictness.
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
            return text, self._choices(text, field, current)
        else:
            return text, self._parenthesized(text, field, current)
        return text, ()

    def _typed(
        self,
        text: Text,
        editor: DialogEditor,
        current: str | int,
        active: bool,
        width: int,
    ) -> None:
        """Append a typed field's value, with the cursor if it is active."""
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
            # At the end of the text, the cursor occupies its own cell.
            text.append(" ", style=cursor)
            written += 1
        text.append(" " * max(0, width - written))

    def _choices(
        self, text: Text, field: ChoiceField, current: str | int
    ) -> tuple[tuple[str, int, int], ...]:
        colors = self.colors
        offered = []
        for choice in field.choices:
            text.append(" ")
            style = colors["option-highlight"] if choice == current else colors["option"]
            at = len(text.plain)
            text.append(choice, style=style)
            offered.append((choice, at, len(text.plain)))
        return tuple(offered)

    @staticmethod
    def _parenthesized(
        text: Text, field: ChoiceField, current: str | int
    ) -> tuple[tuple[str, int, int], ...]:
        """`Mode:(Character)Word`, where the parentheses replace the adjacent spaces.

        Each choice includes the blank before it, so a click between two choices lands
        on one of them.
        """
        offered = []
        for index, choice in enumerate(field.choices):
            at = len(text.plain)
            if choice == current:
                text.append(f"({choice})")
            elif index and field.choices[index - 1] == current:
                text.append(choice)
            else:
                text.append(f" {choice}")
            offered.append((choice, at, len(text.plain)))
        return tuple(offered)


class Panes(Container):
    """The region holding the windows, with their borders and contents.

    The work areas, one per window, are positioned absolutely over it, and the frame
    beneath draws the borders. Both are recomputed on resize, since divider positions
    depend on the size.
    """

    def drawing(self) -> Drawing:
        """The current border drawing, one row taller than this widget."""
        return self.app.windows.frame(self.size.height, self.size.width)

    def on_resize(self) -> None:
        self.app.place_windows()


class Frame(Band):
    """The borders around and between the windows.

    Blank with a single window, as in Derive, where an unsplit screen has no frame. Each
    window's number is drawn in its upper left corner, and the active window's number
    is in inverse video, the only indication of which window commands act on.
    """

    def render(self) -> Text:
        drawing = self.screen.query_one(Panes).drawing()
        colors = self.colors
        text = Text("\n".join(drawing.rows[:-1]), style=colors["frame"], no_wrap=True)
        active = self.app.windows.number - 1
        if active < len(drawing.numbers):
            row, column, width = drawing.numbers[active]
            at = sum(len(line) + 1 for line in drawing.rows[:row]) + column
            text.stylize(colors["selection"], at, at + width)
        return text


class MenuRule(Band):
    """The rule between the work area and the command bands.

    It is the bottom edge of the window frame, so it includes the corners and the ends
    of vertical dividers once the screen is split.
    """

    def render(self) -> Text:
        rule = self.screen.query_one(Panes).drawing().rows[-1]
        return Text(rule[: self.size.width], style=self.colors["frame"], no_wrap=True)


class StatusLine(Band):
    """Bottom line: selection annotation, current file, memory field, pane type.

    Everything on it refers to the active window: the annotation of its highlighted
    expression and its pane type. As in Derive, there is one status line regardless of
    the number of windows.
    """

    annotation = ""
    file = ""
    #: Memory used by the process, in place of Derive's muLISP free-heap gauge. Empty on
    #: platforms that don't report it.
    center = ""
    #: Mode indicators next to the memory field: `Ovr` in overwrite mode, `Lin` when the
    #: arrow keys move within the input line rather than the highlight. Each appears only
    #: in the non-default state, and neither when no line is being edited.
    flags = ""
    pane = "Rederive Algebra"

    def on_mount(self) -> None:
        self._read_memory()
        self.set_interval(_MEMORY_PERIOD, self._read_memory)

    def _read_memory(self) -> None:
        """Update the memory field, repainting only when it changes."""
        center = memory.label()
        if center != self.center:
            self.center = center
            self.refresh()

    def show(
        self, annotation: str, file: str = "", flags: str = "", pane: str = ""
    ) -> None:
        self.annotation = annotation
        self.file = file
        self.flags = flags
        if pane:
            self.pane = pane
        self.refresh()

    def indicate(self, flags: str) -> None:
        """Update only the mode indicators.

        A full refresh would need to know which band the interrupted command had up.
        """
        if flags != self.flags:
            self.flags = flags
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
        if self.flags:
            text.append(" ")
            text.append(self.flags, style=style)
            used += 1 + len(self.flags)
        pane_at = max(used + 1, width - 1 - len(self.pane))
        text.append(" " * (pane_at - used))
        text.append(self.pane, style=style)
        return text
