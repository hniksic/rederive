"""Widgets for the bands of an Algebra pane.

These render model state and nothing else: no navigation rules, no parsing.
Colors are read from the app's palette at render time, so an `Options Color`
command repaints simply by asking for another render.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from rich.cells import cell_len
from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.geometry import Offset, Region
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

#: Columns the label field takes, and so the column every render starts in.
#: `#1:` and `#10:` both leave the expression at column 5, as they do in the
#: original; a number too long for the field pushes it right.
_LABEL_WIDTH = 5

#: Share of the showable width one Ctrl-Right or Ctrl-Left takes, as the
#: original steps it: a third at a time leaves two thirds of the render
#: standing still, which is what makes it readable as a slide rather than as a
#: new screen.
_SCROLL_SHARE = 3

#: Columns of the render kept on show to either side of a highlight the pane
#: has scrolled to follow. The original keeps two, which is what says that the
#: expression goes on past the edge rather than ending there.
_SCROLL_MARGIN = 2

#: Blanks between two fields of an Options dialog.
_FIELD_GAP = "  "

#: Blanks between two words of a menu.
_WORD_GAP = 1

#: Rows a command band takes when what is on it fits in fewer, which is the
#: two the original reserves for a menu however short it is.
_BAND_ROWS = 2

#: Rows a band leaves the rest of the screen when it grows to fit a narrow
#: terminal: the rule above it, the message and status lines below it, and two
#: for the expressions. A screen too short to hold the whole menu as well is
#: one the band gives way on, rather than one it pushes the lines below it off
#: the bottom of.
_BAND_RESERVE = 5

#: Columns a number field's value takes, so that the field beside it does not
#: shift as digits are typed. The original reserves the same width.
_NUMBER_WIDTH = 5

#: Where the status line names the file the session last read or wrote.
_FILE_COLUMN = 22

#: Kills the prompt line keeps for Ctrl-Y to put back. Bash keeps ten and
#: Emacs keeps far more; ten is more than a line of algebra is ever taken
#: apart into, and the oldest going is what keeps a long session from holding
#: on to every scrap of text it was ever asked to delete.
_KILLS_KEPT = 10

#: Seconds between readings of the memory field. The figure is the app plus
#: the engine worker, and the worker is where it moves: a computation that is
#: eating the machine has to show it while there is still time to press Esc,
#: which is a gauge that keeps up rather than one that catches up. Two readings
#: a second cost two `psutil` calls and repaint only when the figure moves.
_MEMORY_PERIOD = 0.5


@dataclass(frozen=True)
class _Place:
    """Where one field of a dialog was drawn, and what can be picked off it.

    The columns are the band's own, so `start` and `end` are what a click on
    that row is measured against. `choices` are the values the field offers,
    each with the columns it took; a field that offers none carries an empty
    tuple.
    """

    row: int
    start: int
    end: int
    index: int
    choices: tuple[tuple[str, int, int], ...]


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


def _painted(entry: Entry, offset: int = 0) -> list[str]:
    """The entry's render, each line prefixed with the label field.

    `offset` is how far the render has been scrolled sideways. The label field
    does not move with it: the number stays where it can be read whatever part
    of a wide expression is on show.
    """
    indent = _indent(entry)
    label = _label(entry).ljust(indent)
    blank = " " * indent
    on = _label_row(entry)
    return [
        (label if row == on else blank) + line[offset:]
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
    offset: int = 0,
) -> None:
    """Highlight the selection, one row of its rectangle at a time.

    The rectangle covers the blanks inside it, so the highlight over a built-up
    fraction is a solid block rather than a ragged outline. `offset` is how far
    the render has been scrolled sideways; the part of the selection that has
    gone off the left edge is not painted, and one scrolled away entirely is
    not painted at all.
    """
    top, left, height, width = rect
    for row in range(top, top + height):
        line = offsets[first + row] + indent
        begin, end = line + left - offset, line + left + width - offset
        if end > max(begin, line):
            text.stylize(style, max(begin, line), end)


def _packed(
    widths: list[int],
    items: range,
    room: int,
    gap: int,
    tails: list[int] | None = None,
) -> list[list[int]]:
    """Which of `items` go on each row of `room` columns, `gap` between them.

    An item too wide to share a row with what is already there opens the next
    one; one too wide for a row of its own is left to be clipped, there being
    nowhere else to put it.

    `tails` is how wide each item is where it ends a row, which is where the
    blanks a field is padded with cost nothing: they are what the field beside
    it stands clear of, and at the end of a row nothing stands beside it.
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
    """A widget that colors itself from the palette the app is showing.

    The two bands that carry commands - the menu and the Options dialog that
    stands where it does - lay themselves out for the width there is rather
    than for the eighty columns the original had. A line too long for the
    terminal is broken again and the band grows a row, which it takes from the
    work area above: the alternative is the row falling off the band, and a
    command nobody can see is a command nobody can reach.
    """

    @property
    def colors(self) -> dict[str, str]:
        return self.app.palette.styles

    @property
    def width(self) -> int:
        """Columns the band has. Before the first layout, the screen's."""
        return self.size.width or self.app.size.width

    def room(self, prefix: int) -> int:
        """Columns a row has for its contents, past a prefix that wide."""
        return max(1, self.width - prefix)

    def put(self, rows: list[Text]) -> int:
        """Show `rows`, the band standing as tall as there are rows to show.

        How many of them it had room for comes back, since a row dropped off
        the bottom carries nothing a click can land on.
        """
        limit = max(_BAND_ROWS, self.app.size.height - _BAND_RESERVE)
        rows = rows[:limit]
        for row in rows:
            # A row wider than the band is wrapped by the widget, and a
            # wrapped row takes two of the rows the band has - which pushes
            # the last of them off it. Cutting at the edge is what makes the
            # band show the rows it was laid out with and no others.
            row.truncate(self.width)
        self.styles.height = max(_BAND_ROWS, len(rows))
        self.update(Text("\n").join(rows))
        return len(rows)

    def within(self, offset: Offset) -> Offset:
        """The screen cell `offset` as a row and column of this band's contents."""
        return offset - self.content_region.offset


class WorkArea(VerticalScroll):
    """The numbered expression history, scrolled to keep the selection visible.

    Each entry is painted as the lines of its own render, labelled down the
    left edge and separated by one blank line, and the entries stack upward
    from the bottom of the pane.

    Nothing here wraps or reflows: a render is as wide as it is, and one wider
    than the pane is clipped at the right edge until Ctrl-Right scrolls it into
    view. That is what keeps the guarantee the selection depends on - one
    subexpression, one rectangle.

    Scrolling sideways moves the selected entry alone, as the original's does.
    The rest of the history is a column of numbered expressions to read down,
    and shifting all of it to see the end of one line would take the others out
    from under their own numbers. Ctrl-Right and Ctrl-Left scroll it by hand,
    and a highlight walking off either edge takes it along.

    Scrolling up and down is Textual's own, since the pane really is a scrollable
    region: the wheel moves it and nothing here has to know. What this does have
    to answer is where a click landed, which only the pane can say - it is the
    one that knows what row each entry was painted on.
    """

    can_focus = False

    def __init__(self, *args, **kwargs) -> None:
        # Built here rather than composed, because a pane is made when its
        # window is and painted in the same breath - before Textual has got
        # round to composing it.
        #
        # A class rather than an id: there is one of these per window, and an
        # id would have to be unique across all of them.
        self.content = Static(classes="work-content")
        super().__init__(self.content, *args, **kwargs)
        #: Columns the selected entry's render is currently shifted left by.
        self.shift = 0
        #: The selection the shift was last brought into view for, so that a
        #: highlight standing still is not chased and a scrolled pane stays
        #: where the user put it.
        self._followed: object = None
        #: The selected entry's width and label field, as the last render saw
        #: them, which is what a shift has to stay inside.
        self._widest = 0
        self._indent = _LABEL_WIDTH
        #: Where the last render put each entry: the row its render starts on,
        #: the column the render starts in, and how many rows it took. What a
        #: click is read against, there being nothing else on screen that says
        #: which expression a row belongs to.
        self._drawn: list[tuple[int, int, int]] = []
        #: Which of them was drawn shifted, since the shift is the selected
        #: entry's alone and a click on it has to be read past the shift.
        self._shifted: int | None = None

    def reset(self) -> None:
        """Forget where this pane stands, as a pane handed to a new window must."""
        self.shift = 0
        self._followed = None
        self._widest = 0
        self._indent = _LABEL_WIDTH
        self._drawn = []
        self._shifted = None

    def show(
        self,
        entries: list[Entry],
        selected: int | None,
        rect: tuple[int, int, int, int] | None,
    ) -> None:
        styles = self.app.palette.styles
        self._follow(entries, selected, rect)
        lines: list[str] = []
        # For each entry, the line it starts on and the column its render does.
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
        text = Text("\n".join(lines), style=styles["work"], no_wrap=True)
        if selected is not None and rect is not None:
            _invert(
                text,
                _offsets(lines),
                starts[selected],
                indents[selected],
                rect,
                styles["selection"],
                self.shift,
            )
        self.content.update(text)
        if selected is not None:
            # After the refresh, so the new content has been laid out and the
            # scrollable region knows how tall it is.
            region = Region(0, starts[selected], 1, entries[selected].height)
            self.call_after_refresh(self.scroll_to_region, region, animate=False)

    def show_help(self, lines: list[str], rows: int, width: int, titled: bool) -> None:
        """Paint a help page over this window's expressions.

        Help is not an overlay: it stands where the expressions stand, in the
        window it was asked from and in no other, with that window's border
        still drawn around it and every other window still showing its own
        worksheet. There is no border of its own, no scrollbar and no shadow,
        which is how the original drew it.

        The page is padded out to the pane's full height, so that a document
        shorter than the pane reads from the top rather than being pushed
        against the bottom the way a history is. A titled page carries its
        title on the first row in inverse video, over the title's own columns
        and no further.
        """
        styles = self.app.palette.styles
        self._drawn, self._shifted = [], None
        page = [line[:width] for line in lines[:rows]]
        page += [""] * (rows - len(page))
        text = Text("\n".join(page), style=styles["work"], no_wrap=True)
        if titled and page[0].strip():
            begin = len(page[0]) - len(page[0].lstrip())
            text.stylize(styles["selection"], begin, len(page[0]))
        self.content.update(text)
        self.call_after_refresh(self.scroll_to, 0, 0, animate=False)

    def show_greeting(
        self, lines: Sequence[str], footer: str, rows: int, width: int
    ) -> None:
        """Paint the opening notice where this window's expressions will stand.

        Centred across the pane and spread down it: the name block stands in
        the upper third and the footer a couple of rows off the bottom, which
        is how a title page is spaced and how the original spaced this one.
        The whole of it is padded out to the pane's full height, so that it
        reads as a page rather than being pushed against the bottom the way a
        history is. A pane too short for both gives up the tail of the block
        rather than the footer, that being the line with a key in it.
        """
        styles = self.app.palette.styles
        self._drawn, self._shifted = [], None
        centred = [line.center(width).rstrip()[:width] for line in lines]
        above = max(1, (rows - len(centred) - 3) // 3)
        page = ([""] * above + centred)[: max(0, rows - 2)]
        at = max(len(page), rows - 3)
        page += [""] * (at - len(page))
        if len(page) < rows:
            page.append(footer.center(width).rstrip()[:width])
        page += [""] * (rows - len(page))
        text = Text("\n".join(page), style=styles["work"], no_wrap=True)
        self.content.update(text)

    def pointed(self, offset: Offset) -> tuple[int, int, int] | None:
        """Which entry the screen cell `offset` is on, and where in its render.

        The row and the column are the entry's own: they index the lines the
        layout painted, so what the caller does with them is a question about
        the expression rather than about the pane. A column left of the render
        comes back negative, which is what a click on the label field is - a
        cell that names the whole expression and no part of it.

        None where the cell is on no entry at all: the blank row between two of
        them, or the space above a history too short to fill the pane.
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
        """Columns the pane has for a render, once the label field is taken."""
        return max(1, self.size.width - self._indent)

    def _inside(self, shift: int) -> int:
        """A shift clipped to the render it scrolls.

        Neither end has anything to show past it, and a pane full of blanks is
        a worse answer than one that has stopped.
        """
        return max(0, min(shift, self._widest - self.showable))

    def _follow(
        self,
        entries: list[Entry],
        selected: int | None,
        rect: tuple[int, int, int, int] | None,
    ) -> None:
        """Take in what a shift needs, and bring a moved highlight into view.

        The render moves as little as it takes and only when the highlight has
        moved, so a pane scrolled by hand stays where it was put until the
        selection walks off it. A couple of columns are kept on show to either
        side, since a subexpression flush against the edge reads as one that
        has been cut off.

        The two ends cannot both be honored for a selection wider than the pane,
        and the left one wins: that is where an expression is read from.
        """
        if selected is None:
            self.shift, self._widest, self._indent = 0, 0, _LABEL_WIDTH
            return
        entry = entries[selected]
        self._widest, self._indent = entry.layout.width, _indent(entry)
        # Before there is a pane to measure there is nothing to bring into it,
        # and the selection stays unfollowed rather than followed wrongly.
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
        """Scroll sideways, which moves no selection and reveals no new entry.

        A third of the showable width at a time, which leaves two thirds of the
        render standing still for the eye to carry across.
        """
        self.shift = self._inside(
            self.shift + direction * max(1, self.showable // _SCROLL_SHARE)
        )


class MessageLine(Band):
    """One line saying what is happening or what input is expected."""

    def show(self, message: str) -> None:
        self.update(Text(f" {message}", style=self.colors["prompt"]))


class PromptLine(Input):
    """The line every command that reads a line reads on.

    Textual's own line inserts and never overwrites, so the original's second
    text input mode is added here: with `overwrite` set, a character typed
    stands on the one at the cursor instead of pushing it right. The key has to
    be taken before the base class sees it, since that is where the insertion
    would happen; a typed character over a selection replaces the selection in
    either mode, which is what the base class does anyway.

    The other addition is a kill ring, as Emacs, bash and every line editor
    descended from them have: the deletions that take a run of text rather than
    a character - Ctrl-U, Ctrl-K, Ctrl-W and Alt-Backspace - keep what they
    took, Ctrl-Y puts the last of it back at the cursor, and Alt-Y straight
    after a Ctrl-Y swaps that for the kill before it and keeps going back
    through the ring. Deletions that follow one another with nothing in between
    join into one kill, so a line taken apart a word at a time comes back
    whole. The ring belongs to the line rather than to the command that put it
    up, and the line is one widget for the whole run of the program: what was
    deleted from an Author line can be put back on the next one, or on a
    Substitute value, for as long as the session lasts.
    """

    BINDINGS = [
        Binding("ctrl+y", "yank", "Restore deleted text", show=False),
        Binding("alt+y", "yank_pop", "Restore an earlier deletion", show=False),
    ]

    overwrite = False

    def __init__(self, **arguments: Any) -> None:
        super().__init__(**arguments)
        #: What the deletions have taken out, oldest first.
        self.kills: list[str] = []
        #: Which way the deletion under way runs - -1 for one that takes text
        #: before the cursor, 1 for one that takes text after it - or None
        #: while what is being deleted is not a kill at all.
        self._killing: int | None = None
        #: The line and cursor the last kill left, which is what says whether
        #: the next one carries straight on from it.
        self._killed_at: tuple[str, int] | None = None
        #: What the last yank put in: which kill it was, where it went, and
        #: the line it left behind. Alt-Y needs all three, to tell a yank
        #: still standing as it landed from one since typed over.
        self._yanked: tuple[int, int, int, str] | None = None

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

    # -- deleting, and putting back what was deleted ------------------------

    def action_delete_left_all(self) -> None:
        """Ctrl-U: take out the line before the cursor, and keep it."""
        self._kill(-1, super().action_delete_left_all)

    def action_delete_left_word(self) -> None:
        """Ctrl-W: take out the word before the cursor, and keep it."""
        self._kill(-1, super().action_delete_left_word)

    def action_delete_right_all(self) -> None:
        """Ctrl-K: take out the line from the cursor on, and keep it."""
        self._kill(1, super().action_delete_right_all)

    def action_delete_right_word(self) -> None:
        """Alt-Backspace: take out the word after the cursor, and keep it."""
        self._kill(1, super().action_delete_right_word)

    def _kill(self, direction: int, delete: Callable[[], None]) -> None:
        """Run `delete`, and take what it removes as a kill running `direction`.

        The base class works out for itself how far each of them reaches, and
        one of them falls back on another, so the reach is not read here: the
        flag says only that whatever `delete` takes out is worth keeping.
        """
        was, self._killing = self._killing, direction
        try:
            delete()
        finally:
            self._killing = was

    def delete(self, start: int, end: int) -> None:
        """Take out the text between `start` and `end`, keeping any kill.

        Every deletion the line makes comes through here, which is what makes
        this the one place the killed text can be caught. Backspace and Delete
        come through it too and pass straight on: a character taken out one at
        a time is a correction rather than something to put back later.
        """
        if self._killing is None:
            super().delete(start, end)
            return
        start, end = sorted((max(0, start), min(len(self.value), end)))
        taken = self.value[start:end]
        before = (self.value, self.cursor_position)
        super().delete(start, end)
        if not taken or self.value == before[0]:
            # A kill with nothing in reach - Ctrl-K at the end of the line -
            # leaves the run it was part of standing, so that the kills to
            # either side of it still join.
            return
        self._keep(taken, joins=before == self._killed_at)
        self._killed_at = (self.value, self.cursor_position)

    def _keep(self, taken: str, joins: bool) -> None:
        """Put `taken` on the ring, or onto the end of the kill before it.

        A kill that starts where the last one finished is a continuation of
        it, so the two are kept as one: three words taken one at a time come
        back as three words. Which end of the kill it joins is which way the
        deletion ran, so text taken leftwards goes in front of what is there.
        """
        if joins and self.kills:
            last = self.kills[-1]
            self.kills[-1] = last + taken if self._killing == 1 else taken + last
            return
        self.kills.append(taken)
        del self.kills[:-_KILLS_KEPT]

    def action_yank(self) -> None:
        """Ctrl-Y: put the last kill back in at the cursor.

        Over whatever is selected, as a typed character is, and with the
        cursor left after it so that the line goes on from where the restored
        text now ends.
        """
        if self.kills:
            self._put(len(self.kills) - 1, *sorted(self.selection))

    def action_yank_pop(self) -> None:
        """Alt-Y: swap what Ctrl-Y just put in for the kill before it.

        Only straight after a yank, with what it put in still standing where
        it landed: anything typed since, or the cursor moved off the end of
        it, and there is no yank left to reconsider. Pressed again it goes on
        back through the ring, and round to the newest again past the oldest.
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
    """The names a half-typed filename could mean, as a list to look through.

    It opens above the line being typed, one name per row, the name the line is
    currently showing highlighted in inverse video. A directory keeps the
    separator that says it is one, so that the list reads as a place to go into
    rather than a file to open.

    The list is a window onto the names rather than all of them: `MAX_ROWS`
    rows at a time, scrolled to keep the highlighted one inside, with the count
    in the border title saying how much is out of sight. That is what makes it
    a thing to look around a directory tree with, which a line of text cut
    short at the pane's edge never was.
    """

    #: The most rows the list takes from the work area above it. Ten is enough
    #: for a directory to be taken in at a glance and still leaves the newest
    #: expressions on screen at the sizes a terminal usually comes in.
    MAX_ROWS = 10

    #: What the screen owes to everything that is not a name in this list: its
    #: own two border rows, then the rule, the two the prompt band takes, the
    #: message line, the status line, and one row of work area. The line being
    #: typed is the one thing a list of names for it can never be worth
    #: covering, so on a short terminal the list gives up rows rather than the
    #: screen giving up the line.
    ROWS_ELSEWHERE = 8

    def __init__(self, *arguments: object, **named: object) -> None:
        super().__init__(*arguments, **named)
        #: What the list is currently showing, kept so that a resize can draw
        #: it again: both the highlight bar and the title are cut to the width,
        #: and the width is not known until the widget has been laid out.
        self._showing: tuple[list[str], int | None, str] = ([], None, "")
        #: Which name the top row is showing, since the list is a window onto
        #: the names and a click names a row rather than a name.
        self._first = 0

    def visible_rows(self, count: int) -> int:
        """How many of `count` names show at once: as many as there is room for."""
        room = self.app.size.height - self.ROWS_ELSEWHERE
        return max(1, min(count, self.MAX_ROWS, room))

    def rows_for(self, count: int) -> int:
        """How tall the list stands to show `count` names, borders included."""
        return self.visible_rows(count) + 2

    def on_resize(self) -> None:
        """Draw again for the new width, and for the new number of rows.

        A terminal that has just become short is why the height is set here as
        well as when the list is filled: the rows the list may take have
        changed, and the list is the one that has to give them up.
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
            # Every row is padded to the full width so that the highlight reads
            # as a bar across the list rather than a box around a word.
            width = max(self.size.width - 2, len(name) + 1)
            text.append(f" {name}".ljust(width) + "\n", style=style)
        self.border_title = self._title(where, len(names), first, len(shown))
        self.update(text)

    def name_at(self, offset: Offset) -> int | None:
        """Which name the screen cell `offset` is on, if it is on one.

        A row is one name across the whole width of the list, the highlight bar
        being drawn that way, so only the row is asked about. None where the
        cell is on the border or past the last name.
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
        """The name the window starts at, chosen to keep `at` inside it."""
        if at is None or count <= rows:
            return 0
        # Centre the highlighted name where there is room on both sides, so
        # that what is coming up is as visible as what has gone by.
        return max(0, min(at - rows // 2, count - rows))

    def _title(self, where: str, count: int, first: int, shown: int) -> str:
        """What the border says: the directory, and how much of it is showing.

        The count is kept whole and the directory is cut to fit around it,
        from the left: the end of a path says where it is, and the start of one
        is usually the part that was the same for everything on offer anyway.
        """
        seen = f"{first + 1}-{first + shown}"
        tally = f"{count}" if shown >= count else f"{seen} of {count}"
        # The title sits in the top border, between the corners, and carries
        # the dashes, the spaces and the dividing hyphen around the two parts.
        room = self.size.width - len(tally) - 10
        if room > 0 and len(where) > room:
            where = "..." + where[-(room - 3) :]
        return f" {where} - {tally} "


class MenuBand(Band):
    """A menu of words, one of them highlighted in inverse video.

    Where each word was drawn is kept as it is drawn, so that a click can be
    answered with the word it landed on. The words are the only thing on the
    band a mouse has any business with: the title is a caption.

    The word the pointer is resting on is underlined, which is the one thing
    the band says about the mouse rather than about the session: a menu of
    words a keyboard drives does not otherwise look clickable, and a mark that
    follows the pointer is how it says that it is.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #: The menu on the band, so that a resize can lay it out again.
        self._showing: tuple[Menu, int | None] | None = None
        #: The columns each word took, as row, first column, last column and
        #: which word of the menu it is.
        self._words: list[tuple[int, int, int, int]] = []
        #: The screen cell the pointer is resting on, or None while it is
        #: resting on nothing this band offers. Kept as a cell rather than as a
        #: word, so that a band drawn again - a submenu opened under a pointer
        #: that has not moved, a terminal resized - marks whatever word now
        #: stands there.
        self._pointer: Offset | None = None

    def say(self, text: str) -> None:
        """Put one line of plain text on the band in place of a menu.

        What a demonstration's script is shown on: the comment above the
        expression it has just run stands where the menu words go.
        """
        self._showing = None
        self._words = []
        self.put([Text(f" {text}", style=self.colors["option"], no_wrap=True)])

    def show(self, menu: Menu, highlighted: int | None) -> None:
        """Render `menu`; `None` highlights nothing, as while Quit asks."""
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
        """Rest the pointer on the screen cell `offset`, or on nothing at all.

        The band is drawn again only when this changes which word the pointer
        is over, a word being as many cells wide as it has letters and the
        pointer crossing every one of them on its way across.
        """
        moved = self.word_at(offset) != self.word_at(self._pointer)
        self._pointer = offset
        if moved and self._showing is not None:
            self.show(*self._showing)

    def word_at(self, offset: Offset | None) -> int | None:
        """Which word of the menu the screen cell `offset` is on, if any is.

        None where the cell is on none of them, and where there is no cell:
        a pointer that is off the band is on no word.
        """
        if offset is None:
            return None
        point = self.within(offset)
        for row, start, end, index in self._words:
            if row == point.y and start <= point.x < end:
                return index
        return None

    def on_resize(self) -> None:
        """Lay the menu out again for the width the terminal now has."""
        if self._showing is not None:
            self.show(*self._showing)

    def _mark(self, rows: list[Text], words: list[tuple[int, int, int, int]]) -> None:
        """Underline the word the pointer is resting on, if it is on one.

        Read off the layout just made rather than off the last one, so that a
        word that has moved under a pointer standing still is the word marked.
        """
        if self._pointer is None:
            return
        point = self.within(self._pointer)
        for row, start, end, _ in words:
            if row == point.y and start <= point.x < end:
                rows[row].stylize(self.colors["option-pointed"], start, end)
                return

    def _rows(self, menu: Menu, prefix: int) -> list[list[int]]:
        """Which words go on each row, as indices into the menu.

        Every row carries the same prefix - the title, or the indent that
        stands under it - so all of them have the same room for words.

        The break the menu names is kept while both of its lines fit, which is
        what keeps eighty columns showing what the original shows. Once one of
        them does not, that break is a convention about a width this terminal
        has not got, and the words are packed across the rows instead: fewer
        rows than breaking each line separately takes, and none half empty.
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
        """One row of the menu, and the columns each word on it took."""
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
    """An Options dialog: its title and its fields, the active one live.

    A field that is not the active one shows its value in parentheses, the way
    the original distinguishes "this is what it is set to" from "this is what
    you are setting".

    A line may also carry a plain word, which is printed where it stands and is
    no field: the variable whose bounds `Declare Variable Interval` is asking
    for stands between the two bounds.

    Where every field and every one of its choices was drawn is kept as it is
    drawn, so that a click can be answered with what it landed on: the field, or
    the one value of it the pointer is over. A dialog laid out for the width
    there is has no fixed columns to compute them from afterwards.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #: The dialog on the band, so that a resize can lay it out again.
        self._showing: DialogEditor | None = None
        #: Where each field went, as its row, the columns it took, which field
        #: of the dialog it is, and the columns each of its choices took.
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
                # A field with no caption prints a space in front of its
                # value, and that space is the one the title would print, so
                # the title goes without: `MANAGE BRANCH: Principal Real Any`.
                # The value still begins in the column the next line indents
                # to, which is what keeps the two lines square.
                prefix = prefix[:-1]
            drawn: list[Text] = []
            # Which field each of them is, and where its choices are inside it;
            # a plain word is neither and belongs to no field at all.
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
            # A line too wide for the terminal is broken between its fields
            # and carried on under the title, the same way a menu too wide for
            # it is. A field is never split: what is asked and what stands as
            # the answer belong beside each other.
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
        """What the screen cell `offset` is on: which field, and which choice.

        The choice is None where the cell is on the field but on no value of it
        - a label, or the blanks a number field is padded out to - which is a
        cell that says which field without saying what to set it to. None
        altogether where the cell is on no field: a plain word, or the title.
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
        """Lay the dialog out again for the width the terminal now has."""
        if self._showing is not None:
            self.show(self._showing)

    def _field(
        self, editor: DialogEditor, field: Field, active: bool
    ) -> tuple[Text, tuple[tuple[str, int, int], ...]]:
        """A field as it is drawn, and where each choice of it went inside that.

        The columns are the field's own; the caller offsets them by where it
        put the field. A field with nothing to pick from - one that is typed
        into, or one whose choices are too many to print - offers none.
        """
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
        """`Mode:(Character)Word` - the parentheses eat the spaces beside them.

        Each choice takes the blank in front of it as well, so that every column
        of the field belongs to the word it stands beside and a click between
        two of them lands on one rather than on neither.
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
    """The region the windows are drawn in: their borders and their contents.

    The work areas are placed over it absolutely, one per window, and the
    frame beneath them draws what is between. Both are asked for again
    whenever this is resized, since where a divider falls is a question about
    how much room there is.
    """

    def drawing(self) -> Drawing:
        """The border as it stands, one row longer than this widget is tall."""
        return self.app.windows.frame(self.size.height, self.size.width)

    def on_resize(self) -> None:
        self.app.place_windows()


class Frame(Band):
    """The borders around and between the windows.

    Blank while there is one window, which is how the original draws an
    unsplit screen: no frame at all, and the whole 80 by 20 for expressions.
    The number in each window's upper left corner is drawn here too, and the
    active window's is the one cell of the frame in inverse video - which is
    the only mark on screen saying which window a command would act on.
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

    It is the window frame's bottom edge, so it carries the corners and the
    feet of every vertical divider once the screen has been split, and is a
    plain rule while it has nothing to join.
    """

    def render(self) -> Text:
        rule = self.screen.query_one(Panes).drawing().rows[-1]
        return Text(rule[: self.size.width], style=self.colors["frame"], no_wrap=True)


class StatusLine(Band):
    """Bottom line: selection annotation, current file, memory field, pane type.

    Everything on it belongs to the active window: the annotation is the one
    on the expression highlighted there, and the pane field names that
    window's type. One line for however many windows are open, as the original
    has it.
    """

    annotation = ""
    file = ""
    #: The original's muLISP heap gauge lived here, saying how much of its
    #: workspace was still free. The slot now says how much memory the process
    #: holds, which is the closest thing a hosted program can report, and stays
    #: empty on a platform that will not say.
    center = ""
    #: The mode words that stand beside the memory field: `Ovr` while the text
    #: input mode is overwrite rather than insert, `Lin` while the arrow keys
    #: belong to the line being edited rather than to the highlight. Each is
    #: named only in the state that is the odd one out, so the line stays quiet
    #: where a line behaves as it usually does, and quiet altogether where
    #: there is no line being typed.
    flags = ""
    pane = "Rederive Algebra"

    def on_mount(self) -> None:
        self._read_memory()
        self.set_interval(_MEMORY_PERIOD, self._read_memory)

    def _read_memory(self) -> None:
        """Take the memory field's reading, repainting only when it moves."""
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
        """Set the mode words alone, the rest of the line standing as it is.

        A key that changes how the line being typed behaves changes nothing
        else on screen, and a whole refresh would have to know what band the
        command it interrupted had up.
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
