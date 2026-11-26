"""The help dialog: a topic tree beside a reading pane, over the dimmed worksheet.

The tree lists the topics, with the sub-topics of the one being read under it. Moving
through the tree shows each page as it is reached, and Enter, Right or Tab moves into
the page to scroll it. Paging past the end of a page goes on to the next one in reading
order, and paging up before its start back to the previous one, so a topic can be read
straight through. Backspace goes back through the pages visited, whichever way they
were reached: the tree, a link, search, or paging.

`/` focuses the search box. While it holds a query, the results take the tree's place,
and Up, Down and Enter choose among them without leaving the box. A result that is a
name or a command lands on its table row, which is highlighted until the page changes
or an item on it is picked. Esc in the box clears it and brings the tree back.

On the page, Tab and Shift-Tab pick its examples, programs and links in turn (see
`HelpScreen`). Enter follows a link or copies the rest, and a click on an example or a
program copies it too.

Below `NARROW` columns the tree and the page take turns at the whole width: Enter shows
the page, and Esc on the page returns to the tree. Esc on the tree, or on the page when
both are shown, closes the dialog.

Every color comes from the `$rd-*` slots, styled in `rederive.tcss`. Code blocks are
shown plain, since syntax highlighting would bring colors of its own.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.geometry import Region, Size
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Input, Label, OptionList, Static
from textual.widgets.markdown import Markdown, MarkdownBlock, MarkdownFence
from textual.widgets.option_list import Option

from rederive.model import help as helps
from rederive.model.settings import Settings

#: Below this width the tree and the page alternate instead of sitting side by side.
NARROW = 80

#: The separator between a topic and a sub-topic, in the line over the page.
CRUMB = " \N{SINGLE RIGHT-POINTING ANGLE QUOTATION MARK} "

#: Tree markers: the topic being read, with its sub-topics open under it, and the
#: page on screen.
OPEN = "\N{BLACK DOWN-POINTING SMALL TRIANGLE} "
SHOWN = "\N{BLACK RIGHT-POINTING SMALL TRIANGLE} "

#: The key hints in the footer, for where the keys go.
TREE_KEYS = (
    ("\N{UPWARDS ARROW}\N{DOWNWARDS ARROW}", "topic"),
    ("Enter", "read"),
    ("Tab", "pane"),
    ("/", "search"),
    ("\N{ERASE TO THE LEFT}", "back"),
    ("Esc", "close"),
)
PAGE_KEYS = (
    ("\N{UPWARDS ARROW}\N{DOWNWARDS ARROW}", "scroll"),
    ("PgUp/PgDn", "page"),
    ("Tab", "pane"),
    ("/", "search"),
    ("\N{ERASE TO THE LEFT}", "back"),
    ("Esc", "close"),
)
#: In the narrow layout, where Tab has nothing beside to move to.
NARROW_TREE_KEYS = tuple(hint for hint in TREE_KEYS if hint[0] != "Tab")
NARROW_PAGE_KEYS = (
    ("\N{UPWARDS ARROW}\N{DOWNWARDS ARROW}", "scroll"),
    ("PgUp/PgDn", "page"),
    ("\N{ERASE TO THE LEFT}", "back"),
    ("Esc", "topics"),
)
SEARCH_KEYS = (
    ("\N{UPWARDS ARROW}\N{DOWNWARDS ARROW}", "result"),
    ("Enter", "go"),
    ("Esc", "clear"),
)

#: The hints for the page's items, put after the paging hint: Tab where the page has
#: items, then Enter for the item picked, and F3 if a line is waiting for it.
ITEM_KEY = ("Tab", "item")
ACT_KEYS = {"link": ("Enter", "follow"), "copy": ("Enter", "copy")}
INSERT_KEY = ("F3", "insert")

#: The style of a link picked. The link's own colors are drawn over any a span gives
#: it, so it is picked by showing them in inverse video.
PICKED_LINK = "reverse"

#: The marker at the right edge of every snippet's first line: it can be copied.
MARKER = "\N{TWO JOINED SQUARES}"

#: What the footer says for a while after a copy, and for how many seconds.
COPIED = "Copied"
COPIED_SECONDS = 1.5

#: What a search result's kind is shown as, after its label.
KIND_NOTES = {"command": "command", "library": "utility"}


class PlainFence(MarkdownFence):
    """A code block shown as written, in the text color.

    A block marked `examples` or `program` holds snippets to copy (`helps.snippets`).
    Each has `MARKER` at the right edge of its first line, in the option color, over
    whatever text is there. The snippet picked from the keyboard is drawn in the
    highlight, and the one under the pointer in the selection color, each across the
    block. A click copies the snippet under the pointer and picks it.
    """

    def __init__(self, markdown: Markdown, token: Any, code: str) -> None:
        super().__init__(markdown, token, code)
        self.snippets = helps.snippets(token.info, code)
        #: The snippet picked and the one under the pointer, by index, or None.
        self.picked: int | None = None
        self.pointed: int | None = None

    @classmethod
    def highlight(
        cls, code: str, language: str, ansi: bool = False, dark: bool = False
    ) -> Content:
        return Content(code)

    def compose(self) -> ComposeResult:
        yield FenceCode(id="code-content")

    def snippet_at(self, line: int) -> int | None:
        """The snippet taking up `line` of the block, or None."""
        for index, snippet in enumerate(self.snippets):
            first, last = snippet.lines
            if first <= line < last:
                return index
        return None

    def pick(self, index: int | None) -> None:
        self.picked = index
        self.query_one(FenceCode).refresh()

    def point(self, index: int | None) -> None:
        if index != self.pointed:
            self.pointed = index
            self.query_one(FenceCode).refresh()

    def drawn(self, width: int) -> Content:
        """The block drawn `width` cells wide."""
        styles = self.app.palette.styles  # type: ignore[attr-defined]
        shown: dict[int, str] = {}
        for index, style in (
            (self.pointed, styles["selection"]),
            (self.picked, styles["option-highlight"]),
        ):
            if index is not None and index < len(self.snippets):
                first, last = self.snippets[index].lines
                shown.update(dict.fromkeys(range(first, last), style))
        starts = {snippet.lines[0] for snippet in self.snippets}
        lines: list[Content] = []
        for number, text in enumerate(self.code.split("\n")):
            style = shown.get(number)
            if number in starts:
                room = max(0, width - len(MARKER) - 1)
                text = text[:room].ljust(room) + " "
                line = Content.assemble(
                    Content.styled(text, style or ""),
                    Content.styled(MARKER, style or styles["option"]),
                )
            elif style is None:
                line = Content(text)
            else:
                line = Content.styled(text.ljust(width), style)
            lines.append(line)
        return Content("\n").join(lines)


class FenceCode(Label):
    """The lines of a code block, which follow the pointer over its snippets."""

    @property
    def fence(self) -> PlainFence:
        parent = self.parent
        assert isinstance(parent, PlainFence)
        return parent

    def render(self) -> Content:
        return self.fence.drawn(self.content_size.width)

    def _snippet(self, event: events.MouseEvent) -> int | None:
        offset = event.get_content_offset(self)
        return None if offset is None else self.fence.snippet_at(offset.y)

    def on_mouse_move(self, event: events.MouseMove) -> None:
        self.fence.point(self._snippet(event))

    def on_leave(self, event: events.Leave) -> None:
        self.fence.point(None)

    def on_click(self, event: events.Click) -> None:
        index = self._snippet(event)
        if index is not None:
            event.stop()
            self.screen.clicked(self.fence, index)  # type: ignore[attr-defined]


class HelpTable(MarkdownBlock):
    """A table, drawn as lines of text rather than a grid of widgets.

    Each column is as wide as its widest cell if the table fits the pane. If it
    doesn't, the last column, where the descriptions are, takes what the others leave,
    and they narrow to leave it at least half. Cells wrap within their column. A rule
    under the header, in the frame color, is the only line drawn.
    """

    #: Columns between two cells.
    GAP = 2

    def __init__(self, markdown: Markdown, token: Any) -> None:
        super().__init__(markdown, token)
        self.headers: list[Content] = []
        self.rows: list[list[Content]] = []
        #: The body row highlighted, counted from 1, or None.
        self.found: int | None = None
        #: The link picked, as its body row, its cell and its span in the cell, or None.
        self.picked: tuple[int, int, int, int] | None = None
        #: The lines of each body row in the drawing, as (first, past the last).
        self.row_lines: dict[int, tuple[int, int]] = {}
        #: The width, highlighted row and picked link the lines were last made for, and
        #: the lines.
        self._made: tuple[int, int | None, tuple[int, ...] | None] | None = None
        self._lines: list[Content] = []

    def compose(self) -> ComposeResult:
        # The header and body cells were parsed as blocks of their own; only their
        # contents are kept.
        for block in _flattened(self):
            if block.name == "th_open":
                self.headers.append(block._content)
            elif block.name == "tr_open":
                self.rows.append([])
            elif block.name == "td_open" and self.rows:
                self.rows[-1].append(block._content)
        self.rows = [row for row in self.rows if row]
        self._blocks.clear()
        return iter(())

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        # The lines are made for the width the layout offers, so the table has its
        # height in the layout that first shows it.
        return len(self.lines(width))

    def render(self) -> Content:
        return Content("\n").join(self.lines(self.size.width))

    def mark(self, row: int | None) -> None:
        """Highlight body row `row`, or none."""
        self.found = row
        self.refresh()

    def pick(self, link: tuple[int, int, int, int] | None) -> None:
        """Highlight a link, given as `links` gives it, or none."""
        self.picked = link
        self.refresh()

    def links(self) -> Iterator[tuple[tuple[int, int, int, int], str]]:
        """Each link in the body rows, as (row, cell, start, end), and its target."""
        for number, row in enumerate(self.rows, 1):
            for column, cell in enumerate(row):
                for start, end, href in _links(cell):
                    yield (number, column, start, end), href

    def lines(self, width: int) -> list[Content]:
        """The table drawn `width` cells wide."""
        if not width or not self.headers:
            return []
        made = (width, self.found, self.picked)
        if self._made == made:
            return self._lines
        styles = self.app.palette.styles  # type: ignore[attr-defined]
        widths = _fitted(self.headers, self.rows, width, self.GAP)
        lines: list[Content] = []
        lines += _row(self.headers, widths, self.GAP, styles["option"] + " bold")
        rule = "\N{BOX DRAWINGS LIGHT HORIZONTAL}" * width
        lines.append(Content.styled(rule, styles["frame"]))
        self.row_lines = {}
        for number, row in enumerate(self.rows, 1):
            first = len(lines)
            style = styles["option-highlight"] if number == self.found else ""
            if self.picked is not None and self.picked[0] == number:
                _, column, start, end = self.picked
                row = list(row)
                row[column] = row[column].stylize(PICKED_LINK, start, end)
            lines += _row(row, widths, self.GAP, style, width)
            self.row_lines[number] = (first, len(lines))
        self._made, self._lines = made, lines
        return lines

    def row_region(self, row: int) -> Region | None:
        """Where body row `row` is, relative to this table."""
        if row not in self.row_lines:
            return None
        first, last = self.row_lines[row]
        return Region(0, first, self.size.width, last - first)


def _links(content: Content) -> Iterator[tuple[int, int, str]]:
    """The links in a block's text: the span of each, and its target."""
    for span in content.spans:
        meta = getattr(span.style, "meta", None) or {}
        action = _LINK.fullmatch(str(meta.get("@click", "")))
        if action is not None:
            yield span.start, span.end, ast.literal_eval(action.group(1))


#: The click action of a link, with its target as a Python literal.
_LINK = re.compile(r"link\((.*)\)")


def _line_of(content: Content, offset: int, width: int) -> int:
    """The line that character `offset` of `content` is on, wrapped `width` wide."""
    plain = content.plain
    at = 0
    for number, line in enumerate(content.wrap(max(1, width))):
        text = line.plain.strip()
        found = plain.find(text, at) if text else at
        if found > offset:
            return max(0, number - 1)
        at = found + len(text)
        if offset < at:
            return number
    return 0


def _flattened(block: MarkdownBlock) -> Iterator[MarkdownBlock]:
    for child in block._blocks:
        yield child
        yield from _flattened(child)


def _fitted(
    headers: list[Content], rows: list[list[Content]], width: int, gap: int
) -> list[int]:
    """Column widths for a table drawn `width` cells wide."""
    columns = len(headers)
    cells = [headers, *rows]
    widths = [
        max(
            (
                max((line.cell_length for line in row[column].split("\n")), default=0)
                for row in cells
                if column < len(row)
            ),
            default=0,
        )
        for column in range(columns)
    ]
    least = [
        max(
            (
                max((len(word) for word in row[column].plain.split()), default=1)
                for row in cells
                if column < len(row)
            ),
            default=1,
        )
        for column in range(columns)
    ]
    room = width - gap * (columns - 1)
    last = columns - 1
    # The last column, where the descriptions are, keeps half the room, or what it
    # needs if that is less: the other columns give up a cell at a time, the widest
    # first, never below their longest word.
    wanted = min(widths[last], max(room // 2, least[last]))
    while sum(widths[:last]) + wanted > room:
        shrinkable = [
            column for column in range(last) if widths[column] > least[column]
        ]
        if not shrinkable:
            break
        widths[max(shrinkable, key=lambda column: widths[column])] -= 1
    widths[last] = max(least[last], min(widths[last], room - sum(widths[:last])))
    if sum(widths) > room:
        # Words longer than their column are folded.
        widths = [max(1, room * share // max(1, sum(widths))) for share in widths]
    return widths


def _row(
    cells: list[Content], widths: list[int], gap: int, style: str, width: int = 0
) -> list[Content]:
    """The lines of one table row, its cells wrapped to their columns."""
    wrapped = [
        cell.wrap(column) if column else [Content()]
        for cell, column in zip(cells, widths)
    ]
    height = max((len(lines) for lines in wrapped), default=1)
    lines: list[Content] = []
    for index in range(height):
        parts: list[Content] = []
        for column, (cell, size) in enumerate(zip(wrapped, widths)):
            text = cell[index] if index < len(cell) else Content()
            text = text.pad_right(size - text.cell_length)
            parts.append(text if column == len(widths) - 1 else text.pad_right(gap))
        line = Content("").join(parts).rstrip()
        if width:
            line = line.pad_right(width - line.cell_length)
        lines.append(line.stylize_before(style) if style else line)
    return lines


class HelpDocument(Markdown):
    """A page of help, its links handled by the dialog rather than a browser."""

    BLOCKS = {
        **Markdown.BLOCKS,
        "fence": PlainFence,
        "code_block": PlainFence,
        "table_open": HelpTable,
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(parser_factory=helps.parser, open_links=False, **kwargs)


@dataclass(frozen=True)
class Spot:
    """An item of the page on screen (`helps.Item`), and where it is drawn.

    `at` says where in `block`: the index of the snippet in a code block, the link as
    `HelpTable.links` gives it in a table, and the span of the link in the text of any
    other block.
    """

    kind: str
    text: str
    block: MarkdownBlock
    at: Any

    @property
    def copied(self) -> bool:
        """Whether the item is something to copy rather than a link."""
        return self.kind != "link"


@dataclass(frozen=True)
class Closed:
    """What the help dialog is dismissed with: the page last shown, and the text to
    insert on the line help was opened from, or None."""

    address: str
    insert: str | None = None


class HelpPage(VerticalScroll, inherit_bindings=False):
    """The reading pane. Paging past either end turns to the next or previous page.

    Enter acts on the item picked, and F3 inserts it on the line help was opened from.
    """

    BINDINGS = [
        Binding("up", "scroll_up", "Scroll up", show=False),
        Binding("down", "scroll_down", "Scroll down", show=False),
        Binding("home", "scroll_home", "Top", show=False),
        Binding("end", "scroll_end", "Bottom", show=False),
        Binding("pageup", "turn(-1)", "Page up", show=False),
        Binding("pagedown", "turn(1)", "Page down", show=False),
        Binding("enter", "screen.act", "Act", show=False),
        Binding("f3", "screen.insert", "Insert", show=False),
    ]

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        if isinstance(self.screen, HelpScreen):
            self.screen.scrolled()

    def on_blur(self, event: events.Blur) -> None:
        if isinstance(self.screen, HelpScreen):
            self.screen.unpick()

    def action_turn(self, direction: int) -> None:
        if direction > 0 and self.scroll_y >= self.max_scroll_y:
            self.screen.turn(1)  # type: ignore[attr-defined]
        elif direction < 0 and self.scroll_y <= 0:
            self.screen.turn(-1)  # type: ignore[attr-defined]
        elif direction > 0:
            self.scroll_page_down(animate=False)
        else:
            self.scroll_page_up(animate=False)


def _line(text: str, style: str = "") -> Text:
    """A line of the tree. The stylesheet cuts it short rather than wrap it."""
    return Text(text, style)


class TopicList(OptionList):
    """The topic tree, or the search results while a query is typed.

    A key that moves the highlight tells the dialog before the next key is handled.
    """

    def _walk(self, move: Any) -> None:
        move()
        self.screen.walked()  # type: ignore[attr-defined]

    def action_cursor_up(self) -> None:
        # The list ends at its first and last lines: the highlight doesn't wrap around.
        if self.highlighted:
            self._walk(super().action_cursor_up)

    def action_cursor_down(self) -> None:
        if self.highlighted is None or self.highlighted < self.option_count - 1:
            self._walk(super().action_cursor_down)

    def action_first(self) -> None:
        self._walk(super().action_first)

    def action_last(self) -> None:
        self._walk(super().action_last)

    def action_page_up(self) -> None:
        self._walk(super().action_page_up)

    def action_page_down(self) -> None:
        self._walk(super().action_page_down)


class HelpScreen(ModalScreen[Closed]):
    """The help dialog. Dismissed with the page last shown, and what to insert.

    The items of the page shown, its examples, programs and links in the order they
    appear, can be picked: Tab and Shift-Tab step through them while the page has the
    keys, Enter follows a link or copies the rest, and, with `line`, when help was
    opened from a line that takes an expression, F3 closes help to insert the example
    or program there. With nothing picked, Tab picks the first item not above the view,
    and Shift-Tab returns to the tree, as Tab and Shift-Tab do past the last and the
    first item. The pick is let go when it is scrolled out of view, when the page loses
    the keys, and when another page is shown.
    """

    BINDINGS = [
        Binding("escape", "escape", "Close", show=False),
        Binding("q", "escape", "Close", show=False),
        Binding("slash", "search", "Search", show=False),
        Binding("backspace", "back", "Back", show=False),
        Binding("right", "read", "Read", show=False),
        Binding("left", "topics", "Topics", show=False),
        Binding("tab", "switch(1)", "Pane", priority=True, show=False),
        Binding("shift+tab", "switch(-1)", "Pane", priority=True, show=False),
    ]

    def __init__(
        self,
        document: helps.Document,
        settings: Settings,
        place: helps.Place,
        line: bool = False,
    ) -> None:
        super().__init__()
        self.document = document
        self.line = line
        self.settings = settings
        self.start = place
        #: The page on screen, and the pages shown before it, the latest last.
        self.address = place.address
        self.history: list[str] = []
        #: Whether the tree shows the pages of the topic being read. Up and Down keep
        #: it as it is, so walking on from an open topic opens the next and walking
        #: over closed topics leaves them closed.
        self.expanded = not self._is_intro(place.address)
        #: The search results the list shows, or None while it shows the tree.
        self.results: list[helps.Entry] | None = None
        #: The query the results are for.
        self._query = ""
        #: The place whose text is still to be shown and whether at its end, or None;
        #: and whether text is being loaded.
        self._wanted: tuple[helps.Place, bool] | None = None
        self.fetching = False
        #: The items of the page shown, found when first wanted, and the one picked.
        self._spots: list[Spot] | None = None
        self.picked: Spot | None = None
        #: The timer that ends the footer saying something was copied, while it does.
        self._copied: Timer | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog") as dialog:
            dialog.border_title = "Help"
            with Horizontal(id="help-body"):
                with Vertical(id="help-side"):
                    yield Input(placeholder="/ search", id="help-search", compact=True)
                    yield TopicList(id="help-topics")
                with Vertical(id="help-reader"):
                    yield Static(id="help-crumb")
                    with HelpPage(id="help-page"):
                        yield HelpDocument(id="help-document")
            yield Static(id="help-keys")

    async def on_mount(self) -> None:
        self._layout(self.size.width)
        self.tree.focus()
        if self._point(self.start, record=False):
            self._wanted = (self.start, False)
            await self._load()

    # -- parts -------------------------------------------------------------

    @property
    def tree(self) -> TopicList:
        return self.query_one(TopicList)

    @property
    def search(self) -> Input:
        return self.query_one("#help-search", Input)

    @property
    def page(self) -> HelpPage:
        return self.query_one(HelpPage)

    @property
    def markdown(self) -> HelpDocument:
        return self.query_one(HelpDocument)

    @property
    def narrow(self) -> bool:
        return self.query_one("#help-dialog").has_class("-narrow")

    @property
    def reading(self) -> bool:
        """Whether the page has the keys (and, when narrow, the whole width)."""
        return self.focused is self.page

    # -- layout ------------------------------------------------------------

    def on_resize(self, event: events.Resize) -> None:
        self._layout(event.size.width)

    def _layout(self, width: int) -> None:
        self.query_one("#help-dialog").set_class(width < NARROW, "-narrow")
        self._shown()

    def _shown(self) -> None:
        """Show what has the keys, in the narrow layout, and its key hints."""
        self.query_one("#help-dialog").set_class(self.reading, "-reading")
        focused = self.focused
        if focused is self.search:
            keys = SEARCH_KEYS
        elif self.narrow:
            keys = NARROW_PAGE_KEYS if focused is self.page else NARROW_TREE_KEYS
        else:
            keys = PAGE_KEYS if focused is self.page else TREE_KEYS
        if focused is self.page:
            keys = self._item_keys(keys)
        styles = self.app.palette.styles  # type: ignore[attr-defined]
        hints = Text(" ")
        if self._copied is not None:
            hints.append(COPIED, styles["option"])
        else:
            for key, meaning in keys:
                hints.append(key, styles["option"])
                hints.append(f" {meaning}   ", styles["work"])
        self.query_one("#help-keys", Static).update(hints)

    def _item_keys(
        self, keys: tuple[tuple[str, str], ...]
    ) -> tuple[tuple[str, str], ...]:
        """The page's hints `keys`, with those for its items after the paging hint.

        While an item is picked, its hints take the place of the scrolling ones, so the
        footer still fits the narrow layout.
        """
        items: list[tuple[str, str]] = []
        if self.spots:
            items.append(ITEM_KEY)
        picked = self.picked
        if picked is not None:
            items.append(ACT_KEYS["copy" if picked.copied else "link"])
            if picked.copied and self.line:
                items.append(INSERT_KEY)
        kept = [hint for hint in keys if not (items and hint[0] == "Tab")]
        at = next(
            (index + 1 for index, hint in enumerate(kept) if hint[0] == "PgUp/PgDn"),
            len(kept),
        )
        if picked is not None:
            kept, at = kept[at:], 0
        return (*kept[:at], *items, *kept[at:])

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        self._shown()

    # -- showing a page ----------------------------------------------------

    def go(
        self, place: helps.Place, record: bool = True, at_end: bool = False
    ) -> None:
        """Go to the page `place` names, scrolled to its row or to the top or end.

        The address, the line over the page and the tree change at once, so keys that
        follow act on the new place. The text follows when it is loaded, and only for
        the place reached last: pages passed on the way are not shown. With `record`,
        the page left goes on the history first, so Backspace returns to it.
        """
        if not self._point(place, record):
            return
        self._wanted = (place, at_end)
        if not self.fetching:
            self.fetching = True
            self.run_worker(self._load(), group="help")

    def _point(self, place: helps.Place, record: bool) -> bool:
        page = self.document.page(place.address)
        if page is None:
            return False
        if record and place.address != self.address:
            self.history.append(self.address)
        self.address = place.address
        if not self._is_intro(place.address):
            self.expanded = True
        self.query_one("#help-crumb", Static).update(CRUMB.join(page.crumbs))
        if self.results is None:
            self._fill_tree()
        return True

    async def _load(self) -> None:
        """Show the text of the place wanted, until none is wanted that isn't shown."""
        try:
            while self._wanted is not None:
                (place, at_end), self._wanted = self._wanted, None
                page = self.document.page(place.address)
                assert page is not None
                self.picked, self._spots = None, None
                await self.markdown.update(self.document.text(page, self.settings))
                self._spots = None
                self.page.scroll_home(animate=False, immediate=True)
                self._shown()
                if place.table is not None and place.row is not None:
                    self._mark(place.table, place.row)
                elif at_end:
                    self.call_after_refresh(
                        self.page.scroll_end, animate=False, immediate=True
                    )
        finally:
            self.fetching = False

    def _mark(self, table: int, row: int) -> None:
        """Highlight a table row, and scroll to it once the page is laid out."""
        tables = list(self.markdown.query(HelpTable).results(HelpTable))
        if table < len(tables):
            tables[table].mark(row)
            self.call_after_refresh(self._reveal, tables[table])

    def _reveal(self, table: HelpTable) -> None:
        """Scroll the highlighted row of `table` near the top of the page."""
        if not table.is_attached or table.found is None:
            return
        region = table.row_region(table.found)
        if region is not None:
            top = table.virtual_region.y + region.y
            self.page.scroll_to(y=max(0, top - 1), animate=False, immediate=True)

    @property
    def marked(self) -> list[str]:
        """The lines of the highlighted table row."""
        for table in self.markdown.query(HelpTable).results(HelpTable):
            if table.found is not None and table.found in table.row_lines:
                first, last = table.row_lines[table.found]
                text = str(table.render()).splitlines()
                return [line.rstrip() for line in text[first:last]]
        return []

    def turn(self, direction: int) -> None:
        """Go on to the next page in reading order (back to the previous for -1)."""
        page = self.document.step(self.address, direction)
        if page is not None:
            self.go(helps.Place(page.address), at_end=direction < 0)

    # -- the tree ----------------------------------------------------------

    def _is_intro(self, address: str) -> bool:
        """Whether `address` is a topic's own page rather than one of its pages."""
        return any(topic.intro.address == address for topic in self.document.topics)

    def _fill_tree(self) -> None:
        """List the topics, the one being read opened if `expanded`, and highlight
        its page."""
        shown = self.document.page(self.address)
        assert shown is not None
        options: list[Option] = []
        highlight = 0
        for topic in self.document.topics:
            opened = self.expanded and topic.title == shown.topic
            if topic.intro.address == self.address:
                highlight = len(options)
            mark = (
                SHOWN
                if topic.intro.address == self.address
                else OPEN
                if opened
                else "  "
            )
            options.append(Option(_line(mark + topic.title), id=topic.intro.address))
            if not opened:
                continue
            for page in topic.pages[1:]:
                if page.address == self.address:
                    highlight = len(options)
                mark = SHOWN if page.address == self.address else "  "
                options.append(Option(_line("  " + mark + page.title), id=page.address))
        tree = self.tree
        tree.set_options(options)
        tree.highlighted = highlight

    def _fill_results(self, query: str) -> None:
        self._query = query
        self.results = self.document.search(query)
        options: list[Option] = []
        styles = self.app.palette.styles  # type: ignore[attr-defined]
        for index, entry in enumerate(self.results):
            label = _line("  " + entry.label, styles["work"])
            note = entry.note or KIND_NOTES.get(entry.kind, "")
            if note:
                label.append(f"  {note}", styles["frame"])
            options.append(Option(label, id=str(index)))
        tree = self.tree
        tree.set_options(options)
        tree.highlighted = 0 if options else None

    def walked(self) -> None:
        """Show the page the tree's highlight was moved to.

        The tree calls this as it handles the key, so the tree is refilled, a topic
        reached from an open one opening under the highlight, before the next key is
        looked at.
        """
        tree = self.tree
        if self.results is not None or tree.highlighted is None:
            return
        address = tree.get_option_at_index(tree.highlighted).id
        if address is not None and address != self.address:
            self.go(helps.Place(address))

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        """Enter or a click on the tree opens a closed topic, and otherwise reads the
        page; on a result, goes there."""
        event.stop()
        if self.results is not None:
            self._go_to_result(event.option_index)
            return
        if event.option_id is not None and event.option_id != self.address:
            self.go(helps.Place(event.option_id))
        if not self._expand():
            self.page.focus()

    # -- search ------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        event.stop()
        if event.value.strip():
            self._fill_results(event.value)
        elif self.results is not None:
            self.results, self._query = None, ""
            self._fill_tree()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        # The change that typed the last character may not have arrived yet.
        if event.value.strip() and self._query != event.value:
            self._fill_results(event.value)
        if self.results:
            self._go_to_result(self.tree.highlighted or 0)

    def on_key(self, event: events.Key) -> None:
        """Up and Down in the search box move through the results."""
        if self.focused is self.search and event.key in ("up", "down"):
            event.stop()
            event.prevent_default()
            if event.key == "up":
                self.tree.action_cursor_up()
            else:
                self.tree.action_cursor_down()

    def _go_to_result(self, index: int) -> None:
        assert self.results is not None
        if not 0 <= index < len(self.results):
            return
        entry = self.results[index]
        self._clear_search()
        self.go(entry.place)
        self.page.focus()

    def _clear_search(self) -> None:
        self.results = None
        self._query = ""
        with self.search.prevent(Input.Changed):
            self.search.value = ""
        self._fill_tree()

    # -- items -------------------------------------------------------------

    @property
    def spots(self) -> list[Spot]:
        """The items of the page shown, in the order they appear."""
        if self._spots is not None:
            return self._spots
        spots: list[Spot] = []
        for block in self.markdown.query(MarkdownBlock).results(MarkdownBlock):
            if isinstance(block, PlainFence):
                kind = helps.ITEM_KINDS.get(block.lexer.strip(), "")
                spots += [
                    Spot(kind, snippet.text, block, index)
                    for index, snippet in enumerate(block.snippets)
                ]
            elif isinstance(block, HelpTable):
                spots += [Spot("link", href, block, at) for at, href in block.links()]
            else:
                spots += [
                    Spot("link", href, block, (start, end))
                    for start, end, href in _links(block._content)
                ]
        if not self.fetching:
            self._spots = spots
        return spots

    def _place(self, spot: Spot) -> tuple[int, int]:
        """The lines of the page `spot` takes up, as (first, past the last)."""
        block = spot.block
        top = 0
        widget: Widget | None = block
        while widget is not None and widget is not self.page:
            top += widget.virtual_region.y
            widget = widget.parent  # type: ignore[assignment]
            if isinstance(widget, Widget) and widget is not self.page:
                top += widget.styles.border.spacing.top
        if isinstance(block, PlainFence):
            first, last = block.snippets[spot.at].lines
        elif isinstance(block, HelpTable):
            first, last = block.row_lines.get(spot.at[0], (0, 1))
        else:
            first = _line_of(block._content, spot.at[0], block.content_size.width)
            last = first + 1
        return top + first, top + last

    def _view(self) -> tuple[int, int]:
        """The lines of the page in view, as (first, past the last)."""
        top = round(self.page.scroll_y)
        return top, top + self.page.scrollable_content_region.height

    def _show(self, spot: Spot, picked: bool) -> None:
        """Draw `spot` picked, or not."""
        block = spot.block
        if isinstance(block, PlainFence):
            block.pick(spot.at if picked else None)
        elif isinstance(block, HelpTable):
            block.pick(spot.at if picked else None)
        else:
            content = block._content
            block.update(content.stylize(PICKED_LINK, *spot.at) if picked else content)

    def pick(self, spot: Spot | None) -> None:
        """Pick `spot`, or nothing, and scroll it into view."""
        if self.picked is not None and self.picked.block.is_attached:
            self._show(self.picked, False)
        self.picked = spot
        if spot is not None:
            # A row found by search gives way to the pick.
            for table in self.markdown.query(HelpTable).results(HelpTable):
                if table.found is not None:
                    table.mark(None)
            self._show(spot, True)
            first, last = self._place(spot)
            self.page.scroll_to_region(
                Region(0, first, 1, last - first), animate=False, immediate=True
            )
        self._shown()

    def unpick(self) -> None:
        if self.picked is not None:
            self.pick(None)

    def scrolled(self) -> None:
        """Let go of the pick once it is out of view, and of the snippet that was under
        the pointer, which the text has moved from under."""
        for fence in self.markdown.query(PlainFence).results(PlainFence):
            fence.point(None)
        if self.picked is None or not self.picked.block.is_attached:
            return
        first, last = self._place(self.picked)
        top, bottom = self._view()
        if last <= top or first >= bottom:
            self.unpick()

    def _step(self, direction: int) -> None:
        """Pick the next item (the previous, for -1), or leave the page past the end."""
        spots = self.spots
        if self.picked in spots:
            index = spots.index(self.picked) + direction  # type: ignore[arg-type]
        elif direction > 0:
            top = self._view()[0]
            index = next(
                (at for at, spot in enumerate(spots) if self._place(spot)[0] >= top),
                len(spots),
            )
        else:
            index = -1
        if 0 <= index < len(spots):
            self.pick(spots[index])
        else:
            self.unpick()
            self._leave(direction)

    def _leave(self, direction: int) -> None:
        """Leave the page for the tree, or for the search box backwards from results."""
        back = direction < 0 and self.results is not None
        (self.search if back else self.tree).focus()

    def action_act(self) -> None:
        """Enter on the page: follow the link picked, or copy the item."""
        spot = self.picked
        if spot is None:
            return
        if spot.copied:
            self.copy(spot.text)
        else:
            self._follow(spot.text)

    def action_insert(self) -> None:
        """F3 on the page: close help, to insert the item picked on the line."""
        spot = self.picked
        if spot is not None and spot.copied and self.line:
            self.dismiss(Closed(self.address, spot.text))

    def clicked(self, fence: PlainFence, index: int) -> None:
        """A click on a snippet: pick it and copy it."""
        spot = next(
            (
                spot
                for spot in self.spots
                if spot.block is fence and spot.at == index
            ),
            None,
        )
        if spot is None:
            return
        self.page.focus()
        self.pick(spot)
        self.copy(spot.text)

    def copy(self, text: str) -> None:
        """Copy `text` as Ctrl-C does, and say so in the footer for a while."""
        self.app.copy_to_clipboard(text)
        if self._copied is not None:
            self._copied.stop()
        self._copied = self.set_timer(COPIED_SECONDS, self._copy_said)
        self._shown()

    def _copy_said(self) -> None:
        self._copied = None
        if self.is_attached:
            self._shown()

    # -- links -------------------------------------------------------------

    def on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        event.stop()
        self._follow(event.href)

    def _follow(self, href: str) -> None:
        place = self.document.resolve(href)
        if place is not None:
            self.go(place)

    # -- keys --------------------------------------------------------------

    def action_search(self) -> None:
        self.search.focus()

    def action_back(self) -> None:
        if self.history:
            self.go(helps.Place(self.history.pop()), record=False)

    def _expand(self) -> bool:
        """Open the topic being read in the tree; whether it was closed."""
        if self.expanded:
            return False
        self.expanded = True
        self._fill_tree()
        return True

    def action_read(self) -> None:
        """Right on the tree: open the topic if it is closed, and otherwise read the
        page."""
        if self.focused is self.tree:
            if self.results is not None:
                self.tree.action_select()
            elif not self._expand():
                self.page.focus()

    def action_topics(self) -> None:
        """Left: from the page to the tree; on the tree, close the open topic."""
        if self.focused is self.page:
            self.tree.focus()
        elif self.focused is self.tree and self.results is None and self.expanded:
            self.expanded = False
            page = self.document.page(self.address)
            assert page is not None
            topic = next(t for t in self.document.topics if t.title == page.topic)
            if topic.intro.address == self.address:
                self._fill_tree()
            else:
                self.go(helps.Place(topic.intro.address))

    def action_switch(self, direction: int) -> None:
        """Tab: from the search box or the tree to the page, and from the page back."""
        if self.focused is self.page:
            self._step(direction)
        elif self.focused is self.search:
            (self.tree if direction > 0 and self.results else self.page).focus()
        else:
            self.page.focus()

    def action_escape(self) -> None:
        """Clear the search, return from the page in the narrow layout, or close."""
        if self.focused is self.search or self.results is not None:
            if self.search.value or self.results is not None:
                self._clear_search()
            self.tree.focus()
        elif self.focused is self.page and self.narrow:
            self.tree.focus()
        else:
            self.dismiss(Closed(self.address))
