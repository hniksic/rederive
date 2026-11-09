"""The help document, and how a pageful of it is laid out.

Derive's help was a plain full-screen reference drawn into the work area of the
current window, with two levels: a menu of subjects, and a subject read a page at
a time. There were no links, index or search. We keep that shape, minus the
references to the paper manual and the subject on utility files (we ship none).
The original's two plot subjects are merged into one, since our plot windows are
mouse-driven with no menus to document, and what an expression plots as is the
same question in 2D and 3D.

Nothing here draws. A topic is a sequence of blocks, each a group of lines kept on
one page. Layout depends on the window size, so pages are computed at paint time
and never stored, and a resized or split pane re-paginates, as in the original.

The settings subject is generated from the same dialogs `Options` and `Manage`
use, so it can't get out of sync. (Derive used `*SETTING*` placeholders, which
silently broke when a setting was renamed.)
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from importlib.resources import files

from rederive.model.settings import DIALOGS, Settings

#: The Help Menu's title, bottom instruction, and final item, which returns to the
#: program.
MENU_TITLE = "Rederive Help Menu"
INSTRUCTION = "Press letter for desired subject"
RESUME = "Resume"
RETURN = "Return to Rederive"

#: Menu options of a topic page besides `Resume`.
NEXT = "Next"
PREVIOUS = "Previous"

#: The subject F1 opens while a line is being edited, and the option that leads
#: from it to the subject menu.
EDITING = "Editing"
SUBJECTS = "Subjects"

#: Rows above a topic page's text: the title and a blank line. The Help Menu
#: includes its title in its body instead.
TITLE_ROWS = 2

#: Below this width, the settings dump uses one column instead of two, to avoid
#: wrapped fragments in narrow panes.
NARROW = 60

#: Source file directives: start of a topic, and placeholder for the settings dump.
_TOPIC = "%topic"
_SETTINGS = "%settings"


@dataclass(frozen=True)
class Topic:
    """One subject of the help document.

    `word` is the option on the `HELP:` menu line (and provides the mnemonic
    letter), `entry` is the line in the Help Menu, and `title` heads each page. As
    in the original, these differ: the menu shows `line Editing commands` for the
    option `Editing`.
    """

    word: str
    entry: str
    title: str
    blocks: tuple[tuple[str, ...], ...] = ()
    settings: bool = False
    """Whether the body is the live settings dump rather than written text."""


def letter(entry: str) -> str:
    """The mnemonic letter of `entry`: its first capital, as in menus."""
    return next((character for character in entry if character.isupper()), entry[:1])


def entries() -> list[str]:
    """The Help Menu's lines: every subject, then the return item."""
    lines = [f"{letter(topic.entry)} - {topic.entry}" for topic in TOPICS]
    return lines + [f"{letter(RETURN)} - {RETURN}"]


def menu_page(rows: int, width: int) -> list[str]:
    """The Help Menu as a page: title, items, and instruction.

    The items are centred as one block so their letters line up. A short pane
    drops the blank lines first, then the instruction.
    """
    items = entries()
    left = " " * max(0, (width - max(len(item) for item in items)) // 2)
    block = [left + item for item in items]
    title, closing = _centred(MENU_TITLE, width), _centred(INSTRUCTION, width)
    for page in (
        [title, "", *block, "", closing],
        [title, *block, closing],
        [title, *block],
    ):
        if len(page) <= rows:
            return page
    return block[:rows]


def pages(topic: Topic, settings: Settings, rows: int, width: int) -> list[list[str]]:
    """`topic` laid out as pages for a pane `rows` tall and `width` wide.

    Each page includes the title and blank row, ready to display.
    """
    body = max(1, rows - TITLE_ROWS)
    if topic.settings:
        blocks = _settings_blocks(settings)
        laid = (
            _paginate(blocks, body)
            if width < NARROW
            else _columns(blocks, body, width)
        )
    else:
        laid = _paginate([_wrapped(block, width) for block in topic.blocks], body)
    title = _centred(topic.title, width)
    return [[title, "", *page] for page in laid]


def _centred(text: str, width: int) -> str:
    return " " * max(0, (width - len(text)) // 2) + text


def _wrapped(block: tuple[str, ...], width: int) -> list[str]:
    """A block's source lines, each wrapped to `width`.

    Continuation lines keep the original line's indent, so wrapped table rows stay
    in their column. (Derive wrapped to the margin, even mid-word.)
    """
    return [row for line in block for row in _wrap(line, width)]


def _wrap(line: str, width: int) -> list[str]:
    if len(line) <= width:
        return [line]
    indent = " " * (len(line) - len(line.lstrip()))
    return textwrap.wrap(
        line,
        width,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [""]


def _paginate(blocks: list[list[str]], rows: int) -> list[list[str]]:
    """Lay out whole blocks on pages, separated by blank lines.

    A block that doesn't fit on the current page starts a new one; a block taller
    than a page is split across pages.
    """
    pages: list[list[str]] = []
    page: list[str] = []
    for block in blocks:
        if page and len(page) + 1 + len(block) > rows:
            pages.append(page)
            page = []
        if page:
            page.append("")
        page.extend(block)
        while len(page) > rows:
            pages.append(page[:rows])
            page = page[rows:]
    if page:
        pages.append(page)
    return pages or [[]]


def _columns(blocks: list[list[str]], rows: int, width: int) -> list[list[str]]:
    """Lay out blocks in two columns per page, left column first.

    The original does this for the settings page, since a single column of short
    `label: value` lines wastes most of the width.
    """
    across = max(1, width // 2)
    columns: list[list[str]] = [[]]
    for block in blocks:
        if columns[-1] and len(columns[-1]) + 1 + len(block) > rows:
            columns.append([])
        if columns[-1]:
            columns[-1].append("")
        columns[-1].extend(block[:rows])
    pages: list[list[str]] = []
    for index in range(0, len(columns), 2):
        left = columns[index]
        right = columns[index + 1] if index + 1 < len(columns) else []
        pages.append(
            [
                (_at(left, row).ljust(across) + _at(right, row)).rstrip()
                for row in range(max(len(left), len(right)))
            ]
        )
    return pages or [[]]


def _at(lines: list[str], row: int) -> str:
    return lines[row] if row < len(lines) else ""


def _settings_blocks(settings: Settings) -> list[list[str]]:
    """The settings dump: one block per screen, its name then its fields.

    Screens come from `DIALOGS`, so new ones appear automatically.
    """
    blocks: list[list[str]] = []
    for dialog in DIALOGS:
        name = _screen_name(dialog.title)
        rows = [name]
        for field in dialog.fields:
            # The only uncaptioned field is alone on its screen, so label it with
            # the screen name's last word.
            label = field.label or name.rpartition(" ")[2]
            rows.append(f"  {label}:  {settings[field.setting]}")
        blocks.append(rows)
    return blocks


def _screen_name(title: str) -> str:
    """Turn a title like `OPTIONS COLOR MENU:` into `Options Color Menu`."""
    return title.rstrip(":").title()


def _read(source: str) -> tuple[Topic, ...]:
    """Parse the document: `%topic` lines, and blank-line-separated blocks."""
    topics: list[Topic] = []
    heads: list[tuple[str, str, str]] = []
    bodies: list[list[list[str]]] = []
    for line in source.splitlines():
        if line.startswith(_TOPIC):
            heads.append(_head(line))
            bodies.append([])
            continue
        if not heads:
            continue
        if line.strip():
            if not bodies[-1] or bodies[-1][-1] == []:
                bodies[-1].append([])
            bodies[-1][-1].append(line.rstrip())
        elif bodies[-1] and bodies[-1][-1] != []:
            bodies[-1].append([])
    for (word, entry, title), body in zip(heads, bodies):
        blocks = [tuple(block) for block in body if block]
        generated = blocks == [(_SETTINGS,)]
        topics.append(
            Topic(
                word,
                entry,
                title,
                () if generated else tuple(blocks),
                settings=generated,
            )
        )
    return tuple(topics)


def _head(line: str) -> tuple[str, str, str]:
    """`%topic Editing | line Editing commands | Line Editing Commands`."""
    word, entry, title = (part.strip() for part in line[len(_TOPIC) :].split("|"))
    return word, entry, title


def _source() -> str:
    return (files("rederive.model") / "help.txt").read_text(encoding="utf-8")


#: All subjects, in Help Menu order.
TOPICS: tuple[Topic, ...] = _read(_source())

#: Subjects by their menu option word.
BY_WORD: dict[str, Topic] = {topic.word: topic for topic in TOPICS}
