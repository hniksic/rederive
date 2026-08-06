"""The help document, and how a pageful of it is laid out.

Derive's help was a full-screen reference painted into the work area of the
window it was asked from - no border, no title bar, no scrollbar, nothing
overlaid and nothing dimmed underneath. It had exactly two levels: a menu of
subjects, and a subject read a page at a time. There were no links, no index
and no search; the section numbers printed through the text pointed at the
paper manual, and following one meant picking the manual up. That shape is
kept here, less the parts that were about paper, less the two plot subjects -
whose menus and framing dialogs are not this program's, its plot windows being
windows of the desktop that are driven with a mouse - and less the subject
listing the utility files, this program shipping none of its own.

Nothing in this module draws anything. A topic is a run of blocks, a block is
a run of source lines that belong on one page together, and laying one out is
a question about the width and the height of the window it is going into -
which is why pages are worked out at paint time and never stored. A pane that
is resized, or split, re-wraps and re-paginates on the next paint, exactly as
the original's did.

The one subject that is generated rather than written is the one saying what
the system is set to. It is read off the same dialogs `Options` and `Manage`
put on the band, so it cannot fall out of step with them. Derive wrote that
page out too, with `*SETTING*` placeholders its viewer substituted, which
means a renamed setting there printed a placeholder and nobody noticed.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from importlib.resources import files

from rederive.model.settings import DIALOGS, Settings

#: What the Help Menu is titled, what it says at the bottom, and the last of
#: its items - the one that is not a subject and leads back to the program.
MENU_TITLE = "Rederive Help Menu"
INSTRUCTION = "Press letter for desired subject"
RESUME = "Resume"
RETURN = "Return to Rederive"

#: What the menu line of a topic page offers besides `Resume`.
NEXT = "Next"
PREVIOUS = "Previous"

#: The two rows a topic page spends before its text: the title, and the blank
#: line under it. The Help Menu carries its own title inside its body, being a
#: centred document rather than a titled page.
TITLE_ROWS = 2

#: Below this many columns the settings dump is laid out in one column rather
#: than two. Two columns of a narrow split pane would be two columns of
#: wrapped fragments, which is worse than a page more of readable lines.
NARROW = 60

#: Directives in the source file: one starts a topic, the other stands in for
#: the settings dump.
_TOPIC = "%topic"
_SETTINGS = "%settings"


@dataclass(frozen=True)
class Topic:
    """One subject of the help document.

    `word` is what the `HELP:` menu line carries and so what the mnemonic
    letter is taken from, `entry` is the line the Help Menu shows, and `title`
    heads every page of the subject. The two are not the same: the menu reads
    `line Editing commands` where the option reads `Editing`, which is the
    original's own arrangement.
    """

    word: str
    entry: str
    title: str
    blocks: tuple[tuple[str, ...], ...] = ()
    settings: bool = False
    """Whether the body is the live settings dump rather than written text."""


def letter(entry: str) -> str:
    """The capital that invokes `entry`, which is the rule menu words follow."""
    return next((character for character in entry if character.isupper()), entry[:1])


def entries() -> list[str]:
    """The lines the Help Menu lists, the way it lists them.

    Every subject, and then the item that is no subject: the one that goes
    back to what the program was doing.
    """
    lines = [f"{letter(topic.entry)} - {topic.entry}" for topic in TOPICS]
    return lines + [f"{letter(RETURN)} - {RETURN}"]


def menu_page(rows: int, width: int) -> list[str]:
    """The Help Menu as a page: a title, the items, and how to choose one.

    The items are one block, centred as a block on the longest of them rather
    than each on its own, so their letters line up down the left. A pane too
    short for the whole thing gives up the blank lines first and the closing
    instruction next: the items are what the page is for.
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
    """`topic` laid out for a pane `rows` tall and `width` wide, a page a list.

    Every page carries the title row and the blank line under it, so what
    comes back is what the pane shows and the caller has nothing left to add.
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

    A line that has to be broken keeps its own indent on the lines it breaks
    onto, so a wrapped table row stays under its own column rather than
    starting again at the margin. Derive broke to the margin, and mid-word at
    that; neither is worth imitating.
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
    """Whole blocks to a page, with a blank line between two on the same page.

    A block taller than a whole page is cut across pages, since there is no
    other way to show it; a block that would only overhang the page it is on
    starts the next one instead.
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
    """Two columns to a page: fill the left one downwards, then the right one.

    Which is how the original lays its settings page out, and it is worth the
    trouble - a column of short `label: value` lines wastes most of eighty.
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

    Every screen the program can put up is listed, in the order the settings
    module holds them, so a screen added there appears here without anything
    being written about it.
    """
    blocks: list[list[str]] = []
    for dialog in DIALOGS:
        name = _screen_name(dialog.title)
        rows = [name]
        for field in dialog.fields:
            # The one field with no caption of its own is the one whose screen
            # has nothing else on it, so the screen's own last word names it.
            label = field.label or name.rpartition(" ")[2]
            rows.append(f"  {label}:  {settings[field.setting]}")
        blocks.append(rows)
    return blocks


def _screen_name(title: str) -> str:
    """`OPTIONS COLOR MENU:` as the command path that reaches it."""
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


#: Every subject, in the order the Help Menu lists them.
TOPICS: tuple[Topic, ...] = _read(_source())

#: Each subject by the option that names it, for the menu that chooses one.
BY_WORD: dict[str, Topic] = {topic.word: topic for topic in TOPICS}
