"""The help document: topics of pages, and the index that links and search land on.

The help is a set of Markdown files in `topics`, one per topic, read in `ORDER`. A file
is `# Topic name`, a few intro lines, and a `## Sub-topic name` section per page. A
page's address is `help:<file stem>/<slug>`, and `help:<file stem>` is the topic's intro
page, which is generated: the intro lines, then a link to each of the topic's pages.

Tables carry the index, by their first header cell:

- A row whose first cell holds code spans is the home of the names they start with:
  `DIF` for `` `DIF(u, x, n)` ``. A `name:DIF` link lands on that row. Symbol
  operators start with no name, so they are not indexed.
- A `Command` table lists commands by their menu path, as in `Calculus Integrate`, and
  its rows are the command index.
- A `Key` table lists keys, which are not indexed.
- The `Area` table, on the utility library page, names an area of the library in the
  first cell and some of its functions as code spans in the second.

Search also finds the old Derive names the reader still accepts, which land on the
home row of the name that replaced them, and every name the pre-defined utility
library (`rederive.library`) defines. A library name the `Area` table lists lands on
its row. One it doesn't list lands on the row that lists the names of its group (the
utility file that defined it, `library.GROUPS`), if they are all on one row, and
otherwise on the utility library page.

A code block's info string says whether it holds things to copy. In an `examples`
block each line is an example: what is copied is the expression, the line up to its
first run of two or more spaces, and the rest is a remark on it. A line that starts
with a blank goes on with the remark of the line before. In a `program` block each
program, the lines between blank lines, is copied whole. A block without an info
string is only read.

The settings page's `%settings` line stands for the settings dump, generated from the
dialogs `Options` and `Manage` use when the page is shown, so it reads the settings as
they stand and can't get out of sync with the screens.

Nothing here draws; `rederive.ui.help` shows the pages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from importlib.resources.abc import Traversable
from urllib.parse import unquote

from markdown_it import MarkdownIt
from markdown_it.token import Token

from rederive import library
from rederive.model.settings import DIALOGS, Settings
from rederive.syntax import names

#: The topic files, by stem, in reading order: the order of the topic tree, and the
#: order paging runs through the pages.
ORDER = (
    "basics",
    "entering",
    "commands",
    "settings",
    "plotting",
    "functions",
    "programming",
)

#: The address scheme of a page link, and of a link to a name's home row.
PAGE_SCHEME = "help:"
NAME_SCHEME = "name:"

#: Where help opens the first time, and the page F1 on an input line opens.
START = PAGE_SCHEME + ORDER[0]
LINE_EDITING = PAGE_SCHEME + "entering/line-editing"

#: The placeholder the settings page holds for the generated dump.
SETTINGS = "%settings"

#: The first header cell of a command table, of a key table, and of the table of the
#: utility library's areas.
COMMAND_HEADER = "Command"
KEY_HEADER = "Key"
AREA_HEADER = "Area"

#: The slug of the page that describes the utility library's groups.
LIBRARY_PAGE = "utility-library"

#: Old names the reader accepts, each with the name that replaced it.
ALIASES: dict[str, str] = {**names.REWRITTEN_ALIASES, **names.KEPT_ALIASES}

#: What heads the list of a topic's pages on its intro page.
CONTENTS = "**Pages in this topic**"

#: The most results a search returns.
RESULTS = 100

#: The leading name of a code span: letters, digits, `_`, `?` and `#`.
_NAME = re.compile(r"[A-Za-z0-9_?#]+")

#: The info strings of the code blocks that hold things to copy.
EXAMPLES = "examples"
PROGRAM = "program"

#: What separates an example's expression from its remark.
_REMARK = re.compile(r" {2,}")


def parser() -> MarkdownIt:
    """The Markdown dialect of the help: CommonMark with tables.

    The reader shows pages with the same parser, so tables are counted the same way in
    the index and on screen.
    """
    return MarkdownIt("commonmark").enable("table")


def slug(title: str) -> str:
    """A page's slug: `Vectors and matrices` is `vectors-and-matrices`."""
    kept = "".join(
        character
        for character in title.lower()
        if character.isalnum() or character in " -"
    )
    return "-".join(kept.split())


@dataclass(frozen=True)
class Place:
    """Where a link or a search result lands: a page, and a table row on it or None.

    `table` counts the tables of the page from 0, and `row` the body rows of the table
    from 1, as the reader's table widget numbers them.
    """

    address: str
    table: int | None = None
    row: int | None = None


@dataclass(frozen=True)
class Page:
    """One page: a topic's intro, or one of its sub-topics."""

    address: str
    topic: str
    title: str
    """The sub-topic's name, or the empty string for the topic's intro page."""
    source: str
    """The Markdown body, without the heading."""

    @property
    def crumbs(self) -> tuple[str, ...]:
        """The topic, then the sub-topic if this isn't an intro page."""
        return (self.topic, self.title) if self.title else (self.topic,)


@dataclass(frozen=True)
class Topic:
    stem: str
    title: str
    pages: tuple[Page, ...]
    """The intro page, then a page per sub-topic."""

    @property
    def intro(self) -> Page:
        return self.pages[0]


@dataclass(frozen=True)
class Entry:
    """One thing search finds: what it is called, what kind it is, and where it lands.

    `note` is shown after the label, as the current name after an old one.
    """

    label: str
    kind: str
    place: Place
    note: str = ""


#: Entry kinds, in the order results of equal match are listed.
KINDS = ("topic", "page", "command", "name", "alias", "library")


@dataclass(frozen=True)
class Row:
    """A table row, as the index reads it: where it is, and what its cells hold."""

    place: Place
    header: str
    """The table's first header cell."""
    texts: tuple[str, ...]
    """Each cell's plain text."""
    cells: tuple[tuple[str, ...], ...]
    """The code spans in each cell."""

    @property
    def text(self) -> str:
        """The first cell's plain text."""
        return self.texts[0] if self.texts else ""

    @property
    def spans(self) -> tuple[str, ...]:
        """The code spans in the first cell."""
        return self.cells[0] if self.cells else ()


class Document:
    """The help, read from the topic files."""

    def __init__(self, topics: tuple[Topic, ...]) -> None:
        self.topics = topics
        #: Every page in reading order.
        self.pages: tuple[Page, ...] = tuple(
            page for topic in topics for page in topic.pages
        )
        self.by_address: dict[str, Page] = {page.address: page for page in self.pages}
        self.rows: tuple[Row, ...] = tuple(
            row for page in self.pages for row in _rows(page)
        )
        #: Each name's home row, by the name case-folded.
        self.names: dict[str, Place] = {}
        #: Each command's row, by its menu path.
        self.commands: dict[str, Place] = {}
        #: The `Area` row of each library name it lists, by the name case-folded.
        self.areas: dict[str, Place] = {}
        homes: dict[str, str] = {}
        for row in self.rows:
            if row.header == COMMAND_HEADER:
                self.commands.setdefault(row.text, row.place)
            elif row.header == AREA_HEADER:
                for span in row.cells[1] if len(row.cells) > 1 else ():
                    name = _NAME.match(span.strip())
                    if name is not None:
                        self.areas.setdefault(name.group().casefold(), row.place)
            elif row.header != KEY_HEADER:
                for name in row_names(row):
                    homes.setdefault(name.casefold(), name)
                    self.names.setdefault(name.casefold(), row.place)
        self.entries = self._entries(homes)

    def _entries(self, homes: dict[str, str]) -> tuple[Entry, ...]:
        entries: list[Entry] = []
        for topic in self.topics:
            entries.append(Entry(topic.title, "topic", Place(topic.intro.address)))
            for page in topic.pages[1:]:
                entries.append(
                    Entry(page.title, "page", Place(page.address), topic.title)
                )
        for command, place in self.commands.items():
            entries.append(Entry(command, "command", place))
        for folded, name in homes.items():
            entries.append(Entry(name, "name", self.names[folded]))
        for alias, current in sorted(ALIASES.items()):
            place = self.names.get(current.casefold())
            if alias.casefold() not in self.names and place is not None:
                entries.append(Entry(alias, "alias", place, current))
        groups = self.group_places()
        for name in sorted(library.NAMES):
            if name.casefold() in self.names:
                continue
            place = self.areas.get(name.casefold()) or groups.get(library.GROUPS[name])
            if place is not None:
                entries.append(Entry(name, "library", place))
        return tuple(entries)

    def group_places(self) -> dict[str, Place]:
        """Where the library names of each group land that the `Area` table doesn't
        list: the one row that lists all its listed names, or else the utility library
        page."""
        page = self.library_page
        if page is None:
            return {}
        rows: dict[str, set[Place]] = {}
        for name in library.NAMES:
            place = self.areas.get(name.casefold())
            if place is not None:
                rows.setdefault(library.GROUPS[name], set()).add(place)
        places: dict[str, Place] = {}
        for group in set(library.GROUPS.values()):
            found = rows.get(group, set())
            places[group] = found.pop() if len(found) == 1 else Place(page.address)
        return places

    @property
    def library_page(self) -> Page | None:
        """The page describing the utility library, if there is one."""
        return next(
            (page for page in self.pages if page.address.endswith("/" + LIBRARY_PAGE)),
            None,
        )

    def page(self, address: str) -> Page | None:
        return self.by_address.get(address)

    def step(self, address: str, direction: int) -> Page | None:
        """The page after `address` in reading order (before, for -1), or None."""
        page = self.by_address[address]
        index = self.pages.index(page) + direction
        return self.pages[index] if 0 <= index < len(self.pages) else None

    def resolve(self, href: str) -> Place | None:
        """Where a `help:` or `name:` link lands, or None if it lands nowhere."""
        if href.startswith(PAGE_SCHEME):
            return Place(href) if href in self.by_address else None
        if href.startswith(NAME_SCHEME):
            name = unquote(href[len(NAME_SCHEME) :]).casefold()
            if name in self.names:
                return self.names[name]
            current = ALIASES.get(name.upper())
            return self.names.get(current.casefold()) if current else None
        return None

    def search(self, query: str) -> list[Entry]:
        """The entries whose labels hold `query`, ignoring case.

        A label equal to the query comes first, then those starting with it, then those
        holding it elsewhere; within each, by kind and then alphabetically.
        """
        wanted = query.strip().casefold()
        if not wanted:
            return []
        found: list[tuple[int, int, str, Entry]] = []
        for entry in self.entries:
            label = entry.label.casefold()
            at = label.find(wanted)
            if at < 0:
                continue
            rank = 0 if label == wanted else 1 if at == 0 else 2
            found.append((rank, KINDS.index(entry.kind), label, entry))
        found.sort(key=lambda item: item[:3])
        return [entry for *_, entry in found[:RESULTS]]

    def text(self, page: Page, settings: Settings) -> str:
        """The Markdown the reader shows for `page`, the settings dump filled in."""
        if SETTINGS not in page.source:
            return page.source
        return page.source.replace(SETTINGS, settings_tables(settings))


def row_names(row: Row) -> list[str]:
    """The names a row is the home of: the leading name of each first-cell code span."""
    found: list[str] = []
    for span in row.spans:
        name = _NAME.match(span.strip())
        if name is not None and name.group() not in found:
            found.append(name.group())
    return found


def _rows(page: Page) -> list[Row]:
    """The body rows of each table on `page`, with the text and code spans of each
    cell."""
    rows: list[Row] = []
    table = -1
    header = ""
    number = 0
    in_head = False
    texts: list[str] = []
    cells: list[tuple[str, ...]] = []
    for token in parser().parse(page.source):
        kind = token.type
        if kind == "table_open":
            table += 1
            header, number = "", 0
        elif kind == "thead_open":
            in_head = True
        elif kind == "thead_close":
            in_head = False
        elif kind == "tr_open":
            texts, cells = [], []
            if not in_head:
                number += 1
        elif kind == "tr_close" and not in_head:
            place = Place(page.address, table, number)
            rows.append(Row(place, header, tuple(texts), tuple(cells)))
        elif kind == "inline" and table >= 0 and (in_head or number):
            if in_head:
                if not header:
                    header = _plain(token) or " "
                continue
            texts.append(_plain(token))
            cells.append(
                tuple(
                    child.content
                    for child in token.children or ()
                    if child.type == "code_inline"
                )
            )
        elif kind == "table_close":
            number = 0
    return rows


def _plain(token: Token) -> str:
    """The text of an inline token, code spans included, without markup."""
    return "".join(
        child.content
        for child in token.children or ()
        if child.type in ("text", "code_inline")
    ).strip()


def links(source: str) -> list[str]:
    """The targets of the links in a Markdown text."""
    found: list[str] = []
    for token in parser().parse(source):
        for child in token.children or ():
            if child.type == "link_open":
                found.append(str(child.attrs.get("href", "")))
    return found


@dataclass(frozen=True)
class Snippet:
    """Something to copy from a code block: the text, and the lines of the block it
    takes up, as (first, past the last), counted from 0."""

    text: str
    lines: tuple[int, int]


def snippets(info: str, code: str) -> list[Snippet]:
    """What there is to copy in a code block with the info string `info`."""
    kind = info.strip()
    rows = code.split("\n")
    while rows and not rows[-1].strip():
        rows.pop()
    found: list[Snippet] = []
    if kind == EXAMPLES:
        for number, row in enumerate(rows):
            if not row.strip():
                continue
            if row[0].isspace():
                if found and found[-1].lines[1] == number:
                    found[-1] = Snippet(found[-1].text, (found[-1].lines[0], number + 1))
                continue
            found.append(Snippet(_REMARK.split(row.rstrip(), 1)[0], (number, number + 1)))
    elif kind == PROGRAM:
        first: int | None = None
        for number, row in enumerate([*rows, ""]):
            if row.strip() and first is None:
                first = number
            elif not row.strip() and first is not None:
                text = "\n".join(line.rstrip() for line in rows[first:number])
                found.append(Snippet(text, (first, number)))
                first = None
    return found


@dataclass(frozen=True)
class Item:
    """Something on a page that can be picked: an `example` or a `program` to copy, or
    a `link`. `text` is what is copied, or where the link leads."""

    kind: str
    text: str


#: The kind of the items of each code block that holds things to copy.
ITEM_KINDS = {EXAMPLES: "example", PROGRAM: "program"}


def items(source: str) -> list[Item]:
    """The items of a Markdown text, in the order they appear."""
    found: list[Item] = []
    for token in parser().parse(source):
        if token.type == "fence":
            kind = ITEM_KINDS.get(token.info.strip())
            if kind is not None:
                found += [Item(kind, s.text) for s in snippets(token.info, token.content)]
        for child in token.children or ():
            if child.type == "link_open":
                found.append(Item("link", str(child.attrs.get("href", ""))))
    return found


def settings_tables(settings: Settings) -> str:
    """The settings dump: a table per settings screen, in the order `DIALOGS` has.

    The first header cell names the screen. Each row gives a setting's caption, the
    name an assignment such as `Precision := Approximate` sets it by, and its value.
    A setting no assignment reaches, such as a color, has no name.
    """
    tables: list[str] = []
    for dialog in DIALOGS:
        screen = _screen_name(dialog.title)
        lines = [f"| {screen} | Name | Value |", "|---|---|---|"]
        for field in dialog.fields:
            # The only uncaptioned field is alone on its screen, so label it with the
            # screen name's last word.
            label = field.label or screen.rpartition(" ")[2]
            name = f"`{field.setting}`" if field.setting in names.SETTINGS else ""
            value = _escaped(str(settings[field.setting]))
            lines.append(f"| {label} | {name} | {value} |")
        tables.append("\n".join(lines))
    return "\n\n".join(tables)


def _escaped(text: str) -> str:
    return re.sub(r"([\\`*_\[\]|<>])", r"\\\1", text)


def _screen_name(title: str) -> str:
    """Turn a title like `OPTIONS COLOR MENU:` into `Options Color Menu`."""
    return title.rstrip(":").title()


def read_topic(stem: str, source: str) -> Topic:
    """Read one topic file: its title, intro lines, and sub-topic sections."""
    title = ""
    intro: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    fenced = False
    for line in source.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        if not fenced and line.startswith("# ") and not title:
            title = line[2:].strip()
        elif not fenced and line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif sections:
            sections[-1][1].append(line)
        else:
            intro.append(line)
    address = PAGE_SCHEME + stem
    contents = "\n".join(
        f"- [{name}]({address}/{slug(name)})" for name, _ in sections
    )
    lead = "\n".join(intro).strip()
    pages = [Page(address, title, "", f"{lead}\n\n{CONTENTS}\n\n{contents}".strip())]
    for name, body in sections:
        pages.append(
            Page(f"{address}/{slug(name)}", title, name, "\n".join(body).strip())
        )
    return Topic(stem, title, tuple(pages))


def read(directory: Traversable, order: tuple[str, ...] = ORDER) -> Document:
    """Read the topic files `order` names from `directory`."""
    return Document(
        tuple(
            read_topic(stem, (directory / f"{stem}.md").read_text(encoding="utf-8"))
            for stem in order
        )
    )


@cache
def document() -> Document:
    """The help as shipped, read on first use."""
    return read(files("rederive.model") / "topics")
