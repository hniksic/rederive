"""The Algebra session: the numbered history and what is selected in it.

Pure Python - no Textual, no rendering, no key names. The UI layer translates
key presses into calls on `Session` and renders the result. This is the seed of
the session layer that will later sit between the TUI and the math engine.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rederive.model.tree import Node

Span = tuple[int, int]
Path = tuple[int, ...]


def _default_parse(text: str) -> Node:
    """Shape an authored line with the placeholder parser.

    Imported at call time: the parser is scaffolding that depends on the model,
    not the other way round, and a caller can pass its own parse function.
    """
    # TODO(display): parse with rederive.syntax.parse_expression instead. The
    # session has to own one ParseState for its whole life and apply each
    # accepted expression's declarations to it, because InputMode, CaseMode and
    # every definition change how later lines lex.
    from rederive.placeholder_parser import parse

    return parse(text)


@dataclass(frozen=True)
class Entry:
    """One numbered line of the history."""

    # TODO(display): an entry also needs its Layout, rendered from the current
    # display options, and re-rendered when those options change.

    number: int
    text: str
    tree: Node


class Session:
    """The expression history plus the selection cursor over it.

    The selection is an entry index together with a path into that entry's
    tree; an empty path means the entry is selected as a whole. Every
    navigation method returns whether it moved the selection, so the UI can
    beep or stay quiet accordingly.
    """

    def __init__(self, parse: Callable[[str], Node] = _default_parse) -> None:
        self._parse = parse
        self.entries: list[Entry] = []
        self.selected: int | None = None
        self.path: Path = ()
        self._next_number = 1

    # -- authoring ---------------------------------------------------------

    def author(self, text: str) -> Entry:
        """Append `text` verbatim as a new entry and select it as a whole.

        Nothing is simplified or validated (R-HIST1) and label numbers only
        ever increase (R-HIST2).
        """
        entry = Entry(self._next_number, text, self._parse(text))
        self._next_number += 1
        self.entries.append(entry)
        self.selected = len(self.entries) - 1
        self.path = ()
        return entry

    # -- selection ---------------------------------------------------------

    @property
    def selected_entry(self) -> Entry | None:
        if self.selected is None:
            return None
        return self.entries[self.selected]

    @property
    def selected_node(self) -> Node | None:
        """The node the path points at, or None when nothing is selected."""
        entry = self.selected_entry
        if entry is None:
            return None
        return self._node_at(entry, self.path)

    def selection_span(self) -> Span | None:
        """Source span to highlight, or None when the history is empty.

        An entry selected as a whole highlights all of its text, even when the
        parse dropped enclosing parentheses from the root node's span.
        """
        # TODO(display): a built-up render makes the selection a rectangle
        # rather than a character span. Add selection_rect(), returning
        # Layout.at(self.route), and retire this once the work area paints
        # layouts.
        entry = self.selected_entry
        if entry is None:
            return None
        if not self.path:
            return (0, len(entry.text))
        node = self._node_at(entry, self.path)
        return (node.start, node.end)

    @staticmethod
    def _node_at(entry: Entry, path: Path) -> Node:
        node = entry.tree
        for index in path:
            node = node.children[index]
        return node

    def _siblings(self, entry: Entry) -> tuple[Node, ...]:
        return self._node_at(entry, self.path[:-1]).children

    # -- navigation --------------------------------------------------------

    # TODO(display): the selection is a route into Layout.root, not a path into
    # the parse tree, so Right and Left step Region.children and Up and Down
    # move between a region and its parent or first child. Region.node is how
    # an operation gets from the selection back to the subexpression.

    def select_entry(self, index: int) -> bool:
        """Select entry `index` as a whole, clamped to the history."""
        if not self.entries:
            return False
        index = max(0, min(index, len(self.entries) - 1))
        moved = index != self.selected or bool(self.path)
        self.selected = index
        self.path = ()
        return moved

    def move_up(self) -> bool:
        """Previous entry, or one level up towards the whole expression."""
        if self.selected is None:
            return False
        if self.path:
            self.path = self.path[:-1]
            return True
        return self.select_entry(self.selected - 1)

    def move_down(self) -> bool:
        """Next entry, or one level down into the first operand."""
        if self.selected is None:
            return False
        if self.path:
            node = self.selected_node
            assert node is not None
            if node.is_atom:
                return False
            self.path += (0,)
            return True
        return self.select_entry(self.selected + 1)

    def move_right(self) -> bool:
        """First operand of the whole expression, or the next sibling."""
        entry = self.selected_entry
        if entry is None:
            return False
        if not self.path:
            if entry.tree.is_atom:
                return False
            self.path = (0,)
            return True
        if self.path[-1] + 1 >= len(self._siblings(entry)):
            return False
        self.path = self.path[:-1] + (self.path[-1] + 1,)
        return True

    def move_left(self) -> bool:
        """Previous sibling; a whole expression has none."""
        if self.selected_entry is None or not self.path or self.path[-1] == 0:
            return False
        self.path = self.path[:-1] + (self.path[-1] - 1,)
        return True

    def move_first_sibling(self) -> bool:
        entry = self.selected_entry
        if entry is None or not self.path:
            return False
        moved = self.path[-1] != 0
        self.path = self.path[:-1] + (0,)
        return moved

    def move_last_sibling(self) -> bool:
        entry = self.selected_entry
        if entry is None or not self.path:
            return False
        last = len(self._siblings(entry)) - 1
        moved = self.path[-1] != last
        self.path = self.path[:-1] + (last,)
        return moved

    def move_first_entry(self) -> bool:
        return self.select_entry(0)

    def move_last_entry(self) -> bool:
        return self.select_entry(len(self.entries) - 1)
