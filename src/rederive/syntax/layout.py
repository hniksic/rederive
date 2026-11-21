"""Laying written text out on several rows, for editing.

The writer puts an expression on one line, which is right for the worksheet, for files,
and for a short expression on the Author line. A program on one line is unreadable,
though, so text headed for the Author line or the clipboard is laid out in the shape
the worksheet displays a program in, keeping the brackets and commas that make it text
to type:

    Rev(v, w) :=
      PROG(
        w := [],
        LOOP(
          IF(v = [], RETURN(w)),
          w := ADJOIN(FIRST(v), w),
          v := REST(v)))

An expression that fits the width stays on its row, unless it holds a `PROG` or `LOOP`
of several statements, whose statements read as a sequence. Otherwise a call, vector or
set puts each item on a row of its own, a level deeper than the row its head is on, and
the closing bracket follows the last item. A definition or assignment whose right side
is laid out starts it on a row of its own, a level deeper. Nothing else is broken, so
anything else too long stays on its row.

The text changes only in blanks, so it parses back to the same expression; a line break
is a blank to the parser. `continued` gives the indentation the Author line starts a
typed row with, by the same rule, so typing continues text the way it is laid out.
"""

from __future__ import annotations

from rederive.model.expr import Kind, Node
from rederive.syntax.names import PROGRAM_FUNCTIONS

__all__ = ["INDENT", "continued", "laid_out"]

#: One level of indentation.
INDENT = "  "

#: Brackets that open and close a list.
_OPENERS = "([{"
_CLOSERS = ")]}"

#: The assignment operators a row may end with, its right side following on the next.
_ASSIGNMENTS = (":=", ":-")

#: Programs whose statements are always laid out one per row, when there are several.
_SEQUENCES = frozenset(("PROG", "LOOP")) & PROGRAM_FUNCTIONS


def continued(text: str) -> str:
    """The indentation for a row that continues `text`.

    It is a level deeper than the row holding the innermost bracket still open, or than
    the last row if that ends with an assignment operator, and otherwise the same as the
    last row's. Brackets in strings don't count.
    """
    opened: list[int] = []
    quoted = False
    for at, character in enumerate(text):
        if character == '"':
            quoted = not quoted
        elif quoted:
            continue
        elif character in _OPENERS:
            opened.append(at)
        elif character in _CLOSERS and opened:
            opened.pop()
    if opened:
        return _indentation(text, opened[-1]) + INDENT
    last = _indentation(text, len(text))
    return last + INDENT if text.rstrip().endswith(_ASSIGNMENTS) else last


def _indentation(text: str, at: int) -> str:
    """The leading blanks of the row of `text` that position `at` is on."""
    row = text[text.rfind("\n", 0, at) + 1 :]
    return row[: len(row) - len(row.lstrip(" "))]


def laid_out(text: str, node: Node, width: int, before: str = "") -> str:
    """`node`'s span of `text`, laid out to fit `width` columns.

    `node` must be a tree of `text`, its spans indexing it, as the writer returns.
    `before` is what precedes the result on the row it starts on, and on the rows above
    it: the result's first row starts after it, and rows are indented from the row it
    ends with.
    """
    layout = _Layout(text, width, before)
    layout.node(node)
    return layout.out[len(before) :]


class _Layout:
    def __init__(self, text: str, width: int, before: str) -> None:
        self.text = text
        self.width = width
        self.out = before

    def node(self, node: Node) -> None:
        """Append `node`'s text, laid out, to `out`."""
        flat = self.text[node.start : node.end]
        if not self._broken(node):
            self.out += flat
            return
        items = self._items(node)
        if items is not None:
            self._list(node, items)
        elif (
            node.kind in (Kind.FUNDEF, Kind.ASSIGN)
            and len(node.children) > 1
            and self._ordered(node)
        ):
            self._assignment(node)
        else:
            self._within(node, flat)

    def _broken(self, node: Node) -> bool:
        """Whether `node` is laid out over rows where it is to be appended."""
        return not self._fits(self.text[node.start : node.end]) or self._holds_sequence(
            node
        )

    def _list(self, node: Node, items: tuple[Node, ...]) -> None:
        """Append a list with each item on a row a level deeper than the head's."""
        deeper = _indentation(self.out, len(self.out)) + INDENT
        self.out += self.text[node.start : items[0].start].rstrip()
        for at, item in enumerate(items):
            if at:
                self.out += self.text[items[at - 1].end : item.start].rstrip()
            self.out += "\n" + deeper
            self.node(item)
        self.out += self.text[items[-1].end : node.end].lstrip()

    def _assignment(self, node: Node) -> None:
        """Append a definition or assignment, its right side a row deeper if laid out."""
        *left, right = node.children
        self._within(node, None, until=right)
        operator = self.text[left[-1].end : right.start]
        if self._broken_after(right, operator):
            self.out += operator.rstrip()
            self.out += "\n" + _indentation(self.out, len(self.out)) + INDENT
        else:
            self.out += operator
        self.node(right)
        self.out += self.text[right.end : node.end]

    def _broken_after(self, node: Node, before: str) -> bool:
        """Whether `node` is laid out if it follows `before` on the current row."""
        saved = self.out
        self.out += before
        try:
            return self._broken(node)
        finally:
            self.out = saved

    def _within(self, node: Node, flat: str | None, until: Node | None = None) -> None:
        """Append a node that is neither a list nor laid out as an assignment, laying
        out what is inside it; with `until`, only the part before that child.

        The text between its children is kept as it is. Children that don't follow one
        another in the text can't be laid out separately, so the node stays as written.
        """
        if not node.children or not self._ordered(node):
            assert flat is not None
            self.out += flat
            return
        at = node.start
        for child in node.children:
            if child is until:
                return
            self.out += self.text[at : child.start]
            self.node(child)
            at = child.end
        self.out += self.text[at : node.end]

    def _ordered(self, node: Node) -> bool:
        """Whether `node`'s children follow one another in the text."""
        children = node.children
        return all(left.end <= right.start for left, right in zip(children, children[1:]))

    def _fits(self, flat: str) -> bool:
        column = len(self.out) - (self.out.rfind("\n") + 1)
        return column + len(flat) <= self.width

    def _items(self, node: Node) -> tuple[Node, ...] | None:
        """The items of a bracketed, comma-separated list, or None for anything else.

        A call's first child is its head, written before the bracket.
        """
        children = node.children
        if node.kind is Kind.CALL:
            children = children[1:]
        if len(children) < 2:
            return None
        opening = self.text[node.start : children[0].start].rstrip()
        closing = self.text[children[-1].end : node.end].strip()
        if not opening or opening[-1] not in _OPENERS:
            return None
        if not closing or closing[0] not in _CLOSERS:
            return None
        for left, right in zip(children, children[1:]):
            if self.text[left.end : right.start].strip() != ",":
                return None
        return children

    def _holds_sequence(self, node: Node) -> bool:
        """Whether `node` is or holds a program whose statements go one per row
        regardless of fit."""
        if node.kind is Kind.CALL and len(node.children) > 2:
            head = node.children[0]
            if head.kind is Kind.NAME and head.value in _SEQUENCES:
                return True
        return any(self._holds_sequence(child) for child in node.children)
