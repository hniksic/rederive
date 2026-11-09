"""Windows created by the `Window` command, and their layout.

Like the session, this is UI-independent: the UI asks for splits and closes, then
asks for the layout.

Windows are leaves of a binary tree of splits. Each leaf holds a stack of
overlays (more than one only after `Window Open`) that share its number and
rectangle; only the top one is shown. Each overlay owns its own `Session`, so a
split copies the worksheet and the two windows then diverge.

Windows are numbered in tree order, so closing one renumbers those after it, as
in the original.

Geometry is in work-area cells. A window owns its top border row and left border
column; `Area` adds the bottom and right edges, owned by the next window, so a
divider is shared by the windows it separates. The rule under which the menu sits
is the bottom border, so `bottom` is one past the last work row.

An unsplit screen's sole window has no border and uses the edge cells too, as in
the original; hence `interior` needs to know whether a frame is drawn.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from rederive.model.session import Session

#: The only window kind in the tree. The original also had two plot window kinds,
#: chosen with `Window Designate`; here plots are desktop windows outside the tree.
#: The kind is kept in case another one is added.
ALGEBRA = "Algebra"

#: How close to the window's edges a split may fall, as in the original: at least
#: two rows from top and bottom, and seven columns from the left and six from the
#: right.
LINE_MARGIN = 2
COLUMN_MARGIN_LEFT = 7
COLUMN_MARGIN_RIGHT = 6

#: Border characters keyed by the directions lines leave the cell. As in the
#: original, horizontals are double and verticals single.
_UP, _DOWN, _LEFT, _RIGHT = 1, 2, 4, 8
_PIECES = {
    _LEFT: "═",
    _RIGHT: "═",
    _LEFT | _RIGHT: "═",
    _UP: "│",
    _DOWN: "│",
    _UP | _DOWN: "│",
    _DOWN | _RIGHT: "╒",
    _DOWN | _LEFT: "╕",
    _UP | _RIGHT: "╘",
    _UP | _LEFT: "╛",
    _UP | _DOWN | _LEFT: "╡",
    _UP | _DOWN | _RIGHT: "╞",
    _UP | _LEFT | _RIGHT: "╧",
    _DOWN | _LEFT | _RIGHT: "╤",
    _UP | _DOWN | _LEFT | _RIGHT: "╪",
}


@dataclass
class Overlay:
    """One window's state and kind."""

    kind: str
    session: Session


class Window:
    """A leaf of the tree: a stack of overlays, the first one on top.

    Not a dataclass because the app keys pane widgets by identity, and windows
    with equal state are still distinct.
    """

    def __init__(self, overlays: list[Overlay]) -> None:
        self.overlays = overlays

    @property
    def kind(self) -> str:
        return self.overlays[0].kind

    @property
    def session(self) -> Session:
        return self.overlays[0].session

    @property
    def stacked(self) -> bool:
        """Whether there are overlays below the visible one."""
        return len(self.overlays) > 1

    def flip(self, step: int = 1) -> bool:
        """Bring the next overlay to the top, as F2 does.

        Return False if there's only one; the original silently does nothing.
        """
        if not self.stacked:
            return False
        if step > 0:
            self.overlays.append(self.overlays.pop(0))
        else:
            self.overlays.insert(0, self.overlays.pop())
        return True


@dataclass
class Split:
    """Two windows side by side or one above the other.

    `at` is the divider position relative to the node's top left corner, so a
    subtree can be laid out independently of its position.
    """

    vertical: bool
    at: int
    first: "Node"
    second: "Node"


Node = Window | Split


@dataclass(frozen=True)
class Area:
    """A node's cells, bounded by its own edges and the ones that close it.

    `top` and `left` are the window's border row and column. `bottom` and `right`
    are shared with the next window's border or the screen's.
    """

    top: int
    left: int
    bottom: int
    right: int


@dataclass(frozen=True)
class Rect:
    """The area of a window's contents, in work-area cells."""

    top: int
    left: int
    height: int
    width: int


@dataclass(frozen=True)
class Drawing:
    """The border as characters, plus where each window number is written.

    `rows` has one row more than the work area: the last is the frame's bottom
    edge, the rule under which the menu sits.
    """

    rows: tuple[str, ...]
    numbers: tuple[tuple[int, int, int], ...]
    """Row, column and width of each window number, in numbering order."""


def interior(area: Area, framed: bool) -> Rect:
    """Where a window in `area` draws its expressions.

    A framed window excludes its borders. An unframed one (the sole window) uses
    them too, giving the original's full 80 by 20 opening screen.
    """
    if not framed:
        return Rect(
            area.top,
            area.left,
            max(0, area.bottom - area.top),
            max(0, area.right - area.left + 1),
        )
    return Rect(
        area.top + 1,
        area.left + 1,
        max(0, area.bottom - area.top - 1),
        max(0, area.right - area.left - 1),
    )


def split_default(size: int) -> int:
    """The default split line or column, halving the window."""
    return (size + 1) // 2


def split_range(vertical: bool, size: int) -> tuple[int, int]:
    """The range of lines or columns where a window `size` across may be split.

    An empty range (high below low) means the window is too small to split.
    """
    if vertical:
        return COLUMN_MARGIN_LEFT, size - COLUMN_MARGIN_RIGHT
    return LINE_MARGIN, size - LINE_MARGIN


def split_offset(vertical: bool, at: int) -> int:
    """The divider position for a requested split at `at`.

    As in the original, vertical dividers round down to an even column, so 39
    and 38 both split at 38.
    """
    return 2 * (at // 2) if vertical else at


class Windows:
    """All windows, and which one is active."""

    def __init__(self, session: Session, kind: str = ALGEBRA) -> None:
        self.root: Node = Window([Overlay(kind, session)])
        self.active: Window = self.root

    # -- what there is -----------------------------------------------------

    @property
    def windows(self) -> list[Window]:
        """Every window, in the order they are numbered."""
        found: list[Window] = []
        _collect(self.root, found)
        return found

    @property
    def number(self) -> int:
        """The active window's number (its position in `windows`)."""
        return self.windows.index(self.active) + 1

    @property
    def framed(self) -> bool:
        """Whether borders are drawn, i.e. there is more than one window."""
        return isinstance(self.root, Split)

    @property
    def session(self) -> Session:
        return self.active.session

    @property
    def kind(self) -> str:
        return self.active.kind

    def numbered(self, number: int) -> Window | None:
        found = self.windows
        return found[number - 1] if 1 <= number <= len(found) else None

    def sessions(self) -> list[Session]:
        """All sessions, including overlaid ones."""
        return [
            overlay.session for window in self.windows for overlay in window.overlays
        ]

    # -- changing the active window ----------------------------------------

    def goto(self, number: int) -> bool:
        window = self.numbered(number)
        if window is None:
            return False
        self.active = window
        return True

    def step(self, direction: int) -> None:
        """Activate the next or previous window, wrapping around."""
        found = self.windows
        self.active = found[(found.index(self.active) + direction) % len(found)]

    # -- changing what there is --------------------------------------------

    def split(self, vertical: bool, at: int, session: Session) -> Window:
        """Split the active window in two, with `session` in the new half.

        As in the original, the active window stays active as the top or left
        half.
        """
        made = Window([Overlay(self.active.kind, session)])
        split = Split(vertical, split_offset(vertical, at), self.active, made)
        self.root = _replaced(self.root, self.active, split)
        return made

    def open(self, kind: str, session: Session) -> None:
        """Overlay a new window on the active one, sharing its number and place."""
        self.active.overlays.insert(0, Overlay(kind, session))

    def close(self, window: Window) -> list[Session]:
        """Close `window`'s top overlay, and the window itself if it was the last.

        The freed space goes to the sibling, and later windows are renumbered.
        Closing the active window activates the first window of that space.
        """
        if window.stacked:
            return [window.overlays.pop(0).session]
        dropped = [overlay.session for overlay in window.overlays]
        heir = _sibling(self.root, window)
        self.root = _without(self.root, window)
        if self.active is window:
            self.active = _first(heir if heir is not None else self.root)
        return dropped

    # -- where everything goes ---------------------------------------------

    def areas(self, height: int, width: int) -> dict[Window, Area]:
        """The cells each window owns, in a work area `height` by `width`.

        `bottom` is `height` rather than one less because the frame's bottom edge
        is the rule below the work area.
        """
        found: dict[Window, Area] = {}
        _place(self.root, Area(0, 0, height, max(0, width - 1)), found)
        return found

    def interior(self, window: Window, height: int, width: int) -> Rect:
        """Where `window` draws its expressions, in the same work area."""
        return interior(self.areas(height, width)[window], self.framed)

    def frame(self, height: int, width: int) -> Drawing:
        """The border around and between the windows.

        A single window has no border, so only the bottom rule is drawn.
        """
        rule = "═" * width
        if not self.framed or height <= 0 or width <= 0:
            return Drawing((" " * width,) * max(0, height) + (rule,), ())
        return _drawn(self.areas(height, width), self.windows, height, width)


# -- the tree ------------------------------------------------------------


def _collect(node: Node, found: list[Window]) -> None:
    if isinstance(node, Window):
        found.append(node)
        return
    _collect(node.first, found)
    _collect(node.second, found)


def _first(node: Node) -> Window:
    """The first (lowest-numbered) window of a subtree."""
    while isinstance(node, Split):
        node = node.first
    return node


def _replaced(node: Node, target: Window, replacement: Node) -> Node:
    """`node` with `target` swapped for `replacement`."""
    if node is target:
        return replacement
    if isinstance(node, Split):
        node.first = _replaced(node.first, target, replacement)
        node.second = _replaced(node.second, target, replacement)
    return node


def _sibling(node: Node, target: Window) -> Node | None:
    """The sibling of `target`, which takes its space when it closes."""
    if isinstance(node, Split):
        if node.first is target:
            return node.second
        if node.second is target:
            return node.first
        return _sibling(node.first, target) or _sibling(node.second, target)
    return None


def _without(node: Node, target: Window) -> Node:
    """`node` with `target` and its parent split removed."""
    if isinstance(node, Split):
        if node.first is target:
            return node.second
        if node.second is target:
            return node.first
        node.first = _without(node.first, target)
        node.second = _without(node.second, target)
    return node


# -- geometry ------------------------------------------------------------


def _place(node: Node, area: Area, found: dict[Window, Area]) -> None:
    if isinstance(node, Window):
        found[node] = area
        return
    if node.vertical:
        at = min(max(area.left + node.at, area.left + 1), area.right - 1)
        _place(node.first, replace(area, right=at), found)
        _place(node.second, replace(area, left=at), found)
        return
    at = min(max(area.top + node.at, area.top + 1), area.bottom - 1)
    _place(node.first, replace(area, bottom=at), found)
    _place(node.second, replace(area, top=at), found)


def _drawn(
    areas: dict[Window, Area], order: list[Window], height: int, width: int
) -> Drawing:
    """All border lines, with proper junction characters where they meet.

    Each cell records the directions lines leave it in, which picks the right
    junction character without special cases.
    """
    arms: dict[tuple[int, int], int] = {}
    for area in _edges(areas.values(), height, width):
        _rule(arms, area)
    rows = [
        "".join(_PIECES.get(arms.get((row, column), 0), " ") for column in range(width))
        for row in range(height + 1)
    ]
    numbers: list[tuple[int, int, int]] = []
    for index, window in enumerate(order):
        area = areas[window]
        text = str(index + 1)[: max(0, width - area.left)]
        numbers.append((area.top, area.left, len(text)))
        # On a screen too small for its windows, a number may fall off the
        # bottom; skip it.
        if 0 <= area.top < len(rows):
            line = rows[area.top]
            rows[area.top] = line[: area.left] + text + line[area.left + len(text) :]
    return Drawing(tuple(rows), tuple(numbers))


def _edges(
    areas: Iterable[Area], height: int, width: int
) -> list[tuple[str, int, int, int]]:
    """All lines to draw, as direction, row or column, and the two ends.

    Each window contributes its top and left edges; the screen's right and bottom
    edges close the rest.
    """
    lines = [
        ("-", height, 0, width - 1),
        ("|", width - 1, 0, height),
    ]
    for area in areas:
        lines.append(("-", area.top, area.left, area.right))
        lines.append(("|", area.left, area.top, area.bottom))
    return lines


def _rule(arms: dict[tuple[int, int], int], line: tuple[str, int, int, int]) -> None:
    direction, at, start, end = line
    for step in range(start, end + 1):
        cell = (at, step) if direction == "-" else (step, at)
        if direction == "-":
            arms[cell] = arms.get(cell, 0) | (_LEFT if step > start else 0)
            arms[cell] |= _RIGHT if step < end else 0
        else:
            arms[cell] = arms.get(cell, 0) | (_UP if step > start else 0)
            arms[cell] |= _DOWN if step < end else 0
