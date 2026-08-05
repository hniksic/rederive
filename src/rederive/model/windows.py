"""The windows the `Window` command makes, and where each one lands.

Pure Python, as the session is: no Textual, no key names. The UI asks for the
tree to be split or closed and then asks where everything goes.

A window is a leaf of a binary tree of splits. Each leaf carries a stack of
overlays, one deep unless `Window Open` has pushed another on: overlays share
their window's number and its rectangle, and only the top one is on screen.
Every overlay owns a whole `Session`, which is what makes two algebra windows
two derivations rather than two views of one - splitting copies the worksheet
and the copies go their own ways from there.

Windows are numbered by where they are and not by when they were made: the
numbers are a walk of the tree, first child before second, recomputed after
every split and close. So closing a window in the middle renumbers the ones
after it, exactly as the original does.

Geometry is in cells of the work area, and a window owns its own top border
row and left border column. `Area` is that pair of edges plus the two that
close it, which the next window along owns; a divider is therefore one row or
column shared by the two windows it separates. The area below the work area -
the rule the menu sits under - is the bottom border, which is why `bottom` is
one past the last work row.

The sole window of an unsplit screen is drawn without any border at all, and
takes those four edge cells for itself. That is the original's screen, and it
is why `interior` needs to be told whether a frame is being drawn.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from rederive.model.session import Session

#: The one kind of window this tree holds. The original had three - two of them
#: plot windows - and chose between them with `Window Designate`; here a plot
#: window is a window of the desktop rather than a leaf of this tree, so there
#: is nothing left to choose between and the kind is carried only because a
#: second kind would want it back.
ALGEBRA = "Algebra"

#: How near an edge a split may fall, as the original allows it: a line no
#: closer than two rows to the top of the window and two to the bottom, a
#: column no closer than seven to the left and six to the right. The two are
#: not the same because a column of expressions is worth less narrow than a
#: page of them is short.
LINE_MARGIN = 2
COLUMN_MARGIN_LEFT = 7
COLUMN_MARGIN_RIGHT = 6

#: Where the pieces of a border meet, keyed by the directions a line runs in
#: from that cell. Horizontals are doubled and verticals are single, which is
#: the original's set.
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
    """One window's worth of state, and what kind of window it is."""

    kind: str
    session: Session


class Window:
    """A leaf of the tree: a stack of overlays, the first of them on top.

    It is a class rather than a dataclass because its identity is what the app
    keys a pane widget by, and two windows holding equal state are still two
    windows.
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
        """Whether `Window Open` has put something under what is showing."""
        return len(self.overlays) > 1

    def flip(self, step: int = 1) -> bool:
        """Bring the next overlay to the top, as F2 does.

        False when there is only one, which is a key with nothing to do rather
        than an error: the original flips a lone window to itself in silence.
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

    `at` is where the divider falls, counted from the node's own top left
    corner, so that a subtree can be laid out without knowing where it sits.
    """

    vertical: bool
    at: int
    first: "Node"
    second: "Node"


Node = Window | Split


@dataclass(frozen=True)
class Area:
    """The cells a node owns: its own two edges, and the two that close it.

    `top` and `left` are the window's border row and column. `bottom` and
    `right` belong to whatever is beyond it - the next window's border, or the
    screen's - and are shared with it.
    """

    top: int
    left: int
    bottom: int
    right: int


@dataclass(frozen=True)
class Rect:
    """Where a window's contents go, in cells of the work area."""

    top: int
    left: int
    height: int
    width: int


@dataclass(frozen=True)
class Drawing:
    """The border as characters, plus where each window number is written.

    `rows` is one row longer than the work area: the last of them is the rule
    the menu sits under, which is the frame's bottom edge.
    """

    rows: tuple[str, ...]
    numbers: tuple[tuple[int, int, int], ...]
    """Each window's number, as row, column and width, in numbering order."""


def interior(area: Area, framed: bool) -> Rect:
    """Where a window in `area` draws its expressions.

    A framed window keeps off its own borders. The sole window of an unsplit
    screen has none, so it takes those rows and columns too - which is what
    makes the original's opening screen a full 80 by 20 rather than one inset
    by a frame nobody asked for.
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
    """The line or column a split is offered at: the window cut in half."""
    return (size + 1) // 2


def split_range(vertical: bool, size: int) -> tuple[int, int]:
    """The lines or columns a split of a window `size` across may fall on.

    An empty range - a high below the low - means the window is too small to
    split at all, which is how the original runs out of screen.
    """
    if vertical:
        return COLUMN_MARGIN_LEFT, size - COLUMN_MARGIN_RIGHT
    return LINE_MARGIN, size - LINE_MARGIN


def split_offset(vertical: bool, at: int) -> int:
    """Where the divider goes for an answer of `at`.

    A vertical divider is rounded down onto an even column, as the original
    rounds it: 39 and 38 both split at 38. A horizontal one falls where it was
    asked for.
    """
    return 2 * (at // 2) if vertical else at


class Windows:
    """Every window there is, and which of them is active."""

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
        """The active window's number, which is its place in that order."""
        return self.windows.index(self.active) + 1

    @property
    def framed(self) -> bool:
        """Whether borders are drawn, which they are as soon as there are two."""
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
        """Every session there is, overlaid ones included."""
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
        """Make the next or the previous window active, wrapping round the ends."""
        found = self.windows
        self.active = found[(found.index(self.active) + direction) % len(found)]

    # -- changing what there is --------------------------------------------

    def split(self, vertical: bool, at: int, session: Session) -> Window:
        """Cut the active window in two, `session` filling the new half.

        The active window stays active and becomes the top or left half, which
        is where the original leaves it: an expression authored straight after
        a split lands in the window the split was issued from.
        """
        made = Window([Overlay(self.active.kind, session)])
        split = Split(vertical, split_offset(vertical, at), self.active, made)
        self.root = _replaced(self.root, self.active, split)
        return made

    def open(self, kind: str, session: Session) -> None:
        """Overlay a new window on the active one, sharing its number and place."""
        self.active.overlays.insert(0, Overlay(kind, session))

    def close(self, window: Window) -> list[Session]:
        """Close what `window` is showing, and the window with its last overlay.

        The space a closed window leaves goes to the window it was split from,
        and the numbers of everything after it move up. Closing the active one
        leaves the first window of that space active.
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
        """The cells each window owns, over a work area `height` by `width`.

        The row below the work area is the frame's bottom edge, which is why
        `bottom` is `height` and not one less: the rule the menu sits under is
        that edge, drawn by whoever draws the rule.
        """
        found: dict[Window, Area] = {}
        _place(self.root, Area(0, 0, height, max(0, width - 1)), found)
        return found

    def interior(self, window: Window, height: int, width: int) -> Rect:
        """Where `window` draws its expressions, over that same work area."""
        return interior(self.areas(height, width)[window], self.framed)

    def frame(self, height: int, width: int) -> Drawing:
        """The border around and between the windows.

        One window is drawn without a border, so all that is left of the frame
        is its bottom edge - which is the plain rule the unsplit screen has
        always had.
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
    """The first window of a subtree, which is the lowest numbered of them."""
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
    """What `target` was split from, and so what takes its space back."""
    if isinstance(node, Split):
        if node.first is target:
            return node.second
        if node.second is target:
            return node.first
        return _sibling(node.first, target) or _sibling(node.second, target)
    return None


def _without(node: Node, target: Window) -> Node:
    """`node` with `target` gone, the split that held it gone with it."""
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
    """Every border line, resolved into the characters where they cross.

    Each cell remembers which ways a line leaves it, so a divider running into
    another one picks the piece with those arms and no rule has to know which
    junction it is making.
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
        # A screen too small to hold the windows it has can put a corner off
        # the bottom of it, and a number off the screen is simply not drawn.
        if 0 <= area.top < len(rows):
            line = rows[area.top]
            rows[area.top] = line[: area.left] + text + line[area.left + len(text) :]
    return Drawing(tuple(rows), tuple(numbers))


def _edges(
    areas: Iterable[Area], height: int, width: int
) -> list[tuple[str, int, int, int]]:
    """Every line to draw, as a direction, a row or column, and its two ends.

    A window contributes its own two edges; the two that close it belong to
    whatever is beyond, and the screen's own right and bottom edges are what
    close the last of them.
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
