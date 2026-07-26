"""The box model and the painter.

Two phases. Layout turns a `Node` into a `Box`: sizes and relative placements
only, no characters. Paint allocates a grid of spaces, blits the boxes into
it, and turns the boxes that are selectable into the region tree.

Painting into a grid rather than concatenating strings is what makes fences,
integral signs and matrix rules simple: each is a box drawn at a known offset.

The invariant that matters: a subexpression occupies one rectangle, and its
region is that rectangle including its own padding. The UI highlights by
inverting those cells.

What is selectable is layout's decision, not painting's: a box names its
selectable operands, in the order the selection cursor visits them, and
painting only supplies the rectangles. A box with no node of its own passes
its operands up, which is how a fence hands over the operand it encloses.
"""

from __future__ import annotations

from dataclasses import dataclass

from rederive.display.glyphs import Fence
from rederive.model.expr import Node


@dataclass(frozen=True)
class Box:
    """A rectangle of output, positioned relative to its own baseline.

    `above` and `below` count rows either side of the baseline row, which is
    the row `text` is drawn on. `parts` are the child boxes, each offset from
    this box's left edge and baseline; `operands` are the boxes that are
    selectable inside this one, in the order the cursor visits them.

    `parts` and `operands` answer different questions and need not agree. A
    call draws its head and its argument list but only the arguments are
    operands; a chain of sums draws two halves but offers three terms.
    """

    width: int
    above: int
    below: int
    node: Node | None
    text: str = ""
    parts: tuple[Placed, ...] = ()
    operands: tuple[Box, ...] = ()

    @property
    def height(self) -> int:
        return self.above + 1 + self.below


@dataclass(frozen=True)
class Placed:
    """A box, offset from its parent's left edge and baseline row."""

    box: Box
    dx: int
    dy: int


@dataclass(frozen=True)
class Region:
    """A selectable subexpression: what it is, and where it was drawn.

    `children` are the operands the user can step into, in the order the
    original steps them. A node that cannot be selected - the name of a call,
    the halves of a flattened chain - has no region at all, so the cursor
    moves over what is on the screen rather than over the parse.
    """

    node: Node
    top: int
    left: int
    height: int
    width: int
    children: tuple[Region, ...] = ()

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return (self.top, self.left, self.height, self.width)


@dataclass(frozen=True)
class Layout:
    """A finished render: its lines, its baseline, and what can be selected.

    Lines are padded to the full width; the caller may strip them. Column and
    row indices in the region tree index `lines` directly.
    """

    lines: tuple[str, ...]
    baseline: int
    root: Region

    @property
    def width(self) -> int:
        return self.root.width

    @property
    def height(self) -> int:
        return len(self.lines)

    def at(self, route: tuple[int, ...]) -> Region | None:
        """The region a route names, or None if the route leads nowhere.

        A route is a tuple of indices into `Region.children`, empty for the
        whole expression. It is what a caller stores as "what is selected",
        and it survives a re-render: the selection tree comes from the
        expression alone, so changing the times operator or the height budget
        moves rectangles without renumbering routes.
        """
        region = self.root
        for index in route:
            if not 0 <= index < len(region.children):
                return None
            region = region.children[index]
        return region


# -- building boxes ---------------------------------------------------------


def text(content: str, node: Node | None = None) -> Box:
    """A one-row box of literal characters."""
    return Box(len(content), 0, 0, node, content)


def row(
    boxes: list[Box],
    node: Node | None = None,
    operands: tuple[Box, ...] = (),
) -> Box:
    """Boxes side by side, baselines aligned, no gaps."""
    parts = []
    width = 0
    for box in boxes:
        parts.append(Placed(box, width, 0))
        width += box.width
    above = max((box.above for box in boxes), default=0)
    below = max((box.below for box in boxes), default=0)
    return Box(width, above, below, node, "", tuple(parts), operands)


def rail(cells: tuple[str, str, str], above: int, below: int, node: Node | None) -> Box:
    """A one-column box: `cells` are its top, middle and bottom characters."""
    parts = []
    for dy in range(-above, below + 1):
        cell = cells[0] if dy == -above else cells[2] if dy == below else cells[1]
        parts.append(Placed(text(cell), 0, dy))
    return Box(1, above, below, node, "", tuple(parts))


def fenced(
    box: Box,
    fence: Fence,
    *,
    pad: int = 0,
    threshold: int = 2,
    node: Node | None = None,
) -> Box:
    """`box` between a pair of delimiters, one column each side and no rows.

    Up to `threshold` rows tall the delimiters are single characters on the
    baseline row; taller than that they become rails, cornered top and bottom.
    `pad` is blank columns inside the fence, which only matrices ask for.

    A fence holds exactly what it was given, so what it encloses is its
    operand: fencing an expression for its precedence leaves the selection
    tree alone, and a bracket that belongs to a vector hands over its
    elements.
    """
    if box.height <= threshold:
        left = text(fence.short[0])
        right = text(fence.short[1])
    else:
        left = rail(fence.left, box.above, box.below, None)
        right = rail(fence.right, box.above, box.below, None)
    inner = pad + box.width + pad
    parts = (
        Placed(left, 0, 0),
        Placed(box, 1 + pad, 0),
        Placed(right, 1 + inner, 0),
    )
    return Box(inner + 2, box.above, box.below, node, "", parts, (box,))


def centered(field: int, width: int) -> int:
    """Where content of `width` sits in a field of `field`. Ties go right."""
    return (field - width + 1) // 2


# -- painting ---------------------------------------------------------------


def paint(box: Box) -> Layout:
    """Draw `box` into a grid of spaces and collect the selection tree."""
    grid = [[" "] * box.width for _ in range(box.height)]
    places: dict[int, tuple[int, int]] = {}
    _blit(grid, box, 0, 0, places)
    lines = tuple("".join(cells) for cells in grid)
    regions = _regions(box, places)
    assert len(regions) == 1, "the outermost box must be one selectable region"
    return Layout(lines, box.above, regions[0])


def _blit(
    grid: list[list[str]],
    box: Box,
    top: int,
    left: int,
    places: dict[int, tuple[int, int]],
) -> None:
    """Draw `box` with its top-left corner at `(top, left)`, and note where."""
    places[id(box)] = (top, left)
    if box.text:
        cells = grid[top + box.above]
        for offset, character in enumerate(box.text):
            cells[left + offset] = character
    for placed in box.parts:
        child = placed.box
        _blit(
            grid,
            child,
            top + box.above + placed.dy - child.above,
            left + placed.dx,
            places,
        )


def _regions(box: Box, places: dict[int, tuple[int, int]]) -> list[Region]:
    """The regions `box` contributes: itself, or its operands if it has no node.

    A box with no node is glue - a fence, a rail, an operator's spelling - and
    stands for whatever it was given to hold.
    """
    inside = [region for operand in box.operands for region in _regions(operand, places)]
    if box.node is None:
        return inside
    top, left = places[id(box)]
    return [Region(box.node, top, left, box.height, box.width, tuple(inside))]
