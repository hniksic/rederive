"""The box model and the painter.

Rendering has two phases. Layout turns a `Node` into a `Box` of sizes and relative
placements, with no characters. Paint blits the boxes into a grid of spaces and builds
the region tree from the selectable boxes. Painting into a grid rather than
concatenating strings keeps fences, integral signs and matrix rules simple: each is a
box drawn at a known offset.

Key invariant: a subexpression occupies one rectangle, and its region is that rectangle
including its padding. The UI highlights by inverting those cells.

Layout decides what is selectable: each box lists its selectable operands in cursor
order, and painting only supplies the rectangles. A box with no node passes its operands
up, which is how a fence exposes the operand it encloses.
"""

from __future__ import annotations

from dataclasses import dataclass

from rederive.display.glyphs import Fence
from rederive.model.expr import Node


@dataclass(frozen=True)
class Box:
    """A rectangle of output, positioned relative to its own baseline.

    `above` and `below` count rows on either side of the baseline row, where `text` is
    drawn. `parts` are the child boxes, offset from this box's left edge and baseline;
    `operands` are the selectable boxes inside this one, in cursor order.

    `parts` and `operands` need not agree. A call draws its head and arguments but only
    the arguments are operands; a chain of sums draws two halves but offers three terms.
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

    `children` are the operands the user can step into, in Derive's order. Unselectable
    nodes (a call's name, the halves of a flattened chain) have no region, so the cursor
    moves over what is on screen rather than over the parse.
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

    def holds(self, row: int, column: int) -> bool:
        """Whether this region covers the cell at `row`, `column`."""
        return (
            self.top <= row < self.top + self.height
            and self.left <= column < self.left + self.width
        )


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

        A route is a tuple of indices into `Region.children`, empty for the whole
        expression. Callers store it as the selection, and it survives re-rendering: the
        selection tree depends only on the expression, so changing the times operator or
        height budget moves rectangles without renumbering routes.
        """
        region = self.root
        for index in route:
            if not 0 <= index < len(region.children):
                return None
            region = region.children[index]
        return region

    def route_at(self, row: int, column: int) -> tuple[int, ...] | None:
        """The route to the smallest region drawn over a cell of `lines`.

        This finds what the mouse points at. Returns None for cells outside the render,
        such as the label field or blanks past the end of a short line. Sibling regions
        never overlap, so descending into the first child that covers the cell is
        unambiguous.
        """
        if not self.root.holds(row, column):
            return None
        route: list[int] = []
        region = self.root
        while True:
            for index, child in enumerate(region.children):
                if child.holds(row, column):
                    route.append(index)
                    region = child
                    break
            else:
                return tuple(route)


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
    """`box` between a pair of delimiters, one column on each side and no extra rows.

    Up to `threshold` rows tall the delimiters are single characters on the baseline;
    taller boxes get rails with corners at top and bottom. `pad` adds blank columns
    inside the fence (used only by matrices).

    The enclosed box is the fence's operand, so parenthesizing for precedence doesn't
    change the selection tree, and a vector's brackets expose its elements.
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
    """Draw `box` with its top-left corner at `(top, left)` and record where."""
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

    A box with no node is glue (a fence, a rail, an operator) and stands for what it
    contains.
    """
    inside = [region for operand in box.operands for region in _regions(operand, places)]
    if box.node is None:
        return inside
    top, left = places[id(box)]
    return [Region(box.node, top, left, box.height, box.width, tuple(inside))]
