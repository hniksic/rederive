"""The 3D plot window: several surfaces in one solid, and a camera that costs nothing.

A surface is read by turning it over, so the whole of this window is arranged
around making that free. The mesh is built once per domain and grid and never
again: every orbit, every wheel click, every preset is a camera move over
vertices that are already on the card, and nothing the mouse can do starts an
evaluation. Only the two toolbar fields - the domain and the grid - do that,
because they are the two things that change what was computed rather than where
it is looked at from.

**The picture is drawn in a box of the program's choosing, not the data's.** The
floor is always the same square, whatever rectangle the domain is, and the
height is the true proportion of the z range to the floor - until that
proportion is one no picture can hold, when it is capped at the floor's own
length or lifted off it. So a surface whose z runs to a million reads as a shape
rather than as a wall, a hemisphere over its own disc reads as a hemisphere, and
a camera that frames one surface frames every other one. The numbers along the
box edges carry the truth; the geometry carries the form. That is also what
makes `Home` a fixed camera rather than a computation: there is only ever one
box, near enough, to look at.

**The z extent is the data's, within reason.** It autoscales to the finite
values of every surface in the window, and where a spike would crush everything
else into a plane it is taken from the 1st to the 99th percentile instead, with
the status bar saying so. Vertices outside the clip leave the mesh exactly as
non-real ones do, so the spike is not drawn flattened against the lid - it is
simply not there, and the sentence says why.

**Holes are holes, and the mesh stops where the surface does.**
`SQRT(1-x^2-y^2)` is a dome over the unit disc and nothing at all outside it,
so faces are kept only where the surface is real, and a cell the domain's edge
crosses is filled up to the boundary the sampler refined - on the closure, to
a fraction of a cell - rather than dropped, so the dome's rim is a circle and
not a staircase. `GLSurfacePlotItem` cannot do any of that - it generates the
full grid of faces from the array's shape - so the triangles are built here and
drawn by a plain `GLMeshItem`. What the item does with a NaN vertex is not a
contract anyone has written down, and a dome with a skirt of garbage triangles
around it is the picture that gets drawn when you trust it.

**A wire surface hides what is behind it.** Derive's plotter drew a grid with
its hidden lines removed, and a see-through grid is a shape a reader has to
solve rather than see - the far side of a dome comes forward, and two surfaces
in one window are one fabric. So the wire is drawn over its own solid, painted
in the color of the canvas and shoved a hair back by the polygon offset: an
invisible body for the depth buffer to hide the lines behind, this surface's
lines and every other item's alike.

**A window that cannot get an OpenGL context says so.** The context is asked for
when the window is first shown, which is inside the Qt event loop, and an
exception there would leave the host dead and every 2D window with it. It is
caught and put on the status bar instead: no picture, but a plot list, a title
and a message naming what happened.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from OpenGL import GL
from pyqtgraph.opengl.GLGraphicsItem import GLOptions
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from rederive.engine.context import Context
from rederive.model.expr import Node
from rederive.plot import evaluate, protocol, theme
from rederive.plot.protocol import Options, PlotKind
from rederive.plot.window2d import (
    CLICK_SLOP_PX,
    CURVE_WIDTH,
    PALETTE,
    PAPER_PALETTE,
    Legend,
    commanded,
    naming,
    pressed,
)
from rederive.syntax import DeriveSyntaxError, ParseState, parse_expression

__all__ = ["Box", "Surface", "Window3D", "mesh", "ticks", "wire"]

#: The surface colors, which are the curve colors turned by one. A shaded solid
#: in white is a white shape with a white shape behind it - the palette's first
#: color is the right one for a stroke on a dark canvas and the worst one for a
#: lit surface - so the order starts at the second and comes round to it last.
SOLID_PALETTE = PALETTE[1:] + PALETTE[:1]
SOLID_PAPER = PAPER_PALETTE[1:] + PAPER_PALETTE[:1]

#: The window's own colors, which are the 2D window's, so that two plot
#: windows side by side are two windows of one program. What is around the
#: picture - the bar, the fields, the status line - is the chrome's business
#: and lives in `theme`.
BACKGROUND = "#0c0c10"
BOX_COLOR = (150, 150, 150, 110)
TICK_COLOR = (150, 150, 150, 200)
TEXT_COLOR = "#d0d0d0"

#: The same three, for the white background every image export is taken on.
PAPER_BOX = (40, 40, 40, 230)
PAPER_TICK = (60, 60, 60, 255)
PAPER_TEXT = "#000000"

#: The rectangle a fresh window evaluates over, and how many samples of it each
#: axis takes. Sixty-four is dense enough that `SIN(x*y)` over the default
#: domain does not alias; the cap is what stops a typed grid from asking for a
#: quarter of a million vertices.
DEFAULT_DOMAIN = (-5.0, 5.0)
DEFAULT_GRID = 64
MAX_GRID = 256

#: The side of the square floor every surface is drawn over, and half of it,
#: which is where the box walls are. World units, and they mean nothing else:
#: the axis numbers are the only quantities in this window with units.
WORLD = 10.0
HALF = WORLD / 2

#: How short the box may be drawn, as a fraction of its floor. The height is
#: the true one - a dome of radius 1 over a floor of 4 is drawn as a dome and
#: not as a bullet - until the surface is so flat that the true height is a
#: line, and then it is exaggerated to this much so that there is a shape to
#: look at rather than a lid.
MIN_HEIGHT = 0.15

#: When the z extent is taken from the percentiles rather than from the data.
#: A surface whose full range is four times its middle 98% has a spike in it,
#: and drawing to the spike is drawing everything else as a floor.
PERCENTILES = (1.0, 99.0)
CLIP_FACTOR = 4.0

#: How a vertex's brightness is arrived at. Height is half of it - the ramp
#: runs from `DIM` at the floor of the box to 1 at the lid - and how the
#: surface lies to a fixed light is the other half. Height alone reads a hill
#: from a valley and nothing else, since two slopes at the same height are the
#: same color; a light is what makes a fold in the middle of a surface show.
#: The light is baked into the vertex colors rather than computed by a shader,
#: which is what keeps the mesh free of normals and the frame rate free of
#: anything.
DIM = 0.30
LIGHT = (0.40, -0.60, 0.69)
AMBIENT = 0.20
HEIGHT_SHARE = 0.5

#: How many row and column lines the wire drawing aims for each way. The full
#: grid at the default 64 x 64 is some eight thousand segments and reads as
#: gray fuzz, so only every k-th row and column is drawn - the thinning is of
#: the drawing, never of the sampling, and every drawn line keeps the full
#: sample spacing. Raising the grid in the toolbar therefore sharpens the
#: wire's lines instead of multiplying them.
WIRE_LINES = 20

#: How far the wire's occluder is pushed away from the camera, as OpenGL's
#: (factor, units) pair. The occluder is the surface's own triangles, so the
#: lines lie exactly on it and the depth test would decide their pixels by
#: rounding - a wire stitched out of dashes. A small shove backwards settles
#: it: the lines win everywhere they touch the surface, and the shove is far
#: too slight to let a line on the far side through.
WIRE_OFFSET = (1.0, 2.0)

#: The GL state that shove is asked for in - the stock opaque state, which is
#: what every other item here draws under, and the polygon offset over it.
WIRE_OCCLUDER = {
    **GLOptions["opaque"],
    GL.GL_POLYGON_OFFSET_FILL: True,
    "glPolygonOffset": WIRE_OFFSET,
}

#: The camera a fresh window opens with and `Home` returns to: a three-quarter
#: view from above, far enough out that the whole cube is in the picture.
CAMERA_ELEVATION = 30.0
CAMERA_AZIMUTH = -60.0
CAMERA_DISTANCE = 23.0

#: The three presets, as elevation and azimuth: face the xy plane from above,
#: the xz plane from the front, the yz plane from the side.
PLANES = {
    "xy": (90.0, -90.0),
    "xz": (0.0, -90.0),
    "yz": (0.0, 0.0),
}

#: How far one arrow key turns the camera, and how fast `R` turns it by itself.
#: A rotation is for reading a shape from every side while the eye stays still,
#: so it is slow: a turn takes half a minute.
ORBIT_DEGREES = 5.0
SPIN_MS = 33
SPIN_DEGREES = 0.4

#: How many tick marks an axis aims for, and the most it may draw - the pool of
#: text items is made once and never grown, since items cannot be added to the
#: view while it is painting.
TICKS = 5
MAX_TICKS = 9

#: What a window with no usable OpenGL says instead of drawing. The reason is
#: the toolkit's own words, which are the only ones that name a missing driver.
NO_OPENGL = "3D drawing is not available: {reason}"

#: How nearly an axis has to point at the camera before its numbers are dropped,
#: as the cosine of the angle between them: about five degrees.
EDGE_ON = 0.996

#: How far a tick mark and its number stand out of the box, in world units.
TICK_OUT = 0.35
LABEL_OUT = 1.05
NAME_OUT = 2.3


@dataclass
class Surface:
    """One entry of a 3D window's plot list: what to draw and what it is called.

    The identity is `worksheet` and `label` together, exactly as a curve's is,
    so re-plotting `#3` replaces its surface and keeps its color while a `#3`
    from another algebra overlay is a second surface.

    `values` is the grid the closure last answered with, kept because the mesh
    is rebuilt whenever the z extent moves - another surface arriving in the
    same window changes what the cube holds without changing what this surface
    is worth.
    """

    worksheet: int
    label: str
    text: str
    kind: PlotKind
    node: Node
    context: Context
    options: Options
    #: The parse state the expression arrived under, which is what the domain
    #: fields read a typed bound with: the same grammar the worksheet reads.
    state: ParseState = field(default_factory=ParseState)
    color: str = SOLID_PALETTE[0]
    paper: str = SOLID_PAPER[0]
    item: Any = None
    #: Whether this surface draws as the wire grid of its samples rather than
    #: as a shaded solid. A property of the surface, not the window: the
    #: toolbar's `mesh` box flips every surface at once, and the legend's
    #: right-click carries the per-surface exception.
    wire: bool = False
    #: The line item the wire look draws with, made beside `item` and shown
    #: over it while `wire` is on - `item` stays, as the shape the lines are
    #: hidden behind.
    wires: Any = None
    #: The lambdified closure, once the sampling thread has built one.
    closure: Callable[..., np.ndarray] | None = None
    trouble: str = ""
    xs: np.ndarray = field(default_factory=lambda: np.empty(0))
    ys: np.ndarray = field(default_factory=lambda: np.empty(0))
    values: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    #: Where the surface stops being real, refined along the grid's edges on
    #: the sampling thread and cached beside the arrays it was found on: the
    #: mesh reads it to end at the boundary rather than a grid step short.
    boundary: evaluate.Boundary | None = None
    hidden: bool = False
    #: Which evaluation these values came from. A job whose generation has
    #: moved on is a job about a domain that is gone.
    generation: int = 0

    @property
    def visible(self) -> bool:
        return not self.hidden

    @property
    def named(self) -> str:
        """How the legend and the status line name this surface."""
        return naming(self.label, self.text)

    @property
    def axes(self) -> tuple[str, str]:
        """The two variables of the floor, in the order the axes take them."""
        names = self.options.variables
        if len(names) >= 2:
            return names[0], names[1]
        return (names[0] if names else "x"), "y"

    @property
    def vertical(self) -> str:
        """What the upright axis is called: the expression's name for it, or z.

        `z = x^2 + y^2` names it and an expression that is only `x^2 + y^2`
        does not, so the second gets the letter every reader supplies anyway.
        """
        return self.options.vertical or "z"


@dataclass(frozen=True)
class Box:
    """The three ranges the picture stands for: the domain, and the z it holds.

    Everything drawn is placed through this and nothing else, which is what
    keeps one mapping between the numbers a reader sees and the geometry the
    card draws. The floor is always the same square, whatever rectangle the
    domain is - the axis numbers say what it is - and the height is the one
    proportion kept, because that is what tells a dome from a bullet.
    """

    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    @property
    def center(self) -> tuple[float, float, float]:
        return tuple(float(low + high) / 2 for low, high in (self.x, self.y, self.z))

    @property
    def lengths(self) -> tuple[float, float, float]:
        return tuple(float(high - low) for low, high in (self.x, self.y, self.z))

    @property
    def height(self) -> float:
        """How tall the box is drawn, in world units.

        In proportion to the floor while the proportion is one a picture can
        hold: a surface whose z runs ten times its domain would be a tower and
        is drawn at the height of the floor instead, and one whose z barely
        varies would be a sheet and is drawn at a fraction of it. Between those
        two the height is true, which is the interesting range - a hemisphere
        over its own disc has to read as a hemisphere.
        """
        across, along, up = self.lengths
        floor = float(np.sqrt(abs(across) * abs(along)))
        if not np.isfinite(floor) or floor <= 0 or not np.isfinite(up) or up <= 0:
            return WORLD
        return WORLD * min(max(up / floor, MIN_HEIGHT), 1.0)

    def across(self, values: Any) -> np.ndarray:
        return place(values, self.x, WORLD)

    def along(self, values: Any) -> np.ndarray:
        return place(values, self.y, WORLD)

    def up(self, values: Any) -> np.ndarray:
        return place(values, self.z, self.height)


def place(values: Any, span: tuple[float, float], length: float) -> np.ndarray:
    """Data coordinates onto the box: the range of `span` onto `length` of world."""
    low, high = float(span[0]), float(span[1])
    width = high - low
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(width) or width == 0:
        return np.zeros_like(array)
    return length * ((array - low) / width - 0.5)


def extent(arrays: Sequence[np.ndarray]) -> tuple[tuple[float, float], bool] | None:
    """The z range the box takes, and whether the percentiles decided it.

    The whole finite data, unless the data is mostly one thing and a little of
    another: a single pole in a corner of the grid can be a thousand times the
    rest of the surface, and drawing to it makes every other feature a flat
    floor. The 1st and 99th percentiles are then the box, the spike leaves the
    mesh, and the window says that it did.

    None where there is nothing finite at all, which is the message rather than
    a picture.
    """
    finite: list[np.ndarray] = []
    for array in arrays:
        flat = np.asarray(array, dtype=np.float64).ravel()
        finite.append(flat[np.isfinite(flat)])
    values = np.concatenate(finite) if finite else np.empty(0)
    if not values.size:
        return None
    low, high = float(np.min(values)), float(np.max(values))
    inner = np.percentile(values, PERCENTILES)
    tight = (float(inner[0]), float(inner[1]))
    clipped = False
    if tight[1] > tight[0] and (high - low) > CLIP_FACTOR * (tight[1] - tight[0]):
        low, high, clipped = tight[0], tight[1], True
    if high <= low:
        # A plane is a surface too, and a box of no height cannot be divided by.
        low, high = low - 1.0, high + 1.0
    return (low, high), clipped


def mesh(
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
    box: Box,
    boundary: evaluate.Boundary | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One surface as vertices in the box, the faces that reach it, and shading.

    A face is kept whole where all four of its corners are real, finite and
    inside the box. A cell that the edge of the surface's domain crosses is
    filled up to the boundary instead of dropped: its defined corners plus the
    refined crossing points `boundary` carries make a triangle, a quadrilateral
    or a pentagon - the marching-squares cases - so the mesh ends where the
    surface does rather than a grid step short. The clip planes come second,
    array-only and evaluation-free because the box moves on the Qt thread: the
    boundary there is a known z, so every partial cell's polygon is trimmed to
    it by linear interpolation along its edges, which is also what saves a
    boundary vertex whose limit is unbounded. Without `boundary` the cells the
    domain's edge crosses are dropped as before; the clip trimming needs no
    refinement data and happens regardless.

    The corners a face never references keep a vertex each - unreferenced, and
    therefore not drawn - because renumbering the survivors would cost more
    than the few kilobytes the holes are worth.

    The shading comes back beside the mesh because it is read off the same
    array: how high a vertex is and which way the surface faces there are both
    questions about z, and answering them here is what leaves the drawing with
    no normals to compute and no lighting to do. A vertex on the boundary is
    not on the grid, so its light is interpolated along the edge it sits on
    rather than read from a gradient that has no cell there.
    """
    zs = np.asarray(values, dtype=np.float64)
    if zs.ndim != 2 or zs.shape[0] < 2 or zs.shape[1] < 2:
        return np.empty((0, 3), dtype=np.float32), _no_faces(), np.empty(0)
    finite = np.isfinite(zs)
    low, high = box.z
    inside = finite & (zs >= low) & (zs <= high)
    heights = np.clip((zs - low) / (high - low), 0.0, 1.0)
    lit = _lit(np.where(finite, box.up(zs), np.nan))
    shading = HEIGHT_SHARE * (DIM + (1.0 - DIM) * heights) + (1.0 - HEIGHT_SHARE) * lit
    upright = np.where(inside, box.up(zs), -box.height / 2)
    across, along = np.meshgrid(box.across(xs), box.along(ys), indexing="ij")
    vertexes = np.stack([across, along, upright], axis=-1).reshape(-1, 3)
    index = np.arange(zs.size).reshape(zs.shape)
    whole = inside[:-1, :-1] & inside[1:, :-1] & inside[1:, 1:] & inside[:-1, 1:]
    corners = [
        index[:-1, :-1][whole],
        index[1:, :-1][whole],
        index[1:, 1:][whole],
        index[:-1, 1:][whole],
    ]
    faces = np.concatenate(
        [
            np.stack(corners[:3], axis=1),
            np.stack([corners[0], corners[2], corners[3]], axis=1),
        ]
    )
    gained, patches = _partial(
        xs, ys, zs, finite, whole, shading, lit, boundary, box, offset=zs.size
    )
    # A vertex with no value has no shading either; it is unreferenced, and the
    # colors it feeds must merely be numbers.
    flat = np.nan_to_num(shading, nan=DIM).reshape(-1)
    if gained:
        points = np.array(gained, dtype=np.float64)
        placed = np.stack(
            [box.across(points[:, 0]), box.along(points[:, 1]), box.up(points[:, 2])],
            axis=1,
        )
        vertexes = np.concatenate([vertexes, placed])
        flat = np.concatenate([flat, points[:, 3]])
        patched = np.array(patches, dtype=np.int64)
        faces = np.concatenate([faces, patched]) if faces.size else patched
    return (
        vertexes.astype(np.float32),
        faces.astype(np.uint32) if faces.size else _no_faces(),
        flat,
    )


#: The corners of a grid cell in perimeter order from its lower-left, so that
#: edge k runs from corner k to corner k+1 and a walk over both visits the
#: cell's rim exactly once.
CELL = ((0, 0), (1, 0), (1, 1), (0, 1))


def _partial(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    finite: np.ndarray,
    whole: np.ndarray,
    shading: np.ndarray,
    lit: np.ndarray,
    boundary: evaluate.Boundary | None,
    box: Box,
    offset: int,
) -> tuple[list[tuple[float, float, float, float]], list[tuple[int, int, int]]]:
    """Fill the cells that stop short of being whole faces, up to where they stop.

    Two kinds of cell arrive here: cells the domain's edge crosses, filled from
    their defined corners and the refined crossings, and cells the clip planes
    cut, whose corners are all real. Both go the same way - build the defined
    polygon, then trim it to the box's z - which is the ordering that lets the
    clip trim a boundary vertex like any other. Cells that are real all over
    but stand entirely above or below the box trim to nothing and are skipped
    without a look.

    Returns the vertices gained, as (x, y, z, shade) in data coordinates, and
    the triangles over them, indexed from `offset` on.
    """
    low, high = box.z
    quads = finite[:-1, :-1] & finite[1:, :-1] & finite[1:, 1:] & finite[:-1, 1:]
    some = finite[:-1, :-1] | finite[1:, :-1] | finite[1:, 1:] | finite[:-1, 1:]
    over, under = zs > high, zs < low
    gone = (over[:-1, :-1] & over[1:, :-1] & over[1:, 1:] & over[:-1, 1:]) | (
        under[:-1, :-1] & under[1:, :-1] & under[1:, 1:] & under[:-1, 1:]
    )
    cells = some & ~whole & ~gone
    if boundary is None:
        cells &= quads
    points: list[tuple[float, float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for i, j in zip(*np.nonzero(cells)):
        filled = _filled(
            int(i), int(j), xs, ys, zs, finite, shading, lit, boundary, box.z
        )
        for polygon in filled:
            trimmed = _trimmed(polygon, low, high)
            if len(trimmed) < 3:
                continue
            base = offset + len(points)
            points.extend(trimmed)
            faces.extend(
                (base, base + k, base + k + 1) for k in range(1, len(trimmed) - 1)
            )
    return points, faces


def _filled(
    i: int,
    j: int,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    finite: np.ndarray,
    shading: np.ndarray,
    lit: np.ndarray,
    boundary: evaluate.Boundary | None,
    zspan: tuple[float, float],
) -> list[list[tuple[float, float, float, float]]]:
    """The marching-squares fill of one partial cell, as polygons in data space.

    The defined corners in perimeter order, with the refined boundary crossing
    inserted on every edge that straddles the domain: one, two or three defined
    corners give a triangle, a quadrilateral or a pentagon, and four give the
    whole quad for the clip to trim. The ambiguous cell - two defined corners
    on a diagonal - is resolved one fixed way, as two separate triangles each
    holding one corner, because a consistent boundary matters more than a guess
    at the local topology. A straddling edge the boundary has no crossing for
    leaves the cell unfilled, which is the un-refined behavior.
    """
    kept = [bool(finite[i + di, j + dj]) for di, dj in CELL]
    corner = [
        (
            float(xs[i + di]),
            float(ys[j + dj]),
            float(zs[i + di, j + dj]),
            float(shading[i + di, j + dj]),
        )
        for di, dj in CELL
    ]
    crossing: list[tuple[float, float, float, float] | None] = [None] * 4
    for k in range(4):
        if kept[k] != kept[(k + 1) % 4]:
            crossing[k] = _crossing(i, j, k, xs, ys, lit, boundary, zspan)
            if crossing[k] is None:
                return []
    if sum(kept) == 2 and kept[0] == kept[2]:
        first = 0 if kept[0] else 1
        return [
            [crossing[(c + 3) % 4], corner[c], crossing[c]]
            for c in (first, first + 2)
        ]
    polygon: list[tuple[float, float, float, float]] = []
    for k in range(4):
        if kept[k]:
            polygon.append(corner[k])
        if crossing[k] is not None:
            polygon.append(crossing[k])
    return [polygon] if len(polygon) >= 3 else []


def _crossing(
    i: int,
    j: int,
    k: int,
    xs: np.ndarray,
    ys: np.ndarray,
    lit: np.ndarray,
    boundary: evaluate.Boundary | None,
    zspan: tuple[float, float],
) -> tuple[float, float, float, float] | None:
    """The refined boundary point on edge `k` of cell `(i, j)`, with its shade.

    The point is off the grid, so its brightness cannot be read from the
    gradient: the height half of its shade is its own z's ramp, clipped to the
    box like any other, and the light half is interpolated along the edge it
    sits on, between the light values of the edge's two grid ends.
    """
    if boundary is None:
        return None
    if k % 2 == 0:  # an edge running along x, at y[j] or y[j+1]
        row = j if k == 0 else j + 1
        x = float(boundary.across[i, row])
        z = float(boundary.across_z[i, row])
        if not np.isfinite(x):
            return None
        t = (x - float(xs[i])) / (float(xs[i + 1]) - float(xs[i]))
        ends = (float(lit[i, row]), float(lit[i + 1, row]))
        y = float(ys[row])
    else:  # an edge running along y, at x[i+1] or x[i]
        column = i + 1 if k == 1 else i
        y = float(boundary.along[column, j])
        z = float(boundary.along_z[column, j])
        if not np.isfinite(y):
            return None
        t = (y - float(ys[j])) / (float(ys[j + 1]) - float(ys[j]))
        ends = (float(lit[column, j]), float(lit[column, j + 1]))
        x = float(xs[column])
    low, high = zspan
    light = (1.0 - t) * ends[0] + t * ends[1]
    ramp = min(max((z - low) / (high - low), 0.0), 1.0)
    shade = HEIGHT_SHARE * (DIM + (1.0 - DIM) * ramp) + (1.0 - HEIGHT_SHARE) * light
    return (x, y, z, shade)


def _trimmed(
    polygon: list[tuple[float, float, float, float]],
    low: float,
    high: float,
) -> list[tuple[float, float, float, float]]:
    """`polygon` cut to the box's z, crossings interpolated along its edges.

    The clip boundary is a known height, so the crossing is plain linear
    interpolation over every component - position and shade alike - and needs
    no evaluation at all, which is why it can run on the Qt thread every time
    the box moves. The vertices all sit on the rim of a convex cell, so what
    survives the cut is convex too and a fan can triangulate it.
    """
    for bound, sign in ((low, 1.0), (high, -1.0)):
        kept: list[tuple[float, float, float, float]] = []
        for at, point in enumerate(polygon):
            previous = polygon[at - 1]
            in_point = sign * (point[2] - bound) >= 0.0
            in_previous = sign * (previous[2] - bound) >= 0.0
            if in_point != in_previous:
                t = (bound - previous[2]) / (point[2] - previous[2])
                kept.append(
                    tuple((1.0 - t) * p + t * q for p, q in zip(previous, point))
                )
            if in_point:
                kept.append(point)
        polygon = kept
        if len(polygon) < 3:
            return []
    return polygon


def wire(
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
    box: Box,
    boundary: evaluate.Boundary | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One surface as the rectangular grid of its own samples, and its shading.

    The row and column lines of the sample arrays, one segment per grid step,
    for a `GLLinePlotItem` in `'lines'` mode: consecutive pairs of the points
    returned are segments, placed in the box, with a shade per point computed
    the way the solid's is. `drawEdges` would have been the wrong tool - the
    faces are triangles, so it draws every diagonal and yields a triangulated
    fabric rather than this grid.

    Only every k-th row and column is drawn, k chosen to leave about
    `WIRE_LINES` lines each way, while every drawn line keeps the full sample
    spacing - the thinning is of the drawing, not of the shape.

    A segment lives by the predicate that decides face survival: both ends
    real, finite and inside the box's z. So holes and clip edges appear in the
    wire exactly where they appear in the solid. Where one end is not real,
    the refined crossing `boundary` carries takes its place, ending the line
    on the boundary rather than a grid step short of it; where an end leaves
    the box's z, the segment is trimmed to the clip plane by the same
    interpolation that trims the faces, which is also what saves a boundary
    end whose limit is unbounded.
    """
    zs = np.asarray(values, dtype=np.float64)
    if zs.ndim != 2 or zs.shape[0] < 2 or zs.shape[1] < 2:
        return np.empty((0, 3), dtype=np.float32), np.empty(0)
    finite = np.isfinite(zs)
    low, high = box.z
    heights = np.clip((zs - low) / (high - low), 0.0, 1.0)
    lit = _lit(np.where(finite, box.up(zs), np.nan))
    shading = HEIGHT_SHARE * (DIM + (1.0 - DIM) * heights) + (1.0 - HEIGHT_SHARE) * lit
    points: list[tuple[float, float, float, float]] = []
    for i in _drawn(zs.shape[0]):
        for j in range(zs.shape[1] - 1):
            _strand(
                points, i, j, False, xs, ys, zs, finite, shading, lit, boundary, box.z
            )
    for j in _drawn(zs.shape[1]):
        for i in range(zs.shape[0] - 1):
            _strand(
                points, i, j, True, xs, ys, zs, finite, shading, lit, boundary, box.z
            )
    if not points:
        return np.empty((0, 3), dtype=np.float32), np.empty(0)
    data = np.array(points, dtype=np.float64)
    placed = np.stack(
        [box.across(data[:, 0]), box.along(data[:, 1]), box.up(data[:, 2])], axis=1
    )
    return placed.astype(np.float32), data[:, 3]


def _drawn(count: int) -> list[int]:
    """Which of `count` rows or columns the wire draws: every k-th, and the last.

    The step leaves about `WIRE_LINES` lines, and the last sample is always
    among them so the wire's rim is the grid's own rim on every side.
    """
    step = max(1, round((count - 1) / WIRE_LINES))
    picked = list(range(0, count, step))
    if picked[-1] != count - 1:
        picked.append(count - 1)
    return picked


def _strand(
    points: list[tuple[float, float, float, float]],
    i: int,
    j: int,
    across: bool,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    finite: np.ndarray,
    shading: np.ndarray,
    lit: np.ndarray,
    boundary: evaluate.Boundary | None,
    zspan: tuple[float, float],
) -> None:
    """One grid step of a wire line: append what survives of it to `points`.

    The segment from sample `(i, j)` to the next sample along x (`across`) or
    along y, in data coordinates with a shade per end. An end that is not real
    is replaced by the refined crossing on the edge - read through `_crossing`,
    against the cell the edge borders, so the wire's boundary points and their
    shades are exactly the mesh's - and a segment with no crossing to end on is
    dropped, which is the un-refined behavior. What remains is trimmed to the
    box's z.
    """
    if across:
        ends = (i, j), (i + 1, j)
        cell, k = ((i, j), 0) if j + 1 < zs.shape[1] else ((i, j - 1), 2)
    else:
        ends = (i, j), (i, j + 1)
        cell, k = ((i, j), 3) if i + 1 < zs.shape[0] else ((i - 1, j), 1)
    kept = bool(finite[ends[0]]), bool(finite[ends[1]])
    if not any(kept):
        return

    def corner(at: tuple[int, int]) -> tuple[float, float, float, float]:
        return (float(xs[at[0]]), float(ys[at[1]]), float(zs[at]), float(shading[at]))

    if all(kept):
        segment = corner(ends[0]), corner(ends[1])
    else:
        crossing = _crossing(cell[0], cell[1], k, xs, ys, lit, boundary, zspan)
        if crossing is None:
            return
        segment = (corner(ends[0]), crossing) if kept[0] else (crossing, corner(ends[1]))
    trimmed = _trimmed_segment(segment[0], segment[1], *zspan)
    if trimmed is not None:
        points.extend(trimmed)


def _trimmed_segment(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    low: float,
    high: float,
) -> tuple[tuple[float, float, float, float], ...] | None:
    """The part of the segment `a`-`b` inside the box's z, or None for none.

    The segment's own case of what `_trimmed` does for a polygon: the clip
    boundary is a known height, so the crossing is plain linear interpolation
    over every component, position and shade alike, and needs no evaluation.
    """
    first, last = 0.0, 1.0
    for bound, sign in ((low, 1.0), (high, -1.0)):
        into_a, into_b = sign * (a[2] - bound), sign * (b[2] - bound)
        if into_a < 0.0 and into_b < 0.0:
            return None
        if into_a < 0.0:
            first = max(first, into_a / (into_a - into_b))
        elif into_b < 0.0:
            last = min(last, into_a / (into_a - into_b))
    if last <= first:
        return None

    def between(t: float) -> tuple[float, float, float, float]:
        x, y, z, shade = ((1.0 - t) * p + t * q for p, q in zip(a, b))
        return (x, y, z, shade)

    return between(first), between(last)


def _no_faces() -> np.ndarray:
    return np.empty((0, 3), dtype=np.uint32)


def _lit(upright: np.ndarray) -> np.ndarray:
    """How much light each vertex catches, from the slope of the grid there.

    The normal of a height field is `(-dz/dx, -dz/dy, 1)` normalized, which is
    two differences and a square root over the array and needs no mesh at all -
    the reason this window computes its own lighting rather than asking for
    face normals it would then have to keep in step with the holes. The light
    is fixed in the world, so turning the picture moves the highlight across
    it, which is what a solid does.
    """
    step_x = WORLD / max(upright.shape[0] - 1, 1)
    step_y = WORLD / max(upright.shape[1] - 1, 1)
    slope_x = np.nan_to_num(np.gradient(upright, step_x, axis=0), nan=0.0)
    slope_y = np.nan_to_num(np.gradient(upright, step_y, axis=1), nan=0.0)
    slope_x = np.clip(slope_x, -1e6, 1e6)
    slope_y = np.clip(slope_y, -1e6, 1e6)
    facing = (-slope_x * LIGHT[0] - slope_y * LIGHT[1] + LIGHT[2]) / np.sqrt(
        slope_x**2 + slope_y**2 + 1.0
    )
    return AMBIENT + (1.0 - AMBIENT) * np.clip(facing, 0.0, 1.0)


def brightened(shading: np.ndarray, color: str) -> np.ndarray:
    """A surface's own color per vertex, scaled by how bright the vertex is."""
    tint = pg.mkColor(color)
    rgb = np.array(tint.getRgbF()[:3], dtype=np.float32)
    scale = np.clip(np.asarray(shading, dtype=np.float32), 0.0, 1.0)[:, None]
    colors = np.ones((int(np.size(shading)), 4), dtype=np.float32)
    colors[:, :3] = rgb * scale
    return colors


def ticks(low: float, high: float, count: int = TICKS) -> list[float]:
    """The round numbers to mark an axis at, about `count` of them.

    One, two, two and a half or five times a power of ten, which is what every
    ruler in the world is divided in and what a reader can add up in their head
    while looking at a picture.
    """
    span = float(high) - float(low)
    if not np.isfinite(span) or span <= 0:
        return []
    rough = span / max(count, 1)
    step = 10.0 ** np.floor(np.log10(rough))
    for multiple in (1.0, 2.0, 2.5, 5.0, 10.0):
        if step * multiple >= rough:
            step *= multiple
            break
    slack = step * 1e-6
    first = np.ceil((low - slack) / step) * step
    found = np.arange(first, high + slack, step)
    return [0.0 if abs(value) < slack else float(value) for value in found]


class View(gl.GLViewWidget):
    """The GL view, saying when the camera has moved and surviving no GL at all.

    Three departures from stock, all of them about what happens around the
    drawing rather than to it. The camera signal is what the tick labels
    re-anchor on: every stock gesture goes through `orbit`, `pan`,
    `setCameraPosition` or the wheel, so overriding those four is a complete
    account of the camera moving, and it is emitted after the move rather than
    during it.

    The second is the right button, which the stock view has no use for at all:
    it orbits on the left and pans on the middle, and a press of the right one
    is recorded and forgotten. So the button is free, and a press let go where
    it was pressed asks for the window's menu - the same click the 2D canvas
    answers, measured by the same slop, so that one hand opens either window's
    menu the same way.

    The third is the guard. `initializeGL` is where a machine with no usable
    OpenGL says so, and it runs inside the Qt event loop, where an exception
    ends the host and takes every other plot window with it. It is remembered
    instead, and the window reports it on its status bar.
    """

    moved = QtCore.Signal()

    #: A right click that stayed put, at the point on the screen it landed on.
    asked = QtCore.Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.broken = ""
        #: Where the right button went down, while it is down.
        self._pressed: Any = None

    def initializeGL(self) -> None:
        try:
            super().initializeGL()
        except Exception as error:
            self.broken = str(error).strip().splitlines()[0]

    def paintGL(self, *arguments: Any, **keywords: Any) -> None:
        if self.broken:
            return
        try:
            super().paintGL(*arguments, **keywords)
        except Exception as error:  # a card that gave out mid-frame
            self.broken = str(error).strip().splitlines()[0]

    def orbit(self, *arguments: Any, **keywords: Any) -> None:
        super().orbit(*arguments, **keywords)
        self.moved.emit()

    def pan(self, *arguments: Any, **keywords: Any) -> None:
        super().pan(*arguments, **keywords)
        self.moved.emit()

    def setCameraPosition(self, *arguments: Any, **keywords: Any) -> None:
        super().setCameraPosition(*arguments, **keywords)
        self.moved.emit()

    def wheelEvent(self, ev: Any) -> None:
        super().wheelEvent(ev)
        self.moved.emit()

    def mousePressEvent(self, ev: Any) -> None:
        """Stock, plus the beginning of a click that may be asking for the menu."""
        super().mousePressEvent(ev)
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            self._pressed = ev.position()

    def mouseReleaseEvent(self, ev: Any) -> None:
        """A right button let go where it went down is the menu being asked for.

        Let go anywhere else it is nothing, which is what a right drag is here
        already: the stock view moves the camera on the other two buttons, so
        nothing is being taken away from the mouse to pay for the menu.
        """
        super().mouseReleaseEvent(ev)
        if ev.button() != QtCore.Qt.MouseButton.RightButton or self._pressed is None:
            return
        moved = ev.position() - self._pressed
        self._pressed = None
        if float(np.hypot(moved.x(), moved.y())) < CLICK_SLOP_PX:
            self.asked.emit(ev.globalPosition().toPoint())

    def mouseMoveEvent(self, ev: Any) -> None:
        """Stock, plus Shift-drag as a second way to say pan.

        The stock view pans on Ctrl-drag and on the middle button; Shift is the
        modifier the rest of this program uses for "the other axis", and a hand
        already holding Shift for a 2D window should not have to learn a second
        key for the same gesture in a 3D one.
        """
        modifiers = ev.modifiers()
        shift = modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier
        control = modifiers & QtCore.Qt.KeyboardModifier.ControlModifier
        if shift and not control and ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
            here = ev.position()
            moved = here - getattr(self, "mousePos", here)
            self.mousePos = here
            self.pan(moved.x(), moved.y(), 0, relative="view")
            return
        super().mouseMoveEvent(ev)


class Window3D(QtWidgets.QMainWindow):
    """One top-level 3D plot window and everything that happens inside it."""

    def __init__(
        self, number: int, host: Any, *, grid: int = DEFAULT_GRID, wire: bool = True
    ) -> None:
        super().__init__()
        self.number = number
        self.kind = protocol.WindowKind.THREE_D
        self.host = host
        self.plots: list[Surface] = []
        self.current = False
        self.xdomain = DEFAULT_DOMAIN
        self.ydomain = DEFAULT_DOMAIN
        # The grid the window opens with is the sticky one the host holds -
        # the grid the last surface was given - clamped here as a typed one
        # is: a window never samples finer than it can draw, whoever asked.
        square = int(min(max(grid, 2), MAX_GRID))
        self.grid = (square, square)
        #: How a surface arriving here is drawn - the sticky look the last
        #: surface anywhere was left in, and thereafter whatever this window's
        #: own `mesh` box says.
        self.wired = bool(wire)
        #: The z range now drawn, and the one the inspector nailed down where
        #: somebody typed it - a typed extent is an answer and is not autoscaled
        #: away by the next surface.
        self.zrange = (-1.0, 1.0)
        #: What the three axes are called, which is the first surface's reading
        #: of the expression until there is one.
        self.axis_names = ("x", "y", "z")
        self._fixed: tuple[float, float] | None = None
        #: The parse state and context the domain fields read typed bounds
        #: under: the last added surface's, since the domain belongs to the
        #: window and the worksheet that last plotted into it is its reader.
        self._state = ParseState()
        self._context = Context()
        #: Which domain edit is the latest, so the numbers a superseded edit
        #: evaluated to cannot land on top of a newer one's.
        self._edit = 0
        self._counter = 0
        self._message = ""
        #: Which of the three furnishings are shown: the box and its ticks, the
        #: numbers and axis names, the legend. All on for a fresh window.
        self._boxed = True
        self._named = True
        self._legend = True
        self._papered = False
        self._inspector: Inspector | None = None
        self._build()
        self.retitle()
        self.home()

    # -- construction ------------------------------------------------------

    def _build(self) -> None:
        self.view = View()
        self.view.setBackgroundColor(BACKGROUND)
        # The keys belong to the window: the stock view has arrow bindings of
        # its own with a repeat timer, and ours orbit by a fixed step.
        self.view.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.view.moved.connect(self._anchor)
        self.view.asked.connect(self._menu_at)
        self.legend = Legend(self.view)
        self.legend.move(12, 12)
        self.legend.picked.connect(self._legend_picked)
        self._make_menu()
        self._furnish()
        self.setCentralWidget(self._laid_out())
        self._spin = QtCore.QTimer(self)
        self._spin.setInterval(SPIN_MS)
        self._spin.timeout.connect(lambda: self.view.orbit(-SPIN_DEGREES, 0.0))
        self.resize(760, 620)

    def _laid_out(self) -> QtWidgets.QWidget:
        """The view between a toolbar and a status line.

        The toolbar holds the two things that change what was computed - the
        rectangle and how finely it is sampled - the `mesh` box that draws
        every surface as the wire grid of its samples, the door to the numbers
        behind the picture, and the clear that starts the picture over.
        Everything else about a 3D window is the camera, and the camera is
        the mouse.
        """
        holder = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toolbar())
        layout.addWidget(self.view, 1)
        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet(f"color: {theme.STATUS_TEXT}; background: transparent;")
        line = QtWidgets.QFrame()
        line.setObjectName("statusline")
        line.setStyleSheet(
            f"QFrame#statusline {{ background: {theme.STATUS};"
            f" border-top: 1px solid {theme.STATUS_EDGE}; }}"
        )
        across = QtWidgets.QHBoxLayout(line)
        across.setContentsMargins(10, 3, 10, 3)
        across.addWidget(self.status, 1)
        layout.addWidget(line)
        return holder

    def _toolbar(self) -> QtWidgets.QWidget:
        """The domain, the grid, and the controls the picture itself answers to.

        The fields stand in three groups with a hairline between them, since a
        domain in x, a domain in y and a grid are three answers rather than six
        numbers in a row.
        """
        bar = QtWidgets.QToolBar()
        bar.setMovable(False)
        self.fields: dict[str, QtWidgets.QLineEdit] = {}
        bar.addWidget(QtWidgets.QLabel("x:"))
        bar.addWidget(self._field("x0", self.xdomain[0]))
        bar.addWidget(QtWidgets.QLabel(" … "))
        bar.addWidget(self._field("x1", self.xdomain[1]))
        bar.addWidget(theme.divider())
        bar.addWidget(QtWidgets.QLabel("y:"))
        bar.addWidget(self._field("y0", self.ydomain[0]))
        bar.addWidget(QtWidgets.QLabel(" … "))
        bar.addWidget(self._field("y1", self.ydomain[1]))
        bar.addWidget(theme.divider())
        bar.addWidget(QtWidgets.QLabel("grid:"))
        bar.addWidget(self._field("nx", self.grid[0]))
        bar.addWidget(QtWidgets.QLabel(" x "))
        bar.addWidget(self._field("ny", self.grid[1]))
        bar.addWidget(theme.divider())
        self.mesh_action = QtGui.QAction("mesh", self)
        self.mesh_action.setCheckable(True)
        self.mesh_action.setChecked(self.wired)
        self.mesh_action.setToolTip(
            "Draw every surface as the wire grid of its samples (M)"
        )
        self.mesh_action.triggered.connect(self._mesh_mode)
        bar.addAction(self.mesh_action)
        self.inspect = QtGui.QAction("view...", self)
        self.inspect.setToolTip("The box and the camera as numbers")
        self.inspect.triggered.connect(self.inspector)
        bar.addAction(self.inspect)
        self.clear_action = QtGui.QAction("clear", self)
        self.clear_action.setToolTip("Remove every surface from this window (Del)")
        self.clear_action.triggered.connect(self.clear)
        bar.addAction(self.clear_action)
        theme.dangerous(bar, self.clear_action)
        return bar

    def _field(self, name: str, value: float) -> QtWidgets.QLineEdit:
        """One toolbar number, which the keyboard only reaches when clicked in.

        The keys of this window are the camera's - `1` faces the xy plane - and
        a field that took the focus when the window opened would swallow them.
        So the fields are click-only, and finishing an edit hands the keyboard
        straight back.
        """
        edit = QtWidgets.QLineEdit(_written(value))
        edit.setFixedWidth(52)
        edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        edit.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        edit.editingFinished.connect(self._edited)
        self.fields[name] = edit
        return edit

    def _furnish(self) -> None:
        """The box, the axis rays, the tick marks and the pool of text items.

        Made once and never added to, because re-anchoring happens while the
        view is live and a view's item list must not be edited from under it. A
        label with nothing to say is set to the empty string instead of being
        taken away.
        """
        self.box = gl.GLBoxItem(
            size=QtGui.QVector3D(WORLD, WORLD, WORLD), color=BOX_COLOR
        )
        self.box.translate(-HALF, -HALF, -HALF)
        self.view.addItem(self.box)
        self.rays = gl.GLAxisItem(size=QtGui.QVector3D(HALF, HALF, HALF))
        self.view.addItem(self.rays)
        self.marks = gl.GLLinePlotItem(
            pos=np.zeros((0, 3), dtype=np.float32), mode="lines", antialias=True
        )
        self.view.addItem(self.marks)
        self.labels = [gl.GLTextItem(text="") for _ in range(3 * MAX_TICKS)]
        self.names = [gl.GLTextItem(text="") for _ in range(3)]
        for item in self.labels + self.names:
            item.setData(font=QtGui.QFont("Helvetica", 10))
            # A number is drawn with a painter and not with the depth buffer, so
            # it is only readable if it is drawn last: the view draws in order
            # of depth value, and a surface added later would otherwise be
            # painted over the numbers naming the edge behind it.
            item.setDepthValue(10)
            self.view.addItem(item)

    def _make_menu(self) -> None:
        """The window's own context menu: what it does, and the keys that do it.

        The GL view has no menu of its own to inherit or to prune, so this one
        is the whole of what a right click on the picture offers. It is the
        camera's list, because a 3D window is a camera: where to look from, and
        the three presets that put an axis edge-on. What is on the toolbar is
        not repeated here - the `mesh` box says its own state where it stands,
        and a menu entry that had to be opened to be read would say it worse.

        The surfaces are listed under `Remove` when the menu opens rather than
        when it is built, since what the window holds changes and the menu does
        not; the submenu of an empty window is offered greyed rather than left
        off, because what it would list is what the window is empty of.
        """
        #: Every key the menu advertises, by the stroke that presses it. The
        #: table `keyPressEvent` dispatches from.
        self._keyed: dict[tuple[int, int], Any] = {}
        self.menu = QtWidgets.QMenu(self)
        self.menu.aboutToShow.connect(self._read_menu)
        self._command("Home view", self.home, ("Home", "0"))
        # The three presets in the order `PLANES` holds them, which is the order
        # the number keys have always been in.
        for plane, key in zip(PLANES, ("1", "2", "3")):
            self._command(
                f"Face the {plane} plane",
                lambda _=False, plane=plane: self.face(plane),
                (key,),
            )
        self._spin_entry = self._command("Rotate", self.spin, ("R",), checkable=True)
        self.menu.addSeparator()
        self._command("View...", self.inspector)
        self._command("Copy image", self.copy_image, ("Ctrl+C",))
        self._command("Export...", self.export, ("Ctrl+S",))
        self.menu.addSeparator()
        self._remove_menu = self.menu.addMenu("Remove")
        self._command("Clear", self.clear, ("Del",))
        self._command("Close", self.close, ("Q", "Ctrl+W"))

    def _command(
        self,
        text: str,
        handler: Callable[..., None],
        keys: Sequence[str] = (),
        checkable: bool = False,
    ) -> Any:
        """One entry of the menu, its keys written on it and in the key table."""
        action = commanded(self, self._keyed, text, handler, keys, checkable)
        self.menu.addAction(action)
        return action

    def _read_menu(self) -> None:
        """Read the menu off the window as it opens: what turns, and what is in it.

        The rotation is a timer and says for itself whether it is running, which
        is what the tick is drawn from; a tick remembering what the menu last
        did would be wrong the first time `R` was pressed instead.
        """
        self._spin_entry.setChecked(self._spin.isActive())
        self._remove_menu.clear()
        for surface in self.plots:
            self._remove_menu.addAction(
                surface.named, lambda _=False, surface=surface: self.remove(surface)
            )
        self._remove_menu.setEnabled(bool(self.plots))

    def _menu_at(self, point: Any) -> None:
        """A right click on the picture: the window's menu, where it was asked for."""
        self.menu.popup(point)

    # -- the plot list -----------------------------------------------------

    def add(self, surface: Surface) -> None:
        """Put a surface in the window, replacing one with the same identity."""
        existing = self.find(surface.worksheet, surface.label)
        if existing is not None:
            # A replacement keeps the look of what it replaces - the color and
            # the wire choice - so re-plotting does not reshuffle the picture.
            surface.color, surface.paper = existing.color, existing.paper
            surface.wire = existing.wire
            self.remove(existing)
        else:
            index = self._counter % len(SOLID_PALETTE)
            surface.color, surface.paper = SOLID_PALETTE[index], SOLID_PAPER[index]
            self._counter += 1
            surface.wire = self.wired
        surface.item = gl.GLMeshItem(
            smooth=True, computeNormals=False, shader=None, glOptions="opaque"
        )
        # Hidden until there is a mesh in it: an item with no faces is asked to
        # draw itself on the next frame, and the sampling has not answered yet.
        surface.item.setVisible(False)
        self.view.addItem(surface.item)
        # The wire drawing of the same samples, shown over the solid while the
        # surface's `wire` is on. Opaque rather than blended, so the depth
        # buffer sorts the wire against the solids beside it - its own among
        # them - the same way it sorts two solids. It draws at the curve's
        # weight, best-effort: a core forward-compatible GL profile refuses
        # line widths other than one, and pyqtgraph skips the width call there.
        surface.wires = gl.GLLinePlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            mode="lines",
            width=CURVE_WIDTH,
            antialias=True,
            glOptions="opaque",
        )
        surface.wires.setVisible(False)
        self.view.addItem(surface.wires)
        self._state, self._context = surface.state, surface.context
        self.plots.append(surface)
        self._relabel()
        self.retitle()
        self._name_axes()
        self._start(surface)
        if self.view.broken:
            self._quiet()

    def find(self, worksheet: int, label: str) -> Surface | None:
        """The surface a worksheet and a label name, if this window has it."""
        for surface in self.plots:
            if surface.worksheet == worksheet and surface.label == label:
                return surface
        return None

    def remove(self, surface: Surface) -> None:
        """Take one surface out of the window, legend row and all."""
        if surface.item is not None:
            self.view.removeItem(surface.item)
            surface.item = None
        if surface.wires is not None:
            self.view.removeItem(surface.wires)
            surface.wires = None
        if surface in self.plots:
            self.plots.remove(surface)
        self._relabel()
        self.retitle()
        self._name_axes()
        self._frame()

    def clear(self) -> None:
        """The toolbar's clear: start this picture over.

        The window it acts on is the window the button is drawn in, so there
        is nothing to infer and nothing to report.
        """
        for surface in list(self.plots):
            self.remove(surface)

    def _relabel(self) -> None:
        """Build the legend again, which is how a removal leaves it right."""
        self.legend.rebuild(
            [
                (
                    surface.named,
                    surface.paper if self._papered else surface.color,
                    surface.hidden,
                )
                for surface in self.plots
            ],
            paper=self._papered,
        )
        # An empty legend is a rectangle over the corner of the picture saying
        # nothing; the toggle is remembered rather than read off the widget, so
        # a legend the user put away stays away when the next surface arrives.
        self.legend.setVisible(self._legend and bool(self.plots))

    def _name_axes(self) -> None:
        """What the three axes are called, taken from the first surface there is.

        The names are the expression's own - the two floor variables in the
        engine's canonical order, and the vertical one where an equation named
        it. A window holding two surfaces of the same two variables is the
        ordinary case, and a window holding two of different ones has to pick:
        the first surface named the axes and the others are drawn over them.
        """
        surface = self.plots[0] if self.plots else None
        if surface is None:
            self.axis_names = ("x", "y", "z")
        else:
            self.axis_names = (*surface.axes, surface.vertical)
        self._anchor()

    # -- evaluation --------------------------------------------------------

    def _start(self, surface: Surface) -> None:
        """Ask the sampling thread for this surface over the domain and grid.

        The only thing that starts an evaluation. A camera move does not come
        near here, which is the promise the window is built on: the mesh is
        about the domain, and the domain is only ever changed by typing in it.
        """
        surface.generation += 1
        generation = surface.generation
        self.host.sample(
            (self.number, id(surface)),
            self._work(surface),
            lambda answer: self._sampled(surface, generation, answer),
        )

    def _work(self, surface: Surface) -> Callable[..., Any]:
        """What the sampling thread is to do, as one callable over nothing else.

        Everything it needs is read here, on the Qt thread, and captured: by
        the time it runs the domain may have been typed over and the surface
        may have been removed, and a job that reached into the window would
        find either.
        """
        node, context, names = surface.node, surface.context, surface.axes
        made = surface.closure
        xrange, yrange = self.xdomain, self.ydomain
        nx, ny = self.grid

        def work(_report: Callable[..., None]) -> Any:
            f = made
            if f is None:
                f = evaluate.closure(node, context, names)
            xs, ys, values = evaluate.grid_eval(f, xrange, yrange, nx, ny)
            boundary = evaluate.grid_boundary(f, xs, ys, values)
            return f, xs, ys, values, boundary

        return work

    def _sampled(self, surface: Surface, generation: int, answer: Any) -> None:
        """The grid is in: rebuild the box around it, and say what it holds."""
        if surface.generation != generation or surface not in self.plots:
            return
        if isinstance(answer, Exception):
            surface.trouble = str(answer)
            self.host.trouble(self.number, surface.label, str(answer))
            self.say(f"{surface.label}: {answer}")
            return
        (
            surface.closure,
            surface.xs,
            surface.ys,
            surface.values,
            surface.boundary,
        ) = answer
        # Whatever the window was saying was about the domain this answer has
        # replaced; the two sentences below are what it has to say about the
        # new one, and they are said in the order they were arrived at.
        self._quiet()
        self._frame()
        if evaluate.finite_fraction(surface.values) == 0.0:
            self.say(f"{surface.label}: no real values over this domain")

    def reevaluate(self) -> None:
        """Every surface again, which is what a new domain or grid asks for."""
        for surface in self.plots:
            self._start(surface)

    # -- the box -----------------------------------------------------------

    def _frame(self) -> None:
        """Work out the z the box stands for, and build every mesh to it.

        One box for the window rather than one per surface: two surfaces in the
        same picture are being compared, and comparing them through two
        different vertical scales would be a picture that lies. So a surface
        arriving rebuilds the others, which costs a mesh each and no evaluation
        at all.
        """
        shown = [
            surface.values
            for surface in self.plots
            if surface.visible and surface.values.size
        ]
        found = extent(shown)
        clipped = False
        if self._fixed is not None:
            self.zrange = self._fixed
        elif found is not None:
            self.zrange, clipped = found
        self._reshape()
        for surface in self.plots:
            self._draw(surface)
        self._anchor()
        if clipped:
            low, high = self.zrange
            self.say(
                "z clipped to the 1st-99th percentile:"
                f" {_written(low)} to {_written(high)}"
            )

    @property
    def box_now(self) -> Box:
        """The three ranges the picture stands for, as one thing to read them from."""
        return Box(self.xdomain, self.ydomain, self.zrange)

    def _reshape(self) -> None:
        """Stand the box and the axis rays up to the height the data asks for.

        The rays start where the data's origin is, when the box holds it, which
        is the same convention the 2D window draws its axes by: an axis belongs
        at zero, and a picture framed away from zero shows it at the edge
        instead of not at all.
        """
        box = self.box_now
        height = box.height
        self.box.setSize(WORLD, WORLD, height)
        self.box.resetTransform()
        self.box.translate(-HALF, -HALF, -height / 2)
        origin = (
            float(np.clip(box.across(0.0), -HALF, HALF)),
            float(np.clip(box.along(0.0), -HALF, HALF)),
            float(np.clip(box.up(0.0), -height / 2, height / 2)),
        )
        self.rays.setSize(HALF - origin[0], HALF - origin[1], height / 2 - origin[2])
        self.rays.resetTransform()
        self.rays.translate(*origin)

    def _draw(self, surface: Surface) -> None:
        """Build one surface's triangles - and its wire, where it wears one.

        A wire surface has its hidden lines removed, the way Derive's plotter
        removed them: what is behind the shape is behind it, and the grid reads
        as a body turned in the light rather than as two grids laid over each
        other. Both items draw for it. The solid keeps the triangles it always
        had - the same ones, stopping at the same rim - painted in the canvas's
        own color, so it is nothing to look at and everything to hide behind;
        the lines draw over it and the depth buffer does the removing, per
        pixel, for the box frame and the other surfaces too.

        The polygon offset is what makes that a wire and not a stitch. The
        lines lie on the very faces they are being tested against, so the
        occluder is pushed a hair away from the camera and the lines win the
        ties.
        """
        if surface.item is None:
            return
        color = surface.paper if self._papered else surface.color
        vertexes, faces, shading = mesh(
            surface.xs, surface.ys, surface.values, self.box_now, surface.boundary
        )
        if surface.wire:
            points, shades = wire(
                surface.xs, surface.ys, surface.values, self.box_now, surface.boundary
            )
            surface.wires.setData(pos=points, color=brightened(shades, color))
            surface.wires.setVisible(surface.visible and bool(points.size))
            surface.item.setGLOptions(WIRE_OCCLUDER)
            # An occluder is a shape and not a picture, so it is given no vertex
            # colors at all: those are what the card reads where a mesh has
            # them, and the one flat color - the canvas behind it, whichever
            # canvas is up - only reaches the fragments in their absence.
            surface.item.setColor(pg.mkColor("w" if self._papered else BACKGROUND))
            if faces.size:
                surface.item.setMeshData(vertexes=vertexes, faces=faces)
            surface.item.setVisible(surface.visible and bool(faces.size))
            return
        surface.wires.setVisible(False)
        surface.item.setGLOptions("opaque")
        if not faces.size:
            surface.item.setVisible(False)
            return
        surface.item.setMeshData(
            vertexes=vertexes, faces=faces, vertexColors=brightened(shading, color)
        )
        surface.item.setVisible(surface.visible)

    def _anchor(self) -> None:
        """Put the tick marks and the numbers on the box edges nearest the camera.

        Three edges carry them: the two bottom edges of the side the camera is
        on, which is where a number sits in front of the picture rather than
        behind it, and for the upright axis the far vertical edge, which is the
        one the surface does not stand in front of. Which edges those are
        changes as the view turns, so this runs on every camera move; it is a
        few dozen positions and no geometry at all.

        An axis pointing at the camera is left unnumbered. Facing the xy plane
        makes the whole z axis one point of the screen, and five numbers stacked
        on that point are five numbers about nothing; the axis keeps its name,
        which is all there is to say about it from there.
        """
        box = self.box_now
        floor = -box.height / 2
        camera = self.view.cameraPosition()
        head_on = self._head_on()
        near_y = -HALF if camera.y() < 0 else HALF
        near_x = -HALF if camera.x() < 0 else HALF
        far_x, far_y = -near_x, -near_y
        out_y, out_x = np.sign(near_y), np.sign(near_x)
        segments: list[tuple[float, float, float]] = []
        written: list[tuple[tuple[float, float, float], str]] = []

        for value in self._ticks(self.xdomain) if not head_on[0] else ():
            at = float(box.across(value))
            segments += [
                (at, near_y, floor),
                (at, near_y + out_y * TICK_OUT, floor),
            ]
            written.append(((at, near_y + out_y * LABEL_OUT, floor), _written(value)))
        for value in self._ticks(self.ydomain) if not head_on[1] else ():
            at = float(box.along(value))
            segments += [
                (near_x, at, floor),
                (near_x + out_x * TICK_OUT, at, floor),
            ]
            written.append(((near_x + out_x * LABEL_OUT, at, floor), _written(value)))
        upright = np.sign(far_x) * 0.7, np.sign(far_y) * 0.7
        for value in self._ticks(self.zrange) if not head_on[2] else ():
            at = float(box.up(value))
            segments += [
                (far_x, far_y, at),
                (far_x + upright[0] * TICK_OUT, far_y + upright[1] * TICK_OUT, at),
            ]
            written.append(
                (
                    (
                        far_x + upright[0] * LABEL_OUT,
                        far_y + upright[1] * LABEL_OUT,
                        at,
                    ),
                    _written(value),
                )
            )

        self.marks.setData(
            pos=np.array(segments, dtype=np.float32).reshape(-1, 3),
            color=pg.glColor(PAPER_TICK if self._papered else TICK_COLOR),
            width=1,
        )
        ink = PAPER_TEXT if self._papered else TEXT_COLOR
        for index, item in enumerate(self.labels):
            if index < len(written) and self._named:
                where, text = written[index]
                item.setData(pos=np.array(where, dtype=np.float64), text=text, color=ink)
            else:
                item.setData(text="")
        places = (
            (0.0, near_y + out_y * NAME_OUT, floor),
            (near_x + out_x * NAME_OUT, 0.0, floor),
            (
                far_x + upright[0] * NAME_OUT,
                far_y + upright[1] * NAME_OUT,
                -floor * 0.9,
            ),
        )
        for item, where, name in zip(self.names, places, self.axis_names):
            item.setData(
                pos=np.array(where, dtype=np.float64),
                text=name if self._named else "",
                color=ink,
            )

    def _head_on(self) -> tuple[bool, bool, bool]:
        """Which axes point so nearly at the camera that they have no length.

        The camera looks along the line from itself to the box's center, so an
        axis is edge-on to the picture exactly when it is parallel to that line.
        The three coordinate presets each put one axis in that position, which
        is why the test is worth making rather than assuming a general view.
        """
        camera = self.view.cameraPosition()
        center = self.view.cameraParams()["center"]
        away = np.array(
            [
                float(camera.x() - center.x()),
                float(camera.y() - center.y()),
                float(camera.z() - center.z()),
            ]
        )
        length = float(np.linalg.norm(away))
        if not length:
            return (False, False, False)
        near = [abs(value) / length > EDGE_ON for value in away]
        return near[0], near[1], near[2]

    def _ticks(self, span: tuple[float, float], count: int = TICKS) -> list[float]:
        """The round numbers of one axis, never more than the label pool holds."""
        return ticks(span[0], span[1], count)[:MAX_TICKS]

    # -- the domain and the grid -------------------------------------------

    def _edited(self) -> None:
        """A toolbar field was left: take what it says, or put back the old text.

        The one place in this window where typing changes what is computed. The
        domain fields read expressions rather than floats - `-π` and `2π` are
        answers - parsed here under the parse state the last surface arrived
        with; what the trees are worth is arithmetic, so they go to the
        sampling thread and the domain moves when the numbers come back. The
        grid is a pair of counts and stays one. A field that does not read is
        reverted rather than argued with, since the value it would take is on
        the screen beside it.
        """
        try:
            grid = (int(self._read("nx")), int(self._read("ny")))
        except ValueError:
            self._show_domain()
            self.say("The grid is a pair of numbers")
            return
        grid = (
            int(min(max(grid[0], 2), MAX_GRID)),
            int(min(max(grid[1], 2), MAX_GRID)),
        )
        try:
            bounds = tuple(
                parse_expression(self.fields[name].text().strip(), self._state).node
                for name in ("x0", "x1", "y0", "y1")
            )
        except DeriveSyntaxError:
            self._show_domain()
            self.say("The domain bounds are expressions, like -π or 2π")
            return
        self._edit += 1
        edit = self._edit
        context = self._context
        known = (*self.xdomain, *self.ydomain)

        def work(_report: Callable[..., None]) -> Any:
            return tuple(
                evaluate.number(node, context, default)
                for node, default in zip(bounds, known)
            )

        self.host.sample(
            (self.number, "domain"),
            work,
            lambda answer: self._reframe_typed(edit, grid, answer),
        )

    def _reframe_typed(
        self, edit: int, grid: tuple[int, int], answer: Any
    ) -> None:
        """The typed bounds are worth numbers: take them, or put back the old ones.

        A bound that would not evaluate came back as the value it was replacing,
        so a field of nonsense reverts by arithmetic rather than by a case; an
        inverted domain is the one refusal left to make here.
        """
        if edit != self._edit or isinstance(answer, Exception):
            return
        x0, x1, y0, y1 = answer
        if x1 <= x0 or y1 <= y0:
            self._show_domain()
            self.say("The domain runs from a lower bound to a higher one")
            return
        if grid != self.grid:
            # A typed grid is sticky: the next surface window opens on it. The
            # sticky value is one count per axis, so a rectangular grid hands
            # on its finer axis. The domain is not sticky - it is a framing,
            # like a 2D view - so only the grid goes back.
            self.host.adjusted(grid=max(grid))
        changed = ((x0, x1), (y0, y1), grid) != (self.xdomain, self.ydomain, self.grid)
        self.xdomain, self.ydomain, self.grid = (x0, x1), (y0, y1), grid
        self._show_domain()
        if changed:
            self.say(f"Evaluating over the new domain at {grid[0]} by {grid[1]}")
            self.reevaluate()

    def _read(self, name: str) -> float:
        return float(self.fields[name].text().strip())

    def _show_domain(self) -> None:
        """Put the window's own numbers back in the fields, however they got there."""
        for name, value in (
            ("x0", self.xdomain[0]),
            ("x1", self.xdomain[1]),
            ("y0", self.ydomain[0]),
            ("y1", self.ydomain[1]),
            ("nx", self.grid[0]),
            ("ny", self.grid[1]),
        ):
            self.fields[name].setText(_written(value))

    def reframe(
        self,
        xdomain: tuple[float, float],
        ydomain: tuple[float, float],
        zrange: tuple[float, float],
    ) -> None:
        """The inspector's answer: a box typed rather than arrived at.

        A typed z extent is an opinion and outranks the autoscale from then on -
        somebody who asked for -1 to 1 wants to see the surface cut off at 1,
        and a picture that reframed itself on the next plot would be answering a
        question nobody asked. The extent that was already on the screen is not
        such an opinion, though: applying a change of camera must not quietly
        freeze the z the data chose, so an unedited field changes nothing.
        """
        moved = (xdomain, ydomain) != (self.xdomain, self.ydomain)
        self.xdomain, self.ydomain = xdomain, ydomain
        self._show_domain()
        if zrange[1] > zrange[0] and not _alike(zrange, self.zrange):
            self._fixed = zrange
        if moved:
            self.reevaluate()
        self._frame()

    def autoscale_z(self) -> None:
        """Give the z extent back to the data, undoing a typed one."""
        self._fixed = None
        self._frame()

    # -- the camera --------------------------------------------------------

    def home(self) -> None:
        """The default camera: a three-quarter view with the whole cube in it."""
        self.view.setCameraPosition(
            pos=pg.Vector(0.0, 0.0, 0.0),
            distance=CAMERA_DISTANCE,
            elevation=CAMERA_ELEVATION,
            azimuth=CAMERA_AZIMUTH,
        )

    def face(self, plane: str) -> None:
        """`1`, `2`, `3`: look straight at one of the three coordinate planes."""
        elevation, azimuth = PLANES[plane]
        self.view.setCameraPosition(elevation=elevation, azimuth=azimuth)
        self.say(f"Facing the {plane} plane")

    def spin(self) -> None:
        """`R`: turn the picture slowly, or stop turning it."""
        if self._spin.isActive():
            self._spin.stop()
            self._quiet()
            return
        self._spin.start()
        self.say("Rotating - R stops it")

    def inspector(self) -> None:
        """The `view...` dialog: the box and the camera as editable numbers."""
        if self._inspector is None:
            self._inspector = Inspector(self)
        self._inspector.refresh()
        self._inspector.show()
        self._inspector.raise_()

    # -- the legend --------------------------------------------------------

    def _legend_picked(self, row: int, button: Any) -> None:
        """A click on a legend row: hide and show, or offer to remove."""
        if not 0 <= row < len(self.plots):
            return
        surface = self.plots[row]
        if button == QtCore.Qt.MouseButton.RightButton:
            menu = QtWidgets.QMenu(self)
            menu.addAction(
                "Draw solid" if surface.wire else "Draw as wire mesh",
                lambda: self.toggle_wire(surface),
            )
            menu.addAction(f"Remove {surface.named}", lambda: self.remove(surface))
            menu.exec(QtGui.QCursor.pos())
            return
        surface.hidden = not surface.hidden
        self._relabel()
        # The box is recomputed over what is visible now, and redrawing every
        # surface to it is also what puts the items of the surface that was
        # clicked away or back.
        self._frame()

    def _mesh_mode(self, checked: bool) -> None:
        """The toolbar's `mesh` box: every surface as wire, or every one solid.

        The property belongs to the surfaces - the legend's right-click
        carries the per-surface exception - so the box flips them all rather
        than holding a state of its own. The look it leaves is the sticky one:
        the next surface, here or in the next window, arrives drawn this way.
        """
        self.wired = bool(checked)
        for surface in self.plots:
            surface.wire = self.wired
            self._draw(surface)
        self.host.adjusted(wire=self.wired)

    def toggle_wire(self, surface: Surface) -> None:
        """One surface between wire and solid: the legend's per-surface override.

        An exception rather than a default, so it hands nothing back to the
        sticky preferences and leaves the toolbar box - the every-surface
        control - where it was.
        """
        surface.wire = not surface.wire
        self._draw(surface)

    # -- export ------------------------------------------------------------

    def copy_image(self) -> None:
        """Ctrl-C: the picture on the clipboard, on paper colors."""
        image = self._photograph()
        if image is None:
            return
        QtWidgets.QApplication.clipboard().setImage(image)
        self.say("Copied the plot to the clipboard")

    def export(self) -> None:
        """Ctrl-S: the picture in a file, on paper colors.

        pyqtgraph's exporters know nothing about a GL view, so this is the
        frame buffer, a file dialog and a PNG - which is every image a 3D plot
        has to give. The colors are the same paper colors the 2D window
        exports on, for the same reason: a dark picture pasted into a document
        is a black rectangle.
        """
        name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save the plot", f"plot{self.number}.png", "PNG image (*.png)"
        )
        if not name:
            return
        image = self._photograph()
        if image is None:
            return
        image.save(name)
        self.say(f"Saved {name}")

    def _photograph(self) -> Any:
        """The view as an image, taken on a white background and put back after.

        `readQImage` is the frame buffer and nothing else: the legend is a Qt
        widget floating over the view, not an item in it, so it would be missing
        from a picture of two surfaces - which is a picture that does not say
        which is which. It is written onto the image afterwards, in the same
        paper colors the surfaces took.
        """
        if self.view.broken:
            self.say(NO_OPENGL.format(reason=self.view.broken))
            return None
        with _on_paper(self):
            image = self.view.readQImage()
            self._name_surfaces(image)
        return image

    def _name_surfaces(self, image: Any) -> None:
        """Write the plot list into the corner of an exported picture."""
        if not self._legend or not self.plots:
            return
        painter = QtGui.QPainter(image)
        # The frame buffer is in device pixels and the widget in logical ones,
        # so a picture on a doubled display is twice the size of the window.
        painter.scale(*([image.height() / max(self.view.height(), 1)] * 2))
        painter.setFont(QtGui.QFont("Helvetica", 9))
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
        for index, surface in enumerate(self.plots):
            painter.setPen(pg.mkColor(surface.paper))
            painter.drawText(16, 20 + index * 14, surface.named)
        painter.end()

    # -- keys --------------------------------------------------------------

    def keyPressEvent(self, ev: Any) -> None:
        """The keys of this window, the menu's own among them.

        Every key a menu entry advertises is dispatched off the menu's own
        table, so the key that works and the key the menu names are one key and
        neither can fire twice. What is left written out here are the keys no
        entry has: the three furnishings, the `mesh` box the toolbar already
        shows the state of, and the arrows, which are the camera's.
        """
        key = ev.key()
        keys = QtCore.Qt.Key
        command = self._keyed.get(pressed(ev))
        if command is not None:
            command.trigger()
        elif key == keys.Key_B:
            self._boxed = not self._boxed
            for item in (self.box, self.rays, self.marks):
                item.setVisible(self._boxed)
        elif key == keys.Key_X:
            self._named = not self._named
            self._anchor()
        elif key == keys.Key_L:
            self._legend = not self._legend
            self.legend.setVisible(self._legend and bool(self.plots))
        elif key == keys.Key_M:
            # M rather than W, deliberately: Ctrl-W closes the window, and a
            # frequent toggle one slipped modifier from destroying the picture
            # would be a poor place for it.
            self.mesh_action.trigger()
        elif key == keys.Key_Left:
            self.view.orbit(ORBIT_DEGREES, 0.0)
        elif key == keys.Key_Right:
            self.view.orbit(-ORBIT_DEGREES, 0.0)
        elif key == keys.Key_Up:
            self.view.orbit(0.0, ORBIT_DEGREES)
        elif key == keys.Key_Down:
            self.view.orbit(0.0, -ORBIT_DEGREES)
        else:
            super().keyPressEvent(ev)
            return
        ev.accept()

    # -- the rest ----------------------------------------------------------

    def say(self, message: str) -> None:
        """Put one line on the status bar, which is the window's whole voice."""
        self._message = message
        self.status.setText(message)

    def _quiet(self) -> None:
        """Nothing to say - unless there is no picture, which always wants saying."""
        self.say(NO_OPENGL.format(reason=self.view.broken) if self.view.broken else "")

    def describe(self) -> protocol.WindowInfo:
        """This window as `Describe` reports it.

        The ranges reported are the domain, since that is what a 3D window is
        framed by.
        """
        return protocol.WindowInfo(
            number=self.number,
            kind=self.kind,
            title=self.windowTitle(),
            current=self.current,
            plots=tuple(
                protocol.PlotInfo(
                    worksheet=surface.worksheet,
                    label=surface.label,
                    text=surface.text,
                    kind=surface.kind,
                    hidden=surface.hidden,
                )
                for surface in self.plots
            ),
            xrange=(float(self.xdomain[0]), float(self.xdomain[1])),
            yrange=(float(self.ydomain[0]), float(self.ydomain[1])),
        )

    def retitle(self, current: bool | None = None) -> None:
        """Title the window by what it holds, and say whether the next plot lands here.

        Called with no argument when the plot list changes, so the title tracks
        the contents while the receiver mark stays as it was.
        """
        if current is not None:
            self.current = current
        self.setWindowTitle(
            protocol.titled(
                self.kind,
                tuple(surface.text or surface.label for surface in self.plots),
                self.current,
            )
        )

    def changeEvent(self, ev: Any) -> None:
        """Activation is the user touching this window, and the host's to know.

        The receiver of the next plot follows the window the user last
        touched, and the activation event is how a click, a raise or an
        alt-tab says so - whatever the platform's focus policy, including one
        that hands focus to whatever the pointer crosses.
        """
        if (
            ev.type() == QtCore.QEvent.Type.ActivationChange
            and self.isActiveWindow()
        ):
            self.host.touched(self.number)
        super().changeEvent(ev)

    def closeEvent(self, ev: Any) -> None:
        """The window manager's business, and the host's to hear about."""
        self._spin.stop()
        if self._inspector is not None:
            self._inspector.close()
        self.host.closed(self.number)
        super().closeEvent(ev)


class Inspector(QtWidgets.QDialog):
    """The numbers behind the picture: the box, and where it is looked at from.

    Everything here is reachable with the mouse and the toolbar; what this adds
    is exactness. Somebody who wants the box to run from -1 to 1 in z, or the
    camera at an azimuth of 45 degrees precisely, has nowhere else to say so -
    and a 3D picture is one of the few places where a number typed is worth
    more than a gesture made.

    The numbers are grouped the way the picture is: the box first, as three
    columns of an axis each, and where it is looked at from second. The camera
    half is a readout as well as a field, following the picture while the
    dialog is up, which is why the heading says so.
    """

    #: The box, as a person reads one: a row of three numbers for where it is
    #: and a row of three for how big it is, under the axis each column is
    #: about. Six fields in a list is the same six numbers with the shape taken
    #: out of them.
    BOX = (("center", ("cx", "cy", "cz")), ("length", ("lx", "ly", "lz")))
    AXES = ("x", "y", "z")

    #: Where the box is looked at from, which is one row of three.
    CAMERA = ("azimuth", "elevation", "distance")

    def __init__(self, window: Window3D) -> None:
        super().__init__(window)
        self._window = window
        self.setWindowTitle(f"View of plot {window.number}")
        self.fields: dict[str, QtWidgets.QLineEdit] = {}
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        layout.addWidget(self._headed())
        layout.addWidget(self._box_group())
        layout.addWidget(self._camera_group())
        layout.addWidget(self._buttons())
        window.view.moved.connect(self._camera_moved)

    def _headed(self) -> QtWidgets.QWidget:
        """What the dialog is about, and the one thing worth knowing about it.

        That the camera numbers follow the picture is the difference between a
        form and an instrument, and it is not a thing anybody would guess from
        three fields with numbers in them.
        """
        holder = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(1)
        title = QtWidgets.QLabel(f"View - plot {self._window.number}")
        title.setStyleSheet(f"color: {theme.FIELD_TEXT}; font-weight: 600;")
        subtitle = QtWidgets.QLabel("follows the camera while open")
        subtitle.setStyleSheet(f"color: {theme.DIM};")
        column.addWidget(title)
        column.addWidget(subtitle)
        return holder

    def _box_group(self) -> QtWidgets.QWidget:
        group = QtWidgets.QGroupBox("Box")
        grid = QtWidgets.QGridLayout(group)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        for column, axis in enumerate(self.AXES):
            grid.addWidget(self._column_head(axis), 0, column + 1)
        for row, (shown, names) in enumerate(self.BOX, start=1):
            grid.addWidget(self._label(shown), row, 0)
            for column, name in enumerate(names):
                grid.addWidget(self._field(name), row, column + 1)
        return group

    def _camera_group(self) -> QtWidgets.QWidget:
        group = QtWidgets.QGroupBox("Camera")
        grid = QtWidgets.QGridLayout(group)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        for column, name in enumerate(self.CAMERA):
            grid.addWidget(self._column_head(name), 0, column)
            grid.addWidget(self._field(name), 1, column)
        return group

    def _buttons(self) -> QtWidgets.QWidget:
        """The three answers: give the z back, leave, or take what is typed.

        One button box rather than a row built by hand, so that the platform
        puts the buttons where the platform puts buttons; `Apply` is the
        accented one, being the only one of the three that does what the
        dialog is for.
        """
        roles = QtWidgets.QDialogButtonBox.ButtonRole
        buttons = QtWidgets.QDialogButtonBox()
        buttons.addButton("Apply", roles.ApplyRole).setObjectName(theme.PRIMARY)
        buttons.addButton("Autoscale z", roles.ResetRole)
        buttons.addButton("Close", roles.RejectRole)
        buttons.clicked.connect(self._clicked)
        return buttons

    def _field(self, name: str) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit()
        edit.setFixedWidth(90)
        edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.fields[name] = edit
        return edit

    def _label(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setStyleSheet(f"color: {theme.TEXT};")
        return label

    def _column_head(self, text: str) -> QtWidgets.QLabel:
        head = QtWidgets.QLabel(text)
        head.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        head.setStyleSheet(f"color: {theme.DIM};")
        return head

    def refresh(self) -> None:
        """Fill the fields from the window as it now stands."""
        box = self._window.box_now
        camera = self._window.view.cameraParams()
        values = dict(zip(("cx", "cy", "cz"), box.center))
        values.update(zip(("lx", "ly", "lz"), box.lengths))
        for name in ("azimuth", "elevation", "distance"):
            values[name] = float(camera.get(name, 0.0))
        for name, value in values.items():
            self.fields[name].setText(_written(value))

    def _camera_moved(self) -> None:
        """Follow the camera while the dialog is up: it is a readout as well."""
        if not self.isVisible():
            return
        camera = self._window.view.cameraParams()
        for name in ("azimuth", "elevation", "distance"):
            self.fields[name].setText(_written(float(camera.get(name, 0.0))))

    def _clicked(self, button: Any) -> None:
        text = button.text()
        if text == "Close":
            self.close()
        elif text == "Autoscale z":
            self._window.autoscale_z()
            self.refresh()
        else:
            self._apply()

    def _apply(self) -> None:
        try:
            center = [float(self.fields[name].text()) for name in ("cx", "cy", "cz")]
            lengths = [float(self.fields[name].text()) for name in ("lx", "ly", "lz")]
            camera = {
                name: float(self.fields[name].text())
                for name in ("azimuth", "elevation", "distance")
            }
        except ValueError:
            self._window.say("The view is described in numbers")
            self.refresh()
            return
        spans = [
            (middle - width / 2, middle + width / 2)
            for middle, width in zip(center, lengths)
        ]
        self._window.reframe(spans[0], spans[1], spans[2])
        self._window.view.setCameraPosition(
            distance=max(camera["distance"], 0.1),
            elevation=camera["elevation"],
            azimuth=camera["azimuth"],
        )
        self.refresh()


class _on_paper:
    """The window in paper colors for as long as an export takes.

    The same bargain the 2D window makes: a picture pasted into a document is
    read on white, so the background goes white, the surfaces take the darkened
    half of the palette, and the box and its numbers go black. Both swaps
    happen between two paints of the widget, and `grabFramebuffer` renders on
    its own rather than through the screen, so the picture is taken on white
    while the screen keeps the dark one.
    """

    def __init__(self, window: Window3D) -> None:
        self._window = window

    def __enter__(self) -> None:
        window = self._window
        window._papered = True
        window.view.setBackgroundColor("w")
        window.box.setColor(PAPER_BOX)
        for surface in window.plots:
            window._draw(surface)
        window._relabel()
        window._anchor()

    def __exit__(self, *_: Any) -> None:
        window = self._window
        window._papered = False
        window.view.setBackgroundColor(BACKGROUND)
        window.box.setColor(BOX_COLOR)
        for surface in window.plots:
            window._draw(surface)
        window._relabel()
        window._anchor()


def _alike(one: tuple[float, float], other: tuple[float, float]) -> bool:
    """Whether two ranges are the same range, allowing for what a field wrote.

    A number that went out to a field as `16.6667` and came back is the same
    number for this purpose: what is being asked is whether somebody typed
    something, not whether two floats are equal.
    """
    span = max(abs(other[1] - other[0]), 1e-12)
    return all(abs(a - b) <= 1e-4 * span for a, b in zip(one, other))


def _written(value: float) -> str:
    """A number as a toolbar field and a tick label write it: short and exact."""
    number = float(value)
    if number == int(number) and abs(number) < 1e15:
        return str(int(number))
    return f"{number:g}"
