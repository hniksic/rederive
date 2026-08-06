"""The geometry of a surface: the box it stands in, its mesh, its wire and its light.

Several hundred lines of arrays and no toolkit at all. What a 3D window draws
is worked out here - where a data coordinate lands in the box, which faces
survive a hole in the domain, where the mesh stops, how bright each vertex is -
and what a window does with the answer is upload it. That is what lets a second
backend draw the same solid: three.js is handed exactly these vertices, these
triangles and these colors.

**The picture is drawn in a box of the program's choosing, not the data's.** The
floor is always the same square, whatever rectangle the domain is, and the
height is the true proportion of the z range to the floor - until that
proportion is one no picture can hold, when it is capped at the floor's own
length or lifted off it. So a surface whose z runs to a million reads as a shape
rather than as a wall, a hemisphere over its own disc reads as a hemisphere, and
a camera that frames one surface frames every other one. The numbers along the
box edges carry the truth; the geometry carries the form.

**The z extent is the data's, within reason.** It autoscales to the finite
values of every surface in the window, and where a spike would crush everything
else into a plane it is taken from the 1st to the 99th percentile instead, with
the window saying so. Vertices outside the clip leave the mesh exactly as
non-real ones do, so the spike is not drawn flattened against the lid - it is
simply not there.

**Holes are holes, and the mesh stops where the surface does.**
`SQRT(1-x^2-y^2)` is a dome over the unit disc and nothing at all outside it,
so faces are kept only where the surface is real, and a cell the domain's edge
crosses is filled up to the boundary the sampler refined - on the closure, to
a fraction of a cell - rather than dropped, so the dome's rim is a circle and
not a staircase.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from rederive.plot import evaluate

__all__ = ["Box", "brightened", "extent", "mesh", "place", "ticks", "wire"]

#: The side of the square floor every surface is drawn over, and half of it,
#: which is where the box walls are. World units, and they mean nothing else:
#: the axis numbers are the only quantities in a 3D window with units.
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

#: How many tick marks an axis aims for.
TICKS = 5


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
    rgb = np.array(_channels(color), dtype=np.float32)
    scale = np.clip(np.asarray(shading, dtype=np.float32), 0.0, 1.0)[:, None]
    colors = np.ones((int(np.size(shading)), 4), dtype=np.float32)
    colors[:, :3] = rgb * scale
    return colors


def _channels(color: str) -> tuple[float, float, float]:
    """The three channels of `#rrggbb`, each from zero to one.

    Read here rather than asked of a toolkit, since a color is three numbers
    and this file is the geometry: every color a plot is drawn in comes from
    the palettes in `model`, which are written this way.
    """
    text = color.lstrip("#")
    if len(text) == 3:
        text = "".join(digit * 2 for digit in text)
    red, green, blue = (int(text[at : at + 2], 16) / 255.0 for at in (0, 2, 4))
    return red, green, blue


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
