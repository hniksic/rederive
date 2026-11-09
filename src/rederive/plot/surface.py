"""Surface geometry: the bounding box, mesh, wireframe and shading.

This module computes, without any toolkit, everything a 3D window draws: where data
coordinates land in the box, which faces survive holes in the domain, where the mesh
ends, and how bright each vertex is. Windows just upload the result, so another backend
(three.js) can draw exactly the same vertices, triangles and colors.

**The box is a fixed cube, not shaped by the data.** Each of the three ranges is
stretched onto it independently. So a surface with a tiny or huge z range still reads
as a shape, `f` and `100*f` look the same, and a camera framing one surface frames
any. The numbers along the box edges show the actual values.

**The z range follows the data, within reason.** It autoscales to the finite values of
all surfaces in the window, but if a spike would flatten everything else, it is clipped
to the 1st-99th percentile and the window says so. Vertices outside the clip are
dropped like non-real ones, so the spike isn't drawn flattened against the lid.

**Holes are holes, and the mesh ends where the surface does.** `SQRT(1-x^2-y^2)` is a
dome over the unit disc and nothing outside it. Faces are kept only where the surface
is real, and cells crossed by the domain's edge are filled up to the boundary refined
by the sampler, so the dome's rim is a circle rather than a staircase.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from rederive.plot import evaluate
from rederive.plot.appearance import LIGHT
from rederive.plot.view import PERCENTILES, Span, pooled

__all__ = ["Box", "brightened", "extent", "mesh", "place", "spanned", "ticks", "wire"]

#: The side of the cube surfaces are drawn in, and half of it, where the walls are. In
#: world units, which have no meaning beyond this.
WORLD = 10.0
HALF = WORLD / 2

#: Vertex brightness parameters. Half the brightness comes from height (from `DIM` at
#: the floor to 1 at the lid) and half from a fixed light, which makes folds visible
#: where height alone would not. The light is baked into vertex colors rather than
#: computed by a shader, so the mesh needs no normals. Its direction is in
#: `plot.appearance`, shared with the page.
DIM = 0.30
AMBIENT = 0.20
HEIGHT_SHARE = 0.5

#: Target number of wireframe lines in each direction. Drawing every grid line would
#: be gray fuzz, so only every k-th row and column is drawn, each at full sample
#: resolution. A finer grid therefore makes lines smoother rather than more numerous.
WIRE_LINES = 20

#: Target number of tick marks per axis.
TICKS = 5


@dataclass(frozen=True)
class Box:
    """The x, y and z ranges mapped onto the cube.

    Everything drawn is placed through this mapping, so the axis numbers and geometry
    always agree. The cube is the same whatever the ranges, so neither a small z range
    nor an elongated domain distorts the picture.
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

    def across(self, values: Any) -> np.ndarray:
        return place(values, self.x, WORLD)

    def along(self, values: Any) -> np.ndarray:
        return place(values, self.y, WORLD)

    def up(self, values: Any) -> np.ndarray:
        return place(values, self.z, WORLD)


def place(values: Any, span: tuple[float, float], length: float) -> np.ndarray:
    """Map data coordinates in `span` onto `length` world units, centered at zero."""
    low, high = float(span[0]), float(span[1])
    width = high - low
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(width) or width == 0:
        return np.zeros_like(array)
    return length * ((array - low) / width - 0.5)


def spanned(values: np.ndarray) -> Span | None:
    """The full and percentile z ranges of one grid's finite values.

    `plot/view.py`'s `pooled` then combines them over all surfaces in the window and
    decides whether to clip.

    Returns None if there are no finite values.
    """
    flat = np.asarray(values, dtype=np.float64).ravel()
    finite = flat[np.isfinite(flat)]
    if not finite.size:
        return None
    inner = np.percentile(finite, PERCENTILES)
    return Span(
        low=float(np.min(finite)),
        high=float(np.max(finite)),
        inner_low=float(inner[0]),
        inner_high=float(inner[1]),
    )


def extent(arrays: Sequence[np.ndarray]) -> tuple[tuple[float, float], bool] | None:
    """The box's z range for these arrays, and whether it was clipped.

    Combines `spanned` and `pooled`, for callers that have all the arrays. (The browser
    keeps the arrays and the box in different interpreters, so it calls them
    separately.)
    """
    spans = [span for span in (spanned(array) for array in arrays) if span is not None]
    return pooled(spans)


def mesh(
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
    box: Box,
    boundary: evaluate.Boundary | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A surface's vertices in the box, its faces, and per-vertex shading.

    A cell is kept as a whole face if all four corners are real, finite and inside the
    box. A cell crossed by the domain's edge is filled up to the boundary instead of
    dropped: its defined corners plus the refined crossings in `boundary` form a
    triangle, quadrilateral or pentagon (the marching-squares cases), so the mesh ends
    where the surface does. Then partial cells are trimmed to the box's z range by
    linear interpolation along their edges. This needs no evaluation, since the box
    changes on the Qt thread, and it also handles boundary vertices whose limit is
    unbounded. Without `boundary`, cells crossed by the domain's edge are dropped, but
    clip trimming still happens.

    Unused grid corners keep their vertex (unreferenced, so not drawn), since
    renumbering would cost more than it saves.

    Shading is computed here from the same array (height and slope), so the drawing
    needs no normals or lighting. Boundary vertices are off the grid, so their light
    is interpolated along their edge.
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
    upright = np.where(inside, box.up(zs), -HALF)
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
    # Vertices without a value are unreferenced, but their shading must still be a
    # number.
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


#: The corners of a grid cell in perimeter order from the lower left, so that edge k
#: runs from corner k to corner k+1.
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
    """Fill the cells that aren't whole faces, as far as the surface extends.

    These are cells crossed by the domain's edge (filled from their defined corners
    and the refined crossings) and cells cut by the clip planes. Both are handled by
    building the defined polygon and then trimming it to the box's z, so boundary
    vertices are trimmed like any other. Cells entirely above or below the box are
    skipped.

    Returns the new vertices as (x, y, z, shade) in data coordinates, and their
    triangles, indexed from `offset`.
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
    """The marching-squares fill of a partial cell, as polygons in data space.

    The defined corners in perimeter order, with the refined crossing inserted on each
    edge that straddles the domain boundary. One, two or three defined corners give a
    triangle, quadrilateral or pentagon; four give the whole quad for clipping. The
    ambiguous case (two diagonal corners) always becomes two separate triangles, since
    consistency matters more than guessing the topology. If a straddling edge has no
    crossing, the cell is left unfilled.
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

    The point is off the grid, so the height part of its shade comes from its own z,
    and the light part is interpolated between the edge's two grid ends.
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
    """`polygon` clipped to the box's z range.

    Crossings are linearly interpolated in all components (position and shade), with
    no evaluation, so this can run on the Qt thread whenever the box changes. The
    result is convex, since the vertices lie on a convex cell's rim, so a fan can
    triangulate it.
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
    """A surface as a wireframe of its sample grid, with shading.

    Returns segments along the grid's rows and columns, one per grid step, for a
    `GLLinePlotItem` in `'lines'` mode: consecutive point pairs are segments, with a
    shade per point computed as for the solid. (`drawEdges` would draw the triangle
    diagonals too.)

    Only every k-th row and column is drawn, about `WIRE_LINES` each way, at full
    sample resolution.

    A segment is kept under the same rule as faces: both ends real, finite and inside
    the box's z, so holes and clip edges match the solid. A non-real end is replaced by
    the refined crossing from `boundary`. An end outside the box's z is trimmed to the
    clip plane by the same interpolation used for faces, which also handles boundary
    ends whose limit is unbounded.
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
    """The indices of rows or columns drawn by the wireframe: every k-th, and the last.

    The step gives about `WIRE_LINES` lines. The last is always included so the
    wireframe reaches the grid's edge on every side.
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
    """Append the visible part of one wireframe segment to `points`.

    The segment runs from sample `(i, j)` to the next sample along x (`across`) or y,
    in data coordinates with a shade per end. A non-real end is replaced by the refined
    crossing from `_crossing`, so boundary points match the mesh exactly; if there is
    no crossing, the segment is dropped. The rest is trimmed to the box's z.
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
    """The part of segment `a`-`b` inside the box's z range, or None.

    The segment counterpart of `_trimmed`, likewise using linear interpolation.
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
    """The light each vertex receives, from the grid's slope there.

    The normal of a height field is `(-dz/dx, -dz/dy, 1)` normalized, which can be
    computed directly on the array without a mesh, avoiding face normals that would
    have to account for holes. The light is fixed in the world, so rotating the picture
    moves the highlight.
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
    """Per-vertex colors: the surface's color scaled by each vertex's brightness."""
    rgb = np.array(_channels(color), dtype=np.float32)
    scale = np.clip(np.asarray(shading, dtype=np.float32), 0.0, 1.0)[:, None]
    colors = np.ones((int(np.size(shading)), 4), dtype=np.float32)
    colors[:, :3] = rgb * scale
    return colors


def _channels(color: str) -> tuple[float, float, float]:
    """The channels of `#rrggbb` (or `#rgb`), each from zero to one.

    Parsed here rather than by a toolkit, since this module is toolkit-free. Plot
    colors come from the palettes in `model`, which use this format.
    """
    text = color.lstrip("#")
    if len(text) == 3:
        text = "".join(digit * 2 for digit in text)
    red, green, blue = (int(text[at : at + 2], 16) / 255.0 for at in (0, 2, 4))
    return red, green, blue


def ticks(low: float, high: float, count: int = TICKS) -> list[float]:
    """About `count` round tick values for an axis.

    Steps are 1, 2, 2.5 or 5 times a power of ten, which are easy to read.
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
