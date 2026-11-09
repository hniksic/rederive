"""Browser plot sampling in the engine worker: closures, samples and readings.

This runs beside sympy in the worker and is the only part of browser plotting that uses
numpy. Its answers are arrays sent straight to the page's JavaScript, never becoming
Python objects on the main thread.

The only state kept between requests is a cache of lambdified closures, which the
worker's statelessness contract allows ("a content-keyed cache the protocol treats as
maybe-empty"). It is keyed by the tree, its context and its variables, so resampling a
dragged view or plotting the same expression in another window reuses a closure, while
a freshly restarted worker simply rebuilds it. Which pane holds what is session state
and lives on the main thread.

The sampling jobs come unchanged from `plot/resample.py`. The record built here has the
same fields as the desktop window's, so both backends sample identically and differ
only in where the result is drawn.

For surfaces, the answer is the geometry from `plot/surface.py` (triangles, wireframe
and per-vertex colors) rather than the sampled grid; the page only displays and rotates
it.

Two things the desktop gets from pyqtgraph are implemented here: marching squares for
the zero contour of an implicit plot, and region shading sent as the boolean grid
itself, one byte per cell, for the page to render as an image.
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Callable
from typing import Any

import numpy as np

from rederive.plot import actions, evaluate, resample, view
from rederive.plot import surface as geometry
from rederive.plot.model import (
    FIELDS,
    FUNCTIONS,
    SOLIDS,
    Plot,
    Surface,
    naming,
    pointed,
    reading,
    roughly,
    written,
)
from rederive.plot.protocol import PARAMETRIZED, PlotKind
from rederive.plot.web import protocol
from rederive.plot.web.protocol import Features, Grid, Numbers, Sample, Trace

__all__ = ["Closures", "methods"]

#: Number of lambdified expressions cached before the oldest is evicted. Enough for
#: several windows of curves; beyond that, rebuilding one only costs a conversion.
CACHED = 32

#: Shapes of a sampling answer, which tell the page how to read its fields. A stroke is
#: two arrays of points; a region is a boolean grid the page shades.
STROKE = "stroke"
REGION = "region"


class Closures:
    """The lambdified expressions this worker has built, keyed by content.

    Eviction is simply oldest-inserted first; since the protocol tolerates an empty
    cache, nothing smarter is needed.
    """

    def __init__(self, held: int = CACHED) -> None:
        self._held = held
        self._made: dict[bytes, Any] = {}

    def keyed(self, plot: Plot, names: tuple[str, ...], reading: str) -> bytes:
        """Return the cache key for an expression.

        The key covers the tree, its context, its variables and the reading wanted,
        since one tree can yield different closures (`u - v` for an implicit curve,
        the truth of `u > v` for a region). It hashes the pickle of these.
        """
        return hashlib.blake2b(
            pickle.dumps((plot.node, plot.context, names, reading)), digest_size=16
        ).digest()

    def get(self, key: bytes) -> Any:
        return self._made.get(key)

    def put(self, key: bytes, made: Any) -> Any:
        self._made[key] = made
        while len(self._made) > self._held:
            del self._made[next(iter(self._made))]
        return made


class Riding(Plot):
    """A plot with its closures, the worker's counterpart of the desktop's `Drawn`.

    It adds the fields sampling reads and writes (the closure, and the pair for a
    parametrized curve), so `plot/resample.py` works on it as on a desktop record.
    """

    closure: Any = None
    pair: Any = None
    xs: Any = None


class Gridded(Surface):
    """A surface with its closure, the 3D counterpart of `Riding`.

    `plot/resample.py`'s grid job reads and stores the closure on it.
    """

    closure: Any = None


def methods(post: Callable[[dict[str, Any]], None]) -> dict[str, Callable[..., Any]]:
    """Return the plot request handlers, sharing one closure cache.

    `post` sends a partial answer to the page: a sampling's uniform pass is drawn while
    refinement continues, and the final answer follows. Surfaces send no partial
    answers, since their grid is evaluated in one pass.
    """
    held = Closures()
    return {
        protocol.SAMPLE: lambda request: _sampled(held, request, post),
        protocol.TRACE: lambda request: _traced(held, request),
        protocol.FEATURES: lambda request: _found(held, request),
        protocol.GRID: lambda request: _gridded(held, request),
        protocol.NUMBERS: _numbers,
    }


# -- typed bounds --------------------------------------------------------------


def _numbers(request: Numbers) -> tuple[float, ...]:
    """Evaluate the bound expressions a form collected, such as `-π`.

    The text was parsed on the main thread, but evaluating it needs sympy. A tree with
    no finite value comes back as NaN rather than an error, and the form reports it.
    """
    return tuple(
        evaluate.number(node, request.context, float("nan")) for node in request.nodes
    )


# -- one sampling -------------------------------------------------------------


def _sampled(
    held: Closures, request: Sample, post: Callable[[dict[str, Any]], None]
) -> dict[str, Any]:
    """Sample one plot over one view into the arrays the page draws."""
    plot = _riding(held, request.model, request.degrees)
    answer = _about(request)
    try:
        work = resample.job(
            plot, request.xrange, request.yrange, request.size, request.degrees
        )
        return answer | _drawn(held, plot, work, request, post)
    except Exception as error:  # reported as the curve's trouble
        return answer | {"shape": STROKE, "trouble": _said(error)}


def _drawn(
    held: Closures,
    plot: Riding,
    work: Callable[..., Any],
    request: Sample,
    post: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Run a sampling job and unpack its answer according to the plot kind."""
    kind = plot.kind
    if kind in FIELDS:
        return _field(held, plot, work(_nothing))
    if kind in PARAMETRIZED:
        return _path(held, plot, work(_nothing), request)
    if kind is PlotKind.DATA:
        xs, ys = work(_nothing)
        return _stroke(xs, ys) | _empty(plot, xs, ys, request)
    made, xs, ys = work(
        lambda ax, ay: post(_about(request) | _stroke(ax, ay) | {"partial": True})
    )
    _remember(held, plot, made)
    return _stroke(xs, ys) | _empty(plot, xs, ys, request)


def _path(
    held: Closures, plot: Riding, answer: Any, request: Sample
) -> dict[str, Any]:
    """Answer for a parametric or polar curve, noting where refinement gave up.

    The parameter of each sample is also sent. A click on a self-crossing curve gives
    only a point in the plane, so the page snaps to the nearest sample and traces from
    its parameter, as the desktop window does.
    """
    pair, inner, sampled, trange = answer
    _remember(held, plot, inner, pair)
    said = ""
    if sampled.gave_up is not None:
        said = (
            f"{plot.label}: gave up refining near"
            f" {plot.parameter} = {roughly(sampled.gave_up)}"
        )
    return (
        _stroke(sampled.xs, sampled.ys)
        | _empty(plot, sampled.xs, sampled.ys, request)
        | {
            "ts": np.ascontiguousarray(sampled.ts, dtype=np.float32).ravel(),
            "trange": [float(trange[0]), float(trange[1])],
            "words": said,
        }
    )


def _field(held: Closures, plot: Riding, answer: Any) -> dict[str, Any]:
    """Answer for an implicit contour or a region over the evaluated grid.

    A contour is line segments separated by gaps, drawn like any other stroke. A
    region is the boolean grid, one byte per cell, in image order (rows from the top)
    so the page can shade it directly.
    """
    made, xs, ys, values = answer
    _remember(held, plot, made)
    extent = [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])]
    if plot.kind is PlotKind.IMPLICIT:
        cx, cy = contour(xs, ys, values)
        # An empty implicit curve gets its own message: the grid evaluated, but there
        # is no solution in view.
        said = "" if cx.size else f"{plot.label}: no solutions in view"
        return _stroke(cx, cy) | {
            "extent": extent,
            "empty": not bool(cx.size),
            "words": said,
        }
    inside = np.asarray(values, dtype=bool)
    return {
        "shape": REGION,
        "mask": np.ascontiguousarray(inside.T[::-1], dtype=np.uint8).ravel(),
        "nx": int(inside.shape[0]),
        "ny": int(inside.shape[1]),
        "extent": extent,
    }


def _about(request: Sample) -> dict[str, Any]:
    """Return the common fields of a sampling answer: pane, plot and generation."""
    return {
        "pane": request.pane,
        "plot": request.plot,
        "generation": request.generation,
        "shape": STROKE,
        "trouble": "",
        "words": "",
        "empty": False,
        "partial": False,
    }


def _stroke(xs: Any, ys: Any) -> dict[str, Any]:
    """Convert two arrays to the page's format: 1-D, contiguous, float32.

    Single precision is plenty for a screen. A contiguous 1-D array crosses as one
    typed array with a single copy, while more dimensions would become nested JS lists.
    """
    return {
        "xs": np.ascontiguousarray(xs, dtype=np.float32).ravel(),
        "ys": np.ascontiguousarray(ys, dtype=np.float32).ravel(),
        "bounds": _bounds(xs, ys),
    }


def _bounds(xs: Any, ys: Any) -> list[float]:
    """Return the bounding rectangle of the finite samples, or [] if there are none.

    Computed here, where the samples are, so the page autoscales by combining four
    numbers per curve instead of scanning arrays. `plot/view.py` frames windows on
    such rectangles.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    keep = np.isfinite(xs) & np.isfinite(ys)
    if not keep.any():
        return []
    xs, ys = xs[keep], ys[keep]
    return [float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())]


def _empty(plot: Riding, xs: Any, ys: Any, request: Sample) -> dict[str, Any]:
    """Explain a plot with nothing to draw in view.

    An unexplained empty picture must be avoided. The message names the empty range,
    since the fix is nearly always to look elsewhere.
    """
    if np.isfinite(np.asarray(ys, dtype=np.float64)).any():
        return {}
    left, right = request.xrange
    name = plot.abscissa or plot.variable
    span = f"{roughly(left)} ≤ {name} ≤ {roughly(right)}"
    return {
        "empty": True,
        "words": f"{plot.label}: no real values for {span} - try A to autoscale",
    }


# -- surfaces ------------------------------------------------------------------


def _gridded(held: Closures, request: Grid) -> dict[str, Any]:
    """Build the geometry of one surface over one domain for the page.

    The triangles, wireframe and per-vertex colors come from `plot/surface.py`, the
    same arrays the desktop gives pyqtgraph. The page only displays and rotates them.
    """
    surface = _standing(held, request.model)
    answer = _about_solid(request)
    try:
        work = resample.grid_job(
            surface, request.xdomain, request.ydomain, request.grid
        )
        made, xs, ys, values, boundary = work(_nothing)
        _remember(held, surface, made)
        return answer | _built(surface, xs, ys, values, boundary, request)
    except Exception as error:  # reported as the surface's trouble
        return answer | {"trouble": _said(error)}


def _built(
    surface: Gridded,
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
    boundary: Any,
    request: Grid,
) -> dict[str, Any]:
    """Build the mesh, wireframe and shading of one grid, placed in one box.

    Three z ranges are returned. The box is where the vertices were placed (the pane's,
    if it has one); `wanted` is the box this surface would use alone; and `span`
    summarizes its values, which the pane pools across surfaces to choose the box. For
    a single surface all three agree.
    """
    span = geometry.spanned(values)
    if span is None:
        return _undrawn() | {
            "empty": True,
            "words": f"{surface.label}: no real values over this domain",
        }
    found = view.pooled([span])
    assert found is not None
    wanted, clipped = found
    zrange = wanted if request.zrange is None else request.zrange
    said = ""
    if clipped and request.zrange is None:
        said = actions.clipped(*wanted)
    box = geometry.Box(request.xdomain, request.ydomain, zrange)
    vertexes, faces, shading = geometry.mesh(xs, ys, values, box, boundary)
    points, shades = geometry.wire(xs, ys, values, box, boundary)
    return {
        "vertices": _placed(vertexes),
        "faces": np.ascontiguousarray(faces, dtype=np.uint32).ravel(),
        "colors": _tinted(shading, surface.color),
        "wire": _placed(points),
        "wirecolors": _tinted(shades, surface.color),
        "wanted": [float(wanted[0]), float(wanted[1])],
        "span": [span.low, span.high, span.inner_low, span.inner_high],
        "clipped": bool(clipped),
        "words": said,
    } | _box(box)


def _about_solid(request: Grid) -> dict[str, Any]:
    """Return the common fields of a surface answer: pane, plot and generation."""
    return {
        "pane": request.pane,
        "plot": request.plot,
        "generation": request.generation,
        "reply": "grid",
        "trouble": "",
        "words": "",
        "empty": False,
    }


def _box(box: geometry.Box) -> dict[str, Any]:
    """Describe the box the vertices are placed in.

    World units are for drawing (the fixed cube the three ranges are stretched onto);
    data units are for display, as in the desktop's inspector and axis labels.

    Display values and tick labels are formatted here with the same function the
    desktop uses, so both show numbers identically. Each tick carries its world
    coordinate, so the page only has to project it.
    """
    return {
        "world": float(geometry.WORLD),
        "center": [float(value) for value in box.center],
        "lengths": [float(value) for value in box.lengths],
        "zrange": [float(box.z[0]), float(box.z[1])],
        "reading": {
            "center": [written(value) for value in box.center],
            "lengths": [written(value) for value in box.lengths],
            "z": [written(box.z[0]), written(box.z[1])],
        },
        "ticks": {
            "x": _marks(box.x, box.across),
            "y": _marks(box.y, box.along),
            "z": _marks(box.z, box.up),
        },
    }


def _marks(
    span: tuple[float, float], place: Callable[..., Any]
) -> list[dict[str, Any]]:
    """Return the ticks of one axis, each with its world coordinate and label.

    Tick positions come from `plot/surface.py` and labels from `written`, as for the
    desktop, so the page formats nothing.
    """
    return [
        {"at": float(place(value)), "text": written(value)}
        for value in geometry.ticks(span[0], span[1])
    ]


def _undrawn() -> dict[str, Any]:
    """Return an empty mesh.

    Sent explicitly so the page removes the previous mesh when the surface has no real
    values over the new domain.
    """
    return {
        "vertices": np.empty(0, dtype=np.float32),
        "faces": np.empty(0, dtype=np.uint32),
        "colors": np.empty(0, dtype=np.float32),
        "wire": np.empty(0, dtype=np.float32),
        "wirecolors": np.empty(0, dtype=np.float32),
        "wanted": [],
        "span": [],
        "clipped": False,
    }


def _placed(points: np.ndarray) -> np.ndarray:
    """Flatten vertices to float32, three per point, as a `BufferAttribute` reads them.

    As with strokes, a 2-D array would cross as nested JS lists, which matters even
    more for surfaces with many more numbers.
    """
    return np.ascontiguousarray(points, dtype=np.float32).ravel()


def _tinted(shading: np.ndarray, color: str) -> np.ndarray:
    """Return per-vertex RGB colors: the surface color scaled by brightness.

    The alpha from `brightened` is dropped, since surfaces are opaque.
    """
    colors = geometry.brightened(shading, color)[:, :3]
    return np.ascontiguousarray(colors, dtype=np.float32).ravel()


# -- tracing and features ------------------------------------------------------


def _traced(held: Closures, request: Trace) -> dict[str, Any]:
    """Evaluate the curve at the trace marker and describe the result.

    All numbers are formatted here; the page displays them as given.
    """
    plot = _ensured(held, request.model, request.degrees)
    answer = {
        "pane": request.pane,
        "plot": request.plot,
        "generation": request.generation,
        "reply": "trace",
        "at": float(request.at),
        "found": False,
        "point": "",
        "words": "",
    }
    try:
        return answer | _read(plot, request)
    except Exception as error:
        return answer | {"words": f"{plot.label}: {_said(error)}"}


def _read(plot: Riding, request: Trace) -> dict[str, Any]:
    """Evaluate one point of a curve according to its kind."""
    at = float(request.at)
    if plot.kind in PARAMETRIZED:
        assert plot.pair is not None
        x, y = _one(plot.pair[0], at), _one(plot.pair[1], at)
        if not (np.isfinite(x) and np.isfinite(y)):
            return {"words": _lost(plot, plot.parameter, at)}
        radius = ""
        if plot.kind is PlotKind.POLAR and plot.closure is not None:
            radius = f"   r = {reading(_one(plot.closure, at))}"
        said = (
            f"Tracing {plot.named}   {plot.parameter} = {reading(at)}"
            f"{radius}   x = {reading(x)}   y = {reading(y)}"
        )
        return _at(x, y, said)
    assert plot.closure is not None
    y = _one(plot.closure, at)
    if not np.isfinite(y):
        return {"words": _lost(plot, plot.variable, at)}
    said = (
        f"Tracing {plot.named}   {plot.variable} = {reading(at)}   y = {reading(y)}"
    )
    return _at(at, y, said)


def _at(x: float, y: float, said: str) -> dict[str, Any]:
    return {
        "found": True,
        "x": float(x),
        "y": float(y),
        "point": pointed(x, y),
        "words": said,
    }


def _lost(plot: Riding, name: str, at: float) -> str:
    """Describe a trace position where the curve has no real, finite value.

    Tracing is how removable singularities are found, so this is a result, not an
    error.
    """
    return f"Tracing {plot.named}: not real and finite at {name} = {reading(at)}"


def _found(held: Closures, request: Features) -> dict[str, Any]:
    """Find all notable points of one curve within its view.

    The whole list is returned so the page can step through it with Tab without a
    round trip per keystroke. The curve is resampled here, which is cheap with cached
    closures, rather than having the worker keep samples.
    """
    plot = _riding(held, request.model, False)
    answer = {
        "pane": request.pane,
        "plot": request.plot,
        "generation": request.generation,
        "reply": "features",
        "features": [],
        "words": "",
    }
    try:
        work = resample.job(plot, request.xrange, request.yrange, request.size)
        made, xs, ys = work(_nothing)
        _remember(held, plot, made)
        others = [_ensured(held, other, False) for other in request.others]
        found = evaluate.features(
            xs, ys, made, [one.closure for one in others if one.closure is not None]
        )
        crossed = [naming(one.label, one.text) for one in others]
        return answer | {"features": [_feature(one, crossed) for one in found]}
    except Exception as error:
        return answer | {"words": f"{plot.label}: {_said(error)}"}


def _feature(found: Any, crossed: list[str]) -> dict[str, Any]:
    """Describe one notable point as the status line shows it."""
    where = f"at x = {reading(found.x)}"
    if found.kind is not evaluate.CROSSING:
        said = f"{found.kind} {where}"
    elif 0 <= found.other < len(crossed):
        said = f"intersection with {crossed[found.other]} {where}"
    else:
        said = f"intersection {where}"
    return {"x": float(found.x), "y": float(found.y), "words": said}


# -- plot records --------------------------------------------------------------


def _riding(held: Closures, model: Plot, degrees: bool) -> Riding:
    """Build a `Riding` record with any cached closures.

    Closures are only looked up here. A job builds what it's missing, and `_remember`
    caches it, so nothing depends on the cache being populated.
    """
    plot = Riding(
        **{name: getattr(model, name) for name in model.__dataclass_fields__}
    )
    made = held.get(held.keyed(plot, _names(plot), _reading_of(plot)))
    if plot.kind is PlotKind.PARAMETRIC:
        plot.pair = made
        return plot
    plot.closure = made
    if plot.kind is PlotKind.POLAR and made is not None:
        # A polar curve's r(θ) is cached under the same key as a function curve of the
        # same tree, so toggling polar mode needs no conversion. Composing the pair is
        # cheap and not cached.
        plot.pair = evaluate.polar_pair(made, degrees)
    return plot


def _standing(held: Closures, model: Surface) -> Gridded:
    """Build a `Gridded` record with any cached closure.

    A surface uses the same kind of closure as a function curve, just of two
    variables, so changing the domain or grid, or replotting the expression, reuses it.
    """
    surface = Gridded(
        **{name: getattr(model, name) for name in model.__dataclass_fields__}
    )
    surface.closure = held.get(
        held.keyed(surface, _names(surface), _reading_of(surface))
    )
    return surface


def _ensured(held: Closures, model: Plot, degrees: bool) -> Riding:
    """Build a `Riding` record, building any closures missing from the cache.

    Sampling jobs build their own closures, but tracing and feature scans don't, so
    they use this. It covers, for example, a worker freshly restarted after Esc.
    """
    plot = _riding(held, model, degrees)
    if plot.kind is PlotKind.PARAMETRIC:
        if plot.pair is None:
            plot.pair = evaluate.pair(plot.node, plot.context, _names(plot))
            _remember(held, plot, None, plot.pair)
        return plot
    if plot.kind is PlotKind.DATA or plot.closure is not None:
        return plot
    plot.closure = _maker(plot)(plot.node, plot.context, _names(plot))
    _remember(held, plot, plot.closure)
    if plot.kind is PlotKind.POLAR:
        plot.pair = evaluate.polar_pair(plot.closure, degrees)
    return plot


def _maker(plot: Plot) -> Callable[..., Any]:
    """Return the closure builder for this plot's reading of its tree."""
    if plot.kind is PlotKind.IMPLICIT:
        return evaluate.difference
    if plot.kind is PlotKind.REGION:
        return evaluate.mask
    return evaluate.closure


def _remember(held: Closures, plot: Riding, made: Any, pair: Any = None) -> None:
    """Cache what a job built.

    For a parametric curve that's the pair; for everything else, including polar
    curves, it's the single closure (the polar pair is cheaply composed from `r(θ)`).
    """
    kept = pair if plot.kind is PlotKind.PARAMETRIC else made
    if kept is not None:
        held.put(held.keyed(plot, _names(plot), _reading_of(plot)), kept)


def _names(plot: Plot) -> tuple[str, ...]:
    """Return the variables of a plot's closure, in axis order."""
    if plot.kind in FIELDS or plot.kind in SOLIDS:
        return plot.axes
    if plot.kind in PARAMETRIZED:
        return plot.options.variables or ("t",)
    if plot.kind in FUNCTIONS:
        return plot.options.variables or ("x",)
    return ()


def _reading_of(plot: Plot) -> str:
    """Return the name of this plot's reading of its tree, part of the cache key.

    One expression can be read several ways: `u = v` is a difference as a contour and a
    truth value as a region.
    """
    if plot.kind is PlotKind.REGION:
        return "mask"
    if plot.kind is PlotKind.IMPLICIT:
        return "difference"
    if plot.kind is PlotKind.PARAMETRIC:
        return "pair"
    return "closure"


def _one(f: Callable[..., np.ndarray], at: float) -> float:
    """Evaluate a closure at one point, returning NaN where it has no value."""
    try:
        return float(np.asarray(f(np.array([at], dtype=np.float64)))[0])
    except Exception:
        return float("nan")


def _nothing(*_arguments: Any) -> None:
    """No-op progress callback, for jobs that report none."""


def _said(error: Exception) -> str:
    """Describe an exception in one line for the message line."""
    text = str(error).strip().splitlines()
    return text[0] if text else type(error).__name__


# -- zero contour --------------------------------------------------------------

#: The four edges of a cell. Marching squares crosses at most two, chosen by the signs
#: of the corners.
_BOTTOM, _RIGHT, _TOP, _LEFT = 0, 1, 2, 3

#: Edge pairs the contour crosses for each of the sixteen corner sign combinations. The
#: two ambiguous saddle cases (opposite corners positive) are always resolved the same
#: way rather than by the center value, so contours don't flicker as the grid changes
#: during a drag.
_CASES: dict[int, tuple[tuple[int, int], ...]] = {
    1: ((_LEFT, _BOTTOM),),
    2: ((_BOTTOM, _RIGHT),),
    3: ((_LEFT, _RIGHT),),
    4: ((_RIGHT, _TOP),),
    5: ((_LEFT, _TOP), (_BOTTOM, _RIGHT)),
    6: ((_BOTTOM, _TOP),),
    7: ((_LEFT, _TOP),),
    8: ((_TOP, _LEFT),),
    9: ((_TOP, _BOTTOM),),
    10: ((_TOP, _RIGHT), (_LEFT, _BOTTOM)),
    11: ((_TOP, _RIGHT),),
    12: ((_RIGHT, _LEFT),),
    13: ((_RIGHT, _BOTTOM),),
    14: ((_BOTTOM, _LEFT),),
}


def contour(
    xs: np.ndarray, ys: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the zero contour of a grid as one stroke with gaps between segments.

    Uses vectorized marching squares and returns unjoined segments, since joining them
    into paths would make no visible difference.

    Cells with any non-finite corner are skipped. Otherwise `SQRT(x) = y` would get a
    spurious wall along the y axis from comparisons against NaN.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] < 2 or values.shape[1] < 2:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    corners = (
        values[:-1, :-1],
        values[1:, :-1],
        values[1:, 1:],
        values[:-1, 1:],
    )
    real = np.isfinite(corners[0])
    for value in corners[1:]:
        real &= np.isfinite(value)
    signs = [np.asarray(value > 0.0) for value in corners]
    case = signs[0] * 1 + signs[1] * 2 + signs[2] * 4 + signs[3] * 8
    across = (float(xs[-1]) - float(xs[0])) / (len(xs) - 1)
    up = (float(ys[-1]) - float(ys[0])) / (len(ys) - 1)
    edges = _crossings(corners, float(xs[0]), float(ys[0]), across, up)
    drawn_x: list[np.ndarray] = []
    drawn_y: list[np.ndarray] = []
    for number, pairs in _CASES.items():
        here = real & (case == number)
        if not here.any():
            continue
        i, j = np.nonzero(here)
        gap = np.full(i.size, np.nan)
        for first, second in pairs:
            # Three entries per segment: the two ends and a NaN gap that keeps the next
            # segment from being joined to it.
            for along, into in ((0, drawn_x), (1, drawn_y)):
                ends = (edges[first][along][i, j], edges[second][along][i, j], gap)
                into.append(np.stack(ends, axis=1).ravel())
    if not drawn_x:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    return np.concatenate(drawn_x), np.concatenate(drawn_y)


def _crossings(
    corners: tuple[np.ndarray, ...], x0: float, y0: float, across: float, up: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Linearly interpolate where the contour meets each of a cell's four edges.

    All four are computed for every cell, which is simpler than computing only the
    ones each case needs, and cheap for grids of at most a quarter million cells.
    """
    shape = corners[0].shape
    i = np.arange(shape[0], dtype=np.float64)[:, None] + np.zeros(shape[1])
    j = np.arange(shape[1], dtype=np.float64)[None, :] + np.zeros((shape[0], 1))
    left, bottom = x0 + i * across, y0 + j * up

    def between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        span = b - a
        share = np.where(span == 0.0, 0.5, -a / np.where(span == 0.0, 1.0, span))
        return np.clip(share, 0.0, 1.0)

    return [
        (left + between(corners[0], corners[1]) * across, bottom),
        (left + across, bottom + between(corners[1], corners[2]) * up),
        (left + between(corners[3], corners[2]) * across, bottom + up),
        (left, bottom + between(corners[0], corners[3]) * up),
    ]
