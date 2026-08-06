"""The far end of a browser plot: closures, samples and readings, in the worker.

This runs beside sympy, in the one interpreter of the pair that is allowed to
know any mathematics, and it is the only part of browser plotting that touches
numpy. What it answers with are arrays, and they go from here to the page's
JavaScript without ever becoming Python objects on the main thread.

The closures are the one thing kept between requests, and keeping them is
licensed rather than smuggled: the worker's statelessness contract carves out
"a content-keyed cache the protocol treats as maybe-empty", and a lambdified
expression is exactly that. It is keyed by what the expression *is* - the tree,
the context it is read under, the names it is a function of - so a re-sample of
a dragged view finds it, a second window plotting the same expression finds it,
and a worker that has just been terminated by Esc finds nothing and rebuilds it.
Nothing else is kept. There is no record here of which pane holds what, because
that is the truth about a session and it lives on the main thread.

The jobs themselves are `plot/resample.py`'s, unaltered: the record built here
carries the same fields the desktop's window carries, so the two backends ask
for the same sampling and differ only in where the answer is drawn.

Two things are written here that the desktop gets from its toolkit. The zero
contour of an implicit grid is marching squares, which pyqtgraph provides on the
desktop and nothing provides here; and a region's shading crosses as the truth
grid itself, one byte per cell, for the page to turn into an image.
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Callable
from typing import Any

import numpy as np

from rederive.plot import evaluate, resample
from rederive.plot.model import (
    FIELDS,
    FUNCTIONS,
    Plot,
    naming,
    pointed,
    reading,
    roughly,
)
from rederive.plot.protocol import PARAMETRIZED, PlotKind
from rederive.plot.web import protocol
from rederive.plot.web.protocol import Features, Sample, Trace

__all__ = ["Closures", "methods"]

#: How many lambdified expressions the cache holds before the oldest goes. A
#: window of eight curves re-sampled on every drag needs eight, and a session
#: that has plotted a dozen things into three windows needs a few more; past
#: that, rebuilding one costs a conversion and nothing is lost.
CACHED = 32

#: What a sampling answer is shaped like, which is how the page reads its
#: fields. A stroke is two arrays of points, whatever kind drew them; a region
#: is a grid of truth values the page shades a rectangle with.
STROKE = "stroke"
REGION = "region"


class Closures:
    """The lambdified expressions this worker has built, keyed by content.

    Insertion-ordered and capped, which is the whole of the eviction policy: a
    cache the protocol treats as maybe-empty owes nobody an accurate one.
    """

    def __init__(self, held: int = CACHED) -> None:
        self._held = held
        self._made: dict[bytes, Any] = {}

    def keyed(self, plot: Plot, names: tuple[str, ...], reading: str) -> bytes:
        """What this expression *is*, as the bytes the cache is keyed by.

        The tree, the context it is read under, the names it is a function of
        and which reading of it is wanted - `u - v` for an implicit curve and
        the truth of `u > v` for a region are two closures over one tree. The
        pickle is the content, since a pickle is what the request arrived as.
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
    """A plot as the sampling holds it: the closures, and nothing drawn.

    The worker's answer to the desktop's `Drawn`. What is added is what a
    sampling reads and writes - the closure, the pair a parametrized curve is
    ridden on - so that `plot/resample.py` can be handed one of these and not
    know which side of the program it is running on.
    """

    closure: Any = None
    pair: Any = None
    xs: Any = None


def methods(post: Callable[[dict[str, Any]], None]) -> dict[str, Callable[..., Any]]:
    """The three plot requests, over one cache of closures.

    `post` is how a partial answer reaches the page: the uniform pass of a
    sampling is drawn while the refinement is still running, and it arrives as
    a message of its own rather than as the answer, which comes later.
    """
    held = Closures()
    return {
        protocol.SAMPLE: lambda request: _sampled(held, request, post),
        protocol.TRACE: lambda request: _traced(held, request),
        protocol.FEATURES: lambda request: _found(held, request),
    }


# -- one sampling -------------------------------------------------------------


def _sampled(
    held: Closures, request: Sample, post: Callable[[dict[str, Any]], None]
) -> dict[str, Any]:
    """One plot over one view, as the arrays the page draws it from."""
    plot = _riding(held, request.model, request.degrees)
    answer = _about(request)
    try:
        work = resample.job(
            plot, request.xrange, request.yrange, request.size, request.degrees
        )
        return answer | _drawn(held, plot, work, request, post)
    except Exception as error:  # the curve's problem, and it says so itself
        return answer | {"shape": STROKE, "trouble": _said(error)}


def _drawn(
    held: Closures,
    plot: Riding,
    work: Callable[..., Any],
    request: Sample,
    post: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Run one job and read its answer the way the kind that asked for it does."""
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
    """A parametric or polar curve, and the note that it gave up refining."""
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
        | {"trange": [float(trange[0]), float(trange[1])], "words": said}
    )


def _field(held: Closures, plot: Riding, answer: Any) -> dict[str, Any]:
    """An implicit contour or a region, over the grid just evaluated.

    A contour is line segments with a gap between the pieces, drawn by whatever
    draws every other stroke. A region is the truth grid itself, one byte a
    cell, laid out the way an image is read - rows from the top - so that the
    page can shade a rectangle with it and nothing has to be transposed twice.
    """
    made, xs, ys, values = answer
    _remember(held, plot, made)
    extent = [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])]
    if plot.kind is PlotKind.IMPLICIT:
        cx, cy = contour(xs, ys, values)
        # An implicit curve gets a sentence of its own where it draws nothing:
        # its grid did evaluate, and what is missing is a solution in the
        # rectangle rather than a real value anywhere.
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
    """The header every answer wears: who it is about, and which view of it."""
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
    """Two arrays as the page takes them: 1-D, contiguous, single precision.

    Single precision because a stroke is drawn on a canvas a thousand pixels
    across and there is nothing in a double that a screen can show; 1-D and
    contiguous because that is the shape that crosses as one typed array with
    one copy, a shape with any more dimensions in it becoming a nested list of
    JavaScript numbers instead.
    """
    return {
        "xs": np.ascontiguousarray(xs, dtype=np.float32).ravel(),
        "ys": np.ascontiguousarray(ys, dtype=np.float32).ravel(),
        "bounds": _bounds(xs, ys),
    }


def _bounds(xs: Any, ys: Any) -> list[float]:
    """The rectangle these samples want, or nothing where none of them is real.

    `plot/view.py` frames a window on rectangles like this one, and the
    rectangle is worked out here because this is where the samples are: what
    the page unions to autoscale is four numbers a curve, and never an array
    it would have to scan itself.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    keep = np.isfinite(xs) & np.isfinite(ys)
    if not keep.any():
        return []
    xs, ys = xs[keep], ys[keep]
    return [float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())]


def _empty(plot: Riding, xs: Any, ys: Any, request: Sample) -> dict[str, Any]:
    """The sentence a plot with nothing to draw in view leaves behind.

    An empty picture with no explanation is the named anti-goal, and the
    sentence names the range it was empty over, since the answer is nearly
    always to look somewhere else.
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


# -- a reading and a scan -----------------------------------------------------


def _traced(held: Closures, request: Trace) -> dict[str, Any]:
    """What the curve is worth where the marker is, and the sentence for it.

    Every number here is formatted where it is computed. The page shows what it
    is handed and writes none of it itself, which is the same rule that keeps
    one spelling of an expression in the program.
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
    """One point of one curve, by the reading its kind is ridden in."""
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
    """The marker over a place the curve has no point at.

    Riding a curve into such a place is how a removable singularity is found,
    so the message is the answer rather than an error.
    """
    return f"Tracing {plot.named}: not real and finite at {name} = {reading(at)}"


def _found(held: Closures, request: Features) -> dict[str, Any]:
    """Every notable point of one curve, for the samples it is drawn from.

    The whole list rather than the next one, because the page steps through it
    with a key and a round trip per keystroke would make Tab feel like a
    question rather than a movement. The samples are taken again here: they are
    what says where to look, the closures they come from are cached, and a
    worker that kept them would be a worker keeping the truth about a picture.
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
    """One notable point, named the way the status line names it."""
    where = f"at x = {reading(found.x)}"
    if found.kind is not evaluate.CROSSING:
        said = f"{found.kind} {where}"
    elif 0 <= found.other < len(crossed):
        said = f"intersection with {crossed[found.other]} {where}"
    else:
        said = f"intersection {where}"
    return {"x": float(found.x), "y": float(found.y), "words": said}


# -- the record a job is run over ---------------------------------------------


def _riding(held: Closures, model: Plot, degrees: bool) -> Riding:
    """The plot as the sampling holds it, with whatever closures are cached.

    The closures are looked up rather than built: a job builds the one it wants
    when it finds none, and `_remember` puts it back. That is what makes the
    cache maybe-empty rather than a thing anything here depends on.
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
        # A polar curve's r(θ) is cached under the same reading a function
        # curve of the same tree is, which is what makes flipping the view's
        # polar toggle cost no conversion at all; composing the pair from it is
        # two multiplications and is not worth keeping.
        plot.pair = evaluate.polar_pair(made, degrees)
    return plot


def _ensured(held: Closures, model: Plot, degrees: bool) -> Riding:
    """The plot with its closures actually built, whatever the cache had.

    A sampling builds its own where it finds none, which is what `job` is; a
    reading and a scan do not, and this is where they get theirs. That is the
    maybe-empty contract in one function: a worker that has just replaced one
    terminated by Esc knows nothing, and is asked for a reading anyway.
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
    """Which of the three readings of a tree this plot wants a closure of."""
    if plot.kind is PlotKind.IMPLICIT:
        return evaluate.difference
    if plot.kind is PlotKind.REGION:
        return evaluate.mask
    return evaluate.closure


def _remember(held: Closures, plot: Riding, made: Any, pair: Any = None) -> None:
    """Keep what a job built, under what the expression is.

    A parametric pair is the pair; everything else, a polar curve included, is
    the single closure - `r(θ)` is what a polar reading is composed from and
    what its readout is taken from, and the composition is not worth a slot.
    """
    kept = pair if plot.kind is PlotKind.PARAMETRIC else made
    if kept is not None:
        held.put(held.keyed(plot, _names(plot), _reading_of(plot)), kept)


def _names(plot: Plot) -> tuple[str, ...]:
    """The variables a plot's closure is a function of, in the axes' order."""
    if plot.kind in FIELDS:
        return plot.axes
    if plot.kind in PARAMETRIZED:
        return plot.options.variables or ("t",)
    if plot.kind in FUNCTIONS:
        return plot.options.variables or ("x",)
    return ()


def _reading_of(plot: Plot) -> str:
    """Which reading of the tree the cached closure is.

    One expression has more than one numeric reading - `u = v` is a difference
    where it is a contour and a truth value where it is a region - so the
    reading is part of what the cache is keyed by.
    """
    if plot.kind is PlotKind.REGION:
        return "mask"
    if plot.kind is PlotKind.IMPLICIT:
        return "difference"
    if plot.kind is PlotKind.PARAMETRIC:
        return "pair"
    return "closure"


def _one(f: Callable[..., np.ndarray], at: float) -> float:
    """A closure at one point, as a float, NaN where it has no value there."""
    try:
        return float(np.asarray(f(np.array([at], dtype=np.float64)))[0])
    except Exception:
        return float("nan")


def _nothing(*_arguments: Any) -> None:
    """A report nobody is listening to, for the jobs that make none."""


def _said(error: Exception) -> str:
    """One line naming an exception, for a message line to carry."""
    text = str(error).strip().splitlines()
    return text[0] if text else type(error).__name__


# -- the zero contour of a grid -----------------------------------------------

#: The four edges of a cell, as the index of the corner each starts at and
#: which way it runs. Marching squares crosses at most two of them, and which
#: two is what the corner signs say.
_BOTTOM, _RIGHT, _TOP, _LEFT = 0, 1, 2, 3

#: Which pair of edges the contour crosses, for each of the sixteen ways the
#: four corners of a cell can be signed. The two ambiguous cases - opposite
#: corners positive - are resolved the same way every time rather than by the
#: centre value: a saddle drawn one way round is a saddle, and a rule that
#: changed with the grid would make a contour flicker under a drag.
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
    """The zero contour of a grid, as one stroke with gaps between its pieces.

    Marching squares over the whole grid at once, answering with loose segments
    rather than joined paths: a stroke drawn with a gap after every segment is
    one polyline to the canvas either way, and joining them would buy nothing
    that is visible.

    Cells with a hole beside them are left out. `SQRT(x) = y` is a curve that
    begins at the origin, not a curve with a wall down the y axis, and the wall
    is what a comparison against a NaN would invent.
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
            # Three columns per segment: the two ends, and the gap that keeps
            # the next segment from being joined to this one.
            for along, into in ((0, drawn_x), (1, drawn_y)):
                ends = (edges[first][along][i, j], edges[second][along][i, j], gap)
                into.append(np.stack(ends, axis=1).ravel())
    if not drawn_x:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    return np.concatenate(drawn_x), np.concatenate(drawn_y)


def _crossings(
    corners: tuple[np.ndarray, ...], x0: float, y0: float, across: float, up: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Where the contour meets each of a cell's four edges, linearly.

    Every cell gets all four, whether or not its own case uses them: computing
    the two that matter per case would be sixteen branches over the same
    arithmetic, and the grid is at most a quarter of a million cells.
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
