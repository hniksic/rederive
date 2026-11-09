"""Resampling policy: when a plot is sampled again, and the sampling job for it.

Sampling is done in screen space and repeated, debounced, on every view change, so
zooming in reveals detail such as spikes narrower than a pixel. A window that was
panned, zoomed, resized or reframed gets a job here, passes it to the session's
sampler, and draws the result if it still matches the current view.

These rules are shared by both backends:

**Every kind uses one job and one key.** A function is sampled over the abscissa, a
parametric curve over its parameter (with error measured as distance in the plane), and
a field over a grid at half the canvas resolution. Jobs are keyed by window and plot, so
all kinds are debounced the same way.

**Data plots are not resampled on view changes.** The drawing side clips and
downsamples them itself.

**A job reads everything it needs up front.** It runs elsewhere (a thread or worker),
by which time the view may have changed or the plot may be gone, so it captures values
and touches no window.

Each plot has a generation counter; results of a job from an outdated generation are
dropped.

`evaluate` is imported inside the job builders, so the browser's main thread can use
this module's policy without loading numpy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rederive.plot.model import DEFAULT_TURN, DEFAULT_TURN_DEGREES, FIELDS, Plot
from rederive.plot.protocol import PARAMETRIZED, PlotKind

__all__ = ["grid_job", "job", "keyed", "wanted"]

#: Delay after the last view change before resampling: long enough that a drag causes
#: one resample rather than sixty, short enough to feel immediate on release.
RESAMPLE_DELAY_MS = 100

#: Resolution of the grid for implicit curves and regions, as a fraction of the canvas
#: resolution, and the maximum points per axis. Half resolution suffices because
#: marching squares interpolates within a cell, so contours are still smooth.
GRID_SHARE = 0.5
MAX_GRID = 512


def keyed(window: int, plot: Plot) -> tuple[int, int]:
    """The job key of a plot in a window; a resample replaces the job with this key."""
    return (window, id(plot))


def wanted(plot: Any) -> bool:
    """Whether a view change asks this plot to be sampled again."""
    return not (plot.kind is PlotKind.DATA and plot.xs.size)


def job(
    plot: Any,
    xrange: tuple[float, float],
    yrange: tuple[float, float],
    size: tuple[float, float],
    degrees: bool = False,
) -> Callable[..., Any]:
    """The sampling job for a 2D plot over a view.

    The result depends on the plot kind: a pair of arrays for a function or data plot,
    a sampled path for a parametric curve, a grid for a field.
    """
    from rederive.plot import evaluate

    node, context = plot.node, plot.context
    if plot.kind in FIELDS:
        return _field(plot, xrange, yrange, size)
    if plot.kind in PARAMETRIZED:
        return _curve(plot, xrange, yrange, size, degrees)
    if plot.kind is PlotKind.DATA:
        return lambda report: evaluate.points(node, context)
    variables = plot.options.variables
    closure = plot.closure

    def work(report: Callable[..., None]) -> Any:
        made = closure
        if made is None:
            made = evaluate.closure(node, context, variables or ("x",))
        xs, ys = evaluate.sample_adaptive(made, xrange, yrange, size, report=report)
        return made, xs, ys

    return work


def grid_job(
    surface: Any,
    xdomain: tuple[float, float],
    ydomain: tuple[float, float],
    grid: tuple[int, int],
) -> Callable[..., Any]:
    """The sampling job for a surface over a domain.

    This is the only evaluation in a 3D window. Camera moves never re-evaluate; only
    typing a new domain does.
    """
    from rederive.plot import evaluate

    node, context, names = surface.node, surface.context, surface.axes
    made = surface.closure
    nx, ny = grid

    def work(_report: Callable[..., None]) -> Any:
        f = made
        if f is None:
            f = evaluate.closure(node, context, names)
        xs, ys, values = evaluate.grid_eval(f, xdomain, ydomain, nx, ny)
        boundary = evaluate.grid_boundary(f, xs, ys, values)
        return f, xs, ys, values, boundary

    return work


def _curve(
    plot: Any,
    xrange: tuple[float, float],
    yrange: tuple[float, float],
    size: tuple[float, float],
    degrees: bool,
) -> Callable[..., Any]:
    """The sampling job for a parametric or polar curve.

    The closures are built once and kept on the plot, so zooming resamples without
    calling sympy. The first sampling is over one full turn. Bounds typed in the range
    fields arrive as expressions (e.g. `-π`) and are evaluated here once.

    A polar curve is turned into a parametric pair here. r(θ) is kept alongside the pair
    because the readout and trace are in r and θ.
    """
    from rederive.plot import evaluate

    node, context, options = plot.node, plot.context, plot.options
    names = options.variables or ("t",)
    made, radius, known = plot.pair, plot.closure, plot.trange
    pending = plot.bounds
    polar = plot.kind is PlotKind.POLAR
    turn = DEFAULT_TURN_DEGREES if degrees else DEFAULT_TURN

    def work(report: Callable[..., None]) -> Any:
        pair, inner, trange = made, radius, known
        if pair is None:
            if polar:
                # A curve switched to polar by the view already has its closure, and
                # r = f(θ) uses the same f.
                if inner is None:
                    inner = evaluate.closure(node, context, names)
                pair = evaluate.polar_pair(inner, degrees)
            else:
                pair = evaluate.pair(node, context, names)
            trange = (-turn, turn)
        if pending is not None:
            low = evaluate.number(pending[0], context, trange[0])
            high = evaluate.number(pending[1], context, trange[1])
            if high > low:
                trange = (low, high)
        sampled = evaluate.sample_curve(
            pair[0], pair[1], trange, xrange, yrange, size, report=report
        )
        return pair, inner, sampled, trange

    return work


def _field(
    plot: Any,
    xrange: tuple[float, float],
    yrange: tuple[float, float],
    size: tuple[float, float],
) -> Callable[..., Any]:
    """The sampling job for an implicit curve or a region.

    Both are fields evaluated over the visible rectangle and recomputed whenever the
    view changes, so zooming into a contour reveals detail rather than magnifying it.
    The grid is at half the canvas resolution (see `GRID_SHARE`).
    """
    from rederive.plot import evaluate

    node, context = plot.node, plot.context
    names = plot.axes
    made = plot.closure
    implicit = plot.kind is PlotKind.IMPLICIT
    nx = int(min(max(size[0] * GRID_SHARE, 2), MAX_GRID))
    ny = int(min(max(size[1] * GRID_SHARE, 2), MAX_GRID))

    def work(_report: Callable[..., None]) -> Any:
        f = made
        if f is None:
            reading = evaluate.difference if implicit else evaluate.mask
            f = reading(node, context, names)
        xs, ys, values = evaluate.grid_eval(f, xrange, yrange, nx, ny)
        return f, xs, ys, values

    return work
