"""When a plot is sampled again, and what the sampling is asked to do for it.

Sampling is in screen space and repeats on every view change, debounced. That
is what makes zooming worth doing: a spike narrower than a pixel is not in the
data until the view asks for it, and then it is. So a window that has been
panned, zoomed, resized or reframed asks here for a job, hands the job to the
session's sampler, and draws whatever comes back that is still about the view
it is showing.

Three rules live here rather than in any window, because both backends need
them and neither should invent its own.

**One job and one key for every kind.** A function is sampled over the
abscissa, a parametric curve over its parameter with the error measured as a
distance in the plane, a field over a grid at half the canvas resolution - and
they are one job, keyed by the window and the plot, so a view change debounces
an implicit grid exactly as it debounces a curve.

**A data plot is not re-sampled on a view change.** A matrix of constants is
the same matrix at every zoom; clipping and downsampling are what the view
change costs it, and the drawing side does both itself.

**Everything the job needs is read before it starts.** The job runs somewhere
else - a thread, a worker - and by the time it does the view may have moved and
the plot may be gone, so it closes over values and touches no window.

The generation counter that goes with all this belongs to the plot: a job whose
generation has moved on is a job about a view that is gone, and its answer is
dropped by the side that would have drawn it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rederive.plot import evaluate
from rederive.plot.model import DEFAULT_TURN, DEFAULT_TURN_DEGREES, FIELDS, Plot
from rederive.plot.protocol import PARAMETRIZED, PlotKind

__all__ = ["grid_job", "job", "keyed", "wanted"]

#: How long to wait after the last view change before re-sampling. Long enough
#: that a drag is one re-sample rather than sixty, short enough that letting go
#: of the mouse feels like the end of the gesture.
RESAMPLE_DELAY_MS = 100

#: How fine the grid an implicit curve or a region is evaluated on is, as a
#: fraction of the canvas, and the most points either axis of it may have. Half
#: the canvas resolution is the design's own figure: marching squares
#: interpolates within a cell, so a contour drawn from half-resolution samples
#: is smooth at full resolution.
GRID_SHARE = 0.5
MAX_GRID = 512


def keyed(window: int, plot: Plot) -> tuple[int, int]:
    """The job key of one plot in one window, which is what a re-sample replaces."""
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
    """What the sampling is to do for one 2D plot over one view, as a callable.

    The answer is read by the same kind that asked for it: a pair of arrays for
    a function or a data plot, a sampled path for a parametric curve, a grid for
    a field.
    """
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
    """What the sampling is to do for one surface over one domain.

    The only thing that starts an evaluation in a 3D window. A camera move does
    not come near it, which is the promise that window is built on: the mesh is
    about the domain, and the domain is only ever changed by typing in it.
    """
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
    """What the sampling does for a parametric or polar curve.

    The two closures are made once and kept on the plot, like a function's one
    is, so that a zoom re-samples without asking sympy anything. The first
    sampling is over one turn, so the picture appears with no question asked;
    bounds typed in the range fields arrive as trees - `-π` is what a person
    types - and what they are worth is arithmetic that need only be done once,
    here, before the range they make goes back to the plot.

    A polar curve is a parametric one composed here rather than at the far end:
    r(θ) is kept beside the pair because the readout and the trace are in r and
    θ, and only the composition knows which closure that is.
    """
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
                # A curve reread polar by the view arrives with its closure
                # already made, and r = f(θ) is the same f.
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
    """What the sampling does for an implicit curve or a region.

    Both are fields over the rectangle now in view rather than curves over an
    abscissa, and both are recomputed for whatever the view has become, which
    is what makes zooming into a contour reveal it rather than magnify it.

    Half the canvas resolution is the design's figure and it is not a
    compromise: marching squares interpolates inside a cell, so a contour drawn
    from half-resolution samples is smooth at full resolution, and a grid at
    the full one would cost four times as much to say the same thing.
    """
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
