"""Plot requests from the browser's main thread to the engine worker.

Like `plot/protocol.py`, this imports no sympy, numpy or toolkit, and the requests are
picklable for sending over `postMessage`. `plot/protocol.py` is what the plot session
speaks to whatever draws; this is what the browser backend sends to the worker.

Each request carries the whole `Plot`, which contains no numbers, so the worker can
build the same record as the desktop window and pass it to `plot/resample.py`.

Answers aren't defined here. Sampling answers are typed arrays sent to the page's
JavaScript; only the request number and any error come back to the main thread, via
the page after it has drawn.
"""

from __future__ import annotations

from dataclasses import dataclass

from rederive.engine.context import Context
from rederive.model.expr import Node
from rederive.plot.model import Plot, Surface

__all__ = ["DRAWING", "Features", "Grid", "Numbers", "Sample", "Trace"]

#: Method names in the worker's request table, alongside the engine calls. A separate
#: channel would gain nothing, since the worker is single-threaded.
SAMPLE = "plot_sample"
TRACE = "plot_trace"
FEATURES = "plot_features"
GRID = "plot_grid"
NUMBERS = "plot_numbers"

#: Requests answered with arrays sent to the page. Others (only `Numbers`) are answered
#: with a pickle to the main thread's Python.
DRAWING = (SAMPLE, TRACE, FEATURES, GRID)


@dataclass(frozen=True)
class Sample:
    """Sample one plot for one view, as described in `plot/resample.py`.

    `pane` and `plot` identify the pane and the curve in it. A plot's real identity (a
    worksheet and a label) is too much to send on every drag, so the pane assigns a
    serial number when the plot is added.

    `generation` works as on the desktop plot record: an answer for a generation the
    pane has moved past is for a stale view and is dropped.
    """

    pane: int
    plot: int
    generation: int
    model: Plot
    xrange: tuple[float, float]
    yrange: tuple[float, float]
    size: tuple[float, float]
    degrees: bool = False


@dataclass(frozen=True)
class Trace:
    """Evaluate one curve at one point for the trace marker.

    The value comes from the closure at the exact point, not from the nearest sample,
    so the request goes to the worker. `at` is the x value for a function curve and the
    parameter for a parametric one.
    """

    pane: int
    plot: int
    generation: int
    model: Plot
    at: float
    degrees: bool = False


@dataclass(frozen=True)
class Features:
    """Find the notable points of one curve within its view.

    Roots, extrema and crossings are located on the samples and refined on the closure.
    The worker resamples the view rather than keeping samples, which the closure cache
    makes cheap and the statelessness contract requires. The whole list is returned,
    so stepping through it needs no further round trips.

    `others` are the curves to find crossings with; the answer refers to them by index.
    """

    pane: int
    plot: int
    generation: int
    model: Plot
    others: tuple[Plot, ...]
    xrange: tuple[float, float]
    yrange: tuple[float, float]
    size: tuple[float, float]


@dataclass(frozen=True)
class Numbers:
    """Evaluate typed view bounds, which can be expressions like `-π`.

    The only request answered with numbers rather than a picture. `plot/forms.py`
    parses the text on the main thread, so the request carries trees. A tree with no
    finite value comes back as NaN, which the form reports.
    """

    pane: int
    nodes: tuple[Node, ...]
    context: Context


@dataclass(frozen=True)
class Grid:
    """Build the geometry of one surface over one domain.

    The only 3D request. Rotating the camera needs no evaluation; only typing a new
    domain does. The answer is `plot/surface.py`'s geometry (vertices, triangles,
    wireframe and per-vertex colors), the same as pyqtgraph gets on the desktop.

    `grid` is the number of samples along each domain axis. Unlike curves, which are
    sampled to match pixels, surfaces use the requested grid.

    `zrange` is the pane's existing box, so that several surfaces share one box. It is
    `None` for the first surface in a pane, whose box is computed from its values.
    """

    pane: int
    plot: int
    generation: int
    model: Surface
    xdomain: tuple[float, float]
    ydomain: tuple[float, float]
    grid: tuple[int, int]
    zrange: tuple[float, float] | None = None
