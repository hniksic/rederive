"""What the main thread asks the engine worker about a picture, and what comes back.

A second vocabulary beside `plot/protocol.py`, and it lives under the same
prohibition: no sympy, no numpy, no toolkit. The app's plot session speaks the
first one to whatever draws; this is what the browser's backend speaks to the
worker that holds the numbers, and the requests are picklable for the same
reason - both ends are the same Python, and `postMessage` carries the bytes.

Every request carries the whole `Plot` rather than a copy of its fields. The
plot is what the sampling is about, it has no numbers in it, and sending it
whole is what lets the worker build exactly the record the desktop's window
builds and hand it to `plot/resample.py` unaltered.

The answers are not here. A sampling answers in arrays, and arrays go to the
page's JavaScript as typed arrays rather than to Python as a pickle: what
crosses back to the main thread is the request number and whatever went wrong,
which the page hands over after it has drawn. That asymmetry is the whole point
of the split - the arrays go where they are drawn, and the words go where they
are said.
"""

from __future__ import annotations

from dataclasses import dataclass

from rederive.engine.context import Context
from rederive.model.expr import Node
from rederive.plot.model import Plot, Surface

__all__ = ["DRAWING", "Features", "Grid", "Numbers", "Sample", "Trace"]

#: The names the requests travel under in the worker's request table. They sit
#: beside the six engine calls rather than in a channel of their own, because
#: the worker is one thread and a second channel would only queue in front of
#: the same interpreter.
SAMPLE = "plot_sample"
TRACE = "plot_trace"
FEATURES = "plot_features"
GRID = "plot_grid"
NUMBERS = "plot_numbers"

#: The four that answer in arrays, which is how the worker's dispatcher tells a
#: request whose answer is the page's from one whose answer is a pickle. What is
#: not on this list comes back to the main thread's Python, `Numbers` being the
#: one such request: four floats are not a picture.
DRAWING = (SAMPLE, TRACE, FEATURES, GRID)


@dataclass(frozen=True)
class Sample:
    """Sample one plot for one view: the job `plot/resample.py` describes.

    `pane` and `plot` are how both ends name the picture and the curve in it -
    the plot's identity is a worksheet and a label, which is too much to carry
    on every drag, so the pane assigns a serial when the plot lands in it and
    both sides use that.

    `generation` is the counter the desktop keeps on its own plot record, doing
    the same job across `postMessage` that it does across a thread: an answer
    wearing a generation the pane has moved past is about a view that is gone,
    and is dropped by the side that would have drawn it.
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
    """What one curve is worth at one place, evaluated rather than read off.

    The closure is what a plot is for: a reading is the function's own value at
    the exact abscissa and never the height of the nearest pixel, so the
    marker's reading is asked of the side that holds the closures. `at` is an
    abscissa for a function curve and a parameter for a parametrized one, which
    is the difference between riding a graph and riding a path.
    """

    pane: int
    plot: int
    generation: int
    model: Plot
    at: float
    degrees: bool = False


@dataclass(frozen=True)
class Features:
    """The notable points of one curve, over the view it is drawn in.

    Roots, extrema and crossings are found on the samples that are drawn and
    refined on the closure, so the request carries the view the samples came
    from and the worker samples again to find them - the closure cache makes
    that cheap, and the alternative would be keeping the samples in a worker
    that promises to keep nothing. What comes back is the whole list, so that
    stepping through it afterwards costs no round trip at all.

    `others` are the curves the crossings are sought against, in the order the
    answer names them by index.
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
    """What typed bounds are worth, asked of the side that can work it out.

    The one request here whose answer is four floats rather than a picture, and
    the reason a page can take `-π` for an edge of a view. The text was read
    where it was typed - `plot/forms.py` parses it under the grammar the plot
    arrived with, and parsing is the half with no mathematics in it - so what
    crosses is trees, and what comes back is what they are worth. A tree that is
    worth no finite number comes back as a NaN, which is a refusal the form
    knows the sentence for.
    """

    pane: int
    nodes: tuple[Node, ...]
    context: Context


@dataclass(frozen=True)
class Grid:
    """One surface over one domain, as the geometry a card can be given.

    The 3D request, and the only thing that starts an evaluation in a 3D pane:
    the camera turns the picture and never comes near here, because the mesh is
    about the domain and the domain is only ever changed by typing in it. What
    comes back is `plot/surface.py`'s answer - vertices, triangles, a wire grid
    and a color a vertex - which is what three.js is handed and what pyqtgraph
    is handed on a desktop.

    `grid` is how many samples each axis of the domain takes, which is the one
    number a 3D window has that a 2D one has not: a curve is sampled to the
    pixels it is drawn on and a surface is sampled to what was asked for.

    `zrange` is the box the vertices are to be placed in, where the pane
    already has one. A pane holding two surfaces draws them in one box or draws
    a picture that lies, so the pane says what the box is and a surface arriving
    into an empty one says what it should be - which is `None` here, and an
    extent worked out from the values themselves.
    """

    pane: int
    plot: int
    generation: int
    model: Surface
    xdomain: tuple[float, float]
    ydomain: tuple[float, float]
    grid: tuple[int, int]
    zrange: tuple[float, float] | None = None
