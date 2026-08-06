"""The arithmetic a view is framed by, in nobody's toolkit.

Framing is never asked about. A fresh window shows x in [-5, 5] with equal
scales - always, so a circle is round - and the mouse does the rest. The one
time a window reframes itself is when the alternative is an empty picture: a
curve added with finite values none of which are in view is fitted, and the
window says so. Everything either of those comes to is here, as functions over
numbers: what the default framing is on a canvas of a given shape, what
rectangle a set of samples wants, whether anything is in view at all, where the
axis lines go when the origin is off the picture, what a point is worth in
polar coordinates, and where the view has been.

Nothing here decides anything. A window asks for a rectangle and then does what
it likes with it - the aspect lock, the padding and the drawing are the
window's, because they are the picture's rather than the mathematics'.

A 3D window is framed by a domain rather than by a view - the mouse turns the
picture and never re-evaluates it - so what it has here is the rectangle it
opens on and how finely that is sampled. They are here rather than in either
window because both backends open on them, and a browser opening on a different
rectangle from a desktop would be two programs.

Numpy is imported where it is used rather than at the top, and the three
functions that use it are the three that take arrays of samples. The rest -
the default framing, where an axis line goes, which reading a view mode gives a
curve, where the view has been - is arithmetic over single numbers, and a
browser's main thread has to be able to ask for it without loading a numerical
library into the instance that paints the screen.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from rederive.plot.protocol import PlotKind

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "DEFAULT_DOMAIN",
    "DEFAULT_GRID",
    "MAX_GRID",
    "History",
    "axis_at",
    "fitting",
    "framing",
    "home_range",
    "polar",
    "reread",
]

#: The default framing: x from -5 to 5, with y following from equal scales.
DEFAULT_HALF_WIDTH = 5.0

#: The rectangle a fresh 3D window evaluates over, and how many samples of it
#: each axis takes. Sixty-four is dense enough that `SIN(x*y)` over the default
#: domain does not alias; the cap is what stops a typed grid from asking for a
#: quarter of a million vertices.
DEFAULT_DOMAIN = (-5.0, 5.0)
DEFAULT_GRID = 64
MAX_GRID = 256

#: A range, as everything here passes one around: two bounds each way.
Ranges = tuple[tuple[float, float], tuple[float, float]]


def home_range(width: float, height: float) -> Ranges:
    """The default framing on a canvas of this shape: the origin centred.

    Both ranges are worked out rather than left to an aspect lock, because a
    lock is free to satisfy itself by widening either axis and which one it
    picks is not something the framing should depend on. The ordinate is
    therefore the abscissa scaled by the shape of the canvas, which is what
    equal scales means.
    """
    half = DEFAULT_HALF_WIDTH * max(height, 1.0) / max(width, 1.0)
    return (-DEFAULT_HALF_WIDTH, DEFAULT_HALF_WIDTH), (-half, half)


def framing(samples: Sequence[tuple[np.ndarray, np.ndarray]]) -> Ranges | None:
    """The rectangle that holds every finite sample given, or None for none.

    What `View all` frames on. The plots with nothing to frame are left out
    rather than counted as nothing: a region agrees with any framing, since it
    is evaluated over whatever the view shows.
    """
    import numpy as np

    xs = [x[np.isfinite(y)] for x, y in samples]
    ys = [y[np.isfinite(y)] for _, y in samples]
    finite = [array for array in ys if array.size]
    if not finite:
        return None
    left = min(float(np.min(array)) for array in xs if array.size)
    right = max(float(np.max(array)) for array in xs if array.size)
    low = min(float(np.min(array)) for array in finite)
    high = max(float(np.max(array)) for array in finite)
    return (left, right), (low, high)


def fitting(xs: np.ndarray, ys: np.ndarray, view: Ranges) -> Ranges | None:
    """The rectangle one plot's samples want, where none of them is in view.

    None where there is something to see already, which is the ordinary case:
    a window whose framing moved under every added curve would be a window
    nobody could compare two curves in.
    """
    import numpy as np

    (left, right), (low, high) = view
    keep = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[keep], ys[keep]
    if not ys.size:
        return None
    inside = (xs >= left) & (xs <= right) & (ys >= low) & (ys <= high)
    if inside.any():
        return None
    return (float(np.min(xs)), float(np.max(xs))), (float(np.min(ys)), float(np.max(ys)))


def axis_at(low: float, high: float, pixel: float) -> float:
    """Where an axis line goes: at zero, or at the edge of the view nearest it.

    A view framed away from the origin would otherwise have no axis line at
    all, and a picture with no reference line in it is hard to read against the
    numbers along its edges. Clamped a pixel inside, so that the line is drawn
    rather than falling on the boundary.
    """
    return min(max(0.0, low + pixel), high - pixel)


def polar(x: float, y: float, degrees: bool) -> tuple[float, float]:
    """A point of the plane as the radius and angle a polar picture is read in.

    The angle is in whatever unit the expressions in the window are written in,
    since a reading in radians beside a curve drawn in degrees is a number
    about nothing.
    """
    import numpy as np

    turn = 180.0 / np.pi if degrees else 1.0
    return float(np.hypot(x, y)), float(np.arctan2(y, x)) * turn


def reread(kind: PlotKind, polar_view: bool) -> PlotKind | None:
    """The kind a view mode reads a plot as, or None where it is unchanged.

    A univariate curve is the one kind with two readings - r = f(θ) while the
    window is polar and y = f(x) while it is not - so those two kinds trade
    places with the toggle and every other kind is its own answer.
    """
    if kind is PlotKind.CURVE and polar_view:
        return PlotKind.POLAR
    if kind is PlotKind.POLAR and not polar_view:
        return PlotKind.CURVE
    return None


class History:
    """Where the view has been, and where in that list it stands.

    Every gesture pushes the range it is about to change before changing it,
    because a range signal arrives after the change and the thing worth keeping
    is what was there before it. Stepping back and then somewhere new throws
    the forward half away, which is what every history in every program does.
    """

    def __init__(self) -> None:
        self._ranges: list[Ranges] = []
        self._at = 0

    def clear(self) -> None:
        self._ranges.clear()
        self._at = 0

    @property
    def last(self) -> Ranges | None:
        """The range pushed most recently, which is where a step back leads."""
        return self._ranges[-1] if self._ranges else None

    def remember(self, current: Ranges) -> None:
        """Push the range as it is now, before something changes it."""
        del self._ranges[self._at :]
        if not self._ranges or self._ranges[-1] != current:
            self._ranges.append(current)
        self._at = len(self._ranges)

    def step(self, direction: int, current: Ranges) -> Ranges | None:
        """The range one step back or forward, or None where there is none."""
        if direction < 0 and self._at == len(self._ranges):
            # Stepping back for the first time has to keep where we are, or
            # there would be nothing to step forward to.
            self.remember(current)
            self._at = len(self._ranges) - 1
        at = self._at + direction
        if not 0 <= at < len(self._ranges):
            return None
        self._at = at
        return self._ranges[at]
