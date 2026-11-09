"""Toolkit-independent arithmetic for framing plot views.

A new window shows x in [-5, 5] with equal scales, so circles are round, and the mouse
does the rest. A window reframes itself only when a newly added curve has finite values
but none in view, and then it says so. This module computes the default framing for a
canvas shape, the rectangle that fits a set of samples, whether anything is in view,
where axis lines go when the origin is off screen, polar coordinates, and the view
history. Windows decide what to do with the results (aspect lock, padding, drawing).

A 3D window is framed by its domain, since the mouse only rotates it. This module has
its default domain, sampling grid, and the shared z range of several surfaces. They are
here so that both backends open on the same picture.

Numpy is imported only inside the functions that take sample arrays, so the rest can be
used from the browser's main thread without loading numpy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rederive.plot.protocol import PlotKind

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "CLIP_FACTOR",
    "DEFAULT_DOMAIN",
    "DEFAULT_GRID",
    "DEFAULT_ZRANGE",
    "MAX_GRID",
    "PERCENTILES",
    "History",
    "Span",
    "axis_at",
    "fitting",
    "framing",
    "home_range",
    "polar",
    "pooled",
    "reread",
]

#: The default framing: x from -5 to 5, with y following from equal scales.
DEFAULT_HALF_WIDTH = 5.0

#: The domain of a new 3D window and its samples per axis. 64 is dense enough that
#: `SIN(x*y)` over the default domain doesn't alias; the cap stops a typed grid from
#: asking for a quarter of a million vertices.
DEFAULT_DOMAIN = (-5.0, 5.0)
DEFAULT_GRID = 64
MAX_GRID = 256

#: The z range of an empty 3D window.
DEFAULT_ZRANGE = (-1.0, 1.0)

#: When to clip the box's z range to the middle percentiles of the values. If the full
#: range is more than four times the middle 98%, there is a spike, and fitting it would
#: flatten everything else.
PERCENTILES = (1.0, 99.0)
CLIP_FACTOR = 4.0

#: A 2D range: (x bounds, y bounds).
Ranges = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class Span:
    """A surface's z range: the full range and the middle (percentile) range.

    Both are reported because whether to clip depends on all surfaces in the window;
    `pooled` decides. Passing four numbers rather than the values lets the browser's
    worker report them to the pane.
    """

    low: float
    high: float
    inner_low: float
    inner_high: float


def pooled(spans: Sequence[Span]) -> tuple[tuple[float, float], bool] | None:
    """The z range shared by all surfaces in a window, and whether it was clipped.

    Surfaces in one window share a vertical scale so they can be compared. The range is
    the union of the full ranges, or, if clipping, the union of the middle ranges.

    Returns None if no surface has a finite value. Uses no numpy, so the browser's main
    thread can call it.
    """
    if not spans:
        return None
    low = min(span.low for span in spans)
    high = max(span.high for span in spans)
    tight = (
        min(span.inner_low for span in spans),
        max(span.inner_high for span in spans),
    )
    clipped = False
    if tight[1] > tight[0] and (high - low) > CLIP_FACTOR * (tight[1] - tight[0]):
        low, high, clipped = tight[0], tight[1], True
    if high <= low:
        # A plane would give the box zero height, which can't be divided by.
        low, high = low - 1.0, high + 1.0
    return (low, high), clipped


def home_range(width: float, height: float) -> Ranges:
    """The default framing for a canvas of this shape, with the origin centered.

    The y range is computed from the canvas shape rather than left to the aspect lock,
    which might widen either axis.
    """
    half = DEFAULT_HALF_WIDTH * max(height, 1.0) / max(width, 1.0)
    return (-DEFAULT_HALF_WIDTH, DEFAULT_HALF_WIDTH), (-half, half)


def framing(samples: Sequence[tuple[np.ndarray, np.ndarray]]) -> Ranges | None:
    """The rectangle holding all finite samples, or None if there are none.

    Used by `View all`. Plots with nothing to frame, such as regions, are left out by the
    caller.
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
    """The rectangle fitting one plot's samples, if none of them is in view.

    Returns None if some sample is already visible, which is the usual case.
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
    """Where to draw an axis line: at zero, or at the nearest edge of the view.

    This keeps a reference line visible when the origin is out of view. It is clamped a
    pixel inside so the line doesn't fall on the boundary.
    """
    return min(max(0.0, low + pixel), high - pixel)


def polar(x: float, y: float, degrees: bool) -> tuple[float, float]:
    """Convert a point to polar radius and angle.

    The angle is in the unit the window's expressions use (degrees or radians).
    """
    import numpy as np

    turn = 180.0 / np.pi if degrees else 1.0
    return float(np.hypot(x, y)), float(np.arctan2(y, x)) * turn


def reread(kind: PlotKind, polar_view: bool) -> PlotKind | None:
    """The kind a plot becomes in the given view mode, or None if unchanged.

    Only univariate curves change: they are r = f(θ) in polar mode and y = f(x)
    otherwise.
    """
    if kind is PlotKind.CURVE and polar_view:
        return PlotKind.POLAR
    if kind is PlotKind.POLAR and not polar_view:
        return PlotKind.CURVE
    return None


class History:
    """The view history and the current position in it.

    Gestures push the current range before changing it, because range signals arrive
    after the change. As usual, moving somewhere new after stepping back discards the
    forward history.
    """

    def __init__(self) -> None:
        self._ranges: list[Ranges] = []
        self._at = 0

    def clear(self) -> None:
        self._ranges.clear()
        self._at = 0

    @property
    def last(self) -> Ranges | None:
        """The most recently pushed range."""
        return self._ranges[-1] if self._ranges else None

    def remember(self, current: Ranges) -> None:
        """Push the current range before something changes it."""
        del self._ranges[self._at :]
        if not self._ranges or self._ranges[-1] != current:
            self._ranges.append(current)
        self._at = len(self._ranges)

    def step(self, direction: int, current: Ranges) -> Ranges | None:
        """The range one step back or forward, or None if there is none."""
        if direction < 0 and self._at == len(self._ranges):
            # On the first step back, save the current range so we can step forward to it.
            self.remember(current)
            self._at = len(self._ranges) - 1
        at = self._at + direction
        if not 0 <= at < len(self._ranges):
            return None
        self._at = at
        return self._ranges[at]
