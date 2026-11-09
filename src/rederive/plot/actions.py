"""The effect of view commands, independent of any backend.

`plot.view` has the arithmetic (default framing, the rectangle that fits a set of
samples, view history). This module answers what a command like Back does: what the
view becomes, whether the aspect lock stays on, and what the window should say.

`Framing` holds a 2D view's state other than the ranges themselves: the history, the
aspect lock, and whether the view was moved from its initial framing. The ranges stay
with the backend, which draws them and handles drags, so each method takes the current
range and returns a `Framed`: the new rectangle and a message, either of which may be
empty.

For 3D there is only the initial camera, the three presets and the step for one key
press, since a camera has no history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rederive.plot import view
from rederive.plot.model import written
from rederive.plot.view import Ranges

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "AUTOSCALED",
    "AUTOSCALED_Y",
    "CAMERA",
    "CLIPPED",
    "FACING",
    "FRAMED_ALL",
    "NOTHING_FRAMED",
    "ORBIT_DEGREES",
    "PAN_SHARE",
    "PLANES",
    "POLAR_OFF",
    "POLAR_ON",
    "ROTATING",
    "SPIN_DEGREES",
    "SPIN_MS",
    "ZOOM_FACTOR",
    "Camera",
    "Framed",
    "Framing",
    "clipped",
    "facing",
]

#: How far the arrow keys pan a 2D view, as a fraction of it, and the factor by which
#: a zoom key changes the scale.
PAN_SHARE = 0.25
ZOOM_FACTOR = 2.0

#: Margin left around a fitted framing, as a fraction of the rectangle, so curves don't
#: run along the edge of the canvas.
FIT_PADDING = 0.05
FRESH_PADDING = 0.1

#: Messages for a plot that arrived entirely out of view and had to be reframed. A
#: function is framed in y only, since it was sampled over the visible x range; a
#: parametric curve or point matrix is framed in both directions.
AUTOSCALED_Y = "{named}: y autoscaled to fit"
AUTOSCALED = "{named}: autoscaled to fit"

#: Messages for `View all`. It may move the view a long way, which is worth confirming,
#: or find nothing to frame, which would otherwise look like the command did nothing.
FRAMED_ALL = "Framed on everything drawn"
NOTHING_FRAMED = "Nothing to frame yet"

#: Messages for toggling polar mode, which applies to every curve in the window.
POLAR_ON = "Polar: curves are read as r = f(θ)"
POLAR_OFF = "Polar off: curves are read as y = f(x)"


@dataclass(frozen=True)
class Framed:
    """The view a framing command produces, and the message to show.

    Ranges of None mean there was nothing to do (e.g. stepping back through an empty
    history, or a plot with no finite samples), and the view should be left alone.
    """

    ranges: Ranges | None = None
    message: str = ""
    #: Margin around the rectangle: none for an exact framing, a little when fitting
    #: samples.
    padding: float = 0.0
    #: Whether to frame only the ordinate, the abscissa being the range the samples
    #: were taken over.
    ordinate: bool = False


class Framing:
    """State of a 2D view apart from its ranges, and the effect of commands on it.

    It holds the history (for Back and Forward), the aspect lock (released by fitting
    framings, restored by Home), and whether the view is still the initial one: an
    unmoved window is reframed on resize, a moved one is not.

    Every method takes `here`, the current range, because the backend owns it: a page's
    interaction loop must handle a drag without asking Python where the view is.
    """

    def __init__(self) -> None:
        self.history = view.History()
        #: Whether both axes have the same scale, so that a circle is round.
        self.equal = True
        #: Whether the view is still the framing the window opened with.
        self.default = True

    def remember(self, here: Ranges | None) -> None:
        """Push the current range onto the history before something changes it.

        Called by gestures rather than from a range signal, because the signal arrives
        after the change.
        """
        if here is None:
            return
        self.default = False
        self.history.remember(here)

    def home(self, here: Ranges | None, width: float, height: float) -> Framed:
        """The default framing: x in [-5, 5], origin centered, equal scales.

        This relocks the scales, restoring exactly the framing a window opens with.
        """
        self.remember(here)
        self.equal = True
        self.default = True
        return Framed(view.home_range(width, height))

    def stepped(self, here: Ranges | None, direction: int) -> Framed:
        """Step back or forward through the view history."""
        if here is None:
            return Framed()
        return Framed(self.unlocked(self.history.step(direction, here)))

    def everything(
        self, here: Ranges | None, samples: list[tuple[np.ndarray, np.ndarray]]
    ) -> Framed:
        """Frame all the given samples, for `View all`.

        The caller leaves out plots with nothing to frame, such as regions, which are
        evaluated over whatever the view shows.
        """
        self.remember(here)
        self.equal = False
        return Framed(view.framing(samples), padding=FIT_PADDING)

    def typed(self, here: Ranges | None, ranges: Ranges) -> Framed:
        """Frame explicitly typed ranges.

        The history is pushed first, so a mistyped range is one Back away. The aspect
        lock is released so the ranges are shown exactly as typed.
        """
        self.remember(here)
        return Framed(self.unlocked(ranges))

    def centred(self, here: Ranges | None, x: float, y: float) -> Framed:
        """Center the view on a point, keeping its size."""
        if here is None:
            return Framed()
        self.remember(here)
        (left, right), (low, high) = here
        width, height = (right - left) / 2, (high - low) / 2
        return Framed(((x - width, x + width), (y - height, y + height)))

    def fresh(
        self,
        here: Ranges | None,
        xs: np.ndarray,
        ys: np.ndarray,
        graph: bool,
        named: str,
    ) -> Framed:
        """Reframe for a new plot that has points, but none of them in view.

        This happens only when the picture would otherwise be empty; reframing for every
        added curve would make curves impossible to compare. `graph` means the plot is
        drawn against the abscissa and is framed in y only.
        """
        if here is None:
            return Framed()
        fitted = view.fitting(xs, ys, here)
        if fitted is None:
            return Framed()
        self.unlocked(fitted)
        self.default = False
        message = (AUTOSCALED_Y if graph else AUTOSCALED).format(named=named)
        return Framed(fitted, message, padding=FRESH_PADDING, ordinate=graph)

    def unlocked(self, ranges: Ranges | None) -> Ranges | None:
        """Release the aspect lock for a fitting framing, and return `ranges` unchanged.

        Returning the input lets callers release the lock and use the range in one
        expression.
        """
        if ranges is not None:
            self.equal = False
        return ranges


@dataclass(frozen=True)
class Camera:
    """A 3D camera position: elevation, azimuth and distance.

    A distance of zero means "keep the current distance": the plane presets only rotate
    the camera.
    """

    elevation: float
    azimuth: float
    distance: float = 0.0


#: The camera of a new 3D window and of `Home view`: a three-quarter view from above,
#: far enough out to show the whole cube.
CAMERA = Camera(elevation=30.0, azimuth=-60.0, distance=23.0)

#: Presets facing the xy plane from above, the xz plane from the front and the yz plane
#: from the side.
PLANES = {
    "xy": Camera(elevation=90.0, azimuth=-90.0),
    "xz": Camera(elevation=0.0, azimuth=-90.0),
    "yz": Camera(elevation=0.0, azimuth=0.0),
}

#: How far one arrow key turns the camera, and the speed of auto-rotation. Rotation is
#: slow (a full turn takes about half a minute) so the shape can be studied.
ORBIT_DEGREES = 5.0
SPIN_MS = 33
SPIN_DEGREES = 0.4

FACING = "Facing the {plane} plane"
ROTATING = "Rotating - {key} stops it"

#: Message shown when spikes caused the 3D box to be clipped to the middle of the value
#: range, so the user knows part of the surface is cut away.
CLIPPED = "z clipped to the 1st-99th percentile: {low} to {high}"


def facing(plane: str) -> Camera:
    """The camera that faces the given coordinate plane."""
    return PLANES[plane]


def clipped(low: float, high: float) -> str:
    """The message for a clipped box, with numbers formatted as the window shows them."""
    return CLIPPED.format(low=written(low), high=written(high))
