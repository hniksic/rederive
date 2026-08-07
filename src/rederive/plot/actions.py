"""What a command does to a view, and nothing that draws one.

`plot.view` has the arithmetic - what a default framing is, what rectangle a
set of samples wants, where the view has been. What was missing beside it is
the answer to the question a command asks: the user pressed Back, so what is
the view now, is the aspect lock still on, and what does the window have to say
about it. That answer is the same in a window and in a page, and it is here.

A `Framing` is therefore the whole of a 2D view's state bar the ranges
themselves: where the view has been, whether the scales are locked equal, and
whether anybody has moved it off the framing it opened with. The ranges stay
with the backend, which is the side that draws them and the side a drag talks
to; every method here is handed the range the view is at and answers with the
range it should go to. `Framed` is that answer - a rectangle to look at and a
sentence to say, either of which may be nothing.

The 3D half is smaller, because a 3D window is a camera and a camera has no
history: what is here is where it opens, the three presets, and how far one key
press turns it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rederive.plot import view
from rederive.plot.view import Ranges

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "AUTOSCALED",
    "AUTOSCALED_Y",
    "CAMERA",
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
    "facing",
]

#: How far the arrow keys pan a 2D view, as a fraction of it, and by what
#: factor a zoom key changes the scale.
PAN_SHARE = 0.25
ZOOM_FACTOR = 2.0

#: How much room a framing that is fitting something leaves around it, as a
#: fraction of the rectangle: a curve drawn along the very edge of the canvas
#: is a curve half of which is off it.
FIT_PADDING = 0.05
FRESH_PADDING = 0.1

#: What a window says when a plot arrives with nothing of it in view and the
#: framing has to move for there to be a picture at all. A function is framed
#: in y alone - its abscissa is the view, which is what it was sampled over -
#: while a parametric curve or a matrix of points can be off the screen in both
#: directions and is framed in both.
AUTOSCALED_Y = "{named}: y autoscaled to fit"
AUTOSCALED = "{named}: autoscaled to fit"

#: What a window says when `View all` has been asked for. The one framing
#: command worth a word either way: it either moved the view a long way, which
#: is worth confirming, or it found nothing to move it around, which looks
#: exactly like a command that did nothing at all.
FRAMED_ALL = "Framed on everything drawn"
NOTHING_FRAMED = "Nothing to frame yet"

#: What a window says when the polar reading goes on and off. The reading is
#: the view's rather than one plot's, so the sentence is about every curve in
#: the window.
POLAR_ON = "Polar: curves are read as r = f(θ)"
POLAR_OFF = "Polar off: curves are read as y = f(x)"


@dataclass(frozen=True)
class Framed:
    """Where a framing command leaves the view, and what it has to say about it.

    Ranges of None mean the command found nothing to do - a step back through
    an empty history, a plot with no finite samples - and a window that applies
    one of these is meant to notice and leave the view alone.
    """

    ranges: Ranges | None = None
    message: str = ""
    #: How much room to leave around the rectangle. A framing that was asked
    #: for exactly takes none; one that is fitting samples leaves a little.
    padding: float = 0.0
    #: Whether the ordinate alone is to be framed, the abscissa being the range
    #: the samples were taken over already.
    ordinate: bool = False


class Framing:
    """Where a 2D view has been, and what a command does to it next.

    Three things live here and none of them is a range. The history, because
    Back is a command and not a gesture; the aspect lock, because every framing
    that is about fitting releases it and Home takes it back, and something has
    to remember which; and whether the view is still the one the window opened
    with, since the default framing is a thing to be given up rather than
    returned to - a window nobody has moved is reframed when it is resized, and
    one that has been moved is not.

    Every method takes `here`, the range the view is at now, because the
    backend is the side that holds it: a page's interaction loop must be able
    to answer a drag without asking Python where the view is.
    """

    def __init__(self) -> None:
        self.history = view.History()
        #: Whether the two axes are held to the same scale, so that a circle
        #: is round.
        self.equal = True
        #: Whether the view is still the framing the window opened with.
        self.default = True

    def remember(self, here: Ranges | None) -> None:
        """Push the range as it is now, before something changes it.

        Called by every gesture rather than by a range signal, because a signal
        arrives after the change and the thing worth keeping is what was there
        before it.
        """
        if here is None:
            return
        self.default = False
        self.history.remember(here)

    def home(self, here: Ranges | None, width: float, height: float) -> Framed:
        """The default framing: x in [-5, 5], the origin centred, equal scales.

        The scales relock, so that Home is the framing every window opens with
        and a released lock is a departure from it like any pan or zoom.
        """
        self.remember(here)
        self.equal = True
        self.default = True
        return Framed(view.home_range(width, height))

    def stepped(self, here: Ranges | None, direction: int) -> Framed:
        """Where the view has been, one step back or forward."""
        if here is None:
            return Framed()
        return Framed(self.unlocked(self.history.step(direction, here)))

    def everything(
        self, here: Ranges | None, samples: list[tuple[np.ndarray, np.ndarray]]
    ) -> Framed:
        """Frame every sample given, which is what `View all` does.

        The plots with nothing to frame are the caller's to leave out: a region
        is evaluated over whatever the view shows and so agrees with any
        framing at all.
        """
        self.remember(here)
        self.equal = False
        return Framed(view.framing(samples), padding=FIT_PADDING)

    def typed(self, here: Ranges | None, ranges: Ranges) -> Framed:
        """A framing written out rather than gestured, which is four numbers.

        The history is pushed first, so a range typed by mistake is one step
        from the range it replaced, and the lock goes the way it goes for every
        framing that is about fitting: somebody who asks for these four numbers
        is asking for these four numbers.
        """
        self.remember(here)
        return Framed(self.unlocked(ranges))

    def centred(self, here: Ranges | None, x: float, y: float) -> Framed:
        """Put a point in the middle of the canvas, the view keeping its size."""
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
        """Reframe for a new plot that has points but none of them in view.

        Exactly when the alternative is an empty picture, and never otherwise:
        a window whose framing moved under every added curve would be a window
        nobody could compare two curves in. `graph` says the plot is drawn
        against the abscissa and is therefore to be framed in y alone.
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
        """Release equal scales for a framing that is about fitting.

        Answers with what it was given, so that the release and the range it is
        the price of are one expression rather than two statements that have to
        be kept in the same order.
        """
        if ranges is not None:
            self.equal = False
        return ranges


@dataclass(frozen=True)
class Camera:
    """Where a 3D window is looked at from: elevation, azimuth and distance.

    A distance of zero is no distance at all rather than a near one: the three
    presets turn the camera and leave it where it stands, so how far out the
    reader is looking from is not theirs to say.
    """

    elevation: float
    azimuth: float
    distance: float = 0.0


#: The camera a fresh 3D window opens with and `Home view` returns to: a
#: three-quarter view from above, far enough out that the whole cube is in it.
CAMERA = Camera(elevation=30.0, azimuth=-60.0, distance=23.0)

#: The three presets, which face the xy plane from above, the xz plane from the
#: front and the yz plane from the side.
PLANES = {
    "xy": Camera(elevation=90.0, azimuth=-90.0),
    "xz": Camera(elevation=0.0, azimuth=-90.0),
    "yz": Camera(elevation=0.0, azimuth=0.0),
}

#: How far one arrow key turns the camera, and how fast a rotation turns it by
#: itself. A rotation is for reading a shape from every side while the eye
#: stays still, so it is slow: a turn takes half a minute.
ORBIT_DEGREES = 5.0
SPIN_MS = 33
SPIN_DEGREES = 0.4

FACING = "Facing the {plane} plane"
ROTATING = "Rotating - {key} stops it"


def facing(plane: str) -> Camera:
    """Where to look from to have one coordinate plane face the reader."""
    return PLANES[plane]
