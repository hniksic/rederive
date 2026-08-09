"""What a plot window is opened on: the whole of what the session asks of a toolkit.

The plot session below this knows how many windows there are, which of them the
next plot lands in, what a `Describe` says and what a preference does to the
next picture. What it does not know is what a window *is* - Qt widgets on the
desktop, panes in a page in the browser - and this is the line between the two.

Two protocols and no base classes. A backend opens windows and can be stopped;
a window takes plots, gives them up, says what it holds and comes to the front
when a plot lands in it. Everything else a window does - the framing, the
gestures, the sampling it asks for, the menus - it does on its own, and reports
back through the session it was opened with. The session is handed to `open`
rather than kept by the backend because that report is the traffic that matters:
a window says it was touched, that it was closed, that a curve would not
evaluate, that a sticky control was moved.
"""

from __future__ import annotations

from typing import Any, Protocol

from rederive.plot.model import Plot
from rederive.plot.protocol import Prefer, WindowInfo, WindowKind

__all__ = ["Backend", "WindowHandle"]


class WindowHandle(Protocol):
    """One open plot window, as the session sees it."""

    #: The number that names this window for the life of the session, and which
    #: kind of picture it holds. The session routes by both.
    number: int
    kind: WindowKind
    #: What the window holds, in the order it was added. The session reads it to
    #: find the elements of a family that a shorter family has left behind.
    plots: list[Any]

    def add(self, plot: Plot) -> Any:
        """Put a plot in the window, replacing one with the same identity."""

    def remove(self, plot: Any) -> None:
        """Take one plot out of the window."""

    def find(self, worksheet: int, label: str) -> Any | None:
        """The plot a worksheet and a label name, if this window has it."""

    def present(self, demonstrating: bool = False) -> None:
        """Show this window and bring it to the front: a plot has landed in it.

        `demonstrating` says the plot is a step of a demonstration, which the
        program is now waiting over a key for. Where a window can be above
        another one it then stays above until `release`, so that the program
        waiting for that key - a terminal, and usually a large one - cannot
        bury the picture it is waiting on.
        """

    def release(self) -> None:
        """No demonstration is waiting on this window any more.

        What `present` did for a demonstration is undone here, and only that:
        the window keeps its place, its plots and its title. An environment
        where windows have no order to be kept at the front of has nothing to
        do about this.
        """

    def retitle(self, current: bool | None = None) -> None:
        """Title the window by what it holds, saying whether it is the receiver."""

    def describe(self) -> WindowInfo:
        """This window as `Describe` reports it."""

    def close(self) -> None:
        """Close this window, which reports itself closed as any other close does."""


class Backend(Protocol):
    """What the session opens windows on."""

    def open(
        self, session: Any, kind: WindowKind, number: int, preferences: Prefer
    ) -> WindowHandle:
        """A new window of `kind`, numbered, built with the sticky preferences.

        The preferences are read here rather than kept by the window, which is
        what makes a changed preference show in the next window and leave the
        open ones as they are.
        """

    def close(self, number: int) -> None:
        """Forget a window that has closed itself: its own event said so."""

    def stop(self) -> None:
        """End every window and whatever runs them. The session is going away."""
