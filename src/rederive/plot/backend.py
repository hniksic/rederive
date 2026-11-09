"""The interface between the plot session and a windowing toolkit.

The session tracks windows, which one receives the next plot, what `Describe` reports
and how preferences apply. What a window actually is (Qt widgets, panes in a browser
page) is up to the backend.

There are two protocols. A backend opens windows and can be stopped; a window adds and
removes plots, describes itself and comes to the front when a plot lands in it.
Everything else (framing, gestures, sampling, menus) the window handles itself, reporting
back through the session passed to `open`: that it was touched or closed, that a curve
failed to evaluate, that a sticky control changed.
"""

from __future__ import annotations

from typing import Any, Protocol

from rederive.plot.model import Plot
from rederive.plot.protocol import Prefer, WindowInfo, WindowKind

__all__ = ["Backend", "WindowHandle"]


class WindowHandle(Protocol):
    """One open plot window, as the session sees it."""

    #: The window's number for the life of the session, and the kind of picture it
    #: holds. The session routes plots by both.
    number: int
    kind: WindowKind
    #: The window's plots in the order added. The session uses it to find leftover
    #: members of a family that was replaced by a shorter one.
    plots: list[Any]

    def add(self, plot: Plot) -> Any:
        """Put a plot in the window, replacing one with the same identity."""

    def remove(self, plot: Any) -> None:
        """Take one plot out of the window."""

    def find(self, worksheet: int, label: str) -> Any | None:
        """The plot a worksheet and a label name, if this window has it."""

    def present(self, demonstrating: bool = False) -> None:
        """Show this window and bring it to the front because a plot landed in it.

        `demonstrating` means the plot is a demonstration step and the program is
        waiting for a key. The window then stays on top until `release`, so the
        terminal waiting for the key doesn't cover it.
        """

    def release(self) -> None:
        """Undo the stay-on-top of a demonstrating `present`.

        The window keeps its position, plots and title. Environments without window
        stacking can do nothing here.
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
        """Open a new window of `kind` with the given number and sticky preferences.

        Preferences are applied only at creation, so a changed preference affects the
        next window but not the open ones.
        """

    def close(self, number: int) -> None:
        """Forget a window that has reported itself closed."""

    def stop(self) -> None:
        """Close every window and shut down the backend as the session ends."""
