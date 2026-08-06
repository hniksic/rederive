"""The plot windows a browser has, which for now are none, and say so.

The app talks to plotting through five calls - `add`, `prefer`, `describe`,
`shutdown` and the callback events arrive on - and on the desktop the object
behind them is a proxy for a child process holding Qt. A browser has no such
process and will not have one: the windows are to be DOM panes drawn by the
page, which is stage 5's work. What it has meanwhile is this, an object with
the same five calls and nothing behind them, which refuses a plot in a sentence
and lets every other command in the program go on working.

The refusal is not new machinery. `PlotError` is what a plot host that will not
start already raises and what the app already prints behind `Plot: `, so a
browser refusing for a reason of its own reads exactly like a desktop refusing
for want of a wheel. What differs is the sentence, and it differs because the
desktop's names a thing to install and a browser cannot install anything.

Preferences are remembered rather than dropped. They are the settings a plot
window would be built with, they arrive whether or not anything is ever
plotted, and a `prefer` that threw them away would leave stage 5 with a window
built out of defaults on a page whose settings say otherwise.
"""

from __future__ import annotations

from collections.abc import Callable

from rederive.plot import protocol
from rederive.plot.protocol import Event
from rederive.plot.proxy import PlotError

__all__ = ["UNDRAWN", "WebPlots"]

#: What the browser says when asked to plot. The one refusal that is about
#: where the program is running rather than about the expression or the
#: install, so it says which part of the program is missing and not what to do
#: about it - there is nothing a reader could do.
UNDRAWN = "plot windows are not in the browser yet - the desktop program draws them"


class WebPlots:
    """The five calls, answered by a page that cannot draw a plot yet."""

    def __init__(self, events: Callable[[Event], None] | None = None) -> None:
        #: Where asynchronous events would go. Nothing sends any yet: a
        #: refusal is the answer to the request that asked for it, and events
        #: are what a window says about itself afterwards.
        self.events = events
        #: The sticky plot preferences the app last held. Kept rather than
        #: used, since what they are for is the next window.
        self.preferences: protocol.Prefer | None = None

    async def add(self, request: protocol.Add) -> protocol.Placed:
        """Refuse, in the words the message line prints behind `Plot: `.

        Awaited because plotting is awaited: on the desktop the wait is a
        thread blocking on the host's acknowledgement, and here there is
        nothing to wait for at all.
        """
        raise PlotError(UNDRAWN)

    def prefer(self, preferences: protocol.Prefer) -> None:
        self.preferences = preferences

    def describe(self) -> tuple[protocol.WindowInfo, ...]:
        """Nothing is open, which is the truth and not a refusal."""
        return ()

    def shutdown(self) -> None:
        """Nothing to end: there is no process here and no window."""
