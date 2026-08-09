"""The plot windows a browser has: panes in the page, and no process behind them.

The app talks to plotting through a handful of calls - `add`, `prefer`,
`describe`, `release`, `shutdown` and the callback events arrive on - and on the
desktop the object behind them is a proxy for a child process holding Qt. A
browser has no such process and needs none: the windows are panes in the page
this app is already drawing on, so what those calls reach is the plot session
itself, in this interpreter, with nothing between them and it.

That is the whole of the browser's answer to the plot host. `PlotSession` is the
same class the desktop runs in its host process, holding the same window
registry and the same keyed job queue; what differs is the backend it opens
windows on and the executor it samples through. So a browser and a desktop route
a plot, pick a receiver and apply a preference by running the same code, and the
two implementations meet only at the two protocols stage 4 drew.

`add` is a coroutine because plotting is awaited. On the desktop the wait is a
thread blocking on the host's acknowledgement; here the answer is known before
the call returns, and the coroutine is what keeps one shape in the app for both
worlds. A refusal is raised as `PlotError`, which is what a host that will not
start already raises and what the app already prints behind `Plot: `.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rederive.plot import protocol
from rederive.plot.protocol import Event
from rederive.plot.proxy import PlotError
from rederive.plot.session import PlotSession, said
from rederive.plot.web.backend import WebBackend, WorkerExecutor

__all__ = ["WebPlots"]


class WebPlots:
    """The same calls, answered by a plot session whose windows are panes."""

    def __init__(self, page: Any, solids: Any, engine: Any) -> None:
        executor = WorkerExecutor(engine)
        backend = WebBackend(page, solids, engine)
        # The one thing either drawing module says that is about no pane in
        # particular: a sampling has been drawn. It goes to the executor, which
        # is what lets go of the request in flight and sends the next one. Both
        # modules are told, because a request in flight may be a curve's or a
        # surface's and the queue is one queue.
        page.attend(backend.handed({"landed": executor.landed}))
        solids.attend(backend.handed({"landed": executor.landed}))
        # A terminated worker takes its sampling with it, and the executor is
        # the side that would otherwise wait forever for the answer.
        engine.lost = executor.lost
        self.session = PlotSession(backend, executor)

    @property
    def events(self) -> Callable[[Event], None] | None:
        """Where asynchronous events go - a pane closed, a curve that would not
        evaluate, a point sent home. The app assigns it, as it does on a proxy."""
        return self.session.events

    @events.setter
    def events(self, listener: Callable[[Event], None] | None) -> None:
        self.session.events = listener

    async def add(self, request: protocol.Add) -> protocol.Placed:
        """Draw one expression, and say which pane took it.

        The refusals are the session's own where it has one, and this call's
        where a window could not be built at all. Both come out as the one
        sentence the message line prints.
        """
        try:
            reply = self.session.add(request)
        except Exception as error:
            raise PlotError(said(error)) from None
        if isinstance(reply, protocol.Refused):
            raise PlotError(reply.message)
        assert isinstance(reply, protocol.Placed)
        return reply

    def prefer(self, preferences: protocol.Prefer) -> None:
        self.session.prefer(preferences)

    async def release(self, finished: bool = False) -> None:
        """The demonstration is over, and its pane goes where it ran to its end.

        A coroutine because the proxy's is one, and its is one because a host
        is a process to wait for. Here the answer is known before the call
        returns, exactly as `add`'s is.
        """
        self.session.release(finished)

    def describe(self) -> tuple[protocol.WindowInfo, ...]:
        """What panes are open and what is in them."""
        return self.session.describe().windows

    def shutdown(self) -> None:
        """Take every pane off the page: the user is leaving the program."""
        self.session.shutdown()
