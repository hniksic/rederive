"""Browser plot windows: panes in the page, with no host process behind them.

The app plots through a few calls (`add`, `prefer`, `describe`, `release`, `shutdown`
and the events callback). On the desktop these go to a proxy for a child process running
Qt. In the browser the windows are panes in the same page, so the calls go directly to
a `PlotSession` in this interpreter.

`PlotSession` is the same class the desktop host process runs, with the same window
registry and keyed job queue; only the backend and the sampling executor differ. So
routing a plot, picking a receiver and applying preferences run the same code on both
platforms.

`add` is a coroutine because the app awaits plotting. On the desktop it waits for the
host's acknowledgement; here the answer is ready immediately, but the coroutine keeps
one interface for both. Refusals raise `PlotError`, as a host that fails to start does,
and the app prints it after `Plot: `.
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
        # Both drawing modules report "landed" when a sampling has been drawn. The
        # executor then releases the request in flight and sends the next one. Both are
        # wired because the single queue holds both curve and surface requests.
        page.attend(backend.handed({"landed": executor.landed}))
        solids.attend(backend.handed({"landed": executor.landed}))
        # A terminated worker loses its sampling; without this the executor would wait
        # forever for the answer.
        engine.lost = executor.lost
        self.session = PlotSession(backend, executor)

    @property
    def events(self) -> Callable[[Event], None] | None:
        """Listener for asynchronous events, such as a pane closing, a curve failing to
        evaluate, or a point being sent home. The app assigns it, as on a proxy."""
        return self.session.events

    @events.setter
    def events(self, listener: Callable[[Event], None] | None) -> None:
        self.session.events = listener

    async def add(self, request: protocol.Add) -> protocol.Placed:
        """Draw one expression and report which pane took it.

        A refusal by the session or a failure to build a window both raise `PlotError`
        with the sentence the message line prints.
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
        """End the demonstration, leaving its pane where it finished.

        A coroutine to match the proxy, which must wait for the host process.
        """
        self.session.release(finished)

    def describe(self) -> tuple[protocol.WindowInfo, ...]:
        """What panes are open and what is in them."""
        return self.session.describe().windows

    def shutdown(self) -> None:
        """Remove every pane from the page when the user quits."""
        self.session.shutdown()
