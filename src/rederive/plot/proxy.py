"""The app side of the plot host: one child process, spawned when first needed.

Shaped like `engine.remote` and different from it in the two ways that matter.
The engine worker is stateless and killed on purpose; the plot host is stateful
- it *is* the windows - so nothing here ever kills it, and its death is always
someone else's doing. And an engine call blocks until the answer is computed,
while a plot request is answered the moment the host has taken it: the drawing
happens afterwards, and what it has to say for itself comes back later as an
event.

So there are two directions of traffic on one pipe. A reader thread owns the
receiving end: replies it hands to whichever call is waiting, events it hands
to the callback the app registered. The app's callback goes through Textual's
`call_from_thread` and lands on the message line, which is why nothing here
touches the screen and why the callback is allowed to be slow.

Spawning is lazy because the host costs a Qt toolkit and a sympy import, and a
session that never plots must not pay for either. It is also lazy in the other
direction: after the host dies the next Plot command starts a new one, so the
recovery path is the ordinary path with nothing in the way of it. Open windows
die with the host, which the app reports and does not try to restore - view
state is deliberately not persisted anywhere, so there is nothing to restore.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import sys
import threading
from collections.abc import Callable, Iterator
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess

from rederive.plot import protocol
from rederive.plot.protocol import Event, Refused, Reply, Request

__all__ = ["DIED", "PlotError", "PlotProxy"]

#: How long to wait for a freshly spawned host to say it has imported Qt,
#: sympy and pyqtgraph and opened its connection to the display. Generous: a
#: cold filesystem is slow and the wait happens once per session.
READY_TIMEOUT = 60.0

#: How long a request may wait for its acknowledgement. Requests are answered
#: before any drawing is done, so this is a deadlock guard and not a budget:
#: nothing the host does between receiving a request and acknowledging it can
#: take a second.
REPLY_TIMEOUT = 30.0

#: What the app says when the host is gone. The one message that is about the
#: process rather than about the picture.
DIED = "plot window process died"


@contextlib.contextmanager
def _real_stderr() -> Iterator[None]:
    """The process's own error stream, for as long as a spawn takes.

    Textual redirects `sys.stderr` at an object that answers -1 to `fileno`,
    which is honest - a screen is not a file descriptor - and fatal to the
    first spawn of the session: the multiprocessing resource tracker hands the
    error stream's descriptor to the child it starts, and -1 is not one, so the
    whole spawn is refused with `bad value(s) in fds_to_keep`. Putting the real
    stream back around `start` costs nothing and is the only place in the
    program that has to know the screen is not a file.
    """
    original = sys.stderr
    if sys.__stderr__ is None or original is sys.__stderr__:
        yield
        return
    sys.stderr = sys.__stderr__
    try:
        yield
    finally:
        sys.stderr = original


class PlotError(Exception):
    """A plot request could not be delivered, or was refused in words.

    One exception for both because the app does the same thing with them: the
    message goes on the message line behind `Plot: `. Which of the two it was
    only matters to the proxy, which respawns after the first kind.
    """


class PlotProxy:
    """The plot host as four method calls, with a process behind them or not.

    One request is in flight at a time. The host is single-threaded from the
    protocol's point of view - its Qt loop takes requests in the order they
    arrive - and nothing the app does needs two answers at once.
    """

    def __init__(self, events: Callable[[Event], None] | None = None) -> None:
        #: Where asynchronous events go. Called on the reader thread, so the
        #: app's own callback is what marshals them onto its event loop.
        self.events = events
        # Spawn on every platform, never fork: the app has Textual's threads
        # and event loop in it, and a forked copy of those in a process that is
        # about to start a Qt event loop is not something to reason about.
        self._context: BaseContext = multiprocessing.get_context("spawn")
        self._call = threading.Lock()
        self._process: BaseProcess | None = None
        self._connection: Connection | None = None
        self._reader: threading.Thread | None = None
        #: The reply the reader thread has taken off the pipe, and the number
        #: it belongs to. A reply wearing another number is an answer to a call
        #: that was abandoned, and is dropped.
        self._replies: dict[int, Reply] = {}
        self._arrived = threading.Condition()
        self._number = 0
        self._ready = False
        self._stopped = False
        #: The sticky plot preferences the app last held - read from the
        #: settings store, whether a state file or a host's own report put
        #: them there - and whether the host has yet to be told about them. A
        #: host that has just started always has: it knows only the protocol's
        #: own defaults.
        self._preferences: protocol.Prefer | None = None
        self._unsaid = False
        #: Why the host is gone, once it is. Cleared by the next spawn, so a
        #: message is reported exactly once.
        self._death: str | None = None
        #: How many hosts have been spawned. Nothing reads it but the tests,
        #: which is where respawn policy is worth asserting about.
        self.starts = 0

    @property
    def running(self) -> bool:
        """Whether there is a host to ask now, without starting one."""
        process = self._process
        return process is not None and process.is_alive()

    # -- the four requests -------------------------------------------------

    def add(self, request: protocol.Add) -> protocol.Placed:
        """Plot an expression, and say which window took it.

        The whole reply comes back rather than the number alone, because the
        acknowledgement message turns on more than the number: whether the
        plot replaced a curve already there is the host's to know.
        """
        reply = self._ask(request)
        if isinstance(reply, protocol.Placed):
            return reply
        raise PlotError(_unexpected(reply))

    def prefer(self, preferences: protocol.Prefer) -> None:
        """Remember what a new window and a new plot are to be built with.

        Nothing is sent from here, and nothing is started. What preferences
        matter to is the next window and the next plot, so they travel in front
        of the next request instead of as traffic of their own - which is why
        this may be called straight from the app's event loop, as the settings
        watcher does: it stores four values and returns.
        """
        if preferences == self._preferences:
            return
        self._preferences = preferences
        self._unsaid = True

    def describe(self) -> tuple[protocol.WindowInfo, ...]:
        """What windows are open and what is in them.

        Answers with nothing where no host has been started, since that is the
        truth and starting one to hear it would open a Qt toolkit to say that
        there are no windows.
        """
        if not self.running:
            return ()
        reply = self._ask(protocol.Describe())
        if isinstance(reply, protocol.Windows):
            return reply.windows
        raise PlotError(_unexpected(reply))

    def shutdown(self) -> None:
        """End the host for good, which is what leaving the app comes to.

        Quiet about everything: a host that has already died is a host that
        needs no shutting down, and the app is on its way out either way.
        """
        with self._call:
            self._stopped = True
            connection, process = self._connection, self._process
            if connection is not None and process is not None and process.is_alive():
                try:
                    connection.send((self._next(), protocol.Shutdown()))
                except (OSError, ValueError):
                    pass
            if process is not None:
                process.join(2.0)
            self._clear()

    # -- one request -------------------------------------------------------

    def _ask(self, request: Request) -> Reply:
        """Send one request and wait for the acknowledgement it is owed.

        Preferences the host has not been told about go first, so that the
        window this request may open is built with them. They are one cheap
        message and only ever sent when something has changed.
        """
        with self._call:
            if self._stopped:
                raise PlotError("plotting has been shut down")
            self._require()
            if self._unsaid and self._preferences is not None:
                self._deliver(self._preferences)
                self._unsaid = False
            return self._deliver(request)

    def _deliver(self, request: Request) -> Reply:
        """One request down the pipe, and the acknowledgement it is owed.

        The caller holds the call lock and has seen to it that there is a host
        on the other end.
        """
        number = self._next()
        connection = self._connection
        assert connection is not None
        try:
            connection.send((number, request))
        except (OSError, ValueError):
            raise self._collapsed() from None
        reply = self._await(number, REPLY_TIMEOUT)
        if isinstance(reply, Refused):
            raise PlotError(reply.message)
        return reply

    def _await(self, number: int, timeout: float) -> Reply:
        """Wait for the reader thread to hand over the answer to `number`."""
        with self._arrived:
            while True:
                reply = self._replies.pop(number, None)
                if reply is not None:
                    return reply
                if self._death is not None:
                    raise self._collapsed()
                if not self._arrived.wait(timeout):
                    self._death = DIED
                    raise self._collapsed()

    def _next(self) -> int:
        self._number += 1
        return self._number

    def _require(self) -> None:
        """See that there is a host to ask, or start one and wait for it.

        A handshake that comes back a refusal is a host that could not start -
        a missing wheel, a display Qt would not open - and it says so in its
        own words rather than in the words the death of a process gets. The
        process is gone either way, so the next request starts another.
        """
        if self.running:
            return
        self._clear()
        self._spawn()
        reply = self._await(protocol.READY, READY_TIMEOUT)
        if isinstance(reply, Refused):
            self._clear()
            raise PlotError(reply.message)
        self._ready = True
        # A fresh host knows only the protocol's defaults, whatever it may have
        # been told before it died.
        self._unsaid = self._preferences is not None

    # -- the process -------------------------------------------------------

    def _spawn(self) -> None:
        """Put a host up. Raises `PlotError` where the system will not have one."""
        from rederive.plot import host

        parent = child = None
        try:
            parent, child = self._context.Pipe(duplex=True)
            process = self._context.Process(
                target=host.serve, args=(child,), daemon=True
            )
            with _real_stderr():
                process.start()
        except (OSError, ValueError) as error:
            _close(parent)
            _close(child)
            raise PlotError(f"the plot window could not be started: {error}") from None
        # The child's end belongs to the child now; holding it here would keep
        # the pipe open past its death and turn every death into a hang.
        child.close()
        self.starts += 1
        self._death = None
        self._connection = parent
        self._process = process
        self._reader = threading.Thread(target=self._reading, args=(parent,), daemon=True)
        self._reader.start()

    def _reading(self, connection: Connection) -> None:
        """The reader thread: replies to whoever waits, events to the app.

        It ends when the pipe does, and the end of the pipe is how the death of
        the host is noticed however it died - a crash, a kill, or the shutdown
        the app asked for.
        """
        while True:
            try:
                number, message = connection.recv()
            except (EOFError, OSError, ValueError):
                break
            if number == protocol.EVENT:
                self._event(message)
                continue
            with self._arrived:
                self._replies[number] = message
                self._arrived.notify_all()
        with self._arrived:
            if self._death is None:
                self._death = DIED
            self._arrived.notify_all()

    def _event(self, event: Event) -> None:
        """Hand one asynchronous event to the app, whatever it makes of it.

        A callback that raises must not take the reader thread down with it:
        the app would then hear about nothing further and never learn why.
        """
        callback = self.events
        if callback is None:
            return
        try:
            callback(event)
        except Exception:
            pass

    def _collapsed(self) -> PlotError:
        """The host is gone: clear it away and say so.

        The next request spawns another, which is the whole of the recovery.
        Open windows are not restored: they died with the process, and the
        design keeps no view state anywhere that could put them back.
        """
        message = self._death or DIED
        self._clear()
        return PlotError(message)

    def _clear(self) -> None:
        """Forget the host, whatever became of it."""
        connection, self._connection = self._connection, None
        _close(connection)
        self._process = None
        self._reader = None
        self._ready = False
        with self._arrived:
            self._replies.clear()


def _unexpected(reply: Reply) -> str:
    """What to say about an answer of a kind the request does not take.

    It cannot happen while both sides are this file and `host.py`, and saying
    so beats an assertion in a program whose plot windows are optional.
    """
    return f"the plot window answered with {type(reply).__name__}"


def _close(connection: Connection | None) -> None:
    if connection is None:
        return
    try:
        connection.close()
    except OSError:
        pass
