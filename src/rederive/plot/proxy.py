"""The app side of the plot host, a child process spawned on first use.

It resembles `engine.remote`, with two key differences. The engine worker is stateless
and killed deliberately; the plot host holds the windows, so nothing here kills it. And
an engine call blocks until the answer is computed, while a plot request is answered as
soon as the host accepts it; drawing happens afterwards, and its outcome comes back
later as an event.

A reader thread receives from the pipe, passing replies to the waiting call and events
to the app's callback. The callback goes through Textual's `call_from_thread` to the
message line, so nothing here touches the screen and the callback may be slow.

The host is spawned lazily because it loads Qt and sympy, which a session that never
plots shouldn't pay for. After the host dies, the next Plot command starts a new one.
Open windows die with it and aren't restored, since view state isn't persisted.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
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

#: How long to wait for a new host to import Qt, sympy and pyqtgraph and connect to the
#: display. Generous, since a cold filesystem is slow and this happens once per session.
READY_TIMEOUT = 60.0

#: How long to wait for a request's acknowledgement. Requests are acknowledged before
#: drawing, so this only guards against deadlock.
REPLY_TIMEOUT = 30.0

#: Message shown when the host process is gone.
DIED = "plot window process died"

#: Modules needed to draw plots. They come with the opt-in `plot` extra, so a working
#: install may lack them.
DRAWN_WITH = ("pyqtgraph", "PySide6", "OpenGL")

#: Message shown when those modules aren't installed, saying how to install them.
UNINSTALLED = "plot windows are not installed - `pip install rederive[plot]` adds them"


@contextlib.contextmanager
def _real_stderr() -> Iterator[None]:
    """Restore the real `sys.stderr` while spawning.

    Textual replaces `sys.stderr` with an object whose `fileno` returns -1. The
    multiprocessing resource tracker passes that descriptor to its child, so the first
    spawn would fail with `bad value(s) in fds_to_keep`.
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
    """A plot request could not be delivered, or was refused.

    One exception for both, since the app shows either on the message line after
    `Plot: `. Only the proxy cares which it was: it respawns after a delivery failure.
    """


class PlotProxy:
    """Method-call interface to the plot host process, which may not be running.

    One request is in flight at a time: the host's Qt loop handles requests in order,
    and the app never needs two answers at once.
    """

    def __init__(self, events: Callable[[Event], None] | None = None) -> None:
        #: Callback for asynchronous events. It is called on the reader thread, so it
        #: must marshal them onto the app's event loop.
        self.events = events
        # Always spawn, never fork: forking a process running Textual's threads and
        # event loop, and then starting a Qt loop in it, would be unpredictable.
        self._context: BaseContext = multiprocessing.get_context("spawn")
        self._call = threading.Lock()
        self._process: BaseProcess | None = None
        self._connection: Connection | None = None
        self._reader: threading.Thread | None = None
        #: Replies received by the reader thread, keyed by request number. Replies to
        #: abandoned calls are dropped.
        self._replies: dict[int, Reply] = {}
        self._arrived = threading.Condition()
        self._number = 0
        self._ready = False
        self._stopped = False
        #: The app's current sticky plot preferences (from a state file or the host's
        #: report), and whether the host has yet to be sent them. A newly started host
        #: always needs them, since it knows only the defaults.
        self._preferences: protocol.Prefer | None = None
        self._unsaid = False
        #: Why the host is gone, if it is. Cleared by the next spawn, so the message is
        #: reported once.
        self._death: str | None = None
        #: How many hosts have been spawned, for tests of the respawn policy.
        self.starts = 0

    @property
    def running(self) -> bool:
        """Whether a host is running, without starting one."""
        process = self._process
        return process is not None and process.is_alive()

    # -- the requests ------------------------------------------------------

    async def add(self, request: protocol.Add) -> protocol.Placed:
        """Plot an expression, returning the placement reply.

        Sending blocks until the host acknowledges, and the first send also waits for
        the host to start, so the call runs on a thread from the loop's pool, as an
        engine call does. A browser backend can answer the same call without waiting.

        The whole reply is returned rather than just the window number, because the
        acknowledgement also depends on whether the plot replaced an existing one.
        """
        return await asyncio.to_thread(self.plot, request)

    def plot(self, request: protocol.Add) -> protocol.Placed:
        """The blocking implementation of `add`."""
        reply = self._ask(request)
        if isinstance(reply, protocol.Placed):
            return reply
        raise PlotError(_unexpected(reply))

    def prefer(self, preferences: protocol.Prefer) -> None:
        """Store the preferences for new windows and plots.

        Nothing is sent or started: the preferences are sent before the next request.
        This makes it safe to call from the app's event loop, as the settings watcher
        does.
        """
        if preferences == self._preferences:
            return
        self._preferences = preferences
        self._unsaid = True

    def describe(self) -> tuple[protocol.WindowInfo, ...]:
        """The open windows and their contents.

        Returns nothing if no host is running, rather than starting one just to learn
        there are no windows.
        """
        if not self.running:
            return ()
        reply = self._ask(protocol.Describe())
        if isinstance(reply, protocol.Windows):
            return reply.windows
        raise PlotError(_unexpected(reply))

    async def release(self, finished: bool = False) -> None:
        """Tell the host the demonstration is over.

        This runs on a thread, like `add`, for an important reason. The demonstration
        may have been ended by a key pressed in the plot window, an event the reader
        thread is still delivering to the app's loop. Waiting for the reply on that
        loop would deadlock until the reply timeout, after which a live host would be
        declared dead. In general, nothing that waits for the host may be called from
        the app's event loop.

        Does nothing if no host is running: a demonstration that drew nothing started
        none, and a dead host was already reported by the demonstration step.
        """
        if not self.running:
            return
        await asyncio.to_thread(self._release, finished)

    def _release(self, finished: bool) -> None:
        """The blocking implementation of `release`."""
        try:
            self._ask(protocol.Release(finished))
        except PlotError:
            pass

    def shutdown(self) -> None:
        """Stop the host for good, when the app exits.

        Errors are ignored: a dead host needs no shutdown, and the app is exiting
        anyway.
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
        """Send a request and wait for its acknowledgement.

        Unsent preferences are sent first, so a window opened by this request uses
        them.
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
        """Send a request and wait for its acknowledgement.

        The caller holds the call lock and has ensured the host is running.
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
        """Wait for the reader thread to receive the reply to request `number`."""
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
        """Ensure a host is running, starting one and waiting for it if needed.

        A refused handshake means the host could not start (e.g. a missing wheel or a
        display Qt can't open), and its message is reported instead of the generic
        `DIED`. The next request will start another host.
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
        # A new host knows only the defaults, whatever its predecessor was told.
        self._unsaid = self._preferences is not None

    # -- the process -------------------------------------------------------

    def _spawn(self) -> None:
        """Start a host process, raising `PlotError` if that's impossible.

        The toolkit modules are checked first. The host would report them missing
        anyway, but as an ImportError naming a module rather than the extra, and only
        after a spawn and Python startup. `find_spec` checks without importing, keeping
        the toolkit out of the app process.
        """
        if not all(_installed(name) for name in DRAWN_WITH):
            raise PlotError(UNINSTALLED)

        from rederive.plot.qt import host

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
        # Close our copy of the child's end, or the pipe would stay open after the child
        # dies and we'd hang instead of noticing.
        child.close()
        self.starts += 1
        self._death = None
        self._connection = parent
        self._process = process
        self._reader = threading.Thread(target=self._reading, args=(parent,), daemon=True)
        self._reader.start()

    def _reading(self, connection: Connection) -> None:
        """Reader thread: pass replies to the waiting call and events to the app.

        It ends when the pipe closes, which is how the host's death is detected,
        whether by crash, kill or requested shutdown.
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
        """Pass an asynchronous event to the app's callback.

        Exceptions from the callback are swallowed so they don't kill the reader
        thread, which would silently cut off all further messages.
        """
        callback = self.events
        if callback is None:
            return
        try:
            callback(event)
        except Exception:
            pass

    def _collapsed(self) -> PlotError:
        """Forget the dead host and return the error to raise.

        The next request spawns a new host. Open windows died with the process and are
        not restored, since no view state is kept.
        """
        message = self._death or DIED
        self._clear()
        return PlotError(message)

    def _clear(self) -> None:
        """Forget the host, whatever its state."""
        connection, self._connection = self._connection, None
        _close(connection)
        self._process = None
        self._reader = None
        self._ready = False
        with self._arrived:
            self._replies.clear()


def _installed(name: str) -> bool:
    """Whether a module is installed, checked without importing it.

    A broken install (e.g. PySide6 without its Qt libraries) passes this check and is
    reported by the host; this only checks whether the wheel was installed.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except ImportError:
        return False


def _unexpected(reply: Reply) -> str:
    """The error message for a reply of an unexpected type.

    This can't happen with the current host, but an error message is friendlier than
    an assertion for an optional feature.
    """
    return f"the plot window answered with {type(reply).__name__}"


def _close(connection: Connection | None) -> None:
    if connection is None:
        return
    try:
        connection.close()
    except OSError:
        pass
