"""The child process the plot windows live in: Qt on the main thread, math off it.

The app process cannot host this. It is deliberately free of sympy and it
already has an event loop - Textual's - and a second one cannot be run beside
it. So plotting is a process, spawned when the first plot is asked for and
kept for the rest of the session, and it is the only process in the program
that owns windows.

Three threads, and which is which is the whole design.

The **Qt thread** is the main thread and does nothing but paint and handle
input. Every request that reaches it has already been decoded, and everything
expensive it starts is started somewhere else, so a window stays live however
slow the expression in it is.

The **reader thread** owns the receiving end of the pipe. It decodes requests
and reposts them into the Qt loop as a queued signal, which is the one safe way
to cross into a Qt event loop from outside it. The end of that pipe is the app
going away, and the host goes with it.

The **sampling thread** owns the mathematics: the engine's substitution pass,
the conversion to sympy, the lambdify, and every sample of every curve. It
takes keyed jobs, so a re-sample of the same curve for a newer view replaces
the one that has not started yet, which is what makes a dragged view cost one
sampling and not sixty. There is no abort path and none is needed: sampling is
depth- and size-capped, so the worst a pathological expression can do is take
a while over its own curve.

The host owns the window bookkeeping too, because it is the only place that
knows when a window closes. One pointer, and it is the window the user last
touched: the windows report their own activation events - a click, a raise, an
alt-tab - and the last-activated window of each kind is the *receiver* the
next plot of that kind lands in. A plot's own arrival counts as a touch, since
the host raises the window it draws into, and when the receiver closes it
falls back to the last-activated survivor of its kind. The receiver says so in
its own title, so the user can see where the next plot will land. The app
never tracks any of this; it asks.

The sticky preferences travel both ways. The app tells the host what the next
surface's grid and the next data plot's points are to be; the host reads them
where a window is built and where a plot is added, which is what keeps a
change out of the windows already open. And when a sticky control moves in a
window, the host keeps the new value and reports it up the pipe, because the
app is the side that outlives the host and writes state files.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
from collections.abc import Callable
from dataclasses import replace
from multiprocessing.connection import Connection
from typing import Any

from rederive.plot import protocol
from rederive.plot.protocol import PlotKind, Where, WindowKind

__all__ = ["preferred", "serve"]


def serve(connection: Connection) -> None:
    """Be the plot host until the app goes away or asks it to stop.

    The handshake is sent once the toolkit is up and a window could be opened,
    so a proxy that has heard it knows the next request pays for nothing but
    its own picture. A failure before that point is sent back in words rather
    than left as a dead process: missing wheels and a display that will not
    open are the two ordinary ones, and both want to be read on the message
    line and not guessed at.
    """
    _take_the_tty()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets

        pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")
        # Two 3D windows are two OpenGL contexts, and the shader programs
        # pyqtgraph compiles are cached per program name rather than per
        # context: without this the second 3D window draws with the first
        # one's program handles and every item in it fails. It has to be set
        # before the application exists, which is why it is here and not in
        # the window.
        QtCore.QCoreApplication.setAttribute(
            QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True
        )
        application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        # The host outlives its windows: closing the last one leaves a process
        # ready for the next Plot command rather than a session that has
        # quietly lost the ability to plot.
        application.setQuitOnLastWindowClosed(False)
        host = Host(connection)
    except Exception as error:  # pragma: no cover - a broken install
        _send(connection, protocol.READY, protocol.Refused(_said(error)))
        return
    _send(connection, protocol.READY, protocol.Done())
    host.listen()
    application.exec()


def _said(error: Exception) -> str:
    """One line naming an exception, for a message line to carry."""
    text = str(error).strip().splitlines()
    return text[0] if text else type(error).__name__


def _take_the_tty() -> None:
    """Keep Qt's chatter off the app's screen.

    The app is a full-screen terminal program and these descriptors are its
    own: a Qt warning about a Wayland protocol printed on them would land in
    the middle of a worksheet. They go to the null device where they would
    otherwise reach a terminal, and are left alone where they would not - under
    a test runner they are a capture buffer, and a traceback in it is the
    fastest way to learn why a host would not start.
    """
    try:
        if not sys.stderr or not sys.stderr.isatty():
            return
        null = os.open(os.devnull, os.O_WRONLY)
    except (OSError, ValueError):
        return
    try:
        os.dup2(null, 1)
        os.dup2(null, 2)
    finally:
        os.close(null)


def _send(connection: Connection, number: int, message: Any) -> None:
    """One message up the pipe, in one piece. A dead pipe is not an error here."""
    try:
        connection.send((number, message))
    except (OSError, ValueError):
        pass


class Sampler:
    """The thread everything numeric happens on, taking one job at a time.

    Jobs are keyed, and a new job displaces a pending one with the same key:
    the third view change in a drag replaces the second's sampling of the same
    curve, which has not started, and the queue therefore never grows longer
    than the number of curves. A job already running is left to finish - it is
    holding numpy arrays and there is nothing to interrupt it with - and its
    answer is discarded by the window if the view has moved on.

    The answer is delivered on the Qt thread, because what happens to it is
    drawing. That is what the signal is for: emitting from this thread into an
    object that lives in the Qt thread is queued by Qt itself, and is the one
    supported way across.
    """

    def __init__(self) -> None:
        from pyqtgraph.Qt import QtCore

        class Postman(QtCore.QObject):
            answered = QtCore.Signal(object, object)

        self._postman = Postman()
        self._postman.answered.connect(
            lambda done, answer: done(answer),
            QtCore.Qt.ConnectionType.QueuedConnection,
        )
        self._lock = threading.Lock()
        self._waiting = threading.Condition(self._lock)
        self._jobs: dict[Any, tuple[Callable[..., Any], Callable[[Any], None]]] = {}
        self._order: list[Any] = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(
        self,
        key: Any,
        work: Callable[..., Any],
        done: Callable[[Any], None],
        report: Callable[..., None] | None = None,
    ) -> None:
        """Queue one job, displacing any pending job with the same key.

        `work` is called with a `report` of its own to hand partial answers to -
        the uniform pass of a sampling, drawn while the refinement runs - and
        both it and `done` are delivered on the Qt thread.
        """

        def reporting(*arguments: Any) -> None:
            if report is not None:
                self._postman.answered.emit(lambda values: report(*values), arguments)

        with self._waiting:
            if key not in self._jobs:
                self._order.append(key)
            self._jobs[key] = (lambda: work(reporting), done)
            self._waiting.notify()

    def drop(self, matches: Callable[[Any], bool]) -> None:
        """Forget the pending jobs whose keys match, a closed window's say."""
        with self._waiting:
            for key in [key for key in self._order if matches(key)]:
                self._order.remove(key)
                self._jobs.pop(key, None)

    def _run(self) -> None:
        while True:
            with self._waiting:
                while not self._order:
                    self._waiting.wait()
                key = self._order.pop(0)
                work, done = self._jobs.pop(key)
            try:
                answer: Any = work()
            except Exception as error:  # the curve's problem, not the host's
                answer = error
            self._postman.answered.emit(done, answer)


def _bridge() -> Any:
    """The object requests cross into the Qt loop on.

    A class made here rather than at module scope because it is a `QObject`,
    and this module is imported by nothing that has a Qt toolkit to make one
    with. Emitting its signal from the reader thread is what reposts a request
    into the Qt event loop.
    """
    from pyqtgraph.Qt import QtCore

    class Bridge(QtCore.QObject):
        arrived = QtCore.Signal(object, object)
        ended = QtCore.Signal()

    return Bridge()


class Host:
    """The window registry, the request handlers, and the two threads' ends."""

    def __init__(self, connection: Connection) -> None:
        self._connection = connection
        self._sending = threading.Lock()
        self.sampler = Sampler()
        self.bridge = _bridge()
        self.bridge.arrived.connect(self._handle)
        self.bridge.ended.connect(self._ended)
        self.windows: dict[int, Any] = {}
        #: The preferences the next window and the next plot are built with.
        #: The dataclass defaults until the app says otherwise, which it does in
        #: front of its first request, so a host driven by a test draws the
        #: pictures the program's own defaults describe.
        self.preferences = protocol.Prefer()
        #: Which window is the receiver for each kind, and the order windows
        #: have been touched in, most recent first - fed by the windows' own
        #: activation events, and what a closed receiver falls back through.
        self._current: dict[WindowKind, int] = {}
        self._used: list[int] = []
        self._made = 0

    # -- the pipe ----------------------------------------------------------

    def listen(self) -> None:
        threading.Thread(target=self._reading, daemon=True).start()

    def _reading(self) -> None:
        """The reader thread: decode, repost, and end when the app is gone."""
        while True:
            try:
                number, request = self._connection.recv()
            except (EOFError, OSError, ValueError):
                break
            self.bridge.arrived.emit(number, request)
        self.bridge.ended.emit()

    def _ended(self) -> None:
        """The app has gone: so do we, windows and all."""
        from pyqtgraph.Qt import QtWidgets

        for window in list(self.windows.values()):
            window.host = _Deaf()
            window.close()
        QtWidgets.QApplication.quit()

    def reply(self, number: int, message: Any) -> None:
        with self._sending:
            _send(self._connection, number, message)

    def event(self, message: Any) -> None:
        """One asynchronous event, up the pipe to whoever is listening."""
        with self._sending:
            _send(self._connection, protocol.EVENT, message)

    # -- what a window asks of the host -------------------------------------

    def sample(
        self,
        key: Any,
        work: Callable[..., Any],
        done: Callable[[Any], None],
        report: Callable[..., None] | None = None,
    ) -> None:
        self.sampler.submit(key, work, done, report)

    def trouble(self, window: int, label: str, message: str) -> None:
        self.event(protocol.Trouble(window, label, message))

    def author(self, worksheet: int, text: str) -> None:
        """A traced point a window sends home, to be authored into a worksheet."""
        self.event(protocol.Traced(worksheet, text))

    def adjusted(self, **values: Any) -> None:
        """A sticky control was left in a new position, named by its `Prefer` field.

        The value is kept, so the next plot follows the last one's look, and
        the whole of the preferences goes up the pipe: the app is the side
        that survives this process and carries the answer into a state file.
        """
        self.preferences = replace(self.preferences, **values)
        self.event(protocol.Preferred(self.preferences))

    def touched(self, number: int) -> None:
        """A window the user touched is the receiver for its kind now.

        Fed by the windows' own activation events - a click, a raise, an
        alt-tab - and by `_add`, since a plot's arrival raises the window it
        draws into and counts as a touch even where the window manager
        declines to hand that window the focus. The same bookkeeping either
        way, which is what makes it bookkeeping that cannot disagree with
        what the user sees.
        """
        window = self.windows.get(number)
        if window is None:
            return
        self._used_now(number)
        self._make_current(number)

    def closed(self, number: int) -> None:
        """A window the user closed: forget it, and hand its kind on.

        The receiver of that kind falls back to the last-activated survivor
        of its kind, so plotting after closing a window lands somewhere
        rather than opening a new window every time.
        """
        window = self.windows.pop(number, None)
        if window is None:
            return
        self.sampler.drop(lambda key: key[0] == number)
        if number in self._used:
            self._used.remove(number)
        kind = window.kind
        if self._current.get(kind) == number:
            del self._current[kind]
            heir = self._recent(kind)
            if heir is not None:
                self._make_current(heir)
        self.event(protocol.Closed(number))

    # -- requests ----------------------------------------------------------

    def _handle(self, number: int, request: Any) -> None:
        """One request, on the Qt thread, answered before anything is drawn."""
        try:
            if isinstance(request, protocol.Add):
                self.reply(number, self._add(request))
            elif isinstance(request, protocol.Describe):
                self.reply(number, self._describe())
            elif isinstance(request, protocol.Prefer):
                self.preferences = request
                self.reply(number, protocol.Done())
            elif isinstance(request, protocol.Shutdown):
                self.reply(number, protocol.Done())
                self._ended()
            else:
                self.reply(number, protocol.Refused(f"unknown request {request!r}"))
        except Exception as error:  # a bug here must not cost the windows
            self.reply(number, protocol.Refused(_said(error)))

    def _add(self, request: protocol.Add) -> Any:
        kind = protocol.dimension(request.kind)
        if request.kind not in protocol.DRAWN:
            return protocol.Refused(protocol.UNDRAWN.format(kind=request.kind.value))
        window = self._target(request.window, kind)
        if isinstance(window, protocol.Refused):
            return window
        drawing = _curves(preferred(request, self.preferences))
        # A family that has grown shorter leaves elements behind, and an
        # element of a plot that is no longer there is a curve nobody can
        # account for.
        kept = {plot.label for plot in drawing}
        for plot in list(window.plots):
            if (
                plot.worksheet == request.worksheet
                and plot.label.startswith(f"{request.label}.")
                and plot.label not in kept
            ):
                window.remove(plot)
        # Whether any of what is about to be drawn replaces a plot already
        # there, which is the word the acknowledgement message turns on.
        replaced = any(
            window.find(plot.worksheet, plot.label) is not None for plot in drawing
        )
        for plot in drawing:
            window.add(plot)
        window.show()
        window.raise_()
        window.activateWindow()
        self.touched(window.number)
        return protocol.Placed(window.number, replaced)

    def _describe(self) -> protocol.Windows:
        return protocol.Windows(
            tuple(self.windows[number].describe() for number in sorted(self.windows))
        )

    # -- the registry ------------------------------------------------------

    def _target(self, where: int | Where | None, kind: WindowKind) -> Any:
        """The window an `Add` is for, making one where that is what is asked.

        Named by nothing, the plot goes to the receiver of its kind - the
        window the user last touched - and opens one where none exists.
        """
        if isinstance(where, int):
            window = self.windows.get(where)
            if window is None:
                return protocol.Refused(f"there is no plot window {where}")
            if window.kind is not kind:
                return protocol.Refused(f"window {where} is a {window.kind} window")
            return window
        if where is Where.NEW:
            return self._open(kind)
        number = self._current.get(kind, self._recent(kind))
        if number is None:
            return self._open(kind)
        return self.windows[number]

    def _open(self, kind: WindowKind) -> Any:
        """A new window of `kind`, numbered where the last one left off.

        One sequence of numbers across both kinds, never reused, so the number
        that keys the protocol names one window for the life of the session.
        The user never reads it: a window is titled by what it holds.

        The two window classes are imported here rather than at the top of the
        file because each of them costs a toolkit: a session that only ever
        draws curves never loads OpenGL, and a machine without it can still
        plot everything else.

        This is the one construction site, and so the one place the sticky
        grid and wire look reach a window. They are read now rather than kept
        by the window, which is what makes a changed preference show in the
        next window and leave the open ones as they are. A 2D window takes no
        preference at all: it always opens with equal scales, and its `1:1`
        toggle is the one-window exception that is deliberately not
        remembered.
        """
        self._made += 1
        if kind is WindowKind.THREE_D:
            from rederive.plot.window3d import Window3D

            window: Any = Window3D(
                self._made,
                self,
                grid=self.preferences.grid,
                wire=self.preferences.wire,
            )
        else:
            from rederive.plot.window2d import Window2D

            window = Window2D(self._made, self)
        self.windows[self._made] = window
        window.retitle(False)
        return window

    def _make_current(self, number: int) -> None:
        """Make a window the receiver for its kind, and say so in the titles."""
        window = self.windows.get(number)
        if window is None:
            return
        self._current[window.kind] = number
        for other in self.windows.values():
            if other.kind is window.kind:
                other.retitle(other.number == number)

    def _used_now(self, number: int) -> None:
        if number in self._used:
            self._used.remove(number)
        self._used.insert(0, number)

    def _recent(self, kind: WindowKind) -> int | None:
        """The last-touched window of `kind`, or the newest one there is."""
        for number in self._used:
            window = self.windows.get(number)
            if window is not None and window.kind is kind:
                return number
        matching = [n for n, w in self.windows.items() if w.kind is kind]
        return max(matching) if matching else None


def preferred(request: protocol.Add, preferences: protocol.Prefer) -> protocol.Add:
    """`request` with the preferences filled in wherever it has no opinion.

    How a data plot draws its points is a property of the plot and not of the
    window it lands in, so the preference is read when the plot is added rather
    than when the window was built: a new data plot follows the preference in
    force now, and one drawn a minute ago keeps whatever its right-click menu
    was told. The grid goes the other way, and is read where a window is made.

    A pure function, so that what a preference does to a request can be asked
    without a toolkit to ask it in.
    """
    options = request.options
    if options.connected is not None and options.point_size is not None:
        return request
    return replace(
        request,
        options=replace(
            options,
            connected=(
                preferences.connected if options.connected is None else options.connected
            ),
            point_size=(
                preferences.point_size
                if options.point_size is None
                else options.point_size
            ),
        ),
    )


def _curves(request: protocol.Add) -> list[Any]:
    """The plots one `Add` becomes: one, or one per element of a family.

    A family is one expression to the user - `#3` is the vector - and several
    curves or surfaces in the window, so its elements are labelled `#3.1`,
    `#3.2` and take the next colors of the palette. Re-plotting `#3` therefore
    replaces its own elements one by one, each keeping the color it had.

    The elements are named with the texts the app wrote, since the host renders
    no expression itself; a family that arrives without them falls back on the
    label, which is at least true.
    """
    if protocol.dimension(request.kind) is WindowKind.THREE_D:
        from rederive.plot.window3d import Surface as Made

        several, single = PlotKind.SURFACES, PlotKind.SURFACE
    else:
        from rederive.plot.window2d import Plot as Made  # type: ignore[assignment]

        several, single = PlotKind.FAMILY, PlotKind.CURVE

    def one(label: str, text: str, node: Any, kind: PlotKind) -> Any:
        return Made(
            worksheet=request.worksheet,
            label=label,
            text=text,
            kind=kind,
            node=node,
            context=request.context,
            options=request.options,
            state=request.state,
        )

    if request.kind is not several:
        return [one(request.label, request.text, request.node, request.kind)]
    texts = request.options.texts
    return [
        one(
            f"{request.label}.{index + 1}",
            texts[index] if index < len(texts) else request.label,
            element,
            single,
        )
        for index, element in enumerate(request.node.children)
    ]


class _Deaf:
    """A host that hears nothing, for windows being closed at shutdown.

    A window tells the host it has closed, and at shutdown every window closes
    at once; without this the registry would be edited from inside its own
    iteration and an event would go up a pipe the app has already let go of.
    """

    def __getattr__(self, name: str) -> Callable[..., None]:
        return lambda *arguments, **keywords: None
