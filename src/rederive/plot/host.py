"""The child process the Qt backend is served in: Qt on the main thread, math off it.

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
the conversion to sympy, the lambdify, and every sample of every curve. It is
the plot session's executor and takes the keyed jobs the session hands it, so a
re-sample of the same curve for a newer view replaces the one that has not
started yet, which is what makes a dragged view cost one sampling and not sixty.

What the windows are, which of them the next plot lands in and what a
preference does to it belong to the plot session rather than to this file. What
is here is the process, the pipe, and the toolkit they run in.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
from multiprocessing.connection import Connection
from typing import Any

from rederive.plot import protocol
from rederive.plot.qt import theme
from rederive.plot.qt.backend import QtBackend, ThreadExecutor
from rederive.plot.session import PlotSession, said

__all__ = ["Host", "serve"]


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
        # Every window this process opens is a dark picture, and the chrome
        # around it is dressed to match here rather than window by window: one
        # sheet on the application reaches the toolbars, the fields, the menus
        # the toolkit puts up for us and the export dialogs we never built.
        theme.dress(application)
        # The host outlives its windows: closing the last one leaves a process
        # ready for the next Plot command rather than a session that has
        # quietly lost the ability to plot.
        application.setQuitOnLastWindowClosed(False)
        host = Host(connection)
    except Exception as error:  # pragma: no cover - a broken install
        _send(connection, protocol.READY, protocol.Refused(said(error)))
        return
    _send(connection, protocol.READY, protocol.Done())
    host.listen()
    application.exec()


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
    """The two ends of the pipe, and the plot session everything on it is about."""

    def __init__(self, connection: Connection) -> None:
        self._connection = connection
        self._sending = threading.Lock()
        self.session = PlotSession(QtBackend(), ThreadExecutor(), self.event)
        self.bridge = _bridge()
        self.bridge.arrived.connect(self._handle)
        # The app has gone: so do we, windows and all.
        self.bridge.ended.connect(self.session.shutdown)

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

    def _handle(self, number: int, request: Any) -> None:
        """One request, on the Qt thread, answered before anything is drawn."""
        self.session.handle(request, lambda message: self.reply(number, message))

    def reply(self, number: int, message: Any) -> None:
        with self._sending:
            _send(self._connection, number, message)

    def event(self, message: Any) -> None:
        """One asynchronous event, up the pipe to whoever is listening."""
        with self._sending:
            _send(self._connection, protocol.EVENT, message)
