"""The child process hosting the Qt backend: Qt on the main thread, math elsewhere.

The app process can't host Qt: it is sympy-free by design and already runs Textual's
event loop. So plotting runs in a separate process, spawned on the first plot and kept
for the rest of the session. It is the only process that owns windows.

It has three threads:

The **Qt thread** (the main thread) only paints and handles input. Requests reach it
already decoded, and expensive work is done elsewhere, so windows stay responsive
however slow an expression is.

The **reader thread** receives from the pipe, decodes requests and reposts them into
the Qt loop as a queued signal. When the pipe closes, the app is gone and the host
exits.

The **sampling thread** does the math: substitution, conversion to sympy, lambdify and
all sampling. It is the plot session's executor and runs keyed jobs, so a resample of a
curve for a newer view replaces a pending one, and a drag costs one sampling, not
sixty.

Window management and preferences belong to the plot session; this module has only the
process, the pipe and the toolkit setup.
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
    """Run the plot host until the app goes away or asks it to stop.

    The handshake is sent once the toolkit is up and a window could be opened, so the
    proxy knows later requests won't pay startup costs. A failure before that point
    (typically missing wheels or a display that won't open) is sent back as a message
    for the message line, rather than leaving a dead process.
    """
    _take_the_tty()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets

        pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")
        # pyqtgraph caches compiled shader programs by name rather than per OpenGL
        # context, so without context sharing a second 3D window would use the first
        # one's program handles and fail to draw. Must be set before the application
        # is created.
        QtCore.QCoreApplication.setAttribute(
            QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True
        )
        application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        # Style the chrome dark to match the plots. Applying it to the application
        # covers toolbars, fields, menus and Qt's own export dialogs.
        theme.dress(application)
        # Keep the host running after the last window closes, ready for the next plot.
        application.setQuitOnLastWindowClosed(False)
        host = Host(connection)
    except Exception as error:  # pragma: no cover - a broken install
        _send(connection, protocol.READY, protocol.Refused(said(error)))
        return
    _send(connection, protocol.READY, protocol.Done())
    host.listen()
    application.exec()


def _take_the_tty() -> None:
    """Keep Qt's warnings off the app's screen.

    The inherited stdout and stderr belong to the full-screen terminal app, where a Qt
    warning would land in the middle of a worksheet. They are redirected to the null
    device only if they are a terminal; under a test runner they are a capture buffer,
    where a traceback helps explain why the host failed to start.
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
    """Send a message up the pipe, ignoring a dead pipe."""
    try:
        connection.send((number, message))
    except (OSError, ValueError):
        pass


def _bridge() -> Any:
    """The object whose signals repost requests from the reader thread into the Qt loop.

    The `QObject` class is defined here rather than at module scope so that importing
    this module doesn't require Qt.
    """
    from pyqtgraph.Qt import QtCore

    class Bridge(QtCore.QObject):
        arrived = QtCore.Signal(object, object)
        ended = QtCore.Signal()

    return Bridge()


class Host:
    """The pipe to the app, and the plot session serving it."""

    def __init__(self, connection: Connection) -> None:
        self._connection = connection
        self._sending = threading.Lock()
        self.session = PlotSession(QtBackend(), ThreadExecutor(), self.event)
        self.bridge = _bridge()
        self.bridge.arrived.connect(self._handle)
        # When the app is gone, shut down along with all windows.
        self.bridge.ended.connect(self.session.shutdown)

    def listen(self) -> None:
        threading.Thread(target=self._reading, daemon=True).start()

    def _reading(self) -> None:
        """Reader thread: receive and repost requests until the app is gone."""
        while True:
            try:
                number, request = self._connection.recv()
            except (EOFError, OSError, ValueError):
                break
            self.bridge.arrived.emit(number, request)
        self.bridge.ended.emit()

    def _handle(self, number: int, request: Any) -> None:
        """Handle a request on the Qt thread, replying before anything is drawn."""
        self.session.handle(request, lambda message: self.reply(number, message))

    def reply(self, number: int, message: Any) -> None:
        with self._sending:
            _send(self._connection, number, message)

    def event(self, message: Any) -> None:
        """Send an asynchronous event up the pipe."""
        with self._sending:
            _send(self._connection, protocol.EVENT, message)
