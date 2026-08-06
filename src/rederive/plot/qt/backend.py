"""The backend the plot session opens Qt windows on, and the thread it samples on.

Two objects, and both are the toolkit's side of a line drawn above them. The
backend makes windows and takes them away again; the executor is the sampling
thread and the one supported way of getting an answer from it back into a Qt
event loop.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from rederive.plot.protocol import Prefer, WindowKind
from rederive.plot.session import Sampler

__all__ = ["QtBackend", "ThreadExecutor"]


class ThreadExecutor:
    """The thread everything numeric happens on, taking one job at a time.

    The answer is delivered on the Qt thread, because what happens to it is
    drawing. That is what the signal is for: emitting from this thread into an
    object that lives in the Qt thread is queued by Qt itself, and is the one
    supported way across.

    There is no abort path and none is needed: sampling is depth- and
    size-capped, so the worst a pathological expression can do is take a while
    over its own curve.
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
        self._waiting = threading.Condition()
        self._sampler: Sampler | None = None

    def serving(self, sampler: Sampler) -> None:
        self._sampler = sampler
        threading.Thread(target=self._run, daemon=True).start()

    def wake(self) -> None:
        with self._waiting:
            self._waiting.notify()

    def deliver(self, done: Callable[[Any], None], answer: Any) -> None:
        self._postman.answered.emit(done, answer)

    def _run(self) -> None:
        assert self._sampler is not None
        while True:
            with self._waiting:
                while (job := self._sampler.take()) is None:
                    self._waiting.wait()
            work, done = job
            self.deliver(done, work())


class QtBackend:
    """Plot windows as Qt windows, made where the session asks for one.

    The two window classes are imported where a window is made rather than at
    the top of the file, because each of them costs a toolkit: a session that
    only ever draws curves never loads OpenGL, and a machine without it can
    still plot everything else.
    """

    def __init__(self) -> None:
        #: The windows now open, so that a shutdown can end them all. A window
        #: the user closed reports itself closed and is dropped here.
        self.windows: dict[int, Any] = {}

    def open(
        self, session: Any, kind: WindowKind, number: int, preferences: Prefer
    ) -> Any:
        """One window, built with the sticky preferences in force now.

        A 2D window takes no preference at all: it always opens with equal
        scales, and its `1:1` toggle is the one-window exception that is
        deliberately not remembered.
        """
        if kind is WindowKind.THREE_D:
            from rederive.plot.qt.window3d import Window3D

            window: Any = Window3D(
                number, session, grid=preferences.grid, wire=preferences.wire
            )
        else:
            from rederive.plot.qt.window2d import Window2D

            window = Window2D(number, session)
        self.windows[number] = window
        return window

    def close(self, number: int) -> None:
        """Forget a window that has closed itself. Its own event said so."""
        self.windows.pop(number, None)

    def stop(self) -> None:
        """End every window, and the Qt loop they were running in.

        The windows are made deaf first. A window tells the session it has
        closed, and here every window closes at once; without this the registry
        would be edited from inside its own iteration and an event would go up a
        channel the app has already let go of.
        """
        from pyqtgraph.Qt import QtWidgets

        for window in list(self.windows.values()):
            window.session = _Deaf()
            window.close()
        self.windows.clear()
        QtWidgets.QApplication.quit()


class _Deaf:
    """A session that hears nothing, for windows being closed at shutdown."""

    def __getattr__(self, name: str) -> Callable[..., None]:
        return lambda *arguments, **keywords: None
