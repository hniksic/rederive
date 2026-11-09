"""The Qt backend for the plot session, and its sampling thread.

The backend opens and closes windows. The executor runs sampling on a separate thread
and delivers results back to the Qt event loop.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from rederive.plot.protocol import Prefer, WindowKind
from rederive.plot.session import Sampler

__all__ = ["QtBackend", "ThreadExecutor"]


class ThreadExecutor:
    """The thread that runs numeric work, one job at a time.

    Results are delivered on the Qt thread, where they are drawn, via a signal emitted
    to an object living in the Qt thread, which Qt queues.

    Jobs can't be aborted, but sampling is depth- and size-capped, so a pathological
    expression can only be slow.
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
    """Opens plot windows as Qt windows.

    The window classes are imported on first use, so a session that only draws curves
    never loads OpenGL, and a machine without OpenGL can still make 2D plots.
    """

    def __init__(self) -> None:
        #: The open windows, so shutdown can close them all. Windows closed by the user
        #: are removed through `close`.
        self.windows: dict[int, Any] = {}

    def open(
        self, session: Any, kind: WindowKind, number: int, preferences: Prefer
    ) -> Any:
        """Open a window with the current sticky preferences.

        2D windows take no preferences: they always open with equal scales, and their
        `1:1` toggle is deliberately not remembered.
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
        """Forget a window that has reported itself closed."""
        self.windows.pop(number, None)

    def stop(self) -> None:
        """Close every window and quit the Qt loop.

        The windows' sessions are replaced with a no-op first. Otherwise each closing
        window would report to the session, modifying the registry during iteration and
        sending events over a channel the app has already dropped.
        """
        from pyqtgraph.Qt import QtWidgets

        for window in list(self.windows.values()):
            window.session = _Deaf()
            window.close()
        self.windows.clear()
        QtWidgets.QApplication.quit()


class _Deaf:
    """A session that ignores everything, for windows closed at shutdown."""

    def __getattr__(self, name: str) -> Callable[..., None]:
        return lambda *arguments, **keywords: None
