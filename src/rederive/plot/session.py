"""Backend-independent plot session: window bookkeeping, routing, and the sample queue.

The session numbers windows, tracks the receiver window of each kind, routes `Add`
requests, answers `Describe`, and applies sticky preferences to new windows and plots.
It knows windows only through the `Backend` protocol.

Windows report their activation events (click, raise, alt-tab), and the last-activated
window of each kind is the *receiver* for the next plot of that kind. A plot arriving
in a window also counts as activation, since the window is brought to the front. When
the receiver closes, the most recently activated remaining window of its kind takes
over. The receiver is marked in its title. The app doesn't track any of this; it asks.

Preferences travel both ways. The app sends the defaults for the next surface's grid
and data plot style; they are applied when a window is built or a plot is added, so open
windows are unaffected. When a sticky control changes in a window, the new value is
stored and reported to the app, which outlives the session and saves state files.

The `Sampler` is a keyed job queue: a new job replaces a pending one with the same key,
so a drag costs one sampling, not sixty. Jobs are run by an `Executor`: a sampling
thread on the desktop, the engine worker in the browser.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Protocol

from rederive.plot import protocol
from rederive.plot.backend import Backend, WindowHandle
from rederive.plot.model import POINT_SIZE, Plot, Surface
from rederive.plot.protocol import PlotKind, Where, WindowKind

__all__ = ["Executor", "PlotSession", "Sampler", "curves", "preferred", "said"]


def said(error: Exception) -> str:
    """A one-line description of an exception, for the message line."""
    text = str(error).strip().splitlines()
    return text[0] if text else type(error).__name__


class Executor(Protocol):
    """Runs the sampler's jobs one at a time and delivers their results.

    On the desktop it is a thread that pulls jobs and passes results back to the Qt
    loop; in the browser it sends jobs to the engine worker.
    """

    def serving(self, sampler: Sampler) -> None:
        """Start running jobs from `sampler`."""

    def wake(self) -> None:
        """Notify that a job is waiting. Called on every `submit`."""

    def deliver(self, done: Callable[[Any], None], answer: Any) -> None:
        """Pass a result to the drawing side, on whatever thread it runs."""


class Sampler:
    """Keyed queue for all numeric jobs, where the latest job for each key wins.

    A new job replaces a pending one with the same key, so the queue is never longer
    than the number of curves. A job already running can't be interrupted and runs to
    completion; the window discards its result if the view has changed.

    A job that raises returns the exception as its result, so the window can report it.
    """

    def __init__(self, executor: Executor) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[Any, tuple[Callable[[], Any], Callable[[Any], None]]] = {}
        self._order: list[Any] = []
        self._executor = executor
        executor.serving(self)

    def submit(
        self,
        key: Any,
        work: Callable[..., Any],
        done: Callable[[Any], None],
        report: Callable[..., None] | None = None,
    ) -> None:
        """Queue a job, replacing any pending job with the same key.

        `work` is called with a `report` callback for partial results (such as the
        uniform pass of a sampling, drawn while refinement runs). Both reports and the
        final result are delivered to the drawing side.
        """

        def reporting(*arguments: Any) -> None:
            if report is not None:
                self._executor.deliver(lambda values: report(*values), arguments)

        with self._lock:
            if key not in self._jobs:
                self._order.append(key)
            self._jobs[key] = (lambda: _answered(work, reporting), done)
        self._executor.wake()

    def drop(self, matches: Callable[[Any], bool]) -> None:
        """Drop pending jobs whose keys match, e.g. those of a closed window."""
        with self._lock:
            for key in [key for key in self._order if matches(key)]:
                self._order.remove(key)
                self._jobs.pop(key, None)

    def take(self) -> tuple[Callable[[], Any], Callable[[Any], None]] | None:
        """The next job and its completion callback, or None if the queue is empty.

        Called by the executor. A taken job is considered running, so submitting the
        same key afterwards queues a new job.
        """
        with self._lock:
            if not self._order:
                return None
            return self._jobs.pop(self._order.pop(0))


def _answered(work: Callable[..., Any], reporting: Callable[..., None]) -> Any:
    try:
        return work(reporting)
    except Exception as error:  # the curve's problem, not the session's
        return error


class PlotSession:
    """The window registry, plot routing, and request handlers.

    On the desktop it lives in the plot host process; in the browser, in the app
    itself. It is the same in both cases.
    """

    def __init__(
        self,
        backend: Backend,
        executor: Executor,
        events: Callable[[Any], None] | None = None,
    ) -> None:
        self.backend = backend
        self.sampler = Sampler(executor)
        #: Callback for asynchronous events: a closed window, a curve that failed to
        #: evaluate, a changed preference.
        self.events = events
        #: Preferences for the next window and plot. They start as the defaults; the
        #: app sends its own before its first request.
        self.preferences = protocol.Prefer()
        self.windows: dict[int, WindowHandle] = {}
        #: The receiver window of each kind, and windows in order of activation, most
        #: recent first, used to pick a new receiver when one closes.
        self._current: dict[WindowKind, int] = {}
        self._used: list[int] = []
        self._made = 0

    # -- requests ----------------------------------------------------------

    def handle(self, request: Any, reply: Callable[[Any], None]) -> None:
        """Handle a request, replying before anything is drawn.

        Shutdown also replies first, since afterwards the channel is closed.
        """
        try:
            if isinstance(request, protocol.Add):
                reply(self.add(request))
            elif isinstance(request, protocol.Describe):
                reply(self.describe())
            elif isinstance(request, protocol.Prefer):
                self.prefer(request)
                reply(protocol.Done())
            elif isinstance(request, protocol.Release):
                self.release(request.finished)
                reply(protocol.Done())
            elif isinstance(request, protocol.Shutdown):
                reply(protocol.Done())
                self.shutdown()
            else:
                reply(protocol.Refused(f"unknown request {request!r}"))
        except Exception as error:  # a bug here must not cost the windows
            reply(protocol.Refused(said(error)))

    def add(self, request: protocol.Add) -> Any:
        """Draw an expression and return the window that received it."""
        kind = protocol.dimension(request.kind)
        if request.kind not in protocol.DRAWN:
            return protocol.Refused(protocol.UNDRAWN.format(kind=request.kind.value))
        window = self._target(request.window, kind)
        if isinstance(window, protocol.Refused):
            return window
        drawing = curves(preferred(request, self.preferences))
        # Remove leftover elements of a family that has grown shorter. A
        # demonstration clears the window entirely.
        kept = {plot.label for plot in drawing}
        for plot in list(window.plots):
            if request.demonstrating or (
                plot.worksheet == request.worksheet
                and plot.label.startswith(f"{request.label}.")
                and plot.label not in kept
            ):
                window.remove(plot)
        # Whether this replaces an existing plot, for the acknowledgement message.
        replaced = any(
            window.find(plot.worksheet, plot.label) is not None for plot in drawing
        )
        for plot in drawing:
            window.add(plot)
        window.present(request.demonstrating)
        self.touched(window.number)
        return protocol.Placed(window.number, replaced)

    def describe(self) -> protocol.Windows:
        """The open windows and their contents."""
        return protocol.Windows(
            tuple(self.windows[number].describe() for number in sorted(self.windows))
        )

    def prefer(self, preferences: protocol.Prefer) -> None:
        """Store the preferences for the next window and plot."""
        self.preferences = preferences

    def release(self, finished: bool = False) -> None:
        """End a demonstration, making its windows ordinary again.

        If it `finished`, its windows close instead. Every window is notified and
        decides for itself, since only it knows whether it showed a demonstration step;
        windows the user opened must not be closed.

        Iterates over a copy, since closing windows remove themselves from the registry.
        """
        for window in list(self.windows.values()):
            window.release(finished)

    def shutdown(self) -> None:
        """Close the windows and stop the backend, as the app has gone."""
        self.backend.stop()

    # -- called by windows ---------------------------------------------------

    def sample(
        self,
        key: Any,
        work: Callable[..., Any],
        done: Callable[[Any], None],
        report: Callable[..., None] | None = None,
    ) -> None:
        self.sampler.submit(key, work, done, report)

    def event(self, message: Any) -> None:
        """Send an asynchronous event to the listener, if any."""
        if self.events is not None:
            self.events(message)

    def trouble(self, window: int, label: str, message: str) -> None:
        self.event(protocol.Trouble(window, label, message))

    def author(self, worksheet: int, text: str) -> None:
        """Send a traced point to be added to a worksheet."""
        self.event(protocol.Traced(worksheet, text))

    def stepped(self) -> None:
        """Forward a key pressed in a window showing a demonstration step.

        The window doesn't use the key, and the demonstration is run by the app, so it
        is simply passed on.
        """
        self.event(protocol.Stepped())

    def adjusted(self, **values: Any) -> None:
        """Record a changed sticky control, given by its `Prefer` field name.

        The value is stored so the next plot follows it, and the full preferences are
        reported to the app, which saves them in the state file.
        """
        self.preferences = replace(self.preferences, **values)
        self.event(protocol.Preferred(self.preferences))

    def touched(self, number: int) -> None:
        """Make an activated window the receiver for its kind.

        Called on the windows' activation events (click, raise, alt-tab) and by `add`,
        since a plot arriving raises its window even if the window manager doesn't give
        it focus.
        """
        window = self.windows.get(number)
        if window is None:
            return
        self._used_now(number)
        self._make_current(number)

    def closed(self, number: int) -> None:
        """Forget a window the user closed, and pick a new receiver if needed.

        The most recently activated remaining window of the same kind becomes the
        receiver, so plotting after closing a window doesn't always open a new one.
        """
        window = self.windows.pop(number, None)
        if window is None:
            return
        self.backend.close(number)
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

    # -- the registry ------------------------------------------------------

    def _target(self, where: int | Where | None, kind: WindowKind) -> Any:
        """The target window of an `Add`, opening one if needed.

        If no window is named, the plot goes to the receiver of its kind, or a new
        window if there is none.
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

    def _open(self, kind: WindowKind) -> WindowHandle:
        """Open a new window of `kind` with the next number.

        Numbers are shared by both kinds and never reused, so a number identifies one
        window for the session. Users never see it; windows are titled by contents.

        Sticky preferences are applied here and only here, so changes affect the next
        window but not open ones.
        """
        self._made += 1
        window = self.backend.open(self, kind, self._made, self.preferences)
        self.windows[self._made] = window
        window.retitle(False)
        return window

    def _make_current(self, number: int) -> None:
        """Make a window the receiver for its kind, and update titles to show it."""
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
        """The most recently activated window of `kind`, or else the newest one."""
        for number in self._used:
            window = self.windows.get(number)
            if window is not None and window.kind is kind:
                return number
        matching = [n for n, w in self.windows.items() if w.kind is kind]
        return max(matching) if matching else None


def preferred(request: protocol.Add, preferences: protocol.Prefer) -> protocol.Add:
    """`request` with unset data-plot options filled in from the preferences.

    Data point style belongs to the plot, not the window, so it is applied when the
    plot is added; existing plots keep their style. (The grid preference, by contrast,
    is applied when a window is created.)

    A pure function, so it can be tested without a toolkit.
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


def curves(request: protocol.Add) -> list[Plot]:
    """The plots an `Add` produces: one, or one per element of a family.

    A family such as the vector `#3` becomes several curves or surfaces labeled `#3.1`,
    `#3.2`, and so on, each taking the next palette color. Re-plotting `#3` replaces the
    elements one by one, keeping their colors.

    Elements are named with the texts rendered by the app, falling back to the label if
    there are none.
    """
    if protocol.dimension(request.kind) is WindowKind.THREE_D:
        made: type[Plot] = Surface
        several, single = PlotKind.SURFACES, PlotKind.SURFACE
    else:
        made, several, single = Plot, PlotKind.FAMILY, PlotKind.CURVE

    def one(label: str, text: str, node: Any, kind: PlotKind) -> Plot:
        return made(
            worksheet=request.worksheet,
            label=label,
            text=text,
            kind=kind,
            node=node,
            context=request.context,
            options=request.options,
            state=request.state,
            # Preferences have already been filled into the request, so the data
            # point style is final here.
            connected=bool(request.options.connected),
            point_size=request.options.point_size or POINT_SIZE,
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
