"""The window bookkeeping, the routing, and the queue everything numeric goes through.

This is the half of plotting that is the same whatever draws the picture, and
it is most of what the plot host used to be: window numbering, which window of
each kind is the receiver, what an activation means, how an `Add` is routed,
how `Describe` is answered, and how the sticky preferences reach the next
window and the next plot. It talks to a `Backend`, and it knows nothing else
about what a window is.

One pointer, and it is the window the user last touched: the windows report
their own activation events - a click, a raise, an alt-tab - and the
last-activated window of each kind is the *receiver* the next plot of that kind
lands in. A plot's own arrival counts as a touch, since a plot is brought to the
front of the window it draws into, and when the receiver closes it falls back to
the last-activated survivor of its kind. The receiver says so in its own title,
so the user can see where the next plot will land. The app never tracks any of
this; it asks.

The sticky preferences travel both ways. The app says what the next surface's
grid and the next data plot's points are to be; they are read where a window is
built and where a plot is added, which is what keeps a change out of the windows
already open. And when a sticky control moves in a window, the new value is kept
and reported to whoever is listening, because the app is the side that outlives
this and writes state files.

The `Sampler` is the third piece. Jobs are keyed, and a new job displaces a
pending one with the same key, which is what makes a dragged view cost one
sampling and not sixty. What actually runs them is an `Executor` handed in: a
sampling thread on the desktop, requests to the engine worker in the browser.
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
    """One line naming an exception, for a message line to carry."""
    text = str(error).strip().splitlines()
    return text[0] if text else type(error).__name__


class Executor(Protocol):
    """What runs the sampler's jobs, one at a time, and delivers their answers.

    The desktop's is a thread that pulls jobs and hands the answers back into
    the Qt loop; the browser's asks the engine worker and resolves what comes
    back. Neither is the sampler's business - what is, is which job runs next.
    """

    def serving(self, sampler: Sampler) -> None:
        """Take the sampler whose jobs are to be run, and start running them."""

    def wake(self) -> None:
        """There is a job waiting. Called wherever `submit` is called."""

    def deliver(self, done: Callable[[Any], None], answer: Any) -> None:
        """Hand one answer to the side that draws, wherever that side runs."""


class Sampler:
    """The keyed job queue everything numeric goes through, latest per key winning.

    A new job displaces a pending one with the same key: the third view change
    in a drag replaces the second's sampling of the same curve, which has not
    started, and the queue therefore never grows longer than the number of
    curves. A job already running is left to finish - it is holding arrays and
    there is nothing to interrupt it with - and its answer is discarded by the
    window if the view has moved on.

    A job that raises answers with the exception rather than losing it: what
    went wrong is the curve's problem and the window is the one that says so.
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
        """Queue one job, displacing any pending job with the same key.

        `work` is called with a `report` of its own to hand partial answers to -
        the uniform pass of a sampling, drawn while the refinement runs - and
        both it and `done` are delivered where the drawing happens.
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
        """Forget the pending jobs whose keys match, a closed window's say."""
        with self._lock:
            for key in [key for key in self._order if matches(key)]:
                self._order.remove(key)
                self._jobs.pop(key, None)

    def take(self) -> tuple[Callable[[], Any], Callable[[Any], None]] | None:
        """The next job and what to do with its answer, or None for none.

        The executor's call, and the only way a job leaves the queue: whatever
        is taken here is running, and a `submit` for the same key while it runs
        queues a new job rather than replacing it.
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
    """The window registry, the routing, and the request handlers.

    Held by the process the windows live in on the desktop, and by the app
    itself in the browser, where there is no such process: what it is is the
    same either way, which is what makes a second backend a backend rather
    than a second program.
    """

    def __init__(
        self,
        backend: Backend,
        executor: Executor,
        events: Callable[[Any], None] | None = None,
    ) -> None:
        self.backend = backend
        self.sampler = Sampler(executor)
        #: Where asynchronous events go - a window closed, a curve that would
        #: not evaluate, a preference the user moved. Assigned rather than
        #: required, since the app is the side that decides what to do with one.
        self.events = events
        #: The preferences the next window and the next plot are built with.
        #: The dataclass defaults until the app says otherwise, which it does in
        #: front of its first request, so a session driven by a test draws the
        #: pictures the program's own defaults describe.
        self.preferences = protocol.Prefer()
        self.windows: dict[int, WindowHandle] = {}
        #: Which window is the receiver for each kind, and the order windows
        #: have been touched in, most recent first - fed by the windows' own
        #: activation events, and what a closed receiver falls back through.
        self._current: dict[WindowKind, int] = {}
        self._used: list[int] = []
        self._made = 0

    # -- requests ----------------------------------------------------------

    def handle(self, request: Any, reply: Callable[[Any], None]) -> None:
        """One request, answered before anything is drawn.

        The acknowledgement goes out first and the drawing happens afterwards,
        which is the protocol's whole shape; a shutdown is the one request that
        does anything after its own reply, and it has to be that way round -
        the answer would otherwise travel a channel that has just been closed.
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
                self.release()
                reply(protocol.Done())
            elif isinstance(request, protocol.Shutdown):
                reply(protocol.Done())
                self.shutdown()
            else:
                reply(protocol.Refused(f"unknown request {request!r}"))
        except Exception as error:  # a bug here must not cost the windows
            reply(protocol.Refused(said(error)))

    def add(self, request: protocol.Add) -> Any:
        """Draw one expression, and say which window took it."""
        kind = protocol.dimension(request.kind)
        if request.kind not in protocol.DRAWN:
            return protocol.Refused(protocol.UNDRAWN.format(kind=request.kind.value))
        window = self._target(request.window, kind)
        if isinstance(window, protocol.Refused):
            return window
        drawing = curves(preferred(request, self.preferences))
        # A family that has grown shorter leaves elements behind, and an
        # element of a plot that is no longer there is a curve nobody can
        # account for. A request that asked to be alone leaves nothing at all.
        kept = {plot.label for plot in drawing}
        for plot in list(window.plots):
            if request.demonstrating or (
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
        window.present(request.demonstrating)
        self.touched(window.number)
        return protocol.Placed(window.number, replaced)

    def describe(self) -> protocol.Windows:
        """What windows are open and what is in them."""
        return protocol.Windows(
            tuple(self.windows[number].describe() for number in sorted(self.windows))
        )

    def prefer(self, preferences: protocol.Prefer) -> None:
        """Remember what the next window and the next plot are to be built with."""
        self.preferences = preferences

    def release(self) -> None:
        """The demonstration is over: every window is an ordinary window again.

        All of them rather than the one that was drawn into. A gallery of
        surfaces and curves fills two windows, both of them were kept in front
        as their steps landed, and what ended ended for both.
        """
        for window in self.windows.values():
            window.release()

    def shutdown(self) -> None:
        """End the windows and whatever runs them: the app has gone."""
        self.backend.stop()

    # -- what a window asks of the session ----------------------------------

    def sample(
        self,
        key: Any,
        work: Callable[..., Any],
        done: Callable[[Any], None],
        report: Callable[..., None] | None = None,
    ) -> None:
        self.sampler.submit(key, work, done, report)

    def event(self, message: Any) -> None:
        """One asynchronous event, to whoever is listening for them."""
        if self.events is not None:
            self.events(message)

    def trouble(self, window: int, label: str, message: str) -> None:
        self.event(protocol.Trouble(window, label, message))

    def author(self, worksheet: int, text: str) -> None:
        """A traced point a window sends home, to be authored into a worksheet."""
        self.event(protocol.Traced(worksheet, text))

    def stepped(self, stopping: bool) -> None:
        """A key pressed in a window a demonstration's step is waiting in.

        Nothing is done with it here: what a key means to a demonstration is
        the app's, this side holding no demonstration at all. All this window
        knew was that its picture was a step and that a key arrived in it.
        """
        self.event(protocol.Stepped(stopping))

    def adjusted(self, **values: Any) -> None:
        """A sticky control was left in a new position, named by its `Prefer` field.

        The value is kept, so the next plot follows the last one's look, and
        the whole of the preferences is reported: the app is the side that
        survives this and carries the answer into a state file.
        """
        self.preferences = replace(self.preferences, **values)
        self.event(protocol.Preferred(self.preferences))

    def touched(self, number: int) -> None:
        """A window the user touched is the receiver for its kind now.

        Fed by the windows' own activation events - a click, a raise, an
        alt-tab - and by `add`, since a plot's arrival raises the window it
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

    def _open(self, kind: WindowKind) -> WindowHandle:
        """A new window of `kind`, numbered where the last one left off.

        One sequence of numbers across both kinds, never reused, so the number
        that keys the protocol names one window for the life of the session.
        The user never reads it: a window is titled by what it holds.

        This is the one construction site, and so the one place the sticky
        preferences reach a window. They are read now rather than kept by the
        window, which is what makes a changed preference show in the next
        window and leave the open ones as they are.
        """
        self._made += 1
        window = self.backend.open(self, kind, self._made, self.preferences)
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


def curves(request: protocol.Add) -> list[Plot]:
    """The plots one `Add` becomes: one, or one per element of a family.

    A family is one expression to the user - `#3` is the vector - and several
    curves or surfaces in the window, so its elements are labelled `#3.1`,
    `#3.2` and take the next colors of the palette. Re-plotting `#3` therefore
    replaces its own elements one by one, each keeping the color it had.

    The elements are named with the texts the app wrote, since nothing here
    renders an expression itself; a family that arrives without them falls back
    on the label, which is at least true.
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
            # How a data plot draws its points is the plot's own and is settled
            # here, where the preferences have already been read into the
            # request: what a backend is handed knows how it is to look.
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
