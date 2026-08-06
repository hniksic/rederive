"""Plot windows as panes in a page, and the worker that samples what they hold.

The `Backend` the plot session opens windows on in a browser. It runs on the
main thread, beside the app, and what it owns is exactly what has no numbers in
it: the plot list, the colors the palette hands out, which pane a plot lands in,
what `Describe` answers, and the routing of everything a pane has to say back.
The picture itself - the view, the gestures, the drawing - is JavaScript's, in
`web/plot2d.js`; the samples are the engine worker's. Nothing here imports
numpy, and nothing here ever holds an array.

That is a three-way split and the seam between each pair is narrow. A pane is
made by calling into the page and is spoken to by calling into the page; the
page speaks back through a handful of proxied functions, in words and numbers
and never in arrays. A sampling is a request posted to the worker, and its
answer goes from the worker to the page directly - the page tells this side only
the request number and whatever went wrong, which is what a `Trouble` is made
of. So the numbers cross once, from where they are computed to where they are
drawn, and the main thread routes without ever touching one.

The sampler is the session's own, unchanged: jobs are keyed, a new job displaces
a pending one with the same key, and the executor below runs one at a time. What
makes one at a time the right number here is that the worker is one thread of
one interpreter - a second request in flight would queue inside it rather than
beside it - and what makes the queue worth having is that a drag would otherwise
post sixty samplings of a curve nobody has looked at yet.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from rederive.engine.context import Angle
from rederive.plot import protocol, resample, view
from rederive.plot.model import FUNCTIONS, PALETTE, PAPER_PALETTE, Plot
from rederive.plot.protocol import PlotKind, Prefer, WindowKind
from rederive.plot.session import said
from rederive.plot.web import protocol as asking

__all__ = ["Pane", "Shown", "UNDRAWN_3D", "WebBackend", "WorkerExecutor"]

#: What the browser says when asked for a solid. Stage 6 is where three.js
#: makes one; until then the refusal names what is missing rather than what to
#: do about it, there being nothing a reader could do.
UNDRAWN_3D = "3D plots are not in the browser yet - the desktop program draws them"

#: What a pane calls its own sampling jobs, beside the one per plot. Keyed like
#: every other job, so a marker dragged along a curve costs one reading and not
#: one per pixel it passed over.
TRACE = "trace"
FEATURES = "features"


class Shown(Plot):
    """A plot as a pane holds it: a name both sides know it by, and no numbers.

    The browser's answer to the Qt window's `Drawn`, and it is shorter for the
    reason the whole design is: what a plot's samples are and what draws them
    are the page's, so nothing of either is here. The serial is what makes the
    pair a pair - a plot's identity is a worksheet and a label, which is too
    much to send on every frame of a drag, so the pane numbers its plots and
    both sides use the number.
    """

    serial: int = 0
    #: Which re-sample the page is drawing. A job whose generation has moved on
    #: is a job about a view that is gone, and the page drops its answer.
    generation: int = 0
    #: Whether this plot has ever been sampled, which is the only thing a data
    #: plot needs remembered: `plot/resample.py`'s rule is that a matrix of
    #: constants is the same matrix at every zoom, so it is sampled once and
    #: clipped by the side that draws it.
    sampled: bool = False
    hidden: bool = False

    @property
    def visible(self) -> bool:
        return not self.hidden


class WebBackend:
    """Plot windows as panes in a page, made where the session asks for one."""

    def __init__(self, page: Any, engine: Any) -> None:
        #: The page's plotting module, which is what a pane is actually made
        #: of, and the engine the samplings are posted through.
        self.page = page
        self.engine = engine
        self.panes: dict[int, Pane] = {}
        self._proxies: list[Any] = []

    def open(
        self, session: Any, kind: WindowKind, number: int, preferences: Prefer
    ) -> Pane:
        """One pane, opened in the page. A solid refuses in words for now.

        The refusal is raised rather than returned because the session's
        `Refused` is a reply to a request and this is a failure to build a
        window; `WebPlots` turns it back into the one sentence the message line
        prints, which is where every plot refusal in the program ends up.
        """
        if kind is WindowKind.THREE_D:
            raise ValueError(UNDRAWN_3D)
        pane = Pane(session, number, self)
        self.panes[number] = pane
        return pane

    def close(self, number: int) -> None:
        """Forget a pane that has closed itself. Its own event said so."""
        self.panes.pop(number, None)

    def stop(self) -> None:
        """Take every pane off the page: the app has gone."""
        for pane in list(self.panes.values()):
            pane.dismiss()
        self.panes.clear()
        self.page.stop()
        for proxy in self._proxies:
            proxy.destroy()
        self._proxies = []

    def handed(self, table: dict[str, Callable[..., Any]]) -> Any:
        """A table of this side's callbacks, as the page can hold and call them.

        Proxied explicitly and kept, because the page holds them for as long as
        a pane is open and a proxy nobody keeps is a proxy the garbage collector
        may take out from under it.
        """
        import js
        from pyodide.ffi import create_proxy, to_js

        proxies = {name: create_proxy(call) for name, call in table.items()}
        self._proxies.extend(proxies.values())
        return to_js(proxies, dict_converter=js.Object.fromEntries)


class Pane:
    """One 2D plot window, drawn by the page and bookkept here.

    Everything the session asks of a window, and nothing about a picture. What
    the page asks back is on the other side of the file: a view that moved, a
    plot to be read out, a point to be sent home, a pane the user closed.
    """

    kind = WindowKind.TWO_D

    def __init__(self, session: Any, number: int, backend: WebBackend) -> None:
        self.session = session
        self.number = number
        self.backend = backend
        self.plots: list[Shown] = []
        self.current = False
        self.title = ""
        #: Whether the expressions in this pane are written in degrees, which
        #: is what a polar curve is composed in and what a reading is in.
        self.degrees = False
        #: Whether the view reads a univariate curve as r = f(θ). The page's
        #: toggle sets it and every curve in the pane is reread on the spot.
        self.polar = False
        self._counter = 0
        self._serial = 0
        self.page = backend.page.open(
            number,
            # How long the view has to stand still before the curves are
            # sampled for it. `plot/resample.py` says it for both backends, so
            # a drag debounces the same in a browser as it does on a desktop.
            resample.RESAMPLE_DELAY_MS,
            backend.handed(
                {
                    "changed": self.changed,
                    "closed": self.dismissed,
                    "touched": self.touched,
                    "traced": self.traced,
                    "scanned": self.scanned,
                    "author": self.author,
                    "polar": self.repolar,
                    "home": self.home,
                    "connect": self.connect,
                    "hide": self.hide,
                    "drop": self.drop,
                }
            ),
        )

    # -- what the session asks ---------------------------------------------

    def add(self, plot: Plot) -> Shown:
        """Put a plot in the pane, replacing one with the same identity.

        Replacing rather than adding is what the zoom-and-plot-again habit
        needs: the same label plotted twice is one curve, and it keeps the
        color it had so that the picture does not reshuffle under the reader.
        """
        fields = Plot.__dataclass_fields__
        shown = Shown(**{name: getattr(plot, name) for name in fields})
        existing = self.find(plot.worksheet, plot.label)
        if existing is not None:
            shown.color, shown.paper = existing.color, existing.paper
            self.remove(existing)
        else:
            index = self._counter % len(PALETTE)
            shown.color, shown.paper = PALETTE[index], PAPER_PALETTE[index]
            self._counter += 1
        self._serial += 1
        shown.serial = self._serial
        # The angle unit is the worksheet's, and a pane reads its numbers out
        # in whatever unit the expressions in it are written in.
        self.degrees = plot.context.angle is Angle.DEGREE
        # Polar is the view's, not the plot's: a univariate curve arriving in a
        # polar pane is read as r = f(θ) from the start.
        reread = view.reread(shown.kind, self.polar)
        if reread is not None:
            shown.kind = reread
        self.plots.append(shown)
        self.page.add(shown.serial, self._spec(shown))
        self.retitle()
        self._start(shown, fresh=True)
        return shown

    def remove(self, plot: Shown) -> None:
        """Take one plot out of the pane."""
        if plot in self.plots:
            self.plots.remove(plot)
            self.page.remove(plot.serial)
        self.retitle()

    def find(self, worksheet: int, label: str) -> Shown | None:
        """The plot a worksheet and a label name, if this pane has it."""
        for plot in self.plots:
            if plot.worksheet == worksheet and plot.label == label:
                return plot
        return None

    def present(self) -> None:
        """A plot has landed here: show the pane and put it in front."""
        self.page.present()

    def retitle(self, current: bool | None = None) -> None:
        """Title the pane by what it holds, saying whether it is the receiver."""
        if current is not None:
            self.current = current
        self.title = protocol.titled(
            self.kind,
            tuple(plot.text or plot.label for plot in self.plots),
            self.current,
        )
        self.page.retitle(self.title, self.current)

    def describe(self) -> protocol.WindowInfo:
        """This pane as `Describe` reports it."""
        (left, right), (low, high) = self._where()
        return protocol.WindowInfo(
            number=self.number,
            kind=self.kind,
            title=self.title,
            current=self.current,
            plots=tuple(
                protocol.PlotInfo(
                    worksheet=plot.worksheet,
                    label=plot.label,
                    text=plot.text,
                    kind=plot.kind,
                    hidden=plot.hidden,
                )
                for plot in self.plots
            ),
            xrange=(left, right),
            yrange=(low, high),
        )

    def close(self) -> None:
        """Close this pane, which reports itself closed as any other close does."""
        self.dismiss()
        self.session.closed(self.number)

    def dismiss(self) -> None:
        """Take the pane off the page and say nothing: the session is going."""
        self.page.dismiss()

    # -- what the page says ------------------------------------------------

    def changed(self) -> None:
        """The view moved: sample every plot again for what is shown now.

        Debounced by the page before it gets here and keyed by the sampler
        after, which between them are what make a drag cost one sampling per
        curve rather than one per frame.
        """
        for plot in self.plots:
            self._start(plot)

    def touched(self) -> None:
        """The user clicked in this pane, so the next plot lands in it."""
        self.session.touched(self.number)

    def dismissed(self) -> None:
        """The user closed this pane."""
        self.session.closed(self.number)

    def author(self, serial: int, text: str) -> None:
        """A traced point sent home, to be authored into its own worksheet.

        The text is the one the worker wrote when it read the point out - the
        page carries it and spells nothing itself, so what lands in the
        worksheet is the number the status line named.
        """
        plot = self._plot(serial)
        if plot is not None and text:
            self.session.author(plot.worksheet, str(text))

    def home(self, width: float, height: float) -> Any:
        """The default framing on a canvas of this shape, as four numbers.

        Asked of this side rather than worked out on the page, so that the
        browser opens on the framing the desktop opens on and `plot/view.py`
        stays the one place that says what that is.
        """
        (left, right), (low, high) = view.home_range(float(width), float(height))
        return _js([left, right, low, high])

    def repolar(self, polar: bool) -> None:
        """The view's polar toggle moved: reread every curve it applies to."""
        self.polar = bool(polar)
        for plot in self.plots:
            reread = view.reread(plot.kind, self.polar)
            if reread is None:
                continue
            plot.kind = reread
            self.page.respec(plot.serial, self._spec(plot))
            self._start(plot)

    def connect(self, serial: int, connected: bool) -> None:
        """A data plot's right-click menu joined its points, or let them loose.

        Sticky, as it is on the desktop: what the menu was last told is what
        the next data plot starts out as, and the app is told so that a state
        file keeps it.
        """
        plot = self._plot(serial)
        if plot is None:
            return
        plot.connected = bool(connected)
        self.page.respec(plot.serial, self._spec(plot))
        self.session.adjusted(connected=plot.connected)

    def hide(self, serial: int, hidden: bool) -> None:
        """A legend row was clicked: take the curve out of the picture, or back.

        Hidden rather than removed, which is what a legend row means everywhere:
        the plot is still in the pane, `Describe` still names it, and it comes
        back when the row is clicked again.
        """
        plot = self._plot(serial)
        if plot is None:
            return
        plot.hidden = bool(hidden)
        self.page.respec(plot.serial, self._spec(plot))
        if not plot.hidden:
            self._start(plot)

    def drop(self, serial: int) -> None:
        """The menu's `Remove`: take one plot out of the pane for good."""
        plot = self._plot(serial)
        if plot is not None:
            self.remove(plot)

    def traced(self, serial: int, at: float) -> None:
        """Read one curve out where the marker now is."""
        plot = self._plot(serial)
        if plot is None:
            return
        request = asking.Trace(
            pane=self.number,
            plot=plot.serial,
            generation=plot.generation,
            model=_bare(plot),
            at=float(at),
            degrees=self.degrees,
        )
        self.session.sample((self.number, TRACE), lambda _report: request, self._said)

    def scanned(self, serial: int) -> None:
        """Find every notable point of one curve, for the view it is drawn in.

        The whole list at once, so that stepping through it with Tab is a key
        press and not a question asked across a worker.
        """
        plot = self._plot(serial)
        if plot is None:
            return
        others = tuple(
            _bare(other)
            for other in self.plots
            if other is not plot and other.visible and other.kind in FUNCTIONS
        )
        (left, right), (low, high) = self._where()
        request = asking.Features(
            pane=self.number,
            plot=plot.serial,
            generation=plot.generation,
            model=_bare(plot),
            others=others,
            xrange=(left, right),
            yrange=(low, high),
            size=self._size(),
        )
        key = (self.number, FEATURES, plot.serial)
        self.session.sample(key, lambda _report: request, self._said)

    # -- sampling ----------------------------------------------------------

    def _start(self, plot: Shown, fresh: bool = False) -> None:
        """Ask the worker for this plot over the view the page is showing.

        `plot/resample.py`'s rule about a data plot is kept here rather than
        called, because the rule reads a plot's samples and this side has none:
        a matrix of constants is the same matrix at every zoom, so it is
        sampled once and the page clips it thereafter.
        """
        if plot.hidden:
            return
        if plot.kind is PlotKind.DATA and plot.sampled:
            return
        plot.generation += 1
        (left, right), (low, high) = self._where()
        request = asking.Sample(
            pane=self.number,
            plot=plot.serial,
            generation=plot.generation,
            model=_bare(plot),
            xrange=(left, right),
            yrange=(low, high),
            size=self._size(),
            degrees=self.degrees,
        )
        plot.sampled = True
        self.page.starting(plot.serial, plot.generation, fresh)
        self.session.sample(
            (self.number, plot.serial),
            lambda _report: request,
            lambda answer: self._answered(plot, answer),
        )

    def _answered(self, plot: Shown, answer: Any) -> None:
        """One sampling is over, as the page reported it.

        The arrays went to the page and were drawn before this ran; what
        arrives here is what could not be drawn, which is the one thing the app
        has to hear about. A curve that will not evaluate reports itself.
        """
        message = _trouble(answer)
        plot.trouble = message
        if message:
            self.session.trouble(self.number, plot.label, message)

    def _said(self, answer: Any) -> None:
        """A reading or a scan is over. Only a failure is worth a word here."""
        message = _trouble(answer)
        if message:
            self.session.trouble(self.number, "", message)

    # -- the pane's own bookkeeping ----------------------------------------

    def _plot(self, serial: Any) -> Shown | None:
        for plot in self.plots:
            if plot.serial == int(serial):
                return plot
        return None

    def _spec(self, plot: Shown) -> Any:
        """How the page is to draw one plot: its look and its name, no numbers.

        `named` is what the app wrote, cut to a line. Nothing on the page ever
        renders an expression, which is the same rule that keeps the desktop's
        windows from doing it.
        """
        return _js(
            {
                "kind": str(plot.kind),
                "label": plot.label,
                "name": plot.named,
                "color": plot.color,
                "connected": bool(plot.connected),
                "size": float(plot.point_size),
                "hidden": bool(plot.hidden),
                "trange": [float(plot.trange[0]), float(plot.trange[1])],
            }
        )

    def _where(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """What the page is showing, as two ranges."""
        shown = self.page.view()
        return (float(shown[0]), float(shown[1])), (float(shown[2]), float(shown[3]))

    def _size(self) -> tuple[float, float]:
        """How many pixels across and up the picture is, which is the tolerance.

        Sampling is in screen space: a quarter of a pixel is what a person can
        see, so how big the canvas is decides how finely a curve is cut.
        """
        shown = self.page.view()
        return (max(float(shown[4]), 100.0), max(float(shown[5]), 100.0))



class WorkerExecutor:
    """The engine worker, running the sampler's jobs one at a time.

    The third implementation of the session's `Executor`, and the one the
    protocol's pull shape was made for: a job is taken when this side is free,
    which is exactly what one request in flight means. The worker is one thread
    of one interpreter, so a second request would wait inside it rather than
    beside it, and waiting inside it would put a stale drag in front of the
    Simplify the user has just pressed.

    What it does not do is wait for its own answers. A sampling answers to the
    page, in arrays, and the page says only that the answer landed - so `landed`
    is what lets go of the flight and pulls the next job, and there is nothing
    here that ever holds a number.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._sampler: Any = None
        #: The request now with the worker, as its number and what to do when
        #: it comes back. One, by design.
        self._flight: tuple[int, Callable[[Any], None]] | None = None

    def serving(self, sampler: Any) -> None:
        self._sampler = sampler

    def wake(self) -> None:
        self._pull()

    def deliver(self, done: Callable[[Any], None], answer: Any) -> None:
        """Hand one answer over. There is one thread here, so this is a call."""
        done(answer)

    def landed(self, number: Any, trouble: Any = "") -> None:
        """The page has drawn the answer to this request, or could not.

        The one thing the page tells this side about a sampling, and what makes
        the queue move: the arrays are already on the screen by the time it is
        called.
        """
        flight = self._flight
        if flight is None or flight[0] != int(number):
            return
        self._flight = None
        self.deliver(flight[1], str(trouble or ""))
        self._pull()

    def lost(self, reason: str) -> None:
        """The worker is gone, taking whatever it was sampling with it.

        Esc terminates the worker, which is the browser's whole abort, and the
        closures and the sampling in flight go with it. The plot list survives,
        because it lives on this side; the cost is a re-lambdify on the next
        view change, which is the maybe-empty cache contract doing its job.
        """
        flight, self._flight = self._flight, None
        if flight is not None:
            self.deliver(flight[1], reason)

    def _pull(self) -> None:
        """Send the next job, if this side is free and there is one to send."""
        if self._flight is not None or self._sampler is None:
            return
        job = self._sampler.take()
        if job is None:
            return
        work, done = job
        request = work()
        if isinstance(request, Exception):
            self.deliver(done, said(request))
            self._pull()
            return
        number = self._engine.numbered()
        self._flight = (number, done)
        asyncio.ensure_future(self._sent(number, request))

    async def _sent(self, number: int, request: Any) -> None:
        """Post one request, and answer for it here where it cannot be posted.

        A worker that is booting is waited for and a worker that is down is
        replaced, both of which are the engine's own policy; what is left is a
        refusal, and a refusal is a `Trouble` like any other.
        """
        try:
            await self._engine.ask(number, _named(request), (request,))
        except Exception as error:
            self.landed(number, said(error))


def _named(request: Any) -> str:
    """Which of the worker's plot methods a request is for."""
    if isinstance(request, asking.Sample):
        return asking.SAMPLE
    if isinstance(request, asking.Trace):
        return asking.TRACE
    return asking.FEATURES


def _bare(plot: Shown) -> Plot:
    """The plot as the worker is sent it: what to draw, and nothing of a pane.

    A `Shown` would travel just as well and would carry the pane's bookkeeping
    with it, which is this side's business and no part of a sampling.
    """
    return Plot(**{name: getattr(plot, name) for name in Plot.__dataclass_fields__})


def _trouble(answer: Any) -> str:
    """What went wrong, out of whatever the page or the queue answered with."""
    if isinstance(answer, Exception):
        return said(answer)
    return str(answer or "")


def _js(value: Any) -> Any:
    """One plain Python structure, as the page reads it: objects and arrays."""
    import js
    from pyodide.ffi import to_js

    return to_js(value, dict_converter=js.Object.fromEntries)
