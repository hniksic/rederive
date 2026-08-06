"""Plot windows as panes in a page, and the worker that samples what they hold.

The `Backend` the plot session opens windows on in a browser. It runs on the
main thread, beside the app, and what it owns is exactly what has no numbers in
it: the plot list, the colors the palette hands out, which pane a plot lands in,
what `Describe` answers, and the routing of everything a pane has to say back.
The picture itself - the view, the gestures, the drawing - is JavaScript's, in
`web/plot2d.js` for a flat one and `web/plot3d.js` for a solid; the samples are
the engine worker's. Nothing here imports numpy, and nothing here ever holds an
array.

The two kinds of pane are two classes here for the reason they are two files
there: what a 2D pane spends itself on is the view - the framing, the gestures,
the marker riding a curve - and a 3D pane has none of those. Its whole business
is the domain, the grid it is sampled on and the box the surfaces stand in, and
the mouse never touches any of them: turning a solid is a camera move over
vertices that are already on the card.

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
from rederive.plot.model import (
    FUNCTIONS,
    PALETTE,
    PAPER_PALETTE,
    SOLID_PALETTE,
    SOLID_PAPER,
    Plot,
    Surface,
    written,
)
from rederive.plot.protocol import PlotKind, Prefer, WindowKind
from rederive.plot.session import said
from rederive.plot.web import protocol as asking

__all__ = ["Pane", "Shown", "Solid", "Standing", "WebBackend", "WorkerExecutor"]

#: What a pane calls its own sampling jobs, beside the one per plot. Keyed like
#: every other job, so a marker dragged along a curve costs one reading and not
#: one per pixel it passed over.
TRACE = "trace"
FEATURES = "features"

#: What a 3D pane says when a field will not read. The domain fields take
#: numbers here and expressions on the desktop, which is the one thing a
#: browser's toolbar cannot do: `-π` is a tree, and what a tree is worth is
#: arithmetic that lives in the worker.
NOT_NUMBERS = "The domain is four numbers, the grid two"
INVERTED = "The domain runs from a lower bound to a higher one"


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


class Standing(Surface):
    """A surface as a pane holds it: a name both sides know it by, and no numbers.

    `Shown` for a solid, and it carries two more things because a surface has
    two more to remember. The z it stands in is the pane's rather than its own -
    two surfaces in one picture are being compared, and comparing them through
    two vertical scales would be a picture that lies - so what it asked for and
    what it was last drawn in are both kept here, and a difference between them
    is what asks for the mesh again.
    """

    serial: int = 0
    #: Which evaluation the page is drawing. A job whose generation has moved
    #: on is a job about a domain that is gone, and the page drops its answer.
    generation: int = 0
    #: The z extent this surface's own values ask for, once they have been
    #: evaluated, and the one its mesh was last built to.
    wanted: tuple[float, float] | None = None
    standing: tuple[float, float] | None = None


class WebBackend:
    """Plot windows as panes in a page, made where the session asks for one."""

    def __init__(self, page: Any, solids: Any, engine: Any) -> None:
        #: The page's two plotting modules, which are what a pane is actually
        #: made of, and the engine the samplings are posted through. Two of
        #: them because a flat picture and a solid one share no drawing at all:
        #: one is a chart library and a canvas, the other is a scene and a card.
        self.page = page
        self.solids = solids
        self.engine = engine
        self.panes: dict[int, Any] = {}
        self._proxies: list[Any] = []

    def open(
        self, session: Any, kind: WindowKind, number: int, preferences: Prefer
    ) -> Any:
        """One pane, opened in the page, with the preferences in force now.

        A 2D pane takes none: it always opens with equal scales, and its polar
        toggle is the one-window exception that is deliberately not remembered.
        A 3D one takes both of its - how finely the next surface is sampled and
        whether it is drawn as the wire grid of its samples - which is what
        makes a grid typed in one pane the grid the next pane opens on.
        """
        if kind is WindowKind.THREE_D:
            pane: Any = Solid(
                session, number, self, preferences.grid, preferences.wire
            )
        else:
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
        self.solids.stop()
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


class Solid:
    """One 3D plot window, drawn by the page and bookkept here.

    The same shape as `Pane` and a shorter one, because a 3D window has less to
    keep: there is no view to move, no marker to ride and no framing to argue
    with. What it owns is the domain, how finely that is sampled, whether the
    surfaces are drawn as wire, and the box they all stand in - and only the
    first two of those start an evaluation, which is the promise the desktop's
    window is built on and the reason turning a solid in a browser costs
    nothing at all.

    The box is the one piece of bookkeeping the 2D side has no counterpart for.
    Two surfaces in one picture are being compared, so they stand in one box or
    the picture lies; the box is the union of what each surface's values ask
    for, and a surface whose mesh was built to a different one is asked for
    again. A pane holding one surface therefore evaluates once for it: the
    surface says what the box is and is drawn in the box it said.
    """

    kind = WindowKind.THREE_D

    def __init__(
        self, session: Any, number: int, backend: WebBackend, grid: int, wire: bool
    ) -> None:
        self.session = session
        self.number = number
        self.backend = backend
        self.plots: list[Standing] = []
        self.current = False
        self.title = ""
        self.xdomain = view.DEFAULT_DOMAIN
        self.ydomain = view.DEFAULT_DOMAIN
        # The grid the pane opens with is the sticky one the session holds -
        # the grid the last surface was given - clamped here as a typed one is:
        # a pane never samples finer than it can draw, whoever asked.
        square = _grid(grid)
        self.grid = (square, square)
        #: How a surface arriving here is drawn: the sticky look the last
        #: surface anywhere was left in, and thereafter this pane's own toggle.
        self.wired = bool(wire)
        #: The z the picture stands in, once anything in it has been evaluated.
        self.zrange: tuple[float, float] | None = None
        self._counter = 0
        self._serial = 0
        self.page = backend.solids.open(
            number,
            backend.handed(
                {
                    "closed": self.dismissed,
                    "touched": self.touched,
                    "framed": self.framed,
                    "stood": self.stood,
                    "wire": self.rewire,
                    "hide": self.hide,
                    "drop": self.drop,
                }
            ),
        )
        self._show()
        self.page.meshed(self.wired)

    # -- what the session asks ---------------------------------------------

    def add(self, plot: Plot) -> Standing:
        """Put a surface in the pane, replacing one with the same identity."""
        fields = Surface.__dataclass_fields__
        standing = Standing(**{name: getattr(plot, name) for name in fields})
        existing = self.find(plot.worksheet, plot.label)
        if existing is not None:
            # A replacement keeps the look of what it replaces - the color and
            # the wire choice - so re-plotting does not reshuffle the picture.
            standing.color, standing.paper = existing.color, existing.paper
            standing.wire = existing.wire
            self.remove(existing)
        else:
            index = self._counter % len(SOLID_PALETTE)
            standing.color, standing.paper = SOLID_PALETTE[index], SOLID_PAPER[index]
            self._counter += 1
            standing.wire = self.wired
        self._serial += 1
        standing.serial = self._serial
        self.plots.append(standing)
        self.page.add(standing.serial, self._spec(standing))
        self.retitle()
        self._start(standing)
        return standing

    def remove(self, plot: Standing) -> None:
        """Take one surface out of the pane, and rebuild the box without it."""
        if plot in self.plots:
            self.plots.remove(plot)
            self.page.remove(plot.serial)
        self.retitle()
        self._rebox()

    def find(self, worksheet: int, label: str) -> Standing | None:
        """The surface a worksheet and a label name, if this pane has it."""
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
        """This pane as `Describe` reports it.

        The ranges reported are the domain, since that is what a 3D window is
        framed by - the same answer the desktop's window gives.
        """
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
            xrange=(float(self.xdomain[0]), float(self.xdomain[1])),
            yrange=(float(self.ydomain[0]), float(self.ydomain[1])),
        )

    def close(self) -> None:
        """Close this pane, which reports itself closed as any other close does."""
        self.dismiss()
        self.session.closed(self.number)

    def dismiss(self) -> None:
        """Take the pane off the page and say nothing: the session is going."""
        self.page.dismiss()

    # -- what the page says ------------------------------------------------

    def touched(self) -> None:
        """The user clicked in this pane, so the next surface lands in it."""
        self.session.touched(self.number)

    def dismissed(self) -> None:
        """The user closed this pane."""
        self.session.closed(self.number)

    def framed(self, *typed: Any) -> None:
        """The domain and the grid as somebody has just typed them.

        The one place in this pane where typing changes what is computed. Six
        numbers, and numbers is what they must be: the desktop's fields read
        expressions - `-π` is an answer there - and what a tree is worth is
        arithmetic that lives where the closures do, which is too far away for
        a field to wait on. A field that does not read is put back rather than
        argued with, since the value it would take is on the screen beside it.
        """
        try:
            bounds = tuple(float(str(value)) for value in typed[:4])
            grid = (_grid(float(str(typed[4]))), _grid(float(str(typed[5]))))
        except ValueError:
            self._show()
            self.page.said(NOT_NUMBERS)
            return
        if bounds[1] <= bounds[0] or bounds[3] <= bounds[2]:
            self._show()
            self.page.said(INVERTED)
            return
        if grid != self.grid:
            # A typed grid is sticky: the next pane opens on it. The sticky
            # value is one count per axis, so a rectangular grid hands on its
            # finer axis. The domain is not sticky - it is a framing, like a 2D
            # view - so only the grid goes back.
            self.session.adjusted(grid=max(grid))
        changed = (bounds[:2], bounds[2:], grid) != (
            self.xdomain,
            self.ydomain,
            self.grid,
        )
        self.xdomain, self.ydomain, self.grid = bounds[:2], bounds[2:], grid
        self._show()
        if changed:
            self.page.said(
                f"Evaluating over the new domain at {grid[0]} by {grid[1]}"
            )
            self.reevaluate()

    def stood(self, serial: Any, wanted: Any, drawn: Any) -> None:
        """A surface has been drawn: what it asks for in z, and what it got.

        The page says it because the numbers came back with the arrays and were
        drawn before this side heard anything at all. What is done with them is
        this side's: the box is the union of what the surfaces ask for, and a
        mesh built to any other box is asked for again.
        """
        plot = self._plot(serial)
        if plot is None:
            return
        plot.wanted = _pair(wanted)
        plot.standing = _pair(drawn)
        self._rebox()

    def rewire(self, wired: Any) -> None:
        """The pane's `mesh` box: every surface as wire, or every one solid.

        The look belongs to the surfaces, so the box flips them all rather than
        holding a state of its own, and the look it leaves is the sticky one:
        the next surface, here or in the next pane, arrives drawn this way.
        Nothing is evaluated for it - both drawings are on the card already.
        """
        self.wired = bool(wired)
        for plot in self.plots:
            plot.wire = self.wired
            self.page.respec(plot.serial, self._spec(plot))
        self.page.meshed(self.wired)
        self.session.adjusted(wire=self.wired)

    def hide(self, serial: Any, hidden: Any) -> None:
        """A legend row was clicked: take the surface out of the picture, or back.

        Hidden rather than removed, as it is everywhere: the surface is still in
        the pane and `Describe` still names it. The box is rebuilt around what
        is left, since a hidden surface is not one the others are compared with.
        """
        plot = self._plot(serial)
        if plot is None:
            return
        plot.hidden = bool(hidden)
        self.page.respec(plot.serial, self._spec(plot))
        self._rebox()

    def drop(self, serial: Any) -> None:
        """The menu's `Remove`: take one surface out of the pane for good."""
        plot = self._plot(serial)
        if plot is not None:
            self.remove(plot)

    # -- evaluation --------------------------------------------------------

    def reevaluate(self) -> None:
        """Every surface again, which is what a new domain or grid asks for."""
        for plot in self.plots:
            self._start(plot)

    def _start(self, plot: Standing) -> None:
        """Ask the worker for this surface over the domain and grid.

        The only thing that starts an evaluation. A camera move does not come
        near here, which is the promise the picture is built on: the mesh is
        about the domain, and the domain is only ever changed by typing in it.
        """
        plot.generation += 1
        request = asking.Grid(
            pane=self.number,
            plot=plot.serial,
            generation=plot.generation,
            model=_solid(plot),
            xdomain=self.xdomain,
            ydomain=self.ydomain,
            grid=self.grid,
            zrange=self._box(plot),
        )
        self.page.starting(plot.serial, plot.generation)
        self.session.sample(
            (self.number, plot.serial),
            lambda _report: request,
            lambda answer: self._answered(plot, answer),
        )

    def _box(self, plot: Standing) -> tuple[float, float] | None:
        """The z this surface is to be placed in, or None for it to say.

        A surface alone in a picture decides the box, because its own values
        are the whole of what the picture holds; one arriving beside others is
        drawn in theirs and may move it once it has been evaluated.
        """
        others = [
            one
            for one in self.plots
            if one is not plot and one.visible and one.standing is not None
        ]
        return self.zrange if others else None

    def _rebox(self) -> None:
        """The box every surface stands in, and the meshes that are not in it."""
        asked = [
            plot.wanted
            for plot in self.plots
            if plot.visible and plot.wanted is not None
        ]
        if not asked:
            return
        self.zrange = (min(low for low, _ in asked), max(high for _, high in asked))
        for plot in self.plots:
            if plot.visible and plot.standing not in (None, self.zrange):
                self._start(plot)

    def _answered(self, plot: Standing, answer: Any) -> None:
        """One evaluation is over, as the page reported it.

        The arrays went to the page and were drawn before this ran; what
        arrives here is what could not be drawn, which is the one thing the app
        has to hear about. A surface that will not evaluate reports itself.
        """
        message = _trouble(answer)
        plot.trouble = message
        if message:
            self.session.trouble(self.number, plot.label, message)

    # -- the pane's own bookkeeping ----------------------------------------

    def _plot(self, serial: Any) -> Standing | None:
        for plot in self.plots:
            if plot.serial == int(serial):
                return plot
        return None

    def _spec(self, plot: Standing) -> Any:
        """How the page is to draw one surface: its look and its name, no numbers."""
        return _js(
            {
                "label": plot.label,
                "name": plot.named,
                "color": plot.color,
                "wire": bool(plot.wire),
                "hidden": bool(plot.hidden),
            }
        )

    def _show(self) -> None:
        """Put the pane's own numbers back in the fields, however they got there."""
        self.page.domain(
            written(self.xdomain[0]),
            written(self.xdomain[1]),
            written(self.ydomain[0]),
            written(self.ydomain[1]),
            written(self.grid[0]),
            written(self.grid[1]),
        )


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
    if isinstance(request, asking.Grid):
        return asking.GRID
    return asking.FEATURES


def _bare(plot: Shown) -> Plot:
    """The plot as the worker is sent it: what to draw, and nothing of a pane.

    A `Shown` would travel just as well and would carry the pane's bookkeeping
    with it, which is this side's business and no part of a sampling.
    """
    return Plot(**{name: getattr(plot, name) for name in Plot.__dataclass_fields__})


def _solid(plot: Standing) -> Surface:
    """The surface as the worker is sent it, `_bare`'s answer for a solid.

    The color travels with it, unlike a curve's: a surface is shaded where its
    geometry is built, so the side that knows how bright each vertex is has to
    know what color it is being made bright in.
    """
    return Surface(
        **{name: getattr(plot, name) for name in Surface.__dataclass_fields__}
    )


def _grid(count: Any) -> int:
    """One axis of a grid, held to what a pane will sample and draw."""
    return int(min(max(int(count), 2), view.MAX_GRID))


def _pair(values: Any) -> tuple[float, float] | None:
    """Two numbers the page handed back, or None where it had none to hand.

    A surface with nothing real over its domain asks for no z at all, and the
    box is then whatever the rest of the picture wanted.
    """
    numbers = [float(value) for value in values] if values is not None else []
    return (numbers[0], numbers[1]) if len(numbers) == 2 else None


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
