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

What a pane offers is the third thing that crosses, and it crosses the way the
samples do not: rarely. `plot/controls.py` says what either kind of window does,
in what words, under what keys and in what order, and a pane here renders none
of it - the page is handed the controls this backend can serve when it is made,
and the menu as it should currently read when a right click asks for one. The
page answers with the name of a control and nothing else. Which means the view
still lives where a drag can reach it without asking anybody, and the words
still live where there is one of them.

What a pane asks to be told crosses the same way, and `plot/forms.py` is where
it is written: the four bounds of a view, the parameter range of a curve, the
domain a solid is evaluated over and the box and camera of its inspector are
fields, words and refusal sentences, and the page draws an overlay and a tool
row out of them exactly as the desktop draws a dialog and a toolbar. The text
comes back here as text. It is parsed on this side, which costs nothing, and
what the trees are worth is asked of the worker - so `-π` is an answer in a
browser for the same reason it is one on a desktop, and the thread that paints
the screen still holds no mathematics.

The one number a 3D pane works out for itself is the box its surfaces stand in,
and it works it out with the desktop's own arithmetic: what crosses from the
worker is four numbers a surface, and `plot/view.py` says how several of those
become one z. A picture in a browser and a picture on a desktop are therefore
clipped at the same height, by one rule written in one place.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from rederive.engine.context import Angle, Context
from rederive.plot import actions, controls, forms, protocol, resample, view
from rederive.plot.model import (
    FIELDS,
    FUNCTIONS,
    PALETTE,
    PAPER_PALETTE,
    SOLID_PALETTE,
    SOLID_PAPER,
    Plot,
    Surface,
    roughly,
    written,
)
from rederive.plot.protocol import PARAMETRIZED, PlotKind, Prefer, WindowKind
from rederive.plot.session import said
from rederive.plot.web import protocol as asking
from rederive.syntax import ParseState

__all__ = ["Pane", "Shown", "Solid", "Standing", "WebBackend", "WorkerExecutor"]

#: What a pane calls its own sampling jobs, beside the one per plot. Keyed like
#: every other job, so a marker dragged along a curve costs one reading and not
#: one per pixel it passed over.
TRACE = "trace"
FEATURES = "features"
#: And the two jobs that read typed bounds - the four of a flat view and the
#: four of a solid's domain. Keyed too, so a form applied twice while the first
#: answer is still out costs one evaluation.
RANGE = "range"
DOMAIN = "domain"

#: What a pane says about a picture that has left it. Copying and exporting are
#: the one pair of commands each backend is left to answer outright - a painter
#: path on one side and a canvas on the other share nothing worth describing -
#: so the words a page says about them are here, beside the page, rather than in
#: the table both backends read.
COPIED_IMAGE = "Copied the plot to the clipboard"
COPIED_POINT = "Copied {text}"
CLIPBOARD_REFUSED = "The browser did not allow the clipboard: {trouble}"
#: The sentence a download owes. It names pixels because that is what it is: a
#: canvas has no painter path to write an SVG out of, so what a page can export
#: is the picture at the size it is drawn, and saying the size is what makes the
#: difference from the desktop's vector file legible rather than a surprise.
DOWNLOADED = "Downloaded {name}, {wide} by {tall} pixels"
DOWNLOAD_REFUSED = "The browser refused the download: {trouble}"


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
    #: What this surface's own values measure in z, once they have been
    #: evaluated, and the box its mesh was last built to. The first is the
    #: summary `plot/view.py` pools into the second over the whole pane.
    span: view.Span | None = None
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
        #: Where the view has been and whether the scales are locked equal. The
        #: ranges themselves stay on the page, which is the side a drag talks
        #: to; what is here is the history, which is a command and not a
        #: gesture, and `plot/actions.py` is where either becomes a framing.
        self._framing = actions.Framing()
        #: The parse state and the context a typed bound is read under: the last
        #: plot's, since a framing belongs to the pane and the worksheet that
        #: last plotted into it is its reader.
        self._state = ParseState()
        self._context = Context()
        self._counter = 0
        self._serial = 0
        #: The plot the last menu was raised over, which is what the entries
        #: about one plot act on. Held here rather than sent back with the
        #: click, as the desktop's windows hold it: what the page knows about a
        #: plot is a serial, and what a command needs is the plot.
        self._pointed: Shown | None = None
        #: The plot the marker was last put on, which is how taking hold of a
        #: curve chooses whose parameter range the tool row is about.
        self._ridden: Shown | None = None
        self.page = backend.page.open(
            number,
            # How long the view has to stand still before the curves are
            # sampled for it. `plot/resample.py` says it for both backends, so
            # a drag debounces the same in a browser as it does on a desktop.
            resample.RESAMPLE_DELAY_MS,
            _controls(controls.FLAT, self.commands()),
            _strip(forms.TRANGE),
            backend.handed(
                {
                    "changed": self.changed,
                    "closed": self.dismissed,
                    "touched": self.touched,
                    "traced": self.traced,
                    "scanned": self.scanned,
                    "author": self.author,
                    "home": self.home,
                    "fitted": self.fitted,
                    "hide": self.hide,
                    "menu": self.menu,
                    "card": self.card,
                    "command": self.command,
                    "remembered": self.remembered,
                    "centred": self.centred,
                    "typed": self.typed,
                    "ranged": self.ranged,
                    "spanned": self.spanned,
                    "copied": self.copied,
                    "exported": self.exported,
                }
            ),
        )
        # The polar reading is this side's and is lit from here; the aspect
        # lock is the page's, being view state a rubber band releases without
        # asking anybody, and the page lights that one itself.
        self.page.lit("view.polar", self.polar)

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
        # in whatever unit the expressions in it are written in; the grammar and
        # the context a typed bound is read under come from the same place.
        self.degrees = plot.context.angle is Angle.DEGREE
        self._state, self._context = plot.state, plot.context
        # Polar is the view's, not the plot's: a univariate curve arriving in a
        # polar pane is read as r = f(θ) from the start.
        reread = view.reread(shown.kind, self.polar)
        if reread is not None:
            shown.kind = reread
        self.plots.append(shown)
        self.page.add(shown.serial, self._spec(shown))
        self.retitle()
        self._axes()
        self._parameter()
        self._start(shown, fresh=True)
        return shown

    def remove(self, plot: Shown) -> None:
        """Take one plot out of the pane."""
        if plot in self.plots:
            self.plots.remove(plot)
            self.page.remove(plot.serial)
        if self._ridden is plot:
            self._ridden = None
        if self._pointed is plot:
            self._pointed = None
        self.retitle()
        self._axes()
        self._parameter()

    def clear(self) -> None:
        """The Del key and the tool row's `clear`: start this picture over.

        The pane it acts on is the pane the key was pressed in, so there is
        nothing to infer and nothing to report. The axis names go with the
        curves that named them and the range fields with the plots that owned
        them.
        """
        for plot in list(self.plots):
            self.remove(plot)

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

    def fitted(self, serial: Any, graph: Any) -> None:
        """A new plot had none of itself in view, so the page reframed for it.

        The sentence is `plot/actions.py`'s, which is the one a desktop window
        says in the same circumstance, and it names the curve the way the plot
        list names it. What the page decides is which of the two happened: a
        graph is framed in y alone, its abscissa being what it was sampled over.
        """
        plot = self._plot(serial)
        if plot is None:
            return
        said = actions.AUTOSCALED_Y if graph else actions.AUTOSCALED
        self.page.said(said.format(named=plot.named))

    def repolar(self, polar: bool) -> None:
        """The polar toggle moved: reread every curve it applies to.

        The reading is the view's rather than one curve's, and it is this
        side's rather than the page's: which kind a curve is drawn as decides
        what is sampled for it, and the page is told so that its own grid and
        its pointer readout follow.
        """
        self.polar = bool(polar)
        self.page.polarized(self.polar)
        self.page.lit("view.polar", self.polar)
        self.page.said(actions.POLAR_ON if self.polar else actions.POLAR_OFF)
        for plot in self.plots:
            reread = view.reread(plot.kind, self.polar)
            if reread is None:
                continue
            plot.kind = reread
            self.page.respec(plot.serial, self._spec(plot))
            self._start(plot)
        # A curve read as r = f(θ) is a curve with a parameter, so the range
        # fields and the axis names both follow the reading.
        self._axes()
        self._parameter()

    def connect(self, plot: Shown, connected: bool) -> None:
        """Join a data plot's points, or let them loose.

        Sticky, as it is on the desktop: what the menu was last told is what
        the next data plot starts out as, and the app is told so that a state
        file keeps it.
        """
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

    def remembered(self, x0: Any, x1: Any, y0: Any, y1: Any) -> None:
        """The page is about to move the view: keep where it stands.

        Called once per gesture rather than once per frame, which is where the
        desktop's windows call it too - a range signal arrives after the change
        and the thing worth keeping is what was there before it. The history is
        this side's because Back is a command and `plot/view.py` already says
        what a step through one comes to; the ranges are the page's because a
        drag must never wait on this side for one.
        """
        self._framing.remember(((float(x0), float(x1)), (float(y0), float(y1))))

    def centred(self, x: Any, y: Any) -> None:
        """A double click on the picture: that point, put in the middle of it.

        The gesture is the page's and the framing is `plot/actions.py`'s, so what
        crosses is the point that was clicked and nothing about the view: the
        history it pushes is the history Back steps through, and a view centred
        by mistake is one step from the view it replaced.
        """
        self._apply(self._framing.centred(self._where(), float(x), float(y)))

    def typed(self, name: Any, values: Any, role: Any = None) -> None:
        """A form was applied: read what was typed into it.

        `role` is which answer of the form was pressed, which for a pane with
        one form and one answer is always the same one and is read anyway: the
        seam is the same seam a solid's inspector uses, and a handler that only
        worked for a form with a single button would be two seams.

        The one form a 2D pane has is its four bounds, and they are expressions
        rather than floats - `-π` is what a person types, and a field that
        special-cased a constant or two would be a parser by stealth. So the
        text is parsed whole by `plot/forms.py`, under the grammar the last plot
        arrived with, and what the trees are worth is asked of the worker, which
        is the side that can answer it. Nothing on the page's thread evaluates
        anything, and a bound is no exception.
        """
        if str(name) != forms.RANGE.name or not _applying(role):
            return
        texts = [str(one) for one in values]
        nodes = forms.parsed(texts, self._state)
        if nodes is None:
            self.page.said(forms.EXPRESSIONS)
            return
        request = asking.Numbers(pane=self.number, nodes=nodes, context=self._context)
        self.session.sample((self.number, RANGE), lambda _report: request, self._framed)

    def ranged(self, serial: Any, low: Any, high: Any) -> None:
        """A parameter range field was left: sample that plot over what it says.

        The same bargain the four bounds make, and the same parse: a text that
        will not read is refused in words and the fields go back to the range
        the plot is actually drawn over, which the page already has. A bound
        that reads but is worth nothing finite reverts when the sampling
        answers, keeping the range the plot had.
        """
        plot = self._plot(serial)
        if plot is None:
            return
        bounds = forms.parsed((str(low), str(high)), plot.state)
        if bounds is None:
            self._parameter()
            self.page.said(
                forms.PARAMETER_EXPRESSIONS.format(parameter=plot.parameter)
            )
            return
        plot.bounds = bounds
        self._start(plot)

    def spanned(self, serial: Any, low: Any, high: Any) -> None:
        """A sampling came back with the parameter range it was taken over.

        The page says it because the numbers arrived with the arrays and were
        drawn before this side heard anything at all - the same shape a 3D pane
        reports the z a surface stood in. What is done with them is this side's:
        the range fields show what the curve is actually drawn over, which is
        also how a bound that was worth nothing announces itself, by reverting.
        """
        plot = self._plot(serial)
        if plot is None:
            return
        plot.trange = (float(low), float(high))
        plot.bounds = None
        if plot is self._ranged_plot():
            self._parameter()

    def copied(self, text: Any, trouble: Any) -> None:
        """The page has put the picture on the clipboard, or was not allowed to."""
        _copied(self.page, text, trouble)

    def exported(self, name: Any, wide: Any, tall: Any, trouble: Any) -> None:
        """The picture has left the tab as a download, or the browser refused."""
        _exported(self.page, name, wide, tall, trouble)

    def traced(self, serial: int, at: float) -> None:
        """Read one curve out where the marker now is."""
        plot = self._plot(serial)
        if plot is None:
            return
        if plot is not self._ridden:
            # Taking hold of a curve is how a plot is chosen, so the range
            # fields follow the marker onto whatever it is riding.
            self._ridden = plot
            self._parameter()
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

    # -- the controls ------------------------------------------------------

    def commands(self) -> dict[str, Callable[[Any], None]]:
        """What this pane does, by the name `plot.controls` gives each thing.

        The dispatch table, and with it the whole of what the pane offers: a
        control that is not here is left off the menu and out of the key
        ladder, so what the browser cannot yet do it does not pretend to. A
        name arrives from the page, is looked up here, and what it names is
        either a call back into the picture - the view is the page's, and a
        drag must never wait on this side - or something about the plot list,
        which is this side's own.
        """
        return {
            "range.set": lambda _value: self._ask_range(),
            "view.all": lambda _value: self._framed_all(),
            "view.home": lambda _value: self._homed(),
            "view.back": lambda _value: self._step(-1),
            "view.forward": lambda _value: self._step(1),
            "view.zoom.in": lambda _value: self.page.zoom(1 / actions.ZOOM_FACTOR),
            "view.zoom.out": lambda _value: self.page.zoom(actions.ZOOM_FACTOR),
            "trace": lambda _value: self.page.trace(),
            "grid": lambda _value: self.page.grid(),
            "legend": lambda _value: self.page.legend(),
            "image.copy": lambda _value: self.page.copy(),
            "image.export": lambda _value: self.page.export(),
            "clear": lambda _value: self.clear(),
            "view.polar": lambda _value: self.repolar(not self.polar),
            "scales.equal": lambda _value: self.page.equalize(),
            "close": lambda _value: self.close(),
            "plot.remove": lambda _value: self._drop(),
            "points.connect": lambda _value: self._reconnect(),
            "points.size": lambda value: self._resize(value),
        }

    def command(self, name: Any, value: Any = None) -> None:
        """Do the thing the page named, which is a control's own name.

        A name the table has nothing for does nothing at all: a menu the user
        left open while the pane emptied under it is the ordinary way that
        happens, and an exception raised into a click handler would be a page
        that stopped listening.
        """
        run = self.commands().get(str(name))
        if run is not None:
            run(_given(value))

    def menu(self, state: Any) -> Any:
        """The context menu of this pane, as it should currently read.

        Answered when the click happens rather than kept, because nothing on it
        is the menu's own: trace is taken hold of by a key and let go of with
        Escape, and a tick standing for what the menu last did would be wrong
        the first time anything else did it.
        """
        return _entries(
            controls.menu(self._snapshot(state), page=True, offers=self.commands())
        )

    def card(self, state: Any) -> Any:
        """The menu one legend row offers, which is about a plot and not a view."""
        return _entries(
            controls.card(self._snapshot(state), page=True, offers=self.commands())
        )

    def _snapshot(self, state: Any) -> controls.Flat:
        """This pane as the description of its controls has to read it.

        Half of it is the page's, since the page is the side that holds the
        view and everything that goes with it, and half is this side's: which
        curve a click was over arrives as a serial, and what a menu has to say
        about that curve is read off the plot the serial names.
        """
        self._pointed = pointed = self._plot(_field(state, "pointed"))
        return controls.Flat(
            tracing=bool(_field(state, "tracing")),
            grid=bool(_field(state, "grid")),
            legend=bool(_field(state, "legend")),
            polar=self.polar,
            equal=bool(_field(state, "equal")),
            pointed=None
            if pointed is None
            else controls.Pointed(
                named=pointed.named, kind=pointed.kind, connected=pointed.connected
            ),
        )

    def _drop(self) -> None:
        """The menu's `Remove`: take the plot it was raised over out for good."""
        if self._pointed is not None:
            self.remove(self._pointed)

    def _reconnect(self) -> None:
        """The menu's `Connect points`, which goes the way the entry read."""
        if self._pointed is not None:
            self.connect(self._pointed, not self._pointed.connected)

    def _resize(self, value: Any) -> None:
        """How big a data plot's points are drawn, off its own menu. Sticky too."""
        plot = self._pointed
        if plot is None or plot.kind is not PlotKind.DATA:
            return
        plot.point_size = float(value)
        self.page.respec(plot.serial, self._spec(plot))
        self.session.adjusted(point_size=plot.point_size)

    # -- framing -----------------------------------------------------------

    def _apply(self, framed: actions.Framed) -> None:
        """Do what a framing command came to, which is where a command ends.

        The arithmetic and the history are `plot/actions.py`'s; what is left is
        the two things the page has to be told - where to look and whether the
        scales are still locked - and the sentence, if there is one. A framing
        of nothing is a command that found nothing to do, and the view stays.
        """
        if framed.ranges is not None:
            (left, right), (low, high) = framed.ranges
            self.page.reframe(left, right, low, high, self._framing.equal)
        if framed.message:
            self.page.said(framed.message)

    def _step(self, direction: int) -> None:
        """Backspace and Shift-Backspace: where the view has been."""
        self._apply(self._framing.stepped(self._where(), direction))

    def _homed(self) -> None:
        """The default framing: x in [-5, 5], the origin centred, equal scales.

        The scales relock, so that Home is the framing every pane opens with and
        a released `1:1` is a departure from it like any pan or zoom.
        """
        width, height = self._size()
        self._apply(self._framing.home(self._where(), width, height))

    def _framed_all(self) -> None:
        """`View all`: frame everything drawn, and say which of the two happened.

        The framing is the page's - the rectangle each curve wants came off the
        worker with its samples, so nothing has to be scanned for it - and the
        words are `plot/actions.py`'s, as every other sentence a pane says is.
        """
        self._framing.remember(self._where())
        found = bool(self.page.autoscale())
        self.page.said(actions.FRAMED_ALL if found else actions.NOTHING_FRAMED)

    def _ask_range(self) -> None:
        """The menu's `Set range...`: the four bounds of the view, typed.

        The one place in this pane where a framing is written rather than
        gestured, and it is `plot/forms.py`'s four fields - the same four the
        desktop's dialog draws, in the same words, filled from the view as it
        stands so that a form opened to change one edge is three edges already
        right.
        """
        (left, right), (low, high) = self._where()
        self.page.ask(
            _form(forms.RANGE, self.number),
            _js([roughly(one) for one in (left, right, low, high)]),
        )

    def _framed(self, answer: Any) -> None:
        """The typed bounds are worth numbers: frame the view on them, or refuse.

        Every way of not being a framing gets its own sentence, which is the
        whole of what this is for: a form that swallowed a bound it could not
        read would be a form nobody could tell from one that worked. A worker
        that never answered is one of those ways, and what it said about itself
        is a better sentence than anything this could write about the bounds.
        """
        if isinstance(answer, str) and answer:
            self.page.said(answer)
            return
        values = tuple(answer) if isinstance(answer, (tuple, list)) else ()
        if len(values) != 4:
            self.page.said(forms.NOT_NUMBERS)
            return
        trouble = forms.framed(values)
        if trouble:
            self.page.said(trouble)
            return
        left, right, bottom, top = values
        self._apply(self._framing.typed(self._where(), ((left, right), (bottom, top))))
        self.page.said(forms.showing(left, right, bottom, top))

    # -- the axes and the parameter range ----------------------------------

    def _axes(self) -> None:
        """Name the axes after what is being plotted against, as the desktop does.

        The abscissa carries the name of the variable, which is the whole of
        what a 2D picture knows about names; the ordinate carries the label of
        the one curve there is, and nothing once there are several - a stack of
        names down the side of the canvas is what the legend is for. The kinds
        with nothing to say say nothing: a parametric pair is not a plot against
        t, and a pane holding one beside `SIN(x)` still has an x axis.
        """
        names = {plot.abscissa for plot in self.plots} - {""}
        single = self.plots[0] if len(self.plots) == 1 else None
        if single is not None and single.kind in FIELDS:
            up = single.axes[1]
        elif single is not None and single.label and single.kind in FUNCTIONS:
            up = single.label
        else:
            up = ""
        self.page.named(names.pop() if len(names) == 1 else "", up)

    def _parameter(self) -> None:
        """Say which plot the tool row's range fields are about, or that none is.

        The curve the marker is riding, since taking hold of a curve is how a
        plot is selected; otherwise the parametrized plot added last. The bounds
        go over spelled, in the function every field and tick label of either
        backend is spelled by: nothing on the page writes a number.
        """
        plot = self._ranged_plot()
        if plot is None:
            self.page.parametrized(None, "", "", "")
            return
        self.page.parametrized(
            plot.serial,
            plot.parameter,
            written(plot.trange[0]),
            written(plot.trange[1]),
        )

    def _ranged_plot(self) -> Shown | None:
        """The plot the range fields are about, or None where there is none."""
        ridden = self._ridden
        if ridden is not None and ridden in self.plots and ridden.kind in PARAMETRIZED:
            return ridden
        for plot in reversed(self.plots):
            if plot.kind in PARAMETRIZED:
                return plot
        return None

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
        if serial is None:
            return None
        for plot in self.plots:
            if plot.serial == int(serial):
                return plot
        return None

    def _spec(self, plot: Shown) -> Any:
        """How the page is to draw one plot: its look and its name, no numbers.

        `named` is what the app wrote, cut to a line. Nothing on the page ever
        renders an expression, which is the same rule that keeps the desktop's
        windows from doing it.

        Two colors rather than one, as the desktop's windows hold: a curve is
        drawn in the first on a dark pane and in the second on the white ground
        a picture leaving the pane is drawn on. Both are `plot/model.py`'s, so
        a plot copied out of a browser is the color it is copied out of a
        window in.
        """
        return _js(
            {
                "kind": str(plot.kind),
                "label": plot.label,
                "name": plot.named,
                "color": plot.color,
                "paper": plot.paper,
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
    the picture lies; the box is what `plot/view.py` pools each surface's
    measurements into, and a surface whose mesh was built to a different one is
    asked for again. A pane holding one surface therefore evaluates once for it:
    the surface says what the box is and is drawn in the box it said.
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
        #: The z the picture stands in, and the one the inspector nailed down
        #: where somebody typed it - a typed extent is an answer and is not
        #: autoscaled away by the next surface.
        self.zrange = view.DEFAULT_ZRANGE
        self._fixed: tuple[float, float] | None = None
        #: The parse state and the context a typed bound is read under: the last
        #: surface's, since the domain belongs to the pane and the worksheet that
        #: last plotted into it is its reader.
        self._state = ParseState()
        self._context = Context()
        #: Which domain edit is the latest, so the numbers a superseded edit
        #: evaluated to cannot land on top of a newer one's.
        self._edit = 0
        self._counter = 0
        self._serial = 0
        #: The surface the last menu was raised over, which is what the entries
        #: about one surface act on.
        self._pointed: Standing | None = None
        self.page = backend.solids.open(
            number,
            _controls(controls.SOLID, self.commands()),
            _strip(forms.DOMAIN),
            _form(forms.VIEW, number),
            backend.handed(
                {
                    "closed": self.dismissed,
                    "touched": self.touched,
                    "framed": self.framed,
                    "stood": self.stood,
                    "hide": self.hide,
                    "menu": self.menu,
                    "card": self.card,
                    "command": self.command,
                    "typed": self.typed,
                    "copied": self.copied,
                    "exported": self.exported,
                }
            ),
        )
        self._show()
        self._axes()
        self.page.lit("mesh", self.wired)
        # The camera a fresh pane opens on, which is `plot/actions.py`'s and not
        # the page's: a browser opening on a different view from a desktop would
        # be two programs.
        self._look(actions.CAMERA)

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
        # The grammar and the context a typed domain is read under come from
        # the worksheet that last plotted here, as a flat pane's bounds do.
        self._state, self._context = plot.state, plot.context
        self.plots.append(standing)
        self.page.add(standing.serial, self._spec(standing))
        self.retitle()
        self._axes()
        self._start(standing)
        return standing

    def remove(self, plot: Standing) -> None:
        """Take one surface out of the pane, and rebuild the box without it."""
        if plot in self.plots:
            self.plots.remove(plot)
            self.page.remove(plot.serial)
        if self._pointed is plot:
            self._pointed = None
        self.retitle()
        self._axes()
        self._rebox()

    def clear(self) -> None:
        """The Del key and the tool row's `clear`: start this picture over.

        The pane it acts on is the pane the key was pressed in, so there is
        nothing to infer and nothing to report. The axis names go with the
        surfaces that named them.
        """
        for plot in list(self.plots):
            self.remove(plot)

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

        The one place in this pane where typing changes what is computed. The
        domain fields read expressions rather than floats - `-π` and `2π` are
        answers - parsed here under the grammar the last surface arrived with;
        what the trees are worth is arithmetic, so they go to the worker and the
        domain moves when the numbers come back. The grid is a pair of counts
        and stays one. A field that will not read is put back rather than argued
        with, since the value it would take is on the screen beside it.
        """
        texts = [str(one) for one in typed]
        grid = forms.counts(texts[4:6])
        if grid is None:
            self._show()
            self.page.said(forms.GRID_NUMBERS)
            return
        bounds = forms.parsed(texts[:4], self._state)
        if bounds is None:
            self._show()
            self.page.said(forms.DOMAIN_EXPRESSIONS)
            return
        self._edit += 1
        edit = self._edit
        request = asking.Numbers(pane=self.number, nodes=bounds, context=self._context)
        self.session.sample(
            (self.number, DOMAIN),
            lambda _report: request,
            lambda answer: self._domained(edit, grid, answer),
        )

    def _domained(self, edit: int, grid: tuple[int, int], answer: Any) -> None:
        """The typed bounds are worth numbers: take them, or put the old ones back.

        A bound that would not evaluate came back as a NaN, which is one of the
        two ways `plot/forms.py` knows a typed domain not to be a rectangle; a
        worker that never answered at all says so in its own words, which are
        better than anything this could write about the bounds.
        """
        if edit != self._edit:
            return
        if isinstance(answer, str) and answer:
            self._show()
            self.page.said(answer)
            return
        values = tuple(answer) if isinstance(answer, (tuple, list)) else ()
        if len(values) != 4:
            self._show()
            self.page.said(forms.DOMAIN_EXPRESSIONS)
            return
        trouble = forms.domained(values)
        if trouble:
            self._show()
            self.page.said(trouble)
            return
        x0, x1, y0, y1 = (float(value) for value in values)
        if grid != self.grid:
            # A typed grid is sticky: the next pane opens on it. The sticky
            # value is one count per axis, so a rectangular grid hands on its
            # finer axis. The domain is not sticky - it is a framing, like a 2D
            # view - so only the grid goes back.
            self.session.adjusted(grid=max(grid))
        changed = ((x0, x1), (y0, y1), grid) != (self.xdomain, self.ydomain, self.grid)
        self.xdomain, self.ydomain, self.grid = (x0, x1), (y0, y1), grid
        self._show()
        if changed:
            self.page.said(forms.evaluating(grid))
            self.reevaluate()

    def stood(self, serial: Any, span: Any, drawn: Any) -> None:
        """A surface has been drawn: what its values measure, and the box it got.

        The page says it because the numbers came back with the arrays and were
        drawn before this side heard anything at all. What is done with them is
        this side's: the box is what `plot/view.py` pools the measurements into,
        and a mesh built to any other box is asked for again.
        """
        plot = self._plot(serial)
        if plot is None:
            return
        plot.span = _span(span)
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
        self.page.lit("mesh", self.wired)
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

    # -- the controls ------------------------------------------------------

    def commands(self) -> dict[str, Callable[[Any], None]]:
        """What this pane does, by the name `plot.controls` gives each thing.

        A 3D pane is a camera and most of its list is where to look from, which
        is the page's arithmetic over vertices that are already on the card:
        nothing here starts an evaluation, and that is the promise the whole
        picture is built on. What is not here is what this backend cannot yet
        do, and what it cannot do it does not offer.
        """
        return {
            "camera.home": lambda _value: self._look(actions.CAMERA),
            "camera.xy": lambda _value: self._face("camera.xy"),
            "camera.xz": lambda _value: self._face("camera.xz"),
            "camera.yz": lambda _value: self._face("camera.yz"),
            "camera.spin": lambda _value: self.page.spin(self._rotating()),
            "view.inspect": lambda _value: self._inspect(),
            "image.copy": lambda _value: self.page.copy(),
            "image.export": lambda _value: self.page.export(),
            "surface.remove": lambda value: self._remove_at(value),
            "clear": lambda _value: self.clear(),
            "close": lambda _value: self.close(),
            "mesh": lambda _value: self.rewire(not self.wired),
            "box": lambda _value: self.page.box(),
            "names": lambda _value: self.page.naming(),
            "legend": lambda _value: self.page.legend(),
            "surface.wire": lambda _value: self._rewire_one(),
            "plot.remove": lambda _value: self._drop(),
        }

    def command(self, name: Any, value: Any = None) -> None:
        """Do the thing the page named, which is a control's own name."""
        run = self.commands().get(str(name))
        if run is not None:
            run(_given(value))

    def menu(self, state: Any) -> Any:
        """The context menu of this pane, as it should currently read."""
        return _entries(
            controls.menu(self._snapshot(state), page=True, offers=self.commands())
        )

    def card(self, state: Any) -> Any:
        """The menu one legend row offers, which is about a surface."""
        return _entries(
            controls.card(self._snapshot(state), page=True, offers=self.commands())
        )

    def _snapshot(self, state: Any) -> controls.Solid:
        """This pane as the description of its controls has to read it.

        What the surfaces are called and how they are drawn is this side's, the
        camera and the panels are the page's, and the two are put together here
        rather than either side keeping a copy of the other's half.
        """
        self._pointed = pointed = self._plot(_field(state, "pointed"))
        return controls.Solid(
            spinning=bool(_field(state, "spinning")),
            wired=self.wired,
            boxed=bool(_field(state, "boxed")),
            named=bool(_field(state, "names")),
            legend=bool(_field(state, "legend")),
            surfaces=tuple(plot.named for plot in self.plots),
            pointed=None
            if pointed is None
            else controls.Pointed(
                named=pointed.named, kind=pointed.kind, wire=pointed.wire
            ),
        )

    def _look(self, camera: actions.Camera) -> None:
        """Put the camera where those three numbers say.

        A distance of zero is what the three presets carry and means no
        distance at all: they turn the camera and leave it standing where it
        was, since how far out the reader is looking from is not theirs to say.
        """
        self.page.look(camera.elevation, camera.azimuth, camera.distance)

    def _face(self, name: str) -> None:
        """Look straight at the plane one of the presets stands for.

        Which plane that is belongs to the control, beside the words it is
        offered under and the key that presses it.
        """
        one = controls.control(name, controls.SOLID)
        self._look(actions.facing(one.value))
        self.page.said(actions.FACING.format(plane=one.value))

    def _rotating(self) -> str:
        """What a pane says while it is turning, naming the key that stops it."""
        keys = controls.keys(controls.control("camera.spin", controls.SOLID), page=True)
        return actions.ROTATING.format(key=keys[0])

    def _remove_at(self, value: Any) -> None:
        """The `Remove` submenu: the surface standing at that place in the list."""
        at = int(value)
        if 0 <= at < len(self.plots):
            self.remove(self.plots[at])

    def _drop(self) -> None:
        """The row menu's `Remove`: take one surface out of the pane for good."""
        if self._pointed is not None:
            self.remove(self._pointed)

    def _rewire_one(self) -> None:
        """One surface between wire and solid: the legend row's own exception.

        An exception rather than a default, so it hands nothing back to the
        sticky preferences and leaves the `mesh` button - the every-surface
        control - where it was. Nothing is evaluated for it: both drawings are
        on the card already.
        """
        plot = self._pointed
        if plot is None:
            return
        plot.wire = not plot.wire
        self.page.respec(plot.serial, self._spec(plot))

    # -- the numbers behind the picture ------------------------------------

    def _inspect(self) -> None:
        """The `View...` form: the box and the camera as numbers to be typed.

        The box is this side's, so it goes over spelled; the camera is the
        page's, being a fact about where a reader is standing rather than a
        value of anything, and the page fills those three itself. Whoever owns
        the number writes it.
        """
        self.page.inspect(_js([written(value) for value in self._reading()]))

    def _reading(self) -> tuple[float, ...]:
        """The box as the inspector asks for it: a center and a length an axis."""
        spans = (self.xdomain, self.ydomain, self.zrange)
        return (
            *((low + high) / 2 for low, high in spans),
            *(high - low for low, high in spans),
        )

    def typed(self, name: Any, values: Any, role: Any = None) -> None:
        """The inspector was answered: take the numbers, or give the z back.

        Three answers and three sentences' worth of difference between them.
        `Autoscale z` hands the extent back to the data, which is what undoes a
        typed one; `Apply` is nine numbers, and nine numbers is what they must
        be - a box and a camera are lengths and angles rather than expressions,
        as they are on the desktop. Either way the fields are filled again, so
        that what the picture is now standing in is what they say.
        """
        if str(name) != forms.VIEW.name:
            return
        if str(role or "") == forms.Role.RESET:
            self.autoscale_z()
            self._showing()
            return
        if not _applying(role):
            return
        read = forms.numbers([str(one) for one in values])
        if read is None or len(read) != 9:
            self.page.said(forms.NUMBERS)
            self._showing()
            return
        spans = [
            forms.spanned(middle, width)
            for middle, width in zip(read[:3], read[3:6])
        ]
        azimuth, elevation, distance = read[6:]
        self.reframe(spans[0], spans[1], spans[2])
        self.page.look(elevation, azimuth, max(distance, 0.1))
        self._showing()

    def reframe(
        self,
        xdomain: tuple[float, float],
        ydomain: tuple[float, float],
        zrange: tuple[float, float],
    ) -> None:
        """The inspector's answer: a box typed rather than arrived at.

        A typed z extent is an opinion and outranks the autoscale from then on -
        somebody who asked for -1 to 1 in z wants to see the surface cut off at
        1, and a picture that reframed itself on the next plot would be
        answering a question nobody asked. The extent that was already on the
        screen is not such an opinion, though: applying a change of camera must
        not quietly freeze the z the data chose, so an unedited field changes
        nothing.
        """
        moved = (xdomain, ydomain) != (self.xdomain, self.ydomain)
        self.xdomain, self.ydomain = xdomain, ydomain
        self._show()
        if zrange[1] > zrange[0] and not forms.alike(zrange, self.zrange):
            self._fixed = self.zrange = zrange
        if moved:
            self.reevaluate()
        else:
            self._rebox()

    def autoscale_z(self) -> None:
        """Give the z extent back to the data, undoing a typed one."""
        self._fixed = None
        self._rebox()

    def _showing(self) -> None:
        """Fill the inspector's box fields again, wherever it is still up."""
        self.page.showing(_js([written(value) for value in self._reading()]))

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
        drawn in theirs and may move it once it has been evaluated. A typed
        extent overrules both, being an answer rather than a measurement.
        """
        if self._fixed is not None:
            return self._fixed
        others = [
            one
            for one in self.plots
            if one is not plot and one.visible and one.standing is not None
        ]
        return self.zrange if others else None

    def _rebox(self) -> None:
        """The box every surface stands in, and the meshes that are not in it.

        The pooling is `plot/view.py`'s, which is the same arithmetic the
        desktop's window arrives at its box by: what crosses from the worker is
        four numbers a surface rather than the values, and the rule for putting
        several of those together is written once and read from both sides.
        """
        asked = [
            plot.span for plot in self.plots if plot.visible and plot.span is not None
        ]
        found = (self._fixed, False) if self._fixed is not None else view.pooled(asked)
        if found is None:
            return
        zrange, clipped = found
        moved, self.zrange = zrange != self.zrange, zrange
        for plot in self.plots:
            if plot.visible and plot.standing not in (None, zrange):
                self._start(plot)
        if clipped and moved:
            self.page.said(actions.clipped(*zrange))

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
        if serial is None:
            return None
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

    def _axes(self) -> None:
        """What the three axes are called, taken from the first surface there is.

        The names are the expression's own - the two floor variables in the
        engine's canonical order, and the vertical one where an equation named
        it. A pane holding two surfaces of the same two variables is the
        ordinary case, and one holding two of different ones has to pick: the
        first surface named the axes and the others are drawn over them.
        """
        first = self.plots[0] if self.plots else None
        names = ("x", "y", "z") if first is None else (*first.axes, first.vertical)
        self.page.named(*names)

    def copied(self, text: Any, trouble: Any) -> None:
        """The page has put the picture on the clipboard, or was not allowed to."""
        _copied(self.page, text, trouble)

    def exported(self, name: Any, wide: Any, tall: Any, trouble: Any) -> None:
        """The picture has left the tab as a download, or the browser refused."""
        _exported(self.page, name, wide, tall, trouble)


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
        self._arrived(number, str(trouble or ""))

    def _arrived(self, number: Any, answer: Any) -> None:
        """One request is over, however it answered, and the queue moves on.

        A sampling answers with whatever the page could not draw, a request for
        what typed bounds are worth answers with the numbers, and neither is
        this class's business past letting go of the flight.
        """
        flight = self._flight
        if flight is None or flight[0] != int(number):
            return
        self._flight = None
        self.deliver(flight[1], answer)
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

        Which door the answer comes back through is the request's own: a picture
        goes to the page and reaches this side as an acknowledgement, and what
        typed bounds are worth comes back here, since nothing draws four floats.
        """
        method = _named(request)
        try:
            if method in asking.DRAWING:
                await self._engine.ask(number, method, (request,))
                return
            answer = await self._engine.value(method, (request,))
        except Exception as error:
            self.landed(number, said(error))
            return
        self._arrived(number, answer)


def _named(request: Any) -> str:
    """Which of the worker's plot methods a request is for."""
    if isinstance(request, asking.Sample):
        return asking.SAMPLE
    if isinstance(request, asking.Trace):
        return asking.TRACE
    if isinstance(request, asking.Grid):
        return asking.GRID
    if isinstance(request, asking.Numbers):
        return asking.NUMBERS
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


def _controls(table: tuple[controls.Control, ...], served: dict[str, Any]) -> Any:
    """The controls a pane offers, as the page needs them to render one.

    Handed over once, when the pane is made, because none of it changes: the
    keys a control answers to, the word a toolbar button carries and the
    sentence it says under the pointer are the table's, and what varies with
    the state of the window is the menu rather than the list. What is left out
    is what this backend has nothing behind, so the page cannot draw a button
    for a thing that would do nothing.
    """
    return _js(
        [
            {
                "name": one.name,
                "keys": list(controls.keys(one, page=True)),
                "bar": one.bar,
                "hint": one.hint,
                "toggle": one.kind is controls.Kind.TOGGLE,
                # The event a page hears this one through instead of a key
                # press, where there is one: the key ladder leaves such a
                # stroke alone, because cancelling it would cancel the thing
                # the browser was about to offer.
                "event": one.event,
            }
            for one in table
            if one.name in served
        ]
    )


def _form(form: forms.Form, number: int) -> Any:
    """One dialog as the page needs it to draw an overlay.

    Everything `plot/forms.py` says about it and nothing else: what is asked
    for, in what words, under what headings, with what answers at the bottom.
    The page lays it out its own way - a page has no dialog and no button box -
    but it invents no field and no word, which is what makes the overlay and the
    desktop's dialog the same form.
    """
    return _js(
        {
            "name": form.name,
            "heading": form.heading.format(number=number),
            "title": form.title.format(number=number),
            "subtitle": form.subtitle,
            "fields": [
                {"name": one.name, "label": one.label, "width": one.width}
                for one in form.fields
            ],
            "groups": [
                {
                    "title": group.title,
                    "heads": list(group.heads),
                    "rows": [
                        {"label": row.label, "fields": list(row.fields)}
                        for row in group.rows
                    ],
                }
                for group in form.groups
            ],
            "buttons": [
                {"label": one.label, "role": str(one.role), "primary": one.primary}
                for one in form.buttons
            ],
        }
    )


def _strip(strip: forms.Strip) -> Any:
    """One run of fields along a tool row, as the page needs it to build one.

    The pieces in the order they stand, a word or a field or a divider each,
    which is the whole of the layout: a strip is a row and not a form.
    """
    return _js(
        {
            "name": strip.name,
            "fields": [
                {"name": one.name, "label": one.label, "width": one.width}
                for one in strip.fields
            ],
            "pieces": [
                {
                    "word": one.word,
                    "entry": one.entry,
                    "divider": one.divider,
                    "name": one.name,
                }
                for one in strip.pieces
            ],
        }
    )


def _entries(entries: tuple[controls.Entry, ...]) -> Any:
    """One menu as it should currently read, as the page draws one.

    Everything resolved: the words, the key this platform presses, the tick,
    the greying and where the rules fall. A page that had to consult anything
    else to draw a line of this has been handed too little.
    """
    return _js(
        [
            {
                "name": entry.name,
                "label": entry.label,
                "keys": list(entry.keys),
                "checked": entry.checked,
                "enabled": entry.enabled,
                "separated": entry.separated,
                "value": entry.value,
                "items": [
                    {
                        "name": item.name,
                        "label": item.label,
                        "keys": [],
                        "checked": False,
                        "enabled": True,
                        "separated": False,
                        "value": item.value,
                        "items": [],
                    }
                    for item in entry.items
                ],
            }
            for entry in entries
        ]
    )


def _field(state: Any, name: str) -> Any:
    """One field of the snapshot the page handed over, or nothing for it.

    The snapshot is a plain object the page builds when a menu opens, and a
    field it has no answer for is a field this backend does not need: a pane
    that cannot show a thing has no state for it either.
    """
    return _given(getattr(state, name, None))


#: What `_nothing` answers where there is no page to have a null of its own: an
#: object no value crossing a seam can be, so that the comparison is false
#: rather than absent.
_NEVER = object()


def _given(value: Any) -> Any:
    """One value off the page, with its two spellings of nothing read as one.

    A page has two words for having no answer - `null` where it means none and
    `undefined` where it has no field at all - and Pyodide keeps them apart:
    `undefined` arrives as `None`, and `null` arrives as a `JsNull` that is
    falsy but is not `None`. A menu raised over no curve carries the first of
    those, so anything asked whether it was handed a plot has to be asked in
    the one word Python has for nothing rather than in either of the page's.
    """
    return None if value is None or value is _nothing() else value


def _nothing() -> Any:
    """The page's own `null`, or a sentinel where there is no page to have one.

    Reached for inside the call, as every other piece of the runtime is: this
    module is read by the tests on a desktop, where `pyodide` is not installed
    and nothing can be a JavaScript null in the first place.
    """
    try:
        from pyodide.ffi import jsnull
    except ImportError:
        return _NEVER
    return jsnull


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


def _span(values: Any) -> view.Span | None:
    """What one surface measured in z, out of the four numbers the page carried."""
    numbers = [float(value) for value in values] if values is not None else []
    return view.Span(*numbers) if len(numbers) == 4 else None


def _applying(role: Any) -> bool:
    """Whether the answer a form came back with is the one that takes the text.

    A form that describes no buttons of its own is answered with the pair every
    dialog has, and its accepting half arrives without a role at all - which is
    the same answer as `Apply` and is read as one.
    """
    return str(role or forms.Role.APPLY) == forms.Role.APPLY


def _copied(page: Any, text: Any, trouble: Any) -> None:
    """What a pane says about a picture that has gone to the clipboard, or not.

    Never silent either way. A page's clipboard is granted or refused - a tab
    Chromium has not seen the user click in is refused outright - and a copy
    that did nothing and said nothing would look exactly like a key that is not
    bound to anything. One function for both kinds of pane, because the
    sentence is about a browser rather than about a picture.
    """
    if trouble:
        page.said(CLIPBOARD_REFUSED.format(trouble=str(trouble)))
        return
    point = str(text or "")
    page.said(COPIED_POINT.format(text=point) if point else COPIED_IMAGE)


def _exported(page: Any, name: Any, wide: Any, tall: Any, trouble: Any) -> None:
    """And what it says about one that has left the tab as a download."""
    if trouble:
        page.said(DOWNLOAD_REFUSED.format(trouble=str(trouble)))
        return
    page.said(DOWNLOADED.format(name=str(name), wide=int(wide), tall=int(tall)))


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
