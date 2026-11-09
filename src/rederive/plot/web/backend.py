"""Plot windows as panes in a page, sampled by the engine worker.

This is the plot session's `Backend` in a browser. It runs on the main thread beside
the app and owns everything without numbers: the plot list, palette colors, which pane
a plot lands in, what `Describe` answers, and routing of the page's callbacks. The
view, gestures and drawing are JavaScript's (`web/plot2d.js` for 2D, `web/plot3d.js`
for 3D), and the samples come from the engine worker. Nothing here imports numpy or
holds an array.

2D and 3D panes are separate classes, as they are separate JS files. A 2D pane is
mostly about the view: framing, gestures and the trace marker. A 3D pane has none of
that; it manages the domain, the sampling grid and the box the surfaces sit in, and
rotating is just a camera move over vertices already on the page.

Panes are created and driven by calling into the page, and the page calls back
through proxied functions with strings and numbers, never arrays. A sampling request
is posted to the worker, which sends the answer straight to the page; the page reports
back only the request number and any error (a `Trouble`). So numbers cross once, from
where they're computed to where they're drawn.

Sampling uses the session's keyed job queue unchanged: a new job replaces a pending one
with the same key, and the executor runs one at a time. One at a time is right because
the worker is single-threaded, and the queue keeps a drag from posting dozens of
samplings for frames nobody will see.

Controls cross rarely. `plot/controls.py` defines each window kind's commands, labels,
keys and order. The page is given the controls this backend supports when a pane is
created, and the current menu on each right click, and it answers with a control's
name. The view stays on the page, where drags can reach it directly, and the labels
stay in Python.

Input forms work the same way, defined in `plot/forms.py`: view bounds, a curve's
parameter range, a surface's domain, and the 3D inspector's box and camera are fields,
labels and error messages that the page renders as an overlay and tool row, like the
desktop's dialogs and toolbars. Typed text comes back as text, is parsed here, and is
evaluated by the worker, so `-π` works as a bound while the main thread stays free of
math.

The one number a 3D pane computes itself is the z range of its box, using
`plot/view.py`'s rule for combining each surface's four-number span. So browser and
desktop clip surfaces at the same height.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from rederive.engine.context import Angle, Context
from rederive.plot import actions, appearance, controls, forms, protocol, resample, view
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

#: Keys of a pane's own jobs, beside the one per plot. Keying means dragging the trace
#: marker costs one reading rather than one per pixel.
TRACE = "trace"
FEATURES = "features"
#: Keys of the jobs that evaluate typed bounds for a 2D view and a 3D domain, so a form
#: applied twice before the first answer arrives costs one evaluation.
RANGE = "range"
DOMAIN = "domain"

#: Messages for copying and exporting. These are the only commands each backend
#: implements entirely on its own, so their messages live here rather than in the
#: shared tables.
COPIED_IMAGE = "Copied the plot to the clipboard"
COPIED_POINT = "Copied {text}"
CLIPBOARD_REFUSED = "The browser did not allow the clipboard: {trouble}"
#: Download message. It names the pixel size because a canvas exports a bitmap at its
#: drawn size, unlike the desktop's vector file.
DOWNLOADED = "Downloaded {name}, {wide} by {tall} pixels"
DOWNLOAD_REFUSED = "The browser refused the download: {trouble}"


class Shown(Plot):
    """A plot as a 2D pane holds it, the browser counterpart of Qt's `Drawn`.

    Samples and drawing belong to the page, so neither is here. A plot's identity (a
    worksheet and a label) is too much to send on every drag frame, so the pane gives
    each plot a serial number that both sides use.
    """

    serial: int = 0
    #: Current resample generation. The page drops answers for older generations.
    generation: int = 0
    #: Whether this plot has been sampled. Data plots are sampled only once, since
    #: per `plot/resample.py` a constant matrix is the same at every zoom.
    sampled: bool = False
    hidden: bool = False

    @property
    def visible(self) -> bool:
        return not self.hidden


class Standing(Surface):
    """A surface as a 3D pane holds it, the counterpart of `Shown`.

    Surfaces in a pane share one z range, since separate vertical scales would
    misrepresent them. So each keeps its own span and the box it was last built in; when
    the pane's box differs, the mesh is rebuilt.
    """

    serial: int = 0
    #: Current evaluation generation. The page drops answers for older generations.
    generation: int = 0
    #: The z span of this surface's values once evaluated, and the z range its mesh was
    #: last built for. `plot/view.py` pools the spans of all surfaces into the latter.
    span: view.Span | None = None
    standing: tuple[float, float] | None = None


class WebBackend:
    """Creates plot windows as panes in the page."""

    def __init__(self, page: Any, solids: Any, engine: Any) -> None:
        #: The page's 2D and 3D plotting modules, which share no drawing code, and the
        #: engine that samplings are posted to.
        self.page = page
        self.solids = solids
        self.engine = engine
        self.panes: dict[int, Any] = {}
        self._proxies: list[Any] = []
        # Visual settings (background, grid, stroke weight, tick offsets) from
        # `plot/appearance.py`, shared with the desktop so the JS modules hardcode none.
        look = _look()
        page.dress(look)
        solids.dress(look)

    def open(
        self, session: Any, kind: WindowKind, number: int, preferences: Prefer
    ) -> Any:
        """Open a pane in the page with the current preferences.

        A 2D pane uses none: it always opens with equal scales, and its polar toggle is
        deliberately not remembered. A 3D pane uses the grid size and wireframe setting,
        so a grid set in one pane carries over to the next.
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
        """Forget a pane that has closed."""
        self.panes.pop(number, None)

    def stop(self) -> None:
        """Remove every pane from the page when the app exits."""
        for pane in list(self.panes.values()):
            pane.dismiss()
        self.panes.clear()
        self.page.stop()
        self.solids.stop()
        for proxy in self._proxies:
            proxy.destroy()
        self._proxies = []

    def handed(self, table: dict[str, Callable[..., Any]]) -> Any:
        """Convert a table of callbacks into a JS object the page can call.

        The proxies are kept, since the page holds them while a pane is open and an
        unreferenced proxy could be collected.
        """
        import js
        from pyodide.ffi import create_proxy, to_js

        proxies = {name: create_proxy(call) for name, call in table.items()}
        self._proxies.extend(proxies.values())
        return to_js(proxies, dict_converter=js.Object.fromEntries)


class Pane:
    """One 2D plot window, drawn by the page and tracked here.

    Implements what the session asks of a window, plus callbacks for the page: the
    view moved, a trace or scan was requested, a point was sent home, the pane closed.
    """

    kind = WindowKind.TWO_D

    def __init__(self, session: Any, number: int, backend: WebBackend) -> None:
        self.session = session
        self.number = number
        self.backend = backend
        self.plots: list[Shown] = []
        self.current = False
        self.title = ""
        #: Whether a demonstration step is showing here, which decides whether the pane
        #: closes when the demonstration ends.
        self.demonstrating = False
        #: Whether the pane's expressions use degrees, for polar curves and readings.
        self.degrees = False
        #: Whether univariate curves are read as r = f(θ). Set by the page's toggle,
        #: which rereads every curve immediately.
        self.polar = False
        #: View history and the equal-scales lock. The current ranges live on the page,
        #: where drags change them; the history is here because Back is a command.
        #: `plot/actions.py` turns either into a framing.
        self._framing = actions.Framing()
        #: Parse state and context for typed bounds, taken from the last plot added.
        self._state = ParseState()
        self._context = Context()
        self._counter = 0
        self._serial = 0
        #: The plot under the last context menu, which per-plot entries act on. Kept
        #: here, as on the desktop, since the page only knows the serial.
        self._pointed: Shown | None = None
        #: The plot last traced, which selects whose parameter range the tool row shows.
        self._ridden: Shown | None = None
        self.page = backend.page.open(
            number,
            # Resample debounce delay, shared with the desktop via `plot/resample.py`.
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
        # Python owns polar mode, so it lights that button. The page owns the aspect lock
        # (a rubber-band zoom releases it) and lights that one itself.
        self.page.lit("view.polar", self.polar)

    # -- session interface -------------------------------------------------

    def add(self, plot: Plot) -> Shown:
        """Add a plot to the pane, replacing one with the same identity.

        Replotting the same label updates the existing curve and keeps its color, so
        the picture doesn't reshuffle.
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
        # The pane takes its angle unit, and the grammar and context for typed bounds,
        # from the latest plot's worksheet.
        self.degrees = plot.context.angle is Angle.DEGREE
        self._state, self._context = plot.state, plot.context
        # Polar mode belongs to the view, so a univariate curve added to a polar pane is
        # read as r = f(θ) from the start.
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
        """Remove one plot from the pane."""
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
        """Remove all plots, for the Del key and the tool row's `clear`.

        Axis names and range fields go with the plots that set them.
        """
        for plot in list(self.plots):
            self.remove(plot)

    def find(self, worksheet: int, label: str) -> Shown | None:
        """Return the plot with this worksheet and label, if the pane has it."""
        for plot in self.plots:
            if plot.worksheet == worksheet and plot.label == label:
                return plot
        return None

    def present(self, demonstrating: bool = False) -> None:
        """Show the pane and bring it to the front after a plot is added."""
        self.demonstrating = demonstrating
        self.page.present(demonstrating)

    def release(self, finished: bool = False) -> None:
        """End a demonstration, closing its pane if it ran to completion.

        Unlike a desktop window, a pane hides nothing that needs restoring. A finished
        gallery's pane closes so the page returns to the worksheet.
        """
        if self.demonstrating and finished:
            self.close()
        self.demonstrating = False

    def retitle(self, current: bool | None = None) -> None:
        """Title the pane by its plots, marking whether it receives the next plot."""
        if current is not None:
            self.current = current
        self.title = protocol.titled(
            self.kind,
            tuple(plot.text or plot.label for plot in self.plots),
            self.current,
        )
        self.page.retitle(self.title, self.current)

    def describe(self) -> protocol.WindowInfo:
        """Describe this pane for `Describe`."""
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
        """Close this pane and tell the session, as for any other close."""
        self.dismiss()
        self.session.closed(self.number)

    def dismiss(self) -> None:
        """Remove the pane from the page silently, since the session is ending."""
        self.page.dismiss()

    # -- page callbacks ----------------------------------------------------

    def changed(self) -> None:
        """Resample every plot after the view moved.

        The page debounces this and the sampler keys the jobs, so a drag costs one
        sampling per curve rather than one per frame.
        """
        for plot in self.plots:
            self._start(plot)

    def touched(self) -> None:
        """Make this pane the receiver of the next plot after the user clicks in it."""
        self.session.touched(self.number)

    def dismissed(self) -> None:
        """The user closed this pane."""
        self.session.closed(self.number)

    def author(self, serial: int, text: str) -> None:
        """Author a traced point into its plot's worksheet.

        The text was formatted by the worker when it read the point, so the worksheet
        gets exactly the number the status line showed.
        """
        plot = self._plot(serial)
        if plot is not None and text:
            self.session.author(plot.worksheet, str(text))

    def home(self, width: float, height: float) -> Any:
        """Return the default framing for a canvas of this shape, as four numbers.

        Computed by `plot/view.py`, so the browser opens on the same framing as the
        desktop.
        """
        (left, right), (low, high) = view.home_range(float(width), float(height))
        return _js([left, right, low, high])

    def fitted(self, serial: Any, graph: Any) -> None:
        """Report that the page reframed because a new plot wasn't in view.

        The message comes from `plot/actions.py`, as on the desktop. The page says
        which case applied: a graph is reframed in y only, since its x range is what
        it was sampled over.
        """
        plot = self._plot(serial)
        if plot is None:
            return
        said = actions.AUTOSCALED_Y if graph else actions.AUTOSCALED
        self.page.said(said.format(named=plot.named))

    def repolar(self, polar: bool) -> None:
        """Toggle polar mode and reread every curve it applies to.

        Polar mode belongs to the view and is tracked here because the curve kind
        decides what is sampled. The page is told so its grid and pointer readout
        follow.
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
        # A curve read as r = f(θ) has a parameter, so the range fields and axis names
        # change too.
        self._axes()
        self._parameter()

    def connect(self, plot: Shown, connected: bool) -> None:
        """Connect or disconnect a data plot's points.

        As on the desktop, the setting carries over to the next data plot, and the app
        is told so the state file keeps it.
        """
        plot.connected = bool(connected)
        self.page.respec(plot.serial, self._spec(plot))
        self.session.adjusted(connected=plot.connected)

    def hide(self, serial: int, hidden: bool) -> None:
        """Hide or show a curve when its legend row is clicked.

        The plot stays in the pane and in `Describe`, and returns on the next click.
        """
        plot = self._plot(serial)
        if plot is None:
            return
        plot.hidden = bool(hidden)
        self.page.respec(plot.serial, self._spec(plot))
        if not plot.hidden:
            self._start(plot)

    def remembered(self, x0: Any, x1: Any, y0: Any, y1: Any) -> None:
        """Record the current view in the history before the page moves it.

        Called once per gesture, not per frame, as on the desktop. The history is kept
        here because Back is a command handled by `plot/view.py`; the ranges stay on the
        page so a drag never waits on Python.
        """
        self._framing.remember(((float(x0), float(x1)), (float(y0), float(y1))))

    def centred(self, x: Any, y: Any) -> None:
        """Center the view on a double-clicked point.

        The framing comes from `plot/actions.py` and is pushed onto the history, so Back
        undoes an accidental recentering.
        """
        self._apply(self._framing.centred(self._where(), float(x), float(y)))

    def typed(self, name: Any, values: Any, role: Any = None) -> None:
        """Handle an applied form.

        `role` is the form button pressed. A 2D pane's single form has one button, but
        `role` is checked anyway so the interface matches the 3D inspector's.

        The form is the four view bounds, which are expressions like `-π`, not just
        floats. `plot/forms.py` parses them with the last plot's grammar, and the worker
        evaluates them, since the main thread evaluates nothing.
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
        """Resample a plot over the parameter range typed into the tool row.

        Parsed like the view bounds. Unparsable text is reported and the fields revert
        to the plot's current range. A bound that parses but has no finite value
        reverts when the sampling answers.
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
        """Record the parameter range a sampling actually used, as the page reports.

        The range arrived with the arrays at the page, which passes it on (as a 3D pane
        does with a surface's z range). The range fields then show the actual range,
        which is how a bound with no finite value reverts.
        """
        plot = self._plot(serial)
        if plot is None:
            return
        plot.trange = (float(low), float(high))
        plot.bounds = None
        if plot is self._ranged_plot():
            self._parameter()

    def copied(self, text: Any, trouble: Any) -> None:
        """Report the result of copying the picture to the clipboard."""
        _copied(self.page, text, trouble)

    def exported(self, name: Any, wide: Any, tall: Any, trouble: Any) -> None:
        """Report the result of downloading the picture."""
        _exported(self.page, name, wide, tall, trouble)

    def traced(self, serial: int, at: float) -> None:
        """Evaluate one curve at the trace marker's position."""
        plot = self._plot(serial)
        if plot is None:
            return
        if plot is not self._ridden:
            # Tracing a curve selects it, so the range fields follow the marker.
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
        """Find every notable point of one curve within the view.

        The whole list is fetched at once, so stepping through with Tab needs no
        round trips to the worker.
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

    # -- controls ----------------------------------------------------------

    def commands(self) -> dict[str, Callable[[Any], None]]:
        """Return the dispatch table, keyed by `plot.controls` names.

        Controls missing from this table are left off the menu and the key bindings, so
        the browser doesn't offer what it can't do. Entries either call back into the
        page (which owns the view) or act on the plot list here.
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
        """Run the control the page named.

        Unknown names are ignored. That happens normally when a menu stays open while
        the pane empties, and raising into a click handler would break the page.
        """
        run = self.commands().get(str(name))
        if run is not None:
            run(_given(value))

    def menu(self, state: Any) -> Any:
        """Return the pane's context menu as it should currently read.

        Built on each click rather than kept, since state like tracing can change by
        other means (a key, Escape), and a stale checkmark would be wrong.
        """
        return _entries(
            controls.menu(self._snapshot(state), page=True, offers=self.commands())
        )

    def card(self, state: Any) -> Any:
        """Return the menu for a legend row, which is about one plot."""
        return _entries(
            controls.card(self._snapshot(state), page=True, offers=self.commands())
        )

    def _snapshot(self, state: Any) -> controls.Flat:
        """Snapshot the pane state that the controls description reads.

        View state comes from the page. The plot under the click arrives as a serial,
        and its details are read from the plot here.
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
        """Remove the plot the menu was opened over."""
        if self._pointed is not None:
            self.remove(self._pointed)

    def _reconnect(self) -> None:
        """Toggle `Connect points` for the plot the menu was opened over."""
        if self._pointed is not None:
            self.connect(self._pointed, not self._pointed.connected)

    def _resize(self, value: Any) -> None:
        """Set a data plot's point size from its menu; carries over to later plots."""
        plot = self._pointed
        if plot is None or plot.kind is not PlotKind.DATA:
            return
        plot.point_size = float(value)
        self.page.respec(plot.serial, self._spec(plot))
        self.session.adjusted(point_size=plot.point_size)

    # -- framing -----------------------------------------------------------

    def _apply(self, framed: actions.Framed) -> None:
        """Apply the result of a framing command.

        `plot/actions.py` does the arithmetic and history; this tells the page the new
        ranges and scale lock, and shows any message. With no ranges, the view stays.
        """
        if framed.ranges is not None:
            (left, right), (low, high) = framed.ranges
            self.page.reframe(left, right, low, high, self._framing.equal)
        if framed.message:
            self.page.said(framed.message)

    def _step(self, direction: int) -> None:
        """Step back or forward through the view history (Backspace, Shift-Backspace)."""
        self._apply(self._framing.stepped(self._where(), direction))

    def _homed(self) -> None:
        """Restore the default framing: x in [-5, 5], origin centered, equal scales.

        The scales relock, so Home always returns to how every pane opens.
        """
        width, height = self._size()
        self._apply(self._framing.home(self._where(), width, height))

    def _framed_all(self) -> None:
        """Frame everything drawn (`View all`) and report the outcome.

        The page computes the framing from bounding rectangles that came with each
        curve's samples; the messages come from `plot/actions.py`.
        """
        self._framing.remember(self._where())
        found = bool(self.page.autoscale())
        self.page.said(actions.FRAMED_ALL if found else actions.NOTHING_FRAMED)

    def _ask_range(self) -> None:
        """Open the `Set range...` form for the four view bounds.

        Uses `plot/forms.py`'s fields, as the desktop dialog does, prefilled with the
        current view.
        """
        (left, right), (low, high) = self._where()
        self.page.ask(
            _form(forms.RANGE, self.number),
            _js([roughly(one) for one in (left, right, low, high)]),
        )

    def _framed(self, answer: Any) -> None:
        """Frame the view on the evaluated bounds, or report why not.

        Each failure gets its own message, so a bad bound is never silently ignored. A
        string answer is the worker's own error, which is reported as is.
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

    # -- axes and parameter range ------------------------------------------

    def _axes(self) -> None:
        """Label the axes, as the desktop does.

        The x axis is named after the variable. The y axis gets the label of the only
        curve, or nothing if there are several (that's what the legend is for). Kinds
        without an abscissa don't contribute, so a parametric pair beside `SIN(x)`
        still leaves an x axis.
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
        """Tell the page which plot the tool row's range fields describe, if any.

        That's the traced curve, or else the last parametrized plot added. Bounds are
        formatted here with `written`, like all fields and tick labels.
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
        """Return the plot the range fields describe, or None."""
        ridden = self._ridden
        if ridden is not None and ridden in self.plots and ridden.kind in PARAMETRIZED:
            return ridden
        for plot in reversed(self.plots):
            if plot.kind in PARAMETRIZED:
                return plot
        return None

    # -- sampling ----------------------------------------------------------

    def _start(self, plot: Shown, fresh: bool = False) -> None:
        """Request a sampling of this plot over the current view.

        `plot/resample.py`'s rule that data plots are sampled only once is
        reimplemented here, since it reads samples and this side has none.
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
        """Handle a finished sampling, as reported by the page.

        The arrays were already drawn by the page; only an error arrives here, and is
        reported to the app.
        """
        message = _trouble(answer)
        plot.trouble = message
        if message:
            self.session.trouble(self.number, plot.label, message)

    def _said(self, answer: Any) -> None:
        """Handle a finished reading or scan, reporting only failures."""
        message = _trouble(answer)
        if message:
            self.session.trouble(self.number, "", message)

    # -- bookkeeping -------------------------------------------------------

    def _plot(self, serial: Any) -> Shown | None:
        if serial is None:
            return None
        for plot in self.plots:
            if plot.serial == int(serial):
                return plot
        return None

    def _spec(self, plot: Shown) -> Any:
        """Describe how the page should draw one plot: appearance and name, no numbers.

        `named` is the app's rendering, cut to one line; the page never renders
        expressions, as desktop windows don't.

        There are two colors, as on the desktop: one for the dark pane and one for the
        white background of copied or exported pictures. Both come from
        `plot/model.py`.
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
        """Return the page's current view as x and y ranges."""
        shown = self.page.view()
        return (float(shown[0]), float(shown[1])), (float(shown[2]), float(shown[3]))

    def _size(self) -> tuple[float, float]:
        """Return the canvas size in pixels, which sets the sampling tolerance.

        Sampling works in screen space, so the canvas size decides how finely a curve
        is sampled.
        """
        shown = self.page.view()
        return (max(float(shown[4]), 100.0), max(float(shown[5]), 100.0))


class Solid:
    """One 3D plot window, drawn by the page and tracked here.

    Like `Pane` but simpler: there's no view to move, trace marker or framing. It owns
    the domain, the grid resolution, the wireframe setting and the shared box. Only
    the domain and grid trigger evaluation, as on the desktop, so rotating costs
    nothing.

    All surfaces in a pane share one box so they can be compared honestly.
    `plot/view.py` pools each surface's span into the box, and a surface whose mesh was
    built for a different box is rebuilt. A pane with one surface evaluates it once.
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
        #: Whether a demonstration step is showing here, which decides whether the pane
        #: closes when the demonstration ends.
        self.demonstrating = False
        self.xdomain = view.DEFAULT_DOMAIN
        self.ydomain = view.DEFAULT_DOMAIN
        # Start with the session's sticky grid (the last surface's), clamped like a
        # typed one so a pane never samples finer than it can draw.
        square = _grid(grid)
        self.grid = (square, square)
        #: Whether surfaces are drawn as wireframe: initially the sticky setting, then
        #: this pane's toggle.
        self.wired = bool(wire)
        #: The box's z range, and the one typed into the inspector, if any. A typed range
        #: isn't autoscaled away by the next surface.
        self.zrange = view.DEFAULT_ZRANGE
        self._fixed: tuple[float, float] | None = None
        #: Parse state and context for typed bounds, taken from the last surface added.
        self._state = ParseState()
        self._context = Context()
        #: Counter of domain edits, so a superseded edit's result can't override a newer
        #: one.
        self._edit = 0
        self._counter = 0
        self._serial = 0
        #: The surface under the last context menu, which per-surface entries act on.
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
        # Initial camera from `plot/actions.py`, so it matches the desktop.
        self._look(actions.CAMERA)

    # -- session interface -------------------------------------------------

    def add(self, plot: Plot) -> Standing:
        """Add a surface to the pane, replacing one with the same identity."""
        fields = Surface.__dataclass_fields__
        standing = Standing(**{name: getattr(plot, name) for name in fields})
        existing = self.find(plot.worksheet, plot.label)
        if existing is not None:
            # A replacement keeps the color and wireframe setting, so replotting doesn't
            # reshuffle the picture.
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
        # Typed domains are read with the grammar and context of the last surface added.
        self._state, self._context = plot.state, plot.context
        self.plots.append(standing)
        self.page.add(standing.serial, self._spec(standing))
        self.retitle()
        self._axes()
        self._start(standing)
        return standing

    def remove(self, plot: Standing) -> None:
        """Remove one surface from the pane and recompute the box without it."""
        if plot in self.plots:
            self.plots.remove(plot)
            self.page.remove(plot.serial)
        if self._pointed is plot:
            self._pointed = None
        self.retitle()
        self._axes()
        self._rebox()

    def clear(self) -> None:
        """Remove all surfaces, for the Del key and the tool row's `clear`."""
        for plot in list(self.plots):
            self.remove(plot)

    def find(self, worksheet: int, label: str) -> Standing | None:
        """Return the surface with this worksheet and label, if the pane has it."""
        for plot in self.plots:
            if plot.worksheet == worksheet and plot.label == label:
                return plot
        return None

    def present(self, demonstrating: bool = False) -> None:
        """Show the pane and bring it to the front after a plot is added."""
        self.demonstrating = demonstrating
        self.page.present(demonstrating)

    def release(self, finished: bool = False) -> None:
        """End a demonstration, closing its pane if it ran to completion.

        Unlike a desktop window, a pane hides nothing that needs restoring. A finished
        gallery's pane closes so the page returns to the worksheet.
        """
        if self.demonstrating and finished:
            self.close()
        self.demonstrating = False

    def retitle(self, current: bool | None = None) -> None:
        """Title the pane by its plots, marking whether it receives the next plot."""
        if current is not None:
            self.current = current
        self.title = protocol.titled(
            self.kind,
            tuple(plot.text or plot.label for plot in self.plots),
            self.current,
        )
        self.page.retitle(self.title, self.current)

    def describe(self) -> protocol.WindowInfo:
        """Describe this pane for `Describe`, reporting the domain as its ranges."""
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
        """Close this pane and tell the session, as for any other close."""
        self.dismiss()
        self.session.closed(self.number)

    def dismiss(self) -> None:
        """Remove the pane from the page silently, since the session is ending."""
        self.page.dismiss()

    # -- page callbacks ----------------------------------------------------

    def touched(self) -> None:
        """Make this pane the receiver of the next plot after the user clicks in it."""
        self.session.touched(self.number)

    def dismissed(self) -> None:
        """The user closed this pane."""
        self.session.closed(self.number)

    def framed(self, *typed: Any) -> None:
        """Handle a domain and grid typed into the tool row.

        This is the only typing in a 3D pane that changes what is computed. Domain fields
        are expressions such as `-π` or `2π`, parsed with the last surface's grammar and
        evaluated by the worker; the domain changes when the numbers come back. The grid
        is a pair of counts. Unreadable fields are reset to their current values.
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
        """Apply the evaluated domain bounds, or restore the old ones.

        A bound that failed to evaluate comes back as NaN, which `plot/forms.py`
        rejects. A string answer is the worker's own error and is reported as is.
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
            # A typed grid carries over to the next pane. The sticky value is one count,
            # so a rectangular grid passes on its finer axis. The domain is a framing,
            # like a 2D view, and doesn't carry over.
            self.session.adjusted(grid=max(grid))
        changed = ((x0, x1), (y0, y1), grid) != (self.xdomain, self.ydomain, self.grid)
        self.xdomain, self.ydomain, self.grid = (x0, x1), (y0, y1), grid
        self._show()
        if changed:
            self.page.said(forms.evaluating(grid))
            self.reevaluate()

    def stood(self, serial: Any, span: Any, drawn: Any) -> None:
        """Record a drawn surface's value span and the z range it was built for.

        The page passes these on from the answer it drew. They're pooled into the
        pane's box via `plot/view.py`, and any mesh built for a different box is
        rebuilt.
        """
        plot = self._plot(serial)
        if plot is None:
            return
        plot.span = _span(span)
        plot.standing = _pair(drawn)
        self._rebox()

    def rewire(self, wired: Any) -> None:
        """Switch every surface between wireframe and solid (the `mesh` control).

        The setting carries over to the next surface, here or in a new pane. No
        evaluation is needed, since the page already has both drawings.
        """
        self.wired = bool(wired)
        for plot in self.plots:
            plot.wire = self.wired
            self.page.respec(plot.serial, self._spec(plot))
        self.page.lit("mesh", self.wired)
        self.session.adjusted(wire=self.wired)

    def hide(self, serial: Any, hidden: Any) -> None:
        """Hide or show a surface when its legend row is clicked.

        The surface stays in the pane and in `Describe`. The box is recomputed without
        hidden surfaces.
        """
        plot = self._plot(serial)
        if plot is None:
            return
        plot.hidden = bool(hidden)
        self.page.respec(plot.serial, self._spec(plot))
        self._rebox()

    # -- controls ----------------------------------------------------------

    def commands(self) -> dict[str, Callable[[Any], None]]:
        """Return the dispatch table, keyed by `plot.controls` names.

        Most entries are camera moves the page handles over existing vertices, and none
        start an evaluation. Controls this backend can't do yet are omitted and not
        offered.
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
        """Run the control the page named."""
        run = self.commands().get(str(name))
        if run is not None:
            run(_given(value))

    def menu(self, state: Any) -> Any:
        """Return the pane's context menu as it should currently read."""
        return _entries(
            controls.menu(self._snapshot(state), page=True, offers=self.commands())
        )

    def card(self, state: Any) -> Any:
        """Return the menu for a legend row, which is about one surface."""
        return _entries(
            controls.card(self._snapshot(state), page=True, offers=self.commands())
        )

    def _snapshot(self, state: Any) -> controls.Solid:
        """Snapshot the pane state that the controls description reads.

        Surface names and wireframe state come from here, camera and panel state from
        the page.
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
        """Move the camera to the given elevation, azimuth and distance.

        The plane presets use distance zero, meaning keep the current distance.
        """
        self.page.look(camera.elevation, camera.azimuth, camera.distance)

    def _face(self, name: str) -> None:
        """Face the plane named by a camera preset control."""
        one = controls.control(name, controls.SOLID)
        self._look(actions.facing(one.value))
        self.page.said(actions.FACING.format(plane=one.value))

    def _rotating(self) -> str:
        """Return the message shown while spinning, naming the key that stops it."""
        keys = controls.keys(controls.control("camera.spin", controls.SOLID), page=True)
        return actions.ROTATING.format(key=keys[0])

    def _remove_at(self, value: Any) -> None:
        """Remove the surface at this index, from the `Remove` submenu."""
        at = int(value)
        if 0 <= at < len(self.plots):
            self.remove(self.plots[at])

    def _drop(self) -> None:
        """Remove the surface the legend row menu was opened over."""
        if self._pointed is not None:
            self.remove(self._pointed)

    def _rewire_one(self) -> None:
        """Toggle one surface between wireframe and solid, from its legend row.

        This is a per-surface exception: it doesn't change the sticky preference or the
        `mesh` button. No evaluation is needed.
        """
        plot = self._pointed
        if plot is None:
            return
        plot.wire = not plot.wire
        self.page.respec(plot.serial, self._spec(plot))

    # -- inspector ---------------------------------------------------------

    def _inspect(self) -> None:
        """Open the `View...` inspector for the box and camera.

        The box is owned here, so its values are formatted here; the camera is owned by
        the page, which fills in those fields itself.
        """
        self.page.inspect(_js([written(value) for value in self._reading()]))

    def _reading(self) -> tuple[float, ...]:
        """Return the box as the inspector shows it: center and length per axis."""
        spans = (self.xdomain, self.ydomain, self.zrange)
        return (
            *((low + high) / 2 for low, high in spans),
            *(high - low for low, high in spans),
        )

    def typed(self, name: Any, values: Any, role: Any = None) -> None:
        """Handle an inspector button.

        `Autoscale z` returns the z range to the data, undoing a typed one. `Apply`
        takes nine plain numbers (lengths and angles, not expressions, as on the
        desktop). Either way the fields are refilled to match the result.
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
        """Apply a box typed into the inspector.

        A typed z range overrides autoscaling from then on, so later plots don't reframe
        it. An unedited z field changes nothing, so applying a camera change doesn't
        freeze the autoscaled range.
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
        """Return the z range to autoscaling, undoing a typed one."""
        self._fixed = None
        self._rebox()

    def _showing(self) -> None:
        """Refill the inspector's box fields, if it is open."""
        self.page.showing(_js([written(value) for value in self._reading()]))

    # -- evaluation --------------------------------------------------------

    def reevaluate(self) -> None:
        """Re-evaluate every surface, after a domain or grid change."""
        for plot in self.plots:
            self._start(plot)

    def _start(self, plot: Standing) -> None:
        """Request this surface's geometry over the domain and grid.

        This is the only thing that starts an evaluation; camera moves never do.
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
        """Return the z range to place this surface in, or None to use its own.

        A surface alone in the pane determines the box. One added beside others is
        drawn in their box and may change it once evaluated. A typed range overrides
        both.
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
        """Recompute the shared box and rebuild meshes built for a different one.

        Spans are pooled with `plot/view.py`, as on the desktop.
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
        """Handle a finished evaluation, as reported by the page.

        The arrays were already drawn by the page; only an error arrives here, and is
        reported to the app.
        """
        message = _trouble(answer)
        plot.trouble = message
        if message:
            self.session.trouble(self.number, plot.label, message)

    # -- bookkeeping -------------------------------------------------------

    def _plot(self, serial: Any) -> Standing | None:
        if serial is None:
            return None
        for plot in self.plots:
            if plot.serial == int(serial):
                return plot
        return None

    def _spec(self, plot: Standing) -> Any:
        """Describe how the page should draw one surface: appearance and name."""
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
        """Fill the domain and grid fields with the pane's current values."""
        self.page.domain(
            written(self.xdomain[0]),
            written(self.xdomain[1]),
            written(self.ydomain[0]),
            written(self.ydomain[1]),
            written(self.grid[0]),
            written(self.grid[1]),
        )

    def _axes(self) -> None:
        """Name the three axes after the first surface.

        The names are the two domain variables in the engine's canonical order and the
        vertical variable if an equation named one. With surfaces of different
        variables, the first one's names win.
        """
        first = self.plots[0] if self.plots else None
        names = ("x", "y", "z") if first is None else (*first.axes, first.vertical)
        self.page.named(*names)

    def copied(self, text: Any, trouble: Any) -> None:
        """Report the result of copying the picture to the clipboard."""
        _copied(self.page, text, trouble)

    def exported(self, name: Any, wide: Any, tall: Any, trouble: Any) -> None:
        """Report the result of downloading the picture."""
        _exported(self.page, name, wide, tall, trouble)


class WorkerExecutor:
    """An `Executor` that runs the sampler's jobs on the engine worker, one at a time.

    A job is pulled only when no request is in flight. The worker is single-threaded,
    so a second request would just queue inside it, possibly putting a stale drag
    ahead of a Simplify the user just pressed.

    It doesn't receive sampling answers itself: those go to the page, which calls
    `landed` once drawn, releasing the flight and pulling the next job.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._sampler: Any = None
        #: Number and completion callback of the single request in flight.
        self._flight: tuple[int, Callable[[Any], None]] | None = None

    def serving(self, sampler: Any) -> None:
        self._sampler = sampler

    def wake(self) -> None:
        self._pull()

    def deliver(self, done: Callable[[Any], None], answer: Any) -> None:
        """Deliver an answer; with one thread, this is a direct call."""
        done(answer)

    def landed(self, number: Any, trouble: Any = "") -> None:
        """Called by the page once it has drawn (or failed to draw) an answer.

        This is what moves the queue; the arrays are already on screen by then.
        """
        self._arrived(number, str(trouble or ""))

    def _arrived(self, number: Any, answer: Any) -> None:
        """Finish the request in flight and pull the next job.

        The answer is an error string for a sampling, or numbers for typed bounds, and
        is passed on as is.
        """
        flight = self._flight
        if flight is None or flight[0] != int(number):
            return
        self._flight = None
        self.deliver(flight[1], answer)
        self._pull()

    def lost(self, reason: str) -> None:
        """Fail the request in flight after the worker died.

        Esc terminates the worker, losing its closures and the sampling in flight. The
        plot list lives here and survives; the cost is rebuilding closures on the next
        view change.
        """
        flight, self._flight = self._flight, None
        if flight is not None:
            self.deliver(flight[1], reason)

    def _pull(self) -> None:
        """Send the next job if none is in flight."""
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
        """Post one request, reporting failure to post as a `Trouble`.

        The engine handles waiting for a booting worker and replacing a dead one. A
        drawing request's answer goes to the page and comes back via `landed`; a
        `Numbers` answer comes back here directly.
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
    """Return the worker method name for a request."""
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
    """Strip a `Shown` down to a plain `Plot` for the worker, without pane bookkeeping."""
    return Plot(**{name: getattr(plot, name) for name in Plot.__dataclass_fields__})


def _solid(plot: Standing) -> Surface:
    """Strip a `Standing` down to a plain `Surface` for the worker.

    Unlike a curve, the color matters to the worker, since it computes the shaded
    per-vertex colors.
    """
    return Surface(
        **{name: getattr(plot, name) for name in Surface.__dataclass_fields__}
    )


def _look() -> Any:
    """Return the visual settings for both drawing modules, sent once at startup.

    All values come from `plot/appearance.py`, shared with the desktop. The only
    conversions are to the page's formats: `rgba(...)` strings for the canvas and
    integer colors with a separate alpha for three.js.
    """
    return _js(
        {
            "ground": appearance.BACKGROUND,
            "axis": appearance.AXIS_COLOR,
            "grid": _rgba(appearance.AXIS_COLOR, appearance.GRID_ALPHA),
            "paper": appearance.PAPER_AXIS,
            "stroke": appearance.CURVE_WIDTH,
            "slop": appearance.CLICK_SLOP_PX,
            "hit": appearance.HIT_PX,
            "region": appearance.REGION_ALPHA,
            "faded": appearance.LEGEND_FADED,
            "marker": {
                "size": appearance.MARKER_PX,
                "width": appearance.MARKER_WIDTH,
                "hairline": appearance.HAIRLINE_ALPHA,
                "chip": appearance.CHIP_OFFSET_PX,
            },
            # Trace marker steps: in pixels when the parameter is the x axis, otherwise
            # as a fraction of the parameter range.
            "nudge": {"slow": appearance.NUDGE_PX, "fast": appearance.NUDGE_FAST_PX},
            "step": {
                "slow": appearance.STEP_SHARE,
                "fast": appearance.STEP_FAST_SHARE,
            },
            "polar": {
                "rings": appearance.POLAR_RINGS,
                "spokes": appearance.POLAR_SPOKES,
                "closest": appearance.POLAR_CLOSEST_PX,
            },
            "box": _ink(appearance.BOX_COLOR),
            "tick": _ink(appearance.TICK_COLOR),
            "lamp": list(appearance.LIGHT),
            "wire": {
                "offset": list(appearance.WIRE_OFFSET),
                "width": appearance.CURVE_WIDTH,
            },
            # Offsets of tick marks, labels and axis names from the box, and how nearly
            # an axis may point at the camera before its labels are dropped.
            "out": {
                "tick": appearance.TICK_OUT,
                "label": appearance.LABEL_OUT,
                "name": appearance.NAME_OUT,
            },
            "edge": appearance.EDGE_ON,
        }
    )


def _rgba(color: str, alpha: float) -> str:
    """Convert `#rrggbb` and an alpha to a canvas `rgba(...)` string."""
    red, green, blue = (int(color[at : at + 2], 16) for at in (1, 3, 5))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _ink(rgba: tuple[int, int, int, int]) -> dict[str, Any]:
    """Convert an RGBA tuple (0-255) to an integer color and a 0-1 alpha for three.js."""
    red, green, blue, alpha = rgba
    return {"color": (red << 16) | (green << 8) | blue, "alpha": alpha / 255}


def _controls(table: tuple[controls.Control, ...], served: dict[str, Any]) -> Any:
    """Describe the pane's controls for the page, sent once when the pane is made.

    Keys, toolbar labels and hints don't change; only the menu varies with state.
    Controls without a handler in `served` are omitted.
    """
    return _js(
        [
            {
                "name": one.name,
                "keys": list(controls.keys(one, page=True)),
                "bar": one.bar,
                "hint": one.hint,
                "toggle": one.kind is controls.Kind.TOGGLE,
                # A browser event that triggers this control instead of a key, if any.
                # Key handling leaves such keystrokes alone, since cancelling them
                # would cancel the browser's own action.
                "event": one.event,
            }
            for one in table
            if one.name in served
        ]
    )


def _form(form: forms.Form, number: int) -> Any:
    """Describe a dialog for the page to draw as an overlay.

    Everything comes from `plot/forms.py`: fields, labels, headings and buttons. The
    page lays it out its own way but adds no fields or text, so it matches the
    desktop dialog.
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
    """Describe a tool row strip: its fields, and its pieces in order (each a word,
    a field or a divider)."""
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
    """Describe a menu for the page, fully resolved.

    Labels, platform keys, checkmarks, enabled state and separators are all included,
    so the page needs nothing else to draw it.
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
    """Read a field from the page's menu state snapshot, or None if it's missing.

    A missing field is one this pane doesn't support.
    """
    return _given(getattr(state, name, None))


#: Sentinel returned by `_nothing` outside a browser, equal to no value from the page.
_NEVER = object()


def _given(value: Any) -> Any:
    """Normalize a value from the page, mapping both JS `null` and `undefined` to None.

    Pyodide converts `undefined` to `None` but `null` to a falsy `JsNull`. A menu
    opened over no curve sends `null`, so it must be normalized before comparing with
    `None`.
    """
    return None if value is None or value is _nothing() else value


def _nothing() -> Any:
    """Return Pyodide's `jsnull`, or a sentinel outside a browser.

    Imported lazily, because the tests import this module on the desktop without
    `pyodide`.
    """
    try:
        from pyodide.ffi import jsnull
    except ImportError:
        return _NEVER
    return jsnull


def _grid(count: Any) -> int:
    """Clamp one grid axis count to what a pane will sample and draw."""
    return int(min(max(int(count), 2), view.MAX_GRID))


def _pair(values: Any) -> tuple[float, float] | None:
    """Return a pair of numbers from the page, or None if it sent none.

    A surface with no real values sends no z range, and the box is then set by the
    other surfaces.
    """
    numbers = [float(value) for value in values] if values is not None else []
    return (numbers[0], numbers[1]) if len(numbers) == 2 else None


def _span(values: Any) -> view.Span | None:
    """Build a surface's z `Span` from the four numbers the page sent."""
    numbers = [float(value) for value in values] if values is not None else []
    return view.Span(*numbers) if len(numbers) == 4 else None


def _applying(role: Any) -> bool:
    """Return whether a form's button role means apply.

    A form with no buttons of its own gets the standard pair, whose accept button
    sends no role; that counts as `Apply`.
    """
    return str(role or forms.Role.APPLY) == forms.Role.APPLY


def _copied(page: Any, text: Any, trouble: Any) -> None:
    """Report the result of a clipboard copy, for both kinds of pane.

    Always reports, since the browser may refuse the clipboard (Chromium does if the
    user hasn't clicked in the tab), and a silent failure would look like an unbound
    key.
    """
    if trouble:
        page.said(CLIPBOARD_REFUSED.format(trouble=str(trouble)))
        return
    point = str(text or "")
    page.said(COPIED_POINT.format(text=point) if point else COPIED_IMAGE)


def _exported(page: Any, name: Any, wide: Any, tall: Any, trouble: Any) -> None:
    """Report the result of a download, for both kinds of pane."""
    if trouble:
        page.said(DOWNLOAD_REFUSED.format(trouble=str(trouble)))
        return
    page.said(DOWNLOADED.format(name=str(name), wide=int(wide), tall=int(tall)))


def _trouble(answer: Any) -> str:
    """Extract an error message from a page or queue answer."""
    if isinstance(answer, Exception):
        return said(answer)
    return str(answer or "")


def _js(value: Any) -> Any:
    """Convert a plain Python structure to JS objects and arrays."""
    import js
    from pyodide.ffi import to_js

    return to_js(value, dict_converter=js.Object.fromEntries)
