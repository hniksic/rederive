"""The 2D plot window: a canvas, a legend, a status line, and no dialogs.

The picture is a measuring instrument rather than an illustration, and every
decision here follows from that.

Axes are drawn through the origin because that is where mathematics puts them.
pyqtgraph, being a data-chart library, draws them as edge spines; here the edge
items keep their numbers and lose their lines, and two infinite lines cross at
(0, 0) instead. A window framed away from the origin therefore has its numbers
along the edges and no axis lines in sight, which is the honest picture.

Framing is never asked about. A fresh window shows x in [-5, 5] with equal
scales, so a circle is round; the mouse does the rest, and the only place exact
bounds can be typed is the stock context menu's per-axis fields - deliberately
the only one. The one time the window reframes itself is when the alternative
is an empty picture: a curve added with finite values none of which are in
view autoscales y and says so.

Sampling is in screen space and repeats on every view change, debounced. That
is what makes zooming worth doing: a spike narrower than a pixel is not in the
data until the view asks for it, and then it is. The sampling itself happens on
the host's sampling thread - this file never evaluates anything on the Qt
thread except a single point under the trace marker, where the whole point is
that the answer is the function's and not the pixel grid's.

What this window draws is function curves. Parametric, polar, data, implicit
and region plots are the same window with other item types in it and are not
written yet; the plot list, the legend, the deletion paths and the trace
machinery are already shaped for them, which is why `Plot` carries its kind.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters  # noqa: F401  - registers the exporters Ctrl-C uses
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from rederive.engine.context import Context
from rederive.model.expr import Node
from rederive.plot import evaluate, protocol
from rederive.plot.protocol import Options, PlotKind

__all__ = ["PALETTE", "Plot", "Window2D"]

#: The window's own colors. Near-black rather than black, so that a curve in
#: black-adjacent color still reads and so that the window does not look like
#: a hole in the desktop.
BACKGROUND = "#0c0c10"
AXIS_COLOR = "#909090"
GRID_ALPHA = 0.18
TEXT_COLOR = "#d0d0d0"
STATUS_BACKGROUND = "#16161c"

#: The curve palette, cycled in this order. Bright on the dark canvas, and
#: eight of them because a ninth curve in a window is rare enough that
#: repeating a color there costs nothing.
PALETTE = (
    "#ffffff",
    "#ffff55",
    "#ff55ff",
    "#ff5555",
    "#55ffff",
    "#55ff55",
    "#ffaa00",
    "#77bbff",
)

#: The same palette for paper. Every image export swaps a white background in,
#: and a white curve on white paper is an empty picture; each color here is its
#: neighbour above, darkened to read on white.
PAPER_PALETTE = (
    "#000000",
    "#9a8000",
    "#b000b0",
    "#c00000",
    "#008080",
    "#008000",
    "#b06000",
    "#0040c0",
)

#: The default framing: x from -5 to 5, with y following from equal scales.
DEFAULT_HALF_WIDTH = 5.0

#: How long the window waits after the last view change before re-sampling.
#: Long enough that a drag is one re-sample rather than sixty, short enough
#: that letting go of the mouse feels like the end of the gesture.
RESAMPLE_DELAY_MS = 100

#: How far the pointer may move between a right-button press and its release
#: and still count as a click that opens the context menu rather than a
#: rubber-band zoom of no area.
CLICK_SLOP_PX = 4.0

#: How near a curve the pointer has to be, in pixels, to be pointing at it.
HIT_PX = 6.0

#: How far the arrow keys move the trace marker, in pixels, plain and with
#: Shift; and how far they pan the view, as a fraction of it.
NUDGE_PX = 1.0
NUDGE_FAST_PX = 10.0
PAN_SHARE = 0.25

#: How the numbers in the status line are written. Six decimals is what a
#: reading is worth from a picture: enough to paste into the algebra window
#: and see it agree, short enough to fit twice on one line.
FORMAT = "{:.6f}"


@dataclass
class Plot:
    """One entry of a window's plot list: what to draw and what it is called.

    The identity is `worksheet` and `label` together, which is what makes
    re-plotting `#3` replace its own curve while a `#3` from another algebra
    overlay is a second plot. `color` survives such a replacement, so that a
    curve keeps its color across the zoom-and-plot-again habit.
    """

    worksheet: int
    label: str
    text: str
    kind: PlotKind
    node: Node
    context: Context
    options: Options
    color: str = PALETTE[0]
    paper: str = PAPER_PALETTE[0]
    item: Any = None
    #: The lambdified closure, once the sampling thread has built one, and
    #: what went wrong if it could not.
    closure: Callable[..., np.ndarray] | None = None
    trouble: str = ""
    #: The samples now drawn, which trace and the feature scan read.
    xs: np.ndarray = field(default_factory=lambda: np.empty(0))
    ys: np.ndarray = field(default_factory=lambda: np.empty(0))
    #: Which re-sample these samples came from. A job whose generation has
    #: moved on is a job whose answer is about a view that is gone.
    generation: int = 0

    @property
    def visible(self) -> bool:
        return self.item is not None and self.item.isVisible()

    @property
    def named(self) -> str:
        """How the legend and the status line name this plot."""
        return f"{self.label}  {self.text}" if self.label else self.text

    @property
    def variable(self) -> str:
        """The name of the abscissa, `x` where the expression named none."""
        return self.options.variables[0] if self.options.variables else "x"


class Canvas(pg.ViewBox):
    """The view box, with the mouse vocabulary this program wants.

    Three departures from stock. The rubber band lives on the *right* button
    rather than on the left, because left-drag is panning and panning is what
    a hand reaches for first; a right press that does not move is still a
    click, and still opens the context menu. The wheel takes Ctrl and Shift to
    mean one axis. And every gesture that changes the range pushes the range it
    changed onto the window's own history first, so Backspace steps back
    through where the view has been rather than through pyqtgraph's rectangle
    stack, which only knows about rubber bands.
    """

    def __init__(self, window: Window2D) -> None:
        super().__init__()
        self._window = window

    def mouseDragEvent(self, ev: Any, axis: int | None = None) -> None:
        if ev.button() != QtCore.Qt.MouseButton.RightButton or axis is not None:
            if ev.isStart():
                self._window.remember()
            super().mouseDragEvent(ev, axis=axis)
            return
        ev.accept()
        start = ev.buttonDownPos(ev.button())
        if not ev.isFinish():
            self.updateScaleBox(start, ev.pos())
            return
        self.rbScaleBox.hide()
        moved = pg.Point(ev.pos()) - pg.Point(start)
        if moved.length() < CLICK_SLOP_PX:
            self.raiseContextMenu(ev)
            return
        rectangle = QtCore.QRectF(pg.Point(start), pg.Point(ev.pos()))
        self._window.remember()
        self.showAxRect(self.childGroup.mapRectFromParent(rectangle))

    def wheelEvent(self, ev: Any, axis: int | None = None) -> None:
        modifiers = ev.modifiers()
        if modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
            axis = 0
        elif modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
            axis = 1
        self._window.remember()
        super().wheelEvent(ev, axis)

    def mouseClickEvent(self, ev: Any) -> None:
        """A click on the canvas: center on it, or take hold of a curve."""
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            super().mouseClickEvent(ev)
            return
        point = self.mapToView(ev.pos())
        if ev.double():
            ev.accept()
            self._window.center_on(point)
            return
        if self._window.clicked(point):
            ev.accept()
            return
        super().mouseClickEvent(ev)

    def getMenu(self, ev: Any) -> Any:
        """The context menu, told what the click was pointing at first."""
        self._window.prepare_menu(self.mapToView(ev.pos()))
        return self.menu


class Legend(pg.LegendItem):
    """The stock legend with the whole entry as the click target.

    pyqtgraph puts hide/show on the color swatch alone, which is a twenty-pixel
    box beside a name that reads like a button and is not one. Here a click
    anywhere on the row toggles the curve, and a right-click anywhere on it
    offers to remove the plot - the same deletion the Plot Delete submenu
    drives from the algebra window.
    """

    #: The row the pointer is over, by index into the plot list, and what a
    #: click on it is about.
    clicked = QtCore.Signal(int, object)

    def hoverEvent(self, ev: Any) -> None:
        """Claim both buttons, so a click anywhere on a row comes here.

        The stock legend claims drags alone, which leaves clicking to the
        twenty-pixel swatch; claiming clicks as well is what widens the target
        to the name beside it.
        """
        ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton)
        ev.acceptClicks(QtCore.Qt.MouseButton.LeftButton)
        ev.acceptClicks(QtCore.Qt.MouseButton.RightButton)

    def mouseClickEvent(self, ev: Any) -> None:
        row = self._row_at(ev.pos())
        if row is None:
            ev.ignore()
            return
        ev.accept()
        self.clicked.emit(row, ev.button())

    def _row_at(self, pos: Any) -> int | None:
        """Which entry the point is in, or None where it is in none of them."""
        for index, (sample, label) in enumerate(self.items):
            for item in (sample, label):
                if item.mapRectToParent(item.boundingRect()).contains(pos):
                    return index
        return None


class Window2D(QtWidgets.QMainWindow):
    """One top-level 2D plot window and everything that happens inside it."""

    def __init__(self, number: int, host: Any) -> None:
        super().__init__()
        self.number = number
        self.kind = protocol.WindowKind.TWO_D
        self.host = host
        self.plots: list[Plot] = []
        self.polar = False
        self.current = False
        #: Ranges the view has been at, and where in that list it stands.
        #: Every gesture pushes before it changes anything, so stepping back is
        #: stepping to the range the last gesture left.
        self._history: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._at = 0
        self._counter = 0
        #: Which curve trace is riding, or None while trace is off.
        self._tracing: int | None = None
        self._trace_x = 0.0
        self._message = ""
        #: The paper colors, while an export dialog is holding them on.
        self._paper: _on_paper | None = None
        self._grid = True
        #: Whether the view is still the default framing, nobody having moved
        #: it. Every gesture that changes the range clears it, since the
        #: default framing is a thing to be given up rather than returned to.
        self._default = True
        self._build()
        self.home()

    # -- construction ------------------------------------------------------

    def _build(self) -> None:
        self.canvas = Canvas(self)
        self.plot = pg.PlotWidget(viewBox=self.canvas, background=BACKGROUND)
        self.item = self.plot.getPlotItem()
        self.item.showGrid(x=True, y=True, alpha=GRID_ALPHA)
        self._strip_spines()
        self._origin_axes()
        self._prune_menu()
        self._extend_menu()
        self.legend = Legend(offset=(12, 12), labelTextSize="9pt")
        self.legend.setParentItem(self.item.vb)
        self.item.legend = self.legend
        self.legend.clicked.connect(self._legend_clicked)
        self.marker = pg.ScatterPlotItem(
            size=11, symbol="s", pen=pg.mkPen(TEXT_COLOR), brush=None
        )
        self.marker.setZValue(20)
        self.marker.setVisible(False)
        self.item.addItem(self.marker, ignoreBounds=True)
        self.setCentralWidget(self._laid_out())
        self.canvas.setAspectLocked(True, ratio=1.0)
        self.canvas.disableAutoRange()
        self.canvas.sigRangeChanged.connect(self._ranged)
        self.canvas.sigResized.connect(self._resized)
        self.item.scene().sigMouseMoved.connect(self._moved)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(RESAMPLE_DELAY_MS)
        self._timer.timeout.connect(self.resample)
        self.plot.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.resize(760, 560)

    def _laid_out(self) -> QtWidgets.QWidget:
        """The canvas over a one-line status bar, and a toolbar over both.

        The toolbar holds the toggles that belong to a window rather than to a
        plot - equal scales now, the polar reading later - because they are the
        two settings a picture is read differently under, and a toggle that
        shows its state is worth more than a menu item that has to be opened
        to be read.
        """
        holder = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        bar = QtWidgets.QToolBar()
        bar.setMovable(False)
        self.equal = QtGui.QAction("1:1", self)
        self.equal.setCheckable(True)
        self.equal.setChecked(True)
        self.equal.setToolTip("Equal scales on both axes")
        self.equal.triggered.connect(self._equal_scales)
        bar.addAction(self.equal)
        layout.addWidget(bar)
        layout.addWidget(self.plot, 1)
        self.status = QtWidgets.QLabel("")
        self.readout = QtWidgets.QLabel("")
        line = QtWidgets.QWidget()
        line.setStyleSheet(f"background: {STATUS_BACKGROUND}; color: {TEXT_COLOR};")
        across = QtWidgets.QHBoxLayout(line)
        across.setContentsMargins(8, 2, 8, 2)
        across.addWidget(self.status, 1)
        across.addWidget(self.readout, 0)
        layout.addWidget(line)
        return holder

    def _strip_spines(self) -> None:
        """Keep the tick numbers along the edges, drop the lines they sit on.

        The axis lines of this window go through the origin, so an edge spine
        would be a second pair of axes drawn where no axes are. pyqtgraph draws
        the spine with the axis pen and the ticks - which are the grid lines,
        the grid being ticks drawn the width of the canvas - with the tick pen,
        so making the first transparent takes the spine and leaves the grid.
        """
        for edge in ("bottom", "left"):
            axis = self.item.getAxis(edge)
            axis.setPen(pg.mkPen(color=(0, 0, 0, 0)))
            axis.setTickPen(pg.mkPen(AXIS_COLOR))
            axis.setTextPen(pg.mkPen(TEXT_COLOR))

    def _origin_axes(self) -> None:
        """The two lines through (0, 0) that a mathematician calls the axes.

        They are infinite lines rather than drawn segments, so they sit at the
        edge of the canvas when the origin is off it - which is the picture a
        reader wants, with the edge numbers saying where the view actually is.
        """
        pen = pg.mkPen(color=AXIS_COLOR, width=1)
        self.axes = (
            pg.InfiniteLine(pos=0, angle=90, pen=pen),
            pg.InfiniteLine(pos=0, angle=0, pen=pen),
        )
        for line in self.axes:
            line.setZValue(-10)
            self.item.addItem(line, ignoreBounds=True)

    def _prune_menu(self) -> None:
        """Keep the stock menu, less the entries that would be wrong answers.

        `Transforms` offers an FFT, a derivative and a phase map of the
        *sampled points*. In a computer algebra system a derivative comes from
        Calculus and is then plotted; a finite difference of screen samples
        would be a wrong answer wearing the right name. `Average` and `Points`
        are about data series and this window draws functions.
        """
        for name in ("Transforms", "Average", "Points"):
            self.item.setContextMenuActionVisible(name, False)

    def _extend_menu(self) -> None:
        """Our own entries, appended to the stock ViewBox menu.

        `Remove` is about whatever the right-click was pointing at, so its text
        is written when the menu opens and it hides itself when the click was
        pointing at nothing.
        """
        menu = self.canvas.menu
        menu.addSeparator()
        self._remove_action = menu.addAction("Remove", self._remove_pointed)
        self._pointed: Plot | None = None

    # -- the plot list -----------------------------------------------------

    def add(self, plot: Plot) -> None:
        """Put a plot in the window, replacing one with the same identity.

        Replacing rather than adding is what the zoom-and-plot-again habit
        needs: the same label plotted twice is one curve, and it keeps the
        color it had so that the picture does not reshuffle under the reader.
        """
        existing = self.find(plot.worksheet, plot.label)
        if existing is not None:
            plot.color, plot.paper = existing.color, existing.paper
            self.remove(existing)
        else:
            index = self._counter % len(PALETTE)
            plot.color, plot.paper = PALETTE[index], PAPER_PALETTE[index]
            self._counter += 1
        plot.item = pg.PlotDataItem(
            np.empty(0),
            np.empty(0),
            pen=pg.mkPen(plot.color, width=1),
            connect="finite",
            antialias=True,
        )
        plot.item.setCurveClickable(True, width=int(HIT_PX))
        # The stroke takes the click before the view box sees it, which is the
        # stock hit test doing exactly what is wanted: clicking a curve is how
        # trace mode takes hold of that curve rather than of the nearest one.
        plot.item.sigClicked.connect(
            lambda _item, ev, plot=plot: self._curve_clicked(plot, ev)
        )
        self.item.addItem(plot.item)
        self.plots.append(plot)
        self._relabel()
        self._axis_names()
        self._start(plot, fresh=True)

    def find(self, worksheet: int, label: str) -> Plot | None:
        """The plot a worksheet and a label name, if this window has it."""
        for plot in self.plots:
            if plot.worksheet == worksheet and plot.label == label:
                return plot
        return None

    def remove(self, plot: Plot) -> None:
        """Take one plot out of the window, legend entry and all."""
        if plot.item is not None:
            self.item.removeItem(plot.item)
        if plot in self.plots:
            self.plots.remove(plot)
        self._relabel()
        if self._tracing is not None and self._tracing >= len(self.plots):
            self._trace_off()

    def take(self, which: protocol.Which, worksheet: int = 0, label: str = "") -> int:
        """The Delete submenu, applied to this window's plot list."""
        if which is protocol.Which.ONE:
            plot = self.find(worksheet, label)
            going = [] if plot is None else [plot]
        elif not self.plots:
            going = []
        elif which is protocol.Which.ALL:
            going = list(self.plots)
        elif which is protocol.Which.FIRST:
            going = self.plots[:1]
        elif which is protocol.Which.LAST:
            going = self.plots[-1:]
        else:
            going = self.plots[:-1]
        for plot in going:
            self.remove(plot)
        return len(going)

    def _relabel(self) -> None:
        """Build the legend again, which is how a removal leaves it right."""
        self.legend.clear()
        for plot in self.plots:
            color = html.escape(plot.color)
            name = html.escape(plot.named).replace(" ", "&nbsp;")
            self.legend.addItem(plot.item, f'<span style="color: {color}">{name}</span>')

    def _axis_names(self) -> None:
        """Label the axes with the variable being plotted against.

        The abscissa carries the name of the variable, which is the whole of
        what a 2D window knows about names; the ordinate carries the label of
        the one curve there is, and nothing once there are several - a stack of
        names down the side of the canvas is what the legend is for.
        """
        names = {plot.variable for plot in self.plots}
        self.item.setLabel("bottom", names.pop() if len(names) == 1 else "")
        if len(self.plots) == 1 and self.plots[0].label:
            self.item.setLabel("left", self.plots[0].label)
        else:
            self.item.setLabel("left", "")

    # -- sampling ----------------------------------------------------------

    def _start(self, plot: Plot, fresh: bool = False) -> None:
        """Ask the sampling thread for this curve over the visible range.

        Everything expensive is on the other side of this call: the conversion,
        the lambdify and the sampling all happen on the host's sampling thread,
        and what comes back is two arrays. A window whose curves are all slow
        is a window that still pans, zooms and closes while they arrive.
        """
        if plot.item is None or not plot.visible:
            return
        plot.generation += 1
        generation = plot.generation
        (left, right), (low, high) = self.canvas.viewRange()
        size = (max(self.plot.width(), 100), max(self.plot.height(), 100))
        node, context, variables = plot.node, plot.context, plot.options.variables
        closure = plot.closure

        def work(report: Callable[..., None]) -> Any:
            made = closure
            if made is None:
                made = evaluate.closure(node, context, variables or ("x",))
            xs, ys = evaluate.sample_adaptive(
                made, (left, right), (low, high), size, report=report
            )
            return made, xs, ys

        self.host.sample(
            (self.number, id(plot)),
            work,
            lambda answer: self._sampled(plot, generation, fresh, answer),
            lambda xs, ys: self._outlined(plot, generation, xs, ys),
        )

    def _outlined(
        self, plot: Plot, generation: int, xs: np.ndarray, ys: np.ndarray
    ) -> None:
        """The uniform pass, drawn while the refinement is still running."""
        if plot.generation != generation or plot.item is None:
            return
        plot.item.setData(xs, ys, connect="finite")

    def _sampled(self, plot: Plot, generation: int, fresh: bool, answer: Any) -> None:
        """The samples are in: draw them, and say so if there is nothing to draw."""
        if plot.generation != generation or plot.item is None:
            return
        if isinstance(answer, Exception):
            plot.trouble = str(answer)
            self.host.trouble(self.number, plot.label, str(answer))
            self.say(f"{plot.label}: {answer}")
            return
        plot.closure, plot.xs, plot.ys = answer
        plot.item.setData(plot.xs, plot.ys, connect="finite")
        if not np.isfinite(plot.ys).any():
            left, right = self.canvas.viewRange()[0]
            self.say(
                f"{plot.label}: no real values for {_short(left)} ≤ {plot.variable}"
                f" ≤ {_short(right)} - try A to autoscale"
            )
        elif fresh:
            self._frame_new(plot)
        if self._tracing is not None:
            self._trace_to(self._trace_x)

    def _frame_new(self, plot: Plot) -> None:
        """Autoscale for a new curve that has values but none of them in view.

        Exactly when the alternative is an empty picture, and never otherwise:
        a window whose framing moved under every added curve would be a window
        nobody could compare two curves in. The aspect lock goes when it
        happens, since a y range chosen to fit is not a y range equal scales
        would have given.
        """
        low, high = self.canvas.viewRange()[1]
        inside = plot.ys[np.isfinite(plot.ys)]
        if not inside.size or ((inside >= low) & (inside <= high)).any():
            return
        self._unlock()
        self._default = False
        bottom, top = float(np.min(inside)), float(np.max(inside))
        self.canvas.setYRange(bottom, top, padding=0.1)
        self.say(f"{plot.label}: y autoscaled to fit")

    def resample(self) -> None:
        """Re-sample every visible curve for the range now shown."""
        for plot in self.plots:
            self._start(plot)

    def _ranged(self, *_: Any) -> None:
        """A view change: schedule a re-sample, and put everything where it goes."""
        self._timer.start()
        self._place_axes()
        if self._tracing is not None:
            self._trace_to(self._trace_x)

    def _place_axes(self) -> None:
        """Keep the axis lines at the origin, or at the edge nearest to it.

        A view framed away from the origin would otherwise have no axis lines
        at all, and a picture with no reference line in it is hard to read
        against the numbers along its edges. Clamped a pixel inside the canvas,
        so that the line is drawn rather than falling on the boundary.
        """
        (left, right), (low, high) = self.canvas.viewRange()
        across, up = self.canvas.viewPixelSize()
        self.axes[0].setPos(min(max(0.0, left + across), right - across))
        self.axes[1].setPos(min(max(0.0, low + up), high - up))

    # -- framing -----------------------------------------------------------

    def home(self) -> None:
        """The default framing: x in [-5, 5], equal scales, the origin centred.

        Both ranges are worked out here rather than left to the aspect lock,
        because the lock is free to satisfy itself by widening either axis, and
        which one it picks is not something the framing should depend on. The
        ordinate is therefore the abscissa scaled by the shape of the canvas,
        which is what equal scales means, and the origin is in the middle of it.
        """
        self.remember()
        self.equal.setChecked(True)
        self.canvas.setAspectLocked(True, ratio=1.0)
        width = max(self.canvas.width(), 1.0)
        height = max(self.canvas.height(), 1.0)
        half = DEFAULT_HALF_WIDTH * height / width
        self.canvas.setRange(
            xRange=(-DEFAULT_HALF_WIDTH, DEFAULT_HALF_WIDTH),
            yRange=(-half, half),
            padding=0,
        )
        self._default = True

    def autoscale(self) -> None:
        """Frame every visible curve, which is what `A` and `View All` do."""
        self.remember()
        self._unlock()
        xs = [plot.xs[np.isfinite(plot.ys)] for plot in self.plots if plot.visible]
        ys = [plot.ys[np.isfinite(plot.ys)] for plot in self.plots if plot.visible]
        finite = [array for array in ys if array.size]
        if not finite:
            self.canvas.autoRange()
            return
        low = min(float(np.min(array)) for array in finite)
        high = max(float(np.max(array)) for array in finite)
        left = min(float(np.min(array)) for array in xs if array.size)
        right = max(float(np.max(array)) for array in xs if array.size)
        self.canvas.setRange(
            xRange=(left, right), yRange=(low, high), padding=0.05
        )

    def _equal_scales(self, checked: bool) -> None:
        """The `1:1` toolbar toggle, which relocks or releases equal scales."""
        self.canvas.setAspectLocked(checked, ratio=1.0)

    def _unlock(self) -> None:
        """Release equal scales, for a framing that is about fitting."""
        self.equal.setChecked(False)
        self.canvas.setAspectLocked(False)

    def center_on(self, point: Any) -> None:
        """Double-click: put the clicked point in the middle of the canvas."""
        self.remember()
        (left, right), (low, high) = self.canvas.viewRange()
        width, height = (right - left) / 2, (high - low) / 2
        self.canvas.setRange(
            xRange=(point.x() - width, point.x() + width),
            yRange=(point.y() - height, point.y() + height),
            padding=0,
        )

    def zoom(self, factor: float) -> None:
        """`+` and `-`: a factor of two about the middle of the view."""
        self.remember()
        self.canvas.scaleBy((factor, factor))

    def pan(self, dx: float, dy: float) -> None:
        """The arrow keys, a quarter of the window at a time."""
        self.remember()
        (left, right), (low, high) = self.canvas.viewRange()
        self.canvas.translateBy(x=(right - left) * dx, y=(high - low) * dy)

    # -- view history ------------------------------------------------------

    def remember(self) -> None:
        """Push the range as it is now, before something changes it.

        Called by every gesture rather than by a range signal, because a signal
        arrives after the change and the thing worth keeping is what was there
        before it. Stepping back and then somewhere new throws the forward half
        away, which is what every history in every program does.
        """
        try:
            ranges = self.canvas.viewRange()
        except Exception:
            return
        current = (tuple(ranges[0]), tuple(ranges[1]))
        self._default = False
        del self._history[self._at :]
        if not self._history or self._history[-1] != current:
            self._history.append(current)  # type: ignore[arg-type]
        self._at = len(self._history)

    def step_history(self, direction: int) -> None:
        """Backspace and Shift-Backspace: where the view has been."""
        if direction < 0 and self._at == len(self._history):
            # Stepping back for the first time has to keep where we are, or
            # there would be nothing to step forward to.
            self.remember()
            self._at = len(self._history) - 1
        at = self._at + direction
        if not 0 <= at < len(self._history):
            return
        self._at = at
        xrange, yrange = self._history[at]
        self._unlock()
        self.canvas.setRange(xRange=xrange, yRange=yrange, padding=0)

    # -- the pointer -------------------------------------------------------

    def _moved(self, position: Any) -> None:
        """The pointer moved over the scene: the readout, and trace if it is on."""
        if not self.item.sceneBoundingRect().contains(position):
            return
        point = self.canvas.mapSceneToView(position)
        self.readout.setText(f"x: {_number(point.x())}   y: {_number(point.y())}")
        if self._tracing is not None:
            self._trace_to(point.x())

    def clicked(self, point: Any) -> bool:
        """A left click on the canvas. True where a curve took it.

        Clicking within a few pixels of a curve is how trace mode is entered
        with the mouse, and how the active curve is changed while it is on.
        """
        plot = self.at(point)
        if plot is None:
            return False
        self._tracing = self.plots.index(plot)
        self._trace_to(point.x())
        return True

    def _curve_clicked(self, plot: Plot, ev: Any) -> None:
        """A click that landed on a curve's own stroke: trace that curve."""
        if plot not in self.plots:
            return
        self._tracing = self.plots.index(plot)
        self._trace_to(self.canvas.mapSceneToView(ev.scenePos()).x())

    def at(self, point: Any) -> Plot | None:
        """The visible curve nearest the point, if one is near enough.

        Measured in pixels against the samples that are drawn, which is the
        same thing the eye is doing: a curve is where its stroke is.
        """
        pixel = self.canvas.viewPixelSize()
        if not pixel[0] or not pixel[1]:
            return None
        best: tuple[float, Plot] | None = None
        for plot in self.plots:
            if not plot.visible or not plot.xs.size:
                continue
            value = self._value_at(plot, point.x())
            if value is None:
                continue
            away = abs(value - point.y()) / pixel[1]
            if away <= HIT_PX and (best is None or away < best[0]):
                best = (away, plot)
        return None if best is None else best[1]

    def _value_at(self, plot: Plot, x: float) -> float | None:
        """What the curve is worth at `x`, evaluated rather than read off.

        The closure is what a plot is for: a reading is the function's own
        value at the exact abscissa, to full precision, and never the height of
        the nearest pixel. Where there is no closure yet the samples answer.
        """
        if plot.closure is not None:
            value = float(np.asarray(plot.closure(np.array([x])))[0])
            return None if not np.isfinite(value) else value
        if not plot.xs.size:
            return None
        index = int(np.argmin(np.abs(plot.xs - x)))
        value = float(plot.ys[index])
        return None if not np.isfinite(value) else value

    # -- trace -------------------------------------------------------------

    def trace(self) -> None:
        """`T` and `F3`: take hold of a curve, or let go of one."""
        if self._tracing is not None:
            self._trace_off()
            return
        if not any(plot.visible for plot in self.plots):
            return
        self._tracing = next(
            index for index, plot in enumerate(self.plots) if plot.visible
        )
        (left, right) = self.canvas.viewRange()[0]
        self._trace_to((left + right) / 2)

    def _trace_off(self) -> None:
        self._tracing = None
        self.marker.setVisible(False)
        self.say("")

    def _trace_step(self, pixels: float) -> None:
        """Left and right while tracing: one pixel, or ten with Shift."""
        self._trace_to(self._trace_x + pixels * self.canvas.viewPixelSize()[0])

    def _trace_curve(self, step: int) -> None:
        """Up and down while tracing: the next curve of the plot list.

        Hidden curves are skipped, as are the kinds with no parametrized curve
        to ride - a region has no `f(x)` a marker could sit on.
        """
        rideable = [
            index
            for index, plot in enumerate(self.plots)
            if plot.visible and plot.kind in TRACEABLE
        ]
        if not rideable or self._tracing is None:
            return
        at = self._tracing
        here = rideable.index(at) if at in rideable else 0
        self._tracing = rideable[(here + step) % len(rideable)]
        self._trace_to(self._trace_x)

    def _trace_to(self, x: float) -> None:
        """Put the marker at this abscissa on the active curve, and say so."""
        if self._tracing is None or self._tracing >= len(self.plots):
            return
        plot = self.plots[self._tracing]
        self._trace_x = x
        value = self._value_at(plot, x)
        if value is None:
            self.marker.setVisible(False)
            self.say(f"Tracing {plot.named}: not real and finite at {_number(x)}")
            return
        self.marker.setData([x], [value])
        self.marker.setPen(pg.mkPen(plot.color))
        self.marker.setVisible(True)
        self.say(
            f"Tracing {plot.named}   {plot.variable} = {_number(x)}"
            f"   y = {_number(value)}"
        )

    @property
    def traced(self) -> tuple[float, float] | None:
        """The point under the marker, for the key that copies it."""
        if self._tracing is None or self._tracing >= len(self.plots):
            return None
        plot = self.plots[self._tracing]
        value = self._value_at(plot, self._trace_x)
        return None if value is None else (self._trace_x, value)

    # -- the legend and the context menu ------------------------------------

    def _legend_clicked(self, row: int, button: Any) -> None:
        """A click on a legend entry: hide and show, or offer to remove."""
        if not 0 <= row < len(self.plots):
            return
        plot = self.plots[row]
        if button == QtCore.Qt.MouseButton.RightButton:
            self._pointed = plot
            self._offer_removal(QtGui.QCursor.pos())
            return
        if plot.item is None:
            return
        plot.item.setVisible(not plot.item.isVisible())
        self.legend.items[row][0].update()
        if plot.visible:
            self._start(plot)
        elif self._tracing == row:
            self._trace_off()

    def _offer_removal(self, at: Any) -> None:
        """A menu of one, where a right-click was about a plot and not a view."""
        menu = QtWidgets.QMenu(self)
        menu.addAction(f"Remove {self._pointed.named}", self._remove_pointed)
        menu.exec(at)

    def prepare_menu(self, point: Any) -> None:
        """Say what a right-click on the canvas was pointing at, if anything."""
        self._pointed = self.at(point)
        self._remove_action.setVisible(self._pointed is not None)
        if self._pointed is not None:
            self._remove_action.setText(f"Remove {self._pointed.named}")

    def _remove_pointed(self) -> None:
        if self._pointed is not None:
            self.remove(self._pointed)
            self._pointed = None

    # -- export ------------------------------------------------------------

    def copy_image(self) -> None:
        """Ctrl-C: the picture on the clipboard, on paper colors.

        The same reflex as Ctrl-C over a highlighted expression, applied to the
        picture, so a plot lands in a document the way an expression lands in a
        message. While tracing it copies the traced point instead, which is the
        one reading a plot produces that the algebra window can take back.
        """
        point = self.traced
        if point is not None:
            text = f"[{_number(point[0])}, {_number(point[1])}]"
            QtWidgets.QApplication.clipboard().setText(text)
            self.say(f"Copied {text}")
            return
        with _on_paper(self):
            image = pg.exporters.ImageExporter(self.item).export(toBytes=True)
        QtWidgets.QApplication.clipboard().setImage(image)
        self.say("Copied the plot to the clipboard")

    def export(self) -> None:
        """Ctrl-S and the context menu's `Export...`: the stock export dialog.

        PNG, SVG, a CSV of the sampled points and a matplotlib window, all of
        them pyqtgraph's. The paper colors go on while the dialog is up rather
        than around the export itself, because the dialog is modeless and the
        export happens inside it: what the preview shows is then what the file
        will hold.
        """
        scene = self.item.scene()
        scene.showExportDialog()
        if self._paper is None:
            self._paper = _on_paper(self)
            self._paper.__enter__()
            scene.exportDialog.installEventFilter(self)

    def eventFilter(self, watched: Any, ev: Any) -> bool:
        """Put the dark theme back when the export dialog goes away."""
        gone = (QtCore.QEvent.Type.Hide, QtCore.QEvent.Type.Close)
        if ev.type() in gone and self._paper is not None:
            self._paper.__exit__()
            self._paper = None
            watched.removeEventFilter(self)
        return False

    # -- keys --------------------------------------------------------------

    def keyPressEvent(self, ev: Any) -> None:
        key = ev.key()
        shift = bool(ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
        control = bool(ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier)
        keys = QtCore.Qt.Key
        if control and key == keys.Key_C:
            self.copy_image()
        elif control and key == keys.Key_S:
            self.export()
        elif control and key == keys.Key_W:
            self.close()
        elif key in (keys.Key_Home, keys.Key_0):
            self.home()
        elif key == keys.Key_A:
            self.autoscale()
        elif key in (keys.Key_Plus, keys.Key_Equal):
            self.zoom(0.5)
        elif key == keys.Key_Minus:
            self.zoom(2.0)
        elif key == keys.Key_Backspace:
            self.step_history(1 if shift else -1)
        elif key in (keys.Key_T, keys.Key_F3):
            self.trace()
        elif key == keys.Key_Escape and self._tracing is not None:
            self._trace_off()
        elif key == keys.Key_L:
            self.legend.setVisible(not self.legend.isVisible())
        elif key == keys.Key_G:
            self._grid = not self._grid
            self.item.showGrid(x=self._grid, y=self._grid, alpha=GRID_ALPHA)
        elif key in (keys.Key_Left, keys.Key_Right):
            step = -1 if key == keys.Key_Left else 1
            if self._tracing is not None:
                self._trace_step(step * (NUDGE_FAST_PX if shift else NUDGE_PX))
            else:
                self.pan(step * PAN_SHARE, 0.0)
        elif key in (keys.Key_Up, keys.Key_Down):
            step = 1 if key == keys.Key_Up else -1
            if self._tracing is not None:
                self._trace_curve(-step)
            else:
                self.pan(0.0, step * PAN_SHARE)
        else:
            super().keyPressEvent(ev)
            return
        ev.accept()

    # -- the rest ----------------------------------------------------------

    def say(self, message: str) -> None:
        """Put one line on the status bar, which is the window's whole voice."""
        self._message = message
        self.status.setText(message)

    def describe(self) -> protocol.WindowInfo:
        """This window as `Describe` reports it."""
        (left, right), (low, high) = self.canvas.viewRange()
        return protocol.WindowInfo(
            number=self.number,
            kind=self.kind,
            title=self.windowTitle(),
            current=self.current,
            plots=tuple(
                protocol.PlotInfo(
                    worksheet=plot.worksheet,
                    label=plot.label,
                    text=plot.text,
                    kind=plot.kind,
                    hidden=not plot.visible,
                )
                for plot in self.plots
            ),
            xrange=(float(left), float(right)),
            yrange=(float(low), float(high)),
            polar=self.polar,
        )

    def retitle(self, current: bool) -> None:
        """Say in the title bar whether the next plot lands here."""
        self.current = current
        self.setWindowTitle(protocol.titled(self.kind, self.number, current))

    def _resized(self, *_: Any) -> None:
        """The canvas has a new size: re-sample, and frame it while it is fresh.

        The default framing needs the shape of the canvas and so cannot be
        worked out before there is one - the window is built, laid out, and
        only then knows how wide it is. A window nobody has moved the view of
        is framed again on every resize, which is both how it gets its first
        framing and how a window dragged wider goes on showing x in [-5, 5]
        rather than drifting outwards as the aspect lock makes room.
        """
        if self._default and self.canvas.width() > 1:
            self.home()
            self._history.clear()
            self._at = 0
        self._timer.start()

    def closeEvent(self, ev: Any) -> None:
        """The window manager's business, and the host's to hear about."""
        self.host.closed(self.number)
        super().closeEvent(ev)


#: The kinds trace can ride. The others have no parametrized curve under the
#: marker, so up and down step over them.
TRACEABLE = frozenset(
    {PlotKind.CURVE, PlotKind.FAMILY, PlotKind.PARAMETRIC, PlotKind.POLAR}
)


class _on_paper:
    """The window in paper colors for as long as an export takes.

    A dark plot pasted into a document is a black rectangle, so every image
    export sets a white background, redraws the curves in the colors that read
    on it, exports, and puts the window back. The window flickers, which is
    honest: the picture that was taken is the picture that was shown.
    """

    def __init__(self, window: Window2D) -> None:
        self._window = window

    def __enter__(self) -> None:
        window = self._window
        self._pens = [(plot, plot.item.opts["pen"]) for plot in window.plots if plot.item]
        window.plot.setBackground("w")
        for plot, _ in self._pens:
            plot.item.setPen(pg.mkPen(plot.paper, width=1))
        for edge in ("bottom", "left"):
            window.item.getAxis(edge).setTextPen(pg.mkPen("k"))
        for line in window.axes:
            line.setPen(pg.mkPen("#404040"))
        QtWidgets.QApplication.processEvents()

    def __exit__(self, *_: Any) -> None:
        window = self._window
        window.plot.setBackground(BACKGROUND)
        for plot, pen in self._pens:
            plot.item.setPen(pen)
        for edge in ("bottom", "left"):
            window.item.getAxis(edge).setTextPen(pg.mkPen(TEXT_COLOR))
        for line in window.axes:
            line.setPen(pg.mkPen(AXIS_COLOR))


def _number(value: float) -> str:
    """A number as the status line writes it: six decimals, or a word."""
    if not np.isfinite(value):
        return "undefined"
    return FORMAT.format(value)


def _short(value: float) -> str:
    """A number as a sentence about the view writes it, with no trailing noise.

    A reading wants every digit it has; a range does not - `-5 ≤ x ≤ 5` is what
    the window is showing, and six decimals of it would be six decimals of
    nothing.
    """
    return f"{value:g}"
