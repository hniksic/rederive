"""The 2D plot window: a canvas, a legend and a status line.

The picture is meant as a measuring instrument rather than an illustration.

Axes are drawn through the origin, as in mathematics, rather than as pyqtgraph's
edge spines: the edge axis items keep their numbers but not their lines, and two
infinite lines cross at (0, 0). A view away from the origin has numbers along the edges
and no axis lines.

A new window shows x in [-5, 5] with equal scales, so circles are round. The `1:1`
toggle releases the lock for that window only, not as a default for the next. The
mouse does the rest; the only place to type exact bounds is the context menu's `Set
range...`. The window reframes itself only when a newly added curve has finite values
but none in view, autoscaling y and saying so.

The framing arithmetic is in `plot.view` and the resampling policy in `plot.resample`,
both toolkit-free; this module handles the drawing.

Sampling is done in screen space and repeated, debounced, on every view change, so
zooming reveals detail such as spikes narrower than a pixel. It runs on the session's
sampling thread. The only evaluation on the Qt thread is the single point under the
trace marker, so its value is the function's rather than the pixel grid's.

The window draws all 2D kinds: function curves, parametric pairs, polar curves, data
points, implicit contours and shaded regions. They share one plot list and one set of
gestures (the legend hides a region as it hides a curve, and Remove removes either),
and differ only in how they are sampled: a function over the abscissa, a parametric
curve over its parameter with error measured as distance in the plane, a field over a
grid at half the canvas resolution. All are re-evaluated on view changes through the
same keyed job, so zooming into an implicit curve reveals detail just as for a
function.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters  # noqa: F401  - registers the exporters copy and export use
from pyqtgraph.Qt import QtCore, QtGui, QtSvg, QtWidgets

from rederive.engine.context import Angle, Context
from rederive.plot import actions, controls, evaluate, forms, protocol, resample, view
from rederive.plot.appearance import (
    AXIS_COLOR,
    BACKGROUND,
    CHIP_OFFSET_PX,
    CLICK_SLOP_PX,
    CURVE_WIDTH,
    GRID_ALPHA,
    HAIRLINE_ALPHA,
    HIT_PX,
    LEGEND_FADED,
    MARKER_PX,
    MARKER_WIDTH,
    NUDGE_FAST_PX,
    NUDGE_PX,
    PAPER_AXIS,
    POLAR_CLOSEST_PX,
    POLAR_RINGS,
    POLAR_SPOKES,
    REGION_ALPHA,
    STEP_FAST_SHARE,
    STEP_SHARE,
)
from rederive.plot.model import (
    FIELDS,
    FUNCTIONS,
    PALETTE,
    PAPER_PALETTE,
    Plot,
    pointed,
    reading,
    roughly,
)
from rederive.plot.protocol import PlotKind
from rederive.plot.qt import theme
from rederive.syntax import ParseState

__all__ = ["Drawn", "Legend", "Sheet", "Window2D"]

#: The color of the axis numbers and trace marker ring, specific to this window. Other
#: canvas colors are in `plot.appearance` (shared with the page), and the chrome's in
#: `theme`.
TEXT_COLOR = "#d0d0d0"

#: Resolution of exported SVGs. Qt's SVG writer defaults to 72 dpi while other Qt paint
#: devices use 96, so this keeps the file the size of the picture.
SVG_DPI = 96

#: The modifiers used in key bindings. Others, such as the keypad flag of a `0` typed on
#: the numeric pad, are masked off before looking up a key in the command table.
MODIFIERS = (
    QtCore.Qt.KeyboardModifier.ControlModifier
    | QtCore.Qt.KeyboardModifier.ShiftModifier
    | QtCore.Qt.KeyboardModifier.AltModifier
    | QtCore.Qt.KeyboardModifier.MetaModifier
)

#: Legend layout in logical pixels: the color sample's length, the gap around it, the
#: width of the eye icon at the row's end, and the row height.
LEGEND_SAMPLE = 18
LEGEND_GAP = 8
LEGEND_EYE = 16
LEGEND_ROW = 19

#: Fade-in duration when a hidden curve is shown from the legend. Hiding is instant;
#: only showing is eased, since that is what the eye follows.
FADE_MS = 150

#: Corner radius of the trace marker's value chip. The marker's other dimensions are in
#: `plot.appearance`.
CHIP_RADIUS = 4.0

#: Kinds parametrized by something other than the abscissa, so the marker moves along t.
PARAMETRIZED = protocol.PARAMETRIZED


def commanded(
    owner: Any,
    keyed: dict[tuple[int, int], Any] | None,
    one: controls.Control,
    handler: Callable[..., None],
) -> Any:
    """Create the menu action for a control, registering its keys.

    Labels, keys and order come from the `plot.controls` table, which both window kinds
    and the browser build their menus from.

    The keys are set on the action and added to `keyed`, the table `keyPressEvent`
    dispatches from. Qt itself never triggers them: its shortcuts would fire alongside
    `keyPressEvent`, which must handle keys anyway, since some mean different things
    with a trace marker shown. `WidgetShortcut` on an action no widget owns therefore
    only displays the keys in the menu's shortcut column, and keys typed into a
    toolbar field stay in the field.
    """
    action = QtGui.QAction(controls.plain(one), owner)
    action.setCheckable(one.kind is controls.Kind.TOGGLE)
    keys = controls.keys(one)
    if keys:
        action.setShortcuts([QtGui.QKeySequence(key) for key in keys])
        action.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetShortcut)
        action.setShortcutVisibleInContextMenu(True)
        if keyed is not None:
            for sequence in action.shortcuts():
                stroke = sequence[0]
                keyed[(stroke.keyboardModifiers().value, int(stroke.key()))] = action
    action.triggered.connect(handler)
    return action


def buttoned(owner: Any, one: controls.Control, handler: Callable[..., None]) -> Any:
    """Create the toolbar button action for a control.

    This is separate from the menu action because the labels differ (`clear` on the
    bar, `Clear` in the menu).
    """
    action = QtGui.QAction(one.bar, owner)
    action.setCheckable(one.kind is controls.Kind.TOGGLE)
    if one.hint:
        action.setToolTip(one.hint)
    action.triggered.connect(handler)
    return action


def pressed(ev: Any) -> tuple[int, int]:
    """A key event as a command table key: (modifiers, key)."""
    return ((ev.modifiers() & MODIFIERS).value, int(ev.key()))


def rendered(
    offered: dict[str, Any], entries: Sequence[controls.Entry]
) -> dict[str, controls.Entry]:
    """Apply menu entries to the actions that draw them.

    The entries decide labels, ticks, enabled state and presence. Actions without an
    entry (e.g. plot entries when no plot was clicked) are hidden rather than disabled.
    """
    shown = {entry.name: entry for entry in entries}
    for name, action in offered.items():
        entry = shown.get(name)
        action.setVisible(entry is not None)
        if entry is None:
            continue
        action.setText(entry.label)
        action.setChecked(entry.checked)
        action.setEnabled(entry.enabled)
    return shown


def loosened(one: controls.Control) -> tuple[int, ...]:
    """A control's keys without modifiers, for matching on the key alone.

    Some keys can't be matched by exact stroke: `+` is a shifted `=` on most keyboards,
    depending on the layout, and plain letter keys should work regardless of modifiers.
    """
    return tuple(
        int(QtGui.QKeySequence(key)[0].key()) for key in controls.keys(one)
    )


@dataclass
class Drawn(Plot):
    """A plot as held by this window, with its drawing items and samples.

    The fields inherited from `Plot` come from the session; the ones added here belong
    to drawing, which keeps `Plot` itself free of numpy. The plot list, legend, trace
    and feature scan all use these.
    """

    item: Any = None
    #: A region's shading image. Only regions need this second item.
    fill: Any = None
    #: The lambdified closure, once the sampling thread has built it.
    closure: Callable[..., np.ndarray] | None = None
    #: The two closures of a parametric pair or polar curve. For a polar curve,
    #: `closure` is r(θ), used by the polar readout.
    pair: tuple[Callable[..., np.ndarray], Callable[..., np.ndarray]] | None = None
    #: The samples currently drawn, used by trace and the feature scan.
    xs: np.ndarray = field(default_factory=lambda: np.empty(0))
    ys: np.ndarray = field(default_factory=lambda: np.empty(0))
    #: The parameter value of each sample, for parametrized kinds, where the marker
    #: sits at a t rather than an x.
    ts: np.ndarray = field(default_factory=lambda: np.empty(0))
    #: The grid a region was last shaded from (axis vectors and truth values), used to
    #: tell whether a click landed inside it.
    grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    #: The curve's features for the current samples, computed when trace first needs
    #: them and dropped when the samples change, and the labels of the curves that the
    #: intersections are with.
    features: tuple[Any, ...] | None = None
    crossed: tuple[str, ...] = ()
    #: The resample these samples came from; results of outdated jobs are dropped.
    generation: int = 0

    @classmethod
    def of(cls, plot: Drawn) -> Drawn:
        """Create this window's record of a plot from the session."""
        return cls(**{field.name: getattr(plot, field.name) for field in fields(Plot)})

    @property
    def visible(self) -> bool:
        return self.item is not None and self.item.isVisible()


class Canvas(pg.ViewBox):
    """The view box, with customized mouse handling.

    It differs from stock in four ways. Rubber-band zoom uses the *right* button,
    leaving left-drag for panning; a right click without movement opens the context
    menu. Ctrl and Shift restrict the wheel to one axis. Every range-changing gesture
    first pushes the range onto the window's history, so Backspace steps back through
    all view changes, unlike pyqtgraph's stack, which only tracks rubber bands. And the
    right-click menu is the window's own.
    """

    def __init__(self, window: Window2D) -> None:
        # Disable the stock menu entirely; the window builds its own.
        super().__init__(enableMenu=False)
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

    def updateScaleBox(self, one: Any, two: Any) -> None:
        """Draw the rubber band in the accent color rather than stock yellow.

        Styled here rather than at construction, since the view box creates the band
        lazily and recreates it when the mouse mode changes.
        """
        box = self.rbScaleBox
        box.setPen(pg.mkPen(theme.ACCENT, width=1))
        box.setBrush(pg.mkBrush(_tinted(theme.ACCENT, 0.08)))
        super().updateScaleBox(one, two)

    def wheelEvent(self, ev: Any, axis: int | None = None) -> None:
        modifiers = ev.modifiers()
        if modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
            axis = 0
        elif modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
            axis = 1
        self._window.remember()
        super().wheelEvent(ev, axis)

    def mouseClickEvent(self, ev: Any) -> None:
        """Handle a canvas click: open the menu, center the view, or select a curve.

        A right click without movement arrives here as a click rather than a drag, so
        the menu is opened here as well as at the end of a zero-length rubber band.
        """
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            self.raiseContextMenu(ev)
            return
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

    def raiseContextMenu(self, ev: Any) -> None:
        """Open the window's menu, prepared for the clicked point.

        The stock implementation lets the scene append `Plot Options` and `Export...`
        to the menu; overriding it shows only the window's own entries.
        """
        self._window.prepare_menu(self.mapToView(ev.pos()))
        self._window.menu.popup(ev.screenPos().toPoint())


class _Row(QtWidgets.QWidget):
    """A legend row: a color sample, the plot's name, and a visibility eye.

    Painted directly, which is simpler than composing styled labels.

    It is transparent to the mouse: the legend card handles all gestures, working out
    which row is hovered or clicked in one place.
    """

    def __init__(self, parent: Any) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFixedHeight(LEGEND_ROW)
        self.name = ""
        self.color = PALETTE[0]
        self.hidden = False
        self.over = False
        self.paper = False

    def describe(self, name: str, color: str, hidden: bool, paper: bool) -> None:
        """Set what the row shows, and repaint.

        The width is a minimum, so the layout stretches all rows to the widest one,
        aligning the eyes in a column and giving all hover highlights the same shape.
        """
        self.name, self.color, self.hidden, self.paper = name, color, hidden, paper
        self.setMinimumWidth(
            LEGEND_GAP
            + LEGEND_SAMPLE
            + LEGEND_GAP
            + self.fontMetrics().horizontalAdvance(name)
            + LEGEND_GAP
            + LEGEND_EYE
        )
        self.update()

    def paintEvent(self, _ev: Any) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        rectangle = QtCore.QRectF(self.rect())
        if self.over:
            wash = theme.CARD_PAPER_ROW_OVER if self.paper else theme.CARD_ROW_OVER
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(*wash))
            painter.drawRoundedRect(rectangle, 5.0, 5.0)
        painter.setOpacity(LEGEND_FADED if self.hidden else 1.0)
        color = QtGui.QColor(self.color)
        middle = rectangle.center().y()
        pen = QtGui.QPen(color)
        pen.setWidthF(CURVE_WIDTH)
        painter.setPen(pen)
        painter.drawLine(
            QtCore.QPointF(LEGEND_GAP, middle),
            QtCore.QPointF(LEGEND_GAP + LEGEND_SAMPLE, middle),
        )
        painter.setPen(color)
        painter.setFont(self.font())
        left = LEGEND_GAP + LEGEND_SAMPLE + LEGEND_GAP
        painter.drawText(
            QtCore.QRectF(left, 0.0, rectangle.width() - left, rectangle.height()),
            int(
                QtCore.Qt.AlignmentFlag.AlignLeft
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            ),
            self.name,
        )
        if self.over:
            painter.setOpacity(1.0)
            self._eye(painter, rectangle)
        painter.end()

    def _eye(self, painter: Any, rectangle: Any) -> None:
        """Draw the eye icon showing what a click would do.

        An open eye for a visible plot and a struck-through one for a hidden plot,
        drawn only under the pointer, so the legend reads as a list rather than as a
        column of buttons.
        """
        room = QtCore.QRectF(
            rectangle.right() - LEGEND_EYE - LEGEND_GAP / 2,
            rectangle.center().y() - LEGEND_EYE / 2,
            LEGEND_EYE,
            LEGEND_EYE,
        )
        almond = QtCore.QRectF(
            room.center().x() - 5.0, room.center().y() - 3.0, 10.0, 6.0
        )
        ink = QtGui.QColor(theme.DIM if self.paper else theme.MUTED)
        painter.setPen(QtGui.QPen(ink, 1.0))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawEllipse(almond)
        painter.setBrush(ink)
        painter.drawEllipse(room.center(), 1.4, 1.4)
        if self.hidden:
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawLine(almond.bottomLeft(), almond.topRight())


class Legend(QtWidgets.QFrame):
    """The plot list as a card floating over the picture, one row per plot.

    It is a widget over the canvas rather than an item in it, so both window kinds can
    share it (a GL view is not a graphics scene).

    Clicking anywhere on a row works: a left click toggles the plot's visibility, and a
    right click opens the plot's menu. The eye shown under the pointer only indicates
    what a click would do.

    Rows are a pool that grows but never shrinks; unused rows are hidden rather than
    deleted, since the legend is rebuilt on every change and deleting widgets from a
    live layout tends to leave stale rows.
    """

    #: A click on a row: the row's index in the plot list, and the mouse button.
    picked = QtCore.Signal(int, object)

    def __init__(self, parent: Any) -> None:
        super().__init__(parent)
        self.setObjectName("legend")
        self.setMouseTracking(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 5, 6, 5)
        layout.setSpacing(1)
        self._rows: list[_Row] = []
        self.shown = 0
        #: The current entries, for drawing the same list elsewhere (e.g. exports).
        self.entries: tuple[tuple[str, str, bool], ...] = ()
        self._over = -1
        self._paper = False
        self._dress()

    def _dress(self) -> None:
        """Style the card: a translucent background, a thin border, round corners."""
        ground = theme.CARD_PAPER if self._paper else theme.CARD
        edge = theme.CARD_PAPER_EDGE if self._paper else theme.CARD_EDGE
        self.setStyleSheet(
            f"QFrame#legend {{ background: {ground}; border: 1px solid {edge};"
            f" border-radius: {theme.CARD_RADIUS}px; }}"
        )

    def rebuild(
        self, entries: Sequence[tuple[str, str, bool]], paper: bool = False
    ) -> None:
        """Show one row per plot: its name in its color, dimmed if hidden.

        Colors are passed in, so an export can rebuild the card in paper colors.
        """
        if paper != self._paper:
            self._paper = paper
            self._dress()
        self.entries = tuple(entries)
        while len(self._rows) < len(entries):
            row = _Row(self)
            self.layout().addWidget(row)
            self._rows.append(row)
        self.shown = len(entries)
        for index, row in enumerate(self._rows):
            if index >= len(entries):
                row.setVisible(False)
                continue
            name, color, hidden = entries[index]
            row.describe(name, color, hidden, paper)
            row.over = index == self._over
            row.setVisible(True)
        # The card floats outside any layout, so it must size itself, after its own
        # layout has sized the rows.
        self.layout().activate()
        self.resize(self.layout().sizeHint())

    def mousePressEvent(self, ev: Any) -> None:
        row = self._at(ev.position().toPoint())
        if row < 0:
            ev.ignore()
            return
        self.picked.emit(row, ev.button())
        ev.accept()

    def mouseMoveEvent(self, ev: Any) -> None:
        self._hover(self._at(ev.position().toPoint()))
        ev.ignore()

    def leaveEvent(self, ev: Any) -> None:
        self._hover(-1)
        super().leaveEvent(ev)

    def _hover(self, row: int) -> None:
        if row == self._over:
            return
        self._over = row
        for index, one in enumerate(self._rows[: self.shown]):
            one.over = index == row
            one.update()

    def _at(self, point: Any) -> int:
        """The index of the row containing the point, or -1."""
        for index, row in enumerate(self._rows[: self.shown]):
            if row.geometry().contains(point):
                return index
        return -1


class _Chip(pg.TextItem):
    """The traced values, in a small dark label next to the marker.

    It paints its own rounded background with a thin border, instead of the stock text
    item's polygon. The base item's fill and border are left empty. The background is
    painted before the text, since child items paint after their parent.
    """

    def __init__(self) -> None:
        super().__init__(html="", anchor=(0.0, 1.0))
        self._fill = pg.mkBrush(_tinted("#0f0f15", 0.85))
        self._edge = pg.mkPen(_tinted("#ffffff", 0.10))
        self.setFont(theme.monospace(8))

    def paint(self, painter: Any, *arguments: Any) -> None:
        super().paint(painter, *arguments)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setPen(self._edge)
        painter.setBrush(self._fill)
        painter.drawRoundedRect(
            self.textItem.mapRectToParent(self.textItem.boundingRect()),
            CHIP_RADIUS,
            CHIP_RADIUS,
        )


class _PolarGrid(pg.UIGraphicsItem):
    """The polar grid of rings and spokes, which pyqtgraph lacks.

    Rings are constant r and spokes constant θ, painted in view coordinates, so with
    equal scales the rings are circles, and otherwise the ellipses that constant r
    actually is. The pen is cosmetic, so lines stay one pixel wide at any zoom, and
    painting is clipped to the view.

    The page draws the same grid by the same rules; see `web/plot2d.js`'s `_polarGrid`.
    """

    def __init__(self) -> None:
        super().__init__()
        self._pen = pg.mkPen(_tinted(AXIS_COLOR, GRID_ALPHA))
        self._pen.setCosmetic(True)

    def paint(self, painter: Any, *arguments: Any) -> None:
        rect = self.boundingRect()
        reach = max(
            abs(rect.left()), abs(rect.right()), abs(rect.top()), abs(rect.bottom())
        )
        step = _ruled(reach)
        across, up = self.pixelWidth(), self.pixelHeight()
        if step <= 0 or not across or not up:
            return
        if min(step / across, step / up) <= POLAR_CLOSEST_PX:
            return
        painter.setClipRect(rect)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setPen(self._pen)
        origin = QtCore.QPointF(0.0, 0.0)
        for ring in range(1, int(reach / step) + 2):
            painter.drawEllipse(origin, ring * step, ring * step)
        # Twice the reach is enough for the spokes to leave the view in every
        # direction.
        far = 2.0 * reach
        for spoke in range(POLAR_SPOKES):
            angle = spoke * 2.0 * np.pi / POLAR_SPOKES
            end = QtCore.QPointF(far * np.cos(angle), far * np.sin(angle))
            painter.drawLine(origin, end)


def _ruled(reach: float) -> float:
    """The ring spacing: a round step of about a fifth of the reach.

    Steps are 1, 2 or 5 times a power of ten, so the rings are at readable numbers.
    """
    if not (reach > 0):
        return 0.0
    rough = reach / POLAR_RINGS
    power = 10.0 ** np.floor(np.log10(rough))
    for multiple in (1.0, 2.0, 5.0):
        if multiple * power >= rough:
            return float(multiple * power)
    return float(10.0 * power)


class Window2D(QtWidgets.QMainWindow):
    """A top-level 2D plot window."""

    def __init__(self, number: int, session: Any) -> None:
        super().__init__()
        self.number = number
        self.kind = protocol.WindowKind.TWO_D
        self.session = session
        self.plots: list[Drawn] = []
        #: Whether univariate curves are read as r = f(θ). A mode of the whole view,
        #: set by the toolbar toggle.
        self.polar = False
        self.current = False
        #: View history, aspect lock, and whether the initial framing was changed.
        self._framing = actions.Framing()
        self._counter = 0
        #: The index of the traced curve, or None while trace is off.
        self._tracing: int | None = None
        self._trace_x = 0.0
        #: The marker's parameter value, on a curve parametrized by something other
        #: than the abscissa.
        self._trace_t = 0.0
        self._message = ""
        #: Whether this window's expressions use degrees, taken from the context of
        #: each added plot, since it is a worksheet setting.
        self.degrees = False
        #: The parse state and context for typed bounds: those of the last added plot.
        self._state = ParseState()
        self._context = Context()
        #: The `Set range...` dialog, once opened.
        self._bounds: Bounds | None = None
        self._grid = True
        #: Whether the legend is shown. Kept separately from the widget, so a legend
        #: the user hid stays hidden when a plot is added.
        self._legend = True
        #: Running fade animations by plot, so clicking a row twice replaces its
        #: animation rather than racing it.
        self._fades: dict[int, Any] = {}
        #: Whether a demonstration step is showing in this window. While it is, the
        #: window stays on top and forwards unused keys to the program.
        self._demonstrating = False
        self._build()
        self.retitle()
        self.home()

    # -- construction ------------------------------------------------------

    def _build(self) -> None:
        self.canvas = Canvas(self)
        self.plot = pg.PlotWidget(viewBox=self.canvas, background=BACKGROUND)
        self.item = self.plot.getPlotItem()
        self._strip_spines()
        self._origin_axes()
        self._polar_grid()
        self._ruling()
        self._make_menu()
        self.legend = Legend(self.plot)
        self.legend.picked.connect(self._legend_clicked)
        self.legend.setVisible(False)
        self._trace_items()
        self.setCentralWidget(self._laid_out())
        self.canvas.setAspectLocked(True, ratio=1.0)
        self.canvas.disableAutoRange()
        # Hide the auto-range button. Auto-range is always off, and would loop with
        # screen-space resampling: each widening fills the view with more curve,
        # giving auto-range a reason to widen again.
        self.item.hideButtons()
        self.canvas.sigRangeChanged.connect(self._ranged)
        self.canvas.sigResized.connect(self._resized)
        self.item.scene().sigMouseMoved.connect(self._moved)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(resample.RESAMPLE_DELAY_MS)
        self._timer.timeout.connect(self.resample)
        self.plot.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.resize(760, 560)

    def _trace_items(self) -> None:
        """Create the trace marker's items, initially hidden.

        A ring in the curve's color, a dashed hairline from it down to the abscissa, and
        a chip showing the values. They are shown, colored and hidden together.
        """
        self.marker = pg.ScatterPlotItem(
            size=MARKER_PX,
            symbol="o",
            pen=pg.mkPen(TEXT_COLOR, width=MARKER_WIDTH),
            brush=None,
        )
        self.marker.setZValue(20)
        self.hairline = pg.PlotCurveItem(
            np.empty(0), np.empty(0), pen=pg.mkPen(TEXT_COLOR)
        )
        self.hairline.setZValue(19)
        self.chip = _Chip()
        self.chip.setZValue(21)
        for item in (self.marker, self.hairline, self.chip):
            item.setVisible(False)
            self.item.addItem(item, ignoreBounds=True)

    def _laid_out(self) -> QtWidgets.QWidget:
        """The toolbar, the canvas, and the status line.

        The toolbar holds window-wide controls: the equal-scales and polar toggles,
        where their state is visible, and Clear.
        """
        holder = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        bar = QtWidgets.QToolBar()
        bar.setMovable(False)
        handlers = self._handlers()
        self.equal = self._button(bar, "scales.equal", handlers)
        self.equal.setChecked(True)
        self.polar_toggle = self._button(bar, "view.polar", handlers)
        bar.addWidget(theme.divider())
        self.clear_action = self._button(bar, "clear", handlers)
        theme.dangerous(bar, self.clear_action)
        # The parameter range of the selected parametric or polar plot, hidden
        # (along with its divider) while there is no such plot.
        self._range_actions = self._range_strip(bar)
        layout.addWidget(bar)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self._status_line())
        return holder

    def _status_line(self) -> QtWidgets.QWidget:
        """The status line: messages on the left, the pointer readout on the right.

        The readout uses a fixed-pitch font so digits stay in place as the pointer
        moves, with coordinate names dimmer than the numbers.
        """
        line = QtWidgets.QFrame()
        line.setObjectName("statusline")
        line.setStyleSheet(
            f"QFrame#statusline {{ background: {theme.STATUS};"
            f" border-top: 1px solid {theme.STATUS_EDGE}; }}"
        )
        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet(f"color: {theme.STATUS_TEXT}; background: transparent;")
        self.readout = QtWidgets.QLabel("")
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.readout.setFont(theme.monospace())
        self.readout.setStyleSheet("background: transparent;")
        across = QtWidgets.QHBoxLayout(line)
        across.setContentsMargins(10, 3, 10, 3)
        across.addWidget(self.status, 1)
        across.addWidget(self.readout, 0)
        return line

    def _button(self, bar: Any, name: str, handlers: dict[str, Any]) -> Any:
        """Add a toolbar button for one of this window's controls."""
        action = buttoned(self, controls.control(name, controls.FLAT), handlers[name])
        bar.addAction(action)
        return action

    def _range_strip(self, bar: Any) -> tuple[Any, ...]:
        """Add the parameter range fields to the toolbar, initially hidden.

        Their layout and labels come from `plot.forms`. Returns the strip's actions,
        including its divider, to show and hide together.
        """
        self.range_fields: dict[str, QtWidgets.QLineEdit] = {}
        raised = []
        for piece in forms.TRANGE.pieces:
            if piece.divider:
                raised.append(bar.addWidget(theme.divider()))
            elif piece.entry:
                edit = self._range_field(forms.TRANGE.field(piece.entry))
                self.range_fields[piece.entry] = edit
                raised.append(bar.addWidget(edit))
            else:
                label = QtWidgets.QLabel("" if piece.name else piece.word)
                if piece.name:
                    self.range_name = label
                raised.append(bar.addWidget(label))
        self.range_low = self.range_fields["low"]
        self.range_high = self.range_fields["high"]
        for action in raised:
            action.setVisible(False)
        return tuple(raised)

    def _range_field(self, one: forms.Field) -> QtWidgets.QLineEdit:
        """A range bound field that only takes focus when clicked.

        Otherwise it could take focus on open and swallow the window's pan, zoom and
        trace keys, as with the 3D window's domain fields.
        """
        edit = theme.field(one.width)
        edit.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        edit.editingFinished.connect(self._range_edited)
        return edit

    def _strip_spines(self) -> None:
        """Keep the tick numbers along the edges, but hide the edge spines.

        The axes go through the origin, so spines would be a second pair of axes.
        pyqtgraph draws the spine with the axis pen and the ticks (including the grid,
        which is ticks spanning the canvas) with the tick pen, so a transparent axis
        pen hides the spine and keeps the grid.
        """
        for edge in ("bottom", "left"):
            axis = self.item.getAxis(edge)
            axis.setPen(pg.mkPen(color=(0, 0, 0, 0)))
            axis.setTickPen(pg.mkPen(AXIS_COLOR))
            axis.setTextPen(pg.mkPen(TEXT_COLOR))

    def _origin_axes(self) -> None:
        """The axes: two infinite lines through (0, 0)."""
        pen = pg.mkPen(color=AXIS_COLOR, width=1)
        self.axes = (
            pg.InfiniteLine(pos=0, angle=90, pen=pen),
            pg.InfiniteLine(pos=0, angle=0, pen=pen),
        )
        for line in self.axes:
            line.setZValue(-10)
            self.item.addItem(line, ignoreBounds=True)

    def _polar_grid(self) -> None:
        """Create the polar grid, shown in polar mode, below the axis lines."""
        self.rings = _PolarGrid()
        self.rings.setZValue(-11)
        self.item.addItem(self.rings, ignoreBounds=True)

    def _ruling(self) -> None:
        """Show the grid that matches the view mode, if the grid is on.

        A polar view shows only the polar grid, a Cartesian view only the Cartesian
        one. The `Grid` toggle applies to whichever is current.
        """
        ruled = self._grid and not self.polar
        self.item.showGrid(x=ruled, y=ruled, alpha=GRID_ALPHA)
        self.rings.setVisible(self._grid and self.polar)

    def _make_menu(self) -> None:
        """Build the window's context menu and its key bindings.

        The stock pyqtgraph menu suits a charting library (mouse modes, transforms that
        would differentiate screen samples, obscure axis submenus) and knows nothing
        of trace or history, so it is replaced with the `plot.controls` table. Each
        entry shows its key, so the menu teaches the keyboard.

        The entries at the end are about the clicked plot, so they are filled in when
        the menu opens.

        Both pyqtgraph menus are disabled, since stock `raiseContextMenu` would append
        `Plot Options` and `Export...` to any raised menu.
        """
        self.item.setMenuEnabled(False, False)
        #: Actions for the menu's keys, by key and modifiers, used by `keyPressEvent`.
        self._keyed: dict[tuple[int, int], Any] = {}
        #: Actions for keys without menu entries, by key alone.
        self._loose: dict[int, Any] = {}
        #: The menu's actions, by control name, updated whenever the menu opens.
        self._offered: dict[str, Any] = {}
        self.menu = QtWidgets.QMenu(self)
        self.menu.aboutToShow.connect(self._read_menu)
        self._make_controls()
        self._pointed: Drawn | None = None

    def _make_controls(self) -> None:
        """Create an action for each control, and add menu entries in table order.

        Labels, keys, order and grouping all come from `plot.controls`; a separator is
        added wherever the group changes.
        """
        handlers = self._handlers()
        group: int | None = None
        for one in controls.FLAT:
            if not one.menu and not controls.keys(one):
                continue
            if one.kind is controls.Kind.CHOICE:
                action = self._choice(one)
            else:
                keyed = self._keyed if one.menu else None
                action = commanded(self, keyed, one, handlers[one.name])
            if not one.menu:
                for key in loosened(one):
                    self._loose[key] = action
                continue
            if group is not None and one.group != group:
                self.menu.addSeparator()
            group = one.group
            self.menu.addAction(action)
            self._offered[one.name] = action

    def _handlers(self) -> dict[str, Callable[..., None]]:
        """The handler of each of this window's controls, by control name."""
        return {
            "range.set": self.set_range,
            "view.all": self.autoscale,
            "view.home": self.home,
            "view.back": lambda: self.step_history(-1),
            "view.forward": lambda: self.step_history(1),
            "view.zoom.in": lambda: self.zoom(1 / actions.ZOOM_FACTOR),
            "view.zoom.out": lambda: self.zoom(actions.ZOOM_FACTOR),
            "trace": self.trace,
            "grid": self.toggle_grid,
            "legend": self.toggle_legend,
            "image.copy": self.copy_image,
            "image.export": self.export,
            "clear": self.clear,
            "close": self.close,
            "plot.remove": self._remove_pointed,
            "points.connect": self._toggle_connected,
            "scales.equal": self._equal_scales,
            "view.polar": self._polar_mode,
        }

    def _choice(self, one: controls.Control) -> Any:
        """A submenu for a choice control, such as point size.

        The alternatives are fixed, so the submenu is built once.
        """
        sizes = QtWidgets.QMenu(controls.plain(one), self)
        for item in controls.choices(one):
            sizes.addAction(
                item.label,
                lambda _=False, size=item.value: self._set_point_size(float(size)),
            )
        return sizes.menuAction()

    def _read_menu(self) -> None:
        """Update the menu from the window's state as it opens.

        The state can change outside the menu (e.g. trace starts by clicking a curve
        and ends with Escape), so the entries are computed from a snapshot by
        `plot.controls` each time.
        """
        rendered(self._offered, controls.menu(self.snapshot()))

    def snapshot(self) -> controls.Flat:
        """A snapshot of the window state that the menu depends on."""
        pointed = self._pointed
        return controls.Flat(
            tracing=self._tracing is not None,
            grid=self._grid,
            legend=self._legend,
            polar=self.polar,
            equal=self._framing.equal,
            pointed=None
            if pointed is None
            else controls.Pointed(
                named=pointed.named, kind=pointed.kind, connected=pointed.connected
            ),
        )

    # -- the plot list -----------------------------------------------------

    def add(self, plot: Plot) -> Drawn:
        """Add a plot to the window, replacing one with the same identity.

        A replaced plot keeps its color, so re-plotting doesn't reshuffle the picture.
        Returns the window's own record of the plot.
        """
        plot = Drawn.of(plot)
        existing = self.find(plot.worksheet, plot.label)
        if existing is not None:
            plot.color, plot.paper = existing.color, existing.paper
            self.remove(existing)
        else:
            index = self._counter % len(PALETTE)
            plot.color, plot.paper = PALETTE[index], PAPER_PALETTE[index]
            self._counter += 1
        # Take the angle unit, parse state and context from the plot's worksheet.
        self.degrees = plot.context.angle is Angle.DEGREE
        self._state, self._context = plot.state, plot.context
        # Polar mode belongs to the view: a univariate curve added to a polar
        # window is read as r = f(θ), and a polar one follows the window's mode.
        self._reread(plot)
        plot.item = self._made(plot)
        plot.item.setCurveClickable(True, width=int(HIT_PX))
        # The curve gets the click before the view box, so clicking a curve traces
        # that curve rather than the nearest one.
        plot.item.sigClicked.connect(
            lambda _item, ev, plot=plot: self._curve_clicked(plot, ev)
        )
        self.item.addItem(plot.item)
        if plot.kind is PlotKind.REGION:
            plot.fill = pg.ImageItem()
            # Set explicitly rather than relying on the global config, since the
            # array is indexed [x, y].
            plot.fill.setOpts(axisOrder="col-major")
            plot.fill.setZValue(-5)
            self.item.addItem(plot.fill, ignoreBounds=True)
        self.plots.append(plot)
        self._relabel()
        self.retitle()
        self._axis_names()
        self._show_trange()
        self._start(plot, fresh=True)
        return plot

    def _pen(self, color: str) -> Any:
        """A cosmetic pen `CURVE_WIDTH` logical pixels wide in the given color.

        Cosmetic, so the width doesn't change with zoom. Qt measures cosmetic widths in
        device pixels, ignoring high-DPI scaling (verified, not assumed), so the device
        pixel ratio is multiplied in.
        """
        return pg.mkPen(
            color, width=CURVE_WIDTH * self.devicePixelRatioF(), cosmetic=True
        )

    def _made(self, plot: Drawn) -> Any:
        """The drawing item for a plot.

        All kinds use the same item type: a stroke with gaps draws functions,
        parametric curves and implicit contours, and with symbols it draws points. So
        the legend, hit testing, coloring and export are shared.

        A region's item is an empty curve carrying its color for the legend; the
        shading is a separate image behind it.
        """
        pen = self._pen(plot.color)
        if plot.kind is PlotKind.DATA:
            item = pg.PlotDataItem(
                np.empty(0),
                np.empty(0),
                pen=pen if plot.connected else None,
                symbol="o",
                symbolSize=plot.point_size,
                # Symbols keep a one-pixel outline, since their size is set by the
                # point size; only the connecting line uses the curve width.
                symbolPen=pg.mkPen(plot.color),
                symbolBrush=pg.mkBrush(plot.color),
                antialias=True,
            )
            # Keep large data plots responsive by drawing only what is in view, at
            # the view's resolution.
            item.setDownsampling(auto=True)
            item.setClipToView(True)
            return item
        return pg.PlotDataItem(
            np.empty(0), np.empty(0), pen=pen, connect="finite", antialias=True
        )

    def find(self, worksheet: int, label: str) -> Drawn | None:
        """The plot with this worksheet and label, if the window has it."""
        for plot in self.plots:
            if plot.worksheet == worksheet and plot.label == label:
                return plot
        return None

    def remove(self, plot: Drawn) -> None:
        """Remove a plot from the window, along with its legend entry."""
        if plot.item is not None:
            self.item.removeItem(plot.item)
        if plot.fill is not None:
            self.item.removeItem(plot.fill)
        if plot in self.plots:
            self.plots.remove(plot)
        self._relabel()
        self.retitle()
        self._show_trange()
        if self._tracing is not None and self._tracing >= len(self.plots):
            self._trace_off()

    def clear(self) -> None:
        """Remove all plots from the window, along with range fields and axis labels."""
        for plot in list(self.plots):
            self.remove(plot)
        self._axis_names()

    def _relabel(self, paper: bool = False) -> None:
        """Rebuild the legend from the current plots, in paper colors if `paper`."""
        self.legend.rebuild(
            [
                (plot.named, plot.paper if paper else plot.color, not plot.visible)
                for plot in self.plots
            ],
            paper=paper,
        )
        # Hide an empty legend. The toggle state is kept separately from the widget,
        # so a legend the user hid stays hidden when a plot is added.
        self.legend.setVisible(self._legend and bool(self.plots))
        self._place_legend()

    def _place_legend(self) -> None:
        """Place the legend in the canvas's top left corner, inside the axes.

        The legend floats over the whole plot widget, so its position depends on the
        width of the left axis numbers, which changes as the view moves.
        """
        corner = self.plot.mapFromScene(
            self.canvas.mapRectToScene(self.canvas.boundingRect()).topLeft()
        )
        self.legend.move(corner.x() + 12, corner.y() + 12)

    def _axis_names(self) -> None:
        """Label the axes.

        The horizontal axis gets the variable's name. The vertical axis gets the label
        of the only curve, or nothing if there are several, since the legend names them.

        Kinds without an opinion on the axes, such as parametric pairs, are ignored, so
        a window with one beside `SIN(x)` still labels the x axis.
        """
        names = {plot.abscissa for plot in self.plots} - {""}
        self.item.setLabel("bottom", names.pop() if len(names) == 1 else "")
        single = self.plots[0] if len(self.plots) == 1 else None
        if single is not None and single.kind in FIELDS:
            self.item.setLabel("left", single.axes[1])
        elif single is not None and single.label and single.kind in FUNCTIONS:
            self.item.setLabel("left", single.label)
        else:
            self.item.setLabel("left", "")

    # -- sampling ----------------------------------------------------------

    def _start(self, plot: Drawn, fresh: bool = False) -> None:
        """Request sampling of this plot over the visible range.

        All expensive work (conversion, lambdify, sampling) happens on the sampling
        thread, so the window stays responsive with slow curves. `resample` decides
        what the job is and whether one is needed.
        """
        if plot.item is None or not plot.visible or not resample.wanted(plot):
            return
        plot.generation += 1
        plot.features = None
        generation = plot.generation
        xrange, yrange = self.canvas.viewRange()
        size = (max(self.plot.width(), 100), max(self.plot.height(), 100))
        self.session.sample(
            resample.keyed(self.number, plot),
            resample.job(plot, xrange, yrange, size, self.degrees),
            lambda answer: self._sampled(plot, generation, fresh, answer),
            lambda xs, ys: self._outlined(plot, generation, xs, ys),
        )

    def _outlined(
        self, plot: Drawn, generation: int, xs: np.ndarray, ys: np.ndarray
    ) -> None:
        """Draw the uniform pass while refinement is still running."""
        if plot.generation != generation or plot.item is None:
            return
        plot.item.setData(xs, ys, connect="finite")

    def _sampled(self, plot: Drawn, generation: int, fresh: bool, answer: Any) -> None:
        """Draw the sampled result, or report if there is nothing to draw."""
        if plot.generation != generation or plot.item is None:
            return
        if isinstance(answer, Exception):
            plot.trouble = str(answer)
            self.session.trouble(self.number, plot.label, str(answer))
            self.say(f"{plot.label}: {answer}")
            return
        if plot.kind in FIELDS:
            self._shaded(plot, answer)
        elif plot.kind in PARAMETRIZED:
            self._ridden(plot, answer)
        elif plot.kind is PlotKind.DATA:
            plot.xs, plot.ys = answer
            plot.item.setData(plot.xs, plot.ys)
        else:
            plot.closure, plot.xs, plot.ys = answer
            plot.item.setData(plot.xs, plot.ys, connect="finite")
        if plot.kind not in FIELDS and not np.isfinite(plot.ys).any():
            self._nothing_real(plot)
        elif fresh:
            self._frame_new(plot)
        if self._tracing is not None:
            self._retrace()

    def _ridden(self, plot: Drawn, answer: Any) -> None:
        """Draw a parametric or polar curve, noting if refinement gave up."""
        plot.pair, plot.closure, sampled, plot.trange = answer
        # The typed bounds are now evaluated into `trange`, which the fields show. A
        # bound that failed to evaluate thus reverts.
        plot.bounds = None
        plot.ts, plot.xs, plot.ys = sampled.ts, sampled.xs, sampled.ys
        plot.item.setData(plot.xs, plot.ys, connect="finite")
        self._show_trange()
        if sampled.gave_up is not None:
            self.say(
                f"{plot.label}: gave up refining near"
                f" {plot.parameter} = {roughly(sampled.gave_up)}"
            )

    def _shaded(self, plot: Drawn, answer: Any) -> None:
        """Draw an implicit contour or a region from the evaluated grid.

        A contour is drawn as line segments separated by NaNs. A region is an image
        behind the curves, one pixel per grid cell, stretched over the evaluated
        rectangle.
        """
        plot.closure, xs, ys, values = answer
        plot.grid = (xs, ys, values)
        if plot.kind is PlotKind.IMPLICIT:
            plot.xs, plot.ys = _contour(xs, ys, values)
            plot.item.setData(plot.xs, plot.ys, connect="finite")
            if not plot.xs.size:
                self._nothing_real(plot)
            return
        plot.fill.setImage(_shading(values, plot.color), autoLevels=False)
        plot.fill.setRect(
            QtCore.QRectF(
                float(xs[0]),
                float(ys[0]),
                float(xs[-1] - xs[0]),
                float(ys[-1] - ys[0]),
            )
        )
        plot.fill.setVisible(plot.item.isVisible())

    def _nothing_real(self, plot: Drawn) -> None:
        """Report that a plot has nothing to draw in view.

        An unexplained empty picture is to be avoided. The message names the range, as
        the fix is usually to look elsewhere. Implicit curves get their own message,
        since their grid evaluated but has no solutions in view.
        """
        left, right = self.canvas.viewRange()[0]
        name = plot.abscissa or plot.variable
        span = f"{roughly(left)} ≤ {name} ≤ {roughly(right)}"
        if plot.kind is PlotKind.IMPLICIT:
            self.say(f"{plot.label}: no solutions in view for {span}")
            return
        # The key comes from the control table, rather than being spelled here.
        key = controls.keys(controls.control("view.all", controls.FLAT))[0]
        self.say(f"{plot.label}: no real values for {span} - try {key} to autoscale")

    def _frame_new(self, plot: Drawn) -> None:
        """Autoscale for a new plot that has points, but none of them in view.

        See `actions.Framing.fresh`. A function is framed in y only, since it was
        sampled over the visible x range; parametric curves and data are framed in
        both directions.
        """
        if plot.kind in FIELDS:
            return
        self._apply(
            self._framing.fresh(
                self._where(),
                plot.xs,
                plot.ys,
                plot.kind in FUNCTIONS,
                plot.label,
            )
        )

    def resample(self) -> None:
        """Resample every visible plot for the current range."""
        for plot in self.plots:
            self._start(plot)

    def _ranged(self, *_: Any) -> None:
        """Handle a view change: schedule a resample and reposition axes and marker."""
        self._timer.start()
        self._place_axes()
        if self._tracing is not None:
            self._trace_to(self._trace_x)

    def _place_axes(self) -> None:
        """Keep the axis lines at the origin, or at the nearest edge (see `axis_at`)."""
        (left, right), (low, high) = self.canvas.viewRange()
        across, up = self.canvas.viewPixelSize()
        self.axes[0].setPos(view.axis_at(left, right, across))
        self.axes[1].setPos(view.axis_at(low, high, up))

    # -- the parameter range -----------------------------------------------

    def _ranged_plot(self) -> Drawn | None:
        """The plot the range fields apply to, or None.

        This is the traced curve, if it is parametrized; otherwise the last added
        parametrized plot.
        """
        active = self._active
        if active is not None and active.kind in PARAMETRIZED:
            return active
        for plot in reversed(self.plots):
            if plot.kind in PARAMETRIZED:
                return plot
        return None

    def _show_trange(self) -> None:
        """Show the selected plot's range in the fields, or hide them if none.

        A field being edited is left alone, so a resample doesn't overwrite typing.
        """
        plot = self._ranged_plot()
        for action in self._range_actions:
            action.setVisible(plot is not None)
        if plot is None:
            return
        named = next(
            piece for piece in forms.TRANGE.pieces if piece.name == "parameter"
        )
        self.range_name.setText(named.word.format(parameter=plot.parameter))
        for name, value in (("low", plot.trange[0]), ("high", plot.trange[1])):
            edit = self.range_fields[name]
            if not edit.hasFocus():
                edit.setText(roughly(value))

    def _range_edited(self) -> None:
        """Resample the selected plot over the range typed in the fields.

        The fields accept expressions such as `-π`, parsed with the plot's parse state
        and evaluated on the sampling thread in the plot's context. Text that doesn't
        parse is reverted, as with the 3D domain; a bound that parses but doesn't
        evaluate to a number reverts when sampling finishes.
        """
        plot = self._ranged_plot()
        if plot is None:
            return
        bounds = forms.parsed(
            (self.range_low.text(), self.range_high.text()), plot.state
        )
        if bounds is None:
            self._show_trange()
            self.say(forms.PARAMETER_EXPRESSIONS.format(parameter=plot.parameter))
            return
        plot.bounds = bounds
        self._start(plot)

    # -- framing -----------------------------------------------------------

    def _apply(self, framed: actions.Framed) -> None:
        """Apply the result of a framing command to the view.

        `plot.actions` computes the result; this sets the aspect lock and range and
        shows the message. Without ranges, the view stays where it is.
        """
        self.equal.setChecked(self._framing.equal)
        self.canvas.setAspectLocked(self._framing.equal, ratio=1.0)
        if framed.ranges is not None:
            if framed.ordinate:
                self.canvas.setYRange(*framed.ranges[1], padding=framed.padding)
            else:
                self.canvas.setRange(
                    xRange=framed.ranges[0],
                    yRange=framed.ranges[1],
                    padding=framed.padding,
                )
        if framed.message:
            self.say(framed.message)

    def home(self) -> None:
        """Reset to the default framing: x in [-5, 5], origin centered, equal scales.

        This also relocks equal scales, restoring the framing a window opens with.
        """
        self._apply(
            self._framing.home(
                self._where(), self.canvas.width(), self.canvas.height()
            )
        )

    def set_range(self) -> None:
        """The menu's `Set range...`: type the view's four bounds.

        The window-modal dialog passes its answer to `_take_range`. Like the 3D
        inspector, it is created once and refilled from the view on every open.
        """
        if self._bounds is None:
            self._bounds = Bounds(self)
            self._bounds.accepted.connect(
                lambda: self._take_range(self._bounds.typed)
            )
        self._bounds.refresh()
        self._bounds.open()

    def _take_range(self, texts: tuple[str, str, str, str]) -> None:
        """Parse four typed bounds and evaluate them on the sampling thread.

        Like the range fields, they accept expressions such as `-π`, parsed with the
        last plot's parse state. Nothing is evaluated on the Qt thread.
        """
        nodes = forms.parsed(texts, self._state)
        if nodes is None:
            self.say(forms.EXPRESSIONS)
            return
        context = self._context

        def work(_report: Callable[..., None]) -> Any:
            return tuple(
                evaluate.number(node, context, float("nan")) for node in nodes
            )

        self.session.sample((self.number, "range"), work, self._framed)

    def _framed(self, answer: Any) -> None:
        """Frame the view on the evaluated bounds, or report why they are invalid.

        See `actions.Framing.typed` for the history and aspect lock handling.
        """
        if isinstance(answer, Exception):
            self.say(forms.NOT_NUMBERS)
            return
        trouble = forms.framed(answer)
        if trouble:
            self.say(trouble)
            return
        left, right, bottom, top = answer
        self._apply(self._framing.typed(self._where(), ((left, right), (bottom, top))))
        self.say(forms.showing(left, right, bottom, top))

    def autoscale(self) -> None:
        """Frame all visible plots, for `A` and `View all`.

        Regions are left out, since they are evaluated over whatever the view shows.
        """
        framed = self._framing.everything(
            self._where(),
            [
                (plot.xs, plot.ys)
                for plot in self.plots
                if plot.visible and plot.kind is not PlotKind.REGION
            ],
        )
        self._apply(framed)
        if framed.ranges is None:
            self.canvas.autoRange()

    def _equal_scales(self, checked: bool) -> None:
        """The `1:1` toolbar toggle, which locks or releases equal scales."""
        self._framing.equal = bool(checked)
        self.canvas.setAspectLocked(checked, ratio=1.0)

    def _polar_mode(self, checked: bool) -> None:
        """The `polar` toolbar toggle, a mode of the whole view.

        Turning it on reads every univariate curve as r = f(θ), and turning it off
        restores them; other kinds (parametric, data, implicit, region) are unaffected.
        A curve switched to polar is drawn over one full turn, adjustable in the range
        fields like any polar plot. The grid, axis labels, pointer readout and trace
        all follow the mode.
        """
        self.polar = checked
        self._ruling()
        active = self._active
        for plot in self.plots:
            if not self._reread(plot):
                continue
            # Drop the polar pair, but keep the closure, which is the same f either
            # way. Sampling resets the range to one full turn.
            plot.pair = None
            plot.bounds = None
            plot.ts = np.empty(0)
            if plot is active:
                self._trace_t = 0.0
            self._start(plot)
        self._axis_names()
        self._show_trange()
        self.say(actions.POLAR_ON if checked else actions.POLAR_OFF)

    def _reread(self, plot: Drawn) -> bool:
        """Set `plot`'s kind for the current view mode, returning whether it changed.

        See `view.reread`.
        """
        reread = view.reread(plot.kind, self.polar)
        if reread is None:
            return False
        plot.kind = reread
        return True

    def center_on(self, point: Any) -> None:
        """Double-click: center the view on the clicked point."""
        self._apply(self._framing.centred(self._where(), point.x(), point.y()))

    def zoom(self, factor: float) -> None:
        """Zoom by a factor around the center of the view."""
        self.remember()
        self.canvas.scaleBy((factor, factor))

    def pan(self, dx: float, dy: float) -> None:
        """Pan by fractions of the view, for the arrow keys."""
        self.remember()
        (left, right), (low, high) = self.canvas.viewRange()
        self.canvas.translateBy(x=(right - left) * dx, y=(high - low) * dy)

    # -- view history ------------------------------------------------------

    def remember(self) -> None:
        """Push the current range onto the history before a gesture changes it."""
        self._framing.remember(self._where())

    def step_history(self, direction: int) -> None:
        """Backspace and Shift-Backspace: step back or forward through the history."""
        self._apply(self._framing.stepped(self._where(), direction))

    def _where(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """The current view range, or None if there is none yet."""
        try:
            ranges = self.canvas.viewRange()
        except Exception:
            return None
        return (tuple(ranges[0]), tuple(ranges[1]))  # type: ignore[return-value]

    # -- the pointer -------------------------------------------------------

    def _moved(self, position: Any) -> None:
        """Update the pointer readout, and move the trace marker on a function.

        A marker on a parametrized curve stays put, since the pointer's x isn't a
        parameter value; it moves with the arrow keys or a click.
        """
        if not self.item.sceneBoundingRect().contains(position):
            return
        point = self.canvas.mapSceneToView(position)
        self.readout.setText(self._readout(point.x(), point.y()))
        plot = self._active
        if plot is not None and plot.kind not in PARAMETRIZED:
            self._trace_to(point.x())

    def _readout(self, x: float, y: float) -> str:
        """The pointer readout, as rich text with names dimmer than the numbers.

        In polar mode, r and θ are shown too.
        """
        readings = [("x", x), ("y", y)]
        if self.polar:
            radius, angle = view.polar(x, y, self.degrees)
            readings.append(("r", radius))
            readings.append(("θ", angle))
        return "&nbsp;&nbsp;&nbsp;".join(_named_value(*reading) for reading in readings)

    def clicked(self, point: Any) -> bool:
        """Handle a left click on the canvas, returning True if it hit a curve.

        Clicking near a curve starts tracing it, or switches trace to it.
        """
        plot = self.at(point)
        if plot is None:
            return False
        self._take_hold(plot, point)
        return True

    def _curve_clicked(self, plot: Drawn, ev: Any) -> None:
        """Trace a curve whose stroke was clicked."""
        if plot not in self.plots:
            return
        self._take_hold(plot, self.canvas.mapSceneToView(ev.scenePos()))

    def _take_hold(self, plot: Drawn, point: Any) -> None:
        """Trace `plot` at the clicked point.

        On a function the marker goes to the click's x. A parametric or polar curve is
        traced by parameter, so the click snaps to the nearest sample's parameter.
        """
        if plot.kind not in TRACEABLE:
            return
        self._tracing = self.plots.index(plot)
        self._show_trange()
        if plot.kind in PARAMETRIZED:
            nearest = self._nearest(plot, point)
            self._trace_at(self._trace_t if nearest is None else nearest)
            return
        self._trace_to(point.x())

    def _nearest(self, plot: Drawn, point: Any) -> float | None:
        """The parameter of the sampled point nearest `point`, in pixels."""
        across, up = self._per_pixel()
        if across is None or up is None or not plot.ts.size:
            return None
        away = np.hypot((plot.xs - point.x()) / across, (plot.ys - point.y()) / up)
        if not np.isfinite(away).any():
            return None
        return float(plot.ts[int(np.nanargmin(away))])

    def _per_pixel(self) -> tuple[float | None, float | None]:
        """The view units per pixel on each axis, or None if zero."""
        across, up = self.canvas.viewPixelSize()
        return (across or None), (up or None)

    def at(self, point: Any) -> Drawn | None:
        """The visible curve nearest the point, if within `HIT_PX` pixels.

        A function is measured by its exact value at the point's x; other curves, which
        can double back, by the distance to their nearest sample.
        """
        across, up = self._per_pixel()
        if across is None or up is None:
            return None
        best: tuple[float, Drawn] | None = None
        for plot in self.plots:
            if not plot.visible or not plot.xs.size:
                continue
            if plot.kind in FUNCTIONS:
                value = self._value_at(plot, point.x())
                away = None if value is None else abs(value - point.y()) / up
            else:
                distances = np.hypot(
                    (plot.xs - point.x()) / across, (plot.ys - point.y()) / up
                )
                away = (
                    float(np.nanmin(distances))
                    if np.isfinite(distances).any()
                    else None
                )
            if away is not None and away <= HIT_PX and (best is None or away < best[0]):
                best = (away, plot)
        return None if best is None else best[1]

    def _region_at(self, point: Any) -> Drawn | None:
        """The region containing the point, according to its shading grid.

        Used only by the context menu, after checking curves, so a curve crossing a
        region takes precedence.
        """
        for plot in reversed(self.plots):
            if plot.kind is not PlotKind.REGION or not plot.visible:
                continue
            if plot.grid is None or not plot.grid[0].size or not plot.grid[1].size:
                continue
            xs, ys, inside = plot.grid
            across = int(np.argmin(np.abs(xs - point.x())))
            up = int(np.argmin(np.abs(ys - point.y())))
            if bool(inside[across, up]):
                return plot
        return None

    def _value_at(self, plot: Drawn, x: float) -> float | None:
        """The curve's value at `x`, evaluated with the closure.

        This gives the function's exact value rather than a pixel's. Before the closure
        exists, the nearest sample is used.
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

    @property
    def _active(self) -> Drawn | None:
        """The traced curve, or None while trace is off."""
        if self._tracing is None or self._tracing >= len(self.plots):
            return None
        return self.plots[self._tracing]

    def trace(self) -> None:
        """`T` and `F3`: toggle trace mode."""
        if self._tracing is not None:
            self._trace_off()
            return
        rideable = self._rideable()
        if not rideable:
            return
        self._tracing = rideable[0]
        self._show_trange()
        plot = self.plots[self._tracing]
        if plot.kind in PARAMETRIZED:
            self._trace_at(sum(plot.trange) / 2)
            return
        (left, right) = self.canvas.viewRange()[0]
        self._trace_to((left + right) / 2)

    def _trace_off(self) -> None:
        self._tracing = None
        self._unmark()
        self._show_trange()
        self.say("")

    def _rideable(self) -> list[int]:
        """Indices of the traceable plots: visible plots of a `TRACEABLE` kind."""
        return [
            index
            for index, plot in enumerate(self.plots)
            if plot.visible and plot.kind in TRACEABLE
        ]

    def _retrace(self) -> None:
        """Reposition the marker after a resample."""
        plot = self._active
        if plot is None:
            return
        if plot.kind in PARAMETRIZED:
            self._trace_at(self._trace_t)
        else:
            self._trace_to(self._trace_x)

    def _trace_step(self, backwards: bool, fast: bool) -> None:
        """Left and right while tracing: move the marker along the curve.

        On a function the marker moves by pixels. A parametric or polar curve's
        parameter isn't on the screen, so it moves by a fraction of the parameter range.
        """
        plot = self._active
        if plot is None:
            return
        direction = -1.0 if backwards else 1.0
        if plot.kind in PARAMETRIZED:
            low, high = plot.trange
            share = STEP_FAST_SHARE if fast else STEP_SHARE
            step = direction * share * (high - low)
            self._trace_at(min(max(self._trace_t + step, low), high))
            return
        pixels = NUDGE_FAST_PX if fast else NUDGE_PX
        step = direction * pixels * self.canvas.viewPixelSize()[0]
        self._trace_to(self._trace_x + step)

    def _trace_curve(self, step: int) -> None:
        """Up and down while tracing: switch to the next or previous curve."""
        rideable = self._rideable()
        if not rideable or self._tracing is None:
            return
        at = self._tracing
        here = rideable.index(at) if at in rideable else 0
        self._tracing = rideable[(here + step) % len(rideable)]
        self._show_trange()
        self._retrace()

    def _trace_to(self, x: float) -> None:
        """Put the marker at `x` on the traced curve, and show the values."""
        plot = self._active
        if plot is None:
            return
        if plot.kind in PARAMETRIZED:
            self._trace_at(self._trace_t)
            return
        self._trace_x = x
        value = self._value_at(plot, x)
        if value is None:
            self._lost(plot, plot.variable, x)
            return
        self._marked(plot, x, value, ((plot.variable, x), ("y", value)))
        self.say(
            f"Tracing {plot.named}   {plot.variable} = {reading(x)}"
            f"   y = {reading(value)}"
        )

    def _trace_at(self, t: float) -> None:
        """Put the marker at parameter `t` on the traced curve, and show the values.

        The point is evaluated by the closures at the exact t, not taken from the
        nearest sample.
        """
        plot = self._active
        if plot is None or plot.pair is None:
            return
        self._trace_t = t
        x = _one(plot.pair[0], t)
        y = _one(plot.pair[1], t)
        if not np.isfinite(x) or not np.isfinite(y):
            self._lost(plot, plot.parameter, t)
            return
        self._trace_x = x
        self._marked(plot, x, y, ((plot.parameter, t), ("x", x), ("y", y)))
        self.say(
            f"Tracing {plot.named}   {plot.parameter} = {reading(t)}"
            f"{self._radius(plot, t)}   x = {reading(x)}   y = {reading(y)}"
        )

    def _radius(self, plot: Drawn, t: float) -> str:
        """The readout of a polar curve's r at θ, or "" for other kinds."""
        if plot.kind is not PlotKind.POLAR or plot.closure is None:
            return ""
        return f"   r = {reading(_one(plot.closure, t))}"

    def _lost(self, plot: Drawn, name: str, at: float) -> None:
        """Hide the marker where the curve has no real value, and say so.

        This is how a removable singularity is found, so it is a message rather than an
        error.
        """
        self._unmark()
        self.say(f"Tracing {plot.named}: not real and finite at {name} = {reading(at)}")

    def _unmark(self) -> None:
        """Hide the marker, its hairline and its chip."""
        for item in (self.marker, self.hairline, self.chip):
            item.setVisible(False)

    def _marked(
        self, plot: Drawn, x: float, y: float, readings: tuple[tuple[str, float], ...]
    ) -> None:
        """Show the marker at a point, in the traced curve's color.

        The ring lets the curve show through, the dashed hairline down to the abscissa
        shows the position along the axis, and the chip repeats the status line's
        values near the point. The color shows which curve is traced.
        """
        color = pg.mkColor(plot.color)
        self.marker.setData([x], [y])
        self.marker.setPen(pg.mkPen(color, width=MARKER_WIDTH))
        self.hairline.setData(np.array([x, x]), np.array([y, self.axes[1].value()]))
        self.hairline.setPen(
            pg.mkPen(
                _tinted(plot.color, HAIRLINE_ALPHA),
                width=1,
                style=QtCore.Qt.PenStyle.DashLine,
            )
        )
        self.chip.setHtml(
            "<div style='white-space: pre'>"
            + "<br/>".join(_named_value(*reading) for reading in readings)
            + "</div>"
        )
        self._place_chip(x, y)
        for item in (self.marker, self.hairline, self.chip):
            item.setVisible(True)

    def _place_chip(self, x: float, y: float) -> None:
        """Place the chip beside the point, on the side that keeps it in view.

        It goes up and to the right of the marker, away from the pointer, and flips
        sides near the edges of the canvas.
        """
        (left, right), (low, high) = self.canvas.viewRange()
        across, up = self.canvas.viewPixelSize()
        near_right = x > right - 0.25 * (right - left)
        near_top = y > high - 0.2 * (high - low)
        self.chip.setAnchor((1.0, 1.0) if near_right else (0.0, 1.0))
        if near_top:
            self.chip.setAnchor((1.0, 0.0) if near_right else (0.0, 0.0))
        self.chip.setPos(
            x + (-CHIP_OFFSET_PX if near_right else CHIP_OFFSET_PX) * across,
            y + (-CHIP_OFFSET_PX if near_top else CHIP_OFFSET_PX) * up,
        )

    @property
    def traced(self) -> tuple[float, float] | None:
        """The point under the trace marker, or None."""
        plot = self._active
        if plot is None:
            return None
        if plot.kind in PARAMETRIZED:
            if plot.pair is None:
                return None
            x = _one(plot.pair[0], self._trace_t)
            y = _one(plot.pair[1], self._trace_t)
            return None if not (np.isfinite(x) and np.isfinite(y)) else (x, y)
        value = self._value_at(plot, self._trace_x)
        return None if value is None else (self._trace_x, value)

    @property
    def traced_text(self) -> str | None:
        """The traced point written for the worksheet, as `[x, y]` with six decimals.

        Ctrl-C and Enter both use this text.
        """
        point = self.traced
        if point is None:
            return None
        return pointed(point[0], point[1])

    def send_home(self) -> None:
        """Enter while tracing: send the traced point to the worksheet.

        The point under the marker (possibly a root or extremum found with Tab) is
        appended to the plot's worksheet as a new algebra entry, with the same text
        that Ctrl-C copies.
        """
        plot = self._active
        text = self.traced_text
        if plot is None or text is None:
            return
        self.session.author(plot.worksheet, text)
        self.say(f"Sent {text} to the worksheet")

    # -- feature snapping ---------------------------------------------------

    def snap(self, backwards: bool) -> None:
        """`Tab` and `Shift-Tab`: move the marker to the next or previous feature.

        Features are roots, local extrema and intersections with other curves (see
        `evaluate.features`). They aren't marked on the canvas, only shown by the marker
        and the status bar.

        Only function curves are supported; features of a parametric path are a
        different problem.

        The scan runs as a sampling job, so the first Tab on a curve moves once the
        scan finishes, and later ones move immediately. The window stays responsive
        meanwhile.
        """
        plot = self._active
        if plot is None:
            return
        if plot.kind not in FUNCTIONS or plot.closure is None:
            self.say(f"Tracing {plot.named}: features are found on function curves")
            return
        if plot.features is None:
            self._find_features(plot, backwards)
            return
        self._snapped(plot, backwards)

    def _find_features(self, plot: Drawn, backwards: bool) -> None:
        """Request a feature scan of this curve on the sampling thread, then step.

        The scan evaluates the closure, so it runs as a sampling job. It is keyed by
        window and curve (as in the browser), so a second Tab before the first finishes
        replaces the pending scan.

        The result is cached on the plot for repeated Tabs, until a resample discards
        it.
        """
        others = [
            other
            for other in self.plots
            if other is not plot
            and other.visible
            and other.kind in FUNCTIONS
            and other.closure is not None
        ]
        assert plot.closure is not None
        generation = plot.generation
        crossed = tuple(other.named for other in others)
        # Captured here rather than in the job, since by the time it runs the arrays
        # may have been replaced (see `resample`).
        xs, ys, closure = plot.xs, plot.ys, plot.closure
        closures = [other.closure for other in others]

        def work(_report: Callable[..., None]) -> Any:
            return evaluate.features(xs, ys, closure, closures)

        self.session.sample(
            (self.number, "features", id(plot)),
            work,
            lambda answer: self._scanned(plot, generation, crossed, backwards, answer),
        )

    def _scanned(
        self,
        plot: Drawn,
        generation: int,
        crossed: tuple[str, ...],
        backwards: bool,
        answer: Any,
    ) -> None:
        """Store the finished scan, and take the requested step.

        The result is dropped if the curve was resampled, trace ended, or the marker
        moved to another curve.
        """
        if plot.generation != generation or self._active is not plot:
            return
        if isinstance(answer, Exception):
            self.say(f"{plot.label}: {answer}")
            return
        plot.features = answer
        plot.crossed = crossed
        self._snapped(plot, backwards)

    def _snapped(self, plot: Drawn, backwards: bool) -> None:
        """Move the marker to the next feature, or report that there is none."""
        found = self._next_feature(plot, backwards)
        if found is None:
            self.say(
                f"Tracing {plot.named}: no further feature"
                f" to the {'left' if backwards else 'right'}"
            )
            return
        self._trace_to(found.x)
        self.say(f"Tracing {plot.named}: {self._named_feature(plot, found)}")

    def _next_feature(self, plot: Drawn, backwards: bool) -> Any:
        """The first feature more than a pixel past the marker in the given direction.

        A feature under the marker is skipped, so repeated Tabs keep moving.
        """
        found = plot.features or ()
        step = self.canvas.viewPixelSize()[0] or 0.0
        if backwards:
            behind = [item for item in found if item.x < self._trace_x - step]
            return behind[-1] if behind else None
        ahead = [item for item in found if item.x > self._trace_x + step]
        return ahead[0] if ahead else None

    def _named_feature(self, plot: Drawn, found: Any) -> str:
        """A description of a feature and its location, for the status bar."""
        where = f"at {plot.variable} = {reading(found.x)}"
        if found.kind is not evaluate.CROSSING:
            return f"{found.kind} {where}"
        crossed = plot.crossed
        if 0 <= found.other < len(crossed):
            return f"intersection with {crossed[found.other]} {where}"
        return f"intersection {where}"

    # -- the legend and the context menu ------------------------------------

    def toggle_grid(self) -> None:
        """`G` and the menu's `Grid`: toggle the grid."""
        self._grid = not self._grid
        self._ruling()

    def toggle_legend(self) -> None:
        """`L` and the menu's `Legend`: toggle the legend.

        The state is kept separately from the widget, so a hidden legend stays hidden
        when a plot is added.
        """
        self._legend = not self._legend
        self.legend.setVisible(self._legend and bool(self.plots))

    def _legend_clicked(self, row: int, button: Any) -> None:
        """Handle a legend row click: toggle visibility, or show the row's menu."""
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
        if plot.fill is not None:
            plot.fill.setVisible(plot.item.isVisible())
        self._relabel()
        if plot.visible:
            self._fade_in(plot)
            self._start(plot)
        elif self._tracing == row:
            self._trace_off()

    def _fade_in(self, plot: Drawn) -> None:
        """Fade a shown curve in, rather than showing it in one frame.

        Hiding is instant, so the window's reported state is immediately correct;
        only showing is eased. The opacity is purely visual and nothing reads it.
        """
        running = self._fades.pop(id(plot), None)
        if running is not None:
            running.stop()
        fade = QtCore.QVariantAnimation(self)
        fade.setDuration(FADE_MS)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        fade.valueChanged.connect(lambda value, plot=plot: self._faded(plot, value))
        fade.finished.connect(lambda plot=plot: self._done_fading(plot))
        self._fades[id(plot)] = fade
        fade.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    def _faded(self, plot: Drawn, value: Any) -> None:
        for item in (plot.item, plot.fill):
            if item is not None:
                item.setOpacity(float(value))

    def _done_fading(self, plot: Drawn) -> None:
        """Finish at full opacity, whatever the last frame was."""
        self._faded(plot, 1.0)
        self._fades.pop(id(plot), None)

    def _offer_removal(self, at: Any) -> None:
        """Show the plot-specific menu for a right-clicked legend row.

        See `controls.card`: the canvas menu is largely about the view, so the row gets
        its own menu with the canvas menu's plot entries.
        """
        handlers = self._handlers()
        menu = QtWidgets.QMenu(self)
        for entry in controls.card(self.snapshot()):
            menu.addAction(entry.label, handlers[entry.name])
        menu.exec(at)

    def prepare_menu(self, point: Any) -> None:
        """Record the plot under a canvas right-click, for the menu about to open."""
        self._pointed = self.at(point) or self._region_at(point)

    def _toggle_connected(self) -> None:
        """Toggle whether a data plot's points are connected by lines.

        The choice is sticky: the session applies it to the next data plot and reports
        it to the app, which saves it in the state file.
        """
        plot = self._pointed
        if plot is None or plot.kind is not PlotKind.DATA:
            return
        plot.connected = not plot.connected
        self._redraw_points(plot)
        self.session.adjusted(connected=plot.connected)

    def _set_point_size(self, size: float) -> None:
        """Set a data plot's point size from its menu. Also sticky."""
        plot = self._pointed
        if plot is None or plot.kind is not PlotKind.DATA:
            return
        plot.point_size = size
        self._redraw_points(plot)
        self.session.adjusted(point_size=size)

    def _redraw_points(self, plot: Drawn) -> None:
        """Apply a data plot's style to its item.

        The item is updated rather than recreated, so its legend entry, hit testing and
        visibility are preserved.
        """
        plot.item.setPen(self._pen(plot.color) if plot.connected else None)
        plot.item.setSymbolSize(plot.point_size)

    def _remove_pointed(self) -> None:
        if self._pointed is not None:
            self.remove(self._pointed)
            self._pointed = None

    # -- export ------------------------------------------------------------

    def copy_image(self) -> None:
        """Ctrl-C: copy the picture to the clipboard, in paper colors.

        While tracing, it copies the traced point instead, which can be pasted into
        the algebra window.
        """
        text = self.traced_text
        if text is not None:
            QtWidgets.QApplication.clipboard().setText(text)
            self.say(f"Copied {text}")
            return
        QtWidgets.QApplication.clipboard().setImage(self._photograph())
        self.say("Copied the plot to the clipboard")

    def export(self) -> None:
        """Ctrl-S and the menu's `Export...`: save the picture, in paper colors.

        PNG saves the same image Ctrl-C copies; SVG saves the scene as vectors, for
        printing or scaling. The format is taken from the typed extension, or else
        from the chosen filter, whose extension is then appended. Both include the
        legend names.
        """
        name, chosen = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save the plot",
            f"plot{self.number}.png",
            "PNG image (*.png);;SVG image (*.svg)",
        )
        if not name:
            return
        kind = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if kind not in ("png", "svg"):
            kind = "svg" if "svg" in chosen.lower() else "png"
            name = f"{name}.{kind}"
        if kind == "png":
            self._photograph().save(name)
        else:
            self._draw_svg(name)
        self.say(f"Saved {name}")

    def _draw_svg(self, name: str) -> None:
        """Paint the scene into an SVG file.

        Qt's SVG writer is a paint device, so the curves come out as vectors and the
        axis numbers as text. pyqtgraph's SVG exporter isn't used: it re-parses Qt's
        output to fix coordinates and fails on the closepath token Qt ends rectangles
        with.
        """
        rect = self.item.sceneBoundingRect()
        page = QtCore.QRectF(0, 0, rect.width(), rect.height())
        writer = QtSvg.QSvgGenerator()
        writer.setFileName(name)
        writer.setSize(rect.size().toSize())
        writer.setViewBox(page)
        # The writer defaults to 72 dpi, which would put labels at three quarters of
        # their tick positions.
        writer.setResolution(SVG_DPI)
        with _on_paper(self):
            painter = QtGui.QPainter(writer)
            # The background belongs to the view, not the scene, so paint it here.
            painter.fillRect(page, pg.mkColor("w"))
            self.item.scene().render(painter, page, rect)
            # The page is the item's rectangle moved to the origin, matching the
            # coordinates of the legend's corner.
            self._name_plots(painter)
            painter.end()

    def _photograph(self) -> Any:
        """The scene as an image in paper colors.

        The colors are switched and restored between two paints, so the screen never
        changes.
        """
        with _on_paper(self):
            image = pg.exporters.ImageExporter(self.item).export(toBytes=True)
            painter = QtGui.QPainter(image)
            painter.scale(*([image.height() / max(self.item.height(), 1)] * 2))
            self._name_plots(painter)
            painter.end()
        return image

    def _name_plots(self, painter: Any) -> None:
        """Paint the plot list into the corner of an exported picture.

        The legend is a widget over the canvas, not part of the scene, so exports would
        lack it. The names are painted in paper colors, as in the 3D window. The painter
        must be scaled to the item's coordinates.
        """
        if not self._legend or not self.plots:
            return
        painter.setFont(QtGui.QFont("Helvetica", 9))
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
        corner = self.canvas.geometry().topLeft()
        for index, plot in enumerate(self.plots):
            painter.setPen(pg.mkColor(plot.paper))
            painter.drawText(
                int(corner.x()) + 16, int(corner.y()) + 24 + index * 14, plot.named
            )

    # -- keys --------------------------------------------------------------

    def event(self, ev: Any) -> bool:
        """Handle Tab while tracing, and update pens on a pixel density change.

        Qt handles Tab as focus navigation in `QWidget.event`, before the key handler,
        so while tracing it is intercepted here. Otherwise Qt handles it.
        """
        tabs = (QtCore.Qt.Key.Key_Tab, QtCore.Qt.Key.Key_Backtab)
        if (
            ev.type() == QtCore.QEvent.Type.KeyPress
            and self._tracing is not None
            and ev.key() in tabs
        ):
            self.keyPressEvent(ev)
            return True
        # Pens include the device pixel ratio (see `_pen`), so update them when it
        # changes.
        if ev.type() == QtCore.QEvent.Type.DevicePixelRatioChange:
            for plot in self.plots:
                if plot.item is None:
                    continue
                if plot.kind is not PlotKind.DATA or plot.connected:
                    plot.item.setPen(self._pen(plot.color))
        return bool(super().event(ev))

    def keyPressEvent(self, ev: Any) -> None:
        """Handle the window's keys.

        Keys shown in the menu are dispatched through the menu's own table, so they
        can't fire twice. Key-only commands (history forward, zoom) use a second table,
        matched on the key alone.

        The gestures handled directly are keys whose meaning depends on whether trace
        is on. Esc ends trace, or otherwise closes the window.

        During a demonstration step, other keys are forwarded to the program, since
        the desktop may have given this window focus: Space and Enter advance the
        demonstration, while the window's own keys keep working.
        """
        key = ev.key()
        shift = bool(ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
        keys = QtCore.Qt.Key
        command = self._keyed.get(pressed(ev)) or self._loose.get(key)
        if command is not None:
            command.trigger()
        elif key == keys.Key_Escape:
            if self._tracing is not None:
                self._trace_off()
            else:
                self.close()
        elif key in (keys.Key_Tab, keys.Key_Backtab) and self._tracing is not None:
            self.snap(shift or key == keys.Key_Backtab)
        elif key in (keys.Key_Return, keys.Key_Enter) and self._tracing is not None:
            self.send_home()
        elif key in (keys.Key_Left, keys.Key_Right):
            step = -1 if key == keys.Key_Left else 1
            if self._tracing is not None:
                self._trace_step(key == keys.Key_Left, shift)
            else:
                self.pan(step * actions.PAN_SHARE, 0.0)
        elif key in (keys.Key_Up, keys.Key_Down):
            step = 1 if key == keys.Key_Up else -1
            if self._tracing is not None:
                self._trace_curve(-step)
            else:
                self.pan(0.0, step * actions.PAN_SHARE)
        elif self._demonstrating:
            self._stepping(ev)
        else:
            super().keyPressEvent(ev)
            return
        ev.accept()

    # -- the rest ----------------------------------------------------------

    def say(self, message: str) -> None:
        """Show a message on the status bar."""
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
        )

    def present(self, demonstrating: bool = False) -> None:
        """Show the window and bring it to the front, as a plot landed in it.

        A demonstration step stays on top until `release`, so the full-screen terminal
        running the demonstration doesn't cover it.

        Keyboard focus is left to the desktop, which tends to focus new windows
        anyway; `keyPressEvent` forwards unused keys to the program during a step.
        """
        self._keep_in_front(demonstrating)
        self.show()
        self.raise_()
        self.activateWindow()

    def release(self, finished: bool = False) -> None:
        """End the demonstration, making this an ordinary window again.

        The window is shown again because changing the flag hid it. If the
        demonstration `finished`, the window closes instead; one that was stopped
        leaves its picture, so Esc can be used to keep it.

        Windows not showing a demonstration step are left alone.
        """
        if not self._demonstrating:
            return
        if finished:
            self.close()
            return
        self._keep_in_front(False)
        self.show()

    def _keep_in_front(self, demonstrating: bool) -> None:
        """Set or clear stay-on-top.

        Changing a window flag hides the window, so callers show it again.
        """
        if demonstrating == self._demonstrating:
            return
        self._demonstrating = demonstrating
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, demonstrating)

    def _stepping(self, ev: Any) -> None:
        """Forward a key the window doesn't use to the program.

        This makes "any key to continue" work in the plot window too. Keys the window
        uses keep working, so a gallery can still be traced and zoomed.
        """
        self.session.stepped()
        ev.accept()

    def retitle(self, current: bool | None = None) -> None:
        """Title the window by its contents, marking it if it is the receiver.

        Called without an argument when the plot list changes, keeping the receiver
        mark as it was.
        """
        if current is not None:
            self.current = current
        self.setWindowTitle(
            protocol.titled(
                self.kind,
                tuple(plot.text or plot.label for plot in self.plots),
                self.current,
            )
        )

    def _resized(self, *_: Any) -> None:
        """Handle a canvas resize: resample, and reframe if the view is unmoved.

        The default framing depends on the canvas shape, which is only known after
        layout. A window whose view hasn't been moved is reframed on every resize,
        which gives it its first framing and keeps it at x in [-5, 5] when resized,
        rather than widening as the aspect lock makes room.
        """
        if self._framing.default and self.canvas.width() > 1:
            self.home()
            self._framing.history.clear()
        self._place_legend()
        self._timer.start()

    def changeEvent(self, ev: Any) -> None:
        """Report activation to the session, which makes this window the receiver.

        Activation covers clicks, raises and alt-tab, under any focus policy, including
        focus-follows-mouse.
        """
        if (
            ev.type() == QtCore.QEvent.Type.ActivationChange
            and self.isActiveWindow()
        ):
            self.session.touched(self.number)
        super().changeEvent(ev)

    def closeEvent(self, ev: Any) -> None:
        """Close the range dialog, if open, and report the close to the session."""
        if self._bounds is not None:
            self._bounds.close()
        self.session.closed(self.number)
        super().closeEvent(ev)


#: The kinds trace can follow. Up and down skip the others.
TRACEABLE = frozenset(
    {PlotKind.CURVE, PlotKind.FAMILY, PlotKind.PARAMETRIC, PlotKind.POLAR}
)


class Sheet(QtWidgets.QDialog):
    """A `plot.forms` form, rendered as a dialog.

    The form defines the fields, labels, groups and buttons; this builds the QDialog.
    Both windows' dialogs use it, so they look alike.

    The dialog only collects strings; the caller fills and validates the fields.
    """

    def __init__(self, window: Any, form: forms.Form) -> None:
        super().__init__(window)
        self._window = window
        self.form = form
        self.setWindowTitle(form.heading.format(number=window.number))
        self.fields: dict[str, QtWidgets.QLineEdit] = {}
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        layout.addWidget(self._headed())
        for group in form.groups:
            layout.addWidget(self._grouped(group))
        layout.addWidget(self._buttons())

    def _headed(self) -> QtWidgets.QWidget:
        """The dialog's title and subtitle."""
        holder = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(1)
        title = QtWidgets.QLabel(self.form.title.format(number=self._window.number))
        title.setStyleSheet(f"color: {theme.FIELD_TEXT}; font-weight: 600;")
        subtitle = QtWidgets.QLabel(self.form.subtitle)
        subtitle.setStyleSheet(f"color: {theme.DIM};")
        column.addWidget(title)
        column.addWidget(subtitle)
        return holder

    def _grouped(self, group: forms.Group) -> QtWidgets.QWidget:
        """A group of fields, laid out as a grid.

        Row labels go at the start of each row, with column headings above. Fields
        with their own labels get them in front.
        """
        box = QtWidgets.QGroupBox(group.title)
        grid = QtWidgets.QGridLayout(box)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        headed = any(row.label for row in group.rows)
        for column, head in enumerate(group.heads):
            grid.addWidget(self._head(head), 0, column + (1 if headed else 0))
        for row, line in enumerate(group.rows, start=1 if group.heads else 0):
            if headed:
                grid.addWidget(self._label(line.label), row, 0)
            for column, name in enumerate(line.fields):
                one = self.form.field(name)
                at = column * (1 if group.heads else 2) + (1 if headed else 0)
                if one.label:
                    grid.addWidget(self._label(one.label), row, at)
                grid.addWidget(self._field(one), row, at + (1 if one.label else 0))
        return box

    def _buttons(self) -> QtWidgets.QWidget:
        """The form's buttons, with the primary one highlighted.

        A button box places buttons according to platform convention. A form without
        buttons gets the standard OK and Cancel.
        """
        if not self.form.buttons:
            standard = QtWidgets.QDialogButtonBox.StandardButton
            buttons = QtWidgets.QDialogButtonBox(standard.Ok | standard.Cancel)
            buttons.button(standard.Ok).setObjectName(theme.PRIMARY)
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            return buttons
        roles = {
            forms.Role.APPLY: QtWidgets.QDialogButtonBox.ButtonRole.ApplyRole,
            forms.Role.RESET: QtWidgets.QDialogButtonBox.ButtonRole.ResetRole,
            forms.Role.CLOSE: QtWidgets.QDialogButtonBox.ButtonRole.RejectRole,
        }
        buttons = QtWidgets.QDialogButtonBox()
        for one in self.form.buttons:
            button = buttons.addButton(one.label, roles[one.role])
            if one.primary:
                button.setObjectName(theme.PRIMARY)
        buttons.clicked.connect(
            lambda button: self.answered(self._role(button.text()))
        )
        return buttons

    def _role(self, label: str) -> forms.Role:
        """The role of a button, found by its label."""
        return next(one.role for one in self.form.buttons if one.label == label)

    def answered(self, role: forms.Role) -> None:
        """Handle one of the form's buttons. Does nothing unless overridden."""

    def _field(self, one: forms.Field) -> QtWidgets.QLineEdit:
        edit = theme.field(one.width)
        self.fields[one.name] = edit
        return edit

    def _label(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setStyleSheet(f"color: {theme.TEXT};")
        return label

    def _head(self, text: str) -> QtWidgets.QLabel:
        head = QtWidgets.QLabel(text)
        head.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        head.setStyleSheet(f"color: {theme.DIM};")
        return head

    def said(self, names: Sequence[str]) -> tuple[str, ...]:
        """The texts of the named fields, in order."""
        return tuple(self.fields[name].text() for name in names)

    def fill(self, values: dict[str, float]) -> None:
        """Fill fields with numbers, formatted as the window formats them."""
        for name, value in values.items():
            self.fields[name].setText(roughly(value))


class Bounds(Sheet):
    """The `Set range...` dialog, for typing the view's four edges.

    Like the 3D inspector, it adds precision to what the mouse can do.

    The fields start with the current view, so changing one edge leaves the others
    as they are. They accept expressions such as `-π` and `2π`, which the window
    parses and validates; the dialog just collects four strings.
    """

    #: The four bounds, in the order the window reads them.
    ORDER = ("left", "right", "bottom", "top")

    def __init__(self, window: Window2D) -> None:
        super().__init__(window, forms.RANGE)
        self.refresh()

    def refresh(self) -> None:
        """Fill the fields from the current view."""
        (left, right), (bottom, top) = self._window.canvas.viewRange()
        self.fill(dict(zip(self.ORDER, (left, right, bottom, top))))

    @property
    def typed(self) -> tuple[str, ...]:
        """The texts of the four fields, in `ORDER`."""
        return self.said(self.ORDER)


def _contour(
    xs: np.ndarray, ys: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The zero contour of a grid, as one stroke with NaN gaps between pieces.

    `isocurve` is pyqtgraph's marching squares, which returns array index coordinates,
    converted here to x and y. Connected paths are requested rather than separate
    segments, which would be much slower to draw.
    """
    if values.shape[0] < 2 or values.shape[1] < 2:
        return np.empty(0), np.empty(0)
    across = (float(xs[-1]) - float(xs[0])) / (len(xs) - 1)
    up = (float(ys[-1]) - float(ys[0])) / (len(ys) - 1)
    # Comparisons with NaN are false, so marching squares would treat a hole in the
    # domain as a huge negative region. Replace NaN with a large finite value instead
    # (infinity would make the interpolation NaN); the spurious contour around the
    # hole is removed below.
    real = np.isfinite(values)
    if not real.any():
        return np.empty(0), np.empty(0)
    outside = 1e6 * max(float(np.max(np.abs(values[real]))), 1.0)
    pieces = pg.functions.isocurve(np.where(real, values, outside), 0.0, connected=True)
    drawn_x: list[np.ndarray] = []
    drawn_y: list[np.ndarray] = []
    for path in pieces:
        if len(path) < 2:
            continue
        points = np.asarray(path, dtype=np.float64)
        keep = _surrounded(points, real)
        here = np.where(keep, float(xs[0]) + points[:, 0] * across, np.nan)
        there = np.where(keep, float(ys[0]) + points[:, 1] * up, np.nan)
        drawn_x.append(np.append(here, np.nan))
        drawn_y.append(np.append(there, np.nan))
    if not drawn_x:
        return np.empty(0), np.empty(0)
    return np.concatenate(drawn_x), np.concatenate(drawn_y)


def _surrounded(points: np.ndarray, real: np.ndarray) -> np.ndarray:
    """Which contour points have the function defined at all four surrounding nodes.

    Points next to a hole are the hole's boundary, not real contour points, and are
    dropped: `SQRT(x) = y` begins at the origin, rather than having a wall along the y
    axis.
    """
    highest = (real.shape[0] - 1, real.shape[1] - 1)
    low = [np.clip(np.floor(points[:, n]).astype(int), 0, highest[n]) for n in (0, 1)]
    high = [np.clip(np.ceil(points[:, n]).astype(int), 0, highest[n]) for n in (0, 1)]
    return (
        real[low[0], low[1]]
        & real[high[0], low[1]]
        & real[low[0], high[1]]
        & real[high[0], high[1]]
    )


def _shading(inside: np.ndarray, color: str) -> np.ndarray:
    """A region's translucent fill image from its boolean grid.

    One pixel per grid cell, in the plot's color at `REGION_ALPHA` where the inequality
    holds and transparent elsewhere. It is stretched over the evaluated rectangle, so
    its resolution is the grid's, which is all the accuracy the boundary has.
    """
    tint = pg.mkColor(color)
    image = np.zeros((*inside.shape, 4), dtype=np.ubyte)
    image[..., 0] = tint.red()
    image[..., 1] = tint.green()
    image[..., 2] = tint.blue()
    image[..., 3] = np.where(inside, int(REGION_ALPHA * 255), 0)
    return image


def _one(f: Callable[..., np.ndarray], at: float) -> float:
    """A closure's value at one point as a float, or NaN."""
    try:
        return float(np.asarray(f(np.array([at], dtype=np.float64)))[0])
    except Exception:
        return float("nan")


class _on_paper:
    """Context manager switching the window to paper colors during an export.

    Exports use a white background and darker curve colors, so they work in
    documents. Exports are synchronous, so both switches happen between two paints
    and the screen stays dark.
    """

    def __init__(self, window: Window2D) -> None:
        self._window = window

    def __enter__(self) -> None:
        window = self._window
        self._pens = [(plot, plot.item.opts["pen"]) for plot in window.plots if plot.item]
        window.plot.setBackground("w")
        for plot, _ in self._pens:
            # Unconnected data plots have no stroke, and setting a pen would add one;
            # only their symbols are recolored. Paper pens use plain `CURVE_WIDTH`,
            # since exporters render at logical size without high-DPI scaling.
            if plot.kind is not PlotKind.DATA or plot.connected:
                plot.item.setPen(pg.mkPen(plot.paper, width=CURVE_WIDTH, cosmetic=True))
            if plot.kind is PlotKind.DATA:
                plot.item.setSymbolPen(pg.mkPen(plot.paper))
                plot.item.setSymbolBrush(pg.mkBrush(plot.paper))
            if plot.fill is not None and plot.grid is not None:
                plot.fill.setImage(_shading(plot.grid[2], plot.paper), autoLevels=False)
        window._relabel(paper=True)
        for edge in ("bottom", "left"):
            window.item.getAxis(edge).setTextPen(pg.mkPen("k"))
        for line in window.axes:
            line.setPen(pg.mkPen(PAPER_AXIS))

    def __exit__(self, *_: Any) -> None:
        window = self._window
        window.plot.setBackground(BACKGROUND)
        for plot, pen in self._pens:
            plot.item.setPen(pen)
            if plot.kind is PlotKind.DATA:
                plot.item.setSymbolPen(pg.mkPen(plot.color))
                plot.item.setSymbolBrush(pg.mkBrush(plot.color))
            if plot.fill is not None and plot.grid is not None:
                plot.fill.setImage(_shading(plot.grid[2], plot.color), autoLevels=False)
        window._relabel()
        for edge in ("bottom", "left"):
            window.item.getAxis(edge).setTextPen(pg.mkPen(TEXT_COLOR))
        for line in window.axes:
            line.setPen(pg.mkPen(AXIS_COLOR))


def _tinted(color: str, alpha: float) -> Any:
    """A color at the given opacity."""
    tint = pg.mkColor(color)
    tint.setAlphaF(alpha)
    return tint


def _named_value(name: str, value: float) -> str:
    """A named value as rich text for the status line and trace chip.

    The name is dimmer than the number, which is what is being read.
    """
    return (
        f'<font color="{theme.DIM}">{html.escape(name)}:</font> '
        f'<font color="{theme.READOUT}">{reading(value)}</font>'
    )


