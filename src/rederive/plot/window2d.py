"""The 2D plot window: a canvas, a legend, a status line, and no dialogs.

The picture is a measuring instrument rather than an illustration, and every
decision here follows from that.

Axes are drawn through the origin because that is where mathematics puts them.
pyqtgraph, being a data-chart library, draws them as edge spines; here the edge
items keep their numbers and lose their lines, and two infinite lines cross at
(0, 0) instead. A window framed away from the origin therefore has its numbers
along the edges and no axis lines in sight, which is the honest picture.

Framing is never asked about. A fresh window shows x in [-5, 5] with equal
scales - always, so a circle is round; the `1:1` toggle releases the lock for
the one odd plot, and that choice is deliberately this window's alone rather
than a default the next window inherits. The mouse does the rest, and the one
place exact bounds can be typed is the context menu's `Set range...` - four
fields and no more, deliberately the only one. The one time the window reframes
itself is when the alternative is an empty picture: a curve added with finite
values none of which are in view autoscales y and says so.

Sampling is in screen space and repeats on every view change, debounced. That
is what makes zooming worth doing: a spike narrower than a pixel is not in the
data until the view asks for it, and then it is. The sampling itself happens on
the host's sampling thread - this file never evaluates anything on the Qt
thread except a single point under the trace marker, where the whole point is
that the answer is the function's and not the pixel grid's.

What this window draws is every 2D kind there is: function curves, parametric
pairs, polar curves, data points, implicit contours and shaded regions. They
are one plot list and one set of gestures because they are one picture - a
legend entry hides a region as it hides a curve, and Remove removes either -
and they differ only in what a sample is. A function is sampled over the
abscissa, a parametric curve over its parameter with the error measured as a
distance in the plane, a field over a grid at half the canvas resolution. All
of them re-evaluate on a view change, debounced, through the same keyed job:
zooming into an implicit curve reveals its detail exactly as zooming into a
function does, which is the whole reason the sampling is redone rather than
the pixels stretched.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters  # noqa: F401  - registers the exporters copy and export use
from pyqtgraph.Qt import QtCore, QtGui, QtSvg, QtWidgets

from rederive.engine.context import Angle, Context
from rederive.model.expr import Node
from rederive.plot import evaluate, protocol, theme
from rederive.plot.protocol import Options, PlotKind
from rederive.syntax import DeriveSyntaxError, ParseState, parse_expression

__all__ = ["PALETTE", "Legend", "Plot", "Window2D"]

#: The window's own colors. Near-black rather than black, so that a curve in
#: black-adjacent color still reads and so that the window does not look like
#: a hole in the desktop. What is around the canvas - the bar, the fields, the
#: status line - is the chrome's business and lives in `theme`.
BACKGROUND = "#0c0c10"
AXIS_COLOR = "#909090"
GRID_ALPHA = 0.18
TEXT_COLOR = "#d0d0d0"

#: How wide the stroke that draws mathematics is, in logical pixels. The curve
#: pen, a data plot's connecting polyline, the legend sample drawn with the
#: item's own pen and the export pens all take their width from here, and the
#: 3D window's wire borrows it, so the weight is the same on screen, in the
#: swatch and in a pasted PNG. The furniture keeps its hairlines - axis lines
#: at one pixel, the grid at its alpha - which is what makes a curve read as
#: the subject rather than as more scaffolding.
CURVE_WIDTH = 2.0

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

#: The pixels an exported SVG is measured in, per inch. Qt's SVG writer counts
#: 72 to the inch and every other Qt paint device counts these, so this is what
#: makes the file the size of the picture it was taken from.
SVG_DPI = 96

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

#: The modifiers a key of these windows is written with. What else the keyboard
#: reports about a press - the keypad flag a `0` typed on the numeric pad
#: carries - says nothing about which key it was, and is masked off before the
#: key is looked up in a window's command table.
MODIFIERS = (
    QtCore.Qt.KeyboardModifier.ControlModifier
    | QtCore.Qt.KeyboardModifier.ShiftModifier
    | QtCore.Qt.KeyboardModifier.AltModifier
    | QtCore.Qt.KeyboardModifier.MetaModifier
)

#: How far the arrow keys move the trace marker, in pixels, plain and with
#: Shift; and how far they pan the view, as a fraction of it.
NUDGE_PX = 1.0
NUDGE_FAST_PX = 10.0
PAN_SHARE = 0.25

#: How the numbers in the status line are written. Six decimals is what a
#: reading is worth from a picture: enough to paste into the algebra window
#: and see it agree, short enough to fit twice on one line.
FORMAT = "{:.6f}"

#: The parameter range a parametric or polar plot takes when nothing says
#: otherwise: one turn, written in whichever angle unit the worksheet is set
#: to. One turn is what draws a closed curve of the ordinary ones, and the
#: toolbar's range fields are where a different piece of the parameter is
#: asked for, while the picture is on the screen.
DEFAULT_TURN = np.pi
DEFAULT_TURN_DEGREES = 180.0

#: How far the arrow keys move the marker along a parametric curve, as a
#: fraction of the parameter range, plain and with Shift. A five-hundredth is
#: about a pixel on a curve that crosses the window once.
STEP_SHARE = 1 / 500
STEP_FAST_SHARE = 1 / 50

#: How fine the grid an implicit curve or a region is evaluated on is, as a
#: fraction of the canvas, and the most points either axis of it may have. Half
#: the canvas resolution is the design's own figure: marching squares
#: interpolates within a cell, so a contour drawn from half-resolution samples
#: is smooth at full resolution.
GRID_SHARE = 0.5
MAX_GRID = 512

#: How solid a region's fill is. Enough to read as shading, little enough that
#: a curve crossing it is still a curve.
REGION_ALPHA = 0.25

#: How big a data point is drawn, and the sizes the right-click menu offers.
POINT_SIZE = 5.0
POINT_SIZES = (3.0, 5.0, 8.0, 12.0)

#: How the legend card is laid out, in logical pixels: the length of the color
#: sample, the gap on either side of it, the room the eye affordance takes at
#: the right end of a row, and how tall a row is drawn.
LEGEND_SAMPLE = 18
LEGEND_GAP = 8
LEGEND_EYE = 16
LEGEND_ROW = 19

#: How much of a hidden row is left standing. Dimmed rather than struck
#: through: a hidden curve is a curve that is still in the window, and the row
#: has to read as one of the list rather than as a mistake in it.
LEGEND_FADED = 0.4

#: How long a curve takes to come back when its legend row is clicked. The
#: hiding is instant - what is hidden is hidden the moment it is asked for -
#: and only the return is eased, since that is the half the eye follows.
FADE_MS = 150

#: The trace marker: a ring rather than a filled dot, so the point it names is
#: still visible under it, and the pen that draws the ring. The hairline down
#: to the axis is drawn in the same color at this much of it, and the value
#: chip rides this far from the point so that a pointer over the marker never
#: covers what the marker says.
MARKER_PX = 11.0
MARKER_WIDTH = 2.0
HAIRLINE_ALPHA = 0.45
CHIP_OFFSET_PX = 12.0
CHIP_RADIUS = 4.0

#: How much of an expression a legend entry, a menu item or a status line shows
#: of it. A data matrix is one expression and a hundred pairs of numbers, and a
#: legend entry three times the width of the window names nothing legibly, so a
#: long one is cut and marked as cut. Nothing is lost: the whole expression is
#: in the worksheet, under the label the entry begins with.
NAME_WIDTH = 48
ELLIPSIS = "…"

#: The kinds parametrized by the abscissa, which are the ones a marker rides
#: at an x and the ones features are sought on.
FUNCTIONS = frozenset({PlotKind.CURVE, PlotKind.FAMILY})

#: The kinds parametrized by something else, which are ridden at a t.
PARAMETRIZED = protocol.PARAMETRIZED

#: The kinds evaluated over a grid rather than along a line. They are areas,
#: recomputed for whatever the view now shows.
FIELDS = frozenset({PlotKind.IMPLICIT, PlotKind.REGION})


def naming(label: str, text: str) -> str:
    """What a plot is called wherever a window has room for one line of it.

    The label first, because that is what identifies it in the worksheet, and
    then as much of the expression as fits. Both window kinds name their plots
    this way, which is why the 3D window borrows this rather than spelling it
    again.
    """
    name = f"{label}  {text}" if label else text
    if len(name) <= NAME_WIDTH:
        return name
    return name[: NAME_WIDTH - len(ELLIPSIS)].rstrip() + ELLIPSIS


def commanded(
    owner: Any,
    keyed: dict[tuple[int, int], Any],
    text: str,
    handler: Callable[..., None],
    keys: Sequence[str] = (),
    checkable: bool = False,
) -> Any:
    """One thing a window does, as the menu entry and the keys that do it.

    The keys are written on the action and put in `keyed`, which is the table
    the window's `keyPressEvent` dispatches from. Qt is never asked to press
    them itself: a shortcut of its own would fire beside the ladder and do
    everything twice, and the ladder has to stay in any case, being where the
    keys that mean one thing while a marker is up and another while it is not
    are decided. `WidgetShortcut` on an action no widget owns is therefore
    lettering rather than binding - the menu writes the keys in its shortcut
    column, the ladder presses them, and a key typed into a toolbar field stays
    text in the field it was typed into, since a field with the keyboard takes
    it first.

    Both window kinds build their menus through this, which is what keeps the
    key a menu entry advertises and the key that works the same key.
    """
    action = QtGui.QAction(text, owner)
    action.setCheckable(checkable)
    if keys:
        action.setShortcuts([QtGui.QKeySequence(key) for key in keys])
        action.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetShortcut)
        action.setShortcutVisibleInContextMenu(True)
        for sequence in action.shortcuts():
            stroke = sequence[0]
            keyed[(stroke.keyboardModifiers().value, int(stroke.key()))] = action
    action.triggered.connect(handler)
    return action


def pressed(ev: Any) -> tuple[int, int]:
    """The stroke a key event is, spelled the way a command table is keyed."""
    return ((ev.modifiers() & MODIFIERS).value, int(ev.key()))


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
    #: The image a region shades itself with, which is the one kind that needs
    #: a second item: everything else is a stroke and `item` draws it.
    fill: Any = None
    #: The lambdified closure, once the sampling thread has built one, and
    #: what went wrong if it could not.
    closure: Callable[..., np.ndarray] | None = None
    #: The two closures of a parametric pair, or of a polar curve composed into
    #: one. `closure` is then r(θ), which is what the polar readout says.
    pair: tuple[Callable[..., np.ndarray], Callable[..., np.ndarray]] | None = None
    trouble: str = ""
    #: The samples now drawn, which trace and the feature scan read.
    xs: np.ndarray = field(default_factory=lambda: np.empty(0))
    ys: np.ndarray = field(default_factory=lambda: np.empty(0))
    #: The parameter each of those samples came from, for the kinds that have
    #: one: a marker on a parametric curve lives at a t and not at an x.
    ts: np.ndarray = field(default_factory=lambda: np.empty(0))
    #: The grid a region was last shaded from - the two axis vectors and the
    #: truth values over them - which is what says whether a click landed
    #: inside the region.
    grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    #: The parse state the expression arrived under, which is what the range
    #: fields read a typed bound with: the same grammar the worksheet reads.
    state: ParseState = field(default_factory=ParseState)
    #: The parameter range now drawn over, and the bounds typed in the range
    #: fields that the sampling thread has yet to turn into one - trees, since
    #: what a person types is `-π` and what it is worth is arithmetic.
    trange: tuple[float, float] = (-DEFAULT_TURN, DEFAULT_TURN)
    bounds: tuple[Node, Node] | None = None
    #: The notable points of this curve for the samples now drawn, worked out
    #: when trace first asks for them and dropped whenever the samples change,
    #: beside the labels of the curves the intersections among them are with.
    features: tuple[Any, ...] | None = None
    crossed: tuple[str, ...] = ()
    #: How a data plot is drawn, resolved from the request and the window's
    #: defaults when the plot is added, and changed by its right-click menu.
    connected: bool = False
    point_size: float = POINT_SIZE
    #: Which re-sample these samples came from. A job whose generation has
    #: moved on is a job whose answer is about a view that is gone.
    generation: int = 0

    @property
    def visible(self) -> bool:
        return self.item is not None and self.item.isVisible()

    @property
    def named(self) -> str:
        """How the legend and the status line name this plot."""
        return naming(self.label, self.text)

    @property
    def variable(self) -> str:
        """The name the plot is parametrized by, `x` where it named none.

        The abscissa of a function, the parameter of a parametric pair, the
        angle of a polar curve: what the status line calls the number the
        marker is moved along.
        """
        return self.options.variables[0] if self.options.variables else "x"

    @property
    def parameter(self) -> str:
        """What the status line calls the number the marker is moved along.

        A polar curve's is θ whatever letter the expression happened to spell
        its variable with: the picture is drawn in an angle and a radius, and a
        line reading `x = 0.50   x = 0.11` for the angle and the abscissa would
        be two different numbers under one name.
        """
        return "θ" if self.kind is PlotKind.POLAR else self.variable

    @property
    def abscissa(self) -> str:
        """What this plot would have the horizontal axis called, if anything.

        A parametric curve has an opinion about neither axis - t is not drawn
        anywhere - and a data plot has no variables at all, so both keep quiet
        and let a function in the same window name the axis.
        """
        if self.kind in FUNCTIONS:
            return self.variable
        if self.kind in FIELDS:
            return self.axes[0]
        return ""

    @property
    def axes(self) -> tuple[str, str]:
        """The two variables a field plot is evaluated over, horizontal first.

        An equation or an inequality in one variable still needs two: `y = 3`
        is a horizontal line and `x = 3` a vertical one, and which it is comes
        from where the name sits in the worksheet's variable order. The lone
        variable is paired with the neighbour that order names, and the two are
        put in the order's own order, so the picture follows the convention the
        reader has in mind.
        """
        names = self.options.variables
        if len(names) >= 2:
            return names[0], names[1]
        order = list(self.context.order)
        lone = names[0] if names else "x"
        other = next((name for name in order[:2] if name != lone), "y")
        if lone in order and other in order and order.index(other) < order.index(lone):
            return other, lone
        return lone, other


class Canvas(pg.ViewBox):
    """The view box, with the mouse vocabulary this program wants.

    Four departures from stock. The rubber band lives on the *right* button
    rather than on the left, because left-drag is panning and panning is what
    a hand reaches for first; a right press that does not move is still a
    click, and still opens the context menu. The wheel takes Ctrl and Shift to
    mean one axis. Every gesture that changes the range pushes the range it
    changed onto the window's own history first, so Backspace steps back
    through where the view has been rather than through pyqtgraph's rectangle
    stack, which only knows about rubber bands. And the menu a right click
    raises is the window's own, the stock one being a chart library's.
    """

    def __init__(self, window: Window2D) -> None:
        # No stock menu at all rather than a stock menu kept out of sight: what
        # a right click raises here is built by the window.
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
        """The rubber band, in this window's accent rather than in stock yellow.

        Colored here rather than once at construction because the box is not
        ours: the view box makes it lazily and throws it away again whenever
        the mouse mode changes, so the one place it is certain to exist is the
        moment it is about to be drawn.
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
        """A click on the canvas: the menu, the middle of the view, or a curve.

        A right press let go where it was pressed reaches the view box as a
        click rather than as a drag of no distance, so the menu is raised from
        here as well as from the end of a rubber band that turned out to be a
        click.
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
        """The window's menu, told what the click was pointing at, and nothing else.

        Stock `raiseContextMenu` hands whatever menu it is given to the scene to
        have the plot item's `Plot Options` and the exporter's `Export...`
        appended to it, which is how a menu written here would come up carrying
        entries nobody wrote. The composition is what is overridden: the menu
        that appears is ours, whole.
        """
        self._window.prepare_menu(self.mapToView(ev.pos()))
        self._window.menu.popup(ev.screenPos().toPoint())


class _Row(QtWidgets.QWidget):
    """One line of a legend card: a sample of the color, a name, and an eye.

    Painted rather than laid out of labels, because what a row shows is three
    things in one strip - the stroke the curve is drawn with, what it is
    called, and whether it is being shown - and a painter says that in a dozen
    lines where a stack of styled labels would need a dozen widgets.

    It is transparent to the mouse: the card above answers every gesture, so
    the row the pointer is on and the row a click is about are worked out in
    one place instead of in as many places as there are rows.
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
        """What this row now stands for, and a repaint to say so.

        The width is a minimum rather than a size, so the layout stretches
        every row to the widest of them: the eyes then stand in one column and
        the hover wash is one shape, whatever the names happen to be.
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
        """The affordance that says what a click here would do, while it can do it.

        An open eye on a plot that is being shown and a struck one on a plot
        that is not, drawn only under the pointer: a row of eyes down the card
        would be a row of buttons, and the card is a list of what the window
        holds first and a set of controls second.
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
    """The plot list as a card over the picture, one row per plot.

    A widget floating over the canvas rather than an item inside it, which is
    what lets the two window kinds share one legend: a GL view is not a
    graphics scene and has nothing an item could hang on, and a card that is a
    widget hangs equally well over either.

    It answers the same two gestures it always has, and the whole row is the
    target for both: a left click hides and shows the plot, a right click
    offers the menu the plot's own stroke offers. The eye that appears under
    the pointer is the only new thing, and it says what the click would do
    rather than being a second place to click.

    The rows are a pool that grows and never shrinks, the row for a plot that
    is gone being emptied and hidden rather than deleted. A plot list is
    rebuilt on every add and every removal, and widgets deleted from under a
    live layout are the shortest way to a legend of stale rows.
    """

    #: The row the pointer is over, by index into the plot list, and which
    #: button the click on it was.
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
        #: What the card now says, as the window described it, for anything
        #: that has to draw the same list somewhere else.
        self.entries: tuple[tuple[str, str, bool], ...] = ()
        self._over = -1
        self._paper = False
        self._dress()

    def _dress(self) -> None:
        """The card itself: a translucent ground, a hairline, and round corners."""
        ground = theme.CARD_PAPER if self._paper else theme.CARD
        edge = theme.CARD_PAPER_EDGE if self._paper else theme.CARD_EDGE
        self.setStyleSheet(
            f"QFrame#legend {{ background: {ground}; border: 1px solid {edge};"
            f" border-radius: {theme.CARD_RADIUS}px; }}"
        )

    def rebuild(
        self, entries: Sequence[tuple[str, str, bool]], paper: bool = False
    ) -> None:
        """One row per plot: its name, in its color, dimmed where it is hidden.

        The colors are given rather than looked up, so that an export in paper
        colors rebuilds the card in them: a name in the screen color of a curve
        is a name in white, and white on white paper is a legend that is not
        there.
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
        # The card is not in any layout - it floats over the picture - so its
        # size is its own business, and the rows have none until the layout has
        # run over them.
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
        """Which row the point is in, or -1 where it is in none of them."""
        for index, row in enumerate(self._rows[: self.shown]):
            if row.geometry().contains(point):
                return index
        return -1


class _Chip(pg.TextItem):
    """The traced numbers, in a small dark label riding the marker.

    A text item that draws its own ground, because the stock one draws a
    polygon around its text and what this wants is a chip: rounded, dark
    enough to be read over a curve, and edged by one hairline rather than by a
    frame. The base item's own fill and border are left empty, so what it does
    here is the transform bookkeeping, and the ground is painted under text
    that has not been drawn yet - a child item paints after its parent.
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


class Window2D(QtWidgets.QMainWindow):
    """One top-level 2D plot window and everything that happens inside it."""

    def __init__(self, number: int, host: Any) -> None:
        super().__init__()
        self.number = number
        self.kind = protocol.WindowKind.TWO_D
        self.host = host
        self.plots: list[Plot] = []
        #: Whether this view reads its univariate curves as r = f(θ). A mode
        #: of the picture, flipped by the toolbar toggle, never of one plot.
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
        #: Where the marker sits on a curve parametrized by something other
        #: than the abscissa, which is a parameter value and not an x.
        self._trace_t = 0.0
        self._message = ""
        #: Whether the expressions in this window are written in degrees. Read
        #: off the context of every plot added, since it is the worksheet's
        #: setting and not the window's; false until there is one to read.
        self.degrees = False
        #: The parse state and the context a typed bound is read under: the
        #: last plot's, since a framing belongs to the window and the worksheet
        #: that last plotted into it is its reader.
        self._state = ParseState()
        self._context = Context()
        #: The four bounds as a form, once somebody has asked to type them.
        self._bounds: Bounds | None = None
        self._grid = True
        #: Whether the legend card is shown, remembered rather than read off
        #: the widget: a legend the user put away stays away when the next plot
        #: arrives and brings the card back with it.
        self._legend = True
        #: The opacity animations now running, by the plot each belongs to, so
        #: that a row clicked twice replaces its own animation rather than
        #: racing it.
        self._fades: dict[int, Any] = {}
        #: Whether the view is still the default framing, nobody having moved
        #: it. Every gesture that changes the range clears it, since the
        #: default framing is a thing to be given up rather than returned to.
        self._default = True
        self._build()
        self.retitle()
        self.home()

    # -- construction ------------------------------------------------------

    def _build(self) -> None:
        self.canvas = Canvas(self)
        self.plot = pg.PlotWidget(viewBox=self.canvas, background=BACKGROUND)
        self.item = self.plot.getPlotItem()
        self.item.showGrid(x=True, y=True, alpha=GRID_ALPHA)
        self._strip_spines()
        self._origin_axes()
        self._make_menu()
        self.legend = Legend(self.plot)
        self.legend.picked.connect(self._legend_clicked)
        self.legend.setVisible(False)
        self._trace_items()
        self.setCentralWidget(self._laid_out())
        self.canvas.setAspectLocked(True, ratio=1.0)
        self.canvas.disableAutoRange()
        # No corner auto-range button: framing is never asked about, so auto-range
        # is off for good and the button would offer it back - tooltip-less, and in
        # a loop with screen-space resampling, since every widening refills the view
        # with curve and gives auto-range a reason to widen again.
        self.item.hideButtons()
        self.canvas.sigRangeChanged.connect(self._ranged)
        self.canvas.sigResized.connect(self._resized)
        self.item.scene().sigMouseMoved.connect(self._moved)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(RESAMPLE_DELAY_MS)
        self._timer.timeout.connect(self.resample)
        self.plot.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.resize(760, 560)

    def _trace_items(self) -> None:
        """The three things a traced point is drawn with, made once and hidden.

        A ring in the curve's own color, a dashed hairline from it down to the
        abscissa, and a chip carrying the numbers the status line is saying.
        Three items rather than one because they are three answers to the same
        question - where on the curve, how far above the axis, and worth what -
        and they are shown, colored and hidden together.
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
        """The canvas over a one-line status bar, and a toolbar over both.

        The toolbar holds what belongs to the window rather than to one plot:
        the two toggles a picture is read differently under - equal scales and
        the polar reading - and the clear that starts the picture over. A
        toggle that shows its state is worth more than a menu item that has to
        be opened to be read.
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
        self.polar_toggle = QtGui.QAction("polar", self)
        self.polar_toggle.setCheckable(True)
        self.polar_toggle.setToolTip("Show every curve as r = f(θ)")
        self.polar_toggle.triggered.connect(self._polar_mode)
        bar.addAction(self.polar_toggle)
        bar.addWidget(theme.divider())
        self.clear_action = QtGui.QAction("clear", self)
        self.clear_action.setToolTip("Remove every plot from this window (Del)")
        self.clear_action.triggered.connect(self.clear)
        bar.addAction(self.clear_action)
        theme.dangerous(bar, self.clear_action)
        # The parameter range of the selected parametric or polar plot. Hidden
        # while the window holds no such plot, since a range of nothing is not
        # a control; the actions are kept so showing it back is one call, and
        # the hairline in front of them goes away with the group it divides.
        self.range_name = QtWidgets.QLabel("")
        self.range_low = self._range_field()
        self.range_high = self._range_field()
        self._range_actions = (
            bar.addWidget(theme.divider()),
            bar.addWidget(self.range_name),
            bar.addWidget(self.range_low),
            bar.addWidget(QtWidgets.QLabel(" … ")),
            bar.addWidget(self.range_high),
        )
        for action in self._range_actions:
            action.setVisible(False)
        layout.addWidget(bar)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self._status_line())
        return holder

    def _status_line(self) -> QtWidgets.QWidget:
        """The window's one line: what it has to say, and where the pointer is.

        The message is the window speaking and is written in the quiet ink a
        sentence takes; the readout is a measurement and is written in figures
        that keep their columns while the pointer moves, with the names of the
        coordinates dimmer than the numbers they carry - the numbers are what
        is being read, and `x` is only there to say which one.
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

    def _range_field(self) -> QtWidgets.QLineEdit:
        """One range bound, which the keyboard only reaches when clicked in.

        The keys of this window pan, zoom and trace, and a field that took the
        focus when the window opened would swallow them - the same bargain the
        3D window's domain fields make.
        """
        edit = QtWidgets.QLineEdit()
        edit.setFixedWidth(64)
        edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        edit.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        edit.editingFinished.connect(self._range_edited)
        return edit

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

    def _make_menu(self) -> None:
        """The window's own context menu: what it does, and the keys that do it.

        The stock menu is a chart library's. It offers a mouse mode named after
        how many buttons a mouse has, transforms that would take a finite
        difference of screen samples and call the answer a derivative, and axis
        submenus whose names say nothing about the fields inside them - while
        saying nothing at all about trace, about the history, or about any other
        word this window knows. So it is not pruned but replaced: every entry
        here is something the window does, and each carries the key that does
        the same thing with no menu open, which makes the menu the place the
        keyboard is learned.

        The tail is about whatever the click was pointing at rather than about
        the window, which is why it is written when the menu opens.

        Both of pyqtgraph's menus are switched off rather than merely left
        unused: `Plot Options` and `Export...` belong to the plot item and the
        scene, and stock `raiseContextMenu` appends them to whatever menu is
        raised.
        """
        self.item.setMenuEnabled(False, False)
        #: Every key the menu advertises, by the stroke that presses it. The
        #: table `keyPressEvent` dispatches from.
        self._keyed: dict[tuple[int, int], Any] = {}
        self.menu = QtWidgets.QMenu(self)
        self.menu.aboutToShow.connect(self._read_toggles)
        self._command("Set range...", self.set_range)
        self._command("View all", self.autoscale, ("A",))
        self._command("Home framing", self.home, ("Home", "0"))
        self._command("Back", lambda: self.step_history(-1), ("Backspace",))
        self.menu.addSeparator()
        self._trace_entry = self._command(
            "Trace", self.trace, ("T", "F3"), checkable=True
        )
        self._grid_entry = self._command(
            "Grid", self.toggle_grid, ("G",), checkable=True
        )
        self._legend_entry = self._command(
            "Legend", self.toggle_legend, ("L",), checkable=True
        )
        self.menu.addSeparator()
        self._command("Copy image", self.copy_image, ("Ctrl+C",))
        self._command("Export...", self.export, ("Ctrl+S",))
        self._command("Clear", self.clear, ("Del",))
        self._command("Close", self.close, ("Q", "Ctrl+W"))
        self.menu.addSeparator()
        self._remove_action = self._command("Remove", self._remove_pointed)
        self._data_actions = self._point_actions(self.menu)
        self._pointed: Plot | None = None

    def _command(
        self,
        text: str,
        handler: Callable[..., None],
        keys: Sequence[str] = (),
        checkable: bool = False,
    ) -> Any:
        """One entry of the menu, its keys written on it and in the key table."""
        action = commanded(self, self._keyed, text, handler, keys, checkable)
        self.menu.addAction(action)
        return action

    def _read_toggles(self) -> None:
        """Read the three switched entries off the window as the menu opens.

        Read rather than remembered, because none of the three is this menu's
        alone: trace is taken hold of by clicking a curve and let go of with
        Escape, and a tick that stood for what the menu last did would be wrong
        the first time anything else did it.
        """
        self._trace_entry.setChecked(self._tracing is not None)
        self._grid_entry.setChecked(self._grid)
        self._legend_entry.setChecked(self._legend)

    def _point_actions(self, menu: Any) -> tuple[Any, ...]:
        """The two things a data plot can be asked, wherever it is right-clicked.

        Connecting the points is the difference between a scatter and a
        polyline, and the size is how a hundred points and a hundred thousand
        are both made readable. Both live on the plot's own menu rather than
        in a settings dialog, and both are sticky: the way a data plot is left
        is the way the next one arrives, which the host is told about below.
        """
        connect = menu.addAction("Connect points", self._toggle_connected)
        sizes = menu.addMenu("Point size")
        for size in POINT_SIZES:
            sizes.addAction(
                f"{size:g} px", lambda _=False, size=size: self._set_point_size(size)
            )
        return (connect, sizes.menuAction())

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
        # The angle unit is the worksheet's, and a window reads its pointer out
        # in whatever unit the expressions in it are written in; the grammar and
        # the context a typed bound is read under come from the same place.
        self.degrees = plot.context.angle is Angle.DEGREE
        self._state, self._context = plot.state, plot.context
        # Polar is the view's, not the plot's: a univariate curve arriving in
        # a polar window is read as r = f(θ) from the start, and one spelled
        # polar by the request is read the way this window reads every curve.
        self._reread(plot)
        plot.connected = bool(plot.options.connected)
        plot.point_size = plot.options.point_size or POINT_SIZE
        plot.item = self._made(plot)
        plot.item.setCurveClickable(True, width=int(HIT_PX))
        # The stroke takes the click before the view box sees it, which is the
        # stock hit test doing exactly what is wanted: clicking a curve is how
        # trace mode takes hold of that curve rather than of the nearest one.
        plot.item.sigClicked.connect(
            lambda _item, ev, plot=plot: self._curve_clicked(plot, ev)
        )
        self.item.addItem(plot.item)
        if plot.kind is PlotKind.REGION:
            plot.fill = pg.ImageItem()
            # Explicit rather than inherited from the global config: the array
            # this window builds is indexed [x, y], and which way an image item
            # reads its data is otherwise a process-wide setting.
            plot.fill.setOpts(axisOrder="col-major")
            plot.fill.setZValue(-5)
            self.item.addItem(plot.fill, ignoreBounds=True)
        self.plots.append(plot)
        self._relabel()
        self.retitle()
        self._axis_names()
        self._show_trange()
        self._start(plot, fresh=True)

    def _pen(self, color: str) -> Any:
        """A stroke of `CURVE_WIDTH` logical pixels in the given color.

        Cosmetic, said explicitly: the width holds while the view zooms, which
        is what a curve wants. Qt reads a cosmetic width in device pixels,
        ignoring the high-DPI scale along with the zoom - measured, not
        presumed - so the screen's ratio is multiplied in to keep the weight
        the same to the eye on a 1x and a 2x screen.
        """
        return pg.mkPen(
            color, width=CURVE_WIDTH * self.devicePixelRatioF(), cosmetic=True
        )

    def _made(self, plot: Plot) -> Any:
        """The drawing item for a plot of this kind.

        One item type for all of them, which is not a coincidence: a stroke
        with gaps in it draws a function, a parametric curve and an implicit
        contour equally, and the same item with no pen and a symbol draws
        points. What that buys is that the legend, the click hit test, the
        color and the export path are written once.

        A region has no stroke at all and hangs an image behind this item,
        which is then an empty curve carrying the color the legend reads.
        """
        pen = self._pen(plot.color)
        if plot.kind is PlotKind.DATA:
            item = pg.PlotDataItem(
                np.empty(0),
                np.empty(0),
                pen=pen if plot.connected else None,
                symbol="o",
                symbolSize=plot.point_size,
                # The symbols keep a one-pixel outline: how big a point draws
                # is the point size's own setting, and only the line joining
                # the points takes the curve's weight.
                symbolPen=pg.mkPen(plot.color),
                symbolBrush=pg.mkBrush(plot.color),
                antialias=True,
            )
            # A matrix of a hundred thousand rows is an ordinary thing to plot
            # and must stay as live as a curve: drawing what is in view, at the
            # resolution the view has, is what makes it one.
            item.setDownsampling(auto=True)
            item.setClipToView(True)
            return item
        return pg.PlotDataItem(
            np.empty(0), np.empty(0), pen=pen, connect="finite", antialias=True
        )

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
        """The toolbar's clear: start this picture over.

        The window it acts on is the window the button is drawn in, so there
        is nothing to infer and nothing to report. The range fields go away
        with the plots that owned them, and the axis labels with the curves
        they named.
        """
        for plot in list(self.plots):
            self.remove(plot)
        self._axis_names()

    def _relabel(self, paper: bool = False) -> None:
        """Build the legend again, which is how a removal leaves it right.

        The colors are handed to the card rather than looked up by it, so an
        export in paper colors rebuilds the legend as well: a name in the
        screen color of a curve is a name in white, and white on white paper is
        a legend that is not there.
        """
        self.legend.rebuild(
            [
                (plot.named, plot.paper if paper else plot.color, not plot.visible)
                for plot in self.plots
            ],
            paper=paper,
        )
        # An empty card is a rectangle over the corner of the picture saying
        # nothing; the toggle is remembered rather than read off the widget, so
        # a legend the user put away stays away when the next plot arrives.
        self.legend.setVisible(self._legend and bool(self.plots))
        self._place_legend()

    def _place_legend(self) -> None:
        """Put the card in the top left corner of the canvas, inside the axes.

        The card is a widget over the whole plot widget and the canvas is only
        part of it, so where the corner is depends on how wide the numbers down
        the left edge have made the axis - which changes as the view moves.
        """
        corner = self.plot.mapFromScene(
            self.canvas.mapRectToScene(self.canvas.boundingRect()).topLeft()
        )
        self.legend.move(corner.x() + 12, corner.y() + 12)

    def _axis_names(self) -> None:
        """Label the axes with the variable being plotted against.

        The abscissa carries the name of the variable, which is the whole of
        what a 2D window knows about names; the ordinate carries the label of
        the one curve there is, and nothing once there are several - a stack of
        names down the side of the canvas is what the legend is for.

        The kinds with nothing to say about the axes say nothing: a parametric
        pair is not a plot against t, and a window holding one beside `SIN(x)`
        still has an x axis.
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

    def _start(self, plot: Plot, fresh: bool = False) -> None:
        """Ask the sampling thread for this plot over the visible range.

        Everything expensive is on the other side of this call: the conversion,
        the lambdify, the sampling and the grid all happen on the host's
        sampling thread, and what comes back is arrays. A window whose curves
        are all slow is a window that still pans, zooms and closes while they
        arrive.

        Which work is done is the plot's kind, and the answer it hands back is
        read by the same kind, which is why there is one job and one key for
        all of them: a view change debounces an implicit grid exactly as it
        debounces a curve.
        """
        if plot.item is None or not plot.visible:
            return
        if plot.kind is PlotKind.DATA and plot.xs.size:
            # A matrix of constants is the same matrix at every zoom. Clipping
            # and downsampling are what the view change costs, and the item
            # does both itself.
            return
        plot.generation += 1
        plot.features = None
        generation = plot.generation
        view = self.canvas.viewRange()
        size = (max(self.plot.width(), 100), max(self.plot.height(), 100))
        self.host.sample(
            (self.number, id(plot)),
            self._work(plot, view, size),
            lambda answer: self._sampled(plot, generation, fresh, answer),
            lambda xs, ys: self._outlined(plot, generation, xs, ys),
        )

    def _work(self, plot: Plot, view: Any, size: Any) -> Callable[..., Any]:
        """What the sampling thread is to do for this plot, as one callable.

        Everything it needs is read here, on the Qt thread, and captured: the
        job must not touch the window while it runs, since by the time it does
        run the view may have moved and the plot may be gone.
        """
        (left, right), (low, high) = view
        node, context = plot.node, plot.context
        if plot.kind in FIELDS:
            return _field_work(plot, (left, right), (low, high), size)
        if plot.kind in PARAMETRIZED:
            return _curve_work(plot, (left, right), (low, high), size, self.degrees)
        if plot.kind is PlotKind.DATA:
            return lambda report: evaluate.points(node, context)
        variables = plot.options.variables
        closure = plot.closure

        def work(report: Callable[..., None]) -> Any:
            made = closure
            if made is None:
                made = evaluate.closure(node, context, variables or ("x",))
            xs, ys = evaluate.sample_adaptive(
                made, (left, right), (low, high), size, report=report
            )
            return made, xs, ys

        return work

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

    def _ridden(self, plot: Plot, answer: Any) -> None:
        """A parametric or polar curve, and the note that it gave up refining."""
        plot.pair, plot.closure, sampled, plot.trange = answer
        # The bounds this sampling was asked over are now the range it came
        # back with, and the fields show what they came to - which is also how
        # a bound that would not evaluate announces itself: by reverting.
        plot.bounds = None
        plot.ts, plot.xs, plot.ys = sampled.ts, sampled.xs, sampled.ys
        plot.item.setData(plot.xs, plot.ys, connect="finite")
        self._show_trange()
        if sampled.gave_up is not None:
            self.say(
                f"{plot.label}: gave up refining near"
                f" {plot.parameter} = {_short(sampled.gave_up)}"
            )

    def _shaded(self, plot: Plot, answer: Any) -> None:
        """An implicit contour or a region, over the grid just evaluated.

        A contour is line segments and is drawn by the same item every other
        kind is drawn by, with a NaN between segments so nothing joins them. A
        region is an image behind the curves, one pixel per grid cell, and the
        view box stretches it over the rectangle it was evaluated on.
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

    def _nothing_real(self, plot: Plot) -> None:
        """The message a plot with nothing to draw in view leaves behind.

        An empty picture with no explanation is the named anti-goal, and the
        sentence names the range it was empty over, since the answer is nearly
        always to look somewhere else. An implicit curve gets a sentence of its
        own: its grid did evaluate, and what is missing is a solution in the
        rectangle rather than a real value anywhere.
        """
        left, right = self.canvas.viewRange()[0]
        name = plot.abscissa or plot.variable
        span = f"{_short(left)} ≤ {name} ≤ {_short(right)}"
        if plot.kind is PlotKind.IMPLICIT:
            self.say(f"{plot.label}: no solutions in view for {span}")
            return
        self.say(f"{plot.label}: no real values for {span} - try A to autoscale")

    def _frame_new(self, plot: Plot) -> None:
        """Autoscale for a new plot that has points but none of them in view.

        Exactly when the alternative is an empty picture, and never otherwise:
        a window whose framing moved under every added curve would be a window
        nobody could compare two curves in. The aspect lock goes when it
        happens, since a range chosen to fit is not a range equal scales would
        have given.

        A function is framed in y alone - its x range is the view, which is
        what it was sampled over - while a parametric curve or a matrix of
        points can be off the screen in both directions and is framed in both.
        """
        if plot.kind in FIELDS:
            return
        (left, right), (low, high) = self.canvas.viewRange()
        keep = np.isfinite(plot.xs) & np.isfinite(plot.ys)
        xs, ys = plot.xs[keep], plot.ys[keep]
        if not ys.size:
            return
        inside = (xs >= left) & (xs <= right) & (ys >= low) & (ys <= high)
        if inside.any():
            return
        self._unlock()
        self._default = False
        bottom, top = float(np.min(ys)), float(np.max(ys))
        if plot.kind in FUNCTIONS:
            self.canvas.setYRange(bottom, top, padding=0.1)
            self.say(f"{plot.label}: y autoscaled to fit")
            return
        self.canvas.setRange(
            xRange=(float(np.min(xs)), float(np.max(xs))),
            yRange=(bottom, top),
            padding=0.1,
        )
        self.say(f"{plot.label}: autoscaled to fit")

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

    # -- the parameter range -----------------------------------------------

    def _ranged_plot(self) -> Plot | None:
        """The plot the range fields are about, or None where there is none.

        The traced curve when the marker is riding a parametrized one, since
        taking hold of a curve is how a plot is selected; otherwise the
        parametrized plot added last.
        """
        active = self._active
        if active is not None and active.kind in PARAMETRIZED:
            return active
        for plot in reversed(self.plots):
            if plot.kind in PARAMETRIZED:
                return plot
        return None

    def _show_trange(self) -> None:
        """Put the selected plot's range in the fields, or put the fields away.

        A field being typed in is left alone: a re-sample landing mid-edit
        must not pull the text out from under the keyboard.
        """
        plot = self._ranged_plot()
        for action in self._range_actions:
            action.setVisible(plot is not None)
        if plot is None:
            return
        self.range_name.setText(f"   {plot.parameter}: ")
        for edit, value in (
            (self.range_low, plot.trange[0]),
            (self.range_high, plot.trange[1]),
        ):
            if not edit.hasFocus():
                edit.setText(_short(value))

    def _range_edited(self) -> None:
        """A range field was left: re-sample the selected plot over what it says.

        The fields read expressions rather than floats - `-π` is what a person
        types, and a field that special-cased a constant or two would be a
        parser by stealth - so the text is parsed whole, under the parse state
        the plot arrived with, and what the trees are worth is arithmetic the
        sampling thread does under the plot's own context. A text that does
        not parse is reverted rather than argued with, as a 3D domain is; a
        bound that parses but has no number in it reverts when the sampling
        answers, keeping the range the plot already had.
        """
        plot = self._ranged_plot()
        if plot is None:
            return
        try:
            low = parse_expression(self.range_low.text().strip(), plot.state).node
            high = parse_expression(self.range_high.text().strip(), plot.state).node
        except DeriveSyntaxError:
            self._show_trange()
            self.say(f"The {plot.parameter} bounds are expressions, like -π or 2π")
            return
        plot.bounds = (low, high)
        self._start(plot)

    # -- framing -----------------------------------------------------------

    def home(self) -> None:
        """The default framing: x in [-5, 5], the origin centred, equal scales.

        Both ranges are worked out here rather than left to the aspect lock,
        because the lock is free to satisfy itself by widening either axis, and
        which one it picks is not something the framing should depend on. The
        ordinate is therefore the abscissa scaled by the shape of the canvas,
        which is what equal scales means, and the origin is in the middle of it.
        The scales relock too, so that Home is the framing every window opens
        with and a released `1:1` is a departure from it like any pan or zoom.
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

    def set_range(self) -> None:
        """The menu's `Set range...`: the four bounds of the view, typed.

        The one place in this window where a framing is written instead of
        gestured, and it is a form of four fields rather than a screen of
        settings: a framing is four numbers, and everything else about the view
        is the mouse's. The dialog is window-modal and hands its answer back
        through `_take_range`, which is where the numbers are argued with; it is
        made once and filled from the view every time it is opened, as the 3D
        window's inspector is.
        """
        if self._bounds is None:
            self._bounds = Bounds(self)
            self._bounds.accepted.connect(
                lambda: self._take_range(self._bounds.typed)
            )
        self._bounds.refresh()
        self._bounds.open()

    def _take_range(self, texts: tuple[str, str, str, str]) -> None:
        """Read four typed bounds, or say in what way they were not four.

        The fields read expressions rather than floats, as the toolbar's range
        fields do - `-π` is what a person types - so the text is parsed whole
        under the grammar the last plot arrived with, and what the trees are
        worth is arithmetic the sampling thread does. Nothing in this window
        evaluates on the Qt thread, and a bound is no exception.
        """
        try:
            nodes = tuple(
                parse_expression(text.strip(), self._state).node for text in texts
            )
        except DeriveSyntaxError:
            self.say("The bounds are expressions, like -π or 2π")
            return
        context = self._context

        def work(_report: Callable[..., None]) -> Any:
            return tuple(
                evaluate.number(node, context, float("nan")) for node in nodes
            )

        self.host.sample((self.number, "range"), work, self._framed)

    def _framed(self, answer: Any) -> None:
        """The typed bounds are worth numbers: frame the view on them, or refuse.

        The history is pushed first, so a range typed by mistake is one
        Backspace from the range it replaced, and the equal-scales lock goes the
        way it goes for every other framing that is about fitting: somebody who
        asks for these four numbers is asking for these four numbers.
        """
        if isinstance(answer, Exception) or not all(np.isfinite(v) for v in answer):
            self.say("The bounds have to be worth numbers, like -π or 2π")
            return
        left, right, bottom, top = answer
        if right <= left or top <= bottom:
            self.say("A range runs from a lower bound to a higher one")
            return
        self.remember()
        self._unlock()
        self.canvas.setRange(xRange=(left, right), yRange=(bottom, top), padding=0)
        self.say(
            f"Showing {_short(left)} ≤ x ≤ {_short(right)}"
            f"   {_short(bottom)} ≤ y ≤ {_short(top)}"
        )

    def autoscale(self) -> None:
        """Frame every visible plot, which is what `A` and `View all` do.

        A region has no points to frame - it is evaluated over whatever the
        view shows and so agrees with any framing - and is left out rather
        than counted as nothing.
        """
        self.remember()
        self._unlock()
        framed = [
            plot
            for plot in self.plots
            if plot.visible and plot.kind is not PlotKind.REGION
        ]
        xs = [plot.xs[np.isfinite(plot.ys)] for plot in framed]
        ys = [plot.ys[np.isfinite(plot.ys)] for plot in framed]
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

    def _polar_mode(self, checked: bool) -> None:
        """The `polar` toolbar toggle, which is a property of the view.

        Flipping it reinterprets every univariate curve in the window as
        r = f(θ), and flipping it back restores them; the kinds with no polar
        reading - parametric, data, implicit, region - keep their own. A
        reinterpreted curve is drawn over one full turn, whatever x-range it
        happened to be viewed at, and the toolbar's range fields are where a
        different piece of θ is asked for - the same way for a curve read
        polar as for one born polar. The axis labels, the pointer readout and
        trace all follow the mode.
        """
        self.polar = checked
        active = self._active
        for plot in self.plots:
            if not self._reread(plot):
                continue
            # The polar composition is the reading's own and goes with it; the
            # closure survives, being the same f either way. The range comes
            # back as one full turn when the sampling answers.
            plot.pair = None
            plot.bounds = None
            plot.ts = np.empty(0)
            if plot is active:
                self._trace_t = 0.0
            self._start(plot)
        self._axis_names()
        self._show_trange()
        self.say(
            "Polar: curves are read as r = f(θ)"
            if checked
            else "Polar off: curves are read as y = f(x)"
        )

    def _reread(self, plot: Plot) -> bool:
        """Give `plot` the kind the view mode reads it as, saying if it moved.

        A univariate curve is the one kind with two readings - r = f(θ) while
        the window is polar and y = f(x) while it is not - so those two kinds
        trade places with the toggle and every other kind is its own answer.
        """
        if plot.kind is PlotKind.CURVE and self.polar:
            plot.kind = PlotKind.POLAR
        elif plot.kind is PlotKind.POLAR and not self.polar:
            plot.kind = PlotKind.CURVE
        else:
            return False
        return True

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
        """The pointer moved over the scene: the readout, and trace if it is on.

        A marker riding a parametric curve stays where it is: its place is a
        parameter value, and the pointer's x is not one - the arrow keys and a
        click are how it moves. A marker on a function follows the pointer,
        which is what makes trace feel like pointing at the curve.
        """
        if not self.item.sceneBoundingRect().contains(position):
            return
        point = self.canvas.mapSceneToView(position)
        self.readout.setText(self._readout(point.x(), point.y()))
        plot = self._active
        if plot is not None and plot.kind not in PARAMETRIZED:
            self._trace_to(point.x())

    def _readout(self, x: float, y: float) -> str:
        """Where the pointer is, in the coordinates this window is read in.

        A polar window adds the polar pair, because a picture drawn as r
        against θ is measured in r and θ: reading a rose off the cartesian
        numbers alone would be arithmetic the window is there to save.

        Written as marked-up text rather than plain, so that the names sit
        behind the numbers they belong to: what is being read is the figures,
        and `x` is only there to say which of them is which.
        """
        readings = [("x", x), ("y", y)]
        if self.polar:
            turn = 180.0 / np.pi if self.degrees else 1.0
            readings.append(("r", float(np.hypot(x, y))))
            readings.append(("θ", float(np.arctan2(y, x)) * turn))
        return "&nbsp;&nbsp;&nbsp;".join(_named_value(*reading) for reading in readings)

    def clicked(self, point: Any) -> bool:
        """A left click on the canvas. True where a curve took it.

        Clicking within a few pixels of a curve is how trace mode is entered
        with the mouse, and how the active curve is changed while it is on.
        """
        plot = self.at(point)
        if plot is None:
            return False
        self._take_hold(plot, point)
        return True

    def _curve_clicked(self, plot: Plot, ev: Any) -> None:
        """A click that landed on a curve's own stroke: trace that curve."""
        if plot not in self.plots:
            return
        self._take_hold(plot, self.canvas.mapSceneToView(ev.scenePos()))

    def _take_hold(self, plot: Plot, point: Any) -> None:
        """Trace `plot`, at the place the click was pointing at.

        Which place that is depends on what the curve is parametrized by. A
        function is ridden at an abscissa and the click names one; a parametric
        or polar curve is ridden at a parameter, which a point in the plane
        does not name at all - so the click snaps to the nearest point that was
        actually sampled, and the parameter it came from is where the marker
        goes.
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

    def _nearest(self, plot: Plot, point: Any) -> float | None:
        """The parameter of the sampled point nearest `point`, in pixels."""
        across, up = self._per_pixel()
        if across is None or up is None or not plot.ts.size:
            return None
        away = np.hypot((plot.xs - point.x()) / across, (plot.ys - point.y()) / up)
        if not np.isfinite(away).any():
            return None
        return float(plot.ts[int(np.nanargmin(away))])

    def _per_pixel(self) -> tuple[float | None, float | None]:
        """How much of each axis one pixel is worth, or None where nothing."""
        across, up = self.canvas.viewPixelSize()
        return (across or None), (up or None)

    def at(self, point: Any) -> Plot | None:
        """The visible curve nearest the point, if one is near enough.

        Measured in pixels against the curve as it is drawn, which is the same
        thing the eye is doing: a curve is where its stroke is. A function
        answers with its own value at the pointer's abscissa, which is exact; a
        curve that can double back has no such answer and is measured against
        its samples, in the plane.
        """
        across, up = self._per_pixel()
        if across is None or up is None:
            return None
        best: tuple[float, Plot] | None = None
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

    def _region_at(self, point: Any) -> Plot | None:
        """The region the point is inside, for a click that landed on no curve.

        A region is an area and not a stroke, so nothing is ever within a few
        pixels of one; what it means to point at a region is to point inside
        it, which the grid it was shaded from answers. Asked only by the
        context menu, and after the curves have had their chance, so that a
        curve crossing a region is still the thing a click on it is about.
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

    @property
    def _active(self) -> Plot | None:
        """The curve the marker is riding, or None while trace is off."""
        if self._tracing is None or self._tracing >= len(self.plots):
            return None
        return self.plots[self._tracing]

    def trace(self) -> None:
        """`T` and `F3`: take hold of a curve, or let go of one."""
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
        """Which plots a marker can sit on, by index into the plot list.

        Hidden curves are out, as are the kinds with no parametrized curve to
        ride: a region has no `f(x)` a marker could be at a point of.
        """
        return [
            index
            for index, plot in enumerate(self.plots)
            if plot.visible and plot.kind in TRACEABLE
        ]

    def _retrace(self) -> None:
        """Put the marker back after a re-sample moved the ground under it."""
        plot = self._active
        if plot is None:
            return
        if plot.kind in PARAMETRIZED:
            self._trace_at(self._trace_t)
        else:
            self._trace_to(self._trace_x)

    def _trace_step(self, backwards: bool, fast: bool) -> None:
        """Left and right while tracing, in whatever the curve is drawn against.

        A function is nudged a pixel at a time, since its parameter is the
        abscissa and a pixel is the smallest step that shows. A parametric or
        polar curve is stepped by a fraction of its parameter range instead:
        the parameter is not on the screen, so a step of it cannot be measured
        in pixels, and a five-hundredth of the range is about a pixel on a
        curve that crosses the window once.
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
        """Up and down while tracing: the next curve of the plot list."""
        rideable = self._rideable()
        if not rideable or self._tracing is None:
            return
        at = self._tracing
        here = rideable.index(at) if at in rideable else 0
        self._tracing = rideable[(here + step) % len(rideable)]
        self._show_trange()
        self._retrace()

    def _trace_to(self, x: float) -> None:
        """Put the marker at this abscissa on the active curve, and say so."""
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
            f"Tracing {plot.named}   {plot.variable} = {_number(x)}"
            f"   y = {_number(value)}"
        )

    def _trace_at(self, t: float) -> None:
        """Put the marker at this parameter value, on the curve that has one.

        The point is the two closures' own values at the exact t rather than
        the nearest sample, for the same reason a function's reading is: what a
        plot is for is the number, and the pixel grid is only how it is shown.
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
            f"Tracing {plot.named}   {plot.parameter} = {_number(t)}"
            f"{self._radius(plot, t)}   x = {_number(x)}   y = {_number(y)}"
        )

    def _radius(self, plot: Plot, t: float) -> str:
        """What a polar curve is worth at θ, which is the reading it is read by."""
        if plot.kind is not PlotKind.POLAR or plot.closure is None:
            return ""
        return f"   r = {_number(_one(plot.closure, t))}"

    def _lost(self, plot: Plot, name: str, at: float) -> None:
        """The marker over a place the curve has no point at.

        Riding a curve into such a place is how a removable singularity is
        found, so the message is the answer rather than an error.
        """
        self._unmark()
        self.say(f"Tracing {plot.named}: not real and finite at {name} = {_number(at)}")

    def _unmark(self) -> None:
        """Take the marker, its hairline and its chip off the canvas together."""
        for item in (self.marker, self.hairline, self.chip):
            item.setVisible(False)

    def _marked(
        self, plot: Plot, x: float, y: float, readings: tuple[tuple[str, float], ...]
    ) -> None:
        """The point, marked in the color of the curve it is being read off.

        A ring rather than a dot, so the curve goes on through it; a dashed
        hairline down to the abscissa, because half of what a reading means is
        where along the axis it is; and the chip, which says what the status
        line says without the eye having to travel to the bottom of the window
        for it. All three take the curve's color, which is what says which
        curve is being ridden when there are several.
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
        """Stand the chip beside the point, on the side that keeps it in view.

        Up and to the right of the marker, which is where the pointer is not,
        and around to the other side within a corner of the canvas, where the
        chip would otherwise be written off the edge of the picture.
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
        """The point under the marker, for the key that copies it."""
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
        """The traced point spelled for the worksheet: `[x, y]`, six decimals.

        One spelling for the two routes home, so what Ctrl-C puts on the
        clipboard and what Enter sends up the event channel are the same text.
        """
        point = self.traced
        if point is None:
            return None
        return f"[{_number(point[0])}, {_number(point[1])}]"

    def send_home(self) -> None:
        """Enter and Return while tracing: the traced point into the worksheet.

        The number the status bar names - the point under the marker, or the
        root or extremum Tab just refined - goes up the event channel to be
        appended to the worksheet the plot came from as a new algebra entry,
        ready to compute with. The same text Ctrl-C copies, so the clipboard
        route and this one enter the same expression; the difference is that
        nothing has to be retyped, which is what makes the plot a measuring
        instrument rather than a poster.
        """
        plot = self._active
        text = self.traced_text
        if plot is None or text is None:
            return
        self.host.author(plot.worksheet, text)
        self.say(f"Sent {text} to the worksheet")

    # -- feature snapping ---------------------------------------------------

    def snap(self, backwards: bool) -> None:
        """`Tab` and `Shift-Tab`: the next notable point of the active curve.

        Roots, local extrema and crossings of the other curves, found on the
        samples that are already drawn and refined on the closure, so what the
        status bar names is the function's own root and not the pixel the
        stroke crossed the axis in. Nothing is drawn for them: a feature exists
        under the marker and in the sentence, and a canvas that sprouted dots
        nobody asked for would be a canvas with an opinion.

        Function curves only in this first version - a parametric curve's
        notable points are a different question, being about a path rather than
        about a graph.
        """
        plot = self._active
        if plot is None:
            return
        if plot.kind not in FUNCTIONS or plot.closure is None:
            self.say(f"Tracing {plot.named}: features are found on function curves")
            return
        if plot.features is None:
            self._find_features(plot)
        found = self._next_feature(plot, backwards)
        if found is None:
            self.say(
                f"Tracing {plot.named}: no further feature"
                f" to the {'left' if backwards else 'right'}"
            )
            return
        self._trace_to(found.x)
        self.say(f"Tracing {plot.named}: {self._named_feature(plot, found)}")

    def _find_features(self, plot: Plot) -> None:
        """Work out this curve's notable points for the samples now drawn.

        Kept on the plot until the samples change, since Tab is pressed
        repeatedly and the scan is over every interval of the curve; every
        re-sample throws them away, which is what keeps a feature about the
        picture that is on the screen.
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
        plot.features = evaluate.features(
            plot.xs, plot.ys, plot.closure, [other.closure for other in others]
        )
        plot.crossed = tuple(other.named for other in others)

    def _next_feature(self, plot: Plot, backwards: bool) -> Any:
        """The first feature past the marker, in the direction asked for.

        Past rather than at, and by more than a pixel, so that pressing Tab
        twice moves twice: a feature the marker is already sitting on is where
        it came from and not where it is going.
        """
        found = plot.features or ()
        step = self.canvas.viewPixelSize()[0] or 0.0
        if backwards:
            behind = [item for item in found if item.x < self._trace_x - step]
            return behind[-1] if behind else None
        ahead = [item for item in found if item.x > self._trace_x + step]
        return ahead[0] if ahead else None

    def _named_feature(self, plot: Plot, found: Any) -> str:
        """One clause naming what was found and where, as the status bar says it."""
        where = f"at {plot.variable} = {_number(found.x)}"
        if found.kind is not evaluate.CROSSING:
            return f"{found.kind} {where}"
        crossed = plot.crossed
        if 0 <= found.other < len(crossed):
            return f"intersection with {crossed[found.other]} {where}"
        return f"intersection {where}"

    # -- the legend and the context menu ------------------------------------

    def toggle_grid(self) -> None:
        """`G` and the menu's `Grid`: the ruling under the picture, or none."""
        self._grid = not self._grid
        self.item.showGrid(x=self._grid, y=self._grid, alpha=GRID_ALPHA)

    def toggle_legend(self) -> None:
        """`L` and the menu's `Legend`: the plot list card, or the bare picture.

        Remembered on the window rather than read off the widget, so a legend
        that was put away stays away when the next plot brings a card back.
        """
        self._legend = not self._legend
        self.legend.setVisible(self._legend and bool(self.plots))

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
        if plot.fill is not None:
            plot.fill.setVisible(plot.item.isVisible())
        self._relabel()
        if plot.visible:
            self._fade_in(plot)
            self._start(plot)
        elif self._tracing == row:
            self._trace_off()

    def _fade_in(self, plot: Plot) -> None:
        """Bring a curve back over a moment rather than in one frame.

        The one animation in this window, and it is deliberately half of one:
        hiding is instant, because what the window reports about a plot has to
        be true the moment the click lands, and only the return is eased. The
        opacity is nothing but drawing, so nothing anywhere reads it.
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

    def _faded(self, plot: Plot, value: Any) -> None:
        for item in (plot.item, plot.fill):
            if item is not None:
                item.setOpacity(float(value))

    def _done_fading(self, plot: Plot) -> None:
        """Land on the end of the ramp, whatever the last frame happened to be."""
        self._faded(plot, 1.0)
        self._fades.pop(id(plot), None)

    def _offer_removal(self, at: Any) -> None:
        """The plot's own menu, where a right-click was about a plot and not a view.

        A menu of its own rather than the canvas's, because half of the canvas
        menu is about the view and a click on a legend row is about one plot -
        but the words it does offer are the canvas menu's own words, which is
        what makes the legend and the curve one target with two places to hit
        it.
        """
        plot = self._pointed
        assert plot is not None
        menu = QtWidgets.QMenu(self)
        menu.addAction(f"Remove {plot.named}", self._remove_pointed)
        if plot.kind is PlotKind.DATA:
            self._name_connect(self._point_actions(menu)[0], plot)
        menu.exec(at)

    def prepare_menu(self, point: Any) -> None:
        """Say what a right-click on the canvas was pointing at, if anything."""
        self._pointed = self.at(point) or self._region_at(point)
        self._remove_action.setVisible(self._pointed is not None)
        data = self._pointed is not None and self._pointed.kind is PlotKind.DATA
        for action in self._data_actions:
            action.setVisible(data)
        if self._pointed is not None:
            self._remove_action.setText(f"Remove {self._pointed.named}")
        if data:
            self._name_connect(self._data_actions[0], self._pointed)

    def _name_connect(self, action: Any, plot: Plot | None) -> None:
        """Say which way the connect entry would go, which is the other way."""
        if plot is not None:
            action.setText("Disconnect points" if plot.connected else "Connect points")

    def _toggle_connected(self) -> None:
        """Connect a data plot's points into a polyline, or take the line away.

        The choice is sticky: the host keeps it as what the next data plot is
        drawn with, and reports it to the app, which is the side that carries
        it into a state file.
        """
        plot = self._pointed
        if plot is None or plot.kind is not PlotKind.DATA:
            return
        plot.connected = not plot.connected
        self._redraw_points(plot)
        self.host.adjusted(connected=plot.connected)

    def _set_point_size(self, size: float) -> None:
        """How big a data plot's points are drawn, off its own menu. Sticky too."""
        plot = self._pointed
        if plot is None or plot.kind is not PlotKind.DATA:
            return
        plot.point_size = size
        self._redraw_points(plot)
        self.host.adjusted(point_size=size)

    def _redraw_points(self, plot: Plot) -> None:
        """Put a data plot's own opinions back on its item.

        The item is kept rather than remade so that the legend entry, the hit
        test and the visibility it already has all survive a change of mind
        about how big a point is.
        """
        plot.item.setPen(self._pen(plot.color) if plot.connected else None)
        plot.item.setSymbolSize(plot.point_size)

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
        text = self.traced_text
        if text is not None:
            QtWidgets.QApplication.clipboard().setText(text)
            self.say(f"Copied {text}")
            return
        QtWidgets.QApplication.clipboard().setImage(self._photograph())
        self.say("Copied the plot to the clipboard")

    def export(self) -> None:
        """Ctrl-S and the menu's `Export...`: the picture in a file, on paper colors.

        A name and two formats, which is the whole of the transaction: the PNG
        is the same photograph Ctrl-C takes, written to the file instead of the
        clipboard, and the SVG is the scene itself, for a document that will be
        printed or scaled. A typed extension says which; without one the chosen
        filter does, and its extension is added to the name.

        Either file carries the legend names, written on the way out the same
        way, so the two formats say the same thing about the same picture.
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
        """The scene into an SVG file, drawn onto it rather than exported to it.

        Qt's SVG writer is a paint device like any other, so the scene paints
        itself onto the file and the curves arrive as vectors and the axis
        numbers as text. pyqtgraph's own SVG exporter goes the long way around -
        it re-reads what Qt wrote to correct the coordinates in it - and cannot
        read the closepath token Qt ends a rectangle with, which makes every
        export through it a traceback.
        """
        rect = self.item.sceneBoundingRect()
        page = QtCore.QRectF(0, 0, rect.width(), rect.height())
        writer = QtSvg.QSvgGenerator()
        writer.setFileName(name)
        writer.setSize(rect.size().toSize())
        writer.setViewBox(page)
        # The file is measured in the pixels of the window it is a picture of,
        # and the writer's own idea of a pixel is 72 to the inch: left at that,
        # every number and label lands three quarters of the way to where its
        # tick is.
        writer.setResolution(SVG_DPI)
        with _on_paper(self):
            painter = QtGui.QPainter(writer)
            # The background is the view's rather than an item's, so a render
            # of the scene alone would come out on nothing at all.
            painter.fillRect(page, pg.mkColor("w"))
            self.item.scene().render(painter, page, rect)
            # The page is the item's own rectangle moved to the origin, so the
            # names go on at the coordinates the legend's corner is measured in.
            self._name_plots(painter)
            painter.end()

    def _photograph(self) -> Any:
        """The scene as an image, taken on paper colors and put back after.

        Both swaps happen between two paints, so what the screen shows never
        changes: the picture goes to paper, is taken, and the window is dark
        again before anything is drawn.
        """
        with _on_paper(self):
            image = pg.exporters.ImageExporter(self.item).export(toBytes=True)
            painter = QtGui.QPainter(image)
            painter.scale(*([image.height() / max(self.item.height(), 1)] * 2))
            self._name_plots(painter)
            painter.end()
        return image

    def _name_plots(self, painter: Any) -> None:
        """Write the plot list into the corner of an exported picture.

        The legend is a card floating over the canvas rather than an item in
        the scene, and an export photographs the scene: a picture of three
        curves with no legend in it is a picture that has lost half of what it
        showed. So the names are written on afterwards, in the paper colors the
        curves themselves took, exactly as the 3D window's are. The painter
        arrives scaled to the item's coordinates, which is what the corner the
        legend sits in is measured in.
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
        """Take Tab while a marker wants it, and re-pen on a density change.

        Qt reads Tab as "next widget" in `QWidget.event`, above the key
        handler, so a key binding on it never fires; while trace is on, Tab is
        the marker's and is taken here. Off trace it goes back to being Qt's.
        """
        tabs = (QtCore.Qt.Key.Key_Tab, QtCore.Qt.Key.Key_Backtab)
        if (
            ev.type() == QtCore.QEvent.Type.KeyPress
            and self._tracing is not None
            and ev.key() in tabs
        ):
            self.keyPressEvent(ev)
            return True
        # The pens carry the screen's pixel ratio (see `_pen`), so landing on
        # a screen of another density is the one event that changes what they
        # should be.
        if ev.type() == QtCore.QEvent.Type.DevicePixelRatioChange:
            for plot in self.plots:
                if plot.item is None:
                    continue
                if plot.kind is not PlotKind.DATA or plot.connected:
                    plot.item.setPen(self._pen(plot.color))
        return bool(super().event(ev))

    def keyPressEvent(self, ev: Any) -> None:
        """The keys of this window, the menu's own among them.

        Every key a menu entry advertises is dispatched off the menu's own
        table, so the key that works and the key the menu names are one key and
        neither can fire twice. What is left written out here are the keys no
        entry has: the ones that mean one thing while a marker is up and another
        while it is not, and the ones that are gestures rather than commands.
        """
        key = ev.key()
        shift = bool(ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
        keys = QtCore.Qt.Key
        command = self._keyed.get(pressed(ev))
        if command is not None:
            command.trigger()
        elif shift and key == keys.Key_Backspace:
            self.step_history(1)
        elif key in (keys.Key_Plus, keys.Key_Equal):
            self.zoom(0.5)
        elif key == keys.Key_Minus:
            self.zoom(2.0)
        elif key == keys.Key_Escape and self._tracing is not None:
            self._trace_off()
        elif key in (keys.Key_Tab, keys.Key_Backtab) and self._tracing is not None:
            self.snap(shift or key == keys.Key_Backtab)
        elif key in (keys.Key_Return, keys.Key_Enter) and self._tracing is not None:
            self.send_home()
        elif key in (keys.Key_Left, keys.Key_Right):
            step = -1 if key == keys.Key_Left else 1
            if self._tracing is not None:
                self._trace_step(key == keys.Key_Left, shift)
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
        )

    def retitle(self, current: bool | None = None) -> None:
        """Title the window by what it holds, and say whether the next plot lands here.

        Called with no argument when the plot list changes, so the title tracks
        the contents while the receiver mark stays as it was.
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
        self._place_legend()
        self._timer.start()

    def changeEvent(self, ev: Any) -> None:
        """Activation is the user touching this window, and the host's to know.

        The receiver of the next plot follows the window the user last
        touched, and the activation event is how a click, a raise or an
        alt-tab says so - whatever the platform's focus policy, including one
        that hands focus to whatever the pointer crosses.
        """
        if (
            ev.type() == QtCore.QEvent.Type.ActivationChange
            and self.isActiveWindow()
        ):
            self.host.touched(self.number)
        super().changeEvent(ev)

    def closeEvent(self, ev: Any) -> None:
        """The window manager's business, and the host's to hear about.

        A range dialog left open goes with the picture it was about, since it
        is a window of its own and would otherwise outlive it.
        """
        if self._bounds is not None:
            self._bounds.close()
        self.host.closed(self.number)
        super().closeEvent(ev)


#: The kinds trace can ride. The others have no parametrized curve under the
#: marker, so up and down step over them.
TRACEABLE = frozenset(
    {PlotKind.CURVE, PlotKind.FAMILY, PlotKind.PARAMETRIC, PlotKind.POLAR}
)


class Bounds(QtWidgets.QDialog):
    """The four edges of the view, as four fields: the framing typed out.

    Everything here is reachable with the mouse, and what this adds is
    exactness - the same bargain the 3D window's inspector makes. It is
    deliberately a form and not a screen: left, right, bottom and top are what
    a framing is, and the four are laid out as the picture is, the horizontal
    pair over the vertical one.

    The fields open on the view as it stands, so a dialog opened to change one
    edge is three edges already right, and they read expressions rather than
    floats - `-π` and `2π` are answers here as they are in the toolbar's range
    fields. What is typed is argued with by the window and not here: this one
    collects four strings and closes.
    """

    #: The four bounds, in the order the window takes them, with the label each
    #: is asked for by and where in the two-by-two form it stands.
    FIELDS = (
        ("left", "Left", 0, 0),
        ("right", "Right", 0, 1),
        ("bottom", "Bottom", 1, 0),
        ("top", "Top", 1, 1),
    )

    def __init__(self, window: Window2D) -> None:
        super().__init__(window)
        self._window = window
        self.setWindowTitle(f"Range of plot {window.number}")
        self.fields: dict[str, QtWidgets.QLineEdit] = {}
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        layout.addWidget(self._headed())
        layout.addWidget(self._form())
        layout.addWidget(self._buttons())
        self.refresh()

    def _headed(self) -> QtWidgets.QWidget:
        """What the dialog is about, and what a field will take."""
        holder = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(1)
        title = QtWidgets.QLabel(f"Range - plot {self._window.number}")
        title.setStyleSheet(f"color: {theme.FIELD_TEXT}; font-weight: 600;")
        subtitle = QtWidgets.QLabel("expressions, like -π or 2π")
        subtitle.setStyleSheet(f"color: {theme.DIM};")
        column.addWidget(title)
        column.addWidget(subtitle)
        return holder

    def _form(self) -> QtWidgets.QWidget:
        group = QtWidgets.QGroupBox("Bounds")
        grid = QtWidgets.QGridLayout(group)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        for name, shown, row, column in self.FIELDS:
            label = QtWidgets.QLabel(shown)
            label.setStyleSheet(f"color: {theme.TEXT};")
            grid.addWidget(label, row, column * 2)
            grid.addWidget(self._field(name), row, column * 2 + 1)
        return group

    def _buttons(self) -> QtWidgets.QWidget:
        """The two answers, the one that frames the picture being the accented one."""
        standard = QtWidgets.QDialogButtonBox.StandardButton
        buttons = QtWidgets.QDialogButtonBox(standard.Ok | standard.Cancel)
        buttons.button(standard.Ok).setObjectName(theme.PRIMARY)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        return buttons

    def _field(self, name: str) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit()
        edit.setFixedWidth(90)
        edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.fields[name] = edit
        return edit

    def refresh(self) -> None:
        """Fill the fields from the view as it now stands."""
        (left, right), (bottom, top) = self._window.canvas.viewRange()
        for name, value in (
            ("left", left),
            ("right", right),
            ("bottom", bottom),
            ("top", top),
        ):
            self.fields[name].setText(_short(value))

    @property
    def typed(self) -> tuple[str, str, str, str]:
        """What the four fields say, in the order the window reads them."""
        said = [self.fields[name].text() for name, *_ in self.FIELDS]
        return said[0], said[1], said[2], said[3]


def _curve_work(
    plot: Plot,
    xrange: tuple[float, float],
    yrange: tuple[float, float],
    size: tuple[float, float],
    degrees: bool,
) -> Callable[..., Any]:
    """What the sampling thread does for a parametric or polar curve.

    The two closures are made once and kept on the plot, like a function's one
    is, so that a zoom re-samples without asking sympy anything. The first
    sampling is over one turn, so the picture appears with no question asked;
    bounds typed in the range fields arrive as trees - `-π` is what a person
    types - and what they are worth is arithmetic that need only be done once,
    here, before the range they make goes back to the plot.

    A polar curve is a parametric one composed here rather than at the far end:
    r(θ) is kept beside the pair because the readout and the trace are in r and
    θ, and only the composition knows which closure that is.
    """
    node, context, options = plot.node, plot.context, plot.options
    names = options.variables or ("t",)
    made, radius, known = plot.pair, plot.closure, plot.trange
    pending = plot.bounds
    polar = plot.kind is PlotKind.POLAR
    turn = DEFAULT_TURN_DEGREES if degrees else DEFAULT_TURN

    def work(report: Callable[..., None]) -> Any:
        pair, inner, trange = made, radius, known
        if pair is None:
            if polar:
                # A curve reread polar by the view arrives with its closure
                # already made, and r = f(θ) is the same f.
                if inner is None:
                    inner = evaluate.closure(node, context, names)
                pair = evaluate.polar_pair(inner, degrees)
            else:
                pair = evaluate.pair(node, context, names)
            trange = (-turn, turn)
        if pending is not None:
            low = evaluate.number(pending[0], context, trange[0])
            high = evaluate.number(pending[1], context, trange[1])
            if high > low:
                trange = (low, high)
        sampled = evaluate.sample_curve(
            pair[0], pair[1], trange, xrange, yrange, size, report=report
        )
        return pair, inner, sampled, trange

    return work


def _field_work(
    plot: Plot,
    xrange: tuple[float, float],
    yrange: tuple[float, float],
    size: tuple[float, float],
) -> Callable[..., Any]:
    """What the sampling thread does for an implicit curve or a region.

    Both are fields over the rectangle now in view rather than curves over an
    abscissa, and both are recomputed for whatever the view has become, which
    is what makes zooming into a contour reveal it rather than magnify it.

    Half the canvas resolution is the design's figure and it is not a
    compromise: marching squares interpolates inside a cell, so a contour drawn
    from half-resolution samples is smooth at full resolution, and a grid at
    the full one would cost four times as much to say the same thing.
    """
    node, context = plot.node, plot.context
    names = plot.axes
    made = plot.closure
    implicit = plot.kind is PlotKind.IMPLICIT
    nx = int(min(max(size[0] * GRID_SHARE, 2), MAX_GRID))
    ny = int(min(max(size[1] * GRID_SHARE, 2), MAX_GRID))

    def work(_report: Callable[..., None]) -> Any:
        f = made
        if f is None:
            reading = evaluate.difference if implicit else evaluate.mask
            f = reading(node, context, names)
        xs, ys, values = evaluate.grid_eval(f, xrange, yrange, nx, ny)
        return f, xs, ys, values

    return work


def _contour(
    xs: np.ndarray, ys: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The zero contour of a grid, as one stroke with gaps between its pieces.

    `isocurve` is pyqtgraph's own marching squares and answers in the index
    coordinates of the array, so the axis vectors are what turn a crossing of
    cell 17.4 into an x. Connected paths rather than loose segments, because a
    path is one polyline and an implicit curve drawn as ten thousand two-point
    segments is ten thousand items to antialias.
    """
    if values.shape[0] < 2 or values.shape[1] < 2:
        return np.empty(0), np.empty(0)
    across = (float(xs[-1]) - float(xs[0])) / (len(xs) - 1)
    up = (float(ys[-1]) - float(ys[0])) / (len(ys) - 1)
    # Marching squares compares, and every comparison with a NaN is false,
    # which would read the hole in a domain as one enormous negative region. A
    # number far above every real one says the same thing and still divides -
    # an infinity would make the interpolation along a crossing edge NaN and
    # put the contour nowhere - and the contour it invents around the hole is
    # taken out below.
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
    """Which contour points have the function defined all round them.

    A point of the contour lies on the edge between two cells, so the four grid
    nodes bracketing it are what say whether it is a crossing of the function
    or of the edge of the function's domain. The ones with a hole beside them
    are the boundary of the hole and are dropped: `SQRT(x) = y` is a curve that
    begins at the origin, not a curve with a wall down the y axis.
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
    """A boolean grid as the translucent image a region is filled with.

    One pixel per grid cell, in the plot's own color at a quarter opacity, and
    transparent everywhere the inequality is false. The view box stretches it
    over the rectangle it was evaluated on, so the resolution of the fill is
    the resolution of the grid and not of the screen - which is exactly right:
    the boundary of a region is only known to the grid's accuracy, and a fill
    drawn sharper than that would be claiming more than was computed.
    """
    tint = pg.mkColor(color)
    image = np.zeros((*inside.shape, 4), dtype=np.ubyte)
    image[..., 0] = tint.red()
    image[..., 1] = tint.green()
    image[..., 2] = tint.blue()
    image[..., 3] = np.where(inside, int(REGION_ALPHA * 255), 0)
    return image


def _one(f: Callable[..., np.ndarray], at: float) -> float:
    """A closure at one point, as a float, NaN where it has no value there."""
    try:
        return float(np.asarray(f(np.array([at], dtype=np.float64)))[0])
    except Exception:
        return float("nan")


class _on_paper:
    """The window in paper colors for as long as an export takes.

    A dark plot pasted into a document is a black rectangle, so every image
    export sets a white background, redraws the curves in the colors that read
    on it, exports, and puts the window back. Every export is synchronous, so
    both swaps fall between two paints and the screen never shows either: the
    file is on paper and the window on screen stays dark throughout.
    """

    def __init__(self, window: Window2D) -> None:
        self._window = window

    def __enter__(self) -> None:
        window = self._window
        self._pens = [(plot, plot.item.opts["pen"]) for plot in window.plots if plot.item]
        window.plot.setBackground("w")
        for plot, _ in self._pens:
            # A data plot with its points loose has no stroke to recolor, and
            # giving it one would draw a polyline nobody asked for; its symbols
            # are what have to change color, and a region's shading is.
            # The paper pens take `CURVE_WIDTH` plain: the exporters render the
            # scene at its logical size with no high-DPI scale, so the plain
            # constant is what puts the screen's weight in the file.
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
            line.setPen(pg.mkPen("#404040"))

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
    """A color at a fraction of its own opacity, for a wash or a hairline."""
    tint = pg.mkColor(color)
    tint.setAlphaF(alpha)
    return tint


def _named_value(name: str, value: float) -> str:
    """One reading, as the status line and the trace chip both write it.

    The name behind the number: what is being read is the figure, and the
    letter in front of it is only there to say which figure it is.
    """
    return (
        f'<font color="{theme.DIM}">{html.escape(name)}:</font> '
        f'<font color="{theme.READOUT}">{_number(value)}</font>'
    )


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
