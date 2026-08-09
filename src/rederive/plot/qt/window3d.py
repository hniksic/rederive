"""The 3D plot window: several surfaces in one solid, and a camera that costs nothing.

A surface is read by turning it over, so the whole of this window is arranged
around making that free. The mesh is built once per domain and grid and never
again: every orbit, every wheel click, every preset is a camera move over
vertices that are already on the card, and nothing the mouse can do starts an
evaluation. Only the two toolbar fields - the domain and the grid - do that,
because they are the two things that change what was computed rather than where
it is looked at from.

**The shape of the picture is `plot.surface`'s and not this file's.** The box a
surface stands in, its triangles, its wire grid and its shading are arrays and
belong to nobody's toolkit; what is here is a `GLMeshItem` fed with them, and
the numbers along the box edges. The one thing worth repeating from there:
`GLSurfacePlotItem` could not have drawn any of it - it generates the full grid
of faces from the array's shape, and what it does with a NaN vertex is not a
contract anyone has written down - so the triangles are built there and drawn
by a plain `GLMeshItem` here. A dome with a skirt of garbage triangles around
it is the picture that gets drawn when you trust it.

Because there is only ever one box, near enough, to look at, `Home` is a fixed
camera rather than a computation.

**A wire surface hides what is behind it.** Derive's plotter drew a grid with
its hidden lines removed, and a see-through grid is a shape a reader has to
solve rather than see - the far side of a dome comes forward, and two surfaces
in one window are one fabric. So the wire is drawn over its own solid, painted
in the color of the canvas and shoved a hair back by the polygon offset: an
invisible body for the depth buffer to hide the lines behind, this surface's
lines and every other item's alike.

**A window that cannot get an OpenGL context says so.** The context is asked for
when the window is first shown, which is inside the Qt event loop, and an
exception there would leave the process dead and every 2D window with it. It is
caught and put on the status bar instead: no picture, but a plot list, a title
and a message naming what happened.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from OpenGL import GL
from pyqtgraph.opengl.GLGraphicsItem import GLOptions
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from rederive.engine.context import Context
from rederive.plot import actions, controls, evaluate, forms, protocol, resample
from rederive.plot.appearance import (
    BACKGROUND,
    BOX_COLOR,
    CLICK_SLOP_PX,
    CURVE_WIDTH,
    EDGE_ON,
    LABEL_OUT,
    NAME_OUT,
    TICK_COLOR,
    TICK_OUT,
    WIRE_OFFSET,
)
from rederive.plot.model import SOLID_PALETTE, SOLID_PAPER, Surface, written
from rederive.plot.qt import theme
from rederive.plot.qt.window2d import (
    Legend,
    Sheet,
    buttoned,
    commanded,
    loosened,
    pressed,
    rendered,
)
from rederive.plot.surface import (
    HALF,
    TICKS,
    WORLD,
    Box,
    brightened,
    extent,
    mesh,
    ticks,
    wire,
)
from rederive.plot.view import DEFAULT_DOMAIN, DEFAULT_GRID, DEFAULT_ZRANGE
from rederive.syntax import ParseState

__all__ = ["Drawn", "Window3D"]

#: What the numbers along the box edges are written in, which is the 2D
#: window's own ink. The ground, the box and the tick marks are
#: `plot.appearance`', which the page's panes draw from as well, so that a
#: window and a pane side by side are two pictures of one program.
TEXT_COLOR = "#d0d0d0"

#: The same three, for the white background every image export is taken on.
PAPER_BOX = (40, 40, 40, 230)
PAPER_TICK = (60, 60, 60, 255)
PAPER_TEXT = "#000000"

#: The GL state the wire's shove is asked for in - the stock opaque state,
#: which is what every other item here draws under, and the polygon offset
#: over it.
WIRE_OCCLUDER = {
    **GLOptions["opaque"],
    GL.GL_POLYGON_OFFSET_FILL: True,
    "glPolygonOffset": WIRE_OFFSET,
}

#: The most tick marks an axis may draw - the pool of text items is made once
#: and never grown, since items cannot be added to the view while it is
#: painting.
MAX_TICKS = 9

#: What a window with no usable OpenGL says instead of drawing. The reason is
#: the toolkit's own words, which are the only ones that name a missing driver.
NO_OPENGL = "3D drawing is not available: {reason}"


@dataclass
class Drawn(Surface):
    """A surface as this window holds it: the items that draw it and its grid.

    The identity, the look and the expression are the surface's own and come
    from the session; everything added here belongs to the side that evaluates
    and draws.

    `values` is the grid the closure last answered with, kept because the mesh
    is rebuilt whenever the z extent moves - another surface arriving in the
    same window changes what the cube holds without changing what this surface
    is worth.
    """

    item: Any = None
    #: The line item the wire look draws with, made beside `item` and shown
    #: over it while `wire` is on - `item` stays, as the shape the lines are
    #: hidden behind.
    wires: Any = None
    #: The lambdified closure, once the sampling thread has built one.
    closure: Callable[..., np.ndarray] | None = None
    xs: np.ndarray = field(default_factory=lambda: np.empty(0))
    ys: np.ndarray = field(default_factory=lambda: np.empty(0))
    values: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    #: Where the surface stops being real, refined along the grid's edges on
    #: the sampling thread and cached beside the arrays it was found on: the
    #: mesh reads it to end at the boundary rather than a grid step short.
    boundary: evaluate.Boundary | None = None
    #: Which evaluation these values came from. A job whose generation has
    #: moved on is a job about a domain that is gone.
    generation: int = 0

    @classmethod
    def of(cls, surface: Surface) -> Drawn:
        """This window's record of a surface the session has just handed it."""
        return cls(
            **{field.name: getattr(surface, field.name) for field in fields(Surface)}
        )


class View(gl.GLViewWidget):
    """The GL view, saying when the camera has moved and surviving no GL at all.

    Three departures from stock, all of them about what happens around the
    drawing rather than to it. The camera signal is what the tick labels
    re-anchor on: every stock gesture goes through `orbit`, `pan`,
    `setCameraPosition` or the wheel, so overriding those four is a complete
    account of the camera moving, and it is emitted after the move rather than
    during it.

    The second is the right button, which the stock view has no use for at all:
    it orbits on the left and pans on the middle, and a press of the right one
    is recorded and forgotten. So the button is free, and a press let go where
    it was pressed asks for the window's menu - the same click the 2D canvas
    answers, measured by the same slop, so that one hand opens either window's
    menu the same way.

    The third is the guard. `initializeGL` is where a machine with no usable
    OpenGL says so, and it runs inside the Qt event loop, where an exception
    ends the process and takes every other plot window with it. It is remembered
    instead, and the window reports it on its status bar.
    """

    moved = QtCore.Signal()

    #: A right click that stayed put, at the point on the screen it landed on.
    asked = QtCore.Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.broken = ""
        #: Where the right button went down, while it is down.
        self._pressed: Any = None

    def initializeGL(self) -> None:
        try:
            super().initializeGL()
        except Exception as error:
            self.broken = str(error).strip().splitlines()[0]

    def paintGL(self, *arguments: Any, **keywords: Any) -> None:
        if self.broken:
            return
        try:
            super().paintGL(*arguments, **keywords)
        except Exception as error:  # a card that gave out mid-frame
            self.broken = str(error).strip().splitlines()[0]

    def orbit(self, *arguments: Any, **keywords: Any) -> None:
        super().orbit(*arguments, **keywords)
        self.moved.emit()

    def pan(self, *arguments: Any, **keywords: Any) -> None:
        super().pan(*arguments, **keywords)
        self.moved.emit()

    def setCameraPosition(self, *arguments: Any, **keywords: Any) -> None:
        super().setCameraPosition(*arguments, **keywords)
        self.moved.emit()

    def wheelEvent(self, ev: Any) -> None:
        super().wheelEvent(ev)
        self.moved.emit()

    def mousePressEvent(self, ev: Any) -> None:
        """Stock, plus the beginning of a click that may be asking for the menu."""
        super().mousePressEvent(ev)
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            self._pressed = ev.position()

    def mouseReleaseEvent(self, ev: Any) -> None:
        """A right button let go where it went down is the menu being asked for.

        Let go anywhere else it is nothing, which is what a right drag is here
        already: the stock view moves the camera on the other two buttons, so
        nothing is being taken away from the mouse to pay for the menu.
        """
        super().mouseReleaseEvent(ev)
        if ev.button() != QtCore.Qt.MouseButton.RightButton or self._pressed is None:
            return
        moved = ev.position() - self._pressed
        self._pressed = None
        if float(np.hypot(moved.x(), moved.y())) < CLICK_SLOP_PX:
            self.asked.emit(ev.globalPosition().toPoint())

    def mouseMoveEvent(self, ev: Any) -> None:
        """Stock, plus Shift-drag as a second way to say pan.

        The stock view pans on Ctrl-drag and on the middle button; Shift is the
        modifier the rest of this program uses for "the other axis", and a hand
        already holding Shift for a 2D window should not have to learn a second
        key for the same gesture in a 3D one.
        """
        modifiers = ev.modifiers()
        shift = modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier
        control = modifiers & QtCore.Qt.KeyboardModifier.ControlModifier
        if shift and not control and ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
            here = ev.position()
            moved = here - getattr(self, "mousePos", here)
            self.mousePos = here
            self.pan(moved.x(), moved.y(), 0, relative="view")
            return
        super().mouseMoveEvent(ev)


class Window3D(QtWidgets.QMainWindow):
    """One top-level 3D plot window and everything that happens inside it."""

    def __init__(
        self, number: int, session: Any, *, grid: int = DEFAULT_GRID, wire: bool = True
    ) -> None:
        super().__init__()
        self.number = number
        self.kind = protocol.WindowKind.THREE_D
        self.session = session
        self.plots: list[Drawn] = []
        self.current = False
        self.xdomain = DEFAULT_DOMAIN
        self.ydomain = DEFAULT_DOMAIN
        # The grid the window opens with is the sticky one the session holds -
        # the grid the last surface was given - clamped as a typed one is: a
        # window never samples finer than it can draw, whoever asked.
        counted = forms.counts((str(grid), str(grid)))
        self.grid = counted if counted is not None else (DEFAULT_GRID, DEFAULT_GRID)
        #: How a surface arriving here is drawn - the sticky look the last
        #: surface anywhere was left in, and thereafter whatever this window's
        #: own `mesh` box says.
        self.wired = bool(wire)
        #: The z range now drawn, and the one the inspector nailed down where
        #: somebody typed it - a typed extent is an answer and is not autoscaled
        #: away by the next surface.
        self.zrange = DEFAULT_ZRANGE
        #: What the three axes are called, which is the first surface's reading
        #: of the expression until there is one.
        self.axis_names = ("x", "y", "z")
        self._fixed: tuple[float, float] | None = None
        #: The parse state and context the domain fields read typed bounds
        #: under: the last added surface's, since the domain belongs to the
        #: window and the worksheet that last plotted into it is its reader.
        self._state = ParseState()
        self._context = Context()
        #: Which domain edit is the latest, so the numbers a superseded edit
        #: evaluated to cannot land on top of a newer one's.
        self._edit = 0
        self._counter = 0
        self._message = ""
        #: Which of the three furnishings are shown: the box and its ticks, the
        #: numbers and axis names, the legend. All on for a fresh window.
        self._boxed = True
        self._named = True
        self._legend = True
        self._papered = False
        self._inspector: Inspector | None = None
        #: The surface a right click on a legend row was about, which is what
        #: that row's own menu is then read for.
        self._pointed: Drawn | None = None
        #: Whether a demonstration's step is waiting in this window: it is kept
        #: above the others while one is, and every key in it belongs to the
        #: program that is running the demonstration rather than to this window.
        self._demonstrating = False
        self._build()
        self.retitle()
        self.home()

    # -- construction ------------------------------------------------------

    def _build(self) -> None:
        self.view = View()
        self.view.setBackgroundColor(BACKGROUND)
        # The keys belong to the window: the stock view has arrow bindings of
        # its own with a repeat timer, and ours orbit by a fixed step.
        self.view.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.view.moved.connect(self._anchor)
        self.view.asked.connect(self._menu_at)
        self.legend = Legend(self.view)
        self.legend.move(12, 12)
        self.legend.picked.connect(self._legend_picked)
        self._make_menu()
        self._furnish()
        self.setCentralWidget(self._laid_out())
        self._spin = QtCore.QTimer(self)
        self._spin.setInterval(actions.SPIN_MS)
        self._spin.timeout.connect(lambda: self.view.orbit(-actions.SPIN_DEGREES, 0.0))
        self.resize(760, 620)

    def _laid_out(self) -> QtWidgets.QWidget:
        """The view between a toolbar and a status line.

        The toolbar holds the two things that change what was computed - the
        rectangle and how finely it is sampled - the `mesh` box that draws
        every surface as the wire grid of its samples, the door to the numbers
        behind the picture, and the clear that starts the picture over.
        Everything else about a 3D window is the camera, and the camera is
        the mouse.
        """
        holder = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toolbar())
        layout.addWidget(self.view, 1)
        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet(f"color: {theme.STATUS_TEXT}; background: transparent;")
        line = QtWidgets.QFrame()
        line.setObjectName("statusline")
        line.setStyleSheet(
            f"QFrame#statusline {{ background: {theme.STATUS};"
            f" border-top: 1px solid {theme.STATUS_EDGE}; }}"
        )
        across = QtWidgets.QHBoxLayout(line)
        across.setContentsMargins(10, 3, 10, 3)
        across.addWidget(self.status, 1)
        layout.addWidget(line)
        return holder

    def _toolbar(self) -> QtWidgets.QWidget:
        """The domain, the grid, and the controls the picture itself answers to.

        The fields stand in three groups with a hairline between them, since a
        domain in x, a domain in y and a grid are three answers rather than six
        numbers in a row.
        """
        bar = QtWidgets.QToolBar()
        bar.setMovable(False)
        self.fields: dict[str, QtWidgets.QLineEdit] = {}
        for piece in forms.DOMAIN.pieces:
            if piece.divider:
                bar.addWidget(theme.divider())
            elif piece.entry:
                bar.addWidget(self._field(forms.DOMAIN.field(piece.entry)))
            else:
                bar.addWidget(QtWidgets.QLabel(piece.word))
        self._show_domain()
        handlers = self._handlers()
        self.mesh_action = self._button(bar, "mesh", self._mesh_mode)
        self.mesh_action.setChecked(self.wired)
        self.inspect = self._button(bar, "view.inspect", handlers["view.inspect"])
        self.clear_action = self._button(bar, "clear", handlers["clear"])
        theme.dangerous(bar, self.clear_action)
        return bar

    def _button(self, bar: Any, name: str, handler: Callable[..., None]) -> Any:
        """One control of this window as the button the toolbar draws it with."""
        action = buttoned(self, controls.control(name, controls.SOLID), handler)
        bar.addAction(action)
        return action

    def _field(self, one: forms.Field) -> QtWidgets.QLineEdit:
        """One toolbar number, which the keyboard only reaches when clicked in.

        The keys of this window are the camera's - `1` faces the xy plane - and
        a field that took the focus when the window opened would swallow them.
        So the fields are click-only, and finishing an edit hands the keyboard
        straight back.
        """
        edit = theme.field(one.width)
        edit.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        edit.editingFinished.connect(self._edited)
        self.fields[one.name] = edit
        return edit

    def _furnish(self) -> None:
        """The box, the axis rays, the tick marks and the pool of text items.

        Made once and never added to, because re-anchoring happens while the
        view is live and a view's item list must not be edited from under it. A
        label with nothing to say is set to the empty string instead of being
        taken away.
        """
        self.box = gl.GLBoxItem(
            size=QtGui.QVector3D(WORLD, WORLD, WORLD), color=BOX_COLOR
        )
        self.box.translate(-HALF, -HALF, -HALF)
        self.view.addItem(self.box)
        self.rays = gl.GLAxisItem(size=QtGui.QVector3D(HALF, HALF, HALF))
        self.view.addItem(self.rays)
        self.marks = gl.GLLinePlotItem(
            pos=np.zeros((0, 3), dtype=np.float32), mode="lines", antialias=True
        )
        self.view.addItem(self.marks)
        self.labels = [gl.GLTextItem(text="") for _ in range(3 * MAX_TICKS)]
        self.names = [gl.GLTextItem(text="") for _ in range(3)]
        for item in self.labels + self.names:
            item.setData(font=QtGui.QFont("Helvetica", 10))
            # A number is drawn with a painter and not with the depth buffer, so
            # it is only readable if it is drawn last: the view draws in order
            # of depth value, and a surface added later would otherwise be
            # painted over the numbers naming the edge behind it.
            item.setDepthValue(10)
            self.view.addItem(item)

    def _make_menu(self) -> None:
        """The window's own context menu: what it does, and the keys that do it.

        The GL view has no menu of its own to inherit or to prune, so this one
        is the whole of what a right click on the picture offers. It is the
        camera's list, because a 3D window is a camera: where to look from, and
        the three presets that put an axis edge-on. What is on the toolbar is
        not repeated here - the `mesh` box says its own state where it stands,
        and a menu entry that had to be opened to be read would say it worse.

        The surfaces are listed under `Remove` when the menu opens rather than
        when it is built, since what the window holds changes and the menu does
        not; the submenu of an empty window is offered greyed rather than left
        off, because what it would list is what the window is empty of.
        """
        #: Every key the menu advertises, by the stroke that presses it. The
        #: table `keyPressEvent` dispatches from, beside the one for the keys
        #: no entry names, which is matched on the key alone.
        self._keyed: dict[tuple[int, int], Any] = {}
        self._loose: dict[int, Any] = {}
        #: The action each entry of the menu is drawn by, by the name of the
        #: control it stands for.
        self._offered: dict[str, Any] = {}
        self.menu = QtWidgets.QMenu(self)
        self.menu.aboutToShow.connect(self._read_menu)
        self._make_controls()

    def _make_controls(self) -> None:
        """Walk the control table: an action each, and the menu in its order.

        The words, the keys, the order and where the rules fall are
        `plot.controls`', so a menu regrouped there is regrouped in both
        backends at once. What happens here is only that they become Qt.
        """
        handlers = self._handlers()
        group: int | None = None
        for one in controls.SOLID:
            if not one.menu and not controls.keys(one):
                continue
            if one.kind is controls.Kind.LIST:
                self._remove_menu = QtWidgets.QMenu(controls.plain(one), self)
                action = self._remove_menu.menuAction()
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
        """What each control of this window does, by the control's own name."""
        return {
            "camera.home": self.home,
            "camera.xy": self._facing("camera.xy"),
            "camera.xz": self._facing("camera.xz"),
            "camera.yz": self._facing("camera.yz"),
            "camera.spin": self.spin,
            "view.inspect": self.inspector,
            "image.copy": self.copy_image,
            "image.export": self.export,
            "clear": self.clear,
            "close": self.close,
            "mesh": lambda: self.mesh_action.trigger(),
            "box": self._toggle_box,
            "names": self._toggle_names,
            "legend": self._toggle_legend,
            "surface.wire": self._wire_pointed,
            "plot.remove": self._remove_pointed,
        }

    def _facing(self, name: str) -> Callable[[], None]:
        """Face the plane one of the three presets stands for.

        Which plane that is belongs to the control, beside the words it is
        offered under and the key that presses it: a preset with the plane
        taken out of it would be three entries and three lambdas that had to be
        kept in the same order.
        """
        plane = controls.control(name, controls.SOLID).value
        return lambda: self.face(plane)

    def _read_menu(self) -> None:
        """Read the menu off the window as it opens, which is where it is decided.

        Nothing on it is the menu's own: the rotation is a timer and says for
        itself whether it is running, and a tick remembering what the menu last
        did would be wrong the first time the key was pressed instead. What the
        entries should say is `plot.controls`' answer to a snapshot of this
        window, and the surfaces under `Remove` are part of that answer, since
        what a window holds changes and the menu does not.
        """
        shown = rendered(self._offered, controls.menu(self.snapshot()))
        self._relist(shown["surface.remove"])

    def _relist(self, entry: controls.Entry) -> None:
        """Fill the `Remove` submenu with the surfaces the window now holds.

        An item names its surface by where it stands in the window's own list,
        which is the one thing about a plot that survives being described.
        """
        self._remove_menu.clear()
        for item in entry.items:
            self._remove_menu.addAction(
                item.label,
                lambda _=False, at=int(item.value or 0): self.remove(self.plots[at]),
            )

    def snapshot(self) -> controls.Solid:
        """This window as the description of its controls has to read it."""
        pointed = self._pointed
        return controls.Solid(
            spinning=self._spin.isActive(),
            wired=self.wired,
            boxed=self._boxed,
            named=self._named,
            legend=self._legend,
            surfaces=tuple(surface.named for surface in self.plots),
            pointed=None
            if pointed is None
            else controls.Pointed(
                named=pointed.named, kind=pointed.kind, wire=pointed.wire
            ),
        )

    def _menu_at(self, point: Any) -> None:
        """A right click on the picture: the window's menu, where it was asked for."""
        self.menu.popup(point)

    # -- the plot list -----------------------------------------------------

    def add(self, surface: Surface) -> Drawn:
        """Put a surface in the window, replacing one with the same identity.

        What the session hands over is the surface; what this window keeps is
        its own record of it, and that is what comes back for a caller that
        wants to watch what became of the picture.
        """
        surface = Drawn.of(surface)
        existing = self.find(surface.worksheet, surface.label)
        if existing is not None:
            # A replacement keeps the look of what it replaces - the color and
            # the wire choice - so re-plotting does not reshuffle the picture.
            surface.color, surface.paper = existing.color, existing.paper
            surface.wire = existing.wire
            self.remove(existing)
        else:
            index = self._counter % len(SOLID_PALETTE)
            surface.color, surface.paper = SOLID_PALETTE[index], SOLID_PAPER[index]
            self._counter += 1
            surface.wire = self.wired
        surface.item = gl.GLMeshItem(
            smooth=True, computeNormals=False, shader=None, glOptions="opaque"
        )
        # Hidden until there is a mesh in it: an item with no faces is asked to
        # draw itself on the next frame, and the sampling has not answered yet.
        surface.item.setVisible(False)
        self.view.addItem(surface.item)
        # The wire drawing of the same samples, shown over the solid while the
        # surface's `wire` is on. Opaque rather than blended, so the depth
        # buffer sorts the wire against the solids beside it - its own among
        # them - the same way it sorts two solids. It draws at the curve's
        # weight, best-effort: a core forward-compatible GL profile refuses
        # line widths other than one, and pyqtgraph skips the width call there.
        surface.wires = gl.GLLinePlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            mode="lines",
            width=CURVE_WIDTH,
            antialias=True,
            glOptions="opaque",
        )
        surface.wires.setVisible(False)
        self.view.addItem(surface.wires)
        self._state, self._context = surface.state, surface.context
        self.plots.append(surface)
        self._relabel()
        self.retitle()
        self._name_axes()
        self._start(surface)
        if self.view.broken:
            self._quiet()
        return surface

    def find(self, worksheet: int, label: str) -> Drawn | None:
        """The surface a worksheet and a label name, if this window has it."""
        for surface in self.plots:
            if surface.worksheet == worksheet and surface.label == label:
                return surface
        return None

    def remove(self, surface: Drawn) -> None:
        """Take one surface out of the window, legend row and all."""
        if surface.item is not None:
            self.view.removeItem(surface.item)
            surface.item = None
        if surface.wires is not None:
            self.view.removeItem(surface.wires)
            surface.wires = None
        if surface in self.plots:
            self.plots.remove(surface)
        self._relabel()
        self.retitle()
        self._name_axes()
        self._frame()

    def clear(self) -> None:
        """The toolbar's clear: start this picture over.

        The window it acts on is the window the button is drawn in, so there
        is nothing to infer and nothing to report.
        """
        for surface in list(self.plots):
            self.remove(surface)

    def _relabel(self) -> None:
        """Build the legend again, which is how a removal leaves it right."""
        self.legend.rebuild(
            [
                (
                    surface.named,
                    surface.paper if self._papered else surface.color,
                    surface.hidden,
                )
                for surface in self.plots
            ],
            paper=self._papered,
        )
        # An empty legend is a rectangle over the corner of the picture saying
        # nothing; the toggle is remembered rather than read off the widget, so
        # a legend the user put away stays away when the next surface arrives.
        self.legend.setVisible(self._legend and bool(self.plots))

    def _name_axes(self) -> None:
        """What the three axes are called, taken from the first surface there is.

        The names are the expression's own - the two floor variables in the
        engine's canonical order, and the vertical one where an equation named
        it. A window holding two surfaces of the same two variables is the
        ordinary case, and a window holding two of different ones has to pick:
        the first surface named the axes and the others are drawn over them.
        """
        surface = self.plots[0] if self.plots else None
        if surface is None:
            self.axis_names = ("x", "y", "z")
        else:
            self.axis_names = (*surface.axes, surface.vertical)
        self._anchor()

    # -- evaluation --------------------------------------------------------

    def _start(self, surface: Drawn) -> None:
        """Ask the sampling thread for this surface over the domain and grid.

        The only thing that starts an evaluation. A camera move does not come
        near here, which is the promise the window is built on: the mesh is
        about the domain, and the domain is only ever changed by typing in it.
        """
        surface.generation += 1
        generation = surface.generation
        self.session.sample(
            resample.keyed(self.number, surface),
            resample.grid_job(surface, self.xdomain, self.ydomain, self.grid),
            lambda answer: self._sampled(surface, generation, answer),
        )

    def _sampled(self, surface: Drawn, generation: int, answer: Any) -> None:
        """The grid is in: rebuild the box around it, and say what it holds."""
        if surface.generation != generation or surface not in self.plots:
            return
        if isinstance(answer, Exception):
            surface.trouble = str(answer)
            self.session.trouble(self.number, surface.label, str(answer))
            self.say(f"{surface.label}: {answer}")
            return
        (
            surface.closure,
            surface.xs,
            surface.ys,
            surface.values,
            surface.boundary,
        ) = answer
        # Whatever the window was saying was about the domain this answer has
        # replaced; the two sentences below are what it has to say about the
        # new one, and they are said in the order they were arrived at.
        self._quiet()
        self._frame()
        if evaluate.finite_fraction(surface.values) == 0.0:
            self.say(f"{surface.label}: no real values over this domain")

    def reevaluate(self) -> None:
        """Every surface again, which is what a new domain or grid asks for."""
        for surface in self.plots:
            self._start(surface)

    # -- the box -----------------------------------------------------------

    def _frame(self) -> None:
        """Work out the z the box stands for, and build every mesh to it.

        One box for the window rather than one per surface: two surfaces in the
        same picture are being compared, and comparing them through two
        different vertical scales would be a picture that lies. So a surface
        arriving rebuilds the others, which costs a mesh each and no evaluation
        at all.
        """
        shown = [
            surface.values
            for surface in self.plots
            if surface.visible and surface.values.size
        ]
        found = extent(shown)
        clipped = False
        if self._fixed is not None:
            self.zrange = self._fixed
        elif found is not None:
            self.zrange, clipped = found
        self._reshape()
        for surface in self.plots:
            self._draw(surface)
        self._anchor()
        if clipped:
            self.say(actions.clipped(*self.zrange))

    @property
    def box_now(self) -> Box:
        """The three ranges the picture stands for, as one thing to read them from."""
        return Box(self.xdomain, self.ydomain, self.zrange)

    def _reshape(self) -> None:
        """Stand the axis rays up where the data's origin is in the box.

        The rays start there when the box holds it, which is the same
        convention the 2D window draws its axes by: an axis belongs at zero,
        and a picture framed away from zero shows it at the edge instead of not
        at all. The box itself never changes shape - it is the cube the three
        ranges are stretched onto - so only the rays are placed here.
        """
        box = self.box_now
        origin = (
            float(np.clip(box.across(0.0), -HALF, HALF)),
            float(np.clip(box.along(0.0), -HALF, HALF)),
            float(np.clip(box.up(0.0), -HALF, HALF)),
        )
        self.rays.setSize(*(HALF - value for value in origin))
        self.rays.resetTransform()
        self.rays.translate(*origin)

    def _draw(self, surface: Drawn) -> None:
        """Build one surface's triangles - and its wire, where it wears one.

        A wire surface has its hidden lines removed, the way Derive's plotter
        removed them: what is behind the shape is behind it, and the grid reads
        as a body turned in the light rather than as two grids laid over each
        other. Both items draw for it. The solid keeps the triangles it always
        had - the same ones, stopping at the same rim - painted in the canvas's
        own color, so it is nothing to look at and everything to hide behind;
        the lines draw over it and the depth buffer does the removing, per
        pixel, for the box frame and the other surfaces too.

        The polygon offset is what makes that a wire and not a stitch. The
        lines lie on the very faces they are being tested against, so the
        occluder is pushed a hair away from the camera and the lines win the
        ties.
        """
        if surface.item is None:
            return
        color = surface.paper if self._papered else surface.color
        vertexes, faces, shading = mesh(
            surface.xs, surface.ys, surface.values, self.box_now, surface.boundary
        )
        if surface.wire:
            points, shades = wire(
                surface.xs, surface.ys, surface.values, self.box_now, surface.boundary
            )
            surface.wires.setData(pos=points, color=brightened(shades, color))
            surface.wires.setVisible(surface.visible and bool(points.size))
            surface.item.setGLOptions(WIRE_OCCLUDER)
            # An occluder is a shape and not a picture, so it is given no vertex
            # colors at all: those are what the card reads where a mesh has
            # them, and the one flat color - the canvas behind it, whichever
            # canvas is up - only reaches the fragments in their absence.
            surface.item.setColor(pg.mkColor("w" if self._papered else BACKGROUND))
            if faces.size:
                surface.item.setMeshData(vertexes=vertexes, faces=faces)
            surface.item.setVisible(surface.visible and bool(faces.size))
            return
        surface.wires.setVisible(False)
        surface.item.setGLOptions("opaque")
        if not faces.size:
            surface.item.setVisible(False)
            return
        surface.item.setMeshData(
            vertexes=vertexes, faces=faces, vertexColors=brightened(shading, color)
        )
        surface.item.setVisible(surface.visible)

    def _anchor(self) -> None:
        """Put the tick marks and the numbers on the box edges nearest the camera.

        Three edges carry them: the two bottom edges of the side the camera is
        on, which is where a number sits in front of the picture rather than
        behind it, and for the upright axis the far vertical edge, which is the
        one the surface does not stand in front of. Which edges those are
        changes as the view turns, so this runs on every camera move; it is a
        few dozen positions and no geometry at all.

        An axis pointing at the camera is left unnumbered. Facing the xy plane
        makes the whole z axis one point of the screen, and five numbers stacked
        on that point are five numbers about nothing; the axis keeps its name,
        which is all there is to say about it from there.
        """
        box = self.box_now
        floor = -HALF
        camera = self.view.cameraPosition()
        head_on = self._head_on()
        near_y = -HALF if camera.y() < 0 else HALF
        near_x = -HALF if camera.x() < 0 else HALF
        far_x, far_y = -near_x, -near_y
        out_y, out_x = np.sign(near_y), np.sign(near_x)
        segments: list[tuple[float, float, float]] = []
        numbered: list[tuple[tuple[float, float, float], str]] = []

        for value in self._ticks(self.xdomain) if not head_on[0] else ():
            at = float(box.across(value))
            segments += [
                (at, near_y, floor),
                (at, near_y + out_y * TICK_OUT, floor),
            ]
            numbered.append(((at, near_y + out_y * LABEL_OUT, floor), written(value)))
        for value in self._ticks(self.ydomain) if not head_on[1] else ():
            at = float(box.along(value))
            segments += [
                (near_x, at, floor),
                (near_x + out_x * TICK_OUT, at, floor),
            ]
            numbered.append(((near_x + out_x * LABEL_OUT, at, floor), written(value)))
        upright = np.sign(far_x) * 0.7, np.sign(far_y) * 0.7
        for value in self._ticks(self.zrange) if not head_on[2] else ():
            at = float(box.up(value))
            segments += [
                (far_x, far_y, at),
                (far_x + upright[0] * TICK_OUT, far_y + upright[1] * TICK_OUT, at),
            ]
            numbered.append(
                (
                    (
                        far_x + upright[0] * LABEL_OUT,
                        far_y + upright[1] * LABEL_OUT,
                        at,
                    ),
                    written(value),
                )
            )

        self.marks.setData(
            pos=np.array(segments, dtype=np.float32).reshape(-1, 3),
            color=pg.glColor(PAPER_TICK if self._papered else TICK_COLOR),
            width=1,
        )
        ink = PAPER_TEXT if self._papered else TEXT_COLOR
        for index, item in enumerate(self.labels):
            if index < len(numbered) and self._named:
                where, text = numbered[index]
                item.setData(pos=np.array(where, dtype=np.float64), text=text, color=ink)
            else:
                item.setData(text="")
        places = (
            (0.0, near_y + out_y * NAME_OUT, floor),
            (near_x + out_x * NAME_OUT, 0.0, floor),
            (
                far_x + upright[0] * NAME_OUT,
                far_y + upright[1] * NAME_OUT,
                -floor * 0.9,
            ),
        )
        for item, where, name in zip(self.names, places, self.axis_names):
            item.setData(
                pos=np.array(where, dtype=np.float64),
                text=name if self._named else "",
                color=ink,
            )

    def _head_on(self) -> tuple[bool, bool, bool]:
        """Which axes point so nearly at the camera that they have no length.

        The camera looks along the line from itself to the box's center, so an
        axis is edge-on to the picture exactly when it is parallel to that line.
        The three coordinate presets each put one axis in that position, which
        is why the test is worth making rather than assuming a general view.
        """
        camera = self.view.cameraPosition()
        center = self.view.cameraParams()["center"]
        away = np.array(
            [
                float(camera.x() - center.x()),
                float(camera.y() - center.y()),
                float(camera.z() - center.z()),
            ]
        )
        length = float(np.linalg.norm(away))
        if not length:
            return (False, False, False)
        near = [abs(value) / length > EDGE_ON for value in away]
        return near[0], near[1], near[2]

    def _ticks(self, span: tuple[float, float], count: int = TICKS) -> list[float]:
        """The round numbers of one axis, never more than the label pool holds."""
        return ticks(span[0], span[1], count)[:MAX_TICKS]

    # -- the domain and the grid -------------------------------------------

    def _edited(self) -> None:
        """A toolbar field was left: take what it says, or put back the old text.

        The one place in this window where typing changes what is computed. The
        domain fields read expressions rather than floats - `-π` and `2π` are
        answers - parsed here under the parse state the last surface arrived
        with; what the trees are worth is arithmetic, so they go to the
        sampling thread and the domain moves when the numbers come back. The
        grid is a pair of counts and stays one. A field that does not read is
        reverted rather than argued with, since the value it would take is on
        the screen beside it.
        """
        grid = forms.counts(
            (self.fields["nx"].text(), self.fields["ny"].text())
        )
        if grid is None:
            self._show_domain()
            self.say(forms.GRID_NUMBERS)
            return
        bounds = forms.parsed(
            [self.fields[name].text() for name in ("x0", "x1", "y0", "y1")],
            self._state,
        )
        if bounds is None:
            self._show_domain()
            self.say(forms.DOMAIN_EXPRESSIONS)
            return
        self._edit += 1
        edit = self._edit
        context = self._context
        known = (*self.xdomain, *self.ydomain)

        def work(_report: Callable[..., None]) -> Any:
            return tuple(
                evaluate.number(node, context, default)
                for node, default in zip(bounds, known)
            )

        self.session.sample(
            (self.number, "domain"),
            work,
            lambda answer: self._reframe_typed(edit, grid, answer),
        )

    def _reframe_typed(
        self, edit: int, grid: tuple[int, int], answer: Any
    ) -> None:
        """The typed bounds are worth numbers: take them, or put back the old ones.

        A bound that would not evaluate came back as the value it was replacing,
        so a field of nonsense reverts by arithmetic rather than by a case; an
        inverted domain is the one refusal left to make here.
        """
        if edit != self._edit or isinstance(answer, Exception):
            return
        trouble = forms.domained(answer)
        if trouble:
            self._show_domain()
            self.say(trouble)
            return
        x0, x1, y0, y1 = answer
        if grid != self.grid:
            # A typed grid is sticky: the next surface window opens on it. The
            # sticky value is one count per axis, so a rectangular grid hands
            # on its finer axis. The domain is not sticky - it is a framing,
            # like a 2D view - so only the grid goes back.
            self.session.adjusted(grid=max(grid))
        changed = ((x0, x1), (y0, y1), grid) != (self.xdomain, self.ydomain, self.grid)
        self.xdomain, self.ydomain, self.grid = (x0, x1), (y0, y1), grid
        self._show_domain()
        if changed:
            self.say(forms.evaluating(grid))
            self.reevaluate()

    def _show_domain(self) -> None:
        """Put the window's own numbers back in the fields, however they got there."""
        for name, value in (
            ("x0", self.xdomain[0]),
            ("x1", self.xdomain[1]),
            ("y0", self.ydomain[0]),
            ("y1", self.ydomain[1]),
            ("nx", self.grid[0]),
            ("ny", self.grid[1]),
        ):
            self.fields[name].setText(written(value))

    def reframe(
        self,
        xdomain: tuple[float, float],
        ydomain: tuple[float, float],
        zrange: tuple[float, float],
    ) -> None:
        """The inspector's answer: a box typed rather than arrived at.

        A typed z extent is an opinion and outranks the autoscale from then on -
        somebody who asked for -1 to 1 wants to see the surface cut off at 1,
        and a picture that reframed itself on the next plot would be answering a
        question nobody asked. The extent that was already on the screen is not
        such an opinion, though: applying a change of camera must not quietly
        freeze the z the data chose, so an unedited field changes nothing.
        """
        moved = (xdomain, ydomain) != (self.xdomain, self.ydomain)
        self.xdomain, self.ydomain = xdomain, ydomain
        self._show_domain()
        if zrange[1] > zrange[0] and not forms.alike(zrange, self.zrange):
            self._fixed = zrange
        if moved:
            self.reevaluate()
        self._frame()

    def autoscale_z(self) -> None:
        """Give the z extent back to the data, undoing a typed one."""
        self._fixed = None
        self._frame()

    # -- the camera --------------------------------------------------------

    def home(self) -> None:
        """The default camera: a three-quarter view with the whole cube in it."""
        self.view.setCameraPosition(
            pos=pg.Vector(0.0, 0.0, 0.0),
            distance=actions.CAMERA.distance,
            elevation=actions.CAMERA.elevation,
            azimuth=actions.CAMERA.azimuth,
        )

    def face(self, plane: str) -> None:
        """Look straight at one of the three coordinate planes."""
        camera = actions.facing(plane)
        self.view.setCameraPosition(
            elevation=camera.elevation, azimuth=camera.azimuth
        )
        self.say(actions.FACING.format(plane=plane))

    def spin(self) -> None:
        """Turn the picture slowly, or stop turning it."""
        if self._spin.isActive():
            self._spin.stop()
            self._quiet()
            return
        self._spin.start()
        key = controls.keys(controls.control("camera.spin", controls.SOLID))[0]
        self.say(actions.ROTATING.format(key=key))

    def inspector(self) -> None:
        """The `view...` dialog: the box and the camera as editable numbers."""
        if self._inspector is None:
            self._inspector = Inspector(self)
        self._inspector.refresh()
        self._inspector.show()
        self._inspector.raise_()

    # -- the legend --------------------------------------------------------

    def _legend_picked(self, row: int, button: Any) -> None:
        """A click on a legend row: hide and show, or offer to remove."""
        if not 0 <= row < len(self.plots):
            return
        surface = self.plots[row]
        if button == QtCore.Qt.MouseButton.RightButton:
            self._pointed = surface
            handlers = self._handlers()
            menu = QtWidgets.QMenu(self)
            for entry in controls.card(self.snapshot()):
                menu.addAction(entry.label, handlers[entry.name])
            menu.exec(QtGui.QCursor.pos())
            return
        surface.hidden = not surface.hidden
        self._relabel()
        # The box is recomputed over what is visible now, and redrawing every
        # surface to it is also what puts the items of the surface that was
        # clicked away or back.
        self._frame()

    def _mesh_mode(self, checked: bool) -> None:
        """The toolbar's `mesh` box: every surface as wire, or every one solid.

        The property belongs to the surfaces - the legend's right-click
        carries the per-surface exception - so the box flips them all rather
        than holding a state of its own. The look it leaves is the sticky one:
        the next surface, here or in the next window, arrives drawn this way.
        """
        self.wired = bool(checked)
        for surface in self.plots:
            surface.wire = self.wired
            self._draw(surface)
        self.session.adjusted(wire=self.wired)

    def toggle_wire(self, surface: Drawn) -> None:
        """One surface between wire and solid: the legend's per-surface override.

        An exception rather than a default, so it hands nothing back to the
        sticky preferences and leaves the toolbar box - the every-surface
        control - where it was.
        """
        surface.wire = not surface.wire
        self._draw(surface)

    def _wire_pointed(self) -> None:
        """The wire override, for the surface a legend row was right-clicked on."""
        if self._pointed is not None:
            self.toggle_wire(self._pointed)

    def _remove_pointed(self) -> None:
        """Take out the surface a legend row was right-clicked on."""
        if self._pointed is not None:
            self.remove(self._pointed)
            self._pointed = None

    # -- the three furnishings ----------------------------------------------

    def _toggle_box(self) -> None:
        """The box, its axis rays and its tick marks, or the picture alone."""
        self._boxed = not self._boxed
        for item in (self.box, self.rays, self.marks):
            item.setVisible(self._boxed)

    def _toggle_names(self) -> None:
        """The numbers along the box edges and the names of the axes, or neither."""
        self._named = not self._named
        self._anchor()

    def _toggle_legend(self) -> None:
        """The plot list card, or the bare picture."""
        self._legend = not self._legend
        self.legend.setVisible(self._legend and bool(self.plots))

    # -- export ------------------------------------------------------------

    def copy_image(self) -> None:
        """Ctrl-C: the picture on the clipboard, on paper colors."""
        image = self._photograph()
        if image is None:
            return
        QtWidgets.QApplication.clipboard().setImage(image)
        self.say("Copied the plot to the clipboard")

    def export(self) -> None:
        """Ctrl-S: the picture in a file, on paper colors.

        pyqtgraph's exporters know nothing about a GL view, so this is the
        frame buffer, a file dialog and a PNG - which is every image a 3D plot
        has to give. The colors are the same paper colors the 2D window
        exports on, for the same reason: a dark picture pasted into a document
        is a black rectangle.
        """
        name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save the plot", f"plot{self.number}.png", "PNG image (*.png)"
        )
        if not name:
            return
        image = self._photograph()
        if image is None:
            return
        image.save(name)
        self.say(f"Saved {name}")

    def _photograph(self) -> Any:
        """The view as an image, taken on a white background and put back after.

        `readQImage` is the frame buffer and nothing else: the legend is a Qt
        widget floating over the view, not an item in it, so it would be missing
        from a picture of two surfaces - which is a picture that does not say
        which is which. It is written onto the image afterwards, in the same
        paper colors the surfaces took.
        """
        if self.view.broken:
            self.say(NO_OPENGL.format(reason=self.view.broken))
            return None
        with _on_paper(self):
            image = self.view.readQImage()
            self._name_surfaces(image)
        return image

    def _name_surfaces(self, image: Any) -> None:
        """Write the plot list into the corner of an exported picture."""
        if not self._legend or not self.plots:
            return
        painter = QtGui.QPainter(image)
        # The frame buffer is in device pixels and the widget in logical ones,
        # so a picture on a doubled display is twice the size of the window.
        painter.scale(*([image.height() / max(self.view.height(), 1)] * 2))
        painter.setFont(QtGui.QFont("Helvetica", 9))
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
        for index, surface in enumerate(self.plots):
            painter.setPen(pg.mkColor(surface.paper))
            painter.drawText(16, 20 + index * 14, surface.named)
        painter.end()

    # -- keys --------------------------------------------------------------

    def keyPressEvent(self, ev: Any) -> None:
        """The keys of this window, the menu's own among them.

        Every key a menu entry advertises is dispatched off the menu's own
        table, so the key that works and the key the menu names are one key and
        neither can fire twice; the keys of the controls no entry names - the
        three furnishings and the `mesh` box the toolbar already shows the state
        of - are the second table, matched on the key alone as they always have
        been. What is left written out here are the arrows, which are the
        camera's own gesture rather than a command, and Esc, which closes the
        window - a picture is dismissed the way every other thing that stands
        over a screen is.

        A key with no meaning here at all is the program's when a
        demonstration's step is waiting in this window, since this is where the
        desktop is likely to have put the keyboard: Space and Enter step the
        demonstration, and the arrows go on turning the surface over.
        """
        key = ev.key()
        keys = QtCore.Qt.Key
        command = self._keyed.get(pressed(ev)) or self._loose.get(key)
        if command is not None:
            command.trigger()
        elif key == keys.Key_Left:
            self.view.orbit(actions.ORBIT_DEGREES, 0.0)
        elif key == keys.Key_Right:
            self.view.orbit(-actions.ORBIT_DEGREES, 0.0)
        elif key == keys.Key_Up:
            self.view.orbit(0.0, actions.ORBIT_DEGREES)
        elif key == keys.Key_Down:
            self.view.orbit(0.0, -actions.ORBIT_DEGREES)
        elif key == keys.Key_Escape:
            self.close()
        elif self._demonstrating:
            self._stepping(ev)
        else:
            super().keyPressEvent(ev)
            return
        ev.accept()

    # -- the rest ----------------------------------------------------------

    def say(self, message: str) -> None:
        """Put one line on the status bar, which is the window's whole voice."""
        self._message = message
        self.status.setText(message)

    def _quiet(self) -> None:
        """Nothing to say - unless there is no picture, which always wants saying."""
        self.say(NO_OPENGL.format(reason=self.view.broken) if self.view.broken else "")

    def describe(self) -> protocol.WindowInfo:
        """This window as `Describe` reports it.

        The ranges reported are the domain, since that is what a 3D window is
        framed by.
        """
        return protocol.WindowInfo(
            number=self.number,
            kind=self.kind,
            title=self.windowTitle(),
            current=self.current,
            plots=tuple(
                protocol.PlotInfo(
                    worksheet=surface.worksheet,
                    label=surface.label,
                    text=surface.text,
                    kind=surface.kind,
                    hidden=surface.hidden,
                )
                for surface in self.plots
            ),
            xrange=(float(self.xdomain[0]), float(self.xdomain[1])),
            yrange=(float(self.ydomain[0]), float(self.ydomain[1])),
        )

    def present(self, demonstrating: bool = False) -> None:
        """A plot has landed here: show this window and put it in front.

        A demonstration's step stays in front, until `release` says the
        demonstration is over: the program running it is a terminal that fills
        the screen, and would bury the picture the moment it was clicked on.

        Which of the two windows then has the keyboard is the desktop's to
        decide and not worth arguing with - it hands a window that has just
        appeared the focus whatever the window asks - so nothing here asks.
        What answers the question instead is `keyPressEvent`: a key this window
        has no use for goes back to the program while a step is waiting here.
        """
        self._keep_in_front(demonstrating)
        self.show()
        self.raise_()
        self.activateWindow()

    def release(self) -> None:
        """The demonstration is over: an ordinary window among the others again.

        Shown again because dropping the flag hid it. Its keys are its own from
        here, there being no demonstration left to hand any of them to.
        """
        if not self._demonstrating:
            return
        self._keep_in_front(False)
        self.show()

    def _keep_in_front(self, demonstrating: bool) -> None:
        """Put the window above the others, or let it back among them.

        Changing a window flag hides the window, which is why every caller
        shows it again.
        """
        if demonstrating == self._demonstrating:
            return
        self._demonstrating = demonstrating
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, demonstrating)

    def _stepping(self, ev: Any) -> None:
        """Hand a key this window has no use for back to the program.

        Which is what makes `any key to continue` true of the window the
        picture is in as well as of the one the message is on. Only the keys
        that are nobody's here travel: what the window itself answers to it
        goes on answering to, so a gallery can be traced, turned over and
        zoomed on its way past.
        """
        self.session.stepped()
        ev.accept()

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
                tuple(surface.text or surface.label for surface in self.plots),
                self.current,
            )
        )

    def changeEvent(self, ev: Any) -> None:
        """Activation is the user touching this window, and the session's to know.

        The receiver of the next plot follows the window the user last
        touched, and the activation event is how a click, a raise or an
        alt-tab says so - whatever the platform's focus policy, including one
        that hands focus to whatever the pointer crosses.
        """
        if (
            ev.type() == QtCore.QEvent.Type.ActivationChange
            and self.isActiveWindow()
        ):
            self.session.touched(self.number)
        super().changeEvent(ev)

    def closeEvent(self, ev: Any) -> None:
        """The window manager's business, and the session's to hear about."""
        self._spin.stop()
        if self._inspector is not None:
            self._inspector.close()
        self.session.closed(self.number)
        super().closeEvent(ev)


class Inspector(Sheet):
    """The numbers behind the picture: the box, and where it is looked at from.

    Everything here is reachable with the mouse and the toolbar; what this adds
    is exactness. Somebody who wants the box to run from -1 to 1 in z, or the
    camera at an azimuth of 45 degrees precisely, has nowhere else to say so -
    and a 3D picture is one of the few places where a number typed is worth
    more than a gesture made.

    What is asked for and how it is grouped is `plot.forms`' `VIEW`: the box
    first, as three columns of an axis each, and where it is looked at from
    second. The camera half is a readout as well as a field, following the
    picture while the dialog is up, which is why the heading says so - and that
    is the whole of what this adds to the form.
    """

    #: The nine numbers, in the three groups they are read and written in.
    CENTER = ("cx", "cy", "cz")
    LENGTHS = ("lx", "ly", "lz")
    CAMERA = ("azimuth", "elevation", "distance")

    def __init__(self, window: Window3D) -> None:
        super().__init__(window, forms.VIEW)
        window.view.moved.connect(self._camera_moved)

    def refresh(self) -> None:
        """Fill the fields from the window as it now stands."""
        box = self._window.box_now
        camera = self._window.view.cameraParams()
        values = dict(zip(self.CENTER, box.center))
        values.update(zip(self.LENGTHS, box.lengths))
        for name in self.CAMERA:
            values[name] = float(camera.get(name, 0.0))
        for name, value in values.items():
            self.fields[name].setText(written(value))

    def _camera_moved(self) -> None:
        """Follow the camera while the dialog is up: it is a readout as well."""
        if not self.isVisible():
            return
        camera = self._window.view.cameraParams()
        for name in self.CAMERA:
            self.fields[name].setText(written(float(camera.get(name, 0.0))))

    def answered(self, role: forms.Role) -> None:
        """One of the three answers: give the z back, leave, or take what is typed."""
        if role is forms.Role.CLOSE:
            self.close()
        elif role is forms.Role.RESET:
            self._window.autoscale_z()
            self.refresh()
        else:
            self._take()

    def _take(self) -> None:
        """Apply the typed box and camera, or say that they are not numbers."""
        read = forms.numbers(self.said((*self.CENTER, *self.LENGTHS, *self.CAMERA)))
        if read is None:
            self._window.say(forms.NUMBERS)
            self.refresh()
            return
        spans = [
            forms.spanned(middle, width)
            for middle, width in zip(read[:3], read[3:6])
        ]
        azimuth, elevation, distance = read[6:]
        self._window.reframe(spans[0], spans[1], spans[2])
        self._window.view.setCameraPosition(
            distance=max(distance, 0.1), elevation=elevation, azimuth=azimuth
        )
        self.refresh()


class _on_paper:
    """The window in paper colors for as long as an export takes.

    The same bargain the 2D window makes: a picture pasted into a document is
    read on white, so the background goes white, the surfaces take the darkened
    half of the palette, and the box and its numbers go black. Both swaps
    happen between two paints of the widget, and `grabFramebuffer` renders on
    its own rather than through the screen, so the picture is taken on white
    while the screen keeps the dark one.
    """

    def __init__(self, window: Window3D) -> None:
        self._window = window

    def __enter__(self) -> None:
        window = self._window
        window._papered = True
        window.view.setBackgroundColor("w")
        window.box.setColor(PAPER_BOX)
        for surface in window.plots:
            window._draw(surface)
        window._relabel()
        window._anchor()

    def __exit__(self, *_: Any) -> None:
        window = self._window
        window._papered = False
        window.view.setBackgroundColor(BACKGROUND)
        window.box.setColor(BOX_COLOR)
        for surface in window.plots:
            window._draw(surface)
        window._relabel()
        window._anchor()

