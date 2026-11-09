"""The 3D plot window: several surfaces in one box, with cheap camera movement.

Surfaces are understood by rotating them, so the window makes that free. The mesh is
built once per domain and grid; orbiting, zooming and presets only move the camera, and
nothing the mouse does triggers evaluation. Only the toolbar's domain and grid fields
do, since they change what is computed.

**The geometry comes from `plot.surface`.** The box, triangles, wireframe and shading
are computed there as toolkit-independent arrays; this module feeds them to a
`GLMeshItem` and draws the tick labels. `GLSurfacePlotItem` isn't used because it
generates faces for the full grid and handles NaN vertices unpredictably, leaving
garbage triangles around a dome.

Since there is essentially one box to look at, `Home` is a fixed camera position.

**Wireframes hide what is behind them**, as Derive's hidden-line plots did. A
see-through grid is hard to read: the far side of a dome shows through, and two
surfaces blend together. So the wireframe is drawn over its own solid, painted in the
canvas color and pushed back slightly by the polygon offset, which hides the lines
behind it.

**A window that can't get an OpenGL context says so.** The context is created when the
window is first shown, inside the Qt event loop, where an exception would kill the
process and every 2D window with it. It is caught and reported on the status bar
instead.
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

#: Tick label color, matching the 2D window. The background, box and tick mark colors
#: are in `plot.appearance`, shared with the page.
TEXT_COLOR = "#d0d0d0"

#: Box, tick and label colors for image exports, which use a white background.
PAPER_BOX = (40, 40, 40, 230)
PAPER_TICK = (60, 60, 60, 255)
PAPER_TEXT = "#000000"

#: GL state of the wireframe's occluding solid: the stock opaque state used by all
#: other items, plus the polygon offset.
WIRE_OCCLUDER = {
    **GLOptions["opaque"],
    GL.GL_POLYGON_OFFSET_FILL: True,
    "glPolygonOffset": WIRE_OFFSET,
}

#: Maximum tick marks per axis. The pool of text items is created once and never
#: grown, since items can't be added to the view while it is painting.
MAX_TICKS = 9

#: Message shown when OpenGL is unusable, with the toolkit's error as the reason.
NO_OPENGL = "3D drawing is not available: {reason}"


@dataclass
class Drawn(Surface):
    """A surface as held by this window, with its drawing items and sampled grid.

    The fields inherited from `Surface` come from the session; the ones added here
    belong to evaluation and drawing.

    `values` is kept because the mesh is rebuilt whenever the z range changes, e.g.
    when another surface is added.
    """

    item: Any = None
    #: The wireframe line item, shown over `item` while `wire` is on; `item` remains as
    #: the occluder hiding lines behind it.
    wires: Any = None
    #: The lambdified closure, once the sampling thread has built one.
    closure: Callable[..., np.ndarray] | None = None
    xs: np.ndarray = field(default_factory=lambda: np.empty(0))
    ys: np.ndarray = field(default_factory=lambda: np.empty(0))
    values: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    #: Where the surface stops being real, computed on the sampling thread and cached
    #: with the arrays, so the mesh ends exactly at the boundary.
    boundary: evaluate.Boundary | None = None
    #: The evaluation these values came from; results of outdated jobs are dropped.
    generation: int = 0

    @classmethod
    def of(cls, surface: Surface) -> Drawn:
        """Create this window's record of a surface from the session."""
        return cls(
            **{field.name: getattr(surface, field.name) for field in fields(Surface)}
        )


class View(gl.GLViewWidget):
    """The GL view, extended with a camera signal, a right-click menu and a GL guard.

    `moved` is emitted after every camera move, so tick labels can re-anchor. All stock
    gestures go through `orbit`, `pan`, `setCameraPosition` or the wheel, so overriding
    these four covers every move.

    The stock view doesn't use the right button, so a right click (released within the
    same slop as in the 2D canvas) opens the window's menu.

    A failure in `initializeGL` (no usable OpenGL) would raise inside the Qt event loop
    and kill the process with all plot windows. It is recorded in `broken` instead and
    reported on the status bar.
    """

    moved = QtCore.Signal()

    #: A right click without movement, with its global screen position.
    asked = QtCore.Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.broken = ""
        #: Where the right button was pressed, while it is down.
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
        except Exception as error:  # GL failure mid-frame
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
        """Stock behavior, plus recording a right press that may open the menu."""
        super().mousePressEvent(ev)
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            self._pressed = ev.position()

    def mouseReleaseEvent(self, ev: Any) -> None:
        """Open the menu if the right button is released where it was pressed.

        A right drag does nothing, as in the stock view, which moves the camera with
        the other two buttons.
        """
        super().mouseReleaseEvent(ev)
        if ev.button() != QtCore.Qt.MouseButton.RightButton or self._pressed is None:
            return
        moved = ev.position() - self._pressed
        self._pressed = None
        if float(np.hypot(moved.x(), moved.y())) < CLICK_SLOP_PX:
            self.asked.emit(ev.globalPosition().toPoint())

    def mouseMoveEvent(self, ev: Any) -> None:
        """Stock behavior, plus Shift-drag to pan.

        The stock view pans with Ctrl-drag and the middle button; Shift-drag is added
        for consistency with the 2D window.
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
    """A top-level 3D plot window."""

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
        # Start with the sticky grid preference, clamped like a typed one.
        counted = forms.counts((str(grid), str(grid)))
        self.grid = counted if counted is not None else (DEFAULT_GRID, DEFAULT_GRID)
        #: Whether new surfaces are drawn as wireframes: initially the sticky
        #: preference, then this window's `mesh` box.
        self.wired = bool(wire)
        #: The current z range, and the one typed in the inspector, if any. A typed
        #: range is not autoscaled away when a surface is added.
        self.zrange = DEFAULT_ZRANGE
        #: The axis names, taken from the first surface.
        self.axis_names = ("x", "y", "z")
        self._fixed: tuple[float, float] | None = None
        #: The parse state and context for bounds typed in the domain fields: those
        #: of the last added surface.
        self._state = ParseState()
        self._context = Context()
        #: Counter of domain edits, so results of a superseded edit are discarded.
        self._edit = 0
        self._counter = 0
        self._message = ""
        #: Visibility of the box and ticks, the tick labels and axis names, and the
        #: legend.
        self._boxed = True
        self._named = True
        self._legend = True
        self._papered = False
        self._inspector: Inspector | None = None
        #: The surface whose legend row was right-clicked, for that row's menu.
        self._pointed: Drawn | None = None
        #: Whether a demonstration step is showing in this window. While it is, the
        #: window stays on top and forwards unused keys to the program.
        self._demonstrating = False
        self._build()
        self.retitle()
        self.home()

    # -- construction ------------------------------------------------------

    def _build(self) -> None:
        self.view = View()
        self.view.setBackgroundColor(BACKGROUND)
        # The window handles the keys: the stock view's arrow bindings use a repeat
        # timer, while ours orbit by a fixed step.
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
        """The view between the toolbar and the status line.

        The toolbar has the domain and grid fields, the `mesh` toggle, the inspector
        button and Clear. Everything else is the camera, controlled with the mouse.
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
        """The toolbar: domain and grid fields, followed by the buttons.

        The fields are in three groups separated by dividers: x domain, y domain and
        grid.
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
        """Add a toolbar button for one of this window's controls."""
        action = buttoned(self, controls.control(name, controls.SOLID), handler)
        bar.addAction(action)
        return action

    def _field(self, one: forms.Field) -> QtWidgets.QLineEdit:
        """A toolbar field that only takes focus when clicked.

        The window's keys control the camera (e.g. `1` faces the xy plane), and a
        field focused on open would swallow them. Finishing an edit returns focus to
        the window.
        """
        edit = theme.field(one.width)
        edit.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        edit.editingFinished.connect(self._edited)
        self.fields[one.name] = edit
        return edit

    def _furnish(self) -> None:
        """Create the box, axis rays, tick marks and pool of text items.

        These are created once and never added to, since re-anchoring happens while
        the view is live and its item list must not change. Unused labels are set to
        the empty string rather than removed.
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
            # Text is drawn with a painter, ignoring the depth buffer, so it must be
            # drawn last or later surfaces would paint over it. The view draws items
            # in order of depth value.
            item.setDepthValue(10)
            self.view.addItem(item)

    def _make_menu(self) -> None:
        """Build the window's context menu and its key bindings.

        The GL view has no menu of its own, so this is the whole right-click menu. It is
        mostly about the camera; toolbar controls aren't repeated in it.

        The surfaces are listed under `Remove` when the menu opens, since they change.
        In an empty window the submenu is shown disabled rather than omitted.
        """
        #: Actions for the menu's keys, by key and modifiers, used by `keyPressEvent`,
        #: and actions for keys without menu entries, by key alone.
        self._keyed: dict[tuple[int, int], Any] = {}
        self._loose: dict[int, Any] = {}
        #: The menu's actions, by control name.
        self._offered: dict[str, Any] = {}
        self.menu = QtWidgets.QMenu(self)
        self.menu.aboutToShow.connect(self._read_menu)
        self._make_controls()

    def _make_controls(self) -> None:
        """Create an action for each control, and add menu entries in table order.

        Labels, keys, order and grouping all come from `plot.controls`.
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
        """The handler of each of this window's controls, by control name."""
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
        """A handler facing the plane of the given camera preset.

        The plane is taken from the control's `value`, so it is defined alongside the
        preset's label and key.
        """
        plane = controls.control(name, controls.SOLID).value
        return lambda: self.face(plane)

    def _read_menu(self) -> None:
        """Update the menu from the window's state as it opens.

        The entries come from `plot.controls` applied to a snapshot of the window, so
        ticks reflect the actual state (e.g. whether the rotation timer runs), even if
        it was changed by a key. The `Remove` submenu is refilled too.
        """
        shown = rendered(self._offered, controls.menu(self.snapshot()))
        self._relist(shown["surface.remove"])

    def _relist(self, entry: controls.Entry) -> None:
        """Fill the `Remove` submenu with the window's current surfaces.

        Items refer to surfaces by their index in the window's plot list.
        """
        self._remove_menu.clear()
        for item in entry.items:
            self._remove_menu.addAction(
                item.label,
                lambda _=False, at=int(item.value or 0): self.remove(self.plots[at]),
            )

    def snapshot(self) -> controls.Solid:
        """A snapshot of this window's state for building its menu."""
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
        """Open the window's menu at the right-clicked point."""
        self.menu.popup(point)

    # -- the plot list -----------------------------------------------------

    def add(self, surface: Surface) -> Drawn:
        """Add a surface to the window, replacing one with the same identity.

        Returns the window's own record of the surface.
        """
        surface = Drawn.of(surface)
        existing = self.find(surface.worksheet, surface.label)
        if existing is not None:
            # A replacement keeps the color and wireframe setting of what it
            # replaces, so re-plotting doesn't change the picture's look.
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
        # Hidden until sampling produces a mesh, since an empty item would still be
        # drawn on the next frame.
        surface.item.setVisible(False)
        self.view.addItem(surface.item)
        # The wireframe, shown over the solid while `wire` is on. Opaque rather than
        # blended, so the depth buffer sorts it against all solids, including its
        # own. The curve width is best-effort: a core forward-compatible GL profile
        # only supports width one, and pyqtgraph skips setting it there.
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
        """The surface with this worksheet and label, if the window has it."""
        for surface in self.plots:
            if surface.worksheet == worksheet and surface.label == label:
                return surface
        return None

    def remove(self, surface: Drawn) -> None:
        """Remove a surface from the window, along with its legend row."""
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
        """Remove all surfaces from the window."""
        for surface in list(self.plots):
            self.remove(surface)

    def _relabel(self) -> None:
        """Rebuild the legend from the current surfaces."""
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
        # Hide an empty legend. The toggle state is kept separately from the widget,
        # so a legend the user hid stays hidden when a surface is added.
        self.legend.setVisible(self._legend and bool(self.plots))

    def _name_axes(self) -> None:
        """Set the axis names from the first surface.

        These are the expression's variables in the engine's canonical order, and the
        vertical name if an equation gave one. If surfaces use different variables, the
        first one names the axes.
        """
        surface = self.plots[0] if self.plots else None
        if surface is None:
            self.axis_names = ("x", "y", "z")
        else:
            self.axis_names = (*surface.axes, surface.vertical)
        self._anchor()

    # -- evaluation --------------------------------------------------------

    def _start(self, surface: Drawn) -> None:
        """Request sampling of this surface over the domain and grid.

        This is the only place evaluation starts; camera moves never get here.
        """
        surface.generation += 1
        generation = surface.generation
        self.session.sample(
            resample.keyed(self.number, surface),
            resample.grid_job(surface, self.xdomain, self.ydomain, self.grid),
            lambda answer: self._sampled(surface, generation, answer),
        )

    def _sampled(self, surface: Drawn, generation: int, answer: Any) -> None:
        """Handle a sampled grid: reframe the box, and report problems."""
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
        # Clear messages about the previous domain before reporting on the new one.
        self._quiet()
        self._frame()
        if evaluate.finite_fraction(surface.values) == 0.0:
            self.say(f"{surface.label}: no real values over this domain")

    def reevaluate(self) -> None:
        """Resample every surface, after a domain or grid change."""
        for surface in self.plots:
            self._start(surface)

    # -- the box -----------------------------------------------------------

    def _frame(self) -> None:
        """Compute the box's z range, and rebuild every mesh for it.

        All surfaces share one box so they can be compared on the same vertical scale.
        So adding a surface rebuilds the other meshes, without re-evaluation.
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
        """The current x, y and z ranges as a `Box`."""
        return Box(self.xdomain, self.ydomain, self.zrange)

    def _reshape(self) -> None:
        """Place the axis rays at the data's origin, clamped to the box.

        As in the 2D window, axes are drawn at zero, or at the box's edge when zero is
        outside it. The box itself is a fixed cube, so only the rays move.
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
        """Build a surface's mesh, and its wireframe if it has one.

        Wireframes have hidden lines removed, as in Derive's plotter. The solid is still
        drawn with the same triangles, but in the canvas color, so it is invisible but
        hides the lines behind it, of this and other surfaces and the box, via the
        depth buffer.

        The lines lie exactly on the faces they are depth-tested against, so the
        occluder is pushed slightly back by the polygon offset to let the lines win.
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
            # The occluder gets no vertex colors, since those would override the flat
            # canvas color it should be drawn in.
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
        """Place the tick marks and labels on the box edges facing the camera.

        The x and y ticks go on the two bottom edges on the camera's side, in front of
        the picture, and the z ticks on the far vertical edge, which the surface
        doesn't cover. These change as the view rotates, so this runs on every camera
        move; it only updates a few dozen positions.

        An axis pointing at the camera gets no tick labels, since they would all pile
        up on one point, but keeps its name.
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
        """Which axes point (nearly) straight at the camera.

        An axis is head-on when it is parallel to the line from the camera to the box's
        center, as happens with each of the plane presets.
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
        """Tick values of an axis, capped at the size of the label pool."""
        return ticks(span[0], span[1], count)[:MAX_TICKS]

    # -- the domain and the grid -------------------------------------------

    def _edited(self) -> None:
        """Apply an edited toolbar field, or restore the old text if invalid.

        The domain fields accept expressions such as `-π` and `2π`, parsed with the last
        surface's parse state and evaluated on the sampling thread; the domain changes
        when the numbers come back. The grid is a pair of integers. Invalid input is
        reverted.
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
        """Apply the evaluated domain bounds, or restore the old ones if invalid.

        A bound that failed to evaluate comes back as its previous value, so only an
        inverted domain needs rejecting here.
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
            # A typed grid is sticky, for the next 3D window. The preference is one
            # count for both axes, so a rectangular grid passes on its finer one. The
            # domain is a framing, like a 2D view, and isn't sticky.
            self.session.adjusted(grid=max(grid))
        changed = ((x0, x1), (y0, y1), grid) != (self.xdomain, self.ydomain, self.grid)
        self.xdomain, self.ydomain, self.grid = (x0, x1), (y0, y1), grid
        self._show_domain()
        if changed:
            self.say(forms.evaluating(grid))
            self.reevaluate()

    def _show_domain(self) -> None:
        """Show the window's current domain and grid in the fields."""
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
        """Apply a box typed in the inspector.

        A typed z range overrides autoscaling from then on: a user who asked for -1 to
        1 wants the surface cut off at 1, even after the next plot. An unedited z field
        changes nothing, so applying a camera change doesn't freeze the autoscaled
        range.
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
        """Return to an autoscaled z range, undoing a typed one."""
        self._fixed = None
        self._frame()

    # -- the camera --------------------------------------------------------

    def home(self) -> None:
        """Reset to the default camera: a three-quarter view showing the whole cube."""
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
        """Toggle slow automatic rotation."""
        if self._spin.isActive():
            self._spin.stop()
            self._quiet()
            return
        self._spin.start()
        key = controls.keys(controls.control("camera.spin", controls.SOLID))[0]
        self.say(actions.ROTATING.format(key=key))

    def inspector(self) -> None:
        """Open the `view...` dialog, which edits the box and camera numerically."""
        if self._inspector is None:
            self._inspector = Inspector(self)
        self._inspector.refresh()
        self._inspector.show()
        self._inspector.raise_()

    # -- the legend --------------------------------------------------------

    def _legend_picked(self, row: int, button: Any) -> None:
        """Handle a legend row click: toggle visibility, or show the row's menu."""
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
        # Recompute the box over the visible surfaces; redrawing also hides or shows
        # the clicked surface's items.
        self._frame()

    def _mesh_mode(self, checked: bool) -> None:
        """The toolbar's `mesh` toggle: draw every surface as wireframe or solid.

        The setting belongs to each surface (the legend's menu can override it for one),
        so this sets it on all of them. It is also stored as the sticky preference for
        new surfaces, here or in new windows.
        """
        self.wired = bool(checked)
        for surface in self.plots:
            surface.wire = self.wired
            self._draw(surface)
        self.session.adjusted(wire=self.wired)

    def toggle_wire(self, surface: Drawn) -> None:
        """Toggle one surface between wireframe and solid, from the legend's menu.

        This is a per-surface exception, so it changes neither the sticky preference
        nor the toolbar's `mesh` toggle.
        """
        surface.wire = not surface.wire
        self._draw(surface)

    def _wire_pointed(self) -> None:
        """Toggle the wireframe of the surface whose legend row was right-clicked."""
        if self._pointed is not None:
            self.toggle_wire(self._pointed)

    def _remove_pointed(self) -> None:
        """Remove the surface whose legend row was right-clicked."""
        if self._pointed is not None:
            self.remove(self._pointed)
            self._pointed = None

    # -- box, labels and legend toggles --------------------------------------

    def _toggle_box(self) -> None:
        """Toggle the box, its axis rays and its tick marks."""
        self._boxed = not self._boxed
        for item in (self.box, self.rays, self.marks):
            item.setVisible(self._boxed)

    def _toggle_names(self) -> None:
        """Toggle the tick labels and axis names."""
        self._named = not self._named
        self._anchor()

    def _toggle_legend(self) -> None:
        """Toggle the legend."""
        self._legend = not self._legend
        self.legend.setVisible(self._legend and bool(self.plots))

    # -- export ------------------------------------------------------------

    def copy_image(self) -> None:
        """Ctrl-C: copy the picture to the clipboard, in paper colors."""
        image = self._photograph()
        if image is None:
            return
        QtWidgets.QApplication.clipboard().setImage(image)
        self.say("Copied the plot to the clipboard")

    def export(self) -> None:
        """Ctrl-S: save the picture to a PNG file, in paper colors.

        pyqtgraph's exporters don't support GL views, so this saves the frame buffer.
        Like the 2D window, it uses paper colors, so the image works in documents.
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
        """The view as an image, rendered in paper colors on a white background.

        `readQImage` captures only the frame buffer, which lacks the legend (a Qt
        widget over the view), so the plot list is painted onto the image afterwards.
        """
        if self.view.broken:
            self.say(NO_OPENGL.format(reason=self.view.broken))
            return None
        with _on_paper(self):
            image = self.view.readQImage()
            self._name_surfaces(image)
        return image

    def _name_surfaces(self, image: Any) -> None:
        """Paint the plot list into the corner of an exported image."""
        if not self._legend or not self.plots:
            return
        painter = QtGui.QPainter(image)
        # The frame buffer is in device pixels and the widget in logical ones, so
        # scale for high-DPI displays.
        painter.scale(*([image.height() / max(self.view.height(), 1)] * 2))
        painter.setFont(QtGui.QFont("Helvetica", 9))
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
        for index, surface in enumerate(self.plots):
            painter.setPen(pg.mkColor(surface.paper))
            painter.drawText(16, 20 + index * 14, surface.named)
        painter.end()

    # -- keys --------------------------------------------------------------

    def keyPressEvent(self, ev: Any) -> None:
        """Handle the window's keys.

        Keys shown in the menu are dispatched through the menu's own table, so they
        can't fire twice. Keys of controls without menu entries (the box, label and
        legend toggles, and `mesh`) use a second table, matched on the key alone. The
        arrows, which orbit the camera, and Esc, which closes the window, are handled
        directly.

        During a demonstration step, other keys are forwarded to the program, since
        the desktop may have given this window focus: Space and Enter advance the
        demonstration, while the arrows still rotate the surface.
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
        """Show a message on the status bar."""
        self._message = message
        self.status.setText(message)

    def _quiet(self) -> None:
        """Clear the status bar, unless OpenGL is broken, which is always reported."""
        self.say(NO_OPENGL.format(reason=self.view.broken) if self.view.broken else "")

    def describe(self) -> protocol.WindowInfo:
        """This window as `Describe` reports it, with the domain as the ranges."""
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
        uses keep working, so a gallery can still be rotated and zoomed.
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
                tuple(surface.text or surface.label for surface in self.plots),
                self.current,
            )
        )

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
        """Stop rotation, close the inspector, and report the close to the session."""
        self._spin.stop()
        if self._inspector is not None:
            self._inspector.close()
        self.session.closed(self.number)
        super().closeEvent(ev)


class Inspector(Sheet):
    """A dialog for editing the box and camera numerically.

    Everything here is also reachable with the mouse and toolbar; the dialog adds
    precision, e.g. a z range of exactly -1 to 1, or an azimuth of exactly 45 degrees.

    The fields and layout come from `plot.forms`' `VIEW`. The camera fields follow the
    camera while the dialog is open, as its subtitle says.
    """

    #: The field names, in three groups.
    CENTER = ("cx", "cy", "cz")
    LENGTHS = ("lx", "ly", "lz")
    CAMERA = ("azimuth", "elevation", "distance")

    def __init__(self, window: Window3D) -> None:
        super().__init__(window, forms.VIEW)
        window.view.moved.connect(self._camera_moved)

    def refresh(self) -> None:
        """Fill the fields from the window's current state."""
        box = self._window.box_now
        camera = self._window.view.cameraParams()
        values = dict(zip(self.CENTER, box.center))
        values.update(zip(self.LENGTHS, box.lengths))
        for name in self.CAMERA:
            values[name] = float(camera.get(name, 0.0))
        for name, value in values.items():
            self.fields[name].setText(written(value))

    def _camera_moved(self) -> None:
        """Update the camera fields when the camera moves, while the dialog is shown."""
        if not self.isVisible():
            return
        camera = self._window.view.cameraParams()
        for name in self.CAMERA:
            self.fields[name].setText(written(float(camera.get(name, 0.0))))

    def answered(self, role: forms.Role) -> None:
        """Handle a button: autoscale z, close, or apply the typed values."""
        if role is forms.Role.CLOSE:
            self.close()
        elif role is forms.Role.RESET:
            self._window.autoscale_z()
            self.refresh()
        else:
            self._take()

    def _take(self) -> None:
        """Apply the typed box and camera, or report that they aren't numbers."""
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
    """Context manager switching the window to paper colors during an export.

    As in the 2D window, the background becomes white, surfaces use the paper palette,
    and the box and labels turn dark. Both switches happen between two paints, and
    `grabFramebuffer` renders offscreen, so the screen keeps showing the dark picture.
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

