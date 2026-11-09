"""The commands of plot windows, defined once for all backends.

Each control's name, menu label, desktop and web keys, kind (action, toggle, choice)
and menu group are defined here, and every backend renders them, so windows can't
disagree on labels, keys or order. This extends the "one spelling of an expression"
rule to controls: if a label, key or menu order appears in two files, something is
wrong.

What a menu should show *now* (whether `Grid` is ticked, which curve `Remove` names, or
whether it is shown at all) depends on the window. The backend passes a small snapshot,
`Flat` for 2D or `Solid` for 3D, and `menu` returns the entries for that state. The
backend keeps the view state itself, so a drag never has to consult Python.

Keys are listed twice because the alphabets differ: `keys` uses the desktop toolkit's
names (also shown in menus), `web` uses browser key event names, where single letters
match either case. Keys the browser reserves, such as Ctrl+W, are omitted from `web`.

Likewise `paged` is an alternative label for the page, used only where the page
behaves differently: Export has no dialog in the page, so its label drops the `...`.

Gestures aren't controls. The arrow keys, Escape, Tab and Enter mean different things
depending on whether a trace marker is shown, have no menu entries, and are handled
by the key handler.
"""

from __future__ import annotations

from collections.abc import Container
from dataclasses import dataclass, replace
from enum import StrEnum

from rederive.plot.protocol import PlotKind

__all__ = [
    "FLAT",
    "POINT_SIZES",
    "SOLID",
    "Control",
    "Entry",
    "Flat",
    "Kind",
    "Pointed",
    "Solid",
    "card",
    "choices",
    "control",
    "keys",
    "listed",
    "menu",
    "plain",
]


class Kind(StrEnum):
    """The kind of control, which determines how it is rendered.

    A toggle shows a tick, or states its state in words using `flipped`. A choice is a
    submenu of fixed alternatives; a list is a submenu of the window's current items,
    taken from the snapshot.
    """

    ACTION = "action"
    TOGGLE = "toggle"
    CHOICE = "choice"
    LIST = "list"


@dataclass(frozen=True)
class Control:
    """A plot window command, described without its implementation.

    `name` is what backends dispatch on. `label` is the menu text, `bar` the toolbar
    button text if there is one, and `hint` the button's tooltip. `{plot}` in a label
    is replaced with the name of the plot that was clicked.

    `group` is the menu group; a separator is drawn where it changes. `menu` says
    whether the control is on the context menu; toolbar toggles and key-only commands
    are defined here too, so their keys are spelled in one place.
    """

    name: str
    label: str = ""
    kind: Kind = Kind.ACTION
    #: The control's keys, as named by the desktop toolkit and by browser key events.
    #: The web keys write letters in lower case and match either case.
    keys: tuple[str, ...] = ()
    web: tuple[str, ...] = ()
    group: int = 0
    menu: bool = True
    bar: str = ""
    hint: str = ""
    #: The label of a toggle while it is on, for toggles that show their state in words
    #: rather than with a tick.
    flipped: str = ""
    #: The label in the page, where the page behaves differently. Only Export uses it:
    #: a trailing `...` promises a dialog, and the page's download has none.
    paged: str = ""
    #: A choice's alternatives, and the format of an item's label.
    choices: tuple[float, ...] = ()
    item: str = ""
    #: The value of a control in a set, such as the plane a camera preset faces.
    value: str = ""
    #: The browser event that triggers this control instead of a key press. Only Copy
    #: image uses it: canceling the Ctrl+C keydown would also cancel the copy, so the
    #: page handles the `copy` event and leaves the keystroke alone. The key is still
    #: listed for display.
    event: str = ""


@dataclass(frozen=True)
class Entry:
    """A fully resolved menu line, ready to render.

    The label is final, the keys are the platform's, and a toggle's tick is decided.
    The renderer needs nothing else.
    """

    name: str
    label: str
    kind: Kind = Kind.ACTION
    keys: tuple[str, ...] = ()
    checked: bool = False
    enabled: bool = True
    #: Whether a separator precedes this entry, starting a new menu group.
    separated: bool = False
    #: The lines of a submenu, for a choice or a list.
    items: tuple[Entry, ...] = ()
    #: The value of a submenu item: a point size, or the index of a plot.
    value: float | int | str | None = None


@dataclass(frozen=True)
class Pointed:
    """The plot that was right-clicked, for the plot-specific menu entries.

    A right click on a curve, legend row or surface targets one plot, and those
    entries are named after it and reflect its style. Only what the menu needs is
    included, not the plot itself.
    """

    named: str
    kind: PlotKind
    connected: bool = False
    wire: bool = False


@dataclass(frozen=True)
class Flat:
    """A snapshot of a 2D window's state, for building its menu.

    Deliberately minimal: only what the controls depend on, so it is cheap to build on
    every right click.
    """

    tracing: bool = False
    grid: bool = True
    legend: bool = True
    polar: bool = False
    equal: bool = True
    pointed: Pointed | None = None


@dataclass(frozen=True)
class Solid:
    """A snapshot of a 3D window's state, for building its menu.

    `surfaces` are the names of the window's plots in order, listed by the `Remove`
    submenu, whose items index into it.
    """

    spinning: bool = False
    wired: bool = True
    boxed: bool = True
    named: bool = True
    legend: bool = True
    surfaces: tuple[str, ...] = ()
    pointed: Pointed | None = None


State = Flat | Solid

#: The available point sizes of a data plot.
POINT_SIZES = (3.0, 5.0, 8.0, 12.0)

#: Placeholder for the clicked plot's name in labels.
PLOT = "{plot}"


#: The 2D window's controls, in menu order. Entries with `menu=False` are toolbar
#: toggles and key-only commands.
FLAT: tuple[Control, ...] = (
    Control(
        name="range.set",
        label="Set range...",
        group=0,
    ),
    Control(
        name="view.all",
        label="View all",
        keys=("A",),
        web=("a",),
        group=0,
    ),
    Control(
        name="view.home",
        label="Home framing",
        keys=("Home", "0"),
        web=("Home", "0", "h"),
        group=0,
    ),
    Control(
        name="view.back",
        label="Back",
        keys=("Backspace",),
        web=("Backspace",),
        group=0,
    ),
    Control(
        name="trace",
        label="Trace",
        kind=Kind.TOGGLE,
        keys=("T", "F3"),
        web=("t", "F3"),
        group=1,
    ),
    Control(
        name="grid",
        label="Grid",
        kind=Kind.TOGGLE,
        keys=("G",),
        web=("g",),
        group=1,
    ),
    Control(
        name="legend",
        label="Legend",
        kind=Kind.TOGGLE,
        keys=("L",),
        web=("l",),
        group=1,
    ),
    Control(
        name="image.copy",
        label="Copy image",
        keys=("Ctrl+C",),
        web=("Ctrl+C",),
        event="copy",
        group=2,
    ),
    Control(
        name="image.export",
        label="Export...",
        paged="Export",
        keys=("Ctrl+S",),
        web=("Ctrl+S",),
        group=2,
    ),
    Control(
        name="clear",
        label="Clear",
        keys=("Del",),
        web=("Delete",),
        group=2,
        bar="clear",
        hint="Remove every plot from this window (Del)",
    ),
    # Ctrl+W belongs to the browser, so the page uses only Q.
    Control(
        name="close",
        label="Close",
        keys=("Q", "Ctrl+W"),
        web=("q",),
        group=2,
    ),
    Control(
        name="plot.remove",
        label=f"Remove {PLOT}",
        group=3,
    ),
    Control(
        name="points.connect",
        label="Connect points",
        flipped="Disconnect points",
        group=3,
    ),
    Control(
        name="points.size",
        label="Point size",
        kind=Kind.CHOICE,
        choices=POINT_SIZES,
        item="{size:g} px",
        group=3,
    ),
    # Toolbar toggles, kept off the menu since the toolbar already shows their state.
    Control(
        name="scales.equal",
        kind=Kind.TOGGLE,
        label="Equal scales",
        menu=False,
        bar="1:1",
        hint="Equal scales on both axes",
    ),
    Control(
        name="view.polar",
        kind=Kind.TOGGLE,
        label="Polar",
        web=("p",),
        menu=False,
        bar="polar",
        hint="Show every curve as r = f(θ)",
    ),
    # Key-only commands: forward through the view history, and zoom.
    Control(
        name="view.forward",
        label="Forward",
        keys=("Shift+Backspace",),
        web=("Shift+Backspace",),
        menu=False,
    ),
    Control(
        name="view.zoom.in",
        label="Zoom in",
        keys=("+", "="),
        web=("+", "="),
        menu=False,
    ),
    Control(
        name="view.zoom.out",
        label="Zoom out",
        keys=("-",),
        web=("-",),
        menu=False,
    ),
)


#: The 3D window's controls. The menu is mostly about the camera; toggles shown on the
#: toolbar aren't repeated in it.
SOLID: tuple[Control, ...] = (
    Control(
        name="camera.home",
        label="Home view",
        keys=("Home", "0"),
        web=("Home", "0", "h"),
        group=0,
    ),
    Control(
        name="camera.xy",
        label="Face the xy plane",
        keys=("1",),
        web=("1",),
        value="xy",
        group=0,
    ),
    Control(
        name="camera.xz",
        label="Face the xz plane",
        keys=("2",),
        web=("2",),
        value="xz",
        group=0,
    ),
    Control(
        name="camera.yz",
        label="Face the yz plane",
        keys=("3",),
        web=("3",),
        value="yz",
        group=0,
    ),
    Control(
        name="camera.spin",
        label="Rotate",
        kind=Kind.TOGGLE,
        keys=("R",),
        web=("r",),
        group=0,
    ),
    Control(
        name="view.inspect",
        label="View...",
        web=("v",),
        group=1,
        bar="view...",
        hint="The box and the camera as numbers",
    ),
    Control(
        name="image.copy",
        label="Copy image",
        keys=("Ctrl+C",),
        web=("Ctrl+C",),
        event="copy",
        group=1,
    ),
    Control(
        name="image.export",
        label="Export...",
        paged="Export",
        keys=("Ctrl+S",),
        web=("Ctrl+S",),
        group=1,
    ),
    Control(
        name="surface.remove",
        label="Remove",
        kind=Kind.LIST,
        group=2,
    ),
    Control(
        name="clear",
        label="Clear",
        keys=("Del",),
        web=("Delete",),
        group=2,
        bar="clear",
        hint="Remove every surface from this window (Del)",
    ),
    Control(
        name="close",
        label="Close",
        keys=("Q", "Ctrl+W"),
        web=("q",),
        group=2,
    ),
    # M rather than W, since Ctrl-W closes the window and a frequent toggle shouldn't be
    # one slipped modifier away from that.
    Control(
        name="mesh",
        kind=Kind.TOGGLE,
        label="Mesh",
        keys=("M",),
        web=("m",),
        menu=False,
        bar="mesh",
        hint="Draw every surface as the wire grid of its samples (M)",
    ),
    Control(
        name="box",
        kind=Kind.TOGGLE,
        label="Box",
        keys=("B",),
        web=("b",),
        menu=False,
    ),
    Control(
        name="names",
        kind=Kind.TOGGLE,
        label="Axis names",
        keys=("X",),
        web=("x",),
        menu=False,
    ),
    Control(
        name="legend",
        kind=Kind.TOGGLE,
        label="Legend",
        keys=("L",),
        web=("l",),
        menu=False,
    ),
    # Per-surface entries of a legend row's menu: wireframe toggle (overriding the
    # `mesh` box) and removal.
    Control(
        name="surface.wire",
        label="Draw as wire mesh",
        flipped="Draw solid",
        menu=False,
    ),
    Control(
        name="plot.remove",
        label=f"Remove {PLOT}",
        menu=False,
    ),
)


def control(name: str, controls: tuple[Control, ...]) -> Control:
    """The control with this name in the given window's table.

    The table must be given because both windows have controls with the same name
    but different labels, e.g. `Clear` removes plots in one and surfaces in the other.
    """
    for one in controls:
        if one.name == name:
            return one
    raise KeyError(name)


def plain(one: Control) -> str:
    """A control's label with the `{plot}` placeholder removed.

    Used for menus built once and reused, which can't name a particular plot.
    """
    return one.label.replace(PLOT, "").strip()


def keys(one: Control, page: bool = False) -> tuple[str, ...]:
    """A control's keys for the desktop or for the page."""
    return one.web if page else one.keys


def listed(controls: tuple[Control, ...]) -> tuple[Control, ...]:
    """The controls on the context menu, in menu order."""
    return tuple(one for one in controls if one.menu)


def choices(one: Control, page: bool = False) -> tuple[Entry, ...]:
    """A choice's alternatives as submenu entries.

    They are fixed rather than taken from the snapshot, since they don't depend on
    the window state.
    """
    return tuple(
        Entry(
            name=one.name,
            label=one.item.format(size=size),
            keys=keys(one, page),
            value=size,
        )
        for size in one.choices
    )


def menu(
    state: State, page: bool = False, offers: Container[str] | None = None
) -> tuple[Entry, ...]:
    """The context menu for a window in this state.

    Labels are resolved, toggles ticked, and inapplicable entries (such as plot entries
    when no plot was clicked) left out.

    `offers`, if given, is the set of controls the backend implements; others are left
    out. This is done here rather than in the renderer so that separators are placed
    correctly, and a group whose entries are all gone loses its separator.
    """
    resolve = _flat if isinstance(state, Flat) else _solid
    controls = FLAT if isinstance(state, Flat) else SOLID
    entries: list[Entry] = []
    group: int | None = None
    for one in listed(controls):
        if offers is not None and one.name not in offers:
            continue
        entry = resolve(one, state, page)
        if entry is None:
            continue
        entries.append(replace(entry, separated=group is not None and one.group != group))
        group = one.group
    return tuple(entries)


def card(
    state: State, page: bool = False, offers: Container[str] | None = None
) -> tuple[Entry, ...]:
    """The menu of a legend row, with only the entries about its plot.

    The canvas menu is largely about the view, so a legend row gets its own menu, but
    with the same entries the canvas offers for a clicked plot.
    """
    if state.pointed is None:
        return ()
    flat = isinstance(state, Flat)
    names = ("plot.remove", "points.connect") if flat else ("surface.wire", "plot.remove")
    controls = FLAT if flat else SOLID
    resolve = _flat if flat else _solid
    found = (
        resolve(control(name, controls), state, page)
        for name in names
        if offers is None or name in offers
    )
    return tuple(entry for entry in found if entry is not None)


def _flat(one: Control, state: Flat, page: bool) -> Entry | None:
    """An entry of a 2D window's menu, or None if it doesn't apply."""
    pointed = state.pointed
    if one.name in ("plot.remove", "points.connect", "points.size"):
        if pointed is None:
            return None
        if one.name != "plot.remove" and pointed.kind is not PlotKind.DATA:
            return None
    checked = {
        "trace": state.tracing,
        "grid": state.grid,
        "legend": state.legend,
        "view.polar": state.polar,
        "scales.equal": state.equal,
    }.get(one.name, False)
    flipped = one.name == "points.connect" and pointed is not None and pointed.connected
    return _made(one, state, page, checked=checked, flipped=flipped)


def _solid(one: Control, state: Solid, page: bool) -> Entry | None:
    """An entry of a 3D window's menu, or None if it doesn't apply."""
    pointed = state.pointed
    if one.name in ("plot.remove", "surface.wire") and pointed is None:
        return None
    checked = {
        "camera.spin": state.spinning,
        "mesh": state.wired,
        "box": state.boxed,
        "names": state.named,
        "legend": state.legend,
    }.get(one.name, False)
    flipped = one.name == "surface.wire" and pointed is not None and pointed.wire
    entry = _made(one, state, page, checked=checked, flipped=flipped)
    if one.name != "surface.remove":
        return entry
    # In an empty window the submenu is shown disabled rather than omitted.
    return replace(
        entry,
        enabled=bool(state.surfaces),
        items=tuple(
            Entry(name=one.name, label=named, value=index)
            for index, named in enumerate(state.surfaces)
        ),
    )


def _made(
    one: Control, state: State, page: bool, checked: bool, flipped: bool
) -> Entry:
    """The menu entry for a control, with its label resolved."""
    if flipped and one.flipped:
        label = one.flipped
    elif page and one.paged:
        label = one.paged
    else:
        label = one.label
    if state.pointed is not None:
        label = label.replace(PLOT, state.pointed.named)
    return Entry(
        name=one.name,
        label=label,
        kind=one.kind,
        keys=keys(one, page),
        checked=checked,
        items=choices(one, page) if one.kind is Kind.CHOICE else (),
    )
