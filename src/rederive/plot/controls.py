"""One spelling of a control: what a plot window offers, written down once.

A window's commands are a vocabulary rather than code. The name of each, the
words a menu writes it with, the keys that press it on a desktop and in a page,
whether it is an action, a toggle or a choice, and which group of the menu it
stands in: all of that is here, once, and every backend renders it. Two windows
that read their menus from this table cannot disagree about a word, a key or an
order, because there is nowhere else either of them could have read one from.

That is the existing "one spelling of an expression" rule applied a level up:
**one spelling of a control**. If a label, a key or a menu order appears in two
files, the seam has leaked.

The other half of a menu is what it should say *now* - `Grid` ticked or not,
`Remove` naming the curve the click was over or absent because it was over
nothing. That cannot be a constant, so it is a function of a snapshot the
backend hands in: `Flat` for a 2D window, `Solid` for a 3D one, and `menu` is
the function from either to the entries as they should currently read. The
backend keeps the view state and hands over a copy of the little of it a menu
depends on, which is what keeps a drag from having to ask Python what it came
to.

Keys are written twice because they are two different alphabets, not two
opinions: `keys` is what a desktop toolkit and a menu's shortcut column spell,
`web` is what a page's key events report, and a single letter there is matched
whichever case it arrives in. A key a page may not have - Ctrl+W and its
neighbours belong to the browser - is simply not in the second column, and the
control says so by being short of one rather than by pretending.

What is not here is the gestures. The arrow keys, Escape, Tab and Enter mean
one thing while a trace marker is up and another while it is not, no menu entry
names them, and they belong to whatever answers a key press. A control is
something that has a name and can be offered.
"""

from __future__ import annotations

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
    """What sort of control an entry is, which is what a renderer draws it as.

    A toggle is drawn with a tick where the toolkit has one and says its state
    in words where it has not, which is what `flipped` is for. A choice is a
    submenu of alternatives written down here; a list is a submenu of whatever
    the window happens to hold, and its items come from the snapshot.
    """

    ACTION = "action"
    TOGGLE = "toggle"
    CHOICE = "choice"
    LIST = "list"


@dataclass(frozen=True)
class Control:
    """One thing a plot window does, as everything about it but the doing.

    `name` is what the backends dispatch on and is the only part of a control
    that is not for reading. `label` is how a menu writes it, `bar` how a
    toolbar button does where it has one, and `hint` what such a button says
    when the pointer rests on it. A label holding `{plot}` is about whatever
    the click was over and is filled in when the menu is read.

    `group` is which run of the menu the entry belongs to: a rule is drawn
    wherever the number changes, so the grouping is a property of the table
    rather than of the code that walks it. `menu` says whether the control is
    on the context menu at all - a toolbar toggle and a key with no entry are
    controls like any other, and a window that could not name them would be
    back to spelling their keys itself.
    """

    name: str
    label: str = ""
    kind: Kind = Kind.ACTION
    #: The keys that press it, as a desktop toolkit spells them and as a page
    #: reports them. A page spells a letter in lower case and matches either.
    keys: tuple[str, ...] = ()
    web: tuple[str, ...] = ()
    group: int = 0
    menu: bool = True
    bar: str = ""
    hint: str = ""
    #: The label a toggle takes while it is on, where it says its state in
    #: words rather than with a tick.
    flipped: str = ""
    #: What a choice offers, and the words one of its items is written with.
    choices: tuple[float, ...] = ()
    item: str = ""
    #: What a control that is one of a set stands for - which plane a camera
    #: preset faces.
    value: str = ""


@dataclass(frozen=True)
class Entry:
    """One line of a menu as it should currently read.

    Resolved: the label is the words to draw, the keys are the ones this
    platform presses, and a toggle's tick is already decided. A renderer that
    has to consult anything else to draw this has been handed too little.
    """

    name: str
    label: str
    kind: Kind = Kind.ACTION
    keys: tuple[str, ...] = ()
    checked: bool = False
    enabled: bool = True
    #: Whether a rule stands above this entry, which is where one group of the
    #: menu ends and the next begins.
    separated: bool = False
    #: The lines of a submenu, for a choice or a list.
    items: tuple[Entry, ...] = ()
    #: What an item of a submenu carries: the size a point is to be drawn at,
    #: or which of the window's plots a line is about.
    value: float | int | str | None = None


@dataclass(frozen=True)
class Pointed:
    """The plot a click was over, which is what the tail of a menu is about.

    A right click on a curve, on a legend row or on a surface is a click about
    one plot, and the entries that answer it are named after it and read its
    two sticky looks. Nothing here is the plot itself: a snapshot carries what
    a menu has to know and not a plot list.
    """

    named: str
    kind: PlotKind
    connected: bool = False
    wire: bool = False


@dataclass(frozen=True)
class Flat:
    """A 2D window as its menu has to read it.

    Small on purpose. It is everything the description of the controls depends
    on and nothing else, so that a backend can build one on every right click
    without thinking about the cost, and so that a field added here is a field
    some entry visibly needs.
    """

    tracing: bool = False
    grid: bool = True
    legend: bool = True
    polar: bool = False
    equal: bool = True
    pointed: Pointed | None = None


@dataclass(frozen=True)
class Solid:
    """A 3D window as its menu has to read it.

    `surfaces` is how the window's own plots name themselves, in the order it
    holds them, which is what the `Remove` submenu lists and what an item of it
    indexes back into.
    """

    spinning: bool = False
    wired: bool = True
    boxed: bool = True
    named: bool = True
    legend: bool = True
    surfaces: tuple[str, ...] = ()
    pointed: Pointed | None = None


State = Flat | Solid

#: The sizes a data plot's points can be drawn at, which is the one choice
#: either window offers.
POINT_SIZES = (3.0, 5.0, 8.0, 12.0)

#: The plot a click was over, in the labels that are about it.
PLOT = "{plot}"


#: What a 2D window does. The order is the order of its menu, the groups are
#: where its rules fall, and the entries with no `menu` are the toolbar's
#: toggles and the keys no entry names.
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
        group=2,
    ),
    Control(
        name="image.export",
        label="Export...",
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
    # Ctrl+W is the desktop's second spelling of Close and belongs to the
    # browser in a page, so the page answers to the letter alone.
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
    # The toolbar's two toggles. They stand where their state is read rather
    # than on a menu that has to be opened to be read, which is why neither is
    # an entry; the words are still this table's.
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
    # The commands with neither an entry nor a button: stepping forward through
    # the view history, which is the other half of Back, and the two zooms.
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


#: What a 3D window does. A 3D window is a camera, so its menu is where to look
#: from; what the toolbar already shows the state of is not repeated on it.
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
        group=1,
    ),
    Control(
        name="image.export",
        label="Export...",
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
    # M rather than W, deliberately: Ctrl-W closes the window, and a frequent
    # toggle one slipped modifier from destroying the picture would be a poor
    # place for it.
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
    # One surface between wire and solid, which is the exception a legend row
    # offers rather than the every-surface `mesh` box, beside the words either
    # window removes a plot with.
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
    """The control this name is of, in the window that offers it.

    The table is asked for rather than searched for, because the two windows
    share names - both clear, both close, both hold a legend - and share none
    of the words: what `Clear` removes is plots in one window and surfaces in
    the other, and a lookup that guessed would sooner or later say the wrong
    one.
    """
    for one in controls:
        if one.name == name:
            return one
    raise KeyError(name)


def plain(one: Control) -> str:
    """A control's label before a menu knows what it is about.

    A menu built once and read many times has to be built with words that are
    true of nothing in particular, and `Remove` is the one entry whose words
    name a plot when a click has picked one out.
    """
    return one.label.replace(PLOT, "").strip()


def keys(one: Control, page: bool = False) -> tuple[str, ...]:
    """The keys that press a control here, which is not the same alphabet."""
    return one.web if page else one.keys


def listed(controls: tuple[Control, ...]) -> tuple[Control, ...]:
    """The controls that are on a context menu, in the order it offers them."""
    return tuple(one for one in controls if one.menu)


def choices(one: Control, page: bool = False) -> tuple[Entry, ...]:
    """What a choice offers, as the lines of its submenu.

    Written down rather than read off a snapshot, since the alternatives are
    the same however the window stands - a menu of point sizes offers four
    sizes whatever a point is currently drawn at.
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


def menu(state: State, page: bool = False) -> tuple[Entry, ...]:
    """The context menu of a window in this state, as it should currently read.

    Labels resolved, toggles ticked, and anything the state has nothing for
    left out: a menu is about the window it was raised over, and an entry that
    would name a curve the click was not near has nothing to name.
    """
    resolve = _flat if isinstance(state, Flat) else _solid
    controls = FLAT if isinstance(state, Flat) else SOLID
    entries: list[Entry] = []
    group: int | None = None
    for one in listed(controls):
        entry = resolve(one, state, page)
        if entry is None:
            continue
        entries.append(replace(entry, separated=group is not None and one.group != group))
        group = one.group
    return tuple(entries)


def card(state: State, page: bool = False) -> tuple[Entry, ...]:
    """The menu one legend row offers, which is about a plot and not a view.

    A menu of its own rather than the canvas's, because half of a canvas menu
    is about the view and a click on a legend row is about one plot - but the
    words are the canvas menu's own words, which is what makes the row and the
    plot one target with two places to hit it.
    """
    if state.pointed is None:
        return ()
    flat = isinstance(state, Flat)
    names = ("plot.remove", "points.connect") if flat else ("surface.wire", "plot.remove")
    controls = FLAT if flat else SOLID
    resolve = _flat if flat else _solid
    found = (resolve(control(name, controls), state, page) for name in names)
    return tuple(entry for entry in found if entry is not None)


def _flat(one: Control, state: Flat, page: bool) -> Entry | None:
    """One entry of a 2D window's menu, or None where it does not apply."""
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
    """One entry of a 3D window's menu, or None where it does not apply."""
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
    # A submenu of an empty window is offered greyed rather than left off,
    # because what it would list is what the window is empty of.
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
    """The entry a control comes to, with the words a menu draws filled in."""
    label = one.flipped if flipped and one.flipped else one.label
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
