"""Messages exchanged between the app and the plot host.

Like `engine/boundary.py`, this module imports no sympy, numpy or Qt, so the app can
build an `Add` without loading either.

All messages are picklable data sent over a multiprocessing pipe. A request gets a
cheap synchronous reply (an acknowledgement, a window number, or a refusal message),
and the heavy work happens in the host afterwards. Later outcomes, such as a window
closed by the user or a curve that failed to evaluate, arrive as separate events.

`PlotKind` covers every plot kind in the design, including any not yet drawable, so
the app can classify an expression and name it in a refusal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from rederive.engine.context import Context
from rederive.model.expr import Node
from rederive.syntax import ParseState

__all__ = [
    "Add",
    "Closed",
    "Describe",
    "Done",
    "Event",
    "Options",
    "PlotInfo",
    "PlotKind",
    "Placed",
    "Prefer",
    "Preferred",
    "Refused",
    "Release",
    "Reply",
    "Request",
    "Shutdown",
    "Stepped",
    "Traced",
    "Trouble",
    "Where",
    "WindowInfo",
    "WindowKind",
    "Windows",
    "DRAWN",
    "PARAMETRIZED",
    "UNDRAWN",
    "dimension",
    "titled",
]

#: Message ids that are not request numbers: the host's one-time startup handshake,
#: and the tag of asynchronous events. Negative so they can't clash with requests.
READY = -1
EVENT = -2


class PlotKind(StrEnum):
    """How a classified expression is drawn.

    The kind is both the classification (design section 5) and the drawing instruction.
    That's why a 3D family is its own kind: one `Add` carries one expression, and the
    kind says how to read it.

    Classification never produces `POLAR`, and no `Add` carries it. Polar is a mode of
    the 2D view that reads every univariate curve as `r = f(θ)`; the kind marks which
    plots in the window's list are read that way.
    """

    CURVE = "curve"
    FAMILY = "family"
    PARAMETRIC = "parametric"
    POLAR = "polar"
    DATA = "data"
    IMPLICIT = "implicit"
    REGION = "region"
    SURFACE = "surface"
    SURFACES = "surfaces"


class WindowKind(StrEnum):
    """The kind of window a plot needs. Each window holds plots of one kind."""

    TWO_D = "2D"
    THREE_D = "3D"


#: The window kind for each plot kind, which decides whether a plot goes to a 2D or a
#: 3D window.
DIMENSION: dict[PlotKind, WindowKind] = {
    PlotKind.CURVE: WindowKind.TWO_D,
    PlotKind.FAMILY: WindowKind.TWO_D,
    PlotKind.PARAMETRIC: WindowKind.TWO_D,
    PlotKind.POLAR: WindowKind.TWO_D,
    PlotKind.DATA: WindowKind.TWO_D,
    PlotKind.IMPLICIT: WindowKind.TWO_D,
    PlotKind.REGION: WindowKind.TWO_D,
    PlotKind.SURFACE: WindowKind.THREE_D,
    PlotKind.SURFACES: WindowKind.THREE_D,
}


def dimension(kind: PlotKind) -> WindowKind:
    """The kind of window `kind` is drawn in."""
    return DIMENSION[kind]


#: The kinds windows can draw, currently all of them. Both sides refuse plots of other
#: kinds. It is kept so that future kinds (a parametric surface, a slider) can be
#: recognized before they are drawable and named in the refusal.
DRAWN = frozenset(PlotKind)

#: Kinds drawn over a parameter, to which the 2D window's range fields apply. They are
#: drawn over one turn until the fields say otherwise.
PARAMETRIZED = frozenset({PlotKind.PARAMETRIC, PlotKind.POLAR})

#: Refusal for a kind not in `DRAWN`, naming the recognized kind.
UNDRAWN = "{kind} plots are not implemented yet"


class Where(StrEnum):
    """A request's target window, other than a numbered one.

    `NEW` asks for a new window. A request naming no window goes to the receiver: the
    window of its kind the user last activated.
    """

    NEW = "new"


#: Window titles, and the mark of the window that receives the next plot of its kind.
#: Windows are titled by their contents, which is what tells them apart in the taskbar
#: and alt-tab list. Windows with the same contents share a title; the window number
#: is only a protocol key and doesn't appear in it.
TITLE = {WindowKind.TWO_D: "Rederive plot", WindowKind.THREE_D: "Rederive 3D plot"}
CURRENT_MARK = " (current)"

#: Maximum length of a title's contents before truncation with "...", to fit a taskbar
#: button.
TITLE_WIDTH = 60


def titled(kind: WindowKind, contents: Sequence[str] = (), current: bool = False) -> str:
    """The window title of a plot window, listing its plots.

    An empty window is titled by its kind alone.
    """
    text = ", ".join(contents)
    if len(text) > TITLE_WIDTH:
        text = text[: TITLE_WIDTH - 3].rstrip() + "..."
    title = f"{text} - {TITLE[kind]}" if text else TITLE[kind]
    return title + CURRENT_MARK if current else title


@dataclass(frozen=True)
class Options:
    """Optional per-kind extras of an `Add`.

    `variables` are the expression's variables in axis order: the abscissa of a curve,
    the parameter of a parametric or polar curve, or the two horizontal axes of a
    surface. The host lambdifies over them, and the window labels axes with them.

    `vertical` is the vertical axis name if the expression gives one, like the `z` of
    `z = x^2 + y^2`.

    `texts` are the one-line renderings of a family's elements, made app-side like
    `text`, since the host never renders expressions.

    The rest describe how a data plot draws its points. None means "use the
    preferences", filled in by the host from the last `Prefer` it received.
    """

    variables: tuple[str, ...] = ()
    vertical: str = ""
    texts: tuple[str, ...] = ()
    connected: bool | None = None
    point_size: float | None = None


# -- requests -----------------------------------------------------------------


@dataclass(frozen=True)
class Add:
    """Plot one expression in the window `window` names.

    With `window` None (the usual case), the plot goes to the receiver of its kind, the
    window the user last activated, opening one if none exists.

    `worksheet` is an opaque id of the algebra worksheet the label belongs to. Plots are
    keyed by both, so re-plotting `#3` from the same worksheet replaces its curve, while
    `#3` from another worksheet is a separate plot.

    `text` is the expression on one line, rendered app-side by the syntax writer. The
    host never renders expressions, it only evaluates the tree.

    `state` is the expression's parse state. The window's fields (parameter range,
    surface domain) accept expressions, which the window parses with this state and
    evaluates in the plot's context.

    `demonstrating` marks a demonstration step rather than a user command. The window
    is cleared of other plots first, and shown without taking keyboard focus, because
    the program is waiting for a key to continue.
    """

    worksheet: int
    node: Node
    context: Context
    kind: PlotKind
    window: int | Where | None = None
    label: str = ""
    text: str = ""
    options: Options = field(default_factory=Options)
    state: ParseState = field(default_factory=ParseState)
    demonstrating: bool = False


@dataclass(frozen=True)
class Describe:
    """Ask which windows are open and what they contain.

    The app keeps no window state of its own and queries it this way when needed, as do
    the tests.
    """


@dataclass(frozen=True)
class Prefer:
    """The sticky plot preferences, as last reported by a host.

    These are separate from `Options`, which describe how one expression is drawn.
    Preferences are the defaults for the next surface and data plot: the grid and
    wireframe of the last surface, and the style the last data plot was left in.

    The host reports changes to sticky controls in a `Preferred` event. The app sends
    the values back in this request when a host starts (a fresh host knows only the
    defaults) and when a loaded state file changes them. They are sent before the next
    request rather than immediately, since they only affect the next plot.

    They are only defaults: open windows keep their grid, and drawn plots keep their
    point size.

    Equal scales is deliberately not a preference. A new window always opens with equal
    scales, and a one-off `1:1` toggle shouldn't change the default.
    """

    grid: int = 64
    connected: bool = False
    point_size: float = 5.0
    wire: bool = True


@dataclass(frozen=True)
class Release:
    """The demonstration is over, so its windows no longer stay on top.

    Demonstration steps arrive with `demonstrating` set and stay above other windows,
    so the waiting terminal doesn't cover them. Only the app knows when the
    demonstration ends: on a key, on Esc, on a refused step, or after the last step.

    `finished` means it ran to the end, and its windows are closed. Otherwise the
    picture stays, so Esc can be used to stop a gallery and keep the current picture.

    Sent only to a running host; there is no point starting one just to release
    nothing.
    """

    finished: bool = False


@dataclass(frozen=True)
class Shutdown:
    """Close every window and end the host, when the app exits."""


Request = Add | Describe | Prefer | Release | Shutdown


# -- replies ------------------------------------------------------------------


@dataclass(frozen=True)
class Done:
    """The request was accepted. Its outcome isn't known yet."""


@dataclass(frozen=True)
class Refused:
    """The request was refused, with a message saying why.

    The app shows the message after `Plot: `, as in `Plot: no plot window`.
    """

    message: str


@dataclass(frozen=True)
class Placed:
    """The number of the window that received the plot.

    `replaced` means the plot replaced one with the same worksheet and label, so the
    acknowledgement says `Replotting`.
    """

    window: int
    replaced: bool = False


@dataclass(frozen=True)
class PlotInfo:
    """One entry of a window's plot list, as `Describe` reports it."""

    worksheet: int
    label: str
    text: str
    kind: PlotKind
    hidden: bool = False


@dataclass(frozen=True)
class WindowInfo:
    """One open window, as `Describe` reports it."""

    number: int
    kind: WindowKind
    title: str
    current: bool
    plots: tuple[PlotInfo, ...] = ()
    xrange: tuple[float, float] = (0.0, 0.0)
    yrange: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class Windows:
    """Every open window, in creation order."""

    windows: tuple[WindowInfo, ...] = ()


Reply = Done | Refused | Placed | Windows


# -- events -------------------------------------------------------------------


@dataclass(frozen=True)
class Closed:
    """The user closed a window. The app reports it on the message line."""

    window: int


@dataclass(frozen=True)
class Trouble:
    """A plot could not be drawn, and why.

    An expression that classifies but won't evaluate (e.g. an operator sympy can't
    evaluate numerically, or a failing lambdify) reports an error rather than silently
    drawing nothing.
    """

    window: int
    label: str
    message: str


@dataclass(frozen=True)
class Traced:
    """A point sent from a plot window to be added as a worksheet entry.

    While tracing, a key appends the point under the marker (or a just-refined root or
    extremum) to the worksheet as a new algebra entry. `worksheet` is the id from the
    plot's `Add`, and `text` is the point exactly as `Ctrl-C` copies it: `[x, y]` with
    six decimals.
    """

    worksheet: int
    text: str


@dataclass(frozen=True)
class Stepped:
    """A key the plot window doesn't use was pressed in it during a demonstration.

    The program prompts for a key to continue, but some desktops (GNOME) give keyboard
    focus to a newly shown window anyway, so the key often lands in the plot window,
    which forwards it.

    Only keys the window doesn't use (Space, Enter and the like) are forwarded. The
    window keeps handling its own keys, so a gallery can be traced, rotated and zoomed,
    and Esc closes the picture rather than ending the demonstration.
    """


@dataclass(frozen=True)
class Preferred:
    """A sticky control changed in a plot window; these are the new preferences.

    The whole `Prefer` is sent rather than the changed value, so adding a preference
    only means adding a field. The host can't keep these for the session (it starts on
    demand, or never on a machine without a display), so the app stores them and saves
    them in the state file.
    """

    preferences: Prefer


Event = Closed | Trouble | Traced | Stepped | Preferred
