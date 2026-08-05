"""The vocabulary the app and the plot host speak, and nothing else.

It mirrors `engine/boundary.py` and lives under the same prohibition: no sympy,
no numpy, no Qt. The app process holds a `Add` in its hands without loading a
computer algebra system or a widget toolkit, and the host is the only side that
knows what either word costs.

Everything here is data, and all of it is picklable, because it travels down a
multiprocessing pipe. A request is answered synchronously with a cheap reply -
an acknowledgement, a window number, or a refusal in words - and the heavy work
it asks for happens in the host afterwards. What that work then has to say for
itself comes back as an event rather than as a reply: a window the user closed
and a curve that would not evaluate arrive long after the request that made
them, so they are their own messages.

The vocabulary is complete for every plot kind the design names, including the
ones no window can draw yet. A kind is a word both sides already know, so later
work adds behaviour rather than protocol - and the app can classify an
expression it cannot yet plot, which is what lets a refusal say what it is
refusing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from rederive.engine.context import Context
from rederive.model.expr import Node

__all__ = [
    "Add",
    "Closed",
    "Delete",
    "Describe",
    "Done",
    "Event",
    "Options",
    "PlotInfo",
    "PlotKind",
    "Placed",
    "Prefer",
    "Refused",
    "Removed",
    "Reply",
    "Request",
    "SetCurrent",
    "Shutdown",
    "Trouble",
    "Where",
    "Which",
    "WindowInfo",
    "WindowKind",
    "Windows",
    "DRAWN",
    "PARAMETRIZED",
    "UNDRAWN",
    "dimension",
    "titled",
]

#: The two ids that carry no request number: the handshake a started host
#: sends once, and the tag every asynchronous event wears. Negative, so no
#: request can be mistaken for either.
READY = -1
EVENT = -2


class PlotKind(StrEnum):
    """What a classified expression is to be drawn as.

    The word is the classification of section 5 and the drawing instruction
    both, which is why the 3D family is a kind of its own rather than a surface
    with several expressions: one `Add` carries one expression, and what the
    kind says is how to read it.

    `POLAR` never comes out of classification. A window in polar mode reads a
    newly added univariate expression as `r = f(θ)`, so the command promotes a
    curve to a polar curve when the target window says it is polar; every
    other kind is unaffected by the toggle.
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
    """Which of the two kinds of window a plot needs. Windows hold one kind."""

    TWO_D = "2D"
    THREE_D = "3D"


#: Which window each kind belongs in. The only thing that decides whether a
#: Plot command lands in a flat window or a solid one.
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


#: The kinds a window can draw today, which is now all of them. Classification
#: knows every kind, so an expression is always recognized for what it is; this
#: is the one place that says which of them a window has been taught to draw,
#: and it is what both sides refuse by. It is kept, empty of exclusions, because
#: the next kind the vocabulary grows - a parametric surface, a slider - is
#: recognized before it is drawn, and a refusal has to be able to name it.
DRAWN = frozenset(PlotKind)

#: The kinds drawn over a parameter, which are the ones the Plot command asks
#: a range for before it sends anything.
PARAMETRIZED = frozenset({PlotKind.PARAMETRIC, PlotKind.POLAR})

#: What a request for one of the others is refused with. The word is the
#: classification's own, so the refusal names what was recognized.
UNDRAWN = "{kind} plots are not implemented yet"


class Where(StrEnum):
    """Which window a request is about, where it is not about a numbered one.

    `CURRENT` is the window of the matching kind the next plot lands in,
    `NEW` asks for one that does not exist yet, and `MRU` is the window that
    last received a plot or a deletion, whatever its kind - which is what the
    Delete submenu acts on.
    """

    CURRENT = "current"
    NEW = "new"
    MRU = "mru"


class Which(StrEnum):
    """Which of a window's plots a `Delete` takes out.

    `ONE` is the deletion the window's own right-click menu drives: the plot
    is named by the worksheet and label it was added under, since that pair is
    what identifies a plot everywhere.
    """

    ALL = "all"
    FIRST = "first"
    LAST = "last"
    BUTLAST = "butlast"
    ONE = "one"


#: How a window is titled, and what says it is the one the next plot of its
#: kind will land in. Both sides know the spelling so that a test can name a
#: window without having asked the host what it calls itself.
TITLE = "Rederive {kind} plot {number}"
CURRENT_MARK = " (current)"


def titled(kind: WindowKind, number: int, current: bool = False) -> str:
    """What the window manager shows over a plot window."""
    title = TITLE.format(kind=kind.value, number=number)
    return title + CURRENT_MARK if current else title


@dataclass(frozen=True)
class Options:
    """The per-kind extras an `Add` carries, all of them optional.

    `variables` is the classification's reading of the expression, in the order
    the axes take them: the abscissa for a curve, the parameter for a
    parametric pair or a polar curve, the two horizontal axes for a surface.
    The host needs them because it lambdifies over them, and the window needs
    them because their names label the axes.

    `vertical` is the name the vertical axis carries where the expression gave
    it one - the `z` of `z = x^2 + y^2`, which classification reads off the
    lone side of the equation.

    `texts` is the one-line rendering of each element of a family, written
    app-side like `text` is: a family is one request that becomes several
    curves, and the host renders no expression of its own to name them with.

    `minimum` and `maximum` are the parameter range a parametric or polar plot
    is drawn over, asked for on a field line before the request is sent. They
    are expressions rather than numbers because `-π` is what the field offers
    and what a person types: turning one into a float is arithmetic, and the
    app does no arithmetic. The host evaluates them once, when the plot is
    added.

    The rest are how a data plot draws its points; None means "whatever the
    preferences say" rather than a value, and the host fills it in from the
    `Prefer` it was last sent. So a plot added without an opinion follows the
    preferences and one that has an opinion keeps it.
    """

    variables: tuple[str, ...] = ()
    vertical: str = ""
    texts: tuple[str, ...] = ()
    minimum: Node | None = None
    maximum: Node | None = None
    connected: bool | None = None
    point_size: float | None = None


# -- requests -----------------------------------------------------------------


@dataclass(frozen=True)
class Add:
    """Plot one expression, in the window `window` names.

    `worksheet` is an opaque id of the algebra worksheet the label belongs to:
    two overlays can each own a `#3`, and a plot is keyed by the pair, so
    re-plotting `#3` from the same worksheet replaces its curve while a `#3`
    from another one is a second plot.

    `text` is the expression written on one line, made app-side with the syntax
    writer. The host never renders an expression itself - it has the tree only
    to evaluate it - which is what keeps one spelling of an expression in the
    program.
    """

    worksheet: int
    node: Node
    context: Context
    kind: PlotKind
    window: int | Where = Where.CURRENT
    label: str = ""
    text: str = ""
    options: Options = field(default_factory=Options)


@dataclass(frozen=True)
class Delete:
    """Take plots out of a window, `which` saying how many and which end."""

    window: int | Where = Where.MRU
    which: Which = Which.LAST
    worksheet: int = 0
    label: str = ""


@dataclass(frozen=True)
class SetCurrent:
    """Make a window the current one for its kind, as `Plot Window` does."""

    window: int


@dataclass(frozen=True)
class Describe:
    """What windows are open and what is in them.

    The app tracks no window state of its own: the prompts that depend on a
    window - the polar toggle, the number the Window field opens on - read it
    from here when the command runs, and the tests observe the host the same
    way.
    """


@dataclass(frozen=True)
class Prefer:
    """The persistent plot preferences, as the settings store now holds them.

    A request of its own rather than more fields on `Options`, because the two
    are about different things: `Options` is what one expression is drawn with
    and this is what the *program* has been told to open its next window with.
    Widening `Options` would have meant sending the same four values with every
    plot and leaving the host to guess which of them it was being told about.

    The app sends one when a host starts - a fresh host knows only the
    dataclass defaults, and the session's preferences may have moved since the
    program opened - and again whenever `Options Plot` or a loaded state file
    changes them. They travel in front of the next request rather than the
    moment they change, since what they matter to is the next window and the
    next plot and nothing sooner.

    Defaults and nothing more. A window already on screen keeps the framing
    lock and the grid it was built with, and a plot already drawn keeps the
    point size its right-click menu gave it; the toggles never write back.
    """

    equal_scales: bool = True
    grid: int = 64
    connected: bool = False
    point_size: float = 5.0


@dataclass(frozen=True)
class Shutdown:
    """Close every window and end the host, which is what leaving the app is."""


Request = Add | Delete | SetCurrent | Describe | Prefer | Shutdown


# -- replies ------------------------------------------------------------------


@dataclass(frozen=True)
class Done:
    """The request was taken. Nothing more is known yet, by design."""


@dataclass(frozen=True)
class Refused:
    """The request cannot be honored, and this is why in words.

    The app puts the message on the message line behind `Plot: `, so it reads
    as one sentence: `Plot: no plot window`.
    """

    message: str


@dataclass(frozen=True)
class Placed:
    """The number of the window that took the plot, or was made current."""

    window: int


@dataclass(frozen=True)
class Removed:
    """How many plots came out of which window, for the message line to say."""

    window: int
    count: int


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
    """One open window, as `Describe` reports it.

    `polar` is None for a 3D window, which has no such mode, rather than False:
    absent and off are different answers and the prompt that reads it wants to
    tell them apart.
    """

    number: int
    kind: WindowKind
    title: str
    current: bool
    plots: tuple[PlotInfo, ...] = ()
    xrange: tuple[float, float] = (0.0, 0.0)
    yrange: tuple[float, float] = (0.0, 0.0)
    polar: bool | None = None


@dataclass(frozen=True)
class Windows:
    """Every open window, in creation order."""

    windows: tuple[WindowInfo, ...] = ()


Reply = Done | Refused | Placed | Removed | Windows


# -- events -------------------------------------------------------------------


@dataclass(frozen=True)
class Closed:
    """The user closed a window. The app is told so that it can say so."""

    window: int


@dataclass(frozen=True)
class Trouble:
    """One plot could not be drawn, and this is what went wrong with it.

    An expression that classifies but will not evaluate - an operator sympy
    has no numeric reading of, a lambdify that raises - reports itself rather
    than drawing nothing, which is the whole of the never-silence rule as it
    applies to the host.
    """

    window: int
    label: str
    message: str


Event = Closed | Trouble
