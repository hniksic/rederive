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

    `POLAR` never comes out of classification, and no `Add` carries it: polar
    is a property of the 2D window's view, whose toggle reads every univariate
    curve in it as `r = f(θ)`. The kind is how such a window's plot list says
    which of its plots are being read that way.
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

#: The kinds drawn over a parameter, which are the ones the 2D window's range
#: fields apply to. Drawn over one turn until those fields say otherwise.
PARAMETRIZED = frozenset({PlotKind.PARAMETRIC, PlotKind.POLAR})

#: What a request for one of the others is refused with. The word is the
#: classification's own, so the refusal names what was recognized.
UNDRAWN = "{kind} plots are not implemented yet"


class Where(StrEnum):
    """Which window a request is about, where it is not about a numbered one.

    `NEW` asks for a window that does not exist yet. A request that names no
    window at all goes to the receiver: the window of its kind the user last
    touched, which the host learns from the windows' own activation events.
    """

    NEW = "new"


#: How a window names itself, and what says it is the one the next plot of its
#: kind will land in. A window is titled by what it holds - the taskbar and the
#: alt-tab list are where a title is read, and there the contents are what tell
#: one plot from another. Windows holding the same expressions share a title,
#: which is how content-named windows behave everywhere; the number stays a
#: protocol key and no part of the name.
TITLE = {WindowKind.TWO_D: "Rederive plot", WindowKind.THREE_D: "Rederive 3D plot"}
CURRENT_MARK = " (current)"

#: How much of the contents a title carries before the rest becomes "...":
#: enough to tell two windows apart, little enough to fit a taskbar button.
TITLE_WIDTH = 60


def titled(kind: WindowKind, contents: Sequence[str] = (), current: bool = False) -> str:
    """What the window manager shows over a plot window: the plots it holds.

    An empty window is titled by its kind alone.
    """
    text = ", ".join(contents)
    if len(text) > TITLE_WIDTH:
        text = text[: TITLE_WIDTH - 3].rstrip() + "..."
    title = f"{text} - {TITLE[kind]}" if text else TITLE[kind]
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

    The rest are how a data plot draws its points; None means "whatever the
    preferences say" rather than a value, and the host fills it in from the
    `Prefer` it was last sent. So a plot added without an opinion follows the
    preferences and one that has an opinion keeps it.
    """

    variables: tuple[str, ...] = ()
    vertical: str = ""
    texts: tuple[str, ...] = ()
    connected: bool | None = None
    point_size: float | None = None


# -- requests -----------------------------------------------------------------


@dataclass(frozen=True)
class Add:
    """Plot one expression, in the window `window` names.

    A `window` of None is the ordinary case: the plot goes to the receiver of
    its kind - the window the user last touched - and opens one where no window
    of that kind exists.

    `worksheet` is an opaque id of the algebra worksheet the label belongs to:
    two overlays can each own a `#3`, and a plot is keyed by the pair, so
    re-plotting `#3` from the same worksheet replaces its curve while a `#3`
    from another one is a second plot.

    `text` is the expression written on one line, made app-side with the syntax
    writer. The host never renders an expression itself - it has the tree only
    to evaluate it - which is what keeps one spelling of an expression in the
    program.

    `state` is the parse state the expression was read under. The window's own
    fields - the parameter range of a parametric or polar plot, a surface's
    domain - take expressions rather than floats, and the syntax reader is
    dependency-free, so the window parses what is typed in them under this
    state and evaluates it under the plot's own context.

    `demonstrating` says this plot is a step of a demonstration rather than a
    command, which changes two things about how it lands. The window is emptied
    of everything else first, a gallery being a sequence of unrelated pictures
    where one drawn over the last is a picture of neither; and the window is
    shown without taking the keyboard, because the program is waiting for a key
    to take the next step and the keyboard has to stay where the program is.
    Nothing a user commands sets it: a plot anyone asked for is welcome to the
    keyboard, and Plot adds where a demonstration replaces.
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
    """What windows are open and what is in them.

    The app tracks no window state of its own: whatever it needs to know about
    a window it reads from here when a command runs, and the tests observe the
    host the same way.
    """


@dataclass(frozen=True)
class Prefer:
    """The sticky plot values: what the app last learned back from a host.

    A request of its own rather than more fields on `Options`, because the two
    are about different things: `Options` is what one expression is drawn with
    and this is what the next surface and the next data plot start out as -
    the grid and the wire look the last surface was given, and the way the
    last data plot was left. Widening `Options` would have meant sending the
    same values with
    every plot and leaving the host to guess which of them it was being told
    about.

    The values move in both directions. The host reports every change of a
    sticky control in a `Preferred` event, and the app hands the values back
    in a request of this shape when a host starts - a fresh host knows only
    the dataclass defaults - and again when a loaded state file changes them.
    They travel in front of the next request rather than the moment they
    change, since what they matter to is the next plot and nothing sooner.

    Defaults and nothing more. A window already on screen keeps the grid it
    was built with, and a plot already drawn keeps the point size its
    right-click menu gave it.

    Equal scales is deliberately absent: a new window always opens with equal
    scales, the window's `1:1` toggle serves the exception, and nothing
    persists - a one-off framing choice must not silently become the default
    that reshapes the next circle.
    """

    grid: int = 64
    connected: bool = False
    point_size: float = 5.0
    wire: bool = True


@dataclass(frozen=True)
class Release:
    """The demonstration is over: no window need stay in front for it any more.

    A step of a gallery arrives with `demonstrating` set and is left above the
    other windows, the program that is waiting for the next key being a
    terminal that would otherwise bury the picture the moment it was touched.
    That has to end when the demonstration does, and only the app knows when
    that is - a demonstration ends on a key, on Esc, on a step that refused, or
    on the last step there was.

    Sent to a host that is running and to no other: a demonstration that opened
    no window has nothing to release, and starting a host to tell it so would
    open a toolkit to say nothing.
    """


@dataclass(frozen=True)
class Shutdown:
    """Close every window and end the host, which is what leaving the app is."""


Request = Add | Describe | Prefer | Release | Shutdown


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
    """The number of the window that took the plot, and how it took it.

    `replaced` says the plot took the place of one already there - same
    worksheet and label - rather than adding a curve, which is the word the
    acknowledgement message turns on: `Replotting` is how replacement teaches
    itself.
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


@dataclass(frozen=True)
class Traced:
    """A point the user sent home from a plot window, to be authored as an entry.

    While tracing, one key appends the point under the marker - or the root or
    extremum just refined - to the worksheet as a new algebra entry, ready to
    compute with. `worksheet` is the id the `Add` that made the plot carried,
    and `text` is the point exactly as `Ctrl-C` puts it on the clipboard -
    `[x, y]`, six decimals - so the clipboard route and this one enter the same
    expression.
    """

    worksheet: int
    text: str


@dataclass(frozen=True)
class Preferred:
    """A sticky control moved in a plot window, and these are the values now.

    The whole of `Prefer` rather than the one value that moved, so that the
    app stores a complete answer and a new sticky value is one more field on
    `Prefer` rather than a message of its own. The host cannot hold these for
    the session - it starts on demand, and on a display-less machine never
    starts at all - so the app is told, keeps them, and writes them into a
    state file; this event is the one place a toggle writes back, and the
    reversal is deliberate.
    """

    preferences: Prefer


Event = Closed | Trouble | Traced | Preferred
