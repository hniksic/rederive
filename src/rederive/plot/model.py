"""What a plot is, apart from whatever draws it: identity, look, and no numbers.

One entry of a window's plot list, as the side that routes plots around
windows holds it. It is what to draw and what it is called, and deliberately
nothing else: no drawing item, because that belongs to the backend that made
it, and no samples, because those belong to the side that does the sampling.
That split is what lets a plot list live somewhere with no numpy in it at all -
the browser's main thread - while the arrays go straight from wherever they are
computed to wherever they are drawn.

The identity is `worksheet` and `label` together, which is what makes
re-plotting `#3` replace its own curve while a `#3` from another algebra
overlay is a second plot. Everything else here is either what the expression
is (the tree, the context it is read in, the variables it holds) or how it is
to look (the color, the points of a data plot, the wire of a surface) - the
things a window is told rather than the things it works out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rederive.engine.context import Context
from rederive.model.expr import Node
from rederive.plot.protocol import Options, PlotKind
from rederive.syntax import ParseState

__all__ = [
    "FIELDS",
    "FUNCTIONS",
    "PALETTE",
    "PAPER_PALETTE",
    "SOLID_PALETTE",
    "SOLID_PAPER",
    "Plot",
    "Surface",
    "naming",
    "pointed",
    "reading",
    "roughly",
]

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

#: The surface colors, which are the curve colors turned by one. A shaded solid
#: in white is a white shape with a white shape behind it - the palette's first
#: color is the right one for a stroke on a dark canvas and the worst one for a
#: lit surface - so the order starts at the second and comes round to it last.
SOLID_PALETTE = PALETTE[1:] + PALETTE[:1]
SOLID_PAPER = PAPER_PALETTE[1:] + PAPER_PALETTE[:1]

#: How much of an expression a legend entry, a menu item or a status line shows
#: of it. A data matrix is one expression and a hundred pairs of numbers, and a
#: legend entry three times the width of the window names nothing legibly, so a
#: long one is cut and marked as cut. Nothing is lost: the whole expression is
#: in the worksheet, under the label the entry begins with.
NAME_WIDTH = 48
ELLIPSIS = "…"

#: How big a data point is drawn where nothing has an opinion about it.
POINT_SIZE = 5.0

#: The parameter range a parametric or polar plot takes when nothing says
#: otherwise: one turn, written in whichever angle unit the worksheet is set
#: to. One turn is what draws a closed curve of the ordinary ones, and the
#: range fields are where a different piece of the parameter is asked for,
#: while the picture is on the screen.
DEFAULT_TURN = math.pi
DEFAULT_TURN_DEGREES = 180.0

#: The kinds parametrized by the abscissa, which are the ones a marker rides
#: at an x and the ones features are sought on.
FUNCTIONS = frozenset({PlotKind.CURVE, PlotKind.FAMILY})

#: The kinds evaluated over a grid rather than along a line. They are areas,
#: recomputed for whatever the view now shows.
FIELDS = frozenset({PlotKind.IMPLICIT, PlotKind.REGION})


def naming(label: str, text: str) -> str:
    """What a plot is called wherever a window has room for one line of it.

    The label first, because that is what identifies it in the worksheet, and
    then as much of the expression as fits. Both window kinds name their plots
    this way, and neither renders an expression itself: `text` is what the app
    wrote, in the one spelling the whole program uses.
    """
    name = f"{label}  {text}" if label else text
    if len(name) <= NAME_WIDTH:
        return name
    return name[: NAME_WIDTH - len(ELLIPSIS)].rstrip() + ELLIPSIS


#: How a number a window reads out is written. Six decimals is what a reading
#: is worth from a picture: enough to paste into the algebra window and see it
#: agree, short enough to fit twice on one line.
FORMAT = "{:.6f}"


def reading(value: float) -> str:
    """A number as a status line writes it: six decimals, or a word.

    Here rather than in a window because both backends read a curve out and a
    point sent home from either has to be the same text.
    """
    if not math.isfinite(value):
        return "undefined"
    return FORMAT.format(value)


def roughly(value: float) -> str:
    """A number as a sentence about the view writes it, with no trailing noise.

    A reading wants every digit it has; a range does not - `-5 ≤ x ≤ 5` is what
    the window is showing, and six decimals of it would be six decimals of
    nothing.
    """
    return f"{value:g}"


def pointed(x: float, y: float) -> str:
    """The point under a trace marker, spelled for the worksheet: `[x, y]`.

    One spelling for every route home, so that what a clipboard carries, what
    an event sends and what a browser pane hands back are the same text.
    """
    return f"[{reading(x)}, {reading(y)}]"


@dataclass
class Plot:
    """One entry of a 2D window's plot list: what to draw and what it is called.

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
    #: What went wrong the last time this plot was sampled, if anything.
    trouble: str = ""
    #: The parse state the expression arrived under, which is what the range
    #: fields read a typed bound with: the same grammar the worksheet reads.
    state: ParseState = field(default_factory=ParseState)
    #: The parameter range now drawn over, and the bounds typed in the range
    #: fields that the sampling has yet to turn into one - trees, since what a
    #: person types is `-π` and what it is worth is arithmetic.
    trange: tuple[float, float] = (-DEFAULT_TURN, DEFAULT_TURN)
    bounds: tuple[Node, Node] | None = None
    #: How a data plot is drawn, resolved from the request and the sticky
    #: preferences when the plot is added, and changed by its right-click menu.
    connected: bool = False
    point_size: float = POINT_SIZE

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


@dataclass
class Surface(Plot):
    """One entry of a 3D window's plot list: what to draw and what it is called.

    A plot like any other, with the two things a solid has an opinion about
    that a curve has not: whether it is drawn as the wire grid of its samples,
    and whether it is being shown at all - a 2D window answers the second from
    its drawing item, and a surface is its own answer.
    """

    color: str = SOLID_PALETTE[0]
    paper: str = SOLID_PAPER[0]
    #: Whether this surface draws as the wire grid of its samples rather than
    #: as a shaded solid. A property of the surface, not the window: the
    #: toolbar's `mesh` box flips every surface at once, and the legend's
    #: right-click carries the per-surface exception.
    wire: bool = False
    hidden: bool = False

    @property
    def visible(self) -> bool:
        return not self.hidden

    @property
    def axes(self) -> tuple[str, str]:
        """The two variables of the floor, in the order the axes take them."""
        names = self.options.variables
        if len(names) >= 2:
            return names[0], names[1]
        return (names[0] if names else "x"), "y"

    @property
    def vertical(self) -> str:
        """What the upright axis is called: the expression's name for it, or z.

        `z = x^2 + y^2` names it and an expression that is only `x^2 + y^2`
        does not, so the second gets the letter every reader supplies anyway.
        """
        return self.options.vertical or "z"
