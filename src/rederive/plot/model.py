"""The backend-independent description of a plot: identity and appearance.

A `Plot` is an entry in a window's plot list: what to draw and its name. It holds no
drawing item (that belongs to the backend) and no samples (those belong to the sampler),
so a plot list can live without numpy, e.g. on the browser's main thread, while arrays
go directly from where they are computed to where they are drawn.

A plot is identified by `worksheet` and `label` together, so re-plotting `#3` replaces
its curve, while `#3` from another algebra overlay is a separate plot. The remaining
fields describe the expression (tree, context, variables) or its appearance (color,
data point style, surface wireframe).
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
    "SOLIDS",
    "SOLID_PALETTE",
    "SOLID_PAPER",
    "Plot",
    "Surface",
    "naming",
    "pointed",
    "reading",
    "roughly",
    "written",
]

#: The curve palette, cycled in this order: bright colors for the dark canvas. Eight is
#: enough since more curves in one window are rare.
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

#: The palette for image exports, which use a white background: each color is the
#: corresponding one above, darkened to read on white.
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

#: Surface colors: the curve palette rotated by one, because white, fine for curves,
#: is the worst color for a shaded surface.
SOLID_PALETTE = PALETTE[1:] + PALETTE[:1]
SOLID_PAPER = PAPER_PALETTE[1:] + PAPER_PALETTE[:1]

#: Maximum length of a plot's name in the legend, menus and status line. Longer names
#: (e.g. a data matrix) are truncated with an ellipsis; the full expression is in the
#: worksheet under the label.
NAME_WIDTH = 48
ELLIPSIS = "…"

#: Default size of a data point.
POINT_SIZE = 5.0

#: Default half-range of a parametric or polar plot's parameter: one full turn, in the
#: worksheet's angle unit, which closes most ordinary curves. The range fields can
#: change it.
DEFAULT_TURN = math.pi
DEFAULT_TURN_DEGREES = 180.0

#: Kinds parametrized by the abscissa: the trace marker follows x, and features are
#: searched for on them.
FUNCTIONS = frozenset({PlotKind.CURVE, PlotKind.FAMILY})

#: Kinds evaluated over a grid rather than along a line, recomputed for the current
#: view.
FIELDS = frozenset({PlotKind.IMPLICIT, PlotKind.REGION})

#: Kinds drawn as a height over a rectangle, held by 3D windows. They are evaluated
#: over a grid like fields, as functions of two variables.
SOLIDS = frozenset({PlotKind.SURFACE, PlotKind.SURFACES})


def naming(label: str, text: str) -> str:
    """A plot's one-line name: the label, then as much of the expression as fits.

    `text` is the expression as rendered by the app; windows never render expressions
    themselves.
    """
    name = f"{label}  {text}" if label else text
    if len(name) <= NAME_WIDTH:
        return name
    return name[: NAME_WIDTH - len(ELLIPSIS)].rstrip() + ELLIPSIS


#: Format of numbers read off a plot: six decimals is precise enough to paste into the
#: algebra window and short enough to fit twice on a line.
FORMAT = "{:.6f}"


def reading(value: float) -> str:
    """Format a number for the status line: six decimals, or "undefined".

    Shared so that both backends produce the same text.
    """
    if not math.isfinite(value):
        return "undefined"
    return FORMAT.format(value)


def roughly(value: float) -> str:
    """Format a number for a message about the view, without trailing zeros.

    Unlike readings, ranges don't need six decimals: `-5 ≤ x ≤ 5` is enough.
    """
    return f"{value:g}"


def written(value: float) -> str:
    """Format a number for a field or tick label, short and exact.

    Whole numbers are written as integers, so that `-5` typed into a field comes back
    as `-5` rather than `-5.00000`.
    """
    number = float(value)
    if number == int(number) and abs(number) < 1e15:
        return str(int(number))
    return f"{number:g}"


def pointed(x: float, y: float) -> str:
    """The point under a trace marker, written for the worksheet as `[x, y]`.

    The clipboard, events and browser panes all use this same text.
    """
    return f"[{reading(x)}, {reading(y)}]"


@dataclass
class Plot:
    """One entry of a 2D window's plot list.

    Identified by `worksheet` and `label` together, so re-plotting `#3` replaces its
    curve while `#3` from another algebra overlay is a separate plot. `color` is kept
    across replacement, so a re-plotted curve keeps its color.
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
    #: The error from the last sampling of this plot, if any.
    trouble: str = ""
    #: The parse state of the expression, used to parse bounds typed in the range fields.
    state: ParseState = field(default_factory=ParseState)
    #: The current parameter range, and bounds typed in the range fields that are yet
    #: to be evaluated (as expressions, since the user may type `-π`).
    trange: tuple[float, float] = (-DEFAULT_TURN, DEFAULT_TURN)
    bounds: tuple[Node, Node] | None = None
    #: Data plot style, set from the request and sticky preferences when the plot is
    #: added, and changed from its right-click menu.
    connected: bool = False
    point_size: float = POINT_SIZE

    @property
    def named(self) -> str:
        """The plot's name in the legend and the status line."""
        return naming(self.label, self.text)

    @property
    def variable(self) -> str:
        """The variable the plot is parametrized by, defaulting to `x`.

        This is the abscissa of a function, the parameter of a parametric pair, or the
        angle of a polar curve.
        """
        return self.options.variables[0] if self.options.variables else "x"

    @property
    def parameter(self) -> str:
        """The status line's name for the value the marker moves along.

        Polar curves always use θ, whatever the expression's variable, to avoid showing
        e.g. `x = 0.50   x = 0.11` for both the angle and the abscissa.
        """
        return "θ" if self.kind is PlotKind.POLAR else self.variable

    @property
    def abscissa(self) -> str:
        """The name this plot suggests for the horizontal axis, or "" for none.

        Parametric and data plots return "", letting a function in the same window name
        the axis.
        """
        if self.kind in FUNCTIONS:
            return self.variable
        if self.kind in FIELDS:
            return self.axes[0]
        return ""

    @property
    def axes(self) -> tuple[str, str]:
        """The two variables a field plot is evaluated over, horizontal first.

        An equation in one variable still needs two: `y = 3` is a horizontal line and
        `x = 3` a vertical one. The lone variable is paired with another from the
        worksheet's variable order, and the two are arranged in that order.
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
    """One entry of a 3D window's plot list.

    Adds wireframe display and visibility. (A 2D plot's visibility is kept by its
    drawing item instead.)
    """

    color: str = SOLID_PALETTE[0]
    paper: str = SOLID_PAPER[0]
    #: Whether to draw this surface as a wireframe rather than shaded. The toolbar's
    #: `mesh` box sets it for every surface, and the legend's right-click menu for one.
    wire: bool = False
    hidden: bool = False

    @property
    def visible(self) -> bool:
        return not self.hidden

    @property
    def axes(self) -> tuple[str, str]:
        """The two horizontal variables, in axis order."""
        names = self.options.variables
        if len(names) >= 2:
            return names[0], names[1]
        return (names[0] if names else "x"), "y"

    @property
    def vertical(self) -> str:
        """The vertical axis name: from the expression, as in `z = x^2 + y^2`, or `z`."""
        return self.options.vertical or "z"
