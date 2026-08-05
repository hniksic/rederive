"""What an expression is a plot of, decided before anything is sent anywhere.

Classification is done app-side because the app is the side that has to act on
it: which kind of window the plot is going to land in is known from the shape
of the tree and the list of free variables, and is not worth a round trip to
find out.

The variables come from `session.variables`, which is a worker call that
already exists and is the authority: it resolves labels, assigned variables and
defined functions, so `#3` classifies as what `#3` means rather than as the
label it is written with. Everything else is read off the pure-Python `Node`,
which is why this module is testable without an engine.

The table is section 5's, first matching row wins, and two of its rows are
precedence decisions worth naming. `[SIN(t), COS(t)]` is a parametric pair and
not a two-curve family - plotting two univariate functions together is two
consecutive Plot commands, which land in the same window anyway - and a
2-vector in two variables, `[x+y, x-y]`, is a two-surface 3D family, because
the parametric reading needs exactly one variable and this has two.

Never silence: an expression that matches no row is refused with a message that
says what stopped it. A refusal that only beeps is the anti-goal.

`preferences` and `learned` are the other pure translations plotting needs,
and they are here for the same reason: the settings store keeps the sticky
plot preferences as words and whole numbers, the host speaks them as booleans
and a pixel count, and these two are where the spellings meet - one for each
direction the values travel.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rederive.model.expr import Kind, Node
from rederive.model.settings import Settings
from rederive.plot.protocol import Options, PlotKind, Prefer

__all__ = ["Plotted", "Unplottable", "classify", "learned", "preferences"]

#: What a refusal says when the expression holds more variables than any
#: reading of it can use. The variables are named, since which ones they are is
#: what the user has to act on.
TOO_MANY = "{text} depends on {names} - reduce to one or two variables"

#: And when the shape itself is not a plot of anything: a three-column matrix,
#: a string, an assignment. There is nothing to reduce, so the message does not
#: pretend there is.
NOT_A_PLOT = "{text} is not an expression this can plot"

#: The relations that bound a region rather than draw a curve. `=` is the
#: implicit curve and everything else is an inequality, `/=` included: the
#: complement of a curve is an area.
INEQUALITIES = frozenset({"<", "<=", ">", ">=", "/=", "≤", "≥", "≠"})

#: The node kinds that join relations into a region: `x > 0 AND y > 0`.
COMBINATIONS = frozenset({Kind.NOT, Kind.AND, Kind.OR, Kind.XOR, Kind.IMP})


class Unplottable(Exception):
    """The expression is not a plot of anything, and this is why in words."""


@dataclass(frozen=True)
class Plotted:
    """What to send: the kind, and the options the kind needs.

    `variables` is in the order the axes take them - the abscissa or the
    parameter first, and for a surface the two horizontal axes in the engine's
    canonical variable order, which is the order they arrive in.
    """

    kind: PlotKind
    variables: tuple[str, ...] = ()
    vertical: str = ""

    @property
    def options(self) -> Options:
        """The classification as the protocol carries it."""
        return Options(variables=self.variables, vertical=self.vertical)


def classify(node: Node, variables: Sequence[str], text: str = "") -> Plotted:
    """What `node` is a plot of, given the variables the engine found in it.

    Raises `Unplottable` where no reading fits, with the message the command
    puts on the message line.
    """
    names = tuple(variables)
    for reading in (_solved, _equation, _region, _vector, _scalar):
        plotted = reading(node, names)
        if plotted is not None:
            return plotted
    raise Unplottable(_refusal(names, text))


def _refusal(names: tuple[str, ...], text: str) -> str:
    if len(names) > 2:
        return TOO_MANY.format(text=text, names=", ".join(names))
    return NOT_A_PLOT.format(text=text)


def _solved(node: Node, names: tuple[str, ...]) -> Plotted | None:
    """`z = f(x, y)`: an equation already solved for its vertical axis.

    The lone variable names the vertical axis and leaves two for the floor,
    which is the one row that reads three variables and refuses nothing.
    """
    if node.kind is not Kind.REL or node.value != "=" or len(node.children) != 2:
        return None
    if len(names) != 3:
        return None
    left, right = node.children
    for side, other in ((left, right), (right, left)):
        if side.kind is not Kind.NAME:
            continue
        name = str(side.value)
        if name not in names or name in _names_in(other):
            continue
        floor = tuple(found for found in names if found != name)
        return Plotted(PlotKind.SURFACE, floor, name)
    return None


def _equation(node: Node, names: tuple[str, ...]) -> Plotted | None:
    """`u = v` in at most two variables: the zero contour of `u - v`."""
    if node.kind is not Kind.REL or node.value != "=" or len(names) > 2:
        return None
    return Plotted(PlotKind.IMPLICIT, names)


def _region(node: Node, names: tuple[str, ...]) -> Plotted | None:
    """An inequality, or a boolean combination of them: an area, not a line."""
    if len(names) > 2:
        return None
    if node.kind in COMBINATIONS or (
        node.kind is Kind.REL and str(node.value) in INEQUALITIES
    ):
        return Plotted(PlotKind.REGION, names)
    return None


def _vector(node: Node, names: tuple[str, ...]) -> Plotted | None:
    """The five readings of a bracketed list, in the order they are tried."""
    if node.kind is not Kind.VECTOR or not node.children:
        return None
    children = node.children
    scalars = [child for child in children if child.kind is not Kind.VECTOR]
    if len(children) == 2 and len(scalars) == 2:
        if not names:
            return Plotted(PlotKind.DATA, names)
        if len(names) == 1:
            return Plotted(PlotKind.PARAMETRIC, names)
    if not names and all(
        child.kind is Kind.VECTOR and len(child.children) == 2 for child in children
    ):
        return Plotted(PlotKind.DATA, names)
    if len(scalars) != len(children):
        return None
    if len(children) >= 3 and len(names) <= 1:
        return Plotted(PlotKind.FAMILY, names)
    if len(children) >= 2 and len(names) == 2:
        return Plotted(PlotKind.SURFACES, names)
    return None


def _scalar(node: Node, names: tuple[str, ...]) -> Plotted | None:
    """One expression: a curve in one variable, a surface in two.

    Nothing that is a statement rather than a value gets this far - an
    assignment, a definition, a string - since those are not expressions the
    question is about.
    """
    if node.kind in (Kind.ASSIGN, Kind.FUNDEF, Kind.STRING, Kind.DOMAIN, Kind.VECTOR):
        return None
    if len(names) <= 1:
        return Plotted(PlotKind.CURVE, names)
    if len(names) == 2:
        return Plotted(PlotKind.SURFACE, names)
    return None


def preferences(settings: Settings) -> Prefer:
    """The sticky plot values as the request that carries them to a host.

    The settings store keeps words and whole numbers, because that is what a
    state file writes; the host wants a boolean and a pixel count. Pure, so
    that the translation is testable without a plot window.
    """
    return Prefer(
        grid=int(settings["PlotGrid"]),
        connected=settings["PlotPoints"] == "Connected",
        point_size=float(settings["PlotPointSize"]),
    )


def learned(preferences: Prefer) -> dict[str, str | int]:
    """A host's report of its sticky values, as the settings store spells them.

    The inverse of `preferences`: a control moved in a plot window comes back
    as a `Prefer`, and what the app stores - and a state file then writes -
    are the settings' own words and whole numbers.
    """
    return {
        "PlotGrid": int(preferences.grid),
        "PlotPoints": "Connected" if preferences.connected else "Discrete",
        "PlotPointSize": int(round(preferences.point_size)),
    }


def _names_in(node: Node) -> set[str]:
    """Every name written anywhere in `node`, function heads included.

    Only ever asked whether one particular variable is among them, so a head
    counted as a name is harmless: a variable that is also a function name is
    not a variable of the expression either way.
    """
    found = {str(node.value)} if node.kind is Kind.NAME else set()
    for child in node.children:
        found |= _names_in(child)
    return found
