"""Classification of an expression into a plot kind, before anything is sent.

This is done app-side because the app needs the answer to pick the window, and
the tree shape and free variables suffice, so no round trip is needed.

The variables come from `session.variables`, which resolves labels, assigned
variables and defined functions, so `#3` classifies by what it refers to. The rest
is read from the pure-Python `Node`, so this module is testable without an engine.

The rules follow section 5's table, first match wins. Two precedence choices
stand out: `[SIN(t), COS(t)]` is a parametric pair rather than a two-curve family
(two curves are just two Plot commands into the same window), and `[x+y, x-y]` is
a pair of surfaces, since parametric needs exactly one variable.

An expression matching no rule is refused with a message explaining why, never
just a beep.

`preferences` and `learned` translate the sticky plot preferences between the
settings store (words and integers) and the plot host (booleans and a pixel
size), one per direction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rederive.model.expr import Kind, Node
from rederive.model.settings import Settings
from rederive.plot.protocol import Options, PlotKind, Prefer

__all__ = ["Plotted", "Unplottable", "classify", "learned", "preferences"]

#: Refusal for an expression with too many variables, named so the user knows
#: what to reduce.
TOO_MANY = "{text} depends on {names} - reduce to one or two variables"

#: Refusal for an expression whose shape can't be plotted, such as a three-column
#: matrix, a string or an assignment.
NOT_A_PLOT = "{text} is not an expression this can plot"

#: Relations that plot as a region rather than a curve. Only `=` is an implicit
#: curve; even `/=` is a region (the complement of a curve).
INEQUALITIES = frozenset({"<", "<=", ">", ">=", "/=", "≤", "≥", "≠"})

#: Node kinds that combine relations into a region, as in `x > 0 AND y > 0`.
COMBINATIONS = frozenset({Kind.NOT, Kind.AND, Kind.OR, Kind.XOR, Kind.IMP})


class Unplottable(Exception):
    """The expression can't be plotted; the message says why."""


@dataclass(frozen=True)
class Plotted:
    """What to send: the plot kind and its options.

    `variables` is in axis order: the abscissa or parameter first, and for a
    surface the two horizontal axes in the engine's canonical (arrival) order.
    """

    kind: PlotKind
    variables: tuple[str, ...] = ()
    vertical: str = ""

    @property
    def options(self) -> Options:
        """The classification as protocol options."""
        return Options(variables=self.variables, vertical=self.vertical)


def classify(
    node: Node, variables: Sequence[str], text: str = "", relation: str | None = None
) -> Plotted:
    """Classify `node` as a plot, given the variables the engine found in it.

    `relation`, from `session.relation`, is the relational operator a non-relation
    tree evaluates to: `IM(u = v)` is a call but plots as the equation. A tree
    written as a relation is classified as written, regardless of evaluation.

    Raises `Unplottable` if nothing fits, with the message for the message line.
    """
    names = tuple(variables)
    if node.kind is not Kind.REL and relation is not None:
        node = _as_relation(node, relation)
    for reading in (_solved, _equation, _region, _vector, _scalar):
        plotted = reading(node, names)
        if plotted is not None:
            return plotted
    raise Unplottable(_refusal(names, text))


def _as_relation(node: Node, relation: str) -> Node:
    """A placeholder relation with operator `relation` standing in for `node`.

    The sides are just `node`; only `_solved` inspects sides, and it needs a bare
    name on one side, which a wrapped call never is.
    """
    return Node(Kind.REL, node.start, node.end, (node, node), relation)


def _refusal(names: tuple[str, ...], text: str) -> str:
    if len(names) > 2:
        return TOO_MANY.format(text=text, names=", ".join(names))
    return NOT_A_PLOT.format(text=text)


def _solved(node: Node, names: tuple[str, ...]) -> Plotted | None:
    """`z = f(x, y)`: an equation solved for the vertical axis.

    The lone variable is the vertical axis and the other two the floor. This is
    the only rule that accepts three variables.
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
    """An inequality, or a boolean combination of relations: a region."""
    if len(names) > 2:
        return None
    if node.kind in COMBINATIONS or (
        node.kind is Kind.REL and str(node.value) in INEQUALITIES
    ):
        return Plotted(PlotKind.REGION, names)
    return None


def _vector(node: Node, names: tuple[str, ...]) -> Plotted | None:
    """The five interpretations of a vector, in the order they are tried."""
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

    Statements and non-values (assignments, definitions, strings) are rejected.
    """
    if node.kind in (Kind.ASSIGN, Kind.FUNDEF, Kind.STRING, Kind.DOMAIN, Kind.VECTOR):
        return None
    if len(names) <= 1:
        return Plotted(PlotKind.CURVE, names)
    if len(names) == 2:
        return Plotted(PlotKind.SURFACE, names)
    return None


def preferences(settings: Settings) -> Prefer:
    """The sticky plot settings as a `Prefer` request for the host.

    Converts the store's words and integers (as written to the state file) to the
    host's booleans and pixel size.
    """
    return Prefer(
        grid=int(settings["PlotGrid"]),
        connected=settings["PlotPoints"] == "Connected",
        point_size=float(settings["PlotPointSize"]),
        wire=settings["PlotMesh"] == "Wire",
    )


def learned(preferences: Prefer) -> dict[str, str | int]:
    """Convert a host's reported `Prefer` back to settings values.

    The inverse of `preferences`, used when a plot window control changes.
    """
    return {
        "PlotGrid": int(preferences.grid),
        "PlotPoints": "Connected" if preferences.connected else "Discrete",
        "PlotPointSize": int(round(preferences.point_size)),
        "PlotMesh": "Wire" if preferences.wire else "Solid",
    }


def _names_in(node: Node) -> set[str]:
    """All names in `node`, including function heads.

    Including heads is harmless: callers only test whether a given variable is
    present.
    """
    found = {str(node.value)} if node.kind is Kind.NAME else set()
    for child in node.children:
        found |= _names_in(child)
    return found
