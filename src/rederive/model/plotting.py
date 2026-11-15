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
from dataclasses import dataclass, replace

from rederive.model.expr import Kind, Node
from rederive.model.settings import Settings
from rederive.plot.protocol import Member, Options, PlotKind, Prefer

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

#: The kinds a member of a vector can have in a 2D window.
FLAT = frozenset({PlotKind.CURVE, PlotKind.IMPLICIT, PlotKind.REGION})


class Unplottable(Exception):
    """The expression can't be plotted; the message says why."""


@dataclass(frozen=True)
class Plotted:
    """What to send: the plot kind and its options.

    `variables` is in axis order: the abscissa or parameter first, and for a
    surface the two horizontal axes in the engine's canonical (arrival) order.

    `node` is what to evaluate when it isn't the tree classified: the right side of
    `y = u(x)` or `z = f(x, y)`, or a vector with such members replaced by their right
    sides. `members` classifies each member of a family or a vector of surfaces.
    """

    kind: PlotKind
    variables: tuple[str, ...] = ()
    vertical: str = ""
    node: Node | None = None
    members: tuple[Plotted, ...] = ()

    @property
    def options(self) -> Options:
        """The classification as protocol options."""
        return Options(
            variables=self.variables,
            vertical=self.vertical,
            members=tuple(
                Member(member.kind, member.variables, member.vertical)
                for member in self.members
            ),
        )


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
    """`z = f(x, y)` or `y = u(x)`: an equation solved for the vertical axis.

    The lone variable is the vertical axis and the others the abscissa or the floor,
    and what is drawn is the other side, so `y = 1/x` is the curve `1/x`, broken at its
    pole, rather than the zero contour of `y - 1/x`, which crosses zero there too. With
    two variables, the lone one must be the second in axis order: `x = 1/y` gives x
    as a function of y, which is drawn as an implicit curve. This is the only rule
    that accepts three variables.
    """
    if node.kind is not Kind.REL or node.value != "=" or len(node.children) != 2:
        return None
    if len(names) not in (2, 3):
        return None
    left, right = node.children
    for side, other in ((left, right), (right, left)):
        if side.kind is not Kind.NAME:
            continue
        name = str(side.value)
        if name not in names or name in _names_in(other):
            continue
        floor = tuple(found for found in names if found != name)
        if len(names) == 3:
            return Plotted(PlotKind.SURFACE, floor, name, other)
        if name == names[1]:
            return Plotted(PlotKind.CURVE, floor, name, other)
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
    """The interpretations of a vector, in the order they are tried.

    A pair of plain expressions is a point or a parametric curve, and constant points,
    alone or in vectors nested to any depth, are a data plot. Otherwise each member is
    plotted as what it is (see `_members`).
    """
    if node.kind is not Kind.VECTOR or not node.children:
        return None
    children = node.children
    plain = [child for child in children if _plain(child)]
    if len(children) == 2 and len(plain) == 2:
        if not names:
            return Plotted(PlotKind.DATA, names)
        if len(names) == 1:
            return Plotted(PlotKind.PARAMETRIC, names)
    if not names and _points(node):
        return Plotted(PlotKind.DATA, names)
    if len(children) < 2 or any(child.kind is Kind.VECTOR for child in children):
        return None
    return _members(node, names)


def _members(node: Node, names: tuple[str, ...]) -> Plotted | None:
    """A vector of things to plot together, each classified as it would be alone.

    So `[x = 4, x - 6, x^2]` is a vertical line and two curves. Its members can share
    a 2D window if they are curves, implicit curves and regions, and the curves share
    one abscissa. Otherwise they are surfaces over one floor: `z = f(x, y)` or an
    expression in those two variables; a member in fewer variables is a surface that is
    constant along the others, as `x` is in `[x + y, x]`.
    """
    members = [_member(child, names) for child in node.children]
    if any(member is None for member in members):
        return None
    found = [member for member in members if member is not None]
    abscissas = {
        member.variables[0]
        for member in found
        if member.kind is PlotKind.CURVE and member.variables
    }
    if all(member.kind in FLAT for member in found) and len(abscissas) <= 1:
        variables = tuple(abscissas) or names
        found = [
            replace(member, variables=tuple(abscissas))
            if member.kind is PlotKind.CURVE
            else member
            for member in found
        ]
        return _family(node, PlotKind.FAMILY, variables, "", found)
    floors = {member.variables for member in found if member.kind is PlotKind.SURFACE}
    if not floors and len(names) == 2:
        floors = {names}
    if len(floors) != 1:
        return None
    floor = floors.pop()

    def standing(member: Plotted) -> bool:
        """Whether the member is a surface over the floor, or can be read as one."""
        if member.kind is PlotKind.SURFACE:
            return True
        # A plain expression, not `y = u(x)`, which is a curve in the plane.
        plain = member.kind is PlotKind.CURVE and member.node is None
        return plain and set(member.variables) <= set(floor)

    if not all(standing(member) for member in found):
        return None
    vertical = next((member.vertical for member in found if member.vertical), "")
    found = [replace(member, kind=PlotKind.SURFACE, variables=floor) for member in found]
    return _family(node, PlotKind.SURFACES, floor, vertical, found)


def _member(node: Node, names: tuple[str, ...]) -> Plotted | None:
    """One member of a vector, classified as it would be alone.

    Solved equations and fields are read in the vector's variables, so they share its
    axes. An expression is read in the variables it holds, so `x^2` beside `y = 1/x` is
    a curve rather than a surface.
    """
    for reading in (_solved, _equation, _region):
        plotted = reading(node, names)
        if plotted is not None:
            return plotted
    held = _names_in(node)
    return _scalar(node, tuple(name for name in names if name in held))


def _family(
    node: Node,
    kind: PlotKind,
    variables: tuple[str, ...],
    vertical: str,
    members: list[Plotted],
) -> Plotted:
    """A vector plotted member by member, with each member's drawn side in place."""
    drawn = None
    if any(member.node is not None for member in members):
        drawn = replace(
            node,
            children=tuple(
                child if member.node is None else member.node
                for child, member in zip(node.children, members, strict=True)
            ),
        )
    stripped = tuple(replace(member, node=None) for member in members)
    return Plotted(kind, variables, vertical, drawn, stripped)


def _plain(node: Node) -> bool:
    """Whether `node` is an expression rather than a vector, relation or statement."""
    return node.kind not in (Kind.VECTOR, Kind.REL, *COMBINATIONS)


def _points(node: Node) -> bool:
    """Whether `node` is a point `[a, b]`, or a vector of such, nested to any depth.

    Each innermost vector of points is drawn as its own line, which is how
    `ISOMETRICS` draws a surface as lines of points.
    """
    if node.kind is not Kind.VECTOR or not node.children:
        return False
    children = node.children
    if len(children) == 2 and all(child.kind is not Kind.VECTOR for child in children):
        return True
    return all(_points(child) for child in children)


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
