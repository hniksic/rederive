"""Plot dialogs described as fields, along with their validation messages.

`Set range...`, the 3D domain and grid, the parameter range of parametric and polar
plots, and the 3D inspector are all sets of named text fields that are read, validated
and applied. Only their rendering differs (a dialog on the desktop, an overlay in the
page), so the fields, labels, layout and error messages are defined here and each
backend renders them.

Bounds are parsed as expressions rather than floats, so users can type `-π`. This module
only parses the text into nodes; evaluating them is up to the caller, and the resulting
numbers are validated here.

Nothing here draws or evaluates. Field widths are hints about how much text is expected,
and a backend may lay out a form its own way.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from rederive.model.expr import Node
from rederive.plot.model import roughly
from rederive.plot.view import MAX_GRID
from rederive.syntax import DeriveSyntaxError, ParseState, parse_expression

__all__ = [
    "BACKWARDS",
    "DOMAIN",
    "DOMAIN_BACKWARDS",
    "DOMAIN_EXPRESSIONS",
    "EXPRESSIONS",
    "GRID_NUMBERS",
    "NOT_NUMBERS",
    "NUMBERS",
    "PARAMETER_EXPRESSIONS",
    "RANGE",
    "TRANGE",
    "VIEW",
    "Button",
    "Field",
    "Form",
    "Group",
    "Piece",
    "Role",
    "Row",
    "Strip",
    "alike",
    "counts",
    "domained",
    "evaluating",
    "framed",
    "numbers",
    "parsed",
    "showing",
    "spanned",
]


@dataclass(frozen=True)
class Field:
    """One input field of a form.

    `label` is shown in front of the field, and is empty when the row or column names
    it. `width` is a hint about how much text is expected (a bound is longer than a
    grid count).
    """

    name: str
    label: str = ""
    width: int = 90


@dataclass(frozen=True)
class Row:
    """One line of a group: its label and the fields on it."""

    label: str
    fields: tuple[str, ...]


@dataclass(frozen=True)
class Group:
    """Fields under one heading, arranged in rows.

    `heads` are column headings when the columns mean something (such as the three
    axes of a box), and empty when each field has its own label.
    """

    title: str
    rows: tuple[Row, ...]
    heads: tuple[str, ...] = ()


class Role(StrEnum):
    """What a form button does, which determines where a toolkit places it."""

    APPLY = "apply"
    RESET = "reset"
    CLOSE = "close"


@dataclass(frozen=True)
class Button:
    """A form button, and whether it is the primary one."""

    label: str
    role: Role
    primary: bool = False


@dataclass(frozen=True)
class Form:
    """A dialog, described by its fields independently of how it is drawn.

    `heading` is the window title and `title` is shown at the top of the form; both
    contain `{number}` for the plot window they refer to. `subtitle` is a short hint
    shown under the title.
    """

    name: str
    heading: str
    title: str
    subtitle: str
    fields: tuple[Field, ...]
    groups: tuple[Group, ...]
    buttons: tuple[Button, ...] = ()

    def field(self, name: str) -> Field:
        """The field with this name."""
        for one in self.fields:
            if one.name == name:
                return one
        raise KeyError(name)


@dataclass(frozen=True)
class Piece:
    """One element of a toolbar strip: a word, a field, or a divider.

    `entry` names one of the strip's fields and `word` is literal text. A divider
    separates groups of fields, such as the x domain, the y domain and the grid. A word
    containing `{parameter}` is rewritten when the strip switches to another plot, and
    `name` lets the window find it.
    """

    word: str = ""
    entry: str = ""
    divider: bool = False
    name: str = ""


@dataclass(frozen=True)
class Strip:
    """A row of fields on a toolbar, in display order."""

    name: str
    fields: tuple[Field, ...]
    pieces: tuple[Piece, ...]

    def field(self, name: str) -> Field:
        """The field with this name."""
        for one in self.fields:
            if one.name == name:
                return one
        raise KeyError(name)


# -- the four forms of a plot window -------------------------------------------


#: The four edges of a 2D view. The horizontal pair is laid out above the vertical one.
RANGE = Form(
    name="range",
    heading="Range of plot {number}",
    title="Range - plot {number}",
    subtitle="expressions, like -π or 2π",
    fields=(
        Field("left", "Left"),
        Field("right", "Right"),
        Field("bottom", "Bottom"),
        Field("top", "Top"),
    ),
    groups=(
        Group(
            title="Bounds",
            rows=(Row("", ("left", "right")), Row("", ("bottom", "top"))),
        ),
    ),
)

#: The 3D inspector: the box as center and length per axis (one column per axis), and
#: the camera position.
VIEW = Form(
    name="view",
    heading="View of plot {number}",
    title="View - plot {number}",
    subtitle="follows the camera while open",
    fields=tuple(
        Field(name)
        for name in (
            *("cx", "cy", "cz", "lx", "ly", "lz"),
            *("azimuth", "elevation", "distance"),
        )
    ),
    groups=(
        Group(
            title="Box",
            heads=("x", "y", "z"),
            rows=(
                Row("center", ("cx", "cy", "cz")),
                Row("length", ("lx", "ly", "lz")),
            ),
        ),
        Group(
            title="Camera",
            heads=("azimuth", "elevation", "distance"),
            rows=(Row("", ("azimuth", "elevation", "distance")),),
        ),
    ),
    buttons=(
        Button("Apply", Role.APPLY, primary=True),
        Button("Autoscale z", Role.RESET),
        Button("Close", Role.CLOSE),
    ),
)

#: The domain a 3D window evaluates over and its sampling grid. Unlike the inspector,
#: changing these requires re-evaluation.
DOMAIN = Strip(
    name="domain",
    fields=(
        Field("x0", width=52),
        Field("x1", width=52),
        Field("y0", width=52),
        Field("y1", width=52),
        Field("nx", width=52),
        Field("ny", width=52),
    ),
    pieces=(
        Piece(word="x:"),
        Piece(entry="x0"),
        Piece(word=" … "),
        Piece(entry="x1"),
        Piece(divider=True),
        Piece(word="y:"),
        Piece(entry="y0"),
        Piece(word=" … "),
        Piece(entry="y1"),
        Piece(divider=True),
        Piece(word="grid:"),
        Piece(entry="nx"),
        Piece(word=" x "),
        Piece(entry="ny"),
        Piece(divider=True),
    ),
)

#: The parameter range of the selected parametric or polar plot, labeled with the
#: parameter's name.
TRANGE = Strip(
    name="trange",
    fields=(Field("low", width=64), Field("high", width=64)),
    pieces=(
        Piece(divider=True),
        Piece(word="   {parameter}: ", name="parameter"),
        Piece(entry="low"),
        Piece(word=" … "),
        Piece(entry="high"),
    ),
)


# -- validation messages -------------------------------------------------------

EXPRESSIONS = "The bounds are expressions, like -π or 2π"
PARAMETER_EXPRESSIONS = "The {parameter} bounds are expressions, like -π or 2π"
NOT_NUMBERS = "The bounds have to be worth numbers, like -π or 2π"
BACKWARDS = "A range runs from a lower bound to a higher one"
SHOWING = "Showing {left} ≤ x ≤ {right}   {bottom} ≤ y ≤ {top}"
DOMAIN_EXPRESSIONS = "The domain bounds are expressions, like -π or 2π"
DOMAIN_BACKWARDS = "The domain runs from a lower bound to a higher one"
GRID_NUMBERS = "The grid is a pair of numbers"
EVALUATING = "Evaluating over the new domain at {nx} by {ny}"
NUMBERS = "The view is described in numbers"


def parsed(texts: Sequence[str], state: ParseState) -> tuple[Node, ...] | None:
    """Parse the texts as expressions, or return None if any fails to parse.

    The plot's parse state is used, since that is the syntax the user is typing in.
    """
    try:
        return tuple(parse_expression(text.strip(), state).node for text in texts)
    except DeriveSyntaxError:
        return None


def counts(texts: Sequence[str], most: int = MAX_GRID) -> tuple[int, int] | None:
    """Read a grid as a pair of counts clamped to [2, `most`], or None if invalid."""
    try:
        read = [int(float(text.strip())) for text in texts]
    except ValueError:
        return None
    return (min(max(read[0], 2), most), min(max(read[1], 2), most))


def numbers(texts: Sequence[str]) -> tuple[float, ...] | None:
    """Read the texts as plain numbers, or None if any isn't one."""
    try:
        return tuple(float(text) for text in texts)
    except ValueError:
        return None


def framed(values: Sequence[float]) -> str:
    """The error message for four invalid view bounds, or "" if they are valid."""
    left, right, bottom, top = values
    if not all(math.isfinite(value) for value in values):
        return NOT_NUMBERS
    if right <= left or top <= bottom:
        return BACKWARDS
    return ""


def domained(values: Sequence[float]) -> str:
    """The error message for an invalid domain, or "" if it is valid.

    On the desktop a bound that doesn't evaluate reverts to its previous value, so
    only an inverted domain can occur. The page sends such a bound as NaN, which is
    rejected here.
    """
    x0, x1, y0, y1 = values
    if not all(math.isfinite(value) for value in values):
        return NOT_NUMBERS
    return DOMAIN_BACKWARDS if x1 <= x0 or y1 <= y0 else ""


def showing(left: float, right: float, bottom: float, top: float) -> str:
    """The message describing a manually framed view."""
    return SHOWING.format(
        left=roughly(left),
        right=roughly(right),
        bottom=roughly(bottom),
        top=roughly(top),
    )


def evaluating(grid: tuple[int, int]) -> str:
    """The message shown when a 3D window re-evaluates over a typed domain."""
    return EVALUATING.format(nx=grid[0], ny=grid[1])


def spanned(center: float, length: float) -> tuple[float, float]:
    """The two edges of an inspector box axis given its center and length."""
    return (center - length / 2, center + length / 2)


def alike(one: Sequence[float], other: Sequence[float]) -> bool:
    """Whether two ranges are equal up to the rounding of a field's display.

    A number shown in a field as `16.6667` and read back counts as unchanged: the
    question is whether the user edited the field, not whether two floats are equal.
    """
    span = max(abs(other[1] - other[0]), 1e-12)
    return all(abs(a - b) <= 1e-4 * span for a, b in zip(one, other))
