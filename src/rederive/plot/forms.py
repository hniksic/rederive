"""The dialogs as fields: what is asked for, and what is said when it will not read.

`Set range...`, the 3D domain and grid, the parameter range of a parametric or
polar plot and the 3D inspector are all one thing: named fields, filled with
text, read, argued with, applied. Only the drawing of them differs - a desktop
renders a dialog and a page renders an overlay - so the fields, their words,
their arrangement and every sentence a refusal is made of are here, and each
backend renders what it finds.

The bounds are expressions rather than floats everywhere they appear, which is
the point of parsing them in one place: `-π` is what a person types, and a
field that special-cased a constant or two would be a parser by stealth. What
the trees are worth is arithmetic and belongs to whoever computes, so this half
reads the text and hands on nodes; the numbers come back later and are argued
with here as well.

Nothing here draws, measures or evaluates. A width is in the units a field is
laid out in and is a hint about how much is going to be typed rather than a
size, and a form that a backend cannot lay out exactly is a form it may lay out
its own way.
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
    """One thing a form asks for: what it is called, and what it is called by.

    `label` is the word standing in front of the field where it has one, and is
    empty where the row or the column it stands in is what names it. `width` is
    how much text is expected in it - a bound is longer than a grid count - and
    is a hint rather than a size.
    """

    name: str
    label: str = ""
    width: int = 90


@dataclass(frozen=True)
class Row:
    """One line of a group: what the line is called, and the fields on it."""

    label: str
    fields: tuple[str, ...]


@dataclass(frozen=True)
class Group:
    """A run of fields under one heading, laid out as the thing they describe.

    `heads` are the words over the columns, where the columns mean something -
    the three axes of a box - and are empty where each field carries its own
    label instead.
    """

    title: str
    rows: tuple[Row, ...]
    heads: tuple[str, ...] = ()


class Role(StrEnum):
    """What a button of a form does, which is what a toolkit places it by."""

    APPLY = "apply"
    RESET = "reset"
    CLOSE = "close"


@dataclass(frozen=True)
class Button:
    """One answer a form offers, and whether it is the accented one."""

    label: str
    role: Role
    primary: bool = False


@dataclass(frozen=True)
class Form:
    """A dialog as the fields it holds rather than as the window it is drawn in.

    `heading` is what the window manager writes over it and `title` what the
    form writes at the top of itself; both name the window they are about, so
    both hold `{number}`. `subtitle` is the one thing worth knowing that three
    fields with numbers in them would not say.
    """

    name: str
    heading: str
    title: str
    subtitle: str
    fields: tuple[Field, ...]
    groups: tuple[Group, ...]
    buttons: tuple[Button, ...] = ()

    def field(self, name: str) -> Field:
        """The field this name is of."""
        for one in self.fields:
            if one.name == name:
                return one
        raise KeyError(name)


@dataclass(frozen=True)
class Piece:
    """One thing standing in a strip of fields along a toolbar.

    A strip is a row rather than a form: a word, a field, another word, another
    field. `entry` names one of the strip's fields and `word` is anything that
    is not one; a divider is where one answer ends and the next begins, since a
    domain in x, a domain in y and a grid are three answers and not six numbers
    in a row. A word holding `{parameter}` is rewritten as the strip changes
    which plot it is about, and `name` is how the window finds it again.
    """

    word: str = ""
    entry: str = ""
    divider: bool = False
    name: str = ""


@dataclass(frozen=True)
class Strip:
    """A run of fields along a toolbar, in the order they stand."""

    name: str
    fields: tuple[Field, ...]
    pieces: tuple[Piece, ...]

    def field(self, name: str) -> Field:
        """The field this name is of."""
        for one in self.fields:
            if one.name == name:
                return one
        raise KeyError(name)


# -- the four things a plot window asks to be told -----------------------------


#: The four edges of a 2D view, typed out. Deliberately a form and not a screen
#: of settings: a framing is four numbers, and everything else about the view
#: belongs to the mouse. The two are laid out as the picture is, the horizontal
#: pair over the vertical one.
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

#: The numbers behind a 3D picture: the box, as a person reads one - where it
#: is and how big it is, under the axis each column is about - and where it is
#: looked at from. Six fields in a list would be the same six numbers with the
#: shape taken out of them.
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

#: The rectangle a 3D window evaluates over and how finely it samples it, which
#: are the two things that change what was computed rather than where it is
#: looked at from.
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

#: The parameter range of the selected parametric or polar plot, which is where
#: a different piece of a curve is asked for. The name is the parameter's own,
#: so the strip says what it is a range of.
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


# -- what a form says when it will not read ------------------------------------

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
    """Read what was typed as expressions, or None where one of them is not.

    Whole texts rather than numbers, under the parse state the plot arrived
    with, because that is the grammar the person typing is writing in. What the
    trees are worth is nobody's business here.
    """
    try:
        return tuple(parse_expression(text.strip(), state).node for text in texts)
    except DeriveSyntaxError:
        return None


def counts(texts: Sequence[str], most: int = MAX_GRID) -> tuple[int, int] | None:
    """A grid as the pair of counts it is, clamped, or None where it is not one.

    A window never samples finer than it can draw, whoever asked, and never
    coarser than a pair of samples an axis.
    """
    try:
        read = [int(float(text.strip())) for text in texts]
    except ValueError:
        return None
    return (min(max(read[0], 2), most), min(max(read[1], 2), most))


def numbers(texts: Sequence[str]) -> tuple[float, ...] | None:
    """What was typed as plain numbers, or None where one of them is not one."""
    try:
        return tuple(float(text) for text in texts)
    except ValueError:
        return None


def framed(values: Sequence[float]) -> str:
    """Why four typed bounds are not a framing, in words, or nothing at all."""
    left, right, bottom, top = values
    if not all(math.isfinite(value) for value in values):
        return NOT_NUMBERS
    if right <= left or top <= bottom:
        return BACKWARDS
    return ""


def domained(values: Sequence[float]) -> str:
    """Why a typed domain is not a rectangle, in words, or nothing at all.

    A bound that would not evaluate came back as the value it was replacing, so
    a field of nonsense reverts by arithmetic rather than by a case; an
    inverted domain is the one refusal left to make.
    """
    x0, x1, y0, y1 = values
    return DOMAIN_BACKWARDS if x1 <= x0 or y1 <= y0 else ""


def showing(left: float, right: float, bottom: float, top: float) -> str:
    """What a view framed by hand is now showing, as the window says it."""
    return SHOWING.format(
        left=roughly(left),
        right=roughly(right),
        bottom=roughly(bottom),
        top=roughly(top),
    )


def evaluating(grid: tuple[int, int]) -> str:
    """What a 3D window says when a typed domain sends it back to the numbers."""
    return EVALUATING.format(nx=grid[0], ny=grid[1])


def spanned(center: float, length: float) -> tuple[float, float]:
    """The two edges an axis of the inspector's box describes."""
    return (center - length / 2, center + length / 2)
