"""The operator ladder, and the functions that do not print as calls.

A special form is keyed by function name and argument count, so `INT(u, x)`
and `INT(u, x, a, b)` are two forms and `INT(u)` is an ordinary call. Each one
receives the engine and returns a box; none of them looks at the tree beyond
the arguments it was handed.

A form also names its own selectable operands, since it alone knows where it
drew them. Every one of them offers its arguments, in the order they were
written rather than in the order they were placed: the original steps an
integral's integrand, then its variable, then its bounds, though it draws the
bounds first.

The ladder lives here because deciding whether an operand needs fences means
knowing what a call renders as: `√u` trails off to the right the way a product
does, where `SIN(u)` delimits itself.
"""

from __future__ import annotations

from collections.abc import Callable

from rederive.display import glyphs
from rederive.display.boxes import Box, Placed, centered, fenced, rail, row, text
from rederive.model.expr import Kind, Node

# The precedence ladder, loosest first. This is how tightly a rendering binds
# as read back off the screen, which is not everywhere how the grammar builds
# it: `-x*y` parses as `-(x·y)`, and a sign already drawn stands to the left
# of a product the way one binding tighter would.
ASSIGNMENT = 0
IMP = 1
XOR = 2
OR = 3
AND = 4
NOT = 5
RELATION = 6
ADD = 7
MUL = 8
NEG = 9
POW = 10
SCRIPT = 11
POSTFIX = 12
ATOM = 13

_BINOPS: dict[str, int] = {
    "/": MUL,
    ".": MUL,
    "^": POW,
}

# What each special form binds like. `√u` takes an operand as tight as a
# power's, so it binds like one; everything else delimits itself well enough
# that no position alone calls for fences.
_FORM_PRECEDENCE: dict[str, int] = {"SQRT": POW}

# How far to the right a form's body reads, for the forms that end in one.
# `Σ k + 1` is a sum of `k + 1`, so a `+` drawn after it has to be fenced off,
# and `Σ k = 1` needs nothing because a relation is looser than the body. The
# body of `d/dx` stops at a product, which is why `d/dx x + 1` takes no fences
# where `d/dx x · 2` does. An integral is absent at any height: `⌠ … dx` is
# closed off by its own `dx`.
_BODY_PRECEDENCE: dict[str, int] = {
    "SUM": ADD,
    "PRODUCT": ADD,
    "LIM": ADD,
    "DIF": MUL,
}

_APPLICATIONS = (Kind.CALL, Kind.APPLY, Kind.FUNCPOW)


def application(node: Node) -> tuple[Node, tuple[Node, ...]]:
    """The function name and the arguments, however the call was written.

    `FUNCPOW` carries its exponent between the two, and that exponent is not
    an argument of the application: `SQRT^3 25` applies `SQRT` to `25`.
    """
    if node.kind is Kind.FUNCPOW:
        return node.children[0], (node.children[2],)
    return node.children[0], node.children[1:]


def special(node: Node) -> Callable[..., Box] | None:
    """The form `node` renders as, or None if it prints as an ordinary call."""
    if node.kind not in _APPLICATIONS:
        return None
    name, arguments = application(node)
    return FORMS.get((str(name.value), len(arguments)))


def body_precedence(node: Node) -> int | None:
    """How far right `node`'s body reads, or None if it closes itself off.

    Not a property of how tightly the form binds: `Σ k` needs fences before a
    `+ 1` and needs none after a `2·`, which no precedence of its own can
    express, since in both cases it sits beside the same operator.
    """
    if special(node) is None:
        return None
    name, _ = application(node)
    return _BODY_PRECEDENCE.get(str(name.value))


def precedence(node: Node) -> int:
    """How tightly `node`'s own rendering binds."""
    match node.kind:
        case Kind.SUM:
            return ADD
        case Kind.PRODUCT:
            return MUL
        case Kind.BINOP:
            return _BINOPS[str(node.value)]
        case Kind.UNOP:
            return NEG
        case Kind.POSTOP:
            return POSTFIX
        case Kind.SUB:
            return SCRIPT
        case Kind.REL | Kind.DOMAIN:
            return RELATION
        case Kind.NOT:
            return NOT
        case Kind.AND:
            return AND
        case Kind.OR:
            return OR
        case Kind.XOR:
            return XOR
        case Kind.IMP:
            return IMP
        case Kind.ASSIGN | Kind.FUNDEF | Kind.SHOWVALUE:
            return ASSIGNMENT
        case Kind.FUNCPOW:
            return POW
        case Kind.CALL | Kind.APPLY:
            name, _ = application(node)
            if special(node) is None:
                return ATOM
            return _FORM_PRECEDENCE.get(str(name.value), ATOM)
        case _:
            return ATOM


# -- the forms --------------------------------------------------------------


def square_root(engine, node, arguments, level, budget) -> Box:
    """`√u`: a prefix and ordinary fences, with no overbar."""
    operand = engine.operand(arguments[0], level, budget, POW)
    return row([text(glyphs.SQRT), operand], node, (operand,))


def absolute_value(engine, node, arguments, level, budget) -> Box:
    """`│u│`: rails on every row, whatever the height."""
    inner = engine.box(arguments[0], level, budget)
    left = rail(glyphs.RAILS.left, inner.above, inner.below, None)
    right = rail(glyphs.RAILS.right, inner.above, inner.below, None)
    return row([left, inner, right], node, (inner,))


def integral(engine, node, arguments, level, budget) -> Box:
    """`⌠ u dx`, with the bounds stacked against the sign when there are any.

    The sign is as tall as the integrand but never shorter than two rows,
    since a lone `⌠` reads as a broken glyph rather than as an integral.
    """
    # The ` dx` is a terminator rather than something the integrand runs into,
    # so nothing follows it for fencing purposes.
    integrand = engine.operand(arguments[0], level, budget, MUL)
    variable = engine.box(arguments[1], level, budget)
    below = integrand.below
    above = max(integrand.above, 1 - below)
    head = rail(glyphs.INTEGRAL, above, below, None)
    operands = (integrand, variable)
    if len(arguments) == 4:
        lower = engine.box(arguments[2], level, budget)
        upper = engine.box(arguments[3], level, budget)
        operands += (lower, upper)
        head = Box(
            1 + max(lower.width, upper.width),
            above + upper.height,
            below + lower.height,
            None,
            "",
            (
                Placed(head, 0, 0),
                # Stacked the way scripts are, but clear of the sign rather
                # than of the baseline: the sign is what they belong to.
                Placed(upper, 1, -above - 1 - upper.below),
                Placed(lower, 1, below + 1 + lower.above),
            ),
        )
    return row([head, text(" "), integrand, text(" d"), variable], node, operands)


def summation(engine, node, arguments, level, budget) -> Box:
    return _big_operator(engine, node, arguments, level, budget, glyphs.SUM)


def multiplication(engine, node, arguments, level, budget) -> Box:
    return _big_operator(engine, node, arguments, level, budget, glyphs.PRODUCT)


def _big_operator(engine, node, arguments, level, budget, glyph: str) -> Box:
    """`Σ` or `Π` on the baseline, `k=a` below it and `b` above it."""
    body = engine.operand(arguments[0], level, budget, MUL)
    index = engine.box(arguments[1], level, budget)
    start = engine.box(arguments[2], level, budget)
    # The index and its first value print tight, as one label.
    lower = row([index, text("="), start])
    upper = engine.box(arguments[3], level, budget)
    sign = text(glyph)
    width = max(sign.width, lower.width, upper.width)
    head = Box(
        width,
        upper.height,
        lower.height,
        None,
        "",
        (
            Placed(sign, centered(width, sign.width), 0),
            Placed(upper, centered(width, upper.width), -1 - upper.below),
            Placed(lower, centered(width, lower.width), 1 + lower.above),
        ),
    )
    return row([head, text(" "), body], node, (body, index, start, upper))


def limit(engine, node, arguments, level, budget) -> Box:
    """`lim` on the baseline with `x→a` centred below it."""
    body = engine.operand(arguments[0], level, budget, MUL)
    variable = engine.box(arguments[1], level, budget)
    target = engine.box(arguments[2], level, budget)
    approach = row([variable, text(glyphs.ARROW), target])
    sign = text(glyphs.LIMIT)
    width = max(sign.width, approach.width)
    head = Box(
        width,
        0,
        approach.height,
        None,
        "",
        (
            Placed(sign, centered(width, sign.width), 0),
            Placed(approach, centered(width, approach.width), 1 + approach.above),
        ),
    )
    return row([head, text(" "), body], node, (body, variable, target))


def derivative(engine, node, arguments, level, budget) -> Box:
    """`d` over `dx`, the order raised, then the operand.

    The bar is the one that gets no padding, and the two `d`s are flush left
    rather than centred so that they line up with each other. The fences are
    there to carry the order, so the two-argument form goes without them.
    """
    operand = engine.operand(arguments[0], level, budget, MUL)
    variable = engine.box(arguments[1], level, budget)
    numerator = text(glyphs.DERIVATIVE)
    denominator = row([text(glyphs.DERIVATIVE), variable])
    width = max(numerator.width, denominator.width)
    quotient = Box(
        width,
        numerator.height,
        denominator.height,
        None,
        "",
        (
            Placed(text(engine.bar(width)), 0, 0),
            Placed(numerator, 0, -1 - numerator.below),
            Placed(denominator, 0, 1 + denominator.above),
        ),
    )
    head = quotient
    operands = (operand, variable)
    if len(arguments) == 3:
        order = engine.box(arguments[2], level + 1, budget)
        operands += (order,)
        head = engine.raised(fenced(quotient, glyphs.PARENS), order)
    return row([head, text(" "), operand], node, operands)


FORMS: dict[tuple[str, int], Callable[..., Box]] = {
    ("SQRT", 1): square_root,
    ("ABS", 1): absolute_value,
    ("INT", 2): integral,
    ("INT", 4): integral,
    ("SUM", 4): summation,
    ("PRODUCT", 4): multiplication,
    ("LIM", 3): limit,
    ("DIF", 2): derivative,
    ("DIF", 3): derivative,
}
