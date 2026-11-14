"""The operator ladder, and the functions that do not print as calls.

Special forms are keyed by function name and argument count, so `INT(u, x)` and
`INT(u, x, a, b)` are two forms and `INT(u)` is an ordinary call. Each form receives the
engine and returns a box, looking only at its own arguments.

Forms list their own selectable operands, since only they know where they drew them.
Operands are offered in the order written, not the order drawn: Derive steps through an
integral's integrand, variable, then bounds, even though the bounds are drawn first.

The precedence ladder lives here because deciding whether an operand needs fences
requires knowing how a call renders: `√u` extends to the right like a product, while
`SIN(u)` delimits itself.
"""

from __future__ import annotations

from collections.abc import Callable

from rederive.display import glyphs
from rederive.display.boxes import Box, Placed, centered, fenced, rail, row, text
from rederive.model.expr import Kind, Node

# The precedence ladder, loosest first. It describes how tightly a rendering binds as
# read on screen, which doesn't always match the grammar: `-x*y` parses as `-(x·y)`, but
# a drawn sign reads as binding tighter than the product after it. A lowered index
# delimits itself, so a subscript binds tighter than anything beside it: `v₁!` is the
# factorial of the element without fences.
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
POSTFIX = 11
SCRIPT = 12
ATOM = 13

_BINOPS: dict[str, int] = {
    "/": MUL,
    ".": MUL,
    "^": POW,
}

# How tightly each special form binds. `√u` takes an operand as tight as a power's, so
# it binds like one; the other forms delimit themselves and never need fences.
_FORM_PRECEDENCE: dict[str, int] = {"SQRT": POW}

# How far right the body extends, for forms that end in a body. `Σ k + 1` reads as a
# sum of `k + 1`, so a following `+` must be fenced off, while `Σ k = 1` needs nothing
# because a relation is looser than the body. The body of `d/dx` stops at a product, so
# `d/dx x + 1` needs no fences but `d/dx x · 2` does. Integrals aren't listed because
# `⌠ … dx` is closed off by its `dx`.
_BODY_PRECEDENCE: dict[str, int] = {
    "SUM": ADD,
    "PRODUCT": ADD,
    "LIM": ADD,
    "DIF": MUL,
}

APPLICATIONS = (Kind.CALL, Kind.APPLY, Kind.FUNCPOW)


def application(node: Node) -> tuple[Node, tuple[Node, ...]]:
    """The function name and arguments, however the call was written.

    `FUNCPOW` has its exponent in between, and the exponent isn't an argument:
    `SQRT^3 25` applies `SQRT` to `25`.
    """
    if node.kind is Kind.FUNCPOW:
        return node.children[0], (node.children[2],)
    return node.children[0], node.children[1:]


def side_of(node: Node) -> str | None:
    """The sign shown for a limit's direction argument, or None if it can't be shown.

    Derive uses `0` for a two-sided limit, a negative number for the left and a positive
    one for the right. The screen shows `-` or `+` after the point, or nothing for a
    two-sided limit. Any other argument returns None.
    """
    negated = node.kind is Kind.UNOP and str(node.value) == "-"
    number = node.children[0] if negated else node
    if number.kind is not Kind.NUMBER:
        return None
    try:
        value = float(str(number.surface or number.value))
    except ValueError:
        return None
    if not value:
        return "" if not negated else None
    return "-" if negated else "+"


def special(node: Node) -> Callable[..., Box] | None:
    """The form `node` renders as, or None if it prints as an ordinary call."""
    if node.kind not in APPLICATIONS:
        return None
    name, arguments = application(node)
    key = (str(name.value), len(arguments))
    form = FORMS.get(key)
    guard = _GUARDS.get(key)
    if form is None or (guard is not None and not guard(arguments)):
        return None
    return form


def is_quotient(node: Node) -> bool:
    return node.kind is Kind.BINOP and node.value == "/"


def body(engine, node: Node, level: int, budget: int | None, required: int) -> Box:
    """A form's body, fenced if `required` calls for it.

    A built-up quotient is never fenced because the bar already delimits it: Derive
    writes `d/dx a/b` bare but fences `d/dx (a·b)`. A quotient drawn linearly (inside a
    script, or when the pane is too short) is treated like any other operand.
    """
    if is_quotient(node) and engine.built_up(node, level, budget):
        return engine.box(node, level, budget)
    return engine.operand(node, level, budget, required)


def body_precedence(node: Node) -> int | None:
    """How far right `node`'s body reads, or None if it closes itself off.

    This is separate from how tightly the form binds: `Σ k` needs fences before `+ 1`
    but not after `2·`, which a single precedence can't express.
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

    The sign is as tall as the integrand but at least two rows, since a lone `⌠` looks
    like a broken glyph. The bounds are drawn like scripts, so a fraction in them stays
    on one row.
    """
    # The ` dx` terminates the integrand, so for fencing purposes nothing follows it.
    integrand = body(engine, arguments[0], level, budget, MUL)
    variable = engine.box(arguments[1], level, budget)
    below = integrand.below
    above = max(integrand.above, 1 - below)
    head = rail(glyphs.INTEGRAL, above, below, None)
    operands = (integrand, variable)
    if len(arguments) == 4:
        lower = engine.box(arguments[2], level + 1, budget)
        upper = engine.box(arguments[3], level + 1, budget)
        operands += (lower, upper)
        head = Box(
            1 + max(lower.width, upper.width),
            above + upper.height,
            below + lower.height,
            None,
            "",
            (
                Placed(head, 0, 0),
                # Stacked like scripts, but positioned relative to the sign rather than
                # the baseline.
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
    """`Σ` or `Π` on the baseline, `k=a` below it and `b` above it.

    The two-argument form (the antidifference or antiquotient, which `Calculus Sum`
    builds when both limits are blank) has no bounds, so only the index goes under the
    sign and nothing goes above. The index and bounds are drawn like scripts, so a
    fraction in them stays on one row.
    """
    inner = body(engine, arguments[0], level, budget, MUL)
    index = engine.box(arguments[1], level + 1, budget)
    if len(arguments) == 2:
        lower, upper = index, None
        operands = (inner, index)
    else:
        start = engine.box(arguments[2], level + 1, budget)
        # The index and its start value print without spaces, as one label.
        lower = row([index, text("="), start])
        upper = engine.box(arguments[3], level + 1, budget)
        operands = (inner, index, start, upper)
    sign = text(glyph)
    width = max(sign.width, lower.width, 0 if upper is None else upper.width)
    placed = [
        Placed(sign, centered(width, sign.width), 0),
        Placed(lower, centered(width, lower.width), 1 + lower.above),
    ]
    if upper is not None:
        placed.append(Placed(upper, centered(width, upper.width), -1 - upper.below))
    head = Box(
        width,
        0 if upper is None else upper.height,
        lower.height,
        None,
        "",
        tuple(placed),
    )
    return row([head, text(" "), inner], node, operands)


def limit(engine, node, arguments, level, budget) -> Box:
    """`lim` on the baseline with `x→a` centered below it.

    `x→a` is drawn like a script, as the original draws it: a fraction in the point
    stays on one row rather than pushing the arrow down under its numerator.

    An optional fourth argument gives the direction, drawn as a sign rather than a
    number: `x→2-` from the left, `x→2+` from the right, and nothing for the two-sided
    limit (written `0`, called Both in the menu). The direction isn't a selectable
    operand, since there's nothing on screen to select.
    """
    inner = body(engine, arguments[0], level, budget, MUL)
    variable = engine.box(arguments[1], level + 1, budget)
    target = engine.box(arguments[2], level + 1, budget)
    side = "" if len(arguments) == 3 else side_of(arguments[3])
    approach = row([variable, text(glyphs.ARROW), target, text(side)])
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
    return row([head, text(" "), inner], node, (inner, variable, target))


def derivative(engine, node, arguments, level, budget) -> Box:
    """`d` over `dx`, the order raised, then the operand.

    This bar gets no padding, and the two `d`s are flush left rather than centered so
    they line up. Fences are only needed to carry the order, so the two-argument form
    has none.

    Its body extends the least far right of the body-ending forms: products and signs
    are fenced (`d/dx (a·b)`, `d/dx (-a)`) where `Σ` and `lim` leave them bare (`Σ a·b`,
    `lim -a`). Only operands binding at least as tightly as a power go unfenced.
    """
    operand = body(engine, arguments[0], level, budget, POW)
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
    ("SUM", 2): summation,
    ("SUM", 4): summation,
    ("PRODUCT", 2): multiplication,
    ("PRODUCT", 4): multiplication,
    ("LIM", 3): limit,
    ("LIM", 4): limit,
    ("DIF", 2): derivative,
    ("DIF", 3): derivative,
}

#: Conditions on the arguments for a form to apply. A four-argument `LIM` draws as a
#: limit only if its direction can be shown as a sign (see `side_of`); otherwise it
#: prints as an ordinary call.
_GUARDS: dict[tuple[str, int], Callable[[tuple[Node, ...]], bool]] = {
    ("LIM", 4): lambda arguments: side_of(arguments[3]) is not None,
}
