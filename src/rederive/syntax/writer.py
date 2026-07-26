"""Expression trees back to author notation - the parser run backwards.

What comes out is canonical, not what was typed: juxtaposition is written `*`,
a name is written the way the symbol table spells it, and a bare application
becomes a call. `x (x + 1)` writes as `x*(x+1)` and `SIN x` as `SIN(x)`, which
is what the original's Transfer Save does, and what makes a saved worksheet a
statement of what the expressions *are* rather than of how they were reached.
Every spelling and every parenthesis here was checked against the original.

The contract is that `parse_expression(write_expression(node))` builds `node`
again, so a file always reads back as the worksheet that wrote it. The
normalizations above are the exceptions in kind but not in meaning: an `APPLY`
comes back a `CALL`, and an `ABS` written `|u|` comes back the same `ABS`.
Writing is therefore idempotent, which is what the round trip over the corpus
checks.

Parentheses come from one rule - an operand looser than its position gets them
- with the position's demand set one step high in three places where the
original writes a pair the grammar would not require: around a signed operand
inside a run, `a*(-b)*c`; around a power inside a power, `a^(b^c)`; and around
a run of the same word operator on the right of one, `a AND (b AND c)`. The
grammar would read all three the same way without them; the original writes
them, and they are the pairs a reader would want.

Only numerals are written as they were typed rather than canonically. A numeral
carries both its value and its digits, and the digits are what the file must
hold: `0FF` under `InputBase := 16` is 255, and writing `255` would read back
as a different number under the base the file itself sets.
"""

from __future__ import annotations

from rederive.model.expr import Kind, Node

# The grammar's precedence ladder, loosest first. This is how tightly the
# parser binds, which is not the ladder the display uses: that one describes
# how tightly a *rendering* reads, where a sign already drawn stands to the
# left of a product rather than around it.
(
    ASSIGN,
    IMP,
    XOR,
    OR,
    AND,
    RELATION,
    SUM,
    NEG,
    PRODUCT,
    POW,
    SUB,
    POSTFIX,
    ATOM,
) = range(13)

_BINOP_PRECEDENCE = {"/": PRODUCT, ".": PRODUCT, "^": POW}

#: The operators spelled as words, which need a blank on each side so that
#: they do not fuse with the names beside them.
_WORD_OPERATORS = {Kind.AND: "AND", Kind.OR: "OR", Kind.XOR: "XOR", Kind.IMP: "IMP"}

#: Plus-or-minus, written the way it is typed - quotes and all - rather than
#: as the `±` glyph the screen shows it as, which is what the original writes
#: and what keeps a file readable where the glyph is not. A bare `+-` would be
#: a plus over a minus, which is a different expression.
_SIGNS = {"+-": '"+-"'}

#: What each logical run demands of its operands. `AND`, `OR` and `XOR` nest to
#: the right, so a nest of the same operator on the *left* is a different tree
#: and has to be fenced; the original fences the right one too.
_LOGICAL_PRECEDENCE = {Kind.AND: AND, Kind.OR: OR, Kind.XOR: XOR, Kind.IMP: IMP}


def write_expression(node: Node) -> str:
    """`node` in author notation, as a file holds it."""
    return _operand(node, ASSIGN)


def _operand(node: Node, required: int) -> str:
    """`node` where an operand binding at least `required` is expected."""
    text = _spell(node)
    return f"({text})" if precedence(node) < required else text


def precedence(node: Node) -> int:
    """How tightly `node`'s written form binds."""
    match node.kind:
        case Kind.ASSIGN | Kind.FUNDEF | Kind.SHOWVALUE:
            return ASSIGN
        case Kind.IMP | Kind.XOR | Kind.OR | Kind.AND:
            return _LOGICAL_PRECEDENCE[node.kind]
        case Kind.REL | Kind.DOMAIN:
            return RELATION
        case Kind.SUM:
            return SUM
        case Kind.UNOP:
            return NEG
        case Kind.PRODUCT:
            return PRODUCT
        case Kind.BINOP:
            return _BINOP_PRECEDENCE[str(node.value)]
        case Kind.SUB:
            return SUB
        case Kind.POSTOP:
            return POSTFIX
        case _:
            # Leaves, and everything that closes itself off: a call, a vector,
            # `ABS(u)`, and `NOT(u)`, which is written as a call as well.
            return ATOM


def _spell(node: Node) -> str:
    """`node` without the fences its position may call for."""
    match node.kind:
        case Kind.NUMBER:
            return str(node.surface or node.value)
        case Kind.NAME | Kind.UNKNOWN:
            return str(node.value)
        case Kind.STRING:
            return f'"{node.value}"'
        case Kind.LABEL:
            return f"#{node.value}"
        case Kind.SUM:
            return _sum(node)
        case Kind.PRODUCT:
            return "*".join(
                _factor(child, first=not index)
                for index, child in enumerate(node.children)
            )
        case Kind.BINOP:
            return _binop(node)
        case Kind.UNOP:
            sign = _SIGNS.get(str(node.value), str(node.value))
            return sign + _operand(node.children[0], NEG)
        case Kind.POSTOP:
            return _operand(node.children[0], POSTFIX) + str(node.value)
        case Kind.SUB:
            left = _operand(node.children[0], SUB)
            return f"{left} SUB {_operand(node.children[1], POSTFIX)}"
        case Kind.ABS:
            # `|u|` writes as the call it means, as the original writes it.
            return f"ABS({write_expression(node.children[0])})"
        case Kind.CALL | Kind.APPLY:
            return f"{node.children[0].value}({_arguments(node.children[1:])})"
        case Kind.FUNCPOW:
            name, exponent, operand = node.children
            return f"{name.value}^{_operand(exponent, SUB)}({write_expression(operand)})"
        case Kind.VECTOR:
            return f"[{_arguments(node.children)}]"
        case Kind.REL:
            left = _operand(node.children[0], RELATION)
            return f"{left}{node.value}{_operand(node.children[1], SUM)}"
        case Kind.NOT:
            return f"NOT({write_expression(node.children[0])})"
        case Kind.AND | Kind.OR | Kind.XOR | Kind.IMP:
            return _logical(node)
        case Kind.ASSIGN:
            return _assignment(node.children[0], node.children[1:], str(node.value))
        case Kind.FUNDEF:
            head = f"{node.value}({_arguments(node.children[0].children)})"
            return _assignment(head, node.children[1:], ":=")
        case Kind.DOMAIN:
            return _domain(node)
        case Kind.INTERVAL:
            opening, closing = str(node.value)
            low, high = (write_expression(child) for child in node.children)
            return f"{opening}{low}, {high}{closing}"
        case Kind.SHOWVALUE:
            return f"{write_expression(node.children[0])}="
    raise AssertionError(f"unwritable node kind: {node.kind}")


def _factor(node: Node, first: bool) -> str:
    """One factor of a product run, fenced if the run would swallow it.

    Two factors bind as loosely as the run itself and have to be looked at:

    * A product inside a product is not the same tree as one flat run - the
      original offers `(a*b)*c` as two operands and `a*b*c` as three - so a
      nested run always keeps its fences.
    * `/` and the dot product are binary and *close* the run to their left,
      taking it as their left operand. A factor that is one may therefore stand
      bare only at the head of the run, where there is nothing to close: the
      first factor of `a/b*c` needs no fences, and the last of `a*(b/c)` needs
      them or the `a` would be read into the numerator.
    """
    closes = node.kind is Kind.BINOP and node.value in ("/", ".")
    if node.kind is Kind.PRODUCT or (closes and not first):
        return f"({write_expression(node)})"
    return _operand(node, PRODUCT)


def _sum(node: Node) -> str:
    """A run of terms and the signs between them.

    The leading term keeps a sign of its own - `-a-b` is a sum of two - and
    every later one is fenced, so that the sign between the terms is the only
    one a reader has to account for. A sum nested inside a sum is fenced by
    either demand, for the reason a nested product is.
    """
    parts = [_operand(node.children[0], NEG)]
    for index, child in enumerate(node.children[1:]):
        parts += [str(node.value)[index], _operand(child, PRODUCT)]
    return "".join(parts)


def _binop(node: Node) -> str:
    """`/`, the dot product, and `^`.

    All three are binary. `/` and `.` fold to the left, so `a/b/c` needs
    nothing on the left and fences on the right. `^` folds to the right, where
    the original fences anyway.
    """
    operator = str(node.value)
    if operator == "^":
        left = _operand(node.children[0], SUB)
        return f"{left}^{_operand(node.children[1], SUB)}"
    # The dot product is spaced, being the one operator whose tight form would
    # fuse with its operands: `2 . 3` written `2.3` is the numeral two-point-
    # three, and the dot would be gone.
    if operator == ".":
        operator = " . "
    left = _operand(node.children[0], PRODUCT)
    return f"{left}{operator}{_operand(node.children[1], POW)}"


def _logical(node: Node) -> str:
    own = _LOGICAL_PRECEDENCE[node.kind]
    word = _WORD_OPERATORS[node.kind]
    # `IMP` is the one that folds to the left, so its left operand may nest.
    left = _operand(node.children[0], own if node.kind is Kind.IMP else own + 1)
    return f"{left} {word} {_operand(node.children[1], own + 1)}"


def _assignment(head: str | Node, right: tuple[Node, ...], operator: str) -> str:
    """`x:=u`, or `x:=` for an assignment with nothing on the right."""
    if isinstance(head, Node):
        head = write_expression(head)
    return head + operator + (write_expression(right[0]) if right else "")


def _domain(node: Node) -> str:
    """`x:epsilonReal (0, inf)`, spelled as tight as the original spells it."""
    text = f"{write_expression(node.children[0])}:epsilon{node.value}"
    if len(node.children) > 1:
        text += f" {write_expression(node.children[1])}"
    return text


def _arguments(children: tuple[Node, ...]) -> str:
    return ",".join(write_expression(child) for child in children)
