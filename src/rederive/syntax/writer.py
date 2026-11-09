"""Writing expression trees back in author notation, the inverse of the parser.

The output is canonical, not what was typed: juxtaposition becomes `*`, names use the
symbol table's spelling, and a bare application becomes a call. `x (x + 1)` is written
`x*(x+1)` and `SIN x` is `SIN(x)`, matching the original's Transfer Save. All spellings
and parentheses were checked against the original.

`parse_expression(write_expression(node))` rebuilds `node`, so a saved file reads back
as the worksheet that wrote it, except for normalizations that preserve meaning: an
`APPLY` comes back as a `CALL`, and `|u|` as the same `ABS`. Writing is therefore
idempotent, which the corpus round-trip test checks.

An operand gets parentheses when it binds more loosely than its position requires. In
three places the original adds parentheses the grammar doesn't need, so we do too: a
signed operand inside a run, `a*(-b)*c`; a power inside a power, `a^(b^c)`; and a
nested word operator on the right, `a AND (b AND c)`.

Numerals alone are written as typed, because their digits depend on the input base:
`0FF` under `InputBase := 16` is 255, and writing `255` would read back differently
under the base the file sets.

With `spaced`, the tree is written as the engine's printer writes a result line: blanks
around sum signs and relations and after commas, `NOT` as a prefix operator rather than
a call, Greek letters by name, and only the parentheses the grammar needs around word
operators. For example, `IF(h<=40,10*h,400+15*(h-40))` is written
`IF(h <= 40, 10*h, 400 + 15*(h - 40))`, and `a OR (b OR c)` is written `a OR b OR c`.
Both forms parse back to the same tree.

The two forms must agree wherever both can write a subtree, because the engine writes
undecidable conditionals this way, and a spelling the printer wouldn't choose would make
such an answer change on a second Simplify.
"""

from __future__ import annotations

from rederive.model.expr import Kind, Node
from rederive.syntax.names import GREEK_GLYPHS

# The parser's precedence levels, loosest first. These differ from the display's levels,
# which describe how a rendering reads.
#
# `NOT` has its own level only in the spaced form, where it's a prefix operator: `NOT x =
# y` negates a relation, so it binds looser than relations and tighter than the logical
# operators. The tight form writes it as the call `NOT(u)`, which needs no level.
(
    ASSIGN,
    IMP,
    XOR,
    OR,
    AND,
    NOT,
    RELATION,
    SUM,
    NEG,
    PRODUCT,
    POW,
    POSTFIX,
    SUB,
    ATOM,
) = range(14)

_BINOP_PRECEDENCE = {"/": PRODUCT, ".": PRODUCT, "^": POW}

#: Operators spelled as words, which need a blank on each side to stay separate from
#: adjacent names.
_WORD_OPERATORS = {Kind.AND: "AND", Kind.OR: "OR", Kind.XOR: "XOR", Kind.IMP: "IMP"}

#: Plus-or-minus is written as typed, with quotes, rather than as the displayed `±`
#: glyph, as the original does. A bare `+-` would be a plus followed by a minus.
_SIGNS = {"+-": '"+-"'}

#: Precedence of each logical operator. `AND`, `OR` and `XOR` are right-associative, so
#: the same operator nested on the left needs parentheses; the original also adds them
#: on the right.
_LOGICAL_PRECEDENCE = {Kind.AND: AND, Kind.OR: OR, Kind.XOR: XOR, Kind.IMP: IMP}


def write_expression(node: Node, spaced: bool = False) -> str:
    """`node` in author notation, as stored in a file.

    With `spaced`, write it as the printer does instead; see the module docstring.
    """
    return _operand(node, ASSIGN, " " if spaced else "")


def _operand(node: Node, required: int, gap: str) -> str:
    """`node` as an operand that must bind at least as tightly as `required`."""
    text = _spell(node, gap)
    return f"({text})" if precedence(node, gap) < required else text


def precedence(node: Node, gap: str = "") -> int:
    """How tightly `node`'s written form binds."""
    match node.kind:
        case Kind.ASSIGN | Kind.FUNDEF | Kind.SHOWVALUE:
            return ASSIGN
        case Kind.IMP | Kind.XOR | Kind.OR | Kind.AND:
            return _LOGICAL_PRECEDENCE[node.kind]
        case Kind.NOT if gap:
            return NOT
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
            # Leaves and self-delimiting forms: calls, vectors, `ABS(u)`, and the tight
            # form's `NOT(u)`.
            return ATOM


def _spell(node: Node, gap: str) -> str:
    """`node` without any parentheses its position may require."""
    match node.kind:
        case Kind.NUMBER:
            return str(node.surface or node.value)
        case Kind.NAME | Kind.UNKNOWN:
            # Greek letters are stored as glyphs. Files get the glyph, result lines the
            # name.
            name = str(node.value)
            return GREEK_GLYPHS.get(name, name) if gap else name
        case Kind.STRING:
            return f'"{node.value}"'
        case Kind.LABEL:
            return f"#{node.value}"
        case Kind.SUM:
            return _sum(node, gap)
        case Kind.PRODUCT:
            return "*".join(
                _factor(child, not index, gap)
                for index, child in enumerate(node.children)
            )
        case Kind.BINOP:
            return _binop(node, gap)
        case Kind.UNOP:
            sign = _SIGNS.get(str(node.value), str(node.value))
            return sign + _operand(node.children[0], NEG, gap)
        case Kind.POSTOP:
            return _operand(node.children[0], POSTFIX, gap) + str(node.value)
        case Kind.SUB:
            left = _operand(node.children[0], POSTFIX, gap)
            # An index is one application, so anything larger, notably a postfix
            # operator, needs parentheses to stay inside it.
            return f"{left} SUB {_operand(node.children[1], ATOM, gap)}"
        case Kind.ABS:
            # `|u|` is written as a call, as the original does.
            return f"ABS({_written(node.children[0], gap)})"
        case Kind.CALL | Kind.APPLY:
            return f"{node.children[0].value}({_arguments(node.children[1:], gap)})"
        case Kind.FUNCPOW:
            name, exponent, operand = node.children
            return (
                f"{name.value}^{_operand(exponent, POSTFIX, gap)}"
                f"({_written(operand, gap)})"
            )
        case Kind.VECTOR:
            return f"[{_arguments(node.children, gap)}]"
        case Kind.REL:
            left = _operand(node.children[0], RELATION, gap)
            right = _operand(node.children[1], SUM, gap)
            return f"{left}{gap}{node.value}{gap}{right}"
        case Kind.NOT:
            if gap:
                return f"NOT {_operand(node.children[0], RELATION, gap)}"
            return f"NOT({_written(node.children[0], gap)})"
        case Kind.AND | Kind.OR | Kind.XOR | Kind.IMP:
            return _logical(node, gap)
        case Kind.ASSIGN:
            return _assignment(
                node.children[0], node.children[1:], str(node.value), gap
            )
        case Kind.FUNDEF:
            head = f"{node.value}({_arguments(node.children[0].children, gap)})"
            return _assignment(head, node.children[1:], ":=", gap)
        case Kind.DOMAIN:
            return _domain(node, gap)
        case Kind.INTERVAL:
            opening, closing = str(node.value)
            low, high = (_written(child, gap) for child in node.children)
            return f"{opening}{low}, {high}{closing}"
        case Kind.SHOWVALUE:
            return f"{_written(node.children[0], gap)}="
    raise AssertionError(f"unwritable node kind: {node.kind}")


def _written(node: Node, gap: str) -> str:
    """`node` without any precedence requirement, in the current form."""
    return _operand(node, ASSIGN, gap)


def _factor(node: Node, first: bool, gap: str) -> str:
    """One factor of a product run, parenthesized where needed.

    Two cases need care:

    * A nested product is a different tree from a flat run (`(a*b)*c` has two operands,
      `a*b*c` three), so it always keeps its parentheses.
    * `/` and the dot product take the run to their left as their left operand, so
      such a factor can go unparenthesized only in first position: `a/b*c` needs none,
      but `a*(b/c)` does, or `a` would end up in the numerator.
    """
    closes = node.kind is Kind.BINOP and node.value in ("/", ".")
    if node.kind is Kind.PRODUCT or (closes and not first):
        return f"({_written(node, gap)})"
    return _operand(node, PRODUCT, gap)


def _sum(node: Node, gap: str) -> str:
    """A run of terms and the signs between them.

    The first term may carry its own sign (`-a-b` is a sum of two terms), but later
    signed terms are parenthesized, so the only sign between terms is the run's. A
    nested sum is always parenthesized, as with nested products.
    """
    parts = [_operand(node.children[0], NEG, gap)]
    for index, child in enumerate(node.children[1:]):
        sign = str(node.value)[index]
        parts += [f"{gap}{sign}{gap}", _operand(child, PRODUCT, gap)]
    return "".join(parts)


def _binop(node: Node, gap: str) -> str:
    """`/`, the dot product, and `^`.

    `/` and `.` are left-associative, so `a/b/c` needs no parentheses on the left but
    `a/(b/c)` needs them on the right. `^` is right-associative, but the original
    parenthesizes nested powers on both sides anyway.
    """
    operator = str(node.value)
    if operator == "^":
        left = _operand(node.children[0], POSTFIX, gap)
        return f"{left}^{_operand(node.children[1], POSTFIX, gap)}"
    # The dot product is always spaced, since `2 . 3` written as `2.3` would be a
    # numeral.
    if operator == ".":
        operator = " . "
    left = _operand(node.children[0], PRODUCT, gap)
    return f"{left}{operator}{_operand(node.children[1], POW, gap)}"


def _logical(node: Node, gap: str) -> str:
    """A word operator and its operands.

    The tight form parenthesizes the same operator nested on either side, as the
    original does. The spaced form parenthesizes only what the grammar requires, on the
    side the operator doesn't associate toward, so `a AND b AND c` and `(a IMP b) IMP c`
    stay as written.
    """
    own = _LOGICAL_PRECEDENCE[node.kind]
    word = _WORD_OPERATORS[node.kind]
    folds_left = node.kind is Kind.IMP
    if gap:
        left, right = own + 1, own + 1 if folds_left else own
    else:
        # `IMP` is left-associative, so its left operand may nest unparenthesized.
        left, right = own if folds_left else own + 1, own + 1
    return (
        f"{_operand(node.children[0], left, gap)} {word} "
        f"{_operand(node.children[1], right, gap)}"
    )


def _assignment(
    head: str | Node, right: tuple[Node, ...], operator: str, gap: str
) -> str:
    """`x:=u`, or `x:=` with an empty right side."""
    if isinstance(head, Node):
        head = _written(head, gap)
    value = _written(right[0], gap) if right else ""
    return f"{head}{gap}{operator}{gap}{value}" if value else f"{head}{gap}{operator}"


def _domain(node: Node, gap: str) -> str:
    """`x:epsilonReal (0, inf)`, as tight as the original writes it.

    The spaced form is `x :epsilon Real (0, inf)`, which is also how it's typed.
    """
    text = f"{_written(node.children[0], gap)}{gap}:epsilon{gap}{node.value}"
    if len(node.children) > 1:
        text += f" {_written(node.children[1], gap)}"
    return text


def _arguments(children: tuple[Node, ...], gap: str) -> str:
    return f",{gap}".join(_written(child, gap) for child in children)
