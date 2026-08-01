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

`spaced` writes the same tree in the engine printer's spelling instead, which is
the spelling of a result line rather than of a file. A blank stands on each side
of a sum's sign and of a relation and after every comma, `NOT` is the prefix
operator it is rather than a call, a Greek letter is written under its name, and
a run of one word operator keeps only the fences the grammar needs: `IF(h<=40,
10*h,400+15*(h-40))` is written `IF(h <= 40, 10*h, 400 + 15*(h - 40))` and
`a OR (b OR c)` is written `a OR b OR c`. The tree is the same either way, and
both spellings read back as it.

The two spellings have to agree wherever both can write a subtree, because the
engine writes an undecidable conditional this way - the one thing a result shows
as authored rather than as computed - and a spelling the printer would not have
chosen would make such an answer move on a second Simplify.
"""

from __future__ import annotations

from rederive.model.expr import Kind, Node
from rederive.syntax.names import GREEK_GLYPHS

# The grammar's precedence ladder, loosest first. This is how tightly the
# parser binds, which is not the ladder the display uses: that one describes
# how tightly a *rendering* reads, where a sign already drawn stands to the
# left of a product rather than around it.
#
# `NOT` is a step of the ladder only in the spaced spelling, where it is
# written as the prefix operator it is: `NOT x = y` is a negated relation, so
# it binds looser than one and tighter than the operators that join them. The
# tight spelling writes it as the call `NOT(u)`, which closes itself off and
# needs no step at all.
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


def write_expression(node: Node, spaced: bool = False) -> str:
    """`node` in author notation, as a file holds it.

    `spaced` writes the printer's spelling of the same tree instead, which is
    the same text with a blank around each sum sign and relation and after each
    comma, `NOT` as a prefix operator, a Greek letter under its name, and no
    fence a run of one word operator does not need.
    """
    return _operand(node, ASSIGN, " " if spaced else "")


def _operand(node: Node, required: int, gap: str) -> str:
    """`node` where an operand binding at least `required` is expected."""
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
            # Leaves, and everything that closes itself off: a call, a vector,
            # `ABS(u)`, and the `NOT(u)` of the tight spelling, which is written
            # as a call as well.
            return ATOM


def _spell(node: Node, gap: str) -> str:
    """`node` without the fences its position may call for."""
    match node.kind:
        case Kind.NUMBER:
            return str(node.surface or node.value)
        case Kind.NAME | Kind.UNKNOWN:
            # A Greek letter is typed and printed under its name and held as
            # the glyph; the file holds the glyph, a result line the name.
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
            # An index reaches one application, so anything written over one -
            # a postfix operator above all - has to be fenced to stay inside it.
            return f"{left} SUB {_operand(node.children[1], ATOM, gap)}"
        case Kind.ABS:
            # `|u|` writes as the call it means, as the original writes it.
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
    """`node` whole, in the spelling the writing already under way is using."""
    return _operand(node, ASSIGN, gap)


def _factor(node: Node, first: bool, gap: str) -> str:
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
        return f"({_written(node, gap)})"
    return _operand(node, PRODUCT, gap)


def _sum(node: Node, gap: str) -> str:
    """A run of terms and the signs between them.

    The leading term keeps a sign of its own - `-a-b` is a sum of two - and
    every later one is fenced, so that the sign between the terms is the only
    one a reader has to account for. A sum nested inside a sum is fenced by
    either demand, for the reason a nested product is.
    """
    parts = [_operand(node.children[0], NEG, gap)]
    for index, child in enumerate(node.children[1:]):
        sign = str(node.value)[index]
        parts += [f"{gap}{sign}{gap}", _operand(child, PRODUCT, gap)]
    return "".join(parts)


def _binop(node: Node, gap: str) -> str:
    """`/`, the dot product, and `^`.

    All three are binary. `/` and `.` fold to the left, so `a/b/c` needs
    nothing on the left and fences on the right. `^` folds to the right, where
    the original fences anyway.
    """
    operator = str(node.value)
    if operator == "^":
        left = _operand(node.children[0], POSTFIX, gap)
        return f"{left}^{_operand(node.children[1], POSTFIX, gap)}"
    # The dot product is spaced, being the one operator whose tight form would
    # fuse with its operands: `2 . 3` written `2.3` is the numeral two-point-
    # three, and the dot would be gone.
    if operator == ".":
        operator = " . "
    left = _operand(node.children[0], PRODUCT, gap)
    return f"{left}{operator}{_operand(node.children[1], POW, gap)}"


def _logical(node: Node, gap: str) -> str:
    """A run of one word operator, and what it demands of each side.

    The tight spelling fences a nest of the same operator on either side, as
    the original does. The spaced spelling fences only what the grammar needs,
    which is where a nest is on the side the operator does not fold to:
    `a AND b AND c` and `(a IMP b) IMP c`, both read back as they stand.
    """
    own = _LOGICAL_PRECEDENCE[node.kind]
    word = _WORD_OPERATORS[node.kind]
    folds_left = node.kind is Kind.IMP
    if gap:
        left, right = own + 1, own + 1 if folds_left else own
    else:
        # `IMP` is the one that folds to the left, so its left operand may nest.
        left, right = own if folds_left else own + 1, own + 1
    return (
        f"{_operand(node.children[0], left, gap)} {word} "
        f"{_operand(node.children[1], right, gap)}"
    )


def _assignment(
    head: str | Node, right: tuple[Node, ...], operator: str, gap: str
) -> str:
    """`x:=u`, or `x:=` for an assignment with nothing on the right."""
    if isinstance(head, Node):
        head = _written(head, gap)
    value = _written(right[0], gap) if right else ""
    return f"{head}{gap}{operator}{gap}{value}" if value else f"{head}{gap}{operator}"


def _domain(node: Node, gap: str) -> str:
    """`x:epsilonReal (0, inf)`, spelled as tight as the original spells it.

    The spaced spelling is the printer's `x :epsilon Real (0, inf)`, which is
    also how it is typed.
    """
    text = f"{_written(node.children[0], gap)}{gap}:epsilon{gap}{node.value}"
    if len(node.children) > 1:
        text += f" {_written(node.children[1], gap)}"
    return text


def _arguments(children: tuple[Node, ...], gap: str) -> str:
    return f",{gap}".join(_written(child, gap) for child in children)
