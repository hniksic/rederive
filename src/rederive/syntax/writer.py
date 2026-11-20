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

from dataclasses import replace

from rederive.model.expr import VARIADIC, Kind, Node, is_chain
from rederive.syntax.names import (
    CONSTANT_GLYPHS,
    DOS_GREEK,
    FUNCTION_GLYPHS,
    GREEK_NAMES,
)

#: Greek letters that would read back as something else, so files spell them by name:
#: `Γ` is the gamma function and `Ω` a small omega until declared otherwise.
_MISREAD_LETTERS = frozenset(GREEK_NAMES) & (
    frozenset(DOS_GREEK) | frozenset(FUNCTION_GLYPHS) | frozenset(CONSTANT_GLYPHS)
)

# The parser's precedence levels, loosest first. These differ from the display's levels,
# which describe how a rendering reads.
#
# `NOT` has its own level only in the spaced form, where it's a prefix operator: `NOT x =
# y` negates a relation, so it binds looser than relations and tighter than the logical
# operators. The tight form writes it as the call `NOT(u)`, which needs no level.
(
    ASSIGN,
    IFF,
    IMP,
    XOR,
    OR,
    AND,
    NOT,
    RELATION,
    UNION,
    INTERSECTION,
    DIFFERENCE,
    SUM,
    DOT,
    CROSS,
    NEG,
    PRODUCT,
    POW,
    POSTFIX,
    SUB,
    ATOM,
) = range(20)

_BINOP_PRECEDENCE = {"/": PRODUCT, ".": PRODUCT, "^": POW}

#: Precedence of each set operator, and how it is written. The difference is written
#: without blanks, as the original does.
_SET_PRECEDENCE = {"UNION": UNION, "INTERSECTION": INTERSECTION, "\\": DIFFERENCE}
_SET_SPELLING = {"UNION": " UNION ", "INTERSECTION": " INTERSECTION ", "\\": "\\"}

#: Operators spelled as words, which need a blank on each side to stay separate from
#: adjacent names.
_WORD_OPERATORS = {
    Kind.AND: "AND",
    Kind.OR: "OR",
    Kind.XOR: "XOR",
    Kind.IMP: "IMP",
    Kind.IFF: "IFF",
}

#: Precedence of the word operators of `INFIX` other than the subscript links.
_INFIX_PRECEDENCE = {"SUBSET": RELATION, "DOTPRODUCT": DOT, "CROSSPRODUCT": CROSS}

#: Plus-or-minus is written as typed, with quotes, rather than as the displayed `±`
#: glyph, as the original does. A bare `+-` would be a plus followed by a minus.
_SIGNS = {"+-": '"+-"'}

#: Precedence of each logical operator. `AND`, `OR` and `XOR` are right-associative, so
#: the same operator nested on the left needs parentheses; the original also adds them
#: on the right.
_LOGICAL_PRECEDENCE = {
    Kind.AND: AND,
    Kind.OR: OR,
    Kind.XOR: XOR,
    Kind.IMP: IMP,
    Kind.IFF: IFF,
}

#: The logical operators that associate to the left.
_FOLDING_LEFT = frozenset((Kind.IMP, Kind.IFF))


def write_expression(node: Node, spaced: bool = False, greek_names: bool = False) -> str:
    """`node` in author notation, as stored in a file.

    With `spaced`, write it as the printer does instead; see the module docstring.
    With `greek_names`, Greek letters are written by name even unspaced, as in an
    entry's text, which spells `sigma` like `pi` and leaves the glyphs to the display.
    """
    if greek_names and not spaced:
        node = _greek_named(node)
    return _operand(node, ASSIGN, " " if spaced else "")


def _greek_named(node: Node) -> Node:
    """`node` with each Greek letter named by its spelled-out name."""
    if node.kind in (Kind.NAME, Kind.UNKNOWN) and str(node.value) in GREEK_NAMES:
        return replace(node, value=GREEK_NAMES[str(node.value)])
    if not node.children:
        return node
    return replace(node, children=tuple(_greek_named(child) for child in node.children))


def _operand(node: Node, required: int, gap: str) -> str:
    """`node` as an operand that must bind at least as tightly as `required`.

    An operand that may bind as loosely as an assignment is one that nothing follows
    but a closing bracket, a comma or the end."""
    if required == ASSIGN and node.kind in _LOGICAL_PRECEDENCE:
        return _logical(node, gap, closed=True)
    text = _spell(node, gap)
    return f"({text})" if precedence(node, gap) < required else text


def precedence(node: Node, gap: str = "") -> int:
    """How tightly `node`'s written form binds."""
    match node.kind:
        case Kind.ASSIGN | Kind.FUNDEF | Kind.SHOWVALUE:
            return ASSIGN
        case Kind.CALL if is_return(node):
            # `RETURN(u)` reads whatever follows into its operand, as `RETURN u` does.
            return ASSIGN
        case Kind.IFF | Kind.IMP | Kind.XOR | Kind.OR | Kind.AND:
            return _LOGICAL_PRECEDENCE[node.kind]
        case Kind.INFIX:
            return SUB if is_chain(node) else _INFIX_PRECEDENCE[str(node.value)]
        case Kind.NOT if gap:
            return NOT
        case Kind.REL | Kind.DOMAIN:
            return RELATION
        case Kind.SETOP:
            return _SET_PRECEDENCE[str(node.value)]
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
            # Leaves and self-delimiting forms: calls, vectors, sets, `ABS(u)`, and the
            # tight form's `NOT(u)`.
            return ATOM


def _spell(node: Node, gap: str) -> str:
    """`node` without any parentheses its position may require."""
    match node.kind:
        case Kind.NUMBER:
            return str(node.surface or node.value)
        case Kind.NAME | Kind.UNKNOWN:
            # Greek letters are stored as glyphs. Files get the glyph, unless it would
            # read back as something else, result lines the name.
            name = str(node.value)
            if gap or name in _MISREAD_LETTERS:
                return GREEK_NAMES.get(name, name)
            return name
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
            operator = str(node.value)
            if operator == "DEG":
                operator = " DEG"
            return _operand(node.children[0], POSTFIX, gap) + operator
        case Kind.SUB:
            return _link(node, "SUB", gap)
        case Kind.INFIX if is_chain(node):
            return _link(node, str(node.value), gap)
        case Kind.INFIX:
            return _infix(node, gap)
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
        case Kind.SET:
            return f"{{{_arguments(node.children, gap)}}}"
        case Kind.ELLIPSIS:
            return str(node.value)
        case Kind.SETOP:
            # Left-associative: the same operator nests bare on the left only.
            own = _SET_PRECEDENCE[str(node.value)]
            left = _operand(node.children[0], own, gap)
            right = _operand(node.children[1], own + 1, gap)
            return f"{left}{_SET_SPELLING[str(node.value)]}{right}"
        case Kind.REL:
            left = _operand(node.children[0], RELATION, gap)
            right = _operand(node.children[1], UNION, gap)
            return f"{left}{gap}{node.value}{gap}{right}"
        case Kind.NOT:
            if gap:
                return f"NOT {_operand(node.children[0], RELATION, gap)}"
            return f"NOT({_written(node.children[0], gap)})"
        case Kind.AND | Kind.OR | Kind.XOR | Kind.IMP | Kind.IFF:
            return _logical(node, gap)
        case Kind.ASSIGN:
            return _assignment(
                node.children[0], node.children[1:], str(node.value), gap
            )
        case Kind.FUNDEF:
            parameters = node.children[0]
            if parameters.value == VARIADIC:
                head = f"{node.value} {_written(parameters.children[0], gap)}"
            else:
                head = f"{node.value}({_arguments(parameters.children, gap)})"
            return _assignment(head, node.children[1:], ":=", gap)
        case Kind.DOMAIN:
            return _domain(node, gap)
        case Kind.INTERVAL:
            opening, closing = str(node.value)
            low, high = (_written(child, gap) for child in node.children)
            return f"{opening}{low}, {high}{closing}"
        case Kind.SHOWVALUE:
            return f"{_written(node.children[0], gap)}="
        case Kind.QUOTE:
            return _quote(node.children[0], gap)
    raise AssertionError(f"unwritable node kind: {node.kind}")


#: What a quote's operand may be without parentheses: the primaries that read back as the
#: one operand of the quote.
_QUOTABLE = frozenset(
    (
        Kind.NAME,
        Kind.NUMBER,
        Kind.STRING,
        Kind.LABEL,
        Kind.UNKNOWN,
        Kind.VECTOR,
        Kind.SET,
        Kind.CALL,
        Kind.APPLY,
        Kind.FUNCPOW,
        Kind.ABS,
        Kind.QUOTE,
    )
)


def _quote(operand: Node, gap: str) -> str:
    """`'u`, the apostrophe directly before its operand, which is parenthesized unless
    it is a name, a number, a string, a vector, a set, a call or another quote."""
    text = _written(operand, gap)
    if operand.kind not in _QUOTABLE or is_return(operand):
        text = f"({text})"
    return f"'{text}"


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
    parts = [_term(node.children[0], NEG, gap)]
    for index, child in enumerate(node.children[1:]):
        sign = str(node.value)[index]
        parts += [f"{gap}{sign}{gap}", _term(child, PRODUCT, gap)]
    return "".join(parts)


def _term(node: Node, required: int, gap: str) -> str:
    """A term of a sum. A vector product binds tighter than the sum, so it needs no
    parentheses, unless a later term would start with a sign."""
    if precedence(node) in (DOT, CROSS) and not (required > NEG and _signed(node)):
        return _spell(node, gap)
    return _operand(node, required, gap)


def _signed(node: Node) -> bool:
    """Whether `node` is written starting with a sign."""
    while node.kind is Kind.INFIX and precedence(node) in (DOT, CROSS):
        node = node.children[0]
    return node.kind is Kind.UNOP


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


def _logical(node: Node, gap: str, closed: bool = False) -> str:
    """A word operator and its operands.

    The tight form parenthesizes the same operator nested on either side, as the
    original does. The spaced form parenthesizes what the grammar requires, on the
    side the operator doesn't associate toward, so `a AND b AND c` and `(a IMP b) IMP c`
    stay as written. Both parenthesize a conjunction inside a disjunction.

    Where nothing follows the operator (`closed`), the tight form writes an assignment
    on its right bare, as the original does: `(m:=v SUB 3) AND c:=v SUB 4`.
    """
    own = _LOGICAL_PRECEDENCE[node.kind]
    word = _WORD_OPERATORS[node.kind]
    folds_left = node.kind in _FOLDING_LEFT
    if gap:
        left, right = own + 1, own + 1 if folds_left else own
    else:
        # `IMP` is left-associative, so its left operand may nest unparenthesized.
        left, right = own if folds_left else own + 1, own + 1
    first, second = node.children
    left = _conjunction_fenced(node, first, left)
    right = _conjunction_fenced(node, second, right)
    if closed and not gap and second.kind is Kind.ASSIGN:
        # `a AND c:=u`: the assignment's value extends to the end, as it is read.
        written = _written(second, gap)
    else:
        written = _operand(second, right, gap)
    return f"{_operand(first, left, gap)} {word} {written}"


def _conjunction_fenced(node: Node, operand: Node, required: int) -> int:
    """The precedence `operand` of the logical `node` must have: a conjunction inside a
    disjunction is parenthesized although `AND` binds more tightly, as the original
    shows and writes a disjunction of solutions, `(x = -a AND y = -a) OR (x = a AND y =
    a)`."""
    if node.kind is Kind.OR and operand.kind is Kind.AND:
        return max(required, AND + 1)
    return required


def _link(node: Node, word: str, gap: str) -> str:
    """A subscript link, `u SUB i`, `A ROW i`, `A COL j` or `A SUB SUB j`.

    The links and postfix operators apply left to right, so the left operand needs no
    parentheses if it is one of them. An index is one primary, so anything larger,
    notably a link or a postfix operator, and a signed number, needs parentheses to
    stay inside it.
    """
    base = node.children[0]
    left = _operand(base, POSTFIX, gap)
    if not gap and base.kind not in _BARE_BASES and not is_chain(base):
        left = f"({_written(base, gap)})"
    if not gap:
        # Files write the column subscript with two blanks between the words.
        word = "  ".join(word.split())
    return f"{left} {word} {_operand(node.children[1], ATOM, gap)}"


#: The left operands of a subscript link that files write without parentheses; any
#: other, a call included, is parenthesized: `(FIRST(t)) SUB 2`.
_BARE_BASES = frozenset(
    (
        Kind.NAME,
        Kind.NUMBER,
        Kind.STRING,
        Kind.VECTOR,
        Kind.SET,
        Kind.SUB,
        Kind.POSTOP,
        Kind.LABEL,
        Kind.UNKNOWN,
        Kind.QUOTE,
    )
)


def _infix(node: Node, gap: str) -> str:
    """`SUBSET`, `DOTPRODUCT` or `CROSSPRODUCT`, all left-associative."""
    own = precedence(node)
    left = _operand(node.children[0], own, gap)
    right = _operand(node.children[1], own + 1, gap)
    return f"{left} {node.value} {right}"


def is_return(node: Node) -> bool:
    """Whether `node` is `RETURN u`, which is written `RETURN(u)`."""
    return (
        node.kind is Kind.CALL
        and len(node.children) == 2
        and node.children[0].value == "RETURN"
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
