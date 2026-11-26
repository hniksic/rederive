"""Writing expression trees in author notation, the inverse of the parser.

The output is canonical, not what was typed: juxtaposition becomes `*`, names use the
symbol table's spelling, and a bare application becomes a call. `x (x + 1)` is written
`x*(x + 1)` and `SIN x` is `SIN(x)`. The same text serves everywhere: answers on the
worksheet, the text of an entry, and saved files.

`parse_expression(write_expression(node))` rebuilds `node`, so a saved file reads back
as the worksheet that wrote it, except for normalizations that preserve meaning: an
`APPLY` comes back as a `CALL`, and `|u|` as the same `ABS`. Writing is therefore
idempotent, which the corpus round-trip test checks.

The text has blanks around sum signs, relations and assignments and after commas, `NOT`
as a prefix operator, Greek letters by name, as `pi` is, and `±` for plus-or-minus.
For example, `IF(h<=40,10*h,400+15*(h-40))` is written `IF(h <= 40, 10*h, 400 + 15*(h -
40))`. An operand gets parentheses when it binds more loosely than its position
requires, so `a OR (b OR c)` is written `a OR b OR c`, and in a few places for the
reader's sake: a signed operand inside a run, `a*(-b)*c`; a power inside a power,
`a^(b^c)`, though `x^-1` and `#e^(-1)` (see `_Writer.exponent`); `(n!)!` rather than
`n!!`; and `(x SUBSET y) = w`.

Numerals alone are written as typed, because their digits depend on the input base:
`0FF` under `InputBase := 16` is 255, and writing `255` would read back differently
under the base the file sets.

`written` also returns the tree of the text it wrote, as the parser would read it: each
node's span indexes the text, and each node carries the surface spelling the text gives
it. That is how the engine's answers get their text.
"""

from __future__ import annotations

from rederive.model.expr import FOLDING_LEFT, VARIADIC, Kind, Node, is_chain
from rederive.syntax.names import GREEK_NAMES

# The parser's precedence levels, loosest first. These differ from the display's levels,
# which describe how a rendering reads.
#
# `NOT` is a prefix operator: `NOT x = y` negates a relation, so it binds looser than
# relations and tighter than the logical operators.
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

#: Plus-or-minus is written as the glyph `±`. A bare `+-` would be a plus followed by a
#: minus.
_SIGNS = {"+-": "±"}

#: Precedence of each logical operator.
_LOGICAL_PRECEDENCE = {
    Kind.AND: AND,
    Kind.OR: OR,
    Kind.XOR: XOR,
    Kind.IMP: IMP,
    Kind.IFF: IFF,
}


def write_expression(node: Node) -> str:
    """`node` in author notation; see the module docstring."""
    return written(node)[0]


def written(node: Node) -> tuple[str, Node]:
    """`node` in author notation, and the tree of that text: `node` as the parser reads
    the text back, its spans indexing the text and its surfaces the spellings there."""
    writer = _Writer()
    tree = writer.operand(node, ASSIGN)
    return writer.text(), tree


def precedence(node: Node) -> int:
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
        case Kind.NOT:
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
            # Leaves and self-delimiting forms: calls, vectors, sets and `ABS(u)`.
            return ATOM


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

#: The left operands of a subscript link that are written without parentheses; any
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


def is_return(node: Node) -> bool:
    """Whether `node` is `RETURN u`, which is written `RETURN(u)`."""
    return (
        node.kind is Kind.CALL
        and len(node.children) == 2
        and node.children[0].value == "RETURN"
    )


class _Writer:
    """One expression being written: the text so far, and the tree of what is written.

    Each method writes a node at the end of the text and returns the node as the parser
    reads it back from there: its span covers the text written for it, without the
    parentheses its caller puts around it, and its surface is its spelling where that
    differs from its value, as a token's is.
    """

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.at = 0

    def text(self) -> str:
        return "".join(self.parts)

    def put(self, text: str) -> None:
        self.parts.append(text)
        self.at += len(text)

    def node(
        self,
        kind: Kind,
        start: int,
        children: tuple[Node, ...] = (),
        value: str | None = None,
        surface: str | None = None,
    ) -> Node:
        """The node of what was written from `start` to here."""
        return Node(kind, start, self.at, children, value, surface)

    def leaf(self, kind: Kind, value: str | None, spelled: str) -> Node:
        """A token spelled `spelled`, with the value `value`."""
        start = self.at
        self.put(spelled)
        surface = spelled if spelled != value else None
        return self.node(kind, start, (), value, surface)

    def operand(self, node: Node, required: int) -> Node:
        """`node` as an operand that must bind at least as tightly as `required`.

        An operand that may bind as loosely as an assignment is one that nothing follows
        but a closing bracket, a comma or the end."""
        if precedence(node) < required:
            return self.fenced(node, whole=False)
        return self.spell(node)

    def fenced(self, node: Node, whole: bool = True) -> Node:
        """`node` in parentheses, which its span excludes: as a whole (see `operand`), or
        just spelled, as an operand too loose for its place."""
        self.put("(")
        tree = self.whole(node) if whole else self.spell(node)
        self.put(")")
        return tree

    def whole(self, node: Node) -> Node:
        """`node` without any precedence requirement, in the current form."""
        return self.operand(node, ASSIGN)

    def spell(self, node: Node) -> Node:
        """`node` without any parentheses its position may require."""
        start = self.at
        match node.kind:
            case Kind.NUMBER:
                self.put(str(node.surface or node.value))
                return self.node(Kind.NUMBER, start, (), node.value, node.surface)
            case Kind.NAME | Kind.UNKNOWN:
                # Greek letters are stored as glyphs and written by name, as `pi` is: a
                # glyph may read back as something else, `Γ` as the gamma function.
                name = str(node.value)
                return self.leaf(node.kind, name, GREEK_NAMES.get(name, name))
            case Kind.STRING:
                self.put(f'"{node.value}"')
                return self.node(Kind.STRING, start, (), node.value)
            case Kind.LABEL:
                self.put(f"#{node.value}")
                return self.node(Kind.LABEL, start, (), node.value)
            case Kind.SUM:
                return self.sum(node)
            case Kind.PRODUCT:
                children = []
                for index, child in enumerate(node.children):
                    if index:
                        self.put("*")
                    children.append(self.factor(child, not index))
                return self.node(Kind.PRODUCT, start, tuple(children), node.value)
            case Kind.BINOP:
                return self.binop(node)
            case Kind.UNOP:
                value = str(node.value)
                sign = _SIGNS.get(value, value)
                self.put(sign)
                child = self.operand(node.children[0], NEG)
                surface = sign if sign != value else None
                return self.node(Kind.UNOP, start, (child,), value, surface)
            case Kind.POSTOP:
                operator = str(node.value)
                child = node.children[0]
                if operator == "!" and _is_factorial(child):
                    # `(n!)!`, not `n!!`, which reads as the double factorial.
                    child = self.fenced(child, whole=False)
                else:
                    child = self.operand(child, POSTFIX)
                self.put(" DEG" if operator == "DEG" else operator)
                return self.node(Kind.POSTOP, start, (child,), operator)
            case Kind.SUB:
                return self.link(node, "SUB")
            case Kind.INFIX if is_chain(node):
                return self.link(node, str(node.value))
            case Kind.INFIX:
                return self.infix(node)
            case Kind.ABS:
                # `|u|` is written as a call, as the original does.
                head = self.leaf(Kind.NAME, "ABS", "ABS")
                return self.called(start, head, node.children)
            case Kind.CALL | Kind.APPLY:
                name = node.children[0]
                head = self.leaf(Kind.NAME, name.value, str(name.value))
                return self.called(start, head, node.children[1:])
            case Kind.FUNCPOW:
                name, exponent, operand = node.children
                head = self.leaf(Kind.NAME, name.value, str(name.value))
                self.put("^")
                power = self.operand(exponent, POSTFIX)
                self.put("(")
                argument = self.whole(operand)
                self.put(")")
                return self.node(Kind.FUNCPOW, start, (head, power, argument))
            case Kind.VECTOR:
                return self.listed(Kind.VECTOR, "[", node.children, "]")
            case Kind.SET:
                return self.listed(Kind.SET, "{", node.children, "}")
            case Kind.ELLIPSIS:
                return self.leaf(Kind.ELLIPSIS, node.value, str(node.value))
            case Kind.SETOP:
                # Left-associative: the same operator nests bare on the left only.
                operator = str(node.value)
                own = _SET_PRECEDENCE[operator]
                left = self.operand(node.children[0], own)
                spelled = _SET_SPELLING[operator]
                self.put(spelled)
                right = self.operand(node.children[1], own + 1)
                word = spelled.strip()
                surface = word if word.isalpha() else None
                return self.node(Kind.SETOP, start, (left, right), operator, surface)
            case Kind.REL:
                first, second = node.children
                if _is_subset(first):
                    # `(x SUBSET y) = w`: a comparison of a subset test.
                    left = self.fenced(first, whole=False)
                else:
                    left = self.operand(first, RELATION)
                self.put(f" {node.value} ")
                right = self.operand(second, UNION)
                return self.node(Kind.REL, start, (left, right), node.value)
            case Kind.NOT:
                self.put("NOT ")
                child = self.operand(node.children[0], RELATION)
                return self.node(Kind.NOT, start, (child,))
            case Kind.AND | Kind.OR | Kind.XOR | Kind.IMP | Kind.IFF:
                return self.logical(node)
            case Kind.ASSIGN:
                target = self.whole(node.children[0])
                return self.assigned(start, (target,), node.children[1:], node.value)
            case Kind.FUNDEF:
                self.put(str(node.value))
                parameters = node.children[0]
                if parameters.value == VARIADIC:
                    self.put(" ")
                    declared = self.whole(parameters.children[0])
                    params = Node(
                        Kind.PARAMS,
                        declared.start,
                        declared.end,
                        (declared,),
                        VARIADIC,
                    )
                else:
                    self.put("(")
                    children = self.arguments(parameters.children)
                    first = children[0].start if children else self.at
                    self.put(")")
                    params = Node(Kind.PARAMS, first, self.at, tuple(children))
                return self.assigned(start, (params,), node.children[1:], ":=", node)
            case Kind.DOMAIN:
                return self.domain(node)
            case Kind.INTERVAL:
                opening, closing = str(node.value)
                self.put(opening)
                low = self.whole(node.children[0])
                self.put(", ")
                high = self.whole(node.children[1])
                self.put(closing)
                return self.node(Kind.INTERVAL, start, (low, high), node.value)
            case Kind.SHOWVALUE:
                child = self.whole(node.children[0])
                self.put("=")
                return self.node(Kind.SHOWVALUE, start, (child,))
            case Kind.QUOTE:
                return self.quote(node.children[0])
        raise AssertionError(f"unwritable node kind: {node.kind}")

    def called(self, start: int, head: Node, arguments: tuple[Node, ...]) -> Node:
        """A call of the name `head` just written, with its arguments in parentheses."""
        self.put("(")
        children = self.arguments(arguments)
        self.put(")")
        return self.node(Kind.CALL, start, (head, *children))

    def listed(
        self, kind: Kind, opening: str, elements: tuple[Node, ...], closing: str
    ) -> Node:
        start = self.at
        self.put(opening)
        children = self.arguments(elements)
        self.put(closing)
        return self.node(kind, start, tuple(children))

    def arguments(self, children: tuple[Node, ...]) -> list[Node]:
        written = []
        for index, child in enumerate(children):
            if index:
                self.put(", ")
            written.append(self.whole(child))
        return written

    def quote(self, operand: Node) -> Node:
        """`'u`, the apostrophe directly before its operand, which is parenthesized
        unless it is a name, a number, a string, a vector, a set, a call or another
        quote."""
        start = self.at
        self.put("'")
        if operand.kind not in _QUOTABLE or is_return(operand):
            child = self.fenced(operand)
        else:
            child = self.whole(operand)
        return self.node(Kind.QUOTE, start, (child,))

    def factor(self, node: Node, first: bool) -> Node:
        """One factor of a product run, parenthesized where needed.

        Two cases need care:

        * A nested product is a different tree from a flat run (`(a*b)*c` has two
          operands, `a*b*c` three), so it always keeps its parentheses.
        * `/` and the dot product take the run to their left as their left operand, so
          such a factor can go unparenthesized only in first position: `a/b*c` needs
          none, but `a*(b/c)` does, or `a` would end up in the numerator.
        """
        closes = node.kind is Kind.BINOP and node.value in ("/", ".")
        if node.kind is Kind.PRODUCT or (closes and not first):
            return self.fenced(node)
        return self.operand(node, PRODUCT)

    def sum(self, node: Node) -> Node:
        """A run of terms and the signs between them.

        The first term may carry its own sign (`-a-b` is a sum of two terms), but later
        signed terms are parenthesized, so the only sign between terms is the run's. A
        nested sum is always parenthesized, as with nested products.
        """
        start = self.at
        children = [self.term(node.children[0], NEG)]
        for index, child in enumerate(node.children[1:]):
            sign = str(node.value)[index]
            self.put(f" {sign} ")
            children.append(self.term(child, PRODUCT))
        return self.node(Kind.SUM, start, tuple(children), node.value)

    def term(self, node: Node, required: int) -> Node:
        """A term of a sum. A vector product binds tighter than the sum, so it needs no
        parentheses, unless a later term would start with a sign."""
        if precedence(node) in (DOT, CROSS) and not (required > NEG and _signed(node)):
            return self.spell(node)
        return self.operand(node, required)

    def binop(self, node: Node) -> Node:
        """`/`, the dot product, and `^`.

        `/` and `.` are left-associative, so `a/b/c` needs no parentheses on the left but
        `a/(b/c)` needs them on the right. `^` is right-associative, but the original
        parenthesizes nested powers on both sides anyway.
        """
        start = self.at
        operator = str(node.value)
        if operator == "^":
            left = self.operand(node.children[0], POSTFIX)
            self.put("^")
            right = self.exponent(node.children[0], node.children[1])
        else:
            left = self.operand(node.children[0], PRODUCT)
            # The dot product is always spaced, since `2 . 3` written as `2.3` would be
            # a numeral.
            self.put(" . " if operator == "." else operator)
            right = self.operand(node.children[1], POW)
        return self.node(Kind.BINOP, start, (left, right), operator)

    def exponent(self, base: Node, exponent: Node) -> Node:
        """The exponent of a power of `base`.

        Every power and every signed operand there is parenthesized, except that the
        exponential `#e^u` is written as the function it is, which needs no parentheses
        as an exponent, `#i^#e^u`, and parentheses around a negative argument,
        `#e^(-1)`; while a negative whole power of anything else is written as the
        original shows it, `x^-1` and `10^-7`. A square root is the power it is, and
        parenthesized like one, `x^(SQRT(2))`, and a decimal is set apart from the
        product that may follow, `t^(1.9)`.
        """
        if _is_exponential(exponent):
            return self.spell(exponent)
        if _is_root(exponent) or _is_decimal(exponent):
            return self.fenced(exponent, whole=False)
        if _is_signed_integer(exponent) and not _is_e(base):
            return self.spell(exponent)
        return self.operand(exponent, POSTFIX)

    def logical(self, node: Node) -> Node:
        """A word operator and its operands.

        Parentheses go where the grammar requires them, on the side the operator doesn't
        associate toward, so `a AND b AND c` and `(a IMP b) IMP c` stay as written, and a
        conjunction inside a disjunction needs none: `x = -2 AND a = b OR x = 2 AND a =
        b`.
        """
        start = self.at
        own = _LOGICAL_PRECEDENCE[node.kind]
        word = _WORD_OPERATORS[node.kind]
        left, right = own + 1, own + 1 if node.kind in FOLDING_LEFT else own
        first, second = node.children
        one = self.operand(first, left)
        self.put(f" {word} ")
        other = self.operand(second, right)
        return self.node(node.kind, start, (one, other))

    def link(self, node: Node, word: str) -> Node:
        """A subscript link, `u SUB i`, `A ROW i`, `A COL j` or `A SUB SUB j`.

        The links and postfix operators apply left to right, so the left operand needs no
        parentheses if it is one of them. An index is one primary, so anything larger,
        notably a link or a postfix operator, and a signed number, needs parentheses to
        stay inside it.
        """
        start = self.at
        base = node.children[0]
        if base.kind not in _BARE_BASES and not is_chain(base):
            left = self.fenced(base)
        else:
            left = self.operand(base, POSTFIX)
        self.put(f" {word} ")
        index = self.operand(node.children[1], ATOM)
        surface = word.split()[0]
        return self.node(node.kind, start, (left, index), node.value, surface)

    def infix(self, node: Node) -> Node:
        """`SUBSET`, `DOTPRODUCT` or `CROSSPRODUCT`, all left-associative."""
        start = self.at
        own = precedence(node)
        left = self.operand(node.children[0], own)
        self.put(f" {node.value} ")
        right = self.operand(node.children[1], own + 1)
        return self.node(Kind.INFIX, start, (left, right), node.value, node.value)

    def assigned(
        self,
        start: int,
        head: tuple[Node, ...],
        right: tuple[Node, ...],
        operator: str | None,
        definition: Node | None = None,
    ) -> Node:
        """`x:=u`, or `x:=` with an empty right side, the head already written; a
        definition's head is its name and `PARAMS`."""
        if right:
            self.put(f" {operator} ")
            children = (*head, self.whole(right[0]))
        else:
            self.put(f" {operator}")
            children = head
        if definition is not None:
            return self.node(Kind.FUNDEF, start, children, definition.value)
        return self.node(Kind.ASSIGN, start, children, operator)

    def domain(self, node: Node) -> Node:
        """`x :epsilon Real (0, inf)`, as it's typed."""
        start = self.at
        children = [self.whole(node.children[0])]
        self.put(f" :epsilon {node.value}")
        if len(node.children) > 1:
            self.put(" ")
            children.append(self.whole(node.children[1]))
        return self.node(Kind.DOMAIN, start, tuple(children), node.value)


def _is_factorial(node: Node) -> bool:
    return node.kind is Kind.POSTOP and node.value == "!"


def _is_subset(node: Node) -> bool:
    return node.kind is Kind.INFIX and node.value == "SUBSET"


def _is_e(node: Node) -> bool:
    return node.kind is Kind.NAME and node.value == "#e"


def _is_exponential(node: Node) -> bool:
    """Whether `node` is `#e^u`."""
    return node.kind is Kind.BINOP and node.value == "^" and _is_e(node.children[0])


def _is_root(node: Node) -> bool:
    return node.kind is Kind.CALL and node.children[0].value == "SQRT"


def _is_decimal(node: Node) -> bool:
    """Whether `node` is a numeral with a point, `1.5`."""
    return node.kind is Kind.NUMBER and "." in str(node.surface or node.value)


def _is_signed_integer(node: Node) -> bool:
    """Whether `node` is a negative whole numeral, `-1`."""
    if node.kind is not Kind.UNOP or node.value != "-":
        return False
    number = node.children[0]
    return number.kind is Kind.NUMBER and not _is_decimal(number)


def _signed(node: Node) -> bool:
    """Whether `node` is written starting with a sign."""
    while node.kind is Kind.INFIX and precedence(node) in (DOT, CROSS):
        node = node.children[0]
    return node.kind is Kind.UNOP

