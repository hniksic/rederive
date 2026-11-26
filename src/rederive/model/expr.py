"""Expression trees.

A tree is the only thing that crosses the engine/UI boundary: a node kind, the
source span it covers, its operands, and - for leaves and operators - the text
that identifies it.

There is one generic node type rather than a class per kind, because the main consumers
(subexpression navigation, selection highlighting, the typesetter) walk generically over
`(kind, span, children)`. Making `Kind` a `StrEnum` recovers most of the type safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Kind(StrEnum):
    """What a node is."""

    NUMBER = "number"
    NAME = "name"
    STRING = "string"
    LABEL = "label"
    UNKNOWN = "unknown"
    SUM = "sum"
    PRODUCT = "product"
    BINOP = "binop"
    UNOP = "unop"
    POSTOP = "postop"
    SUB = "sub"
    ABS = "abs"
    CALL = "call"
    APPLY = "apply"
    FUNCPOW = "funcpow"
    VECTOR = "vector"
    SET = "set"
    #: The `...` of `[0, 2, ..., 10]`, an element of a vector or set.
    ELLIPSIS = "ellipsis"
    #: A union, intersection or difference of sets; `value` is the operator, one of
    #: `names.SET_OPERATORS`.
    SETOP = "setop"
    REL = "rel"
    NOT = "not"
    AND = "and"
    OR = "or"
    XOR = "xor"
    IMP = "imp"
    IFF = "iff"
    #: A word operator of Derive 6 between two operands; `value` is the word: `ROW`,
    #: `COL`, `SUB SUB` (with one more `SUB` per level deeper), `DOTPRODUCT`,
    #: `CROSSPRODUCT` or `SUBSET`.
    INFIX = "infix"
    #: `target := u`, an update `target :+ u` (`:-`, `:*`, `:/`), or `target :=` with no
    #: value; `value` is the operator. The target is a name, a string, or a name with
    #: subscripts (`a SUB i`).
    ASSIGN = "assign"
    #: A definition `F(x, y := 0) := u`; `value` is the name, the children are the
    #: `PARAMS` and the body, if any.
    FUNDEF = "fundef"
    #: A definition's parameters: each a `NAME`, or an `ASSIGN` of its default. `value`
    #: is `VARIADIC` for the head `F v :=`, whose one parameter takes all the arguments.
    PARAMS = "params"
    DOMAIN = "domain"
    INTERVAL = "interval"
    SHOWVALUE = "showvalue"
    #: The quote `'u`, whose one child is the quoted primary: `u` as written, not
    #: evaluated (a held value).
    QUOTE = "quote"


@dataclass(frozen=True)
class Node:
    """One node of an expression tree.

    `start` and `end` are offsets into `Source.text`; they delimit the text the
    highlight covers when this node is selected. Parentheses are not nodes, and
    a parenthesized operand's span excludes them: in `x (x + 1)` the sum's span
    covers `x + 1`.

    `value` is the canonical spelling: the operator, or a leaf's text. `surface` is
    how it was actually written, or `None` if canonically, so the printer loses
    nothing at parse time.

    `SUM` and `PRODUCT` hold all their terms at once, as in the original, rather than
    being binary. `value` and `surface` then have one character per gap between terms:
    `a-b+c` is a `SUM` of three with `value` `"-+"`. A `PRODUCT`'s value is all `*`,
    and its surface records each gap as `*`, `·`, or a space for juxtaposition.
    """

    kind: Kind
    start: int
    end: int
    children: tuple[Node, ...] = ()
    value: str | None = None
    surface: str | None = None

    @property
    def is_atom(self) -> bool:
        return not self.children


#: The `value` of the `PARAMS` of a variadic head.
VARIADIC = "variadic"

#: The logical operators that associate to the left; `AND`, `OR` and `XOR` associate to
#: the right.
FOLDING_LEFT = frozenset((Kind.IMP, Kind.IFF))

#: The chain operators of `INFIX`, which bind like `SUB`.
CHAIN_WORDS = frozenset(("ROW", "COL"))


def is_chain(node: Node) -> bool:
    """Whether `node` is a link of the subscript chain other than `SUB`: `A ROW i`, `A
    COL j`, `A SUB SUB j`."""
    return node.kind is Kind.INFIX and (node.value in CHAIN_WORDS or is_deep(node))


def is_deep(node: Node) -> bool:
    """Whether `node` is `A SUB SUB j`, or a longer run of `SUB`."""
    return node.kind is Kind.INFIX and set(str(node.value).split()) == {"SUB"}


def depth(node: Node) -> int:
    """How many levels down the index of `A SUB SUB j` applies: one per extra `SUB`."""
    return len(str(node.value).split()) - 1


def parameters(params: Node) -> tuple[tuple[str, Node | None], ...]:
    """The parameters of a definition's `PARAMS`, each a name and its default or None.

    A string parameter, a string variable, is named by its text in quotes, which no other
    name can be.
    """
    found = []
    for child in params.children:
        default = None
        if child.kind is Kind.ASSIGN:
            default = child.children[1] if len(child.children) > 1 else None
            child = child.children[0]
        found.append((variable_key(child), default))
    return tuple(found)


def parameter_names(params: Node) -> tuple[str, ...]:
    """The names of the parameters of a definition's `PARAMS`."""
    return tuple(name for name, _ in parameters(params))


def variable_key(node: Node) -> str:
    """The name of the variable a `NAME` or `STRING` node is, as `parameters` names it:
    a string variable by its text in quotes, so `"a1"` and `a1` are two variables."""
    if node.kind is Kind.STRING:
        return f'"{node.value}"'
    return str(node.value)


#: `parameters`, under a name that leaves the word free for local variables.
parameters_of = parameters
