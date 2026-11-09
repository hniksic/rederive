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
    REL = "rel"
    NOT = "not"
    AND = "and"
    OR = "or"
    XOR = "xor"
    IMP = "imp"
    ASSIGN = "assign"
    FUNDEF = "fundef"
    PARAMS = "params"
    DOMAIN = "domain"
    INTERVAL = "interval"
    SHOWVALUE = "showvalue"


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
