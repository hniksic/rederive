"""Expression trees.

A tree is the only thing that crosses the engine/UI boundary: a node kind, the
source span it covers, its operands, and - for leaves and operators - the text
that identifies it.

One generic node type rather than a class per kind. The dominant consumers
(subexpression navigation, inverse-video selection, the built-up typesetter)
all walk generically over `(kind, span, children)`, and a class hierarchy would
force `isinstance` dispatch through all of it for no gain. `Kind` is a
`StrEnum`, which recovers most of the type safety.
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

    `value` carries the canonical spelling - the operator, or a leaf's text.
    `surface` carries how it was actually written when that differs, so that
    nothing a printer would need is discarded at parse time: `None` means
    "spelled canonically".

    `SUM` and `PRODUCT` are runs rather than pairs, holding all their terms at
    once the way the original does. Their operators sit between the terms, so
    `value` and `surface` carry one character per gap and are one shorter than
    `children`: `a-b+c` is a `SUM` of three with `value` `"-+"`. A `PRODUCT`
    is all `*` canonically, and its surface says how each gap was written -
    `*`, `·`, or a space for juxtaposition, which is written as nothing at
    all.
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
