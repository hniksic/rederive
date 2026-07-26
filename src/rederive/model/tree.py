"""Expression trees.

A tree is the only thing the rest of the program knows about the structure of
an authored expression: a node kind, the source span it covers, and its
operands. Selection and navigation are defined purely in these terms, so the
placeholder parser can be replaced by a real engine without touching anything
else.

TODO(display): superseded by rederive.model.expr.Node. Delete with
rederive.placeholder_parser once the UI is wired to the real parser.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Node:
    """One node of an expression tree.

    `start` and `end` are offsets into the raw text the expression was
    authored from; they delimit the text the highlight covers when this node
    is selected. Parentheses are not nodes, and a parenthesized operand's span
    excludes them: in `x (x + 1)` the sum's span covers `x + 1`.
    """

    kind: str
    start: int
    end: int
    children: tuple[Node, ...] = field(default_factory=tuple)

    @property
    def is_atom(self) -> bool:
        return not self.children
