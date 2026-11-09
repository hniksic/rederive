"""Data types shared by both halves of the engine.

An `Amount` goes to the worker and a `Result` comes back. This module imports nothing
that reaches sympy, so the app can use them without loading it (see
`rederive.engine.client`). Anything that needs an expression evaluated belongs on the
worker side.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from rederive.model.expr import Node

__all__ = ["DEFAULT_AMOUNT", "Amount", "Result"]


class Amount(StrEnum):
    """How hard Factor tries, spelled as in the menu.

    `raDical` is capitalized at its mnemonic letter (`R` is taken by Rational), which
    the menu reads off the word.
    """

    TRIVIAL = "Trivial"
    SQUAREFREE = "Squarefree"
    RATIONAL = "Rational"
    RADICAL = "raDical"
    COMPLEX = "Complex"


#: The default amount, per the manual and the menu.
DEFAULT_AMOUNT = Amount.RATIONAL


@dataclass(frozen=True)
class Result:
    """An expression as the worksheet holds it: the text and its tree.

    `node`'s spans index `text`.

    `value` is the exact answer, which differs from `node` only where a number is shown
    as something other than its ratio: one third *shown* as `0.333333` is still one
    third. The worksheet saves and draws `node`; the next command computes with `value`.
    """

    node: Node
    text: str
    value: Node | None = None

    @property
    def exact(self) -> Node:
        """The tree to compute with: `value` if set, else `node`."""
        return self.node if self.value is None else self.value
