"""The vocabulary both sides of the engine speak.

An `Amount` goes down to the worker and a `Result` comes back, so both halves name
them and neither may own them. They live here, apart from the mathematics, because of
what this module may not import: nothing under `sympy`, and nothing that reaches it.
That is what lets the app side hold an answer and name an amount without loading a
computer algebra system to do it - `rederive.engine.client` says the rest.

Everything here is data. A rule that needs an expression evaluated belongs on the
worker side, whatever it is about.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from rederive.model.expr import Node

__all__ = ["DEFAULT_AMOUNT", "Amount", "Result"]


class Amount(StrEnum):
    """How hard Factor tries, in the words the menu spells them with.

    `raDical` carries its capital because that is where its mnemonic letter is
    - `R` belongs to Rational - and the menu reads the letter off the word.
    """

    TRIVIAL = "Trivial"
    SQUAREFREE = "Squarefree"
    RATIONAL = "Rational"
    RADICAL = "raDical"
    COMPLEX = "Complex"


#: What the amount is when nobody says. The manual's default, and the menu's.
DEFAULT_AMOUNT = Amount.RATIONAL


@dataclass(frozen=True)
class Result:
    """An expression as the worksheet holds it: the text, and its tree.

    `node`'s spans index `text`.

    `value` is the same answer written exactly, which is a different tree only
    where the notation writes a number as something other than the ratio it
    is: one third *shown* as `0.333333` is still one third, and three times it
    is still one. What is shown is what the worksheet saves and draws; what is
    here is what the next command computes with.
    """

    node: Node
    text: str
    value: Node | None = None

    @property
    def exact(self) -> Node:
        """The tree to compute with: the exact one where they differ."""
        return self.node if self.value is None else self.value
