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

__all__ = ["DEFAULT_AMOUNT", "SETTLING", "Amount", "PlotForm", "Result"]


class Amount(StrEnum):
    """How hard Factor tries, spelled as in the menu.

    `raDical` is capitalized at its mnemonic letter (`R` is taken by Rational), which
    the menu reads off the word, as are `tUring` and `scHmidt`, which factor a matrix
    (`T` and `S` are taken by Trivial and Squarefree).
    """

    TRIVIAL = "Trivial"
    SQUAREFREE = "Squarefree"
    RATIONAL = "Rational"
    RADICAL = "raDical"
    COMPLEX = "Complex"
    TURING = "tUring"
    SCHMIDT = "scHmidt"


#: The default amount, per the manual and the menu.
DEFAULT_AMOUNT = Amount.RATIONAL


@dataclass(frozen=True)
class Result:
    """An expression as the worksheet holds it: the text and its tree.

    `node`'s spans index `text`.

    `value` is the exact answer, which differs from `node` only where a number is shown
    as something other than its ratio: one third *shown* as `0.333333` is still one
    third. The worksheet saves and draws `node`; the next command computes with `value`.

    `performed` is what the command's programs did to the session on the way, for the
    session to do after it: an `ASSIGN` tree for each global assigned or cleared (with
    its final value), and the `FUNDEF` and `DOMAIN` trees of definitions and
    declarations, in order (see `Store`).

    `displayed` holds the texts the command's `DISPLAY`s showed, in order, for the
    session to add before the answer, and `written` the last text a `WRITE` wrote, for
    the status line. `needs` names a file a `LOAD` asked for that the session hasn't
    looked for yet; the command then ended there, and the rest of the answer means
    nothing (see `FileNeeded`). `random` is the state of the random generator after the
    command's draws, None if it drew nothing (see `Context.random_state`).
    """

    node: Node
    text: str
    value: Node | None = None
    performed: tuple[Node, ...] = ()
    displayed: tuple[str, ...] = ()
    written: str | None = None
    needs: str | None = None
    random: int | None = None

    @property
    def exact(self) -> Node:
        """The tree to compute with: `value` if set, else `node`."""
        return self.node if self.value is None else self.value


#: Seconds the simplification a plot starts from may take before its programs are
#: given up and run at each point instead (see `Context.deadline`). A client that can
#: stop its engine gives the whole request half again as long and then stops it (see
#: `PlotForm`).
SETTLING = 1.5


@dataclass(frozen=True)
class PlotForm(Result):
    """What a plot draws, as the Plot command's simplification finds it.

    Derive simplifies an expression before plotting it, programs running and their
    assignments to globals made, and plots the result. So does Rederive: the fields of
    `Result` are that simplification (`performed` and `random` for the session to take
    over as for Simplify), and the answer is drawn when it is a closed form, one that
    evaluates as numbers at every point by itself.

    Where it isn't, `pointwise` is true and `node` is the expression as written, which is
    evaluated anew at each point the plot samples, programs running: a function defined
    by a program whose flow depends on the variable, a recursion standing on it, a
    built-in that only computes on numbers (`PRIMEPI(x)`), a `RANDOM` drawn at every
    point. Nothing the point evaluations do is kept.

    `variables` are the variables of what is drawn, most main first, without the names
    a program standing in it assigns, and `relation` the relation operator it comes to,
    if any (see `relation_of`).
    """

    variables: tuple[str, ...] = ()
    relation: str | None = None
    pointwise: bool = False
