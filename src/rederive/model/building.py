"""The operators of the Build menu, and the tree each one makes.

Build is like Author, but operands are picked off the screen rather than typed. It takes
an expression, then an operator, then (for a binary operator) another expression, and
repeats until Done. The result is appended unsimplified, like an authored line.

The operator list is the original's, in its order: just the few operators worth a menu
entry, not everything the grammar supports.

Each operator builds a new node from one or two trees and provides the status line's
annotation of the result: `#1+#2`, `SIN(#1)`, or `User` for an operand that was typed.
Spans of built nodes are meaningless, because the built expression is written out and
reparsed before it becomes an entry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rederive.model.expr import Kind, Node

#: The `1` in `Recip`'s `1/u`.
ONE = Node(Kind.NUMBER, 0, 0, (), "1")

#: The last word of the operator menu, which finishes building rather than applying
#: an operator.
DONE = "Done"


@dataclass(frozen=True)
class Operator:
    """One word of the Build operator menu.

    `arity` is the number of operands, which determines whether choosing it asks for
    another expression. `annotation` is a format string for the status line's
    description of the result.
    """

    word: str
    arity: int
    build: Callable[..., Node]
    annotation: str

    def annotate(self, *operands: str) -> str:
        return self.annotation.format(*operands)


def _sum(sign: str) -> Callable[[Node, Node], Node]:
    """Build `u + v` or `u - v`, a two-term `SUM`."""
    return lambda left, right: Node(Kind.SUM, 0, 0, (left, right), sign)


def _product(left: Node, right: Node) -> Node:
    return Node(Kind.PRODUCT, 0, 0, (left, right), "*")


def _binop(symbol: str) -> Callable[[Node, Node], Node]:
    return lambda left, right: Node(Kind.BINOP, 0, 0, (left, right), symbol)


def _relation(left: Node, right: Node) -> Node:
    return Node(Kind.REL, 0, 0, (left, right), "=")


def _negation(operand: Node) -> Node:
    return Node(Kind.UNOP, 0, 0, (operand,), "-")


def _reciprocal(operand: Node) -> Node:
    return Node(Kind.BINOP, 0, 0, (ONE, operand), "/")


def _postfix(symbol: str) -> Callable[[Node], Node]:
    return lambda operand: Node(Kind.POSTOP, 0, 0, (operand,), symbol)


def _call(name: str) -> Callable[[Node], Node]:
    return lambda operand: Node(
        Kind.CALL, 0, 0, (Node(Kind.NAME, 0, 0, (), name), operand)
    )


def _binary(word: str, build: Callable[[Node, Node], Node]) -> Operator:
    """Return an infix operator."""
    return Operator(word, 2, build, "{0}" + word + "{1}")


def _function(word: str) -> Operator:
    """Return a one-argument function call operator."""
    return Operator(word, 1, _call(word.upper()), word.upper() + "({0})")


#: The operator menu, in the original's order. `Minus` negates, `Recip` builds `1/u`,
#: and the backquote, `!` and `%` are postfix.
OPERATORS: tuple[Operator, ...] = (
    _binary("+", _sum("+")),
    _binary("-", _sum("-")),
    _binary("*", _product),
    _binary("/", _binop("/")),
    _binary("^", _binop("^")),
    _binary(".", _binop(".")),
    Operator("`", 1, _postfix("`"), "`({0})"),
    _binary("=", _relation),
    Operator("Minus", 1, _negation, "-({0})"),
    Operator("Recip", 1, _reciprocal, "1/({0})"),
    _function("Ln"),
    _function("Exp"),
    _function("Tan"),
    _function("Sin"),
    _function("Cos"),
    _function("Atan"),
    Operator("!", 1, _postfix("!"), "!({0})"),
    Operator("%", 1, _postfix("%"), "%({0})"),
)

#: The menu's words: the operators, then `Done`.
WORDS: tuple[str, ...] = tuple(operator.word for operator in OPERATORS) + (DONE,)

_BY_WORD = {operator.word: operator for operator in OPERATORS}


def operator(word: str) -> Operator | None:
    """Return the operator named `word`, or None for `Done`."""
    return _BY_WORD.get(word)
