"""Build: the operators its menu offers, and the tree each one makes.

Build is Author for people who would rather point than type. It takes an
expression, then an operator, then - for a binary one - another expression,
and goes on taking operators until Done; what comes out is appended
unsimplified, exactly as an authored line is. Its whole point is that the
operands are picked off the screen, so an expression too long to retype, or a
subexpression buried in one, costs the same three keystrokes as `x`.

The list is the original's, in the original's order, and it is not the list of
everything the grammar can write: it is the handful of operators worth a
keystroke. Anything else is quicker to author.

Nothing here computes. Each operator says how to hang two trees, or one, off a
new node, and what the status line calls the result - `#1+#2` for a sum of two
entries, `SIN(#1)` for a sine of one, `User` standing in for an operand that
was typed rather than pointed at. The spans of what is built index nothing,
because a built expression is written out and read back before it becomes an
entry, the way every derived expression is.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rederive.model.expr import Kind, Node

#: The `1` of `Recip`, which is the one operand no operator is given.
ONE = Node(Kind.NUMBER, 0, 0, (), "1")

#: What the operator menu's last word is called. Not an operator: it says that
#: what has been built is finished with.
DONE = "Done"


@dataclass(frozen=True)
class Operator:
    """One word of the Build operator menu.

    `arity` is how many operands it takes, and so whether choosing it asks for
    another expression. `annotation` is how the status line spells the result:
    a binary operator writes itself between its operands, and a unary one
    writes itself around its one, whether it is a function name or a sign.
    """

    word: str
    arity: int
    build: Callable[..., Node]
    annotation: str

    def annotate(self, *operands: str) -> str:
        return self.annotation.format(*operands)


def _sum(sign: str) -> Callable[[Node, Node], Node]:
    """`u + v` or `u - v`, which is one run of two terms with a sign between."""
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
    """An operator that writes itself between its two operands."""
    return Operator(word, 2, build, "{0}" + word + "{1}")


def _function(word: str) -> Operator:
    """A named function of one operand, invoked and annotated by that name."""
    return Operator(word, 1, _call(word.upper()), word.upper() + "({0})")


#: The operator menu, in the original's order. The four that are neither a
#: function name nor an infix symbol are the interesting ones: `Minus` negates,
#: `Recip` writes `1/u`, and the two symbols beside them are postfix.
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

#: The words the menu shows, which are the operators and then `Done`.
WORDS: tuple[str, ...] = tuple(operator.word for operator in OPERATORS) + (DONE,)

_BY_WORD = {operator.word: operator for operator in OPERATORS}


def operator(word: str) -> Operator | None:
    """The operator `word` names, or None for `Done`, which is not one."""
    return _BY_WORD.get(word)
