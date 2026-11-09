"""Manage Substitute: replace variables or subexpressions with what the user typed.

This command computes nothing. It substitutes and stops, so `a*x^2 + b*x + c` with 2
for x, 3 for a and 5 for b gives `3*2^2 + 5*2 + c`, not 22. The manual (section 4.8)
notes that substitution doesn't commute with simplification, so the user may want to
see the result before simplifying.

This differs from the evaluation other commands start with, which substitutes labels,
assignments and definitions on the way into sympy. This one never touches sympy, which
would simplify.

A replacement is a pair of trees: what to find and what to put in its place. Matching
is structural and exact, so substituting `k` for `t^3` in `t^6 + t^3` gives `t^6 + k`,
not `k^2 + k`. Every match is replaced and substituted values aren't searched again, so
substitution is simultaneous, as in Derive: `x + 2*y` with `y` for `x` and `x` for `y`
is `y + 2*x`, not `x + 2*x`.

The answer is printed and reparsed so the entry's spans index its text.
"""

from __future__ import annotations

from collections.abc import Sequence

from rederive.engine.boundary import Result
from rederive.model.expr import Node
from rederive.syntax import ParseState, parse_expression, write_expression

__all__ = ["Replacement", "replace", "replaced"]

#: One replacement: the subtree to find, and what to put in its place.
Replacement = tuple[Node, Node]


def replace(
    node: Node,
    replacements: Sequence[Replacement] = (),
    state: ParseState | None = None,
) -> Result:
    """Derive's Manage Substitute: `node` with `replacements` written into it.

    Nothing is simplified. `state` is the symbol table used to reparse the answer; a
    session passes its own, which already knows every name the answer can contain.
    """
    text = write_expression(replaced(node, replacements))
    return Result(parse_expression(text, state or ParseState()).node, text)


def replaced(node: Node, replacements: Sequence[Replacement]) -> Node:
    """The substituted tree, before it is written back out.

    The tree-level entry point, as `simplified` is for Simplify. Its spans are
    meaningless, since substituted subtrees keep offsets from wherever they were
    parsed; hence the command prints and reparses.
    """
    for target, value in replacements:
        if same_expression(node, target):
            return value
    children = tuple(replaced(child, replacements) for child in node.children)
    if children == node.children:
        return node
    return Node(node.kind, node.start, node.end, children, node.value, node.surface)


def same_expression(one: Node, other: Node) -> bool:
    """Whether two trees are the same expression, ignoring position and spelling.

    Offsets differ between occurrences of `t^3` in one line. Spelling is also ignored:
    `2x` and `2*x` are the same product, a numeral compares by value, and redundant
    parentheses don't count.
    """
    return (
        one.kind is other.kind
        and one.value == other.value
        and len(one.children) == len(other.children)
        and all(
            same_expression(a, b)
            for a, b in zip(one.children, other.children, strict=True)
        )
    )
