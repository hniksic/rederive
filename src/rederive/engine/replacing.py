"""Manage Substitute: what the user says a name stands for, written in.

The one command that computes nothing. It puts a value where a variable was,
or one expression where a highlighted subexpression was, writes the answer out
and stops - so `a*x^2 + b*x + c` with 2 for x, 3 for a and 5 for b comes back
as `3*2^2 + 5*2 + c` and not as 22. The manual makes a point of this (section
4.8): substitution does not commute with simplification, so what was written
in is worth looking at before anything is done to it.

Not to be confused with `substitute`, the pre-pass every other command runs.
That one writes in what a label, an assignment or a definition already stands
for, on the way into sympy. This one writes in what the user has just typed,
and never goes near sympy - there is nothing there for it to do that would not
also simplify.

A replacement is a pair of trees: what to look for, and what to write in its
place. Matching is structural and exact, so substituting `k` for `t^3` in
`t^6 + t^3` gives `t^6 + k` and never `k^2 + k`. Every match is replaced, and
nothing inside what has just been written in is looked at again - which is
what makes the substitution simultaneous, as the original's is: `x + 2*y` with
`y` for `x` and `x` for `y` is `y + 2*x`, not `x + 2*x`.

The answer is written out and read back, as every other command's is, so that
the entry's spans index the text it shows.
"""

from __future__ import annotations

from collections.abc import Sequence

from rederive.engine.boundary import Result
from rederive.model.expr import Node
from rederive.syntax import ParseState, parse_expression, write_expression

__all__ = ["Replacement", "replace", "replaced"]

#: One replacement: the subtree to look for, and what to write in its place.
Replacement = tuple[Node, Node]


def replace(
    node: Node,
    replacements: Sequence[Replacement] = (),
    state: ParseState | None = None,
) -> Result:
    """Derive's Manage Substitute: `node` with `replacements` written into it.

    Nothing is simplified. `state` is the symbol table the answer is reparsed
    with, and a session passes its own: every name the answer can carry is one
    its own lines have already read.
    """
    text = write_expression(replaced(node, replacements))
    return Result(parse_expression(text, state or ParseState()).node, text)


def replaced(node: Node, replacements: Sequence[Replacement]) -> Node:
    """The substituted tree, before it is written back out.

    The tree-level entry point, as `simplified` is for Simplify. Its spans
    index nothing: what was written in brings the offsets of wherever it was
    read from, which is why the command prints and reparses before the answer
    becomes an entry.
    """
    for target, value in replacements:
        if _same(node, target):
            return value
    children = tuple(replaced(child, replacements) for child in node.children)
    if children == node.children:
        return node
    return Node(node.kind, node.start, node.end, children, node.value, node.surface)


def _same(one: Node, other: Node) -> bool:
    """Whether two trees are the same expression, wherever they were written.

    A `Node` carries where it was read from as well as what it says, and two
    occurrences of `t^3` in one line carry different offsets. How it was
    spelled is not part of the question either: `2x` and `2*x` are one product,
    and a numeral is what it is worth rather than the digits it was typed as.
    """
    return (
        one.kind is other.kind
        and one.value == other.value
        and len(one.children) == len(other.children)
        and all(_same(a, b) for a, b in zip(one.children, other.children, strict=True))
    )
