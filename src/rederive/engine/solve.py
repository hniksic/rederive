"""soLve: the command, on top of `solving` and the Simplify pipeline.

Derive's soLve is Simplify and then solving, the way Factor is Simplify and
then factoring, and the order of the two is the same for the same reason: the
pipeline runs at Exact precision whatever the session is set to, the solving
runs next, and the rounding runs last. Rounding first would leave a float
equation with nothing exact left to solve.

What makes this command unlike every other one is its answer. `simplify`,
`factor` and `expand` each turn one expression into one expression; soLve turns
one expression into *any number of them*, one per solution, and the session
appends them all. So this returns a tuple of `Result` rather than a `Result`,
and the empty tuple is a real answer - "no solutions found", after which
nothing is appended at all.

`solve` works on a whole authored line and not on a subtree, and that is the
one place it parts company with the other commands. Solving half of an
expression produces something there is no sensible way to splice back into the
other half, so the session names the entry and never the highlight.
"""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp

from rederive.engine.context import Context, Precision
from rederive.engine.from_sympy import Result, from_sympy
from rederive.engine.pipeline import approximated, simplified
from rederive.engine.solving import solutions
from rederive.engine.to_sympy import to_sympy
from rederive.model.expr import Kind, Node
from rederive.syntax.state import ParseState

__all__ = ["equations_in", "solve", "solved"]


def solve(
    node: Node,
    context: Context | None = None,
    variables: Sequence[str] = (),
    bounds: tuple[Node, Node] | None = None,
    state: ParseState | None = None,
) -> tuple[Result, ...]:
    """Derive's soLve: what `node` says about the variables it is solved for.

    Each `Result` is one worksheet entry, in the order they are to be appended;
    the empty tuple means no solutions were found. `variables` names what to
    solve for, in the order the user chose them, empty meaning work it out.
    `bounds` is the interval a numeric search is confined to, which Approximate
    precision asks for and the other two modes never do. `state` is the symbol
    table the answers are reparsed with; a session working in a non-default
    input or case mode must pass its own.
    """
    context = context or Context()
    return tuple(
        from_sympy(answer, context, state)
        for answer in solved(node, context, variables, bounds)
    )


def solved(
    node: Node,
    context: Context | None = None,
    variables: Sequence[str] = (),
    bounds: tuple[Node, Node] | None = None,
) -> tuple[sp.Basic, ...]:
    """The solutions, before they are written back out.

    The sympy-level entry point, as `simplified` is for Simplify and `factored`
    for Factor.
    """
    context = context or Context()
    exact = context.with_precision(Precision.EXACT)
    expression = simplified(node, exact)
    interval = (
        None
        if bounds is None
        else (to_sympy(bounds[0], exact), to_sympy(bounds[1], exact))
    )
    return tuple(
        approximated(answer, context)
        for answer in solutions(expression, context, variables, interval)
    )


def equations_in(node: Node) -> int:
    """How many equations `node` is a system of, zero where it is no system.

    Which is the question the UI has to answer before it can ask anything: a
    system of E equations in more than E variables is asked about E times, and
    a scalar equation at most once. Syntactic, like `decomposes`: a system is a
    vector of relations as it was written, and reading that off the tree costs
    nothing.
    """
    if node.kind is not Kind.VECTOR or not node.children:
        return 0
    if not all(child.kind is Kind.REL for child in node.children):
        return 0
    return len(node.children)
