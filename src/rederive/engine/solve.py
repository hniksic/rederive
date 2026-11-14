"""soLve: the command, on top of `solving` and the Simplify pipeline.

Like Factor, Derive's soLve is Simplify followed by solving. The pipeline runs at Exact
precision regardless of the session, solving runs next, and rounding last; rounding
first would leave a float equation with nothing exact to solve.

Simplify is told not to decide the relation. On its own it answers `x = x` with `true`,
as Derive does, but that leaves no unknown; soLve must answer `x = @1` (any value),
which is only reachable from the equation as written.

Unlike other commands, soLve returns *any number* of results, one per solution, and
the session appends them all. Hence a tuple of `Result`; the empty tuple means no
solutions were found.

`solve` also works only on a whole entry, never a subtree, since a solved part of an
expression can't be spliced back into the rest.
"""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp

from rederive.engine.boundary import Result
from rederive.engine.context import Context, Precision
from rederive.engine.from_sympy import from_sympy
from rederive.engine.printer import shadowing_built_ins
from rederive.engine.approximation import approximated, noting_approximations
from rederive.engine.evaluate import divisors
from rederive.engine.pipeline import simplified
from rederive.engine.solving import minting, solutions
from rederive.engine.to_sympy import to_sympy
from rederive.model.expr import Node
from rederive.syntax.state import ParseState

__all__ = ["solve", "solved"]


@noting_approximations
@shadowing_built_ins
def solve(
    node: Node,
    context: Context | None = None,
    variables: Sequence[str] = (),
    bounds: tuple[Node, Node] | None = None,
    state: ParseState | None = None,
) -> tuple[Result, ...]:
    """Derive's soLve: solve `node` for the given variables.

    Each `Result` is one worksheet entry, in append order; the empty tuple means no
    solutions were found. `variables` names what to solve for in the user's order,
    empty meaning determine it automatically. `bounds` limits a numeric search, used
    only in Approximate mode. `state` is the symbol table used to reparse the answers;
    a session in a non-default input or case mode must pass its own.
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
    """The solutions as sympy expressions, before they are written back out.

    The sympy-level entry point, like `simplified` for Simplify and `factored` for
    Factor.
    """
    context = context or Context()
    with minting(context):
        exact = context.with_precision(Precision.EXACT)
        expression = simplified(node, exact, decide=False)
        divided = []
        for divisor in divisors(node):
            try:
                divided.append(simplified(divisor, exact, decide=False))
            except Exception:
                continue
        interval = (
            None
            if bounds is None
            else (to_sympy(bounds[0], exact), to_sympy(bounds[1], exact))
        )
        answers = solutions(expression, context, variables, interval, divided)
    return tuple(approximated(answer, context) for answer in answers)
