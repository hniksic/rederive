"""Factor: the command, on top of `factoring` and the Simplify pipeline.

Derive's Factor is Simplify followed by factoring, so its answer may be Simplify's work:
Derive factors `SIN(x)^2 - 1` to `-COS(x)^2` because its Simplify gives `-COS(x)^2` too.

The pipeline runs at Exact precision regardless of the session, factoring runs next, and
rounding last. Radical factoring must reach `SQRT(2)` before Approximate mode shows it as
`1.41421`; rounding first would leave a float with nothing to factor.

`factor` works on any subtree, so the session can factor a highlighted selection.
"""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp

from rederive.engine.boundary import DEFAULT_AMOUNT, Amount, Result
from rederive.engine.context import Context, Precision
from rederive.engine.factoring import factored_expression
from rederive.engine.from_sympy import from_sympy
from rederive.engine.approximation import approximated, noting_approximations
from rederive.engine.pipeline import simplified
from rederive.model.expr import Node
from rederive.syntax.state import ParseState

__all__ = ["factor", "factored"]


@noting_approximations
def factor(
    node: Node,
    context: Context | None = None,
    amount: Amount = DEFAULT_AMOUNT,
    variables: Sequence[str] = (),
    state: ParseState | None = None,
) -> Result:
    """Derive's Factor: `node` written as a product.

    `variables` names the factorization variables in the order chosen, empty meaning
    all. `state` is the symbol table used to reparse the answer; a session in a
    non-default input or case mode must pass its own.
    """
    context = context or Context()
    return from_sympy(factored(node, context, amount, variables), context, state)


def factored(
    node: Node,
    context: Context | None = None,
    amount: Amount = DEFAULT_AMOUNT,
    variables: Sequence[str] = (),
) -> sp.Basic:
    """The factored sympy expression, before it is written back out.

    Like `simplified` for Simplify, this lets a caller keep computing without printing
    and reparsing.
    """
    context = context or Context()
    expression = simplified(node, context.with_precision(Precision.EXACT))
    return factored_expression(
        expression, amount, variables, lambda e: approximated(e, context)
    )
