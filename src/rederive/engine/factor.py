"""Factor: the command, on top of `factoring` and the Simplify pipeline.

Derive's Factor is Simplify plus factoring, not factoring on its own, and this
file is that sum: `pipeline` runs first and `factoring` runs on what it
produced. It is what the manual means by saying Factor and Simplify both reach
a sufficiently simple form and Factor goes further, and it means an answer the
original prints may be Simplify's work rather than `factoring`'s - Derive
factors `SIN(x)^2 - 1` to `-COS(x)^2`, and it does that because its Simplify
answers `-COS(x)^2` too.

The order of the two is the whole of what this file decides, and it is not the
obvious one. The pipeline runs at Exact precision whatever the session is set
to, the factoring runs next, and the rounding runs last - because radical
factoring has to reach `SQRT(2)` before Approximate mode has anything to show
as `1.41421`. Rounding first would leave a float with nothing to factor.

`factor` works on any subtree, not only on a whole authored line, so the
session can factor what the user has highlighted and put the answer back.
"""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp

from rederive.engine.boundary import DEFAULT_AMOUNT, Amount, Result
from rederive.engine.context import Context, Precision
from rederive.engine.factoring import factored_expression
from rederive.engine.from_sympy import from_sympy
from rederive.engine.pipeline import approximated, simplified
from rederive.model.expr import Node
from rederive.syntax.state import ParseState

__all__ = ["factor", "factored"]


def factor(
    node: Node,
    context: Context | None = None,
    amount: Amount = DEFAULT_AMOUNT,
    variables: Sequence[str] = (),
    state: ParseState | None = None,
) -> Result:
    """Derive's Factor: `node` written as a product.

    `variables` names the factorization variables in the order they were
    chosen, empty meaning all of them. `state` is the symbol table the answer
    is reparsed with; a session working in a non-default input or case mode
    must pass its own.
    """
    context = context or Context()
    return from_sympy(factored(node, context, amount, variables), context, state)


def factored(
    node: Node,
    context: Context | None = None,
    amount: Amount = DEFAULT_AMOUNT,
    variables: Sequence[str] = (),
) -> sp.Basic:
    """The factored expression, before it is written back out.

    The sympy-level entry point, as `simplified` is for Simplify: a later
    command that wants to keep computing has no reason to print and reparse in
    between.
    """
    context = context or Context()
    expression = simplified(node, context.with_precision(Precision.EXACT))
    return factored_expression(
        expression, amount, variables, lambda e: approximated(e, context)
    )
