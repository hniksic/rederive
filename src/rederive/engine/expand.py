"""Expand: the command, on top of `expanding` and the Simplify pipeline.

Derive's Expand is Simplify plus expanding, exactly as Factor is Simplify plus
factoring, and this file is that sum in the same order and for the same
reasons: the pipeline runs at Exact precision whatever the session is set to,
the expanding runs next, and the rounding runs last, so that a radical partial
fraction reaches `SQRT(2)` before Approximate mode has anything to show as
`1.41421`.

The manual is explicit that both commands reach a sufficiently simple form and
that Expand goes further, and it warns that Simplify is usually faster and
usually stays closer to what was written. An answer the original prints may
therefore be Simplify's work and not the expansion's - and since Simplify's
normal form already writes every sum about its own primary variable, over a
sum in one variable there is nothing left for Expand to do. Where the two part
company is a second variable, a ratio to split into partial fractions, and a
product or a power, which Simplify never distributes and Expand does.

`expand` works on any subtree, not only on a whole authored line, so the
session can expand what the user has highlighted and put the answer back -
which is the difference the manual draws between expanding `2*x*(x - 3)^2`
whole and expanding only the `(x - 3)^2` inside it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import sympy as sp

from rederive.engine.boundary import DEFAULT_AMOUNT, Amount, Result
from rederive.engine.context import Context, Precision
from rederive.engine.expanding import expanded_expression
from rederive.engine.from_sympy import from_sympy
from rederive.engine.pipeline import approximated, simplified
from rederive.model.expr import Node
from rederive.syntax.state import ParseState

__all__ = ["expand", "expanded"]


def expand(
    node: Node,
    context: Context | None = None,
    amount: Amount = DEFAULT_AMOUNT,
    variables: Sequence[str] = (),
    state: ParseState | None = None,
) -> Result:
    """Derive's Expand: `node` written as a sum.

    `variables` names the expansion variables in the order they were chosen,
    empty meaning all of them. `state` is the symbol table the answer is
    reparsed with; a session working in a non-default input or case mode must
    pass its own.

    The expansion variables temporarily override the order list, as the manual
    says they do, and the answer is written in that order: expanding
    `(x + 2*y + 1)^3` about `y` and then `x` gives a polynomial led by `8*y^3`
    where the same expansion about `x` and then `y` is led by `x^3`.
    """
    context = context or Context()
    answer = expanded(node, context, amount, variables)
    return from_sympy(answer, _about(context, variables), state)


def _about(context: Context, variables: Sequence[str]) -> Context:
    """`context` with the chosen expansion variables at the head of its order
    list, which is where an expansion puts them.

    The rest of the list follows, so that a variable nobody chose keeps the
    place the session gave it. Nothing is chosen where the expansion was about
    all of them, and then the list is the session's own.
    """
    if not variables:
        return context
    rest = [name for name in context.order if name not in variables]
    return replace(context, order=(*variables, *rest))


def expanded(
    node: Node,
    context: Context | None = None,
    amount: Amount = DEFAULT_AMOUNT,
    variables: Sequence[str] = (),
) -> sp.Basic:
    """The expanded expression, before it is written back out.

    The sympy-level entry point, as `simplified` is for Simplify: a later
    command that wants to keep computing has no reason to print and reparse in
    between.
    """
    context = context or Context()
    expression = simplified(node, context.with_precision(Precision.EXACT))
    return expanded_expression(
        expression,
        amount,
        variables,
        lambda e: approximated(e, context),
        context.order,
    )
