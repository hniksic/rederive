"""Expand: the command, on top of `expanding` and the Simplify pipeline.

Like Factor, Derive's Expand is Simplify followed by expanding. The pipeline runs at
Exact precision regardless of the session, expanding runs next, and rounding last, so a
radical partial fraction reaches `SQRT(2)` before Approximate mode shows it as
`1.41421`.

Derive's answer may be Simplify's work rather than the expansion's: Simplify's normal
form already writes every sum about its main variable, so a sum in one variable has
nothing left to expand. Expand differs only for a second variable, a ratio to split into
partial fractions, and a product or power, which Simplify never distributes.

`expand` works on any subtree, so the session can expand a highlighted selection, e.g.
only the `(x - 3)^2` inside `2*x*(x - 3)^2`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import sympy as sp

from rederive.engine.boundary import DEFAULT_AMOUNT, Amount, Result
from rederive.engine.context import Context, Precision
from rederive.engine.expanding import expanded_expression
from rederive.engine.from_sympy import from_sympy
from rederive.engine.printer import shadowing_built_ins
from rederive.engine.approximation import approximated, noting_approximations
from rederive.engine.pipeline import simplified
from rederive.model.expr import Node
from rederive.syntax.state import ParseState

__all__ = ["expand", "expanded"]


@noting_approximations
@shadowing_built_ins
def expand(
    node: Node,
    context: Context | None = None,
    amount: Amount = DEFAULT_AMOUNT,
    variables: Sequence[str] = (),
    state: ParseState | None = None,
) -> Result:
    """Derive's Expand: `node` written as a sum.

    `variables` names the expansion variables in the order chosen, empty meaning all.
    `state` is the symbol table used to reparse the answer; a session in a non-default
    input or case mode must pass its own.

    As in Derive, the expansion variables temporarily override the order list: expanding
    `(x + 2*y + 1)^3` about `y` then `x` gives a polynomial led by `8*y^3`, about `x`
    then `y` one led by `x^3`.
    """
    context = context or Context()
    answer = expanded(node, context, amount, variables)
    return from_sympy(answer, _about(context, variables), state)


def _about(context: Context, variables: Sequence[str]) -> Context:
    """`context` with the chosen expansion variables at the head of its order list.

    The remaining variables keep their session order. With no variables chosen, the
    context is returned unchanged.
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
    """The expanded sympy expression, before it is written back out.

    Like `simplified` for Simplify, this lets a caller keep computing without printing
    and reparsing.
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
