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
therefore be Simplify's work and not the expansion's.

`expand` works on any subtree, not only on a whole authored line, so the
session can expand what the user has highlighted and put the answer back -
which is the difference the manual draws between expanding `2*x*(x - 3)^2`
whole and expanding only the `(x - 3)^2` inside it.
"""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp

from rederive.engine.context import Context, Precision
from rederive.engine.expanding import expanded_expression
from rederive.engine.factoring import DEFAULT_AMOUNT, Amount
from rederive.engine.from_sympy import Result, from_sympy
from rederive.engine.pipeline import approximated, simplified
from rederive.model.expr import Kind, Node
from rederive.syntax.state import ParseState

__all__ = ["expand", "expanded", "written_as_ratio"]


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
    """
    context = context or Context()
    return from_sympy(expanded(node, context, amount, variables), context, state)


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


def written_as_ratio(node: Node) -> bool:
    """Whether `node` is written as a quotient, and so has a denominator to factor.

    Which is the question the original asks before it asks for an amount: only
    a ratio can become partial fractions, and only partial fractions need a
    denominator factored. The test is on how the expression is written and not
    on what it is worth, which is the original's own rule and an unforgiving
    one - it asks for an amount for `x^2/x`, whose value has no denominator at
    all, and asks for none for `2/x + 1/x`, whose value does. `x^-1` is a
    power and not a quotient, so it is not asked about either.

    An amount changes nothing where there is no denominator to factor. What it
    changes is how many questions are put.
    """
    return node.kind is Kind.BINOP and node.value == "/"
