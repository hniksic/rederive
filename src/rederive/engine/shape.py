"""Applying a scalar rewrite inside a non-scalar entry.

Factor and Expand rewrite scalars but receive whole entries, which may be relations,
vectors, matrices or definitions. The structure must survive the rewrite, so it is
walked here and the command sees only the scalars inside, as in Simplify.
"""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean

from rederive.engine.to_sympy import Assign, Declare, FunDef, InertVector, Logical

__all__ = ["attempt", "distributed"]

#: A rewrite of one scalar.
Leaf = Callable[[sp.Basic], sp.Basic]


def distributed(expression: sp.Basic, leaf: Leaf) -> sp.Basic:
    """Apply `leaf` to every scalar in `expression`.

    A relation is rewritten side by side, a vector or matrix element-wise, and a
    definition on its value only.

    Other booleans are returned untouched. Derive rewrites those into its own normal
    forms, which are different operations not yet implemented.
    """

    def again(part: sp.Basic) -> sp.Basic:
        return distributed(part, leaf)

    if isinstance(expression, Relational):
        left, right = again(expression.lhs), again(expression.rhs)
        try:
            return expression.func(left, right, evaluate=False)
        except Exception:
            return expression.func(left, right)
    if isinstance(expression, Declare):
        return expression
    if isinstance(expression, (Assign, FunDef)):
        head, operator, *value = expression.args
        return expression.func(head, operator, *(again(part) for part in value))
    if isinstance(expression, sp.MatrixBase):
        return expression.applyfunc(again)
    if isinstance(expression, (InertVector, Logical)):
        return expression.func(*(again(argument) for argument in expression.args))
    if isinstance(expression, Boolean):
        return expression
    return leaf(expression)


def attempt(expression: sp.Expr, rewrite: Callable[[sp.Expr], sp.Basic]) -> sp.Expr:
    """Apply `rewrite`, or return the expression unchanged if it fails.

    Unlike Simplify's gated rewrites this doesn't require the answer to be shorter,
    since the rewrite was explicitly requested: `(x + 2)*(x - 2)` is Factor's answer
    though it has more operations than `x^2 - 4`.
    """
    try:
        candidate = rewrite(expression)
    except Exception:
        return expression
    return candidate if isinstance(candidate, sp.Expr) else expression
