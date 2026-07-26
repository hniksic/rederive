"""Running a scalar rewrite over whatever shape an entry came in.

Factor and Expand both rewrite scalars, and both are handed whole worksheet
entries, which need not be scalars at all: a relation, a vector or matrix, a
definition. What makes an entry one of those has to survive the rewrite for
the line to still read as one, so the shape is walked here and the command
sees only the scalars inside it. It is the same distribution Simplify makes,
and for the same reason.
"""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean

from rederive.engine.to_sympy import Assign, Declare, FunDef, InertVector, Logical

__all__ = ["attempt", "distributed"]

#: What to do to one scalar.
Leaf = Callable[[sp.Basic], sp.Basic]


def distributed(expression: sp.Basic, leaf: Leaf) -> sp.Basic:
    """Apply `leaf` to every scalar `expression` is built out of.

    A relation is rewritten side by side, a vector or matrix element by
    element, a definition on its value alone.

    A boolean beyond a single relation comes back untouched. Derive rewrites
    those into normal forms of its own, which are different operations sharing
    a name, and not this milestone's.
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
    """`rewrite`, or the expression unchanged if it will not run.

    Unlike Simplify's gated rewrites this does not ask whether the answer got
    shorter.
    The rewrite was asked for by name, and `(x + 2)*(x - 2)` is the answer to
    Factor even though it counts more operations than `x^2 - 4`.
    """
    try:
        candidate = rewrite(expression)
    except Exception:
        return expression
    return candidate if isinstance(candidate, sp.Expr) else expression
