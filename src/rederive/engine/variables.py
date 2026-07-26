"""Which variables an expression holds.

The order they come out in is `ordering`'s, most main first, which is the order
Factor and Expand offer their variables in - `x^2 - a^2` lists `x,a`, not `a,x`
- and what makes the first one chosen the primary variable.
"""

from __future__ import annotations

import sympy as sp

from rederive.engine.context import Context
from rederive.engine.ordering import main_order
from rederive.engine.substitute import substitute
from rederive.engine.to_sympy import to_sympy
from rederive.model.expr import Node

__all__ = ["expression_variables"]


def expression_variables(node: Node, context: Context | None = None) -> tuple[str, ...]:
    """The variables `node` holds, most main first.

    What a command has to know before it can ask anything, because whether it
    asks at all depends on the answer: one variable is not a choice, so the
    original puts up no prompt unless there are two or more.

    What an assignment has given a value is not among them, since substitution
    has already replaced it by the time there is anything to expand or factor.
    """
    context = context or Context()
    try:
        expression = to_sympy(substitute(node, context), context)
        symbols = expression.free_symbols
    except Exception:
        return ()
    # `type is` rather than `isinstance`: a string literal is a `Symbol`
    # subclass, and a string is data rather than a variable to work about.
    return main_order(
        (s.name for s in symbols if type(s) is sp.Symbol), context.order
    )
