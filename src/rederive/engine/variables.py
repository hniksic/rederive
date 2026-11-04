"""Which variables an expression holds.

The order they come out in is `ordering`'s, most main first, which is the order
Factor and Expand offer their variables in - `x^2 - a^2` lists `x,a`, not `a,x`
- and what makes the first one chosen the primary variable.
"""

from __future__ import annotations

import sympy as sp

from rederive.engine.context import Context
from rederive.engine.ordering import main_order
from rederive.engine.evaluate import evaluated
from rederive.model.expr import Node

__all__ = ["expression_variables"]


def expression_variables(
    node: Node, context: Context | None = None, written: bool = False
) -> tuple[str, ...]:
    """The variables `node` holds, most main first.

    What a command has to know before it can ask anything, because whether it
    asks at all depends on the answer: one variable is not a choice, so the
    original puts up no prompt unless there are two or more.

    What an assignment has given a value is not among them, since substitution
    has already replaced it by the time there is anything to expand or factor.

    With `written`, an expression that simplifies to one holding no variable
    answers with the variables it was written in instead, so that soLve still has
    something to offer for `x/x = 1`. Those are read off the evaluation with
    sympy's own simplification held back, which leaves a bound variable and an
    assigned one out just as the full evaluation does.
    """
    context = context or Context()
    try:
        symbols = evaluated(node, context).free_symbols
        if written and not _variables(symbols):
            with sp.evaluate(False):
                symbols = evaluated(node, context).free_symbols
    except Exception:
        return ()
    return main_order(_variables(symbols), context.order)


def _variables(symbols: set[sp.Basic]) -> list[str]:
    """The names of the symbols that are variables."""
    # `type is` rather than `isinstance`: a string literal is a `Symbol`
    # subclass, and a string is data rather than a variable to work about.
    return [s.name for s in symbols if type(s) is sp.Symbol]
