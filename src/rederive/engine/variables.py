"""Which variables an expression holds, and whether it is a relation.

The order the variables come out in is `ordering`'s, most main first, which is the
order Factor and Expand offer their variables in - `x^2 - a^2` lists `x,a`, not
`a,x` - and what makes the first one chosen the primary variable.

Both questions are asked of the evaluation rather than of the tree, because the
tree can be written around what it means: `#3` is a label, and `IM(u = v)` is a
function applied to an equation, which the engine answers with an equation.
"""

from __future__ import annotations

import sympy as sp
from sympy.core.relational import Relational

from rederive.engine.context import Context
from rederive.engine.ordering import main_order
from rederive.engine.evaluate import evaluated
from rederive.model.expr import Node

__all__ = ["expression_variables", "relation_of"]

#: How a relation's operator is written, keyed by sympy's spelling of it. The
#: orderings are spelled alike on both sides and are not listed.
_RELATIONS = {"==": "=", "!=": "/="}


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


def relation_of(node: Node, context: Context | None = None) -> str | None:
    """The operator `node` evaluates to a relation under, or None where it does not.

    Written as the notation writes it: `=`, `/=`, `<`, `<=`, `>`, `>=`. What the Plot
    command asks before it decides what a line is a plot of, since a function
    applied to an equation, `IM(2^(x + #i*y) = (x + #i*y)^6)`, is a call on the tree
    and an equation once evaluated, and it is the equation that is drawn.
    """
    try:
        value = evaluated(node, context or Context())
    except Exception:
        return None
    if not isinstance(value, Relational):
        return None
    return _RELATIONS.get(value.rel_op, value.rel_op)


def _variables(symbols: set[sp.Basic]) -> list[str]:
    """The names of the symbols that are variables."""
    # `type is` rather than `isinstance`: a string literal is a `Symbol`
    # subclass, and a string is data rather than a variable to work about.
    return [s.name for s in symbols if type(s) is sp.Symbol]
