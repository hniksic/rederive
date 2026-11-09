"""Which variables an expression holds, and whether it is a relation.

Variables are listed most main first, per `ordering`, which is the order Factor and
Expand offer them in: `x^2 - a^2` lists `x,a`, not `a,x`.

Both questions are answered from the evaluated expression rather than the tree, since
the tree may not show its meaning: `#3` is a label, and `IM(u = v)` is a function call
that evaluates to an equation.
"""

from __future__ import annotations

import sympy as sp
from sympy.core.relational import Relational

from rederive.engine.context import Context
from rederive.engine.ordering import main_order
from rederive.engine.evaluate import evaluated
from rederive.model.expr import Node

__all__ = ["expression_variables", "relation_of"]

#: Derive spelling of relation operators that sympy spells differently.
_RELATIONS = {"==": "=", "!=": "/="}


def expression_variables(
    node: Node, context: Context | None = None, written: bool = False
) -> tuple[str, ...]:
    """The variables `node` holds, most main first.

    Commands need this before prompting: Derive only asks for a variable when there
    are two or more. Assigned variables are excluded, since substitution has already
    replaced them.

    With `written`, an expression that simplifies to one without variables returns the
    variables as written instead, so soLve has something to offer for `x/x = 1`. These
    come from evaluating with sympy's automatic simplification disabled, which still
    excludes bound and assigned variables.
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
    """The relation operator `node` evaluates to, or None if it isn't a relation.

    The operator is in Derive notation: `=`, `/=`, `<`, `<=`, `>`, `>=`. Plot uses
    this to decide what to draw, since `IM(2^(x + #i*y) = (x + #i*y)^6)` is a call in
    the tree but an equation once evaluated, and the equation is what gets drawn.
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
    # `type is` rather than `isinstance`: string literals are a `Symbol` subclass, and
    # aren't variables.
    return [s.name for s in symbols if type(s) is sp.Symbol]
