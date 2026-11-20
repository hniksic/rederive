"""Which variables an expression holds, and whether it is a relation.

Variables are listed most main first, per `ordering`, which is the order Factor and
Expand offer them in: `x^2 - a^2` lists `x,a`, not `a,x`.

Both questions are answered from the evaluated expression rather than the tree, since
the tree may not show its meaning: `#3` is a label, and `IM(u = v)` is a function call
that evaluates to an equation. The value is taken as evaluated, before an undecided
conditional in it is written out for display: writing out one nested in many others
(an `ITERATE` over an `IF`) is the costly part, and tells nothing about the variables.

A name a program assigns is no variable where the program stands unrun: in `IF(x < 0,
PROG(a_ := -x, a_), x^2)` only `x` is.
"""

from __future__ import annotations

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational

from rederive.engine.context import Context
from rederive.engine.evaluate import Environment, evaluate, value_of
from rederive.engine.ordering import main_order
from rederive.engine.vectors import Vector
from rederive.model.expr import Node

__all__ = [
    "evaluated_names",
    "expression_variables",
    "relation_in",
    "relation_of",
    "variables_in",
]

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
        value, inert = evaluated_names(node, context)
        names = variables_in(value, inert)
        if written and not names:
            with sp.evaluate(False):
                value, inert = evaluated_names(node, context)
            names = variables_in(value, inert)
    except Exception:
        return ()
    return main_order(names, context.order)


def relation_of(node: Node, context: Context | None = None) -> str | None:
    """The relation operator `node` evaluates to, or None if it isn't a relation.

    The operator is in Derive notation: `=`, `/=`, `<`, `<=`, `>`, `>=`. Plot uses
    this to decide what to draw, since `IM(2^(x + #i*y) = (x + #i*y)^6)` is a call in
    the tree but an equation once evaluated, and the equation is what gets drawn.
    """
    try:
        value, _ = evaluated_names(node, context or Context())
    except Exception:
        return None
    return relation_in(value)


def relation_in(value: sp.Basic) -> str | None:
    """The relation operator of `value`, in Derive notation, or None."""
    if not isinstance(value, Relational):
        return None
    return _RELATIONS.get(value.rel_op, value.rel_op)


def evaluated_names(node: Node, context: Context) -> tuple[sp.Basic, frozenset[str]]:
    """`node` evaluated where nothing runs, as it stands before display, and the names
    the programs standing in it assign (see `Store.inert`)."""
    context = context.fresh()
    value = value_of(evaluate(node, Environment(context)))
    return value, frozenset(context.store.inert)


def variables_in(value: sp.Basic, inert: frozenset[str] = frozenset()) -> list[str]:
    """The names of the variables in `value`, leaving out the names in `inert`."""
    return _variables(_free(value), inert)


def _free(value: sp.Basic) -> set[sp.Basic]:
    """The free symbols of `value`, where the variable of a summation, generated vector
    or other binding call that stands is bound in its body, as sympy's own `Sum` binds
    its index: `PRODUCT(1 - 1/v_ SUB 1, v_, FACTORS(n))` holds only `n`."""
    standing = [
        found
        for found in value.atoms(AppliedUndef)
        if found.func.__name__ in _BINDING and len(found.args) >= 2
    ]
    if not standing:
        return value.free_symbols
    return _unbound(value)


def _unbound(value: sp.Basic) -> set[sp.Basic]:
    """`_free`, by a walk."""
    if isinstance(value, AppliedUndef) and value.func.__name__ in _BINDING:
        body, index, *rest = value.args
        bound = {getattr(symbol, "name", None) for symbol in _indexes(index)}
        inside = {
            symbol
            for symbol in _unbound(body)
            if getattr(symbol, "name", "").split(" SUB ")[0] not in bound
        }
        return inside.union(*(_unbound(argument) for argument in rest))
    if not value.args:
        return set(value.free_symbols)
    if not value.atoms(AppliedUndef):
        return set(value.free_symbols)
    return set().union(*(_unbound(argument) for argument in value.args))


def _indexes(index: sp.Basic) -> tuple[sp.Basic, ...]:
    """The variables a binding call's index names: one, or a vector of them."""
    return tuple(index.args) if isinstance(index, Vector) else (index,)


#: The binding built-ins whose result no longer depends on their variable. (`DIF`,
#: `TAYLOR` and an indefinite `INT` do, and leave it free.)
_BINDING = frozenset(
    "EVERY ITERATE ITERATES MAP MAP_LIST PRODUCT SELECT SOME SUM TABLE VECTOR".split()
)


def _variables(symbols: set[sp.Basic], inert: frozenset[str] = frozenset()) -> list[str]:
    """The names of the symbols that are variables."""
    # `type is` rather than `isinstance`: string literals are a `Symbol` subclass, and
    # aren't variables.
    return [s.name for s in symbols if type(s) is sp.Symbol and s.name not in inert]
