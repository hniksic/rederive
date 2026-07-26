"""Factor: the command, on top of `factoring` and the Simplify pipeline.

Derive's Factor is Simplify plus factoring, not factoring on its own, and this
file is that sum: `pipeline` runs first and `factoring` runs on what it
produced. It is what the manual means by saying Factor and Simplify both reach
a sufficiently simple form and Factor goes further, and it means an answer the
original prints may be Simplify's work rather than `factoring`'s - Derive
factors `SIN(x)^2 - 1` to `-COS(x)^2`, and it does that because its Simplify
answers `-COS(x)^2` too.

The order of the two is the whole of what this file decides, and it is not the
obvious one. The pipeline runs at Exact precision whatever the session is set
to, the factoring runs next, and the rounding runs last - because radical
factoring has to reach `SQRT(2)` before Approximate mode has anything to show
as `1.41421`. Rounding first would leave a float with nothing to factor.

`factor` works on any subtree, not only on a whole authored line, so the
session can factor what the user has highlighted and put the answer back.
"""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp

from rederive.engine.context import Context, Precision
from rederive.engine.factoring import DEFAULT_AMOUNT, Amount, factored_expression
from rederive.engine.from_sympy import Result, from_sympy
from rederive.engine.ordering import main_order
from rederive.engine.pipeline import approximated, simplified
from rederive.engine.substitute import substitute
from rederive.engine.to_sympy import to_sympy
from rederive.model.expr import Kind, Node
from rederive.syntax.state import ParseState

__all__ = ["decomposes", "factor", "factor_variables", "factored"]


def factor(
    node: Node,
    context: Context | None = None,
    amount: Amount = DEFAULT_AMOUNT,
    variables: Sequence[str] = (),
    state: ParseState | None = None,
) -> Result:
    """Derive's Factor: `node` written as a product.

    `variables` names the factorization variables in the order they were
    chosen, empty meaning all of them. `state` is the symbol table the answer
    is reparsed with; a session working in a non-default input or case mode
    must pass its own.
    """
    context = context or Context()
    return from_sympy(factored(node, context, amount, variables), context, state)


def factored(
    node: Node,
    context: Context | None = None,
    amount: Amount = DEFAULT_AMOUNT,
    variables: Sequence[str] = (),
) -> sp.Basic:
    """The factored expression, before it is written back out.

    The sympy-level entry point, as `simplified` is for Simplify: a later
    command that wants to keep computing has no reason to print and reparse in
    between.
    """
    context = context or Context()
    expression = simplified(node, context.with_precision(Precision.EXACT))
    return factored_expression(
        expression, amount, variables, lambda e: approximated(e, context)
    )


def factor_variables(node: Node, context: Context | None = None) -> tuple[str, ...]:
    """The variables Factor would offer for `node`, in the order it offers them.

    Most main first, which is what the original lists: `y^2 - x^2` offers `x,y`
    and `z^2 - a^2` offers `z,a`. What an assignment has given a value is not
    among them, since substitution has already replaced it by the time there is
    anything to factor.

    The command needs this before it can ask anything, because whether it asks
    at all depends on the answer: one variable is not a choice, so the original
    puts up no prompt unless there are two or more.
    """
    context = context or Context()
    try:
        expression = to_sympy(substitute(node, context), context)
        symbols = expression.free_symbols
    except Exception:
        return ()
    # `type is` rather than `isinstance`: a string literal is a `Symbol`
    # subclass, and a string is data rather than a variable to factor about.
    return main_order(s.name for s in symbols if type(s) is sp.Symbol)


def decomposes(node: Node) -> bool:
    """Whether `node` is written as a rational number, and nothing else.

    Which is the question the original asks before it asks anything else: a
    number has a prime decomposition and no amount to choose, so the amount
    menu never comes up for one. The test is on how the expression is written
    rather than on what it is worth. `1234567890/49` is a rational number, and
    `10!` is a factorial that has one for a value - the original asks for an
    amount for the second and not the first.

    An amount would change nothing either way, since a number is decomposed
    whatever was asked for. What it changes is how many questions are put.
    """
    if node.kind is Kind.NUMBER:
        return True
    if node.kind is Kind.UNOP and node.value in ("-", "+"):
        return all(decomposes(child) for child in node.children)
    if node.kind is Kind.BINOP and node.value == "/":
        return all(decomposes(child) for child in node.children)
    return False
