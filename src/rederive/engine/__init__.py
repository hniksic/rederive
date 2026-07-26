"""The math engine: expression trees in, expression trees out.

Nothing outside this package may import from inside it except through the
names re-exported here, and nothing inside it may import from the UI. What
crosses the boundary is a `Node` tree and a `Context`; what comes back is a
`Result` carrying both the answer's text and its reparsed tree.

Two doors, shared by every command the engine will ever grow:

* `to_sympy` translates a tree into sympy faithfully. It maps each node to the
  object that means the same thing and simplifies nothing beyond sympy's own
  automatic evaluation. What it reads is the context: the precision mode, the
  angle mode, the input base, and the domains that say what a variable is.
* `from_sympy` writes an expression back as author notation and reparses it,
  which is how any result becomes a worksheet entry.

The mapping is total in both directions. Every node kind converts, and
anything the engine has no mathematics for becomes an inert head that carries
its operands, survives untouched and prints back to the notation it came from.
A construct sympy will not take - a call with arguments it cannot use, a
product of shapes it cannot multiply - becomes such a head as well, rather
than a guess at what was meant.

Commands are built on top of these two doors and the converters know nothing
about them. The first of them is Simplify.

`simplify(node, context)` promises Derive's Simplify: the sufficiently simple
form of an expression - no superfluous variables, roots, functions or reducible
degrees - reached by transforming as little as necessary. It reads only its
`Context`: precision, branch, the Trigonometry, Trigpower, Exponential and
Logarithm directions, angle measure, input base, and the domains, assignments,
function definitions and labels the session has recorded. It has no other state
and no side effects, so the same tree and context always give the same answer.

Two promises hold across every input:

* It never raises on anything the parser produced. A rewrite that fails is a
  rewrite not taken, and the previous form stands.
* It never guesses. A transformation that needs a variable to be real, or
  positive, or an integer, fires only where a declaration says so; where
  nothing says so the expression comes back as it went in. An undeclared
  variable is real, which is Derive's own default.

`approx` is the same pipeline with the precision mode set to Approximate, which
is what the manual says the approX command is.

`factor(node, context, amount, variables)` is Derive's Factor: the same
expression written as a product. It is Simplify and then factoring, which is
what the manual means by saying both commands reach a sufficiently simple form
and Factor goes further, and it makes the same two promises. `Amount` is how
far it goes - Trivial, Squarefree, Rational, raDical or Complex, each doing
everything the one before it does - and `variables` names the factorization
variables, empty meaning all of them.

The mathematics of it is a file of its own below the pipeline rather than
above, because `FACTOR(u, amount, x, y, ...)` is an authored line's own way of
asking for the same thing, and Simplify is what evaluates that.

Both commands work on any subtree, not only on a whole authored line, so the
session can act on what the user has highlighted and put the answer back.
"""

from __future__ import annotations

from rederive.engine.context import (
    Angle,
    Branch,
    Context,
    Definition,
    Direction,
    Domain,
    DomainKind,
    Precision,
    TrigPower,
    domain_of_node,
)
from rederive.engine.factor import decomposes, factor, factor_variables
from rederive.engine.factoring import Amount
from rederive.engine.from_sympy import Result, from_sympy, parse_state_for
from rederive.engine.ordering import ORDER_LIST, main_order
from rederive.engine.pipeline import approx, simplify
from rederive.engine.printer import author_text
from rederive.engine.substitute import substitute
from rederive.engine.to_sympy import to_sympy

__all__ = [
    "ORDER_LIST",
    "Amount",
    "Angle",
    "Branch",
    "Context",
    "Definition",
    "Direction",
    "Domain",
    "DomainKind",
    "Precision",
    "Result",
    "TrigPower",
    "approx",
    "author_text",
    "decomposes",
    "domain_of_node",
    "factor",
    "factor_variables",
    "from_sympy",
    "main_order",
    "parse_state_for",
    "simplify",
    "substitute",
    "to_sympy",
]
