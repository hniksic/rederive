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
degrees - with every sum in normal form, reached by transforming as little else
as necessary. It reads only its `Context`: precision, branch, the Trigonometry,
Trigpower, Exponential and Logarithm directions, angle measure, input base, the
variable order list, and the domains, assignments, function definitions and
labels the session has recorded. It has no other state and no side effects, so
the same tree and context always give the same answer.

The normal form is a sum written as a rational function of the most main
variable it holds, which the order list is what decides. So `(x + 1)^9 + y` is
a ninth-degree polynomial in `x`, `(y + 1)^9 + x` is left as it was written,
and `Manage Ordering` changes both answers. Only sums: a product or a power
that is not itself a sum is never distributed, which is why `2*x*(x - 3)^2` and
`(x + 1)*(y + 1)` come back untouched.

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

`expand(node, context, amount, variables)` is Derive's Expand: the same
expression written as a sum. It is Simplify and then expanding, and it makes
the same two promises. `variables` names the expansion variables, empty
meaning all of them, and everything free of them is left alone - which is what
writes `(x + 2*y + 1)^3` about `x` in powers of `2*y + 1`. A ratio whose
denominator holds an expansion variable becomes partial fractions instead, and
`amount` says how far the denominator is factored on the way; Expand offers
four of the five amounts, `Complex` being Factor's alone.

The mathematics of both is a file of its own below the pipeline rather than
above, because `FACTOR(u, amount, x, y, ...)` and `EXPAND(u, amount, x, y,
...)` are an authored line's own way of asking for the same things, and
Simplify is what evaluates those.

`replace(node, replacements, state)` is Derive's Manage Substitute, and it is
the one command that computes nothing: each replacement is a subtree to look
for and what to write in its place, every match is replaced at once, and the
answer is written out unsimplified. It is not `substitute`, which is the
pre-pass every other command runs to write in what a name already stands for.

`expression_variables(node, context)` is what a command offers before it can
ask anything: the variables the expression holds, most main first.

Every command works on any subtree, not only on a whole authored line, so the
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
from rederive.engine.expand import expand, written_as_ratio
from rederive.engine.factor import decomposes, factor
from rederive.engine.factoring import Amount
from rederive.engine.from_sympy import Result, from_sympy, parse_state_for
from rederive.engine.ordering import ORDER_LIST, main_order
from rederive.engine.pipeline import approx, simplify
from rederive.engine.printer import author_text
from rederive.engine.replacing import Replacement, replace
from rederive.engine.substitute import substitute
from rederive.engine.to_sympy import to_sympy
from rederive.engine.variables import expression_variables

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
    "Replacement",
    "Result",
    "TrigPower",
    "approx",
    "author_text",
    "decomposes",
    "domain_of_node",
    "expand",
    "expression_variables",
    "factor",
    "from_sympy",
    "main_order",
    "parse_state_for",
    "replace",
    "simplify",
    "substitute",
    "to_sympy",
    "written_as_ratio",
]
