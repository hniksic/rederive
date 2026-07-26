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
about them. The first of them, Simplify, is not here yet: its pipeline and the
substitution pre-pass it shares with every later command are the next
increment.
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
from rederive.engine.from_sympy import Result, from_sympy, parse_state_for
from rederive.engine.printer import author_text
from rederive.engine.to_sympy import to_sympy

__all__ = [
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
    "author_text",
    "domain_of_node",
    "from_sympy",
    "parse_state_for",
    "to_sympy",
]
