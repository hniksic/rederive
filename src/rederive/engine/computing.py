"""The half of the engine that computes: the commands, and sympy underneath them.

Importing this is asking for sympy - some four hundred modules of it, a third of a
second, thirty-odd megabytes - so importing it is also a claim about which process is
doing it. The worker makes that claim in `worker.serve`, once the log and the memory
limits are in place. Tests and direct callers make it because computing in place is
exactly what they want. The app process must never make it: it holds a `RemoteEngine`
and the computing happens at the other end of a pipe, which is what lets Esc kill a
computation and what keeps a worksheet savable when the mathematics has died.

Everything `client` offers is offered here too, so a caller who may compute writes
one import and not two. The subset is the point of the pair: `client` is what anyone
may hold, and this is what anyone who may compute may hold. Which one a module
imports is its declaration of what it is allowed to do, and the only such
declaration - there is no third way in, and `tests/test_packages.py` is what says so.

The commands themselves - what Simplify promises, what Factor's amounts mean, what
soLve answers with - are documented in `rederive.engine`.
"""

from __future__ import annotations

from rederive.engine.client import (
    ORDER_LIST,
    Amount,
    Angle,
    Branch,
    Context,
    Definition,
    Direction,
    Domain,
    DomainKind,
    EngineAborted,
    EngineBug,
    EngineDied,
    EngineDown,
    EngineError,
    EngineMemoryExceeded,
    Precision,
    RemoteEngine,
    Replacement,
    Result,
    TrigPower,
    decomposes,
    domain_of_node,
    equations_in,
    main_order,
    replace,
    written_as_ratio,
)
from rederive.engine.expand import expand
from rederive.engine.factor import factor
from rederive.engine.from_sympy import from_sympy, parse_state_for
from rederive.engine.pipeline import approx, simplify
from rederive.engine.printer import author_text
from rederive.engine.solve import solve, solved
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
    "EngineAborted",
    "EngineBug",
    "EngineDied",
    "EngineDown",
    "EngineError",
    "EngineMemoryExceeded",
    "Precision",
    "RemoteEngine",
    "Replacement",
    "Result",
    "TrigPower",
    "approx",
    "author_text",
    "decomposes",
    "domain_of_node",
    "equations_in",
    "expand",
    "expression_variables",
    "factor",
    "from_sympy",
    "main_order",
    "parse_state_for",
    "replace",
    "simplify",
    "solve",
    "solved",
    "substitute",
    "to_sympy",
    "written_as_ratio",
]
