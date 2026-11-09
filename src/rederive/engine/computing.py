"""The half of the engine that computes: the commands, and sympy underneath them.

Importing this loads sympy (some four hundred modules, a third of a second, thirty-odd
megabytes), so only a process meant to compute should do it. The worker imports it in
`worker.serve` once logging and memory limits are set up; tests and direct callers
import it to compute in-process. The app process never does: it holds a `RemoteEngine`
and computes over a pipe, so Esc can kill a computation and a worksheet stays savable
when the worker dies.

This re-exports everything in `client`, so a caller that may compute needs one import.
Which of the two a module imports declares what it may do; there is no third way in,
as `tests/test_packages.py` enforces.

The commands themselves are documented in `rederive.engine`.
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
from rederive.engine.evaluate import evaluated
from rederive.engine.expand import expand
from rederive.engine.factor import factor
from rederive.engine.from_sympy import from_sympy, parse_state_for
from rederive.engine.pipeline import approx, simplify
from rederive.engine.printer import author_text
from rederive.engine.solve import solve, solved
from rederive.engine.to_sympy import authored_conditionals, to_sympy
from rederive.engine.variables import expression_variables, relation_of

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
    "Replacement",
    "Result",
    "TrigPower",
    "approx",
    "author_text",
    "authored_conditionals",
    "decomposes",
    "domain_of_node",
    "equations_in",
    "evaluated",
    "expand",
    "expression_variables",
    "factor",
    "from_sympy",
    "main_order",
    "parse_state_for",
    "relation_of",
    "replace",
    "simplify",
    "solve",
    "solved",
    "to_sympy",
    "written_as_ratio",
]
