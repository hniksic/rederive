"""The half of the engine the app may hold: everything that is cheap to import.

The worker half is the mathematics (`pipeline`, `to_sympy`, `from_sympy`, `factor`,
`expand`, `solve` and their dependencies). It imports sympy and runs in the worker,
loaded by `worker.serve`. The app process never imports it: it sends a tree and gets a
`Result` back.

This module must not import sympy, directly or indirectly. Sympy is some four hundred
modules, a third of a second and thirty-odd megabytes, and the worker loads its own copy
anyway. Without it the app can still author, draw, navigate and save a worksheet, so a
session keeps working while the engine is down.

The worker-death exceptions live here rather than in `remote` because other transports
(e.g. a browser port) need them too; recovery lives in `policy` for the same reason.
`remote` keeps only the child-process machinery.

This module offers:

* The exceptions a dead worker raises.
* `Amount` and `Result` from `boundary`.
* `Context` and its settings, which the app edits and the worker only reads.
* `replace` and `main_order`, which read the tree rather than compute with it.
* The questions menus ask before prompting the user. They are syntactic on purpose:
  Derive decides whether to ask for a factoring amount from how an expression is
  *written*, not its value, so a tree walk suffices.

`tests/test_packages.py` fails if anything imported here reaches sympy.
"""

from __future__ import annotations

from rederive.engine.boundary import Amount, Result
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
from rederive.engine.ordering import ORDER_LIST, main_order
from rederive.engine.replacing import Replacement, replace
from rederive.model.expr import Kind, Node

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
    "decomposes",
    "domain_of_node",
    "equations_in",
    "main_order",
    "replace",
    "written_as_ratio",
]


class EngineError(Exception):
    """Base class for failures of an engine call due to the worker."""

    #: Path of the file holding the dead worker's output, if any.
    log: str = ""


class EngineAborted(EngineError):
    """The user pressed Esc and the computation was taken away."""


class EngineMemoryExceeded(EngineError):
    """The worker met the memory cap, from the inside or from the watchdog."""


class EngineDied(EngineError):
    """The worker went away for a reason neither it nor the user chose."""

    def __init__(self, message: str, log: str = "") -> None:
        super().__init__(f"{message} - see {log}" if log else message)
        self.log = log


class EngineDown(EngineError):
    """There is no worker and starting one is not working."""

    def __init__(self, message: str, log: str = "") -> None:
        super().__init__(f"{message} - see {log}" if log else message)
        self.log = log


class EngineBug(EngineError):
    """The engine raised, which it promises never to do on parser output.

    The worker survives this, so only the command is lost.
    """

    def __init__(self, message: str, traceback_text: str = "") -> None:
        super().__init__(message)
        self.traceback_text = traceback_text


def decomposes(node: Node) -> bool:
    """Whether `node` is written as a rational number.

    Derive doesn't ask for a factoring amount for a number, since it is decomposed into
    primes regardless. The test is on how the expression is written, not its value:
    `1234567890/49` qualifies, but `10!` doesn't, and Derive asks for an amount for it.
    """
    if node.kind is Kind.NUMBER:
        return True
    if node.kind is Kind.UNOP and node.value in ("-", "+"):
        return all(decomposes(child) for child in node.children)
    if node.kind is Kind.BINOP and node.value == "/":
        return all(decomposes(child) for child in node.children)
    return False


def written_as_ratio(node: Node) -> bool:
    """Whether `node` is written as a quotient, and so has a denominator to factor.

    Expand asks for an amount only in this case, since only a ratio becomes partial
    fractions. Like Derive, the test is purely syntactic: it asks for `x^2/x`, whose
    value has no denominator, but not for `2/x + 1/x`, whose value does. `x^-1` is a
    power, not a quotient.
    """
    return node.kind is Kind.BINOP and node.value == "/"


def equations_in(node: Node) -> int:
    """How many equations `node` is a system of, or zero if it isn't one.

    The UI needs this before prompting: a system of E equations in more than E variables
    asks for variables E times, a scalar equation at most once. Like `decomposes` it is
    syntactic: a system is a vector of relations or expressions as written.
    """
    if node.kind is not Kind.VECTOR or not node.children:
        return 0
    if any(child.kind in _NO_EQUATION for child in node.children):
        return 0
    return len(node.children)


#: Vector element kinds that can't be an equation of a system. Any other element is a
#: relation, or an expression `u` read as `u = 0` by `equations_of`.
_NO_EQUATION = frozenset(
    {
        Kind.VECTOR,
        Kind.STRING,
        Kind.NOT,
        Kind.AND,
        Kind.OR,
        Kind.XOR,
        Kind.IMP,
        Kind.ASSIGN,
        Kind.FUNDEF,
        Kind.DOMAIN,
    }
)
