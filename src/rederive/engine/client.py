"""The half of the engine the app may hold: everything that costs nothing to import.

The engine has two halves and one rule between them.

The worker half is the mathematics - `pipeline`, `to_sympy`, `from_sympy`, `factor`,
`expand`, `solve` and what they stand on. It imports sympy, it runs in the worker,
and `worker.serve` is what imports it there. The app process never does: a client
ships a tree across and a `Result` comes back, and a proxy needs none of the
mathematics it is a proxy for.

The client half is this module, and its rule is that it imports no sympy, directly or
through anything else. That is not tidiness. Sympy is some four hundred modules, a
third of a second and thirty-odd megabytes, and the app paying that buys nothing:
whatever it loaded, the worker loads its own copy where the computing happens. What
is left when it is gone is enough to author, draw, navigate and save a worksheet - so
a session whose engine is down goes on working, which is the property every engine
client is built around.

The rule has a second edge, and it is what the exceptions are doing here. A client
is written against a transport - `remote` spawns a child process and talks down a
pipe - and a port has another one. So the vocabulary a dead worker raises lives here,
where anyone may hold it, rather than beside the machinery of any one transport; the
recovery every death funnels into is in `policy` beside it, for the same reason. What
is left in `remote` is the child process itself, which is the only part a browser
cannot use.

What the client half offers:

* The exceptions a dead worker raises.
* The vocabulary that crosses to the worker, from `boundary`: `Amount` and `Result`.
* The session's own state - `Context` and the settings in it - since a context is
  data the app edits and the worker only reads.
* `replace` and `main_order`, the two things a command asks for that are answered by
  reading the tree rather than by computing with it.
* The questions the menus have to answer before they can put a question of their own.
  Each is syntactic on purpose, which is what makes it answerable here: Derive asks
  for a factoring amount based on how an expression is *written*, not on what it is
  worth, so the answer is a walk over the tree and not a computation.

`tests/test_packages.py` holds the rule to it. Anything imported here that reaches
sympy fails that test, whatever else it does.
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
    """What every engine call can fail with, the worker being mortal."""

    #: Where the dead worker's output went, where there is a file to read.
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

    The worker survives it - one bad answer is not worth a process - so this is
    the one engine failure that costs nothing but the command.
    """

    def __init__(self, message: str, traceback_text: str = "") -> None:
        super().__init__(message)
        self.traceback_text = traceback_text


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


def written_as_ratio(node: Node) -> bool:
    """Whether `node` is written as a quotient, and so has a denominator to factor.

    Which is the question the original asks before it asks for an amount: only
    a ratio can become partial fractions, and only partial fractions need a
    denominator factored. The test is on how the expression is written and not
    on what it is worth, which is the original's own rule and an unforgiving
    one - it asks for an amount for `x^2/x`, whose value has no denominator at
    all, and asks for none for `2/x + 1/x`, whose value does. `x^-1` is a
    power and not a quotient, so it is not asked about either.

    An amount changes nothing where there is no denominator to factor. What it
    changes is how many questions are put.
    """
    return node.kind is Kind.BINOP and node.value == "/"


def equations_in(node: Node) -> int:
    """How many equations `node` is a system of, zero where it is no system.

    Which is the question the UI has to answer before it can ask anything: a
    system of E equations in more than E variables is asked about E times, and
    a scalar equation at most once. Syntactic, like `decomposes`: a system is a
    vector of relations as it was written, and reading that off the tree costs
    nothing.
    """
    if node.kind is not Kind.VECTOR or not node.children:
        return 0
    if not all(child.kind is Kind.REL for child in node.children):
        return 0
    return len(node.children)
