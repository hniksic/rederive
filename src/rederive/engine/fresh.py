"""Every command starts from the same state of sympy's random generators and caches.

Sympy draws random numbers in a few places that decide no answer by themselves but can
change its form or how long it takes: the order in which it asks an object's
assumptions, and the points at which it compares two expressions numerically while
folding nested conditionals. Unseeded, the generators start from the operating system's
entropy in each process, so the same command could come out differently from one run
to the next. Seeded at the start of each command, an answer depends only on the
command.

Sympy's caches, and the engine's own small caches of sympy results (see `forgotten`),
are emptied at the start of each command too. They change no answer, but they change
how much work a step does: a step that finds its subresults cached constructs fewer
objects, and the budgets of optional steps are counted in work done (see `budget`).
Emptied, the work a step does depends only on the command, not on what the session
computed before. A costly result a later command may well repeat, an integral, is
remembered across commands (see `remembered`), but a budgeted step only sees what its
own command computed.

So does the first import of a module: sympy imports many of its parts only when a
computation first needs them, and an import constructs the module's constants, which
then sit in the caches for the rest of that command. The parts the engine's commands
reach are imported here, with the engine, so that no command is the first to.
"""

from __future__ import annotations

import functools
import importlib
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from sympy.core import random as sympy_random
from sympy.core.cache import clear_cache

from rederive.engine.budget import budgeted

__all__ = ["afresh", "forgotten", "remembered"]

#: The parts of sympy that the engine's commands import on first use (found by running
#: a sample of commands of every kind), imported with the engine (see the module's
#: docstring).
_PRELOADED = """
    sympy.assumptions.wrapper sympy.codegen.ast sympy.combinatorics
    sympy.concrete.delta sympy.integrals.heurisch sympy.integrals.laplace
    sympy.integrals.manualintegrate sympy.integrals.meijerint sympy.integrals.prde
    sympy.integrals.rde sympy.integrals.risch sympy.integrals.transforms
    sympy.physics.matrices sympy.physics.units sympy.polys.domains.old_fractionfield
    sympy.polys.domains.old_polynomialring sympy.polys.polymatrix sympy.series.formal
    sympy.series.fourier sympy.series.gruntz sympy.sets.handlers.functions
    sympy.sets.handlers.issubset sympy.sets.setexpr sympy.simplify.fu
    sympy.simplify.hyperexpand sympy.solvers.diophantine sympy.solvers.inequalities
    sympy.solvers.ode sympy.solvers.polysys sympy.solvers.recurr sympy.solvers.solveset
    sympy.tensor.tensor
""".split()

for _name in _PRELOADED:
    importlib.import_module(_name)

P = ParamSpec("P")
R = TypeVar("R")

#: The engine's own caches of results that depend on sympy objects, emptied with
#: sympy's at the start of each command (see `forgotten`).
_CACHES: list[Callable[[], None]] = []


def forgotten(clear: Callable[[], None]) -> Callable[[], None]:
    """Register `clear` to empty one of the engine's caches at the start of each
    command; returns it."""
    _CACHES.append(clear)
    return clear


#: The number of commands started, which tells what a `remembered` function computed in
#: the running one.
_commands = 0


def remembered(size: int) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a function to remember its last `size` values across commands, like
    `functools.lru_cache`, for a costly computation a later command may repeat, such as
    an integral.

    Inside a work budget (see `budget`) only a value the running command computed is
    taken from memory, so that the work a budgeted step does doesn't depend on what
    earlier commands computed; anything else is computed again, and remembered anew.
    The arguments must be hashable.
    """

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        memory: OrderedDict[Any, tuple[int, R]] = OrderedDict()

        @functools.wraps(function)
        def remembering(*args: P.args, **kwargs: P.kwargs) -> R:
            key = (args, tuple(sorted(kwargs.items())))
            found = memory.get(key)
            if found is not None and (found[0] == _commands or not budgeted()):
                memory.move_to_end(key)
                return found[1]
            value = function(*args, **kwargs)
            memory[key] = (_commands, value)
            memory.move_to_end(key)
            if len(memory) > size:
                memory.popitem(last=False)
            return value

        remembering.forget = memory.clear  # type: ignore[attr-defined]
        return remembering

    return decorate


def afresh(command: Callable[P, R]) -> Callable[P, R]:
    """Decorate `command` to start sympy's random generators and the caches from the
    same state."""

    @functools.wraps(command)
    def seeded(*args: P.args, **kwargs: P.kwargs) -> R:
        global _commands
        _commands += 1
        sympy_random.seed(0)
        clear_cache()
        for clear in _CACHES:
            clear()
        return command(*args, **kwargs)

    return seeded
