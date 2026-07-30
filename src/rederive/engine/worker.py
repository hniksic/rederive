"""The child process the engine runs in: requests down a pipe, answers back.

The worker holds no session state. Every request carries the whole of what the
answer may depend on - the tree, the `Context`, the `ParseState` - and the
worker keeps nothing between requests but warm imports and sympy's own cache.
That is what makes killing it a recovery rather than a loss: a fresh worker is
indistinguishable from the one that died, so Esc can abort a computation by
taking the process out from under it, and every other death is the same thing
happening for a different reason.

Statelessness is a contract rather than an accident. Should the worker ever
grow per-session state for speed, it may only ever be a content-keyed cache the
protocol treats as maybe-empty - the parent re-sends what the worker says it
does not know - never the truth about a session.

The tty belongs to the app. Nothing the worker writes may land on it, so
stdout and stderr go to a log file named after the process, which is where a
traceback the parent could not carry back is to be read.

Two private methods sit in the table beside the six public ones, so that the
parent's abort and memory tests do not have to find a real computation slow
enough or hungry enough to test with.
"""

from __future__ import annotations

import os
import signal
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any

__all__ = [
    "BUG",
    "MEMORY",
    "READY",
    "log_path",
    "serve",
]

#: The id the ready handshake carries. No request wears it, so the parent can
#: tell the handshake from an answer without a second channel.
READY = -1

#: The two kinds of error an answer can name. A bug is an exception the engine
#: promised not to raise, and the worker survives it; memory is a `MemoryError`,
#: after which the worker exits, a heap that has just failed a huge allocation
#: not being worth trusting.
BUG = "bug"
MEMORY = "memory"

#: How the log file is named, the pid being what the parent knows about a
#: worker it cannot otherwise ask.
_LOG_NAME = "rederive-worker-{pid}.log"

#: The log file, held open for as long as the worker runs.
_log: Any = None

#: What `allocate` is holding, so that the heap keeps it while it hangs.
_held: Any = None


def log_path(pid: int) -> str:
    """Where the worker with this pid writes what it would have printed."""
    return os.path.join(tempfile.gettempdir(), _LOG_NAME.format(pid=pid))


def serve(connection: Connection, cap: int | None) -> None:
    """Be the engine, for as long as the parent keeps asking.

    The order of the startup steps is the point of it: the tty is given up
    before anything can print to it, SIGINT is disowned before the parent's
    terminal can deliver one, the limits are in place before the engine is
    imported, and the handshake goes out only once the import has cost what it
    costs - so a parent that has heard `ready` knows the next request pays for
    nothing but its own answer.
    """
    _take_the_log(os.getpid())
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _apply_limits(cap)
    methods = _methods()
    connection.send((READY, "ready"))
    _loop(connection, methods)


def _loop(connection: Connection, methods: dict[str, Callable[..., Any]]) -> None:
    """One request at a time, until the parent hangs up or the heap gives out."""
    while True:
        try:
            request = connection.recv()
        except (EOFError, OSError):
            return
        number, method, args = request
        try:
            connection.send((number, "ok", methods[method](*args)))
        except MemoryError:
            # Answer first, exit second: the parent wants to know which of the
            # several ways of running out of memory this was, and the pipe is
            # the only way to tell it.
            _report(connection, number, MEMORY)
            sys.exit(1)
        except Exception:
            # Including `RecursionError`, which costs the answer and not the
            # process: the engine promises never to raise on parser output, so
            # anything arriving here is a bug worth surfacing and not worth a
            # respawn.
            _report(connection, number, BUG)


def _report(connection: Connection, number: int, kind: str) -> None:
    """Send what went wrong, and write the traceback where it can be read.

    The parent gets the traceback text as well as the message, since a bug in
    the engine is worth more than its last line; the log keeps a copy for the
    deaths where there is no pipe left to carry one.
    """
    text = traceback.format_exc()
    print(text, file=sys.stderr)
    try:
        connection.send((number, "error", kind, _message(), text))
    except (OSError, ValueError):
        pass


def _message() -> str:
    """The current exception in one line, as a message line can show it."""
    kind, value, _ = sys.exc_info()
    name = "error" if kind is None else kind.__name__
    return f"{name}: {value}" if str(value) else name


def _take_the_log(pid: int) -> None:
    """Point both output streams at this worker's own log file.

    The descriptors are redirected rather than the Python objects, so that a
    C extension writing to fd 2 lands there too. Anything that will not
    redirect is left alone: a worker without a log is worth more than no
    worker.
    """
    global _log
    try:
        _log = open(log_path(pid), "a", buffering=1, encoding="utf-8", errors="replace")
        os.dup2(_log.fileno(), 1)
        os.dup2(_log.fileno(), 2)
    except OSError:
        _log = None


def _apply_limits(cap: int | None) -> None:
    """Ask the kernel to enforce the memory cap, where it will.

    Belt to the parent's braces: the watchdog polls, and a single enormous
    allocation can be made and filled between two polls. `RLIMIT_AS` catches
    that one on Linux; macOS ignores it and Windows has nothing like it, which
    is why the watchdog and not this is the enforcement. Core dumps are turned
    off because a worker that segfaults holding a gigabyte of bignums should
    not write a gigabyte to disk.
    """
    try:
        import resource
    except ImportError:
        return
    _limit(resource, "RLIMIT_CORE", 0)
    if cap is not None and sys.platform.startswith("linux"):
        _limit(resource, "RLIMIT_AS", cap)


def _limit(resource: Any, name: str, value: int) -> None:
    """Set one soft limit, where the platform has it and will take it."""
    which = getattr(resource, name, None)
    if which is None:
        return
    try:
        _, hard = resource.getrlimit(which)
        if hard != resource.RLIM_INFINITY:
            value = min(value, hard)
        resource.setrlimit(which, (value, hard))
    except (ValueError, OSError):
        pass


def _methods() -> dict[str, Callable[..., Any]]:
    """The table of what a request may ask for.

    The engine is imported here rather than at module scope: importing sympy is
    the second-costliest thing the worker ever does, and it must happen after
    the log and the limits are in place. `computing` rather than `client` is the
    worker saying which half of the engine it is: it is the half that computes,
    and it is the only process that may be.
    """
    from rederive.engine import computing

    return {
        "simplify": computing.simplify,
        "approx": computing.approx,
        "factor": computing.factor,
        "expand": computing.expand,
        "solve": computing.solve,
        "expression_variables": computing.expression_variables,
        "hang": _hang,
        "allocate": _allocate,
    }


def _hang() -> None:
    """Block forever, so that an abort has something certain to abort."""
    while True:
        time.sleep(3600)


def _allocate(megabytes: int) -> None:
    """Take that much memory and hold it, so the cap has something to catch."""
    global _held
    _held = [bytearray(1 << 20) for _ in range(megabytes)]
    _hang()
