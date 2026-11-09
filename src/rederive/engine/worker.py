"""The child process the engine runs in: requests come in over a pipe, answers go back.

The worker holds no session state. Each request carries everything the answer depends
on (the tree, the `Context`, the `ParseState`), and the worker keeps only warm imports
and sympy's cache between requests. A fresh worker is therefore indistinguishable from
a dead one, so Esc can abort by killing the process, and any other death is recovered
the same way.

This is a contract. Any per-session state added for speed must be a content-keyed cache
the protocol treats as possibly empty (the parent re-sends what the worker lacks),
never the authoritative session state.

The tty belongs to the app, so both descriptors are pointed at the null device before
anything can print. The per-process log file, which holds tracebacks the parent
couldn't receive, is opened on first write: most workers are killed without writing
anything, and opening eagerly would leave an empty file per worker.

The table also has two private methods so the parent's abort and memory tests needn't
find a real computation slow or hungry enough.

`methods` is public because a browser worker serves the same table over `postMessage`.
The rest of this module (pipe, tty, limits) is specific to a child process; the
connection type is imported only for annotations, so runtimes without
`_multiprocessing` can still import the shared parts.
"""

from __future__ import annotations

import io
import os
import signal
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

__all__ = [
    "BUG",
    "MEMORY",
    "READY",
    "log_path",
    "methods",
    "serve",
]

#: The id of the ready handshake. No request uses it, so the parent can tell the
#: handshake from an answer.
READY = -1

#: The two kinds of error an answer can report. A bug is an exception the engine
#: promised not to raise, and the worker survives it. Memory is a `MemoryError`, after
#: which the worker exits, since the heap can't be trusted after a failed allocation.
BUG = "bug"
MEMORY = "memory"

#: Log file name, keyed by pid so the parent can find it without asking the worker.
_LOG_NAME = "rederive-worker-{pid}.log"

#: The replacement for both output streams once they are off the tty. The file
#: behind it is created on first write, if any.
_log: Any = None

#: Memory held by `_allocate` so it isn't freed while it hangs.
_held: Any = None


def log_path(pid: int) -> str:
    """Path of the log file for the worker with this pid."""
    return os.path.join(tempfile.gettempdir(), _LOG_NAME.format(pid=pid))


def serve(connection: Connection, cap: int | None) -> None:
    """Serve engine requests until the parent hangs up.

    The startup order matters: release the tty before anything can print, ignore
    SIGINT before the terminal can deliver one, set limits before importing the
    engine, and send the handshake only after the import, so the next request pays
    only for its own answer.
    """
    _take_the_log(os.getpid())
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _apply_limits(cap)
    table = methods()
    connection.send((READY, "ready"))
    _loop(connection, table)


def _loop(connection: Connection, methods: dict[str, Callable[..., Any]]) -> None:
    """Handle requests one at a time until the parent hangs up or memory runs out."""
    while True:
        try:
            request = connection.recv()
        except (EOFError, OSError):
            return
        number, method, args = request
        try:
            connection.send((number, "ok", methods[method](*args)))
        except MemoryError:
            # Report before exiting, so the parent knows how memory ran out.
            _report(connection, number, MEMORY)
            sys.exit(1)
        except Exception:
            # Including `RecursionError`. The engine promises not to raise on parser
            # output, so this is a bug to report, but not worth a respawn.
            _report(connection, number, BUG)


def _report(connection: Connection, number: int, kind: str) -> None:
    """Send the error and its traceback to the parent, and log the traceback.

    The log copy covers deaths where the pipe is gone.
    """
    text = traceback.format_exc()
    print(text, file=sys.stderr)
    try:
        connection.send((number, "error", kind, _message(), text))
    except (OSError, ValueError):
        pass


def _message() -> str:
    """The current exception as a one-line message."""
    kind, value, _ = sys.exc_info()
    name = "error" if kind is None else kind.__name__
    return f"{name}: {value}" if str(value) else name


class _LazyLog(io.TextIOBase):
    """An output stream that creates the log file on first write.

    Most workers write nothing, and should leave no file. Creating the file also points
    fds 1 and 2 at it, so C extension output lands there from then on; C extension
    output before the first Python write goes to the null device.

    Every write is flushed, since most workers die by signal and buffered output would
    be lost.
    """

    def __init__(self, pid: int) -> None:
        self._pid = pid
        self._file: Any = None
        self._failed = False

    def write(self, text: str) -> int:
        """Write to the log, opening it on first use."""
        file = self._materialize()
        if file is None:
            return len(text)
        try:
            written = file.write(text)
            file.flush()
        except (OSError, ValueError):
            return len(text)
        return written

    def flush(self) -> None:
        if self._file is not None:
            try:
                self._file.flush()
            except (OSError, ValueError):
                pass

    def fileno(self) -> int:
        """The log's file descriptor, creating the log if needed."""
        file = self._materialize()
        if file is None:
            raise OSError("worker log unavailable")
        return int(file.fileno())

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def _materialize(self) -> Any:
        """The open log file, opening it and redirecting fds 1 and 2 if needed.

        If the file won't open, the failure is remembered and later writes are
        discarded rather than retried.
        """
        if self._file is not None or self._failed:
            return self._file
        try:
            self._file = open(
                log_path(self._pid), "a", buffering=1, encoding="utf-8", errors="replace"
            )
        except OSError:
            self._failed = True
            return None
        try:
            os.dup2(self._file.fileno(), 1)
            os.dup2(self._file.fileno(), 2)
        except OSError:
            pass
        return self._file


def _take_the_log(pid: int) -> None:
    """Detach both output streams from the tty, logging lazily if needed.

    The inherited descriptors point at the app's screen and must be redirected before
    the first import can print, long before it is known whether a log is needed. So
    they go to the null device, and `_LazyLog` opens the file on demand.

    Redirection failures are ignored: a worker without a log beats no worker.
    """
    global _log
    try:
        null = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        return
    try:
        os.dup2(null, 1)
        os.dup2(null, 2)
    except OSError:
        pass
    finally:
        os.close(null)
    _log = _LazyLog(pid)
    sys.stdout = _log
    sys.stderr = _log


def _apply_limits(cap: int | None) -> None:
    """Ask the kernel to enforce the memory cap, where supported.

    This backs up the parent's polling watchdog, which can miss a huge allocation made
    between two polls. `RLIMIT_AS` catches that on Linux; macOS ignores it and Windows
    lacks it, so the watchdog remains the primary enforcement. Core dumps are disabled
    so a segfaulting worker holding a gigabyte of bignums doesn't write it to disk.
    """
    try:
        import resource
    except ImportError:
        return
    _limit(resource, "RLIMIT_CORE", 0)
    if cap is not None and sys.platform.startswith("linux"):
        _limit(resource, "RLIMIT_AS", cap)


def _limit(resource: Any, name: str, value: int) -> None:
    """Set one soft limit, if the platform supports and accepts it."""
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


def methods() -> dict[str, Callable[..., Any]]:
    """The table of methods a request may call.

    The engine is imported here rather than at module scope because importing sympy is
    expensive and must happen after the log and limits are set up. The worker imports
    `computing` rather than `client` because it is the process that computes.
    """
    from rederive.engine import computing

    return {
        "simplify": computing.simplify,
        "factor": computing.factor,
        "expand": computing.expand,
        "solve": computing.solve,
        "expression_variables": computing.expression_variables,
        "relation_of": computing.relation_of,
        "hang": _hang,
        "allocate": _allocate,
    }


def _hang() -> None:
    """Block forever, giving abort tests something to abort."""
    while True:
        time.sleep(3600)


def _allocate(megabytes: int) -> None:
    """Allocate and hold `megabytes` of memory, for testing the memory cap."""
    global _held
    _held = [bytearray(1 << 20) for _ in range(megabytes)]
    _hang()
