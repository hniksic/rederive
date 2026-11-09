"""The app side of the engine worker: engine calls over a pipe to a child process.

`RemoteEngine` implements the engine calls `Session` uses, so a session given one
computes in a child process transparently. Each call sends the tree, context and parse
state over a pipe and blocks for the answer on an `asyncio.to_thread` thread, leaving
the event loop free to repaint, update the elapsed time, and handle Esc. Everything
below the calls is blocking code: a receive loop with deadlines, memory polling
between reads, waiting for the handshake.

The child process exists for Esc. Sympy has no cooperative cancellation, so aborting
means killing the process. That is routine because the worker is stateless, so its
replacement loses nothing.

How deaths are handled, including the three-strikes rule, is decided in `policy`,
shared with browser workers. This module has the process-specific part: spawning,
signaling, and reading a pipe.

The memory cap is enforced by watching from outside, since that is portable:
`RLIMIT_AS` works on Linux, is ignored on macOS, and doesn't exist on Windows. The
watchdog polls the worker's resident set during a request. Where it can't be read (no
psutil, or nothing to read), there is no watchdog, and the cap is whatever the child's
kernel enforces. There is deliberately no time limit: Derive users let computations
run for hours, and Esc is the time limit.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import signal
import threading
import time
from collections.abc import Sequence
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from typing import Any

from rederive import memory
from rederive.engine import worker
from rederive.engine.boundary import Amount, Result
from rederive.engine.client import (
    EngineAborted,
    EngineBug,
    EngineDied,
    EngineDown,
    EngineError,
    EngineMemoryExceeded,
)
from rederive.engine.context import Context
from rederive.engine.policy import Deaths
from rederive.model.expr import Node
from rederive.syntax.state import ParseState

__all__ = [
    "DEFAULT_CAP",
    "EngineAborted",
    "EngineBug",
    "EngineDied",
    "EngineDown",
    "EngineError",
    "EngineMemoryExceeded",
    "RemoteEngine",
    "default_cap",
]

#: The worker's memory cap as a divisor of physical memory, i.e. half. Generous enough
#: for any legitimate computation (`10000!` is a megabyte), yet leaving room for the
#: app, terminal and system.
DEFAULT_CAP_SHARE = 2

#: The cap when physical memory is unknown. Two gigabytes is more than any worksheet
#: needs and less than any desktop has.
DEFAULT_CAP = 2 << 30

#: Seconds between readings of the worker's resident set during a request. Frequent
#: enough to catch a runaway, rare enough to cost nothing.
WATCH_PERIOD = 0.5

#: Seconds to wait for a new worker to finish importing sympy. Generous, since a cold
#: filesystem on a loaded machine is slow, and the wait is abortable anyway.
READY_TIMEOUT = 30.0


def default_cap() -> int:
    """Half of physical memory, or `DEFAULT_CAP` if that is unknown."""
    total = memory.total_bytes()
    return DEFAULT_CAP if total is None else total // DEFAULT_CAP_SHARE


class RemoteEngine:
    """The engine calls, answered by a killable child process.

    The calls are awaitable and everything below them blocks, bridged by
    `asyncio.to_thread`. A thread waits on the pipe, Esc kills the worker, and the
    waiting call raises an exception naming the cause.

    The `_call` lock allows one request at a time, which is all a single-threaded
    session needs. `abort` is called from the event loop while a waiting request
    holds that lock, so it deliberately doesn't take it.
    """

    def __init__(
        self,
        cap: int | None = None,
        period: float = WATCH_PERIOD,
        ready_timeout: float = READY_TIMEOUT,
    ) -> None:
        self.cap = default_cap() if cap is None else cap
        self._period = period
        self._ready_timeout = ready_timeout
        #: Whether the resident set can be read, and so whether to run the watchdog.
        self._watching = memory.measures_processes()
        # Always spawn, never fork: forking a process with a Textual app's threads and
        # event loop is unsafe, and consistent startup everywhere is worth the extra
        # second of imports.
        self._context: BaseContext = multiprocessing.get_context("spawn")
        self._call = threading.Lock()
        self._guard = threading.Lock()
        self._process: BaseProcess | None = None
        self._connection: Connection | None = None
        self._ready = False
        #: Why we killed the worker, if we did. Set before the kill and read by the
        #: death path, to tell an abort from a crash.
        self._reason: EngineError | None = None
        #: The transport-independent death policy (three strikes, down state).
        self._deaths = Deaths()
        #: Number of aborts requested. A call reads it before starting and gives up if
        #: it changes, so Esc pressed while a call is still queued for a worker aborts
        #: it rather than letting it run on the next worker.
        self._aborts = 0
        self._stopped = False
        self._number = 0
        #: Number of workers spawned, used by tests of the respawn policy.
        self.starts = 0

    # -- the engine interface ----------------------------------------------
    #
    # Awaitable calls over blocking machinery. Each runs `_ask` on a thread and awaits
    # the thread, leaving the loop free while the answer is computed.

    async def simplify(
        self, node: Node, context: Context, state: ParseState | None = None
    ) -> Result:
        return await self._await("simplify", (node, context, state))

    async def factor(
        self,
        node: Node,
        context: Context,
        amount: Amount = Amount.RATIONAL,
        variables: Sequence[str] = (),
        state: ParseState | None = None,
    ) -> Result:
        arguments = (node, context, amount, tuple(variables), state)
        return await self._await("factor", arguments)

    async def expand(
        self,
        node: Node,
        context: Context,
        amount: Amount = Amount.RATIONAL,
        variables: Sequence[str] = (),
        state: ParseState | None = None,
    ) -> Result:
        arguments = (node, context, amount, tuple(variables), state)
        return await self._await("expand", arguments)

    async def solve(
        self,
        node: Node,
        context: Context,
        variables: Sequence[str] = (),
        bounds: tuple[Node, Node] | None = None,
        state: ParseState | None = None,
    ) -> tuple[Result, ...]:
        arguments = (node, context, tuple(variables), bounds, state)
        return await self._await("solve", arguments)

    async def expression_variables(
        self, node: Node, context: Context | None = None, written: bool = False
    ) -> tuple[str, ...]:
        return await self._await("expression_variables", (node, context, written))

    async def relation_of(
        self, node: Node, context: Context | None = None
    ) -> str | None:
        return await self._await("relation_of", (node, context))

    async def _await(self, method: str, args: tuple[Any, ...]) -> Any:
        """Run one request on a thread from the loop's pool and await it."""
        return await asyncio.to_thread(self._ask, method, args)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start a worker, letting it import sympy in the background.

        Returns once the process exists; the handshake is awaited in the background,
        so the first command usually finds a worker with its imports done.
        """
        with self._call:
            self._stopped = False
            if self._process is None:
                self._spawn()
                self._warm()

    def abort(self) -> None:
        """Abort the current computation by killing the worker.

        Called from the event loop while a thread is blocked on the pipe. The reason
        is recorded first so the resulting death is reported as an abort, not a crash.
        """
        with self._guard:
            self._aborts += 1
        self._record(EngineAborted("Aborted"))
        self._kill()

    def shutdown(self) -> None:
        """Stop the worker permanently, on app exit."""
        with self._guard:
            self._stopped = True
        self._kill()
        with self._guard:
            process, self._process = self._process, None
        if process is not None:
            _reap(process)
        connection, self._connection = self._connection, None
        _close(connection)
        self._ready = False
        memory.register_worker(None)

    # -- one request -------------------------------------------------------

    def _ask(self, method: str, args: tuple[Any, ...]) -> Any:
        """Send one request and wait for its answer or the worker's death.

        The abort count is read before queueing, so Esc pressed while waiting for the
        lock or for a worker to start aborts the call too.
        """
        aborts = self._aborts
        with self._call:
            self._abandoned(aborts)
            self._require()
            self._abandoned(aborts)
            self._number += 1
            number = self._number
            connection = self._connection
            if connection is None:
                raise self._collapse() from None
            try:
                connection.send((number, method, args))
            except (OSError, ValueError, EOFError):
                raise self._collapse() from None
            while True:
                answer = self._receive()
                # An answer with another number belongs to an abandoned request. This
                # can't happen since aborted workers are always killed, but the check
                # is cheap insurance.
                if answer[0] == number:
                    return self._answered(answer)

    def _abandoned(self, aborts: int) -> None:
        """Raise `EngineAborted` if an abort arrived since this call started."""
        if self._aborts != aborts:
            raise EngineAborted("Aborted")

    def _require(self) -> None:
        """Ensure a ready worker exists, or raise why there isn't one.

        A client that is down retries once here; that is the whole retry policy (see
        `Deaths.retrying`).
        """
        if self._stopped:
            raise EngineDown("The engine has been shut down")
        if self._deaths.down is not None:
            self._deaths.retrying()
            self._spawn()
        elif self._process is None:
            self._spawn()
        self._await_ready()

    def _await_ready(self) -> None:
        """Wait for the worker's handshake after it imports the engine."""
        if self._ready:
            return
        deadline = time.monotonic() + self._ready_timeout
        while True:
            answer = self._receive(deadline)
            if answer[0] == worker.READY:
                self._ready = True
                self._deaths.ready()
                return

    def _receive(self, deadline: float | None = None) -> Any:
        """The next message from the worker, checking its memory while waiting.

        Any other end to the wait is a death, classified by the death path.
        """
        connection = self._connection
        if connection is None:
            raise self._collapse() from None
        while True:
            try:
                if connection.poll(self._period):
                    return connection.recv()
            except (EOFError, OSError, ValueError):
                raise self._collapse() from None
            process = self._process
            if process is None or not process.is_alive():
                raise self._collapse() from None
            if self._watching:
                self._watch(process)
            if deadline is not None and time.monotonic() > deadline:
                self._record(EngineDied("The engine did not start in time"))
                self._kill()

    def _watch(self, process: BaseProcess) -> None:
        """Kill the worker if its resident set exceeds the cap.

        A failed reading is ignored: the worker has probably just died, and the death
        path will handle it. Not called where resident sets can't be read.
        """
        held = memory.process_bytes(process.pid)
        if held is not None and held > self.cap:
            self._record(EngineMemoryExceeded(self._out_of_memory()))
            self._kill()

    def _answered(self, answer: Any) -> Any:
        """Interpret one response: a value, a survivable bug, or a death."""
        if answer[1] == "ok":
            return answer[2]
        _, _, kind, message, traceback_text = answer
        if kind == worker.MEMORY:
            # The worker exits after this, so run the death path now rather than
            # surprising the next command.
            self._record(EngineMemoryExceeded(self._out_of_memory()))
            raise self._collapse()
        raise EngineBug(message, traceback_text)

    def _out_of_memory(self) -> str:
        return f"The engine ran out of memory - the cap is {memory.written(self.cap)}"

    # -- the one death path ------------------------------------------------

    def _collapse(self) -> EngineError:
        """Handle a dead worker: determine why, clean up, and start a replacement.

        Every kind of death comes here. How many failures in a row are too many is
        decided by `policy`. The exception is returned rather than raised so each
        caller can raise it where it suits.
        """
        with self._guard:
            process, self._process = self._process, None
            reason, self._reason = self._reason, None
        connection, self._connection = self._connection, None
        ready, self._ready = self._ready, False
        memory.register_worker(None)
        if process is not None:
            _kill_off(process)
            _reap(process)
        _close(connection)
        error = reason if reason is not None else _death_of(process, ready)
        respawn = self._deaths.died(error, ready)
        if self._stopped or not respawn:
            return error
        try:
            self._spawn()
        except EngineDown:
            return error
        self._warm()
        return error

    # -- spawning ----------------------------------------------------------

    def _spawn(self) -> None:
        """Start a worker. Raises `EngineDown` if the process can't be started."""
        parent = child = None
        try:
            parent, child = self._context.Pipe(duplex=True)
            process = self._context.Process(
                target=worker.serve, args=(child, self.cap), daemon=True
            )
            process.start()
        except (OSError, ValueError) as error:
            _close(parent)
            _close(child)
            down = EngineDown(f"The engine could not be started: {error}")
            self._deaths.down = down
            raise down from None
        # Close our copy of the child's end, or the pipe would stay open past the
        # child's death and every death would become a hang.
        child.close()
        self.starts += 1
        self._connection = parent
        self._ready = False
        with self._guard:
            self._process = process
        memory.register_worker(process.pid)

    def _warm(self) -> None:
        """Wait for the handshake on a background thread, so a death is noticed.

        Otherwise a worker that can't start would go unnoticed until the next
        command, and the three-strikes rule would need three commands to trigger.
        """
        threading.Thread(target=self._warming, daemon=True).start()

    def _warming(self) -> None:
        with self._call:
            if self._stopped or self._connection is None:
                # Shut down or replaced while this thread waited for the lock.
                return
            try:
                self._await_ready()
            except EngineError:
                # The death path has already handled it, and no command is waiting.
                pass

    # -- killing -----------------------------------------------------------

    def _record(self, reason: EngineError) -> None:
        """Record the reason for an imminent kill.

        Only if there is a worker to kill, so Esc pressed as a computation finishes
        can't mislabel the next worker's death.
        """
        with self._guard:
            if self._process is not None and self._reason is None:
                self._reason = reason

    def _kill(self) -> None:
        """Kill the worker now, without waiting (called from the event loop)."""
        with self._guard:
            process = self._process
        if process is not None:
            _kill_off(process)


def _kill_off(process: BaseProcess) -> None:
    """Terminate the process, then kill it if it is still alive shortly after."""
    try:
        if process.is_alive():
            process.terminate()
            process.join(0.2)
        if process.is_alive():
            process.kill()
    except (OSError, ValueError, AssertionError):
        pass


def _reap(process: BaseProcess) -> None:
    try:
        process.join(1.0)
    except (OSError, ValueError, AssertionError):
        pass


def _close(connection: Connection | None) -> None:
    if connection is None:
        return
    try:
        connection.close()
    except OSError:
        pass


def _death_of(process: BaseProcess | None, ready: bool) -> EngineError:
    """The error for a worker that died unintentionally.

    Based on the exit code: a negative one is the signal that ended the process, and
    an unexpected `SIGKILL` is almost always the system's OOM killer.
    """
    log = "" if process is None or process.pid is None else worker.log_path(process.pid)
    code = None if process is None else process.exitcode
    if not ready:
        return EngineDied("The engine would not start", log)
    if code is not None and code == -getattr(signal, "SIGKILL", 9):
        return EngineDied("The engine was killed, most likely for memory", log)
    if code is not None and code < 0:
        return EngineDied(f"The engine crashed with signal {-code}", log)
    return EngineDied("The engine stopped", log)
