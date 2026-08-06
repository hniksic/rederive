"""The app side of the engine worker: six calls, one pipe, one child process.

`RemoteEngine` answers the six calls `Session` makes the engine by, so a session
handed one computes in a child process without knowing it. Each call ships the
tree, the context and the parse state down a pipe and blocks on the answer, on a
thread `asyncio.to_thread` lends it: the event loop stays free to repaint, to tick
the elapsed time, and to hear Esc. Everything under those six calls is written in
the blocking style that thread is for - a pull-based receive loop with deadlines,
a resident set polled between reads, a handshake waited out.

Esc is the whole reason for the child. Sympy cannot be politely interrupted
mid-bignum - there is no cooperative cancellation to ask for - so an abort is
the process being taken away from under the computation. That makes killing the
worker an ordinary operation rather than a catastrophe, and it is only ordinary
because the worker is stateless: the replacement knows everything the dead one
knew, which is nothing.

What every death then comes to is not this module's to decide. It is one path for
all of them, it is the same path for a worker that is a browser tab's rather than a
process, and it is `policy` - the three-strikes rule included. What is here is the
half that cannot travel: spawning, signalling, and reading a pipe.

The memory cap is enforced from here rather than in the child, because the
portable answer is to watch from outside. `RLIMIT_AS` is enforced on Linux, is
famously not enforced on macOS, and does not exist on Windows; the watchdog
polls the worker's resident set while a request is in flight and works
everywhere a resident set can be read. Where none can - an environment with no
psutil, and any environment with no such thing to read - there is no watchdog at
all rather than one polling for a figure that will never come, and the cap is
whatever the child's own kernel will enforce. There is deliberately no limit on
time: Derive users let computations run for hours, and Esc is the time limit.
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

#: How much of the machine the worker may hold before it is taken away: half of
#: physical memory. Generous enough that no legitimate computation meets it -
#: `10000!` is a megabyte - and small enough that the app, the terminal and the
#: rest of the system are still there when it does.
DEFAULT_CAP_SHARE = 2

#: The cap on a machine that will not say how much memory it has. Two
#: gigabytes is more than any worksheet needs and less than any desktop has.
DEFAULT_CAP = 2 << 30

#: Seconds between readings of the worker's resident set while a request is in
#: flight. Often enough to catch a runaway before it hurts, rare enough that
#: the reading costs nothing next to the computation it is watching.
WATCH_PERIOD = 0.5

#: How long to wait for a freshly spawned worker to finish importing sympy.
#: Generous: a cold filesystem on a loaded machine is slow, and the wait is
#: abortable anyway.
READY_TIMEOUT = 30.0


def default_cap() -> int:
    """Half of physical memory, or a fixed figure where the machine will not say."""
    total = memory.total_bytes()
    return DEFAULT_CAP if total is None else total // DEFAULT_CAP_SHARE


class RemoteEngine:
    """The six engine calls, answered by a child process that can be killed.

    The calls are awaitable and everything under them blocks, `asyncio.to_thread`
    being what joins the two. That is the desktop reading of the session's async
    engine seam, and it is today's semantics to the letter: a thread waits on the
    pipe, Esc kills the worker, and the waiting call raises the exception naming
    the cause.

    One request is in flight at a time, which is what the lock is for: the
    session is single-threaded and the worker holds nothing between requests,
    so there is nothing a second concurrent request could be about. `abort`
    is the exception - it is called from the event loop while the lock is held
    by whoever is waiting, and it deliberately takes no lock of its own, since
    a busy worker's cooperation is exactly what it cannot have.
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
        #: Whether there is a resident set to read, and so a watchdog to run.
        #: Asked once: an environment does not grow the ability while it runs.
        self._watching = memory.measures_processes()
        # Spawn on every platform, never fork: forking a process with a Textual
        # app's threads and event loop in it is not safe, and one start-up
        # behaviour everywhere is worth the second of import it costs.
        self._context: BaseContext = multiprocessing.get_context("spawn")
        self._call = threading.Lock()
        self._guard = threading.Lock()
        self._process: BaseProcess | None = None
        self._connection: Connection | None = None
        self._ready = False
        #: Why we did the killing, when we did. Set before the kill and read by
        #: the death path, which is what tells an abort from a segfault.
        self._reason: EngineError | None = None
        #: The three-strikes rule and what put the proxy down, which are the
        #: parts of dying that have nothing to do with a child process.
        self._deaths = Deaths()
        #: How many aborts have been asked for. A call reads it before it
        #: starts and gives up if it has moved: an Esc pressed while the call
        #: is still queueing for a worker is as much an abort as one pressed
        #: while sympy is grinding, and the user must not find the command run
        #: anyway on the worker that turns up next.
        self._aborts = 0
        self._stopped = False
        self._number = 0
        #: How many workers have been spawned. Nothing reads it but the tests,
        #: which is where respawn policy is worth asserting about.
        self.starts = 0

    # -- the engine interface ----------------------------------------------
    #
    # Six awaitable calls over a blocking machine. Each hands `_ask` to a thread
    # and waits for that thread rather than for the pipe, which is what leaves the
    # loop free while the answer is being computed: the thread is where the wait
    # happens, and it is the thread the kill lands under.

    async def simplify(
        self, node: Node, context: Context, state: ParseState | None = None
    ) -> Result:
        return await self._await("simplify", (node, context, state))

    async def approx(
        self,
        node: Node,
        context: Context,
        digits: int | None = None,
        state: ParseState | None = None,
    ) -> Result:
        return await self._await("approx", (node, context, digits, state))

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
        self, node: Node, context: Context | None = None
    ) -> tuple[str, ...]:
        return await self._await("expression_variables", (node, context))

    async def _await(self, method: str, args: tuple[Any, ...]) -> Any:
        """One request, waited for on a thread of the loop's own pool."""
        return await asyncio.to_thread(self._ask, method, args)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Put a worker up, and let it import sympy while the user reads the menu.

        Returns as soon as the process exists; the handshake is waited for in
        the background, so that the first command usually finds a worker that
        has already paid for its imports.
        """
        with self._call:
            self._stopped = False
            if self._process is None:
                self._spawn()
                self._warm()

    def abort(self) -> None:
        """Take the computation away, there being no way to ask it to stop.

        Called from the event loop while a thread is blocked on the pipe. The
        reason is recorded first so that the death the kill causes is read as
        an abort rather than as a crash.
        """
        with self._guard:
            self._aborts += 1
        self._record(EngineAborted("Aborted"))
        self._kill()

    def shutdown(self) -> None:
        """End the worker for good, which is what leaving the app comes to."""
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
        """Send one request and wait for its answer, whatever becomes of it.

        The abort count is read before the queueing starts, so that an Esc
        pressed while this call is still waiting - for the lock, or for a
        worker to finish importing sympy - stops it as surely as one pressed
        while the answer is being computed.
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
                # An answer wearing another number belongs to a request that
                # was abandoned. It cannot happen while an aborted worker is
                # always killed rather than reused, and it is cheap insurance.
                if answer[0] == number:
                    return self._answered(answer)

    def _abandoned(self, aborts: int) -> None:
        """Give up where an abort has arrived since this call started."""
        if self._aborts != aborts:
            raise EngineAborted("Aborted")

    def _require(self) -> None:
        """See that there is a worker to ask, or say why there will not be one.

        A proxy that has gone down is retried exactly once here, which is the
        whole of the retry policy: the user pressing a command key is the
        clock, and nothing spins behind their back.
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
        """Wait for the worker to say it has finished importing the engine."""
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
        """The next message from the worker, watching its appetite while it waits.

        Every way the wait can end other than a message is a death, and the
        death path is what decides which one it was.
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
        """The memory cap, polled from outside where there is a figure to poll.

        A reading that fails is no reading: a worker that has just died has
        nothing to report, and the death path is about to hear about it anyway.
        Where the environment reads no resident set at all this is never called,
        and the poll below is left doing the one job it has either way, which is
        noticing that the worker has gone.
        """
        held = memory.process_bytes(process.pid)
        if held is not None and held > self.cap:
            self._record(EngineMemoryExceeded(self._out_of_memory()))
            self._kill()

    def _answered(self, answer: Any) -> Any:
        """What one response means: a value, a survivable bug, or a death."""
        if answer[1] == "ok":
            return answer[2]
        _, _, kind, message, traceback_text = answer
        if kind == worker.MEMORY:
            # The worker exits after saying this, so the death path runs now
            # rather than surprising the next command.
            self._record(EngineMemoryExceeded(self._out_of_memory()))
            raise self._collapse()
        raise EngineBug(message, traceback_text)

    def _out_of_memory(self) -> str:
        return f"The engine ran out of memory - the cap is {memory.written(self.cap)}"

    # -- the one death path ------------------------------------------------

    def _collapse(self) -> EngineError:
        """A worker is gone: say why, clear it away, and put another in its place.

        Every cause arrives here, which is what makes the recovery one piece of
        code. What is decided here is only what a child process makes of it; how
        many failures in a row are too many is `policy`'s. The exception is
        returned rather than raised so that each caller raises it from where its
        own error handling reads best.
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
        """Put a worker up. Raises `EngineDown` when the system will not have one."""
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
        # The child's end belongs to the child now; holding it here would keep
        # the pipe open past its death and turn every death into a hang.
        child.close()
        self.starts += 1
        self._connection = parent
        self._ready = False
        with self._guard:
            self._process = process
        memory.register_worker(process.pid)

    def _warm(self) -> None:
        """Wait for the handshake off the calling thread, so a death is noticed.

        Without this a worker that cannot start would sit dead in its pipe
        until the next command, and the three-strikes rule would need three
        commands to count to three instead of noticing on its own.
        """
        threading.Thread(target=self._warming, daemon=True).start()

    def _warming(self) -> None:
        with self._call:
            if self._stopped or self._connection is None:
                # Shut down, or replaced, while this thread waited its turn.
                return
            try:
                self._await_ready()
            except EngineError:
                # The death path has already put a replacement up, or has
                # decided not to. There is no command waiting to be told.
                pass

    # -- killing -----------------------------------------------------------

    def _record(self, reason: EngineError) -> None:
        """Say why the kill that is about to happen is happening.

        Only while there is something to kill, so that an Esc arriving as a
        computation finishes cannot label the next worker's death.
        """
        with self._guard:
            if self._process is not None and self._reason is None:
                self._reason = reason

    def _kill(self) -> None:
        """End the worker now. Called from the event loop, so it waits for nothing."""
        with self._guard:
            process = self._process
        if process is not None:
            _kill_off(process)


def _kill_off(process: BaseProcess) -> None:
    """Terminate, then kill: politeness first, but not for long."""
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
    """What to say about a worker nobody meant to kill.

    The exit code is the whole of the evidence, and it reads well: a negative
    one is the signal that ended the process, and `SIGKILL` from a process
    nobody signalled is the system's OOM killer nine times in ten.
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
