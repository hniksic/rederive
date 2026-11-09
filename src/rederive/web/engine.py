"""The app side of the engine in a browser: one Web Worker, no threads.

`WebEngine` is the third implementation of the session's `Runner`, alongside the
in-process one and the one that talks to a child process over a pipe. A request is a
`postMessage`, the answer arrives later on `onmessage`, and a future joins the two, so
only the awaiting coroutine is suspended.

Everything but transport is shared: the exceptions come from `engine.client`, so the app
reports a browser worker's death the same way as a desktop one, and recovery (including
the three-strikes rule) comes from `engine.policy`. This module only creates and kills
workers and turns messages into answers.

`terminate()` is the browser's SIGKILL and the only way to stop sympy mid-computation.
It's affordable because the worker is stateless. A replacement costs a fresh Pyodide and
sympy import, about two thirds of a second with precompiled packages, so no standby
worker is kept.

There is no memory cap or watchdog, since a tab can't read a worker's heap (see
`platform.web`); running out of memory arrives as an ordinary death. There is no log
file either: a dying worker's output is in the browser console, and error messages
point there.

Requests and answers are pickled, as over the pipe (trees, `Context`, `ParseState`,
`Result`), and `postMessage` copies the bytes as a `Uint8Array`.

Plot sampling uses the same worker but goes through `ask`. Its answers are arrays that
go straight to the page for drawing, so `ask` posts the request and expects nothing
back; the page draws the answer and notifies the plot session. `ask` takes no lock,
because the lock guards the single pending-answer slot. The worker itself, being single
threaded, orders sampling against engine calls. `value` is the exception: its answer is
four floats needed on this side, so it uses the pending slot like an engine call.

The worker also sends unprompted heap-size messages, after each answer and during long
computations. They are fed to the memory gauge and nothing waits on them.

Sharing the worker has costs: a drag queued behind an integral waits for it, and a
Simplify during a drag waits for the sampling in flight (which is bounded). Esc
terminates the worker for both kinds of request, so `lost` tells the plot side that its
pending sampling will never arrive.
"""

from __future__ import annotations

import asyncio
import pickle
from collections.abc import Callable, Sequence
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

__all__ = ["READY_TIMEOUT", "WebEngine"]

#: How long a new worker gets to boot Pyodide, install the package and import the engine.
#: Generous because a cold cache means a download, and the wait can be aborted anyway.
READY_TIMEOUT = 120.0

#: Where error messages send the user for details. The browser tells the page almost
#: nothing about why a worker failed; the console has the rest.
CONSOLE = "the browser console"


class WebEngine:
    """The engine calls, answered by a terminable Web Worker.

    The lock keeps one request in flight at a time. `abort` takes no lock: it's called
    from the event loop while a call awaits its answer, and kills the worker without
    needing its cooperation.
    """

    def __init__(
        self,
        spawn: Callable[[], Any],
        ready_timeout: float = READY_TIMEOUT,
    ) -> None:
        #: JS function returning a new worker that is already loading its runtime. The
        #: page creates it because the page knows the URL and module type.
        self._spawn = spawn
        self._ready_timeout = ready_timeout
        self._call = asyncio.Lock()
        self._worker: Any = None
        #: Proxies held by JS for this worker's lifetime, destroyed with it.
        self._proxies: list[Any] = []
        #: Resolved on the worker's handshake, or failed if it dies first.
        self._starting: asyncio.Future[None] | None = None
        self._ready = False
        #: Number and future of the request awaiting an answer.
        self._pending: tuple[int, asyncio.Future[Any]] | None = None
        #: The three-strikes rule and the reason the engine went down.
        self._deaths = Deaths()
        #: Abort counter. A call notes it at the start and gives up if it changes, so
        #: Esc also cancels a call still waiting for the worker to boot.
        self._aborts = 0
        self._stopped = False
        self._number = 0
        #: Optional callback invoked when a worker dies, so plot sampling (whose answers
        #: go to the page) can stop waiting.
        self.lost: Callable[[str], None] | None = None
        #: Number of workers created, used by tests to check respawn policy.
        self.starts = 0

    # -- the engine interface ----------------------------------------------

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

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Create a worker and let it boot in the background.

        Returns immediately. Loading the runtime, the package and sympy happens on the
        worker's thread, so the first command usually finds it ready.
        """
        self._stopped = False
        if self._worker is None:
            self._new()

    def abort(self) -> None:
        """Abort the computation by terminating the worker.

        Called from the event loop while a call awaits its answer. A terminated worker
        sends nothing, not even an end of stream, so the death is handled here directly
        rather than detected.
        """
        self._aborts += 1
        if self._worker is not None:
            self._collapse(EngineAborted("Aborted"))

    def shutdown(self) -> None:
        """Stop the worker permanently, when the user leaves the page."""
        self._stopped = True
        self._release()
        starting, self._starting = self._starting, None
        pending, self._pending = self._pending, None
        waiting = (starting, None if pending is None else pending[1])
        for future in waiting:
            if future is not None and not future.done():
                future.cancel()
        self._ready = False

    # -- one request -------------------------------------------------------

    async def _await(self, method: str, args: tuple[Any, ...]) -> Any:
        """Send one request and await its answer.

        The abort count is read before waiting for the lock, so Esc pressed while this
        call waits for the lock or for the worker to boot also aborts it.
        """
        aborts = self._aborts
        async with self._call:
            self._abandoned(aborts)
            await self._require()
            self._abandoned(aborts)
            self._number += 1
            number = self._number
            answer: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
            self._pending = (number, answer)
            try:
                _post(self._worker, pickle.dumps((number, method, args)))
            except Exception as error:
                raise self._collapse(EngineDied(f"The engine stopped: {error}")) from None
            return await answer

    # -- plot requests -----------------------------------------------------

    def numbered(self) -> int:
        """Allocate the number for the next request.

        Allocated before sending, because sending is a coroutine and the caller needs the
        number immediately. Engine and plot requests share one sequence, so numbers are
        unique.
        """
        self._number += 1
        return self._number

    async def value(self, method: str, args: tuple[Any, ...]) -> Any:
        """Send a plot request whose answer is a value for Python rather than a picture.

        Evaluating four typed bounds yields four floats, so this uses the lock and pending
        slot like the engine calls.
        """
        return await self._await(method, args)

    async def ask(self, number: int, method: str, args: tuple[Any, ...]) -> None:
        """Send a plot sampling request, whose answer goes to the page, not to Python.

        No lock is needed because the lock guards the pending slot, and this request has
        no answer to put there. The worker orders it against other requests by handling
        messages in arrival order.
        """
        await self._require()
        try:
            _post(self._worker, pickle.dumps((number, method, args)))
        except Exception as error:
            raise self._collapse(EngineDied(f"The engine stopped: {error}")) from None

    def _abandoned(self, aborts: int) -> None:
        """Raise `EngineAborted` if an abort arrived since this call started."""
        if self._aborts != aborts:
            raise EngineAborted("Aborted")

    async def _require(self) -> None:
        """Ensure there is a ready worker, or raise explaining why there isn't.

        An engine that went down is retried once per command; nothing retries in the
        background.
        """
        if self._stopped:
            raise EngineDown("The engine has been shut down")
        if self._deaths.down is not None:
            self._deaths.retrying()
            self._new()
        elif self._worker is None:
            self._new()
        starting = self._starting
        if starting is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(starting), self._ready_timeout)
        except asyncio.TimeoutError:
            died = EngineDied("The engine did not start in time", CONSOLE)
            raise self._collapse(died) from None

    # -- worker messages ---------------------------------------------------

    def _message(self, event: Any) -> None:
        """Handle a worker message, treating any failure as the worker's death.

        An exception escaping this JS callback would go unseen and leave the waiting call
        hanging, so a message that can't be decoded counts as a dead worker.
        """
        try:
            self._heard(event.data)
        except Exception as error:
            self._collapse(EngineDied(f"The engine answered with nonsense: {error}"))

    def _heard(self, data: Any) -> None:
        """Dispatch a handshake, an answer, or a page report that the worker failed.

        Python always sends pickles. A string comes from the page and means the worker
        couldn't be built (the runtime didn't load or the package didn't install), which
        counts as a death.
        """
        if isinstance(data, str):
            self._collapse(EngineDied(data, CONSOLE))
            return
        if getattr(data, "pane", None) is not None:
            # A sampling answer, handled by the page's JavaScript. Recognized by a field
            # rather than by type, since pickles also arrive as typed arrays.
            return
        held = getattr(data, "heap", None)
        if held is not None:
            # A heap-size report from the worker, for the memory gauge.
            memory.worker_holds(int(held))
            return
        message = pickle.loads(_bytes(data))
        if message[0] == worker.READY:
            self._greeted()
            return
        self._answered(message)

    def _failed(self, event: Any) -> None:
        """Handle an uncaught error in the worker, which kills it."""
        message = getattr(event, "message", "") or "The engine stopped"
        self._collapse(EngineDied(str(message), CONSOLE))

    def _greeted(self) -> None:
        """Handle the worker's handshake, releasing calls waiting for it to start."""
        self._ready = True
        self._deaths.ready()
        starting = self._starting
        if starting is not None and not starting.done():
            starting.set_result(None)

    def _answered(self, message: tuple[Any, ...]) -> None:
        """Handle an answer: a value, a survivable bug, or a death.

        An answer with a different number belongs to an abandoned request and is
        ignored. That can't happen while aborted workers are always terminated, but it's
        cheap insurance.
        """
        pending = self._pending
        if pending is None or pending[0] != message[0]:
            return
        _, answer = pending
        if message[1] == "ok":
            self._pending = None
            answer.set_result(message[2])
            return
        _, _, kind, said, traceback_text = message
        if kind == worker.MEMORY:
            # The worker closes after sending this, so handle the death now rather than
            # on the next command.
            self._collapse(EngineMemoryExceeded("The engine ran out of memory"))
            return
        self._pending = None
        answer.set_exception(EngineBug(said, traceback_text))

    # -- worker death ------------------------------------------------------

    def _collapse(self, reason: EngineError) -> EngineError:
        """Handle a dead worker: fail its pending futures, discard it, start another.

        Every cause of death ends up here: abort, failure to start, termination by the
        browser, or running out of memory. `policy` decides when too many failures in a
        row stop respawning. The exception is also returned so the caller can raise it.
        """
        self._release()
        ready, self._ready = self._ready, False
        starting, self._starting = self._starting, None
        pending, self._pending = self._pending, None
        _fail(starting, reason)
        if pending is not None:
            _fail(pending[1], reason)
        if self.lost is not None:
            # Plot sampling has no future to fail, since its answers go to the page, so it
            # must be told explicitly.
            self.lost(str(reason))
        respawn = self._deaths.died(reason, ready)
        if not self._stopped and respawn:
            self._new()
        return reason

    # -- worker lifecycle --------------------------------------------------

    def _new(self) -> None:
        """Create a worker and subscribe to its messages. Doesn't raise: a worker that
        fails to boot reports it in a message, like any other death."""
        from pyodide.ffi import create_proxy

        made = self._spawn()
        message = create_proxy(self._message)
        failure = create_proxy(self._failed)
        made.onmessage = message
        made.onerror = failure
        self._proxies = [message, failure]
        self._worker = made
        self._ready = False
        self._starting = asyncio.get_running_loop().create_future()
        self.starts += 1

    def _release(self) -> None:
        """Terminate the worker and destroy the proxies JS holds for it."""
        made, self._worker = self._worker, None
        proxies, self._proxies = self._proxies, []
        if made is not None:
            made.terminate()
        for proxy in proxies:
            proxy.destroy()
        # A terminated worker's heap is freed, so clear its figure from the gauge.
        memory.worker_holds(None)


def _post(made: Any, payload: bytes) -> None:
    """Send a pickle to the worker as a `Uint8Array`."""
    import js

    buffer = js.Uint8Array.new(len(payload))
    buffer.assign(payload)
    made.postMessage(buffer)


def _bytes(data: Any) -> bytes:
    """Copy the pickle out of a `Uint8Array`."""
    return bytes(data.to_py())


def _fail(future: asyncio.Future[Any] | None, reason: EngineError) -> None:
    """Fail a future, if there is one and it isn't done.

    The exception is immediately retrieved because nobody may be awaiting the future (a
    worker can die between commands), and the loop would otherwise log it as
    unretrieved.
    """
    if future is None or future.done():
        return
    future.set_exception(reason)
    future.exception()
