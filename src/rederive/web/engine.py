"""The app side of the engine in a browser: six calls, one Web Worker, no threads.

`WebEngine` is the third implementation of the session's `Runner`, beside the
one that computes in this process and the one that talks to a child process
down a pipe, and it is the shape the async seam was made for. There is no
thread here for a call to block in and no pipe to block on: a request is
`postMessage`, the answer arrives later on `onmessage`, and what joins the two
is a future the message resolves. Nothing suspends but the coroutine that asked.

What it borrows is everything that is not transport. The exceptions are
`engine.client`'s, so the app's reporting reads a browser death exactly as it
reads a desktop one; the recovery is `engine.policy`'s, three-strikes rule
included. What is written here is the half that cannot travel: making a worker,
killing one, and turning a message into an answer.

Killing is the whole reason for the worker. `terminate()` is the browser's
SIGKILL - immediate, uncooperative, and the only way to stop sympy in the
middle of a computation - and it is affordable for the same reason it is on the
desktop: the worker is stateless, so the replacement knows everything the dead
one knew. It costs a fresh Pyodide and a fresh sympy import, which stage 0
measured at about two thirds of a second on the precompiled channel. That is a
pause and not a hang, and it is why nothing here keeps a standby worker warm.

Two desktop things have no browser equivalent and are deliberately absent. There
is no memory cap and no watchdog, because a tab cannot read a worker's heap -
`platform.web` says so, and an allocation that goes too far arrives here as a
death like any other. And there is no log file: what a dying worker printed is
in the page's console, which is where the message sends a reader.

Requests and answers cross as pickles. It is the same vocabulary the pipe
carries - a tree, a `Context`, a `ParseState`, a `Result` - and the same
serialisation, since both ends are the same Python running the same package;
`postMessage` carries the bytes as a `Uint8Array`, which is a copy and nothing
more.

There is a second family of requests on the same worker, and it is the plot
sampling. It goes out through `ask` rather than through the six calls, and the
difference is where its answer goes: a sampling answers in arrays, and arrays
belong on the page where they are drawn rather than in the interpreter that
routes them. So `ask` numbers a request, posts it, and expects nothing back -
the page hears the answer, draws it, and says so to the plot session. That is
also why `ask` takes no lock. The lock guards the one pending slot the six calls
share, and a request with no answer to this side needs no slot; what actually
orders a sampling against a Simplify is the worker, which is one thread of one
interpreter and takes its messages as they come.

The price of sharing is worth naming. A drag posted behind an integral waits for
the integral, and a Simplify pressed mid-drag waits for the one sampling in
flight - bounded, since sampling is depth- and size-capped, but real. And Esc
terminates the worker under both families at once, which is why `lost` is here:
the side waiting for a sampling that will never arrive has to be told, or its
queue stops moving.
"""

from __future__ import annotations

import asyncio
import pickle
from collections.abc import Callable, Sequence
from typing import Any

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

#: How long a freshly made worker has to boot Pyodide, install the package and
#: import the engine before it is given up on. Generous: it is a download on a
#: cold cache, and the wait is abortable anyway.
READY_TIMEOUT = 120.0

#: What a worker that never came up is reported as. The browser tells a page
#: almost nothing about why a worker failed, so the console is where the rest
#: of it is, and the message says so rather than pretending to know.
CONSOLE = "the browser console"


class WebEngine:
    """The six engine calls, answered by a Web Worker that can be terminated.

    One request is in flight at a time, which is what the lock is for: the
    session is single-threaded, the worker holds nothing between requests, and
    there is nothing a second concurrent request could be about. `abort` is the
    exception - it is called from the event loop while a call is awaiting its
    answer, and it takes no lock, since a busy worker's cooperation is exactly
    what it cannot have.
    """

    def __init__(
        self,
        spawn: Callable[[], Any],
        ready_timeout: float = READY_TIMEOUT,
    ) -> None:
        #: What the page does to make a worker: a JS function answering with a
        #: fresh one, already loading its runtime. Making it is the page's
        #: business because the URL and the module type are the page's.
        self._spawn = spawn
        self._ready_timeout = ready_timeout
        self._call = asyncio.Lock()
        self._worker: Any = None
        #: The proxies JS holds for as long as this worker lives, and which
        #: are destroyed with it.
        self._proxies: list[Any] = []
        #: Resolved when the worker has answered its handshake, and failed
        #: with the death of a worker that never did.
        self._starting: asyncio.Future[None] | None = None
        self._ready = False
        #: The request waiting for an answer, as its number and its future.
        self._pending: tuple[int, asyncio.Future[Any]] | None = None
        #: The three-strikes rule and what put the engine down, which are the
        #: parts of dying that have nothing to do with a Web Worker.
        self._deaths = Deaths()
        #: How many aborts have been asked for. A call reads it before it
        #: starts and gives up if it has moved: an Esc pressed while the call
        #: is still waiting for a worker to boot is as much an abort as one
        #: pressed while sympy is grinding.
        self._aborts = 0
        self._stopped = False
        self._number = 0
        #: Told when a worker dies, so that whoever was waiting on an answer
        #: this side will never see - the plot sampling, whose answers go to
        #: the page - can stop waiting. Assigned rather than required.
        self.lost: Callable[[str], None] | None = None
        #: How many workers have been made. Nothing reads it but the tests,
        #: which is where respawn policy is worth asserting about.
        self.starts = 0

    # -- the engine interface ----------------------------------------------

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

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Put a worker up, and let it boot while the user reads the menu.

        Returns as soon as the worker exists. Everything expensive about it -
        the runtime, the package, the sympy import - happens on the worker's
        own thread, so the first command usually finds one that has already
        paid for its imports.
        """
        self._stopped = False
        if self._worker is None:
            self._new()

    def abort(self) -> None:
        """Take the computation away, there being no way to ask it to stop.

        Called from the event loop while a call awaits its answer. Nothing
        comes back from a terminated worker - not even an end of stream - so
        the death is worked out here rather than noticed, which is the one
        place the browser needs more code than the pipe does.
        """
        self._aborts += 1
        if self._worker is not None:
            self._collapse(EngineAborted("Aborted"))

    def shutdown(self) -> None:
        """End the worker for good, which is what leaving the page comes to."""
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
        """Send one request and await its answer, whatever becomes of it.

        The abort count is read before the queueing starts, so that an Esc
        pressed while this call is still waiting - for the lock, or for a
        worker to finish booting - stops it as surely as one pressed while the
        answer is being computed.
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

    # -- the second family -------------------------------------------------

    def numbered(self) -> int:
        """The number the next request will travel under.

        Taken before the sending rather than during it, because the sending is
        a coroutine and the side that asked has to know what it is waiting for
        the moment it asks. One sequence for both families, so that an answer
        never wears a number two requests could claim.
        """
        self._number += 1
        return self._number

    async def ask(self, number: int, method: str, args: tuple[Any, ...]) -> None:
        """Send one request whose answer is the page's rather than Python's.

        The plot sampling, and nothing else. No lock, because there is nothing
        here for a second request to collide with: what the lock protects is the
        pending slot the six calls share, and this request has no answer to put
        in one. What it does share is the worker, and the worker orders the two
        families itself by taking its messages in the order they arrive.
        """
        await self._require()
        try:
            _post(self._worker, pickle.dumps((number, method, args)))
        except Exception as error:
            raise self._collapse(EngineDied(f"The engine stopped: {error}")) from None

    def _abandoned(self, aborts: int) -> None:
        """Give up where an abort has arrived since this call started."""
        if self._aborts != aborts:
            raise EngineAborted("Aborted")

    async def _require(self) -> None:
        """See that there is a worker to ask, or say why there will not be one.

        An engine that has gone down is retried exactly once here, which is the
        whole of the retry policy: the user pressing a command key is the
        clock, and nothing spins behind their back.
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

    # -- what the worker says ----------------------------------------------

    def _message(self, event: Any) -> None:
        """One message off the worker, and a death where it will not be read.

        The callback is JS's, so an exception raised under it lands nowhere a
        user would look and leaves the waiting call waiting forever. A message
        that will not unpickle is a dead worker like any other.
        """
        try:
            self._heard(event.data)
        except Exception as error:
            self._collapse(EngineDied(f"The engine answered with nonsense: {error}"))

    def _heard(self, data: Any) -> None:
        """The handshake, an answer, or the page saying it has no worker.

        Everything Python sends is a pickle. A string is the page's own way of
        saying that the worker could not be built at all - a runtime that would
        not load, a package that would not install - which is a death before
        the handshake and counts as one.
        """
        if isinstance(data, str):
            self._collapse(EngineDied(data, CONSOLE))
            return
        if getattr(data, "pane", None) is not None:
            # A sampling answer, which is the page's traffic and not this
            # side's: the arrays in it are drawn by whoever is listening for
            # them in JavaScript, and are never read here. Recognized by a
            # field rather than by a type, because what a pickle arrives as is
            # a typed array and so is half of what a sampling answers with.
            return
        message = pickle.loads(_bytes(data))
        if message[0] == worker.READY:
            self._greeted()
            return
        self._answered(message)

    def _failed(self, event: Any) -> None:
        """The worker raised where nothing was there to catch it, and is gone."""
        message = getattr(event, "message", "") or "The engine stopped"
        self._collapse(EngineDied(str(message), CONSOLE))

    def _greeted(self) -> None:
        """The worker has said it is up, which is what a pending call waits for."""
        self._ready = True
        self._deaths.ready()
        starting = self._starting
        if starting is not None and not starting.done():
            starting.set_result(None)

    def _answered(self, message: tuple[Any, ...]) -> None:
        """What one answer means: a value, a survivable bug, or a death.

        An answer wearing another number belongs to a request that was
        abandoned. It cannot happen while an aborted worker is always
        terminated rather than reused, and it is cheap insurance.
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
            # The worker stops after saying this, so the death path runs now
            # rather than surprising the next command.
            self._collapse(EngineMemoryExceeded("The engine ran out of memory"))
            return
        self._pending = None
        answer.set_exception(EngineBug(said, traceback_text))

    # -- the one death path ------------------------------------------------

    def _collapse(self, reason: EngineError) -> EngineError:
        """A worker is gone: fail what it owed, clear it away, put another up.

        Every cause arrives here - an abort, a worker that would not start, one
        the browser took away, one that ran out of memory - which is what makes
        the recovery one piece of code. How many failures in a row are too many
        is `policy`'s to say. The exception is returned as well as delivered so
        that a caller can raise it from where its own error handling reads best.
        """
        self._release()
        ready, self._ready = self._ready, False
        starting, self._starting = self._starting, None
        pending, self._pending = self._pending, None
        _fail(starting, reason)
        if pending is not None:
            _fail(pending[1], reason)
        if self.lost is not None:
            # The second family has no future to fail: its answers go to the
            # page, so the only way to tell it that one is never coming is to
            # say so.
            self.lost(str(reason))
        respawn = self._deaths.died(reason, ready)
        if not self._stopped and respawn:
            self._new()
        return reason

    # -- the worker itself -------------------------------------------------

    def _new(self) -> None:
        """Make a worker, and listen to it. Raises nothing: the page's job is
        to answer with an object, and a worker that will not boot says so in a
        message like any other death."""
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
        """Terminate the worker and give the page back the proxies it holds."""
        made, self._worker = self._worker, None
        proxies, self._proxies = self._proxies, []
        if made is not None:
            made.terminate()
        for proxy in proxies:
            proxy.destroy()


def _post(made: Any, payload: bytes) -> None:
    """Hand one pickle to the worker, as the bytes a structured clone can carry."""
    import js

    buffer = js.Uint8Array.new(len(payload))
    buffer.assign(payload)
    made.postMessage(buffer)


def _bytes(data: Any) -> bytes:
    """The pickle inside a `Uint8Array`, copied out of the page's memory."""
    return bytes(data.to_py())


def _fail(future: asyncio.Future[Any] | None, reason: EngineError) -> None:
    """Fail one future, where there is one and it is still waiting.

    The exception is read straight back out because there may be nobody
    awaiting this future - a worker can die between commands - and an exception
    a future was never asked for is one the loop reports to the console as
    though something had gone unnoticed.
    """
    if future is None or future.done():
        return
    future.set_exception(reason)
    future.exception()
