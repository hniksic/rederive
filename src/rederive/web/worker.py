"""The engine worker under Pyodide: the desktop request table, over `postMessage`.

The desktop worker is a child process reading a pipe; this is a Web Worker reading
`onmessage`. Both serve `engine.worker.methods`, imported here so the two can't drift
apart, under the same statelessness contract: nothing persists between requests except
warm imports and sympy's cache, so terminating the worker loses nothing.

This worker also does plot sampling, which needs sympy and numpy; a separate Pyodide
would cost another hundred megabytes and a second sympy startup. Its only state is the
content-keyed cache of lambdified expressions the contract allows, which the protocol
treats as possibly empty.

Engine answers are pickles read by Python. Sampling answers are typed arrays sent
directly to the page's JavaScript, which the main thread's Python ignores, keeping numpy
out of the instance that draws the screen.

The worker also reports its heap size unprompted, since the main thread can't measure a
worker's memory and the status line gauge covers both. It reports between requests and,
during long computations, from the garbage collector callback.

Unlike the desktop worker, it doesn't detach from the tty or open a log; its output goes
to the browser console.

On `MemoryError`, the error is reported before the worker closes itself, so the client
can tell which kind of out-of-memory failure occurred, as with the child process.
"""

from __future__ import annotations

import gc
import pickle
import sys
import time
import traceback
from collections.abc import Callable
from typing import Any

from rederive.engine.worker import BUG, MEMORY, READY, methods
from rederive.plot.web import protocol as plots

__all__ = ["serve"]

#: Minimum interval between heap readings during a computation. Collections can run
#: thousands of times a second during an expansion, and reading the heap on each would
#: be too costly.
HEAP_PERIOD_SECONDS = 0.25

#: Time of the last heap reading and the last size reported. Unchanged sizes aren't
#: resent; WebAssembly memory only grows, so most readings repeat the last one.
_looked = 0.0
_said: int | None = None

#: Guards against reentry: reporting allocates, which can trigger a collection, which
#: would report again.
_reporting = False


def serve() -> None:
    """Serve engine and sampling requests for the lifetime of the worker.

    As on the desktop, the engine and sampler (with numpy) are imported before `ready`
    is sent, so no request after the handshake pays import costs.
    """
    import js
    from pyodide.ffi import create_proxy

    from rederive.plot.web import sampler

    table = methods() | sampler.methods(_drawing)
    js.self.onmessage = create_proxy(lambda event: _asked(table, event.data))
    gc.callbacks.append(_collected)
    _post(pickle.dumps((READY, "ready")))
    _held()


def _asked(table: dict[str, Callable[..., Any]], data: Any) -> None:
    """One request, answered however it turns out."""
    number, method, args = pickle.loads(bytes(data.to_py()))
    try:
        answer = table[method](*args)
        if method in plots.DRAWING:
            _drawing({"number": number} | answer)
        else:
            _post(pickle.dumps((number, "ok", answer)))
    except MemoryError:
        _post(_trouble(number, MEMORY))
        _close()
        return
    except Exception:
        # Including `RecursionError`, which fails the request but not the worker. The
        # engine never raises on parser output, so anything here is a bug worth
        # reporting, not worth restarting Pyodide.
        _post(_trouble(number, BUG))
    _held()


def _held() -> None:
    """Report this worker's heap size to the page.

    A Web Worker's memory can't be measured from outside, so it reports at startup and
    after each request. The report is a plain message with a `heap` field rather than a
    pickled, numbered answer, since it isn't a reply to anything.
    """
    global _looked, _said, _reporting

    _reporting = True
    try:
        import js
        from pyodide.ffi import to_js

        from rederive.platform.web import heap

        _looked = time.monotonic()
        size = heap()
        if size is None or size == _said:
            return
        _said = size
        js.self.postMessage(to_js({"heap": size}, dict_converter=js.Object.fromEntries))
    except Exception:
        # This runs in the middle of other work, and a gauge isn't worth an error; a
        # failure just skips one reading.
        pass
    finally:
        _reporting = False


def _collected(phase: str, _info: dict[str, Any]) -> None:
    """Report the heap from a garbage collector callback during a computation.

    While a request runs, no timers fire and no messages are read, so the collector is
    the only hook that runs. A computation that fills the heap collects often, so the
    gauge keeps up; one that doesn't allocate has nothing new to report anyway.

    Reports only at the end of a collection, at most once per `HEAP_PERIOD_SECONDS`.
    """
    if phase != "stop" or _reporting:
        return
    if time.monotonic() - _looked < HEAP_PERIOD_SECONDS:
        return
    _held()


def _drawing(answer: dict[str, Any]) -> None:
    """Send a sampling to the page's JavaScript as typed arrays, not a pickle.

    A contiguous 1-D float32 array crosses as one typed array with a single copy. A
    multi-dimensional array would become nested JS lists, which Pyodide documents as
    very slow and would make dragging sluggish.
    """
    import js
    from pyodide.ffi import to_js

    js.self.postMessage(to_js(answer, dict_converter=js.Object.fromEntries))


def _trouble(number: int, kind: str) -> bytes:
    """Build an error answer, including the traceback, and print it to the console.

    The console copy survives when the worker dies before the answer can be sent.
    """
    text = traceback.format_exc()
    print(text, file=sys.stderr)
    return pickle.dumps((number, "error", kind, _message(), text))


def _message() -> str:
    """Return the current exception as one line for the message line."""
    kind, value, _ = sys.exc_info()
    name = "error" if kind is None else kind.__name__
    return f"{name}: {value}" if str(value) else name


def _post(payload: bytes) -> None:
    """Send a pickle to the page as a `Uint8Array`."""
    import js

    buffer = js.Uint8Array.new(len(payload))
    buffer.assign(payload)
    js.self.postMessage(buffer)


def _close() -> None:
    """Close this worker, whose heap can't be trusted after a failed allocation. The
    client replaces it anyway; closing stops it from answering anything else first."""
    import js

    js.self.close()
