"""The engine worker under Pyodide: the same request table, over `postMessage`.

The desktop worker is a child process reading a pipe; this one is a Web Worker
reading its own `onmessage`, and between the two there is one table of methods
and one set of rules. `engine.worker.methods` is that table, imported from here
so that a browser cannot drift into answering a different set of questions than
a desktop does, and the statelessness the table is served under is the same
contract: nothing is kept between requests but warm imports and sympy's own
cache, which is what makes terminating this worker a recovery rather than a
loss.

This worker serves one family the desktop's does not, and it is the plot
sampling. It is here rather than in a third interpreter because sampling wants
sympy and numpy, and a Pyodide of its own would be another hundred megabytes and
a second sympy boot for a picture. The state it keeps is the one the contract
licenses - a content-keyed cache of lambdified expressions, which the protocol
treats as maybe-empty and rebuilds on demand - and nothing else.

Its answers leave by a different door. Everything the engine says is a pickle,
because Python is what reads it; a sampling says arrays, and arrays go to the
JavaScript that draws them, as a message of typed arrays that the page's own
listener picks up. The main thread's Python sees such a message go past and
reads nothing out of it, which is what keeps a numerical library out of the
instance that paints the screen.

What the desktop worker does that this one does not is take the tty off the
program and open a log. A worker has no tty to take, and what it prints goes to
the page's console, which is the browser's log file and needs no opening.

The last message a worker sends is the interesting one. A `MemoryError` is
answered before the worker closes itself, because the client wants to know
which of the several ways of running out of memory this was and the message
channel is the only way to tell it - the same order, and for the same reason,
as the child process's.
"""

from __future__ import annotations

import pickle
import sys
import traceback
from collections.abc import Callable
from typing import Any

from rederive.engine.worker import BUG, MEMORY, READY, methods
from rederive.plot.web import protocol as plots

__all__ = ["serve"]


def serve() -> None:
    """Be the engine, for as long as the page keeps this worker alive.

    The order is the point of it, as it is on the desktop: the engine is
    imported before the handshake goes out, so a page that has heard `ready`
    knows the next request pays for nothing but its own answer. What comes
    before the import here is the page's own - the runtime, the package - and
    is finished by the time this is called.

    The sampling table is built after the engine's, and its cost is numpy and
    this package's own plotting half - a fraction of what sympy costs, and paid
    before the handshake with everything else.
    """
    import js
    from pyodide.ffi import create_proxy

    from rederive.plot.web import sampler

    table = methods() | sampler.methods(_drawing)
    js.self.onmessage = create_proxy(lambda event: _asked(table, event.data))
    _post(pickle.dumps((READY, "ready")))


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
    except Exception:
        # Including `RecursionError`, which costs the answer and not the
        # worker: the engine promises never to raise on parser output, so
        # anything arriving here is a bug worth surfacing and not worth a
        # fresh Pyodide.
        _post(_trouble(number, BUG))


def _drawing(answer: dict[str, Any]) -> None:
    """Hand one picture to the page, as the arrays a canvas can draw.

    Not a pickle, and deliberately so: what is in it is samples, and samples
    are for the side that draws. A 1-D contiguous array of single precision
    crosses as one typed array with one copy; anything with a second dimension
    in it would cross as a nested list of JavaScript numbers, which the runtime
    documents as extremely slow and which a drag would feel every frame of.
    """
    import js
    from pyodide.ffi import to_js

    js.self.postMessage(to_js(answer, dict_converter=js.Object.fromEntries))


def _trouble(number: int, kind: str) -> bytes:
    """Say what went wrong, and leave the traceback where it can be read.

    The client gets the traceback text as well as the message, since a bug in
    the engine is worth more than its last line; the console keeps a copy for
    the deaths where there is nothing left to send one with.
    """
    text = traceback.format_exc()
    print(text, file=sys.stderr)
    return pickle.dumps((number, "error", kind, _message(), text))


def _message() -> str:
    """The current exception in one line, as a message line can show it."""
    kind, value, _ = sys.exc_info()
    name = "error" if kind is None else kind.__name__
    return f"{name}: {value}" if str(value) else name


def _post(payload: bytes) -> None:
    """Hand one pickle to the page, as the bytes a structured clone can carry."""
    import js

    buffer = js.Uint8Array.new(len(payload))
    buffer.assign(payload)
    js.self.postMessage(buffer)


def _close() -> None:
    """End this worker from the inside, a heap that has just failed a huge
    allocation not being worth trusting. The client is about to replace it
    either way; closing is what stops it answering anything else first."""
    import js

    js.self.close()
