"""The engine in a worker process: its answers, and every way the worker can die.

Spawning a worker costs about a second of sympy import, so tests that only want an
answer share one. Tests that kill the worker each get their own.

The abort and the memory cap are tested with two private methods in the worker's table
(`hang` and `allocate`). Relying on a real computation being slow or memory-hungry
enough would make the tests depend on sympy rather than on the recovery path.

For the same reason no kill is timed with a sleep. Each is triggered by an engine
event (a request being sent, a worker not yet up) by wrapping one method on the
instance, so it lands at the intended point on both fast and loaded machines.
"""

import signal
import threading
import time

import psutil
import pytest

from rederive import memory
from rederive.engine import worker as worker_module
from rederive.engine.client import Context
from rederive.engine.policy import STARTS_BEFORE_DOWN
from rederive.engine.remote import (
    EngineAborted,
    EngineDied,
    EngineMemoryExceeded,
    RemoteEngine,
)
from rederive.model.session import Session
from rederive.syntax import ParseState, parse_expression

#: A cap small enough to hit on purpose, yet large enough for a worker with sympy
#: imported to stay under otherwise.
SMALL_CAP = 300 << 20

#: How long a test waits on a background thread, long enough that a loaded machine
#: spawning workers isn't mistaken for a broken one.
PATIENCE = 60.0


def tree(text):
    """Parse an expression the way a fresh session would."""
    return parse_expression(text, ParseState()).node


def until(condition, patience=PATIENCE):
    """Wait for a condition set by a background thread; return whether it became true."""
    deadline = time.monotonic() + patience
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def fire(event, action, patience=PATIENCE):
    """Run `action` on another thread once `event` is set.

    The wait is bounded, so an event that never arrives fails the test rather than
    hanging it.
    """

    def run():
        event.wait(patience)
        action()

    threading.Thread(target=run, daemon=True).start()


def after_the_request(engine):
    """An event set once the engine's next request has been sent.

    `_ask` sends and then reads the answer, so the first read follows the send, and a
    worker killed on this event is killed while handling the request. That matters
    because a kill before the send would instead be the abandoned-call case tested
    separately (`_ask` reads the abort count on entry).
    """
    sent = threading.Event()
    reading = engine._receive

    def watched(deadline=None):
        sent.set()
        return reading(deadline)

    engine._receive = watched
    return sent


class Queueing:
    """A wrapper around the engine's call lock that signals when a thread queues for it.

    Wrapped rather than replaced, so threads already holding the lock keep the same
    one. The event is set inside the acquire, after `_ask` has read the abort count.
    """

    def __init__(self, lock, thread, event):
        self._lock = lock
        self._thread = thread
        self._event = event

    def __enter__(self):
        if threading.current_thread() is self._thread:
            self._event.set()
        return self._lock.__enter__()

    def __exit__(self, *raised):
        return self._lock.__exit__(*raised)


def a_worker_that_is_still_starting(engine):
    """Start a worker, returning an event set when this thread's call queues for it.

    At that moment the worker must not be up yet, and the call must have started (an
    abort before `_ask` reads the abort count is not an abandoned call). Rather than
    guess with a sleep, the starting thread is held at its first read, the one awaiting
    the handshake, until this thread is queueing for the lock it holds. So when the
    event is set, the handshake is unread and the call is under way, however long the
    sympy import takes.
    """
    arrived = threading.Event()
    queueing = threading.Event()
    waiting = threading.Event()
    reading = engine._receive

    def watched(deadline=None):
        if not waiting.is_set():
            arrived.set()
            queueing.wait(PATIENCE)
            waiting.set()
        return reading(deadline)

    engine._receive = watched
    engine.start()
    engine._call = Queueing(engine._call, threading.current_thread(), queueing)
    assert arrived.wait(PATIENCE)
    return waiting


@pytest.fixture(scope="module")
def remote():
    """A worker shared by every test that only wants answers from it.

    Capped, because one test computes a hostile expression, and polled often enough
    that the cap stops a runaway in time.
    """
    engine = RemoteEngine(cap=SMALL_CAP, period=0.05)
    engine.start()
    yield engine
    engine.shutdown()


@pytest.fixture
def mortal():
    """Make workers private to one test, for tests that kill them."""
    engines = []

    def make(**arguments):
        engine = RemoteEngine(**arguments)
        engines.append(engine)
        return engine

    yield make
    for engine in engines:
        engine.shutdown()


# -- the five answers --------------------------------------------------------


async def test_simplify_comes_back_from_the_child(remote):
    assert (await remote.simplify(tree("2 + 3"), Context())).text == "5"


async def test_factor_comes_back_from_the_child(remote):
    assert (await remote.factor(tree("x^2 - 1"), Context())).text == "(x + 1)*(x - 1)"


async def test_expand_comes_back_from_the_child(remote):
    assert (await remote.expand(tree("(x + 1)^2"), Context())).text == "x^2 + 2*x + 1"


async def test_solve_comes_back_from_the_child(remote):
    """Unlike the other operations, soLve sends back a tuple of results."""
    answers = await remote.solve(tree("x^2 - 5*x + 6 = 0"), Context())
    assert [answer.text for answer in answers] == ["x = 2", "x = 3"]


async def test_a_solve_with_no_solutions_comes_back_empty(remote):
    assert await remote.solve(tree("x = x + 1"), Context()) == ()


async def test_the_variables_come_back_from_the_child(remote):
    assert await remote.expression_variables(tree("x^2 - a^2"), Context()) == ("x", "a")


async def test_the_relation_comes_back_from_the_child(remote):
    assert await remote.relation_of(tree("IM(x = y)"), Context()) == "="
    assert await remote.relation_of(tree("x^2"), Context()) is None


async def test_a_session_given_one_computes_through_it(remote):
    session = Session(runner=remote)
    session.author("(x + 1)*(x - 1)")
    assert (await session.expand("#1")).text == "x^2 - 1"


# -- the deaths --------------------------------------------------------------


async def test_an_abort_kills_the_worker_and_the_next_one_is_already_coming(mortal):
    engine = mortal()
    # Make a request rather than just start, so the worker is past its imports before
    # `hang` is sent.
    assert (await engine.simplify(tree("1 + 1"), Context())).text == "2"
    fire(after_the_request(engine), engine.abort)
    with pytest.raises(EngineAborted):
        engine._ask("hang", ())
    # The replacement is spawned by the death itself, not by the next command.
    assert engine.starts == 2
    assert (await engine.simplify(tree("1 + 1"), Context())).text == "2"


async def test_an_abort_while_a_call_is_still_waiting_stops_it_too(mortal):
    """Esc must work even before the worker is up.

    The abort lands while the call is queueing for a worker still importing sympy. The
    command must not wait out the import and then run on the replacement; since the
    request is `hang`, that failure would show as a hang.
    """
    engine = mortal()
    fire(a_worker_that_is_still_starting(engine), engine.abort)
    with pytest.raises(EngineAborted):
        engine._ask("hang", ())
    assert (await engine.simplify(tree("1 + 1"), Context())).text == "2"


async def test_a_worker_over_the_cap_is_taken_away(mortal):
    engine = mortal(cap=SMALL_CAP, period=0.05)
    engine.start()
    with pytest.raises(EngineMemoryExceeded) as raised:
        engine._ask("allocate", (2 * SMALL_CAP >> 20,))
    # The error names the cap in force, formatted like other sizes (the format itself
    # is tested in `test_memory`).
    assert memory.written(SMALL_CAP) in str(raised.value)
    assert (await engine.simplify(tree("1 + 1"), Context())).text == "2"


async def test_a_worker_killed_from_outside_is_replaced(mortal):
    engine = mortal()
    engine.start()
    await engine.simplify(tree("1 + 1"), Context())
    pid = engine._process.pid
    # Killed through psutil rather than a signal, since Windows has no `SIGKILL`. This
    # simulates something outside the app (the OOM killer, a task manager) killing it.
    fire(after_the_request(engine), lambda: psutil.Process(pid).kill())
    with pytest.raises(EngineDied) as raised:
        engine._ask("hang", ())
    # Only a signal in the exit code shows the worker was killed rather than exited, and
    # Windows reports an ordinary code. Recovery works either way; only naming the
    # cause needs a signal.
    if hasattr(signal, "SIGKILL"):
        assert str(raised.value).startswith("The engine was killed")
    assert engine.starts == 2
    assert (await engine.simplify(tree("1 + 1"), Context())).text == "2"


async def test_a_bug_in_the_engine_costs_the_answer_and_not_the_worker(remote):
    """Uses the shared worker, since the point is that it survives."""
    starts = remote.starts
    with pytest.raises(Exception) as raised:
        remote._ask("no_such_method", ())
    assert "no_such_method" in str(raised.value)
    assert remote.starts == starts
    assert (await remote.simplify(tree("1 + 1"), Context())).text == "2"


def refuses_to_start(connection, cap):
    """A worker that dies before the handshake.

    Module-level because a spawned child imports this module and looks its target up by
    name, which wouldn't work for a closure or local.
    """
    raise SystemExit(1)


async def test_a_worker_that_will_not_start_puts_the_proxy_down(monkeypatch, mortal):
    monkeypatch.setattr(worker_module, "serve", refuses_to_start)
    engine = mortal()
    with pytest.raises(EngineDied):
        await engine.simplify(tree("1 + 1"), Context())
    assert until(lambda: engine._deaths.down is not None)
    assert engine.starts == STARTS_BEFORE_DOWN
    # It stays down: a command issued while down tries exactly one more spawn, and
    # nothing else spawns. The proxy has no timer, backoff or retry thread, so `starts`
    # after the next command proves nothing respawned in the background.
    with pytest.raises(EngineDied):
        await engine.simplify(tree("1 + 1"), Context())
    assert engine.starts == STARTS_BEFORE_DOWN + 1


# -- a hostile expression, never run without a cap ---------------------------


async def test_a_hostile_power_comes_back_inert_and_the_session_stands(remote):
    """`10^10^10` is returned unevaluated without killing the worker.

    Sympy evaluates `Integer**Integer` on construction, which here would mean ten
    billion digits and four gigabytes, so the engine declines to build it and returns
    the power as written. The cap is a precaution; the test asserts it isn't needed:
    the worker survives and the session remains usable.
    """
    starts = remote.starts
    session = Session(runner=remote)
    session.author("10^10^10")
    assert (await session.simplify("#1")).text == "10^10000000000"
    session.author("2 + 3")
    assert (await session.simplify("#3")).text == "5"
    assert remote.starts == starts
