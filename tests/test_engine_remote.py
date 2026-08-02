"""The engine at arm's length: the answers, and every way the worker can die.

Spawning a worker costs about a second of sympy import, so the tests that only
want an answer share one. The tests that want a death each get their own,
since a death is what they are about.

The two private methods in the worker's table are what the abort and the cap
are tested with. Timing a real computation to be slow enough or hungry enough
would make these tests about sympy's mood rather than about the recovery path.

For the same reason no death is aimed with a sleep. Each is fired off the
engine's own signal - a request going out, a worker not up yet - by wrapping
one method on the instance, so that the death lands where the test says it
does on a fast machine and a loaded one alike.
"""

import os
import signal
import threading
import time

import pytest

from rederive import memory
from rederive.engine import worker as worker_module
from rederive.engine.client import Context
from rederive.engine.remote import (
    STARTS_BEFORE_DOWN,
    EngineAborted,
    EngineDied,
    EngineMemoryExceeded,
    RemoteEngine,
)
from rederive.model.session import Session
from rederive.syntax import ParseState, parse_expression

#: A cap small enough to be met on purpose and large enough for a worker with
#: sympy imported to live under while it is.
SMALL_CAP = 300 << 20

#: How long a test waits for something a background thread is doing. Long
#: enough that a loaded machine spawning workers is not mistaken for a broken
#: one.
PATIENCE = 60.0


def tree(text):
    """One expression, parsed the way a fresh session would parse it."""
    return parse_expression(text, ParseState()).node


def until(condition, patience=PATIENCE):
    """Wait for something a background thread is doing, and say whether it did."""
    deadline = time.monotonic() + patience
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def fire(event, action, patience=PATIENCE):
    """Do something on another thread, at the moment the event says has come.

    The wait is bounded so that a signal that never arrives fails the test
    waiting on the action rather than hanging it.
    """

    def run():
        event.wait(patience)
        action()

    threading.Thread(target=run, daemon=True).start()


def after_the_request(engine):
    """An event set once the engine's next request is on the wire.

    `_ask` sends and then reads for the answer, so the first read is the first
    thing that happens after the send, and a worker taken away on this event is
    taken away from a request it already has. That is the distinction these
    tests turn on: a kill that landed before the send would be the abandoned
    call another test is about, `_ask` having read the abort count on the way
    in.
    """
    sent = threading.Event()
    reading = engine._receive

    def watched(deadline=None):
        sent.set()
        return reading(deadline)

    engine._receive = watched
    return sent


class Queueing:
    """The engine's call lock, saying when one thread has begun queueing for it.

    Wrapped rather than replaced, so that the threads holding the lock already
    keep holding the same one. The event is set from inside the acquire, which
    is past the point where `_ask` reads the abort count it will be judged
    against.
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
    """Start a worker, and say when a call of this thread's is queueing for it.

    Two things have to be true at that moment and neither is worth guessing at
    with a sleep. The worker must not be up yet, which is what the test using
    this is about; and the call must have started, since an abort arriving
    before `_ask` reads the abort count is not an abandoned call at all.

    So the warming thread is held at the top of its first read - the read that
    waits for the handshake - until this thread is queueing for the lock it
    holds. When the event is set, the handshake is still unread and the call is
    under way, whatever the machine's idea of how long importing sympy takes.
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
    """One worker for every test that only wants an answer out of it.

    Capped, because one of the tests through it computes a hostile expression
    and a hostile expression is never run without a cap, and watched often
    enough that the cap would stop a runaway rather than follow it.
    """
    engine = RemoteEngine(cap=SMALL_CAP, period=0.05)
    engine.start()
    yield engine
    engine.shutdown()


@pytest.fixture
def mortal():
    """A worker of one test's own, for a test that means to kill it."""
    engines = []

    def make(**arguments):
        engine = RemoteEngine(**arguments)
        engines.append(engine)
        return engine

    yield make
    for engine in engines:
        engine.shutdown()


# -- the six answers ---------------------------------------------------------


def test_simplify_comes_back_from_the_child(remote):
    assert remote.simplify(tree("2 + 3"), Context()).text == "5"


def test_approx_comes_back_from_the_child(remote):
    assert remote.approx(tree("1/3"), Context(), None, None).text == "0.333333"


def test_factor_comes_back_from_the_child(remote):
    assert remote.factor(tree("x^2 - 1"), Context()).text == "(x + 1)*(x - 1)"


def test_expand_comes_back_from_the_child(remote):
    assert remote.expand(tree("(x + 1)^2"), Context()).text == "x^2 + 2*x + 1"


def test_solve_comes_back_from_the_child(remote):
    """Several results down one pipe, which is what makes soLve unlike the
    other five: a tuple crosses where the others send one answer."""
    answers = remote.solve(tree("x^2 - 5*x + 6 = 0"), Context())
    assert [answer.text for answer in answers] == ["x = 2", "x = 3"]


def test_a_solve_with_no_solutions_comes_back_empty(remote):
    assert remote.solve(tree("x = x + 1"), Context()) == ()


def test_the_variables_come_back_from_the_child(remote):
    assert remote.expression_variables(tree("x^2 - a^2"), Context()) == ("x", "a")


def test_a_session_given_one_computes_through_it(remote):
    session = Session(runner=remote)
    session.author("(x + 1)*(x - 1)")
    assert session.expand("#1").text == "x^2 - 1"


# -- the deaths --------------------------------------------------------------


def test_an_abort_kills_the_worker_and_the_next_one_is_already_coming(mortal):
    engine = mortal()
    # Asked for rather than started, so that the worker is up and past its
    # imports before anything hangs: a `hang` that never got sent is not what
    # this test is about.
    assert engine.simplify(tree("1 + 1"), Context()).text == "2"
    fire(after_the_request(engine), engine.abort)
    with pytest.raises(EngineAborted):
        engine._ask("hang", ())
    # The replacement is spawned by the death itself, not by the next command.
    assert engine.starts == 2
    assert engine.simplify(tree("1 + 1"), Context()).text == "2"


def test_an_abort_while_a_call_is_still_waiting_stops_it_too(mortal):
    """An Esc pressed before the worker is even up must still mean Esc.

    Here the abort lands while the call is queueing for a worker that is still
    importing sympy. The command must not wait that import out and then run on
    the replacement, which is the one thing an abort must never leave
    happening - and which is a hang rather than a wrong answer, `hang` being
    what was asked for.
    """
    engine = mortal()
    fire(a_worker_that_is_still_starting(engine), engine.abort)
    with pytest.raises(EngineAborted):
        engine._ask("hang", ())
    assert engine.simplify(tree("1 + 1"), Context()).text == "2"


def test_a_worker_over_the_cap_is_taken_away(mortal):
    engine = mortal(cap=SMALL_CAP, period=0.05)
    engine.start()
    with pytest.raises(EngineMemoryExceeded) as raised:
        engine._ask("allocate", (2 * SMALL_CAP >> 20,))
    # The refusal says what the cap was, written the way every other size the
    # program reports is written. How that reads is `test_memory` to pin; what
    # this one is about is that the figure named is the cap in force.
    assert memory.written(SMALL_CAP) in str(raised.value)
    assert engine.simplify(tree("1 + 1"), Context()).text == "2"


def test_a_worker_killed_from_outside_is_replaced(mortal):
    engine = mortal()
    engine.start()
    engine.simplify(tree("1 + 1"), Context())
    pid = engine._process.pid
    fire(after_the_request(engine), lambda: os.kill(pid, signal.SIGKILL))
    with pytest.raises(EngineDied) as raised:
        engine._ask("hang", ())
    assert str(raised.value).startswith("The engine was killed")
    assert engine.starts == 2
    assert engine.simplify(tree("1 + 1"), Context()).text == "2"


def test_a_bug_in_the_engine_costs_the_answer_and_not_the_worker(remote):
    """The shared worker, since the point is that it is still there afterwards."""
    starts = remote.starts
    with pytest.raises(Exception) as raised:
        remote._ask("no_such_method", ())
    assert "no_such_method" in str(raised.value)
    assert remote.starts == starts
    assert remote.simplify(tree("1 + 1"), Context()).text == "2"


def refuses_to_start(connection, cap):
    """A worker that dies before the handshake.

    A module-level function because a spawned child is given its target by
    name: it imports this module and looks the name up, which a closure or a
    local would not survive.
    """
    raise SystemExit(1)


def test_a_worker_that_will_not_start_puts_the_proxy_down(monkeypatch, mortal):
    monkeypatch.setattr(worker_module, "serve", refuses_to_start)
    engine = mortal()
    with pytest.raises(EngineDied):
        engine.simplify(tree("1 + 1"), Context())
    assert until(lambda: engine._down is not None)
    assert engine.starts == STARTS_BEFORE_DOWN
    # And it stays down: a command asked for while it is down tries exactly one
    # more spawn, and nothing else spawns at all. There is no timer, no backoff
    # and no retry thread in the proxy - `starts` moves only where a command
    # moves it - so the count after the next command is the whole of the
    # evidence that nothing spun behind the user's back in between.
    with pytest.raises(EngineDied):
        engine.simplify(tree("1 + 1"), Context())
    assert engine.starts == STARTS_BEFORE_DOWN + 1


# -- a hostile expression, never run without a cap ---------------------------


def test_a_hostile_power_comes_back_inert_and_the_session_stands(remote):
    """`10^10^10` is the input that used to take the app and the worksheet.

    Sympy builds `Integer**Integer` as the power is constructed, so converting
    this one at all means ten billion digits and four gigabytes; the engine
    declines to build it and hands back the power as it was written. The worker
    is capped because a hostile input is never run without a cap, not because
    this one still needs saving - and the assertion is that it is not needed:
    the worker is the same worker afterwards, and the session on this side is
    intact and computable with.
    """
    starts = remote.starts
    session = Session(runner=remote)
    session.author("10^10^10")
    assert session.simplify("#1").text == "10^10000000000"
    session.author("2 + 3")
    assert session.simplify("#3").text == "5"
    assert remote.starts == starts
