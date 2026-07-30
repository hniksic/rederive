"""The engine at arm's length: the answers, and every way the worker can die.

Spawning a worker costs about a second of sympy import, so the tests that only
want an answer share one. The tests that want a death each get their own,
since a death is what they are about.

The two private methods in the worker's table are what the abort and the cap
are tested with. Timing a real computation to be slow enough or hungry enough
would make these tests about sympy's mood rather than about the recovery path.
"""

import os
import signal
import threading
import time

import pytest

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


@pytest.fixture(scope="module")
def remote():
    """One worker for every test that only wants an answer out of it."""
    engine = RemoteEngine()
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
    assert remote.factor(tree("x^2 - 1"), Context()).text == "(x - 1)*(x + 1)"


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
    threading.Timer(0.2, engine.abort).start()
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
    engine.start()
    threading.Timer(0.2, engine.abort).start()
    with pytest.raises(EngineAborted):
        engine._ask("hang", ())
    assert engine.simplify(tree("1 + 1"), Context()).text == "2"


def test_a_worker_over_the_cap_is_taken_away(mortal):
    engine = mortal(cap=SMALL_CAP, period=0.05)
    engine.start()
    with pytest.raises(EngineMemoryExceeded) as raised:
        engine._ask("allocate", (2 * SMALL_CAP >> 20,))
    assert "300M" in str(raised.value)
    assert engine.simplify(tree("1 + 1"), Context()).text == "2"


def test_a_worker_killed_from_outside_is_replaced(mortal):
    engine = mortal()
    engine.start()
    engine.simplify(tree("1 + 1"), Context())
    pid = engine._process.pid
    threading.Timer(0.2, lambda: os.kill(pid, signal.SIGKILL)).start()
    with pytest.raises(EngineDied) as raised:
        engine._ask("hang", ())
    assert str(raised.value).startswith("The engine was killed")
    assert engine.starts == 2
    assert engine.simplify(tree("1 + 1"), Context()).text == "2"


def test_a_bug_in_the_engine_costs_the_answer_and_not_the_worker(mortal):
    engine = mortal()
    engine.start()
    with pytest.raises(Exception) as raised:
        engine._ask("no_such_method", ())
    assert "no_such_method" in str(raised.value)
    assert engine.starts == 1
    assert engine.simplify(tree("1 + 1"), Context()).text == "2"


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
    # And it stays down: nothing spins behind the user's back.
    time.sleep(0.5)
    assert engine.starts == STARTS_BEFORE_DOWN
    # A command asked for while it is down tries exactly one more spawn.
    with pytest.raises(EngineDied):
        engine.simplify(tree("1 + 1"), Context())
    assert engine.starts == STARTS_BEFORE_DOWN + 1


# -- a hostile expression, never run without a cap ---------------------------


def test_a_hostile_power_comes_back_inert_and_the_session_stands(mortal):
    """`10^10^10` is the input that used to take the app and the worksheet.

    Sympy builds `Integer**Integer` as the power is constructed, so converting
    this one at all means ten billion digits and four gigabytes; the engine
    declines to build it and hands back the power as it was written. The cap
    is here because a hostile input is never run without one, not because this
    one still needs saving - and the assertion is that it is not needed: the
    worker lives, and the session on this side is intact and computable with.
    """
    engine = mortal(cap=200 << 20, period=0.05)
    session = Session(runner=engine)
    session.author("10^10^10")
    assert session.simplify("#1").text == "10^10000000000"
    session.author("2 + 3")
    assert session.simplify("#3").text == "5"
