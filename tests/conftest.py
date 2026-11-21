"""Test-wide configuration, a check that every answer stays readable, and a time limit.

Textual's pilot sleeps 20 ms twice per simulated keystroke, using a wall-clock heuristic
to guess whether the app has settled. The guess is expensive and unreliable anyway (a
process descheduled on a busy machine looks idle), so shrink the granularity to make a
test cost only what its work costs.

What pilot.press does guarantee is a drain: every widget's message queue emptied once.
That is not a layout pass. Showing a widget or setting its height writes a style, and
the resulting size is assigned later, when the screen is idle, so a test that reads
`size` or `region` must wait for layout - `laid_out` in `screen.py` does that.
"""

import faulthandler
import os
import signal
import sys
import threading

import pytest
import textual._wait as _wait

# Read before assigning, so an upstream rename fails loudly rather than silently creating
# an unused attribute.
assert _wait.SLEEP_GRANULARITY and _wait.SLEEP_IDLE
_wait.SLEEP_GRANULARITY = 0.002
_wait.SLEEP_IDLE = _wait.SLEEP_GRANULARITY / 20.0


def pytest_configure(config):
    """Run the processes that run the tests with string hashing seeded as the engine's
    worker seeds it (see `rederive.engine.remote`), so that the work budgets of
    optional steps cut off what they cut off in the app. A single process (`-n0`) has
    hashed strings since it started and keeps its own seed."""
    os.environ["PYTHONHASHSEED"] = "0"


#: Module holding the engine's record of answers author notation could not express. Named
#: rather than imported so a test that never loads the engine doesn't pay for sympy (and
#: such a test produced no answers anyway).
_FROM_SYMPY = "rederive.engine.from_sympy"


def _record():
    """The engine's record of unreadable answers, or None if it is not loaded."""
    module = sys.modules.get(_FROM_SYMPY)
    return None if module is None else module.UNREADABLE


@pytest.fixture(autouse=True)
def answers_stay_readable():
    """Fail any test that produces an answer not expressible in author notation.

    A head the notation has no name for is written as inert text. That's the right
    fallback for an unforeseen head, but it must not go unnoticed: `RootSum` and
    `AccumBounds` both slipped through this way, since a dead result looks like a live
    one.

    Checking every test covers what a dedicated corpus would (the shipped library and
    parser cases already run through Simplify in `test_simplify_corpus.py`, and
    Approximate, Factor, Expand and soLve have their own files) at the cost of a list
    length per test.

    A test that expects unreadable answers requests `unreadable_answers` and inspects
    the record itself.
    """
    record = _record()
    if record is None:
        # Nothing can produce answers yet, and a record that appears during the test
        # may hold entries that aren't this test's.
        yield
        return
    seen = len(record)
    yield
    assert record[seen:] == [], "an answer the notation cannot say"


#: The seconds a test may run. The engine has no time limit of its own: what the user
#: asks for runs until it ends or is interrupted. So a test that would run on fails at
#: this limit instead of hanging the suite. Far above the longest test, also on a busy
#: machine.
TEST_SECONDS = 300

#: The seconds after `TEST_SECONDS` at which a test that hasn't stopped (stuck in code
#: that doesn't see the interruption) ends its whole process, which fails it too.
_GRACE_SECONDS = 60


class TimedOut(BaseException):
    """A test ran past `TEST_SECONDS`. A `BaseException`, so that no handler in the
    engine that turns a failure into an unevaluated call takes it for one."""


def _timed_out(signum, frame):
    # Raised again a little later, in case a handler swallowed it.
    signal.setitimer(signal.ITIMER_REAL, 1)
    raise TimedOut(f"the test ran longer than {TEST_SECONDS} s")


@pytest.fixture(autouse=True)
def limited_time():
    """Fail a test that runs longer than `TEST_SECONDS`, where an alarm can interrupt
    it (the main thread of a POSIX process); past the grace period, end the process."""
    if not hasattr(signal, "setitimer") or (
        threading.current_thread() is not threading.main_thread()
    ):
        yield
        return
    previous = signal.signal(signal.SIGALRM, _timed_out)
    signal.setitimer(signal.ITIMER_REAL, TEST_SECONDS)
    faulthandler.dump_traceback_later(TEST_SECONDS + _GRACE_SECONDS, exit=True)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def unreadable_answers(monkeypatch):
    """A fresh, empty record of unreadable answers, for a test that expects some.

    Not the shared one, since the engine records each text only once per process and an
    earlier test may already have recorded it.
    """
    import rederive.engine.from_sympy as from_sympy

    record = []
    monkeypatch.setattr(from_sympy, "UNREADABLE", record)
    return record
