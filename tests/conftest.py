"""Test-wide configuration, and a check that every answer stays readable.

Textual's pilot sleeps 20 ms twice per simulated keystroke, using a wall-clock heuristic
to guess whether the app has settled. The guess is expensive and unreliable anyway (a
process descheduled on a busy machine looks idle), so shrink the granularity to make a
test cost only what its work costs.

What pilot.press does guarantee is a drain: every widget's message queue emptied once.
That is not a layout pass. Showing a widget or setting its height writes a style, and
the resulting size is assigned later, when the screen is idle, so a test that reads
`size` or `region` must wait for layout - `laid_out` in `screen.py` does that.
"""

import sys

import pytest
import textual._wait as _wait

# Read before assigning, so an upstream rename fails loudly rather than silently creating
# an unused attribute.
assert _wait.SLEEP_GRANULARITY and _wait.SLEEP_IDLE
_wait.SLEEP_GRANULARITY = 0.002
_wait.SLEEP_IDLE = _wait.SLEEP_GRANULARITY / 20.0

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

    A result reaches the worksheet by being printed and parsed back, and one the parser
    can't read is kept as inert text. That's the right fallback for an unforeseen head,
    but it must not go unnoticed: `RootSum` and `AccumBounds` both slipped through this
    way, since a dead result looks like a live one.

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
