"""Test-wide configuration, and the net under every answer.

Textual's pilot sleeps 20 ms twice per simulated keystroke, comparing wall clock
against process time to guess whether the app has settled.  The guess costs far more
than the app it watches, and buys little: it is a wall-clock heuristic, and a process
that a busy machine has descheduled looks idle to it, so what it settles is not
something a test can rest on either way.  Shrink the granularity so a test costs what
its work costs.

What pilot.press does promise is a drain: every widget's message queue emptied once.
A drain is not a layout pass.  Showing a widget or setting how tall it stands writes a
style, and the size that follows is assigned later, on the screen's own idle, so a test
that reads `size` or `region` has to wait for the layout itself - `laid_out` in
`screen.py` is how.
"""

import sys

import pytest
import textual._wait as _wait

# Read before assigning, so a rename upstream fails loudly rather than quietly
# creating an attribute nobody looks at.
assert _wait.SLEEP_GRANULARITY and _wait.SLEEP_IDLE
_wait.SLEEP_GRANULARITY = 0.002
_wait.SLEEP_IDLE = _wait.SLEEP_GRANULARITY / 20.0

#: Where the engine keeps its record of answers author notation could not say.
#: Named rather than imported: a test that never loads the engine should not
#: pay for sympy, and one that never loads it produced no answer either.
_FROM_SYMPY = "rederive.engine.from_sympy"


def _record():
    """The engine's record of unreadable answers, or None if it is not loaded."""
    module = sys.modules.get(_FROM_SYMPY)
    return None if module is None else module.UNREADABLE


@pytest.fixture(autouse=True)
def answers_stay_readable():
    """No test may produce an answer that is not author notation.

    A result reaches the worksheet by being printed and read back, and one the
    reader cannot take is kept as inert text: legible, and dead. That is the
    right thing to do with a head nobody foresaw and the wrong thing to leave
    unnoticed - `RootSum` was a wrong answer for as long as it was, and
    `AccumBounds` an unusable one, because a dead result and a live one look
    alike from the outside.

    So the question is asked of every test rather than of a corpus of its own.
    What the suite already runs is what a corpus would hold - the shipped
    library and the parser's cases go through Simplify in
    `test_simplify_corpus.py`, and Approximate, Factor, Expand and soLve have
    files of their own - and asking here costs a list length per test instead
    of a second run of all of it.

    A test that means to produce one asks for `unreadable_answers` and reads
    the record itself.
    """
    record = _record()
    if record is None:
        # Nothing here can answer anything, and a record that appeared while
        # the test ran is one whose earlier entries are not this test's.
        yield
        return
    seen = len(record)
    yield
    assert record[seen:] == [], "an answer the notation cannot say"


@pytest.fixture
def unreadable_answers(monkeypatch):
    """An empty record of unsayable answers, for a test that expects some.

    Empty rather than shared, since the engine notes each text once per process
    and a text some earlier test met would leave nothing to see.
    """
    import rederive.engine.from_sympy as from_sympy

    record = []
    monkeypatch.setattr(from_sympy, "UNREADABLE", record)
    return record
