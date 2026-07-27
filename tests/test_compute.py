"""The screen while a command computes: modal, ticking, and abortable.

The engine under these is one that waits to be let go rather than one that
works, because everything here is about timing and real sympy is the wrong
clock to read it by. What it answers with, once it is let go, is what the real
engine answers, so what lands on the worksheet is what a real command would
have put there.
"""

import threading
import time

import pytest
from screen import entries, message, prompt

from rederive import engine
from rederive.engine import EngineAborted
from rederive.model.session import Session
from rederive.ui.app import (
    ENTER_TO_DERIVE,
    MODE_COMPUTE,
    MODE_MENU,
    SIMPLIFYING,
    RederiveApp,
    _elapsed,
    _whole,
)

#: How long anything here waits for a thread to get where it is going.
PATIENCE = 10.0


class Blocking:
    """An engine that answers only once the test says so.

    It satisfies the five calls a session makes and the `abort` the app looks
    for, which is all that decides whether the app runs a command on a thread.
    """

    def __init__(self) -> None:
        #: Set while a call is waiting, so a test can be sure one has started.
        self.running = threading.Event()
        self._go = threading.Event()
        self._aborted = False

    def release(self) -> None:
        """Let the call in hand answer."""
        self._go.set()

    def abort(self) -> None:
        """What Esc reaches: the call in hand fails as an aborted one does."""
        self._aborted = True
        self._go.set()

    def _wait(self) -> None:
        self.running.set()
        self._go.wait(PATIENCE)
        self.running.clear()
        self._go.clear()
        if self._aborted:
            self._aborted = False
            raise EngineAborted("Aborted")

    def simplify(self, node, context, state=None):
        self._wait()
        return engine.simplify(node, context, state)

    def approx(self, node, context, digits=None, state=None):
        self._wait()
        return engine.approx(node, context, digits, state)

    def factor(self, node, context, amount=None, variables=(), state=None):
        self._wait()
        return engine.factor(node, context, amount, variables, state)

    def expand(self, node, context, amount=None, variables=(), state=None):
        self._wait()
        return engine.expand(node, context, amount, variables, state)

    def expression_variables(self, node, context=None):
        self._wait()
        return engine.expression_variables(node, context)


@pytest.fixture
def runner():
    return Blocking()


@pytest.fixture
def app(runner):
    return RederiveApp(Session(runner=runner))


@pytest.mark.parametrize(
    ("seconds", "written"),
    [
        (0, "0.0s"),
        (0.04, "0.0s"),
        (3.14, "3.1s"),
        (9.94, "9.9s"),
        # Rounded to ten, so it is written the way ten seconds is written.
        (9.96, "10s"),
        (42.4, "42s"),
        (60, "1m 0s"),
        (608, "10m 8s"),
        (3600, "1h 0m 0s"),
        (7808, "2h 10m 8s"),
    ],
    ids=str,
)
def test_an_elapsed_time_is_written_as_coarsely_as_it_can_be(seconds, written):
    assert _elapsed(seconds) == written


@pytest.mark.parametrize(
    ("seconds", "written"),
    [
        (0, "0s"),
        (3.14, "3s"),
        (9.94, "10s"),
        (42.4, "42s"),
        (60, "1m 0s"),
        (608, "10m 8s"),
        (7808, "2h 10m 8s"),
    ],
    ids=str,
)
def test_a_running_clock_counts_whole_seconds(seconds, written):
    assert _whole(seconds) == written


def _refused(said):
    """Whether the message line has stopped asking or computing and said no."""
    return said != ENTER_TO_DERIVE and not said.startswith(SIMPLIFYING)


async def settle(pilot, condition, patience=PATIENCE):
    """Wait for something a thread is doing, and say whether it happened."""
    deadline = time.monotonic() + patience
    while time.monotonic() < deadline:
        if condition():
            return True
        await pilot.pause(0.01)
    return condition()


async def author(pilot, text):
    await pilot.press("a")
    await pilot.press(*text)
    await pilot.press("enter")


async def simplify(pilot, runner):
    """Ask for a Simplify of the entry offered, and wait until it is running.

    One press per call: the second key would otherwise be delivered before the
    prompt the first one puts up has taken the focus.
    """
    await pilot.press("s")
    await pilot.press("enter")
    assert await settle(pilot, runner.running.is_set)


async def test_a_computation_holds_the_screen_and_says_how_long_it_has(app, runner):
    async with app.run_test() as pilot:
        await author(pilot, "1+1")
        await simplify(pilot, runner)
        assert app.mode == MODE_COMPUTE
        # Nothing is said for the first second: the message line still asks
        # what the prompt asked, so a command that answers at once passes
        # without a clock flashing on the way.
        assert message(app) == ENTER_TO_DERIVE
        assert await settle(pilot, lambda: message(app).startswith(SIMPLIFYING))
        assert "ESC aborts" in message(app)
        # And once it is up the clock moves, which is the whole point of the
        # event loop being free.
        first = message(app)
        assert await settle(pilot, lambda: message(app) != first)
        runner.release()
        assert await settle(pilot, lambda: app.mode == MODE_MENU)
        assert entries(app) == ["1+1", "2"]
        assert message(app).startswith("Compute time:")


async def test_a_command_that_answers_at_once_leaves_the_message_line_alone(
    app, runner
):
    """Almost every command does, and the clock would only flicker.

    The line the prompt left up stands until there is a wait worth reporting,
    and the compute time replaces it when the answer lands.
    """
    async with app.run_test() as pilot:
        await author(pilot, "1+1")
        await simplify(pilot, runner)
        assert message(app) == ENTER_TO_DERIVE
        runner.release()
        assert await settle(pilot, lambda: app.mode == MODE_MENU)
        assert message(app).startswith("Compute time:")


async def test_no_key_but_escape_means_anything_while_one_runs(app, runner):
    async with app.run_test() as pilot:
        await author(pilot, "1+1")
        await simplify(pilot, runner)
        await pilot.press("a", "tab", "down", "enter", "q")
        assert app.mode == MODE_COMPUTE
        assert entries(app) == ["1+1"]
        runner.release()
        assert await settle(pilot, lambda: app.mode == MODE_MENU)
        assert entries(app) == ["1+1", "2"]


async def test_escape_aborts_and_the_worksheet_is_as_the_command_found_it(app, runner):
    async with app.run_test() as pilot:
        await author(pilot, "1+1")
        await simplify(pilot, runner)
        await pilot.press("escape")
        assert await settle(pilot, lambda: app.mode == MODE_MENU)
        assert message(app).startswith("Aborted after")
        assert entries(app) == ["1+1"]


async def test_a_command_after_an_abort_works(app, runner):
    async with app.run_test() as pilot:
        await author(pilot, "1+1")
        await simplify(pilot, runner)
        await pilot.press("escape")
        assert await settle(pilot, lambda: app.mode == MODE_MENU)
        await simplify(pilot, runner)
        runner.release()
        assert await settle(pilot, lambda: app.mode == MODE_MENU)
        assert entries(app) == ["1+1", "2"]
        assert message(app).startswith("Compute time:")


async def test_a_line_that_will_not_read_stays_up_to_be_corrected(app, runner):
    """The line is read by the command, so the refusal comes back off the thread.

    Nothing reaches the engine - the runner is never called - and the prompt
    has to come out of compute mode as its own again, cursor and all.
    """
    async with app.run_test() as pilot:
        await pilot.press("s")
        await pilot.press(*"1+")
        await pilot.press("enter")
        assert await settle(pilot, lambda: _refused(message(app)))
        assert not runner.running.is_set()
        # The prompt is still there, still the user's, and still focused.
        assert prompt(app)[1] == "1+"
        assert app.focused is app.query_one("#prompt-input")
        assert entries(app) == []


async def test_control_enter_simplifies_what_it_entered_on_the_thread(app, runner):
    async with app.run_test() as pilot:
        await pilot.press("a")
        await pilot.press(*"2+3")
        await pilot.press("ctrl+j")
        assert await settle(pilot, runner.running.is_set)
        assert app.mode == MODE_COMPUTE
        assert entries(app) == ["2+3"]
        runner.release()
        assert await settle(pilot, lambda: app.mode == MODE_MENU)
        assert entries(app) == ["2+3", "5"]


@pytest.fixture
def demo(tmp_path):
    path = tmp_path / "show.dmo"
    path.write_text("; adds two numbers\n2 + 3\n\n; and a symbolic one\n(x+1)^2\n")
    return path


async def test_a_demonstration_waits_on_the_step_it_dispatched(app, runner, demo):
    """Each step is authored here and simplified on the thread.

    So the step is as abortable as any other command, and the comment does not
    go up until its answer is in.
    """
    async with app.run_test() as pilot:
        await pilot.press("t", "d")
        await pilot.press(*str(demo))
        await pilot.press("enter")
        assert await settle(pilot, runner.running.is_set)
        assert app.mode == MODE_COMPUTE
        assert entries(app) == ["2 + 3"]
        runner.release()
        assert await settle(pilot, lambda: message(app) == "Press any key to continue")
        assert entries(app) == ["2 + 3", "5"]


async def test_an_abort_ends_the_demonstration_where_it_stands(app, runner, demo):
    async with app.run_test() as pilot:
        await pilot.press("t", "d")
        await pilot.press(*str(demo))
        await pilot.press("enter")
        assert await settle(pilot, runner.running.is_set)
        await pilot.press("escape")
        assert await settle(pilot, lambda: app.mode == MODE_MENU)
        assert message(app).startswith("Aborted after")
        assert entries(app) == ["2 + 3"]
        # Suspended rather than lost: naming the file again picks it up.
        await pilot.press("t", "d")
        await pilot.press(*str(demo))
        await pilot.press("enter")
        assert await settle(pilot, runner.running.is_set)
        runner.release()
        assert await settle(pilot, lambda: message(app) == "Press any key to continue")
        assert entries(app) == ["2 + 3", "(x+1)^2", "(x + 1)^2"]


async def test_control_enter_aborted_leaves_what_it_entered_standing(app, runner):
    async with app.run_test() as pilot:
        await pilot.press("a")
        await pilot.press(*"2+3")
        await pilot.press("ctrl+j")
        assert await settle(pilot, runner.running.is_set)
        await pilot.press("escape")
        assert await settle(pilot, lambda: app.mode == MODE_MENU)
        assert entries(app) == ["2+3"]
        assert message(app).startswith("Aborted after")
