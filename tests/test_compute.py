"""The screen while a command computes: modal, ticking, and abortable.

These tests are about timing, so the engine blocks until the test releases it instead of
depending on how long sympy takes. Once released it returns the real engine's answer, so
the worksheet ends up as it would after a real command.
"""

import asyncio
import threading
import time

import pytest
from screen import entries, message, prompt

from rederive.engine import computing
from rederive.engine.client import EngineAborted
from rederive.model.session import Session
from rederive.ui import app as app_module
from rederive.ui.app import (
    CLOCK_AFTER,
    ENTER_TO_DERIVE,
    MODE_COMPUTE,
    MODE_MENU,
    SIMPLIFYING,
    RederiveApp,
    _elapsed,
    _whole,
)

#: How long to wait for a thread to reach a given point. Long enough for a loaded machine,
#: short enough that a deadlock fails quickly.
PATIENCE = 2.0


class Blocking:
    """An engine that answers only when the test releases it.

    It implements the five calls a session makes plus the `abort` the app checks for to
    decide on compute mode. Like `RemoteEngine`, it waits in a thread from the loop's
    pool, so the event loop stays free while a call is pending.
    """

    def __init__(self) -> None:
        #: Set while a call is waiting, so a test can be sure one has started.
        self.running = threading.Event()
        self._go = threading.Event()
        self._aborted = False

    def release(self) -> None:
        """Let the pending call answer."""
        self._go.set()

    def abort(self) -> None:
        """Abort the pending call, as Esc does."""
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

    async def _held(self):
        await asyncio.to_thread(self._wait)

    async def simplify(self, node, context, state=None):
        await self._held()
        return computing.simplify(node, context, state)

    async def factor(self, node, context, amount=None, variables=(), state=None):
        await self._held()
        return computing.factor(node, context, amount, variables, state)

    async def expand(self, node, context, amount=None, variables=(), state=None):
        await self._held()
        return computing.expand(node, context, amount, variables, state)

    async def expression_variables(self, node, context=None, written=False):
        await self._held()
        return computing.expression_variables(node, context, written)

    async def relation_of(self, node, context=None):
        await self._held()
        return computing.relation_of(node, context)


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
        # Rounds to ten, so it is written in the whole-seconds format.
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
    """Whether the message line shows a refusal rather than the prompt or the clock."""
    return said != ENTER_TO_DERIVE and not said.startswith(SIMPLIFYING)


async def settle(pilot, condition, patience=PATIENCE):
    """Wait until `condition` holds or time runs out, and return whether it held."""
    deadline = time.monotonic() + patience
    while time.monotonic() < deadline:
        if condition():
            return True
        await pilot.pause(0.002)
    return condition()


async def author(pilot, text):
    await pilot.press("a")
    await pilot.press(*text)
    await pilot.press("enter")


async def simplify(pilot, runner):
    """Simplify the highlighted entry and wait until the computation is running.

    The keys are pressed in separate calls; otherwise the second would arrive before the
    prompt opened by the first has focus.
    """
    await pilot.press("s")
    await pilot.press("enter")
    assert await settle(pilot, runner.running.is_set)


async def test_a_computation_holds_the_screen_and_says_how_long_it_has(
    app, runner, monkeypatch
):
    """The clock is driven by moving `app.started` back rather than by waiting.

    Waiting out the real delays would make the test slow and dependent on the machine.
    Moving the start back also proves the displayed figure comes from the running clock.
    """
    monkeypatch.setattr(app_module, "TICK", 0.01)
    async with app.run_test() as pilot:
        await author(pilot, "1+1")
        await simplify(pilot, runner)
        assert app.mode == MODE_COMPUTE
        # No clock for the first second, so a command that answers at once doesn't flash
        # one; the message line still shows the prompt's text.
        await pilot.pause(0.05)
        assert message(app) == ENTER_TO_DERIVE
        # The clock appears after a second.
        app.started -= CLOCK_AFTER
        assert await settle(pilot, lambda: message(app).startswith(SIMPLIFYING))
        assert "ESC aborts" in message(app)
        # Once shown, the clock keeps updating because the event loop is free. Moving back
        # a full minute yields a reading the test's own running time couldn't produce.
        assert "1m" not in message(app)
        app.started -= 60.0
        assert await settle(pilot, lambda: "1m" in message(app))
        runner.release()
        assert await settle(pilot, lambda: app.mode == MODE_MENU)
        assert entries(app) == ["1+1", "2"]
        assert message(app).startswith("Compute time:")


async def test_a_command_that_answers_at_once_leaves_the_message_line_alone(
    app, runner
):
    """Most commands answer at once, and a clock would only flicker.

    The prompt's message stays until the wait is worth reporting, and is replaced by the
    compute time when the answer arrives.
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
    """The command parses the line, so the refusal comes back from the thread.

    The engine is never called, and the prompt must leave compute mode intact, with its
    text and focus.
    """
    async with app.run_test() as pilot:
        await pilot.press("s")
        await pilot.press(*"1+")
        await pilot.press("enter")
        assert await settle(pilot, lambda: _refused(message(app)))
        assert not runner.running.is_set()
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
def demo(tmp_path, monkeypatch):
    """A demo file in the current directory, so the prompt accepts its bare name."""
    path = tmp_path / "show.dmo"
    path.write_text("; adds two numbers\n2 + 3\n\n; and a symbolic one\n(x+1)^2\n")
    monkeypatch.chdir(tmp_path)
    return path


async def test_a_demonstration_waits_on_the_step_it_dispatched(app, runner, demo):
    """Each step is simplified on the thread, so it can be aborted like any command.

    The next comment is not shown until the step's answer is in.
    """
    async with app.run_test() as pilot:
        await pilot.press("t", "d", "o")
        await pilot.press(*demo.name)
        await pilot.press("enter")
        assert await settle(pilot, runner.running.is_set)
        assert app.mode == MODE_COMPUTE
        assert entries(app) == ["2+3"]
        runner.release()
        waiting = "Press any key to continue, ESC to stop"
        assert await settle(pilot, lambda: message(app) == waiting)
        assert entries(app) == ["2+3", "5"]


async def test_an_abort_ends_the_demonstration_where_it_stands(app, runner, demo):
    async with app.run_test() as pilot:
        await pilot.press("t", "d", "o")
        await pilot.press(*demo.name)
        await pilot.press("enter")
        assert await settle(pilot, runner.running.is_set)
        await pilot.press("escape")
        assert await settle(pilot, lambda: app.mode == MODE_MENU)
        assert message(app).startswith("Aborted after")
        assert entries(app) == ["2+3"]
        # The demo is suspended, not lost: naming the file again resumes it.
        await pilot.press("t", "d", "o")
        await pilot.press(*demo.name)
        await pilot.press("enter")
        assert await settle(pilot, runner.running.is_set)
        runner.release()
        waiting = "Press any key to continue, ESC to stop"
        assert await settle(pilot, lambda: message(app) == waiting)
        assert entries(app) == ["2+3", "(x+1)^2", "(x + 1)^2"]


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
