"""Undo and redo: what a step puts back, what makes a step, and the keys.

The session tests drive a `History` as the app does, one `begin` per command, so that
what makes a step is tested along with what a step holds. The app tests check the keys:
where they apply, what they say, and that they leave lines and plot windows alone.
"""

import pytest
from screen import entries, message, prompt

from rederive.model import undo
from rederive.model.session import Session
from rederive.model.undo import History
from rederive.model.expr import Kind
from rederive.syntax import DeriveSyntaxError, InputMode, parse_expression
from rederive.ui.app import (
    NOTHING_TO_REDO,
    NOTHING_TO_UNDO,
    RederiveApp,
)


@pytest.fixture
def session():
    return Session()


@pytest.fixture
def history():
    return History()


def labels(session):
    """Each entry's label and text, as the work area shows them."""
    return [f"#{entry.number}: {entry.text}" for entry in session.entries]


def undone(history):
    """Undo the last step, and say what command it was, or None with none to undo."""
    step = history.undo()
    return None if step is None else step.description


def redone(history):
    """Redo the last undone step, and say what command it was, or None with none."""
    step = history.redo()
    return None if step is None else step.description


def run(history, session, name, *calls):
    """Run one command named `name`, made of `calls` on the session."""
    history.begin(session, name)
    for call in calls:
        call()


def applied(history, session, name, **values):
    """Commit a dialog of settings named `name`, setting `values`."""
    run(history, session, name, lambda: session.settings.apply(values))


def authored(history, session, *texts):
    """Author each of `texts` as a command of its own."""
    for text in texts:
        run(history, session, "Author", lambda text=text: session.author(text))


# -- what an undo puts back ----------------------------------------------------


def test_undoing_an_author_takes_the_entry_away_and_redoing_brings_it_back(
    session, history
):
    authored(history, session, "x", "y")
    assert undone(history) == "Author"
    assert labels(session) == ["#1: x"]
    assert redone(history) == "Author"
    assert labels(session) == ["#1: x", "#2: y"]


def test_the_next_entry_takes_the_label_an_undo_freed(session, history):
    authored(history, session, "x", "y")
    undone(history)
    authored(history, session, "z")
    assert labels(session) == ["#1: x", "#2: z"]


def test_undoing_a_remove_puts_the_block_back_and_the_buffer_as_it_was(
    session, history
):
    authored(history, session, "x", "y", "z")
    run(history, session, "Remove", lambda: session.remove(1, 1))
    run(history, session, "Remove", lambda: session.remove(2, 3))
    undone(history)
    assert labels(session) == ["#2: y", "#3: z"]
    # The buffer holds what the first Remove took, so Unremove still restores that.
    assert session.unremove() == 1
    assert labels(session) == ["#2: y", "#3: z", "#1: x"]


def test_undoing_an_unremove_takes_the_restored_entries_out_again(session, history):
    authored(history, session, "x", "y")
    run(history, session, "Remove", lambda: session.remove(1, 1))
    run(history, session, "Unremove", lambda: session.unremove())
    assert undone(history) == "Unremove"
    assert labels(session) == ["#2: y"]


def test_undoing_a_move_and_a_renumber_restores_the_order_and_the_labels(
    session, history
):
    authored(history, session, "x", "y", "#1 + 1")
    run(history, session, "Move", lambda: session.move_block(None, 1, 1))
    run(history, session, "Manage Renumber", session.renumber)
    assert labels(session) == ["#1: y", "#2: #3 + 1", "#3: x"]
    undone(history)
    assert labels(session) == ["#2: y", "#3: #1 + 1", "#1: x"]
    undone(history)
    assert labels(session) == ["#1: x", "#2: y", "#3: #1 + 1"]


def test_undoing_an_annotation_restores_the_old_one(session, history):
    authored(history, session, "x")
    run(history, session, "Manage Annotate", lambda: session.annotate(1, "mine"))
    undone(history)
    assert session.entries[0].annotation == "User"


def test_undoing_a_clear_brings_the_worksheet_and_its_numbering_back(
    session, history
):
    authored(history, session, "x", "y")
    run(history, session, "Transfer Clear Expressions", session.clear_expressions)
    undone(history)
    authored(history, session, "z")
    assert labels(session) == ["#1: x", "#2: y", "#3: z"]


def test_an_undone_definition_is_gone_and_a_redone_one_is_back(session, history):
    authored(history, session, "a := 7", "F(x) := x^2", "t :epsilon Integer")
    for _ in range(3):
        undone(history)
    assert session.assignments == {} and session.functions == {}
    assert session.domains == {}
    assert session.state.lookup("F") is None
    # `F(3)` reads as a product again, as before the definition.
    assert parse_expression("F(3)", session.state).node.kind is Kind.PRODUCT
    for _ in range(3):
        redone(history)
    assert "a" in session.assignments and "F" in session.functions
    assert session.state.is_function("F")


def test_undoing_a_clear_of_the_variables_and_functions_brings_them_back(
    session, history
):
    authored(history, session, "a := 7", "F(x) := x^2")
    run(history, session, "Transfer Clear Variables", session.clear_variables)
    run(history, session, "Transfer Clear Functions", session.clear_functions)
    undone(history)
    undone(history)
    assert "a" in session.assignments and "F" in session.functions
    assert not session.state.withdrawn


async def test_one_command_that_enters_several_expressions_is_one_step(
    session, history
):
    authored(history, session, "x^2 = 1")
    history.begin(session, "Solve")
    solved = await session.solve("#1")
    assert len(solved) == 2
    assert undone(history) == "Solve"
    assert labels(session) == ["#1: x^2 = 1"]


async def test_an_undone_solve_gives_its_arbitrary_values_back(session, history):
    authored(history, session, "x = x")
    history.begin(session, "Solve")
    [first] = await session.solve("#1")
    undone(history)
    history.begin(session, "Solve")
    [again] = await session.solve("#1")
    assert "@1" in first.text and again.text == first.text


async def test_undoing_a_simplify_restores_the_selection_it_found(session, history):
    authored(history, session, "(1 + 1)*x")
    assert session.move_right()
    route = session.route
    history.begin(session, "Simplify")
    await session.simplify("#1")
    undone(history)
    assert session.selected == 0 and session.route == route


def test_a_load_is_one_step_and_the_file_it_named_stays(session, history, tmp_path):
    path = tmp_path / "three.mth"
    path.write_text("a\nb\nc\n")
    authored(history, session, "x")
    run(history, session, "Transfer Load Derive", lambda: session.load(path))
    assert len(session.entries) == 3
    assert undone(history) == "Transfer Load Derive"
    assert labels(session) == ["#1: x"]
    assert session.file == path and session.source == path


def test_undoing_past_a_save_leaves_the_next_save_going_to_its_file(
    session, history, tmp_path
):
    path = tmp_path / "kept.mth"
    authored(history, session, "x")
    run(history, session, "Transfer Save Derive", lambda: session.save(path))
    undone(history)
    assert labels(session) == []
    assert session.file == path


# -- settings ----------------------------------------------------------------


def test_undoing_an_options_dialog_puts_the_setting_and_its_record_back(
    session, history
):
    def notation():
        session.settings.apply({"Notation": "Mixed"})
        session.record("Notation")

    run(history, session, "Options Notation", notation)
    assert undone(history) == "Options Notation"
    assert session.settings["Notation"] == "Rational"
    assert session.entries == []
    redone(history)
    assert session.settings["Notation"] == "Mixed"
    assert labels(session) == ["#1: Notation := Mixed"]


def test_a_precision_is_put_back_with_the_notation_it_had(session, history):
    session.settings.apply({"Precision": "Approximate"})
    session.settings.apply({"Notation": "Decimal"})
    applied(history, session, "Options Precision", Precision="Exact")
    undone(history)
    # Not the Scientific notation a change to Approximate would bring.
    assert session.settings["Precision"] == "Approximate"
    assert session.settings["Notation"] == "Decimal"


def test_an_undone_input_mode_is_how_the_next_line_reads(session, history):
    authored(history, session, "InputMode := Word")
    undone(history)
    assert session.settings["InputMode"] == "Character"
    assert session.state.input_mode is InputMode.CHARACTER
    assert session.author("ab").text == "a*b"


def test_a_settings_change_alone_is_a_step(session, history):
    applied(history, session, "Options Color Work", WorkColor="Yellow")
    undone(history)
    assert session.settings["WorkColor"] == "White"


def test_undoing_options_defaults_restores_every_setting_and_the_order(
    session, history
):
    authored(history, session, "Notation := Mixed", "VariableOrder := [y, x]")
    order = session.order
    run(history, session, "Options Defaults", session.restore_defaults)
    assert session.settings["Notation"] == "Rational"
    assert undone(history) == "Options Defaults"
    assert session.settings["Notation"] == "Mixed"
    assert session.order == order


def test_a_plot_preference_is_not_part_of_a_step(session, history):
    authored(history, session, "x")
    # As a plot window reports it, while the last command is still the open one.
    session.settings.apply({"PlotMesh": "Solid"})
    undone(history)
    assert session.entries == []
    assert session.settings["PlotMesh"] == "Solid"


# -- several windows -----------------------------------------------------------


def test_undo_goes_back_through_every_windows_commands_in_order(history):
    first = Session()
    second = first.copy()
    run(history, first, "Author", lambda: first.author("x"))
    applied(history, first, "Options Notation", Notation="Mixed")
    run(history, second, "Author", lambda: second.author("y"))
    applied(history, second, "Options Notation", Notation="Decimal")
    assert undone(history) == "Options Notation"
    assert first.settings["Notation"] == "Mixed"
    assert labels(second) == ["#1: y"]
    assert undone(history) == "Author"
    assert labels(second) == []
    assert undone(history) == "Options Notation"
    assert first.settings["Notation"] == "Rational"
    assert undone(history) == "Author"
    assert labels(first) == []
    assert undone(history) is None
    for _ in range(4):
        redone(history)
    assert labels(first) == ["#1: x"] and labels(second) == ["#1: y"]
    assert first.settings["Notation"] == "Decimal"


def test_a_setting_two_windows_changed_comes_back_through_each_value(history):
    first = Session()
    second = first.copy()
    applied(history, first, "Options Color Work", WorkColor="Yellow")
    applied(history, second, "Options Color Work", WorkColor="Green")
    undone(history)
    assert first.settings["WorkColor"] == "Yellow"
    undone(history)
    assert first.settings["WorkColor"] == "White"
    redone(history)
    assert first.settings["WorkColor"] == "Yellow"
    redone(history)
    assert first.settings["WorkColor"] == "Green"


def test_another_window_reads_by_the_input_mode_an_undo_puts_back(history):
    first = Session()
    second = first.copy()
    run(history, first, "Author", lambda: first.author("InputMode := Word"))
    assert second.state.input_mode is InputMode.WORD
    undone(history)
    assert second.state.input_mode is InputMode.CHARACTER
    assert second.author("ab").text == "a*b"


def test_a_closed_windows_steps_are_forgotten(history):
    first = Session()
    second = first.copy()
    run(history, first, "Author", lambda: first.author("x"))
    applied(history, second, "Options Color Work", WorkColor="Yellow")
    run(history, second, "Author", lambda: second.author("y"))
    undone(history)
    # Closing, as a command of the window closed.
    history.begin(second, "Window Close")
    second.discard()
    history.forget(second)
    assert all(step.session is first for step in (*history.undone, *history.redone))
    assert redone(history) is None
    assert undone(history) == "Author"
    assert labels(first) == []
    # The settings are back as that step found them, before the closed window set any.
    assert first.settings["WorkColor"] == "White"
    assert undone(history) is None


def test_steps_before_a_split_stay_with_the_window_split(session, history):
    authored(history, session, "x")
    history.settle()
    copy = session.copy()
    assert undone(history) == "Author"
    assert labels(session) == []
    assert labels(copy) == ["#1: x"]


# -- what makes a step -------------------------------------------------------


def test_a_command_that_changes_nothing_leaves_no_step(session, history):
    authored(history, session, "x")
    run(history, session, "Jump", lambda: session.select_entry(0))
    run(history, session, "Move", lambda: session.move_block(1, 1, 1))
    run(history, session, "Manage Annotate", lambda: session.annotate(1, "User"))
    assert undone(history) == "Author"


def test_a_command_that_fails_leaves_no_step(session, history):
    authored(history, session, "x")
    history.begin(session, "Author")
    with pytest.raises(DeriveSyntaxError):
        session.author("x +* 2")
    history.begin(session, "Remove")
    with pytest.raises(KeyError):
        session.remove(7, 7)
    assert undone(history) == "Author"
    assert undone(history) is None


def test_a_new_change_after_an_undo_ends_the_redo(session, history):
    authored(history, session, "x", "y")
    undone(history)
    authored(history, session, "z")
    assert redone(history) is None
    assert labels(session) == ["#1: x", "#2: z"]


def test_moving_the_highlight_after_an_undo_keeps_the_redo(session, history):
    authored(history, session, "x", "y", "z")
    undone(history)
    session.move_up()
    run(history, session, "Jump", lambda: session.select_entry(0))
    assert redone(history) == "Author"
    assert labels(session) == ["#1: x", "#2: y", "#3: z"]


def test_only_redo_follows_an_undo(session, history):
    authored(history, session, "x")
    assert redone(history) is None
    undone(history)
    assert redone(history) == "Author"
    assert redone(history) is None


def test_the_stacks_keep_the_latest_steps_only(session, history):
    authored(history, session, *(f"x{number}" for number in range(undo.DEPTH + 5)))
    count = 0
    while undone(history) is not None:
        count += 1
    assert count == undo.DEPTH
    assert len(session.entries) == 5
    count = 0
    while redone(history) is not None:
        count += 1
    assert count == undo.DEPTH


def test_a_step_aside_leaves_the_open_command_its_own(session, history):
    authored(history, session, "x")
    history.begin(session, "Author")
    with history.aside(session, "Trace point"):
        session.author("[1, 2]")
    session.author("y")
    assert undone(history) == "Author"
    assert labels(session) == ["#1: x", "#2: [1, 2]"]
    assert undone(history) == "Trace point"
    assert labels(session) == ["#1: x"]


# -- the keys ------------------------------------------------------------------


@pytest.fixture
def app():
    return RederiveApp()


@pytest.fixture
def beeps(app, monkeypatch):
    """How many times the app beeped."""
    heard = []
    monkeypatch.setattr(app, "bell", lambda: heard.append(True))
    return heard


async def author(pilot, text):
    await pilot.press("a")
    await pilot.press(*text)
    await pilot.press("enter")


async def test_ctrl_z_and_ctrl_y_undo_and_redo_at_the_command_menu(app):
    async with app.run_test() as pilot:
        await author(pilot, "x")
        await author(pilot, "y")
        await pilot.press("ctrl+z")
        assert entries(app) == ["x"]
        assert message(app) == "Undid Author"
        await pilot.press("ctrl+y")
        assert entries(app) == ["x", "y"]
        assert message(app) == "Redid Author"


async def test_undo_names_the_command_by_its_menus(app, tmp_path):
    path = tmp_path / "two.mth"
    path.write_text("a\nb\n")
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        await pilot.press(*str(path), "enter")
        assert entries(app) == ["a", "b"]
        await pilot.press("o", "n", "m", "enter")
        await pilot.press("ctrl+z")
        assert message(app) == "Undid Options Notation"
        await pilot.press("ctrl+z")
        assert message(app) == "Undid Transfer Load Derive"
        assert entries(app) == []


async def test_a_clear_confirmed_is_undone_and_one_refused_is_no_step(app):
    async with app.run_test() as pilot:
        await author(pilot, "x")
        await pilot.press("t", "c", "e", "n")
        await pilot.press("t", "c", "e", "y")
        assert entries(app) == []
        await pilot.press("ctrl+z")
        assert message(app) == "Undid Transfer Clear Expressions"
        assert entries(app) == ["x"]
        await pilot.press("ctrl+z")
        assert message(app) == "Undid Author"


async def test_with_nothing_to_undo_or_redo_the_keys_beep_and_say_so(app, beeps):
    async with app.run_test() as pilot:
        await pilot.press("ctrl+z")
        assert message(app) == NOTHING_TO_UNDO
        await pilot.press("ctrl+y")
        assert message(app) == NOTHING_TO_REDO
        assert len(beeps) == 2


async def test_on_a_line_the_keys_are_the_lines_and_undo_nothing(app):
    async with app.run_test() as pilot:
        await author(pilot, "x")
        await pilot.press("a", *"y+1", "ctrl+w")
        assert prompt(app) == ("AUTHOR expression:", "y+")
        # Ctrl-Y yanks the deleted word back.
        await pilot.press("ctrl+y")
        assert prompt(app) == ("AUTHOR expression:", "y+1")
        await pilot.press("ctrl+z")
        assert prompt(app)[0] == "AUTHOR expression:"
        assert entries(app) == ["x"]


async def test_in_a_submenu_or_a_dialog_the_keys_undo_nothing(app):
    async with app.run_test() as pilot:
        await author(pilot, "x")
        await pilot.press("t", "ctrl+z")
        assert entries(app) == ["x"]
        await pilot.press("escape", "o", "n", "ctrl+z")
        assert entries(app) == ["x"]


async def test_undo_takes_the_last_command_of_any_window_and_names_it(app):
    async with app.run_test(size=(80, 25)) as pilot:
        await author(pilot, "x")
        await pilot.press("w", "s", "h", "enter", "f1")
        await author(pilot, "y")
        await pilot.press("f1")
        await pilot.press("ctrl+z")
        assert message(app) == "Undid Author in window 2"
        assert app.windows.number == 1
        assert [entry.text for entry in app.windows.numbered(2).session.entries] == [
            "x"
        ]
        await pilot.press("ctrl+z")
        assert message(app) == "Undid Author"
        assert entries(app) == []
        await pilot.press("ctrl+y", "ctrl+y")
        assert message(app) == "Redid Author in window 2"
        assert app.windows.number == 1
        assert entries(app) == ["x"]


async def test_closing_a_window_leaves_the_other_windows_steps(app):
    async with app.run_test(size=(80, 25)) as pilot:
        await author(pilot, "x")
        await pilot.press("w", "s", "h", "enter", "f1")
        await author(pilot, "y")
        await pilot.press("w", "c", "enter", "y")
        assert len(app.windows.windows) == 1
        await pilot.press("ctrl+z")
        assert message(app) == "Undid Author"
        assert entries(app) == []
        await pilot.press("ctrl+z")
        assert message(app) == NOTHING_TO_UNDO


async def test_each_step_of_a_demonstration_is_undone_alone(app, tmp_path, monkeypatch):
    (tmp_path / "show.dmo").write_text("; adds\n2 + 3\n\n; and again\n4 + 5\n")
    monkeypatch.chdir(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press("t", "d", "o")
        await pilot.press(*"show.dmo", "enter")
        await pilot.press("space", "escape")
        assert entries(app) == ["2 + 3", "5", "4 + 5", "9"]
        await pilot.press("ctrl+z")
        assert message(app) == "Undid Demo step"
        assert entries(app) == ["2 + 3", "5"]
