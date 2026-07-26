"""The Manage commands, driven the way a user drives them.

Four of them set something and are dialogs; Renumber, Annotate and Ordering act
on the session instead, and ask for what they need on a line of their own.

The expected band text and message lines are what the original puts on the same
screens, minus the leading blank the pane indents by. Every version shows the
same words on every one of them.

Only the keys peculiar to these screens are exercised here; the rest of what a
dialog does belongs to `test_options` and `test_settings`, since it is the same
machinery underneath.
"""

import pytest
from screen import annotation, band, entries, highlighted, message, prompt

from rederive.engine.context import Angle, Branch, Direction, TrigPower
from rederive.model.session import Session
from rederive.ui import menu as menus
from rederive.ui.app import RederiveApp


@pytest.fixture
def app():
    return RederiveApp()


@pytest.fixture
def session():
    return Session()


async def manage(pilot, *keys):
    """Open the Manage menu and press on."""
    await pilot.press("m", *keys)


def authored(session, *texts):
    for text in texts:
        session.author(text)
    return session


def labels(session):
    """The history as the work area shows it: each entry's label and its text."""
    return [f"#{entry.number}: {entry.text}" for entry in session.entries]


def annotations(session):
    return [entry.annotation for entry in session.entries]


# -- the menu ----------------------------------------------------------------


async def test_manage_lists_eight_commands_over_two_lines(app):
    async with app.run_test() as pilot:
        await manage(pilot)
        assert band(app) == [
            " MANAGE: Annotate Branch Exponential Logarithm Ordering Renumber Substitute",
            "         Trigonometry",
        ]
        assert highlighted(app) == "Annotate"
        assert message(app) == "Enter option"


def test_every_manage_word_gives_its_own_letter():
    letters = [menus.mnemonic(word) for word in menus.MANAGE.words]
    assert letters == list("abelorst")


async def test_substitute_is_the_one_command_still_inert(app):
    async with app.run_test() as pilot:
        await manage(pilot, "s")
        assert message(app) == "Substitute: not implemented yet"
        # The menu is still up, with the word that did nothing on it.
        assert band(app)[0].startswith(" MANAGE: ")


# -- the screens -------------------------------------------------------------


async def test_branch_is_a_selection_field_with_no_caption(app):
    async with app.run_test() as pilot:
        await manage(pilot, "b")
        assert band(app) == [" MANAGE BRANCH: Principal Real Any"]
        assert highlighted(app) == "Principal"
        assert message(app) == "Select preferred branch for roots"


async def test_exponential_and_logarithm_ask_for_a_direction(app):
    async with app.run_test() as pilot:
        await manage(pilot, "e")
        assert band(app) == [" MANAGE EXPONENTIAL: Direction: Auto Collect Expand"]
        assert message(app) == "Exponential transformations"
        await pilot.press("escape")
        await pilot.press("l")
        assert band(app) == [" MANAGE LOGARITHM: Direction: Auto Collect Expand"]
        assert message(app) == "Logarithm transformations"


async def test_the_exponential_field_opens_on_auto(app):
    """Section 6.1 of the manual says Collect, which was the 1.x default and
    stale by the later releases: the original's own DERIVE.INI says
    `*EXP-EXPD* |Auto|`."""
    async with app.run_test() as pilot:
        await manage(pilot, "e")
        assert highlighted(app) == "Auto"
        assert app.settings["Exponential"] == "Auto"


async def test_trigonometry_asks_three_questions_over_two_lines(app):
    async with app.run_test() as pilot:
        await manage(pilot, "t")
        assert band(app) == [
            " MANAGE TRIGONOMETRY: Direction: Auto Collect Expand"
            "  Toward:(Auto)Sines Cosines",
            "                      Angle: Degree(Radian)",
        ]
        assert message(app) == "Angle sums & multiple angles"
        await pilot.press("tab")
        assert message(app) == "Trig power transformations"
        assert highlighted(app) == "Auto"
        await pilot.press("tab")
        assert message(app) == "Select angle mode"
        assert highlighted(app) == "Radian"
        # And round to the first field again, committing nothing on the way.
        await pilot.press("tab")
        assert message(app) == "Angle sums & multiple angles"
        assert entries(app) == []


# -- answering ---------------------------------------------------------------


async def test_a_letter_settles_a_one_field_screen_outright(app):
    async with app.run_test() as pilot:
        await manage(pilot, "b", "r")
        assert app.settings["Branch"] == "Real"
        assert entries(app) == ["Branch := Real"]
        assert highlighted(app) == "Author"


async def test_a_letter_on_a_wider_screen_moves_to_the_next_field(app):
    async with app.run_test() as pilot:
        await manage(pilot, "t", "e")
        assert message(app) == "Trig power transformations"
        assert entries(app) == []
        assert band(app)[0].startswith(
            " MANAGE TRIGONOMETRY: Direction: Auto Collect(Expand)"
        )


async def test_space_and_backspace_step_the_active_field(app):
    async with app.run_test() as pilot:
        await manage(pilot, "b")
        await pilot.press("space")
        assert highlighted(app) == "Real"
        await pilot.press("backspace")
        assert highlighted(app) == "Principal"
        # Backspace wraps around the near end rather than leaving the field.
        await pilot.press("backspace")
        assert highlighted(app) == "Any"


async def test_committing_records_one_expression_per_changed_field(app):
    async with app.run_test() as pilot:
        # Direction to Expand, past Toward untouched, Angle to Degree.
        await manage(pilot, "t", "e", "tab", "d", "enter")
        assert app.settings["Trigonometry"] == "Expand"
        assert app.settings["Trigpower"] == "Auto"
        assert app.settings["Angle"] == "Degree"
        assert entries(app) == ["Trigonometry := Expand", "Angle := Degree"]
        assert highlighted(app) == "Author"


async def test_a_screen_that_changed_nothing_records_nothing(app):
    async with app.run_test() as pilot:
        await manage(pilot, "t", "enter")
        assert entries(app) == []
        assert highlighted(app) == "Author"


async def test_escape_abandons_every_pending_change(app):
    async with app.run_test() as pilot:
        await manage(pilot, "t", "c", "s", "d")
        await pilot.press("escape")
        assert app.settings["Trigonometry"] == "Auto"
        assert app.settings["Trigpower"] == "Auto"
        assert app.settings["Angle"] == "Radian"
        assert entries(app) == []
        # And one Esc lands on the Manage menu, not past it.
        assert band(app)[0].startswith(" MANAGE: ")


# -- the same settings from the author line ----------------------------------


async def test_an_authored_assignment_changes_the_setting_it_names(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"Branch := Real", "enter")
        assert app.settings["Branch"] == "Real"
        assert app.session.context.branch is Branch.REAL
        # And the screen that owns it opens on what was authored.
        await manage(pilot, "b")
        assert highlighted(app) == "Real"


def test_every_manage_setting_reaches_the_context(session):
    session.settings.apply(
        {
            "Branch": "Any",
            "Exponential": "Collect",
            "Logarithm": "Expand",
            "Trigonometry": "Collect",
            "Trigpower": "Cosines",
            "Angle": "Degree",
        }
    )
    context = session.context
    assert context.branch is Branch.ANY
    assert context.exponential is Direction.COLLECT
    assert context.logarithm is Direction.EXPAND
    assert context.trigonometry is Direction.COLLECT
    assert context.trigpower is TrigPower.COSINES
    assert context.angle is Angle.DEGREE


async def test_the_branch_setting_changes_the_answer(app):
    """The point of the step: the principal cube root of -8 is complex, and the
    real branch of it is -2."""
    async with app.run_test() as pilot:
        await pilot.press("a", *"(-8)^(1/3)", "enter")
        await pilot.press("s", "enter")
        assert entries(app)[-1] == "2*(-1)^(1/3)"
        await manage(pilot, "b", "r")
        await pilot.press("j", *"1", "enter")
        await pilot.press("s", "enter")
        assert entries(app)[-1] == "-2"


# -- Manage Renumber ---------------------------------------------------------


def test_renumber_puts_the_labels_back_in_sequence(session):
    """The manual's own case: a remove leaves 1, 3, 4, and renumbering makes it
    1, 2, 3 - the annotation of the last entry moving with the label it names."""
    authored(session, "x", "y", "z")
    session.remove(2, 2)
    session.simplify("#3")
    assert labels(session) == ["#1: x", "#3: z", "#4: z"]
    assert annotations(session) == ["User", "User", "Simp(#3)"]
    session.renumber()
    assert labels(session) == ["#1: x", "#2: z", "#3: z"]
    assert annotations(session) == ["User", "User", "Simp(#2)"]


def test_renumber_follows_the_order_the_entries_sit_in(session):
    """What a move, or a remove and an unremove, leaves: labels out of sequence
    because the entries are no longer stacked in the order they were numbered."""
    authored(session, "a", "b", "c", "d", "e")
    session.remove(2, 3)
    session.unremove()
    assert labels(session) == ["#1: a", "#4: d", "#5: e", "#2: b", "#3: c"]
    session.renumber()
    assert labels(session) == ["#1: a", "#2: d", "#3: e", "#4: b", "#5: c"]


def test_renumber_moves_a_reference_inside_an_expression(session):
    """Rederive keeps `#3` as a reference where the original resolved it as it
    read the line, so renumbering has to move it or it points somewhere else."""
    authored(session, "x", "y", "z")
    session.remove(2, 2)
    session.author("#3 + 1")
    session.renumber()
    assert labels(session) == ["#1: x", "#2: z", "#3: #2 + 1"]
    # And it still names the same expression it named before.
    assert session.simplify("#3").text == "z + 1"


def test_a_reference_in_a_renumbered_line_is_drawn_as_it_now_reads(session):
    authored(session, "x", "y", "z")
    session.remove(2, 2)
    session.author("#3 + 1")
    session.renumber()
    assert session.entries[-1].layout.lines == ("#2 + 1",)


def test_the_next_label_follows_the_renumbered_history(session):
    authored(session, "x", "y", "z")
    session.remove(2, 2)
    session.renumber()
    session.author("q")
    assert labels(session) == ["#1: x", "#2: z", "#3: q"]


def test_renumbering_a_history_already_in_sequence_changes_nothing(session):
    authored(session, "x", "y")
    before = list(session.entries)
    session.renumber()
    assert session.entries == before


def test_renumbering_an_empty_history_changes_nothing(session):
    session.renumber()
    session.author("x")
    assert labels(session) == ["#1: x"]


def test_renumber_leaves_the_highlight_where_it_was(session):
    authored(session, "x + 1", "y", "z")
    session.remove(2, 2)
    session.select_entry(0)
    session.move_right()
    session.renumber()
    assert session.selected == 0 and session.route == (0,)
    assert session.selected_node.value == "x"


async def test_renumber_runs_on_the_keystroke(app):
    """No prompt, no dialog, no message: the command menu comes straight back."""
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter", "a", *"y", "enter")
        await pilot.press("r", *"2", "tab", *"2", "enter")
        await pilot.press("a", *"z", "enter")
        assert [entry.number for entry in app.session.entries] == [1, 3]
        await manage(pilot, "r")
        assert [entry.number for entry in app.session.entries] == [1, 2]
        assert band(app)[0].startswith(" COMMAND: ")
        assert message(app) == "Enter option"


# -- Manage Annotate ---------------------------------------------------------


async def test_annotate_asks_for_a_label_and_then_for_the_text(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter", "a", *"y", "enter")
        await manage(pilot, "a")
        assert band(app) == [" MANAGE ANNOTATE: Expression: 2"]
        assert message(app) == "Enter label number"
        await pilot.press("enter")
        assert prompt(app) == ("ANNOTATION:", "User")
        assert message(app) == "Enter annotation"


async def test_the_label_field_follows_the_highlight(app):
    """As `Remove` and `Unremove` do: the manual would rather you picked the
    expression than typed its number."""
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter", "a", *"y", "enter")
        await manage(pilot, "a")
        await pilot.press("up")
        assert band(app) == [" MANAGE ANNOTATE: Expression: 1"]


async def test_an_annotation_reaches_the_status_line(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x + 1", "enter")
        await manage(pilot, "a", "enter", *"Kirchhoff", "enter")
        assert annotation(app) == "Kirchhoff"
        assert app.session.entries[0].annotation == "Kirchhoff"


async def test_annotating_appends_nothing_and_moves_no_highlight(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter", "a", *"y", "enter")
        await manage(pilot, "a", "enter", *"Mine", "enter")
        assert entries(app) == ["x", "y"]
        assert app.session.selected == 1
        assert band(app)[0].startswith(" COMMAND: ")


async def test_the_annotation_offered_is_the_one_the_entry_carries(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x + 1", "enter")
        await pilot.press("s", "enter")
        await manage(pilot, "a", "enter")
        assert prompt(app) == ("ANNOTATION:", "Simp(#1)")


async def test_escape_from_the_annotation_line_lands_on_the_manage_menu(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter")
        await manage(pilot, "a", "enter", *"Mine")
        await pilot.press("escape")
        assert app.session.entries[0].annotation == "User"
        assert band(app)[0].startswith(" MANAGE: ")


async def test_a_label_that_names_no_entry_is_refused(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter")
        await manage(pilot, "a", *"7", "enter")
        # The question stays up with what was typed still on it to correct.
        assert band(app) == [" MANAGE ANNOTATE: Expression: 7"]


async def test_annotate_asks_nothing_of_an_empty_history(app):
    async with app.run_test() as pilot:
        await manage(pilot, "a")
        assert band(app)[0].startswith(" COMMAND: ")
        assert entries(app) == []


def test_an_annotation_is_written_above_the_expression_it_belongs_to(
    session, tmp_path
):
    authored(session, "x + 1")
    session.annotate(1, "Kirchhoff")
    file = tmp_path / "sheet.mth"
    session.save(file)
    assert file.read_text().splitlines()[:2] == [";Kirchhoff", "x+1"]


def test_an_annotation_read_back_is_the_one_that_was_written(session, tmp_path):
    authored(session, "x + 1")
    session.annotate(1, "Kirchhoff")
    file = tmp_path / "sheet.mth"
    session.save(file)
    other = Session()
    other.load(file)
    assert annotations(other) == ["Kirchhoff"]


def test_an_annotation_moves_with_its_label_when_the_history_is_renumbered(session):
    authored(session, "x", "y", "z")
    session.remove(2, 2)
    session.simplify("#3")
    session.annotate(4, "from #3, twice over")
    session.renumber()
    assert annotations(session)[-1] == "from #2, twice over"


# -- Manage Ordering ---------------------------------------------------------


async def test_ordering_offers_the_list_in_force(app):
    async with app.run_test() as pilot:
        await manage(pilot, "o")
        assert prompt(app) == ("MANAGE ORDER variables:", "x y z")
        assert message(app) == "Enter variables in desired order"


async def test_ordering_takes_a_new_list_and_records_nothing(app):
    async with app.run_test() as pilot:
        await manage(pilot, "o", *"y x z", "enter")
        assert app.session.order == ("y", "x", "z")
        # Unlike the four settings screens, this one appends no expression.
        assert entries(app) == []
        assert band(app)[0].startswith(" COMMAND: ")
        # And the next Manage Ordering opens on what was entered.
        await manage(pilot, "o")
        assert prompt(app) == ("MANAGE ORDER variables:", "y x z")


async def test_a_line_that_is_not_variables_is_refused(app):
    """Derive stores `q 2 + w` verbatim; reproducing that would be a bug."""
    async with app.run_test() as pilot:
        await manage(pilot, "o", *"q 2 + w", "enter")
        assert app.session.order == ("x", "y", "z")
        assert prompt(app) == ("MANAGE ORDER variables:", "q 2 + w")


def test_a_variable_may_not_be_named_twice(session):
    assert session.order_list("x y x") is None


def test_a_predefined_name_is_not_a_variable(session):
    assert session.order_list("sin") is None
    assert session.order_list("pi") is None


def test_a_variable_goes_on_the_list_the_way_an_expression_records_it(session):
    assert session.order_list("Y, X") == ("y", "x")


def test_an_empty_line_is_an_empty_list(session):
    assert session.order_list("") == ()


def test_the_order_list_changes_which_variable_factor_offers_first(session):
    authored(session, "x^2 - y^2")
    assert session.variables("#1") == ("x", "y")
    session.order = ("y", "x", "z")
    assert session.variables("#1") == ("y", "x")


def test_a_variable_off_the_list_is_less_main_than_one_on_it(session):
    authored(session, "x^2 - a^2")
    assert session.variables("#1") == ("x", "a")
    session.order = ("a",)
    assert session.variables("#1") == ("a", "x")


async def test_the_order_list_reaches_the_variables_factor_offers(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x^2 - y^2", "enter")
        await manage(pilot, "o", *"y x z", "enter")
        await pilot.press("f", "enter")
        assert message(app) == "Return for all or select 1: y,x"


def test_the_order_list_decides_a_declared_functions_parameters(session):
    session.order = ("y", "x", "z")
    entry = session.declare_function("F", "x^2 + y")
    assert entry.text == "F(y, x) := x^2 + y"


# -- the state file ----------------------------------------------------------


def test_a_state_file_carries_the_manage_settings(session, tmp_path):
    file = tmp_path / "state.ini"
    settings = {
        "Branch": "Real",
        "Exponential": "Collect",
        "Logarithm": "Expand",
        "Trigonometry": "Expand",
        "Trigpower": "Sines",
        "Angle": "Degree",
    }
    session.settings.apply(settings)
    session.save_state(file)
    lines = file.read_text().splitlines()
    for name, value in settings.items():
        assert f"{name} := {value}" in lines
    other = Session()
    assert other.load_state(file) == 0
    for name, value in settings.items():
        assert other.settings[name] == value


def test_a_state_file_carries_the_variable_order_list(session, tmp_path):
    """Not a setting - no field owns it and nothing records it as an assignment
    - but Derive kept it in DERIVE.INI, so a session opens the way you left it."""
    file = tmp_path / "state.ini"
    session.order = ("y", "x", "z")
    session.save_state(file)
    assert "VariableOrder := y x z" in file.read_text().splitlines()
    other = Session()
    assert other.load_state(file) == 0
    assert other.order == ("y", "x", "z")


def test_an_order_line_that_names_no_variables_is_counted(session, tmp_path):
    file = tmp_path / "state.ini"
    file.write_text("VariableOrder := q 2 + w\n")
    assert session.load_state(file) == 1
    assert session.order == ("x", "y", "z")
