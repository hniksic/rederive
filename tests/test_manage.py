"""The four Manage commands that set something, driven the way a user drives them.

The expected band text and message lines are what the original puts on the same
screens, minus the leading blank the pane indents by. Every version shows the
same words on all four.

Only the keys peculiar to these screens are exercised here; the rest of what a
dialog does belongs to `test_options` and `test_settings`, since it is the same
machinery underneath.
"""

import pytest
from screen import band, entries, highlighted, message

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


async def test_the_four_commands_that_are_not_settings_are_inert(app):
    async with app.run_test() as pilot:
        for word in ("Annotate", "Ordering", "Renumber", "Substitute"):
            await manage(pilot, menus.mnemonic(word))
            assert message(app) == f"{word}: not implemented yet"
            # The menu is still up, with the word that did nothing on it.
            assert band(app)[0].startswith(" MANAGE: ")
            await pilot.press("escape")


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
