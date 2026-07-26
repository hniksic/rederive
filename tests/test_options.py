"""The Options command, driven through the app the way a user drives it.

The expected band text is what the original puts on the same screen, minus the
leading blank the pane indents by.
"""

import pytest
from screen import band, entries, highlighted, message

from rederive.model.settings import COLORS
from rederive.ui import menu as menus
from rederive.ui.app import RederiveApp


@pytest.fixture
def app():
    return RederiveApp()


async def options(pilot, *keys):
    """Open the Options menu and press on."""
    await pilot.press("o", *keys)


# -- the menus ---------------------------------------------------------------


async def test_options_lists_the_commands_that_still_mean_something(app):
    async with app.run_test() as pilot:
        await options(pilot)
        assert band(app) == [
            " OPTIONS: Color Input Mute Notation Output Precision Radix"
        ]
        assert highlighted(app) == "Color"
        assert message(app) == "Enter option"


def test_every_menu_gives_each_word_its_own_letter():
    for menu in (menus.ALGEBRA, menus.OPTIONS, menus.COLOR, menus.COLORS):
        letters = [menus.mnemonic(word) for word in menu.words]
        assert len(set(letters)) == len(letters), menu.title


def test_the_color_words_spell_the_colors_they_stand_for():
    assert tuple(menus.COLOR_WORDS) == COLORS
    assert tuple(word.capitalize() for word in menus.COLOR_WORDS.values()) == COLORS


async def test_escape_climbs_back_one_level_at_a_time(app):
    async with app.run_test() as pilot:
        await options(pilot, "c")
        assert band(app) == [" OPTIONS COLOR: Menu Work"]
        await pilot.press("escape")
        assert highlighted(app) == "Color"
        await pilot.press("escape")
        assert highlighted(app) == "Author"
        assert message(app) == "Enter option"


# -- the dialogs -------------------------------------------------------------


async def test_a_dialog_parenthesizes_the_fields_that_are_not_active(app):
    async with app.run_test() as pilot:
        await options(pilot, "i")
        assert band(app) == [
            " OPTIONS INPUT: Mode: Character Word  Case:(Insensitive)Sensitive",
            "                Use:(LineEdit)Subexpression",
        ]
        assert highlighted(app) == "Character"
        assert message(app) == "Select input mode"
        await pilot.press("tab")
        assert band(app)[0].endswith("Mode:(Character)Word  Case: Insensitive Sensitive")
        assert highlighted(app) == "Insensitive"
        assert message(app) == "Select case mode"


async def test_a_letter_answers_the_field_and_moves_to_the_next(app):
    async with app.run_test() as pilot:
        await options(pilot, "p")
        assert message(app) == "Select arithmetic mode"
        await pilot.press("a")
        assert message(app) == "Enter significant digits"
        assert band(app)[0].startswith(
            " OPTIONS PRECISION: Mode:(Approximate)Exact Mixed"
        )


async def test_space_steps_the_value_under_the_highlight(app):
    async with app.run_test() as pilot:
        await options(pilot, "o")
        assert highlighted(app) == "Normal"
        await pilot.press("space")
        assert highlighted(app) == "Compressed"


async def test_a_lone_field_settles_the_dialog_on_one_keystroke(app):
    async with app.run_test() as pilot:
        await options(pilot, "m")
        assert band(app) == [" OPTIONS MUTE: Active: Yes No"]
        await pilot.press("y")
        assert app.settings["Mute"] == "Yes"
        assert highlighted(app) == "Author"


async def test_escape_leaves_the_settings_alone(app):
    async with app.run_test() as pilot:
        await options(pilot, "p", "a")
        await pilot.press("escape")
        assert app.settings["Precision"] == "Exact"
        assert entries(app) == []


# -- committing --------------------------------------------------------------


async def test_committing_records_one_expression_per_changed_field(app):
    async with app.run_test() as pilot:
        await options(pilot, "p", "a", "9", "enter")
        assert app.settings["Precision"] == "Approximate"
        assert app.settings["PrecisionDigits"] == 9
        assert entries(app) == ["Precision := Approximate", "PrecisionDigits := 9"]
        # And the last of them is what the highlight is left on.
        assert app.session.selected == 1


async def test_a_dialog_that_changed_nothing_records_nothing(app):
    async with app.run_test() as pilot:
        await options(pilot, "p", "enter")
        assert entries(app) == []
        assert highlighted(app) == "Author"


async def test_mute_and_color_are_remembered_but_not_recorded(app):
    async with app.run_test() as pilot:
        await options(pilot, "m", "y")
        await options(pilot, "c", "m", "space", "l", "enter")
        assert app.settings["Mute"] == "Yes"
        assert app.settings["FrameColor"] == "Lime"
        assert entries(app) == []


async def test_records_are_written_in_the_notation_they_select(app):
    async with app.run_test() as pilot:
        # Compressed format reaches the spacing of its own record.
        await options(pilot, "o", "c", "enter")
        assert entries(app) == ["DisplayFormat:=Compressed"]
        # And an octal output base reaches the digits of the next one.
        await options(pilot, "r", "tab", "o", "enter")
        await options(pilot, "n", "tab", "1", "2", "enter")
        assert entries(app)[1:] == ["OutputBase:=Octal", "NotationDigits:=14"]


async def test_radix_other_asks_for_a_base(app):
    async with app.run_test() as pilot:
        await options(pilot, "r", "space", "space")
        assert highlighted(app) == "Other"
        await pilot.press("enter")
        assert band(app) == [" OPTIONS RADIX: Input: 10"]
        assert message(app) == "Enter radix base"
        # The field overwrites as it is typed, so 10 becomes 70 - a base Derive
        # will not take - until the leftover digit is deleted.
        await pilot.press("7")
        assert band(app) == [" OPTIONS RADIX: Input: 70"]
        await pilot.press("enter")
        assert message(app) == "Enter radix base"
        await pilot.press("delete", "enter")
        assert app.settings["InputBase"] == 7
        assert entries(app) == ["InputBase := 7"]


# -- reacting ----------------------------------------------------------------


async def test_mute_silences_the_beep_a_stray_key_earns(app):
    async with app.run_test() as pilot:
        beeps = []
        app.bell = lambda: beeps.append("beep")
        await pilot.press("z")
        assert beeps == ["beep"]
        await options(pilot, "m", "y")
        await pilot.press("z")
        assert beeps == ["beep"]


async def test_a_color_reaches_the_screen_when_the_dialog_commits(app):
    async with app.run_test() as pilot:
        before = app.palette.styles["option"]
        await options(pilot, "c", "m")
        assert band(app) == [
            " OPTIONS COLOR MENU: Frame: Red  Option: Yellow  Prompt: Red  Status: Green",
            "                     Background: Black  Border: Black",
        ]
        assert message(app) == "Select frame color (press Space for the list)"
        # The list is a submenu, opening on the color the slot already has.
        await pilot.press("space")
        assert band(app) == [
            " COLOR: Black blUe Green Cyan Red Magenta brOwn grAy",
            "        Darkgray Skyblue Lime aQua Pink Violet Yellow White",
        ]
        assert highlighted(app) == "Red"
        # Picking one returns to the dialog, on the next slot along.
        await pilot.press("s")
        assert app.palette.styles["option"] == before
        assert message(app) == "Select option color (press Space for the list)"
        await pilot.press("p", "enter")
        assert app.settings["FrameColor"] == "Skyblue"
        assert app.palette.styles["option"] != before
        assert app.palette.css_variables()["rd-frame"] == "#5555ff"
        assert app.query_one("#menu").styles.color.hex.lower() == "#ff5555"
