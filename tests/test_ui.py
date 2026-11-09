"""Smoke tests driving the real app through Textual's pilot."""

import pytest
from textual import events
from screen import (
    annotation,
    band,
    entries,
    flags,
    highlighted,
    highlighted_expression,
    highlighted_rows,
    laid_out,
    message,
    prompt,
    resized,
    text_of,
    work_area,
)

from rederive import __version__, platform
from rederive.ui import menu as menus
from rederive.ui.app import COPIED, COPIED_TEXT, CUT_TEXT, NOTHING_TO_CUT, RederiveApp
from rederive.ui.widgets import WorkArea


def highlighted_menu_option(app):
    return highlighted(app)


@pytest.fixture
def app():
    return RederiveApp()


async def author(pilot, text):
    await pilot.press("a")
    await pilot.press(*text)
    await pilot.press("enter")


def authored(app, *texts):
    """Add entries through the session instead of typing them.

    The Author line just hands its text to `Session.author`, so this leaves the same
    worksheet, with the last entry selected and the panes painted. Use it for setup
    when the keys under test come afterwards: each key pressed through the pilot waits
    for the app to go idle, so typing a six-key line costs ~70ms versus ~1ms here.
    Type the line for real when it ends in `=` (the Author line does more with it), or
    when the Author line itself or the layout it leaves is what's being tested.
    """
    for text in texts:
        app.session.author(text)
    app.place_windows()


# -- the opening notice --------------------------------------------------------

# As in the original, the notice is just what was painted, not a state: it survives
# menus, dialogs and abandoned lines, and is gone for good once anything else draws the
# work area (an expression, a help page, a window command, a Clear).


async def test_the_opening_notice_stands_across_the_empty_pane(app):
    async with app.run_test():
        shown = work_area(app)
        said = [line.strip() for line in shown if line]
        assert said == [
            "R E D E R I V E",
            "A Mathematical Assistant",
            f"Version {__version__}",
            "Press H for help, T D for demos",
        ]
        # Centered, with blank rows at the top and bottom.
        width = app.query_one("#panes").size.width
        assert all(line == line.strip().center(width).rstrip() for line in shown if line)
        assert shown[0] == "" and shown[-1] == ""
        # The name block sits high and the help line low, with more than one blank
        # row between them.
        rows = [at for at, line in enumerate(shown) if line]
        assert rows[0] < len(shown) // 3 < rows[-1]
        assert rows[-1] - rows[-2] > 1


async def test_the_opening_notice_gives_the_pane_up_for_the_first_expression(app):
    async with app.run_test() as pilot:
        authored(app, "x")
        assert work_area(app) == ["#1:  x"]
        # The notice doesn't come back when the worksheet is emptied later.
        await pilot.press("t", "c", "e", "y")
        assert work_area(app) == []


async def test_a_split_takes_the_opening_notice_off_both_panes(app):
    async with app.run_test(size=(80, 25)) as pilot:
        await pilot.press("w", "s", "h", "enter")
        assert work_area(app, 1) == [] and work_area(app, 2) == []


async def test_a_clear_with_nothing_to_clear_takes_the_opening_notice_away(app):
    async with app.run_test() as pilot:
        await pilot.press("t", "c", "e")
        assert work_area(app) == []


async def test_a_menu_and_a_dialog_leave_the_opening_notice_standing(app):
    async with app.run_test() as pilot:
        # A menu opened and escaped, a dialog committed, and a line abandoned.
        await pilot.press("w", "escape")
        await pilot.press("o", "n", "enter")
        await pilot.press("a", "escape")
        assert "Press H for help" in "\n".join(work_area(app))


async def test_escape_on_the_command_menu_returns_the_highlight_to_author(app):
    async with app.run_test() as pilot:
        await pilot.press("tab", "tab", "tab")
        assert highlighted_menu_option(app) == "Declare"
        await pilot.press("escape")
        assert highlighted_menu_option(app) == "Author"
        assert message(app) == "Enter option"


async def test_menu_highlight_cycles_and_wraps(app):
    async with app.run_test() as pilot:
        assert highlighted_menu_option(app) == "Author"
        assert message(app) == "Enter option"
        await pilot.press("tab", "tab")
        assert highlighted_menu_option(app) == "Calculus"
        await pilot.press("shift+tab")
        assert highlighted_menu_option(app) == "Build"
        await pilot.press("shift+tab", "shift+tab")
        assert highlighted_menu_option(app) == "approX"


async def test_mnemonic_invokes_without_moving_the_highlight(app):
    async with app.run_test() as pilot:
        await pilot.press("u")
        assert message(app) == "Unremove buffer empty"
        assert highlighted_menu_option(app) == "Author"


async def test_a_menu_comes_back_up_on_its_first_word(app):
    async with app.run_test() as pilot:
        await pilot.press(*["tab"] * 10)
        assert highlighted_menu_option(app) == "Options"
        await pilot.press("enter")
        assert highlighted_menu_option(app) == "Color"
        # Leaving the submenu forgets where Tab left the highlight below it.
        await pilot.press("escape")
        assert highlighted_menu_option(app) == "Author"


async def test_a_command_that_ran_leaves_the_highlight_where_it_starts(app):
    async with app.run_test() as pilot:
        authored(app, "1+1")
        await pilot.press(*["shift+tab"] * 6)
        assert highlighted_menu_option(app) == "Simplify"
        await pilot.press("enter", "enter")
        assert entries(app) == ["1+1", "2"]
        assert highlighted_menu_option(app) == "Author"


async def test_an_abandoned_command_leaves_the_highlight_alone(app):
    async with app.run_test() as pilot:
        await pilot.press(*["shift+tab"] * 6)
        await pilot.press("enter", "escape")
        assert highlighted_menu_option(app) == "Simplify"


# -- the band across a terminal that is not eighty columns --------------------


async def test_eighty_columns_break_the_menu_where_the_menu_says(app):
    async with app.run_test(size=(80, 24)) as pilot:
        assert band(app) == [
            " COMMAND: Author Build Calculus Declare Expand Factor Help Jump soLve Manage",
            "          Options Plot Quit Remove Simplify Transfer Unremove moVe Window"
            " approX",
        ]


@pytest.mark.parametrize("width", [70, 60, 45, 30], ids=str)
async def test_a_narrow_terminal_keeps_every_word_of_the_menu(app, width):
    async with app.run_test(size=(width, 24)) as pilot:
        lines = band(app)
        assert " ".join(line.strip() for line in lines).split() == [
            "COMMAND:",
            *menus.ALGEBRA.words,
        ]
        # The band is as tall as its lines, so none wraps or is cut off.
        assert app.query_one("#menu").size.height == len(lines)
        assert max(len(line) for line in lines) <= width


async def test_a_narrow_terminal_keeps_both_lines_of_a_dialog(app):
    async with app.run_test(size=(60, 24)) as pilot:
        await pilot.press("o", "r")
        # The band reaches the height the fields need only after the layout catches up.
        await laid_out(pilot, app, "#fields")
        lines = band(app)
        assert lines[0].startswith(" OPTIONS RADIX: Input:")
        assert lines[-1].strip().startswith("Output:")
        assert app.query_one("#fields").size.height == len(lines)


async def test_a_resized_terminal_lays_the_menu_out_again(app):
    async with app.run_test(size=(80, 24)) as pilot:
        assert len(band(app)) == 2
        await resized(pilot, app, 60, 24)
        assert len(band(app)) == 3
        await resized(pilot, app, 80, 24)
        assert len(band(app)) == 2


async def test_a_terminal_too_short_for_the_menu_keeps_its_other_lines(app):
    async with app.run_test(size=(30, 10)) as pilot:
        # Five rows go to the rule, message and status lines and two rows of
        # expressions; the menu gets what's left and is cut off.
        assert len(band(app)) == 5
        assert message(app) == "Enter option"
        await pilot.pause()
        assert app.work_area.size.height == 2


async def test_author_appends_and_selects_the_new_entry(app):
    async with app.run_test() as pilot:
        await author(pilot, "x (x + 1)")
        assert [entry.text for entry in app.session.entries] == ["x*(x+1)"]
        # Typed as juxtaposition, drawn with the times operator.
        assert work_area(app) == ["#1:  x·(x + 1)"]
        assert highlighted_expression(app) == "x·(x + 1)"
        assert message(app) == "Enter option"
        assert text_of(app.query_one("#status")).plain.strip().startswith("User")


@pytest.mark.parametrize("key", ["enter", "ctrl+j"], ids=str)
async def test_a_trailing_equals_enters_the_value_beside_what_was_typed(app, key):
    async with app.run_test() as pilot:
        await pilot.press("a", *"2^3=", key)
        # One entry, annotated as the user's: the `=` was typed, not a command. The
        # value is already there, so Ctrl-Enter has nothing left to simplify.
        assert entries(app) == ["2^3 = 8"]
        assert annotation(app) == "User"
        # It was computed, so it is timed.
        assert message(app).startswith("Compute time:")


@pytest.mark.parametrize("key", ["ctrl+j", "ctrl+enter"], ids=str)
async def test_ctrl_enter_authors_and_simplifies_in_one(app, key):
    async with app.run_test() as pilot:
        await pilot.press("a")
        await pilot.press(*"2+3")
        await pilot.press(key)
        assert entries(app) == ["2+3", "5"]
        assert annotation(app) == "Simp(#1)"
        assert message(app).startswith("Compute time:")


async def test_ctrl_enter_is_enter_on_a_line_that_simplifies_already(app):
    async with app.run_test() as pilot:
        authored(app, "2+3")
        await pilot.press("s")
        await pilot.press("ctrl+j")
        assert entries(app) == ["2+3", "5"]


async def test_ctrl_enter_enters_nothing_of_its_own_to_simplify(app):
    async with app.run_test() as pilot:
        authored(app, "2+3")
        await pilot.press("j")
        await pilot.press("1", "ctrl+j")
        assert entries(app) == ["2+3"]


async def test_a_vector_is_entered_and_simplified_at_once(app):
    async with app.run_test() as pilot:
        await pilot.press("d", "r", "2", "enter")
        await pilot.press("1", "enter")
        await pilot.press(*"2+3")
        await pilot.press("ctrl+j")
        assert entries(app) == ["[1,2+3]", "[1, 5]"]


async def test_a_syntax_error_leaves_the_author_line_up(app):
    async with app.run_test() as pilot:
        await pilot.press("a")
        await pilot.press(*"12.34.5")
        await pilot.press("enter")
        assert app.session.entries == []
        assert message(app) == "Syntax error detected at cursor"
        author_input = app.query_one("#prompt-input")
        assert author_input.value == "12.34.5"
        # Positioned where Derive stops reading: the second decimal point.
        assert author_input.cursor_position == 5
        # The line stays up to be corrected.
        await pilot.press("delete")
        await pilot.press("enter")
        assert [entry.text for entry in app.session.entries] == ["12.345"]
        assert message(app) == "Enter option"


async def test_a_built_up_render_is_painted_and_labelled(app):
    async with app.run_test() as pilot:
        authored(app, "(x+1)/(x^2+2x+3)")
        # The label sits on the middle row (rounding down the screen), and the
        # render's rows share one indent.
        assert work_area(app) == [
            "          x + 1",
            "     ──────────────",
            "#1:    2",
            "      x  + 2·x + 3",
        ]


async def test_the_selection_is_a_rectangle_over_the_render(app):
    async with app.run_test() as pilot:
        authored(app, "(x+1)/(x^2+2x+3)")
        assert highlighted_rows(app) == [
            "     x + 1",
            "──────────────",
            "  2",
            " x  + 2·x + 3",
        ]
        await pilot.press("right")
        assert highlighted_rows(app) == ["x + 1"]
        await pilot.press("right")
        assert highlighted_rows(app) == [" 2", "x  + 2·x + 3"]


async def test_escape_abandons_the_author_line(app):
    async with app.run_test() as pilot:
        await pilot.press("a")
        await pilot.press(*"abc")
        await pilot.press("escape")
        assert app.session.entries == []
        assert message(app) == "Enter option"


async def test_arrow_keys_walk_the_expression_structure(app):
    async with app.run_test() as pilot:
        authored(app, "x (x + 1)")
        authored(app, "x y")
        authored(app, "x x x")
        assert highlighted_expression(app) == "x·x·x"
        await pilot.press("up", "up")
        assert highlighted_expression(app) == "x·(x + 1)"
        await pilot.press("right")
        assert highlighted_expression(app) == "x"
        await pilot.press("right")
        assert highlighted_expression(app) == "x + 1"
        await pilot.press("down")
        assert highlighted_expression(app) == "x"
        await pilot.press("up", "up")
        assert highlighted_expression(app) == "x·(x + 1)"
        # The menu highlight never moved while the arrows walked the history.
        assert highlighted_menu_option(app) == "Author"


async def test_simplify_offers_the_highlighted_expression(app):
    async with app.run_test() as pilot:
        authored(app, "2 (8 + 7) / 3^2")
        await pilot.press("s")
        assert prompt(app) == ("SIMPLIFY expression:", "#1")
        assert message(app) == "Enter expression"
        await pilot.press("enter")
        assert work_area(app) == [
            "      2·(8 + 7)",
            "     ───────────",
            "#1:        2",
            "          3",
            "",
            "      10",
            "#2:  ────",
            "       3",
        ]
        assert highlighted_expression(app) == " 10\n────\n  3"
        assert message(app).startswith("Compute time:")
        assert annotation(app) == "Simp(#1)"


async def test_simplify_of_a_part_copies_the_rest_of_the_expression(app):
    async with app.run_test() as pilot:
        authored(app, "2 (8 + 7) / 3^2")
        await pilot.press("right", "right")
        assert highlighted_rows(app) == [" 2", "3"]
        await pilot.press("s", "enter")
        assert [entry.text for entry in app.session.entries][-1] == "2*(8+7)/9"
        assert highlighted_expression(app) == " 2·(8 + 7)\n───────────\n     9"
        # The quote says that only part of the expression was simplified.
        assert annotation(app) == "Simp(#1')"


async def test_a_typed_expression_is_simplified_as_the_users_own(app):
    async with app.run_test() as pilot:
        await pilot.press("s")
        # Nothing is highlighted, so nothing is offered.
        assert prompt(app) == ("SIMPLIFY expression:", "")
        await pilot.press(*"2+3")
        await pilot.press("enter")
        assert work_area(app) == ["#1:  5"]
        assert annotation(app) == "Simp(User)"


async def test_a_typed_label_replaces_the_one_offered(app):
    async with app.run_test() as pilot:
        authored(app, "3 + 4")
        authored(app, "5 + 6")
        # The number comes up selected, so typing one replaces it.
        await pilot.press("s")
        await pilot.press("1")
        assert prompt(app) == ("SIMPLIFY expression:", "#1")
        await pilot.press("enter")
        assert work_area(app)[-1] == "#3:  7"
        assert annotation(app) == "Simp(#1)"


async def test_the_annotation_follows_the_selection(app):
    async with app.run_test() as pilot:
        authored(app, "2^3")
        await pilot.press("s", "enter")
        assert annotation(app) == "Simp(#1)"
        await pilot.press("up")
        assert highlighted_expression(app) == " 3\n2"
        assert annotation(app) == "User"


@pytest.mark.parametrize("key", ["s", "f", "l"], ids=str)
async def test_an_expression_line_that_does_not_read_is_left_up(app, key):
    """Covers the three line readers: Simplify/approX, Factor/Expand, and soLve."""
    async with app.run_test() as pilot:
        authored(app, "x")
        await pilot.press(key)
        await pilot.press(*"+")
        await pilot.press("enter")
        assert message(app) == "Syntax error detected at cursor"
        assert [entry.text for entry in app.session.entries] == ["x"]
        await pilot.press("escape")
        assert message(app) == "Enter option"
        assert highlighted_menu_option(app) == "Author"


async def test_quit_asks_before_abandoning_expressions(app):
    async with app.run_test() as pilot:
        authored(app, "x")
        await pilot.press("q")
        assert message(app) == "Abandon expressions (Y/N)?"
        # The original takes the highlight off the menu while it asks.
        assert highlighted_menu_option(app) is None
        await pilot.press("n")
        assert message(app) == "Enter option"
        assert highlighted_menu_option(app) == "Author"
        assert app.is_running
        await pilot.press("q", "y")
        await pilot.pause()
        assert not app.is_running


# -- Factor, which asks three questions before it answers ---------------------
#
# Every screen asserted here was checked against the original. The question order
# follows the original, not the manual, which has the amount asked before the variables.


async def test_factor_asks_for_expression_then_variables_then_amount(app):
    async with app.run_test() as pilot:
        authored(app, "6x^2 + 10x^3/y")
        await pilot.press("f")
        assert prompt(app) == ("FACTOR expression:", "#1")
        assert message(app) == "Enter expression"
        await pilot.press("enter")
        assert prompt(app) == ("FACTOR variable 1:", "")
        assert message(app) == "Return for all or select 1: x,y"
        await pilot.press("enter")
        assert band(app) == [
            " FACTOR: Amount: Trivial Squarefree Rational raDical Complex"
        ]
        assert message(app) == "Select amount of factoring"
        await pilot.press("t")
        assert work_area(app)[-4:] == [
            "         2",
            "      2·x ·(5·x + 3·y)",
            "#2:  ──────────────────",
            "              y",
        ]
        assert message(app).startswith("Compute time:")
        assert annotation(app) == "Fctr(#1)"


async def test_a_number_is_decomposed_without_being_asked_about(app):
    async with app.run_test() as pilot:
        authored(app, "1234567890")
        await pilot.press("f", "enter")
        assert entries(app)[-1] == "2*3^2*5*3607*3803"
        assert message(app).startswith("Compute time:")
        assert highlighted_menu_option(app) == "Author"


async def test_the_amount_menu_opens_on_what_was_chosen_last(app):
    async with app.run_test() as pilot:
        authored(app, "x^2 - 4")
        # One variable is no choice, so none is asked for.
        await pilot.press("f", "enter")
        assert message(app) == "Select amount of factoring"
        assert highlighted_menu_option(app) == "Rational"
        await pilot.press("t")
        authored(app, "x^2 - 9")
        await pilot.press("f", "enter")
        assert highlighted_menu_option(app) == "Trivial"


async def test_the_variables_are_collected_one_at_a_time(app):
    async with app.run_test() as pilot:
        authored(app, "x^2 y^2 - x^2 - y^4 + y^2")
        await pilot.press("f", "enter")
        assert message(app) == "Return for all or select 1: x,y"
        await pilot.press("x", "enter")
        assert prompt(app) == ("FACTOR variable 2:", "")
        assert message(app) == "Return for no more or select next: y"
        # Ending the list leaves y out, which is what keeps y^2 - 1 whole.
        await pilot.press("enter", "r")
        assert entries(app)[-1] == "(x + y)*(x - y)*(y^2 - 1)"


async def test_choosing_the_last_variable_ends_the_list(app):
    async with app.run_test() as pilot:
        authored(app, "x^2 y^2 - x^2 - y^4 + y^2")
        await pilot.press("f", "enter")
        await pilot.press("x", "enter")
        await pilot.press("y", "enter")
        assert message(app) == "Select amount of factoring"
        await pilot.press("r")
        assert entries(app)[-1] == "(x + y)*(x - y)*(y + 1)*(y - 1)"


async def test_a_name_that_is_no_variable_is_refused(app):
    async with app.run_test() as pilot:
        authored(app, "x^2 - y^2")
        await pilot.press("f", "enter")
        await pilot.press("q", "enter")
        # The question is put again, unchanged and with nothing chosen.
        assert prompt(app) == ("FACTOR variable 1:", "")
        assert message(app) == "Return for all or select 1: x,y"


# -- Expand, which asks the same three questions ------------------------------
#
# Every screen asserted here was checked against the original. Expand shares Factor's
# flow, but asks for an amount only for a ratio (which has a denominator to factor), and
# its amount menu lacks Complex.


async def test_expand_asks_for_the_expression_and_answers(app):
    async with app.run_test() as pilot:
        authored(app, "2x(x - 3)^2")
        await pilot.press("e")
        assert prompt(app) == ("EXPAND expression:", "#1")
        assert message(app) == "Enter expression"
        await pilot.press("enter")
        assert entries(app)[-1] == "2*x^3 - 12*x^2 + 18*x"
        assert message(app).startswith("Compute time:")
        assert annotation(app) == "Expd(#1)"
        assert highlighted_menu_option(app) == "Author"


async def test_expand_asks_for_the_variables_when_there_are_two(app):
    async with app.run_test() as pilot:
        authored(app, "(x + 2y + 1)^3")
        await pilot.press("e", "enter")
        assert prompt(app) == ("EXPAND variable 1:", "")
        assert message(app) == "Return for all or select 1: x,y"
        await pilot.press("x", "enter")
        assert prompt(app) == ("EXPAND variable 2:", "")
        assert message(app) == "Return for no more or select next: y"
        # Ending the list leaves y out, which is what keeps 2*y + 1 whole.
        await pilot.press("enter")
        assert entries(app)[-1] == (
            "x^3 + 3*x^2*(2*y + 1) + 3*x*(2*y + 1)^2 + (2*y + 1)^3"
        )


async def test_a_ratio_is_asked_about_and_a_polynomial_is_not(app):
    async with app.run_test() as pilot:
        authored(app, "1/(x^2 - 1)")
        await pilot.press("e", "enter")
        assert band(app) == [
            " EXPAND: Amount: Trivial Squarefree Rational raDical"
        ]
        assert message(app) == "Select amount of factoring"
        assert highlighted_menu_option(app) == "Rational"
        await pilot.press("r")
        assert entries(app)[-1] == "1/(2*(x - 1)) - 1/(2*(x + 1))"


async def test_the_two_amount_menus_remember_separately(app):
    """As in the original, Factor left on Trivial still opens Expand on Rational."""
    async with app.run_test() as pilot:
        authored(app, "x^2 - 4")
        await pilot.press("f", "enter", "t")
        authored(app, "1/(x^2 - 4)")
        await pilot.press("e", "enter")
        assert highlighted_menu_option(app) == "Rational"
        await pilot.press("escape")
        authored(app, "x^2 - 9")
        await pilot.press("f", "enter")
        assert highlighted_menu_option(app) == "Trivial"


async def test_the_variables_come_before_the_amount(app):
    async with app.run_test() as pilot:
        authored(app, "(x + 1)^2/(a^2 - 1)")
        await pilot.press("e", "enter")
        assert prompt(app) == ("EXPAND variable 1:", "")
        assert message(app) == "Return for all or select 1: x,a"
        await pilot.press("x", "enter")
        # Ending the variable list is what brings the amount menu up.
        await pilot.press("enter")
        assert message(app) == "Select amount of factoring"
        await pilot.press("r")
        assert entries(app)[-1] == "x^2/(a^2 - 1) + 2*x/(a^2 - 1) + 1/(a^2 - 1)"


async def test_expand_of_a_part_copies_the_rest_of_the_expression(app):
    """The manual's own example of highlighting a subexpression."""
    async with app.run_test() as pilot:
        authored(app, "2x(x - 3)^2")
        await pilot.press("right", "end")
        assert highlighted_expression(app) == "       2\n(x - 3)"
        await pilot.press("e", "enter")
        assert work_area(app)[-1] == "#2:  2·x·(x  - 6·x + 9)"
        assert annotation(app) == "Expd(#1')"


# -- soLve, which asks up to three questions and appends any number of answers -
#
# Every screen asserted here was checked against the original. The variable line is
# unnumbered, as the original asks it once per variable needed without saying which,
# and the interval is asked only in Approximate precision.


async def test_solve_asks_for_the_expression_and_appends_every_solution(app):
    async with app.run_test() as pilot:
        authored(app, "x^2 - 5x + 6 = 0")
        await pilot.press("l")
        assert prompt(app) == ("SOLVE expression:", "#1")
        assert message(app) == "Enter expression"
        # One variable settles the question, so none is asked.
        await pilot.press("enter")
        assert entries(app) == ["x^2-5*x+6=0", "x = 2", "x = 3"]
        assert highlighted_expression(app) == "x = 3"
        assert message(app).startswith("Compute time:")
        assert annotation(app) == "Solve(#1)"
        assert highlighted_menu_option(app) == "Author"


async def test_two_variables_bring_the_variable_line_up(app):
    async with app.run_test() as pilot:
        authored(app, "a x + b = 0")
        await pilot.press("l", "enter")
        # The most main variable is offered, and typing replaces it.
        assert prompt(app) == ("SOLVE variable:", "x")
        assert message(app) == "Enter variable"
        await pilot.press("enter")
        assert entries(app)[-1] == "x = -b/a"


async def test_the_offered_variable_can_be_typed_over(app):
    async with app.run_test() as pilot:
        authored(app, "a x + b = 0")
        await pilot.press("l", "enter")
        await pilot.press("a", "enter")
        assert entries(app)[-1] == "a = -b/x"


async def test_an_underdetermined_system_is_asked_about_once_per_equation(app):
    async with app.run_test() as pilot:
        authored(app, "[x + y + z = 1, x - y = 0]")
        await pilot.press("l", "enter")
        assert prompt(app) == ("SOLVE variable:", "x")
        await pilot.press("enter")
        # The next default is the next variable not yet chosen.
        assert prompt(app) == ("SOLVE variable:", "y")
        await pilot.press("enter")
        assert entries(app)[-1] == "[x = (1 - z)/2, y = (1 - z)/2]"


async def test_a_square_system_is_asked_nothing(app):
    async with app.run_test() as pilot:
        authored(app, "[x + y = 3, x - y = 1]")
        await pilot.press("l", "enter")
        assert entries(app)[-1] == "[x = 2, y = 1]"


async def test_a_name_that_is_no_variable_leaves_the_line_up(app):
    async with app.run_test() as pilot:
        authored(app, "a x + b = 0")
        await pilot.press("l", "enter")
        await pilot.press("backspace", *"2", "enter")
        assert prompt(app) == ("SOLVE variable:", "2")
        assert entries(app) == ["a*x+b=0"]


async def test_approximate_precision_asks_for_the_interval(app):
    async with app.run_test() as pilot:
        authored(app, "Precision := Approximate")
        authored(app, "x^5 - x + 1 = 0")
        await pilot.press("l", "enter")
        assert band(app) == [" SOLVE: Lower: -10                Upper: 10"]
        assert message(app) == "Enter bound on solution"
        await pilot.press("enter")
        assert entries(app)[-1] == "x = -1.16730"


async def test_the_other_two_precisions_never_ask_for_an_interval(app):
    async with app.run_test() as pilot:
        authored(app, "Precision := Mixed")
        authored(app, "3^x = x^2")
        await pilot.press("l", "enter")
        assert entries(app)[-1] == "x = -0.686026"


async def test_no_solutions_appends_nothing_and_says_so(app):
    async with app.run_test() as pilot:
        authored(app, "x^2 - 4")
        authored(app, "x = x + 1")
        await pilot.press("l", "enter")
        assert message(app) == "No solutions found"
        assert entries(app) == ["x^2-4", "x=x+1"]
        # Nothing was appended, so nothing moved the highlight off #2.
        assert highlighted_expression(app) == "x = x + 1"


async def test_solving_twice_appends_the_answer_twice(app):
    async with app.run_test() as pilot:
        authored(app, "2x = 8")
        await pilot.press("l", "enter")
        await pilot.press("ctrl+home")
        await pilot.press("l", "enter")
        assert entries(app) == ["2*x=8", "x = 4", "x = 4"]


async def test_the_arbitrary_counter_runs_across_the_session(app):
    async with app.run_test() as pilot:
        authored(app, "x = x")
        await pilot.press("l", "enter")
        assert entries(app)[-1] == "x = @1"
        authored(app, "2y = y + y")
        await pilot.press("l", "enter")
        assert entries(app)[-1] == "y = @2"


# -- approX, which asks the one question Simplify asks ------------------------
#
# Every screen asserted here was checked against the original. The command is Simplify
# at approximate precision. It doesn't ask for digits, which come from Options
# Precision.


async def test_approx_offers_the_highlighted_expression(app):
    async with app.run_test() as pilot:
        authored(app, "pi")
        await pilot.press("x")
        assert prompt(app) == ("APPROX expression:", "#1")
        assert message(app) == "Enter expression"
        await pilot.press("enter")
        assert work_area(app) == ["#1:  π", "", "#2:  3.14159"]
        assert highlighted_expression(app) == "3.14159"
        assert message(app).startswith("Compute time:")
        assert annotation(app) == "Approx(#1)"
        assert highlighted_menu_option(app) == "Author"


async def test_approx_leaves_the_precision_setting_alone(app):
    """approX is approximate for that command only, as a later Simplify shows."""
    async with app.run_test() as pilot:
        authored(app, "1/3")
        await pilot.press("x", "enter")
        assert entries(app)[-1] == "0.333333"
        await pilot.press("s")
        assert prompt(app) == ("SIMPLIFY expression:", "#2")
        await pilot.press(*"1")
        await pilot.press("enter")
        assert entries(app)[-1] == "1/3"


async def test_an_abandoned_approx_leaves_the_history_alone(app):
    async with app.run_test() as pilot:
        authored(app, "pi")
        await pilot.press("x", "escape")
        assert entries(app) == ["pi"]
        assert message(app) == "Enter option"
        assert highlighted_menu_option(app) == "Author"


# -- Remove and Unremove ------------------------------------------------------
#
# Every screen asserted here was checked against the original.


def numbered(app):
    """The history as `#label: text` lines, in linear notation.

    For the rendered 2D display, use `work_area` instead.
    """
    return [f"#{entry.number}: {entry.text}" for entry in app.session.entries]


def selected_number(app):
    """The label of the expression the highlight is on."""
    return app.session.selected_entry.number


async def test_remove_offers_the_highlighted_expression_at_both_ends(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y", "z", "w", "v")
        await pilot.press("r")
        assert band(app) == [" REMOVE: Start: 5      End: 5"]
        assert message(app) == "Enter label number"
        await pilot.press("2", "tab", "4", "enter")
        assert numbered(app) == ["#1: x", "#5: v"]
        assert message(app) == "Enter option"
        assert highlighted_menu_option(app) == "Author"


async def test_enter_alone_removes_the_highlighted_expression(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y", "z")
        await pilot.press("r", "enter")
        assert numbered(app) == ["#1: x", "#2: y"]


async def test_the_history_keys_all_label_the_field(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y", "z", "w")
        await pilot.press("r")
        await pilot.press("ctrl+home")
        assert band(app) == [" REMOVE: Start: 1      End: 4"]
        assert highlighted_expression(app) == "x"
        await pilot.press("down")
        assert band(app) == [" REMOVE: Start: 2      End: 4"]
        await pilot.press("ctrl+end")
        assert band(app) == [" REMOVE: Start: 4      End: 4"]
        await pilot.press("tab", "up")
        # The highlight moved with the key, and the End field took it.
        assert band(app) == [" REMOVE: Start: 4      End: 3"]
        assert highlighted_expression(app) == "z"
        await pilot.press("up")
        assert band(app) == [" REMOVE: Start: 4      End: 2"]
        await pilot.press("enter")
        assert numbered(app) == ["#1: x"]


async def test_a_label_that_names_no_expression_leaves_the_question_up(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y", "z")
        await pilot.press("r", "9", "tab", "9", "enter")
        assert band(app) == [" REMOVE: Start: 9      End: 9"]
        assert message(app) == "Enter label number"
        assert numbered(app) == ["#1: x", "#2: y", "#3: z"]


async def test_escape_abandons_remove_and_leaves_the_history_alone(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y")
        await pilot.press("r", "1", "tab", "2")
        await pilot.press("escape")
        assert numbered(app) == ["#1: x", "#2: y"]
        assert message(app) == "Enter option"
        assert highlighted_menu_option(app) == "Author"


async def test_unremove_asks_where_and_puts_them_back(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y", "z", "w", "v")
        await pilot.press("r", "2", "tab", "4", "enter")
        await pilot.press("u")
        assert band(app) == [" UNREMOVE: Before: 5"]
        assert message(app) == 'Enter label number or type "end"'
        await pilot.press("enter")
        assert numbered(app) == ["#1: x", "#2: y", "#3: z", "#4: w", "#5: v"]
        assert highlighted_expression(app) == "v"


async def test_end_puts_them_after_the_last_expression(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y", "z")
        await pilot.press("r", "1", "tab", "2", "enter")
        await pilot.press("u")
        await pilot.press(*"END", "enter")
        assert numbered(app) == ["#3: z", "#1: x", "#2: y"]
        assert highlighted_expression(app) == "y"


async def test_unremove_says_so_when_there_is_nothing_to_put_back(app):
    async with app.run_test() as pilot:
        authored(app, "x")
        await pilot.press("u")
        assert message(app) == "Unremove buffer empty"
        assert band(app)[0].startswith(" COMMAND:")


async def test_an_engine_command_empties_the_buffer(app):
    async with app.run_test() as pilot:
        authored(app, "2 + 3", "y")
        await pilot.press("r", "2", "tab", "2", "enter")
        await pilot.press("s", "enter")
        await pilot.press("u")
        assert message(app) == "Unremove buffer empty"


async def test_unremove_asks_nothing_when_the_history_is_empty(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y")
        await pilot.press("r", "1", "tab", "2", "enter")
        assert numbered(app) == []
        await pilot.press("u")
        assert numbered(app) == ["#1: x", "#2: y"]
        assert message(app) == "Enter option"


async def test_the_buffer_outlives_the_unremove_that_used_it(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y", "z")
        await pilot.press("r", "1", "tab", "2", "enter")
        await pilot.press("u", "3", "enter")
        assert numbered(app) == ["#1: x", "#2: y", "#3: z"]
        # The same block again, under labels that are free this time.
        await pilot.press("u", "3", "enter")
        assert numbered(app) == ["#1: x", "#2: y", "#4: x", "#5: y", "#3: z"]


# -- moVe ---------------------------------------------------------------------
#
# Every screen asserted here matches the original.


async def test_move_offers_the_highlighted_expression_in_all_three_fields(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y", "z", "w", "v")
        await pilot.press("v")
        assert band(app) == [" MOVE: Before: 5      Start: 5      End: 5"]
        assert message(app) == 'Enter label number or type "end"'
        await pilot.press("1", "tab", "3", "tab", "4", "enter")
        assert numbered(app) == ["#3: z", "#4: w", "#1: x", "#2: y", "#5: v"]
        assert highlighted_expression(app) == "x"
        assert message(app) == "Enter option"
        assert highlighted_menu_option(app) == "Author"


async def test_only_the_destination_field_asks_for_a_word(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y")
        await pilot.press("v", "tab")
        assert message(app) == "Enter label number"
        await pilot.press("tab")
        assert message(app) == "Enter label number"
        await pilot.press("tab")
        assert message(app) == 'Enter label number or type "end"'


async def test_end_sends_the_block_past_the_last_expression(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y", "z")
        await pilot.press("v")
        await pilot.press(*"END", "tab", "1", "tab", "2", "enter")
        assert numbered(app) == ["#3: z", "#1: x", "#2: y"]
        assert highlighted_expression(app) == "y"


async def test_enter_alone_moves_the_highlighted_expression_nowhere(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y", "z")
        await pilot.press("v", "enter")
        assert numbered(app) == ["#1: x", "#2: y", "#3: z"]
        assert highlighted_expression(app) == "z"


async def test_the_arrows_walk_the_history_and_label_the_move_field(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y", "z", "w")
        await pilot.press("v", "up", "up", "up")
        # The highlight moved with the key, and the Before field took it.
        assert band(app) == [" MOVE: Before: 1      Start: 4      End: 4"]
        assert highlighted_expression(app) == "x"
        await pilot.press("enter")
        assert numbered(app) == ["#4: w", "#1: x", "#2: y", "#3: z"]


async def test_a_label_that_names_no_expression_leaves_the_move_question_up(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y")
        await pilot.press("v", "9", "tab", "1", "tab", "1", "enter")
        assert band(app) == [" MOVE: Before: 9      Start: 1      End: 1"]
        # The refusal puts focus back on the wrong field, ready for correction.
        assert message(app) == 'Enter label number or type "end"'
        await pilot.press("2")
        assert band(app) == [" MOVE: Before: 2      Start: 1      End: 1"]
        assert numbered(app) == ["#1: x", "#2: y"]


async def test_the_field_a_move_was_refused_over_is_the_one_it_goes_back_to(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y", "z")
        await pilot.press("v", "1", "tab", "9", "tab", "3", "enter")
        assert message(app) == "Enter label number"
        await pilot.press("3")
        assert band(app) == [" MOVE: Before: 1      Start: 3      End: 3"]
        await pilot.press("enter")
        assert numbered(app) == ["#3: z", "#1: x", "#2: y"]


async def test_escape_abandons_move_and_leaves_the_history_alone(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y")
        await pilot.press("v", "1", "tab", "2", "tab", "2")
        await pilot.press("escape")
        assert numbered(app) == ["#1: x", "#2: y"]
        assert message(app) == "Enter option"
        assert highlighted_menu_option(app) == "Author"


# -- Declare, whose four commands ask their questions in every form -----------
#
# Every screen asserted here was checked against the original.


async def test_declare_lists_the_four_things_that_can_be_declared(app):
    async with app.run_test() as pilot:
        await pilot.press("d")
        assert band(app) == [" DECLARE: Function Variable Matrix vectoR"]
        assert highlighted_menu_option(app) == "Function"
        assert message(app) == "Enter option"


async def test_a_variable_is_named_then_given_a_domain_then_an_interval(app):
    async with app.run_test() as pilot:
        await pilot.press("d", "v")
        assert prompt(app) == ("DECLARE VARIABLE name:", "")
        assert message(app) == 'Enter name or type "default"'
        await pilot.press(*"x", "enter")
        assert band(app) == [
            " DECLARE VARIABLE: Value Integer Real Complex Nonscalar"
        ]
        assert highlighted_menu_option(app) == "Real"
        assert message(app) == "Select value or domain of x"
        await pilot.press("r")
        assert band(app) == [
            " DECLARE VARIABLE: All Positive Negative nonpoSitive nonneGative Interval"
        ]
        assert highlighted_menu_option(app) == "All"
        assert message(app) == "Select interval of x"
        await pilot.press("p")
        assert work_area(app) == ["#1:  x :ε Real (0, ∞)"]
        assert annotation(app) == "User"
        assert highlighted_menu_option(app) == "Author"


async def test_a_domain_with_no_interval_is_the_last_question(app):
    async with app.run_test() as pilot:
        await pilot.press("d", "v")
        await pilot.press(*"z", "enter", "c")
        assert work_area(app) == ["#1:  z :ε Complex"]
        assert highlighted_menu_option(app) == "Author"


async def test_the_bounds_screen_asks_for_both_ends_and_their_strictness(app):
    async with app.run_test() as pilot:
        await pilot.press("d", "v")
        await pilot.press(*"n", "enter", "i", "i")
        assert band(app) == [
            " DECLARE VARIABLE: Bounds: -∞      (<)≤  n  (<)≤   ∞"
        ]
        assert message(app) == "Enter left bound"
        await pilot.press("1", "delete", "tab", "space")
        # The live field shows both symbols and highlights the one in force;
        # a field that is not live parenthesizes it instead.
        assert band(app)[0].endswith("Bounds: 1        < ≤  n  (<)≤   ∞")
        assert highlighted_menu_option(app) == "≤"
        await pilot.press("tab")
        assert message(app) == "Enter right bound"
        assert band(app)[0].endswith("Bounds: 1        <(≤)  n   < ≤   ∞")
        await pilot.press("tab", "5", "enter")
        assert work_area(app) == ["#1:  n :ε Integer [1, 5)"]


async def test_a_bound_that_is_not_a_number_is_not_taken(app):
    async with app.run_test() as pilot:
        await pilot.press("d", "v")
        await pilot.press(*"p", "enter", "r", "i")
        await pilot.press(*"a+1", "delete", "enter")
        # The question stays up, with what was typed still on it to correct.
        assert band(app)[0].startswith(" DECLARE VARIABLE: Bounds: a+1")
        assert entries(app) == []


async def test_a_variable_is_given_a_value_on_a_line_of_its_own(app):
    async with app.run_test() as pilot:
        await pilot.press("d", "v")
        await pilot.press(*"area", "enter", "v")
        assert prompt(app) == ("DECLARE VARIABLE value:", "")
        assert message(app) == "Enter expression"
        await pilot.press(*"pi r^2", "enter")
        assert entries(app) == ["area:=pi*r^2"]
        assert work_area(app)[-1] == "#1:  area := π·r"


async def test_the_domain_menu_opens_on_what_the_variable_already_is(app):
    async with app.run_test() as pilot:
        await pilot.press("d", "v")
        await pilot.press(*"area", "enter", "v")
        await pilot.press(*"5", "enter")
        await pilot.press("d", "v")
        await pilot.press(*"area", "enter")
        assert highlighted_menu_option(app) == "Value"


async def test_a_pre_defined_name_is_refused_and_the_question_put_again(app):
    async with app.run_test() as pilot:
        await pilot.press("d", "v")
        await pilot.press(*"sin", "enter")
        assert prompt(app) == ("DECLARE VARIABLE name:", "")
        assert message(app) == 'Enter name or type "default"'


async def test_a_function_takes_its_parameters_from_its_definition(app):
    async with app.run_test() as pilot:
        await pilot.press("d", "f")
        assert prompt(app) == ("DECLARE FUNCTION name:", "")
        assert message(app) == "Enter name"
        await pilot.press(*"hyp", "enter")
        assert prompt(app) == ("DECLARE FUNCTION value:", "")
        assert message(app) == "Enter expression"
        await pilot.press(*"sqrt(a^2+b^2)", "enter")
        assert work_area(app)[-1] == "#1:  HYP(a, b) := √(a  + b )"
        assert highlighted_menu_option(app) == "Author"


async def test_a_blank_definition_asks_for_the_variables_instead(app):
    async with app.run_test() as pilot:
        await pilot.press("d", "f")
        await pilot.press(*"f", "enter", "enter")
        assert prompt(app) == ("DECLARE FUNCTION variable:", "")
        assert message(app) == "Enter variable or press ENTER"
        await pilot.press(*"x", "enter")
        await pilot.press(*"y", "enter")
        await pilot.press("enter")
        assert entries(app) == ["F(x,y):="]
        assert work_area(app) == ["#1:  F(x, y) :="]


async def test_a_vector_asks_for_a_dimension_then_an_element_at_a_time(app):
    async with app.run_test() as pilot:
        await pilot.press("d", "r")
        assert band(app) == [" DECLARE VECTOR: Dimension:"]
        assert message(app) == "Enter number of elements"
        await pilot.press("3", "enter")
        assert prompt(app) == ("VECTOR element:", "")
        assert message(app) == "Enter vector element 1"
        await pilot.press("1", "enter")
        assert message(app) == "Enter vector element 2"
        await pilot.press("2", "enter")
        await pilot.press(*"x", "enter")
        assert entries(app) == ["[1,2,x]"]
        assert work_area(app) == ["#1:  [1, 2, x]"]


async def test_a_matrix_asks_for_its_shape_and_offers_zero_for_every_cell(app):
    async with app.run_test() as pilot:
        await pilot.press("d", "m")
        assert band(app) == [" DECLARE MATRIX: Rows: 3      Columns: 3"]
        assert message(app) == "Enter number of rows"
        await pilot.press("2", "tab", "2", "enter")
        assert prompt(app) == ("MATRIX element:", "0")
        assert message(app) == "Enter matrix element (1,1)"
        await pilot.press("1", "enter")
        assert message(app) == "Enter matrix element (1,2)"
        await pilot.press("enter", "enter", "4", "enter")
        assert entries(app) == ["[[1,0],[0,4]]"]
        assert work_area(app) == ["     ┌ 1  0 ┐", "#1:  │      │", "     └ 0  4 ┘"]


async def test_a_shape_is_offered_again_the_next_time_one_is_asked_for(app):
    async with app.run_test() as pilot:
        await pilot.press("d", "m")
        await pilot.press("2", "tab", "2", "enter")
        await pilot.press("enter", "enter", "enter", "enter")
        await pilot.press("d", "m")
        assert band(app) == [" DECLARE MATRIX: Rows: 2      Columns: 2"]
        await pilot.press("escape", "escape")
        await pilot.press("d", "r", "2", "enter")
        await pilot.press("1", "enter", "2", "enter")
        await pilot.press("d", "r")
        assert band(app) == [" DECLARE VECTOR: Dimension: 2"]


async def test_an_element_that_does_not_read_leaves_the_line_up(app):
    async with app.run_test() as pilot:
        await pilot.press("d", "r", "2", "enter")
        await pilot.press(*"2+*", "enter")
        assert message(app) == "Syntax error detected at cursor"
        assert prompt(app) == ("VECTOR element:", "2+*")


@pytest.mark.parametrize(
    ("keys", "step"),
    [
        (("d", "v", "enter"), "no name"),
        (("d", "f", "enter"), "no function name"),
        (("d", "r", "2", "enter", "enter"), "no vector element"),
    ],
    ids=str,
)
async def test_a_blank_answer_that_names_nothing_abandons_the_command(app, keys, step):
    async with app.run_test() as pilot:
        await pilot.press(*keys)
        assert band(app) == [" DECLARE: Function Variable Matrix vectoR"]
        assert entries(app) == []


# -- Build, which asks for operands and operators until Done ------------------
#
# Every screen asserted here was checked against the original, and is the same in all
# its versions.


async def test_build_asks_for_an_operand_then_an_operator_then_another(app):
    async with app.run_test() as pilot:
        authored(app, "x^2*y")
        await pilot.press("b")
        assert prompt(app) == ("BUILD first expression:", "#1")
        assert message(app) == "Enter expression"
        await pilot.press("enter")
        operators = "+ - * / ^ . ` = Minus Recip Ln Exp Tan Sin Cos Atan ! % Done"
        assert band(app) == [f" BUILD: Operator: {operators}"]
        assert message(app) == "Select operator"
        assert highlighted(app) == "+"
        await pilot.press("+")
        assert prompt(app) == ("BUILD next expression:", "#1")
        await pilot.press("enter")
        # After an operand, the menu highlight is on Done.
        assert highlighted(app) == "Done"
        await pilot.press("d")
        assert entries(app) == ["x^2*y", "x^2*y+x^2*y"]
        assert annotation(app) == "#1+#1"
        assert band(app)[0].startswith(" COMMAND:")


async def test_a_unary_operator_asks_for_nothing_more(app):
    async with app.run_test() as pilot:
        authored(app, "x")
        await pilot.press("b", "enter")
        await pilot.press("s")
        assert band(app)[0].startswith(" BUILD: Operator:")
        assert highlighted(app) == "Done"
        await pilot.press("d")
        assert entries(app) == ["x", "SIN(x)"]
        assert annotation(app) == "SIN(#1)"


async def test_the_operators_chain_until_done(app):
    async with app.run_test() as pilot:
        authored(app, "2")
        authored(app, "3")
        authored(app, "4")
        await pilot.press("b", "up", "up", "enter")
        await pilot.press("+", "down", "enter")
        await pilot.press("*", "down", "enter")
        await pilot.press("d")
        # Built left to right with the needed parentheses; the annotation is
        # written flat, as the original writes it.
        assert entries(app)[-1] == "(2+3)*4"
        assert annotation(app) == "#1+#2*#3"


async def test_unary_operators_nest(app):
    async with app.run_test() as pilot:
        authored(app, "x")
        await pilot.press("b", "enter", "s", "c", "d")
        assert entries(app)[-1] == "COS(SIN(x))"
        assert annotation(app) == "COS(SIN(#1))"


async def test_the_symbols_are_their_own_mnemonics(app):
    async with app.run_test() as pilot:
        authored(app, "x")
        await pilot.press("b", "enter", "!")
        assert highlighted(app) == "Done"
        await pilot.press("d")
        assert entries(app)[-1] == "x!"


async def test_space_steps_the_operator_menu(app):
    async with app.run_test() as pilot:
        authored(app, "x")
        await pilot.press("b", "enter")
        await pilot.press("space", "space")
        assert highlighted(app) == "*"


async def test_the_arrow_keys_pick_the_operand_off_the_screen(app):
    async with app.run_test() as pilot:
        authored(app, "2")
        authored(app, "3")
        await pilot.press("b")
        assert prompt(app) == ("BUILD first expression:", "#2")
        await pilot.press("up")
        assert prompt(app) == ("BUILD first expression:", "#1")


async def test_a_highlighted_part_is_what_is_built_from(app):
    async with app.run_test() as pilot:
        authored(app, "SIN(a*x^2) + 5")
        await pilot.press("f6", "right", "down")
        assert highlighted_expression(app) == "   2\na·x"
        await pilot.press("b")
        # The line names the entry, but the highlighted part is what's used.
        assert prompt(app) == ("BUILD first expression:", "#1")
        await pilot.press("enter", "d")
        assert entries(app)[-1] == "a*x^2"
        assert annotation(app) == "#1'"


async def test_build_takes_a_typed_expression_too(app):
    async with app.run_test() as pilot:
        authored(app, "x")
        await pilot.press("b")
        await pilot.press("backspace", "backspace")
        await pilot.press(*"2+3", "enter")
        await pilot.press("d")
        assert entries(app)[-1] == "2+3"
        assert annotation(app) == "User"


async def test_a_line_that_does_not_read_stays_up(app):
    async with app.run_test() as pilot:
        authored(app, "x")
        await pilot.press("b")
        await pilot.press("backspace", "backspace")
        await pilot.press(*"2+*", "enter")
        assert message(app) == "Syntax error detected at cursor"
        assert prompt(app) == ("BUILD first expression:", "2+*")


async def test_a_blank_operand_line_abandons_the_build(app):
    async with app.run_test() as pilot:
        authored(app, "x")
        await pilot.press("b")
        await pilot.press("backspace", "backspace", "enter")
        assert band(app)[0].startswith(" COMMAND:")
        assert entries(app) == ["x"]


async def test_build_asks_the_same_question_of_an_empty_history(app):
    async with app.run_test() as pilot:
        await pilot.press("b")
        assert prompt(app) == ("BUILD first expression:", "")
        assert message(app) == "Enter expression"


async def test_ctrl_enter_on_done_enters_one_expression(app):
    async with app.run_test() as pilot:
        authored(app, "x^2*y")
        await pilot.press("b", "enter", "+", "enter")
        await pilot.press("ctrl+j")
        # One entry, not two: the built expression isn't added, and the annotation
        # records both how it was built and that it was simplified.
        assert entries(app) == ["x^2*y", "2*x^2*y"]
        assert annotation(app) == "Simp(#1+#1)"


# -- Calculus, whose seven commands ask an expression, a variable, and a line -


async def test_the_calculus_menu_lists_seven_commands(app):
    async with app.run_test() as pilot:
        await pilot.press("c")
        assert band(app) == [
            " CALCULUS: Differentiate Integrate Limit Product Sum Taylor Vector"
        ]
        assert message(app) == "Enter option"
        assert highlighted(app) == "Differentiate"


async def test_differentiate_asks_expression_then_variable_then_order(app):
    async with app.run_test() as pilot:
        authored(app, "x^2*y")
        await pilot.press("c", "d")
        assert prompt(app) == ("CALCULUS DIFFERENTIATE expression:", "#1")
        assert message(app) == "Enter expression"
        await pilot.press("enter")
        assert prompt(app) == ("CALCULUS DIFFERENTIATE variable:", "x")
        assert message(app) == "Enter variable"
        await pilot.press("enter")
        assert band(app) == [" CALCULUS DIFFERENTIATE: Order: 1"]
        assert message(app) == "Enter expression"
        await pilot.press("enter")
        assert work_area(app)[-3:] == ["     d    2", "#2:  ── (x ·y)", "     dx"]
        assert entries(app)[-1] == "DIF(x^2*y,x)"
        assert annotation(app) == "Dif(#1,x)"
        assert band(app)[0].startswith(" COMMAND:")


#: The line each command finishes on, after its expression and its variable.
LAST_LINES = {
    "d": " CALCULUS DIFFERENTIATE: Order: 1",
    "i": " CALCULUS INTEGRATE: Lower limit:                    Upper limit:",
    "l": " CALCULUS LIMIT: Point: 0                  From:(Both)Left Right",
    "p": " CALCULUS PRODUCT: Lower limit: 1                  Upper limit: n",
    "s": " CALCULUS SUM: Lower limit: 1                  Upper limit: n",
    "t": " CALCULUS TAYLOR: Degree: 5                  Point: 0",
    "v": " CALCULUS VECTOR: Start: 1               End:                 Step: 1",
}


@pytest.mark.parametrize(
    ("keys", "text", "note"),
    [
        (("d", "enter"), "DIF(x^2*y,x)", "the order is left off when it is one"),
        (("d", "3", "enter"), "DIF(x^2*y,x,3)", "and written when it is not"),
        (("i", "enter"), "INT(x^2*y,x)", "no limits is the indefinite integral"),
        (("i", "0", "enter", "1", "enter"), "INT(x^2*y,x,0,1)", "both is definite"),
        (("l", "enter"), "LIM(x^2*y,x,0,0)", "Both is written as a zero"),
        (("l", "tab", "l", "enter"), "LIM(x^2*y,x,0,-1)", "Left as a minus one"),
        (("l", "tab", "r", "enter"), "LIM(x^2*y,x,0,1)", "Right as a one"),
        (("s", "enter"), "SUM(x^2*y,x,1,n)", "the limits offered are 1 to n"),
        (
            ("s", "delete", "tab", "delete", "enter"),
            "SUM(x^2*y,x)",
            "and blanking both asks for the antidifference",
        ),
        (("p", "enter"), "PRODUCT(x^2*y,x,1,n)", "a product asks the same"),
        (("t", "enter"), "TAYLOR(x^2*y,x,0,5)", "the point is written first"),
        (("t", "3", "tab", "1", "enter"), "TAYLOR(x^2*y,x,1,3)", "though asked last"),
        (("v", "5", "enter"), "VECTOR(x^2*y,x,1,5)", "the step is left off at one"),
        (("v", "5", "tab", "2", "enter"), "VECTOR(x^2*y,x,1,5,2)", "and written at two"),
    ],
    ids=str,
)
async def test_what_each_command_enters(app, keys, text, note):
    """The linear forms come from the original's own Transfer Save."""
    async with app.run_test() as pilot:
        authored(app, "x^2*y")
        await pilot.press("c", *keys[:1], "enter", "enter")
        assert band(app) == [LAST_LINES[keys[0]]]
        await pilot.press(*keys[1:])
        assert entries(app)[-1] == text


async def test_the_limit_direction_is_chosen_off_the_line(app):
    async with app.run_test() as pilot:
        authored(app, "x^2*y")
        await pilot.press("c", "l", "enter", "enter")
        assert message(app) == "Enter limit point"
        await pilot.press("tab")
        assert band(app) == [
            " CALCULUS LIMIT: Point: 0                  From: Both Left Right"
        ]
        assert message(app) == "Select approach direction"
        assert highlighted(app) == "Both"
        await pilot.press("space")
        assert highlighted(app) == "Left"
        # A mnemonic chooses and hands the line back to the point.
        await pilot.press("r")
        assert band(app) == [
            " CALCULUS LIMIT: Point: 0                  From: Both Left(Right)"
        ]
        assert message(app) == "Enter limit point"


async def test_the_taylor_line_says_which_field_it_is_asking_about(app):
    async with app.run_test() as pilot:
        authored(app, "x^2*y")
        await pilot.press("c", "t", "enter", "enter")
        assert message(app) == "Enter maximum degree"
        await pilot.press("tab")
        assert message(app) == "Enter expansion point"


async def test_the_vector_line_opens_on_the_field_it_cannot_do_without(app):
    async with app.run_test() as pilot:
        authored(app, "x^2*y")
        await pilot.press("c", "v", "enter", "enter")
        await pilot.press("7")
        assert band(app) == [
            " CALCULUS VECTOR: Start: 1               End: 7               Step: 1"
        ]


@pytest.mark.parametrize(
    ("keys", "step"),
    [
        (("v", "enter"), "an end the vector has to run to"),
        (("d", "delete", "enter"), "an order to differentiate to"),
        (("l", "delete", "enter"), "a point to take the limit at"),
    ],
    ids=str,
)
async def test_a_field_that_has_to_be_answered_refuses_to_commit(app, keys, step):
    async with app.run_test() as pilot:
        authored(app, "x^2*y")
        await pilot.press("c", *keys[:1], "enter", "enter", *keys[1:])
        assert entries(app) == ["x^2*y"]
        assert band(app)[0].startswith(" CALCULUS ")


async def test_one_limit_of_a_pair_sends_the_line_to_the_other(app):
    async with app.run_test() as pilot:
        authored(app, "x^2*y")
        await pilot.press("c", "i", "enter", "enter")
        await pilot.press("0", "enter")
        # Neither committed nor refused: focus moves to the empty limit, and
        # filling it commits.
        assert entries(app) == ["x^2*y"]
        assert band(app) == [
            " CALCULUS INTEGRATE: Lower limit: 0                  Upper limit:"
        ]
        await pilot.press("1", "enter")
        assert entries(app)[-1] == "INT(x^2*y,x,0,1)"


async def test_the_variable_offered_is_the_primary_one(app):
    async with app.run_test() as pilot:
        authored(app, "SIN(a*x^2)")
        await pilot.press("c", "d", "enter")
        assert prompt(app) == ("CALCULUS DIFFERENTIATE variable:", "x")


async def test_a_bound_variable_leaves_the_line_empty(app):
    async with app.run_test() as pilot:
        authored(app, "INT(x^2, x, 0, 1)")
        await pilot.press("c", "d", "enter")
        assert prompt(app) == ("CALCULUS DIFFERENTIATE variable:", "")


async def test_a_line_that_names_no_variable_is_refused_without_a_word(app):
    async with app.run_test() as pilot:
        authored(app, "x^2*y")
        await pilot.press("c", "d", "enter")
        await pilot.press("backspace")
        await pilot.press(*"2+3", "enter")
        assert prompt(app) == ("CALCULUS DIFFERENTIATE variable:", "2+3")
        assert message(app) == "Enter variable"
        assert entries(app) == ["x^2*y"]


async def test_a_highlighted_part_is_what_is_taken(app):
    async with app.run_test() as pilot:
        authored(app, "SIN(a*x^2) + 5")
        await pilot.press("f6", "right", "down")
        assert highlighted_expression(app) == "   2\na·x"
        await pilot.press("c", "d", "enter", "enter", "enter")
        # The part is extracted, not substituted: DIF wraps just the part.
        assert entries(app)[-1] == "DIF(a*x^2,x)"
        assert annotation(app) == "Dif(#1',x)"


async def test_a_blank_expression_line_abandons_the_command(app):
    async with app.run_test() as pilot:
        authored(app, "x^2*y")
        await pilot.press("c", "d")
        await pilot.press("backspace", "backspace", "enter")
        assert band(app) == [
            " CALCULUS: Differentiate Integrate Limit Product Sum Taylor Vector"
        ]
        assert entries(app) == ["x^2*y"]


async def test_ctrl_enter_on_the_last_field_enters_one_expression(app):
    async with app.run_test() as pilot:
        authored(app, "x^2*y")
        await pilot.press("c", "d", "enter", "enter")
        await pilot.press("ctrl+j")
        assert entries(app) == ["x^2*y", "2*x*y"]
        assert annotation(app) == "Simp(Dif(#1,x))"


async def test_nothing_is_computed_until_a_simplify_asks_for_it(app):
    async with app.run_test() as pilot:
        authored(app, "x^2*y")
        await pilot.press("c", "d", "enter", "enter", "enter")
        assert entries(app)[-1] == "DIF(x^2*y,x)"
        await pilot.press("s", "enter")
        assert entries(app)[-1] == "2*x*y"
        assert annotation(app) == "Simp(#2)"

# -- Esc in the middle of a command -------------------------------------------
#
# Esc abandons the command at any step, on a line, submenu or dialog, and returns to
# the menu the command was picked from.

COMMAND_MENU = " COMMAND:"
DECLARE_MENU = " DECLARE: Function Variable Matrix vectoR"
CALCULUS_MENU = " CALCULUS: Differentiate Integrate Limit Product Sum Taylor Vector"


@pytest.mark.parametrize(
    ("setup", "keys", "menu", "first"),
    [
        pytest.param(
            ["x^2 y^2 - 1"], ("f", "enter"), COMMAND_MENU, "Author", id="a line"
        ),
        pytest.param(
            ["x^2 y^2 - 1"],
            ("f", "enter", "enter"),
            COMMAND_MENU,
            "Author",
            id="the amount menu",
        ),
        pytest.param(
            ["Precision := Approximate", "x^5 - x + 1 = 0"],
            ("l", "enter"),
            COMMAND_MENU,
            "Author",
            id="the interval",
        ),
        pytest.param(
            ["x"], ("b", "enter"), COMMAND_MENU, "Author", id="the operator menu"
        ),
        pytest.param(
            [],
            ("d", "f", "f", "enter", "enter"),
            DECLARE_MENU,
            "Function",
            id="a Declare line",
        ),
        pytest.param(
            [],
            ("d", "v", "x", "enter"),
            DECLARE_MENU,
            "Function",
            id="the domain menu",
        ),
        pytest.param(
            [],
            ("d", "v", "x", "enter", "r", "i"),
            DECLARE_MENU,
            "Function",
            id="the bounds",
        ),
        pytest.param(
            ["x^2*y"],
            ("c", "d", "enter", "enter"),
            CALCULUS_MENU,
            "Differentiate",
            id="the order",
        ),
    ],
)
async def test_escape_abandons_the_command_for_the_menu_it_came_from(
    app, setup, keys, menu, first
):
    async with app.run_test() as pilot:
        authored(app, *setup)
        before = entries(app)
        await pilot.press(*keys)
        await pilot.press("escape")
        assert band(app)[0].startswith(menu)
        assert highlighted_menu_option(app) == first
        assert message(app) == "Enter option"
        assert entries(app) == before
        assert app.asking is None
        assert app.solving is None
        assert app.building is None
        assert app.calculating is None
        assert (app.declaring, app.defining, app.entering) == (None, None, None)


# -- Jump ---------------------------------------------------------------------
#
# Every screen asserted here was checked against the original.


async def test_jump_asks_for_a_label_and_highlights_what_it_names(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y", "z")
        await pilot.press("j")
        # Nothing is offered, as the current label would be pointless to jump to.
        assert prompt(app) == ("JUMP to:", "")
        assert message(app) == "Enter label number"
        await pilot.press("1", "enter")
        assert highlighted_expression(app) == "x"
        assert message(app) == "Enter option"
        assert highlighted_menu_option(app) == "Author"


async def test_a_label_no_expression_carries_lands_on_the_one_above_it(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y", "z")
        await pilot.press("r", "2", "tab", "2", "enter")
        await pilot.press("j", "2", "enter")
        assert highlighted_expression(app) == "z"


async def test_a_label_past_the_last_one_leaves_the_line_up(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y")
        await pilot.press("j", "9", "enter")
        assert prompt(app) == ("JUMP to:", "9")
        assert message(app) == "Enter label number"
        assert highlighted_expression(app) == "y"
        # The line stays up to be corrected.
        await pilot.press("backspace", "1", "enter")
        assert highlighted_expression(app) == "x"


async def test_a_line_that_is_no_label_number_is_refused(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y")
        await pilot.press("j")
        await pilot.press(*"1x", "enter")
        assert prompt(app) == ("JUMP to:", "1x")
        assert highlighted_expression(app) == "y"


async def test_a_jump_line_with_nothing_on_it_leaves_the_highlight_alone(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y")
        await pilot.press("j", "enter")
        assert highlighted_expression(app) == "y"
        assert message(app) == "Enter option"
        assert highlighted_menu_option(app) == "Author"


async def test_escape_abandons_the_jump_line(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y")
        await pilot.press("j", "1")
        await pilot.press("escape")
        assert highlighted_expression(app) == "y"
        assert message(app) == "Enter option"


async def test_jumping_to_the_expression_you_are_inside_of_keeps_the_part(app):
    async with app.run_test() as pilot:
        authored(app, "x (x + 1)")
        await pilot.press("right", "right")
        assert highlighted_expression(app) == "x + 1"
        await pilot.press("j", "1", "enter")
        assert highlighted_expression(app) == "x + 1"


@pytest.mark.parametrize(
    ("key", "line"),
    [
        ("s", "SIMPLIFY expression:"),
        ("f", "FACTOR expression:"),
        ("l", "SOLVE expression:"),
        ("r", None),
        ("v", None),
        ("j", None),
    ],
    ids=str,
)
async def test_an_empty_history_asks_for_nothing(app, key, line):
    """One command per code path: the lines shared by Simplify/approX and by
    Factor/Expand, soLve's line, and three commands that ask nothing when the history
    is empty."""
    async with app.run_test() as pilot:
        await pilot.press(key)
        if line is not None:
            # Nothing is offered, and Enter on the empty line does nothing.
            assert prompt(app) == (line, "")
            await pilot.press("enter")
        assert app.session.entries == []
        assert band(app)[0].startswith(" COMMAND:")
        assert message(app) == "Enter option"


# -- walking the history with a question up -----------------------------------
#
# Keys that move the highlight between expressions keep working while a prompt line is
# up; keys that move the cursor along the line belong to the line. Every screen
# asserted here was checked against the original.


async def test_the_history_keys_walk_while_a_prompt_line_is_up(app):
    async with app.run_test() as pilot:
        authored(app, "2 + 3", "y", "z")
        await pilot.press("s")
        assert prompt(app) == ("SIMPLIFY expression:", "#3")
        # The line takes the label of wherever the highlight lands, which the manual
        # recommends over typing the number.
        await pilot.press("up")
        assert prompt(app)[1] == "#2"
        assert highlighted_expression(app) == "y"
        await pilot.press("ctrl+home")
        assert prompt(app)[1] == "#1"
        await pilot.press("enter")
        assert numbered(app)[-1] == "#4: 5"
        assert annotation(app) == "Simp(#1)"


async def test_a_label_walked_onto_is_selected_as_the_offered_one_was(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y", "z")
        await pilot.press("s", "up")
        assert prompt(app)[1] == "#2"
        # Typing replaces it, just like the originally offered label.
        await pilot.press("3")
        assert prompt(app)[1] == "#3"


async def test_a_line_that_was_typed_on_is_left_as_it_is(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y", "z")
        await pilot.press("s")
        await pilot.press(*"2+3")
        assert prompt(app)[1] == "#2+3"
        await pilot.press("up")
        # The highlight moved, but the line, now edited, is kept.
        assert highlighted_expression(app) == "y"
        assert prompt(app)[1] == "#2+3"


async def test_a_cursor_key_hands_the_line_over_for_good(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y", "z")
        await pilot.press("s")
        # End deselects the offered label, and the line stops taking labels.
        await pilot.press("end", "up")
        assert highlighted_expression(app) == "y"
        assert prompt(app)[1] == "#3"


async def test_a_stretch_marked_out_by_hand_hands_the_line_over_too(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y", "z")
        await pilot.press("s")
        # A user-made selection isn't the offered label, even if it looks the same,
        # so walking leaves the line and selection alone.
        await pilot.press("home", "shift+end")
        await pilot.press("up")
        assert highlighted_expression(app) == "y"
        assert prompt(app)[1] == "#3"
        line = app.query_one("#prompt-input")
        assert line.selected_text == "#3"


async def test_the_cursor_keys_move_no_highlight_while_a_line_is_up(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y", "x (x + 1)")
        await pilot.press("s")
        for key in ("left", "right", "home", "end"):
            await pilot.press(key)
            assert highlighted_expression(app) == "x·(x + 1)"


# -- copying an expression and pasting it -------------------------------------
#
# Ctrl-C copies the highlighted expression and Ctrl-V pastes it on the line being
# typed. The copy goes to both the system clipboard and the app's own, but headless runs
# leave the system clipboard alone, so these tests check the app's copy.


async def test_ctrl_c_copies_the_highlighted_expression_and_ctrl_v_pastes_it(app):
    async with app.run_test() as pilot:
        authored(app, "x + 1", "y")
        await pilot.press("ctrl+c")
        assert app.clipboard == "y"
        # The message confirms the copy, which is otherwise invisible.
        assert message(app) == "Copied the highlighted expression"
        await pilot.press("a", "ctrl+v")
        assert prompt(app) == ("AUTHOR expression:", "y")


async def test_a_headless_copy_keeps_off_the_desktop_clipboard(app, monkeypatch):
    # Tests pressing Ctrl-C must not write to the developer's clipboard. This tests
    # the headless guard in `copy_to_clipboard` that prevents it.
    touched = []
    environment = type(platform.current())
    monkeypatch.setattr(environment, "copy", lambda self, text: touched.append(text))
    async with app.run_test() as pilot:
        authored(app, "y")
        await pilot.press("ctrl+c")
        assert app.clipboard == "y"
        assert touched == []


async def test_ctrl_v_pastes_the_system_clipboard_where_it_is_read(app, monkeypatch):
    # Outside tests Ctrl-V reads the system clipboard, falling back on the app's copy
    # when that yields nothing. This simulates a non-headless run with a fake system
    # clipboard.
    held = ["a + b\nsecond line"]
    environment = type(platform.current())
    monkeypatch.setattr(environment, "paste", lambda self: held[0])
    monkeypatch.setattr(RederiveApp, "is_headless", property(lambda self: False))
    monkeypatch.setattr(environment, "copy", lambda self, text, refused=None: None)
    async with app.run_test() as pilot:
        authored(app, "y")
        await pilot.press("ctrl+c")
        await pilot.press("a", "ctrl+v")
        assert prompt(app)[1] == "a + b"
        held[0] = None
        await pilot.press("ctrl+v")
        assert prompt(app)[1] == "a + by"


async def test_ctrl_c_copies_only_the_highlighted_part(app):
    async with app.run_test() as pilot:
        authored(app, "x (x + 1)")
        await pilot.press("right", "right")
        assert highlighted_expression(app) == "x + 1"
        await pilot.press("ctrl+c")
        assert app.clipboard == "x+1"


async def test_the_highlight_can_be_walked_and_copied_under_a_line(app):
    async with app.run_test() as pilot:
        authored(app, "x + 1", "y")
        # The main use: build a new expression from one on screen without leaving
        # the line.
        await pilot.press("a", *"2 (")
        await pilot.press("up")
        assert highlighted_expression(app) == "x + 1"
        await pilot.press("ctrl+c", "ctrl+v", *")", "enter")
        assert numbered(app)[-1] == "#3: 2*(x+1)"


async def test_a_copy_outlives_the_line_it_was_pasted_on(app):
    async with app.run_test() as pilot:
        authored(app, "sin(x)")
        await pilot.press("ctrl+c")
        await pilot.press("a", "ctrl+v", "escape")
        # The clipboard belongs to the app, not the command, so it persists.
        await pilot.press("a", "ctrl+v", "ctrl+v")
        assert prompt(app)[1] == "SIN(x)SIN(x)"


async def test_pasting_takes_the_offered_label_hash_and_all(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y")
        await pilot.press("ctrl+c")
        await pilot.press("s")
        assert prompt(app) == ("SIMPLIFY expression:", "#2")
        # As with F3, the paste replaces the whole offered label, `#` included.
        await pilot.press("ctrl+v")
        assert prompt(app)[1] == "y"


async def test_a_terminal_paste_takes_the_offered_label_too(app):
    # The terminal's paste key, and Ctrl-V in the browser, send a paste event rather
    # than a key. It should behave like Ctrl-V.
    async with app.run_test() as pilot:
        authored(app, "x", "y")
        await pilot.press("s")
        assert prompt(app) == ("SIMPLIFY expression:", "#2")
        line = app.query_one("#prompt-input")
        line.post_message(events.Paste("a + b\nsecond line"))
        await pilot.pause()
        assert prompt(app)[1] == "a + b"


async def test_ctrl_v_pastes_onto_any_line_but_not_at_the_menu(app):
    async with app.run_test() as pilot:
        authored(app, "x")
        await pilot.press("ctrl+c")
        await pilot.press("t", "s", "d")
        assert prompt(app)[0] == "TRANSFER SAVE DERIVE file:"
        # Pasting works on lines that don't take an expression, too.
        await pilot.press("ctrl+v")
        assert prompt(app)[1] == "x"
        await pilot.press("escape", "escape", "escape")
        await pilot.press("ctrl+v")
        assert highlighted_menu_option(app) == "Author"


async def test_ctrl_c_with_nothing_highlighted_copies_nothing(app):
    async with app.run_test() as pilot:
        await pilot.press("ctrl+c")
        assert app.clipboard == ""
        assert message(app) == "Enter option"


async def test_ctrl_v_with_nothing_copied_writes_nothing(app):
    async with app.run_test() as pilot:
        await pilot.press("a", "ctrl+v")
        assert prompt(app) == ("AUTHOR expression:", "")


# -- copying and cutting what is marked out on the line ------------------------
#
# Text selected on the line takes precedence over the highlighted expression: Ctrl-C
# copies the highlighted expression only when nothing is selected. Ctrl-X works only on
# the line's selection, and says so when there is none.


async def test_ctrl_c_copies_what_is_marked_out_on_the_line(app):
    async with app.run_test() as pilot:
        authored(app, "a/b")
        await pilot.press("a")
        await pilot.press(*"2 ", "alt+p")
        await pilot.press("shift+left")
        await pilot.press("ctrl+c")
        assert app.clipboard == "π"
        assert message(app) == COPIED_TEXT
        # Copying leaves the line unchanged.
        assert prompt(app)[1] == "2 π"


async def test_the_label_a_line_offers_is_not_a_selection_of_the_users(app):
    async with app.run_test() as pilot:
        authored(app, "x + 1", "y")
        # Simplify preselects `#2` so typing replaces it. That's not a user selection,
        # so Ctrl-C still copies the highlighted expression.
        await pilot.press("s")
        assert prompt(app) == ("SIMPLIFY expression:", "#2")
        await pilot.press("ctrl+c")
        assert app.clipboard == "y"
        assert message(app) == COPIED
        # Also after walking rewrites the offered label.
        await pilot.press("up")
        assert prompt(app)[1] == "#1"
        await pilot.press("ctrl+c")
        assert app.clipboard == "x+1"


async def test_ctrl_x_cuts_what_is_marked_out_and_keeps_the_copy(app):
    async with app.run_test() as pilot:
        await pilot.press("a")
        await pilot.press(*"12+", "alt+p")
        await pilot.press("shift+left")
        await pilot.press("ctrl+x")
        assert prompt(app)[1] == "12+"
        assert app.clipboard == "π"
        assert message(app) == CUT_TEXT
        # A cut is also a copy, so it can be pasted back.
        await pilot.press("ctrl+v")
        assert prompt(app)[1] == "12+π"


async def test_ctrl_x_with_nothing_marked_out_says_so(app):
    async with app.run_test() as pilot:
        authored(app, "a/b")
        await pilot.press("a")
        await pilot.press(*"x+1")
        await pilot.press("ctrl+x")
        # The line is unchanged, and the highlighted expression is not cut.
        assert prompt(app)[1] == "x+1"
        assert app.clipboard == ""
        assert message(app) == NOTHING_TO_CUT


async def test_ctrl_x_does_not_cut_the_label_a_line_offers(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y")
        await pilot.press("s")
        await pilot.press("ctrl+x")
        assert prompt(app) == ("SIMPLIFY expression:", "#2")
        assert app.clipboard == ""
        assert message(app) == NOTHING_TO_CUT


async def test_ctrl_x_at_the_menu_cuts_nothing_and_says_nothing(app):
    async with app.run_test() as pilot:
        authored(app, "x+1")
        await pilot.press("ctrl+x")
        # There's no line at the menu, so no message either.
        assert app.clipboard == ""
        assert message(app) == "Enter option"


# -- F3 and F4, the original's keys for the same thing -------------------------
#
# They write the highlighted expression onto the line directly. The help offers copy and
# paste instead, but these are kept for users following the manual.


async def test_f3_writes_the_highlighted_expression_onto_the_author_line(app):
    async with app.run_test() as pilot:
        authored(app, "x + 1", "y")
        await pilot.press("a", "f3")
        assert prompt(app) == ("AUTHOR expression:", "y")
        # Walking the highlight under the line picks what F3 takes.
        await pilot.press("up", "f3")
        assert prompt(app)[1] == "yx+1"


async def test_f4_fences_what_it_writes(app):
    async with app.run_test() as pilot:
        authored(app, "x + 1")
        await pilot.press("a", *"2", "f4", "enter")
        assert numbered(app)[-1] == "#2: 2*(x+1)"


async def test_f3_writes_at_the_cursor_and_leaves_it_after(app):
    async with app.run_test() as pilot:
        authored(app, "x + 1")
        await pilot.press("a", *"ab", "left", "f3")
        assert prompt(app)[1] == "ax+1b"
        # The cursor ends up after the inserted text.
        await pilot.press(*"c")
        assert prompt(app)[1] == "ax+1cb"


async def test_f3_takes_only_the_highlighted_part(app):
    async with app.run_test() as pilot:
        authored(app, "x (x + 1)")
        # Descend before opening the line, since with a line up the sideways keys
        # belong to the line.
        await pilot.press("right", "right")
        assert highlighted_expression(app) == "x + 1"
        await pilot.press("a", "f3")
        assert prompt(app)[1] == "x+1"


@pytest.mark.parametrize(
    ("keys", "line"),
    [
        (("a",), "AUTHOR expression:"),
        (("d", "v", *"a", "enter", "v"), "DECLARE VARIABLE value:"),
        (("d", "f", *"g", "enter"), "DECLARE FUNCTION value:"),
        (("d", "r", "2", "enter"), "VECTOR element:"),
    ],
    ids=str,
)
async def test_f3_writes_onto_every_line_an_expression_is_written_on(app, keys, line):
    async with app.run_test() as pilot:
        authored(app, "x + 1")
        await pilot.press(*keys)
        assert prompt(app)[0] == line
        await pilot.press("f3")
        assert prompt(app)[1] == "x+1"


async def test_f3_writes_onto_the_substitute_value_line(app):
    async with app.run_test() as pilot:
        authored(app, "x + 1", "a")
        await pilot.press("m", "s", "enter")
        assert prompt(app)[0] == "MANAGE SUBSTITUTE value:"
        # The manual names F3 for this line, as the substituted value is often an
        # expression already on screen.
        await pilot.press("f3")
        assert prompt(app)[1] == "a"


async def test_the_substitute_value_line_takes_what_the_highlight_walks_onto(app):
    async with app.run_test() as pilot:
        authored(app, "x (x + 1)", "a + b")
        await pilot.press("m", "s", "enter")
        assert prompt(app) == ("MANAGE SUBSTITUTE value:", "a")
        # The target expression is settled; the value is found by walking to it,
        # into it, and taking it.
        await pilot.press("up")
        assert highlighted_expression(app) == "x·(x + 1)"
        await pilot.press("f6", "right", "right")
        assert highlighted_expression(app) == "x + 1"
        await pilot.press("f3")
        assert prompt(app)[1] == "x+1"


async def test_f3_writes_nothing_on_a_line_that_takes_no_expression(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y")
        # Jump takes a label number and Declare a name, not an expression.
        for keys in (("j",), ("d", "v")):
            await pilot.press(*keys)
            await pilot.press("f3", "f4")
            assert prompt(app)[1] == ""
            await pilot.press("escape")


async def test_f3_takes_the_offered_label_hash_and_all(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y")
        await pilot.press("s")
        assert prompt(app) == ("SIMPLIFY expression:", "#2")
        # A typed digit replaces only the number, but F3 writes an expression, so it
        # replaces the `#` too.
        await pilot.press("f3")
        assert prompt(app)[1] == "y"


async def test_f3_on_an_empty_worksheet_writes_nothing(app):
    async with app.run_test() as pilot:
        await pilot.press("a", "f3", "f4")
        assert prompt(app) == ("AUTHOR expression:", "")


async def test_the_alt_keys_write_glyphs_onto_the_line(app):
    async with app.run_test() as pilot:
        await pilot.press("a")
        await pilot.press("alt+a", "alt+p", "alt+0", "alt+q", "alt+e", "alt+i")
        assert prompt(app)[1] == "απ∞√#e#i"


async def test_alt_minus_writes_the_plus_or_minus_operator(app):
    async with app.run_test() as pilot:
        # The key needs a terminal that distinguishes Alt from Escape, which the
        # keyboard protocol Textual requests provides.
        await pilot.press("a", "alt+minus", *"x", "enter")
        # Drawn as the glyph, and stored as `"+-"` (with quotes), as in the original.
        assert work_area(app) == ["#1:  ±x"]
        assert numbered(app)[-1] == '#1: "+-"x'


async def test_a_glyph_written_by_an_alt_key_parses(app):
    async with app.run_test() as pilot:
        await pilot.press("a", "alt+p", "enter")
        assert work_area(app) == ["#1:  π"]
        # The subscript operator is a word here (a glyph in the original), so the
        # key writes it spaced as the printer does.
        await pilot.press("a", *"x", "alt+v", "2", "enter")
        assert numbered(app)[-1] == "#2: x SUB 2"


async def test_the_alt_keys_write_on_any_line_but_not_at_the_menu(app):
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        assert prompt(app)[0] == "TRANSFER LOAD DERIVE file:"
        await pilot.press("alt+p")
        assert prompt(app)[1].endswith("π")
        await pilot.press("escape", "escape", "escape")
        # At the menu, Alt-P is not the P mnemonic: nothing runs.
        await pilot.press("alt+p")
        assert highlighted_menu_option(app) == "Author"
        assert message(app) == "Enter option"


async def test_f6_hands_the_sideways_keys_to_the_highlight_and_back(app):
    async with app.run_test() as pilot:
        authored(app, "x (x + 1)")
        await pilot.press("a")
        assert flags(app) == ["Lin"]
        await pilot.press("f6")
        assert flags(app) == []
        await pilot.press("right", "right")
        assert highlighted_expression(app) == "x + 1"
        await pilot.press("f3")
        assert prompt(app)[1] == "x+1"
        # Back in line-edit mode the keys move the cursor again.
        await pilot.press("f6")
        assert flags(app) == ["Lin"]
        await pilot.press("home", *"2")
        assert prompt(app)[1] == "2x+1"
        assert highlighted_expression(app) == "x + 1"


async def test_the_arrow_key_setting_says_which_mode_a_line_starts_in(app):
    async with app.run_test() as pilot:
        authored(app, "x + 1")
        authored(app, "ArrowKeyMode := Subexpression")
        await pilot.press("a")
        assert flags(app) == []
        # F6 applies to the current line only; the next starts per the setting.
        await pilot.press("f6")
        assert flags(app) == ["Lin"]
        await pilot.press("escape", "a")
        assert flags(app) == []


async def test_no_line_up_means_no_mode_words_to_report(app):
    async with app.run_test() as pilot:
        authored(app, "x")
        assert flags(app) == []
        # Overwrite persists, but with no line up it isn't shown.
        await pilot.press("a", "insert", "escape")
        assert flags(app) == []


async def test_ins_toggles_overwrite(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"abc", "home")
        assert flags(app) == ["Lin"]
        await pilot.press("insert")
        assert flags(app) == ["Ovr", "Lin"]
        await pilot.press(*"XY")
        assert prompt(app)[1] == "XYc"
        # At the end of the line, overwrite appends.
        await pilot.press("end", *"d")
        assert prompt(app)[1] == "XYcd"
        await pilot.press("insert")
        assert flags(app) == ["Lin"]
        await pilot.press("home", *"Z")
        assert prompt(app)[1] == "ZXYcd"
        # In the original Ctrl-V was also Ins; here it pastes and leaves the mode.
        await pilot.press("ctrl+v")
        assert flags(app) == ["Lin"]


async def test_overwrite_stands_until_it_is_turned_back(app):
    async with app.run_test() as pilot:
        await pilot.press("a", "insert", "escape")
        # It's a typing preference, not part of a command, so it persists across
        # lines.
        await pilot.press("a", *"abc", "home", *"X")
        assert flags(app) == ["Ovr", "Lin"]
        assert prompt(app)[1] == "Xbc"


async def test_overwrite_replaces_what_is_selected_as_inserting_does(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y")
        await pilot.press("s", "insert")
        # Typing replaces the whole selected label rather than overwriting its
        # first character.
        assert prompt(app)[1] == "#2"
        await pilot.press("1")
        assert prompt(app)[1] == "#1"


# -- the kill ring: what a deletion took out, and Ctrl-Y putting it back ------


async def test_a_kill_is_put_back_at_the_cursor_and_the_cursor_follows_it(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"a + b")
        # Nothing has been deleted yet, so there is nothing to put back.
        await pilot.press("ctrl+y", "alt+y")
        assert prompt(app)[1] == "a + b"
        # Ctrl-K takes the line from the cursor on, so the whole of it here.
        await pilot.press("home", "ctrl+k", "escape")
        # The kill ring outlives the command, so kills from an abandoned line can
        # still be yanked.
        await pilot.press("a", *"2 (", "ctrl+y", *")")
        assert prompt(app) == ("AUTHOR expression:", "2 (a + b)")
        await pilot.press("enter")
        assert entries(app) == ["2*(a+b)"]


async def test_deletions_that_follow_one_another_come_back_as_one(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"a + b + c")
        # Consecutive kills join into one, as in Emacs and bash.
        await pilot.press("ctrl+w", "ctrl+w")
        assert prompt(app)[1] == "a + "
        await pilot.press("ctrl+w")
        assert prompt(app)[1] == ""
        await pilot.press("ctrl+y")
        assert prompt(app)[1] == "a + b + c"
        # Backspace and Delete aren't kills, so Ctrl-Y still yanks what Ctrl-U took.
        await pilot.press("ctrl+u", *"cde", "backspace", "left", "delete")
        assert prompt(app)[1] == "c"
        await pilot.press("ctrl+y")
        assert prompt(app)[1] == "ca + b + c"


async def test_a_deletion_after_something_else_starts_a_kill_of_its_own(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"a b", "ctrl+w", *"c", "ctrl+w")
        assert prompt(app)[1] == "a "
        # Typing in between makes them separate kills, so Ctrl-Y yanks only the last.
        await pilot.press("ctrl+y")
        assert prompt(app)[1] == "a c"


async def test_alt_y_walks_back_through_the_kills_before_the_last(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"first", "ctrl+u", *"second", "ctrl+u")
        await pilot.press("ctrl+y")
        assert prompt(app)[1] == "second"
        await pilot.press("alt+y")
        assert prompt(app)[1] == "first"
        # Past the oldest it wraps around to the newest.
        await pilot.press("alt+y")
        assert prompt(app)[1] == "second"
        # Once something is typed after the yank, Alt-Y does nothing.
        await pilot.press(*")")
        await pilot.press("alt+y")
        assert prompt(app)[1] == "second)"


async def test_the_jump_line_walks_and_enter_alone_keeps_where_it_went(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y", "z")
        await pilot.press("j")
        await pilot.press("up", "up")
        assert highlighted_expression(app) == "x"
        assert prompt(app)[1] == ""
        await pilot.press("enter")
        assert highlighted_expression(app) == "x"
        assert message(app) == "Enter option"


async def test_the_page_keys_walk_a_paneful_at_a_time(app):
    async with app.run_test(size=(80, 25)) as pilot:
        # Twenty-five rows leave a twenty-row pane, as in the original.
        authored(app, *(str(number) for number in range(1, 31)))
        await pilot.press("j", *"20", "enter")
        await pilot.press("pageup")
        assert selected_number(app) == 11
        await pilot.press("pageup")
        assert selected_number(app) == 2
        await pilot.press("pagedown")
        assert selected_number(app) == 10
        # The original's two key pairs for first and last.
        await pilot.press("ctrl+pageup")
        assert selected_number(app) == 1
        await pilot.press("ctrl+pagedown")
        assert selected_number(app) == 30
        await pilot.press("ctrl+home")
        assert selected_number(app) == 1


async def test_the_page_keys_walk_under_a_prompt_line_too(app):
    async with app.run_test(size=(80, 25)) as pilot:
        authored(app, *(str(number) for number in range(1, 31)))
        await pilot.press("j", *"20", "enter")
        await pilot.press("s")
        assert prompt(app)[1] == "#20"
        await pilot.press("pageup")
        assert selected_number(app) == 11
        assert prompt(app)[1] == "#11"


async def test_the_page_keys_label_a_dialogs_field_too(app):
    async with app.run_test(size=(80, 25)) as pilot:
        authored(app, *(str(number) for number in range(1, 31)))
        await pilot.press("j", *"20", "enter")
        await pilot.press("r")
        assert band(app) == [" REMOVE: Start: 20     End: 20"]
        await pilot.press("pageup")
        assert band(app) == [" REMOVE: Start: 11     End: 20"]


async def test_factors_variable_line_takes_no_notice_of_them(app):
    async with app.run_test() as pilot:
        authored(app, "x", "y", "x^2 - y^2")
        await pilot.press("f", "enter")
        assert prompt(app)[0] == "FACTOR variable 1:"
        # The expression is already chosen, so the highlight stays put.
        for key in ("up", "down", "pageup", "pagedown", "ctrl+home", "ctrl+end"):
            await pilot.press(key)
            assert highlighted_expression(app) == " 2    2\nx  - y"


@pytest.mark.parametrize(
    ("keys", "line"),
    [
        (("c", "d", "enter"), "CALCULUS DIFFERENTIATE variable:"),
        (("d", "m", "2", "tab", "2", "enter"), "MATRIX element:"),
    ],
    ids=str,
)
async def test_the_lines_that_name_no_expression_walk_under_it_too(app, keys, line):
    """A line asking for a name, and a line for entering an element."""
    async with app.run_test() as pilot:
        authored(app, "x", "y", "a z + b")
        await pilot.press(*keys)
        offered = prompt(app)[1]
        assert prompt(app)[0] == line
        await pilot.press("up")
        assert highlighted_expression(app) == "y"
        await pilot.press("ctrl+home")
        assert highlighted_expression(app) == "x"
        # Neither names an expression, so neither takes the label walked onto.
        assert prompt(app) == (line, offered)


async def test_a_line_that_takes_no_expression_is_offered_no_f6(app):
    async with app.run_test() as pilot:
        authored(app, "x (x + 1)", "y")
        await pilot.press("m", "a", "enter")
        assert prompt(app)[0] == "ANNOTATION:"
        # The line takes no expression, so it has no arrow-key mode to show.
        assert flags(app) == []
        await pilot.press("f6")
        assert flags(app) == []


async def test_subexpression_mode_leaves_such_a_line_its_sideways_keys(app):
    async with app.run_test() as pilot:
        authored(app, "ArrowKeyMode := Subexpression")
        authored(app, "x (x + 1)")
        await pilot.press("c", "d", "enter")
        assert prompt(app) == ("CALCULUS DIFFERENTIATE variable:", "x")
        await pilot.press("end", *"y")
        assert prompt(app)[1] == "xy"
        assert highlighted_expression(app) == "x·(x + 1)"


# -- keeping the selected entry in view ---------------------------------------


async def test_a_new_entry_is_scrolled_to_even_when_the_scroll_beats_the_layout(
    app, monkeypatch
):
    # Textual may run an after-refresh callback before the layout that makes room for
    # the new entry, clamping the scroll to the old height. Running callbacks
    # immediately forces that ordering.
    monkeypatch.setattr(
        WorkArea,
        "call_after_refresh",
        lambda self, callback, *args, **kwargs: callback(*args, **kwargs),
    )
    async with app.run_test(size=(80, 24)) as pilot:
        for number in range(15):
            authored(app, f"x+{number}")
        await pilot.pause()
        pane = app.work_area
        assert pane.max_scroll_y > 0
        assert pane.scroll_offset.y == pane.max_scroll_y


async def test_solutions_that_first_overflow_the_pane_are_scrolled_to(app):
    async with app.run_test(size=(100, 40)) as pilot:
        authored(app, "x^12=12")
        await pilot.press("l", "enter")
        for _ in range(20):
            await pilot.pause()
        pane = app.work_area
        assert pane.max_scroll_y > 0
        assert pane.scroll_offset.y == pane.max_scroll_y


#: A render much wider than an 80-column pane, drawn exactly as authored, so the
#: visible columns can be computed by slicing.
WIDE = " + ".join(str(number) for number in range(100, 130))

#: Columns available to a render in an 80-column pane (the label takes five), and the
#: distance of one Ctrl-Right (a third of that).
SHOWABLE = 75
STEP = 25

#: Columns kept visible beyond a highlight that the pane scrolled to follow.
MARGIN = 2


def from_the_right(term):
    """The minimal scroll that brings `term` into view from the right."""
    return WIDE.index(term) + len(term) + MARGIN - SHOWABLE


async def scrolled(pilot, keys):
    """Set up a worksheet with `WIDE` in the middle, select it, and press `keys`."""
    authored(pilot.app, "x+1", WIDE, "x+1")
    # Scrolling needs the pane's width, so let it lay out first.
    await pilot.pause()
    await pilot.press("up")
    await pilot.press(*keys)


async def test_scrolling_sideways_moves_the_selected_entry_alone(app):
    async with app.run_test() as pilot:
        await scrolled(pilot, ["ctrl+right"])
        assert work_area(app) == [
            "#1:  x + 1",
            "",
            "#2:  " + WIDE[STEP:],
            "",
            "#3:  x + 1",
        ]
        # The highlight scrolls too, cut off at the left edge.
        assert highlighted_expression(app) == WIDE[STEP:]


async def test_each_step_sideways_is_a_third_of_what_the_pane_shows(app):
    async with app.run_test() as pilot:
        await scrolled(pilot, ["ctrl+right", "ctrl+right"])
        assert work_area(app)[2] == "#2:  " + WIDE[2 * STEP :]
        await pilot.press("ctrl+left")
        assert work_area(app)[2] == "#2:  " + WIDE[STEP:]


async def test_a_scroll_stops_at_either_end_of_the_render(app):
    async with app.run_test() as pilot:
        await scrolled(pilot, ["ctrl+right"] * 10)
        # Stops at the render's last column, with no blanks past it.
        assert work_area(app)[2] == "#2:  " + WIDE[len(WIDE) - 75 :]
        await pilot.press(*["ctrl+left"] * 10)
        assert work_area(app)[2] == "#2:  " + WIDE


async def test_a_narrow_entry_has_nowhere_to_scroll_to(app):
    async with app.run_test() as pilot:
        await author(pilot, "x+1")
        await pilot.press("ctrl+right")
        assert work_area(app) == ["#1:  x + 1"]


async def test_a_highlight_that_moves_brings_the_render_back_with_it(app):
    async with app.run_test() as pilot:
        await scrolled(pilot, ["ctrl+right"])
        # Down and back up reselects the whole entry, which starts at the left edge,
        # so the scroll resets.
        await pilot.press("down", "up")
        assert work_area(app)[2] == "#2:  " + WIDE
        # So does moving to its first part.
        await pilot.press("ctrl+right")
        assert work_area(app)[2] == "#2:  " + WIDE[STEP:]
        await pilot.press("right")
        assert work_area(app)[2] == "#2:  " + WIDE
        assert highlighted_expression(app) == "100"


async def test_a_highlight_walking_off_the_right_takes_the_render_with_it(app):
    async with app.run_test() as pilot:
        # Twelve terms in, the highlight is still visible unscrolled.
        await scrolled(pilot, ["right"] * 12)
        assert highlighted_expression(app) == "111"
        assert work_area(app)[2] == "#2:  " + WIDE
        # The thirteenth is past the edge, so the render scrolls just enough to show
        # it with the margin.
        await pilot.press("right")
        assert highlighted_expression(app) == "112"
        assert from_the_right("112") == MARGIN
        assert work_area(app)[2] == "#2:  " + WIDE[MARGIN:]
        await pilot.press("right")
        assert work_area(app)[2] == "#2:  " + WIDE[from_the_right("113") :]


async def test_a_highlight_walking_back_left_brings_the_render_back(app):
    async with app.run_test() as pilot:
        await scrolled(pilot, ["right"] * 25)
        assert highlighted_expression(app) == "124"
        assert work_area(app)[2] == "#2:  " + WIDE[from_the_right("124") :]
        # Walking back, the render follows once the highlight reaches the left edge,
        # keeping the same margin.
        await pilot.press(*["left"] * 12)
        assert highlighted_expression(app) == "112"
        assert work_area(app)[2] == "#2:  " + WIDE[WIDE.index("112") - MARGIN :]
        await pilot.press("home")
        assert highlighted_expression(app) == "100"
        assert work_area(app)[2] == "#2:  " + WIDE


async def test_a_highlight_already_on_show_moves_no_render(app):
    async with app.run_test() as pilot:
        # Scrolled by hand with the highlight still visible: the pane stays put
        # rather than snapping to the highlight.
        await scrolled(pilot, ["right"] * 11 + ["ctrl+right"])
        assert highlighted_expression(app) == "110"
        assert work_area(app)[2] == "#2:  " + WIDE[STEP:]
        await pilot.press("right")
        assert highlighted_expression(app) == "111"
        assert work_area(app)[2] == "#2:  " + WIDE[STEP:]


async def test_a_selection_too_wide_to_show_is_shown_from_its_left(app):
    async with app.run_test() as pilot:
        # The entry is wider than the pane, so it's shown from its left end.
        await scrolled(pilot, ["ctrl+right", "down", "up"])
        assert work_area(app)[2] == "#2:  " + WIDE


async def test_a_highlight_scrolled_off_the_left_is_not_painted(app):
    async with app.run_test() as pilot:
        await scrolled(pilot, ["right"])
        assert highlighted_expression(app) == "100"
        await pilot.press("ctrl+right")
        assert work_area(app)[2] == "#2:  " + WIDE[STEP:]
        assert highlighted_expression(app) == ""


# -- and the same scroll spelled without a modifier ----------------------------
#
# `<` and `>` scroll like Ctrl-Left and Ctrl-Right, for window managers that grab those
# keys. Being characters, they scroll only where nothing is waiting for typed input.


async def test_the_angle_brackets_scroll_as_the_ctrl_keys_do(app):
    async with app.run_test() as pilot:
        await scrolled(pilot, [">", ">"])
        assert work_area(app)[2] == "#2:  " + WIDE[2 * STEP :]
        await pilot.press("<")
        assert work_area(app)[2] == "#2:  " + WIDE[STEP:]


async def test_an_angle_bracket_typed_on_a_line_is_text(app):
    async with app.run_test() as pilot:
        await pilot.press("a")
        await pilot.press(*"x<3")
        assert prompt(app) == ("AUTHOR expression:", "x<3")
        await pilot.press("enter")
        assert entries(app)[-1] == "x<3"


async def test_an_angle_bracket_typed_into_a_field_is_text(app):
    async with app.run_test() as pilot:
        authored(app, "x^2*y")
        await pilot.press("c", "l", "enter", "enter")
        await pilot.press("<")
        assert band(app) == [
            " CALCULUS LIMIT: Point: <                  From:(Both)Left Right"
        ]


# -- what a touch screen's keyboard is for -----------------------------------


async def test_only_a_line_or_a_text_field_wants_typing(app):
    async with app.run_test() as pilot:
        assert not app.wants_typing
        await pilot.press("a")
        await pilot.pause()
        assert app.wants_typing
        await pilot.press("x", "enter")
        await pilot.pause()
        assert not app.wants_typing
        # Menus and dialog choices respond to taps; text fields need a keyboard.
        await pilot.press("o", "r")
        assert not app.wants_typing
        await pilot.press("space", "space", "enter")
        assert app.wants_typing
        await pilot.press("escape", "escape", "escape")
        assert not app.wants_typing
