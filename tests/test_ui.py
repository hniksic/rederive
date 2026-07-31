"""Smoke tests driving the real app through Textual's pilot."""

import pytest
from screen import (
    annotation,
    band,
    entries,
    flags,
    highlighted,
    highlighted_expression,
    highlighted_rows,
    message,
    prompt,
    text_of,
    work_area,
)

from rederive import __version__
from rederive.ui import menu as menus
from rederive.ui.app import RederiveApp


def highlighted_menu_option(app):
    return highlighted(app)


@pytest.fixture
def app():
    return RederiveApp()


async def author(pilot, text):
    await pilot.press("a")
    await pilot.press(*text)
    await pilot.press("enter")


# -- the opening notice --------------------------------------------------------

# The original's notice is a painting rather than a state: it survives a menu
# opened and escaped, a dialog committed and a line abandoned, and is gone for
# good the moment anything else draws the work area - an expression, a help
# page, a window command, a Clear. These say the same of this one.


async def test_the_opening_notice_stands_across_the_empty_pane(app):
    async with app.run_test():
        shown = work_area(app)
        said = [line.strip() for line in shown if line]
        assert said == [
            "R E D E R I V E",
            "A Mathematical Assistant",
            f"Version {__version__}",
            "Press H for help",
        ]
        # Centred across the pane, and clear of the top of it and the bottom.
        width = app.query_one("#panes").size.width
        assert all(line == line.strip().center(width).rstrip() for line in shown if line)
        assert shown[0] == "" and shown[-1] == ""
        # The name block stands high and the help line low, with the pane's
        # empty middle between them rather than a single blank row.
        rows = [at for at, line in enumerate(shown) if line]
        assert rows[0] < len(shown) // 3 < rows[-1]
        assert rows[-1] - rows[-2] > 1


async def test_the_opening_notice_gives_the_pane_up_for_the_first_expression(app):
    async with app.run_test() as pilot:
        await author(pilot, "x")
        assert work_area(app) == ["#1:  x"]
        # A worksheet emptied later is an empty worksheet, not a program that
        # has just started, so the notice does not come back.
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
        await pilot.press("p")
        assert message(app) == "Plot: not implemented yet"
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
        await author(pilot, "1+1")
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


async def resized(pilot, app, width, height):
    """Resize the terminal, and wait for the new size to reach the band."""
    await pilot.resize_terminal(width, height)
    for _ in range(20):
        if app.query_one("#menu").size.width == width:
            return
        await pilot.pause()
    raise AssertionError("the resize never reached the band")


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
        # Every row is a row of the band, so none of them is a wrap of the one
        # above it and none has fallen off the bottom.
        assert app.query_one("#menu").size.height == len(lines)
        assert max(len(line) for line in lines) <= width


async def test_a_narrow_terminal_keeps_both_lines_of_a_dialog(app):
    async with app.run_test(size=(60, 24)) as pilot:
        await pilot.press("o", "r")
        await pilot.pause()
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
        # Five rows go to the rule, the message and status lines and two rows
        # of expressions; the menu takes what is left and is cut off there.
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
        await author(pilot, "2+3")
        await pilot.press("s")
        await pilot.press("ctrl+j")
        assert entries(app) == ["2+3", "5"]


async def test_ctrl_enter_enters_nothing_of_its_own_to_simplify(app):
    async with app.run_test() as pilot:
        await author(pilot, "2+3")
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
        # The line is still there to be corrected and entered again.
        await pilot.press("delete")
        await pilot.press("enter")
        assert [entry.text for entry in app.session.entries] == ["12.345"]
        assert message(app) == "Enter option"


async def test_a_built_up_render_is_painted_and_labelled(app):
    async with app.run_test() as pilot:
        await author(pilot, "(x+1)/(x^2+2x+3)")
        # The label sits on the vertically centred row, biased downward, and
        # the render's own rows are indented to the same column.
        assert work_area(app) == [
            "          x + 1",
            "     ──────────────",
            "#1:    2",
            "      x  + 2·x + 3",
        ]


async def test_the_selection_is_a_rectangle_over_the_render(app):
    async with app.run_test() as pilot:
        await author(pilot, "(x+1)/(x^2+2x+3)")
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
        await author(pilot, "x (x + 1)")
        await author(pilot, "x y")
        await author(pilot, "x x x")
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
        await author(pilot, "2 (8 + 7) / 3^2")
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
        await author(pilot, "2 (8 + 7) / 3^2")
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
        await author(pilot, "3 + 4")
        await author(pilot, "5 + 6")
        # The number comes up selected, so typing one replaces it.
        await pilot.press("s")
        await pilot.press("1")
        assert prompt(app) == ("SIMPLIFY expression:", "#1")
        await pilot.press("enter")
        assert work_area(app)[-1] == "#3:  7"
        assert annotation(app) == "Simp(#1)"


async def test_the_annotation_follows_the_selection(app):
    async with app.run_test() as pilot:
        await author(pilot, "2^3")
        await pilot.press("s", "enter")
        assert annotation(app) == "Simp(#1)"
        await pilot.press("up")
        assert highlighted_expression(app) == " 3\n2"
        assert annotation(app) == "User"


async def test_simplify_leaves_a_line_that_does_not_read_up(app):
    async with app.run_test() as pilot:
        await author(pilot, "x")
        await pilot.press("s")
        await pilot.press(*"+")
        await pilot.press("enter")
        assert message(app) == "Syntax error detected at cursor"
        assert [entry.text for entry in app.session.entries] == ["x"]
        await pilot.press("escape")
        assert message(app) == "Enter option"
        assert highlighted_menu_option(app) == "Author"


async def test_simplify_asks_for_nothing_when_the_history_is_empty(app):
    async with app.run_test() as pilot:
        await pilot.press("s", "enter")
        assert app.session.entries == []
        assert message(app) == "Enter option"


async def test_quit_asks_before_abandoning_expressions(app):
    async with app.run_test() as pilot:
        await author(pilot, "x")
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
# Every screen asserted here was checked against the original.
# The order is the original's and not the manual's, which describes the amount
# as being asked for before the variables.


async def test_factor_asks_for_expression_then_variables_then_amount(app):
    async with app.run_test() as pilot:
        await author(pilot, "6x^2 + 10x^3/y")
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


async def test_one_variable_is_no_choice_so_none_is_asked_for(app):
    async with app.run_test() as pilot:
        await author(pilot, "x^2 - 4")
        await pilot.press("f", "enter")
        assert highlighted_menu_option(app) == "Rational"
        await pilot.press("r")
        assert entries(app)[-1] == "(x - 2)*(x + 2)"


async def test_a_number_is_decomposed_without_being_asked_about(app):
    async with app.run_test() as pilot:
        await author(pilot, "1234567890")
        await pilot.press("f", "enter")
        assert entries(app)[-1] == "2*3^2*5*3607*3803"
        assert message(app).startswith("Compute time:")
        assert highlighted_menu_option(app) == "Author"


async def test_the_amount_menu_opens_on_what_was_chosen_last(app):
    async with app.run_test() as pilot:
        await author(pilot, "x^2 - 4")
        await pilot.press("f", "enter")
        assert highlighted_menu_option(app) == "Rational"
        await pilot.press("t")
        await author(pilot, "x^2 - 9")
        await pilot.press("f", "enter")
        assert highlighted_menu_option(app) == "Trivial"


async def test_the_variables_are_collected_one_at_a_time(app):
    async with app.run_test() as pilot:
        await author(pilot, "x^2 y^2 - x^2 - y^4 + y^2")
        await pilot.press("f", "enter")
        assert message(app) == "Return for all or select 1: x,y"
        await pilot.press("x", "enter")
        assert prompt(app) == ("FACTOR variable 2:", "")
        assert message(app) == "Return for no more or select next: y"
        # Ending the list leaves y out, which is what keeps y^2 - 1 whole.
        await pilot.press("enter", "r")
        assert entries(app)[-1] == "(x - y)*(x + y)*(y^2 - 1)"


async def test_choosing_the_last_variable_ends_the_list(app):
    async with app.run_test() as pilot:
        await author(pilot, "x^2 y^2 - x^2 - y^4 + y^2")
        await pilot.press("f", "enter")
        await pilot.press("x", "enter")
        await pilot.press("y", "enter")
        assert message(app) == "Select amount of factoring"
        await pilot.press("r")
        assert entries(app)[-1] == "(x - y)*(x + y)*(y - 1)*(y + 1)"


async def test_a_name_that_is_no_variable_is_refused(app):
    async with app.run_test() as pilot:
        await author(pilot, "x^2 - y^2")
        await pilot.press("f", "enter")
        await pilot.press("q", "enter")
        # The question is put again, unchanged and with nothing chosen.
        assert prompt(app) == ("FACTOR variable 1:", "")
        assert message(app) == "Return for all or select 1: x,y"


async def test_factor_of_a_part_copies_the_rest_of_the_expression(app):
    async with app.run_test() as pilot:
        await author(pilot, "(x^2 - 1) + SIN(z)")
        await pilot.press("right")
        assert highlighted_expression(app) == " 2\nx  - 1"
        # Only x is in the highlighted part, so no variable is asked for.
        await pilot.press("f", "enter", "r")
        assert work_area(app)[-1] == "#2:  (x - 1)·(x + 1) + SIN(z)"
        assert annotation(app) == "Fctr(#1')"


async def test_a_typed_expression_is_factored_as_the_users_own(app):
    async with app.run_test() as pilot:
        await pilot.press("f")
        assert prompt(app) == ("FACTOR expression:", "")
        await pilot.press(*"x^2-9")
        await pilot.press("enter", "r")
        # The line that was typed is not an entry; only the answer is.
        assert entries(app) == ["(x - 3)*(x + 3)"]
        assert annotation(app) == "Fctr(User)"


@pytest.mark.parametrize(
    ("keys", "step"),
    [
        (("f",), "the expression"),
        (("f", "enter"), "the variables"),
        (("f", "enter", "enter"), "the amount"),
    ],
    ids=str,
)
async def test_escape_abandons_factor_from_any_of_its_questions(app, keys, step):
    """One Esc returns to the command menu, whichever question is up."""
    async with app.run_test() as pilot:
        await author(pilot, "x^2 y^2 - 1")
        await pilot.press(*keys)
        await pilot.press("escape")
        assert message(app) == "Enter option"
        assert highlighted_menu_option(app) == "Author"
        assert app.asking is None
        assert entries(app) == ["x^2*y^2-1"]


async def test_factor_leaves_a_line_that_does_not_read_up(app):
    async with app.run_test() as pilot:
        await author(pilot, "x")
        await pilot.press("f")
        await pilot.press(*"+")
        await pilot.press("enter")
        assert message(app) == "Syntax error detected at cursor"
        assert entries(app) == ["x"]


async def test_factor_asks_for_nothing_when_the_history_is_empty(app):
    async with app.run_test() as pilot:
        await pilot.press("f", "enter")
        assert app.session.entries == []
        assert message(app) == "Enter option"


# -- Expand, which asks the same three questions ------------------------------
#
# Every screen asserted here was checked against the original.
# Expand asks for an amount where Factor asks for one - the two are one flow -
# but for a different reason and off a menu of its own: only a ratio has a
# denominator to factor, and Complex is not among the amounts on offer.


async def test_expand_asks_for_the_expression_and_answers(app):
    async with app.run_test() as pilot:
        await author(pilot, "2x(x - 3)^2")
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
        await author(pilot, "(x + 2y + 1)^3")
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
        await author(pilot, "1/(x^2 - 1)")
        await pilot.press("e", "enter")
        assert band(app) == [
            " EXPAND: Amount: Trivial Squarefree Rational raDical"
        ]
        assert message(app) == "Select amount of factoring"
        assert highlighted_menu_option(app) == "Rational"
        await pilot.press("r")
        assert entries(app)[-1] == "1/(2*(x - 1)) - 1/(2*(x + 1))"


async def test_the_two_amount_menus_remember_separately(app):
    """The original keeps a choice made on one off the other: Factor left on
    Trivial still opens Expand on Rational."""
    async with app.run_test() as pilot:
        await author(pilot, "x^2 - 4")
        await pilot.press("f", "enter", "t")
        await author(pilot, "1/(x^2 - 4)")
        await pilot.press("e", "enter")
        assert highlighted_menu_option(app) == "Rational"
        await pilot.press("escape")
        await author(pilot, "x^2 - 9")
        await pilot.press("f", "enter")
        assert highlighted_menu_option(app) == "Trivial"


async def test_the_variables_come_before_the_amount(app):
    async with app.run_test() as pilot:
        await author(pilot, "(x + 1)^2/(a^2 - 1)")
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
        await author(pilot, "2x(x - 3)^2")
        await pilot.press("right", "end")
        assert highlighted_expression(app) == "       2\n(x - 3)"
        await pilot.press("e", "enter")
        assert work_area(app)[-1] == "#2:  2·x·(x  - 6·x + 9)"
        assert annotation(app) == "Expd(#1')"


@pytest.mark.parametrize(
    ("keys", "step"),
    [
        (("e",), "the expression"),
        (("e", "enter"), "the variables"),
        (("e", "enter", "enter"), "the amount"),
    ],
    ids=str,
)
async def test_escape_abandons_expand_from_any_of_its_questions(app, keys, step):
    """One Esc returns to the command menu, whichever question is up."""
    async with app.run_test() as pilot:
        await author(pilot, "1/(x^2 y^2 - 1)")
        await pilot.press(*keys)
        await pilot.press("escape")
        assert message(app) == "Enter option"
        assert highlighted_menu_option(app) == "Author"
        assert app.asking is None
        assert entries(app) == ["1/(x^2*y^2-1)"]


async def test_expand_leaves_a_line_that_does_not_read_up(app):
    async with app.run_test() as pilot:
        await author(pilot, "x")
        await pilot.press("e")
        await pilot.press(*"+")
        await pilot.press("enter")
        assert message(app) == "Syntax error detected at cursor"
        assert entries(app) == ["x"]


async def test_expand_asks_for_nothing_when_the_history_is_empty(app):
    async with app.run_test() as pilot:
        await pilot.press("e", "enter")
        assert app.session.entries == []
        assert message(app) == "Enter option"


# -- soLve, which asks up to three questions and appends any number of answers -
#
# Every screen asserted here was checked against the original. The variable
# line carries no number, the original asking it once per variable it wants
# and never saying which one is up, and the interval comes up only in
# Approximate precision.


async def test_solve_asks_for_the_expression_and_appends_every_solution(app):
    async with app.run_test() as pilot:
        await author(pilot, "x^2 - 5x + 6 = 0")
        await pilot.press("l")
        assert prompt(app) == ("SOLVE expression:", "#1")
        assert message(app) == "Enter expression"
        await pilot.press("enter")
        assert entries(app) == ["x^2-5*x+6=0", "x = 2", "x = 3"]
        assert message(app).startswith("Compute time:")
        assert annotation(app) == "Solve(#1)"
        assert highlighted_menu_option(app) == "Author"


async def test_the_highlight_lands_on_the_last_solution(app):
    async with app.run_test() as pilot:
        await author(pilot, "x^2 - 5x + 6 = 0")
        await pilot.press("l", "enter")
        assert highlighted_expression(app) == "x = 3"


async def test_one_variable_settles_the_question_so_none_is_asked(app):
    async with app.run_test() as pilot:
        await author(pilot, "2x + 3 = 7")
        await pilot.press("l", "enter")
        assert entries(app)[-1] == "x = 2"


async def test_two_variables_bring_the_variable_line_up(app):
    async with app.run_test() as pilot:
        await author(pilot, "a x + b = 0")
        await pilot.press("l", "enter")
        # The most main variable is offered, and typing replaces it.
        assert prompt(app) == ("SOLVE variable:", "x")
        assert message(app) == "Enter variable"
        await pilot.press("enter")
        assert entries(app)[-1] == "x = -b/a"


async def test_the_offered_variable_can_be_typed_over(app):
    async with app.run_test() as pilot:
        await author(pilot, "a x + b = 0")
        await pilot.press("l", "enter")
        await pilot.press("a", "enter")
        assert entries(app)[-1] == "a = -b/x"


async def test_an_underdetermined_system_is_asked_about_once_per_equation(app):
    async with app.run_test() as pilot:
        await author(pilot, "[x + y + z = 1, x - y = 0]")
        await pilot.press("l", "enter")
        assert prompt(app) == ("SOLVE variable:", "x")
        await pilot.press("enter")
        # The default walks the pool, the chosen one having left it.
        assert prompt(app) == ("SOLVE variable:", "y")
        await pilot.press("enter")
        assert entries(app)[-1] == "[x = (1 - z)/2, y = (1 - z)/2]"


async def test_a_square_system_is_asked_nothing(app):
    async with app.run_test() as pilot:
        await author(pilot, "[x + y = 3, x - y = 1]")
        await pilot.press("l", "enter")
        assert entries(app)[-1] == "[x = 2, y = 1]"


async def test_a_name_that_is_no_variable_leaves_the_line_up(app):
    async with app.run_test() as pilot:
        await author(pilot, "a x + b = 0")
        await pilot.press("l", "enter")
        await pilot.press("backspace", *"2", "enter")
        assert prompt(app) == ("SOLVE variable:", "2")
        assert entries(app) == ["a*x+b=0"]


async def test_approximate_precision_asks_for_the_interval(app):
    async with app.run_test() as pilot:
        await author(pilot, "Precision := Approximate")
        await author(pilot, "x^5 - x + 1 = 0")
        await pilot.press("l", "enter")
        assert band(app) == [" SOLVE: Lower: -10                Upper: 10"]
        assert message(app) == "Enter bound on solution"
        await pilot.press("enter")
        assert entries(app)[-1] == "x = -1.16730"


async def test_the_other_two_precisions_never_ask_for_an_interval(app):
    async with app.run_test() as pilot:
        await author(pilot, "Precision := Mixed")
        await author(pilot, "3^x = x^2")
        await pilot.press("l", "enter")
        assert entries(app)[-1] == "x = -0.686026"


async def test_no_solutions_appends_nothing_and_says_so(app):
    async with app.run_test() as pilot:
        await author(pilot, "x^2 - 4")
        await author(pilot, "x = x + 1")
        await pilot.press("l", "enter")
        assert message(app) == "No solutions found"
        assert entries(app) == ["x^2-4", "x=x+1"]
        # Nothing was appended, so nothing moved the highlight off #2.
        assert highlighted_expression(app) == "x = x + 1"


async def test_a_typed_expression_is_solved_as_the_users_own(app):
    async with app.run_test() as pilot:
        await pilot.press("l")
        assert prompt(app) == ("SOLVE expression:", "")
        await pilot.press(*"2x=8")
        await pilot.press("enter")
        assert entries(app) == ["x = 4"]
        assert annotation(app) == "Solve(User)"


@pytest.mark.parametrize(
    ("keys", "step"),
    [
        (("l",), "the expression"),
        (("l", "enter"), "the variable"),
        (("l", "enter", "enter"), "the second variable"),
    ],
    ids=str,
)
async def test_escape_abandons_solve_from_any_of_its_questions(app, keys, step):
    """One Esc returns to the command menu, whichever question is up, and the
    worksheet is untouched."""
    async with app.run_test() as pilot:
        await author(pilot, "[x + y + z = 1, x - y = 0]")
        await pilot.press(*keys)
        await pilot.press("escape")
        assert message(app) == "Enter option"
        assert highlighted_menu_option(app) == "Author"
        assert app.solving is None
        assert entries(app) == ["[x+y+z=1,x-y=0]"]


async def test_escape_abandons_solve_from_the_interval(app):
    async with app.run_test() as pilot:
        await author(pilot, "Precision := Approximate")
        await author(pilot, "x^5 - x + 1 = 0")
        await pilot.press("l", "enter")
        await pilot.press("escape")
        assert message(app) == "Enter option"
        assert app.solving is None
        assert entries(app)[-1] == "x^5-x+1=0"


async def test_solve_leaves_a_line_that_does_not_read_up(app):
    async with app.run_test() as pilot:
        await author(pilot, "x")
        await pilot.press("l")
        await pilot.press(*"+")
        await pilot.press("enter")
        assert message(app) == "Syntax error detected at cursor"
        assert entries(app) == ["x"]


async def test_solve_asks_for_nothing_when_the_history_is_empty(app):
    async with app.run_test() as pilot:
        await pilot.press("l", "enter")
        assert app.session.entries == []
        assert message(app) == "Enter option"


async def test_solving_twice_appends_the_answer_twice(app):
    async with app.run_test() as pilot:
        await author(pilot, "2x = 8")
        await pilot.press("l", "enter")
        await pilot.press("ctrl+home")
        await pilot.press("l", "enter")
        assert entries(app) == ["2*x=8", "x = 4", "x = 4"]


async def test_the_arbitrary_counter_runs_across_the_session(app):
    async with app.run_test() as pilot:
        await author(pilot, "x = x")
        await pilot.press("l", "enter")
        assert entries(app)[-1] == "x = @1"
        await author(pilot, "2y = y + y")
        await pilot.press("l", "enter")
        assert entries(app)[-1] == "y = @2"


# -- approX, which asks the one question Simplify asks ------------------------
#
# Every screen asserted here was checked against the original. The command is
# Simplify at approximate precision, and its line says so and no more: there
# is no question about digits, which come from Options Precision.


async def test_approx_offers_the_highlighted_expression(app):
    async with app.run_test() as pilot:
        await author(pilot, "pi")
        await pilot.press("x")
        assert prompt(app) == ("APPROX expression:", "#1")
        assert message(app) == "Enter expression"
        await pilot.press("enter")
        assert work_area(app) == ["#1:  π", "", "#2:  3.14159"]
        assert highlighted_expression(app) == "3.14159"
        assert message(app).startswith("Compute time:")
        assert annotation(app) == "Approx(#1)"
        assert highlighted_menu_option(app) == "Author"


async def test_approx_of_a_part_copies_the_rest_of_the_expression(app):
    async with app.run_test() as pilot:
        await author(pilot, "SQRT(2) + x^2")
        await pilot.press("right")
        assert highlighted_expression(app) == "√2"
        await pilot.press("x", "enter")
        assert work_area(app)[-2:] == ["                2", "#2:  1.41421 + x"]
        # The quote says that only part of the expression was approximated.
        assert annotation(app) == "Approx(#1')"


async def test_a_typed_expression_is_approximated_as_the_users_own(app):
    async with app.run_test() as pilot:
        await pilot.press("x")
        assert prompt(app) == ("APPROX expression:", "")
        await pilot.press(*"SQRT(3)")
        await pilot.press("enter")
        assert work_area(app) == ["#1:  1.73205"]
        assert annotation(app) == "Approx(User)"


async def test_approx_leaves_the_precision_setting_alone(app):
    """The mode is approximate for the one command, and Simplify says so."""
    async with app.run_test() as pilot:
        await author(pilot, "1/3")
        await pilot.press("x", "enter")
        assert entries(app)[-1] == "0.333333"
        await pilot.press("s")
        assert prompt(app) == ("SIMPLIFY expression:", "#2")
        await pilot.press(*"1")
        await pilot.press("enter")
        assert entries(app)[-1] == "1/3"


async def test_approx_leaves_a_line_that_does_not_read_up(app):
    async with app.run_test() as pilot:
        await author(pilot, "x")
        await pilot.press("x")
        await pilot.press(*"+")
        await pilot.press("enter")
        assert message(app) == "Syntax error detected at cursor"
        assert entries(app) == ["x"]


async def test_approx_asks_for_nothing_when_the_history_is_empty(app):
    async with app.run_test() as pilot:
        await pilot.press("x", "enter")
        assert app.session.entries == []
        assert message(app) == "Enter option"


async def test_an_abandoned_approx_leaves_the_history_alone(app):
    async with app.run_test() as pilot:
        await author(pilot, "pi")
        await pilot.press("x", "escape")
        assert entries(app) == ["pi"]
        assert message(app) == "Enter option"
        assert highlighted_menu_option(app) == "Author"


# -- Remove and Unremove ------------------------------------------------------
#
# Every screen asserted here was checked against the original.


def numbered(app):
    """The history as text: each entry's label, and the notation it holds.

    Not what the work area draws, which is built up from the same tree and
    written in glyphs; `work_area` is what asks for that.
    """
    return [f"#{entry.number}: {entry.text}" for entry in app.session.entries]


def selected_number(app):
    """The label of the expression the highlight is on."""
    return app.session.selected_entry.number


async def worksheet(pilot, *texts):
    for text in texts:
        await author(pilot, text)


async def test_remove_offers_the_highlighted_expression_at_both_ends(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y", "z", "w", "v")
        await pilot.press("r")
        assert band(app) == [" REMOVE: Start: 5      End: 5"]
        assert message(app) == "Enter label number"
        await pilot.press("2", "tab", "4", "enter")
        assert numbered(app) == ["#1: x", "#5: v"]
        assert message(app) == "Enter option"
        assert highlighted_menu_option(app) == "Author"


async def test_enter_alone_removes_the_highlighted_expression(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y", "z")
        await pilot.press("r", "enter")
        assert numbered(app) == ["#1: x", "#2: y"]


async def test_the_arrows_walk_the_history_and_label_the_field(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y", "z", "w")
        await pilot.press("r", "tab")
        await pilot.press("up")
        # The highlight moved with the key, and the End field took it.
        assert band(app) == [" REMOVE: Start: 4      End: 3"]
        assert highlighted_expression(app) == "z"
        await pilot.press("up")
        assert band(app) == [" REMOVE: Start: 4      End: 2"]
        await pilot.press("enter")
        assert numbered(app) == ["#1: x"]


async def test_a_label_that_names_no_expression_leaves_the_question_up(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y", "z")
        await pilot.press("r", "9", "tab", "9", "enter")
        assert band(app) == [" REMOVE: Start: 9      End: 9"]
        assert message(app) == "Enter label number"
        assert numbered(app) == ["#1: x", "#2: y", "#3: z"]


async def test_escape_abandons_remove_and_leaves_the_history_alone(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y")
        await pilot.press("r", "1", "tab", "2")
        await pilot.press("escape")
        assert numbered(app) == ["#1: x", "#2: y"]
        assert message(app) == "Enter option"
        assert highlighted_menu_option(app) == "Author"


async def test_remove_asks_nothing_when_the_history_is_empty(app):
    async with app.run_test() as pilot:
        await pilot.press("r")
        assert band(app)[0].startswith(" COMMAND:")
        assert message(app) == "Enter option"


async def test_unremove_asks_where_and_puts_them_back(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y", "z", "w", "v")
        await pilot.press("r", "2", "tab", "4", "enter")
        await pilot.press("u")
        assert band(app) == [" UNREMOVE: Before: 5"]
        assert message(app) == 'Enter label number or type "end"'
        await pilot.press("enter")
        assert numbered(app) == ["#1: x", "#2: y", "#3: z", "#4: w", "#5: v"]
        assert highlighted_expression(app) == "v"


async def test_end_puts_them_after_the_last_expression(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y", "z")
        await pilot.press("r", "1", "tab", "2", "enter")
        await pilot.press("u")
        await pilot.press(*"END", "enter")
        assert numbered(app) == ["#3: z", "#1: x", "#2: y"]
        assert highlighted_expression(app) == "y"


async def test_unremove_says_so_when_there_is_nothing_to_put_back(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x")
        await pilot.press("u")
        assert message(app) == "Unremove buffer empty"
        assert band(app)[0].startswith(" COMMAND:")


async def test_an_engine_command_empties_the_buffer(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "2 + 3", "y")
        await pilot.press("r", "2", "tab", "2", "enter")
        await pilot.press("s", "enter")
        await pilot.press("u")
        assert message(app) == "Unremove buffer empty"


async def test_unremove_asks_nothing_when_the_history_is_empty(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y")
        await pilot.press("r", "1", "tab", "2", "enter")
        assert numbered(app) == []
        await pilot.press("u")
        assert numbered(app) == ["#1: x", "#2: y"]
        assert message(app) == "Enter option"


async def test_the_buffer_outlives_the_unremove_that_used_it(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y", "z")
        await pilot.press("r", "1", "tab", "2", "enter")
        await pilot.press("u", "3", "enter")
        assert numbered(app) == ["#1: x", "#2: y", "#3: z"]
        # The same block again, under labels that are free this time.
        await pilot.press("u", "3", "enter")
        assert numbered(app) == ["#1: x", "#2: y", "#4: x", "#5: y", "#3: z"]


async def test_the_history_keys_all_label_the_field(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y", "z", "w")
        await pilot.press("r")
        await pilot.press("ctrl+home")
        assert band(app) == [" REMOVE: Start: 1      End: 4"]
        assert highlighted_expression(app) == "x"
        await pilot.press("down")
        assert band(app) == [" REMOVE: Start: 2      End: 4"]
        await pilot.press("ctrl+end")
        assert band(app) == [" REMOVE: Start: 4      End: 4"]


# -- moVe ---------------------------------------------------------------------
#
# Every screen asserted here matches the original.


async def test_move_offers_the_highlighted_expression_in_all_three_fields(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y", "z", "w", "v")
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
        await worksheet(pilot, "x", "y")
        await pilot.press("v", "tab")
        assert message(app) == "Enter label number"
        await pilot.press("tab")
        assert message(app) == "Enter label number"
        await pilot.press("tab")
        assert message(app) == 'Enter label number or type "end"'


async def test_end_sends_the_block_past_the_last_expression(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y", "z")
        await pilot.press("v")
        await pilot.press(*"END", "tab", "1", "tab", "2", "enter")
        assert numbered(app) == ["#3: z", "#1: x", "#2: y"]
        assert highlighted_expression(app) == "y"


async def test_enter_alone_moves_the_highlighted_expression_nowhere(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y", "z")
        await pilot.press("v", "enter")
        assert numbered(app) == ["#1: x", "#2: y", "#3: z"]
        assert highlighted_expression(app) == "z"


async def test_the_arrows_walk_the_history_and_label_the_move_field(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y", "z", "w")
        await pilot.press("v", "up", "up", "up")
        # The highlight moved with the key, and the Before field took it.
        assert band(app) == [" MOVE: Before: 1      Start: 4      End: 4"]
        assert highlighted_expression(app) == "x"
        await pilot.press("enter")
        assert numbered(app) == ["#4: w", "#1: x", "#2: y", "#3: z"]


async def test_a_label_that_names_no_expression_leaves_the_move_question_up(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y")
        await pilot.press("v", "9", "tab", "1", "tab", "1", "enter")
        assert band(app) == [" MOVE: Before: 9      Start: 1      End: 1"]
        # The refusal takes the highlight back to the field that was wrong,
        # so the correction is typed where the mistake is.
        assert message(app) == 'Enter label number or type "end"'
        await pilot.press("2")
        assert band(app) == [" MOVE: Before: 2      Start: 1      End: 1"]
        assert numbered(app) == ["#1: x", "#2: y"]


async def test_the_field_a_move_was_refused_over_is_the_one_it_goes_back_to(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y", "z")
        await pilot.press("v", "1", "tab", "9", "tab", "3", "enter")
        assert message(app) == "Enter label number"
        await pilot.press("3")
        assert band(app) == [" MOVE: Before: 1      Start: 3      End: 3"]
        await pilot.press("enter")
        assert numbered(app) == ["#3: z", "#1: x", "#2: y"]


async def test_escape_abandons_move_and_leaves_the_history_alone(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y")
        await pilot.press("v", "1", "tab", "2", "tab", "2")
        await pilot.press("escape")
        assert numbered(app) == ["#1: x", "#2: y"]
        assert message(app) == "Enter option"
        assert highlighted_menu_option(app) == "Author"


async def test_move_asks_nothing_when_the_history_is_empty(app):
    async with app.run_test() as pilot:
        await pilot.press("v")
        assert band(app)[0].startswith(" COMMAND:")
        assert message(app) == "Enter option"


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
        (("d", "v"), "the name"),
        (("d", "v", "x", "enter"), "the domain"),
        (("d", "v", "x", "enter", "r"), "the interval"),
        (("d", "v", "x", "enter", "r", "i"), "the bounds"),
        (("d", "v", "x", "enter", "v"), "the value"),
        (("d", "f"), "the function name"),
        (("d", "f", "f", "enter"), "the definition"),
        (("d", "f", "f", "enter", "enter"), "the function variables"),
        (("d", "m"), "the shape"),
        (("d", "m", "enter"), "a matrix element"),
        (("d", "r"), "the dimension"),
        (("d", "r", "2", "enter"), "a vector element"),
    ],
    ids=str,
)
async def test_escape_abandons_declare_from_any_of_its_questions(app, keys, step):
    """One Esc returns to the Declare menu, whichever question is up."""
    async with app.run_test() as pilot:
        await pilot.press(*keys)
        await pilot.press("escape")
        assert band(app) == [" DECLARE: Function Variable Matrix vectoR"]
        assert message(app) == "Enter option"
        assert entries(app) == []
        assert (app.declaring, app.defining, app.entering) == (None, None, None)


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
# Every screen asserted here was checked against the original, whose versions
# draw all of them the same way.


async def test_build_asks_for_an_operand_then_an_operator_then_another(app):
    async with app.run_test() as pilot:
        await author(pilot, "x^2*y")
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
        # An operand answered leaves the menu open on the word that finishes.
        assert highlighted(app) == "Done"
        await pilot.press("d")
        assert entries(app) == ["x^2*y", "x^2*y+x^2*y"]
        assert annotation(app) == "#1+#1"
        assert band(app)[0].startswith(" COMMAND:")


async def test_a_unary_operator_asks_for_nothing_more(app):
    async with app.run_test() as pilot:
        await author(pilot, "x")
        await pilot.press("b", "enter")
        await pilot.press("s")
        assert band(app)[0].startswith(" BUILD: Operator:")
        assert highlighted(app) == "Done"
        await pilot.press("d")
        assert entries(app) == ["x", "SIN(x)"]
        assert annotation(app) == "SIN(#1)"


async def test_the_operators_chain_until_done(app):
    async with app.run_test() as pilot:
        await author(pilot, "2")
        await author(pilot, "3")
        await author(pilot, "4")
        await pilot.press("b", "up", "up", "enter")
        await pilot.press("+", "down", "enter")
        await pilot.press("*", "down", "enter")
        await pilot.press("d")
        # Left to right, with the fences the grouping calls for - and an
        # annotation written flat, which is how the original writes it.
        assert entries(app)[-1] == "(2+3)*4"
        assert annotation(app) == "#1+#2*#3"


async def test_unary_operators_nest(app):
    async with app.run_test() as pilot:
        await author(pilot, "x")
        await pilot.press("b", "enter", "s", "c", "d")
        assert entries(app)[-1] == "COS(SIN(x))"
        assert annotation(app) == "COS(SIN(#1))"


async def test_the_symbols_are_their_own_mnemonics(app):
    async with app.run_test() as pilot:
        await author(pilot, "x")
        await pilot.press("b", "enter", "!")
        assert highlighted(app) == "Done"
        await pilot.press("d")
        assert entries(app)[-1] == "x!"


async def test_space_steps_the_operator_menu(app):
    async with app.run_test() as pilot:
        await author(pilot, "x")
        await pilot.press("b", "enter")
        await pilot.press("space", "space")
        assert highlighted(app) == "*"


async def test_the_arrow_keys_pick_the_operand_off_the_screen(app):
    async with app.run_test() as pilot:
        await author(pilot, "2")
        await author(pilot, "3")
        await pilot.press("b")
        assert prompt(app) == ("BUILD first expression:", "#2")
        await pilot.press("up")
        assert prompt(app) == ("BUILD first expression:", "#1")


async def test_a_highlighted_part_is_what_is_built_from(app):
    async with app.run_test() as pilot:
        await author(pilot, "SIN(a*x^2) + 5")
        await pilot.press("f6", "right", "down")
        assert highlighted_expression(app) == "   2\na·x"
        await pilot.press("b")
        # The line still names the entry; the part is what it stands for.
        assert prompt(app) == ("BUILD first expression:", "#1")
        await pilot.press("enter", "d")
        assert entries(app)[-1] == "a*x^2"
        assert annotation(app) == "#1'"


async def test_build_takes_a_typed_expression_too(app):
    async with app.run_test() as pilot:
        await author(pilot, "x")
        await pilot.press("b")
        await pilot.press("backspace", "backspace")
        await pilot.press(*"2+3", "enter")
        await pilot.press("d")
        assert entries(app)[-1] == "2+3"
        assert annotation(app) == "User"


async def test_a_line_that_does_not_read_stays_up(app):
    async with app.run_test() as pilot:
        await author(pilot, "x")
        await pilot.press("b")
        await pilot.press("backspace", "backspace")
        await pilot.press(*"2+*", "enter")
        assert message(app) == "Syntax error detected at cursor"
        assert prompt(app) == ("BUILD first expression:", "2+*")


@pytest.mark.parametrize(
    ("keys", "step"),
    [
        (("b",), "the first operand"),
        (("b", "enter"), "the operator"),
        (("b", "enter", "+"), "the second operand"),
        (("b", "enter", "+", "enter"), "the operator again"),
    ],
    ids=str,
)
async def test_escape_abandons_the_whole_build(app, keys, step):
    """There is nothing to step back to: one Esc gives the command menu back."""
    async with app.run_test() as pilot:
        await author(pilot, "x")
        await pilot.press(*keys)
        await pilot.press("escape")
        assert band(app)[0].startswith(" COMMAND:")
        assert entries(app) == ["x"]
        assert app.building is None


async def test_a_blank_operand_line_abandons_the_build(app):
    async with app.run_test() as pilot:
        await author(pilot, "x")
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
        await author(pilot, "x^2*y")
        await pilot.press("b", "enter", "+", "enter")
        await pilot.press("ctrl+j")
        # One entry, not two: what was built never reaches the history, and
        # the annotation says both what it was and that it was taken.
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
        await author(pilot, "x^2*y")
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


@pytest.mark.parametrize(
    ("key", "line"),
    [
        ("i", " CALCULUS INTEGRATE: Lower limit:                    Upper limit:"),
        ("l", " CALCULUS LIMIT: Point: 0                  From:(Both)Left Right"),
        ("p", " CALCULUS PRODUCT: Lower limit: 1                  Upper limit: n"),
        ("s", " CALCULUS SUM: Lower limit: 1                  Upper limit: n"),
        ("t", " CALCULUS TAYLOR: Degree: 5                  Point: 0"),
        ("v", " CALCULUS VECTOR: Start: 1               End:                 Step: 1"),
    ],
    ids=str,
)
async def test_each_command_finishes_on_a_line_of_its_own(app, key, line):
    async with app.run_test() as pilot:
        await author(pilot, "x^2*y")
        await pilot.press("c", key, "enter", "enter")
        assert band(app) == [line]


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
        await author(pilot, "x^2*y")
        await pilot.press("c", *keys[:1], "enter", "enter", *keys[1:])
        assert entries(app)[-1] == text


async def test_the_limit_direction_is_chosen_off_the_line(app):
    async with app.run_test() as pilot:
        await author(pilot, "x^2*y")
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
        await author(pilot, "x^2*y")
        await pilot.press("c", "t", "enter", "enter")
        assert message(app) == "Enter maximum degree"
        await pilot.press("tab")
        assert message(app) == "Enter expansion point"


async def test_the_vector_line_opens_on_the_field_it_cannot_do_without(app):
    async with app.run_test() as pilot:
        await author(pilot, "x^2*y")
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
        await author(pilot, "x^2*y")
        await pilot.press("c", *keys[:1], "enter", "enter", *keys[1:])
        assert entries(app) == ["x^2*y"]
        assert band(app)[0].startswith(" CALCULUS ")


async def test_one_limit_of_a_pair_sends_the_line_to_the_other(app):
    async with app.run_test() as pilot:
        await author(pilot, "x^2*y")
        await pilot.press("c", "i", "enter", "enter")
        await pilot.press("0", "enter")
        # Neither taken nor refused outright: the empty half is what is asked
        # for now, and answering it commits.
        assert entries(app) == ["x^2*y"]
        assert band(app) == [
            " CALCULUS INTEGRATE: Lower limit: 0                  Upper limit:"
        ]
        await pilot.press("1", "enter")
        assert entries(app)[-1] == "INT(x^2*y,x,0,1)"


async def test_the_variable_offered_is_the_primary_one(app):
    async with app.run_test() as pilot:
        await author(pilot, "SIN(a*x^2)")
        await pilot.press("c", "d", "enter")
        assert prompt(app) == ("CALCULUS DIFFERENTIATE variable:", "x")


async def test_a_bound_variable_leaves_the_line_empty(app):
    async with app.run_test() as pilot:
        await author(pilot, "INT(x^2, x, 0, 1)")
        await pilot.press("c", "d", "enter")
        assert prompt(app) == ("CALCULUS DIFFERENTIATE variable:", "")


async def test_a_line_that_names_no_variable_is_refused_without_a_word(app):
    async with app.run_test() as pilot:
        await author(pilot, "x^2*y")
        await pilot.press("c", "d", "enter")
        await pilot.press("backspace")
        await pilot.press(*"2+3", "enter")
        assert prompt(app) == ("CALCULUS DIFFERENTIATE variable:", "2+3")
        assert message(app) == "Enter variable"
        assert entries(app) == ["x^2*y"]


async def test_a_highlighted_part_is_what_is_taken(app):
    async with app.run_test() as pilot:
        await author(pilot, "SIN(a*x^2) + 5")
        await pilot.press("f6", "right", "down")
        assert highlighted_expression(app) == "   2\na·x"
        await pilot.press("c", "d", "enter", "enter", "enter")
        # Extraction, not substitution: the part alone is what the head goes
        # round, and the entry it came out of is left as it was.
        assert entries(app)[-1] == "DIF(a*x^2,x)"
        assert annotation(app) == "Dif(#1',x)"


@pytest.mark.parametrize(
    ("keys", "step"),
    [
        (("c", "d"), "the expression"),
        (("c", "d", "enter"), "the variable"),
        (("c", "d", "enter", "enter"), "the order"),
        (("c", "v", "enter", "enter"), "the values a vector runs over"),
    ],
    ids=str,
)
async def test_escape_abandons_the_command_for_the_calculus_menu(app, keys, step):
    async with app.run_test() as pilot:
        await author(pilot, "x^2*y")
        await pilot.press(*keys)
        await pilot.press("escape")
        assert band(app) == [
            " CALCULUS: Differentiate Integrate Limit Product Sum Taylor Vector"
        ]
        assert entries(app) == ["x^2*y"]
        assert app.calculating is None


async def test_a_blank_expression_line_abandons_the_command(app):
    async with app.run_test() as pilot:
        await author(pilot, "x^2*y")
        await pilot.press("c", "d")
        await pilot.press("backspace", "backspace", "enter")
        assert band(app) == [
            " CALCULUS: Differentiate Integrate Limit Product Sum Taylor Vector"
        ]
        assert entries(app) == ["x^2*y"]


async def test_ctrl_enter_on_the_last_field_enters_one_expression(app):
    async with app.run_test() as pilot:
        await author(pilot, "x^2*y")
        await pilot.press("c", "d", "enter", "enter")
        await pilot.press("ctrl+j")
        assert entries(app) == ["x^2*y", "2*x*y"]
        assert annotation(app) == "Simp(Dif(#1,x))"


async def test_nothing_is_computed_until_a_simplify_asks_for_it(app):
    async with app.run_test() as pilot:
        await author(pilot, "x^2*y")
        await pilot.press("c", "d", "enter", "enter", "enter")
        assert entries(app)[-1] == "DIF(x^2*y,x)"
        await pilot.press("s", "enter")
        assert entries(app)[-1] == "2*x*y"
        assert annotation(app) == "Simp(#2)"

# -- Jump ---------------------------------------------------------------------
#
# Every screen asserted here was checked against the original.


async def test_jump_asks_for_a_label_and_highlights_what_it_names(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y", "z")
        await pilot.press("j")
        # Nothing is offered: the command is for going somewhere else, so the
        # label of where the highlight already is would be no use.
        assert prompt(app) == ("JUMP to:", "")
        assert message(app) == "Enter label number"
        await pilot.press("1", "enter")
        assert highlighted_expression(app) == "x"
        assert message(app) == "Enter option"
        assert highlighted_menu_option(app) == "Author"


async def test_a_label_no_expression_carries_lands_on_the_one_above_it(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y", "z")
        await pilot.press("r", "2", "tab", "2", "enter")
        await pilot.press("j", "2", "enter")
        assert highlighted_expression(app) == "z"


async def test_a_label_past_the_last_one_leaves_the_line_up(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y")
        await pilot.press("j", "9", "enter")
        assert prompt(app) == ("JUMP to:", "9")
        assert message(app) == "Enter label number"
        assert highlighted_expression(app) == "y"
        # The line is still there to be corrected and entered again.
        await pilot.press("backspace", "1", "enter")
        assert highlighted_expression(app) == "x"


async def test_a_line_that_is_no_label_number_is_refused(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y")
        await pilot.press("j")
        await pilot.press(*"1x", "enter")
        assert prompt(app) == ("JUMP to:", "1x")
        assert highlighted_expression(app) == "y"


async def test_a_jump_line_with_nothing_on_it_leaves_the_highlight_alone(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y")
        await pilot.press("j", "enter")
        assert highlighted_expression(app) == "y"
        assert message(app) == "Enter option"
        assert highlighted_menu_option(app) == "Author"


async def test_escape_abandons_the_jump_line(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y")
        await pilot.press("j", "1")
        await pilot.press("escape")
        assert highlighted_expression(app) == "y"
        assert message(app) == "Enter option"


async def test_jumping_to_the_expression_you_are_inside_of_keeps_the_part(app):
    async with app.run_test() as pilot:
        await author(pilot, "x (x + 1)")
        await pilot.press("right", "right")
        assert highlighted_expression(app) == "x + 1"
        await pilot.press("j", "1", "enter")
        assert highlighted_expression(app) == "x + 1"


async def test_jump_asks_nothing_when_the_history_is_empty(app):
    async with app.run_test() as pilot:
        await pilot.press("j")
        assert band(app)[0].startswith(" COMMAND:")
        assert message(app) == "Enter option"


# -- walking the history with a question up -----------------------------------
#
# The keys that move the highlight from one expression to another go on doing
# it while a prompt line has the screen; the ones that move a cursor along the
# line belong to the line. Every screen asserted here was checked against the
# original.


async def test_the_history_keys_walk_while_a_prompt_line_is_up(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "2 + 3", "y", "z")
        await pilot.press("s")
        assert prompt(app) == ("SIMPLIFY expression:", "#3")
        # The line takes the label of wherever the highlight lands, which is
        # what the manual recommends over typing the number.
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
        await worksheet(pilot, "x", "y", "z")
        await pilot.press("s", "up")
        assert prompt(app)[1] == "#2"
        # Typing replaces it, exactly as it replaces the label first offered.
        await pilot.press("3")
        assert prompt(app)[1] == "#3"


async def test_a_line_that_was_typed_on_is_left_as_it_is(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y", "z")
        await pilot.press("s")
        await pilot.press(*"2+3")
        assert prompt(app)[1] == "#2+3"
        await pilot.press("up")
        # The highlight moved; the line is the user's now, so it stands.
        assert highlighted_expression(app) == "y"
        assert prompt(app)[1] == "#2+3"


async def test_a_cursor_key_hands_the_line_over_for_good(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y", "z")
        await pilot.press("s")
        # End collapses the selection the offered label came up with, and the
        # line stops taking labels from then on.
        await pilot.press("end", "up")
        assert highlighted_expression(app) == "y"
        assert prompt(app)[1] == "#3"


async def test_the_cursor_keys_move_no_highlight_while_a_line_is_up(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y", "x (x + 1)")
        await pilot.press("s")
        for key in ("left", "right", "home", "end"):
            await pilot.press(key)
            assert highlighted_expression(app) == "x·(x + 1)"


async def test_the_author_line_walks_but_is_never_labelled(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y", "z")
        await pilot.press("a")
        await pilot.press(*"w")
        await pilot.press("up")
        assert highlighted_expression(app) == "y"
        assert prompt(app) == ("AUTHOR expression:", "w")


# -- copying an expression and pasting it -------------------------------------
#
# Ctrl-C takes what is highlighted and Ctrl-V puts it back on the line being
# typed. The copy goes to the terminal's clipboard as well as to the program's
# own; what the terminal does with it is nothing a test can see, so these check
# the copy that is kept here.


async def test_ctrl_c_copies_the_highlighted_expression_and_ctrl_v_pastes_it(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x + 1", "y")
        await pilot.press("ctrl+c")
        assert app.clipboard == "y"
        # Text handed to a clipboard leaves no other mark, so the message line
        # is what says the key landed.
        assert message(app) == "Copied the highlighted expression"
        await pilot.press("a", "ctrl+v")
        assert prompt(app) == ("AUTHOR expression:", "y")


async def test_ctrl_c_copies_only_the_highlighted_part(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x (x + 1)")
        await pilot.press("right", "right")
        assert highlighted_expression(app) == "x + 1"
        await pilot.press("ctrl+c")
        assert app.clipboard == "x+1"


async def test_the_highlight_can_be_walked_and_copied_under_a_line(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x + 1", "y")
        # Which is the point of the pair: build a new expression out of one
        # already on screen without leaving the line it is being built on.
        await pilot.press("a", *"2 (")
        await pilot.press("up")
        assert highlighted_expression(app) == "x + 1"
        await pilot.press("ctrl+c", "ctrl+v", *")", "enter")
        assert numbered(app)[-1] == "#3: 2*(x+1)"


async def test_a_copy_outlives_the_line_it_was_pasted_on(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "sin(x)")
        await pilot.press("ctrl+c")
        await pilot.press("a", "ctrl+v", "escape")
        # The clipboard belongs to the program rather than to the command that
        # read the line, so it is still there for the next one.
        await pilot.press("a", "ctrl+v", "ctrl+v")
        assert prompt(app)[1] == "SIN(x)SIN(x)"


async def test_pasting_takes_the_offered_label_hash_and_all(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y")
        await pilot.press("ctrl+c")
        await pilot.press("s")
        assert prompt(app) == ("SIMPLIFY expression:", "#2")
        # As F3 does: what is pasted stands where an expression goes, and a `#`
        # in front of one says nothing.
        await pilot.press("ctrl+v")
        assert prompt(app)[1] == "y"


async def test_ctrl_v_pastes_onto_any_line_but_not_at_the_menu(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x")
        await pilot.press("ctrl+c")
        await pilot.press("t", "s", "d")
        assert prompt(app)[0] == "TRANSFER SAVE DERIVE file:"
        # Pasting is about the text on the line, so it applies to the lines that
        # collect something other than an expression too.
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


# -- F3 and F4, the original's keys for the same thing -------------------------
#
# They write the highlighted expression onto the line directly. Copy and paste
# are what the help offers, but a session driven from the manual still reaches
# for these, so they go on working.


async def test_f3_writes_the_highlighted_expression_onto_the_author_line(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x + 1", "y")
        await pilot.press("a", "f3")
        assert prompt(app) == ("AUTHOR expression:", "y")
        # Walking under the line is what picks what F3 takes.
        await pilot.press("up", "f3")
        assert prompt(app)[1] == "yx+1"


async def test_f4_fences_what_it_writes(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x + 1")
        await pilot.press("a", *"2", "f4", "enter")
        assert numbered(app)[-1] == "#2: 2*(x+1)"


async def test_f3_writes_at_the_cursor_and_leaves_it_after(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x + 1")
        await pilot.press("a", *"ab", "left", "f3")
        assert prompt(app)[1] == "ax+1b"
        # The cursor sits after what was written, so the next thing typed
        # follows it rather than landing back where the line was.
        await pilot.press(*"c")
        assert prompt(app)[1] == "ax+1cb"


async def test_f3_takes_only_the_highlighted_part(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x (x + 1)")
        # Descending happens before the line goes up: with a prompt line on
        # screen the sideways keys are the line's own.
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
        await worksheet(pilot, "x + 1")
        await pilot.press(*keys)
        assert prompt(app)[0] == line
        await pilot.press("f3")
        assert prompt(app)[1] == "x+1"


async def test_f3_writes_onto_the_substitute_value_line(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x + 1", "a")
        await pilot.press("m", "s", "enter")
        assert prompt(app)[0] == "MANAGE SUBSTITUTE value:"
        # The manual gives F3 for this line by name: the value put in the place
        # of a variable is as often an expression already on screen as it is
        # something typed.
        await pilot.press("f3")
        assert prompt(app)[1] == "a"


async def test_the_substitute_value_line_takes_what_the_highlight_walks_onto(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x (x + 1)", "a + b")
        await pilot.press("m", "s", "enter")
        assert prompt(app) == ("MANAGE SUBSTITUTE value:", "a")
        # Which expression is being substituted into is settled, but what goes
        # in the variable's place is not, and this is how it is found: walk to
        # it, walk into it, and take it.
        await pilot.press("up")
        assert highlighted_expression(app) == "x·(x + 1)"
        await pilot.press("f6", "right", "right")
        assert highlighted_expression(app) == "x + 1"
        await pilot.press("f3")
        assert prompt(app)[1] == "x+1"


async def test_f3_writes_nothing_on_a_line_that_takes_no_expression(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y")
        # Jump takes a label number, Declare a variable's name: neither is an
        # expression, so neither takes what F3 would write.
        for keys in (("j",), ("d", "v")):
            await pilot.press(*keys)
            await pilot.press("f3", "f4")
            assert prompt(app)[1] == ""
            await pilot.press("escape")


async def test_f3_takes_the_offered_label_hash_and_all(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y")
        await pilot.press("s")
        assert prompt(app) == ("SIMPLIFY expression:", "#2")
        # The `#` stands outside the selection a typed digit replaces, but what
        # F3 writes is an expression and there is no `#` in front of one.
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
        # The key needs a terminal that tells a modifier from an Escape, which
        # is what the keyboard protocol Textual asks for is for.
        await pilot.press("a", "alt+minus", *"x", "enter")
        # The glyph is drawn, and written `"+-"` - the spelling the original
        # writes it in, quotes and all - where the entry is text.
        assert work_area(app) == ["#1:  ±x"]
        assert numbered(app)[-1] == '#1: "+-"x'


async def test_a_glyph_written_by_an_alt_key_parses(app):
    async with app.run_test() as pilot:
        await pilot.press("a", "alt+p", "enter")
        assert work_area(app) == ["#1:  π"]
        # The subscript operator is a word here and a glyph in the original, so
        # the key writes it spaced the way the printer spaces it.
        await pilot.press("a", *"x", "alt+v", "2", "enter")
        assert numbered(app)[-1] == "#2: x SUB 2"


async def test_the_alt_keys_write_on_any_line_but_not_at_the_menu(app):
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        assert prompt(app)[0] == "TRANSFER LOAD DERIVE file:"
        await pilot.press("alt+p")
        assert prompt(app)[1].endswith("π")
        await pilot.press("escape", "escape", "escape")
        # At the menu the letter under Alt is not a mnemonic either: nothing
        # runs, and the menu is where it was.
        await pilot.press("alt+p")
        assert highlighted_menu_option(app) == "Author"
        assert message(app) == "Enter option"


async def test_f6_hands_the_sideways_keys_to_the_highlight_and_back(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x (x + 1)")
        await pilot.press("a")
        assert flags(app) == ["Lin"]
        await pilot.press("f6")
        assert flags(app) == []
        await pilot.press("right", "right")
        assert highlighted_expression(app) == "x + 1"
        await pilot.press("f3")
        assert prompt(app)[1] == "x+1"
        # Back in line-edit mode the same keys are the cursor's again.
        await pilot.press("f6")
        assert flags(app) == ["Lin"]
        await pilot.press("home", *"2")
        assert prompt(app)[1] == "2x+1"
        assert highlighted_expression(app) == "x + 1"


async def test_the_arrow_key_setting_says_which_mode_a_line_starts_in(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x + 1")
        await author(pilot, "ArrowKeyMode := Subexpression")
        await pilot.press("a")
        assert flags(app) == []
        # F6 is for the line in hand only: the next one starts where the
        # setting says again.
        await pilot.press("f6")
        assert flags(app) == ["Lin"]
        await pilot.press("escape", "a")
        assert flags(app) == []


async def test_no_line_up_means_no_mode_words_to_report(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x")
        assert flags(app) == []
        # Overwrite outlives the line it was set on, but there is no line for
        # the status line to say it of.
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
        # At the end of the line there is nothing to stand on, so it grows.
        await pilot.press("end", *"d")
        assert prompt(app)[1] == "XYcd"
        await pilot.press("insert")
        assert flags(app) == ["Lin"]
        await pilot.press("home", *"Z")
        assert prompt(app)[1] == "ZXYcd"
        # The original spelled Ins as Ctrl-V too; that key pastes here, so it
        # leaves the mode alone.
        await pilot.press("ctrl+v")
        assert flags(app) == ["Lin"]


async def test_overwrite_stands_until_it_is_turned_back(app):
    async with app.run_test() as pilot:
        await pilot.press("a", "insert", "escape")
        # It is how the user is typing rather than what a command asked, so it
        # outlives the line it was set on.
        await pilot.press("a", *"abc", "home", *"X")
        assert flags(app) == ["Ovr", "Lin"]
        assert prompt(app)[1] == "Xbc"


async def test_overwrite_replaces_what_is_selected_as_inserting_does(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y")
        await pilot.press("s", "insert")
        # The offered label comes up selected, and typing takes it away whole
        # rather than standing on its first character.
        assert prompt(app)[1] == "#2"
        await pilot.press("1")
        assert prompt(app)[1] == "#1"


# -- the kill ring: what a deletion took out, and Ctrl-Y putting it back ------


async def test_ctrl_y_puts_back_what_ctrl_u_took_out(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x + 1")
        await pilot.press("ctrl+u")
        assert prompt(app)[1] == ""
        await pilot.press("ctrl+y")
        assert prompt(app) == ("AUTHOR expression:", "x + 1")
        await pilot.press("enter")
        assert entries(app) == ["x+1"]


async def test_a_kill_is_put_back_at_the_cursor_and_the_cursor_follows_it(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"a + b", "home")
        # Ctrl-K takes the line from the cursor on, so the whole of it here.
        await pilot.press("ctrl+k", *"2 (", "ctrl+y", *")")
        assert prompt(app)[1] == "2 (a + b)"


async def test_what_one_line_deleted_comes_back_on_the_next(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"sin(x)/x", "ctrl+u", "escape")
        # The ring belongs to the line rather than to the command, so a line
        # abandoned altogether still leaves what it deleted to be put back.
        await pilot.press("a", "ctrl+y", "enter")
        assert entries(app) == ["SIN(x)/x"]
        # And on a line another command reads, not only on another Author: the
        # label Simplify offers is taken out and the expression put in instead.
        await pilot.press("s", "ctrl+u", "ctrl+y", "enter")
        assert entries(app) == ["SIN(x)/x", "SIN(x)/x"]


async def test_deletions_that_follow_one_another_come_back_as_one(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"a + b + c")
        # Three words taken a word at a time are one kill, as Emacs and bash
        # join theirs, so the line comes back whole rather than a word of it.
        await pilot.press("ctrl+w", "ctrl+w")
        assert prompt(app)[1] == "a + "
        await pilot.press("ctrl+w")
        assert prompt(app)[1] == ""
        await pilot.press("ctrl+y")
        assert prompt(app)[1] == "a + b + c"


async def test_a_deletion_after_something_else_starts_a_kill_of_its_own(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"a b", "ctrl+w", *"c", "ctrl+w")
        assert prompt(app)[1] == "a "
        # Typing between the two says they are separate deletions, so the last
        # of them is what Ctrl-Y puts back.
        await pilot.press("ctrl+y")
        assert prompt(app)[1] == "a c"


async def test_alt_y_walks_back_through_the_kills_before_the_last(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"first", "ctrl+u", *"second", "ctrl+u")
        await pilot.press("ctrl+y")
        assert prompt(app)[1] == "second"
        await pilot.press("alt+y")
        assert prompt(app)[1] == "first"
        # Past the oldest it comes round to the newest again.
        await pilot.press("alt+y")
        assert prompt(app)[1] == "second"


async def test_alt_y_takes_the_place_of_the_yank_and_leaves_the_rest(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"one", "ctrl+u", *"two", "ctrl+u")
        await pilot.press(*"f(", "ctrl+y", *")")
        assert prompt(app)[1] == "f(two)"
        # The cursor has moved on since the yank, so there is nothing left to
        # reconsider: what Alt-Y would have swapped stands as it is.
        await pilot.press("alt+y")
        assert prompt(app)[1] == "f(two)"


async def test_alt_y_says_nothing_when_the_last_key_was_not_a_yank(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "ctrl+u", *"y")
        await pilot.press("alt+y")
        assert prompt(app)[1] == "y"


async def test_ctrl_y_on_a_line_that_deleted_nothing_writes_nothing(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x")
        await pilot.press("ctrl+y", "alt+y")
        assert prompt(app)[1] == "x"


async def test_a_character_deleted_on_its_own_is_not_kept(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"ab", "ctrl+u", *"cd", "backspace", "delete")
        # Backspace and Delete are corrections rather than kills, so the line
        # Ctrl-U took is still what comes back.
        await pilot.press("ctrl+y")
        assert prompt(app)[1] == "cab"


async def test_the_jump_line_walks_and_enter_alone_keeps_where_it_went(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y", "z")
        await pilot.press("j")
        await pilot.press("up", "up")
        assert highlighted_expression(app) == "x"
        assert prompt(app)[1] == ""
        await pilot.press("enter")
        assert highlighted_expression(app) == "x"
        assert message(app) == "Enter option"


async def test_the_page_keys_walk_a_paneful_at_a_time(app):
    async with app.run_test(size=(80, 25)) as pilot:
        # Twenty-five rows leave a twenty-row pane, which is the original's.
        for number in range(1, 31):
            await author(pilot, str(number))
        await pilot.press("j", *"20", "enter")
        await pilot.press("pageup")
        assert selected_number(app) == 11
        await pilot.press("pageup")
        assert selected_number(app) == 2
        await pilot.press("pagedown")
        assert selected_number(app) == 10
        # The two spellings of first and last the original gives them.
        await pilot.press("ctrl+pageup")
        assert selected_number(app) == 1
        await pilot.press("ctrl+pagedown")
        assert selected_number(app) == 30
        await pilot.press("ctrl+home")
        assert selected_number(app) == 1


async def test_the_page_keys_walk_under_a_prompt_line_too(app):
    async with app.run_test(size=(80, 25)) as pilot:
        for number in range(1, 31):
            await author(pilot, str(number))
        await pilot.press("j", *"20", "enter")
        await pilot.press("s")
        assert prompt(app)[1] == "#20"
        await pilot.press("pageup")
        assert selected_number(app) == 11
        assert prompt(app)[1] == "#11"


async def test_the_page_keys_label_a_dialogs_field_too(app):
    async with app.run_test(size=(80, 25)) as pilot:
        for number in range(1, 31):
            await author(pilot, str(number))
        await pilot.press("j", *"20", "enter")
        await pilot.press("r")
        assert band(app) == [" REMOVE: Start: 20     End: 20"]
        await pilot.press("pageup")
        assert band(app) == [" REMOVE: Start: 11     End: 20"]


async def test_factors_variable_line_takes_no_notice_of_them(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y", "x^2 - y^2")
        await pilot.press("f", "enter")
        assert prompt(app)[0] == "FACTOR variable 1:"
        # What to factor has been settled by now, so the highlight stays put.
        for key in ("up", "down", "pageup", "pagedown", "ctrl+home", "ctrl+end"):
            await pilot.press(key)
            assert highlighted_expression(app) == " 2    2\nx  - y"


@pytest.mark.parametrize(
    ("keys", "line"),
    [
        (("c", "d", "enter"), "CALCULUS DIFFERENTIATE variable:"),
        (("l", "enter"), "SOLVE variable:"),
        (("m", "a", "enter"), "ANNOTATION:"),
        (("m", "s", "enter"), "MANAGE SUBSTITUTE value:"),
    ],
    ids=str,
)
async def test_the_lines_that_collect_a_name_walk_under_it_too(app, keys, line):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y", "a z + b")
        await pilot.press(*keys)
        offered = prompt(app)[1]
        assert prompt(app)[0] == line
        await pilot.press("up")
        assert highlighted_expression(app) == "y"
        await pilot.press("ctrl+home")
        assert highlighted_expression(app) == "x"
        # None of them names an expression, so none takes the label walked onto.
        assert prompt(app) == (line, offered)


async def test_a_line_that_takes_no_expression_is_offered_no_f6(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x (x + 1)", "y")
        await pilot.press("m", "a", "enter")
        assert prompt(app)[0] == "ANNOTATION:"
        # There is no expression on this line to walk, so there is no mode to
        # be in and nothing for the status line to say.
        assert flags(app) == []
        await pilot.press("f6")
        assert flags(app) == []


async def test_subexpression_mode_leaves_such_a_line_its_sideways_keys(app):
    async with app.run_test() as pilot:
        await author(pilot, "ArrowKeyMode := Subexpression")
        await worksheet(pilot, "x (x + 1)")
        await pilot.press("c", "d", "enter")
        assert prompt(app) == ("CALCULUS DIFFERENTIATE variable:", "x")
        await pilot.press("end", *"y")
        assert prompt(app)[1] == "xy"
        assert highlighted_expression(app) == "x·(x + 1)"


@pytest.mark.parametrize(
    ("keys", "line"),
    [
        (("d", "v"), "DECLARE VARIABLE name:"),
        (("d", "v", *"a", "enter", "v"), "DECLARE VARIABLE value:"),
        (("d", "f"), "DECLARE FUNCTION name:"),
        (("d", "f", *"g", "enter"), "DECLARE FUNCTION value:"),
        (("d", "f", *"g", "enter", "enter"), "DECLARE FUNCTION variable:"),
        (("d", "r", "2", "enter"), "VECTOR element:"),
        (("d", "m", "2", "tab", "2", "enter"), "MATRIX element:"),
    ],
    ids=str,
)
async def test_the_declare_lines_walk_but_are_never_labelled(app, keys, line):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y", "z")
        await pilot.press(*keys)
        offered = prompt(app)[1]
        assert prompt(app)[0] == line
        await pilot.press("up")
        assert highlighted_expression(app) == "y"
        await pilot.press("ctrl+home")
        assert highlighted_expression(app) == "x"
        # None of them names an expression, so none takes the label walked onto.
        assert prompt(app) == (line, offered)


#: A render far wider than the eighty-column pane, drawn exactly as authored,
#: so the columns on show can be read off it by slicing.
WIDE = " + ".join(str(number) for number in range(100, 130))

#: Columns a render has to itself in an eighty-column pane, the label field
#: holding the other five, and what one Ctrl-Right takes: a third of them.
SHOWABLE = 75
STEP = 25

#: Columns kept on show beyond a highlight the pane has scrolled to follow.
MARGIN = 2


def from_the_right(term):
    """The shift that brings `term` in from the right edge, and no more."""
    return WIDE.index(term) + len(term) + MARGIN - SHOWABLE


async def scrolled(pilot, keys):
    """A worksheet with the wide entry in the middle, selected and scrolled."""
    await author(pilot, "x+1")
    await author(pilot, WIDE)
    await author(pilot, "x+1")
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
        # The highlight goes with it, cut off where the render now starts.
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
        # The last column of the render, and no blanks past it.
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
        # Down to the entry below and back up. The entry comes back selected
        # whole, which is a selection starting at the left edge of its render.
        await pilot.press("down", "up")
        assert work_area(app)[2] == "#2:  " + WIDE
        # And a move to the first part of it asks for the same thing.
        await pilot.press("ctrl+right")
        assert work_area(app)[2] == "#2:  " + WIDE[STEP:]
        await pilot.press("right")
        assert work_area(app)[2] == "#2:  " + WIDE
        assert highlighted_expression(app) == "100"


async def test_a_highlight_walking_off_the_right_takes_the_render_with_it(app):
    async with app.run_test() as pilot:
        # Twelve terms in, the render still reaches the highlight unscrolled.
        await scrolled(pilot, ["right"] * 12)
        assert highlighted_expression(app) == "111"
        assert work_area(app)[2] == "#2:  " + WIDE
        # The thirteenth is past the edge, so the render moves - by the two
        # columns that bring it in with a margin, and by no more than that.
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
        # Walking back, the render follows once the highlight reaches its edge,
        # and stops with the same margin standing in front of it.
        await pilot.press(*["left"] * 12)
        assert highlighted_expression(app) == "112"
        assert work_area(app)[2] == "#2:  " + WIDE[WIDE.index("112") - MARGIN :]
        await pilot.press("home")
        assert highlighted_expression(app) == "100"
        assert work_area(app)[2] == "#2:  " + WIDE


async def test_a_highlight_already_on_show_moves_no_render(app):
    async with app.run_test() as pilot:
        # Scrolled by hand, with the highlight still inside what is on show:
        # the pane stays where it was put rather than snapping to the highlight.
        await scrolled(pilot, ["right"] * 11 + ["ctrl+right"])
        assert highlighted_expression(app) == "110"
        assert work_area(app)[2] == "#2:  " + WIDE[STEP:]
        await pilot.press("right")
        assert highlighted_expression(app) == "111"
        assert work_area(app)[2] == "#2:  " + WIDE[STEP:]


async def test_a_selection_too_wide_to_show_is_shown_from_its_left(app):
    async with app.run_test() as pilot:
        # The whole entry is wider than the pane, so its two ends cannot both
        # be brought in. The end it is read from wins.
        await scrolled(pilot, ["ctrl+right", "down", "up"])
        assert work_area(app)[2] == "#2:  " + WIDE


async def test_a_highlight_scrolled_off_the_left_is_not_painted(app):
    async with app.run_test() as pilot:
        await scrolled(pilot, ["right"])
        assert highlighted_expression(app) == "100"
        await pilot.press("ctrl+right")
        assert work_area(app)[2] == "#2:  " + WIDE[STEP:]
        assert highlighted_expression(app) == ""
