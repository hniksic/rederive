"""Smoke tests driving the real app through Textual's pilot."""

import pytest
from screen import (
    annotation,
    band,
    entries,
    highlighted,
    highlighted_expression,
    highlighted_rows,
    message,
    prompt,
    text_of,
    work_area,
)

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
        await pilot.press("b")
        assert message(app) == "Build: not implemented yet"
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


async def test_author_appends_and_selects_the_new_entry(app):
    async with app.run_test() as pilot:
        await author(pilot, "x (x + 1)")
        assert [entry.text for entry in app.session.entries] == ["x (x + 1)"]
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
        assert entries(app) == ["[1, 2+3]", "[1, 5]"]


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
        assert [entry.text for entry in app.session.entries][-1] == "2 (8 + 7) / 9"
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
        assert entries(app) == ["x^2 y^2 - 1"]


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
        assert entries(app) == ["1/(x^2 y^2 - 1)"]


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
    """The history as the work area labels it."""
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
        assert entries(app) == ["z :ε Complex"]
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
        assert entries(app) == ["n :ε Integer [1, 5)"]


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
        assert entries(app) == ["area := pi r^2"]
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
        assert entries(app) == ["f(x, y) :="]
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
        assert entries(app) == ["[1, 2, x]"]
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
        assert entries(app) == ["[[1, 0], [0, 4]]"]
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


async def test_f3_writes_the_highlighted_expression_onto_the_author_line(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x + 1", "y")
        await pilot.press("a", "f3")
        assert prompt(app) == ("AUTHOR expression:", "y")
        # Walking under the line is what picks what F3 takes.
        await pilot.press("up", "f3")
        assert prompt(app)[1] == "yx + 1"


async def test_f4_fences_what_it_writes(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x + 1")
        await pilot.press("a", *"2", "f4", "enter")
        assert numbered(app)[-1] == "#2: 2(x + 1)"


async def test_f3_writes_at_the_cursor_and_leaves_it_after(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x + 1")
        await pilot.press("a", *"ab", "left", "f3")
        assert prompt(app)[1] == "ax + 1b"
        # The cursor sits after what was written, so the next thing typed
        # follows it rather than landing back where the line was.
        await pilot.press(*"c")
        assert prompt(app)[1] == "ax + 1cb"


async def test_f3_takes_only_the_highlighted_part(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x (x + 1)")
        # Descending happens before the line goes up: with a prompt line on
        # screen the sideways keys are the line's own.
        await pilot.press("right", "right")
        assert highlighted_expression(app) == "x + 1"
        await pilot.press("a", "f3")
        assert prompt(app)[1] == "x + 1"


async def test_f3_writes_nothing_on_any_other_line(app):
    async with app.run_test() as pilot:
        await worksheet(pilot, "x", "y")
        for keys, line in ((("s",), "#2"), (("j",), ""), (("d", "v"), "")):
            await pilot.press(*keys)
            await pilot.press("f3", "f4")
            assert prompt(app)[1] == line
            await pilot.press("escape")


async def test_f3_on_an_empty_worksheet_writes_nothing(app):
    async with app.run_test() as pilot:
        await pilot.press("a", "f3", "f4")
        assert prompt(app) == ("AUTHOR expression:", "")


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
