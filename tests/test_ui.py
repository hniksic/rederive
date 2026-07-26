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


async def test_author_appends_and_selects_the_new_entry(app):
    async with app.run_test() as pilot:
        await author(pilot, "x (x + 1)")
        assert [entry.text for entry in app.session.entries] == ["x (x + 1)"]
        # Typed as juxtaposition, drawn with the times operator.
        assert work_area(app) == ["#1:  x·(x + 1)"]
        assert highlighted_expression(app) == "x·(x + 1)"
        assert message(app) == "Enter option"
        assert text_of(app.query_one("#status")).plain.strip().startswith("User")


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
        assert app.factoring is None
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


# -- Remove and Unremove ------------------------------------------------------
#
# Every screen asserted here was checked against the original.


def numbered(app):
    """The history as the work area labels it."""
    return [f"#{entry.number}: {entry.text}" for entry in app.session.entries]


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
