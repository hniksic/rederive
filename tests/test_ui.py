"""Smoke tests driving the real app through Textual's pilot."""

import pytest
from screen import (
    highlighted,
    highlighted_expression,
    highlighted_rows,
    message,
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
        await pilot.press("f")
        assert message(app) == "Factor: not implemented yet"
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
        author_input = app.query_one("#author-input")
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
