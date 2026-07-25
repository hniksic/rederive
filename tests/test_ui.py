"""Smoke tests driving the real app through Textual's pilot."""

import pytest
from rich.text import Text
from textual.widgets import Static

from rederive.ui.app import RederiveApp
from rederive.ui.theme import STYLES


def _text(widget):
    """The Rich text a widget currently shows."""
    content = getattr(widget, "content", None)
    return content if isinstance(content, Text) else widget.render()


def _styled(widget, style):
    """The runs of `widget`'s rendered text carrying `style`."""
    text = _text(widget)
    return [
        text.plain[span.start : span.end]
        for span in text.spans
        if span.style == style
    ]


def highlighted_menu_option(app):
    (option,) = _styled(app.query_one("#menu"), STYLES["option-highlight"])
    return option


def highlighted_expression(app):
    return "".join(_styled(app.query_one("#work-content", Static), STYLES["selection"]))


def message(app):
    return _text(app.query_one("#message")).plain.strip()


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
        assert highlighted_expression(app) == "x (x + 1)"
        assert message(app) == "Enter option"
        assert _text(app.query_one("#status")).plain.strip().startswith("User")


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
        assert highlighted_expression(app) == "x x x"
        await pilot.press("up", "up")
        assert highlighted_expression(app) == "x (x + 1)"
        await pilot.press("right")
        assert highlighted_expression(app) == "x"
        await pilot.press("right")
        assert highlighted_expression(app) == "x + 1"
        await pilot.press("down")
        assert highlighted_expression(app) == "x"
        await pilot.press("up", "up")
        assert highlighted_expression(app) == "x (x + 1)"
        # The menu highlight never moved while the arrows walked the history.
        assert highlighted_menu_option(app) == "Author"


async def test_quit_asks_before_abandoning_expressions(app):
    async with app.run_test() as pilot:
        await author(pilot, "x")
        await pilot.press("q")
        assert message(app) == "Abandon expressions (Y/N)?"
        await pilot.press("n")
        assert message(app) == "Enter option"
        assert app.is_running
        await pilot.press("q", "y")
        await pilot.pause()
        assert not app.is_running
