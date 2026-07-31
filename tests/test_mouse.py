"""The mouse: clicking, pointing at a subexpression, and the wheel.

Every gesture here has a key that does the same thing, and these say so: what a
click on a menu word does is what that word's own letter does, what a click on
an expression does is what the arrows do, and what the wheel does is what
Ctrl-Right and the paging keys do. So the tests are mostly about geometry - the
cell the pointer was over, and what was drawn there - since that is the one part
the keyboard never had to answer.

The bands lay themselves out for the width there is, so nothing here counts
columns by hand: each cell is found in what the band actually drew.
"""

import pytest
from screen import (
    band,
    band_id,
    chosen,
    completions,
    content,
    entries,
    highlighted,
    highlighted_expression,
    message,
    prompt,
    work_area,
)
from test_compute import Blocking, settle
from textual.events import (
    MouseScrollDown,
    MouseScrollLeft,
    MouseScrollRight,
    MouseScrollUp,
)

from rederive.model.session import Session
from rederive.ui.app import COPIED, COPIED_TEXT, MODE_COMPUTE, RederiveApp


@pytest.fixture
def app():
    return RederiveApp()


async def author(pilot, text):
    await pilot.press("a")
    await pilot.press(*text)
    await pilot.press("enter")


async def click_cell(pilot, app, row, column, number=None):
    """Click a cell of a window's work area, as its lines are numbered.

    The pause is what makes the cell the right one: the panes are placed
    absolutely as the windows are, one turn of the loop after the keystroke that
    moved them, so a region read without waiting is where the pane was. A
    terminal has no such problem - it reports the cell a user pointed at, on the
    screen as composited - so this is the test's own bookkeeping.
    """
    await pilot.pause()
    await pilot.click(content(app, number), offset=(column, row))


async def click_on(pilot, app, text, row=None, number=None):
    """Click the first cell of `text` where the work area has drawn it."""
    lines = work_area(app, number)
    rows = range(len(lines)) if row is None else (row,)
    for at in rows:
        column = lines[at].find(text)
        if column != -1:
            await click_cell(pilot, app, at, column, number)
            return
    raise AssertionError(f"{text!r} is not on the work area")


async def click_word(pilot, app, word):
    """Click a word of the command band, wherever it has been laid out."""
    await pilot.pause()
    widget = app.query_one(band_id(app))
    for row, line in enumerate(band(app)):
        column = line.find(word)
        if column != -1:
            await pilot.click(widget, offset=(column, row))
            return
    raise AssertionError(f"{word!r} is not on the band")


async def wheel(
    pilot, app, widget, direction, offset=(0, 0), sideways=False, shift=False
):
    """Turn the wheel one notch over a cell of `widget`.

    Textual's pilot drives clicks and hovers but not the wheel, so the event is
    posted the way a terminal reports one: at a screen cell, naming no widget.
    """
    await pilot.pause()
    forward, backward = (
        (MouseScrollRight, MouseScrollLeft)
        if sideways
        else (MouseScrollDown, MouseScrollUp)
    )
    at = widget.region.offset + offset
    app.screen._forward_event(
        (forward if direction > 0 else backward)(
            None,
            x=at.x,
            y=at.y,
            delta_x=0,
            delta_y=0,
            button=0,
            shift=shift,
            meta=False,
            ctrl=False,
            screen_x=at.x,
            screen_y=at.y,
        )
    )
    await pilot.pause()


# -- pointing at an expression -------------------------------------------------


async def test_a_click_on_an_expression_selects_that_expression(app):
    async with app.run_test() as pilot:
        await author(pilot, "x+1")
        await author(pilot, "y+2")
        assert highlighted_expression(app) == "y + 2"
        await click_on(pilot, app, "x")
        assert app.session.selected == 0
        assert app.session.route == ()
        assert highlighted_expression(app) == "x + 1"


async def test_a_click_inside_the_selected_expression_goes_one_level_in(app):
    """A term of a long sum is reachable without hitting its one character."""
    async with app.run_test() as pilot:
        await author(pilot, "(x+1)/y")
        assert highlighted_expression(app) == " x + 1\n───────\n   y"
        # The `x` of the numerator: the numerator first, then the `x` itself,
        # and an atom is where it stops.
        await click_on(pilot, app, "x")
        assert highlighted_expression(app) == "x + 1"
        await click_on(pilot, app, "x")
        assert highlighted_expression(app) == "x"
        await click_on(pilot, app, "x")
        assert highlighted_expression(app) == "x"


async def test_a_click_on_another_branch_takes_it_where_the_two_part(app):
    async with app.run_test() as pilot:
        await author(pilot, "(x+1)/(y+2)")
        await click_on(pilot, app, "x")
        await click_on(pilot, app, "x")
        assert highlighted_expression(app) == "x"
        # The denominator is a different branch, so the click lands on the whole
        # of it - the level at which it and the `x` differ - and the next one
        # goes in.
        await click_on(pilot, app, "y")
        assert highlighted_expression(app) == "y + 2"
        await click_on(pilot, app, "y")
        assert highlighted_expression(app) == "y"


async def test_a_click_on_an_operator_goes_back_out_to_what_owns_it(app):
    """The `+` between two terms belongs to the sum and not to either term."""
    async with app.run_test() as pilot:
        await author(pilot, "x+y")
        await click_on(pilot, app, "x")
        assert highlighted_expression(app) == "x"
        await click_on(pilot, app, "+")
        assert highlighted_expression(app) == "x + y"


async def test_a_click_on_the_label_selects_the_whole_expression(app):
    """The number in front of a render names all of it and no part of it."""
    async with app.run_test() as pilot:
        await author(pilot, "x+y")
        await click_on(pilot, app, "x")
        assert app.session.route == (0,)
        await click_on(pilot, app, "#1")
        assert app.session.route == ()


async def test_a_click_past_the_end_of_a_render_selects_it_whole(app):
    async with app.run_test() as pilot:
        await author(pilot, "x+y")
        await click_on(pilot, app, "x")
        assert app.session.route == (0,)
        await click_cell(pilot, app, 0, len(work_area(app)[0]) + 5)
        assert app.session.route == ()


async def test_a_click_on_the_blank_row_between_two_entries_moves_nothing(app):
    async with app.run_test() as pilot:
        await author(pilot, "x")
        await author(pilot, "y")
        assert work_area(app) == ["#1:  x", "", "#2:  y"]
        await click_cell(pilot, app, 1, 5)
        assert app.session.selected == 1


async def test_a_click_above_a_short_history_moves_nothing(app):
    """The pane is taller than the history, and the space above it is nobody's."""
    async with app.run_test() as pilot:
        await author(pilot, "x")
        await author(pilot, "y")
        await pilot.click(app.work_area, offset=(3, 0))
        assert (app.session.selected, app.session.route) == (1, ())


async def test_a_click_on_the_opening_notice_does_nothing(app):
    async with app.run_test() as pilot:
        await pilot.click(content(app), offset=(10, 5))
        assert app.session.selected is None
        assert "R E D E R I V E" in "".join(work_area(app))


async def test_a_click_lands_where_a_scrolled_render_is_drawn(app):
    """A render scrolled sideways is read at the columns it is showing."""
    async with app.run_test() as pilot:
        await author(pilot, "+".join("abcdefghijklmnopqrstuvwxyz"))
        await pilot.press("ctrl+right")
        assert app.work_area.shift
        # The render has moved under the label field, so the letter on screen is
        # not the letter that column held before the scroll.
        shown = work_area(app)[0]
        assert not shown.startswith("#1:  a")
        column = next(at for at in range(5, len(shown)) if shown[at].isalpha())
        await click_cell(pilot, app, 0, column)
        assert highlighted_expression(app) == shown[column]


async def test_a_click_on_the_label_of_a_scrolled_render_selects_it_whole(app):
    """The label field does not move with the render, and neither does its rule."""
    async with app.run_test() as pilot:
        await author(pilot, "+".join("abcdefghijklmnopqrstuvwxyz"))
        await click_cell(pilot, app, 0, 5)
        assert app.session.route != ()
        await pilot.press("ctrl+right")
        await click_cell(pilot, app, 0, 1)
        assert app.session.route == ()


# -- clicking with a line up ---------------------------------------------------


async def test_a_click_relabels_the_line_that_was_offered_a_label(app):
    """Which is what an arrow key does, and what the manual recommends."""
    async with app.run_test() as pilot:
        await author(pilot, "x+1")
        await author(pilot, "y+2")
        await pilot.press("s")
        assert prompt(app) == ("SIMPLIFY expression:", "#2")
        await click_on(pilot, app, "x")
        assert prompt(app) == ("SIMPLIFY expression:", "#1")
        # And the line still has the keyboard: nothing on screen took it.
        assert app.focused is app.query_one("#prompt-input")


async def test_a_click_reaches_a_subexpression_under_a_line(app):
    """The arrow key mode is about a key two things want; nothing wants the mouse."""
    async with app.run_test() as pilot:
        await author(pilot, "x+1")
        await pilot.press("s")
        assert app.line_edit
        await click_on(pilot, app, "x")
        await click_on(pilot, app, "x")
        assert highlighted_expression(app) == "x"


async def test_a_click_moves_nothing_on_the_one_line_that_holds_the_highlight(app):
    """The variable line Factor collects on, where the original holds it still."""
    async with app.run_test() as pilot:
        await author(pilot, "x*y+x")
        await pilot.press("f", "enter")
        assert message(app).startswith("Return for all")
        await click_on(pilot, app, "x")
        assert app.session.route == ()


# -- clicking a menu word ------------------------------------------------------


async def test_a_click_on_a_menu_word_runs_that_command(app):
    async with app.run_test() as pilot:
        await click_word(pilot, app, "Author")
        assert prompt(app) == ("AUTHOR expression:", "")


async def test_a_click_on_a_word_of_the_second_row_runs_that_command(app):
    """Where a word is drawn is what a click has to be read against."""
    async with app.run_test() as pilot:
        assert len(band(app)) == 2 and "Window" in band(app)[1]
        await click_word(pilot, app, "Window")
        assert band(app)[0].startswith(" WINDOW:")


async def test_a_click_on_a_menu_word_leaves_the_highlight_where_it_is(app):
    """As typing a mnemonic letter does: running a command is not stepping to it."""
    async with app.run_test() as pilot:
        await click_word(pilot, app, "Window")
        assert highlighted(app) == "Close"
        await pilot.press("escape")
        assert highlighted(app) == "Author"


async def test_a_click_between_two_menu_words_does_nothing(app):
    async with app.run_test() as pilot:
        line = band(app)[0]
        gap = line.index("Author") - 1
        assert line[gap] == " "
        await pilot.click(app.query_one("#menu"), offset=(gap, 0))
        assert band(app)[0].startswith(" COMMAND:")
        assert message(app) == "Enter option"


async def test_a_click_on_the_menu_title_does_nothing(app):
    async with app.run_test() as pilot:
        await pilot.click(app.query_one("#menu"), offset=(2, 0))
        assert band(app)[0].startswith(" COMMAND:")


async def test_a_click_on_a_menu_word_does_nothing_while_a_question_is_up(app):
    """A confirmation leaves the menu up with nothing on it to pick."""
    async with app.run_test() as pilot:
        await author(pilot, "x")
        await click_word(pilot, app, "Quit")
        assert message(app) == "Abandon expressions (Y/N)?"
        await click_word(pilot, app, "Author")
        assert message(app) == "Abandon expressions (Y/N)?"
        await pilot.press("n")


async def test_a_click_chooses_a_help_subject_and_comes_back(app):
    async with app.run_test() as pilot:
        await click_word(pilot, app, "Help")
        assert band(app)[0].startswith(" HELP:")
        await click_word(pilot, app, "Basics")
        assert work_area(app)[0].strip()
        assert band(app)[0].startswith(" HELP BASICS:")
        await click_word(pilot, app, "Resume")
        assert band(app)[0].startswith(" HELP:")
        await click_word(pilot, app, "Resume")
        assert band(app)[0].startswith(" COMMAND:")


# -- clicking an Options dialog ------------------------------------------------


async def test_a_click_on_a_value_sets_it_and_moves_the_highlight_on(app):
    """Which is what typing that value's own letter does."""
    async with app.run_test() as pilot:
        await pilot.press("o", "n")
        assert app.editor.values["Notation"] == "Rational"
        await click_word(pilot, app, "Decimal")
        assert app.editor.values["Notation"] == "Decimal"
        assert app.editor.field.setting == "NotationDigits"
        # And nothing is applied until the dialog is committed.
        assert app.settings["Notation"] == "Rational"
        await pilot.press("enter")
        assert app.settings["Notation"] == "Decimal"


async def test_a_click_on_a_value_of_a_one_field_dialog_settles_it(app):
    async with app.run_test() as pilot:
        assert app.settings["Mute"] == "No"
        await pilot.press("o", "m")
        assert band(app)[0].startswith(" OPTIONS MUTE:")
        await click_word(pilot, app, "Yes")
        assert app.settings["Mute"] == "Yes"
        assert band(app)[0].startswith(" COMMAND:")


async def test_a_click_on_a_field_moves_the_highlight_onto_it_and_no_more(app):
    async with app.run_test() as pilot:
        await pilot.press("o", "n")
        assert app.editor.field.setting == "Notation"
        await click_word(pilot, app, "Digits")
        assert app.editor.field.setting == "NotationDigits"
        assert app.editor.values["Notation"] == "Rational"


async def test_escape_after_a_click_still_abandons_the_dialog(app):
    async with app.run_test() as pilot:
        await pilot.press("o", "n")
        await click_word(pilot, app, "Scientific")
        await pilot.press("escape")
        assert app.settings["Notation"] == "Rational"


async def test_a_click_on_a_word_that_is_no_field_does_nothing(app):
    """`Declare Variable Interval` prints the variable between its two bounds."""
    async with app.run_test() as pilot:
        # Declare Variable w, Integer, Interval: the bounds, with `w` between.
        await pilot.press("d", "v", *"w", "enter", "i", "i")
        line = band(app)[0]
        assert " w " in line
        active = app.editor.field.setting
        await pilot.pause()
        await pilot.click(app.query_one(band_id(app)), offset=(line.index(" w ") + 1, 0))
        assert app.editor.field.setting == active
        await pilot.press("escape")


async def test_a_click_on_an_expression_fills_a_field_that_names_one(app):
    """Remove and Unremove let the highlight pick the expression, mouse included."""
    async with app.run_test() as pilot:
        for text in ("x", "y", "z"):
            await author(pilot, text)
        await pilot.press("r")
        assert band(app)[0] == " REMOVE: Start: 3      End: 3"
        await click_on(pilot, app, "x")
        assert band(app)[0] == " REMOVE: Start: 1      End: 3"
        await pilot.press("enter")
        assert entries(app) == []


async def test_a_click_on_an_expression_does_nothing_under_another_dialog(app):
    """A dialog whose fields name no expression takes no notice of the highlight."""
    async with app.run_test() as pilot:
        await author(pilot, "x")
        await author(pilot, "y")
        await pilot.press("o", "n")
        await click_on(pilot, app, "x")
        assert app.session.selected == 1
        await pilot.press("escape")


# -- windows -------------------------------------------------------------------


async def test_a_click_in_another_window_makes_it_active_and_selects_there(app):
    async with app.run_test(size=(80, 25)) as pilot:
        await author(pilot, "x")
        await author(pilot, "y")
        await pilot.press("w", "s", "v", "enter")
        assert app.windows.number == 1
        await click_on(pilot, app, "x", number=2)
        assert app.windows.number == 2
        assert app.session.selected == 0


async def test_a_click_in_another_window_is_refused_under_a_submenu(app):
    """Switching windows out from under a half-answered question is not offered."""
    async with app.run_test(size=(80, 25)) as pilot:
        await author(pilot, "x")
        await pilot.press("w", "s", "v", "enter")
        await pilot.press("w")
        await click_on(pilot, app, "x", number=2)
        assert app.windows.number == 1


# -- the wheel -----------------------------------------------------------------


async def test_the_wheel_scrolls_the_pane_and_leaves_the_highlight(app):
    async with app.run_test(size=(80, 25)) as pilot:
        for number in range(30):
            await author(pilot, f"x{number}")
        pane = app.work_area
        was = pane.scroll_offset.y
        assert was > 0
        for _ in range(3):
            await wheel(pilot, app, pane, -1, offset=(2, 2))
        assert pane.scroll_offset.y < was
        assert app.session.selected == 29


async def test_the_wheel_with_shift_scrolls_a_wide_render_sideways(app):
    async with app.run_test() as pilot:
        await author(pilot, "+".join("abcdefghijklmnopqrstuvwxyz"))
        assert app.work_area.shift == 0
        await wheel(pilot, app, app.work_area, 1, offset=(2, 0), shift=True)
        assert app.work_area.shift > 0
        assert app.session.route == ()
        await wheel(pilot, app, app.work_area, -1, offset=(2, 0), shift=True)
        assert app.work_area.shift == 0


async def test_a_sideways_wheel_scrolls_a_wide_render_sideways(app):
    async with app.run_test() as pilot:
        await author(pilot, "+".join("abcdefghijklmnopqrstuvwxyz"))
        await wheel(pilot, app, app.work_area, 1, offset=(2, 0), sideways=True)
        assert app.work_area.shift > 0
        await wheel(pilot, app, app.work_area, -1, offset=(2, 0), sideways=True)
        assert app.work_area.shift == 0


async def test_the_wheel_turns_the_pages_of_a_help_subject(app):
    async with app.run_test() as pilot:
        await pilot.press("h", "a")
        assert app.helping.pages > 1
        assert app.helping.page == 0
        await wheel(pilot, app, app.work_area, 1, offset=(2, 2))
        assert app.helping.page == 1
        await wheel(pilot, app, app.work_area, -1, offset=(2, 2))
        assert app.helping.page == 0


async def test_the_wheel_turns_no_page_on_the_subject_menu(app):
    """There are no pages to turn there, as there is nothing for Down to turn."""
    async with app.run_test() as pilot:
        await pilot.press("h")
        page = work_area(app)
        await wheel(pilot, app, app.work_area, 1, offset=(2, 2))
        assert work_area(app) == page


# -- the list of names a file prompt opens -------------------------------------


async def test_a_click_takes_a_name_off_the_list_without_reading_it(app, tmp_path):
    (tmp_path / "one.mth").write_text("x\n")
    (tmp_path / "two.mth").write_text("y\n")
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        await pilot.press(*f"{tmp_path}/", "tab")
        assert completions(app) == ["one.mth", "two.mth"]
        listing = app.query_one("#completions")
        # Row 1 of the list, inside its border, is the second name.
        await pilot.pause()
        await pilot.click(listing, offset=(2, 2))
        assert prompt(app)[1] == f"{tmp_path}/two.mth"
        # The list is put away and nothing has been read: Enter is still owed.
        assert completions(app) is None
        assert entries(app) == []
        await pilot.press("enter")
        assert entries(app) == ["y"]


async def test_the_wheel_walks_the_list_of_names(app, tmp_path):
    for name in ("one", "two", "three"):
        (tmp_path / f"{name}.mth").write_text("x\n")
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        await pilot.press(*f"{tmp_path}/", "tab")
        names = completions(app)
        assert len(names) == 3
        listing = app.query_one("#completions")
        # One notch is Down and one is Up, down to where they start from.
        await wheel(pilot, app, listing, 1, offset=(2, 2))
        assert chosen(app) == names[0]
        await wheel(pilot, app, listing, 1, offset=(2, 2))
        assert chosen(app) == names[1]
        await wheel(pilot, app, listing, -1, offset=(2, 2))
        assert chosen(app) == names[0]
        # And the name it points at is on the line, as every way of walking it
        # leaves it.
        assert prompt(app)[1].endswith(names[0])


# -- a demonstration -----------------------------------------------------------


async def test_a_click_steps_a_demonstration(app, tmp_path):
    """Any key steps one, and a click is one more way of saying go on."""
    path = tmp_path / "show.dmo"
    path.write_text("; adds two numbers\n2 + 3\n\n; and a symbolic one\n(x+1)^2\n")
    async with app.run_test() as pilot:
        await pilot.press("t", "d", *str(path), "enter")
        assert band(app) == [" adds two numbers"]
        await pilot.click(content(app), offset=(2, 0))
        assert band(app) == [" and a symbolic one"]
        assert entries(app) == ["2+3", "5", "(x+1)^2", "(x + 1)^2"]


# -- copying -------------------------------------------------------------------


async def test_swept_text_is_what_ctrl_c_copies_while_a_sweep_stands(app):
    """And the highlighted expression again once the sweep has been cleared."""
    async with app.run_test() as pilot:
        await author(pilot, "x+1")
        await pilot.pause()
        pane = content(app)
        await pilot.mouse_down(pane, offset=(0, 0))
        await pilot.hover(pane, offset=(5, 0))
        await pilot.mouse_up(pane, offset=(5, 0))
        assert app.screen.get_selected_text() == "#1:  x"
        await pilot.press("ctrl+c")
        assert app.clipboard == "#1:  x"
        assert message(app) == COPIED_TEXT
        # The sweep is gone with the copy, so the key means the expression again.
        await pilot.press("ctrl+c")
        assert app.clipboard == "x+1"
        assert message(app) == COPIED


async def test_a_click_clears_a_sweep(app):
    async with app.run_test() as pilot:
        await author(pilot, "x+1")
        await pilot.pause()
        pane = content(app)
        await pilot.mouse_down(pane, offset=(0, 0))
        await pilot.hover(pane, offset=(5, 0))
        await pilot.mouse_up(pane, offset=(5, 0))
        assert app.screen.get_selected_text()
        await click_on(pilot, app, "x")
        assert not app.screen.get_selected_text()


async def test_repeated_clicks_go_in_rather_than_sweeping_the_pane(app):
    """A double click is Textual's "select all the text"; here it is one level in."""
    async with app.run_test() as pilot:
        await author(pilot, "(x+1)/y")
        line = work_area(app)[0]
        await pilot.pause()
        await pilot.click(content(app), offset=(line.index("x"), 0), times=2)
        assert not app.screen.get_selected_text()
        assert highlighted_expression(app) == "x"


# -- while a computation runs --------------------------------------------------


async def test_a_click_says_nothing_while_a_computation_runs():
    """It is modal down to the last key, and the mouse is no way round that."""
    runner = Blocking()
    app = RederiveApp(Session(runner=runner))
    async with app.run_test() as pilot:
        await author(pilot, "1+1")
        await pilot.press("s", "enter")
        assert await settle(pilot, lambda: app.mode == MODE_COMPUTE)
        await click_word(pilot, app, "Author")
        await click_on(pilot, app, "1")
        assert app.mode == MODE_COMPUTE
        runner.release()
        assert await settle(pilot, lambda: app.mode != MODE_COMPUTE)
        assert entries(app) == ["1+1", "2"]
