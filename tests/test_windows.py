"""The Window command: the tree of windows, where they land, and the commands.

Every geometry rule asserted here was checked against the original, whose
versions agree on all of it: the frame characters, where a divider falls for a
given answer, what answers are taken, and what each command asks.
"""

import pytest
from screen import (
    band,
    frame,
    highlighted,
    message,
    styled,
    window_type,
    work_area,
)

from rederive.model import windows
from rederive.model.session import Session
from rederive.model.windows import Windows
from rederive.ui.app import RederiveApp
from rederive.ui.widgets import Panes

#: The screen the original ran on, and the one every dump here was taken from.
SIZE = (80, 25)

#: What the work area is then worth: twenty rows above the rule.
WORK, COLUMNS = 20, 80


@pytest.fixture
def app():
    return RederiveApp()


async def author(pilot, text):
    await pilot.press("a")
    await pilot.press(*text)
    await pilot.press("enter")


async def resized(pilot, app, width, height):
    """Resize the terminal, and wait for the new size to reach the panes."""
    await pilot.resize_terminal(width, height)
    for _ in range(20):
        if app.query_one(Panes).size.width == width:
            return
        await pilot.pause()
    raise AssertionError("the resize never reached the panes")


def rows(windows_, height=WORK, width=COLUMNS):
    """Each window's interior, in numbering order."""
    areas = windows_.areas(height, width)
    return [
        windows.interior(areas[window], windows_.framed) for window in windows_.windows
    ]


# -- the tree -----------------------------------------------------------------


def test_a_new_session_is_one_algebra_window():
    tree = Windows(Session())
    assert len(tree.windows) == 1
    assert tree.number == 1
    assert tree.kind == windows.ALGEBRA
    assert not tree.framed


def test_splitting_numbers_the_new_window_after_the_active_one():
    tree = Windows(Session())
    made = tree.split(False, 10, Session())
    assert tree.windows == [tree.active, made]
    # The window the split was issued from stays active, so an expression
    # authored next lands where the user was looking.
    assert tree.number == 1


def test_numbers_are_a_walk_of_the_tree_and_not_an_order_of_creation():
    tree = Windows(Session())
    right = tree.split(True, 40, Session())
    # Splitting the left column puts a window between the two that exist.
    below = tree.split(False, 10, Session())
    assert tree.windows == [tree.active, below, right]


def test_closing_a_window_gives_its_space_back_and_renumbers():
    tree = Windows(Session())
    middle = tree.split(False, 7, Session())
    tree.active = middle
    last = middle_split = tree.split(False, 7, Session())
    assert len(tree.windows) == 3
    tree.close(middle)
    assert tree.windows == [tree.windows[0], last]
    assert middle_split is last
    # The closed window was active, so what took its space is.
    assert tree.active is last


def test_closing_the_only_window_of_a_stack_takes_the_window_with_it():
    tree = Windows(Session())
    tree.split(False, 10, Session())
    second = tree.windows[1]
    tree.close(second)
    assert tree.windows == [tree.active]
    assert not tree.framed


def test_an_opened_window_shares_the_number_and_flips_under_it():
    tree = Windows(Session())
    first = tree.session
    second = Session()
    tree.open(windows.ALGEBRA, second)
    assert len(tree.windows) == 1
    assert tree.session is second
    assert tree.active.flip()
    assert tree.session is first
    tree.active.flip(-1)
    assert tree.session is second


def test_stepping_wraps_round_both_ends():
    tree = Windows(Session())
    tree.split(False, 7, Session())
    tree.split(False, 4, Session())
    assert tree.number == 1
    tree.step(-1)
    assert tree.number == 3
    tree.step(1)
    assert tree.number == 1


# -- geometry -----------------------------------------------------------------


def test_the_only_window_takes_the_whole_work_area():
    tree = Windows(Session())
    assert rows(tree) == [windows.Rect(0, 0, WORK, COLUMNS)]


def test_a_horizontal_split_puts_the_divider_on_the_line_it_was_given():
    tree = Windows(Session())
    tree.split(False, 10, Session())
    # Rows 1 to 9 and 11 to 19, the divider taking row 10 and the frame rows 0
    # and 20 - which is the original's own screen, cell for cell.
    assert rows(tree) == [
        windows.Rect(1, 1, 9, 78),
        windows.Rect(11, 1, 9, 78),
    ]


def test_a_vertical_split_rounds_the_column_down_onto_an_even_one():
    tree = Windows(Session())
    tree.split(True, 39, Session())
    assert [rect.left for rect in rows(tree)] == [1, 39]
    other = Windows(Session())
    other.split(True, 38, Session())
    assert rows(other) == rows(tree)


def test_a_vertical_split_at_forty_leaves_thirty_nine_and_thirty_eight():
    tree = Windows(Session())
    tree.split(True, 40, Session())
    assert rows(tree) == [
        windows.Rect(1, 1, 19, 39),
        windows.Rect(1, 41, 19, 38),
    ]


def test_a_split_is_measured_from_the_window_it_splits():
    tree = Windows(Session())
    lower = tree.split(False, 10, Session())
    tree.active = lower
    tree.split(False, 2, Session())
    # The lower window's corner is row 10, so its line 2 is row 12.
    assert [rect.top for rect in rows(tree)] == [1, 11, 13]


def test_the_default_splits_the_window_in_half():
    assert windows.split_default(20) == 10
    assert windows.split_default(19) == 10
    assert windows.split_default(80) == 40
    assert windows.split_default(78) == 39
    assert windows.split_default(9) == 5


def test_the_answers_a_split_takes_are_the_original_s():
    assert windows.split_range(False, 20) == (2, 18)
    assert windows.split_range(False, 9) == (2, 7)
    assert windows.split_range(True, 80) == (7, 74)
    assert windows.split_range(True, 38) == (7, 32)


def test_a_window_too_small_to_split_offers_no_answer_at_all():
    low, high = windows.split_range(False, 3)
    assert high < low


# -- the border ---------------------------------------------------------------


def test_one_window_is_drawn_without_a_border():
    tree = Windows(Session())
    drawing = tree.frame(WORK, COLUMNS)
    assert set("".join(drawing.rows[:-1])) == {" "}
    assert drawing.rows[-1] == "═" * COLUMNS


def test_a_horizontal_divider_carries_the_number_and_ends_on_the_frame():
    tree = Windows(Session())
    tree.split(False, 10, Session())
    drawing = tree.frame(WORK, COLUMNS)
    assert drawing.rows[0] == "1" + "═" * 78 + "╕"
    assert drawing.rows[1] == "│" + " " * 78 + "│"
    assert drawing.rows[10] == "2" + "═" * 78 + "╡"
    assert drawing.rows[20] == "╘" + "═" * 78 + "╛"


def test_a_vertical_divider_meets_the_bottom_frame_in_a_tee():
    tree = Windows(Session())
    tree.split(True, 40, Session())
    drawing = tree.frame(WORK, COLUMNS)
    assert drawing.rows[0] == "1" + "═" * 39 + "2" + "═" * 38 + "╕"
    assert drawing.rows[1] == "│" + " " * 39 + "│" + " " * 38 + "│"
    assert drawing.rows[20] == "╘" + "═" * 39 + "╧" + "═" * 38 + "╛"


def test_a_divider_running_into_another_takes_the_piece_with_those_arms():
    tree = Windows(Session())
    right = tree.split(True, 40, Session())
    tree.split(False, 10, Session())
    tree.active = right
    tree.split(False, 10, Session())
    drawing = tree.frame(WORK, COLUMNS)
    # The lower left divider stops against the vertical one, which goes on
    # above and below it; the lower right one starts there.
    assert drawing.rows[10] == "2" + "═" * 39 + "4" + "═" * 38 + "╡"


def test_the_number_of_a_window_past_the_ninth_takes_two_cells():
    tree = Windows(Session())
    for _ in range(9):
        tree.active = tree.split(False, 2, Session())
    drawing = tree.frame(WORK, COLUMNS)
    assert drawing.numbers[9][2] == 2
    assert drawing.rows[drawing.numbers[9][0]].startswith("10═")


# -- the commands -------------------------------------------------------------


async def test_the_window_menu_lists_the_eight_commands_on_one_line(app):
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("w")
        assert band(app) == [
            " WINDOW: Close Designate Flip Goto Next Open Previous Split"
        ]
        assert message(app) == "Enter option"
        assert highlighted(app) == "Close"


async def test_split_asks_which_way_and_then_where(app):
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("w", "s")
        assert band(app) == [" WINDOW SPLIT: Horizontal Vertical"]
        assert highlighted(app) == "Horizontal"
        await pilot.press("h")
        assert band(app) == [" WINDOW SPLIT HORIZONTAL: At line: 10"]
        assert message(app) == "Enter line number"


async def test_a_vertical_split_asks_for_a_column(app):
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("w", "s", "v")
        assert band(app) == [" WINDOW SPLIT VERTICAL: At column: 40"]
        assert message(app) == "Enter column number"


async def test_a_split_copies_the_worksheet_and_the_histories_then_fork(app):
    async with app.run_test(size=SIZE) as pilot:
        await author(pilot, "x")
        await pilot.press("w", "s", "h", "enter")
        assert work_area(app, 1) == ["#1:  x"]
        assert work_area(app, 2) == ["#1:  x"]
        # The window the split was issued from is the one that stays active.
        await author(pilot, "y")
        assert work_area(app, 1) == ["#1:  x", "", "#2:  y"]
        assert work_area(app, 2) == ["#1:  x"]


async def test_the_frame_appears_the_moment_there_are_two_windows(app):
    async with app.run_test(size=SIZE) as pilot:
        assert frame(app)[0].strip() == ""
        await pilot.press("w", "s", "h", "enter")
        drawn = frame(app)
        assert drawn[0] == "1" + "═" * 78 + "╕"
        assert drawn[10] == "2" + "═" * 78 + "╡"
        assert drawn[20] == "╘" + "═" * 78 + "╛"


async def test_a_line_the_window_is_too_small_for_is_refused(app):
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("w", "s", "h")
        await pilot.press("1", "9")
        await pilot.press("enter")
        # The field stays up with what was typed still on it, as the original
        # leaves it: no message, and nothing split.
        assert band(app) == [" WINDOW SPLIT HORIZONTAL: At line: 19"]
        assert len(app.windows.windows) == 1


async def test_f1_and_shift_f1_walk_the_windows(app):
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("w", "s", "h", "enter")
        assert app.windows.number == 1
        await pilot.press("f1")
        assert app.windows.number == 2
        await pilot.press("f1")
        assert app.windows.number == 1
        await pilot.press("shift+f1")
        assert app.windows.number == 2


async def test_goto_offers_the_next_window_and_close_the_active_one(app):
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("w", "s", "h", "enter")
        await pilot.press("w", "g")
        assert band(app) == [" WINDOW GOTO: Window: 2"]
        assert message(app) == "Enter window number"
        await pilot.press("enter")
        assert app.windows.number == 2
        await pilot.press("w", "c")
        assert band(app) == [" WINDOW CLOSE: Window: 2"]
        assert message(app) == "Enter window number"


async def test_close_asks_before_it_throws_expressions_away(app):
    async with app.run_test(size=SIZE) as pilot:
        await author(pilot, "x")
        await pilot.press("w", "s", "h", "enter")
        await pilot.press("w", "c", "enter")
        # The number entered stands on the band while the question is asked.
        assert band(app) == [" WINDOW CLOSE: Window: 1"]
        assert message(app) == "Abandon expressions (Y/N)?"
        await pilot.press("n")
        assert len(app.windows.windows) == 2
        await pilot.press("w", "c", "enter", "y")
        assert len(app.windows.windows) == 1


async def test_close_is_refused_outright_while_there_is_one_window(app):
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("w", "c")
        assert band(app) == [
            " WINDOW: Close Designate Flip Goto Next Open Previous Split"
        ]
        assert message(app) == "Enter option"


async def test_designate_offers_the_three_types_and_opens_on_this_one(app):
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("w", "d")
        assert band(app) == [" WINDOW DESIGNATE: Type: 2D-plot 3D-plot Algebra"]
        assert message(app) == "Enter window type"
        assert highlighted(app) == "Algebra"


async def test_designating_a_window_makes_it_over_empty(app):
    async with app.run_test(size=SIZE) as pilot:
        await author(pilot, "x")
        await pilot.press("w", "d", "enter")
        assert message(app) == "Abandon expressions (Y/N)?"
        await pilot.press("y")
        assert work_area(app) == []


async def test_a_plot_window_is_offered_and_refused(app):
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("w", "d", "2")
        assert message(app) == "2D-plot: not implemented yet"
        assert app.windows.kind == "Algebra"


async def test_open_overlays_a_window_that_flip_brings_back(app):
    async with app.run_test(size=SIZE) as pilot:
        await author(pilot, "x")
        await pilot.press("w", "o")
        assert band(app) == [" WINDOW OPEN: Type: 2D-plot 3D-plot Algebra"]
        await pilot.press("enter")
        # Still one window, and still no frame: an overlay shares both.
        assert len(app.windows.windows) == 1
        assert work_area(app) == []
        await pilot.press("f2")
        assert work_area(app) == ["#1:  x"]


async def test_the_active_window_is_the_one_whose_number_is_in_inverse_video(app):
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("w", "s", "h", "enter")
        inverse = app.palette.styles["selection"]
        assert styled(app.query_one("#frame"), inverse) == ["1"]
        await pilot.press("f1")
        assert styled(app.query_one("#frame"), inverse) == ["2"]


async def test_a_resized_terminal_lays_the_windows_out_again(app):
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("w", "s", "h", "enter")
        await resized(pilot, app, 60, 20)
        drawn = frame(app)
        # Six rows below the work area: the rule, the three rows the menu
        # takes across sixty columns, the message line and the status line.
        assert len(drawn) == 15
        assert drawn[0] == "1" + "═" * 58 + "╕"
        assert drawn[10] == "2" + "═" * 58 + "╡"
        assert drawn[-1] == "╘" + "═" * 58 + "╛"


async def test_the_status_line_names_the_active_window_s_type(app):
    async with app.run_test(size=SIZE) as pilot:
        assert window_type(app) == "Rederive Algebra"


async def test_quit_asks_once_for_every_window(app):
    async with app.run_test(size=SIZE) as pilot:
        await author(pilot, "x")
        await pilot.press("w", "s", "h", "enter")
        # The active window is cleared, but the copy still holds the history.
        await pilot.press("t", "c", "a", "y")
        assert work_area(app, 1) == []
        await pilot.press("q")
        assert message(app) == "Abandon expressions (Y/N)?"
