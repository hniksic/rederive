"""The help system: the document, and the two-level reader over it."""

import pytest
from screen import band, highlighted, highlighted_rows, message, prompt, work_area

from rederive.model import help as helps
from rederive.model.settings import Settings
from rederive.ui.app import RederiveApp


@pytest.fixture
def app():
    return RederiveApp()


def title(app):
    """The first row of the work area, which is a help page's title."""
    return work_area(app)[0].strip()


# -- the document ----------------------------------------------------------


def test_every_subject_has_a_letter_of_its_own():
    letters = [helps.letter(topic.entry) for topic in helps.TOPICS]
    assert len(set(letters)) == len(letters)
    assert helps.letter(helps.RETURN) not in letters


def test_a_subject_word_matches_the_letter_its_menu_entry_carries():
    for topic in helps.TOPICS:
        assert helps.letter(topic.word) == helps.letter(topic.entry)


def test_every_page_carries_the_title_and_fits_the_pane():
    topic = helps.BY_WORD["Editing"]
    for page in helps.pages(topic, Settings(), 20, 80):
        assert page[0].strip() == topic.title
        assert page[1] == ""
        assert len(page) <= 20
        assert max(len(line) for line in page) <= 80


def test_a_narrower_pane_wraps_and_takes_more_pages():
    topic = helps.BY_WORD["Basics"]
    wide = helps.pages(topic, Settings(), 20, 80)
    narrow = helps.pages(topic, Settings(), 20, 44)
    assert max(len(line) for page in narrow for line in page) <= 44
    assert len(narrow) > len(wide)


def test_a_wrapped_line_keeps_its_own_indent():
    rows = helps._wrapped(("  F1    " + "word " * 30,), 40)
    assert len(rows) > 1
    assert all(row.startswith("  ") for row in rows[1:])


def test_the_menu_page_gives_up_its_blank_lines_before_its_items():
    items = len(helps.entries())
    assert helps.menu_page(items + 4, 80).count("") == 2
    assert helps.menu_page(items + 2, 80).count("") == 0
    assert len(helps.menu_page(items + 1, 80)) == items + 1


def _state(settings):
    pages = helps.pages(helps.BY_WORD["State"], settings, 20, 80)
    return "\n".join(line for page in pages for line in page)


def test_the_state_subject_reads_the_settings_as_they_stand():
    settings = Settings()
    assert "Mode:  Exact" in _state(settings)
    settings.apply({"Precision": "Approximate"})
    assert "Mode:  Approximate" in _state(settings)


def test_the_state_subject_lays_out_in_one_column_where_two_would_not_fit():
    settings = Settings()
    wide = helps.pages(helps.BY_WORD["State"], settings, 20, 80)
    narrow = helps.pages(helps.BY_WORD["State"], settings, 20, 40)
    # Two settings to a row where there is room for two columns, one where
    # there is not.
    assert any(line.count(":") == 2 for page in wide for line in page)
    assert all(line.count(":") <= 1 for page in narrow for line in page)
    assert len(narrow) > len(wide)


# -- the reader ------------------------------------------------------------


async def test_help_opens_over_the_worksheet(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x+1", "enter")
        assert work_area(app) == ["#1:  x + 1"]
        await pilot.press("h")
        assert band(app) == [
            " HELP: Basics Editing Functions Algebra Plot State Resume"
        ]
        assert highlighted(app) == "Basics"
        assert message(app) == "Enter option"
        assert title(app) == helps.MENU_TITLE
        assert helps.INSTRUCTION in "\n".join(work_area(app))


async def test_a_letter_opens_the_subject_it_names(app):
    async with app.run_test() as pilot:
        await pilot.press("h", "e")
        assert band(app) == [" HELP EDITING: Next Previous Resume"]
        assert highlighted(app) == "Next"
        assert title(app) == helps.BY_WORD["Editing"].title


async def test_next_and_previous_turn_the_pages(app):
    async with app.run_test() as pilot:
        await pilot.press("h", "e")
        first = work_area(app)
        await pilot.press("n")
        assert work_area(app) != first
        await pilot.press("p")
        assert work_area(app) == first
        # There is nothing before the first page, and asking stays put.
        await pilot.press("p")
        assert work_area(app) == first


async def test_the_history_keys_turn_pages_too(app):
    async with app.run_test() as pilot:
        await pilot.press("h", "e")
        first = work_area(app)
        await pilot.press("down")
        second = work_area(app)
        assert second != first
        await pilot.press("up")
        assert work_area(app) == first
        await pilot.press("pagedown")
        assert work_area(app) == second


async def test_past_the_last_page_is_back_to_the_subject_menu(app):
    async with app.run_test() as pilot:
        await pilot.press("h", "b")
        for _ in range(len(helps.pages(helps.BY_WORD["Basics"], app.settings, 18, 80))):
            await pilot.press("n")
        assert title(app) == helps.MENU_TITLE
        assert app.helping is not None and app.helping.topic is None


async def test_resume_climbs_one_level_at_a_time(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x+1", "enter")
        await pilot.press("h", "f")
        await pilot.press("r")
        assert title(app) == helps.MENU_TITLE
        await pilot.press("r")
        assert app.helping is None
        assert work_area(app) == ["#1:  x + 1"]
        assert highlighted(app) == "Author"


async def test_escape_is_resume(app):
    async with app.run_test() as pilot:
        await pilot.press("h", "a", "escape")
        assert title(app) == helps.MENU_TITLE
        await pilot.press("escape")
        assert app.helping is None
        assert band(app)[0].startswith(" COMMAND:")


async def test_f1_opens_line_editing_and_gives_the_line_back(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x^2+1")
        assert prompt(app) == ("AUTHOR expression:", "x^2+1")
        await pilot.press("f1")
        assert title(app) == helps.BY_WORD[helps.EDITING].title
        assert band(app) == [" HELP EDITING: Next Previous Subjects Resume"]
        # One Resume gives the line back, wherever in the subject it is pressed
        # and with what was typed on it still there.
        await pilot.press("n", "escape")
        assert app.helping is None
        assert prompt(app) == ("AUTHOR expression:", "x^2+1")
        assert message(app) == "Enter expression (press F1 for help)"
        await pilot.press("enter")
        assert [entry.text for entry in app.session.entries] == ["x^2+1"]


async def test_reading_the_editing_subject_through_ends_at_the_line(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x^2+1")
        await pilot.press("f1")
        editing = helps.BY_WORD[helps.EDITING]
        for _ in range(len(helps.pages(editing, app.settings, 18, 80))):
            await pilot.press("n")
        assert app.helping is None
        assert prompt(app) == ("AUTHOR expression:", "x^2+1")


async def test_subjects_opens_the_rest_of_the_document_over_the_line(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x^2+1")
        await pilot.press("f1", "s")
        assert title(app) == helps.MENU_TITLE
        # A subject picked off that menu is read under it and comes back to it,
        # which is where it was opened from.
        await pilot.press("f")
        assert title(app) == helps.BY_WORD["Functions"].title
        await pilot.press("escape")
        assert title(app) == helps.MENU_TITLE
        # And the menu itself stands where the subject stood: over the line.
        await pilot.press("escape")
        assert app.helping is None
        assert prompt(app) == ("AUTHOR expression:", "x^2+1")


async def test_f1_at_the_command_menu_is_still_the_next_window(app):
    async with app.run_test() as pilot:
        await pilot.press("w", "s", "h", "enter")
        assert app.windows.number == 1
        await pilot.press("f1")
        assert app.windows.number == 2
        assert app.helping is None


async def test_help_stands_in_the_window_it_was_asked_from(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x+1", "enter")
        await pilot.press("w", "s", "h", "enter")
        await pilot.press("h", "e")
        assert work_area(app, 1)[0].strip() == helps.BY_WORD["Editing"].title
        assert work_area(app, 2) == ["#1:  x + 1"]


async def test_the_title_row_is_the_one_run_in_inverse_video(app):
    async with app.run_test() as pilot:
        await pilot.press("h", "e")
        assert highlighted_rows(app) == [helps.BY_WORD["Editing"].title]


async def test_the_subject_menu_shows_no_title_in_inverse_video(app):
    async with app.run_test() as pilot:
        await pilot.press("h")
        assert highlighted_rows(app) == []
