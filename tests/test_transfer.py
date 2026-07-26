"""Saving and loading expressions: the file format, and the Transfer commands.

The expected band text, the message lines and the file layout are what the
original puts on the same screens and writes to the same files. How an
expression itself is spelled in a file belongs to `test_writer`. Where Rederive
differs on purpose - it says so when a line of a file will not parse - the test
that pins the difference says why.
"""

import pytest
from screen import (
    annotation,
    band,
    entries,
    highlighted,
    highlighted_expression,
    message,
    prompt,
    text_of,
)

from rederive.model import worksheet
from rederive.model.session import Session
from rederive.ui.app import RederiveApp


@pytest.fixture
def session():
    return Session()


@pytest.fixture
def app():
    return RederiveApp()


@pytest.fixture
def file(tmp_path):
    return tmp_path / "work.mth"


def written(session, path, *block):
    session.save(path, *block)
    return path.read_text(encoding="utf-8")


def annotations(session):
    return [entry.annotation for entry in session.entries]


def texts(session):
    return [entry.text for entry in session.entries]


# -- what a file looks like --------------------------------------------------


def test_a_file_is_an_expression_a_line_with_blanks_between(session, file):
    session.author("x^2 + 1")
    session.author("SIN(x)/2")
    assert written(session, file) == "x^2+1\n\nSIN(x)/2\n"


def test_an_entry_is_written_from_its_tree_and_not_from_its_text(session, file):
    """A file says what the expressions are, not how they were typed.

    `test_writer` is where the spellings themselves are pinned against the
    original; what matters here is that saving goes through them at all.
    """
    session.author("x (x + 1)")
    session.author("2 (8 + 7) / 3^2")
    assert written(session, file) == "x*(x+1)\n\n2*(8+7)/3^2\n"


def test_an_annotation_is_the_comment_above_its_expression(session, file):
    session.author("SIN(x)/2")
    session.simplify("#1")
    assert written(session, file) == "SIN(x)/2\n\n;Simp(#1)\nSIN(x)/2\n"


def test_a_line_the_user_wrote_needs_no_comment_to_say_so(session, file):
    session.author("x")
    assert ";" not in written(session, file)


def test_the_annotation_option_leaves_the_comments_out(session, file):
    session.author("SIN(x)/2")
    session.simplify("#1")
    session.settings.apply({"SaveAnnotation": "Omit"})
    assert written(session, file) == "SIN(x)/2\n\nSIN(x)/2\n"


def test_the_range_option_writes_the_block_it_is_given(session, file):
    for text in ("a", "b", "c", "d"):
        session.author(text)
    assert written(session, file, 2, 3) == "b\n\nc\n"


def test_a_long_line_is_broken_and_continued_with_a_tilde(session, file):
    session.author("(a+b)^3 (c+d)^3 (e+f)^3 (g+h)^3 (i+j)^3")
    session.settings.apply({"SaveLength": 30})
    assert written(session, file) == "(a+b)^3*(c+d)^3*(e+f)^3*(g+h)~\n^3*(i+j)^3\n"


def test_a_continued_line_reads_back_as_the_one_expression(session, file):
    session.author("(a+b)^3 (c+d)^3 (e+f)^3 (g+h)^3 (i+j)^3")
    session.settings.apply({"SaveLength": 30})
    session.save(file)
    other = Session()
    other.load(file)
    assert texts(other) == ["(a+b)^3*(c+d)^3*(e+f)^3*(g+h)^3*(i+j)^3"]


def test_a_name_with_no_extension_is_a_math_file():
    assert worksheet.path_of("work") == worksheet.path_of("work.mth")
    assert str(worksheet.path_of("work")).endswith(".mth")
    # One the user spelled out is left as it is.
    assert str(worksheet.path_of("notes.txt")).endswith("notes.txt")


# -- loading -----------------------------------------------------------------


def test_loading_replaces_the_history_and_numbers_it_again(session, file):
    session.author("a")
    session.author("b")
    session.save(file)
    session.author("c")
    assert session.load(file) == 0
    assert texts(session) == ["a", "b"]
    assert [entry.number for entry in session.entries] == [1, 2]
    assert session.selected == 1 and session.route == ()


def test_an_annotation_comes_back_with_its_expression(session, file):
    session.author("SIN(x)/2")
    session.simplify("#1")
    session.save(file)
    session.load(file)
    assert annotations(session) == ["User", "Simp(#1)"]


def test_what_a_loaded_line_defines_reaches_the_next_command(session, file):
    session.author("f(y) := y^2 - 1")
    session.author("f(3)")
    session.save(file)
    other = Session()
    other.load(file)
    assert other.simplify("#2").text == "8"


def test_a_setting_a_loaded_line_assigns_takes_effect(session, file):
    session.author("Notation := Mixed")
    session.save(file)
    other = Session()
    other.load(file)
    assert other.settings["Notation"] == "Mixed"


def test_blank_lines_separate_nothing(session, file):
    """One expression per line is the rule; a blank line only spaces the file
    out, so a file written without them reads the same."""
    file.write_text("x+1\ny+2\nz+3\n")
    session.load(file)
    assert texts(session) == ["x+1", "y+2", "z+3"]


def test_the_conventions_of_a_file_the_original_wrote_are_read(session, file):
    """CRLF line endings, a Ctrl-Z ending the file, and a tilde continuation."""
    file.write_bytes(b";Mine\r\na+~\r\nb\r\n\r\nc*d\r\n\x1a")
    assert session.load(file) == 0
    assert texts(session) == ["a+b", "c*d"]
    assert annotations(session) == ["Mine", "User"]


def test_a_file_written_in_code_page_437_reads(session, file):
    """Rederive writes UTF-8. The original wrote code page 437, where 0xE9 is
    the capital theta a variable in its own GRAPHICS.MTH is named for."""
    file.write_bytes(b"\xe9n + 1\r\n\x1a")
    assert session.load(file) == 0
    assert texts(session) == ["Θn + 1"]


def test_a_line_that_does_not_parse_is_left_out_and_counted(session, file):
    """The original drops such a line without a word. The rest of the file
    still loads, which is the part that matters; the count is what lets the
    screen say that something was dropped."""
    file.write_text("x^2+1\n\ny +\n\nz\n")
    assert session.load(file) == 1
    assert texts(session) == ["x^2+1", "z"]


def test_a_file_that_is_not_there_leaves_the_history_alone(session, file):
    session.author("x")
    with pytest.raises(FileNotFoundError):
        session.load(file.with_name("nothing.mth"))
    assert texts(session) == ["x"]
    assert session.file is None


def test_the_session_remembers_the_file_it_used(session, file):
    session.author("x")
    session.save(file)
    assert session.file == file


# -- merging -----------------------------------------------------------------


def test_merging_appends_and_carries_the_numbering_on(session, file):
    session.author("a")
    session.author("b")
    session.save(file)
    assert session.merge(file) == 0
    assert texts(session) == ["a", "b", "a", "b"]
    assert [entry.number for entry in session.entries] == [1, 2, 3, 4]


def test_a_merged_annotation_names_the_entry_it_now_means(session, file):
    session.author("SIN(x)/2")
    session.simplify("#1")
    session.save(file)
    session.merge(file)
    # The file says Simp(#1); merged behind two entries, that entry is #3.
    assert annotations(session) == ["User", "Simp(#1)", "User", "Simp(#3)"]


def test_a_merged_reference_still_means_the_expression_it_meant(session, file):
    session.author("p")
    session.author("#1 + 2")
    session.save(file)
    other = Session()
    for text in ("a", "b", "c"):
        other.author(text)
    other.merge(file)
    assert texts(other) == ["a", "b", "c", "p", "#4+2"]
    assert other.simplify("#5").text == "p + 2"


def test_loading_shifts_nothing(session, file):
    """A load starts the numbering again, so every reference already fits."""
    session.author("p")
    session.author("#1 + 2")
    session.save(file)
    session.load(file)
    assert texts(session) == ["p", "#1+2"]


# -- the menus ---------------------------------------------------------------


async def test_transfer_lists_the_commands_the_original_lists(app):
    async with app.run_test() as pilot:
        await pilot.press("t")
        assert band(app) == [" TRANSFER: Load Save Merge Clear Demo Print"]
        assert highlighted(app) == "Load"
        await pilot.press("l")
        assert band(app) == [" TRANSFER LOAD: Derive State daTa Utility"]
        await pilot.press("escape", "s")
        assert band(app) == [
            " TRANSFER SAVE: Derive Basic C Fortran Pascal Options State"
        ]


async def test_a_transfer_command_with_nothing_behind_it_says_so(app):
    async with app.run_test() as pilot:
        await pilot.press("t", "d")
        assert message(app) == "Demo: not implemented yet"


# -- the save options dialog -------------------------------------------------


async def test_the_save_options_are_the_three_the_original_offers(app):
    async with app.run_test() as pilot:
        await pilot.press("t", "s", "o")
        assert band(app) == [
            " TRANSFER SAVE OPTIONS: Range: All Some  Annotation:(Save)Omit  Length: 79"
        ]
        assert message(app) == "Select save range"
        await pilot.press("tab")
        assert message(app) == "Select annotation mode"
        await pilot.press("tab")
        assert message(app) == "Enter line length"


async def test_the_save_options_leave_their_own_menu_up(app):
    """They say how the next save writes its file, so the original leaves you
    where the save is - unlike an Options dialog, which is the whole command."""
    async with app.run_test() as pilot:
        await pilot.press("t", "s", "o", "s", "enter")
        assert app.settings["SaveRange"] == "Some"
        assert band(app)[0].startswith(" TRANSFER SAVE:")
        # And nothing is recorded: these are not the system's settings.
        assert entries(app) == []


# -- saving through the app --------------------------------------------------


async def test_saving_asks_for_a_name_and_writes_the_file(app, file):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x^2+1", "enter")
        await pilot.press("t", "s", "d")
        assert prompt(app) == ("TRANSFER SAVE DERIVE file:", "")
        assert message(app) == "Enter filename"
        await pilot.press(*str(file), "enter")
        assert file.read_text() == "x^2+1\n"
        assert message(app) == "Enter option"
        # The command ran, so the whole path it was reached by is done with.
        assert band(app)[0].startswith(" COMMAND:")


async def test_the_status_line_names_the_file_the_session_is_on(app, file):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter")
        await pilot.press("t", "s", "d", *str(file), "enter")
        status = text_of(app.query_one("#status")).plain
        assert "work.mth" in status
        # Beside the annotation, not in place of it.
        assert annotation(app) == "User"


async def test_the_file_last_used_is_offered_back(app, file):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter")
        await pilot.press("t", "s", "d", *str(file), "enter")
        await pilot.press("t", "s", "d")
        assert prompt(app) == ("TRANSFER SAVE DERIVE file:", str(file))
        # All of it is selected, so a name typed over it replaces it whole.
        await pilot.press("q")
        assert prompt(app)[1] == "q"


async def test_the_history_keys_walk_without_touching_the_name(app, file):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter")
        await pilot.press("a", *"y", "enter")
        await pilot.press("t", "s", "d", *str(file), "enter")
        await pilot.press("t", "s", "d")
        await pilot.press("up")
        # The highlight walks, as it does under any prompt line, but a file
        # name is no label and is left exactly as it was offered.
        assert highlighted_expression(app) == "x"
        assert prompt(app) == ("TRANSFER SAVE DERIVE file:", str(file))


async def test_a_history_with_nothing_in_it_is_not_written(app):
    async with app.run_test() as pilot:
        beeps = []
        app.bell = lambda: beeps.append("beep")
        await pilot.press("t", "s", "d")
        assert beeps == ["beep"]
        assert band(app)[0].startswith(" TRANSFER SAVE:")


async def test_the_some_range_asks_which_block_before_the_name(app, file):
    async with app.run_test() as pilot:
        for text in ("a", "b", "c"):
            await pilot.press("a", *text, "enter")
        await pilot.press("t", "s", "o", "s", "enter")
        await pilot.press("d")
        # The whole history is offered, and the two fields keep their columns.
        assert band(app) == [" TRANSFER SAVE DERIVE: Start: 1      End: 3"]
        assert message(app) == "Enter label number"
        await pilot.press("2", "tab", "3", "enter")
        assert prompt(app)[0] == "TRANSFER SAVE DERIVE file:"
        await pilot.press(*str(file), "enter")
        assert file.read_text() == "b\n\nc\n"


# -- loading and merging through the app -------------------------------------


async def test_loading_replaces_what_is_on_screen(app, file):
    file.write_text("a\n\nb\n")
    async with app.run_test() as pilot:
        await pilot.press("a", *"z", "enter")
        await pilot.press("t", "l", "d")
        assert prompt(app) == ("TRANSFER LOAD DERIVE file:", "")
        assert message(app) == "Enter filename (press F1 for list)"
        await pilot.press(*str(file), "enter")
        assert entries(app) == ["a", "b"]
        assert message(app) == "Enter option"


async def test_merging_adds_to_what_is_on_screen(app, file):
    file.write_text("a\n\nb\n")
    async with app.run_test() as pilot:
        await pilot.press("a", *"z", "enter")
        await pilot.press("t", "m")
        assert prompt(app)[0] == "TRANSFER MERGE file:"
        await pilot.press(*str(file), "enter")
        assert entries(app) == ["z", "a", "b"]


async def test_a_name_that_is_nothing_leaves_the_line_up_to_be_corrected(app, file):
    async with app.run_test() as pilot:
        await pilot.press("a", *"z", "enter")
        await pilot.press("t", "l", "d")
        await pilot.press(*str(file.with_name("nothing.mth")), "enter")
        assert message(app) == "File not found"
        assert prompt(app)[1].endswith("nothing.mth")
        assert entries(app) == ["z"]
        # Esc goes back to the menu the command was picked from, not past it.
        await pilot.press("escape")
        assert band(app)[0].startswith(" TRANSFER LOAD:")


async def test_a_file_with_a_line_that_will_not_parse_says_how_many(app, file):
    file.write_text("x^2+1\n\ny +\n\nz\n")
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d", *str(file), "enter")
        assert entries(app) == ["x^2+1", "z"]
        assert message(app) == "1 expression could not be read"
