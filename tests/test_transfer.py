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
from rederive.syntax import LANGUAGES
from rederive.ui import menu as menus
from rederive.ui.app import RederiveApp
from rederive.ui.menu import mnemonic


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


# -- clearing ----------------------------------------------------------------


def test_clearing_expressions_starts_the_numbering_again(session):
    session.author("a")
    session.author("b")
    session.clear_expressions()
    assert texts(session) == [] and session.selected is None
    session.author("c")
    assert [entry.number for entry in session.entries] == [1]


def test_clearing_expressions_leaves_what_they_defined(session):
    """Four commands rather than degrees of one: the manual is explicit."""
    session.author("v := 7")
    session.clear_expressions()
    session.author("v + 1")
    assert session.simplify("#1").text == "8"


def test_clearing_variables_leaves_the_expressions(session):
    session.author("v := 7")
    session.author("v + 1")
    session.clear_variables()
    assert texts(session) == ["v := 7", "v + 1"]
    assert session.simplify("#2").text == "v + 1"


def test_clearing_functions_takes_the_body_away(session):
    """The name is left undefined, so a call on it is stuck rather than nine."""
    session.author("F(u) := u^2")
    session.author("F(3)")
    assert session.simplify("#2").text == "9"
    session.clear_functions()
    session.author("F(3)")
    assert session.simplify("#4").text == "F(3)"


def test_clearing_all_is_the_other_three_at_once(session):
    session.author("v := 7")
    session.author("F(u) := u^2")
    session.clear_all()
    assert texts(session) == []
    session.author("v + F(3)")
    assert session.simplify("#1").text == "v + F(3)"


# -- loading a utility file --------------------------------------------------


def test_a_utility_file_defines_without_being_displayed(session, file):
    file.write_text("CUBE(t) := t^3\n\nk := 11\n")
    assert session.load_utility(file) == 0
    assert texts(session) == []
    session.author("CUBE(k)")
    assert session.simplify("#1").text == "1331"


def test_a_utility_file_does_not_become_the_session_file(session, file):
    """It is a library, not the worksheet, so the status line goes on naming
    whatever worksheet is open."""
    session.author("x")
    session.save(file)
    other = file.with_name("lib.mth")
    other.write_text("CUBE(t) := t^3\n")
    session.load_utility(other)
    assert session.file == file


def test_a_utility_line_that_does_not_parse_is_counted(session, file):
    file.write_text("CUBE(t) := t^3\n\ny +\n")
    assert session.load_utility(file) == 1


# -- loading a data file -----------------------------------------------------


def test_each_block_of_a_data_file_is_one_matrix(session, file):
    file.write_text("1 2 3\n4 5 6\n\n7 8\n9 10\n")
    assert session.load_data(file) == 0
    assert texts(session) == ["[[1,2,3],[4,5,6]]", "[[7,8],[9,10]]"]


def test_a_block_of_one_line_is_a_vector(session, file):
    """Which is what the original makes of it, rather than a one-row matrix."""
    file.write_text("1 2 3\n")
    session.load_data(file)
    assert texts(session) == ["[1,2,3]"]


def test_a_data_file_appends_to_what_is_already_there(session, file):
    file.write_text("1 2\n")
    session.author("marker")
    session.load_data(file)
    assert texts(session) == ["marker", "[1,2]"]


def test_numbers_may_be_parted_by_spaces_or_commas_or_both(session, file):
    file.write_text("1, 2 3 ,4\n")
    session.load_data(file)
    assert texts(session) == ["[1,2,3,4]"]


def test_an_exponent_becomes_the_power_it_means(session, file):
    """Derive's expression syntax has no exponent notation, so a data file's
    `-2.325E-7` has to be written as the power it stands for."""
    file.write_text("-2.325E-7 1.5D3 4.5\n")
    session.load_data(file)
    assert texts(session) == ["[-2.325*10^-7,1.5*10^3,4.5]"]


def test_a_block_that_is_not_numbers_is_left_out_and_counted(session, file):
    file.write_text("1 2\n\nx y\n\n3 4\n")
    assert session.load_data(file) == 1
    assert texts(session) == ["[1,2]", "[3,4]"]


# -- the state file ----------------------------------------------------------


def test_a_state_file_is_the_settings_as_assignments(session, file):
    session.settings.apply({"Notation": "Mixed", "PrecisionDigits": 12})
    session.save_state(file)
    lines = file.read_text().splitlines()
    assert "Notation := Mixed" in lines
    assert "PrecisionDigits := 12" in lines


def test_a_state_file_reads_back_as_the_settings_that_wrote_it(session, file):
    session.settings.apply({"Notation": "Mixed", "TimesOperator": "Asterisk"})
    session.save_state(file)
    other = Session()
    assert other.load_state(file) == 0
    assert other.settings["Notation"] == "Mixed"
    assert other.settings["TimesOperator"] == "Asterisk"


def test_a_state_file_holds_decimal_numerals_whatever_the_radix(session, file):
    """It is read back by a session whose radix is whatever it is, so a base
    the file itself selects cannot be the base the file is read in."""
    session.settings.apply({"OutputBase": 16, "SaveLength": 60})
    session.save_state(file)
    assert "SaveLength := 60" in file.read_text().splitlines()


def test_a_state_file_leaves_the_history_and_the_file_alone(session, file):
    session.author("x")
    session.save(file)
    session.save_state(file.with_name("state.ini"))
    assert texts(session) == ["x"] and session.file == file


def test_a_line_of_a_state_file_that_sets_nothing_is_counted(session, file):
    file.write_text("Notation := Mixed\nNoSuchThing := 3\nNotation := Nonsense\n")
    assert session.load_state(file) == 2
    assert session.settings["Notation"] == "Mixed"


# -- the menus ---------------------------------------------------------------


async def test_transfer_lists_the_commands_the_original_lists(app):
    """Word for word, but for Print, whose four commands are a printer driver.

    Derive lists `Load Save Merge Clear Demo Print`. Print's own menu is printer
    type, paper size and PCL font strings, and paper is a non-goal, so the word
    goes - as `Display` and `Execute` go from the Options menu.
    """
    async with app.run_test() as pilot:
        await pilot.press("t")
        assert band(app) == [" TRANSFER: Load Save Merge Clear Demo"]
        assert highlighted(app) == "Load"
        await pilot.press("l")
        assert band(app) == [" TRANSFER LOAD: Derive State daTa Utility"]
        await pilot.press("escape", "c")
        assert band(app) == [" TRANSFER CLEAR: All Expressions Functions Variables"]


async def test_the_save_targets_are_languages_of_this_century(app):
    """The original's are Basic, C, Fortran and Pascal.

    The command is the same command and only the list of targets is dated, so
    it is dated forward. Every word keeps a mnemonic letter of its own, which
    is what the menu needs of it.
    """
    async with app.run_test() as pilot:
        await pilot.press("t", "s")
        assert band(app) == [" TRANSFER SAVE: Derive C Python Rust Julia Options State"]
        letters = [mnemonic(word) for word in menus.TRANSFER_SAVE.words]
        assert len(set(letters)) == len(letters)


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


async def test_ctrl_enter_simplifies_every_expression_it_reads(app, file):
    file.write_text("2+3\n\n4 5\n")
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d", *str(file))
        await pilot.press("ctrl+j")
        assert entries(app) == ["2+3", "4 5", "5", "20"]
        assert message(app).startswith("Compute time:")


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


async def test_loading_a_utility_file_shows_nothing_and_defines_everything(app, file):
    file.write_text("CUBE(t) := t^3\n")
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "u")
        assert prompt(app) == ("TRANSFER LOAD UTILITY file:", "")
        await pilot.press(*str(file), "enter")
        assert entries(app) == []
        await pilot.press("a", *"CUBE(2)", "enter")
        await pilot.press("s", "enter")
        assert entries(app)[-1] == "8"


async def test_loading_a_data_file_puts_its_matrices_on_screen(app, tmp_path):
    numbers = tmp_path / "grid.dat"
    numbers.write_text("1 2\n3 4\n")
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "t")
        assert prompt(app) == ("TRANSFER LOAD DATA file:", "")
        await pilot.press(*str(numbers), "enter")
        assert entries(app) == ["[[1,2],[3,4]]"]


async def test_a_name_with_no_extension_gets_the_one_its_command_reads(app, tmp_path):
    """As the original supplied MTH, DAT or DMO by which command was asking."""
    (tmp_path / "grid.dat").write_text("1 2\n")
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "t", *str(tmp_path / "grid"), "enter")
        assert entries(app) == ["[1,2]"]


# -- saving source code ------------------------------------------------------


async def test_saving_a_language_writes_it_under_that_language_suffix(app, tmp_path):
    async with app.run_test() as pilot:
        await pilot.press("a", *"SQRT(x)+1", "enter")
        await pilot.press("t", "s", "p")
        assert prompt(app) == ("TRANSFER SAVE PYTHON file:", "")
        await pilot.press(*str(tmp_path / "work"), "enter")
        assert (tmp_path / "work.py").read_text() == "math.sqrt(x) + 1\n"
        assert band(app)[0].startswith(" COMMAND:")


@pytest.mark.parametrize(
    "letter, name, expected",
    [
        ("c", "work.c", "pow(x, 2)\n"),
        ("p", "work.py", "x ** 2\n"),
        ("r", "work.rs", "x.powi(2)\n"),
        ("j", "work.jl", "x ^ 2\n"),
    ],
)
async def test_each_language_is_reached_by_its_own_letter(
    app, tmp_path, letter, name, expected
):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x^2", "enter")
        await pilot.press("t", "s", letter, *str(tmp_path / "work"), "enter")
        assert (tmp_path / name).read_text() == expected


async def test_an_annotation_goes_behind_the_target_comment_marker(app, tmp_path):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x^2", "enter")
        await pilot.press("s", "enter")
        await pilot.press("t", "s", "j", *str(tmp_path / "work"), "enter")
        assert (tmp_path / "work.jl").read_text() == "x ^ 2\n\n# Simp(#1)\nx ^ 2\n"


async def test_a_language_save_writes_the_block_the_range_option_asked_for(
    app, tmp_path
):
    async with app.run_test() as pilot:
        for text in ("a", "b", "c"):
            await pilot.press("a", *text, "enter")
        await pilot.press("t", "s", "o", "s", "enter")
        await pilot.press("c")
        assert band(app) == [" TRANSFER SAVE DERIVE: Start: 1      End: 3"]
        await pilot.press("2", "tab", "3", "enter")
        await pilot.press(*str(tmp_path / "work"), "enter")
        assert (tmp_path / "work.c").read_text() == "b\n\nc\n"


def test_a_language_save_does_not_break_long_lines(session, tmp_path):
    """The line length is a math file's business; none of the four minds one."""
    session.author("(a+b)^3 (c+d)^3 (e+f)^3")
    session.settings.apply({"SaveLength": 20})
    julia = next(l for l in LANGUAGES if l.word == "Julia")
    session.save_source(tmp_path / "work.jl", julia)
    assert (tmp_path / "work.jl").read_text().splitlines() == [
        "(a + b) ^ 3 * (c + d) ^ 3 * (e + f) ^ 3"
    ]


# -- the state file through the app ------------------------------------------


async def test_the_state_commands_offer_a_file_of_their_own(app, tmp_path):
    """The worksheet is not a settings file and neither name suits the other."""
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter")
        await pilot.press("t", "s", "d", *str(tmp_path / "work"), "enter")
        await pilot.press("t", "s", "s")
        assert prompt(app) == ("TRANSFER SAVE STATE file:", "rederive.ini")


async def test_saving_and_loading_the_state_carries_the_settings_over(app, tmp_path):
    async with app.run_test() as pilot:
        await pilot.press("o", "n", "m", "enter")
        assert app.settings["Notation"] == "Mixed"
        await pilot.press("t", "s", "s", *str(tmp_path / "state"), "enter")
        assert (tmp_path / "state.ini").exists()
        await pilot.press("o", "n", "r", "enter")
        assert app.settings["Notation"] == "Rational"
        await pilot.press("t", "l", "s")
        # The file just written is what the load offers back.
        assert prompt(app) == ("TRANSFER LOAD STATE file:", str(tmp_path / "state.ini"))
        await pilot.press("enter")
        assert app.settings["Notation"] == "Mixed"


async def test_loading_a_state_leaves_the_history_alone(app, tmp_path):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter")
        await pilot.press("t", "s", "s", *str(tmp_path / "state"), "enter")
        await pilot.press("t", "l", "s", "enter")
        assert entries(app) == ["x"]
        assert message(app) == "Enter option"


# -- clearing through the app ------------------------------------------------


async def test_clearing_expressions_asks_before_it_throws_them_away(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter")
        await pilot.press("t", "c", "e")
        assert message(app) == "Abandon expressions (Y/N)?"
        assert highlighted(app) is None
        await pilot.press("n")
        assert entries(app) == ["x"]
        # Refused, so the menu it was picked from is where it leaves you.
        assert band(app)[0].startswith(" TRANSFER CLEAR:")
        await pilot.press("e", "y")
        assert entries(app) == []
        assert band(app)[0].startswith(" COMMAND:")


async def test_clearing_all_asks_the_same_question(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter")
        await pilot.press("t", "c", "a")
        assert message(app) == "Abandon expressions (Y/N)?"
        await pilot.press("y")
        assert entries(app) == []


async def test_a_history_with_nothing_in_it_is_nothing_to_ask_about(app):
    async with app.run_test() as pilot:
        await pilot.press("t", "c", "a")
        assert message(app) == "Enter option"
        assert band(app)[0].startswith(" COMMAND:")


async def test_clearing_variables_and_functions_asks_nothing(app):
    """Nothing on screen goes, so there is nothing to lose by mistake."""
    async with app.run_test() as pilot:
        await pilot.press("a", *"v:=7", "enter")
        await pilot.press("t", "c", "v")
        assert entries(app) == ["v:=7"]
        assert message(app) == "Enter option"
        await pilot.press("a", *"v+1", "enter")
        await pilot.press("s", "enter")
        assert entries(app)[-1] == "v + 1"


# -- the demonstration -------------------------------------------------------


@pytest.fixture
def demo(tmp_path):
    path = tmp_path / "show.dmo"
    path.write_text("; adds two numbers\n2 + 3\n\n; and a symbolic one\n(x+1)^2\n")
    return path


async def test_a_demonstration_authors_and_simplifies_a_step_at_a_time(app, demo):
    async with app.run_test() as pilot:
        await pilot.press("t", "d")
        assert prompt(app) == ("TRANSFER DEMO file:", "")
        assert message(app) == "Enter filename (press F1 for list)"
        await pilot.press(*str(demo), "enter")
        # The comment takes the band the menu was on, and it waits there.
        assert band(app) == [" adds two numbers"]
        assert message(app) == "Press any key to continue"
        assert entries(app) == ["2 + 3", "5"]
        await pilot.press("space")
        assert band(app) == [" and a symbolic one"]
        assert entries(app) == ["2 + 3", "5", "(x+1)^2", "(x + 1)^2"]
        # The last step done, the command is over.
        await pilot.press("space")
        assert band(app)[0].startswith(" COMMAND:")
        assert message(app) == "Enter option"


async def test_escape_suspends_a_demonstration_where_it_stands(app, demo):
    async with app.run_test() as pilot:
        await pilot.press("t", "d", *str(demo), "enter")
        await pilot.press("escape")
        assert band(app)[0].startswith(" COMMAND:")
        # Free to do anything, and naming the same file picks it up again.
        await pilot.press("a", *"z", "enter")
        await pilot.press("t", "d", *str(demo), "enter")
        assert band(app) == [" and a symbolic one"]
        assert entries(app) == ["2 + 3", "5", "z", "(x+1)^2", "(x + 1)^2"]


async def test_a_demonstration_that_has_run_out_starts_over(app, demo):
    async with app.run_test() as pilot:
        await pilot.press("t", "d", *str(demo), "enter")
        await pilot.press("space", "space")
        await pilot.press("t", "d", *str(demo), "enter")
        assert band(app) == [" adds two numbers"]


async def test_a_demonstration_file_that_is_nothing_leaves_the_line_up(app, tmp_path):
    async with app.run_test() as pilot:
        await pilot.press("t", "d", *str(tmp_path / "nothing.dmo"), "enter")
        assert message(app) == "File not found"
        assert prompt(app)[0] == "TRANSFER DEMO file:"


def test_a_demonstration_is_its_comments_and_the_lines_under_them(tmp_path):
    path = tmp_path / "show.dmo"
    path.write_text("; one\na+~\nb\n\nno comment here\n\n; two\nc\n")
    assert worksheet.demonstration(path) == (
        ("one", "a+b"),
        ("", "no comment here"),
        ("two", "c"),
    )
