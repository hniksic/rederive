"""Saving and loading expressions: the file format, and the Transfer commands.

The expected band text, message lines and file layout match the original. How an
expression is spelled in a file is tested in `test_writer`. Where Rederive differs on
purpose (it reports lines that fail to parse), the test says why.
"""

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from screen import (
    annotation,
    band,
    chosen,
    completions,
    entries,
    highlighted,
    highlighted_expression,
    laid_out,
    listing_title,
    message,
    prompt,
    text_of,
)

from rederive import fetch
from rederive.model import worksheet
from rederive.model.session import Session
from rederive.platform.desktop import DesktopStorage
from rederive.syntax import LANGUAGES
from rederive.ui import menu as menus
from rederive.ui.app import RederiveApp
from rederive.ui.menu import mnemonic


@pytest.fixture(autouse=True)
def here(tmp_path, monkeypatch):
    """Run each test in its temporary directory, so files can be named without a path.

    Tests where the absolute path matters (the file list's title, walking down a tree)
    spell it out.
    """
    monkeypatch.chdir(tmp_path)


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
    """A file records the expressions, not how they were typed.

    The spellings themselves are tested in `test_writer`; this checks that saving uses
    them.
    """
    session.author("x (x + 1)")
    session.author("2 (8 + 7) / 3^2")
    assert written(session, file) == "x*(x+1)\n\n2*(8+7)/3^2\n"


async def test_an_annotation_is_the_comment_above_its_expression(session, file):
    session.author("SIN(x)/2")
    await session.simplify("#1")
    assert written(session, file) == "SIN(x)/2\n\n;Simp(#1)\nSIN(x)/2\n"


def test_a_line_the_user_wrote_needs_no_comment_to_say_so(session, file):
    session.author("x")
    assert ";" not in written(session, file)


async def test_the_annotation_option_leaves_the_comments_out(session, file):
    session.author("SIN(x)/2")
    await session.simplify("#1")
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
    # An explicit extension is kept.
    assert str(worksheet.path_of("notes.txt")).endswith("notes.txt")


# -- completing a half-typed name --------------------------------------------


def test_a_name_completes_to_the_file_it_could_be(tmp_path):
    (tmp_path / "work.mth").touch()
    assert worksheet.completions(f"{tmp_path}/wo") == [f"{tmp_path}/work.mth"]


def test_every_name_on_offer_comes_back_in_order(tmp_path):
    for name in ("beta.mth", "alpha.mth"):
        (tmp_path / name).touch()
    assert worksheet.completions(f"{tmp_path}/") == [
        f"{tmp_path}/alpha.mth",
        f"{tmp_path}/beta.mth",
    ]


def test_only_the_files_the_command_can_use_are_offered(tmp_path):
    (tmp_path / "work.txt").touch()
    (tmp_path / "work.dat").touch()
    assert worksheet.completions(f"{tmp_path}/wo") == []
    offered = worksheet.completions(f"{tmp_path}/wo", worksheet.DATA_SUFFIX)
    assert offered == [f"{tmp_path}/work.dat"]


def test_an_extension_is_matched_however_it_is_cased(tmp_path):
    """The original's shipped files have upper-case names."""
    (tmp_path / "TRIG.MTH").touch()
    assert worksheet.completions(f"{tmp_path}/TR") == [f"{tmp_path}/TRIG.MTH"]


def test_a_directory_completes_with_the_separator_after_it(tmp_path):
    (tmp_path / "utility").mkdir()
    assert worksheet.completions(f"{tmp_path}/uti") == [f"{tmp_path}/utility/"]


def test_a_name_nothing_could_grow_from_completes_to_nothing(tmp_path):
    (tmp_path / "work.mth").touch()
    # A complete name, no matching file, and a missing directory.
    assert worksheet.completions(f"{tmp_path}/work.mth") == []
    assert worksheet.completions(f"{tmp_path}/nothing") == []
    assert worksheet.completions(f"{tmp_path}/gone/wo") == []


def test_a_hidden_name_is_offered_only_to_a_dot(tmp_path):
    (tmp_path / ".hidden.mth").touch()
    assert worksheet.completions(f"{tmp_path}/") == []
    assert worksheet.completions(f"{tmp_path}/.") == [f"{tmp_path}/.hidden.mth"]


def test_what_was_typed_is_kept_as_it_was_typed(tmp_path, monkeypatch):
    """A `~` is used for completion but not expanded in the result."""
    # Set both, since POSIX expands `~` from HOME and Windows from USERPROFILE;
    # otherwise Windows would use the real home directory.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    (tmp_path / "work.mth").touch()
    assert worksheet.completions("~/wo") == ["~/work.mth"]


def test_a_name_typed_out_in_full_is_still_one_of_the_matches(tmp_path):
    """Matches (shown in the list) differ from completions (what Tab can add).

    A fully typed name has no completions, but it still matches, so the list keeps
    showing it after a name is taken from it.
    """
    for name in ("work.mth", "workbook.mth"):
        (tmp_path / name).touch()
    assert worksheet.matches(f"{tmp_path}/work.mth") == [f"{tmp_path}/work.mth"]
    assert worksheet.completions(f"{tmp_path}/work.mth") == []
    assert worksheet.matches(f"{tmp_path}/work") == [
        f"{tmp_path}/work.mth",
        f"{tmp_path}/workbook.mth",
    ]


def test_matches_answers_the_same_questions_completions_does(tmp_path):
    """Apart from the exact name, matches and completions apply the same rules."""
    (tmp_path / "utility").mkdir()
    (tmp_path / "work.txt").touch()
    (tmp_path / ".hidden.mth").touch()
    assert worksheet.matches(f"{tmp_path}/") == [f"{tmp_path}/utility/"]
    assert worksheet.matches(f"{tmp_path}/.") == [f"{tmp_path}/.hidden.mth"]
    assert worksheet.matches(f"{tmp_path}/gone/wo") == []


# -- a name is a file in the working directory and nowhere else ---------------


def test_a_name_means_a_file_of_your_own():
    """Names resolve relative to the working directory."""
    worksheet.path_of("aardvark").write_text("x\n", encoding="utf-8")
    assert worksheet.path_of("aardvark") == worksheet.path_of("aardvark.mth")
    assert not worksheet.path_of("aardvark").is_absolute()
    assert worksheet.matches("") == ["aardvark.mth"]


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


async def test_an_annotation_comes_back_with_its_expression(session, file):
    session.author("SIN(x)/2")
    await session.simplify("#1")
    session.save(file)
    session.load(file)
    assert annotations(session) == ["User", "Simp(#1)"]


async def test_what_a_loaded_line_defines_reaches_the_next_command(session, file):
    session.author("f(y) := y^2 - 1")
    session.author("f(3)")
    session.save(file)
    other = Session()
    other.load(file)
    assert (await other.simplify("#2")).text == "8"


def test_a_setting_a_loaded_line_assigns_takes_effect(session, file):
    session.author("Notation := Mixed")
    session.save(file)
    other = Session()
    other.load(file)
    assert other.settings["Notation"] == "Mixed"


def test_a_file_is_read_in_word_mode(session, file):
    """As in the original, whatever the session's mode: `ab` is one name in a file.

    The session stays in Character mode, so the same text typed afterwards splits.
    """
    file.write_text("ab + y1\n")
    session.load(file)
    session.merge(file)
    assert texts(session) == ["ab+y1", "ab+y1"]
    assert session.author("ab + y1").text == "a*b+y*1"


def test_a_file_that_sets_the_input_mode_reads_the_rest_in_it(session, file):
    """The setting applies from its line on, and is still in force after the load."""
    file.write_text("ab\nInputMode := Character\nab\n")
    session.load(file)
    assert texts(session) == ["ab", "InputMode:=Character", "a*b"]
    session.settings.assign("InputMode", "Character")
    file.write_text("InputMode := Word\n")
    session.load_utility(file)
    assert session.author("ab").text == "ab"


def test_a_parameter_from_a_utility_file_reads_whole_afterwards(session, file):
    """ODE.MTH's functions take `y1` as a parameter, and its demonstration types `y1`."""
    file.write_text("ODE(y1, u) := y1 + u + qq7\n")
    session.load_utility(file)
    assert session.author("y1 + u + qq7").text == "y1+u+q*q*7"


def test_blank_lines_separate_nothing(session, file):
    """There is one expression per line; blank lines are only spacing."""
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
    """Rederive writes UTF-8, but the original wrote code page 437, where 0xE9 is the
    capital theta in the name `Θn` that its GRAPHICS.MTH uses. A file is read in Word
    mode, so `Θn` is one name, shown as the original shows it."""
    file.write_bytes(b"\xe9n + 1\r\n\x1a")
    assert session.load(file) == 0
    assert texts(session) == ["Θn+1"]


def test_a_line_that_does_not_parse_is_left_out_and_counted(session, file):
    """The original drops such lines silently. The rest of the file still loads, and
    the count lets the screen report the drop."""
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


async def test_a_merged_annotation_names_the_entry_it_now_means(session, file):
    session.author("SIN(x)/2")
    await session.simplify("#1")
    session.save(file)
    session.merge(file)
    # The file says Simp(#1); merged after two entries, that entry is #3.
    assert annotations(session) == ["User", "Simp(#1)", "User", "Simp(#3)"]


async def test_a_merged_reference_still_means_the_expression_it_meant(session, file):
    session.author("p")
    session.author("#1 + 2")
    session.save(file)
    other = Session()
    for text in ("a", "b", "c"):
        other.author(text)
    other.merge(file)
    assert texts(other) == ["a", "b", "c", "p", "#4+2"]
    assert (await other.simplify("#5")).text == "p + 2"


def test_loading_shifts_nothing(session, file):
    """A load restarts the numbering, so references need no adjustment."""
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


async def test_clearing_expressions_leaves_what_they_defined(session):
    """Per the manual, these are four separate commands, not levels of one."""
    session.author("v := 7")
    session.clear_expressions()
    session.author("v + 1")
    assert (await session.simplify("#1")).text == "8"


async def test_clearing_variables_leaves_the_expressions(session):
    session.author("v := 7")
    session.author("v + 1")
    session.clear_variables()
    assert texts(session) == ["v:=7", "v+1"]
    assert (await session.simplify("#2")).text == "v + 1"


async def test_clearing_functions_takes_the_body_away(session):
    """The name is forgotten, so `F(3)` parses as a product, like any undefined name
    before a parenthesis, rather than as nine."""
    session.author("F(u) := u^2")
    session.author("F(3)")
    assert (await session.simplify("#2")).text == "9"
    session.clear_functions()
    session.author("F(3)")
    assert (await session.simplify("#4")).text == "3*f"


async def test_clearing_all_is_the_other_three_at_once(session):
    session.author("v := 7")
    session.author("F(u) := u^2")
    session.clear_all()
    assert texts(session) == []
    session.author("v + F(3)")
    assert (await session.simplify("#1")).text == "3*f + v"


# -- loading a utility file --------------------------------------------------


async def test_a_utility_file_defines_without_being_displayed(session, file):
    file.write_text("CUBE(t) := t^3\n\nk := 11\n")
    assert session.load_utility(file) == 0
    assert texts(session) == []
    session.author("CUBE(k)")
    assert (await session.simplify("#1")).text == "1331"


def test_a_utility_file_does_not_become_the_session_file(session, file):
    """It is a library, not the worksheet, so the status line keeps naming the open
    worksheet."""
    session.author("x")
    session.save(file)
    other = file.with_name("lib.mth")
    other.write_text("CUBE(t) := t^3\n")
    session.load_utility(other)
    assert session.file == file


def test_a_utility_line_that_does_not_parse_is_counted(session, file):
    file.write_text("CUBE(t) := t^3\n\ny +\n")
    assert session.load_utility(file) == 1


async def test_a_utility_files_names_are_read_whole(session, file):
    """A library is always read in Word input mode.

    The session here is in Character mode, where `ab` is `a*b`, but in the file `ab` is
    one variable. This is what makes SOLVE.MTH's `xk` a variable rather than a product.
    """
    file.write_text("SECOND(ab, cd) := cd - ab\n")
    assert session.load_utility(file) == 0
    assert (await session.simplify("SECOND(2, 5)")).text == "3"


async def test_loading_a_utility_file_leaves_the_input_mode_alone(session, file):
    """Word mode applies only while reading the file; the session's setting is kept."""
    file.write_text("CUBE(t) := t^3\n")
    session.load_utility(file)
    assert (await session.simplify("ab")).text == "a*b"


# -- loading a data file -----------------------------------------------------


def test_each_block_of_a_data_file_is_one_matrix(session, file):
    file.write_text("1 2 3\n4 5 6\n\n7 8\n9 10\n")
    assert session.load_data(file) == 0
    assert texts(session) == ["[[1,2,3],[4,5,6]]", "[[7,8],[9,10]]"]


def test_a_block_of_one_line_is_a_vector(session, file):
    """As in the original, rather than a one-row matrix."""
    file.write_text("1 2 3\n")
    session.load_data(file)
    assert texts(session) == ["[1,2,3]"]


def test_a_data_file_appends_to_what_is_already_there(session, file):
    file.write_text("1 2\n")
    session.author("z")
    session.load_data(file)
    assert texts(session) == ["z", "[1,2]"]


def test_numbers_may_be_parted_by_spaces_or_commas_or_both(session, file):
    file.write_text("1, 2 3 ,4\n")
    session.load_data(file)
    assert texts(session) == ["[1,2,3,4]"]


def test_an_exponent_becomes_the_power_it_means(session, file):
    """Derive's syntax has no exponent notation, so `-2.325E-7` is written as a power of
    ten."""
    file.write_text("-2.325E-7 1.5D3 4.5\n")
    session.load_data(file)
    assert texts(session) == ["[-2.325*10^(-7),1.5*10^3,4.5]"]


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
    """The file may be read by a session with any radix, so it can't use the base it
    selects."""
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
    """Word for word, except Print.

    Derive lists `Load Save Merge Clear Demo Print`. Print's submenu configures a
    printer (type, paper size, PCL fonts), and printing is a non-goal, so it is
    omitted, as `Display` and `Execute` are from the Options menu.
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
    """The original's targets are Basic, C, Fortran and Pascal, updated here.

    Each word must still have its own mnemonic letter.
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
    """They configure the next save, so, as in the original, the save menu stays up,
    unlike after an Options dialog."""
    async with app.run_test() as pilot:
        await pilot.press("t", "s", "o", "s", "enter")
        assert app.settings["SaveRange"] == "Some"
        assert band(app)[0].startswith(" TRANSFER SAVE:")
        # Nothing is recorded in the worksheet: these are not system settings.
        assert entries(app) == []


# -- saving through the app --------------------------------------------------


async def test_saving_asks_for_a_name_and_writes_the_file(app, file):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x^2+1", "enter")
        await pilot.press("t", "s", "d")
        assert prompt(app) == ("TRANSFER SAVE DERIVE file:", "")
        assert message(app) == "Enter filename (TAB completes, opens the list)"
        await pilot.press(*file.name, "enter")
        assert file.read_text() == "x^2+1\n"
        assert message(app) == "Enter option"
        # The command ran, so the menus return to the top.
        assert band(app)[0].startswith(" COMMAND:")


async def test_the_status_line_names_the_file_the_session_is_on(app, file):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter")
        await pilot.press("t", "s", "d", *file.name, "enter")
        status = text_of(app.query_one("#status")).plain
        assert "work.mth" in status
        # Shown beside the annotation, not instead of it.
        assert annotation(app) == "User"


async def test_the_file_last_used_is_offered_back(app, file):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter")
        await pilot.press("t", "s", "d", *file.name, "enter")
        await pilot.press("t", "s", "d")
        assert prompt(app) == ("TRANSFER SAVE DERIVE file:", file.name)
        # The offered name is selected, so typing replaces it.
        await pilot.press("q")
        assert prompt(app)[1] == "q"


async def test_the_history_keys_walk_without_touching_the_name(app, file):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter")
        await pilot.press("a", *"y", "enter")
        await pilot.press("t", "s", "d", *file.name, "enter")
        await pilot.press("t", "s", "d")
        await pilot.press("up")
        # The highlight moves as under any prompt, but a file name is not a label, so
        # the line is unchanged.
        assert highlighted_expression(app) == "x"
        assert prompt(app) == ("TRANSFER SAVE DERIVE file:", file.name)


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
        await pilot.press(*file.name, "enter")
        assert file.read_text() == "b\n\nc\n"


# -- loading and merging through the app -------------------------------------


async def test_loading_replaces_what_is_on_screen(app, file):
    file.write_text("a\n\nb\n")
    async with app.run_test() as pilot:
        await pilot.press("a", *"z", "enter")
        await pilot.press("t", "l", "d")
        assert prompt(app) == ("TRANSFER LOAD DERIVE file:", "")
        assert message(app) == "Enter filename (TAB completes, opens the list)"
        await pilot.press(*file.name, "enter")
        assert entries(app) == ["a", "b"]
        assert message(app) == "Enter option"


async def test_ctrl_enter_simplifies_every_expression_it_reads(app, file):
    file.write_text("2+3\n\n4 5\n")
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d", *file.name)
        await pilot.press("ctrl+j")
        assert entries(app) == ["2+3", "4*5", "5", "20"]
        assert message(app).startswith("Compute time:")


async def test_merging_adds_to_what_is_on_screen(app, file):
    file.write_text("a\n\nb\n")
    async with app.run_test() as pilot:
        await pilot.press("a", *"z", "enter")
        await pilot.press("t", "m")
        assert prompt(app)[0] == "TRANSFER MERGE file:"
        await pilot.press(*file.name, "enter")
        assert entries(app) == ["z", "a", "b"]


async def test_a_name_that_is_nothing_leaves_the_line_up_to_be_corrected(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"z", "enter")
        await pilot.press("t", "l", "d")
        await pilot.press(*"nothing.mth", "enter")
        assert message(app) == "File not found"
        assert prompt(app)[1].endswith("nothing.mth")
        assert entries(app) == ["z"]
        # Esc returns to the menu the command was chosen from.
        await pilot.press("escape")
        assert band(app)[0].startswith(" TRANSFER LOAD:")


async def test_a_file_with_a_line_that_will_not_parse_says_how_many(app, file):
    file.write_text("x^2+1\n\ny +\n\nz\n")
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d", *file.name, "enter")
        assert entries(app) == ["x^2+1", "z"]
        assert message(app) == "1 expression could not be read"


async def test_loading_a_utility_file_shows_nothing_and_defines_everything(app, file):
    file.write_text("CUBE(t) := t^3\n")
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "u")
        assert prompt(app) == ("TRANSFER LOAD UTILITY file:", "")
        await pilot.press(*file.name, "enter")
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
        await pilot.press(*numbers.name, "enter")
        assert entries(app) == ["[[1,2],[3,4]]"]


async def test_a_name_with_no_extension_gets_the_one_its_command_reads(app, tmp_path):
    """As in the original, the extension (MTH, DAT or DMO) depends on the command."""
    (tmp_path / "grid.dat").write_text("1 2\n")
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "t", *"grid", "enter")
        assert entries(app) == ["[1,2]"]


# -- completing a name on the line -------------------------------------------


async def test_tab_completes_the_name_and_the_file_reads(app, file):
    file.write_text("a\n\nb\n")
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        await pilot.press(*"wo", "tab")
        assert prompt(app)[1] == file.name
        await pilot.press("enter")
        assert entries(app) == ["a", "b"]


async def test_tab_writes_out_what_the_names_share_and_opens_the_list(app, tmp_path):
    """The line is extended to the common prefix, and the list shows the candidates."""
    for name in ("workbook.mth", "worksheet.mth"):
        (tmp_path / name).touch()
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        await pilot.press(*"w", "tab")
        assert prompt(app)[1] == "work"
        assert completions(app) == ["workbook.mth", "worksheet.mth"]
        # Opening the list selects nothing.
        assert chosen(app) is None


async def test_the_list_says_which_directory_it_is_showing(app, tmp_path):
    for name in ("alpha.mth", "beta.mth"):
        (tmp_path / name).touch()
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        await pilot.press(*f"{tmp_path}/", "tab")
        where, _, count = listing_title(app).rpartition(" - ")
        assert count == "2"
        # A directory too long for the border is truncated from the left, keeping the
        # informative end.
        assert f"{tmp_path}/".endswith(where.removeprefix("..."))


async def test_tab_with_the_list_up_steps_through_it(app, tmp_path):
    for name in ("workbook.mth", "worksheet.mth"):
        (tmp_path / name).touch()
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        await pilot.press(*"work", "tab")
        assert chosen(app) is None
        await pilot.press("tab")
        assert (chosen(app), prompt(app)[1]) == ("workbook.mth", "workbook.mth")
        await pilot.press("tab")
        assert (chosen(app), prompt(app)[1]) == ("worksheet.mth", "worksheet.mth")
        # The list wraps around.
        await pilot.press("tab")
        assert chosen(app) == "workbook.mth"


async def test_shift_tab_steps_back_through_the_list(app, tmp_path):
    """Unlike the original's completion, overshooting takes one press to undo."""
    for name in ("alpha.mth", "beta.mth", "gamma.mth"):
        (tmp_path / name).touch()
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        await pilot.press("tab", "tab", "tab", "tab")
        assert chosen(app) == "gamma.mth"
        await pilot.press("shift+tab")
        assert (chosen(app), prompt(app)[1]) == ("beta.mth", "beta.mth")
        await pilot.press("shift+tab")
        assert chosen(app) == "alpha.mth"
        # It wraps backwards too.
        await pilot.press("shift+tab")
        assert chosen(app) == "gamma.mth"


async def test_the_arrows_walk_the_list_the_same_way(app, tmp_path):
    for name in ("alpha.mth", "beta.mth"):
        (tmp_path / name).touch()
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        await pilot.press("tab", "down")
        assert chosen(app) == "alpha.mth"
        await pilot.press("down")
        assert chosen(app) == "beta.mth"
        await pilot.press("up")
        assert chosen(app) == "alpha.mth"


async def test_the_arrows_still_walk_the_history_with_no_list_up(app, tmp_path):
    """Up and Down move in the list only while it is shown."""
    (tmp_path / "work.mth").touch()
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter")
        await pilot.press("a", *"y", "enter")
        await pilot.press("t", "l", "d")
        assert completions(app) is None
        await pilot.press("up")
        assert highlighted_expression(app) == "x"


async def test_enter_goes_into_a_directory_and_the_list_follows(app, tmp_path):
    """Walking down a tree: one key per level, each level listed."""
    (tmp_path / "utility").mkdir()
    (tmp_path / "utility" / "trig.mth").touch()
    (tmp_path / "utility" / "vector.mth").touch()
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        await pilot.press(*f"{tmp_path}/uti", "tab")
        # A single match, so Tab takes it and lists its contents.
        assert prompt(app)[1] == f"{tmp_path}/utility/"
        assert completions(app) == ["trig.mth", "vector.mth"]
        await pilot.press("down", "enter")
        assert prompt(app)[1] == f"{tmp_path}/utility/trig.mth"
        # Choosing a file closes the list without reading the file.
        assert completions(app) is None
        assert app.session.file is None
        await pilot.press("enter")
        assert app.session.file == tmp_path / "utility" / "trig.mth"


async def test_enter_on_a_directory_in_the_list_opens_it(app, tmp_path):
    (tmp_path / "math").mkdir()
    (tmp_path / "math" / "algebra").mkdir()
    (tmp_path / "math" / "algebra" / "groups.mth").touch()
    (tmp_path / "notes.mth").touch()
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        await pilot.press("tab")
        assert completions(app) == ["math/", "notes.mth"]
        await pilot.press("down", "enter")
        assert completions(app) == ["algebra/"]
        await pilot.press("enter")
        assert completions(app) == ["groups.mth"]
        assert prompt(app)[1] == "math/algebra/"


async def test_typing_narrows_the_list_without_taking_the_line(app, tmp_path):
    """The list filters as you type, and typed text is never overwritten."""
    for name in ("alpha.mth", "beta.mth", "berry.mth"):
        (tmp_path / name).touch()
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        await pilot.press("tab")
        assert completions(app) == ["alpha.mth", "berry.mth", "beta.mth"]
        await pilot.press("b")
        assert completions(app) == ["berry.mth", "beta.mth"]
        assert prompt(app)[1] == "b"
        await pilot.press("e", "t")
        assert completions(app) == ["beta.mth"]
        assert prompt(app)[1] == "bet"


async def test_backspacing_opens_the_list_back_out(app, tmp_path):
    """Backspacing over a separator walks back up the tree."""
    (tmp_path / "utility").mkdir()
    (tmp_path / "utility" / "trig.mth").touch()
    (tmp_path / "work.mth").touch()
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        await pilot.press(*"uti", "tab")
        assert completions(app) == ["trig.mth"]
        await pilot.press("backspace")
        assert completions(app) == ["utility/"]
        await pilot.press(*["backspace"] * 7)
        assert prompt(app)[1] == ""
        assert completions(app) == ["utility/", "work.mth"]


async def test_a_name_the_list_has_no_match_for_closes_it(app, tmp_path):
    for name in ("alpha.mth", "beta.mth"):
        (tmp_path / name).touch()
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        await pilot.press("tab")
        assert completions(app) == ["alpha.mth", "beta.mth"]
        await pilot.press("z")
        assert completions(app) is None
        assert message(app) == "Enter filename (TAB completes, opens the list)"


async def test_one_name_and_no_other_is_taken_with_no_list_to_look_at(app, tmp_path):
    """A single match is taken without opening the list."""
    (tmp_path / "alpha.mth").touch()
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        await pilot.press("tab")
        assert prompt(app)[1] == "alpha.mth"
        assert completions(app) is None


async def test_escape_puts_the_list_away_before_it_leaves_the_command(app, tmp_path):
    """The first Escape closes the list; only the second leaves the command."""
    for name in ("alpha.mth", "beta.mth"):
        (tmp_path / name).touch()
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        await pilot.press("tab", "tab")
        assert completions(app) == ["alpha.mth", "beta.mth"]
        await pilot.press("escape")
        # The list closes, but the prompt and the name browsed to remain.
        assert completions(app) is None
        assert prompt(app)[1] == "alpha.mth"
        await pilot.press("escape")
        assert band(app)[0].startswith(" TRANSFER LOAD:")


async def test_a_long_list_scrolls_and_says_how_much_is_showing(app, tmp_path):
    for number in range(40):
        (tmp_path / f"utility{number:02}.mth").touch()
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        await pilot.press("tab")
        rows = completions(app)
        assert rows == [f"utility{number:02}.mth" for number in range(10)]
        assert listing_title(app).endswith("1-10 of 40")
        # Page Down moves a screenful and the view follows the highlight.
        await pilot.press("pagedown")
        assert chosen(app) == "utility10.mth"
        assert chosen(app) in completions(app)
        # Paging stops at the end rather than wrapping.
        await pilot.press(*["pagedown"] * 6)
        assert chosen(app) == "utility39.mth"
        assert listing_title(app).endswith("31-40 of 40")


@pytest.mark.parametrize("height", [24, 16, 12, 10])
async def test_the_list_gives_up_rows_rather_than_the_line(app, tmp_path, height):
    """A short terminal shortens the list, never the prompt line or bands."""
    for number in range(40):
        (tmp_path / f"utility{number:02}.mth").touch()
    async with app.run_test(size=(80, height)) as pilot:
        await pilot.press("t", "l", "d")
        await pilot.press("tab")
        assert completions(app)
        await laid_out(pilot, app, "#prompt-band")
        # Everything below the list keeps its height, so the list pushed nothing off.
        for identifier, rows in (
            ("#prompt-band", 2),
            ("#message", 1),
            ("#status", 1),
        ):
            assert app.query_one(identifier).size.height == rows, identifier
        listing = app.query_one("#completions")
        assert listing.region.bottom <= app.query_one("#rule").region.y


async def test_a_name_that_completes_to_nothing_is_the_beep(app):
    async with app.run_test() as pilot:
        beeps = []
        app.bell = lambda: beeps.append("beep")
        await pilot.press("t", "l", "d")
        await pilot.press(*"nothing", "tab")
        assert beeps == ["beep"]
        assert prompt(app)[1] == "nothing"


async def test_a_command_completes_to_the_files_it_reads(app, tmp_path):
    """Load Derive offers no data files, and daTa offers only data files."""
    (tmp_path / "numbers.dat").touch()
    (tmp_path / "numbers.mth").touch()
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        await pilot.press(*"num", "tab")
        assert prompt(app)[1] == "numbers.mth"
        await pilot.press("escape")
        await pilot.press("t")
        await pilot.press(*"num", "tab")
        assert prompt(app)[1] == "numbers.dat"


async def test_a_name_being_saved_completes_too(app, file):
    """Completion works when saving, to overwrite an existing file."""
    file.write_text("a\n")
    async with app.run_test() as pilot:
        await pilot.press("a", *"z", "enter")
        await pilot.press("t", "s", "d")
        await pilot.press(*"wo", "tab", "enter")
        assert file.read_text() == "z\n"


async def test_tab_on_a_line_that_names_no_file_leaves_it_alone(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x+1", "tab")
        assert prompt(app) == ("AUTHOR expression:", "x+1")


async def test_the_line_offers_the_completion_as_it_is_typed(app, file):
    """The suggestion is shown dimmed after the cursor, and Right accepts it."""
    file.write_text("a\n")
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        await pilot.press(*"wo")
        await pilot.pause()
        await pilot.press("right")
        assert prompt(app)[1] == file.name


# -- saving source code ------------------------------------------------------


async def test_saving_a_language_writes_it_under_that_language_suffix(app, tmp_path):
    async with app.run_test() as pilot:
        await pilot.press("a", *"SQRT(x)+1", "enter")
        await pilot.press("t", "s", "p")
        assert prompt(app) == ("TRANSFER SAVE PYTHON file:", "")
        await pilot.press(*"work", "enter")
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
        await pilot.press("t", "s", letter, *"work", "enter")
        assert (tmp_path / name).read_text() == expected


async def test_an_annotation_goes_behind_the_target_comment_marker(app, tmp_path):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x^2", "enter")
        await pilot.press("s", "enter")
        await pilot.press("t", "s", "j", *"work", "enter")
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
        await pilot.press(*"work", "enter")
        assert (tmp_path / "work.c").read_text() == "b\n\nc\n"


def test_a_language_save_does_not_break_long_lines(session, tmp_path):
    """The line length only applies to math files; none of the four languages need it."""
    session.author("(a+b)^3 (c+d)^3 (e+f)^3")
    session.settings.apply({"SaveLength": 20})
    julia = next(l for l in LANGUAGES if l.word == "Julia")
    session.save_source(tmp_path / "work.jl", julia)
    assert (tmp_path / "work.jl").read_text().splitlines() == [
        "(a + b) ^ 3 * (c + d) ^ 3 * (e + f) ^ 3"
    ]


# -- the state file through the app ------------------------------------------


async def test_the_state_commands_offer_a_file_of_their_own(app):
    """State commands offer their own file name, separate from the worksheet's."""
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter")
        await pilot.press("t", "s", "d", *"work", "enter")
        await pilot.press("t", "s", "s")
        assert prompt(app) == ("TRANSFER SAVE STATE file:", "rederive.ini")


async def test_saving_and_loading_the_state_carries_the_settings_over(app, tmp_path):
    async with app.run_test() as pilot:
        await pilot.press("o", "n", "m", "enter")
        assert app.settings["Notation"] == "Mixed"
        await pilot.press("t", "s", "s", *"state", "enter")
        assert (tmp_path / "state.ini").exists()
        await pilot.press("o", "n", "r", "enter")
        assert app.settings["Notation"] == "Rational"
        await pilot.press("t", "l", "s")
        # The load offers the file just written.
        assert prompt(app) == ("TRANSFER LOAD STATE file:", "state.ini")
        await pilot.press("enter")
        assert app.settings["Notation"] == "Mixed"


async def test_loading_a_state_leaves_the_history_alone(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter")
        await pilot.press("t", "s", "s", *"state", "enter")
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
        # Declined, so the Clear menu stays up.
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
    """Nothing on screen is removed, so there is nothing to lose by mistake."""
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
        await pilot.press("t", "d", "o")
        assert prompt(app) == ("TRANSFER DEMO OTHER file:", "")
        assert message(app) == "Enter filename (TAB completes, opens the list)"
        await pilot.press(*demo.name, "enter")
        # The comment replaces the menu band and waits for a key.
        assert band(app) == [" adds two numbers"]
        assert message(app) == "Press any key to continue, ESC to stop"
        assert entries(app) == ["2+3", "5"]
        await pilot.press("space")
        assert band(app) == [" and a symbolic one"]
        assert entries(app) == ["2+3", "5", "(x+1)^2", "(x + 1)^2"]
        # After the last step, the command ends.
        await pilot.press("space")
        assert band(app)[0].startswith(" COMMAND:")
        assert message(app) == "Enter option"


async def test_a_demonstration_step_is_read_in_word_mode(app, tmp_path):
    """Like any line read from a file, while the Author line stays in Character mode."""
    path = tmp_path / "words.dmo"
    path.write_text("; one name\nab + 1\n")
    async with app.run_test() as pilot:
        await pilot.press("t", "d", "o", *path.name, "enter")
        assert entries(app) == ["ab+1", "ab + 1"]
        await pilot.press("space")
    assert app.session.author("ab").text == "a*b"


async def test_escape_suspends_a_demonstration_where_it_stands(app, demo):
    async with app.run_test() as pilot:
        await pilot.press("t", "d", "o", *demo.name, "enter")
        await pilot.press("escape")
        assert band(app)[0].startswith(" COMMAND:")
        # Other work is possible, and naming the same file resumes it.
        await pilot.press("a", *"z", "enter")
        await pilot.press("t", "d", "o", *demo.name, "enter")
        assert band(app) == [" and a symbolic one"]
        assert entries(app) == ["2+3", "5", "z", "(x+1)^2", "(x + 1)^2"]


async def test_a_demonstration_that_has_run_out_starts_over(app, demo):
    async with app.run_test() as pilot:
        await pilot.press("t", "d", "o", *demo.name, "enter")
        await pilot.press("space", "space")
        await pilot.press("t", "d", "o", *demo.name, "enter")
        assert band(app) == [" adds two numbers"]


async def test_a_demonstration_can_start_from_anywhere_on_the_screen(app, demo):
    """The page's demo menu is clicked, and a click can come on any screen."""
    async with app.run_test() as pilot:
        await pilot.press("a", *"x+")
        app.demonstrate(demo.name)
        await pilot.pause()
        # The interrupted author line is gone, and the step is shown.
        assert band(app) == [" adds two numbers"]
        assert entries(app) == ["2+3", "5"]
        # The abandoned line isn't resumed: the demonstration ends at the command menu.
        await pilot.press("space", "space")
        assert band(app)[0].startswith(" COMMAND:")
        assert message(app) == "Enter option"


async def test_a_demonstration_file_that_is_nothing_leaves_the_line_up(app):
    async with app.run_test() as pilot:
        await pilot.press("t", "d", "o", *"nothing.dmo", "enter")
        assert message(app) == "File not found"
        assert prompt(app)[0] == "TRANSFER DEMO OTHER file:"


# -- the original's own demonstrations ----------------------------------------
#
# Here the desktop departs from the original's Transfer menu: Derive asked for a file
# name because the demonstrations were on its diskette, while here they are listed by
# name and downloaded. The download itself is tested in `test_demos`; these tests cover
# the menu and what happens to a downloaded demonstration: it is shown and kept in
# memory, and the working directory is left untouched.


SHOW = b"; adds two numbers\r\n2 + 3\r\n\x1a"


@pytest.fixture
def downloaded(monkeypatch):
    """Fake the download, recording which files were requested."""
    asked = []

    def fetched(demo, keep=None):
        asked.append(demo.file)
        if isinstance(brought.get(demo.file, SHOW), OSError):
            raise brought[demo.file]
        for file, data in also.items():
            keep(file, data)
        return brought.get(demo.file, SHOW)

    brought: dict[str, object] = {}
    also: dict[str, bytes] = {}
    monkeypatch.setattr(fetch, "fetched", fetched)
    return SimpleNamespace(asked=asked, brought=brought, also=also)


async def settled(pilot, condition, patience=2.0):
    """Wait for the download task, which runs off the event loop."""
    deadline = time.monotonic() + patience
    while time.monotonic() < deadline and not condition():
        await pilot.pause(0.002)
    return condition()


async def test_the_demonstrations_are_offered_as_a_menu_of_their_own(app):
    """Nine demonstrations plus the original's `Other` command."""
    async with app.run_test() as pilot:
        await pilot.press("t", "d")
        assert band(app) == [
            " TRANSFER DEMO: Arithmetic alGebra Trigonometry Functions Calculus Matrices",
            "                2D-plots Parametric 3D-plots Other",
        ]
        assert highlighted(app) == "Arithmetic"
        # Each word has its own mnemonic; the galleries use a digit.
        words = menus.TRANSFER_DEMO.words
        assert len({mnemonic(word) for word in words}) == len(words)


async def test_a_demonstration_that_is_not_here_is_downloaded_and_remembered(
    app, downloaded
):
    async with app.run_test() as pilot:
        await pilot.press("t", "d", "a")
        assert await settled(pilot, lambda: band(app) == [" adds two numbers"])
        assert entries(app) == ["2+3", "5"]
        assert downloaded.asked == ["ARITH.DMO"]
        # Nothing is written: the user asked for a demonstration, not a file.
        assert not Path("ARITH.DMO").exists()
        # It is kept for the rest of the run, so a second viewing needs no download.
        await pilot.press("space", "t", "d", "a")
        assert band(app) == [" adds two numbers"]
        assert downloaded.asked == ["ARITH.DMO"]


async def test_a_demonstration_that_is_here_is_not_downloaded_again(app, downloaded):
    Path("ARITH.DMO").write_text("; already here\n1 + 1\n")
    async with app.run_test() as pilot:
        await pilot.press("t", "d", "a")
        assert band(app) == [" already here"]
        assert downloaded.asked == []


async def test_the_files_the_diskette_brought_with_it_are_kept_too(app, downloaded):
    """The other files from the diskette are kept, so they aren't downloaded again."""
    downloaded.also["TRIG.DMO"] = b"; the sine\r\nSIN(0)\r\n\x1a"
    async with app.run_test() as pilot:
        await pilot.press("t", "d", "a")
        assert await settled(pilot, lambda: band(app) == [" adds two numbers"])
        # The extra file runs without a download and isn't written to disk either.
        await pilot.press("space", "t", "d", "t")
        assert band(app) == [" the sine"]
        assert downloaded.asked == ["ARITH.DMO"]
        assert not Path("TRIG.DMO").exists()


async def test_a_demonstration_that_will_not_come_says_so_and_leaves_the_menu(
    app, downloaded
):
    downloaded.brought["ARITH.DMO"] = OSError("the server said 503")
    async with app.run_test() as pilot:
        await pilot.press("t", "d", "a")
        assert await settled(pilot, lambda: "503" in message(app))
        assert message(app) == "ARITH.DMO could not be downloaded: the server said 503"
        # The menu stays up for another choice.
        assert band(app)[0].startswith(" TRANSFER DEMO:")


async def test_a_second_demonstration_waits_for_the_one_on_its_way(app, monkeypatch):
    holding = threading.Event()

    def fetched(demo, keep=None):
        holding.wait(2.0)
        return SHOW

    monkeypatch.setattr(fetch, "fetched", fetched)
    async with app.run_test() as pilot:
        await pilot.press("t", "d", "a")
        assert message(app) == "Fetching ARITH.DMO from the Internet Archive..."
        await pilot.press("t", "d", "g")
        assert message(app) == (
            "ARITH.DMO is still on its way; the next one can be asked for after"
        )
        holding.set()
        assert await settled(pilot, lambda: band(app) == [" adds two numbers"])
        assert not Path("ALGEBRA.DMO").exists()


async def test_watching_a_demonstration_writes_nothing_anywhere(
    app, downloaded, monkeypatch
):
    """Not the demonstration's file, nor anything under another name.

    The check is on the storage write that every program write goes through, since a
    write elsewhere would leave the directory looking clean.
    """
    monkeypatch.setattr(
        DesktopStorage, "write", lambda *_: (_ for _ in ()).throw(OSError("read-only"))
    )
    async with app.run_test() as pilot:
        await pilot.press("t", "d", "a")
        assert await settled(pilot, lambda: band(app) == [" adds two numbers"])
        assert entries(app) == ["2+3", "5"]
        assert list(Path().iterdir()) == []


async def test_the_environments_own_download_is_what_a_demonstration_comes_by(
    app, downloaded
):
    """A tab has no sockets, so the environment supplies the fetcher.

    The fetcher also decides where the bytes go: a page stores them by name, a desktop
    keeps them in memory for the run.
    """
    asked = []

    async def brought(demo):
        asked.append(demo.file)
        return SHOW

    app.fetcher = brought
    async with app.run_test() as pilot:
        await pilot.press("t", "d", "a")
        assert await settled(pilot, lambda: band(app) == [" adds two numbers"])
        assert asked == ["ARITH.DMO"]
        assert downloaded.asked == []
        assert not Path("ARITH.DMO").exists()


async def test_a_gallery_is_refused_before_anything_is_downloaded(
    app, downloaded, monkeypatch
):
    """A gallery needs a plot window, which isn't available here."""
    monkeypatch.setattr("rederive.ui.app.available", lambda: False)
    async with app.run_test() as pilot:
        await pilot.press("t", "d", "3")
        assert message(app) == "Plot: needs a graphical display"
        assert downloaded.asked == []


def test_a_demonstration_is_its_comments_and_the_lines_under_them(tmp_path):
    path = tmp_path / "show.dmo"
    path.write_text("; one\na+~\nb\n\nno comment here\n\n; two\nc\n")
    assert worksheet.demonstration(path) == (
        ("one", "a+b"),
        ("", "no comment here"),
        ("two", "c"),
    )


def test_a_quoted_line_is_the_comment_of_the_step_under_it(tmp_path):
    """This is how the original's plot galleries are written."""
    path = tmp_path / "gallery.mth"
    path.write_text('"A spike:"\n\n2*SIN(LN(ABS(x)))\n\n"A saddle:"\n\ny^2-x^2\n')
    assert worksheet.demonstration(path) == (
        ("A spike:", "2*SIN(LN(ABS(x)))"),
        ("A saddle:", "y^2-x^2"),
    )


def test_a_line_with_more_on_it_than_a_string_is_a_step(tmp_path):
    """A string combined with anything else is an ordinary expression step."""
    path = tmp_path / "gallery.mth"
    path.write_text('"a" + "b"\n\n["x", 1]\n')
    assert worksheet.demonstration(path) == (("", '"a" + "b"'), ("", '["x", 1]'))


def test_the_end_of_a_dos_file_is_not_a_step(tmp_path):
    """Every file the original wrote ends with Ctrl-Z, which is not a step."""
    path = tmp_path / "show.dmo"
    path.write_bytes(b"; one\r\na+b\r\n\x1a")
    assert worksheet.demonstration(path) == (("one", "a+b"),)
