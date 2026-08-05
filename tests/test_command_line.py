"""Files named on the command line, and how each is decided to be read.

The original took them after the program name and said what each was with a
slash and a letter behind it - `DERIVE NUMBER/U PLOT2D/M`. A slash cannot mean
that here, so the letters are switches in front of the name; what they choose
between, and what an extension chooses when nothing else does, is the
original's.
"""

import pytest
from screen import band, entries, message

from rederive import __version__
from rederive.__main__ import Usage, named, provenance, read
from rederive.model.session import Session
from rederive.ui.app import RederiveApp


@pytest.fixture
def session():
    return Session()


@pytest.fixture
def files(tmp_path, monkeypatch):
    """A directory of one file of each kind, and the shell sitting in it."""
    (tmp_path / "work.mth").write_text("x + 1\n2 x\n", encoding="utf-8")
    (tmp_path / "more.mth").write_text("y\n", encoding="utf-8")
    (tmp_path / "lib.mth").write_text("SQ(u) := u^2\n", encoding="utf-8")
    (tmp_path / "nums.dat").write_text("1 2\n3 4\n", encoding="utf-8")
    (tmp_path / "show.dmo").write_text(";a comment\n2 + 3\n", encoding="utf-8")
    (tmp_path / "arith.dmo").write_text(";adding up\n1 + 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def kinds(arguments):
    """What each name on a command line is taken to be."""
    return [(kind, path.name) for kind, path in named(arguments)]


def test_an_extension_says_what_a_file_is(files):
    assert kinds(["work.mth", "nums.dat", "show.dmo"]) == [
        ("math", "work.mth"),
        ("data", "nums.dat"),
        ("demo", "show.dmo"),
    ]


def test_a_name_without_one_is_looked_for_under_each(files):
    # The manual's own example: `DERIVE ARITH` runs ARITH.DMO, there being no
    # ARITH.MTH beside it.
    assert kinds(["arith"]) == [("demo", "arith.dmo")]
    assert kinds(["work"]) == [("math", "work.mth")]
    assert kinds(["nums"]) == [("data", "nums.dat")]


def test_a_name_that_is_nowhere_is_reported_as_the_likeliest_file(files):
    assert kinds(["missing"]) == [("math", "missing.mth")]


def test_a_switch_says_what_the_extension_would_not(files):
    # A utility file is a math file read a different way, so the switch is the
    # only thing that can say it is one.
    assert kinds(["-u", "lib.mth"]) == [("utility", "lib.mth")]
    assert kinds(["-t", "work.mth"]) == [("data", "work.mth")]


def test_a_switch_holds_until_the_next_one(files):
    assert kinds(["-u", "lib", "more", "-m", "work"]) == [
        ("utility", "lib.mth"),
        ("utility", "more.mth"),
        ("math", "work.mth"),
    ]


def test_a_switch_supplies_its_own_extension(files):
    assert kinds(["-t", "nums"]) == [("data", "nums.dat")]
    assert kinds(["-d", "show"]) == [("demo", "show.dmo")]


def test_the_long_spellings_mean_the_same(files):
    assert kinds(["--utility", "lib"]) == kinds(["-u", "lib"])
    assert kinds(["--demo", "show"]) == kinds(["-d", "show"])


def test_an_unknown_switch_is_refused(files):
    with pytest.raises(Usage, match="unknown option"):
        named(["-x", "work"])


def test_the_files_are_read_into_the_session(session, files):
    demo, message = read(session, named(["work.mth"]))
    assert demo is None
    assert message == ""
    assert [entry.text for entry in session.entries] == ["x+1", "2*x"]


def test_a_second_math_file_adds_to_the_first(session, files):
    read(session, named(["work", "more"]))
    # Merged rather than loaded, so naming two does not leave only the last.
    assert [entry.text for entry in session.entries] == ["x+1", "2*x", "y"]


def test_a_utility_file_is_read_without_being_shown(session, files):
    read(session, named(["-u", "lib"]))
    assert session.entries == []
    assert session.author("SQ(3)").text == "SQ(3)"


def test_a_data_file_becomes_a_matrix(session, files):
    read(session, named(["nums.dat"]))
    assert [entry.text for entry in session.entries] == ["[[1,2],[3,4]]"]


def test_a_demonstration_is_left_for_the_screen_to_run(session, files):
    demo, _ = read(session, named(["arith"]))
    assert demo is not None and demo.name == "arith.dmo"
    # It is not read here: a demonstration is a session to be driven rather
    # than a file to be taken in.
    assert session.entries == []


def test_only_one_demonstration_can_be_run(session, files):
    with pytest.raises(Usage, match="only one demonstration"):
        read(session, named(["show", "arith"]))


def test_a_file_that_is_not_there_is_refused(session, files):
    with pytest.raises(Usage, match="no such file"):
        read(session, named(["missing"]))


def test_lines_that_will_not_parse_are_counted(session, files, tmp_path):
    (tmp_path / "bent.mth").write_text("x + 1\n2 +\n) (\n", encoding="utf-8")
    _, message = read(session, named(["bent"]))
    assert message == "2 lines not read"
    assert [entry.text for entry in session.entries] == ["x+1"]


def test_one_such_line_is_counted_in_the_singular(session, files, tmp_path):
    (tmp_path / "bent.mth").write_text("x + 1\n2 +\n", encoding="utf-8")
    _, message = read(session, named(["bent"]))
    assert message == "1 line not read"


# -- what the screen opens on ------------------------------------------------


def opened(arguments):
    """The app as the command line leaves it, before it is run."""
    session = Session()
    demo, opening = read(session, named(arguments))
    return RederiveApp(session, demo=demo, opening=opening)


async def test_the_worksheet_is_on_screen_from_the_first_frame(files):
    app = opened(["work"])
    async with app.run_test():
        assert entries(app) == ["x+1", "2*x"]
        assert message(app) == "Enter option"


async def test_what_would_not_read_is_said_in_place_of_the_invitation(files, tmp_path):
    (tmp_path / "bent.mth").write_text("x + 1\n2 +\n", encoding="utf-8")
    app = opened(["bent"])
    async with app.run_test():
        assert message(app) == "1 line not read"


async def test_a_demonstration_runs_as_soon_as_there_is_a_screen(files):
    app = opened(["arith"])
    async with app.run_test() as pilot:
        # Already on its first step, with no Transfer Demo command issued.
        assert band(app) == [" adding up"]
        assert entries(app) == ["1+1", "2"]
        await pilot.press("space")
        assert band(app)[0].startswith(" COMMAND:")


async def test_files_read_first_are_there_for_the_demonstration_to_use(files):
    app = opened(["work", "arith"])
    async with app.run_test():
        assert entries(app) == ["x+1", "2*x", "1+1", "2"]


# -- what the program says it is -------------------------------------------


def test_provenance_names_the_program_and_what_it_runs_on():
    """One `name version` per line, which is the form the build workflow reads.

    A release carries its own interpreter and its own sympy, so these four lines are
    the only account of what is inside a binary; the workflow checks the middle two
    against `.python-version` and `uv.lock`.
    """
    reported = dict(line.split(maxsplit=1) for line in provenance().splitlines())
    assert reported["rederive"] == __version__
    assert reported["Python"].split(".")[0] == "3"
    assert reported["sympy"][0].isdigit()
    assert "platform" in reported
