"""Files in a tab, asked without one: the store, the download, the link.

A browser has no filesystem and three gestures instead, and every question
worth asking about them is a question about which gesture a file took and what
was said about it. What is real here is all of the Python: the store that gives
a name back its file, the rule that a save leaves the tab before it is
remembered, the encoding a link carries a worksheet in, and the words for a
clipboard that would not take one.

The tab is faked to the depth the code touches it - a store with a ceiling, a
clipboard that can refuse, an address - and no further. Nothing here draws
anything, and no browser is started.
"""

from pathlib import Path

import pytest
from fakepage import Clipboard, FakeTab, Store, browser

from rederive import platform
from rederive.model import worksheet
from rederive.model.session import Session
from rederive.platform.web import CLIPBOARD_REFUSED, KEPT, Web, WebStorage
from rederive.web import files as gestures
from rederive.web.files import Files

SHEET = Path("work.mth")


@pytest.fixture
def page():
    return FakeTab()


@pytest.fixture(autouse=True)
def environment(page):
    """The browser in force, as `web.boot` puts it, and the desktop back after.

    Not decoration. A link asks the platform for a clipboard on its way out, and
    a test that left the desktop in place would ask the machine's own.
    """
    browser()
    platform.use(Web(page))
    yield
    platform.use(None)


@pytest.fixture
def storage():
    return platform.current().storage()


def again(page, hash=""):
    """The next visit to this page: a new environment over the same store.

    Which is the whole of what a reload is. MEMFS, the session and the storage
    object all go; localStorage and the address are what carry across, and they
    are what everything here is about.
    """
    import js

    browser(store=js.localStorage, hash=hash)
    platform.use(Web(page))
    return platform.current().storage()


# -- the store ----------------------------------------------------------------


def test_a_save_leaves_the_tab_and_is_remembered_by_name(storage, page):
    """The two halves of a save, in the order that makes the file safe.

    The download is where the save actually goes - it is the only copy the user
    can keep - and the store is what makes the name mean it again afterwards.
    """
    storage.write(SHEET, "x^2 - 4\n")
    assert page.downloads == [("work.mth", "x^2 - 4\n")]
    assert storage.read(SHEET) == b"x^2 - 4\n"
    assert storage.exists(SHEET)


def test_what_the_store_holds_outlives_the_page_that_wrote_it(page, storage):
    """Which is the whole reason there is a store rather than a dictionary.

    MEMFS is gone by the next visit, and a worksheet that lived there alone
    would be gone with it.
    """
    storage.write(SHEET, "sin(x)\n")
    assert again(page).read(SHEET) == b"sin(x)\n"


def test_a_store_with_no_room_costs_the_name_and_not_the_file(page):
    """The rule that the worksheet is always savable, in the one place it bites.

    The download has already gone by the time the store refuses, so nothing is
    lost but the tab's memory of the name - and that is said out loud rather
    than swallowed.
    """
    import js

    js.localStorage = Store(room=10)
    WebStorage(page).write(SHEET, "a very long worksheet indeed\n")
    assert page.downloads == [("work.mth", "a very long worksheet indeed\n")]
    assert page.notices == [
        "work.mth was downloaded, but this page has no room left to remember it by name"
    ]


def test_a_file_handed_over_is_kept_and_never_sent_back_out(storage, page):
    """A picked or dropped file is already on a disk; downloading it is nonsense."""
    storage.keep(Path("theirs.mth"), "x + 1\n")
    assert storage.read(Path("theirs.mth")) == b"x + 1\n"
    assert page.downloads == []


def test_a_file_on_the_filesystem_is_read_when_the_store_has_none(storage, tmp_path):
    """The store first and MEMFS behind it, for a path that names something real."""
    on_disk = tmp_path / "ondisk.mth"
    on_disk.write_text("x + 1\n", encoding="utf-8", newline="\n")
    assert storage.exists(on_disk)
    assert storage.read(on_disk) == b"x + 1\n"


def test_a_file_of_your_own_wins_over_the_one_on_the_filesystem(storage, tmp_path):
    """The store is asked first, wherever the path points."""
    on_disk = tmp_path / "ondisk.mth"
    on_disk.write_text("theirs\n", encoding="utf-8")
    storage.keep(on_disk, "mine\n")
    assert storage.read(on_disk) == b"mine\n"


def test_a_listing_offers_the_store_and_the_filesystem_together(
    storage, tmp_path, monkeypatch
):
    """What a file prompt completes from, in a tab where both have something.

    A tab's working directory is empty - MEMFS holds the package and nothing
    the user put there - so the store is nearly always the whole of the answer.
    The filesystem is in it anyway, since the worksheets the wheel carries are
    on it and a path typed in full should find them.
    """
    monkeypatch.chdir(tmp_path)
    storage.keep(Path("second.mth"), "")
    storage.keep(Path("first.mth"), "")
    assert list(storage.names(Path())) == ["first.mth", "second.mth"]
    (tmp_path / "onthedisk.mth").write_text("")
    assert list(storage.names(Path())) == ["first.mth", "onthedisk.mth", "second.mth"]
    # A name typed in full is filed under the path and listed under the name,
    # which is what a listing of a directory is.
    storage.keep(tmp_path / "elsewhere.mth", "")
    assert list(storage.names(tmp_path)) == ["elsewhere.mth", "onthedisk.mth"]


def test_a_store_that_will_not_be_read_leaves_the_program_running(page):
    """A tab told to keep no site data keeps none, and starts all the same."""
    import js

    del js.localStorage
    storage = WebStorage(page)
    assert not storage.remembers
    storage.write(SHEET, "x\n")
    assert page.downloads == [("work.mth", "x\n")]
    assert storage.read(SHEET) == b"x\n"


# -- the clipboard ------------------------------------------------------------


def test_a_clipboard_that_takes_the_text_says_nothing(page):
    import js

    said = []
    Web(page).copy("x^2", said.append)
    assert js.navigator.clipboard.written == ["x^2"]
    assert said == []


@pytest.mark.parametrize(
    "clipboard",
    [Clipboard(allow=False), Clipboard(raises=True), None],
    ids=["refused", "threw", "absent"],
)
def test_a_clipboard_that_will_not_take_the_text_says_so(page, clipboard):
    """Never silence, in the one place the browser can refuse a key outright.

    Three ways to be refused - a rejected promise, a call that throws, a page
    that has no clipboard at all - and one sentence, because the difference
    between them is nothing the user can act on.
    """
    import js

    said = []
    js.navigator.clipboard = clipboard
    Web(page).copy("x^2", said.append)
    assert said == [CLIPBOARD_REFUSED]


# -- the link -----------------------------------------------------------------


def _session(*expressions):
    session = Session()
    for expression in expressions:
        session.author(expression)
    return session


def test_a_link_carries_the_worksheet_and_comes_back_as_the_same_one(page, storage):
    """The round trip, which is the whole of what a permalink promises."""
    session = _session("x^2 - 4", "sin(x)/x")
    Files(page, session, storage).linked()
    assert page.fragment.startswith("w=")

    arrived = Session()
    Files(page, arrived, again(page, hash=page.fragment)).shared()
    assert [entry.text for entry in arrived.entries] == [
        entry.text for entry in session.entries
    ]


def test_a_link_is_a_file_in_the_store_like_any_other(page, storage):
    """So that Transfer Save writes it back out and the status line names it."""
    Files(page, _session("x + 1"), storage).linked()

    returning = again(page, hash=page.fragment)
    arrived = Session()
    Files(page, arrived, returning).shared()
    assert returning.exists(Path(gestures.SHARED))
    assert arrived.file == Path(gestures.SHARED)


def test_a_worksheet_too_long_for_a_link_is_refused_in_words(page, storage):
    """And the refusal names the size, since nothing can be done to shrink it."""
    session = _session(*["x^2 - 4 + sin(x)/x"] * 500)
    Files(page, session, storage).linked()
    assert page.fragment is None
    assert page.notices[-1].startswith("This worksheet is ")
    assert "too long for a link" in page.notices[-1]


def test_an_address_carrying_nothing_readable_says_so(page):
    """A link cut in half on its way here is the ordinary way this happens."""
    storage = again(page, hash="w=this-is-not-a-worksheet")
    assert Files(page, Session(), storage).shared() == gestures.UNREADABLE_LINK


def test_an_address_with_nothing_in_it_is_the_ordinary_visit(page, storage):
    session = Session()
    assert Files(page, session, storage).opening() == ""
    assert session.entries == []


def test_a_tab_that_will_remember_nothing_says_so_before_anything_else(page):
    """The one thing about a private tab a user cannot find out by looking."""
    import js

    del js.localStorage
    files = Files(page, Session(), WebStorage(page))
    assert files.opening() == gestures.FORGETFUL


def test_a_link_the_clipboard_refuses_still_leaves_the_address(page, storage):
    """The address is the link; the clipboard is a convenience on top of it."""
    import js

    js.navigator.clipboard = Clipboard(allow=False)
    Files(page, _session("x"), storage).linked()
    assert page.fragment.startswith("w=")
    assert page.notices[-1].endswith("the address bar carries the link either way")


# -- the settings -------------------------------------------------------------


def test_the_settings_are_written_when_the_tab_goes_and_read_when_it_comes_back(
    page, storage
):
    """What a tab has instead of a state file, and the whole of how it remembers."""
    import js

    session = Session()
    session.settings.assign("Notation", "Decimal")
    Files(page, session, storage).remember()
    assert js.localStorage.getItem(KEPT) is not None

    returning = Session()
    Files(page, returning, again(page)).settings()
    assert returning.settings["Notation"] == "Decimal"


def test_a_first_visit_has_no_settings_to_read_and_says_nothing_about_it(page, storage):
    session = Session()
    Files(page, session, storage).settings()
    assert page.notices == []
    assert session.settings["Notation"] == "Rational"


def test_the_settings_are_kept_and_not_downloaded(page, storage):
    """A tab hidden and shown again would otherwise rain files on the user."""
    Files(page, Session(), storage).remember()
    assert page.downloads == []


# -- what a file arriving is told to itself -----------------------------------


def test_a_dropped_file_names_the_command_that_reads_it(page, storage):
    """It arrived without being asked for, so the notice says what to do with it."""
    Files(page, Session(), storage).arrived("theirs.mth", "x^3\n")
    assert storage.read(Path("theirs.mth")) == b"x^3\n"
    assert "theirs.mth" in page.notices[-1]
    assert "Transfer Load Derive" in page.notices[-1]
