"""File handling in the browser, tested without one: the store, downloads, and links.

A browser has no filesystem, only the store, downloads and the link, so these tests
check which of them a file went through and what the user was told. All the Python is
real: the store mapping names to files, the rule that a save is downloaded before it
is stored, the link encoding of a worksheet, and the message for a refused clipboard.

The tab is faked only as far as the code uses it (a store with a size limit, a
clipboard that can refuse, an address). No browser is started.
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
    """Install the browser platform as `web.boot` does, restoring the desktop after.

    Needed because creating a link uses the platform's clipboard, which on the desktop
    platform would be the real one.
    """
    browser()
    platform.use(Web(page))
    yield
    platform.use(None)


@pytest.fixture
def storage():
    return platform.current().storage()


def again(page, hash=""):
    """Simulate a reload: a new environment over the same store.

    MEMFS, the session and the storage object are discarded; only localStorage and the
    address survive.
    """
    import js

    browser(store=js.localStorage, hash=hash)
    platform.use(Web(page))
    return platform.current().storage()


# -- the store ----------------------------------------------------------------


def test_a_save_leaves_the_tab_and_is_remembered_by_name(storage, page):
    """A save downloads the file first, then stores it under its name.

    The download is the only copy the user can keep; the store lets the name be
    opened again later.
    """
    storage.write(SHEET, "x^2 - 4\n")
    assert page.downloads == [("work.mth", "x^2 - 4\n")]
    assert storage.read(SHEET) == b"x^2 - 4\n"
    assert storage.exists(SHEET)


def test_what_the_store_holds_outlives_the_page_that_wrote_it(page, storage):
    """The reason for a persistent store: MEMFS doesn't survive to the next visit."""
    storage.write(SHEET, "sin(x)\n")
    assert again(page).read(SHEET) == b"sin(x)\n"


def test_a_store_with_no_room_costs_the_name_and_not_the_file(page):
    """When the store is full, the worksheet is still saved.

    The download has already happened when the store refuses, so only the stored name
    is lost, and the user is told so.
    """
    import js

    js.localStorage = Store(room=10)
    WebStorage(page).write(SHEET, "a very long worksheet indeed\n")
    assert page.downloads == [("work.mth", "a very long worksheet indeed\n")]
    assert page.notices == [
        "work.mth was downloaded, but this page has no room left to remember it by name"
    ]


def test_a_file_handed_over_is_kept_and_never_sent_back_out(storage, page):
    """A picked or dropped file is already on disk, so it isn't downloaded again."""
    storage.keep(Path("theirs.mth"), "x + 1\n")
    assert storage.read(Path("theirs.mth")) == b"x + 1\n"
    assert page.downloads == []


def test_a_file_on_the_filesystem_is_read_when_the_store_has_none(storage, tmp_path):
    """MEMFS is the fallback for a path the store doesn't hold."""
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
    """File prompt completion lists both the store and the filesystem.

    A tab's working directory is usually empty (MEMFS holds just the package), so the
    store is nearly all of it. The filesystem is included so that a full path finds
    worksheets shipped in the wheel.
    """
    monkeypatch.chdir(tmp_path)
    storage.keep(Path("second.mth"), "")
    storage.keep(Path("first.mth"), "")
    assert list(storage.names(Path())) == ["first.mth", "second.mth"]
    (tmp_path / "onthedisk.mth").write_text("")
    assert list(storage.names(Path())) == ["first.mth", "onthedisk.mth", "second.mth"]
    # A file stored under a full path is listed by name in its directory.
    storage.keep(tmp_path / "elsewhere.mth", "")
    assert list(storage.names(tmp_path)) == ["elsewhere.mth", "onthedisk.mth"]


def test_a_store_that_will_not_be_read_leaves_the_program_running(page):
    """With site data blocked, nothing is stored, but the program still runs."""
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
    """A refused clipboard write is always reported.

    The three failures (a rejected promise, a throwing call, no clipboard at all) get
    the same message, since the user can't act on the difference.
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
    """A worksheet survives the round trip through a permalink."""
    session = _session("x^2 - 4", "sin(x)/x")
    Files(page, session, storage).linked()
    assert page.fragment.startswith("w=")

    arrived = Session()
    Files(page, arrived, again(page, hash=page.fragment)).shared()
    assert [entry.text for entry in arrived.entries] == [
        entry.text for entry in session.entries
    ]


def test_a_link_is_a_file_in_the_store_like_any_other(page, storage):
    """So Transfer Save can write it back and the status line shows its name."""
    Files(page, _session("x + 1"), storage).linked()

    returning = again(page, hash=page.fragment)
    arrived = Session()
    Files(page, arrived, returning).shared()
    assert returning.exists(Path(gestures.SHARED))
    assert arrived.file == Path(gestures.SHARED)


def test_a_worksheet_too_long_for_a_link_is_refused_in_words(page, storage):
    """The refusal names the size, since the user can't shrink the link."""
    session = _session(*["x^2 - 4 + sin(x)/x"] * 500)
    Files(page, session, storage).linked()
    assert page.fragment is None
    assert page.notices[-1].startswith("This worksheet is ")
    assert "too long for a link" in page.notices[-1]


def test_an_address_carrying_nothing_readable_says_so(page):
    """Typically caused by a truncated link."""
    storage = again(page, hash="w=this-is-not-a-worksheet")
    assert Files(page, Session(), storage).shared() == gestures.UNREADABLE_LINK


def test_an_address_with_nothing_in_it_is_the_ordinary_visit(page, storage):
    session = Session()
    assert Files(page, session, storage).opening() == ""
    assert session.entries == []


def test_a_tab_that_will_remember_nothing_says_so_before_anything_else(page):
    """A private tab is announced first, since the user can't otherwise tell."""
    import js

    del js.localStorage
    files = Files(page, Session(), WebStorage(page))
    assert files.opening() == gestures.FORGETFUL


def test_a_link_the_clipboard_refuses_still_leaves_the_address(page, storage):
    """The address is the link; copying it to the clipboard is a convenience."""
    import js

    js.navigator.clipboard = Clipboard(allow=False)
    Files(page, _session("x"), storage).linked()
    assert page.fragment.startswith("w=")
    assert page.notices[-1].endswith("the address bar carries the link either way")


# -- the settings -------------------------------------------------------------


def test_the_settings_are_written_when_the_tab_goes_and_read_when_it_comes_back(
    page, storage
):
    """localStorage replaces the desktop's state file."""
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
    files = Files(page, session, storage)
    files.settings()
    assert page.notices == []
    assert files.opening() == ""
    assert session.settings["Notation"] == "Rational"


def test_a_visit_that_restores_changed_settings_says_so_as_it_opens(page, storage):
    """Without a notice, a mode switched on in an earlier visit would look like a bug.

    The notice says where the settings came from and how to reset them. Stored settings
    equal to the defaults produce no notice.
    """
    session = Session()
    Files(page, session, storage).remember()
    files = Files(page, Session(), again(page))
    files.settings()
    assert files.opening() == ""

    session = Session()
    session.settings.assign("Angle", "Degree")
    Files(page, session, storage).remember()
    files = Files(page, Session(), again(page))
    files.settings()
    assert files.opening() == gestures.RESTORED
    assert "Reset settings" in gestures.RESTORED


def test_a_link_and_restored_settings_are_both_said(page, storage):
    session = Session()
    session.settings.assign("Angle", "Degree")
    Files(page, session, storage).remember()
    text = "x^2\n"
    fragment = gestures.FIELD + gestures._encoded(text)
    files = Files(page, Session(), again(page, hash=fragment))
    files.settings()
    opening = files.opening()
    assert opening.startswith("1 expression from the link")
    assert opening.endswith(gestures.RESTORED)


def test_the_settings_are_kept_and_not_downloaded(page, storage):
    """Otherwise every time the tab was hidden it would download a file."""
    Files(page, Session(), storage).remember()
    assert page.downloads == []


def test_a_reset_puts_the_settings_back_for_this_visit_and_the_next(page, storage):
    """Reset restores the defaults now and stores them for the next visit.

    Otherwise a mode switched on in an earlier visit would persist. The defaults are
    stored immediately, without waiting for the tab to be hidden.
    """
    session = Session()
    session.settings.assign("Angle", "Degree")
    session.order = ("z", "y")
    files = Files(page, session, storage)
    files.remember()
    files.reset()
    assert session.settings["Angle"] == "Radian"
    assert session.order == ("x", "y", "z")
    assert "default" in page.notices[-1]
    assert page.downloads == []

    returning = Session()
    Files(page, returning, again(page)).settings()
    assert returning.settings["Angle"] == "Radian"
    assert returning.order == ("x", "y", "z")


# -- files arriving unrequested -----------------------------------------------


def test_a_dropped_file_names_the_command_that_reads_it(page, storage):
    """It arrived unrequested, so the notice says how to open it."""
    Files(page, Session(), storage).arrived("theirs.mth", "x^3\n")
    assert storage.read(Path("theirs.mth")) == b"x^3\n"
    assert "theirs.mth" in page.notices[-1]
    assert "Transfer Load Derive" in page.notices[-1]
