"""File handling in the browser, tested without one: the store, downloads, links, and
the file dialog of a command that reads a file.

A browser has no filesystem, only the store, downloads and the link, so these tests
check which of them a file went through and what the user was told. All the Python is
real: the store mapping names to files, the rule that a save is downloaded before it
is stored, the link encoding of a worksheet, and the message for a refused clipboard.

The tab is faked only as far as the code uses it (a store with a size limit, a
clipboard that can refuse, an address). No browser is started.
"""

import json
from pathlib import Path

import pytest
from dfwsheet import Sheet
from fakepage import Clipboard, FakeTab, Store, browser
from screen import band, message, prompt

from rederive import platform
from rederive.model import worksheet
from rederive.model.session import Session
from rederive.platform.web import CLIPBOARD_REFUSED, KEPT, Web, WebStorage
from rederive.ui.app import CHOOSING_FILE, ENTER_OPTION, RederiveApp
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
    assert files.opening() == ""
    assert files.greeting() == gestures.FORGETFUL


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
    files.welcome()
    assert page.notices == []
    assert files.opening() == ""
    assert session.settings["Notation"] == "Rational"


def test_a_visit_that_restores_changed_settings_says_so_as_it_opens(page, storage):
    """Without a notice, a mode switched on in an earlier visit would look like a bug.

    The notice says where the settings came from and how to reset them. It goes to the
    page rather than the message line. Stored settings equal to the defaults produce no
    notice.
    """
    session = Session()
    Files(page, session, storage).remember()
    files = Files(page, Session(), again(page))
    files.settings()
    assert files.greeting() == ""

    session = Session()
    session.settings.assign("Angle", "Degree")
    Files(page, session, storage).remember()
    files = Files(page, Session(), again(page))
    files.settings()
    assert files.opening() == ""
    files.welcome()
    assert page.notices == [gestures.RESTORED]
    assert "Reset settings" in gestures.RESTORED


def test_a_link_and_restored_settings_are_both_said(page, storage):
    session = Session()
    session.settings.assign("Angle", "Degree")
    Files(page, session, storage).remember()
    text = "x^2\n"
    fragment = gestures.FIELD + gestures._encoded(text)
    files = Files(page, Session(), again(page, hash=fragment))
    files.settings()
    assert files.opening().startswith("1 expression from the link")
    assert files.greeting() == gestures.RESTORED


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


def test_a_reset_is_undone_like_a_command(page, storage):
    session = Session()
    session.settings.assign("Angle", "Degree")
    session.order = ("z", "y")
    files = Files(page, session, storage)
    files.reset()
    assert files.history.undo().description == "Reset settings"
    assert session.settings["Angle"] == "Degree"
    assert session.order == ("z", "y")


# -- files arriving unrequested -----------------------------------------------


def test_a_dropped_file_says_what_finds_it(page, storage):
    """It arrived unrequested, so the notice says what it is there for."""
    Files(page, Session(), storage).arrived("theirs.mth", b"x^3\n")
    assert storage.read(Path("theirs.mth")) == b"x^3\n"
    assert page.notices[-1] == gestures.ARRIVED.format(name="theirs.mth")


# -- worksheets of Derive for Windows, which are binary -------------------------

DFW = Path("work.dfw")


def _worksheet(*expressions):
    sheet = Sheet()
    for expression in expressions:
        sheet.expression(expression)
    return sheet.bytes()


def test_a_dropped_worksheet_is_kept_as_its_bytes_and_loads(page, storage):
    """As on the desktop: the bytes Load reads are the bytes that were dropped."""
    data = _worksheet("x^2", "#1+1")
    Files(page, Session(), storage).arrived(DFW.name, data)
    assert storage.read(DFW) == data
    assert page.notices[-1] == gestures.ARRIVED.format(name=DFW.name)
    session = Session()
    session.load(DFW)
    assert [entry.text for entry in session.entries] == ["x^2", "#1 + 1"]


def test_a_worksheet_is_remembered_by_the_next_visit(page, storage):
    data = _worksheet("x")
    Files(page, Session(), storage).arrived(DFW.name, data)
    returning = again(page)
    assert returning.read(DFW) == data
    session = Session()
    session.merge(DFW)
    assert [entry.text for entry in session.entries] == ["x"]


def test_a_worksheet_too_large_to_remember_still_loads_now(page):
    """A worksheet with plot images can outgrow the whole store.

    It stays for this visit and the user is told the next one won't have it. Leaving it
    out of what is written keeps the store working for every other file.
    """
    import js

    js.localStorage = Store(room=2000)
    platform.use(Web(page))
    storage = platform.current().storage()
    files = Files(page, Session(), storage)
    files.arrived("small.mth", b"x + 1\n")
    files.arrived(DFW.name, _worksheet(*["x^2 + 1"] * 200))
    assert "no room to remember it" in page.notices[-1]
    session = Session()
    session.load(DFW)
    assert len(session.entries) == 200
    storage.write(SHEET, "y\n")
    returning = again(page)
    assert not returning.exists(DFW)
    assert returning.read(Path("small.mth")) == b"x + 1\n"
    assert returning.read(SHEET) == b"y\n"


def test_a_store_from_before_binary_files_still_reads(page):
    """An earlier visit wrote every file as a plain string, and they all still read."""
    import js

    store = Store()
    store.setItem(KEPT, json.dumps({"old.mth": "x + 1\n", "odd": 7}))
    js.localStorage = store
    returning = again(page)
    assert returning.read(Path("old.mth")) == b"x + 1\n"
    assert not returning.exists(Path("odd"))


def test_a_dos_math_file_is_decoded_as_on_the_desktop(page, storage):
    """Code page 437, where `\\xe3` is pi, as Derive for DOS wrote it."""
    raw = b"\xe3 + 1\r\n"
    Files(page, Session(), storage).arrived("old.mth", raw)
    assert storage.read(Path("old.mth")).decode("utf-8") == worksheet.decoded(raw)
    session = Session()
    session.load(Path("old.mth"))
    assert [entry.text for entry in session.entries] == ["pi + 1"]


def test_a_worksheet_is_linked_as_the_math_file_it_saves_as(page, storage):
    """A link carries what Save writes: the expressions, as text."""
    Files(page, Session(), storage).arrived(DFW.name, _worksheet("x^2"))
    session = Session()
    session.load(DFW)
    assert session.file == Path("work.mth")
    Files(page, session, storage).linked()
    arrived = Session()
    Files(page, arrived, again(page, hash=page.fragment)).shared()
    assert [entry.text for entry in arrived.entries] == ["x^2"]


# -- the file dialog of a command that reads a file -------------------------------


def _app(page, storage):
    """The app with the page's file operations attached, as `web.boot` does."""
    app = RederiveApp()
    files = Files(page, app.session, storage)
    files.attend()
    app.chooser = files.choose
    app.unfound = files.unfound
    return app


def _prompted(app):
    """Whether the file prompt is up."""
    return app.query_one("#prompt-band").display


async def test_load_opens_the_dialog_and_loads_the_file_picked(page, storage):
    """The command asks the page for a file instead of a name, and the file picked is
    stored under its name and loaded as if that name had been typed."""
    app = _app(page, storage)
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        assert page.dialogs == [".mth,.dfw,text/plain"]
        assert not _prompted(app)
        assert message(app) == CHOOSING_FILE
        page.handlers["picked"]("theirs.mth", b"x^2\n#1 + 1\n")
        await pilot.pause()
        assert [entry.text for entry in app.session.entries] == ["x^2", "#1 + 1"]
        assert app.session.file == Path("theirs.mth")
        assert storage.read(Path("theirs.mth")) == b"x^2\n#1 + 1\n"
        assert not _prompted(app)
        assert band(app)[0].startswith(" COMMAND:")
        assert message(app) == ENTER_OPTION
        # Loading tells the user nothing about the page: the file is already loaded.
        assert page.notices == []


async def test_a_picked_worksheet_of_derive_for_windows_loads_as_its_bytes(page, storage):
    app = _app(page, storage)
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        page.handlers["picked"](DFW.name, _worksheet("x^3"))
        await pilot.pause()
        assert [entry.text for entry in app.session.entries] == ["x^3"]


async def test_a_picked_file_without_an_extension_loads(page, storage):
    """At the prompt a name without an extension gets the command's, so the file is
    read, and stored, under that name, rather than not found under its own."""
    app = _app(page, storage)
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        page.handlers["picked"]("notes", b"x + 2\n")
        await pilot.pause()
        assert [entry.text for entry in app.session.entries] == ["x + 2"]
        assert not _prompted(app)
        assert message(app) == ENTER_OPTION
        assert storage.read(Path("notes.mth")) == b"x + 2\n"


@pytest.mark.parametrize(
    ("keys", "accepted"),
    [
        (("t", "m"), ".mth,.dfw,text/plain"),
        (("t", "l", "s"), ".ini,text/plain"),
        (("t", "l", "t"), ".dat,text/plain"),
        (("t", "l", "u"), ".mth,.dfw,text/plain"),
        (("t", "d", "o"), ".dmo,text/plain"),
    ],
)
async def test_every_command_that_reads_a_file_opens_the_dialog(
    page, storage, keys, accepted
):
    app = _app(page, storage)
    async with app.run_test() as pilot:
        await pilot.press(*keys)
        assert page.dialogs == [accepted]
        assert not _prompted(app)


async def test_a_command_that_writes_a_file_still_asks_for_its_name(page, storage):
    """A save goes out as a download, which the name typed names."""
    app = _app(page, storage)
    async with app.run_test() as pilot:
        authored = app.session
        authored.author("x")
        await pilot.press("t", "s", "d")
        assert page.dialogs == []
        assert prompt(app)[0] == "TRANSFER SAVE DERIVE file:"


async def test_a_dismissed_dialog_lands_where_esc_at_the_prompt_does(page, storage):
    app = _app(page, storage)
    async with app.run_test() as pilot:
        page.opens = False
        await pilot.press("t", "l", "d", "escape")
        typed = (band(app), message(app))
        await pilot.press("escape", "escape")
        page.opens = True
        await pilot.press("t", "l", "d")
        page.handlers["dismissed"]()
        await pilot.pause()
        assert (band(app), message(app)) == typed
        assert not _prompted(app)
        assert app.session.entries == []


async def test_a_dialog_that_will_not_open_leaves_the_name_prompt(page, storage):
    """Without user activation, or `showPicker`, the page refuses, and the command asks
    for the name as on the desktop, finding a file added to the page by it."""
    Files(page, Session(), storage).arrived("theirs.mth", b"y\n")
    page.opens = False
    app = _app(page, storage)
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        assert prompt(app)[0] == "TRANSFER LOAD DERIVE file:"
        await pilot.press(*"theirs", "enter")
        assert [entry.text for entry in app.session.entries] == ["y"]


async def test_a_file_picked_after_the_program_moved_on_is_only_kept(page, storage):
    """A dialog that never said it was dismissed leaves no command waiting on it."""
    app = _app(page, storage)
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d", "escape", "escape", "a")
        page.handlers["picked"]("late.mth", b"z\n")
        await pilot.pause()
        assert app.session.entries == []
        assert storage.read(Path("late.mth")) == b"z\n"
        assert page.notices == [gestures.ARRIVED.format(name="late.mth")]


async def test_a_picked_file_that_cannot_be_read_leaves_its_name_up(page, storage):
    """As a typed name does, for correction."""
    app = _app(page, storage)
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "u")
        page.handlers["picked"]("bad.mth", b"x\n")
        storage.read = lambda path: (_ for _ in ()).throw(PermissionError(path))
        await pilot.pause()
        assert prompt(app) == ("TRANSFER LOAD UTILITY file:", "bad.mth")
        assert message(app) == "Cannot read file"


# -- the files a worksheet loads ------------------------------------------------


async def test_a_worksheet_whose_load_misses_a_file_says_how_to_supply_it(page, storage):
    """The message line says what could not be found, and the page says how to add
    it, for long enough to follow."""
    app = _app(page, storage)
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        page.handlers["picked"]("main.mth", b'LOAD("other.mth")\nx\n')
        await pilot.pause()
        assert message(app) == "other.mth could not be found"
        assert page.notices == [
            "other.mth is not in this page. Drop it on the page or add it with"
            " File > Add file..., then load again"
        ]
        assert page.kept == [True]
        # Supplied, the same worksheet loads it.
        Files(page, app.session, storage).arrived("other.mth", b"f(x) := x^2\n")
        await pilot.press("t", "l", "d")
        page.handlers["picked"]("main.mth", b'LOAD("other.mth")\nx\n')
        await pilot.pause()
        assert message(app) == ENTER_OPTION
        assert len(page.notices) == 2


async def test_a_load_typed_at_the_author_line_says_how_to_supply_its_file(
    page, storage
):
    app = _app(page, storage)
    async with app.run_test() as pilot:
        await pilot.press("a", *'LOAD("a.mth")', "enter")
        assert message(app) == "a.mth could not be found"
        assert page.notices == [
            "a.mth is not in this page. Drop it on the page or add it with"
            " File > Add file..., then load again"
        ]


async def test_every_file_a_load_missed_is_named_once(page, storage):
    app = _app(page, storage)
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d")
        page.handlers["picked"](
            "main.mth", b'LOAD("a.mth")\nLOAD("b.mth")\nLOAD("a.mth")\n'
        )
        await pilot.pause()
        assert page.notices == [
            "a.mth and b.mth are not in this page. Drop them on the page or add them"
            " with File > Add file..., then load again"
        ]
