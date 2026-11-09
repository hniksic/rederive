"""The original's demonstrations: the catalog, and the page code that fetches them.

The catalog lists nine files and where to download them. It lives in Python because a
desktop menu needs the same files. The tests check that galleries are flagged correctly
and that every entry is available from every source.

The page side covers what happens once a file has arrived: it is stored in the page's
storage under its original name, and the app is told to run it as a script or a
gallery. The browser parts (menu, download, fallback relay) are `web/demos.js` and are
not tested here. The app is faked only as deep as this code touches it; running a
demonstration is tested with a real app in `test_transfer` and `test_plot`.

Further sections cover `Transfer Demo` in a page, which has to download through the
page because a tab has no sockets, and the desktop's download paths, with `urlopen`
answered from a table.
"""

import re
import time
import zipfile
from functools import partial
from io import BytesIO
from pathlib import Path

import pytest
from fakepage import FakeTab, browser, jsnull

from rederive import fetch, platform
from rederive.model import demos as catalogue
from rederive.model import worksheet
from rederive.platform.web import Web
from rederive.ui.app import RederiveApp
from rederive.web.boot import PageApp
from rederive.web.demos import Demos

SCRIPT = b"; adds two numbers\r\n2 + 3\r\n\x1a"
GALLERY = b'"A spike:"\r\n\r\n2*SIN(LN(ABS(x)))\r\n\x1a'


# -- the catalogue ------------------------------------------------------------


def test_every_demonstration_is_named_by_the_file_it_is():
    """A demonstration is named by its file name, since that is what it is saved as."""
    assert catalogue.named("ALGEBRA.DMO").title == "Algebra"
    assert catalogue.named("algebra.dmo") is None
    assert catalogue.named("NOTHING.DMO") is None


def test_the_galleries_are_the_files_whose_steps_are_drawn():
    drawn = [demo.file for demo in catalogue.DEMOS if demo.plotting]
    assert drawn == ["PLOT2D.MTH", "PLOTPARA.MTH", "PLOT3D.MTH"]
    # The extension agrees with the flag: a gallery's steps are read as worksheet
    # lines, not as a script's.
    for demo in catalogue.DEMOS:
        expected = worksheet.SUFFIX if demo.plotting else worksheet.DEMO_SUFFIX
        assert demo.file.lower().endswith(expected)


def test_every_demonstration_can_be_had_from_every_source():
    for demo in catalogue.DEMOS:
        assert len(demo.urls) == len(catalogue.SOURCES)
        for url in demo.urls:
            assert url.startswith("https://")
            assert url.endswith(demo.file)


# -- what the two sides promise each other ------------------------------------


def test_the_page_module_exports_everything_python_calls_on_it():
    """The fake page accepts any call, so the real module's exports must be checked.

    Python calls into the `demos.js` module by name. A missing export would go unnoticed
    by tests using the fake page, and in a browser would raise with the demonstration
    half started.
    """
    source = (Path(__file__).resolve().parents[1] / "web" / "demos.js").read_text()
    pattern = r"^export (?:async\s+)?(?:function\s+(\w+)|\{([^}]*)\})"
    exported = set()
    for name, group in re.findall(pattern, source, re.M):
        exported.update([name] if name else (word.strip() for word in group.split(",")))
    # `wire` is used by the page itself; the other three are called from Python.
    assert {"wire", "attend", "said", "brought"} <= exported


# -- the page's menu ----------------------------------------------------------


@pytest.fixture
def page():
    return FakeTab()


class FakeApp:
    """The app as the menu sees it: whether it is running, and a way to hand it a file.

    The menu enters the app through `call_next`, since a click arrives on a DOM callback
    and the screen belongs to the event loop. Here it runs the call immediately, which
    keeps the tests simple; with `deferring` it queues the call instead, for the test
    about the handoff itself.
    """

    def __init__(self, running=True, deferring=False):
        self.is_running = running
        self.deferring = deferring
        self.demonstrated = []
        self.waiting = []

    def call_next(self, call, *arguments):
        waiting = partial(call, *arguments)
        if self.deferring:
            self.waiting.append(waiting)
            return
        waiting()

    def pump(self):
        """Run the queued calls, as the loop would."""
        waiting, self.waiting = self.waiting, []
        for call in waiting:
            call()

    def demonstrate(self, name, plotting=False):
        self.demonstrated.append((name, plotting))


@pytest.fixture
def app():
    return FakeApp()


@pytest.fixture(autouse=True)
def environment(page):
    browser()
    platform.use(Web(page))
    yield
    platform.use(None)


@pytest.fixture
def demos(page, app):
    made = Demos(page, app, platform.current().storage())
    made.attend()
    return made


def chose(page, name, data=None):
    """Choose a demonstration in the menu, calling the handler JS calls."""
    page.handlers["chose"](name, data)


def kept(page, name):
    return page.handlers["kept"](name)


def keep(page, name, data):
    page.handlers["keep"](name, data)


def test_the_page_is_handed_the_menu_it_is_to_build(demos, page):
    offered = page.handlers["demos"]
    assert [entry["title"] for entry in offered] == [
        demo.title for demo in catalogue.DEMOS
    ]
    # Each entry carries its URLs: Python decides where the file is, but only the page
    # can fetch it.
    assert offered[0]["urls"] == list(catalogue.DEMOS[0].urls)
    # Entries also carry the menu group headings, in order.
    assert [entry["group"] for entry in offered[:1] + offered[-1:]] == [
        catalogue.SCRIPTS,
        catalogue.GALLERIES,
    ]


def test_the_app_is_entered_through_its_loop_and_not_from_the_click(page):
    """A click arrives on a DOM callback, but the screen belongs to the event loop.

    Started directly from the callback, a demonstration would run but never be drawn;
    plot pane events cross to the loop for the same reason.
    """
    clicked = FakeApp(deferring=True)
    Demos(page, clicked, platform.current().storage()).attend()
    chose(page, "ALGEBRA.DMO", SCRIPT)
    assert clicked.demonstrated == []
    clicked.pump()
    assert clicked.demonstrated == [("ALGEBRA.DMO", False)]


def test_a_script_lands_in_the_page_and_runs(demos, page, app):
    chose(page, "ALGEBRA.DMO", SCRIPT)
    assert app.demonstrated == [("ALGEBRA.DMO", False)]
    # It is then stored under its own name, which Transfer Demo can use to run it again.
    assert worksheet.demonstration(Path("ALGEBRA.DMO")) == (
        ("adds two numbers", "2 + 3"),
    )


def test_a_gallery_runs_as_the_kind_that_is_drawn(demos, page, app):
    """The kind comes from the catalog, never from the page."""
    chose(page, "PLOT2D.MTH", GALLERY)
    assert app.demonstrated == [("PLOT2D.MTH", True)]
    assert worksheet.demonstration(Path("PLOT2D.MTH")) == (
        ("A spike:", "2*SIN(LN(ABS(x)))"),
    )


def test_the_page_is_told_where_the_whole_diskette_is(demos, page):
    """The page gets the diskette image URL, which bypasses the archive's extractor."""
    assert page.handlers["image"] == catalogue.IMAGE


def test_a_demonstration_can_be_kept_without_being_run(demos, page, app):
    """The other eight files from a downloaded diskette are kept without running them."""
    keep(page, "PLOT2D.MTH", GALLERY)
    assert app.demonstrated == []
    assert kept(page, "PLOT2D.MTH")
    # Choosing it later runs it from storage.
    chose(page, "PLOT2D.MTH")
    assert app.demonstrated == [("PLOT2D.MTH", True)]


def test_nothing_is_kept_under_a_name_that_is_on_no_list(demos, page):
    keep(page, "MALICE.DMO", SCRIPT)
    assert not platform.current().storage().exists(Path("MALICE.DMO"))


def test_a_demonstration_the_page_has_is_run_without_a_download(demos, page, app):
    """Storage outlives the tab, so a second viewing needs no download.

    The page then passes JS `null`, which is falsy but is not `None`; a check by identity
    would treat it as bytes and try to read from it.
    """
    assert not kept(page, "ALGEBRA.DMO")
    chose(page, "ALGEBRA.DMO", SCRIPT)
    assert kept(page, "ALGEBRA.DMO")
    chose(page, "ALGEBRA.DMO", jsnull)
    assert app.demonstrated == [("ALGEBRA.DMO", False)] * 2
    assert worksheet.demonstration(Path("ALGEBRA.DMO")) == (
        ("adds two numbers", "2 + 3"),
    )


def test_nothing_is_kept_for_a_name_that_is_on_no_list(demos, page):
    assert not kept(page, "MALICE.DMO")


def test_a_name_that_is_on_no_list_runs_nothing(demos, page, app):
    chose(page, "MALICE.DMO", SCRIPT)
    assert app.demonstrated == []
    assert page.notices == ["MALICE.DMO is not one of the original's demonstrations"]
    assert not platform.current().storage().exists(Path("MALICE.DMO"))


def test_the_notice_names_the_command_that_runs_it_again(demos, page):
    chose(page, "ALGEBRA.DMO", SCRIPT)
    chose(page, "PLOT2D.MTH", GALLERY)
    # Demo reads a script and Load reads a worksheet, and a gallery is a worksheet, so
    # each notice must name the right command.
    assert page.notices == [
        "ALGEBRA.DMO is in this page now; Transfer Demo runs it again by that name",
        "PLOT2D.MTH is in this page now; Transfer Load Derive reads it by that name",
    ]


def test_a_file_the_original_wrote_is_read_in_its_own_encoding(demos, page):
    """The original's files are in code page 437, including their Greek letters."""
    chose(page, "ARITH.DMO", "; the angle α\r\nSIN(x)\r\n\x1a".encode("cp437"))
    assert worksheet.demonstration(Path("ARITH.DMO")) == (("the angle α", "SIN(x)"),)


def test_a_demonstration_asked_for_too_early_is_refused_in_words(page):
    """The button appears on the first frame, before the program is running."""
    starting = FakeApp(running=False)
    Demos(page, starting, platform.current().storage()).attend()
    chose(page, "ALGEBRA.DMO", SCRIPT)
    assert starting.demonstrated == []
    assert page.notices == [
        "Rederive is still starting; try the demonstration again in a moment"
    ]
    # Nothing is kept; the file is downloaded again when the user retries.
    assert not platform.current().storage().exists(Path("ALGEBRA.DMO"))


# -- the command that offers the same nine ------------------------------------
#
# `Transfer Demo` is available in a page too, but a tab has no sockets, so the command
# fetches through the page and keeps the file in page storage.


async def test_the_command_downloads_through_the_page(demos, page, arith):
    page.archive["ARITH.DMO"] = SCRIPT
    assert await demos.brought(arith) == SCRIPT
    assert page.fetched == ["ARITH.DMO"]
    # It is stored under its original name, so it isn't downloaded twice and the
    # Transfer commands can find it.
    assert kept(page, "ARITH.DMO")
    assert worksheet.demonstration(Path("ARITH.DMO")) == (
        ("adds two numbers", "2 + 3"),
    )
    # Nothing is downloaded to the user's disk, as a save in a page would be.
    assert page.downloads == []


async def test_a_download_the_page_refuses_says_so_in_one_sentence(demos, arith):
    """The message line shows a one-sentence error, not the browser's stack trace.

    The fake page offers nothing, as when every route to the archive is down.
    """
    with pytest.raises(OSError, match="^the server said 503$"):
        await demos.brought(arith)
    assert not platform.current().storage().exists(Path("ARITH.DMO"))


def test_the_pages_opening_notice_points_at_the_button_and_not_the_keystrokes():
    """In a page, demonstrations are a click away, so the greeting doesn't suggest T D."""
    assert "T D" in RederiveApp.GREETING_LINE
    assert "T D" not in PageApp.GREETING_LINE
    assert "help" in PageApp.GREETING_LINE


# -- the desktop's own download -----------------------------------------------
#
# The desktop counterpart of `web/demos.js`: downloading a demonstration that is not on
# disk. `urlopen` is answered from a table, so these tests cover the fallback paths and
# validity checks without opening sockets.


class Answer:
    """The part of a `urlopen` response that `fetch` uses."""

    def __init__(self, raw: bytes) -> None:
        self.raw = raw

    def read(self, size: int = -1) -> bytes:
        return self.raw if size < 0 else self.raw[:size]

    def __enter__(self) -> "Answer":
        return self

    def __exit__(self, *_: object) -> bool:
        return False


def serving(held, slow=()):
    """A fake `urlopen` serving `held` by URL and raising OSError for other URLs.

    URLs in `slow` answer after a short delay, so a test can control who wins a race.
    """

    def opened(url, timeout=None):
        if url in slow:
            time.sleep(0.05)
        if url not in held:
            raise OSError(f"the server said 503: {url}")
        return Answer(held[url])

    return opened


def diskette(files):
    """A zip laid out like the archive's, with the files inside a directory."""
    raw = BytesIO()
    with zipfile.ZipFile(raw, "w") as image:
        for name, data in files.items():
            image.writestr(f"DERIVE_3.14/{name}", data)
    return raw.getvalue()


@pytest.fixture
def arith():
    return catalogue.named("ARITH.DMO")


def test_the_first_source_to_answer_is_the_one_the_file_comes_off(
    monkeypatch, arith
):
    """Both sources are asked at once, so a slow one costs nothing if the other is up."""
    first, second = arith.urls
    monkeypatch.setattr(fetch, "urlopen", serving({second: SCRIPT}, slow=[first]))
    assert fetch.fetched(arith) == SCRIPT


def test_a_source_that_refuses_is_passed_over_for_one_that_does_not(
    monkeypatch, arith
):
    monkeypatch.setattr(fetch, "urlopen", serving({arith.urls[1]: SCRIPT}))
    assert fetch.fetched(arith) == SCRIPT


@pytest.mark.parametrize(
    "answered",
    [b"", b"<html>the archive is sorry</html>", b"x" * (fetch.LARGEST_FILE + 1)],
    ids=["nothing", "an apology", "something far too big"],
)
def test_what_comes_back_has_to_be_a_file_of_the_originals(
    monkeypatch, arith, answered
):
    """A relay may serve its own error page with status 200, so the content is checked."""
    held = dict.fromkeys(arith.urls, answered)
    monkeypatch.setattr(fetch, "urlopen", serving(held))
    with pytest.raises(OSError, match="not a file of the original"):
        fetch.fetched(arith)


def test_the_whole_diskette_is_the_road_left_when_neither_source_opens_one(
    monkeypatch, arith
):
    """Both sources rely on the same extractor, so they tend to fail together."""
    image = diskette({demo.file: SCRIPT for demo in catalogue.DEMOS})
    monkeypatch.setattr(fetch, "urlopen", serving({catalogue.IMAGE: image}))
    assert fetch.fetched(arith) == SCRIPT


def test_the_diskette_hands_over_the_eight_nobody_asked_for(monkeypatch, arith):
    """The other files in the image are kept, so they never need downloading again."""
    image = diskette({demo.file: SCRIPT for demo in catalogue.DEMOS})
    monkeypatch.setattr(fetch, "urlopen", serving({catalogue.IMAGE: image}))
    kept = {}
    fetch.fetched(arith, lambda file, data: kept.update({file: data}))
    assert sorted(kept) == sorted(
        demo.file for demo in catalogue.DEMOS if demo.file != arith.file
    )


def test_the_archives_refusal_is_the_one_reported_when_no_road_is_open(
    monkeypatch, arith
):
    """The error from the first route, the one expected to work, is the one reported."""
    monkeypatch.setattr(fetch, "urlopen", serving({}))
    with pytest.raises(OSError, match="the server said 503"):
        fetch.fetched(arith)


def test_a_diskette_that_will_not_open_leaves_the_archives_refusal_standing(
    monkeypatch, arith
):
    monkeypatch.setattr(fetch, "urlopen", serving({catalogue.IMAGE: b"not a zip"}))
    with pytest.raises(OSError, match="the server said 503"):
        fetch.fetched(arith)
