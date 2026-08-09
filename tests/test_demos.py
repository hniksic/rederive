"""The original's demonstrations: the catalogue, and the page that fetches one.

The catalogue is nine names and where they can be had, and it is Python rather
than page furniture because a desktop wanting the same menu wants the same nine
files. What is asked of it here is what a reader of it is entitled to assume:
that a gallery is a gallery, and that every entry can be fetched from every
source.

The page seam is the other half. What the browser does - a menu, a download, a
relay when nothing answers - is `web/demos.js` and is not here. What is here is
what happens to a file once it has arrived: it becomes a file of the page's
store under the name the original gave it, and the app is told to run it as one
of the two kinds of demonstration.

The app is faked to the depth this code touches it, which is one question and
one call. What a demonstration then *does* is the app's own, and is tested with
a real app in `test_transfer` and `test_plot`.

The desktop's half of the same seam is the last section: the roads a machine
with sockets takes to the same nine files, over a `urlopen` that answers out of
a table. The menu those come in by is `test_transfer` too.
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
from rederive.web.demos import Demos

SCRIPT = b"; adds two numbers\r\n2 + 3\r\n\x1a"
GALLERY = b'"A spike:"\r\n\r\n2*SIN(LN(ABS(x)))\r\n\x1a'


# -- the catalogue ------------------------------------------------------------


def test_every_demonstration_is_named_by_the_file_it_is():
    """The name is the file's own, since it is downloaded and kept under it."""
    assert catalogue.named("ALGEBRA.DMO").title == "Algebra"
    assert catalogue.named("algebra.dmo") is None
    assert catalogue.named("NOTHING.DMO") is None


def test_the_galleries_are_the_files_whose_steps_are_drawn():
    drawn = [demo.file for demo in catalogue.DEMOS if demo.plotting]
    assert drawn == ["PLOT2D.MTH", "PLOTPARA.MTH", "PLOT3D.MTH"]
    # The extension says the same thing the flag does, and a step of a gallery
    # is read as a worksheet line rather than as a script's.
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
    """The fake page below answers to anything; the real one has to be checked.

    Python is handed the `demos.js` module object and calls into it by name, so
    a call this side makes and that side does not export is a hole no test with
    a fake page in it can see - and in a browser it is an exception with the
    demonstration half started.
    """
    source = (Path(__file__).resolve().parents[1] / "web" / "demos.js").read_text()
    pattern = r"^export (?:function\s+(\w+)|\{([^}]*)\})"
    exported = set()
    for name, group in re.findall(pattern, source, re.M):
        exported.update([name] if name else (word.strip() for word in group.split(",")))
    # `wire` is the page's own; the other two are what this module reaches for.
    assert {"wire", "attend", "said"} <= exported


# -- the page's menu ----------------------------------------------------------


@pytest.fixture
def page():
    return FakeTab()


class FakeApp:
    """The program as the menu speaks to it: is it up, and here is a file.

    `call_next` is what the app is entered by, since a click arrives on a DOM
    callback and the screen belongs to the loop. It runs what it is handed at
    once, which is what the loop does a moment later and what leaves every test
    below reading in one line. `deferring` holds it instead, for the one test
    that is about the handing over itself.
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
        """Run what the loop would have run."""
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
    """The menu's own road in: what JS calls, by the name JS calls it."""
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
    # Every entry carries its own addresses: which host has the file is not the
    # page's decision, and the page is the only side that can ask.
    assert offered[0]["urls"] == list(catalogue.DEMOS[0].urls)
    # And the headings a menu groups them under, in the order they are offered.
    assert [entry["group"] for entry in offered[:1] + offered[-1:]] == [
        catalogue.SCRIPTS,
        catalogue.GALLERIES,
    ]


def test_the_app_is_entered_through_its_loop_and_not_from_the_click(page):
    """A click lands on a DOM callback, and the screen belongs to the loop.

    Reached from the callback, a demonstration is set going and never drawn -
    which is what an event off a plot pane crosses to the loop to avoid.
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
    # A file like any other afterwards: it is in the store under its own name,
    # which is the name Transfer Demo would be given to run it again.
    assert worksheet.demonstration(Path("ALGEBRA.DMO")) == (
        ("adds two numbers", "2 + 3"),
    )


def test_a_gallery_runs_as_the_kind_that_is_drawn(demos, page, app):
    """The kind is read off the catalogue, never believed from the page."""
    chose(page, "PLOT2D.MTH", GALLERY)
    assert app.demonstrated == [("PLOT2D.MTH", True)]
    assert worksheet.demonstration(Path("PLOT2D.MTH")) == (
        ("A spike:", "2*SIN(LN(ABS(x)))"),
    )


def test_the_page_is_told_where_the_whole_diskette_is(demos, page):
    """The road that does not go through the archive's extractor."""
    assert page.handlers["image"] == catalogue.IMAGE


def test_a_demonstration_can_be_kept_without_being_run(demos, page, app):
    """The eight the page did not ask for, out of a diskette it paid for once."""
    keep(page, "PLOT2D.MTH", GALLERY)
    assert app.demonstrated == []
    assert kept(page, "PLOT2D.MTH")
    # And it is a file like any other: the next choice of it runs off the store.
    chose(page, "PLOT2D.MTH")
    assert app.demonstrated == [("PLOT2D.MTH", True)]


def test_nothing_is_kept_under_a_name_that_is_on_no_list(demos, page):
    keep(page, "MALICE.DMO", SCRIPT)
    assert not platform.current().storage().exists(Path("MALICE.DMO"))


def test_a_demonstration_the_page_has_is_run_without_a_download(demos, page, app):
    """The second viewing of one: the store outlives the tab, so nothing waits.

    Nothing is brought along, and the page says so the way a page says it: with
    its own `null`, which is falsy and is not `None`. Asked for by identity it
    reads as bytes, and the bytes it has none of are then read out of it.
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
    # Demo reads a script and Load reads a worksheet, and a gallery is a
    # worksheet by extension: naming the wrong one would send nobody anywhere.
    assert page.notices == [
        "ALGEBRA.DMO is in this page now; Transfer Demo runs it again by that name",
        "PLOT2D.MTH is in this page now; Transfer Load Derive reads it by that name",
    ]


def test_a_file_the_original_wrote_is_read_in_its_own_encoding(demos, page):
    """Code page 437, which is what a Greek name in one of these files is."""
    chose(page, "ARITH.DMO", "; the angle α\r\nSIN(x)\r\n\x1a".encode("cp437"))
    assert worksheet.demonstration(Path("ARITH.DMO")) == (("the angle α", "SIN(x)"),)


def test_a_demonstration_asked_for_too_early_is_refused_in_words(page):
    """The button is there from the first frame and the program is not."""
    starting = FakeApp(running=False)
    Demos(page, starting, platform.current().storage()).attend()
    chose(page, "ALGEBRA.DMO", SCRIPT)
    assert starting.demonstrated == []
    assert page.notices == [
        "Rederive is still starting; try the demonstration again in a moment"
    ]
    # And nothing was kept: the file is downloaded again next time, which is
    # the next thing the user will do.
    assert not platform.current().storage().exists(Path("ALGEBRA.DMO"))


# -- the desktop's own download -----------------------------------------------
#
# The other half of the seam `web/demos.js` is: what a machine with sockets does
# when one of the nine is asked for and is not on the disk. Nothing here opens
# one - `urlopen` is answered by a table of what each address holds - so what is
# tested is the roads and the guards, which is all of this module that is not
# the standard library.


class Answer:
    """What `urlopen` hands back, as much of it as this reads."""

    def __init__(self, raw: bytes) -> None:
        self.raw = raw

    def read(self, size: int = -1) -> bytes:
        return self.raw if size < 0 else self.raw[:size]

    def __enter__(self) -> "Answer":
        return self

    def __exit__(self, *_: object) -> bool:
        return False


def serving(held, slow=()):
    """An answering `urlopen`: `held` by address, an OSError where there is none.

    An address in `slow` answers after a moment, which is how a race is put the
    way round a test wants it.
    """

    def opened(url, timeout=None):
        if url in slow:
            time.sleep(0.05)
        if url not in held:
            raise OSError(f"the server said 503: {url}")
        return Answer(held[url])

    return opened


def diskette(files):
    """A zip of the shape the archive holds, its names inside a directory."""
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
    """Both are asked at once, so the slow one costs nothing while the other is up."""
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
    """A relay answers its own error page with a 200, so the status says little."""
    held = dict.fromkeys(arith.urls, answered)
    monkeypatch.setattr(fetch, "urlopen", serving(held))
    with pytest.raises(OSError, match="not a file of the original"):
        fetch.fetched(arith)


def test_the_whole_diskette_is_the_road_left_when_neither_source_opens_one(
    monkeypatch, arith
):
    """One piece of machinery opens both images, so both shut at once."""
    image = diskette({demo.file: SCRIPT for demo in catalogue.DEMOS})
    monkeypatch.setattr(fetch, "urlopen", serving({catalogue.IMAGE: image}))
    assert fetch.fetched(arith) == SCRIPT


def test_the_diskette_hands_over_the_eight_nobody_asked_for(monkeypatch, arith):
    """They have been paid for, so none of them is ever downloaded again."""
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
    """The first road is the one that was meant to work, so it is the one read."""
    monkeypatch.setattr(fetch, "urlopen", serving({}))
    with pytest.raises(OSError, match="the server said 503"):
        fetch.fetched(arith)


def test_a_diskette_that_will_not_open_leaves_the_archives_refusal_standing(
    monkeypatch, arith
):
    monkeypatch.setattr(fetch, "urlopen", serving({catalogue.IMAGE: b"not a zip"}))
    with pytest.raises(OSError, match="the server said 503"):
        fetch.fetched(arith)
