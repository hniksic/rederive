"""The environment a browser tab provides: a page's clipboard, a store, a heap.

Three of the four answers here are shorter than the desktop's because a tab is
a smaller machine. The memory gauge reads the size of this instance's own
WebAssembly heap, there being no resident set and nothing outside an instance
that can measure it; there is no Qt, so what a plot would be drawn with is
nothing; and the clipboard is the page's, asked for through
`navigator.clipboard` and refused as often as it is granted - which is why it
is the one answer here that reports a refusal in words.

Storage is the long one, because a tab has no filesystem and has three things
instead, and a file goes to whichever of them fits what it is for:

* **the store**, a dictionary this class holds and localStorage keeps, keyed by
  the path the user typed. It is what makes a name mean the same file twice,
  and it is what survives a reload - MEMFS does not, and a worksheet written
  there would be gone with the tab.
* **a download**, which is where a save actually goes. It is the only way out
  of a tab and the only copy the user can keep, open in another program or
  send to anyone.
* **MEMFS**, read-only as far as this is concerned, which is where the wheel is
  unpacked. A path that names something on it is read by `pathlib` exactly as
  it is on a desktop, which is why the store is asked first and the filesystem
  behind it.

A file the user hands over - picked or dropped on the page - arrives through
`keep`, which is the store without the download: it came from a disk and
sending it back there is not what was asked for. The settings arrive the same
way, every time the tab is hidden.

Nothing selects this: `rederive.web.boot` calls `platform.use` before it builds
anything, which is the whole of how the browser says what it is.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

__all__ = ["Web", "WebStorage", "heap"]

#: What localStorage keeps the store under. One key holding the whole of it, as
#: JSON: worksheets are a few kilobytes each, and a store written in one piece
#: cannot half-survive a write that was refused part way through.
KEPT = "rederive.files"

#: What the page is told when the clipboard would not take the text. The first
#: half is what happened and the second is what to do instead, since the
#: terminal's own copy still went out and a browser's own copy key still works.
CLIPBOARD_REFUSED = "The browser did not allow the clipboard; select and copy instead"

#: What the page is told when the store will not hold any more. The save itself
#: went out as a download before this, so what is lost is the tab's memory of
#: the name and not the file.
NO_ROOM = "{name} was downloaded, but this page has no room left to remember it by name"


class WebStorage:
    """Files, as a tab has them: a store, a download, and the wheel's own MEMFS.

    The store is read once, when the page is built, and written whole on every
    change. Both are cheap - it is a few kilobytes of text - and neither has a
    failure worth a retry: a browser that will not keep it says so at once.
    """

    def __init__(self, page: Any = None) -> None:
        #: The page's file module, which owns everything with a DOM in it: the
        #: download, the picker, the drop target and the notice. None where
        #: there is no page, which is every reading of this module outside a
        #: browser.
        self._page = page
        self._files: dict[str, str] = {}
        #: Whether localStorage answered at all. A tab in private mode, or one
        #: whose user has turned site data off, keeps nothing across a reload,
        #: and the difference is worth a sentence at the top of the screen.
        self.remembers = self._read_store()

    # -- what the program asks for -----------------------------------------

    def read(self, path: Path) -> bytes:
        """A file's bytes: the store's copy, or the one the wheel carries.

        The store first, so that a worksheet of the user's own wins over one of
        the program's under the same name - which is the rule the desktop gets
        for nothing by looking in the working directory before the library.
        """
        kept = self._files.get(_key(path))
        if kept is not None:
            return kept.encode("utf-8")
        return path.read_bytes()

    def write(self, path: Path, text: str) -> None:
        """Send a file out of the tab, and remember it by name.

        The download goes first and the store second, because the download is
        the save: it is the copy the user keeps, and a store that has run out of
        room must not stop a worksheet from being written. A store that refuses
        therefore costs the name and not the file, and says so.

        A download the browser refuses outright is the other way round: nothing
        left the tab, so it is an OSError like any other unwritable file and the
        command reports it where a command reports such things.
        """
        if self._page is not None:
            try:
                self._page.download(path.name, text)
            except Exception as refused:
                raise OSError(f"the browser refused the download: {refused}") from None
        self._files[_key(path)] = text
        if not self._write_store() and self._page is not None:
            self._page.said(NO_ROOM.format(name=path.name))

    def names(self, directory: Path) -> Sequence[str]:
        """What is in a directory: the store's own names and MEMFS's behind them.

        A prompt completes from this, so what it offers is what a name would
        find. A tab's working directory holds nothing the user did not put
        there, so the store is nearly always the whole of the answer.
        """
        found = {Path(name).name for name in self._files if _in(name, directory)}
        try:
            found.update(entry.name for entry in directory.iterdir())
        except OSError:
            pass
        return sorted(found)

    def exists(self, path: Path) -> bool:
        return _key(path) in self._files or path.exists()

    def is_directory(self, path: Path) -> bool:
        """The store holds no directories, so this is MEMFS's answer alone."""
        return path.is_dir()

    # -- what the page asks for --------------------------------------------

    def keep(self, path: Path, text: str) -> None:
        """Remember a file by name without sending it anywhere.

        For the files that came from somewhere else: one the user picked or
        dropped on the page, which is already on their disk; the worksheet a
        link arrived carrying; and the settings, which the page writes every
        time it is hidden and which nobody asked to have a copy of.

        A store that will not take it is not reported, unlike a save's. Nothing
        was asked for here, the file is in this page's memory either way, and
        what is lost is only the next visit's memory of it.
        """
        self._files[_key(path)] = text
        self._write_store()

    def kept(self) -> list[str]:
        """Every name the store holds, which is what the page shows of itself."""
        return sorted(self._files)

    # -- the store ---------------------------------------------------------

    def _read_store(self) -> bool:
        """Take the store out of localStorage, and say whether there is one.

        Anything unreadable there is treated as nothing rather than as an
        error: a store written by some older shape of this is not worth
        refusing to start over.
        """
        storage = _local()
        if storage is None:
            return False
        try:
            held = storage.getItem(KEPT)
            if held:
                self._files = {
                    name: text
                    for name, text in json.loads(held).items()
                    if isinstance(text, str)
                }
        except Exception:
            return True
        return True

    def _write_store(self) -> bool:
        """Put the store back, and say whether it went.

        False is a browser that would not take it, which is nearly always the
        quota: five megabytes is the usual ceiling and a worksheet is small, so
        a tab reaches it only by saving a great many of them.
        """
        storage = _local()
        if storage is None:
            return False
        try:
            storage.setItem(KEPT, json.dumps(self._files))
        except Exception:
            return False
        return True


def _key(path: Path) -> str:
    """What a file is filed under: the path, as the user typed it.

    `worksheet.path_of` has already supplied the extension and expanded a
    leading `~`, so two spellings of one name arrive here as one string.
    """
    return str(path)


def _in(name: str, directory: Path) -> bool:
    """Whether a stored name lies directly in `directory`, as a listing asks."""
    return Path(name).parent == directory


def _local() -> Any:
    """The tab's localStorage, or nothing where there is none to be had.

    Reading it can raise as well as be missing: a browser told to keep no site
    data refuses the property itself rather than answering an empty store.
    """
    try:
        import js
    except ImportError:  # read outside a browser, which the tests do
        return None
    try:
        return js.localStorage
    except Exception:
        return None


class Web:
    """The four answers, as a browser tab gives them."""

    def __init__(self, page: Any = None) -> None:
        self._storage = WebStorage(page)

    # -- the clipboard -----------------------------------------------------

    def copy(self, text: str, refused: Callable[[str], None] | None = None) -> None:
        """Hand text to the page's clipboard, and say so where it will not take it.

        `navigator.clipboard` is absent outside a secure context and its write
        is a promise that may be refused - a tab that has not been clicked in
        is not allowed to write a clipboard, and Firefox asks the user about a
        page it has not seen do this before. A desktop can be quiet about all
        of that because the copy took the terminal's road as well; here there
        is no such road, xterm.js answering the escape sequence with nothing,
        so a refusal means the copy did not happen and the key did nothing.
        """
        import js

        clipboard = getattr(js.navigator, "clipboard", None)
        if clipboard is None:
            _say(refused, CLIPBOARD_REFUSED)
            return
        try:
            promise = clipboard.writeText(text)
        except Exception:
            _say(refused, CLIPBOARD_REFUSED)
            return
        # Nothing waits on a copy, so the promise is not awaited; what it is
        # given is somewhere to put a rejection, which is both what keeps the
        # user informed and what keeps an uncaught rejection out of the
        # console.
        catch = getattr(promise, "catch", None)
        if catch is not None:
            catch(lambda _error: _say(refused, CLIPBOARD_REFUSED))

    # -- files -------------------------------------------------------------

    def storage(self) -> WebStorage:
        return self._storage

    # -- memory ------------------------------------------------------------

    def process_bytes(self, pid: int | None = None) -> int | None:
        """This interpreter's heap: the WebAssembly memory it has taken.

        Not a resident set, a tab having none, but the nearest reading there
        is - and nearer in spirit to what this gauge replaced than a resident
        set ever was, the original having shown how much of its own workspace
        muLISP was holding.

        `pid` is ignored. There are no processes here to tell apart: an
        instance can read its own heap and nobody else's, so the question this
        answers is always about the interpreter asking it.
        """
        return heap()

    def measures_processes(self) -> bool:
        """No: a heap is a gauge and the watchdog wants something else.

        What this turns off is the desktop engine's cap, which reads a worker's
        resident set from outside and kills it at a ceiling. Neither half of
        that is here. Nothing outside a Web Worker can read its memory - the
        figure the gauge shows is one the worker sends - and the figure itself
        is a high-water mark that never comes down, so a cap read off it would
        go on holding a program to a size it let go of long ago.

        Nor is there a better reading to be had. `performance.memory` is
        Chromium's alone, is quantised to the point of meaninglessness and does
        not count WebAssembly memory; `measureUserAgentSpecificMemory` is
        Chromium's too, wants the page cross-origin isolated, and resolves at
        the next collection rather than when it is asked. So the engine runs
        uncapped here and an allocation that goes too far is the browser
        killing the worker, which arrives as a death like any other.
        """
        return False

    def total_bytes(self) -> int | None:
        """Nothing: how much memory the machine has is not a tab's to know."""
        return None

    # -- what this is ------------------------------------------------------

    def provenance(self) -> str:
        """What the page carries: the runtime under it and no toolkit at all.

        The first line is the shape the desktop's takes - what a plot would be
        drawn with, which here is nothing - so that `--version` reads the same
        way wherever it is asked.
        """
        toolkit = "Qt unusable (no toolkit in the browser)"
        return f"{toolkit}\n{_runtime()}\nplatform {_agent()}"


def heap() -> int | None:
    """How many bytes of WebAssembly memory this instance has, or None outside one.

    `_module` is Pyodide's own handle on the Emscripten instance and its heap
    views are the linear memory itself, so the length of one is the size of the
    heap. Two things to know about the figure. It is what the runtime has taken
    rather than what Python is using, since the allocator hands out of it and
    gives back into it; and WebAssembly memory grows and never shrinks, so it
    is a high-water mark - a program that has let go of a large expression goes
    on reporting what holding it cost.

    Both are why this is the gauge and not the watchdog: it says what the tab
    is carrying, which is what somebody watching the status line wants to know,
    and it cannot say that a computation has just released anything.

    The handle is private to Pyodide, hence the guard around every step: a
    release that moves it costs the field its figure and nothing else.

    Called by the engine worker as well as by the page, each about itself,
    which is why it is a function here rather than a method on `Web` - the
    worker sets no platform up, having nothing else to ask one for.
    """
    try:
        import pyodide_js
    except ImportError:  # read outside a browser, which the tests do
        return None
    try:
        return int(pyodide_js._module.HEAPU8.length)
    except Exception:
        return None


def _say(refused: Callable[[str], None] | None, sentence: str) -> None:
    """Report a refusal wherever the caller left somewhere to report it."""
    if refused is not None:
        refused(sentence)


def _runtime() -> str:
    """Which Pyodide is running this, or that the question has no answer here."""
    try:
        import pyodide
    except ImportError:  # read outside a browser, which the tests do
        return "Pyodide absent"
    return f"Pyodide {pyodide.__version__}"


def _agent() -> str:
    """The browser, as it names itself. The nearest thing a tab has to a machine."""
    try:
        import js
    except ImportError:
        return "emscripten"
    return str(js.navigator.userAgent)
