"""The browser environment: the page's clipboard, a persistent store, the Wasm heap.

Memory is the size of this instance's WebAssembly heap, since a tab has no resident set.
There is no Qt. The clipboard is `navigator.clipboard`, which is often refused, so
this is the only implementation that reports a refused copy.

A tab has no real filesystem, so storage uses three things:

* **The store**, a dict kept in localStorage and keyed by the path the user typed. It
  makes a name refer to the same file again, and unlike MEMFS it survives a reload.
* **A download**, where a save actually goes. It is the only way to get a file out of
  the tab.
* **MEMFS**, where the wheel is unpacked, read-only for our purposes. Reads try the
  store first and then MEMFS through `pathlib`.

Files the user picks or drops on the page arrive through `keep`, which stores them
without a download, since they already came from disk. The settings are stored the same
way whenever the tab is hidden.

`rederive.web.boot` installs this with `platform.use` before building anything.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

__all__ = ["Web", "WebStorage", "heap"]

#: localStorage key holding the whole store as JSON. Worksheets are small, and writing
#: the store in one piece means a failed write can't leave it half-updated.
KEPT = "rederive.files"

#: Message shown when the clipboard refuses the text, suggesting the browser's own copy.
CLIPBOARD_REFUSED = "The browser did not allow the clipboard; select and copy instead"

#: Message shown when the store is full. The save already went out as a download, so
#: only the stored copy under that name is lost.
NO_ROOM = "{name} was downloaded, but this page has no room left to remember it by name"


class WebStorage:
    """File storage in a browser tab: the store, downloads, and MEMFS.

    The store is read once at startup and written whole on every change. Both are
    cheap, and failures aren't worth retrying.
    """

    def __init__(self, page: Any = None) -> None:
        #: The page's file module, which handles the DOM side: downloads, the file
        #: picker, the drop target and notices. None outside a browser.
        self._page = page
        self._files: dict[str, str] = {}
        #: Whether localStorage is available. In private mode or with site data
        #: disabled it isn't, and nothing survives a reload, which the screen mentions.
        self.remembers = self._read_store()

    # -- what the program asks for -----------------------------------------

    def read(self, path: Path) -> bytes:
        """Return a file's bytes from the store, or else from MEMFS.

        The store comes first so a user's file shadows a bundled one of the same name,
        as the working directory does on the desktop.
        """
        kept = self._files.get(_key(path))
        if kept is not None:
            return kept.encode("utf-8")
        return path.read_bytes()

    def write(self, path: Path, text: str) -> None:
        """Download the file and keep it in the store.

        The download is the real save, so it goes first, and a full store only
        produces a notice. A refused download raises OSError, like any unwritable
        file.
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
        """List a directory's entries from both the store and MEMFS.

        Used for prompt completion. The working directory normally only has what the
        user put there, so in practice this is mostly the store.
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
        """Check MEMFS only, since the store holds no directories."""
        return path.is_dir()

    # -- what the page asks for --------------------------------------------

    def keep(self, path: Path, text: str) -> None:
        """Store a file without downloading it.

        Used for files that came from elsewhere: picked or dropped by the user, carried
        by a link, or the settings, which the page writes whenever it is hidden.

        Unlike in `write`, a full store isn't reported: the file is still in memory, and
        only the next visit loses it.
        """
        self._files[_key(path)] = text
        self._write_store()

    def kept(self) -> list[str]:
        """Return every name in the store, for the page to display."""
        return sorted(self._files)

    # -- the store ---------------------------------------------------------

    def _read_store(self) -> bool:
        """Load the store from localStorage, returning whether localStorage exists.

        Unreadable contents (e.g. from an older format) are ignored rather than treated
        as an error.
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
        """Save the store to localStorage, returning whether it succeeded.

        Failure nearly always means the quota (usually 5 MB) is exceeded, which takes a
        great many worksheets.
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
    """Return the store key for `path`, which is the path as typed.

    `worksheet.path_of` has already added the extension and expanded `~`, so
    equivalent spellings arrive as the same string.
    """
    return str(path)


def _in(name: str, directory: Path) -> bool:
    """Whether a stored name lies directly in `directory`."""
    return Path(name).parent == directory


def _local() -> Any:
    """Return the tab's localStorage, or None if unavailable.

    Accessing it can raise, e.g. when the browser is set to keep no site data.
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
    """The `Platform` implementation for a browser tab."""

    def __init__(self, page: Any = None) -> None:
        self._storage = WebStorage(page)

    # -- the clipboard -----------------------------------------------------

    def copy(self, text: str, refused: Callable[[str], None] | None = None) -> None:
        """Copy text to the page's clipboard, reporting through `refused` if refused.

        `navigator.clipboard` is absent outside a secure context, and its write returns
        a promise that may be rejected: an unfocused tab may not write, and Firefox asks
        the user the first time. Unlike on the desktop, the terminal copy doesn't help
        here (xterm.js ignores the escape sequence), so a refusal means nothing was
        copied.
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
        # The promise isn't awaited, but a rejection handler informs the user and keeps
        # an uncaught rejection out of the console.
        catch = getattr(promise, "catch", None)
        if catch is not None:
            catch(lambda _error: _say(refused, CLIPBOARD_REFUSED))

    def paste(self) -> str | None:
        """Return None: the page lets the browser handle Ctrl-V itself.

        Reading `navigator.clipboard` would require a permission prompt, while the
        browser's own paste needs none.
        """
        return None

    # -- files -------------------------------------------------------------

    def storage(self) -> WebStorage:
        return self._storage

    # -- memory ------------------------------------------------------------

    def process_bytes(self, pid: int | None = None) -> int | None:
        """Return this interpreter's WebAssembly heap size.

        A tab has no resident set, but the heap size is the closest equivalent, and
        closer in spirit to the original's muLISP workspace gauge anyway.

        `pid` is ignored, since an instance can only read its own heap.
        """
        return heap()

    def measures_processes(self) -> bool:
        """Return False, which disables the engine's memory watchdog.

        The desktop watchdog reads the worker's resident set from outside and kills it
        at a limit. Neither works here: a Web Worker's memory can't be read from
        outside (the gauge shows what the worker reports), and the heap size is a
        high-water mark that never shrinks.

        There's no better alternative. `performance.memory` is Chromium-only, heavily
        quantized, and excludes WebAssembly memory; `measureUserAgentSpecificMemory` is
        Chromium-only, requires cross-origin isolation, and resolves only at the next
        GC. So the engine runs uncapped, and if it allocates too much the browser kills
        the worker, which is handled like any other worker death.
        """
        return False

    def total_bytes(self) -> int | None:
        """Return None, since a tab can't know the machine's memory."""
        return None

    # -- what this is ------------------------------------------------------

    def provenance(self) -> str:
        """Return the (absent) toolkit, the Pyodide runtime, and the browser.

        The first line mirrors the desktop's toolkit line, so `--version` output has
        the same shape everywhere.
        """
        toolkit = "Qt unusable (no toolkit in the browser)"
        return f"{toolkit}\n{_runtime()}\nplatform {_agent()}"


def heap() -> int | None:
    """Return the size of this instance's WebAssembly memory, or None outside one.

    `_module` is Pyodide's handle on the Emscripten instance, and its heap views span
    the linear memory, so a view's length is the heap size. This is memory the runtime
    has reserved, not what Python is using, and since WebAssembly memory never shrinks
    it is a high-water mark. That makes it fine for the gauge but useless for a
    watchdog.

    `_module` is private to Pyodide, so every step is guarded; if it changes, the
    memory field just goes blank.

    This is a function rather than a `Web` method because the engine worker calls it
    too, and the worker sets up no platform.
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
    """Call `refused` with `sentence`, if given."""
    if refused is not None:
        refused(sentence)


def _runtime() -> str:
    """Return the Pyodide version line, or "Pyodide absent"."""
    try:
        import pyodide
    except ImportError:  # read outside a browser, which the tests do
        return "Pyodide absent"
    return f"Pyodide {pyodide.__version__}"


def _agent() -> str:
    """Return the browser's user agent, the nearest thing a tab has to a machine."""
    try:
        import js
    except ImportError:
        return "emscripten"
    return str(js.navigator.userAgent)
