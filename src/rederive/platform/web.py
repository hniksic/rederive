"""The environment a browser tab provides: a page's clipboard, MEMFS, no gauge.

Three of the four answers here are shorter than the desktop's because a tab is
a smaller machine. There is no resident set to read, so the memory gauge is
empty and the engine's watchdog never runs; there is no Qt, so what a plot
would be drawn with is nothing; and the clipboard is the page's, asked for
through `navigator.clipboard` and refused as often as it is granted.

The one that is not smaller is storage. Pyodide's MEMFS is a real filesystem
with real paths on it - the package's own worksheets are unpacked into it with
the wheel, and `pathlib` reads them exactly as it does on a desktop - so files
work here without any of the work stage 7 owes: Save writing a download and
Load opening a file picker are what a tab has instead of a home directory, and
until they exist a file written here lives as long as the tab does.

Nothing selects this: `rederive.web.boot` calls `platform.use` before it builds
anything, which is the whole of how the browser says what it is.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

__all__ = ["Web", "WebStorage"]


class WebStorage:
    """Files through `pathlib`, MEMFS being a filesystem like any other.

    What it is not is a filesystem that outlives the tab. A worksheet saved
    here is saved into the page's own memory and is gone on the next visit,
    which is stage 7's to fix by making Save a download and Load a file the
    user hands over; the reading half already works, since the worksheets the
    package ships with travel in the wheel and are unpacked with it.
    """

    def read(self, path: Path) -> bytes:
        return path.read_bytes()

    def write(self, path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")

    def names(self, directory: Path) -> Sequence[str]:
        """What is in a directory, or nothing where it will not be read."""
        try:
            return sorted(entry.name for entry in directory.iterdir())
        except OSError:
            return []

    def exists(self, path: Path) -> bool:
        return path.exists()

    def is_directory(self, path: Path) -> bool:
        return path.is_dir()


class Web:
    """The four answers, as a browser tab gives them."""

    def __init__(self) -> None:
        self._storage = WebStorage()

    # -- the clipboard -----------------------------------------------------

    def copy(self, text: str) -> None:
        """Hand text to the page's clipboard, quiet where the page has none.

        `navigator.clipboard` is absent outside a secure context and its write
        is a promise that may be refused - a tab that has not been clicked in
        is not allowed to write a clipboard - so this is best effort in the
        same way the desktop's is: the copy took the terminal's road as well,
        and a refusal is not worth a message about a key that appeared to
        work.
        """
        import js

        clipboard = getattr(js.navigator, "clipboard", None)
        if clipboard is None:
            return
        try:
            promise = clipboard.writeText(text)
        except Exception:
            return
        # The promise is not awaited - nothing waits on a copy - but a
        # rejection nobody catches is an error in the page's console, which is
        # noise about something that was always allowed to fail.
        catch = getattr(promise, "catch", None)
        if catch is not None:
            catch(lambda _error: None)

    # -- files -------------------------------------------------------------

    def storage(self) -> WebStorage:
        return self._storage

    # -- memory ------------------------------------------------------------

    def process_bytes(self, pid: int | None = None) -> int | None:
        """Nothing: a tab cannot see a worker's heap, its own included."""
        return None

    def measures_processes(self) -> bool:
        """No, which is what turns the engine's memory watchdog off.

        There is no reading to be had. `performance.memory` is Chromium's
        alone, is the whole tab rather than the worker, and is quantised to
        the point of meaninglessness; a Web Worker's heap is not exposed at
        all. So the engine runs uncapped here and an allocation that goes too
        far is the browser killing the worker, which arrives as a death like
        any other.
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
