"""The desktop environment: system clipboard, `pathlib` files, psutil, and Qt.

psutil and pyperclip (the `desktop` extra) and pyqtgraph and Qt (the `plot` extra) are
optional, so each is imported in a try, and a missing one yields an empty answer rather
than an ImportError.

Runtime failures of installed packages, such as a missing clipboard tool or an
unreadable resident set, are also caught, since they are routine on a desktop and not
worth an error.
"""

from __future__ import annotations

import platform
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # the `desktop` extra was not installed
    psutil: Any = None

try:
    import pyperclip
except ImportError:  # likewise
    pyperclip: Any = None

__all__ = ["Desktop", "DesktopStorage"]


class DesktopStorage:
    """File storage through `pathlib`."""

    def read(self, path: Path) -> bytes:
        return path.read_bytes()

    def write(self, path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")

    def names(self, directory: Path) -> Sequence[str]:
        """List a directory's entries, or nothing if it can't be read.

        Used for prompt completion, where a missing or unreadable directory simply
        completes to nothing.
        """
        try:
            return sorted(entry.name for entry in directory.iterdir())
        except OSError:
            return []

    def exists(self, path: Path) -> bool:
        return path.exists()

    def is_directory(self, path: Path) -> bool:
        return path.is_dir()


class Desktop:
    """The `Platform` implementation for a desktop machine."""

    def __init__(self) -> None:
        self._storage = DesktopStorage()

    # -- the clipboard -----------------------------------------------------

    def copy(self, text: str, refused: Callable[[str], None] | None = None) -> None:
        """Copy text to the system clipboard through whatever tool is available.

        pyperclip uses wl-copy, xclip or xsel, and silently does nothing if none is
        found or pyperclip itself is missing. `refused` is never called, because the
        copy has also gone out through the terminal (the only way across ssh anyway),
        so the key still worked.
        """
        if pyperclip is None:
            return
        try:
            pyperclip.copy(text)
        except pyperclip.PyperclipException:
            pass

    def paste(self) -> str | None:
        """Return the system clipboard's text, using the same tool as `copy`.

        None if there is no tool, which is usual across ssh, where the terminal's own
        paste key is the way to paste.
        """
        if pyperclip is None:
            return None
        try:
            return pyperclip.paste()
        except pyperclip.PyperclipException:
            return None

    # -- files -------------------------------------------------------------

    def storage(self) -> DesktopStorage:
        return self._storage

    # -- memory ------------------------------------------------------------

    def process_bytes(self, pid: int | None = None) -> int | None:
        """Return a process's resident set size, or None if it can't be read.

        None covers a missing psutil, a platform that doesn't report it, and a process
        that is gone.
        """
        if psutil is None:
            return None
        try:
            return psutil.Process(pid).memory_info().rss
        except (psutil.Error, OSError, ValueError, AttributeError):
            return None

    def measures_processes(self) -> bool:
        """Whether psutil is available.

        Even with psutil, a particular process may be unreadable (gone, or owned by
        another user); that is `process_bytes` returning None, not this returning False.
        """
        return psutil is not None

    def total_bytes(self) -> int | None:
        """Return the machine's physical memory, or None if unknown."""
        if psutil is None:
            return None
        try:
            return psutil.virtual_memory().total
        except (psutil.Error, OSError, ValueError, AttributeError):
            return None

    # -- what this is ------------------------------------------------------

    def provenance(self) -> str:
        """Return the plotting toolkit and the machine.

        The bundled Qt version is the first thing to check when a plot misbehaves. The
        Qt backend reports it.
        """
        from rederive.plot.qt import toolkit

        machine = f"{platform.system().lower()} {platform.machine()}"
        return f"{toolkit()}\nplatform {machine}"
