"""The environment an ordinary machine provides: a clipboard, files, psutil, Qt.

Three of the four answers here are given by a wheel that an install need not
have. psutil and pyperclip are the `desktop` extra, pyqtgraph and Qt the
`plot` one, and all of them are opt-in - so each is imported in a try, and an
install without one gets the answer that says so rather than an ImportError
from a module that has nothing to do with what was asked.

The failures a wheel that *is* installed can produce are caught here too, and
for the same reason: a clipboard tool that is not on the machine and a
resident set the kernel will not report are ordinary on a desktop, and neither
is worth an error to a status line or a copy.
"""

from __future__ import annotations

import platform
from collections.abc import Sequence
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
    """Files through `pathlib`, which is what a machine with a filesystem has."""

    def read(self, path: Path) -> bytes:
        return path.read_bytes()

    def write(self, path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")

    def names(self, directory: Path) -> Sequence[str]:
        """What is in a directory, or nothing where it will not be read.

        A prompt completes from this, and a directory that is not there or not
        readable is one that completes to nothing rather than one that refuses.
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
    """The four answers, as a machine with a desktop on it gives them."""

    def __init__(self) -> None:
        self._storage = DesktopStorage()

    # -- the clipboard -----------------------------------------------------

    def copy(self, text: str) -> None:
        """Hand text to the desktop's own clipboard, through whatever tool is there.

        pyperclip finds wl-copy, xclip or xsel and uses it, and is quiet where
        it finds none. An install without pyperclip is quiet in the same way:
        the copy still took the terminal's road, which is the only road there
        is across ssh anyway.
        """
        if pyperclip is None:
            return
        try:
            pyperclip.copy(text)
        except pyperclip.PyperclipException:
            pass

    # -- files -------------------------------------------------------------

    def storage(self) -> DesktopStorage:
        return self._storage

    # -- memory ------------------------------------------------------------

    def process_bytes(self, pid: int | None = None) -> int | None:
        """One process's resident set, or None where it will not be read.

        None covers a machine without psutil, a platform that does not report
        a resident set, and a process that is no longer there. They are the
        same thing to a caller: no figure to show.
        """
        if psutil is None:
            return None
        try:
            return psutil.Process(pid).memory_info().rss
        except (psutil.Error, OSError, ValueError, AttributeError):
            return None

    def total_bytes(self) -> int | None:
        """How much physical memory the machine has, or None where that is unknown."""
        if psutil is None:
            return None
        try:
            return psutil.virtual_memory().total
        except (psutil.Error, OSError, ValueError, AttributeError):
            return None

    # -- what this is ------------------------------------------------------

    def provenance(self) -> str:
        """The toolkit a plot would be drawn with, and the machine underneath.

        Which Qt a build carries is the first question a plot that misbehaves
        raises, and a bundle carries its own, so nothing outside it can answer.
        The import is the answer either way: a machine whose Qt will not load
        is one where the Plot command cannot work, and saying so here costs
        nothing, where refusing to print a version at all would cost a user the
        rest of the lines.
        """
        machine = f"{platform.system().lower()} {platform.machine()}"
        return f"{_toolkit()}\nplatform {machine}"


def _toolkit() -> str:
    """The Qt and pyqtgraph a plot is drawn with, or the reason there is none."""
    try:
        import pyqtgraph
        from pyqtgraph.Qt import QtCore
    except ImportError as missing:
        return f"Qt unusable ({missing})"

    return f"Qt {QtCore.qVersion()}\npyqtgraph {pyqtgraph.__version__}"
