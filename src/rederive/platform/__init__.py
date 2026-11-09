"""Services that depend on the environment, behind one protocol.

The program needs a clipboard, file storage, memory readings, and version information,
and each works differently on a desktop than in a browser (which has no resident set
and no Qt). Gathering them here lets each environment supply its own implementation
instead of every caller checking where it runs.

The desktop implementation is the default. Nothing checks `sys.platform`; another
environment (the browser) calls `use` before anything else asks.

The desktop implementation tolerates a partial install. psutil and pyperclip (the
`desktop` extra) and the toolkit (the `plot` extra) are all optional, and an install
without them can still author, compute, and save. A missing package therefore yields an
empty answer rather than an error: copying falls back to the terminal alone, and the
memory field stays empty.

All file I/O goes through `Storage`: `pathlib` on the desktop, a page-managed store,
downloads and uploads in the browser.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

__all__ = ["Platform", "Storage", "current", "use"]


class Storage(Protocol):
    """File access for the worksheet commands.

    `worksheet.text_of` reads bytes rather than text, because files written by the
    original are in code page 437 and only the caller knows that. `Session.save`,
    `save_source` and `save_state` write text. The file prompt lists directories for
    browsing and completion.

    Calls are keyed by path, since that is what the user typed and the status line
    shows. How an environment without a filesystem interprets a path is up to it;
    callers must not depend on it.
    """

    def read(self, path: Path) -> bytes: ...

    def write(self, path: Path, text: str) -> None: ...

    def names(self, directory: Path) -> Sequence[str]: ...

    def exists(self, path: Path) -> bool: ...

    def is_directory(self, path: Path) -> bool: ...


class Platform(Protocol):
    """The services an environment provides."""

    def copy(self, text: str, refused: Callable[[str], None] | None = None) -> None:
        """Copy text to the clipboard, if there is one.

        The environment may call `refused` with a message if the copy failed and that
        is worth reporting: a desktop without a clipboard tool has still copied
        through the terminal, but a browser page has not. It is a callback because it
        may be called after this returns, as a browser's clipboard is asynchronous.
        """

    def paste(self) -> str | None:
        """Return the clipboard's text, or None if it can't be read.

        None makes Ctrl-V fall back to the program's internal clipboard. That covers a
        machine with no clipboard tool, and the browser, where the page lets the
        browser handle pasting itself.
        """

    def storage(self) -> Storage:
        """Return the file storage for this environment."""

    def process_bytes(self, pid: int | None = None) -> int | None:
        """Return a process's resident set size, or None if it can't be read."""

    def measures_processes(self) -> bool:
        """Whether this environment can measure process memory at all.

        This differs from `process_bytes` returning None, which also happens for a
        process that just died. The engine's memory watchdog runs only if this is true.
        """

    def total_bytes(self) -> int | None:
        """Return the machine's physical memory, or None if unknown."""

    def provenance(self) -> str:
        """Return environment-specific `--version` lines, one `name version` per line.

        These are the plotting toolkit and the machine.
        """


#: The current environment, set on first use or by `use`.
_platform: Platform | None = None


def current() -> Platform:
    """Return the current environment, the desktop unless `use` said otherwise.

    The desktop module is imported lazily because it loads psutil and pyperclip, which
    a process that never needs them shouldn't pay for.
    """
    global _platform
    if _platform is None:
        from rederive.platform.desktop import Desktop

        _platform = Desktop()
    return _platform


def use(platform: Platform | None) -> None:
    """Set the environment before anything asks for it. None means the default."""
    global _platform
    _platform = platform
