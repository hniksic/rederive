"""What the environment provides, and the one place that knows which one it is.

Four things the program needs and cannot work out for itself: a clipboard, a
filesystem, a memory reading, and the truth about what it is running on. Every
one of them has an answer on a desktop that it does not have somewhere else -
a browser has no resident set to read and no Qt to report - so they are
gathered behind one protocol rather than reached for where they are used. A
port then supplies another implementation instead of teaching every caller
what it is running on.

`desktop.py` is the only implementation there is. Nothing here picks by
looking at `sys.platform`: the ordinary environment is the default, and an
environment that is not the ordinary one is the one that knows it, so it calls
`use` before anything asks.

The desktop implementation is written for an install that may not have
everything. psutil and pyperclip are the `desktop` extra and the toolkit is
the `plot` one, all three opt-in, and an install that took none of them is a
supported install: it authors, computes and saves. So a missing wheel is an
answer here rather than an error - no clipboard is a copy that took the
terminal's road alone, and no reading is an empty field on the status line.

Storage is defined and not yet used. Every file the program reads or writes
still goes through `pathlib` where the command is, which is right for as long
as there is one filesystem; the protocol is written down now so that the side
that has no filesystem has something to satisfy rather than something to
invent.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

__all__ = ["Platform", "Storage", "current", "use"]


class Storage(Protocol):
    """Files, as the worksheet commands ask for them.

    Five calls, and each one is a place the program already touches a path:
    `worksheet.text_of` reads bytes rather than text, because a file the
    original wrote is code page 437 and only the caller knows that;
    `Session.save`, `save_source` and `save_state` write text; and the file
    prompt browses and completes from a directory listing.
    """

    def read(self, path: Path) -> bytes: ...

    def write(self, path: Path, text: str) -> None: ...

    def names(self, directory: Path) -> Sequence[str]: ...

    def exists(self, path: Path) -> bool: ...

    def is_directory(self, path: Path) -> bool: ...


class Platform(Protocol):
    """The environment, as the program asks it for things."""

    def copy(self, text: str) -> None:
        """Hand text to whatever clipboard there is, quiet where there is none."""

    def storage(self) -> Storage:
        """The files this environment keeps worksheets in."""

    def process_bytes(self, pid: int | None = None) -> int | None:
        """One process's resident set, or None where it will not be read."""

    def total_bytes(self) -> int | None:
        """How much memory the machine has, or None where that is unknown."""

    def provenance(self) -> str:
        """What this environment carries, one `name version` per line.

        The lines `--version` adds to the ones that are true wherever the
        program runs: the toolkit a plot would be drawn with, and the machine.
        """


#: The environment in force, once anything has asked or anything has said.
_platform: Platform | None = None


def current() -> Platform:
    """The environment this program is running in.

    The desktop unless something has said otherwise, and the import is here
    rather than at the top of the module because it is the import that costs -
    it is what reaches for psutil and pyperclip, and a process that never asks
    for a clipboard or a memory reading never loads either.
    """
    global _platform
    if _platform is None:
        from rederive.platform.desktop import Desktop

        _platform = Desktop()
    return _platform


def use(platform: Platform | None) -> None:
    """Say what environment this is, before anything asks. None means the default."""
    global _platform
    _platform = platform
