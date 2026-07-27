"""How much memory the program is holding, asked of the platform directly.

The status line shows the figure where the original showed its muLISP heap
gauge. psutil is what asks: it is already a dependency for the engine worker's
memory cap, and one library that answers on Linux, macOS and Windows is worth
more than three hand-written ways of asking. Where a platform will not answer,
or a reading fails, the answer is None and the field stays empty rather than
showing a figure that might be wrong.

The program is two processes once the engine worker is up, and the gauge is
about the program: `register_worker` says which process the second one is, and
the reading is the two added together. A worker that has died or has not been
spawned yet contributes nothing, silently - a gauge is not worth an error.
"""

from __future__ import annotations

import psutil

#: Bytes in the units a size is written in, largest first. Resident sets are
#: measured in megabytes on every machine this runs on, but a small one reads
#: better in kilobytes and a large one in gigabytes.
_UNITS = (("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10))

#: The engine worker, once one has been spawned. Module state because the
#: status line reads the gauge and the engine proxy owns the worker, and
#: neither has any business knowing about the other.
_worker: int | None = None


def register_worker(pid: int | None) -> None:
    """Say which process the engine worker is, or None once it is gone."""
    global _worker
    _worker = pid


def process_bytes(pid: int | None = None) -> int | None:
    """One process's resident set, or None where it will not be read.

    None covers both a platform that does not report a resident set and a
    process that is no longer there, which are the same thing to a caller: no
    figure to show.
    """
    try:
        return psutil.Process(pid).memory_info().rss
    except (psutil.Error, OSError, ValueError, AttributeError):
        return None


def resident_bytes() -> int | None:
    """What the program holds: this process, plus the worker where there is one.

    None when this process will not be read at all. A worker that will not be
    read is simply left out, since the figure without it is still true about
    the program and better than no figure.
    """
    own = process_bytes()
    if own is None:
        return None
    if _worker is None:
        return own
    worker = process_bytes(_worker)
    return own if worker is None else own + worker


def total_bytes() -> int | None:
    """How much physical memory the machine has, or None where that is unknown."""
    try:
        return psutil.virtual_memory().total
    except (psutil.Error, OSError, ValueError, AttributeError):
        return None


def written(size: int) -> str:
    """A byte count in the largest unit that leaves at least one whole digit."""
    for unit, scale in _UNITS:
        if size >= scale:
            return f"{size / scale:.0f}{unit}"
    return f"{size}B"


def label() -> str:
    """The status line's memory field, empty where there is no figure to show."""
    size = resident_bytes()
    return "" if size is None else f"Memory:{written(size)}"
