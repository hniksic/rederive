"""How much memory the program is holding, and how the figure is written.

The status line shows the figure where the original showed its muLISP heap
gauge. The reading itself belongs to the environment and comes from
`rederive.platform`, which asks psutil on a desktop and reads the size of its
own WebAssembly heap in a browser; what is here is what the gauge makes of it.
Where the platform will not answer the answer is None and the field stays empty
rather than showing a figure that might be wrong.

The program is two interpreters once the engine worker is up, and the gauge is
about the program, so the reading is the two added together. How the second one
is read is the difference between the environments, and there is a call for
each:

* `register_worker` says which *process* it is, for an environment that can
  measure a program it did not start from outside. That is the desktop, and the
  figure is as fresh as the moment it is asked for.
* `worker_holds` says what it *reported*, for an environment that cannot. That
  is the browser, where nothing outside a Web Worker can read its heap, so the
  worker says what it holds on its way past. Such a figure is as old as the
  worker's last answer - a worker in the middle of a computation answers
  nothing, its own timers included - which is as fresh as a reading there can
  be.

A worker that has died, has not been spawned yet or has said nothing yet
contributes nothing, silently - a gauge is not worth an error.
"""

from __future__ import annotations

from rederive import platform

#: Bytes in the units a size is written in, largest first. Resident sets are
#: measured in mebibytes on every machine this runs on, but a small one reads
#: better in kibibytes and a large one in gibibytes. The units are the binary
#: ones and are named as such, since that is what the figure counts.
_UNITS = (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10))

#: The engine worker, once one has been spawned. Module state because the
#: status line reads the gauge and the engine proxy owns the worker, and
#: neither has any business knowing about the other.
_worker: int | None = None

#: And what a worker that can only be read from the inside last said it holds.
#: The other half of the same slot, kept apart from it because the two are read
#: differently: this figure is already a size, where `_worker` is something to
#: go and measure.
_reported: int | None = None


def register_worker(pid: int | None) -> None:
    """Say which process the engine worker is, or None once it is gone."""
    global _worker
    _worker = pid


def worker_holds(size: int | None) -> None:
    """Say what the engine worker has just reported holding, or None once it is gone.

    For a worker nobody outside it can measure. The figure replaces whatever
    the last one said rather than adding to it: it is a reading of the same
    heap and not a second heap.
    """
    global _reported
    _reported = size


def process_bytes(pid: int | None = None) -> int | None:
    """One process's resident set, or None where it will not be read.

    None covers a platform that does not report a resident set and a process
    that is no longer there, which are the same thing to a caller: no figure to
    show.
    """
    return platform.current().process_bytes(pid)


def measures_processes() -> bool:
    """Whether a resident set can be read here at all.

    What the engine's memory watchdog turns on: an environment that measures no
    process runs without one. Asked of the platform rather than guessed from a
    None reading, since a None is also what a process that has died gives.
    """
    return platform.current().measures_processes()


def resident_bytes() -> int | None:
    """What the program holds: this interpreter, plus the worker where there is one.

    None when this one will not be read at all. A worker that will not be read
    is simply left out, since the figure without it is still true about the
    program and better than no figure.
    """
    own = process_bytes()
    if own is None:
        return None
    worker = _reported if _worker is None else process_bytes(_worker)
    return own if worker is None else own + worker


def total_bytes() -> int | None:
    """How much physical memory the machine has, or None where that is unknown."""
    return platform.current().total_bytes()


def written(size: int) -> str:
    """A byte count in the largest unit that leaves at least one whole digit.

    A figure in gibibytes carries a decimal, since whole gibibytes are a coarse
    enough step that the gauge would sit still while the program grew. Past ten
    gibibytes the decimal is dropped again: that much is a figure to read at a
    glance, not to watch.
    """
    for unit, scale in _UNITS:
        if size >= scale:
            value = size / scale
            if unit == "GiB" and round(value, 1) < 10:
                return f"{value:.1f} {unit}"
            return f"{value:.0f} {unit}"
    return f"{size} B"


def label() -> str:
    """The status line's memory field, empty where there is no figure to show."""
    size = resident_bytes()
    return "" if size is None else f"Memory: {written(size)}"
