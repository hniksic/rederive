"""Memory usage of the program, and how it is formatted.

The status line shows this where the original showed its muLISP heap gauge. The
measurement comes from `rederive.platform`: psutil on the desktop, the WebAssembly heap
size in the browser. Where the platform can't measure, the reading is None and the
field stays empty.

Once the engine worker is running, the figure is the sum of both interpreters. The
worker is measured in one of two ways:

* `register_worker` records its process ID, for the desktop, which can measure another
  process directly. The figure is current whenever it is read.
* `worker_holds` records the size the worker reported, for the browser, where a Web
  Worker's heap can't be read from outside. The worker reports after each answer and
  from its garbage collector while computing, so the gauge moves during long
  computations.

A worker that has died, hasn't started, or hasn't reported yet is silently left out.
"""

from __future__ import annotations

from rederive import platform

#: Binary units for displaying sizes, largest first.
_UNITS = (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10))

#: PID of the engine worker, once spawned. Kept as module state so that the status line
#: (which reads the gauge) and the engine proxy (which owns the worker) stay decoupled.
_worker: int | None = None

#: Last size reported by a worker that can't be measured from outside. Unlike `_worker`,
#: this is already a size rather than something to measure.
_reported: int | None = None


def register_worker(pid: int | None) -> None:
    """Record the engine worker's PID, or None once it is gone."""
    global _worker
    _worker = pid


def worker_holds(size: int | None) -> None:
    """Record the size the engine worker just reported, or None once it is gone.

    Used for a worker that can't be measured from outside. Each report replaces the
    previous one.
    """
    global _reported
    _reported = size


def process_bytes(pid: int | None = None) -> int | None:
    """Return a process's resident set size, or None if it can't be read.

    None means either that the platform doesn't report it or that the process is gone.
    """
    return platform.current().process_bytes(pid)


def measures_processes() -> bool:
    """Whether this platform can measure process memory at all.

    The engine's memory watchdog runs only if it can. This is asked of the platform
    rather than inferred from a None reading, which a dead process also gives.
    """
    return platform.current().measures_processes()


def resident_bytes() -> int | None:
    """Return memory held by this interpreter plus the worker, if any.

    None if this process can't be measured. A worker that can't be measured is left
    out, since a partial figure is better than none.
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
    """Format a byte count in the largest unit that keeps at least one whole digit.

    Gibibytes below 10 get one decimal, since whole GiB steps are too coarse to show
    growth.
    """
    for unit, scale in _UNITS:
        if size >= scale:
            value = size / scale
            if unit == "GiB" and round(value, 1) < 10:
                return f"{value:.1f} {unit}"
            return f"{value:.0f} {unit}"
    return f"{size} B"


def label() -> str:
    """Return the status line's memory field, or "" if there is no figure."""
    size = resident_bytes()
    return "" if size is None else f"Memory: {written(size)}"
