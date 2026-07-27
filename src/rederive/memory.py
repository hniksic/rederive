"""How much memory this process is holding, asked of the platform directly.

The status line shows the figure where the original showed its muLISP heap
gauge. A number that only decorates a status line is not worth a dependency,
so each platform is asked the cheapest way it answers: a line of `/proc` on
Linux, one `task_info` call on macOS, one `GetProcessMemoryInfo` call on
Windows. Anywhere else, and anywhere a call fails, the answer is None and the
field stays empty rather than showing a figure that might be wrong.
"""

from __future__ import annotations

import ctypes
import os
import sys

#: Bytes in the units a size is written in, largest first. Resident sets are
#: measured in megabytes on every machine this runs on, but a small one reads
#: better in kilobytes and a large one in gigabytes.
_UNITS = (("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10))


class _MachTaskBasicInfo(ctypes.Structure):
    """macOS `mach_task_basic_info`, of which only `resident_size` is wanted."""

    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_max", ctypes.c_uint64),
        ("user_time", ctypes.c_int32 * 2),
        ("system_time", ctypes.c_int32 * 2),
        ("policy", ctypes.c_int32),
        ("suspend_count", ctypes.c_int32),
    ]


#: The `task_info` flavor that fills in the structure above.
_MACH_TASK_BASIC_INFO = 20


class _ProcessMemoryCounters(ctypes.Structure):
    """Windows `PROCESS_MEMORY_COUNTERS`, of which `WorkingSetSize` is wanted."""

    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("page_fault_count", ctypes.c_uint32),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    ]


def _from_proc() -> int | None:
    """Resident pages from `/proc/self/statm`, on Linux and kin.

    The second field is the resident set in pages, which is the one number
    wanted and the cheapest of the several places `/proc` keeps it.
    """
    try:
        with open("/proc/self/statm", "rb") as handle:
            fields = handle.read().split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def _from_mach() -> int | None:
    """Resident bytes from the mach kernel, on macOS."""
    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        libc.mach_task_self.restype = ctypes.c_uint32
        info = _MachTaskBasicInfo()
        count = ctypes.c_uint32(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint32))
        failed = libc.task_info(
            libc.mach_task_self(),
            ctypes.c_uint32(_MACH_TASK_BASIC_INFO),
            ctypes.byref(info),
            ctypes.byref(count),
        )
        return None if failed else int(info.resident_size)
    except (OSError, AttributeError, ValueError):
        return None


def _from_psapi() -> int | None:
    """The working set, which is what Windows calls a resident set."""
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        # Vista onward exports the call from kernel32 under a K32 name; the
        # older home is psapi, still there for programs that ask for it.
        try:
            call = kernel32.K32GetProcessMemoryInfo
        except AttributeError:
            call = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo  # type: ignore[attr-defined]
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        ok = call(
            ctypes.c_void_p(kernel32.GetCurrentProcess()),
            ctypes.byref(counters),
            ctypes.sizeof(counters),
        )
        return int(counters.working_set_size) if ok else None
    except (OSError, AttributeError, ValueError):
        return None


def resident_bytes() -> int | None:
    """This process's resident set, or None where the platform will not say."""
    if sys.platform == "darwin":
        return _from_mach()
    if sys.platform == "win32":
        return _from_psapi()
    # Linux, and any other system that mounts a Linux-shaped `/proc`.
    return _from_proc()


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
