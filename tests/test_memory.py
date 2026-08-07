"""The status line's memory field: the reading, and how it is written."""

import os
import re
import sys

import pytest
from screen import text_of

from rederive import memory, platform
from rederive.ui.app import RederiveApp


def test_this_process_holds_a_plausible_amount():
    size = memory.resident_bytes()
    if size is None:
        pytest.skip(f"{sys.platform} does not report a resident set")
    # A Python interpreter with sympy loaded is megabytes at the very least,
    # and no run of the suite is a gigabyte.
    assert 1 << 20 < size < 8 << 30


@pytest.mark.parametrize(
    ("size", "written"),
    [
        (0, "0 B"),
        (999, "999 B"),
        (1 << 10, "1 KiB"),
        (99 << 10, "99 KiB"),
        (1 << 20, "1 MiB"),
        ((3 << 20) + (512 << 10), "4 MiB"),
        (150 << 20, "150 MiB"),
        (1 << 30, "1.0 GiB"),
        ((1 << 30) + (100 << 20), "1.1 GiB"),
        (9 << 30, "9.0 GiB"),
        (10 << 30, "10 GiB"),
        (64 << 30, "64 GiB"),
    ],
    ids=str,
)
def test_a_size_is_written_in_its_largest_whole_unit(size, written):
    assert memory.written(size) == written


def test_the_gauge_counts_the_worker_in_where_there_is_one():
    """The gauge is about the program, and the program is two processes.

    This one stands in for the worker as its own worker, which is the one
    process certain to be there and certain to be readable.
    """
    alone = memory.resident_bytes()
    if alone is None:
        pytest.skip(f"{sys.platform} does not report a resident set")
    memory.register_worker(os.getpid())
    try:
        assert memory.resident_bytes() > alone
    finally:
        memory.register_worker(None)
    assert memory.resident_bytes() <= alone * 1.5


def test_a_worker_that_is_gone_is_left_out_rather_than_refused():
    alone = memory.resident_bytes()
    if alone is None:
        pytest.skip(f"{sys.platform} does not report a resident set")
    # A pid nothing answers to: the figure is still true about the app.
    memory.register_worker(-1)
    try:
        assert memory.resident_bytes() == pytest.approx(alone, rel=0.5)
    finally:
        memory.register_worker(None)


def test_a_worker_that_can_only_be_read_from_inside_reports_instead():
    """The browser's half of the gauge: a Web Worker nothing outside can measure.

    The figure arrives from the worker rather than being taken off it, and the
    gauge adds it in exactly as it adds a resident set it read itself.
    """
    alone = memory.resident_bytes()
    if alone is None:
        pytest.skip(f"{sys.platform} does not report a resident set")
    memory.worker_holds(64 << 20)
    try:
        assert memory.resident_bytes() == alone + (64 << 20)
        # A later figure replaces the earlier one: it is the same heap read
        # again, not a second heap.
        memory.worker_holds(96 << 20)
        assert memory.resident_bytes() == alone + (96 << 20)
    finally:
        memory.worker_holds(None)
    # And a worker that has gone takes its figure with it.
    assert memory.resident_bytes() == alone


def test_a_worker_that_can_be_measured_is_measured():
    """A process to read beats a figure somebody sent, there being one truth.

    Neither environment does both, so this is about the gauge and not about
    either of them: what the two calls mean is `go and read this` and `this is
    what it said`, and a reading nobody has to be asked for is the better one.
    """
    alone = memory.resident_bytes()
    if alone is None:
        pytest.skip(f"{sys.platform} does not report a resident set")
    memory.register_worker(os.getpid())
    memory.worker_holds(64 << 20)
    try:
        assert memory.resident_bytes() == pytest.approx(2 * alone, rel=0.5)
    finally:
        memory.register_worker(None)
        memory.worker_holds(None)


def test_a_tab_reads_its_own_heap_and_says_nothing_outside_one():
    """The reading a browser has, asked for where there is no browser.

    Outside Pyodide there is no heap to read, and the gauge is empty rather
    than broken - which is also what the whole of this suite sees, since it
    runs on a desktop.
    """
    from rederive.platform.web import Web, heap

    assert heap() is None
    assert Web().process_bytes() is None
    assert not Web().measures_processes()


def test_a_platform_that_will_not_answer_leaves_the_field_empty(monkeypatch):
    """An install without psutil is the ordinary shape of this, and it is supported.

    The gauge is the one part of the program that has nothing to fall back on, so
    what it does instead is say nothing - a field that is not there reads better
    than a figure that is not true.
    """
    environment = type(platform.current())
    monkeypatch.setattr(environment, "process_bytes", lambda self, pid=None: None)
    assert memory.resident_bytes() is None
    assert memory.label() == ""


async def test_the_status_line_shows_what_the_process_holds():
    app = RederiveApp()
    async with app.run_test():
        status = text_of(app.query_one("#status")).plain
        if memory.resident_bytes() is None:
            assert "Memory:" not in status
        else:
            assert re.search(r"Memory: \d+(\.\d)? (B|KiB|MiB|GiB)", status)
