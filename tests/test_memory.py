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
    # Python with sympy loaded takes at least megabytes, and the suite never reaches
    # gigabytes.
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
    """The gauge covers the whole program, which runs as two processes.

    The test process registers itself as the worker, since it is certain to exist and
    be readable.
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
    # A nonexistent pid: the figure still reports the app alone.
    memory.register_worker(-1)
    try:
        assert memory.resident_bytes() == pytest.approx(alone, rel=0.5)
    finally:
        memory.register_worker(None)


def test_a_worker_that_can_only_be_read_from_inside_reports_instead():
    """In the browser, a Web Worker can't be measured from outside, so it reports itself.

    The gauge adds the reported figure just like a resident set it measured itself.
    """
    alone = memory.resident_bytes()
    if alone is None:
        pytest.skip(f"{sys.platform} does not report a resident set")
    memory.worker_holds(64 << 20)
    try:
        assert memory.resident_bytes() == alone + (64 << 20)
        # A later figure replaces the earlier one rather than adding to it.
        memory.worker_holds(96 << 20)
        assert memory.resident_bytes() == alone + (96 << 20)
    finally:
        memory.worker_holds(None)
    # Clearing the worker drops its figure.
    assert memory.resident_bytes() == alone


def test_a_worker_that_can_be_measured_is_measured():
    """A measurable worker process takes precedence over a reported figure.

    No real environment provides both; this pins the gauge's rule that a direct
    measurement beats a reported one.
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


def test_a_busy_worker_reports_out_of_its_collector_and_not_on_every_pass(monkeypatch):
    """How often the browser's gauge updates while a computation runs.

    No real garbage collection is triggered; the test only checks which collector
    callbacks lead to reading the heap, without depending on the machine.
    """
    import time

    from rederive.web import worker

    looks = []
    monkeypatch.setattr(worker, "_held", lambda: looks.append(1))
    # The end of a collection, with no recent reading: report.
    monkeypatch.setattr(worker, "_looked", 0.0)
    worker._collected("stop", {})
    assert len(looks) == 1
    # The start of a collection is not reported.
    worker._collected("start", {})
    assert len(looks) == 1
    # A collection right after a reading is skipped: an expansion can collect thousands
    # of times a second, far faster than anyone can read the gauge.
    monkeypatch.setattr(worker, "_looked", time.monotonic())
    worker._collected("stop", {})
    assert len(looks) == 1


def test_a_tab_reads_its_own_heap_and_says_nothing_outside_one():
    """Outside Pyodide there is no heap to read, so the web gauge is empty, not broken."""
    from rederive.platform.web import Web, heap

    assert heap() is None
    assert Web().process_bytes() is None
    assert not Web().measures_processes()


def test_a_platform_that_will_not_answer_leaves_the_field_empty(monkeypatch):
    """This is the normal case for an install without psutil, which is supported.

    With no reading available, the field is omitted rather than showing a wrong figure.
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
