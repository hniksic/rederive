"""The status line's memory field: the reading, and how it is written."""

import os
import re
import sys

import pytest
from screen import text_of

from rederive import memory
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
        (0, "0B"),
        (999, "999B"),
        (1 << 10, "1K"),
        (99 << 10, "99K"),
        (1 << 20, "1M"),
        ((3 << 20) + (512 << 10), "4M"),
        (150 << 20, "150M"),
        (1 << 30, "1G"),
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


async def test_the_status_line_shows_what_the_process_holds():
    app = RederiveApp()
    async with app.run_test():
        status = text_of(app.query_one("#status")).plain
        if memory.resident_bytes() is None:
            assert "Memory:" not in status
        else:
            assert re.search(r"Memory:\d+[BKMG]", status)
