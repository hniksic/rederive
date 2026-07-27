"""The status line's memory field: the reading, and how it is written."""

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


async def test_the_status_line_shows_what_the_process_holds():
    app = RederiveApp()
    async with app.run_test():
        status = text_of(app.query_one("#status")).plain
        if memory.resident_bytes() is None:
            assert "Memory:" not in status
        else:
            assert re.search(r"Memory:\d+[BKMG]", status)
