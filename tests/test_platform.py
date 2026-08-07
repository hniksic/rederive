"""The seam between the program and the environment it happens to be running in.

Most of what the platform answers is checked where it is used - the memory gauge in
`test_memory.py`, the clipboard in `test_ui.py`, `--version` in `test_command_line.py`.
What is left here is the seam itself: that the desktop answers everything the protocol
asks, that saying which environment this is works, and storage, which is written down
ahead of the commands that will go through it and would otherwise have nothing holding
it to a filesystem that exists.
"""

import os

import pytest

from rederive import platform
from rederive.platform.desktop import Desktop, DesktopStorage
from rederive.platform.web import Web, WebStorage


def _asked_of(protocol):
    """Every call a protocol makes, which is what an implementation owes it."""
    return {
        name
        for name in vars(protocol)
        if not name.startswith("_") and callable(getattr(protocol, name))
    }


@pytest.mark.parametrize(
    ("protocol", "implementation"),
    [
        (platform.Platform, Desktop),
        (platform.Storage, DesktopStorage),
        (platform.Platform, Web),
        (platform.Storage, WebStorage),
    ],
    ids=lambda given: given.__name__,
)
def test_every_environment_answers_everything_the_protocol_asks(protocol, implementation):
    """Nothing type-checks this suite, so each implementation is asked by name.

    The browser's is asked here rather than where it is exercised because this is
    the question that has nothing to do with a browser: whether the two
    environments still answer the same calls, which is the whole claim the seam
    makes.
    """
    assert _asked_of(protocol) <= _asked_of(implementation)


def test_the_environment_is_the_desktop_unless_something_says_otherwise():
    assert isinstance(platform.current(), Desktop)


def test_an_environment_that_says_what_it_is_gets_asked_instead():
    """What a port does at boot, and the whole of how it takes over."""

    class Elsewhere:
        def copy(self, text):
            pass

    somewhere = Elsewhere()
    platform.use(somewhere)
    try:
        assert platform.current() is somewhere
    finally:
        platform.use(None)
    assert isinstance(platform.current(), Desktop)


def test_a_file_reads_back_as_the_bytes_it_was_written_as(tmp_path):
    """Written as text and read back as bytes, which is the asymmetry the callers have.

    A file Rederive writes is UTF-8; a file it reads may be the code page 437 the
    original wrote, and only the caller knows enough to try both. The line endings are
    the ones the system writes text with, which on Windows is what a Derive file has
    always carried.
    """
    storage = DesktopStorage()
    path = tmp_path / "sheet.mth"
    storage.write(path, "x^2 - 4\n")
    assert storage.read(path) == b"x^2 - 4" + os.linesep.encode()
    assert storage.exists(path)
    assert not storage.is_directory(path)
    assert storage.is_directory(tmp_path)


def test_a_directory_is_listed_in_order_and_one_that_is_not_there_is_empty(tmp_path):
    """A prompt completes off this, so a directory that will not be read offers nothing."""
    storage = DesktopStorage()
    (tmp_path / "second.mth").write_text("")
    (tmp_path / "first.mth").write_text("")
    assert list(storage.names(tmp_path)) == ["first.mth", "second.mth"]
    assert list(storage.names(tmp_path / "nowhere")) == []
