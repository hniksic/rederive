"""The platform layer between the program and its environment.

Most platform services are tested where they are used: the memory gauge in
`test_memory.py`, the clipboard in `test_ui.py`, `--version` in `test_command_line.py`.
This file tests the layer itself: that implementations cover the protocols, that the
current environment can be switched, and storage, which no command uses yet and would
otherwise go untested against a real filesystem.
"""

import os

import pytest

from rederive import platform
from rederive.platform.desktop import Desktop, DesktopStorage
from rederive.platform.web import Web, WebStorage


def _asked_of(protocol):
    """The public methods a protocol declares, which an implementation must provide."""
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
    """Nothing type-checks this suite, so check method names directly.

    The web implementation is checked here rather than in the browser tests because
    this needs no browser: it only asks whether both environments provide the same
    calls.
    """
    assert _asked_of(protocol) <= _asked_of(implementation)


def test_the_environment_is_the_desktop_unless_something_says_otherwise():
    assert isinstance(platform.current(), Desktop)


def test_an_environment_that_says_what_it_is_gets_asked_instead():
    """A port installs its platform at boot, and that is all it takes."""

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
    """Files are written as text but read back as bytes.

    Rederive writes UTF-8, but a file it reads may be in the original's code page 437,
    and only the caller knows enough to try both. Line endings are the system's, which
    on Windows matches what Derive files have always used.
    """
    storage = DesktopStorage()
    path = tmp_path / "sheet.mth"
    storage.write(path, "x^2 - 4\n")
    assert storage.read(path) == b"x^2 - 4" + os.linesep.encode()
    assert storage.exists(path)
    assert not storage.is_directory(path)
    assert storage.is_directory(tmp_path)


def test_a_directory_is_listed_in_order_and_one_that_is_not_there_is_empty(tmp_path):
    """File prompts complete from this, so an unreadable directory offers nothing."""
    storage = DesktopStorage()
    (tmp_path / "second.mth").write_text("")
    (tmp_path / "first.mth").write_text("")
    assert list(storage.names(tmp_path)) == ["first.mth", "second.mth"]
    assert list(storage.names(tmp_path / "nowhere")) == []
