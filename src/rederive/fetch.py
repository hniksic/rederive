"""Download the original's demonstrations on the desktop.

The counterpart of `web/demos.js`: query all of a demonstration's URLs at once and take
the first answer, falling back to the whole diskette image when the archive's extractor
is down. Unlike the browser, the desktop needs no CORS relay and no handwritten zip
reader; it fetches the image directly and opens it with `zipfile`.

This only runs when a demonstration is requested and not already on disk.

Requests are raced so a stalled source doesn't delay one that answers, and they run in
daemon threads with timeouts so a stalled source can't block quitting either.
"""

from __future__ import annotations

import queue
import threading
import zipfile
from collections.abc import Callable
from io import BytesIO
from urllib.request import urlopen

from rederive.model.demos import DEMOS, IMAGE, Demo

__all__ = ["fetched"]

#: Timeout for one request: long enough for a slow server, short enough that a dead one
#: is detected while the user is still waiting.
TIMEOUT = 10.0

#: Size limits for a single demonstration and for the diskette image. The largest
#: demonstration is a few KB and the diskette about 300 KB, so anything larger is not
#: what we asked for.
LARGEST_FILE = 65536
LARGEST_IMAGE = 4 * 1024 * 1024

#: How an HTML error page starts. Some hosts serve error pages with status 200, so the
#: status alone can't be trusted.
HTML = b"<"

NOT_A_FILE = "what came back was not a file of the original"


def fetched(demo: Demo, keep: Callable[[str, bytes], None] | None = None) -> bytes:
    """Return the contents of `demo`. Raises OSError if no source works.

    Normally the file is extracted server-side by the archive. When that fails (it
    fails for all URLs at once, being the same service), the whole diskette is
    downloaded and unpacked here. Since the other demonstrations then come for free,
    they are passed to `keep` so they never need to be fetched again.
    """
    try:
        return _raced(demo.urls)
    except OSError as refused:
        try:
            return _unpacked(demo.file, keep)
        except OSError:
            # Report the primary source's error, which is the more informative one.
            raise refused from None


def _raced(urls: tuple[str, ...]) -> bytes:
    """Return the first successful response from `urls`, or raise the first error.

    The URLs are queried in parallel so that slow servers' waits don't add up. Threads
    that lose the race finish into a queue nobody reads.
    """
    answers: queue.Queue[tuple[bytes | None, OSError | None]] = queue.Queue()
    for url in urls:
        thread = threading.Thread(target=_answering, args=(url, answers), daemon=True)
        thread.start()
    refused: OSError | None = None
    for _ in urls:
        got, trouble = answers.get()
        if got is not None:
            return got
        assert trouble is not None
        refused = refused or trouble
    assert refused is not None
    raise refused


def _answering(url: str, answers: queue.Queue) -> None:
    """Fetch `url` and put the result or the error into `answers`."""
    try:
        answers.put((_got(url, LARGEST_FILE), None))
    except OSError as trouble:
        answers.put((None, trouble))


def _got(url: str, largest: int) -> bytes:
    """Return the contents of `url`, if they look like one of the original's files.

    Reads at most one byte past `largest`, so an oversized response is rejected
    without downloading it in full.
    """
    with urlopen(url, timeout=TIMEOUT) as answer:
        raw: bytes = answer.read(largest + 1)
    if not raw or len(raw) > largest or raw.startswith(HTML):
        raise OSError(NOT_A_FILE)
    return raw


def _unpacked(file: str, keep: Callable[[str, bytes], None] | None) -> bytes:
    """Extract `file` from the whole diskette, passing the other demos to `keep`."""
    raw = _got(IMAGE, LARGEST_IMAGE)
    try:
        image = zipfile.ZipFile(BytesIO(raw))
        found = {name.rpartition("/")[2].upper(): name for name in image.namelist()}
        wanted = found.get(file.upper())
        if wanted is None:
            raise OSError(f"the diskette has no {file}")
        if keep is not None:
            for demo in DEMOS:
                inside = found.get(demo.file.upper())
                if inside is not None and demo.file != file:
                    keep(demo.file, image.read(inside))
        return image.read(wanted)
    except (zipfile.BadZipFile, EOFError, ValueError) as trouble:
        # A broken zip means a failed download, so report it as OSError like the rest.
        raise OSError(f"the diskette could not be opened: {trouble}") from None
