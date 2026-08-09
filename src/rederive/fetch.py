"""The original's demonstrations, fetched the way a machine with sockets fetches.

The other half of `web/demos.js`, and it does the same two things: ask every
address a demonstration has at once and take whichever answers, and fall back
on the whole diskette when the archive's extractor is the thing that is down.
What it does *not* do is the half of that file which is the browser's own
predicament - a relay in front of the diskette, and a zip reader written by
hand. A page may not read that address at all, having no CORS headers to go on;
a desktop reads it directly, and opens the zip with the one in the standard
library.

Nothing here is reached unless one of the nine is asked for and is not on the
disk already, so a program that is never shown off never opens a socket.

The threads are daemons and the waits are bounded. A source that has stopped
answering must not hold the demonstration up behind a source that is answering,
which is why they are raced; and it must not hold *quitting* up either, once
the race is over and nobody is listening to it any more.
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

#: How long one address is given to answer. Long enough for a server that is
#: thinking, short enough that a road which is closed is found to be closed
#: while the user is still expecting something.
TIMEOUT = 10.0

#: What a demonstration may be before it is taken for one, and what the whole
#: diskette may be. The largest of the nine is a few kilobytes and the diskette
#: is three hundred; anything past these is something else wearing the name.
LARGEST_FILE = 65536
LARGEST_IMAGE = 4 * 1024 * 1024

#: What an apology begins with. A host that answers an error page answers it
#: with a 200, so the status alone would let one onto the worksheet.
HTML = b"<"

NOT_A_FILE = "what came back was not a file of the original"


def fetched(demo: Demo, keep: Callable[[str, bytes], None] | None = None) -> bytes:
    """The bytes of `demo`, off whichever road is open. Raises OSError if none is.

    The ordinary road is the file itself, a few kilobytes out of a diskette
    image that the archive opens at its end. When that road is shut - and it is
    one piece of machinery for both images, so it shuts for both at once - the
    whole diskette comes over instead and is opened here. Every demonstration
    in it is then paid for, which is what `keep` is for: the eight that were not
    asked for are handed over to whoever is holding on to them, so that none of
    them is ever fetched twice.
    """
    try:
        return _raced(demo.urls)
    except OSError as refused:
        try:
            return _unpacked(demo.file, keep)
        except OSError:
            # The archive's own refusal rather than the diskette's: the first
            # road is the one that was meant to work, and its failure is the one
            # worth reading.
            raise refused from None


def _raced(urls: tuple[str, ...]) -> bytes:
    """The first of `urls` to answer with a file, or the first refusal there was.

    All at once rather than one after the other: the slow case is a server
    thinking, and asking in turn would add the waits together. The threads that
    lose the race are left to finish into a queue nobody reads.
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
    """One runner in the race, reporting whichever way it went."""
    try:
        answers.put((_got(url, LARGEST_FILE), None))
    except OSError as trouble:
        answers.put((None, trouble))


def _got(url: str, largest: int) -> bytes:
    """What is at `url`, if what is there can be a file of the original's.

    One byte more than the largest is read, so that something far too big is
    refused for its size rather than pulled down in full and refused after.
    """
    with urlopen(url, timeout=TIMEOUT) as answer:
        raw: bytes = answer.read(largest + 1)
    if not raw or len(raw) > largest or raw.startswith(HTML):
        raise OSError(NOT_A_FILE)
    return raw


def _unpacked(file: str, keep: Callable[[str, bytes], None] | None) -> bytes:
    """`file` out of the whole diskette, the rest of it kept on the way past."""
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
        # A zip that will not open is a download that went wrong, which is what
        # every other failure on this road already reads as.
        raise OSError(f"the diskette could not be opened: {trouble}") from None
