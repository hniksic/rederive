"""The math file: a worksheet as plain text.

Rederive's own format rather than the original's, though the two look alike on
purpose. A file is lines of text and nothing else:

* one expression per line, in the canonical author notation `write_expression`
  produces - what the expression is, rather than how it happened to be typed;
* a `;` line carries the annotation of the expression under it, so that a
  Simplify's answer comes back knowing where it came from;
* a line too long for the file's line length is broken and continued with a
  trailing `~`.

Blank lines carry nothing; one is written between records because it makes the
file easier to read. That is the whole format: no header, no version, nothing
that has to be kept in step with the program. A worksheet can be read, edited
and diffed with any text editor, and the `.MTH` utility files the original
shipped are already in it.

Reading is the syntax package's job - `Source.from_file` strips the comments
and joins the continuations. This module says what those comments *meant*, and
how a file is written.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

#: What a file is called when the name typed has no extension of its own.
SUFFIX = ".mth"

#: The two other extensions Transfer reads, for the two commands that read
#: something else: a demonstration script and a file of numbers.
DEMO_SUFFIX = ".dmo"
DATA_SUFFIX = ".dat"

COMMENT = ";"
CONTINUATION = "~"

#: The line length a file is written to, and the shortest one that can be
#: written at all: a line needs room for a character and the tilde that
#: continues it.
LINE_LENGTH = 79
MINIMUM_LENGTH = 2


@dataclass(frozen=True)
class Record:
    """One expression of a file, with the annotation written above it.

    An empty annotation is written as nothing: a line the user authored is the
    ordinary case and needs no comment to say so.
    """

    text: str
    annotation: str = ""


def write(
    records: Iterable[Record],
    length: int | None = LINE_LENGTH,
    annotations: bool = True,
    comment: str = COMMENT,
) -> str:
    """The file text for `records`, one record per paragraph.

    `comment` is what an annotation is written behind, and `length` None asks
    for no continuation at all. Both are for the source files `Transfer Save C`
    and its neighbours write: those carry the same records behind the target
    language's own comment marker, and none of the four minds a long line.
    """
    lines: list[str] = []
    for record in records:
        if annotations and record.annotation:
            lines.append(comment + record.annotation)
        lines.extend([record.text] if length is None else _continued(record.text, length))
        lines.append("")
    return "\n".join(lines)


def _continued(text: str, length: int) -> list[str]:
    """`text` as lines of at most `length`, each but the last ending in `~`.

    The break is by character count and nothing else, as the original's is: it
    may cut a name or a numeral in half, and joining the lines back together is
    what makes that harmless.
    """
    if length < MINIMUM_LENGTH or len(text) <= length:
        return [text]
    room = length - len(CONTINUATION)
    chunks = [text[at : at + room] for at in range(0, len(text), room)]
    return [chunk + CONTINUATION for chunk in chunks[:-1]] + chunks[-1:]


def annotations_of(text: str) -> dict[int, str]:
    """The annotation of each expression of `text`, by the line it starts on.

    Lines are numbered from one, as `Source.locate` reports them. An annotation
    applies to the next expression under it, and a second `;` line before that
    expression replaces the first rather than adding to it.
    """
    found: dict[int, str] = {}
    pending = ""
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(COMMENT):
            pending = stripped[len(COMMENT) :].strip()
        else:
            if pending:
                found[number] = pending
            pending = ""
    return found


def text_of(path: Path) -> str:
    """A file's text. Raises, as reading a file does, before anything changes.

    UTF-8 is what Rederive writes. A file the original wrote is code page 437,
    which only shows in one where a glyph left ASCII - a Greek variable name,
    almost always - and that is what the fallback reads. Code page 437 decodes
    any byte at all, so a file is never refused for what is in it.
    """
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp437")


def demonstration(path: Path) -> tuple[tuple[str, str], ...]:
    """The steps of a demonstration file: each comment with the line under it.

    A demonstration file is a math file whose comments carry the script rather
    than an annotation, so it is read the same way and the comments are what is
    kept. A `~` continuation still joins its lines, and a step with no comment
    above it is one whose band is blank.
    """
    text = text_of(path)
    comments = annotations_of(text)
    steps = []
    pending: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(COMMENT):
            continue
        pending.append(stripped)
        if stripped.endswith(CONTINUATION):
            pending[-1] = stripped[: -len(CONTINUATION)]
            continue
        # The comment belongs to the line the expression started on.
        start = number - len(pending) + 1
        steps.append((comments.get(start, ""), "".join(pending)))
        pending = []
    return tuple(steps)


def path_of(name: str, suffix: str = SUFFIX) -> Path:
    """The file a typed name asks for.

    A name with no extension gets `suffix`, the way the original supplied MTH,
    DAT, INI or the extension of whichever language was being saved. A leading
    `~` is the shell's home directory here, never a continuation: the tilde
    only continues lines inside a file.
    """
    path = Path(name.strip()).expanduser()
    return path if path.suffix else path.with_suffix(suffix)
