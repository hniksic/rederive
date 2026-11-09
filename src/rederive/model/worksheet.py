"""The math file: a worksheet as plain text.

Rederive's own format, deliberately similar to the original's:

* one expression per line, in the canonical notation `write_expression` produces
  rather than as typed;
* a `;` line holds the annotation of the expression below it, so that, e.g., a
  Simplify result keeps its origin;
* a line exceeding the line length is continued with a trailing `~`.

Blank lines are ignored; one is written between records for readability. There is
no header or version, so files can be edited and diffed as text, and the original's
`.MTH` utility files are already in this format.

`Source.from_file` in the syntax package strips comments and joins continuations
when reading. This module interprets the comments and writes files.

All file access goes through `platform.current().storage()` (`pathlib` on the
desktop, a page-local store in a browser); path interpretation stays here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rederive import platform
from rederive.syntax.source import END_MARKER

#: The extension added to a name typed without one.
SUFFIX = ".mth"

#: Extensions of demonstration scripts and data files, which Transfer also reads.
DEMO_SUFFIX = ".dmo"
DATA_SUFFIX = ".dat"

COMMENT = ";"
CONTINUATION = "~"

#: A line consisting only of a quoted string is also prose. The original's plot
#: galleries use this for captions, e.g. `"A spike:"` above the expression.
QUOTE = '"'

#: The default line length, and the minimum: room for one character and the `~`.
LINE_LENGTH = 79
MINIMUM_LENGTH = 2


@dataclass(frozen=True)
class Record:
    """One expression of a file, with its annotation.

    An empty annotation (the usual case for authored lines) isn't written.
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

    `comment` is the annotation prefix, and `length` None disables continuation.
    Both exist for the source files written by `Transfer Save C` and similar,
    which use the language's comment marker and allow long lines.
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

    As in the original, lines break purely by character count, possibly inside a
    name or number; reading joins them back.
    """
    if length < MINIMUM_LENGTH or len(text) <= length:
        return [text]
    room = length - len(CONTINUATION)
    chunks = [text[at : at + room] for at in range(0, len(text), room)]
    return [chunk + CONTINUATION for chunk in chunks[:-1]] + chunks[-1:]


def annotations_of(text: str) -> dict[int, str]:
    """The annotation of each expression of `text`, keyed by its starting line.

    Lines are numbered from one, as `Source.locate` reports them. An annotation
    applies to the next expression; a later `;` line before it replaces the
    earlier one.
    """
    found: dict[int, str] = {}
    pending = ""
    for number, line in enumerate(_lines(text), start=1):
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


def _lines(text: str) -> list[str]:
    """A file's lines, up to the DOS Ctrl-Z end marker the original always wrote."""
    end = text.find(END_MARKER)
    return (text if end < 0 else text[:end]).splitlines()


def decoded(raw: bytes) -> str:
    """Decode a math file as UTF-8, falling back to code page 437.

    Rederive writes UTF-8; the original wrote code page 437, which matters only for
    non-ASCII glyphs such as Greek names. Code page 437 decodes any byte, so
    decoding never fails.
    """
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp437")


def text_of(path: Path) -> str:
    """A file's text, decoded by `decoded`. Raises on read errors."""
    return decoded(platform.current().storage().read(path))


def demonstration(path: Path) -> tuple[tuple[str, str], ...]:
    """The steps of the demonstration file at `path`. Raises on read errors."""
    return steps_of(text_of(path))


def steps_of(text: str) -> tuple[tuple[str, str], ...]:
    """The steps of a demonstration: each comment paired with the line below it.

    A demonstration is a math file whose comments are script text rather than
    annotations. `~` continuations are joined, and a step without a comment has a
    blank band.

    A line that is only a quoted string also counts as a comment, so the original's
    plot galleries (a caption, then the expression) work as demonstrations.

    Takes text rather than a path so a downloaded demonstration can run without
    being saved first.
    """
    comments = annotations_of(text)
    steps = []
    pending: list[str] = []
    caption = ""
    for number, line in enumerate(_lines(text), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(COMMENT):
            continue
        if _captioning(stripped):
            caption = stripped[1:-1].strip()
            continue
        pending.append(stripped)
        if stripped.endswith(CONTINUATION):
            pending[-1] = stripped[: -len(CONTINUATION)]
            continue
        # The comment belongs to the line the expression started on.
        start = number - len(pending) + 1
        steps.append((comments.get(start, caption), "".join(pending)))
        pending = []
        caption = ""
    return tuple(steps)


def _captioning(line: str) -> bool:
    """Whether a line is a single quoted string, and so a comment.

    Strings inside larger expressions, such as `["x", 1]` or `"a" + "b"`, don't
    count.
    """
    if len(line) < 2 or not line.startswith(QUOTE) or not line.endswith(QUOTE):
        return False
    return QUOTE not in line[1:-1]


def path_of(name: str, suffix: str = SUFFIX) -> Path:
    """The path for a typed file name.

    A name without an extension gets `suffix`, as the original supplied MTH, DAT,
    INI or a language extension. A leading `~` means the home directory.
    """
    path = Path(name.strip()).expanduser()
    return path if path.suffix else path.with_suffix(suffix)


def matches(name: str, suffix: str = SUFFIX) -> list[str]:
    """All names that `name` could be completed to, in order, including itself.

    Used by file prompts for browsing and completion. Offers files with extension
    `suffix` (case-insensitively, since the original's files are uppercase) and all
    directories, with a trailing separator. Dotfiles are offered only if `name`'s
    last component starts with a dot, as in a shell.

    The typed prefix, including `~`, is kept verbatim, so any result can replace
    the input directly. An exact match is included so the list stays consistent
    after a pick.

    Library files are not offered, since save prompts complete from the same list
    and must not suggest overwriting the program's own files. The library is
    reached by typing its names, which the help text and README document.
    """
    storage = platform.current().storage()
    head, separator, typed = name.rpartition("/")
    directory = Path(head + separator).expanduser() if separator else Path()
    offered = []
    for entry in storage.names(directory):
        if not entry.startswith(typed):
            continue
        if entry.startswith(".") and not typed.startswith("."):
            continue
        if storage.is_directory(directory / entry):
            offered.append(f"{head}{separator}{entry}/")
        elif entry.lower().endswith(suffix.lower()):
            offered.append(f"{head}{separator}{entry}")
    return offered


def completions(name: str, suffix: str = SUFFIX) -> list[str]:
    """The matches that extend `name`, i.e. `matches` without `name` itself."""
    return [grown for grown in matches(name, suffix) if len(grown) > len(name)]
