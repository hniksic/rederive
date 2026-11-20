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

`Source.from_file` in the syntax package reads the text: it joins continuations,
expands escapes, and pairs each expression with its annotation. This module decodes
files and writes them.

Wherever a math file is read, a Derive for Windows worksheet is read too: `read` tells
them apart by content, since the extension is not reliable, and `dfw` reads the binary
format.

All file access goes through `platform.current().storage()` (`pathlib` on the
desktop, a page-local store in a browser); path interpretation stays here.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from rederive import platform
from rederive.model import dfw
from rederive.syntax import Source, source_lines
from rederive.syntax.source import COMMENT

#: The extension added to a name typed without one.
SUFFIX = ".mth"

#: Extensions of demonstration scripts and data files, which Transfer also reads.
DEMO_SUFFIX = ".dmo"
DATA_SUFFIX = ".dat"

#: The extension of Derive for Windows worksheets, which are read but never written.
DFW_SUFFIX = ".dfw"

#: The extensions offered when choosing a worksheet to read.
READABLE = (SUFFIX, DFW_SUFFIX)

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

    No annotation (None, the usual case for authored lines) isn't written; an empty one
    is a bare `;`.
    """

    text: str
    annotation: str | None = None


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
        if annotations and record.annotation is not None:
            lines.append((comment + record.annotation).rstrip())
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


def decoded(raw: bytes) -> str:
    """Decode a math file as UTF-8, falling back to code page 437.

    Rederive writes UTF-8; Derive for DOS and Derive 5 wrote code page 437, which
    matters only for non-ASCII glyphs such as Greek names. Code page 437 decodes any
    byte, so decoding never fails. A byte order mark is dropped.

    Derive 6 writes a character beyond ASCII as an escape, `\x7f2211` for `∑`. Escapes
    are left for `Source.from_file`, since a line break may fall inside one.
    """
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp437")


def text_of(path: Path) -> str:
    """A file's text, decoded by `decoded`. Raises on read errors."""
    return decoded(platform.current().storage().read(path))


def read(path: Path) -> str | dfw.Worksheet:
    """The math file or Derive for Windows worksheet at `path`.

    Raises `OSError` on read errors, including a worksheet that can't be read whole.
    """
    raw = platform.current().storage().read(path)
    return dfw.read(raw) if dfw.is_worksheet(raw) else decoded(raw)


def demonstration(path: Path) -> tuple[tuple[str, str], ...]:
    """The steps of the demonstration file at `path`. Raises on read errors."""
    return steps_of(text_of(path))


def steps_of(text: str) -> tuple[tuple[str, str], ...]:
    """The steps of a demonstration: each comment paired with the line below it.

    A demonstration is a math file whose comments are script text rather than
    annotations, and it is read as one (`Source.from_file`). A step without a comment
    has a blank band.

    A line that is only a quoted string also counts as a comment, so the original's
    plot galleries (a caption, then the expression) work as demonstrations.

    Takes text rather than a path so a downloaded demonstration can run without
    being saved first.
    """
    source = Source.from_file(text)
    steps = []
    caption = ""
    for start, stop in source_lines(source):
        line = source.text[start:stop].strip()
        if _captioning(line):
            caption = line[1:-1].strip()
            continue
        steps.append((source.annotations.get(start) or caption, line))
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


#: The separators of the parts of a path as `LOAD` names it, and a drive letter.
_SEPARATORS = re.compile(r"[\\/]")
_DRIVE = re.compile(r"[A-Za-z]:")

#: The subdirectory of the working directory where `LOAD` also looks, as Derive kept its
#: utility files in its `Math` directory.
MATH_DIRECTORY = "math"


def file_name(written: str) -> str:
    """The name of the file a path names, as `LOAD` writes it: its last part."""
    return _SEPARATORS.split(written)[-1]


def located(written: str, near: Sequence[Path] = ()) -> Path | None:
    """The file `LOAD(written)` loads, or None if there is none.

    Derive's worksheets name files by Windows paths of the machine they were made on
    (`C:\\Derive6\\Math\\Number.mth`, `..\\math\\vector`), so the path is split at
    either kind of slash and a drive letter is dropped. It is looked for as given, a
    relative path against the first of `near` (the directory of the file being read),
    or the working directory without one; then by its name alone in each of `near`, the
    working directory and the latter's `Math` subdirectory. Names are compared
    whatever their case, and a name without an extension is looked for with `.mth`,
    then `.dfw`. Only math files and worksheets are loaded: a name with another
    extension finds nothing.
    """
    parts = _SEPARATORS.split(written.strip())
    if parts and _DRIVE.match(parts[0]):
        parts[0] = parts[0][2:]
    *directories, name = parts
    if not name or name in (".", ".."):
        return None
    suffix = Path(name).suffix.lower()
    if suffix and suffix not in READABLE:
        return None
    names = [name] if suffix else [name + extension for extension in READABLE]
    if directories and not directories[0]:
        given = Path(os.sep).joinpath(*directories[1:])
    else:
        given = (near[0] if near else Path()).joinpath(*directories)
    places = [given, *near, Path(), *_math_directories()]
    for place in dict.fromkeys(places):
        found = _named_in(place, names)
        if found is not None:
            return found
    return None


def _math_directories() -> list[Path]:
    """The working directory's `Math` subdirectory, in whatever case it is spelled."""
    storage = platform.current().storage()
    return [
        Path(entry)
        for entry in storage.names(Path())
        if entry.lower() == MATH_DIRECTORY and storage.is_directory(Path(entry))
    ]


def _named_in(directory: Path, names: Sequence[str]) -> Path | None:
    """The file in `directory` with the first of `names`, whatever the case; one with
    the case as written comes first."""
    storage = platform.current().storage()
    entries = storage.names(directory)
    for name in names:
        matching = [entry for entry in entries if entry.lower() == name.lower()]
        for entry in sorted(matching, key=lambda entry: entry != name):
            if not storage.is_directory(directory / entry):
                return directory / entry
    return None


def same_file(one: Path, other: Path) -> bool:
    """Whether two paths name the same file."""
    return os.path.abspath(one) == os.path.abspath(other)


def path_of(name: str, suffix: str = SUFFIX) -> Path:
    """The path for a typed file name.

    A name without an extension gets `suffix`, as the original supplied MTH, DAT,
    INI or a language extension. A leading `~` means the home directory.
    """
    path = Path(name.strip()).expanduser()
    return path if path.suffix else path.with_suffix(suffix)


def matches(name: str, suffix: str | tuple[str, ...] = SUFFIX) -> list[str]:
    """All names that `name` could be completed to, in order, including itself.

    Used by file prompts for browsing and completion. Offers files with extension
    `suffix`, or with any of several, case-insensitively since the original's files
    are uppercase, and all directories, with a trailing separator. Dotfiles are
    offered only if `name`'s last component starts with a dot, as in a shell.

    The typed prefix, including `~`, is kept verbatim, so any result can replace
    the input directly. An exact match is included so the list stays consistent
    after a pick.

    Library files are not offered, since save prompts complete from the same list
    and must not suggest overwriting the program's own files. The library is
    reached by typing its names, which the help text and README document.
    """
    storage = platform.current().storage()
    suffixes = tuple(
        offered.lower() for offered in ((suffix,) if isinstance(suffix, str) else suffix)
    )
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
        elif entry.lower().endswith(suffixes):
            offered.append(f"{head}{separator}{entry}")
    return offered


def completions(name: str, suffix: str | tuple[str, ...] = SUFFIX) -> list[str]:
    """The matches that extend `name`, i.e. `matches` without `name` itself."""
    return [grown for grown in matches(name, suffix) if len(grown) > len(name)]
