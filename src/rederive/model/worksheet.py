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

Nothing here opens a file itself. Reading one, listing a directory and asking
whether a name is there all go through `platform.current().storage()`, which on
a desktop is `pathlib` with one call in front of it and in a browser is a store
the page keeps. What a path *means* stays here, since that is the same question
wherever the file lands.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rederive import platform
from rederive.syntax.source import END_MARKER

#: What a file is called when the name typed has no extension of its own.
SUFFIX = ".mth"

#: The two other extensions Transfer reads, for the two commands that read
#: something else: a demonstration script and a file of numbers.
DEMO_SUFFIX = ".dmo"
DATA_SUFFIX = ".dat"

COMMENT = ";"
CONTINUATION = "~"

#: The other way a file carries a line of prose: a line that is nothing but a
#: quoted string. The original's own plot galleries are written that way -
#: `"A spike:"` above the expression it names - because a string is an
#: expression there and a worksheet had nowhere else to put a caption.
QUOTE = '"'

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
    """A file's lines, as the reader sees them.

    The Ctrl-Z that ends a DOS text file is not one of them, and neither is
    anything behind it: every file the original wrote carries one, and a line
    made of it would be read as an expression nobody wrote.
    """
    end = text.find(END_MARKER)
    return (text if end < 0 else text[:end]).splitlines()


def decoded(raw: bytes) -> str:
    """What a math file's bytes say, whichever of the two encodings wrote them.

    UTF-8 is what Rederive writes. A file the original wrote is code page 437,
    which only shows in one where a glyph left ASCII - a Greek variable name,
    almost always - and that is what the fallback reads. Code page 437 decodes
    any byte at all, so a file is never refused for what is in it.
    """
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp437")


def text_of(path: Path) -> str:
    """A file's text. Raises, as reading a file does, before anything changes.

    The two encodings are `decoded`'s; what is here is the reading itself.
    """
    return decoded(platform.current().storage().read(path))


def demonstration(path: Path) -> tuple[tuple[str, str], ...]:
    """The steps of the demonstration file at `path`. Raises, as reading does."""
    return steps_of(text_of(path))


def steps_of(text: str) -> tuple[tuple[str, str], ...]:
    """The steps of a demonstration: each comment with the line under it.

    A demonstration file is a math file whose comments carry the script rather
    than an annotation, so it is read the same way and the comments are what is
    kept. A `~` continuation still joins its lines, and a step with no comment
    above it is one whose band is blank.

    A line that is nothing but a quoted string is a comment here too. The
    original's plot galleries are scripts of that shape - a caption, then the
    expression it names - and they are demonstrations in everything but the
    extension they were given.

    The text rather than the file, because a demonstration that has just been
    downloaded is text before it is a file, and is to run whether or not the
    directory it landed in would take it.
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
    """Whether a line is a quoted string and nothing else, and so a comment.

    Only a whole line counts, and only one string on it. A string is an
    expression like any other where it stands beside one - `["x", 1]` is a
    vector with a string in it, `"a" + "b"` is a sum of two - and what makes
    this one prose is that the line is the string and nothing more.
    """
    if len(line) < 2 or not line.startswith(QUOTE) or not line.endswith(QUOTE):
        return False
    return QUOTE not in line[1:-1]


def path_of(name: str, suffix: str = SUFFIX) -> Path:
    """The file a typed name asks for.

    A name with no extension gets `suffix`, the way the original supplied MTH,
    DAT, INI or the extension of whichever language was being saved. A leading
    `~` is the shell's home directory here, never a continuation: the tilde
    only continues lines inside a file.
    """
    path = Path(name.strip()).expanduser()
    return path if path.suffix else path.with_suffix(suffix)


def matches(name: str, suffix: str = SUFFIX) -> list[str]:
    """Every name `name` could mean, in order, itself included.

    What a file prompt browses and completes from. Only what the command can
    use is on offer - files whose extension is `suffix`, matched however it is
    cased, since the original's own files are named in capitals - along with
    every directory, offered with the separator after it so that going on from
    there carries on inside it. A name beginning with a dot is offered only to
    letters that begin with one, as a shell does.

    What was typed is kept as it stands, `~` and all: only the last component
    grows, so any of the answers can go straight back on the line. A name that
    is already one of them is among them, which is what lets the list stay up
    and stay honest once a name has been taken from it.

    What is *not* on offer is the library: the same list is what a save prompt
    completes from, and offering a name that names a file of the program's own
    in a prompt that is about to write one of yours would be a lie. The library
    is reached by typing the name, and named in the help text and the README so
    that it can be.
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
    """The matches that would grow `name`, which is what completing it can use.

    `matches` less the name itself: a name typed out in full is no completion
    of itself, and offering it as one would let a keystroke look like it had
    done nothing.
    """
    return [grown for grown in matches(name, suffix) if len(grown) > len(name)]
