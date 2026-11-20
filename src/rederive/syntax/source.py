"""Preprocessing of source text, with offsets mapping back to the original.

The parser takes `str`; decoding is up to the caller.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    """A piece of Derive text, prepared for lexing.

    `text` is what the lexer and parser index; `original` is the caller's input.
    Preprocessing deletes characters, so offsets into `text` don't match `original`;
    `locate` translates them.
    """

    text: str
    original: str
    origin: str | None = None
    offsets: tuple[int, ...] = field(default=(), repr=False)
    """For each character of `text`, its index in `original`."""
    annotations: Mapping[int, str] = field(default_factory=dict, repr=False)
    """For a file, the annotation of each expression line that has one, keyed by where
    the line starts in `text`."""

    def map_offset(self, offset: int) -> int:
        """Where an offset into `text` falls in `original`."""
        if not self.offsets:
            return min(offset, len(self.original))
        if offset < len(self.offsets):
            return self.offsets[offset]
        return len(self.original)

    def locate(self, offset: int) -> tuple[int, int]:
        """(line, column), 1-based, in `original`."""
        index = self.map_offset(offset)
        line_start = self.original.rfind("\n", 0, index) + 1
        return self.original.count("\n", 0, index) + 1, index - line_start + 1

    @classmethod
    def from_line(cls, text: str) -> Source:
        """The author line: no comments, no splices, identity offset map."""
        return cls(text=text, original=text, origin=None, offsets=())

    @classmethod
    def from_file(cls, text: str, origin: str | None = None) -> Source:
        """A math or demonstration file's decoded text, prepared for reading.

        The steps run in this order, each on the result of the one before:

        1. cut the text at the DOS end marker, Ctrl-Z;
        2. normalize CR LF, CR and LF line ends to LF;
        3. join each line ending in `~` to the next, on raw characters, so that a line
           Derive broke inside a name, a number, a string or an escape is whole again;
        4. expand the `\\x7fXXXX` escapes;
        5. classify the lines: a line starting with `;` is an annotation, which applies
           to the next expression line (a later one before it replaces it) and is
           blanked; in any other line a `;` outside strings and square brackets starts a
           comment, which is cut off.

        Strings are tracked on the joined line, so a `;` in a string that a wrap split
        is still data. Blank lines remain, so `source_lines` finds each expression
        line, and `annotations` holds the annotation of each.
        """
        body, offsets = _identity(text)
        body, offsets = _cut_at_end_marker(body, offsets)
        body, offsets = _normalize_line_endings(body, offsets)
        body, offsets = _splice_continuations(body, offsets)
        body, offsets = _decode_escapes(body, offsets)
        body, offsets, annotations = _classify_lines(body, offsets)
        return cls(
            text=body,
            original=text,
            origin=origin,
            offsets=tuple(offsets),
            annotations=annotations,
        )


#: Derive 6's escape for a character outside ASCII: DEL and four hex digits, so that
#: `\x7f2211` is U+2211, the summation sign. Its math files and worksheets use it.
_ESCAPE = re.compile("\x7f([0-9A-Fa-f]{4})")


def unescaped(text: str) -> str:
    """`text` with each Derive 6 escape replaced by the character it stands for.

    An escape for a control character or a surrogate is left as it is: neither can
    stand in a line of Derive text.
    """
    return _decode_escapes(text, [])[0] if "\x7f" in text else text


def _decode_escapes(text: str, offsets: list[int]) -> tuple[str, list[int]]:
    """Decode the escapes; a character from one maps to where the escape started.

    `offsets` may be empty when the caller has no use for them.
    """
    out: list[str] = []
    kept: list[int] = []
    index = 0
    for found in _ESCAPE.finditer(text):
        character = chr(int(found.group(1), 16))
        if unicodedata.category(character) in ("Cc", "Cs"):
            continue
        out.append(text[index : found.start()])
        out.append(character)
        if offsets:
            kept.extend(offsets[index : found.start() + 1])
        index = found.end()
    out.append(text[index:])
    if offsets:
        kept.extend(offsets[index:])
    return "".join(out), kept


#: End-of-file marker of a DOS text file. Every file the original wrote has one, and
#: anything after it is ignored.
END_MARKER = "\x1a"


def _identity(text: str) -> tuple[str, list[int]]:
    return text, list(range(len(text)))


def _cut_at_end_marker(text: str, offsets: list[int]) -> tuple[str, list[int]]:
    """Drop the Ctrl-Z that ends a DOS text file, and anything after it."""
    end = text.find(END_MARKER)
    if end < 0:
        return text, offsets
    return text[:end], offsets[:end]


def _normalize_line_endings(text: str, offsets: list[int]) -> tuple[str, list[int]]:
    """Accept CRLF, LF and CR; normalize to LF."""
    out: list[str] = []
    kept: list[int] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\r":
            out.append("\n")
            kept.append(offsets[index])
            index += 2 if text.startswith("\r\n", index) else 1
            continue
        out.append(char)
        kept.append(offsets[index])
        index += 1
    return "".join(out), kept


#: What starts an annotation line, and a comment after an expression.
COMMENT = ";"


def _classify_lines(
    text: str, offsets: list[int]
) -> tuple[str, list[int], dict[int, str]]:
    """Blank the annotation lines and cut the comments off the expression lines.

    Returns the text, its offsets, and the annotation of each expression line, keyed by
    where the line starts in the returned text.
    """
    out: list[str] = []
    kept: list[int] = []
    annotations: dict[int, str] = {}
    pending: str | None = None
    start = 0
    while start <= len(text):
        stop = text.find("\n", start)
        if stop < 0:
            stop = len(text)
        line = text[start:stop]
        stripped = line.strip()
        if stripped.startswith(COMMENT):
            pending = stripped[len(COMMENT) :].strip()
        else:
            end = start + _comment_start(line)
            if stripped:
                if pending is not None:
                    annotations[len(kept)] = pending
                pending = None
            out.append(text[start:end])
            kept.extend(offsets[start:end])
        if stop < len(text):
            out.append("\n")
            kept.append(offsets[stop])
        start = stop + 1
    return "".join(out), kept, annotations


def _comment_start(line: str) -> int:
    """Where the comment of an expression line starts, or its length if it has none.

    A `;` inside a string is data, and one inside square brackets separates the rows of
    a matrix.
    """
    in_string = False
    depth = 0
    for index, char in enumerate(line):
        if char == '"':
            in_string = not in_string
        elif in_string:
            continue
        elif char == "[":
            depth += 1
        elif char == "]":
            depth = max(0, depth - 1)
        elif char == COMMENT and not depth:
            return index
    return len(line)


def _splice_continuations(text: str, offsets: list[int]) -> tuple[str, list[int]]:
    """Join a line ending in `~` to the next one.

    This deliberately operates on raw characters before lexing, so a splice may join
    the halves of an identifier, string literal, or operator expression.
    """
    out: list[str] = []
    kept: list[int] = []
    index = 0
    while index < len(text):
        if text.startswith("~\n", index):
            index += 2
            continue
        out.append(text[index])
        kept.append(offsets[index])
        index += 1
    return "".join(out), kept
