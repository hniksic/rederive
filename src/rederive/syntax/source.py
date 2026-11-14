"""Preprocessing of source text, with offsets mapping back to the original.

The parser takes `str`; decoding is up to the caller.
"""

from __future__ import annotations

import re
import unicodedata
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
        """A .MTH/.DMO source, with comments stripped and `~` continuations joined.

        Comments are stripped before splicing, so a `~` inside a comment doesn't join
        lines: `; a comment ~` followed by `1+2` loads `1+2` on its own. Escapes are
        decoded after splicing, since Derive breaks a line wherever it is long, even
        inside an escape.
        """
        body, offsets = _identity(text)
        body, offsets = _cut_at_end_marker(body, offsets)
        body, offsets = _normalize_line_endings(body, offsets)
        body, offsets = _strip_comments(body, offsets)
        body, offsets = _splice_continuations(body, offsets)
        body, offsets = _decode_escapes(body, offsets)
        return cls(text=body, original=text, origin=origin, offsets=tuple(offsets))


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


def _strip_comments(text: str, offsets: list[int]) -> tuple[str, list[int]]:
    """Drop `;` to end of line. A `;` inside a string literal is data."""
    out: list[str] = []
    kept: list[int] = []
    in_string = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            in_string = not in_string
        elif char == "\n":
            in_string = False
        elif char == ";" and not in_string:
            index = text.find("\n", index)
            if index < 0:
                break
            continue
        out.append(char)
        kept.append(offsets[index])
        index += 1
    return "".join(out), kept


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
