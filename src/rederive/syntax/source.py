"""Textual preprocessing, and the map back to what the caller handed over.

The parser takes `str`: decoding is the caller's business, and no encoding
appears in this API.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    """A piece of Derive text, prepared for lexing.

    `text` is what the lexer and parser index; `original` is what the caller
    handed over. The preprocessing transforms only ever delete characters, so
    offsets into `text` no longer index `original`, and `locate` is what puts
    the cursor in the right place.
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
        """A .MTH/.DMO source: comments and `~` splices apply.

        The transforms run in this order, and comments are stripped before
        splicing, not after: a first line of `; a comment ~` followed by `1+2`
        loads both, because the `~` inside the comment is gone before splicing
        ever looks for one.
        """
        body, offsets = _identity(text)
        body, offsets = _cut_at_end_marker(body, offsets)
        body, offsets = _normalize_line_endings(body, offsets)
        body, offsets = _strip_comments(body, offsets)
        body, offsets = _splice_continuations(body, offsets)
        return cls(text=body, original=text, origin=origin, offsets=tuple(offsets))


#: What ends a DOS text file. Every file the original wrote carries one, and
#: nothing after it is text.
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
    """Accept CRLF, LF and CR; normalise to LF."""
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

    A raw character operation on the whole text, deliberately: the splice
    happens before lexing and may cut an identifier, a string literal or an
    operand from its operator in half.
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
