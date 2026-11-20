"""Reading the worksheets Derive for Windows saves: DfW files, which are binary.

Versions 5 and 6 write the same layout:

* a one-line banner, `DERIVE for Windows version 5.06 DfW file saved on ...` or `Derive
  version 6.10 DfW file saved on ...`, ending in CR LF and the DOS end marker Ctrl-Z;
* the state block: a DWORD count, then that many CR LF terminated lines, each a Derive
  expression. They define every function and variable the session knew when it was
  saved (including those of utility files, which the worksheet never shows), declare
  domains, and in version 6 assign the settings;
* the worksheet objects as an MFC `CArchive` list: a WORD count, then each object;
* a trailer: a DWORD (the next expression number), 28 bytes, a DWORD count and that
  many WORDs.

All integers are little-endian. An archived object starts with a WORD tag: 0xFFFF
introduces a class (a WORD schema, a WORD name length and the name) whose object
follows, 0x8000 plus an index is an object of a class seen before, 0x7FFF says the tag
is in the following DWORD instead, and anything else refers back to an object already
read. Classes and objects share one index sequence, starting at 1. A string is a BYTE
length, or 0xFF and a WORD length, or 0xFFFF and a DWORD length; the prefix FF FE FF
marks a UTF-16 string, after which the length prefix repeats.

The objects are:

* `CExpnObj`, an expression: a rectangle, a BYTE, the annotation, a double, the
  expression's number, a DWORD and its text, in the notation of math files;
* `CTextObj` and `CRuleObj`, text: a rectangle, a BYTE and a string holding an RTF
  document;
* `CDispOleObj`, an embedded OLE object: a rectangle, a BYTE and a `CDispItem`, whose
  18 header bytes and DWORD size are followed by an OLE compound file of that size;
* `CPlotObj`, a plot: a rectangle, a BYTE, a WORD dimension and a `C2DPlotView` or
  `C3DPlotView`. The view holds its labels (`C2DText`, `C3DText`), a count of plot
  items, the items themselves and its settings, ending in a snapshot as a BMP image.

The plot items are compiled Derive code whose layout is unknown, so a plot is skipped
by finding the end of its snapshot: the first BMP whose stated size ends where an object
or the trailer starts. The items it passes over still take indexes, one per item and one
per class introduced among them. That is a heuristic, but a wrong guess cannot go
unnoticed: the references that follow would fail to resolve, or the archive would not
end exactly with a well-formed trailer, and either refuses the whole file.

Text is decoded here, so callers see Unicode. Version 6 writes Windows code page 1252,
and characters beyond it as `\\x7fXXXX` escapes, as in its math files. Version 5 writes
its own font's codes, of which only some are known; see `DERIVE_CHARACTERS`.
"""

from __future__ import annotations

import errno
import re
import struct
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from rederive.syntax.names import SETTINGS
from rederive.syntax.source import unescaped

#: The banner that identifies a worksheet, up to and including the end marker.
_BANNER = re.compile(
    rb"(?:DERIVE for Windows|Derive) version (\S+) DfW file saved on [^\r\n]*\r\n\x1a"
)

#: The longest banner looked for; real ones are well under this.
_BANNER_LENGTH = 200

#: Version 5's codes for characters outside ASCII, as far as they are known. They are
#: neither code page 437 nor ISO 8859-1: they were found by aligning the same definitions
#: as version 5 and version 6 worksheets save them, or a state line with the typed
#: expressions of the same worksheet that spell its letter by name (`rho`, `kappa`,
#: `PHI`); the NOT sign is ISO 8859-1's (`IF(¬ DIM(a) = n ∨ ¬ DIM(b) = n, ...)`). Where
#: version 6 writes a constant or function in the Private Use Area (the constants e, i
#: and pi, GAMMA, ZETA), so does this table. The German letters are as in ISO 8859-1.
#: `NAME_TO_CODES` and `CODES_TO_NAME` use these codes too, as Derive's own programs test
#: them; the bracket pieces at 0x80, 0x82 and 0x84 are what Derive 6's CODES_TO_NAME
#: gives for those codes.
DERIVE_CHARACTERS = {
    0x80: "⎛",
    0x82: "⎞",
    0x84: "⎡",
    0x8B: "√",
    0x8E: "∪",
    0x8F: "∧",
    0x90: "∨",
    0x92: "≥",
    0x93: "≤",
    0x94: "≠",
    0x95: "⋅",
    0x96: "∞",
    0x99: "↓",
    0x9B: "\ue7c4",
    0x9D: "Δ",
    0x9E: "Θ",
    0xA3: "∏",
    0xA4: "∑",
    0xA5: "Φ",
    0xA8: "Ω",
    0xA9: "α",
    0xAA: "γ",
    0xAB: "δ",
    0xAC: "¬",
    0xAD: "ε",
    0xAE: "\ue7c5",
    0xB0: "°",
    0xB1: "±",
    0xB2: "θ",
    0xB4: "κ",
    0xB5: "μ",
    0xB7: "·",
    0xB9: "\ue7c2",
    0xBA: "ρ",
    0xBC: "τ",
    0xBE: "φ",
    0xBF: "λ",
    0xDE: "σ",
    0xDF: "β",
    0xEA: "\ue7c0",
    0xEE: "\ue7c1",
    0xF7: "ω",
    0xFE: "ν",
} | {code: chr(code) for code in b"\xc4\xd6\xdc\xe4\xf6\xfc"}

#: The letters of ISO 8859-1 that version 5 writes as they are, and ASCII's.
_LATIN_LETTERS = frozenset(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\xc4\xd6\xdc\xe4\xf6\xfc"
)


def _version5_text(stored: bytes) -> str:
    """Version 5 text as Unicode (see `DERIVE_CHARACTERS`).

    The code 0xDF is beta in Derive's font and sharp s in ISO 8859-1, which a German
    user typing text gets: it is sharp s inside a string, and between two letters of a
    word (`"größter Wert"`), and beta otherwise (`β0_`).
    """
    characters = []
    quoted = False
    for place, code in enumerate(stored):
        if code == 0x22:
            quoted = not quoted
        if code == 0xDF and (
            quoted
            or (
                place
                and stored[place - 1] in _LATIN_LETTERS
                and place + 1 < len(stored)
                and stored[place + 1] in _LATIN_LETTERS
            )
        ):
            characters.append("ß")
            continue
        characters.append(DERIVE_CHARACTERS.get(code, chr(code)))
    return "".join(characters)


#: Annotations of expressions the user typed, in the languages Derive was localized to.
#: Derive 6 annotates an expression simplified with a trailing `=` as `User=Simp(User)`,
#: and that is typed too.
_TYPED = frozenset("User Benutzer Utilisateur Utente Nueva Användare".split())

#: The classes of the objects a worksheet lists.
_TOP_CLASSES = ("CExpnObj", "CTextObj", "CRuleObj", "CDispOleObj", "CPlotObj")

#: A line of the state block that assigns a setting, if the name is one.
_SETTING = re.compile(r"\s*([A-Za-z]+)\s*:=(.*)", re.DOTALL)

#: Every class a plot's items were seen to use.
_PLOT_CLASSES = frozenset(
    """
    C2DPlotView C3DPlotView C2DText C3DText CExplicitPlot CPointListPlot
    C3DParametricPlot C3DParaPointPlot CImplicitPlot CParametricPlot C3DExplicitPlot
    C3DDataPointPlot CRelationPlot CXofYPlot
    """.split()
)

#: A class introduced among a plot's items: the new-class tag, a small schema, the name's
#: length and a name of that length.
_NEW_CLASS = re.compile(rb"\xff\xff[\x00-\x0f]\x00([\x01-\x40])\x00(C[A-Za-z0-9]+)")

#: A BMP file header followed by the size of a known info header.
_BITMAP = re.compile(rb"BM(.{4})\x00{4}.{4}[\x28\x6c\x7c]\x00{3}", re.DOTALL)

#: The bytes of the trailer before its list of WORDs.
_TRAILER = 36

#: The signature of an OLE compound file.
_COMPOUND = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

#: Bytes of a `LOGFONT` before its face name, and the face name's characters.
_LOGFONT = 28
_FACE = 32


class Damaged(OSError):
    """A worksheet that can't be read: truncated, corrupt, or of an unknown kind.

    An `OSError`, so it is reported like any other file that can't be read.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(
            errno.EINVAL, f"not a readable Derive for Windows worksheet: {reason}"
        )


@dataclass(frozen=True)
class Expression:
    """An expression object: its number, annotation and text as typed."""

    number: int
    annotation: str
    text: str

    @property
    def typed(self) -> bool:
        """Whether the user typed the expression rather than a command producing it."""
        return typed(self.annotation)


@dataclass(frozen=True)
class Text:
    """A text object, as its paragraphs of plain text."""

    paragraphs: tuple[str, ...]


@dataclass(frozen=True)
class Worksheet:
    """What a worksheet holds that Rederive can use.

    `state` has a line of the state block wherever it was decoded with certainty and
    None where it holds a version 5 code whose meaning is unknown. `objects` are the
    expressions and texts in order, without the plots and embedded objects, of which
    `plots` and `embedded` count how many there were.
    """

    state: tuple[str | None, ...]
    objects: tuple[Expression | Text, ...]
    plots: int = 0
    embedded: int = 0


def typed(annotation: str) -> bool:
    """Whether `annotation` says the user typed the expression, in any of Derive's
    languages: `User`, `Benutzer`, ..., alone or as `User=Simp(User)`, which a typed
    `expr=` gets. A digit glued to the word, as in `Simp(User02)`, is what Derive made of
    a reference to an expression it deleted, and is no such annotation."""
    name, _, rest = annotation.strip().partition("=")
    name = name.strip()
    return name in _TYPED and rest.strip() in ("", f"Simp({name})")


def setting_of(line: str) -> tuple[str, str] | None:
    """The setting a state line assigns and its value, or None if it assigns none.

    Derive 6 saves its settings at the end of the state block, as `Name:=Value`.
    """
    found = _SETTING.fullmatch(line)
    if found is None or found.group(1) not in SETTINGS:
        return None
    return found.group(1), found.group(2).strip()


def is_worksheet(raw: bytes) -> bool:
    """Whether `raw` starts with a DfW banner, and so is read here rather than as text."""
    return _BANNER.match(raw[:_BANNER_LENGTH]) is not None


def read(raw: bytes) -> Worksheet:
    """The worksheet `raw` holds. Raises `Damaged` unless all of it reads."""
    banner = _BANNER.match(raw[:_BANNER_LENGTH])
    if banner is None:
        raise Damaged("no banner")
    reader = _Archive(raw, banner.end(), modern=not banner.group(1).startswith(b"5"))
    state = tuple(reader.state_line() for _ in range(reader.count(reader.dword())))
    objects = []
    for _ in range(reader.count(reader.list_length())):
        found = reader.top_object()
        if found is not None:
            objects.append(found)
    reader.trailer()
    return Worksheet(state, tuple(objects), reader.plots, reader.embedded)


class _Archive:
    """A cursor over the file, reading archived values and objects.

    Every read is checked against the data left, so a truncated or corrupt file raises
    `Damaged` rather than reading past its end or taking a garbage count at face value.
    """

    def __init__(self, data: bytes, at: int, modern: bool) -> None:
        self.data = data
        self.at = at
        #: Whether this is version 6, which decides how text is encoded.
        self.modern = modern
        #: The class names by index. Objects take indexes too, so this is sparse.
        self.classes: dict[int, str] = {}
        #: The last index taken.
        self.taken = 0
        #: How many plots and embedded objects were skipped.
        self.plots = 0
        self.embedded = 0

    # -- values -------------------------------------------------------------

    def _take(self, size: int) -> int:
        """Advance over `size` bytes and return where they start."""
        start = self.at
        if size < 0 or size > len(self.data) - start:
            raise Damaged(f"truncated at offset {start}")
        self.at += size
        return start

    def _unpack(self, form: str) -> int | float:
        return struct.unpack_from(form, self.data, self._take(struct.calcsize(form)))[0]

    def byte(self) -> int:
        return int(self._unpack("<B"))

    def word(self) -> int:
        return int(self._unpack("<H"))

    def dword(self) -> int:
        return int(self._unpack("<I"))

    def double(self) -> float:
        return float(self._unpack("<d"))

    def raw(self, size: int) -> bytes:
        start = self._take(size)
        return self.data[start : self.at]

    def skip(self, size: int) -> None:
        self._take(size)

    def count(self, count: int) -> int:
        """`count`, checked against the bytes left, as each counted thing takes one."""
        if count > len(self.data) - self.at:
            raise Damaged(f"impossible count {count} at offset {self.at}")
        return count

    def list_length(self) -> int:
        """An object list's length: a WORD, or 0xFFFF and a DWORD."""
        length = self.word()
        return self.dword() if length == 0xFFFF else length

    def string(self) -> bytes | str:
        """A string: bytes as stored, or text if it was stored as UTF-16."""
        length = self.byte()
        wide = False
        if length == 0xFF:
            length = self.word()
            if length == 0xFFFE:
                wide = True
                length = self.byte()
                if length == 0xFF:
                    length = self.word()
                    if length == 0xFFFF:
                        length = self.dword()
            elif length == 0xFFFF:
                length = self.dword()
        if wide:
            return self.raw(2 * length).decode("utf-16-le", "replace")
        return self.raw(length)

    def text(self, stored: bytes | str) -> str:
        """A string in the file as Unicode, expanding the version 6 escapes.

        A version 5 code whose meaning is unknown is taken as ISO 8859-1. In an
        expression shown, that is at worst a wrong letter in sight of the user; for the
        state block, see `state_line`.
        """
        if isinstance(stored, str):
            return unescaped(stored)
        if self.modern:
            return unescaped(stored.decode("cp1252", "replace"))
        return unescaped(_version5_text(stored))

    def state_line(self) -> str | None:
        """The next line of the state block, or None if it can't be decoded for certain.

        The state block defines names, and a name with a guessed character is a different
        name, so a line with a code that isn't known is not read at all.
        """
        end = self.data.find(b"\r\n", self.at)
        if end < 0:
            raise Damaged(f"unterminated state line at offset {self.at}")
        line = self.raw(end - self.at)
        self.skip(2)
        if self.modern:
            try:
                decoded = line.decode("cp1252")
            except UnicodeDecodeError:
                return None
        else:
            if any(code > 0x7F and code not in DERIVE_CHARACTERS for code in line):
                return None
            decoded = self.text(line)
        decoded = unescaped(decoded)
        return None if "\x7f" in decoded else decoded

    def rectangle(self) -> None:
        self.skip(16)

    # -- objects ------------------------------------------------------------

    def top_object(self) -> Expression | Text | None:
        """The next object of the worksheet's list, or None for one that is skipped."""
        return self._object(_TOP_CLASSES)

    def _object(self, expected: tuple[str, ...]) -> Expression | Text | None:
        """Read an object whose class must be one of `expected`."""
        start = self.at
        tag = self.word()
        if tag == 0xFFFF:
            self.word()
            name = self.raw(self.word()).decode("ascii", "replace")
            self.taken += 1
            self.classes[self.taken] = name
        else:
            if tag == 0x7FFF:
                tag = self.dword()
                new_object, index = tag & 0x80000000, tag & 0x7FFFFFFF
            else:
                new_object, index = tag & 0x8000, tag & 0x7FFF
            if not new_object:
                raise Damaged(f"unexpected reference at offset {start}")
            if index not in self.classes:
                raise Damaged(f"unknown class {index} at offset {start}")
            name = self.classes[index]
        if name not in expected:
            raise Damaged(f"unexpected {name} at offset {start}")
        self.taken += 1
        return self._readers[name](self)

    def _expression(self) -> Expression:
        self.rectangle()
        self.byte()
        annotation = self.text(self.string())
        self.double()
        number = self.dword()
        self.dword()
        return Expression(number, annotation, self.text(self.string()))

    def _text(self) -> Text:
        self.rectangle()
        self.byte()
        stored = self.string()
        if isinstance(stored, bytes):
            stored = stored.decode("cp1252", "replace")
        return Text(rtf_paragraphs(stored))

    def _embedded(self) -> None:
        self.rectangle()
        self.byte()
        self._object(("CDispItem",))
        self.embedded += 1

    def _embedded_item(self) -> None:
        self.skip(18)
        size = self.dword()
        start = self._take(size)
        if not self.data.startswith(_COMPOUND, start):
            raise Damaged(f"embedded object at offset {start} is not an OLE file")

    def _plot(self) -> None:
        self.rectangle()
        self.byte()
        self.word()
        self._object(("C2DPlotView", "C3DPlotView"))
        self.plots += 1

    def _plot_view(self) -> None:
        """A plot view: labels, then items and settings, skipped as the module says."""
        for _ in range(self.count(self.word())):
            self._object(("C2DText", "C3DText"))
        items = self.word()
        body = self.at
        for found in _BITMAP.finditer(self.data, body):
            end = found.start() + int.from_bytes(found.group(1), "little")
            if found.end() <= end <= len(self.data) and self._resumes(end):
                break
        else:
            raise Damaged(f"no end found for the plot at offset {body}")
        for introduced in _NEW_CLASS.finditer(self.data, body, found.start()):
            name = introduced.group(2)[: introduced.group(1)[0]].decode("ascii")
            if name in _PLOT_CLASSES:
                self.taken += 1
                self.classes[self.taken] = name
        self.taken += items
        self.at = end

    def _resumes(self, at: int) -> bool:
        """Whether an object or the trailer could start at `at`."""
        left = len(self.data) - at
        if _TRAILER <= left and (left - _TRAILER) % 2 == 0:
            words = struct.unpack_from("<I", self.data, at + _TRAILER - 4)[0]
            if 2 * words == left - _TRAILER:
                return True
        if left < 2:
            return False
        tag = struct.unpack_from("<H", self.data, at)[0]
        if tag == 0xFFFF:
            return _NEW_CLASS.match(self.data, at) is not None
        return bool(tag & 0x8000)

    def _plot_label(self) -> None:
        """A plot label: a DWORD, a rectangle, the text, a font and where it is."""
        self.dword()
        self.rectangle()
        wide = isinstance(self.string(), str)
        self.skip(_LOGFONT + _FACE * (2 if wide else 1))
        self.skip(16)

    def _plot_label_3d(self) -> None:
        """The same in 3D, placed by three integers rather than two doubles."""
        self.dword()
        self.rectangle()
        wide = isinstance(self.string(), str)
        self.skip(_LOGFONT + _FACE * (2 if wide else 1))
        self.skip(12)

    #: How to read each class, which `_object` knows by name.
    _readers: ClassVar[dict[str, Callable[[_Archive], Expression | Text | None]]] = {
        "CExpnObj": _expression,
        "CTextObj": _text,
        "CRuleObj": _text,
        "CDispOleObj": _embedded,
        "CDispItem": _embedded_item,
        "CPlotObj": _plot,
        "C2DPlotView": _plot_view,
        "C3DPlotView": _plot_view,
        "C2DText": _plot_label,
        "C3DText": _plot_label_3d,
    }

    def trailer(self) -> None:
        """Read the trailer, which must end the file."""
        self.dword()
        self.skip(_TRAILER - 8)
        self.skip(2 * self.count(self.dword()))
        if self.at != len(self.data):
            raise Damaged(f"{len(self.data) - self.at} bytes after the trailer")


# -- RTF ----------------------------------------------------------------------------

#: An RTF token: a control word with its optional numeric argument and the one space
#: that ends it, a hex-escaped byte, a control symbol, a brace, or a run of plain text.
_RTF_TOKEN = re.compile(
    r"\\([a-zA-Z]+)(-?\d+)? ?|\\'([0-9a-fA-F]{2})|\\(.)|([{}])|([^\\{}\r\n]+)|[\r\n]+",
    re.DOTALL,
)

#: Groups whose content is not text.
_RTF_HIDDEN = frozenset(
    """
    fonttbl colortbl stylesheet info pict object header footer headerl headerr
    footerl footerr listtable listoverridetable rsidtbl generator xmlnstbl themedata
    latentstyles datastore filetbl revtbl
    """.split()
)

#: Control words that stand for characters.
_RTF_CHARACTERS = {
    "par": "\n",
    "line": "\n",
    "sect": "\n",
    "page": "\n",
    "tab": " ",
    "emdash": "\u2014",
    "endash": "\u2013",
    "bullet": "\u2022",
    "lquote": "\u2018",
    "rquote": "\u2019",
    "ldblquote": "\u201c",
    "rdblquote": "\u201d",
}

#: Control symbols that stand for characters; others, such as the optional hyphen `\-`,
#: stand for none. A backslash before a line break is a paragraph mark.
_RTF_SYMBOLS = {
    "\\": "\\",
    "{": "{",
    "}": "}",
    "~": " ",
    "_": "-",
    "\n": "\n",
    "\r": "\n",
}


def rtf_paragraphs(document: str) -> tuple[str, ...]:
    """The text of an RTF document as its non-empty paragraphs, each on one line.

    This is as much of RTF as a word processor's text needs: groups that hold no text
    (the font table, pictures) are left out, `\\'hh` is a byte in the document's code
    page, and `\\uN` is a character followed by `\\ucN` fallback characters to skip.
    Formatting is dropped.
    """
    codepage = "cp1252"
    output: list[str] = []
    #: For each open group: whether it is hidden, and the fallback count in it.
    groups: list[tuple[bool, int]] = [(False, 1)]
    hidden, fallback = groups[-1]
    skipping = 0
    at_group_start = False
    for token in _RTF_TOKEN.finditer(document):
        word, argument, hexadecimal, symbol, brace, plain = token.groups()
        if brace == "{":
            groups.append((hidden, fallback))
            at_group_start = True
            continue
        if brace == "}":
            if len(groups) > 1:
                groups.pop()
            hidden, fallback = groups[-1]
            at_group_start = False
            continue
        starting, at_group_start = at_group_start, False
        if word is not None or symbol == "*":
            if starting and (symbol == "*" or word in _RTF_HIDDEN):
                hidden = True
                groups[-1] = (hidden, fallback)
                continue
            if word in _RTF_HIDDEN:
                hidden = True
                groups[-1] = (hidden, fallback)
            elif word == "ansicpg" and argument is not None:
                codepage = _codepage(int(argument), codepage)
            elif word == "uc" and argument is not None:
                fallback = max(0, int(argument))
                groups[-1] = (hidden, fallback)
            elif word == "u" and argument is not None and not hidden:
                output.append(chr(int(argument) % 0x10000))
                skipping = fallback
            elif word in _RTF_CHARACTERS and not hidden:
                output.append(_RTF_CHARACTERS[word])
                skipping = 0
            continue
        if hidden:
            continue
        if hexadecimal is not None:
            if skipping:
                skipping -= 1
            else:
                output.append(bytes([int(hexadecimal, 16)]).decode(codepage, "replace"))
        elif symbol is not None:
            output.append(_RTF_SYMBOLS.get(symbol, ""))
        elif plain is not None:
            if skipping:
                dropped = min(skipping, len(plain))
                plain = plain[dropped:]
                skipping -= dropped
            output.append(plain)
    lines = "".join(output).split("\n")
    return tuple(" ".join(line.split()) for line in lines if line.strip())


def _codepage(number: int, current: str) -> str:
    """The Python codec for Windows code page `number`, or `current` if there is none."""
    name = f"cp{number}"
    try:
        "".encode(name)
    except LookupError:
        return current
    return name
