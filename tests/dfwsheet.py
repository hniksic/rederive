"""Derive for Windows worksheets built byte by byte, for the tests that read them.

`Sheet` follows the layout described in `rederive.model.dfw`.
"""

import struct
from pathlib import Path

MODERN = b"Derive version 6.10 DfW file saved on 03/04/2005 12:00:00\r\n\x1a"
OLD = b"DERIVE for Windows version 5.06 DfW file saved on 03/04/2001 12:00:00\r\n\x1a"


def escaped(text: str) -> bytes:
    """Text as version 6 stores it: code page 1252, and an escape, DEL and four hex
    digits, for a character beyond it."""
    out = b""
    for character in text:
        try:
            out += character.encode("cp1252")
        except UnicodeEncodeError:
            out += b"\x7f" + f"{ord(character):04x}".encode()
    return out


#: The codes of version 5's font for the characters the tests use.
_DERIVE5 = {
    "√": 0x8B,
    "≥": 0x92,
    "≤": 0x93,
    "∞": 0x96,
    "↓": 0x99,
    "∏": 0xA3,
    "∑": 0xA4,
    "ε": 0xAD,
    "·": 0xB7,
    "π": 0xB9,
    "τ": 0xBC,
    "λ": 0xBF,
    "σ": 0xDE,
    "ê": 0xEA,
    "ö": 0xF6,
}


def derive5(text: str) -> bytes:
    """Text as version 5 stores it, in its own font's codes."""
    return bytes(_DERIVE5.get(character, ord(character)) for character in text)


def rtf(*paragraphs: str) -> str:
    """An RTF document of `paragraphs`, as a text object holds it."""
    return _RTF_HEAD + r"\par ".join(map(_rtf_text, paragraphs)) + "}"


_RTF_HEAD = r"{\rtf1\ansi\ansicpg1252\deff0{\fonttbl{\f0\fswiss Arial;}}\pard "


def _rtf_text(text: str) -> str:
    """Text in RTF, a character beyond ASCII as `\\uN?`, N a signed 16-bit number."""
    return "".join(
        character if ord(character) < 0x80 else rf"\u{_signed(ord(character))}?"
        for character in text
    )


def _signed(code: int) -> int:
    return code - 0x10000 if code > 0x7FFF else code


def string(text: str | bytes, wide: bool = False) -> bytes:
    """An archived string, as bytes or, if `wide`, as UTF-16. Text beyond code page
    1252 is escaped, as version 6 escapes it."""
    if wide:
        data = text.encode("utf-16-le")
        return b"\xff\xfe\xff" + _length(len(data) // 2) + data
    data = text if isinstance(text, bytes) else escaped(text)
    return _length(len(data)) + data


def _length(length: int) -> bytes:
    if length < 0xFF:
        return bytes([length])
    return b"\xff" + struct.pack("<H", length)


class Sheet:
    """A worksheet under construction: a state block, objects, and the trailer.

    Classes and objects take archive indexes as they would in the file, so an object of
    a class already written refers to it by index.
    """

    def __init__(self, banner: bytes = MODERN, state: list[str | bytes] = ()) -> None:
        self.banner = banner
        self.state = list(state)
        self.objects: list[bytes] = []
        self.classes: dict[str, int] = {}
        self.taken = 0
        self.numbers = 0

    def tag(self, name: str) -> bytes:
        """The tag that starts an object of class `name`, taking its indexes."""
        if name in self.classes:
            self.taken += 1
            return struct.pack("<H", 0x8000 | self.classes[name])
        self.taken += 1
        self.classes[name] = self.taken
        self.taken += 1
        return b"\xff\xff\x01\x00" + struct.pack("<H", len(name)) + name.encode()

    def expression(
        self, text: str | bytes, annotation: str = "User", wide: bool = False
    ) -> "Sheet":
        self.numbers += 1
        self.objects.append(
            self.tag("CExpnObj")
            + bytes(16)
            + b"\x00"
            + string(annotation, wide)
            + struct.pack("<d", -1.0)
            + struct.pack("<II", self.numbers, 0)
            + string(text, wide)
        )
        return self

    def text(self, rtf: str, wide: bool = False, kind: str = "CTextObj") -> "Sheet":
        self.objects.append(self.tag(kind) + bytes(16) + b"\x00" + string(rtf, wide))
        return self

    def embedded(self, size: int = 600) -> "Sheet":
        """An OLE object, whose compound file is `size` bytes."""
        compound = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + bytes(size - 8)
        item = self.tag("CDispItem") + bytes(18) + struct.pack("<I", size) + compound
        self.objects.append(self.tag("CDispOleObj") + bytes(16) + b"\x00" + item)
        return self

    def plot(self, snapshot: bool = True) -> "Sheet":
        """A 2D plot with a label, two items (one of a new class) and a snapshot."""
        head = self.tag("CPlotObj") + bytes(16) + b"\x00" + struct.pack("<H", 2)
        view = self.tag("C2DPlotView") + struct.pack("<H", 1)
        label = (
            self.tag("C2DText")
            + bytes(4)
            + bytes(16)
            + string("a label")
            + bytes(28 + 32)
            + struct.pack("<dd", 1.0, 2.0)
        )
        # The items: compiled code in a layout no reader decodes. The first introduces a
        # class, which takes an index, and each item takes one.
        items = (
            b"\xff\xff\x01\x00\x0d\x00CExplicitPlot"
            + bytes(range(1, 40))
            + struct.pack("<H", 0x8000 | (self.taken + 1))
            + bytes(range(50, 90))
        )
        self.taken += 1 + 2
        settings = bytes(range(100, 130))
        pixels = bytes(64)
        info = struct.pack("<I", 40) + bytes(36)
        bitmap = (
            b"BM"
            + struct.pack("<I", 14 + len(info) + len(pixels))
            + bytes(4)
            + struct.pack("<I", 14 + len(info))
            + info
            + pixels
        )
        if not snapshot:
            bitmap = bytes(len(bitmap))
        self.objects.append(
            head + view + label + struct.pack("<H", 2) + items + settings + bitmap
        )
        return self

    def bytes(self) -> bytes:
        lines = [
            line if isinstance(line, bytes) else escaped(line) for line in self.state
        ]
        return (
            self.banner
            + struct.pack("<I", len(lines))
            + b"".join(line + b"\r\n" for line in lines)
            + struct.pack("<H", len(self.objects))
            + b"".join(self.objects)
            + struct.pack("<I", self.numbers + 1)
            + bytes(28)
            + struct.pack("<I", 2)
            + bytes(4)
        )

    def write(self, path: Path) -> Path:
        path.write_bytes(self.bytes())
        return path
