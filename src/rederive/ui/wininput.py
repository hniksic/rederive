"""Windows console input records, translated into the text a terminal would send.

With VT input off, a Windows console program reads its input as records: a key record
names the key, the character it types and the modifiers held, and a mouse record gives
the pointer's cell and buttons. Unlike the text ConPTY produces with VT input on, these
keep Shift on Enter. `Translator` turns them into the sequences Textual's parser reads
from a terminal, so the rest of Textual is unchanged. The forms are those of a terminal
with the kitty keyboard protocol's first level, which is what Textual asks for
elsewhere: legacy sequences for the cursor and function keys, `CSI code;mods u` for
keys that have no legacy form with modifiers.

Records are plain values here, so the translation is the same, and tested, everywhere.

A paste reaches the program as a burst of key records, with no bracketed-paste markers.
`Translator.feed` takes the records read together, and treats them as a paste when they
type both text and at least one line break, as prompt_toolkit does: a person doesn't
type a character and Enter within one read. The typed runs are then sent as one
bracketed paste, so a pasted line break starts a row instead of entering the line.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["FocusRecord", "KeyRecord", "MouseRecord", "Record", "Translator"]

# `dwControlKeyState` bits.
RIGHT_ALT = 0x0001
LEFT_ALT = 0x0002
RIGHT_CTRL = 0x0004
LEFT_CTRL = 0x0008
SHIFT = 0x0010

# `dwButtonState` bits, and the wheel's delta in its high word.
LEFT_BUTTON = 0x0001
RIGHT_BUTTON = 0x0002
MIDDLE_BUTTON = 0x0004

#: The buttons, each with its number in SGR mouse reports.
BUTTONS = ((LEFT_BUTTON, 0), (MIDDLE_BUTTON, 1), (RIGHT_BUTTON, 2))

# `dwEventFlags` values.
MOUSE_MOVED = 0x0001
MOUSE_WHEELED = 0x0004
MOUSE_HWHEELED = 0x0008

# Virtual-key codes.
VK_BACK = 0x08
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_MENU = 0x12
VK_ESCAPE = 0x1B

#: Keys that only modify others, or lock: Shift, Ctrl, Alt (each also as left and
#: right), the Windows keys, Caps Lock, Num Lock and Scroll Lock. Pressed alone they
#: type nothing.
MODIFIER_KEYS = frozenset({0x10, 0x11, 0x12, 0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5})
MODIFIER_KEYS |= {0x5B, 0x5C, 0x14, 0x90, 0x91}

#: Keys sent as `CSI final` alone and `CSI 1;mods final` with modifiers.
CURSOR_KEYS = {
    0x26: "A",  # up
    0x28: "B",  # down
    0x27: "C",  # right
    0x25: "D",  # left
    0x24: "H",  # home
    0x23: "F",  # end
}

#: F1 to F4, sent as `SS3 final` alone and `CSI 1;mods final` with modifiers.
SS3_KEYS = {0x70: "P", 0x71: "Q", 0x72: "R", 0x73: "S"}

#: Keys sent as `CSI number ~`, with `;mods` added for modifiers.
TILDE_KEYS = {
    0x2D: 2,  # insert
    0x2E: 3,  # delete
    0x21: 5,  # page up
    0x22: 6,  # page down
    0x74: 15,  # F5
    0x75: 17,
    0x76: 18,
    0x77: 19,
    0x78: 20,
    0x79: 21,
    0x7A: 23,
    0x7B: 24,  # F12
}

CSI = "\x1b["
PASTE_START = "\x1b[200~"
PASTE_END = "\x1b[201~"


@dataclass(frozen=True)
class KeyRecord:
    """A `KEY_EVENT_RECORD`: `char` is `uChar.UnicodeChar`, `state` the modifiers."""

    down: bool
    vk: int
    char: str = ""
    state: int = 0
    repeat: int = 1


@dataclass(frozen=True)
class MouseRecord:
    """A `MOUSE_EVENT_RECORD`, with the console's 0-based cell."""

    x: int
    y: int
    buttons: int = 0
    state: int = 0
    flags: int = 0


@dataclass(frozen=True)
class FocusRecord:
    """A `FOCUS_EVENT_RECORD`, which the console documents as internal: ignored."""

    focused: bool


Record = KeyRecord | MouseRecord | FocusRecord


def _modifiers(state: int) -> tuple[bool, bool, bool]:
    """Shift, Alt and Ctrl, from a record's `dwControlKeyState`."""
    return (
        bool(state & SHIFT),
        bool(state & (LEFT_ALT | RIGHT_ALT)),
        bool(state & (LEFT_CTRL | RIGHT_CTRL)),
    )


def _xterm_modifier(shift: bool, alt: bool, ctrl: bool) -> int:
    """The modifier parameter of xterm and kitty sequences: 1 plus the held bits."""
    return 1 + shift + 2 * alt + 4 * ctrl


def _character(record: KeyRecord) -> str:
    """The character the key typed, or "" when it typed none."""
    return "" if record.char == "\0" else record.char


def _is_text(record: KeyRecord) -> bool:
    """Whether the key typed a printable character as text.

    Ctrl and Alt together with a printable character are AltGr, which is how keyboard
    layouts type characters such as `@`, `{` and `€`, so they type text. Alt alone
    makes the key a shortcut instead.
    """
    character = _character(record)
    if not character or not (character.isprintable() or _is_surrogate(character)):
        return False
    _, alt, ctrl = _modifiers(record.state)
    return ctrl or not alt


def _is_surrogate(character: str) -> bool:
    return "\ud800" <= character <= "\udfff"


def _is_line_break(record: KeyRecord) -> bool:
    """Whether the key is Enter, or Ctrl-Enter (which types LF), without Alt."""
    _, alt, _ = _modifiers(record.state)
    return _character(record) in ("\r", "\n") and not alt


def translate_key(record: KeyRecord) -> str:
    """The terminal input for one key record, repeats included."""
    if not record.down:
        # Alt with a code typed on the numeric keypad delivers the character on the
        # release of Alt.
        if record.vk == VK_MENU and _character(record):
            return _character(record)
        return ""
    return _one_key(record) * max(record.repeat, 1)


def _one_key(record: KeyRecord) -> str:
    vk = record.vk
    if vk in MODIFIER_KEYS:
        return ""
    shift, alt, ctrl = _modifiers(record.state)
    modifier = _xterm_modifier(shift, alt, ctrl)
    character = _character(record)
    if vk == VK_RETURN:
        if modifier == 1:
            return "\r"
        # Textual reads LF as Ctrl-J, which is how terminals without the keyboard
        # protocol send Ctrl-Enter.
        if modifier == 5:
            return "\n"
        return f"{CSI}13;{modifier}u"
    if vk == VK_TAB:
        if modifier == 1:
            return "\t"
        if modifier == 2:
            return f"{CSI}Z"
        return f"{CSI}9;{modifier}u"
    if vk == VK_BACK:
        return "\x7f" if modifier == 1 else f"{CSI}127;{modifier}u"
    # Esc is sent as the protocol reports it, since a bare ESC is also how sequences
    # start: Textual would wait to see whether more follows. Ctrl-[ types ESC too.
    if character == "\x1b" and vk != VK_ESCAPE:
        return f"{CSI}27u"
    if vk == VK_ESCAPE:
        return f"{CSI}27u" if modifier == 1 else f"{CSI}27;{modifier}u"
    if final := CURSOR_KEYS.get(vk):
        return f"{CSI}{final}" if modifier == 1 else f"{CSI}1;{modifier}{final}"
    if final := SS3_KEYS.get(vk):
        return f"\x1bO{final}" if modifier == 1 else f"{CSI}1;{modifier}{final}"
    if number := TILDE_KEYS.get(vk):
        return f"{CSI}{number}~" if modifier == 1 else f"{CSI}{number};{modifier}~"
    if _is_text(record):
        return character
    if character and (alt or ctrl) and character.isprintable():
        # Alt with a character, as the protocol reports it. A bare ESC before the
        # character would be read only after a pause, and ESC with `[` or `O` starts a
        # sequence. Shift is in the character, except for a letter, which the protocol
        # reports in lower case with Shift.
        if shift and character.isalpha() and character.isupper():
            return f"{CSI}{ord(character.lower())};{_xterm_modifier(True, alt, ctrl)}u"
        return f"{CSI}{ord(character)};{_xterm_modifier(False, alt, ctrl)}u"
    if character:
        # A control character, which is what Ctrl with a letter types.
        return character
    if (alt or ctrl) and (0x30 <= vk <= 0x39 or 0x41 <= vk <= 0x5A):
        # Ctrl with a digit, or Ctrl-Alt with a letter, types no character.
        return f"{CSI}{ord(chr(vk).lower())};{modifier}u"
    if ctrl and vk == 0x20:
        return "\0"
    return ""


def _pasted(record: Record) -> str | None:
    """What a record adds to a paste, or None if it isn't part of one.

    Releases and modifier keys add nothing but don't interrupt the paste, since a
    pasted capital comes with Shift pressed and released around it.
    """
    if not isinstance(record, KeyRecord):
        return None
    if not record.down or record.vk in MODIFIER_KEYS:
        return ""
    repeat = max(record.repeat, 1)
    if _is_line_break(record):
        return "\n" * repeat
    if _character(record) == "\t" or _is_text(record):
        return _character(record) * repeat
    return None


def is_paste(records: Sequence[Record]) -> bool:
    """Whether records read together are a paste: text and at least one line break."""
    keys = [r for r in records if isinstance(r, KeyRecord) and r.down]
    return any(map(_is_line_break, keys)) and any(map(_is_text, keys))


class Translator:
    """Turns console input records into terminal input, keeping state between reads.

    The state is the mouse buttons held, since a mouse record reports which are down
    rather than which changed, and a high surrogate whose low half is in the next read.
    """

    def __init__(self) -> None:
        self._buttons = 0
        self._pending = ""

    def feed(self, records: Sequence[Record]) -> str:
        """The terminal input for records read together."""
        pieces: list[str] = []
        paste: list[str] | None = None
        pasting = is_paste(records)
        for record in records:
            text = _pasted(record) if pasting else None
            if text is not None and (text or paste is not None):
                if paste is None:
                    paste = []
                paste.append(text)
                continue
            if paste is not None:
                pieces.append(PASTE_START + "".join(paste) + PASTE_END)
                paste = None
            if isinstance(record, KeyRecord):
                pieces.append(translate_key(record))
            elif isinstance(record, MouseRecord):
                pieces.append(self._mouse(record))
        if paste is not None:
            pieces.append(PASTE_START + "".join(paste) + PASTE_END)
        text = self._pending + "".join(pieces)
        self._pending = ""
        if text and "\ud800" <= text[-1] <= "\udbff":
            self._pending, text = text[-1], text[:-1]
        # Characters outside the BMP come as two records, one per UTF-16 half.
        return text.encode("utf-16", "surrogatepass").decode("utf-16", "surrogatepass")

    def _mouse(self, record: MouseRecord) -> str:
        """SGR mouse reports for a mouse record: `CSI < button;x;y M` or `m`."""
        shift, alt, ctrl = _modifiers(record.state)
        bits = 4 * shift + 8 * alt + 16 * ctrl
        x, y = record.x + 1, record.y + 1
        if record.flags & (MOUSE_WHEELED | MOUSE_HWHEELED):
            delta = (record.buttons >> 16) & 0xFFFF
            positive = 0 < delta < 0x8000
            if record.flags & MOUSE_WHEELED:
                # Away from the user is positive, and scrolls up.
                button = 64 if positive else 65
            else:
                button = 67 if positive else 66
            return f"{CSI}<{button | bits};{x};{y}M"
        held = record.buttons & (LEFT_BUTTON | MIDDLE_BUTTON | RIGHT_BUTTON)
        changed = held ^ self._buttons
        self._buttons = held
        reports = []
        for mask, button in BUTTONS:
            if changed & mask:
                final = "M" if held & mask else "m"
                reports.append(f"{CSI}<{button | bits};{x};{y}{final}")
        if not reports and record.flags & MOUSE_MOVED:
            # Motion reports the first button held, or 3 for none.
            button = next((b for mask, b in BUTTONS if held & mask), 3)
            reports.append(f"{CSI}<{32 | button | bits};{x};{y}M")
        return "".join(reports)
