"""Terminal input the drivers read before Textual's parser sees it.

Whether Shift-Enter can be told from Enter depends on the terminal: only one with the
kitty keyboard protocol reports it, as `CSI 13;2u`. Textual turns the protocol on
without checking that the terminal took it, so the driver asks: `QUERY` requests the
protocol's flags, which only a terminal with the protocol answers, followed by the
primary device attributes, which every terminal answers. A flags reply means Shift-Enter
arrives as itself. Neither reply is input, and Textual's parser would turn them into
stray keys, so they are taken out here.

Alt-Enter is the fallback for a new row, but without the protocol it arrives as ESC CR,
which Textual reads as a plain Enter. It is rewritten to the protocol's report of
Alt-Enter, which Textual reads as `alt+enter`.

With the protocol, a key's report may carry the text it types. Textual takes text on an
Alt key for a character Alt composed, as Option does on a Mac, and drops the Alt. But
xterm.js reports text on Alt-E that is just the key's own letter, so the program would
see `e` and not `alt+e`. Where the text is the key's letter in either case, Alt
composed nothing, and the text is taken out of the report.
"""

from __future__ import annotations

import re

__all__ = ["PROTOCOL_OFF", "PROTOCOL_ON", "QUERY", "screened"]

#: Turn the kitty keyboard protocol on with the flags Textual's terminal driver uses
#: (disambiguate, report all keys, report text), and back off. For a driver that
#: Textual doesn't already do this for.
PROTOCOL_ON = "\x1b[>25u"
PROTOCOL_OFF = "\x1b[<u"

#: The kitty keyboard flags query, then the primary device attributes query.
QUERY = "\x1b[?u\x1b[c"

#: A terminal's answer to either query: `CSI ? flags u` or `CSI ? attributes c`. Keys
#: never start with `CSI ?`, so these can't be mistaken for typing.
_REPLY = re.compile(r"\x1b\[\?[\d;]*([uc])")

#: Alt-Enter from a terminal without the protocol, and as the protocol reports it.
_ALT_ENTER = "\x1b\r"
_ALT_ENTER_REPORTED = "\x1b[13;3u"

#: A key reported with modifiers and text: `CSI key[:alternates] ; mods[:event] ; text u`.
_KEY_WITH_TEXT = re.compile(r"\x1b\[(\d+)((?::\d*)*);(\d+)((?::\d+)?);([\d:]+)u")
#: The Alt bit of the protocol's modifier parameter, which is 1 plus the held bits.
_ALT = 0b10


def screened(text: str) -> tuple[str, bool]:
    """`text` without the replies to `QUERY`, and whether it held the flags reply."""
    flags = False

    def drop(reply: re.Match[str]) -> str:
        nonlocal flags
        flags = flags or reply.group(1) == "u"
        return ""

    text = _REPLY.sub(drop, text).replace(_ALT_ENTER, _ALT_ENTER_REPORTED)
    return _KEY_WITH_TEXT.sub(_uncomposed, text), flags


def _uncomposed(report: re.Match[str]) -> str:
    """A key's report without its text, where the text is only what Alt held down on the
    key's letter would leave."""
    key, alternates, modifiers, event, typed = report.groups()
    held = int(modifiers) - 1
    if held & _ALT and typed.isdigit() and chr(int(typed)).lower() == chr(int(key)):
        return f"\x1b[{key}{alternates};{modifiers}{event}u"
    return report.group()
