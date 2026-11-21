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


def screened(text: str) -> tuple[str, bool]:
    """`text` without the replies to `QUERY`, and whether it held the flags reply."""
    flags = False

    def drop(reply: re.Match[str]) -> str:
        nonlocal flags
        flags = flags or reply.group(1) == "u"
        return ""

    text = _REPLY.sub(drop, text).replace(_ALT_ENTER, _ALT_ENTER_REPORTED)
    return text, flags
