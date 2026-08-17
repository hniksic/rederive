"""Textual's terminal driver, made to leave when the terminal does.

Textual reads the keyboard on a thread of its own, and on every platform but Windows
that thread takes the end of its input for nothing having been typed: it goes back to
waiting, the descriptor is readable still, and the two make a loop that holds a core
for as long as the process lives. That is what an app whose terminal has gone away is
left doing - an ssh session dropped without a SIGHUP behind it, a pty whose master was
closed by the harness that opened it - and what one started with its input on the null
device does from the first frame. Textualize/textual#6690 is the upstream report.

Here the end of input is the end of the session. There is nobody left to type, no
screen anything can be seen on, and a worker that would otherwise hold its memory for
a worksheet nobody will read again; so the driver asks the app to leave, the way Quit
does once it has been answered, and gives the thread back. What is here besides that
is upstream's own loop, which has to be here in full because the read that has to be
looked at is in the middle of it.
"""

from __future__ import annotations

import os
import selectors
from codecs import getincrementaldecoder

from textual._parser import ParseError
from textual._xterm_parser import XTermParser
from textual.drivers.linux_driver import LinuxDriver

__all__ = ["TerminalDriver"]

#: How long one wait for input lasts before the loop looks at whether it has been
#: asked to stop, as upstream has it.
_POLL = 0.1

#: How much is read at a time, as upstream has it.
_CHUNK = 4096


class TerminalDriver(LinuxDriver):
    """The Linux and macOS driver, with the terminal's hangup read as Quit."""

    def run_input_thread(self) -> None:
        """Wait for input and dispatch events, until asked to stop or hung up on."""
        selector = selectors.SelectSelector()
        selector.register(self.fileno, selectors.EVENT_READ)
        parser = XTermParser(self._debug)
        decode = getincrementaldecoder("utf-8")().decode

        def pump(ready: list[tuple[selectors.SelectorKey, int]], final: bool) -> bool:
            """Feed the parser what is readable; false once the input has ended.

            An empty read is the end of it, and so is a read the kernel refuses: a
            pty whose master has gone answers with `EIO` on some systems and with
            nothing on others, and neither is a terminal worth waiting on.
            """
            for at, (_, mask) in enumerate(ready):
                if not mask & selectors.EVENT_READ:
                    continue
                try:
                    data = os.read(self.fileno, _CHUNK)
                except OSError:
                    data = b""
                if not data:
                    return False
                text = decode(data, final=final and at == len(ready) - 1)
                for event in parser.feed(text):
                    self.process_message(event)
            for event in parser.tick():
                self.process_message(event)
            return True

        try:
            while not self.exit_event.is_set():
                if not pump(selector.select(_POLL), final=False):
                    self._hung_up()
                    return
            selector.unregister(self.fileno)
            pump(selector.select(_POLL), final=True)
        finally:
            selector.close()
            try:
                for _ in parser.feed(""):
                    pass
            except (EOFError, ParseError):
                pass

    def _hung_up(self) -> None:
        """Ask the app to leave, from the input thread, which the app's loop is not."""
        self._app.call_later(self._app.exit)
