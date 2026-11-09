"""Textual's terminal driver, changed to exit when the terminal goes away.

On every platform except Windows, Textual's input thread treats end of input as "nothing
typed yet" and goes back to waiting. Since the descriptor stays readable, this becomes a
busy loop that pins a core for the life of the process. It happens when the terminal
disappears (an ssh session dropped without SIGHUP, a pty whose master was closed) and,
from the start, when stdin is the null device. The upstream report is
Textualize/textual#6690.

This driver treats end of input as the end of the session: it asks the app to exit, as
a confirmed Quit does, and ends the thread. The rest is a copy of upstream's input loop,
needed in full because the read that must be checked sits in the middle of it.
"""

from __future__ import annotations

import os
import selectors
from codecs import getincrementaldecoder

from textual._parser import ParseError
from textual._xterm_parser import XTermParser
from textual.drivers.linux_driver import LinuxDriver

__all__ = ["TerminalDriver"]

#: How long each wait for input lasts before checking whether to stop (as upstream).
_POLL = 0.1

#: Bytes read at a time (as upstream).
_CHUNK = 4096


class TerminalDriver(LinuxDriver):
    """The Linux and macOS driver, treating a terminal hangup as Quit."""

    def run_input_thread(self) -> None:
        """Wait for input and dispatch events, until asked to stop or hung up on."""
        selector = selectors.SelectSelector()
        selector.register(self.fileno, selectors.EVENT_READ)
        parser = XTermParser(self._debug)
        decode = getincrementaldecoder("utf-8")().decode

        def pump(ready: list[tuple[selectors.SelectorKey, int]], final: bool) -> bool:
            """Feed readable input to the parser; return false once input has ended.

            Both an empty read and a failed read count as the end: a pty whose master
            is gone returns `EIO` on some systems and an empty read on others.
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
        """Ask the app to exit. Runs on the input thread, not the app's loop."""
        self._app.call_later(self._app.exit)
