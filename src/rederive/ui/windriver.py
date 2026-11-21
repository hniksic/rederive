"""Textual's Windows driver, reading console input records instead of VT input.

Textual sets the console's input to `ENABLE_VIRTUAL_TERMINAL_INPUT`, where the console
hands over the terminal's text. Under ConPTY, which is how Windows Terminal hosts a
program, that text is the console's rendering of key records, and it renders Shift-Enter
as a bare CR. With VT input off, the records arrive as they are, Shift included. This
driver turns VT input off and translates the records into terminal input (see
`wininput`), so that Textual's parser and everything after it work as on other
platforms. It is the approach crossterm takes on Windows.

Copied from upstream, because the parts to change sit inside them: the start of
application mode, without asking for the kitty keyboard protocol (the records already
report every modifier) or for bracketed paste (VT input being off, no markers would
come; `wininput` recognizes a paste instead), and the input thread's loop, reading
records as `win32.EventMonitor.run` does but translating them all rather than keeping
only their characters. Window size changes are handled as upstream does.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Callable
from ctypes import byref, wintypes

from textual import constants
from textual._xterm_parser import XTermParser
from textual.drivers import win32
from textual.drivers._writer_thread import WriterThread
from textual.drivers.windows_driver import WindowsDriver

from rederive.ui.wininput import FocusRecord, KeyRecord, MouseRecord, Record, Translator

__all__ = ["ConsoleDriver"]

#: Input modes: window size and mouse records, with quick edit (selecting with the
#: mouse in the console window) off, which `ENABLE_EXTENDED_FLAGS` without
#: `ENABLE_QUICK_EDIT_MODE` does.
INPUT_MODE = (
    win32.ENABLE_WINDOW_INPUT | win32.ENABLE_MOUSE_INPUT | win32.ENABLE_EXTENDED_FLAGS
)

KEY_EVENT = 0x0001
MOUSE_EVENT = 0x0002
WINDOW_BUFFER_SIZE_EVENT = 0x0004
FOCUS_EVENT = 0x0010

#: Records read at a time (as upstream).
_MAX_RECORDS = 1024

#: How long to wait for each read before checking whether to stop, in milliseconds
#: (as upstream).
_POLL = 100

#: After a read with keys in it, records arriving within this many milliseconds belong
#: with it, so that a paste the console delivers in parts is still read together.
_GATHER = 10

#: The longest a gathering of records goes on, in seconds.
_GATHER_LIMIT = 0.5


def _application_mode() -> Callable[[], None]:
    """Set the console modes; return what restores the previous ones."""
    terminal_in, terminal_out = sys.__stdin__, sys.__stdout__
    mode_in = win32.get_console_mode(terminal_in)
    mode_out = win32.get_console_mode(terminal_out)

    def restore() -> None:
        win32.set_console_mode(terminal_in, mode_in)
        win32.set_console_mode(terminal_out, mode_out)

    win32.set_console_mode(
        terminal_out, mode_out | win32.ENABLE_VIRTUAL_TERMINAL_PROCESSING
    )
    win32.set_console_mode(terminal_in, INPUT_MODE)
    return restore


class ConsoleDriver(WindowsDriver):
    """The Windows driver, reading console input records, Shift-Enter included."""

    def start_application_mode(self) -> None:
        """Start as upstream does, with VT input off and the records' input thread."""
        loop = asyncio.get_running_loop()
        self._restore_console = _application_mode()
        self._writer_thread = WriterThread(self._file)
        self._writer_thread.start()
        self.write("\x1b[?1049h")  # Enable alt screen
        self._enable_mouse_support()
        self.write("\x1b[?25l")  # Hide cursor
        self.flush()
        self._app.shift_enter = True
        self._event_thread = _InputThread(
            loop, self._app, self.exit_event, self.process_message
        )
        self._event_thread.start()


class _InputThread(win32.EventMonitor):
    """Reads console input records and sends what they translate to into the app."""

    def run(self) -> None:
        parser = XTermParser(debug=constants.DEBUG)
        translator = Translator()
        handle = win32.GetStdHandle(win32.STD_INPUT_HANDLE)
        buffer = (win32.INPUT_RECORD * _MAX_RECORDS)()
        count = wintypes.DWORD(0)
        read = win32.KERNEL32.ReadConsoleInputW

        def gather() -> tuple[list[Record], tuple[int, int] | None]:
            """Read the records waiting, and any that follow closely."""
            records: list[Record] = []
            size = None
            deadline = time.monotonic() + _GATHER_LIMIT
            while True:
                read(handle, byref(buffer), _MAX_RECORDS, byref(count))
                keys = False
                for record in buffer[: count.value]:
                    if record.EventType == WINDOW_BUFFER_SIZE_EVENT:
                        new = record.Event.WindowBufferSizeEvent.dwSize
                        size = (new.X, new.Y)
                    elif (plain := _plain(record)) is not None:
                        records.append(plain)
                        keys = keys or isinstance(plain, KeyRecord)
                # Only keys are waited for, so that mouse motion isn't held back.
                if (
                    not keys
                    or time.monotonic() > deadline
                    or win32.wait_for_handles([handle], _GATHER) is None
                ):
                    return records, size

        try:
            while not self.exit_event.is_set():
                for event in parser.tick():
                    self.process_event(event)
                if win32.wait_for_handles([handle], _POLL) is None:
                    continue
                records, size = gather()
                # An empty feed is the end of input to the parser.
                if text := translator.feed(records):
                    for event in parser.feed(text):
                        self.process_event(event)
                if size is not None:
                    self.on_size_change(*size)
        except Exception as error:
            self.app.log.error("EVENT MONITOR ERROR", error)


def _plain(record: win32.INPUT_RECORD) -> Record | None:
    """A console input record as a plain value, if it is one `wininput` reads."""
    kind = record.EventType
    if kind == KEY_EVENT:
        key = record.Event.KeyEvent
        return KeyRecord(
            bool(key.bKeyDown),
            key.wVirtualKeyCode,
            key.uChar.UnicodeChar,
            key.dwControlKeyState,
            key.wRepeatCount,
        )
    if kind == MOUSE_EVENT:
        mouse = record.Event.MouseEvent
        return MouseRecord(
            mouse.dwMousePosition.X,
            mouse.dwMousePosition.Y,
            mouse.dwButtonState,
            mouse.dwControlKeyState,
            mouse.dwEventFlags,
        )
    if kind == FOCUS_EVENT:
        return FocusRecord(bool(record.Event.FocusEvent.bSetFocus))
    return None
