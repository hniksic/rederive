"""The app in a Windows pseudo-console, typed at as Windows Terminal types.

Windows Terminal hosts a program through ConPTY and, asked by ConPTY for
win32-input-mode, sends each key as `CSI Vk;Sc;Uc;Kd;Cs;Rc _`: the key's records, with
the modifiers held. This test plays the terminal's part: it starts the app in a ConPTY
pseudo-console, writes keys that way (and a paste and a click as the terminal sends
them, as text), and reads what the app shows from the screen ConPTY draws, rendered by a
terminal emulator. It checks what only the Windows console driver makes work: Shift-Enter
starting a row, and a paste keeping its line breaks.

Windows only, since it needs ConPTY.
"""

from __future__ import annotations

import re
import sys
import threading
import time

import pytest

if sys.platform != "win32":
    pytest.skip("needs the Windows pseudo-console", allow_module_level=True)

import ctypes

import pyte
from winpty import Backend, PtyProcess

ROWS, COLUMNS = 25, 80

SHIFT_PRESSED = 0x0010
ENHANCED_KEY = 0x0100
VK_SHIFT, VK_RETURN, VK_ESCAPE, VK_LEFT = 0x10, 0x0D, 0x1B, 0x25

USER32 = ctypes.windll.user32


def record(vk: int, char: str, down: bool, state: int) -> str:
    """One key record in win32-input-mode."""
    scan = USER32.MapVirtualKeyW(vk, 0)
    code = ord(char) if char else 0
    return f"\x1b[{vk};{scan};{code};{int(down)};{state};1_"


def key(vk: int, char: str = "", state: int = 0) -> str:
    """A key pressed and released, Shift around it if it is held."""
    keys = [record(vk, char, True, state), record(vk, char, False, state)]
    if state & SHIFT_PRESSED:
        keys = [record(VK_SHIFT, "", True, SHIFT_PRESSED), *keys]
        keys.append(record(VK_SHIFT, "", False, 0))
    return "".join(keys)


def character(text: str) -> str:
    """The key typing a character, on the keyboard layout in effect."""
    scan = USER32.VkKeyScanW(ord(text))
    vk, shifts = scan & 0xFF, (scan >> 8) & 0xFF
    return key(vk, text, SHIFT_PRESSED if shifts & 1 else 0)


class Console:
    """The app in a pseudo-console, and the screen it draws."""

    def __init__(self, cwd: str) -> None:
        self.screen = pyte.Screen(COLUMNS, ROWS)
        self._stream = pyte.Stream(self.screen)
        self._lock = threading.Lock()
        self.process = PtyProcess.spawn(
            [sys.executable, "-m", "rederive"],
            cwd=cwd,
            dimensions=(ROWS, COLUMNS),
            backend=Backend.ConPTY,
        )
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        # ConPTY blocks the program once its output pipe is full, so keep reading.
        while self.process.isalive():
            try:
                output = self.process.read(65536)
            except EOFError:
                return
            with self._lock:
                self._stream.feed(output)

    def lines(self) -> list[str]:
        with self._lock:
            return [line.rstrip() for line in self.screen.display]

    def text(self) -> str:
        return "\n".join(self.lines())

    def wait_for(self, pattern: str, timeout: float = 15) -> re.Match[str]:
        """Wait until the screen shows `pattern`, a regular expression."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if found := re.search(pattern, self.text(), re.MULTILINE):
                return found
            time.sleep(0.1)
        raise AssertionError(f"never showed {pattern!r}; the screen:\n{self.text()}")

    def write(self, text: str) -> None:
        """Send input, then leave time enough that the next key is read separately."""
        self.process.write(text)
        time.sleep(0.15)

    def type(self, text: str) -> None:
        for each in text:
            self.write(character(each))

    def close(self) -> None:
        if self.process.isalive():
            self.process.terminate(force=True)


@pytest.fixture
def console(tmp_path):
    console = Console(str(tmp_path))
    yield console
    console.close()


def entry(number: int, text: str) -> str:
    return rf"^#{number}:\s+{re.escape(text)}$"


def test_keys_pastes_and_the_mouse_reach_the_app(console):
    console.wait_for(r"COMMAND:", timeout=60)

    # The hint names Shift-Enter, which the driver says the keyboard reports.
    console.type("a")
    console.wait_for(r"Shift-Enter: new line")

    # Shift-Enter starts a row on a finished expression; Enter enters it.
    console.type("23")
    console.write(key(VK_RETURN, "\r", SHIFT_PRESSED))
    console.type("+ 45")
    console.wait_for(r"^\s+\+ 45$")
    assert not re.search(r"^#1:", console.text(), re.MULTILINE)
    console.write(key(VK_RETURN, "\r"))
    console.wait_for(entry(1, "23 + 45"))

    # Plain Enter on an unfinished expression starts a row instead.
    console.type("a")
    console.type("[1, 2,")
    console.write(key(VK_RETURN, "\r"))
    console.type("3]")
    console.wait_for(r"^\s+3\]$")
    assert not re.search(r"^#2:", console.text(), re.MULTILINE)
    console.write(key(VK_RETURN, "\r"))
    console.wait_for(entry(2, "[1, 2, 3]"))

    # A paste arrives as text, which ConPTY turns into key records all at once. Its line
    # break makes a row, though typing Enter there would have entered `77`.
    console.type("a")
    console.write("77\r+ 88")
    console.wait_for(r"^\s+\+ 88$")
    assert not re.search(r"^#3:", console.text(), re.MULTILINE)
    console.write(key(VK_RETURN, "\r"))
    console.wait_for(entry(3, "77 + 88"))

    # An arrow key moves the cursor on the line.
    console.type("a")
    console.type("12")
    console.write(key(VK_LEFT, "", ENHANCED_KEY))
    console.type("3")
    console.write(key(VK_RETURN, "\r"))
    console.wait_for(entry(4, "132"))

    # Clicking a menu word invokes it, and Esc backs out of the command again.
    console.wait_for(r"COMMAND: Author")
    lines = console.lines()
    row = next(at for at, line in enumerate(lines) if "COMMAND: Author" in line)
    # SGR mouse reports count from 1; this is the word's second cell.
    x, y = lines[row].index("Author") + 2, row + 1
    console.write(f"\x1b[<0;{x};{y}M\x1b[<0;{x};{y}m")
    console.wait_for(r"AUTHOR expression:")
    console.write(key(VK_ESCAPE, "\x1b"))
    console.wait_for(r"COMMAND:")
