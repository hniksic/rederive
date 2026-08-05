#!/usr/bin/env python3
"""Ask a built executable the four questions a bundle can fail on.

Run from the repository root, against what the spec left in `dist`:

    uv run python packaging/smoke.py dist/rederive

An executable that imports cleanly is not an executable that works. Everything a
bundle can lose it loses silently at build time and loudly at run time, in a place no
import can reach: the help text is not read until the user asks for help, the engine
does not run until there is something to compute, and the stylesheet is not read by
anything the analysis can see at all. So the checks here drive the real program
through a terminal rather than importing anything from it, and each one stands for a
way the bundle can be wrong:

* The command line, which says the archive unpacked and the app's own modules import.
* The first frame, which says the stylesheet was found - Textual refuses to start
  without it.
* The Help menu, which says `help.txt` was found and read as a resource.
* An expansion, which says the worker was spawned, re-entered the executable as a
  child, imported sympy there and sent an answer back. It is the check that costs the
  most and covers the most: nothing else exercises the child process at all.

Only the first runs on Windows, there being no pty to drive the rest through. That
leaves the Windows build covered for unpacking and not for anything above it, which
is worth knowing about rather than papering over, so it is reported and not skipped
silently.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

#: The terminal the app is driven in. Wide enough that the menu bar is not wrapped
#: into something the expectations below cannot find.
COLUMNS, ROWS = 100, 30

#: How long any one thing is waited for. Generous: this runs on build machines whose
#: load is nobody's to predict, and a slow answer is not a wrong one.
PATIENCE = 30.0

#: What the expansion of `(x + 1)^7` puts on the screen. A fragment of the middle of
#: the line, so that neither the 2D layout of the exponents above it nor the width of
#: the window can move it out of reach.
EXPANSION = "21·x"

#: The label the command menu carries, which is how the app says it is back at the
#: top and ready for the next command.
COMMAND_MENU = "COMMAND:"

#: What Quit asks before it goes.
ABANDON = "Abandon expressions (Y/N)?"


def report(check: str, detail: str = "") -> None:
    print(f"  ok   {check}{' - ' + detail if detail else ''}", flush=True)


class Failed(Exception):
    """A check the executable did not pass, named the way the report reads."""


def usage_check(binary: Path) -> None:
    """A named file that is not there is refused, which takes the whole app to say.

    Reaching this message means the archive unpacked, the interpreter started and
    every module from the entry point down to the command line parser imported. It is
    the cheapest question that has a real answer, so it is asked first.
    """
    finished = subprocess.run(
        [str(binary), "no-such-worksheet.mth"],
        capture_output=True,
        text=True,
        timeout=PATIENCE,
    )
    if finished.returncode != 2 or "no such file" not in finished.stderr:
        raise Failed(
            f"expected the usage refusal, got exit {finished.returncode} "
            f"and {finished.stderr!r}"
        )
    report("command line", "refuses a file that is not there")


class Terminal:
    """The app in a pty, and the two things a check needs: type this, wait for that."""

    def __init__(self, binary: Path) -> None:
        import fcntl
        import pty
        import struct
        import termios

        self._master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLUMNS, 0, 0))
        self.process = subprocess.Popen(
            [str(binary)],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=dict(os.environ, TERM="xterm-256color", COLORTERM="truecolor"),
        )
        os.close(slave)
        self._seen = b""

    def type(self, keys: str) -> None:
        os.write(self._master, keys.encode())

    def escape(self) -> None:
        """Go back one step, and be sure the app got there before typing again.

        Escape is the one key that cannot be sent with another behind it. A terminal
        writes Alt-A as Escape then `A`, so a key arriving in the same breath as an
        Escape is read as the two of them together and lands somewhere nobody asked
        for. Waiting for the command menu to come back separates them, and checks
        that going back worked while it is at it.
        """
        self.type("\x1b")
        self.wait_for(COMMAND_MENU)

    def wait_for(self, text: str, patience: float = PATIENCE) -> None:
        """Read until `text` is on the screen, or say what was there instead."""
        import select

        self._seen = b""
        deadline = time.monotonic() + patience
        while time.monotonic() < deadline:
            if text in _plain(self._seen):
                return
            if select.select([self._master], [], [], 0.05)[0]:
                try:
                    self._seen += os.read(self._master, 65536)
                except OSError:
                    break
        raise Failed(
            f"waited {patience:.0f}s for {text!r}, screen held:\n{_shown(self._seen)}"
        )

    def finish(self) -> int:
        try:
            return self.process.wait(timeout=PATIENCE)
        except subprocess.TimeoutExpired:
            self.process.kill()
            raise Failed("the app did not leave when it was asked to") from None


def _plain(data: bytes) -> str:
    """The screen with the escape sequences that drew it taken back out."""
    text = data.decode("utf-8", "replace")
    for pattern in (
        r"\x1b\][^\x07\x1b]*(\x07|\x1b\\)",
        r"\x1b[\[\?][0-9;]*[a-zA-Z]",
        r"\x1b[()][A-B0-9]",
        r"\x1b.",
    ):
        text = re.sub(pattern, "", text)
    return text


def _shown(data: bytes) -> str:
    """The last of the screen, for a failure to be read against."""
    lines = [line.rstrip() for line in _plain(data).splitlines() if line.strip()]
    return "\n".join(f"    {line}" for line in lines[-12:])


def session_checks(binary: Path) -> None:
    """Start the app, ask it for help, make it compute, and let it go."""
    terminal = Terminal(binary)
    try:
        terminal.wait_for("Press H for help")
        report("first frame", "the stylesheet was found")

        terminal.type("H")
        terminal.wait_for("Rederive Help Menu")
        report("help", "help.txt was found")
        terminal.escape()

        terminal.type("A")
        terminal.wait_for("AUTHOR expression:")
        terminal.type("(x + 1)^7\r")
        terminal.wait_for(COMMAND_MENU)
        terminal.type("E")
        terminal.wait_for("EXPAND expression:")
        terminal.type("\r")
        terminal.wait_for(EXPANSION)
        report("engine", "the worker spawned and expanded (x + 1)^7")

        terminal.type("Q")
        terminal.wait_for(ABANDON)
        terminal.type("Y")
        code = terminal.finish()
        if code != 0:
            raise Failed(f"leaving the app gave exit {code}")
        report("quit", "left with nothing behind it")
    finally:
        if terminal.process.poll() is None:
            terminal.process.kill()


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("binary", type=Path, help="the executable the spec built")
    parsed = parser.parse_args(arguments)
    binary = parsed.binary
    if not binary.exists():
        print(f"smoke: {binary}: no such file", file=sys.stderr)
        return 2

    print(f"smoke: {binary}", flush=True)
    try:
        usage_check(binary)
        if sys.platform == "win32":
            print("  --   screen, help and engine not checked: no pty on Windows")
        else:
            session_checks(binary)
    except Failed as failure:
        print(f"  FAIL {failure}", file=sys.stderr)
        return 1
    print("smoke: passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
