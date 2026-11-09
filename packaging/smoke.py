#!/usr/bin/env python3
"""Smoke-test a built executable for the ways a bundle can break.

Run from the repository root, against what the spec left in `dist`:

    uv run python packaging/smoke.py dist/rederive

A bundle that imports cleanly can still be missing files that are only read later: the
help text, the engine's dependencies, the stylesheet. So these checks drive the real
program through a terminal, each covering one way the bundle can be wrong:

* The command line: the archive unpacked and the app's modules import.
* `--version`: what the bundle actually carries. A release ships its own interpreter,
  sympy and Qt, and the reported versions are the only evidence of what was built; a
  bundle that picked up the build machine's Python would pass every other check. It
  also shows that Qt and pyqtgraph survived the build, since the program imports them
  to report their versions.
* The first frame: the stylesheet was found (Textual won't start without it).
* The Help menu: `help.txt` was found and read as a resource.
* A computation from each of several areas of mathematics: the worker spawned,
  re-entered the executable as a child, imported sympy and answered. Nothing else
  exercises the child process, and a module missing from the build only shows up when
  something uses it.

On Windows only the first two run, since there's no pty. The gap is reported rather than
skipped silently.

Plotting isn't checked either, and is reported the same way: there's no display, and
the app refuses to plot on Linux without one. `--version` proves Qt and pyqtgraph import,
but not that a window opens or that the PyOpenGL backend chosen at run time was bundled.
The test suite draws offscreen, but against the source tree.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

#: Terminal size. Wide enough that the menu bar doesn't wrap and break the expectations.
COLUMNS, ROWS = 100, 30

#: Timeout for each wait. Generous, since build machines can be heavily loaded.
PATIENCE = 30.0

#: Maximum session transcript written on failure. A session is typically around 100 KB
#: and all of it is useful; the cap only guards against runaways. The end is kept, since
#: that's where the check failed, and the dropped amount is reported.
TRANSCRIPT = 512 * 1024

#: How long the screen must be still before a failure calls it settled. Over a second,
#: because a running computation redraws its elapsed time every second.
SETTLED = 1.5

#: A terminal control sequence. Written out by hand so this script has no dependencies.
#: The first branch is a CSI, whose parameters and final byte are captured because one of
#: them moves the cursor. The parameters are split so that private sequences like
#: `?1049h` leave `parameters` empty and are skipped. The remaining branches (titles,
#: character sets, two-byte escapes) are matched only to be skipped.
#:
#: The fourth branch matches a sequence cut off at the end of the data, when the wait
#: gave up mid-write. Without it the two-byte escape branch would consume `\x1b[` and
#: leave the parameter digits to be drawn as text.
CONTROL = re.compile(
    r"\x1b\[(?P<parameters>[0-9;]*)[0-9;?<>=!\"$' ]*(?P<final>[@-~])"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|\x1b[()][A-B0-9]"
    r"|\x1b[\[\]][^\x1b]*\Z"
    r"|\x1b."
)

#: Final bytes of absolute cursor moves, the only motion the app uses.
CURSOR = ("H", "f")

#: A fragment from the middle of the expansion of `(x + 1)^7`, chosen so that neither the
#: 2D exponent layout nor the window width can hide it.
EXPANSION = "21·x"

#: Expressions to simplify, the expected screen fragment, and the area of mathematics
#: each covers. A missing part of sympy only shows up when used, so each area needs its
#: own check; a build without integrals would pass the expansion check.
#:
#: Expected fragments were read off a real screen. Answers are typeset in 2D, so only
#: answers that fit on a single line are used (`#e` draws as `ê`; a fraction would
#: span three lines).
COMPUTATIONS = (
    ("INT(1/(1 + x^2), x)", "ATAN(x)", "integrals"),
    ("SOLVE(x^2 - 4, x)", "[x = -2, x = 2]", "solving"),
    ("EIGENVALUES([[2, 0], [0, 3]])", "[w = 2, w = 3]", "matrices"),
    ("LIM((1 + 1/n)^n, n, inf)", "ê", "limits"),
)

#: Prompts for Author and Simplify.
AUTHOR_PROMPT = "AUTHOR expression:"
SIMPLIFY_PROMPT = "SIMPLIFY expression:"

#: Command menu label, shown when the app is ready for the next command.
COMMAND_MENU = "COMMAND:"

#: Quit's confirmation prompt.
ABANDON = "Abandon expressions (Y/N)?"


def report(check: str, detail: str = "") -> None:
    print(f"  ok   {check}{' - ' + detail if detail else ''}", flush=True)


class Failed(Exception):
    """A failed check, described as the report shows it."""


def usage_check(binary: Path) -> None:
    """Check that a nonexistent file argument is refused.

    Getting this message means the archive unpacked, the interpreter started and the
    modules up to the command line parser imported. It's the cheapest real check, so it
    runs first.
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


#: Components `--version` must report. Only a version-like value is checked; the build
#: workflow compares actual versions against `.python-version` and `uv.lock`. Reporting
#: `Qt` and `pyqtgraph` requires importing them, which is this script's only check that
#: plotting survived the build.
CARRIED = ("rederive", "Python", "sympy", "Qt", "pyqtgraph")


def version_check(binary: Path) -> None:
    """Check that `--version` reports the bundled interpreter, sympy and Qt."""
    finished = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True, timeout=PATIENCE
    )
    reported = dict(
        line.split(maxsplit=1)
        for line in finished.stdout.splitlines()
        if len(line.split(maxsplit=1)) == 2
    )
    missing = [name for name in CARRIED if not reported.get(name, "")[:1].isdigit()]
    if finished.returncode != 0 or missing:
        raise Failed(
            f"expected a version for each of {', '.join(CARRIED)}; got exit "
            f"{finished.returncode} and {finished.stdout!r}"
        )
    carried = ", ".join(f"{name} {reported[name]}" for name in CARRIED[1:])
    report("version", f"carries {carried}")


class Terminal:
    """The app running in a pty, with methods to type keys and wait for output."""

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
        self._session = b""

    def type(self, keys: str) -> None:
        os.write(self._master, keys.encode())

    def escape(self) -> None:
        """Press Escape and wait for the command menu before typing anything else.

        Alt-A is sent as Escape followed by `A`, so a key sent right after Escape would
        be read as an Alt combination. Waiting for the menu separates them and confirms
        that going back worked.
        """
        self.type("\x1b")
        self.wait_for(COMMAND_MENU)

    def wait_for(self, text: str, patience: float = PATIENCE) -> None:
        """Read until `text` appears, or raise `Failed` describing what was there.

        Only output arriving after the call can match; the app redraws the same command
        menu after every command, so matching earlier output would pass on the previous
        frame.

        On failure, the screen is replayed from `self._session` (everything drawn since
        startup), because frames are written as differences from the previous one. Volume
        and timing are reported from `self._seen`, the output during this wait.
        """
        import select

        self._seen = b""
        started = time.monotonic()
        deadline = started + patience
        drawn: float | None = None
        while time.monotonic() < deadline:
            if text in _plain(self._seen):
                return
            if select.select([self._master], [], [], 0.05)[0]:
                try:
                    arrived = os.read(self._master, 65536)
                except OSError:
                    break
                drawn = time.monotonic()
                self._seen += arrived
                self._session += arrived
        raise Failed(
            f"waited {patience:.0f}s for {text!r}. "
            f"{_arrival(self._seen, None if drawn is None else deadline - drawn)}\n"
            f"{_held(self._session, self._seen, text)}\n"
            f"The screen it waited in front of:\n{_shown(self._session)}\n"
            f"{_transcript(self._session)}"
        )

    def finish(self) -> int:
        try:
            return self.process.wait(timeout=PATIENCE)
        except subprocess.TimeoutExpired:
            self.process.kill()
            raise Failed("the app did not leave when it was asked to") from None


def _plain(data: bytes) -> str:
    """Return the output with escape sequences removed."""
    text = data.decode("utf-8", "replace")
    for pattern in (
        r"\x1b\][^\x07\x1b]*(\x07|\x1b\\)",
        r"\x1b[\[\?][0-9;]*[a-zA-Z]",
        r"\x1b[()][A-B0-9]",
        r"\x1b.",
    ):
        text = re.sub(pattern, "", text)
    return text


def _arrival(seen: bytes, quiet: float | None) -> str:
    """Classify a timeout into one of three unrelated failures.

    Nothing drawn means the app never responded: it's stuck, or gone with the pty still
    open. Drawing that stopped means the app settled on something other than expected.
    Drawing that continued to the deadline means the event loop is alive (the clock and
    memory gauge keep repainting) but the app is stuck waiting on something.
    """
    if not seen or quiet is None:
        return "Nothing was drawn in that time."
    volume = f"{len(seen)} bytes were drawn and none held it"
    if quiet > SETTLED:
        return f"{volume}; the last of them {quiet:.1f}s before the deadline."
    return f"{volume}, the last still arriving as it ran out."


def _held(session: bytes, seen: bytes, text: str) -> str:
    """Report which of four successive forms of the output contain the expected text.

    Each form is derived from the previous one, so this distinguishes an app that never
    wrote the text (a program bug) from this script misreading correct bytes (a script
    bug). The raw bytes come first, as the primary record. Only facts are reported, not
    conclusions drawn from them.
    """
    wanted = text.encode()
    offset = session.find(wanted)
    drawn = [line for line in _screen(session) if text in line]
    places = (
        ("in the bytes the app has written", offset >= 0),
        ("in the bytes this wait was reading", wanted in seen),
        ("in those with the escape sequences taken out", text in _plain(seen)),
        ("on the screen replayed from them", bool(drawn)),
    )
    lines = [f"    {'yes' if found else 'no':<4}{where}" for where, found in places]
    if offset >= 0:
        lines[0] += f", from byte {offset}"
    if drawn:
        lines[3] += f", on {'; '.join(line.strip() for line in drawn)!r}"
    return "Where the text was:\n" + "\n".join(lines)


def _transcript(session: bytes) -> str:
    """Format everything the app wrote so it can be logged and exactly recovered.

    The replayed screen is this script's interpretation and could be wrong, so the raw
    bytes are logged too for later analysis.

    They're written as Python bytes literals rather than base64, so escape sequences stay
    readable (`\\x1b[5;1H`) and mangled characters are visible. Each line is a complete
    literal, so trailing spaces survive log indentation and prefixes; joining the
    evaluated literals recovers the stream exactly.
    """
    dropped = max(len(session) - TRANSCRIPT, 0)
    kept = session[dropped:]
    rows = [kept[at : at + 64] for at in range(0, len(kept), 64)]
    said = f"{len(session)} bytes"
    if dropped:
        said += f", of which the first {dropped} are dropped here"
    return (
        f"Everything the app wrote ({said}), as one bytes literal a line:\n"
        + "\n".join(f"    {row!r}" for row in rows)
    )


def _screen(data: bytes) -> list[str]:
    """Replay the output onto a terminal grid and return its rows.

    Escape sequences are interpreted rather than stripped, since stripping splices
    repaints together (a status bar written in a corner ends up appended to the previous
    line, and a frame drawn twice appears twice).

    The app repaints by addressing a row and writing all of it, and never erases,
    scrolls or inserts, so only absolute cursor moves are honored. Other sequences
    (color, alternate screen, mouse reporting, terminal replies) don't affect placement
    and are skipped, as is a sequence truncated by the timeout.
    """
    grid = [[" "] * COLUMNS for _ in range(ROWS)]
    text = data.decode("utf-8", "replace")
    row = column = position = 0
    while position < len(text):
        control = CONTROL.match(text, position) if text[position] == "\x1b" else None
        if control is not None:
            position = control.end()
            if control["final"] in CURSOR:
                where = (control["parameters"].split(";") + ["", ""])[:2]
                row = min(max(int(where[0] or 1) - 1, 0), ROWS - 1)
                column = min(max(int(where[1] or 1) - 1, 0), COLUMNS - 1)
            continue
        character = text[position]
        position += 1
        if character == "\r":
            column = 0
        elif character == "\n":
            row = min(row + 1, ROWS - 1)
        elif character >= " " and column < COLUMNS:
            grid[row][column] = character
            column += 1
    return ["".join(cells).rstrip() for cells in grid]


def _shown(data: bytes) -> str:
    """Return the replayed screen, indented for the failure report.

    Blank rows between written ones are kept as layout; leading and trailing blank rows
    are dropped.
    """
    lines = _screen(data)
    while lines and not lines[-1]:
        lines.pop()
    while lines and not lines[0]:
        lines.pop(0)
    if not lines:
        return "    (nothing was ever drawn on it)"
    return "\n".join(f"    {line}" for line in lines)


def _simplify(terminal: Terminal, expression: str, expected: str) -> None:
    """Author `expression`, Simplify it, and wait for the answer to appear.

    Simplify defaults to the expression just authored, so an empty line accepts it.
    """
    terminal.type("A")
    terminal.wait_for(AUTHOR_PROMPT)
    terminal.type(f"{expression}\r")
    terminal.wait_for(COMMAND_MENU)
    terminal.type("S")
    terminal.wait_for(SIMPLIFY_PROMPT)
    terminal.type("\r")
    terminal.wait_for(expected)


def session_checks(binary: Path) -> None:
    """Start the app, open help, run computations, and quit."""
    terminal = Terminal(binary)
    try:
        terminal.wait_for("Press H for help")
        report("first frame", "the stylesheet was found")

        terminal.type("H")
        terminal.wait_for("Rederive Help Menu")
        report("help", "help.txt was found")
        terminal.escape()

        terminal.type("A")
        terminal.wait_for(AUTHOR_PROMPT)
        terminal.type("(x + 1)^7\r")
        terminal.wait_for(COMMAND_MENU)
        terminal.type("E")
        terminal.wait_for("EXPAND expression:")
        terminal.type("\r")
        terminal.wait_for(EXPANSION)
        report("engine", "the worker spawned and expanded (x + 1)^7")

        for expression, expected, area in COMPUTATIONS:
            _simplify(terminal, expression, expected)
            report(area, expression)

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
        version_check(binary)
        if sys.platform == "win32":
            print("  --   screen, help and engine not checked: no pty on Windows")
        else:
            session_checks(binary)
        print("  --   plotting not drawn: no display to open a plot window on")
    except Failed as failure:
        print(f"  FAIL {failure}", file=sys.stderr)
        return 1
    print("smoke: passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
