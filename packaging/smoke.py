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
* `--version`, which says what the bundle actually carries. A release is a binary
  with its own interpreter, its own sympy and its own Qt inside it, so the versions
  it reports are the only evidence of what was built - a machine that quietly
  supplied its own Python produces a bundle that passes every other check here and
  is still not the program the suite tested. It is also the whole of what says the
  plot toolkit survived the build, since the program has to import Qt and pyqtgraph
  to answer.
* The first frame, which says the stylesheet was found - Textual refuses to start
  without it.
* The Help menu, which says `help.txt` was found and read as a resource.
* A computation from each of several areas of the mathematics, which says the worker
  was spawned, re-entered the executable as a child, imported sympy there and sent an
  answer back. These cost the most and cover the most: nothing else exercises the
  child process at all, and a module the build's analysis failed to bring along is
  not missing until something reaches for it.

The first two run on Windows, there being no pty to drive the rest through. That
leaves the Windows build covered for unpacking and for what it carries, but not for
anything above that, which is worth knowing about rather than papering over, so it is
reported and not skipped silently.

Plotting is the other gap, and it is reported the same way. A plot is a window in a
child process, and this runs where there is nothing to put one on - the app refuses
the command outright on a Linux machine with no display. What `--version` proves is
that the bundle carries an importable Qt and pyqtgraph. Whether a window actually
opens, and whether the backend PyOpenGL picks by name at run time came along with it,
nothing here asks: the suite draws offscreen but does it against the source tree, so
a build is only answered for that by someone drawing a 3D plot from it by hand.
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

#: How long the screen has to have stood still for a failure to call it settled. Set
#: above a second, which is the rate of the fastest thing that repaints without anyone
#: asking: a running computation writes its elapsed time in whole seconds, so a screen
#: with one under way is redrawn that often and nothing about that is progress.
SETTLED = 1.5

#: A control sequence, which is everything in the stream that is not a character to be
#: placed. Written out rather than reached for from a library so that this script goes
#: on being one file with nothing behind it. The first branch is a CSI, whose parameters
#: and final byte are read because the cursor is moved by one of them; the rest carry a
#: title, name a character set, or are the two-byte escapes, and are matched to be
#: skipped. The parameters are split in two so that a private sequence - `?1049h` and
#: the like, which no reading here wants - leaves `parameters` empty and is stepped over
#: with the others.
#:
#: The fourth branch is the one whose terminator never came. It can only be the last
#: thing in the data, the app having been mid-write when the wait gave up, and it has to
#: be recognised where it is: what would otherwise match is the two-byte escape below
#: it, which takes the `\x1b[` and leaves the parameter digits to be drawn as if they
#: were an answer.
CONTROL = re.compile(
    r"\x1b\[(?P<parameters>[0-9;]*)[0-9;?<>=!\"$' ]*(?P<final>[@-~])"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|\x1b[()][A-B0-9]"
    r"|\x1b[\[\]][^\x1b]*\Z"
    r"|\x1b."
)

#: The final bytes that move the cursor somewhere absolute, which is the whole of the
#: motion the app writes: it addresses a row and writes the row.
CURSOR = ("H", "f")

#: What the expansion of `(x + 1)^7` puts on the screen. A fragment of the middle of
#: the line, so that neither the 2D layout of the exponents above it nor the width of
#: the window can move it out of reach.
EXPANSION = "21·x"

#: What Simplify is asked for, what says the answer arrived, and the part of the
#: mathematics each one stands for. One per area the bundle has to have brought whole,
#: because an area is only missing once something reaches for it: the polynomial above
#: reaches none of these, and a build without sympy's integrals answers every other
#: check here correctly.
#:
#: Each expected fragment was read off a real screen rather than derived from the text
#: form of the answer. An answer is typeset in two dimensions, so only a piece of one
#: line can be waited for, and only answers that come out on a single line are asked
#: for here - `#e` draws as `ê`, and a fraction would put its halves on three lines.
COMPUTATIONS = (
    ("INT(1/(1 + x^2), x)", "ATAN(x)", "integrals"),
    ("SOLVE(x^2 - 4, x)", "[x = -2, x = 2]", "solving"),
    ("EIGENVALUES([[2, 0], [0, 3]])", "[w = 2, w = 3]", "matrices"),
    ("LIM((1 + 1/n)^n, n, inf)", "ê", "limits"),
)

#: What the Author line asks, and what Simplify asks after it.
AUTHOR_PROMPT = "AUTHOR expression:"
SIMPLIFY_PROMPT = "SIMPLIFY expression:"

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


#: The lines `--version` has to produce for the bundle to be worth checking further.
#: A version-shaped answer for each is all this asks; which versions they ought to be
#: is a question for the build workflow, which has `.python-version` and `uv.lock` to
#: compare them against and this script does not. `Qt` and `pyqtgraph` are here
#: because the program has to import the toolkit to print them, which is the only
#: word this script gets on whether plotting survived the build.
CARRIED = ("rederive", "Python", "sympy", "Qt", "pyqtgraph")


def version_check(binary: Path) -> None:
    """The bundle says what it is, and what interpreter, sympy and Qt it brought."""
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
        self._session = b""

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
        """Read until `text` is on the screen, or say what was there instead.

        Only what arrives from here on can match. The app draws the same command menu
        after every command, so a wait that accepted what was on the screen already
        would pass on the last command's frame rather than this one's.

        That makes a failure's evidence narrow, and the two halves of what is reported
        answer the two halves of the question. `self._session` is everything the app
        has drawn since it started, which is what a screen has to be replayed from: a
        frame is written as the differences from the one before it, so bytes that begin
        at this wait draw an incomplete picture. The volume and the timing come from
        `self._seen` alone, which is the only part of the stream this wait is about.
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
            f"The screen it waited in front of:\n{_shown(self._session)}"
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


def _arrival(seen: bytes, quiet: float | None) -> str:
    """Which of the three failures this was, they having nothing to do with each other.

    Nothing drawn at all means the app never answered: it is wedged behind something,
    or gone with the pty still open. Drawing that stopped and stayed stopped means it
    did answer, settled, and settled on something other than what was expected, which
    is a question about the answer rather than about whether one came. Drawing that
    went on to the deadline means the loop is alive and something under it is not: a
    screen is repainted for the clock and the memory gauge whatever else has stalled,
    so paint arriving to the last moment says the app is waiting too.

    The three are worth a different morning each, and a bare timeout says none of them.
    """
    if not seen or quiet is None:
        return "Nothing was drawn in that time."
    volume = f"{len(seen)} bytes were drawn and none held it"
    if quiet > SETTLED:
        return f"{volume}; the last of them {quiet:.1f}s before the deadline."
    return f"{volume}, the last still arriving as it ran out."


def _screen(data: bytes) -> list[str]:
    """The rows of the terminal, replayed from everything written to it.

    Escape sequences are obeyed here rather than deleted. Deleting them concatenates
    the fragments of every repaint into something that reads like a screen and is not
    one - a status bar written at the far corner lands spliced onto the end of whatever
    line was drawn before it, and a frame drawn twice appears twice.

    Little has to be understood to place a character correctly, because little is what
    the app writes. It repaints by addressing a row and writing the whole of it, so the
    only moves this honours are the absolute ones; nothing it sends erases, scrolls or
    inserts. Everything else - colour, the alternate screen, mouse reporting, the
    terminal's replies to its own questions - decides how a character looks or how the
    session behaves and never where a character goes, so it is recognised in order to
    be stepped over. So is the one the timeout cut in half, which is why a failure's
    screen ends in the app's own last row rather than in the digits of a colour.
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
    """The screen as the terminal held it, indented to sit under the failure.

    Blank rows between two written ones are the layout the app chose and are kept. The
    ones above and below everything it wrote are the empty part of a worksheet that
    fills from the foot, and say nothing worth the lines they take in a log.
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

    Simplify is offered the expression just authored, so the prompt takes an empty
    line the way the expansion above does.
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
