#!/usr/bin/env python3
"""Boot the built demo in a real browser and ask it one question.

    uv run --with playwright python tools/smoke_web.py
    ... --directory build/web --browser chromium --shots build/shots

What it does is what a user does in their first ten seconds: open the page, wait
for the menu, author `(x + 1)^2 - (x - 1)^2`, simplify it, and read `4·x` off
the screen. Everything between those keystrokes and that answer is the whole
demo - Pyodide in two instances, the Textual driver, xterm.js, the engine worker
and the pickles between them - so a run that gets the answer has exercised the
lot, and one that does not says which step it stopped at.

The screen is read out of xterm.js's own buffer rather than off the pixels. That
is what the program wrote, character for character, and it is the only reading a
test can assert against: a screenshot says whether a person would like it and
nothing about whether the mathematics arrived.

**Nothing here is a timing.** Stage 0 measured Playwright's Firefox running
WebAssembly some eight times slower than the machine's own, so the waits below
are generous on purpose and none of them means anything about how fast the demo
is.

The page is served by `python -m http.server`, which is what a static host is
without the compression - the demo names no other host at run time, so a
directory and a port are the whole of what it needs.
"""

from __future__ import annotations

import argparse
import http.server
import sys
import threading
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: What is authored, and what the program must put on the screen for it. The
#: difference of two squares, because it is the README's own example and
#: because its answer is one line of two characters: an answer that has to be
#: matched across a two-dimensional render would be testing the display rather
#: than the demo.
AUTHORED = "(x+1)^2-(x-1)^2"
ANSWER = "4·x"

#: A word off the opening menu, which is how the page says the program is up.
#: Not a timing: it is waited for until it appears or the clock runs out.
MENU = "Author"

#: How long each step may take before the run is called a failure. Firefox
#: under Playwright is slow enough at WebAssembly that a budget written for
#: Chromium would fail it for no reason, so both get the slower one.
BOOT_SECONDS = 180
ANSWER_SECONDS = 120

#: How the buffer is read back: every row of the screen as text, which is
#: exactly what a terminal would have shown.
SCREEN = """() => {
  const term = window.rederive && window.rederive.term;
  if (!term) return '';
  const buffer = term.buffer.active;
  const lines = [];
  for (let row = 0; row < buffer.length; row += 1) {
    lines.push(buffer.getLine(row).translateToString(true));
  }
  return lines.join('\\n');
}"""


def main(arguments: list[str] | None = None) -> int:
    parsed = _command_line(arguments)
    directory = Path(parsed.directory).resolve()
    if not (directory / "index.html").is_file():
        raise SystemExit(f"{directory}: not a built demo - run tools/build_web.py")
    shots = Path(parsed.shots).resolve() if parsed.shots else None
    if shots is not None:
        shots.mkdir(parents=True, exist_ok=True)

    server, port = _serving(directory)
    try:
        failures = [
            name
            for name in parsed.browser
            if not _smoke(name, f"http://127.0.0.1:{port}/", shots)
        ]
    finally:
        server.shutdown()
    if failures:
        print(f"FAILED: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"passed: {', '.join(parsed.browser)}", file=sys.stderr)
    return 0


def _command_line(arguments: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the built browser demo.")
    parser.add_argument("--directory", default=str(ROOT / "build" / "web"))
    parser.add_argument(
        "--browser",
        action="append",
        choices=("chromium", "firefox", "webkit"),
        help="which browser to run in; repeatable, both by default",
    )
    parser.add_argument("--shots", help="write a screenshot of each run here")
    parsed = parser.parse_args(arguments)
    parsed.browser = parsed.browser or ["chromium", "firefox"]
    return parsed


def _serving(directory: Path) -> tuple[http.server.HTTPServer, int]:
    """A static server on a port the machine chose, in a thread of its own."""
    handler = partial(_Quiet, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


class _Quiet(http.server.SimpleHTTPRequestHandler):
    """The same server with its log off: 800 requests of it are not a report."""

    def log_message(self, format: str, *args: object) -> None:
        pass


def _smoke(name: str, url: str, shots: Path | None) -> bool:
    from playwright.sync_api import sync_playwright

    print(f"-- {name} {'-' * 40}", file=sys.stderr)
    with sync_playwright() as playwright:
        browser = getattr(playwright, name).launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        troubles: list[str] = []
        page.on("pageerror", lambda error: troubles.append(str(error)))
        page.on(
            "console",
            lambda message: (
                troubles.append(message.text) if message.type == "error" else None
            ),
        )
        try:
            return _asked(page, url, name, shots, troubles)
        finally:
            browser.close()


def _asked(page, url: str, name: str, shots: Path | None, troubles: list[str]) -> bool:
    page.goto(url)
    if not _waiting(page, MENU, BOOT_SECONDS):
        return _failed(name, "the program never put its menu up", page, shots, troubles)

    # Author, and then Simplify what was authored: two commands off the menu,
    # each a letter, and each ending in the Enter that takes the line.
    page.keyboard.press("a")
    page.wait_for_timeout(500)
    page.keyboard.type(AUTHORED)
    page.keyboard.press("Enter")
    if not _waiting(page, "#1:", 30):
        return _failed(name, "the expression was not authored", page, shots, troubles)

    page.keyboard.press("s")
    page.wait_for_timeout(500)
    page.keyboard.press("Enter")
    if not _waiting(page, ANSWER, ANSWER_SECONDS):
        return _failed(name, f"Simplify never answered {ANSWER}", page, shots, troubles)

    if shots is not None:
        page.screenshot(path=str(shots / f"{name}.png"))
    if troubles:
        return _failed(name, "the page reported errors", page, shots, troubles)
    print(f"{name}: authored, simplified, and read {ANSWER} off the screen",
          file=sys.stderr)
    return True


def _waiting(page, wanted: str, seconds: int) -> bool:
    """Watch the terminal's own buffer until it holds `wanted`."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if wanted in page.evaluate(SCREEN):
            return True
        page.wait_for_timeout(250)
    return False


def _failed(name, what, page, shots: Path | None, troubles: list[str]) -> bool:
    """Say what went wrong, and leave the screen behind to be looked at."""
    print(f"{name}: {what}", file=sys.stderr)
    print(page.evaluate(SCREEN), file=sys.stderr)
    for trouble in troubles:
        print(f"{name}: page said: {trouble}", file=sys.stderr)
    if shots is not None:
        page.screenshot(path=str(shots / f"{name}-failed.png"))
    return False


if __name__ == "__main__":
    sys.exit(main())
