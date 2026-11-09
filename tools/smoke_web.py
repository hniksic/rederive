#!/usr/bin/env python3
"""Boot the web build in a real browser and ask it one question.

    uv run --with playwright python tools/smoke_web.py
    ... --directory build/web --browser chromium --shots build/shots

It opens the page, waits for the menu, authors `(x + 1)^2 - (x - 1)^2`, simplifies
it, and reads `4·x` off the screen. That exercises the whole build (two Pyodide
instances, the Textual driver, xterm.js, the engine worker and the pickles between
them), and a failure reports which step it stopped at.

The screen is read from xterm.js's buffer rather than from pixels, giving the exact
characters the program wrote.

**Nothing here is a timing.** Playwright's Firefox runs WebAssembly about eight times
slower than native, so the waits are deliberately generous and say nothing about the
build's speed.

The page is served by `http.server`; the build needs no other host at run time.
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

#: The expression authored and the expected answer. This is the README's example, and
#: its answer fits on one line, so the test doesn't depend on 2D rendering.
AUTHORED = "(x+1)^2-(x-1)^2"
ANSWER = "4·x"

#: A word from the opening menu, whose appearance means the program is up.
MENU = "Author"

#: Time limits per step. Both browsers get Firefox's budget, since it is much slower at
#: WebAssembly under Playwright.
BOOT_SECONDS = 180
ANSWER_SECONDS = 120

#: JS that reads every row of the terminal buffer as text.
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
        raise SystemExit(f"{directory}: not a web build - run tools/build_web.py")
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
    parser = argparse.ArgumentParser(description="Smoke-test the web build.")
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
    """Start a static server on a free port in a background thread."""
    handler = partial(_Quiet, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


class _Quiet(http.server.SimpleHTTPRequestHandler):
    """A request handler with logging disabled, to avoid hundreds of log lines."""

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

    # Author the expression, then Simplify it.
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
    """Poll the terminal buffer until it contains `wanted` or time runs out."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if wanted in page.evaluate(SCREEN):
            return True
        page.wait_for_timeout(250)
    return False


def _failed(name, what, page, shots: Path | None, troubles: list[str]) -> bool:
    """Report a failure with the screen contents, page errors and a screenshot."""
    print(f"{name}: {what}", file=sys.stderr)
    print(page.evaluate(SCREEN), file=sys.stderr)
    for trouble in troubles:
        print(f"{name}: page said: {trouble}", file=sys.stderr)
    if shots is not None:
        page.screenshot(path=str(shots / f"{name}-failed.png"))
    return False


if __name__ == "__main__":
    sys.exit(main())
