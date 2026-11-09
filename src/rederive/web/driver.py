"""Textual driver for a page: xterm.js for the screen, `onData` for the keys.

Output goes to `term.write`; input arrives as strings on `term.onData` and is parsed by
Textual's own `XTermParser`, as on a desktop. The driver also handles the alternate
screen, mouse modes, size and timeouts.

Instead of `SIGWINCH`, xterm.js reports size changes through `onResize`. The driver sends
Textual a resize event and sets `COLUMNS` and `LINES`, which is where Rich looks, since
there's no ioctl to query.

`XTermParser` can't tell a lone Esc from the start of an escape sequence until a moment
passes with no further input. On a desktop the input thread polls it between reads; here
a ticker task does. Without it, Esc, which aborts a computation, would never arrive.

`pyodide.ffi` is imported inside the method that needs it (callbacks passed to JS must be
proxied so they aren't collected), so the module still imports in the desktop test suite.
"""

from __future__ import annotations

import asyncio
import os
from functools import partial
from typing import Any, Callable

from textual import events
from textual._xterm_parser import XTermParser
from textual.driver import Driver
from textual.geometry import Size

__all__ = ["BrowserDriver"]

#: How often to check the parser for timed-out input: fast enough for Esc to feel
#: immediate, cheap enough for an idle tab.
TICK = 0.05

#: Mouse modes to enable, in terminal order. 1006 (SGR) matters most: it's the only
#: encoding that reports which button was released, which the right-click menus need.
MOUSE_ON = ("\x1b[?1000h", "\x1b[?1003h", "\x1b[?1006h")
MOUSE_OFF = ("\x1b[?1000l", "\x1b[?1003l", "\x1b[?1006l")


class BrowserDriver(Driver):
    """A Textual driver backed by an xterm.js terminal in the page.

    Textual constructs the driver itself, so `on` binds the page's terminal to the
    class in advance.
    """

    def __init__(
        self,
        app: Any,
        *,
        terminal: Any,
        debug: bool = False,
        mouse: bool = True,
        size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(app, debug=debug, mouse=mouse, size=size)
        self._terminal = terminal
        self._parser = XTermParser(debug)
        #: Proxies held by JS and the subscriptions it returned, released when the app
        #: stops. Dropping a proxy JS still holds leaves it calling freed memory.
        self._proxies: list[Any] = []
        self._subscriptions: list[Any] = []
        self._ticker: asyncio.Task[None] | None = None

    @classmethod
    def on(cls, terminal: Any) -> Callable[..., "BrowserDriver"]:
        """Return this driver class bound to `terminal`, for `driver_class`."""
        return partial(cls, terminal=terminal)

    @property
    def is_web(self) -> bool:
        return True

    @property
    def can_suspend(self) -> bool:
        """False: Ctrl-Z returns to a shell, and a tab has none."""
        return False

    # -- the screen --------------------------------------------------------

    def write(self, data: str) -> None:
        """Write directly; xterm.js does its own buffering and parsing."""
        self._terminal.write(data)

    # -- application mode --------------------------------------------------

    def start_application_mode(self) -> None:
        """Take over the screen and start listening for input."""
        self._listen()
        self._sized(self._terminal.cols, self._terminal.rows)
        self.write("\x1b[?1049h")  # the alternate screen
        self._enable_mouse()
        self.write("\x1b[?25l")  # hide the cursor
        self.write("\x1b[?2004h")  # bracketed paste
        self._ticker = asyncio.get_running_loop().create_task(self._ticking())

    def disable_input(self) -> None:
        """Stop listening and release the page's callbacks and subscriptions."""
        if self._ticker is not None:
            self._ticker.cancel()
            self._ticker = None
        for subscription in self._subscriptions:
            subscription.dispose()
        self._subscriptions.clear()
        for proxy in self._proxies:
            proxy.destroy()
        self._proxies.clear()
        self._disable_mouse()

    def stop_application_mode(self) -> None:
        """Restore the screen so whatever the page shows next is legible."""
        self.disable_input()
        self.write("\x1b[?2004l")
        self.write("\x1b[?1049l")
        self.write("\x1b[?25h")

    # -- input -------------------------------------------------------------

    def _listen(self) -> None:
        """Subscribe to the terminal's key and resize events."""
        from pyodide.ffi import create_proxy

        typed = create_proxy(self._typed)
        resized = create_proxy(self._resized)
        self._proxies = [typed, resized]
        self._subscriptions = [
            self._terminal.onData(typed),
            self._terminal.onResize(resized),
        ]

    def _typed(self, data: str) -> None:
        """Parse a string of terminal input into events."""
        try:
            for event in self._parser.feed(data):
                self.process_message(event)
        except Exception:
            self._panicked()

    def _resized(self, size: Any) -> None:
        """Handle a terminal resize (the counterpart of `SIGWINCH`)."""
        self._sized(size.cols, size.rows)

    def _sized(self, columns: int, rows: int) -> None:
        """Tell the app its new size, and set it in the environment for Rich.

        Textual gets the event, while Rich reads the environment. If they disagree, the
        screen is drawn to one size and cleared to another.
        """
        os.environ["COLUMNS"] = str(columns)
        os.environ["LINES"] = str(rows)
        size = Size(columns, rows)
        self.send_message(events.Resize(size, size))

    async def _ticking(self) -> None:
        """Periodically flush timed-out input from the parser.

        This is how a lone Esc is recognized as Esc rather than the start of a sequence.
        The task runs for as long as application mode does.
        """
        while True:
            await asyncio.sleep(TICK)
            for event in self._parser.tick():
                self.process_message(event)

    def _panicked(self) -> None:
        """Show a parse failure on the app's panic screen.

        An exception escaping a JS callback would otherwise go nowhere the user would see.
        """
        import rich.traceback

        self._app.call_later(self._app.panic, rich.traceback.Traceback())

    # -- the mouse ---------------------------------------------------------

    def _enable_mouse(self) -> None:
        if not self._mouse:
            return
        for mode in MOUSE_ON:
            self.write(mode)

    def _disable_mouse(self) -> None:
        if not self._mouse:
            return
        for mode in MOUSE_OFF:
            self.write(mode)
