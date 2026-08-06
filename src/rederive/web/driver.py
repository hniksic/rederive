"""The Textual driver a page is: xterm.js for the screen, `onData` for the keys.

The abstract `Driver` is four methods, and a browser can answer all four. Bytes
that would go to a terminal go to `term.write` instead; bytes that would come
back off a tty arrive as strings on `term.onData`, and Textual's own
`XTermParser` turns them into events exactly as it does on a desktop - it
parses a stream of characters and has never cared where the stream came from.
What is left is the housekeeping a terminal does for a program without being
asked: the alternate screen, the mouse modes, the size, and the timeouts.

Two of those are worth naming.

The size is not a signal here. `SIGWINCH` is what tells a program on a desktop
that its window changed, and xterm.js has `onResize` in its place; what has to
happen either way is that Rich and Textual find the new figures where they look
for them, which for Rich is `COLUMNS` and `LINES` in the environment. A tab
sets them, since it has no ioctl to be asked.

The timeouts are the reason for the ticker. `XTermParser` cannot tell a lone Esc
from the beginning of an escape sequence until a moment has passed with nothing
following it, so it holds the key and waits to be asked whether the moment is
up. On a desktop the input thread asks every tenth of a second between reads;
here nothing reads, so a task asks instead. Without it the Esc that aborts a
computation never arrives, which is the one key this program cannot do without.

`pyodide.ffi` is imported inside the method that needs it. A callback handed to
JS has to be wrapped in a proxy or it is collected while JS still holds it, but
this module is also read where there is no `pyodide` to import - the suite does
exactly that - and a module that will not import there is a module the rules
about what the app process may hold cannot be applied to.
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

#: How often the parser is asked whether what it is holding has timed out.
#: Twenty times a second: fast enough that Esc feels immediate, and cheap
#: enough that a tab doing nothing is doing almost nothing.
TICK = 0.05

#: The mouse modes the app asks for, in the order a terminal wants them. 1006
#: is the one that matters: SGR is the only encoding that reports which button
#: was released, and the program's right-click menus turn on knowing.
MOUSE_ON = ("\x1b[?1000h", "\x1b[?1003h", "\x1b[?1006h")
MOUSE_OFF = ("\x1b[?1000l", "\x1b[?1003l", "\x1b[?1006l")


class BrowserDriver(Driver):
    """Textual's four calls, answered by a terminal drawn in a page.

    Held together by one terminal object, which the page owns and hands over
    before the app is built. `on` is how it arrives: Textual constructs the
    driver itself, so what it is given is this class with the terminal already
    bound to it rather than an instance.
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
        #: The proxies JS holds and the subscriptions it gave back, so that
        #: both can be given up when the app stops. A proxy that is dropped
        #: while JS still has it is a callback into freed memory.
        self._proxies: list[Any] = []
        self._subscriptions: list[Any] = []
        self._ticker: asyncio.Task[None] | None = None

    @classmethod
    def on(cls, terminal: Any) -> Callable[..., "BrowserDriver"]:
        """This driver bound to one terminal, in the form `driver_class` takes."""
        return partial(cls, terminal=terminal)

    @property
    def is_web(self) -> bool:
        return True

    @property
    def can_suspend(self) -> bool:
        """No. Ctrl-Z hands a program back to a shell, and a tab has none."""
        return False

    # -- the screen --------------------------------------------------------

    def write(self, data: str) -> None:
        """Straight through: xterm.js does its own buffering and its own parsing."""
        self._terminal.write(data)

    # -- application mode --------------------------------------------------

    def start_application_mode(self) -> None:
        """Take the screen, and start listening for what comes back off it."""
        self._listen()
        self._sized(self._terminal.cols, self._terminal.rows)
        self.write("\x1b[?1049h")  # the alternate screen
        self._enable_mouse()
        self.write("\x1b[?25l")  # hide the cursor
        self.write("\x1b[?2004h")  # bracketed paste, which is how a paste arrives
        self._ticker = asyncio.get_running_loop().create_task(self._ticking())

    def disable_input(self) -> None:
        """Stop listening, and give the page back everything it lent."""
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
        """Give the screen back, so that whatever the page shows next is legible."""
        self.disable_input()
        self.write("\x1b[?2004l")
        self.write("\x1b[?1049l")
        self.write("\x1b[?25h")

    # -- what comes back off the terminal ----------------------------------

    def _listen(self) -> None:
        """Subscribe to the two things a terminal has to say: keys and a size."""
        from pyodide.ffi import create_proxy

        typed = create_proxy(self._typed)
        resized = create_proxy(self._resized)
        self._proxies = [typed, resized]
        self._subscriptions = [
            self._terminal.onData(typed),
            self._terminal.onResize(resized),
        ]

    def _typed(self, data: str) -> None:
        """One string off the terminal, parsed into whatever events it holds."""
        try:
            for event in self._parser.feed(data):
                self.process_message(event)
        except Exception:
            self._panicked()

    def _resized(self, size: Any) -> None:
        """The terminal changed shape, which on a desktop would be a signal."""
        self._sized(size.cols, size.rows)

    def _sized(self, columns: int, rows: int) -> None:
        """Tell the app its new size, and leave it where Rich looks for it too.

        Textual hears the event; Rich reads the environment, having no terminal
        to ask. Both have to be right or the screen is drawn to one size and
        cleared to another.
        """
        os.environ["COLUMNS"] = str(columns)
        os.environ["LINES"] = str(rows)
        size = Size(columns, rows)
        self.send_message(events.Resize(size, size))

    async def _ticking(self) -> None:
        """Ask the parser whether the sequence it is holding has timed out.

        Which is how a lone Esc becomes an Esc rather than the start of
        something longer. The task lives as long as application mode does.
        """
        while True:
            await asyncio.sleep(TICK)
            for event in self._parser.tick():
                self.process_message(event)

    def _panicked(self) -> None:
        """Put a broken parse on the app's panic screen, where it can be read.

        A callback JS holds is the end of the line for an exception: nothing
        above it prints anything anywhere a user would look.
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
