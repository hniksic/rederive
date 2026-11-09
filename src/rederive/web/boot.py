"""Browser entry point: build the program from browser parts and start it.

`start` mirrors `__main__.main`, with browser substitutes: a Web Worker instead of the
engine child process, `platform.web` as the environment, an in-process plot session
drawing panes in the page instead of a plot host, and xterm.js as the screen via a
Textual driver. The rest of the program is unchanged and unaware of the difference.

Where `__main__` reads the command line, `web.files` reads the page address and storage:
a worksheet carried in the link and the saved settings, both applied before the app is
built.

`platform.use` and the environment variables come first: the platform must be in place
before anything asks for a clipboard or memory reading, and Rich reads `COLORTERM` when
Textual builds its Console. Without it Rich falls back to eight colors.

`get_driver_class` is overridden because `termios` and `tty` import fine under Pyodide,
so Textual would silently choose its Linux driver, which later fails when it can't start
a thread.

The app runs via `run_async` under `runPythonAsync` rather than `App.run`, since there's
one thread and the browser owns the event loop. The coroutine ends when the user quits.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rederive import platform
from rederive.model.session import Session
from rederive.platform.web import Web
from rederive.ui import app as ui
from rederive.ui.app import RederiveApp
from rederive.web.demos import Demos
from rederive.web.driver import BrowserDriver
from rederive.web.engine import WebEngine
from rederive.web.files import Files
from rederive.web.plots import WebPlots

__all__ = ["start"]

#: Tells Rich that xterm.js is a truecolor terminal. `force_terminal` already makes Rich
#: emit escapes; these select which ones, and a tab has no other source for them.
TERMINAL = {"TERM": "xterm-256color", "COLORTERM": "truecolor"}


class PageApp(RederiveApp):
    """The app running on the page's xterm.js terminal.

    `CSS_PATH` is absolute because Textual resolves it relative to the file defining
    the class, and this subclass lives elsewhere than the stylesheet.
    """

    CSS_PATH = str(Path(ui.__file__).parent / RederiveApp.CSS_PATH)
    #: The greeting's last line, without the demonstrations hint: the page has its own
    #: button for them.
    GREETING_LINE = ui.GREETING_HELP

    def __init__(
        self,
        session: Session,
        terminal: Any,
        plots: WebPlots,
        opening: str = "",
        keyboard: Callable[[bool], None] | None = None,
        welcome: Callable[[], None] | None = None,
    ) -> None:
        # Set before the base class init, which asks for the driver.
        self._terminal = terminal
        self._keyboard = keyboard
        #: Shows the page's own opening notice once the screen is drawn.
        self._welcome = welcome
        #: Last typing state sent to the page, so only changes are reported.
        self._typing: bool | None = None
        super().__init__(session, plotter=plots, opening=opening)

    def get_driver_class(self) -> Any:
        """Always the browser driver, with no fallback."""
        return BrowserDriver.on(self._terminal)

    def on_mount(self) -> None:
        # Textual runs the base class's `on_mount` too, so it isn't called here.
        if self._welcome is not None:
            self.call_after_refresh(self._welcome)

    # -- touch-screen keyboard ---------------------------------------------
    #
    # A phone shows its keyboard while the terminal has focus, but only the program
    # knows when it's waiting for text. So the page is told whenever that changes and
    # decides what to do with its keyboard.
    #
    # Changes are caught in two places: a redraw follows every mode change, and the
    # prompt line appears and disappears through focus changes, which Textual applies
    # shortly after the redraw.

    def refresh_screen(self) -> None:
        super().refresh_screen()
        self._tell_typing()

    def on_descendant_focus(self) -> None:
        self._tell_typing()

    def on_descendant_blur(self) -> None:
        self._tell_typing()

    def _tell_typing(self) -> None:
        typing = self.wants_typing
        if typing == self._typing or self._keyboard is None:
            return
        self._typing = typing
        self._keyboard(typing)


async def start(
    terminal: Any,
    spawn: Any,
    plots: Any,
    solids: Any,
    page: Any,
    demos: Any,
    keyboard: Any = None,
) -> None:
    """Run Rederive on this page until the user leaves it.

    The arguments are page objects Python can't create: the xterm.js terminal, a
    factory for the engine worker, the 2D and 3D plot modules (which share no drawing
    code), the file operations (download, picker and address), the demonstrations menu,
    and a callback that shows or hides a touch-screen keyboard.

    Saved settings and any worksheet in the address are loaded before the screen
    appears, so the user first sees what the link asked for.
    """
    _naming_tasks()
    os.environ.update(TERMINAL)
    platform.use(Web(page))
    engine = WebEngine(spawn)
    engine.start()
    session = Session(runner=engine)
    files = Files(page, session, platform.current().storage())
    files.settings()
    app = PageApp(
        session,
        terminal,
        WebPlots(plots, solids, engine),
        files.opening(),
        keyboard,
        files.welcome,
    )
    files.attend()
    # Kept alive while the program runs, since it holds the proxies the menu calls;
    # otherwise JS would call freed callbacks.
    demonstrations = Demos(demos, app, platform.current().storage())
    demonstrations.attend()
    # The demo-download command uses the same fetcher, since the app's default (`urlopen`
    # on a thread) doesn't work in a tab.
    app.fetcher = demonstrations.brought
    try:
        await app.run_async()
    finally:
        files.remember()
        engine.shutdown()


def _naming_tasks() -> None:
    """Put back the private asyncio helper Pyodide's event loop still calls.

    `WebLoop.create_task` copies CPython's `BaseEventLoop.create_task`, including the
    call to `asyncio.tasks._set_task_name`, which Python 3.14 (used by Pyodide 314) no
    longer has. The call is on the task-factory branch, and Textual installs a factory
    at the start of `run_async`, so without this every task raises AttributeError.

    The helper only called `Task.set_name`. The fix lives here because it's a bug in the
    browser runtime only.
    """
    if hasattr(asyncio.tasks, "_set_task_name"):
        return

    def named(task: Any, name: Any) -> None:
        if name is not None:
            task.set_name(name)

    asyncio.tasks._set_task_name = named
