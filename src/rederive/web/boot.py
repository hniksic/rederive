"""What the page runs: build the program on a browser's four answers, and start.

`start` is the whole of the browser's entry point, and it is `__main__.main` with
each of its three machines swapped for the thing a tab has instead. The engine
worker is a Web Worker rather than a child process; the environment is
`platform.web` rather than a desktop; the plot host is a plot session in this
very interpreter, opening panes in the page rather than windows on a desktop;
and the screen is xterm.js, reached through a Textual driver. Nothing else about
the program changes, and nothing else in the program knows any of it.

Two of the swaps have to happen before anything is built, which is why they are
the first lines. `platform.use` has to be in force before the first thing asks
for a clipboard or a memory reading, and the environment variables have to be
set before Textual builds its Console: Rich decides how many colours it may use
by reading `COLORTERM`, and a page that leaves it unset gets eight of them and a
display that looks nothing like the program.

The driver is the third, and it is not a swap so much as the removal of an
alternative. `termios` and `tty` import cleanly under Pyodide, so Textual's
Linux driver is a working import here and its automatic choice would be taken in
silence, to break much later on a thread it cannot start. `get_driver_class` is
where that choice is made, so it is overridden rather than overruled: the page's
app has one driver and no fallback behind it.

The app is never run with `App.run`. There is one thread and the loop is the
browser's, so the entry has to be `run_async` under `runPythonAsync` - and this
coroutine ends when the user leaves the program, which is when the page has
nothing left to show.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from rederive import platform
from rederive.model.session import Session
from rederive.platform.web import Web
from rederive.ui import app as ui
from rederive.ui.app import RederiveApp
from rederive.web.driver import BrowserDriver
from rederive.web.engine import WebEngine
from rederive.web.plots import WebPlots

__all__ = ["start"]

#: What the page tells Rich about itself. `force_terminal` already makes Rich
#: write escapes; what these decide is which escapes - a truecolor terminal, as
#: xterm.js is - and neither has anywhere else to come from in a tab.
TERMINAL = {"TERM": "xterm-256color", "COLORTERM": "truecolor"}


class PageApp(RederiveApp):
    """The program, on the terminal this page drew.

    The stylesheet is named absolutely because Textual resolves `CSS_PATH`
    against the file the class was defined in, and this class is not defined in
    that file. One stylesheet still, and one place that reads it.
    """

    CSS_PATH = str(Path(ui.__file__).parent / RederiveApp.CSS_PATH)

    def __init__(self, session: Session, terminal: Any, plots: WebPlots) -> None:
        # Before the base class, which asks for the driver as it starts.
        self._terminal = terminal
        super().__init__(session, plotter=plots)

    def get_driver_class(self) -> Any:
        """This page's terminal, and nothing behind it to fall back to."""
        return BrowserDriver.on(self._terminal)


async def start(terminal: Any, spawn: Any, plots: Any, solids: Any) -> None:
    """Run Rederive on this page until the user leaves it.

    Four things the page owns and this side cannot make: the xterm.js the
    program draws on, what makes an engine worker, and the two modules that
    open a plot pane - one for a flat picture and one for a solid, which share
    no drawing at all. All of them are DOM elements, script URLs and canvases,
    and none of them is anything Python could have built.
    """
    _naming_tasks()
    os.environ.update(TERMINAL)
    platform.use(Web())
    engine = WebEngine(spawn)
    engine.start()
    app = PageApp(Session(runner=engine), terminal, WebPlots(plots, solids, engine))
    try:
        await app.run_async()
    finally:
        engine.shutdown()


def _naming_tasks() -> None:
    """Put back the private asyncio helper Pyodide's event loop still calls.

    `WebLoop.create_task` is a copy of CPython's `BaseEventLoop.create_task`,
    call to `asyncio.tasks._set_task_name` and all - and Python 3.14, which
    Pyodide 314 carries, no longer has that function. The call is on the branch
    a loop with a task factory takes, and Textual installs one on the first
    line of `run_async`, so without this every task the app creates raises an
    AttributeError from inside the loop.

    Naming a task is the whole of what the helper did, and `Task.set_name` is
    what it did it with. The repair belongs here rather than anywhere the
    program is read on a desktop: it is a browser runtime's bug and nothing
    else's.
    """
    if hasattr(asyncio.tasks, "_set_task_name"):
        return

    def named(task: Any, name: Any) -> None:
        if name is not None:
            task.set_name(name)

    asyncio.tasks._set_task_name = named
