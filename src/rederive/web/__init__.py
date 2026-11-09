"""The browser port: what a tab must supply in place of a terminal and an OS.

`driver` provides the screen and keyboard via xterm.js; `engine` and `worker` are the two
ends of a Web Worker that replaces the engine's child process; `plots` provides plot
windows as page panes; and `boot` wires them together when the page loads.

No code outside this package knows about it. The app takes a driver class and a session
whose runner is a `WebEngine`, plus `platform.use`, all seams the desktop program already
has. The port adds this directory and no branches elsewhere.

Every module here must import without a browser, because the test suite checks them on
the desktop against the client-half rules. So `js` and `pyodide.ffi` are imported inside
the functions that use them, and a module that can't do that says why.
"""

from __future__ import annotations

__all__: list[str] = []
