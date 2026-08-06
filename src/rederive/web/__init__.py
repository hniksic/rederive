"""The browser half: the same program, with a page where the machine was.

Four things a tab has to supply that a terminal and an operating system supply
everywhere else, and nothing more. `driver` is the screen and the keyboard,
xterm.js standing in for the terminal; `engine` and `worker` are the two ends
of a Web Worker standing in for the child process the engine computes in;
`plots` is the plot windows a tab does not have yet; and `boot` is what the
page runs to put them together.

Nothing above this package knows any of it exists. The app is built with a
driver class and handed a session whose runner is a `WebEngine`, which are the
two seams the desktop program already has - `platform.use` is the third - and
what is left over is this package. That is the test of the port: it adds a
directory, and it does not add a branch anywhere else.

Every module here is written to import where there is no browser, because the
suite reads them on a desktop to hold them to the client-half rules. So `js`
and `pyodide.ffi` are reached for inside the functions that use them rather
than at the top of a module, and a module that cannot do that says why.
"""

from __future__ import annotations

__all__: list[str] = []
