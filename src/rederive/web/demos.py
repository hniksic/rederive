"""The demonstration menu of the page: what is on it, and what a choice does.

The list itself is `rederive.model.demos`, which is nine names, nine titles and
the addresses they can be downloaded from - Python rather than page furniture,
because a desktop wanting the same menu wants the same nine files. What is here
is the two halves of the seam: the catalogue is handed to the page to build a
menu out of, and a file the page has downloaded comes back to be run.

Only the file name comes back, and the bytes where there were any to bring: a
demonstration the store already has is asked for by name alone, which is what
`kept` is for and what makes a second viewing of one instant. Which kind of
demonstration it is, and so what a step does with its expression, is read off
the catalogue here rather than believed from the page: the page is given the
list and hands one of them back, and a name that is not on it is not a
demonstration.

What happens after that is the program's ordinary machinery. The file goes into
the store under its own name, exactly as a picked file does, and then the app is
told to demonstrate it. That it came off the network is not something the rest
of the program has to know, and afterwards the file is in the page like any
other - `Transfer Demo ALGEBRA` runs it again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rederive.model import demos as catalogue
from rederive.model import worksheet
from rederive.platform.web import WebStorage
from rederive.ui.app import RederiveApp

__all__ = ["Demos"]

#: What the page says once a script is running. It names the command that runs
#: it again, since the file is in the page now and the menu is where a second
#: viewing comes from.
RUNNING = "{name} is in this page now; Transfer Demo runs it again by that name"

#: And once a gallery is. The command named is the other one: a gallery is a
#: worksheet by extension and Demo would not find it.
DRAWING = "{name} is in this page now; Transfer Load Derive reads it by that name"

#: What it says when a demonstration is asked for before there is a screen to
#: run it on. The window is a fraction of a second wide - the button is there
#: from the first frame and the program is not - but a click that vanished into
#: nothing would be the page's fault and read as the program's.
TOO_SOON = "Rederive is still starting; try the demonstration again in a moment"

#: And when a name arrives that is not on the list the page was given. Nothing
#: a user can do reaches this; what it is for is saying so out loud rather than
#: running a file nobody named.
UNKNOWN = "{name} is not one of the original's demonstrations"


class Demos:
    """The page's demonstration menu, bound to one app and one store."""

    def __init__(self, page: Any, app: RederiveApp, storage: WebStorage) -> None:
        self._page = page
        self._app = app
        self._storage = storage
        #: The proxies JS holds. A callback given to JS without one is
        #: collected while JS still points at it.
        self._proxies: list[Any] = []

    def attend(self) -> None:
        """Hand the page the menu, the relays and what a choice on it does.

        All at once, since none of them is any use alone: a menu with nothing
        behind it offers what cannot be run, and a handler with no menu in
        front of it is never reached. The proxy is kept for as long as the page
        is, as every proxy handed to JS is.
        """
        import js
        from pyodide.ffi import create_proxy, to_js

        table = {"chose": self.chose, "kept": self.kept}
        proxies = {name: create_proxy(call) for name, call in table.items()}
        self._proxies = list(proxies.values())
        given = {**proxies, "demos": _offered()}
        self._page.attend(to_js(given, dict_converter=js.Object.fromEntries))

    def kept(self, file: str) -> bool:
        """Whether this demonstration is in the page already.

        What spares the second viewing its download. The store outlives the
        tab, so a demonstration run last week is on the screen as fast as the
        program can author it - and the file it runs from is the one that was
        downloaded then, which is the same file it would download now.
        """
        demo = catalogue.named(file)
        return demo is not None and self._storage.exists(Path(demo.file))

    def chose(self, file: str, data: Any = None) -> None:
        """Run this demonstration, keeping the bytes the page brought with it.

        The file lands in the store before it is run, because that is what makes
        it a file: the name on the status line is a name the Transfer commands
        can be given afterwards, and a demonstration suspended with Esc is
        picked up by naming it. Nothing arrives with a file the page did not
        have to download, since keeping it again would only write what is
        already there.
        """
        demo = catalogue.named(file)
        if demo is None:
            self._page.said(UNKNOWN.format(name=file))
            return
        if not self._app.is_running:
            self._page.said(TOO_SOON)
            return
        if data is not None:
            self._storage.keep(Path(demo.file), worksheet.decoded(_bytes(data)))
        said = DRAWING if demo.plotting else RUNNING
        self._page.said(said.format(name=demo.file))
        self._app.demonstrate(demo.file, demo.plotting)


def _offered() -> list[dict[str, Any]]:
    """The catalogue in the shape the page builds a menu out of.

    Plain dictionaries, because what crosses this seam is data and the page has
    no use for a class. The addresses travel with each entry: which host has the
    file is not the page's decision, and the page is the only side that can ask.
    """
    return [
        {
            "file": demo.file,
            "title": demo.title,
            "group": demo.group,
            "urls": list(demo.urls),
        }
        for demo in catalogue.DEMOS
    ]


def _bytes(data: Any) -> bytes:
    """The file, however the side that read it holds its bytes.

    A browser hands over a `Uint8Array`, which is copied out of the page's
    memory the way every other array of bytes crossing this seam is; everywhere
    else - a test above all - bytes are bytes.
    """
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    return bytes(data.to_py())
