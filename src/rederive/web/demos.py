"""The page's demonstrations menu.

The catalogue lives in `rederive.model.demos` (nine names, titles and download URLs), in
Python so a desktop menu could share it. This module hands the catalogue to the page to
build a menu, and runs the files the page downloads.

The page sends back a file name, plus the bytes if it had to download them. A demo
already in storage is requested by name alone (see `kept`), which makes a second viewing
instant. The demo's kind, which determines what a step does with its expression, is
looked up in the catalogue rather than trusted from the page, and unknown names are
rejected.

The file is then stored under its own name, like a picked file, and the app is told to
demonstrate it. Afterwards it's an ordinary file in the page, so `Transfer Demo ALGEBRA`
runs it again.

`Transfer Demo` offers the same nine demos from the menu band. A tab has no sockets, so
the app is given `brought` as its fetcher, which downloads through the page.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rederive.model import demos as catalogue
from rederive.model import worksheet
from rederive.platform.web import WebStorage
from rederive.ui.app import RederiveApp

__all__ = ["Demos"]

#: Shown when a demo script starts, naming the command that runs it again.
RUNNING = "{name} is in this page now; Transfer Demo runs it again by that name"

#: Shown when a gallery starts. A gallery has a worksheet extension, so Demo wouldn't
#: find it and a different command is named.
DRAWING = "{name} is in this page now; Transfer Load Derive reads it by that name"

#: Shown when a demo is chosen before the app is running. The window is brief, since the
#: button appears before the program does, but a silently ignored click would look like
#: a bug.
TOO_SOON = "Rederive is still starting; try the demonstration again in a moment"

#: Shown when the page sends a name not in the catalogue. Users can't trigger this; it
#: guards against running an unlisted file.
UNKNOWN = "{name} is not one of the original's demonstrations"

#: Fallback for a failed download with no error message. Browsers normally give one.
NOT_FETCHED = "the browser refused the download"


class Demos:
    """The page's demonstration menu, bound to one app and one store."""

    def __init__(self, page: Any, app: RederiveApp, storage: WebStorage) -> None:
        self._page = page
        self._app = app
        self._storage = storage
        #: Proxies held by JS; without them, callbacks would be collected while JS still
        #: refers to them.
        self._proxies: list[Any] = []

    def attend(self) -> None:
        """Give the page the catalogue, the image URL and the choice handlers.

        The proxies are kept for the lifetime of the page.
        """
        import js
        from pyodide.ffi import create_proxy, to_js

        table = {"chose": self.chose, "keep": self.keep, "kept": self.kept}
        proxies = {name: create_proxy(call) for name, call in table.items()}
        self._proxies = list(proxies.values())
        given = {**proxies, "demos": _offered(), "image": catalogue.IMAGE}
        self._page.attend(to_js(given, dict_converter=js.Object.fromEntries))

    def kept(self, file: str) -> bool:
        """Return whether this demonstration is already stored in the page.

        This lets a repeat viewing skip the download. Storage outlives the tab, so a
        demo downloaded last week is still available.
        """
        demo = catalogue.named(file)
        return demo is not None and self._storage.exists(Path(demo.file))

    def keep(self, file: str, data: Any) -> None:
        """Store a demonstration without running it.

        Downloading the diskette yields all nine files at once, so the page keeps the
        eight it didn't ask for rather than fetching them again later.
        """
        demo = catalogue.named(file)
        if demo is None:
            return
        self._storage.keep(Path(demo.file), worksheet.decoded(_bytes(data)))

    def chose(self, file: str, data: Any = None) -> None:
        """Run this demonstration, storing any bytes the page downloaded.

        The file is stored before running so that its name works with the Transfer
        commands, including resuming a demo suspended with Esc. No bytes arrive for
        a file that was already stored.
        """
        demo = catalogue.named(file)
        if demo is None:
            self._page.said(UNKNOWN.format(name=file))
            return
        if not self._app.is_running:
            self._page.said(TOO_SOON)
            return
        # Test truthiness rather than `is None`: an omitted argument arrives as `None`,
        # but an explicit `null` arrives as a JavaScript null, which is falsy but not
        # `None`.
        if data:
            self._storage.keep(Path(demo.file), worksheet.decoded(_bytes(data)))
        said = DRAWING if demo.plotting else RUNNING
        self._page.said(said.format(name=demo.file))
        # Schedule on the loop: this runs in a DOM callback, and a demonstration started
        # from here would never be drawn.
        self._app.call_next(self._app.demonstrate, demo.file, demo.plotting)

    async def brought(self, demo: catalogue.Demo) -> bytes:
        """Download this demonstration through the page and store it by name.

        This is `Transfer Demo`'s fetcher in a tab, which can only download through the
        page. Unlike on the desktop, the file goes into page storage rather than being
        saved, since saving in a tab means a download to the user's disk. The other
        eight files from the diskette were already stored by the page.
        """
        try:
            raw = _bytes(await self._page.brought(demo.file))
        except Exception as trouble:
            raise OSError(_sentence(trouble)) from None
        self._storage.keep(Path(demo.file), worksheet.decoded(raw))
        return raw


def _offered() -> list[dict[str, Any]]:
    """Return the catalogue as plain dictionaries for the page's menu.

    Each entry includes its download URLs, since Python decides where files come from
    but only the page can fetch them.
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


def _sentence(trouble: Exception) -> str:
    """Reduce a browser rejection to one line for the message line.

    The exception text is the error name, message and possibly a stack. Keep only the
    first line and drop a generic `Error: ` prefix.
    """
    said = str(trouble).strip().splitlines()
    return said[0].removeprefix("Error: ") if said else NOT_FETCHED


def _bytes(data: Any) -> bytes:
    """Convert file data to bytes.

    The browser passes a `Uint8Array`, which is copied out of JS memory; tests pass
    bytes directly.
    """
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    return bytes(data.to_py())
