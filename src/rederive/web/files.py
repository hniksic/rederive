"""The three file gestures a tab has, and what the program makes of each.

A page cannot open a file and cannot write one, so the browser's answer to
Transfer Load and Transfer Save is not one seam but three, and only one of them
is a command:

* **Save** is `platform.web.WebStorage`'s and happens where the command is: the
  file is downloaded and remembered under its name. Nothing here.
* **A file handed over** - picked with the page's button, or dropped on it -
  arrives at `arrived` and goes into the store, at which point the Load prompt
  can find it and complete it like any other name. It is not loaded on arrival:
  Load replaces the worksheet, and a file falling on the page is not a command
  to throw the current one away.
* **A link** is the address. `linked` puts the worksheet in the fragment, which
  is the one part of a URL that is never sent to a server, and `shared` takes
  it out again on the next visit.

The settings are here too, for want of anywhere better: a tab has no shell to
be started from and no home directory to keep a state file in, so it writes one
into the store every time it is hidden and reads it back before the screen is
built. That is what makes the web build remember how it was left.

Every DOM call is the page's, through the module `web/files.js`. What is here
is which file goes where and what is said about it.
"""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import Any

from rederive import platform
from rederive.model.session import Session
from rederive.platform.web import WebStorage
from rederive.ui.app import STATE_FILE

__all__ = ["Files", "SHARED", "STATE"]

#: What the fragment names its one field, so that an address carrying a
#: worksheet is legible as such and so that anything else in a fragment - an
#: anchor, another program's parameter - is left alone.
FIELD = "w="

#: The file a link's worksheet arrives as. It is a name like any other once it
#: is in the store: it can be loaded again, merged, saved over.
SHARED = "shared.mth"

#: Where the settings live between visits. The file the desktop's own Transfer
#: Save State prompt offers, taken from where that prompt takes it: it is the
#: same file in the same format, and a user who moves between the two should not
#: have to learn two names for it.
STATE = STATE_FILE

#: How long a fragment may be before a link is refused. Browsers themselves
#: take far more - Chromium and Firefox both carry tens of thousands of
#: characters in an address - but a link that cannot be pasted into a message
#: without being cut in half is not a link anyone can use, and a worksheet that
#: long has a Save waiting for it.
LONGEST = 8000

#: What the page says when a link has been made. The address bar is named
#: because that is where the link is, whatever the clipboard did.
LINKED = "This page's address now carries the worksheet; copy it to share it"

#: And when the worksheet will not fit in one. The number is the whole of the
#: explanation: there is no way to shorten it that keeps it a worksheet.
TOO_LONG = (
    "This worksheet is {size} characters and too long for a link;"
    " Transfer Save writes it to a file instead"
)

#: What the page says when a file has been picked or dropped. It names the
#: command as well as the file, since a file that arrived without being asked
#: for is one nobody knows what to do with.
ARRIVED = "{name} is in this page now; Transfer Load Derive reads it by that name"

#: And when a link's worksheet was on the address and could not be read. What
#: it usually means is a link that something cut in half on its way here.
UNREADABLE_LINK = "The worksheet in this page's address could not be read"

#: What the opening message says about a worksheet a link brought. The count is
#: the shape `__main__` reports a file in, and for the same reason: what the
#: screen is showing was not asked for by whoever is looking at it.
FROM_LINK = "{count} expression{s} from the link in this page's address"

#: And what it says in a tab that is keeping nothing - private browsing, or a
#: browser told to hold no site data. Everything works; nothing outlives the
#: tab, and that is the one thing about this environment a user cannot find out
#: by looking at it.
FORGETFUL = "This browser keeps no site data: files and settings go with the tab"


class Files:
    """The page's file gestures, bound to one session and one store.

    Built before the app is, because two of the four things it does happen
    before there is a screen: the settings are applied to the session the app
    is about to be built on, and a link's worksheet is read into it.
    """

    def __init__(self, page: Any, session: Session, storage: WebStorage) -> None:
        self._page = page
        self._session = session
        self._storage = storage
        #: The proxies JS holds. A callback given to JS without one is
        #: collected while JS still points at it.
        self._proxies: list[Any] = []

    # -- before the screen -------------------------------------------------

    def settings(self) -> None:
        """Apply the settings this page was left with, if it was left with any.

        Quietly, and before the app exists: what a state file could not take is
        a count the desktop reports on the message line at the moment the user
        asked for it, and nobody asked for this one. A file the store does not
        have is the first visit, which is nothing to report either.
        """
        state = Path(STATE)
        if not self._storage.exists(state):
            return
        try:
            self._session.load_state(state)
        except OSError:
            return

    def opening(self) -> str:
        """What the message line says before the user has asked for anything.

        A link's worksheet if the address carries one, and otherwise a word
        about a tab that will not remember anything, since that is the one thing
        about this environment that cannot be found out by looking at it. A
        first visit to an ordinary tab has nothing to say and says nothing.
        """
        said = self.shared()
        if said or self._storage.remembers:
            return said
        return FORGETFUL

    def shared(self) -> str:
        """Read the worksheet out of the address, and say what came of it.

        The text becomes a file in the store before it becomes expressions, so
        that a link is a worksheet in every way that matters afterwards - it has
        a name, the status line shows it, and Transfer Save writes it back out.
        """
        fragment = _fragment()
        if fragment is None:
            return ""
        text = _decoded(fragment)
        if text is None:
            return UNREADABLE_LINK
        path = Path(SHARED)
        self._storage.keep(path, text)
        try:
            skipped = self._session.merge(path)
        except OSError:
            return UNREADABLE_LINK
        count = len(self._session.entries)
        opening = FROM_LINK.format(count=count, s="" if count == 1 else "s")
        return opening if not skipped else f"{opening}, {skipped} of them unreadable"

    # -- while the screen is up --------------------------------------------

    def attend(self) -> None:
        """Wire the page's two controls, and the tab going away, to this session.

        The proxies are kept for as long as the page is: a callback JS holds
        without one behind it is a call into memory the collector has taken.
        """
        import js
        from pyodide.ffi import create_proxy, to_js

        table = {"arrived": self.arrived, "linked": self.linked}
        proxies = {name: create_proxy(call) for name, call in table.items()}
        leaving = create_proxy(self.remember)
        self._proxies = [*proxies.values(), leaving]
        self._page.attend(to_js(proxies, dict_converter=js.Object.fromEntries))
        # Both, because neither is enough on its own: `pagehide` is what a tab
        # being closed or navigated away from fires, and a phone that puts the
        # browser in the background may only ever fire the other.
        js.window.addEventListener("pagehide", leaving)
        js.document.addEventListener("visibilitychange", leaving)

    def arrived(self, name: str, text: str) -> None:
        """A file the user handed over: keep it, and say what it is called now."""
        self._storage.keep(Path(name), text)
        self._page.said(ARRIVED.format(name=name))

    def linked(self) -> None:
        """Put the worksheet in the address, and on the clipboard if it is allowed.

        The address is the link; the clipboard is a convenience on top of it,
        and it is the one part of this that can be refused - which is why the
        sentence names the address rather than the clipboard.
        """
        import js

        text = self._session.written()
        fragment = FIELD + _encoded(text)
        if len(fragment) > LONGEST:
            self._page.said(TOO_LONG.format(size=len(text)))
            return
        self._page.address(fragment)
        self._page.said(LINKED)
        platform.current().copy(str(js.location.href), self._refused)

    def remember(self, _event: Any = None) -> None:
        """Write the settings into the store, the tab being on its way out.

        Only the settings. The worksheet is not saved behind the user's back -
        Transfer Save is a command and stays one - and a tab that quietly kept
        every session would hand the next visitor somebody else's screen.
        """
        self._storage.keep(Path(STATE), self._session.state_written())

    def _refused(self, why: str) -> None:
        """What the page says when the clipboard would not take the link."""
        self._page.said(f"{why} - the address bar carries the link either way")


def _fragment() -> str | None:
    """The worksheet field of this page's address, if it carries one."""
    import js

    fragment = str(js.location.hash).lstrip("#")
    if not fragment.startswith(FIELD):
        return None
    return fragment[len(FIELD) :]


def _encoded(text: str) -> str:
    """A worksheet as one field of a URL: base64 in the alphabet a URL allows.

    Unpadded, since `=` in a fragment is what separates a field from its value
    and a reader that has to be told to ignore it is a reader waiting to be got
    wrong.
    """
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


def _decoded(fragment: str) -> str | None:
    """The worksheet a fragment carries, or nothing where it carries no such thing.

    Three ways to fail and one answer to all of them: what arrives here came off
    the address bar of a browser and may have been through a chat window, a mail
    client and a line-wrapping text editor on the way.
    """
    padded = fragment + "=" * (-len(fragment) % 4)
    try:
        return base64.urlsafe_b64decode(padded).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None
