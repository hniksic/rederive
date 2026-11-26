"""File handling in a tab, which can't open or write files directly.

* **Save** is handled by `platform.web.WebStorage` in the command itself: the file is
  downloaded and also kept in page storage under its name.
* **A command that reads a file**, such as Transfer Load, opens the browser's file
  dialog through `choose`. The file picked arrives at `picked`, as bytes, goes into
  storage under its name, and the command goes on as if that name had been typed at its
  prompt. Where the dialog can't open, the command asks for the name as on the desktop.
* **A file added or dropped** on the page arrives at `arrived` and goes into storage,
  where a worksheet's `LOAD` finds it: the files a worksheet loads are named inside it,
  and no dialog asks for them. It isn't loaded on arrival, since Load replaces the
  worksheet and a dropped file shouldn't discard the current one. `unfound` tells the
  user which files a `LOAD` missed, and how to supply them.
* **A link** carries the worksheet in the URL fragment, which is never sent to a server.
  `linked` writes it there and `shared` reads it back on the next visit. It carries
  what Save would write, so a worksheet read from a Derive for Windows file goes as a
  math file, without the state that file saved.

Settings are handled here too. A tab has no home directory for a state file, so the
settings are written to storage whenever the tab is hidden and read back before the
screen is built. Because this happens unasked, the page's menu has `reset`, which
restores the defaults now and for later visits, and `welcome` names it in the page's
notice rather than on the program's message line.

All DOM calls go through `web/files.js`; this module decides which file goes where and
what to tell the user.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rederive import platform
from rederive.engine.client import ORDER_LIST
from rederive.model import dfw, worksheet
from rederive.model.session import Session
from rederive.model.settings import DEFAULTS
from rederive.model.undo import History
from rederive.platform.web import WebStorage, received
from rederive.ui.app import STATE_FILE

__all__ = ["Files", "SHARED", "STATE"]

#: Prefix of the worksheet field in the fragment, which identifies it and leaves any
#: other fragment content (an anchor, say) alone.
FIELD = "w="

#: File name for a worksheet from a link. Once stored, it's an ordinary file.
SHARED = "shared.mth"

#: Where settings are stored between visits: the same file name and format the desktop's
#: Transfer Save State prompt offers.
STATE = STATE_FILE

#: Maximum fragment length for a link. Browsers accept far more, but longer links tend to
#: get cut when pasted into messages, and such a worksheet can be saved instead.
LONGEST = 8000

#: Page notice when restored settings differ from the defaults. A mode switched on
#: months ago could otherwise look like a bug, so it names the menu item that undoes it.
RESTORED = "Settings restored from your last visit; File > Reset settings forgets them"

#: Shown after a settings reset. It mentions the next visit because the stored reset
#: isn't otherwise visible.
RESET = "Every setting is back at its default, and will be on the next visit"

#: How undo names a reset.
RESET_STEP = "Reset settings"

#: Shown when a link is made. It names the address bar, which has the link regardless of
#: whether copying to the clipboard succeeded.
LINKED = "This page's address now carries the worksheet; copy it to share it"

#: Shown when the worksheet is too long for a link.
TOO_LONG = (
    "This worksheet is {size} characters and too long for a link;"
    " Transfer Save writes it to a file instead"
)

#: Shown when a file is added or dropped, saying what finds it there.
ARRIVED = "{name} is in this page now, where a worksheet's LOAD finds it"

#: The same for a file the store has no room for, which only this visit has.
ARRIVED_UNREMEMBERED = (
    "{name} is in this page until it is closed, with no room to remember it longer;"
    " a worksheet's LOAD finds it until then"
)

#: Page notice for the files a `LOAD` could not find, with the remedy: the message line
#: has already said that they could not be found.
UNFOUND = (
    "{names} {verb} not in this page. Drop {pronoun} on the page or add {pronoun} with"
    " File > Add file..., then load again"
)

#: What a file dialog accepts besides the extensions a command reads, as the page's
#: own picker does: a worksheet is plain text whatever it is called.
ALSO_ACCEPTED = "text/plain"

#: Shown when the worksheet in the address can't be decoded, usually because the link
#: was truncated.
UNREADABLE_LINK = "The worksheet in this page's address could not be read"

#: Opening message for a worksheet loaded from a link, in the same form `__main__` uses
#: for a file given on the command line.
FROM_LINK = "{count} expression{s} from the link in this page's address"

#: Page notice when the browser keeps no site data (e.g. private browsing). Everything
#: works, but nothing outlives the tab, which the user couldn't otherwise tell.
FORGETFUL = "This browser keeps no site data: files and settings go with the tab"


class Files:
    """The page's file operations, bound to one session and one store.

    Created before the app, because settings and a linked worksheet are loaded into the
    session before the screen exists.
    """

    def __init__(self, page: Any, session: Session, storage: WebStorage) -> None:
        self._page = page
        self._session = session
        self._storage = storage
        #: Proxies held by JS; without them, callbacks would be collected while JS still
        #: refers to them.
        self._proxies: list[Any] = []
        #: Whether `settings` changed anything, for `opening` to report.
        self._restored = False
        #: The history a reset is a step of, for Ctrl-Z to undo; the app's, once there
        #: is one.
        self.history = History()
        #: The command waiting on the file dialog, as the callbacks `choose` was given,
        #: or None.
        self._choosing: (
            tuple[Callable[[str], str | None], Callable[[], None]] | None
        ) = None

    # -- before the screen -------------------------------------------------

    def settings(self) -> None:
        """Apply the stored settings, if any, before the app exists.

        This is silent: unlike on the desktop, the user didn't ask for it, so skipped
        lines aren't reported, and a missing file just means a first visit. It records
        whether anything changed for `welcome` to mention.
        """
        state = Path(STATE)
        if not self._storage.exists(state):
            return
        before = self._session.state_written()
        try:
            self._session.load_state(state)
        except OSError:
            return
        self._restored = self._session.state_written() != before

    def opening(self) -> str:
        """Return the opening message line, which reports a worksheet from the link.

        It's empty without one. What concerns the page rather than the worksheet goes
        to `welcome`.
        """
        return self.shared()

    def greeting(self) -> str:
        """Return the page's opening notice, or an empty string if there is none.

        It mentions restored non-default settings, or warns that the browser keeps no
        data.
        """
        if self._restored:
            return RESTORED
        return "" if self._storage.remembers else FORGETFUL

    def welcome(self) -> None:
        """Show `greeting` in the page's notice, once the screen is up."""
        greeting = self.greeting()
        if greeting:
            self._page.said(greeting)

    def shared(self) -> str:
        """Load the worksheet from the address and return a message about it.

        The text is stored as a file before being merged, so it behaves like any other
        worksheet: it has a name, the status line shows it, and Transfer Save works.
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
            report = self._session.merge(path)
        except OSError:
            return UNREADABLE_LINK
        count = len(self._session.entries)
        opening = FROM_LINK.format(count=count, s="" if count == 1 else "s")
        return opening if not report else f"{opening}; {report}"

    # -- while the screen is up --------------------------------------------

    def attend(self) -> None:
        """Connect the page's file controls and tab-closing events to this session.

        The proxies are kept for the lifetime of the page.
        """
        import js
        from pyodide.ffi import create_proxy, to_js

        table = {
            "arrived": self.arrived,
            "picked": self.picked,
            "dismissed": self.dismissed,
            "linked": self.linked,
            "reset": self.reset,
        }
        proxies = {name: create_proxy(call) for name, call in table.items()}
        leaving = create_proxy(self.remember)
        self._proxies = [*proxies.values(), leaving]
        self._page.attend(to_js(proxies, dict_converter=js.Object.fromEntries))
        # Both are needed: `pagehide` fires when a tab is closed or navigated away from,
        # but a phone backgrounding the browser may fire only `visibilitychange`.
        js.window.addEventListener("pagehide", leaving)
        js.document.addEventListener("visibilitychange", leaving)

    def arrived(self, name: str, data: Any) -> None:
        """Store a file the user added or dropped, and tell them its name."""
        self._said_arrived(name, self._kept(name, data))

    def choose(
        self,
        suffixes: tuple[str, ...],
        chosen: Callable[[str], str | None],
        dismissed: Callable[[], None],
    ) -> bool:
        """Open the file dialog for a command that reads a file, as the app's
        `chooser`, and return whether it opened.

        It doesn't without the user activation a browser requires of a dialog, which
        the key that chose the command normally grants, or in a browser that can't open
        one on request; the command then asks for the name instead.
        """
        if not self._page.choose(",".join([*suffixes, ALSO_ACCEPTED])):
            return False
        self._choosing = (chosen, dismissed)
        return True

    def picked(self, name: str, data: Any) -> None:
        """Give the name of the file picked in the dialog to the command, and store
        the file under the name the command reads it by.

        A command that no longer wants it leaves it stored under its own name, as an
        added file.
        """
        choosing, self._choosing = self._choosing, None
        named = None if choosing is None else choosing[0](name)
        if named is None:
            self._said_arrived(name, self._kept(name, data))
        else:
            self._kept(named, data)

    def dismissed(self) -> None:
        """Abandon the command whose dialog was dismissed."""
        choosing, self._choosing = self._choosing, None
        if choosing is not None:
            choosing[1]()

    def unfound(self, names: list[str]) -> None:
        """Tell the user how to supply the files a `LOAD` could not find, as the
        app's `unfound`.

        The notice is kept until the user dismisses it or another takes its place, as
        the one for a file supplied does: it is an instruction, followed while the user
        goes to find the files.
        """
        several = len(names) > 1
        listed = names[0] if not several else f"{', '.join(names[:-1])} and {names[-1]}"
        self._page.said(
            UNFOUND.format(
                names=listed,
                verb="are" if several else "is",
                pronoun="them" if several else "it",
            ),
            True,
        )

    def _kept(self, name: str, data: Any) -> bool:
        """Store a file the page handed over, and say whether it will be remembered.

        The page hands over the file's bytes. A Derive for Windows worksheet is kept as
        them; anything else is text, decoded as on the desktop, so a file in the DOS
        code page reads the same here.
        """
        raw = received(data)
        contents = raw if dfw.is_worksheet(raw) else worksheet.decoded(raw)
        return self._storage.keep(Path(name), contents)

    def _said_arrived(self, name: str, remembered: bool) -> None:
        """Tell the user a file is in the page, and what finds it there."""
        if remembered:
            self._page.said(ARRIVED.format(name=name))
        else:
            self._page.said(ARRIVED_UNREMEMBERED.format(name=name))

    def linked(self) -> None:
        """Put the worksheet in the address, and copy it to the clipboard if allowed.

        The message names the address bar, since the clipboard copy can be refused.
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

    def reset(self) -> None:
        """Reset the settings and `Manage Ordering` variable order to their defaults.

        The store is written immediately rather than by `remember`, so the reset survives
        a forced close. As with any settings change, existing entries keep their
        rendering. Like a command, it can be undone.
        """
        with self.history.aside(self._session, RESET_STEP):
            self._session.settings.apply(DEFAULTS)
            self._session.order = ORDER_LIST
        self._storage.keep(Path(STATE), self._session.state_written())
        self._page.said(RESET)

    def remember(self, _event: Any = None) -> None:
        """Store the settings as the tab is hidden or closed.

        The worksheet isn't saved implicitly: saving is the user's choice, and the next
        visitor shouldn't see someone else's work.
        """
        self._storage.keep(Path(STATE), self._session.state_written())

    def _refused(self, why: str) -> None:
        """Report that the clipboard refused the link."""
        self._page.said(f"{why} - the address bar carries the link either way")


def _fragment() -> str | None:
    """Return the worksheet field of the page's address, if present."""
    import js

    fragment = str(js.location.hash).lstrip("#")
    if not fragment.startswith(FIELD):
        return None
    return fragment[len(FIELD) :]


def _encoded(text: str) -> str:
    """Encode a worksheet as URL-safe base64.

    Padding is stripped, since `=` separates a fragment field from its value.
    """
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


def _decoded(fragment: str) -> str | None:
    """Decode a worksheet from a fragment, or return None if it can't be decoded.

    Any decoding failure returns None; links may have been mangled by chat, mail or
    line-wrapping editors on the way.
    """
    padded = fragment + "=" * (-len(fragment) % 4)
    try:
        return base64.urlsafe_b64decode(padded).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None
