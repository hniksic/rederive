"""Undo and redo: the state as each command found it, in one history for all windows.

Like the session, this is pure Python, with no Textual and no key names. A step is a
`Snapshot` of the session one command changed, from before that command: the entries,
the definitions and declarations, the counters, the unremove buffer, the selection, and
the settings. Taking one is cheap, since entries are immutable and shared, and the
tables are copied shallowly. Undoing puts the snapshot back whole rather than reversing
what the command did, so no command needs an inverse of its own, and a step reverses
exactly what its command changed, however many entries that appended.

There is one `History` for all windows, since the settings are shared by every window's
session: with the steps in the order their commands ran, putting back the settings a
step found, all of them, returns them to how they were before that command, whichever
window it ran in. Each window's worksheet is its own, so a step holds the one session
its command acted on; no command acts on more than one. Undoing a step of a window
other than the active one changes that window, and nothing switches to it. Closing a
window forgets its steps. A window made by splitting starts as a copy, and the steps
before the split stay those of the window split.

The sticky plot preferences are not part of a snapshot, since a plot window sets them,
and undoing never touches a plot window. Where the worksheet is saved is not part of
its state either: undoing past a Save or a Load leaves the next save going where it
would have gone.

What makes a step is the command, not the calls it makes on the session, which are
many and varied. `History.begin` takes a snapshot when a command starts, and the command
is over when the next one starts, or when something is undone or redone: commands end
in many ways (an answer arriving, a question abandoned, a dialog committed), but none of
them changes the worksheet after another command has started. A command that changes
nothing (one abandoned, refused or failed) leaves no step, and neither does moving the
highlight.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rederive.model.session import Entry, Route, Session
    from rederive.syntax import ParseState

#: How many steps each stack keeps; past that, the oldest is forgotten.
DEPTH = 100


@dataclass(frozen=True)
class Snapshot:
    """A session's state at one moment, with the settings: everything an undo puts back.

    The tables are the session's own, copied; nothing changes them once they are here.
    `settings` holds every setting but the plot preferences.
    """

    entries: tuple[Entry, ...]
    removed: tuple[Entry, ...]
    state: ParseState
    assignments: Mapping[str, Any]
    functions: Mapping[str, Any]
    domains: Mapping[str, Any]
    order: tuple[str, ...]
    arbitrary: int
    random: int
    next_number: int
    unsaved_objects: bool
    settings: Mapping[str, str | int]
    selected: int | None
    route: Route

    def same(self, other: Snapshot) -> bool:
        """Whether `other` holds the same worksheet, settings and definitions.

        The selection is not compared, since moving the highlight changes nothing to
        undo, and neither is whether the plots a worksheet was read with are still to
        be reported, which only a save or a load changes. Entries are compared by
        identity first, so this is quick when little changed.
        """
        return (
            self.entries == other.entries
            and self.removed == other.removed
            and _parsing(self.state) == _parsing(other.state)
            and self.assignments == other.assignments
            and self.functions == other.functions
            and self.domains == other.domains
            and self.order == other.order
            and self.arbitrary == other.arbitrary
            and self.random == other.random
            and self.next_number == other.next_number
            and self.settings == other.settings
        )


def _parsing(state: ParseState) -> tuple:
    """What of a parse state reading depends on; not its caches or revision."""
    return (
        state.input_mode,
        state.case_mode,
        state.input_base,
        state.functions,
        state.variables,
        state.withdrawn,
    )


@dataclass(frozen=True)
class Step:
    """One entry of a stack: the session a command acted on, the state to put back,
    and the command's name."""

    description: str
    session: Session
    snapshot: Snapshot


@dataclass(frozen=True)
class _Open:
    """The command in progress: its session, the state it started from, its name."""

    session: Session
    before: Snapshot
    description: str


class History:
    """The undo and redo stacks of all windows, each at most `DEPTH` deep, and the
    command in progress, which becomes a step when it is over.

    One command runs at a time, so what a command changed becomes a step when the next
    command starts, whichever window that is in, and a change to the shared settings
    belongs to the command that made it. A change made outside any command, if there is
    one, becomes part of the open command's step. A new step empties the redo stack,
    since what was undone no longer follows from the state.
    """

    def __init__(self, depth: int = DEPTH) -> None:
        self.undone: deque[Step] = deque(maxlen=depth)
        self.redone: deque[Step] = deque(maxlen=depth)
        self._open: _Open | None = None

    def begin(self, session: Session, description: str) -> None:
        """Start a command on `session`, ending the one before it."""
        self.settle()
        self._open = _Open(session, session.snapshot(), description)

    def settle(self) -> None:
        """Make what the open command has changed so far a step, and go on from here.

        The command stays open, so what it changes later is a step of its own.
        """
        if self._open is None:
            return
        session, before = self._open.session, self._open.before
        if not before.same(session.snapshot()):
            self.undone.append(Step(self._open.description, session, before))
            self.redone.clear()
        self._refresh()

    @contextmanager
    def aside(self, session: Session, description: str) -> Iterator[None]:
        """Make what happens inside one step of `session`, whatever command is open.

        This is for what arrives from outside the command in progress, such as a point
        sent from a plot window while a question is asked: the open command goes on
        afterwards, from the state the step left.
        """
        interrupted = self._open
        self.begin(session, description)
        try:
            yield
        finally:
            self.settle()
            if interrupted is not None:
                self._open = replace(
                    interrupted, before=interrupted.session.snapshot()
                )

    def undo(self) -> Step | None:
        """Return to the state before the last step, and say what step that was.

        None, with nothing changed, when there is no step to undo.
        """
        return self._moved(self.undone, self.redone)

    def redo(self) -> Step | None:
        """Return to the state the last undo left, and say what step it undid.

        None, with nothing changed, when nothing has been undone since the last step.
        """
        return self._moved(self.redone, self.undone)

    def forget(self, session: Session) -> None:
        """Drop the steps of `session`, whose window has closed, and its open command.

        The other windows' steps stay, in their order: undoing the one before a dropped
        step returns the settings to how that one found them.
        """
        for stack in (self.undone, self.redone):
            kept = [step for step in stack if step.session is not session]
            stack.clear()
            stack.extend(kept)
        if self._open is not None and self._open.session is session:
            self._open = None

    def _moved(self, source: deque[Step], target: deque[Step]) -> Step | None:
        """Pop the last step of `source`, return its session to the step's snapshot,
        and push the reverse, from the state as it is now, onto `target`."""
        self.settle()
        if not source:
            return None
        step = source.pop()
        target.append(replace(step, snapshot=step.session.snapshot()))
        step.session.return_to(step.snapshot)
        self._refresh()
        return step

    def _refresh(self) -> None:
        """Have the open command start from now, which an undo in any window moves,
        since it puts the settings back."""
        if self._open is not None:
            self._open = replace(self._open, before=self._open.session.snapshot())
