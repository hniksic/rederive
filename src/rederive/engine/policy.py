"""What every engine client does when its worker dies, whatever the worker was.

The transports have nothing in common. A desktop worker is a child process behind a
pipe, killed with a signal; a browser worker is a Web Worker behind `postMessage`,
killed with `terminate()`. What they share is the policy, and it is the policy rather
than the machinery that the app's reporting is written against.

Every death funnels into one path. The user pressing Esc, a memory cap met, the
system taking the process away, a crash and a worker that simply stopped are one
shape: the pending call fails with the exception that names the cause, and a
replacement is started straight away. That is only ordinary because the worker is
stateless - the replacement knows everything the dead one knew, which is nothing -
and because a fresh worker is started every time the app starts, so the recovery path
is exercised on every run and not only in emergencies.

The one thing that is not ordinary is a worker that cannot start at all, and `Deaths`
is the rule for it. Three deaths before the handshake in a row stop the automatic
respawn, since a broken install would otherwise spin: the client goes down, engine
commands fail fast saying why, and the next command tries one more start - retry at
the pace of a person pressing keys, with no backoff machinery to get wrong.
Everything that does not need the engine goes on working while it is down, which is
the property that matters: the mathematics may die, the worksheet must always be
savable.
"""

from __future__ import annotations

from rederive.engine.client import EngineError

__all__ = ["STARTS_BEFORE_DOWN", "Deaths"]

#: Deaths before the handshake, in a row, after which a client stops starting
#: workers by itself and waits to be asked.
STARTS_BEFORE_DOWN = 3


class Deaths:
    """The run of workers that never answered, and what it decides.

    One counter and one exception. The counter is of starts that died before the
    handshake, in a row, and it goes back to nothing the moment a worker answers
    one - a worker that ran and then died says nothing about whether the next one
    will start. The exception is what put the client down, which is what a command
    asked for while it is down is refused with.
    """

    def __init__(self, limit: int = STARTS_BEFORE_DOWN) -> None:
        self.limit = limit
        #: What put the client down, or None while it is up.
        self.down: EngineError | None = None
        self._failures = 0

    def ready(self) -> None:
        """A worker has answered its handshake, so the run of failed starts is over."""
        self._failures = 0

    def died(self, error: EngineError, ready: bool) -> bool:
        """Note one death, and say whether to put a replacement up straight away.

        `ready` is whether the worker that died had finished starting. One that
        had says nothing about the install, however it died; one that had not is
        the start failing, and enough of those in a row are what puts the client
        down.
        """
        self._failures = 0 if ready else self._failures + 1
        if self._failures < self.limit:
            return True
        self.down = error
        return False

    def retrying(self) -> None:
        """A command has asked again, which is the whole of the retry policy.

        The user pressing a command key is the clock: the client tries one more
        start and nothing spins behind their back in between.
        """
        self.down = None
