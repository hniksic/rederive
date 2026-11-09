"""What every engine client does when its worker dies, regardless of transport.

A desktop worker is a child process behind a pipe, killed with a signal; a browser
worker is a Web Worker behind `postMessage`, killed with `terminate()`. They share this
policy, and the app's error reporting is written against it.

Every death is handled the same way, whether caused by Esc, the memory cap, the OS, a
crash, or the worker just stopping: the pending call fails with an exception naming the
cause, and a replacement starts immediately. This is safe because the worker is
stateless, and well tested because every app start launches a fresh worker.

The exception is a worker that can't start at all, handled by `Deaths`. Three deaths in
a row before the handshake stop automatic respawning, so a broken install doesn't spin:
the client goes down, engine commands fail fast with the reason, and each new command
tries one more start. Retrying at the pace of keypresses avoids backoff machinery.
Everything that doesn't need the engine keeps working, so the worksheet stays savable.
"""

from __future__ import annotations

from rederive.engine.client import EngineError

__all__ = ["STARTS_BEFORE_DOWN", "Deaths"]

#: Consecutive deaths before the handshake after which a client stops restarting workers
#: on its own and waits for the next command.
STARTS_BEFORE_DOWN = 3


class Deaths:
    """Tracks consecutive failed worker starts and decides when the client goes down.

    The counter resets as soon as a worker answers its handshake, since a worker that
    ran and then died says nothing about whether the next will start. `down` holds the
    error that took the client down, which commands are refused with until a retry.
    """

    def __init__(self, limit: int = STARTS_BEFORE_DOWN) -> None:
        self.limit = limit
        #: The error that put the client down, or None while it is up.
        self.down: EngineError | None = None
        self._failures = 0

    def ready(self) -> None:
        """Record that a worker answered its handshake, resetting the failure count."""
        self._failures = 0

    def died(self, error: EngineError, ready: bool) -> bool:
        """Record a death and return whether to start a replacement immediately.

        `ready` is whether the worker had finished starting. Only deaths before that
        count as failed starts; enough in a row put the client down.
        """
        self._failures = 0 if ready else self._failures + 1
        if self._failures < self.limit:
            return True
        self.down = error
        return False

    def retrying(self) -> None:
        """Bring the client back up because a command was issued.

        This is the whole retry policy: each command the user issues allows one more
        start attempt, and nothing retries in the background.
        """
        self.down = None
