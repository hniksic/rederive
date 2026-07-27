"""Entry point: `rederive` / `python -m rederive`.

The engine worker is spawned here rather than by the app, because it is the
program that owns it and not the screen: the worker starts before the first
frame, so its sympy import is paid for while the user is still reading the
menu, and it is ended in a `finally`, so however the app leaves there is no
child left behind.

Nothing at import time may have an effect, and nothing may run outside the
guard below. A spawned worker re-imports whatever this module is, and a side
effect here would start a second copy of the app inside the child.
"""

from __future__ import annotations

from rederive.engine import RemoteEngine
from rederive.model.session import Session
from rederive.ui.app import RederiveApp


def main() -> None:
    runner = RemoteEngine()
    runner.start()
    try:
        RederiveApp(Session(runner=runner)).run()
    finally:
        runner.shutdown()


if __name__ == "__main__":
    main()
