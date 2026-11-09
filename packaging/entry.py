"""PyInstaller entry point, which calls `freeze_support` before `main`.

In a frozen program, spawning the worker re-runs the executable, so the child starts here
with multiprocessing's arguments. `freeze_support` recognizes those and runs the worker
instead of the app. It must run before `main`, which rejects unknown arguments. That is
why the program is frozen from this module rather than from `rederive/__main__.py`.
"""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()

    from rederive.__main__ import main

    sys.exit(main())
