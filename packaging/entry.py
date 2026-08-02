"""What PyInstaller compiles, and the one call it has to make before anything else.

A frozen program cannot start its worker the way an installed one does. Spawning
re-runs the executable rather than the interpreter, so the child arrives back at this
module carrying the arguments multiprocessing gave it, and something has to recognise
those and hand the process to the worker instead of to the app. `freeze_support` is
that something, and it has to run before the command line is read: `main` refuses
arguments it does not know, and the child's are arguments it does not know.

Putting the call in front of `main` is the whole of this module, and the reason the
program is frozen from here rather than from `rederive/__main__.py`. Nothing else
about starting up changes; `main` is reached with the command line the user typed, and
in the child it is never reached at all.
"""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()

    from rederive.__main__ import main

    sys.exit(main())
