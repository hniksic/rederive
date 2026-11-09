"""Entry point: `rederive` / `python -m rederive`.

The engine worker is spawned here rather than by the app, so that it starts before the
first frame (its sympy import overlaps with the user reading the menu) and is shut down
in a `finally`, leaving no child process behind however the app exits.

Files named on the command line are also read here, before the first frame, as the
original did: `DERIVE ARITH` starts on ARITH rather than on an empty worksheet.

Nothing may have an effect at import time or run outside the `__main__` guard, because a
spawned worker re-imports this module. For the same reason Textual is imported inside
`main`, which the child never calls, rather than at module level.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from rederive import __version__
from rederive.model import worksheet

if TYPE_CHECKING:
    from rederive.model.session import Session

USAGE = "usage: rederive [--version] [-m|-u|-t|-d] FILE ..."

#: Print version information instead of starting the app. A release binary bundles its
#: own interpreter, sympy and Qt, so this is the only way to tell what it runs. The build
#: workflow checks its output against `.python-version` and `uv.lock`.
VERSION = "--version"

#: Switches that say how the following files are read. The original used a slash suffix
#: (`NUMBER/U`), which would clash with path separators, so these are prefix switches,
#: each in effect until the next: `NUMBER/U PLOT2D/M` becomes `-u NUMBER -m PLOT2D`.
SWITCHES = {
    "-m": "math",
    "--math": "math",
    "-u": "utility",
    "--utility": "utility",
    "-t": "data",
    "--data": "data",
    "-d": "demo",
    "--demo": "demo",
}

#: File kind implied by each extension when no switch is given. The order is also the
#: search order for a name given without an extension, as in the original.
BY_SUFFIX = {
    worksheet.SUFFIX: "math",
    worksheet.DATA_SUFFIX: "data",
    worksheet.DEMO_SUFFIX: "demo",
}

#: Default extension for each kind. A utility file is a math file read differently, so
#: it uses MTH.
SUFFIXES = {
    "math": worksheet.SUFFIX,
    "utility": worksheet.SUFFIX,
    "data": worksheet.DATA_SUFFIX,
    "demo": worksheet.DEMO_SUFFIX,
}


class Usage(Exception):
    """The command line does not say anything that can be acted on."""


def named(arguments: Sequence[str]) -> list[tuple[str, Path]]:
    """Return the files named on the command line, each with its kind.

    A switch holds until the next one. Without a switch, the kind comes from the
    extension, and a name without an extension is looked up under each in turn, so
    `rederive arith` finds ARITH.DMO.
    """
    kind: str | None = None
    files: list[tuple[str, Path]] = []
    for argument in arguments:
        if argument in SWITCHES:
            kind = SWITCHES[argument]
        elif argument.startswith("-") and argument != "-":
            raise Usage(f"unknown option: {argument}")
        else:
            files.append(_resolved(kind, Path(argument).expanduser()))
    return files


def _resolved(kind: str | None, path: Path) -> tuple[str, Path]:
    """Return the kind and actual file for `path`.

    A switch decides the kind, otherwise the extension does. A name with neither is
    looked up under each extension, falling back to a math file, so a missing file is
    reported as the one the user most likely meant.
    """
    if kind is not None:
        return kind, worksheet.path_of(str(path), SUFFIXES[kind])
    if path.suffix:
        return BY_SUFFIX.get(path.suffix.lower(), "math"), path
    for suffix, found in BY_SUFFIX.items():
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            return found, candidate
    return "math", path.with_suffix(worksheet.SUFFIX)


def read(session: Session, files: Sequence[tuple[str, Path]]) -> tuple[Path | None, str]:
    """Read every file except the demonstration.

    Returns the demonstration to run once the screen is up, and a message to show.

    Math files are merged rather than loaded: into an empty worksheet the two are the
    same, and merging lets each subsequent file add to the worksheet instead of
    replacing it.
    """
    readers: dict[str, Callable[[Path], int]] = {
        "math": session.merge,
        "utility": session.load_utility,
        "data": session.load_data,
    }
    demo: Path | None = None
    skipped = 0
    for kind, path in files:
        if kind == "demo":
            if demo is not None:
                raise Usage("only one demonstration can be run")
            demo = path
            continue
        try:
            skipped += readers[kind](path)
        except FileNotFoundError:
            raise Usage(f"{path}: no such file") from None
        except OSError as error:
            raise Usage(f"{path}: {error.strerror or 'cannot be read'}") from None
    return demo, _reported(skipped)


def _reported(skipped: int) -> str:
    """Return the message about lines that failed to parse, if any."""
    if not skipped:
        return ""
    lines = "line" if skipped == 1 else "lines"
    return f"{skipped} {lines} not read"


def provenance() -> str:
    """Return the versions of this program, Python, sympy, and the platform.

    One `name version` per line, readable by people and easy for a workflow to check.
    The lines after the first three come from `rederive.platform`, since they depend on
    the environment (e.g. whether there is Qt).

    The app process otherwise never imports sympy (it belongs to the worker) or the GUI
    toolkit (it belongs to the plot host), as `tests/test_packages.py` enforces. Doing
    it here is fine because this path prints and exits without opening a session.
    """
    import platform

    import sympy

    from rederive.platform import current

    return (
        f"rederive {__version__}\n"
        f"Python {platform.python_version()}\n"
        f"sympy {sympy.__version__}\n"
        f"{current().provenance()}"
    )


def _refused(refused: Usage) -> int:
    print(f"rederive: {refused}\n{USAGE}", file=sys.stderr)
    return 2


def main(arguments: Sequence[str] | None = None) -> int:
    """Read the files named on the command line, then run the app on the session.

    The command line is validated before the worker starts, so a bad one fails fast.

    The order of the local imports matters: the worker (which needs only the cheap
    engine proxy) is started before the screen is imported, so the child's sympy import
    and the parent's Textual import run in parallel.
    """
    given = sys.argv[1:] if arguments is None else arguments
    if VERSION in given:
        print(provenance())
        return 0

    try:
        files = named(given)
    except Usage as refused:
        return _refused(refused)

    from rederive.engine.remote import RemoteEngine

    runner = RemoteEngine()
    runner.start()
    try:
        from rederive.model.session import Session
        from rederive.ui.app import RederiveApp

        session = Session(runner=runner)
        try:
            demo, message = read(session, files)
        except Usage as refused:
            return _refused(refused)
        RederiveApp(session, demo=demo, opening=message).run()
    finally:
        runner.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
