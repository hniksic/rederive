"""Entry point: `rederive` / `python -m rederive`.

The engine worker is spawned here rather than by the app, because it is the
program that owns it and not the screen: the worker starts before the first
frame, so its sympy import is paid for while the user is still reading the
menu, and it is ended in a `finally`, so however the app leaves there is no
child left behind.

Files named on the command line are read here too, into the session the app is
then handed. The original reads them the same way - `DERIVE ARITH` starts on
ARITH rather than on an empty worksheet - and the point of doing it before the
first frame is the same as for the worker: what the user is shown is already
what they asked for.

Nothing at import time may have an effect, and nothing may run outside the
guard below. A spawned worker re-imports whatever this module is, and a side
effect here would start a second copy of the app inside the child.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from rederive.engine import RemoteEngine
from rederive.model import worksheet
from rederive.model.session import Session
from rederive.ui.app import RederiveApp

USAGE = "usage: rederive [-m|-u|-t|-d] FILE ..."

#: How a switch says the files after it are to be read. The original wrote
#: these after the name and behind a slash - `NUMBER/U` - which cannot be
#: written here, a slash being what separates a directory from what is in it.
#: They are switches in front of the name instead, each holding until the next
#: one, so that the original's `NUMBER/U PLOT2D/M` is `-u NUMBER -m PLOT2D`.
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

#: What a file with each extension is taken to be when no switch says. In
#: search order, which is what a name given without an extension is looked for
#: under: the original supplied MTH, DAT and DMO the same way.
BY_SUFFIX = {
    worksheet.SUFFIX: "math",
    worksheet.DATA_SUFFIX: "data",
    worksheet.DEMO_SUFFIX: "demo",
}

#: The extension each kind supplies to a name given without one. A utility file
#: is a math file read a different way, so it is looked for under MTH.
SUFFIXES = {
    "math": worksheet.SUFFIX,
    "utility": worksheet.SUFFIX,
    "data": worksheet.DATA_SUFFIX,
    "demo": worksheet.DEMO_SUFFIX,
}


class Usage(Exception):
    """The command line does not say anything that can be acted on."""


def named(arguments: Sequence[str]) -> list[tuple[str, Path]]:
    """The files the command line names, each with how it is to be read.

    A switch holds until the next one; a name with no switch in front of it is
    read by its extension, and one given without an extension is looked for
    under each in turn, so that `rederive arith` finds ARITH.DMO.
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
    """What `path` is to be read as, and the file that name means.

    A switch settles it. Without one an extension does, and a name carrying
    neither is looked for under each extension in turn - falling back on a math
    file, so that a name that is nowhere is reported as the file the user most
    likely meant rather than as three that are all missing.
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
    """Read every file but the demonstration, and say what there is to report.

    Returns the demonstration to run once the screen is up, since that one is
    not a file to be read but a session to be driven, and whatever the reading
    has to say for itself.

    A math file is merged rather than loaded even though `Transfer Load Derive`
    loads: the two are the same thing into a worksheet with nothing in it, and
    merging is what lets a second one named after the first add to it instead
    of replacing it.
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
    """What the message line says about lines that would not parse, if any."""
    if not skipped:
        return ""
    lines = "line" if skipped == 1 else "lines"
    return f"{skipped} {lines} not read"


def _refused(refused: Usage) -> int:
    print(f"rederive: {refused}\n{USAGE}", file=sys.stderr)
    return 2


def main(arguments: Sequence[str] | None = None) -> int:
    """Read what the command line names, then hand the session to the screen.

    The command line is looked at before the worker is started, so that a line
    that says nothing usable costs nothing to find out about.
    """
    try:
        files = named(sys.argv[1:] if arguments is None else arguments)
    except Usage as refused:
        return _refused(refused)
    runner = RemoteEngine()
    runner.start()
    try:
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
