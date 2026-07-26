"""The state file: every system control setting as a line of text.

`Transfer Save State` writes one, `Transfer Load State` reads it back, and the
point of it is that a session opens the way you left it.

The format is the worksheet's own, narrowed to assignments: one
`Name := Value` per line, in the order the Options screens present them. That
is deliberate. Derive's DERIVE.INI was a Lisp reader's view of its internals -
`*TRIG-EXPD* |Auto|`, `*DEFAULT-DOMAIN* (|Real| ((((1 INF) . -1) . T) ...))` -
which is exactly the kind of file that has to be kept in step with the program.
These lines are the ones a user could have authored, and authoring one has
always changed the setting, so the file says nothing the program does not
already accept from the Author line.

Numerals are decimal here whatever `Options Radix` is set to. A state file is
read back by a session whose radix is whatever it happens to be, and a base
that only the file itself selects cannot be used to read the file.
"""

from __future__ import annotations

from rederive.model.settings import FIELDS, Settings

ASSIGN = ":="

#: What a state file is called when the name typed has no extension of its own.
SUFFIX = ".ini"


def write(settings: Settings) -> str:
    """Every setting as an assignment, in the order the dialogs present them."""
    return "".join(f"{name} {ASSIGN} {settings[name]}\n" for name in FIELDS)


def read(text: str, settings: Settings) -> int:
    """Apply the assignments in `text`, and say how many lines would not take.

    A line naming no setting, or giving one a value it will not have, is
    counted and passed over: the rest of the file still applies, for the reason
    one bad line of a worksheet does not cost you the other two hundred.
    """
    refused = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        name, found, value = line.partition(ASSIGN)
        if not found or not settings.assign(name.strip(), value.strip()):
            refused += 1
    return refused
