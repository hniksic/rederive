"""The state file: every system control setting as a line of text.

`Transfer Save State` writes one, `Transfer Load State` reads it back, and the
point of it is that a session opens the way you left it.

The format is the worksheet's own, narrowed to assignments: one
`Name := Value` per line, in the order the Options screens present them, plus
the variable order list at the end. That is deliberate. Derive's DERIVE.INI was
a Lisp reader's view of its internals - `*TRIG-EXPD* |Auto|`,
`*DEFAULT-DOMAIN* (|Real| ((((1 INF) . -1) . T) ...))` - which is exactly the
kind of file that has to be kept in step with the program. The setting lines
here are ones a user could have authored, and authoring one has always changed
the setting, so the file says nothing about them the program does not already
accept from the Author line. The order list has no such spelling - nothing
authors it, and `Manage Ordering` is the only way to write it - so it borrows
the same shape rather than a format of its own.

Numerals are decimal here whatever `Options Radix` is set to. A state file is
read back by a session whose radix is whatever it happens to be, and a base
that only the file itself selects cannot be used to read the file.
"""

from __future__ import annotations

from collections.abc import Sequence

from rederive.model.settings import FIELDS, Settings

ASSIGN = ":="

#: What a state file is called when the name typed has no extension of its own.
SUFFIX = ".ini"

#: The one line here that is not a system control setting: the variable order
#: list `Manage Ordering` keeps. It is session state and not a `Settings`
#: entry - no dialog field owns it, and no `Name := Value` records it - but a
#: state file is how a session opens the way you left it, and Derive's own
#: DERIVE.INI carried the list as `*VARIABLE-ORDER* (\x \y \z)`. Written in the
#: same shape as the settings, since that is the one shape this file has.
ORDER = "VariableOrder"


def write(settings: Settings, order: Sequence[str] = ()) -> str:
    """Every setting as an assignment, in the order the dialogs present them.

    The variable order list follows them, written as the words `Manage
    Ordering` takes on its own line.
    """
    lines = [f"{name} {ASSIGN} {settings[name]}\n" for name in FIELDS]
    lines.append(f"{ORDER} {ASSIGN} {' '.join(order)}\n")
    return "".join(lines)


def read(text: str, settings: Settings) -> tuple[int, str | None]:
    """Apply the assignments in `text`: how many would not take, and the order.

    A line naming no setting, or giving one a value it will not have, is
    counted and passed over: the rest of the file still applies, for the reason
    one bad line of a worksheet does not cost you the other two hundred.

    The variable order line is handed back rather than applied, because what a
    variable name is depends on the symbol table the session is reading with
    and only the session holds one. None means the file carried no such line.
    """
    refused = 0
    order: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        name, found, value = line.partition(ASSIGN)
        if found and name.strip() == ORDER:
            order = value.strip()
        elif not found or not settings.assign(name.strip(), value.strip()):
            refused += 1
    return refused, order
