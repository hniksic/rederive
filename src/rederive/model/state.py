"""The state file: every system control setting as a line of text.

`Transfer Save State` writes one and `Transfer Load State` reads it back, so a session
can be restored the way you left it.

Each line is a `Name := Value` assignment, in the order the Options screens show them,
followed by the variable order list. Unlike Derive's DERIVE.INI, which dumped Lisp
internals (`*TRIG-EXPD* |Auto|`), these are lines a user could author to change the
setting, so the format needs no separate maintenance. The variable order can't be
authored (only `Manage Ordering` sets it), but uses the same shape.

Numerals are always decimal regardless of `Options Radix`, since the file is read back
by a session whose radix is unknown.
"""

from __future__ import annotations

from collections.abc import Sequence

from rederive.model.settings import FIELDS, Settings

ASSIGN = ":="

#: Default extension for a state file.
SUFFIX = ".ini"

#: Name for the variable order list kept by `Manage Ordering`. It is session state rather
#: than a `Settings` entry, but belongs in the state file, as it did in DERIVE.INI
#: (`*VARIABLE-ORDER* (\x \y \z)`).
ORDER = "VariableOrder"


def write(settings: Settings, order: Sequence[str] = ()) -> str:
    """Write every setting as an assignment, in the order the dialogs show them.

    The variable order list follows on its own line, written as `Manage Ordering`
    accepts it.
    """
    lines = [f"{name} {ASSIGN} {settings[name]}\n" for name in FIELDS]
    lines.append(f"{ORDER} {ASSIGN} {' '.join(order)}\n")
    return "".join(lines)


def read(text: str, settings: Settings) -> tuple[int, str | None]:
    """Apply the assignments in `text`, returning the number refused and the order.

    A line with an unknown setting or an invalid value is counted and skipped, and the
    rest of the file still applies.

    The variable order line is returned rather than applied, because parsing variable
    names requires the session's symbol table. None means the file had no such line.
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
