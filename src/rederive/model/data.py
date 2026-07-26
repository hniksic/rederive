"""Numeric data files: rows of numbers in, matrices out.

The format is the original's, and it is the plainest one in the program. A line
is a list of numbers separated by spaces, commas or both, and each line is one
row of a matrix; a blank line ends the matrix and starts the next. That is all
a DAT file is - no header, no shape, nothing to keep in step with the program -
which is why instruments and other math programs could write one.

The one thing that has to be translated is the exponent. A data file may write
`-2.325E-7`, and Derive's own expression syntax has no exponent notation at
all, so the number becomes `-2.325*10^-7` on the way in. `D` is accepted for
`E`, Fortran having written double precision that way.
"""

from __future__ import annotations

import re

_MAGNITUDE = r"[-+]?(?:\d+\.?\d*|\.\d+)"

#: A number the way a data file writes one: integer, decimal or rational, with
#: an optional exponent. Anything else on a row is not a number, and the row it
#: is on is not data.
NUMBER = re.compile(rf"^(?P<size>{_MAGNITUDE})(?:[EeDd](?P<power>[-+]?\d+))?"
                    rf"(?P<over>/\d+)?$")

#: What separates two numbers on a row.
SEPARATORS = re.compile(r"[\s,]+")


def matrix(block: list[str]) -> str:
    """One block of lines as Derive text, ready to author.

    A block of several lines is a matrix of that many rows; a block of one line
    is that row alone, as a vector, which is what the original makes of it.

    Raises `ValueError` when a line holds something that is not a number, which
    is what a caller reports as a block it could not read.
    """
    rows = [f"[{','.join(_numbers(line))}]" for line in block]
    return rows[0] if len(rows) == 1 else f"[{','.join(rows)}]"


def blocks(text: str) -> list[list[str]]:
    """The file's lines, grouped into the matrices the blank lines delimit."""
    found: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip():
            current.append(line)
        elif current:
            found.append(current)
            current = []
    return found + ([current] if current else [])


def _numbers(line: str) -> list[str]:
    """One row's numbers, each in Derive notation."""
    return [_number(word) for word in SEPARATORS.split(line.strip()) if word]


def _number(word: str) -> str:
    """One number, with any exponent spelled as the power it means.

    Raises `ValueError` when `word` is not a number at all.
    """
    found = NUMBER.match(word)
    if found is None:
        raise ValueError(word)
    size, power, over = found.group("size", "power", "over")
    return f"{size}{over or ''}" if power is None else f"{size}*10^{power}{over or ''}"
