"""Numeric data files: rows of numbers in, matrices out.

The format is the original's DAT format: each line is a matrix row of numbers separated
by blanks, tabs, commas or semicolons, and a blank line starts a new matrix. There is no
header, which made the format easy for instruments and other programs to write. A line
whose first character that isn't blank is `;` is the annotation of the next matrix, as
Derive 5 reads it. A decimal comma can't be told from a separator, and separates.

Derive syntax has no exponent notation, so `-2.325E-7` is translated to
`-2.325*10^-7`. `D` is accepted in place of `E`, as Fortran writes double precision.
"""

from __future__ import annotations

import re

_MAGNITUDE = r"[-+]?(?:\d+\.?\d*|\.\d+)"

#: A number in a data file: integer, decimal or rational, with an optional exponent.
NUMBER = re.compile(rf"^(?P<size>{_MAGNITUDE})(?:[EeDd](?P<power>[-+]?\d+))?"
                    rf"(?P<over>/\d+)?$")

SEPARATORS = re.compile(r"[\s,;]+")

#: What starts the annotation line of a matrix.
COMMENT = ";"


def matrix(block: list[str]) -> str:
    """Convert a block of lines to Derive text, ready to author.

    A multi-line block becomes a matrix, and a single line a vector, as in the original.

    Raises `ValueError` if a line contains something other than numbers.
    """
    rows = [f"[{','.join(_numbers(line))}]" for line in block]
    return rows[0] if len(rows) == 1 else f"[{','.join(rows)}]"


def blocks(text: str) -> list[tuple[list[str], str | None]]:
    """Split the file's lines into blocks separated by blank lines, each with the
    annotation an annotation line before it gives, or None."""
    found: list[tuple[list[str], str | None]] = []
    current: list[str] = []
    annotation: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(COMMENT):
            if current:
                found.append((current, annotation))
                current = []
            annotation = stripped[len(COMMENT) :].strip()
        elif stripped:
            current.append(line)
        elif current:
            found.append((current, annotation))
            current = []
            annotation = None
    return found + ([(current, annotation)] if current else [])


def _numbers(line: str) -> list[str]:
    """Return a row's numbers in Derive notation."""
    return [_number(word) for word in SEPARATORS.split(line.strip()) if word]


def _number(word: str) -> str:
    """Convert a number to Derive notation, writing any exponent as a power of 10.

    Raises `ValueError` if `word` is not a number.
    """
    found = NUMBER.match(word)
    if found is None:
        raise ValueError(word)
    size, power, over = found.group("size", "power", "over")
    return f"{size}{over or ''}" if power is None else f"{size}*10^{power}{over or ''}"
