"""The syntax error exception.

The lexer and parser both raise this single exception type, with one message and one
position. There is deliberately no hierarchy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rederive.syntax.source import Source


class DeriveSyntaxError(Exception):
    """A rejected expression.

    `offset` is where the parser stopped, which may not be where the mistake is.
    `expected` is a short machine-readable hint, unused by the TUI but available to a
    richer front end.
    """

    def __init__(self, offset: int, expected: str, source: Source | None = None) -> None:
        super().__init__(expected)
        self.offset = offset
        self.expected = expected
        self.source = source

    def __str__(self) -> str:
        return "Syntax error detected at cursor"
