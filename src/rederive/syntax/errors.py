"""The syntax error exception.

The lexer and parser both raise this single exception type, with one message and one
position. There is deliberately no hierarchy: a line refused for another reason than its
syntax, such as an invalid value for a setting, carries its own message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rederive.syntax.source import Source


class DeriveSyntaxError(Exception):
    """A rejected expression.

    `offset` is where the parser stopped, which may not be where the mistake is.
    `expected` is a short machine-readable hint, unused by the TUI but available to a
    richer front end. `message` replaces the usual one.
    """

    def __init__(
        self,
        offset: int,
        expected: str,
        source: Source | None = None,
        message: str | None = None,
    ) -> None:
        super().__init__(expected)
        self.offset = offset
        self.expected = expected
        self.source = source
        self.message = message

    def __str__(self) -> str:
        return self.message or "Syntax error detected at cursor"
