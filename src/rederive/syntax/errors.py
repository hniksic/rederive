"""The one syntax error.

One exception type, one message, one position. Lexer and parser both raise it;
there is deliberately no hierarchy and no second message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rederive.syntax.source import Source


class DeriveSyntaxError(Exception):
    """A rejected expression.

    `offset` is where the parser stopped, which is not necessarily where the
    mistake is. `expected` is a short machine-readable hint that the v1 TUI
    never shows; it exists so a richer front end can do better without changing
    the engine.
    """

    def __init__(self, offset: int, expected: str, source: Source | None = None) -> None:
        super().__init__(expected)
        self.offset = offset
        self.expected = expected
        self.source = source

    def __str__(self) -> str:
        return "Syntax error detected at cursor"
