"""Tokens.

A token's `value` is its canonical spelling - the operator, or the name the
symbol table knows. `surface` is how it was written, when that differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TokenKind(StrEnum):
    NUMBER = "number"
    NAME = "name"
    STRING = "string"
    LABEL = "label"
    OP = "op"
    EOF = "eof"


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    start: int
    end: int
    value: str = ""
    surface: str | None = None
    is_function: bool = False
    is_defhead: bool = False
    """Set on a name that a `:=` lookahead proved to be a definition head."""
    word: str | None = None
    """The word operator this name is where an operator can stand, such as `ROW` for a
    `row` with no user name of that spelling, or `DEG` for the constant `deg`. The name
    itself is the token; the operator, if the parser takes it, extends to `word_end`,
    which in Character mode may lie beyond the token's end."""
    word_end: int = 0
