"""Derive expression syntax: text in, expression trees out.

Nothing outside this package may import from inside it except through the
names re-exported here.

Parsing produces no mathematics. `2+3` yields a sum node, never `5`: authored
expressions stay inert until the user asks for a Simplify.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from rederive.model.expr import Kind, Node
from rederive.syntax.errors import DeriveSyntaxError
from rederive.syntax.lexer import Lexer
from rederive.syntax.parser import Parser
from rederive.syntax.source import Source
from rederive.syntax.state import (
    CaseMode,
    ClearWhat,
    Declaration,
    DomainDeclaration,
    FunctionDeclaration,
    FunctionInfo,
    InputMode,
    ParseState,
    SettingDeclaration,
    VariableDeclaration,
    VariableInfo,
)

__all__ = [
    "CaseMode",
    "ClearWhat",
    "Declaration",
    "DeriveSyntaxError",
    "DomainDeclaration",
    "FunctionDeclaration",
    "FunctionInfo",
    "InputMode",
    "Kind",
    "Node",
    "ParseResult",
    "ParseState",
    "SettingDeclaration",
    "Source",
    "VariableDeclaration",
    "VariableInfo",
    "parse_expression",
    "parse_source",
]


@dataclass(frozen=True)
class ParseResult:
    """One expression, and what accepting it would declare.

    `declarations` is what this expression would add to the symbol table:
    variable definitions, function definitions and setting changes. The parser
    does not apply them to the caller's state at top level; the caller does.
    """

    node: Node
    declarations: tuple[Declaration, ...]
    source: Source


def parse_expression(text: str, state: ParseState) -> ParseResult:
    """Parse exactly one expression. Raises DeriveSyntaxError.

    Used by the Author line. For a file, go through `Source.from_file` and
    `parse_source` instead, so comments and `~` splices are handled.
    """
    source = Source.from_line(text)
    return _parse_one(source, state, 0, len(source.text))


def parse_source(source: Source, state: ParseState) -> Iterator[ParseResult]:
    """Parse a .MTH/.DMO source, one expression at a time.

    Lazy by contract, not by convenience: `InputMode := Word` on line 3 must
    change how line 5 lexes, so the caller applies each result's declarations
    before pulling the next expression.
    """
    text = source.text
    start = 0
    while start <= len(text):
        stop = text.find("\n", start)
        if stop < 0:
            stop = len(text)
        if text[start:stop].strip():
            yield _parse_one(source, state, start, stop)
        if stop >= len(text):
            return
        start = stop + 1


def _parse_one(source: Source, state: ParseState, start: int, limit: int) -> ParseResult:
    parser = Parser(Lexer(source, state, start, limit))
    node = parser.parse()
    return ParseResult(node, tuple(parser.declarations), source)
