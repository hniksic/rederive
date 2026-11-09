"""Derive expression syntax: parsing text into expression trees, and writing them back.

Code outside this package should import only the names re-exported here.

`write_expression` is the inverse of the parser: what it writes parses back to the same
tree. `write_source` writes a tree as C, Python, Rust or Julia, with no such round-trip
guarantee.

Parsing does no math: `2+3` yields a sum node, not `5`. Authored expressions stay inert
until the user simplifies them.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from rederive.model.expr import Kind, Node
from rederive.syntax.errors import DeriveSyntaxError
from rederive.syntax.languages import LANGUAGES, Language, write_source
from rederive.syntax.lexer import Lexer, is_name
from rederive.syntax.names import PARSING_SETTINGS
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
from rederive.syntax.writer import write_expression

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
    "LANGUAGES",
    "Language",
    "Node",
    "PARSING_SETTINGS",
    "ParseResult",
    "ParseState",
    "SettingDeclaration",
    "Source",
    "VariableDeclaration",
    "VariableInfo",
    "is_name",
    "parse_expression",
    "parse_source",
    "source_lines",
    "write_expression",
    "write_source",
]


@dataclass(frozen=True)
class ParseResult:
    """A parsed expression and the declarations it makes.

    `declarations` lists the variable definitions, function definitions and setting
    changes the expression would add to the symbol table. The caller, not the parser,
    applies them.
    """

    node: Node
    declarations: tuple[Declaration, ...]
    source: Source


def parse_expression(text: str, state: ParseState) -> ParseResult:
    """Parse exactly one expression, raising DeriveSyntaxError on failure.

    Used by the Author line. For files, use `Source.from_file` and `parse_source`, which
    handle comments and `~` continuations.
    """
    source = Source.from_line(text)
    return _parse_one(source, state, 0, len(source.text))


def source_lines(source: Source) -> Iterator[tuple[int, int]]:
    """Yield the span in `source.text` of each expression in a prepared file.

    After comment stripping and `~` splicing, each non-blank line is one expression.
    Callers that must recover from a line that fails to parse iterate the spans
    themselves.
    """
    text = source.text
    start = 0
    while start <= len(text):
        stop = text.find("\n", start)
        if stop < 0:
            stop = len(text)
        if text[start:stop].strip():
            yield start, stop
        if stop >= len(text):
            return
        start = stop + 1


def parse_source(source: Source, state: ParseState) -> Iterator[ParseResult]:
    """Parse a .MTH/.DMO source lazily, one expression at a time.

    Laziness is required: `InputMode := Word` on line 3 changes how line 5 lexes, so the
    caller must apply each result's declarations before pulling the next one.
    """
    for start, stop in source_lines(source):
        yield _parse_one(source, state, start, stop)


def _parse_one(source: Source, state: ParseState, start: int, limit: int) -> ParseResult:
    parser = Parser(Lexer(source, state, start, limit))
    node = parser.parse()
    return ParseResult(node, tuple(parser.declarations), source)
