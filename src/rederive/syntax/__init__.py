"""Derive expression syntax: text in, expression trees out, and text again.

Nothing outside this package may import from inside it except through the
names re-exported here.

`write_expression` is the parser run backwards, and the two are kept together
because they are one contract: what the writer writes, the parser reads back as
the same tree.

Parsing produces no mathematics. `2+3` yields a sum node, never `5`: authored
expressions stay inert until the user asks for a Simplify.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from rederive.model.expr import Kind, Node
from rederive.syntax.errors import DeriveSyntaxError
from rederive.syntax.lexer import Lexer
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
    "Node",
    "PARSING_SETTINGS",
    "ParseResult",
    "ParseState",
    "SettingDeclaration",
    "Source",
    "VariableDeclaration",
    "VariableInfo",
    "parse_expression",
    "parse_source",
    "source_lines",
    "write_expression",
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


def source_lines(source: Source) -> Iterator[tuple[int, int]]:
    """The span of each expression of a prepared file, in `source.text`.

    One line, one expression: comments are already gone and a `~` continuation
    has already joined its lines into one, so what is left is either blank or a
    whole expression. This is where that rule lives; a caller that has to
    survive a line which does not parse walks the spans itself.
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
    """Parse a .MTH/.DMO source, one expression at a time.

    Lazy by contract, not by convenience: `InputMode := Word` on line 3 must
    change how line 5 lexes, so the caller applies each result's declarations
    before pulling the next expression.
    """
    for start, stop in source_lines(source):
        yield _parse_one(source, state, start, stop)


def _parse_one(source: Source, state: ParseState, start: int, limit: int) -> ParseResult:
    parser = Parser(Lexer(source, state, start, limit))
    node = parser.parse()
    return ParseResult(node, tuple(parser.declarations), source)
