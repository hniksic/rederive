"""sympy -> Node, by way of the printer and the parser.

A result becomes a worksheet entry only here, whoever computed it. The route
is print then reparse rather than build a tree directly, and that is
deliberate: the reparsed tree has spans indexing the very text the entry
shows, so subexpression navigation, highlighting and the built-up display all
work on a result exactly as they work on something the user authored.

The reparse needs a symbol table - in Character mode `tera` is one name only
once something has declared it - so the caller may supply a `ParseState`, and
`parse_state_for` builds one that knows every name the expression mentions.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from sympy.core.function import AppliedUndef

from rederive.engine.context import Context
from rederive.engine.printer import author_text
from rederive.engine.to_sympy import FunDef, StringLiteral
from rederive.model.expr import Kind, Node
from rederive.syntax import (
    DeriveSyntaxError,
    FunctionDeclaration,
    ParseState,
    VariableDeclaration,
    parse_expression,
)

#: What joins a subscripted symbol's parts in its name.
_SUBSCRIPT = " SUB "


@dataclass(frozen=True)
class Result:
    """An expression as the worksheet holds it: the text, and its tree.

    `node`'s spans index `text`.
    """

    node: Node
    text: str


def from_sympy(
    expression: sp.Basic,
    context: Context | None = None,
    state: ParseState | None = None,
) -> Result:
    """Write `expression` as author notation, and read it back as a tree."""
    context = context or Context()
    text = author_text(expression, context)
    if state is None:
        state = parse_state_for(expression, context)
    try:
        return Result(parse_expression(text, state).node, text)
    except DeriveSyntaxError:
        # Nothing the engine produces should be unreadable, but a result that
        # is stays inert and legible rather than taking the command down.
        return Result(Node(Kind.STRING, 0, len(text), (), text), text)


def parse_state_for(expression: sp.Basic, context: Context) -> ParseState:
    """A parse state that reads back everything `expression` can mention.

    Every name in the expression, and every name the context knows, is
    declared, so that a multi-character name lexes as the one name it is.

    The modes are the default ones - Character input, case-insensitive - and
    only the input base comes from the context, which is all the context
    carries. A session working in another mode has a state of its own and
    should pass it: under `CaseMode := Sensitive`, `n` and `N` are two
    variables, and a state that folds them would read one result as the
    other.
    """
    state = ParseState(input_base=context.input_base)
    # Every symbol, not only the free ones: the index of a sum and the
    # variable of a limit have to lex as themselves too.
    for symbol in expression.atoms(sp.Symbol):
        if isinstance(symbol, StringLiteral):
            continue
        for part in symbol.name.split(_SUBSCRIPT):
            _declare_variable(state, part)
    for name in (*context.domains, *context.assignments):
        _declare_variable(state, name)
    # Functions last: a name is one or the other, and the later declaration
    # displaces the earlier.
    for call in expression.atoms(AppliedUndef):
        _declare_function(state, type(call).__name__, len(call.args))
    for definition in expression.atoms(FunDef):
        name, parameters = definition.args[0], definition.args[1]
        _declare_function(state, str(name), len(parameters))
    for name, (parameters, _) in context.functions.items():
        _declare_function(state, name, len(parameters))
    return state


def _declare_variable(state: ParseState, name: str) -> None:
    if _is_name(name):
        state.declare(VariableDeclaration(name))


def _declare_function(state: ParseState, name: str, arity: int) -> None:
    if _is_name(name):
        parameters = tuple(f"_{index}" for index in range(arity))
        state.declare(FunctionDeclaration(name, parameters, has_body=False))


def _is_name(text: str) -> bool:
    """Whether `text` is a name the lexer would read whole.

    Labels, arbitrary constants and subscripted symbols carry punctuation the
    lexer has its own rules for, and declaring them would teach it nothing.
    """
    return bool(text) and text[0].isalpha() and all(
        character.isalnum() or character == "_" for character in text
    )
