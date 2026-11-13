"""sympy -> Node, by way of the printer and the parser.

Every result becomes a worksheet entry here. It is printed and reparsed rather than
converted directly so that the tree's spans index the entry's text, making navigation,
highlighting and the built-up display work as they do on authored input.

Reparsing needs a symbol table (in Character mode `tera` is one name only once
declared), so the caller may supply a `ParseState`; `parse_state_for` builds one that
knows every name in the expression.

Under a decimal notation a result is written twice: as shown, which the worksheet draws
and saves, and as exact ratios, which the next command computes with. The latter is
`Result.value`, None when both texts are identical.
"""

from __future__ import annotations

import sys
from dataclasses import replace

import sympy as sp
from sympy.core.function import AppliedUndef

from rederive.engine.boundary import Result
from rederive.engine.context import Context, Notation
from rederive.engine.printer import author_text, is_written, named
from rederive.engine.to_sympy import FunDef, StringLiteral
from rederive.model.expr import Kind, Node
from rederive.syntax import (
    DeriveSyntaxError,
    FunctionDeclaration,
    ParseState,
    VariableDeclaration,
    parse_expression,
)

#: Separator between the parts of a subscripted symbol's name.
_SUBSCRIPT = " SUB "

#: Every distinct answer that couldn't be written in author notation.
#:
#: Falling back to an inert string is right for unforeseen heads, but not silently:
#: `RootSum`, `AccumBounds` and the set heads all reached the worksheet as dead text
#: unnoticed. So each is recorded here and logged, and the test suite checks that the
#: list stays empty.
UNREADABLE: list[str] = []


def _unreadable(text: str) -> None:
    """Record and log an answer that isn't author notation.

    The worker's stderr is its log file, so this lands in the log of the process
    that computed the answer. Each distinct text is logged once per process to keep
    the log readable.
    """
    if text in UNREADABLE:
        return
    UNREADABLE.append(text)
    print(f"result not readable as author notation: {text}", file=sys.stderr)


def from_sympy(
    expression: sp.Basic,
    context: Context | None = None,
    state: ParseState | None = None,
    authored: dict[sp.Basic, str] | None = None,
) -> Result:
    """Write `expression` as author notation, and read it back as a tree.

    `authored` is passed to the printer, which writes each expression it maps as the
    given text. The answer carries what the command performed in `context`.
    """
    context = context or Context()
    # Name dummies first so the symbol table is built from the names in the text.
    expression = named(expression)
    text = author_text(expression, context, authored)
    if state is None:
        state = parse_state_for(expression, context)
    else:
        state = _knowing(state, expression, context)
    try:
        node = parse_expression(text, state).node
    except DeriveSyntaxError:
        # This shouldn't happen, but an unreadable result is kept as an inert string
        # rather than failing the command.
        _unreadable(text)
        string = Node(Kind.STRING, 0, len(text), (), text)
        return Result(string, text, performed=tuple(context.performed.nodes))
    if node.kind is Kind.STRING and not isinstance(expression, StringLiteral):
        # The printer had no name for a head and quoted it as a string. It reparses
        # but is just as dead, so record it too. Authored strings are not failures.
        _unreadable(text)
    elif not is_written(expression):
        # Likewise for a function head quoted inside a larger answer.
        _unreadable(text)
    exact = _exact(expression, context, state, text, authored)
    return Result(node, text, exact, tuple(context.performed.nodes))


def _exact(
    expression: sp.Basic,
    context: Context,
    state: ParseState,
    text: str,
    authored: dict[sp.Basic, str] | None = None,
) -> Node | None:
    """The answer written with exact ratios, or None if that is what `text` is.

    A decimal style writes a number only to the digits shown, so reading it back gives
    a different number. Derive saves the shown form but computes with the exact one,
    so both are kept.
    """
    if context.notation is Notation.RATIONAL:
        return None
    exact = author_text(
        expression, replace(context, notation=Notation.RATIONAL), authored
    )
    if exact == text:
        return None
    try:
        return parse_expression(exact, state).node
    except DeriveSyntaxError:
        return None


def _knowing(
    state: ParseState, expression: sp.Basic, context: Context
) -> ParseState:
    """A copy of the caller's state, with the result's new names declared.

    The caller's modes are kept. Its symbol table may lack names the answer introduces
    (e.g. a bound variable sympy invents for the Leibniz rule), and in Character mode
    `k1` reads as `k*1` until declared.

    The state is copied so the engine's working names don't leak into the session's
    table as variables the user never wrote.
    """
    known = replace(
        state, functions=dict(state.functions), variables=dict(state.variables)
    )
    for symbol in expression.atoms(sp.Symbol, sp.MatrixSymbol):
        if isinstance(symbol, StringLiteral):
            continue
        for part in symbol.name.split(_SUBSCRIPT):
            if state.resolve(part) is None:
                _declare_variable(known, part)
    for call in expression.atoms(AppliedUndef):
        name = type(call).__name__
        if state.resolve(name) is None:
            _declare_function(known, name, len(call.args))
    return known


def parse_state_for(expression: sp.Basic, context: Context) -> ParseState:
    """A parse state that reads back everything `expression` can mention.

    Every name in the expression and the context is declared, so multi-character
    names lex as one name.

    Modes are the defaults (Character input, case-insensitive); only the input base
    comes from the context. A session in another mode should pass its own state: under
    `CaseMode := Sensitive`, `n` and `N` are different variables.
    """
    state = ParseState(input_base=context.input_base)
    # All symbols, not only free ones: a sum's index and a limit's variable must lex
    # too. Declared nonscalars are `MatrixSymbol`s, not `Symbol`s.
    for symbol in expression.atoms(sp.Symbol, sp.MatrixSymbol):
        if isinstance(symbol, StringLiteral):
            continue
        for part in symbol.name.split(_SUBSCRIPT):
            _declare_variable(state, part)
    for name in (*context.domains, *context.assignments):
        _declare_variable(state, name)
    # Functions last, so they override a variable of the same name.
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

    Labels, arbitrary constants and subscripted symbols contain punctuation the lexer
    handles itself, so declaring them is pointless.
    """
    return bool(text) and text[0].isalpha() and all(
        character.isalnum() or character == "_" for character in text
    )
