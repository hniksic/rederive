"""sympy -> Node, by way of the printer and the parser.

A result becomes a worksheet entry only here, whoever computed it. The route
is print then reparse rather than build a tree directly, and that is
deliberate: the reparsed tree has spans indexing the very text the entry
shows, so subexpression navigation, highlighting and the built-up display all
work on a result exactly as they work on something the user authored.

The reparse needs a symbol table - in Character mode `tera` is one name only
once something has declared it - so the caller may supply a `ParseState`, and
`parse_state_for` builds one that knows every name the expression mentions.

A result under a decimal notation is written twice: once as it is shown, which
is what the worksheet draws and saves, and once as the ratios it is made of,
which is what the next command computes with. `Result.value` is the second,
and it is None when the two would be the same text.
"""

from __future__ import annotations

import sys
from dataclasses import replace

import sympy as sp
from sympy.core.function import AppliedUndef

from rederive.engine.boundary import Result
from rederive.engine.context import Context, Notation
from rederive.engine.printer import author_text, named
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

#: Every answer the notation could not say, each the first time it was met.
#:
#: Falling back to an inert string is right for a head nobody foresaw. Doing it
#: silently is not: `RootSum`, `AccumBounds` and the set heads all reached the
#: worksheet as dead text and none of them was found by anything but somebody
#: going to look. So each is noted here and written to the log, and the suite
#: reads the list to ask of every command it runs that the list stayed empty.
UNREADABLE: list[str] = []


def _unreadable(text: str) -> None:
    """Note an answer that is not author notation, where it can be read.

    The worker has no terminal and its standard error is its log file, opened
    by the first thing written to it, so a line here is a line in the log of
    the process that computed the answer - which is where a head nobody
    anticipated should turn up, rather than in somebody's reading of a node
    kind.

    Once per distinct text and per process. An answer met a thousand times is
    one thing to name, and a log that repeats itself is a log nobody reads.
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

    `authored` is passed to the printer, which writes what it names as the text
    it is given rather than as the expression it holds.
    """
    context = context or Context()
    # Named first, so that the symbol table below is built from the names the
    # text actually carries rather than from the dummies behind them.
    expression = named(expression)
    text = author_text(expression, context, authored)
    if state is None:
        state = parse_state_for(expression, context)
    else:
        state = _knowing(state, expression, context)
    try:
        node = parse_expression(text, state).node
    except DeriveSyntaxError:
        # Nothing the engine produces should be unreadable, but a result that
        # is stays inert and legible rather than taking the command down.
        _unreadable(text)
        return Result(Node(Kind.STRING, 0, len(text), (), text), text)
    if node.kind is Kind.STRING and not isinstance(expression, StringLiteral):
        # A head the printer had no word for, written as the quoted string the
        # notation keeps unreadable text in. It reparses, and it is as dead as
        # one that did not, so it is worth the same notice. A string the author
        # wrote is a string and not a failure.
        _unreadable(text)
    return Result(node, text, _exact(expression, context, state, text, authored))


def _exact(
    expression: sp.Basic,
    context: Context,
    state: ParseState,
    text: str,
    authored: dict[sp.Basic, str] | None = None,
) -> Node | None:
    """The same answer written as the ratios it is made of, or None if that is
    what `text` already is.

    A decimal style writes a number to as many digits as it is shown to, and
    reading that back is reading a different number - which is what the
    original saves to a file, and not what it goes on computing with. So the
    exact writing is kept beside the shown one.
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
    """The caller's state, plus the names only the result mentions.

    The caller's modes are what the session is working in and are kept exactly.
    Its symbol table is not enough on its own: an answer can name something the
    author line never did - the bound variable sympy invents for the Leibniz
    rule, say - and in Character mode `k1` is `k*1` until something declares
    it, so the answer would read back as a different expression.

    A copy, because the caller's state belongs to the caller. Declaring the
    engine's own working names into the session's table would leave them there
    as variables the user never wrote.
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
    # variable of a limit have to lex as themselves too. A declared nonscalar
    # is a `MatrixSymbol` and no `Symbol`, and its name has to lex as well.
    for symbol in expression.atoms(sp.Symbol, sp.MatrixSymbol):
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
