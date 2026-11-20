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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import sympy as sp
from sympy.core.function import AppliedUndef

from rederive.engine.boundary import Result
from rederive.engine.context import (
    CLEARED,
    CLEARED_SETTING,
    Context,
    Notation,
    SettingValue,
)
from rederive.engine.printer import HeldBase, author_text, is_written, named
from rederive.engine.to_sympy import Assign, FunDef, StringLiteral, Vector
from rederive.model.expr import Kind, Node
from rederive.syntax import (
    CaseMode,
    DeriveSyntaxError,
    FunctionDeclaration,
    ParseState,
    VariableDeclaration,
    is_name,
    parse_expression,
    reserved,
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
    with deep_recursion(_ANSWER_DEPTH):
        return _from_sympy(expression, context or Context(), state, authored)


@contextmanager
def deep_recursion(limit: int) -> Iterator[None]:
    """Raise the Python recursion limit to `limit` while the block runs."""
    previous = sys.getrecursionlimit()
    if previous < limit:
        sys.setrecursionlimit(limit)
    try:
        yield
    finally:
        sys.setrecursionlimit(previous)


#: The recursion limit for writing an answer out and reading it back. A deep recursion
#: of user functions builds an answer as deeply nested, a pending call inside thousands
#: of operations, and the printer and parser recurse once per level.
_ANSWER_DEPTH = 200_000


def _from_sympy(
    expression: sp.Basic,
    context: Context,
    state: ParseState | None,
    authored: dict[sp.Basic, str] | None,
) -> Result:
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
        return Result(
            string,
            text,
            performed=effects(context, state),
            displayed=tuple(context.store.displayed),
            written=context.store.written,
            random=context.store.random,
        )
    if node.kind is Kind.STRING and not isinstance(expression, StringLiteral):
        # The printer had no name for a head and quoted it as a string. It reparses
        # but is just as dead, so record it too. Authored strings are not failures.
        _unreadable(text)
    elif not is_written(expression):
        # Likewise for a function head quoted inside a larger answer.
        _unreadable(text)
    exact = _exact(expression, context, state, text, authored)
    store = context.store
    return Result(
        node,
        text,
        exact,
        effects(context, state),
        tuple(store.displayed),
        store.written,
        random=store.random,
    )


def effects(context: Context, state: ParseState | None = None) -> tuple[Node, ...]:
    """What the command's programs did, as trees for the session to perform (see
    `Store`): an `ASSIGN` of each global assigned or cleared, with its final value
    written exactly, and each definition and declaration, in order.

    A value that doesn't read back is left out and recorded as unreadable, like an
    answer.
    """
    store = context.store
    if not store.effects:
        return ()
    # The evaluator imports this module, so its helper is imported when needed.
    from rederive.engine.evaluate import released

    exact = replace(context, notation=Notation.RATIONAL)
    found: list[Node] = []
    for (kind, name), node in store.effects.items():
        if node is not None:
            found.append(node)
            continue
        if kind == "set":
            setting = store.settings[name]
            value = CLEARED if setting == CLEARED_SETTING else setting_value(setting)
        else:
            value = store.values[name]
        if name.startswith('"'):
            target: sp.Basic = StringLiteral(name[1:-1])
        else:
            target = sp.Symbol(name)
        parts = [target, sp.Symbol(":=")]
        if value is not CLEARED:
            parts.append(released(value))
        assignment = Assign(*parts)
        text = author_text(assignment, exact)
        known = parse_state_for(assignment, exact)
        if state is not None:
            known = _knowing(state, assignment, exact)
        try:
            tree = parse_expression(text, known).node
        except DeriveSyntaxError:
            tree = None
        if tree is None or tree.kind is not Kind.ASSIGN:
            _unreadable(text)
            continue
        found.append(tree)
    return tuple(found)


def setting_value(value: SettingValue) -> sp.Basic:
    """A setting's value as an expression: a number, its keyword as a name, a truth
    value, or the vector of the variables in order."""
    if isinstance(value, int):
        return sp.Integer(value)
    if isinstance(value, tuple):
        return Vector(*(sp.Symbol(name) for name in value))
    if value in ("true", "false"):
        return sp.true if value == "true" else sp.false
    return sp.Symbol(value)


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
    for call in _called(expression):
        name = type(call).__name__
        if state.resolve(name) is None:
            _declare_function(known, name, len(call.args))
    _know_held(known, expression)
    return known


def _know_held(state: ParseState, expression: sp.Basic) -> None:
    """Declare the names the held values in `expression` mention, which are in their
    trees rather than among its atoms: variables, and the functions they call that
    `state` doesn't know."""
    try:
        atoms = expression.atoms(HeldBase)
    except Exception:
        return
    for atom in atoms:
        _know_tree(state, type(atom).node)


def _know_tree(state: ParseState, node: Node) -> None:
    if node.kind is Kind.NAME:
        name = str(node.value)
        if state.resolve(name) is None:
            _declare_variable(state, name)
        return
    children = node.children
    if node.kind in (Kind.CALL, Kind.APPLY):
        name = str(children[0].value)
        if state.resolve(name) is None:
            _declare_function(state, name, len(children) - 1)
        children = children[1:]
    for child in children:
        _know_tree(state, child)


def written_tree(value: sp.Basic, context: Context) -> Node:
    """`value` as a tree: its text, exact, read back as an answer is (see
    `from_sympy`), without the answer's other parts."""
    exact = replace(context, notation=Notation.RATIONAL)
    value = named(value)
    text = author_text(value, exact)
    return parse_expression(text, parse_state_for(value, exact)).node


def parse_state_for(expression: sp.Basic, context: Context) -> ParseState:
    """A parse state that reads back everything `expression` can mention.

    Every name in the expression and the context is declared, so multi-character
    names lex as one name.

    The input mode is the default, Character, with every name declared. The input base
    and case mode come from the context: under `CaseMode := Sensitive`, `n` and `N` are
    different variables.
    """
    case_mode = CaseMode.SENSITIVE if context.case_sensitive else CaseMode.INSENSITIVE
    state = ParseState(input_base=context.input_base, case_mode=case_mode)
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
    for call in _called(expression):
        _declare_function(state, type(call).__name__, len(call.args))
    for definition in expression.atoms(FunDef):
        name, parameters = definition.args[0], definition.args[1]
        # A variadic head has its one parameter alone, not in a tuple.
        arity = len(parameters) if isinstance(parameters, sp.Tuple) else 1
        _declare_function(state, str(name), arity)
    for name, definition in context.functions.items():
        _declare_function(state, name, len(definition.names))
    _know_held(state, expression)
    return state


def _called(expression: sp.Basic) -> list[AppliedUndef]:
    """The calls of undefined functions in `expression` whose names must be declared
    as functions to read back.

    A name that is also a variable in it, as `l` in `l*o*e*l(v, w, n)`, stays one when
    no call of it has a single argument: a name before `(` with no argument or several
    is read as a call anyway, while one declared a function couldn't stand alone.
    """
    calls = [
        call
        for call in expression.atoms(AppliedUndef)
        if not isinstance(call, HeldBase)
    ]
    symbols = {
        symbol.name
        for symbol in expression.atoms(sp.Symbol)
        if not isinstance(symbol, StringLiteral)
    }
    single = {type(call).__name__ for call in calls if len(call.args) == 1}
    return [
        call
        for call in calls
        if type(call).__name__ not in symbols or type(call).__name__ in single
    ]


def _declare_variable(state: ParseState, name: str) -> None:
    """Declare `name` a variable, unless it is reserved, as the keyword a `Logical`
    carries is, which must keep reading as the keyword."""
    if _is_name(name) and not reserved(name):
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
    return is_name(text)
