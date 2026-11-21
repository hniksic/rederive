"""sympy -> Node: the tree of every result, and its text.

Every result becomes a worksheet entry here. The printer builds its tree (`printer.tree`)
and the writer writes the tree's text, giving the tree spans that index the text, so
navigation, highlighting and the built-up display work as they do on authored input.

Under a decimal notation a result is built twice: as shown, which the worksheet draws
and saves, and with exact ratios, which the next command computes with. The latter is
`Result.value`, None when both are the same.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import sympy as sp

from rederive.engine.boundary import Result
from rederive.engine.context import (
    CLEARED,
    CLEARED_SETTING,
    Context,
    Notation,
    SettingValue,
)
from rederive.engine.native import given
from rederive.engine.printer import is_written, named, tree
from rederive.engine.to_sympy import Assign, StringLiteral, Vector
from rederive.model.expr import Kind, Node
from rederive.syntax import written

#: Every distinct answer with a head the notation has no name for.
#:
#: Writing such a head as an inert string is right for unforeseen heads, but not
#: silently: `RootSum`, `AccumBounds` and the set heads all reached the worksheet as dead
#: text unnoticed. So each is recorded here and logged, and the test suite checks that
#: the list stays empty.
UNREADABLE: list[str] = []


def _unreadable(text: str) -> None:
    """Record and log an answer the notation can't say.

    The worker's stderr is its log file, so this lands in the log of the process
    that computed the answer. Each distinct text is logged once per process to keep
    the log readable.
    """
    if text in UNREADABLE:
        return
    UNREADABLE.append(text)
    print(f"result not expressible in author notation: {text}", file=sys.stderr)


def from_sympy(
    expression: sp.Basic,
    context: Context | None = None,
    authored: dict[sp.Basic, Node] | None = None,
) -> Result:
    """`expression` as the tree and text of a worksheet entry.

    `authored` is passed to the printer, which puts in each tree it maps for its
    expression. The answer carries what the command performed in `context`.
    """
    with deep_recursion(_ANSWER_DEPTH):
        return _from_sympy(expression, context or Context(), authored)


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


#: The recursion limit for building an answer's tree and writing it. A deep recursion of
#: user functions builds an answer as deeply nested, a pending call inside thousands of
#: operations, and the printer and writer recurse once per level.
_ANSWER_DEPTH = 200_000


def _from_sympy(
    expression: sp.Basic,
    context: Context,
    authored: dict[sp.Basic, Node] | None,
) -> Result:
    expression = named(expression)
    text, node = tree_of(expression, context, authored)
    if node.kind is Kind.STRING and not isinstance(expression, StringLiteral):
        # The printer had no name for a head and quoted it as a string, which is just
        # as dead. Authored strings are not failures.
        _unreadable(text)
    elif not is_written(expression):
        # Likewise for a function head quoted inside a larger answer.
        _unreadable(text)
    exact = None
    if context.notation is not Notation.RATIONAL:
        # A decimal style writes a number only to the digits shown, so reading it back
        # gives a different number. Derive saves the shown form but computes with the
        # exact one, so both are kept.
        exactly = replace(context, notation=Notation.RATIONAL)
        exact_text, exact = tree_of(expression, exactly, authored)
        if exact_text == text:
            exact = None
    store = context.store
    return Result(
        node,
        text,
        exact,
        effects(context),
        tuple(store.displayed),
        store.written,
        random=store.random,
    )


def effects(context: Context) -> tuple[Node, ...]:
    """What the command's programs did, as trees for the session to perform (see
    `Store`): an `ASSIGN` of each global assigned or cleared, with its final value
    written exactly, and each definition and declaration, in order.
    """
    store = context.store
    done = [
        (at, kind, name, node) for (kind, name), (at, node) in store.effects.items()
    ]
    done.extend(
        (cell.at, "assign", name, None) for name, cell in store.cells.items() if cell.at
    )
    if not done:
        return ()
    done.sort(key=lambda each: each[0])
    # The evaluator imports this module, so its helper is imported when needed.
    from rederive.engine.evaluate import released

    exact = replace(context, notation=Notation.RATIONAL)
    found: list[Node] = []
    for _, kind, name, node in done:
        if node is not None:
            found.append(node)
            continue
        if kind == "set":
            setting = store.settings[name]
            value = CLEARED if setting == CLEARED_SETTING else setting_value(setting)
        else:
            value = given(store, name)
        if name.startswith('"'):
            target: sp.Basic = StringLiteral(name[1:-1])
        else:
            target = sp.Symbol(name)
        parts = [target, sp.Symbol(":=")]
        if value is not CLEARED:
            parts.append(released(value))
        found.append(tree_of(Assign(*parts), exact)[1])
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


def written_tree(value: sp.Basic, context: Context) -> Node:
    """`value` as a tree, exact, as an answer is (see `from_sympy`), without the
    answer's other parts."""
    exact = replace(context, notation=Notation.RATIONAL)
    return tree_of(named(value), exact)[1]


def tree_of(
    expression: sp.Basic,
    context: Context,
    authored: dict[sp.Basic, Node] | None = None,
) -> tuple[str, Node]:
    """`expression`'s text under `context`, and its tree, which the text indexes."""
    return written(tree(expression, context, authored))
