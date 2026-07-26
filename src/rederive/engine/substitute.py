"""The pre-pass every command runs before converting: what a name stands for.

Labels, assigned variables and defined functions are replaced on the tree,
before sympy ever sees it, because that is where Derive replaces them: a
function body is a tree with its parameters written in, not a closure sympy
could hold. Simplify runs this first, and so will Expand, Factor, approX and
the rest; nothing in it belongs to any one of them.

Three rules make it terminate and stay honest:

* A variable is expanded at most once along a branch. `pn := pn + 1` therefore
  simplifies to `pn + 1` rather than spinning, and the name that could not be
  expanded again is left standing.
* A function is likewise expanded at most once along a branch, so a recursive
  definition leaves its own call in place.
* A call with fewer arguments than the definition has parameters substitutes
  the leading parameters only. `ACCELERATION(f, m) := f/m` applied to one
  argument gives `6/m`: the parameters nobody supplied stay as the names they
  are, which is Derive's partial application.
* A function that names a variable of its own binds it. With `x := 5`,
  `SUM(x^2, x, 1, 3)` is a sum over `x` and not the nonsense `SUM(25, 5, 1, 3)`:
  inside the first argument the bound name stands for itself, and it shadows a
  parameter of the same name.

What a definition names is never substituted for. `u := u + 1` keeps its own
left-hand side, `F(x) := x` keeps its parameter list, and a declaration keeps
the variable it declares - otherwise a definition line could not be written
twice.

The spans of the tree this returns index nothing: a body spliced in from
another entry brings that entry's offsets with it. That is harmless, because
the substituted tree is only ever converted, and the tree a `Result` carries
comes from reparsing the printed answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from rederive.engine.context import Context
from rederive.model.expr import Kind, Node
from rederive.syntax import ParseState
from rederive.syntax.names import BINDING_FUNCTIONS

__all__ = ["named_as_declared", "substitute"]


def named_as_declared(node: Node, state: ParseState | None) -> Node:
    """`node` with every name spelled the way the symbol table records it.

    Case-insensitively - which is the default - `x` and `X` are one variable,
    and which of the two a line carries depends on what had been declared when
    it was read. `X := ELEMENT(x, 1)` declares `X` and mentions `x` before that
    declaration exists, so the line comes back holding both spellings; the
    engine has to see one name for one variable, or the answer holds two and
    reading it back turns them into one.

    Without a symbol table nothing can be resolved and the tree stands as it
    is, which is what a caller with no session of its own gets.
    """
    if state is None:
        return node
    return _named(node, state)


def _named(node: Node, state: ParseState) -> Node:
    """The walk. A call's head is a name of another kind and is left alone.

    What the table records for a variable is the spelling to write it under,
    and for anything else it is the spelling of something else entirely: the
    setting value `Exact` for `EXACT(...)`, the letter `δ` for `DELTA(...)`.
    Neither reads back as the call it was written as.
    """
    head = node.children[0] if node.kind in (Kind.CALL, Kind.APPLY) else None
    children = tuple(
        child if child is head else _named(child, state) for child in node.children
    )
    if node.kind is Kind.NAME:
        known = state.resolve(str(node.value))
        if known is not None and not known.is_function and known.canonical != node.value:
            return replace(node, value=known.canonical, children=children)
    return replace(node, children=children) if children != node.children else node


@dataclass(frozen=True)
class _Scope:
    """What a name means at one point in the walk.

    `arguments` are the trees a call's parameters stand for, already
    substituted, so they are spliced in rather than walked again. `bound` are
    the names that stand for themselves: a definition's own parameters, and the
    variable a sum, an integral or a generated vector names.
    """

    arguments: Mapping[str, Node] = field(default_factory=dict)
    bound: frozenset[str] = frozenset()
    variables: frozenset[str] = frozenset()
    functions: frozenset[str] = frozenset()
    labels: frozenset[int] = frozenset()


def substitute(node: Node, context: Context) -> Node:
    """`node` with every label, assigned variable and defined function in it."""
    return _walk(node, _Scope(), context)


def _walk(node: Node, scope: _Scope, context: Context) -> Node:
    match node.kind:
        case Kind.NAME:
            return _name(node, scope, context)
        case Kind.LABEL:
            return _label(node, scope, context)
        case Kind.CALL | Kind.APPLY:
            return _call(node, scope, context)
        case Kind.ASSIGN:
            return _keeping_first(node, scope, context)
        case Kind.FUNDEF:
            return _fundef(node, scope, context)
        case Kind.DOMAIN:
            return node
    return _children(node, scope, context)


def _children(node: Node, scope: _Scope, context: Context) -> Node:
    return replace(node, children=tuple(_walk(c, scope, context) for c in node.children))


def _keeping_first(node: Node, scope: _Scope, context: Context) -> Node:
    """An assignment: its value is substituted, its target is not."""
    rest = tuple(_walk(child, scope, context) for child in node.children[1:])
    return replace(node, children=(node.children[0], *rest))


def _name(node: Node, scope: _Scope, context: Context) -> Node:
    name = str(node.value)
    argument = scope.arguments.get(name)
    if argument is not None:
        return argument
    if name in scope.bound or name in scope.variables:
        return node
    value = context.assignments.get(name)
    if value is None:
        return node
    deeper = replace(scope, arguments={}, variables=scope.variables | {name})
    return _walk(value, deeper, context)


def _label(node: Node, scope: _Scope, context: Context) -> Node:
    """`#3`, or itself when nobody knows it or it reaches itself."""
    try:
        number = int(str(node.value))
    except ValueError:
        return node
    target = context.labels.get(number)
    if target is None or number in scope.labels:
        return node
    deeper = replace(scope, arguments={}, labels=scope.labels | {number})
    return _walk(target, deeper, context)


def _call(node: Node, scope: _Scope, context: Context) -> Node:
    """A call, with its arguments substituted and its body written in."""
    head, *arguments = node.children
    name = str(head.value)
    if name in BINDING_FUNCTIONS:
        return _binding(node, arguments, scope, context)
    substituted = tuple(_walk(argument, scope, context) for argument in arguments)
    definition = context.functions.get(name)
    if definition is None or name in scope.functions:
        return replace(node, children=(head, *substituted))
    parameters, body = definition
    # Partial application: the parameters nobody supplied are left standing.
    paired = dict(zip(parameters, substituted, strict=False))
    deeper = _Scope(
        arguments=paired,
        variables=scope.variables,
        functions=scope.functions | {name},
        labels=scope.labels,
    )
    return _walk(body, deeper, context)


def _binding(
    node: Node, arguments: list[Node], scope: _Scope, context: Context
) -> Node:
    """`SUM(u, k, m, n)`: `k` stands for itself in `u`, and nowhere else.

    The name itself is left as written - substituting for it would leave the
    call with no variable to range over - and the arguments after it are read in
    the enclosing scope, because a limit is not part of what it bounds.

    A second argument that is no name binds nothing. `SUM(v)` over a vector's
    elements has none either, and both walk like any other call.
    """
    head = node.children[0]
    if len(arguments) < 2 or arguments[1].kind is not Kind.NAME:
        walked = tuple(_walk(argument, scope, context) for argument in arguments)
        return replace(node, children=(head, *walked))
    body, index, *rest = arguments
    name = str(index.value)
    inner = replace(
        scope,
        arguments={
            parameter: written
            for parameter, written in scope.arguments.items()
            if parameter != name
        },
        bound=scope.bound | {name},
    )
    written = (
        _walk(body, inner, context),
        index,
        *(_walk(argument, scope, context) for argument in rest),
    )
    return replace(node, children=(head, *written))


def _fundef(node: Node, scope: _Scope, context: Context) -> Node:
    """A definition: the body is substituted, its parameters stand for themselves."""
    parameters, *body = node.children
    names = frozenset(str(child.value) for child in parameters.children)
    deeper = replace(scope, arguments={}, bound=scope.bound | names)
    written = tuple(_walk(part, deeper, context) for part in body)
    return replace(node, children=(parameters, *written))
