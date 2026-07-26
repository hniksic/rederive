"""Expression trees as s-expressions, for the conformance corpus.

The mapping is total: every node has exactly one written form, and
`(juxt a b)` is distinct from `(* a b)`.

A sum or a product is a run of terms rather than a nest of pairs, and carries
one operator per gap. When every gap is the same the operator heads the form,
`(+ a b c)` and `(juxt x y z)`; when they differ the run is named and the
operators are written between the terms it joins, `(sum a - b + c)` and
`(prod 2 juxt x * y)`.
"""

from __future__ import annotations

from rederive.model.expr import Kind, Node

_BINOPS = {"/": "/", "^": "^", ".": "dot"}
_UNOPS = {"-": "neg", "+": "pos", "+-": "pm"}
_POSTOPS = {"!": "fact", "%": "pct", "`": "transpose"}


def to_sexpr(node: Node) -> str:
    """Write `node` the way the corpus spells it."""
    children = [to_sexpr(child) for child in node.children]
    match node.kind:
        case Kind.NUMBER | Kind.NAME | Kind.UNKNOWN:
            return str(node.value)
        case Kind.STRING:
            return f'"{node.value}"'
        case Kind.LABEL:
            return f"(label {node.value})"
        case Kind.SUM | Kind.PRODUCT:
            return _run(node, children)
        case Kind.BINOP:
            return _form(_BINOPS[str(node.value)], children)
        case Kind.UNOP:
            return _form(_UNOPS[str(node.value)], children)
        case Kind.POSTOP:
            return _form(_POSTOPS[str(node.value)], children)
        case Kind.SUB:
            return _form("sub", children)
        case Kind.ABS:
            return _form("abs", children)
        case Kind.CALL:
            return _form("call", children)
        case Kind.APPLY:
            return _form("apply", children)
        case Kind.FUNCPOW:
            return _form("funcpow", children)
        case Kind.VECTOR:
            return _form("vec", children)
        case Kind.REL:
            # Binary, and the operator is written between its operands.
            return _form("rel", [children[0], str(node.value), children[1]])
        case Kind.NOT | Kind.AND | Kind.OR | Kind.XOR | Kind.IMP:
            return _form(str(node.kind), children)
        case Kind.ASSIGN:
            return _form(str(node.value), children)
        case Kind.FUNDEF:
            return _form("fundef", [str(node.value), *children])
        case Kind.PARAMS:
            return _form("params", children)
        case Kind.DOMAIN:
            return _form("domain", [children[0], str(node.value), *children[1:]])
        case Kind.INTERVAL:
            opening, closing = str(node.value)
            return _form("ivl", [f'"{opening}"', *children, f'"{closing}"'])
        case Kind.SHOWVALUE:
            return _form("showvalue", children)
    raise AssertionError(f"unwritable node kind: {node.kind}")


def _form(head: str, parts: list[str]) -> str:
    return "(" + " ".join([head, *parts]) + ")"


def _run(node: Node, children: list[str]) -> str:
    """A sum or a product, written by its gaps."""
    gaps = [_gap(node, index) for index in range(len(children) - 1)]
    if len(set(gaps)) == 1:
        return _form(gaps[0], children)
    head = "sum" if node.kind is Kind.SUM else "prod"
    parts = [children[0]]
    for gap, child in zip(gaps, children[1:], strict=True):
        parts += [gap, child]
    return _form(head, parts)


def _gap(node: Node, index: int) -> str:
    """The operator joining term `index` to the one after it."""
    if node.kind is Kind.SUM:
        return str(node.value)[index]
    written = node.surface or str(node.value)
    return "juxt" if written[index] == " " else "*"
