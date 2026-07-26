"""Expression trees as s-expressions, for the conformance corpus.

The mapping is total: every node has exactly one written form, and
`(juxt a b)` is distinct from `(* a b)`.
"""

from __future__ import annotations

from rederive.model.expr import Kind, Node

_BINOPS = {"*": "*", "/": "/", "+": "+", "-": "-", "^": "^", ".": "dot"}
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
        case Kind.BINOP:
            juxtaposed = node.value == "*" and node.surface == ""
            head = "juxt" if juxtaposed else _BINOPS[str(node.value)]
            return _form(head, children)
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
            return _form("rel", _interleave(children, str(node.value).split()))
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


def _interleave(operands: list[str], operators: list[str]) -> list[str]:
    parts = [operands[0]]
    for operator, operand in zip(operators, operands[1:], strict=True):
        parts += [operator, operand]
    return parts
