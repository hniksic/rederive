"""TEMPORARY placeholder parser - throwaway scaffolding.

Milestone 1 does no mathematics, but subexpression selection still has to move
over the *structure* of an expression rather than over characters. This module
is the cheapest thing that gives authored text a plausible shape: a naive
tokenizer and recursive-descent parser that knows numbers, identifiers,
`+ - * / ^`, parentheses, and juxtaposition as implicit multiplication.

It validates nothing and computes nothing. Input that does not parse degrades
to a single atom covering the whole line - input is never rejected.

The whole module is deleted when the real engine lands. Nothing outside it may
depend on its internals; consumers see only `Node` trees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rederive.model.tree import Node

_TOKEN_RE = re.compile(
    r"""
      (?P<space>\s+)
    | (?P<number>\d+(?:\.\d*)?|\.\d+)
    | (?P<name>[A-Za-z_][A-Za-z_0-9]*)
    | (?P<op>[-+*/^()])
    """,
    re.VERBOSE,
)

_STARTS_FACTOR = ("number", "name")


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str
    start: int
    end: int


class _ParseError(Exception):
    pass


@dataclass(frozen=True)
class _Parsed:
    """A parsed node plus the extent of the source text that produced it.

    `node.start`/`node.end` is the highlight span, which excludes enclosing
    parentheses; `lo`/`hi` include them, so that an enclosing node's span
    reaches over the parentheses of its operands.
    """

    node: Node
    lo: int
    hi: int


def parse(text: str) -> Node:
    """Parse `text` into a tree, degrading to a single atom on any error."""
    try:
        parser = _Parser(_tokenize(text))
        parsed = parser.parse_sum()
        if parser.peek() is not None:
            raise _ParseError
        return parsed.node
    except _ParseError:
        return Node("atom", 0, len(text))


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN_RE.match(text, pos)
        if match is None:
            raise _ParseError
        kind = match.lastgroup
        assert kind is not None
        if kind != "space":
            tokens.append(_Token(kind, match.group(), match.start(), match.end()))
        pos = match.end()
    if not tokens:
        raise _ParseError
    return tokens


class _Parser:
    def __init__(self, tokens: list[_Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    def peek(self) -> _Token | None:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _take(self) -> _Token:
        token = self.peek()
        if token is None:
            raise _ParseError
        self._pos += 1
        return token

    def _at_op(self, *texts: str) -> bool:
        token = self.peek()
        return token is not None and token.kind == "op" and token.text in texts

    def parse_sum(self) -> _Parsed:
        """Additive chain, flattened into one n-ary sum."""
        terms = [self.parse_product()]
        while self._at_op("+", "-"):
            operator = self._take()
            term = self.parse_product()
            if operator.text == "-":
                # `a - b` is a sum whose second term is the negation `- b`.
                term = _wrap("neg", operator.start, term)
            terms.append(term)
        if len(terms) == 1:
            return terms[0]
        return _combine("sum", terms)

    def parse_product(self) -> _Parsed:
        """Multiplicative chain: `*`, juxtaposition, and `/`, left to right."""
        factors = [self.parse_unary()]
        while True:
            if self._at_op("*"):
                self._take()
                factors.append(self.parse_unary())
            elif self._at_op("/"):
                self._take()
                left = _combine("product", factors)
                right = self.parse_unary()
                factors = [_combine("quotient", [left, right])]
            elif self._starts_factor():
                factors.append(self.parse_unary())
            else:
                break
        return _combine("product", factors)

    def parse_unary(self) -> _Parsed:
        if self._at_op("+", "-"):
            operator = self._take()
            operand = self.parse_unary()
            kind = "neg" if operator.text == "-" else "pos"
            return _wrap(kind, operator.start, operand)
        return self.parse_power()

    def parse_power(self) -> _Parsed:
        base = self.parse_primary()
        if self._at_op("^"):
            self._take()
            # Right-associative, and the exponent may carry its own sign.
            return _combine("power", [base, self.parse_unary()])
        return base

    def parse_primary(self) -> _Parsed:
        token = self._take()
        if token.kind in _STARTS_FACTOR:
            return _Parsed(Node("atom", token.start, token.end), token.start, token.end)
        if token.kind == "op" and token.text == "(":
            inner = self.parse_sum()
            closing = self._take()
            if closing.text != ")":
                raise _ParseError
            # The parentheses widen the extent but not the highlight span.
            return _Parsed(inner.node, token.start, closing.end)
        raise _ParseError

    def _starts_factor(self) -> bool:
        token = self.peek()
        if token is None:
            return False
        return token.kind in _STARTS_FACTOR or (token.kind == "op" and token.text == "(")


def _combine(kind: str, parts: list[_Parsed]) -> _Parsed:
    """Build a node spanning `parts`, or pass a lone part straight through."""
    if len(parts) == 1:
        return parts[0]
    node = Node(kind, parts[0].lo, parts[-1].hi, tuple(part.node for part in parts))
    return _Parsed(node, node.start, node.end)


def _wrap(kind: str, start: int, operand: _Parsed) -> _Parsed:
    node = Node(kind, start, operand.hi, (operand.node,))
    return _Parsed(node, node.start, node.end)
