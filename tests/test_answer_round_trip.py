"""The text of an answer reads back as the answer's tree.

The engine builds an answer's tree from the value it computed and writes the text from
the tree (`from_sympy`). Nothing reads the text back on the way, so this is the check
that the writer and the parser agree on what the engine builds: a worksheet that is
saved, or an answer that is edited, must read back as the tree it was, spans and
all.

The values are the answers to a sample of the corpus, under the notations, bases and
settings the engine writes differently, and values made directly for the heads the
corpus doesn't reach.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import sympy as sp
from test_simplify_corpus import GROUPS

from rederive.engine.computing import Context, from_sympy, simplify
from rederive.engine.context import Notation
from rederive.engine.to_sympy import PlusMinus, StringLiteral, Vector
from rederive.model.expr import Kind, Node
from rederive.syntax import (
    DeriveSyntaxError,
    FunctionDeclaration,
    ParseState,
    VariableDeclaration,
    is_name,
    parse_expression,
    reserved,
    write_expression,
)

#: Every how many corpus lines one is taken, which keeps the run to a few seconds.
_EVERY = 12

#: The settings that change how an answer is written.
CONTEXTS = {
    "exact": Context(),
    "decimal": Context(notation=Notation.DECIMAL, notation_digits=6),
    "mixed": Context(notation=Notation.MIXED),
    "scientific": Context(notation=Notation.SCIENTIFIC, notation_digits=4),
    "hexadecimal": Context(input_base=16),
}


def _sample() -> list[tuple[str, ParseState]]:
    lines = [line for group in GROUPS.values() for line in group]
    return lines[::_EVERY]


SAMPLE = _sample()


def _knowing(state: ParseState, node: Node) -> ParseState:
    """A copy of `state` that also knows the names `node` mentions, as the session
    whose answer it is knows them: variables, and functions where they are called."""
    known = replace(
        state, functions=dict(state.functions), variables=dict(state.variables)
    )
    _declare(known, node)
    return known


def _declare(state: ParseState, node: Node) -> None:
    if node.kind is Kind.NAME:
        name = str(node.value)
        if is_name(name) and not reserved(name) and state.resolve(name) is None:
            state.declare(VariableDeclaration(name))
        return
    children = node.children
    if node.kind is Kind.CALL:
        name = str(children[0].value)
        if is_name(name) and state.resolve(name) is None:
            arity = len(children) - 1
            parameters = tuple(f"_{index}" for index in range(arity))
            state.declare(FunctionDeclaration(name, parameters, has_body=False))
        children = children[1:]
    for child in children:
        _declare(state, child)


def _has_decimal(node: Node) -> bool:
    """Whether `node` has a decimal in it. A decimal is written in base ten whatever
    the input base, and reads back in the input base, so outside base ten it doesn't
    read back as itself."""
    if node.kind is Kind.NUMBER:
        return "." in str(node.surface or node.value)
    return any(_has_decimal(child) for child in node.children)


def _reads_back(node: Node, text: str, state: ParseState) -> None:
    known = _knowing(state, node)
    assert parse_expression(text, known).node == node, text


@pytest.mark.parametrize("setting", list(CONTEXTS))
def test_an_answer_reads_back_as_its_tree(setting):
    context = CONTEXTS[setting]
    for text, original in SAMPLE:
        state = replace(
            original,
            functions=dict(original.functions),
            variables=dict(original.variables),
            input_base=context.input_base,
        )
        try:
            parsed = parse_expression(text, state)
        except DeriveSyntaxError:
            continue
        for declaration in parsed.declarations:
            state.declare(declaration)
        computing = replace(context, withdrawn=state.withdrawn)
        answer = simplify(parsed.node, computing, state)
        if context.input_base != 10 and _has_decimal(answer.node):
            continue
        _reads_back(answer.node, answer.text, state)
        if answer.value is not None:
            exact = write_expression(answer.value)
            _reads_back(answer.value, exact, state)


x, y, k = sp.symbols("x y k", real=True)

#: Values of heads and forms the corpus sample may not reach.
VALUES = [
    sp.Rational(-3, 4) * x,
    PlusMinus(x) + 1,
    sp.zoo * x,
    -sp.oo,
    sp.nan,
    sp.Piecewise((x, x > 0), (-x, True)),
    sp.And(x > -2, x < 2),
    sp.Or(sp.And(sp.Eq(x, 1), sp.Eq(y, 2)), sp.Eq(x, 3)),
    sp.Not(sp.Eq(x, y)),
    sp.Implies(x > 0, y > 0),
    sp.Interval(0, 1),
    sp.EmptySet,
    Vector(x, Vector(1, 2), StringLiteral("text")),
    sp.Symbol("x SUB 1", real=True) * sp.Symbol("α", real=True),
    sp.Sum(x**k, (k, 0, sp.Symbol("n"))),
    sp.Integral(sp.exp(-(x**2)), (x, 0, sp.oo)),
    sp.Derivative(sp.Function("f")(x), x, 2),
    sp.Limit(sp.sin(x) / x, x, 0, "+"),
    sp.factorial(sp.factorial(x)),
    sp.exp(sp.exp(x)) + x ** (x**2) + x ** sp.sqrt(2),
    sp.Float("1.5e-7") * x,
    sp.Rational(1, 10**7),
    sp.Mul(2, 3, 3, evaluate=False),
    sp.RootSum(sp.Poly(x**5 + x + 1), sp.Lambda(x, x**2)),
    sp.CRootOf(x**5 + x + 1, 0),
]


@pytest.mark.parametrize("value", VALUES, ids=str)
@pytest.mark.parametrize("setting", ["exact", "decimal", "hexadecimal"])
def test_a_value_reads_back_as_its_tree(value, setting, unreadable_answers):
    context = CONTEXTS[setting]
    answer = from_sympy(value, context)
    if context.input_base != 10 and _has_decimal(answer.node):
        return
    _reads_back(answer.node, answer.text, ParseState(input_base=context.input_base))

