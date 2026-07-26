"""Expression trees back to author notation.

Two things are checked. The spellings are what the original's Transfer Save
writes for each shape. The round trip is the writer's contract: every
expression in both corpora is parsed, written, and parsed again, and has to
come back as the tree it started as - which is how a saved worksheet is
guaranteed to reopen as itself.
"""

from __future__ import annotations

import expr_cases
import pytest
from sexpr import to_sexpr
from test_expr_library import GROUPS

from rederive.model.expr import Kind, Node
from rederive.syntax import (
    ParseState,
    SettingDeclaration,
    parse_expression,
    write_expression,
)

#: Authored, and what the original writes for it. Every pair was confirmed
#: against the original itself.
SPELLINGS = [
    # juxtaposition becomes a `*`, and the spacing goes
    ("x (x + 1)", "x*(x+1)"),
    ("3 x^2 - 2 x + 1", "3*x^2-2*x+1"),
    ("1/(2 x)", "1/(2*x)"),
    ("2.5 + 1/2", "2.5+1/2"),
    ("a = b", "a=b"),
    ("a <= b", "a<=b"),
    ("[1, 2, 3]", "[1,2,3]"),
    ("INT(x^2, x)", "INT(x^2,x)"),
    ("x := 3", "x:=3"),
    # a name is written as the symbol table spells it
    ("f(y) := y^2", "F(y):=y^2"),
    # a bare application becomes a call, and `|u|` the call it means
    ("SIN x", "SIN(x)"),
    ("(SIN x)^2", "SIN(x)^2"),
    ("|x|", "ABS(x)"),
    # a run keeps the fences that say it is one
    ("(a-b)+c", "(a-b)+c"),
    ("a-b+c", "a-b+c"),
    ("(a*b)*c", "(a*b)*c"),
    # `/` closes the run to its left, so where it stands decides the fences
    ("a*b/c*d", "a*b/c*d"),
    ("a/b*c", "a/b*c"),
    ("a*(b/c)", "a*(b/c)"),
    ("(a/b)/c", "a/b/c"),
    ("a/(b/c)", "a/(b/c)"),
    ("p*(1+t)*((n-1)/i)", "p*(1+t)*((n-1)/i)"),
    # a sign is fenced anywhere but at the head of a run
    ("-x*y", "-x*y"),
    ("-a-b", "-a-b"),
    ("-(a+b)", "-(a+b)"),
    ("a*-b*c", "a*(-b)*c"),
    ("(-a)^2", "(-a)^2"),
    ("-a^2", "-a^2"),
    # powers: the exponent is fenced when it is a sign or a power of its own
    ("a^b^c", "a^(b^c)"),
    ("a^-b", "a^(-b)"),
    ("(a^b)^c", "(a^b)^c"),
    ("a^(b*c)", "a^(b*c)"),
    ("a^SIN(x)", "a^SIN(x)"),
    ("a^x!", "a^x!"),
    ("a^(x SUB 2)", "a^x SUB 2"),
    ("2^(1/2)", "2^(1/2)"),
    ("a*b^c", "a*b^c"),
    ("(a*b)^c", "(a*b)^c"),
    # the word operators keep the blanks that stop them fusing
    ("x SUB 2", "x SUB 2"),
    ("a SUB b SUB c", "a SUB b SUB c"),
    ("x SUB (a+b)", "x SUB (a+b)"),
    ("(a+b) SUB 2", "(a+b) SUB 2"),
    ("a AND b OR c", "a AND b OR c"),
    ("a XOR b IMP c", "a XOR b IMP c"),
    ("a AND (b AND c)", "a AND (b AND c)"),
    ("NOT a AND b", "NOT(a) AND b"),
    ("NOT (a=b)", "NOT(a=b)"),
    # plus-or-minus is written as it is typed, quotes and all
    ('"+-" x', '"+-"x'),
    ('a + "+-" b', 'a+("+-"b)'),
    # the dot product is the one operator that would fuse into a numeral
    ("2 . 3", "2 . 3"),
    ("[1,2] . [3,4]", "[1,2] . [3,4]"),
    # and the rest, left as they are
    ("n!", "n!"),
    ("(a+b)!", "(a+b)!"),
    ("3!/2", "3!/2"),
    ("50%", "50%"),
    ("(x + 1)/(x - 1)", "(x+1)/(x-1)"),
    ("#e^(#i pi)", "#e^(#i*pi)"),
    ("SUM(k, k, 1, 10)", "SUM(k,k,1,10)"),
    ("IF(x > 0, 1, -1)", "IF(x>0,1,-1)"),
    ('"a string"', '"a string"'),
    ("x :epsilon Real (0, inf)", "x:epsilonReal (0, inf)"),
    ("[[1,2],[3,4]]", "[[1,2],[3,4]]"),
    ("-inf", "-inf"),
]


@pytest.mark.parametrize(
    ("authored", "expected"), SPELLINGS, ids=[authored for authored, _ in SPELLINGS]
)
def test_the_original_writes_it_this_way(authored: str, expected: str) -> None:
    state = ParseState()
    assert write_expression(parse_expression(authored, state).node) == expected


def normalized(node: Node) -> Node:
    """`node` with the three differences the writer is allowed, applied.

    Writing is canonical, so three kinds of node come back as the form they
    mean rather than the one they were built from: a juxtaposed product comes
    back written `*`, a bare application comes back a call, and `|u|` comes
    back `ABS(u)`. Spans and surface spellings go too, neither being part of
    what an expression is.
    """
    children = tuple(normalized(child) for child in node.children)
    kind = node.kind
    if kind is Kind.APPLY:
        kind = Kind.CALL
    elif kind is Kind.ABS:
        kind = Kind.CALL
        children = (Node(Kind.NAME, 0, 0, (), "ABS"), *children)
    return Node(kind, 0, 0, children, node.value, None)


def round_trip(text: str, state: ParseState) -> tuple[str, str, str]:
    """What `text` parses to, what that writes as, and what that parses to."""
    node = parse_expression(text, state).node
    written = write_expression(node)
    again = parse_expression(written, state).node
    return written, to_sexpr(normalized(node)), to_sexpr(normalized(again))


CORPUS = [case for case in expr_cases.load() if not case.expected.startswith("ERROR")]


@pytest.mark.parametrize("case", CORPUS, ids=[case.id for case in CORPUS])
def test_a_conformance_case_survives_being_written(case: expr_cases.Case) -> None:
    written, before, after = round_trip(case.text, case.state())
    assert after == before, f"{case.text!r} was written {written!r}"


@pytest.mark.parametrize("group", GROUPS, ids=list(GROUPS))
def test_a_shipped_file_survives_being_written(group: str) -> None:
    """Every expression of the utility files the original shipped."""
    state = ParseState()
    for text in GROUPS[group]:
        result = parse_expression(text, state)
        for declaration in result.declarations:
            state.declare(declaration)
        written, before, after = round_trip(text, state)
        assert after == before, f"{text!r} was written {written!r}"


def test_what_is_written_is_written_the_same_way_twice() -> None:
    """The canonical form is a fixed point: saving a loaded file changes nothing."""
    state = ParseState()
    for text in ("x (x + 1)", "SIN x", "|x|", "a*-b*c", "(a-b)+c", "a^b^c"):
        once = write_expression(parse_expression(text, state).node)
        assert write_expression(parse_expression(once, state).node) == once


def test_a_numeral_is_written_in_the_digits_it_was_typed_in() -> None:
    """A numeral's value is decimal, but the digits are what the file holds.

    `0FF` is 255, and writing `255` would read back as a different number under
    the base the file itself sets.
    """
    state = ParseState()
    state.declare(SettingDeclaration("InputBase", "16"))
    assert write_expression(parse_expression("0FF + 1", state).node) == "0FF+1"
