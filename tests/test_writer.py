"""Expression trees back to author notation.

Spellings are checked against what the original's Transfer Save writes. The round trip
is the writer's contract: every expression in both corpora must parse, write and parse
back to the same tree, so a saved worksheet reopens unchanged.
"""

from __future__ import annotations

import expr_cases
import pytest
from sexpr import to_sexpr
from test_expr_library import GROUPS, starting_state

from rederive.model.expr import Kind, Node
from rederive.syntax import (
    ParseState,
    SettingDeclaration,
    parse_expression,
    write_expression,
)

#: Authored text and what the original writes for it, each confirmed against the
#: original.
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
    # a nested run keeps the parentheses that mark it as separate
    ("(a-b)+c", "(a-b)+c"),
    ("a-b+c", "a-b+c"),
    ("(a*b)*c", "(a*b)*c"),
    # `/` closes the run to its left, so its position decides the parentheses
    ("a*b/c*d", "a*b/c*d"),
    ("a/b*c", "a/b*c"),
    ("a*(b/c)", "a*(b/c)"),
    ("(a/b)/c", "a/b/c"),
    ("a/(b/c)", "a/(b/c)"),
    ("p*(1+t)*((n-1)/i)", "p*(1+t)*((n-1)/i)"),
    # a sign is parenthesized except at the start of a run
    ("-x*y", "-x*y"),
    ("-a-b", "-a-b"),
    ("-(a+b)", "-(a+b)"),
    ("a*-b*c", "a*(-b)*c"),
    ("(-a)^2", "(-a)^2"),
    ("-a^2", "-a^2"),
    # powers: the exponent is parenthesized when it is signed or itself a power
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
    # word operators keep the spaces that keep them from fusing with operands
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
    # the dot product is the one operator that could fuse into a numeral
    ("2 . 3", "2 . 3"),
    ("[1,2] . [3,4]", "[1,2] . [3,4]"),
    # the rest are unchanged
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


SPACINGS = [
    ("IF(h <= 40, 10*h, 400 + 15*(h - 40))", "IF(h <= 40, 10*h, 400 + 15*(h - 40))"),
    ("x (x+1)-1", "x*(x + 1) - 1"),
    ("[1,2,3]", "[1, 2, 3]"),
    ("-x^2+2*x-1", "-x^2 + 2*x - 1"),
    ("a/(b+c)", "a/(b + c)"),
    ("f(x):=x^2", "F(x) := x^2"),
    # `NOT` is prefix, parenthesized only around looser-binding operands.
    ("NOT p AND q", "NOT p AND q"),
    ("NOT(p AND q)", "NOT (p AND q)"),
    ("NOT(x=y)", "NOT x = y"),
    # A chain of one word operator keeps only the parentheses the grammar needs: `AND`
    # groups to the right and `IMP` to the left.
    ("a OR b OR c", "a OR b OR c"),
    ("(a OR b) OR c", "(a OR b) OR c"),
    ("a AND (b OR c)", "a AND (b OR c)"),
    ("a IMP b IMP c", "(a IMP b) IMP c"),
    ("a IMP (b IMP c)", "a IMP (b IMP c)"),
    # A Greek letter is stored as its glyph and written as its name.
    ("alpha + 1", "alpha + 1"),
]


@pytest.mark.parametrize(
    ("authored", "expected"), SPACINGS, ids=[authored for authored, _ in SPACINGS]
)
def test_the_printers_spelling_puts_the_blanks_back(authored: str, expected: str):
    """`spaced` writes the text as a result line shows it."""
    state = ParseState()
    node = parse_expression(authored, state).node
    assert write_expression(node, spaced=True) == expected


def test_a_spaced_spelling_reads_back_as_what_was_written() -> None:
    """Spaces don't change meaning: every corpus case reparses to the same tree."""
    for case in CORPUS:
        state = case.state()
        node = parse_expression(case.text, state).node
        written = write_expression(node, spaced=True)
        again = parse_expression(written, state).node
        assert to_sexpr(normalized(again)) == to_sexpr(normalized(node)), (
            f"{case.text!r} was written {written!r}"
        )


def normalized(node: Node) -> Node:
    """`node` with the writer's three permitted changes applied.

    Writing is canonical: a juxtaposed product comes back with `*`, a bare application
    as a call, and `|u|` as `ABS(u)`. Spans and surface spellings are dropped too.
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
    """The written form of `text`, and the s-expressions before and after the trip."""
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
    """Every expression in the original's shipped utility files."""
    state = starting_state(group)
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
    """A numeral is written in its original digits, not as a decimal value.

    `0FF` is 255, but `255` would read back as a different number under the file's
    input base.
    """
    state = ParseState()
    state.declare(SettingDeclaration("InputBase", "16"))
    assert write_expression(parse_expression("0FF + 1", state).node) == "0FF+1"
