"""Expression trees back to author notation.

The round trip is the writer's contract: every expression in both corpora must parse,
write and parse back to the same tree, so a saved worksheet reopens unchanged.
"""

from __future__ import annotations

import expr_cases
import pytest
from sexpr import to_sexpr
from test_expr_library import GROUPS, refused, starting_state

from rederive.model.expr import Kind, Node
from rederive.syntax import (
    ParseState,
    SettingDeclaration,
    parse_expression,
    write_expression,
)

#: Authored text and how it is written.
SPELLINGS = [
    # juxtaposition becomes a `*`, with blanks around sum signs and relations and after
    # commas
    ("x (x + 1)", "x*(x + 1)"),
    ("3 x^2 - 2 x + 1", "3*x^2 - 2*x + 1"),
    ("1/(2 x)", "1/(2*x)"),
    ("2.5 + 1/2", "2.5 + 1/2"),
    ("a = b", "a = b"),
    ("a <= b", "a <= b"),
    ("[1, 2, 3]", "[1, 2, 3]"),
    ("INT(x^2, x)", "INT(x^2, x)"),
    ("x := 3", "x := 3"),
    # a name is written as it was declared, a built-in in upper case
    ("f(y) := y^2", "f(y) := y^2"),
    ("sin(y)+Sin(y)", "SIN(y) + SIN(y)"),
    # a bare application becomes a call, and `|u|` the call it means
    ("SIN x", "SIN(x)"),
    ("(SIN x)^2", "SIN(x)^2"),
    ("|x|", "ABS(x)"),
    # a nested run keeps the parentheses that mark it as separate
    ("(a-b)+c", "(a - b) + c"),
    ("a-b+c", "a - b + c"),
    ("(a*b)*c", "(a*b)*c"),
    # `/` closes the run to its left, so its position decides the parentheses
    ("a*b/c*d", "a*b/c*d"),
    ("a/b*c", "a/b*c"),
    ("a*(b/c)", "a*(b/c)"),
    ("(a/b)/c", "a/b/c"),
    ("a/(b/c)", "a/(b/c)"),
    ("p*(1+t)*((n-1)/i)", "p*(1 + t)*((n - 1)/i)"),
    # a sign is parenthesized except at the start of a run
    ("-x*y", "-x*y"),
    ("-a-b", "-a - b"),
    ("-(a+b)", "-(a + b)"),
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
    ("x SUB (a+b)", "x SUB (a + b)"),
    ("(a+b) SUB 2", "(a + b) SUB 2"),
    # a word operator gets only the parentheses the grammar needs: `AND` groups to the
    # right and `IMP` to the left
    ("a AND b OR c", "a AND b OR c"),
    ("a XOR b IMP c", "a XOR b IMP c"),
    ("a AND (b AND c)", "a AND b AND c"),
    ("(a OR b) OR c", "(a OR b) OR c"),
    ("a AND (b OR c)", "a AND (b OR c)"),
    ("a IMP b IMP c", "(a IMP b) IMP c"),
    ("a IMP (b IMP c)", "a IMP (b IMP c)"),
    # `NOT` is prefix, parenthesized only around looser-binding operands
    ("NOT a AND b", "NOT a AND b"),
    ("NOT (a=b)", "NOT a = b"),
    ("NOT(p AND q)", "NOT (p AND q)"),
    # plus-or-minus is written as the glyph
    ('"+-" x', "±x"),
    ('a + "+-" b', "a + (±b)"),
    # the dot product is the one operator that could fuse into a numeral
    ("2 . 3", "2 . 3"),
    ("[1,2] . [3,4]", "[1, 2] . [3, 4]"),
    # the rest are unchanged
    ("n!", "n!"),
    ("(a+b)!", "(a + b)!"),
    ("3!/2", "3!/2"),
    ("50%", "50%"),
    ("(x + 1)/(x - 1)", "(x + 1)/(x - 1)"),
    ("#e^(#i pi)", "#e^(#i*pi)"),
    ("SUM(k, k, 1, 10)", "SUM(k, k, 1, 10)"),
    ("IF(x > 0, 1, -1)", "IF(x > 0, 1, -1)"),
    ('"a string"', '"a string"'),
    ("x :epsilon Real (0, inf)", "x :epsilon Real (0, inf)"),
    ("[[1,2],[3,4]]", "[[1, 2], [3, 4]]"),
    ("-inf", "-inf"),
    # sets, and the set operators: the difference unspaced, the words spaced
    ("{1, ..., 10} \\ {2, 3, 5, 7}", "{1, ..., 10}\\{2, 3, 5, 7}"),
    ("{1} union {2} = {1, 2}", "{1} UNION {2} = {1, 2}"),
    # Derive 6's word operators, in upper case and spaced; glyphs as words
    ("p iff q", "p IFF q"),
    ("u ⊆ v", "u SUBSET v"),
    ("a UNION b SUBSET c", "a UNION b SUBSET c"),
    ("a - b DOTPRODUCT c + d", "a - b DOTPRODUCT c + d"),
    ("c - (-a) DOTPRODUCT b", "c - (-a DOTPRODUCT b)"),
    ("(a+b) CROSSPRODUCT c", "(a + b) CROSSPRODUCT c"),
    ("x deg", "x DEG"),
    ("100°", "100 DEG"),
    ("m ROW 2", "m ROW 2"),
    # an index is one primary, so a signed or larger one is parenthesized
    ("v SUB -1", "v SUB (-1)"),
    ("a SUB (i SUB j)", "a SUB (i SUB j)"),
    ("(a SUB i)!", "a SUB i!"),
    ("a SUB (i!)", "a SUB (i!)"),
    # `RETURN u` as the call, parenthesized where something follows it
    ("IF(v=[], RETURN w)", "IF(v = [], RETURN(w))"),
    ("(RETURN x)+1", "(RETURN(x)) + 1"),
    ("IF(i>n, exit)", "IF(i > n, exit)"),
    # assignments, updates and their targets; `:==` is `:=`
    ("k :+ 1", "k :+ 1"),
    ("w SUB j := q", "w SUB j := q"),
    ("x :== 3", "x := 3"),
    # matrix rows are written as nested vectors
    ("[a, b; c, d]", "[[a, b], [c, d]]"),
    ("m :epsilon Real [0, inf)", "m :epsilon Real [0, inf)"),
    ("x :epsilon Vector", "x :epsilon Vector"),
    # definition heads
    ("f(x, n := 0, tmp) := x + n", "f(x, n := 0, tmp) := x + n"),
    ("Block commands := commands SUB DIM(commands)", (
        "Block commands := commands SUB DIM(commands)"
    )),
    ("F() := 5", "F() := 5"),
    # the column subscript; a call indexed is parenthesized
    ("m↓↓2", "m SUB SUB 2"),
    ("FIRST(t) SUB 2", "(FIRST(t)) SUB 2"),
    ("SOLUTIONS(e, [a, b]) COL 1", "(SOLUTIONS(e, [a, b])) COL 1"),
    ("v SUB 1 SUB 1", "v SUB 1 SUB 1"),
    ("v SUB DIM(v)", "v SUB DIM(v)"),
    # an assignment inside a logical operator is parenthesized
    ("x=1 AND y=-2 OR x=3 AND y=0", "x = 1 AND y = -2 OR x = 3 AND y = 0"),
    ("(m:=v SUB 3) AND c:=v SUB 4", "(m := v SUB 3) AND (c := v SUB 4)"),
    ("[(m:=3) AND c:=4, 1]", "[(m := 3) AND (c := 4), 1]"),
    ("((m:=3) AND c:=4) OR d", "(m := 3) AND (c := 4) OR d"),
    # the quote, its operand parenthesized unless a primary
    ("BRUCH(a, b) := '(a/b)", "BRUCH(a, b) := '(a/b)"),
    ("s_ := INSERT('[a = q_*b + r_], s_, 0)", "s_ := INSERT('[a = q_*b + r_], s_, 0)"),
    ("[a = '(q_*b) + r_]", "[a = '(q_*b) + r_]"),
    ("'p^2", "'p^2"),
    ("'(-x)", "'(-x)"),
    ("''(1 + 2)", "''(1 + 2)"),
    ("'IF(x >= 0, g)", "'IF(x >= 0, g)"),
    ("'(RETURN x)", "'(RETURN(x))"),
    # the minus sign is the hyphen
    ("a − b", "a - b"),
    # a Greek letter is stored as its glyph and written as its name
    ("alpha + 1", "alpha + 1"),
]


@pytest.mark.parametrize(
    ("authored", "expected"), SPELLINGS, ids=[authored for authored, _ in SPELLINGS]
)
def test_the_original_writes_it_this_way(authored: str, expected: str) -> None:
    state = ParseState()
    assert write_expression(parse_expression(authored, state).node) == expected


#: Lines entered in turn, and how the last is written: names as they were declared,
#: built-ins in upper case, and old names of Derive's either replaced by the Derive 6
#: name or kept.
NAMED = [
    (["first([a,b,c])"], "FIRST([a, b, c])"),
    (["Transform(f):=f", "transform(x)+TRANSFORM(x)"], "Transform(x) + Transform(x)"),
    (["DIMENSION(v)"], "DIM(v)"),
    (["AUX(a,b):=", "CONV(a,x_):=[REVERSE_VECTOR(AUX(a,x_))]"], (
        "CONV(a, x_) := [REVERSE(AUX(a, x_))]"
    )),
    (["DELETE_ELEMENT(v,1)"], "DELETE(v, 1)"),
    (["REPLACE_ELEMENT(c,p,n)"], "REPLACE(c, p, n)"),
    (["IF(NUMBER(n),1,0)"], "IF(NUMBER(n), 1, 0)"),
    (["ELEMENT(v,2)"], "ELEMENT(v, 2)"),
    (['LOAD("D:\\DfD\\x.mth")'], 'LOAD("D:\\DfD\\x.mth")'),
    # A Greek variable is written by name, which reads back as the letter even where
    # the glyph would not: `Γ` is the gamma function and `Ω` a small omega.
    (["gamma+Omega+Gamma+Theta"], "gamma + Omega + Gamma + Theta"),
    (["gamma(k):=k+1"], "γ(k) := k + 1"),
]


@pytest.mark.parametrize(
    ("lines", "expected"), NAMED, ids=[str(lines) for lines, _ in NAMED]
)
def test_a_name_is_written_as_it_was_declared(lines: list[str], expected: str) -> None:
    state = ParseState()
    for line in lines:
        result = parse_expression(line, state)
        for declaration in result.declarations:
            state.declare(declaration)
    written = write_expression(result.node)
    assert written == expected
    assert write_expression(parse_expression(written, state).node) == expected


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
        if refused(text):
            continue
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
    assert write_expression(parse_expression("0FF + 1", state).node) == "0FF + 1"
