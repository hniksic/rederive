"""Laying written text out on rows for the Author line and the clipboard."""

from pathlib import Path

import pytest

from rederive.model.session import Session
from rederive.syntax import DeriveSyntaxError, parse_expression, write_expression, written
from rederive.syntax.layout import laid_out

LIBRARY = Path(__file__).parent / "corpus" / "library-expressions.txt"


def layout(text, width=60, before=""):
    session = Session()
    node = parse_expression(text, session.state).node
    written_text, tree = written(node)
    return laid_out(written_text, tree, width, before)


def test_what_fits_stays_on_one_row():
    assert layout("IF(x > 0, 1, 2)") == "IF(x > 0, 1, 2)"


def test_a_list_too_wide_puts_each_item_on_a_row_a_level_deeper():
    assert layout("[111111, 222222, 333333]", width=20) == (
        "[\n  111111,\n  222222,\n  333333]"
    )


def test_a_program_is_laid_out_as_the_worksheet_shows_it():
    # The example from Derive 6's help on procedural programming, which displays it in
    # the same shape without the brackets and commas.
    rev = "Rev(v,w) := PROG(w:=[], LOOP(IF(v=[], RETURN w), w:=ADJOIN(FIRST(v),w), v:=REST(v)))"
    assert layout(rev) == (
        "Rev(v, w) :=\n"
        "  PROG(\n"
        "    w := [],\n"
        "    LOOP(\n"
        "      IF(v = [], RETURN(w)),\n"
        "      w := ADJOIN(FIRST(v), w),\n"
        "      v := REST(v)))"
    )


def test_a_program_of_one_statement_that_fits_stays_on_its_row():
    assert layout("f(n) := PROG(RETURN n)") == "f(n) := PROG(RETURN(n))"


def test_rows_are_indented_from_the_row_the_text_continues():
    assert layout("PROG(a, b)", before="x :=\n  IF(c, ") == "PROG(\n    a,\n    b)"


def test_what_cannot_be_broken_stays_long():
    assert layout("a + b + c + d + e + f + g", width=5) == "a + b + c + d + e + f + g"


def _library():
    for line in LIBRARY.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            yield line


def test_laid_out_text_reads_back_as_the_same_expression():
    # Narrow, so that most lists are broken. Only blanks change, and a line break is a
    # blank to the parser.
    session = Session()
    checked = 0
    for line in _library():
        try:
            node = parse_expression(line, session.state).node
        except DeriveSyntaxError:
            continue
        text, tree = written(node)
        rows = laid_out(text, tree, 20)
        assert write_expression(parse_expression(rows, session.state).node) == text
        checked += 1
    assert checked > 1000
