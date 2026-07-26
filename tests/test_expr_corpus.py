"""The conformance corpus: the parser is not done until it passes."""

from __future__ import annotations

import expr_cases
import pytest
from sexpr import to_sexpr

from rederive.syntax import DeriveSyntaxError, parse_expression

CASES = expr_cases.load()


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_corpus_case(case: expr_cases.Case) -> None:
    assert _parse(case) == case.expected


def _parse(case: expr_cases.Case) -> str:
    try:
        return to_sexpr(parse_expression(case.text, case.state()).node)
    except DeriveSyntaxError as error:
        return f"ERROR@{error.offset}"
