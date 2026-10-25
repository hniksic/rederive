"""Simplify's answers are fixed points of Simplify, over every case the suite states.

`test_simplify.py` already asks this of the inputs it lists in `EVERY_CASE`, under
whatever context each one uses. This asks it of every `simp(text) == expected` the
suite states with the default context, wherever it is stated - the parametrized lists
and the inline asserts of `test_simplify.py` and `test_manual.py` alike - and asks it
twice: the answer to an input simplifies to itself, and so does the expected text. The
second is the stronger of the two, because an expected value is written by hand, and a
form the engine would not keep is a form it could never have answered with.

The pairs are read out of the two files' source rather than imported, so that a case
added to either is covered here without being listed twice.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from rederive.engine.computing import Context, simplify
from rederive.syntax import ParseState, parse_expression

_HERE = Path(__file__).parent
_SOURCES = ("test_simplify.py", "test_manual.py")


def _is_text(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _simp_calls(node: ast.AST) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "simp"
    ]


def _pairs_in(path: Path) -> list[tuple[str, str]]:
    """The default-context (input, expected) pairs one test file states.

    A list counts where a parametrized test consumes it and calls `simp` with the text
    alone; an inline assert counts where it compares `simp` of one literal with another.
    """
    tree = ast.parse(path.read_text())
    consumed: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        calls = _simp_calls(node)
        if not calls or any(len(call.args) != 1 or call.keywords for call in calls):
            continue
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                consumed.update(
                    argument.id
                    for argument in decorator.args
                    if isinstance(argument, ast.Name)
                )
    pairs = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.List)
            and any(getattr(target, "id", None) in consumed for target in node.targets)
        ):
            pairs.extend(
                (element.elts[0].value, element.elts[1].value)
                for element in node.value.elts
                if isinstance(element, ast.Tuple)
                and len(element.elts) == 2
                and all(_is_text(part) for part in element.elts)
            )
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
            left = node.test.left
            if (
                isinstance(left, ast.Call)
                and getattr(left.func, "id", None) == "simp"
                and len(left.args) == 1
                and not left.keywords
                and _is_text(left.args[0])
                and len(node.test.ops) == 1
                and isinstance(node.test.ops[0], ast.Eq)
                and _is_text(node.test.comparators[0])
            ):
                pairs.append((left.args[0].value, node.test.comparators[0].value))
    return pairs


def _every_pair() -> list[tuple[str, str]]:
    seen: dict[tuple[str, str], None] = {}
    for name in _SOURCES:
        for pair in _pairs_in(_HERE / name):
            seen.setdefault(pair, None)
    return list(seen)


EVERY_PAIR = _every_pair()

_TRUTH_TABLE = "TRUTH_TABLE(p, q, p AND q, p OR q, p XOR q, p IMP q)"

#: Inputs whose answer a second Simplify changes, input to reason.
SETTLES_LATER = {
    "SOLVE(-2*x + 3*y <= 7, x)": (
        "the bound's quotient distributes on a second pass, where the original keeps "
        "(3*y - 7)/2 as read; sympy distributes a number over a sum as it builds one"
    ),
    _TRUTH_TABLE: (
        "a heading is simplified once the table is a matrix on the worksheet, which "
        "the original does too: p XOR q reads back as NOT p AND q OR p AND NOT q"
    ),
}

#: Expected texts that do not simplify to themselves beyond the answers of the inputs
#: above, expected text to reason. These are the manual's printed spellings that
#: `test_manual.py` records in `NEVER_TO_HOLD` and `NOT_YET_HELD` as decided or not yet
#: kept.
_BARS = "the manual's bars; the writer spells every absolute value ABS"
MANUAL_SPELLINGS = {
    "|x - y|/2 + (x + y)/2": _BARS,
    "(x + y)/2 - |x - y|/2": _BARS,
    "|m*n|/GCD(n, m)": _BARS,
    "|x|": _BARS,
    "(ERF(SQRT(2)*z/(2*s) - SQRT(2)*m/(2*s)) + 1)/2": (
        "the manual's preserved quotient, which sympy distributes"
    ),
    "1.732050807": "an approximation read back is the rational it prints as",
    "(2 - w)*(b - w) - 3*a": "the manual's factor order, which sympy sorts",
    "[3*(x + y)]": "the manual's undistributed coefficient, which sympy distributes",
}


#: Expected texts that do not simplify to themselves because the spelling Derive is
#: observed to print contradicts a rule it follows elsewhere, expected text to reason.
#: `test_simplify.py` records the same cases as not held.
CONFLICTING_SPELLINGS = {
    "1/2 - ERF(SQRT(2)*m/(2*s) - SQRT(2)*z/(2*s))/2": (
        "Derive's (ERF(...) + 1)/2 contradicts its semi-fraction rule elsewhere, "
        "(x + 1)^2/y + z -> (x^2 + 2*x + y*z + 1)/y"
    ),
}


def simp(text: str) -> str:
    return simplify(parse_expression(text, ParseState()).node, Context()).text


def _marked(inputs: dict[str, str], expectations: dict[str, str] | None = None) -> list:
    expectations = expectations or {}
    cases = []
    for text, expected in EVERY_PAIR:
        reason = inputs.get(text) or expectations.get(expected)
        marks = pytest.mark.xfail(reason=reason, strict=True) if reason else ()
        cases.append(pytest.param(text, expected, marks=marks))
    return cases


def test_the_pairs_are_found():
    """The collection reads something: an empty list would pass every case below."""
    assert len(EVERY_PAIR) > 500


@pytest.mark.parametrize(("text", "expected"), _marked(SETTLES_LATER), ids=str)
def test_simplifying_an_answer_again_changes_nothing(text, expected):
    once = simp(text)
    assert simp(once) == once


@pytest.mark.parametrize(
    ("text", "expected"),
    _marked(SETTLES_LATER, MANUAL_SPELLINGS | CONFLICTING_SPELLINGS),
    ids=str,
)
def test_an_expected_answer_simplifies_to_itself(text, expected):
    assert simp(expected) == expected
