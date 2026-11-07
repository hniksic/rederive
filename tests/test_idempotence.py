"""Simplify's answers are fixed points of Simplify, over every case the suite states.

`test_simplify.py` already asks this of the inputs it lists in `EVERY_CASE`, under
whatever context each one uses. This asks it of every `simp(text) == expected` the
suite states with the default context, wherever it is stated - the parametrized lists
and the inline asserts of `test_simplify.py`, `test_calculus.py`, `test_matrices.py`,
`test_logic.py`, `test_approximate.py`, `test_functions.py`, `test_integrate.py` and
`test_manual.py` alike: the expected text simplifies to itself. An expected value is
written by hand, and a form the engine would not keep is a form it could never have
answered with.

That the answer to an input simplifies to itself is the same question, where the case
holds: its test says the answer is the expected text, and this one says the expected
text stays put. So the answer is simplified again only for the cases their own files
record as not held - the manual's spellings `test_manual.py` lists in `NEVER_TO_HOLD`
and `NOT_YET_HELD`, and the ones `test_functions.py` keeps under a strict xfail - whose
answer may be something other than the expected text, and so a form of its own to ask
about. Asking it of every case would simplify every input a second time, on top of
the run its own test gives it, for an answer the two tests together already know.

The pairs are read out of the files' source rather than imported, so that a case
added to any of them is covered here without being listed twice; what each file
records as not held is imported from it, so that a case struck off there is asked here
again as it should be.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from rederive.engine.computing import Context, simplify
from rederive.syntax import FunctionDeclaration, ParseState, parse_expression

import test_functions
import test_manual

_HERE = Path(__file__).parent
_SOURCES = (
    "test_simplify.py",
    "test_calculus.py",
    "test_matrices.py",
    "test_logic.py",
    "test_approximate.py",
    "test_functions.py",
    "test_integrate.py",
    "test_manual.py",
)

#: A default-context case, as this module reads it: input, expected text, the
#: functions the input is read with declared, and the id of the test that states it.
Pair = tuple[str, str, tuple[str, ...], str]


def _is_text(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _simp_calls(node: ast.AST) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "simp"
    ]


def _declared_in(function: ast.FunctionDef) -> dict[str, tuple[str, ...]]:
    """The states a test builds with `declaring`, by the name it binds each one to."""
    found = {}
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "declaring"
            and all(_is_text(argument) for argument in node.value.args)
        ):
            found[node.targets[0].id] = tuple(arg.value for arg in node.value.args)
    return found


def _functions_read_with(
    call: ast.Call, declared: dict[str, tuple[str, ...]]
) -> tuple[str, ...] | None:
    """The functions a `simp` call of one literal reads its text with, or None where
    it is given a context, or a state that is not one `declaring` built."""
    if len(call.args) != 1 or not _is_text(call.args[0]):
        return None
    if not call.keywords:
        return ()
    if len(call.keywords) == 1 and call.keywords[0].arg == "state":
        value = call.keywords[0].value
        if isinstance(value, ast.Name) and value.id in declared:
            return declared[value.id]
    return None


def _pairs_in(path: Path) -> list[Pair]:
    """The default-context cases one test file states.

    A list counts where a parametrized test consumes it and calls `simp` with the text
    alone; an inline assert counts where it compares `simp` of one literal with another.
    The functions are those the input is read with declared, which an inline assert
    names by a state `declaring` built in the same test. A case out of a list is
    identified the way pytest identifies it, by the test's name and the `ids=str` of
    its two texts; an inline assert by the test's name alone.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    consumed: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        calls = _simp_calls(node)
        if not calls or any(len(call.args) != 1 or call.keywords for call in calls):
            continue
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                consumed.update(
                    (argument.id, node.name)
                    for argument in decorator.args
                    if isinstance(argument, ast.Name)
                )
    pairs = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.List)):
            continue
        names = (getattr(target, "id", None) for target in node.targets)
        tests = [consumed[name] for name in names if name in consumed]
        if not tests:
            continue
        for element in node.value.elts:
            if (
                isinstance(element, ast.Tuple)
                and len(element.elts) == 2
                and all(_is_text(part) for part in element.elts)
            ):
                text, expected = (part.value for part in element.elts)
                pairs.append((text, expected, (), f"{tests[0]}[{text}-{expected}]"))
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        declared = _declared_in(function)
        for node in ast.walk(function):
            if not (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare)):
                continue
            left = node.test.left
            if not (
                isinstance(left, ast.Call)
                and getattr(left.func, "id", None) == "simp"
                and len(node.test.ops) == 1
                and isinstance(node.test.ops[0], ast.Eq)
                and _is_text(node.test.comparators[0])
            ):
                continue
            functions = _functions_read_with(left, declared)
            if functions is not None:
                expected = node.test.comparators[0].value
                pairs.append((left.args[0].value, expected, functions, function.name))
    return pairs


def _every_pair() -> list[Pair]:
    seen: dict[tuple[str, str, tuple[str, ...]], Pair] = {}
    for name in _SOURCES:
        for pair in _pairs_in(_HERE / name):
            seen.setdefault(pair[:3], pair)
    return list(seen.values())


EVERY_PAIR = _every_pair()


def _unheld(pair: Pair) -> bool:
    """Whether the test stating this case expects it to fail, so that the answer to
    the input is not the expected text and is to be asked about on its own."""
    text, _, _, test_id = pair
    return (
        test_id in test_manual.NEVER_TO_HOLD
        or test_id in test_manual.NOT_YET_HELD
        or text in test_functions.INDICATOR_AND_NORMAL_NOT_HELD
    )


#: The cases whose answer is not the expected text, by their own tests' account.
UNHELD_PAIRS = [pair for pair in EVERY_PAIR if _unheld(pair)]

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
#: `test_functions.py` records the same cases as not held.
CONFLICTING_SPELLINGS = {
    "1/2 - ERF(SQRT(2)*m/(2*s) - SQRT(2)*z/(2*s))/2": (
        "Derive's (ERF(...) + 1)/2 contradicts its semi-fraction rule elsewhere, "
        "(x + 1)^2/y + z -> (x^2 + 2*x + y*z + 1)/y"
    ),
}


def simp(text: str, functions: tuple[str, ...] = ()) -> str:
    state = ParseState()
    for name in functions:
        state.declare(FunctionDeclaration(name))
    return simplify(parse_expression(text, state).node, Context(), state).text


def _marked(
    pairs: list[Pair],
    inputs: dict[str, str],
    expectations: dict[str, str] | None = None,
) -> list:
    expectations = expectations or {}
    cases = []
    for text, expected, functions, _ in pairs:
        reason = inputs.get(text) or expectations.get(expected)
        marks = pytest.mark.xfail(reason=reason, strict=True) if reason else ()
        cases.append(pytest.param(text, expected, functions, marks=marks))
    return cases


def test_the_pairs_are_found():
    """The collection reads something: an empty list would pass every case below,
    and the record of what is not held is read against the ids it names."""
    assert len(EVERY_PAIR) > 500
    assert len(UNHELD_PAIRS) > 5


@pytest.mark.parametrize(
    ("text", "expected", "functions"), _marked(UNHELD_PAIRS, SETTLES_LATER), ids=str
)
def test_simplifying_an_unheld_answer_again_changes_nothing(text, expected, functions):
    once = simp(text, functions)
    assert simp(once, functions) == once


@pytest.mark.parametrize(
    ("text", "expected", "functions"),
    _marked(EVERY_PAIR, SETTLES_LATER, MANUAL_SPELLINGS | CONFLICTING_SPELLINGS),
    ids=str,
)
def test_an_expected_answer_simplifies_to_itself(text, expected, functions):
    assert simp(expected, functions) == expected
