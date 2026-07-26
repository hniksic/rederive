"""Every expression Derive shipped in its utility files must parse.

Breadth, where `test_expr_corpus.py` is depth: these are thousands of real
expressions, checked only for being accepted and for the declarations they make
carrying over to the expressions after them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rederive.syntax import DeriveSyntaxError, ParseState, parse_expression

LIBRARY = Path(__file__).parent / "corpus" / "library-expressions.txt"


def load(path: Path = LIBRARY) -> dict[str, list[str]]:
    """The expressions of each `##` group, in file order."""
    groups: dict[str, list[str]] = {}
    current: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("##"):
            current = groups.setdefault(line[2:].strip(), [])
        elif line.strip() and not line.startswith("#"):
            current.append(line)
    return groups


GROUPS = load()


@pytest.mark.parametrize("group", GROUPS, ids=list(GROUPS))
def test_a_shipped_file_parses(group: str) -> None:
    state = ParseState()
    for expression in GROUPS[group]:
        try:
            result = parse_expression(expression, state)
        except DeriveSyntaxError as error:
            raise AssertionError(
                f"{group}: rejected at offset {error.offset}\n"
                f"  {expression}\n"
                f"  {' ' * error.offset}^"
            ) from None
        for declaration in result.declarations:
            state.declare(declaration)


def test_the_library_is_substantial() -> None:
    assert sum(len(expressions) for expressions in GROUPS.values()) > 1500
