"""Every expression Derive shipped in its utility files must parse.

Unlike the in-depth `test_expr_corpus.py`, this covers thousands of real expressions,
checking only that they're accepted and that their declarations carry over to later
expressions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rederive.syntax import (
    DeriveSyntaxError,
    FunctionDeclaration,
    ParseState,
    parse_expression,
)

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

_USERS = "derive-3.14-dos/USERS/"

#: Utility files each demo loads first, where they differ from the demo's namesake.
#: Without them, calls to the functions they define would parse as products, the way
#: Derive reads `foo(x)` for an unknown `foo`.
LOADS: dict[str, tuple[str, ...]] = {
    f"{_USERS}FIT_.DMO": (f"{_USERS}FIT_.MTH", "derive-3.14-dos/APPROX.MTH"),
    f"{_USERS}TENSOR1.DMO": (f"{_USERS}TENSOR.MTH",),
    f"{_USERS}TENSOR2.DMO": (f"{_USERS}TENSOR.MTH",),
}

#: Functions that files written for a later Derive call as built-ins. The original
#: lacks them, so they're declared up front.
LATER_BUILTINS: dict[str, tuple[str, ...]] = {
    "hawaii-coursework/add-head.mth": ("LOAD",),
    "hawaii-coursework/add-util.mth": (
        "DIM",
        "DSOLVE2_IV",
        "PROG",
        "SOLUTIONS",
        "SUBST",
    ),
}


def _replay(state: ParseState, expressions: list[str]) -> None:
    for expression in expressions:
        for declaration in parse_expression(expression, state).declarations:
            state.declare(declaration)


def starting_state(group: str) -> ParseState:
    """Return the symbol table for parsing a group: its loaded files and any later-Derive
    built-ins it calls."""
    state = ParseState()
    for name in LATER_BUILTINS.get(group, ()):
        state.declare(FunctionDeclaration(name, has_body=True))
    if group.endswith(".DMO"):
        companion = f"{group.removesuffix('.DMO')}.MTH"
        loaded = LOADS.get(group, (companion,) if companion in GROUPS else ())
        for file in loaded:
            _replay(state, GROUPS[file])
    return state


@pytest.mark.parametrize("group", GROUPS, ids=list(GROUPS))
def test_a_shipped_file_parses(group: str) -> None:
    state = starting_state(group)
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
