"""Every expression Derive shipped in its utility files must parse.

Breadth, where `test_expr_corpus.py` is depth: these are thousands of real
expressions, checked only for being accepted and for the declarations they make
carrying over to the expressions after them.
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

#: The utility files a demo loads before its first line, demo to files, where they are
#: not just the demo's namesake. A demo calls the functions those files define, and
#: Derive reads a call to a function it does not know as a product, as it reads
#: `foo(x)`, so without them a demo's one-argument calls would be products here too.
LOADS: dict[str, tuple[str, ...]] = {
    f"{_USERS}FIT_.DMO": (f"{_USERS}FIT_.MTH", "derive-3.14-dos/APPROX.MTH"),
    f"{_USERS}TENSOR1.DMO": (f"{_USERS}TENSOR.MTH",),
    f"{_USERS}TENSOR2.DMO": (f"{_USERS}TENSOR.MTH",),
}

#: The functions a file written for a later Derive calls as built in, file to names.
#: The original has none of them, so they are declared the way the session the file
#: was written in knew them.
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
    """The symbol table a group's first line is read with: the files it loads, read
    first, and the functions of a later Derive it calls."""
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
