"""Both invariants, over every expression the project has on file.

`test_simplify.py` is where the mathematics is: inline cases, each one naming
what it is about. This is the net underneath it - every expression in
`corpus/`, which is Derive's shipped utility files, its demo scripts and the
parser's own cases, some three thousand of them - each simplified, printed,
read back and simplified again. What it catches is a change that breaks
something no inline case covers. What it reports is the line it broke on, which
is a starting point and not a diagnosis: the fix is usually a new inline case
in `test_simplify.py` and then the code to pass it.

About a minute across every core, where the rest of the suite is forty seconds,
so it is marked and left out of the default run:

    uv run pytest -m slow -n auto

One case per expression, not per file. The cost of an expression runs from a
microsecond to ten seconds, and a case per file would leave one worker grinding
through the hypergeometric integrals while the others sat idle; a case apiece
gives the distributor something to distribute, and the tail is one expression
rather than one file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rederive.engine.computing import (
    Context,
    authored_conditionals,
    from_sympy,
    simplify,
    to_sympy,
)
from rederive.syntax import DeriveSyntaxError, ParseState, parse_expression

CORPUS = Path(__file__).parent / "corpus"

#: The expressions whose written form settles only on a second Simplify, each
#: by the name it is defined under. Every one is a correct answer written in an
#: order it does not keep, and every one is expected here exactly: one that
#: starts settling fails this test as loudly as one that stops, since a record
#: nobody prunes is a record nobody trusts.
#:
#: * `BESSEL_Y_SERIES` writes `2/pi*(u - v/2)` with the two outside the sum and
#:   distributes it into the sum on the second pass. Distributing it in
#:   `_canonical` is not the answer: it settles this one and unsettles two
#:   others, where sympy then picks the other sign for a cosine's argument or
#:   multiplies out a term it had left folded.
#: * `LIN2_CCF_HOM` writes `(p^2/4 - d/4)^(x/2)` as it stands and
#:   `(p^2 - d)^(x/2)/2^x` on the second pass: splitting the base did not pay
#:   until the form it was offered had changed.
UNSETTLED = (
    "BESSEL_Y_SERIES(",
    "LIN2_CCF_HOM(",
)


def _library() -> dict[str, list[str]]:
    """The shipped utility files, by the `##` group each was read from."""
    groups: dict[str, list[str]] = {}
    current: list[str] = []
    for line in (CORPUS / "library-expressions.txt").read_text().splitlines():
        if line.startswith("##"):
            current = groups.setdefault(line[2:].strip(), [])
        elif line.strip() and not line.startswith("#"):
            current.append(line)
    return groups


def _parser_cases() -> list[str]:
    """The parser corpus, which writes an expression and its tree per line."""
    lines = []
    for line in (CORPUS / "parser-cases.txt").read_text().splitlines():
        text = line.split("=>")[0].strip()
        if text and not text.startswith("#"):
            lines.append(text)
    return lines


def _groups() -> dict[str, list[str]]:
    groups = dict(_library())
    groups["parser-cases"] = _parser_cases()
    return groups


GROUPS = _groups()


def _settles(text: str) -> bool:
    """Whether simplifying `text` reaches a form that stays where it is.

    The two invariants at once: printing the answer is a fixed point, and
    simplifying it again changes nothing. A line the parser rejects is not
    this module's business - `test_expr_library.py` is where that is checked.

    Printing takes the answer's own conditionals with it. An undecidable `IF`
    is shown as it was written rather than as it was converted, so what its
    arms are spelled as is not a property of the expression and no print of one
    can recover it; the record is part of how such an answer is printed at all,
    and printing without it would be asking a different question.
    """
    state = ParseState()
    try:
        parsed = parse_expression(text, state)
    except DeriveSyntaxError:
        return True
    # What the line declares, before it is simplified, because that is the
    # order a session does it in: `X := ELEMENT(x, 1)` declares `X`, and only
    # then is the `x` in it known to be the same variable.
    for declaration in parsed.declarations:
        state.declare(declaration)
    node = parsed.node
    context = Context()
    once = simplify(node, context, state)
    authored = authored_conditionals(once.node, context)
    printed = from_sympy(to_sympy(once.node, context), context, state, authored).text
    return printed == once.text == simplify(once.node, context, state).text


CASES = [(group, text) for group in sorted(GROUPS) for text in GROUPS[group]]


@pytest.mark.slow
@pytest.mark.parametrize(
    ("group", "text"), CASES, ids=[f"{group}: {text[:60]}" for group, text in CASES]
)
def test_a_shipped_expression_settles(group: str, text: str) -> None:
    """Both invariants over one expression, and totality along with them.

    Nothing here catches an exception, so Simplify raising on something the
    parser accepted fails this as an error with its traceback, which is what
    the promise that it never raises looks like from the outside.
    """
    assert _settles(text) is not text.startswith(UNSETTLED)


def test_the_record_of_unsettled_expressions_is_current() -> None:
    # Cheap enough for the default run, and it is what keeps the record above
    # from naming an expression the corpus no longer holds.
    for name in UNSETTLED:
        assert any(
            text.startswith(name) for texts in GROUPS.values() for text in texts
        ), name
