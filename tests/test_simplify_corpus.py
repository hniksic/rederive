"""The Simplify invariants (print round-trip and idempotence) over the whole corpus.

The mathematics is tested by inline cases in `test_simplify.py`, `test_calculus.py`,
`test_matrices.py`, `test_logic.py`, `test_approximate.py` and `test_functions.py`. This
is a safety net beneath them: every line of `corpus/` (Derive's utility files, demo
scripts and the parser's cases, about three thousand lines) is simplified, printed,
parsed back and simplified again. It catches breakage no inline case covers. A failing
line is a starting point, not a diagnosis; the fix is usually a new inline case in the
relevant file, then the code to pass it.

It takes about twenty seconds across all cores (the rest of the suite takes
twenty-five) and runs by default, since skipping it would give green runs that never
checked.

Cases are per expression, not per file: costs range from a microsecond to ten seconds,
and per-file cases would leave one worker stuck on the hypergeometric integrals while
others idle.

Expressions are deduplicated by shape (see `_shape`): names and numbers are dropped and
only the first nine nodes are considered. For example, the six `FUN.MTH` definitions of
the form `<coefficient>*DIF(z^(a-1)*..., z, ...)` are one case, and the forty-four
`<unit>:=<n>*<unit>` definitions another. This drops eleven hundred of the twenty-seven
hundred lines and nearly a third of the run time.

The trade-off is real: lines of one shape are different expressions (the kept case may
have `LN(1 - z)` where a dropped one had `ASIN(SQRT(z))`), so a bug only the dropped one
would hit goes unnoticed. The loss is limited because shapes keep function names
(`SIN(x)` and `COS(x)` differ), and because nine nodes was chosen as the coarsest cut
at which no shape mixes lines that settle with lines that don't. `test_expr_library.py`
still parses every line.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import expr_cases
import pytest

from rederive.engine.computing import (
    Context,
    authored_conditionals,
    from_sympy,
    simplify,
    to_sympy,
)
from rederive.model.expr import Kind, Node
from rederive.syntax import DeriveSyntaxError, ParseState, parse_expression

CORPUS = Path(__file__).parent / "corpus"

#: Names of expressions that settle only on a second Simplify. The list must be exact:
#: one that starts settling fails just like one that stops, so the list stays accurate.
#: Currently empty.
UNSETTLED: tuple[str, ...] = ()


def _library() -> dict[str, list[tuple[str, ParseState]]]:
    """The shipped utility file lines, grouped by their `##` header.

    Nothing in these files sets a mode, so every line is parsed with a fresh state.
    """
    groups: dict[str, list[tuple[str, ParseState]]] = {}
    current: list[tuple[str, ParseState]] = []
    for line in (CORPUS / "library-expressions.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("##"):
            current = groups.setdefault(line[2:].strip(), [])
        elif line.strip() and not line.startswith("#"):
            current.append((line, ParseState()))
    return groups


def _parser_cases() -> list[tuple[str, ParseState]]:
    """The conformance corpus, read with the same loader as `test_expr_corpus.py`.

    The format has directives, group headers and comments besides expressions, and some
    cases themselves start with `#` (as in `#e` and `#i`), so it is best parsed in one
    place.

    Each case gets the state its `[state:]` and `[declared:]` directives set, since
    parsing depends on them: `gh(1,2)` is a call only if `GH` is declared, and `lhs-rhs`
    is a subtraction only in Sensitive case mode. The input base is reset to ten because
    output is always decimal: under `InputBase=16`, `2A` is 42, prints as `42`, and
    reads back as 66. That's a notation issue, not a Simplify one.
    """
    cases: list[tuple[str, ParseState]] = []
    for case in expr_cases.load():
        state = case.state()
        state.input_base = 10
        cases.append((case.text, state))
    return cases


def _groups() -> dict[str, list[tuple[str, ParseState]]]:
    groups = dict(_library())
    groups["parser-cases"] = _parser_cases()
    return groups


GROUPS = _groups()


def _settles(text: str, state: ParseState) -> bool:
    """Whether simplifying `text` reaches a stable form.

    Checks both invariants: the answer's printed form round-trips, and simplifying it
    again changes nothing. The same `state` parses the line and the reprinted answer,
    as in one session. Lines the parser rejects are skipped; `test_expr_library.py`
    covers those.

    Printing passes the answer's authored conditionals along: an undecidable `IF` is
    shown as written, not as converted, and its original spelling can't be recovered
    from the converted form.
    """
    try:
        parsed = parse_expression(text, state)
    except DeriveSyntaxError:
        return True
    # Apply the line's declarations before simplifying, as a session does:
    # `X := ELEMENT(x, 1)` declares `X`, and only then is its `x` known to be the
    # same variable.
    for declaration in parsed.declarations:
        state.declare(declaration)
    node = parsed.node
    context = Context()
    once = simplify(node, context, state)
    authored = authored_conditionals(once.node, context)
    printed = from_sympy(to_sympy(once.node, context), context, state, authored).text
    return printed == once.text == simplify(once.node, context, state).text


Case = tuple[str, str, ParseState]

#: Leaf kinds whose values a shape ignores. Variables, numbers and parameter names are
#: mostly what distinguishes lines of one family.
ANONYMOUS = (Kind.NUMBER, Kind.NAME)

#: Kinds whose operands are treated as an unordered set. `2*SQRT(pi)*c!/d*DIF(u)` and
#: `-c!/d*DIF(u)` differ only in factor order and count, so they get one shape.
COMMUTATIVE = (Kind.SUM, Kind.PRODUCT, Kind.AND, Kind.OR, Kind.XOR)

#: Kinds whose first child is the function, not an argument. The function name is the
#: one leaf a shape keeps, since different functions take different paths through the
#: engine: `SIN(x)` and `LN(x)` are two shapes.
APPLICATIONS = (Kind.CALL, Kind.APPLY, Kind.FUNCPOW)

#: How many nodes, breadth first, a shape covers. Nine reaches a definition's outline
#: (header, body operator, and the body's terms or factors) and stops above the
#: arguments, where family members differ. It is tuned against the full corpus as the
#: smallest value at which no shape mixes settling and unsettling lines. At eight,
#: Kummer's function joins `BESSEL_Y_SERIES`, which doesn't settle, and Derive's two-line
#: ODE solvers join `LIN2_CCF_HOM`, which doesn't either.
OUTLINE = 9


def _outline(node: Node) -> tuple[object, ...]:
    """`node`'s structure as nested `(tag, operands)` tuples.

    Drops names and numbers, offsets and `surface` spellings. A definition's own name
    is dropped too, so `F_1A(a,b,c,z):=` and `F_1B(a,b,c,z):=` match, and so is a
    leading minus, which is just a coefficient.

    Keeps the operators, function names, nesting and operand counts; for commutative
    kinds, the set of distinct operands, ignoring order and repeats.
    """
    if node.kind in ANONYMOUS:
        return ((node.kind,), ())
    if node.kind is Kind.UNOP and node.value == "-":
        return _outline(node.children[0])
    children = node.children
    tag: tuple[object, ...] = (node.kind, node.value)
    if node.kind is Kind.FUNDEF:
        tag = (node.kind,)
    elif node.kind in APPLICATIONS and children and children[0].kind is Kind.NAME:
        tag = (node.kind, children[0].value)
        children = children[1:]
    operands = [_outline(child) for child in children]
    if node.kind in COMMUTATIVE:
        return ((node.kind,), tuple(sorted(set(operands), key=repr)))
    return (tag, tuple(operands))


def _shape(node: Node) -> tuple[object, ...]:
    """`node`'s outline truncated to its first `OUTLINE` nodes, breadth first.

    Breadth first, so a shape covers the top of the expression rather than its left
    edge; truncated by node count rather than depth, so small expressions are captured
    exactly.

    A level-order list of tags with child counts determines the tree prefix, so two
    expressions with equal tuples share exactly their first nine nodes.
    """
    order: list[object] = []
    queue = deque([_outline(node)])
    while queue and len(order) < OUTLINE:
        tag, operands = queue.popleft()
        order.append((tag, len(operands)))
        queue.extend(operands)
    return tuple(order)


def _representatives(cases: list[Case]) -> list[Case]:
    """Keep the first line of each shape in `cases`, dropping the rest.

    This drops eleven hundred of twenty-seven hundred lines and nearly a third of the
    work, mostly from `FUN.MTH`'s hypergeometric definitions, where forty duplicates
    took half a minute. Every shape is still run, so the expensive one-of-a-kind lines
    (`F_2L`'s integral, `HYP_AUX`, `PSI`, `BESSEL_Y_SERIES`) still cost what they did.

    A line the parser rejects has no shape, so it is keyed by its text.

    Keeping the first of each shape in `cases` order keeps test ids stable between
    runs.
    """
    seen: set[object] = set()
    kept: list[Case] = []
    for case in cases:
        _, text, state = case
        try:
            shape: object = _shape(parse_expression(text, state).node)
        except DeriveSyntaxError:
            shape = ("unparsed", text)
        if shape in seen:
            continue
        seen.add(shape)
        kept.append(case)
    return kept


CASES = _representatives(
    [(group, text, state) for group in sorted(GROUPS) for text, state in GROUPS[group]]
)


@pytest.mark.parametrize(
    ("group", "text", "state"),
    CASES,
    ids=[f"{group}: {text[:60]}" for group, text, _ in CASES],
)
def test_a_shipped_expression_settles(group: str, text: str, state: ParseState) -> None:
    """Both invariants for one expression, plus that Simplify never raises.

    Exceptions aren't caught, so Simplify raising on a parsed line fails with its
    traceback.
    """
    assert _settles(text, state) is not text.startswith(UNSETTLED)


def test_the_record_of_unsettled_expressions_is_current() -> None:
    # Ensures every `UNSETTLED` name is actually run. Being in the corpus isn't enough,
    # since a line can be dropped as a duplicate shape.
    for name in UNSETTLED:
        assert any(text.startswith(name) for _, text, _ in CASES), name
