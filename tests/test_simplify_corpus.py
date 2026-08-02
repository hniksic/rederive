"""Both invariants, over the whole corpus.

`test_simplify.py` is where the mathematics is: inline cases, each one naming
what it is about. This is the net underneath it - `corpus/`, which is Derive's
shipped utility files, its demo scripts and the parser's own cases, some three
thousand lines saying some sixteen hundred distinct things - each simplified,
printed, read back and simplified again. What it catches is a change that
breaks something no inline case covers. What it reports is the line it broke
on, which is a starting point and not a diagnosis: the fix is usually a new
inline case in `test_simplify.py` and then the code to pass it.

Twenty seconds across every core, against the twenty-five the rest of the suite
takes, and it runs with the rest: what a marker holding it back would buy is a
green run that had not asked the question.

One case per expression, not per file. The cost of an expression runs from a
microsecond to ten seconds, and a case per file would leave one worker grinding
through the hypergeometric integrals while the others sat idle; a case apiece
gives the distributor something to distribute, and the tail is one expression
rather than one file.

One case per shape, though, and not one per expression. A line's shape is what
`_shape` makes of its parse: the names and the numbers taken out, and nothing
below the ninth node read at all. The six definitions in `FUN.MTH` that read
`<coefficient>*DIF(z^(a-1)*..., z, ...)` are one shape and one case, the
forty-four unit definitions that read `<unit>:=<n>*<unit>` are another, and
eleven hundred of the corpus's twenty-seven hundred lines are runs of Simplify
over a form it has already been run over. Dropping them takes close to a third
off the work.

This one is a trade and not a free one. Two lines of a shape are built the same
way but they are not the same expression, and where the case that runs has
`LN(1 - z)` the one dropped had `ASIN(SQRT(z))`, so a gap that only the second
would find is a gap this file no longer finds. What holds the loss down is that
a shape is read with the function names in it - `SIN(x)` and `COS(x)` are two
shapes, not one - and that no family here disagrees about settling, which was
checked over the whole corpus when the rule was written and is what the ninth
node is: the coarsest reading at which every family still speaks with one
voice. What the project has on file, in every spelling, is
`test_expr_library.py`'s question, and it still asks it of every one of them;
what shrinks here is what Simplify is run over, not what the project reads.
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
#: * `F_3A` divides `SECH(z/k)^(k+1)` by `2^(k+1)` and writes the base of the
#:   power as `1/(2*(#e^(z/k)/2 + #e^(-z/k)/2))`, with the two outside the sum,
#:   and multiplies it through on the second pass to reach
#:   `1/(#e^(z/k) + #e^(-z/k))`. The same two as `BESSEL_Y_SERIES`, standing in
#:   a power's base rather than in front of a sum.
#: * `LIN2_CCF_HOM` writes `(p^2/4 - d/4)^(x/2)` as it stands and
#:   `(p^2 - d)^(x/2)/2^x` on the second pass: splitting the base did not pay
#:   until the form it was offered had changed.
UNSETTLED = (
    "BESSEL_Y_SERIES(",
    "F_3A(",
    "LIN2_CCF_HOM(",
)


def _library() -> dict[str, list[tuple[str, ParseState]]]:
    """The shipped utility files, by the `##` group each was read from.

    Nothing in these files sets a mode, so every line is read the way a session
    that had just started would read it.
    """
    groups: dict[str, list[tuple[str, ParseState]]] = {}
    current: list[tuple[str, ParseState]] = []
    for line in (CORPUS / "library-expressions.txt").read_text().splitlines():
        if line.startswith("##"):
            current = groups.setdefault(line[2:].strip(), [])
        elif line.strip() and not line.startswith("#"):
            current.append((line, ParseState()))
    return groups


def _parser_cases() -> list[tuple[str, ParseState]]:
    """The conformance corpus, through the loader `test_expr_corpus.py` uses.

    One reader for the format, which holds more than an expression per line:
    directive lines, group headers, comments, and cases whose own text begins
    with `#`, which is how the constants `#e` and `#i` are written.

    Each case is read under the state its `[state:]` and `[declared:]`
    directives put it in, since what a line parses as depends on them: `gh(1,2)`
    is a call only where `GH` is declared, and `lhs-rhs` is a subtraction only in
    Sensitive case mode. The input base is the one setting left at its default,
    because answers are printed in base ten and there is no output base to match
    a reader set to another one: under `InputBase=16` the case `2A` is 42, which
    prints as `42` and reads back as 66. That is a property of the notation
    rather than of Simplify, and Simplify is what this module asks about.
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
    """Whether simplifying `text` reaches a form that stays where it is.

    The two invariants at once: printing the answer is a fixed point, and
    simplifying it again changes nothing. `state` reads the line and reads the
    print of the answer back, so both halves are one session's. A line the
    parser rejects is not this module's business - `test_expr_library.py` is
    where that is checked.

    Printing takes the answer's own conditionals with it. An undecidable `IF`
    is shown as it was written rather than as it was converted, so what its
    arms are spelled as is not a property of the expression and no print of one
    can recover it; the record is part of how such an answer is printed at all,
    and printing without it would be asking a different question.
    """
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


Case = tuple[str, str, ParseState]

#: The leaves whose text says what an expression is about rather than what it
#: is built out of. Which variable, which number, and which of `a`, `b`, `c`
#: and `z` a definition took as its parameters is most of what separates one
#: line of a family from the next, and a shape has none of it.
ANONYMOUS = (Kind.NUMBER, Kind.NAME)

#: The runs whose operands are a bag and not a list. `2*SQRT(pi)*c!/d*DIF(u)`
#: and `-c!/d*DIF(u)` differ in the order and the number of their factors and
#: in nothing else, and a coefficient is not a shape.
COMMUTATIVE = (Kind.SUM, Kind.PRODUCT, Kind.AND, Kind.OR, Kind.XOR)

#: The kinds whose first child is the function rather than an argument. That
#: name is the one leaf a shape keeps, because it is what an expression is made
#: of and not what it is about: `SIN(x)` and `LN(x)` are two shapes, and a
#: family that collapsed the calls into each other would collapse the engine's
#: paths through them too.
APPLICATIONS = (Kind.CALL, Kind.APPLY, Kind.FUNCPOW)

#: How much of an expression its shape is read from, in nodes, breadth first.
#: Nine reaches the outline of a definition - the header, the body's operator,
#: and what the body is a product or a sum of - and stops above the arguments,
#: which is where a family's members differ. It is a tuned number: the corpus
#: was run in full and every family's verdict compared, and nine is the
#: coarsest reading at which no family holds two verdicts. At eight, Kummer's
#: function joins `BESSEL_Y_SERIES`, which does not settle, and Derive's
#: two-line ODE solvers join `LIN2_CCF_HOM`, which does not either - and a
#: family that disagrees about settling is the evidence that its members are
#: not one shape after all.
OUTLINE = 9


def _outline(node: Node) -> tuple[object, ...]:
    """`node` as the thing it is built out of, in `(tag, operands)` form.

    Everything a name or a number carries goes, along with the offsets and the
    `surface` spelling that `_settles` has no use for either. A definition's
    own name goes with them, since `F_1A(a,b,c,z):=` and `F_1B(a,b,c,z):=` are
    one header. A leading minus goes too, being the shortest way there is to
    write a coefficient.

    What stays is the frame: which operator, which function, how deep, and how
    many operands - and for a commutative run, which operands, without their
    order and without repeating one that appears twice.
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
    """`node`'s outline down to its first `OUTLINE` nodes, breadth first.

    Breadth first and not depth first, so that what a shape describes is the
    whole of the top of an expression rather than the whole of its left edge,
    and by node count and not by depth, so that an expression small enough to
    fit is described exactly and only the big ones are read as an outline.
    Nothing in the corpus under nine nodes is grouped with anything it does not
    match node for node.

    The tags and the operand counts are the whole of it: a level-order walk
    that records how many children each node has can be read back into the
    tree, so two expressions with this tuple in common have the same first nine
    nodes and no more is claimed about them.
    """
    order: list[object] = []
    queue = deque([_outline(node)])
    while queue and len(order) < OUTLINE:
        tag, operands = queue.popleft()
        order.append((tag, len(operands)))
        queue.extend(operands)
    return tuple(order)


def _representatives(cases: list[Case]) -> list[Case]:
    """`cases` with one line of each shape kept and the rest left out.

    Eleven hundred of the twenty-seven hundred lines go, and with them close to
    a third of the work: `FUN.MTH`'s hypergeometric definitions are where it
    tells, forty of them being another writing of a shape already run and half
    a minute between them. What is left is not a subset anyone chose - every
    shape the corpus writes is still run, including every shape written once,
    which is what the expensive end of the file is now made of: `F_2L`'s
    integral, `HYP_AUX`, `PSI` and `BESSEL_Y_SERIES` are each the only line of
    their shape and each still costs what they cost.

    The parse decides, as it does in `_settles`, and it is asked the same way,
    so a line the parser rejects stands for its own text: there is no tree to
    take a shape of and no simplifying to be done either.

    The first line of a shape in `cases` order survives, so the test id a case
    carries does not move between runs.
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
    """Both invariants over one expression, and totality along with them.

    Nothing here catches an exception, so Simplify raising on something the
    parser accepted fails this as an error with its traceback, which is what
    the promise that it never raises looks like from the outside.
    """
    assert _settles(text, state) is not text.startswith(UNSETTLED)


def test_the_record_of_unsettled_expressions_is_current() -> None:
    # Cheap enough for the default run, and it is what keeps the record above
    # from naming an expression no case runs - which the corpus holding it is
    # not enough for, since a case can be dropped as another of its shape.
    for name in UNSETTLED:
        assert any(text.startswith(name) for _, text, _ in CASES), name
