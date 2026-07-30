"""What soLve must produce.

Author text in, author text out, as the Simplify and Factor cases are, and for
the same reason: the text the worksheet shows is the whole of what the command
promises. What differs is the shape of the answer - a list of entries rather
than one - so every case below names the whole list, and the empty list is a
real expectation and not a missing one.

Almost every value here was checked against the original, and the rest are
quoted in the 3.14 manual. Where this engine's answer differs from the
original's the engine's is recorded and the case says why; nothing is weakened
to make a test pass. There are four such places and they are all deliberate:
the order of several solutions, which representatives a periodic equation is
sampled at, which variable of a singular system is the arbitrary one, and the
last digit of a rounded root.
"""

from __future__ import annotations

import pytest
import sympy as sp

from rederive.engine.computing import (
    Context,
    domain_of_node,
    simplify,
    solve,
    to_sympy,
)
from rederive.engine.context import Angle, Precision
from rederive.syntax import ParseState, parse_expression

EXACT = Context()
APPROXIMATE = Context().with_precision(Precision.APPROXIMATE)
MIXED = Context().with_precision(Precision.MIXED)
DEGREES = Context(angle=Angle.DEGREE)


def parse(text, state=None):
    return parse_expression(text, state or ParseState()).node


def sol(text, variables=(), context=None, bounds=None):
    """`text` solved, as the worksheet would show the entries it appends."""
    interval = None if bounds is None else (parse(bounds[0]), parse(bounds[1]))
    answers = solve(parse(text), context or EXACT, variables, interval)
    return [answer.text for answer in answers]


# -- equations ----------------------------------------------------------------

EQUATIONS = [
    ("2*x + 3 = 7", ["x = 2"]),
    ("x^2 - 5*x + 6 = 0", ["x = 2", "x = 3"]),
    # A bare expression is solved as `u = 0`. Derive prints `2` before `-2`;
    # the order here is reals ascending, which the case below is about.
    ("x^2 - 4", ["x = -2", "x = 2"]),
    # Complex roots come out in rectangular form, and a conjugate pair is
    # written with the positive imaginary part first, as the original writes it.
    ("x^2 + 1", ["x = #i", "x = -#i"]),
    ("x^3 - 2*x^2 + x - 2", ["x = 2", "x = #i", "x = -#i"]),
    # Multiplicity is collapsed: the double root at zero is listed once.
    ("x^2*(8*x - 9) = 0", ["x = 0", "x = 9/8"]),
    ("LN(x) = 5", ["x = #e^5"]),
    # Nothing here can solve either of these, so the residual equation is what
    # comes back - no warning, no guess. Derive prints the quintic's residual
    # as `x^5 - x = -1`, which is its Simplify's normal form of the same thing.
    ("x^5 - x + 1 = 0", ["x^5 - x + 1 = 0"]),
    ("3^x = x^2", ["3^x - x^2 = 0"]),
    ("x = x + 1", []),
]


@pytest.mark.parametrize(("text", "expected"), EQUATIONS, ids=str)
def test_an_equation_solves_to_one_entry_per_solution(text, expected):
    assert sol(text) == expected


def test_a_symbolic_equation_solves_for_whichever_variable_was_chosen():
    assert sol("a*x + b = 0", ("x",)) == ["x = -b/a"]
    assert sol("a*x + b = 0", ("a",)) == ["a = -b/x"]


def test_a_cubic_is_solved_in_radicals():
    """Cardano, symbolically: three exact solutions and no numbers in sight.

    Their spelling is sympy's and is not the original's, so what is asserted is
    that there are three of them, that each is exact, and that each satisfies
    the equation for a case the equation can be checked on.
    """
    answers = solve(parse("x^3 + a*x + b = 0"), EXACT, ("x",))
    assert len(answers) == 3
    assert not any("." in answer.text for answer in answers)
    a, b, x = sp.symbols("a b x", real=True)
    for answer in answers:
        root = to_sympy(answer.node, EXACT).rhs
        residual = (x**3 + a * x + b).subs(x, root).subs({a: 2, b: 3})
        assert abs(complex(residual.evalf())) < 1e-9


def test_the_order_of_several_solutions_is_ours_and_not_the_originals():
    """Reals ascending, then the complex ones. Derive's order is its solver's:
    `x^2 - 5*x + 6` comes out `2, 3` there and `x^2 - 4` comes out `2, -2`, so
    there is no rule to be faithful to and one worth having."""
    assert sol("x^2 - 5*x + 6 = 0") == ["x = 2", "x = 3"]
    assert sol("x^2 - 4") == ["x = -2", "x = 2"]


# -- periodic equations -------------------------------------------------------
#
# Derive has no way of writing `x = 2*pi*n` and neither has this engine, so a
# periodic equation gets a finite sample near the origin. What Derive prints is
# redundant - `COS(x) = 1/2` gives `pi/3`, `-pi/3` and `5*pi/3`, the last two
# being one point - and what is printed here is one representative per family.


def test_a_periodic_equation_is_sampled_once_per_family():
    # Derive prints `pi/6`, `5*pi/6` and `-7*pi/6`, the third being the first
    # again a period down.
    assert sol("SIN(x) = 1/2") == ["x = pi/6", "x = 5*pi/6"]
    # Derive prints `0`, `pi` and `-pi`.
    assert sol("SIN(x) = 0") == ["x = 0", "x = pi"]
    # Derive prints `pi/3`, `-pi/3` and `5*pi/3`.
    assert sol("COS(x) = 1/2") == ["x = -pi/3", "x = pi/3"]


def test_a_solution_is_measured_in_whatever_the_angle_setting_says():
    assert sol("SIN(x) = 1/2", context=DEGREES) == ["x = 30", "x = 150"]


# -- identities ---------------------------------------------------------------


def test_an_identity_solves_to_an_arbitrary_value():
    assert sol("x = x") == ["x = @1"]


def test_the_arbitrary_counter_comes_from_the_context():
    """Which is what lets the session keep one counter across every command,
    and what keeps the answer a function of the tree and the context alone."""
    assert sol("x = x", context=Context(arbitrary_index=2)) == ["x = @2"]
    assert sol("x = x", context=Context(arbitrary_index=7)) == ["x = @7"]


# -- inequalities -------------------------------------------------------------


INEQUALITIES = [
    # One interval is one chained entry.
    ("x^2 < 4", ["-2 < x < 2"]),
    ("x^2 <= 4", ["-2 <= x <= 2"]),
    # A union is one entry per piece, exactly as several roots are.
    ("ABS(3*x + 2) > 5", ["x < -7/3", "x > 1"]),
    # A half-line is a single relation, with the variable on the left.
    ("2*x > 6", ["x > 3"]),
    ("2*x >= 6", ["x >= 3"]),
]


@pytest.mark.parametrize(("text", "expected"), INEQUALITIES, ids=str)
def test_an_inequality_solves_to_the_range_it_describes(text, expected):
    assert sol(text) == expected


def test_a_symbolic_inequality_turns_under_a_negative_coefficient():
    assert sol("-2*x + 3*y <= 7", ("x",)) == ["x >= (3*y - 7)/2"]


def test_an_unequality_is_solved_too():
    """`/=` is the one relation the original is content to punt on. Sympy
    reduces it, so it is answered rather than punted on."""
    assert sol("x /= 3") == ["x < 3", "x > 3"]


def test_an_inequality_that_always_holds_is_an_identity():
    assert sol("x^2 >= 0") == ["x = @1"]


def test_an_unsolved_inequality_keeps_its_operator():
    """The residual of a `<` is a `<`: moving everything to the left of one
    does not turn it into an equation, and an entry that said so would be
    saying something nobody asked."""
    assert sol("x^3 < SIN(x)") == ["x^3 - SIN(x) < 0"]


def test_a_chained_answer_reads_back_as_what_it_says():
    """Every result reparses its own text, so the chain has to be something the
    grammar takes. A relation chain nests to the left, which is what the layer
    that draws it already knows how to draw flat."""
    answer = solve(parse("x^2 < 4"), EXACT)[0]
    assert answer.text == "-2 < x < 2"
    assert answer.node.kind is parse("-2 < x < 2").kind
    assert answer.node == parse("-2 < x < 2")


# -- systems ------------------------------------------------------------------


SYSTEMS = [
    ("[x + y = 3, x - y = 1]", (), ["[x = 2, y = 1]"]),
    # Linear in the solution variables, arbitrary in everything else.
    (
        "[2*a^2*x + 3*y = 7, x - 5*y = 0]",
        (),
        ["[x = 35/(10*a^2 + 3), y = 7/(10*a^2 + 3)]"],
    ),
    # A singular consistent system parametrises its freedom, and the free
    # variable is replaced by the arbitrary value rather than left as itself.
    ("[x + 3*y = 1, 2*x + 6*y = 2]", (), ["[x = @1, y = (1 - @1)/3]"]),
    # An inconsistent one has no solutions, and appends nothing.
    ("[x + 3*y = 1, 2*x + 6*y = 3]", (), []),
    # Variables nobody chose stay as themselves on the right.
    ("[x + y + z = 1, x - y = 0]", ("x", "y"), ["[x = (1 - z)/2, y = (1 - z)/2]"]),
    ("[x + y + z = 1, x - y = 0]", ("x", "z"), ["[x = y, z = 1 - 2*y]"]),
]


@pytest.mark.parametrize(("text", "variables", "expected"), SYSTEMS, ids=str)
def test_a_system_solves_to_one_entry_holding_the_solution_vector(
    text, variables, expected
):
    assert sol(text, variables) == expected


def test_a_system_nonlinear_in_its_variables_is_solved_too():
    """Beyond the original, which handles only the linear case and refers the
    rest to a five-step manual recipe. One entry per solution, as everywhere."""
    assert sol("[x^2 + y^2 = 1, y = x]", ("x", "y")) == [
        "[x = -SQRT(2)/2, y = -SQRT(2)/2]",
        "[x = SQRT(2)/2, y = SQRT(2)/2]",
    ]


def test_a_system_nobody_can_solve_falls_back_to_its_own_residuals():
    """Never "no solutions found", which would be a lie: not knowing and there
    being nothing to know are different answers, and only one of them is
    something to say."""
    assert sol("[3^x = x^2, LN(y) = COS(y)]", ("x", "y")) == [
        "[3^x - x^2 = 0, LN(y) - COS(y) = 0]"
    ]


# -- the numeric modes --------------------------------------------------------


def test_approximate_precision_searches_the_interval_it_was_given():
    assert sol("x^2 - 5*x + 6 = 0", context=APPROXIMATE, bounds=("0", "2.5")) == [
        "x = 2"
    ]
    assert sol("x^2 - 5*x + 6 = 0", context=APPROXIMATE, bounds=("10", "20")) == []


def test_a_numeric_root_is_shown_to_the_precision_digits():
    assert sol("x^5 - x + 1 = 0", context=APPROXIMATE, bounds=("-10", "10")) == [
        "x = -1.16730"
    ]


def test_every_real_root_in_the_interval_comes_back_not_just_one():
    """Where the original returns one root even when the interval holds
    several: `x^2 - 5*x + 6` over `[-10, 10]` gives `x = 3` there. For a
    polynomial all of them cost the same as one."""
    assert sol("x^2 - 5*x + 6 = 0", context=APPROXIMATE, bounds=("-10", "10")) == [
        "x = 2",
        "x = 3",
    ]


def test_mixed_precision_searches_without_being_told_where():
    # Derive prints `-0.686026`, cutting the digit this engine rounds: the
    # root is -0.68602672..., which is `-0.686027` to six significant digits.
    assert sol("3^x = x^2", context=MIXED) == ["x = -0.686026"]


def test_mixed_precision_solves_exactly_where_it_can():
    assert sol("x^2 - 5*x + 6 = 0", context=MIXED) == ["x = 2", "x = 3"]


# -- domains ------------------------------------------------------------------


def test_a_declared_domain_does_not_filter_the_solutions():
    """The manual is explicit about it: a declaration says what a variable is,
    which decides how an answer simplifies, and never which answers are
    wanted. Solutions are sought over the complexes whatever was declared."""
    state = ParseState()
    name, domain = domain_of_node(parse("x :epsilon Real", state))
    context = Context(domains={name: domain})
    assert sol("x^2 + 1", context=context) == ["x = #i", "x = -#i"]


# -- the promises -------------------------------------------------------------

# Every kind of thing the parser produces, solved. None of these may raise;
# what they come back as is not the point.
TOTAL = [
    "x",
    "?",
    "1/0",
    "INT(#e^(x^2), x)",
    "DIF(F(x)^3, x)",
    "LIM(SIN(x)/x, x, 0)",
    "SUM(k^2, k, 1, n)",
    "SOLVE(x^2 = 2, x)",
    "IF(x > 0, x, -x)",
    "x :epsilon Real [0, inf)",
    "F(y) := y^2 - 1",
    "[[1, 2], [3, 4]]",
    "x > 1 AND x < 3",
    "#e^(#i*pi)",
    "(x^2 - 1)^(1/3)",
    "RANDOM(5)",
    "x^2 - 1 = y^2 - 1",
    "[1, 2, 3]",
    "[]",
    "2 = 2",
]


@pytest.mark.parametrize("text", TOTAL, ids=str)
@pytest.mark.parametrize(
    "context", [EXACT, APPROXIMATE, MIXED], ids=lambda c: str(c.precision)
)
def test_solving_never_raises(text, context):
    bounds = None if context is not APPROXIMATE else (parse("-10"), parse("10"))
    solve(parse(text), context, (), bounds)


def test_a_variable_the_expression_does_not_hold_has_no_solutions():
    assert sol("x^2 - 4", ("w",)) == []


# -- SOLVE as an authored function --------------------------------------------
#
# `SOLVE(u, x)` is the author-line spelling of the command, so Simplify is what
# evaluates it. The answer is a *vector* of relations, which is the Derive 3
# and 4 shape; Derive 5 turned it into a disjunction, and the shipped libraries
# this engine is meant to run are written against the vector.

HEADS = [
    ("SOLVE(x^2 - 5*x + 6 = 0, x)", "[x = 2, x = 3]"),
    # Derive writes `[x = 2, x = -2]`; the order is this engine's, as above.
    ("SOLVE(x^2 - 4, x)", "[x = -2, x = 2]"),
    ("SOLVE(x = x + 1, x)", "[]"),
    ("RHS(SOLVE(x^2 - 5*x + 6 = 0, x))", "[2, 3]"),
    ("LHS(SOLVE(x^2 - 5*x + 6 = 0, x))", "[x, x]"),
    ("SOLVE(- 2*x + 3*y <= 7, x)", "[x >= (3*y - 7)/2]"),
    ("SOLVE(x^2 < 4, x)", "[-2 < x < 2]"),
    # A system contributes one inner vector per solution, so that the outer
    # vector is always the list of entries the command would have appended.
    ("SOLVE([x + y = 3, x - y = 1], [x, y])", "[[x = 2, y = 1]]"),
]


@pytest.mark.parametrize(("text", "expected"), HEADS, ids=str)
def test_simplify_evaluates_a_solve_head(text, expected):
    assert simplify(parse(text), EXACT).text == expected


def test_the_dimension_of_a_solve_is_how_many_solutions_there_are():
    """Which is what the shipped libraries branch on, and why the empty vector
    matters: no solutions is a vector of nothing rather than an error."""
    assert simplify(parse("DIMENSION(SOLVE(x^2 - 4, x))"), EXACT).text == "2"
    assert simplify(parse("DIMENSION(SOLVE(x = x + 1, x))"), EXACT).text == "0"


def test_a_degenerate_solve_head_mints_an_arbitrary_value():
    assert simplify(parse("SOLVE(x = x, x)"), EXACT).text == "[x = @1]"


def test_the_four_argument_form_is_numeric_in_every_mode():
    """Derive's help says "if in approximate mode"; the shipped files use it as
    though it always applied, and always-numeric is what makes the call mean
    one thing. In Exact mode the answer is still an approximate number, and an
    approximate number under Rational notation is written as the ratio it is."""
    assert simplify(parse("SOLVE(x^5 - x + 1 = 0, x, -10, 10)"), EXACT).text == (
        "[x = -1221/1046]"
    )
    approximate = simplify(parse("SOLVE(x^5 - x + 1 = 0, x, -10, 10)"), APPROXIMATE)
    assert approximate.text == "[x = -1.16730]"


def test_a_solve_head_is_solved_where_it_stands():
    """It is an expression like any other, so it may be part of a larger one."""
    assert simplify(parse("DIMENSION(RHS(SOLVE(x^2 - 4, x)))"), EXACT).text == "2"


def test_one_solution_can_be_read_out_of_the_vector():
    """`RHS((SOLVE(z, y)) SUB 1)` is how the shipped ODE library takes the
    first solution, so a subscript has to reach through the vector too."""
    assert simplify(parse("(SOLVE(x^2 - 4, x)) SUB 1"), EXACT).text == "x = -2"
    assert simplify(parse("RHS((SOLVE(x^2 - 4, x)) SUB 1)"), EXACT).text == "-2"
