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
from rederive.model.expr import Kind
from rederive.syntax import ParseState, parse_expression
from rederive.syntax.state import InputMode

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
    # Two roots of the substituted quadratic, a conjugate pair, cube to the same
    # number, which is one solution and is listed once.
    ("x^(1/3)/3 + 3/x^(1/3) = 1", ["x = -27"]),
    # Nothing here can solve either of these, so the residual equation is what
    # comes back - no warning, no guess. Derive prints the quintic's residual
    # as `x^5 - x = -1`, which is its Simplify's normal form of the same thing.
    ("x^5 - x + 1 = 0", ["x^5 - x + 1 = 0"]),
    ("COS(x) = x", ["COS(x) - x = 0"]),
    # Derive gives up on this one too. Lambert W inverts it, and its one real root
    # is the answer.
    ("3^x = x^2", ["x = -2*LAMBERTW(LN(3)/2)/LN(3)"]),
    # Both real branches of Lambert W, where an equation has a root on each.
    ("x = LN(x) + 2", ["x = -LAMBERTW(-#e^(-2))", "x = -LAMBERTW(-#e^(-2), -1)"]),
    ("x*EXP(x) = 1", ["x = LAMBERTW(1)"]),
    ("x^x = 2", ["x = #e^LAMBERTW(LN(2))"]),
    # There is no real root, but there are complex ones, so finding no real one
    # leaves the equation standing rather than answering that there are none.
    ("x*EXP(x) = -1", ["x*#e^x + 1 = 0"]),
    # `LAMBERTW(a)` is one root of several for a negative `a`, and nothing here
    # proves it a root at all, so this stands too.
    ("x*EXP(x) = a", ["x*#e^x - a = 0"]),
    ("x = x + 1", []),
    # Infinity is no solution: `1/x` only tends to zero there.
    ("1/x = 0", []),
    # The first has the two integer roots and a negative one Lambert W writes; the
    # second holds at 1 and, `(-1)^(-1)` being -1, at -1.
    ("2^x = x^2", ["x = -2*LAMBERTW(LN(2)/2)/LN(2)", "x = 2", "x = 4"]),
    ("x^x = x", ["x = -1", "x = 1"]),
]


@pytest.mark.parametrize(("text", "expected"), EQUATIONS, ids=str)
def test_an_equation_solves_to_one_entry_per_solution(text, expected):
    assert sol(text) == expected


#: Equations a solver rewrites before it solves them - logarithms combined, a
#: square root squared - pick up candidates that the rewritten equation has and
#: the authored one has not, so each candidate is put back. The principal
#: logarithm of a negative number carries `pi*#i`, which is what rules out the
#: negative candidates here: `LN(-2) + LN(-3)` is `LN(6) + 2*pi*#i`.
EXTRANEOUS = [
    ("LN(x) + LN(x - 1) = LN(6)", ["x = 3"]),
    ("LN(x) + LN(x + 1) = LN(2)", ["x = 1"]),
    ("LN(x - 2) + LN(x - 3) = LN(2)", ["x = 4"]),
    ("LN(x) + LN(x - 3) = LN(4)", ["x = 4"]),
    ("2*LN(x) = LN(x + 2)", ["x = 2"]),
    ("2*LN(x) = LN(4)", ["x = 2"]),
    ("LOG(x, 2) + LOG(x - 2, 2) = 3", ["x = 4"]),
    ("LOG(x, 10) + LOG(x + 21, 10) = 2", ["x = 4"]),
    # Both candidates hold here: `LN((-3)^2)` is `LN(9)`, and so is `2*LN(3)`.
    ("LN(x^2) = 2*LN(3)", ["x = -3", "x = 3"]),
    ("LN(x^2 - 1) = LN(3)", ["x = -2", "x = 2"]),
    # Squaring adds `x = 1` and `x = -1`, where the root is the negation.
    ("SQRT(x) = x - 2", ["x = 4"]),
    ("SQRT(x + 5) = x - 1", ["x = 4"]),
    ("SQRT(2*x + 3) = x", ["x = 3"]),
    ("SQRT(x) = -2", []),
    # The principal cube root of -8 is `1 + SQRT(3)*#i`, so nothing cubes to -2.
    ("x^(1/3) = -2", []),
]


@pytest.mark.parametrize(("text", "expected"), EXTRANEOUS, ids=str)
def test_a_candidate_that_does_not_hold_is_no_solution(text, expected):
    assert sol(text) == expected


#: An `ABS` of a real argument is the argument or its negation, so an equation in
#: one is finitely many equations, each over the range where its signs hold. The
#: answer can be a range too, written as an inequality is.
ABSOLUTE = [
    ("ABS(x - 1) = 2", ["x = -1", "x = 3"]),
    ("ABS(2*x - 3) = 5", ["x = -1", "x = 4"]),
    ("ABS(x) = 3", ["x = -3", "x = 3"]),
    ("ABS(x) = -1", []),
    ("ABS(x^2 - 4) = 0", ["x = -2", "x = 2"]),
    ("ABS(x^2 - 1) = 3", ["x = -2", "x = 2"]),
    ("ABS(x + 1) = ABS(x - 3)", ["x = 1"]),
    ("ABS(x - 1) = ABS(2*x)", ["x = -1", "x = 1/3"]),
    ("ABS(x - 1) + ABS(x + 1) = 4", ["x = -2", "x = 2"]),
    ("ABS(x) = x + 1", ["x = -1/2"]),
    ("ABS(x) + x^2 = 2", ["x = -1", "x = 1"]),
    # An undeclared parameter is real too.
    ("ABS(x - a) = 1", ["x = a - 1", "x = a + 1"]),
    ("ABS(x) = x", ["x >= 0"]),
    ("ABS(x - 1) + ABS(x + 1) = 2", ["-1 <= x <= 1"]),
    # Each family of a periodic one is sampled once, as without the `ABS`.
    ("ABS(SIN(x)) = 1/2", ["x = -5*pi/6", "x = -pi/6", "x = pi/6", "x = 5*pi/6"]),
]


@pytest.mark.parametrize(("text", "expected"), ABSOLUTE, ids=str)
def test_an_equation_in_abs_is_solved_sign_by_sign(text, expected):
    assert sol(text) == expected


def test_an_equation_in_abs_of_a_complex_variable_stands():
    """`ABS(x - 1) = 2` over the complexes is a circle, which no list of
    relations describes."""
    state = ParseState()
    name, domain = domain_of_node(parse("x :epsilon Complex", state))
    context = Context(domains={name: domain})
    assert sol("ABS(x - 1) = 2", ("x",), context=context) == ["ABS(x - 1) - 2 = 0"]


#: A power of any positive base is a power of `#e`, and is solved as one: each
#: equation has a family of complex solutions per root of the quadratic in `b^x`,
#: sampled at its member nearest the origin.
EXPONENTIALS = [
    ("3^(2*x) - 4*3^x + 3 = 0", ["x = 0", "x = 1"]),
    ("2^(2*x) - 5*2^x + 4 = 0", ["x = 0", "x = 2"]),
    ("9^x - 10*3^x + 9 = 0", ["x = 0", "x = 2"]),
    ("#e^(2*x) - 3*#e^x + 2 = 0", ["x = 0", "x = LN(2)"]),
    # `2^x = -2` holds where `x*LN(2)` is `LN(2) + pi*#i`.
    ("4^x - 2^(x + 1) - 8 = 0", ["x = 2", "x = 1 + #i*pi/LN(2)"]),
    ("#e^(2*x) - 2*#e^x - 8 = 0", ["x = LN(4)", "x = LN(2) + #i*pi"]),
    ("(1/2)^x = 4", ["x = -2"]),
]


@pytest.mark.parametrize(("text", "expected"), EXPONENTIALS, ids=str)
def test_an_equation_in_powers_of_a_number_is_solved_through_logarithms(
    text, expected
):
    assert sol(text) == expected


def test_a_solution_is_written_in_its_plainer_spelling():
    """No negated quotient over a negative number, and no factored radicand where
    the expanded one is shorter."""
    assert sol("2^x = 3^(x - 1)") == ["x = LN(3)/(LN(3) - LN(2))"]
    assert sol("5^x = 2^(x + 1)") == ["x = LN(2)/(LN(5) - LN(2))"]
    assert sol("x^2 + y^2 = r^2", ("y",)) == [
        "y = -SQRT(r^2 - x^2)",
        "y = SQRT(r^2 - x^2)",
    ]


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


def test_three_real_roots_are_written_without_an_imaginary_unit():
    """The casus irreducibilis: an irreducible cubic whose roots are all real.

    Cardano's formula reaches them only through the cube root of a complex
    number, so the radicals carry an `#i` that cancels and cannot be
    cancelled - and there is no way around that in radicals at all, which is
    what the case is named for. Viete's trigonometric form is the way around
    it, and its answers are the original's.
    """
    assert sol("x^3 - 3*x + 1 = 0", ("x",)) == [
        "x = -2*COS(pi/9)",
        "x = 2*COS(4*pi/9)",
        "x = 2*COS(2*pi/9)",
    ]


def test_one_real_root_is_written_in_two_real_cube_roots():
    """Cardano's formula as the original writes it: the real root is `u + v`, a sum
    of real cube roots, each radicand's denominator brought out of the root and a
    negative radicand's sign written in front of it. The complex pair is built from
    the same two roots."""
    assert sol("x^3 + 3*x + 1 = 0") == [
        "x = (4*SQRT(5) - 4)^(1/3)/2 - (4*SQRT(5) + 4)^(1/3)/2",
        "x = (4*SQRT(5) + 4)^(1/3)/4 - (4*SQRT(5) - 4)^(1/3)/4"
        " + SQRT(3)*#i*((4*SQRT(5) + 4)^(1/3) + (4*SQRT(5) - 4)^(1/3))/4",
        "x = (4*SQRT(5) + 4)^(1/3)/4 - (4*SQRT(5) - 4)^(1/3)/4"
        " - SQRT(3)*#i*((4*SQRT(5) + 4)^(1/3) + (4*SQRT(5) - 4)^(1/3))/4",
    ]


@pytest.mark.parametrize(
    ("text", "real"),
    [
        # Both radicands negative, and `SQRT(D)` rational.
        ("x^3 - 6*x + 6 = 0", "x = -2^(1/3) - 2^(2/3)"),
        ("x^3 - 6*x - 6 = 0", "x = 2^(2/3) + 2^(1/3)"),
        # Both radicands negative, `SQRT(D)` irrational.
        ("x^3 - 3*x + 3 = 0", "x = -(12 - 4*SQRT(5))^(1/3)/2 - (4*SQRT(5) + 12)^(1/3)/2"),
        # A cube comes out of the radicand.
        ("x^3 + 48*x + 64 = 0", "x = 2*(4*SQRT(5) - 4)^(1/3) - 2*(4*SQRT(5) + 4)^(1/3)"),
        # A leading coefficient, and a rational `SQRT(D)`.
        ("2*x^3 + 6*x + 3 = 0", "x = 2^(2/3)/2 - 2^(1/3)"),
        # A square term: the shift stands after the cube roots.
        (
            "x^3 + x^2 + 1 = 0",
            "x = -(116 - 12*SQRT(93))^(1/3)/6 - (12*SQRT(93) + 116)^(1/3)/6 - 1/3",
        ),
        (
            "2*x^3 - 3*x^2 + 5*x + 7 = 0",
            "x = (3*SQRT(27273) - 486)^(1/3)/6 - (3*SQRT(27273) + 486)^(1/3)/6 + 1/2",
        ),
        # A cube once shifted, with one radicand zero.
        ("x^3 + 3*x^2 + 3*x - 1 = 0", "x = 2^(1/3) - 1"),
    ],
    ids=str,
)
def test_each_root_of_a_cubic_with_one_real_root_solves_it(text, real):
    """The real root is pinned and the pair is checked: every one of the three
    satisfies the equation, the real one is the only real one, and each is written
    as it simplifies again."""
    answers = solve(parse(text), EXACT, ("x",))
    assert [answer.text for answer in answers][0] == real
    assert len(answers) == 3
    equation = to_sympy(parse(text), EXACT)
    residual = equation.lhs - equation.rhs
    (x,) = residual.free_symbols
    for answer in answers:
        root = to_sympy(answer.node, EXACT).rhs
        value = complex(residual.subs(x, root).evalf(30))
        assert abs(value) < 1e-20
        assert simplify(answer.node, EXACT).text == answer.text
    assert simplify(parse(f"SOLVE({text}, x, Real)"), EXACT).text == f"[{real}]"


def test_a_cubic_factor_is_written_as_the_cubic_is():
    assert sol("(x - 2)*(x^3 + 3*x + 1) = 0")[:2] == [
        "x = (4*SQRT(5) - 4)^(1/3)/2 - (4*SQRT(5) + 4)^(1/3)/2",
        "x = 2",
    ]


def test_the_order_of_several_solutions_is_ours_and_not_the_originals():
    """Reals ascending, then the complex ones. Derive's order is its solver's:
    `x^2 - 5*x + 6` comes out `2, 3` there and `x^2 - 4` comes out `2, -2`, so
    there is no rule to be faithful to and one worth having."""
    assert sol("x^2 - 5*x + 6 = 0") == ["x = 2", "x = 3"]
    assert sol("x^2 - 4") == ["x = -2", "x = 2"]


def test_complex_solutions_go_round_the_circle():
    """The complex ones by their angle from the positive real axis, each conjugate
    pair together with its positive imaginary part first."""
    assert sol("x^4 = -4") == ["x = 1 + #i", "x = 1 - #i", "x = -1 + #i", "x = -1 - #i"]
    assert sol("x^4 = 16") == ["x = -2", "x = 2", "x = 2*#i", "x = -2*#i"]


def test_a_binomial_is_solved_as_one_real_root_times_the_roots_of_unity():
    """The real root is written once per solution, not once per term of it."""
    answers = sol("x^100 = 12")
    assert len(answers) == 100
    assert answers[:4] == [
        "x = -12^(1/100)",
        "x = 12^(1/100)",
        "x = 12^(1/100)*(COS(pi/50) + #i*SIN(pi/50))",
        "x = 12^(1/100)*(COS(pi/50) - #i*SIN(pi/50))",
    ]
    assert sol("2*x^3 + 5 = 0") == [
        "x = -20^(1/3)/2",
        "x = 20^(1/3)*(1/2 + SQRT(3)*#i/2)/2",
        "x = 20^(1/3)*(1/2 - SQRT(3)*#i/2)/2",
    ]


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
    assert sol("SIN(x) = a", context=DEGREES) == ["x = 180 - ASIN(a)", "x = ASIN(a)"]


# -- identities ---------------------------------------------------------------


def test_an_identity_solves_to_an_arbitrary_value():
    assert sol("x = x") == ["x = @1"]


@pytest.mark.parametrize(
    "text", ["x/x = 1", "x/(x + 2) + 2/(x + 2) = 1", "SOLVE(x/x = 1)"]
)
def test_an_equation_whose_variable_cancels_is_solved_for_it(text):
    """`x/x = 1` cancels to `1 = 1` before a variable is chosen, and is still an
    equation in `x`, the variable it divides by: every value of it, but the poles,
    is a solution."""
    if text.startswith("SOLVE"):
        assert simplify(parse(text), EXACT).text == "[x = @1]"
    else:
        assert sol(text) == ["x = @1"]


@pytest.mark.parametrize(
    "text",
    [
        "(x^2 - 1)/(x - 1) = x + 1",
        # Zero once over one denominator, which solveset answers with no solution.
        "1/(x - 1) + 1 = x/(x - 1)",
        "1/x + 1/(x*(x - 1)) = 1/(x - 1)",
    ],
)
def test_a_rational_identity_solves_to_an_arbitrary_value(text):
    """Every value where both sides are defined is a solution, and no finite list of
    relations names every value but the poles."""
    assert sol(text) == ["x = @1"]


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
    # The variable is isolated, not left inside `-3 < x - 2 < 3`.
    ("ABS(x - 2) < 3", ["-1 < x < 5"]),
    # A periodic inequality is sampled once per family, as a periodic equation is.
    ("SIN(x) > 0", ["0 < x < pi"]),
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


# -- the sets sympy answers with ----------------------------------------------


#: One equation or inequality per set head sympy hands the solver, and the
#: relations Derive says the same thing in. `Interval`, `Union`, `EmptySet`,
#: `FiniteSet` and the whole complex plane each arrive as a set, and author
#: notation has no word for any of them: a chain is how a range is written, an
#: entry apiece is how a union is, no entries at all is how emptiness is, and
#: `@n` is how "every value there is" is. `Complement` is the sixth head and
#: has a section of its own below, because what it comes to depends on what can
#: be proved about the values it excludes.
SETS = [
    ("x^2 < 1", ["-1 < x < 1"]),
    ("x^2 > 1", ["x < -1", "x > 1"]),
    ("#e^x = 0", []),
    ("x^2 + 1 = 0", ["x = #i", "x = -#i"]),
    ("x^2 + 1 > 0", ["x = @1"]),
]


@pytest.mark.parametrize(("text", "expected"), SETS, ids=str)
def test_a_solution_set_is_written_in_the_notation_derive_has(text, expected):
    """The conversion happens in the solver, and it may not stop happening.

    Every one of these would otherwise reach the printer as the set it is, and
    a set is not something author notation can say: it would be shown as the
    inert text `Interval(-1, 1)` and be dead on arrival, or - worse, now that
    the notation has an `INTERVAL` of its own - be read back as the range a
    limit keeps to, which is a different thing that looks like an answer.

    So each entry is asserted to be a live relation as well as to read the way
    it does. `Kind.STRING` is what a lost one looks like.
    """
    answers = solve(parse(text), EXACT)
    assert [answer.text for answer in answers] == expected
    for answer in answers:
        assert answer.node.kind is not Kind.STRING


def test_a_chained_answer_reads_back_as_what_it_says():
    """Every result reparses its own text, so the chain has to be something the
    grammar takes. A relation chain nests to the left, which is what the layer
    that draws it already knows how to draw flat."""
    answer = solve(parse("x^2 < 4"), EXACT)[0]
    assert answer.text == "-2 < x < 2"
    assert answer.node.kind is parse("-2 < x < 2").kind
    assert answer.node == parse("-2 < x < 2")


# -- excluded values ----------------------------------------------------------
#
# A rational equation whose numerator holds a parameter is the shape: sympy
# answers `Complement({2*m}, {0})` for `1 - 2*m/r = 0`, because the division by
# `r` it undid is only allowed where `r` is not zero. The reading is the
# conservative one - a candidate is dropped only where it is provably excluded -
# and it lands on the original's own answers.


EXCLUSIONS = [
    # The original answers `[r = 2*m]`, and so does this: `2*m` is not provably
    # zero. Nothing is hidden by keeping it, since `m` being zero makes the
    # residual the constant 1 and leaves no solution to offer at all.
    ("1 - 2*m/r = 0", "r", ["r = 2*m"]),
    ("1 - 2*0/r = 0", "r", []),
    # Provably excluded, and the only candidate there was: no solution.
    ("1/r = 0", "r", []),
    ("a/x = b", "x", ["x = a/b"]),
    ("(x^2 - a^2)/(x*(x - a)) = 0", "x", ["x = -a"]),
    ("1/x + 1/(x + a) = 0", "x", ["x = -a/2"]),
    ("(r^2 - a)/(r - 1) = 0", "r", ["r = -SQRT(a)", "r = SQRT(a)"]),
    # Simplify cancels each of these before anything solves it, and the one
    # candidate left is where the equation as written divides by zero.
    ("(x^2 - 1)/(x - 1) = 2", "x", []),
    ("x^2/x = 0", "x", []),
    ("(x - 1)^2/(x - 1) = 0", "x", []),
    ("x^(-1)*x^2 = 0", "x", []),
    ("x/(x - 2) = 2/(x - 2)", "x", []),
    ("(x^2 - 1)/(x - 1) = 3", "x", ["x = 2"]),
    ("x^2*(x - 2)/(x - 2) = 0", "x", ["x = 0"]),
    # `r = 2*m` divides by zero only for one value of `m`, which is no proof.
    ("(1 - 2*m/r)*r/r = 0", "r", ["r = 2*m"]),
]


@pytest.mark.parametrize(("text", "variable", "expected"), EXCLUSIONS, ids=str)
def test_a_candidate_survives_unless_it_is_provably_excluded(text, variable, expected):
    assert sol(text, (variable,)) == expected


def test_a_declaration_that_settles_the_exclusion_answers_the_same():
    """Declaring the parameter nonzero decides what was undecided, and the
    answer may not move for it: a proof that the exclusion does not bite says
    the same thing the absence of a proof that it does was already worth."""
    state = ParseState()
    name, domain = domain_of_node(parse("m :epsilon Real (0, inf)", state))
    context = Context(domains={name: domain})
    assert sol("1 - 2*m/r = 0", ("r",), context=context) == ["r = 2*m"]


def test_an_exclusion_from_a_solution_family_is_left_unanswered():
    """`SIN(x)/(x - a) = 0` excludes `a` from the multiples of pi, and there is
    no way to write what is left: the family is infinite, so removing one of
    its members from it leaves a set no finite list of relations describes. The
    residual equation is the honest answer, and an invented one is not."""
    assert sol("SIN(x)/(x - a) = 0", ("x",)) == ["SIN(x)/(x - a) = 0"]


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
    # An expression in a system is the equation `u = 0`, as a bare one is.
    ("[a - 1, b - 2]", ("a", "b"), ["[a = 1, b = 2]"]),
    ("[x - y, y = 2]", ("x", "y"), ["[x = 2, y = 2]"]),
    ("[0, 0]", ("x", "y"), ["[x = @1, y = @2]"]),
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
    assert sol("[x*y = 6, x + y = 5]", ("x", "y")) == [
        "[x = 2, y = 3]",
        "[x = 3, y = 2]",
    ]


def test_a_nonlinear_system_with_a_cubic_resolvent_has_all_four_solutions():
    """Eliminating `y` leaves a quartic with the root 2 and an irreducible cubic.

    The cubic's three real roots are written in Cardano's radicals, which are long
    enough that only their count is pinned here; each is a real solution of both
    equations.
    """
    answers = sol("[x^2 + y = 7, x + y^2 = 11]", ("x", "y"))
    assert len(answers) == 4
    assert answers[0] == "[x = 2, y = 3]"


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


INFINITE_BOUNDS = [
    ("COS(x) = x", ("-inf", "inf"), ["x = 0.739085"]),
    ("COS(x) = x", ("0", "inf"), ["x = 0.739085"]),
    ("#e^x = 3 - x", ("-inf", "inf"), ["x = 0.792059"]),
    ("x^2 - 2 = 0", ("-inf", "inf"), ["x = -1.41421", "x = 1.41421"]),
    # A search that finds nothing proves nothing past where it looked, so the
    # equation stands rather than being said to have no root.
    ("COS(x) = x", ("-inf", "-5"), ["COS(x) - x = 0"]),
    ("#e^x = 0", ("-inf", "inf"), ["#e^x = 0"]),
]


@pytest.mark.parametrize(("text", "bounds", "expected"), INFINITE_BOUNDS, ids=str)
def test_an_infinite_bound_is_searched_as_far_as_a_search_reaches(
    text, bounds, expected
):
    """The 6.01 changelog allows infinite limits to NSOLVE; here they work for
    every equation, a polynomial's roots being had at once and anything else's
    sought in a window that widens as Mixed mode's search does."""
    assert sol(text, context=APPROXIMATE, bounds=bounds) == expected


def test_a_value_too_small_for_a_float_is_no_root():
    """`#e^-2000` is 0.0 as a float, and still not zero."""
    assert sol("#e^x = 0", context=APPROXIMATE, bounds=("-2000", "10")) == []


def test_a_multiple_root_in_the_interval_comes_back_once():
    """As the exact path collapses it: `(x - 1)^2` has the one solution 1."""
    assert sol("(x - 1)^2 = 0", context=APPROXIMATE, bounds=("0", "3")) == ["x = 1"]
    assert sol("(x - 1)^2*(x + 2) = 0", context=APPROXIMATE, bounds=("-3", "3")) == [
        "x = -2",
        "x = 1",
    ]
    assert simplify(parse("SOLVE((x - 1)^2 = 0, x, 0, 3)"), EXACT).text == "[x = 1]"


def test_mixed_precision_searches_without_being_told_where():
    # Derive prints `-0.686026`, cutting the digit this engine rounds: the
    # root is -0.68602672..., which is `-0.686027` to six significant digits.
    assert sol("3^x = x^2", context=MIXED) == ["x = -0.686026"]
    assert sol("COS(x) = x", context=MIXED) == ["x = 0.739085"]


def test_mixed_precision_solves_exactly_where_it_can():
    assert sol("x^2 - 5*x + 6 = 0", context=MIXED) == ["x = 2", "x = 3"]


@pytest.mark.parametrize("context", [APPROXIMATE, MIXED], ids=lambda c: str(c.precision))
def test_a_complex_solution_is_approximated_and_not_decided(context):
    """The solve variable is real, and an equation of it to a complex number is no
    statement to answer `false`: it is a solution, approximated as Exact gives it."""
    assert sol("x^2 + 3 = 0", context=context) == ["x = 1.73205*#i", "x = -1.73205*#i"]
    assert sol("x^3 - x + 5 = 0", context=context) == [
        "x = -1.90416",
        "x = 0.952080 + 1.31124*#i",
        "x = 0.952080 - 1.31124*#i",
    ]
    assert simplify(parse("SOLVE(x^2 + 3 = 0, x)"), context).text == (
        "[x = 1.73205*#i, x = -1.73205*#i]"
    )


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
    "SOLVE(x^2 + 1 = 0, x, Real)",
    "SOLVE(x^2 + 1 = 0, Real)",
    "SOLVE(x^2 + 1 = 0, x, Reals)",
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
    ("SOLVE(x*EXP(x) = 1, x)", "[x = LAMBERTW(1)]"),
    ("SOLVE(LN(x) + LN(x - 1) = LN(6), x)", "[x = 3]"),
    ("SOLVE(ABS(x) = x, x)", "[x >= 0]"),
    # The head sees what its equation divides by as written, as the command does.
    ("SOLVE((x^2 - 1)/(x - 1) = 2, x)", "[]"),
    ("SOLVE(x^2/x = 0, x)", "[]"),
    # A system contributes one inner vector per solution, so that the outer
    # vector is always the list of entries the command would have appended.
    ("SOLVE([x + y = 3, x - y = 1], [x, y])", "[[x = 2, y = 1]]"),
    # A vector of expressions is a system too, which is how VECTOR.MTH's
    # EXACT_EIGENVECTOR states the one it solves.
    ("SOLVE([a - 1, b - 2], [a, b])", "[[a = 1, b = 2]]"),
    ("SOLVE([q - p, p - q], [p, q])", "[[p = @1, q = @1]]"),
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
    """The original's help says "if in approximate mode"; the shipped files use
    it as
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


# -- real solutions only ------------------------------------------------------
#
# `SOLVE(u, x, Real)` keeps the solutions not having an imaginary component, which
# is Derive 6's third argument. Only a solution known to have one is left out.

REAL = [
    # The 6.1 manual's own example.
    ("SOLVE(x^3 + x^2 + x + 1, x, Real)", "[x = -1]"),
    (
        "SOLVE(x^3 + 3*x + 1 = 0, x, Real)",
        "[x = (4*SQRT(5) - 4)^(1/3)/2 - (4*SQRT(5) + 4)^(1/3)/2]",
    ),
    # Real roots all stay.
    ("SOLVE(x^2 - 4, x, Real)", "[x = -2, x = 2]"),
    # Every solution non-real is no solution at all, as for an equation with none.
    ("SOLVE(x^2 + 1 = 0, x, Real)", "[]"),
    # A periodic family whose members are all complex goes, and one with a real
    # representative stays.
    ("SOLVE(EXP(x) = -2, x, Real)", "[]"),
    ("SOLVE(EXP(x) = 2, x, Real)", "[x = LN(2)]"),
    # A value with a parameter in it is real for some values of the parameter,
    # and stands.
    ("SOLVE(x^2 = a, x, Real)", "[x = -SQRT(a), x = SQRT(a)]"),
    # What is not a list of values is not filtered: an identity, a range, and the
    # residual equation of what could not be solved.
    ("SOLVE(x = x, x, Real)", "[x = @1]"),
    ("SOLVE(x^2 < 4, x, Real)", "[-2 < x < 2]"),
    ("SOLVE(COS(x) = x, x, Real)", "[COS(x) - x = 0]"),
    # A system's solution vector goes where one of its values would.
    (
        "SOLVE([x^2 + y^2 = 2, x - y = 0], [x, y], Real)",
        "[[x = -1, y = -1], [x = 1, y = 1]]",
    ),
    ("SOLVE([x + y = #i, x - y = 0], [x, y], Real)", "[]"),
    # Complex names the default domain, and filters nothing.
    ("SOLVE(x^2 + 1 = 0, x, Complex)", "[x = #i, x = -#i]"),
]


@pytest.mark.parametrize(("text", "expected"), REAL, ids=str)
def test_real_asks_for_the_solutions_without_an_imaginary_component(text, expected):
    assert simplify(parse(text), EXACT).text == expected


def word_mode():
    state = ParseState()
    state.input_mode = InputMode.WORD
    return state


# A call that is none of SOLVE's forms is not answered, and stands as written: a
# misspelled domain answered as though it were absent would answer another question.
STANDING = [
    # A third argument that names no domain.
    "SOLVE(x^2 = 1, x, reals)",
    "SOLVE(x^2 = 1, x, imaginary)",
    "SOLVE(x^2 = 1, x, 5)",
    "SOLVE(x^2 = -1, x, y)",
    # More than four arguments.
    "SOLVE(x^2 = 1, x, -2, 2, 3)",
    # A domain word where the variable belongs.
    "SOLVE(x^2 = 1, Real)",
    "SOLVE(x^2 = 1, Complex)",
]


@pytest.mark.parametrize("text", STANDING, ids=str)
@pytest.mark.parametrize(
    "context", [EXACT, APPROXIMATE, MIXED], ids=lambda c: str(c.precision)
)
def test_a_solve_head_of_no_form_stands_as_written(text, context):
    state = word_mode()
    answer = simplify(parse(text, state), context).text
    assert answer == text
    assert simplify(parse(answer, state), context).text == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("DIMENSION(SOLVE(x^2 = 1, x, Reals))", "DIMENSION(SOLVE(x^2 = 1, x, reals))"),
        ("RHS(SOLVE(x^2 = 1, x, Reals))", "SOLVE(x^2 = 1, x, reals)"),
        ("(SOLVE(x^2 = 1, x, Reals)) SUB 1", "(SOLVE(x^2 = 1, x, reals)) SUB 1"),
    ],
    ids=str,
)
def test_a_standing_solve_head_is_taken_as_the_call_it_is(text, expected):
    """As a standing `RATE` is: `RHS` of what is no relation is the thing itself."""
    assert simplify(parse(text, word_mode()), EXACT).text == expected


def test_an_assigned_third_argument_is_read_for_its_value():
    """What reaches the head is the value: a name assigned a number is no domain, and
    one assigned the word is that word."""
    state = ParseState()
    number = Context(assignments={"y": parse("3", state)})
    assert simplify(parse("SOLVE(x^2 = -1, x, y)", state), number).text == (
        "SOLVE(x^2 = -1, x, 3)"
    )
    word = Context(assignments={"y": parse("Real", state)})
    assert simplify(parse("SOLVE(x^2 = -1, x, y)", state), word).text == "[]"


def test_a_real_root_written_through_the_imaginary_unit_is_kept():
    """Ferrari's radicals reach the four real roots of this quartic through `#i`
    terms that cancel, which evaluate to a residue of an imaginary part and not to
    none; that is no imaginary component."""
    text = "SOLVE(x^4 - 4*x^2 + x + 1 = 0, x, Real)"
    answer = simplify(parse(text), EXACT)
    assert "#i" in answer.text
    assert simplify(parse(f"DIMENSION({text})"), EXACT).text == "4"


def test_real_filters_complex_roots_however_they_are_written():
    assert simplify(parse("DIMENSION(SOLVE(x^4 + x + 1 = 0, x, Real))"), EXACT).text == (
        "0"
    )


def test_real_is_answered_in_every_precision_mode():
    for context in (APPROXIMATE, MIXED):
        assert simplify(parse("SOLVE(x^3 - x + 5 = 0, x, Real)"), context).text == (
            "[x = -1.90416]"
        )


def test_real_does_not_make_a_declared_domain_filter():
    state = ParseState()
    name, domain = domain_of_node(parse("x :epsilon Real", state))
    context = Context(domains={name: domain})
    assert simplify(parse("SOLVE(x^2 + 1, x)"), context).text == "[x = #i, x = -#i]"
