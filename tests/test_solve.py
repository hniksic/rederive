"""What soLve, `SOLVE` and `SOLUTIONS` must produce.

Like the Simplify and Factor cases, these go from author text to author text. soLve
appends a list of entries, so every case gives the whole list; an empty list is a real
expectation.

Almost every value was checked against the original; the rest are quoted from the 3.14
manual and the 6.1 help. Where the engine deliberately differs, its answer is recorded
and the case says why; nothing is weakened to make a test pass. The deliberate
differences are: the order of solutions, which representatives of a periodic family are
shown, and the last digit of a rounded root.
"""

from __future__ import annotations

import itertools

import pytest
import sympy as sp

from rederive.engine.computing import (
    Context,
    domain_of_node,
    simplify,
    solve,
    to_sympy,
)
from rederive.engine.context import Angle, Branch, Precision
from rederive.model.expr import Kind
from rederive.model.session import Session
from rederive.syntax import ParseState, parse_expression
from rederive.syntax.state import CaseMode, InputMode

EXACT = Context()
APPROXIMATE = Context().with_precision(Precision.APPROXIMATE)
MIXED = Context().with_precision(Precision.MIXED)
DEGREES = Context(angle=Angle.DEGREE)


def parse(text, state=None):
    return parse_expression(text, state or ParseState()).node


def sol(text, variables=(), context=None, bounds=None):
    """Solve `text` and return the entries the worksheet would append."""
    interval = None if bounds is None else (parse(bounds[0]), parse(bounds[1]))
    answers = solve(parse(text), context or EXACT, variables, interval)
    return [answer.text for answer in answers]


# -- equations ----------------------------------------------------------------

EQUATIONS = [
    ("2*x + 3 = 7", ["x = 2"]),
    ("x^2 - 5*x + 6 = 0", ["x = 2", "x = 3"]),
    # A bare expression is solved as `u = 0`. Derive prints `2` before `-2`; here reals
    # are ascending (see the test on ordering below).
    ("x^2 - 4", ["x = -2", "x = 2"]),
    # Complex roots are in rectangular form, with the positive imaginary part of a
    # conjugate pair first, as in the original.
    ("x^2 + 1", ["x = #i", "x = -#i"]),
    ("x^3 - 2*x^2 + x - 2", ["x = 2", "x = #i", "x = -#i"]),
    # The double root at zero is listed once.
    ("x^2*(8*x - 9) = 0", ["x = 0", "x = 9/8"]),
    ("LN(x) = 5", ["x = #e^5"]),
    # Two roots of the substituted quadratic, a conjugate pair, cube to the same
    # number, listed once.
    ("x^(1/3)/3 + 3/x^(1/3) = 1", ["x = -27"]),
    # Unsolvable, so the equation is returned without warning or guess, with the
    # number term on the right, as Derive writes it.
    ("x^5 - x + 1 = 0", ["x^5 - x = -1"]),
    ("COS(x) = x", ["COS(x) - x = 0"]),
    # Derive gives up on this too; Lambert W gives its one real root.
    ("3^x = x^2", ["x = -2*LAMBERTW(LN(3)/2)/LN(3)"]),
    # A root on each real branch of Lambert W.
    ("x = LN(x) + 2", ["x = -LAMBERTW(-#e^(-2))", "x = -LAMBERTW(-#e^(-2), -1)"]),
    ("x*EXP(x) = 1", ["x = LAMBERTW(1)"]),
    ("x^x = 2", ["x = #e^LAMBERTW(LN(2))"]),
    # No real root, but there are complex ones, so the equation stays rather than
    # claiming there are no solutions.
    ("x*EXP(x) = -1", ["x*#e^x = -1"]),
    # For negative `a`, `LAMBERTW(a)` is only one of several roots, and nothing proves
    # it a root at all, so this stays too.
    ("x*EXP(x) = a", ["x*#e^x - a = 0"]),
    ("x = x + 1", []),
    # Infinity is not a solution: `1/x` only tends to zero.
    ("1/x = 0", []),
    # The first has two integer roots plus a negative one via Lambert W; the second
    # holds at 1 and at -1, since `(-1)^(-1)` is -1.
    ("2^x = x^2", ["x = -2*LAMBERTW(LN(2)/2)/LN(2)", "x = 2", "x = 4"]),
    ("x^x = x", ["x = -1", "x = 1"]),
]


@pytest.mark.parametrize(("text", "expected"), EQUATIONS, ids=str)
def test_an_equation_solves_to_one_entry_per_solution(text, expected):
    assert sol(text) == expected


#: Rewriting an equation before solving (combining logarithms, squaring a root) can add
#: extraneous candidates, so each one is checked in the original equation. The
#: principal logarithm of a negative number carries `pi*#i`, which rules out negative
#: candidates here: `LN(-2) + LN(-3)` is `LN(6) + 2*pi*#i`.
EXTRANEOUS = [
    ("LN(x) + LN(x - 1) = LN(6)", ["x = 3"]),
    ("LN(x) + LN(x + 1) = LN(2)", ["x = 1"]),
    ("LN(x - 2) + LN(x - 3) = LN(2)", ["x = 4"]),
    ("LN(x) + LN(x - 3) = LN(4)", ["x = 4"]),
    ("2*LN(x) = LN(x + 2)", ["x = 2"]),
    ("2*LN(x) = LN(4)", ["x = 2"]),
    ("LOG(x, 2) + LOG(x - 2, 2) = 3", ["x = 4"]),
    ("LOG(x, 10) + LOG(x + 21, 10) = 2", ["x = 4"]),
    # Both candidates hold: `LN((-3)^2)` and `2*LN(3)` are both `LN(9)`.
    ("LN(x^2) = 2*LN(3)", ["x = -3", "x = 3"]),
    ("LN(x^2 - 1) = LN(3)", ["x = -2", "x = 2"]),
    # Squaring adds `x = 1` and `x = -1`, where the root is the negated right side.
    ("SQRT(x) = x - 2", ["x = 4"]),
    ("SQRT(x + 5) = x - 1", ["x = 4"]),
    ("SQRT(2*x + 3) = x", ["x = 3"]),
    ("SQRT(x) = -2", []),
    # The candidate holds for some values of a parameter.
    ("SQRT(x) = a", ["x = IF(a >= 0, a^2)"]),
    # An inverse trigonometric function takes back only the angles in its range.
    ("ATAN(x) = a", ["x = IF(-pi/2 < a < pi/2, TAN(a))"]),
    ("2*ATAN(x) = a", ["x = IF(-pi < a < pi, TAN(a/2))"]),
    ("ASIN(x) = a", ["x = IF(-pi/2 <= a <= pi/2, SIN(a))"]),
    ("ACOS(x) = 2*a + 1", ["x = IF(-1/2 <= a <= pi/2 - 1/2, COS(2*a + 1))"]),
    ("ASIN(x) = 2", []),
    (
        "2*ATAN(SQRT(3)*(#e^(SQRT(3)*x) - 1)/(3*(#e^(SQRT(3)*x) + 1))) = a",
        [
            "x = IF(-pi < a < pi, SQRT(3)*LN(-(SQRT(3)*TAN(a/2) + 1)"
            "/(SQRT(3)*TAN(a/2) - 1))/3)"
        ],
    ),
    # The principal cube root of -8 is `1 + SQRT(3)*#i`, so there is no solution.
    ("x^(1/3) = -2", []),
    # As a quadratic in `x^(1/3)` this offers -125 too, whose principal cube root is
    # not -5.
    ("x^(2/3) - 2*x^(1/3) - 35 = 0", ["x = 343"]),
]


@pytest.mark.parametrize(("text", "expected"), EXTRANEOUS, ids=str)
def test_a_candidate_that_does_not_hold_is_no_solution(text, expected):
    assert sol(text) == expected


#: `ABS` of a real argument is the argument or its negation, so the equation splits into
#: finitely many cases, each over the range where its signs hold. The answer can be a
#: range, written as an inequality.
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
    # Each periodic family is sampled once, as without the `ABS`.
    ("ABS(SIN(x)) = 1/2", ["x = -5*pi/6", "x = -pi/6", "x = pi/6", "x = 5*pi/6"]),
    # Three `ABS` at once, as a `MAX` on each side gives.
    ("MAX(2 - x^2, x) = MAX(-x, x^3/9)", ["x = -1", "x = 3"]),
]


def test_an_equation_in_abs_keeps_a_cubic_root_of_one_piece():
    """`MAX(2 - x^2, x) = x^3/9` is `x^3 + 9*x^2 - 18 = 0` for `-2 < x < 1`, where one
    of the cubic's roots lies; the other two pieces give -3 and 3."""
    answers = sol("MAX(2 - x^2, x) = x^3/9")
    assert len(answers) == 3
    assert answers[0] == "x = -3" and answers[2] == "x = 3"
    assert "ACOS(-2/3)" in answers[1]


@pytest.mark.parametrize(("text", "expected"), ABSOLUTE, ids=str)
def test_an_equation_in_abs_is_solved_sign_by_sign(text, expected):
    assert sol(text) == expected


def test_an_equation_in_abs_of_a_complex_variable_stands():
    """Over the complexes `ABS(x - 1) = 2` is a circle, which no list of relations can
    describe."""
    state = ParseState()
    name, domain = domain_of_node(parse("x :epsilon Complex", state))
    context = Context(domains={name: domain})
    assert sol("ABS(x - 1) = 2", ("x",), context=context) == ["ABS(x - 1) = 2"]


def test_a_polynomial_in_a_root_without_radical_roots_is_left_unsolved_at_once():
    # In `p^(1/10)` it is of degree 17, irreducible; Approximate searches for the root.
    text = "7*p^(17/10) - 225000*p^(2/5) = 2625000"
    assert sol(text, ("p",)) == [text]
    over = "80*(7*p^(17/10) - 225000*p^(2/5) - 2625000)/p^2"
    assert len(sol(over, ("p",))) == 1
    assert sol("x^(2/3) - 2*x^(1/3) - 35 = 0", ("x",)) == ["x = 343"]


def test_an_abs_of_a_parameter_is_a_nonnegative_constant():
    found = sol("(t^2 - b^2)/(ABS(b) + t) = 2*a^2/(ABS(b) - t)", ("t",))
    assert len(found) == 3
    assert sol("ABS(b)*x + 1 = 0", ("x",)) == ["x = -1/ABS(b)"]


#: A power of a positive base is solved as a power of `#e`. Each root of the quadratic
#: in `b^x` gives a family of complex solutions, represented by the member nearest the
#: origin.
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


def test_powers_of_parameters_are_solved_through_logarithms_too():
    assert sol("a*b^x = c*d^x", ("x",)) == ["x = (LN(c) - LN(a))/(LN(b) - LN(d))"]


def test_a_polynomial_in_a_function_of_the_variable_is_solved_through_its_roots():
    """`SIN(x)^2 + 1` is zero where `SIN(x)` is `#i` or `-#i`; the factors of the
    product are solved each on its own."""
    answers = sol("(x + 1)*(SIN(x)^2 + 1)^2*COS(3*x)^3 = 0")
    assert answers[:3] == ["x = -1", "x = -pi/6", "x = pi/6"]
    assert "x = #i*LN(SQRT(2) + 1)" in answers
    assert len(answers) == 7


EQUAL_ARCS = [
    ("ASIN(x) = ATAN(x)", ["x = 0"]),
    ("ACOS(x) = ATAN(x)", ["x = SQRT(SQRT(5)/2 - 1/2)"]),
    ("ACOS(x) = ASIN(x)", ["x = SQRT(2)/2"]),
]


@pytest.mark.parametrize(("text", "expected"), EQUAL_ARCS, ids=str)
def test_an_equation_between_two_arcs_is_solved_through_their_tangents(
    text, expected
):
    """The negative root of the tangents' equation is no solution of the arcs'."""
    assert sol(text) == expected


def test_a_solution_is_written_in_its_plainer_spelling():
    """No negated quotient over a negative denominator, and no factored radicand when the
    expanded one is shorter."""
    assert sol("2^x = 3^(x - 1)") == ["x = LN(3)/(LN(3) - LN(2))"]
    assert sol("5^x = 2^(x + 1)") == ["x = LN(2)/(LN(5) - LN(2))"]
    assert sol("x^2 + y^2 = r^2", ("y",)) == [
        "y = -SQRT(r^2 - x^2)",
        "y = SQRT(r^2 - x^2)",
    ]


def test_a_logarithm_comes_back_whole():
    """Derive 4.11: `[y=LN(x)]`. Sympy writes the solution as `LN(ABS(x)) +
    #i*PHASE(x)`, the same for nonzero `x`, which then defeats integration."""
    assert sol("#e^y = x", ("y",)) == ["y = LN(x)"]
    assert simplify(parse("SOLVE(#e^y = x, y)"), EXACT).text == "y = LN(x)"
    # The modulus is the same for `-x`, whose phase sympy takes.
    assert simplify(parse("SOLVE(x + #e^(-y) = 0, y)"), EXACT).text == "y = -LN(-x)"


def test_a_variable_alone_is_the_equation_that_it_is_zero():
    # InvLaplace finds the pole of a factor `s` of a denominator so.
    assert simp("SOLVE(x, x)") == "x = 0"
    assert simp("SOLUTIONS(x, x)") == "[0]"


def test_a_solution_in_a_function_with_no_name_leaves_the_equation_unsolved():
    # Sympy's inverse error function can't be typed back; the equation stands instead.
    assert sol("ERF(x) = y", ("x",)) == ["ERF(x) - y = 0"]


def test_a_symbolic_equation_solves_for_whichever_variable_was_chosen():
    assert sol("a*x + b = 0", ("x",)) == ["x = -b/a"]
    assert sol("a*x + b = 0", ("a",)) == ["a = -b/x"]


def test_a_cubic_is_solved_in_radicals():
    """Cardano, symbolically: three exact solutions.

    The form is sympy's, not the original's, so the test checks only that there are
    three exact solutions, each satisfying the equation for sample parameter values.
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
    """The casus irreducibilis: an irreducible cubic with three real roots.

    In radicals the roots can only be written via cube roots of complex numbers, with
    `#i` terms that cancel but can't be eliminated. Viete's trigonometric form avoids
    this and matches the original.
    """
    assert sol("x^3 - 3*x + 1 = 0", ("x",)) == [
        "x = -2*COS(pi/9)",
        "x = 2*COS(4*pi/9)",
        "x = 2*COS(2*pi/9)",
    ]


def test_one_real_root_is_written_in_two_real_cube_roots():
    """Cardano's formula as the original writes it: the real root is `u + v`, a sum of
    real cube roots, with each radicand's denominator taken out of the root and a
    negative radicand's sign moved in front. The complex pair uses the same two roots."""
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
        # With a square term, the shift follows the cube roots.
        (
            "x^3 + x^2 + 1 = 0",
            "x = -(116 - 12*SQRT(93))^(1/3)/6 - (12*SQRT(93) + 116)^(1/3)/6 - 1/3",
        ),
        (
            "2*x^3 - 3*x^2 + 5*x + 7 = 0",
            "x = (3*SQRT(27273) - 486)^(1/3)/6 - (3*SQRT(27273) + 486)^(1/3)/6 + 1/2",
        ),
        # A cube after shifting, with one radicand zero.
        ("x^3 + 3*x^2 + 3*x - 1 = 0", "x = 2^(1/3) - 1"),
    ],
    ids=str,
)
def test_each_root_of_a_cubic_with_one_real_root_solves_it(text, real):
    """The real root is pinned, and all three roots are checked to satisfy the equation
    and be fixed points of Simplify; the real one is the only real one."""
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
    assert simplify(parse(f"SOLVE({text}, x, Real)"), EXACT).text == real


def test_a_cubic_factor_is_written_as_the_cubic_is():
    assert sol("(x - 2)*(x^3 + 3*x + 1) = 0")[:2] == [
        "x = (4*SQRT(5) - 4)^(1/3)/2 - (4*SQRT(5) + 4)^(1/3)/2",
        "x = 2",
    ]


def test_a_polynomial_is_solved_factor_by_factor_where_one_factor_is_not():
    """A factor whose roots have no radicals stays as its equation beside the roots of
    the others, rather than the whole equation coming back unsolved."""
    assert sol("(x - 1)*(x^5 + x + 3) = 0") == ["x = 1", "x^5 + x = -3"]
    # Multiplied out, with a parameter.
    answers = sol("x^7 + x^6 + x^5 + x^3 + (a + 1)*x^2 + (a + 1)*x + a = 0")
    assert len(answers) == 3 and answers[2] == "x^5 + x + a = 0"


def test_the_order_of_several_solutions_is_ours_and_not_the_originals():
    """Reals ascending, then complex ones. Derive's order is an artifact of its solver
    (`2, 3` for `x^2 - 5*x + 6` but `2, -2` for `x^2 - 4`), so there is no rule to
    follow, and a consistent one is better."""
    assert sol("x^2 - 5*x + 6 = 0") == ["x = 2", "x = 3"]
    assert sol("x^2 - 4") == ["x = -2", "x = 2"]


def test_complex_solutions_go_round_the_circle():
    """Complex solutions are ordered by angle from the positive real axis, keeping
    conjugate pairs together with the positive imaginary part first."""
    assert sol("x^4 = -4") == ["x = 1 + #i", "x = 1 - #i", "x = -1 + #i", "x = -1 - #i"]
    assert sol("x^4 = 16") == ["x = -2", "x = 2", "x = 2*#i", "x = -2*#i"]


def test_a_binomial_is_solved_as_one_real_root_times_the_roots_of_unity():
    """The real root appears once per solution, as a factor, not once per term."""
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
# Neither Derive nor this engine can write `x = 2*pi*n`, so a periodic equation gets a
# finite sample near the origin. Derive's output is redundant (`COS(x) = 1/2` gives
# `pi/3`, `-pi/3` and `5*pi/3`, the last two in one family); here each family appears
# once.


def test_a_periodic_equation_is_sampled_once_per_family():
    # Derive prints `pi/6`, `5*pi/6` and `-7*pi/6`, the third being the second shifted
    # by a period.
    assert sol("SIN(x) = 1/2") == ["x = pi/6", "x = 5*pi/6"]
    # Derive prints `0`, `pi` and `-pi`.
    assert sol("SIN(x) = 0") == ["x = 0", "x = pi"]
    # Derive prints `pi/3`, `-pi/3` and `5*pi/3`.
    assert sol("COS(x) = 1/2") == ["x = -pi/3", "x = pi/3"]


def test_a_family_shared_by_two_others_adds_no_solution():
    """Sympy also lists where the two families of `COS(t) = c` meet, which are members
    of both."""
    text = "SOLUTIONS(x = (a^2 - r^2)*COS(t)/(2*(a*COS(t) - r)), t)"
    assert simp(text) == (
        "[-ACOS(2*r*x/(2*a*x - a^2 + r^2)), ACOS(2*r*x/(2*a*x - a^2 + r^2))]"
    )


def test_an_infinite_side_is_no_value_reached_at_a_pole():
    """An equation with an infinite or undefined side has no solution, though the other
    side has a pole: a pole is no point where a function takes a value."""
    assert simp("SOLVE(TAN(2*a) = 4*SQRT(3)/0, a)") == "false"
    assert simp("SOLVE(1/(x^2 + 1) = 1/0, x)") == "false"
    assert simp("SOLVE(x = 1/0, x)") == "false"


def test_a_solution_not_known_to_be_inside_the_bounds_is_kept_under_that_condition():
    assert simp("SOLVE(x^2 = 4, x, 0, 3)") == "x = 2"
    assert simp("SOLVE(x^2 = -4, x, 0, 3)") == "false"
    assert simp("SOLVE(x^2 = a, x, 1, 2)") == (
        "x = SQRT(a) AND 1 <= SQRT(a) AND SQRT(a) <= 2"
    )


def test_a_solution_angle_is_shown_in_whatever_unit_the_angle_setting_says():
    assert sol("SIN(x) = 1/2", context=DEGREES) == ["x = 30 DEG", "x = 150 DEG"]
    assert sol("SIN(x) = a", context=DEGREES) == ["x = pi - ASIN(a)", "x = ASIN(a)"]
    assert simp("SOLVE(TAN(a) = 1, a, Real)", DEGREES) == "a = 45 DEG"


# -- identities ---------------------------------------------------------------


def test_an_identity_solves_to_an_arbitrary_value():
    assert sol("x = x") == ["x = @1"]


def test_an_assignment_or_a_definition_is_solved_for_its_value():
    assert sol("a := x^2 - 9", ("x",)) == ["x = -3", "x = 3"]
    assert sol("W(k) := k^2 - 4") == ["k = -2", "k = 2"]


@pytest.mark.parametrize(
    "text", ["x/x = 1", "x/(x + 2) + 2/(x + 2) = 1", "SOLVE(x/x = 1)"]
)
def test_an_equation_whose_variable_cancels_is_solved_for_it(text):
    """`x/x = 1` cancels to `1 = 1` before a variable is chosen, but is still an equation
    in `x`: every value except the poles is a solution."""
    if text.startswith("SOLVE"):
        assert simplify(parse(text), EXACT).text == "true"
    else:
        assert sol(text) == ["x = @1"]


@pytest.mark.parametrize(
    "text",
    [
        "(x^2 - 1)/(x - 1) = x + 1",
        # Zero over a common denominator, for which solveset finds no solution.
        "1/(x - 1) + 1 = x/(x - 1)",
        "1/x + 1/(x*(x - 1)) = 1/(x - 1)",
    ],
)
def test_a_rational_identity_solves_to_an_arbitrary_value(text):
    """Every value where both sides are defined is a solution, and the arbitrary value
    is the only way to express "everything except the poles"."""
    assert sol(text) == ["x = @1"]


def test_the_arbitrary_counter_comes_from_the_context():
    """This lets the session keep one counter across commands, while the answer depends
    only on the tree and the context."""
    assert sol("x = x", context=Context(arbitrary_index=2)) == ["x = @2"]
    assert sol("x = x", context=Context(arbitrary_index=7)) == ["x = @7"]


# -- inequalities -------------------------------------------------------------


INEQUALITIES = [
    # An interval is one chained entry.
    ("x^2 < 4", ["-2 < x < 2"]),
    ("x^2 <= 4", ["-2 <= x <= 2"]),
    # A union gives one entry per piece, like several roots.
    ("ABS(3*x + 2) > 5", ["x < -7/3", "x > 1"]),
    # A half-line is a single relation, with the variable on the left.
    ("2*x > 6", ["x > 3"]),
    ("2*x >= 6", ["x >= 3"]),
    # The variable is isolated, not left inside `-3 < x - 2 < 3`.
    ("ABS(x - 2) < 3", ["-1 < x < 5"]),
    # A periodic inequality is sampled once per family, like a periodic equation.
    ("SIN(x) > 0", ["0 < x < pi"]),
]


@pytest.mark.parametrize(("text", "expected"), INEQUALITIES, ids=str)
def test_an_inequality_solves_to_the_range_it_describes(text, expected):
    assert sol(text) == expected


def test_a_symbolic_inequality_turns_under_a_negative_coefficient():
    assert sol("-2*x + 3*y <= 7", ("x",)) == ["x >= (3*y - 7)/2"]


def test_an_unequality_is_solved_too():
    """Every value but the real solutions of the equation."""
    assert sol("x /= 3") == ["x /= 3"]
    assert sol("x^2 /= 4") == ["x /= -2 AND x /= 2"]
    assert sol("x^2 /= -4") == ["x = @1"]
    assert sol("x /= x") == []


def test_an_inequality_that_always_holds_is_an_identity():
    assert sol("x^2 >= 0") == ["x = @1"]


def test_inequalities_in_several_variables_are_left_unsolved():
    # The solutions are a region, which no list of values describes.
    assert sol("[x + y > 0, x - y < 0]", ("x", "y")) == ["x + y > 0 AND x - y < 0"]


def test_an_unsolved_inequality_keeps_its_operator():
    """The residual of a `<` is still a `<`, not an equation."""
    assert sol("x^3 < SIN(x)") == ["x^3 - SIN(x) < 0"]


# -- the sets sympy answers with ----------------------------------------------


#: One case per set head sympy returns, with the equivalent Derive relations. Author
#: notation has no sets, so `Interval` becomes a chain, `Union` one entry per piece,
#: `EmptySet` no entries, `FiniteSet` one entry per element, and the whole complex
#: plane `@n`. `Complement` has its own section below, since its result depends on
#: what can be proved about the excluded values.
SETS = [
    ("x^2 < 1", ["-1 < x < 1"]),
    ("x^2 > 1", ["x < -1", "x > 1"]),
    ("#e^x = 0", []),
    ("x^2 + 1 = 0", ["x = #i", "x = -#i"]),
    ("x^2 + 1 > 0", ["x = @1"]),
]


@pytest.mark.parametrize(("text", "expected"), SETS, ids=str)
def test_a_solution_set_is_written_in_the_notation_derive_has(text, expected):
    """The solver must convert sets to relations.

    Otherwise a set reaches the printer and is shown as inert text like
    `Interval(-1, 1)`, or worse, read back as the notation's `INTERVAL`, a limit's
    range, which looks like an answer but isn't.

    So each entry is checked to be a real relation, not `Kind.STRING`, as well as for
    its text.
    """
    answers = solve(parse(text), EXACT)
    assert [answer.text for answer in answers] == expected
    for answer in answers:
        assert answer.node.kind is not Kind.STRING


def test_a_chained_answer_reads_back_as_what_it_says():
    """Results are reparsed from their text, so the chain must be valid syntax. It
    nests to the left, which the renderer already draws flat."""
    answer = solve(parse("x^2 < 4"), EXACT)[0]
    assert answer.text == "-2 < x < 2"
    assert answer.node.kind is parse("-2 < x < 2").kind
    assert answer.node == parse("-2 < x < 2")


# -- excluded values ----------------------------------------------------------
#
# Typically a rational equation with a parameter in the numerator: sympy answers
# `Complement({2*m}, {0})` for `1 - 2*m/r = 0`, since clearing the denominator requires
# `r` to be nonzero. A candidate is dropped only if provably excluded, which matches
# the original's answers.


EXCLUSIONS = [
    # The original answers `[r = 2*m]`, and so does this: `2*m` is not provably zero.
    # Keeping it hides nothing, since with `m` zero there is no solution at all.
    ("1 - 2*m/r = 0", "r", ["r = 2*m"]),
    ("1 - 2*0/r = 0", "r", []),
    # The only candidate is provably excluded: no solution.
    ("1/r = 0", "r", []),
    ("a/x = b", "x", ["x = a/b"]),
    ("(x^2 - a^2)/(x*(x - a)) = 0", "x", ["x = -a"]),
    ("1/x + 1/(x + a) = 0", "x", ["x = -a/2"]),
    ("(r^2 - a)/(r - 1) = 0", "r", ["r = -SQRT(a)", "r = SQRT(a)"]),
    # Simplify cancels these before solving, and the remaining candidate is where the
    # original equation divides by zero.
    ("(x^2 - 1)/(x - 1) = 2", "x", []),
    ("x^2/x = 0", "x", []),
    ("(x - 1)^2/(x - 1) = 0", "x", []),
    ("x^(-1)*x^2 = 0", "x", []),
    ("x/(x - 2) = 2/(x - 2)", "x", []),
    ("(x^2 - 1)/(x - 1) = 3", "x", ["x = 2"]),
    ("x^2*(x - 2)/(x - 2) = 0", "x", ["x = 0"]),
    # `r = 2*m` divides by zero only for one value of `m`, so it isn't provably excluded.
    ("(1 - 2*m/r)*r/r = 0", "r", ["r = 2*m"]),
    # The divisor vanishes on a family of its own, which excludes no member of the
    # solution families.
    ("4/(3*(COS(t) - 1)) + 4 = 0", "t", ["t = -ACOS(2/3)", "t = ACOS(2/3)"]),
]


@pytest.mark.parametrize(("text", "variable", "expected"), EXCLUSIONS, ids=str)
def test_a_candidate_survives_unless_it_is_provably_excluded(text, variable, expected):
    assert sol(text, (variable,)) == expected


def test_a_declaration_that_settles_the_exclusion_answers_the_same():
    """Declaring the parameter nonzero settles the exclusion, and the answer must not
    change: a candidate kept for lack of proof is the same as one proven valid."""
    state = ParseState()
    name, domain = domain_of_node(parse("m :epsilon Real (0, inf)", state))
    context = Context(domains={name: domain})
    assert sol("1 - 2*m/r = 0", ("r",), context=context) == ["r = 2*m"]


def test_an_exclusion_from_a_solution_family_is_left_unanswered():
    """`SIN(x)/(x - a) = 0` excludes `a` from the multiples of pi. An infinite family
    minus one member can't be written as a finite list of relations, so the equation is
    returned, as its numerator: where the denominator vanishes there is no solution."""
    assert sol("SIN(x)/(x - a) = 0", ("x",)) == ["SIN(x) = 0"]


# -- systems ------------------------------------------------------------------


SYSTEMS = [
    ("[x + y = 3, x - y = 1]", (), ["x = 2 AND y = 1"]),
    # Linear in the solution variables, arbitrary in everything else.
    (
        "[2*a^2*x + 3*y = 7, x - 5*y = 0]",
        (),
        ["x = 35/(10*a^2 + 3) AND y = 7/(10*a^2 + 3)"],
    ),
    # A singular consistent system is parametrized, with the free variable replaced by
    # an arbitrary value.
    ("[x + 3*y = 1, 2*x + 6*y = 2]", (), ["x = @1 AND y = (1 - @1)/3"]),
    # An inconsistent system has no solutions.
    ("[x + 3*y = 1, 2*x + 6*y = 3]", (), []),
    # Variables not solved for remain on the right side.
    ("[x + y + z = 1, x - y = 0]", ("x", "y"), ["x = (1 - z)/2 AND y = (1 - z)/2"]),
    ("[x + y + z = 1, x - y = 0]", ("x", "z"), ["x = y AND z = 1 - 2*y"]),
    # An expression in a system means `u = 0`, as on its own.
    ("[a - 1, b - 2]", ("a", "b"), ["a = 1 AND b = 2"]),
    ("[x - y, y = 2]", ("x", "y"), ["x = 2 AND y = 2"]),
    ("[0, 0]", ("x", "y"), ["x = @1 AND y = @2"]),
    # A conjunction of equations is a system too.
    ("x + y = 3 AND x - y = 1", ("x", "y"), ["x = 2 AND y = 1"]),
    # So is an equation between matrices, here between column vectors.
    ("[[1, 2], [3, 4]] . [[x], [y]] = [[1], [2]]", ("x", "y"), ["x = 0 AND y = 1/2"]),
]


@pytest.mark.parametrize(("text", "variables", "expected"), SYSTEMS, ids=str)
def test_a_system_solves_to_one_entry_holding_the_solution_vector(
    text, variables, expected
):
    assert sol(text, variables) == expected


def test_a_system_nonlinear_in_its_variables_is_solved_too():
    """Beyond the original, which only solves linear systems and refers the rest to a
    five-step manual procedure. One entry per solution, as elsewhere."""
    assert sol("[x^2 + y^2 = 1, y = x]", ("x", "y")) == [
        "x = -SQRT(2)/2 AND y = -SQRT(2)/2",
        "x = SQRT(2)/2 AND y = SQRT(2)/2",
    ]
    assert sol("[x*y = 6, x + y = 5]", ("x", "y")) == [
        "x = 2 AND y = 3",
        "x = 3 AND y = 2",
    ]


def test_a_nonlinear_system_gives_its_free_variables_arbitrary_values():
    """As a singular linear system does, each solution with its own."""
    assert sol("[x*y = 1]", ("x", "y")) == ["x = 1/@1 AND y = @1"]
    assert sol("[x^2 + y^2 = t^2]", ("x", "y", "t")) == [
        "x = @1 AND y = @2 AND t = -SQRT(@1^2 + @2^2)",
        "x = @3 AND y = @4 AND t = SQRT(@3^2 + @4^2)",
    ]


def test_a_nonlinear_system_with_a_cubic_resolvent_has_all_four_solutions():
    """Eliminating `y` leaves a quartic with the root 2 and an irreducible cubic.

    The cubic's three real roots are written in Viete's trigonometric form, both `x`
    and `y` as a root of its own eliminating cubic, as a single equation writes them.
    """
    answers = sol("[x^2 + y = 7, x + y^2 = 11]", ("x", "y"))
    assert len(answers) == 4
    assert "x = 2 AND y = 3" in answers
    assert (
        "x = -2*SQRT(34)*SIN(pi/6 - ACOS(317*SQRT(34)/2312)/3)/3 - 2/3"
        " AND y = 8*SQRT(3)*COS(ACOS(69*SQRT(3)/128)/3)/3 - 1"
    ) in answers
    assert not any("#i" in answer for answer in answers)


def test_a_polynomial_system_writes_each_variable_as_a_root_of_its_eliminant():
    """`a`, `b` and `c` are each a root of `t^3 - 3*t^2 + 1`, written in cosines as
    for `SOLVE(t^3 - 3*t^2 + 1 = 0, t)`, and the solutions are its six permutations.
    Solving the basis by substitution would write `b` through a quadratic in Cardano's
    complex cube roots of `c`."""
    answers = sol(
        "[a^2 + b^2 + c^2 = 9, a^3 + b^3 + c^3 = 24, a + b + c = 3]", ("a", "b", "c")
    )
    roots = ["2*COS(pi/9) + 1", "1 - 2*COS(4*pi/9)", "1 - 2*COS(2*pi/9)"]
    assert sorted(answers) == sorted(
        f"a = {p} AND b = {q} AND c = {r}"
        for p, q, r in itertools.permutations(roots)
    )
    # A cubic with one real root has it and the complex pair in Cardano's radicals.
    answers = sol(
        "[a^2 + b^2 + c^2 = 9, a^3 + b^3 + c^3 = 30, a + b + c = 3]", ("a", "b", "c")
    )
    assert len(answers) == 6
    assert all(answer.count("#i") == 2 for answer in answers)
    # A rational root and two quadratic ones.
    assert "a = 1 AND b = 1 - SQRT(2) AND c = SQRT(2) + 1" in sol(
        "[a^2 + b^2 + c^2 = 7, a^3 + b^3 + c^3 = 15, a + b + c = 3]", ("a", "b", "c")
    )


def test_a_polynomial_system_keeps_its_non_real_solutions():
    """As for `SOLVE(x^5 + x + 1 = 0, x)`: the roots of `x^2 + x + 1` are as much
    solutions as the complex roots of the cubic factor."""
    answers = sol("[x^5 + x + y = 0, y = 1]", ("x", "y"))
    assert len(answers) == 5
    assert "x = -1/2 - SQRT(3)*#i/2 AND y = 1" in answers


def test_a_polynomial_system_keeps_the_roots_of_a_quintic_as_its_equations():
    """Besides `[x = -1, y = 1, z = 1]`, `[x = 1, y = 1, z = 1]` and four non-real
    solutions on `3*z^2 - 2*z + 1`, `z` can be a root of a quintic with no radicals.
    Those solutions come last, as the reduced equations of that part alone, with `y`
    written in `z`."""
    eqn = "[x^2*y + 3*y*z = 4, 2*y^2 - 3*x^2*z = -1, 2*y*z^2 - z^2 = 1]"
    answers = sol(eqn, ("x", "y", "z"))
    assert len(answers) == 7
    assert answers[0] == "x = -1 AND y = 1 AND z = 1"
    rest = answers[-1]
    assert rest.endswith(" AND 6*z^5 - 6*z^4 - 9*z^3 - 7*z^2 - 3*z = 1")
    assert " AND y = " in rest and "x = " not in rest


def test_a_system_in_shape_position_writes_the_others_in_the_last_variable():
    """`y = 0` makes `x` a root of the quintic, written as its equation, as for `SOLVE((x
    - 1)*(x^5 + x + 3) = 0, x)`."""
    assert sol("[y = x^5 + x + 3, (x - 1)*y = 0]", ("y", "x")) == [
        "y = 5 AND x = 1",
        "y = 0 AND x^5 + x = -3",
    ]
    assert sol("[x^5 + x + 3 = y, x = y + 1]", ("x", "y")) == [
        "x = y + 1 AND y^5 + 5*y^4 + 10*y^3 + 10*y^2 + 5*y = -5"
    ]


def test_a_complex_equation_in_real_variables_is_split_into_its_parts():
    state = ParseState()
    domains = dict(
        domain_of_node(parse(f"{name} :epsilon Real", state)) for name in "xy"
    )
    context = Context(domains=domains)
    assert sol("3*x - (2 + #i)*y = -3*#i", ("x", "y"), context=context) == [
        "x = 2 AND y = 3"
    ]


def test_the_conjugate_of_a_complex_variable_makes_two_real_unknowns():
    state = ParseState()
    name, domain = domain_of_node(parse("z :epsilon Complex", state))
    context = Context(domains={name: domain})
    eqn = "(1 + #i)*z + (2 - #i)*CONJ(z) = -3*#i"
    assert sol(eqn, ("z",), context=context) == ["z = 2 + 3*#i"]


def test_a_complex_variable_of_given_modulus_is_on_a_circle():
    state = ParseState()
    name, domain = domain_of_node(parse("z :epsilon Complex", state))
    context = Context(domains={name: domain})
    assert sol("ABS(z) = 2", ("z",), context=context) == ["z = 2*unit_circle"]
    assert sol("3*ABS(z) - 6 = 0", ("z",), context=context) == ["z = 2*unit_circle"]
    assert sol("ABS(z) = -1", ("z",), context=context) == []
    # A real variable of given absolute value is one of two numbers.
    assert sol("ABS(x) = 5", ("x",)) == ["x = -5", "x = 5"]


def test_a_system_nobody_can_solve_falls_back_to_its_own_residuals():
    """Never "no solutions": failing to solve is different from having no solutions."""
    assert sol("[3^x = x^2, LN(y) = COS(y)]", ("x", "y")) == [
        "3^x - x^2 = 0 AND LN(y) - COS(y) = 0"
    ]


# -- the numeric modes --------------------------------------------------------


def test_approximate_precision_searches_the_interval_it_was_given():
    assert sol("x^2 - 5*x + 6 = 0", context=APPROXIMATE, bounds=("0", "2.5")) == [
        "x = 2"
    ]
    assert sol("x^2 - 5*x + 6 = 0", context=APPROXIMATE, bounds=("10", "20")) == []


def test_a_numeric_root_is_shown_to_the_precision_digits():
    assert sol("x^5 - x + 1 = 0", context=APPROXIMATE, bounds=("-10", "10")) == [
        "x = -1.167303978"
    ]


def test_every_real_root_in_the_interval_comes_back_not_just_one():
    """The original returns one root even if the interval has several (`x = 3` for
    `x^2 - 5*x + 6` over `[-10, 10]`). For a polynomial, finding all costs no more."""
    assert sol("x^2 - 5*x + 6 = 0", context=APPROXIMATE, bounds=("-10", "10")) == [
        "x = 2",
        "x = 3",
    ]


INFINITE_BOUNDS = [
    ("COS(x) = x", ("-inf", "inf"), ["x = 0.7390851332"]),
    ("COS(x) = x", ("0", "inf"), ["x = 0.7390851332"]),
    ("#e^x = 3 - x", ("-inf", "inf"), ["x = 0.7920599684"]),
    ("x^2 - 2 = 0", ("-inf", "inf"), ["x = -1.414213562", "x = 1.414213562"]),
    # Interval arithmetic proves that `COS(x) - x` is positive below -5. A search that
    # finds nothing and proves nothing leaves the equation unsolved rather than claiming
    # there is no root: `#e^x` comes ever closer to 0.
    ("COS(x) = x", ("-inf", "-5"), []),
    ("#e^x = 0", ("-inf", "inf"), ["#e^x = 0"]),
]


@pytest.mark.parametrize(("text", "bounds", "expected"), INFINITE_BOUNDS, ids=str)
def test_an_infinite_bound_is_searched_as_far_as_a_search_reaches(
    text, bounds, expected
):
    """The 6.01 changelog allows infinite limits for NSOLVE; here they work for every
    equation. Polynomial roots are found directly; others are searched in a widening
    window, as in Mixed mode."""
    assert sol(text, context=APPROXIMATE, bounds=bounds) == expected


def test_a_value_too_small_for_a_float_is_no_root():
    """`#e^-2000` is 0.0 as a float, and still not zero."""
    assert sol("#e^x = 0", context=APPROXIMATE, bounds=("-2000", "10")) == []


def test_a_multiple_root_in_the_interval_comes_back_once():
    """As in exact mode, `(x - 1)^2` has the single solution 1."""
    assert sol("(x - 1)^2 = 0", context=APPROXIMATE, bounds=("0", "3")) == ["x = 1"]
    assert sol("(x - 1)^2*(x + 2) = 0", context=APPROXIMATE, bounds=("-3", "3")) == [
        "x = -2",
        "x = 1",
    ]
    assert simplify(parse("SOLVE((x - 1)^2 = 0, x, 0, 3)"), EXACT).text == "x = 1"


def test_mixed_precision_searches_without_being_told_where():
    # Derive prints `-0.686026`, truncating where this engine rounds: the root is
    # -0.68602672..., which is `-0.686027` to six significant digits.
    assert sol("3^x = x^2", context=MIXED) == ["x = -0.6860267245"]
    assert sol("COS(x) = x", context=MIXED) == ["x = 0.7390851332"]


def test_approximate_precision_searches_where_a_solution_has_no_name():
    """Derive 4.11: `[z=0.272462]`. The exact solution is in sympy's inverse error
    function, which can't be written; under Approximate the search finds the number. This
    is CALLOPTN.MTH's `STD_DEV`, a SOLVE for the volatility in a Black-Scholes price."""
    assert sol("ERF(z) = 0.3", context=APPROXIMATE) == ["z = 0.2724627147"]
    assert sol("NORMAL(z) = 0.6", context=APPROXIMATE) == ["z = 0.2533471031"]
    assert sol(
        "47.5614*ERF(0.353553*z - 0.0353553/z) + 50*ERF(0.353553*z + 0.0353553/z)"
        " = 8.01207",
        context=APPROXIMATE,
    ) == ["z = 0.2000003997"]
    assert sol("ERF(z) = 0.3") == ["10*ERF(z) = 3"]


POLES = [
    # Derive 4.11: `[z=0.284708]`. Each is undefined at 0, a point of the scan, and has
    # its root between 0 and the next point.
    ("ATAN(z - 0.1/z) + ATAN(z + 0.1/z) = 0.5", ["z = 0.2847079663"]),
    (
        "47.5614*ERF(0.353553*z - 0.0353553/z) + 50*ERF(0.353553*z + 0.0353553/z)"
        " - 8.01207 = 0",
        ["z = 0.2000003997"],
    ),
    # `ATAN(z) + ATAN(1/z)` is `pi/2` or `-pi/2`.
    ("ATAN(z) + ATAN(1/z) = 1", []),
]


@pytest.mark.parametrize(("text", "expected"), POLES, ids=str)
def test_a_root_beside_a_discontinuity_is_found(text, expected):
    """The manual: "Even if the equation is discontinuous in the interval [...] the soLve
    command usually finds a solution if one exists in the interval"."""
    assert sol(text, ("z",), APPROXIMATE, ("-10", "10")) == expected


def test_mixed_precision_solves_exactly_where_it_can():
    assert sol("x^2 - 5*x + 6 = 0", context=MIXED) == ["x = 2", "x = 3"]


@pytest.mark.parametrize("context", [APPROXIMATE, MIXED], ids=lambda c: str(c.precision))
def test_a_complex_solution_is_approximated_and_not_decided(context):
    """Although the solve variable is real, a complex solution is not answered `false`:
    it is approximated from the exact solution."""
    assert sol("x^2 + 3 = 0", context=context) == [
        "x = 1.732050807*#i",
        "x = -1.732050807*#i",
    ]
    assert sol("x^3 - x + 5 = 0", context=context) == [
        "x = -1.904160859",
        "x = 0.9520804295 + 1.311248044*#i",
        "x = 0.9520804295 - 1.311248044*#i",
    ]
    assert simplify(parse("SOLVE(x^2 + 3 = 0, x)"), context).text == (
        "x = -1.732050807*#i OR x = 1.732050807*#i"
    )


# -- domains ------------------------------------------------------------------


def test_a_declared_domain_does_not_filter_the_solutions():
    """As the manual states, a declaration affects how answers simplify, not which
    answers are wanted. Solutions are always sought over the complexes."""
    state = ParseState()
    name, domain = domain_of_node(parse("x :epsilon Real", state))
    context = Context(domains={name: domain})
    assert sol("x^2 + 1", context=context) == ["x = #i", "x = -#i"]


# -- the promises -------------------------------------------------------------

# Every kind of node the parser produces. None may raise; the result doesn't matter.
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


def test_a_variable_the_expression_does_not_hold_is_solved_by_a_condition_on_the_rest():
    # Every `w` solves it once `x^2 = 4`, so "no solutions" would be wrong; a false
    # equation has none.
    assert sol("x^2 - 4", ("w",)) == ["x^2 = 4"]
    assert sol("v = 2*u + 1", ("y",)) == ["2*u - v = -1"]
    assert sol("1 = 2", ("y",)) == []


# -- SOLVE as an authored function --------------------------------------------
#
# `SOLVE(u, x)` answers with a Boolean expression equivalent to `u`: one disjunct per
# solution, `true` when every value is a solution and `false` when none is. The order
# of the solutions is the engine's, so answers are compared as sets of disjuncts.


def named(value):
    """A test id: the input, and not the set of answers, whose order varies."""
    return value if isinstance(value, str) else "-"


def words():
    """Word mode, case-sensitive, so names like `Ieff` are read and written as typed."""
    state = ParseState()
    state.input_mode = InputMode.WORD
    state.case_mode = CaseMode.SENSITIVE
    return state


def simp(text, context=EXACT, state=None):
    return simplify(parse(text, state or words()), context).text


def disjuncts(text, context=EXACT, state=None):
    """The disjuncts of what `text` simplifies to, as a set."""
    return set(simp(text, context, state).split(" OR "))


def elements(text, context=EXACT):
    """The elements of the vector `text` simplifies to, as a set."""
    answer = simp(text, context)
    assert answer.startswith("[") and answer.endswith("]"), answer
    inner = answer[1:-1]
    parts, depth, current = [], 0, ""
    for char in inner:
        depth += char in "(["
        depth -= char in ")]"
        if char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current.strip())
    return set(parts)


ANSWERS = [
    # The 6.1 help's examples.
    ("SOLVE(x^2 - 1 = 0, x)", {"x = -1", "x = 1"}),
    ("SOLVE(x^3 + x^2 + x + 1, x)", {"x = -#i", "x = #i", "x = -1"}),
    ("SOLVE(x^3 + x^2 + x + 1, x, Real)", {"x = -1"}),
    ("SOLVE(x = x + 1, x)", {"false"}),
    ("SOLVE(ABS(x - 2) >= 0, x)", {"true"}),
    ("SOLVE(1680 = 48*q, q)", {"q = 35"}),
    ("SOLVE(57120 = p*48 + 35*1680, p)", {"p = -35"}),
    ("SOLVE(x^2 - 4*x + 5, x)", {"x = 2 - #i", "x = 2 + #i"}),
    (
        "SOLVE(x^2 + x/4 + 2 = 0, x)",
        {"x = -1/8 - SQRT(127)*#i/8", "x = -1/8 + SQRT(127)*#i/8"},
    ),
    ("SOLVE(2*x^2 + x + 4 = 0, x, Real)", {"false"}),
    # A system's solution is a conjunction in the order of the variables.
    ("SOLVE(x^2 + y^2 = 2*a^2 AND x - y = 0, [x, y])", {
        "x = -a AND y = -a", "x = a AND y = a"
    }),
    ("SOLVE(4*x + y = 8 AND 8*x + 3*y = 48, [x, y])", {"x = -6 AND y = 32"}),
    # A Boolean answer is a valid first argument.
    ("SOLVE(SOLVE(x^2 = 4, x), x)", {"x = -2", "x = 2"}),
    ("SOLVE(x^2 - 1 = 0 OR x = 5, x)", {"x = 5", "x = -1", "x = 1"}),
    ("SOLVE(x*EXP(x) = 1, x)", {"x = LAMBERTW(1)"}),
    ("SOLVE(LN(x) + LN(x - 1) = LN(6), x)", {"x = 3"}),
    ("SOLVE(ABS(x) = x, x)", {"x >= 0"}),
    # Like the command, the head sees the divisions in the equation as written.
    ("SOLVE((x^2 - 1)/(x - 1) = 2, x)", {"false"}),
    ("SOLVE(x^2/x = 0, x)", {"false"}),
    ("SOLVE(z + 5.95897 + 0.00759*#i, z)", {"z = -595897/100000 - 759*#i/100000"}),
]


@pytest.mark.parametrize(("text", "expected"), ANSWERS, ids=named)
def test_solve_answers_with_a_disjunction_of_solutions(text, expected):
    assert disjuncts(text) == expected


def test_a_solution_of_a_trigonometric_system_is_rationalized():
    # Reuleaux.dfw (Derive User Group Newsletter 108), as Derive gives it.
    text = (
        "SOLVE(x = -a*COS(t) - SQRT(3)*a*SIN(t) + a AND y = SQRT(3)*a*COS(t)"
        " + a*SIN(t) - a, [y, t])"
    )
    assert disjuncts(text) == {
        "y = -(SQRT(-x^2 + 2*a*x + 3*a^2) + SQRT(3)*x - (SQRT(3) - 2)*a)/2 AND t ="
        " 2*ATAN((SQRT(-x^2 + 2*a*x + 3*a^2) + SQRT(3)*a)/(2*a - x))",
        "y = (SQRT(-x^2 + 2*a*x + 3*a^2) - SQRT(3)*x + (SQRT(3) - 2)*a)/2 AND t ="
        " 2*ATAN((SQRT(3)*a - SQRT(-x^2 + 2*a*x + 3*a^2))/(2*a - x))",
    }


def test_a_solution_of_a_system_follows_the_order_of_the_variables():
    assert simp("SOLVE([x + y = 3, x - y = 1], [y, x])") == "[y = 1 AND x = 2]"
    assert simp("SOLVE([a*x + b*y = 3, a*y + c*z = 4, c*y + a*z = 0], [z, x, y])") == (
        "[z = -4*c/(a^2 - c^2) AND x = (3*a^2 - 4*a*b - 3*c^2)/(a^3 - a*c^2)"
        " AND y = 4*a/(a^2 - c^2)]"
    )


ONE_ARGUMENT = [
    # Without a variable, a relation is decided.
    ("SOLVE(3 = 3)", "true"),
    ('SOLVE("abc" = "abc")', "true"),
    ("SOLVE(97 > 1)", "true"),
    ("SOLVE(MOD(2^(10^100 + 8), 10^100 + 9) = 1)", "false"),
    # Otherwise it is solved for all its variables, in the variable order.
    ("SOLVE(x^2 = 1)", "x = -1 OR x = 1"),
    ("SOLVE(x + y = 3 AND x - y = 1)", "x = 2 AND y = 1"),
    ("VECTOR(SOLVE(x^2 - 8*x = -15), x, 1, 5)", "[false, false, true, false, true]"),
]


@pytest.mark.parametrize(("text", "expected"), ONE_ARGUMENT, ids=str)
def test_solve_of_one_argument_solves_for_every_variable(text, expected):
    assert simp(text, state=word_mode()) == expected


VECTORS = [
    # A vector of relations gives the vector of the disjuncts.
    ("SOLVE([x^2 - 1 = 0], x)", {"x = -1", "x = 1"}),
    ("SOLVE([x^2 - x - 6], x)", {"x = 3", "x = -2"}),
    ("SOLVE([x^2 + y^2 = 2*a^2, x - y = 0], [x, y])", {
        "x = a AND y = a", "x = -a AND y = -a"
    }),
    ("SOLVE([2*x + 3*y = 7, 4*x - 7*y = 1], [x, y])", {"x = 2 AND y = 1"}),
    (
        "SOLVE([[19*x + 18*y + 61 = 0], [10*x + 4*y - 50 = 0], [36*x - 49*y - 1131 = 0],"
        " [17*x - 2*y - 217 = 0], [38*x - 11*y - 583 = 0]], [x, y])",
        {"x = 11 AND y = -15"},
    ),
    (
        "SOLVE([a = 0, a/8 - SQRT(127)*b/8 = -2], [a, b], Real)",
        {"a = 0 AND b = 16*SQRT(127)/127"},
    ),
    ("SOLVE([a - 1, b - 2], [a, b])", {"a = 1 AND b = 2"}),
    # A vector of variables inside the vector, as a utility file passes its unknowns, is
    # solved for in its order rather than the order of the names.
    ("SOLVE([c = 1, b = 2, a = 3], [[c, b, a]])", {"c = 1 AND b = 2 AND a = 3"}),
    (
        "SOLVE([@8 = 1, @9 = 2, @10 = 3], [[@8, @9, @10]])",
        {"@8 = 1 AND @9 = 2 AND @10 = 3"},
    ),
    ("[SOLVE([w = 1], w), SOLVE([w = 2], w)]`", {"[w = 1, w = 2]"}),
]


@pytest.mark.parametrize(("text", "expected"), VECTORS, ids=named)
def test_a_vector_of_relations_solves_to_a_vector(text, expected):
    assert elements(text) == expected


def test_a_vector_without_solutions_is_empty():
    assert simp("SOLVE([x = x + 1], x)") == "[]"
    assert simp("SOLVE([x + y = 1, 2 = 3], [x, y])") == "[]"
    assert simp("SOLVE([x + 3*y = 1, 2*x + 6*y = 3], [x, y])") == "[]"
    assert simp("SOLVE([x = x], x)") == "[true]"


def test_a_variable_with_a_value_is_solved_for_as_a_variable():
    """Naming a variable with a value as the variable to solve for solves for the
    variable, which is then not replaced by its value in the equation."""
    state = words()
    context = Context(
        assignments={"d1": parse("(a1*e2 - a2*e1)/(e2 - a2)", state), "x": parse("3")}
    )
    text = "SOLVE(0 - a2 = (e2 - a2)/(e1 - a1)*(d1 - a1), d1)"
    assert simp(text, context, state) == "d1 = (a2*e1 - a1*e2)/(a2 - e2)"
    assert simp("SOLVE(x^2 = 9, x)", context, state) == "x = -3 OR x = 3"
    assert simp("SOLUTIONS(x^2 = 9, x)", context, state) == "[3, -3]"
    assert simp("x^2", context, state) == "9"


def test_a_name_holding_the_variables_names_them():
    state = words()
    context = Context(assignments={"vars1": parse("[x, y]", state)})
    assert simp("SOLVE([x + y = 3, x - y = 1], vars1)", context, state) == (
        "[x = 2 AND y = 1]"
    )


def test_one_equation_in_several_variables_is_solved_factor_by_factor():
    assert disjuncts("SOLVE(2*xo*(yo^2 - b^2), [xo, yo], Real)") == {
        "xo = 0",
        "yo = -b",
        "yo = b",
    }
    assert disjuncts("SOLVE(2*yo*(xo^2 - a^2), [xo, yo], Real)") == {
        "xo = -a",
        "xo = a",
        "yo = 0",
    }
    # A factor in two of the variables stays an equation.
    assert simp("SOLVE(x + y = 3, [x, y])") == "x + y = 3"


def test_a_chained_relation_is_the_conjunction_of_its_links():
    # The link without the solve variable is a condition every solution carries.
    text = "SOLVE(2*x*SIN(theta) + 2*y*COS(theta) = 3*SIN(2*theta) = 0, y)"
    assert simp(text) == "y = 3*SIN(theta) - x*TAN(theta) AND 3*SIN(2*theta) = 0"


def test_a_chain_an_operand_computed_is_the_conjunction_of_its_links():
    """`DIF` of an equation is the equation of the derivatives, so `DIF(v = h^2, h) = 0`
    is the chain `0 = 2*h = 0`."""
    assert simp("SOLVE(DIF(v = h^2, h) = 0, h)") == "h = 0"
    assert simp("SOLUTIONS(DIF(v = h^2, h) = 0, h)") == "[0]"
    assert simp("SOLVE(DIF(v = h^2 - 2*h, h) = 1, h)") == "false"


def test_an_equation_without_the_variables_is_a_condition_on_the_solutions():
    """Such an equation is neither dropped nor taken for "no solutions": the solutions
    hold where it does. One decided false leaves none."""
    assert simp("SOLVE([x + y = 1, a = 0], [x, y])") == "[x + y = 1 AND a = 0]"
    assert simp("SOLVE(x^2 = 4 AND a = b, x)") == "(x = -2 OR x = 2) AND a = b"
    assert simp("SOLVE(x = x AND a = b, x)") == "a = b"
    assert simp("SOLUTIONS(x = 2 AND a = b, x)") == "[IF(a = b, 2)]"


SHARED_CONDITIONS = [
    # Derive 4.11 states a condition every solution shares once, `(x = 2 OR x = -2) AND
    # a = b`; the solutions keep this engine's order.
    ("SOLVE(x^2 = 4 AND a > 0, x)", "(x = -2 OR x = 2) AND a > 0"),
    ("SOLVE(x^2 > 4 AND a = b, x)", "(x > 2 OR x < -2) AND a = b"),
    ("SOLVE(4 < x^2 AND x^2 < 9 AND a = b, x)", "(2 < x < 3 OR -3 < x < -2) AND a = b"),
    ("SOLVE(x^2 = 4 AND a = b AND c > 0, x)", "(x = -2 OR x = 2) AND a = b AND c > 0"),
    ("SOLVE((x - 1)*(x^5 + x + 3) = 0 AND a = b, x)", "(x^5 + x = -3 OR x = 1) AND a = b"),
    ("SOLVE(a*x^2 = a AND b = c, x)", "(a = 0 OR x = -1 OR x = 1) AND b = c"),
    (
        "SOLVE(x^2 + y^2 = 2 AND x = y AND a = b, [x, y])",
        "(x = -1 AND y = -1 OR x = 1 AND y = 1) AND a = b",
    ),
    # Solutions under the same conditions are one group, whichever conjunction of the
    # first argument they come from, in the order they have without the conditions,
    # `x = 3 OR x = -2 OR x = 2`.
    ("SOLVE((x^2 = 4 OR x = 3) AND a = b, x)", "(x = 3 OR x = -2 OR x = 2) AND a = b"),
    # A condition only some solutions share is stated for those.
    ("SOLVE(x^2 = 4 AND a = b OR x = 3, x)", "(x = -2 OR x = 2) AND a = b OR x = 3"),
    (
        "SOLVE(x^2 = 4 AND a = b OR x^2 = 9 AND c = d, x)",
        "(x = -3 OR x = 3) AND c = d OR (x = -2 OR x = 2) AND a = b",
    ),
    # A single solution keeps its conditions beside it, its equations in order.
    ("SOLVE(x = 2 AND a = b, x)", "x = 2 AND a = b"),
    ("SOLVE(x + y = 3 AND x - y = 1 AND a = b, [x, y])", "x = 2 AND y = 1 AND a = b"),
    ("SOLVE(x^2 /= 4 AND a = b, x)", "x /= -2 AND x /= 2 AND a = b"),
    # The answer is equivalent to the first argument, and reads back as it is written.
    ("SOLVE(SOLVE(x^2 = 4 AND a = b, x), x)", "(x = -2 OR x = 2) AND a = b"),
    ("(x = -2 OR x = 2) AND a = b", "(x = -2 OR x = 2) AND a = b"),
]


@pytest.mark.parametrize(("text", "expected"), SHARED_CONDITIONS, ids=str)
def test_a_condition_the_solutions_share_is_stated_once(text, expected):
    assert simp(text) == expected


def test_a_factored_answer_keeps_one_solution_per_element_and_listing():
    """The vector form has one element per solution and SOLUTIONS one value each, the
    condition with each."""
    assert elements("SOLVE([x^2 = 4, a = b], x)") == {
        "x = 2 AND a = b",
        "x = -2 AND a = b",
    }
    assert simp("SOLUTIONS(x^2 = 4 AND a = b, x)") == "[IF(a = b, 2), IF(a = b, -2)]"
    assert simp("SOLUTIONS(SOLVE(x^2 = 4 AND a = b, x), x)") == (
        "[IF(a = b, 2), IF(a = b, -2)]"
    )
    assert simp("SOLUTIONS(x^2 + y^2 = 2 AND x = y AND a = b, [x, y])") == (
        "[[IF(a = b, 1), IF(a = b, 1)], [IF(a = b, -1), IF(a = b, -1)]]"
    )
    # A free value under a condition is listed as it is without one.
    assert simp("SOLUTIONS(x + y = 1, [x, y])") == "[[@1, 1 - @1]]"
    assert simp("SOLUTIONS(x + y = 1 AND a = b, [x, y])") == (
        "[[IF(a = b, @1), IF(a = b, 1 - @1)]]"
    )


def test_the_sides_of_a_factored_answer_are_the_solutions_and_the_conditions():
    assert simp("LHS(SOLVE(x^2 = 4 AND a = b, x))") == "x = -2 OR x = 2"
    assert simp("RHS(SOLVE(x^2 = 4 AND a = b, x))") == "a = b"


def test_conditioned_angles_are_shown_in_the_angle_unit():
    assert simp("SOLVE(SIN(x) = 1/2 AND a = b, x)", DEGREES) == (
        "(x = 150 DEG OR x = 30 DEG) AND a = b"
    )


BOUNDED_CONDITIONS = [
    # A solution under a condition is inside or outside the bounds as it is without it.
    ("SOLVE(x^2 = 4 AND a = b, x, 1, 3)", "x = 2 AND a = b"),
    ("SOLVE(x^2 = 4 AND a = b, x, 3, 5)", "false"),
    ("SOLVE(x^2 = 4 AND a = b AND c > 0, x, -3, 0)", "x = -2 AND a = b AND c > 0"),
    ("SOLVE([x^2 = 4, a = b], x, 1, 3)", "[x = 2 AND a = b]"),
    ("SOLUTIONS(x^2 = 4 AND a = b, x, 1, 3)", "[IF(a = b, 2)]"),
    ("SOLUTIONS(x^2 = 4 AND a = b, x, 3, 5)", "[]"),
    # One not known to be inside holds where it is, besides the condition, and one
    # that is inside for no value of the parameters is no solution: `-SQRT(a)` is
    # never in `[1, 2]`.
    (
        "SOLVE(x^2 = a AND a > 0, x, 1, 2)",
        "x = SQRT(a) AND 1 <= SQRT(a) AND SQRT(a) <= 2 AND a > 0",
    ),
    (
        "SOLUTIONS(x^2 = a AND a > 0, x, 1, 2)",
        "[IF(a > 0 AND SQRT(a) >= 1 AND SQRT(a) <= 2, SQRT(a))]",
    ),
    # So does one under a condition the solver found.
    ("SOLVE(x = a AND x = b, x, 0, 1)", "x = a AND 0 <= a AND a <= 1 AND a - b = 0"),
]


@pytest.mark.parametrize(("text", "expected"), BOUNDED_CONDITIONS, ids=str)
def test_bounds_apply_to_solutions_under_conditions(text, expected):
    assert simp(text) == expected


BOUNDED_EVERYTHING = [
    # Without bounds every value is a solution; with them, every value in them is.
    ("SOLVE(x = x, x)", "true"),
    ("SOLVE(x = x, x, 1, 3)", "1 <= x <= 3"),
    ("SOLVE(x + 1 = 1 + x, x, 1, 3)", "1 <= x <= 3"),
    ("SOLVE(x/x = 1, x, 1, 3)", "1 <= x <= 3"),
    ("SOLVE(SIN(x)^2 + COS(x)^2 = 1, x, 0, 1)", "0 <= x <= 1"),
    ("SOLVE(x = x, x, SQRT(2), pi)", "SQRT(2) <= x <= pi"),
    ("SOLVE([x = x], x, 1, 3)", "[1 <= x <= 3]"),
    # An inequality that holds everywhere.
    ("SOLVE(x^2 + 1 > 0, x, 1, 3)", "1 <= x <= 3"),
    ("SOLVE(x^2 >= 0, x, 1, 3)", "1 <= x <= 3"),
    # Every other solution found is one of these.
    ("SOLVE(x^2 = 4 OR x = x, x, 1, 3)", "1 <= x <= 3"),
    # An infinite bound bounds nothing, and bounds that meet hold one value.
    ("SOLVE(x = x, x, -inf, 3)", "x <= 3"),
    ("SOLVE(x = x, x, -inf, inf)", "true"),
    ("SOLVE(x = x, x, 1, 1)", "x = 1"),
    ("SOLVE(x = x, x, 3, 1)", "false"),
    # Under a condition on the parameters, every value in the bounds is a solution
    # where the condition holds.
    ("SOLVE(x = x AND a = b, x)", "a = b"),
    ("SOLVE(x = x AND a = b, x, 1, 3)", "1 <= x AND x <= 3 AND a = b"),
    ("SOLVE([x = x, a = b], x, 1, 3)", "[1 <= x AND x <= 3 AND a = b]"),
    # A vector lists no interval of values, so `SOLUTIONS` stays as written, but
    # without bounds every value under a condition is an arbitrary one under it.
    ("SOLUTIONS(x = x, x)", "[@1]"),
    ("SOLUTIONS(x = x, x, 1, 3)", "SOLUTIONS(true, x, 1, 3)"),
    ("SOLUTIONS(x = x, x, 1, 1)", "[1]"),
    ("SOLUTIONS(x = x AND a = b, x)", "[IF(a = b, @1)]"),
    ("SOLUTIONS(x = x AND a = b, x, 1, 3)", "SOLUTIONS(a = b, x, 1, 3)"),
    # A condition in none of the variables is solved by every value where it holds.
    ("SOLVE(a = b, x, 1, 3)", "1 <= x AND x <= 3 AND a - b = 0"),
    ("SOLVE(a = b OR x = 2, x, 1, 3)", "1 <= x AND x <= 3 AND a - b = 0 OR x = 2"),
]


@pytest.mark.parametrize(("text", "expected"), BOUNDED_EVERYTHING, ids=str)
def test_every_value_in_the_bounds_solves_an_identity(text, expected):
    assert simp(text) == expected


#: Solutions under conditions on the parameters, in answers a second Simplify keeps. A
#: condition in none of the variables is solved by every value of them where it holds,
#: and a solution under conditions that hold for no value of the parameters is none:
#: `-SQRT(a)` is never in `[1, 2]`.
SETTLED_CONDITIONS = [
    ("SOLVE(a = b, x)", "a - b = 0"),
    ("SOLUTIONS(a = b, x)", "[IF(a - b = 0, @1)]"),
    ("SOLUTIONS(a > b, x)", "[IF(a - b > 0, @1)]"),
    ("SOLUTIONS(a = b, [x, y])", "[[IF(a - b = 0, @1), IF(a - b = 0, @2)]]"),
    ("SOLUTIONS(a = b OR x = 2, x)", "[2, IF(a - b = 0, @1)]"),
    ("SOLUTIONS(a = b, x, 1, 1)", "[IF(a - b = 0, 1)]"),
    ("SOLUTIONS(1 = 2, x, 1, 3)", "[]"),
    # Every value in the bounds is a solution there, which no vector lists.
    ("SOLUTIONS(a = b, x, 1, 3)", "SOLUTIONS(a = b, x, 1, 3)"),
    ("SOLUTIONS(a = b OR x = 2, x, 1, 3)", "SOLUTIONS(x = 2 OR a = b, x, 1, 3)"),
    ("SOLUTIONS(x^2 = a, x, 1, 2)", "[IF(SQRT(a) >= 1 AND SQRT(a) <= 2, SQRT(a))]"),
    (
        "SOLUTIONS(x^2 = a, x, -2, -1)",
        "[IF(SQRT(a) >= 1 AND SQRT(a) <= 2, -SQRT(a))]",
    ),
    (
        "SOLUTIONS(x^2 = a AND a > 0, x, 1, 2)",
        "[IF(a > 0 AND SQRT(a) >= 1 AND SQRT(a) <= 2, SQRT(a))]",
    ),
    ("SOLUTIONS(x^2 = a AND a < 0, x, 1, 2)", "[]"),
    ("SOLUTIONS(x^2 = 4 AND SQRT(a) <= -1 AND SQRT(a) >= -5, x)", "[]"),
]


@pytest.mark.parametrize(("text", "expected"), SETTLED_CONDITIONS, ids=str)
def test_solutions_under_conditions_are_settled_at_once(text, expected):
    once = simp(text)
    assert once == expected
    assert simp(once) == once


BOUNDED_RANGES = [
    ("SOLVE(x^2 > 4, x, 0, 5)", "2 < x <= 5"),
    ("SOLVE(x^2 > 4, x, -5, 5)", "2 < x <= 5 OR -5 <= x < -2"),
    ("SOLVE(x^2 > 4, x, -1, 1)", "false"),
    ("SOLVE(x^2 > 4, x, 0, inf)", "x > 2"),
    ("SOLVE(x > 0, x, 1, 3)", "1 <= x <= 3"),
    ("SOLVE(x^2 > 2, x, 0, 5)", "SQRT(2) < x <= 5"),
    ("SOLVE(x^5 + x > 3, x, 0, 2)", "ROOT_OF(x^5 + x - 3, x, 0) < x <= 2"),
    ("SOLVE(x /= 2, x, 0, 5)", "2 < x <= 5 OR 0 <= x < 2"),
    ("SOLVE([x^2 > 4], x, 0, 5)", "[2 < x <= 5]"),
    # A range cut to a single value is that solution.
    ("SOLVE(x^2 >= 4, x, 0, 2)", "x = 2"),
    ("SOLUTIONS(x^2 >= 4, x, 0, 2)", "[2]"),
    ("SOLVE(x^2 > 4 AND a = b, x, 0, 5)", "2 < x AND x <= 5 AND a = b"),
    # A range with an end at a parameter holds where it is inside the bounds.
    ("SOLVE(x > a, x, 0, 5)", "x > a AND 0 <= x AND x <= 5"),
]


@pytest.mark.parametrize(("text", "expected"), BOUNDED_RANGES, ids=str)
def test_bounds_cut_a_range_of_solutions(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize("context", [EXACT, APPROXIMATE], ids=["Exact", "Approximate"])
def test_solve_command_bounds_an_identity_and_a_range(context):
    def bounded(text, bounds):
        return sol(text, context=context, bounds=bounds)

    assert sol("x = x", context=context) == ["x = @1"]
    assert bounded("x = x", ("1", "3")) == ["1 <= x <= 3"]
    assert bounded("x + 1 = 1 + x", ("1", "3")) == ["1 <= x <= 3"]
    assert bounded("x/x = 1", ("1", "3")) == ["1 <= x <= 3"]
    assert bounded("x^2 + 1 > 0", ("1", "3")) == ["1 <= x <= 3"]
    assert bounded("x = x", ("-inf", "inf")) == ["x = @1"]
    assert bounded("x^2 > 4", ("0", "5")) == ["2 < x <= 5"]
    assert bounded("x^2 > 4", ("-1", "1")) == []
    assert bounded("x^2 >= 4", ("0", "2")) == ["x = 2"]
    assert bounded("x /= 2", ("0", "5")) == ["0 <= x < 2", "2 < x <= 5"]
    assert bounded("x > a", ("0", "5")) == ["x > a AND 0 <= x AND x <= 5"]


def test_bounds_search_an_unsolved_equation_under_conditions():
    assert simp("SOLVE(x^5 + x + 3 = 0, x, -2, 0)", APPROXIMATE) == "x = -1.132997565"
    assert simp("SOLVE(x^5 + x + 3 = 0 AND a = b, x, -2, 0)", APPROXIMATE) == (
        "x = -1.132997565 AND a = b"
    )


SOLVED_UNDER_CONDITIONS = [
    # soLve writes each solution on its own line, with the conditions as given, as
    # SOLVE does.
    ("x^2 = 4 AND a = b", ("x",), ["x = -2 AND a = b", "x = 2 AND a = b"]),
    ("x^2 = 4 AND a > 0", ("x",), ["x = -2 AND a > 0", "x = 2 AND a > 0"]),
    ("x^2 = a AND a > 0", ("x",), ["x = -SQRT(a) AND a > 0", "x = SQRT(a) AND a > 0"]),
    ("x^2 > 4 AND a > 0", ("x",), ["x > 2 AND a > 0", "x < -2 AND a > 0"]),
    (
        "x^2 = 4 AND a = b AND c > 0",
        ("x",),
        ["x = -2 AND a = b AND c > 0", "x = 2 AND a = b AND c > 0"],
    ),
    ("x > 1 AND a = b AND c > 0", ("x",), ["x > 1 AND a = b AND c > 0"]),
    ("x + y = 3 AND x - y = 1 AND a > 0", ("x", "y"), ["x = 2 AND y = 1 AND a > 0"]),
    # A condition decided is dropped, or leaves no solution.
    ("x^2 = 4 AND 2 < 3", ("x",), ["x = -2", "x = 2"]),
    ("x^2 = 4 AND 2 < 1", ("x",), []),
    # A condition the solver finds is written as an unsolved equation.
    ("x = a AND x = b", ("x",), ["x = a AND a - b = 0"]),
]


@pytest.mark.parametrize(("text", "variables", "expected"), SOLVED_UNDER_CONDITIONS)
def test_solve_command_solves_under_conditions(text, variables, expected):
    assert sol(text, variables) == expected


def test_solve_command_bounds_the_equation_beside_conditions():
    bounds = ("1", "3")
    assert sol("x^2 = 4", ("x",), APPROXIMATE, bounds) == ["x = 2"]
    assert sol("x^2 = 4 AND a = b", ("x",), APPROXIMATE, bounds) == ["x = 2 AND a = b"]
    assert sol("x^2 = 4 AND a > 0", ("x",), APPROXIMATE, bounds) == ["x = 2 AND a > 0"]


# A call matching none of SOLVE's forms stays as written; ignoring a misspelled domain
# would answer a different question.
STANDING = [
    # A third argument that isn't a domain.
    "SOLVE(x^2 = 1, x, reals)",
    "SOLVE(x^2 = 1, x, imaginary)",
    "SOLVE(x^2 = 1, x, 5)",
    "SOLVE(x^2 = -1, x, y)",
    # More than four arguments.
    "SOLVE(x^2 = 1, x, -2, 2, 3)",
    # A domain word where the variable belongs.
    "SOLVE(x^2 = 1, Real)",
    "SOLVE(x^2 = 1, Complex)",
    # Something other than a variable to solve for.
    "SOLVE(x^2 = 4, 3)",
    "SOLVE(x^2 = 4, [x, x])",
    "SOLVE(x^2 = 4, [])",
    # Bounds that aren't numbers.
    "SOLVE(x^2 = 4, x, a, 10)",
    # A set of equations has no order to solve them in.
    "SOLVE({x = 1, y = 2}, [x, y])",
]


def word_mode():
    state = ParseState()
    state.input_mode = InputMode.WORD
    return state


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
        ("DIMENSION(SOLVE(x^2 = 1, x, Reals))", "DIM(SOLVE(x^2 = 1, x, reals))"),
        ("RHS(SOLVE(x^2 = 1, x, Reals))", "SOLVE(x^2 = 1, x, reals)"),
        ("(SOLVE(x^2 = 1, x, Reals)) SUB 1", "(SOLVE(x^2 = 1, x, reals)) SUB 1"),
    ],
    ids=str,
)
def test_a_standing_solve_head_is_taken_as_the_call_it_is(text, expected):
    """As with an unevaluated `RATE`, `RHS` of a non-relation is the thing itself."""
    assert simplify(parse(text, word_mode()), EXACT).text == expected


def test_an_assigned_third_argument_is_read_for_its_value():
    """The head sees the assigned value: a name assigned a number is not a domain, but
    one assigned a domain word is that domain."""
    state = ParseState()
    number = Context(assignments={"y": parse("3", state)})
    assert simplify(parse("SOLVE(x^2 = -1, x, y)", state), number).text == (
        "SOLVE(x^2 = -1, x, 3)"
    )
    word = Context(assignments={"y": parse("Real", state)})
    assert simplify(parse("SOLVE(x^2 = -1, x, y)", state), word).text == "false"


# -- the sides of an answer --------------------------------------------------------


SIDES = [
    # The operands count in the order they are written out: `x = 2 + #i OR x = 2 - #i`.
    ("RHS(x = 2 - #i OR x = 2 + #i)", "x = 2 - #i"),
    ("RHS(RHS(x = 2 - #i OR x = 2 + #i))", "2 - #i"),
    ("LHS(x = 2 - #i OR x = 2 + #i)", "x = 2 + #i"),
    ("RHS(x = 1 OR x = 2 OR x = 3)", "x = 2 OR x = 3"),
    ("RHS(p = 1/6 AND q = 3/4 AND s = 4)", "q = 3/4 AND s = 4"),
    ("RHS(q = 3/4 AND s = 4)", "s = 4"),
    # One solution is an equation, as in Derive 3 and 4.
    ("RHS(SOLVE(1680 = 48*q, q))", "35"),
    ("RHS(SOLVE(p = 1/6 AND q = 3/4 AND s = 4, [p, q, s]))", "q = 3/4 AND s = 4"),
    # A vector of solutions distributes.
    ("RHS(SOLVE([x^2 - 5*x + 6 = 0], x))", "[2, 3]"),
    # A disjunction has no elements to select, and the selection that stands may be an
    # equation, so its sides stand too.
    ("RHS((SOLVE(x^2 = 4, x)) SUB 1)", "RHS((x = -2 OR x = 2) SUB 1)"),
    ("LHS(ELEMENT(x = 2 OR x = -2, 1))", "LHS(ELEMENT(x = -2 OR x = 2, 1))"),
]


@pytest.mark.parametrize(("text", "expected"), SIDES, ids=str)
def test_the_sides_of_a_logical_answer_are_its_first_operand_and_the_rest(
    text, expected
):
    assert simp(text) == expected


def test_a_logical_answer_is_not_a_vector_of_solutions():
    """Derive 6 broke the Derive 3 idioms: SOLUTIONS replaces them."""
    assert simp("DIMENSION(SOLVE(x^2 - 4, x))") == "DIM(x = -2 OR x = 2)"
    assert simp("DIMENSION(SOLUTIONS(x^2 - 4, x))") == "2"
    assert simp("DIMENSION(SOLUTIONS(x = x + 1, x))") == "0"
    assert simp("(SOLUTIONS(x^2 - 4, x)) SUB 1") == "2"


def test_a_conjunction_indexes_its_conjuncts_and_a_disjunction_does_not():
    """A system's solution is a conjunction of one equation per variable, in the order
    of the variables, and `SUB` picks one; the solutions of a disjunction are not
    indexed (`SOLUTIONS` lists them)."""
    assert simp("(x = 6 AND y = 9/2) SUB 1") == "x = 6"
    assert simp("(SOLVE([z + x = 3, z - x = 1], [z, x])) SUB 1 SUB 2") == "x = 1"
    assert simp("LHS((SOLVE([z + x = 3, z - x = 1], [z, x])) SUB 1)") == "z = 2"
    assert simp("(x = 6 AND y = 9/2) SUB 3") == "(x = 6 AND y = 9/2) SUB 3"
    assert simp("(x = 1 OR x = 2) SUB 1") == "(x = 1 OR x = 2) SUB 1"


# -- conditions on parameters ----------------------------------------------------


CONDITIONS = [
    # A factor free of the variable adds the condition under which every value solves
    # the equation.
    (
        "SOLVE(Ieff^2*R*T = Imax^2*R*T/2, Ieff)",
        {"Ieff = -SQRT(2)*Imax/2", "Ieff = SQRT(2)*Imax/2", "R = 0", "T = 0"},
    ),
    ("SOLVE((U0/(RI + RL))^2*RL = U0^2/(4*RI), RL)", {"RL = RI", "U0 = 0"}),
    ("SOLVE(-t*(c*(t + 5) + 10) = t^2 - 5*t, c)", {"c = -1", "t = -5", "t = 0"}),
    ("SOLVE(r*t*(3*b - v)/(3*(v - b)^3), v)", {"v = 3*b", "r = 0", "t = 0"}),
    # Nothing else makes a condition: not a coefficient, nor a denominator.
    ("SOLVE(a*x + b = 0, x)", {"x = -b/a"}),
    ("SOLVE(2*P/Ub = Ub/(2*Rc), Rc)", {"Rc = Ub^2/(4*P)"}),
    ("SOLVE(U0^2 - 4*PLmax*RI = 0, PLmax)", {"PLmax = U0^2/(4*RI)"}),
    ("SOLVE((x^4 - x^2 + y - 1)/x^4 = 0, y)", {"y = -x^4 + x^2 + 1"}),
    ("SOLVE((z + 10*t)/(t*(t + 5)) = c, z)", {"z = t*(c*(t + 5) - 10)"}),
    ("SOLVE(r^2*SIN(phi)^2 = r^3*COS(phi)^3/(a - r*COS(phi)), r)", {
        "r = a*SIN(phi)^2/COS(phi)", "r = 0"
    }),
    # Without the variable, the relation is the answer, or its truth value.
    ("SOLVE(a = 0, x)", {"a = 0"}),
    ("SOLVE(5, x)", {"false"}),
    ("SOLVE(SUBST(x^2*y - x - y + 1, x, 1), y)", {"true"}),
    ("SOLVE(SUBST(x^2*y - x - y + 1, x, 0), y)", {"y = 1"}),
    ("SOLVE(SUBST(x*(y^2 - 1) - y + 1, x, 1), y)", {"y = 1", "y = 0"}),
    # Undefined points and infinity are no solutions.
    ("SOLVE(x/(2*x - 8) + (x - 6)/(x - 4) = 3, x)", {"false"}),
    ("SOLVE(x/(2*x - 8) + (x - 6)/(x - 4) = 3/2, x)", {"true"}),
    ("SOLVE(x + 1 + 1/(x - 2) = 3 + 1/(x - 2), x)", {"false"}),
    ("SOLVE((x - 1)/(x^2 - 1) = 0, x)", {"false"}),
    ("SOLVE(1/x = 0, x)", {"false"}),
    (
        "SOLVE(DIF(1000*(x - 1)/((101*x - 100)*(100*x - 99)), x), x)",
        {"x = 1 - SQRT(101)/1010", "x = SQRT(101)/1010 + 1"},
    ),
]


@pytest.mark.parametrize(("text", "expected"), CONDITIONS, ids=named)
def test_a_factor_without_the_variable_is_a_condition_on_the_parameters(
    text, expected
):
    assert disjuncts(text) == expected


def test_a_condition_that_cannot_hold_is_dropped():
    state = ParseState()
    name, domain = domain_of_node(parse("a :epsilon Real (0, inf)", state))
    context = Context(domains={name: domain})
    assert simplify(parse("SOLVE(a*x^2 = a, x)", state), context).text == (
        "x = -1 OR x = 1"
    )
    assert disjuncts("SOLVE(a*x^2 = a, x)") == {"x = -1", "x = 1", "a = 0"}
    # A factorial has no zeros, as a factor of the parameters or of the variable.
    assert simp("SOLVE(n!*(k - n*t) = 0, k)") == "k = n*t"
    assert simp("SOLVE(n!/(k!*(n - k)!)*(k - n*t) = 0, k)") == "k = n*t"


def test_a_power_to_a_symbolic_exponent_stays_an_equation():
    """Whether `t^(k - 1)` vanishes depends on its exponent, so it stays as an equation
    beside the other factor's solution, solved for the base's variable or for the
    exponent's."""
    bernstein = "t^(k - 1)*(1 - t)^(-k + n - 1)*(k - n*t)*n!/(k!*(n - k)!)"
    assert simp(f"SOLVE(0 = {bernstein}, t)") == (
        "(1 - t)^(-k + n - 1) = 0 OR t^(k - 1) = 0 OR t = k/n"
    )
    assert simp(f"SOLVE(0 = {bernstein}, k)") == (
        "(1 - t)^(-k + n - 1) = 0 OR t^(k - 1) = 0 OR k = n*t"
    )


def test_solutions_shows_no_conditions():
    assert elements("SOLUTIONS(Ieff^2*R*T = Imax^2*R*T/2, Ieff)") == {
        "-SQRT(2)*Imax/2",
        "SQRT(2)*Imax/2",
    }


# -- equations left unsolved ------------------------------------------------------


UNSOLVED = [
    # The number term goes to the right, with integer coefficients and the first term
    # positive.
    ("SOLVE(x^5 + x + 3 = 0, x)", {"x^5 + x = -3"}),
    (
        "SOLVE(x^5 + 2*x^4 - 3*x^3 + 4*x^2 - 5*x + 6 = 0, x)",
        {"x^5 + 2*x^4 - 3*x^3 + 4*x^2 - 5*x = -6"},
    ),
    ("SOLVE((x - 1)*(x^5 + x + 3) = 0, x)", {"x^5 + x = -3", "x = 1"}),
    ("SOLVE(SIN(x) = 1 - x/6, x)", {"6*SIN(x) + x = 6"}),
    ("SOLVE(1.5*x*COS(x) - SIN(x)/2 = 0, x)", {"3*x*COS(x) - SIN(x) = 0"}),
    ("SOLVE(SIN(x^2) + 2*COS(x) = 1, x, Real)", {"2*COS(x) + SIN(x^2) = 1"}),
    ("SOLVE(TAN(x) = 2*x, x)", {"TAN(x) - 2*x = 0"}),
    (
        "SOLVE(80*(7*p^(17/10) - 225000*p^(2/5) - 2625000)/p^2, p)",
        {"7*p^(17/10) - 225000*p^(2/5) = 2625000"},
    ),
    ("SOLVE(FLOOR(x^2) = 3, x)", {"FLOOR(x^2) = 3"}),
]


@pytest.mark.parametrize(("text", "expected"), UNSOLVED, ids=named)
def test_an_unsolved_equation_is_returned_rearranged(text, expected):
    assert disjuncts(text) == expected


def test_solutions_lists_the_solutions_found_and_drops_what_is_unsolved():
    """`[]` says that no solution was found; programs loop until they get it."""
    assert simp("SOLUTIONS(x^5 + x + 3 = 0, x)") == "[]"
    assert simp("SOLUTIONS(FLOOR(x^2) = 3, x)") == "[]"
    assert simp("SOLUTIONS((x - 1)*(x^5 + x + 3) = 0, x)") == "[1]"


def test_a_linear_congruence_has_derive_s_three_representatives():
    """`MOD(a*x + b, m) = r` holds where `a*x + b` is `r`, `r + m` or `r - m`, with `b`
    reduced modulo `m` first: the values Derive lists (a Caesar cipher worksheet
    decodes with them). A remainder outside `[0, m)` has none."""
    assert simp("SOLUTIONS(MOD(8*77 + z, 31) + 65 = 65, z)") == "[4, -27, -58]"
    assert simp("SOLUTIONS(MOD(65*33 + z, 97) + 32 = 65, z)") == "[22, -75, 119]"
    assert simp("SOLUTIONS(MOD(81*54 + z, 101) + 32 = 65, z)") == "[2, -99, 103]"
    assert simp("SOLUTIONS(MOD(29*83 + z, 101) + 32 = 65, z)") == "[50, -51, -152]"
    assert simp("SOLVE(MOD(2*x + 3, 5) = 4, x)") == "x = 1/2 OR x = 3 OR x = -2"
    assert simp("SOLVE(MOD(x, 5) = 7, x)") == "false"


ORDERED_SOLUTIONS = [
    # Rational solutions first, by denominator, then size, the positive first.
    ("SOLUTIONS(3*x^2 + 44*x + 96, x)", "[-12, -8/3]"),
    ("SOLUTIONS(250*x^3 - 1575*x^2 - 14700*x + 10976, x)", "[-28/5, 56/5, 7/10]"),
    ("SOLUTIONS(x^2 - 1, x)", "[1, -1]"),
    # The two signs of a root together, the one adding it first.
    ("SOLUTIONS(x^2 + 2*x - 4 = 0, x)", "[SQRT(5) - 1, -SQRT(5) - 1]"),
    ("SOLUTIONS(x^2 - 4*x + 5, x)", "[2 + #i, 2 - #i]"),
    ("SOLUTIONS(x^4 + 4*x^2 + 3 = 0, x)", "[#i, -#i, SQRT(3)*#i, -SQRT(3)*#i]"),
    # Opposite roots by the square-free part of their square.
    ("SOLUTIONS(x^4 - 55*x^2 + 450, x)", "[3*SQRT(5), -3*SQRT(5), SQRT(10), -SQRT(10)]"),
    ("SOLUTIONS(x^4 - 33*x^2 + 90, x)", "[SQRT(3), -SQRT(3), SQRT(30), -SQRT(30)]"),
    # The three real roots of a cubic as Derive's trigonometric formula lists them: here
    # the middle, the smallest, the largest one.
    (
        "SOLUTIONS(2*z^3 - z^2 - z + 1/4 = 0, z)",
        "[1/6 - SQRT(7)*SIN(pi/6 - ACOS(-SQRT(7)/14)/3)/3, "
        "1/6 - SQRT(7)*COS(pi/3 - ACOS(-SQRT(7)/14)/3)/3, "
        "SQRT(7)*COS(ACOS(-SQRT(7)/14)/3)/3 + 1/6]",
    ),
    # SOLVE lists them back to front.
    ("SOLVE(x^2 - 4*x + 5, x)", "x = 2 - #i OR x = 2 + #i"),
    (
        "SOLVE(x + 2*x = c + x*x, x)",
        "x = (3 - SQRT(9 - 4*c))/2 OR x = (SQRT(9 - 4*c) + 3)/2",
    ),
    ("RHS(SOLVE(x^2 - 4*x + 5, x))", "x = 2 + #i"),
    ("RHS(RHS(SOLVE(x^2 - 4*x + 5, x)))", "2 + #i"),
    # A system's solutions in the order of their values, variable by variable.
    (
        "SOLUTIONS((x - m)^2 + y^2 = (r + d)^2 AND x = d, [x, y])",
        "[[d, SQRT((2*d - m + r)*(m + r))], [d, -SQRT((2*d - m + r)*(m + r))]]",
    ),
]


@pytest.mark.parametrize(("text", "expected"), ORDERED_SOLUTIONS, ids=named)
def test_solutions_come_in_derive_s_order(text, expected):
    """Programs take the k-th solution (`(SOLUTIONS(...)) SUB 2`, `RHS(SOLVE(...))`),
    so the order is part of the answer: the order of Derive 6's recorded results."""
    assert simp(text) == expected


# -- inequalities ---------------------------------------------------------------


INEQUALITY_ANSWERS = [
    ("SOLVE(ABS(ABS(2*x + 1) - 5) > 2, x)", {"x < -4", "-2 < x < 1", "x > 3"}),
    (
        "SOLVE(-2*v^2 + 2*v + 1 >= 0, v)",
        {"1/2 - SQRT(3)/2 <= v <= SQRT(3)/2 + 1/2"},
    ),
    ("SOLVE(2*SQRT(2)*v + 1 >= 0, v)", {"v >= -SQRT(2)/4"}),
    ("SOLVE(-10/3 <= -30/k <= -3/3, k)", {"9 <= k <= 30"}),
    ("SOLVE(-10/3 <= -30/k <= -3/2, k)", {"9 <= k <= 20"}),
    ("SOLVE(3*x + 5 > 29, x)", {"x > 8"}),
    ("SOLVE(2*x + 1 < 5 OR 2*x + 3 > 8, x)", {"x < 2", "x > 5/2"}),
    ("SOLVE(-2*x + 3*y <= 7, x)", {"x >= (3*y - 7)/2"}),
    ("SOLVE(x > 1 AND x < 3, x)", {"1 < x < 3"}),
    ("SOLVE(x^2 <= 0, x)", {"x = 0"}),
    ("SOLVE(x^2 < 0, x)", {"false"}),
    ("SOLVE(x^2 >= 0, x)", {"true"}),
    ("SOLVE(x^2 /= 4, x)", {"x /= -2 AND x /= 2"}),
    ("SOLVE(x^2 < 4, x)", {"-2 < x < 2"}),
]


@pytest.mark.parametrize(("text", "expected"), INEQUALITY_ANSWERS, ids=named)
def test_an_inequality_solves_to_intervals(text, expected):
    assert disjuncts(text) == expected


def test_solutions_of_an_inequality_lists_points_only():
    assert simp("SOLUTIONS(x > 8, x)") == "SOLUTIONS(x > 8, x)"
    assert simp("SOLUTIONS(x^2 <= 0, x)") == "[0]"
    assert simp("SOLUTIONS(ABS(x - 2) >= 0, x)") == "[@1]"


# -- domains and bounds -----------------------------------------------------------
#
# `SOLVE(u, x, Real)`, Derive 6's third argument, keeps solutions without an imaginary
# component. Only solutions known to have one are dropped.

REAL = [
    ("SOLVE(x^3 + x^2 + x + 1, x, Real)", {"x = -1"}),
    (
        "SOLVE(x^3 + 3*x + 1 = 0, x, Real)",
        {"x = (4*SQRT(5) - 4)^(1/3)/2 - (4*SQRT(5) + 4)^(1/3)/2"},
    ),
    ("SOLVE(x^2 - 4, x, Real)", {"x = -2", "x = 2"}),
    ("SOLVE(x^2 + 1 = 0, x, Real)", {"false"}),
    # A family of complex values is dropped; one with a real representative stays.
    ("SOLVE(EXP(x) = -2, x, Real)", {"false"}),
    ("SOLVE(EXP(x) = 2, x, Real)", {"x = LN(2)"}),
    # A solution with a parameter may be real, so it stays.
    ("SOLVE(x^2 = a, x, Real)", {"x = -SQRT(a)", "x = SQRT(a)"}),
    # Answers that aren't values are not filtered: an identity, a range, and an
    # unsolved equation.
    ("SOLVE(x = x, x, Real)", {"true"}),
    ("SOLVE(x^2 < 4, x, Real)", {"-2 < x < 2"}),
    ("SOLVE(COS(x) = x, x, Real)", {"COS(x) - x = 0"}),
    # A system's solution is filtered as a whole.
    (
        "SOLVE([x^2 + y^2 = 2, x - y = 0], [x, y], Real)",
        {"[x = 1 AND y = 1, x = -1 AND y = -1]"},
    ),
    ("SOLVE([x + y = #i, x - y = 0], [x, y], Real)", {"[]"}),
    # Complex is the default domain and filters nothing.
    ("SOLVE(x^2 + 1 = 0, x, Complex)", {"x = #i", "x = -#i"}),
    ("SOLVE(x^3 - 2*x^2 + x - 2, x, Real)", {"x = 2"}),
    ("SOLVE(x^4 - 4 = 0, x, Real)", {"x = -SQRT(2)", "x = SQRT(2)"}),
]


@pytest.mark.parametrize(("text", "expected"), REAL, ids=named)
def test_real_asks_for_the_solutions_without_an_imaginary_component(text, expected):
    assert disjuncts(text) == expected


def test_a_real_root_written_through_the_imaginary_unit_is_kept():
    """Ferrari's radicals write this quartic's four real roots with `#i` terms that
    cancel, which doesn't make them imaginary."""
    text = "SOLUTIONS(x^4 - 4*x^2 + x + 1 = 0, x, Real)"
    answer = simplify(parse(text), EXACT)
    assert "#i" in answer.text
    assert simplify(parse(f"DIMENSION({text})"), EXACT).text == "4"


def test_real_filters_complex_roots_however_they_are_written():
    assert simp("SOLVE(x^4 + x + 1 = 0, x, Real)") == "false"


def test_real_is_answered_in_every_precision_mode():
    for context in (APPROXIMATE, MIXED):
        assert simp("SOLVE(x^3 - x + 5 = 0, x, Real)", context) == "x = -1.904160859"


def test_real_does_not_make_a_declared_domain_filter():
    state = ParseState()
    name, domain = domain_of_node(parse("x :epsilon Real", state))
    context = Context(domains={name: domain})
    assert set(simplify(parse("SOLVE(x^2 + 1, x)"), context).text.split(" OR ")) == {
        "x = #i",
        "x = -#i",
    }
    positive = Context(domains=dict([domain_of_node(parse("x :epsilon Real (0, inf)"))]))
    assert set(simplify(parse("SOLVE(x^2 = 4, x)"), positive).text.split(" OR ")) == {
        "x = -2",
        "x = 2",
    }


def test_bounds_keep_the_real_solutions_between_them():
    assert simp("SOLVE(x^2 = 4, x, 0, 10)") == "x = 2"
    assert simp("SOLUTIONS(x^2 = 4, x, 0, 10)") == "[2]"
    assert simp("SOLVE(x^2 = 4, x, 3, 10)") == "false"
    assert simp("SOLVE((x - 1)^2 = 0, x, 0, 3)") == "x = 1"


def test_bounds_search_numerically_only_where_numbers_are_asked_for():
    """In Exact precision an equation without exact solutions stays unsolved; in
    Approximate precision it is searched between the bounds."""
    assert simp("SOLVE(x^5 - x + 1 = 0, x, -10, 10)") == "x^5 - x = -1"
    assert simp("SOLVE(x^5 - x + 1 = 0, x, -10, 10)", APPROXIMATE) == "x = -1.167303978"


# -- linear systems -------------------------------------------------------------


LINEAR = [
    ("SOLVE(5*x + 4*y = 60 AND x + 2*y = 18, [x, y])", "x = 8 AND y = 5"),
    ("SOLVE(4*x + y = 8 AND 8*x + 13*y = 48, [x, y])", "x = 14/11 AND y = 32/11"),
    ("SOLVE(0.3*x + 0.25*y = 10 AND 0.4*x + 0.5*y = 16, [x, y])", "x = 20 AND y = 16"),
    (
        "SOLVE(a*x + b*y + (a + b) = 0 AND c*x + d*y + (c + d) = 0, [x, y])",
        "x = -1 AND y = -1",
    ),
    (
        "SOLVE(p*x + q*y - (a*p + b*q) = 0 AND r*x + s*y - (a*r + b*s) = 0, [x, y])",
        "x = a AND y = b",
    ),
    (
        "SOLVE(x + 16*y + 3*z = 6 AND 9*x - 9*y + 2*z = 4 AND 21*x + 12*y - 35*z = -2,"
        " [x, y, z])",
        "x = 2006/3447 AND y = 850/3447 AND z = 188/383",
    ),
    ("SOLVE([-3*x + 2*y = 10, 2*x - 3*y = 5], [x, y])", "[x = -8 AND y = -7]"),
    (
        "SOLVE([12*a - 3*b - c - 19*d = 2, -3*a - 19*c = 0, -19*a + 3*c = 1,"
        " -a - 19*b - 3*(4*c - d) = 0], [a, b, c, d])",
        "[a = -19/370 AND b = -809/34225 AND c = 3/370 AND d = -9199/68450]",
    ),
    # Underdetermined: the reduced system, pivots as far left as they go.
    ("SOLVE(2*x + 6*y = 2 AND 3*x = 3 - 9*y, [x, y])", "x + 3*y = 1"),
    ("SOLVE(x + 3*y = 1 AND 2*x + 6*y = 3, [x, y])", "false"),
    (
        "SOLVE([x - y + 3*z = 10, 2*x + 4*y + 3*z = 2], [x, y, z])",
        "[x + 5*z/2 = 7 AND y - z/2 = -3]",
    ),
    (
        "SOLVE([x - y + 3*z = 10, 2*x + 4*y + 3*z = 2], [x, y])",
        "[x = 7 - 5*z/2 AND y = z/2 - 3]",
    ),
    (
        "SOLVE([2*x + 6*y - 3*z = -6, 4*x + 3*y + 3*z = 6, 4*x - 3*y + 9*z = k],"
        " [x, y, z])",
        "[x + 3*z/2 = 3 AND y - z = -2 AND k = 18]",
    ),
    ("SOLVE([q - p, p - q], [p, q])", "[p - q = 0]"),
    # Inconsistent for most values of the parameters: the system holds where they meet
    # the condition, and in an identity the elimination doesn't know, everywhere.
    ("SOLVE([x + y = 1, 2*x + 2*y = k], [x, y])", "[x + y = 1 AND k = 2]"),
    ("SOLVE([x = a, x = b], x)", "[x = a AND a - b = 0]"),
    ("SOLVE(a = COS(t)^2 + 1 AND a = 2 - SIN(t)^2, a)", "a = 2 - SIN(t)^2"),
    ("SOLUTIONS(a = COS(t)^2 + 1 AND a = 2 - SIN(t)^2, a)", "[2 - SIN(t)^2]"),
    ("SOLVE([a = COS(t)^2, a = 2 - SIN(t)^2], a)", "[]"),
    # No equation in the variables: the conditions alone, unless they contradict.
    ("SOLVE([u = a, u = b], [p, q])", "[a - u = 0 AND b - u = 0]"),
    ("SOLVE([u = 1, u = 2], [p, q])", "[]"),
    ("SOLVE([x = k, x = 1 + k, k = m], x)", "[]"),
    (
        "SOLVE([6*a/(5*COS(t)^2 + 4)^(3/2) + 1, 6*a/(9 - 5*SIN(t)^2)^(3/2) + 1] = [0, 0],"
        " a)",
        "a = 5*SQRT(9 - 5*SIN(t)^2)*SIN(t)^2/6 - 3*SQRT(9 - 5*SIN(t)^2)/2",
    ),
]


@pytest.mark.parametrize(("text", "expected"), LINEAR, ids=str)
def test_a_linear_system_is_solved_or_reduced(text, expected):
    assert simp(text) == expected


TWISTER = (
    "[2*a + l = 9, a + 2*f + e = 18, 2*e + l + f + n + t + a = 40, 2*e + n + t = 29,"
    " r + a + b + e = 22, r + a + 2*t + e = 31, 2*t + 2*a + r + n + l + e = 45,"
    " z + e + b + r + a = 23]"
)


def test_an_underdetermined_system_reduces_to_the_rows_of_its_echelon_form():
    """Eight equations in nine unknowns: `z` is determined, `t` stays free."""
    assert simp(f"SOLVE({TWISTER}, [a, b, e, f, l, n, r, t, z])") == (
        "[a - t/5 = 4/5 AND b - 2*t = -9 AND e + 3*t/5 = 58/5 AND f - t/5 = 14/5"
        " AND l + 2*t/5 = 37/5 AND n - t/5 = 29/5 AND r + 8*t/5 = 93/5 AND z = 1]"
    )
    assert simp(f"SOLVE({TWISTER}, [a, b, e, f, l, n, r, z])") == (
        "[a = (t + 4)/5 AND b = 2*t - 9 AND e = (58 - 3*t)/5 AND f = (t + 14)/5"
        " AND l = (37 - 2*t)/5 AND n = (t + 29)/5 AND r = (93 - 8*t)/5 AND z = 1]"
    )
    # `z` is not solved for, and the system holds for no `z` but 1.
    assert simp(f"SOLVE({TWISTER}, [a, b, e, f, l, n, r, t])") == (
        "[a - t/5 = 4/5 AND b - 2*t = -9 AND e + 3*t/5 = 58/5 AND f - t/5 = 14/5"
        " AND l + 2*t/5 = 37/5 AND n - t/5 = 29/5 AND r + 8*t/5 = 93/5 AND z = 1]"
    )
    assert simp(f"SOLUTIONS({TWISTER}, [a, b, e, f, l, n, r, t, z])") == (
        "[[@1, 10*@1 - 17, 14 - 3*@1, @1 + 2, 9 - 2*@1, @1 + 5, 25 - 8*@1, 5*@1 - 4, 1]]"
    )


# -- nonlinear systems ----------------------------------------------------------


NONLINEAR = [
    (
        "SOLVE(x^2 + y^2 = 65 AND x - 5*y = -13, [x, y])",
        {"x = -8 AND y = 1", "x = 7 AND y = 4"},
    ),
    (
        "SOLVE(a + b = -7/2 AND a*b = 11/2, [a, b])",
        {
            "a = -7/4 - SQRT(39)*#i/4 AND b = -7/4 + SQRT(39)*#i/4",
            "a = -7/4 + SQRT(39)*#i/4 AND b = -7/4 - SQRT(39)*#i/4",
        },
    ),
    ("SOLVE(x^2 + y^2 = 1 AND x^2 + y^2 = 2, [x, y])", {"false"}),
    # Inconsistent only where `a` and `b` differ, so not `false`.
    (
        "SOLVE(x^2 + y^2 = a AND x^2 + y^2 = b, [x, y])",
        {"x^2 + y^2 - a = 0 AND x^2 + y^2 - b = 0"},
    ),
    (
        "SOLVE((3*u^2 - 1)/(10*u^2) = 0 AND -(3*u^4 - 4*u^2 + 1)/(10*u^2) = 0, u)",
        {"u = -SQRT(3)/3", "u = SQRT(3)/3"},
    ),
]


@pytest.mark.parametrize(("text", "expected"), NONLINEAR, ids=named)
def test_a_nonlinear_system_answers_with_its_solutions(text, expected):
    assert disjuncts(text) == expected


def test_a_vector_of_nonlinear_equations_answers_with_a_vector():
    assert elements("SOLVE([x^2 + y^2 = 65, x - 5*y = -13], [x, y])") == {
        "x = 7 AND y = 4",
        "x = -8 AND y = 1",
    }
    assert simp("SOLVE([x^2 + y^2 = 1, x^2 + y^2 = 2], [x, y])") == "[]"
    assert elements(
        "SOLVE([x^2 - x*y - y - 113, y^2 - x*y + x = -41], [x, y])"
    ) == {"x = 15 AND y = 7", "x = -61/5 AND y = -16/5"}


# -- SOLUTIONS ------------------------------------------------------------------


SOLUTIONS = [
    ("SOLUTIONS(x^2 - 1 = 0, x)", {"1", "-1"}),
    ("SOLUTIONS(2*x - 2*t - 5 = 0, x)", {"t + 5/2"}),
    ("SOLUTIONS((4 + k*3)*2^(1/2)*1/8 = 1, k)", {"4*SQRT(2)/3 - 4/3"}),
    ("SOLUTIONS(m*lambda^2 + b*lambda + k = 0, lambda)", {
        "-(SQRT(b^2 - 4*k*m) + b)/(2*m)", "(SQRT(b^2 - 4*k*m) - b)/(2*m)"
    }),
    ("SOLUTIONS(s^2 + 0.1*s + 9 = 0, s)", {
        "-1/20 + SQRT(3599)*#i/20", "-1/20 - SQRT(3599)*#i/20"
    }),
    ("SOLUTIONS(DIF((3*u^2 + 1)/(10*u), u), u)", {"SQRT(3)/3", "-SQRT(3)/3"}),
    ("SOLUTIONS(SQRT(3)/5 = (3*u^2 + 1)/(10*u), u)", {"SQRT(3)/3"}),
    ("SOLUTIONS(r^4 - 2*r^3 + 2*r^2 - 2*r + 1, r)", {"1", "#i", "-#i"}),
    ("SOLUTIONS(r^3 + 3*r^2 - 4*r - 12, r)", {"2", "-2", "-3"}),
    ("SOLUTIONS(x^4 - 1, x)", {"1", "-1", "#i", "-#i"}),
    ("SOLUTIONS(28*x^3 - 64*x^2 + 3*x + 45, x)", {"3/2", "-5/7"}),
    ("SOLUTIONS(SQRT(y) = x, y)", {"IF(x >= 0, x^2)"}),
    ("SOLUTIONS(x^2 + y^2 = 2*a^2 AND x - y = 0, [x, y])", {"[a, a]", "[-a, -a]"}),
    ("SOLUTIONS(a + b = 1 AND a*b = -2, [a, b])", {"[-1, 2]", "[2, -1]"}),
    (
        "SOLUTIONS([x + a*y + a^2*z = a^3, x + b*y + b^2*z = b^3,"
        " x + c*y + c^2*z = c^3], [x, y, z])",
        {"[a*b*c, -a*(b + c) - b*c, a + b + c]"},
    ),
    (
        "SOLUTIONS([0 = 10000*x^3 - 7500*x^2*y - 2200*x + 1800*y,"
        " 0 = -2500*x^3 + 1800*x - 1350*y], [x, y])",
        {"[0, 0]", "[SQRT(3)/5, 2*SQRT(3)/9]", "[-SQRT(3)/5, -2*SQRT(3)/9]"},
    ),
]


@pytest.mark.parametrize(("text", "expected"), SOLUTIONS, ids=named)
def test_solutions_lists_the_values(text, expected):
    assert elements(text) == expected


def test_solutions_without_solutions_or_with_every_value():
    assert simp("SOLUTIONS(x = x + 1, x)") == "[]"
    assert simp("SOLUTIONS(0 = 0, x)") == "[@1]"
    assert simp("SOLUTIONS(x/(2*x - 8) + (x - 6)/(x - 4) = 3/2, [x])") == "[[@1]]"
    assert simp("SOLUTIONS([x + 3*y = 1, 2*x + 6*y = 3], [x, y])") == "[]"


def test_solutions_of_an_underdetermined_system_is_a_parametric_row():
    assert simp("SOLUTIONS(2*x + 6*y = 2 AND 3*x = 3 - 9*y, [x, y])") == (
        "[[@1, (1 - @1)/3]]"
    )
    assert simp("SOLUTIONS([x + y = 1], [x, y])") == "[[@1, 1 - @1]]"
    assert simp("SOLUTIONS([x - y + 3*z = 10 AND 2*x + 4*y + 3*z = 2], [x, y, z])") == (
        "[[@1, -(@1 + 8)/5, (14 - 2*@1)/5]]"
    )


def test_solutions_is_an_ordinary_vector():
    assert elements("VECTOR(1/k, k, SOLUTIONS(x^2 - 5*x + 6, x))") == {"1/2", "1/3"}
    assert simp("SELECT(x > 0, x, SOLUTIONS(x^2 = 4, x))") == "[2]"
    text = "FIRST(SOLUTIONS(a = 3 AND b = -3.2 AND c = #e + 0.5*#i, [a, b, c]))"
    assert simp(text) == "[3, -16/5, #e + #i/2]"
    assert simp(
        "[VARIABLES(a = 3 AND b = 7), (SOLUTIONS(a = 3 AND b = 7,"
        " VARIABLES(a = 3 AND b = 7))) SUB 1]"
    ) == "[[a, b], [3, 7]]"


def test_solutions_needs_a_variable():
    assert simp("SOLUTIONS(x^2 = 4)") == "SOLUTIONS(x^2 = 4)"


# -- numeric solving ------------------------------------------------------------

TEN = Context(precision_digits=10, notation_digits=10)


NUMERIC = [
    (
        "NSOLVE(x^5 - 2*x^3 + 4*x^2 + 2*x - 3 = 0, x)",
        {
            "x = 1.013396467 - 1.152395734*#i",
            "x = 1.013396467 + 1.152395734*#i",
            "x = 0.7265501430",
            "x = -1.753343077",
            "x = -1",
        },
    ),
    (
        "NSOLVE(x^5 - 2*x^3 + 4*x^2 + 2*x - 3 = 0, x, Real)",
        {"x = 0.7265501430", "x = -1.753343077", "x = -1"},
    ),
    ("NSOLVE(x*(x - 1)*(x + 3)*(3*x - 4), x)", {
        "x = 1.333333333", "x = -3", "x = 1", "x = 0"
    }),
    ("NSOLVE(x^5 + x + 1 = 0, x, Real)", {"x = -0.7548776662"}),
    ("NSOLVE(x^3 - 3*x + 1, x)", {
        "x = -1.879385241", "x = 0.3472963553", "x = 1.532088886"
    }),
    ("NSOLVE((6*k + 1)*(12*k + 1)*(18*k + 1) = 1000, k, Real)", {"k = 0.8165433244"}),
    (
        "NSOLVE(z^4 - 20*z^3 - 66*z^2 + 1420*z - 2315"
        " - #i*(24*z^3 - 340*z^2 + 736*z + 1100), z)",
        {"z = 8 + 9*#i", "z = 6 + 7*#i", "z = 4 + 5*#i", "z = 2 + 3*#i"},
    ),
    ("NSOLVE(w^2 - 1.360404999*w + 0.2393721784, w, Real)", {
        "w = 1.152752244", "w = 0.2076527540"
    }),
    ("NSOLVE(0.7071067811*x = 0.3535533904, x)", {"x = 0.4999999997"}),
    ("NSOLVE(x^4 - 4 = 0, x, Real)", {"x = -1.414213562", "x = 1.414213562"}),
    ("NSOLVE(x^4 - 4 = 0, x, -10, 10)", {"x = -1.414213562", "x = 1.414213562"}),
    ("NSOLVE((x - pi)^4 = 0, x, 3, 4)", {"x = 3.141592653"}),
    (
        "NSOLVE(x^4 - 4*pi*x^3 + 6*pi^2*x^2 - 4*pi^3*x + pi^4 = 0, x, 3, 4)",
        {"x = 3.141592653"},
    ),
    ("NSOLVE(#e^x*(x - 5) + 5 = 0, x)", {"x = 0"}),
    ("NSOLVE(#e^x*(x - 5) + 5 = 0, x, 1, 10)", {"x = 4.965114231"}),
    ("NSOLVE(SIN(x) = 1 - x/6, x, 0, 3)", {"x = 0.9885669796"}),
    ("NSOLVE(SIN(x) = 1 - x/6, x, 1, 3)", {"x = 2.523600393"}),
    ("NSOLVE(TAN(x) = 2*x, x, 0.1, 10)", {"x = 1.165561185"}),
    ("NSOLVE(3*x*COS(x) - SIN(x) = 0, x, 1, 3)", {"x = 1.324194449"}),
    ("NSOLVE(x^(2/3) - 2*x^(1/3) - 35 = 0, x, -1000, 0)", {"false"}),
    ("NSOLVE(x^(2/3) - 2*x^(1/3) - 35 = 0, x, Real)", {"x = 343"}),
    ("NSOLVE(1 = x*#e^x, x)", {"x = 0.5671432904"}),
    ("NSOLVE(#e^x + 1 = 0, x)", {"false"}),
    ("NSOLVE(1/x = 0, x, -1, 1)", {"false"}),
    ("NSOLVE(0 = g*75/15*(1 - #e^(-15*0/75)) + c, c, Real)", {"c = 0"}),
]


@pytest.mark.parametrize(("text", "expected"), NUMERIC, ids=named)
def test_nsolve_answers_with_numbers(text, expected):
    assert disjuncts(text, TEN) == expected


def test_nsolutions_answers_with_a_vector_of_numbers():
    assert elements("NSOLUTIONS(x*(x - 1)*(x + 3)*(3*x - 4), x)", TEN) == {
        "0", "1", "-3", "1.333333333"
    }
    assert elements("NSOLUTIONS(20*b^3 - 54*b^2 - 114*b + 200, b)", TEN) == {
        "1.329420146", "-2.141669350", "3.512249203"
    }
    assert elements("NSOLUTIONS(x^3 + 3*x^2 - 4*x + 10, x)", TEN) == {
        "-4.417802089", "0.7089010446 + 1.327037430*#i", "0.7089010446 - 1.327037430*#i"
    }
    assert simp("NSOLUTIONS(#e^x + 1 = 0, x)", TEN) == "[]"
    assert simp("CEILING(NSOLUTIONS(1 - (5/6)^x = 0.9, x))") == "[13]"
    text = "CEILING(RHS(NSOLVE((6*k + 1)*(12*k + 1)*(18*k + 1) = 1000, k, Real)))"
    assert simp(text) == "1"


def test_nsolve_in_bounds_finds_the_roots_there():
    text = "VECTOR(NSOLUTIONS(1.5*x*COS(x) - SIN(x)/2 = 0, x, a, a + 1), a, [1, 2, 4, 7])"
    assert simp(text, TEN) == "[[1.324194449], [], [4.640683630], [7.811334475]]"
    text = (
        "APPEND(VECTOR(NSOLUTIONS(1.5*x*COS(x) - SIN(x)/2 = 0, x, a, a + 1), a, "
        "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))"
    )
    assert simp(text, TEN) == "[0, 1.324194449, 4.640683630, 7.811334475]"


def test_nsolve_tells_roots_apart_that_lie_closer_than_its_scan():
    """`SIN(x^2) = #e^x` has pairs of roots about 0.00015 apart near each zero of
    `SIN(x^2)`; between them the scan sees no sign change, interval arithmetic does."""
    assert simp("NSOLVE(SIN(x^2) - #e^x, x, -6.9, -6.8)", TEN) == "x = -6.864608198"
    assert simp("NSOLVE(SIN(x^2) - #e^x, x, -7, -6.5)", TEN) == "x = -6.864608198"


def test_nsolve_finds_its_root_to_every_digit_asked_for():
    context = Context(precision_digits=40, notation_digits=40)
    assert simp("NSOLVE(1 = x*#e^x, x)", context) == (
        "x = 0.5671432904097838729999686622103555497538"
    )


def test_nsolve_answers_false_only_where_it_proves_there_is_no_root():
    """A search that finds nothing proves nothing, so the call stays; interval
    arithmetic proves that `COS(x) - x` has no root below -5, and the imaginary part
    of `x^(2/3) - 2*x^(1/3) - 35` that it has none below 0."""
    assert simp("NSOLVE(COS(x) = x, x, -inf, -5)", TEN) == "false"
    assert simp("NSOLVE(x^(2/3) - 2*x^(1/3) = 35, x, -10, 0)", TEN) == "false"
    assert simp("NSOLVE(#e^(-x) = 0, x)", TEN) == "NSOLVE(#e^(-x) = 0, x)"
    assert simp("NSOLUTIONS(#e^(-x) = 0, x)", TEN) == "[]"


async def test_nsolve_evaluates_a_conditional_at_each_point_it_tries():
    """A user's function whose `IF` can't be decided for a symbolic argument is
    evaluated at the numbers the search substitutes."""
    session = Session(step_limit=-1)
    session.author("F(t) := IF(t < 0, 1 - F(-t), 1 - #e^(-t)/2)")
    answer = await session.simplify("NSOLVE(F(t) = 0.9, t)")
    assert answer.text == "t = 1.609437912"
    answer = await session.simplify("NSOLVE(F(t) = 0.1, t)")
    assert answer.text == "t = -1.609437912"


NUMERIC_STANDING = [
    "NSOLVE([3*x^2 + z - 10, x - 4*y^2 + 2*z + 10, x^2 + y^2 - 10], [x, y, z])",
    "NSOLVE(x^2 < 2, x)",
    "NSOLVE(x^2 = a, x)",
    "NSOLVE(SIN(x) = 0, x, 4, 3)",
    "NSOLUTIONS(x^2 = a, x)",
]


@pytest.mark.parametrize("text", NUMERIC_STANDING, ids=str)
def test_nsolve_takes_one_equation_in_one_variable(text):
    assert simp(text, state=word_mode()) == text


# -- approximate solving --------------------------------------------------------


APPROXIMATED = [
    (
        "APPROX(SOLVE(x^5 + x + 3 = 0, x))",
        {
            "x = -0.4753807566 - 1.129701725*#i",
            "x = -0.4753807566 + 1.129701725*#i",
            "x = 1.041879539 - 0.8228703381*#i",
            "x = 1.041879539 + 0.8228703381*#i",
            "x = -1.132997565",
        },
    ),
    (
        "APPROX(SOLVE(-2*x^3 + 4*x^2 + 2*x - 3 = 0, x))",
        {"x = 0.7703185293", "x = 2.139726157", "x = -0.9100446871"},
    ),
    ("APPROX(SOLVE(x^3 = 2*4^3, x, Real))", {"x = 5.039684199"}),
    (
        "APPROX(SOLVE(x^4 - 4 = 0, x))",
        {
            "x = -1.414213562*#i",
            "x = 1.414213562*#i",
            "x = -1.414213562",
            "x = 1.414213562",
        },
    ),
    ("APPROX(SOLVE(SIN(x) = 1 - x/6, x))", {"x = 0.9885669796"}),
    # A number is distributed over the sum it multiplies.
    (
        "APPROX(SOLVE(-2.97695*COS(s) - 0.243867*SIN(s) - 0.0936123*t - 8.99998, t))",
        {"t = -31.80084241*COS(s) - 2.605074333*SIN(s) - 96.14099856"},
    ),
]


@pytest.mark.parametrize(("text", "expected"), APPROXIMATED, ids=named)
def test_approximating_a_solve_approximates_its_solutions(text, expected):
    assert disjuncts(text, TEN) == expected


def test_approximate_precision_solves_numerically_where_solving_exactly_fails():
    context = TEN.with_precision(Precision.APPROXIMATE)
    assert simp("SOLVE(x^5 + x + 3 = 0, x, Real)", context) == "x = -1.132997565"
    assert elements("SOLUTIONS(-3*x^2 - 18*x + 1040 = 0, x)", context) == {
        "15.85912687", "-21.85912687"
    }
    assert simp("SOLVE(4*(4.64816) + 4*k = 40, k)", context) == "k = 5.35184"


def test_any_branch_solves_a_symbolic_power_whatever_the_base():
    # ODE.MTH's POLES finds the second singular point of the hypergeometric equation in
    # `(x - r)^k`. A power to an exponent not known to be positive is no zero factor.
    context = Context(branch=Branch.ANY)
    text = "-(x - r)*(1 - z*(x - r)^k)/(2*k*z*(x - r)^k) = 0"
    assert sol(text, ("x",), context) == ["x = (1/z)^(1/k) + r"]
    assert sol("x^k = z", ("x",), context) == ["x = z^(1/k)"]
    assert sol("x^k = z", ("x",)) == ["x^k - z = 0"]


def test_a_power_to_a_symbolic_exponent_stays_among_the_solutions_of_the_others():
    """Whether `t^(k - 1)` vanishes depends on the sign of `k - 1`."""
    text = "SOLVE(0 = DIF(COMB(n, k)*t^k*(1 - t)^(n - k), t), t)"
    assert disjuncts(text) == {"t = k/n", "t^(k - 1) = 0", "(1 - t)^(-k + n - 1) = 0"}
    text = "SOLUTIONS(0 = DIF(COMB(n, k)*t^k*(1 - t)^(n - k), t), t)"
    assert simp(text) == "[k/n]"


def test_a_system_whose_radical_roots_are_huge_is_solved_in_numbers():
    """The six Lagrange equations for the distance between an ellipse and a circle
    eliminate to a polynomial of degree 8 whose radical roots, of some 200 operations
    each, `solve` would substitute back for minutes. Under Approximate every part of
    the reduced basis is solved numerically, the points where the curves meet
    included."""
    lagrange = (
        "[2*x1*(l + 1) - 2*x2 = 0, 2*x2*(u + 1) - 2*x1 + 6*u = 0,"
        " 2*y1*(4*l + 1) - 2*y2 = 0, 2*y2*(u + 1) - 2*y1 - 4*u = 0,"
        " x1^2 + 4*y1^2 = 4, (x2 + 3)^2 + (y2 - 2)^2 = 49]"
    )
    context = Context(precision_digits=5, notation_digits=5).with_precision(
        Precision.APPROXIMATE
    )
    text = f"(SOLUTIONS({lagrange}, [x1, y1, x2, y2, l, u], Real)) COL [1, 2, 3, 4]"
    assert elements(text, context) == {
        "[1.9516, -0.21860, -9.3881, 4.8622]",
        "[-1.7254, 0.50570, -7.5427, 7.3257]",
        "[-1.7254, 0.50570, 1.5427, -3.3257]",
        "[1.9516, -0.21860, 3.3880, -0.86222]",
    }


def test_the_real_branch_takes_the_real_root_of_a_negative_number():
    real = Context(branch=Branch.REAL)
    text = "SOLVE(x^(2/3) - 2*x^(1/3) - 35 = 0, x, Real)"
    assert disjuncts(text, real) == {"x = 343", "x = -125"}
    assert disjuncts(text) == {"x = 343"}
    assert simp("SOLVE(x^(1/3) = -2, x)", real) == "x = -8"


def test_a_declared_parameter_decides_a_conditional_solution():
    state = words()
    for declaration, expected in (
        ("x :epsilon Real [0, inf)", "y = x^2"),
        ("x :epsilon Real (-inf, 0)", "false"),
    ):
        name, domain = domain_of_node(parse(declaration, state))
        context = Context(domains={name: domain})
        assert simp("SOLVE(SQRT(y) = x, y)", context) == expected
    assert simp("SOLVE(SQRT(y) = x, y)") == "y = IF(x >= 0, x^2)"


def test_a_polynomial_system_is_solved_numerically_where_its_roots_have_no_radicals():
    """In Exact precision the solutions on the eliminating sextic stay as its reduced
    system; approximated, its roots are found and substituted back."""
    text = "SOLVE(x^2 + y^2 = 5 AND x^2 + z + y = 10 AND y^2 + x*z = 20, [x, y, z])"
    assert " = -366025" in simp(text)
    approximated = disjuncts(f"APPROX({text})")
    assert len(approximated) == 6
    assert (
        "x = -4.172073972 + 0.4309770664*#i AND y = 0.5089874532 + 3.532637573*#i"
        " AND z = -7.729447452 + 0.06349882973*#i"
    ) in approximated
    assert disjuncts(
        "APPROX(SOLVE(3*x^2 + z = 10 AND x - 4*y^2 + 2*z + 15 = 0 AND x^2 + y^2 = 8,"
        " [x, y, z]))"
    ) == {
        "x = -1 AND y = -2.645751311 AND z = 7",
        "x = -1 AND y = 2.645751311 AND z = 7",
        "x = 1.5 AND y = -2.397915761 AND z = 3.25",
        "x = 1.5 AND y = 2.397915761 AND z = 3.25",
    }


def test_a_numeric_solution_of_a_system_satisfies_every_equation():
    """A root of one equation of the reduced system, `b^2 + 2*b + c^2 = 0` with `c` a root
    of the quintic, need not satisfy the next one in the same unknowns: only the
    branches that solve all of them are solutions."""
    text = (
        "APPROX(SOLVE([2*a + b + c + d, a*(b + c + d) + b*(c + d - 1) + c*d,"
        " a*(b*(c + d) + c*d) + b*c*d + c, -a*b*c*d + d], [a, b, c, d], Real))"
    )
    assert elements(text) == {
        "a = 0 AND b = 0 AND c = 0 AND d = 0",
        "a = 1 AND b = -1 AND c = -1 AND d = 0",
        "a = 1 AND b = -2 AND c = 0 AND d = 0",
        "a = 1 AND b = -1.754877666 AND c = -0.5698402909 AND d = 0.3247179572",
        "a = 0.5651977173 AND b = -1.769292354 AND c = 0.6388969194 AND d = 0",
    }


def test_a_root_of_a_quotient_in_a_system_s_solution_is_simplified():
    """Substituting `d` into the second equation gives the root of an expanded quartic
    over `(r1 - r2)^2`; the solution is written as Simplify writes that root."""
    text = (
        "SOLVE(x = (2*d*(r1 - r2) + m^2 + r1^2 - r2^2)/(2*m) AND y = -SQRT((m^2 - r1^2"
        " + 2*r1*r2 - r2^2)*(4*d^2 + 4*d*(r1 + r2) - m^2 + r1^2 + 2*r1*r2 + r2^2))/(2*m),"
        " [y, d])"
    )
    assert simp(text) == (
        "y = -SQRT((-m - r1 + r2)*(-m + r1 - r2)*(2*x - m + r1 - r2)*(2*x - m - r1 + r2))"
        "*SIGN(m)/(2*ABS(r1 - r2)) AND d = (2*m*x - m^2 - r1^2 + r2^2)/(2*(r1 - r2))"
    )


def test_a_linear_system_in_functions_of_a_parameter_is_solved_in_their_identities():
    text = (
        "SOLUTIONS(r*COS(phi) = 2*c*COS(phi)^2 - c AND r*SIN(phi) = 2*c*SIN(phi)*COS(phi)"
        " - c*SIN(phi)/COS(phi), [r, x])"
    )
    assert simp(text) == "[[c*(2*COS(phi) - 1/COS(phi)), @1]]"


MANUAL = [
    ("(2*x - 3 = 5) + 3", "2*x = 8"),
    ("(2*x = 8)/2", "x = 4"),
    ("(x^2 + 5*x + 6 = 0) - 6", "x^2 + 5*x = -6"),
    ("#e^(LN(x) = 5)", "x = #e^5"),
    ("(3*x - 4 = 5 + 8*x) - 8*x", "-5*x - 4 = 5"),
    ("(-5*x = 9)/(-5)", "x = -9/5"),
    ("LN(#e^(x + 3) = 5)", "x + 3 = LN(5)"),
    ("((x + 1)^(1/3) = 8 - x)^3", "x + 1 = (8 - x)^3"),
    ("(2^(2 - x) + 3 = 9) - 3", "2^(2 - x) = 6"),
    # Squaring adds a solution; nothing checks it.
    ("(SQRT(x) = -1)^2", "x = 1"),
]


@pytest.mark.parametrize(("text", "expected"), MANUAL, ids=str)
def test_an_equation_is_solved_by_hand_by_operating_on_both_sides(text, expected):
    assert simp(text) == expected


# -- equations between statements ------------------------------------------------


def test_an_equation_between_two_equations_is_their_equivalence():
    """Equal truth values: both hold or neither does. `SOLUTIONS` lists the points where
    both hold; where neither does, a region in both variables, nothing is solved."""
    assert simp("(1 = 1) = (2 = 3)") == "false"
    assert simp("(1 = 1) /= (2 = 3)") == "true"
    assert simp("SOLUTIONS((y = 2*x) = (y = 3 - x), [x, y])") == "[[1, 2]]"
    assert simp("SOLUTIONS(x /= 1, x)") == "SOLUTIONS(x /= 1, x)"


async def test_an_equation_between_two_equations_passed_as_arguments():
    """An intersection program written with `l1 = l2` for two lines."""
    session = Session(step_limit=-1)
    session.author("InputMode := Word")
    session.author("IP(l1, l2) := (SOLUTIONS(l1 = l2, [x, y])) SUB 1")
    answer = await session.simplify("IP(y = 2*x, y = 3 - x)")
    assert answer.text == "[1, 2]"


async def test_solutions_leave_out_the_zeros_of_a_denominator_through_its_values():
    # The divisors are read with the values of their names: `x - a` vanishes at a
    # solution of the numerator when a is 1.
    session = Session()
    session.author("a := 1")
    answer = await session.simplify("SOLUTIONS((x^2 - 1)/(x - a) = 0, x)")
    assert answer.text == "[-1]"


def test_solutions_of_a_conjunction_holding_a_solve_answer():
    # The solve answer's equations, in their order, are a conjunction like any other.
    assert simp("SOLUTIONS(p = 8 AND SOLVE([a = 2, c = 3], [a, c]), [a, c, p])") == (
        "[[2, 3, 8]]"
    )


def test_a_root_next_to_an_end_where_the_equation_is_undefined():
    # The piece next to 0, where the residual is undefined, is scanned again until the
    # root at 10^-6 is bracketed.
    assert simp("NSOLVE((x - 10^(-6))/x, x, 0, 1)") == "x = 10^-6"


def test_a_candidate_offered_under_a_condition_holds_where_its_parameter_allows():
    assert simp("SOLVE(t + 2*SQRT(v) = 24, v)") == "v = IF(t <= 24, (t - 24)^2/4)"


def test_a_candidate_holds_on_the_side_of_zero_where_its_root_is_written():
    # The candidate leaves the root of a perfect square, the absolute value of what is
    # squared, whose sign decides it.
    assert simp("SOLUTIONS(LN(SQRT(v) - x) = k, v)") == (
        "[IF(#e^k + x >= 0, (#e^k + x)^2)]"
    )
    # `LN(x)` is real only where `x > 0`.
    assert simp("SOLUTIONS(-LN(SQRT(x^2 + 4*v) - x) = LN(x) + k, v)") == (
        "[IF(x > 0, (2*x^2*#e^k + 1)*#e^(-2*k)/(4*x^2))]"
    )
    # The real candidates hold for `x >= 0`, the complex ones, where `v^2 + 1` is a
    # cube root of unity times `ABS(x)^(2/3)`, for `x <= 0`.
    assert simp("SOLUTIONS((v^2 + 1)^(3/2) = x, v)") == (
        "[IF(x <= 0, -SQRT(2)*SQRT((-1 + SQRT(3)*#i)*(-x)^(2/3) - 2)/2), "
        "IF(x <= 0, SQRT(2)*SQRT((-1 + SQRT(3)*#i)*(-x)^(2/3) - 2)/2), "
        "IF(x <= 0, -SQRT(2)*SQRT(-(1 + SQRT(3)*#i)*(-x)^(2/3) - 2)/2), "
        "IF(x <= 0, SQRT(2)*SQRT(-(1 + SQRT(3)*#i)*(-x)^(2/3) - 2)/2), "
        "IF(x >= 0, -SQRT(x^(2/3) - 1)), IF(x >= 0, SQRT(x^(2/3) - 1))]"
    )
    # At `x = 0` the candidates are undefined, though the equation evaluates to 0.
    assert simp("SOLUTIONS(x - 2/(v^2 + 1)^(3/2) = 0, v)") == (
        "[IF(x < 0, -SQRT(2)*SQRT(-2^(2/3)/(-x)^(2/3) - 2^(2/3)*SQRT(3)*#i/(-x)^(2/3)"
        " - 2)/2), "
        "IF(x < 0, SQRT(2)*SQRT(-2^(2/3)/(-x)^(2/3) - 2^(2/3)*SQRT(3)*#i/(-x)^(2/3)"
        " - 2)/2), "
        "IF(x < 0, -SQRT(2)*SQRT(-2^(2/3)/(-x)^(2/3) + 2^(2/3)*SQRT(3)*#i/(-x)^(2/3)"
        " - 2)/2), "
        "IF(x < 0, SQRT(2)*SQRT(-2^(2/3)/(-x)^(2/3) + 2^(2/3)*SQRT(3)*#i/(-x)^(2/3)"
        " - 2)/2), "
        "IF(x > 0, -SQRT(2^(2/3)/x^(2/3) - 1)), IF(x > 0, SQRT(2^(2/3)/x^(2/3) - 1))]"
    )
    assert simp("SOLUTIONS(SQRT(v^2 + 1) = x, v)") == (
        "[IF(x >= 0, -SQRT(x^2 - 1)), IF(x >= 0, SQRT(x^2 - 1))]"
    )


def test_a_principal_power_of_a_root_of_unity_is_decided_exactly():
    # `v^(5/2)` is real for `v` at an angle of 0 or `+-4*pi/5` where `x >= 0`, and at
    # `+-2*pi/5` where `x <= 0`.
    assert simp("SOLVE(x = v^(5/2), v)") == (
        "v = IF(x <= 0, #i*(SQRT(2)*SQRT(SQRT(5) + 5) - SQRT(5)*#i + #i)*(-x)^(2/5)/4)"
        " OR v = IF(x <= 0, -#i*(SQRT(2)*SQRT(SQRT(5) + 5) + SQRT(5)*#i - #i)"
        "*(-x)^(2/5)/4)"
        " OR v = IF(x >= 0, -#i*(SQRT(2)*SQRT(5 - SQRT(5)) - SQRT(5)*#i - #i)"
        "*x^(2/5)/4)"
        " OR v = IF(x >= 0, #i*(SQRT(2)*SQRT(5 - SQRT(5)) + SQRT(5)*#i + #i)*x^(2/5)/4)"
        " OR v = IF(x >= 0, x^(2/5))"
    )
    assert simp("SOLVE(x = v^(7/2), v)") == (
        "v = IF(x <= 0, (COS(2*pi/7) + #i*SIN(2*pi/7))*(-x)^(2/7))"
        " OR v = IF(x <= 0, (COS(2*pi/7) - #i*SIN(2*pi/7))*(-x)^(2/7))"
        " OR v = IF(x <= 0, -(COS(pi/7) + #i*SIN(pi/7))*(-x)^(2/7))"
        " OR v = IF(x <= 0, -(COS(pi/7) - #i*SIN(pi/7))*(-x)^(2/7))"
        " OR v = IF(x >= 0, -(COS(3*pi/7) + #i*SIN(3*pi/7))*x^(2/7))"
        " OR v = IF(x >= 0, -(COS(3*pi/7) - #i*SIN(3*pi/7))*x^(2/7))"
        " OR v = IF(x >= 0, x^(2/7))"
    )


def test_a_candidate_at_a_point_where_another_gives_its_value_is_dropped():
    # The cube roots of unity times `x^(4/3)` are solutions only at `x = 0`, where they
    # are 0, as `x^(4/3)` is.
    assert simp("SOLVE(x = v^(3/4), v)") == "v = IF(x >= 0, x^(4/3))"


def test_an_integral_to_infinity_is_no_infinite_side():
    # Its limit is infinite, not its value; a case of a conditional may be infinite.
    assert simp("SOLVE(y = INT(#e^(-s*t), t, 0, inf), y)") == "y = IF(s > 0, 1/s, inf)"
    assert (
        simp("SOLUTIONS(2*y = INT(#e^(-s*t)*SIN(t), t, 0, inf), y)")
        == "[IF(s > 0, 1/(s^2 + 1), INT(#e^(-s*t)*SIN(t), t, 0, inf))/2]"
    )
    assert simp("SOLVE(TAN(2*a) = 4/0, a)") == "false"


def test_a_root_isolated_in_several_parameters_gives_guarded_solutions():
    # Each root of the square solves the equation where the side the root equals is
    # not negative: here at a = b = 1, x = 2, y = 3 only the second does.
    assert simp("SOLVE(SQRT(k^2 + b) = y - k*x, k)") == (
        "k = IF((x*SQRT(b*x^2 + y^2 - b) + y)/((x + 1)*(x - 1)) <= 0, "
        "(SQRT(b*x^2 + y^2 - b) + x*y)/((x + 1)*(x - 1))) OR "
        "k = IF((x*SQRT(b*x^2 + y^2 - b) - y)/((x + 1)*(x - 1)) >= 0, "
        "(x*y - SQRT(b*x^2 + y^2 - b))/((x + 1)*(x - 1)))"
    )
