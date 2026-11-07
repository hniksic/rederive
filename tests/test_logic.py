"""What Simplify makes of relations, logic and conditionals.

A relation is decided where the arithmetic or the declared domains settle it and stands
where they do not: `x^2 >= 0` is `true`, `x < x + 1` stands of a complex `x`, since
only the reals are ordered, and two matrices are related element by element, so
`[x, y] = [1, 2]` is `[x = 1, y = 2]`. Logic has an algebra, `p AND (q OR p)` being
`p`, and a truth table its columns. A conditional takes the branch its test decides and
comes back as written where nothing does, `IF(x > 0, 1, -1)` being `1` only once `x` is
declared positive. Each case is the author's text in and the worksheet's text out,
written inline as in `test_simplify.py` so that a failure names the mathematics.
"""

from __future__ import annotations

import pytest
from simplifying import COMPLEX_X, POSITIVE_X, declared, parse, simp

from rederive.engine.computing import Context


# -- relations and logic ------------------------------------------------------


def test_a_relation_that_decides_is_decided_and_one_that_does_not_stands():
    """A relation is a statement, and a statement that says something definite
    is simplified by saying it. Every row here was checked against the
    original, which answers all six `true` or `false`.

    The judgement is three-valued: `3*x = x^2 + c` settles nothing and so comes
    back as it was authored, sides simplified and the relation standing. That
    is what leaves a conditional's test alone, and what leaves soLve something
    to solve."""
    assert simp("2 < 3") == "true"
    assert simp("2 = 2") == "true"
    assert simp("1 = 2") == "false"
    assert simp("x = x") == "true"
    assert simp("2 <= 2") == "true"
    assert simp("x < x + 1") == "true"
    assert simp("x + 2*x = c + x*x") == "3*x = x^2 + c"


def test_a_relation_that_holds_for_every_value_is_decided():
    """Each holds for every value the variables can take, the last two by the
    intervals declared for them."""
    assert simp("x^2 >= 0") == "true"
    positive = declared("p :epsilon Real (0, inf)", "q :epsilon Real (0, inf)")
    assert simp("p + q > 0", positive) == "true"
    assert simp("ABS(g) < 1", declared("g :epsilon Real (0, 1)")) == "true"


def test_a_relation_between_two_matrices_is_stated_of_each_pair_of_elements():
    """Two matrices of one shape are related element by element rather than
    compared whole, and each of the relations that come out is judged on its own.
    That is what section 8.5 reads a solved system off: `[[x], [y], [z]] = A^-1 .
    b` is the three equations it stands for. The answer keeps the shape it was
    made from, except that a single row is written flat as any one-row matrix is.

    Sides the pairing does not fit are a statement about the two things
    themselves, and there the two written values are all there is to compare:
    the original answers `[1, 2] = [1, 2, 3]` false, and leaves `[x, y] = [1,
    2, 3]` standing as the statement about `x` and `y` that it is."""
    assert simp("[x, y] = [1, 2]") == "[x = 1, y = 2]"
    assert simp("[[x], [y]] = [[1], [2]]") == "[[x = 1], [y = 2]]"
    assert simp("[x, y] < [1, 2]") == "[x < 1, y < 2]"
    spread = simp("[[a, b], [c, d]] = [[1, 2], [3, 4]]")
    assert spread == "[[a = 1, b = 2], [c = 3, d = 4]]"
    assert simp("[1, 2] = [1, 2, 3]") == "false"
    assert simp("[x, y] = [1, 2, 3]") == "[x, y] = [1, 2, 3]"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Two matrices are equal when every pair of elements is, and different when
        # any pair is: an equation of two of them is one statement, and where the
        # elements settle it the answer is one truth value.
        ("[1, 2] = [1, 2]", "true"),
        ("[1, 2] = [1, 3]", "false"),
        ("[[1, 2], [3, 4]] = [[1, 2], [3, 4]]", "true"),
        ("[[1, 2], [3, 4]] = [[1, 2], [3, 5]]", "false"),
        ("[SIN(x)^2 + COS(x)^2, 2] = [1, 2]", "true"),
        # One pair that differs is enough, whatever the others say.
        ("[1, 2] = [x, 3]", "false"),
        # Not yet settled, the equation is the system of its elements.
        ("[1, 2] = [x, 2]", "[1 = x, true]"),
        # An inequation says that some pair differs, which none of its elements
        # says alone, so one not yet settled stands whole.
        ("[1, 2] /= [1, 3]", "true"),
        ("[1, 2] /= [1, 2]", "false"),
        ("[1, 2] /= [x, 3]", "true"),
        ("[1, 2] /= [x, 2]", "[1, 2] /= [x, 2]"),
        ("[x, y] /= [1, 2]", "[x, y] /= [1, 2]"),
        ("[1, 2] /= [1, 2, 3]", "true"),
        ("IF([1, 2] = [1, 2], a, b)", "a"),
    ],
    ids=str,
)
def test_an_equation_of_two_matrices_is_one_truth_value(text, expected):
    assert simp(text) == expected


def test_the_system_of_section_eight_five_is_solved_by_inverting_its_matrix():
    """The manual's own use for the rule above, in exact arithmetic: the three
    quotients are what its 2.91059, 0.175496 and 1.58278 round to."""
    solved = simp(
        "[[x], [y], [z]] = [[5, 3, -7], [2, -8, 1], [-1, 9, 4]]^-1 . [[4], [6], [5]]"
    )
    assert solved == "[[x = 879/302], [y = 53/302], [z = 239/151]]"


def test_only_the_reals_are_ordered_so_a_complex_variable_decides_nothing():
    """`x < x + 1` is a fact about the reals and says nothing about the complex
    plane, where `<` has no meaning to be true of. Equality is a question every
    domain answers, so it is decided there too."""
    assert simp("x < x + 1", COMPLEX_X) == "x < x + 1"
    assert simp("x = x", COMPLEX_X) == "true"


def test_an_equation_is_answered_by_the_arithmetic_and_not_by_the_domains():
    """The original does not put an equation to the declared domains. `x = #i`
    is where a quadratic with a negative discriminant lands, and it stands as
    written whether `x` was declared real, declared complex or declared nothing
    at all - a name nobody declared being real is what makes `SQRT(x^2)` into
    `ABS(x)`, and that is a different question. `ABS(x) + 1 = 0` is satisfied
    by no real and stands too: an equation asks about a value rather than
    claiming something to be refuted from what the value is known to be.

    What answers one is the difference of the two sides coming out a number,
    zero for one value and a nonzero real for two. A difference that is a
    number and not real answers nothing, which is the answer the original
    gives: `#i = 1` and `2*#i = #i` stand."""
    assert simp("x = #i") == "x = #i"
    assert simp("x = SQRT(-1)") == "x = #i"
    assert simp("x = #i", declared("x :epsilon Real")) == "x = #i"
    assert simp("x = #i", COMPLEX_X) == "x = #i"
    assert simp("x + #i = 0") == "x + #i = 0"
    assert simp("x^2 + 1 = 0") == "x^2 + 1 = 0"
    assert simp("x /= #i") == "x /= #i"
    assert simp("ABS(x) + 1 = 0") == "ABS(x) + 1 = 0"
    assert simp("#e^x = 0") == "#e^x = 0"
    assert simp("n = 1/2", declared("n :epsilon Integer")) == "n = 1/2"
    assert simp("p = -1", declared("p :epsilon Real (0, inf)")) == "p = -1"
    assert simp("y = 5", declared("y :epsilon Real (0, 1)")) == "y = 5"
    # What the arithmetic answers is answered, and a value declaration is
    # arithmetic: the name is the number it was declared to stand for.
    assert simp("x + 1 = x") == "false"
    assert simp("x /= x") == "false"
    assert simp("1 = 2") == "false"
    assert simp("SQRT(-4) = 2*#i") == "true"
    assert simp("2*x = x + x") == "true"
    assert simp("v = 5", Context(assignments={"v": parse("7")})) == "false"
    # A number that is not real leaves the two sides where they were.
    assert simp("#i = 1") == "#i = 1"
    assert simp("2*#i = #i") == "2*#i = #i"
    assert simp("x + #i = x") == "x + #i = x"
    # Two sides written the same are one value whatever they are written out
    # of, which is the whole of what answers the ones that have no difference.
    assert simp("INF = INF") == "true"
    assert simp("-INF = INF") == "false"
    assert simp("true = true") == "true"
    assert simp('"no" = "no"') == "true"
    assert simp('"no" = "yes"') == '"no" = "yes"'


def test_a_quadratic_whose_discriminant_turns_negative_is_answered_in_roots():
    """The manual's quadratic with numbers substituted into it, which is where
    an equation the domains falsify would take an answer away from a user: `10`
    for `a`, `1.3333` for `b` and `30` for `c` puts the discriminant below zero,
    and the root is what the original prints there."""
    values = {name: parse(value) for name, value in (("a", "10"), ("b", "1.3333"))}
    context = Context(assignments=values)
    formula = "x = (SQRT(b^2 - 4*a*c) - b)/(2*a)"
    assert simp(formula, context) == (
        "x = SQRT(177768889 - 4000000000*c)/200000 - 13333/200000"
    )
    context = Context(assignments={**values, "c": parse("30")})
    assert simp(formula, context) == (
        "x = -13333/200000 + SQRT(119822231111)*#i/200000"
    )


def test_relations_joined_over_one_variable_are_solved():
    assert simp("6 >= -2*x AND 3*x /= -9") == "x > -3"
    assert simp("x < 1 AND x > 3") == "false"


def test_a_range_is_written_as_the_chain_the_original_writes():
    """Two bounds on one variable are a range, and a range has a spelling of
    its own that the grammar reads back. The two strictnesses are independent,
    so a half-open range is a chain too."""
    assert simp("x > -2 AND x < 2") == "-2 < x < 2"
    assert simp("x >= -2 AND x < 2") == "-2 <= x < 2"


def test_a_conjunction_it_cannot_solve_keeps_its_shape():
    assert simp("x >= 1 OR x <= -1") == "x >= 1 OR x <= -1"
    # Nothing is solved here - two variables are two unknowns - and the pair is
    # still a range around `y`, which is written as the chain it is.
    assert simp("x < y AND y < 1") == "x < y < 1"


LOGIC = [
    ("NOT NOT p", "p"),
    ("NOT p", "NOT p"),
    # The algebra: absorption, the operators that are spellings for the other
    # three, a variable and its negation, and a conjunct common to every
    # disjunct. A product of sums is left folded, which is where the algebra
    # stops - Derive does not multiply one out into a sum of products.
    ("p AND (q OR p)", "p"),
    ("p OR NOT p", "true"),
    ("p AND NOT p", "false"),
    ("p IMP q", "NOT p OR q"),
    ("p XOR q", "NOT p AND q OR p AND NOT q"),
    ("NOT (p AND q)", "NOT p OR NOT q"),
    ("p AND (q OR r)", "p AND (q OR r)"),
    ("p AND r OR p AND q", "p AND (q OR r)"),
    # Operands are put in the order list's order, a term sorting by the
    # variable it leads with, and a negated literal coming first on a tie.
    ("s AND r OR q AND p", "p AND q OR r AND s"),
    ("c OR b AND a", "a AND b OR c"),
    ("p AND q OR NOT p AND r", "NOT p AND r OR p AND q"),
    # Six variables is past where the algebra pays: both normal forms are read
    # off a truth table, and this one comes back longer than it went in.
    ("a XOR b XOR c XOR d XOR e XOR f", "a XOR b XOR c XOR d XOR e XOR f"),
    # Boolean on booleans, bitwise on integers.
    ("3 OR 5", "7"),
    ("NOT 5", "-6"),
    ("12 AND 10", "8"),
    ("3 XOR 5", "6"),
    ("3 AND p", "3 AND p"),
]


@pytest.mark.parametrize(("text", "expected"), LOGIC, ids=str)
def test_logic(text, expected):
    assert simp(text) == expected


def test_a_bitwise_operator_is_read_again_once_its_operands_are_numbers():
    """`12 AND k_` waits, and both readings are offered afresh when it stops.

    Which of the two an operator is depends on what stands under it, and what
    stands there is a variable until something writes a value in.
    `NUMBER.MTH`'s `LUCAS` tests a bit of its argument with `(n AND d_) = 0`,
    where `d_` is the iteration's own variable and is a number only once an
    iterate is taken.
    """
    assert simp("VECTOR(12 AND k_, k_, 1, 4)") == "[0, 0, 0, 4]"
    assert simp("ITERATES(IF((12 AND d_) = 0, 2*d_, d_/2), d_, 1, 3)") == "[1, 2, 4, 2]"


#: A truth table: the columns as they were written, then one row per
#: assignment, the last variable changing fastest and `true` before `false`.
TRUTH_TABLES = [
    (
        "TRUTH_TABLE(p, q, p AND q, p OR q, p XOR q, p IMP q)",
        "[[p, q, p AND q, p OR q, p XOR q, p IMP q], "
        "[true, true, true, true, false, true], "
        "[true, false, false, true, true, false], "
        "[false, true, false, true, true, true], "
        "[false, false, false, false, false, true]]",
    ),
    # Nothing but variables is the table of the assignments themselves.
    ("TRUTH_TABLE(p)", "[[p], [true], [false]]"),
    (
        "TRUTH_TABLE(p, q)",
        "[[p, q], [true, true], [true, false], [false, true], [false, false]]",
    ),
    ("TRUTH_TABLE(p, NOT p)", "[[p, NOT p], [true, false], [false, true]]"),
    # A leading argument that is no variable leaves nothing to vary.
    ("TRUTH_TABLE(3, p)", "TRUTH_TABLE(3, p)"),
]


@pytest.mark.parametrize(("text", "expected"), TRUTH_TABLES, ids=str)
def test_truth_tables(text, expected):
    assert simp(text) == expected


# -- IF, and the test it is asked whether it holds -----------------------------

CONDITIONALS = [
    ("IF(2 = 2, 1, 2)", "1", None),
    ("IF(2 = 3, 1, 2)", "2", None),
    # A test that is a conjunction of relations is decided link by link, which
    # is how `NUMBER.MTH` writes `FIBONACCI`'s guard on a whole number.
    ("IF(2 >= 0 AND FLOOR(2) = 2, 1, 2)", "1", None),
    ("IF(2 >= 0 AND FLOOR(5/2) = 5/2, 1, 2)", "2", None),
    # No else clause and a false test: the value is unknown.
    ("IF(2 = 3, 1)", "?", None),
    ("IF(x > 0, 1, -1)", "1", POSITIVE_X),
    # Undecidable, with an unknown clause to fall back on.
    ("IF(x > 0, 1, -1, 0)", "0", None),
    # Undecidable, with none: the whole expression comes back.
    ("IF(x > 0, 1, -1)", "IF(x > 0, 1, -1)", None),
]


@pytest.mark.parametrize(("text", "expected", "context"), CONDITIONALS, ids=str)
def test_a_conditional_takes_the_branch_its_test_decides(text, expected, context):
    assert simp(text, context) == expected


def test_an_undecidable_conditional_keeps_its_branches_unsimplified():
    # Derive's own behaviour, and the safe one: a branch that is never taken
    # may be nonsense the guard exists to avoid.
    assert simp("IF(x > 0, (x + 1)^2 - x^2, -1)") == "IF(x > 0, (x + 1)^2 - x^2, -1)"


UNTOUCHED_CONDITIONALS = [
    # Not even the arithmetic: an arm is shown as it was written.
    ("IF(x > 0, 2 + 3, 4 + 5)", "IF(x > 0, 2 + 3, 4 + 5)"),
    # Nor the order the operands were written in, nor a product a coefficient
    # would be multiplied through.
    ("IF(x > 0, b*a, 400 + 15*(x - 40))", "IF(x > 0, b*a, 400 + 15*(x - 40))"),
    # Nor the side of the test each half was written on.
    ("IF(40 >= h, 1, 2)", "IF(40 >= h, 1, 2)"),
    # An arm of the conditional that was decided is simplified as any other
    # expression is; only the one left standing is shown as written.
    ("IF(2 = 2, 2 + 3, 4 + 5)", "5"),
    ("IF(x > 0, 2 + 3, 4 + 5, 6 + 7)", "13"),
    # A conditional inside a larger expression keeps its place in it.
    ("2*IF(x > 0, 2 + 3, 4 + 5)", "2*IF(x > 0, 2 + 3, 4 + 5)"),
    ("IF(x > 0, IF(y > 0, 1 + 1, 2), 3)", "IF(x > 0, IF(y > 0, 1 + 1, 2), 3)"),
]


@pytest.mark.parametrize(("text", "expected"), UNTOUCHED_CONDITIONALS, ids=str)
def test_an_undecidable_conditional_comes_back_as_it_was_written(text, expected):
    """10.3: Derive answers an undecidable `IF` with the `IF` the author typed.

    Simplifying an arm is what the conditional exists to prevent, and doing the
    arithmetic in one is simplifying it. What comes back is therefore the text
    that went in - which is also what makes the answer stand still, a second
    Simplify having nothing left to change.
    """
    assert simp(text) == expected
    assert simp(expected) == expected


def test_an_undecidable_conditional_keeps_the_variables_it_depends_on():
    """A held-back `IF` must not look constant to what surrounds it.

    Standing it aside under a plain symbol would hide `k` from the sum, and
    sympy would answer `n*IF(k > 0, k, 0)` - having been shown a summand that
    no longer depends on what it is summed over.
    """
    assert simp("SUM(IF(k > 0, k, 0), k, 1, n)") == "SUM(IF(k > 0, k, 0), k, 1, n)"
    assert simp("DIF(IF(x > 0, x^2, 0), x)") == "DIF(IF(x > 0, x^2, 0), x)"


def test_a_decided_conditional_simplifies_the_branch_it_took():
    assert simp("IF(2 = 2, (x + 1)^2 - x^2, -1)") == "2*x + 1"
    assert simp("1 + IF(2 = 2, 3, 4)") == "4"


def test_a_branch_a_written_out_sum_decides_is_worked_out_too():
    """A test undecidable where it was written can be decided where it lands.

    The conditional is set aside whole while the sum is written out, and each
    term brings it back with an index in it - which is a test that decides, and
    a branch nothing has yet run over. So the heads in it are offered once
    more, and the answer is a number rather than a head standing in a sum.
    NUMBER.MTH's `DISTINCT_PARTS` reaches its answer exactly this way.
    """
    assert simp("SUM(IF(k = 2, INT(x, x, 0, k), 0), k, 1, 2)") == "2"
    assert simp("SUM(IF(k = 2, SUM(j, j, 1, k), 0), k, 1, 2)") == "3"


def test_a_conditional_answer_keeps_the_test_that_qualifies_it():
    """A sum that has a closed form only where it converges answers with that
    condition attached, and the condition is exactly the kind nothing can
    decide: `ABS(x) < 1` is true for some `x` and false for others. So it
    stands, and the answer goes on saying what it is worth.
    """
    assert simp("SUM(k^2*x^k, k, 1, inf)") == (
        "IF(ABS(x) < 1, -x*(x + 1)/(x - 1)^3, SUM(k^2*x^k, k, 1, inf))"
    )


def test_a_test_holding_a_calculus_head_is_worked_out_before_it_is_asked():
    """`DIF(u, x)` is how ODE1.MTH asks whether `u` depends on `x`.

    Read as the comparison with zero that a test with no relation in it is, an
    unevaluated `Derivative` decides nothing, and a four-argument `IF` would go
    straight to its unknown clause where Derive answers. So the heads in a test
    are taken first: the answer is the derivative and not the `Derivative`.
    """
    assert simp("IF(DIF(y, x), 1, 2, 3)") == "1"
    assert simp("IF(DIF(x, x), 1, 2, 3)") == "2"
    assert simp("IF(DIF((DIF(x^2 + 1, x) - DIF(2*x*y, y))/(2*x*y), x), 1, 2, 3)") == "1"


def test_an_assignment_in_a_test_is_performed_and_read_back_in_the_branches():
    """`IF("no" = a := u, ..., a, a)` is how ODE1.MTH's `DSOLVE1` is written.

    The assignment happens where the test is evaluated, and the name it assigns
    stands for the value in the branches that come after it - which is what lets
    one method be tried once and its answer read back out. A branch that assigns
    the name again binds only within itself, since only one branch is ever taken.
    """
    assert simp('IF("no" = a_ := 2 + 3, 1, a_, a_)') == "5"
    assert simp('IF("no" = a_ := "no", a_, 2, 2)') == '"no"'
    assert simp('IF(1 = a_ := 1, IF(2 = b_ := 2 + 3, 0, b_, b_), a_, a_)') == "5"
