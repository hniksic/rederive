"""Simplify on relations, logic and conditionals.

A relation is decided when arithmetic or declared domains settle it, and left alone
otherwise: `x^2 >= 0` is `true`, but `x < x + 1` stays for complex `x` since only reals
are ordered. Two vectors are equal when their elements are, so `[x, y] = [1, 2]` is
`x = 1 AND y = 2`. Logic is simplified algebraically (`p AND (q OR p)` is `p`). A
conditional takes the branch its test decides and is returned as written otherwise, so
`IF(x > 0, 1, -1)` is `1` only when `x` is declared positive. Cases are written inline,
as in `test_simplify.py`, so a failure shows the mathematics.
"""

from __future__ import annotations

import pytest
from simplifying import COMPLEX_X, POSITIVE_X, declared, declaring, parse, simp

from rederive.engine.computing import Context, Definition
from rederive.model.session import Session


# -- relations and logic ------------------------------------------------------


def element(text, context=None):
    """What `text` simplifies to as the element of a vector, where an equation is
    decided like any relation; the whole expression being an equation keeps its form."""
    answer = simp(f"[{text}]", context)
    assert answer.startswith("[") and answer.endswith("]"), answer
    return answer[1:-1]


def test_a_relation_that_decides_is_decided_and_one_that_does_not_stands():
    """A definite relation simplifies to `true` or `false`.

    The result is three-valued: `3*x = x^2 + c` is undecided, so it stays a relation
    with its sides simplified. That keeps conditional tests intact and leaves soLve
    something to solve."""
    assert simp("2 < 3") == "true"
    assert element("2 = 2") == "true"
    assert element("1 = 2") == "false"
    assert element("x = x") == "true"
    assert simp("2 <= 2") == "true"
    assert simp("x < x + 1") == "true"
    assert simp("x + 2*x = c + x*x") == "3*x = x^2 + c"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A top-level equation keeps its form, so that it can be worked on by hand.
        ("2 = 2", "2 = 2"),
        ("x = x + 1", "x = x + 1"),
        ("e := 2 = 2", "e := 2 = 2"),
        ("(x = x + 1) - x", "0 = 1"),
        ("(2 = 2) + (3 = 3)", "5 = 5"),
        ("EXP(LN(x) = 5)", "x = #e^5"),
        ("a = a - x", "a = a - x"),
        ("3 < x", "3 < x"),
        # Anything else is decided where it can be: an inequation or inequality, an
        # element, an argument, an operand of a logical operator, an `IF` test, and an
        # equation a function returns.
        ("x /= x", "false"),
        ("x + 1 < x + SQRT(2)", "true"),
        ("[x = x + 1, x + 1 = x + 1]", "[false, true]"),
        ("VECTOR(x^2 - 8*x = -15, x, 1, 5)", "[false, false, true, false, true]"),
        ("VECTOR(x^2 - 8*x < -15, x, 1, 5)", "[false, false, false, true, false]"),
        ("2 = 2 AND x > 0", "x > 0"),
        ("IF(2 = 2, 1, 0)", "1"),
        ("SUBST(a*x + b = 0, x, -b/a)", "true"),
        ("EXPAND((x + 1)^2 = x^2 + 2*x + 1)", "true"),
        ("FACTOR(a = a - x)", "a = a - x"),
        ("APPROX(2 = 2)", "true"),
        # An equation of vectors is the conjunction of the element equations, unless
        # its sides are the same, which stays whatever their kind, as `2 = 2` does.
        ("[1, 2] = [1, 2]", "[1, 2] = [1, 2]"),
        ("[[-3], [-3]] = [[-3], [-3]]", "[[-3], [-3]] = [[-3], [-3]]"),
        ('"a" = "a"', '"a" = "a"'),
        ("[1, 2] = [1, 3]", "false"),
        ("[x, 2] = [1, 2]", "x = 1"),
        # The sides of an equation are taken apart, never decided.
        ("LHS(2*x + 3 = 5)", "2*x + 3"),
        ("RHS(x = x + 1)", "x + 1"),
        ("RHS(2*x + 3 < 5)", "5"),
        # A relation in arithmetic is decided and stays a truth value there.
        ("(2 < 3) - 2", "true - 2"),
        ("(x^2 >= 0)*5", "5*true"),
    ],
    ids=str,
)
def test_a_top_level_equation_keeps_its_form(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Set literals are compared by their elements, as simplified expressions.
        ("{x} = {1}", "false"),
        ("{x} = {y}", "false"),
        ("[{x} = {1}]", "[false]"),
        ("IF({x} = {1}, 1, 2)", "2"),
        ("{x} SUBSET {1}", "false"),
        ("{1, 2} = {2, 1, 2}", "{1, 2} = {1, 2}"),
        ("{1} = {2}", "false"),
        ("{1, 2} = {x}", "false"),
        ("IF({1, 2} = {x}, 1, 2)", "2"),
        ("{3} SUBSET {1, 2}", "false"),
        ("{3} SUBSET {1, x}", "false"),
        # A set that isn't a literal has unknown elements.
        ("[{x} = s]", "[{x} = s]"),
        # A statement equals true where it holds and false where it fails.
        ("(x > 1) = true", "x > 1"),
        ("(x > 1) = false", "x <= 1"),
        ("(x = 1) = true", "x = 1"),
        ("true = (x = 1)", "x = 1"),
        ("(x > 1) /= true", "x <= 1"),
        ("[(x > 1) = false]", "[x <= 1]"),
        ("(2 > 1) = true", "true"),
        # A variable may be a number, so its equation with a truth value stands.
        ("a = true", "a = true"),
    ],
    ids=str,
)
def test_an_equation_of_sets_or_truth_values_is_decided_only_where_it_holds(
    text, expected
):
    assert simp(text) == expected


async def test_an_equation_assigned_or_referred_to_keeps_its_form_at_top_level():
    session = Session()
    for line in ("e := x = x + 1", "EVAL1(t) := t"):
        await session.simplify(line)
    answers = [
        (await session.simplify(line)).text
        for line in ("e", "e - x", "[e]", "EVAL1(2 = 2)", "EVAL1(a = a - x)")
    ]
    assert answers == ["x = x + 1", "0 = 1", "[false]", "true", "a = a - x"]


def test_a_word_compared_with_a_relation_is_left_undecided_at_once():
    # ODE.MTH's `IF(fail = (local :== ...), ...)` gives `local` in both other cases;
    # simplifying an equation of integrals to decide it takes minutes.
    text = "IF(fail = (INT(TAN(x - y + 1)/ABS(TAN(x - y + 1)), x) = c), 1, 2, 2)"
    assert simp(text) == "2"


def test_a_word_compared_with_an_expression_without_it_is_left_undecided():
    # ODE.MTH's `IF(fail = (local :== ...), ...)` with a method's long answer: it holds
    # for one value of `fail`, so it is undecided whatever the answer simplifies to.
    assert simp("IF(fail = SQRT((a^2 - 2*a*b + b^2)/(x - r)^2), 1, 2, 3)") == "3"


def test_a_relation_that_holds_for_every_value_is_decided():
    """Each holds for all values of its variables, the last two by their declared
    intervals."""
    assert simp("x^2 >= 0") == "true"
    positive = declared("p :epsilon Real (0, inf)", "q :epsilon Real (0, inf)")
    assert simp("p + q > 0", positive) == "true"
    assert simp("ABS(g) < 1", declared("g :epsilon Real (0, 1)")) == "true"


def test_an_ordering_of_multiples_of_one_expression_is_a_comparison_with_zero():
    """`c*u > d*u` is `u > 0` for `c > d` and `u < 0` for `c < d`, also as the test of
    a conditional answer."""
    assert simp("p > -p") == "p > 0"
    assert simp("2*(x - y^3) <= 3*(x - y^3)") == "x - y^3 >= 0"
    assert simp("-SIN(x) >= SIN(x)/2") == "SIN(x) <= 0"
    assert simp("INT(ABS(x)*COS(pi*x/p), x, -p, p)") == (
        "IF(p > 0, -4*p^2/pi^2, 4*p^2/pi^2)"
    )


def test_an_equation_of_two_vectors_is_the_conjunction_of_their_elements():
    """Same-length vectors are equal when each pair of elements is, so the equation is
    the conjunction of the element equations, each judged separately, and a matrix is
    compared element by element. The inequation is the disjunction of the element
    inequations. Vectors of different lengths, at any depth, are different values.
    Vectors are not ordered, so an ordering of vectors stands."""
    assert simp("[x, y] = [1, 2]") == "x = 1 AND y = 2"
    assert simp("[[x], [y]] = [[1], [2]]") == "x = 1 AND y = 2"
    assert simp("[x, y] < [1, 2]") == "[x, y] < [1, 2]"
    assert simp("[1, 2, 3] < [2, 3, 4]") == "[1, 2, 3] < [2, 3, 4]"
    spread = simp("[[a, b], [c, d]] = [[1, 2], [3, 4]]")
    assert spread == "a = 1 AND b = 2 AND c = 3 AND d = 4"
    assert simp("[a, b] = [c, d]") == "a = c AND b = d"
    assert simp("[a, b] /= [c, d]") == "a /= c OR b /= d"
    assert simp("[1, 2] = [1, 2, 3]") == "false"
    assert simp("[x, y] = [1, 2, 3]") == "false"
    assert simp("[[x, y]] = [[1, 2, 3]]") == "false"
    assert simp("[[] = []]") == "[true]"
    assert simp("[x] = []") == "false"


def test_a_vector_is_no_number_string_or_truth_value():
    """A vector compared with a number, a string or a truth value is different from it;
    compared with a variable, it stands."""
    assert simp("[1, 2, 2] = 7") == "false"
    assert simp("[1, 2] /= 3") == "true"
    assert simp("[[1, 2, 2] = 7]") == "[false]"
    assert simp('[1] = "a"') == "false"
    assert simp("[1] = true") == "false"
    assert simp("[a, b] = x") == "[a, b] = x"
    # A relation as an element is compared with nothing.
    assert simp("[x = 1, y = 2] = [1, 2]") == "[x = 1, y = 2] = [1, 2]"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Vectors are equal when every element pair is, and unequal when any pair
        # differs; when the elements settle it, the answer is a single truth value. At
        # top level, an equation whose sides are the same stays, as `2 = 2` does.
        ("[1, 2] = [1, 2]", "[1, 2] = [1, 2]"),
        ("[[1, 2] = [1, 2]]", "[true]"),
        ("[1, 2] = [1, 3]", "false"),
        ("[[[1, 2], [3, 4]] = [[1, 2], [3, 4]]]", "[true]"),
        ("[[1, 2], [3, 4]] = [[1, 2], [3, 5]]", "false"),
        ("[SIN(x)^2 + COS(x)^2, 2] = [1, 2]", "[1, 2] = [1, 2]"),
        ("[[SIN(x)^2 + COS(x)^2, 2] = [1, 2]]", "[true]"),
        ("[[true, false] = [true, false]]", "[true]"),
        # One differing pair is enough.
        ("[1, 2] = [x, 3]", "false"),
        ("[1, x] = [2, x]", "false"),
        # If unsettled, the equation is the conjunction of the open element equations.
        ("[1, 2] = [x, 2]", "1 = x"),
        ("[1, 2] /= [1, 3]", "true"),
        ("[1, 2] /= [1, 2]", "false"),
        ("[1, 2] /= [x, 3]", "true"),
        ("[1, 2] /= [x, 2]", "1 /= x"),
        ("[x, y] /= [1, 2]", "x /= 1 OR y /= 2"),
        ("[1, 2] /= [1, 2, 3]", "true"),
        ("IF([1, 2] = [1, 2], a, b)", "a"),
        ("IF([1, 2] = [1, 3], a, b)", "b"),
    ],
    ids=str,
)
def test_an_equation_of_two_matrices_is_one_truth_value(text, expected):
    assert simp(text) == expected


def test_the_system_of_section_eight_five_is_solved_by_inverting_its_matrix():
    """The manual's use of the rule above, in exact arithmetic; the quotients round to
    its 2.91059, 0.175496 and 1.58278."""
    solved = simp(
        "[[x], [y], [z]] = [[5, 3, -7], [2, -8, 1], [-1, 9, 4]]^-1 . [[4], [6], [5]]"
    )
    assert solved == "x = 879/302 AND y = 53/302 AND z = 239/151"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Strings are equal exactly when their texts are, case counting.
        ('["Hallo" = "Hallo"]', "[true]"),
        ('"Hallo" = "hallo"', "false"),
        ('IF("Hallo" = "hallo", 1, 2)', "2"),
        ('"a" /= "b"', "true"),
        # A string is no vector, set or truth value. Beside a number it is an unknown,
        # as in arithmetic.
        ('"3" = 3', '"3" = 3'),
        ('"a" = [1]', "false"),
        ('"a" = {1}', "false"),
        ('"a" /= true', "true"),
        # Beside a variable it stands, and strings are not ordered.
        ('"a" = x', '"a" = x'),
        ('"a" < "b"', '"a" < "b"'),
        # The unknown equals itself and no number, string or vector.
        ("[? = ?]", "[true]"),
        ("? = 3", "false"),
        ('? /= "a"', "true"),
        ("? = x", "? = x"),
        ('IF(3 = "+-"3, 1, 2)', "1"),
    ],
    ids=str,
)
def test_strings_compare_by_their_text(text, expected):
    assert simp(text) == expected


def test_only_the_reals_are_ordered_so_a_complex_variable_decides_nothing():
    """`x < x + 1` holds for reals, but `<` is meaningless on complex numbers. Equality
    is meaningful in every domain, so it is still decided."""
    assert simp("x < x + 1", COMPLEX_X) == "x < x + 1"
    assert element("x = x", COMPLEX_X) == "true"


def test_an_equation_is_answered_by_the_arithmetic_and_not_by_the_domains():
    """The original never decides an equation from declared domains. `x = #i`, where a
    quadratic with negative discriminant lands, stands whether `x` is declared real,
    complex or not at all; undeclared names being real matters for `SQRT(x^2)` ->
    `ABS(x)`, but not here. `ABS(x) + 1 = 0` has no real solution and stands too: an
    equation asks about a value rather than making a claim to refute.

    An equation is decided only when the difference of its sides is a number: zero
    means true, a nonzero real means false. A non-real difference decides nothing, as
    in the original: `#i = 1` and `2*#i = #i` stand."""
    assert element("x = #i") == "x = #i"
    assert element("x = SQRT(-1)") == "x = #i"
    assert element("x = #i", declared("x :epsilon Real")) == "x = #i"
    assert element("x = #i", COMPLEX_X) == "x = #i"
    assert element("x + #i = 0") == "x + #i = 0"
    assert element("x^2 + 1 = 0") == "x^2 + 1 = 0"
    assert element("x /= #i") == "x /= #i"
    assert element("ABS(x) + 1 = 0") == "ABS(x) + 1 = 0"
    assert element("#e^x = 0") == "#e^x = 0"
    assert element("n = 1/2", declared("n :epsilon Integer")) == "n = 1/2"
    assert element("p = -1", declared("p :epsilon Real (0, inf)")) == "p = -1"
    assert element("y = 5", declared("y :epsilon Real (0, 1)")) == "y = 5"
    # Arithmetic decides these; an assigned value counts as arithmetic.
    assert element("x + 1 = x") == "false"
    assert element("x /= x") == "false"
    assert element("1 = 2") == "false"
    assert element("SQRT(-4) = 2*#i") == "true"
    assert element("2*x = x + x") == "true"
    assert element("v = 5", Context(assignments={"v": parse("7")})) == "false"
    # A non-real difference leaves the equation alone.
    assert element("#i = 1") == "#i = 1"
    assert element("2*#i = #i") == "2*#i = #i"
    assert element("x + #i = x") == "x + #i = x"
    # Identical sides are equal whatever they are; that alone decides cases where no
    # difference can be taken.
    assert element("INF = INF") == "true"
    assert element("-INF = INF") == "false"
    assert element("true = true") == "true"
    assert element('"no" = "no"') == "true"
    assert element('"no" = "yes"') == "false"


def test_a_quadratic_whose_discriminant_turns_negative_is_answered_in_roots():
    """The manual's quadratic with numbers substituted, where deciding equations from
    domains would lose the user's answer: `a = 10`, `b = 1.3333`, `c = 30` make the
    discriminant negative, and the root is what the original prints."""
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
    """Two bounds on one variable are written as a chain, which the grammar can read
    back. Each bound's strictness is independent, so half-open ranges chain too."""
    assert simp("x > -2 AND x < 2") == "-2 < x < 2"
    assert simp("x >= -2 AND x < 2") == "-2 <= x < 2"


def test_a_conjunction_it_cannot_solve_keeps_its_shape():
    assert simp("x >= 1 OR x <= -1") == "x >= 1 OR x <= -1"
    # With two unknowns nothing is solved, but the pair is still a range around `y`,
    # written as a chain.
    assert simp("x < y AND y < 1") == "x < y < 1"


LOGIC = [
    ("NOT NOT p", "p"),
    ("NOT p", "NOT p"),
    # The algebra: absorption, operators rewritten in terms of NOT/AND/OR, a variable
    # with its negation, and factoring out a common conjunct. A product of sums stays
    # folded, since Derive doesn't multiply it out into a sum of products.
    ("p AND (q OR p)", "p"),
    ("p OR NOT p", "true"),
    ("p AND NOT p", "false"),
    ("p IMP q", "NOT p OR q"),
    ("p XOR q", "(NOT p AND q) OR (p AND NOT q)"),
    ("p IFF q", "(NOT p AND NOT q) OR (p AND q)"),
    ("true IFF false", "false"),
    ("p IFF p", "true"),
    ("NOT (p AND q)", "NOT p OR NOT q"),
    ("p AND (q OR r)", "p AND (q OR r)"),
    ("p AND r OR p AND q", "p AND (q OR r)"),
    # Operands are sorted by the order list, each term by its leading variable, with a
    # negated literal first on a tie.
    ("s AND r OR q AND p", "(p AND q) OR (r AND s)"),
    ("c OR b AND a", "(a AND b) OR c"),
    ("p AND q OR NOT p AND r", "(NOT p AND r) OR (p AND q)"),
    # With six variables the algebra isn't worth it: both normal forms come from a
    # truth table, and here they are longer than the input.
    ("a XOR b XOR c XOR d XOR e XOR f", "a XOR b XOR c XOR d XOR e XOR f"),
    # Boolean on booleans, bitwise on integers.
    ("3 OR 5", "7"),
    ("NOT 5", "-6"),
    ("12 AND 10", "8"),
    ("3 XOR 5", "6"),
    ("3 AND p", "3 AND p"),
    # `IFF` on integers is the complement of `XOR`.
    ("5 IFF 3", "-7"),
    # A number beside a statement is no truth value.
    ("p IFF 3", "p IFF 3"),
    ("p IMP 3", "p IMP 3"),
    ("p XOR 3", "p XOR 3"),
    # A truth value is not a number, so arithmetic on it is left alone, treating it
    # like an unknown.
    ("x*PRIME(4)", "false*x"),
    ("PRIME(5) + 1", "true + 1"),
    ("true + true", "2*true"),
    ("-true", "-true"),
    ("true/2", "true/2"),
    ("x + (p AND q)", "(p AND q) + x"),
    ("x*(a < b)", "x*(a < b)"),
]


@pytest.mark.parametrize(("text", "expected"), LOGIC, ids=str)
def test_logic(text, expected):
    assert simp(text) == expected


def test_a_bitwise_operator_is_read_again_once_its_operands_are_numbers():
    """`12 AND k_` waits, and is reconsidered as bitwise once `k_` gets a value.

    Whether an operator is logical or bitwise depends on its operands, and a variable
    has no value until one is substituted. `NUMBER.MTH`'s `LUCAS` tests a bit with
    `(n AND d_) = 0`, where `d_` is the iteration variable.
    """
    assert simp("VECTOR(12 AND k_, k_, 1, 4)") == "[0, 0, 0, 4]"
    assert simp("ITERATES(IF((12 AND d_) = 0, 2*d_, d_/2), d_, 1, 3)") == "[1, 2, 4, 2]"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # `n AND d` of variables may be a number once they are integers, so comparing
        # it with a number decides nothing, wherever the relation is built.
        ("IF((n AND d) = 0, 1, 2)", "IF((d AND n) = 0, 1, 2)"),
        ("IF((n OR d) /= 3, 1, 2)", "IF((d OR n) /= 3, 1, 2)"),
        ("IF((NOT n) = 0, 1, 2)", "IF((NOT n) = 0, 1, 2)"),
        ("[SUBST(a = 0, a, n AND d)]", "[(d AND n) = 0]"),
        ("IF((6 AND 3) = 2, 1, 2)", "1"),
        # A conjunction of relations is a truth value, never a number.
        ("IF((x > 1 AND y < 2) = 0, 1, 2)", "2"),
        ("IF((p AND q) = (q AND p), 1, 2)", "1"),
    ],
    ids=str,
)
def test_a_logical_operator_on_variables_may_be_bitwise(text, expected):
    assert simp(text) == expected


async def test_a_bitwise_test_waits_for_its_operands_in_a_function():
    session = Session()
    session.author("F(n, d) := IF((n AND d) = 0, 1, 2)")
    for call, expected in [("F(6, 3)", "2"), ("F(4, 3)", "1")]:
        assert (await session.simplify(call)).text == expected
    session.author("G(a) := IF(a = 0, 1, 2)")
    assert (await session.simplify("G(n AND d)")).text == "IF(0 = (d AND n), 1, 2)"


def test_variables_declared_vector_may_be_equal():
    """Sympy takes a matrix symbol for unequal to any other matrix; a variable declared
    Vector may hold any vector, so the test stands. A vector is still never a number."""
    context = declared("v :epsilon Vector", "w :epsilon Vector")
    assert simp("IF(v = w, 1, 2)", context) == "IF(v = w, 1, 2)"
    assert simp("IF(v /= [1, 2], 1, 2)", context) == "IF(v /= [1, 2], 1, 2)"
    assert simp("IF(v = v, 1, 2)", context) == "1"
    assert simp("IF(v = 0, 1, 2)", context) == "2"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # An ordering of vectors as a test holds when it holds between each pair of
        # elements, as Groebner basis programs test that one monomial divides another.
        ("IF([1, 2] <= [2, 2], 1, 2)", "1"),
        ("IF([1, 3] <= [2, 2], 1, 2)", "2"),
        ("IF([4, 1, 0] >= [2, 1, 0], 1, 2)", "1"),
        ("IF([a, 4] <= [2, 3], 1, 2)", "2"),
        ("IF([a, 2] <= [2, 3], 1, 2)", "IF([a, 2] <= [2, 3], 1, 2)"),
        ("IF([1, 2] <= [2, 3, 4], 1, 2)", "IF([1, 2] <= [2, 3, 4], 1, 2)"),
    ],
    ids=str,
)
def test_an_ordering_of_vectors_as_a_test_compares_their_elements(text, expected):
    assert simp(text) == expected


#: A truth table: a header row of the columns as written, then one row per assignment,
#: the last variable changing fastest and `true` before `false`.
TRUTH_TABLES = [
    (
        "TRUTH_TABLE(p, q, p AND q, p OR q, p XOR q, p IMP q)",
        "[[p, q, p AND q, p OR q, p XOR q, p IMP q], "
        "[true, true, true, true, false, true], "
        "[true, false, false, true, true, false], "
        "[false, true, false, true, true, true], "
        "[false, false, false, false, false, true]]",
    ),
    (
        "TRUTH_TABLE(p, q, p IFF q)",
        "[[p, q, p IFF q], [true, true, true], [true, false, false], "
        "[false, true, false], [false, false, true]]",
    ),
    # With only variables, the table is just the assignments.
    ("TRUTH_TABLE(p)", "[[p], [true], [false]]"),
    (
        "TRUTH_TABLE(p, q)",
        "[[p, q], [true, true], [true, false], [false, true], [false, false]]",
    ),
    ("TRUTH_TABLE(p, NOT p)", "[[p, NOT p], [true, false], [false, true]]"),
    # A leading argument that isn't a variable leaves nothing to vary.
    ("TRUTH_TABLE(3, p)", "TRUTH_TABLE(3, p)"),
]


@pytest.mark.parametrize(("text", "expected"), TRUTH_TABLES, ids=str)
def test_truth_tables(text, expected):
    assert simp(text) == expected


# -- IF and its test ----------------------------------------------------------

CONDITIONALS = [
    ("IF(2 = 2, 1, 2)", "1", None),
    ("IF(2 = 3, 1, 2)", "2", None),
    # Values all the same are the value, whatever the test.
    ("IF(? >= ?, ?, ?)", "?", None),
    ("IF(? >= ?, a, a, a)", "a", None),
    # A conjunction of relations is decided term by term, as in `NUMBER.MTH`'s
    # whole-number guard in `FIBONACCI`.
    ("IF(2 >= 0 AND FLOOR(2) = 2, 1, 2)", "1", None),
    ("IF(2 >= 0 AND FLOOR(5/2) = 5/2, 1, 2)", "2", None),
    # No else clause and a false test: the value is unknown.
    ("IF(2 = 3, 1)", "?", None),
    ("IF(x > 0, 1, -1)", "1", POSITIVE_X),
    # Undecidable, with an unknown clause to fall back on.
    ("IF(x > 0, 1, -1, 0)", "0", None),
    # A truth table's heading is a formula, true for some values and not others.
    ("IF((TRUTH_TABLE(x, y, x AND y)) SUB 1 SUB 3 = true, 1, 0, 2)", "2", None),
    # Beside a truth value, a number is a test, `u = 0`, as an `IF` test is.
    ("IF(false OR 0, 1, 2)", "1", None),
    ("false OR 2", "false", None),
    ("IF(20 = 0 OR 2, 1, 2)", "2", None),
    # Undecidable, with no fallback: the whole expression comes back.
    ("IF(x > 0, 1, -1)", "IF(x > 0, 1, -1)", None),
    # A test is decided as Simplify would: an empty range is false, and sides that
    # simplify to the same value are equal.
    ("IF(x < 1 AND x > 3, a, b)", "b", None),
    ("IF((x^2 + x^2*y^2)/(x^2*y) = (1 + y^2)/y, 1, 2, 3)", "1", None),
    ("IF(DIF((x + x*y)/x, x), 1, 2, 3)", "1", None),
    # Complex numbers are compared multiplied out.
    ("IF((2 + 3*#i)^2 = 1, 1, 2)", "2", None),
    ("IF((2 + 3*#i)^2 = -5 + 12*#i, 1, 2)", "1", None),
]


@pytest.mark.parametrize(("text", "expected", "context"), CONDITIONALS, ids=str)
def test_a_conditional_takes_the_branch_its_test_decides(text, expected, context):
    assert simp(text, context) == expected


def test_an_undecidable_conditional_keeps_its_branches_unsimplified():
    # As in Derive, and safer: an untaken branch may be nonsense the guard exists to
    # avoid.
    assert simp("IF(x > 0, (x + 1)^2 - x^2, -1)") == "IF(x > 0, (x + 1)^2 - x^2, -1)"


UNTOUCHED_CONDITIONALS = [
    # Not even arithmetic: an arm is shown as written.
    ("IF(x > 0, 2 + 3, 4 + 5)", "IF(x > 0, 2 + 3, 4 + 5)"),
    # Nor is operand order changed or a coefficient distributed.
    ("IF(x > 0, b*a, 400 + 15*(x - 40))", "IF(x > 0, b*a, 400 + 15*(x - 40))"),
    # Nor are the sides of the test swapped.
    ("IF(40 >= h, 1, 2)", "IF(40 >= h, 1, 2)"),
    # A decided conditional's chosen arm is simplified normally; only an undecided one
    # is shown as written.
    ("IF(2 = 2, 2 + 3, 4 + 5)", "5"),
    ("IF(x > 0, 2 + 3, 4 + 5, 6 + 7)", "13"),
    # A conditional inside a larger expression stays in place.
    ("2*IF(x > 0, 2 + 3, 4 + 5)", "2*IF(x > 0, 2 + 3, 4 + 5)"),
    ("IF(x > 0, IF(y > 0, 1 + 1, 2), 3)", "IF(x > 0, IF(y > 0, 1 + 1, 2), 3)"),
]


@pytest.mark.parametrize(("text", "expected"), UNTOUCHED_CONDITIONALS, ids=str)
def test_an_undecidable_conditional_comes_back_as_it_was_written(text, expected):
    """10.3: Derive answers an undecidable `IF` with the `IF` exactly as typed.

    The conditional exists to keep its arms from being simplified, even arithmetically.
    So the input comes back unchanged, which also makes it stable under a second
    Simplify.
    """
    assert simp(text) == expected
    assert simp(expected) == expected


def test_an_undecidable_conditional_keeps_the_variables_it_depends_on():
    """A held-back `IF` must not look constant to its surroundings.

    Replacing it with a plain symbol would hide `k` from the sum, and sympy would
    answer `n*IF(k > 0, k, 0)`, treating the summand as independent of `k`.
    """
    assert simp("SUM(IF(k > 0, k, 0), k, 1, n)") == "SUM(IF(k > 0, k, 0), k, 1, n)"
    assert simp("DIF(IF(x > 0, x^2, 0), x)") == "IF(x > 0, 2*x, 0)"


def test_a_decided_conditional_simplifies_the_branch_it_took():
    assert simp("IF(2 = 2, (x + 1)^2 - x^2, -1)") == "2*x + 1"
    assert simp("1 + IF(2 = 2, 3, 4)") == "4"


def test_a_branch_a_written_out_sum_decides_is_worked_out_too():
    """A test undecidable as written can be decided after substitution.

    The conditional is held whole while the sum is expanded, and each term brings it
    back with a numeric index, so the test decides and the chosen branch has not been
    evaluated yet. Its heads are evaluated then, giving a number rather than an
    unevaluated head. NUMBER.MTH's `DISTINCT_PARTS` relies on this.
    """
    assert simp("SUM(IF(k = 2, INT(x, x, 0, k), 0), k, 1, 2)") == "2"
    assert simp("SUM(IF(k = 2, SUM(j, j, 1, k), 0), k, 1, 2)") == "3"


def test_a_conditional_answer_keeps_the_test_that_qualifies_it():
    """A sum with a closed form only where it converges answers with that condition
    attached. `ABS(x) < 1` is true for some `x` and false for others, so it stays.
    """
    assert simp("SUM(k^2*x^k, k, 1, inf)") == (
        "IF(ABS(x) < 1, -x*(x + 1)/(x - 1)^3, SUM(k^2*x^k, k, 1, inf))"
    )


def test_each_case_of_a_conditional_answer_is_simplified_where_it_applies():
    """A case is evaluated only where its test holds and the tests before it fail."""
    # An equation puts its value for its side.
    assert simp("SUM(SIN((2*k - 1)*x), k, 1, n)") == (
        "IF(SIN(x) = 0, 0, SIN(n*x)^2/SIN(x))"
    )
    # A range settles a power of zero, and the last case is at the point `a = -1`.
    assert simp("INT(ABS(a)*x^a, x, 0, 1)") == (
        "IF(a >= 0, a/(a + 1), IF(a /= -1, a*(0^(a + 1) - 1)/(a + 1), inf))"
    )
    # Past `n = 0`, `pi*n/p /= 0` holds, so the case after it is never reached. The
    # first test was `n = 0 OR n = 0 AND pi*n/p /= 0`.
    assert simp("INT(x*SIN(n*pi*x/p), x, -p, p)") == (
        "IF(n = 0, 0, 2*p^2*(SIN(pi*n)/(pi*n) - COS(pi*n))/(pi*n))"
    )
    # Undefined at `a = 0`, the last case is not given a value there.
    assert simp("INT(1/SQRT(a^2 - x^2), x, 0, ABS(a))") == (
        "IF(a /= 0, pi/2, #i*LN(2*#i*ABS(a)) - #i*LN(-2*ABS(a)))"
    )


def test_a_test_holding_a_calculus_head_is_worked_out_before_it_is_asked():
    """`DIF(u, x)` is how ODE1.MTH asks whether `u` depends on `x`.

    A test without a relation is compared with zero, and an unevaluated `Derivative`
    decides nothing, so a four-argument `IF` would take its unknown clause where Derive
    answers. So heads in a test are evaluated first.
    """
    assert simp("IF(DIF(y, x), 1, 2, 3)") == "1"
    assert simp("IF(DIF(x, x), 1, 2, 3)") == "2"
    assert simp("IF(DIF((DIF(x^2 + 1, x) - DIF(2*x*y, y))/(2*x*y), x), 1, 2, 3)") == "1"


def test_an_assignment_in_a_test_is_performed_and_read_back_in_the_branches():
    """`IF("no" = a := u, ..., a, a)` is how ODE1.MTH's `DSOLVE1` is written.

    The assignment happens when the test is evaluated, and the name holds the value in
    the branches, so a method is tried once and its answer reused. A branch that
    reassigns the name binds only within itself, since only one branch is taken.
    """
    assert simp('IF("no" = a_ := 2 + 3, 1, a_, a_)') == "5"
    assert simp('IF("no" = a_ := "no", a_, 2, 2)') == '"no"'
    assert simp('IF(1 = a_ := 1, IF(2 = b_ := 2 + 3, 0, b_, b_), a_, a_)') == "5"


def test_a_name_assigned_in_a_test_is_assigned_again_in_a_nested_test():
    """ODE.MTH chains its methods as `IF(fail = (local :== M1(w)), IF(fail = (local
    :== M2(w)), ..., local, local), local, local)` inside its functions. The second
    assignment is to `local` again, not to `fail`, the value the first gave it.
    """
    body = parse("IF(f = (a :== f), IF(f = (a :== w), 7, a, a))")
    context = Context(functions={"G": Definition(("w",), body)})
    assert simp("G(5)", context) == "5"
    assert simp("f", context) == "f"


#: Truth values as Derive computes with them, each checked in Derive.
TRUTHS = [
    # A `false` operand decides a conjunction and a `true` one a disjunction, whatever
    # the others are.
    ("1/2 AND false", "false"),
    ("1/2 OR true", "true"),
    # Also where the deciding operand is decided only once simplified, as in FUN.MTH's
    # F21 at `c = 3/2`.
    ("MOD(3/2 - 1/2) = 0 OR MOD(3/2) AND 3/2 >= 2", "true"),
    ("MOD(3/2) AND 3/2 >= 2", "false"),
    # A test that isn't a relation means `test = 0`, and a vector is zero when each
    # element is.
    ("IF([0, 0], 1, 2, 3)", "1"),
    ("IF([0, 1], 1, 2, 3)", "2"),
    ("IF([0, x], 1, 2, 3)", "3"),
    # Vectors of different lengths are unequal, whatever their elements.
    ("IF([a] = [], 1, 2, 3)", "2"),
    ("IF([a] /= [b, c], 1, 2, 3)", "1"),
    # Truth values add up like unknowns. In a sum the index is substituted before the
    # test is decided, so `NUMBER(k)` is true at each value.
    ("SUM(PRIME(k), k, 1, 3)", "false + 2*true"),
    ("SUM(NUMBER(k), k, 1, 3)", "3*true"),
    ("PRODUCT(PRIME(k), k, 1, 3)", "false*true^2"),
    # A relation of numbers in a vector is decided like any other.
    ("[1 = 1]", "[true]"),
]


@pytest.mark.parametrize(("text", "expected"), TRUTHS, ids=str)
def test_truth_values(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[(x + 1)^2 = x^2 + 2*x + 1]", "[true]"),
        ("IF((x + 1)^2 = x^2 + 2*x + 1, 1, 2)", "1"),
        ("[(a + b)^3 = a^3 + 3*a^2*b + 3*a*b^2 + b^3]", "[true]"),
        ("[(SQRT(x) + 1)^2 = x + 2*SQRT(x) + 1]", "[true]"),
        # Equal where both sides are defined, as `[(x^2 - 1)/(x - 1) = x + 1]` is.
        ("[1/(x + 1) + 1/(x - 1) = 2*x/(x^2 - 1)]", "[true]"),
        ("[(x + 1)^2 /= x^2 + 2*x + 1]", "[false]"),
        # Sides that differ by a nonzero number are never equal.
        ("[(x + 1)^2 = x^2 + 2*x + 2]", "[false]"),
        # A difference still in `x` decides nothing: the equation holds for some `x`.
        ("[(x + 1)^2 = x^2 + 1]", "[(x + 1)^2 = x^2 + 1]"),
        ("IF((x + 1)^2 = x^2 + 1, 1, 2)", "IF((x + 1)^2 = x^2 + 1, 1, 2)"),
        # A top-level equation is not decided, so it can still be worked on.
        ("(x + 1)^2 = x^2 + 2*x + 1", "(x + 1)^2 = x^2 + 2*x + 1"),
    ],
    ids=str,
)
def test_an_equation_of_equal_expressions_in_different_forms_is_true(text, expected):
    assert simp(text) == expected


def test_a_predicate_awaiting_its_argument_is_a_test_as_written():
    """`IF(PRIME(m), 2, 3)` stays as typed, reading back as itself; comparing the
    predicate with zero would ask a different question.
    """
    context = Context(functions={"H": Definition(("n",), parse("IF(PRIME(n), 2, 3)"))})
    assert simp("H(m)", context) == "IF(PRIME(m), 2, 3)"
    assert simp("IF(PRIME(m), 2, 3)") == "IF(PRIME(m), 2, 3)"
    assert simp("H(5)", context) == "2"
    assert simp("H(4)", context) == "3"


def test_the_last_iterate_of_a_system_may_be_a_conditional():
    """The test at the last step is undecided, so the `IF` is the last iterate, and
    nothing needs to be taken apart for a further step.
    """
    assert simp(
        "ITERATES(IF(DIF(ELEMENT([x, y^2], n), y) = 0, [p + 10, n + 1], [p + 1, n + 1]),"
        " [p, n], [0, 1], 2)"
    ) == "[[0, 1], [10, 2], IF(2*y = 0, [20, 3], [11, 3])]"


def test_an_undecidable_conditional_shows_its_test_worked_out_and_its_arms_as_written():
    # As Derive 4.11 shows it (D).
    assert simp("IF(DIF(x^2, x) = 0, 1 + 1, 2 + 2)") == "IF(2*x = 0, 1 + 1, 2 + 2)"
    assert simp("IF(x = 0, 0, INT(t*x, t, 0, 1))") == "IF(x = 0, 0, INT(t*x, t, 0, 1))"
    body = parse("IF(x=0,0,INT(t*x,t,0,1))")
    context = Context(functions={"F": Definition(("x",), body)})
    assert simp("F(y) + 1 + 1", context) == "IF(y = 0, 0, INT(t*y, t, 0, 1)) + 2"


# -- Boolean simplification ---------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("p IFF p", "true"),
        ("p IFF true", "p"),
        ("true IFF false", "false"),
        ("p IMP q IFF NOT q IMP NOT p", "true"),
        ("p AND ((p AND q) OR r)", "p AND (q OR r)"),
        ("NOT (x < 3)", "x >= 3"),
        ("x < 3 AND x < 5", "x < 3"),
        ("x > 3 OR x <= 3", "true"),
        ("[3 OR 5, NOT 5, 3 AND 5, 3 XOR 5]", "[7, -6, 1, 6]"),
        ("{1, 2} AND {2, 3}", "{2}"),
        ("{1, 2} OR {2, 3}", "{1, 2, 3}"),
        ("(2 < 3) UNION (3 < 2)", "true"),
        # A solution is simplified with the statements it is combined with.
        ("SOLVE(x^2 = 4, x) AND x > 0", "x = 2"),
        ("(x = 1 OR x = 3) AND x > 2", "x = 3"),
        ("SOLVE(x^2 = 4, x) OR x = 7", "x = -2 OR x = 2 OR x = 7"),
    ],
    ids=str,
)
def test_boolean_simplification(text, expected):
    assert simp(text) == expected


# -- domains ------------------------------------------------------------------------


SETS = declared("a :epsilon Set", "b :epsilon Set")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a UNION a", "a"),
        ("a``", "a"),
        ("a INTERSECTION a`", "{}"),
        ("[(a INTERSECTION b)` = a` UNION b`]", "[true]"),
        ("[(a UNION b)` = a` INTERSECTION b`]", "[true]"),
        ("a OR b", "a UNION b"),
        ("NOT a", "a`"),
        ("SET_TYPE?(a)", "true"),
        ("a = b", "a = b"),
    ],
    ids=str,
)
def test_a_variable_declared_a_set_obeys_the_set_algebra(text, expected):
    assert simp(text, SETS) == expected


def test_the_complement_of_a_union_or_intersection_of_sets_is_taken_inside():
    assert simp("(a UNION b)`", SETS) == "a` INTERSECTION b`"
    assert simp("(a INTERSECTION b)`", SETS) == "a` UNION b`"
    assert simp("(a UNION {1})`", SETS) == "a` INTERSECTION {1}`"


def test_an_undeclared_variable_in_a_set_operation_is_a_set():
    """There its transpose is its complement, while alone it is a scalar's, itself."""
    assert simp("[(a UNION b)` = a` INTERSECTION b`]") == "[true]"
    assert simp("[(a INTERSECTION b)` = a` UNION b`]") == "[true]"
    assert simp("(a UNION b)`") == "a` INTERSECTION b`"
    assert simp("a`") == "a"


def test_products_of_variables_declared_vectors_do_not_commute():
    vectors = declared("a :epsilon Vector", "b :epsilon Vector")
    assert simp("a*b - b*a", vectors) == "a . b - b . a"
    assert simp("VECTOR_TYPE?(a)", vectors) == "true"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("IF(q, 1, 2)", "IF(q, 1, 2)"),
        ("q AND NOT q", "false"),
        ("q`", "NOT q"),
        ("LOGICAL_TYPE?(q)", "true"),
        ("NUMBER_TYPE?(q)", "false"),
    ],
    ids=str,
)
def test_a_variable_declared_logical_is_a_truth_value(text, expected):
    assert simp(text, declared("q :epsilon Logical")) == expected


# -- conditionals -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # An undecided test is shown simplified, its clauses as written.
        (
            "IF(t < -SQRT(1/2), t + SQRT(2), IF(t < SQRT(1/2), SQRT(1 - t^2), "
            "SQRT(2) - t))",
            "IF(t < -SQRT(2)/2, t + SQRT(2), IF(t < SQRT(1/2), SQRT(1 - t^2), "
            "SQRT(2) - t))",
        ),
        ("IF(x + x > 3, 1 + 1, 3)", "IF(2*x > 3, 1 + 1, 3)"),
        ("IF(1 < x, 1, 2)", "IF(1 < x, 1, 2)"),
        ("v = IF(t <= 24, (t - 24)^2/4)", "v = IF(t <= 24, (t - 24)^2/4)"),
        ("IF(ABS(x) <= 5, 3*x^2/25 - 3, ?)", "IF(ABS(x) <= 5, 3*x^2/25 - 3, ?)"),
        ("IF(x^2 >= 0, 1, 2)", "1"),
    ],
    ids=str,
)
def test_an_undecided_test_is_shown_simplified(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('IF(F(a) = 4, "OK", "no")', 'IF(a^2 = 4, "OK", "no")'),
        ("IF(F(a) < 4, 1 + 1, 3)", "IF(a^2 < 4, 1 + 1, 3)"),
        ("IF(F(a) = 4 AND b > 0, 1, 2)", "IF(a^2 = 4 AND b > 0, 1, 2)"),
        ("IF(NOT F(a) = 4, 1, 2)", "IF(a^2 /= 4, 1, 2)"),
        ("IF(F(a) = 4)", "IF(a^2 = 4)"),
        ("IF(INT(F(t), t, 0, a) = 1, 1, 2)", "IF(a^3/3 = 1, 1, 2)"),
        # The clauses stay as written.
        ("IF(a > 0, F(a), F(2))", "IF(a > 0, F(a), F(2))"),
        ("IF(F(a) = 4, IF(F(b) = 1, 1, 2), 3)", "IF(a^2 = 4, IF(F(b) = 1, 1, 2), 3)"),
        # A test in a called body is shown so too.
        ("G(a)", "IF(a^2 = 4, 1, 2)"),
        ("H(a)", "IF(a^2 = 4, 1, 2)"),
    ],
    ids=str,
)
def test_an_undecided_test_is_shown_with_its_calls_made(text, expected):
    context = Context(
        functions={
            "F": Definition(("x",), parse("x^2")),
            "G": Definition(("y",), parse("IF(F(y) = 4, 1, 2)", declaring("F"))),
            "H": Definition(("y",), parse("PROG(IF(F(y) = 4, 1, 2))", declaring("F"))),
        }
    )
    assert simp(text, context) == expected


def test_a_declared_interval_decides_a_test():
    below = declared("h :epsilon Real (-inf, 30)")
    assert simp("IF(h <= 40, 10*h, 400 + 15*(h - 40))", below) == "10*h"
    assert simp("IF(t > 1, 1, 0)", declared("t :epsilon Real (1, b)")) == "1"
