"""How the engine prints results, and that printed results parse back.

Each printed form is checked for its spelling in author notation and for parsing back to
the original expression. The worksheet stores results as text and rereads them, so the
round trip is asserted on the tree, not the text.
"""

from __future__ import annotations

import pytest
import sympy as sp
from sexpr import to_sexpr
from sympy.core.function import UndefinedFunction

from rederive.engine.computing import Context, author_text, from_sympy, to_sympy
from rederive.engine.context import Precision
from rederive.engine.evaluate import Suspended, released
from rederive.engine.printer import numeral
from rederive.engine.to_sympy import (
    Assign,
    Declare,
    Dot,
    FunDef,
    Logical,
    PlusMinus,
    StringLiteral,
    Subscript,
    Transposed,
    Vector,
)
from rederive.model.expr import Kind
from rederive.syntax import ParseState, parse_expression

x, y, z = sp.symbols("x y z", real=True)
n = sp.Symbol("n", integer=True)
F = sp.Function("F")
xi = sp.Dummy("xi")


def written(expression, context=None):
    return author_text(expression, context or Context())


def reread(expression, context=None):
    """Print the expression and parse it back."""
    context = context or Context()
    result = from_sympy(expression, context)
    return to_sympy(result.node, context)


# -- spellings ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (x**2, "x^2"),
        (sp.sqrt(x), "SQRT(x)"),
        (1 / sp.sqrt(x), "1/SQRT(x)"),
        (x ** sp.Rational(3, 2), "x^(3/2)"),
        (1 / x, "1/x"),
        (x ** -2, "1/x^2"),
        (sp.exp(x), "#e^x"),
        (sp.exp(x + 1), "#e^(x + 1)"),
        (sp.E, "#e"),
        (sp.I, "#i"),
        (sp.pi, "pi"),
        (sp.oo, "inf"),
        (-sp.oo, "-inf"),
        (sp.zoo, "±inf"),
        (sp.nan, "?"),
        (sp.EulerGamma, "euler_gamma"),
        (sp.true, "true"),
        (sp.false, "false"),
        (sp.log(x), "LN(x)"),
        (sp.Abs(x), "ABS(x)"),
        (sp.factorial(x), "x!"),
        (sp.factorial(x + 1), "(x + 1)!"),
        (sp.sign(x), "SIGN(x)"),
        (sp.Heaviside(x, sp.Rational(1, 2)), "STEP(x)"),
        (sp.atan2(y, x), "ATAN(y, x)"),
        (sp.arg(x), "PHASE(x)"),
        (sp.conjugate(sp.Symbol("z", complex=True)), "CONJ(z)"),
        (sp.binomial(5, 2), "10"),
        (sp.Max(x, y), "MAX(x, y)"),
        (sp.Min(x, y), "MIN(x, y)"),
        (sp.asinh(x), "ASINH(x)"),
        (sp.Symbol("alpha"), "alpha"),
        (sp.Symbol("α"), "alpha"),
        (Vector(1, 2, 3), "[1, 2, 3]"),
        (Vector(Vector(1, 2), Vector(3, 4)), "[[1, 2], [3, 4]]"),
        (Vector(Vector(1, 2)), "[[1, 2]]"),
        (Vector(), "[]"),
        # A sympy matrix, which only a product with a symbolic matrix holds, is
        # written as rows even if it has only one.
        (sp.ImmutableMatrix([[1, 2]]), "[[1, 2]]"),
        (sp.Eq(x, 1, evaluate=False), "x = 1"),
        (sp.Ne(x, 1, evaluate=False), "x /= 1"),
        (sp.Le(x, 1, evaluate=False), "x <= 1"),
        (sp.Function("SOLVE")(x, y), "SOLVE(x, y)"),
    ],
    ids=str,
)
def test_how_an_expression_is_written(expression, expected):
    assert written(expression) == expected


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (sp.Derivative(sp.sin(x), x, evaluate=False), "DIF(SIN(x), x)"),
        (sp.Derivative(sp.sin(x), (x, 2), evaluate=False), "DIF(SIN(x), x, 2)"),
        (sp.Integral(x, x), "INT(x, x)"),
        (sp.Integral(x, (x, 1, 2)), "INT(x, x, 1, 2)"),
        (sp.Sum(x, (x, 1, n)), "SUM(x, x, 1, n)"),
        (sp.Product(x, (x, 1, n)), "PRODUCT(x, x, 1, n)"),
        (sp.Limit(sp.sin(x) / x, x, 0, dir="+-"), "LIM(SIN(x)/x, x, 0)"),
        (sp.Limit(sp.sign(x), x, 0, dir="+"), "LIM(SIGN(x), x, 0, 1)"),
        (sp.Limit(sp.sign(x), x, 0, dir="-"), "LIM(SIGN(x), x, 0, -1)"),
    ],
    ids=str,
)
def test_an_unevaluated_calculus_head_is_written_as_its_own_call(expression, expected):
    assert written(expression) == expected


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (sp.Derivative(F(x), x, evaluate=False), "F'(x)"),
        (sp.Derivative(F(x), (x, 2), evaluate=False), "F''(x)"),
        # The chain rule leaves the derivative evaluated at a non-variable point, which
        # sympy represents as a substitution.
        (sp.Subs(sp.Derivative(F(xi), xi, evaluate=False), (xi,), (x**2,)), "F'(x^2)"),
        # With two arguments the variable must be named, so it stays a `DIF`.
        (sp.Derivative(F(x, y), x, evaluate=False), "DIF(F(x, y), x)"),
        # Primes are only for undefined functions; known heads keep their name.
        (sp.Derivative(sp.sin(x), x, evaluate=False), "DIF(SIN(x), x)"),
    ],
    ids=str,
)
def test_the_derivative_of_an_undefined_function_is_written_with_primes(
    expression, expected
):
    assert written(expression) == expected


def test_plus_or_minus_is_written_as_the_glyph():
    """Deliberate; don't "correct" it to `+-`.

    `±` and the quoted `"+-"` are the operator's two input spellings. A bare `+-u` is
    unary plus applied to a minus, so `±1` written that way would reread as `-1`.
    """
    assert written(PlusMinus(sp.Integer(1))) == "±1"
    assert written(sp.zoo) == "±inf"
    assert to_sympy(parse_expression("±1", ParseState()).node) == PlusMinus(1)
    assert to_sympy(parse_expression('"+-"1', ParseState()).node) == PlusMinus(1)
    assert to_sympy(parse_expression("+-1", ParseState()).node) == -1


def test_a_logical_operator_is_written_with_its_keyword():
    p, q, r = sp.symbols("p q r")
    assert written(sp.Not(p)) == "NOT p"
    assert written(sp.And(p, q)) == "p AND q"
    assert written(sp.Implies(p, q)) == "p IMP q"
    # A looser-binding operand is parenthesized. Commutative operands are ordered by
    # their leading variable, so the group comes first here.
    assert written(sp.And(sp.Or(p, q), r)) == "(p OR q) AND r"
    assert written(sp.Not(sp.And(p, q))) == "NOT (p AND q)"


def test_the_operands_of_a_commutative_operator_are_put_in_order():
    """`AND` and `OR` operands are ordered as the original does: by leading variable,
    negated first on a tie. `IMP` is not commutative and keeps its order."""
    p, q, r, s = sp.symbols("p q r s")
    assert written(sp.Or(r, q, p)) == "p OR q OR r"
    assert written(sp.Or(sp.And(s, r), sp.And(q, p))) == "p AND q OR r AND s"
    assert written(sp.Or(q, sp.Not(p))) == "NOT p OR q"
    assert written(sp.Or(sp.And(p, q), sp.And(sp.Not(p), r))) == (
        "NOT p AND r OR p AND q"
    )
    # Variables in the order list come first, the rest alphabetically.
    assert written(sp.And(*sp.symbols("a z x"))) == "x AND z AND a"
    assert written(sp.Implies(q, p)) == "q IMP p"


def test_the_terms_of_a_sum_are_put_in_order():
    """Terms sort by leading variable in order-list order, then by descending degree;
    terms with no variable go last."""
    a, x, y, z = sp.symbols("a x y z")
    assert written(5 * y**3 + 2 * x**2 - 3 * a * x) == "2*x^2 - 3*a*x + 5*y^3"
    assert written(a + z) == "z + a"
    assert written(sp.Symbol("w") + x) == "x + w"
    assert written(x**2 + sp.Symbol("c") + 1) == "x^2 + c + 1"
    # The real part of a complex number always comes first.
    assert written(7 * sp.I + 1) == "1 + 7*#i"


#: The original's order for terms tied on variable and degree, checked pairwise against
#: it. It is not alphabetical: `LN` comes before `COS` and `TAN` before `SIN`.
HEADS = {
    "ABS": sp.Abs,
    "ASIN": sp.asin,
    "ATAN": sp.atan,
    "LN": sp.log,
    "COS": sp.cos,
    "TAN": sp.tan,
    "SIN": sp.sin,
}


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (first, second)
        for index, first in enumerate(HEADS)
        for second in list(HEADS)[index + 1 :]
    ],
    ids=str,
)
def test_which_function_head_leads_a_sum_of_two(first, second):
    """The sum is written the same way whichever order it is built in."""
    z = sp.Symbol("z")
    expected = f"{first}(z) + {second}(z)"
    assert written(HEADS[first](z) + HEADS[second](z)) == expected
    assert written(HEADS[second](z) + HEADS[first](z)) == expected


def test_the_variable_of_a_term_outranks_its_head():
    """All `t` terms come before all `u` terms, regardless of the function heads."""
    t, u = sp.symbols("t u")
    assert written(sp.cos(u) + sp.sin(t)) == "SIN(t) + COS(u)"


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        # Sympy splits these on evaluation; the original writes them whole.
        (sp.Integer(12) ** sp.Rational(1, 100), "12^(1/100)"),
        (sp.Integer(12) ** sp.Rational(1, 4), "12^(1/4)"),
        (sp.Integer(72) ** sp.Rational(1, 4), "72^(1/4)"),
        (x * sp.Integer(12) ** sp.Rational(1, 4) / 7, "12^(1/4)*x/7"),
        # Sympy rationalizes the denominator, as the original does.
        (1 / sp.Integer(12) ** sp.Rational(1, 4), "108^(1/4)/6"),
        # Roots aren't combined into one of higher order than either factor's.
        (sp.Integer(2) ** sp.Rational(2, 3) * sp.sqrt(3), "2^(2/3)*SQRT(3)"),
        # Nor into one longer than the factors it would replace.
        (
            5 / sp.Integer(12) ** sp.Rational(1, 100),
            "5*2^(49/50)*3^(99/100)/6",
        ),
        # An integer coefficient stays separate.
        (sp.sqrt(12), "2*SQRT(3)"),
    ],
    ids=str,
)
def test_roots_of_integers_in_a_product_are_written_as_one(expression, expected):
    assert written(expression) == expected
    assert sp.simplify(reread(expression) - expression) == 0


def test_two_bounds_on_one_variable_are_written_as_a_chain():
    """A conjunction forming a range is written as a chain, `-2 < x < 2`, as in the
    original. Each bound may be written either way round, strict or not."""
    x, y = sp.symbols("x y", real=True)
    assert written(sp.And(x > -2, x < 2)) == "-2 < x < 2"
    assert written(sp.And(sp.Lt(-2, x), sp.Le(x, 2))) == "-2 < x <= 2"
    assert written(sp.And(x < 2, x >= -2)) == "-2 <= x < 2"
    assert written(sp.And(x > -2, x < 2 * y)) == "-2 < x < 2*y"


def test_a_pair_that_brackets_nothing_keeps_its_keyword():
    """Two bounds from the same side, or on unrelated variables, are not a range.

    The shared variable determines the chain, so `x < y AND y < 1` is a range around `y`.
    """
    x, y, z = sp.symbols("x y z", real=True)
    assert written(sp.And(x < y, y < 1)) == "x < y < 1"
    assert written(sp.And(x < 1, y < 1)) == "x < 1 AND y < 1"
    assert written(sp.And(x > -2, x > 1)) == "x > -2 AND x > 1"
    assert written(sp.And(x < y, y < z)) == "x < y < z"
    # `/=` is not a bound, so this stays two statements, in sympy's order.
    assert written(sp.And(sp.Ne(x, 1), x < 2)) == "x < 2 AND x /= 1"


# -- numerals ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("number", "base", "expected"),
    [(255, 10, "255"), (255, 16, "0FF"), (5, 2, "101"), (-14, 16, "-0E"), (0, 16, "0")],
    ids=str,
)
def test_a_numeral_is_written_in_its_base(number, base, expected):
    # A numeral starting with a letter gets a leading zero so it isn't read as a name.
    assert numeral(number, base) == expected


def test_numerals_are_written_in_the_input_base_so_that_they_read_back():
    context = Context(input_base=16)
    assert written(sp.Integer(255), context) == "0FF"
    assert written(sp.Rational(255, 16), context) == "0FF/10"
    state = ParseState(input_base=16)
    assert parse_expression("0FF", state).node.value == "255"


def test_a_float_is_written_at_the_precision_of_the_context():
    context = Context(precision=Precision.APPROXIMATE, precision_digits=6)
    assert written(sp.pi.evalf(30), context) == "3.14159"
    assert written(sp.sqrt(2).evalf(30), context) == "1.41421"


def test_a_float_is_written_as_plain_digits_and_keeps_its_point():
    context = Context(precision=Precision.APPROXIMATE)
    # Exponent notation can't be read back, and a whole float without its point would
    # read back as an integer.
    assert written(sp.Float("1e20", 20), context) == "100000000000000000000."
    assert "." in written(sp.Float(3), context)


def test_a_float_is_written_to_the_digits_asked_for():
    context = Context(precision=Precision.APPROXIMATE, precision_digits=12)
    assert written(sp.pi.evalf(30), context) == "3.14159265359"


# -- reading back ------------------------------------------------------------

REREAD = [
    sp.Integer(-7),
    sp.Rational(-3, 4),
    x**2 * y - 1,
    (x + 1) / (x - 1),
    sp.sqrt(x + 1),
    x ** sp.Rational(-3, 2),
    (x**y) ** sp.Symbol("z", real=True),
    sp.exp(2 * x),
    sp.factorial(x + 1),
    sp.factorial(sp.Rational(3, 2)),
    sp.Abs(x - y),
    sp.sin(x) ** 2 + sp.cos(x) ** 2,
    Vector(x, -y),
    Vector(Vector(x, 1), Vector(0, y)),
    Vector(Vector(x, 1)),
    sp.Eq(x**2, y, evaluate=False),
    sp.Integral(sp.sin(x) / x, (x, 0, sp.oo)),
    sp.Derivative(F(x), (x, 3), evaluate=False),
    sp.Subs(sp.Derivative(F(xi), xi, evaluate=False), (xi,), (x**2,)),
    sp.Function("SOLVE")(sp.Eq(x, 1, evaluate=False), x),
    sp.Symbol("x SUB 1", real=True) + 1,
    -sp.Symbol("tera", real=True),
]

#: Heads with no sympy equivalent. They are why output goes through a printer rather
#: than a formatter, so they are round-tripped too.
INERT = [
    PlusMinus(x),
    # `PlusMinus(oo)` is omitted on purpose: `±inf` denotes unsigned infinity and reads
    # back as sympy's `zoo`.
    StringLiteral("note"),
    Subscript(x + 1, 2),
    Subscript(x, -1),
    # Sympy transposes whatever it can (a scalar's transpose is the scalar), so the
    # inert head only remains around a non-expression.
    Transposed(sp.Eq(x, 1)),
    Dot(sp.Symbol("a", real=True), Vector(1, 2)),
    Vector(Vector(1, 2), Vector(3)),
    Logical(sp.Symbol("AND"), sp.Integer(3), sp.Symbol("p", real=True)),
    Assign(sp.Symbol("u", real=True), sp.Symbol(":="), sp.Integer(5)),
    Assign(sp.Symbol("u", real=True), sp.Symbol(":==")),
    FunDef(sp.Symbol("F"), sp.Tuple(x, y), x + y),
    FunDef(sp.Symbol("F"), sp.Tuple(x)),
    Declare(sp.Symbol("x", real=True), sp.Symbol("Real")),
    # The declared symbol carries the declared assumptions.
    Declare(
        sp.Symbol("k", integer=True, nonnegative=True),
        sp.Symbol("Integer"),
        sp.Symbol("[)"),
        0,
        sp.oo,
    ),
]


@pytest.mark.parametrize("expression", REREAD + INERT, ids=str)
def test_what_is_printed_parses_back_to_the_same_expression(expression):
    assert reread(expression) == expression


@pytest.mark.parametrize("expression", REREAD + INERT, ids=str)
def test_printing_is_a_fixed_point(expression):
    once = from_sympy(expression)
    twice = from_sympy(to_sympy(once.node))
    assert twice.text == once.text
    assert to_sexpr(twice.node) == to_sexpr(once.node)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (sp.erfi(x), "ERFI(x)"),
        (sp.Ei(x), "EI(x)"),
        (sp.Si(x), "SI(x)"),
        (sp.Ci(x), "CI(x)"),
        (sp.LambertW(x), "LAMBERTW(x)"),
        (sp.LambertW(x, -1), "LAMBERTW(x, -1)"),
        (sp.polylog(2, x), "POLYLOG(2, x)"),
        (sp.polygamma(1, x), "POLYGAMMA(1, x)"),
        # Sympy represents digamma as polygamma of order zero.
        (sp.digamma(x), "DIGAMMA(x)"),
        (sp.polygamma(0, x), "DIGAMMA(x)"),
        (sp.gamma(x) * sp.polygamma(0, x), "GAMMA(x)*DIGAMMA(x)"),
    ],
    ids=str,
)
def test_a_special_function_is_written_under_its_name_and_reads_back(
    expression, expected
):
    assert written(expression) == expected
    assert reread(expression) == expression


def test_a_multi_character_name_reads_back_as_one_name():
    # In Character mode `tera` is four factors unless declared; the state `from_sympy`
    # builds declares it.
    result = from_sympy(sp.Symbol("tera", real=True) * 2)
    assert result.text == "2*tera"
    assert to_sympy(result.node) == sp.Symbol("tera", real=True) * 2


def test_a_caller_may_supply_the_parse_state():
    state = ParseState()
    from_sympy(sp.Symbol("kilo", real=True), Context(), state)
    # The caller's state is used as given, not replaced.
    assert "kilo" not in state.variables


# -- what `.doit()` can hand back --------------------------------------------
#
# Sympy may answer integrals and derivatives with heads author notation lacks, and with
# invented bound variables. Each must print as valid input, or the entry can't be
# read back.


def test_a_case_split_is_written_as_the_conditional_the_notation_has():
    piecewise = sp.Piecewise((1 / x, sp.Ne(x, 0)), (sp.oo, True))
    assert written(piecewise) == "IF(x /= 0, 1/x, inf)"
    # Without a final `true` case there is no else clause: outside every condition there
    # is no value.
    assert written(sp.Piecewise((1 / x, sp.Ne(x, 0)))) == "IF(x /= 0, 1/x)"


def test_a_head_carrying_tuples_writes_them_as_vectors():
    # `SUBS(F(y), (y,), (x,))` would not parse, since a parenthesized list is not an
    # expression; a vector is.
    assert written(sp.Subs(sp.Function("F")(y), y, x)) == "SUBS(F(y), [y], [x])"


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (sp.hyper((1, 2), (3,), x), '"hyper([1, 2], [3], x)"'),
        (sp.besselj(1, x) + 1, '"besselj(1, x)" + 1'),
        (sp.uppergamma(y, x), '"uppergamma(y, x)"'),
    ],
    ids=str,
)
def test_a_sympy_function_without_a_name_is_written_as_a_string(
    expression, expected, unreadable_answers
):
    """A sympy function with no name in the notation isn't given one.

    Upper-casing `hyper` to `HYPER` would invent a name only sympy defines. Instead it
    is quoted, like a set, and recorded as unreadable. The calculus commands never
    return one: an answer containing one leaves the original head unevaluated.
    """
    assert written(expression) == expected
    from_sympy(expression)
    assert unreadable_answers == [expected]


SETS = [
    (sp.S.EmptySet, '"EmptySet"'),
    (sp.S.Reals, '"Reals"'),
    (sp.S.Integers, '"Integers"'),
    (sp.S.Naturals, '"Naturals"'),
    (sp.S.Naturals0, '"Naturals0"'),
    (sp.S.Rationals, '"Rationals"'),
    (sp.S.Complexes, '"Complexes"'),
    (sp.S.UniversalSet, '"UniversalSet"'),
    (sp.Interval(0, 1), '"Interval(0, 1)"'),
    (sp.FiniteSet(1, 2), '"{1, 2}"'),
    (
        sp.Union(sp.Interval(0, 1), sp.Interval(2, 3)),
        '"Union(Interval(0, 1), Interval(2, 3))"',
    ),
    (
        sp.ImageSet(sp.Lambda(n, 2 * sp.pi * n), sp.S.Integers),
        '"ImageSet(Lambda(n, 2*pi*n), Integers)"',
    ),
]


@pytest.mark.parametrize(("expression", "expected"), SETS, ids=str)
def test_a_set_is_written_as_the_string_the_notation_holds_it_in(
    expression, expected, unreadable_answers
):
    """A sympy set has no author notation and must not look as if it had one.

    soLve converts sets to chained relations and `@n` before printing, as Derive does,
    so these shouldn't be reached. This pins what happens if some path does reach
    them: the answer is marked as unwritable.

    Unquoted, `EmptySet` would read back in Character mode as `e*m*p*t*y*s*e*t`, and
    `Interval(0, 1)` as the built-in `INTERVAL`, which means a limit's range rather
    than a set (the same kind of collision `LI` and `CHI` were). A string is inert.
    """
    assert written(expression) == expected
    result = from_sympy(expression)
    assert result.node.kind is Kind.STRING
    # The engine also reports it, so new cases don't go unnoticed.
    assert unreadable_answers == [expected]


def test_a_set_inside_a_set_is_not_quoted_twice():
    """Only the outermost set is quoted.

    Its parts are written as usual, in author notation where possible, which keeps the
    text legible enough to diagnose.
    """
    condition = sp.ConditionSet(x, sp.Eq(x**2, 1), sp.S.Reals)
    assert written(condition) == '"ConditionSet(x, x^2 = 1, Reals)"'


def test_a_string_the_author_wrote_is_not_an_unreadable_answer(unreadable_answers):
    """Both unwritable heads and string literals produce `Kind.STRING`, but only the
    former is a failure."""
    assert from_sympy(StringLiteral("HELP.MTH")).node.kind is Kind.STRING
    assert unreadable_answers == []


def test_a_sympy_bound_variable_is_written_as_an_ordinary_name():
    # Sympy prints a `Dummy` as `_k1`, which doesn't lex, and `k1` and `xi` lex in
    # Character mode as `k*1` and `x*i`. A single letter is one name in every mode.
    assert written(sp.Dummy("k1")) == "k"
    assert written(sp.Dummy("xi_2") + 1) == "x + 1"
    assert written(sp.Dummy("xi") + x) == "x + x_"
    assert parse_expression("x + x_", ParseState()).node.children[1].value == "x_"


def test_two_bound_variables_of_one_name_are_written_apart():
    """Sympy distinguishes its invented variables by identity, which text can't hold.

    Differentiating a product `n` times twice gives two dummies both named `k1`; writing
    both as `k1` would merge them into one variable. Distinct names also keep the term
    order of a sum deterministic.
    """
    first, second = sp.Dummy("k1"), sp.Dummy("k1")
    assert written(first + 2 * second) in ("k_ + 2*k", "k + 2*k_")
    # A name already in use is not reused.
    assert written(sp.Dummy("k1") + sp.Symbol("k", real=True)) == "k + k_"


def test_a_dot_product_keeps_its_parentheses_inside_a_negated_product():
    # `-x*a . b` reads as `(-x*a) . b`, since `.` and `*` have the same precedence.
    a, b = sp.symbols("a b", real=True)
    assert written(-x * Dot(a, b)) == "-x*(a . b)"
    assert written(x * Dot(a, b)) == "x*(a . b)"


def test_a_power_of_a_vector_is_written_in_author_notation():
    assert written(sp.sqrt(Vector(sp.Rational(1, 12)))) == "SQRT([1/12])"


def test_an_inert_head_is_parenthesised_by_what_it_is_written_as():
    """`±u` is parenthesized like a sum, not like a call.

    `#e^±inf*z` would read back as `(#e^±inf)*z`, and `±inf*z` as `±(inf*z)`, since
    plus-or-minus takes a whole product. So unsigned infinity (`±inf`) inside a product
    needs parentheses.
    """
    assert written(sp.exp(sp.zoo * z)) == "#e^((±inf)*z)"
    assert to_sympy(from_sympy(sp.exp(sp.zoo * z)).node).args[0] == sp.zoo * z
    # Plus-or-minus over anything else round-trips as the `PlusMinus` head.
    assert to_sympy(from_sympy(PlusMinus(sp.oo * z)).node) == PlusMinus(sp.oo * z)


@pytest.mark.parametrize(
    "expression",
    [
        sp.Piecewise((1 / x, sp.Ne(x, 0)), (sp.oo, True)),
        sp.Subs(sp.Function("F")(y), y, x),
        sp.Dummy("k1") * 2,
        sp.exp(sp.zoo * z),
    ],
    ids=str,
)
def test_what_doit_hands_back_reads_back(expression):
    result = from_sympy(expression)
    assert result.node.kind is not Kind.STRING
    assert from_sympy(to_sympy(result.node)).text == result.text


def test_a_supplied_state_learns_the_names_only_the_answer_uses():
    """Names that appear only in the answer are declared in a copy of the caller's state.

    An answer can contain a name never typed on the worksheet, and in Character mode
    `k1` is `k*1` unless declared. The engine must not modify the caller's state.
    """
    state = ParseState()
    invented = sp.Symbol("k1", real=True) * 2
    result = from_sympy(invented, Context(), state)
    assert result.text == "2*k1"
    assert to_sympy(result.node) == sp.Symbol("k1", real=True) * 2
    assert "k1" not in state.variables


# -- suspended conditionals ----------------------------------------------------

i, j = sp.symbols("i j", real=True)


def _suspended(display, *variables):
    """A suspended conditional over `variables` showing as `display`, as the evaluator
    makes them."""
    head = UndefinedFunction(
        "IF_9",
        bases=(Suspended,),
        __dict__={
            "node": None,
            "environment": None,
            "display": display,
            "variables": variables,
        },
    )
    return head(*variables)


def test_a_suspended_conditional_prints_as_its_conditional():
    """Its class name is internal: TENSOR.MTH's `PERM_EL([1, 2, 3])` once printed
    `IF_2(iIp, jJp)`."""
    conditional = sp.Piecewise((i, i < j), (2, True))
    assert written(_suspended(conditional, i, j)) == "IF(i < j, i, 2)"
    # Applied to values other than its variables, it shows them substituted.
    head = type(_suspended(conditional, i, j))
    assert written(head(i, x)) == "IF(i < x, i, 2)"


def test_a_suspended_conditional_is_released_where_its_parent_rejects_it():
    """`1 > DIMENSION(APPEND([], ...))` around a conditional can't be rebuilt with the
    conditional in place, since sympy holds a function of a matrix non-real; it is
    rebuilt unevaluated instead."""
    conditional = sp.Piecewise((i, i < j), (2, True))
    dimension, append = sp.Function("DIMENSION"), sp.Function("APPEND")
    with sp.evaluate(False):
        test = sp.StrictGreaterThan(
            1, dimension(append(sp.ImmutableMatrix([]), _suspended(conditional, i, j)))
        )
    shown = released(test)
    assert not shown.atoms(Suspended)
    assert written(shown) == "1 > DIMENSION(APPEND([], IF(i < j, i, 2)))"
