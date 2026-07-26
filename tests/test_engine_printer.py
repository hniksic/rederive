"""What the engine's results look like, and that they read back.

Two things are checked of every printed form: that it is spelled the way the
author line spells it, and that parsing it gives back the expression it came
from. The second is the invariant the worksheet rests on - a result is stored
as text and read again - so it is asserted on the tree, not on the text.
"""

from __future__ import annotations

import pytest
import sympy as sp
from sexpr import to_sexpr

from rederive.engine import Context, author_text, from_sympy, to_sympy
from rederive.engine.context import Precision
from rederive.engine.printer import numeral
from rederive.engine.to_sympy import (
    Assign,
    Declare,
    Dot,
    FunDef,
    InertVector,
    Logical,
    PlusMinus,
    StringLiteral,
    Subscript,
    Transposed,
)
from rederive.model.expr import Kind
from rederive.syntax import ParseState, parse_expression

x, y, z = sp.symbols("x y z", real=True)
n = sp.Symbol("n", integer=True)


def written(expression, context=None):
    return author_text(expression, context or Context())


def reread(expression, context=None):
    """The expression, printed and parsed again."""
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
        (sp.Matrix(1, 3, [1, 2, 3]), "[1, 2, 3]"),
        (sp.Matrix([[1, 2], [3, 4]]), "[[1, 2], [3, 4]]"),
        (sp.Matrix(0, 0, []), "[]"),
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


def test_plus_or_minus_is_written_as_the_glyph():
    """Deliberate, and not to be "corrected" to `+-`.

    `±` and the quoted `"+-"` are the operator's two input spellings; a bare
    `+-u` is a plus applied to a minus, so writing a result that way turns
    `±1` into `-1` the next time it is read.
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
    # An operand that binds more loosely takes parentheses.
    assert written(sp.And(sp.Or(p, q), r)) == "r AND (p OR q)"
    assert written(sp.Not(sp.And(p, q))) == "NOT (p AND q)"


# -- numerals ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("number", "base", "expected"),
    [(255, 10, "255"), (255, 16, "0FF"), (5, 2, "101"), (-14, 16, "-0E"), (0, 16, "0")],
    ids=str,
)
def test_a_numeral_is_written_in_its_base(number, base, expected):
    # A numeral that would begin with a letter takes a leading zero, so that
    # it cannot be read as a name.
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
    # There is no exponent notation to read back, and a whole float that lost
    # its point would read back as an integer.
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
    sp.Matrix(1, 2, [x, -y]),
    sp.Matrix([[x, 1], [0, y]]),
    sp.Eq(x**2, y, evaluate=False),
    sp.Integral(sp.sin(x) / x, (x, 0, sp.oo)),
    sp.Derivative(sp.Function("F")(x), (x, 3), evaluate=False),
    sp.Function("SOLVE")(sp.Eq(x, 1, evaluate=False), x),
    sp.Symbol("x SUB 1", real=True) + 1,
    -sp.Symbol("tera", real=True),
]

#: The heads sympy has no object for. They are the whole reason the outbound
#: path is a printer and not a formatter, so they are read back too.
INERT = [
    PlusMinus(x),
    # `PlusMinus(oo)` is absent on purpose: `±inf` is the notation for unsigned
    # infinity, and reading it back gives sympy's own object for that.
    StringLiteral("note"),
    Subscript(x + 1, 2),
    Subscript(x, -1),
    Transposed(sp.Symbol("a", real=True)),
    Dot(sp.Symbol("a", real=True), sp.Symbol("b", real=True)),
    InertVector(sp.Matrix(1, 2, [1, 2]), sp.Matrix(1, 1, [3])),
    Logical(sp.Symbol("AND"), sp.Integer(3), sp.Symbol("p", real=True)),
    Assign(sp.Symbol("u", real=True), sp.Symbol(":="), sp.Integer(5)),
    Assign(sp.Symbol("u", real=True), sp.Symbol(":==")),
    FunDef(sp.Symbol("F"), sp.Tuple(x, y), x + y),
    FunDef(sp.Symbol("F"), sp.Tuple(x)),
    Declare(sp.Symbol("x", real=True), sp.Symbol("Real")),
    # The declared symbol carries what the declaration says about it.
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


def test_a_multi_character_name_reads_back_as_one_name():
    # In Character mode `tera` is four factors until something declares it,
    # and the state `from_sympy` builds is what declares it.
    result = from_sympy(sp.Symbol("tera", real=True) * 2)
    assert result.text == "2*tera"
    assert to_sympy(result.node) == sp.Symbol("tera", real=True) * 2


def test_a_caller_may_supply_the_parse_state():
    state = ParseState()
    from_sympy(sp.Symbol("kilo", real=True), Context(), state)
    # The caller's state is used as given; it is not the engine's to replace.
    assert "kilo" not in state.variables


# -- what `.doit()` can hand back --------------------------------------------
#
# Sympy answers some integrals and derivatives with heads the notation has no
# word for, and with bound variables it invented on the spot. Each has to come
# out as something the author line could have been typed with, or the result is
# unreadable and the entry is lost.


def test_a_case_split_is_written_as_the_conditional_the_notation_has():
    piecewise = sp.Piecewise((1 / x, sp.Ne(x, 0)), (sp.oo, True))
    assert written(piecewise) == "IF(x /= 0, 1/x, inf)"
    # Without a final `true` the last case has no else clause, which is right:
    # outside every condition there is no value.
    assert written(sp.Piecewise((1 / x, sp.Ne(x, 0)))) == "IF(x /= 0, 1/x)"


def test_a_head_carrying_tuples_writes_them_as_vectors():
    # `HYPER((1, 2), (3,), x)` is not readable - a parenthesised list is no
    # expression - and a vector is.
    assert written(sp.hyper((1, 2), (3,), x)) == "HYPER([1, 2], [3], x)"
    assert written(sp.Subs(sp.Function("F")(y), y, x)) == "SUBS(F(y), [y], [x])"


def test_a_sympy_bound_variable_is_written_as_an_ordinary_name():
    # Sympy writes a `Dummy` as `_k1`, and `_k1` does not lex.
    assert written(sp.Dummy("k1")) == "k1"
    assert written(sp.Dummy("xi_2") + 1) == "xi_2 + 1"


def test_two_bound_variables_of_one_name_are_written_apart():
    """Sympy tells its invented variables apart by an identity text cannot hold.

    Differentiating a product `n` times twice over gives two of them, both
    called `k1`, and writing both as `k1` would produce text that reads back as
    an expression in one variable where there were two. The names are also what
    the terms of a sum are ordered by, so a name kept for two things would make
    the written order depend on which of them came first.
    """
    first, second = sp.Dummy("k1"), sp.Dummy("k1")
    assert written(first + 2 * second) in ("k1_ + 2*k1", "k1 + 2*k1_")
    # And a name already in use is not taken over.
    assert written(sp.Dummy("k1") + sp.Symbol("k1", real=True)) == "k1 + k1_"


def test_a_dot_product_keeps_its_parentheses_inside_a_negated_product():
    # `-x*a . b` reads as `(-x*a) . b`: `.` and `*` are one level in the
    # grammar, and the left operand would swallow the factor in front.
    a, b = sp.symbols("a b", real=True)
    assert written(-x * Dot(a, b)) == "-x*(a . b)"
    assert written(x * Dot(a, b)) == "x*(a . b)"


def test_a_power_of_a_matrix_is_written_in_author_notation():
    # Sympy writes this one `Matrix(...)**(1/2)`, and `**` is no operator here:
    # the text would read back as a product with an empty factor.
    assert written(sp.sqrt(sp.Matrix(1, 1, [sp.Rational(1, 12)]))) == "SQRT([1/12])"


def test_an_inert_head_is_parenthesised_by_what_it_is_written_as():
    """`±u` is written like a sum, not like a call.

    `#e^±inf*z` would read back as `(#e^±inf)*z`, which is a different
    expression, and `±inf*z` as `±(inf*z)`: the plus-or-minus takes a whole
    product, so unsigned infinity standing in one needs its parentheses.
    `±inf` is the notation for unsigned infinity, and comes back as that.
    """
    assert written(sp.exp(sp.zoo * z)) == "#e^((±inf)*z)"
    assert to_sympy(from_sympy(sp.exp(sp.zoo * z)).node).args[0] == sp.zoo * z
    # A plus-or-minus over anything else is the head it is written as.
    assert to_sympy(from_sympy(PlusMinus(sp.oo * z)).node) == PlusMinus(sp.oo * z)


@pytest.mark.parametrize(
    "expression",
    [
        sp.Piecewise((1 / x, sp.Ne(x, 0)), (sp.oo, True)),
        sp.hyper((1, 2), (3,), x),
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
    """A caller's symbol table does not know what the engine invented.

    Differentiating a product `n` times gives a sum over a bound variable
    sympy names itself, and in Character mode `k1` is `k*1` until something
    declares it. The caller's state is not the engine's to write to, so the
    declaration goes into a copy.
    """
    state = ParseState()
    invented = sp.Dummy("k1") * 2
    result = from_sympy(invented, Context(), state)
    assert result.text == "2*k1"
    assert to_sympy(result.node) == sp.Symbol("k1", real=True) * 2
    # The session's own table is left as it was.
    assert "k1" not in state.variables
