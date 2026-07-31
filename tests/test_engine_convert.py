"""Node -> sympy and back: the engine's two doors.

What is checked here is the translation, not the mathematics. A case asserts
either what sympy object a construct becomes or that the text survives the
round trip - author text -> tree -> sympy -> text - which is the property
every command depends on and the only thing that keeps an unknown construct
alive.
"""

from __future__ import annotations

import pytest
import sympy as sp
from sexpr import to_sexpr
from sympy.core.function import AppliedUndef

from rederive.engine.computing import (
    Context,
    Domain,
    DomainKind,
    from_sympy,
    to_sympy,
)
from rederive.engine.context import Angle, Precision
from rederive.engine.to_sympy import (
    DIMENSION,
    FUNCTIONS,
    Assign,
    Declare,
    FunDef,
    InertVector,
    PlusMinus,
    StringLiteral,
    Subscript,
    Taylor,
    Transposed,
)
from rederive.model.expr import Kind
from rederive.syntax import ParseState, parse_expression
from rederive.syntax.names import BUILTIN_FUNCTIONS


def parse(text, state=None):
    return parse_expression(text, state or ParseState()).node


def convert(text, context=None, state=None):
    """The sympy object `text` converts to."""
    return to_sympy(parse(text, state), context or Context())


def written(text, context=None, state=None):
    """`text` converted and written back out."""
    context = context or Context()
    return from_sympy(convert(text, context, state), context).text


def roundtrip(text, context=None, state=None):
    """The text after one pass through the engine, and after two.

    Equal values mean the printer's output is a fixed point, which is what
    makes a result safe to put in the worksheet and read back later.
    """
    context = context or Context()
    first = from_sympy(convert(text, context, state), context)
    second = from_sympy(to_sympy(first.node, context), context)
    return first.text, second.text


# -- leaves ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2", sp.Integer(2)),
        ("0.1", sp.Rational(1, 10)),
        ("2.5", sp.Rational(5, 2)),
        ("2.", sp.Integer(2)),
        ("pi", sp.pi),
        ("#e", sp.E),
        ("#i", sp.I),
        ("inf", sp.oo),
        ("deg", sp.pi / 180),
        ("true", sp.true),
        ("false", sp.false),
        ("euler_gamma", sp.EulerGamma),
        ("?", sp.nan),
    ],
    ids=str,
)
def test_a_leaf_converts_to_its_sympy_counterpart(text, expected):
    assert convert(text) == expected


def test_e_and_i_are_variables():
    # Only `#e` and `#i` are the constants.
    assert convert("e") == sp.Symbol("e", real=True)
    assert convert("i^2") == sp.Symbol("i", real=True) ** 2


def test_a_decimal_is_rounded_to_the_precision_in_approximate_mode():
    """An approximate number is a rational: the simplest one the precision allows.

    One tenth needs no digits, so it approximates itself. A number that needs
    more digits than there are does not: six of them cannot tell
    `123456789/10000` from `308642/25`, so that is what approximate mode reads
    it as - the original's own answer - and it is the rounding on the way in
    that makes approximate arithmetic approximate.
    """
    approximate = Context(precision=Precision.APPROXIMATE)
    assert convert("0.1", approximate) == sp.Rational(1, 10)
    assert convert("12345.6789", approximate) == sp.Rational(308642, 25)


def test_a_string_is_inert_and_is_not_the_variable_of_that_name():
    assert convert('"x"') == StringLiteral("x")
    assert convert('"x"') != sp.Symbol("x")


def test_a_label_resolves_to_what_it_names():
    context = Context(labels={3: parse("x + 1")})
    assert convert("#3", context) == sp.Symbol("x", real=True) + 1


def test_an_unresolved_label_stays_inert():
    assert convert("#3") == sp.Symbol("#3")


def test_a_label_that_reaches_itself_stops():
    context = Context(labels={1: parse("#1 + 1")})
    assert convert("#1", context) == sp.Symbol("#1") + 1


# -- operators ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1 - 2 + 3", sp.Integer(2)),
        ("2*3*4", sp.Integer(24)),
        ("6/3", sp.Integer(2)),
        ("2^10", sp.Integer(1024)),
        ("-x", -sp.Symbol("x", real=True)),
        ("+x", sp.Symbol("x", real=True)),
        ("4!", sp.Integer(24)),
        ("50%", sp.Rational(1, 2)),
        ("|-3|", sp.Integer(3)),
        ("MAX(1, 7, 3)", sp.Integer(7)),
    ],
    ids=str,
)
def test_an_operator_converts(text, expected):
    assert convert(text) == expected


def test_plus_or_minus_is_inert():
    assert convert("±3") == PlusMinus(3)
    # Arithmetic does not reach inside it.
    assert convert("±3 + 1") == PlusMinus(3) + 1


def test_a_subscripted_variable_is_one_symbol():
    assert convert("x SUB 1") == sp.Symbol("x SUB 1", real=True)
    assert convert("x SUB 1").free_symbols == {sp.Symbol("x SUB 1", real=True)}


def test_a_subscripted_variable_inherits_the_domain_of_its_base():
    context = Context(domains={"n": Domain(DomainKind.INTEGER)})
    assert convert("n SUB 1", context).is_integer


def test_a_computed_subscript_keeps_its_operands_reachable():
    subscript = convert("x SUB (i + 1)")
    assert isinstance(subscript, Subscript)
    assert sp.Symbol("i", real=True) in subscript.free_symbols


def test_a_subscript_on_a_vector_selects_an_element_counting_from_one():
    assert convert("[10, 20, 30] SUB 2") == sp.Integer(20)


def test_a_subscript_on_a_matrix_selects_a_row():
    assert convert("[[1, 2], [3, 4]] SUB 2") == sp.Matrix(1, 2, [3, 4])


def test_a_transpose_of_a_matrix_is_a_matrix_and_of_a_scalar_is_the_scalar():
    # 8.5 p.205: the transpose of a scalar is the scalar, and everything not
    # declared nonscalar is one. What is no expression at all keeps the head.
    assert convert("[[1, 2], [3, 4]]`") == sp.Matrix([[1, 3], [2, 4]])
    assert convert("a`") == sp.Symbol("a", real=True)
    assert convert("a``") == sp.Symbol("a", real=True)
    assert convert("(x = 1)`") == Transposed(sp.Eq(sp.Symbol("x", real=True), 1))


def test_a_transpose_of_a_declared_nonscalar_is_held_as_one():
    nonscalar = Context(domains={"a": Domain(DomainKind.NONSCALAR)})
    matrix = sp.MatrixSymbol("a", DIMENSION, DIMENSION)
    assert convert("a`", nonscalar) == matrix.T
    assert convert("a``", nonscalar) == matrix


def test_a_dot_product_of_vectors_is_a_number():
    assert convert("[2, 3] . [4, 5]") == sp.Integer(23)


def test_a_dot_of_conforming_matrices_is_their_product():
    assert convert("[[1, 2], [3, 4]] . [[5], [6]]") == sp.Matrix([[17], [39]])
    assert convert("[[1], [2]] . [[3, 4]]") == sp.Matrix([[3, 4], [6, 8]])


def test_a_dot_of_shapes_that_will_not_multiply_stays_inert():
    assert type(convert("[[1, 2], [3, 4]] . [1, 2, 3]")).__name__ == "Dot"
    assert type(convert("a . b")).__name__ == "Dot"


def test_multiplying_two_vectors_is_their_dot_product():
    # Nothing else can be meant, and sympy will not multiply the shapes.
    assert convert("[2, 3] * [4, 5]") == sp.Integer(23)


def test_a_function_power_applies_then_raises():
    assert convert("SIN^2 x") == sp.sin(sp.Symbol("x", real=True)) ** 2


def test_a_negative_function_power_is_a_reciprocal():
    assert convert("SIN^-1 x") == 1 / sp.sin(sp.Symbol("x", real=True))


# -- relations, logic and vectors --------------------------------------------


def test_a_relation_is_assembled_and_not_decided():
    assert convert("2 = 2") == sp.Eq(2, 2, evaluate=False)
    assert convert("x /= 1").rel_op == "!="


def test_a_relation_chain_is_the_conjunction_it_means():
    """`1 <= a <= b` says two things about `a`, which is what the shipped
    library writes it for. Reading it as the grammar nests it would be
    comparing the truth value of the first link with `b`.

    The links are held in canonical form, a conjunction holding its operands in
    a set: `1 <= a` and `a >= 1` have to be one object or the order the pair
    comes back in would depend on which of the two spellings was written."""
    a, b, c = sp.symbols("a b c", real=True)
    assert convert("1 <= a <= b") == sp.And(sp.Ge(a, 1), sp.Le(a, b))
    assert convert("a >= 1 AND a <= b") == convert("1 <= a <= b")
    assert convert("a < b < c") == sp.And(sp.Lt(a, b), sp.Lt(b, c))
    assert convert("a = b = c") == sp.And(sp.Eq(a, b), sp.Eq(b, c))


def test_a_boolean_operator_on_booleans_is_boolean():
    assert convert("NOT NOT p") == sp.Symbol("p", real=True)
    assert convert("p AND q") == sp.And(*sp.symbols("p q", real=True))


@pytest.mark.parametrize(
    ("text", "expected"),
    [("3 OR 5", 7), ("NOT 5", -6), ("12 AND 10", 8), ("3 XOR 5", 6)],
    ids=str,
)
def test_a_boolean_operator_on_integers_is_bitwise(text, expected):
    assert convert(text) == sp.Integer(expected)


def test_a_flat_vector_is_a_row_and_a_nested_one_is_a_matrix():
    assert convert("[1, 2, 3]") == sp.Matrix(1, 3, [1, 2, 3])
    assert convert("[[1, 2], [3, 4]]") == sp.Matrix([[1, 2], [3, 4]])


def test_a_ragged_vector_stays_inert():
    assert isinstance(convert("[[1, 2], [3]]"), InertVector)


def test_a_vector_of_relations_is_a_container_and_not_a_matrix():
    assert isinstance(convert("[x = 1, y = 2]"), InertVector)


# -- definitions -------------------------------------------------------------


def test_an_assignment_converts_its_value_and_keeps_its_shape():
    assignment = convert("u := 2 + 3")
    assert isinstance(assignment, Assign)
    assert assignment.args[0] == sp.Symbol("u", real=True)
    assert assignment.args[2] == sp.Integer(5)


def test_a_function_definition_converts_its_body_only():
    definition = convert("F(x, y) := x + x")
    assert isinstance(definition, FunDef)
    assert definition.args[2] == 2 * sp.Symbol("x", real=True)


def test_a_domain_declaration_is_inert():
    assert isinstance(convert("x :epsilon Real"), Declare)


def test_showing_a_value_converts_the_child():
    assert convert("2 + 3 =") == sp.Integer(5)


# -- the function table ------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SQRT(9)", 3),
        ("EXP(0)", 1),
        ("LN(1)", 0),
        ("LOG(8, 2)", 3),
        ("SIGN(-3)", -1),
        ("STEP(2)", 1),
        ("FLOOR(5.73)", 5),
        ("FLOOR(7, 2)", 3),
        ("MOD(7, 3)", 1),
        ("MOD(7, 0)", 7),
        ("MODS(7, 3)", 1),
        ("GCD(12, 18, 27)", 3),
        ("LCM(4, 6)", 12),
        ("NUMERATOR(3/4)", 3),
        ("DENOMINATOR(3/4)", 4),
        ("MIN(2, 5)", 2),
        ("COMB(5, 2)", 10),
        ("PERM(5, 2)", 20),
        ("GAMMA(5)", 24),
        ("RE(3 + 2*#i)", 3),
        ("IM(3 + 2*#i)", 2),
        ("CONJ(3 + 2*#i)", sp.Integer(3) - 2 * sp.I),
        ("PHASE(-1)", sp.pi),
        ("SIN(pi/4)", sp.sqrt(2) / 2),
        ("ATAN(1, 1)", sp.pi / 4),
        ("COSH(0)", 1),
        ("DET([[1, 2], [3, 4]])", -2),
        ("TRACE([[1, 2], [3, 4]])", 5),
        ("DIMENSION([1, 2, 3])", 3),
        ("ELEMENT([1, 2, 3], 2)", 2),
        ("IDENTITY_MATRIX(2)", sp.eye(2)),
        ("CROSS([1, 0, 0], [0, 1, 0])", sp.Matrix(1, 3, [0, 0, 1])),
    ],
    ids=str,
)
def test_a_table_function_converts(text, expected):
    assert convert(text) == expected


def test_sign_of_zero_is_plus_or_minus_one():
    assert convert("SIGN(0)") == PlusMinus(1)


def test_a_calculus_head_converts_unevaluated():
    assert isinstance(convert("DIF(SIN(x), x)"), sp.Derivative)
    assert isinstance(convert("INT(x, x, 1, 2)"), sp.Integral)
    assert isinstance(convert("SUM(k, k, 1, 5)"), sp.Sum)
    assert isinstance(convert("PRODUCT(k, k, 1, 5)"), sp.Product)
    assert isinstance(convert("LIM(SIN(x)/x, x, 0)"), sp.Limit)
    # A series has no sympy head to be held in, so the engine has one, and it
    # waits where the rest of them wait.
    assert isinstance(convert("TAYLOR(SIN(x), x, 0, 3)"), Taylor)
    assert convert("TAYLOR(SIN(x), x, 0, 3)").doit() == convert("x - x^3/6")


def test_a_two_sided_limit_stays_two_sided_and_a_side_is_kept():
    assert convert("LIM(SIGN(x), x, 0)").args[3] == sp.Symbol("+-")
    assert convert("LIM(SIGN(x), x, 0, 1)").args[3] == sp.Symbol("+")
    assert convert("LIM(SIGN(x), x, 0, -1)").args[3] == sp.Symbol("-")


def test_a_conditional_converts_to_the_case_split_sympy_writes_it_as():
    """The way back from a `Piecewise`, which is what makes a result settle.

    Sympy answers a conditional integral with a `Piecewise` and the printer
    writes it as the `IF` the notation has; reading that back as anything else
    would leave a result whose written form changed on the next pass.
    """
    conditional = convert("IF(x > 0, 1, -1)")
    assert isinstance(conditional, sp.Piecewise)
    assert conditional == sp.Piecewise((1, sp.Symbol("x", real=True) > 0), (-1, True))
    assert convert("IF(x > 0, 1)") == sp.Piecewise((1, sp.Symbol("x", real=True) > 0))


def test_a_substitution_converts_back_from_the_vectors_it_is_written_as():
    # Sympy holds a derivative taken at a point as a `Subs`, and it is the head
    # that says the derivative inside is not to be evaluated.
    y = sp.Symbol("y", real=True)
    substitution = convert("SUBS(DIF(F(v), v), [v], [y])")
    assert isinstance(substitution, sp.Subs)
    assert substitution.point == (y,)


def test_a_hypergeometric_head_converts_back_from_the_vectors_it_is_written_as():
    assert convert("HYPER([1, 2], [3], x)") == sp.hyper(
        (1, 2), (3,), sp.Symbol("x", real=True)
    )


def test_a_sympy_head_the_printer_named_converts_back_to_that_head():
    """The inverse of writing a sympy function as its name upper-cased.

    A Bessel series comes back carrying heads the notation was never given a
    spelling for, and reading them as inert would leave the same mathematics
    under a different object - one that sorts by another name, so the answer
    would settle only on a second Simplify.
    """
    z = sp.Symbol("z", real=True)
    assert convert("BESSELI(1, z)") == sp.besseli(1, z)
    assert convert("EI(z)") == sp.Ei(z)
    assert convert("LOWERGAMMA(1, z)") == sp.lowergamma(1, z)


@pytest.mark.parametrize(
    "name", ["SOLVE", "FIT", "ITERATE", "TRUTH_TABLE", "RANDOM"], ids=str
)
def test_a_name_derive_defines_is_not_displaced_by_a_sympy_head(name):
    # The inventory has a reading of its own for each of these, and this call
    # is one none of them can take; nothing found among sympy's classes may
    # step in and answer it instead.
    assert isinstance(convert(f"{name}(x, 2)"), AppliedUndef)


def test_the_arbitrary_point_on_the_unit_circle_carries_no_assumptions():
    # Real is exactly what it is not: 1, -1, #i and -#i are all of them points
    # on it, so nothing that needs a real variable may fire on it.
    assert convert("unit_circle").is_real is None
    assert written("SQRT(unit_circle^2)") == "SQRT(unit_circle^2)"


def test_an_arbitrary_function_is_an_opaque_head_that_can_be_differentiated():
    x = sp.Symbol("x", real=True)
    call = sp.Function("F")(x)
    derivative = convert("DIF(F(x)^3, x)").doit()
    assert derivative == 3 * call**2 * sp.Derivative(call, x)


def test_a_table_function_given_arguments_it_cannot_take_stays_opaque():
    # Better an untouched call than a guess at what was meant.
    assert isinstance(convert("SIN(1, 2)"), AppliedUndef)
    assert written("SIN(1, 2)") == "SIN(1, 2)"


#: Everything the table does not cover must survive as an inert head.
OPAQUE = sorted(BUILTIN_FUNCTIONS - set(FUNCTIONS))


@pytest.mark.parametrize("name", OPAQUE)
def test_a_function_outside_the_table_is_opaque_and_round_trips(name):
    text = f"{name}(x, 2)"
    value = convert(text)
    assert isinstance(value, AppliedUndef)
    assert type(value).__name__ == name
    assert roundtrip(text) == (text, text)


# -- the angle mode ----------------------------------------------------------


def test_degree_mode_measures_trig_arguments_in_degrees():
    degrees = Context(angle=Angle.DEGREE)
    assert convert("SIN(45)", degrees) == sp.sqrt(2) / 2
    assert convert("ASIN(1/2)", degrees) == 30


def test_radian_mode_still_understands_the_degree_constant():
    assert convert("SIN(45 deg)") == sp.sqrt(2) / 2


# -- round trips -------------------------------------------------------------

#: One case per node kind, so that nothing the parser can build is lost on the
#: way through sympy. `PARAMS` and `INTERVAL` ride along inside the definition
#: and the declaration that own them.
KINDS = [
    "2/3",
    "x",
    '"note"',
    "#3",
    "?",
    "x - y + 2",
    "2*x*y",
    "x/y",
    "x^2",
    "a . b",
    "-x",
    "±x",
    "x!",
    "a`",
    "x SUB 1",
    "|x|",
    "ABS(x)",
    "SIN(x)",
    "SIN x",
    "LN(x)",
    "SIN^2 x",
    "[1, 2, 3]",
    "x = 1",
    "NOT p",
    "p AND q",
    "p OR q",
    "p XOR q",
    "p IMP q",
    "u := 5",
    "F(x, y) := x + y",
    "x :epsilon Real",
    "x :epsilon Real [0, inf)",
    "x + 1 =",
]


@pytest.mark.parametrize("text", KINDS)
def test_a_construct_survives_the_round_trip(text):
    first, second = roundtrip(text)
    assert first == second


#: Expressions already in the form the engine writes. These must come back
#: character for character: what sympy has nothing to do to, it must not
#: change. Anything sympy does evaluate belongs in the cases above instead.
UNCHANGED = [
    "x",
    "-x",
    "x^2",
    "x + y",
    "x - y",
    "2*x",
    "x/y",
    "1/x",
    "x^2*y",
    "SQRT(x)",
    "#e^x",
    "LN(x)",
    "SIN(x)",
    "ABS(x)",
    "x!",
    "x SUB 1",
    "x SUB (i + 1)",
    "(x + 1) SUB 2",
    "(x = 1)`",
    "a . b",
    "±x",
    '"note"',
    "?",
    "inf",
    "-inf",
    "pi",
    "#i",
    "euler_gamma",
    "[1, 2, 3]",
    "[[1, 2], [3, 4]]",
    "[]",
    "x = 1",
    "x /= 1",
    "x <= 1",
    "NOT p",
    "p AND q",
    "p IMP q",
    "u := 5",
    "F(x, y) := x + y",
    "x :epsilon Real",
    "x :epsilon Integer [0, inf)",
    "DIF(F(x), x)",
    "DIF(F(x), x, 2)",
    "INT(F(x), x)",
    "INT(F(x), x, 1, 2)",
    "SUM(F(k), k, 1, n)",
    "PRODUCT(F(k), k, 1, n)",
    "LIM(F(x), x, 0)",
    "LIM(F(x), x, 0, 1)",
    "TAYLOR(F(x), x, 0, 3)",
    "SOLVE(x^2 = 1, x)",
    "IF(x > 0, 1, -1)",
    "IF(x > 0, 1)",
    "HYPER([1, 2], [3], x)",
    "MY_FUNCTION(x, 2)",
]


@pytest.mark.parametrize("text", UNCHANGED)
def test_an_expression_already_in_engine_form_comes_back_unchanged(text):
    assert written(text) == text


def test_every_node_kind_is_covered_by_a_round_trip_case():
    covered = set()
    for text in KINDS:
        stack = [parse(text)]
        while stack:
            node = stack.pop()
            covered.add(node.kind)
            stack.extend(node.children)
    assert covered == set(Kind)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1/0", "±inf"),
        ("0/0", "?"),
        ("inf - inf", "?"),
        ("SIGN(0)", "±1"),
        ("-inf", "-inf"),
        ("x :epsilon Real [0, inf)", "x :epsilon Real [0, inf)"),
        ("F(x) :=", "F(x) :="),
        ("u :==", "u :=="),
        ("[[1, 2], [3]]", "[[1, 2], [3]]"),
        ("SOLVE(x^2 = 1, x)", "SOLVE(x^2 = 1, x)"),
        # Four arguments, the last of them the value where the test cannot be
        # decided: no `Piecewise` holds that, and the head stays inert.
        ("IF(x > 0, 1, -1, 0)", "IF(x > 0, 1, -1, 0)"),
    ],
    ids=str,
)
def test_what_a_construct_is_written_as(text, expected):
    assert written(text) == expected


def test_a_result_carries_a_tree_whose_spans_index_its_text():
    result = from_sympy(convert("(x + 1)*(x - 1)"))
    assert result.text[result.node.start : result.node.end] == result.text
    assert to_sexpr(result.node) == to_sexpr(parse(result.text))
