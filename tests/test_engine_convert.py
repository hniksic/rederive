"""Conversion from nodes to sympy and back.

These tests check the translation, not the mathematics. A case asserts either which
sympy object a construct becomes, or that the text survives the round trip (text -> tree
-> sympy -> text). Every command depends on the round trip, and it is the only thing that
preserves constructs the engine doesn't understand.
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
    PlusMinus,
    StringLiteral,
    Subscript,
    Taylor,
    Transposed,
    Vector,
)
from rederive.model.expr import Kind
from rederive.syntax import (
    DeriveSyntaxError,
    FunctionDeclaration,
    ParseState,
    parse_expression,
)
from rederive.syntax.names import BUILTIN_FUNCTIONS


def parse(text, state=None):
    return parse_expression(text, state or ParseState()).node


def declaring(*names):
    """A parse state in which each of `names` is declared a function.

    Upper case alone doesn't make a name a function: an undefined name before a
    parenthesis is a product (in Character mode, of its letters).
    """
    state = ParseState()
    for name in names:
        state.declare(FunctionDeclaration(name))
    return state


def convert(text, context=None, state=None):
    """The sympy object `text` converts to."""
    return to_sympy(parse(text, state), context or Context())


def written(text, context=None, state=None):
    """`text` converted and written back out."""
    context = context or Context()
    return from_sympy(convert(text, context, state), context).text


def roundtrip(text, context=None, state=None):
    """The text after one pass through the engine, and after two.

    Equal values mean the printer's output is a fixed point, so a result can safely be
    stored in the worksheet and read back.
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
    """An approximate number becomes the simplest rational the precision allows.

    One tenth is exact. With six digits, `123456789/10000` is indistinguishable from
    `308642/25`, so approximate mode reads it as the latter, matching the original. This
    rounding on input is what makes approximate arithmetic approximate.
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
    assert convert("[[1, 2], [3, 4]] SUB 2") == Vector(3, 4)


def test_a_subscript_reaches_through_every_dimension_there_is():
    """`SUB` is another spelling of `ELEMENT`, so it reaches equally deep.

    This includes a vector of matrices. Each index in the chain selects from the result
    of the previous one."""
    assert convert("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]] SUB 2 SUB 1 SUB 2") == 6
    assert convert("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]] SUB [2, 1, 2]") == 6


def test_a_subscript_on_a_vector_with_no_index_yet_keeps_the_access():
    """A variable index defers the access.

    A generated `VECTOR` body relies on this: the body is converted while the index is
    still a variable, and re-evaluated once a number is substituted."""
    held = convert("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]] SUB i")
    assert isinstance(held, AppliedUndef)
    assert type(held).__name__ == "ELEMENT"


def test_a_transpose_of_a_matrix_is_a_matrix_and_of_a_scalar_is_the_scalar():
    # 8.5 p.205: the transpose of a scalar is the scalar, and anything not declared
    # nonscalar is a scalar. A non-expression such as an equation keeps the head.
    assert convert("[[1, 2], [3, 4]]`") == Vector(Vector(1, 3), Vector(2, 4))
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
    assert convert("[[1, 2], [3, 4]] . [[5], [6]]") == Vector(Vector(17), Vector(39))
    assert convert("[[1], [2]] . [[3, 4]]") == Vector(Vector(3, 4), Vector(6, 8))
    # A one-row matrix is a matrix, so its product with a vector is a vector.
    assert convert("[[1, 2]] . [3, 4]") == Vector(11)


def test_a_dot_of_shapes_that_will_not_multiply_stays_inert():
    assert type(convert("[[1, 2], [3, 4]] . [1, 2, 3]")).__name__ == "Dot"
    assert type(convert("a . [1, 2]")).__name__ == "Dot"


def test_a_dot_of_scalars_is_their_product():
    # Anything not declared nonscalar is a scalar (section 8.5).
    assert convert("a . b") == convert("a*b")
    assert convert("SIN(x) . 2") == convert("2*SIN(x)")


def test_multiplying_two_vectors_is_their_dot_product():
    # Nothing else makes sense, and sympy won't multiply these shapes.
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
    """`1 <= a <= b` constrains `a` on both sides, as the shipped library intends.
    Nesting it as the grammar does would compare the truth value of `1 <= a` with `b`.

    The links are canonicalized, because a conjunction stores its operands in a set:
    `1 <= a` and `a >= 1` must be the same object, or the order of the pair would depend
    on which spelling was written."""
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


def test_a_vector_is_its_elements_and_a_matrix_a_vector_of_rows():
    assert convert("[1, 2, 3]") == Vector(1, 2, 3)
    assert convert("[[1, 2], [3, 4]]") == Vector(Vector(1, 2), Vector(3, 4))
    # A one-row matrix is not the vector it holds.
    assert convert("[[1, 2]]") == Vector(Vector(1, 2))
    assert convert("[]") == Vector()


def test_a_vector_of_matrices_keeps_every_dimension():
    """Tables of Christoffel symbols need the third dimension."""
    vector = convert("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]")
    assert vector.args == (
        Vector(Vector(1, 2), Vector(3, 4)),
        Vector(Vector(5, 6), Vector(7, 8)),
    )


def test_a_ragged_vector_or_one_of_relations_is_a_vector_like_any_other():
    assert convert("[[1, 2], [3]]") == Vector(Vector(1, 2), Vector(3))
    x, y = sp.symbols("x y", real=True)
    assert convert("[x = 1, y = 2]") == Vector(sp.Eq(x, 1), sp.Eq(y, 2))


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
        ("ELEMENT([[2, 3, 5], [7, 1, 4]], 2, 3)", 4),
        ("ELEMENT([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 2, 1, 2)", 6),
        ("IDENTITY_MATRIX(2)", Vector(Vector(1, 0), Vector(0, 1))),
        ("CROSS([1, 0, 0], [0, 1, 0])", Vector(0, 0, 1)),
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
    # Sympy has no unevaluated series head, so the engine provides one.
    assert isinstance(convert("TAYLOR(SIN(x), x, 0, 3)"), Taylor)
    assert convert("TAYLOR(SIN(x), x, 0, 3)").doit() == convert("x - x^3/6")


def test_a_two_sided_limit_stays_two_sided_and_a_side_is_kept():
    assert convert("LIM(SIGN(x), x, 0)").args[3] == sp.Symbol("+-")
    assert convert("LIM(SIGN(x), x, 0, 1)").args[3] == sp.Symbol("+")
    assert convert("LIM(SIGN(x), x, 0, -1)").args[3] == sp.Symbol("-")


def test_a_conditional_converts_to_the_case_split_sympy_writes_it_as():
    """`IF` converts to `Piecewise`, so printed results round-trip.

    Sympy answers conditional integrals with a `Piecewise`, which the printer writes as
    `IF`. Reading that back as anything else would change the written form on the next
    pass.
    """
    conditional = convert("IF(x > 0, 1, -1)")
    assert isinstance(conditional, sp.Piecewise)
    assert conditional == sp.Piecewise((1, sp.Symbol("x", real=True) > 0), (-1, True))
    assert convert("IF(x > 0, 1)") == sp.Piecewise((1, sp.Symbol("x", real=True) > 0))


def test_a_substitution_converts_back_from_the_vectors_it_is_written_as():
    # Sympy represents a derivative at a point as a `Subs`, which keeps the inner
    # derivative unevaluated.
    y = sp.Symbol("y", real=True)
    substitution = convert("SUBS(DIF(F(v), v), [v], [y])", state=declaring("F"))
    assert isinstance(substitution, sp.Subs)
    assert substitution.point == (y,)


@pytest.mark.parametrize(
    ("text", "letters"),
    [
        ("HYPER(x)", "h*y*p*e*r*x"),
        ("MEIJERG(x)", "m*EI(j)*e*r*g*x"),
        ("BESSELI(z)", "b*e*s*s*e*l*i*z"),
        ("LOWERGAMMA(1, z)", "l*o*w*e*r*GAMMA(1, z)"),
        ("LI(x)", "l*i*x"),
    ],
    ids=str,
)
def test_a_sympy_class_name_is_no_function_name(text, letters):
    """Function names come from the notation, not from sympy's exported classes.

    Sympy's hypergeometric and Bessel functions, incomplete gammas and logarithmic
    integral have no name here, so their sympy names are read like any undefined name:
    letter by letter, with known functions found among the letters as in any word. The
    logarithmic integral is written `EI(LN(x))`.
    """
    converted = convert(text)
    assert converted == convert(letters)
    assert not converted.atoms(
        sp.hyper, sp.meijerg, sp.besseli, sp.lowergamma, sp.li
    )


@pytest.mark.parametrize(
    "text",
    [
        "HYPER([1, 2], [3], x)",
        "MEIJERG([[1, 1], []], [[1], [0]], x)",
        "BESSELI(1, z)",
    ],
    ids=str,
)
def test_a_sympy_class_name_takes_no_arguments(text):
    # An undefined name isn't a function, and an argument list is only a valid factor
    # when it holds one argument.
    with pytest.raises(DeriveSyntaxError):
        parse(text)


def test_an_interval_converts_back_to_the_bounds_a_limit_answered_with():
    """`INTERVAL` converts back to sympy's `AccumBounds`.

    Sympy computes with the object itself, so reading `INTERVAL(-1, 1)` as anything else
    would print the same but behave differently in arithmetic.
    """
    assert convert("INTERVAL(-1, 1)") == sp.AccumBounds(-1, 1)
    assert roundtrip("INTERVAL(-1, 1)") == ("INTERVAL(-1, 1)", "INTERVAL(-1, 1)")


@pytest.mark.parametrize(
    "text",
    ["INTERVAL(1)", "INTERVAL(-1, 0, 1)", "INTERVAL(3, 1)", "INTERVAL(x, 2)"],
    ids=str,
)
def test_an_interval_whose_bounds_are_no_bounds_stays_opaque(text):
    """An interval needs exactly two bounds, in increasing order.

    Sympy's constructor can't always check this: nothing says whether `x` is below two,
    and a possibly reversed range is not a valid one. Each case stays the call it was
    written as.
    """
    assert isinstance(convert(text), AppliedUndef)


def test_a_sum_over_the_roots_of_a_polynomial_converts_back_from_its_bound_form():
    """`ROOT_SUM` converts back to a `RootSum`, whose summand is a `Lambda`.

    The notation has no lambda; like every binding head, it names the bound variable as
    the second argument. Reading back must rebuild the `RootSum` itself, since a sum
    reassembled from its parts is a different object that prints differently.
    """
    x, t = sp.Symbol("x", real=True), sp.Symbol("t", real=True)
    summed = convert("ROOT_SUM(t^3 + t + 1, t, t*LN(x - t))")
    assert summed == sp.RootSum(t**3 + t + 1, sp.Lambda(t, t * sp.log(x - t)), t)
    assert roundtrip("ROOT_SUM(t^3 + t + 1, t, t*LN(x - t))") == (
        "ROOT_SUM(t^3 + t + 1, t, t*LN(x - t))",
        "ROOT_SUM(t^3 + t + 1, t, t*LN(x - t))",
    )


def test_a_single_root_of_a_polynomial_carries_the_index_that_picks_it():
    """`ROOT_OF(p, t, n)` maps to sympy's `CRootOf`, which holds the polynomial and index.

    The generator isn't an argument of the class, so the printer reads it from the
    polynomial. The index keeps different roots of the same quintic distinct.
    """
    t = sp.Symbol("t", real=True)
    assert convert("ROOT_OF(t^5 - t - 1, t, 0)") == sp.CRootOf(t**5 - t - 1, t, 0)
    assert convert("ROOT_OF(t^5 - t - 1, t, 1)") != convert("ROOT_OF(t^5 - t - 1, t, 0)")
    assert roundtrip("ROOT_OF(t^5 - t - 1, t, 0)") == (
        "ROOT_OF(t^5 - t - 1, t, 0)",
        "ROOT_OF(t^5 - t - 1, t, 0)",
    )


@pytest.mark.parametrize(
    "text",
    [
        "ROOT_SUM(x, 2)",
        "ROOT_SUM(SIN(t), t, t)",
        "ROOT_SUM(2, t, t)",
        "ROOT_SUM(t^3 + u, t, t*LN(x - t))",
        "ROOT_SUM(t^3 + t + 1, 2, t)",
        "ROOT_OF(x, 2)",
        "ROOT_OF(t^5 - t - 1, t, 1/2)",
        "ROOT_OF(t^5 - t - 1, t, 9)",
    ],
    ids=str,
)
def test_a_root_head_given_arguments_it_cannot_take_stays_opaque(text):
    """None of these denote roots, so each stays the call it was written as.

    A sine and a constant aren't polynomials, a polynomial in two variables has no roots
    until one is chosen, a half is not an index, and a quintic has no ninth root. The
    inventory's totality test requires every name to behave this way.
    """
    assert isinstance(convert(text), AppliedUndef)


def test_the_hyperbolic_integrals_are_named_rather_than_upper_cased():
    """Sympy's `Chi` and `Shi` are named `COSH_INT` and `SINH_INT`.

    Upper-casing `Chi` would give `CHI`, which in Derive is the chi-square distribution,
    silently conflating two functions. Derive had no cosh-integral, so `COSH_INT` is free.
    """
    x = sp.Symbol("x", real=True)
    assert convert("COSH_INT(x)") == sp.Chi(x)
    assert convert("SINH_INT(x)") == sp.Shi(x)
    assert roundtrip("COSH_INT(x)") == ("COSH_INT(x)", "COSH_INT(x)")
    assert roundtrip("SINH_INT(x)") == ("SINH_INT(x)", "SINH_INT(x)")
    # `CHI` is still the distribution.
    assert convert("CHI(x)") == sp.sign(x) / 2 - sp.sign(x - 1) / 2


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("EI(x)", sp.Ei(sp.Symbol("x", real=True))),
        ("SI(x)", sp.Si(sp.Symbol("x", real=True))),
        ("CI(x)", sp.Ci(sp.Symbol("x", real=True))),
        ("ERFI(x)", sp.erfi(sp.Symbol("x", real=True))),
        ("DIGAMMA(x)", sp.digamma(sp.Symbol("x", real=True))),
        ("POLYGAMMA(1, x)", sp.polygamma(1, sp.Symbol("x", real=True))),
        ("POLYLOG(2, x)", sp.polylog(2, sp.Symbol("x", real=True))),
        ("LAMBERTW(x)", sp.LambertW(sp.Symbol("x", real=True))),
    ],
    ids=str,
)
def test_a_special_function_of_rederives_own_converts_and_reads_back(text, expected):
    assert convert(text) == expected
    assert roundtrip(text) == (text, text)


def test_the_polygamma_function_of_order_zero_is_the_digamma_function():
    # Sympy has no separate digamma class, so the two spellings are the same object.
    assert convert("POLYGAMMA(0, x)") == convert("DIGAMMA(x)")
    assert written("POLYGAMMA(0, x)") == "DIGAMMA(x)"


@pytest.mark.parametrize(
    "name", ["SOLVE", "FIT", "ITERATE", "TRUTH_TABLE", "RANDOM"], ids=str
)
def test_a_name_derive_defines_is_not_displaced_by_a_sympy_head(name):
    # Each of these has its own reading in the inventory that rejects this call; no sympy
    # class of the same name may answer it instead.
    assert isinstance(convert(f"{name}(x, 2)"), AppliedUndef)


def test_the_arbitrary_point_on_the_unit_circle_carries_no_assumptions():
    # It isn't real: 1, -1, #i and -#i are all on the circle, so no rule that needs a
    # real variable may fire.
    assert convert("unit_circle").is_real is None
    assert written("SQRT(unit_circle^2)") == "SQRT(unit_circle^2)"


def test_an_arbitrary_function_is_an_opaque_head_that_can_be_differentiated():
    x = sp.Symbol("x", real=True)
    call = sp.Function("F")(x)
    derivative = convert("DIF(F(x)^3, x)", state=declaring("F")).doit()
    assert derivative == 3 * call**2 * sp.Derivative(call, x)


def test_a_table_function_given_arguments_it_cannot_take_stays_opaque():
    # Better an untouched call than a guess.
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

#: One case per node kind, so nothing the parser builds is lost through sympy. `PARAMS`
#: and `INTERVAL` are covered inside the definition and declaration that contain them.
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


#: Expressions already in engine form, which must come back character for character.
#: Anything sympy evaluates belongs in the cases above instead.
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
    "a . [1, 2]",
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
    "F'(x)",
    "F''(x)",
    "DIF(F(x, y), x)",
    "INT(F(x), x)",
    "INT(F(x), x, 1, 2)",
    "SUM(F(k), k, 1, n)",
    "PRODUCT(F(k), k, 1, n)",
    "LIM(F(x), x, 0)",
    "LIM(F(x), x, 0, 1)",
    "TAYLOR(F(x), x, 0, 3)",
    "INTERVAL(-1, 1)",
    "ROOT_SUM(t^3 + t + 1, t, t*LN(x - t))",
    "ROOT_OF(t^5 - t - 1, t, 0)",
    "SOLVE(x^2 = 1, x)",
    "IF(x > 0, 1, -1)",
    "IF(x > 0, 1)",
    "MY_FUNCTION(x, 2)",
]


@pytest.mark.parametrize("text", UNCHANGED)
def test_an_expression_already_in_engine_form_comes_back_unchanged(text):
    # Declare the functions the cases call, as an author would.
    assert written(text, state=declaring("F", "MY_FUNCTION")) == text


#: Quotients with a sum's integer content in front, as the normal form writes them. The
#: reader multiplies a number into a sum it stands alone beside, and the printer takes
#: it out again.
CONTENT_IN_FRONT = [
    "2*(x + 1)/y",
    "-2*(x + 2)/(x^2 + 1)",
    "2*(2*x + 3)/(3*(x^2 + 1))",
    "1/(2*(x + 1))",
    "-1/(2*(x + 1))",
    "(x + 1)/(2*(x^2 + 2))",
    "3/(2*(2*x + 1))",
]


@pytest.mark.parametrize("text", CONTENT_IN_FRONT)
def test_a_sum_s_content_in_front_survives_the_round_trip(text):
    assert roundtrip(text) == (text, text)


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
        # The fourth argument is the value when the test is undecidable. `Piecewise`
        # can't express that, so the head stays inert.
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
