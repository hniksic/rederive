"""The evaluation state, what a declaration is worth, and what a name stands for.

A domain declaration is the whole of the "don't guess" rule: a rewrite fires
because a declaration justifies it, so what matters here is which sympy
assumption each declaration turns into, and that an undeclared variable is
real.

The substitution pre-pass is the other half of the state: assignments, function
definitions and labels are written into the tree before anything converts it,
and the cases below are about where that stops - a variable that reaches
itself, a call with fewer arguments than parameters, a name a definition owns.
"""

from __future__ import annotations

import pytest
import sympy as sp
from sexpr import to_sexpr

from rederive.engine.computing import (
    Context,
    Domain,
    DomainKind,
    domain_of_node,
    substitute,
    to_sympy,
)
from rederive.engine.context import Angle, Branch, Direction, Precision, TrigPower
from rederive.model.settings import Settings
from rederive.syntax import (
    DomainDeclaration,
    ParseState,
    VariableDeclaration,
    parse_expression,
)


def parse(text, state=None):
    return parse_expression(text, state or ParseState()).node


def written(text, context, state=None):
    """`text` with everything the context knows substituted into it."""
    return to_sexpr(substitute(parse(text, state), context))


def same_as(text, state=None):
    """What `written` should equal when the answer is `text` itself."""
    return to_sexpr(parse(text, state))


def symbol(declaration):
    """The symbol a declaration declares, as the converter builds it."""
    name, domain = domain_of_node(parse(declaration))
    return to_sympy(parse(name), Context(domains={name: domain}))


# -- factory values ----------------------------------------------------------


def test_the_factory_context():
    context = Context()
    assert context.precision is Precision.EXACT
    assert context.precision_digits == 6
    assert context.branch is Branch.PRINCIPAL
    assert context.exponential is Direction.AUTO
    assert context.logarithm is Direction.AUTO
    assert context.trigonometry is Direction.AUTO
    assert context.trigpower is TrigPower.AUTO
    assert context.angle is Angle.RADIAN
    assert context.input_base == 10
    assert not context.domains and not context.assignments
    assert not context.functions and not context.labels


def test_an_undeclared_variable_is_real():
    assert Context().domain("q") == Domain(DomainKind.REAL)
    assert to_sympy(parse("q")).is_real


def test_approximating_changes_the_precision_and_nothing_else():
    context = Context(input_base=8, precision_digits=6)
    approximate = context.with_precision(Precision.APPROXIMATE, 20)
    assert approximate.precision is Precision.APPROXIMATE
    assert approximate.precision_digits == 20
    assert approximate.input_base == 8
    assert context.precision is Precision.EXACT


# -- from the session --------------------------------------------------------


def test_a_context_is_built_from_the_settings_store():
    settings = Settings()
    settings.apply({"Precision": "Approximate", "PrecisionDigits": 12, "InputBase": 16})
    context = Context.from_settings(settings)
    assert context.precision is Precision.APPROXIMATE
    assert context.precision_digits == 12
    assert context.input_base == 16


def test_the_manage_settings_reach_the_context():
    settings = Settings()
    assert Context.from_settings(settings).angle is Angle.RADIAN
    settings.apply(
        {
            "Branch": "Real",
            "Exponential": "Collect",
            "Logarithm": "Expand",
            "Trigonometry": "Expand",
            "Trigpower": "Sines",
            "Angle": "Degree",
        }
    )
    context = Context.from_settings(settings)
    assert context.branch is Branch.REAL
    assert context.exponential is Direction.COLLECT
    assert context.logarithm is Direction.EXPAND
    assert context.trigonometry is Direction.EXPAND
    assert context.trigpower is TrigPower.SINES
    assert context.angle is Angle.DEGREE


def test_declarations_become_domains():
    context = Context.from_settings(
        Settings(), [DomainDeclaration("k", "Integer"), DomainDeclaration("z", "Complex")]
    )
    assert context.domain("k").kind is DomainKind.INTEGER
    assert context.domain("z").kind is DomainKind.COMPLEX


def test_a_domain_given_with_an_interval_outranks_the_declaration():
    name, domain = domain_of_node(parse("x :epsilon Real (0, inf)"))
    context = Context.from_settings(
        Settings(), [DomainDeclaration(name, "Real")], domains={name: domain}
    )
    assert context.domain("x").low is not None


# -- reading a declaration ---------------------------------------------------


def test_a_declaration_without_an_interval():
    declared = domain_of_node(parse("x :epsilon Integer"))
    assert declared == ("x", Domain(DomainKind.INTEGER))


def test_a_declaration_with_an_interval_keeps_its_brackets():
    name, domain = domain_of_node(parse("x :epsilon Real [0, inf)"))
    assert name == "x"
    assert domain.kind is DomainKind.REAL
    assert domain.closed_low and not domain.closed_high
    assert domain.low.value == "0"


def test_an_unrecognised_domain_is_not_an_error():
    _, domain = domain_of_node(parse("x :epsilon Fictional"))
    assert domain.kind is DomainKind.UNKNOWN


# -- declarations as assumptions ---------------------------------------------


@pytest.mark.parametrize(
    ("declaration", "facts"),
    [
        ("x :epsilon Real", {"is_real": True}),
        ("x :epsilon Integer", {"is_integer": True}),
        ("x :epsilon Complex", {"is_real": None}),
        ("x :epsilon Real (0, inf)", {"is_positive": True}),
        ("x :epsilon Real [0, inf)", {"is_nonnegative": True, "is_positive": None}),
        ("x :epsilon Real (-inf, 0)", {"is_negative": True}),
        ("x :epsilon Real (-inf, 0]", {"is_nonpositive": True, "is_negative": None}),
        ("x :epsilon Real [2, 5]", {"is_positive": True}),
        ("x :epsilon Integer [1, inf)", {"is_integer": True, "is_positive": True}),
        ("x :epsilon Real (-inf, inf)", {"is_real": True, "is_positive": None}),
    ],
    ids=str,
)
def test_what_a_declaration_tells_sympy(declaration, facts):
    value = symbol(declaration)
    for fact, expected in facts.items():
        assert getattr(value, fact) is expected, fact


def test_a_nonscalar_variable_does_not_commute():
    assert not symbol("m :epsilon Nonscalar").is_commutative


def test_a_one_point_interval_is_a_value():
    assert symbol("x :epsilon Real [7, 7]") == sp.Integer(7)
    name, domain = domain_of_node(parse("x :epsilon Real [7, 7]"))
    assert to_sympy(parse("x + 1"), Context(domains={name: domain})) == sp.Integer(8)


def test_an_open_bound_is_still_a_bound_at_infinity():
    # No variable is infinite, so `inf` as a bound says only which side.
    assert symbol("x :epsilon Real (0, inf)").is_positive


def test_a_declaration_only_reaches_the_variable_it_names():
    name, domain = domain_of_node(parse("x :epsilon Real (0, inf)"))
    context = Context(domains={name: domain})
    assert to_sympy(parse("x"), context).is_positive
    assert to_sympy(parse("y"), context).is_positive is None


def test_a_domain_decides_whether_a_rewrite_is_allowed():
    # The whole point of the declarations: no domain, no rewrite.
    unknown = Context(domains={"x": Domain(DomainKind.COMPLEX)})
    name, domain = domain_of_node(parse("x :epsilon Real (0, inf)"))
    real = to_sympy(parse("SQRT(x^2)"))
    complex_x = to_sympy(parse("SQRT(x^2)"), unknown)
    positive = to_sympy(parse("SQRT(x^2)"), Context(domains={name: domain}))
    assert real == sp.Abs(sp.Symbol("x", real=True))
    assert complex_x == sp.sqrt(sp.Symbol("x", complex=True) ** 2)
    assert positive == sp.Symbol("x", positive=True)


# -- substitution: assignments -----------------------------------------------


def test_an_assigned_variable_is_replaced_by_its_value():
    context = Context(assignments={"u": parse("2*w")})
    assert written("u + 1", context) == same_as("2*w + 1")


def test_an_assignment_is_followed_as_far_as_it_goes():
    context = Context(assignments={"u": parse("v"), "v": parse("3")})
    assert written("u", context) == same_as("3")


def test_a_variable_that_reaches_itself_is_expanded_once_and_left():
    # `pn := pn + 1` is a legitimate thing to author. It must not spin, and it
    # must not silently drop the name it could not expand again.
    state = ParseState()
    state.declare(VariableDeclaration("pn", True))
    context = Context(assignments={"pn": parse("pn + 1", state)})
    assert written("pn", context, state) == same_as("pn + 1", state)


def test_two_variables_that_reach_each_other_stop():
    context = Context(assignments={"a": parse("b"), "b": parse("a")})
    assert written("a", context) == same_as("a")


def test_an_assignment_keeps_the_name_it_defines():
    # Otherwise `u := u + 1` could not be written a second time.
    context = Context(assignments={"u": parse("5")})
    assert written("u := u + 1", context) == same_as("u := 5 + 1")


def test_a_declaration_keeps_the_variable_it_declares():
    context = Context(assignments={"x": parse("5")})
    assert written("x :epsilon Real", context) == same_as("x :epsilon Real")


# -- substitution: function definitions ---------------------------------------


def test_a_call_becomes_the_body_with_its_arguments_written_in():
    context = Context(functions={"F": (("x", "y"), parse("x^2 + y"))})
    assert written("F(a, 3)", context) == same_as("a^2 + 3")


def test_a_call_with_fewer_arguments_substitutes_the_leading_parameters():
    # Derive's partial application: `ACCELERATION(6)` is `6/m`.
    context = Context(functions={"ACCELERATION": (("f", "m"), parse("f/m"))})
    assert written("ACCELERATION(6)", context) == same_as("6/m")


def test_an_argument_is_substituted_before_it_is_written_in():
    context = Context(
        assignments={"u": parse("7")}, functions={"F": (("x",), parse("x + 1"))}
    )
    assert written("F(u)", context) == same_as("7 + 1")


def test_a_recursive_definition_leaves_its_own_call_standing():
    context = Context(functions={"G": (("x",), parse("G(x) + 1"))})
    assert written("G(2)", context) == same_as("G(2) + 1")


def test_a_parameter_stands_for_itself_inside_the_definition():
    # The body of `F(u) := u + 1` is about the parameter, not about whatever
    # `u` happens to be assigned elsewhere.
    context = Context(assignments={"u": parse("7")})
    assert written("F(u) := u + 1", context) == same_as("F(u) := u + 1")


def test_a_function_the_context_does_not_know_is_left_alone():
    context = Context(assignments={"u": parse("7")})
    assert written("H(u)", context) == same_as("H(7)")


# -- substitution: bound variables --------------------------------------------


def test_the_variable_a_sum_names_is_not_substituted_for():
    # `SUM(25, 5, 1, 3)` is not a sum over anything, and it is what writing an
    # assigned value into every position would produce.
    context = Context(assignments={"x": parse("5")})
    assert written("SUM(x^2, x, 1, 3)", context) == same_as("SUM(x^2, x, 1, 3)")


def test_an_argument_after_the_bound_variable_is_read_outside_the_binding():
    # `ITERATES(TAN(x), x, x, -1)` is from the manual: the third argument is the
    # starting value, and the `x` in it is whatever `x` means outside.
    context = Context(assignments={"x": parse("5")})
    assert written("SUM(x^2, x, 1, x)", context) == same_as("SUM(x^2, x, 1, 5)")


def test_a_bound_variable_shadows_a_parameter_of_the_same_name():
    context = Context(functions={"F": (("x",), parse("SUM(x^2, x, 1, 3)"))})
    assert written("F(9)", context) == same_as("SUM(x^2, x, 1, 3)")


def test_a_binding_function_given_no_variable_binds_nothing():
    context = Context(assignments={"v": parse("[1, 2, 3]")})
    assert written("SUM(v)", context) == same_as("SUM([1, 2, 3])")


def test_a_vector_of_names_binds_every_one_of_them():
    # An iterated limit ranges over each of them, so an assigned `x` reaches
    # neither the vector nor the expression under it.
    context = Context(assignments={"x": parse("5")})
    assert written("LIM(x + y, [x, y], [1, 2])", context) == same_as(
        "LIM(x + y, [x, y], [1, 2])"
    )


def test_a_parameter_supplied_with_a_vector_of_names_is_written_in():
    # SOLVE.MTH's `NEWTON_AUX(a, x, x0, n)` limits over its parameter `x`, and
    # `NEWTONS` calls it with the vector of variables to solve for. Left
    # standing, the limit would be taken in a variable nobody named. The body
    # takes the vector too, a definition being textual substitution throughout.
    context = Context(functions={"F": (("x",), parse("LIM(x*y, x, 1)"))})
    assert written("F([u, v])", context) == same_as("LIM([u, v]*y, [u, v], 1)")


# -- substitution: labels -----------------------------------------------------


def test_a_label_is_replaced_by_the_expression_it_names():
    context = Context(labels={3: parse("x + 1")})
    assert written("#3*2", context) == same_as("(x + 1)*2")


def test_a_label_nobody_knows_stays_a_label():
    assert written("#3 + 1", Context()) == same_as("#3 + 1")


def test_a_label_that_reaches_itself_stops():
    context = Context(labels={1: parse("#1 + 1")})
    assert written("#1", context) == same_as("#1 + 1")


def test_substitution_leaves_a_tree_it_has_nothing_to_do_to_alone():
    assert written("x + SIN(y)", Context()) == same_as("x + SIN(y)")
