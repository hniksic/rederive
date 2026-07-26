"""The evaluation state, and what a declaration is worth.

A domain declaration is the whole of the "don't guess" rule: a rewrite fires
because a declaration justifies it, so what matters here is which sympy
assumption each declaration turns into, and that an undeclared variable is
real.
"""

from __future__ import annotations

import pytest
import sympy as sp

from rederive.engine import Context, Domain, DomainKind, domain_of_node, to_sympy
from rederive.engine.context import Angle, Branch, Direction, Precision, TrigPower
from rederive.model.settings import Settings
from rederive.syntax import DomainDeclaration, ParseState, parse_expression


def parse(text, state=None):
    return parse_expression(text, state or ParseState()).node


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


def test_a_setting_no_dialog_owns_yet_keeps_its_factory_value():
    assert Context.from_settings(Settings()).angle is Angle.RADIAN


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
