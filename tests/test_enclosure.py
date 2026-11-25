"""The interval enclosures behind NSOLVE's finding that an equation has no root: each
must hold the value it encloses, which a 60-digit evaluation checks."""

from __future__ import annotations

import math

import mpmath
import pytest
import sympy as sp
from simplifying import simp

from rederive.engine.computing import Context
from rederive.engine.solving import _constant_enclosure, _outward

CONSTANTS = [
    sp.pi,
    sp.sqrt(2),
    sp.E,
    sp.sin(1),
    -sp.pi / 7,
    sp.Float("0.1", 30),
    sp.Float("-2.71828182845904523536028747135", 30),
    sp.exp(1000),
    sp.exp(-1000),
    sp.Rational(1, 3),
    sp.Integer(10) ** 400 + 1,
    sp.Rational(1, 10**400 + 7),
    sp.Rational(-884279719003555, 281474976710656),
    sp.I * sp.pi,
    2 + sp.I * sp.sqrt(3),
    sp.exp(sp.I),
    sp.exp(sp.I) * sp.exp(-1000),
    sp.sqrt(2) ** sp.I,
]


def _holds(interval, value) -> bool:
    low, high = (mpmath.mpf(end) for end in interval._mpi_)
    return low <= value <= high


@pytest.mark.parametrize("constant", CONSTANTS, ids=str)
def test_a_constant_enclosure_holds_the_constant(constant):
    real, imaginary = _constant_enclosure(constant)
    with mpmath.workdps(60):
        value = constant.evalf(60)
        exact_real, exact_imaginary = value.as_real_imag()
        assert _holds(real, mpmath.mpf(sp.Float(exact_real, 60)._mpf_))
        if imaginary is None:
            assert exact_imaginary == 0
        else:
            assert _holds(imaginary, mpmath.mpf(sp.Float(exact_imaginary, 60)._mpf_))


def test_a_constant_enclosure_is_narrow():
    """Wide enough to hold the constant and no wider than its 30-digit value allows."""
    real, imaginary = _constant_enclosure(sp.pi)
    assert imaginary is None
    assert real.b - real.a < mpmath.mpf(10) ** -15


def test_nsolve_proves_no_root_only_where_a_constant_allows_it():
    """The constant term is about -3.7*10^-17, so `ABS(x)` meets it at two points in
    the interval; the double nearest `pi` is no stand-in for `pi`."""
    text = (
        "NSOLVE(ABS(x) - pi + 884279719003555/281474976710656 + #e^(-37) = 0, x, -1, 2)"
    )
    assert simp(text, Context(precision_digits=10, notation_digits=10)) != "false"


BOUNDS = [sp.sqrt(2), sp.pi, -sp.pi, sp.Rational(1, 3), sp.Float("0.1"), sp.E**40]


@pytest.mark.parametrize("bound", BOUNDS, ids=str)
def test_the_explored_range_holds_the_bounds(bound):
    """The doubles the search starts from lie on the far side of each bound or at it."""
    below, above = _outward(bound, -math.inf), _outward(bound, math.inf)
    assert sp.Rational(below) <= bound <= sp.Rational(above)
    assert math.nextafter(below, math.inf) >= above
