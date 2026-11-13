"""Integrals from Bronstein's table that Simplify answers correctly.

The cases come from `BRONSTEIN-INTEGRALS.md`: Bronstein & Semendyayev's table of basic
integrals, the worked examples of the integration methods, the numbered table of
indefinite integrals, and the table of definite integrals. Each case pairs the integral
as transcribed with the text the worksheet shows.

This is a safety net under `test_integrate.py`, whose cases each target one integration
rule: the full tables catch breakage those cases miss.

Every expected value was verified before being recorded, not copied from the engine:
antiderivatives by numerical differentiation at points where the integrand is real,
definite integrals against the book's value or quadrature. The form is what is pinned,
so a failure may be a correct answer written differently; check it by differentiating.

An id is the book's entry number (`D` for the definite table, `pNNN.k` for unnumbered
entries on page NNN). `@n=3` marks a concrete instance of an entry with a symbolic
exponent, used to test reduction formulas, whose general form stays unevaluated because
the book's right-hand side contains another integral.

Entries that Simplify doesn't answer, answers wrongly, or takes over ten seconds on are
omitted; `BRONSTEIN-FAILURES.md` lists them.
"""

from __future__ import annotations

import pytest

from rederive.engine.computing import Context, simplify
from rederive.syntax import ParseState, parse_expression


def simp(text):
    state = ParseState()
    return simplify(parse_expression(text, state).node, Context(), state).text


CASES = [
    # -- Basic table ---------------------------------------------------------------------
    ("p386.1", "INT(x^n, x)", "(x^(n + 1) - 1)/(n + 1)"),
    ("p386.2", "INT(1/x, x)", "LN(x)"),
    ("p386.3", "INT(#e^x, x)", "#e^x"),
    ("p386.4", "INT(a^x, x)", "(a^x - 1)/LN(a)"),
    ("p386.5", "INT(SIN(x), x)", "-COS(x)"),
    ("p386.6", "INT(COS(x), x)", "SIN(x)"),
    ("p386.7", "INT(TAN(x), x)", "-LN(COS(x))"),
    ("p386.8", "INT(COT(x), x)", "LN(SIN(x))"),
    ("p386.9", "INT(1/COS(x)^2, x)", "TAN(x)"),
    ("p386.10", "INT(1/SIN(x)^2, x)", "-1/TAN(x)"),
    ("p386.11", "INT(SINH(x), x)", "#e^x/2 + #e^(-x)/2"),
    ("p386.12", "INT(COSH(x), x)", "#e^x/2 - #e^(-x)/2"),
    ("p386.13", "INT(TANH(x), x)", "LN(#e^x/2 + #e^(-x)/2)"),
    ("p386.14", "INT(COTH(x), x)", "LN(#e^x/2 - #e^(-x)/2)"),
    ("p386.15", "INT(1/COSH(x)^2, x)", "(#e^(2*x) - 1)/(#e^(2*x) + 1)"),
    ("p386.16", "INT(1/SINH(x)^2, x)", "-2/(#e^(2*x) - 1) - 1"),
    ("p386.17", "INT(1/(a^2 + x^2), x)", "ATAN(x/a)/a"),
    ("p386.18", "INT(1/(a^2 - x^2), x)", "(LN(x + a) - LN(x - a))/(2*a)"),
    ("p386.19", "INT(1/(x^2 - a^2), x)", "(LN(x - a) - LN(x + a))/(2*a)"),
    ("p386.20", "INT(1/SQRT(a^2 - x^2), x)", "ATAN(x/SQRT(a^2 - x^2))"),
    ("p386.21", "INT(1/SQRT(a^2 + x^2), x)", "LN(SQRT(x^2 + a^2) + x)"),
    ("p386.22", "INT(1/SQRT(x^2 - a^2), x)", "LN(SQRT(x^2 - a^2) + x)"),
    # -- General rules of integration ----------------------------------------------------
    ("p387.1", "INT((x + 3)^2*(x^2 + 1), x)", "x^5/5 + 3*x^4/2 + 10*x^3/3 + 3*x^2 + 9*x"),
    ("p387.2", "INT(SIN(2*x)*COS(x), x)", "-2*COS(x)^3/3"),
    ("p387.3", "INT(SIN(a*x), x)", "(1 - COS(a*x))/a"),
    ("p387.4", "INT(#e^(a*x + b), x)", "(#e^(a*x + b) - #e^b)/a"),
    ("p387.5", "INT(1/(1 + (x + a)^2), x)", "ATAN(x + a)"),
    ("p387.6", "INT((2*x + 3)/(x^2 + 3*x - 5), x)", "LN(x^2 + 3*x - 5)"),
    # -- Real simple roots of the denominator --------------------------------------------
    ("p389.1", "INT(c/(x - alpha), x)", "c*LN(x - alpha)"),
    (
        "p389.2",
        "INT((2*x + 3)/(x^3 + x^2 - 2*x), x)",
        "-3*LN(x)/2 + 5*LN(x - 1)/3 - LN(x + 2)/6",
    ),
    # -- Real multiple roots of the denominator ------------------------------------------
    ("p389.3", "INT(c/(x - alpha)^k, x)", "c*((x - alpha)^(1 - k) - 1)/(1 - k)"),
    ("p389.4", "INT((x^3 + 1)/(x*(x - 1)^3), x)", "-LN(x) + 2*LN(x - 1) - x/(x - 1)^2"),
    # -- Simple complex roots of the denominator -----------------------------------------
    (
        "p390.1",
        "INT((c*x + d)/(x^2 + p*x + q), x)",
        "c*LN(x^2 + p*x + q)/2 + (2*d - c*p)*ATAN((2*x + p)/SQRT(4*q - p^2))/SQRT(4*q "
        "- p^2)",
    ),
    ("p390.2", "INT(4/(x^3 + 4*x), x)", "LN(x) - LN(x^2 + 4)/2"),
    # -- Multiple complex roots of the denominator ---------------------------------------
    (
        "p391.1",
        "INT(c/2*(2*x + p)/(x^2 + p*x + q)^m, x)",
        "c*((x^2 + p*x + q)^(1 - m) - 1)/(2*(1 - m))",
    ),
    (
        "p391.2@m=2",
        "INT(1/(x^2 + p*x + q)^2, x)",
        "4*ATAN((2*x + p)/SQRT(4*q - p^2))/(4*q - p^2)^(3/2) - (2*x + p)/((p^2 - "
        "4*q)*(x^2 + p*x + q))",
    ),
    (
        "p392.1",
        "INT((2*x^2 + 2*x + 13)/((x - 2)*(x^2 + 1)^2), x)",
        "-4*ATAN(x) + LN(x - 2) - LN(x^2 + 1)/2 + (3 - 4*x)/(2*(x^2 + 1))",
    ),
    ("p392.2", "INT(1/(x^2 + 1)^2, x)", "ATAN(x)/2 + x/(2*(x^2 + 1))"),
    # -- Ostrogradsky's method, worked example -------------------------------------------
    (
        "p394.1",
        "INT((x^4 + x^3 + 4*x^2 + 3*x + 2)/((x + 1)^2*(x^2 + 1)^2), x)",
        "3*ATAN(x)/4 - (x^2 - x + 4)/(4*(x + 1)*(x^2 + 1))",
    ),
    ("p394.2", "INT((x + 1)/((x + 1)*(x^2 + 1)), x)", "ATAN(x)"),
    # -- Binomial differentials, worked example ------------------------------------------
    (
        "p397.1",
        "INT((1 + x^(1/4))^(1/3)/SQRT(x), x)",
        "3*(4*x^(1/4) - 3)*(x^(1/4) + 1)^(4/3)/7",
    ),
    # -- Universal substitution, worked example ------------------------------------------
    (
        "p399.1",
        "INT((1 + SIN(x))/(SIN(x)*(1 + COS(x))), x)",
        "TAN(x/2)^2/4 + LN(TAN(x/2))/2 + TAN(x/2)",
    ),
    # -- Simplified methods --------------------------------------------------------------
    (
        "p400.1",
        "INT(SIN(x)^2*COS(x)^4, x)",
        "-SIN(x)*COS(x)^5/6 + SIN(x)*COS(x)^3/24 + SIN(x)*COS(x)/16 + x/16",
    ),
    ("p400.2@n=3", "INT(TAN(x)^3, x)", "LN(COS(x)) + 1/(2*COS(x)^2)"),
    ("p400.3", "INT(TAN(x), x)", "-LN(COS(x))"),
    # -- Integrals with ax + b -----------------------------------------------------------
    ("1", "INT((a*x + b)^n, x)", "((a*x + b)^(n + 1) - b^(n + 1))/(a*(n + 1))"),
    ("2", "INT(1/(a*x + b), x)", "(LN(a*x + b) - LN(b))/a"),
    (
        "3",
        "INT(x*(a*x + b)^n, x)",
        "(a*x + b)^n*(a^2*x^2*(n + 1) + a*b*n*x - b^2)/(a^2*(n^2 + 3*n + 2))",
    ),
    (
        "4@n=3,m=2",
        "INT(x^2*(a*x + b)^3, x)",
        "a^3*x^6/6 + 3*a^2*b*x^5/5 + 3*a*b^2*x^4/4 + b^3*x^3/3",
    ),
    ("5", "INT(x/(a*x + b), x)", "-(b*LN(a*x + b) - b*LN(b) - a*x)/a^2"),
    (
        "6",
        "INT(x/(a*x + b)^2, x)",
        "(a*x*LN(a*x + b) + b*LN(a*x + b) - b*LN(b) - a*x*(LN(b) + 1))/(a^2*(a*x + b))",
    ),
    ("7", "INT(x/(a*x + b)^3, x)", "-(2*a*x + b)/(2*a^2*(a*x + b)^2)"),
    (
        "8",
        "INT(x/(a*x + b)^n, x)",
        "(a*x + b)^(-n)*(a^2*x^2*(1 - n) - a*b*n*x - b^2)/(a^2*(n^2 - 3*n + 2))",
    ),
    (
        "9",
        "INT(x^2/(a*x + b), x)",
        "(2*b^2*LN(a*x + b) - 2*b^2*LN(b) + a^2*x^2 - 2*a*b*x)/(2*a^3)",
    ),
    (
        "10",
        "INT(x^2/(a*x + b)^2, x)",
        "-(2*a*b*x*LN(a*x + b) + 2*b^2*LN(a*x + b) - 2*b^2*LN(b) - a^2*x^2 - "
        "a*b*x*(2*LN(b) + 2))/(a^3*(a*x + b))",
    ),
    (
        "11",
        "INT(x^2/(a*x + b)^3, x)",
        "LN(a*x + b)/a^3 - (2*LN(b) + 3)/(2*a^3) + b*(4*a*x + 3*b)/(2*a^3*(a*x + b)^2)",
    ),
    (
        "12",
        "INT(x^2/(a*x + b)^n, x)",
        "(a*x + b)^(-n)*(a^3*x^3*(-n^2 + 3*n - 2) + a^2*b*x^2*(n - n^2) - 2*a*b^2*n*x "
        "- 2*b^3)/(a^3*(n^3 - 6*n^2 + 11*n - 6))",
    ),
    (
        "13",
        "INT(x^3/(a*x + b), x)",
        "-(6*b^3*LN(a*x + b) - 6*b^3*LN(b) - 2*a^3*x^3 + 3*a^2*b*x^2 - "
        "6*a*b^2*x)/(6*a^4)",
    ),
    (
        "14",
        "INT(x^3/(a*x + b)^2, x)",
        "(6*a*b^2*x*LN(a*x + b) + 6*b^3*LN(a*x + b) - 6*b^3*LN(b) + a^3*x^3 - "
        "3*a^2*b*x^2 - a*b^2*x*(6*LN(b) + 6))/(2*a^4*(a*x + b))",
    ),
    (
        "15",
        "INT(x^3/(a*x + b)^3, x)",
        "-3*b*LN(a*x + b)/a^4 + x/a^3 + b*(6*LN(b) + 5)/(2*a^4) - b^2*(6*a*x + "
        "5*b)/(2*a^4*(a*x + b)^2)",
    ),
    (
        "16",
        "INT(x^3/(a*x + b)^4, x)",
        "LN(a*x + b)/a^4 - (6*LN(b) + 11)/(6*a^4) + b*(18*a^2*x^2 + 27*a*b*x + "
        "11*b^2)/(6*a^4*(a*x + b)^3)",
    ),
    (
        "17",
        "INT(x^3/(a*x + b)^n, x)",
        "(a*x + b)^(-n)*(a^4*x^4*(-n^3 + 6*n^2 - 11*n + 6) + a^3*b*x^3*(-n^3 + 3*n^2 - "
        "2*n) + a^2*b^2*x^2*(3*n - 3*n^2) - 6*a*b^3*n*x - 6*b^4)/(a^4*(n^4 - 10*n^3 + "
        "35*n^2 - 50*n + 24))",
    ),
    ("18", "INT(1/(x*(a*x + b)), x)", "(LN(a) - LN(b/x + a))/b"),
    ("19", "INT(1/(x*(a*x + b)^2), x)", "(LN(x) - LN(x + b/a))/b^2 + 1/(b*(a*x + b))"),
    (
        "20",
        "INT(1/(x*(a*x + b)^3), x)",
        "(LN(x) - LN(x + b/a))/b^3 + (2*a*x + 3*b)/(2*b^2*(a*x + b)^2)",
    ),
    (
        "21@n=3",
        "INT(1/(x*(a*x + b)^3), x)",
        "(LN(x) - LN(x + b/a))/b^3 + (2*a*x + 3*b)/(2*b^2*(a*x + b)^2)",
    ),
    ("22", "INT(1/(x^2*(a*x + b)), x)", "a*(LN(x + b/a) - LN(x))/b^2 - 1/(b*x)"),
    (
        "23",
        "INT(1/(x^2*(a*x + b)^2), x)",
        "2*a*(LN(x + b/a) - LN(x))/b^3 - (2*a*x + b)/(b^2*x*(a*x + b))",
    ),
    (
        "24",
        "INT(1/(x^2*(a*x + b)^3), x)",
        "-(6*a^3*x^3*LN(x) - 6*a^3*x^3*LN(a*x + b) + 12*a^2*b*x^2*LN(x) - "
        "12*a^2*b*x^2*LN(a*x + b) + 6*a*b^2*x*LN(x) - 6*a*b^2*x*LN(a*x + b) + "
        "6*a^2*b*x^2 + 9*a*b^2*x + 2*b^3)/(2*b^4*x*(a*x + b)^2)",
    ),
    (
        "25@n=3",
        "INT(1/(x^2*(a*x + b)^3), x)",
        "-(6*a^3*x^3*LN(x) - 6*a^3*x^3*LN(a*x + b) + 12*a^2*b*x^2*LN(x) - "
        "12*a^2*b*x^2*LN(a*x + b) + 6*a*b^2*x*LN(x) - 6*a*b^2*x*LN(a*x + b) + "
        "6*a^2*b*x^2 + 9*a*b^2*x + 2*b^3)/(2*b^4*x*(a*x + b)^2)",
    ),
    (
        "26",
        "INT(1/(x^3*(a*x + b)), x)",
        "(2*a*x - b)/(2*b^2*x^2) + a^2*(LN(x) - LN(x + b/a))/b^3",
    ),
    (
        "27",
        "INT(1/(x^3*(a*x + b)^2), x)",
        "3*a^2*(LN(x) - LN(x + b/a))/b^4 + (6*a^2*x^2 + 3*a*b*x - "
        "b^2)/(2*b^3*x^2*(a*x + b))",
    ),
    (
        "28",
        "INT(1/(x^3*(a*x + b)^3), x)",
        "(12*a^4*x^4*LN(x) - 12*a^4*x^4*LN(a*x + b) + 24*a^3*b*x^3*LN(x) - "
        "24*a^3*b*x^3*LN(a*x + b) + 12*a^2*b^2*x^2*LN(x) - 12*a^2*b^2*x^2*LN(a*x + b) "
        "+ 12*a^3*b*x^3 + 18*a^2*b^2*x^2 + 4*a*b^3*x - b^4)/(2*b^5*x^2*(a*x + b)^2)",
    ),
    (
        "29@n=3",
        "INT(1/(x^3*(a*x + b)^3), x)",
        "(12*a^4*x^4*LN(x) - 12*a^4*x^4*LN(a*x + b) + 24*a^3*b*x^3*LN(x) - "
        "24*a^3*b*x^3*LN(a*x + b) + 12*a^2*b^2*x^2*LN(x) - 12*a^2*b^2*x^2*LN(a*x + b) "
        "+ 12*a^3*b*x^3 + 18*a^2*b^2*x^2 + 4*a*b^3*x - b^4)/(2*b^5*x^2*(a*x + b)^2)",
    ),
    (
        "30@n=3,m=2",
        "INT(1/(x^2*(a*x + b)^3), x)",
        "-(6*a^3*x^3*LN(x) - 6*a^3*x^3*LN(a*x + b) + 12*a^2*b*x^2*LN(x) - "
        "12*a^2*b*x^2*LN(a*x + b) + 6*a*b^2*x*LN(x) - 6*a*b^2*x*LN(a*x + b) + "
        "6*a^2*b*x^2 + 9*a*b^2*x + 2*b^3)/(2*b^4*x*(a*x + b)^2)",
    ),
    (
        "31",
        "INT((a*x + b)/(f*x + g), x)",
        "-((a*g - b*f)*LN(f*x + g) - a*g*LN(g) + b*f*LN(g) - a*f*x)/f^2",
    ),
    ("32", "INT(1/((a*x + b)*(f*x + g)), x)", "(LN(a*x + b) - LN(f*x + g))/(a*g - b*f)"),
    (
        "33",
        "INT(x/((a*x + b)*(f*x + g)), x)",
        "g*LN(x + g*(a*g/f + b)/(a*g + b*f))/(f*(a*g - b*f)) - b*LN(x + b*(b*f/a + "
        "g)/(a*g + b*f))/(a*(a*g - b*f))",
    ),
    (
        "34",
        "INT(1/((a*x + b)^2*(f*x + g)), x)",
        "-(a*f*x*LN(a*x + b) - a*f*x*LN(f*x + g) + b*f*LN(a*x + b) - b*f*LN(f*x + g) "
        "+ a*g - b*f)/((a*g - b*f)^2*(a*x + b))",
    ),
    (
        "35",
        "INT(x/((a + x)*(b + x)^2), x)",
        "-(a*x*LN(x + a) - a*x*LN(x + b) + a*b*LN(x + a) - a*b*LN(x + b) - b*(a - "
        "b))/((a - b)^2*(x + b))",
    ),
    (
        "36",
        "INT(x^2/((a + x)*(b + x)^2), x)",
        "-(b*x*(2*a - b)*LN(x + b) - a^2*x*LN(x + a) + b^2*(2*a - b)*LN(x + b) - "
        "a^2*b*LN(x + a) + b^2*(a - b))/((a - b)^2*(x + b))",
    ),
    (
        "37",
        "INT(1/((a + x)^2*(b + x)^2), x)",
        "(2*x^2*LN(x + a) - 2*x^2*LN(x + b) + x*(2*a + 2*b)*LN(x + a) - x*(2*a + "
        "2*b)*LN(x + b) + 2*a*b*LN(x + a) - 2*a*b*LN(x + b) - 2*x*(a - b) - (a + b)*(a "
        "- b))/((a - b)^3*(x + a)*(x + b))",
    ),
    (
        "38",
        "INT(x/((a + x)^2*(b + x)^2), x)",
        "-(x^2*(a + b)*LN(x + a) - x^2*(a + b)*LN(x + b) + x*(a + b)^2*LN(x + a) - x*(a "
        "+ b)^2*LN(x + b) + a*b*(a + b)*LN(x + a) - a*b*(a + b)*LN(x + b) - x*(a + "
        "b)*(a - b) - 2*a*b*(a - b))/((a - b)^3*(x + a)*(x + b))",
    ),
    (
        "39",
        "INT(x^2/((a + x)^2*(b + x)^2), x)",
        "(2*a*b*x^2*LN(x + a) - 2*a*b*x^2*LN(x + b) + x*(2*a^2*b + 2*a*b^2)*LN(x + a) - "
        "x*(2*a^2*b + 2*a*b^2)*LN(x + b) + 2*a^2*b^2*LN(x + a) - 2*a^2*b^2*LN(x + b) - "
        "x*(a - b)*(a^2 + b^2) - (a - b)*(a^2*b + a*b^2))/((a - b)^3*(x + a)*(x + b))",
    ),
    # -- Integrals with ax^2 + bx + c ----------------------------------------------------
    (
        "40a",
        "INT(1/(a*x^2 + b*x + c), x)",
        "2*ATAN((2*a*x + b)/SQRT(4*a*c - b^2))/SQRT(4*a*c - b^2)",
    ),
    (
        "40b",
        "INT(1/(a*x^2 + b*x + c), x)",
        "2*ATAN((2*a*x + b)/SQRT(4*a*c - b^2))/SQRT(4*a*c - b^2)",
    ),
    (
        "41",
        "INT(1/(a*x^2 + b*x + c)^2, x)",
        "(2*a*x + b)/((4*a*c - b^2)*(a*x^2 + b*x + c)) + 4*a*ATAN((2*a*x + b)/SQRT(4*a*c "
        "- b^2))/(4*a*c - b^2)^(3/2)",
    ),
    (
        "42",
        "INT(1/(a*x^2 + b*x + c)^3, x)",
        "(12*a^3*x^3 + 18*a^2*b*x^2 + x*(20*a^2*c + 4*a*b^2) + 10*a*b*c - "
        "b^3)/(2*(4*a*c - b^2)^2*(a*x^2 + b*x + c)^2) + 12*a^2*ATAN((2*a*x + "
        "b)/SQRT(4*a*c - b^2))/(4*a*c - b^2)^(5/2)",
    ),
    (
        "43@n=3",
        "INT(1/(a*x^2 + b*x + c)^3, x)",
        "(12*a^3*x^3 + 18*a^2*b*x^2 + x*(20*a^2*c + 4*a*b^2) + 10*a*b*c - "
        "b^3)/(2*(4*a*c - b^2)^2*(a*x^2 + b*x + c)^2) + 12*a^2*ATAN((2*a*x + "
        "b)/SQRT(4*a*c - b^2))/(4*a*c - b^2)^(5/2)",
    ),
    (
        "44",
        "INT(x/(a*x^2 + b*x + c), x)",
        "LN(a*x^2 + b*x + c)/(2*a) - b*ATAN((2*a*x + b)/SQRT(4*a*c - "
        "b^2))/(a*SQRT(4*a*c - b^2))",
    ),
    (
        "45",
        "INT(x/(a*x^2 + b*x + c)^2, x)",
        "-(b*x + 2*c)/((4*a*c - b^2)*(a*x^2 + b*x + c)) - 2*b*ATAN((2*a*x + "
        "b)/SQRT(4*a*c - b^2))/(4*a*c - b^2)^(3/2)",
    ),
    (
        "46@n=3",
        "INT(x/(a*x^2 + b*x + c)^3, x)",
        "-(6*a^2*b*x^3 + 9*a*b^2*x^2 + x*(10*a*b*c + 2*b^3) + 8*a*c^2 + "
        "b^2*c)/(2*(4*a*c - b^2)^2*(a*x^2 + b*x + c)^2) - 6*a*b*ATAN((2*a*x + "
        "b)/SQRT(4*a*c - b^2))/(4*a*c - b^2)^(5/2)",
    ),
    (
        "47",
        "INT(x^2/(a*x^2 + b*x + c), x)",
        "-b*LN(a*x^2 + b*x + c)/(2*a^2) + x/a - (2*a*c - b^2)*ATAN((2*a*x + "
        "b)/SQRT(4*a*c - b^2))/(a^2*SQRT(4*a*c - b^2))",
    ),
    (
        "48",
        "INT(x^2/(a*x^2 + b*x + c)^2, x)",
        "(x*(b^2 - 2*a*c) + b*c)/(a*(4*a*c - b^2)*(a*x^2 + b*x + c)) + 4*c*ATAN((2*a*x + "
        "b)/SQRT(4*a*c - b^2))/(4*a*c - b^2)^(3/2)",
    ),
    (
        "49@n=3",
        "INT(x^2/(a*x^2 + b*x + c)^3, x)",
        "(x^3*(4*a^2*c + 2*a*b^2) + x^2*(6*a*b*c + 3*b^3) + x*(10*b^2*c - 4*a*c^2) + "
        "6*b*c^2)/(2*(4*a*c - b^2)^2*(a*x^2 + b*x + c)^2) + 2*(2*a*c + "
        "b^2)*ATAN((2*a*x + b)/SQRT(4*a*c - b^2))/(4*a*c - b^2)^(5/2)",
    ),
    (
        "50@n=3,m=2",
        "INT(x^2/(a*x^2 + b*x + c)^3, x)",
        "(x^3*(4*a^2*c + 2*a*b^2) + x^2*(6*a*b*c + 3*b^3) + x*(10*b^2*c - 4*a*c^2) + "
        "6*b*c^2)/(2*(4*a*c - b^2)^2*(a*x^2 + b*x + c)^2) + 2*(2*a*c + "
        "b^2)*ATAN((2*a*x + b)/SQRT(4*a*c - b^2))/(4*a*c - b^2)^(5/2)",
    ),
    (
        "51@n=3",
        "INT(x^5/(a*x^2 + b*x + c)^3, x)",
        "LN(a*x^2 + b*x + c)/(2*a^3) + (x^3*(50*a^3*b*c^2 - 30*a^2*b^3*c + 4*a*b^5) + "
        "x^2*(32*a^3*c^3 + 11*a^2*b^2*c^2 - 19*a*b^4*c + 3*b^6) + x*(62*a^2*b*c^3 - "
        "44*a*b^3*c^2 + 6*b^5*c) + 24*a^2*c^4 - 21*a*b^2*c^3 + "
        "3*b^4*c^2)/(2*a^3*(4*a*c - b^2)^2*(a*x^2 + b*x + c)^2) - b*(30*a^2*c^2 - "
        "10*a*b^2*c + b^4)*ATAN((2*a*x + b)/SQRT(4*a*c - b^2))/(a^3*(4*a*c - "
        "b^2)^(5/2))",
    ),
    (
        "52",
        "INT(1/(x*(a*x^2 + b*x + c)), x)",
        "LN(x)/c - LN(a*x^2 + b*x + c)/(2*c) - b*ATAN((2*a*x + b)/SQRT(4*a*c - "
        "b^2))/(c*SQRT(4*a*c - b^2))",
    ),
    (
        "53@n=3",
        "INT(1/(x*(a*x^2 + b*x + c)^3), x)",
        "LN(x)/c^3 - LN(a*x^2 + b*x + c)/(2*c^3) + (x^3*(2*a^2*b^3 - 14*a^3*b*c) + "
        "x^2*(16*a^3*c^2 - 29*a^2*b^2*c + 4*a*b^4) + x*(-2*a^2*b*c^2 - 12*a*b^3*c + "
        "2*b^5) + 24*a^2*c^3 - 21*a*b^2*c^2 + 3*b^4*c)/(2*c^2*(4*a*c - b^2)^2*(a*x^2 "
        "+ b*x + c)^2) - b*(30*a^2*c^2 - 10*a*b^2*c + b^4)*ATAN((2*a*x + "
        "b)/SQRT(4*a*c - b^2))/(c^3*(4*a*c - b^2)^(5/2))",
    ),
    (
        "54",
        "INT(1/(x^2*(a*x^2 + b*x + c)), x)",
        "-b*LN(x)/c^2 + b*LN(a*x^2 + b*x + c)/(2*c^2) - 1/(c*x) - (2*a*c - "
        "b^2)*ATAN((2*a*x + b)/SQRT(4*a*c - b^2))/(c^2*SQRT(4*a*c - b^2))",
    ),
    (
        "55@n=3,m=2",
        "INT(1/(x^2*(a*x^2 + b*x + c)^3), x)",
        "-3*b*LN(x)/c^4 + 3*b*LN(a*x^2 + b*x + c)/(2*c^4) + (x^4*(-60*a^4*c^2 + "
        "42*a^3*b^2*c - 6*a^2*b^4) + x^3*(-138*a^3*b*c^2 + 87*a^2*b^3*c - 12*a*b^5) + "
        "x^2*(-100*a^3*c^3 - 14*a^2*b^2*c^2 + 36*a*b^4*c - 6*b^6) + x*(-122*a^2*b*c^3 "
        "+ 68*a*b^3*c^2 - 9*b^5*c) - 32*a^2*c^4 + 16*a*b^2*c^3 - "
        "2*b^4*c^2)/(2*c^3*x*(4*a*c - b^2)^2*(a*x^2 + b*x + c)^2) - 3*(20*a^3*c^3 - "
        "30*a^2*b^2*c^2 + 10*a*b^4*c - b^6)*ATAN((2*a*x + b)/SQRT(4*a*c - "
        "b^2))/(c^4*(4*a*c - b^2)^(5/2))",
    ),
    (
        "56",
        "INT(1/((f*x + g)*(a*x^2 + b*x + c)), x)",
        "(2*a*g - b*f)*ATAN((2*a*x + b)/SQRT(4*a*c - b^2))/((a*g^2 - b*f*g + "
        "c*f^2)*SQRT(4*a*c - b^2)) + f*LN(f*x + g)/(a*g^2 - b*f*g + c*f^2) - "
        "f*LN(a*x^2 + b*x + c)/(2*(a*g^2 - b*f*g + c*f^2))",
    ),
    # -- Integrals with a^2 ± x^2 --------------------------------------------------------
    ("57a", "INT(1/(a^2 + x^2), x)", "ATAN(x/a)/a"),
    ("57b", "INT(1/(a^2 - x^2), x)", "(LN(x + a) - LN(x - a))/(2*a)"),
    ("58a", "INT(1/(a^2 + x^2)^2, x)", "ATAN(x/a)/(2*a^3) + x/(2*a^2*(x^2 + a^2))"),
    (
        "58b",
        "INT(1/(a^2 - x^2)^2, x)",
        "(LN(x + a) - LN(x - a))/(4*a^3) - x/(2*a^2*(x^2 - a^2))",
    ),
    (
        "59a",
        "INT(1/(a^2 + x^2)^3, x)",
        "3*ATAN(x/a)/(8*a^5) + x*(3*x^2 + 5*a^2)/(8*a^4*(x^2 + a^2)^2)",
    ),
    (
        "59b",
        "INT(1/(a^2 - x^2)^3, x)",
        "(3*LN(x + a) - 3*LN(x - a))/(16*a^5) + x*(5*a^2 - 3*x^2)/(8*a^4*(x^2 - a^2)^2)",
    ),
    (
        "60a@n=3",
        "INT(1/(a^2 + x^2)^(3 + 1), x)",
        "5*ATAN(x/a)/(16*a^7) + x*(15*x^4 + 40*a^2*x^2 + 33*a^4)/(48*a^6*(x^2 + a^2)^3)",
    ),
    (
        "60b@n=3",
        "INT(1/(a^2 - x^2)^(3 + 1), x)",
        "(5*LN(x + a) - 5*LN(x - a))/(32*a^7) - x*(15*x^4 - 40*a^2*x^2 + "
        "33*a^4)/(48*a^6*(x^2 - a^2)^3)",
    ),
    ("61a", "INT(x/(a^2 + x^2), x)", "LN(x^2 + a^2)/2"),
    ("61b", "INT(x/(a^2 - x^2), x)", "-LN(x^2 - a^2)/2"),
    ("62a", "INT(x/(a^2 + x^2)^2, x)", "-1/(2*(x^2 + a^2))"),
    ("62b", "INT(x/(a^2 - x^2)^2, x)", "1/(2*(a^2 - x^2))"),
    ("63a", "INT(x/(a^2 + x^2)^3, x)", "-1/(4*(x^2 + a^2)^2)"),
    ("63b", "INT(x/(a^2 - x^2)^3, x)", "1/(4*(a^2 - x^2)^2)"),
    ("64a", "INT(x/(a^2 + x^2)^(n + 1), x)", "1/(2*n) - (x^2 + a^2)^(-n)/(2*n)"),
    ("64b", "INT(x/(a^2 - x^2)^(n + 1), x)", "(a^2 - x^2)^(-n)/(2*n) - 1/(2*n)"),
    ("65a", "INT(x^2/(a^2 + x^2), x)", "x - a*ATAN(x/a)"),
    ("65b", "INT(x^2/(a^2 - x^2), x)", "-a*LN(x - a)/2 + a*LN(x + a)/2 - x"),
    ("66a", "INT(x^2/(a^2 + x^2)^2, x)", "ATAN(x/a)/(2*a) - x/(2*(x^2 + a^2))"),
    (
        "66b",
        "INT(x^2/(a^2 - x^2)^2, x)",
        "(LN(x - a) - LN(x + a))/(4*a) - x/(2*(x^2 - a^2))",
    ),
    (
        "67a",
        "INT(x^2/(a^2 + x^2)^3, x)",
        "ATAN(x/a)/(8*a^3) + x*(x^2 - a^2)/(8*a^2*(x^2 + a^2)^2)",
    ),
    (
        "67b",
        "INT(x^2/(a^2 - x^2)^3, x)",
        "(LN(x - a) - LN(x + a))/(16*a^3) + x*(x^2 + a^2)/(8*a^2*(x^2 - a^2)^2)",
    ),
    (
        "68a@n=3",
        "INT(x^2/(a^2 + x^2)^(3 + 1), x)",
        "ATAN(x/a)/(16*a^5) + x*(3*x^4 + 8*a^2*x^2 - 3*a^4)/(48*a^4*(x^2 + a^2)^3)",
    ),
    (
        "68b@n=3",
        "INT(x^2/(a^2 - x^2)^(3 + 1), x)",
        "(3*x^6*LN(x - a) - 3*x^6*LN(x + a) - 9*a^2*x^4*LN(x - a) + 9*a^2*x^4*LN(x + a) "
        "+ 9*a^4*x^2*LN(x - a) - 9*a^4*x^2*LN(x + a) - 3*a^6*LN(x - a) + 3*a^6*LN(x + a) "
        "+ 6*a*x^5 - 16*a^3*x^3 - 6*a^5*x)/(96*a^5*(x^2 - a^2)^3)",
    ),
    ("69a", "INT(x^3/(a^2 + x^2), x)", "x^2/2 - a^2*LN(x^2 + a^2)/2"),
    ("69b", "INT(x^3/(a^2 - x^2), x)", "-a^2*LN(x^2 - a^2)/2 - x^2/2"),
    ("70a", "INT(x^3/(a^2 + x^2)^2, x)", "LN(x^2 + a^2)/2 + a^2/(2*(x^2 + a^2))"),
    ("70b", "INT(x^3/(a^2 - x^2)^2, x)", "LN(x^2 - a^2)/2 - a^2/(2*(x^2 - a^2))"),
    ("71a", "INT(x^3/(a^2 + x^2)^3, x)", "-(2*x^2 + a^2)/(4*(x^2 + a^2)^2)"),
    ("71b", "INT(x^3/(a^2 - x^2)^3, x)", "(2*x^2 - a^2)/(4*(x^2 - a^2)^2)"),
    (
        "72a",
        "INT(x^3/(a^2 + x^2)^(n + 1), x)",
        "-(x^2 + a^2)^(-n - 1)*(n*x^4 + a^2*x^2*(n + 1) + a^4)/(2*(n^2 - n))",
    ),
    (
        "72b",
        "INT(x^3/(a^2 - x^2)^(n + 1), x)",
        "-(a^2 - x^2)^(-n - 1)*(n*x^4 - a^2*x^2*(n + 1) + a^4)/(2*(n^2 - n))",
    ),
    ("73a", "INT(1/(x*(a^2 + x^2)), x)", "(2*LN(x) - LN(x^2 + a^2))/(2*a^2)"),
    ("73b", "INT(1/(x*(a^2 - x^2)), x)", "-LN(1 - a^2/x^2)/(2*a^2)"),
    (
        "74a",
        "INT(1/(x*(a^2 + x^2)^2), x)",
        "LN(x)/a^4 - LN(x^2 + a^2)/(2*a^4) + 1/(2*a^2*(x^2 + a^2))",
    ),
    (
        "74b",
        "INT(1/(x*(a^2 - x^2)^2), x)",
        "LN(x)/a^4 - LN(x^2 - a^2)/(2*a^4) - 1/(2*a^2*(x^2 - a^2))",
    ),
    (
        "75a",
        "INT(1/(x*(a^2 + x^2)^3), x)",
        "LN(x)/a^6 - LN(x^2 + a^2)/(2*a^6) + (2*x^2 + 3*a^2)/(4*a^4*(x^2 + a^2)^2)",
    ),
    (
        "75b",
        "INT(1/(x*(a^2 - x^2)^3), x)",
        "LN(x)/a^6 - LN(x^2 - a^2)/(2*a^6) + (3*a^2 - 2*x^2)/(4*a^4*(x^2 - a^2)^2)",
    ),
    ("76a", "INT(1/(x^2*(a^2 + x^2)), x)", "-ATAN(x/a)/a^3 - 1/(a^2*x)"),
    ("76b", "INT(1/(x^2*(a^2 - x^2)), x)", "(LN(x + a) - LN(x - a))/(2*a^3) - 1/(a^2*x)"),
    (
        "77a",
        "INT(1/(x^2*(a^2 + x^2)^2), x)",
        "-3*ATAN(x/a)/(2*a^5) - (3*x^2 + 2*a^2)/(2*a^4*x*(x^2 + a^2))",
    ),
    (
        "77b",
        "INT(1/(x^2*(a^2 - x^2)^2), x)",
        "(3*LN(x + a) - 3*LN(x - a))/(4*a^5) + (2*a^2 - 3*x^2)/(2*a^4*x*(x^2 - a^2))",
    ),
    (
        "78a",
        "INT(1/(x^2*(a^2 + x^2)^3), x)",
        "-15*ATAN(x/a)/(8*a^7) - (15*x^4 + 25*a^2*x^2 + 8*a^4)/(8*a^6*x*(x^2 + a^2)^2)",
    ),
    (
        "78b",
        "INT(1/(x^2*(a^2 - x^2)^3), x)",
        "(15*LN(x + a) - 15*LN(x - a))/(16*a^7) - (15*x^4 - 25*a^2*x^2 + "
        "8*a^4)/(8*a^6*x*(x^2 - a^2)^2)",
    ),
    (
        "79a",
        "INT(1/(x^3*(a^2 + x^2)), x)",
        "-LN(x)/a^4 + LN(x^2 + a^2)/(2*a^4) - 1/(2*a^2*x^2)",
    ),
    (
        "79b",
        "INT(1/(x^3*(a^2 - x^2)), x)",
        "LN(x)/a^4 - LN(x^2 - a^2)/(2*a^4) - 1/(2*a^2*x^2)",
    ),
    (
        "80a",
        "INT(1/(x^3*(a^2 + x^2)^2), x)",
        "-(4*x^4*LN(x) - 2*x^4*LN(x^2 + a^2) + 4*a^2*x^2*LN(x) - 2*a^2*x^2*LN(x^2 + "
        "a^2) + 2*a^2*x^2 + a^4)/(2*a^6*x^2*(x^2 + a^2))",
    ),
    (
        "80b",
        "INT(1/(x^3*(a^2 - x^2)^2), x)",
        "-(2*x^4*LN(a^2 - x^2) - 4*x^4*LN(ABS(x)) - 2*a^2*x^2*LN(a^2 - x^2) + "
        "4*a^2*x^2*LN(ABS(x)) + 2*a^2*x^2 - a^4)/(2*a^6*x^2*(x^2 - a^2))",
    ),
    (
        "81a",
        "INT(1/(x^3*(a^2 + x^2)^3), x)",
        "-3*LN(x)/a^8 + 3*LN(x^2 + a^2)/(2*a^8) - (6*x^4 + 9*a^2*x^2 + "
        "2*a^4)/(4*a^6*x^2*(x^2 + a^2)^2)",
    ),
    (
        "81b",
        "INT(1/(x^3*(a^2 - x^2)^3), x)",
        "-(6*x^6*LN(a^2 - x^2) - 12*x^6*LN(ABS(x)) - 12*a^2*x^4*LN(a^2 - x^2) + "
        "24*a^2*x^4*LN(ABS(x)) + 6*a^4*x^2*LN(a^2 - x^2) - 12*a^4*x^2*LN(ABS(x)) + "
        "6*a^2*x^4 - 9*a^4*x^2 + 2*a^6)/(4*a^8*x^2*(x^2 - a^2)^2)",
    ),
    (
        "82a",
        "INT(1/((b + c*x)*(a^2 + x^2)), x)",
        "b*ATAN(x/ABS(a))/((a^2*c^2 + b^2)*ABS(a)) - c*LN(x^2 + a^2)/(2*(a^2*c^2 + "
        "b^2)) + c*LN(c*x + b)/(a^2*c^2 + b^2)",
    ),
    (
        "82b",
        "INT(1/((b + c*x)*(a^2 - x^2)), x)",
        "-LN(x - a)/(2*a*(a*c + b)) - LN(x + a)/(2*a*(a*c - b)) + c*LN(c*x + b)/((a*c + "
        "b)*(a*c - b))",
    ),
    # -- Integrals with a^3 ± x^3 --------------------------------------------------------
    (
        "83a",
        "INT(1/(a^3 + x^3), x)",
        "SQRT(3)*ATAN(SQRT(3)*(2*x - a)/(3*ABS(a)))/(3*a*ABS(a)) + LN(x + a)/(3*a^2) - "
        "LN(x^2 - a*x + a^2)/(6*a^2)",
    ),
    (
        "83b",
        "INT(1/(a^3 - x^3), x)",
        "SQRT(3)*ATAN(SQRT(3)*(2*x + a)/(3*ABS(a)))/(3*a*ABS(a)) - LN(x - a)/(3*a^2) + "
        "LN(x^2 + a*x + a^2)/(6*a^2)",
    ),
    (
        "84a",
        "INT(1/(a^3 + x^3)^2, x)",
        "-(2*SQRT(3)*x^3*ABS(a)*ATAN(SQRT(3)*(a - 2*x)/(3*ABS(a))) + "
        "2*SQRT(3)*a^3*ABS(a)*ATAN(SQRT(3)*(a - 2*x)/(3*ABS(a))) - 2*a*x^3*LN(x + a) + "
        "a*x^3*LN(x^2 - a*x + a^2) - 2*a^4*LN(x + a) + a^4*LN(x^2 - a*x + a^2) - "
        "3*a^3*x)/(9*a^6*(x^3 + a^3))",
    ),
    (
        "84b",
        "INT(1/(a^3 - x^3)^2, x)",
        "(2*SQRT(3)*x^3*ABS(a)*ATAN(SQRT(3)*(2*x + a)/(3*ABS(a))) - "
        "2*SQRT(3)*a^3*ABS(a)*ATAN(SQRT(3)*(2*x + a)/(3*ABS(a))) - 2*a*x^3*LN(x - a) + "
        "a*x^3*LN(x^2 + a*x + a^2) + 2*a^4*LN(x - a) - a^4*LN(x^2 + a*x + a^2) - "
        "3*a^3*x)/(9*a^6*(x^3 - a^3))",
    ),
    (
        "85a",
        "INT(x/(a^3 + x^3), x)",
        "SQRT(3)*ATAN(SQRT(3)*(2*x - a)/(3*ABS(a)))/(3*ABS(a)) - LN(x + a)/(3*a) + "
        "LN(x^2 - a*x + a^2)/(6*a)",
    ),
    (
        "85b",
        "INT(x/(a^3 - x^3), x)",
        "-SQRT(3)*ATAN(SQRT(3)*(2*x + a)/(3*ABS(a)))/(3*ABS(a)) - LN(x - a)/(3*a) + "
        "LN(x^2 + a*x + a^2)/(6*a)",
    ),
    (
        "86a",
        "INT(x/(a^3 + x^3)^2, x)",
        "-(2*SQRT(3)*x^3*SIGN(a)*ATAN(SQRT(3)*(a - 2*x)/(3*ABS(a))) + "
        "2*SQRT(3)*a^3*SIGN(a)*ATAN(SQRT(3)*(a - 2*x)/(3*ABS(a))) + 2*x^3*LN(x + a) - "
        "x^3*LN(x^2 - a*x + a^2) + 2*a^3*LN(x + a) - a^3*LN(x^2 - a*x + a^2) - "
        "6*a*x^2)/(18*a^4*(x^3 + a^3))",
    ),
    (
        "86b",
        "INT(x/(a^3 - x^3)^2, x)",
        "-(2*SQRT(3)*x^3*SIGN(a)*ATAN(SQRT(3)*(2*x + a)/(3*ABS(a))) - "
        "2*SQRT(3)*a^3*SIGN(a)*ATAN(SQRT(3)*(2*x + a)/(3*ABS(a))) + 2*x^3*LN(x - a) - "
        "x^3*LN(x^2 + a*x + a^2) - 2*a^3*LN(x - a) + a^3*LN(x^2 + a*x + a^2) + "
        "6*a*x^2)/(18*a^4*(x^3 - a^3))",
    ),
    ("87a", "INT(x^2/(a^3 + x^3), x)", "LN(x^3 + a^3)/3"),
    ("87b", "INT(x^2/(a^3 - x^3), x)", "-LN(x^3 - a^3)/3"),
    ("88a", "INT(x^2/(a^3 + x^3)^2, x)", "-1/(3*(x^3 + a^3))"),
    ("88b", "INT(x^2/(a^3 - x^3)^2, x)", "1/(3*(a^3 - x^3))"),
    (
        "89a",
        "INT(x^3/(a^3 + x^3), x)",
        "SQRT(3)*ABS(a)*ATAN(SQRT(3)*(a - 2*x)/(3*ABS(a)))/3 - a*LN(x + a)/3 + "
        "a*LN(x^2 - a*x + a^2)/6 + x",
    ),
    (
        "89b",
        "INT(x^3/(a^3 - x^3), x)",
        "SQRT(3)*ABS(a)*ATAN(SQRT(3)*(2*x + a)/(3*ABS(a)))/3 - a*LN(x - a)/3 + "
        "a*LN(x^2 + a*x + a^2)/6 - x",
    ),
    (
        "90a",
        "INT(x^3/(a^3 + x^3)^2, x)",
        "SQRT(3)*ATAN(SQRT(3)*(2*x - a)/(3*ABS(a)))/(9*a*ABS(a)) + LN(x + a)/(9*a^2) - "
        "LN(x^2 - a*x + a^2)/(18*a^2) - x/(3*(x^3 + a^3))",
    ),
    (
        "90b",
        "INT(x^3/(a^3 - x^3)^2, x)",
        "-SQRT(3)*ATAN(SQRT(3)*(2*x + a)/(3*ABS(a)))/(9*a*ABS(a)) + LN(x - a)/(9*a^2) - "
        "LN(x^2 + a*x + a^2)/(18*a^2) - x/(3*(x^3 - a^3))",
    ),
    ("91a", "INT(1/(x*(a^3 + x^3)), x)", "(3*LN(x) - LN(x^3 + a^3))/(3*a^3)"),
    ("91b", "INT(1/(x*(a^3 - x^3)), x)", "(3*LN(x) - LN(x^3 - a^3))/(3*a^3)"),
    (
        "92a",
        "INT(1/(x*(a^3 + x^3)^2), x)",
        "LN(x)/a^6 - LN(x^3 + a^3)/(3*a^6) + 1/(3*a^3*(x^3 + a^3))",
    ),
    (
        "92b",
        "INT(1/(x*(a^3 - x^3)^2), x)",
        "LN(x)/a^6 - LN(x^3 - a^3)/(3*a^6) - 1/(3*a^3*(x^3 - a^3))",
    ),
    (
        "93a",
        "INT(1/(x^2*(a^3 + x^3)), x)",
        "(2*SQRT(3)*x*SIGN(a)*ATAN(SQRT(3)*(a - 2*x)/(3*ABS(a))) + 2*x*LN(x + a) - "
        "x*LN(x^2 - a*x + a^2) - 6*a)/(6*a^4*x)",
    ),
    (
        "93b",
        "INT(1/(x^2*(a^3 - x^3)), x)",
        "-(2*SQRT(3)*x*SIGN(a)*ATAN(SQRT(3)*(2*x + a)/(3*ABS(a))) + 2*x*LN(x - a) - "
        "x*LN(x^2 + a*x + a^2) + 6*a)/(6*a^4*x)",
    ),
    (
        "94a",
        "INT(1/(x^2*(a^3 + x^3)^2), x)",
        "(4*SQRT(3)*x^4*ABS(a)*ATAN(SQRT(3)*(a - 2*x)/(3*ABS(a))) + "
        "4*SQRT(3)*a^3*x*ABS(a)*ATAN(SQRT(3)*(a - 2*x)/(3*ABS(a))) + 4*a*x^4*LN(x + "
        "a) - 2*a*x^4*LN(x^2 - a*x + a^2) + 4*a^4*x*LN(x + a) - 2*a^4*x*LN(x^2 - a*x "
        "+ a^2) - 12*a^2*x^3 - 9*a^5)/(9*a^8*x*(x^3 + a^3))",
    ),
    (
        "94b",
        "INT(1/(x^2*(a^3 - x^3)^2), x)",
        "-(4*SQRT(3)*x^4*ABS(a)*ATAN(SQRT(3)*(2*x + a)/(3*ABS(a))) - "
        "4*SQRT(3)*a^3*x*ABS(a)*ATAN(SQRT(3)*(2*x + a)/(3*ABS(a))) + 4*a*x^4*LN(x - "
        "a) - 2*a*x^4*LN(x^2 + a*x + a^2) - 4*a^4*x*LN(x - a) + 2*a^4*x*LN(x^2 + a*x "
        "+ a^2) + 12*a^2*x^3 - 9*a^5)/(9*a^8*x*(x^3 - a^3))",
    ),
    (
        "95a",
        "INT(1/(x^3*(a^3 + x^3)), x)",
        "-SQRT(3)*ABS(a)*ATAN(SQRT(3)*(2*x - a)/(3*ABS(a)))/(3*a^6) - LN(x + "
        "a)/(3*a^5) + LN(x^2 - a*x + a^2)/(6*a^5) - 1/(2*a^3*x^2)",
    ),
    (
        "95b",
        "INT(1/(x^3*(a^3 - x^3)), x)",
        "SQRT(3)*ABS(a)*ATAN(SQRT(3)*(2*x + a)/(3*ABS(a)))/(3*a^6) - LN(x - a)/(3*a^5) "
        "+ LN(x^2 + a*x + a^2)/(6*a^5) - 1/(2*a^3*x^2)",
    ),
    (
        "96a",
        "INT(1/(x^3*(a^3 + x^3)^2), x)",
        "(10*SQRT(3)*x^5*SIGN(a)*ATAN(SQRT(3)*(a - 2*x)/(3*ABS(a))) + "
        "10*SQRT(3)*a^3*x^2*SIGN(a)*ATAN(SQRT(3)*(a - 2*x)/(3*ABS(a))) - 10*x^5*LN(x + "
        "a) + 5*x^5*LN(x^2 - a*x + a^2) - 10*a^3*x^2*LN(x + a) + 5*a^3*x^2*LN(x^2 - "
        "a*x + a^2) - 15*a^2*x^3 - 9*a^5)/(18*a^8*x^2*(x^3 + a^3))",
    ),
    (
        "96b",
        "INT(1/(x^3*(a^3 - x^3)^2), x)",
        "(10*SQRT(3)*x^5*SIGN(a)*ATAN(SQRT(3)*(2*x + a)/(3*ABS(a))) - "
        "10*SQRT(3)*a^3*x^2*SIGN(a)*ATAN(SQRT(3)*(2*x + a)/(3*ABS(a))) - 10*x^5*LN(x - "
        "a) + 5*x^5*LN(x^2 + a*x + a^2) + 10*a^3*x^2*LN(x - a) - 5*a^3*x^2*LN(x^2 + "
        "a*x + a^2) - 15*a^2*x^3 + 9*a^5)/(18*a^8*x^2*(x^3 - a^3))",
    ),
    # -- Integrals with a^4 + x^4 --------------------------------------------------------
    (
        "97",
        "INT(1/(a^4 + x^4), x)",
        "-SQRT(2)*(2*ATAN(1 - SQRT(2)*x/a) - 2*ATAN(SQRT(2)*x/a + 1) + LN(x^2 - "
        "SQRT(2)*a*x + a^2) - LN(x^2 + SQRT(2)*a*x + a^2))/(8*a^3)",
    ),
    ("98", "INT(x/(a^4 + x^4), x)", "ATAN(x^2/a^2)/(2*a^2)"),
    (
        "99",
        "INT(x^2/(a^4 + x^4), x)",
        "-SQRT(2)*(2*ATAN(1 - SQRT(2)*x/a) - 2*ATAN(SQRT(2)*x/a + 1) - LN(x^2 - "
        "SQRT(2)*a*x + a^2) + LN(x^2 + SQRT(2)*a*x + a^2))/(8*a)",
    ),
    ("100", "INT(x^3/(a^4 + x^4), x)", "LN(x^4 + a^4)/4"),
    # -- Integrals with a^4 - x^4 --------------------------------------------------------
    (
        "101",
        "INT(1/(a^4 - x^4), x)",
        "(2*ABS(a)*ATAN(x/ABS(a)) + a*(LN(x + a) - LN(x - a)))/(4*a^4)",
    ),
    ("102", "INT(x/(a^4 - x^4), x)", "(LN(x^2 + a^2) - LN(x^2 - a^2))/(4*a^2)"),
    (
        "103",
        "INT(x^2/(a^4 - x^4), x)",
        "(LN(x + a) - LN(x - a))/(4*a) - ATAN(x/ABS(a))/(2*ABS(a))",
    ),
    ("104", "INT(x^3/(a^4 - x^4), x)", "-LN(x^4 - a^4)/4"),
    # -- Integrals with sqrt(x) and a^2 ± b^2*x ------------------------------------------
    (
        "109a",
        "INT(SQRT(x)/(a^2 + b^2*x), x)",
        "2*SQRT(x)/b^2 - 2*a*ATAN(b*SQRT(x)/a)/b^3",
    ),
    (
        "109b",
        "INT(SQRT(x)/(a^2 - b^2*x), x)",
        "-(a*LN(SQRT(x) - a/b) - a*LN(SQRT(x) + a/b) + 2*b*SQRT(x) - a*(LN(-a) - "
        "LN(a)))/b^3",
    ),
    (
        "110a",
        "INT(SQRT(x)^3/(a^2 + b^2*x), x)",
        "2*a^3*ATAN(b*SQRT(x)/a)/b^5 + 2*x^(3/2)/(3*b^2) - 2*a^2*SQRT(x)/b^4",
    ),
    (
        "110b",
        "INT(SQRT(x)^3/(a^2 - b^2*x), x)",
        "-(3*a^3*LN(SQRT(x) - a/b) - 3*a^3*LN(SQRT(x) + a/b) + 2*b^3*x^(3/2) + "
        "6*a^2*b*SQRT(x) - 3*a^3*(LN(-a) - LN(a)))/(3*b^5)",
    ),
    (
        "111a",
        "INT(SQRT(x)/(a^2 + b^2*x)^2, x)",
        "ATAN(b*SQRT(x)/a)/(a*b^3) - SQRT(x)/(b^2*(b^2*x + a^2))",
    ),
    (
        "111b",
        "INT(SQRT(x)/(a^2 - b^2*x)^2, x)",
        "(b^2*x*LN(SQRT(x) - a/b) - b^2*x*LN(SQRT(x) + a/b) - a^2*LN(SQRT(x) - a/b) + "
        "a^2*LN(SQRT(x) + a/b) + b^2*x*(LN(a) - LN(-a)) - 2*a*b*SQRT(x) - a^2*(LN(a) "
        "- LN(-a)))/(2*a*b^3*(b^2*x - a^2))",
    ),
    (
        "112a",
        "INT(SQRT(x)^3/(a^2 + b^2*x)^2, x)",
        "SQRT(x)*(2*b^2*x + 3*a^2)/(b^4*(b^2*x + a^2)) - 3*a*ATAN(b*SQRT(x)/a)/b^5",
    ),
    (
        "112b",
        "INT(SQRT(x)^3/(a^2 - b^2*x)^2, x)",
        "(3*a*b^2*x*LN(SQRT(x) - a/b) - 3*a*b^2*x*LN(SQRT(x) + a/b) - "
        "3*a^3*LN(SQRT(x) - a/b) + 3*a^3*LN(SQRT(x) + a/b) + 4*b^3*x^(3/2) + "
        "3*a*b^2*x*(LN(a) - LN(-a)) - 6*a^2*b*SQRT(x) - 3*a^3*(LN(a) - "
        "LN(-a)))/(2*b^5*(b^2*x - a^2))",
    ),
    ("113a", "INT(1/((a^2 + b^2*x)*SQRT(x)), x)", "2*ATAN(b*SQRT(x)/a)/(a*b)"),
    (
        "113b",
        "INT(1/((a^2 - b^2*x)*SQRT(x)), x)",
        "-(LN(SQRT(x) - a/b) - LN(SQRT(x) + a/b) - LN(-a) + LN(a))/(a*b)",
    ),
    (
        "114a",
        "INT(1/((a^2 + b^2*x)*SQRT(x)^3), x)",
        "-2*b*ATAN(b*SQRT(x)/a)/a^3 - 2/(a^2*SQRT(x))",
    ),
    (
        "114b",
        "INT(1/((a^2 - b^2*x)*SQRT(x)^3), x)",
        "b*(LN(SQRT(x) + a/b) - LN(SQRT(x) - a/b))/a^3 - 2/(a^2*SQRT(x))",
    ),
    (
        "115a",
        "INT(1/((a^2 + b^2*x)^2*SQRT(x)), x)",
        "ATAN(b*SQRT(x)/a)/(a^3*b) + SQRT(x)/(a^2*(b^2*x + a^2))",
    ),
    (
        "115b",
        "INT(1/((a^2 - b^2*x)^2*SQRT(x)), x)",
        "-(b^2*x*LN(SQRT(x) - a/b) - b^2*x*LN(SQRT(x) + a/b) - a^2*LN(SQRT(x) - a/b) "
        "+ a^2*LN(SQRT(x) + a/b) - b^2*x*(LN(-a) - LN(a)) + 2*a*b*SQRT(x) + "
        "a^2*(LN(-a) - LN(a)))/(2*a^3*b*(b^2*x - a^2))",
    ),
    (
        "116a",
        "INT(1/((a^2 + b^2*x)^2*SQRT(x)^3), x)",
        "-3*b*ATAN(b*SQRT(x)/a)/a^5 - (3*b^2*x + 2*a^2)/(a^4*SQRT(x)*(b^2*x + a^2))",
    ),
    (
        "116b",
        "INT(1/((a^2 - b^2*x)^2*SQRT(x)^3), x)",
        "3*b*(LN(SQRT(x) + a/b) - LN(SQRT(x) - a/b))/(2*a^5) + (2*a^2 - "
        "3*b^2*x)/(a^4*SQRT(x)*(b^2*x - a^2))",
    ),
    # -- Other integrals with sqrt(x) ----------------------------------------------------
    (
        "117",
        "INT(SQRT(x)/(a^4 + x^2), x)",
        "-SQRT(2)*(2*ATAN(1 - SQRT(2)*SQRT(x)/a) - 2*ATAN(SQRT(2)*SQRT(x)/a + 1) - "
        "LN(x - SQRT(2)*a*SQRT(x) + a^2) + LN(x + SQRT(2)*a*SQRT(x) + a^2))/(4*a)",
    ),
    (
        "118",
        "INT(1/((a^4 + x^2)*SQRT(x)), x)",
        "-SQRT(2)*(2*ATAN(1 - SQRT(2)*SQRT(x)/a) - 2*ATAN(SQRT(2)*SQRT(x)/a + 1) + "
        "LN(x - SQRT(2)*a*SQRT(x) + a^2) - LN(x + SQRT(2)*a*SQRT(x) + a^2))/(4*a^3)",
    ),
    (
        "119",
        "INT(SQRT(x)/(a^4 - x^2), x)",
        "-(2*ATAN(SQRT(x)/a) + LN(SQRT(x) - a) - LN(SQRT(x) + a))/(2*a)",
    ),
    (
        "120",
        "INT(1/((a^4 - x^2)*SQRT(x)), x)",
        "(2*ATAN(SQRT(x)/a) - LN(SQRT(x) - a) + LN(SQRT(x) + a))/(2*a^3)",
    ),
    # -- Integrals with sqrt(a*x + b) ----------------------------------------------------
    ("121", "INT(SQRT(a*x + b), x)", "2*(a*x + b)^(3/2)/(3*a)"),
    ("122", "INT(x*SQRT(a*x + b), x)", "2*(3*a*x - 2*b)*(a*x + b)^(3/2)/(15*a^2)"),
    (
        "123",
        "INT(x^2*SQRT(a*x + b), x)",
        "2*(a*x + b)^(3/2)*(15*a^2*x^2 - 12*a*b*x + 8*b^2)/(105*a^3)",
    ),
    ("124", "INT(1/SQRT(a*x + b), x)", "2*SQRT(a*x + b)/a"),
    ("125", "INT(x/SQRT(a*x + b), x)", "2*SQRT(a*x + b)*(a*x - 2*b)/(3*a^2)"),
    (
        "126",
        "INT(x^2/SQRT(a*x + b), x)",
        "2*SQRT(a*x + b)*(3*a^2*x^2 - 4*a*b*x + 8*b^2)/(15*a^3)",
    ),
    ("127a", "INT(1/(x*SQRT(a*x + b)), x)", "2*ATAN(SQRT(a*x + b)/SQRT(-b))/SQRT(-b)"),
    ("127b", "INT(1/(x*SQRT(a*x + b)), x)", "2*ATAN(SQRT(a*x + b)/SQRT(-b))/SQRT(-b)"),
    (
        "128",
        "INT(SQRT(a*x + b)/x, x)",
        "2*SQRT(a*x + b) - 2*SQRT(-b)*ATAN(SQRT(a*x + b)/SQRT(-b))",
    ),
    (
        "129",
        "INT(1/(x^2*SQRT(a*x + b)), x)",
        "a*ATAN(SQRT(a*x + b)/SQRT(-b))/(-b)^(3/2) - SQRT(a*x + b)/(b*x)",
    ),
    (
        "130",
        "INT(SQRT(a*x + b)/x^2, x)",
        "a*ATAN(SQRT(a*x + b)/SQRT(-b))/SQRT(-b) - SQRT(a*x + b)/x",
    ),
    (
        "131@n=3",
        "INT(1/(x^3*SQRT(a*x + b)), x)",
        "SQRT(a*x + b)*(3*a*x - 2*b)/(4*b^2*x^2) + 3*a^2*ATAN(SQRT(a*x + "
        "b)/SQRT(-b))/(4*(-b)^(5/2))",
    ),
    ("132", "INT(SQRT(a*x + b)^3, x)", "2*(a*x + b)^(5/2)/(5*a)"),
    ("133", "INT(x*SQRT(a*x + b)^3, x)", "2*(5*a*x - 2*b)*(a*x + b)^(5/2)/(35*a^2)"),
    (
        "134",
        "INT(x^2*SQRT(a*x + b)^3, x)",
        "2*(a*x + b)^(5/2)*(35*a^2*x^2 - 20*a*b*x + 8*b^2)/(315*a^3)",
    ),
    (
        "135",
        "INT(SQRT(a*x + b)^3/x, x)",
        "2*(-b)^(3/2)*ATAN(SQRT(a*x + b)/SQRT(-b)) + 2*a*x*SQRT(a*x + b)/3 + "
        "8*b*SQRT(a*x + b)/3",
    ),
    ("136", "INT(x/SQRT(a*x + b)^3, x)", "(2*a*x + 4*b)/(a^2*SQRT(a*x + b))"),
    (
        "137",
        "INT(x^2/SQRT(a*x + b)^3, x)",
        "(2*a^2*x^2 - 8*a*b*x - 16*b^2)/(3*a^3*SQRT(a*x + b))",
    ),
    (
        "138",
        "INT(1/(x*SQRT(a*x + b)^3), x)",
        "2/(b*SQRT(a*x + b)) - 2*ATAN(SQRT(a*x + b)/SQRT(-b))/(-b)^(3/2)",
    ),
    (
        "139",
        "INT(1/(x^2*SQRT(a*x + b)^3), x)",
        "-(3*a*x + b)/(b^2*x*SQRT(a*x + b)) - 3*a*ATAN(SQRT(a*x + "
        "b)/SQRT(-b))/(-b)^(5/2)",
    ),
    (
        "140a",
        "INT((a*x + b)^(n/2), x)",
        "(2*(a*x + b)^(n/2 + 1) - 2*b^(n/2 + 1))/(a*(n + 2))",
    ),
    (
        "140b",
        "INT((a*x + b)^(-n/2), x)",
        "(2*(a*x + b)^(1 - n/2) - 2*b^(1 - n/2))/(a*(2 - n))",
    ),
    (
        "141a",
        "INT(x*(a*x + b)^(n/2), x)",
        "(a*x + b)^(n/2)*(a^2*x^2*(2*n + 4) + 2*a*b*n*x - 4*b^2)/(a^2*(n^2 + 6*n + 8))",
    ),
    (
        "141b",
        "INT(x*(a*x + b)^(-n/2), x)",
        "-2*(a*x + b)^(-n)*(a^2*x^2*(n - 2) + a*b*n*x + 2*b^2)*SQRT((a*x + "
        "b)^n)/(a^2*(n^2 - 6*n + 8))",
    ),
    (
        "142a",
        "INT(x^2*(a*x + b)^(n/2), x)",
        "(a*x + b)^(n/2)*(a^3*x^3*(2*n^2 + 12*n + 16) + a^2*b*x^2*(2*n^2 + 4*n) - "
        "8*a*b^2*n*x + 16*b^3)/(a^3*(n^3 + 12*n^2 + 44*n + 48))",
    ),
    (
        "142b",
        "INT(x^2*(a*x + b)^(-n/2), x)",
        "-2*(a*x + b)^(-n)*(a^3*x^3*(n^2 - 6*n + 8) + a^2*b*x^2*(n^2 - 2*n) + "
        "4*a*b^2*n*x + 8*b^3)*SQRT((a*x + b)^n)/(a^3*(n^3 - 12*n^2 + 44*n - 48))",
    ),
    (
        "143@n=3",
        "INT((a*x + b)^(3/2)/x, x)",
        "2*(-b)^(3/2)*ATAN(SQRT(a*x + b)/SQRT(-b)) + 2*a*x*SQRT(a*x + b)/3 + "
        "8*b*SQRT(a*x + b)/3",
    ),
    (
        "144@n=3",
        "INT(1/(x*(a*x + b)^(3/2)), x)",
        "2/(b*SQRT(a*x + b)) - 2*ATAN(SQRT(a*x + b)/SQRT(-b))/(-b)^(3/2)",
    ),
    (
        "145@n=3",
        "INT(1/(x^2*(a*x + b)^(3/2)), x)",
        "-(3*a*x + b)/(b^2*x*SQRT(a*x + b)) - 3*a*ATAN(SQRT(a*x + "
        "b)/SQRT(-b))/(-b)^(5/2)",
    ),
    # -- Integrals with sqrt(a*x + b) and sqrt(f*x + g) ----------------------------------
    (
        "146a",
        "INT(1/SQRT((a*x + b)*(f*x + g)), x)",
        "LN(2*SQRT(a*f)*SQRT(a*f*x^2 + x*(a*g + b*f) + b*g) + 2*a*f*x + a*g + "
        "b*f)/SQRT(a*f)",
    ),
    (
        "146b",
        "INT(1/SQRT((a*x + b)*(f*x + g)), x)",
        "LN(2*SQRT(a*f)*SQRT(a*f*x^2 + x*(a*g + b*f) + b*g) + 2*a*f*x + a*g + "
        "b*f)/SQRT(a*f)",
    ),
    (
        "147",
        "INT(x/SQRT((a*x + b)*(f*x + g)), x)",
        "SQRT(a*f*x^2 + x*(a*g + b*f) + b*g)/(a*f) - (a*g + "
        "b*f)*LN(2*SQRT(a*f)*SQRT(a*f*x^2 + x*(a*g + b*f) + b*g) + 2*a*f*x + a*g + "
        "b*f)/(2*a*f*SQRT(a*f))",
    ),
    (
        "148",
        "INT(1/(SQRT(a*x + b)*SQRT(f*x + g)^3), x)",
        "2*SQRT(a*x + b)/((a*g - b*f)*SQRT(f*x + g))",
    ),
    (
        "149a",
        "INT(1/((f*x + g)*SQRT(a*x + b)), x)",
        "2*ATAN(SQRT(f)*SQRT(a*x + b)/SQRT(a*g - b*f))/(SQRT(f)*SQRT(a*g - b*f))",
    ),
    (
        "149b",
        "INT(1/((f*x + g)*SQRT(a*x + b)), x)",
        "2*ATAN(SQRT(f)*SQRT(a*x + b)/SQRT(a*g - b*f))/(SQRT(f)*SQRT(a*g - b*f))",
    ),
    (
        "150",
        "INT(SQRT((a*x + b)*(f*x + g)), x)",
        "x*SQRT(a*f*x^2 + x*(a*g + b*f) + b*g)/2 + (a*g + b*f)*SQRT(a*f*x^2 + x*(a*g "
        "+ b*f) + b*g)/(4*a*f) - (a*g - b*f)^2*LN(2*SQRT(a*f)*SQRT(a*f*x^2 + x*(a*g + "
        "b*f) + b*g) + 2*a*f*x + a*g + b*f)/(8*a*f*SQRT(a*f))",
    ),
    (
        "151",
        "INT(SQRT((f*x + g)/(a*x + b)), x)",
        "x*SQRT((f*x + g)/(a*x + b)) + b*SQRT((f*x + g)/(a*x + b))/a - (a*g - "
        "b*f)*ATAN(SQRT(a)*SQRT((f*x + g)/(a*x + b))/SQRT(-f))/(a^(3/2)*SQRT(-f))",
    ),
    (
        "152",
        "INT(SQRT(a*x + b)/(f*x + g), x)",
        "2*SQRT(a*x + b)/f - 2*SQRT(a*g - b*f)*ATAN(SQRT(f)*SQRT(a*x + b)/SQRT(a*g - "
        "b*f))/f^(3/2)",
    ),
    (
        "153@n=3",
        "INT((f*x + g)^3/SQRT(a*x + b), x)",
        "(10*a^3*f^3*x^3*SQRT(a*x + b) + 2*x^2*(21*a^3*f^2*g - 6*a^2*b*f^3)*SQRT(a*x "
        "+ b) + 2*x*(35*a^3*f*g^2 - 28*a^2*b*f^2*g + 8*a*b^2*f^3)*SQRT(a*x + b) + "
        "2*(35*a^3*g^3 - 70*a^2*b*f*g^2 + 56*a*b^2*f^2*g - 16*b^3*f^3)*SQRT(a*x + b) "
        "- 70*a^3*SQRT(b)*g^3 + 140*a^2*b^(3/2)*f*g^2 - 112*a*b^(5/2)*f^2*g + "
        "32*b^(7/2)*f^3)/(35*a^4)",
    ),
    (
        "154@n=3",
        "INT(1/(SQRT(a*x + b)*(f*x + g)^3), x)",
        "SQRT(a*x + b)*(3*a*f*x + 5*a*g - 2*b*f)/(4*(a*g - b*f)^2*(f*x + g)^2) + "
        "3*a^2*ATAN(SQRT(f)*SQRT(a*x + b)/SQRT(a*g - b*f))/(4*SQRT(f)*(a*g - "
        "b*f)^(5/2))",
    ),
    (
        "155@n=3",
        "INT(SQRT(a*x + b)*(f*x + g)^3, x)",
        "(70*a^4*f^3*x^4*SQRT(a*x + b) + 2*x^3*(135*a^4*f^2*g + 5*a^3*b*f^3)*SQRT(a*x "
        "+ b) + 2*x^2*(189*a^4*f*g^2 + 27*a^3*b*f^2*g - 6*a^2*b^2*f^3)*SQRT(a*x + b) "
        "+ 2*x*(105*a^4*g^3 + 63*a^3*b*f*g^2 - 36*a^2*b^2*f^2*g + "
        "8*a*b^3*f^3)*SQRT(a*x + b) + 2*b*(105*a^3*g^3 - 126*a^2*b*f*g^2 + "
        "72*a*b^2*f^2*g - 16*b^3*f^3)*SQRT(a*x + b) - 210*a^3*b^(3/2)*g^3 + "
        "252*a^2*b^(5/2)*f*g^2 - 144*a*b^(7/2)*f^2*g + 32*b^(9/2)*f^3)/(315*a^4)",
    ),
    (
        "156@n=3",
        "INT(SQRT(a*x + b)/(f*x + g)^3, x)",
        "SQRT(a*x + b)*(a*f*x - a*g + 2*b*f)/(4*f*(a*g - b*f)*(f*x + g)^2) + "
        "a^2*ATAN(SQRT(f)*SQRT(a*x + b)/SQRT(a*g - b*f))/(4*f^(3/2)*(a*g - b*f)^(3/2))",
    ),
    # -- Integrals with sqrt(a^2 - x^2) --------------------------------------------------
    (
        "157",
        "INT(SQRT(a^2 - x^2), x)",
        "a^2*ATAN(x/SQRT(a^2 - x^2))/2 + x*SQRT(a^2 - x^2)/2",
    ),
    ("158", "INT(x*SQRT(a^2 - x^2), x)", "-(a^2 - x^2)^(3/2)/3"),
    (
        "159",
        "INT(x^2*SQRT(a^2 - x^2), x)",
        "a^4*ATAN(x/SQRT(a^2 - x^2))/8 + x^3*SQRT(a^2 - x^2)/4 - a^2*x*SQRT(a^2 - "
        "x^2)/8",
    ),
    ("160", "INT(x^3*SQRT(a^2 - x^2), x)", "-(3*x^2 + 2*a^2)*(a^2 - x^2)^(3/2)/15"),
    (
        "161",
        "INT(SQRT(a^2 - x^2)/x, x)",
        "a*LN((SQRT(a^2 - x^2) - a)/x) + SQRT(a^2 - x^2)",
    ),
    (
        "162",
        "INT(SQRT(a^2 - x^2)/x^2, x)",
        "-ATAN(x/SQRT(a^2 - x^2)) - SQRT(a^2 - x^2)/x",
    ),
    (
        "163",
        "INT(SQRT(a^2 - x^2)/x^3, x)",
        "-LN((SQRT(a^2 - x^2) - a)/x)/(2*a) - SQRT(a^2 - x^2)/(2*x^2)",
    ),
    ("164", "INT(1/SQRT(a^2 - x^2), x)", "ATAN(x/SQRT(a^2 - x^2))"),
    ("165", "INT(x/SQRT(a^2 - x^2), x)", "-SQRT(a^2 - x^2)"),
    (
        "166",
        "INT(x^2/SQRT(a^2 - x^2), x)",
        "a^2*ATAN(x/SQRT(a^2 - x^2))/2 - x*SQRT(a^2 - x^2)/2",
    ),
    ("167", "INT(x^3/SQRT(a^2 - x^2), x)", "-SQRT(a^2 - x^2)*(x^2 + 2*a^2)/3"),
    ("168", "INT(1/(x*SQRT(a^2 - x^2)), x)", "LN((SQRT(a^2 - x^2) - a)/x)/a"),
    ("169", "INT(1/(x^2*SQRT(a^2 - x^2)), x)", "-SQRT(a^2 - x^2)/(a^2*x)"),
    (
        "170",
        "INT(1/(x^3*SQRT(a^2 - x^2)), x)",
        "LN((SQRT(a^2 - x^2) - a)/x)/(2*a^3) - SQRT(a^2 - x^2)/(2*a^2*x^2)",
    ),
    (
        "171",
        "INT(SQRT(a^2 - x^2)^3, x)",
        "3*a^4*ATAN(x/SQRT(a^2 - x^2))/8 - x^3*SQRT(a^2 - x^2)/4 + 5*a^2*x*SQRT(a^2 - "
        "x^2)/8",
    ),
    ("172", "INT(x*SQRT(a^2 - x^2)^3, x)", "-(a^2 - x^2)^(5/2)/5"),
    (
        "173",
        "INT(x^2*SQRT(a^2 - x^2)^3, x)",
        "a^6*ATAN(x/SQRT(a^2 - x^2))/16 - x^5*SQRT(a^2 - x^2)/6 + 7*a^2*x^3*SQRT(a^2 "
        "- x^2)/24 - a^4*x*SQRT(a^2 - x^2)/16",
    ),
    ("174", "INT(x^3*SQRT(a^2 - x^2)^3, x)", "-(5*x^2 + 2*a^2)*(a^2 - x^2)^(5/2)/35"),
    (
        "175",
        "INT(SQRT(a^2 - x^2)^3/x, x)",
        "a^3*LN((SQRT(a^2 - x^2) - a)/x) - x^2*SQRT(a^2 - x^2)/3 + 4*a^2*SQRT(a^2 - "
        "x^2)/3",
    ),
    (
        "176",
        "INT(SQRT(a^2 - x^2)^3/x^2, x)",
        "-3*a^2*ATAN(x/SQRT(a^2 - x^2))/2 - SQRT(a^2 - x^2)*(x^2 + 2*a^2)/(2*x)",
    ),
    (
        "177",
        "INT(SQRT(a^2 - x^2)^3/x^3, x)",
        "-3*a*LN((SQRT(a^2 - x^2) - a)/x)/2 - SQRT(a^2 - x^2)*(2*x^2 + a^2)/(2*x^2)",
    ),
    ("178", "INT(1/SQRT(a^2 - x^2)^3, x)", "x/(a^2*SQRT(a^2 - x^2))"),
    ("179", "INT(x/SQRT(a^2 - x^2)^3, x)", "1/SQRT(a^2 - x^2)"),
    (
        "180",
        "INT(x^2/SQRT(a^2 - x^2)^3, x)",
        "x/SQRT(a^2 - x^2) - ATAN(x/SQRT(a^2 - x^2))",
    ),
    ("181", "INT(x^3/SQRT(a^2 - x^2)^3, x)", "(2*a^2 - x^2)/SQRT(a^2 - x^2)"),
    (
        "182",
        "INT(1/(x*SQRT(a^2 - x^2)^3), x)",
        "LN((SQRT(a^2 - x^2) - a)/x)/a^3 + 1/(a^2*SQRT(a^2 - x^2))",
    ),
    (
        "183",
        "INT(1/(x^2*SQRT(a^2 - x^2)^3), x)",
        "(2*x^2 - a^2)/(a^4*x*SQRT(a^2 - x^2))",
    ),
    (
        "184",
        "INT(1/(x^3*SQRT(a^2 - x^2)^3), x)",
        "3*LN((SQRT(a^2 - x^2) - a)/x)/(2*a^5) + (3*x^2 - a^2)/(2*a^4*x^2*SQRT(a^2 - "
        "x^2))",
    ),
    # -- Integrals with sqrt(x^2 + a^2) --------------------------------------------------
    (
        "185",
        "INT(SQRT(x^2 + a^2), x)",
        "a^2*LN(SQRT(x^2 + a^2) + x)/2 + x*SQRT(x^2 + a^2)/2",
    ),
    ("186", "INT(x*SQRT(x^2 + a^2), x)", "(x^2 + a^2)^(3/2)/3"),
    (
        "187",
        "INT(x^2*SQRT(x^2 + a^2), x)",
        "-a^4*LN(SQRT(x^2 + a^2) + x)/8 + x^3*SQRT(x^2 + a^2)/4 + a^2*x*SQRT(x^2 + "
        "a^2)/8",
    ),
    ("188", "INT(x^3*SQRT(x^2 + a^2), x)", "(3*x^2 - 2*a^2)*(x^2 + a^2)^(3/2)/15"),
    (
        "189",
        "INT(SQRT(x^2 + a^2)/x, x)",
        "a*LN((SQRT(x^2 + a^2) - a)/x) + SQRT(x^2 + a^2)",
    ),
    (
        "190",
        "INT(SQRT(x^2 + a^2)/x^2, x)",
        "LN(SQRT(x^2 + a^2) + x) - SQRT(x^2 + a^2)/x",
    ),
    (
        "191",
        "INT(SQRT(x^2 + a^2)/x^3, x)",
        "LN((SQRT(x^2 + a^2) - a)/x)/(2*a) - SQRT(x^2 + a^2)/(2*x^2)",
    ),
    ("192", "INT(1/SQRT(x^2 + a^2), x)", "LN(SQRT(x^2 + a^2) + x)"),
    ("193", "INT(x/SQRT(x^2 + a^2), x)", "SQRT(x^2 + a^2)"),
    (
        "194",
        "INT(x^2/SQRT(x^2 + a^2), x)",
        "x*SQRT(x^2 + a^2)/2 - a^2*LN(SQRT(x^2 + a^2) + x)/2",
    ),
    ("195", "INT(x^3/SQRT(x^2 + a^2), x)", "SQRT(x^2 + a^2)*(x^2 - 2*a^2)/3"),
    ("196", "INT(1/(x*SQRT(x^2 + a^2)), x)", "LN((SQRT(x^2 + a^2) - a)/x)/a"),
    ("197", "INT(1/(x^2*SQRT(x^2 + a^2)), x)", "-SQRT(x^2 + a^2)/(a^2*x)"),
    (
        "198",
        "INT(1/(x^3*SQRT(x^2 + a^2)), x)",
        "-LN((SQRT(x^2 + a^2) - a)/x)/(2*a^3) - SQRT(x^2 + a^2)/(2*a^2*x^2)",
    ),
    (
        "199",
        "INT(SQRT(x^2 + a^2)^3, x)",
        "3*a^4*LN(SQRT(x^2 + a^2) + x)/8 + x^3*SQRT(x^2 + a^2)/4 + 5*a^2*x*SQRT(x^2 + "
        "a^2)/8",
    ),
    ("200", "INT(x*SQRT(x^2 + a^2)^3, x)", "(x^2 + a^2)^(5/2)/5"),
    (
        "201",
        "INT(x^2*SQRT(x^2 + a^2)^3, x)",
        "-a^6*LN(SQRT(x^2 + a^2) + x)/16 + x^5*SQRT(x^2 + a^2)/6 + 7*a^2*x^3*SQRT(x^2 + "
        "a^2)/24 + a^4*x*SQRT(x^2 + a^2)/16",
    ),
    ("202", "INT(x^3*SQRT(x^2 + a^2)^3, x)", "(5*x^2 - 2*a^2)*(x^2 + a^2)^(5/2)/35"),
    (
        "203",
        "INT(SQRT(x^2 + a^2)^3/x, x)",
        "a^3*LN((SQRT(x^2 + a^2) - a)/x) + x^2*SQRT(x^2 + a^2)/3 + 4*a^2*SQRT(x^2 + "
        "a^2)/3",
    ),
    (
        "204",
        "INT(SQRT(x^2 + a^2)^3/x^2, x)",
        "3*a^2*LN(SQRT(x^2 + a^2) + x)/2 + SQRT(x^2 + a^2)*(x^2 - 2*a^2)/(2*x)",
    ),
    (
        "205",
        "INT(SQRT(x^2 + a^2)^3/x^3, x)",
        "3*a*LN((SQRT(x^2 + a^2) - a)/x)/2 + SQRT(x^2 + a^2)*(2*x^2 - a^2)/(2*x^2)",
    ),
    ("206", "INT(1/SQRT(x^2 + a^2)^3, x)", "x/(a^2*SQRT(x^2 + a^2))"),
    ("207", "INT(x/SQRT(x^2 + a^2)^3, x)", "-1/SQRT(x^2 + a^2)"),
    (
        "208",
        "INT(x^2/SQRT(x^2 + a^2)^3, x)",
        "LN(SQRT(x^2 + a^2) + x) - x/SQRT(x^2 + a^2)",
    ),
    ("209", "INT(x^3/SQRT(x^2 + a^2)^3, x)", "(x^2 + 2*a^2)/SQRT(x^2 + a^2)"),
    (
        "210",
        "INT(1/(x*SQRT(x^2 + a^2)^3), x)",
        "LN((SQRT(x^2 + a^2) - a)/x)/a^3 + 1/(a^2*SQRT(x^2 + a^2))",
    ),
    (
        "211",
        "INT(1/(x^2*SQRT(x^2 + a^2)^3), x)",
        "-(2*x^2 + a^2)/(a^4*x*SQRT(x^2 + a^2))",
    ),
    (
        "212",
        "INT(1/(x^3*SQRT(x^2 + a^2)^3), x)",
        "-3*LN((SQRT(x^2 + a^2) - a)/x)/(2*a^5) - (3*x^2 + a^2)/(2*a^4*x^2*SQRT(x^2 + "
        "a^2))",
    ),
    # -- Integrals with sqrt(x^2 - a^2) --------------------------------------------------
    (
        "213",
        "INT(SQRT(x^2 - a^2), x)",
        "x*SQRT(x^2 - a^2)/2 - a^2*LN(SQRT(x^2 - a^2) + x)/2",
    ),
    ("214", "INT(x*SQRT(x^2 - a^2), x)", "(x^2 - a^2)^(3/2)/3"),
    (
        "215",
        "INT(x^2*SQRT(x^2 - a^2), x)",
        "-a^4*LN(SQRT(x^2 - a^2) + x)/8 + x^3*SQRT(x^2 - a^2)/4 - a^2*x*SQRT(x^2 - "
        "a^2)/8",
    ),
    ("216", "INT(x^3*SQRT(x^2 - a^2), x)", "(3*x^2 + 2*a^2)*(x^2 - a^2)^(3/2)/15"),
    ("217", "INT(SQRT(x^2 - a^2)/x, x)", "SQRT(x^2 - a^2) - a*ATAN(SQRT(x^2 - a^2)/a)"),
    (
        "218",
        "INT(SQRT(x^2 - a^2)/x^2, x)",
        "LN(SQRT(x^2 - a^2) + x) - SQRT(x^2 - a^2)/x",
    ),
    (
        "219",
        "INT(SQRT(x^2 - a^2)/x^3, x)",
        "ATAN(SQRT(x^2 - a^2)/a)/(2*a) - SQRT(x^2 - a^2)/(2*x^2)",
    ),
    ("220", "INT(1/SQRT(x^2 - a^2), x)", "LN(SQRT(x^2 - a^2) + x)"),
    ("221", "INT(x/SQRT(x^2 - a^2), x)", "SQRT(x^2 - a^2)"),
    (
        "222",
        "INT(x^2/SQRT(x^2 - a^2), x)",
        "a^2*LN(SQRT(x^2 - a^2) + x)/2 + x*SQRT(x^2 - a^2)/2",
    ),
    ("223", "INT(x^3/SQRT(x^2 - a^2), x)", "SQRT(x^2 - a^2)*(x^2 + 2*a^2)/3"),
    ("224", "INT(1/(x*SQRT(x^2 - a^2)), x)", "ATAN(SQRT(x^2 - a^2)/a)/a"),
    ("225", "INT(1/(x^2*SQRT(x^2 - a^2)), x)", "SQRT(x^2 - a^2)/(a^2*x)"),
    (
        "226",
        "INT(1/(x^3*SQRT(x^2 - a^2)), x)",
        "ATAN(SQRT(x^2 - a^2)/a)/(2*a^3) + SQRT(x^2 - a^2)/(2*a^2*x^2)",
    ),
    (
        "227",
        "INT(SQRT(x^2 - a^2)^3, x)",
        "3*a^4*LN(SQRT(x^2 - a^2) + x)/8 + x^3*SQRT(x^2 - a^2)/4 - 5*a^2*x*SQRT(x^2 - "
        "a^2)/8",
    ),
    ("228", "INT(x*SQRT(x^2 - a^2)^3, x)", "(x^2 - a^2)^(5/2)/5"),
    (
        "229",
        "INT(x^2*SQRT(x^2 - a^2)^3, x)",
        "a^6*LN(SQRT(x^2 - a^2) + x)/16 + x^5*SQRT(x^2 - a^2)/6 - 7*a^2*x^3*SQRT(x^2 - "
        "a^2)/24 + a^4*x*SQRT(x^2 - a^2)/16",
    ),
    ("230", "INT(x^3*SQRT(x^2 - a^2)^3, x)", "(5*x^2 + 2*a^2)*(x^2 - a^2)^(5/2)/35"),
    (
        "231",
        "INT(SQRT(x^2 - a^2)^3/x, x)",
        "a^3*ATAN(SQRT(x^2 - a^2)/a) + x^2*SQRT(x^2 - a^2)/3 - 4*a^2*SQRT(x^2 - a^2)/3",
    ),
    (
        "232",
        "INT(SQRT(x^2 - a^2)^3/x^2, x)",
        "SQRT(x^2 - a^2)*(x^2 + 2*a^2)/(2*x) - 3*a^2*LN(SQRT(x^2 - a^2) + x)/2",
    ),
    (
        "233",
        "INT(SQRT(x^2 - a^2)^3/x^3, x)",
        "SQRT(x^2 - a^2)*(2*x^2 + a^2)/(2*x^2) - 3*a*ATAN(SQRT(x^2 - a^2)/a)/2",
    ),
    ("234", "INT(1/SQRT(x^2 - a^2)^3, x)", "-x/(a^2*SQRT(x^2 - a^2))"),
    ("235", "INT(x/SQRT(x^2 - a^2)^3, x)", "-1/SQRT(x^2 - a^2)"),
    (
        "236",
        "INT(x^2/SQRT(x^2 - a^2)^3, x)",
        "LN(SQRT(x^2 - a^2) + x) - x/SQRT(x^2 - a^2)",
    ),
    ("237", "INT(x^3/SQRT(x^2 - a^2)^3, x)", "(x^2 - 2*a^2)/SQRT(x^2 - a^2)"),
    (
        "238",
        "INT(1/(x*SQRT(x^2 - a^2)^3), x)",
        "-ATAN(SQRT(x^2 - a^2)/a)/a^3 - 1/(a^2*SQRT(x^2 - a^2))",
    ),
    (
        "239",
        "INT(1/(x^2*SQRT(x^2 - a^2)^3), x)",
        "(a^2 - 2*x^2)/(a^4*x*SQRT(x^2 - a^2))",
    ),
    (
        "240",
        "INT(1/(x^3*SQRT(x^2 - a^2)^3), x)",
        "(a^2 - 3*x^2)/(2*a^4*x^2*SQRT(x^2 - a^2)) - 3*ATAN(SQRT(x^2 - a^2)/a)/(2*a^5)",
    ),
    # -- Integrals with sqrt(a*x^2 + b*x + c) --------------------------------------------
    (
        "241a",
        "INT(1/SQRT(a*x^2 + b*x + c), x)",
        "LN(2*SQRT(a)*SQRT(a*x^2 + b*x + c) + 2*a*x + b)/SQRT(a)",
    ),
    (
        "241b",
        "INT(1/SQRT(a*x^2 + b*x + c), x)",
        "LN(2*SQRT(a)*SQRT(a*x^2 + b*x + c) + 2*a*x + b)/SQRT(a)",
    ),
    (
        "241c",
        "INT(1/SQRT(a*x^2 + b*x + b^2/(4*a)), x)",
        "2*SQRT(a*x^2 + b*x + b^2/(4*a))*LN(2*a*x + b)/(2*a*x + b)",
    ),
    (
        "241d",
        "INT(1/SQRT(a*x^2 + b*x + c), x)",
        "LN(2*SQRT(a)*SQRT(a*x^2 + b*x + c) + 2*a*x + b)/SQRT(a)",
    ),
    (
        "242",
        "INT(1/((a*x^2 + b*x + c)*SQRT(a*x^2 + b*x + c)), x)",
        "(4*a*x + 2*b)/((4*a*c - b^2)*SQRT(a*x^2 + b*x + c))",
    ),
    (
        "243",
        "INT(1/((a*x^2 + b*x + c)^2*SQRT(a*x^2 + b*x + c)), x)",
        "2*(2*a*x + b)*(8*a^2*x^2 + 8*a*b*x + 12*a*c - b^2)/(3*(4*a*c - b^2)^2*(a*x^2 "
        "+ b*x + c)^(3/2))",
    ),
    (
        "244@n=3",
        "INT(1/(a*x^2 + b*x + c)^((2*3 + 1)/2), x)",
        "2*(2*a*x + b)*(128*a^4*x^4 + 256*a^3*b*x^3 + x^2*(320*a^3*c + 112*a^2*b^2) + "
        "x*(320*a^2*b*c - 16*a*b^3) + 240*a^2*c^2 - 40*a*b^2*c + 3*b^4)/(15*(4*a*c - "
        "b^2)^3*(a*x^2 + b*x + c)^(5/2))",
    ),
    (
        "245",
        "INT(SQRT(a*x^2 + b*x + c), x)",
        "(4*a*c - b^2)*LN(2*SQRT(a)*SQRT(a*x^2 + b*x + c) + 2*a*x + b)/(8*a^(3/2)) + "
        "x*SQRT(a*x^2 + b*x + c)/2 + b*SQRT(a*x^2 + b*x + c)/(4*a)",
    ),
    (
        "246",
        "INT((a*x^2 + b*x + c)*SQRT(a*x^2 + b*x + c), x)",
        "3*(4*a*c - b^2)^2*LN(2*SQRT(a)*SQRT(a*x^2 + b*x + c) + 2*a*x + "
        "b)/(128*a^(5/2)) + a*x^3*SQRT(a*x^2 + b*x + c)/4 + 3*b*x^2*SQRT(a*x^2 + b*x "
        "+ c)/8 + x*(b^2/a + 20*c)*SQRT(a*x^2 + b*x + c)/32 + b*(20*a*c - "
        "3*b^2)*SQRT(a*x^2 + b*x + c)/(64*a^2)",
    ),
    (
        "247",
        "INT((a*x^2 + b*x + c)^2*SQRT(a*x^2 + b*x + c), x)",
        "5*(4*a*c - b^2)^3*LN(2*SQRT(a)*SQRT(a*x^2 + b*x + c) + 2*a*x + "
        "b)/(1024*a^(7/2)) + a^2*x^5*SQRT(a*x^2 + b*x + c)/6 + 5*a*b*x^4*SQRT(a*x^2 + "
        "b*x + c)/12 + x^3*(52*a*c + 27*b^2)*SQRT(a*x^2 + b*x + c)/96 + b*x^2*(b^2/a "
        "+ 156*c)*SQRT(a*x^2 + b*x + c)/192 + x*(b^2*(48*a*c - 5*b^2)/a^2 + "
        "528*c^2)*SQRT(a*x^2 + b*x + c)/768 + b*(528*a^2*c^2 - 160*a*b^2*c + "
        "15*b^4)*SQRT(a*x^2 + b*x + c)/(1536*a^3)",
    ),
    (
        "248@n=3",
        "INT((a*x^2 + b*x + c)^((2*3 + 1)/2), x)",
        "35*(4*a*c - b^2)^4*LN(2*SQRT(a)*SQRT(a*x^2 + b*x + c) + 2*a*x + "
        "b)/(32768*a^(9/2)) + a^3*x^7*SQRT(a*x^2 + b*x + c)/8 + "
        "7*a^2*b*x^6*SQRT(a*x^2 + b*x + c)/16 + x^5*(100*a^2*c + "
        "101*a*b^2)*SQRT(a*x^2 + b*x + c)/192 + b*x^4*(500*a*c + 85*b^2)*SQRT(a*x^2 + "
        "b*x + c)/384 + x^3*(2608*a*c^2 + 3*b^4/a + 2696*b^2*c)*SQRT(a*x^2 + b*x + "
        "c)/3072 + b*x^2*(b^2*(88*a*c - 7*b^2)/a^2 + 7824*c^2)*SQRT(a*x^2 + b*x + "
        "c)/6144 + x*(b^2*(2256*a^2*c^2 - 476*a*b^2*c + 35*b^4)/a^3 + "
        "17856*c^3)*SQRT(a*x^2 + b*x + c)/24576 + b*(17856*a^3*c^3 - 8176*a^2*b^2*c^2 "
        "+ 1540*a*b^4*c - 105*b^6)*SQRT(a*x^2 + b*x + c)/(49152*a^4)",
    ),
    (
        "249",
        "INT(x/SQRT(a*x^2 + b*x + c), x)",
        "SQRT(a*x^2 + b*x + c)/a - b*LN(2*SQRT(a)*SQRT(a*x^2 + b*x + c) + 2*a*x + "
        "b)/(2*a^(3/2))",
    ),
    (
        "250",
        "INT(x/((a*x^2 + b*x + c)*SQRT(a*x^2 + b*x + c)), x)",
        "-(2*b*x + 4*c)/((4*a*c - b^2)*SQRT(a*x^2 + b*x + c))",
    ),
    (
        "251@n=3",
        "INT(x/(a*x^2 + b*x + c)^((2*3 + 1)/2), x)",
        "-(256*a^4*b*x^5 + 640*a^3*b^2*x^4 + 2*x^3*(320*a^3*b*c + 240*a^2*b^3) + "
        "2*x^2*(480*a^2*b^2*c + 40*a*b^4) + 2*x*(240*a^2*b*c^2 + 120*a*b^3*c - 5*b^5) "
        "+ 192*a^2*c^3 + 96*a*b^2*c^2 - 4*b^4*c)/(15*(4*a*c - b^2)^3*(a*x^2 + b*x + "
        "c)^(5/2))",
    ),
    (
        "252",
        "INT(x^2/SQRT(a*x^2 + b*x + c), x)",
        "-(4*a*c - 3*b^2)*LN(2*SQRT(a)*SQRT(a*x^2 + b*x + c) + 2*a*x + b)/(8*a^(5/2)) "
        "+ x*SQRT(a*x^2 + b*x + c)/(2*a) - 3*b*SQRT(a*x^2 + b*x + c)/(4*a^2)",
    ),
    (
        "253",
        "INT(x^2/((a*x^2 + b*x + c)*SQRT(a*x^2 + b*x + c)), x)",
        "LN(2*SQRT(a)*SQRT(a*x^2 + b*x + c) + 2*a*x + b)/a^(3/2) + (2*b*c - "
        "2*x*(2*a*c - b^2))/(a*(4*a*c - b^2)*SQRT(a*x^2 + b*x + c))",
    ),
    (
        "254",
        "INT(x*SQRT(a*x^2 + b*x + c), x)",
        "-b*(4*a*c - b^2)*LN(2*SQRT(a)*SQRT(a*x^2 + b*x + c) + 2*a*x + "
        "b)/(16*a^(5/2)) + x^2*SQRT(a*x^2 + b*x + c)/3 + b*x*SQRT(a*x^2 + b*x + "
        "c)/(12*a) + (8*a*c - 3*b^2)*SQRT(a*x^2 + b*x + c)/(24*a^2)",
    ),
    (
        "255",
        "INT(x*(a*x^2 + b*x + c)*SQRT(a*x^2 + b*x + c), x)",
        "-3*b*(4*a*c - b^2)^2*LN(2*SQRT(a)*SQRT(a*x^2 + b*x + c) + 2*a*x + "
        "b)/(256*a^(7/2)) + a*x^4*SQRT(a*x^2 + b*x + c)/5 + 11*b*x^3*SQRT(a*x^2 + b*x "
        "+ c)/40 + x^2*(32*a*c + b^2)*SQRT(a*x^2 + b*x + c)/(80*a) + b*x*(28*a*c - "
        "5*b^2)*SQRT(a*x^2 + b*x + c)/(320*a^2) + (128*a^2*c^2 - 100*a*b^2*c + "
        "15*b^4)*SQRT(a*x^2 + b*x + c)/(640*a^3)",
    ),
    (
        "256@n=3",
        "INT(x*(a*x^2 + b*x + c)^((2*3 + 1)/2), x)",
        "-35*b*(4*a*c - b^2)^4*LN(2*SQRT(a)*SQRT(a*x^2 + b*x + c) + 2*a*x + "
        "b)/(65536*a^(11/2)) + a^3*x^8*SQRT(a*x^2 + b*x + c)/9 + "
        "55*a^2*b*x^7*SQRT(a*x^2 + b*x + c)/144 + x^6*(131072*a^2*c + "
        "132096*a*b^2)*SQRT(a*x^2 + b*x + c)/294912 + x^5*(316416*a*b*c + "
        "53504*b^3)*SQRT(a*x^2 + b*x + c)/294912 + x^4*(1536*a^2*c^2 + 1572*a*b^2*c + "
        "b^4)*SQRT(a*x^2 + b*x + c)/(2304*a) + b*x^3*(16752*a^2*c^2 + 104*a*b^2*c - "
        "9*b^4)*SQRT(a*x^2 + b*x + c)/(18432*a^2) + x^2*(16384*a^3*c^3 + "
        "1104*a^2*b^2*c^2 - 264*a*b^4*c + 21*b^6)*SQRT(a*x^2 + b*x + c)/(36864*a^3) + "
        "b*x*(11968*a^3*c^3 - 6768*a^2*b^2*c^2 + 1428*a*b^4*c - 105*b^6)*SQRT(a*x^2 + "
        "b*x + c)/(147456*a^4) + (32768*a^4*c^4 - 53568*a^3*b^2*c^3 + "
        "24528*a^2*b^4*c^2 - 4620*a*b^6*c + 315*b^8)*SQRT(a*x^2 + b*x + "
        "c)/(294912*a^5)",
    ),
    (
        "257",
        "INT(x^2*SQRT(a*x^2 + b*x + c), x)",
        "-(4*a*c - b^2)*(4*a*c - 5*b^2)*LN(2*SQRT(a)*SQRT(a*x^2 + b*x + c) + 2*a*x + "
        "b)/(128*a^(7/2)) + x^3*SQRT(a*x^2 + b*x + c)/4 + b*x^2*SQRT(a*x^2 + b*x + "
        "c)/(24*a) + x*(12*a*c - 5*b^2)*SQRT(a*x^2 + b*x + c)/(96*a^2) + b*(15*b^2 - "
        "52*a*c)*SQRT(a*x^2 + b*x + c)/(192*a^3)",
    ),
    (
        "258a",
        "INT(1/(x*SQRT(a*x^2 + b*x + c)), x)",
        "LN((2*SQRT(c)*SQRT(a*x^2 + b*x + c) - b*x - 2*c)/x)/SQRT(c)",
    ),
    (
        "258b",
        "INT(1/(x*SQRT(a*x^2 + b*x + c)), x)",
        "LN((2*SQRT(c)*SQRT(a*x^2 + b*x + c) - b*x - 2*c)/x)/SQRT(c)",
    ),
    (
        "258c",
        "INT(1/(x*SQRT(b^2/(4*c)*x^2 + b*x + c)), x)",
        "2*(LN(x) - LN(x + 2*c/b))*SQRT(b^2*x^2/(4*c) + b*x + c)/(b*x + 2*c)",
    ),
    (
        "258d",
        "INT(1/(x*SQRT(a*x^2 + b*x + c)), x)",
        "LN((2*SQRT(c)*SQRT(a*x^2 + b*x + c) - b*x - 2*c)/x)/SQRT(c)",
    ),
    (
        "259",
        "INT(1/(x^2*SQRT(a*x^2 + b*x + c)), x)",
        "-b*LN((2*SQRT(c)*SQRT(a*x^2 + b*x + c) - b*x - 2*c)/x)/(2*c^(3/2)) - "
        "SQRT(a*x^2 + b*x + c)/(c*x)",
    ),
    (
        "260",
        "INT(SQRT(a*x^2 + b*x + c)/x, x)",
        "b*LN(2*SQRT(a)*SQRT(a*x^2 + b*x + c) + 2*a*x + b)/(2*SQRT(a)) + "
        "SQRT(c)*LN(2*SQRT(c)*SQRT(a*x^2 + b*x + c)/x - 2*c/x - b) + SQRT(a*x^2 + b*x "
        "+ c)",
    ),
    (
        "261",
        "INT(SQRT(a*x^2 + b*x + c)/x^2, x)",
        "SQRT(a)*LN(2*SQRT(a)*SQRT(a*x^2 + b*x + c) + 2*a*x + b) + "
        "b*LN((2*SQRT(c)*SQRT(a*x^2 + b*x + c) - b*x - 2*c)/x)/(2*SQRT(c)) - "
        "SQRT(a*x^2 + b*x + c)/x",
    ),
    (
        "262@n=3",
        "INT((a*x^2 + b*x + c)^((2*3 + 1)/2)/x, x)",
        "b*(2240*a^3*c^3 - 560*a^2*b^2*c^2 + 84*a*b^4*c - "
        "5*b^6)*LN(2*SQRT(a)*SQRT(a*x^2 + b*x + c) + 2*a*x + b)/(2048*a^(7/2)) + "
        "c^(7/2)*LN(2*SQRT(c)*SQRT(a*x^2 + b*x + c)/x - 2*c/x - b) + "
        "a^3*x^6*SQRT(a*x^2 + b*x + c)/7 + 43*a^2*b*x^5*SQRT(a*x^2 + b*x + c)/84 + "
        "x^4*(67584*a^2*c + 68480*a*b^2)*SQRT(a*x^2 + b*x + c)/107520 + "
        "x^3*(177728*a*b*c + 30480*b^3)*SQRT(a*x^2 + b*x + c)/107520 + "
        "x^2*(15616*a^2*c^2 + 16428*a*b^2*c + 35*b^4)*SQRT(a*x^2 + b*x + c)/(13440*a) "
        "+ b*x*(111184*a^2*c^2 + 2520*a*b^2*c - 175*b^4)*SQRT(a*x^2 + b*x + "
        "c)/(53760*a^2) + (180224*a^3*c^3 + 48720*a^2*b^2*c^2 - 8120*a*b^4*c + "
        "525*b^6)*SQRT(a*x^2 + b*x + c)/(107520*a^3)",
    ),
    ("263", "INT(1/(x*SQRT(a*x^2 + b*x)), x)", "-2*SQRT(a*x^2 + b*x)/(b*x)"),
    ("264", "INT(1/SQRT(2*a*x - x^2), x)", "-ATAN((a - x)/SQRT(2*a*x - x^2))"),
    (
        "265",
        "INT(x/SQRT(2*a*x - x^2), x)",
        "-a*ATAN((a - x)/SQRT(x*(2*a - x))) - SQRT(2*a*x - x^2)",
    ),
    (
        "266",
        "INT(SQRT(2*a*x - x^2), x)",
        "-a^2*ATAN((a - x)/SQRT(x*(2*a - x)))/2 + x*SQRT(2*a*x - x^2)/2 - "
        "a*SQRT(2*a*x - x^2)/2",
    ),
    # -- Integrals with irrational expressions of other forms ----------------------------
    (
        "267a",
        "INT(1/((a*x^2 + b)*SQRT(f*x^2 + g)), x)",
        "ATAN(x*SQRT(a*g - b*f)/(SQRT(b)*SQRT(f*x^2 + g)))/(SQRT(b)*SQRT(a*g - b*f))",
    ),
    (
        "267b",
        "INT(1/((a*x^2 + b)*SQRT(f*x^2 + g)), x)",
        "ATAN(x*SQRT(a*g - b*f)/(SQRT(b)*SQRT(f*x^2 + g)))/(SQRT(b)*SQRT(a*g - b*f))",
    ),
    (
        "268",
        "INT((a*x + b)^(1/n), x)",
        "n*((a*x + b)^(1/n + 1) - b^(1/n + 1))/(a*(n + 1))",
    ),
    (
        "269",
        "INT(1/(a*x + b)^(1/n), x)",
        "n*((a*x + b)^(1 - 1/n) - b^(1 - 1/n))/(a*(n - 1))",
    ),
    (
        "270",
        "INT(1/(x*SQRT(x^n + a^2)), x)",
        "(LN(SQRT(x^n + a^2) - a) - LN(SQRT(x^n + a^2) + a))/(a*n)",
    ),
    ("271", "INT(1/(x*SQRT(x^n - a^2)), x)", "2*ATAN(SQRT(x^n - a^2)/a)/(a*n)"),
    # -- Reduction formulas for binomial integrals ---------------------------------------
    (
        "273@n=3,m=2",
        "INT(x^2*(a*x^3 + b)^p, x)",
        "IF(a /= 0, IF(p /= -1, (a*x^3 + b)^(p + 1)/(p + 1), LN(a*x^3 + b))/(3*a), "
        "b^p*x^3/3)",
    ),
    # -- Integrals with sine -------------------------------------------------------------
    ("274", "INT(SIN(a*x), x)", "(1 - COS(a*x))/a"),
    ("275", "INT(SIN(a*x)^2, x)", "(2*a*x - SIN(2*a*x))/(4*a)"),
    ("276", "INT(SIN(a*x)^3, x)", "(COS(a*x) + 2)*(COS(a*x) - 1)^2/(3*a)"),
    ("277", "INT(SIN(a*x)^4, x)", "-(8*SIN(2*a*x) - SIN(4*a*x) - 12*a*x)/(32*a)"),
    ("278@n=3", "INT(SIN(a*x)^3, x)", "(COS(a*x) + 2)*(COS(a*x) - 1)^2/(3*a)"),
    ("279", "INT(x*SIN(a*x), x)", "(SIN(a*x)/a - x*COS(a*x))/a"),
    (
        "280",
        "INT(x^2*SIN(a*x), x)",
        "-(a^2*x^2*COS(a*x) - 2*COS(a*x) - 2*a*x*SIN(a*x) + 2)/a^3",
    ),
    (
        "281",
        "INT(x^3*SIN(a*x), x)",
        "-(x^3*COS(a*x) - 6*x*COS(a*x)/a^2 - 3*x^2*SIN(a*x)/a + 6*SIN(a*x)/a^3)/a",
    ),
    (
        "282@n=3",
        "INT(x^3*SIN(a*x), x)",
        "-(x^3*COS(a*x) - 6*x*COS(a*x)/a^2 - 3*x^2*SIN(a*x)/a + 6*SIN(a*x)/a^3)/a",
    ),
    ("283", "INT(SIN(a*x)/x, x)", "SI(a*x)"),
    ("284", "INT(SIN(a*x)/x^2, x)", "a*CI(a*x) - SIN(a*x)/x"),
    (
        "285@n=3",
        "INT(SIN(a*x)/x^3, x)",
        "-a*COS(a*x)/(2*x) - SIN(a*x)/(2*x^2) - a^2*SI(a*x)/2",
    ),
    (
        "286",
        "INT(1/SIN(a*x), x)",
        "IF(a /= 0, (LN(COS(a*x) - 1) - LN(COS(a*x) + 1))/(2*a), 0)",
    ),
    ("287", "INT(1/SIN(a*x)^2, x)", "IF(a /= 0, -1/(a*TAN(a*x)), 0)"),
    (
        "288",
        "INT(1/SIN(a*x)^3, x)",
        "IF(a /= 0, (LN(COS(a*x) - 1) - LN(COS(a*x) + 1) - "
        "2*COS(a*x)/SIN(a*x)^2)/(4*a), 0)",
    ),
    (
        "289@n=3",
        "INT(1/SIN(a*x)^3, x)",
        "IF(a /= 0, (LN(COS(a*x) - 1) - LN(COS(a*x) + 1) - "
        "2*COS(a*x)/SIN(a*x)^2)/(4*a), 0)",
    ),
    (
        "291",
        "INT(x/SIN(a*x)^2, x)",
        "(a*x*TAN(a*x/2)^2 - 2*TAN(a*x/2)*LN(1/(COS(a*x) + 1)) + "
        "2*TAN(a*x/2)*LN(TAN(a*x/2)) - 2*LN(2)*TAN(a*x/2) - a*x)/(2*a^2*TAN(a*x/2))",
    ),
    ("293", "INT(1/(1 + SIN(a*x)), x)", "1/a - COS(a*x)/(a*(SIN(a*x) + 1))"),
    ("294", "INT(1/(1 - SIN(a*x)), x)", "COS(a*x)/(a*(1 - SIN(a*x))) - 1/a"),
    (
        "295",
        "INT(x/(1 + SIN(a*x)), x)",
        "-(LN(1/(COS(a*x) + 1)) - 2*LN(TAN(a*x/2) + 1) + a*x/TAN(a*x/2 + pi/4) + "
        "LN(2))/a^2",
    ),
    (
        "296",
        "INT(x/(1 - SIN(a*x)), x)",
        "(2*LN(1 - TAN(a*x/2)) - LN(1/(COS(a*x) + 1)) + a*x*TAN(a*x/2 + pi/4) - LN(2))"
        "/a^2",
    ),
    (
        "297a",
        "INT(SIN(a*x)/(1 + SIN(a*x)), x)",
        "x - 1/a + COS(a*x)/(a*(SIN(a*x) + 1))",
    ),
    (
        "297b",
        "INT(SIN(a*x)/(1 - SIN(a*x)), x)",
        "-x - 1/a + COS(a*x)/(a*(1 - SIN(a*x)))",
    ),
    (
        "298a",
        "INT(1/(SIN(a*x)*(1 + SIN(a*x))), x)",
        "(LN(TAN(a*x/2)) + COS(a*x)/(SIN(a*x) + 1))/a",
    ),
    (
        "298b",
        "INT(1/(SIN(a*x)*(1 - SIN(a*x))), x)",
        "(LN(TAN(a*x/2)) + COS(a*x)/(1 - SIN(a*x)))/a",
    ),
    (
        "299",
        "INT(1/(1 + SIN(a*x))^2, x)",
        "2/(3*a) - (SIN(a*x) + 2)*COS(a*x)/(3*a*(SIN(a*x) + 1)^2)",
    ),
    (
        "300",
        "INT(1/(1 - SIN(a*x))^2, x)",
        "-2/(3*a) - (SIN(a*x) - 2)*COS(a*x)/(3*a*(1 - SIN(a*x))^2)",
    ),
    (
        "301",
        "INT(SIN(a*x)/(1 + SIN(a*x))^2, x)",
        "-(COS(a*x) + SIN(2*a*x) - (SIN(a*x) + 1)^2)/(3*a*(SIN(a*x) + 1)^2)",
    ),
    (
        "302",
        "INT(SIN(a*x)/(1 - SIN(a*x))^2, x)",
        "-(COS(a*x) - SIN(2*a*x) - (SIN(a*x) - 1)^2)/(3*a*(SIN(a*x) - 1)^2)",
    ),
    (
        "303",
        "INT(1/(1 + SIN(a*x)^2), x)",
        "SQRT(2)*(ATAN(SQRT(2)*TAN(a*x)) + pi*FLOOR(a*x/pi + 1/2))/(2*a)",
    ),
    ("304", "INT(1/(1 - SIN(a*x)^2), x)", "TAN(a*x)/a"),
    (
        "305",
        "INT(SIN(a*x)*SIN(b*x), x)",
        "(b*SIN(a*x)*COS(b*x) - a*SIN(b*x)*COS(a*x))/(a^2 - b^2)",
    ),
    (
        "306a",
        "INT(1/(b + c*SIN(a*x)), x)",
        "(2*ATAN((b*TAN(a*x/2) + c)/SQRT(b^2 - c^2)) + 2*pi*SIGN(b*(b^2 - "
        "c^2))*FLOOR(a*x/(2*pi) + 1/2) - 2*ATAN(c/SQRT(b^2 - c^2)))/(a*SQRT(b^2 - "
        "c^2))",
    ),
    (
        "306b",
        "INT(1/(b + c*SIN(a*x)), x)",
        "(2*ATAN((b*TAN(a*x/2) + c)/SQRT(b^2 - c^2)) + 2*pi*SIGN(b*(b^2 - "
        "c^2))*FLOOR(a*x/(2*pi) + 1/2) - 2*ATAN(c/SQRT(b^2 - c^2)))/(a*SQRT(b^2 - "
        "c^2))",
    ),
    (
        "307",
        "INT(SIN(a*x)/(b + c*SIN(a*x)), x)",
        "-(2*b*ATAN((b*TAN(a*x/2) + c)/SQRT(b^2 - c^2)) + 2*pi*b*SIGN(b*(b^2 - "
        "c^2))*FLOOR(a*x/(2*pi) + 1/2) - 2*b*ATAN(c/SQRT(b^2 - c^2)) - a*x*SQRT(b^2 - "
        "c^2))/(a*c*SQRT(b^2 - c^2))",
    ),
    (
        "308",
        "INT(1/(SIN(a*x)*(b + c*SIN(a*x))), x)",
        "-(2*c*ATAN((b*TAN(a*x/2) + c)/SQRT(b^2 - c^2)) - SQRT(b^2 - "
        "c^2)*LN(TAN(a*x/2)) + 2*pi*c*SIGN(b*(b^2 - c^2))*FLOOR(a*x/(2*pi) + "
        "1/2))/(a*b*SQRT(b^2 - c^2))",
    ),
    (
        "309",
        "INT(1/(b + c*SIN(a*x))^2, x)",
        "(2*b^2*(c*SIN(a*x) + b)*ATAN((b*TAN(a*x/2) + c)/SQRT(b^2 - c^2)) + "
        "b*c*SQRT(b^2 - c^2)*COS(a*x) + (c*SIN(a*x) + b)*(2*pi*b^2*SIGN(b*(b^2 - "
        "c^2))*FLOOR(a*x/(2*pi) + 1/2) - 2*b^2*ATAN(c/SQRT(b^2 - c^2)) - c*SQRT(b^2 - "
        "c^2)))/(a*b*(b^2 - c^2)^(3/2)*(c*SIN(a*x) + b))",
    ),
    (
        "310",
        "INT(SIN(a*x)/(b + c*SIN(a*x))^2, x)",
        "-(2*c*(c*SIN(a*x) + b)*ATAN((b*TAN(a*x/2) + c)/SQRT(b^2 - c^2)) + b*SQRT(b^2 "
        "- c^2)*COS(a*x) - (c*SIN(a*x) + b)*(-2*pi*c*SIGN(b*(b^2 - "
        "c^2))*FLOOR(a*x/(2*pi) + 1/2) + 2*c*ATAN(c/SQRT(b^2 - c^2)) + SQRT(b^2 - "
        "c^2)))/(a*(b^2 - c^2)^(3/2)*(c*SIN(a*x) + b))",
    ),
    (
        "311",
        "INT(1/(b^2 + c^2*SIN(a*x)^2), x)",
        "(ATAN(SQRT(b^2 + c^2)*TAN(a*x)/ABS(b)) + pi*FLOOR(a*x/pi + 1/2))/(a*SQRT(b^2 "
        "+ c^2)*ABS(b))",
    ),
    (
        "312a",
        "INT(1/(b^2 - c^2*SIN(a*x)^2), x)",
        "(ATAN(SQRT(b^2 - c^2)*TAN(a*x)/ABS(b)) + pi*FLOOR(a*x/pi + 1/2))/(a*SQRT(b^2 "
        "- c^2)*ABS(b))",
    ),
    (
        "312b",
        "INT(1/(b^2 - c^2*SIN(a*x)^2), x)",
        "(ATAN(SQRT(b^2 - c^2)*TAN(a*x)/ABS(b)) + pi*FLOOR(a*x/pi + 1/2))/(a*SQRT(b^2 "
        "- c^2)*ABS(b))",
    ),
    # -- Integrals with cosine -----------------------------------------------------------
    ("313", "INT(COS(a*x), x)", "SIN(a*x)/a"),
    ("314", "INT(COS(a*x)^2, x)", "(SIN(2*a*x) + 2*a*x)/(4*a)"),
    ("315", "INT(COS(a*x)^3, x)", "(3 - SIN(a*x)^2)*SIN(a*x)/(3*a)"),
    ("316", "INT(COS(a*x)^4, x)", "(8*SIN(2*a*x) + SIN(4*a*x) + 12*a*x)/(32*a)"),
    ("317@n=3", "INT(COS(a*x)^3, x)", "(3 - SIN(a*x)^2)*SIN(a*x)/(3*a)"),
    ("318", "INT(x*COS(a*x), x)", "(COS(a*x) + a*x*SIN(a*x) - 1)/a^2"),
    ("319", "INT(x^2*COS(a*x), x)", "(2*x*COS(a*x)/a + x^2*SIN(a*x) - 2*SIN(a*x)/a^2)/a"),
    (
        "320",
        "INT(x^3*COS(a*x), x)",
        "(3*a^2*x^2*COS(a*x) - 6*COS(a*x) + a^3*x^3*SIN(a*x) - 6*a*x*SIN(a*x) + 6)/a^4",
    ),
    (
        "321@n=3",
        "INT(x^3*COS(a*x), x)",
        "(3*a^2*x^2*COS(a*x) - 6*COS(a*x) + a^3*x^3*SIN(a*x) - 6*a*x*SIN(a*x) + 6)/a^4",
    ),
    ("322", "INT(COS(a*x)/x, x)", "CI(a*x)"),
    ("323", "INT(COS(a*x)/x^2, x)", "-COS(a*x)/x - a*SI(a*x)"),
    (
        "324@n=3",
        "INT(COS(a*x)/x^3, x)",
        "-COS(a*x)/(2*x^2) + a*SIN(a*x)/(2*x) - a^2*CI(a*x)/2",
    ),
    ("325", "INT(1/COS(a*x), x)", "(LN(SIN(a*x) + 1) - LN(1 - SIN(a*x)))/(2*a)"),
    ("326", "INT(1/COS(a*x)^2, x)", "TAN(a*x)/a"),
    (
        "327",
        "INT(1/COS(a*x)^3, x)",
        "-(LN(1 - SIN(a*x)) - LN(SIN(a*x) + 1) - 2*SIN(a*x)/COS(a*x)^2)/(4*a)",
    ),
    (
        "328@n=3",
        "INT(1/COS(a*x)^3, x)",
        "-(LN(1 - SIN(a*x)) - LN(SIN(a*x) + 1) - 2*SIN(a*x)/COS(a*x)^2)/(4*a)",
    ),
    (
        "330",
        "INT(x/COS(a*x)^2, x)",
        "-(LN(1/(COS(a*x) + 1)) - LN(TAN(a*x/2) - 1) - LN(TAN(a*x/2) + 1) - "
        "a*x*TAN(a*x/2)/COS(a*x) - a*x*TAN(a*x/2) + LN(2))/a^2",
    ),
    ("332", "INT(1/(1 + COS(a*x)), x)", "TAN(a*x/2)/a"),
    ("333", "INT(1/(1 - COS(a*x)), x)", "-1/(a*TAN(a*x/2))"),
    (
        "334",
        "INT(x/(1 + COS(a*x)), x)",
        "-(LN(1/(COS(a*x) + 1)) - a*x*TAN(a*x/2) + LN(2))/a^2",
    ),
    (
        "335",
        "INT(x/(1 - COS(a*x)), x)",
        "-(TAN(a*x/2)*LN(1/(COS(a*x) + 1)) - 2*TAN(a*x/2)*LN(TAN(a*x/2)) + "
        "LN(2)*TAN(a*x/2) + a*x)/(a^2*TAN(a*x/2))",
    ),
    ("336", "INT(COS(a*x)/(1 + COS(a*x)), x)", "x - TAN(a*x/2)/a"),
    ("337", "INT(COS(a*x)/(1 - COS(a*x)), x)", "-1/(a*TAN(a*x/2)) - x"),
    (
        "338",
        "INT(1/(COS(a*x)*(1 + COS(a*x))), x)",
        "-(LN(1 - SIN(a*x)) - LN(SIN(a*x) + 1) + 2*TAN(a*x/2))/(2*a)",
    ),
    (
        "339",
        "INT(1/(COS(a*x)*(1 - COS(a*x))), x)",
        "-(LN(1 - SIN(a*x)) - LN(SIN(a*x) + 1) + 2/TAN(a*x/2))/(2*a)",
    ),
    ("340", "INT(1/(1 + COS(a*x))^2, x)", "(TAN(a*x/2)^2 + 3)*TAN(a*x/2)/(6*a)"),
    ("341", "INT(1/(1 - COS(a*x))^2, x)", "-(1/TAN(a*x/2)^2 + 3)/(6*a*TAN(a*x/2))"),
    ("342", "INT(COS(a*x)/(1 + COS(a*x))^2, x)", "(3 - TAN(a*x/2)^2)*TAN(a*x/2)/(6*a)"),
    ("343", "INT(COS(a*x)/(1 - COS(a*x))^2, x)", "(3 - 1/TAN(a*x/2)^2)/(6*a*TAN(a*x/2))"),
    (
        "344",
        "INT(1/(1 + COS(a*x)^2), x)",
        "SQRT(2)*(ATAN(SQRT(2)*TAN(a*x)/2) + pi*FLOOR(a*x/pi + 1/2))/(2*a)",
    ),
    ("345", "INT(1/(1 - COS(a*x)^2), x)", "-1/(a*TAN(a*x))"),
    (
        "346",
        "INT(COS(a*x)*COS(b*x), x)",
        "(a*SIN(a*x)*COS(b*x) - b*SIN(b*x)*COS(a*x))/(a^2 - b^2)",
    ),
    (
        "347a",
        "INT(1/(b + c*COS(a*x)), x)",
        "(2*ATAN((b - c)*TAN(a*x/2)/SQRT(b^2 - c^2)) + 2*pi*SIGN(b + "
        "c)*FLOOR(a*x/(2*pi) + 1/2))/(a*SQRT(b^2 - c^2))",
    ),
    (
        "347b",
        "INT(1/(b + c*COS(a*x)), x)",
        "(2*ATAN((b - c)*TAN(a*x/2)/SQRT(b^2 - c^2)) + 2*pi*SIGN(b + "
        "c)*FLOOR(a*x/(2*pi) + 1/2))/(a*SQRT(b^2 - c^2))",
    ),
    (
        "348",
        "INT(COS(a*x)/(b + c*COS(a*x)), x)",
        "-(2*b*ATAN((b - c)*TAN(a*x/2)/SQRT(b^2 - c^2)) + 2*pi*b*SIGN(b + "
        "c)*FLOOR(a*x/(2*pi) + 1/2) - a*x*SQRT(b^2 - c^2))/(a*c*SQRT(b^2 - c^2))",
    ),
    (
        "349",
        "INT(1/(COS(a*x)*(b + c*COS(a*x))), x)",
        "-(4*c*ATAN((b - c)*TAN(a*x/2)/SQRT(b^2 - c^2)) + 4*pi*c*SIGN(b + "
        "c)*FLOOR(a*x/(2*pi) + 1/2) - SQRT(b^2 - c^2)*(LN(SIN(a*x) + 1) - LN(1 - "
        "SIN(a*x))))/(2*a*b*SQRT(b^2 - c^2))",
    ),
    (
        "350",
        "INT(1/(b + c*COS(a*x))^2, x)",
        "(2*b*(c*COS(a*x) + b)*ATAN((b - c)*TAN(a*x/2)/SQRT(b^2 - c^2)) + "
        "2*pi*b*c*SIGN(b + c)*FLOOR(a*x/(2*pi) + 1/2)*COS(a*x) - c*SQRT(b^2 - "
        "c^2)*SIN(a*x) + 2*pi*b^2*SIGN(b + c)*FLOOR(a*x/(2*pi) + 1/2))/(a*(b^2 - "
        "c^2)^(3/2)*(c*COS(a*x) + b))",
    ),
    (
        "351",
        "INT(COS(a*x)/(b + c*COS(a*x))^2, x)",
        "-(2*c*(c*COS(a*x) + b)*ATAN((b - c)*TAN(a*x/2)/SQRT(b^2 - c^2)) + "
        "2*pi*c^2*SIGN(b + c)*FLOOR(a*x/(2*pi) + 1/2)*COS(a*x) - b*SQRT(b^2 - "
        "c^2)*SIN(a*x) + 2*pi*b*c*SIGN(b + c)*FLOOR(a*x/(2*pi) + 1/2))/(a*(b^2 - "
        "c^2)^(3/2)*(c*COS(a*x) + b))",
    ),
    (
        "352",
        "INT(1/(b^2 + c^2*COS(a*x)^2), x)",
        "(ATAN(ABS(b)*TAN(a*x)/SQRT(b^2 + c^2)) + pi*FLOOR(a*x/pi + 1/2))/(a*SQRT(b^2 "
        "+ c^2)*ABS(b))",
    ),
    (
        "353a",
        "INT(1/(b^2 - c^2*COS(a*x)^2), x)",
        "(ATAN(ABS(b)*TAN(a*x)/SQRT(b^2 - c^2)) + pi*SIGN(b^2 - c^2)*FLOOR(a*x/pi + "
        "1/2))/(a*SQRT(b^2 - c^2)*ABS(b))",
    ),
    (
        "353b",
        "INT(1/(b^2 - c^2*COS(a*x)^2), x)",
        "(ATAN(ABS(b)*TAN(a*x)/SQRT(b^2 - c^2)) + pi*SIGN(b^2 - c^2)*FLOOR(a*x/pi + "
        "1/2))/(a*SQRT(b^2 - c^2)*ABS(b))",
    ),
    # -- Integrals with sine and cosine --------------------------------------------------
    ("354", "INT(SIN(a*x)*COS(a*x), x)", "SIN(a*x)^2/(2*a)"),
    ("355", "INT(SIN(a*x)^2*COS(a*x)^2, x)", "(4*a*x - SIN(4*a*x))/(32*a)"),
    (
        "356",
        "INT(SIN(a*x)^n*COS(a*x), x)",
        "(SIN(a*x)^(n + 1) - 1)/(a*(n + 1))",
    ),
    (
        "357",
        "INT(SIN(a*x)*COS(a*x)^n, x)",
        "(1 - COS(a*x)^(n + 1))/(a*(n + 1))",
    ),
    (
        "358a@n=3,m=2",
        "INT(SIN(a*x)^3*COS(a*x)^2, x)",
        "COS(a*x)^5/(5*a) - COS(a*x)^3/(3*a) + 2/(15*a)",
    ),
    (
        "358b@n=3,m=2",
        "INT(SIN(a*x)^3*COS(a*x)^2, x)",
        "COS(a*x)^5/(5*a) - COS(a*x)^3/(3*a) + 2/(15*a)",
    ),
    (
        "359",
        "INT(1/(SIN(a*x)*COS(a*x)), x)",
        "IF(a /= 0, (2*LN(SIN(a*x)) - LN(SIN(a*x)^2 - 1))/(2*a), 0)",
    ),
    (
        "360",
        "INT(1/(SIN(a*x)^2*COS(a*x)), x)",
        "IF(a /= 0, -(LN(SIN(a*x) - 1) - LN(SIN(a*x) + 1) + 2/SIN(a*x))/(2*a), 0)",
    ),
    (
        "361",
        "INT(1/(SIN(a*x)*COS(a*x)^2), x)",
        "IF(a /= 0, (LN(COS(a*x) - 1) - LN(COS(a*x) + 1) + 2/COS(a*x))/(2*a), 0)",
    ),
    (
        "362",
        "INT(1/(SIN(a*x)^3*COS(a*x)), x)",
        "IF(a /= 0, -(LN(SIN(a*x)^2 - 1) - 2*LN(SIN(a*x)) + 1/SIN(a*x)^2)/(2*a), 0)",
    ),
    (
        "363",
        "INT(1/(SIN(a*x)*COS(a*x)^3), x)",
        "IF(a /= 0, (LN(COS(a*x)^2 - 1) - 2*LN(COS(a*x)) + 1/COS(a*x)^2)/(2*a), 0)",
    ),
    ("364", "INT(1/(SIN(a*x)^2*COS(a*x)^2), x)", "IF(a /= 0, -2/(a*TAN(2*a*x)), 0)"),
    (
        "365",
        "INT(1/(SIN(a*x)^2*COS(a*x)^3), x)",
        "IF(a /= 0, -(3*LN(SIN(a*x) - 1) - 3*LN(SIN(a*x) + 1) - 2*SIN(a*x)/COS(a*x)^2 "
        "+ 4/SIN(a*x))/(4*a), 0)",
    ),
    (
        "366",
        "INT(1/(SIN(a*x)^3*COS(a*x)^2), x)",
        "IF(a /= 0, (3*LN(COS(a*x) - 1) - 3*LN(COS(a*x) + 1) - 2*COS(a*x)/SIN(a*x)^2 "
        "+ 4/COS(a*x))/(4*a), 0)",
    ),
    (
        "367@n=3",
        "INT(1/(SIN(a*x)*COS(a*x)^3), x)",
        "IF(a /= 0, (LN(COS(a*x)^2 - 1) - 2*LN(COS(a*x)) + 1/COS(a*x)^2)/(2*a), 0)",
    ),
    (
        "368@n=3",
        "INT(1/(SIN(a*x)^3*COS(a*x)), x)",
        "IF(a /= 0, -(LN(SIN(a*x)^2 - 1) - 2*LN(SIN(a*x)) + 1/SIN(a*x)^2)/(2*a), 0)",
    ),
    (
        "369a@n=3,m=2",
        "INT(1/(SIN(a*x)^3*COS(a*x)^2), x)",
        "IF(a /= 0, (3*LN(COS(a*x) - 1) - 3*LN(COS(a*x) + 1) - 2*COS(a*x)/SIN(a*x)^2 "
        "+ 4/COS(a*x))/(4*a), 0)",
    ),
    (
        "369b@n=3,m=2",
        "INT(1/(SIN(a*x)^3*COS(a*x)^2), x)",
        "IF(a /= 0, (3*LN(COS(a*x) - 1) - 3*LN(COS(a*x) + 1) - 2*COS(a*x)/SIN(a*x)^2 "
        "+ 4/COS(a*x))/(4*a), 0)",
    ),
    ("370", "INT(SIN(a*x)/COS(a*x)^2, x)", "(1/COS(a*x) - 1)/a"),
    ("371", "INT(SIN(a*x)/COS(a*x)^3, x)", "TAN(a*x)^2/(2*a)"),
    (
        "372",
        "INT(SIN(a*x)/COS(a*x)^n, x)",
        "(COS(a*x)^(1 - n) - 1)/(a*(n - 1))",
    ),
    (
        "373",
        "INT(SIN(a*x)^2/COS(a*x), x)",
        "-(LN(1 - SIN(a*x)) - LN(SIN(a*x) + 1) + 2*SIN(a*x))/(2*a)",
    ),
    (
        "374",
        "INT(SIN(a*x)^2/COS(a*x)^3, x)",
        "(LN(1 - SIN(a*x)) - LN(SIN(a*x) + 1) + 2*SIN(a*x)/COS(a*x)^2)/(4*a)",
    ),
    (
        "375@n=3",
        "INT(SIN(a*x)^2/COS(a*x)^3, x)",
        "(LN(1 - SIN(a*x)) - LN(SIN(a*x) + 1) + 2*SIN(a*x)/COS(a*x)^2)/(4*a)",
    ),
    ("376", "INT(SIN(a*x)^3/COS(a*x), x)", "-(SIN(a*x)^2 + 2*LN(COS(a*x)))/(2*a)"),
    ("377", "INT(SIN(a*x)^3/COS(a*x)^2, x)", "(COS(a*x) - 1)^2/(a*COS(a*x))"),
    (
        "378@n=4",
        "INT(SIN(a*x)^3/COS(a*x)^4, x)",
        "((1 - 3*COS(a*x)^2)/COS(a*x)^3 + 2)/(3*a)",
    ),
    ("379@n=3", "INT(SIN(a*x)^3/COS(a*x), x)", "-(SIN(a*x)^2 + 2*LN(COS(a*x)))/(2*a)"),
    ("380a@n=3,m=2", "INT(SIN(a*x)^3/COS(a*x)^2, x)", "(COS(a*x) - 1)^2/(a*COS(a*x))"),
    ("380b@n=3,m=2", "INT(SIN(a*x)^3/COS(a*x)^2, x)", "(COS(a*x) - 1)^2/(a*COS(a*x))"),
    ("380c@n=3,m=2", "INT(SIN(a*x)^3/COS(a*x)^2, x)", "(COS(a*x) - 1)^2/(a*COS(a*x))"),
    ("381", "INT(COS(a*x)/SIN(a*x)^2, x)", "IF(a /= 0, -1/(a*SIN(a*x)), 0)"),
    ("382", "INT(COS(a*x)/SIN(a*x)^3, x)", "IF(a /= 0, -1/(2*a*SIN(a*x)^2), 0)"),
    (
        "383",
        "INT(COS(a*x)/SIN(a*x)^n, x)",
        "(SIN(a*x)^(1 - n) - 1)/(a*(1 - n))",
    ),
    (
        "384",
        "INT(COS(a*x)^2/SIN(a*x), x)",
        "IF(a /= 0, (LN(COS(a*x) - 1) - LN(COS(a*x) + 1) + 2*COS(a*x))/(2*a), 0)",
    ),
    (
        "385",
        "INT(COS(a*x)^2/SIN(a*x)^3, x)",
        "IF(a /= 0, -(LN(COS(a*x) - 1) - LN(COS(a*x) + 1) + "
        "2*COS(a*x)/SIN(a*x)^2)/(4*a), 0)",
    ),
    (
        "386@n=3",
        "INT(COS(a*x)^2/SIN(a*x)^3, x)",
        "IF(a /= 0, -(LN(COS(a*x) - 1) - LN(COS(a*x) + 1) + "
        "2*COS(a*x)/SIN(a*x)^2)/(4*a), 0)",
    ),
    (
        "387",
        "INT(COS(a*x)^3/SIN(a*x), x)",
        "IF(a /= 0, (2*LN(SIN(a*x)) - SIN(a*x)^2)/(2*a), 0)",
    ),
    ("388", "INT(COS(a*x)^3/SIN(a*x)^2, x)", "IF(a /= 0, -(SIN(a*x) + 1/SIN(a*x))/a, 0)"),
    (
        "389@n=4",
        "INT(COS(a*x)^3/SIN(a*x)^4, x)",
        "IF(a /= 0, (3 - 1/SIN(a*x)^2)/(3*a*SIN(a*x)), 0)",
    ),
    (
        "390@n=3",
        "INT(COS(a*x)^3/SIN(a*x), x)",
        "IF(a /= 0, (2*LN(SIN(a*x)) - SIN(a*x)^2)/(2*a), 0)",
    ),
    (
        "391a@n=3,m=2",
        "INT(COS(a*x)^3/SIN(a*x)^2, x)",
        "IF(a /= 0, -(SIN(a*x) + 1/SIN(a*x))/a, 0)",
    ),
    (
        "391b@n=3,m=2",
        "INT(COS(a*x)^3/SIN(a*x)^2, x)",
        "IF(a /= 0, -(SIN(a*x) + 1/SIN(a*x))/a, 0)",
    ),
    (
        "391c@n=3,m=2",
        "INT(COS(a*x)^3/SIN(a*x)^2, x)",
        "IF(a /= 0, -(SIN(a*x) + 1/SIN(a*x))/a, 0)",
    ),
    (
        "392a",
        "INT(1/(SIN(a*x)*(1 + COS(a*x))), x)",
        "(LN(1 - COS(a*x)) - LN(COS(a*x) + 1) + 2/(COS(a*x) + 1))/(4*a)",
    ),
    (
        "392b",
        "INT(1/(SIN(a*x)*(1 - COS(a*x))), x)",
        "(LN(1 - COS(a*x)) - LN(COS(a*x) + 1) + 2/(COS(a*x) - 1))/(4*a)",
    ),
    (
        "393a",
        "INT(1/(COS(a*x)*(1 + SIN(a*x))), x)",
        "-((SIN(a*x) + 1)*LN(1 - SIN(a*x)) - (SIN(a*x) + 1)*LN(SIN(a*x) + 1) - "
        "2*SIN(a*x))/(4*a*(SIN(a*x) + 1))",
    ),
    (
        "393b",
        "INT(1/(COS(a*x)*(1 - SIN(a*x))), x)",
        "-((SIN(a*x) - 1)*LN(1 - SIN(a*x)) - (SIN(a*x) - 1)*LN(SIN(a*x) + 1) + "
        "2*SIN(a*x))/(4*a*(SIN(a*x) - 1))",
    ),
    (
        "394a",
        "INT(SIN(a*x)/(COS(a*x)*(1 + COS(a*x))), x)",
        "(LN(COS(a*x) + 1) - LN(COS(a*x)) - LN(2))/a",
    ),
    (
        "394b",
        "INT(SIN(a*x)/(COS(a*x)*(1 - COS(a*x))), x)",
        "(LN(1 - COS(a*x)) - LN(COS(a*x)))/a",
    ),
    (
        "395a",
        "INT(COS(a*x)/(SIN(a*x)*(1 + SIN(a*x))), x)",
        "(LN(SIN(a*x)) - LN(SIN(a*x) + 1))/a",
    ),
    (
        "395b",
        "INT(COS(a*x)/(SIN(a*x)*(1 - SIN(a*x))), x)",
        "(LN(SIN(a*x)) - LN(1 - SIN(a*x)))/a",
    ),
    (
        "396a",
        "INT(SIN(a*x)/(COS(a*x)*(1 + SIN(a*x))), x)",
        "-((SIN(a*x) + 1)*LN(1 - SIN(a*x)) - (SIN(a*x) + 1)*LN(SIN(a*x) + 1) + "
        "2*SIN(a*x))/(4*a*(SIN(a*x) + 1))",
    ),
    (
        "396b",
        "INT(SIN(a*x)/(COS(a*x)*(1 - SIN(a*x))), x)",
        "((SIN(a*x) - 1)*LN(1 - SIN(a*x)) - (SIN(a*x) - 1)*LN(SIN(a*x) + 1) - "
        "2*SIN(a*x))/(4*a*(SIN(a*x) - 1))",
    ),
    (
        "397a",
        "INT(COS(a*x)/(SIN(a*x)*(1 + COS(a*x))), x)",
        "(LN(1 - COS(a*x)) - LN(COS(a*x) + 1) - 2/(COS(a*x) + 1))/(4*a)",
    ),
    (
        "397b",
        "INT(COS(a*x)/(SIN(a*x)*(1 - COS(a*x))), x)",
        "-(LN(1 - COS(a*x)) - LN(COS(a*x) + 1) - 2/(COS(a*x) - 1))/(4*a)",
    ),
    (
        "398a",
        "INT(SIN(a*x)/(SIN(a*x) + COS(a*x)), x)",
        "x/2 - LN(COS(a*x) + SIN(a*x))/(2*a)",
    ),
    (
        "398b",
        "INT(SIN(a*x)/(SIN(a*x) - COS(a*x)), x)",
        "LN(COS(a*x) - SIN(a*x))/(2*a) + x/2",
    ),
    (
        "399a",
        "INT(COS(a*x)/(SIN(a*x) + COS(a*x)), x)",
        "LN(COS(a*x) + SIN(a*x))/(2*a) + x/2",
    ),
    (
        "399b",
        "INT(COS(a*x)/(SIN(a*x) - COS(a*x)), x)",
        "LN(COS(a*x) - SIN(a*x))/(2*a) - x/2",
    ),
    (
        "400a",
        "INT(1/(SIN(a*x) + COS(a*x)), x)",
        "SQRT(2)*(LN((1 - COS(a*x + pi/4))/SIN(a*x + pi/4)) - LN(SQRT(2) - 1))/(2*a)",
    ),
    (
        "400b",
        "INT(1/(SIN(a*x) - COS(a*x)), x)",
        "SQRT(2)*(LN(SQRT(2) + 1) - LN(TAN(a*x + pi/4) + 1/COS(a*x + pi/4)))/(2*a)",
    ),
    ("401a", "INT(1/(1 + COS(a*x) + SIN(a*x)), x)", "LN(TAN(a*x/2) + 1)/a"),
    ("401b", "INT(1/(1 + COS(a*x) - SIN(a*x)), x)", "-LN(1 - TAN(a*x/2))/a"),
    (
        "402",
        "INT(1/(b*SIN(a*x) + c*COS(a*x)), x)",
        "(LN(-(b*COS(a*x) - c*SIN(a*x) - SQRT(b^2 + c^2))/(c*COS(a*x) + b*SIN(a*x))) "
        "- LN((SQRT(b^2 + c^2) - b)/c))/(a*SQRT(b^2 + c^2))",
    ),
    (
        "403",
        "INT(SIN(a*x)/(b + c*COS(a*x)), x)",
        "(LN(b + c) - LN(c*COS(a*x) + b))/(a*c)",
    ),
    ("404", "INT(COS(a*x)/(b + c*SIN(a*x)), x)", "(LN(c*SIN(a*x) + b) - LN(b))/(a*c)"),
    (
        "405",
        "INT(1/(b + c*COS(a*x) + f*SIN(a*x)), x)",
        "(2*ATAN(((b - c)*TAN(a*x/2) + f)/SQRT(b^2 - c^2 - f^2)) + 2*pi*SIGN((b - "
        "c)*(b^2 - c^2 - f^2))*FLOOR(a*x/(2*pi) + 1/2) - 2*ATAN(f/SQRT(b^2 - c^2 - "
        "f^2)))/(a*SQRT(b^2 - c^2 - f^2))",
    ),
    (
        "406",
        "INT(1/(b^2*COS(a*x)^2 + c^2*SIN(a*x)^2), x)",
        "(ATAN(ABS(c)*TAN(a*x)/ABS(b)) + pi*FLOOR(a*x/pi + 1/2))/(a*ABS(b)*ABS(c))",
    ),
    (
        "407",
        "INT(1/(b^2*COS(a*x)^2 - c^2*SIN(a*x)^2), x)",
        "LN((b*COS(a*x) + c*SIN(a*x))/(b*COS(a*x) - c*SIN(a*x)))/(2*a*b*c)",
    ),
    (
        "408",
        "INT(SIN(a*x)*COS(b*x), x)",
        "-(a*COS(a*x)*COS(b*x) + b*SIN(a*x)*SIN(b*x))/(a^2 - b^2)",
    ),
    # -- Integrals with tangent ----------------------------------------------------------
    ("409", "INT(TAN(a*x), x)", "-LN(COS(a*x))/a"),
    ("410", "INT(TAN(a*x)^2, x)", "(TAN(a*x) - a*x)/a"),
    ("411", "INT(TAN(a*x)^3, x)", "(TAN(a*x)^2 + 2*LN(COS(a*x)))/(2*a)"),
    ("412@n=3", "INT(TAN(a*x)^3, x)", "(TAN(a*x)^2 + 2*LN(COS(a*x)))/(2*a)"),
    ("415", "INT(TAN(a*x)^n/COS(a*x)^2, x)", "(TAN(a*x)^(n + 1) - 1)/(a*(n + 1))"),
    ("415@n=3", "INT(TAN(a*x)^3/COS(a*x)^2, x)", "TAN(a*x)^4/(4*a)"),
    ("416a", "INT(1/(TAN(a*x) + 1), x)", "LN(COS(a*x) + SIN(a*x))/(2*a) + x/2"),
    ("416b", "INT(1/(TAN(a*x) - 1), x)", "LN(COS(a*x) - SIN(a*x))/(2*a) - x/2"),
    ("417a", "INT(TAN(a*x)/(TAN(a*x) + 1), x)", "x/2 - LN(COS(a*x) + SIN(a*x))/(2*a)"),
    ("417b", "INT(TAN(a*x)/(TAN(a*x) - 1), x)", "LN(COS(a*x) - SIN(a*x))/(2*a) + x/2"),
    # -- Integrals with cotangent --------------------------------------------------------
    ("418", "INT(COT(a*x), x)", "IF(a /= 0, LN(SIN(a*x))/a, 0)"),
    ("419", "INT(COT(a*x)^2, x)", "IF(a /= 0, -(1/TAN(a*x) + a*x)/a, 0)"),
    (
        "420",
        "INT(COT(a*x)^3, x)",
        "IF(a /= 0, -(2*LN(SIN(a*x)) + 1/SIN(a*x)^2)/(2*a), 0)",
    ),
    (
        "421@n=3",
        "INT(COT(a*x)^3, x)",
        "IF(a /= 0, -(2*LN(SIN(a*x)) + 1/SIN(a*x)^2)/(2*a), 0)",
    ),
    ("424", "INT(COT(a*x)^n/SIN(a*x)^2, x)", "(1 - COT(a*x)^(n + 1))/(a*(n + 1))"),
    (
        "424@n=3",
        "INT(COT(a*x)^3/SIN(a*x)^2, x)",
        "IF(a /= 0, -COS(2*a*x)/(4*a*SIN(a*x)^4), 0)",
    ),
    ("425a", "INT(1/(1 + COT(a*x)), x)", "x/2 - LN(COS(a*x) + SIN(a*x))/(2*a)"),
    ("425b", "INT(1/(1 - COT(a*x)), x)", "LN(COS(a*x) - SIN(a*x))/(2*a) + x/2"),
    # -- Integrals of hyperbolic functions -----------------------------------------------
    ("426", "INT(SINH(a*x), x)", "#e^(a*x)/(2*a) + #e^(-a*x)/(2*a) - 1/a"),
    ("427", "INT(COSH(a*x), x)", "(#e^(a*x) - #e^(-a*x))/(2*a)"),
    (
        "428",
        "INT(SINH(a*x)^2, x)",
        "#e^(2*a*x)/(8*a) - #e^(-2*a*x)/(8*a) - x/2",
    ),
    (
        "429",
        "INT(COSH(a*x)^2, x)",
        "#e^(2*a*x)/(8*a) - #e^(-2*a*x)/(8*a) + x/2",
    ),
    (
        "430a@n=3",
        "INT(SINH(a*x)^3, x)",
        "#e^(3*a*x)/(24*a) - 3*#e^(a*x)/(8*a) - 3*#e^(-a*x)/(8*a) + #e^(-3*a*x)/(24*a) "
        "+ 2/(3*a)",
    ),
    (
        "431a@n=3",
        "INT(COSH(a*x)^3, x)",
        "#e^(3*a*x)/(24*a) + 3*#e^(a*x)/(8*a) - 3*#e^(-a*x)/(8*a) - #e^(-3*a*x)/(24*a)",
    ),
    (
        "432",
        "INT(1/SINH(a*x), x)",
        "LN((#e^(a*x/2) - #e^(-a*x/2))/(#e^(a*x/2) + #e^(-a*x/2)))/a",
    ),
    (
        "433",
        "INT(1/COSH(a*x), x)",
        "2*ATAN((#e^(a*x/2) - #e^(-a*x/2))/(#e^(a*x/2) + #e^(-a*x/2)))/a",
    ),
    (
        "434",
        "INT(x*SINH(a*x), x)",
        "x*(#e^(a*x) + #e^(-a*x))/(2*a) + (#e^(-a*x) - #e^(a*x))/(2*a^2)",
    ),
    (
        "435",
        "INT(x*COSH(a*x), x)",
        "x*(#e^(a*x) - #e^(-a*x))/(2*a) - (#e^(a*x) + #e^(-a*x))/(2*a^2) + 1/a^2",
    ),
    ("436", "INT(TANH(a*x), x)", "LN(#e^(a*x)/2 + #e^(-a*x)/2)/a"),
    ("437", "INT(COTH(a*x), x)", "IF(a /= 0, LN(#e^(a*x)/2 - #e^(-a*x)/2)/a, (±inf)*x)"),
    (
        "438",
        "INT(TANH(a*x)^2, x)",
        "((-LN(2 - 2*#e^(a*x)/(#e^(a*x) + #e^(-a*x))) + LN(2/(#e^(a*x) + "
        "#e^(-a*x))) + a*x - 2)*#e^(2*a*x) - LN(2 - 2*#e^(a*x)/(#e^(a*x) + "
        "#e^(-a*x))) + LN(2/(#e^(a*x) + #e^(-a*x))) + a*x + 2)/(2*a*(#e^(2*a*x) + 1))",
    ),
    ("439", "INT(COTH(a*x)^2, x)", "x - 2/(a*#e^(2*a*x) - a)"),
    (
        "440",
        "INT(SINH(a*x)*SINH(b*x), x)",
        "(#e^(b*x)/(a + b) - #e^(-b*x)/(a - b))*#e^(a*x)/4 + (#e^(b*x)/(a - b) - "
        "#e^(-b*x)/(a + b))*#e^(-a*x)/4",
    ),
    (
        "441",
        "INT(COSH(a*x)*COSH(b*x), x)",
        "(#e^(b*x)/(a + b) + #e^(-b*x)/(a - b))*#e^(a*x)/4 - (#e^(b*x)/(a - b) + "
        "#e^(-b*x)/(a + b))*#e^(-a*x)/4",
    ),
    (
        "442",
        "INT(COSH(a*x)*SINH(b*x), x)",
        "(#e^(b*x)/(a + b) - #e^(-b*x)/(a - b))*#e^(a*x)/4 + (#e^(-b*x)/(a + b) - "
        "#e^(b*x)/(a - b))*#e^(-a*x)/4",
    ),
    (
        "443",
        "INT(SINH(a*x)*SIN(a*x), x)",
        "SQRT(2)*(SIN(a*x + pi/4)*#e^(-a*x) - COS(a*x + pi/4)*#e^(a*x))/(4*a)",
    ),
    (
        "444",
        "INT(COSH(a*x)*COS(a*x), x)",
        "SQRT(2)*(SIN(a*x + pi/4)*#e^(a*x) - COS(a*x + pi/4)*#e^(-a*x))/(4*a)",
    ),
    (
        "445",
        "INT(SINH(a*x)*COS(a*x), x)",
        "(SQRT(2)*COS(a*x + pi/4)*#e^(-a*x) + SQRT(2)*SIN(a*x + pi/4)*#e^(a*x) - "
        "2)/(4*a)",
    ),
    (
        "446",
        "INT(COSH(a*x)*SIN(a*x), x)",
        "-(SQRT(2)*COS(a*x + pi/4)*#e^(a*x) + SQRT(2)*SIN(a*x + pi/4)*#e^(-a*x) - "
        "2)/(4*a)",
    ),
    # -- Integrals of exponential functions ----------------------------------------------
    ("447", "INT(#e^(a*x), x)", "(#e^(a*x) - 1)/a"),
    ("448", "INT(x*#e^(a*x), x)", "(a*x*#e^(a*x) - #e^(a*x) + 1)/a^2"),
    (
        "449",
        "INT(x^2*#e^(a*x), x)",
        "(a^2*x^2*#e^(a*x) - 2*a*x*#e^(a*x) + 2*#e^(a*x) - 2)/a^3",
    ),
    (
        "450@n=3",
        "INT(x^3*#e^(a*x), x)",
        "(a^3*x^3*#e^(a*x) - 3*a^2*x^2*#e^(a*x) + 6*a*x*#e^(a*x) - 6*#e^(a*x) + 6)/a^4",
    ),
    ("451", "INT(#e^(a*x)/x, x)", "EI(a*x)"),
    (
        "452@n=3",
        "INT(#e^(a*x)/x^3, x)",
        "-a*#e^(a*x)/(2*x) - #e^(a*x)/(2*x^2) + a^2*EI(a*x)/2",
    ),
    ("453", "INT(1/(1 + #e^(a*x)), x)", "-(LN(#e^(a*x) + 1) - a*x - LN(2))/a"),
    (
        "454",
        "INT(1/(b + c*#e^(a*x)), x)",
        "-(LN(#e^(a*x) + b/c) - LN(b/c + 1) - a*x)/(a*b)",
    ),
    (
        "455",
        "INT(#e^(a*x)/(b + c*#e^(a*x)), x)",
        "(LN(c*#e^(a*x) + b) - LN(b + c))/(a*c)",
    ),
    (
        "456a",
        "INT(1/(b*#e^(a*x) + c*#e^(-a*x)), x)",
        "(ATAN(SQRT(b)*#e^(a*x)/SQRT(c)) - ATAN(SQRT(b)/SQRT(c)))/(a*SQRT(b)*SQRT(c))",
    ),
    (
        "456b",
        "INT(1/(b*#e^(a*x) + c*#e^(-a*x)), x)",
        "(ATAN(SQRT(b)*#e^(a*x)/SQRT(c)) - ATAN(SQRT(b)/SQRT(c)))/(a*SQRT(b)*SQRT(c))",
    ),
    ("457", "INT(x*#e^(a*x)/(1 + a*x)^2, x)", "#e^(a*x)/(a^3*x + a^2)"),
    ("458", "INT(#e^(a*x)*LN(x), x)", "(LN(x)*#e^(a*x) - EI(a*x))/a"),
    (
        "459",
        "INT(#e^(a*x)*SIN(b*x), x)",
        "(a*SIN(b*x) - b*COS(b*x))*#e^(a*x)/(a^2 + b^2)",
    ),
    (
        "460",
        "INT(#e^(a*x)*COS(b*x), x)",
        "(a*COS(b*x) + b*SIN(b*x))*#e^(a*x)/(a^2 + b^2)",
    ),
    (
        "461@n=3",
        "INT(#e^(a*x)*SIN(x)^3, x)",
        "((a^3 + a)*SIN(x)^3 - ((3*a^2 + 3)*SIN(x)^2 + 6)*COS(x) + "
        "6*a*SIN(x))*#e^(a*x)/(a^4 + 10*a^2 + 9)",
    ),
    (
        "462@n=3",
        "INT(#e^(a*x)*COS(x)^3, x)",
        "((a^3 + a)*COS(x)^3 + (3*a^2 + 3)*SIN(x)*COS(x)^2 + 6*a*COS(x) + "
        "6*SIN(x))*#e^(a*x)/(a^4 + 10*a^2 + 9)",
    ),
    (
        "463",
        "INT(x*#e^(a*x)*SIN(b*x), x)",
        "-(x*(a^2*b + b^3)*COS(b*x) - 2*a*b*COS(b*x) - x*(a^3 + a*b^2)*SIN(b*x) - (b^2 "
        "- a^2)*SIN(b*x))*#e^(a*x)/(a^4 + 2*a^2*b^2 + b^4)",
    ),
    (
        "464",
        "INT(x*#e^(a*x)*COS(b*x), x)",
        "(x*(a^3 + a*b^2)*COS(b*x) + (b^2 - a^2)*COS(b*x) + x*(a^2*b + b^3)*SIN(b*x) - "
        "2*a*b*SIN(b*x))*#e^(a*x)/(a^4 + 2*a^2*b^2 + b^4)",
    ),
    # -- Integrals of logarithmic functions ----------------------------------------------
    ("465", "INT(LN(x), x)", "x*LN(x) - x"),
    ("466", "INT(LN(x)^2, x)", "x*LN(x)^2 - 2*x*LN(x) + 2*x"),
    ("467", "INT(LN(x)^3, x)", "x*LN(x)^3 - 3*x*LN(x)^2 + 6*x*LN(x) - 6*x"),
    ("468@n=3", "INT(LN(x)^3, x)", "x*LN(x)^3 - 3*x*LN(x)^2 + 6*x*LN(x) - 6*x"),
    ("469", "INT(1/LN(x), x)", "EI(LN(x))"),
    ("470@n=3", "INT(1/LN(x)^3, x)", "EI(LN(x))/2 - x*(LN(x) + 1)/(2*LN(x)^2)"),
    (
        "471",
        "INT(x^m*LN(x), x)",
        "(x^(m + 1)*(m + 1)*LN(x) - x^(m + 1) + 1)/(m^2 + 2*m + 1)",
    ),
    (
        "472@n=3,m=2",
        "INT(x^2*LN(x)^3, x)",
        "x^3*LN(x)^3/3 - x^3*LN(x)^2/3 + 2*x^3*LN(x)/9 - 2*x^3/27",
    ),
    ("473", "INT(LN(x)^n/x, x)", "(LN(x)^(n + 1) - 1)/(n + 1)"),
    ("474", "INT(LN(x)/x^m, x)", "x^(-m)*(x*(1 - m)*LN(x) - x + x^m)/(m^2 - 2*m + 1)"),
    ("475@n=3,m=2", "INT(LN(x)^3/x^2, x)", "-(LN(x)^3 + 3*LN(x)^2 + 6*LN(x) + 6)/x"),
    ("476", "INT(x^m/LN(x), x)", "EI((m + 1)*LN(x))"),
    (
        "477@n=3,m=2",
        "INT(x^2/LN(x)^3, x)",
        "9*EI(3*LN(x))/2 - x^3*(3*LN(x) + 1)/(2*LN(x)^2)",
    ),
    ("478", "INT(1/(x*LN(x)), x)", "LN(LN(x))"),
    ("479", "INT(1/(x^n*LN(x)), x)", "EI((1 - n)*LN(x))"),
    ("480", "INT(1/(x*LN(x)^n), x)", "(LN(x)^(1 - n) - 1)/(1 - n)"),
    ("485", "INT(SIN(LN(x)), x)", "x*SIN(LN(x))/2 - x*COS(LN(x))/2"),
    ("486", "INT(COS(LN(x)), x)", "x*COS(LN(x))/2 + x*SIN(LN(x))/2"),
    ("487", "INT(#e^(a*x)*LN(x), x)", "(LN(x)*#e^(a*x) - EI(a*x))/a"),
    # -- Integrals of inverse trigonometric functions ------------------------------------
    ("488", "INT(ASIN(x/a), x)", "x*ASIN(x/a) + a*SQRT(1 - x^2/a^2)"),
    (
        "489",
        "INT(x*ASIN(x/a), x)",
        "x^2*ASIN(x/a)/2 + a^3*SQRT(-1/a^2)*LN(2*SQRT(-1/a^2)*SQRT(1 - x^2/a^2) - "
        "2*x/a^2)/4 + a*x*SQRT(1 - x^2/a^2)/4",
    ),
    (
        "490",
        "INT(x^2*ASIN(x/a), x)",
        "x^3*ASIN(x/a)/3 + a*x^2*SQRT(1 - x^2/a^2)/9 + 2*a^3*SQRT(1 - x^2/a^2)/9",
    ),
    ("492", "INT(ASIN(x/a)/x^2, x)", "LN((SQRT(1 - x^2/a^2) - 1)/x)/a - ASIN(x/a)/x"),
    ("493", "INT(ACOS(x/a), x)", "x*ACOS(x/a) - a*SQRT(1 - x^2/a^2)"),
    (
        "494",
        "INT(x*ACOS(x/a), x)",
        "-a^3*SQRT(-1/a^2)*LN(2*SQRT(-1/a^2)*SQRT(1 - x^2/a^2) - 2*x/a^2)/4 + "
        "x^2*ACOS(x/a)/2 - a*x*SQRT(1 - x^2/a^2)/4",
    ),
    (
        "495",
        "INT(x^2*ACOS(x/a), x)",
        "x^3*ACOS(x/a)/3 - a*x^2*SQRT(1 - x^2/a^2)/9 - 2*a^3*SQRT(1 - x^2/a^2)/9",
    ),
    ("497", "INT(ACOS(x/a)/x^2, x)", "-LN((SQRT(1 - x^2/a^2) - 1)/x)/a - ACOS(x/a)/x"),
    ("498", "INT(ATAN(x/a), x)", "x*ATAN(x/a) - a*LN(x^2 + a^2)/2"),
    (
        "499",
        "INT(x*ATAN(x/a), x)",
        "x^2*ATAN(x/a)/2 + a^2*ATAN(x/a)/2 - a*x/2",
    ),
    ("500", "INT(x^2*ATAN(x/a), x)", "x^3*ATAN(x/a)/3 + a^3*LN(x^2 + a^2)/6 - a*x^2/6"),
    (
        "501@n=3",
        "INT(x^3*ATAN(x/a), x)",
        "x^4*ATAN(x/a)/4 - a^4*ATAN(x/a)/4 - a*x^3/12 + a^3*x/4",
    ),
    ("503", "INT(ATAN(x/a)/x^2, x)", "-ATAN(x/a)/x - LN(a^2/x^2 + 1)/(2*a)"),
    (
        "504@n=3",
        "INT(ATAN(x/a)/x^3, x)",
        "-ATAN(x/a)/(2*x^2) - (ATAN(x/ABS(a))/ABS(a) + 1/x)/(2*a)",
    ),
    ("505", "INT(ACOT(x/a), x)", "a*LN(x^2 + a^2)/2 + x*ACOT(x/a)"),
    (
        "506",
        "INT(x*ACOT(x/a), x)",
        "-a^2*ATAN(x/a)/2 + x^2*ACOT(x/a)/2 + a*x/2",
    ),
    ("507", "INT(x^2*ACOT(x/a), x)", "-a^3*LN(x^2 + a^2)/6 + x^3*ACOT(x/a)/3 + a*x^2/6"),
    (
        "508@n=3",
        "INT(x^3*ACOT(x/a), x)",
        "a^4*ATAN(x/a)/4 + x^4*ACOT(x/a)/4 + a*x^3/12 - a^3*x/4",
    ),
    ("510", "INT(ACOT(x/a)/x^2, x)", "LN(a^2/x^2 + 1)/(2*a) - ACOT(x/a)/x"),
    (
        "511@n=3",
        "INT(ACOT(x/a)/x^3, x)",
        "(ATAN(x/ABS(a))/ABS(a) + 1/x)/(2*a) - ACOT(x/a)/(2*x^2)",
    ),
    # -- Integrals of inverse hyperbolic (area) functions --------------------------------
    ("512", "INT(ASINH(x/a), x)", "x*LN(SQRT(x^2/a^2 + 1) + x/a) - a*SQRT(x^2/a^2 + 1)"),
    (
        "513",
        "INT(ACOSH(x/a), x)",
        "2*x*LN(SQRT(x/a - 1) + SQRT(x/a + 1)) - a*SQRT(x/a + 1)*SQRT(x/a - 1) - LN(2)*x",
    ),
    (
        "514",
        "INT(ATANH(x/a), x)",
        "-x*LN(1 - x/a)/2 + x*LN(x/a + 1)/2 + a*LN(x^2 - a^2)/2",
    ),
    ("515", "INT(ACOTH(x/a), x)", "x*LN(2/(x/a - 1) + 1)/2 + a*LN(x^2 - a^2)/2"),
    # -- Substitution rule ---------------------------------------------------------------
    (
        "p454.1",
        "INT(SQRT(a^2 - x^2), x, 0, a)",
        "IF(a /= 0, a^2*ASIN(a/ABS(a))/2, -#i*a^2*LN(-a)/2 + #i*a^2*LN(#i*ABS(a))/2)",
    ),
    ("p454.2", "INT(COS(z), z, 0, pi)", "0"),
    # -- Integration by parts ------------------------------------------------------------
    ("p454.3", "INT(x*#e^x, x, 0, 1)", "1"),
    # -- Differentiation under the integral sign -----------------------------------------
    ("p455.1", "INT((x - 1)/LN(x), x, 0, 1)", "LN(2)"),
    ("p455.3", "INT(x^t, x, 0, 1)", "IF(t > -1, 1/(t + 1), inf)"),
    ("p455.4", "INT(1/(s + 1), s, 0, t)", "LN(t + 1)"),
    # -- Integration by series expansion -------------------------------------------------
    ("p456.1", "INT(#e^(-x^2), x)", "SQRT(pi)*ERF(x)/2"),
    ("p456.2", "INT(#e^(-x^2), x, 0, 1/2)", "SQRT(pi)*ERF(1/2)/2"),
    # -- Applications of definite integrals: general principle ---------------------------
    ("p460.1", "INT(s*h^2/c^2, h, 0, c)", "c*s/3"),
    # -- Improper integrals with infinite limits -----------------------------------------
    ("p466.1", "INT(1/x, x, 1, inf)", "inf"),
    ("p466.2", "INT(1/x^2, x, 2, inf)", "1/2"),
    ("p466.3", "INT(1/(1 + x^2), x, -inf, inf)", "pi"),
    ("p468.2", "INT(x^(3/2)/(1 + x^2), x, 0, inf)", "inf"),
    # -- Improper integrals of unbounded functions ---------------------------------------
    ("p470.1", "INT(1/SQRT(x), x, 0, b)", "2*SQRT(b)"),
    ("p470.2", "INT(TAN(x), x, 0, pi/2)", "inf"),
    ("p471.1", "INT(2*x/(x^2 - 1), x, -2, 2)", "?"),
    # -- Integrals depending on a parameter ----------------------------------------------
    ("p472.1", "INT(x^(y - 1)*#e^(-x), x, 0, inf)", "IF(y > 0, (y - 1)!, inf)"),
    ("p472.2", "DIF(INT(ATAN(x/y), x, 0, 1), y)", "-LN(1/y^2 + 1)/2"),
    ("p472.3", "INT(x/(x^2 + y^2), x, 0, 1)", "LN(y^2 + 1)/2 - LN(ABS(y))"),
    ("p473.1", "INT(ATAN(x/y), x, 0, 1)", "ATAN(1/y) - y*LN(1/y^2 + 1)/2"),
    # -- Integration under the integral sign ---------------------------------------------
    ("p473.3", "INT(1/(1 + y), y, a, b)", "LN(b + 1) - LN(a + 1)"),
    ("p473.4", "INT(x^y, y, a, b)", "IF(x > 0 AND x /= 1, (x^b - x^a)/LN(x), b - a)"),
    ("p474.1", "INT((y^2 - x^2)/(x^2 + y^2)^2, x, 0, 1)", "1/(y^2 + 1)"),
    ("p474.2", "INT(1/(1 + y^2), y, 0, 1)", "pi/4"),
    ("p474.3", "INT((y^2 - x^2)/(x^2 + y^2)^2, y, 0, 1)", "-1/(x^2 + 1)"),
    ("p474.4", "INT(-1/(x^2 + 1), x, 0, 1)", "-pi/4"),
    ("p474.5", "INT(INT((y^2 - x^2)/(x^2 + y^2)^2, x, 0, 1), y, 0, 1)", "pi/4"),
    ("p474.6", "INT(INT((y^2 - x^2)/(x^2 + y^2)^2, y, 0, 1), x, 0, 1)", "-pi/4"),
    # -- Exponential functions -----------------------------------------------------------
    ("D1a@n=3", "INT(x^3*#e^(-a*x), x, 0, inf)", "IF(a > 0, 6/a^4, inf)"),
    ("D1b@n=3", "INT(x^3*#e^(-a*x), x, 0, inf)", "IF(a > 0, 6/a^4, inf)"),
    ("D2a@n=3", "INT(x^3*#e^(-a*x^2), x, 0, inf)", "IF(a > 0, 1/(2*a^2), inf)"),
    ("D3", "INT(#e^(-a^2*x^2), x, 0, inf)", "SQRT(pi)/(2*ABS(a))"),
    ("D4", "INT(x^2*#e^(-a^2*x^2), x, 0, inf)", "SQRT(pi)*ABS(a)/(4*a^4)"),
    ("D9", "INT(#e^(-x)*LN(x), x, 0, inf)", "-euler_gamma"),
    # -- Trigonometric functions ---------------------------------------------------------
    ("D11a", "INT(SIN(a*x)/x, x, 0, inf)", "pi*SIGN(a)/2"),
    ("D11b", "INT(SIN(a*x)/x, x, 0, inf)", "pi*SIGN(a)/2"),
    ("D12", "INT(COS(a*x)/x, x, 0, alpha)", "inf"),
    ("D16a", "INT(SIN(x)/SQRT(x), x, 0, inf)", "SQRT(2)*SQRT(pi)/2"),
    ("D16b", "INT(COS(x)/SQRT(x), x, 0, inf)", "SQRT(2)*SQRT(pi)/2"),
    (
        "D18",
        "INT(COS(a*x)/(1 + x^2), x, 0, inf)",
        "pi*#e^a/4 - pi*#e^ABS(a)/4 + pi*#e^(-ABS(a))/4 + pi*#e^(-a)/4",
    ),
    ("D19", "INT(SIN(a*x)^2/x^2, x, 0, inf)", "pi*ABS(a)/2"),
    ("D20a", "INT(SIN(x^2), x, -inf, inf)", "SQRT(2)*SQRT(pi)/2"),
    ("D20b", "INT(COS(x^2), x, -inf, inf)", "SQRT(2)*SQRT(pi)/2"),
    # -- Logarithmic functions -----------------------------------------------------------
    ("D27", "INT(LN(x)/(x - 1), x, 0, 1)", "pi^2/6"),
    ("D34", "INT(SIN(x)*LN(SIN(x)), x, 0, pi/2)", "LN(2) - 1"),
    ("D37", "INT(LN(TAN(x)), x, 0, pi/2)", "0"),
    # -- Algebraic functions -------------------------------------------------------------
    (
        "D44",
        "INT(1/(1 + 2*x*COS(a) + x^2), x, 0, 1)",
        "(ATAN((COS(a) + 1)/ABS(SIN(a))) - ASIN(COS(a)))/ABS(SIN(a))",
    ),
    (
        "D45",
        "INT(1/(1 + 2*x*COS(a) + x^2), x, 0, inf)",
        "-(2*ASIN(COS(a)) - pi)*ABS(SIN(a))/(2*SIN(a)^2)",
    ),
]


@pytest.mark.parametrize(
    ("text", "expected"),
    [pytest.param(text, expected, id=id_) for id_, text, expected in CASES],
)
def test_bronstein_integral(text, expected):
    assert simp(text) == expected
