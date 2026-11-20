"""Expressions as data: strings and their character codes, `STRING`, the recognizers and
type predicates, identity, and `SUBST`.

The recognizers (`INTEGER?` ... `LOGICAL?`) test the form of their argument's simplified
value and the type predicates (`INTEGER_TYPE?` ... `LOGICAL_TYPE?`) what it is known to
be; both answer true or false at once. `STRING` takes a value apart into its operator
and operands. Cases are inline, as in `test_simplify.py`, so that a failure shows the
mathematics; those needing an assignment, a declaration or a setting run in a session.
"""

from __future__ import annotations

import pytest
from simplifying import simp

from rederive.model.session import Session


async def run(*lines):
    """Run `lines` in a fresh session: a line starting with `>` is entered, any other is
    a pair of an input and the text Simplify answers with."""
    session = Session(step_limit=-1)
    for line in lines:
        if isinstance(line, str):
            assert line.startswith("> ")
            session.author(line[2:])
            continue
        text, expected = line
        assert (await session.simplify(text)).text == expected, text


# -- recognizers ----------------------------------------------------------------------

RECOGNIZED = [
    ("[INTEGER?(-10), INTEGER?(SQRT(2))]", "[true, false]"),
    ("[INTEGER?(5), INTEGER?(0), INTEGER?(-1), INTEGER?(3)]", "[true, true, true, true]"),
    # The form counts, not what a value could be.
    ("[INTEGER?(FLOOR(x)), INTEGER?(SIGN(x))]", "[false, false]"),
    # The argument is simplified first.
    ("INTEGER?(4/2)", "true"),
    ("INTEGER?(SQRT(4))", "true"),
    ("INTEGER?(SIN(x)^2 + COS(x)^2)", "true"),
    ("INTEGER?(x)", "false"),
    ("INTEGER?([1, 2])", "false"),
    ("[RATIONAL?(0.28999), RATIONAL?(SQRT(2)/2)]", "[true, false]"),
    (
        "[RATIONAL?(2/3), RATIONAL?(-7/2), RATIONAL?(5), RATIONAL?(0)]",
        "[true, true, true, true]",
    ),
    ("RATIONAL?(pi)", "false"),
    # Approximated, a real number is a rational.
    ("APPROX(RATIONAL?(pi))", "true"),
    ("[EVEN?(4), EVEN?(0), EVEN?(-6)]", "[true, true, true]"),
    ("[ODD?(1), ODD?(-7)]", "[true, true]"),
    # Parity is a property of integers only.
    ("[EVEN?(2/3), EVEN?(3/2), ODD?(3/5)]", "[false, false, false]"),
    ("EVEN?(n)", "false"),
    (
        "[NUMBER?(3), NUMBER?(-2/3), NUMBER?(2*#i), NUMBER?(3 - 5*#i)]",
        "[true, true, true, true]",
    ),
    ("[NUMBER?(SIN(x)), NUMBER?(x^2)]", "[false, false]"),
    (
        "[NUMBER?(pi), NUMBER?(SQRT(2)), NUMBER?(inf), NUMBER?(?)]",
        "[false, false, false, false]",
    ),
    ("APPROX(NUMBER?(pi))", "true"),
    ("NUMBER(2*#i)", "true"),
    ("[VARIABLE?(x), VARIABLE?(a), VARIABLE?(Real)]", "[true, true, true]"),
    ("[VARIABLE?(pi), VARIABLE?(x + 1), VARIABLE?(3)]", "[false, false, false]"),
    # A string is a variable of that name, as a Groebner basis program naming its
    # variables "x1", "x2", ... tests.
    ('VARIABLE?("x")', "true"),
    ('[STRING?("a=b"), STRING?("Derive")]', "[true, true]"),
    ("STRING?(x)", "false"),
    ("STRING?(CODES_TO_NAME([65]))", "true"),
    # The codes spell the integer 1.
    ("STRING?(CODES_TO_NAME([49]))", "false"),
    # The operator at the top, as `STRING` names it.
    ("[SUM?(x + y), SUM?(x - y)]", "[true, true]"),
    ("[SUM?(x), SUM?(3), SUM?(-x)]", "[false, false, false]"),
    ("[PRODUCT?(x*y), PRODUCT?(x/y)]", "[true, true]"),
    # A negation is a product with -1, as FACTORS takes it apart.
    ("[PRODUCT?(2/3), PRODUCT?(-x), PRODUCT?(2*x)]", "[false, true, true]"),
    ("[PRODUCT?(-3), PRODUCT?(-x - y)]", "[false, false]"),
    ("[POWER?(x^2), POWER?(SQRT(x))]", "[true, true]"),
    ("[POWER?(x^n), POWER?(#e^x), POWER?(4)]", "[false, false, false]"),
    (
        "[VECTOR?([a, b, c]), VECTOR?([[a, b], [c, d]]), VECTOR?([])]",
        "[true, true, true]",
    ),
    (
        "[MATRIX?([[a, b], [c, d]]), MATRIX?([[a, b], [c]]), "
        "MATRIX?([a, b]), MATRIX?([])]",
        "[true, false, false, false]",
    ),
    ("MATRIX?([[1, 2]])", "true"),
    ("[SET?({a, b, c}), SET?({}), SET?(s UNION t)]", "[true, true, false]"),
    (
        "[LOGICAL?(true), LOGICAL?(false), LOGICAL?(p), LOGICAL?(x = 1)]",
        "[true, true, false, false]",
    ),
    ("LOGICAL?(1 = 1)", "true"),
    ("SELECT(VECTOR?(j), j, [1, [2], x, [3, 4]])", "[[2], [3, 4]]"),
]


@pytest.mark.parametrize(("text", "expected"), RECOGNIZED, ids=str)
def test_recognizers(text, expected):
    assert simp(text) == expected


TYPES = [
    (
        "[INTEGER_TYPE?(3), INTEGER_TYPE?(FLOOR(x)), INTEGER_TYPE?(SIGN(x))]",
        "[true, true, true]",
    ),
    ("INTEGER_TYPE?(x)", "false"),
    (
        "[REAL_TYPE?(3), REAL_TYPE?(-2/3), REAL_TYPE?(x + y), REAL_TYPE?(SIN(x))]",
        "[true, true, true, true]",
    ),
    (
        "[REAL_TYPE?(SQRT(x)), REAL_TYPE?(#e + 0.5*#i), REAL_TYPE?(inf)]",
        "[false, false, false]",
    ),
    (
        "SELECT(REAL_TYPE?(k) AND k > 0, k, [3, -3.2, #e + 0.5*#i])",
        "[3]",
    ),
    (
        "[NUMBER_TYPE?(a - 3*#i), NUMBER_TYPE?(SQRT(x)), NUMBER_TYPE?(LN(x))]",
        "[true, true, true]",
    ),
    (
        '[NUMBER_TYPE?([1]), NUMBER_TYPE?("a"), NUMBER_TYPE?(x = 1)]',
        "[false, false, false]",
    ),
    ("[VECTOR_TYPE?([a, b, c]), VECTOR_TYPE?([[a, b], [c, d]])]", "[true, true]"),
    ("VECTOR_TYPE?(x)", "false"),
    ("[SET_TYPE?({a, b, c}), SET_TYPE?(a UNION b)]", "[true, true]"),
    ("SET_TYPE?(a)", "false"),
    (
        "[LOGICAL_TYPE?(true), LOGICAL_TYPE?(p AND q), LOGICAL_TYPE?(x < 1)]",
        "[true, true, true]",
    ),
    ("LOGICAL_TYPE?(PRIME?(n))", "true"),
    ("LOGICAL_TYPE?(p)", "false"),
]


@pytest.mark.parametrize(("text", "expected"), TYPES, ids=str)
def test_type_predicates(text, expected):
    assert simp(text) == expected


async def test_predicates_read_values_and_declarations():
    await run(
        "> k := pi",
        ("NUMBER?(2 + k)", "false"),
        ("NUMBER_TYPE?(2 + k)", "true"),
        "> n :epsilon Integer",
        (
            "[INTEGER_TYPE?(n), INTEGER_TYPE?(2*n + 1), INTEGER?(n)]",
            "[true, true, false]",
        ),
        "> v :epsilon Nonscalar",
        ("[VECTOR_TYPE?(v), VECTOR?(v)]", "[true, false]"),
    )


async def test_a_definition_keeps_a_test_on_the_form_of_its_parameter():
    """What a parameter's argument is depends on the call, so a shown definition keeps
    the test; called, it is decided."""
    await run(
        ("F(x) := IF(NUMBER?(x), 1, 2)", "F(x) := IF(NUMBER?(x), 1, 2)"),
        ("[F(3), F(y)]", "[1, 2]"),
        ("G(x) := STRING(x)", "G(x) := STRING(x)"),
        ("G(a + b)", '["+", a, b]'),
    )


# -- identity -------------------------------------------------------------------------

IDENTITY = [
    ("IDENTICAL?(x + x, 2*x)", "true"),
    ("IDENTICAL?((x + 1)^2, x^2 + 2*x + 1)", "false"),
    ("IDENTICAL?(1/2, 0.5)", "true"),
    ("IDENTICAL?([a, b], [a, b])", "true"),
    ("IDENTICAL?({b, a}, {a, b})", "true"),
    ('IDENTICAL?("=", "=")', "true"),
    ('IDENTICAL?("x", x)', "false"),
    ("IDENTICAL?(a, b)", "false"),
    ("IDENTICAL?(ABS(x0), inf)", "false"),
]


@pytest.mark.parametrize(("text", "expected"), IDENTITY, ids=str)
def test_identity(text, expected):
    assert simp(text) == expected


async def test_identity_is_of_values():
    await run("> x0 := -inf", ("IDENTICAL?(ABS(x0), inf)", "true"))


# -- strings ---------------------------------------------------------------------------

STRINGS = [
    ('"a" + "a"', '2*"a"'),
    ('"-3.14" + "3.14"', '"-3.14" + "3.14"'),
    ('DET([["c1", "c2"], ["c2", "c3"]])', '"c1"*"c3" - "c2"^2'),
    ('["Period", "Balance"]', '["Period", "Balance"]'),
    ('["Derive" = "Derive"]', "[true]"),
    ("STRING(123456789)", '"123456789"'),
    ("STRING(x + y - z)", '["+", x, y, -z]'),
    ("STRING(COS(x^2))", '["COS", x^2]'),
    ("STRING(x)", '"x"'),
    # It distributes over vectors and sets.
    ("STRING([x + 1, y])", '[["+", x, 1], "y"]'),
    ("STRING({a, b})", '{"a", "b"}'),
    # A non-integer is a quotient, in Rational notation, and the decimal shown
    # otherwise.
    ("STRING(3.14159)", '["/", 314159, 100000]'),
    ("STRING(-3.14159)", '["-", 314159/100000]'),
    ("APPROX(STRING(3.14159))", '"3.14159"'),
    # A negative decimal is its text with the sign, like a negative integer.
    ("APPROX(STRING(-3.14159))", '"-3.14159"'),
    ('APPROX(APPEND("p=", STRING(-1/8)))', '"p=-0.125"'),
    (
        "APPROX(STRING(COS(pi/257)), 100)",
        '"0.99992528666973255521280535914555624069162982964292120118813770997100714752'
        '77558301156004129496103686"',
    ),
    ('APPEND("d", STRING(5))', '"d5"'),
    ('APPEND("d", STRING(12))', '"d12"'),
    # A negative integer is its numeral with the sign, as NAME_TO_CODES reads it.
    ("STRING(-14)", '"-14"'),
    ('APPEND("x = ", STRING(-7))', '"x = -7"'),
    ("REVERSE(STRING(1000))", '"0001"'),
    ("STRING(-z)", '["-", z]'),
    ("STRING(-2*x^3)", '["-", 2*x^3]'),
    ("STRING(2*x^3)", '["·", 2, x^3]'),
    ("STRING(x^3)", '["^", x, 3]'),
    ("STRING(#e^(2*x))", '["^", #e, 2*x]'),
    ("STRING(SQRT(x))", '["^", x, 1/2]'),
    ("STRING(x/y)", '["/", x, y]'),
    ("STRING(x <= 1)", '["≤", x, 1]'),
    ("STRING(x = 1)", '["=", x, 1]'),
    ("STRING(a AND b AND c)", '["∧", a, b AND c]'),
    ("STRING(SIGN(x - 1))", '["SIGN", x - 1]'),
    ("STRING(INT(1/SQRT(1 - x^4), x))", '["∫", 1/SQRT(1 - x^4), x]'),
    ("STRING(pi)", '"pi"'),
    ("STRING(?)", '"?"'),
    ('STRING("abc")', '"abc"'),
    ("STRING(x SUB 1)", '["↓", x, 1]'),
    ("FIRST(STRING(x^2))", '"^"'),
    ("NAME_TO_CODES(FIRST(STRING(COS(2*x))))", "[67, 79, 83]"),
    ('IDENTICAL?(FIRST(STRING(x = 1)), "=")', "true"),
]


@pytest.mark.parametrize(("text", "expected"), STRINGS, ids=str)
def test_strings(text, expected):
    assert simp(text) == expected


CODES = [
    ('NAME_TO_CODES("Exact")', "[69, 120, 97, 99, 116]"),
    ("CODES_TO_NAME([69, 120, 97, 99, 116])", '"Exact"'),
    # Codes that spell an integer give the integer.
    ("CODES_TO_NAME([45, 52, 49])", "-41"),
    ("NAME_TO_CODES(1104)", "[49, 49, 48, 52]"),
    ("NAME_TO_CODES(47110815)", "[52, 55, 49, 49, 48, 56, 49, 53]"),
    ("VECTOR(CODES_TO_NAME(v), v, NAME_TO_CODES(1104))", "[1, 1, 0, 4]"),
    (
        "VECTOR(CODES_TO_NAME(k), k, [52, 55, 49, 49, 48, 56, 49, 53])",
        "[4, 7, 1, 1, 0, 8, 1, 5]",
    ),
    # A number is taken as the current notation shows it.
    (
        "NAME_TO_CODES(-3.14159)",
        "[45, 51, 49, 52, 49, 53, 57, 47, 49, 48, 48, 48, 48, 48]",
    ),
    ("APPROX(CODES_TO_NAME(NAME_TO_CODES(-3.14159)))", '"-3.14159"'),
    ('NAME_TO_CODES("-3140")', "[45, 51, 49, 52, 48]"),
    ('NAME_TO_CODES("3.14")', "[51, 46, 49, 52]"),
    ("CODES_TO_NAME(48)", "0"),
    ("CODES_TO_NAME([51, 53])", "35"),
    ("CODES_TO_NAME([48, 51, 53])", "35"),
    ('CODES_TO_NAME(NAME_TO_CODES("0123"))', "123"),
    ('CODES_TO_NAME(NAME_TO_CODES("-1.23"))', '"-1.23"'),
    ("CODES_TO_NAME(65)", '"A"'),
    (
        "CODES_TO_NAME([72, 97, 112, 112, 121, 32, 78, 101, 119, 32, 89, 101, 97, 114, "
        "32, 50, 48, 48, 48])",
        '"Happy New Year 2000"',
    ),
    ("CODES_TO_NAME(VECTOR(v + 64, v, [9, 0, 12]))", '"I@L"'),
    # A symbol of Derive's font has Derive's code; a letter beyond ASCII, or any
    # character beyond 255, is DEL and four hexadecimal digits.
    ('NAME_TO_CODES("≠≤≥±")', "[148, 147, 146, 177]"),
    ("CODES_TO_NAME([148, 147, 146])", '"≠≤≥"'),
    ("NAME_TO_CODES(FIRST(STRING(a /= b)))", "[148]"),
    ("CODES_TO_NAME([127, 50, 53, 97, 48, 127, 50, 48, 50, 55])", '"■‧"'),
    ("DIM(CODES_TO_NAME([127, 50, 53, 97, 48]))", "1"),
    ('NAME_TO_CODES("■")', "[127, 50, 53, 97, 48]"),
    ('NAME_TO_CODES("Ä")', "[127, 48, 48, 99, 52]"),
    ("NAME_TO_CODES(CODES_TO_NAME([129, 131]))", "[129, 131]"),
    # Derive's font has bracket pieces at 128, 130 and 132.
    ("CODES_TO_NAME([82, 128, 130, 132])", '"R⎛⎞⎡"'),
    ('NAME_TO_CODES("⎛⎞⎡")', "[128, 130, 132]"),
    ('NAME_TO_CODES("")', "[]"),
    ("CODES_TO_NAME([])", '""'),
    ("CODES_TO_NAME([45])", '"-"'),
    ("CODES_TO_NAME([45, 48])", "0"),
    # No string holds a double quote, a line break or a NUL.
    ("CODES_TO_NAME([65, 34])", "CODES_TO_NAME([65, 34])"),
    ("CODES_TO_NAME([65, 10])", "CODES_TO_NAME([65, 10])"),
    ("CODES_TO_NAME([300])", '"Ĭ"'),
    ("CODES_TO_NAME([55296])", "CODES_TO_NAME([55296])"),
    ("CODES_TO_NAME([127, 50, 53])", "CODES_TO_NAME([127, 50, 53])"),
    ("CODES_TO_NAME([x])", "CODES_TO_NAME([x])"),
    ("NAME_TO_CODES(x)", "[120]"),
    ("NAME_TO_CODES(x + 1)", "NAME_TO_CODES(x + 1)"),
    ('NAME_TO_CODES(["a", "bc"])', "[[97], [98, 99]]"),
    ("SUM(NAME_TO_CODES(9^(9^3))) - 48*DIM(NAME_TO_CODES(9^(9^3)))", "3249"),
]


@pytest.mark.parametrize(("text", "expected"), CODES, ids=str)
def test_character_codes(text, expected):
    assert simp(text) == expected


async def test_numbers_as_text_follow_the_output_base_and_the_notation():
    await run(
        "> OutputBase := Binary",
        ("STRING(65)", '"1000001"'),
        "> OutputBase := Hexadecimal",
        ("STRING(255)", '"0FF"'),
        "> OutputBase := Decimal",
        "> Notation := Decimal",
        ("NAME_TO_CODES(3.45)", "[51, 46, 52, 53]"),
    )


# -- substitution -----------------------------------------------------------------------

SUBSTITUTED = [
    ("SUBST(a*x + b, x, z^3)", "a*z^3 + b"),
    # All at once: a replacement is not substituted into again.
    ("SUBST(x*y, [x, y], [x + y, 2 + x])", "(x + y)*(x + 2)"),
    ("SUBST(y - x^2, [x, y], [x + y, x - y])", "-x^2 + x*(1 - 2*y) - y^2 - y"),
    ("SUBST(y - x^2, x, x + y)", "-x^2 - 2*x*y - y^2 + y"),
    # One pair after another.
    ("SUBST(x*y, y, x, x^2, z)", "z"),
    # Only a subexpression matches: x^2 is no part of x^4.
    ("SUBST(a*x^4 + b*x^3 + c*x^2 + d*x + e, x^2, y)", "a*x^4 + b*x^3 + d*x + c*y + e"),
    ("SUBST(x*y*z, x*y, w)", "x*y*z"),
    # The result is simplified: an equation may be decided.
    ("SUBST(y = x + c1^3 - 3*c1, c1, 1)", "y = x - 2"),
    ("SUBST(a*x + b*y + (a + b) = 0, [x, y], [-1, -1])", "true"),
    ("SUBST(a*x + (a + d)*y + a + 2*d = 0, [x, y], [1, -2])", "true"),
    (
        "VECTOR(SUBST(8*x^2 - 24*x*y + 18*y^2 - 6*x - 9*y - 209, [x, y], k), k, "
        "[[4130, 2701], [5014, 3285]])",
        "[0, 0]",
    ),
    (
        'APPROX(SUBST("dz" = x*"dx"/SQRT(x^2 + y^2) + y*"dy"/SQRT(x^2 + y^2), '
        '[x, y, "dx", "dy"], [3, 4, 0.04, -0.02]))',
        '"dz" = 0.008',
    ),
    ("SUBST([x, x + 9]/2, x, 2)", "[1, 11/2]"),
    # A bound variable is not replaced.
    ("SUBST(SUM(k*x, k, 1, n), k, 5)", "n*x*(n + 1)/2"),
    ("SUBST(INT(x^2, x, 0, a), x, 3)", "a^3/3"),
    # Substitution takes no limit.
    ("SUBST(x/SIN(x), x, 0)", "?"),
    ("SUBST(x^2, x)", "SUBST(x^2, x)"),
    ("SUBST(x + y, [x, y], [1])", "SUBST(x + y, [x, y], [1])"),
    # A number for a vector of variables replaces each of them.
    ("SUBST(GRAD(x^4*y, [x, y])/(x^4*y), [x, y], 1)", "[4, 1]"),
    ("SUBST(x^2*y + 3*x, [x, y], 2)", "14"),
    ("SUBS(x^2, [x], [3])", "9"),
]


@pytest.mark.parametrize(("text", "expected"), SUBSTITUTED, ids=str)
def test_substitution(text, expected):
    assert simp(text) == expected


async def test_substitution_values_come_from_variables_and_bound_names_stay():
    await run(
        "> ell := a^2*x^2 + b^2*y^2 = 4",
        ("SUBST(ell, [a, b], [1, 2])", "x^2 + 4*y^2 = 4"),
        "> vs := [1, 1]",
        ("SUBST(x^2 + y, [x, y], vs)", "2"),
        # A parameter may be a vector of values, so it doesn't replace each variable.
        "> G(k) := SUBST(x^2 + y, [x, y], k)",
        ("G([1, 2])", "3"),
        "> F(x) :=",
        ("SUBST(SUM(F(k), k, 1, n), k, 3)", "SUM(F(k), k, 1, n)"),
        ("SUBST(SUM(F(k)*x, k, 1, n), x, 3)", "SUM(3*F(k), k, 1, n)"),
        ("SUBST(INT(F(x), x, 0, a), a, 3)", "INT(F(x), x, 0, 3)"),
        ("SUBST(DIF(F(x), x), x, 3)", "F'(3)"),
    )
