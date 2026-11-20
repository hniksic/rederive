"""State variables (the settings as expressions) and the modes they select.

A setting is assigned by `Name := value` wherever an assignment may stand, and read back
as an operand. Its factory defaults are Derive 6's. The modes change only what an
identity valid for every value of the variables allows: `Angle := Degree` changes how an
angle is shown and nothing else; `Exponential`, `Logarithm` and `Trigpower` choose the
direction of their rules.

Each case is a list of lines run in a fresh session, in order: a line starting with `>`
is entered, a line starting with `~` is approximated, any other is a pair of an input and
the text Simplify answers with, or `(error: message)` for a line that is refused.
"""

import pytest

from rederive.model.session import Session
from rederive.syntax import DeriveSyntaxError


async def run(*lines):
    """Run `lines` in a fresh session (see the module docstring)."""
    session = Session(step_limit=-1)
    for line in lines:
        if isinstance(line, str):
            assert line.startswith("> ")
            session.author(line[2:])
            continue
        text, expected = line
        if expected.startswith("(error: "):
            count = len(session.entries)
            with pytest.raises(DeriveSyntaxError, match=expected[len("(error: ") : -1]):
                session.author(text)
            assert len(session.entries) == count
        elif text.startswith("~ "):
            assert (await session.approx(text[2:])).text == expected, text
        else:
            assert (await session.simplify(text)).text == expected, text
    return session


# -- assigning and reading ----------------------------------------------------


async def test_a_setting_reads_as_its_value():
    await run(
        "> Precision := Approximate",
        ("[Precision, PrecisionDigits]", "[Approximate, 10]"),
        "> Precision := Exact",
        ("[Precision = Exact, Precision = Mixed]", "[true, false]"),
    )


async def test_a_setting_assignment_simplifies_to_itself():
    await run(("TimesOperator := Asterisk", "TimesOperator := Asterisk"))


async def test_a_setting_changes_for_what_is_evaluated_after_it():
    await run(
        ("[PrecisionDigits := 20, PrecisionDigits + 1]", "[PrecisionDigits := 20, 21]")
    )


async def test_a_radix_with_a_name_reads_as_its_name():
    await run(
        "> OutputBase := 2",
        ("OutputBase", "Binary"),
        "> OutputBase := 10",
        ("OutputBase", "Decimal"),
    )


async def test_a_setting_in_another_case_takes_a_value_it_accepts():
    await run(
        "> notation := decimal",
        ("Notation", "Decimal"),
        "> angle := 30",
        ("[angle, Angle]", "[30, Radian]"),
    )


async def test_an_empty_assignment_restores_the_factory_default():
    await run(
        "> PrecisionDigits := 20",
        "> PrecisionDigits :=",
        ("PrecisionDigits", "10"),
        "> VariableOrder := [y, x]",
        "> VariableOrder :=",
        ("VariableOrder", "[x, y, z]"),
    )


async def test_a_setting_that_is_a_parameter_is_restored_on_return():
    await run(
        "> P1(x, PrecisionDigits := 30) := x + PrecisionDigits",
        ("[P1(1), PrecisionDigits]", "[31, 10]"),
    )


async def test_a_setting_assigned_in_a_program_keeps_its_value():
    await run(
        "> P2(x) := PROG(PrecisionDigits := 30, x)",
        ("[P2(1), PrecisionDigits]", "[1, 30]"),
    )


async def test_the_variable_order_is_saved_and_restored_by_a_program():
    await run(
        "> P3(u, store) := PROG(store := VariableOrder, VariableOrder := [u], "
        "VariableOrder := store, store)",
        ("P3(t)", "[x, y, z]"),
    )


async def test_a_value_a_setting_does_not_take_is_refused():
    session = await run(("Angle := 30", "(error: Invalid value for Angle)"))
    assert session.settings["Angle"] == "Radian"
    await run(
        ("Precision := Rational", "(error: Invalid value for Precision)"),
        ("VariableOrder := [x, 2]", "(error: Invalid value for VariableOrder)"),
    )


async def test_a_value_held_by_a_variable_is_not_refused_on_input():
    session = await run("> trig_ := Expand", "> Trigonometry := trig_")
    assert session.settings["Trigonometry"] == "Auto"
    await session.simplify("Trigonometry := trig_")
    assert session.settings["Trigonometry"] == "Expand"


# -- the table and its factory defaults ---------------------------------------


async def test_the_factory_defaults_are_derive_6s():
    await run(
        ("[PrecisionDigits, NotationDigits]", "[10, 10]"),
        ("[Precision, Notation, Branch, Angle]", "[Exact, Rational, Principal, Radian]"),
        ("[Exponential, Logarithm, Trigonometry, Trigpower]", "[Auto, Auto, Auto, Auto]"),
        (
            "[CaseMode, InputMode, InputBase, OutputBase]",
            "[Insensitive, Character, Decimal, Decimal]",
        ),
        ("[DisplayFormat, TimesOperator, DisplaySteps]", "[Normal, Dot, false]"),
        ("VariableOrder", "[x, y, z]"),
        ("SimplifyRelations", "false"),
        ("~ SQRT(24)", "4.898979485"),
        ("~ pi", "3.141592653"),
    )


async def test_the_inert_settings_read_back_and_change_nothing():
    await run(
        "> DisplaySteps := true",
        ("[DisplaySteps, (x + 1)^2 - x^2 - 2*x]", "[true, 1]"),
        "> SimplifyRelations := true",
        ("[SimplifyRelations, 2*x + 3 = 5]", "[true, 2*x + 3 = 5]"),
    )


async def test_the_plot_cross_names_are_variables():
    await run(
        ("hCross + 1", "hCross + 1"),
        "> xCross := 3",
        ("xCross + vCross", "vCross + 3"),
    )


# -- VariableOrder --------------------------------------------------------------


async def test_the_variable_order_orders_terms_and_factors():
    await run(
        ("z^2*x^3*y^5", "x^3*y^5*z^2"),
        ("z^3*b^2*a*2*x^5*c", "2*a*b^2*c*x^5*z^3"),
        ("5*y^3 + 2*x^2 - 3*a*x", "2*x^2 - 3*a*x + 5*y^3"),
        ("-x + y", "y - x"),
        ("(x + 1)^2 - x^2", "2*x + 1"),
        ("(alpha - 1)^3", "(alpha - 1)^3"),
    )


async def test_the_most_main_variable_is_expanded_in():
    await run(
        (
            "(x + 1)^9 + y",
            "x^9 + 9*x^8 + 36*x^7 + 84*x^6 + 126*x^5 + 126*x^4 + 84*x^3 + 36*x^2 + 9*x"
            " + y + 1",
        ),
        "> VariableOrder := [y, x, z]",
        ("(x + 1)^9 + y", "y + (x + 1)^9"),
        ("VariableOrder", "[y, x, z]"),
    )


async def test_a_repeated_name_counts_once_and_orders_the_variables():
    await run(
        "> VariableOrder := [b, a, b]",
        ("VariableOrder", "[b, a]"),
        ("VARIABLES(a + b + c)", "[b, a, c]"),
        '> VariableOrder := [x, y, z, "dx", "dy"]',
        ("VariableOrder", "[x, y, z, dx, dy]"),
        "> VariableOrder := []",
        ("VariableOrder", "[]"),
    )


# -- Angle ---------------------------------------------------------------------


async def test_trigonometric_functions_take_radians_in_either_mode():
    await run(
        ("SIN(45*deg)", "SQRT(2)/2"),
        ("ASIN(1/2)", "pi/6"),
        ("30*deg", "pi/6"),
        "> Angle := Degree",
        ("SIN(45*deg)", "SQRT(2)/2"),
        ("SIN(pi/4)", "SQRT(2)/2"),
        ("TAN(45*deg)", "1"),
        ("TAN(45)", "TAN(45)"),
        ("~ TAN(45)", "1.619775190"),
    )


async def test_degree_mode_shows_an_angle_in_degrees():
    await run(
        "> Angle := Degree",
        ("ASIN(1/2)", "30 DEG"),
        ("ATAN(1)", "45 DEG"),
        ("~ ATAN(1)", "0.7853981633"),
        ("ATAN(1)/(1*deg)", "45"),
        ("ATAN(2)", "ATAN(2)"),
        ("PHASE(1 + #i)", "45 DEG"),
        ("pi/6", "pi/6"),
        ("30*deg", "30 DEG"),
        ("~ deg", "0.01745329251"),
        ("SOLVE(TAN(a) = 1, a, Real)", "a = 45 DEG"),
        ("ARCSIN(1/2)", "30"),
    )


# -- precision -----------------------------------------------------------------


async def test_exact_mode_denests_radicals():
    await run(
        ("SQRT(4 - 2*SQRT(3))", "SQRT(3) - 1"),
        ("SQRT(5 + 2*SQRT(6))", "SQRT(3) + SQRT(2)"),
        ("(243*SQRT(5) - 294*SQRT(3))^(1/3)", "3*SQRT(5) - 2*SQRT(3)"),
        (
            "SQRT(12 + 2*SQRT(6) + 2*SQRT(14) + SQRT(6)*SQRT(14))",
            "SQRT(7) + SQRT(3) + SQRT(2)",
        ),
        ("(SQRT(5) + 2)^(1/3) + (8*SQRT(5) + 16)^(1/3) - (27*SQRT(5) + 54)^(1/3)", "0"),
    )


async def test_precision_brings_its_notation_and_digits():
    await run(
        "> Precision := Approximate",
        ("[Notation, NotationDigits]", "[Scientific, 10]"),
        ("SQRT(4 + 2*SQRT(3))", "2.732050807"),
        ("(5/8)^2", "0.390625"),
        # A typed numeral is the fraction it spells, whatever the precision.
        ("0.123456789012345*10^15", "1.234567890*10^14"),
        "> Precision := Mixed",
        ("Notation", "Mixed"),
        ("SQRT(3422357/2313 - 1140443/771)", "2/3"),
        "> Precision := Exact",
        "> PrecisionDigits := 12",
        ("NotationDigits", "12"),
        "> NotationDigits := 5",
        ("PrecisionDigits", "12"),
    )


# -- the transformation modes --------------------------------------------------


async def test_exponential_modes():
    await run(
        ("x^a*#e^(b + c)", "x^a*#e^(b + c)"),
        ("#e^x*#e^(2*x)", "#e^(3*x)"),
        "> Exponential := Expand",
        ("x^a*#e^(b + c)", "x^a*#e^b*#e^c"),
        "> Exponential := Collect",
        ("#e^b*#e^c*x^a", "x^a*#e^(b + c)"),
        ("(#e^x)^3", "#e^(3*x)"),
        # Drawn into the exponent only where `x^a = #e^(a*LN(x))` holds.
        "> x :epsilon Real (0, inf)",
        ("x^a*#e^(b + c)", "#e^(a*LN(x) + b + c)"),
    )


async def test_logarithm_auto_factors_an_argument_with_a_repeated_factor():
    await run(
        ("LN(x*y^4 + 2*x*y^2 + x)/2", "LN(x*(y^2 + 1)^2)/2"),
        ("LN(2*x)", "LN(2*x)"),
        ("LN(x^2 - x) - LN(x)", "LN(x^2 - x) - LN(x)"),
        ("LN(x^2) - 2*LN(ABS(x))", "0"),
        ("LOG(8, 2)", "3"),
        ("LN(-3)", "LN(3) + #i*pi"),
        # A real power of a positive number leaves its logarithm.
        ("LN(2^(2 - x) = 6)", "LN(2)*(2 - x) = LN(6)"),
        "> x :epsilon Real [0, inf)",
        ("LN(x^2 - x) - LN(x)", "LN(x - 1)"),
    )


async def test_logarithm_expand():
    await run(
        "> Logarithm := Expand",
        ("LN(x*y^4 + 2*x*y^2 + x)/2", "LN(x)/2 + LN(y^2 + 1)"),
        ("LN(x*y)", "LN(x*y)"),
        # The power of a real `x` leaves as the power of its absolute value.
        ("LN(x^2)", "2*LN(ABS(x))"),
        ("LN(SQRT(x))", "LN(x)/2"),
    )


async def test_logarithm_collect():
    await run(
        "> Logarithm := Collect",
        ("LN(x*y^4 + 2*x*y^2 + x)/2", "LN(SQRT(x)*(y^2 + 1))"),
        ("LN(x)/2 + LN(y^2 + 1)", "LN(SQRT(x)*(y^2 + 1))"),
        # `2*LN(x)` is `LN(x^2)` only for a nonnegative `x`.
        ("2*LN(x)", "2*LN(x)"),
        ("LN(x) + LN(y)", "LN(x) + LN(y)"),
    )


async def test_trigonometry_modes():
    await run(
        ("SIN(pi/4)", "SQRT(2)/2"),
        "> Trigonometry := Expand",
        ("SIN(2*x + y)", "2*COS(x)^2*SIN(y) + 2*SIN(x)*COS(x)*COS(y) - SIN(y)"),
        ("SIN(x) - SIN(x + pi/3) + COS(x + pi/6)", "0"),
        "> Trigonometry := Collect",
        ("2*COS(x)^2*SIN(y) + 2*SIN(x)*COS(x)*COS(y) - SIN(y)", "SIN(2*x + y)"),
        ("SIN(3*x) + COS(3*x)", "SQRT(2)*SIN(3*x + pi/4)"),
        ("8*SIN(pi/18)*COS(pi/9)*COS(2*pi/9)", "1"),
    )


async def test_trigpower_writes_a_polynomial_in_one_function():
    await run(
        "> Trigpower := Cosines",
        ("SIN(x)^4", "COS(x)^4 - 2*COS(x)^2 + 1"),
        "> Trigpower := Sines",
        ("COS(x)^4 - 2*COS(x)^2 + 1", "SIN(x)^4"),
        ("COS(x)^4", "SIN(x)^4 - 2*SIN(x)^2 + 1"),
    )


async def test_branch_modes():
    await run(
        ("(-8)^(1/3)", "1 + SQRT(3)*#i"),
        ("(1 + #i)^3", "-2 + 2*#i"),
        "> Branch := Real",
        ("(-8)^(1/3)", "-2"),
        ("a - (-1)^(1/9)", "a + 1"),
    )


# -- APPROX ---------------------------------------------------------------------


async def test_approx_forms():
    await run(
        ("APPROX(pi, 50)", "3.1415926535897932384626433832795028841971693993751"),
        ("APPROX(pi, 20)", "3.1415926535897932384"),
        ("APPROX(SIN(pi/4), 35)", "0.70710678118654752440084436210484903"),
        (
            "APPROX(5/173, 50)",
            "0.028901734104046242774566473988439306358381502890173",
        ),
        ("APPROX(a*(SQRT(21)/24 - 1/8))", "0.06594065395*a"),
        ("APPROX(x, 0)", "APPROX(x, 0)"),
        ("[APPROX(pi, 6)]", "[355/113]"),
        "> FOO(x) := APPROX(x, 6)",
        ("FOO(pi)", "355/113"),
    )


async def test_an_approximated_vector_is_approximated_elementwise():
    await run(
        (
            "APPROX(VECTOR(SIN(z), z, 0, pi/4, 0.2))",
            "[0, 0.1986693307, 0.3894183423, 0.5646424733]",
        )
    )


async def test_an_approximated_definition_needs_no_approx_call():
    """An `APPROX` in a definition stays a call, evaluated where the function is called,
    unless the definition is approximated itself: then it has nothing left to do."""
    await run(
        ("F(a) := APPROX(a/3)", "F(a) := APPROX(a/3)"),
        ("~ F(a) :== APPROX(a/3)", "F(a) := 0.3333333333*a"),
        ("~ F(a) :== APPROX(SQRT(a) + 1/3)", "F(a) := SQRT(a) + 0.3333333333"),
    )


async def test_an_approximated_assignment_shows_digits():
    await run("> a := 0.9333333333", ("b := APPROX(a, 6)", "b := 0.933333"))


async def test_approx_of_relations_and_of_exact_values():
    await run(
        ("APPROX(pi = 3)", "false"),
        ("APPROX(SQRT(2)^2 = 2)", "true"),
        (
            "APPROX(x = 1/2 - SQRT(5)/2 OR x = SQRT(5)/2 + 1/2 OR x = -1/2 OR x = -2)",
            "x = -2 OR x = -0.6180339887 OR x = -0.5 OR x = 1.618033988",
        ),
        ("APPROX(LIM((TAN(x) - 1)/(SIN(x) - 1/SQRT(2)), x, pi/4))", "2.828427124"),
        ("APPROX(SIN(10^11), 10)", "0.9286936604"),
        ("APPROX(SIN(10^11 + 0.5))", "0.9927992642"),
        ("APPROX(1234^56)", "1.299119025*10^173"),
    )


# -- number display ----------------------------------------------------------------


async def test_digits_are_truncated():
    await run(
        ("~ 5/7", "0.7142857142"),
        ("~ [3/13, 5/17]", "[0.2307692307, 0.2941176470]"),
        ("~ SQRT(24)", "4.898979485"),
        ("~ -1286732833/567", "-2.269370075*10^6"),
        (
            "~ [448000/3, 449000/3, 1000, -21/500, 1/1000000, 12599]",
            "[1.493333333*10^5, 1.496666666*10^5, 1000, -0.042, 10^-6, 1.2599*10^4]",
        ),
        (
            "~ [0, 3023599993/10^13, -7558999984/10^14]",
            "[0, 0.0003023599993, -7.558999984*10^-5]",
        ),
        ("~ (#i + 7)/(3*#i + 4)", "1.24 - 0.68*#i"),
        ("~ [1/2, 1/16, 1/128, 1/1024]", "[0.5, 0.0625, 0.0078125, 0.0009765625]"),
    )


async def test_the_whole_part_and_one_more_digit_are_shown():
    await run(
        ("APPROX((365/(5 + SQRT(6)))^(1/3), 3)", "3.65"),
        ("APPROX(100*(365/(5 + SQRT(6)))^(1/3), 3)", "365.9"),
    )


async def test_notation_digits_cut_every_style():
    await run(
        "> NotationDigits := 4",
        "> Notation := Decimal",
        ("1234999/1000000", "1.234"),
        "> Notation := Scientific",
        ("1234999/1000000", "1.234"),
        "> NotationDigits := 6",
        "> Notation := Decimal",
        (
            "[467835792/10^9, 3196875/10^7, 4038507/10^8]",
            "[0.467835, 0.319687, 0.0403850]",
        ),
    )


async def test_the_output_base_writes_decimal_numerals_too():
    session = await run("> Notation := Decimal", "> OutputBase := Hexadecimal")
    answer = await session.simplify("[12, 29, 3645, 24.5]")
    assert answer.layout.lines == ("[0C, 1D, 0E3D, 18.8]",)
