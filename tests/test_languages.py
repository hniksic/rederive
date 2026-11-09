"""Expression trees as C, Python, Rust and Julia source.

The original offered Basic, C, Fortran and Pascal, so most of what is tested here has no
original to match. Three rules do come from the original, and are marked below: a name
the target lacks is passed through in the target's casing, numerals are written as
typed, and `#e^u` becomes the exponential.

Nothing is checked against a compiler. Like the original's, the writer is a
transliterator, and it doesn't attempt translations it can't make faithfully.
"""

from __future__ import annotations

import pytest

from rederive.syntax import LANGUAGES, ParseState, parse_expression, write_source

#: Each target language keyed by its Transfer Save menu word.
BY_WORD = {language.word: language for language in LANGUAGES}


def written(text: str, word: str) -> str:
    return write_source(parse_expression(text, ParseState()).node, BY_WORD[word])


def spellings(text: str) -> dict[str, str]:
    return {language.word: written(text, language.word) for language in LANGUAGES}


# -- the elementary functions ------------------------------------------------


@pytest.mark.parametrize(
    "authored, expected",
    [
        ("SIN(x)", {"C": "sin(x)", "Python": "math.sin(x)", "Rust": "x.sin()",
                    "Julia": "sin(x)"}),
        ("SQRT(x + 1)", {"C": "sqrt(x + 1)", "Python": "math.sqrt(x + 1)",
                         "Rust": "(x + 1).sqrt()", "Julia": "sqrt(x + 1)"}),
        # Derive spells the natural logarithm LN; only Rust agrees.
        ("LN(x)", {"C": "log(x)", "Python": "math.log(x)", "Rust": "x.ln()",
                   "Julia": "log(x)"}),
        ("ABS(x)", {"C": "fabs(x)", "Python": "abs(x)", "Rust": "x.abs()",
                    "Julia": "abs(x)"}),
        ("MAX(x, y)", {"C": "fmax(x, y)", "Python": "max(x, y)",
                       "Rust": "x.max(y)", "Julia": "max(x, y)"}),
    ],
)
def test_a_function_is_spelled_the_way_its_target_spells_it(authored, expected):
    assert spellings(authored) == expected


def test_the_two_argument_logarithm_takes_its_base_where_the_target_wants_it():
    """Julia's `log` takes the base first, Derive's last, and C has no two-argument
    logarithm, so it gets the change-of-base formula."""
    assert spellings("LOG(x, 10)") == {
        "C": "(log(x) / log(10))",
        "Python": "math.log(x, 10)",
        "Rust": "x.log(10)",
        "Julia": "log(10, x)",
    }


def test_the_two_argument_arctangent_is_the_one_with_two_arguments():
    assert spellings("ATAN(y, x)") == {
        "C": "atan2(y, x)",
        "Python": "math.atan2(y, x)",
        "Rust": "y.atan2(x)",
        "Julia": "atan(y, x)",
    }


def test_the_factorial_is_a_call_where_it_is_anything():
    """C has only the gamma function, and Rust has neither, so Rust gets a plain call,
    as the original wrote for what it couldn't spell."""
    assert spellings("n!") == {
        "C": "tgamma(n + 1)",
        "Python": "math.factorial(n)",
        "Rust": "factorial(n)",
        "Julia": "factorial(n)",
    }


def test_a_constant_is_the_target_constant_and_not_a_free_variable():
    """Derive wrote `pi` into C, where it is undeclared; it should be M_PI."""
    assert spellings("pi + inf") == {
        "C": "M_PI + INFINITY",
        "Python": "math.pi + math.inf",
        "Rust": "std::f64::consts::PI + f64::INFINITY",
        "Julia": "pi + Inf",
    }


def test_the_imaginary_unit_reaches_the_two_targets_that_have_one():
    """From the original: what a target can't spell passes through unchanged."""
    assert spellings("#i") == {
        "C": "#i",
        "Python": "1j",
        "Rust": "#i",
        "Julia": "im",
    }


# -- powers ------------------------------------------------------------------


def test_a_power_is_written_the_way_the_target_writes_one():
    assert spellings("x^2 + 1") == {
        "C": "pow(x, 2) + 1",
        "Python": "x ** 2 + 1",
        "Rust": "x.powi(2) + 1",
        "Julia": "x ^ 2 + 1",
    }


def test_rust_picks_its_power_method_by_the_exponent():
    """`powi` takes a whole number and `powf` a real one."""
    assert written("x^2", "Rust") == "x.powi(2)"
    assert written("x^-3", "Rust") == "x.powi(-3)"
    assert written("x^(2/3)", "Rust") == "x.powf(2 / 3)"


def test_a_numeral_receiving_a_rust_method_is_written_with_a_point():
    """`2.sqrt()` doesn't tokenize, since the lexer reads `2.` as the numeral."""
    assert written("SQRT(2)", "Rust") == "2.0.sqrt()"
    assert written("10^100", "Rust") == "10.0.powi(100)"


def test_the_base_e_power_is_the_exponential():
    """The original wrote `EXP(x)` for it in all four of its languages."""
    assert spellings("#e^x") == {
        "C": "exp(x)",
        "Python": "math.exp(x)",
        "Rust": "x.exp()",
        "Julia": "exp(x)",
    }


def test_a_power_of_a_power_keeps_the_shape_it_was_written_in():
    assert written("2^3^4", "Julia") == "2 ^ 3 ^ 4"
    assert written("(2^3)^4", "Julia") == "(2 ^ 3) ^ 4"
    assert written("2^3^4", "C") == "pow(2, pow(3, 4))"


# -- operators ---------------------------------------------------------------


def test_derives_equality_is_the_targets_equality():
    """Derive wrote `x=y` into C, which is an assignment. Derive's `=` is a relation,
    so it should be `==`."""
    assert written("x = y", "C") == "x == y"
    assert written("x /= y", "C") == "x != y"


def test_the_logical_operators_are_words_only_where_the_target_wants_words():
    assert spellings("x AND NOT y") == {
        "C": "x && !y",
        "Python": "x and not y",
        "Rust": "x && !y",
        "Julia": "x && !y",
    }


def test_a_conditional_becomes_the_target_conditional():
    assert spellings("IF(x > 0, 1, -1)") == {
        "C": "(x > 0 ? 1 : -1)",
        "Python": "(1 if x > 0 else -1)",
        "Rust": "(if x > 0 { 1 } else { -1 })",
        "Julia": "ifelse(x > 0, 1, -1)",
    }


def test_a_run_of_factors_keeps_the_fences_that_say_where_a_quotient_ends():
    """All four languages associate `*` and `/` left to right, so a division can go
    unparenthesized only at the start of a run, as in the Derive writer."""
    assert written("a*b/c*d", "C") == "a * b / c * d"
    assert written("a*(b/c)", "C") == "a * (b / c)"


def test_a_vector_is_the_nearest_thing_the_target_has():
    assert spellings("[[1, 2], [3, 4]]") == {
        "C": "{{1, 2}, {3, 4}}",
        "Python": "[[1, 2], [3, 4]]",
        "Rust": "[[1, 2], [3, 4]]",
        "Julia": "[[1, 2], [3, 4]]",
    }


# -- what does not translate -------------------------------------------------


def test_a_numeral_is_written_as_it_was_written():
    """From the original, which wrote Derive's exact `1/3` into C as `1/3`, an integer
    division. Writing `1.0/3.0` would be guessing what the reader meant."""
    for language in LANGUAGES:
        assert written("1/3 + 2.5", language.word) == "1 / 3 + 2.5"


def test_a_name_no_target_has_is_passed_through_spelled_their_way():
    """The original wrote `INT(SIN(x), x)` into C as `int(sin(x),x)`, which no compiler
    accepts but every reader understands."""
    assert written("INT(SIN(x), x)", "C") == "int(sin(x), x)"
    assert written("SOLVE(x^2 = 4, x)", "Julia") == "solve(x ^ 2 == 4, x)"


def test_indexing_passes_through_rather_than_guessing_where_it_starts():
    """Julia indexes from one like Derive, the others from zero, so `[]` would be off
    by one in three targets."""
    assert written("x SUB 2", "Python") == "sub(x, 2)"


def test_a_shape_with_no_form_at_all_stays_in_derive_notation():
    assert written("x :epsilon Real (0, inf)", "C") == "x:epsilonReal (0, inf)"
    # No target has an `IMP` operator, so it is written as in Derive.
    assert written("a IMP b", "C") == "a IMP b"


def test_what_passes_through_is_fenced_where_it_has_to_be():
    """The precedence of an untranslatable expression is unknown, so anything but a
    leaf is assumed to bind as loosely as possible."""
    assert written("(a IMP b) + 1", "C") == "(a IMP b) + 1"
    assert written("#3 + 1", "C") == "#3 + 1"


# -- definitions -------------------------------------------------------------


def test_a_function_definition_takes_the_form_its_target_has():
    """Julia writes exactly this, Python needs a lambda, and the two without a one-line
    form get the assignment the original wrote."""
    assert spellings("G(u) := u^2 - 1") == {
        "C": "G(u) = pow(u, 2) - 1",
        "Python": "G = lambda u: u ** 2 - 1",
        "Rust": "G(u) = u.powi(2) - 1",
        "Julia": "G(u) = u ^ 2 - 1",
    }


def test_an_assignment_is_an_assignment_everywhere():
    for language in LANGUAGES:
        assert written("y := 5", language.word) == "y = 5"


# -- the file ----------------------------------------------------------------


def test_every_target_has_a_suffix_and_a_comment_marker_of_its_own():
    assert {language.word: language.suffix for language in LANGUAGES} == {
        "C": ".c",
        "Python": ".py",
        "Rust": ".rs",
        "Julia": ".jl",
    }
    assert {language.comment for language in LANGUAGES} == {"//", "#"}
