r"""Expression trees as LaTeX math, for the Markdown that Transfer Save writes.

The shape follows the screen: a quotient is a `\frac` where the screen draws a bar, and
parentheses appear where the screen draws them. The display corpus specifies the screen;
these tests check that the LaTeX makes the same choices, and spell out where LaTeX does
better than a terminal grid.

Nothing here runs a TeX engine. The output was checked with KaTeX and MathJax over the
parser corpus and the expressions of Derive's utility files and demos; the last test
keeps the cheapest part of that, that every formula is well nested.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from expr_cases import load

from rederive.display import ASTERISK, IMPLICIT, DisplayOptions, write_latex
from rederive.syntax import DeriveSyntaxError, InputMode, ParseState, parse_expression

LIBRARY = Path(__file__).parent / "corpus" / "library-expressions.txt"


DEFAULTS = DisplayOptions()


def latex(text: str, options: DisplayOptions = DEFAULTS, **state) -> str:
    return write_latex(parse_expression(text, ParseState(**state)).node, options)


# -- fractions ---------------------------------------------------------------


def test_a_quotient_is_built_up_as_on_screen():
    assert latex("x/3 + 1/3") == r"\frac{x}{3} + \frac{1}{3}"
    assert latex("(x^2 - 1)/(x - 1)") == r"\frac{x^{2} - 1}{x - 1}"


def test_a_quotient_in_a_script_is_linear_as_on_screen():
    """The screen writes `x^(1/2)` and `2^(1/(1 + x))` on one row, fenced as linear
    operands are."""
    assert latex("x^(1/2)") == "x^{1/2}"
    assert latex("2^(1/(1 + x))") == "2^{1/(1 + x)}"
    assert latex("e^(-x^2/2)") == "e^{-x^{2}/2}"


def test_a_nested_quotient_keeps_the_size_the_screen_gives_it():
    """LaTeX shrinks a fraction inside another or in a matrix; the screen doesn't."""
    assert latex("1/(1 + 1/x)") == r"\frac{1}{1 + \dfrac{1}{x}}"
    assert latex("[[1/2, x], [3, 4]]") == (
        r"\begin{bmatrix} \dfrac{1}{2} & x \cr 3 & 4 \end{bmatrix}"
    )
    assert latex("SIN(1/x)/2") == r"\frac{\sin\left( \dfrac{1}{x} \right)}{2}"


def test_a_run_of_factors_with_a_quotient_is_drawn_as_the_screen_draws_it():
    assert latex("a*b/c*d") == r"\frac{a \cdot b}{c} \cdot d"


# -- powers, roots and signs -------------------------------------------------


@pytest.mark.parametrize(
    "authored, expected",
    [
        ("x^2", "x^{2}"),
        ("2^3^4", "2^{3^{4}}"),
        ("(x^y)^z", "(x^{y})^{z}"),
        ("(-2)^x", "(-2)^{x}"),
        ("-x^2", "-x^{2}"),
        ("(-x)^2", "(-x)^{2}"),
        ("(a/b)^2", r"\left( \frac{a}{b} \right)^{2}"),
        ("x^-1", "x^{-1}"),
    ],
)
def test_a_power_fences_its_base_as_the_screen_does(authored, expected):
    assert latex(authored) == expected


def test_an_exponent_tower_is_raised_however_deep():
    """The screen writes the third level flat, `x^y`, for lack of rows; LaTeX can
    keep raising."""
    assert latex("a^b^c^d") == "a^{b^{c^{d}}}"


def test_a_root_takes_an_overbar_instead_of_parentheses():
    assert latex("SQRT(x^2 + 1)") == r"\sqrt{x^{2} + 1}"
    assert latex("SQRT(1 + SQRT(x))/2") == r"\frac{\sqrt{1 + \sqrt{x}}}{2}"
    # Raised, the root is fenced, as on screen.
    assert latex("SQRT(x)^2") == r"(\sqrt{x})^{2}"


def test_a_sign_and_a_product_are_fenced_as_on_screen():
    assert latex("a*(-b)") == r"a \cdot (-b)"
    assert latex("(a + b)(c - d)^2") == r"(a + b) \cdot (c - d)^{2}"
    assert latex("a - (b - c)") == "a - (b - c)"
    assert latex("a + (b + c)") == "a + (b + c)"
    assert latex('"+-"x') == r"\pm x"


def test_the_times_operator_is_the_one_the_screen_shows():
    assert latex("2 a") == r"2 \cdot a"
    assert latex("2 a", DisplayOptions(times=ASTERISK)) == r"2 \ast a"
    assert latex("2 a", DisplayOptions(times=IMPLICIT)) == r"2 \thinspace a"


def test_a_power_never_follows_another_superscript_directly():
    """TeX rejects `x^{a}^{b}`, so a base ending in a raised mark is braced."""
    assert latex("(30 DEG)^2") == r"{30^{\circ}}^{2}"
    assert latex("A``") == r"{a^{\mathrm{T}}}^{\mathrm{T}}"
    assert latex("x^deg") == r"x^{{}^{\circ}}"


# -- delimiters --------------------------------------------------------------


def test_delimiters_grow_only_around_something_taller_than_a_line():
    r"""`\left(` after a function name sets it off with a space, so it's kept for the
    tall operands that need it."""
    assert latex("SIN(x)") == r"\sin(x)"
    assert latex("SIN(x + pi/4)") == r"\sin\left( x + \frac{\pi}{4} \right)"
    assert latex("ABS(x - 1)") == r"\lvert x - 1 \rvert"
    assert latex("ABS(1/x)") == r"\left| \frac{1}{x} \right|"


def test_vectors_sets_and_intervals():
    assert latex("[1, ..., 5]") == r"[1, \ldots, 5]"
    assert latex("[[1, 2], [3]]") == "[[1, 2], [3]]"
    assert latex("{1, 2} UNION {3} \\ {2}") == (
        r"\lbrace 1, 2 \rbrace \cup \lbrace 3 \rbrace \setminus \lbrace 2 \rbrace"
    )
    assert latex("x :epsilon Real (0, inf]") == r"x \in \mathbb{R}\thinspace (0, \infty]"
    assert latex("n :epsilon Nonscalar") == r"n \in \mathrm{Nonscalar}"


def test_a_matrix_is_built_up_where_the_screen_builds_it_up():
    assert latex("[[1, 2], [3, 4]]") == r"\begin{bmatrix} 1 & 2 \cr 3 & 4 \end{bmatrix}"
    assert latex("x^[[1, 2], [3, 4]]") == "x^{[[1, 2], [3, 4]]}"
    # A call drops its parentheses around one, as on screen.
    assert latex("DET([[1, 2], [3, 4]])") == (
        r"\det\thinspace \begin{bmatrix} 1 & 2 \cr 3 & 4 \end{bmatrix}"
    )


# -- names -------------------------------------------------------------------


@pytest.mark.parametrize(
    "authored, expected",
    [
        ("alpha + Gamma + epsilon + phi", r"\alpha + \Gamma + \varepsilon + \varphi"),
        ("Beta + omicron", r"\mathrm{B} + o"),
        ("pi + inf + #e + #i", r"\pi + \infty + \mathrm{e} + \mathrm{i}"),
        ("true", r"\mathrm{true}"),
        ("é + 1", r"\text{é} + 1"),
    ],
)
def test_a_name_is_shown_as_the_screen_shows_it(authored, expected):
    assert latex(authored) == expected


def test_a_longer_name_is_one_word_and_not_a_product():
    word = InputMode.WORD
    assert latex("abc + x1", input_mode=word) == r"\mathit{abc} + \mathit{x1}"


#: An underscore as drawn in a name; see `test_no_backslash_precedes_punctuation`.
UNDERSCORE = r"\underline{\hphantom{x}}"


def test_a_letter_with_an_index_after_an_underscore_is_subscripted():
    word = InputMode.WORD
    assert latex("x_12 + a_max", input_mode=word) == r"x_{12} + a_{\mathit{max}}"
    assert latex("my_var + c1_", input_mode=word) == (
        rf"\mathit{{my{UNDERSCORE}var}} + \mathit{{c1{UNDERSCORE}}}"
    )
    # As a function's name too.
    assert latex("F_00(x) := x", input_mode=word) == r"F_{00}(x) \mathrel{:=} x"


def test_a_subscripted_name_is_fenced_before_another_subscript():
    """TeX rejects `x_{1}_{2}`, and `{x_{1}}_{2}` would read as `x_{12}`."""
    word = InputMode.WORD
    assert latex("x_1 SUB 2", input_mode=word) == "(x_{1})_{2}"
    assert latex("g_ij SUB j SUB k", input_mode=word) == (
        r"(g_{\mathit{ij}})_{j,k}"
    )
    # A power or a call after the name needs nothing.
    assert latex("x_1^2", input_mode=word) == "x_{1}^{2}"
    assert latex("f_1'(x) SUB 2", input_mode=word) == "f_{1}'(x)_{2}"


def test_a_name_beyond_ascii_letters_is_escaped():
    word = InputMode.WORD
    assert latex("SEC_δ(v_) := 1", input_mode=word) == (
        rf"\operatorname{{SEC{UNDERSCORE}{{\delta}}}}(\mathit{{v{UNDERSCORE}}})"
        r" \mathrel{:=} 1"
    )

def test_a_string_is_text_with_its_special_characters_set_as_math():
    """MathJax's text mode knows none of LaTeX's escapes, and KaTeX's rejects `_`."""
    assert latex('"hi there"') == r"\text{“hi there”}"
    assert latex(r'"a_b #1 $5 & 50% {x} ~ ^ \"') == (
        rf"\text{{“a}}{UNDERSCORE}\text{{b }}\#\text{{1 }}\$\text{{5 }}\And\text{{ 50}}\%"
        r"\text{ }\lbrace\text{x}\rbrace"
        r"\text{ }\sim\text{ }\hat{}\text{ }\backslash\text{”}"
    )


def test_primes_stay_outside_the_name():
    word = InputMode.WORD
    assert latex("f'(x) + BB'(x)", input_mode=word) == r"f'(x) + \operatorname{bb}'(x)"


def test_a_subscript_is_lowered_and_nested_ones_merge_as_on_screen():
    assert latex("v SUB 1 SUB 2") == "v_{1,2}"
    assert latex("[a, b] SUB (i + 1)") == "[a, b]_{i + 1}"
    assert latex("(x^2) SUB 1") == "(x^{2})_{1}"


# -- operators ---------------------------------------------------------------


def test_relations_and_logic():
    assert latex("x <= y AND x /= 0") == r"x \le y \land x \ne 0"
    assert latex("a AND NOT b OR c XOR d") == r"a \land \lnot b \lor c \veebar d"
    assert latex("a IMP b IFF c") == r"a \Rightarrow b \Leftrightarrow c"
    assert latex("a OR b OR c") == r"a \lor b \lor c"
    assert latex("(a OR b) OR c") == r"(a \lor b) \lor c"
    assert latex("a AND b AND c") == r"a \land b \land c"
    assert latex("(a IMP b) IMP c") == r"a \Rightarrow b \Rightarrow c"
    assert latex("a IMP (b IMP c)") == r"a \Rightarrow (b \Rightarrow c)"
    assert latex("a SUBSET b") == r"a \subseteq b"


def test_the_postfix_operators():
    assert latex("n!") == "n!"
    assert latex("(n + 1)!") == "(n + 1)!"
    assert latex("5%") == r"5\%"
    assert latex("30 DEG") == r"30^{\circ}"
    assert latex("A`") == r"a^{\mathrm{T}}"


def test_the_products_of_vectors():
    assert latex("a.b") == r"a \bullet b"
    assert latex("a CROSSPRODUCT b") == r"a \times b"


# -- the special forms -------------------------------------------------------


def test_an_integral_with_and_without_bounds():
    assert latex("INT(x SIN(x), x)") == r"\int x \cdot \sin(x) \thinspace dx"
    assert latex("INT(1/x, x, 1, #e)") == (
        r"\int_{1}^{\mathrm{e}} \frac{1}{x} \thinspace dx"
    )
    assert latex("INT(a + b, x)") == r"\int (a + b) \thinspace dx"


def test_a_limit_shows_its_direction_as_a_sign():
    assert latex("LIM(SIN(x)/x, x, 0)") == r"\lim_{x \to 0} \frac{\sin(x)}{x}"
    assert latex("LIM(1/x, x, 0, -1)") == r"\lim_{x \to {0}^{-}} \frac{1}{x}"
    assert latex("LIM(1/x, x, 0, 1)") == r"\lim_{x \to {0}^{+}} \frac{1}{x}"


def test_a_sum_and_a_product_with_and_without_bounds():
    assert latex("SUM(1/k^2, k, 1, inf)") == r"\sum_{k=1}^{\infty} \frac{1}{k^{2}}"
    assert latex("PRODUCT(k, k, 1, n)") == r"\prod_{k=1}^{n} k"
    assert latex("SUM(k, k)") == r"\sum_{k} k"


def test_a_form_ending_in_its_body_is_fenced_off_as_on_screen():
    """`Σ k + 1` would read as the sum of `k + 1`."""
    assert latex("SUM(k, k) + 1") == r"\left( \sum_{k} k \right) + 1"
    assert latex("2 SUM(k, k)") == r"2 \cdot \sum_{k} k"


def test_a_derivative_is_written_in_leibniz_notation():
    """The screen raises the operator, `(d/dx)^2`; in print the order goes on the
    `d`s."""
    assert latex("DIF(x^3, x)") == r"\frac{d}{dx} x^{3}"
    assert latex("DIF(a*b, x)") == r"\frac{d}{dx} (a \cdot b)"
    assert latex("DIF(x^3, x, 2)") == r"\frac{d^{2}}{dx^{2}} x^{3}"


@pytest.mark.parametrize(
    "authored, expected",
    [
        ("LN(x) + EXP(x) + COS(x)", r"\ln(x) + \exp(x) + \cos(x)"),
        ("ASIN(x) + ASINH(x)", r"\arcsin(x) + \operatorname{arsinh}(x)"),
        ("LOG(x, 2)", r"\log_{2}(x)"),
        ("GAMMA(x) + ERF(x)", r"\Gamma(x) + \operatorname{erf}(x)"),
        (
            "FLOOR(x/2) + CEILING(y)",
            r"\left\lfloor \frac{x}{2} \right\rfloor + \lceil y \rceil",
        ),
        ("SIN^2(x)", r"\sin(x)^{2}"),
    ],
)
def test_a_familiar_function_has_its_usual_spelling(authored, expected):
    assert latex(authored) == expected


def test_anything_else_is_a_call_as_written():
    assert latex("ITERATE(x/2, x, 1, 5)") == (
        r"\operatorname{ITERATE}\left( \frac{x}{2}, x, 1, 5 \right)"
    )
    assert latex("PROG(a := 1, RETURN a)") == (
        r"\operatorname{PROG}(a \mathrel{:=} 1, \operatorname{RETURN}(a))"
    )
    assert latex("MOD(a, b)") == r"\operatorname{MOD}(a, b)"


# -- definitions and the rest ------------------------------------------------


def test_assignments_and_definitions():
    assert latex("y := 5") == r"y \mathrel{:=} 5"
    assert latex("y :=") == r"y \mathrel{:=}"
    assert latex("a :+ 1") == r"a \mathrel{{:}{+}} 1"
    assert latex("f(x) := IF(x > 0, x^2, -x)") == (
        r"f(x) \mathrel{:=} \operatorname{IF}(x > 0, x^{2}, -x)"
    )
    assert latex("area(r) := pi r^2", input_mode=InputMode.WORD) == (
        r"\operatorname{area}(r) \mathrel{:=} \pi \cdot r^{2}"
    )
    assert latex("F x := x") == r"F\thinspace x \mathrel{:=} x"


def test_the_leaves_that_are_not_mathematics():
    assert latex("#3 + 1") == r"\#3 + 1"
    assert latex("?") == "?"
    assert latex("'x") == r"\text{'}x"
    assert latex('"x" :epsilon Real') == r"\text{“x”} \in \mathbb{R}"


def test_a_number_is_shown_in_the_output_base():
    hexadecimal = DisplayOptions(output_base=16)
    assert latex("255 + 10") == "255 + 10"
    assert latex("255 + 10", hexadecimal) == r"\mathrm{0FF} + \mathrm{0A}"


def corpus_formulas() -> list[tuple[str, str]]:
    """Every expression of the parser corpus and of Derive's utility and demo files, with
    its LaTeX. A group of the latter is read in order with the declarations its lines
    make, so that a call parses as a call."""
    formulas = []
    for case in load():
        if not case.expected.startswith("ERROR"):
            node = parse_expression(case.text, case.state()).node
            formulas.append((case.text, write_latex(node)))
    state = ParseState()
    for line in LIBRARY.read_text(encoding="utf-8").splitlines():
        if line.startswith("##"):
            state = ParseState()
        elif line.strip() and not line.startswith("#"):
            try:
                result = parse_expression(line, state)
            except DeriveSyntaxError:
                continue
            for declaration in result.declarations:
                state.declare(declaration)
            formulas.append((line, write_latex(result.node)))
    return formulas


@pytest.fixture(scope="module")
def formulas() -> list[tuple[str, str]]:
    return corpus_formulas()


def test_every_formula_is_well_nested(formulas):
    r"""Each brace and `\left` is closed, in the order opened."""
    for text, latex in formulas:
        stack = []
        for at, character in enumerate(latex):
            escaped = at > 0 and latex[at - 1] == "\\"
            if character == "{" and not escaped:
                stack.append(text)
            elif character == "}" and not escaped:
                assert stack.pop() == text
        assert not stack, text
        assert latex.count(r"\left") == latex.count(r"\right"), text


#: The pairs of a backslash and ASCII punctuation that are written anyway, since nothing
#: else both KaTeX and MathJax accept spells these characters.
UNAVOIDABLE = {r"\#", r"\$", r"\%"}


def test_no_backslash_precedes_punctuation(formulas):
    r"""GitHub removes a backslash before ASCII punctuation inside math, as Markdown
    escaping, before MathJax sees it, so `\\` would end no matrix row and `\{` draw
    no brace."""
    pairs = re.compile(r"\\[!-/:-@\[-`{-~]")
    for text, latex in formulas:
        assert set(pairs.findall(latex)) <= UNAVOIDABLE, (text, latex)
