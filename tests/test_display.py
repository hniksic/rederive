"""The display conformance corpus.

Every block below is the original's own output and therefore the
specification, never something to edit until an implementation passes. The
render blocks are its reference renders; the fencing, sign and selection
sections were checked against the original, as were the renders.

Two blocks carry a correction, marked where each occurs. The transcription
they reached us in disagrees with the program itself, and in both cases with
the rest of the corpus as well.

Tests may look inside `rederive.display`; the rule that nothing outside the
package imports from inside it is about the application, not about its tests.
"""

from __future__ import annotations

from textwrap import dedent

import pytest
from sexpr import to_sexpr

from rederive.display import DisplayOptions, boxes, glyphs, render
from rederive.syntax import ParseState, names, parse_expression

DEFAULTS = DisplayOptions()


def layout_of(text: str, options: DisplayOptions = DEFAULTS):
    return render(parse_expression(text, ParseState()).node, options)


def rendered(text: str, options: DisplayOptions = DEFAULTS) -> list[str]:
    return [line.rstrip() for line in layout_of(text, options).lines]


def block(text: str) -> list[str]:
    return [line.rstrip() for line in dedent(text).strip("\n").split("\n")]


def check(text: str, expected: str, options: DisplayOptions = DEFAULTS) -> None:
    """Compare a render against its block, rstripped, and report readably.

    A line-by-line diff of two multi-row renders is unreadable, so failures
    print the two blocks one above the other instead.
    """
    actual = rendered(text, options)
    wanted = block(expected)
    if actual != wanted:
        report = "\n".join(
            [f"{text}", "expected:", *wanted, "actual:", *actual, ""]
        )
        pytest.fail(report, pytrace=False)


# -- fractions --------------------------------------------------------------

FRACTIONS = [
    (
        "(x+1)/(x^2+2x+3)",
        """
             x + 1
        ──────────────
          2
         x  + 2·x + 3
        """,
    ),
    (
        "a/b/c/d",
        """
           a
          ───
           b
         ─────
           c
        ───────
           d
        """,
    ),
    (
        "1+1/(1+1/(1+1/x))",
        """
                   1
        1 + ───────────────
                     1
             1 + ─────────
                       1
                  1 + ───
                       x
        """,
    ),
    (
        # Corrected against the program: the reference block puts `x` one
        # column left of centre, where every other block - a one-wide
        # numerator over bars of 3, 5 and 9 - centres exactly.
        "x/(a+b)",
        """
           x
        ───────
         a + b
        """,
    ),
    (
        "SIN(x)/COS(x)",
        """
         SIN(x)
        ────────
         COS(x)
        """,
    ),
    (
        "a/(b/c)",
        """
          a
        ─────
          b
         ───
          c
        """,
    ),
    (
        "x!/(n+1)!",
        """
            x!
        ──────────
         (n + 1)!
        """,
    ),
    (
        "-(a/b)",
        """
           a
        - ───
           b
        """,
    ),
    (
        "a+b/c+d",
        """
             b
        a + ─── + d
             c
        """,
    ),
]

# -- superscripts and subscripts --------------------------------------------

SCRIPTS = [
    (
        "SQRT(x^2+1)",
        """
           2
        √(x  + 1)
        """,
    ),
    (
        "2^(x+1)",
        """
         x + 1
        2
        """,
    ),
    (
        "x^(1/2)",
        """
         1/2
        x
        """,
    ),
    (
        "x^(y^z)",
        """
          z
         y
        x
        """,
    ),
    (
        "(a/b)^(c/d)",
        """
        ┌ a ┐c/d
        │───│
        └ b ┘
        """,
    ),
    (
        "x^((a+b)/(c+d))",
        """
         (a + b)/(c + d)
        x
        """,
    ),
    (
        "x^(y^(z^w))",
        """
          z^w
         y
        x
        """,
    ),
    (
        "x^(y^(z^(w^(v^(u^(t^s))))))",
        """
          z^w^v^u^t^s
         y
        x
        """,
    ),
    (
        "x^-1",
        """
         -1
        x
        """,
    ),
    (
        "2^(-x/y)",
        """
         - x/y
        2
        """,
    ),
    (
        "(x^2+1)^3",
        """
          2     3
        (x  + 1)
        """,
    ),
    (
        "(a/b+1)^2",
        """
        ┌ a     ┐2
        │─── + 1│
        └ b     ┘
        """,
    ),
    (
        "x^(y+1/z)",
        """
         y + 1/z
        x
        """,
    ),
    (
        "x^(SIN(a/b))",
        """
         SIN(a/b)
        x
        """,
    ),
    (
        "(x^(1/2))/(y+1)",
        """
           1/2
          x
        ───────
         y + 1
        """,
    ),
    (
        "x SUB (n+1)",
        """
        x
         n + 1
        """,
    ),
    (
        "v SUB 1 SUB 2",
        """
        v
         1,2
        """,
    ),
    (
        "SQRT^3 25",
        """
           3
        √25
        """,
    ),
]

# -- fences -----------------------------------------------------------------

FENCES = [
    (
        "SIN(x^2+1)",
        """
             2
        SIN(x  + 1)
        """,
    ),
    (
        "SIN(a SUB (b+c))",
        """
        SIN(a     )
             b + c
        """,
    ),
    (
        "SIN((x+1)/(x-1))",
        """
           ┌ x + 1 ┐
        SIN│───────│
           └ x - 1 ┘
        """,
    ),
    (
        "SQRT((x+1)/(x-1))",
        """
         ┌ x + 1 ┐
        √│───────│
         └ x - 1 ┘
        """,
    ),
    (
        "ABS((x+1)/(x-1))",
        """
        │ x + 1 │
        │───────│
        │ x - 1 │
        """,
    ),
    (
        "(x^2+1)(y+1)",
        """
          2
        (x  + 1)·(y + 1)
        """,
    ),
]

# -- vectors and matrices ---------------------------------------------------

VECTORS = [
    (
        "[[1,2],[3,4]]",
        """
        ┌ 1  2 ┐
        │      │
        └ 3  4 ┘
        """,
    ),
    (
        "[[1,2,3],[4,5,6],[7,8,9]]",
        """
        ┌ 1  2  3 ┐
        │         │
        │ 4  5  6 │
        │         │
        └ 7  8  9 ┘
        """,
    ),
    (
        "[[a/b,2],[3,SQRT(x)]]",
        """
        ┌  a      ┐
        │ ───   2 │
        │  b      │
        │         │
        └  3   √x ┘
        """,
    ),
    (
        "[(a+b)/c, d]",
        """
        ┌ a + b    ┐
        │───────, d│
        └   c      ┘
        """,
    ),
    (
        "[1,2].[3,4]",
        """
        [1, 2] ∙ [3, 4]
        """,
    ),
]

# -- calculus forms ---------------------------------------------------------

CALCULUS = [
    (
        "INT(x^2/(x+1),x)",
        """
        ⌠     2
        │    x
        │ ─────── dx
        ⌡  x + 1
        """,
    ),
    (
        "INT(x^2,x,0,1)",
        """
         1
        ⌠   2
        ⌡  x  dx
         0
        """,
    ),
    (
        "SUM(k^2,k,1,n)",
        """
         n   2
         Σ  k
        k=1
        """,
    ),
    (
        # Corrected against the program: the reference block drops the `x` of
        # the denominator, which shares its row with `x→0` because `lim` sits
        # on the bar row.
        "LIM((SIN x)/x,x,0)",
        """
             SIN(x)
        lim ────────
        x→0     x
        """,
    ),
    (
        "DIF(x^3,x,2)",
        """
        ┌d ┐2  3
        │──│  x
        └dx┘
        """,
    ),
]

CORPUS = FRACTIONS + SCRIPTS + FENCES + VECTORS + CALCULUS


@pytest.mark.parametrize(
    ("text", "expected"), CORPUS, ids=[text for text, _ in CORPUS]
)
def test_corpus(text: str, expected: str) -> None:
    check(text, expected)


# -- single-row forms -------------------------------------------------------

SINGLE_ROW = [
    ("-x-(-y)", "-x - -y"),
    ("3-(-2)", "3 - -2"),
    ("3+-2", "3 + -2"),
    ("a=b<c", "a = b < c"),
    ("x>=1 AND NOT y<2", "x ≥ 1 AND NOT y < 2"),
    ('"hello"', '"hello"'),
    ("5%+x", "5% + x"),
    ("x:epsilon Real [0,inf)", "x :ε Real [0, ∞)"),
    # An interval's bounds are written in author notation rather than built
    # up, so a bound that is a fraction or a power stays on the row.
    ("x:epsilon Real (1/2, 5)", "x :ε Real (1/2, 5)"),
    ("y:epsilon Real (2^10, 1/3)", "y :ε Real (2^10, 1/3)"),
    ("p:epsilon Real [-2/3, 5)", "p :ε Real [- 2/3, 5)"),
    ("90 deg", "90·°"),
]


@pytest.mark.parametrize(("text", "expected"), SINGLE_ROW)
def test_single_row_forms(text: str, expected: str) -> None:
    assert rendered(text) == [expected]


def test_constants_print_as_their_glyphs() -> None:
    check(
        "#e^(#i pi)+?",
        """
         î·π
        ê    + ?
        """,
    )


def test_a_definition_prints_its_head_on_the_baseline() -> None:
    check(
        "f(x):=x^2/2",
        """
                  2
                 x
        F(x) := ────
                  2
        """,
    )


# -- output options ---------------------------------------------------------

OPTION_CASE = "a b + c/d + (x+1)^2 - SIN(x)"

OPTIONS = [
    (
        DisplayOptions(times="Dot"),
        """
               c           2
        a·b + ─── + (x + 1)  - SIN(x)
               d
        """,
    ),
    (
        DisplayOptions(times="Asterisk"),
        """
               c           2
        a*b + ─── + (x + 1)  - SIN(x)
               d
        """,
    ),
    (
        DisplayOptions(times="Implicit"),
        """
               c           2
        a b + ─── + (x + 1)  - SIN(x)
               d
        """,
    ),
    (
        # The bar's last cell is `╴`, half the width of `─`.
        DisplayOptions(compressed=True, times="Dot"),
        """
             c       2
        a·b+──╴+(x+1) -SIN(x)
             d
        """,
    ),
    (
        DisplayOptions(compressed=True, times="Implicit"),
        """
             c       2
        a b+──╴+(x+1) -SIN(x)
             d
        """,
    ),
]


@pytest.mark.parametrize(("options", "expected"), OPTIONS)
def test_output_options(options: DisplayOptions, expected: str) -> None:
    check(OPTION_CASE, expected, options)


@pytest.mark.parametrize(
    ("text", "base", "expected"),
    [("255", 16, "0FF"), ("255", 10, "255"), ("10", 2, "1010"), ("-14", 16, "-0E")],
)
def test_output_radix(text: str, base: int, expected: str) -> None:
    assert rendered(text, DisplayOptions(output_base=base)) == [expected]


def test_a_numeral_with_a_radix_point_is_shown_as_written() -> None:
    # Converting the fraction would be arithmetic, which this layer does none of.
    assert rendered("1.5", DisplayOptions(output_base=16)) == ["1.5"]


@pytest.mark.parametrize(
    ("value", "base", "expected"),
    [("1/3", 10, "1/3"), ("1/3", 3, "1/10"), ("10/3", 16, "0A/3")],
)
def test_output_radix_of_a_ratio(value: str, base: int, expected: str) -> None:
    # What a numeral carries when no finite decimal is worth it: `0.1` read in
    # base three. Both halves are whole, so both convert.
    assert glyphs.numeral(value, base, 6) == expected


# -- notation digits --------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.123456789", "0.123456"),
        ("12345.67891", "12345.6"),
        ("1234567.891", "1234567.8"),
        ("0.000123456789", "0.000123456"),
        # Cut, not rounded: the digit after the last one shown never carries.
        ("0.9999999", "0.999999"),
        ("1.9999999", "1.99999"),
        # A cut that leaves nothing but zeros leaves the point its one digit.
        ("0.10000000000000001", "0.1"),
        ("1.0000001", "1.0"),
        # A whole number is exact, and is shown in full.
        ("123456789", "123456789"),
    ],
)
def test_notation_digits(text: str, expected: str) -> None:
    assert rendered(text) == [expected]


def test_a_ratio_is_shown_in_full() -> None:
    # Rational notation says what it is worth however long that is; there is
    # nothing to cut, the two halves being whole.
    assert glyphs.numeral("1/3", 10, 6) == "1/3"


@pytest.mark.parametrize(
    ("digits", "expected"),
    [(1, "3.1"), (3, "3.14"), (12, "3.14159265358")],
)
def test_notation_digits_is_the_setting(digits: int, expected: str) -> None:
    options = DisplayOptions(notation_digits=digits)
    assert rendered("3.14159265358979", options) == [expected]


# -- the selection tree -----------------------------------------------------

# A route is a tuple of indices into `Region.children`, empty for the whole
# expression. The rectangles are the ones the render section fixes; only the
# addresses are the selection tree's business.


@pytest.mark.parametrize(
    ("text", "route", "rect"),
    [
        # The three terms of `a + b/c + d` hang directly off the root, and
        # `a + b/c` is not addressable at all.
        ("a+b/c+d", (), (0, 0, 3, 11)),
        ("a+b/c+d", (0,), (1, 0, 1, 1)),
        ("a+b/c+d", (1,), (0, 4, 3, 3)),
        ("a+b/c+d", (1, 0), (0, 5, 1, 1)),
        ("a+b/c+d", (1, 1), (2, 5, 1, 1)),
        ("a+b/c+d", (2,), (1, 10, 1, 1)),
        # A fraction has two operands, drawn built up or not.
        ("(x+1)/(x^2+2x+3)", (0,), (0, 5, 1, 5)),
        ("(x+1)/(x^2+2x+3)", (1,), (2, 1, 2, 12)),
        # The head is dropped, so the argument moves to `(0,)`.
        ("SIN(x+1)", (0,), (0, 4, 1, 5)),
        ("a^b^c", (1,), (0, 1, 2, 2)),
    ],
)
def test_routes(text: str, route: tuple[int, ...], rect: tuple[int, ...]) -> None:
    region = layout_of(text).at(route)
    assert region is not None
    assert region.rect == rect


def test_a_route_that_leads_nowhere_is_none() -> None:
    assert layout_of("a+b").at((5,)) is None
    assert layout_of("a+b").at((0, 0)) is None


def test_a_route_names_the_subexpression_it_covers() -> None:
    """`Region.node` is how an operation gets back to what is selected."""
    layout = layout_of("a+b/c+d")
    assert to_sexpr(layout.at(()).node) == "(+ a (/ b c) d)"
    assert to_sexpr(layout.at((1,)).node) == "(/ b c)"
    assert to_sexpr(layout.at((1, 1)).node) == "c"


def test_a_region_is_the_rectangle_a_subexpression_covers() -> None:
    # `b/c` in `a + b/c + d`: the blanks beside `b` and `c` are inside it.
    layout = layout_of("a+b/c+d")
    region = layout.at((1,))
    assert region is not None
    covered = [
        line[region.left : region.left + region.width]
        for line in layout.lines[region.top : region.top + region.height]
    ]
    assert covered == [" b ", "───", " c "]


# -- which operands are selectable ------------------------------------------


def operands(text: str, route: tuple[int, ...] = ()) -> list[str]:
    """What the cursor visits from `route`, in order."""
    region = layout_of(text).at(route)
    assert region is not None
    return [to_sexpr(child.node) for child in region.children]


def covers(text: str, route: tuple[int, ...]) -> list[str]:
    """The cells a route's rectangle covers, rstripped."""
    layout = layout_of(text)
    region = layout.at(route)
    assert region is not None
    return [
        line[region.left : region.left + region.width].rstrip()
        for line in layout.lines[region.top : region.top + region.height]
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a+b+c", ["a", "b", "c"]),
        # The sign belongs to the run, so the third term is `c`, not `-c`.
        ("a+b-c", ["a", "b", "c"]),
        ("a b c", ["a", "b", "c"]),
        ("a*b*c", ["a", "b", "c"]),
        # `/` and the dot product are binary and drawn as groups of their own.
        ("a*b/c", ["(* a b)", "c"]),
        ("a*b/c*d", ["(/ (* a b) c)", "d"]),
        ("[1,2].[3,4]", ["(vec 1 2)", "(vec 3 4)"]),
        # Parentheses are a run of their own, whichever operator they group.
        ("a-(b-c)", ["a", "(- b c)"]),
        ("a+(b+c)", ["a", "(+ b c)"]),
        ("a*(b*c)", ["a", "(* b c)"]),
    ],
)
def test_a_run_offers_its_terms(text: str, expected: list[str]) -> None:
    assert operands(text) == expected


def test_a_parenthesised_run_keeps_its_fences() -> None:
    """`a + (b + c)` is two terms in the original, and says so on the screen.

    Only the parentheses tell it from `a + b + c`, which is three, so they
    are drawn even though `+` would be read the same way without them.
    """
    check("a+(b+c)", "a + (b + c)")
    check("a*(b*c)", "a·(b·c)")
    check("a+b+c", "a + b + c")


def test_a_flattened_term_covers_the_term_alone() -> None:
    assert covers("a+b-c", (2,)) == ["c"]
    assert covers("a b c", (1,)) == ["b"]


def test_a_head_is_not_an_operand() -> None:
    assert operands("SIN(x+1)") == ["(+ x 1)"]
    assert operands("SIN x") == ["x"]
    # `SQRT^3 25` offers the application and the exponent, not the name.
    assert operands("SQRT^3 25") == ["(funcpow SQRT 3 25)", "3"]
    assert covers("SQRT^3 25", (0,)) == ["√25"]
    assert operands("SQRT^3 25", (0,)) == ["25"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("INT(x^2,x)", ["(^ x 2)", "x"]),
        # Argument order, though the bounds are drawn first.
        ("INT(x^2,x,0,1)", ["(^ x 2)", "x", "0", "1"]),
        ("SUM(k^2,k,1,n)", ["(^ k 2)", "k", "1", "n"]),
        ("PRODUCT(k,k,1,n)", ["k", "k", "1", "n"]),
        ("LIM((SIN x)/x,x,0)", ["(/ (apply SIN x) x)", "x", "0"]),
        ("DIF(x^3,x,2)", ["(^ x 3)", "x", "2"]),
        ("DIF(x,x)", ["x", "x"]),
        ("SQRT(x+1)", ["(+ x 1)"]),
        ("ABS(x+1)", ["(+ x 1)"]),
        ("|x+1|", ["(+ x 1)"]),
    ],
)
def test_a_special_form_names_its_own_operands(text: str, expected: list[str]) -> None:
    assert operands(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("x^2", ["x", "2"]),
        ("x SUB 1", ["x", "1"]),
        ("x!", ["x"]),
        ("NOT x", ["x"]),
        ("x AND y", ["x", "y"]),
        ("[a,b,c]", ["a", "b", "c"]),
        ("[[1,2],[3,4]]", ["(vec 1 2)", "(vec 3 4)"]),
        ("x:=1", ["x", "1"]),
        ("x+1=", ["(+ x 1)"]),
    ],
)
def test_everything_else_follows_the_tree(text: str, expected: list[str]) -> None:
    assert operands(text) == expected


def test_a_matrix_offers_rows_and_then_cells() -> None:
    assert operands("[[1,2],[3,4]]", (0,)) == ["1", "2"]
    assert covers("[[1,2],[3,4]]", (0,)) == ["1  2"]


# -- confirmed against the original -----------------------------------------

# Checked against the original: with the expression authored, these are the
# cells that go to inverse video under Right.


def test_a_subscript_chain_nests_rather_than_flattening() -> None:
    """`v SUB 1 SUB 2` offers `v SUB 1` and `2`, not `v`, `1` and `2`."""
    assert operands("v SUB 1 SUB 2") == ["(sub v 1)", "2"]
    assert covers("v SUB 1 SUB 2", (0,)) == ["v", " 1"]
    assert operands("v SUB 1 SUB 2", (0,)) == ["v", "1"]


def test_a_definition_offers_its_head_and_its_body() -> None:
    """The head selects whole, `F(x)`, and steps into its parameters."""
    assert operands("f(x):=x^2/2") == ["(params x)", "(/ (^ x 2) 2)"]
    assert covers("f(x):=x^2/2", (0,)) == ["F(x)"]
    assert operands("f(x):=x^2/2", (0,)) == ["x"]


def test_a_quotient_of_a_product_offers_the_product_whole() -> None:
    assert operands("a*b/c") == ["(* a b)", "c"]
    assert operands("a*b/c", (0,)) == ["a", "b"]


def test_the_operand_of_a_unary_sign_is_selectable() -> None:
    assert operands("-x") == ["x"]
    assert covers("-x", (0,)) == ["x"]


def test_a_relation_chain_selects_as_nested_pairs() -> None:
    # `(a = b) < c`: two operands at the root, and the left one steps into
    # its own pair. The original does the same.
    assert operands("a=b<c") == ["(rel a = b)", "c"]
    assert covers("a=b<c", (0,)) == ["a = b"]
    assert operands("a=b<c", (0,)) == ["a", "b"]


def test_a_domain_and_its_interval_select_together() -> None:
    assert covers("x:epsilon Real [0,inf)", (1,)) == ["Real [0, ∞)"]
    assert operands("x:epsilon Real [0,inf)") == ["x", '(ivl "[" 0 inf ")")']


# -- routes survive a re-render ---------------------------------------------


@pytest.mark.parametrize(
    "options",
    [
        DisplayOptions(),
        DisplayOptions(compressed=True),
        DisplayOptions(times="Implicit"),
        # Low enough to force the fraction into its linear form.
        DisplayOptions(height=1),
    ],
)
def test_a_route_names_the_same_node_however_it_is_drawn(
    options: DisplayOptions,
) -> None:
    text = "a+b/c+d"
    node = parse_expression(text, ParseState()).node
    reference = render(node)
    other = render(node, options)
    for route in [(), (0,), (1,), (1, 0), (1, 1), (2,)]:
        here, there = reference.at(route), other.at(route)
        assert here is not None and there is not None, route
        assert here.node is there.node, route


def test_a_degraded_fraction_keeps_its_operands() -> None:
    # The bar becomes a slash, and numerator and denominator stay put.
    assert rendered("a+b/c+d", DisplayOptions(height=1)) == ["a + b/c + d"]
    node = parse_expression("a+b/c+d", ParseState()).node
    layout = render(node, DisplayOptions(height=1))
    assert to_sexpr(layout.at((1,)).node) == "(/ b c)"
    assert layout.at((1, 0)).rect == (0, 4, 1, 1)


# -- the height budget ------------------------------------------------------


def continued_fraction(depth: int) -> str:
    """`1+1/(1+1/(...))` nested to `depth`, innermost operand `x`."""
    text = "x"
    for _ in range(depth):
        text = f"1+1/({text})"
    return text


@pytest.mark.parametrize(
    ("depth", "height"), [(9, 19), (10, 21), (11, 23)]
)
def test_a_continued_fraction_is_fully_built_up_when_it_may_be(
    depth: int, height: int
) -> None:
    assert len(rendered(continued_fraction(depth))) == height


@pytest.mark.parametrize("depth", [9, 10, 11])
def test_degrading_stops_as_soon_as_the_render_fits(depth: int) -> None:
    lines = rendered(continued_fraction(depth), DisplayOptions(height=19))
    assert len(lines) == 19
    # Outermost first: one linear division per level of overflow, and none
    # deeper than that.
    assert lines[1].count("1/") == depth - 9


def test_the_budget_degrades_a_chain_of_divisions() -> None:
    check(
        "1/(2/(3/(4/(5/(6/(7/(8/(9/(a/b)))))))))",
        # The built-up fraction operand of the linear `/` takes no fences: the
        # bar already delimits it.
        """
                   2
        1/───────────────────
                   3
           ─────────────────
                   4
            ───────────────
                   5
             ─────────────
                   6
              ───────────
                   7
               ─────────
                   8
                ───────
                   9
                 ─────
                   a
                  ───
                   b
        """,
        DisplayOptions(height=19),
    )


def test_a_matrix_that_busts_the_budget_falls_back_to_the_flat_form() -> None:
    check(
        "[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11]]",
        """
        [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]
        """,
        DisplayOptions(height=19),
    )


def test_the_outermost_division_degrades_first() -> None:
    check(
        continued_fraction(10),
        # The multi-row operand takes fences because it is a sum, not a
        # fraction.
        """
              ┌                             1                         ┐
        1 + 1/│1 + ───────────────────────────────────────────────────│
              │                               1                       │
              │     1 + ───────────────────────────────────────────── │
              │                                 1                     │
              │          1 + ───────────────────────────────────────  │
              │                                   1                   │
              │               1 + ─────────────────────────────────   │
              │                                     1                 │
              │                    1 + ───────────────────────────    │
              │                                       1               │
              │                         1 + ─────────────────────     │
              │                                         1             │
              │                              1 + ───────────────      │
              │                                           1           │
              │                                   1 + ─────────       │
              │                                             1         │
              │                                        1 + ───        │
              └                                             x         ┘
        """,
        DisplayOptions(height=19),
    )


def test_two_levels_degrade_when_one_is_not_enough() -> None:
    check(
        continued_fraction(11),
        """
              ┌      ┌                             1                         ┐┐
        1 + 1/│1 + 1/│1 + ───────────────────────────────────────────────────││
              │      │                               1                       ││
              │      │     1 + ───────────────────────────────────────────── ││
              │      │                                 1                     ││
              │      │          1 + ───────────────────────────────────────  ││
              │      │                                   1                   ││
              │      │               1 + ─────────────────────────────────   ││
              │      │                                     1                 ││
              │      │                    1 + ───────────────────────────    ││
              │      │                                       1               ││
              │      │                         1 + ─────────────────────     ││
              │      │                                         1             ││
              │      │                              1 + ───────────────      ││
              │      │                                           1           ││
              │      │                                   1 + ─────────       ││
              │      │                                             1         ││
              │      │                                        1 + ───        ││
              └      └                                             x         ┘┘
        """,
        DisplayOptions(height=19),
    )


# -- fencing a special form -------------------------------------------------

# Every block in this section was checked against the original.
#
# A form that ends in its body is fenced by what would be read into that body:
# an operator drawn to its right that binds at least as tightly. `Σ k + 1` is a
# sum of `k + 1`, so a `+` after one needs fences and a `=` does not; the body
# of `d/dx` stops at a product, so `d/dx x + 1` needs none where `d/dx x · 2`
# does. Nothing about this is a precedence: a sum is fenced left of a `·` and
# bare right of one. A comma, a closing bracket and an integral's `dx` are
# terminators rather than operators, and fence nothing.

FORM_FENCES = [
    # An integral is closed off by its own `dx`, so it is never fenced.
    (
        "INT(x^2,x)+1",
        """
        ⌠  2
        ⌡ x  dx + 1
        """,
    ),
    (
        "2 INT(x^2,x)",
        """
          ⌠  2
        2·⌡ x  dx
        """,
    ),
    (
        "INT(x^2/(x+1),x)+1",
        """
        ⌠     2
        │    x
        │ ─────── dx + 1
        ⌡  x + 1
        """,
    ),
    (
        "INT(SUM(k,k,1,n),x)",
        """
        ⌠  n
        │  Σ  k dx
        ⌡ k=1
        """,
    ),
    # A sum, a product and a limit end in their body.
    (
        "SUM(k,k,1,n)+1",
        """
        ┌ n   ┐
        │ Σ  k│ + 1
        └k=1  ┘
        """,
    ),
    (
        "PRODUCT(k,k,1,n)+1",
        """
        ┌ n   ┐
        │ Π  k│ + 1
        └k=1  ┘
        """,
    ),
    (
        "SUM(k,k,1,n)*2",
        """
        ┌ n   ┐
        │ Σ  k│·2
        └k=1  ┘
        """,
    ),
    (
        "SUM(k,k,1,n)^2",
        """
        ┌ n   ┐2
        │ Σ  k│
        └k=1  ┘
        """,
    ),
    (
        # A relation is looser than the body, so it is read correctly as it is.
        "SUM(k,k,1,n)=1",
        """
         n
         Σ  k = 1
        k=1
        """,
    ),
    (
        # Nothing is drawn to its right.
        "2 SUM(k,k,1,n)",
        """
           n
        2· Σ  k
          k=1
        """,
    ),
    (
        # Nor here: the `+ 1` follows the product, not the sum.
        "2 SUM(k,k,1,n)+1",
        """
           n
        2· Σ  k + 1
          k=1
        """,
    ),
    (
        "1-SUM(k,k,1,n)",
        """
             n
        1 -  Σ  k
            k=1
        """,
    ),
    (
        "-SUM(k,k,1,n)",
        """
           n
        -  Σ  k
          k=1
        """,
    ),
    (
        # The bar closes the numerator off, as it does for every operand.
        "SUM(k,k,1,n)/2",
        """
          n
          Σ  k
         k=1
        ───────
           2
        """,
    ),
    (
        # A comma terminates the body rather than being read into it.
        "[SUM(k,k,1,n),2]",
        """
        ┌ n      ┐
        │ Σ  k, 2│
        └k=1     ┘
        """,
    ),
    (
        # So does the call's own closing parenthesis.
        "SIN(SUM(k,k,1,n))",
        """
           ┌ n   ┐
        SIN│ Σ  k│
           └k=1  ┘
        """,
    ),
    (
        # The radical takes an operand as tight as a power's, so it fences one.
        "SQRT(SUM(k,k,1,n))",
        """
         ┌ n   ┐
        √│ Σ  k│
         └k=1  ┘
        """,
    ),
    (
        "SUM(SUM(k,k,1,n),j,1,m)",
        """
         m   n
         Σ   Σ  k
        j=1 k=1
        """,
    ),
    (
        "LIM(x,x,0)+1",
        """
        (lim x) + 1
         x→0
        """,
    ),
    (
        "LIM(x,x,0)*2",
        """
        (lim x)·2
         x→0
        """,
    ),
    (
        "2 LIM(x,x,0)",
        """
        2·lim x
          x→0
        """,
    ),
    # `d/dx` reads only as far as a product, and its fences are there to carry
    # the order: the two-argument form goes without them entirely.
    (
        "DIF(x,x)",
        """
        d
        ── x
        dx
        """,
    ),
    (
        "DIF(x,x)+1",
        """
        d
        ── x + 1
        dx
        """,
    ),
    (
        "DIF(x,x)*2",
        """
        ┌d   ┐
        │── x│·2
        └dx  ┘
        """,
    ),
    (
        "2 DIF(x,x)",
        """
          d
        2·── x
          dx
        """,
    ),
    (
        "DIF(x,x)/2",
        """
         d
         ── x
         dx
        ──────
           2
        """,
    ),
    (
        "DIF(x^3,x,2)+1",
        """
        ┌d ┐2  3
        │──│  x  + 1
        └dx┘
        """,
    ),
]


@pytest.mark.parametrize(
    ("text", "expected"), FORM_FENCES, ids=[text for text, _ in FORM_FENCES]
)
def test_a_form_is_fenced_by_what_would_be_read_into_its_body(
    text: str, expected: str
) -> None:
    check(text, expected)


# -- the unary sign ---------------------------------------------------------

# Also from the original: the sign abuts a leaf and stands off anything
# else, whether or not the operand is more than a row tall.

SIGNED = [
    ("-x-(-y)", "-x - -y"),
    ("3-(-2)", "3 - -2"),
    ("3+-2", "3 + -2"),
    ("-#e", "-ê"),
    ("-?", "-?"),
    ('-"s"', '-"s"'),
    ("-SIN(x)", "- SIN(x)"),
    ("-SQRT(x)", "- √x"),
    ("-[1,2]", "- [1, 2]"),
    ("-ABS(x)", "- │x│"),
    ("-x!", "- x!"),
    ("-x y", "- x·y"),
]


@pytest.mark.parametrize(("text", "expected"), SIGNED)
def test_a_sign_abuts_a_leaf_and_stands_off_anything_else(
    text: str, expected: str
) -> None:
    assert rendered(text) == [expected]


def test_a_sign_stands_off_a_raised_or_lowered_operand() -> None:
    check(
        "-x^2",
        """
           2
        - x
        """,
    )
    check(
        "-x SUB 1",
        """
        - x
           1
        """,
    )


def test_a_sign_over_a_quotient_stands_beside_the_bar() -> None:
    """Where the sign is drawn follows the tree, and nothing compensates.

    `-x/y` is `-(x/y)`, whose operand is a quotient rather than a leaf, so the
    ordinary rule that a sign stands off anything but a leaf puts it beside
    the bar. `(-x)/y` really does put `-x` over the bar. The original draws
    both this way.
    """
    check(
        "-x/y",
        """
           x
        - ───
           y
        """,
    )
    check(
        "(-x)/y",
        """
         -x
        ────
          y
        """,
    )


# -- the pieces the corpus only exercises indirectly ------------------------


@pytest.mark.parametrize(
    ("field", "width", "offset"),
    [(1, 1, 0), (3, 1, 1), (4, 2, 1), (4, 1, 2), (5, 2, 2)],
)
def test_centering_ties_go_right(field: int, width: int, offset: int) -> None:
    assert boxes.centered(field, width) == offset


def test_a_tie_puts_a_numerator_right_of_centre() -> None:
    # A one-wide numerator over a two-wide denominator: the bar is four wide,
    # so the numerator has three columns of slack and takes two of them.
    check(
        "x/(-a)",
        """
          x
        ────
         -a
        """,
    )


def test_a_two_row_operand_takes_single_character_fences() -> None:
    inner = boxes.Box(3, 1, 0, None, "abc")
    fence = boxes.fenced(inner, glyphs.PARENS)
    assert boxes.paint(_owned(fence)).lines == ("     ", "(abc)")


def test_a_three_row_operand_takes_built_up_fences() -> None:
    inner = boxes.Box(3, 1, 1, None, "abc")
    fence = boxes.fenced(inner, glyphs.PARENS)
    assert boxes.paint(_owned(fence)).lines == ("┌   ┐", "│abc│", "└   ┘")


def _owned(box: boxes.Box) -> boxes.Box:
    """The painter records regions, so the outermost box needs a node."""
    node = parse_expression("x", ParseState()).node
    inside = (boxes.Placed(box, 0, 0),)
    return boxes.Box(box.width, box.above, box.below, node, "", inside)


def test_a_fraction_is_built_up_at_level_zero_and_linear_above_it() -> None:
    assert rendered("a/b") == [" a", "───", " b"]
    assert rendered("x^(a/b)") == [" a/b", "x"]


def test_a_power_stops_raising_at_the_second_script_level() -> None:
    # Level 0 and level 1 raise; level 2 and deeper print `a^b` linearly.
    assert rendered("y^z") == [" z", "y"]
    assert rendered("x^(y^z)") == ["  z", " y", "x"]
    assert rendered("x^(y^(z^w))") == ["  z^w", " y", "x"]


def test_a_subscript_lowers_at_the_first_script_level() -> None:
    check(
        "x^(a SUB b)",
        """
         a
          b
        x
        """,
    )


def test_the_flat_form_of_a_subscript_writes_an_arrow() -> None:
    # `↓` where a superscript too deep to raise writes `^`. The original holds
    # this glyph at 0x19 of its own font.
    check(
        "x^(y^(a SUB b))",
        """
          a↓b
         y
        x
        """,
    )


def test_a_matrix_is_flat_inside_a_script() -> None:
    assert rendered("x^([[1,2],[3,4]])") == [" [[1, 2], [3, 4]]", "x"]


def test_every_name_the_lexer_knows_has_an_output_spelling() -> None:
    """The two inventories are separate tables, so guard the overlap."""
    for name in names.GREEK_VARIABLES:
        assert name in glyphs.NAMES, name
    for name in names.CONSTANTS:
        assert name in glyphs.NAMES, name
