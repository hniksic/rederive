"""The session's engine commands: what Simplify does to the history.

The engine's own answers are tested in `test_simplify`. What matters here is
the wiring: that a line's definitions reach the next command, that the answer
is appended as an entry like any other, and that simplifying part of an entry
brings the rest of it along.
"""

import pytest

from rederive import engine
from rederive.model import building
from rederive.model.session import Session
from rederive.syntax import DeriveSyntaxError


@pytest.fixture
def session():
    return Session()


def texts(session):
    return [entry.text for entry in session.entries]


def part(session, *route):
    """Highlight a subexpression of the selected entry, by successive steps."""
    for step in route:
        getattr(session, f"move_{step}")()
    return session.selected_node


# -- the answer as an entry --------------------------------------------------


def test_the_answer_is_appended_and_selected_as_a_whole(session):
    session.author("2 (8 + 7) / 3^2")
    answer = session.simplify("#1")
    assert texts(session) == ["2 (8 + 7) / 3^2", "10/3"]
    assert answer.number == 2
    assert session.selected == 1 and session.route == ()
    assert answer.annotation == "Simp(#1)"


def test_a_typed_expression_is_the_users_own(session):
    assert session.simplify("2 + 3").text == "5"
    assert session.entries[0].annotation == "Simp(User)"


def test_a_label_stands_for_the_entry_it_names(session):
    session.author("2^3")
    assert session.simplify("#1 + 1").text == "9"
    # Written into an expression rather than named alone, so it is the user's.
    assert session.entries[-1].annotation == "Simp(User)"


def test_a_line_that_does_not_parse_appends_nothing(session):
    session.author("x")
    with pytest.raises(DeriveSyntaxError):
        session.simplify("x +")
    assert texts(session) == ["x"]


def test_an_answer_can_be_simplified_again(session):
    session.author("2 (8 + 7) / 3^2")
    session.simplify("#1")
    assert session.simplify("#2").text == "10/3"
    assert session.entries[-1].annotation == "Simp(#2)"


# -- part of an entry --------------------------------------------------------


def test_simplifying_part_of_an_entry_copies_the_rest_of_it(session):
    session.author("2 (8 + 7) / 3^2")
    assert part(session, "right", "right").value == "^"
    answer = session.simplify("#1")
    assert answer.text == "2 (8 + 7) / 9"
    # The quote is what says that only part of the entry was simplified.
    assert answer.annotation == "Simp(#1')"


def test_a_spliced_answer_is_fenced_where_the_line_needs_it(session):
    session.author("x := y + 1")
    session.author("2 x")
    part(session, "right", "right")
    assert session.simplify("#2").text == "2 (y + 1)"


def test_a_spliced_line_reads_back_as_the_text_it_shows(session):
    """The new entry's spans index its own text, as an authored line's do."""
    session.author("(8 + 7) (x + 1)")
    part(session, "right")
    answer = session.simplify("#1")
    # The fences the line was written with stay, and are drawn as precedence
    # asks rather than as they were typed.
    assert answer.text == "(15) (x + 1)"
    assert answer.layout.lines == ("15·(x + 1)",)
    session.move_right()
    assert answer.text[session.selected_node.start : session.selected_node.end] == "15"


def test_the_whole_entry_is_simplified_when_the_whole_entry_is_selected(session):
    session.author("2 (8 + 7) / 3^2")
    part(session, "right", "right", "up", "up")
    assert session.route == ()
    assert session.simplify("#1").text == "10/3"


def test_another_entry_is_simplified_whole(session):
    session.author("2 (8 + 7) / 3^2")
    session.author("x + x")
    session.select_entry(0)
    part(session, "right", "right")
    # The highlight is in #1, so naming #2 asks for the whole of #2.
    answer = session.simplify("#2")
    assert answer.text == "2*x"
    assert answer.annotation == "Simp(#2)"


# -- what the lines have defined ---------------------------------------------


def test_an_assignment_reaches_the_next_command(session):
    session.author("x := 5")
    session.author("2 x + 1")
    assert session.simplify("#2").text == "11"


def test_an_emptied_assignment_is_forgotten(session):
    session.author("x := 5")
    session.author("x :=")
    session.author("2 x")
    assert session.simplify("#3").text == "2*x"


def test_a_function_definition_reaches_the_next_command(session):
    session.author("f(y) := y^2 - 1")
    session.author("f(3)")
    assert session.simplify("#2").text == "8"


def test_every_definition_on_a_line_is_recorded(session):
    session.author("[a := 2, b := 3]")
    session.author("a b")
    assert session.simplify("#2").text == "6"


def test_a_setting_is_not_a_variable(session):
    session.author("Notation := Mixed")
    assert session.assignments == {}
    assert session.settings["Notation"] == "Mixed"


def test_a_declared_domain_is_what_lets_a_rewrite_fire(session):
    session.author("z :epsilon Real (0, inf)")
    session.author("SQRT(z^2)")
    assert session.simplify("#2").text == "z"


def test_nothing_is_guessed_about_a_variable_nobody_declared(session):
    session.author("SQRT(w^2)")
    assert session.simplify("#1").text == "ABS(w)"


def test_the_precision_setting_reaches_the_command(session):
    session.author("1/3")
    session.settings.assign("Precision", "Approximate")
    assert session.simplify("#1").text.startswith("0.333333")


# -- Factor ------------------------------------------------------------------
#
# The engine's own answers are tested in `test_factor`. What matters here is
# that Factor reaches the history the way Simplify does, and that the two
# questions it asks first are answered from the expression it would act on.


def test_a_factored_answer_is_appended_and_annotated(session):
    session.author("x^2 - 4")
    answer = session.factor("#1")
    assert texts(session) == ["x^2 - 4", "(x - 2)*(x + 2)"]
    assert answer.annotation == "Fctr(#1)"


def test_a_typed_expression_is_factored_as_the_users_own(session):
    assert session.factor("x^2 - 9").text == "(x - 3)*(x + 3)"
    assert session.entries[0].annotation == "Fctr(User)"


def test_factoring_part_of_an_entry_copies_the_rest_of_it(session):
    session.author("(x^2 - 1) + SIN(z)")
    part(session, "right")
    answer = session.factor("#1")
    # The splice is fenced and the line's own fences stay, so the text carries
    # a pair more than it needs. What is drawn comes from the tree, where a
    # fence is a matter of precedence, so the extra pair shows up nowhere.
    assert answer.text == "(((x - 1)*(x + 1))) + SIN(z)"
    assert answer.layout.lines == ("(x - 1)·(x + 1) + SIN(z)",)
    assert answer.annotation == "Fctr(#1')"


def test_the_amount_reaches_the_command(session):
    session.author("2 x^3 - 12 x^2 + 18 x")
    assert session.factor("#1", engine.Amount.TRIVIAL).text == "2*x*(x^2 - 6*x + 9)"
    assert session.factor("#1", engine.Amount.SQUAREFREE).text == "2*x*(x - 3)^2"


def test_the_factorization_variables_reach_the_command(session):
    session.author("x^2 y^2 - x^2 - y^4 + y^2")
    whole = session.factor("#1", engine.Amount.RATIONAL, ("x", "y"))
    assert whole.text == "(x - y)*(x + y)*(y - 1)*(y + 1)"
    about_x = session.factor("#1", engine.Amount.RATIONAL, ("x",))
    assert about_x.text == "(x - y)*(x + y)*(y^2 - 1)"


def test_a_line_that_does_not_parse_factors_nothing(session):
    session.author("x")
    with pytest.raises(DeriveSyntaxError):
        session.factor("x +")
    assert texts(session) == ["x"]


def test_an_assignment_reaches_the_factoring(session):
    session.author("k := 4")
    session.author("x^2 - k")
    assert session.factor("#2").text == "(x - 2)*(x + 2)"


# -- what Factor asks before it factors ---------------------------------------


def test_the_variables_on_offer_are_most_main_first(session):
    session.author("y^2 - x^2")
    assert session.variables("#1") == ("x", "y")
    session.author("b a - c^2")
    assert session.variables("#2") == ("a", "b", "c")
    # The order list is x, y, z, and a variable on it outranks one that is not.
    session.author("z^2 - a^2")
    assert session.variables("#3") == ("z", "a")


def test_the_variables_on_offer_come_from_the_highlighted_part(session):
    """Only the subexpression is factored, so only its variables are offered."""
    session.author("(x^2 - 1) + SIN(z)")
    assert session.variables("#1") == ("x", "z")
    part(session, "right")
    assert session.variables("#1") == ("x",)


def test_an_assigned_name_is_no_longer_a_variable(session):
    session.author("k := 4")
    session.author("x^2 - k")
    assert session.variables("#2") == ("x",)


def test_a_number_is_recognised_before_anything_is_asked(session):
    session.author("1234567890")
    session.author("1234567890/49")
    session.author("-12")
    session.author("10!")
    session.author("x^2 - 4")
    assert [session.decomposes(f"#{n}") for n in (1, 2, 3, 4, 5)] == [
        True,
        True,
        True,
        # A factorial has a number for a value but is not written as one, and
        # the original asks for an amount for it.
        False,
        False,
    ]


# -- Expand -------------------------------------------------------------------


def test_the_answer_is_appended_as_an_expansion(session):
    session.author("2 x (x - 3)^2")
    answer = session.expand("#1")
    assert texts(session) == ["2 x (x - 3)^2", "2*x^3 - 12*x^2 + 18*x"]
    assert answer.annotation == "Expd(#1)"


def test_expanding_part_of_an_entry_copies_the_rest_of_it(session):
    """The manual's own example: expanding only the square inside the product
    leaves the product standing around it."""
    session.author("2 x (x - 3)^2")
    part(session, "right", "last_sibling")
    answer = session.expand("#1")
    assert answer.layout.lines == ("      2           ", "2·x·(x  - 6·x + 9)")
    assert answer.annotation == "Expd(#1')"


def test_the_expansion_variables_reach_the_command(session):
    session.author("(x + 2 y + 1)^3")
    about_x = session.expand("#1", engine.Amount.RATIONAL, ("x",))
    assert about_x.text == "x^3 + 3*x^2*(2*y + 1) + 3*x*(2*y + 1)^2 + (2*y + 1)^3"


def test_the_amount_reaches_the_expansion(session):
    session.author("1/(x^2 - 1)")
    assert session.expand("#1", engine.Amount.TRIVIAL).text == "1/(x^2 - 1)"
    assert session.expand("#1").text == "1/(2*(x - 1)) - 1/(2*(x + 1))"


def test_an_assignment_reaches_the_expansion(session):
    session.author("k := 3")
    session.author("(x + k)^2")
    assert session.expand("#2").text == "x^2 + 6*x + 9"


def test_a_ratio_is_recognised_before_an_amount_is_asked_for(session):
    session.author("1/(x^2 - 1)")
    session.author("7/12")
    session.author("x^-1")
    session.author("2/x + 1/x")
    assert [session.written_as_ratio(f"#{n}") for n in (1, 2, 3, 4)] == [
        True,
        True,
        # A power, not a quotient, however it is drawn.
        False,
        # Its value has a denominator; the expression as written has none, and
        # the original goes by how it is written.
        False,
    ]


# -- approX -------------------------------------------------------------------
#
# Simplify with the precision temporarily approximate, and nothing else: the
# command asks for an expression and no more.


def test_an_approximated_answer_is_appended_and_annotated(session):
    session.author("pi")
    answer = session.approx("#1")
    assert texts(session) == ["pi", "3.14159"]
    assert answer.annotation == "Approx(#1)"


def test_a_typed_expression_is_approximated_as_the_users_own(session):
    assert session.approx("SQRT(3)").text == "1.73205"
    assert session.entries[0].annotation == "Approx(User)"


def test_approximating_part_of_an_entry_copies_the_rest_of_it(session):
    session.author("SQRT(2) + x^2")
    part(session, "right")
    answer = session.approx("#1")
    assert answer.layout.lines == ("           2", "1.41421 + x ")
    assert answer.annotation == "Approx(#1')"


def test_the_settings_are_left_where_they_were(session):
    """The precision is approximate for the one command, not from now on."""
    session.author("1/3")
    assert session.approx("#1").text == "0.333333"
    assert session.settings["Precision"] == "Exact"
    assert session.simplify("#1").text == "1/3"


def test_the_precision_digits_setting_reaches_the_command(session):
    session.author("pi")
    session.settings.assign("PrecisionDigits", 12)
    # Twelve digits of pi, cut rather than rounded, which is what the
    # original answers here: `3.14159265358`, not `...359`.
    assert session.approx("#1").text == "3.14159265358"


def test_what_has_no_number_in_it_is_simplified_and_no_more(session):
    session.author("x + x")
    assert session.approx("#1").text == "2*x"


def test_a_line_that_does_not_parse_approximates_nothing(session):
    session.author("x")
    with pytest.raises(DeriveSyntaxError):
        session.approx("x +")
    assert texts(session) == ["x"]


def test_an_assignment_reaches_the_approximation(session):
    session.author("k := 2")
    session.author("SQRT(k)")
    assert session.approx("#2").text == "1.41421"


# -- Build --------------------------------------------------------------------


def built(session, word, *requests):
    """Build one operator over the expressions `requests` names."""
    operator = building.operator(word)
    resolved = [session.named_target(request) for request in requests]
    node = operator.build(*(node for node, _ in resolved))
    return session.build(node, operator.annotate(*(name for _, name in resolved)))


def test_a_built_expression_is_appended_unsimplified(session):
    session.author("2 + 3")
    answer = built(session, "*", "#1", "#1")
    assert answer.text == "(2+3)*(2+3)"
    assert answer.annotation == "#1*#1"


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("+", "x^2*y+x^2*y"),
        ("-", "x^2*y-x^2*y"),
        ("=", "x^2*y=x^2*y"),
        ("Minus", "-x^2*y"),
        ("Recip", "1/(x^2*y)"),
        ("Sin", "SIN(x^2*y)"),
        ("Ln", "LN(x^2*y)"),
        ("!", "(x^2*y)!"),
        ("%", "(x^2*y)%"),
        ("`", "(x^2*y)`"),
    ],
    ids=str,
)
def test_each_operator_writes_what_the_original_writes(session, word, expected):
    """The forms the original's own Transfer Save writes."""
    session.author("x^2*y")
    operator = building.operator(word)
    requests = ["#1"] * operator.arity
    assert built(session, word, *requests).text == expected


@pytest.mark.parametrize(
    ("word", "expected"),
    [("+", "#1+#1"), ("Minus", "-(#1)"), ("Recip", "1/(#1)"), ("Sin", "SIN(#1)")],
    ids=str,
)
def test_the_annotation_names_the_operands(session, word, expected):
    session.author("x^2*y")
    operator = building.operator(word)
    assert built(session, word, *["#1"] * operator.arity).annotation == expected


def test_a_typed_operand_is_the_users_own(session):
    session.author("x")
    assert built(session, "+", "#1", "2 + 3").annotation == "#1+User"


def test_a_highlighted_part_is_the_operand_and_carries_a_quote(session):
    session.author("SIN(a*x^2) + 5")
    part(session, "right", "down")
    answer = built(session, "Sin", "#1")
    # Extraction, not substitution: what is built is the part alone, and the
    # rest of the entry it came out of is left where it was.
    assert answer.text == "SIN(a*x^2)"
    assert answer.annotation == "SIN(#1')"


def test_building_a_part_alone_extracts_it(session):
    session.author("SIN(a*x^2) + 5")
    part(session, "right", "down")
    node, name = session.named_target("#1")
    assert session.build(node, name).text == "a*x^2"
    assert session.entries[-1].annotation == "#1'"


def test_operators_chain_left_to_right(session):
    """`#1 + #2` then `* #3` is `(#1 + #2)·#3`, and the annotation is flat."""
    for text in ("2", "3", "4"):
        session.author(text)
    first = built(session, "+", "#1", "#2")
    operator = building.operator("*")
    node = operator.build(first.node, session.target("#3"))
    answer = session.build(node, operator.annotate(first.annotation, "#3"))
    assert answer.text == "(2+3)*4"
    assert answer.annotation == "#1+#2*#3"


def test_a_build_taken_with_ctrl_enter_appends_one_entry(session):
    session.author("x^2*y")
    node, name = session.named_target("#1")
    operator = building.operator("+")
    built_node = operator.build(node, node)
    answer = session.build(built_node, operator.annotate(name, name), True)
    assert texts(session) == ["x^2*y", "2*x^2*y"]
    assert answer.annotation == "Simp(#1+#1)"


# -- Calculus -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("head", "prefix", "arguments", "expected"),
    [
        ("DIF", "Dif", ("x",), "DIF(x^2*y,x)"),
        ("DIF", "Dif", ("x", "3"), "DIF(x^2*y,x,3)"),
        ("INT", "Int", ("x",), "INT(x^2*y,x)"),
        ("INT", "Int", ("x", "0", "1"), "INT(x^2*y,x,0,1)"),
        ("LIM", "Lim", ("x", "0", "0"), "LIM(x^2*y,x,0,0)"),
        ("LIM", "Lim", ("x", "2", "-1"), "LIM(x^2*y,x,2,-1)"),
        ("SUM", "Sum", ("x",), "SUM(x^2*y,x)"),
        ("SUM", "Sum", ("x", "1", "n"), "SUM(x^2*y,x,1,n)"),
        ("PRODUCT", "Product", ("x", "1", "n"), "PRODUCT(x^2*y,x,1,n)"),
        ("TAYLOR", "Taylor", ("x", "1", "3"), "TAYLOR(x^2*y,x,1,3)"),
        ("VECTOR", "Vector", ("x", "1", "5"), "VECTOR(x^2*y,x,1,5)"),
        ("VECTOR", "Vector", ("x", "1", "5", "2"), "VECTOR(x^2*y,x,1,5,2)"),
    ],
    ids=str,
)
def test_each_head_is_written_as_the_original_writes_it(
    session, head, prefix, arguments, expected
):
    """The linear forms come from the original's own Transfer Save."""
    session.author("x^2*y")
    assert session.calculus(head, prefix, "#1", arguments).text == expected


def test_nothing_is_computed(session):
    """The point of the command: a Simplify after it is what takes the answer."""
    session.author("x^2*y")
    session.calculus("DIF", "Dif", "#1", ("x",))
    assert texts(session) == ["x^2*y", "DIF(x^2*y,x)"]
    assert session.simplify("#2").text == "2*x*y"


def test_the_annotation_names_the_expression_and_the_variable(session):
    session.author("x^2*y")
    # The order is not in it: `DIF(u, x, 3)` is annotated as the first
    # derivative is.
    assert session.calculus("DIF", "Dif", "#1", ("x", "3")).annotation == "Dif(#1,x)"


def test_a_typed_expression_is_calculated_as_the_users_own(session):
    assert session.calculus("INT", "Int", "SIN(z)", ("z",)).annotation == "Int(User,z)"


def test_a_highlighted_part_is_taken_alone_and_carries_a_quote(session):
    session.author("SIN(a*x^2) + 5")
    part(session, "right", "down")
    answer = session.calculus("DIF", "Dif", "#1", ("x",))
    assert answer.text == "DIF(a*x^2,x)"
    assert answer.annotation == "Dif(#1',x)"


def test_a_calculus_command_taken_with_ctrl_enter_appends_one_entry(session):
    session.author("x^2*y")
    answer = session.calculus("DIF", "Dif", "#1", ("x",), True)
    assert texts(session) == ["x^2*y", "2*x*y"]
    assert answer.annotation == "Simp(Dif(#1,x))"


def test_an_argument_that_does_not_parse_appends_nothing(session):
    session.author("x")
    with pytest.raises(DeriveSyntaxError):
        session.calculus("DIF", "Dif", "#1", ("x", "2 +"))
    assert texts(session) == ["x"]


def test_a_two_sided_limit_is_written_with_the_direction_the_menu_chose(session):
    """`Calculus Limit` writes `0` for Both, and `0` is two-sided."""
    session.author("ABS(x)/x")
    both = session.calculus("LIM", "Lim", "#1", ("x", "0", "0"))
    assert both.text == "LIM(ABS(x)/x,x,0,0)"
    assert session.simplify("#2").text == "±1"
    right = session.calculus("LIM", "Lim", "#1", ("x", "0", "1"))
    assert right.text == "LIM(ABS(x)/x,x,0,1)"
    assert session.simplify("#4").text == "1"


# -- what a Calculus variable field takes -------------------------------------


def test_the_variable_offered_is_the_primary_one(session):
    session.author("SIN(a*x^2)")
    assert session.variables("#1")[0] == "x"


def test_a_bound_variable_is_not_offered(session):
    session.author("INT(x^2*y, x, 0, 1)")
    assert session.variables("#1") == ("y",)


@pytest.mark.parametrize("text", ["x", "k", " x "], ids=str)
def test_a_name_is_a_variable(session, text):
    assert session.is_variable(text)


# `abc` is three names under the Character input mode the session starts in,
# which is a product and so no answer to a question that wants one variable.
@pytest.mark.parametrize(
    "text", ["", "2 + 3", "2", "SIN", "#e", "x +", "abc"], ids=str
)
def test_anything_that_is_not_one_name_is_not(session, text):
    assert not session.is_variable(text)
