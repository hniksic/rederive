"""The session's engine commands: what Simplify does to the history.

The engine's own answers are tested in `test_simplify`. What matters here is
the wiring: that a line's definitions reach the next command, that the answer
is appended as an entry like any other, and that simplifying part of an entry
brings the rest of it along.
"""

import pytest

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
