"""Selection and navigation rules, as pure model operations.

The selection is a route into what was drawn, so these tests read it back as
the cells it covers - what would be in inverse video on screen.
"""

import pytest
from sexpr import to_sexpr

from rederive.model.session import Session
from rederive.model.settings import Settings
from rederive.syntax import DeriveSyntaxError


@pytest.fixture
def session():
    session = Session()
    for text in ("x (x + 1)", "x y", "x x x"):
        session.author(text)
    return session


def selected_text(session):
    """The selected rectangle, as its rows with the padding stripped."""
    entry = session.selected_entry
    top, left, height, width = session.selection_rect()
    rows = entry.layout.lines[top : top + height]
    return "\n".join(row[left : left + width].rstrip() for row in rows)


def test_authoring_selects_the_new_entry_as_a_whole():
    session = Session()
    session.author("12.345")
    assert session.entries[0].number == 1
    assert session.selected == 0
    assert session.route == ()
    assert selected_text(session) == "12.345"


def test_label_numbers_only_increase():
    session = Session()
    session.author("a")
    session.author("b")
    assert [entry.number for entry in session.entries] == [1, 2]


def test_a_line_that_does_not_parse_is_not_entered():
    session = Session()
    with pytest.raises(DeriveSyntaxError) as caught:
        session.author("12.34.5")
    assert session.entries == []
    assert session.selected is None
    # The manual's own example: the cursor lands on the second decimal point.
    assert caught.value.offset == 5


def test_empty_history_has_no_selection():
    session = Session()
    assert session.selection_rect() is None
    assert session.selected_node is None
    assert not session.move_up()
    assert not session.move_right()


def test_worked_example(session):
    """The walkthrough from the milestone brief, key by key."""
    assert selected_text(session) == "x·x·x"
    session.move_up()
    session.move_up()
    assert session.selected_entry.number == 1
    assert selected_text(session) == "x·(x + 1)"
    session.move_right()
    assert selected_text(session) == "x"
    session.move_right()
    # The parentheses are the product's, not the sum's, so they stay outside.
    assert selected_text(session) == "x + 1"
    session.move_down()
    assert selected_text(session) == "x"
    session.move_up()
    assert selected_text(session) == "x + 1"
    session.move_up()
    assert selected_text(session) == "x·(x + 1)"


def test_the_selection_names_the_subexpression_it_covers(session):
    """`selected_node` is how an operation gets back to what is selected."""
    session.move_first_entry()
    assert to_sexpr(session.selected_node) == "(juxt x (+ x 1))"
    session.move_right()
    session.move_right()
    assert to_sexpr(session.selected_node) == "(+ x 1)"


def test_entry_movement_stops_at_the_ends(session):
    assert not session.move_down()
    session.move_first_entry()
    assert session.selected == 0
    assert not session.move_up()
    session.move_last_entry()
    assert session.selected == 2


def test_sibling_movement_stops_at_the_ends(session):
    session.move_first_entry()
    session.move_right()
    assert not session.move_left()
    session.move_right()
    assert not session.move_right()


def test_home_and_end_move_between_siblings(session):
    session.author("a + b + c")
    session.move_right()
    assert selected_text(session) == "a"
    assert session.move_last_sibling()
    assert selected_text(session) == "c"
    assert session.move_first_sibling()
    assert selected_text(session) == "a"
    # A whole expression has no siblings to move among.
    session.move_up()
    assert not session.move_first_sibling()
    assert not session.move_last_sibling()


def test_atoms_have_nothing_to_descend_into(session):
    session.author("x")
    assert not session.move_right()
    session.author("x y")
    session.move_right()
    assert not session.move_down()


def test_a_run_offers_its_terms_and_not_its_pairs():
    """What the cursor visits is decided by the render, not by the parse."""
    session = Session()
    session.author("a + b - c")
    session.move_right()
    assert selected_text(session) == "a"
    session.move_right()
    assert selected_text(session) == "b"
    session.move_right()
    # The sign belongs to the run, so the third term is `c` and not `- c`.
    assert selected_text(session) == "c"
    assert not session.move_right()


def test_a_call_offers_its_arguments_and_never_its_name():
    session = Session()
    session.author("SIN(x + 1)")
    session.move_right()
    assert selected_text(session) == "x + 1"
    assert not session.move_right()


def test_a_selection_is_the_rectangle_a_subexpression_covers():
    session = Session()
    session.author("a + b/c + d")
    session.move_right()
    session.move_right()
    assert session.selection_rect() == (0, 4, 3, 3)
    assert selected_text(session) == " b\n───\n c"


# -- what the settings reach ------------------------------------------------


def test_the_input_radix_reaches_the_parser():
    session = Session()
    session.author("InputBase := Hexadecimal")
    assert session.settings["InputBase"] == "Hexadecimal"
    assert session.author("1A").layout.lines == ("26",)


def test_the_output_radix_reaches_the_renderer():
    session = Session()
    session.author("OutputBase := Hexadecimal")
    assert session.author("26").layout.lines == ("1A",)


def test_the_times_operator_and_the_format_reach_the_renderer():
    settings = Settings()
    session = Session(settings)
    settings.apply({"TimesOperator": "Asterisk", "DisplayFormat": "Compressed"})
    assert session.author("a b + c").layout.lines == ("a*b+c",)


def test_an_entry_keeps_the_render_it_was_authored_with():
    """The original never redraws an expression already on screen."""
    session = Session()
    first = session.author("a b")
    session.author("TimesOperator := Asterisk")
    second = session.author("a b")
    assert first.layout.lines == ("a·b",)
    assert second.layout.lines == ("a*b",)


def test_a_definition_reaches_the_lines_that_follow():
    session = Session()
    session.author("F(mx, mf) := mx + mf")
    # Without the definition, Character mode would read this as a product.
    assert session.author("mx + mf").layout.lines == ("mx + mf",)


def test_an_authored_assignment_changes_the_setting_a_dialog_changes():
    session = Session()
    session.author("InputMode := Word")
    assert session.settings["InputMode"] == "Word"
    assert session.author("xyz").layout.lines == ("xyz",)


def test_a_setting_takes_effect_before_its_own_line_is_drawn():
    session = Session()
    entry = session.author("DisplayFormat := Compressed")
    assert entry.layout.lines == ("DisplayFormat:=Compressed",)


def test_a_value_no_field_takes_changes_nothing():
    """A refused value must not reach the parse state either."""
    session = Session()
    session.author("InputBase := 1")
    assert session.settings["InputBase"] == "Decimal"
    assert session.state.input_base == 10
    assert session.author("19").layout.lines == ("19",)
