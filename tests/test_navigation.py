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
    assert to_sexpr(session.selected_node) == "(* x (+ x 1))"
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


# -- stepping in --------------------------------------------------------------
#
# Every rule asserted here was checked against the original, key by key.


def test_either_horizontal_arrow_steps_into_a_whole_expression():
    """A whole expression has nothing to either side, so both arrows go in."""
    session = Session()
    session.author("(a + b) (c + d)")
    assert session.move_left()
    assert selected_text(session) == "a + b"
    session.move_up()
    assert session.move_right()
    assert selected_text(session) == "a + b"


def test_stepping_back_in_returns_to_the_operand_you_left():
    session = Session()
    session.author("(a + b) (c + d)")
    session.move_right()
    session.move_right()
    assert selected_text(session) == "c + d"
    session.move_up()
    assert selected_text(session) == "(a + b)·(c + d)"
    session.move_right()
    assert selected_text(session) == "c + d"
    # Whichever arrow steps in, and however deep it was left.
    session.move_down()
    session.move_right()
    assert selected_text(session) == "d"
    session.move_up()
    session.move_up()
    session.move_left()
    assert selected_text(session) == "c + d"
    session.move_down()
    assert selected_text(session) == "d"


def test_each_operand_remembers_its_own_place():
    """The place is the node's own, so a detour through a sibling keeps it."""
    session = Session()
    session.author("(p + q) (r + s)")
    session.move_right()
    session.move_down()
    session.move_right()
    assert selected_text(session) == "q"
    session.move_up()
    session.move_right()
    session.move_down()
    # An operand not stepped into before starts at the first, and not wherever
    # its sibling happens to stand.
    assert selected_text(session) == "r"
    session.move_up()
    session.move_left()
    session.move_down()
    assert selected_text(session) == "q"


def test_end_leaves_a_place_to_return_to(session):
    session.author("a + b + c")
    session.move_right()
    session.move_last_sibling()
    session.move_up()
    session.move_right()
    assert selected_text(session) == "c"


def test_only_one_expression_remembers_at_a_time():
    """Stepping into another expression is what takes the memory over.

    Selecting one as a whole on the way past does not, which is what makes
    looking down the history and coming back land where it left off.
    """
    session = Session()
    session.author("(a + b) (c + d)")
    session.author("(p + q) (r + s)")
    session.move_up()
    session.move_right()
    session.move_right()
    assert selected_text(session) == "c + d"
    session.move_up()
    # Down to the second expression and back, without stepping into it.
    session.move_down()
    session.move_up()
    session.move_right()
    assert selected_text(session) == "c + d"
    # Now step into it, and the first expression has forgotten.
    session.move_up()
    session.move_down()
    session.move_right()
    assert selected_text(session) == "p + q"
    session.move_up()
    session.move_up()
    session.move_right()
    assert selected_text(session) == "a + b"


def test_a_new_expression_starts_at_its_first_operand():
    session = Session()
    session.author("(a + b) (c + d)")
    session.move_right()
    session.move_right()
    assert selected_text(session) == "c + d"
    session.author("(p + q) (r + s)")
    session.move_right()
    assert selected_text(session) == "p + q"


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


# -- paging -------------------------------------------------------------------
#
# A page is however much of the history is on screen, so these tests say how
# tall the pane is: 20 rows, which is the original's. The landings were checked
# against it, key by key.


def paged(session):
    return session.selected_entry.number


def test_a_page_is_the_expressions_a_pane_holds():
    session = Session()
    for number in range(1, 31):
        session.author(str(number))
    session.jump(20)
    # Ten one-line expressions fit, blank lines between them counted, so a
    # page keeps the expression it started from in view at the far edge.
    assert session.move_page_up(20) and paged(session) == 11
    assert session.move_page_up(20) and paged(session) == 2


def test_a_page_of_built_up_expressions_is_fewer_of_them():
    session = Session()
    for number in range(1, 21):
        session.author(f"{number}/2")
    session.jump(20)
    # Three rows each, so five to a pane rather than ten.
    assert session.move_page_up(20) and paged(session) == 16
    assert session.move_page_up(20) and paged(session) == 12
    assert session.move_page_down(20) and paged(session) == 16


def test_paging_down_takes_the_bottom_of_the_pane_before_it_scrolls():
    session = Session()
    for number in range(1, 31):
        session.author(str(number))
    session.jump(2)
    # The pane cannot scroll past the first expression, so the highlight is
    # not at its bottom edge: the first page down is to that edge.
    assert session.move_page_down(20) and paged(session) == 10
    assert session.move_page_down(20) and paged(session) == 19
    assert session.move_page_down(20) and paged(session) == 28


def test_paging_stops_at_the_ends(session):
    session.move_first_entry()
    assert not session.move_page_up(20)
    session.move_last_entry()
    assert not session.move_page_down(20)


def test_an_expression_too_tall_for_the_pane_is_a_page_of_its_own():
    session = Session()
    session.author("x")
    session.author("[[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]")
    session.author("y")
    assert session.entries[1].height > 20
    session.jump(2)
    assert session.move_page_up(20) and paged(session) == 1
    session.jump(2)
    assert session.move_page_down(20) and paged(session) == 3


def test_a_page_selects_the_expression_it_lands_on_whole(session):
    session.move_first_entry()
    session.move_right()
    assert selected_text(session) == "x"
    session.move_page_down(20)
    assert session.route == ()


def test_a_page_that_cannot_move_keeps_the_subexpression(session):
    """A page with nowhere to go moves nothing at all, the route included."""
    session.move_first_entry()
    session.move_right()
    assert selected_text(session) == "x"
    assert not session.move_page_up(20)
    assert selected_text(session) == "x"


def test_an_empty_history_has_nothing_to_page():
    empty = Session()
    assert not empty.move_page_up(20)
    assert not empty.move_page_down(20)


# -- Jump ---------------------------------------------------------------------
#
# Every rule asserted here was checked against the original.


def test_jumping_selects_the_entry_a_label_names(session):
    assert session.jump(1)
    assert session.selected_entry.number == 1
    assert selected_text(session) == "x·(x + 1)"


def test_a_label_no_entry_carries_lands_on_the_one_above_it(session):
    session.remove(2, 2)
    assert session.jump(2)
    assert session.selected_entry.number == 3


def test_zero_lands_on_the_first_entry(session):
    session.remove(1, 1)
    assert session.jump(0)
    assert session.selected_entry.number == 2


def test_a_label_past_the_last_one_names_nothing(session):
    session.select_entry(0)
    assert not session.jump(4)
    assert session.selected_entry.number == 1


def test_it_is_the_label_that_is_looked_up_and_not_the_position(session):
    session.remove(1, 1)
    session.unremove()
    # The history now reads #2 #3 #1, which is not the order its labels read.
    assert [entry.number for entry in session.entries] == [2, 3, 1]
    assert session.jump(1)
    assert session.selected == 2


def test_jumping_to_the_entry_you_are_inside_of_keeps_the_subexpression(session):
    session.move_first_entry()
    session.move_right()
    assert selected_text(session) == "x"
    assert session.jump(1)
    assert selected_text(session) == "x"


def test_jumping_to_any_other_entry_selects_it_whole(session):
    session.move_first_entry()
    session.move_right()
    session.jump(2)
    session.jump(1)
    assert selected_text(session) == "x·(x + 1)"


def test_an_empty_history_has_nothing_to_jump_to():
    assert not Session().jump(1)


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


def test_the_notation_digits_reach_the_renderer():
    session = Session()
    assert session.author("0.123456789").layout.lines == ("0.123456",)
    session.author("NotationDigits := 9")
    assert session.author("0.123456789").layout.lines == ("0.123456789",)


def test_showing_fewer_digits_is_not_keeping_fewer():
    """A cut render says nothing about the number: the value is the whole of it."""
    session = Session()
    assert session.author("0.10000000000000001").layout.lines == ("0.100000",)
    session.author("0.10000000000000001 - 1/10")
    assert session.simplify("#2").text == "1/100000000000000000"


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
