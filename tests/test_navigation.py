"""Selection and navigation rules, as pure model operations."""

import pytest

from rederive.model import Session


@pytest.fixture
def session():
    session = Session()
    for text in ("x (x + 1)", "x y", "x x x"):
        session.author(text)
    return session


def selected_text(session):
    entry = session.selected_entry
    start, end = session.selection_span()
    return entry.text[start:end]


def test_authoring_selects_the_new_entry_as_a_whole():
    session = Session()
    session.author("12.345")
    assert session.entries[0].number == 1
    assert session.selected == 0
    assert session.path == ()
    assert selected_text(session) == "12.345"


def test_label_numbers_only_increase():
    session = Session()
    session.author("a")
    session.author("b")
    assert [entry.number for entry in session.entries] == [1, 2]


def test_empty_history_has_no_selection():
    session = Session()
    assert session.selection_span() is None
    assert not session.move_up()
    assert not session.move_right()


def test_worked_example(session):
    """The walkthrough from the milestone brief, key by key."""
    assert selected_text(session) == "x x x"
    session.move_up()
    session.move_up()
    assert session.selected_entry.number == 1
    assert selected_text(session) == "x (x + 1)"
    session.move_right()
    assert selected_text(session) == "x"
    session.move_right()
    assert selected_text(session) == "x + 1"
    session.move_down()
    assert selected_text(session) == "x"
    session.move_up()
    assert selected_text(session) == "x + 1"
    session.move_up()
    assert selected_text(session) == "x (x + 1)"


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
