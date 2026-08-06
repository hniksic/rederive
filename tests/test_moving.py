"""moVe at the session level: which block goes where, and what stays behind.

Every rule asserted here matches the original.
"""

import pytest

from rederive.model.session import Session


@pytest.fixture
def session():
    return Session()


def labels(session):
    """The history as the work area shows it: each entry's label and its text."""
    return [f"#{entry.number}: {entry.text}" for entry in session.entries]


def authored(session, *texts):
    for text in texts:
        session.author(text)
    return session


# -- where the block lands ----------------------------------------------------


def test_a_block_goes_in_front_of_the_entry_that_was_named(session):
    authored(session, "x", "y", "z", "w", "v")
    assert session.move_block(1, 3, 4) == 2
    assert labels(session) == ["#3: z", "#4: w", "#1: x", "#2: y", "#5: v"]


def test_a_block_moved_forwards_lands_in_front_of_that_entry_too(session):
    authored(session, "x", "y", "z", "w", "v")
    session.move_block(5, 2, 3)
    assert labels(session) == ["#1: x", "#4: w", "#2: y", "#3: z", "#5: v"]


def test_naming_one_label_twice_moves_that_entry_alone(session):
    authored(session, "x", "y", "z")
    assert session.move_block(1, 3, 3) == 1
    assert labels(session) == ["#3: z", "#1: x", "#2: y"]


def test_either_end_of_the_block_may_be_named_first(session):
    authored(session, "x", "y", "z", "w", "v")
    session.move_block(1, 4, 3)
    assert labels(session) == ["#3: z", "#4: w", "#1: x", "#2: y", "#5: v"]


def test_no_destination_sends_the_block_past_the_last_entry(session):
    authored(session, "x", "y", "z")
    assert session.move_block(None, 1, 2) == 2
    assert labels(session) == ["#3: z", "#1: x", "#2: y"]


def test_the_entries_keep_the_labels_they_moved_with(session):
    authored(session, "x", "y", "z")
    session.move_block(1, 2, 2)
    session.author("q")
    # Nothing was renumbered, so the next label goes on counting from three.
    assert labels(session) == ["#2: y", "#1: x", "#3: z", "#4: q"]


def test_the_block_is_physically_contiguous_and_not_numerically(session):
    authored(session, "a", "b", "c", "d", "e")
    session.move_block(1, 4, 5)
    # The history now reads #4 #5 #1 #2 #3, so #5 and #1 are neighbours.
    assert labels(session) == ["#4: d", "#5: e", "#1: a", "#2: b", "#3: c"]
    session.move_block(None, 5, 1)
    assert labels(session) == ["#4: d", "#2: b", "#3: c", "#5: e", "#1: a"]


def test_a_label_that_names_no_entry_moves_nothing(session):
    authored(session, "x", "y")
    with pytest.raises(KeyError):
        session.move_block(9, 1, 1)
    with pytest.raises(KeyError):
        session.move_block(1, 9, 9)
    assert labels(session) == ["#1: x", "#2: y"]


# -- the destination the command does nothing about ---------------------------


def test_a_destination_inside_the_block_moves_nothing(session):
    authored(session, "x", "y", "z", "w", "v")
    session.select_entry(4)
    assert session.move_block(3, 2, 4) == 0
    assert labels(session) == ["#1: x", "#2: y", "#3: z", "#4: w", "#5: v"]
    # Not even the highlight moves: the original does nothing at all with it.
    assert session.selected_entry.text == "v"


def test_moving_an_expression_to_where_it_already_is_does_nothing(session):
    authored(session, "x", "y", "z")
    session.select_entry(0)
    # Which is what Enter alone answers, every field opening on the highlight.
    assert session.move_block(2, 2, 2) == 0
    assert labels(session) == ["#1: x", "#2: y", "#3: z"]
    assert session.selected_entry.text == "x"


def test_the_entry_just_past_the_block_is_a_destination_like_any_other(session):
    authored(session, "x", "y", "z", "w")
    session.select_entry(0)
    # The order comes out unchanged, but the move was made: the highlight went
    # to the destination, which is how the original tells the two apart.
    assert session.move_block(4, 2, 3) == 2
    assert labels(session) == ["#1: x", "#2: y", "#3: z", "#4: w"]
    assert session.selected_entry.text == "w"


# -- what is left where ------------------------------------------------------


def test_the_entry_the_block_went_in_front_of_is_selected(session):
    authored(session, "x", "y", "z", "w")
    session.move_block(1, 3, 4)
    assert session.selected_entry.text == "x"
    assert session.route == ()


def test_the_last_moved_entry_is_selected_when_the_block_goes_at_the_end(session):
    authored(session, "x", "y", "z")
    session.move_block(None, 1, 2)
    assert session.selected_entry.text == "y"


def test_a_whole_expression_moves_however_little_of_it_was_highlighted(session):
    authored(session, "x", "y + z")
    session.select_entry(1)
    session.move_right()
    assert session.route == (0,)
    session.move_block(1, 2, 2)
    assert labels(session) == ["#2: y+z", "#1: x"]
    assert session.route == ()


async def test_an_annotation_travels_with_its_expression(session):
    session.author("2 + 3")
    await session.simplify("#1")
    session.move_block(1, 2, 2)
    assert labels(session) == ["#2: 5", "#1: 2+3"]
    assert session.entries[0].annotation == "Simp(#1)"


async def test_what_a_moved_line_defined_stays_defined(session):
    authored(session, "k := 4", "x")
    session.move_block(None, 1, 1)
    assert (await session.simplify("k + 1")).text == "5"


def test_moving_leaves_the_unremove_buffer_alone(session):
    authored(session, "x", "y", "z")
    session.remove(1, 1)
    session.move_block(2, 3, 3)
    assert labels(session) == ["#3: z", "#2: y"]
    assert session.unremove(2) == 1
    assert labels(session) == ["#3: z", "#1: x", "#2: y"]


def test_renumber_puts_the_labels_a_move_disordered_back_in_sequence(session):
    authored(session, "x", "y", "z")
    session.move_block(1, 3, 3)
    session.renumber()
    assert labels(session) == ["#1: z", "#2: x", "#3: y"]
