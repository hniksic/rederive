"""Remove and Unremove at the session level: which entries go, and what comes
back.

The two commands are one mechanism seen from both ends, so they are tested
together: what Remove takes out is exactly what Unremove has to be able to put
back, labels and annotations included.

Every rule asserted here was checked against the original.
"""

import pytest

from rederive.model.session import Session
from rederive.syntax import DeriveSyntaxError


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


# -- what Remove takes out ----------------------------------------------------


def test_a_block_goes_and_the_entries_that_stay_keep_their_labels(session):
    authored(session, "x", "y", "z", "w", "v")
    assert session.remove(2, 4) == 3
    assert labels(session) == ["#1: x", "#5: v"]


def test_naming_one_label_twice_removes_that_entry_alone(session):
    authored(session, "x", "y", "z")
    assert session.remove(2, 2) == 1
    assert labels(session) == ["#1: x", "#3: z"]


def test_either_end_of_the_block_may_be_named_first(session):
    authored(session, "x", "y", "z", "w")
    session.remove(3, 2)
    assert labels(session) == ["#1: x", "#4: w"]


def test_a_removed_label_is_not_handed_out_again(session):
    authored(session, "x", "y", "z")
    session.remove(1, 2)
    session.author("q")
    assert labels(session) == ["#3: z", "#4: q"]


def test_the_block_is_physically_contiguous_and_not_numerically(session):
    authored(session, "a", "b", "c", "d", "e")
    session.remove(2, 3)
    session.unremove()
    # The history now reads #1 #4 #5 #2 #3, so #5 and #2 are neighbours.
    assert labels(session) == ["#1: a", "#4: d", "#5: e", "#2: b", "#3: c"]
    session.remove(5, 2)
    assert labels(session) == ["#1: a", "#4: d", "#3: c"]


async def test_what_a_removed_line_defined_is_left_standing(session):
    authored(session, "k := 4", "x")
    session.remove(1, 1)
    # The value outlives the line that gave it, as it does in the original.
    assert (await session.simplify("k + 1")).text == "5"


def test_a_label_that_names_no_entry_removes_nothing(session):
    authored(session, "x", "y")
    with pytest.raises(KeyError):
        session.remove(1, 9)
    assert labels(session) == ["#1: x", "#2: y"]


# -- where the selection lands ------------------------------------------------


def test_the_entry_that_takes_the_blocks_place_is_selected(session):
    authored(session, "x", "y", "z", "w", "v")
    session.remove(2, 3)
    assert session.selected_entry.text == "w"
    assert session.route == ()


def test_removing_the_last_block_selects_the_new_last_entry(session):
    authored(session, "x", "y", "z", "w")
    session.remove(3, 4)
    assert session.selected_entry.text == "y"


def test_removing_everything_leaves_nothing_selected(session):
    authored(session, "x", "y")
    session.remove(1, 2)
    assert session.entries == []
    assert session.selected is None and session.selected_entry is None


# -- what Unremove puts back --------------------------------------------------


def test_the_removed_expressions_come_back_where_they_are_asked_for(session):
    authored(session, "x", "y", "z", "w", "v")
    session.remove(2, 3)
    assert session.unremove(5) == 2
    assert labels(session) == ["#1: x", "#4: w", "#2: y", "#3: z", "#5: v"]


def test_they_go_after_the_last_entry_when_no_entry_is_named(session):
    authored(session, "x", "y", "z")
    session.remove(1, 2)
    session.unremove()
    assert labels(session) == ["#3: z", "#1: x", "#2: y"]


async def test_an_annotation_comes_back_with_its_expression(session):
    session.author("2 + 3")
    await session.simplify("#1")
    assert session.entries[1].annotation == "Simp(#1)"
    session.remove(2, 2)
    session.unremove()
    assert session.entries[-1].annotation == "Simp(#1)"


def test_the_entry_the_block_went_in_front_of_stays_selected(session):
    authored(session, "x", "y", "z", "w")
    session.remove(2, 3)
    session.select_entry(0)
    session.unremove(4)
    assert session.selected_entry.text == "w"
    assert session.route == ()


def test_the_last_restored_entry_is_selected_when_they_go_at_the_end(session):
    authored(session, "x", "y", "z")
    session.remove(1, 2)
    session.unremove()
    assert session.selected_entry.text == "y"


def test_a_label_that_names_no_entry_puts_nothing_back(session):
    authored(session, "x", "y", "z")
    session.remove(1, 1)
    with pytest.raises(KeyError):
        session.unremove(9)
    assert labels(session) == ["#2: y", "#3: z"]


# -- the buffer ---------------------------------------------------------------


def test_the_buffer_holds_only_the_last_removal(session):
    authored(session, "x", "y", "z")
    session.remove(1, 1)
    session.remove(2, 2)
    session.unremove()
    assert labels(session) == ["#3: z", "#2: y"]


def test_unremoving_does_not_empty_the_buffer(session):
    authored(session, "x", "y", "z")
    session.remove(1, 2)
    session.unremove()
    assert session.unremove() == 2


def test_a_label_already_taken_is_given_afresh_on_the_way_back(session):
    authored(session, "x", "y", "z")
    session.remove(1, 2)
    session.author("q")
    session.unremove(3)
    assert labels(session) == ["#1: x", "#2: y", "#3: z", "#4: q"]
    # The second copy cannot have #1 and #2, so it continues the counter.
    session.unremove(3)
    assert labels(session) == [
        "#1: x",
        "#2: y",
        "#5: x",
        "#6: y",
        "#3: z",
        "#4: q",
    ]


async def test_an_engine_command_empties_the_buffer(session):
    authored(session, "x", "2 + 3")
    session.remove(1, 1)
    await session.simplify("#2")
    assert session.removed == []
    assert session.unremove() == 0


def test_authoring_leaves_the_buffer_alone(session):
    authored(session, "x", "y")
    session.remove(1, 1)
    session.author("z")
    assert session.unremove() == 1


async def test_a_command_that_does_not_run_leaves_the_buffer_alone(session):
    authored(session, "x", "y")
    session.remove(1, 1)
    with pytest.raises(DeriveSyntaxError):
        await session.simplify("y +")
    assert session.unremove() == 1
