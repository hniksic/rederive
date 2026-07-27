"""The Manage commands, driven the way a user drives them.

Four of them set something and are dialogs; Renumber, Annotate and Ordering act
on the session instead, and ask for what they need on a line of their own.

The expected band text and message lines are what the original puts on the same
screens, minus the leading blank the pane indents by. Every version shows the
same words on every one of them.

Only the keys peculiar to these screens are exercised here; the rest of what a
dialog does belongs to `test_options` and `test_settings`, since it is the same
machinery underneath.
"""

import pytest
from screen import (
    annotation,
    band,
    entries,
    highlighted,
    highlighted_expression,
    message,
    prompt,
    work_area,
)

from rederive.engine.context import Angle, Branch, Direction, TrigPower
from rederive.model.session import Session
from rederive.syntax import DeriveSyntaxError
from rederive.ui import menu as menus
from rederive.ui.app import RederiveApp


@pytest.fixture
def app():
    return RederiveApp()


@pytest.fixture
def session():
    return Session()


async def manage(pilot, *keys):
    """Open the Manage menu and press on."""
    await pilot.press("m", *keys)


def authored(session, *texts):
    for text in texts:
        session.author(text)
    return session


def labels(session):
    """The history as the work area shows it: each entry's label and its text."""
    return [f"#{entry.number}: {entry.text}" for entry in session.entries]


def annotations(session):
    return [entry.annotation for entry in session.entries]


# -- the menu ----------------------------------------------------------------


async def test_manage_lists_eight_commands_over_two_lines(app):
    async with app.run_test() as pilot:
        await manage(pilot)
        assert band(app) == [
            " MANAGE: Annotate Branch Exponential Logarithm Ordering Renumber Substitute",
            "         Trigonometry",
        ]
        assert highlighted(app) == "Annotate"
        assert message(app) == "Enter option"


def test_every_manage_word_gives_its_own_letter():
    letters = [menus.mnemonic(word) for word in menus.MANAGE.words]
    assert letters == list("abelorst")


# -- the screens -------------------------------------------------------------


async def test_branch_is_a_selection_field_with_no_caption(app):
    async with app.run_test() as pilot:
        await manage(pilot, "b")
        assert band(app) == [" MANAGE BRANCH: Principal Real Any"]
        assert highlighted(app) == "Principal"
        assert message(app) == "Select preferred branch for roots"


async def test_exponential_and_logarithm_ask_for_a_direction(app):
    async with app.run_test() as pilot:
        await manage(pilot, "e")
        assert band(app) == [" MANAGE EXPONENTIAL: Direction: Auto Collect Expand"]
        assert message(app) == "Exponential transformations"
        await pilot.press("escape")
        await pilot.press("l")
        assert band(app) == [" MANAGE LOGARITHM: Direction: Auto Collect Expand"]
        assert message(app) == "Logarithm transformations"


async def test_the_exponential_field_opens_on_auto(app):
    """Section 6.1 of the manual says Collect, which was the 1.x default and
    stale by the later releases: the original's own DERIVE.INI says
    `*EXP-EXPD* |Auto|`."""
    async with app.run_test() as pilot:
        await manage(pilot, "e")
        assert highlighted(app) == "Auto"
        assert app.settings["Exponential"] == "Auto"


async def test_trigonometry_asks_three_questions_over_two_lines(app):
    async with app.run_test() as pilot:
        await manage(pilot, "t")
        assert band(app) == [
            " MANAGE TRIGONOMETRY: Direction: Auto Collect Expand"
            "  Toward:(Auto)Sines Cosines",
            "                      Angle: Degree(Radian)",
        ]
        assert message(app) == "Angle sums & multiple angles"
        await pilot.press("tab")
        assert message(app) == "Trig power transformations"
        assert highlighted(app) == "Auto"
        await pilot.press("tab")
        assert message(app) == "Select angle mode"
        assert highlighted(app) == "Radian"
        # And round to the first field again, committing nothing on the way.
        await pilot.press("tab")
        assert message(app) == "Angle sums & multiple angles"
        assert entries(app) == []


# -- answering ---------------------------------------------------------------


async def test_a_letter_settles_a_one_field_screen_outright(app):
    async with app.run_test() as pilot:
        await manage(pilot, "b", "r")
        assert app.settings["Branch"] == "Real"
        assert entries(app) == ["Branch := Real"]
        assert highlighted(app) == "Author"


async def test_a_letter_on_a_wider_screen_moves_to_the_next_field(app):
    async with app.run_test() as pilot:
        await manage(pilot, "t", "e")
        assert message(app) == "Trig power transformations"
        assert entries(app) == []
        assert band(app)[0].startswith(
            " MANAGE TRIGONOMETRY: Direction: Auto Collect(Expand)"
        )


async def test_space_and_backspace_step_the_active_field(app):
    async with app.run_test() as pilot:
        await manage(pilot, "b")
        await pilot.press("space")
        assert highlighted(app) == "Real"
        await pilot.press("backspace")
        assert highlighted(app) == "Principal"
        # Backspace wraps around the near end rather than leaving the field.
        await pilot.press("backspace")
        assert highlighted(app) == "Any"


async def test_committing_records_one_expression_per_changed_field(app):
    async with app.run_test() as pilot:
        # Direction to Expand, past Toward untouched, Angle to Degree.
        await manage(pilot, "t", "e", "tab", "d", "enter")
        assert app.settings["Trigonometry"] == "Expand"
        assert app.settings["Trigpower"] == "Auto"
        assert app.settings["Angle"] == "Degree"
        assert entries(app) == ["Trigonometry := Expand", "Angle := Degree"]
        assert highlighted(app) == "Author"


async def test_a_screen_that_changed_nothing_records_nothing(app):
    async with app.run_test() as pilot:
        await manage(pilot, "t", "enter")
        assert entries(app) == []
        assert highlighted(app) == "Author"


async def test_escape_abandons_every_pending_change(app):
    async with app.run_test() as pilot:
        await manage(pilot, "t", "c", "s", "d")
        await pilot.press("escape")
        assert app.settings["Trigonometry"] == "Auto"
        assert app.settings["Trigpower"] == "Auto"
        assert app.settings["Angle"] == "Radian"
        assert entries(app) == []
        # And one Esc lands on the Manage menu, not past it.
        assert band(app)[0].startswith(" MANAGE: ")


# -- the same settings from the author line ----------------------------------


async def test_an_authored_assignment_changes_the_setting_it_names(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"Branch := Real", "enter")
        assert app.settings["Branch"] == "Real"
        assert app.session.context.branch is Branch.REAL
        # And the screen that owns it opens on what was authored.
        await manage(pilot, "b")
        assert highlighted(app) == "Real"


def test_every_manage_setting_reaches_the_context(session):
    session.settings.apply(
        {
            "Branch": "Any",
            "Exponential": "Collect",
            "Logarithm": "Expand",
            "Trigonometry": "Collect",
            "Trigpower": "Cosines",
            "Angle": "Degree",
        }
    )
    context = session.context
    assert context.branch is Branch.ANY
    assert context.exponential is Direction.COLLECT
    assert context.logarithm is Direction.EXPAND
    assert context.trigonometry is Direction.COLLECT
    assert context.trigpower is TrigPower.COSINES
    assert context.angle is Angle.DEGREE


async def test_the_branch_setting_changes_the_answer(app):
    """The point of the step: the principal cube root of -8 is complex, and the
    real branch of it is -2."""
    async with app.run_test() as pilot:
        await pilot.press("a", *"(-8)^(1/3)", "enter")
        await pilot.press("s", "enter")
        assert entries(app)[-1] == "2*(-1)^(1/3)"
        await manage(pilot, "b", "r")
        await pilot.press("j", *"1", "enter")
        await pilot.press("s", "enter")
        assert entries(app)[-1] == "-2"


# -- Manage Renumber ---------------------------------------------------------


def test_renumber_puts_the_labels_back_in_sequence(session):
    """The manual's own case: a remove leaves 1, 3, 4, and renumbering makes it
    1, 2, 3 - the annotation of the last entry moving with the label it names."""
    authored(session, "x", "y", "z")
    session.remove(2, 2)
    session.simplify("#3")
    assert labels(session) == ["#1: x", "#3: z", "#4: z"]
    assert annotations(session) == ["User", "User", "Simp(#3)"]
    session.renumber()
    assert labels(session) == ["#1: x", "#2: z", "#3: z"]
    assert annotations(session) == ["User", "User", "Simp(#2)"]


def test_renumber_follows_the_order_the_entries_sit_in(session):
    """What a move, or a remove and an unremove, leaves: labels out of sequence
    because the entries are no longer stacked in the order they were numbered."""
    authored(session, "a", "b", "c", "d", "e")
    session.remove(2, 3)
    session.unremove()
    assert labels(session) == ["#1: a", "#4: d", "#5: e", "#2: b", "#3: c"]
    session.renumber()
    assert labels(session) == ["#1: a", "#2: d", "#3: e", "#4: b", "#5: c"]


def test_renumber_moves_a_reference_inside_an_expression(session):
    """Rederive keeps `#3` as a reference where the original resolved it as it
    read the line, so renumbering has to move it or it points somewhere else."""
    authored(session, "x", "y", "z")
    session.remove(2, 2)
    session.author("#3 + 1")
    session.renumber()
    assert labels(session) == ["#1: x", "#2: z", "#3: #2 + 1"]
    # And it still names the same expression it named before.
    assert session.simplify("#3").text == "z + 1"


def test_a_reference_in_a_renumbered_line_is_drawn_as_it_now_reads(session):
    authored(session, "x", "y", "z")
    session.remove(2, 2)
    session.author("#3 + 1")
    session.renumber()
    assert session.entries[-1].layout.lines == ("#2 + 1",)


def test_the_next_label_follows_the_renumbered_history(session):
    authored(session, "x", "y", "z")
    session.remove(2, 2)
    session.renumber()
    session.author("q")
    assert labels(session) == ["#1: x", "#2: z", "#3: q"]


def test_renumbering_a_history_already_in_sequence_changes_nothing(session):
    authored(session, "x", "y")
    before = list(session.entries)
    session.renumber()
    assert session.entries == before


def test_renumbering_an_empty_history_changes_nothing(session):
    session.renumber()
    session.author("x")
    assert labels(session) == ["#1: x"]


def test_renumber_leaves_the_highlight_where_it_was(session):
    authored(session, "x + 1", "y", "z")
    session.remove(2, 2)
    session.select_entry(0)
    session.move_right()
    session.renumber()
    assert session.selected == 0 and session.route == (0,)
    assert session.selected_node.value == "x"


async def test_renumber_runs_on_the_keystroke(app):
    """No prompt, no dialog, no message: the command menu comes straight back."""
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter", "a", *"y", "enter")
        await pilot.press("r", *"2", "tab", *"2", "enter")
        await pilot.press("a", *"z", "enter")
        assert [entry.number for entry in app.session.entries] == [1, 3]
        await manage(pilot, "r")
        assert [entry.number for entry in app.session.entries] == [1, 2]
        assert band(app)[0].startswith(" COMMAND: ")
        assert message(app) == "Enter option"


# -- Manage Annotate ---------------------------------------------------------


async def test_annotate_asks_for_a_label_and_then_for_the_text(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter", "a", *"y", "enter")
        await manage(pilot, "a")
        assert band(app) == [" MANAGE ANNOTATE: Expression: 2"]
        assert message(app) == "Enter label number"
        await pilot.press("enter")
        assert prompt(app) == ("ANNOTATION:", "User")
        assert message(app) == "Enter annotation"


async def test_the_label_field_follows_the_highlight(app):
    """As `Remove` and `Unremove` do: the manual would rather you picked the
    expression than typed its number."""
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter", "a", *"y", "enter")
        await manage(pilot, "a")
        await pilot.press("up")
        assert band(app) == [" MANAGE ANNOTATE: Expression: 1"]


async def test_an_annotation_reaches_the_status_line(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x + 1", "enter")
        await manage(pilot, "a", "enter", *"Kirchhoff", "enter")
        assert annotation(app) == "Kirchhoff"
        assert app.session.entries[0].annotation == "Kirchhoff"


async def test_annotating_appends_nothing_and_moves_no_highlight(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter", "a", *"y", "enter")
        await manage(pilot, "a", "enter", *"Mine", "enter")
        assert entries(app) == ["x", "y"]
        assert app.session.selected == 1
        assert band(app)[0].startswith(" COMMAND: ")


async def test_the_annotation_offered_is_the_one_the_entry_carries(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x + 1", "enter")
        await pilot.press("s", "enter")
        await manage(pilot, "a", "enter")
        assert prompt(app) == ("ANNOTATION:", "Simp(#1)")


async def test_escape_from_the_annotation_line_lands_on_the_manage_menu(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter")
        await manage(pilot, "a", "enter", *"Mine")
        await pilot.press("escape")
        assert app.session.entries[0].annotation == "User"
        assert band(app)[0].startswith(" MANAGE: ")


async def test_a_label_that_names_no_entry_is_refused(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter")
        await manage(pilot, "a", *"7", "enter")
        # The question stays up with what was typed still on it to correct.
        assert band(app) == [" MANAGE ANNOTATE: Expression: 7"]


async def test_annotate_asks_nothing_of_an_empty_history(app):
    async with app.run_test() as pilot:
        await manage(pilot, "a")
        assert band(app)[0].startswith(" COMMAND: ")
        assert entries(app) == []


def test_an_annotation_is_written_above_the_expression_it_belongs_to(
    session, tmp_path
):
    authored(session, "x + 1")
    session.annotate(1, "Kirchhoff")
    file = tmp_path / "sheet.mth"
    session.save(file)
    assert file.read_text().splitlines()[:2] == [";Kirchhoff", "x+1"]


def test_an_annotation_read_back_is_the_one_that_was_written(session, tmp_path):
    authored(session, "x + 1")
    session.annotate(1, "Kirchhoff")
    file = tmp_path / "sheet.mth"
    session.save(file)
    other = Session()
    other.load(file)
    assert annotations(other) == ["Kirchhoff"]


def test_an_annotation_moves_with_its_label_when_the_history_is_renumbered(session):
    authored(session, "x", "y", "z")
    session.remove(2, 2)
    session.simplify("#3")
    session.annotate(4, "from #3, twice over")
    session.renumber()
    assert annotations(session)[-1] == "from #2, twice over"


# -- Manage Ordering ---------------------------------------------------------


async def test_ordering_offers_the_list_in_force(app):
    async with app.run_test() as pilot:
        await manage(pilot, "o")
        assert prompt(app) == ("MANAGE ORDER variables:", "x y z")
        assert message(app) == "Enter variables in desired order"


async def test_ordering_takes_a_new_list_and_records_nothing(app):
    async with app.run_test() as pilot:
        await manage(pilot, "o", *"y x z", "enter")
        assert app.session.order == ("y", "x", "z")
        # Unlike the four settings screens, this one appends no expression.
        assert entries(app) == []
        assert band(app)[0].startswith(" COMMAND: ")
        # And the next Manage Ordering opens on what was entered.
        await manage(pilot, "o")
        assert prompt(app) == ("MANAGE ORDER variables:", "y x z")


async def test_a_line_that_is_not_variables_is_refused(app):
    """Derive stores `q 2 + w` verbatim; reproducing that would be a bug."""
    async with app.run_test() as pilot:
        await manage(pilot, "o", *"q 2 + w", "enter")
        assert app.session.order == ("x", "y", "z")
        assert prompt(app) == ("MANAGE ORDER variables:", "q 2 + w")


def test_a_variable_may_not_be_named_twice(session):
    assert session.order_list("x y x") is None


def test_a_predefined_name_is_not_a_variable(session):
    assert session.order_list("sin") is None
    assert session.order_list("pi") is None


def test_a_variable_goes_on_the_list_the_way_an_expression_records_it(session):
    assert session.order_list("Y, X") == ("y", "x")


def test_an_empty_line_is_an_empty_list(session):
    assert session.order_list("") == ()


def test_the_order_list_changes_which_variable_factor_offers_first(session):
    authored(session, "x^2 - y^2")
    assert session.variables("#1") == ("x", "y")
    session.order = ("y", "x", "z")
    assert session.variables("#1") == ("y", "x")


def test_a_variable_off_the_list_is_less_main_than_one_on_it(session):
    authored(session, "x^2 - a^2")
    assert session.variables("#1") == ("x", "a")
    session.order = ("a",)
    assert session.variables("#1") == ("a", "x")


async def test_the_order_list_reaches_the_variables_factor_offers(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x^2 - y^2", "enter")
        await manage(pilot, "o", *"y x z", "enter")
        await pilot.press("f", "enter")
        assert message(app) == "Return for all or select 1: y,x"


def test_the_order_list_decides_a_declared_functions_parameters(session):
    session.order = ("y", "x", "z")
    entry = session.declare_function("F", "x^2 + y")
    assert entry.text == "F(y, x) := x^2 + y"


# -- Manage Substitute -------------------------------------------------------
#
# The manual's own exercises (section 4.8) make the corpus: the answers below
# are what the original appends for them, unsimplified and annotated `Sub(#1)`
# whether a whole expression or a subexpression was highlighted.


def highlight(session, *route):
    """Highlight a subexpression of the selected entry, by successive steps."""
    for step in route:
        getattr(session, f"move_{step}")()
    return session.selected_node


def test_a_value_is_written_in_for_each_variable(session):
    authored(session, "a x^2 + b x + c")
    entry = session.substitute("#1", {"x": "2", "a": "3", "b": "5", "c": "c"})
    # Left exactly as it was written: 22 is what a Simplify of it would say.
    assert entry.text == "3*2^2+5*2+c"
    assert entry.annotation == "Sub(#1)"


def test_an_expression_may_be_written_in_as_well_as_a_number(session):
    authored(session, "a x^2 + b x + c")
    assert session.substitute("#1", {"x": "x + 1"}).text == "a*(x+1)^2+b*(x+1)+c"


def test_the_substitutions_all_go_in_at_once(session):
    """Interchanging two variables interchanges them: `x` for `y` is not read
    again once `y` has been written in for `x`."""
    authored(session, "x + 2 y")
    assert session.substitute("#1", {"x": "y", "y": "x"}).text == "y+2*x"


def test_two_variables_are_interchanged_wherever_they_stand(session):
    authored(session, "a x^2 + b x + c")
    assert session.substitute("#1", {"a": "c", "c": "a"}).text == "c*x^2+b*x+a"


def test_a_variable_answered_with_its_own_name_is_left_alone(session):
    authored(session, "a x^2 + b x + c")
    assert session.substitute("#1", {"x": "2", "a": "a"}).text == "a*2^2+b*2+c"


def test_a_blank_value_leaves_its_variable_alone(session):
    authored(session, "a x + b")
    assert session.substitute("#1", {"a": "", "b": "2"}).text == "a*x+2"


def test_nothing_to_write_in_writes_the_expression_out_again(session):
    authored(session, "2 + 3")
    assert session.substitute("#1", {}).text == "2+3"


def test_a_typed_expression_is_substituted_into_as_the_users_own(session):
    assert session.substitute("x + 1", {"x": "5"}).text == "5+1"
    assert session.entries[-1].annotation == "Sub(User)"


def test_a_line_that_does_not_parse_appends_nothing(session):
    authored(session, "x")
    with pytest.raises(DeriveSyntaxError):
        session.substitute("#1", {"x": "2 +"})
    assert labels(session) == ["#1: x"]


def test_the_result_is_appended_unsimplified_and_can_be_simplified_after(session):
    authored(session, "a x^2 + b x + c")
    session.substitute("#1", {"x": "2", "a": "3", "b": "5", "c": "0"})
    assert session.simplify("#2").text == "22"


# -- a highlighted subexpression ---------------------------------------------


def test_every_match_of_the_highlighted_part_is_replaced(session):
    """The manual's own example: only the whole highlighted product matches, so
    the radical in the denominator is left where it is."""
    authored(session, "3 SQRT(x^2 + y^2)/(2 + SQRT(x^2 + y^2))")
    assert highlight(session, "right").kind == "product"
    entry = session.substitute_part("#1", "r")
    assert entry.text == "r/(2+SQRT(x^2+y^2))"
    # No quote on the label: the answer is derived from the whole entry, since
    # that is where the substitution happened.
    assert entry.annotation == "Sub(#1)"


def test_a_match_is_the_whole_subexpression_and_not_a_part_of_one(session):
    """`k` for `t^3` gives `t^6 + k`, never `k^2 + k`: `t^6` is not a match."""
    authored(session, "t^6 + t^3")
    assert highlight(session, "right", "right").value == "^"
    assert session.substitute_part("#1", "k").text == "t^6+k"


def test_the_same_subexpression_is_replaced_wherever_it_stands(session):
    authored(session, "(x + 1) y + (x + 1)")
    highlight(session, "right", "right")
    assert session.substitute_part("#1", "u").text == "u*y+u"


def test_a_match_is_of_the_expression_and_not_of_how_it_was_written(session):
    """`2 x` and `2*x` are one product written two ways, and a match is a match
    of the tree."""
    authored(session, "2 x + 2*x")
    highlight(session, "right")
    assert session.substitute_part("#1", "u").text == "u+u"


def test_a_blank_replacement_writes_nothing_in(session):
    authored(session, "t^6 + t^3")
    highlight(session, "right", "right")
    assert session.substitute_part("#1", "").text == "t^6+t^3"


def test_a_highlighted_part_is_what_makes_the_command_ask_once(session):
    authored(session, "t^6 + t^3")
    assert not session.substitutes_part("#1")
    highlight(session, "right", "right")
    assert session.substitutes_part("#1")
    # A highlight in another entry says nothing about the one being named.
    authored(session, "z")
    session.select_entry(0)
    highlight(session, "right")
    assert not session.substitutes_part("#2")


# -- the command as it is driven ---------------------------------------------


async def test_substitute_asks_for_the_expression_and_then_for_each_variable(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"a x^2 + b x + c", "enter")
        await manage(pilot, "s")
        assert prompt(app) == ("MANAGE SUBSTITUTE expression:", "#1")
        assert message(app) == "Enter expression"
        await pilot.press("enter")
        # The variables come in main order: x is on the order list, and the
        # rest sort alphabetically behind it.
        assert prompt(app) == ("MANAGE SUBSTITUTE value:", "x")
        assert message(app) == "Enter replacement for x"
        await pilot.press(*"2", "enter")
        assert prompt(app) == ("MANAGE SUBSTITUTE value:", "a")
        assert message(app) == "Enter replacement for a"
        await pilot.press(*"3", "enter")
        assert prompt(app) == ("MANAGE SUBSTITUTE value:", "b")
        await pilot.press(*"5", "enter")
        assert prompt(app) == ("MANAGE SUBSTITUTE value:", "c")
        # Enter on the name it offers leaves that variable standing.
        await pilot.press("enter")
        assert entries(app)[-1] == "3*2^2+5*2+c"
        assert annotation(app) == "Sub(#1)"
        assert message(app).startswith("Compute time:")
        assert highlighted(app) == "Author"


async def test_the_answer_is_drawn_as_it_stands(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"a x^2 + b x + c", "enter")
        await manage(pilot, "s", "enter", *"2", "enter", *"3", "enter")
        await pilot.press(*"5", "enter", "enter")
        assert work_area(app)[-2:] == ["        2", "#2:  3·2  + 5·2 + c"]


async def test_a_subexpression_is_asked_about_once_and_offered_nothing(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"t^6 + t^3", "enter")
        await pilot.press("right", "right")
        assert highlighted_expression(app) == " 3\nt"
        await manage(pilot, "s")
        assert prompt(app) == ("MANAGE SUBSTITUTE expression:", "#1")
        await pilot.press("enter")
        assert prompt(app) == ("MANAGE SUBSTITUTE value:", "")
        assert message(app) == "Enter replacement for subexpression"
        await pilot.press(*"k", "enter")
        assert entries(app)[-1] == "t^6+k"
        assert annotation(app) == "Sub(#1)"


async def test_an_expression_with_nothing_in_it_to_replace_asks_nothing(app):
    """`2 + 3` has no variable and no highlight inside it, so there is nothing
    to ask and nothing to append - and the Manage menu is still up."""
    async with app.run_test() as pilot:
        await pilot.press("a", *"2 + 3", "enter")
        await manage(pilot, "s", "enter")
        assert entries(app) == ["2 + 3"]
        assert band(app)[0].startswith(" MANAGE: ")


async def test_ctrl_enter_simplifies_what_was_substituted(app):
    """The original nests the annotation as `Simp(Sub(#1))`; every Ctrl-Enter
    here appends a separate entry instead, and this one is no exception."""
    async with app.run_test() as pilot:
        await pilot.press("a", *"a x^2 + b x + c", "enter")
        await manage(pilot, "s", "enter", *"2", "enter", *"3", "enter")
        await pilot.press(*"5", "enter", *"0")
        await pilot.press("ctrl+j")
        assert entries(app)[-2:] == ["3*2^2+5*2+0", "22"]
        assert annotations(app.session)[-2:] == ["Sub(#1)", "Simp(#2)"]


async def test_a_value_that_does_not_read_leaves_the_line_up(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x + 1", "enter")
        await manage(pilot, "s", "enter")
        await pilot.press(*"2 +", "enter")
        assert message(app) == "Syntax error detected at cursor"
        assert entries(app) == ["x + 1"]
        assert prompt(app) == ("MANAGE SUBSTITUTE value:", "2 +")


@pytest.mark.parametrize(
    ("keys", "step"), [(("s",), "the expression"), (("s", "enter"), "a value")], ids=str
)
async def test_escape_abandons_substitute_from_either_question(app, keys, step):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x + 1", "enter")
        await manage(pilot, *keys)
        await pilot.press("escape")
        assert entries(app) == ["x + 1"]
        assert app.substituting is None
        # One Esc lands on the Manage menu, as it does from any of its lines.
        assert band(app)[0].startswith(" MANAGE: ")


async def test_the_expression_line_follows_the_highlight(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x", "enter", "a", *"y", "enter")
        await manage(pilot, "s")
        assert prompt(app) == ("MANAGE SUBSTITUTE expression:", "#2")
        await pilot.press("up")
        assert prompt(app) == ("MANAGE SUBSTITUTE expression:", "#1")


# -- the state file ----------------------------------------------------------


def test_a_state_file_carries_the_manage_settings(session, tmp_path):
    file = tmp_path / "state.ini"
    settings = {
        "Branch": "Real",
        "Exponential": "Collect",
        "Logarithm": "Expand",
        "Trigonometry": "Expand",
        "Trigpower": "Sines",
        "Angle": "Degree",
    }
    session.settings.apply(settings)
    session.save_state(file)
    lines = file.read_text().splitlines()
    for name, value in settings.items():
        assert f"{name} := {value}" in lines
    other = Session()
    assert other.load_state(file) == 0
    for name, value in settings.items():
        assert other.settings[name] == value


def test_a_state_file_carries_the_variable_order_list(session, tmp_path):
    """Not a setting - no field owns it and nothing records it as an assignment
    - but Derive kept it in DERIVE.INI, so a session opens the way you left it."""
    file = tmp_path / "state.ini"
    session.order = ("y", "x", "z")
    session.save_state(file)
    assert "VariableOrder := y x z" in file.read_text().splitlines()
    other = Session()
    assert other.load_state(file) == 0
    assert other.order == ("y", "x", "z")


def test_an_order_line_that_names_no_variables_is_counted(session, tmp_path):
    file = tmp_path / "state.ini"
    file.write_text("VariableOrder := q 2 + w\n")
    assert session.load_state(file) == 1
    assert session.order == ("x", "y", "z")
