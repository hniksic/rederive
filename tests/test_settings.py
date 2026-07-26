"""The settings model and the dialog editor, with no UI involved."""

import pytest
from sexpr import to_sexpr

from rederive.model.settings import (
    COLOR_MENU,
    COLORS,
    DIALOGS,
    INPUT,
    MUTE,
    NOTATION,
    OTHER,
    OUTPUT,
    PRECISION,
    RADIX,
    ChoiceField,
    DialogEditor,
    Settings,
)
from rederive.syntax import (
    DeriveSyntaxError,
    ParseState,
    SettingDeclaration,
    parse_expression,
)


@pytest.fixture
def settings():
    return Settings()


def editor(dialog, settings):
    return DialogEditor(dialog, settings)


def test_defaults_are_the_originals(settings):
    assert settings["Precision"] == "Exact"
    assert settings["PrecisionDigits"] == 6
    assert settings["Notation"] == "Rational"
    assert settings["InputMode"] == "Character"
    assert settings["CaseMode"] == "Insensitive"
    assert settings["ArrowKeyMode"] == "LineEdit"
    assert settings["DisplayFormat"] == "Normal"
    assert settings["TimesOperator"] == "Dot"
    assert settings["InputBase"] == "Decimal"
    assert settings["OutputBase"] == "Decimal"
    assert settings["Mute"] == "No"


def test_watchers_hear_which_settings_changed(settings):
    heard = []
    settings.watch(heard.append)
    changed = settings.apply({"Precision": "Mixed", "PrecisionDigits": 6})
    assert changed == ("Precision",)
    assert heard == [frozenset({"Precision"})]
    # Applying the same values again is not a change, and says nothing.
    assert settings.apply({"Precision": "Mixed"}) == ()
    assert len(heard) == 1


def test_choosing_a_value_moves_on_to_the_next_field(settings):
    dialog = editor(PRECISION, settings)
    assert dialog.field.label == "Mode"
    assert dialog.choose("Approximate")
    assert dialog.field.label == "Digits"
    assert dialog.values["Precision"] == "Approximate"


def test_space_steps_through_the_choices_and_wraps(settings):
    dialog = editor(OUTPUT, settings)
    assert dialog.value(dialog.field) == "Normal"
    dialog.cycle()
    assert dialog.value(dialog.field) == "Compressed"
    dialog.cycle()
    assert dialog.value(dialog.field) == "Normal"


def test_backspace_steps_back_through_the_choices_and_wraps(settings):
    dialog = editor(RADIX, settings)
    assert dialog.value(dialog.field) == "Decimal"
    dialog.cycle(-1)
    assert dialog.value(dialog.field) == "Octal"
    dialog.cycle(-1)
    assert dialog.value(dialog.field) == "Binary"
    dialog.cycle(-1)
    assert dialog.value(dialog.field) == "Other"


def test_tab_wraps_through_every_field(settings):
    dialog = editor(INPUT, settings)
    labels = []
    for _ in range(4):
        labels.append(dialog.field.label)
        dialog.next_field()
    assert labels == ["Mode", "Case", "Use", "Mode"]


def test_typing_overwrites_the_number_under_the_cursor(settings):
    dialog = editor(NOTATION, settings)
    dialog.next_field()
    assert dialog.text == "6"
    dialog.type_character("1")
    dialog.type_character("2")
    assert dialog.text == "12"
    dialog.erase()
    assert dialog.text == "1"
    assert dialog.commit()["NotationDigits"] == 1


def test_a_digit_count_below_one_is_refused(settings):
    dialog = editor(PRECISION, settings)
    dialog.next_field()
    dialog.type_character("0")
    assert dialog.commit() is None
    # The dialog stays put, with the offending digit still there to correct.
    assert dialog.text == "0"
    assert settings["PrecisionDigits"] == 6


def test_other_asks_for_a_base_between_2_and_36(settings):
    dialog = editor(RADIX, settings)
    dialog.cycle()  # Decimal -> Hexadecimal
    dialog.cycle()  # -> Other
    assert dialog.commit() is None
    assert dialog.prompt is not None
    # The base offered is the one in effect, not the one just stepped over.
    assert dialog.text == "10"
    dialog.type_character("4")
    dialog.type_character("0")
    assert dialog.commit() is None
    dialog.erase()
    dialog.erase()
    dialog.type_character("7")
    assert dialog.commit()["InputBase"] == 7


def test_a_lone_field_settles_as_soon_as_it_is_answered(settings):
    dialog = editor(MUTE, settings)
    assert dialog.settles_on_choice
    assert not editor(PRECISION, settings).settles_on_choice


def test_only_the_changed_fields_are_reported(settings):
    dialog = editor(INPUT, settings)
    dialog.choose("Word")
    assert [field.setting for field in dialog.changes(settings)] == ["InputMode"]
    assert [field.setting for field in editor(INPUT, settings).changes(settings)] == []


def test_a_record_is_written_the_way_the_settings_say_to_write_it(settings):
    settings.apply({"Precision": "Approximate"})
    assert settings.assignment("Precision") == "Precision := Approximate"
    settings.apply({"DisplayFormat": "Compressed"})
    assert settings.assignment("DisplayFormat") == "DisplayFormat:=Compressed"
    settings.apply({"OutputBase": "Octal", "NotationDigits": 12})
    assert settings.assignment("NotationDigits") == "NotationDigits:=14"


def test_a_record_is_the_expression_the_parser_would_have_read(settings):
    """A record is built, not parsed, but the two must not disagree.

    Every value of every recorded field goes round: the parser reads the record
    back as the setting it records, and as the same expression. `Normal` and
    `Expand` are the ones this catches - both are built-in functions too.
    """
    for dialog in DIALOGS:
        for field in dialog.fields:
            if not field.recorded or not isinstance(field, ChoiceField):
                continue
            for choice in field.choices:
                if choice == OTHER:
                    continue
                settings.apply({field.setting: choice})
                text = settings.assignment(field.setting)
                result = parse_expression(text, ParseState())
                assert result.declarations == (
                    SettingDeclaration(field.setting, choice),
                ), text
                assert result.node == settings.assignment_node(field.setting), text


def test_a_numeric_record_is_built_because_it_may_not_lex(settings):
    # `InputBase := 7` is written in base ten and read in base seven, where 7
    # is not a digit. The record is the expression, and it says seven.
    settings.apply({"InputBase": 7})
    node = settings.assignment_node("InputBase")
    assert settings.assignment("InputBase") == "InputBase := 7"
    assert to_sexpr(node) == '(:= InputBase 7)'
    with pytest.raises(DeriveSyntaxError):
        parse_expression("InputBase := 7", ParseState(input_base=7))


def test_numerals_take_a_leading_zero_when_they_start_with_a_letter(settings):
    settings.apply({"OutputBase": "Hexadecimal"})
    assert settings.numeral(14) == "0E"
    assert settings.numeral(31) == "1F"
    assert settings.numeral(9) == "9"
    settings.apply({"OutputBase": 36})
    assert settings.numeral(35) == "0Z"
    assert settings.numeral(-35) == "-0Z"


def test_every_color_slot_starts_from_a_named_color(settings):
    for field in COLOR_MENU.fields:
        assert settings[field.setting] in COLORS
