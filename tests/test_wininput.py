"""Windows console input records translated into terminal input.

Each translation is checked by what Textual's parser makes of it, since that is what
the app sees: key names, mouse events and pastes.
"""

import pytest
from textual import events
from textual._xterm_parser import XTermParser

from rederive.ui.wininput import (
    LEFT_ALT,
    LEFT_BUTTON,
    LEFT_CTRL,
    MIDDLE_BUTTON,
    MOUSE_HWHEELED,
    MOUSE_MOVED,
    MOUSE_WHEELED,
    RIGHT_ALT,
    RIGHT_BUTTON,
    SHIFT,
    FocusRecord,
    KeyRecord,
    MouseRecord,
    Translator,
)

CTRL, ALT = LEFT_CTRL, LEFT_ALT


def press(vk, char="", state=0, repeat=1):
    """A key's press and release, as the console reports a typed key."""
    return [KeyRecord(True, vk, char, state, repeat), KeyRecord(False, vk, char, state)]


def typed(text):
    """Records typing `text`, letters and digits by their own keys."""
    records = []
    for character in text:
        if character == "\r":
            records += press(0x0D, "\r")
        elif character.isupper():
            records += [KeyRecord(True, 0x10, "", SHIFT)]
            records += press(ord(character), character, SHIFT)
            records += [KeyRecord(False, 0x10, "", 0)]
        elif character.isalnum():
            records += press(ord(character.upper()), character)
        else:
            records += press(0, character)
    return records


def parsed(*batches):
    """What Textual's parser makes of the input for each batch of records."""
    translator = Translator()
    parser = XTermParser()
    found = []
    for batch in batches:
        # An empty feed is the end of input to the parser.
        if text := translator.feed(batch):
            found += parser.feed(text)
    return found


def keys(*batches):
    found = parsed(*batches)
    assert all(isinstance(event, events.Key) for event in found), found
    return [event.key for event in found]


# -- keys ------------------------------------------------------------------------------


def test_typed_characters_are_themselves():
    assert keys(typed("a1Z ")) == ["a", "1", "Z", "space"]


def test_only_presses_type_and_the_repeat_count_is_honored():
    assert keys([KeyRecord(False, 0x41, "a")]) == []
    assert keys([KeyRecord(True, 0x41, "a", repeat=3)]) == ["a", "a", "a"]


def test_a_modifier_pressed_alone_types_nothing():
    for vk, state in ((0x10, SHIFT), (0x11, CTRL), (0x12, ALT), (0xA5, RIGHT_ALT)):
        assert keys(press(vk, "", state)) == []


def test_altgr_types_the_character_rather_than_a_shortcut():
    # AltGr is reported as Ctrl and Alt held, with the character the layout gives.
    assert [(e.key, e.character) for e in parsed(press(0x51, "@", CTRL | RIGHT_ALT))] == [
        ("at", "@")
    ]
    assert [e.character for e in parsed(press(0x45, "€", CTRL | RIGHT_ALT))] == ["€"]


def test_ctrl_with_a_letter_is_the_control_character():
    assert keys(press(0x41, "\x01", CTRL), press(0x56, "\x16", CTRL)) == [
        "ctrl+a",
        "ctrl+v",
    ]


def test_alt_with_a_character_is_alt_and_the_key():
    assert keys(press(0x41, "a", ALT)) == ["alt+a"]
    assert keys(press(0x41, "A", ALT | SHIFT)) == ["alt+shift+a"]
    # Not the start of a sequence, as ESC [ would be.
    assert keys(press(0xDB, "[", ALT), typed("A")) == ["alt+left_square_bracket", "A"]


def test_ctrl_with_a_digit_and_ctrl_alt_with_a_letter():
    assert keys(press(0x31, "", CTRL)) == ["ctrl+1"]
    assert keys(press(0x41, "", CTRL | ALT)) == ["alt+ctrl+a"]


@pytest.mark.parametrize(
    "state, key",
    [
        (0, "enter"),
        (SHIFT, "shift+enter"),
        (ALT, "alt+enter"),
        (CTRL, "ctrl+j"),
        (CTRL | SHIFT, "ctrl+shift+enter"),
    ],
)
def test_enter(state, key):
    char = "\n" if state & CTRL else "\r"
    assert keys(press(0x0D, char, state)) == [key]


@pytest.mark.parametrize(
    "vk, char, state, key",
    [
        (0x09, "\t", 0, "tab"),
        (0x09, "\t", SHIFT, "shift+tab"),
        (0x09, "\t", CTRL, "ctrl+tab"),
        (0x08, "\x08", 0, "backspace"),
        (0x08, "\x7f", CTRL, "ctrl+backspace"),
        (0x08, "\x08", ALT, "alt+backspace"),
        (0x1B, "\x1b", 0, "escape"),
        (0xDB, "\x1b", CTRL, "escape"),
    ],
)
def test_editing_keys(vk, char, state, key):
    assert keys(press(vk, char, state)) == [key]


@pytest.mark.parametrize(
    "vk, name",
    [
        (0x26, "up"),
        (0x28, "down"),
        (0x27, "right"),
        (0x25, "left"),
        (0x24, "home"),
        (0x23, "end"),
        (0x21, "pageup"),
        (0x22, "pagedown"),
        (0x2D, "insert"),
        (0x2E, "delete"),
        (0x70, "f1"),
        (0x73, "f4"),
        (0x74, "f5"),
        (0x7B, "f12"),
    ],
)
@pytest.mark.parametrize(
    "state, prefix",
    [
        (0, ""),
        (SHIFT, "shift+"),
        (CTRL, "ctrl+"),
        (ALT, "alt+"),
        (CTRL | SHIFT, "ctrl+shift+"),
    ],
)
def test_cursor_and_function_keys_with_modifiers(vk, name, state, prefix):
    assert keys(press(vk, "", state)) == [prefix + name]


def test_a_character_outside_the_bmp_is_joined_from_its_halves():
    halves = "😀".encode("utf-16-le")
    high = halves[:2].decode("utf-16-le", "surrogatepass")
    low = halves[2:].decode("utf-16-le", "surrogatepass")
    assert [e.character for e in parsed(press(0, high) + press(0, low))] == ["😀"]
    # Even when the halves are read separately.
    assert [e.character for e in parsed(press(0, high), press(0, low))] == ["😀"]


def test_the_character_from_an_alt_keypad_code_comes_with_releasing_alt():
    records = [
        KeyRecord(True, 0x12, "", ALT),
        *press(0x62, "", ALT),
        *press(0x63, "", ALT),
    ]
    assert keys([*records, KeyRecord(False, 0x12, "é", 0)]) == ["é"]


def test_focus_records_are_ignored():
    assert parsed([FocusRecord(True), FocusRecord(False)]) == []


# -- pastes ----------------------------------------------------------------------------


def test_text_with_a_line_break_read_together_is_a_paste():
    found = parsed(typed("f(x) :=\r  X + 1"))
    assert [type(e) for e in found] == [events.Paste]
    assert found[0].text == "f(x) :=\n  X + 1"


def test_a_pasted_lf_and_tab_stay_in_the_paste():
    found = parsed(
        [*typed("a"), *press(0x0D, "\n", CTRL), *press(0x09, "\t"), *typed("b")]
    )
    assert [type(e) for e in found] == [events.Paste]
    assert found[0].text == "a\n\tb"


def test_a_typed_enter_is_a_key_even_after_typing_read_separately():
    assert keys(typed("23"), press(0x0D, "\r")) == ["2", "3", "enter"]


def test_text_without_a_line_break_is_typing():
    assert keys(typed("abc")) == ["a", "b", "c"]


def test_line_breaks_without_text_are_keys():
    assert keys(press(0x0D, "\r") + press(0x0D, "\r")) == ["enter", "enter"]


def test_shift_enter_alone_is_a_key():
    assert keys(press(0x0D, "\r", SHIFT)) == ["shift+enter"]


def test_other_keys_in_a_paste_stay_keys_around_it():
    found = parsed([*press(0x26, ""), *typed("a\rb"), *press(0x41, "a", ALT)])
    assert [type(e) for e in found] == [events.Key, events.Paste, events.Key]
    assert [found[0].key, found[1].text, found[2].key] == ["up", "a\nb", "alt+a"]


# -- the mouse -------------------------------------------------------------------------


def mouse(found):
    return [
        (type(e).__name__, e.button, e.x, e.y, e.shift, e.meta, e.ctrl)
        for e in found
        if isinstance(e, events.MouseEvent)
    ]


def test_a_click_is_a_press_and_a_release_at_the_cell():
    found = parsed([MouseRecord(4, 2, LEFT_BUTTON), MouseRecord(4, 2, 0)])
    assert mouse(found) == [
        ("MouseDown", 1, 4, 2, False, False, False),
        ("MouseUp", 1, 4, 2, False, False, False),
    ]


def test_each_button_and_the_modifiers():
    found = parsed(
        [
            MouseRecord(0, 0, MIDDLE_BUTTON, CTRL),
            MouseRecord(0, 0, MIDDLE_BUTTON | RIGHT_BUTTON, SHIFT | ALT),
            MouseRecord(0, 0, 0),
        ]
    )
    assert mouse(found) == [
        ("MouseDown", 2, 0, 0, False, False, True),
        ("MouseDown", 3, 0, 0, True, True, False),
        ("MouseUp", 2, 0, 0, False, False, False),
        ("MouseUp", 3, 0, 0, False, False, False),
    ]


def test_motion_with_and_without_a_button():
    found = parsed(
        [
            MouseRecord(1, 1, 0, flags=MOUSE_MOVED),
            MouseRecord(1, 1, LEFT_BUTTON),
            MouseRecord(5, 1, LEFT_BUTTON, flags=MOUSE_MOVED),
            MouseRecord(5, 1, 0),
        ]
    )
    assert mouse(found) == [
        ("MouseMove", 0, 1, 1, False, False, False),
        ("MouseDown", 1, 1, 1, False, False, False),
        ("MouseMove", 1, 5, 1, False, False, False),
        ("MouseUp", 1, 5, 1, False, False, False),
    ]


def test_the_wheel_both_ways_and_across():
    up, down = 120 << 16, (0x10000 - 120) << 16
    found = parsed(
        [
            MouseRecord(3, 3, up, flags=MOUSE_WHEELED),
            MouseRecord(3, 3, down, flags=MOUSE_WHEELED),
            MouseRecord(3, 3, up, flags=MOUSE_HWHEELED),
            MouseRecord(3, 3, down, flags=MOUSE_HWHEELED),
        ]
    )
    assert [type(e).__name__ for e in found] == [
        "MouseScrollUp",
        "MouseScrollDown",
        "MouseScrollRight",
        "MouseScrollLeft",
    ]
    assert all((e.x, e.y) == (3, 3) for e in found)
