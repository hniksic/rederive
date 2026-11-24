"""Screening terminal input before Textual parses it: the replies to the keyboard query,
Alt-Enter from a terminal without the keyboard protocol, and the text on an Alt key."""

from textual._xterm_parser import XTermParser

from rederive.ui.keyboard import screened


def test_a_flags_reply_says_shift_enter_is_reported_and_goes():
    assert screened("\x1b[?1u\x1b[?62;22c") == ("", True)


def test_only_the_attributes_reply_means_it_is_not():
    assert screened("\x1b[?65;1;9c") == ("", False)


def test_the_replies_are_taken_out_of_typing_around_them():
    assert screened("a\x1b[?0ub\x1b[?6cc") == ("abc", True)


def test_alt_enter_is_rewritten_as_the_protocol_reports_it():
    assert screened("x\x1b\ry") == ("x\x1b[13;3uy", False)


def test_keys_the_protocol_reports_pass_unchanged():
    assert screened("\x1b[13;2u\x1b[97;5u") == ("\x1b[13;2u\x1b[97;5u", False)


def test_alt_with_the_keys_own_letter_loses_the_text():
    assert screened("\x1b[101;3;101u") == ("\x1b[101;3u", False)


def test_alt_shift_with_the_capital_loses_the_text():
    assert screened("\x1b[101;4;69u") == ("\x1b[101;4u", False)


def test_a_character_alt_composed_keeps_the_text():
    assert screened("\x1b[101;3;233u") == ("\x1b[101;3;233u", False)


def test_text_without_alt_is_kept():
    assert screened("\x1b[101;2;69u") == ("\x1b[101;2;69u", False)


def test_textual_reads_alt_e_from_xterm_js_as_alt_e():
    (data, _) = screened("\x1b[101;3;101u")
    assert [event.key for event in XTermParser().feed(data)] == ["alt+e"]
