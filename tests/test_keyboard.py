"""Screening terminal input before Textual parses it: the replies to the keyboard query,
and Alt-Enter from a terminal without the keyboard protocol."""

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
