"""Reading Derive for Windows worksheets, the binary `.dfw` files of Derive 5 and 6.

The worksheets are built byte by byte, by `dfwsheet.Sheet`.
"""

import struct
from pathlib import Path

import pytest
from dfwsheet import MODERN, OLD, Sheet
from screen import entries, message

from rederive.__main__ import Usage, named, read
from rederive.model import dfw, worksheet
from rederive.model.session import Session
from rederive.syntax import (
    InputMode,
    ParseState,
    SettingDeclaration,
    parse_expression,
    write_expression,
)
from rederive.ui.app import RederiveApp

#: The settings Derive 6 saves at the end of every state block.
SETTINGS = [
    "Precision:=Exact",
    "PrecisionDigits:=10",
    "Notation:=Rational",
    "NotationDigits:=10",
    "Branch:=Principal",
    "Exponential:=Auto",
    "Logarithm:=Auto",
    "Trigonometry:=Auto",
    "Trigpower:=Auto",
    "Angle:=Radian",
    "CaseMode:=Insensitive",
    "VariableOrder:=[x,y,z]",
    "OutputBase:=Decimal",
    "InputBase:=Decimal",
    "InputMode:=Character",
    "DisplayFormat:=Normal",
    "TimesOperator:=Dot",
    "DisplaySteps:=false",
]


@pytest.fixture
def session():
    return Session()


@pytest.fixture
def file(tmp_path):
    return tmp_path / "work.dfw"


def texts(session):
    return [entry.text for entry in session.entries]


def annotations(session):
    return [entry.annotation for entry in session.entries]


RTF = (
    r"{\rtf1\ansi\ansicpg1252\deff0{\fonttbl{\f0\fswiss Arial;}}"
    r"{\colortbl ;\red255\green0\blue0;}"
    r"\pard\f0\fs20 First \b paragraph\b0 .\par Second, with \'e4 and "
    r"\uc1\u945?.\par}"
)


# -- the worksheet's objects -------------------------------------------------


def test_expressions_load_in_order_with_their_annotations(session, file):
    Sheet().expression("SIN(x)/2").expression("SIN(x)/2", "Simp(#1)").write(file)
    assert not session.load(file)
    assert texts(session) == ["SIN(x)/2", "SIN(x)/2"]
    assert annotations(session) == ["User", "Simp(#1)"]


def test_a_localized_or_one_line_simplification_counts_as_typed(session, file):
    """Derive 6 annotates `2+3=` simplified in place as `User=Simp(User)`."""
    sheet = Sheet().expression("a", "Benutzer").expression("2+3=5", "User=Simp(User)")
    sheet.expression("b", "Simp(User)").write(file)
    session.load(file)
    assert annotations(session) == ["User", "User", "Simp(User)"]
    assert texts(session) == ["a", "2+3=5", "b"]


def test_text_becomes_a_string_per_paragraph(session, file):
    Sheet().text(RTF).expression("x").write(file)
    session.load(file)
    assert texts(session) == ['"First paragraph."', '"Second, with ä and α."', "x"]


def test_a_quote_in_text_does_not_end_the_string(session, file):
    Sheet().text(r'{\rtf1 He said "yes".\par}').write(file)
    session.load(file)
    assert texts(session) == ["\"He said 'yes'.\""]


def test_label_references_follow_expressions_past_the_text(session, file):
    """Text takes numbers here, so `#1` in the file is entry 2 once loaded."""
    sheet = Sheet().text(r"{\rtf1 Title\par}").expression("x^2")
    sheet.expression("#1+1").expression("x^2+1", "Simp(#2)").write(file)
    session.load(file)
    assert texts(session) == ['"Title"', "x^2", "#2+1", "x^2+1"]
    assert annotations(session)[3] == "Simp(#3)"


def test_merging_numbers_the_worksheet_after_what_is_there(session, file):
    session.author("y")
    Sheet().expression("x").expression("#1+1").write(file)
    session.merge(file)
    assert texts(session) == ["y", "x", "#2+1"]


def test_a_utf16_annotation_and_expression_are_read(session, file):
    Sheet().expression("λ+1", "Benutzer", wide=True).write(file)
    session.load(file)
    # An entry's text spells a Greek letter by name.
    assert texts(session) == ["lambda+1"]
    assert annotations(session) == ["User"]


def test_utf16_text_is_read(session, file):
    Sheet().text(RTF, wide=True).write(file)
    session.load(file)
    assert texts(session)[1] == '"Second, with ä and α."'


def test_a_rule_is_text_too(session, file):
    Sheet().text(r"{\rtf1 Ruled\par}", kind="CRuleObj").write(file)
    session.load(file)
    assert texts(session) == ['"Ruled"']


def test_derive_6_characters_in_an_expression_are_read(session, file):
    """Version 6 escapes what code page 1252 lacks, and has its own `#e` and `#i`."""
    Sheet().expression(b"\x7fe7c0^(\x7fe7c1\x7f00b7x)+\x7f221a2").write(file)
    session.load(file)
    assert texts(session) == ["#e^(#i*x)+SQRT(2)"]


def test_derive_5_codes_in_an_expression_are_read(session, file):
    Sheet(OLD).expression(b"v\x99 2\x93\xb9").write(file)
    session.load(file)
    assert texts(session) == ["v SUB 2<=pi"]


def test_derive_5_greek_letters_in_the_state_are_the_names_derive_6_saves(session, file):
    """0xDE 0xBF 0xAD are sigma, lambda and epsilon, as aligning the definitions that
    version 5 and 6 worksheets both save shows; 0xDF is beta, not a sharp s."""
    Sheet(OLD, state=[b"\xde(\xbf, \xad):=\xad + \xbf", b"\xdf:=2 + \x8b2"]).write(file)
    assert not session.load(file)
    assert session.functions["σ"].names == ("λ", "ε")
    assert list(session.assignments) == ["β"]


async def test_derive_5_not_sign_and_greek_letters_in_the_state(session, file):
    """0xAC is the NOT sign, as in ISO 8859-1 (`IF(¬ DIM(a) = n ∨ ¬ DIM(b) = n, ...)`);
    0xBA, 0xB4 and 0xA5 are rho, kappa and capital phi, the letters the expressions of
    the same worksheets spell as `rho`, `kappa` and `PHI`."""
    state = [
        b"t(a, n):=IF(\xac DIM(a) = n \x90 \xac n > 0, 1, 2)",
        b"\xba(\xb4):=\xb4 + 1",
    ]
    Sheet(OLD, state=[*state, b"\xa5:=7"]).write(file)
    assert not session.load(file)
    assert session.functions["ρ"].names == ("κ",)
    assert (await session.simplify("[t([1], 1), t([1], 2), ρ(1)]")).text == "[2, 1, 2]"
    assert list(session.assignments) == ["φ"]


def test_derive_5_sharp_s_in_a_word_or_string_is_a_sharp_s(session, file):
    """0xDF between two letters of a word, or in a string, is the sharp s a German
    user typed, not beta."""
    Sheet(OLD).expression(b'["gr\xf6\xdfter Wert:", \xdf0_]').write(file)
    session.load(file)
    assert texts(session) == ['["größter Wert:",β0_]']


def test_plots_and_embedded_objects_are_left_out_and_counted(session, file):
    sheet = Sheet().expression("a").plot().embedded().expression("b")
    sheet.text(r"{\rtf1 After\par}").plot().expression("c").write(file)
    assert str(session.load(file)) == "2 plots and 1 embedded object not loaded"
    assert texts(session) == ["a", "b", '"After"', "c"]


def test_the_worksheet_is_saved_as_a_math_file(session, file):
    """Saving never writes text over the binary original."""
    Sheet().expression("a").write(file)
    session.load(file)
    assert session.file == file.with_suffix(".mth")


def test_an_expression_that_does_not_parse_is_kept_as_a_string(session, file):
    """It keeps its number, so a later reference to the expression after it still
    means that expression."""
    Sheet().expression("a").expression("2 +", "Simp(#1)").expression("#3").write(file)
    report = session.load(file)
    assert texts(session) == ["a", '"2 +"', "#3"]
    assert annotations(session) == ["User", "Simp(#1)", "User"]
    assert str(report) == "1 line kept as strings (#2)"


# -- the saved state -------------------------------------------------------


async def test_definitions_saved_with_the_worksheet_are_restored(session, file):
    """A utility file's functions are saved in the state block, though never shown."""
    Sheet(state=["CUBE(t):=t^3", "k:=5"]).expression("CUBE(k)").write(file)
    session.load(file)
    assert texts(session) == ["CUBE(k)"]
    assert (await session.simplify("#1")).text == "125"


async def test_the_saved_state_overrides_what_the_expressions_define(session, file):
    """The state block was saved after everything shown, so it has the last word."""
    sheet = Sheet(state=["f(x):=x^3", "a:=2"])
    sheet.expression("f(x):=x^2").expression("a:=1").write(file)
    session.load(file)
    assert (await session.simplify("f(a)")).text == "8"


async def test_what_the_saved_session_no_longer_had_is_forgotten(session, file):
    sheet = Sheet(state=["g(x):=x+1"]).expression("a:=7").expression("g(x):=x+1")
    sheet.write(file)
    session.load(file)
    assert "a" not in session.assignments
    assert (await session.simplify("a+g(1)")).text == "a + 2"


async def test_nothing_is_forgotten_when_the_state_cannot_all_be_read(session, file):
    """An unreadable state line might have defined the name, so it is kept."""
    sheet = Sheet(OLD, state=[b"\x81:=1", b"g(x):=x+1"]).expression("a:=7")
    sheet.write(file)
    session.load(file)
    assert (await session.simplify("a+g(1)")).text == "9"


async def test_a_state_line_with_an_unknown_derive_5_code_is_left_out(session, file):
    Sheet(OLD, state=[b"\x81:=1", b"SQ(u):=u\x93 2"]).write(file)
    session.load(file)
    assert list(session.assignments) == []
    assert (await session.simplify("SQ(1)")).text == "true"


def test_the_saved_settings_are_restored(session, file):
    saved = [
        line.replace("Rational", "Decimal").replace("[x,y,z]", "[y,x]")
        for line in SETTINGS
    ]
    Sheet(state=saved).expression("x").write(file)
    session.load(file)
    assert session.settings["Notation"] == "Decimal"
    assert session.order == ("y", "x")


def test_a_setting_rederive_lacks_is_ignored(session, file):
    """`DisplaySteps` is a Derive 6 setting, not a variable to define."""
    Sheet(state=SETTINGS).write(file)
    session.load(file)
    assert "displaysteps" not in session.assignments
    assert not session.state.variables


def test_the_expressions_read_with_the_saved_case_mode(session, file):
    sheet = Sheet(state=["CaseMode:=Sensitive", "A:=1", "a:=2"]).expression("A+a")
    sheet.write(file)
    session.load(file)
    assert session.assignments.keys() >= {"A", "a"}


def test_the_saved_settings_apply_before_the_definitions(session, file):
    """They are saved last, but the definitions were saved under them."""
    Sheet(state=["A:=1", "a:=2", "CaseMode:=Sensitive"]).expression("A+a").write(file)
    assert not session.load(file)
    assert session.assignments.keys() >= {"A", "a"}
    assert texts(session) == ["A+a"]


async def test_a_definition_may_use_a_function_defined_further_down(session, file):
    """The state block is sorted by name, so the order of definition is lost."""
    state = ["EXTREM(a):=EXTR_ONE(a, 2) + F_ONE(a)", "EXTR_ONE(a, b):=a*b", "F_ONE(u):=u"]
    Sheet(state=state).expression("EXTREM(3)").write(file)
    assert not session.load(file)
    assert (await session.simplify("#1")).text == "9"


async def test_a_state_line_that_does_not_parse_is_reported_by_its_name(session, file):
    """The rest of the state is restored, and the name is no longer declared."""
    Sheet(state=["F(x):=x+", "G(x):=F(x, 1)", "k:="]).write(file)
    report = session.load(file)
    assert str(report) == "1 definition not restored (F)"
    assert "G" in session.functions
    assert "F" not in session.state.functions


def test_an_undecodable_state_line_is_counted_without_a_name(session, file):
    Sheet(OLD, state=[b"\x81:=1", b"g(x):=x+"]).write(file)
    assert str(session.load(file)) == "2 definitions not restored (g)"


def test_a_saved_setting_rederive_lacks_is_reported(session, file):
    Sheet(state=[*SETTINGS, "ScientificThreshold:=5"]).write(file)
    assert str(session.load(file)) == "1 setting ignored"


async def test_a_utility_worksheet_reports_the_lines_it_could_not_read(session, file):
    Sheet(state=["CUBE(t):=t^3", "SQ(t):=t^", "Notation:=Decimal"]).write(file)
    assert str(session.load_utility(file)) == "1 line not readable (SQ)"
    assert (await session.simplify("CUBE(2)")).text == "8"


async def test_a_utility_worksheet_gives_its_definitions_but_not_its_settings(
    session, file
):
    Sheet(state=["CUBE(t):=t^3", "Notation:=Decimal"]).expression("x").write(file)
    assert not session.load_utility(file)
    assert session.entries == []
    assert session.settings["Notation"] == "Rational"
    assert (await session.simplify("CUBE(2)")).text == "8"


async def test_a_function_name_beside_an_operand_in_the_state_multiplies(session, file):
    """The state is written as Derive displays it, where only a built-in applies to
    an operand without parentheses: a function defined further down doesn't make the
    product a call."""
    Sheet(state=["g(x):=k x + d", "k(x):=x^2"]).write(file)
    assert not session.load(file)
    assert write_expression(session.functions["g"].body) == "k*x+d"
    assert (await session.simplify("k(3)")).text == "9"


def test_where_a_plot_s_cross_was_defines_nothing(session, file):
    state = [
        "hCross:=APPROX(39615384615384617/10000000000000000)",
        "vCross:=1",
        "zCross:=2",
    ]
    Sheet(state=state).write(file)
    assert not session.load(file)
    assert not session.assignments and not session.state.variables


def test_a_declaration_repeated_is_one(session, file):
    Sheet(state=["k:=", "k:=", "k:=", '"$by1":=']).write(file)
    assert not session.load(file)
    assert "k" in session.state.variables


async def test_domain_lines_in_every_notation(session, file):
    state = [
        "d:epsilon Real (-0.5, 0.5)",
        "x:epsilon Real (0, ∞)",
        "n:epsilonInteger [0, inf)",
        "u:epsilon Vector",
        "NIL:epsilon Error",
    ]
    Sheet(state=state).write(file)
    assert not session.load(file)
    answer = await session.simplify("[SQRT(x^2), SQRT((d - 1)^2), SIN(n*pi)]")
    assert answer.text == "[x, 1 - d, 0]"
    assert {"u", "NIL"} <= set(session.domains)


async def test_a_setting_assigned_in_a_saved_body_is_left_for_the_body(session, file):
    state = [
        "F(t_):=PROG(t_ ≔ Trigonometry, Trigonometry ≔ Collect, Trigonometry ≔ t_)",
        *SETTINGS,
    ]
    Sheet(state=state).write(file)
    assert not session.load(file)
    assert session.settings["Trigonometry"] == "Auto"


async def test_a_saved_definition_of_a_built_in_is_refused(session, file):
    Sheet(state=["EI(x, m):=\ue7c3 + LN(x) + ∑(x^n_/(n_·n_!), n_, 1, m)"]).write(file)
    assert str(session.load(file)) == "1 refused as built-in (EI)"
    assert (await session.simplify("INT(EXP(x)/x, x)")).text == "EI(x)"
    assert (await session.simplify("EI(x, 1)")).text == "LN(x) + x + euler_gamma"


# -- what is read as what --------------------------------------------------


def test_a_worksheet_named_as_a_math_file_is_read_as_a_worksheet(session, tmp_path):
    path = Sheet().expression("x").write(tmp_path / "binary.mth")
    session.load(path)
    assert texts(session) == ["x"]


def test_a_math_file_named_as_a_worksheet_is_read_as_text(session, file):
    file.write_text("x+1\n\ny\n", encoding="utf-8")
    session.load(file)
    assert texts(session) == ["x+1", "y"]


def test_a_math_file_decodes_derive_6_escapes(session, tmp_path):
    path = tmp_path / "escaped.mth"
    path.write_bytes(b"\x7f2211(k,k,1,n)\r\n\r\n\x7fe7c0^x\r\n\r\n\"\x7f03b1\"\r\n")
    session.load(path)
    assert texts(session) == ["SUM(k,k,1,n)", "#e^x", '"α"']


def test_an_escape_broken_across_a_continuation_is_decoded(session, tmp_path):
    """Derive breaks long lines by length, even in the middle of an escape."""
    path = tmp_path / "broken.mth"
    path.write_bytes(b"x\x7f22~\r\n64y\r\n")
    session.load(path)
    assert texts(session) == ["x<=y"]


def test_a_truncated_worksheet_is_refused_and_the_history_kept(session, file):
    data = Sheet().expression("a").expression("b").bytes()
    session.author("x")
    for cut in (len(MODERN) + 2, len(data) // 2, len(data) - 3):
        file.write_bytes(data[:cut])
        with pytest.raises(OSError):
            session.load(file)
    assert texts(session) == ["x"]


def test_bytes_after_the_trailer_are_refused(session, file):
    file.write_bytes(Sheet().expression("a").bytes() + b"\x00")
    with pytest.raises(dfw.Damaged):
        session.merge(file)


def test_an_impossible_count_is_refused(session, file):
    data = bytearray(Sheet().expression("a").bytes())
    data[len(MODERN) : len(MODERN) + 4] = struct.pack("<I", 0xFFFFFFFF)
    file.write_bytes(bytes(data))
    with pytest.raises(dfw.Damaged):
        session.load(file)


def test_a_plot_whose_end_cannot_be_found_is_refused(session, file):
    Sheet().expression("a").plot(snapshot=False).expression("b").write(file)
    with pytest.raises(dfw.Damaged, match="plot"):
        session.load(file)


def test_an_unknown_class_is_refused(session, file):
    data = Sheet().expression("a").bytes().replace(b"CExpnObj", b"CMystery")
    file.write_bytes(data)
    with pytest.raises(dfw.Damaged, match="CMystery"):
        session.load(file)


def test_the_command_line_reads_a_worksheet_and_reports_a_damaged_one(
    session, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    Sheet().expression("x").write(tmp_path / "good.dfw")
    (tmp_path / "bad.dfw").write_bytes(MODERN + b"\x05")
    assert named(["good"]) == [("math", Path("good.dfw"))]
    read(session, named(["good.dfw"]))
    assert texts(session) == ["x"]
    with pytest.raises(Usage, match="bad.dfw: not a readable Derive for Windows"):
        read(session, named(["bad.dfw"]))


def test_the_load_prompt_offers_worksheets_of_both_kinds(tmp_path):
    for name in ("a.mth", "b.dfw", "c.dat"):
        (tmp_path / name).write_text("")
    assert worksheet.matches(f"{tmp_path}/", worksheet.READABLE) == [
        f"{tmp_path}/a.mth",
        f"{tmp_path}/b.dfw",
    ]
    assert worksheet.matches(f"{tmp_path}/") == [f"{tmp_path}/a.mth"]


async def test_transfer_load_reports_a_damaged_worksheet(file, monkeypatch):
    monkeypatch.chdir(file.parent)
    file.write_bytes(MODERN)
    app = RederiveApp()
    async with app.run_test() as pilot:
        await pilot.press("t", "l", "d", *file.name, "enter")
        assert message(app) == "Cannot read file"
        assert entries(app) == []


# -- pieces ----------------------------------------------------------------


def test_rtf_keeps_the_text_and_drops_the_rest():
    assert dfw.rtf_paragraphs(RTF) == ("First paragraph.", "Second, with ä and α.")


def test_rtf_skips_hidden_groups_and_reads_escaped_symbols():
    document = r"{\rtf1{\*\generator Msftedit;}a\{b\}\\c\~d\-e\par\par{\pict 0101}f}"
    assert dfw.rtf_paragraphs(document) == (r"a{b}\c de", "f")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("v↓2", "v SUB 2"),
        ("x ≔ 1", "x:=1"),
        ("∑(k, k, 1, n)·∏(k, k, 1, n)", "SUM(k,k,1,n)*PRODUCT(k,k,1,n)"),
        ("∂(x^2, x) + ∫(x, x)", "DIF(x^2,x)+INT(x,x)"),
        ("¬ p ∧ q ∨ r → s", "(NOT(p) AND q) OR r IMP s"),
        ("a ≠ b", "a/=b"),
        ("u ⋅ v", "u . v"),
        ("(3) +  + ", "GAMMA(3)+euler_gamma+pi"),
    ],
)
def test_derive_6_spellings_read_as_the_names_they_stand_for(text, expected):
    state = ParseState()
    state.declare(SettingDeclaration("InputMode", str(InputMode.WORD)))
    assert write_expression(parse_expression(text, state).node) == expected
