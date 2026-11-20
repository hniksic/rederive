"""`LOAD`, and what reading Derive's math files keeps: annotations in every language,
strings, the files a `LOAD` names on a machine they were not made on, data files,
demonstrations, saving with the settings, and definitions of names a file may not
take.

Each test runs in a temporary directory of its own, which is the working directory, so
files can be named without a path.
"""

import pytest

from rederive.model import worksheet
from rederive.model.session import Session


@pytest.fixture(autouse=True)
def here(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def session():
    return Session()


def texts(session):
    return [entry.text for entry in session.entries]


def annotations(session):
    return [entry.annotation for entry in session.entries]


def math_file(path, *lines):
    """Write a math file as Derive writes one: records of CR LF lines, a blank line
    between them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("\r\n\r\n".join(lines).encode("utf-8") + b"\r\n")
    return path


# -- annotations ----------------------------------------------------------------------


def test_a_typed_annotation_in_any_language_is_the_users(session, tmp_path):
    math_file(
        tmp_path / "a.mth",
        ";Benutzer=Simp(Benutzer)\r\nHESSE(0,0)=[[-2*b^2,0],[0,-2*a^2]]",
        ";Utilisateur\r\nx^2",
        ";Nueva\r\nx",
    )
    assert not session.load(tmp_path / "a.mth")
    assert annotations(session) == ["User", "User", "User"]


def test_a_reference_in_a_localized_annotation_follows_a_merge(session, tmp_path):
    math_file(
        tmp_path / "b.mth", "xo=0", ";Simp(L\x7f00f6se(#1,[xo,yo]))\r\nxo=0 OR yo=b"
    )
    for text in ("a", "b", "c"):
        session.author(text)
    assert not session.merge(tmp_path / "b.mth")
    assert annotations(session)[3:] == ["User", "Simp(Löse(#4,[xo,yo]))"]


def test_a_damaged_reference_is_kept_as_written(session, tmp_path):
    """Derive replaced a deleted `#102` by `User` and left `02` behind; the digits are
    no reference."""
    math_file(tmp_path / "c.mth", ";Simp(User02)\r\n7")
    for text in ("a", "b", "c"):
        session.author(text)
    assert not session.merge(tmp_path / "c.mth")
    assert annotations(session)[3:] == ["Simp(User02)"]


def test_free_text_and_the_empty_annotation_are_kept(session, tmp_path):
    math_file(tmp_path / "d.mth", ";Interest/Discount\r\nID(i,nth,mth):=i", ";\r\nx")
    assert not session.load(tmp_path / "d.mth")
    assert annotations(session) == ["Interest/Discount", ""]
    session.save(tmp_path / "e.mth")
    assert (tmp_path / "e.mth").read_text() == (
        ";Interest/Discount\nID(i,nth,mth):=i\n\n;\nx\n"
    )


# -- strings ------------------------------------------------------------------------


def test_a_backslash_in_a_string_is_a_character(session, tmp_path):
    math_file(tmp_path / "a.mth", 'LOAD("D:\\DfD\\DNL\\DNL93\\MTH12\\MaxMin_ut.mth")')
    assert str(session.load(tmp_path / "a.mth")) == "MaxMin_ut.mth could not be found"
    assert texts(session) == ['LOAD("D:\\DfD\\DNL\\DNL93\\MTH12\\MaxMin_ut.mth")']


def test_a_string_line_is_an_entry_but_no_definition_of_a_utility(session, tmp_path):
    math_file(tmp_path / "b.mth", '"Circles1"', "VECTOR(SQRT(1-(x+5-2*a)^2)-5,a,0,5)")
    assert not session.load(tmp_path / "b.mth")
    assert texts(session) == ['"Circles1"', "VECTOR(SQRT(1-(x+5-2*a)^2)-5,a,0,5)"]
    other = Session()
    assert not other.load_utility(tmp_path / "b.mth")
    assert texts(other) == [] and not other.assignments and not other.functions


def test_strings_of_a_backslash_and_a_tilde_survive_the_wrap(session, tmp_path):
    (tmp_path / "f.mth").write_bytes(
        b'L:=[["~",[[1.6,7],[inf,~\r\ninf]]]]\r\n\r\n'
        b'char_names_n1:=["#","$","@","[","\\","]","^","`","{","|","}","~"]\r\n\x1a'
    )
    assert not session.load(tmp_path / "f.mth")
    names = session.entries[1].node.children[1]
    assert [child.value for child in names.children] == list("#$@[\\]^`{|}~")
    assert texts(session)[0] == 'L:=[["~",[[1.6,7],[inf,inf]]]]'


async def test_a_semicolon_in_a_wrapped_string_is_data(session, tmp_path):
    (tmp_path / "h.mth").write_bytes(
        b'o1:=APPEND(["Ho:"],["s(x1-x2~\r\n);","Z:"])\r\n'
    )
    assert not session.load(tmp_path / "h.mth")
    assert (await session.simplify("o1")).text == '["Ho:", "s(x1-x2);", "Z:"]'


def test_a_tilde_at_the_end_of_a_piece_is_written_twice(session, tmp_path):
    session.settings.assign("SaveLength", "5")
    session.author('"ab~cd"')
    session.save(tmp_path / "t.mth")
    assert (tmp_path / "t.mth").read_text() == '"ab~~\ncd"\n'
    other = Session()
    other.load(tmp_path / "t.mth")
    assert texts(other) == ['"ab~cd"']


# -- LOAD on entry -------------------------------------------------------------------


def test_a_load_line_of_a_file_loads_beside_the_file(tmp_path):
    folder = tmp_path / "D"
    math_file(folder / "ALL.MTH", "DOW(d,m,y):=d")
    math_file(
        folder / "a.mth",
        'LOAD("C:\\Users\\Kcozart\\Desktop\\New folder (5)\\ALL.MTH")',
        "DOW(7,12,52)",
    )
    session = Session()
    assert not session.load(folder / "a.mth")
    assert "DOW" in session.functions
    assert len(session.entries) == 2


async def test_a_relative_path_without_an_extension_finds_the_file_in_any_case(
    tmp_path,
):
    folder = tmp_path / "D"
    math_file(folder / "VECTOR.MTH", "OUTER(v,w):=[v,w]")
    math_file(folder / "b.mth", 'LOAD("..\\math\\vector")')
    session = Session()
    assert not session.load(folder / "b.mth")
    assert (await session.simplify("OUTER(1, 2)")).text == "[1, 2]"


async def test_a_typed_load_finds_the_file_in_the_math_directory(session, tmp_path):
    math_file(tmp_path / "Math" / "NumberTheoryFunctions.mth", "NTH_PRIME(n):=n+1000")
    session.author('LOAD("C:\\Derive6\\Math\\NumberTheoryFunctions.mth")')
    assert session.take_notice() is None
    assert (await session.simplify("#1")).text == "true"
    assert (await session.simplify("NTH_PRIME(1)")).text == "1001"


async def test_a_file_name_is_not_translated(session, tmp_path):
    math_file(tmp_path / "Math" / "NumberTheoryFunctions.mth", "NTH_PRIME(n):=n+1000")
    session.author('LOAD("D:\\DERIVE\\NUMBER.MTH")')
    assert session.take_notice() == "NUMBER.MTH could not be found"
    assert (await session.simplify("#1")).text == 'LOAD("D:\\DERIVE\\NUMBER.MTH")'
    assert session.take_notice() == "NUMBER.MTH could not be found"


def test_only_math_files_and_worksheets_are_loaded(session, tmp_path):
    (tmp_path / "notes.txt").write_text("F(x):=x\n")
    session.author('LOAD("notes.txt")')
    assert session.take_notice() == "notes.txt could not be found"
    assert "F" not in session.functions


async def test_a_loaded_file_takes_definitions_and_settings_but_enters_nothing(
    session, tmp_path
):
    math_file(tmp_path / "c.mth", "F(x):=x+1", '"comment"', "Precision:=Approximate")
    session.author("F(x) := 0")
    session.author('LOAD("c.mth")')
    assert texts(session) == ["F(x):=0", 'LOAD("c.mth")']
    assert session.settings["Precision"] == "Approximate"
    assert (await session.simplify("F(2)")).text == "3"


def test_a_file_that_loads_itself_is_loaded_once(session, tmp_path):
    math_file(tmp_path / "d.mth", 'LOAD("d.mth")', "G(x):=x")
    session.author('LOAD("d.mth")')
    assert session.take_notice() is None
    assert "G" in session.functions


async def test_a_vector_of_loads_and_other_things_loads_when_simplified(
    session, tmp_path
):
    math_file(tmp_path / "c.mth", "F(x, y):=x+y")
    session.author('[LOAD("c.mth"), F(1, 1)]')
    assert "F" not in session.functions
    assert (await session.simplify("#1")).text == "[true, 2]"
    assert "F" in session.functions


def test_a_missing_file_does_not_stop_the_file_that_loads_it(session, tmp_path):
    math_file(
        tmp_path / "DS-GAUSS.MTH",
        '"Elementary matrix program. Copyright 1998 by David Stegenga."',
        "[[],a,1,1,1,0]",
        "ROW_SWAP(v,k):=",
        "ROW_SCALE(v):=",
        'LOAD("..\\math\\vector")',
        "COLS(a):=DIMENSION(a`)",
        "ROWS(a):=DIMENSION(a)",
    )
    report = session.load(tmp_path / "DS-GAUSS.MTH")
    assert str(report) == "vector could not be found"
    assert len(session.entries) == 7
    assert {"ROW_SWAP", "ROW_SCALE", "COLS", "ROWS"} <= set(session.functions)


async def test_a_load_of_a_worksheet_takes_its_state(session, tmp_path):
    from dfwsheet import Sheet

    Sheet(state=["CUBE(t):=t^3", "Notation:=Decimal"]).expression("x").write(
        tmp_path / "w.dfw"
    )
    session.author('LOAD("w")')
    assert (await session.simplify("CUBE(2)")).text == "8"
    assert session.settings["Notation"] == "Rational"


# -- definitions a file may not make ------------------------------------------------


async def test_a_utility_file_s_definitions_of_built_ins_are_refused(
    session, tmp_path
):
    math_file(
        tmp_path / "n.mth",
        "CEILING(z,w):=-FLOOR(-z,w)",
        "ROUND(z,w):=FLOOR(z/w+1/2)",
        "NTH_PRIME(n):=n+1000",
    )
    report = session.load_utility(tmp_path / "n.mth")
    assert str(report) == "2 refused as built-in (CEILING, ROUND)"
    assert (await session.simplify("NTH_PRIME(1)")).text == "1001"
    assert (await session.simplify("CEILING(573, 100)")).text == "6"


async def test_a_load_reports_the_built_ins_its_file_defines(session, tmp_path):
    math_file(tmp_path / "v.mth", "RANK(m):=0", "ADJOIN_ELEMENT(e,v):=APPEND([e],v)")
    session.author('LOAD("v.mth")')
    assert session.take_notice() == "1 refused as built-in (RANK)"
    assert (await session.simplify("RANK([[1, 2], [2, 4]])")).text == "1"
    assert (await session.simplify("ADJOIN_ELEMENT(a, [b, c])")).text == "[a, b, c]"


async def test_a_vector_of_definitions_makes_all_but_the_refused_ones(session, tmp_path):
    math_file(
        tmp_path / "d.mth",
        "[G(x) := x + 1, SORT(v) := v, H(x) := 2*x]",
        "G(3)",
    )
    report = session.load(tmp_path / "d.mth")
    assert str(report) == "1 refused as built-in (SORT)"
    assert [entry.text for entry in session.entries] == [
        '"[G(x) := x + 1, SORT(v) := v, H(x) := 2*x]"',
        "G(3)",
    ]
    assert (await session.simplify("[G(3), H(3)]")).text == "[4, 6]"
    assert (await session.simplify("SORT([2, 1])")).text == "[1, 2]"
    other = Session()
    report = other.load_utility(tmp_path / "d.mth")
    assert str(report) == "1 refused as built-in (SORT)"
    assert (await other.simplify("H(4)")).text == "8"


async def test_definitions_of_special_functions_are_refused(session, tmp_path):
    math_file(
        tmp_path / "z.mth", "DILOG(z):=z", "POLYLOG(n,z,m):=SUM(z^k/k^n,k,1,m)"
    )
    math_file(
        tmp_path / "p.mth",
        "POLYGAMMA(n,z):=IF(n=-1,LN(GAMMA(z)),IF(n=0,DIGAMMA(z),"
        "(-1)^(n+1)*n!*ZETA(n+1,z)))",
    )
    report = session.load_utility(tmp_path / "z.mth")
    assert str(report) == "2 refused as built-in (DILOG, POLYLOG)"
    assert (await session.simplify("POLYLOG(2, z, 3)")).text == "z^3/9 + z^2/4 + z"
    report = session.load_utility(tmp_path / "p.mth")
    assert str(report) == "1 refused as built-in (POLYGAMMA)"
    assert (await session.simplify("POLYGAMMA(-1, 2)")).text == "0"
    assert (await session.simplify("DIF(DIGAMMA(x), x)")).text == "POLYGAMMA(1, x)"


async def test_another_spelling_of_a_reserved_name_is_the_users(session):
    session.author("ei(x) := x + 1")
    assert (await session.simplify("ei(2)")).text == "3"


# -- data files and demonstrations ----------------------------------------------------


def test_a_data_file_may_annotate_and_separate_by_semicolons(session, tmp_path):
    (tmp_path / "a.dat").write_bytes(b";Measured values\r\n1;2;3\r\n4 , 5 , 6\r\n")
    assert not session.load_data(tmp_path / "a.dat")
    assert texts(session) == ["[[1,2,3],[4,5,6]]"]
    assert annotations(session) == ["Measured values"]


def test_a_data_file_may_separate_by_tabs(session, tmp_path):
    (tmp_path / "b.dat").write_bytes(b"1\t2\t3\r\n")
    assert not session.load_data(tmp_path / "b.dat")
    assert texts(session) == ["[1,2,3]"]
    assert annotations(session) == ["User"]


# -- saving -------------------------------------------------------------------------


def test_the_state_option_writes_the_settings_first(session, tmp_path):
    session.settings.assign("Precision", "Approximate")
    session.settings.assign("SaveState", "Save")
    session.author("x")
    session.save(tmp_path / "s.mth")
    written = (tmp_path / "s.mth").read_text()
    assert written.startswith(
        "Precision:=Approximate\n\nPrecisionDigits:=10\n\nNotation:=Scientific\n\n"
    )
    assert written.endswith(
        "InputMode:=Character\n\nDisplayFormat:=Normal\n\nTimesOperator:=Dot\n\nx\n"
    )
    assert "VariableOrder:=[x,y,z]\n\nOutputBase:=Decimal\n\n" in written
    other = Session()
    other.load(tmp_path / "s.mth")
    assert other.settings["Precision"] == "Approximate"
    assert len(other.entries) == 18
    other.save(tmp_path / "t.mth")
    assert (tmp_path / "t.mth").read_text() == written


def test_settings_are_not_saved_by_default(session, tmp_path):
    session.author("x")
    session.save(tmp_path / "s.mth")
    assert (tmp_path / "s.mth").read_text() == "x\n"


def test_a_name_beyond_ascii_is_written_as_itself(session, tmp_path):
    session.author("θ := 1")
    session.save(tmp_path / "g.mth")
    assert (tmp_path / "g.mth").read_bytes() == "θ:=1\n".encode()


def test_a_saved_file_reads_back_as_it_was(session, tmp_path):
    """Derive 6's constructs are written the way Derive 6 writes them, so they read
    back to the same entries."""
    lines = [
        "f(x, y := 0) := PROG(k := 0, LOOP(IF(k > x, exit), k :+ 1), RETURN k)",
        "x :epsilon Real (0, ∞)",
        "[a, b; c, d]",
        "u ROW 2",
        "u COL 1",
        "[1, 2] DOTPRODUCT [3, 4]",
        "PRIME?(7)",
        "a SUB 1 SUB 2",
        '"C:\\x"',
        "x = 1 OR y = 2 AND NOT z",
        "g(v, n := 2) := PROG(n :* 2, v SUB 1 :- n, v SUB n := 0, v)",
        "B v := v SUB DIM(v)",
        "p IFF q",
        "{1, 2} UNION {3} SUBSET {1, ..., 5} \\ {4}",
        "u CROSSPRODUCT v",
        "m↓↓2",
        "FIRST(t) SUB 2",
        "a ± b",
        "45°",
        "(m := 3) AND c := 4",
        "h(a, b) := '(a/b) + 'IF(a > 0, ['(b SUB 1), \"s\"])",
        "[''(1 + 2), 'x^2]",
    ]
    for line in lines:
        session.author(line)
    session.save(tmp_path / "r.mth")
    other = Session()
    assert not other.load(tmp_path / "r.mth")
    assert texts(other) == texts(session)
    other.save(tmp_path / "s.mth")
    assert (tmp_path / "s.mth").read_bytes() == (tmp_path / "r.mth").read_bytes()
    written = (tmp_path / "r.mth").read_text()
    assert ";" not in written.replace('"', "")
    for spelling in [
        "B v:=v SUB DIM(v)",
        "m SUB  SUB 2",
        "(FIRST(t)) SUB 2",
        'a+("+-"b)',
        "45 DEG",
        "(m:=3) AND c:=4",
        "h(a,b):='(a/b)+'IF(a>0,['(b SUB 1),\"s\"])",
        "[''(1+2),'x^2]",
    ]:
        assert spelling in written, spelling


def test_a_worksheet_saved_says_its_plots_are_lost_once(session, tmp_path):
    from dfwsheet import Sheet

    Sheet().expression("x").plot().write(tmp_path / "p.dfw")
    assert str(session.load(tmp_path / "p.dfw")) == "1 plot not loaded"
    session.save(tmp_path / "p.mth")
    assert session.take_notice() == "plots and embedded objects are not saved"
    session.save(tmp_path / "p.mth")
    assert session.take_notice() is None


def test_located_tries_the_given_path_then_the_name_alone(tmp_path):
    math_file(tmp_path / "x" / "Fastf.MTH", "F(x):=x")
    written = "C:\\DfD\\Mth00\\compiler\\Fastf.mth"
    assert worksheet.located(written, [tmp_path / "x"]) == tmp_path / "x" / "Fastf.MTH"
    assert worksheet.located("x/fastf", []) == worksheet.Path("x/Fastf.MTH")
    assert worksheet.located("fastf", []) is None
    assert worksheet.located("C:\\", []) is None


async def test_a_demonstration_step_loads_beside_the_demonstration(session, tmp_path):
    folder = tmp_path / "demos"
    math_file(folder / "vector.mth", "OUTER(v,w):=VECTOR(VECTOR(a_*b_,b_,w),a_,v)")
    demo = math_file(
        folder / "b.dmo",
        ';Load the vector utility first\r\nLOAD("vector")',
        ";Outer product\r\nOUTER([1,2],[3,4])",
    )
    steps = worksheet.demonstration(demo)
    assert [comment for comment, _ in steps] == [
        "Load the vector utility first",
        "Outer product",
    ]
    for _, text in steps:
        with session.reading_file(demo):
            session.author(text)
    assert session.take_notice() is None
    assert (await session.simplify("#2")).text == "[[3, 4], [6, 8]]"


# -- names read before their definitions --------------------------------------------


async def test_a_greek_letter_called_before_its_definition_is_a_call(session, tmp_path):
    """In a file a name directly followed by `(` is a call, a Greek letter's name too:
    `alpha` and `beta` are defined each in terms of the other."""
    math_file(
        tmp_path / "r.mth",
        "S(x):=x+1",
        "PRED(x):=IF(x=0,0,x-1)",
        "alpha(x):=IF(x=0,0,IF(alpha(PRED(x))<beta(PRED(x)),beta(PRED(x)),"
        "IF(alpha(PRED(x))>beta(PRED(x)),S(beta(PRED(x))),0)))",
        "beta(x):=IF(x=0,0,IF(alpha(PRED(x))=beta(PRED(x)),S(alpha(PRED(x))),"
        "alpha(PRED(x))))",
        "psi(x):=[alpha(x),beta(x)]",
    )
    assert not session.load(tmp_path / "r.mth")
    answer = await session.simplify("VECTOR(psi(n), n, 0, 5)")
    assert answer.text == "[[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [2, 0]]"


async def test_a_parameter_assigned_in_a_body_leaves_a_function_of_its_name(session):
    """The assignment is to the local: the function `g` is still the function."""
    with session.reading_file():
        session.author("g(y):=y+1")
        session.author("k(n,g,l:=[]):=PROG(g:=n,g)")
    assert (await session.simplify("g(2)")).text == "3"

