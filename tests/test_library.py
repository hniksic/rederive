"""The pre-defined utility library: Derive's utility functions in every new session.

The mechanism first: every text reads as a definition, the names are ordinary
definitions that the user, a file or a worksheet's state replaces, they call each other
by name, a library function is a name only where it is called, and none of it is ever
saved or reported. Then the functions themselves, utility file by utility file, each
case in a fresh session.
"""

import pytest
from dfwsheet import Sheet

from rederive import library
from rederive.engine.context import library_definition
from rederive.model.expr import Kind
from rederive.model.session import Session
from rederive.syntax import CaseMode, ParseState, SettingDeclaration, parse_expression


@pytest.fixture(autouse=True)
def here(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def session():
    return Session()


async def simplified(session, text):
    return (await session.simplify(text)).text


async def answers(*lines):
    """What Simplify answers for each of `lines` in turn, in one fresh session; a line
    starting with `>` is entered, not simplified."""
    session = Session()
    found = []
    for line in lines:
        if line.startswith("> "):
            session.author(line.removeprefix("> "))
        else:
            found.append(await simplified(session, line))
    return found


def math_file(path, *lines):
    path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    return path


# -- the texts ----------------------------------------------------------------------


def test_the_library_has_404_definitions_two_of_them_variables():
    assert len(library.DEFINED) == 404
    assert library.VARIABLES == {"default1", "random_sign"}
    assert len(library.GROUPS) == 404
    assert len(set(library.GROUPS.values())) == 23


@pytest.mark.parametrize("name", sorted(library.DEFINED))
def test_every_text_reads_as_the_definition_of_its_name(name):
    definition = library_definition(name)
    assert definition.body is not None
    node = parse_expression(library.text(name), ParseState()).node
    assert node.kind in (Kind.FUNDEF, Kind.ASSIGN)
    assert (node.value if node.kind is Kind.FUNDEF else node.children[0].value) == name


def test_no_library_name_is_a_built_in_or_reserved():
    from rederive.syntax import reserved
    from rederive.syntax.names import KNOWN_FUNCTIONS

    assert not library.DEFINED & KNOWN_FUNCTIONS
    assert not [name for name in library.DEFINED if reserved(name, function=True)]


def test_the_functions_rederive_builds_in_are_not_defined():
    assert not library.DEFINED & {"EI", "SI", "CI", "POLYGAMMA", "POLYLOG", "DILOG"}


# -- in force, and replaced ---------------------------------------------------------


async def test_a_library_function_is_defined_with_nothing_loaded():
    assert await answers("DIVISORS(6)") == ["[1, 2, 3, 6]"]


async def test_a_definition_replaces_the_library_s_and_the_others_call_it():
    assert await answers(
        "> DIVISORS(n) := [n, 1]",
        "DIVISORS(6)",
        "DIVISOR_TAU(6)",
        "CYCLOTOMIC(2, x)",
        "CYCLOTOMIC(4, x)",
    ) == ["[6, 1]", "4", "x + 1", "x^4 - 1"]


async def test_an_empty_definition_leaves_an_arbitrary_function():
    assert await answers(
        "> DIVISORS(n) := [n, 1]", "> DIVISORS(n) :=", "DIVISORS(6)"
    ) == ["DIVISORS(6)"]


async def test_library_functions_call_each_other_by_name():
    assert await answers(
        "> STIRLING2(n, k) := n + k",
        "STIRLING2(4, 2)",
        "STIRLING_SUBSET(4, 2)",
        "> STIRLING(n, k) := 100",
        "STIRLING_SUBSET(4, 2)",
        "STIRLING1(4, 2)",
    ) == ["6", "7", "100", "100"]


async def test_a_worksheet_s_state_replaces_the_definitions_it_holds(session, tmp_path):
    state = [
        "MOEBIUS_MU(n):=∏(- MAX(2 - v_↓2, 0), v_, FACTORS(n))",
        "SQUAREFREE(n):=MOEBIUS_MU(n)^2 + 100",
    ]
    assert not session.load(Sheet(state=state).expression("x").write(tmp_path / "t.dfw"))
    assert await simplified(session, "MOEBIUS_MU(30)") == "-1"
    assert await simplified(session, "SQUAREFREE(30)") == "101"
    assert await simplified(session, "EULER_PHI(100)") == "40"


async def test_a_worksheet_s_state_keeps_an_old_text_and_the_library_the_rest(
    session, tmp_path
):
    state = [
        "STUDENT(t, v):=1 - INCOMPLETE_BETA(v/(v + t^2), v/2, 1/2)",
        "default1:=1",
        "LINE_AUX(m, x0, y0, x):=m",
    ]
    assert not session.load(Sheet(state=state).expression("x").write(tmp_path / "s.dfw"))
    assert await simplified(session, "STUDENT(1, 1)") == "1/2"
    assert await simplified(session, "LINE_AUX(2, 0, 0, x)") == "2"
    assert await simplified(session, "DIVISORS(12)") == "[1, 2, 3, 4, 6, 12]"


async def test_a_loaded_file_replaces_library_definitions_and_reports_nothing(
    session, tmp_path
):
    math_file(tmp_path / "u.mth", "DIVISORS(n):=[n]", "NTH_PRIME(n):=n+1000")
    assert not session.load_utility(tmp_path / "u.mth")
    assert await simplified(session, "DIVISORS(6)") == "[6]"
    assert await simplified(session, "NTH_PRIME(1)") == "1001"


async def test_a_loaded_file_s_built_ins_are_refused_its_library_names_replaced(
    session, tmp_path
):
    math_file(tmp_path / "e.mth", "EI(x,m):=x+m", "EN(n,z):=n+z")
    assert await simplified(session, "EI(x, 1)") == "LN(x) + x + euler_gamma"
    assert str(session.load_utility(tmp_path / "e.mth")) == "1 refused as built-in (EI)"
    assert await simplified(session, "EI(x, 1)") == "LN(x) + x + euler_gamma"
    assert await simplified(session, "EN(1, 2)") == "3"
    assert await simplified(session, "INT(#e^x/x, x)") == "EI(x)"


async def test_clearing_the_functions_takes_the_library_s_away(session):
    session.clear_functions()
    assert session.context.function("DIVISORS") is None
    assert session.author("DIVISORS(6)").node.kind is not Kind.CALL
    assert await simplified(session, "default1") == "1"
    session.clear_variables()
    assert session.context.assigned("default1") is None
    assert await simplified(session, "default1") == "a*d*e*f*l*t*u"


async def test_a_definition_after_clearing_is_the_user_s(session):
    session.clear_functions()
    session.author("DIVISORS(n) := [n]")
    assert await simplified(session, "DIVISORS(6)") == "[6]"


def test_a_split_window_keeps_what_was_cleared(session):
    session.clear_functions()
    assert session.copy().context.function("DIVISORS") is None


# -- DOS-era files loaded over the library ------------------------------------------


async def test_a_dos_era_text_replaces_the_library_s_and_computes_as_written():
    assert await answers(
        "MOEBIUS_MU(30)",
        "> MOEBIUS_MU(n):=PRODUCT(IF(PRIME(k_),-1,0),k_,FACTORS(FACTOR(n)))",
        "MOEBIUS_MU(30)",
        "EULER_PHI(30)",
    ) == ["-1", "0", "8"]


async def test_the_dos_era_separate_meets_derive_6_factors():
    assert await answers(
        "SEPARATE(x^2*SIN(y), x, y)",
        "> INDEP_OF(f,x):=IF(DIMENSION(SELECT(i_=x,i_,VARIABLES(f))),true,false)",
        "> SEPARATE_HELP(f,x,y):=ITERATE(IF(INDEP_OF(f SUB n,y),[p*f SUB n,q,n+1],"
        "IF(INDEP_OF(f SUB n,x),[p,q*f SUB n,n+1],[0,0,n+1])),[p,q,n],[1,1,1],"
        "DIMENSION(f))",
        "> SEPARATE(u,x,y):=SEPARATE_HELP(FACTORS(u),x,y)",
        "SEPARATE(x^2*SIN(y), x, y)",
    ) == ["[x^2, SIN(y), 3]", "[[x, 2], [SIN(y), 1], 3]"]


async def test_the_library_separates_variables():
    assert await answers(
        "SEPARATE(x*y^3/(1 + x^2), x, y)",
        "SEPARATE(x + y, x, y)",
        "SEP(x*y, -1, x, y, 0, 1)",
        "SEP(x + y, -1, x, y, 0, 1)",
    ) == ["[x/(x^2 + 1), y^3, 4]", "[0, 0, 2]", "LN(y) = x^2/2", '"inapplicable"']


async def test_a_dos_era_file_s_student_replaces_the_library_s(session, tmp_path):
    math_file(
        tmp_path / "p.mth",
        "STUDENT(t,v):=1-INCOMPLETE_BETA(v/(v+t^2),v/2,1/2)",
        "INCOMPLETE_BETA(x,z,w):=x",
    )
    assert await simplified(session, "STUDENT(1, 1)") == "3/4"
    assert not session.load_utility(tmp_path / "p.mth")
    assert await simplified(session, "STUDENT(1, 1)") == "1/2"


async def test_a_definition_of_a_library_name_is_never_reported(session, tmp_path):
    math_file(tmp_path / "d.mth", "DIVISORS(n):=[n]")
    assert not session.load_utility(tmp_path / "d.mth")
    assert await simplified(session, "DIVISORS(6)") == "[6]"


async def test_a_loaded_file_s_helper_is_defined_and_the_library_s_others_stay(
    session, tmp_path
):
    math_file(
        tmp_path / "r.mth",
        "CHEBYSHEV_T_LIST(n,x):=RECURRENCE('(2*x*v_ SUB 1-v_ SUB 2),v_,[x,1],n)",
    )
    assert not session.load_utility(tmp_path / "r.mth")
    # The held recurrence stays held at each pass of the iteration.
    assert await simplified(session, "CHEBYSHEV_T_LIST(4, x)") == (
        "[1, x, 2*x*v_ SUB 1 - v_ SUB 2, 2*x*v_ SUB 1 - v_ SUB 2]"
    )
    assert await simplified(session, "CHEBYSHEV_T_SERIES(x^3, x, -1, 1, 1, 3)") == "3*x/4"


# -- the units files, which the library leaves out ----------------------------------


async def test_metric_units_convert():
    assert await answers(
        "> InputMode := Word",
        "> milli:=0.001",
        "> kilo:=1000",
        "> second:=",
        "> minute:=60*second",
        "> hour:=60*minute",
        "> day:=24*hour",
        "> year:=365.2422*day",
        "> century:=100*year",
        "> newton:=kilo*gram*meter/second^2",
        "> joule:=newton*meter",
        "> erg:=joule/10000000",
        "5*milli*erg/century",
        "APPROX(5*milli*erg/century)",
    ) == [
        "gram*meter^2/(6311385216000000*second^3)",
        "1.584438226*10^-16*gram*meter^2/second^3",
    ]


async def test_english_units_convert_and_the_last_file_s_units_win():
    assert await answers(
        "> InputMode := Word",
        "> second:=",
        "> minute:=60*second",
        "> hour:=60*minute",
        "> day:=24*hour",
        "> week:=7*day",
        "> fortnight:=2*week",
        "> foot:=",
        "> mile:=5280*foot",
        "> furlong:=1/8*mile",
        "5*furlong/fortnight",
        "APPROX(5*furlong/fortnight)",
        "> foot:=0.3048*meter",
        "APPROX(5*furlong/fortnight)",
        "> c_:=299792458*(meter/second)",
        "c_*second/meter",
    ) == [
        "11*foot/(4032*second)",
        "0.002728174603*foot/second",
        "0.0008315476190*meter/second",
        "299792458",
    ]


# -- saving -------------------------------------------------------------------------


async def test_a_save_writes_the_entries_not_the_library(session, tmp_path):
    session.author("DIVISORS(28)")
    await session.simplify("#1")
    session.save(tmp_path / "d.mth")
    assert (tmp_path / "d.mth").read_text() == (
        "DIVISORS(28)\n\n;Simp(#1)\n[1,2,4,7,14,28]\n"
    )
    other = Session()
    other.load(tmp_path / "d.mth")
    assert await simplified(other, "#1") == "[1, 2, 4, 7, 14, 28]"


def test_a_redefinition_is_an_entry_and_is_saved(session, tmp_path):
    session.author("DIVISORS(n):=[n, 1]")
    session.author("DIVISORS(6)")
    session.save(tmp_path / "r.mth")
    assert (tmp_path / "r.mth").read_text() == "DIVISORS(n):=[n,1]\n\nDIVISORS(6)\n"


# -- reading names ------------------------------------------------------------------


async def test_in_character_mode_a_library_name_is_a_whole_run_before_a_parenthesis():
    assert await answers(
        "DIVISORS(28)",
        "divisors(28)",
        "PRIME_POWER?(27)",
        "star(3)",
        "star + 1",
        "kilo",
    ) == [
        "[1, 2, 4, 7, 14, 28]",
        "[1, 2, 4, 7, 14, 28]",
        "true",
        "37",
        "a*r*s*t + 1",
        "i*k*l*o",
    ]


def test_a_library_name_inside_a_longer_run_is_letters(session):
    assert session.author("xstar(3)").text == "x*s*t*a*r*3"


def test_in_word_mode_a_library_name_before_a_parenthesis_is_a_call(session):
    session.author("InputMode := Word")
    assert session.author("STAR (3)").node.kind is Kind.CALL
    assert session.author("star").node.kind is Kind.NAME


def test_in_sensitive_mode_only_the_exact_spelling_is_called(session):
    session.author("CaseMode := Sensitive")
    assert session.author("STAR(3)").node.kind is Kind.CALL
    assert session.author("star(3)").text == "s*t*a*r*3"


async def test_a_user_s_name_of_that_spelling_wins(session):
    session.author("star := 5")
    assert await simplified(session, "star(3)") == "15"
    assert await simplified(session, "STAR(3)") == "37"


def test_the_parameters_of_the_texts_are_no_variables_of_the_session(session):
    assert session.state.resolve("dk") is None
    assert session.author("dk").text == "d*k"
    assert session.state.resolve("default1") is not None


def test_the_parse_state_withdraws_only_library_names():
    state = ParseState()
    state.withdraw(["DIVISORS", "F"])
    assert state.withdrawn == {"DIVISORS"}
    assert state.library_function("DIVISORS") is None
    assert state.library_function("STAR") == "STAR"
    state.declare(SettingDeclaration("CaseMode", str(CaseMode.SENSITIVE)))
    assert state.library_function("star") is None


# -- library variables and the built-ins beside them --------------------------------


async def test_default1_is_the_value_of_an_omitted_argument():
    assert await answers(
        "default1", "PARA_DIF([t, t^2], t)", "> default1 := 2", "PARA_DIF([t, t^2], t)"
    ) == ["1", "2*t", "2"]


async def test_taking_default1_away_takes_the_library_s_value_too(session):
    session.author("default1 :=")
    assert await simplified(session, "default1") == "default1"


async def test_random_sign_and_the_built_ins_that_serve_utility_functions():
    assert await answers(
        "ABS(random_sign)",
        "POLYGAMMA(1, 1)",
        "POLYGAMMA(-1, 2)",
        "SI(inf)",
        "EI(x, 1)",
    ) == ["1", "pi^2/6", "0", "pi/2", "LN(x) + x + euler_gamma"]


# -- the functions, utility file by utility file ------------------------------------

#: Each case is the library function it tests and the lines of one fresh session: a
#: line `> text` is entered, or simplified when it defines nothing (`RANDOM(-10)`
#: seeds the generator), and a pair is an input and what it simplifies to. Sessions
#: approximate to ten digits, as Derive 6's do.
LIBRARY_CASES = [
    # -- CombinatorialFunctions --------------------------------------------------------
    (
        "CATALAN",
        ("CATALAN(10)", "16796"),
        ("VECTOR(CATALAN(n), n, 0, 5)", "[1, 1, 2, 5, 14, 42]"),
    ),
    (
        "BERNOULLI",
        (
            "VECTOR(BERNOULLI(n), n, 0, 8)",
            "[1, -1/2, 1/6, 0, -1/30, 0, 1/42, 0, -1/30]",
        ),
        ("BERNOULLI(12)", "-691/2730"),
        ("BERNOULLI(1, #e^x, -1)", "-1/2"),
    ),
    (
        "BERNOULLI_POLY",
        (
            "BERNOULLI_POLY(10, x)",
            "x^10 - 5*x^9 + 15*x^8/2 - 7*x^6 + 5*x^4 - 3*x^2/2 + 5/66",
        ),
        ("BERNOULLI_POLY(2, x)", "x^2 - x + 1/6"),
        ("BERNOULLI_POLY(3, x)", "x*(x - 1)*(2*x - 1)/2"),
        ("BERNOULLI_POLY(4, 1)", "-1/30"),
    ),
    (
        "EULER",
        ("EULER(10)", "-50521"),
        ("VECTOR(EULER(n), n, 0, 10)", "[1, 0, -1, 0, 5, 0, -61, 0, 1385, 0, -50521]"),
        ("EULER(12)", "2702765"),
    ),
    (
        "EULER_POLY",
        (
            "EULER_POLY(10, x)",
            "(x - 1)*(x^9 - 4*x^8 - 4*x^7 + 26*x^6 + 26*x^5 - 100*x^4 - 100*x^3 + "
            "155*x^2 + 155*x)",
        ),
        ("EULER_POLY(2, x)", "x*(x - 1)"),
        ("EULER_POLY(3, x)", "x^3 - 3*x^2/2 + 1/4"),
        ("EULER_POLY(4, x)", "(x - 1)*(x^3 - x^2 - x)"),
    ),
    (
        "PARTS",
        ("PARTS(4)", "5"),
        ("VECTOR(PARTS(n), n, 0, 10)", "[1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42]"),
        ("PARTS(50)", "204226"),
        ("PARTS(100)", "190569292"),
        ("MOD(PARTS(5*1000 + 4), 5)", "0"),
    ),
    (
        "PARTS_LIST",
        ("PARTS_LIST(4)", "[1, 1, 2, 3, 5]"),
        ("PARTS_LIST(10)", "[1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42]"),
    ),
    (
        "DISTINCT_PARTS",
        ("DISTINCT_PARTS(4)", "2"),
        ("DISTINCT_PARTS(20)", "64"),
        ("VECTOR(DISTINCT_PARTS(n), n, 0, 10)", "[1, 1, 1, 2, 2, 3, 4, 5, 6, 8, 10]"),
    ),
    (
        "BELL",
        (
            "VECTOR(BELL(n), n, 0, 10)",
            "[1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975]",
        ),
        ("BELL(0)", "1"),
    ),
    (
        "RECURRENCE",
        ("RECURRENCE(v SUB 1 + v SUB 2, v, [1, 1], 8)", "[1, 1, 2, 3, 5, 8, 13, 21]"),
        ("RECURRENCE(v SUB 1 + v SUB 2, v, [1, 0], 8)", "[0, 1, 1, 2, 3, 5, 8, 13]"),
        (
            "RECURRENCE(2*x*v SUB 1 - v SUB 2, v, [x, 1], 4)",
            "[1, x, 2*x^2 - 1, 4*x^3 - 3*x]",
        ),
    ),
    (
        "TRIANGULAR",
        ("VECTOR(TRIANGULAR(n), n, 1, 5)", "[1, 3, 6, 10, 15]"),
        ("TRIANGULAR(n)", "n*(n + 1)/2"),
    ),
    (
        "TETRAHEDRAL",
        ("VECTOR(TETRAHEDRAL(n), n, 1, 6)", "[1, 4, 10, 20, 35, 56]"),
        ("TETRAHEDRAL(n)", "n*(n + 1)*(n + 2)/6"),
    ),
    (
        "PENTATOPE",
        ("VECTOR(PENTATOPE(n), n, 1, 6)", "[1, 5, 15, 35, 70, 126]"),
        ("PENTATOPE(n)", "n*(n + 1)*(n + 2)*(n + 3)/24"),
    ),
    (
        "POLYGONAL",
        ("POLYGONAL(10, 3)", "55"),
        ("POLYGONAL(4, 5)", "22"),
        ("VECTOR(POLYGONAL(n, 4), n, 1, 6)", "[1, 4, 9, 16, 25, 36]"),
    ),
    (
        "POLYGONAL_PYRAMID",
        ("POLYGONAL_PYRAMID(n, 3, 2)", "n*(n + 1)/2"),
        ("POLYGONAL_PYRAMID(n, 3, 3)", "n*(n + 1)*(n + 2)/6"),
        ("POLYGONAL_PYRAMID(n, 3, 4)", "n*(n + 1)*(n + 2)*(n + 3)/24"),
        ("VECTOR(POLYGONAL_PYRAMID(n, 4, 3), n, 1, 6)", "[1, 5, 14, 30, 55, 91]"),
        ("VECTOR(POLYGONAL_PYRAMID(n, 5, 2), n, 1, 6)", "[1, 5, 12, 22, 35, 51]"),
    ),
    (
        "HEX",
        ("VECTOR(HEX(n), n, 1, 5)", "[1, 7, 19, 37, 61]"),
        ("HEX(n)", "3*n^2 - 3*n + 1"),
    ),
    (
        "STAR",
        ("VECTOR(STAR(n), n, 1, 6)", "[1, 13, 37, 73, 121, 181]"),
        ("STAR(n)", "6*n^2 - 6*n + 1"),
    ),
    (
        "CENTERED",
        ("VECTOR(CENTERED(n, 5), n, 1, 5)", "[1, 6, 16, 31, 51]"),
        ("CENTERED(n, p)", "n^2*p/2 - n*p/2 + 1"),
    ),
    (
        "OCTAHEDRAL",
        ("VECTOR(OCTAHEDRAL(n), n, 1, 6)", "[1, 6, 19, 44, 85, 146]"),
        ("OCTAHEDRAL(n)", "n*(2*n^2 + 1)/3"),
    ),
    (
        "CENTERED_PYRAMID",
        ("CENTERED_PYRAMID(n, 6, 2)", "3*n^2 - 3*n + 1"),
        ("CENTERED_PYRAMID(n, 4, 3)", "n*(2*n^2 + 1)/3"),
        ("VECTOR(CENTERED_PYRAMID(n, 6, 3), n, 1, 5)", "[1, 8, 27, 64, 125]"),
        ("CENTERED_PYRAMID(3, 6, 2)", "19"),
    ),
    (
        "CENTERED_CUBE",
        ("VECTOR(CENTERED_CUBE(n, 3), n, 1, 5)", "[1, 9, 35, 91, 189]"),
        ("CENTERED_CUBE(2, 2)", "5"),
    ),
    (
        "RHOMBIC_DODECAHEDRAL",
        ("VECTOR(RHOMBIC_DODECAHEDRAL(n), n, 1, 5)", "[1, 15, 65, 175, 369]"),
        ("RHOMBIC_DODECAHEDRAL(n)", "4*n^3 - 6*n^2 + 4*n - 1"),
    ),
    (
        "CENTERED_HEX",
        ("CENTERED_HEX(n, 3)", "4*n^3 - 6*n^2 + 4*n - 1"),
        ("VECTOR(CENTERED_HEX(n, 2), n, 1, 5)", "[1, 7, 19, 37, 61]"),
        ("CENTERED_HEX(2, 4)", "31"),
    ),
    (
        "STIRLING",
        (
            "TABLE([STIRLING(-5, -n), STIRLING(n, 5)], n, 5, 7)",
            "[[5, 1, 1], [6, 15, 15], [7, 175, 140]]",
        ),
        ("STIRLING(0, 0)", "1"),
        ("STIRLING(3, 5)", "0"),
        ("STIRLING(-2, 3)", "0"),
        ("[STIRLING(0, -2), STIRLING(-2, 0), STIRLING(3, 0)]", "[0, 0, 0]"),
    ),
    (
        "STIRLING1",
        (
            "VECTOR(STIRLING1(n, 5), n, 1, 10)",
            "[0, 0, 0, 0, 1, 15, 175, 1960, 22449, 269325]",
        ),
        (
            "VECTOR([n, VECTOR(STIRLING1(n, k), k, 0, n)], n, 0, 4)",
            "[[0, [1]], [1, [0, 1]], [2, [0, 1, 1]], [3, [0, 2, 3, 1]], [4, [0, 6, "
            "11, 6, 1]]]",
        ),
        ("[STIRLING1(10, 7), STIRLING2(-7, -10)]", "[9450, 9450]"),
        ("VECTOR(STIRLING1(6, k_), k_, 0, 6)", "[0, 120, 274, 225, 85, 15, 1]"),
    ),
    (
        "STIRLING_CYCLE",
        ("STIRLING_CYCLE(4, 2)", "11"),
        ("STIRLING_CYCLE(5, 3)", "35"),
    ),
    (
        "STIRLING2",
        (
            "VECTOR(STIRLING2(n, 5), n, 1, 10)",
            "[0, 0, 0, 0, 1, 15, 140, 1050, 6951, 42525]",
        ),
        (
            "VECTOR([n, VECTOR(STIRLING2(n, k), k, 0, n)], n, 0, 4)",
            "[[0, [1]], [1, [0, 1]], [2, [0, 1, 1]], [3, [0, 1, 3, 1]], [4, [0, 1, "
            "7, 6, 1]]]",
        ),
        ("STIRLING2(13, 7)", "5715424"),
        ("APPROX(STIRLING2(30, 10)*10!*10^(2 - 30))", "62.91371892"),
    ),
    (
        "STIRLING_SUBSET",
        ("STIRLING_SUBSET(4, 2)", "7"),
        ("STIRLING_SUBSET(6, 3)", "90"),
    ),
    # -- NumberTheoryFunctions ---------------------------------------------------------
    (
        "EXTENDED_GCD",
        ("EXTENDED_GCD(20580, 85176)", "[84, [149, -36]]"),
        ("EXTENDED_GCD(229, 541)", "[1, [189, -80]]"),
        ("EXTENDED_GCD(-4, 6)", "[2, [1, 1]]"),
        ("EXTENDED_GCD(0, 0)", "[0, [0, 0]]"),
    ),
    (
        "SOLVE_MOD",
        ("SOLVE_MOD(3*x - 6, x, 9)", "[2, 5, 8]"),
        ("SOLVE_MOD(3*x, x, 9)", "[0, 3, 6]"),
        ("SOLVE_MOD(3*x = 30, x, 9)", "[1, 4, 7]"),
        ("SOLVE_MOD(29*x = 1, x, 660)", "[569]"),
        ("SOLVE_MOD(2*x = 1, x, 4)", "[]"),
    ),
    (
        "CRT",
        ("CRT([1, 2, 3], [4, 5, 11])", "157"),
        ("CRT([MOD(83, 5), MOD(83, 7), MOD(83, 16)], [5, 7, 16])", "83"),
        ("CRT([483296, 4], [101^3, 103])", "2543898"),
    ),
    (
        "NTH_PRIME",
        ("NTH_PRIME(9)", "23"),
        ("NTH_PRIME(100)", "541"),
        ("VECTOR(NTH_PRIME(n), n, 1, 10)", "[2, 3, 5, 7, 11, 13, 17, 19, 23, 29]"),
    ),
    (
        "PRIMEPI",
        (
            "[PRIMEPI(10000, 6, 1), PRIMEPI(10000, 6, 5) + 2, PRIMEPI(10000)]",
            "[611, 618, 1229]",
        ),
        ("PRIMEPI(2)", "1"),
        ("PRIMEPI(1)", "0"),
        ("PRIMEPI(21/2)", "4"),
        ("PRIMEPI(10, 3, 1)", "1"),
        ("PRIMEPI(-3)", "0"),
    ),
    (
        "FAREY",
        ("FAREY(6)", "[1/6, 1/5, 1/4, 1/3, 2/5, 1/2, 3/5, 2/3, 3/4, 4/5, 5/6, 1]"),
        (
            "FAREY(7)",
            "[1/7, 1/6, 1/5, 1/4, 2/7, 1/3, 2/5, 3/7, 1/2, 4/7, 3/5, 2/3, 5/7, "
            "3/4, 4/5, 5/6, 6/7, 1]",
        ),
        ("FAREY(1)", "[1]"),
    ),
    (
        "DIVISORS",
        ("DIVISORS(28)", "[1, 2, 4, 7, 14, 28]"),
        (
            "DIVISORS(23456)",
            "[1, 2, 4, 8, 16, 32, 733, 1466, 2932, 5864, 11728, 23456]",
        ),
        (
            "DIVISORS(3450)",
            "[1, 2, 3, 5, 6, 10, 15, 23, 25, 30, 46, 50, 69, 75, 115, 138, 150, "
            "230, 345, 575, 690, 1150, 1725, 3450]",
        ),
        (
            "DIVISORS(360)",
            "[1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 60, "
            "72, 90, 120, 180, 360]",
        ),
        ("DIVISORS(1)", "[1]"),
        ("DIVISORS(x)", "[1, x]"),
    ),
    (
        "DIVISOR_SIGMA",
        ("DIVISOR_SIGMA(1, 360)", "1170"),
        ("DIVISOR_SIGMA(3, 1000)", "1151792460"),
        (
            "VECTOR(DIVISOR_SIGMA(1, n_), n_, 1, 10)",
            "[1, 3, 4, 7, 6, 12, 8, 15, 13, 18]",
        ),
        ("DIVISOR_SIGMA(k, 4)", "2^(2*k) + 2^k + 1"),
    ),
    (
        "DIVISOR_TAU",
        ("DIVISOR_TAU(28)", "6"),
        ("VECTOR(DIVISOR_TAU(n_), n_, 1, 10)", "[1, 2, 2, 3, 2, 4, 2, 4, 3, 4]"),
    ),
    (
        "EULER_PHI",
        ("EULER_PHI(100)", "40"),
        ("EULER_PHI(1)", "1"),
        ("EULER_PHI(x)", "x - 1"),
        ("VECTOR(EULER_PHI(n_), n_, 1, 10)", "[1, 1, 2, 2, 4, 2, 6, 4, 6, 4]"),
        ("INVERSE_MOD(301, EULER_PHI(15251))", "14701"),
    ),
    (
        "PRIME_POWER?",
        (
            "SELECT(PRIME_POWER?(n), n, 1, 30)",
            "[2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29]",
        ),
        (
            "[PRIME_POWER?(9), PRIME_POWER?(10), PRIME_POWER?(8), PRIME_POWER?(1)]",
            "[true, false, true, false]",
        ),
        ("PRIME_POWER?(2^61 - 1)", "true"),
        ("PRIME_POWER?(12)", "false"),
        (
            "[PRIME_POWER?(0), PRIME_POWER?(-8), PRIME_POWER?(1/2), "
            "PRIME_POWER?(SQRT(2))]",
            "[false, false, false, false]",
        ),
    ),
    (
        "PRIMITIVE_ROOT",
        ("PRIMITIVE_ROOT(54)", "5"),
        ("PRIMITIVE_ROOT(101)", "2"),
        ("PRIMITIVE_ROOT(2551)", "6"),
        ("PRIMITIVE_ROOT(3397745833)", "5"),
        (
            "SELECT(PRIMITIVE_ROOT(p, 10) = 10, p, SELECT(PRIME?(q_), q_, 1, 100))",
            "[7, 17, 19, 23, 29, 47, 59, 61, 97]",
        ),
        ("PRIMITIVE_ROOT(5, 10)", "12"),
        ("PRIMITIVE_ROOT(18)", "5"),
        ("PRIMITIVE_ROOT(8)", "?"),
        ("PRIMITIVE_ROOT(15)", "?"),
        ("PRIMITIVE_ROOT(1)", "0"),
        ("PRIMITIVE_ROOT(6)", "5"),
    ),
    (
        "MOEBIUS_MU",
        ("MOEBIUS_MU(100)", "0"),
        ("[MOEBIUS_MU(2431), MOEBIUS_MU(1000), MOEBIUS_MU(1122)]", "[-1, 0, 1]"),
        ("VECTOR(MOEBIUS_MU(n_), n_, 1, 10)", "[1, -1, -1, 0, -1, 1, -1, 0, 0, 1]"),
    ),
    (
        "SQUAREFREE",
        ("SQUAREFREE(837)", "false"),
        (
            "VECTOR(SQUAREFREE(n_), n_, 1, 10)",
            "[true, true, true, false, true, true, true, false, false, true]",
        ),
    ),
    (
        "CYCLOTOMIC",
        (
            "VECTOR([n, CYCLOTOMIC(n, x)], n, 1, 6)",
            "[[1, x - 1], [2, x + 1], [3, x^2 + x + 1], [4, x^2 + 1], [5, x^4 + "
            "x^3 + x^2 + x + 1], [6, x^2 - x + 1]]",
        ),
        ("CYCLOTOMIC(12, x)", "x^4 - x^2 + 1"),
        (
            "CYCLOTOMIC(105, x)",
            "x^48 + x^47 + x^46 - x^43 - x^42 - 2*x^41 - x^40 - x^39 + x^36 + x^35 "
            "+ x^34 + x^33 + x^32 + x^31 - x^28 - x^26 - x^24 - x^22 - x^20 + x^17 "
            "+ x^16 + x^15 + x^14 + x^13 + x^12 - x^9 - x^8 - 2*x^7 - x^6 - x^5 + "
            "x^2 + x + 1",
        ),
        (
            "SELECT(ABS(PRODUCT(SUBST(TERMS(CYCLOTOMIC(n, x)), x, 1))) > 1, n, 200)",
            "[105, 165, 195]",
        ),
    ),
    (
        "GEN_LUCAS",
        ("GEN_LUCAS(10, 1, -1, 0, 1)", "55"),
        ("GEN_LUCAS(5, 3, 2, 0, 1)", "31"),
    ),
    (
        "U_LUCAS",
        ("U_LUCAS(30, 1, -1)", "832040"),
        ("U_LUCAS(10, 3, 2)", "1023"),
    ),
    (
        "V_LUCAS",
        ("V_LUCAS(31, 1, -1)", "3010349"),
        ("V_LUCAS(10, 3, 2)", "1025"),
    ),
    (
        "U_MOD",
        ("U_MOD(30, 1, -1, 31)", "0"),
        ("U_MOD(10, 1, -1, 7)", "6"),
    ),
    (
        "V_MOD",
        ("V_MOD(31, 1, -1, 31)", "1"),
    ),
    (
        "LUCAS",
        ("VECTOR(LUCAS(n), n, 0, 10)", "[2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123]"),
    ),
    (
        "FIBONACCI",
        ("VECTOR(FIBONACCI(n), n, 0, 10)", "[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]"),
        ("FIBONACCI(100)", "354224848179261915075"),
        (
            "REVERSE(VECTOR(FIBONACCI(k), k, 2, 16))",
            "[987, 610, 377, 233, 144, 89, 55, 34, 21, 13, 8, 5, 3, 2, 1]",
        ),
    ),
    (
        "PELL",
        ("VECTOR(PELL(k), k, 1, 10)", "[0, 1, 2, 5, 12, 29, 70, 169, 408, 985]"),
        ("VECTOR(PELL(n), n, 0, 10)", "[1, 0, 1, 2, 5, 12, 29, 70, 169, 408, 985]"),
    ),
    (
        "LUCAS_LEHMER",
        ("LUCAS_LEHMER(31)", "true"),
        ("LUCAS_LEHMER(11)", "false"),
    ),
    (
        "NEXT_MERSENNE_DEGREE",
        ("NEXT_MERSENNE_DEGREE(31)", "61"),
        ("NEXT_MERSENNE_DEGREE(1)", "2"),
    ),
    (
        "MERSENNE_LIST",
        ("MERSENNE_LIST(12)", "[2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127]"),
        ("MERSENNE_LIST(5)", "[2, 3, 5, 7, 13]"),
    ),
    (
        "MERSENNE_DEGREE",
        ("MERSENNE_DEGREE(38)", "6972593"),
        ("MERSENNE_DEGREE(39)", "13466917"),
        ("MERSENNE_DEGREE(-1)", "13466917"),
        ("VARIABLES(MERSENNE_DEGREE(n))", "[n]"),
    ),
    (
        "MERSENNE",
        ("MERSENNE(10)", "618970019642690137449562111"),
        ("VECTOR(MERSENNE(n), n, 1, 5)", "[3, 7, 31, 127, 8191]"),
        ("MERSENNE(8)", "2147483647"),
    ),
    (
        "PERFECT",
        ("PERFECT(2)", "28"),
        ("VECTOR(PERFECT(n), n, 1, 4)", "[6, 28, 496, 8128]"),
        ("PERFECT(5)", "33550336"),
    ),
    (
        "CONTINUED_FRACTION",
        ("APPROX(CONTINUED_FRACTION(#e, 8))", "[2, 1, 2, 1, 1, 4, 1, 1, 6]"),
        ("APPROX(CONTINUED_FRACTION(pi, 7))", "[3, 7, 15, 1, 292, 1, 1, 1]"),
        ("CONTINUED_FRACTION(SQRT(2), 3)", "[1, 2, 2, 2]"),
    ),
    (
        "CONVERGENTS",
        (
            "CONVERGENTS(SQRT(2), 10)",
            "[1, 3/2, 7/5, 17/12, 41/29, 99/70, 239/169, 577/408, 1393/985, "
            "3363/2378, 8119/5741]",
        ),
        ("CONVERGENTS(7/3, 5)", "[2, 7/3]"),
        (
            "CONVERGENTS(0.9333333333, 5)",
            "[0, 1, 13/14, 14/15, 1866666661/1999999994, 3733333336/4000000003]",
        ),
        ("CONVERGENTS(3, 4)", "[3]"),
        ("CONVERGENTS(x, 0)", "[FLOOR(x)]"),
        ("CONVERGENTS(pi, 1/2)", "?"),
    ),
    (
        "CONVERGENT",
        ("CONVERGENT(SQRT(2), 10)", "8119/5741"),
        ("CONVERGENT(pi, 3)", "355/113"),
        ("CONVERGENT(7/3, 0)", "2"),
        ("CONVERGENT(7/3, 4)", "7/3"),
        ("CONVERGENT(x, 0)", "FLOOR(x)"),
        ("[CONVERGENT(pi, -1), CONVERGENT(pi, k)]", "[?, ?]"),
    ),
    (
        "JACOBI",
        ("JACOBI(2, 23)", "1"),
        ("JACOBI(10^60 + 1, 10^68 + 99)", "1"),
        ("JACOBI(5, 16937389168607)", "-1"),
        ("JACOBI(6, 9)", "0"),
        ("JACOBI(3, 8)", "?"),
    ),
    (
        "SQUARE_ROOT",
        ("SQUARE_ROOT(2, 23)", "18"),
        ("SQUARE_ROOT(3, 23)", "16"),
        ("SQUARE_ROOT(5, 23)", "?"),
        ("SQUARE_ROOT(4, 13)", "11"),
        ("SQUARE_ROOT(3, 13)", "9"),
        ("SQUARE_ROOT(2, 17)", "6"),
        ("SQUARE_ROOT(2, 41)", "17"),
        ("SQUARE_ROOT(2, 97)", "83"),
        ("SQUARE_ROOT(0, 7)", "0"),
        ("SQUARE_ROOT(3, 2)", "1"),
        ("SQUARE_ROOT(2, 15)", "1"),
    ),
    # -- EquationSolving ---------------------------------------------------------------
    (
        "NEWTON",
        (
            "APPROX(NEWTON(x^2 - 3, x, 2))",
            "[2, 1.75, 1.732142857, 1.732050810, 1.732050807, 1.732050807]",
        ),
        ("APPROX(NEWTON(x^2 - 3, x, 2, 2))", "[2, 1.75, 1.732142857]"),
        ("NEWTON(x^2 - 2, x, 1, 3)", "[1, 3/2, 17/12, 577/408]"),
    ),
    (
        "NEWTONS",
        (
            "APPROX(NEWTONS([EXP(x*y) - x - 2*y, ATAN(x*y) - 2*x - y], [x, y], "
            "[-1, -1]))",
            "[[-1, -1], [0.4928806511, -0.9645356430], [-0.04339581048, "
            "0.7570251808], [-0.4463079803, 0.5634802058], [-0.4249948246, "
            "0.5997033149], [-0.4246672327, 0.5998876906], [-0.4246672072, "
            "0.5998877071], [-0.4246672072, 0.5998877071]]",
        ),
        (
            "APPROX(NEWTONS([3*x^2 + z - 10, x - 4*y^2 + 2*z + 15, x^2 + y^2 - 8], "
            "[x, y, z], [1, -2, -1]))",
            "[[1, -2, -1], [1.666666666, -2.416666666, 3], [1.509803921, "
            "-2.396974306, 3.235294117], [1.500038147, -2.397911967, 3.249942778], "
            "[1.500000000, -2.397915761, 3.249999999], [1.5, -2.397915761, 3.25], "
            "[1.5, -2.397915761, 3.25]]",
        ),
        (
            "NEWTONS([x^2 - 2, y - 1], [x, y], [1, 0], 2)",
            "[[1, 0], [3/2, 1], [17/12, 1]]",
        ),
    ),
    (
        "FIXED_POINT",
        (
            "APPROX(FIXED_POINT(2*COS(x), x, 1.5, 10))",
            "[1.5, 0.1414744033, 1.980018354, -0.7957914253, 1.399439154, "
            "0.3410396031, 1.884814923, -0.6177664155, 1.630348433, -0.1190338253, "
            "1.985847670]",
        ),
        (
            "APPROX(FIXED_POINT((x + 2*COS(x))/2, x, 1.5, 10))",
            "[1.5, 0.8207372016, 1.092050620, 1.006691570, 1.038005274, "
            "1.026942146, 1.030909002, 1.029493846, 1.029999627, 1.029818978, "
            "1.029883515]",
        ),
        "> g := 8 - (x + 1)^(1/3)",
        (
            "APPROX(FIXED_POINT(g, x, 6))",
            "[6, 6.087068817, 6.079170206, 6.079884066, 6.079819527, 6.079825362, "
            "6.079824834, 6.079824882, 6.079824878, 6.079824878, 6.079824878, "
            "6.079824878]",
        ),
        ("FIXED_POINT(x/2 + 1, x, 0, 3)", "[0, 1, 3/2, 7/4]"),
        (
            "APPROX(FIXED_POINT(COS(x), x, 1, 3))",
            "[1, 0.5403023058, 0.8575532158, 0.6542897904]",
        ),
    ),
    (
        "TAYLOR_SOLVE",
        (
            "TAYLOR_SOLVE(SIN(y) + y + x*EXP(x), x, y, 0, 0, 3)",
            "-25*x^3/96 - x^2/2 - x/2",
        ),
        ("TAYLOR_SOLVE(x^2 + y^2 - 1, x, y, 0, 1, 4)", "-x^4/8 - x^2/2 + 1"),
    ),
    (
        "TAYLOR_INVERSE",
        ("TAYLOR_INVERSE(x*EXP(x), x, y, 0, 3)", "3*y^3/2 - y^2 + y"),
        ("TAYLOR_INVERSE(SIN(x), x, y, 0, 5)", "3*y^5/40 + y^3/6 + y"),
        (
            "TAYLOR_INVERSE(x^2, x, y, 0, 3)",
            "\"Taylor expansion of inverse does not exist\"",
        ),
    ),
    # -- LinearAlgebra -----------------------------------------------------------------
    (
        "SCALE_ELEMENT",
        "> A := [1, 2, 3; 4, 5, 6; 7, 8, 9]",
        ("SCALE_ELEMENT(A, 1, 4)", "[[4, 8, 12], [4, 5, 6], [7, 8, 9]]"),
        (
            "SCALE_ELEMENT([1, 0, 5, 2, -3, 0; 0, 1, -3, -1, 2, 0; 0, 0, -5, -3, "
            "4, 1], 3, -1/5)",
            "[[1, 0, 5, 2, -3, 0], [0, 1, -3, -1, 2, 0], [0, 0, 1, 3/5, -4/5, -1/5]]",
        ),
    ),
    (
        "SWAP_ELEMENTS",
        "> A := [1, 2, 3; 4, 5, 6; 7, 8, 9]",
        ("SWAP_ELEMENTS(A, 1, 3)", "[[7, 8, 9], [4, 5, 6], [1, 2, 3]]"),
        ("SWAP_ELEMENTS([a, b, c], 1, 2)", "[b, [[1, 2, 3], [4, 5, 6], [7, 8, 9]], c]"),
    ),
    (
        "SUBTRACT_ELEMENTS",
        "> A := [1, 2, 3; 4, 5, 6; 7, 8, 9]",
        ("SUBTRACT_ELEMENTS(A, 2, 1, 4)", "[[1, 2, 3], [0, -3, -6], [7, 8, 9]]"),
        ("SUBTRACT_ELEMENTS(A, 2, 1)", "[[1, 2, 3], [3, 3, 3], [7, 8, 9]]"),
        (
            "SUBTRACT_ELEMENTS([2, 6, -3, -6; 0, -9, 9, 18; 0, -15, 15, k + 12], "
            "3, 2, 15/9)",
            "[[2, 6, -3, -6], [0, -9, 9, 18], [0, 0, 0, k - 18]]",
        ),
    ),
    (
        "FORCE0",
        "> A := [1, 2, 3; 4, 5, 6; 7, 8, 9]",
        ("FORCE0(A, 2, 1, 1)", "[[1, 2, 3], [0, -3, -6], [7, 8, 9]]"),
        ("FORCE0(A, 3, 2, 2)", "[[1, 2, 3], [4, 5, 6], [3/5, 0, -3/5]]"),
    ),
    (
        "PIVOT",
        "> A := [1, 2, 3; 4, 5, 6; 7, 8, 9]",
        ("PIVOT(A, 1, 1)", "[[1, 2, 3], [0, -3, -6], [0, -6, -12]]"),
        ("PIVOT(PIVOT(A, 1, 1), 2, 2)", "[[1, 2, 3], [0, -3, -6], [0, 0, 0]]"),
    ),
    (
        "MINOR",
        "> A := [1, 2, 3; 4, 5, 6; 7, 8, 9]",
        ("MINOR(A, 2, 2)", "[[1, 3], [7, 9]]"),
        ("MINOR(A, 1, 3)", "[[4, 5], [7, 8]]"),
    ),
    (
        "COFACTOR",
        "> A := [1, 2, 3; 4, 5, 6; 7, 8, 9]",
        ("COFACTOR(A, 1, 2)", "6"),
        ("COFACTOR(A, 2, 2)", "-12"),
    ),
    (
        "ADJOINT",
        ("ADJOINT([2, x; 1, 3])", "[[3, -x], [-1, 2]]"),
        ("ADJOINT([1, 2; 3, 4])*[1, 2; 3, 4]", "[[-2, 0], [0, -2]]"),
    ),
    (
        "NULL_SPACE",
        ("NULL_SPACE([1, 2, 3; 1, 2, 3; 1, 2, 3])", "[[2, 3], [-1, 0], [0, -1]]"),
        (
            "[1, 2, 3; 1, 2, 3; 1, 2, 3]*NULL_SPACE([1, 2, 3; 1, 2, 3; 1, 2, 3])",
            "[[0, 0], [0, 0], [0, 0]]",
        ),
        ("NULL_SPACE([1, 0; 0, 1])", "[]"),
        ("NULL_SPACE([[1, 2]])", "[[2], [-1]]"),
        ("NULL_SPACE([0, 0; 0, 0])", "[[-1, 0], [0, -1]]"),
        ("NULL_SPACE([1, 2, 0, 3; 0, 0, 1, 4])", "[[2, 3], [-1, 0], [0, 4], [0, -1]]"),
        ("NULL_SPACE([1, 0; 0, x - 2])", "[]"),
    ),
    (
        "EXACT_EIGENVECTOR",
        ("EXACT_EIGENVECTOR([1, 2; 1, 1], SQRT(2) + 1)", "[[-SQRT(2)], [-1]]"),
        ("EXACT_EIGENVECTOR([1, 2; 1, 1], 1 - SQRT(2))", "[[SQRT(2)], [-1]]"),
        ("EXACT_EIGENVECTOR([2, 0; 0, 2], 2)", "[[-1, 0], [0, -1]]"),
        ("EXACT_EIGENVECTOR([1, 2; 1, 1], 1)", "[]"),
        ("EXACT_EIGENVECTOR([1, 2; 1, 1], 2.414213562)", "[]"),
        (
            "EXACT_EIGENVECTOR([-2, 1, 0, 0; 2, -9/5, 0, 0; 0, 1/5, 0, 0; 0, 3/5, "
            "0, 0], 0)",
            "[[0, 0], [0, 0], [-1, 0], [0, -1]]",
        ),
    ),
    (
        "APPROX_EIGENVECTOR",
        "> Motive := [1, 1, 7, 1/5, 1/8, 3; 1, 1, 9, 2, 1/8, 1/4; 1/7, 1/9, 1, 2, "
        "1/5, 4; 5, 1/2, 1/2, 1, 1/3, 2; 8, 8, 5, 3, 1, 6; 1/3, 4, 1/4, 1/2, 1/6, 1]",
        "> NotationDigits := 6",
        (
            "APPROX(APPROX_EIGENVECTOR(Motive, 9.52509))",
            "[-0.261811, -0.293876, -0.175191, -0.257476, -0.844939, -0.184882]",
        ),
        "> A := [-2, 1, 0, 0; 2, -1.8, 0, 0; 0, 0.2, 0, 0; 0, 0.6, 0, 0]",
        "> PrecisionDigits := 6",
        (
            "APPROX(APPROX_EIGENVECTOR(A, -0.482255))",
            "[0.370995, 0.563076, -0.233518, -0.700554]",
        ),
        ("APPROX(APPROX_EIGENVECTOR([2, 0; 0, 3], 2.1))", "[0.998460, 0.0554700]"),
        ("APPROX(APPROX_EIGENVECTOR([2, 1; 1, 2], 2.99))", "[-0.696367, -0.717685]"),
    ),
    # -- VectorMatrixFunctions ---------------------------------------------------------
    (
        "OUTER",
        ("OUTER([a, b], [2, 3, 4])", "[[2*a, 3*a, 4*a], [2*b, 3*b, 4*b]]"),
        ("OUTER([1, 2], [1, 2])", "[[1, 2], [2, 4]]"),
    ),
    (
        "KRONECKER",
        "> matel(m, k, n, c1, c2) := IF(k = 0, KRONECKER(m, n), c1*SQRT(m + "
        "1)*matel(m + 1, k - 1, n, c1, c2) + c2*SQRT(m)*matel(m - 1, k - 1, n, c1, "
        "c2))",
        ("matel(m, 1, m)", "0"),
        ("matel(m, 1, m + 1)", "c1*SQRT(m + 1)"),
        ("matel(m, 2, m)", "c1*c2*(2*m + 1)"),
        ("matel(m, 2, m + 2)", "c1^2*SQRT(m + 1)*SQRT(m + 2)"),
        (
            "matel(m, 1, n)",
            "c1*SQRT(m + 1)*IF(n = m + 1, 1, 0) + c2*SQRT(m)*IF(n = m - 1, 1, 0)",
        ),
        ("[KRONECKER(2, 2), KRONECKER(1, 2)]", "[1, 0]"),
        (
            "VECTOR(VECTOR(KRONECKER(i, j), j, 3), i, 3)",
            "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]",
        ),
    ),
    (
        "APPEND_COLUMNS",
        (
            "APPEND_COLUMNS([a, b; c, d], [e; f], [g; h])",
            "[[a, b, e, g], [c, d, f, h]]",
        ),
        (
            "APPEND_COLUMNS([[a, b; c, d], [e; f], [g; h]])",
            "[[a, b, e, g], [c, d, f, h]]",
        ),
        (
            "APPEND_COLUMNS([[a], [b]], [[c, d], [e, f]], [[g], [h]])",
            "[[a, c, d, g], [b, e, f, h]]",
        ),
        ("APPEND_COLUMNS([a, b; c, d], [e, f])", "[[a, b, e], [c, d, f]]"),
        ("APPEND_COLUMNS()", "[]"),
        (
            "APPEND_COLUMNS([1; 1; 1], [0; 1; 2], [2; 1; 0])",
            "[[1, 0, 2], [1, 1, 1], [1, 2, 0]]",
        ),
        ("DET(APPEND_COLUMNS([1; 1; 1], [0; 1; 2], [2; 1; 0]))", "0"),
        ("APPEND_COLUMNS([u1, u2, u3], [c1], [c2])", "[u, 2*u, 3*u, c, 2*c]"),
        ("APPEND_COLUMNS([1, 2; 3, 4])", "[[1, 2], [3, 4]]"),
    ),
    (
        "PARTITION",
        ("PARTITION([1, 2, 3, 4, 5], 3, 2)", "[[1, 2, 3], [3, 4, 5]]"),
        ("PARTITION([1, 2, 3, 4, 5, 6], 2)", "[[1, 2], [3, 4], [5, 6]]"),
        ("PARTITION([1, 2, 3, 4, 5], 2)", "[[1, 2], [3, 4]]"),
    ),
    (
        "REMOVE_ELEMENTS",
        ("REMOVE_ELEMENTS([a, b, b, c, d, d, d, d, e, e], [b, d, e])", "[a, c]"),
        ("REMOVE_ELEMENTS([a, b, b, c, d, d, d, d, e, e], d)", "[a, b, b, c, e, e]"),
        ("REMOVE_ELEMENTS([1, 2, 3], [])", "[1, 2, 3]"),
        ("REMOVE_ELEMENTS([], a)", "[]"),
        ("REMOVE_ELEMENTS([x, 2*x, x + x], 2*x)", "[x]"),
    ),
    (
        "MATPROD",
        ("MATPROD([1, 2; 3, 4], [5, 6; 7, 8], 2, 1)", "43"),
        ("MATPROD([a, b; c, d], [e, f; g, h], 1, 2)", "a*f + b*h"),
    ),
    (
        "JACOBIAN",
        ("JACOBIAN([(w^2 - v^2)/2, w*v], [w, v])", "[[w, -v], [v, w]]"),
        (
            "JACOBIAN([3*x^2 + z, x - 4*y^2 + 2*z, x^2 + y^2], [x, y, z])",
            "[[6*x, 0, 1], [1, -8*y, 2], [2*x, 2*y, 0]]",
        ),
    ),
    (
        "COVARIANT_METRIC_TENSOR",
        ("COVARIANT_METRIC_TENSOR([w, -v; v, w])", "[[v^2 + w^2, 0], [0, v^2 + w^2]]"),
        ("COVARIANT_METRIC_TENSOR([1, 2; 0, 1])", "[[1, 2], [2, 5]]"),
    ),
    (
        "GEOMETRY_MATRIX",
        (
            "GEOMETRY_MATRIX([u, v], [v^2 + w^2, 0; 0, v^2 + w^2])",
            "[[u, v], [SQRT(v^2 + w^2), SQRT(v^2 + w^2)]]",
        ),
        ("GEOMETRY_MATRIX([r, θ], [1, 0; 0, r^2])", "[[r, theta], [1, ABS(r)]]"),
    ),
    (
        "RECTANGULAR_TO_POLAR",
        ("RECTANGULAR_TO_POLAR(3, 4)", "[5, ATAN(4/3)]"),
        ("APPROX(RECTANGULAR_TO_POLAR(3, 4))", "[5, 0.9272952180]"),
        ("RECTANGULAR_TO_POLAR(-1, 0)", "[1, pi]"),
        ("RECTANGULAR_TO_POLAR(0, -2)", "[2, -pi/2]"),
    ),
    (
        "POLAR_TO_RECTANGULAR",
        ("POLAR_TO_RECTANGULAR(2, pi/3)", "[1, SQRT(3)]"),
        ("POLAR_TO_RECTANGULAR(r, 0)", "[r, 0]"),
    ),
    (
        "POLAR_SUM",
        ("POLAR_SUM([1, pi/2], [2, pi/3])", "[SQRT(2*SQRT(3) + 5), ATAN(SQRT(3) + 1)]"),
        ("APPROX(POLAR_SUM([1, pi/2], [2, pi/3]))", "[2.909312911, 1.219916915]"),
        ("POLAR_SUM([1, 0], [1, pi/2])", "[SQRT(2), pi/4]"),
    ),
    # -- RationalApproximation ---------------------------------------------------------
    (
        "PADE",
        (
            "PADE(#e^x, x, 0, 3, 3)",
            "(x^3 + 12*x^2 + 60*x + 120)/(-x^3 + 12*x^2 - 60*x + 120)",
        ),
        ("PADE(EXP(x), x, 0, 3, 2)", "(x^3 + 9*x^2 + 36*x + 60)/(3*(x^2 - 8*x + 20))"),
        ("PADE(1/(1 - x), x, 0, 0, 1)", "1/(1 - x)"),
    ),
    (
        "CHEBYSHEV_T_SERIES",
        (
            "CHEBYSHEV_T_SERIES(EXP(x), x, -1/2, 1/2, 4, 5)",
            "x^4/24 + 65*x^3/384 + x^2/2 + 6143*x/6144 + 1",
        ),
        (
            "CHEBYSHEV_T_SERIES(EXP(x), x, -1/2, 1/2, 5, 5)",
            "x^5/120 + x^4/24 + x^3/6 + x^2/2 + x + 1",
        ),
        ("CHEBYSHEV_T_SERIES(x^2, x, -1, 1, 0, 2)", "1/2"),
        ("CHEBYSHEV_T_SERIES(x^2, x, 0, 2, 0, 2)", "3/2"),
        ("CHEBYSHEV_T_SERIES(x^3, x, -1, 1, 1, 3)", "3*x/4"),
        ("CHEBYSHEV_T_SERIES(EXP(x), x, -1, 1, 3, 2)", "\"k must be <= n\""),
    ),
    # -- MiscellaneousFunctions --------------------------------------------------------
    (
        "SQUARE_WAVE",
        ("SQUARE_WAVE(1/2)", "1"),
        ("SQUARE_WAVE(3/2)", "-1"),
        ("SQUARE_WAVE(-1/2)", "-1"),
        ("SQUARE_WAVE(x)", "(-1)^FLOOR(x)"),
    ),
    (
        "RATIO_TEST",
        ("RATIO_TEST(1/n!, n)", "0"),
        ("RATIO_TEST(2^n/n, n)", "2"),
    ),
    (
        "LIM2",
        ("LIM2((x^2 - y^2)/(x^2 + y^2), x, y, 0, 0)", "(1 - @1^2)/(@1^2 + 1)"),
        ("LIM2((SIN(x^2) - SIN(y^2))/(x - y), x, y, 0, 0)", "0"),
    ),
    (
        "LEFT_RIEMANN",
        ("LEFT_RIEMANN(x, x, 0, 1, 4)", "3/8"),
        ("LEFT_RIEMANN(x^2, x, 0, 1, 2)", "1/8"),
    ),
    (
        "INT_PARTS",
        ("INT_PARTS(x, COS(x), x)", "COS(x) + x*SIN(x)"),
        ("INT_PARTS(x, #e^x, x)", "x*#e^x - #e^x"),
        ("INT_PARTS(LN(x), 1, x)", "x*LN(x) - x"),
        ("INT_PARTS(1, #e^x, x)", "#e^x"),
    ),
    (
        "DEF_INT_PARTS",
        ("DEF_INT_PARTS(x, COS(x), x, 0, pi)", "-2"),
        ("DEF_INT_PARTS(x, #e^x, x, 0, 1)", "1"),
    ),
    (
        "INT_SUBST",
        ("INT_SUBST(t*SIN(t^2), t, t^2)", "-COS(t^2)/2"),
        ("INT_SUBST(2*x*#e^(x^2), x, x^2)", "#e^(x^2)"),
    ),
    (
        "DEF_INT_SUBST",
        ("DEF_INT_SUBST(t*SIN(t^2), t, t^2, 0, SQRT(pi))", "1"),
        ("DEF_INT_SUBST(2*x*#e^(x^2), x, x^2, 0, 1)", "#e - 1"),
    ),
    (
        "INVERSE",
        ("INVERSE(SIN(x/b), x)", "b*ASIN(x)"),
        ("INVERSE(x^3 + 1, x)", "(x - 1)^(1/3)"),
    ),
    (
        "PROVE_SUM",
        ("PROVE_SUM(k, k, 1, n, n*(n + 1)/2)", "[0, 0]"),
        ("PROVE_SUM(k^3, k, 1, n, n^2*(n + 1)^2/4)", "[0, 0]"),
        ("PROVE_SUM(k, k, 1, n, n^2)", "[0, -n]"),
    ),
    (
        "POLY_COEFF",
        ("POLY_COEFF((3*x^3 + 2*x - 1)*(3*x - 3*x^2 + 2*x^4 - x^5/2), x, 5)", "-9/2"),
        ("POLY_COEFF(PRODUCT(1 + x^i, i, 1, 20), x, 20)", "64"),
        (
            "TABLE(POLY_COEFF((x + x^2 + x^3 + x^4 + x^5 + x^6)^4, x, k), k, 4, 8)`",
            "[[4, 5, 6, 7, 8], [1, 4, 10, 20, 35]]",
        ),
    ),
    (
        "POLY_DEGREE",
        ("POLY_DEGREE((3*x^3 + 2*x - 1)*(3*x - 3*x^2 + 2*x^4 - x^5/2), x)", "8"),
        ("POLY_DEGREE(PRODUCT(1 + x^i, i, 1, 20), x)", "210"),
        ("POLY_DEGREE(0, x)", "-1"),
        ("POLY_DEGREE(5, x)", "0"),
    ),
    (
        "RANDOM_POLY",
        ("POLY_COEFF(RANDOM_POLY(x, 3), x, 3)", "1"),
        ("POLY_DEGREE(RANDOM_POLY(u, 2, 20), u)", "2"),
        ("POLY_COEFF(RANDOM_POLY(x, 4, 9), x, 4)", "1"),
    ),
    (
        "RANDOM_VECTOR",
        "> RANDOM(-10)",
        ("RANDOM_VECTOR(4, 6)", "[-1, -4, 3, 1]"),
        ("DIM(RANDOM_VECTOR(5, 10))", "5"),
    ),
    (
        "RANDOM_MATRIX",
        (
            "[DIM(RANDOM_MATRIX(3, 4, 10)), DIM(FIRST(RANDOM_MATRIX(3, 4, 10)))]",
            "[3, 4]",
        ),
        "> RANDOM(-10)",
        ("RANDOM_MATRIX(2, 2, 6)", "[[-1, -4], [3, 1]]"),
    ),
    (
        "RANDOM_NORMAL",
        ("RANDOM_NORMAL(0, 5)", "5"),
        ("NUMBER?(APPROX(RANDOM_NORMAL(1, 0)))", "true"),
    ),
    (
        "GOODNESS_OF_FIT",
        ("GOODNESS_OF_FIT(x, x, [0, 0; 1, 1; 2, 3])", "SQRT(3)/3"),
        ("GOODNESS_OF_FIT(2*x, x, [0, 0; 1, 2])", "0"),
    ),
    (
        "LINEAR_CORRELATION_COEFFICIENT",
        ("LINEAR_CORRELATION_COEFFICIENT([1, 2; 2, 4; 3, 7])", "5*SQRT(57)/38"),
        ("LINEAR_CORRELATION_COEFFICIENT([1, 3; 2, 1; 3, -1])", "-1"),
    ),
    (
        "POLY_INTERPOLATE",
        ("POLY_INTERPOLATE([[0, 0], [1, 1], [2, 2], [3, 0]], x)", "3*x^2/2 - x^3/2"),
        ("POLY_INTERPOLATE([[0, 1], [1, 2], [2, 5]], x)", "x^2 + 1"),
    ),
    (
        "POLY_INTERPOLATE_EXPRESSION",
        ("POLY_INTERPOLATE_EXPRESSION(x^3, x, [0, 1, 2])", "3*x^2 - 2*x"),
        ("POLY_INTERPOLATE_EXPRESSION(SIN(x), x, [0, pi/2, pi])", "-4*x*(x - pi)/pi^2"),
    ),
    # -- DifferentiationApplications ---------------------------------------------------
    (
        "CURVATURE",
        ("CURVATURE(x^2, x)", "2/(4*x^2 + 1)^(3/2)"),
        ("CURVATURE(x^3, x)", "6*x/(9*x^4 + 1)^(3/2)"),
    ),
    (
        "CENTER_OF_CURVATURE",
        ("CENTER_OF_CURVATURE(x^2, x)", "[-4*x^3, 3*x^2 + 1/2]"),
        ("CENTER_OF_CURVATURE(SQRT(1 - x^2), x)", "[0, 0]"),
    ),
    (
        "TANGENT",
        ("TANGENT(x^2, x, 1)", "2*x - 1"),
        ("TANGENT(SQRT(x^2 + 1), x, inf)", "x"),
        ("TANGENT(SIN(x), x, 0)", "x"),
    ),
    (
        "PERPENDICULAR",
        ("PERPENDICULAR(x^2, x, 1)", "3/2 - x/2"),
        ("PERPENDICULAR(#e^x, x, 0)", "1 - x"),
    ),
    (
        "OSCULATING_CIRCLE",
        (
            "SUBST(OSCULATING_CIRCLE(x^2, x, θ), x, 0)",
            "[COS(theta)/2, SIN(theta)/2 + 1/2]",
        ),
        ("SUBST(OSCULATING_CIRCLE(x^2/2, x, θ), x, 0)", "[COS(theta), SIN(theta) + 1]"),
    ),
    (
        "PARA_DIF",
        ("PARA_DIF([COS(t), SIN(t)], t)", "-COT(t)"),
        ("PARA_DIF([t, t^3], t, 2)", "6*t"),
    ),
    (
        "PARA_CURVATURE",
        ("PARA_CURVATURE([t, t^2], t)", "2/(4*t^2 + 1)^(3/2)"),
        ("SUBST(PARA_CURVATURE([2*COS(t), 2*SIN(t)], t), t, pi/2)", "-1/2"),
    ),
    (
        "PARA_CENTER_OF_CURVATURE",
        ("PARA_CENTER_OF_CURVATURE([t, t^2], t)", "[-4*t^3, 3*t^2 + 1/2]"),
        ("PARA_CENTER_OF_CURVATURE([COS(t), SIN(t)], t)", "[0, 0]"),
    ),
    (
        "PARA_TANGENT",
        ("PARA_TANGENT([t, t^2], t, 1, x)", "2*x - 1"),
        (
            "y = PARA_TANGENT([a*COS(t)^(2/n), b*SIN(t)^(2/n)], t, θ, x)",
            "y = b*(SIN(theta)^(2/n - 2)*COS(theta)^2 - x*SIN(theta)^(2/n - "
            "2)*COS(theta)^(2 - 2/n)/a + SIN(theta)^(2/n))",
        ),
    ),
    (
        "PARA_PERPENDICULAR",
        ("PARA_PERPENDICULAR([t, t^2], t, 1, x)", "3/2 - x/2"),
        ("PARA_PERPENDICULAR([t, t^3], t, 1, x)", "4/3 - x/3"),
    ),
    (
        "PARA_OSCULATING_CIRCLE",
        ("PARA_OSCULATING_CIRCLE([t, t^2], t, 0, φ)", "[COS(phi)/2, SIN(phi)/2 + 1/2]"),
        (
            "PARA_OSCULATING_CIRCLE([2*COS(t), 2*SIN(t)], t, pi/2, φ)",
            "[-2*COS(phi), -2*SIN(phi)]",
        ),
    ),
    (
        "POLAR_DIF",
        ("POLAR_DIF(1, θ)", "-COT(theta)"),
        (
            "POLAR_DIF(θ, θ)",
            "(theta*COS(theta) + SIN(theta))/(COS(theta) - theta*SIN(theta))",
        ),
    ),
    (
        "POLAR_CURVATURE",
        ("SUBST(POLAR_CURVATURE(2, θ), θ, pi/2)", "-1/2"),
        ("SUBST(POLAR_CURVATURE(1, θ), θ, pi/2)", "-1"),
    ),
    (
        "POLAR_CENTER_OF_CURVATURE",
        ("POLAR_CENTER_OF_CURVATURE(2, θ)", "[0, 0]"),
        ("POLAR_CENTER_OF_CURVATURE(1, θ)", "[0, 0]"),
    ),
    (
        "POLAR_TANGENT",
        ("POLAR_TANGENT(1, θ, pi/4, x)", "SQRT(2) - x"),
        ("POLAR_TANGENT(2, θ, pi/2, x)", "2"),
    ),
    (
        "POLAR_PERPENDICULAR",
        ("POLAR_PERPENDICULAR(1, θ, pi/4, x)", "x"),
        ("POLAR_PERPENDICULAR(2, θ, pi/4, x)", "x"),
    ),
    (
        "POLAR_OSCULATING_CIRCLE",
        ("POLAR_OSCULATING_CIRCLE(2, θ, pi/2, φ)", "[-2*COS(phi), -2*SIN(phi)]"),
        ("POLAR_OSCULATING_CIRCLE(1, θ, pi/2, φ)", "[-COS(phi), -SIN(phi)]"),
    ),
    (
        "IMP_DIF",
        ("2*x*y + (1 + x^2)*IMP_DIF(x^2*y + y - 1, x, y)", "0"),
        (
            "IMP_DIF(x + SQRT(x*y) - y - 4)",
            "-y*(SQRT(x*y) + 2*x)/(x*(SQRT(x*y) - 2*y))",
        ),
        (
            "IMP_DIF(y^3*SIN(x*y) - x^(4.5))",
            "(9*x^(7/2) - 2*y^4*COS(x*y))/(2*y^2*(x*y*COS(x*y) + 3*SIN(x*y)))",
        ),
        ("IMP_DIF(x^2 + y^2 - 1, x, y, 2)", "-x^2/y^3 - 1/y"),
        "> f4(x, y) := x^2/2 + y^2/2 - x - 2*y",
        (
            "SOLUTIONS(x + 4*y - 5 AND IMP_DIF(f4(x, y), x, y, 1) = IMP_DIF(x + "
            "4*y - 5, x, y, 1), [x, y])",
            "[[13/17, 18/17]]",
        ),
    ),
    (
        "IMP_CURVATURE",
        ("SUBST(IMP_CURVATURE(x^2 + y^2 - 1, x, y), [x, y], [0, 1])", "-1"),
        ("SUBST(IMP_CURVATURE(y - x^2, x, y), [x, y], [0, 0])", "2"),
    ),
    (
        "IMP_CENTER_OF_CURVATURE",
        ("IMP_CENTER_OF_CURVATURE(x^2 + y^2 - 4, x, y)", "[0, 0]"),
        ("IMP_CENTER_OF_CURVATURE(y - x^2, x, y)", "[-4*x^3, 2*x^2 + y + 1/2]"),
    ),
    (
        "IMP_TANGENT",
        ("IMP_TANGENT(x^2 + y^2 - 2, x, y, 1, 1)", "2 - x"),
        ("IMP_TANGENT(y - x^2, x, y, 1, 1)", "2*x - 1"),
    ),
    (
        "IMP_PERPENDICULAR",
        ("IMP_PERPENDICULAR(x^2 + y^2 - 2, x, y, 1, 1)", "x"),
        ("IMP_PERPENDICULAR(y - x^2, x, y, 1, 1)", "3/2 - x/2"),
    ),
    (
        "IMP_OSCULATING_CIRCLE",
        (
            "IMP_OSCULATING_CIRCLE(x^2 + y^2 - 4, x, y, 0, 2, θ)",
            "[-2*COS(theta), -2*SIN(theta)]",
        ),
        (
            "IMP_OSCULATING_CIRCLE(y - x^2, x, y, 0, 0, θ)",
            "[COS(theta)/2, SIN(theta)/2 + 1/2]",
        ),
        (
            "IMP_OSCULATING_CIRCLE(y - x^2, x, y, 1, 1, θ)",
            "[5*SQRT(5)*COS(theta)/2 - 4, 5*SQRT(5)*SIN(theta)/2 + 7/2]",
        ),
    ),
    (
        "TANGENT_PLANE",
        (
            "TANGENT_PLANE(27*x^2 + 18*x*y*z + 4*x*z^3 - 4*y^3 - y^2*z^2, [x, y, "
            "z], [5/27, 1, 1])",
            "32*x - 32*y/3 + 32*z/9 + 32/27",
        ),
        (
            "TANGENT_PLANE(x^2 + y^2 + z^2 - 3, [x, y, z], [1, 1, 1])",
            "2*x + 2*y + 2*z - 6",
        ),
    ),
    (
        "NORMAL_LINE",
        (
            "NORMAL_LINE(x^2 + y^2 + z^2 - 3, [x, y, z], [1, 1, 1], t)",
            "[2*t + 1, 2*t + 1, 2*t + 1]",
        ),
        (
            "NORMAL_LINE(z - x^2 - y^2, [x, y, z], [1, 1, 2], t)",
            "[1 - 2*t, 1 - 2*t, t + 2]",
        ),
    ),
    # -- IntegrationApplications -------------------------------------------------------
    (
        "LAPLACE",
        "> s :epsilon Real (0, inf)",
        ("LAPLACE(t^2, t, s)", "2/s^3"),
        ("LAPLACE(2 + t/2, t, s)", "(4*s + 1)/(2*s^2)"),
        ("LAPLACE((2 + t/2)*SIN(2*t), t, s)", "2*(2*s^2 + s + 8)/(s^2 + 4)^2"),
    ),
    (
        "LAPLACE",
        "> s :epsilon Real (5, inf)",
        ("LAPLACE(t^2*EXP(5*t), t, s)", "-2/(5 - s)^3"),
    ),
    (
        "LAPLACE",
        # The transform is defined where its integral converges, here for s > 0, and
        # is its value there: the integral alone is a conditional for an undeclared s.
        ("LAPLACE(SIN(t), t, s)", "1/(s^2 + 1)"),
        ("LAPLACE(t^2, t, s)", "2/s^3"),
        ("INT(#e^(-s*t)*t^2, t, 0, inf)", "IF(s > 0, 2/s^3, inf)"),
        ("LAPLACE(SIN(t), t, 2)", "1/5"),
    ),
    (
        "FOURIER",
        ("FOURIER(t, t, -pi, pi, 2)", "2*SIN(t) - SIN(2*t)"),
    ),
    (
        "ARC_LENGTH",
        ("ARC_LENGTH(x, x, 0, 1)", "SQRT(2)"),
        ("ARC_LENGTH(x^(3/2), x, 0, 4)", "80*SQRT(10)/27 - 8/27"),
        ("APPROX(ARC_LENGTH(-x^3/6 + x, x, 0, 2.5))", "3.327921319"),
    ),
    (
        "POLAR_ARC_LENGTH",
        ("POLAR_ARC_LENGTH(1, θ, 0, 2*pi)", "2*pi"),
        ("POLAR_ARC_LENGTH(2*COS(θ), θ, 0, pi)", "2*pi"),
    ),
    (
        "PARA_ARC_LENGTH",
        "> bez3(a_, b_, c_, d_, t) := (1 - t)^3*a_ + 3*(1 - t)^2*t*b_ + 3*(1 - "
        "t)*t^2*c_ + t^3*d_",
        "> cv(t) := bez3([-10, 9], [15, -20], [7, 20], [-5, -6], t)",
        ("APPROX(PARA_ARC_LENGTH(cv(t), t, 0, 1))", "39.81314121"),
        ("PARA_ARC_LENGTH([COS(t), SIN(t)], t, 0, pi)", "pi"),
        ("PARA_ARC_LENGTH([t, 0], t, 0, 2, t)", "2"),
        ("PARA_ARC_LENGTH([COS(t), SIN(t), t], t, 0, 2*pi)", "2*SQRT(2)*pi"),
    ),
    (
        "AREA",
        ("AREA(x, 0, 1, y, 0, x)", "1/2"),
        ("AREA(x, 0, 1, y, 0, x, x*y)", "1/8"),
    ),
    (
        "AREA_CENTROID",
        ("AREA_CENTROID(x, 0, 1, y, 0, x)", "[2/3, 1/3]"),
        ("AREA_CENTROID(x, 0, 2, y, 0, 1)", "[1, 1/2]"),
    ),
    (
        "AREA_INERTIA",
        ("AREA_INERTIA(x, 0, 1, y, 0, 1)", "[[1/3, -1/4], [-1/4, 1/3]]"),
        ("AREA_INERTIA(x, 0, 2, y, 0, 1)", "[[1/3, -1/2], [-1/2, 4/3]]"),
    ),
    (
        "POLAR_AREA",
        ("POLAR_AREA(r, 0, 1, θ, 0, 2*pi)", "pi"),
        ("POLAR_AREA(r, 0, 2*COS(θ), θ, -pi/2, pi/2)", "pi"),
    ),
    (
        "SURFACE_AREA",
        ("SURFACE_AREA(0, x, 0, 1, y, 0, 1)", "1"),
        ("SURFACE_AREA(x + y, x, 0, 1, y, 0, 1)", "SQRT(3)"),
    ),
    (
        "VOLUME",
        ("VOLUME(x, 0, 1, y, 0, 1, z, 0, 1)", "1"),
        ("VOLUME(x, 0, 1, y, 0, x, z, 0, y)", "1/6"),
    ),
    (
        "VOLUME_CENTROID",
        ("VOLUME_CENTROID(x, 0, 1, y, 0, 1, z, 0, 1)", "[1/2, 1/2, 1/2]"),
        ("VOLUME_CENTROID(x, 0, 1, y, 0, 1, z, 0, 2)", "[1/2, 1/2, 1]"),
    ),
    (
        "VOLUME_INERTIA",
        (
            "VOLUME_INERTIA(x, 0, 1, y, 0, 1, z, 0, 1)",
            "[[2/3, -1/4, -1/4], [-1/4, 2/3, -1/4], [-1/4, -1/4, 2/3]]",
        ),
        (
            "VOLUME_INERTIA(x, 0, 2, y, 0, 1, z, 0, 1)",
            "[[2/3, -1/2, -1/2], [-1/2, 5/3, -1/4], [-1/2, -1/4, 5/3]]",
        ),
    ),
    (
        "SPHERICAL_VOLUME",
        ("SPHERICAL_VOLUME(r, 0, 1, θ, 0, pi, φ, 0, 2*pi)", "4*pi/3"),
        ("SPHERICAL_VOLUME(r, 0, 2, θ, 0, pi/2, φ, 0, 2*pi)", "16*pi/3"),
    ),
    (
        "CYLINDRICAL_VOLUME",
        ("CYLINDRICAL_VOLUME(r, 0, 1, θ, 0, 2*pi, z, 0, 2)", "2*pi"),
        ("CYLINDRICAL_VOLUME(r, 0, 2, θ, 0, pi, z, 0, 1)", "2*pi"),
    ),
    (
        "VOLUME_OF_REVOLUTION",
        ("VOLUME_OF_REVOLUTION(x, x, 0, 1)", "pi/3"),
        ("VOLUME_OF_REVOLUTION(SQRT(x), x, 0, 1)", "pi/2"),
    ),
    (
        "AREA_OF_REVOLUTION",
        ("AREA_OF_REVOLUTION(1, x, 0, 1)", "2*pi"),
        ("AREA_OF_REVOLUTION(x, x, 0, 1)", "SQRT(2)*pi"),
    ),
    (
        "VOLUMEY_OF_REVOLUTION",
        ("VOLUMEY_OF_REVOLUTION(x^2, x, 0, 1)", "pi/2"),
        ("VOLUMEY_OF_REVOLUTION(1, x, 0, 1)", "pi"),
    ),
    (
        "AREAY_OF_REVOLUTION",
        ("AREAY_OF_REVOLUTION(x, x, 0, 1)", "SQRT(2)*pi"),
        ("AREAY_OF_REVOLUTION(1, x, 0, 1)", "pi"),
    ),
    # -- NumericalApproximation --------------------------------------------------------
    (
        "DIF_NUMERIC",
        "> F5(x) := ABS(SIGN(x))",
        ("DIF_NUMERIC(F5(x), x, 0, h)", "0"),
        ("DIF_NUMERIC(x^3, x, 1, 1/10, 1)", "1201/400"),
        ("DIF_NUMERIC(x^3, x, 1, 1/10)", "1201/400"),
        ("DIF_NUMERIC(x^3, x, 1, 1/10, 2)", "6"),
    ),
    (
        "SMOOTH_VECTOR",
        ("SMOOTH_VECTOR([1, 2, 4, 8])", "[1, 7/3, 14/3, 8]"),
        ("SMOOTH_VECTOR([0, 3, 0, 3, 0])", "[0, 1, 2, 1, 0]"),
    ),
    (
        "SMOOTH_COLUMN",
        ("SMOOTH_COLUMN([1, 2; 3, 4; 5, 9], 2)", "[[1, 2], [3, 5], [5, 9]]"),
        ("SMOOTH_COLUMN([1, 2; 3, 4; 5, 9], 1)", "[[1, 2], [3, 4], [5, 9]]"),
    ),
    (
        "DIF_DATA",
        "> InputMode := Word",
        "> CaseMode := Sensitive",
        "> Precision := Approximate",
        "> DPA1(pKm1, pKm2) := 1/(10^(pKm1 - pH) + 1 + 10^(pH - pKm2))",
        "> DPA2(pKm1, pKm2) := 1/(10^(pKm1 + pKm2 - 2*pH) + 10^(pKm2 - pH) + 1)",
        "> VOL_BUR_DP(Vf, CA0, CB, pKw, pKm1, pKm2, γH) := "
        "VECTOR([Vf*(CA0*γH*(DPA1(pKm1, pKm2) + 2*DPA2(pKm1, pKm2)) + 10^(pH - "
        "pKw) - 10^(-pH))/(CB*γH + 10^(-pH) - 10^(pH - pKw)), pH], pH, "
        "FLOOR(-LOG(2*CA0, 10)), CEILING(14 + LOG(CA0, 10)), 0.1)",
        "> m := VOL_BUR_DP(20, 0.002, 0.02, 14, 2.74, 5.36, 0.772)",
        (
            "DIF_DATA(m) SUB [1, 2, 3]",
            "[[-7.674608566, 0.08601676820], [-6.547594726, 0.09144334169], "
            "[-5.483814654, 0.09788232195]]",
        ),
        ("DIF_DATA([[0, 0], [1, 1], [2, 4]])", "[[0, 0], [1, 2], [2, 4]]"),
        ("DIF_DATA([[0, 1], [1, 3], [3, 7]])", "[[0, 2], [1, 2], [3, 2]]"),
    ),
    (
        "DIF2_DATA",
        ("DIF2_DATA([[0, 0], [1, 1], [2, 4]])", "[[0, 2], [1, 2], [2, 2]]"),
        ("DIF2_DATA([[0, 1], [1, 3], [3, 7]])", "[[0, 0], [1, 0], [3, 0]]"),
    ),
    (
        "INT_DATA",
        ("INT_DATA([[0, 0], [1, 1], [2, 4]])", "[[0, 0], [1, 1/2], [2, 3]]"),
        ("INT_DATA([[0, 2], [3, 2]])", "[[0, 0], [3, 6]]"),
    ),
    # -- FirstOrderODEs ----------------------------------------------------------------
    (
        "DSOLVE1",
        ("DSOLVE1(2*x*y, 1 + x^2, x, y, 0, 1)", "x^2*y + y - 1 = 0"),
        ("SOLVE(DSOLVE1(-x*y^2, 1, x, y, 0, 1), y)", "y = -2/(x^2 - 2)"),
        "> Branch := Real",
    ),
    (
        "DSOLVE1_GEN",
        ("DSOLVE1_GEN(-1/x*v - x*SIN(x), 1, x, v)", "COS(x) + v/x = c"),
        ("DSOLVE1_GEN(v/(v^2 + 1), -1, v, y)", "LN(v^2 + 1)/2 - y = c"),
        ("DSOLVE1_GEN(y/v, -1, v, y)", "LN(v) - LN(y) = c"),
        (
            "DSOLVE1_GEN(c*x - x*COS(x), -1, x, y, k)",
            "-COS(x) - x*SIN(x) + c*x^2/2 - y = k",
        ),
    ),
    (
        "SEPARABLE",
        (
            "SEPARABLE(-1/x, (1 + y^2)/y, x, y, 3, 2)",
            "LN(y^2 + 1)/2 - LN(5)/2 = LN(3) - LN(x)",
        ),
        (
            "SEPARABLE(p*p*t*α*#e^(-p*t), 1, t, c, 0, 0)",
            "c = alpha*(1 - (p*t + 1)*#e^(-p*t))",
        ),
    ),
    (
        "SEPARABLE_GEN",
        ("SEPARABLE_GEN(1/SIN(x)^2, COS(y)^2)", "TAN(y) = c - COT(x)"),
        ("SEPARABLE_GEN(x, y, x, y)", "LN(y) = x^2/2 + c"),
    ),
    (
        "LINEAR1",
        ("LINEAR1(-4/x, -(2*x^2 + 4)/x, x, y, 1, -3)", "y = -5*x^4 + x^2 + 1"),
        (
            "LINEAR1(TAN(x), SIN(x)/COS(x)^2, x, y, pi/6, 1)",
            "y = SQRT(3)*((12 - 4*SQRT(3))*COS(x)^2 + 3*SQRT(3))/(18*COS(x))",
        ),
    ),
    (
        "LINEAR1_GEN",
        (
            "LINEAR1_GEN(x/(x^2 - 1), 1/(1 - x^2))",
            "y = (c - LN(SQRT(x^2 - 1) + x))/SQRT(x^2 - 1)",
        ),
        ("LINEAR1_GEN(1, 1)", "y = c*#e^(-x) + 1"),
    ),
    (
        "HOMOGENEOUS",
        (
            "HOMOGENEOUS((y + x)/(y - x))",
            "LN(-(x0^2 + 2*x0*y0 - y0^2)/x0^2)/2 - LN(y*(y - 2*x)/x^2 - 1)/2 = "
            "LN(x) - LN(x0)",
        ),
        (
            "HOMOGENEOUS((y + x)/(y - x), x, y, 1, 2)",
            "-LN(y*(2*x - y)/x^2 + 1)/2 = LN(x)",
        ),
        (
            "HOMOGENEOUS((y*#e^(y/x) - x)/(x*#e^(y/x)), x, y, 1/2, -3/2)",
            "#e^(-3) - #e^(y/x) = LN(2*x)",
        ),
        (
            "HOMOGENEOUS((y^2 - x^2)/(2*x*y), x, y, 2, -1)",
            "LN(5/4) - LN(y^2/x^2 + 1) = LN(x/2)",
        ),
        (
            "HOMOGENEOUS(2*x*y/(y^2 - x^2), x, y, 2, -1)",
            "LN(11/8)/3 - LN(y*(y^2 - 3*x^2)/x^3)/3 = LN(x/2)",
        ),
        (
            "HOMOGENEOUS((3*x^2*y - y^3)/(x^3 - 3*x*y^2), x, y, -3, -2)",
            "LN(y/x)/2 - LN(y^2/x^2 + 1) + LN(13*SQRT(6)/18) = LN(-x/3)",
        ),
        ("HOMOGENEOUS(1 + y/x, x, y, 1, 0)", "y/x = LN(x)"),
        ("HOMOGENEOUS(x*y, x, y, 1, 1)", "\"inapplicable\""),
    ),
    (
        "HOMOGENEOUS_GEN",
        ("HOMOGENEOUS_GEN((y*#e^(y/x) - x)/(x*#e^(y/x)))", "-#e^(y/x) = LN(x) + c"),
        ("HOMOGENEOUS_GEN((y*#e^(y/x) - 1)/(x*#e^(y/x)))", "\"inapplicable\""),
    ),
    (
        "EXACT",
        (
            "EXACT(2*x^2 - y^2 + y, -(2*x*y - x + 4*y), x, y, 1, -3/2)",
            "2*x^3/3 + x*(y - y^2) - 2*y^2 + 91/12 = 0",
        ),
        ("EXACT(y, 2*x, x, y, 1, 1)", "\"inapplicable\""),
    ),
    (
        "EXACT_GEN",
        (
            "EXACT_GEN(2*x^2 - y^2 + y, -(2*x*y - x + 4*y))",
            "2*x^3/3 + x*(y - y^2) - 2*y^2 = c",
        ),
        ("EXACT_GEN(2*x + #e^x*LN(y), #e^x/y)", "#e^x*LN(y) + x^2 = c"),
        ("EXACT_GEN(y, 2*x)", "\"inapplicable\""),
    ),
    (
        "INTEGRATING_FACTOR",
        (
            "INTEGRATING_FACTOR(2*x*y, -2*x^2 - y, x, y, -2, 3)",
            "x^2/y^2 + 1/y - 7/9 = 0",
        ),
        (
            "INTEGRATING_FACTOR(x^5 + a*x^4 + x^2*y + y^3, -x^3 - x*y^2)",
            "(3*x0^3*x^5 + 6*a*x0^3*x^4 + x^3*(-6*a*x0^4 - 3*x0^5 + 6*x0^2*y0 + "
            "2*y0^3) - 6*x0^3*x^2*y - 2*x0^3*y^3)/(6*x0^3*x^3) = 0",
        ),
    ),
    (
        "INTEGRATING_FACTOR_GEN",
        ("INTEGRATING_FACTOR_GEN(2*x^3*y^2 - y, 2*x^2*y^3 - x)", "\"inapplicable\""),
        (
            "INTEGRATING_FACTOR_GEN(1/(x^2*y^2)*(2*x^3*y^2 - y), "
            "1/(x^2*y^2)*(2*x^2*y^3 - x))",
            "x^2 + 1/(x*y) + y^2 = c",
        ),
    ),
    (
        "MONOMIAL_TEST",
        ("MONOMIAL_TEST(2*x^3*y^2 - y, 2*x^2*y^3 - x)", "1/(x^2*y^2)"),
        ("MONOMIAL_TEST(y, 2*x)", "1/SQRT(x)"),
    ),
    (
        "BERNOULLI_ODE",
        ("BERNOULLI_ODE(1, #e^x, -1, x, y, 0, 2)", "y^2 = 2*#e^x/3 + 10*#e^(-2*x)/3"),
        (
            "BERNOULLI_ODE(1, #e^x, -1)",
            "y^2 = y0^2*#e^(2*x0 - 2*x) + 2*#e^x/3 - 2*#e^(3*x0 - 2*x)/3",
        ),
    ),
    (
        "BERNOULLI_ODE_GEN",
        ("BERNOULLI_ODE_GEN(1, #e^x, -1)", "y^2 = c*#e^(-2*x) + 2*#e^x/3"),
    ),
    (
        "GEN_HOM",
        (
            "GEN_HOM(y^2/x^3, x, y, 1, 1)",
            "LN(2 - y/x^2)/2 - LN(y/x^2)/2 = LN(x)",
        ),
        ("GEN_HOM(x + y, x, y, 1, 1)", "\"inapplicable\""),
    ),
    (
        "GEN_HOM_GEN",
        ("GEN_HOM_GEN(y^2/x^3, x, y)", "LN(y/x^2 - 2)/2 - LN(y/x^2)/2 = LN(x) + c"),
        ("GEN_HOM_GEN(x + y, x, y)", "\"inapplicable\""),
    ),
    (
        "FUN_LIN_CCF",
        ("FUN_LIN_CCF((x + y)^2, 1, 1, 0, x, y, 0, 0)", "ATAN(x + y) = x"),
        ("FUN_LIN_CCF(x + y, 1, 1, 0, x, y, 0, 0)", "LN(x + y + 1) = x"),
    ),
    (
        "FUN_LIN_CCF_GEN",
        (
            "FUN_LIN_CCF_GEN((-x + 2*y - 5)/(2*x - 4*y + 4), 2, -4, 4)",
            "-3*LN(2*x - 4*y + 7)/4 + x/2 - y + 1 = x + c",
        ),
        ("FUN_LIN_CCF_GEN((x + y)^2, 1, 1, 0)", "ATAN(x + y) = x + c"),
    ),
    (
        "LIN_FRAC",
        (
            "LIN_FRAC((-3*x + 2*y - 10)/(2*x - 3*y - 5), -3, 2, -10, 2, -3, -5, x, "
            "y, 0, 3)",
            "-LN(-(x - y + 1)/(x + 8))/6 - 5*LN((x + y + 15)/(x + 8))/6 "
            "+ LN(3*3^(2/3)/4) = LN(x/8 + 1)",
        ),
    ),
    (
        "LIN_FRAC_GEN",
        (
            "LIN_FRAC_GEN((-3*x + 2*y - 10)/(2*x - 3*y - 5), -3, 2, -10, 2, -3, -5)",
            "-LN(-(x - y + 1)/(x + 8))/6 - 5*LN((x + y + 15)/(x + 8))/6 = LN(x + 8) "
            "+ c",
        ),
    ),
    (
        "ALMOST_LIN",
        ("ALMOST_LIN(#e^y, #e^y, 1, x, x, y, 0, 0)", "#e^y = 2*#e^(-x) + x - 1"),
        ("ALMOST_LIN(1, y^2, 1, x, x, y, 0, 1)", "\"inapplicable\""),
    ),
    (
        "ALMOST_LIN_GEN",
        ("ALMOST_LIN_GEN(#e^y, #e^y, 1, x)", "#e^y = c*#e^(-x) + x - 1"),
        ("ALMOST_LIN_GEN(1, y^2, 1, x)", "\"inapplicable\""),
    ),
    (
        "CLAIRAUT",
        (
            "CLAIRAUT((x*v - y)^2, v^2 + 1, x, y, v)",
            "[c^2*x^2 - 2*c*x*y + y^2 - c^2 - 1 = 0, 2*v*x^2 - 2*x*y - 2*v = 0]",
        ),
        ("CLAIRAUT(x*v - y, v^2 - v)", "[c*x - y - c^2 + c = 0, x - 2*v + 1 = 0]"),
    ),
    # -- SecondOrderODEs ---------------------------------------------------------------
    (
        "DSOLVE2",
        ("DSOLVE2(0, 1, 0, x, c1, c2)", "c*(COS(x) + 2*SIN(x))"),
        ("DSOLVE2(0, -9, 0)", "c1*#e^(3*x) + c2*#e^(-3*x)"),
        ("DSOLVE2(-1/x, 0, x*SIN(x))", "\"inapplicable\""),
        ("DSOLVE2(-3/x, 8/x^2, (x^2 + 2*x)/x^2)", "\"inapplicable\""),
    ),
    (
        "DSOLVE2_IV",
        ("DSOLVE2_IV(-2/x, 2/x^2, 0, x, 1, 3, 1)", "5*x - 2*x^2"),
        ("DSOLVE2_IV(2/x, -2/x^2, 0, x, 2, 10, 15)", "\"inapplicable\""),
        (
            "DSOLVE2_IV(-4, 8, 0, x, pi/2, 1, -1)",
            "(3*SIN(2*x)/2 - COS(2*x))*#e^(2*x - pi)",
        ),
        (
            "DSOLVE2_IV(0.25, 6.7, 0, x, 0, 1, 0)",
            "(COS(SQRT(10695)*x/40) + "
            "SQRT(10695)*SIN(SQRT(10695)*x/40)/2139)*#e^(-x/8)",
        ),
    ),
    (
        "DSOLVE2_BV",
        ("DSOLVE2_BV(0, 1, 0, x, 0, 0, pi/2, 1)", "SIN(x)"),
        ("DSOLVE2_BV(0, 0, 2, x, 0, 0, 1, 0)", "x^2 - x"),
    ),
    (
        "AUTONOMOUS_CONSERVATIVE",
        (
            "AUTONOMOUS_CONSERVATIVE(1/(4*SQRT(y)))",
            "x = 8*v0^2*IF(v0 = 0, 1, SIGN(v0))*ABS(v0)/3 - 4*SQRT(y0)*IF(v0 = 0, "
            "1, SIGN(v0))*ABS(v0) + 4*SQRT(y)*SQRT(SQRT(y) + v0^2 - "
            "SQRT(y0))*IF(v0 = 0, 1, SIGN(v0))/3 + (8*SQRT(y0) - "
            "8*v0^2)*SQRT(SQRT(y) + v0^2 - SQRT(y0))*IF(v0 = 0, 1, SIGN(v0))/3 + x0",
        ),
        ("AUTONOMOUS_CONSERVATIVE(0, x, y, 0, 0, 1)", "x = y"),
        (
            "AUTONOMOUS_CONSERVATIVE(y, x, y, 0, 1, 1)",
            "x = IF(y < 1, 2*LN(ABS(y)/4 - ABS(ABS(y)/2 + y/2 - 1)/2 + y/4 + 1/2) "
            "- LN(y), LN(y))",
        ),
    ),
    (
        "LIOUVILLE",
        ("LIOUVILLE(0, 0, x, y, c1, c2)", "2*c*x + y - c = 0"),
        ("LIOUVILLE(0, 1, x, y, c1, c2)", "#e^y + 2*c*x - c = 0"),
    ),
    (
        "AUTONOMOUS",
        ("AUTONOMOUS(v^2 + 1)", "(v^2 + 1)/v"),
        ("AUTONOMOUS(v^2/y)", "v/y"),
    ),
    (
        "EXACT2",
        ("EXACT2(1, 0, x, y, v, c)", "v = c"),
        ("EXACT2(x, v, x, y, v, c)", "v*x = c"),
    ),
    # -- ODEApproximation --------------------------------------------------------------
    (
        "TAYLOR_ODE1",
        ("TAYLOR_ODE1(EXP(x*y), x, y, 0, 1, 4)", "5*x^4/12 + x^3/2 + x^2/2 + x + 1"),
        ("TAYLOR_ODE1(y, x, y, 0, 1, 3)", "x^3/6 + x^2/2 + x + 1"),
    ),
    (
        "PICARD",
        ("PICARD(-y + 2*x + 1, 1, x, y, 0, 1)", "x^2 + 1"),
        (
            "EXPAND(ITERATES(PICARD(-y + 2*x + 1, v, x, y, 0, 1), v, 1, 4))",
            "[1, x^2 + 1, -x^3/3 + x^2 + 1, x^4/12 - x^3/3 + x^2 + 1, -x^5/60 + "
            "x^4/12 - x^3/3 + x^2 + 1]",
        ),
        (
            "EXPAND(PICARD(y^2 + SQRT(x), x + 1, x, y, 0, 1))",
            "x^3/3 + x^2 + 2*x^(3/2)/3 + x + 1",
        ),
    ),
    (
        "TAYLOR_ODES",
        "> InputMode := Word",
        (
            "TAYLOR_ODES([y2, y1], x, [y1, y2], 0, [1, 0], 3)",
            "[x^2/2 + 1, x^3/6 + x]",
        ),
        (
            "TAYLOR_ODES([y1, 2*y2], x, [y1, y2], 0, [1, 1], 2)",
            "[x^2/2 + x + 1, 2*x^2 + 2*x + 1]",
        ),
    ),
    (
        "TAYLOR_ODE2",
        (
            "TAYLOR_ODE2(2*x*v + x^2*y + 3*x, x, y, v, 0, 0, 1, 5)",
            "3*x^5/10 + 5*x^3/6 + x",
        ),
        (
            "TAYLOR_ODE2(x + y^2 + v, x, y, v, 0, 1, -1, 5)",
            "-x^5/120 + x^4/24 - x^3/6 - x + 1",
        ),
        (
            "TAYLOR_ODE2((v + 4*x*t)/(2*x - 9), x, t, v, 0, 8, -2, 5)",
            "41*x^5/131220 + 73*x^4/2916 - 143*x^3/243 + x^2/9 - 2*x + 8",
        ),
    ),
    (
        "EULER_ODE",
        (
            "APPROX(EULER_ODE(26/(3 + (x + y)^2), x, y, 1, -2, 1/4, 4))",
            "[[1, -2], [1.25, -0.375], [1.5, 1.351141078], [1.75, 1.935200456], "
            "[2, 2.327222449]]",
        ),
        (
            "APPROX(EULER_ODE(x*y^2, x, y, 0, 1, 0.05, 6))",
            "[[0, 1], [0.05, 1], [0.1, 1.0025], [0.15, 1.007525031], [0.2, "
            "1.015138331], [0.25, 1.025443389], [0.3, 1.038587566]]",
        ),
        ("EULER_ODE(y, x, y, 0, 1, 1/2, 2)", "[[0, 1], [1/2, 3/2], [1, 9/4]]"),
    ),
    (
        "RK",
        "> g := 9.81",
        (
            "APPROX(RK([-g, v], [t, v, x], [0, 100, 0], 1, 3))",
            "[[0, 100, 0], [1, 90.19, 95.095], [2, 80.38, 180.38], [3, 70.57, "
            "255.855]]",
        ),
        (
            "APPROX(RK([x - 0.1*x*y, -0.5*y + 0.02*x*y], [t, x, y], [0, 25, 3], "
            "0.1, 100) SUB 101)",
            "[10, 24.37369317, 3.000687933]",
        ),
        ("RK([y], [x, y], [0, 1], 1, 1)", "[[0, 1], [1, 65/24]]"),
    ),
    (
        "DIRECTION_FIELD",
        ("DIM(DIRECTION_FIELD(x*y, x, 0, 1, 2, y, 0, 1, 3))", "4"),
        (
            "DIRECTION_FIELD(0, x, 0, 1, 1, y, 0, 1, 1) SUB 1 SUB 1",
            "[[-1/4, 0], [1/4, 0]]",
        ),
    ),
    (
        "EXTRACT_2_COLUMNS",
        ("EXTRACT_2_COLUMNS([1, 2, 3; 4, 5, 6], 1, 3)", "[[1, 3], [4, 6]]"),
        ("EXTRACT_2_COLUMNS([1, 2, 3; 4, 5, 6], 3, 2)", "[[3, 2], [6, 5]]"),
    ),
    # -- RecurrenceEquations -----------------------------------------------------------
    (
        "LIN1_DIFFERENCE",
        ("LIN1_DIFFERENCE(m, (m + 1)!, m, 1, 2)", "(m^2/2 + m/2 + 1)*(m - 1)!"),
        ("LIN1_DIFFERENCE(2, 0, x, 0, 1)", "2^x"),
        ("LIN1_DIFFERENCE(1, x, x, 0, 0)", "x*(x - 1)/2"),
    ),
    (
        "RECURRENCE1",
        ("RECURRENCE1(2*y, x, y, 0, 1, 3)", "[[0, 1], [1, 2], [2, 4], [3, 8]]"),
        ("RECURRENCE1(y + x, x, y, 0, 0, 3)", "[[0, 0], [1, 0], [2, 1], [3, 3]]"),
    ),
    (
        "GEOMETRIC1",
        ("GEOMETRIC1(2, 2, 0, x, 1, 1)", "x"),
        ("GEOMETRIC1(3, 1, 1, x, 1, 0)", "LN(x)/LN(3)"),
    ),
    (
        "CLAIRAUT_DIF",
        ("CLAIRAUT_DIF((y - x*d)^2, d^2)", "c^2*x^2 - 2*c*x*y + y^2 - c^2 = 0"),
        ("CLAIRAUT_DIF(y - x*d, d^2)", "-c*x + y - c^2 = 0"),
    ),
    (
        "LIN2_CCF",
        (
            "LIN2_CCF(-8, 16, 4*(n + 1)^2, n)",
            "c1*2^(2*n) + c2*n*2^(2*n) + 4*n^2/9 + 40*n/27 + 44/27",
        ),
        ("LIN2_CCF(-6, 9, 3^(n + 2), n)", "(c1 + c2*n + n^2/2 - n/2)*3^n"),
        ("LIN2_CCF(6, 9, 10, x, c1, c2)", "2*c*x*(-3)^x + c*(-3)^x + 5/8"),
        ("LIN2_CCF(-3, 2, 0, x, c1, c2)", "c*2^x + 2*c"),
        ("LIN2_CCF(-2, 1, 0, x, c1, c2)", "2*c*x + c"),
    ),
    (
        "LIN2_CCF_BV",
        ("LIN2_CCF_BV(-3, 2, 0, x, 0, 0, 2, 3)", "2^x - 1"),
        ("LIN2_CCF_BV(-2, 1, 0, x, 0, 1, 2, 5)", "2*x + 1"),
    ),
    # -- GraphicsFunctions -------------------------------------------------------------
    (
        "AreaUnderCurve",
        (
            "AreaUnderCurve(x^2, x, 0, 1, y)",
            "[x^2, x >= 0 AND x <= 1 AND y < x^2 AND y > 0]",
        ),
        (
            "AreaUnderCurve(x^2, x, 0, 1)",
            "[x^2, x >= 0 AND x <= 1 AND y < x^2 AND y > 0]",
        ),
    ),
    (
        "AreaOverCurve",
        (
            "AreaOverCurve(x^2 - 1, x, -1, 1, y)",
            "[x^2 - 1, x >= -1 AND x <= 1 AND y > x^2 - 1 AND y < 0]",
        ),
        (
            "AreaOverCurve(-x, x, 0, 1, y)",
            "[-x, x >= 0 AND x <= 1 AND x > -y AND y < 0]",
        ),
    ),
    (
        "AreaBetweenCurves",
        (
            "AreaBetweenCurves(x, x^2, x, 0, 1, y)",
            "[x, x^2, x >= 0 AND x <= 1 AND (y - x)*(x^2 - y) > 0]",
        ),
        (
            "AreaBetweenCurves(0, 1, x, 0, 2, y)",
            "[0, 1, x >= 0 AND x <= 2 AND y*(1 - y) > 0]",
        ),
    ),
    (
        "PlotInt",
        (
            "PlotInt(x, x, 0, 1, y)",
            "[x, x >= 0 AND x <= 1 AND x > y AND y > 0, x >= 0 AND x <= 1 AND x < "
            "y AND y < 0]",
        ),
        (
            "PlotInt(x, x, 1, 0, y)",
            "[x, x >= 0 AND x <= 1 AND x < y AND y < 0, x >= 0 AND x <= 1 AND x > "
            "y AND y > 0]",
        ),
    ),
    (
        "POLAR",
        ("POLAR(2, pi/2)", "[0, 2]"),
        ("POLAR(r, 0)", "[r, 0]"),
    ),
    (
        "SPHERE",
        ("SPHERE(1, 0, pi/2)", "[1, 0, 0]"),
        ("SPHERE(2, θ, 0)", "[0, 0, 2]"),
    ),
    (
        "CYLINDER",
        ("CYLINDER(2, pi/2, 3)", "[0, 2, 3]"),
        ("CYLINDER(r, 0, 5)", "[r, 0, 5]"),
    ),
    (
        "CONE",
        ("CONE(pi/4, 0, 2)", "[SQRT(2), 0, 2]"),
        ("CONE(pi/2, pi, 1)", "[-1, 0, 1]"),
    ),
    (
        "TORUS",
        ("TORUS(2, 1, 0, 0)", "[2, 0, 1]"),
        ("TORUS(2, 1, pi/2, pi/2)", "[0, 1, 0]"),
    ),
    (
        "ISOMETRIC",
        ("ISOMETRIC([1, 2, 3])", "[1, 3/2]"),
        "> k1 := [SQRT(1 - 4*v^2)/2, v*COS(0), 0.5 + v*SIN(0)]",
        "> k2 := [-SQRT(1 - 4*v^2)/2, v*COS(0), 0.5 + v*SIN(0)]",
        (
            "[ISOMETRIC(k1), ISOMETRIC(k2)]",
            "[[v - SQRT(1 - 4*v^2)/2, -SQRT(1 - 4*v^2)/4 - v/2 + 1/2], [SQRT(1 - "
            "4*v^2)/2 + v, SQRT(1 - 4*v^2)/4 - v/2 + 1/2]]",
        ),
    ),
    (
        "ROTATE_X",
        ("ROTATE_X(pi/2)", "[[1, 0, 0], [0, 0, -1], [0, 1, 0]]"),
        (
            "ROTATE_X(30*deg)*ROTATE_Y(45*deg)*ROTATE_Z(60*deg)",
            "[[SQRT(2)/4, -SQRT(6)/4, SQRT(2)/2], [SQRT(2)/8 + 3/4, SQRT(3)/4 - "
            "SQRT(6)/8, -SQRT(2)/4], [SQRT(3)/4 - SQRT(6)/8, 3*SQRT(2)/8 + 1/4, "
            "SQRT(6)/4]]",
        ),
    ),
    (
        "ROTATE_Y",
        ("ROTATE_Y(pi/2)", "[[0, 0, 1], [0, 1, 0], [-1, 0, 0]]"),
        (
            "(ROTATE_X(x)`*ROTATE_Y(y)`*ROTATE_Z(z)`)`",
            "[[COS(y)*COS(z), SIN(x)*SIN(y)*COS(z) - COS(x)*SIN(z), "
            "COS(x)*SIN(y)*COS(z) + SIN(x)*SIN(z)], [COS(y)*SIN(z), COS(x)*COS(z) "
            "+ SIN(x)*SIN(y)*SIN(z), COS(x)*SIN(y)*SIN(z) - SIN(x)*COS(z)], "
            "[-SIN(y), SIN(x)*COS(y), COS(x)*COS(y)]]",
        ),
    ),
    (
        "ROTATE_Z",
        (
            "(ROTATE_X(30*deg)`*ROTATE_Y(45*deg)`*ROTATE_Z(60*deg)`)`",
            "[[SQRT(2)/4, SQRT(2)/8 - 3/4, SQRT(3)/4 + SQRT(6)/8], [SQRT(6)/4, "
            "SQRT(3)/4 + SQRT(6)/8, 3*SQRT(2)/8 - 1/4], [-SQRT(2)/2, SQRT(2)/4, "
            "SQRT(6)/4]]",
        ),
        (
            "APPROX((ROTATE_X(150*deg)`*ROTATE_Y(135*deg)`*ROTATE_Z(120*deg)`)`)",
            "[[0.3535533905, 0.5732233047, 0.7391989197], [-0.6123724357, "
            "0.7391989197, -0.2803300858], [-0.7071067811, -0.3535533905, "
            "0.6123724357]]",
        ),
        ("ROTATE_Z(pi/2)", "[[0, -1, 0], [1, 0, 0], [0, 0, 1]]"),
        ("ROTATE_Z(σ)*ROTATE_Z(-σ)", "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"),
    ),
    (
        "ISOMETRICS",
        (
            "APPROX(FIRST(ISOMETRICS(SPHERE(1.5), φ, 0, pi, 6, θ, -pi, pi, 12)))",
            "[[0, 1.5], [0.75, 1.674038105], [1.299038105, 1.399519052], [1.5, "
            "0.75], [1.299038105, -0.1004809471], [0.75, -0.9240381056], [0, -1.5]]",
        ),
        (
            "ISOMETRICS([s, t, 0], s, 0, 1, 1, t, 0, 1, 1)",
            "[[[0, 0], [-1, -1/2]], [[1, -1/2], [0, -1]]]",
        ),
        ("DIM(FIRST(ISOMETRICS([s, t, s*t], s, 0, 1, 2, t, 0, 1, 1)))", "3"),
    ),
    (
        "COPROJECTION",
        ("COPROJECTION([[1, 2], [3, 4], [5, 6]])", "[[1, 3, 5], [2, 4, 6]]"),
        ("COPROJECTION([[1, 2, 3]])", "[[1], [2], [3]]"),
    ),
    (
        "RAYS",
        ("DIM(RAYS(z^2, z, 0, 1 + #i, 1, 1))", "2"),
        ("DIM(FIRST(RAYS(z, z, 0, 2 + #i, 2, 1)))", "3"),
    ),
    (
        "HORIZONTALS",
        (
            "HORIZONTALS(z^2, z, 0, 1 + #i, 1, 1)",
            "[[[0, 0], [1, 0]], [[-1, 0], [0, 2]]]",
        ),
        ("HORIZONTALS(z, z, 0, 1 + #i, 1, 1)", "[[[0, 0], [1, 0]], [[0, 1], [1, 1]]]"),
    ),
    (
        "ARCS",
        (
            "APPROX(FIRST(COPROJECTION(ARCS(#e^z, z, 0, pi, 4, -pi, pi, 24))))",
            "[[1, 0], [0.4559381277, 0], [0.2078795763, 0], [0.09478022484, 0], "
            "[0.04321391826, 0]]",
        ),
        ("ARCS(z, z, 1, 2, 1, 0, pi, 1)", "[[[1, 0], [-1, 0]], [[2, 0], [-2, 0]]]"),
        ("ARCS(z^2, z, 1, 1, 1, 0, pi/2, 1)", "[[[1, 0], [-1, 0]], [[1, 0], [-1, 0]]]"),
    ),
    (
        "GRID_LINES",
        ("DIM(GRID_LINES(0, 2))", "3"),
        # The text writes the segments `[M1; M2; ...]`, a column, so each segment is a
        # row of one element.
        ("FIRST(GRID_LINES(0, 2)) SUB 1", "[[[0, 0, 0], [2, 0, 0]]]"),
    ),
    (
        "GRID_POINTS",
        ("DIM(GRID_POINTS(0, 1))", "2"),
        ("GRID_POINTS(0, 1) SUB 1 SUB 2", "[[[1, 0, 0]], [[1, 0, 0]], [[0, 1, 0]]]"),
    ),
    (
        "POLYGON_FILL",
        (
            "FIRST(POLYGON_FILL([[0, 0], [1, 0], [0, 1]]))",
            "[[0, 0], [1, 0], [0, 1], [0, 0]]",
        ),
        ("DIM(POLYGON_FILL([[0, 0], [1, 0], [1, 1], [0, 1]]) SUB 2)", "2"),
    ),
    # -- BesselFunctions ---------------------------------------------------------------
    (
        "BESSEL_J",
        (
            "APPROX(VECTOR(BESSEL_J(j_, 2), j_, 5, 0, -1))",
            "[0.007039629755, 0.03399571980, 0.1289432494, 0.3528340286, "
            "0.5767248077, 0.2238907791]",
        ),
        ("APPROX(BESSEL_J(0, 1))", "0.7651976865"),
        ("APPROX(BESSEL_J(1, 2))", "0.5767248077"),
        ("APPROX(BESSEL_J(-1, 1))", "-0.4400505857"),
        ("BESSEL_J(1/2, pi/2)", "2/pi"),
        ("[BESSEL_J(0, 0), BESSEL_J(2, 0)]", "[1, 0]"),
    ),
    (
        "BESSEL_J_LIST",
        ("DIM(BESSEL_J_LIST(5, 1))", "7"),
        ("APPROX(BESSEL_J_LIST(20, 1) SUB 2)", "0.4400505857"),
    ),
    (
        "BESSEL_J_SERIES",
        ("BESSEL_J_SERIES(0, z, 2)", "z^4/64 - z^2/4 + 1"),
        ("BESSEL_J_SERIES(1, z, 1)", "z*(8 - z^2)/16"),
    ),
    (
        "BESSEL_J_ASYMP",
        ("BESSEL_J_ASYMP(1/2, z)", "SQRT(2)*SQRT(1/z)*SIN(z)/SQRT(pi)"),
        ("BESSEL_J_ASYMP(0, z)", "SQRT(2)*SQRT(1/z)*SIN(z + pi/4)/SQRT(pi)"),
    ),
    (
        "BESSEL_Y",
        ("APPROX(BESSEL_Y(0, 2))", "0.5103756726"),
        ("APPROX(BESSEL_Y(1, 1))", "-0.7812128213"),
        ("APPROX(BESSEL_Y(2, 1))", "-1.650682606"),
        ("BESSEL_Y(0, 0)", "-inf"),
    ),
    (
        "BESSEL_Y_SERIES",
        (
            "BESSEL_Y_SERIES(0, z, 1)",
            "-(z^2*LN(z) - 4*LN(z/2) - (LN(2) - euler_gamma + 1)*z^2 - "
            "4*euler_gamma)/(2*pi)",
        ),
        ("BESSEL_Y_SERIES(0, z, 0)", "2*(LN(z/2) + euler_gamma)/pi"),
    ),
    (
        "BESSEL_Y_ASYMP",
        ("BESSEL_Y_ASYMP(1/2, z)", "-SQRT(2)*SQRT(1/z)*COS(z)/SQRT(pi)"),
        ("BESSEL_Y_ASYMP(0, z)", "-SQRT(2)*SQRT(1/z)*COS(z + pi/4)/SQRT(pi)"),
    ),
    (
        "BESSEL_I",
        ("APPROX(BESSEL_I(1, 5))", "24.33564214"),
        ("APPROX(BESSEL_I(5, 5))", "2.157974547"),
        ("BESSEL_I(0, 0)", "1"),
        ("APPROX(BESSEL_I(0, 1))", "1.266065877"),
        ("APPROX(BESSEL_I(1, 1))", "0.5651591039"),
    ),
    (
        "BESSEL_I_SERIES",
        ("APPROX(BESSEL_I_SERIES(1, 5, 10))", "24.33564206"),
        ("APPROX(BESSEL_I_SERIES(5, 5, 10))", "2.157974547"),
        ("BESSEL_I_SERIES(1, z, 1)", "z*(z^2 + 8)/16"),
        ("BESSEL_I_SERIES(0, z, 2)", "z^4/64 + z^2/4 + 1"),
    ),
    (
        "BESSEL_I_ASYMP",
        ("BESSEL_I_ASYMP(1/2, z)", "SQRT(2)*#e^z/(2*SQRT(pi)*SQRT(z))"),
        ("BESSEL_I_ASYMP(0, z)", "SQRT(2)*(1/z + 8)*#e^z/(16*SQRT(pi)*SQRT(z))"),
    ),
    (
        "BESSEL_K",
        ("APPROX(BESSEL_K(0, 1))", "0.4210244382"),
        ("APPROX(BESSEL_K(1/2, 1))", "0.4610685044"),
    ),
    (
        "BESSEL_K_ASYMP",
        ("BESSEL_K_ASYMP(1/2, z)", "SQRT(2)*SQRT(pi)*SQRT(1/z)*#e^(-z)/2"),
        ("BESSEL_K_ASYMP(0, z)", "SQRT(2)*SQRT(pi)*(8 - 1/z)*SQRT(1/z)*#e^(-z)/16"),
    ),
    (
        "SPHERICAL_BESSEL_J",
        (
            "VECTOR(SPHERICAL_BESSEL_J(k_, z), k_, 5, 0, -1)",
            "[-(z^5*COS(z) - 105*z^3*COS(z) + 945*z*COS(z) - 15*z^4*SIN(z) + "
            "420*z^2*SIN(z) - 945*SIN(z))/z^6, (10*z^3*COS(z) - 105*z*COS(z) + "
            "z^4*SIN(z) - 45*z^2*SIN(z) + 105*SIN(z))/z^5, (z^3*COS(z) - "
            "15*z*COS(z) - 6*z^2*SIN(z) + 15*SIN(z))/z^4, -(3*z*COS(z) + "
            "z^2*SIN(z) - 3*SIN(z))/z^3, (SIN(z) - z*COS(z))/z^2, SIN(z)/z]",
        ),
        (
            "VECTOR(SPHERICAL_BESSEL_J(k_, 2), k_, 5, 0, -1)",
            "[-541*COS(2)/32 - 495*SIN(2)/64, -65*COS(2)/16 - 59*SIN(2)/32, "
            "-11*COS(2)/8 - 9*SIN(2)/16, -3*COS(2)/4 - SIN(2)/8, SIN(2)/4 - "
            "COS(2)/2, SIN(2)/2]",
        ),
        (
            "APPROX(VECTOR(SPHERICAL_BESSEL_J(k_, 2), k_, 2, 0, -1))",
            "[0.1984479490, 0.4353977749, 0.4546487134]",
        ),
        ("SPHERICAL_BESSEL_J(0, z)", "SIN(z)/z"),
        ("SPHERICAL_BESSEL_J(1, z)", "(SIN(z) - z*COS(z))/z^2"),
        ("SPHERICAL_BESSEL_J(-1, z)", "COS(z)/z"),
    ),
    (
        "SPHERICAL_BESSEL_J_LIST",
        ("DIM(SPHERICAL_BESSEL_J_LIST(4, 1))", "6"),
        ("APPROX(SPHERICAL_BESSEL_J_LIST(20, 1) SUB 1)", "0.8414709848"),
    ),
    (
        "SPHERICAL_BESSEL_Y",
        (
            "VECTOR(SPHERICAL_BESSEL_Y(k_, z), k_, 0, 5)",
            "[-COS(z)/z, -(COS(z) + z*SIN(z))/z^2, (z^2*COS(z) - 3*COS(z) - "
            "3*z*SIN(z))/z^3, (6*z^2*COS(z) - 15*COS(z) + z^3*SIN(z) - "
            "15*z*SIN(z))/z^4, -(z^4*COS(z) - 45*z^2*COS(z) + 105*COS(z) - "
            "10*z^3*SIN(z) + 105*z*SIN(z))/z^5, -(15*z^4*COS(z) - 420*z^2*COS(z) + "
            "945*COS(z) + z^5*SIN(z) - 105*z^3*SIN(z) + 945*z*SIN(z))/z^6]",
        ),
        (
            "VECTOR(SPHERICAL_BESSEL_Y(k_, 2), k_, 0, 5)",
            "[-COS(2)/2, -COS(2)/4 - SIN(2)/2, COS(2)/8 - 3*SIN(2)/4, 9*COS(2)/16 "
            "- 11*SIN(2)/8, 59*COS(2)/32 - 65*SIN(2)/16, 495*COS(2)/64 - "
            "541*SIN(2)/32]",
        ),
        (
            "APPROX(VECTOR(SPHERICAL_BESSEL_Y(k_, 2), k_, 0, 5))",
            "[0.2080734182, -0.3506120042, -0.7339914246, -1.484366557, "
            "-4.461291526, -18.59144531]",
        ),
        ("SPHERICAL_BESSEL_Y(0, z)", "-COS(z)/z"),
        ("SPHERICAL_BESSEL_Y(1, z)", "-(COS(z) + z*SIN(z))/z^2"),
    ),
    (
        "AI_SERIES",
        ("APPROX(AI_SERIES(0, 5))", "0.3550280538"),
        ("APPROX(AI_SERIES(1, 10))", "0.1352924163"),
    ),
    (
        "BI_SERIES",
        ("APPROX(BI_SERIES(0, 5))", "0.6149266274"),
        ("APPROX(BI_SERIES(1, 10))", "1.207423594"),
    ),
    # -- EllipticIntegrals -------------------------------------------------------------
    (
        "ELLIPTIC_F",
        ("ELLIPTIC_F(phi, 0)", "phi"),
        ("APPROX(ELLIPTIC_F(pi/2, 1/2))", "1.854074677"),
    ),
    (
        "ELLIPTIC_E",
        ("ELLIPTIC_E(pi/2, 0)", "pi/2"),
        ("APPROX(ELLIPTIC_E(pi/2, 1/2))", "1.350643881"),
    ),
    (
        "ELLIPTIC_PI",
        ("ELLIPTIC_PI(phi, 0, 0)", "phi"),
        ("APPROX(ELLIPTIC_PI(pi/2, 1/2, 1/4))", "2.167619360"),
    ),
    (
        "JACOBI_AM",
        ("APPROX(JACOBI_AM(1/2, 1/2, 6))", "0.4901412054"),
        ("APPROX(JACOBI_AM(1, 0, 4))", "1"),
    ),
    (
        "KI",
        ("KI(0, 5)", "pi/2"),
        ("APPROX(KI(1/2, 6))", "1.854074677"),
    ),
    (
        "EIF",
        ("APPROX(EIF(pi/2, 1/2, 10))", "1.854074677"),
        ("EIF(0, m, 3)", "0"),
    ),
    # -- ExponentialIntegrals ----------------------------------------------------------
    (
        "LI",
        ("LI(x, 1)", "LN(x) + LN(LN(x)) + euler_gamma"),
        ("APPROX(LI(10, 20))", "6.165599504"),
    ),
    (
        "EN",
        ("EN(0, 1)", "#e^(-1)"),
        ("APPROX(EN(1, 1))", "0.2193839343"),
        ("APPROX(EN(2, 1))", "0.1484955067"),
    ),
    (
        "EN_ASYMP",
        ("EN_ASYMP(1, z, 1)", "(1 - 1/z)*#e^(-z)/z"),
        ("EN_ASYMP(2, z, 0)", "#e^(-z)/z"),
    ),
    (
        "EI1",
        ("EI1(z, 2)", "-LN(z) - z^2/4 + z - euler_gamma"),
        ("APPROX(EI1(1, 20))", "0.2193839343"),
    ),
    # -- FresnelIntegrals --------------------------------------------------------------
    (
        "FRESNEL_SIN",
        ("APPROX(FRESNEL_SIN(2))", "0.3434156783"),
        ("APPROX(FRESNEL_SIN(1))", "0.4382591473"),
        ("FRESNEL_SIN(0)", "0"),
    ),
    (
        "FRESNEL_COS",
        ("APPROX(FRESNEL_COS(4))", "0.4984260330"),
        ("APPROX(FRESNEL_COS(3))", "0.6057207892"),
        ("APPROX(FRESNEL_COS(2))", "0.4882534060"),
        ("APPROX(FRESNEL_COS(1))", "0.7798934003"),
        ("FRESNEL_COS(0)", "0"),
    ),
    (
        "FRESNEL_SIN_SERIES",
        ("FRESNEL_SIN_SERIES(z, 0)", "pi*z^3/6"),
        ("FRESNEL_SIN_SERIES(z, 1)", "pi*z^3/6 - pi^3*z^7/336"),
    ),
    (
        "FRESNEL_COS_SERIES",
        ("FRESNEL_COS_SERIES(z, 1)", "z - pi^2*z^5/40"),
        ("FRESNEL_COS_SERIES(z, 0)", "z"),
    ),
    (
        "FRESNEL_SIN_J",
        ("APPROX(FRESNEL_SIN_J(1, 10))", "0.4382591473"),
        ("APPROX(FRESNEL_SIN_J(1/2, 10))", "0.06473243285"),
    ),
    (
        "FRESNEL_COS_J",
        ("APPROX(FRESNEL_COS_J(1, 10))", "0.7798934003"),
        ("APPROX(FRESNEL_COS_J(1/2, 10))", "0.4923442258"),
    ),
    (
        "FRESNEL_SIN_ASYMP",
        ("APPROX(FRESNEL_SIN_ASYMP(5))", "0.4991914016"),
        ("APPROX(FRESNEL_SIN_ASYMP(10))", "0.4681699789"),
    ),
    (
        "FRESNEL_COS_ASYMP",
        ("APPROX(FRESNEL_COS_ASYMP(5))", "0.5636310157"),
        ("APPROX(FRESNEL_COS_ASYMP(10))", "0.4998986942"),
    ),
    (
        "DAWSON",
        ("DAWSON(0)", "0"),
        ("APPROX(DAWSON(1))", "0.5380795069"),
    ),
    # -- HypergeometricFunctions -------------------------------------------------------
    (
        "HYPERGEOMETRIC_SERIES",
        ("HYPERGEOMETRIC_SERIES([], [], z, 2)", "z^2/2 + z + 1"),
        ("HYPERGEOMETRIC_SERIES([1, 1], [2], z, 2)", "z^2/3 + z/2 + 1"),
    ),
    (
        "KUMMER",
        ("KUMMER(1, 2, z)", "(#e^z - 1)/z"),
        ("APPROX(KUMMER(1, 2, 1))", "1.718281828"),
    ),
    (
        "KUMMER_SERIES",
        ("KUMMER_SERIES(1, 1, z, 3)", "z^3/6 + z^2/2 + z + 1"),
        ("KUMMER_SERIES(1, 2, 1, 5)", "1237/720"),
    ),
    (
        "GAUSS",
        ("GAUSS(1, 1, 3, 1)", "2"),
        ("APPROX(GAUSS(1, 1, 2, 1/2))", "1.386294361"),
    ),
    (
        "GAUSS_SERIES",
        ("GAUSS_SERIES(1, 1, 1, z, 3)", "z^3 + z^2 + z + 1"),
        ("GAUSS_SERIES(1, 1, 2, z, 2)", "z^2/3 + z/2 + 1"),
    ),
    # -- OrthogonalPolynomials ---------------------------------------------------------
    (
        "CHEBYCHEV_T",
        ("CHEBYCHEV_T(6, x)", "32*x^6 - 48*x^4 + 18*x^2 - 1"),
        ("CHEBYCHEV_T(6, 1/2)", "1"),
    ),
    (
        "CHEBYCHEV_T_LIST",
        ("DIM(CHEBYCHEV_T_LIST(7, x))", "7"),
        ("CHEBYCHEV_T_LIST(7, x) SUB 7", "32*x^6 - 48*x^4 + 18*x^2 - 1"),
        ("CHEBYCHEV_T_LIST(4, x)", "[1, x, 2*x^2 - 1, 4*x^3 - 3*x]"),
        ("CHEBYCHEV_T_LIST(5, 1/2)", "[1, 1/2, -1/2, -1, -1/2]"),
    ),
    (
        "CHEBYCHEV_U",
        ("CHEBYCHEV_U(6, x)", "64*x^6 - 80*x^4 + 24*x^2 - 1"),
        ("CHEBYCHEV_U(3, 1/2)", "-1"),
    ),
    (
        "CHEBYCHEV_U_LIST",
        ("DIM(CHEBYCHEV_U_LIST(7, x))", "7"),
        ("CHEBYCHEV_U_LIST(7, x) SUB 7", "64*x^6 - 80*x^4 + 24*x^2 - 1"),
        ("CHEBYCHEV_U_LIST(4, x)", "[1, 2*x, 4*x^2 - 1, 8*x^3 - 4*x]"),
        ("CHEBYCHEV_U_LIST(2, x)", "[1, 2*x]"),
    ),
    (
        "LEGENDRE_P",
        ("LEGENDRE_P(6, x)", "231*x^6/16 - 315*x^4/16 + 105*x^2/16 - 5/16"),
        ("LEGENDRE_P(3, 1/2)", "-7/16"),
    ),
    (
        "LEGENDRE_P_LIST",
        ("DIM(LEGENDRE_P_LIST(7, x))", "7"),
        ("LEGENDRE_P_LIST(7, x) SUB 7", "231*x^6/16 - 315*x^4/16 + 105*x^2/16 - 5/16"),
        ("LEGENDRE_P_LIST(4, x)", "[1, x, 3*x^2/2 - 1/2, 5*x^3/2 - 3*x/2]"),
        ("LEGENDRE_P_LIST(2, x)", "[1, x]"),
    ),
    (
        "ASSOCIATED_LEGENDRE_P",
        ("ASSOCIATED_LEGENDRE_P(4, 2, x)", "(1 - x^2)*(105*x^2 - 15)/2"),
        ("ASSOCIATED_LEGENDRE_P(1, 1, x)", "-SQRT(1 - x^2)"),
    ),
    (
        "HERMITE_H",
        ("HERMITE_H(6, x)", "64*x^6 - 480*x^4 + 720*x^2 - 120"),
        ("VECTOR(HERMITE_H(k, x), k, 0, 3)", "[1, 2*x, 4*x^2 - 2, 8*x^3 - 12*x]"),
    ),
    (
        "HERMITE_H_LIST",
        ("DIM(HERMITE_H_LIST(7, x))", "7"),
        ("HERMITE_H_LIST(7, x) SUB 7", "64*x^6 - 480*x^4 + 720*x^2 - 120"),
        ("HERMITE_H_LIST(4, x)", "[1, 2*x, 4*x^2 - 2, 8*x^3 - 12*x]"),
        ("HERMITE_H_LIST(2, x)", "[1, 2*x]"),
    ),
    (
        "HERMITE_HE",
        ("HERMITE_HE(6, x)", "x^6 - 15*x^4 + 45*x^2 - 15"),
        ("HERMITE_HE(2, x)", "x^2 - 1"),
    ),
    (
        "HERMITE_HE_LIST",
        ("DIM(HERMITE_HE_LIST(7, x))", "7"),
        ("HERMITE_HE_LIST(7, x) SUB 7", "x^6 - 15*x^4 + 45*x^2 - 15"),
        ("HERMITE_HE_LIST(4, x)", "[1, x, x^2 - 1, x^3 - 3*x]"),
        ("HERMITE_HE_LIST(2, x)", "[1, x]"),
    ),
    (
        "WEBER_D",
        ("WEBER_D(6, x)", "(x^6 - 15*x^4 + 45*x^2 - 15)*#e^(-x^2/4)"),
        ("WEBER_D(0, x)", "#e^(-x^2/4)"),
    ),
    (
        "LAGUERRE_L",
        ("LAGUERRE_L(4, x)", "x^4/24 - 2*x^3/3 + 3*x^2 - 4*x + 1"),
        ("LAGUERRE_L(2, x)", "x^2/2 - 2*x + 1"),
    ),
    (
        "LAGUERRE_L_LIST",
        ("DIM(LAGUERRE_L_LIST(5, x))", "5"),
        ("LAGUERRE_L_LIST(5, x) SUB 5", "x^4/24 - 2*x^3/3 + 3*x^2 - 4*x + 1"),
        (
            "LAGUERRE_L_LIST(4, x)",
            "[1, 1 - x, x^2/2 - 2*x + 1, -x^3/6 + 3*x^2/2 - 3*x + 1]",
        ),
        ("LAGUERRE_L_LIST(2, x)", "[1, 1 - x]"),
    ),
    (
        "GENERALIZED_LAGUERRE",
        ("GENERALIZED_LAGUERRE(4, 2, x)", "x^4/24 - x^3 + 15*x^2/2 - 20*x + 15"),
        ("GENERALIZED_LAGUERRE(1, a, x)", "-x + a + 1"),
    ),
    (
        "GENERALIZED_LAGUERRE_LIST",
        ("DIM(GENERALIZED_LAGUERRE_LIST(5, 2, x))", "5"),
        (
            "GENERALIZED_LAGUERRE_LIST(5, 2, x) SUB 5",
            "x^4/24 - x^3 + 15*x^2/2 - 20*x + 15",
        ),
        (
            "GENERALIZED_LAGUERRE_LIST(3, a, x)",
            "[1, -x + a + 1, x^2/2 - x*(a + 2) + a^2/2 + 3*a/2 + 1]",
        ),
        ("GENERALIZED_LAGUERRE_LIST(2, 0, x)", "[1, 1 - x]"),
    ),
    (
        "JACOBI_P",
        ("JACOBI_P(4, 2, 1, x)", "165*x^4/8 + 15*x^3/2 - 45*x^2/4 - 5*x/2 + 5/8"),
        ("JACOBI_P(1, a, b, x)", "x*(a/2 + b/2 + 1) + a/2 - b/2"),
    ),
    (
        "JACOBI_P_LIST",
        ("DIM(JACOBI_P_LIST(5, 2, 1, x))", "5"),
        (
            "JACOBI_P_LIST(5, 2, 1, x) SUB 5",
            "165*x^4/8 + 15*x^3/2 - 45*x^2/4 - 5*x/2 + 5/8",
        ),
        ("JACOBI_P_LIST(3, 1, 1, x)", "[1, 2*x, 15*x^2/4 - 3/4]"),
        ("JACOBI_P_LIST(2, 0, 0, x)", "[1, x]"),
    ),
    (
        "GEGENBAUER_C",
        ("GEGENBAUER_C(4, 2, x)", "80*x^4 - 48*x^2 + 3"),
        ("GEGENBAUER_C(2, a, x)", "a*(2*x^2*(a + 1) - 1)"),
    ),
    (
        "GEGENBAUER_C_LIST",
        ("DIM(GEGENBAUER_C_LIST(5, 2, x))", "5"),
        ("GEGENBAUER_C_LIST(5, 2, x) SUB 5", "80*x^4 - 48*x^2 + 3"),
        ("GEGENBAUER_C_LIST(3, a, x)", "[1, 2*a*x, a*(2*x^2*(a + 1) - 1)]"),
        ("GEGENBAUER_C_LIST(2, 1, x)", "[1, 2*x]"),
    ),
    # -- ProbabilityFunctions ----------------------------------------------------------
    (
        "POCHHAMMER",
        ("POCHHAMMER(a, 3)", "a*(a + 1)*(a + 2)"),
        ("POCHHAMMER(1, 5)", "120"),
    ),
    (
        "DIGAMMA_PSI",
        ("APPROX(DIGAMMA_PSI(1))", "-0.5772156649"),
        ("APPROX(DIGAMMA_PSI(2))", "0.4227843350"),
    ),
    (
        "INCOMPLETE_GAMMA",
        "> cdf_gam(p, l, x) := INCOMPLETE_GAMMA(p, x/l)",
        ("cdf_gam(3, 2, 1.5)", "1 - 65*#e^(-3/4)/32"),
        ("APPROX(cdf_gam(3, 2, 1.5))", "0.04050543974"),
    ),
    (
        "INCOMPLETE_GAMMA",
        ("APPROX(INCOMPLETE_GAMMA(0.5, 3))", "0.9856941215"),
        ("INCOMPLETE_GAMMA(1, w)", "1 - #e^(-w)"),
        ("APPROX(INCOMPLETE_GAMMA(2, 1))", "0.2642411176"),
    ),
    (
        "INCOMPLETE_GAMMA_SERIES",
        ("INCOMPLETE_GAMMA_SERIES(1, w, 2)", "w*(w^2/6 + w/2 + 1)*#e^(-w)"),
        ("INCOMPLETE_GAMMA_SERIES(1, w, 0)", "w*#e^(-w)"),
    ),
    (
        "EULER_BETA",
        ("EULER_BETA(1/2, 2)", "4/3"),
        ("APPROX(EULER_BETA(0.6, 5))", "0.5806391675"),
        ("EULER_BETA(2, 3)", "1/12"),
    ),
    (
        "INCOMPLETE_BETA",
        ("INCOMPLETE_BETA(x, 1, 1)", "x"),
        ("INCOMPLETE_BETA(1.5, 2, 4)", "9/16"),
        ("INCOMPLETE_BETA(-1/2, 1/2, 2)", "7*SQRT(2)*#i/8"),
        ("INCOMPLETE_BETA(1/2, 2, 3)", "11/16"),
    ),
    (
        "POISSON_DENSITY",
        ("POISSON_DENSITY(2, t)", "t^2*#e^(-t)/2"),
        ("POISSON_DENSITY(0, 1)", "#e^(-1)"),
    ),
    (
        "POISSON_DISTRIBUTION",
        ("POISSON_DISTRIBUTION(2, t)", "(t^2/2 + t + 1)*#e^(-t)"),
        ("POISSON_DISTRIBUTION(0, t)", "#e^(-t)"),
    ),
    (
        "BINOMIAL_DENSITY",
        ("BINOMIAL_DENSITY(3, 6, 3/4)", "135/1024"),
        ("APPROX(BINOMIAL_DENSITY(6, 21, 1/6))", "0.07548933698"),
    ),
    (
        "BINOMIAL_DISTRIBUTION",
        ("1 - BINOMIAL_DISTRIBUTION(5, 10, 1/4)", "10343/524288"),
        ("APPROX(1 - BINOMIAL_DISTRIBUTION(1, 6, 0.75))", "0.9953613281"),
        (
            "APPROX(TABLE(BINOMIAL_DISTRIBUTION(k, 50, 1/3), k, 15, 17))",
            "[[15, 0.3689669031], [16, 0.4867947092], [17, 0.6046225153]]",
        ),
    ),
    (
        "HYPERGEOMETRIC_DENSITY",
        ("HYPERGEOMETRIC_DENSITY(3, 10, 10, 20)", "3600/46189"),
        ("HYPERGEOMETRIC_DENSITY(1, 2, 2, 4)", "2/3"),
    ),
    (
        "HYPERGEOMETRIC_DISTRIBUTION",
        ("HYPERGEOMETRIC_DISTRIBUTION(1, 2, 2, 4)", "5/6"),
        ("HYPERGEOMETRIC_DISTRIBUTION(2, 2, 2, 4)", "1"),
    ),
    (
        "STUDENT",
        ("STUDENT(0, 5)", "1/2"),
        ("STUDENT(1, 1)", "3/4"),
        ("STUDENT(1, 2)", "SQRT(3)/6 + 1/2"),
        ("STUDENT(-1, 1)", "1/4"),
        # Over the interval searched the test of the IF is decided.
        ("NSOLVE(STUDENT(t, 16) = 0.95, t, 0, inf)", "t = 1.745883676"),
        (
            "STUDENT(x, df)",
            "IF(x < 0, 1 - STUDENT(-x, d*f), 1 - INCOMPLETE_BETA(d*f/(x^2 + d*f), "
            "d*f/2, 1/2)/2)",
        ),
    ),
    (
        "STUDENT",
        ("NSOLVE(STUDENT(t, 19) = 99%, t)", "t = 2.539483190"),
    ),
    (
        "F_DISTRIBUTION",
        ("APPROX(F_DISTRIBUTION(9, 2, 15))", "0.002702899474"),
        ("APPROX(F_DISTRIBUTION(105/19, 2, 10))", "0.02418065429"),
        ("APPROX(F_DISTRIBUTION(549/397, 2, 9))", "0.2994361085"),
        ("APPROX(F_DISTRIBUTION(4077/397, 2, 9))", "0.004756718049"),
        ("APPROX(F_DISTRIBUTION(504/397, 4, 9))", "0.3503277693"),
        ("APPROX(F_DISTRIBUTION(3897/677, 2, 6))", "0.04021658460"),
        ("APPROX(2*F_DISTRIBUTION(12/5, 24, 15))", "0.08245520715"),
        ("F_DISTRIBUTION(1, 2, 2)", "1/2"),
        ("F_DISTRIBUTION(0, 3, 4)", "1"),
    ),
    (
        "F_DISTRIBUTION",
        ("NSOLVE(F_DISTRIBUTION(x, 7, 7) = 0.05, x)", "x = 3.787043539"),
    ),
    (
        "CHI_SQUARE",
        ("CHI_SQUARE(x, 2)", "1 - #e^(-x/2)"),
        ("NSOLVE(CHI_SQUARE(x, 5) = 0.95, x)", "x = 11.07049769"),
    ),
    # -- ZetaFunctions -----------------------------------------------------------------
    (
        "HURWITZ_ZETA",
        ("HURWITZ_ZETA(2, 1, 0)", "5/3"),
        ("APPROX(HURWITZ_ZETA(2, 1, 20))", "1.644942445"),
    ),
    (
        "LERCH_PHI",
        ("LERCH_PHI(z, 1, 1, 2)", "z^2/3 + z/2 + 1"),
        ("LERCH_PHI(1/2, 1, 1, 3)", "131/96"),
    ),
]


def _case_ids(cases):
    """`NAME` for a function's first case, `NAME-2` for its second, and so on."""
    seen = {}
    ids = []
    for name, *_ in cases:
        seen[name] = seen.get(name, 0) + 1
        ids.append(name if seen[name] == 1 else f"{name}-{seen[name]}")
    return ids


@pytest.mark.parametrize("case", LIBRARY_CASES, ids=_case_ids(LIBRARY_CASES))
async def test_a_library_function_computes_its_specified_results(case):
    _, *lines = case
    session = Session()
    session.author("PrecisionDigits := 10")
    session.author("NotationDigits := 10")
    expected, found = [], []
    for line in lines:
        if isinstance(line, str):
            text = line.removeprefix("> ")
            if ":=" in text or ":epsilon" in text:
                session.author(text)
            else:
                await session.simplify(text)
            continue
        given, answer = line
        expected.append(answer)
        found.append(await simplified(session, given))
    assert found == expected
