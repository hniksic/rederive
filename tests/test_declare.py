"""The Declare commands: what each of them writes, and what it then means.

Every expected expression here is what the original appends for the same
answers. The commands themselves are driven through the app in `test_ui`;
what is tested here is the session side - that the expression is the one the
original writes, and that authoring it defines what it says it defines.
"""

import pytest

from rederive.model.session import NAMED_INTERVALS, Bounds, Session

#: Each word of the interval menu, and the declaration choosing it writes.
INTERVALS = [
    ("All", "x :ε Real"),
    ("Positive", "x :ε Real (0, ∞)"),
    ("Negative", "x :ε Real (-∞, 0)"),
    ("nonpoSitive", "x :ε Real (-∞, 0]"),
    ("nonneGative", "x :ε Real [0, ∞)"),
]


@pytest.fixture
def session():
    return Session()


def texts(session):
    return [entry.text for entry in session.entries]


def drawn(entry):
    """The one line an entry is drawn on.

    What a declaration is written in is the notation it is *shown* in - `:ε`
    and `∞` are glyphs and never text - so a test about the notation asks the
    render rather than the entry's own text, which is the canonical author
    notation a file holds.
    """
    assert len(entry.layout.lines) == 1
    return entry.layout.lines[0]


# -- Declare Variable, the domains -------------------------------------------


@pytest.mark.parametrize(("word", "expected"), INTERVALS, ids=str)
def test_each_named_interval_is_written_in_standard_notation(session, word, expected):
    bounds = NAMED_INTERVALS.get(word)
    assert drawn(session.declare_domain("x", "Real", bounds)) == expected


def test_the_domains_with_no_interval_are_written_alone(session):
    assert drawn(session.declare_domain("z", "Complex")) == "z :ε Complex"
    assert drawn(session.declare_domain("w", "Nonscalar")) == "w :ε Nonscalar"


def test_an_integer_domain_carries_its_interval_too(session):
    entry = session.declare_domain("n", "Integer", Bounds("1", "5", True, True))
    assert drawn(entry) == "n :ε Integer [1, 5]"
    assert entry.annotation == "User"


def test_an_infinite_bound_is_open_however_it_was_set(session):
    """A variable cannot be declared infinite, so `≤ ∞` is still `∞)`."""
    bounds = Bounds(closed_low=True, closed_high=True)
    assert drawn(session.declare_domain("x", "Real", bounds)) == "x :ε Real (-∞, ∞)"


async def test_a_declared_domain_is_what_lets_a_rewrite_fire(session):
    session.declare_domain("z", "Real", Bounds("0", "∞"))
    session.author("SQRT(z^2)")
    assert (await session.simplify("#2")).text == "z"


async def test_the_default_domain_stands_for_every_variable_nobody_named(session):
    session.author("SQRT(w^2)")
    assert (await session.simplify("#1")).text == "ABS(w)"
    session.declare_domain("default", "Complex")
    session.author("SQRT(q^2)")
    assert (await session.simplify("#4")).text == "SQRT(q^2)"


# -- Declare Variable Value --------------------------------------------------


async def test_a_value_is_written_as_an_assignment(session):
    entry = session.declare_value("area", "pi r^2")
    assert entry.text == "area:=pi*r^2"
    assert entry.layout.lines == ("           2", "area := π·r ")
    session.author("2 area")
    assert (await session.simplify("#2")).text == "2*pi*r^2"


def test_a_blank_value_leaves_the_name_an_unassigned_variable(session):
    session.declare_value("area", "pi r^2")
    assert session.declare_value("area").text == "area:="
    assert session.assignments == {}


# -- Declare Function --------------------------------------------------------


async def test_the_definitions_own_variables_become_the_parameters(session):
    entry = session.declare_function("hyp", "sqrt(a^2 + b^2)")
    assert entry.text == "HYP(a,b):=SQRT(a^2+b^2)"
    # The name is a function's now, so it is drawn as one: upper case.
    assert entry.layout.lines[-1] == "HYP(a, b) := √(a  + b )"
    session.author("HYP(3, 4)")
    assert (await session.simplify("#2")).text == "5"


def test_the_parameters_are_not_variables_of_the_session(session):
    """The command declares a function, and nothing else.

    Its parameters are known names only while the body is read: a later
    `A(t)` is a call, and would not be if `a` were a variable now.
    """
    session.declare_function("hyp", "sqrt(a^2 + b^2)")
    assert session.state.variables == {}
    assert session.declarable("a")


def test_the_parameters_are_ordered_most_main_first(session):
    """`x`, `y` and `z` are the order list; the rest follow alphabetically."""
    assert session.declare_function("p", "y + a + x + z + b").text.startswith(
        "P(x,y,z,a,b):="
    )


def test_a_variable_with_a_value_is_still_a_variable(session):
    session.declare_value("k", "5")
    assert session.declare_function("q", "k + z + m").text.startswith("Q(z,k,m):=")


def test_a_constant_and_a_function_are_not_variables(session):
    assert session.declare_function("f", "pi SIN(x)").text.startswith("F(x):=")


def test_a_definition_with_no_variables_defines_no_function(session):
    """Which is what the original writes for one: a variable with a value."""
    assert session.declare_function("q", "5").text == "q:=5"
    assert session.assignments["q"].value == "5"


def test_an_arbitrary_function_is_its_parameters_and_no_body(session):
    entry = session.declare_arbitrary("f", ("x", "y"))
    assert entry.text == "F(x,y):="
    assert entry.layout.lines == ("F(x, y) :=",)
    assert "F" in session.functions or session.state.functions["F"].params == ("x", "y")


def test_an_arbitrary_function_with_no_parameters_is_a_variable(session):
    assert session.declare_arbitrary("g", ()).text == "g:="
    assert "g" not in session.state.functions


# -- Declare Matrix and Declare vectoR ---------------------------------------


def test_a_vector_is_its_elements_between_brackets(session):
    entry = session.declare_vector(("1", "2", "x"))
    assert entry.text == "[1,2,x]"
    assert entry.layout.lines == ("[1, 2, x]",)


def test_a_matrix_is_a_vector_of_rows(session):
    entry = session.declare_matrix((("1", "2", "3"), ("4", "5", "6")))
    assert entry.text == "[[1,2,3],[4,5,6]]"
    assert entry.layout.lines == ("┌ 1  2  3 ┐", "│         │", "└ 4  5  6 ┘")


# -- what the screens open on ------------------------------------------------


def test_a_name_nothing_declared_is_a_real_variable(session):
    session.author("q + 1")
    assert session.declared_as("q") == "Real"
    assert session.declared_interval("q") == "All"
    assert session.bounds_of("q") == Bounds()


def test_a_variable_declared_without_a_domain_is_answered_by_value(session):
    session.declare_value("area", "pi r^2")
    assert session.declared_as("area") == "Value"
    # Clearing the value does not make it undeclared again.
    session.declare_value("area")
    assert session.declared_as("area") == "Value"


def test_a_declared_domain_answers_the_menu_it_was_chosen_from(session):
    session.declare_domain("p", "Real", Bounds("0", "∞"))
    assert session.declared_as("p") == "Real"
    assert session.declared_interval("p") == "Positive"
    session.declare_domain("z", "Complex")
    assert session.declared_as("z") == "Complex"


def test_a_name_is_looked_up_however_it_was_cased(session):
    """Case-insensitive mode is the default, so `AREA` is the same variable."""
    session.declare_value("area", "pi r^2")
    assert session.declared_as("AREA") == "Value"
    session.declare_domain("p", "Real", Bounds("0", "∞"))
    assert session.declared_interval("P") == "Positive"


def test_bounds_with_no_name_of_their_own_answer_interval(session):
    session.declare_domain("p", "Real", Bounds("1", "5", True, False))
    assert session.declared_interval("p") == "Interval"
    assert session.bounds_of("p") == Bounds("1", "5", True, False)


def test_the_bounds_screen_opens_on_the_bounds_the_variable_has(session):
    session.declare_domain("p", "Real", Bounds("-∞", "1/2", closed_high=True))
    assert session.bounds_of("p") == Bounds("-∞", "1/2", False, True)


# -- what the screens will take ----------------------------------------------


@pytest.mark.parametrize(
    ("name", "declarable"),
    [
        ("x", True),
        ("area", True),
        ("light_years", True),
        ("x1", True),
        ("default", True),
        ("", False),
        ("2x", False),
        ("x+1", False),
        ("SIN", False),
        ("pi", False),
        ("AND", False),
    ],
    ids=str,
)
def test_a_pre_defined_name_cannot_be_declared(session, name, declarable):
    assert session.declarable(name) is declarable


def test_a_declared_function_name_can_be_taken_back(session):
    """The Declare Variable command turns a function's name into a variable."""
    session.declare_function("hyp", "a + b")
    assert session.declarable("hyp")
    session.declare_domain("hyp", "Real")
    assert "HYP" not in session.state.functions


@pytest.mark.parametrize(
    ("text", "accepted"),
    [
        ("0", True),
        ("-5", True),
        ("1/2", True),
        ("2.5", True),
        ("∞", True),
        ("-∞", True),
        ("inf", True),
        ("a + 1", False),
        ("x", False),
        ("", False),
        ("2 +", False),
    ],
    ids=str,
)
def test_a_bound_is_a_number_or_an_infinity(session, text, accepted):
    assert session.is_bound(text) is accepted
