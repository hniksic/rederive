"""Sessions the Derive 3.14 manual describes, tested as the manual states them.

Every case here comes from a session the manual walks a reader through: type
this, get that. The expected value is what the printed manual says appears,
translated from its typeset form into author text - `·` written `*`, `√`
written `SQRT`, raised exponents written `^` - and nothing else. Where the
manual states an outcome only in prose ("simplifies to 6 and 60,
respectively") the prose is the expectation; where it states no outcome at
all, the case is not here.

The manual's own printed form is what these assert, so a failure means one of
two things: the engine does not do what the manual promises, or it does the
same mathematics in a different spelling. Both are worth seeing, and neither
is worth papering over here - `test_simplify` is where the engine's own
spelling is recorded.

Where the manual is wrong, a case asserts what the program was seen to do
instead, and says so in a comment beside it. Five such places are known: the
initial Exponential setting (6.1 contradicts 6.2, and the original's DERIVE.INI
says Auto; the research is at `model/settings.py`), the COVARIANT_METRIC_TENSOR
entry on p.238 (the program prints `v^2 + w^2`), ACOSH's radicals in 6.6
(`SQRT(z - 1)` leads), SIGN(3 + 4*#i) in 6.8 (the fraction sits under the #i),
and TAYLOR_INVERSE's example in 9.1, which passes four arguments to the
five-parameter function the same section declares - the manual's own answer,
written in y, proves the call it meant was the five-argument one.

Some expectations follow rules that a typeset page cannot state and only the
program itself can. The ones these tests lean on, taken from the original:
Simplify orders a sum in descending kernel order - function terms first
whatever their variable, compared variable-first among themselves along the
chain #e^x, ABS, ASIN, ATAN, LN, COS, TAN, SIN - and a product the other way
up: number, bare variables, powers of sums, then function terms with that
chain reversed. Factors of a factorization ascend by main variable, degree of
the base, multiplicity, then coefficients from the leading one down by
magnitude, the positive sign winning ties. A two-term sum whose canonical
head is negative is shown turned round (`2*x - x^2`, `(x + 2)*(2 - x)`);
longer sums keep their leading minus. A quotient the program's own arithmetic
produced survives to the display, where an identically shaped authored sum
stays apart. Two divergences between the program's DOS releases are known -
dot-product nesting, and LN(COS(x))'s second derivative, `- TAN(x)^2 - 1` on
the earlier only, which is what this engine follows. One divergence from the
program is deliberate and recorded here: a recursion that exceeds the size
guard returns what was worked out so far, where Derive exhausts memory and
returns nothing.

Plotting is out of scope for this project, so Chapter 5 and every plot
exercise elsewhere is left out.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager, suppress

import pytest

from rederive.engine.computing import (
    Amount,
    Branch,
    Context,
    Domain,
    DomainKind,
    approx,
    domain_of_node,
    expand,
    factor,
    simplify,
    solve,
)
from rederive.engine.context import Angle, Direction, Precision, TrigPower
from rederive.engine.remote import RemoteEngine
from rederive.model.session import Session
from rederive.syntax import ParseState, parse_expression
from sexpr import to_sexpr


def parse(text, state=None):
    return parse_expression(text, state or ParseState()).node


def simp(text, context=None, state=None):
    """`text` simplified, as the worksheet would show it."""
    return simplify(parse(text, state), context or Context(), state).text


def apx(text, context=None, state=None):
    """`text` approximated: what the approX command would show."""
    return approx(parse(text, state), context or Context(), state).text


def declared(*declarations, **settings):
    """A context whose domains are what these declarations say."""
    domains = {}
    for declaration in declarations:
        name, domain = domain_of_node(parse(declaration))
        domains[name] = domain
    return Context(domains=domains, **settings)


def expanded(text, *variables, amount=Amount.RATIONAL, context=None):
    """What Expand would put on the worksheet."""
    return expand(parse(text), context or Context(), amount, variables).text


def factored(text, amount=Amount.RATIONAL, *variables, context=None):
    """What Factor would put on the worksheet, at the amount chosen."""
    return factor(parse(text), context or Context(), amount, variables).text


def solutions(text, *variables, context=None):
    """The lines soLve would append, in order."""
    return [result.text for result in solve(parse(text), context or Context(), variables)]


@pytest.fixture
def session():
    return Session()


async def answer(session, *lines):
    """Author every line but the last, and simplify the last one."""
    for line in lines[:-1]:
        session.author(line)
    return (await session.simplify(lines[-1])).text


#: How long a manual session is given to answer before it is called a hang.
#: Two of the manual's own lines do not come back at all, and a test that
#: wedges the suite says less than one that fails - so those two are asked
#: through the worker engine, which is the half of the program Esc can stop.
#: The bound only has to be long enough to tell "never" from "slow": these two
#: run out of it every time, so keeping it short is what keeps the suite quick
#: while still reporting an unexpected pass if either ever starts answering.
PATIENCE = 2.0


@contextmanager
def abortable():
    """A session whose engine computes in a worker, so a runaway can be killed."""
    engine = RemoteEngine()
    try:
        yield Session(runner=engine), engine
    finally:
        engine.shutdown()


async def answered_within(session, engine, ask, patience=PATIENCE):
    """What `ask(session)` answers, or a failure saying it never did.

    `ask` hands back the command to wait for, so the bound can be put on the
    waiting and not on the command. One that runs out of it is stopped the way
    a user stops one - `engine.abort` is Esc - and what is reported is that it
    never answered rather than whatever the abort raised.
    """
    asking = asyncio.ensure_future(ask(session))
    answered, _ = await asyncio.wait((asking,), timeout=patience)
    if not answered:
        engine.abort()
        with suppress(BaseException):
            await asyncio.wait_for(asking, patience)
        pytest.fail(f"no answer in {patience:.0f} seconds")
    return asking.result()


#: The manual's promises this engine has decided not to keep, test id to
#: reason. These are decisions, not gaps: an unexpected pass here means the
#: engine drifted off a recorded decision, and the drift is what to look at.
NEVER_TO_HOLD = {
    # The bars read in and the screen draws them; what will not carry them is
    # `.text`, which is the author spelling and has one name for the function
    # whichever way it was typed. `|u|` is a `Kind.ABS` and the `ABS(u)` a
    # simplified answer holds is a call, so the two are not one tree either.
    # MAX and MIN want the single bar of `(x + y)/2` besides.
    "test_the_number_theory_functions[LCM(m, n)-|m*n|/GCD(n, m)]":
        "the writer spells every absolute value ABS; the screen draws the bars",
    "test_the_piecewise_functions[MAX(x, y)-|x - y|/2 + (x + y)/2]":
        "the writer spells every absolute value ABS; the screen draws the bars",
    "test_the_piecewise_functions[MIN(x, y)-(x + y)/2 - |x - y|/2]":
        "the writer spells every absolute value ABS; the screen draws the bars",
    "test_a_square_rooted_is_an_absolute_value_where_the_variable_is_real":
        "the writer spells every absolute value ABS; the screen draws the bars",
    "test_trigonometry[ACOS(z)-pi/2 - ASIN(z)]":
        "the arc rules of 6.4 are equivalences, not rewrites; ACOS stays ACOS",
    "test_trigonometry[ACOT(z)-pi/2 - ATAN(z)]":
        "ACOT is sympy's odd branch; the identity is false on the negative reals",
}


#: The manual's promises this engine does not keep yet, named by test id.
#:
#: These run and are expected to fail rather than being skipped, so that one
#: which starts holding is reported as an unexpected pass and can be struck
#: off. Nothing here says the manual is wrong: where the manual is wrong, the
#: case asserts the program's answer and passes.
NOT_YET_HELD = {
    # A quotient must survive from computation to display. Derive keeps the
    # quotient its own arithmetic produced - BERNOULLI_POLY's /66, NORMAL's /2,
    # EULER_POLY's common factor - and never builds one out of an authored sum.
    # Sympy distributes a number over a sum at construction, so by the time the
    # pipeline runs there is no quotient left to preserve; holding these needs
    # a preserved-quotient representation threaded through conversion, the
    # pipeline and the canonical rebuild.
    "test_the_error_and_zeta_functions[NORMAL(z, m, "
    "s)-(ERF(SQRT(2)*z/(2*s) - SQRT(2)*m/(2*s)) + 1)/2]",
    "test_the_worked_examples_of_the_number_theory_file[BERNOULLI_POLY("
    "10, x)-(66*x^10 - 330*x^9 + 495*x^8 - 462*x^6 + 330*x^4 - 99*x^2 + 5)/66]",
    "test_the_worked_examples_of_the_number_theory_file[EULER_POLY(10, "
    "x)-x*(x^9 - 5*x^8 + 30*x^6 - 126*x^4 + 255*x^2 - 155)]",
    # The accumulators carry the manual's own values - `FG(4)` is
    # `[3*epsilon*mu + mu^2 - 15*mu*sigma^2, 6*mu*sigma]` - and the page factors
    # the `mu` its three terms share. `normal.py` says why no common factor is
    # taken back out of a finished sum: the original does not do it
    # consistently, `y*x^2 + 2*y*x + 2*y` authored as it stands coming back as
    # it stands, so it has two fixed points for the one expression. This is that
    # decision met from the manual's side, and a difference in spelling only.
    "test_the_manuals_mutual_recursion_written_with_accumulators",
    # Sympy never comes back for these two of the manual's own sessions; they
    # are asked through the worker engine under PATIENCE and fail saying so,
    # instead of wedging the suite.
    "test_the_integral_the_manual_does_by_substitution",
    "test_the_iterates_of_the_manuals_fixed_point",
    # Each half is written over its own bar, as the original writes it, and the
    # arguments are what is left: sympy signs an even or odd function's argument
    # by its own alphabetical order, giving `COS(w - z)` where the original
    # writes `COS(z - w)`, and turning the sine's terms round with it. The order
    # list makes `z` the main name, which is what says which way round it goes.
    "test_a_product_of_sines_collects_into_the_manuals_identity[COS(z)*"
    "COS(w)-COS(z - w)/2 + COS(z + w)/2]",
    "test_a_product_of_sines_collects_into_the_manuals_identity[SIN(z)*"
    "COS(w)-SIN(z - w)/2 + SIN(z + w)/2]",
    "test_a_product_of_sines_collects_into_the_manuals_identity[SIN(z)*"
    "SIN(w)-COS(z - w)/2 - COS(z + w)/2]",
    # An approximate number is a rational here and in the original both, the
    # simplest one standing for the value: under Rational notation the original
    # shows pi as 355/113 and SQRT(3) as 5042/2911, neither of them a power of
    # two underneath. What differs is the tolerance the simplest one is sought
    # within, and the digits these want are what that difference comes to.
    # SQRT(10001) is exactly 20001/200 here, so the cancellation is total and
    # 0.005 stands where the manual reports 0.00499722; the two fractions of the
    # SUM are quotients of whole numbers, exact at six digits, so Approximate
    # mode reaches the 2/3 the manual keeps for Mixed, and rounding every step
    # instead answers 0.666534 rather than the page's 0.666622. Reaching these
    # means finding the rule the original picks its rational by.
    "test_catastrophic_cancellation_is_reported_as_the_manual_reports_it",
    "test_mixed_mode_subtracts_the_fractions_before_it_rounds",
    # The lesson of 3.8 p.45 holds and its digits do not: an approximation
    # re-approximated keeps the error it was made with. What the page has is a
    # decimal read back in, which the original answers 1.73205 as this engine
    # does; re-simplifying the approximation itself is what keeps the error, and
    # the original answers that with the 1.732050841 its 5042/2911 comes to.
    # This case asks the first and expects the second.
    "test_reapproximating_an_approximation_keeps_its_error",
    # The value is the manual's and the spelling is the engine's: APPROX(SQRT(3),
    # 10) is 262087/151316, which is 1.732050807 to ten digits, and Rational
    # notation writes the ratio. Two decisions stand between the two - an
    # approximate number is a rational, and how many digits it is shown to is the
    # notation's business rather than the call's - and `test_simplify` records
    # both.
    "test_approx_takes_the_digits_it_is_given",
    # Two causes in one case. Exact mode needs the denesting the radical below
    # wants: the line is (SQRT(3) + 1)^(3/2) over twice itself, and simplify,
    # radsimp and sqrtdenest all leave it standing. Mixed mode then differs in
    # noise and not in mathematics, and the two pull against each other, since
    # rounding is the last step of the pipeline and an exact mode that reached
    # 1/2 would have Mixed show 1/2 too.
    "test_the_three_modes_on_an_expression_worth_one_half",
    # The denesting holds and the order it is written in does not. A kernel that
    # holds no variable sorts by its spelling, so a sum of pure surds comes out
    # by ascending radicand. Neither direction fits the manual: 3.8 p.42 prints
    # `SQRT(3) + SQRT(2)` and 6.3 p.104 prints `-SQRT(2)/8 - SQRT(6)/8 + 1/2`.
    "test_exact_mode_denests_the_manuals_radicals[SQRT(5 + "
    "2*SQRT(6))-SQRT(3) + SQRT(2)]",
    # The determinant is multiplied out where the original leaves it factored,
    # and then written about the most main variable it holds: with none of `a`,
    # `b`, `w` on the order list that is `a`, so `(2 - w)*(b - w) - 3*a` comes
    # back as `-3*a + b*(2 - w) + w^2 - 2*w`. The list cannot simply grow a `w`:
    # COVARIANT_METRIC_TENSOR's `v^2 + w^2` on p.238 has `v` more main than `w`,
    # which is what unlisted names being alphabetical gives.
    "test_the_characteristic_variable_defaults_to_w",
    # NEWTONS comes back as the ITERATES it is defined as. Three things stand in
    # the way: `GRAD(u, x)` over a vector `u` is an inert head and not the
    # Jacobian SOLVE.MTH builds its augmented matrix from; `LIM(m, [x, y],
    # [a, b])` over a matrix is not carried out; and the file's iteration
    # variable `xk` lexes as `x*k` under Character input, so ITERATES is handed a
    # product where a variable belongs.
    "test_newtons_finds_the_manuals_nonlinear_solution",
    # Two gaps, both in what ODE1.MTH's own definitions need. A four-argument
    # `IF` decides its test before the calculus pass runs, so `IF(DIF(u, x), ...)`
    # - how the file asks whether an expression depends on a variable - is
    # undecidable where Derive answers it. And an assignment written inside a
    # function body is performed when the body is defined rather than when the
    # call is made, so loading the file leaves `a_` standing for
    # `SEP(p, q, x, y, x0, y0)` with the parameters unwritten - and `a_` is where
    # DSOLVE1 reads its answer back out of.
    "test_the_first_order_equation_the_manual_solves",
    # `LIM(u, x, a)` where `a` holds `x` is no limit - nothing approaches a
    # moving point - and the file means the substitution: `LIM((x*v - y)^2, y,
    # x*v - y)` is `y^2`. The engine builds a sympy limit, so the second
    # condition comes back as a tower of SUBS over DIF over LIM. Reading such a
    # call as the substitution gets it to `2*v*x^2 - 2*x*y - 2*v = 0`, one common
    # factor short of the manual's line, and costs test_simplify's
    # `test_a_derivative_a_substitution_binds_is_not_evaluated`.
    "test_the_clairaut_equation_the_manual_solves",
    # `F(4)` comes back an unfinished tower of `DIF` over `F(0)`. A calculus
    # head over a call still waiting for its body waits for it, which is what
    # makes `F(3)` the `3*mu*sigma` it is; but a pass writes each pending body
    # in one level at a time, and nothing collapses until the innermost `F(0)`
    # answers, so by four levels down the tree is past the size guard. Derive
    # works such a call out where it stands, innermost first, which is what
    # keeps every intermediate small. Reached with the guard lifted, the value
    # is the manual's and the spelling is
    # `3*epsilon*mu + mu^2 - 15*mu*sigma^2` - the same common `mu` that holds
    # the accumulator version above. The empty `G(n) :=` changes nothing either
    # way, the same session without it answering the same.
    "test_an_empty_definition_makes_a_name_a_function_and_not_a_variable",
    # The brackets are read as the vector they are and the scalar goes in; what
    # comes back is `[3*x + 3*y]`. The element is then written in the normal form
    # of its main variable, which multiplies the three through where the manual's
    # `[3*(x + y)]` keeps it folded.
    "test_square_brackets_make_a_vector_rather_than_a_group",
    # `2^3=` parses to a `showvalue` node, and turning that into `2^3 = 8` means
    # computing the 8 while the line is being authored. `Session.author` is the
    # one entry point that computes nothing: every calculation the session makes
    # is awaited through `Runner`, which the app fills with a child process so
    # that a long one can be aborted and the drawing process knows no sympy.
    "test_an_expression_typed_with_a_trailing_equals_shows_both_sides",
    # The terms are the manual's own; the order they are printed in is not.
    # Expand's answer is ordered by the rule every sum is, where a term with a
    # sum under the bar closes the sum it belongs to - which is the original's
    # rule too, `(x^3 + 1)/(x + 2) + y` being `x^2 - 2*x + y - 7/(x + 2) + 4`.
    # Here the manual puts the partial fractions first instead, `(x - 3)^-1`
    # ahead of `(x + 2)^-2` ahead of `(x + 2)^-1`, which is no order in the
    # degree of x. One rule cannot give both.
    "test_the_partial_fraction_expansion_of_the_manual",
}


@pytest.fixture(autouse=True)
def what_is_not_held(request):
    """Expect the failure of every case the two manifests name."""
    test_id = request.node.nodeid.split("::", 1)[1]
    if test_id in NEVER_TO_HOLD:
        request.node.add_marker(pytest.mark.xfail(reason=NEVER_TO_HOLD[test_id]))
    elif test_id in NOT_YET_HELD:
        request.node.add_marker(
            pytest.mark.xfail(reason="the manual says so; the engine does not yet")
        )


COMPLEX_Z = Context(domains={"z": Domain(DomainKind.COMPLEX)})
REAL_XY = declared("x :epsilon Real", "y :epsilon Real")
DEGREES = Context(angle=Angle.DEGREE)


# == Chapter 3: Arithmetic ====================================================

# -- 3.1 entering expressions: what the operators mean ------------------------

#: The manual states the precedence and associativity rules but prints no
#: values for them; the value each rule picks out is what a test can see.
PRECEDENCE = [
    # `*` and `/` bind tighter than `+` and `-` (3.1, p.24).
    ("3 + 4 * 5", "23"),
    # Equal precedence applies left to right, so this is (1/2)*3 (3.1, p.25).
    ("1 / 2 * 3", "3/2"),
    # A space or a parenthesis between two expressions multiplies (3.1, p.25).
    ("2 (3 + 5)", "16"),
    # `3½ - 2¼` is written with plus signs, or the juxtaposition would
    # multiply (3.1, p.25).
    ("(3+1/2) - (2+1/4)", "5/4"),
    # `^` binds tighter than the four, so `9^1/3` is not a cube root.
    ("9^1/3", "3"),
    # `^` associates to the right, unlike the rest (3.1, p.26).
    ("4^3^2", "262144"),
    # Percent is one hundredth, and binds tighter than any of them (3.1, p.26).
    ("22% + 36%", "29/50"),  # the manual's "equal to 0.58", written exactly
    # A function of one integer argument needs no parentheses, and a power
    # written on the name applies to the result (3.4, p.32).
    ("SQRT 25", "5"),
    ("SQRT^3 25", "125"),
]


@pytest.mark.parametrize(("text", "expected"), PRECEDENCE, ids=str)
def test_the_operators_group_the_way_the_manual_says(text, expected):
    assert simp(text) == expected


async def test_the_manuals_first_arithmetic_is_not_simplified_until_asked(session):
    # 3.1 p.27: authoring `2 (8 + 7)/3^2` displays it built up, unsimplified.
    entry = session.author("2 (8 + 7) / 3^2")
    assert entry.text == "2*(8+7)/3^2"
    lines = (" 2·(8 + 7) ", "───────────", "      2    ", "     3     ")
    assert entry.layout.lines == lines
    # 3.2 p.28: Simplify answers 10/3, and says which line it came from.
    simplified = await session.simplify("#1")
    assert simplified.text == "10/3"
    assert simplified.annotation == "Simp(#1)"


def test_an_expression_typed_with_a_trailing_equals_shows_both_sides(session):
    # 3.2 p.29: `2^3=` displays the unsimplified form equated to the answer.
    assert session.author("2^3=").text == "2^3 = 8"


# -- 3.4 the functions the chapter introduces ---------------------------------

FUNCTIONS = [
    ("GCD(12, 30)", "6"),
    ("LCM(12, 30)", "60"),
    ("AVERAGE(3, 7)", "5"),
    ("|-8|", "8"),
    ("SQRT(4 + 5)", "3"),
    # 3.4 p.33: the root of a fraction is rationalized but not approximated.
    ("SQRT(1/12)", "SQRT(3)/6"),
]


@pytest.mark.parametrize(("text", "expected"), FUNCTIONS, ids=str)
def test_the_functions_of_chapter_three(text, expected):
    assert simp(text) == expected


# -- 3.8 precision ------------------------------------------------------------

def test_exact_mode_keeps_a_surd_and_takes_the_root_it_can():
    # 3.8 p.42: √4 is 2, √3 stays symbolic.
    assert simp("SQRT(4)") == "2"
    assert simp("SQRT(3)") == "SQRT(3)"


DENESTING = [
    # 3.8 p.42 and p.47: the nested radicals the manual denests in exact mode.
    ("SQRT(4 - 2*SQRT(3))", "SQRT(3) - 1"),
    ("SQRT(5 + 2*SQRT(6))", "SQRT(3) + SQRT(2)"),
    ("(243*SQRT(5) - 294*SQRT(3))^(1/3)", "3*SQRT(5) - 2*SQRT(3)"),
]


@pytest.mark.parametrize(("text", "expected"), DENESTING, ids=str)
def test_exact_mode_denests_the_manuals_radicals(text, expected):
    assert simp(text) == expected


def test_an_exact_power_of_a_fraction_is_the_whole_ratio():
    # 3.8 p.43: (7/8)^99 is computed exactly, as a reduced ratio.
    assert simp("(7/8)^99") == f"{7**99}/{8**99}"


# Taking the precision carries the notation with it, the way Options Precision
# does in the program and the way section 3.9 says it does: approximate
# arithmetic is read in the style that suits it.
APPROXIMATE_6 = Context().with_precision(Precision.APPROXIMATE)
APPROXIMATE_10 = Context().with_precision(Precision.APPROXIMATE, 10)
APPROXIMATE_100 = Context().with_precision(Precision.APPROXIMATE, 100)
MIXED = Context().with_precision(Precision.MIXED)


def test_approximate_mode_gives_the_digits_the_precision_asks_for():
    # 3.8 pp.44-45: √3 at six digits, then ten, then a hundred.
    assert simp("SQRT(3)", APPROXIMATE_6) == "1.73205"
    assert simp("SQRT(3)", APPROXIMATE_10) == "1.732050807"
    assert simp("SQRT(3)", APPROXIMATE_100).startswith(
        "1.7320508075688772935274463415058723669428052538103806280558069794519330169"
    )


def test_reapproximating_an_approximation_keeps_its_error():
    # 3.8 p.45: 1.73205 is only good to six digits, so simplifying *it* at ten
    # digits answers 1.732050657 rather than √3's 1.732050807.
    assert simp("1.73205", APPROXIMATE_10) == "1.732050657"


def test_mixed_mode_subtracts_the_fractions_before_it_rounds():
    # 3.8 p.46: approximate mode rounds first and loses it; mixed mode does
    # the rational arithmetic exactly and lands on 2/3.
    radical = "SQRT(3422357/2313 - 1140443/771)"
    assert simp(radical, APPROXIMATE_6) == "0.666622"
    assert simp(radical, MIXED) == "2/3"


def test_the_three_modes_on_an_expression_worth_one_half():
    # 3.8 p.47: exact says 1/2, approximate says 0.5, mixed drifts to 0.499999.
    text = "(SQRT(SQRT(3) + 1) + SQRT(3*SQRT(3) + 3))/(SQRT(24*SQRT(3) + 40))"
    assert simp(text) == "1/2"
    assert simp(text, APPROXIMATE_6) == "0.5"
    assert simp(text, MIXED) == "0.499999"


def test_catastrophic_cancellation_is_reported_as_the_manual_reports_it():
    # 3.8 p.46: at six digits √10001 - 100 answers 0.00499722, of which the
    # manual says only the first three digits are right.
    assert simp("SQRT(10001) - 100", APPROXIMATE_6) == "0.00499722"


# -- 3.9 notation, 3.10 approximating -----------------------------------------

def test_pi_is_approximated_as_digits_or_as_a_ratio():
    # 3.9 p.50: six digits of π, then the simplest ratio good to six digits.
    from dataclasses import replace

    from rederive.engine.context import Notation

    assert simp("pi", APPROXIMATE_6) == "3.14159"
    assert simp("pi", replace(APPROXIMATE_6, notation=Notation.RATIONAL)) == "355/113"


def test_exact_mode_leaves_pi_alone_and_approx_does_not():
    # 3.10 p.51: Simplify keeps π; approX is the command that spends it.
    assert simp("pi") == "pi"
    assert apx("pi") == "3.14159"


def test_approx_takes_the_digits_it_is_given():
    # 3.10 p.51: APPROX(u, n) approximates u to n digits.
    assert simp("APPROX(SQRT(3), 10)") == "1.732050807"


# == Chapter 6: Functions and Constants =======================================

# -- 6.1 exponential, 6.2 logarithmic -----------------------------------------

EXPONENTIAL_AND_LOG = [
    # 6.0 p.98: roots of fractions, exactly.
    ("SQRT(4/9)", "2/3"),
    ("SQRT(8/9)", "2*SQRT(2)/3"),
    # 6.0 p.98: a complex argument in rectangular form answers in it.
    ("SQRT(3 - 4*#i)", "2 - #i"),
    # 6.1 p.99.
    ("EXP(z)", "#e^z"),
    # 6.2 p.101.
    ("LN(#e^3)", "3"),
    ("LOG(8, 2)", "3"),
    ("LOG(z, w)", "LN(z)/LN(w)"),
    ("LOG(z)", "LN(z)"),
]


@pytest.mark.parametrize(("text", "expected"), EXPONENTIAL_AND_LOG, ids=str)
def test_exponentials_and_logarithms(text, expected):
    assert simp(text) == expected


def test_the_exponential_direction_starts_out_automatic():
    # 6.1 says "The initial direction is Collect", and is the one place here
    # where the manual is taken to be wrong rather than unmet: Collect was the
    # 1.x default and 6.1 is a leftover of it, where the original's own
    # DERIVE.INI says `*EXP-EXPD* |Auto|` and so does its screen.
    # `model.settings` carries the evidence.
    assert Context().exponential is Direction.AUTO


def test_the_logarithm_and_trigonometry_directions_start_out_automatic():
    # 6.2 p.102 and 6.3 p.105: both initially Auto.
    assert Context().logarithm is Direction.AUTO
    assert Context().trigonometry is Direction.AUTO
    assert Context().trigpower is TrigPower.AUTO


def test_a_collected_product_of_exponentials_is_one_exponential():
    # 6.1 p.99: ê^z·ê^w <-> ê^(z + w), collecting toward the right.
    collect = Context(exponential=Direction.COLLECT)
    assert simp("#e^z*#e^w", collect) == "#e^(z + w)"


def test_an_expanded_exponential_of_a_sum_is_a_product():
    # 6.1 p.99 states the rule as `#e^z*#e^w <-> #e^(z + w)`; what comes back
    # puts the off-list `w` first, a product ordering a name the list does not
    # know ahead of one it does.
    expand = Context(exponential=Direction.EXPAND)
    assert simp("#e^(z + w)", expand) == "#e^w*#e^z"


def test_logarithms_collect_and_expand_where_the_domain_allows_it():
    # 6.2 p.101: LN(x) + LN(z) <-> LN(x·z) needs x nonnegative.
    positive = declared("x :epsilon Real (0, inf)", logarithm=Direction.COLLECT)
    assert simp("LN(x) + LN(z)", positive) == "LN(x*z)"
    expand = declared("x :epsilon Real (0, inf)", logarithm=Direction.EXPAND)
    assert simp("LN(x*z)", expand) == "LN(x) + LN(z)"


def test_a_logarithm_is_not_collected_where_the_domain_does_not_allow_it():
    # 6.0 p.98: LN(x²)/2 -> LN(x) is invalid for negative x, and Derive does
    # not make a transformation it cannot justify.
    assert simp("LN(x^2)/2") != "LN(x)"


# -- 6.3 trigonometric, 6.4 inverse trigonometric -----------------------------

TRIGONOMETRY = [
    # 6.3 p.103: deg is π/180 radians.
    ("60*deg", "pi/3"),
    ("SEC(z)", "1/COS(z)"),
    ("CSC(z)", "1/SIN(z)"),
    # 6.3 p.107.
    ("SIN(pi/4)", "SQRT(2)/2"),
    ("ASIN(1/2)", "pi/6"),
    # 6.4 p.108. A table of equivalences rather than a rewrite system: the
    # `ACOS` line applied to the `ASEC` line's answer leaves `pi/2 -
    # ASIN(1/z)`, which is neither form the section prints. The engine keeps
    # `ASIN`, `ACOS` and `ATAN` and writes the reciprocal arcs over them, so
    # the two lines about reciprocals hold and the `ACOS` line does not.
    #
    # The `ACOT` line is about a branch this engine does not use: its `ACOT` is
    # the odd one, where `ACOT(-1)` is `-pi/4`, and `pi/2 - ATAN(z)` is the
    # other one - the two agree above zero and are a half turn apart below it.
    ("ACOT(z)", "pi/2 - ATAN(z)"),
    ("ACOT(x, y)", "ATAN(y, x)"),
    ("ACOS(z)", "pi/2 - ASIN(z)"),
    ("ASEC(z)", "ACOS(1/z)"),
    ("ACSC(z)", "ASIN(1/z)"),
]


@pytest.mark.parametrize(("text", "expected"), TRIGONOMETRY, ids=str)
def test_trigonometry(text, expected):
    assert simp(text) == expected


DEGREE_MODE = [
    ("SIN(45)", "SQRT(2)/2"),
    ("ASIN(1/2)", "30"),
]


@pytest.mark.parametrize(("text", "expected"), DEGREE_MODE, ids=str)
def test_degree_mode_reads_and_writes_degrees(text, expected):
    # 6.3 p.107: the same two lines, in the other angular unit.
    assert simp(text, DEGREES) == expected


def test_the_angle_unit_starts_out_radian():
    # 6.3 p.106: "The initial angular unit is radian."
    assert Context().angle is Angle.RADIAN


#: 6.3 p.104 states these as the transformations the Expand direction applies,
#: written the way an identity is written rather than the way an answer comes
#: back. What the original prints puts the off-list `w` first, which is where a
#: product puts a name the order list does not know, and is the order of the
#: `z!/(w!*(z - w)!)` that COMB(z, w) prints on its own page.
EXPANDED_TRIGONOMETRY = [
    ("SIN(z + w)", "SIN(w)*COS(z) + COS(w)*SIN(z)"),
    ("COS(z + w)", "COS(w)*COS(z) - SIN(w)*SIN(z)"),
]


@pytest.mark.parametrize(("text", "expected"), EXPANDED_TRIGONOMETRY, ids=str)
def test_an_angle_sum_expands_into_the_manuals_identity(text, expected):
    assert simp(text, Context(trigonometry=Direction.EXPAND)) == expected


#: The converse transformations of 6.3 p.104, again as the page states them
#: rather than as they come back: the original writes each half over its own
#: bar, and its arguments run `z - w`.
COLLECTED_TRIGONOMETRY = [
    ("SIN(z)*SIN(w)", "COS(z - w)/2 - COS(z + w)/2"),
    ("COS(z)*COS(w)", "COS(z - w)/2 + COS(z + w)/2"),
    ("SIN(z)*COS(w)", "SIN(z - w)/2 + SIN(z + w)/2"),
]


@pytest.mark.parametrize(("text", "expected"), COLLECTED_TRIGONOMETRY, ids=str)
def test_a_product_of_sines_collects_into_the_manuals_identity(text, expected):
    assert simp(text, Context(trigonometry=Direction.COLLECT)) == expected


def test_a_square_is_pushed_toward_the_function_that_is_asked_for():
    # 6.3 p.106: Toward Sines rewrites cosines, and the other way about.
    assert simp("COS(z)^2", Context(trigpower=TrigPower.SINES)) == "1 - SIN(z)^2"
    assert simp("SIN(z)^2", Context(trigpower=TrigPower.COSINES)) == "1 - COS(z)^2"


def test_a_sine_of_a_sixteenth_of_pi_expands_into_radicals():
    # 6.3 p.104: arguments of the form π/n for n a power of two times 1, 3, 5
    # or 15 become algebraic numbers written with radicals.
    assert simp("SIN(pi/24)", Context(trigonometry=Direction.EXPAND)) == (
        "SQRT(-SQRT(2)/8 - SQRT(6)/8 + 1/2)"
    )


# -- 6.5 hyperbolic, 6.6 inverse hyperbolic -----------------------------------

HYPERBOLIC = [
    ("SINH(z)", "#e^z/2 - #e^(-z)/2"),
    ("COSH(z)", "#e^z/2 + #e^(-z)/2"),
    ("TANH(z)", "(#e^(2*z) - 1)/(#e^(2*z) + 1)"),
    ("COTH(z)", "(#e^(2*z) + 1)/(#e^(2*z) - 1)"),
    ("SECH(z)", "2*#e^z/(#e^(2*z) + 1)"),
    ("CSCH(z)", "2*#e^z/(#e^(2*z) - 1)"),
]


@pytest.mark.parametrize(("text", "expected"), HYPERBOLIC, ids=str)
def test_the_hyperbolic_functions_become_exponentials(text, expected):
    assert simp(text) == expected


INVERSE_HYPERBOLIC = [
    ("ASINH(z)", "LN(SQRT(z^2 + 1) + z)"),
    # 6.6 p.111 prints the radicals as SQRT(z + 1) + SQRT(z - 1), but the
    # original answers with SQRT(z - 1) first, as its radical order elsewhere
    # demands.
    ("ACOSH(z)", "2*LN(SQRT(z - 1) + SQRT(z + 1)) - LN(2)"),
    ("ATANH(z)", "LN(z + 1)/2 - LN(1 - z)/2"),
    ("ACOTH(z)", "LN((z + 1)/(z - 1))/2"),
]


@pytest.mark.parametrize(("text", "expected"), INVERSE_HYPERBOLIC, ids=str)
def test_the_inverse_hyperbolic_functions_become_logarithms(text, expected):
    assert simp(text) == expected


# -- 6.7 piecewise continuous -------------------------------------------------

PIECEWISE = [
    ("|5|", "5"),
    ("|-5|", "5"),
    ("SIGN(3)", "1"),
    ("SIGN(-3)", "-1"),
    # 6.7 p.111: the sign of zero is indeterminate, and ±1 is a value.
    ("SIGN(0)", "±1"),
    ("STEP(x)", "SIGN(x)/2 + 1/2"),
    ("FLOOR(5.73)", "5"),
    ("MOD(m, 0)", "m"),
    ("MODS(m, 0)", "m"),
    # 6.7 p.112: a nonnumeric argument leaves the floor form behind.
    ("MOD(m, n)", "m - n*FLOOR(m, n)"),
    ("MODS(m, n)", "m - n*FLOOR(m + n/2, n)"),
    # 6.7 p.111: the characteristic function of an interval.
    ("CHI(1, 2, 3)", "1"),
    ("CHI(2, 1, 3)", "0"),
    ("CHI(1, 3, 2)", "0"),
    ("MAX(x, y)", "|x - y|/2 + (x + y)/2"),
    ("MIN(x, y)", "(x + y)/2 - |x - y|/2"),
    ("CHI(a, x, b)", "SIGN(x - a)/2 - SIGN(x - b)/2"),
]


@pytest.mark.parametrize(("text", "expected"), PIECEWISE, ids=str)
def test_the_piecewise_functions(text, expected):
    assert simp(text) == expected


def test_a_reversed_characteristic_interval_changes_sign():
    # 6.7 p.111: CHI(a, x, b) with a > b is -CHI(b, x, a). The page states that
    # as an equivalence and no CHI survives to be printed: a characteristic
    # function comes back as the difference of signs it stands for, and the two
    # reversed intervals cancelling is what says the sign changed.
    assert simp("CHI(3, x, 1)") == "SIGN(x - 3)/2 - SIGN(x - 1)/2"
    assert simp("CHI(3, x, 1) + CHI(1, x, 3)") == "0"


# -- 6.8 complex variable functions -------------------------------------------

COMPLEX_FUNCTIONS = [
    ("|x + #i*y|", "SQRT(x^2 + y^2)"),
    ("RE(x + #i*y)", "x"),
    ("IM(x + #i*y)", "y"),
    ("CONJ(x + #i*y)", "x - #i*y"),
]


@pytest.mark.parametrize(("text", "expected"), COMPLEX_FUNCTIONS, ids=str)
def test_the_complex_functions_on_a_rectangular_argument(text, expected):
    assert simp(text, REAL_XY) == expected


def test_the_sign_of_a_complex_number_is_its_point_on_the_unit_circle():
    # 6.8 p.113, which prints 4/5 as a coefficient; Derive puts the #i in the
    # numerator, 3/5 + 4*#i/5.
    assert simp("SIGN(3 + 4*#i)") == "3/5 + 4*#i/5"


# -- 6.9 probability, 6.10 statistical ----------------------------------------

PROBABILITY = [
    ("5!", "120"),
    ("(3/2)!", "3*SQRT(pi)/4"),
    ("GAMMA(z)", "(z - 1)!"),
    ("PERM(z, w)", "z!/(z - w)!"),
    ("COMB(z, w)", "z!/(w!*(z - w)!)"),
    ("AVERAGE(2, 3, 4)", "3"),
    ("AVERAGE([2, 3, 4])", "3"),
    # 6.10 p.118: a matrix is worked row by row, answering a vector.
    ("RMS([[3, 4, 5], [5, 12, 13]])", "[5*SQRT(6)/3, 13*SQRT(6)/3]"),
]


@pytest.mark.parametrize(("text", "expected"), PROBABILITY, ids=str)
def test_probability_and_statistics(text, expected):
    assert simp(text) == expected


def test_a_random_integer_lands_inside_the_range_it_names():
    # 6.9 p.116: RANDOM(6) is one of 0..5, with equal probability.
    draws = {simp("RANDOM(6)") for _ in range(50)}
    assert draws <= {"0", "1", "2", "3", "4", "5"}


def test_the_least_squares_fit_of_the_manuals_parabola():
    # 6.10 p.119: three points and three parameters, so the fit is exact.
    assert simp("FIT([x, a*x^2 + b*x + c], [[-1.5, 0], [0.5, -2], [1.5, -1.5]])") == (
        "x^2/2 - x/2 - 15/8"
    )


def test_the_least_squares_fit_of_the_manuals_plane():
    # 6.10 p.119: four points and three parameters, approximated.
    fit = apx(
        "FIT([x, y, a*x + b*y + c], "
        "[[2.75, -2.3, 2.4], [-3.5, 4.5, 4.2], [5, 3.5, 5.8], [-4, -5, 1.3]])"
    )
    assert fit == "0.153644*x + 0.357749*y + 3.35279"


def test_the_least_squares_fit_that_is_not_linear_in_its_data_variable():
    # 6.10 p.120. The coefficients come out as the manual has them; only the
    # order of the two terms differs, Derive sorting function terms by an
    # internal ordinal that puts ATAN before SIN where this sorts by name.
    fit = apx("FIT([t, q*ATAN(t) + r*SIN(t)], [[-4, -1], [-1, -1.9], [2, 2.25]])")
    assert fit == "1.29723*ATAN(t) + 0.961378*SIN(t)"


# -- 6.11 error and zeta ------------------------------------------------------

ERROR_AND_ZETA = [
    ("ERF(z, w)", "ERF(w) - ERF(z)"),
    ("ERFC(z)", "1 - ERF(z)"),
    ("NORMAL(z, m, s)", "(ERF(SQRT(2)*z/(2*s) - SQRT(2)*m/(2*s)) + 1)/2"),
    ("ZETA(2)", "pi^2/6"),
    # 6.11 p.122: the sum that is the zeta function, for s a number above 1.
    ("SUM(k^(-3), k, 1, inf)", "ZETA(3)"),
]


@pytest.mark.parametrize(("text", "expected"), ERROR_AND_ZETA, ids=str)
def test_the_error_and_zeta_functions(text, expected):
    assert simp(text) == expected


# -- 6.12 financial -----------------------------------------------------------

def test_the_payment_on_the_manuals_car_loan():
    # 6.12 p.124: ten digits of precision, as the section insists on.
    assert apx("PMT(10%/12, 4*12, 15000)", APPROXIMATE_10) == "-380.4387515"


def test_the_balance_on_the_manuals_savings_account():
    # 6.12 p.124: payments at the beginning of each period.
    assert apx("FVAL(5.5%/12, 5*12, -100, -1000, 1)", APPROXIMATE_10) == "8235.356459"


# -- 6.13 number theory, 6.14 decomposition -----------------------------------

NUMBER_THEORY = [
    ("GCD(6, 15)", "3"),
    ("LCM([6, 15, 14])", "210"),
    ("LCM(m, n)", "|m*n|/GCD(n, m)"),
    ("PRIME(7)", "true"),
    ("PRIME(9)", "false"),
    ("NUMBER(SQRT(3^2 + 4^2))", "true"),
    ("NEXT_PRIME(1000)", "1009"),
]


@pytest.mark.parametrize(("text", "expected"), NUMBER_THEORY, ids=str)
def test_the_number_theory_functions(text, expected):
    assert simp(text) == expected


DECOMPOSITION = [
    ("NUMERATOR((3*x^2 - 5*x + 7)/(x - 4))", "3*x^2 - 5*x + 7"),
    ("QUOTIENT(x^4 + 3*x^3 + 5*x + 6, x^2 - 5)", "x^2 + 3*x + 5"),
    ("REMAINDER(x^4 + 3*x^3 + 5*x + 6, x^2 - 5)", "20*x + 31"),
    ("POLY_GCD(x^3 + 3*x^2 + 5*x + 6, x^3 + 2*x - 3)", "x^2 + x + 3"),
    ("LHS(2*x + 3 = 5)", "2*x + 3"),
    ("RHS(2*x + 3 = 5)", "5"),
    # 6.14 p.128: RHS distributes over soLve's vector of solutions.
    ("RHS(SOLVE(x^2 - 5*x + 6 = 0, x))", "[2, 3]"),
]


@pytest.mark.parametrize(("text", "expected"), DECOMPOSITION, ids=str)
def test_the_decomposition_functions(text, expected):
    assert simp(text) == expected


# == Chapter 7: Calculus ======================================================

# -- 7.1 limits ---------------------------------------------------------------

LIMITS = [
    # 7.1 p.169: the difference quotient that is the derivative of x².
    ("LIM(((x + h)^2 - x^2)/((x + h) - x), h, 0)", "2*x"),
    # One-sided and two-sided limits of a discontinuity.
    ("LIM(SIGN(x), x, 0, 1)", "1"),
    ("LIM(SIGN(x), x, 0, -1)", "-1"),
    ("LIM(SIGN(x), x, 0)", "±1"),
    ("LIM(1/x, x, 0, 1)", "inf"),
    ("LIM(1/x, x, 0, -1)", "-inf"),
    ("LIM(1/x, x, 0)", "±inf"),
    # A limit at infinity.
    ("LIM(a*x/(x + 1), x, inf)", "a"),
    # 7.1 p.171: the limit exists where the substitution does not.
    ("LIM(x/SIN(x), x, 0)", "1"),
    # A vector of variables takes a vector of values.
    ("LIM(x^2 + y^2, [x, y], [2, 3])", "13"),
]


@pytest.mark.parametrize(("text", "expected"), LIMITS, ids=str)
def test_the_limits_of_the_manual(text, expected):
    assert simp(text) == expected


def test_substituting_the_limit_point_gives_the_ambiguity_the_limit_resolves():
    # 7.1 p.171: 0/SIN(0) is `?` where the limit is 1.
    assert simp("0/SIN(0)") == "?"


# -- 7.2 differentiation ------------------------------------------------------

DERIVATIVES = [
    ("DIF(SIN(a*x^2), x)", "2*a*x*COS(a*x^2)"),
    ("DIF(LN(COS(x)), x, 2)", "-TAN(x)^2 - 1"),
    ("DIF(DIF((a*x + b*y)^3, x), y)", "6*a*b*(a*x + b*y)"),
    # 7.2 p.173: nothing of the variable in it, so nothing of it left.
    ("DIF(a, x)", "0"),
]


@pytest.mark.parametrize(("text", "expected"), DERIVATIVES, ids=str)
def test_the_derivatives_of_the_manual(text, expected):
    assert simp(text) == expected


async def test_the_derivative_of_a_declared_arbitrary_function(session):
    # 7.2 p.173: with F an arbitrary function, the chain rule stops at F'.
    session.declare_arbitrary("F", ["x"])
    assert (await session.simplify("DIF(F(x)^3, x)")).text == "3*F(x)^2*DIF(F(x), x)"


def test_a_negative_order_of_differentiation_is_an_antiderivative():
    # 7.4 p.177: the inverse of the second derivative on p.173.
    assert simp("DIF(-TAN(x)^2 - 1, x, -2)") == "LN(COS(x))"


# -- 7.3 Taylor polynomials ---------------------------------------------------

TAYLOR = [
    ("TAYLOR(#e^x, x, 0, 5)", "x^5/120 + x^4/24 + x^3/6 + x^2/2 + x + 1"),
    # 7.3 p.175: the degree can fall short of the order asked for.
    ("TAYLOR(c*SIN(x), x, 0, 6)", "c*x^5/120 - c*x^3/6 + c*x"),
    # 7.3 p.176: every derivative vanishes at 0, so every order answers 0.
    ("TAYLOR(EXP(-1/x^2), x, 0, 4)", "0"),
    ("TAYLOR(EXP(-1/x^2), x, 0, 5)", "0"),
    ("TAYLOR(EXP(-1/x^2), x, 0, 6)", "0"),
    # 7.3 p.176: √x has no finite derivative at 0 past the zeroth.
    ("TAYLOR(SQRT(x), x, 0, 0)", "0"),
    ("TAYLOR(SQRT(x), x, 0, 2)", "?"),
]


@pytest.mark.parametrize(("text", "expected"), TAYLOR, ids=str)
def test_the_taylor_polynomials_of_the_manual(text, expected):
    assert simp(text) == expected


# -- 7.4 integration ----------------------------------------------------------

INTEGRALS = [
    ("INT(x/(a^2 + b^2*x^4), x)", "ATAN(b*x^2/a)/(2*a*b)"),
    # 7.4 p.178: an iterated integral whose inner limits carry the outer
    # variable.
    ("INT(INT(x*LN(y), y, x, 2*x), x, 0, 2)", "8*LN(2) - 32/9"),
    # 7.4 pp.180-181: the improper integrals, including the two the manual
    # documents as wrong and as principal values.
    ("INT(1/x^2, x, 1, inf)", "1"),
    ("INT(1/SQRT(x), x, 0, 1)", "2"),
    ("INT(1/x^3, x, -1, 2)", "3/8"),
    ("INT(1/x^2, x, -1, 1)", "-2"),
    ("INT(1/x^2, x, -1, 0) + INT(1/x^2, x, 0, 1)", "inf"),
]


@pytest.mark.parametrize(("text", "expected"), INTEGRALS, ids=str)
def test_the_integrals_of_the_manual(text, expected):
    assert simp(text) == expected


# -- 7.5 sums, 7.6 products ---------------------------------------------------

SUMS_AND_PRODUCTS = [
    # 7.5 p.183: an antidifference, and the definite sum it answers.
    ("SUM(1/2^n, n)", "-2^(1 - n)"),
    ("SUM(1/2^n, n, 1, m)", "1 - 2^(-m)"),
    ("SUM(1/k^2, k, 1, 5)", "5269/3600"),
    ("SUM(k^2, k, [2, 3, 5, 7, 11])", "208"),
    ("SUM([2, 3, 5, 7, 11])", "28"),
    # 7.5 p.182: an empty range sums to zero.
    ("SUM(k^2, k, 3, 2)", "0"),
    # 7.6 p.185: the antiquotient, and the definite product.
    ("PRODUCT(n^2, n)", "(n - 1)!^2"),
    ("PRODUCT(n^2, n, 1, m)", "m!^2"),
    ("PRODUCT(k^2+1, k, 1, 5)", "44200"),
    ("PRODUCT(k^2, k, [2, 3, 5, 7, 11])", "5336100"),
    ("PRODUCT([2, 3, 5, 7, 11])", "2310"),
    # 7.6 p.184: an empty range multiplies to one.
    ("PRODUCT(k^2, k, 3, 2)", "1"),
]


@pytest.mark.parametrize(("text", "expected"), SUMS_AND_PRODUCTS, ids=str)
def test_the_sums_and_products_of_the_manual(text, expected):
    assert simp(text) == expected


# == Chapter 8: Vectors and Matrices ==========================================

GENERATED = [
    ("VECTOR(x^2, x, 5)", "[1, 4, 9, 16, 25]"),
    ("VECTOR(j!, j, 0, 4)", "[1, 1, 2, 6, 24]"),
    ("VECTOR(k^2, k, [2, 3, 5, 7, 11])", "[4, 9, 25, 49, 121]"),
    (
        "VECTOR(VECTOR(j + k, k, 1, 4), j, 1, 3)",
        "[[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]",
    ),
    ("IDENTITY_MATRIX(3)", "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"),
    ("DIMENSION([[a, b, c], [1, 2, 3]])", "2"),
    ("DIMENSION(ELEMENT([[a, b, c], [1, 2, 3]], 1))", "3"),
    ("ABS([a, b, c])", "SQRT(a^2 + b^2 + c^2)"),
]


@pytest.mark.parametrize(("text", "expected"), GENERATED, ids=str)
def test_the_generated_vectors_of_the_manual(text, expected):
    assert simp(text) == expected


def test_a_vector_generated_over_a_step_is_approximated_elementwise():
    # 8.2 p.195: the step is 0.2 and the range stops short of π/4.
    assert apx("VECTOR(SIN(z), z, 0, pi/4, 0.2)") == (
        "[0, 0.198669, 0.389418, 0.564642]"
    )


MANIPULATION = [
    ("[a, b, c] SUB 2", "b"),
    ("[[2, 3, 5], [7, 1, 4]] SUB 2 SUB 3", "4"),
    ("[[2, 3, 5], [7, 1, 4]] SUB [2, 3]", "4"),
    ("ELEMENT([a, b, c], 2)", "b"),
    ("ELEMENT([[2, 3, 5], [7, 1, 4]], 2, 3)", "4"),
    ("APPEND([a, b], [c, d], [e, f])", "[a, b, c, d, e, f]"),
    ("APPEND([[a, b], [c, d], [e, f]])", "[a, b, c, d, e, f]"),
    ("DELETE_ELEMENT([a, b, c], 2)", "[a, c]"),
    ("INSERT_ELEMENT(d, [a, b, c], 2)", "[a, d, b, c]"),
    # 8.3 p.199: the position defaults to the front, and one past the end
    # appends.
    ("INSERT_ELEMENT(d, [a, b, c])", "[d, a, b, c]"),
    ("INSERT_ELEMENT(d, [a, b, c], 4)", "[a, b, c, d]"),
    ("REPLACE_ELEMENT(d, [a, b, c], 2)", "[a, d, c]"),
    ("REPLACE_ELEMENT(d, [a, b, c])", "[d, b, c]"),
    ("REVERSE_VECTOR([a, b, c])", "[c, b, a]"),
    ("SELECT(PRIME(k), k, [3, 5, 7, 9, 11])", "[3, 5, 7, 11]"),
]


@pytest.mark.parametrize(("text", "expected"), MANIPULATION, ids=str)
def test_the_vector_manipulation_functions(text, expected):
    assert simp(text) == expected


def test_selecting_the_primes_below_a_hundred():
    # 8.3 p.200: every prime the range covers.
    primes = [n for n in range(2, 101) if all(n % d for d in range(2, n))]
    assert simp("SELECT(PRIME(k), k, 1, 100)") == (
        "[" + ", ".join(str(p) for p in primes) + "]"
    )


VECTOR_OPERATIONS = [
    ("[2*x, -5, x^2] + [-x, 8, 2*x]", "[x, 3, x^2 + 2*x]"),
    ("3 [2*x, -5, x^2]", "[6*x, -15, 3*x^2]"),
    ("[2, a, 5] . [2*a, 3, -1]", "7*a - 5"),
    ("[[a, b], [c, d]] . [2, 3]", "[2*a + 3*b, 2*c + 3*d]"),
    ("[2, 3] . [[a, b], [c, d]]", "[2*a + 3*c, 2*b + 3*d]"),
    ("[[a, b], [c, d]] . [[2], [3]]", "[[2*a + 3*b], [2*c + 3*d]]"),
    (
        "[[2, 3], [4, 5]] . [[a, b, 1], [c, d, 1]]",
        "[[2*a + 3*c, 2*b + 3*d, 5], [4*a + 5*c, 4*b + 5*d, 9]]",
    ),
    ("CROSS([1, 2, 3], [a, b, c])", "[2*c - 3*b, 3*a - c, b - 2*a]"),
    ("CROSS([1, 2], [a, b])", "b - 2*a"),
]


@pytest.mark.parametrize(("text", "expected"), VECTOR_OPERATIONS, ids=str)
def test_the_vector_operations_of_the_manual(text, expected):
    assert simp(text) == expected


MATRIX_OPERATIONS = [
    ("[[a, b, c], [1, 2, 3]]`", "[[a, 1], [b, 2], [c, 3]]"),
    ("DET([[2, 3], [a, b]])", "2*b - 3*a"),
    ("TRACE([[1, 2, 3], [4, 5, 6], [7, 8, 9]])", "15"),
    (
        "[[2, 3], [a, b]]^2",
        "[[3*a + 4, 3*b + 6], [a*(b + 2), 3*a + b^2]]",
    ),
    ("ROW_REDUCE([[1, 2], [5, 6]], [[3, 4], [7, 8]])", "[[1, 0, -1, -2], [0, 1, 2, 3]]"),
    (
        "ROW_REDUCE([[1, 1], [2, 2]], [[1, 1], [2, 1]])",
        "[[1, 1, 1, 0], [0, 0, 0, 1]]",
    ),
    ("CHARPOLY([[2, 3], [a, b]], z)", "z^2 - z*(b + 2) - 3*a + 2*b"),
    ("EIGENVALUES([[2, 3], [0, b]], z)", "[z = 2, z = b]"),
]


@pytest.mark.parametrize(("text", "expected"), MATRIX_OPERATIONS, ids=str)
def test_the_matrix_operations_of_the_manual(text, expected):
    assert simp(text) == expected


def test_a_matrix_inverted_twice_is_the_matrix_it_started_as():
    # 8.5 p.207: and with no roundoff, the exercise's point.
    assert simp("([[2, 3], [a, b]]^-1)^-1") == "[[2, 3], [a, b]]"


def test_a_singular_matrix_has_no_inverse_to_answer_with():
    # 8.5 p.207: the determinant is zero, so the inverse stays as written.
    assert simp("[[2, 3], [4*a, 6*a]]^-1") == "[[2, 3], [4*a, 6*a]]^-1"


def test_the_manuals_three_by_three_system_solved_by_inversion():
    # 8.5 p.208.
    solved = apx(
        "[[x], [y], [z]] = [[5, 3, -7], [2, -8, 1], [-1, 9, 4]]^-1 . [[4], [6], [5]]"
    )
    assert solved == "[[x = 2.91059], [y = 0.175496], [z = 1.58278]]"


def test_the_characteristic_variable_defaults_to_w():
    # 8.7 p.210: "The default value of the variable used by CHARPOLY is w." The
    # page prints the polynomial only for the call that names its variable, and
    # what the default call answers is the determinant left factored.
    assert simp("CHARPOLY([[2, 3], [a, b]])") == "(2 - w)*(b - w) - 3*a"
    assert simp("EIGENVALUES([[2, 3], [0, b]])") == "[w = 2, w = b]"


NONSCALAR = declared(
    "a :epsilon Nonscalar", "b :epsilon Nonscalar", "c :epsilon Nonscalar"
)

MATRIX_ALGEBRA = [
    ("a``", "a"),
    ("(a^-1)`", "a`^-1"),
    ("(a + b)`", "a` + b`"),
    ("(a . b)`", "b` . a`"),
    ("a . (b + c)", "a . b + a . c"),
    ("(b + c) . a", "b . a + c . a"),
    ("(a . b) . c", "a . (b . c)"),
    ("(a . b)^-1", "b^-1 . a^-1"),
    ("DET(a^-1)", "1/DET(a)"),
]


@pytest.mark.parametrize(("text", "expected"), MATRIX_ALGEBRA, ids=str)
def test_the_algebra_of_declared_nonscalars(text, expected):
    assert simp(text, NONSCALAR) == expected


VECTOR_CALCULUS = [
    ("GRAD(x*y^2*z^3)", "[y^2*z^3, 2*x*y*z^3, 3*x*y^2*z^2]"),
    ("GRAD(c*w + x^2 + y^3 + z^4, [w, x, y, z])", "[c, 2*x, 3*y^2, 4*z^3]"),
    ("DIV([y^2*z^3, 2*x*y*z^3, 3*x*y^2*z^2])", "x*(6*y^2*z + 2*z^3)"),
    ("LAPLACIAN(x*y^2*z^3)", "x*(6*y^2*z + 2*z^3)"),
    ("CURL([y^2, 2*x*z, 0])", "[-2*x, 0, 2*z - 2*y]"),
    # 8.9 p.216: a planar field's curl is answered as the scalar it amounts to.
    ("CURL([v^2, u], [[u, v], [1, 1]])", "1 - 2*v"),
    ("POTENTIAL([y^2*z^3, 2*x*y*z^3, 3*x*y^2*z^2])", "x*y^2*z^3"),
    ("VECTOR_POTENTIAL([x, 0, y - z])", "[-y^2/2, -x*z, 0]"),
]


@pytest.mark.parametrize(("text", "expected"), VECTOR_CALCULUS, ids=str)
def test_the_vector_calculus_of_the_manual(text, expected):
    assert simp(text) == expected


async def test_the_gradient_of_an_arbitrary_function_is_its_partial_derivatives(session):
    # 8.9 p.213.
    session.declare_arbitrary("F", ["x", "y", "z"])
    assert (await session.simplify("GRAD(F(x, y, z))")).text == (
        "[DIF(F(x, y, z), x), DIF(F(x, y, z), y), DIF(F(x, y, z), z)]"
    )


def test_a_potential_is_checked_by_taking_its_gradient_back():
    # 8.10 p.217: a potential is unique only to a constant, so the manual's
    # own check is the one worth making.
    field = "[y^2*z^3, 2*x*y*z^3, 3*x*y^2*z^2]"
    assert simp(f"GRAD(POTENTIAL({field}))") == simp(field)


def test_a_vector_potential_is_checked_by_taking_its_curl_back():
    field = "[x, 0, y - z]"
    assert simp(f"CURL(VECTOR_POTENTIAL({field}))") == simp(field)


# == Chapter 10: Programming ==================================================

async def test_the_iterates_of_the_manuals_fixed_point():
    # 10.1 p.253: iteration stops when a value repeats, and x0 heads the
    # vector. The manual is explicit that this one has to be approximated:
    # iterated exactly it never repeats, so it never stops.
    with abortable() as (session, engine):
        answer = await answered_within(
            session, engine, lambda s: s.approx("ITERATES(#e^(-x/20), x, 1)")
        )
    assert answer.text == (
        "[1, 0.951229, 0.953551, 0.953441, 0.953446, 0.953446, 0.953446]"
    )


async def test_an_iteration_count_bounds_the_iterates(session):
    # 10.1 p.254: n iterations, and n + 1 elements with x0.
    assert (await session.approx("ITERATES(#e^(-x/20), x, 1, 5)")).text == (
        "[1, 0.951229, 0.953551, 0.953441, 0.953446, 0.953446]"
    )


def test_a_negative_count_iterates_the_inverse():
    # 10.1 p.254: which is how MISC.MTH's INVERSE is built.
    assert simp("ITERATES(TAN(x), x, x, -1)") == "[x, ATAN(x)]"


async def test_the_manuals_fibonacci_by_iteration(session):
    # 10.2 p.257: the pair is the hundredth and the hundred and first.
    session.author("FIB(n) := ITERATE([v SUB 2, v SUB 1 + v SUB 2], v, [0, 1], n)")
    assert (await session.simplify("FIB(100)")).text == (
        "[354224848179261915075, 573147844013817084101]"
    )


async def test_the_same_fibonacci_written_with_a_vector_of_variables(session):
    # 10.2 p.258: the same thing without subscripts.
    session.author("FIB(n) := ITERATE([k, j+k], [j, k], [0, 1], n)")
    assert (await session.simplify("FIB(100)")).text == (
        "[354224848179261915075, 573147844013817084101]"
    )


async def test_a_value_assigned_decides_the_manuals_pay_schedule(session):
    # 10.3 p.261.
    session.author("h := 50")
    assert (await session.simplify("IF(h <= 40, 10h, 400 + 15*(h - 40))")).text == "550"


async def test_a_declared_domain_decides_the_manuals_pay_schedule(session):
    # 10.3 p.261: h under thirty, so the first branch is the answer even
    # without a value.
    session.declare_domain("h", "Real", None)
    session.author("h :epsilon Real (-inf, 30)")
    assert (await session.simplify("IF(h <= 40, 10h, 400 + 15*(h - 40))")).text == "10*h"


def test_an_undecidable_condition_keeps_the_whole_conditional():
    # 10.3 p.261.
    text = "IF(h <= 40, 10*h, 400 + 15*(h - 40))"
    assert simp(text) == text


def test_a_one_armed_conditional_counts_what_it_tests():
    # 10.3 p.263: IF with only a test answers 1 or 0, so the sum counts the
    # primes below a hundred.
    assert simp("SUM(IF(PRIME(n)), n, 1, 100)") == "25"


def test_the_conditional_defaults_the_manual_documents():
    # 10.3 p.262: no else clause means `?`.
    assert simp("IF(1 = 2, a)") == "?"
    assert simp("IF(1 = 1, a)") == "a"
    # A test that is not a relation is read as `test = 0`.
    assert simp("IF(0, a, b)") == "a"
    assert simp("IF(5, a, b)") == "b"


def test_the_manuals_search_for_the_first_prime_pair_above_a_hundred_thousand():
    # 10.3 p.264.
    assert simp("ITERATE(IF(PRIME(n) AND PRIME(n+2), n, n+2), n, 100001)") == "100151"


async def test_the_manuals_quadrant_function(session):
    # 10.4 p.266: the origin and both positive axes fall in quadrant one, the
    # negative x-axis in two and the negative y-axis in four.
    session.author("QUADRANT(x, y) := IF(x >= 0, IF(y >= 0, 1, 4), IF(y >= 0, 2, 3))")
    corners = ["(1, 1)", "(-1, 1)", "(-1, -1)", "(1, -1)"]
    assert [(await session.simplify(f"QUADRANT{at}")).text for at in corners] == [
        "1",
        "2",
        "3",
        "4",
    ]
    axes = ["(1, 0)", "(0, 1)", "(-1, 0)", "(0, -1)", "(0, 0)"]
    assert [(await session.simplify(f"QUADRANT{at}")).text for at in axes] == [
        "1",
        "1",
        "2",
        "4",
        "1",
    ]


async def test_the_manuals_triangle_category_is_inclusive(session):
    # 10.5 p.269: OR is and/or, so an equilateral triangle is isosceles.
    session.author('CATEGORY(a, b, c) := IF(a=b OR a=c OR b=c, "Isosceles", "Scalene")')
    assert (await session.simplify("CATEGORY(1, 1, 1)")).text == '"Isosceles"'


def test_and_binds_tighter_than_or(session):
    # 10.5 p.271: AND is to OR as multiplication is to addition.
    loose = session.author("a<=b AND b<=c OR a>=b AND b>=c")
    tight = session.author("(a<=b AND b<=c) OR (a>=b AND b>=c)")
    assert loose.text == tight.text


async def test_the_manuals_recursive_factorial(session):
    # 10.6 p.273: the recursion the chapter opens with.
    session.author("FACT(n) := IF(n = 0, 1, n*FACT(n - 1))")
    assert (await session.simplify("FACT(0)")).text == "1"
    assert (await session.simplify("FACT(5)")).text == "120"
    assert (await session.simplify("FACT(64)")).text == simp("64!")


async def test_the_manuals_accumulating_fibonacci(session):
    # 10.6 p.277: the helper carries what would otherwise be recomputed.
    session.author("FIB_AUX(n, f1, f0) := IF(n=0, f0, FIB_AUX(n - 1, f1 + f0, f1))")
    session.author("FIB_FAST(n) := FIB_AUX(n, 1, 0)")
    assert (await session.simplify("FIB_FAST(100)")).text == "354224848179261915075"


async def test_an_empty_definition_makes_a_name_a_function_and_not_a_variable(session):
    # 10.7 p.279: `G(n) :=` first, or the G inside F's definition is read as a
    # variable and the mutual recursion never closes.
    session.author("m := -3*mu*sigma")
    session.author("s := epsilon - 2*sigma^2")
    session.author("e := -sigma*(mu + 2*epsilon)")
    session.author("G(n) :=")
    session.author(
        "F(n) := IF(n=0, 1, m*DIF(F(n-1), mu) + s*DIF(F(n-1), sigma) "
        "+ e*DIF(F(n-1), epsilon) - mu*G(n-1))"
    )
    session.author(
        "G(n) := IF(n=0, 0, m*DIF(G(n-1), mu) + s*DIF(G(n-1), sigma) "
        "+ e*DIF(G(n-1), epsilon) + F(n-1))"
    )
    assert (await session.simplify("F(4)")).text == "mu*(3*epsilon + mu - 15*sigma^2)"


async def test_the_manuals_mutual_recursion_written_with_accumulators(session):
    # 10.7 p.280: the pair F(4) and G(4), computed once each.
    session.author("m := -3*mu*sigma")
    session.author("s := epsilon - 2*sigma^2")
    session.author("e := -sigma*(mu + 2*epsilon)")
    session.author(
        "FG_AUX(n, f, g) := IF(n=0, [f, g], FG_AUX(n-1, "
        "m*DIF(f,mu) + s*DIF(f,sigma) + e*DIF(f,epsilon) - mu*g, "
        "m*DIF(g,mu) + s*DIF(g,sigma) + e*DIF(g,epsilon) + f))"
    )
    session.author("FG(n) := FG_AUX(n, 1, 0)")
    assert (await session.simplify("FG(4)")).text == (
        "[mu*(3*epsilon + mu - 15*sigma^2), 6*mu*sigma]"
    )


# == Chapter 4: Algebra =======================================================

# -- 4.2 what Simplify does to an algebraic expression ------------------------

ALGEBRA = [
    ("2*y*3", "6*y"),
    ("x^2*y*x", "x^3*y"),
    ("3*x + 7 + x", "4*x + 7"),
    ("x + 0", "x"),
    ("1*x", "x"),
    ("(3*x*y^3)^2", "9*x^2*y^6"),
    ("(x^2 + 2*x*y + y^2)/(x^2 - y^2)", "(x + y)/(x - y)"),
    ("(x + 1)^2 - x^2", "2*x + 1"),
    ("2*x/(x^2 - 1) - 1/(x - 1)", "1/(x + 1)"),
    # 4.2 p.60: the difference of squares, seen without multiplying out the
    # fiftieth power that would bury it.
    ("x^2 - (x + (y+1)^50)*(x - (y+1)^50)", "(y + 1)^100"),
]


@pytest.mark.parametrize(("text", "expected"), ALGEBRA, ids=str)
def test_the_algebra_the_manual_simplifies(text, expected):
    assert simp(text) == expected


async def test_simplifying_a_subexpression_leaves_the_rest_of_the_line_alone(session):
    # 4.2 p.61: with the parenthesized sum highlighted, the answer is a copy of
    # the whole line in which only that part is simplified, and the annotation
    # says so with a prime. The command is asked for `#1` and not for `#1'`:
    # 3.3 p.51 states that a subexpression cannot be named by label at all, so
    # the prime is what the derivation is written with rather than a request.
    session.author("(5*x - 3*x + 1)^7 - x")
    session.select_entry(0)
    session.move_right()
    session.move_down()
    assert session.highlighted_text == "5*x-3*x+1"
    part = await session.simplify("#1")
    assert to_sexpr(part.node) == to_sexpr(parse("(2*x + 1)^7 - x"))
    assert part.annotation == "Simp(#1')"


# -- 4.3 ordering variables ---------------------------------------------------

ORDERING = [
    # 4.3 p.63: the order list is x y z, and exponents do not reorder factors.
    ("z^2*x^3*y^5", "x^3*y^5*z^2"),
    # Off-list names sort alphabetically, after the number and before the list.
    ("z^3*b^2*a*2*x^5*c", "2*a*b^2*c*x^5*z^3"),
    # Terms in the main variable lead, highest degree first.
    ("5*y^3 + 2*x^2 - 3*a*x", "2*x^2 - 3*a*x + 5*y^3"),
    # A leading minus sign is written away where the sum allows it.
    ("-x + y", "y - x"),
]


@pytest.mark.parametrize(("text", "expected"), ORDERING, ids=str)
def test_the_order_the_manual_writes_a_result_in(text, expected):
    assert simp(text) == expected


def test_which_variable_is_main_decides_whether_a_power_is_multiplied_out():
    # 4.3 p.65: with x main the ninth power is expanded; with y main it is not.
    assert simp("(x + 1)^9 + y") == (
        "x^9 + 9*x^8 + 36*x^7 + 84*x^6 + 126*x^5 + 126*x^4 + 84*x^3 + 36*x^2 "
        "+ 9*x + y + 1"
    )
    assert simp("(x + 1)^9 + y", Context(order=("y", "x", "z"))) == "y + (x + 1)^9"


# -- 4.4 expanding polynomials ------------------------------------------------

def test_expanding_the_manuals_product():
    # 4.4 p.66.
    assert expanded("2*x*(x - 3)^2") == "2*x^3 - 12*x^2 + 18*x"


EXPANSIONS = [
    # 4.4 p.67: expanding about some variables and not others.
    (
        ("3*a^2*x^2*y^3 + 7*(b + 1)^2*x^2*y^3", ("x", "y")),
        "x^2*y^3*(3*a^2 + 7*(b + 1)^2)",
    ),
    (
        ("(x + 2*y + 1)^3", ("x",)),
        "x^3 + 3*x^2*(2*y + 1) + 3*x*(2*y + 1)^2 + (2*y + 1)^3",
    ),
    (
        ("(x + 2*y + 1)^3", ("y",)),
        "8*y^3 + 12*y^2*(x + 1) + 6*y*(x + 1)^2 + (x + 1)^3",
    ),
    (
        ("(x + 2*y + 1)^3", ("x", "y")),
        "x^3 + 6*x^2*y + 3*x^2 + 12*x*y^2 + 12*x*y + 3*x + 8*y^3 + 12*y^2 + 6*y + 1",
    ),
    (
        ("(x + 2*y + 1)^3", ("y", "x")),
        "8*y^3 + 12*y^2*x + 12*y^2 + 6*y*x^2 + 12*y*x + 6*y + x^3 + 3*x^2 + 3*x + 1",
    ),
]


@pytest.mark.parametrize(("call", "expected"), EXPANSIONS, ids=str)
def test_expanding_about_the_variables_the_manual_chooses(call, expected):
    text, variables = call
    assert expanded(text, *variables) == expected


# -- 4.5 complex branches -----------------------------------------------------

def test_a_complex_result_is_written_in_rectangular_form():
    # 4.5 p.71.
    assert simp("(1 + #i)^3") == "-2 + 2*#i"
    assert apx("(#i + 7)/(3*#i + 4)") == "1.24 - 0.68*#i"


def test_the_branch_the_manual_starts_on_is_the_principal_one():
    # 4.5 p.72: "Principal is the initial setting".
    assert Context().branch is Branch.PRINCIPAL
    assert simp("9^(1/2)") == "3"
    assert simp("(-8)^(1/3)") == "1 + SQRT(3)*#i"


def test_the_real_branch_takes_the_real_root_where_there_is_one():
    # 4.5 p.73.
    assert simp("(-8)^(1/3)", Context(branch=Branch.REAL)) == "-2"


def test_any_branch_drops_the_caution_the_others_keep():
    # 4.5 p.73: √(x²) is x, sign unknown or not.
    assert simp("(x^2)^(1/2)", Context(branch=Branch.ANY)) == "x"


# -- 4.6 factoring ------------------------------------------------------------

def test_factoring_the_manuals_number():
    # 4.6 p.75.
    assert factored("1234567890") == "2*3^2*5*3607*3803"


FACTORINGS = [
    # 4.6 pp.76-78: the same expression at rising amounts, and the manual's
    # other worked amounts.
    (("3 - 1/(x + 2)", Amount.TRIVIAL), "(3*x + 5)/(x + 2)"),
    (("2*x^3 - 12*x^2 + 18*x", Amount.TRIVIAL), "2*x*(x^2 - 6*x + 9)"),
    (("2*x^3 - 12*x^2 + 18*x", Amount.SQUAREFREE), "2*x*(x - 3)^2"),
    (
        ("x^4 + 2*x^3 - 3*x^2 - 8*x - 4", Amount.SQUAREFREE),
        "(x + 1)^2*(x^2 - 4)",
    ),
    (
        ("x^4 + 2*x^3 - 3*x^2 - 8*x - 4", Amount.RATIONAL),
        "(x + 2)*(x - 2)*(x + 1)^2",
    ),
    (("4*x^2 - 9", Amount.RATIONAL), "(2*x + 3)*(2*x - 3)"),
    (("x^2 - 2", Amount.RADICAL), "(x + SQRT(2))*(x - SQRT(2))"),
    (("x^2 + 2", Amount.COMPLEX), "(x + SQRT(2)*#i)*(x - SQRT(2)*#i)"),
]


@pytest.mark.parametrize(("call", "expected"), FACTORINGS, ids=str)
def test_the_factorings_of_the_manual(call, expected):
    text, amount = call
    assert factored(text, amount) == expected


def test_an_approximate_radical_factoring_is_written_in_digits():
    # 4.6 p.77: the same factoring in approximate mode.
    assert factored("x^2 - 2", Amount.RADICAL, context=APPROXIMATE_6) == (
        "(x + 1.41421)*(x - 1.41421)"
    )
    assert factored("x^2 + 2", Amount.COMPLEX, context=APPROXIMATE_6) == (
        "(x + 1.41421*#i)*(x - 1.41421*#i)"
    )


MULTIVARIATE_FACTORINGS = [
    (
        ("x^2*y^2 - x^2 - y^4 + y^2", ("x", "y")),
        "(x + y)*(x - y)*(y + 1)*(y - 1)",
    ),
    (("x^2*y^2 - x^2 - y^4 + y^2", ("x",)), "(x + y)*(x - y)*(y^2 - 1)"),
    (
        ("x^2*y^4 - 2*x^2*y^2 + x^2 + 2*x*y^2 - 2*x + 1", ("x",)),
        "(x*(y^2 - 1) + 1)^2",
    ),
]


@pytest.mark.parametrize(("call", "expected"), MULTIVARIATE_FACTORINGS, ids=str)
def test_factoring_about_the_variables_the_manual_chooses(call, expected):
    text, variables = call
    assert factored(text, Amount.RATIONAL, *variables) == expected


def test_the_quadratic_formula_as_a_radical_factoring():
    # 4.6 p.79.
    assert factored("a*x^2 + b*x + c", Amount.RADICAL, "x") == (
        "a*(x + (SQRT(b^2 - 4*a*c) + b)/(2*a))*(x - (SQRT(b^2 - 4*a*c) - b)/(2*a))"
    )


# -- 4.7 partial fractions ----------------------------------------------------

def test_the_partial_fraction_expansion_of_the_manual():
    # 4.7 p.81: each term super-proper in x, and the numerator that has no x
    # in it is left whole.
    text = "(25*x^4 + 81*a*x^2 + 324*a*x + 324*a)/(x^3 + x^2 - 8*x - 12)"
    assert expanded(text, "x") == (
        "81*(a + 1)/(x - 3) - 80/(x + 2)^2 + 144/(x + 2) + 25*x - 25"
    )


# -- 4.8 substituting values --------------------------------------------------

def test_substitutions_for_variables_are_made_all_at_once(session):
    # 4.8 p.85: which is what lets a pair of variables trade places. What the
    # page states is the simultaneity - `x + 2*y` interchanged is `y + 2*x` and
    # not the `x + 2*x` that writing one in and then the other would give. The
    # tree is what says so: a substituted line is held in the author spelling,
    # the manual prints its own typeset one, and the two are one expression.
    session.author("x + 2*y")
    substituted = session.substitute("#1", {"x": "y", "y": "x"})
    assert to_sexpr(substituted.node) == to_sexpr(parse("y + 2*x"))


def test_substituting_a_subexpression_takes_only_what_matches_it(session):
    # 4.8 p.86: k for the subexpression t³ in t⁶ + t³ is `t⁶ + k`, and not the
    # `k² + k` that the same replacement made of t would reach, because a
    # subexpression goes in only where it stands written.
    session.author("t^6 + t^3")
    session.select_entry(0)
    session.move_right()
    session.move_right()
    assert session.highlighted_text == "t^3"
    assert to_sexpr(session.substitute_part("#1", "k").node) == to_sexpr(
        parse("t^6 + k")
    )


def test_substituting_a_root_for_a_variable_reaches_what_a_subexpression_cannot(
    session,
):
    # 4.8 p.86: `k² + k` is what the page says to reach by writing k^(1/3) in
    # for t and simplifying. It prints no intermediate line, and there is no
    # spelling of one to print: `^` folds right, so the `k^(1/3)^6` a flat
    # writing would give reads back as `k^((1/3)^6)`.
    session.author("t^6 + t^3")
    substituted = session.substitute("#1", {"t": "k^(1/3)"})
    assert simp(substituted.text) == "k^2 + k"


def test_substituting_before_simplifying_can_lose_the_answer_simplifying_keeps():
    # 4.8 p.88: the ratio has a removable singularity at -1. Substituted into
    # the unsimplified form it is 0/0; simplified first, it is 0.
    assert simp("((-1)^2 + 2*(-1) + 1)/((-1)^2 - 1)") == "?"
    assert simp("(x^2 + 2*x + 1)/(x^2 - 1)") == "(x + 1)/(x - 1)"
    assert simp("(-1 + 1)/(-1 - 1)") == "0"


# -- 4.9 infinite values ------------------------------------------------------

INFINITIES = [
    ("TAN(pi/2)", "±inf"),
    ("1/0", "±inf"),
    ("SIGN(0)", "±1"),
    ("inf - inf", "?"),
    ("inf/inf", "?"),
    ("0/0", "?"),
]


@pytest.mark.parametrize(("text", "expected"), INFINITIES, ids=str)
def test_the_infinities_and_indeterminates_of_the_manual(text, expected):
    assert simp(text) == expected


# -- 4.10 domains -------------------------------------------------------------

def test_a_root_squared_is_what_it_was_whatever_it_was():
    # 4.10 p.91.
    assert simp("(SQRT(z))^2", COMPLEX_Z) == "z"


def test_a_square_rooted_is_an_absolute_value_where_the_variable_is_real():
    # 4.10 p.91: real is what an undeclared variable is.
    assert simp("SQRT(x^2)") == "|x|"


def test_an_integer_declaration_settles_a_sine_of_a_multiple_of_pi():
    # 4.10 p.91.
    assert simp("SIN(n*pi)", declared("n :epsilon Integer")) == "0"


QUOTIENTS_OF_ONE = [
    ("x/x", "1"),
    ("x^0", "1"),
    ("0^0", "1"),
]


@pytest.mark.parametrize(("text", "expected"), QUOTIENTS_OF_ONE, ids=str)
def test_the_quotients_the_manual_answers_one_for(text, expected):
    # 4.10 p.94: all three, whatever the domain of x takes in.
    assert simp(text) == expected


def test_a_one_point_interval_is_an_assignment():
    # 4.10 p.95: `7 ≤ y ≤ 7` gives y the value 7.
    assert simp("y + 1", declared("y :epsilon Real [7, 7]")) == "8"


# -- 4.11 assigning values ----------------------------------------------------

async def test_an_assigned_variable_is_replaced_where_it_stands(session):
    # 4.11 p.98.
    session.author("area_circle := pi*r^2")
    answered = await session.simplify("2*area_circle + 2*pi*h*r")
    assert answered.text == "2*pi*h*r + 2*pi*r^2"


async def test_an_assignment_is_followed_as_far_as_it_goes(session):
    # 4.11 p.99: radius is s², s is 5, so radius is 25.
    session.author("radius := s^2")
    session.author("s := 5")
    assert (await session.simplify("radius")).text == "25"


# -- 4.12 defining functions --------------------------------------------------

async def test_a_function_definition_takes_the_arguments_it_is_given(session):
    # 4.12 p.104: and leaves the parameters it is not given standing.
    session.author("ACCELERATION(f, m) := f/m")
    assert (await session.simplify("ACCELERATION(6, 2)")).text == "3"
    assert (await session.simplify("ACCELERATION(6)")).text == "6/m"


# -- 4.13 equations and relations ---------------------------------------------

def test_an_equation_is_simplified_a_side_at_a_time():
    # 4.13 p.107.
    assert simp("x + 2*x = c + x*x") == "3*x = x^2 + c"


EQUATION_ARITHMETIC = [
    # 4.13 pp.108-110: the manual completes a square by doing the same thing
    # to both sides of the equation, a line at a time.
    ("(x^2 + 5*x + 6 = 0) - 6", "x^2 + 5*x = -6"),
    ("4*(x^2 + 5*x + 6 = 0)", "4*x^2 + 20*x + 24 = 0"),
    # 4.13 p.111: a function applied to an equation is applied to both sides.
    ("EXP(LN(x) = 5)", "x = #e^5"),
]


@pytest.mark.parametrize(("text", "expected"), EQUATION_ARITHMETIC, ids=str)
def test_arithmetic_on_a_whole_equation(text, expected):
    assert simp(text) == expected


def test_an_equation_factors_as_a_whole():
    # 4.13 p.110.
    assert factored("x^2 - 5*x + 6 = 0") == "(x - 2)*(x - 3) = 0"


# -- 4.14 solving -------------------------------------------------------------

def test_the_solve_function_answers_a_vector_of_solutions():
    # 4.14 p.113: the function's answer is bracketed where the command's is not.
    assert simp("SOLVE(-2*x + 3*y <= 7, x)") == "[x >= (3*y - 7)/2]"


def test_the_manuals_quadratic_is_solved_as_two_equations():
    # 4.14 p.114.
    assert solutions("x^2 - 5*x + 6 = 0", "x") == ["x = 2", "x = 3"]


def test_an_equation_that_cannot_be_solved_is_answered_as_an_implicit_one():
    # 4.14 p.115: the variable is inextricable, so what comes back says so.
    assert solutions("3^x = x^2", "x") == ["3^x - x^2 = 0"]


def test_an_impossible_equation_has_no_solutions_to_answer_with():
    # 4.14 p.117.
    assert solutions("x = x + 1", "x") == []


def test_a_degenerate_equation_is_solved_by_an_arbitrary_value():
    # 4.14 p.117: every x solves it, and @1 is how that is written.
    assert solutions("2*x = x + x", "x") == ["x = @1"]


# -- 4.15 systems -------------------------------------------------------------

def test_the_manuals_linear_system():
    # 4.15 p.119: linear in x and y, whatever a does.
    assert solutions("[2*a^2*x + 3*y = 7, x - 5*y = 0]", "x", "y") == [
        "[x = 35/(10*a^2 + 3), y = 7/(10*a^2 + 3)]"
    ]


def test_an_inconsistent_system_has_no_solutions_to_answer_with():
    # 4.15 p.121.
    assert solutions("[x + 3*y = 1, 2*x + 6*y = 3]", "x", "y") == []


def test_the_pair_of_the_manuals_nonlinear_system_that_solve_can_take():
    # 4.15 p.124: the first two equations, solved for z and w.
    system = "[x^2 - z + w + 1 = 0, y^2 - z - w + 3 = 0]"
    assert solutions(system, "z", "w") == [
        "[z = (x^2 + y^2 + 4)/2, w = -(x^2 - y^2 - 2)/2]"
    ]


def test_the_quadratic_the_manuals_elimination_arrives_at():
    # 4.15 p.125.
    assert solutions("5*y^2 - 5*y - 10 = 0", "y") == ["y = -1", "y = 2"]


# -- 4.16 boolean operators ---------------------------------------------------

BOOLEAN = [
    ("NOT true", "false"),
    ("NOT false", "true"),
    ("true AND false", "false"),
    ("true OR false", "true"),
    ("true XOR true", "false"),
    ("true XOR false", "true"),
    ("true IMP false", "false"),
    ("false IMP true", "true"),
    # 4.16 p.128: the forms the unknown cases take. The page fences the XOR
    # form for the reader; the program prints it unfenced, relying on AND
    # binding tighter than OR, and that is what is asserted here.
    ("p XOR q", "NOT p AND q OR p AND NOT q"),
    ("p IMP q", "NOT p OR q"),
    ("NOT NOT p", "p"),
    ("p AND (p AND q OR r)", "p AND (q OR r)"),
    ("p OR NOT p", "true"),
    # 4.16 p.130: relations joined by them are simplified where they can be.
    ("6 >= -2*x AND 3*x /= -9", "x > -3"),
    # 4.16 p.132: integers make them bitwise, on a half-infinite two's
    # complement.
    ("3 OR 5", "7"),
    ("NOT 5", "-6"),
]


@pytest.mark.parametrize(("text", "expected"), BOOLEAN, ids=str)
def test_the_boolean_operators(text, expected):
    assert simp(text) == expected


def test_not_binds_tighter_than_and_and_and_tighter_than_or(session):
    # 4.16 p.129: NOT, AND, OR, XOR, IMP, in falling order.
    assert session.author("NOT p OR q AND r").text == session.author(
        "(NOT p) OR (q AND r)"
    ).text


def test_the_manuals_truth_table():
    # 4.16 p.131: the header names the columns, and the rows run TT, TF, FT, FF.
    assert simp("TRUTH_TABLE(p, q, p AND q, p OR q, p XOR q, p IMP q)") == (
        "[[p, q, p AND q, p OR q, p XOR q, p IMP q], "
        "[true, true, true, true, false, true], "
        "[true, false, false, true, true, false], "
        "[false, true, false, true, true, true], "
        "[false, false, false, false, false, true]]"
    )


# == The appendices ===========================================================

def test_the_bare_letters_are_variables_and_not_the_constants():
    # Q&A pp.286-287: `e` and `i` are the user's to spend; the constants are
    # written `#e` and `#i`.
    assert simp("LN(e)") == "LN(e)"
    assert simp("i^2") == "i^2"
    assert simp("LN(#e)") == "1"
    assert simp("(#i)^2") == "-1"


def test_a_logarithm_difference_waits_for_a_domain_that_justifies_it():
    # Q&A p.287: not simplified while x may be negative, simplified once it
    # may not be.
    text = "LN(x^2 - x) - LN(x)"
    assert simp(text) != "LN(x - 1)"
    assert simp(text, declared("x :epsilon Real [0, inf)")) == "LN(x - 1)"


def test_square_brackets_make_a_vector_rather_than_a_group():
    # Q&A p.288: `3 [x+y]` is three times a one-element vector.
    assert simp("3 [x+y]") == "[3*(x + y)]"


def test_integrating_a_derivative_need_not_give_back_what_was_differentiated():
    # Q&A p.288: the antiderivative differs from x/(x + 1) by the constant 1.
    assert simp("INT(DIF(x/(x + 1), x), x)") == "-1/(x + 1)"


# == Chapter 9: Utility Files =================================================

# The chapter is one long walk through the MTH files Derive shipped with, so
# these load the definitions it asks about and put the manual's own questions
# to them. Only what these cases reach is here, plus whatever it calls in turn.

UTILITY_SOURCE = {
    "INT_APPS.MTH": """\
LAPLACE(y,t,s):=INT(#e^(-s*t)*y,t,0,inf)
""",
    "MISC.MTH": """\
INVERSE(u,x):=ITERATE(u,x,x,-1)
INT_SUBST(y,x,u):=LIM(INT(LIM(y,x,INVERSE(u,x))*DIF(INVERSE(u,x),x),x),x,u)
""",
    "NUMBER.MTH": """\
POWER_MOD(n,d,m):=MOD(n^d,m)
NTH_PRIME(n):=ITERATE(NEXT_PRIME(k),k,1,n)
CATALAN(n):=COMB(2*n,n)/(n+1)
BERNOULLI(n):=IF(n=0,1,IF(n=1,-1/2,-n*ZETA(1-n)))
BERNOULLI_POLY(n,x):=IF(n=1,x-1/2,n*LIM(SUM(k^(n-1),k,1,x_-1),x_,x)+BERNOULLI(n))
EULER_POLY(n,x):=2/(n+1)*(BERNOULLI_POLY(n+1,x)-2^(n+1)*BERNOULLI_POLY(n+1,x/2))
EULER(n):=IF(MOD(n,2)=1,0,2^n*EULER_POLY(n,1/2))
LUCAS(n):=(ITERATE(IF((n AND d_)=0,[a_^2-2*(-1)^c_,a_*b_-(-1)^c_,2*c_,d_/2],[a_*b_-(-1)^c_,b_^2+2*(-1)^c_,2*c_+1,d_/2]),[a_,b_,c_,d_],[2,1,0,2^FLOOR(LOG(n,2))],FLOOR(LOG(n,2))+1)) SUB 1
FIBONACCI(n):=IF(n>=0 AND FLOOR(n)=n,(2*LUCAS(n+1)-LUCAS(n))/5,1/SQRT(5)*(((1+SQRT(5))/2)^n-((SQRT(5)-1)/2)^n*COS(pi*n)))
NEXT_MERSENNE_DEGREE(n):=IF(PRIME(2^(n+1)-1),n+1,NEXT_MERSENNE_DEGREE(n+1))
MERSENNE(n):=2^ITERATE(NEXT_MERSENNE_DEGREE(j),j,1,n)-1
MERSENNE_DEGREE(n):=ELEMENT([2,3,5,7,13,17,19,31,61,89,107,127,521,607,1279,2203,2281,3217,4253,4423,9689,9941,11213,19937,21701,23209,44497,86243,110503,132049,216091,756839,859433],n)
MERSENNE(n):=2^MERSENNE_DEGREE(n)-1
CONTINUED_FRACTION(u,n):=FLOOR(ITERATES(1/MOD(x_),x_,u,n))
CONVERGENT_AUX(v,k,a):=IF(k=0,a,CONVERGENT_AUX(v,k-1,v SUB k+1/a))
CONVERGENT(v):=CONVERGENT_AUX(v,DIMENSION(v),inf)
PARTS_AUX(n,m):=IF(n<2*m,1,1+SUM(PARTS_AUX(n-k_,k_),k_,m,FLOOR(n,2)))
PARTS(n):=IF(n<1,0,PARTS_AUX(n,1))
DISTINCT_PARTS_AUX(n,m):=IF(n<2*m,1,1+SUM(DISTINCT_PARTS_AUX(n-k_,k_+1),k_,m,FLOOR(n,2)))
DISTINCT_PARTS(n):=IF(n<1,0,DISTINCT_PARTS_AUX(n-1,1))
PARTS(n):=IF(n<2,1,FLOOR(APPROX(SUM(1/pi*SQRT(k_/2)*SUM(IF(GCD(h_,k_)=1,COS(pi*(SUM((i_/k_-1/2)*(MOD(i_*h_/k_)-1/2),i_,1,k_-1)-2*n*h_/k_)),0),h_,1,k_)*(-2*SQRT(6)*#e^(-pi*SQRT(24*n-1)/(6*k_))*(#e^(pi*SQRT(24*n-1)/(3*k_))*(6*k_-pi*SQRT(24*n-1))-6*k_-pi*SQRT(24*n-1))/(k_*(24*n-1)^(3/2))),k_,1,SQRT(n)/LOG(n,11)),LOG(1/(4*n*SQRT(3))*EXP(pi*SQRT(2*n/3)),10)+5)+0.5))
DISTINCT_PARTS(n):=SUM(IF(8*n-16*i_+1=FLOOR(SQRT(8*n-16*i_+1))^2,PARTS(i_),0),i_,0,n/2)
""",
    "ODE1.MTH": """\
SEPARABLE(p,q,x,y,x0,y0):=INT(1/q,y,y0,y)=INT(p,x,x0,x)
HOMOGENEOUS(r,x,y,x0,y0):=IF(DIF(LIM(r,y,x*y),x),LIM(SEPARABLE(1/x,LIM(r,x,1)-y,x,y,x0,y0/x0),y,y/x),"inapplicable","inapplicable")
INTF_AUX(mu,p,q,x,y,x0,y0):=INT(LIM(mu*q,x,x0),y,y0,y)+INT(mu*p,x,x0,x)=0
INTF_HLP(p,q,x,y,x0,y0):= IF(DIF((DIF(p,y)-DIF(q,x))/q,y),    INTF_AUX(#e^INT((DIF(p,y)-DIF(q,x))/q,x),p,q,x,y,x0,y0),    "inapplicable",    "inapplicable")
INTEGRATING_FACTOR(p,q,x,y,x0,y0):= IF(DIF((DIF(q,x)-DIF(p,y))/p,x),    INTF_AUX(#e^INT((DIF(q,x)-DIF(p,y))/p,y),p,q,x,y,x0,y0),    INTF_HLP(p,q,x,y,x0,y0),    INTF_HLP(p,q,x,y,x0,y0))
GEN_HOM_AUX(k,r,x,y,x0,y0):=IF(DIF(k,x)=0 AND DIF(k,y)=0,LIM(SEPARABLE(1/x,y*(k+LIM(r*x/y,y,y/x^k)),x,y,x0,y0*x0^k),y,y*x^k),"inapplicable","inapplicable")
GEN_HOM(r,x,y,x0,y0):=GEN_HOM_AUX(x*DIF(x*r/y,x)/(y*DIF(x*r/y,y)),r,x,y,x0,y0)
CLAIRAUT(p,q,x,y,v,c):=[LIM(p-q,v,c)=0,x*LIM(DIF(LIM(p,y,x*v-y),y),y,x*v-y)-DIF(q,v)=0]
SEP_HLP(p,q,x,y,x0,y0):= IF(DIF(p,x)=0 and DIF(q,y)=0,    SEPARABLE(1/q,-p,x,y,x0,y0),    "inapplicable",    "inapplicable")
SEP(p,q,x,y,x0,y0):= IF(DIF(p,y)=0 and DIF(q,x)=0,    SEPARABLE(p,-1/q,x,y,x0,y0),    SEP_HLP(p,q,x,y,x0,y0),    SEP_HLP(p,q,x,y,x0,y0))
DSOLVE1(p,q,x,y,x0,y0,a_):= IF("inapplicable"=a_:=INTEGRATING_FACTOR(p,q,x,y,x0,y0),    IF("inapplicable"=a_:=HOMOGENEOUS(-p/q,x,y,x0,y0),       IF("inapplicable"=a_:=SEP(p,q,x,y,x0,y0),	  GEN_HOM(-p/q,x,y,x0,y0),a_,a_),a_,a_),a_,a_)
""",
    "ODE_APPR.MTH": """\
TAYLOR_ODE1(r,x,y,x0,y0,n,v_):=ELEMENT(ITERATE([v_ SUB 1+1,DIF(v_ SUB 2,x)+r*DIF(v_ SUB 2,y),v_ SUB 3+(x-x0)^v_ SUB 1*LIM(v_ SUB 2,[x,y],[x0,y0])/v_ SUB 1!],v_,[1,r,y0],n),3)
PICARD(r,p,x,y,x0,y0):=y0+INT(LIM(r,y,p),x,x0,x)
TAYLOR_ODE2_AUX(x,y,v,x0,y0,v0,n,aux):=y0+v0*(x-x0)+SUM(LIM(aux SUB (k_-1),[v,y,x],[v0,y0,x0])/k_!*(x-x0)^k_,k_,2,n)
TAYLOR_ODE2(r,x,y,v,x0,y0,v0,n):=TAYLOR_ODE2_AUX(x,y,v,x0,y0,v0,n,ITERATES(DIF(g_,v)*r+DIF(g_,y)*v+DIF(g_,x),g_,r,n-2))
EULER(r,x,y,x0,y0,h,n):=ITERATES(v_+h*[1,LIM(r,[x,y],v_)],v_,[x0,y0],n)
""",
    "RECUREQN.MTH": """\
LIN1_DIFFERENCE(p,q,x,x0,y0):=PRODUCT(p,x,x0,x-1)*(y0+SUM(q/PRODUCT(p,x,x0,x),x,x0,x-1))
CLAIRAUT_DIF(p,q,d,x,y,c):=LIM(p-q,d,c)=0
""",
    "SOLVE.MTH": """\
NEWTON_AUX(a,x,x0,n):=ITERATES(xk-ELEMENT(ROW_REDUCE(LIM(a,x,xk))`,DIMENSION(a)+1),xk,x0,n)
NEWTONS(u,x,x0,n):=NEWTON_AUX(APPEND(GRAD(u,x),[u])`,x,x0,n)
TAYLOR_SOLVE_AUX2(x,y,x0,y0,n,aux):=y0+SUM(LIM(ELEMENT(aux,k_),[y,x],[y0,x0])/k_!*(x-x0)^k_,k_,1,n)
TAYLOR_SOLVE_AUX(x,y,x0,y0,n,v):=TAYLOR_SOLVE_AUX2(x,y,x0,y0,n,ITERATES(DIF(g_,y)*v+DIF(g_,x),g_,v,n-1))
TAYLOR_SOLVE(u,x,y,x0,y0,n):=TAYLOR_SOLVE_AUX(x,y,x0,y0,n,-DIF(u,x)/DIF(u,y))
LAGRANGE_EXPANSION(u,x,y,x0,y0,n):=x0+SUM((y-y0)^k_/k_!*LIM(DIF(((x-x0)/(u-y0))^k_,x,k_-1),x,x0),k_,1,n)
TAYLOR_INVERSE(u,x,y,x0,n):=IF(LIM(DIF(u,x),x,x0)=0,"Taylor expansion of inverse does not exist",LAGRANGE_EXPANSION(TAYLOR(u,x,x0,n),x,y,x0,LIM(u,x,x0),n),LAGRANGE_EXPANSION(TAYLOR(u,x,x0,n),x,y,x0,LIM(u,x,x0),n))
""",
    "VECTOR.MTH": """\
i_:=[1,0,0]
j_:=[0,1,0]
k_:=[0,0,1]
cylindrical:=[[r,theta,z],[1,r,1]]
spherical:=[[r,theta,phi],[1,r*SIN(phi),r]]
OUTER(v,w):=VECTOR([v SUB n_],n_,DIMENSION(v)) . [w]
ADJOIN_ELEMENT(e,v):=APPEND([e],v)
APPEND_COLUMNS(a,b):=APPEND(a`,b`)`
MINOR(a,i,j):=DELETE_ELEMENT(DELETE_ELEMENT(a,i)`,j)`
SWAP_ELEMENTS(v,i,j):=VECTOR(IF(m_=i,v SUB j,IF(m_=j,v SUB i,v SUB m_)),m_,DIMENSION(v))
SCALE_ELEMENT(v,i,s):=VECTOR(IF(m_=i,s*v SUB i,v SUB m_),m_,DIMENSION(v))
COFACTOR(a,i,j):=COS(pi*(i+j))*DET(MINOR(a,i,j))
ADJOINT(a):=VECTOR(VECTOR(COFACTOR(a,m_,n_),m_,DIMENSION(a)),n_,DIMENSION(a))
ZERO_VECTOR(v,i):=IF(i>DIMENSION(v),0,IF(v SUB i=0,ZERO_VECTOR(v,i+1),1,1))
NONZERO_ROWS(m):=SUM(ZERO_VECTOR(m SUB n_,1),n_,1,DIMENSION(m))
RANK(m):=NONZERO_ROWS(ROW_REDUCE(m))
JACOBIAN(u,alpha):=VECTOR(GRAD(u SUB m_,alpha),m_,DIMENSION(u))
COVARIANT_METRIC_TENSOR(a):=a` . a
GEOMETRY_MATRIX(alpha,g):=[alpha,VECTOR(SQRT(g SUB m_ SUB m_),m_,DIMENSION(g))]
""",
}


def written(tmp_path, name):
    """A utility file on disk, so `load_utility` is given what it expects."""
    path = tmp_path / name
    path.write_text(UTILITY_SOURCE[name], encoding="utf-8")
    return path


@pytest.fixture
def loaded(tmp_path):
    """A session with the named utility files loaded, as Transfer Load Utility
    would leave it."""

    def load(*names, word_input=False):
        session = Session()
        if word_input:
            session.author("InputMode := Word")
        for name in names:
            session.load_utility(written(tmp_path, name))
        return session

    return load


# -- 9.1 SOLVE.MTH ------------------------------------------------------------

async def test_the_series_solution_of_an_equation_that_has_no_closed_one(loaded):
    # 9.1 p.229.
    session = loaded("SOLVE.MTH")
    answered = await session.simplify("TAYLOR_SOLVE(SIN(y) + y + x*#e^x, x, y, 0, 0, 3)")
    assert answered.text == (
        "-25*x^3/96 - x^2/2 - x/2"
    )


async def test_the_series_inverse_of_a_function_that_has_no_closed_one(loaded):
    # 9.1 p.230, where the printed call is one argument short: it passes four
    # to the `TAYLOR_INVERSE(u, x, y, x0, n)` the same page declares. Read as
    # written it binds `y := 0` and `x0 := 3` and leaves `n` free, and the
    # original answers it with a sum over that free `n` about the point 3 -
    # which is what this engine answers it with too. The line the manual prints
    # is what the five-argument call gives, and the answer being written in `y`
    # is what says so.
    session = loaded("SOLVE.MTH")
    assert (await session.simplify("TAYLOR_INVERSE(x*#e^x, x, y, 0, 3)")).text == (
        "3*y^3/2 - y^2 + y"
    )


async def test_newtons_finds_the_manuals_nonlinear_solution(loaded):
    # 9.1 p.228: seven rows, and the last of them is the solution.
    session = loaded("SOLVE.MTH")
    rows = (await session.approx(
        "NEWTONS([#e^(x*y) - x - 2*y, ATAN(x*y) - 2*x - y], [x, y], [-1, -1], 6)"
    )).text
    assert rows.endswith("[-0.424667, 0.599887]]")


# -- 9.2 VECTOR.MTH -----------------------------------------------------------

VECTOR_MTH = [
    ("i_", "[1, 0, 0]"),
    ("j_", "[0, 1, 0]"),
    ("k_", "[0, 0, 1]"),
    ("OUTER([a, b], [2, 3, 4])", "[[2*a, 3*a, 4*a], [2*b, 3*b, 4*b]]"),
    ("ADJOIN_ELEMENT(a, [b, c])", "[a, b, c]"),
    (
        "APPEND_COLUMNS([[a, b], [c, d]], [[e, f], [g, h]])",
        "[[a, b, e, f], [c, d, g, h]]",
    ),
    ("SWAP_ELEMENTS([a, b, c], 1, 2)", "[b, a, c]"),
    ("SCALE_ELEMENT([a, b, c], 2, 3)", "[a, 3*b, c]"),
    ("ADJOINT([[2, x], [1, 3]])", "[[3, -x], [-1, 2]]"),
    ("RANK([[2, 3, 5], [4, 6, 10], [1, 2, 3]])", "2"),
    ("JACOBIAN([(w^2 - v^2)/2, w*v], [w, v])", "[[w, -v], [v, w]]"),
    (
        # Manual p.238 misprints this as `w^2 + v^2`; the DOS releases both
        # answer `v^2 + w^2`.
        "COVARIANT_METRIC_TENSOR([[w, -v], [v, w]])",
        "[[v^2 + w^2, 0], [0, v^2 + w^2]]",
    ),
    (
        "GEOMETRY_MATRIX([u, v], [[u^2 + v^2, 0], [0, u^2 + v^2]])",
        "[[u, v], [SQRT(u^2 + v^2), SQRT(u^2 + v^2)]]",
    ),
]


@pytest.mark.parametrize(("text", "expected"), VECTOR_MTH, ids=str)
async def test_the_worked_examples_of_the_vector_utility_file(text, expected, loaded):
    session = loaded("VECTOR.MTH", word_input=True)
    assert (await session.simplify(text)).text == expected


async def test_the_coordinate_geometries_the_vector_file_assigns(loaded):
    # 9.2 p.236.
    session = loaded("VECTOR.MTH", word_input=True)
    assert (await session.simplify("cylindrical")).text == "[[r, theta, z], [1, r, 1]]"
    assert (await session.simplify("spherical")).text == (
        "[[r, theta, phi], [1, r*SIN(phi), r]]"
    )


# -- 9.5 INT_APPS.MTH ---------------------------------------------------------

async def test_the_laplace_transform_of_the_manuals_square(loaded):
    # 9.5 p.240: s must be declared positive or the integral does not converge.
    session = loaded("INT_APPS.MTH")
    session.author("s :epsilon Real (0, inf)")
    assert (await session.simplify("LAPLACE(t^2, t, s)")).text == "2/s^3"


# -- 9.6 ODE1.MTH -------------------------------------------------------------

async def test_the_first_order_equation_the_manual_solves(loaded):
    # 9.6 p.247: and the check that the answer meets the initial condition.
    session = loaded("ODE1.MTH")
    assert (await session.simplify("DSOLVE1(2*x*y, 1 + x^2, x, y, 0, 1)")).text == (
        "x^2*y + y - 1 = 0"
    )
    assert (await session.simplify("0^2*1 + 1 - 1 = 0")).text == "0 = 0"


async def test_the_clairaut_equation_the_manual_solves(loaded):
    # 9.6 p.251: a general solution in c, and an equation to find v from.
    session = loaded("ODE1.MTH")
    assert (await session.simplify("CLAIRAUT((x*v - y)^2, v^2 + 1)")).text == (
        "[c^2*x^2 - 2*c*x*y + y^2 - c^2 - 1 = 0, 2*(v*x^2 - x*y - v) = 0]"
    )


# -- 9.8 ODE_APPR.MTH ---------------------------------------------------------

async def test_the_series_solution_of_the_manuals_first_order_equation(loaded):
    # 9.8 p.255.
    session = loaded("ODE_APPR.MTH")
    assert (await session.simplify("TAYLOR_ODE1(#e^(x*y), x, y, 0, 1, 4)")).text == (
        "5*x^4/12 + x^3/2 + x^2/2 + x + 1"
    )


async def test_the_picard_iterate_of_the_manuals_equation(loaded):
    # 9.8 p.256: expanded about x, which is the form the manual prints.
    session = loaded("ODE_APPR.MTH")
    entry = session.author("PICARD(y^2 + SQRT(x), x + 1, x, y, 0, 1)")
    assert (await session.expand(f"#{entry.number}", variables=("x",))).text == (
        "x^3/3 + x^2 + 2*x^(3/2)/3 + x + 1"
    )


async def test_the_series_solution_of_the_manuals_second_order_equation(loaded):
    # 9.8 p.259.
    session = loaded("ODE_APPR.MTH")
    assert (await session.simplify(
        "TAYLOR_ODE2(2*x*v + x^2*y + 3*x, x, y, v, 0, 0, 1, 5)"
    )).text == "3*x^5/10 + 5*x^3/6 + x"


async def test_the_euler_steps_the_manual_takes(loaded):
    # 9.8 p.260: five coordinate pairs, the first of them the initial point.
    session = loaded("ODE_APPR.MTH")
    answered = await session.approx("EULER(26/(3 + (x + y)^2), x, y, 1, -2, 1/4, 4)")
    assert answered.text == (
        "[[1, -2], [1.25, -0.375], [1.5, 1.35114], [1.75, 1.93520], [2, 2.32722]]"
    )


# -- 9.9 RECUREQN.MTH ---------------------------------------------------------

async def test_the_difference_equation_the_manual_solves(loaded):
    # 9.9 p.264, and the check of its initial condition.
    session = loaded("RECUREQN.MTH")
    assert (await session.simplify("LIN1_DIFFERENCE(m, (m+1)!, m, 1, 2)")).text == (
        "(m^2/2 + m/2 + 1)*(m - 1)!"
    )
    assert (await session.simplify("(1^2/2 + 1/2 + 1)*(1 - 1)!")).text == "2"


async def test_the_clairaut_difference_equation_the_manual_solves(loaded):
    # 9.9 p.266.
    session = loaded("RECUREQN.MTH")
    assert (await session.simplify("CLAIRAUT_DIF((y - x*d)^2, d^2)")).text == (
        "c^2*x^2 - 2*c*x*y + y^2 - c^2 = 0"
    )


# -- 9.20 NUMBER.MTH ----------------------------------------------------------

NUMBER_MTH = [
    ("POWER_MOD(2, 500, 100)", "76"),
    ("NTH_PRIME(3)", "5"),
    ("CATALAN(10)", "16796"),
    ("BERNOULLI(10)", "5/66"),
    (
        "BERNOULLI_POLY(10, x)",
        "(66*x^10 - 330*x^9 + 495*x^8 - 462*x^6 + 330*x^4 - 99*x^2 + 5)/66",
    ),
    ("EULER(10)", "-50521"),
    ("EULER_POLY(10, x)", "x*(x^9 - 5*x^8 + 30*x^6 - 126*x^4 + 255*x^2 - 155)"),
    ("FIBONACCI(10)", "55"),
    ("MERSENNE(10)", "618970019642690137449562111"),
    ("CONVERGENT([2, 1, 2, 1, 1, 4, 1, 1, 6])", "1264/465"),
    ("PARTS(4)", "5"),
    ("DISTINCT_PARTS(4)", "2"),
]


@pytest.mark.parametrize(("text", "expected"), NUMBER_MTH, ids=str)
async def test_the_worked_examples_of_the_number_theory_file(text, expected, loaded):
    session = loaded("NUMBER.MTH")
    assert (await session.simplify(text)).text == expected


async def test_the_continued_fraction_of_e(loaded):
    # 9.20 p.283: nine partial quotients of ê.
    session = loaded("NUMBER.MTH")
    assert (await session.approx("CONTINUED_FRACTION(#e, 8)")).text == (
        "[2, 1, 2, 1, 1, 4, 1, 1, 6]"
    )


# -- 9.21 MISC.MTH ------------------------------------------------------------

async def test_the_integral_the_manual_does_by_substitution(tmp_path):
    # 9.21 p.285.
    with abortable() as (session, engine):
        session.load_utility(written(tmp_path, "MISC.MTH"))
        answer = await answered_within(
            session,
            engine,
            lambda s: s.simplify("INT_SUBST(t*SIN(t^2), t, t^2)"),
        )
    assert answer.text == "-COS(t^2)/2"


async def test_the_inverse_of_the_manuals_function(loaded):
    # 9.21 p.285.
    session = loaded("MISC.MTH")
    assert (await session.simplify("INVERSE(SIN(x/b), x)")).text == "b*ASIN(x)"
