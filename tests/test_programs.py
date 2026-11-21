"""Programs: definitions with defaults, calls through activations, assignments, the
constructs `PROG`, `LOOP`, `exit`, `RETURN` and `ASSIGN`, the built-ins that evaluate an
expression per value of a variable, settings in programs, and the step limit.

Each case is a list of lines run in a fresh session, in order: a line starting with `>`
is entered, any other is a pair of an input and the text Simplify answers with, or
`(error: message)` for a command that fails with that message and appends nothing.
"""

import time

import pytest
import sympy as sp
from sympy.core.parameters import global_parameters

from rederive.engine import budget, evaluate
from rederive.engine.client import EngineError
from rederive.model.session import Session
from rederive.syntax import write_expression


def entered(text):
    return f"> {text}"


async def run(*lines, step_limit=-1):
    """Run `lines` in a fresh session (see the module docstring); return the session.

    `step_limit` is the session's, by default that of a session computing in-process.
    """
    session = Session(step_limit=step_limit)
    for line in lines:
        if isinstance(line, str):
            assert line.startswith("> ")
            session.author(line[2:])
            continue
        text, expected = line
        if expected.startswith("(error: "):
            count = len(session.entries)
            with pytest.raises(EngineError, match=expected[len("(error: ") : -1]):
                await session.simplify(text)
            assert len(session.entries) == count
            continue
        assert (await session.simplify(text)).text == expected, text
    return session


# -- definitions --------------------------------------------------------------


async def test_an_omitted_argument_takes_its_default():
    await run(
        "> SUMSQ(x, y := 0, z := 0) := x^2 + y^2 + z^2",
        ("[SUMSQ(5), SUMSQ(5, 3), SUMSQ(5, 3, 2)]", "[25, 34, 38]"),
        "> SUMSQ2(x, y, z) := x^2 + y^2 + z^2",
        ("[SUMSQ2(5), SUMSQ2(5, 3), SUMSQ2(5, 3, 2)]", "[y^2 + z^2 + 25, z^2 + 34, 38]"),
    )


async def test_a_prime_differentiates_the_function_where_a_loop_names_its_parameter():
    """`f'(t)` is the derivative of f at t, also where a loop gives t its values: the
    body is differentiated in a variable of its own, then taken at each value."""
    await run(
        "> f(t) := t^3",
        ("TABLE(f'(t), t, [7, 21])", "[[7, 147], [21, 1323]]"),
        ("VECTOR(f'(t), t, 1, 3)", "[3, 12, 27]"),
        ("f'(r^2)", "3*r^4"),
        "> F(x, a) := a*x^2",
        ("F'(a, a)", "2*a^2"),
    )


async def test_a_generator_with_nothing_to_range_over_stands_as_written():
    await run(
        "> e := x^2 + b",
        ("VECTOR(SUBST(e, b, s))", "VECTOR(SUBST(e, b, s))"),
        ("SELECT(x > 1)", "SELECT(x > 1)"),
        ("VECTOR(SUBST(e, b, s), s, [1, 2])", "[x^2 + 1, x^2 + 2]"),
    )


async def test_a_default_sees_the_parameters_before_it():
    await run("> P1(x, d := 2*x) := x + d", ("[P1(3), P1(3, 1)]", "[9, 4]"))


async def test_a_call_of_a_function_without_a_body_is_completed_from_its_parameters():
    await run(
        "> FOO(x, y := 3, z) :=",
        ("FOO(5)", "FOO(5, 3, z)"),
        ("FOO(5, 4)", "FOO(5, 4, z)"),
        ("FOO(1, 2, 3)", "FOO(1, 2, 3)"),
    )


async def test_an_omitted_argument_without_a_default_is_its_name():
    await run(
        "> HYPOTENUSE(a, b) := SQRT(a^2 + b^2)",
        ("HYPOTENUSE(3)", "SQRT(b^2 + 9)"),
        "> addrc(a, b, n) := IF(b = 0, n, addrc(a, b - 1, n + 1))",
        ("addrc(3, 2)", "n + 2"),
        ("addrc(3, 2, 3)", "5"),
    )


async def test_mutual_recursion_declares_the_second_function_first():
    await run(
        "> G1(n) :=",
        "> F1(n) := IF(n = 0, 0, 1 + G1(n - 1))",
        "> G1(n) := IF(n = 0, 0, 1 + F1(n - 1))",
        ("F1(5)", "5"),
    )


async def test_a_variadic_function_takes_its_arguments_as_a_vector():
    await run(
        "> FOO v := SUM(v)",
        ("FOO(2, 3, 5, 7, 11)", "28"),
        "> foo2 v := [AVERAGE(v), VARIANCE(v)]",
        ("foo2(1, 4, 8)", "[13/3, 37/3]"),
        "> Block commands := commands SUB DIM(commands)",
        ("Block(1 + 1, 2 + 2, 3 + 3)", "6"),
        "> CNT v := DIM(v)",
        ("[CNT(), CNT([1, 2]), CNT(1, 2)]", "[0, 1, 2]"),
    )


async def test_extra_arguments_are_ignored():
    await run("> EX2(a, b) := a - b", ("EX2(5, 3, 7)", "2"))


async def test_a_definition_holding_a_program_simplifies_to_itself():
    await run(
        "> od(n) := LOOP(IF(MOD(n, 2) = 1, RETURN n), n := n/2)",
        ("#1", "od(n) := LOOP(IF(MOD(n, 2) = 1, RETURN(n)), n := n/2)"),
    )


# -- calls ----------------------------------------------------------------------


async def test_a_local_is_assigned_in_its_activation_only():
    await run(
        "> LOC(a, t) := PROG(t := a + 1, t^2)",
        "> t := 99",
        ("LOC(2)", "9"),
        ("t", "99"),
    )


async def test_a_name_that_is_no_parameter_is_global():
    await run(
        "> k := 5",
        "> prime_power?(n) := PROG(k := 1, t := 2, LOOP(IF(t > n, RETURN false), "
        "IF(PRIME?(n^(1/k)), exit), k := k + 1, t := 2*t))",
        ("prime_power?(9)", "true"),
        ("k", "2"),
        "> count(l) := PROG(k := 0, LOOP(IF(l = [], RETURN k), "
        "IF(prime_power?(FIRST(l)), k :+ 1), l := REST(l)))",
        ("count([1, ..., 10])", "4"),
    )


async def test_locals_are_restored_after_the_call():
    await run(
        "> k_ := 5",
        "> PRIME_POWER?(n, k_ := 1, t_ := 2) := LOOP(IF(t_ > n, RETURN false), "
        "IF(PRIME?(n^(1/k_)), exit), k_ :+ 1, t_ :* 2)",
        "> COUNT(l, k_ := 0) := LOOP(IF(l = [], RETURN k_), "
        "IF(PRIME_POWER?(FIRST(l)), k_ :+ 1), l := REST(l))",
        ("COUNT([1, ..., 10])", "7"),
        ("k_", "5"),
    )


async def test_the_variables_of_a_global_value_are_not_the_parameters():
    await run(
        "> b := 200 - pi*r",
        "> a(r) := 2*r*b",
        ("a(100/pi)", "200*(200 - pi*r)/pi"),
    )


async def test_a_variable_added_to_an_infinite_value_is_among_the_variables():
    # The time-value-of-money program in Derive User Group Newsletter 55 finds the one
    # unknown among its arguments this way, the number of periods `inf` for a
    # perpetuity.
    await run(
        "> InputMode := Word",
        "> n := inf",
        ("VARIABLES(n + a)", "[a]"),
        "> UNKNOWN(n, im, pv, pmt, fv) := VARIABLES(n + im + pv + pmt + fv)",
        ("UNKNOWN(inf, rate, 10, -1, 0)", "[rate]"),
        ("UNKNOWN(12, rate, 10, -1, 0)", "[rate]"),
    )


async def test_a_parameter_without_an_argument_reads_the_global():
    await run(
        "> default1 := 1",
        "> AREA0(x, x1, x2, default1) := INT(default1, x, x1, x2)",
        ("AREA0(x, 0, 3)", "3"),
        ("AREA0(x, 0, 3, x)", "9/2"),
    )


# -- recursion ----------------------------------------------------------------


async def test_recursion_runs_thousands_of_levels_deep():
    await run(
        "> succ(n) := n + 1",
        "> addr(a, b) := IF(b = 0, a, succ(addr(a, b - 1)))",
        ("addr(121, 345)", "466"),
        "> CNT(n) := IF(n = 0, 0, 1 + CNT(n - 1))",
        ("CNT(5000)", "5000"),
    )


async def test_a_return_ends_only_its_own_activation():
    await run(
        "> sudan(n, x, y) := IF(n = 0, RETURN(x + y), IF(y = 0, RETURN(x), "
        "RETURN(sudan(n - 1, sudan(n, x, y - 1), sudan(n, x, n - 1) + y))))",
        ("sudan(1, 1, 3)", "10"),
        ("sudan(1, 0, 3)", "6"),
        "> dreia(n) := PROG(IF(n = 1, RETURN 19), n := dreia(n - 1), "
        "IF(MOD(n, 2) = 0, n/2, 3*n + 1))",
        ("VECTOR(dreia(n), n, 1, 10)", "[19, 58, 29, 88, 44, 22, 11, 34, 17, 52]"),
    )


async def test_runaway_recursion_leaves_the_pending_call():
    await run("> RUN(n) := RUN(n + 1)", ("RUN(1)", "RUN(10001)"))


# -- assignments ----------------------------------------------------------------


async def test_a_simplified_assignment_stores_the_simplified_value():
    await run(
        ("x := 2 + 3", "x := 5"),
        ("x", "5"),
        "> r := s^2",
        "> s := 5",
        ("r", "25"),
    )


async def test_an_assignment_to_a_global_is_the_value_of_a_call():
    await run(
        "> test(n, x, y) := PROG(x := n^2, y := n^3, res := x + y)",
        ("test(5)", "res := 150"),
        ("res", "150"),
        ("test(5) + 1", "151"),
        "> test2(n, x, y, res) := PROG(x := n^2, y := n^3, res := x + y)",
        ("test2(5)", "150"),
    )


async def test_a_program_at_top_level_assigns_globals_when_simplified():
    await run(
        "> [aa := 1, bb := 5]",
        "> PROG(xx := 3, yy := 4)",
        ("xx + aa", "xx + 1"),
        ("PROG(xx := 3, yy := 4)", "yy := 4"),
        ("xx + aa", "4"),
        ("PROG(A := VECTOR([k, k^2], k, 1, 3), [])", "[]"),
        ("A", "[[1, 1], [2, 4], [3, 9]]"),
    )


async def test_a_definition_in_a_program_is_made():
    await run(
        ("PROG(xt2(t) := t, yt2(t) := t/2 + 9/2)", "yt2(t) := t/2 + 9/2"),
        ("xt2(7)", "7"),
    )


async def test_an_assignment_in_an_element_of_a_vector_keeps_its_form():
    await run(
        "> setup0(dummy) := PROG(n0 := 24961, true)",
        ("[setup0(), e0 := 2]", "[true, e0 := 2]"),
        ("[n0, e0]", "[24961, 2]"),
        ("[w1 := 2, w1^2, w1 := w1 + 1]", "[w1 := 2, 4, w1 := 3]"),
    )


async def test_an_assignment_as_an_operand_is_its_value():
    await run(
        "> NEWA(u) := (a9 := u)",
        ("0*NEWA(w + 7)", "0"),
        ("a9", "w + 7"),
        ("1 + PROG(u1 := 2, u1^2)", "5"),
    )


async def test_globals_persist_from_one_command_to_the_next():
    await run(
        "> [x := 0, subtotal := 0]",
        "> stepwise(n) := PROG(LOOP(x :+ 1, subtotal := subtotal + x^2, "
        'IF(x > n, RETURN "done"), RETURN [x, subtotal]))',
        ("stepwise(5)", "[1, 1]"),
        ("stepwise(5)", "[2, 5]"),
        ("stepwise(5)", "[3, 14]"),
        ("stepwise(5)", "[4, 30]"),
        ("stepwise(5)", "[5, 55]"),
        ("stepwise(5)", '"done"'),
    )


async def test_clearing_a_variable_takes_its_value_away():
    session = await run(
        "> [p1 := 124, pa := 207/2, p2 :=]",
        ("p2", "p2"),
        "> clr(dummy) := [p1 :=, 7]",
        ("clr()", "[p1 :=, 7]"),
        ("p1", "p1"),
    )
    assert "p1" not in session.assignments


async def test_a_string_is_a_variable_of_its_own():
    await run(
        ('ASSIGN(APPEND("a", 1), 25)', '"a1" := 25'),
        ('"a1"', "25"),
    )


async def test_assign_assigns_the_variable_its_target_evaluates_to():
    await run(
        '> SAVEV(s, prefix := "s", i_ := 0) := '
        'PROG(ASSIGN(APPEND(prefix, i_, "_"), s), i_)',
        ("SAVEV([7, 8])", "0"),
        ('"s0_"', "[7, 8]"),
        "> tgt := 5",
        ("ASSIGN(tgt, 1)", "ASSIGN(5, 1)"),
        ("ASSIGN(2 + 3, 1)", "ASSIGN(5, 1)"),
        "> nm := q",
        ("ASSIGN(nm, 7)", "q := 7"),
        ("q", "7"),
    )


async def test_a_vector_of_assignments_is_evaluated_in_order_and_x_assigned_x_clears():
    await run(
        "> InputMode := Word",
        ("[x := y, y^2 - x, x := x]", "[x := y, y^2 - y, x :=]"),
        ("x", "x"),
        # An element written out is selected before anything is evaluated.
        ("[x := w^2, y^2 - x, x := x] SUB 2", "y^2 - x"),
        ("x", "x"),
    )


async def test_an_assignment_to_a_formal_holding_a_variable_assigns_the_variable():
    """Outside a program a formal supplied with a variable's name stands for it as a
    target too, and an argument mentioning the variable sees its value: the
    substitution idiom of the newsletter."""
    await run(
        "> InputMode := Word",
        "> PART(v_, n_) := v_ SUB n_",
        "> SIMPLE_SUBST(expr_, var_, svar_) := "
        "PART([var_ := svar_, expr_, var_ := var_], 2)",
        ("SIMPLE_SUBST(a*x + b, x, z^3)", "a*z^3 + b"),
        ("x", "x"),
        "> S(v_, a_, b_) := [1 + v_ SUB 1, SIMPLE_SUBST(v_ SUB 2, a_ SUB (v_ SUB 1), "
        "b_ SUB (v_ SUB 1))]",
        "> S2(expr_, a_, b_) := ITERATE(S(w_, a_, b_), w_, [1, expr_], DIMENSION(a_))",
        ("S2(x^2 - y*z, [x, y, z], [w^3, a^3, 5])", "[4, w^6 - 5*a^3]"),
        # In a program a formal is a local like any other.
        "> T(u, v) := PROG(v := 3, u)",
        ("T(x + 1, x)", "x + 1"),
    )


async def test_a_truth_table_is_over_its_variables_whatever_their_values():
    await run(
        "> InputMode := Word",
        '> a1 := "11110000"',
        (
            "TRUTH_TABLE(a1, a2, a1 OR a2)",
            "[[a1, a2, a1 OR a2], [true, true, true], [true, false, true], [false, true, "
            "true], [false, false, false]]",
        ),
    )


# -- updates and element assignment -----------------------------------------------


async def test_an_update_operates_on_the_variable():
    await run(
        "> CNTUP(n, k_ := 0, s_ := 1) := "
        "LOOP(IF(k_ = n, RETURN [k_, s_]), k_ :+ 1, s_ :* 2)",
        ("CNTUP(10)", "[10, 1024]"),
        "> HALVE(s_) := PROG(LOOP(s_ :/ 2, IF(MOD(s_, 2) = 1, exit)), s_)",
        ("HALVE(96)", "3"),
        "> g6(x) := PROG(x :* 1 + 0*#i, [x, x*x])",
        ("[g6(5), g6(0)]", "[[5, 25], [0, 0]]"),
        "> zz := 10",
        ("zz :- 3", "zz := 7"),
        ("zz :* 2 + 1", "zz := 21"),
        "> vv := [1, 2, 3]",
        ("vv :+ [10, 10, 10]", "vv := [11, 12, 13]"),
        ("vv SUB 2 :+ 5", "vv SUB 2 := 17"),
        ("vv", "[11, 17, 13]"),
        # A local vector updated by another local takes no share in that local.
        "> upd(y, p) := PROG(y := [10, 20], p := 2, y :* p, [y, p])",
        ("upd()", "[[20, 40], 2]"),
    )


async def test_a_string_parameter_is_a_local_variable_that_can_be_declared():
    """A string among the formal parameters is a variable like any other, here the label
    of the result; declared a vector, an equation of it with a vector stands."""
    await run(
        ('"R" = [1, 2]', "false"),
        '> lbl(a, "R") := IF(DIM(a) = 1, "R" = a SUB 1, PROG("R" :epsilon Vector,'
        ' "R" = a))',
        ("lbl([5])", '"R" = 5'),
        ("lbl([1, 2])", '"R" = [1, 2]'),
    )


async def test_an_element_assignment_replaces_an_element_when_simplified():
    await run(
        "> A := [[1, 2, 3, 4], [5, 6, 7, 9]]",
        "> A SUB 2 SUB 3 := A SUB 2 SUB 3 + 1",
        ("A", "[[1, 2, 3, 4], [5, 6, 7, 9]]"),
        ("A SUB 2 SUB 3 := A SUB 2 SUB 3 + 1", "A SUB 2 SUB 3 := 8"),
        ("A", "[[1, 2, 3, 4], [5, 6, 8, 9]]"),
    )


async def test_an_element_assignment_takes_a_vector_of_indices():
    await run(
        "> a := [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]",
        ("a SUB 1 := [x, y, z, u]", "a SUB 1 := [x, y, z, u]"),
        (
            "a SUB [1, 3] := [[r, r, r, r], [s, s, s, s]]",
            "a SUB [1, 3] := [[r, r, r, r], [s, s, s, s]]",
        ),
        ("a", "[[r, r, r, r], [5, 6, 7, 8], [s, s, s, s], [13, 14, 15, 16]]"),
        ("a SUB SUB 1 := [0, 0, 0, 0]", "a SUB SUB 1 := [0, 0, 0, 0]"),
        ("a SUB SUB 1", "[r, 5, s, 13]"),
        ("a SUB 9 := 0", "a SUB 9 := 0"),
        ("a SUB (-1) SUB (-1) := 0", "a SUB (-1) SUB (-1) := 0"),
        ("a SUB 4", "[13, 14, 15, 16]"),
        "> SWAPR(h, m, i, j) := "
        "PROG(j := [m, m + 1], i := REVERSE(j), h SUB i := h SUB j, h)",
        ("SWAPR([[1, 1], [2, 2], [3, 3]], 2)", "[[1, 1], [3, 3], [2, 2]]"),
        "> SLICE(h, i, j, k) := PROG(k := [1, ..., j], "
        "h SUB i SUB k := h SUB i SUB k - 10*h SUB 1 SUB k, h)",
        ("SLICE([[1, 2, 3], [4, 5, 6]], 2, 2)", "[[1, 2, 3], [-6, -15, 6]]"),
    )


async def test_an_element_assignment_at_a_position_not_counted_from_1_changes_nothing():
    # Reading counts a negative index from the end, but Derive assigns only at a
    # position counted from 1.
    await run(
        ("PROG(v := [1, 2, 3], v SUB (-1) := 9, v)", "[1, 2, 3]"),
        ("v SUB (-1)", "3"),
        ("PROG(v := [1, 2, 3], v SUB 0 := 9, v)", "[1, 2, 3]"),
        ("PROG(v := [1, 2, 3], v SUB 4 := 9, v)", "[1, 2, 3]"),
        ("PROG(v := [1, 2, 3], v SUB (-1) :+ 9, v)", "[1, 2, 3]"),
        ("PROG(v := [1, 2, 3], v SUB [1, -1] := [8, 9], v)", "[1, 2, 3]"),
        "> m := [[1, 2], [3, 4]]",
        ("PROG(m SUB 2 SUB (-1) := 9, m)", "[[1, 2], [3, 4]]"),
        ("PROG(m SUB (-1) SUB 2 := 9, m)", "[[1, 2], [3, 4]]"),
        ("PROG(m SUB 2 SUB [1, -1] := [8, 9], m)", "[[1, 2], [3, 4]]"),
        ("PROG(m SUB 2 SUB [1, 2] := [8, 9], m)", "[[1, 2], [8, 9]]"),
        "> w := [1, 2, 3]",
        ("w SUB (-1) := 9", "w SUB (-1) := 9"),
        ("w", "[1, 2, 3]"),
        "> invstep(m, sp, t := 1) := PROG(i := sp, m SUB i := m SUB i/m SUB i SUB i, "
        "LOOP(i :+ t, m SUB i := m SUB i - m SUB sp*m SUB i SUB sp, "
        "IF(t = 1 AND i > DIM(m), PROG(inv := m, RETURN(inv))), "
        "IF(t = -1 AND i < 0, PROG(inv := m, RETURN(inv)))))",
        (
            "invstep([[1, 3/2, 1/2, 1/2, 0, 0], [0, 1, -3, -1, 2, 0], "
            "[0, 0, 1, 3/5, -4/5, -1/5]], 3, -1)",
            "[[1, 3/2, 0, 1/5, 2/5, 1/10], [0, 1, 0, 4/5, -2/5, -3/5], "
            "[0, 0, 1, 3/5, -4/5, -1/5]]",
        ),
    )


async def test_two_locals_share_a_vector_one_was_given_by_name():
    await run(
        "> ALIAS(v, w) := PROG(w := v, w SUB 1 := 99, [v, w])",
        ("ALIAS([1, 2])", "[[99, 2], [99, 2]]"),
        "> ALIAS2(v, w) := PROG(w := v + 0*v, w SUB 1 := 99, [v, w])",
        ("ALIAS2([1, 2])", "[[1, 2], [99, 2]]"),
        "> ALIAS3(v, w) := PROG(w := v, w := [7, 7], w SUB 1 := 99, [v, w])",
        ("ALIAS3([1, 2])", "[[1, 2], [99, 7]]"),
        "> setfirst(v) := PROG(v SUB 1 := 0, true)",
        "> callsf(w) := PROG(w := [7, 8], setfirst(w), w)",
        ("callsf()", "[0, 8]"),
        "> callsf2(w) := PROG(w := [7, 8], setfirst(w + 0*w), w)",
        ("callsf2()", "[7, 8]"),
        "> zz1 := [5, 6]",
        ("setfirst(zz1)", "true"),
        ("zz1", "[5, 6]"),
        "> glb(v) := PROG(gv := v, v SUB 1 := 0, [v, gv])",
        ("glb([1, 2])", "[[0, 2], [1, 2]]"),
    )


async def test_a_program_assigns_elements_of_a_global():
    await run(
        "> OMEGA := IDENTITY_MATRIX(2)",
        "> init2(n, i_ := 1, j_) := PROG(LOOP(j_ := 1, LOOP(OMEGA SUB i_ SUB j_ := "
        "i_ + j_, j_ :+ 1, IF(j_ > n, exit)), i_ :+ 1, IF(i_ > n, exit)))",
        ("init2(2)", "true"),
        ("OMEGA", "[[2, 3], [3, 4]]"),
    )


# -- PROG, LOOP, exit, RETURN -------------------------------------------------------


async def test_prog_runs_its_statements_in_order():
    await run(
        "> Rev1(v, w) := PROG(w := [], LOOP(IF(v = [], RETURN w), "
        "w := ADJOIN(FIRST(v), w), v := REST(v)))",
        ("Rev1([a, b, c])", "[c, b, a]"),
        "> data := [1, 2]",
        (
            "PROG(s := 0, d := data, LOOP(IF(DIM(d) = 0, RETURN(SQRT(s))), "
            "s :+ FIRST(d)^2, d := REST(d)))",
            "SQRT(5)",
        ),
        "> fq(n) := PROG(q := VECTOR(k^2, k, n, 0, -1), RETURN q)",
        ("fq(3)", "[9, 4, 1, 0]"),
        "> fq2(n) := PROG(q := VECTOR(k^2, k, n, 0, -1))",
        ("fq2(3)", "q := [9, 4, 1, 0]"),
        "> JD(m, y) := PROG(IF(m > 2, m := m - 3, PROG(m := m + 9, y := y - 1)), [m, y])",
        ("[JD(1, 2002), JD(5, 2002)]", "[[10, 2001], [2, 2002]]"),
    )


async def test_a_loop_left_by_exit_has_the_last_value():
    await run(
        "> odd_part(n) := LOOP(IF(MOD(n, 2) = 1, RETURN(n)), n := n/2)",
        ("odd_part(96)", "3"),
        "> PRIME_POWER?(n, k_ := 1, t_ := 2) := LOOP(IF(t_ > n, RETURN false), "
        "IF(PRIME?(n^(1/k_)), exit), k_ :+ 1, t_ :* 2)",
        (
            "[PRIME_POWER?(9), PRIME_POWER?(10), PRIME_POWER?(8), PRIME_POWER?(1)]",
            "[true, false, true, false]",
        ),
        "> XV(a, n) := LOOP(IF(a = [], RETURN n), "
        "IF(FIRST(a) < 0, [a := REST(a), exit]), n :+ 1, a := REST(a))",
        ("XV([1, 2, -3, 4], 0)", "[4]"),
        "> XE(k) := LOOP(IF(k < 3, k :+ 1, exit))",
        ("XE(0)", "false"),
        "> fr(x, f, i) := PROG(f := 1/x SUB 1, i := 2, LOOP(IF(i = DIM(x), exit), "
        "f := 1/(f + x SUB i), i :+ 1), x SUB i + f)",
        ("fr([5, 1, 3])", "23/6"),
    )


async def test_exit_ends_the_innermost_loop_or_prog_of_its_activation():
    await run(
        # A `PROG` inside a `LOOP` is what its `exit` ends; the loop goes on, as
        # LaplaceTransforms' loop over the terms of a sum relies on.
        "> XP(k, r) := PROG(r := 0, LOOP(IF(k = 0, exit), "
        "PROG(r :+ 1, exit, r :+ 100), k :- 1), r + 10)",
        ("XP(3)", "13"),
        "> XQ(show) := PROG(IF(NOT show, exit), 7)",
        ("[XQ(true), XQ(false)]", "[7, true]"),
        "> inner(k) := IF(k > 2, exit, k)",
        "> outer(k, s) := "
        "PROG(s := 0, LOOP(IF(k > 5, exit), s :+ 1, inner(k), k :+ 1), s)",
        ("outer(0)", "6"),
        "> perm4(i := 1, j := 1, k := 1, l := 1, liste := []) := "
        "PROG(LOOP(IF(i > 4, exit), LOOP(IF(j > 4, exit), IF(i /= j, "
        "LOOP(IF(k > 4, exit), IF(i /= k AND j /= k, LOOP(IF(l > 4, exit), "
        "IF(i /= l AND j /= l AND k /= l, liste := APPEND(liste, "
        "[[i, j, k, l]])), l :+ 1)), l := 1, k :+ 1)), k := 1, j :+ 1), j := 1, i :+ 1), "
        "RETURN(liste))",
        ("[DIM(perm4()), FIRST(perm4())]", "[24, [1, 2, 3, 4]]"),
    )


async def test_exit_outside_every_construct_is_a_name():
    await run(("[1, exit]", "[1, exit]"))


async def test_return_ends_the_activation_from_any_depth():
    await run(
        "> agm0(x, y) := PROG(IF(NOT NUMBER?(x) OR NOT NUMBER?(y), "
        'RETURN "invalid arguments"), (x + y)/2)',
        ("[agm0(a, 7), agm0(1, 7)]", '["invalid arguments", 4]'),
        "> R3(n) := PROG(LOOP(LOOP(IF(n > 3, RETURN n), n :+ 1)), 99)",
        ("R3(0)", "4"),
        "> R4(v, w) := PROG(w := VECTOR(IF(k = 3, RETURN \"found\", k), k, v), w)",
        ("[R4([1, 2]), R4([1, 3, 5])]", '[[1, 2], "found"]'),
        "> R5(x) := x + IF(x > 0, RETURN 100, 1)",
        ("[R5(1), R5(-1)]", "[100, 0]"),
    )


async def test_return_at_top_level_ends_the_command():
    await run(("RETURN 2 + 3", "5"), ("PROG(1, RETURN 2, 3)", "2"))


# -- IF, AND, OR ----------------------------------------------------------------------


async def test_an_if_statement_with_an_undecided_test_runs_no_clause():
    await run(
        "> one(c) := PROG(IF(c > 0, \"pos\"))",
        ("[one(1), one(-1)]", '["pos", ?]'),
        "> evn(n) := PROG(IF(MOD(n, 2), RETURN \"even\"), \"odd\")",
        ("[evn(4), evn(7)]", '["even", "odd"]'),
        # A conditional that only gives a value is that value.
        "> cnd(x) := PROG(y := 2*x, IF(x > 0, y, -y))",
        ("cnd(z)", "IF(z > 0, 2*z, -2*z)"),
    )


async def test_a_program_that_cannot_decide_its_way_leaves_the_call_standing():
    await run(
        "> und(u, k) := PROG(k := 1, IF(u > 0, k := 2), k)",
        ("[und(3), und(-3), und(w)]", "[2, 1, und(w)]"),
        "> stp(x) := PROG(IF(x < 0, RETURN 0), IF(x < 1, RETURN x), 1)",
        ("[stp(-3), stp(1/2), stp(5), stp(x)]", "[0, 1/2, 1, stp(x)]"),
        ("stp(x) + 1", "stp(x) + 1"),
        "> g(y) := stp(y) + 1",
        ("g(t)", "stp(t) + 1"),
        # Nothing the call did is kept.
        "> cnt := 0",
        "> tick(x) := PROG(cnt := cnt + 1, IF(x > 0, RETURN 1), 0)",
        ("[tick(x), cnt]", "[tick(x), 0]"),
        ("[tick(2), cnt]", "[1, 1]"),
    )


async def test_a_program_passes_over_an_equation_it_cannot_decide():
    """An equation asked of symbolic values means "identically", as in the library's
    POLY_DEGREE, and a test over an element that doesn't exist can't be decided by
    any value: Derive goes past both, and so do programs written for it."""
    await run(
        "> nz(u) := PROG(IF(u = 0, RETURN \"zero\"), \"nonzero\")",
        ("[nz(0), nz(x^2 - 1)]", '["zero", "nonzero"]'),
        "> fst(v) := PROG(IF(v SUB 2 > 0, RETURN 1), 2)",
        ("fst([7])", "2"),
    )


async def test_a_call_past_an_undecided_equation_nests_little_deeper():
    """Going on past an undecided equation, a recursion over a symbolic value need not
    end: the call stands once it nests a little further, with nothing it did kept, and
    the same call over a number still computes. A loop may pass such a test any number
    of times."""
    await run(
        "> fact(m) := PROG(IF(m = 0, RETURN 1), RETURN m*fact(m - 1))",
        ("fact(5)", "120"),
        ("fact(n)", "fact(n)"),
        "> cnt := 0",
        "> deep(m) := PROG(cnt := cnt + 1, IF(m = 0, RETURN 1), RETURN deep(m - 1))",
        ("[deep(n), cnt]", "[deep(n), 0]"),
        ("[deep(100), cnt]", "[1, 101]"),
        "> many(u, n) := PROG(i := 0, LOOP(IF(i = n, RETURN i), IF(u = 0, RETURN -1), "
        "i := i + 1))",
        ("many(x, 3000)", "3000"),
    )


async def test_a_display_works_out_each_undecided_conditional_once(monkeypatch):
    """A program that adds up conditionals over undecided tests builds an answer whose
    conditionals contain the earlier ones in their tests, many times over. Its display
    works out each distinct conditional once, with its parents evaluated, and builds
    what that can't from the displays as written: sympy's folding of a conditional
    in a test into the other, which works through the sets the tests hold on, isn't
    run with evaluation off, where it builds ever larger expressions and simplifies
    them for seconds (more or fewer, in the order a set of its relations is visited
    in, so with the process's hash seed)."""
    attempts = []
    within = evaluate.within

    def counted(limit, compute):
        attempts.append(limit)
        return within(limit, compute)

    unevaluated = []
    rewrite = sp.Piecewise._eval_rewrite_as_ITE

    def folded(self, *args, **hints):
        if not global_parameters.evaluate:
            unevaluated.append(self)
        return rewrite(self, *args, **hints)

    monkeypatch.setattr(evaluate, "within", counted)
    monkeypatch.setattr(sp.Piecewise, "_eval_rewrite_as_ITE", folded)
    a = "IF(q < [1, 2, 3] AND [1, 2, 3] < 0, 4)"
    b = f"IF(q < [1, 2, 3] AND {a} > [1, 2, 3], 4)"
    c = f"IF(q < [1, 2, 3] AND [1, 2, 3] < {b} + {a}, 4)"
    await run(
        "> acc(p, q, k) := PROG(k := 0, k :+ IF(q < p < k, 4), k :+ IF(q < p < k, 4), "
        "k :+ IF(q < p < k, 4), k)",
        ("acc([1, 2, 3])", f"{b} + {a} + {c}"),
    )
    # The whole answer and the two conditionals whose tests hold others, once each.
    assert attempts.count(budget.REWRITING) == 3
    assert not unevaluated


async def test_a_program_passes_over_a_test_of_a_parameter_left_out():
    """A tolerance the call leaves out has no value: Derive's Romberg program then
    builds its whole table, and so does this."""
    await run(
        "> tol(x, eps) := PROG(IF(ABS(x) < eps, RETURN 0), x)",
        ("[tol(3), tol(3, 5), tol(3, 1)]", "[3, 0, 3]"),
        ("tol(y, 1)", "tol(y, 1)"),
    )


async def test_an_assignment_in_a_test_is_made():
    await run(
        "> cnt2(v, n, i) := PROG(n := DIM(v), i := 0, LOOP(IF(n < i := i + 1, exit)), i)",
        ("cnt2([a, b, c])", "4"),
        '> try1(x) := "inapplicable"',
        "> try2(x) := x^2",
        '> dsol(x, a_) := IF("inapplicable" = a_ := try1(x), IF("inapplicable" = '
        'a_ := try2(x), "none", a_, a_), a_, a_)',
        ("dsol(3)", "9"),
    )


async def test_and_and_or_evaluate_their_right_operand_only_when_needed():
    await run(
        "> seq1(b, f) := PROG(b := 2, (b :* 5) AND f := 7, [b, f])",
        ("seq1()", "[10, 7]"),
        "> seq2(b) := PROG(b := 1, (b := 2) AND RETURN [true, b], \"not reached\")",
        ("seq2()", "[true, 2]"),
        "> sc(n) := PROG(n := 0, IF(false AND 1 > n := 5, 1, 2), n)",
        ("sc()", "0"),
    )


async def test_an_equation_with_a_plus_or_minus_holds_if_either_sign_makes_it_hold():
    await run(
        '> jtest(b) := IF(MODS(b, 8) = ±3, "yes", "no")',
        ("[jtest(3), jtest(5), jtest(7)]", '["yes", "yes", "no"]'),
    )


async def test_the_infinity_of_a_division_by_zero_is_either_infinity_in_a_test():
    """`1/0` is `±inf`, so the continued fraction of a rational number ends where the
    remainder is 0."""
    await run(
        "> cf_vector(v) := IF(v SUB 1 = inf, v SUB 2, "
        "cf_vector([1/MOD(v SUB 1), APPEND(v SUB 2, [FLOOR(v SUB 1)])]), ?)",
        ("cf_vector([127/46, []])", "[2, 1, 3, 5, 2]"),
        ("[IF(1/0 = -inf, 1, 2, 3), IF(1/0 = 5, 1, 2, 3)]", "[1, 2]"),
    )


# -- statements -----------------------------------------------------------------------


async def test_statement_vectors_and_strings():
    await run(
        "> ea0(a, b, q_, r_) := PROG([q_ := FLOOR(a, b), r_ := a - q_*b], "
        "[a := b, b := r_], [a, b, q_, r_])",
        ("ea0(8991, 3293)", "[3293, 2405, 2, 2405]"),
        "> swp(u, v, s_) := PROG([s_ := u, u := v, v := s_], [u, v])",
        ("swp(1, 2)", "[2, 1]"),
        '> cm(n) := PROG("n := 0", "@ a comment", n + 1)',
        ("cm(4)", "5"),
        "> vst(n) := PROG([n :+ 1, n :* 2])",
        ("vst(3)", "[4, 8]"),
    )


async def test_a_declaration_in_a_program_declares_the_global():
    session = await run(
        "> dcl(n) := PROG(m :epsilon Integer, SIN(m*pi))",
        ("dcl(1)", "0"),
    )
    assert "m" in session.domains


# -- iterating built-ins ----------------------------------------------------------


async def test_a_binder_body_is_evaluated_anew_at_each_value_with_its_effects():
    await run(
        ("ITERATES(DIF(f, x), f, x^10, 3)", "[x^10, 10*x^9, 90*x^8, 720*x^7]"),
        "> acc(n, t) := PROG(t := [], VECTOR(t := APPEND(t, [k^2]), k, 1, n), t)",
        ("acc(4)", "[1, 4, 9, 16]"),
        "> acc2(m, t) := PROG(t := [], MAP(t := APPEND(t, [k]), k, m, 1, -1), t)",
        ("acc2(3)", "[3, 2, 1]"),
        "> g2(n, s, m) := PROG(s := 0, m := 0, "
        "LOOP(IF(m = n, RETURN s), s :+ SUM(j, j, 1, m), m :+ 1))",
        ("g2(4)", "10"),
    )


async def test_a_binder_variable_hides_a_global_only_for_the_iteration():
    await run(
        "> i := 77",
        ("VECTOR(i^2, i, 1, 3)", "[1, 4, 9]"),
        ("i", "77"),
        "> inn(n) := SUM(k_, k_, 1, n)",
        ("SUM(inn(k_), k_, 1, 3)", "10"),
    )


async def test_a_stored_value_read_under_a_binder_keeps_its_own_bound_variables():
    await run(
        "> w := INT(f(t, x), t, 0, x)",
        ("VECTOR(w, x, 1, 2)", "[INT(f(t, 1), t, 0, 1), INT(f(t, 2), t, 0, 2)]"),
        ("VECTOR(w, t, 1, 2)", "[INT(f(t, x), t, 0, x), INT(f(t, x), t, 0, x)]"),
        "> v := DIF(g(x, t), x) + t",
        ("VECTOR(v, t, 1, 2)", "[DIF(g(x, 1), x) + 1, DIF(g(x, 2), x) + 2]"),
    )


async def test_a_return_in_a_binder_body_ends_the_activation():
    await run(
        "> rr(k) := PROG(k := 0, VECTOR(IF(i = 3, RETURN i*100, i), i, 1, 5))",
        ("rr()", "300"),
    )


async def test_random_draws_anew_at_each_evaluation():
    await run(
        ("DIM(VECTOR(RANDOM(10^9), i, 1, 10) \\ {})", "10"),
        "> die(n) := RANDOM(10^9)",
        "> d := die()",
        ("DIM({d, d, d, d})", "4"),
    )


async def test_iterates_runs_until_a_value_repeats_any_earlier_one():
    await run(
        "> collatz(n) := IF(MOD(n, 2) = 0, n/2, 3*n + 1)",
        (
            "ITERATES(collatz(k), k, 17, 20)",
            "[17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1, 4, 2, 1, 4, 2, 1, 4, 2]",
        ),
        (
            "ITERATES(collatz(k), k, 17)",
            "[17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1, 4]",
        ),
        ("ITERATE(collatz(k), k, 17)", "?"),
        ("ITERATES(FLOOR(x, 10), x, 258)", "[258, 25, 2, 0, 0]"),
        "> cyc(n, m) := ITERATES([MOD(v SUB 1 + 1, n), v SUB 2], v, [0, 0], m)",
        ("DIM(cyc(1560, inf)) - 1", "1560"),
    )


async def test_an_iterate_undefined_at_its_values_comes_from_the_simplified_body():
    """A divided difference is 0/0 where the iterates have converged; simplified with
    the variable symbolic, as Derive simplifies the body, it is a polynomial."""
    await run(
        (
            "ITERATES([m SUB 2, (m SUB 2^3 - m SUB 1^3)/(m SUB 2 - m SUB 1)], m, "
            "[10, 10], 1)",
            "[[10, 10], [10, 300]]",
        ),
        (
            "ITERATES([ELEMENT(m, 2), (ELEMENT(m, 2)^3 - ELEMENT(m, 1)^3)/"
            "(ELEMENT(m, 2) - ELEMENT(m, 1))], m, [10, 10], 1)",
            "[[10, 10], [10, 300]]",
        ),
    )


async def test_a_count_that_is_infinite_or_an_unassigned_variable_is_no_count():
    await run(
        "> palindrome_process(n, m) := "
        "ITERATES(IF(k = REVERSE(k), k, k + REVERSE(k)), k, n, m)",
        ("palindrome_process(19)", "[19, 110, 121, 121]"),
        ("palindrome_process(196, 3)", "[196, 887, 1675, 7436]"),
        ("ITERATES(1/x, x, 2, inf)", "[2, 1/2, 2]"),
    )


async def test_iterate_over_vectors_and_calls():
    await run(
        (
            "ITERATES([n + 1, ADJOIN(0, v) + APPEND(v, [0])], [n, v], [0, [1]], 2)",
            "[[0, [1]], [1, [1, 1]], [2, [1, 2, 1]]]",
        ),
        "> F(x) := x^2",
        "> G(x) := 2*x + 3",
        ("ITERATES(F(x), x, x, 3)", "[x, x^2, x^4, x^8]"),
        ("ITERATES(G(x), x, x, 3)", "[x, 2*x + 3, 4*x + 9, 8*x + 21]"),
        ("ITERATES(G(x), x, 2, 3)", "[2, 7, 17, 37]"),
        ("ITERATES(NEXT_PRIME(p_^3), p_, 2, 3)", "[2, 11, 1361, 2521008887]"),
        "> FAKIT(n) := (ITERATE([i + 1, (i + 1)*j], [i, j], [0, 1], n)) SUB 2",
        ("FAKIT(5)", "120"),
        "> SUBEQ(eq, expr) := ITERATE(eq, LHS(expr), RHS(expr), 1)",
        ("SUBEQ([x/(x^2 + y^2), SQRT(1 - x*y)], [x = 1, y = 0])", "[1, 1]"),
    )


async def test_map_list_maps_over_a_collection_or_a_range():
    await run(
        ("MAP_LIST(k^2, k, [2, 3, 5])", "[4, 9, 25]"),
        ("MAP_LIST(k^2, k, {-1, 1, 2})", "{1, 4}"),
        ("MAP_LIST(k^2, k, 0, 3)", "[0, 1, 4, 9]"),
        ("MAP_LIST(k, k, 5, 1, -2)", "[5, 3, 1]"),
        ("MAP_LIST(k, k, 3, 1)", "[]"),
        ("MAP_LIST(k^2, k, 3)", "[1, 4, 9]"),
        ('MAP_LIST(k, k, "abc")', '["a", "b", "c"]'),
        ("MAP_LIST(DIM(TERMS(EXPAND(f))), f, [(x + 1)^2, x*(x + 1), 7])", "[3, 2, 1]"),
        ("MAP(k^2, k, [1, 2, 3])", "true"),
        ("MAP_LIST(k^2, k, w)", "MAP_LIST(k^2, k, w)"),
        "> tocyc(c_) := MAP_LIST(c_ SUB k_, k_, {1, ..., DIM(c_)})",
        ("tocyc([7, 5, 7])", "{5, 7}"),
    )


async def test_map_list_over_a_set_gives_a_set():
    await run(
        "> G := [[1, 0, 0, 0, 0, 1], [0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0]]",
        (
            "MAP_LIST(MOD(v*G, 2), v, {0, 1}^3)",
            "{[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 0], "
            "[0, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 1], [1, 0, 1, 0, 1, 1], "
            "[1, 1, 0, 1, 0, 1], [1, 1, 1, 1, 1, 1]}",
        ),
    )


async def test_a_body_whose_values_are_unknown_is_not_evaluated():
    await run(
        "> cnt := 0",
        ("MAP(cnt := cnt + 1, k, w)", "MAP(cnt := 1, k, w)"),
        ("cnt", "0"),
    )


async def test_every_and_some_stop_at_the_value_that_decides():
    await run(
        ("EVERY(MOD(x^14701, 15251) = x, x, 0, 100)", "true"),
        ("EVERY(k > 0, k, [1, 2, 3])", "true"),
        ("EVERY(k > 1, k, [1, 2, 3])", "false"),
        ("SOME(k > 2, k, [1, 2, 3])", "true"),
        ("SOME(k > 5, k, 1, 3)", "false"),
        ("[EVERY(k > 0, k, []), SOME(k > 0, k, [])]", "[true, false]"),
        ("EVERY(k > w, k, [1, 2])", "EVERY(k > w, k, [1, 2])"),
        ("SOME(k > w, k, [1, 2])", "SOME(k > w, k, [1, 2])"),
        # The values of a range are made as they are reached, not all ahead.
        ("[EVERY(k > k, k, 1, 10^20), SOME(k > 5, k, 10^30)]", "[false, true]"),
        "> evl(n) := PROG([EVERY(k > k, k, 1, n), SOME(k > 5, k, 1, n, 2)])",
        ("evl(10^20)", "[false, true]"),
        "> evs(n, c) := PROG(c := 0, EVERY(PROG(c :+ 1, k < 3), k, 1, n), c)",
        ("evs(10)", "3"),
        "> inap(x, n) := IF(SOME(NOT RATIONAL?(t_), t_, x) OR EVERY(t_ = 0, t_, x) "
        'OR 2 > n := DIM(x), "inapplicable", n)',
        ('[inap([0, 0]), inap([1])]', '["inapplicable", "inapplicable"]'),
        "> SQUAREFREE(n) := EVERY(v_ SUB 2 = 1, v_, FACTORS(n))",
        ("[SQUAREFREE(30), SQUAREFREE(12)]", "[true, false]"),
    )


# -- settings -----------------------------------------------------------------------


async def test_a_setting_reads_as_its_value_and_changes_at_once():
    await run(
        "> tp(s, m) := PROG(s := Precision, Precision := Approximate, m := Precision, "
        "Precision := s, [s, m])",
        ("tp()", "[Exact, Approximate]"),
        ("Precision", "Exact"),
        "> ta(x) := PROG(Precision := Approximate, SQRT(x))",
        ("ta(2)", "1.414213562"),
        ("Precision", "Approximate"),
    )


async def test_only_the_statements_after_a_setting_feel_it():
    await run(
        "> tr(x, s, t) := PROG(t := RATIONAL?(SQRT(x) - 1), Precision := Approximate, "
        "s := RATIONAL?(SQRT(x) - 1), Precision := Exact, [t, s])",
        ("tr(2)", "[false, true]"),
        "> bits(n) := PROG(OutputBase := Binary, n := NAME_TO_CODES(STRING(n)), "
        "OutputBase := Decimal, n - 48)",
        ("bits(10)", "[1, 0, 1, 0]"),
    )


async def test_residues_at_roots_a_program_finds_are_taken_at_the_roots():
    """An inverse Laplace transform as the sum of the residues of `F(s)*#e^(s*t)` at
    the roots of its denominator (Beaudin, DUG Newsletter 111). Under Approximate the
    roots stay exact in the program, so each limit is a residue, not 0."""
    await run(
        "> Precision := Approximate",
        "> ilt(F, s, t, r, k, v) := PROG(r := SOLUTIONS(DENOMINATOR(F), s), k := 1, "
        "v := 0, LOOP(IF(k > DIM(r), RETURN v), "
        "v :+ LIM((s - r SUB k)*F*#e^(s*t), s, r SUB k), k :+ 1))",
        (
            "ilt(1/(s^2 + 0.1*s + 9), s, t)",
            "0.3333796392*SIN(2.999583304*t)*#e^(-0.05*t)",
        ),
    )


async def test_an_integer_is_a_sequence_of_the_digits_the_output_base_shows():
    await run(
        "> nb(n, s) := PROG(OutputBase := Binary, s := [DIM(n), n SUB 2, "
        "REVERSE(n), APPEND(\"0\", n)], OutputBase := Decimal, s)",
        ("nb(74)", '[7, 0, 41, "01001010"]'),
        ("[DIM(74), 74 SUB 2]", "[2, 4]"),
    )


async def test_a_setting_assigned_in_a_program_persists():
    session = await run(
        "> tg(dummy) := PROG(Trigonometry := Collect, 0)",
        ("tg()", "0"),
        ("Trigonometry", "Collect"),
        (
            "[Trigonometry := Auto, Notation := Rational]",
            "[Trigonometry := Auto, Notation := Rational]",
        ),
        ("Trigonometry", "Auto"),
    )
    assert session.settings["Trigonometry"] == "Auto"


async def test_entering_a_program_changes_no_setting():
    session = await run("> tq(dummy) := PROG(Precision := Approximate, 0)")
    assert session.settings["Precision"] == "Exact"


async def test_a_parameter_named_after_a_setting_is_a_local_setting():
    await run(
        "> PrecisionDigits := 10",
        "> tq(x, PrecisionDigits := 20) := PrecisionDigits",
        ("[tq(1), tq(1, 30)]", "[20, 30]"),
        ("PrecisionDigits", "10"),
        "> pd(x, d := PrecisionDigits) := d",
        ("[pd(1), pd(1, 30)]", "[10, 30]"),
        # The value is computed to ten digits; outside, it is the exact number those
        # digits are, shown as Exact shows it.
        "> ap(x, Precision := Approximate) := SQRT(x)",
        ("[ap(2), SQRT(2)]", "[131836323/93222358, SQRT(2)]"),
    )


async def test_a_setting_a_program_assigns_holds_for_the_rest_of_the_command():
    """`appi` raises the precision to compute digits of pi: its value keeps them, and
    `STRING` of it writes them under the digits `appi` set."""
    await run(
        "> Precision := Approximate",
        "> PrecisionDigits := 10",
        "> appi(x, nd) := PROG(PrecisionDigits := nd + 5, NotationDigits := nd, "
        "t := 0, k := 2*nd + 1, LOOP(t := 1/k - t*x^2, k := k - 2, IF(k < 1, exit)), "
        "RETURN(t*x))",
        "> pg(nd, s) := PROG(s := STRING(16*appi(1/5, nd) - 4*appi(1/239, nd)), s)",
        ('pg(30)', '"3.14159265358979323846264338327"'),
    )


async def test_an_invalid_setting_value_changes_nothing():
    await run(
        "> tb(dummy) := PROG(Precision := Sideways, 0)",
        ("tb()", "0"),
        ("Precision", "Exact"),
    )


# -- variables holding programs ---------------------------------------------------


async def test_a_variable_holding_a_program_runs_it_at_each_reference():
    await run(
        "> new := PROG(xcor_ := 0, ycor_ := 0, trace_ := [[0, 0]], RETURN trace_)",
        "> xcor_ := 5",
        ("new", "[[0, 0]]"),
        ("xcor_", "0"),
        "> tk := 0",
        "> tick := PROG(tk := tk + 1, tk)",
        ("[tick, tick, tick]", "[1, 2, 3]"),
        ("tick + tick", "9"),
        "> usep(n) := PROG(n := tick, n*10)",
        ("usep()", "60"),
    )


# -- limits -----------------------------------------------------------------------


async def test_a_command_past_the_step_limit_is_abandoned_without_effect():
    await run(
        "> gl := 1",
        "> spin2(n) := LOOP(gl := gl + 1)",
        ("spin2(0)", "(error: Step limit exceeded)"),
        ("gl", "1"),
        "> spin3(u, k) := PROG(k := 0, LOOP(IF(k > u, exit), k :+ 1), k)",
        ("spin3(3)", "4"),
        # Over a symbolic bound the loop can't decide its way: the call stands.
        ("spin3(w)", "spin3(w)"),
        ("ITERATES(k + 1, k, 0)", "(error: Step limit exceeded)"),
        step_limit=1000,
    )


def test_a_session_computing_in_process_has_a_step_limit():
    assert Session().step_limit is not None
    assert Session(step_limit=None).step_limit is None


# -- definitions simplified -------------------------------------------------------


async def test_a_simplified_definition_is_made_with_the_simplified_body():
    session = await run(
        "> gg(x) := SIN(x)^2 + COS(x)^2",
        ("gg(x) := SIN(x)^2 + COS(x)^2", "gg(x) := 1"),
        ("gg(a + b)", "1"),
        "> ff(x, k := 2) := x^k + 1 + 1",
        ("ff(x, k := 2) := x^k + 1 + 1", "ff(x, k := 2) := x^k + 2"),
        ("ff(3)", "11"),
    )
    assert session.functions["gg"].body.value == "1"


async def test_a_program_called_in_a_definition_being_shown_runs():
    await run(
        "> InputMode := Word",
        "> P(u, a, b) := PROG(a := u^2, b := a + 1, RETURN(b*t))",
        ("h(t) := P(3)", "h(t) := 10*t"),
        ("g(t) := P(t)", "g(t) := t*(t^2 + 1)"),
        # One that does not end over a symbolic argument stands.
        "> Q(n) := LOOP(IF(n = 0, RETURN(1)), n :+ 0)",
        ("k(s) := Q(s)", "k(s) := Q(s)"),
    )


async def test_a_definition_takes_apart_an_expression_in_its_parameters():
    """`TERMS` of a polynomial in a parameter is worked out, the parameter standing for
    itself, as every other part of a body shown is: the coefficients reduced modulo a
    prime."""
    await run(
        "> InputMode := Word",
        "> POLYMODS(u, m, x) := SUM(VECTOR(MODS(LIM(u_, x, 1), m)*u_/LIM(u_, x, 1), u_, "
        "TERMS(EXPAND(u))))",
        (
            "F1(x) := POLYMODS(x^4 + 75*x^3 + 6*x^2 + 12*x + 5, 79, x)",
            "F1(x) := x^4 - 4*x^3 + 6*x^2 + 12*x + 5",
        ),
    )


async def test_a_binding_over_a_collection_still_to_come_keeps_its_body():
    """The elements may mention any variable the collection does, so nothing in the
    body treats the bound variable as independent of them."""
    await run(
        "> InputMode := Word",
        (
            "H(y) := SUM(VECTOR(DIF(u_, x), u_, TERMS(y*x^2 + 5*x)))",
            "H(y) := SUM(VECTOR(DIF(u_, x), u_, TERMS(x^2*y + 5*x)))",
        ),
        ("H(2)", "4*x + 5"),
        ("SUM(k^2, k, m)", "m*(m + 1)*(2*m + 1)/6"),
    )


# -- display ------------------------------------------------------------------------


def shown(text):
    """The lines an entered `text` is drawn in, right blanks stripped."""
    entry = Session().author(text)
    return [line.rstrip() for line in entry.layout.lines]


def test_a_program_is_drawn_as_an_indented_block():
    assert shown(
        "Rev(v, w) := PROG(w := [], LOOP(IF(v = [], RETURN w), "
        "w := ADJOIN(FIRST(v), w), v := REST(v)))"
    ) == [
        "Rev(v, w) :=",
        "  Prog",
        "    w := []",
        "    Loop",
        "      If v = []",
        "         RETURN w",
        "      w := ADJOIN(FIRST(v), w)",
        "      v := REST(v)",
    ]


def test_each_clause_of_a_block_if_is_a_line_under_the_test():
    assert shown(
        "ea(a, b, q_, r_, s_ := []) := LOOP(IF(b = 0, RETURN(s_)), "
        "[q_ := FLOOR(a, b), r_ := a - q_*b], IF(r_ > 0, s_ := INSERT([a = q_*b + r_], "
        "s_, 0), s_ := INSERT([a = q_*b], s_, 0)), [a := b, b := r_])"
    ) == [
        "ea(a, b, q_, r_, s_ := []) :=",
        "  Loop",
        "    If b = 0",
        "       RETURN s_",
        "    [q_ := FLOOR(a, b), r_ := a - q_·b]",
        "    If r_ > 0",
        "       s_ := INSERT([a = q_·b + r_], s_, 0)",
        "       s_ := INSERT([a = q_·b], s_, 0)",
        "    [a := b, b := r_]",
    ]


def test_an_if_that_exits_stays_on_its_line():
    assert shown(
        "half(n, s_) := PROG(s_ := n - 1, LOOP(s_ :/ 2, IF(MOD(s_, 2) = 1, exit)), s_)"
    ) == [
        "half(n, s_) :=",
        "  Prog",
        "    s_ := n - 1",
        "    Loop",
        "      s_ :/ 2",
        "      If MOD(s_, 2) = 1 exit",
        "    s_",
    ]


def test_nested_ifs_in_block_position_are_blocks():
    assert shown("sg(a) := IF(a = 0, 0, IF(a > 0, 1, -1))") == [
        "sg(a) :=",
        "  If a = 0",
        "     0",
        "     If a > 0",
        "        1",
        "        -1",
    ]


def test_a_construct_that_is_an_operand_or_a_value_is_drawn_as_a_call():
    assert shown('pq(c) := PROG(sig := IF(c, x, y), "text", sig)') == [
        "pq(c) :=",
        "  Prog",
        "    sig := IF(c, x, y)",
        '    "text"',
        "    sig",
    ]
    assert shown("1 + IF(x > 0, 2, 3)") == ["1 + IF(x > 0, 2, 3)"]


def test_a_top_level_assignment_of_a_program_is_a_block():
    assert shown("tick := PROG(k := k + 1, k)") == [
        "tick :=",
        "  Prog",
        "    k := k + 1",
        "    k",
    ]


# -- output and files -----------------------------------------------------------------


async def displays(session, text, expected, displayed):
    """Simplify `text` in `session`, checking its answer, and that the string entries
    before it hold the texts `displayed`, in order."""
    count = len(session.entries)
    answer = await session.simplify(text)
    assert answer.text == expected, text
    added = [entry.text for entry in session.entries[count:-1]]
    assert added == [f'"{each}"' for each in displayed], text


async def test_display_shows_a_value_before_the_answer():
    session = await run("> testd(a) := PROG(DISPLAY(a), a*a)")
    await displays(session, "testd(3)", "9", ["3"])
    await displays(
        session,
        "PROG(a := EXPAND((x + 1)^3), b := TERMS(a), DISPLAY(DIM(b)))",
        "4",
        ["4"],
    )
    await displays(session, 'DISPLAY(APPEND(STRING(57), "-gon"))', '"57-gon"', ["57-gon"])


async def test_display_shows_a_string_as_its_characters():
    session = await run(
        '> tf(q) := PROG(DISPLAY("Vereinfache step auf step - Simplify step by step"), '
        'DISPLAY(""))'
    )
    await displays(
        session, "tf(1)", '""', ["Vereinfache step auf step - Simplify step by step", ""]
    )


async def test_display_shows_each_time_it_is_evaluated():
    session = await run(
        "> shuf(n, k) := PROG(k := 0, LOOP(IF(k = n, RETURN k), "
        "DISPLAY(VECTOR(i, i, 1, k + 1)), k :+ 1))",
        "> dq(x) := DISPLAY(x) + 1",
    )
    await displays(session, "shuf(3)", "3", ["[1]", "[1, 2]", "[1, 2, 3]"])
    await displays(session, "VECTOR(dq(k), k, 1, 2)", "[2, 3]", ["1", "2"])


async def test_display_shows_a_value_as_the_settings_in_force_write_it():
    session = await run()
    await displays(
        session,
        "PROG(Notation := Decimal, Precision := Approximate, DISPLAY(1/4), 1/2)",
        "0.5",
        ["0.25"],
    )


async def test_a_display_with_a_wrong_line_count_stands():
    session = await run(
        ("DISPLAY(x, 0)", "DISPLAY(x, 0)"), ("DISPLAY(x, 1, 2)", "DISPLAY(x, 1, 2)")
    )
    assert len(session.entries) == 2


async def test_nothing_is_shown_when_a_definition_is_entered():
    session = await run("> dz(x) := DISPLAY(x)", ("dz(5)", "5"))
    texts = [entry.text for entry in session.entries]
    assert texts == ["dz(x) := DISPLAY(x)", '"5"', "5"]


async def test_what_was_displayed_stays_when_the_command_is_abandoned():
    session = Session(step_limit=1000)
    session.author("dl(k) := LOOP(DISPLAY(k), k :+ 1, IF(k > 2, LOOP(k :+ 1)))")
    with pytest.raises(EngineError):
        await session.simplify("dl(1)")
    assert [entry.text for entry in session.entries[1:]] == ['"1"', '"2"']


async def test_write_shows_its_text_on_the_message_line():
    session = await run(
        "> wr(n, m) := PROG(m := 0, LOOP(WRITE(m), IF(m >= n, exit), m :+ 1), m)"
    )
    count = len(session.entries)
    assert (await session.simplify("wr(5)")).text == "5"
    assert len(session.entries) == count + 1
    assert session.take_notice() == "5"
    text = "Sorry: Unable to find this particular Laplace transform"
    assert (await session.simplify(f'WRITE("{text}")')).text == f'"{text}"'
    assert session.take_notice() == text
    assert session.take_notice() is None


async def test_progress_is_reported_as_the_program_runs():
    session = Session()
    heard = []
    session.runner.progress = lambda text, displayed: heard.append((text, displayed))
    session.author("pr(n) := PROG(WRITE(n), DISPLAY(n + 1), n + 2)")
    assert (await session.simplify("pr(1)")).text == "3"
    assert heard == [("1", False), ("2", True)]


@pytest.fixture
def util1(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "util1.mth").write_text("SQ1(x) := x^2\n\nADD1(x, y) := x + y\n")


async def test_load_loads_a_utility_file(util1):
    session = await run(
        ('LOAD("util1.mth")', "true"),
        ("SQ1(4)", "16"),
        ('LOAD("no_such_file.mth")', 'LOAD("no_such_file.mth")'),
        "> ldr(f) := PROG(LOAD(f), SQ1(5))",
        ('ldr("util1.mth")', "25"),
        ("LOAD(3)", "LOAD(3)"),
        ("LOAD(x)", "LOAD(x)"),
    )
    assert session.take_notice() is None


async def test_load_in_a_program_loads_before_the_statements_after_it(util1):
    session = await run(
        "> InputMode := Word", ('PROG(LOAD("util1.mth"), ADD1(3, 4))', "7")
    )
    assert "SQ1" in session.functions


async def test_a_missing_file_is_named_on_the_message_line(util1):
    session = Session()
    assert (await session.simplify('LOAD("C:\\nowhere\\nothing.mth")')).text == (
        'LOAD("C:\\nowhere\\nothing.mth")'
    )
    assert session.take_notice() == "nothing.mth could not be found"


# -- complete programs ----------------------------------------------------------------


async def test_josephus_deletes_nothing_past_the_end():
    """The survivor of 41 counted off by threes; at each step the program deletes the
    element one past the end of the shorter of its two pieces."""
    await run(
        "> InputMode := Word",
        "> r(x) := VECTOR(y, y, x)",
        "> g(v, k) := VECTOR(v SUB n, n, IF(k = DIM(v), DIM(v) - 1, MOD(k, DIM(v))))",
        "> f(v, k) := VECTOR(v SUB n, n, IF(k = DIM(v), DIM(v) + 1, MOD(k, DIM(v)) + 1), "
        "DIM(v))",
        "> h(v, k) := APPEND(f(v, k), g(v, k))",
        "> i(v, k) := DELETE(h(v, k), DIM(v))",
        "> jo(x, k) := (ITERATE(i(w, k), w, r(x), x - 1)) SUB 1",
        ("jo(41, 3)", "31"),
    )


async def test_a_pade_approximant_matches_coefficients_by_their_string_names():
    await run(
        "> InputMode := Word",
        "> pade(y, x, x0, n, d, j_ := 0, t_ := 0) := PROG(LOOP(t_ :+ LIM(y, x, x0)*x^j_, "
        "j_ :+ 1, IF(j_ > n + d, exit), y := DIF(y, x)/j_), y := t_*(SUM(STRING(k_ + "
        "n)*x^k_, k_, 1, d) + 1), y := REVERSE(TERMS(y - SUM(STRING(k_)*x^k_, k_, 0, n), "
        "x)), y := LIM(y SUB [1, ..., n + d + 1], x, 1), y := FIRST(SOLUTIONS(y, "
        "STRING([0, ..., n + d]))), SUM(y SUB (k_ + 1)*(x - x0)^k_, k_, 0, n)/(SUM(y SUB "
        "(k_ + n + 1)*(x - x0)^k_, k_, 1, d) + 1))",
        ("pade(EXP(x), x, 0, 3, 2)", "(x^3 + 9*x^2 + 36*x + 60)/(3*(x^2 - 8*x + 20))"),
    )


async def test_estimating_pi_shows_the_polygon_it_stops_at():
    session = await run(
        "> InputMode := Word",
        '> OdhadPi(n) := PROG(k := 5, Precision := Approximate, LOOP(d := k*SIN(pi/k), '
        "h := k*TAN(pi/k), IF(FLOOR(d*10^n) = FLOOR(h*10^n), "
        'PROG(DISPLAY(APPEND(STRING(k), "-uhelnik")), RETURN APPROX([d, h], 10))), '
        "k := k + 1))",
    )
    await session.simplify("OdhadPi(2)")
    assert session.entries[-2].text == '"57-uhelnik"'
    assert session.settings["Precision"] == "Approximate"
    assert (await session.simplify("k")).text == "57"


async def test_the_arithmetic_geometric_mean_checks_its_arguments():
    """An `APPROX` call inside a vector answers a rational, shown by the notation in
    force; approximated, it shows the digits the original printed, 15.33074234."""
    session = await run(
        "> InputMode := Word",
        "> agm(x, y, e, a, g, n, q) := PROG(IF(NOT NUMBER?(x) OR NOT NUMBER?(y), "
        'RETURN "invalid arguments"), q := [x, y], n := 1, '
        "LOOP(IF(ABS(q SUB 1 - q SUB 2) < 10^(-e), exit), "
        "a := AVERAGE(q SUB 1, q SUB 2), g := SQRT(q SUB 1*q SUB 2), q := [a, g], "
        "n :+ 1), [APPROX(a), n - 1])",
        ("agm(27, 7, 3)", "[802733/52361, 3]"),
        ("agm(12, 7, 10)", "[550531/58995, 4]"),
        ("agm(a, 7, 10)", '"invalid arguments"'),
    )
    assert (await session.approx("agm(27, 7, 3)")).text == "[15.33074234, 3]"
    assert (await session.approx("agm(12, 7, 10)")).text == "[9.331824730, 4]"


async def test_pollard_s_rho_shows_how_many_iterations_it_took():
    session = await run(
        "> InputMode := Word",
        "> rho(n, x := 3, y := 3, d_, i_ := 0) := PROG(LOOP(i_ :+ 1, "
        "x := MOD(x^2 + 1, n), y := MOD(MOD(y^2 + 1, n)^2 + 1, n), "
        "d_ := GCD(x - y, n), IF(d_ > 1, exit)), "
        'DISPLAY(APPEND(STRING(i_), " iterations")), MIN(d_, n/d_))',
    )
    await displays(session, "rho(35)", "5", ["1 iterations"])


async def test_the_jacobi_symbol():
    await run(
        "> JACOBI(a, b, s_ := 1, t_) := PROG(IF(b < 0 OR MOD(b, 2) = 0, RETURN ?), "
        "IF(GCD(a, b) > 1, RETURN 0), LOOP(IF(a = 1 OR b = 1, RETURN s_), "
        "a := MODS(a, b), IF(a < 0 AND MOD(b, 4) = 3, s_ := -s_), a := ABS(a), t_ := 0, "
        "LOOP(IF(MOD(a, 2) = 1, exit), t_ :+ 1, a :/ 2), "
        "IF(MOD(t_, 2) = 1 AND MODS(b, 8) = ±3, s_ := -s_), t_ := a, a := b, b := t_, "
        "IF(MOD(a, 4) = 3 AND MOD(b, 4) = 3, s_ := -s_)))",
        ("JACOBI(10^60 + 1, 10^68 + 99)", "1"),
        ("JACOBI(5, 16937389168607)", "-1"),
        ("[JACOBI(2, 7), JACOBI(2, 3), JACOBI(3, 9), JACOBI(3, 8)]", "[1, -1, 0, ?]"),
    )


async def test_a_loop_runs_thousands_of_passes_quickly():
    start = time.perf_counter()
    await run(
        "> SQS(n, k, s) := PROG(k := 0, s := [], LOOP(IF(k = n, RETURN DIM(s)), "
        "k :+ 1, s := APPEND(s, [k^2])))",
        ("SQS(2000)", "2000"),
    )
    assert time.perf_counter() - start < 20


async def test_a_list_built_or_consumed_element_by_element_takes_linear_time():
    """`ADJOIN`, `APPEND` and `REST` share the elements of the list they are given, so
    a program building or consuming a long list one element at a time does not copy it
    at every step."""
    start = time.perf_counter()
    await run(
        "> BLD(n, v := [], i := 0) := LOOP(i :+ 1, v := ADJOIN(i, v), "
        "IF(i >= n, RETURN v))",
        "> APN(n, v := [], i := 0) := LOOP(i :+ 1, v := APPEND(v, [i]), "
        "IF(i >= n, RETURN v))",
        "> CNS(v, s := 0) := LOOP(IF(v = [], RETURN s), s :+ FIRST(v), v := REST(v))",
        ("[CNS(BLD(30000)), CNS(APN(30000))]", "[450015000, 450015000]"),
    )
    assert time.perf_counter() - start < 20


async def test_lists_made_from_one_list_keep_their_own_elements():
    await run(
        "> APN(n, v := [], i := 0) := LOOP(i :+ 1, v := APPEND(v, [i]), "
        "IF(i >= n, RETURN v))",
        "> w := APN(20)",
        (
            "[ADJOIN(0, w) SUB 1, ADJOIN(-1, w) SUB 1, APPEND(w, [21]) SUB -1, "
            "APPEND(w, [22, 23]) SUB 21, APPEND([0], w) SUB 1, DIM(w), w SUB -1]",
            "[0, -1, 21, 22, 0, 20, 20]",
        ),
        (
            "[REST(w, 18), REST(REST(w), -17), ADJOIN(0, REST(w, 17))]",
            "[[19, 20], [2, 3], [0, 18, 19, 20]]",
        ),
    )


async def test_a_list_built_element_by_element_under_approximate_precision():
    await run(
        "> Precision := Approximate",
        "> APX(n, v := [], i := 0) := LOOP(i :+ 1, v := APPEND(v, [i/7]), "
        "IF(i >= n, RETURN v))",
        ("REST(APX(20), 17)", "[2.571428571, 2.714285714, 2.857142857]"),
    )


async def test_number_theory_utilities():
    await run(
        "> SOLVE_MOD(u, x, m, a_, b_, d_) := PROG(a_ := DIF(LHS(u) - RHS(u), x), "
        "b_ := LIM(RHS(u) - LHS(u), x, 0), d_ := GCD(a_, m), "
        "IF(MOD(b_, d_) /= 0, RETURN []), m := m/d_, "
        "VECTOR(MOD(b_/d_*INVERSE_MOD(a_/d_, m), m) + k_*m, k_, 0, d_ - 1))",
        ("SOLVE_MOD(29*x = 1, x, 660)", "[569]"),
        ("SOLVE_MOD(3*x = 0, x, 9)", "[0, 3, 6]"),
        ("SOLVE_MOD(3*x = 6, x, 9)", "[2, 5, 8]"),
        ("SOLVE_MOD(3*x = 1, x, 9)", "[]"),
        "> CRT(a, m, m_ := 1, x_ := 0) := LOOP(IF(m = [], RETURN x_), "
        "x_ := x_ + (FIRST(a) - x_)*m_*INVERSE_MOD(m_, FIRST(m)), m_ :* FIRST(m), "
        "x_ := MOD(x_, m_), a := REST(a), m := REST(m))",
        ("CRT([1, 2, 3], [4, 5, 11])", "157"),
        "> FAREY(n, a_ := 1, b_ := 1, l_ := [1]) := LOOP(IF(a_ = b_, IF(b_ = n, "
        "RETURN SORT(l_), [b_ :+ 1, a_ := 1])), IF(GCD(a_, b_) = 1, "
        "l_ := APPEND(l_, [a_/b_])), a_ :+ 1)",
        ("FAREY(6)", "[1/6, 1/5, 1/4, 1/3, 2/5, 1/2, 3/5, 2/3, 3/4, 4/5, 5/6, 1]"),
        (
            "FAREY(7)",
            "[1/7, 1/6, 1/5, 1/4, 2/7, 1/3, 2/5, 3/7, 1/2, 4/7, 3/5, 2/3, 5/7, 3/4, "
            "4/5, 5/6, 6/7, 1]",
        ),
        "> rabin_miller(n, a, s_) := PROG(s_ := n - 1, LOOP(s_ :/ 2, "
        "IF(MOD(s_, 2) = 1, exit)), a := -ABS(MODS(a^s_, n)), LOOP(IF(a = -1, "
        "RETURN true), s_ :* 2, IF(s_ = n - 1, RETURN false), a := MODS(a^2, n)))",
        ("rabin_miller(561, 2)", "false"),
        ("rabin_miller(10^100 + 267, 2)", "true"),
        "> pprime?(n, k := 6, d_ := 2, s_, t_ := 5) := PROG(IF(GCD(n, 6) > 1, "
        "RETURN MEMBER?(n, [2, 3])), s_ := MIN(FLOOR(SQRT(n)), 1021), LOOP(IF(t_ > s_, "
        "exit), IF(MOD(n, t_) = 0, RETURN false), t_ :+ d_, d_ := 6 - d_), "
        "IF(n < 1062961, RETURN SOLVE(n > 1)), RANDOM(-n), LOOP(IF(k = 0, RETURN true), "
        "IF(NOT rabin_miller(n, RANDOM(n - 3) + 2), RETURN false), k :- 1))",
        ("pprime?(10^100 + 267)", "true"),
        (
            "[pprime?(9), pprime?(2), pprime?(25), pprime?(97)]",
            "[false, true, false, true]",
        ),
        "> rho(n, x := 3, y := 3, d_, i_ := 0) := PROG(LOOP(i_ :+ 1, "
        "x := MOD(x^2 + 1, n), y := MOD(MOD(y^2 + 1, n)^2 + 1, n), "
        "d_ := GCD(x - y, n), IF(d_ > 1, exit)), MIN(d_, n/d_))",
        ("rho(35)", "5"),
    )


async def test_card_shuffles():
    await run(
        "> Inshuffle(n, cds, newcds, oldcds, j, k) := PROG(cds := VECTOR(i, i, 1, n), "
        '"DISPLAY(cds)", k := 0, oldcds := cds, LOOP(IF(newcds = cds, RETURN(k)), '
        "j := 1, newcds := [], LOOP(IF(j > n/2, exit), newcds := APPEND(newcds, "
        'oldcds SUB [j, n/2 + j]), j :+ 1), "DISPLAY(newcds)", oldcds := newcds, '
        "k :+ 1))",
        ("Inshuffle(100)", "30"),
        "> Outshuffle_o(n, cds, newcds, oldcds, j, k) := PROG(cds := VECTOR(i, i, 1, n), "
        "k := 0, oldcds := cds, LOOP(IF(newcds = cds, RETURN k), j := 1, newcds := [], "
        "LOOP(IF(j > n/2, exit), newcds := APPEND(newcds, oldcds SUB [n/2 + j, j]), "
        "j :+ 1), oldcds := newcds, k :+ 1))",
        ("Outshuffle_o(52)", "52"),
    )


async def test_approximate_loops():
    session = await run(
        "> PrecisionDigits := 10",
        "> hank(n, k_ := 2, s_ := 0, t_ := 1, u_ := 1) := LOOP(IF(s_ = u_, "
        "RETURN(s_/n)), u_ := s_, s_ :+ t_*k_*(k_ - 1), t_ :* (n - k_ + 1)/n, k_ :+ 1)",
        "> agm(x, y, e, a, g, n, q) := PROG(IF(NOT NUMBER?(x) OR NOT NUMBER?(y), "
        'RETURN "invalid arguments"), q := [x, y], n := 1, '
        "LOOP(IF(ABS(q SUB 1 - q SUB 2) < 10^(-e), exit), "
        "a := AVERAGE(q SUB 1, q SUB 2), g := SQRT(q SUB 1*q SUB 2), q := [a, g], "
        "n :+ 1), [APPROX(a), n - 1])",
        ("agm(a, 7, 10)", '"invalid arguments"'),
        ("(agm(27, 7, 3)) SUB 2", "3"),
        ("(agm(12, 7, 10)) SUB 2", "4"),
        "> OdhadPi(n) := PROG(k := 5, Precision := Approximate, LOOP(d := k*SIN(pi/k), "
        "h := k*TAN(pi/k), IF(FLOOR(d*10^n) = FLOOR(h*10^n), "
        "PROG(RETURN APPROX([d, h], 10))), k := k + 1))",
        ("OdhadPi(2)", "[3.140002340, 3.144777633]"),
        ("Precision", "Approximate"),
        ("k", "57"),
    )
    assert (await session.approx("hank(1000)")).text == "40.30321292"


async def test_turtle_graphics():
    await run(
        "> new := PROG(xcor_ := 0, ycor_ := 0, heading_ := 0, trace_ := [[0, 0]], "
        "RETURN trace_)",
        "> fd(l) := PROG(xcor_ := xcor_ + l*COS(heading_*deg), "
        "ycor_ := ycor_ + l*SIN(heading_*deg), "
        "trace_ := APPEND(trace_, [[xcor_, ycor_]]), RETURN trace_)",
        "> lt(a) := PROG(heading_ := MOD(heading_ + a, 360), RETURN trace_)",
        "> rt(a) := lt(-a)",
        "> Block commands := commands SUB DIM(commands)",
        "> last(list) := list SUB DIM(list)",
        "> repeat(n, c) := last(VECTOR(c, i, 1, n))",
        "> koch(s, t) := PROG(IF(t = 0, RETURN fd(s)), koch(s/3, t - 1), lt(60), "
        "koch(s/3, t - 1), rt(120), koch(s/3, t - 1), lt(60), koch(s/3, t - 1))",
        ("new", "[[0, 0]]"),
        (
            "Block(fd(2), lt(120), fd(2), lt(120), fd(2))",
            "[[0, 0], [2, 0], [1, SQRT(3)], [0, 0]]",
        ),
        ("new", "[[0, 0]]"),
        ("koch(3, 1)", "[[0, 0], [1, 0], [3/2, SQRT(3)/2], [2, 0], [3, 0]]"),
        (
            "repeat(2, Block(fd(1), lt(90)))",
            "[[0, 0], [1, 0], [3/2, SQRT(3)/2], [2, 0], [3, 0], [4, 0]]",
        ),
    )


async def test_elliptic_curve_addition():
    await run(
        "> ecurveop(p, q, a, b, c, o := [inf, inf], r, s) := IF(p = o, q, IF(q = o, p, "
        "PROG(r := IF(p /= q, (q SUB 2 - p SUB 2)/(q SUB 1 - p SUB 1), "
        "(3*p SUB 1^2 + 2*a*p SUB 1 + b)/(2*p SUB 2)), IF(0 + 1/r = 0, o, "
        "PROG(s := r^2 - a - p SUB 1 - q SUB 1, [s, -(p SUB 2 + r*(s - p SUB 1))])))))",
        ("ecurveop([0, #i], [1, 0], 0, 0, -1)", "[-2, -3*#i]"),
        ("ecurveop([1, 0], [0, #i], 0, 0, -1)", "[-2, -3*#i]"),
        ("ecurveop([inf, inf], [1, 0], 0, 0, -1)", "[1, 0]"),
    )


async def test_a_lottery_draw_needs_a_new_number_at_each_pass():
    await run(
        "> LOTTO(n, k, POOL) := PROG(POOL := {1, ..., n}, LOOP(IF(DIM(POOL) = n - k, "
        "RETURN {1, ..., n} \\ POOL), POOL := POOL \\ {1 + RANDOM(n)}))",
        ("DIM(LOTTO(49, 6))", "6"),
        ("LOTTO(3, 3)", "{1, 2, 3}"),
    )


async def test_a_setting_saved_and_restored_by_a_program():
    await run(
        "> NULL_SPACE(M, R_, b_, t_, c_, k_, id_, i_ := 1, ns_ := []) := "
        "PROG(R_ := ROW_REDUCE(M), c_ := DIM(FIRST(M)), k_ := RANK(R_), "
        "IF(c_ = k_, RETURN []), b_ := VECTOR(POSITION(1, R_ SUB t_), t_, 1, k_), "
        "id_ := -IDENTITY_MATRIX(c_ - k_), LOOP(IF(i_ > c_, RETURN REVERSE(ns_)), "
        "IF(MEMBER?(i_, b_), PROG(ns_ := ADJOIN(DELETE(FIRST(R_), b_), ns_), "
        "R_ := REST(R_)), PROG(ns_ := ADJOIN(FIRST(id_), ns_), id_ := REST(id_))), "
        "i_ :+ 1))",
        "> EXACT_EIGENVECTOR(M, lambda, r_, c_, trig_) := "
        "PROG(c_ := M - lambda*IDENTITY_MATRIX(DIM(M)), r_ := NULL_SPACE(c_), "
        "IF(r_ = [], PROG(trig_ := Trigonometry, Trigonometry := Collect, "
        "r_ := NULL_SPACE(c_), Trigonometry := trig_)), r_)",
        ("NULL_SPACE([[1, 2, 3], [1, 2, 3], [1, 2, 3]])", "[[2, 3], [-1, 0], [0, -1]]"),
        ("NULL_SPACE([[1, 0], [0, 1]])", "[]"),
        "> A := [[-2, 1, 0, 0], [2, -9/5, 0, 0], [0, 1/5, 0, 0], [0, 3/5, 0, 0]]",
        ("EXACT_EIGENVECTOR(A, 0)", "[[0, 0], [0, 0], [-1, 0], [0, -1]]"),
        ("EXACT_EIGENVECTOR([[1, 0], [0, 1]], 5)", "[]"),
        ("Trigonometry", "Auto"),
    )


async def test_sums_of_three_squares_by_nested_loops():
    await run(
        "> S3C(n) := PROG(v := [], IF(MOD(n, 8) = 7, RETURN v), "
        "m := FLOOR(SQRT(n/3)), a := 0, LOOP(IF(a > m, RETURN v), b := a, "
        "LOOP(IF(b^2 > (n - a^2)/2, exit), c := SQRT(n - a^2 - b^2), "
        "IF(INTEGER?(c), v := APPEND(v, [[a, b, c]])), b :+ 1), a :+ 1))",
        ("S3C(101)", "[[0, 1, 10], [1, 6, 8], [2, 4, 9], [4, 6, 7]]"),
        ("S3C(7)", "[]"),
        # `v` is global and keeps the last run's value.
        ("v", "[]"),
        ("DIM(S3C(1001))", "10"),
    )


async def test_the_smallest_taylor_order_within_a_tolerance():
    """The test compares an exact combination of sines and cosines of integers with a
    rational, which is decided exactly."""
    await run(
        "> napprox(u, x, x0, x1, epsilon, n := 0) := LOOP(IF(ABS(SUBST(u - "
        "TAYLOR(u, x, x0, n), x, x1)) < epsilon, RETURN n), n :+ 1)",
        ("napprox(SIN(x), x, 1, 2, 1/1000)", "6"),
    )


async def test_the_extended_gcd_over_the_gaussian_integers():
    await run(
        "> EXTENDED_GCD(a, b, q_, r_) := PROG(a := [a, [1, 0]], b := [b, [0, 1]], "
        "LOOP(IF(FIRST(b) = 0, exit), q_ := FIRST(a)/FIRST(b), "
        "q_ := ROUND(RE(q_)) + ROUND(IM(q_))*#i, r_ := a - q_*b, a := b, b := r_), "
        "IF(FIRST(a) = 0, RETURN [0, [0, 0]]), LOOP(IF(0 <= PHASE(FIRST(a)) < pi/2, "
        "RETURN a), a :* #i))",
        ("EXTENDED_GCD(229, 541)", "[1, [189, -80]]"),
        ("EXTENDED_GCD(1234567, 234567)", "[1, [9640, -50737]]"),
        ("EXTENDED_GCD(17 + 13*#i, 14 + 11*#i)", "[1, [1 - 10*#i, -1 + 12*#i]]"),
        ("EXTENDED_GCD(0, 0)", "[0, [0, 0]]"),
        "> diophl(lst, xy, tmp, tmp1) := PROG(IF(MOD(lst SUB 3, GCD(lst SUB 2, "
        'lst SUB 1)) /= 0, RETURN "no solution"), tmp1 := EXTENDED_GCD(lst SUB 1, '
        "lst SUB 2), tmp := [lst SUB 2, -lst SUB 1]*k_/GCD(lst SUB 2, lst SUB 1) - "
        "lst SUB 3/GCD(lst SUB 2, lst SUB 1)*[tmp1 SUB 2 SUB 1, tmp1 SUB 2 SUB 2], "
        "RETURN [xy SUB 1 = tmp SUB 1 AND xy SUB 2 = tmp SUB 2])",
        ("diophl([12, 34, -6], [p, q])", "[p = 17*k_ + 9 AND q = -6*k_ - 3]"),
        ("diophl([12, 34, 5], [p, q])", '"no solution"'),
    )


async def test_a_primitive_root_from_the_value_a_loop_left():
    await run(
        "> EULER_PHI(n) := n*PRODUCT(1 - 1/v_ SUB 1, v_, FACTORS(n))",
        "> PRIME_POWER?(n, k_ := 1, t_ := 2) := LOOP(IF(t_ > n, RETURN false), "
        "IF(PRIME?(n^(1/k_)), exit), k_ :+ 1, t_ :* 2)",
        "> PRIMITIVE_ROOT(n, a_ := 2, n_, l_, p_) := IF(n < 5, n - 1, IF(MOD(n, 4) > 0, "
        "IF(PRIME_POWER?(n/(2 - MOD(n, 2))), PROG(n_ := EULER_PHI(n), "
        "l_ := (FACTORS(n_)) SUB SUB 1, LOOP(IF(GCD(a_, n) = 1, "
        "IF(EVERY(MOD(a_^(n_/p_), n) /= 1, p_, l_), RETURN a_)), a_ :+ 1)))))",
        ("PRIMITIVE_ROOT(101)", "2"),
        ("PRIMITIVE_ROOT(2551)", "6"),
        ("PRIMITIVE_ROOT(3397745833)", "5"),
        (
            "SELECT(PRIMITIVE_ROOT(p, 10) = 10, p, SELECT(PRIME(q_), q_, 1, 100))",
            "[7, 17, 19, 23, 29, 47, 59, 61, 97]",
        ),
        ("PRIMITIVE_ROOT(8)", "?"),
    )


async def test_integer_relations_by_pslq():
    """Element and slice assignment with vector indices, updates of subscripted
    targets, assignments in tests, `exit` in nested loops, a default read from a
    setting."""
    await run(
        "> pslq_aux(n, i, j0, h, b, y, j, k, t) := LOOP(j := MIN(i - 1, j0), "
        "LOOP(IF(0 /= t := ROUND(h SUB i SUB j, h SUB j SUB j), PROG(k := [1, ..., j], "
        "h SUB i SUB k := h SUB i SUB k - t*h SUB j SUB k, b SUB j :+ t*b SUB i, "
        "y SUB j :+ t*y SUB i)), IF(1 > j := j - 1, exit)), IF(n < i := i + 1, exit))",
        "> pslq(x, ds := 1, d := PrecisionDigits, n, h, b, y, i, j, m, s, t) := "
        "PROG(x := APPROX(x, d), IF(SOME(NOT(RATIONAL?(t_)), t_, x) OR EVERY(t_ = 0, "
        't_, x) OR 2 > n := DIM(x), RETURN("inapplicable")), s := APPROX(VECTOR('
        "SQRT(t_), t_, FIRST(ITERATE([ADJOIN(FIRST(t_) + x SUB i_^2, t_), i_ - 1], "
        "[t_, i_], [[x SUB n^2], n - 1], n - 1))), d), y := ROUND(10^d*x/s SUB 1), "
        "b := IDENTITY_MATRIX(n), IF(ABS(y SUB n) < 10^ds, RETURN([b SUB n, 0, 1])), "
        "h := VECTOR(VECTOR(IF(j_ < i_, ROUND(-10^d*x SUB i_*x SUB j_/(s SUB j_*"
        "s SUB (j_ + 1))), IF(j_ = i_, ROUND(10^d*s SUB (j_ + 1)/s SUB j_), 0)), j_, 1, "
        "n - 1), i_, 1, n), pslq_aux(n, 2, n - 1, h, b, y), LOOP(m := FIRST(ITERATE("
        "[IF(ABS(y SUB j_) > ABS(y SUB i_), i_, j_), i_ + 1], [j_, i_], [1, 2], n - 1)), "
        "IF(y SUB m^2 < 100^ds*b SUB m^2, RETURN([b SUB m, APPROX(ABS(y SUB m)/MAX("
        "VECTOR(ABS(t_), t_, y)), d), APPROX(10^d/MAX(VECTOR(ABS(h SUB i_ SUB i_), i_, "
        "1, n - 1)), d)])), m := FIRST(ITERATE([IF(h SUB j_ SUB j_^2 < (4/3)^(i_ - j_)*"
        "h SUB i_ SUB i_^2, i_, j_), i_ + 1], [j_, i_], [1, 2], n - 2)), "
        "j := [m, m + 1], "
        "i := REVERSE(j), h SUB i := h SUB j, b SUB i := b SUB j, y SUB i := y SUB j, "
        "IF(n - 1 > i := m, PROG(s := APPROX(SIGN(h SUB m SUB j), d), LOOP(t := h SUB i "
        "SUB j, h SUB i SUB j := ROUND([s*t, CROSS(s, t)]), IF(n < i := i + 1, exit)))), "
        "pslq_aux(n, m + 1, m + 1, h, b, y)))",
        ("pslq([0, 0])", '"inapplicable"'),
        ("pslq([1])", '"inapplicable"'),
        ("pslq([x, 1])", '"inapplicable"'),
        "> PrecisionDigits := 30",
        ("DIM(pslq([1, 2]))", "3"),
        ("ABS(FIRST(pslq([1, 2])) . [1, 2])", "0"),
        ("ABS(FIRST(pslq([1, 1, 1])) . [1, 1, 1])", "0"),
    )


async def test_a_permutation_as_cycles():
    await run(
        "> tocycles(p, c_, p_ := [], s_) := PROG(IF(VECTOR?(FIRST(p)), p := p SUB 2), "
        "s_ := {1, ..., DIM(p)}, LOOP(IF(s_ = {}, RETURN(REVERSE(p_))), "
        "c_ := ITERATES(p SUB k_, k_, MIN(s_)), IF(DIM(c_) > 2, "
        "p_ := ADJOIN(DELETE(c_, -1), p_)), s_ := s_ \\ MAP_LIST(c_ SUB k_, k_, "
        "{1, ..., DIM(c_)})))",
        ("tocycles([2, 4, 7, 1, 5, 6, 3])", "[[1, 2, 4], [3, 7]]"),
        ("tocycles([5, 2, 6, 1, 4, 3, 7])", "[[1, 5, 4], [3, 6]]"),
        ("tocycles([[1, 2, 3], [2, 3, 1]])", "[[1, 2, 3]]"),
    )


async def test_a_recursive_program_over_expression_structure():
    await run(
        "> POLY_DEGREE(u, x) := PROG(IF(u = 0, RETURN -1), IF(IDENTICAL?(u, x), "
        "RETURN 1), IF(SUM?(u), RETURN MAX(VECTOR(POLY_DEGREE(v_, x), v_, TERMS(u)))), "
        "IF(POWER?(u) OR PRODUCT?(u), SUM(POLY_DEGREE(v_ SUB 1, x)*v_ SUB 2, v_, "
        "FACTORS(u)), 0))",
        ("POLY_DEGREE(PRODUCT(1 + x^i, i, 1, 20), x)", "210"),
        ("POLY_DEGREE((3*x^3 + 2*x - 1)*(3*x - 3*x^2 + 2*x^4 - x^5/2), x)", "8"),
        (
            "[POLY_DEGREE(0, x), POLY_DEGREE(x, x), POLY_DEGREE(7, x), "
            "POLY_DEGREE(x^3 + 2*x, x)]",
            "[-1, 1, 0, 3]",
        ),
    )


# -- the quote: held values ---------------------------------------------------------
#
# A held value is shown as written, its locals filled in: a sum or product put in for a
# name joins the one around it, and a negative number is subtracted.


async def test_a_quote_holds_its_expression_as_written():
    await run(
        "> InputMode := Word",
        ("'(2 + 3)", "2 + 3"),
        ("TERMS('(x + x + x))", "[x, x, x]"),
        ("TERMS(x + x + x)", "[3*x]"),
        ("''(1 + 2)", "'(1 + 2)"),
        ("'(xq := 3)", "xq := 3"),
        ("xq", "xq"),
        ("1 + '(1/2)", "1/2 + 1"),
        ("'(1/2) - '(1/2)", "0"),
        "> gq := 5",
        ("'(gq + 1)", "gq + 1"),
        "> F2(a, b) := a + b",
        ("'F2(1, 2)", "F2(1, 2)"),
        ("EXPAND('F2(1, 2))", "3"),
    )


async def test_a_quote_fills_in_the_locals_in_scope():
    await run(
        "> InputMode := Word",
        "> BRUCH(a, b) := '(a/b)",
        ("BRUCH(15, 3)", "15/3"),
        ("BRUCH(9, BRUCH(15, 3))", "9/(15/3)"),
        "> KB(m, n) := IF(MOD(m, n) /= 0, FLOOR(m, n) + BRUCH(1, KB(n, MOD(m, n))), m/n)",
        ("KB(5, 3)", "1/(1/2 + 1) + 1"),
        ("KB(131, 100)", "1/(1/(1/(1/3 + 2) + 4) + 3) + 1"),
        "> SUMORDER(a, b) := '(a + b)",
        ("[SUMORDER(2, 3), SUMORDER(3, 2)]", "[2 + 3, 3 + 2]"),
        "> KBO(m, n) := IF(MOD(m, n) /= 0, SUMORDER(FLOOR(m, n), "
        "BRUCH(1, KBO(n, MOD(m, n)))), m/n)",
        ("KBO(131, 100)", "1 + 1/(3 + 1/(4 + 1/(2 + 1/3)))"),
        "> qg(n) := '(n + gq)",
        "> gq := 5",
        ("qg(2)", "2 + gq"),
        ("EXPAND(qg(2))", "7"),
        ("VECTOR('(k + k), k, 1, 3)", "[1 + 1, 2 + 2, 3 + 3]"),
        "> LinCom3C(a_, b_, c_) := '(lambda*a_ + mu*b_ + nu*c_)",
        "> PlaneC(a_, s_, u_, t_, v_) := [[x], [y], [z]] = '(a_ + s_*u_ + t_*v_)",
        "> [a := [[1], [2], [3]], b := [[2], [-1], [2]], c := [[1], [1], [-3]], "
        "d := [[-4], [3], [2]]]",
        (
            "LinCom3C(a, b, c)",
            "lambda*[[1], [2], [3]] + mu*[[2], [-1], [2]] + nu*[[1], [1], [-3]]",
        ),
        (
            "PlaneC(b, s, c, t, d)",
            "[[x], [y], [z]] = [[2], [-1], [2]] + s*[[1], [1], [-3]] + t*[[-4], [3], "
            "[2]]",
        ),
    )


async def test_a_held_value_accumulates_in_a_loop():
    await run(
        "> InputMode := Word",
        "> fee(list, s1_, res_) := PROG(res_ := FIRST(list), list := REST(list), "
        "LOOP(IF(list = [], RETURN res_), s1_ := FIRST(list), res_ := '(res_ + s1_), "
        "list := REST(list)))",
        ("fee([1, 2, 3, 4])", "1 + 2 + 3 + 4"),
        ("fee([1/(x - 1), -1/(x + 1)])", "1/(x - 1) - 1/(x + 1)"),
        ("FACTOR(fee([1/(x - 1), -1/(x + 1)]))", "2/((x + 1)*(x - 1))"),
    )


async def test_a_held_value_is_passed_on_as_it_is():
    await run(
        "> InputMode := Word",
        "> lv(u, w) := PROG(w := u, w)",
        ("lv('(2 + 3))", "2 + 3"),
        "> idq(u) := u",
        ("idq('(1 + 1))", "1 + 1"),
        "> hv(v) := VECTOR(v SUB i + 1, i, 1, 2)",
        ("hv(['(2*3), '(4*5)])", "[2*3 + 1, 4*5 + 1]"),
        ("SUM(['(2*3), 4])", "2*3 + 4"),
        ("'(1 + 1) = 2", "1 + 1 = 2"),
        "> li(n, z) := IF(n = 1, -LN(1 - z), INT(li(n - 1, t)/t, t, 0, z), 'li(n, z))",
        ("li(k, w)", "li(k, w)"),
        ("li(1, w)", "-LN(1 - w)"),
        "> heq(a, b, a_, b_, equ) := PROG(a_ := 10, b_ := -15, equ := a_*x + b_*y - "
        "a_*a - b_*b = 0, equ := LHS(equ) - RHS(equ), [['(equ = 0)]])",
        ("heq(1, -2)", "[[10*x - 15*y - 40 = 0]]"),
    )


async def test_built_ins_that_take_an_expression_apart_see_it_as_written():
    await run(
        "> InputMode := Word",
        ('STRING(\'(x + x))', '["+", x, x]'),
        ("NUMBER?('(2 + 3))", "false"),
        ("SUM?('(2 + 3))", "true"),
        ("[PRODUCT?('(2/3)), POWER?('(x^2)), VARIABLE?('x)]", "[true, true, true]"),
        ("LHS('(1 + 1 = 2))", "1 + 1"),
        ("NUMERATOR('(2/4))", "2"),
        ("DENOMINATOR('(2/4))", "4"),
    )


async def test_a_held_value_is_released_where_its_value_is_computed_with():
    session = await run(
        "> InputMode := Word",
        ("IF('(1 + 1) = 2, \"yes\", \"no\")", '"yes"'),
        ("SUBST('(x + x), x, 2)", "4"),
        ("DIF('(x^2 + x^2), x)", "4*x"),
        ("SQRT('(2 + 2))", "2"),
        ("SELECT(k > '(1 + 1), k, [1, 2, 3, 4])", "[3, 4]"),
        "> sv(u) := PROG(gh := u, 0)",
        ("sv('(2 + 3))", "0"),
        ("gh", "5"),
        "> xq2 := '(1 + 1)",
        ("xq2", "1 + 1"),
        ("xq3 := '(2 + 2)", "xq3 := 2 + 2"),
        ("xq3", "4"),
        "> RECORD2(u) := PROG(o2 := [], eq2 := u, e2 := u)",
        (
            "RECORD2('((2*x - 3)/15 - (4*x - 9)/20 = (8*x - 27)/30 - (16*x - 81)/24 - "
            "9/40))",
            "e2 := (2*x - 3)/15 - (4*x - 9)/20 = (8*x - 27)/30 - (16*x - 81)/24 - 9/40",
        ),
        ("e2", "1/4 - x/15 = 9/4 - 2*x/5"),
    )
    assert (await session.approx("'(1/4 + 1/4)")).text == "0.5"


async def test_a_definition_with_a_quote_is_shown_with_it():
    await run(
        "> InputMode := Word",
        "> BR2(a, b) := '(a/b)",
        ("BR2(a, b) := '(a/b)", "BR2(a, b) := '(a/b)"),
    )


async def test_display_shows_a_held_value_as_written():
    session = await run(
        "> InputMode := Word",
        "> chq(b, c, s_) := PROG(s_ := '(s^2 + b*s + c = 0), DISPLAY(s_), b^2 - 4*c)",
    )
    await displays(session, "chq(0, -4)", "16", ["s^2 + 0*s - 4 = 0"])
    await displays(session, "chq(-6, 9)", "0", ["s^2 - 6*s + 9 = 0"])


async def test_the_euclidean_algorithm_as_a_table_of_equations():
    await run(
        "> InputMode := Word",
        "> ea(a, b, q_, r_, s_ := []) := LOOP(IF(b = 0, RETURN(s_)), [q_ := FLOOR(a, b), "
        "r_ := a - q_*b], IF(r_ > 0, s_ := INSERT('[a = q_*b + r_], s_, 0), "
        "s_ := INSERT('[a = q_*b], s_, 0)), [a := b, b := r_])",
        (
            "ea(8991, 3293)",
            "[[8991 = 2*3293 + 2405], [3293 = 1*2405 + 888], [2405 = 2*888 + 629], "
            "[888 = 1*629 + 259], [629 = 2*259 + 111], [259 = 2*111 + 37], "
            "[111 = 3*37]]",
        ),
        ("DIM(ea(89, 55))", "9"),
        "> EA(a, b, q_, r_, s_ := []) := PROG(LOOP(IF(b = 0, RETURN(REVERSE(s_))), "
        "q_ := a/b, q_ := ROUND(RE(q_)) + ROUND(IM(q_))*#i, r_ := a - q_*b, "
        "IF(r_ /= 0, s_ := ADJOIN([a = '(q_*b) + r_], s_), "
        "s_ := ADJOIN([a = '(q_*b)], s_)), a := b, b := r_))",
        (
            "EA(17 + 13*#i, 14 + 11*#i)",
            "[[17 + 13*#i = 1*(14 + 11*#i) + 3 + 2*#i], [14 + 11*#i = 5*(3 + 2*#i) - 1 "
            "+ #i], [3 + 2*#i = (-2*#i)*(-1 + #i) + 1], [-1 + #i = (-1 + #i)*1]]",
        ),
    )


async def test_turtle_graphics_with_a_held_block():
    await run(
        "> new := PROG(xcor_ := 0, ycor_ := 0, heading_ := 0, trace_ := [[0, 0]], "
        "RETURN trace_)",
        "> fd(l) := PROG(xcor_ := xcor_ + l*COS(heading_*deg), "
        "ycor_ := ycor_ + l*SIN(heading_*deg), "
        "trace_ := APPEND(trace_, [[xcor_, ycor_]]), RETURN trace_)",
        "> lt(a) := PROG(heading_ := MOD(heading_ + a, 360), RETURN trace_)",
        "> rt(a) := lt(-a)",
        "> Block commands := commands SUB DIM(commands)",
        "> last(list) := list SUB DIM(list)",
        "> repeat(n, c) := last(VECTOR(c, i, 1, n))",
        "> koch(s, t) := PROG(IF(t = 0, RETURN fd(s)), koch(s/3, t - 1), lt(60), "
        "koch(s/3, t - 1), rt(120), koch(s/3, t - 1), lt(60), koch(s/3, t - 1))",
        "> kochstar(s, t) := repeat(3, 'Block(koch(s, t), rt(120)))",
        ("new", "[[0, 0]]"),
        # The held body stays held at each pass of `VECTOR`: no turtle moves.
        ("repeat(4, 'Block(fd(2), lt(90)))", "Block(fd(2), lt(90))"),
        ("trace_", "[[0, 0]]"),
        ("kochstar(1, 3)", "Block(koch(1, 3), rt(120))"),
    )


async def test_solving_an_equation_step_by_step():
    await run(
        "> InputMode := Word",
        "> RECORD(u) := PROG(o := [], @1 := LHS(u), @2 := RHS(u), l := [@1], r := [@2], "
        "u)",
        "> do(u, t := 0, t_) := PROG(t_ := (NAME_TO_CODES(t)) SUB 1, IF(t_ = 101, "
        'PROG(@1 := EXPAND(@1, "Radical"), @2 := EXPAND(@2, "Radical"), o := '
        'ADJOIN(["expand(@)"], o))), IF(t_ = 48, PROG(@1 := SUBST(u, @, @1), '
        "@2 := SUBST(u, @, @2), o := ADJOIN([u], o))), l := ADJOIN(@1, l), "
        'r := ADJOIN(@2, r), [["", "v", ""], [@1, "=", @2]])',
        "> UNDO(u) := PROG(o := REST(o), l := REST(l), r := REST(r), @1 := FIRST(l), "
        '@2 := FIRST(r), [["", "v", ""], [@1, "=", @2]])',
        '> play := [REVERSE(l), VECTOR("=", k_, DIM(l)), REVERSE(r), '
        'APPEND(REVERSE(o), [""])]`',
        ("RECORD('(3^(2*x + 1) - 7 = 20))", "3^(2*x + 1) - 7 = 20"),
        ("do(@ + 7)", '[["", "v", ""], [3^(2*x + 1), "=", 27]]'),
        ("do(LOG(@, 3))", '[["", "v", ""], [2*x + 1, "=", 3]]'),
        ("do(@/2)", '[["", "v", ""], [x + 1/2, "=", 3/2]]'),
        ("UNDO()", '[["", "v", ""], [2*x + 1, "=", 3]]'),
        ("do(@ - 1)", '[["", "v", ""], [2*x, "=", 2]]'),
        ("do(@/2)", '[["", "v", ""], [x, "=", 1]]'),
        (
            "play",
            '[[3^(2*x + 1) - 7, "=", 20, [@ + 7]], [3^(2*x + 1), "=", 27, '
            '[LN(@)/LN(3)]], [2*x + 1, "=", 3, [@ - 1]], [2*x, "=", 2, [@/2]], '
            '[x, "=", 1, ""]]',
        ),
        (
            "RECORD('(2*(x - 1) + x/3 = 4*x - 9 + (x - 2)/3))",
            "2*(x - 1) + x/3 = 4*x - 9 + (x - 2)/3",
        ),
        ('do(@, "e")', '[["", "v", ""], [7*x/3 - 2, "=", 13*x/3 - 29/3]]'),
        ("do(@*3)", '[["", "v", ""], [7*x - 6, "=", 13*x - 29]]'),
        ("do(@ - 7*x + 29)", '[["", "v", ""], [23, "=", 6*x]]'),
    )


async def test_completing_the_square_step_by_step():
    session = await run(
        "> InputMode := Word",
        "> [steps :=, stn :=]",
        "> transform(q, v, c2, c1, c0, c4, c5) := PROG(v := (VARIABLES(q)) SUB 1, "
        "c2 := DIF(q, v, 2)/2, c1 := DIF(q, v), c1 := SUBST(c1, v, 0), "
        "c0 := SUBST(q, v, 0), c4 := c1/(2*c2), c5 := c4^2, "
        "steps := '[['(c2*(v^2 + c1*v/c2) + c0)], "
        "['(c2*(v^2 + c1/c2*v + c4^2) + c0 - c2*(c1/(2*c2))^2)], "
        "['(c2*(v + c4)^2) + '(c0 - c2*c5)], ['(c2*(v + c4)^2) + c0 - "
        'c2*(c1/(2*c2))^2], ["fertig - ready!"]], stn := 0, DISPLAY("Simplify step '
        'by step"))',
        "> step_(dummy) := PROG(stn := stn + 1, steps SUB stn SUB 1)",
        "> step := step_()",
    )
    shown = ["Simplify step by step"]
    await displays(session, "transform(11*x^2 - 4*x - 13)", f'"{shown[0]}"', shown)
    # The global holds the inner quotes; reading it releases them.
    held = write_expression(session.assignments["steps"].children[0])
    assert held == "['(11*(x^2 - 4*x/11) - 13)]"
    assert (await session.simplify("steps SUB 1")).text == "[11*(x^2 - 4*x/11) - 13]"
    for expected in [
        "11*(x^2 - 4*x/11) - 13",
        "11*(x^2 - 4/11*x + (-2/11)^2) - 13 - 11*(-4/(2*11))^2",
        "11*(x - 2/11)^2 - 13 - 11*4/121",
        "11*(x - 2/11)^2 - 147/11",
        '"fertig - ready!"',
    ]:
        assert (await session.simplify("step")).text == expected
    await displays(session, "transform(u^2 + 19*u + 18)", f'"{shown[0]}"', shown)
    for expected in [
        "1*(u^2 + 19*u/1) + 18",
        "1*(u^2 + 19/1*u + (19/2)^2) + 18 - 1*(19/(2*1))^2",
        "1*(u + 19/2)^2 + 18 - 1*361/4",
        "1*(u + 19/2)^2 - 289/4",
    ]:
        assert (await session.simplify("step")).text == expected


async def test_random_equations_through_a_given_point():
    await run(
        "> InputMode := Word",
        "> sys_6(a, b, k, n, dummy, a_, b_, aux, list, i, equ) := PROG(dummy := "
        "RANDOM(0), list := [], i := 1, LOOP(a_ := (RANDOM(k) + 1)*(-1)^RANDOM(2), "
        "b_ := (RANDOM(k) + 1)*(-1)^RANDOM(2), equ := a_*x + b_*y - (a_*a + b_*b) = 0, "
        "equ := LHS(equ) - RHS(equ), list := APPEND([['(equ = 0)]], list), "
        "IF(i = n, RETURN(list)), i :+ 1))",
        ("DIM(sys_6(1, -2, 20, 5))", "5"),
        (
            "SELECT(SUBST(LHS(e SUB 1), [x, y], [1, -2]) /= 0, e, sys_6(1, -2, 20, 5))",
            "[]",
        ),
        (
            "SELECT(SUBST(LHS(e SUB 1), [x, y], [11, -15]) /= 0, e, "
            "sys_6(11, -15, 50, 5))",
            "[]",
        ),
    )


async def test_a_string_names_the_variable_it_spells():
    # TaylorODE names the derivatives of its unknown `"y0"`, `"y1"`, ... and iterates,
    # differentiates and takes gradients over them.
    await run(
        (
            'ITERATES([x + 1, 2*"y0"], [x, "y0"], [0, 1], 3)',
            "[[0, 1], [1, 2], [2, 4], [3, 8]]",
        ),
        ('GRAD("y0"^2*"y1", ["y0", "y1"])', '[2*"y0"*"y1", "y0"^2]'),
        "> GG(q) := PROG(lv := \"y4\", DIF(\"y4\"^2, lv))",
        ("GG(1)", '2*"y4"'),
        "> HH(w) := PROG(w := [x, \"y0\"], ITERATES([x + 1, 2*\"y0\"], w, [0, 1], 2))",
        ("HH(1)", "[[0, 1], [1, 2], [2, 4]]"),
    )


async def test_a_subscript_is_evaluated_before_what_it_selects_from():
    # The index assigns the list it reads, so the element is the one it appended, as
    # titbit15.mth's DEF relies on.
    await run(
        "> NEWL(u) := (l := u)",
        "> l := [a]",
        ("l SUB DIM(NEWL(APPEND(l, [b])))", "b"),
    )


async def test_nsolve_evaluates_a_program_at_each_point():
    # Over a variable the loop's test is undecided and the call stands; a numeric
    # search runs it at each point instead.
    await run(
        "> sq(a, y) := PROG(y := a, LOOP(IF(ABS(y^2 - a) < 10^(-15), RETURN y), "
        "y := (y + a/y)/2))",
        ("sq(x)", "sq(x)"),
        ("NSOLVE(sq(x) = 3, x, 1, 20)", "x = 9"),
        ("NSOLUTIONS(sq(x) = 3, x, 1, 20)", "[9]"),
        ("NSOLVE(sq(x) = -3, x, 1, 20)", "NSOLVE(sq(x) = -3, x, 1, 20)"),
    )
