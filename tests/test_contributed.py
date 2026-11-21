"""Packages users contributed to Derive 6, loaded from their files.

They are not part of the pre-defined library: a session knows none of their functions
until the user loads the file, with Transfer Load Utility or `LOAD`. Each test writes
the package's definitions to a file and loads it, then checks the package's examples
against what Derive 6 computed for them.
"""

from __future__ import annotations

import pytest
from test_native import stored_twice

from rederive.model.session import Session


@pytest.fixture(autouse=True)
def here(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def package(path, definitions):
    """A math file holding `definitions`, one per line, as Derive writes one."""
    path.write_text("\r\n".join(definitions) + "\r\n", encoding="utf-8")
    return path


async def answers(session, *lines):
    """What Simplify answers for each of `lines` in turn; a line starting with `>` is
    entered, not simplified."""
    found = []
    for line in lines:
        if line.startswith("> "):
            session.author(line.removeprefix("> "))
        else:
            found.append((await session.simplify(line)).text)
    return found


def loaded(tmp_path, name, definitions):
    """A new session with the package loaded as a utility file."""
    session = Session()
    assert not session.load_utility(package(tmp_path / f"{name}.mth", definitions))
    return session


#: The definitions of LaplaceTransforms.
LAPLACE = [
    (
        "TREE(expn, expnstr):=PROG(expnstr := STRING(expn), IF(VECTOR?(expnstr), "
        "ADJOIN(FIRST(expnstr), VECTOR(TREE(u), u, REST(expnstr))), expn))"
    ),
    "ClosedFormIntegral?(f):=PROG(f := TREE(f), ¬ MEMBER?(\"∫\", VARIABLES(f)))",
    "Contains?(u, z):=MEMBER?(z, VARIABLES(u))",
    (
        "Division_by_t?(f, t, facts, t_test):=PROG(facts := FACTORS(f, t), "
        "t_test := SELECT(FIRST(v) = t, v, facts), IF(DIM(t_test) ≠ 1, "
        "RETURN [false]), IF((FIRST(t_test))↓2 = -1, RETURN [true, t·f], [false]))"
    ),
    (
        "Linear?(f, x, m, c):=PROG(IF(¬ Contains?(f, x), RETURN [true, 0, f]), "
        "m := ∂(f, x), IF(Contains?(m, x), RETURN [false]), [true, m, f - m·x])"
    ),
    (
        "LinearCosine?(f, x, xoccurs, size, str, consts, "
        "linear):=PROG(f := FACTORS(f), xoccurs := SELECT(Contains?(u, x), u, f), "
        "size := DIM(xoccurs), IF(size = 0 ∨ size > 1, RETURN [false]), "
        "xoccurs := FIRST(xoccurs), IF(xoccurs↓2 ≠ 1, RETURN [false]), "
        "str := STRING(FIRST(xoccurs)), IF(¬ NAME_TO_CODES(FIRST(str)) = [67, 79, 83], "
        "RETURN [false]), linear := Linear?(str↓2, x), IF(¬ FIRST(linear), "
        "RETURN [false]), consts := SELECT(¬ Contains?(u, x), u, f), [true, ∏(v↓1^v↓2, "
        "v, consts), linear↓2, linear↓3])"
    ),
    "Natural?(n):=PROG(IF(n ≥ 0 ∧ INTEGER?(n), RETURN true), false)",
    (
        "Simple_Exponential?(f, x, str):=PROG(str := STRING(f), "
        "IF((FIRST(str) = \"^\" ∧ ¬ Contains?(str↓2, x)) ∧ ¬ Contains?(str↓3/x), "
        "RETURN [true, str↓2^(str↓3/x)]), [false])"
    ),
    (
        "ExponPolyCosine?(f, x, store, consts, base, simpexpon, lincosine, index, m, "
        "c, n):=PROG(IF(¬ Contains?(f, x), RETURN [true, f, 1, 0, 0, 0]), "
        "store := VariableOrder, VariableOrder := [x], f := FACTORS(f), "
        "consts := ∏(v↓1^v↓2, v, SELECT(¬ Contains?(u, x), u, f)), "
        "f := SELECT(Contains?(u, x), u, f), [base := 1, m := 0, c := 0, n := 0], "
        "LOOP(simpexpon := Simple_Exponential?(f↓1↓1, x), IF(FIRST(simpexpon), "
        "PROG((base :* simpexpon↓2^f↓1↓2) ∧ f := DELETE(f, 1), IF(DIM(f) = 0, "
        "(VariableOrder := store) ∧ RETURN [true, consts, base, n, m, c])), exit)), "
        "lincosine := LinearCosine?(FIRST(FIRST(f)), x), IF(FIRST(lincosine), "
        "PROG(index := (FIRST(f))↓2, IF(index ≠ 1 ∧ Natural?(index), "
        "(VariableOrder := store) ∧ RETURN [false, true]), "
        "((m := lincosine↓3) ∧ c := lincosine↓4) ∧ f := DELETE(f, 1)), "
        "IF(FIRST(FIRST(f)) ≠ x, (VariableOrder := store) ∧ RETURN [false], true, "
        "(VariableOrder := store) ∧ RETURN [false])), IF(DIM(f) > 1, "
        "(VariableOrder := store) ∧ RETURN [false]), IF(FIRST(FIRST(f)) = x, "
        "n := f↓1↓2), (VariableOrder := store) ∧ RETURN [true, consts, base, n, m, c])"
    ),
    (
        "LinearSine?(f, x, xoccurs, size, str, consts, linear):=PROG(f := FACTORS(f), "
        "xoccurs := SELECT(Contains?(u, x), u, f), size := DIM(xoccurs), "
        "IF(size = 0 ∨ size > 1, RETURN [false]), xoccurs := FIRST(xoccurs), "
        "IF(xoccurs↓2 ≠ 1, RETURN [false]), str := STRING(FIRST(xoccurs)), "
        "IF(¬ NAME_TO_CODES(FIRST(str)) = [83, 73, 78], RETURN [false]), "
        "linear := Linear?(str↓2, x), IF(¬ FIRST(linear), RETURN [false]), "
        "consts := SELECT(¬ Contains?(u, x), u, f), [true, ∏(v↓1^v↓2, v, consts), "
        "linear↓2, linear↓3])"
    ),
    (
        "ExponPolySine?(f, x, store, xoccurs, consts, base, simpexpon, linsine, index, "
        "m, c, n):=PROG(IF(¬ Contains?(f, x), RETURN [false]), store := VariableOrder, "
        "VariableOrder := [x], f := FACTORS(f), consts := ∏(v↓1^v↓2, v, "
        "SELECT(¬ Contains?(u, x), u, f)), f := SELECT(Contains?(u, x), u, f), "
        "[base := 1, m := 0, c := 0, n := 0], "
        "LOOP(simpexpon := Simple_Exponential?(f↓1↓1, x), IF(FIRST(simpexpon), "
        "PROG((base :* simpexpon↓2^f↓1↓2) ∧ f := DELETE(f, 1), IF(DIM(f) = 0, "
        "(VariableOrder := store) ∧ RETURN [true, consts, base, n, m, c])), exit)), "
        "linsine := LinearSine?(FIRST(FIRST(f)), x), IF(FIRST(linsine), "
        "PROG(index := (FIRST(f))↓2, IF(index ≠ 1 ∧ Natural?(index), "
        "(VariableOrder := store) ∧ RETURN [false, true]), "
        "((m := linsine↓3) ∧ c := linsine↓4) ∧ f := DELETE(f, 1)), "
        "IF(FIRST(FIRST(f)) ≠ x, (VariableOrder := store) ∧ RETURN [false], true, "
        "(VariableOrder := store) ∧ RETURN [false])), IF(DIM(f) > 1, "
        "(VariableOrder := store) ∧ RETURN [false]), IF(FIRST(FIRST(f)) = x, "
        "n := f↓1↓2), (VariableOrder := store) ∧ RETURN [true, consts, base, n, m, c])"
    ),
    (
        "FlattenTree(v, list := []):=LOOP(IF(DIM(v) = 0, RETURN list), "
        "IF(VECTOR?(FIRST(v)), PROG(v := FIRST(v), list := ADJOIN(FIRST(v), list), "
        "v := APPEND(REST(v))), PROG(list := ADJOIN(FIRST(v), list), "
        "v := APPEND(REST(v)))))"
    ),
    (
        "QUOTDIF(vect, x, n):=ITERATE([v·∂(u, x) - u·∂(v, x), v^2], [u, v], "
        "[FIRST(vect), FIRST(REST(vect))], n)"
    ),
    (
        "InvLaplaceRatPol(f, s, t, facts, numer, denom, poles, polynumdegree, "
        "polydenomdegree, orders, order, counter, size, resdenom, residue, residues, "
        "sol, constterm, test, constfacts, varfacts, tmpfacts, "
        "tmporders):=PROG(numer := NUMERATOR(f), denom := DENOMINATOR(f), "
        "facts := FACTORS(denom, Complex, s), constfacts := [], varfacts := [], "
        "counter := 1, size := DIM(facts), LOOP(IF(counter > size, exit), "
        "test := facts↓counter, IF(Contains?(test, s), varfacts := ADJOIN(test, "
        "varfacts), constfacts := ADJOIN(test, constfacts)), counter :+ 1), "
        "IF(DIM(constfacts) = 0, constterm := 1, constterm := ∏(VECTOR(v↓1^v↓2, v, "
        "constfacts))), orders := varfacts COL 2, facts := varfacts COL 1, "
        "poles := [], size := DIM(facts), counter := 1, LOOP(IF(counter > size, exit), "
        "sol := facts↓counter, poles := ADJOIN(RHS(SOLVE(sol, s)), poles), "
        "counter :+ 1), poles := REVERSE(poles), residues := [], counter := 1, "
        "size := DIM(poles), LOOP(IF(counter > size, exit), tmpfacts := DELETE(facts, "
        "counter), tmporders := DELETE(orders, counter), "
        "resdenom := ∏(tmpfacts↓i^tmporders↓i, i, 1, size - 1), "
        "order := orders↓counter, residue := LIM(QUOTDIF([#e^(s·t)·numer, resdenom], "
        "s, order - 1), s, poles↓counter), "
        "residue := FIRST(residue)/FIRST(REST(residue))·(1/(order - 1)!), "
        "residues := ADJOIN(residue, residues), counter :+ 1), ∑(residues)/constterm)"
    ),
    (
        "InvLaplaceProdExp(nonexpterm, shift, s, "
        "t):=STEP(t + shift)·SUBST(InvLaplaceRatPol(nonexpterm, s, t), t, t + shift)"
    ),
    (
        "LinearExp?(f, x, str):=PROG(str := STRING(f), IF(DIM(str) ≠ 3, RETURN false), "
        "IF((FIRST(str) = \"^\" ∧ str↓2 = #e) ∧ ¬ Contains?(str↓3/x, x), RETURN true), "
        "false)"
    ),
    (
        "ProductofLinExp?(f, x, facts, expterms, "
        "nonexpterms):=PROG(facts := FACTORS(f, x), expterms := SELECT(LinearExp?(v↓1, "
        "x), v, facts), IF(DIM(expterms) ≠ 1, RETURN [false]), "
        "nonexpterms := SELECT(¬ LinearExp?(v↓1, x), v, facts), [true, "
        "LN(FIRST(FIRST(expterms)))/x·FIRST(REST(FIRST(expterms))), ∏(v↓1^v↓2, v, "
        "nonexpterms)])"
    ),
    (
        "Power?(f, x, expr, xexprs):=PROG(IF(¬ Contains?(f, x), RETURN [true, 0]), "
        "IF(¬ Contains?(f/x, x), RETURN [true, 1]), expr := STRING(f), "
        "IF(FIRST(expr) = \"-\", expr := STRING(FIRST(REST(expr)))), "
        "IF(FIRST(expr) = \"^\", IF(IDENTICAL?(FIRST(REST(expr)), "
        "x) ∧ Natural?(FIRST(REST(REST(expr)))), RETURN [true, "
        "FIRST(REST(REST(expr)))])), IF(FIRST(expr) = \"·\", "
        "PROG(xexprs := SELECT(Contains?(v, x), v, REST(expr)), IF(DIM(xexprs) ≠ 1, "
        "RETURN [false, -1]), expr := STRING(FIRST(xexprs)), IF(FIRST(expr) = \"^\", "
        "IF(IDENTICAL?(FIRST(REST(expr)), x) ∧ Natural?(FIRST(REST(REST(expr)))), "
        "RETURN [true, FIRST(REST(REST(expr)))]), RETURN [false, -1]))), [false, -1])"
    ),
    (
        "Polynomial?(f, x, terms, size, test, sum, tests, "
        "counter):=PROG(terms := TERMS(EXPAND(f, x), x), size := DIM(terms), "
        "counter := 1, sum := 0, tests := [], LOOP(IF(counter > size, exit), "
        "test := Power?(terms↓counter, x), IF(FIRST(test), "
        "(sum :+ 1) ∧ tests := ADJOIN(test, tests)), counter :+ 1), IF(sum = size, "
        "RETURN [true, MAX(tests COL 2)]), [false])"
    ),
    (
        "RationalPolynomial?(f, x, numer, "
        "denom):=PROG(numer := Polynomial?(NUMERATOR(f), x), "
        "denom := Polynomial?(DENOMINATOR(f), x), IF(¬ FIRST(numer) ∨ ¬ FIRST(denom), "
        "RETURN false), IF(numer↓2 ≥ denom↓2, RETURN false), true)"
    ),
    (
        "InvLaplace(f, s, t, terms, counter, size, term, inverses, "
        "prodexp):=PROG(terms := TERMS(EXPAND(f, s)), counter := 1, "
        "size := DIM(terms), inverses := [], LOOP(IF(counter > size, exit), "
        "term := terms↓counter, PROG(IF(RationalPolynomial?(term, s), "
        "(inverses := ADJOIN(InvLaplaceRatPol(term, s, t), inverses)) ∧ exit), "
        "prodexp := ProductofLinExp?(term, s), "
        "IF(FIRST(prodexp) ∧ RationalPolynomial?(prodexp↓3, s), "
        "inverses := ADJOIN(InvLaplaceProdExp(prodexp↓3, prodexp↓2, s, t), inverses), "
        "RETURN 'InvLaplace(f, s, t))), counter :+ 1), ∑(inverses))"
    ),
    (
        "LaplaceExponPolyCosine(s, a, b, n, m, "
        "c):=a·SUBST((-1)^n·∂((s·COS(c) - m·SIN(c))/(m^2 + s^2), s, n), s, s - LN(b))"
    ),
    (
        "LaplaceExponPolySine(s, a, b, n, m, "
        "c):=a·SUBST((-1)^n·∂((m·COS(c) + s·SIN(c))/(m^2 + s^2), s, n), s, s - LN(b))"
    ),
    (
        "Power_of_t?(f, t, facts, index, t_term, facts_t, liear, "
        "shift):=PROG(facts := FACTORS(f), t_term := SELECT(FIRST(Linear?(facts↓r↓1, "
        "t)) ∧ Contains?(facts↓r, t), r, DIM(facts)), IF(DIM(t_term) ≠ 1, "
        "RETURN [false]), facts_t := DELETE(facts, FIRST(t_term)), "
        "IF(Contains?(facts_t, t), RETURN [false]), t_term := facts↓FIRST(t_term), "
        "index := t_term↓2, linear := Linear?(t_term, t), shift := linear↓3/linear↓2, "
        "IF(Natural?(-index), RETURN [false]), [true, ∏(v↓1^v↓2, v, facts_t), index, "
        "shift])"
    ),
    (
        "LinearSign?(f, x, str, expr, consts, xoccurs, size, "
        "linear):=PROG(f := FACTORS(f), xoccurs := SELECT(Contains?(u, x), u, f), "
        "size := DIM(xoccurs), IF(size = 0 ∨ size > 1, RETURN [false]), "
        "xoccurs := FIRST(xoccurs), IF(xoccurs↓2 ≠ 1, RETURN [false]), "
        "str := STRING(FIRST(xoccurs)), IF(¬ NAME_TO_CODES(FIRST(str)) = [83, 73, 71, "
        "78], RETURN [false]), linear := Linear?(str↓2, x), IF(¬ FIRST(linear), "
        "RETURN [false]), consts := SELECT(¬ Contains?(u, x), u, f), [true, ∏(v↓1^v↓2, "
        "v, consts), linear↓2, linear↓3])"
    ),
    (
        "ProductofSign?(f, x, facts, signterm, nonsignterms, counter, size, "
        "linsign?):=PROG(f := FACTORS(f), signterm := [], nonsignterms := [], "
        "counter := 1, size := DIM(f), LOOP(IF(counter > size, exit), "
        "IF(FIRST(LinearSign?(f↓counter↓1, x)), signterm := ADJOIN(f↓counter, "
        "signterm), nonsignterms := ADJOIN(f↓counter, nonsignterms)), counter :+ 1), "
        "IF(DIM(signterm) ≠ 1, RETURN [false]), "
        "linsign? := LinearSign?(FIRST(FIRST(signterm)), x), IF(¬ FIRST(linsign?), "
        "RETURN [false]), IF((FIRST(signterm))↓2 = 1, RETURN [true, linsign?↓3, "
        "linsign?↓4, ∏(v↓1^v↓2, v, nonsignterms)]), [false])"
    ),
    (
        "Laplace(f, t, s, divbyt, varordstore, terms, term, size, counter, trigstore, "
        "poweroft, expolycos, expolysin, termplus, divbyt, lapdivbyt, lap, transforms, "
        "testsign, signflag, signscale):=PROG(trigstore := Trigonometry, "
        "varordstore := VariableOrder, VariableOrder := [t], "
        "divbyt := Division_by_t?(f, t), IF(FIRST(divbyt), "
        "PROG(lapdivbyt := LapDivByt(divbyt↓2, t, s), IF(FIRST(lapdivbyt), "
        "RETURN lapdivbyt↓2, "
        "PROG(WRITE(\"Sorry: Unable to find this particular Laplace transform\"), "
        "RETURN false)))), terms := TERMS(EXPAND(f, t)), size := DIM(terms), "
        "counter := 1, transforms := [], LOOP(IF(counter > size, exit), "
        "term := terms↓counter, signflag := false, testsign := ProductofSign?(term, "
        "t), IF(FIRST(testsign), PROG(signflag := true, "
        "signscale := testsign↓3/testsign↓2, IF(NUMBER?(signscale) ∧ signscale ≥ 0, "
        "((signflag := false) ∧ term := testsign↓4) ∧ exit), "
        "termplus := SUBST(testsign↓4, t, t - signscale), "
        "transforms := ADJOIN(2·#e^(s·signscale)·Laplace(termplus, t, "
        "s) - Laplace(testsign↓4, t, s), transforms))), PROG(IF(signflag, exit), "
        "poweroft := Power_of_t?(term, t), IF(FIRST(poweroft), "
        "(transforms := ADJOIN(#e^(s·poweroft↓4)·poweroft↓2·GAMMA(poweroft↓3 + 1)/s^(po"
        "weroft↓3 + 1), transforms)) ∧ exit), expolycos := ExponPolyCosine?(term, t), "
        "IF(¬ FIRST(expolycos) ∧ expolycos↓2, "
        "(((Trigonometry := Collect) ∧ transforms := ADJOIN(Laplace(term, t, s), "
        "transforms)) ∧ Trigonometry := trigstore) ∧ exit), "
        "IF(FIRST(expolycos) ∧ Natural?(expolycos↓4), "
        "(transforms := ADJOIN(LaplaceExponPolyCosine(s, expolycos↓2, expolycos↓3, "
        "expolycos↓4, expolycos↓5, expolycos↓6), transforms)) ∧ exit), "
        "expolysin := ExponPolySine?(term, t), IF(¬ FIRST(expolysin) ∧ expolysin↓2, "
        "(((Trigonometry := Collect) ∧ transforms := ADJOIN(Laplace(term, t, s), "
        "transforms)) ∧ Trigonometry := trigstore) ∧ exit), "
        "IF(FIRST(expolysin) ∧ Natural?(expolysin↓4), "
        "(transforms := ADJOIN(LaplaceExponPolySine(s, expolysin↓2, expolysin↓3, "
        "expolysin↓4, expolysin↓5, expolysin↓6), transforms)) ∧ exit), "
        "divbyt := Division_by_t?(term, t), IF(FIRST(divbyt), "
        "PROG(lapdivt := LapDivByt(FIRST(REST(divbyt)), t, s), IF(FIRST(lap·DIV(t)), "
        "(transforms := ADJOIN(lap·DIV(t↓2), transforms)) ∧ exit))), "
        "lap := ∫(#e^(- s·t)·term, t, 0, ∞), IF(ClosedFormIntegral?(lap), "
        "(tranforms := ADJOIN(lap, transforms)) ∧ exit), s :epsilon Real (0, ∞), "
        "lap := ∫(#e^(- s·t)·term, t, 0, ∞), IF(ClosedFormIntegral?(lap), "
        "((s :=) ∧ transforms := ADJOIN(lap, transforms)) ∧ exit), "
        "WRITE(\"Sorry: Unable to find this particular Laplace transform\"), "
        "RETURN false), counter :+ 1), VariableOrder := varordstore, ∑(transforms))"
    ),
    (
        "LapDivByt(f, t, s, lapl, laplint):=PROG(lapl := Laplace(f, t, s), "
        "IF(lapl = false, RETURN [false]), laplint := ∫(lapl, s, s, ∞), "
        "IF(ClosedFormIntegral?(laplint), PROG(IF(laplint = ∞, RETURN [false]), "
        "IF(¬ MEMBER?(∞, FlattenTree(TREE(laplint))), RETURN [true, laplint]))), "
        "[false])"
    ),
]


#: The definitions of NonlinearRegression.
NONLINEAR = [
    (
        "position(e, u, d, i_):=PROG(d := DIM(u), i_ := 1, LOOP(IF(e = u↓i_, "
        "RETURN i_), IF(i_ > d, RETURN 0), i_ :+ 1))"
    ),
    (
        "GAUSS_NEWTON(eq, parm, β0_, data, n_ := 30, sn_ := 10, cc_ := 1/10^8, dvar, "
        "ivars, ivars2, obs_, data1, k_, pos_, x_, y_, m_, ycpt, eq1, j_, p_, sse_, "
        "t_, sst_, lsse_, δ_, old_, subit_, eps_, std_, prob_, ssr_, fd_, a1_, a2_, "
        "a3_, a4_, title, stats):=PROG(eqq := RHS(eq), dvar := LHS(eq), "
        "ivars2 := VARIABLES(eqq), ivars := SELECT(¬ MEMBER?(i_, parm), i_, ivars2), "
        "ivars1 := ivars, parm1 := parm, obs_ := DIM(data) - 1, vars := data↓1, "
        "data1 := data↓[2, ..., obs_ + 1], k_ := DIM(parm), pos_ := position(dvar, "
        "vars), IF(pos_ = 0, RETURN \"Undefined dependent variable!\", "
        "y_ := data1↓↓pos_), pos_ := VECTOR(position(ivars↓i_, vars), i_, 1, "
        "DIM(ivars)), IF(MEMBER?(0, pos_), "
        "RETURN \"Undefined independent variable(s)!\", m_ := data1↓↓pos_), "
        "ycpt := IF(MEMBER?(1, GRAD(eqq, parm)), 1, 0, 0), k_ :- ycpt, β_ := β0_, "
        "eq1 := VECTOR(SUBST(eqq, ivars, m_↓i), i, obs_), j_ := GRAD(eq1, parm)`, "
        "p_ := SUBST(eq1, parm, β_), r_ := y_ - p_, sse_ := r_·r_, t_ := IF(ycpt = 1, "
        "y_ - VECTOR(AVERAGE(y_), i, obs_), y_), sst_ := t_·t_, iter := [APPEND([0], "
        "β_, [sse_])], eps_ := 1, iter_ := 1, LOOP(IF(iter_ > n_ ∨ eps_ < cc_, exit), "
        "x_ := SUBST(j_, parm, β_), lsse_ := sse_, xpxi := 1/(x_`·x_), "
        "δ_ := xpxi·x_`·r_, old_ := β_, β_ :+ δ_, p_ := SUBST(eq1, parm, β_), "
        "r_ := y_ - p_, sse_ := r_·r_, iter := APPEND(iter, [APPEND([iter_], β_, "
        "[sse_])]), subit_ := 1, LOOP(IF(sse_ ≤ lsse_ ∨ subit_ > sn_, exit), δ_ :/ 2, "
        "β_ := old_ + δ_, p_ := SUBST(eq1, parm, β_), r_ := y_ - p_, sse_ := r_·r_, "
        "iter := APPEND(iter, [APPEND([iter_ + subit_/100], β_, [sse_])]), "
        "subit_ :+ 1), IF(subit_ > sn_, exit), "
        "eps_ := ABS((lsse_ - sse_)/(sse_ + 1/10^6)), iter_ :+ 1), IF(iter_ > n_, "
        "msg := APPEND(\"Convergence failed after \", n_, \" iterations!\"), "
        "IF(subit_ > sn_, msg := \"SSE did not improve after 10 halvings!\", "
        "msg := \"Convergence criteria met!\")), dfe_ := obs_ - k_ - ycpt, "
        "mse_ := sse_/dfe_, se_ := √mse_, std_ := VECTOR(√(xpxi↓i_↓i_·mse_), i_, "
        "k_ + ycpt), t_ := VECTOR(β_↓i_/std_↓i_, i_, k_ + ycpt), "
        "prob_ := VECTOR(1 - STUDENT(t_↓i_, dfe_), i_, k_ + ycpt), "
        "ssr_ := sst_ - sse_, rsq := ssr_/sst_, "
        "arsq := 1 - sse_/(obs_ - k_ - ycpt)·((obs_ - ycpt)/sst_), "
        "fd_ := F_DISTRIBUTION(ssr_·dfe_/(sse_·k_), k_, dfe_), a1_ := [\"Source\", "
        "\"DF\", \"SS\", \"MS\", \"F\", \"Prob(F)\"], a2_ := [\"Reg\", k_, ssr_, "
        "ssr_/k_, ssr_·dfe_/(sse_·k_), IF(fd_ < 0, 0, fd_)], a3_ := [\"Error\", dfe_, "
        "sse_, sse_/dfe_, \" \", \" \"], a4_ := [\"Total\", dfe_ + k_, sst_, \" \", "
        "\" \", \" \"], anova := APPEND([a1_], [a2_], [a3_], [a4_]), "
        "feq := dvar = SUBST(eqq, parm, β_), iter := APPEND([APPEND([\"Iter\"], parm, "
        "[\"SSE\"])], iter), title := [\"Parm\"; \"Value\"; \"STD\"; APPEND(\"t(\", "
        "dfe_, \")\"); \"Prob(t)\"], output := APPEND_COLUMNS(title, APPEND([parm], "
        "[β_], [std_], [t_], [prob_]))`, stats := [\"SE\", \"R^2\", \"AdjR^2\"; se_, "
        "rsq, arsq], "
        "RETURN [\"Gauss_Newton Method\"; \" \"; msg; \" \"; output; \" \"; anova; \" "
        "\"; stats; \" \"; feq])"
    ),
    (
        "MARQUARDT(eq, parm, β0_, data, n_ := 50, cc_ := 1/10^8, dvar, ivars, ivars2, "
        "obs_, data1, k_, pos_, x_, y_, ycpt, eq1, j_, p_, sse_, t_, sst_, λ_, eps_, "
        "iter_, lsse_, xpx, diag_, δ_, mse_, std_, prob_, ssr_, fd_, a1_, a2_, a3_, "
        "a4_, title, stats):=PROG(eqq := RHS(eq), dvar := LHS(eq), "
        "ivars2 := VARIABLES(eqq), ivars := SELECT(¬ MEMBER?(i_, parm), i_, ivars2), "
        "parm1 := parm, ivars1 := ivars, obs_ := DIM(data) - 1, vars := data↓1, "
        "data1 := data↓[2, ..., obs_ + 1], k_ := DIM(parm), pos_ := position(dvar, "
        "vars), IF(pos_ = 0, RETURN \"Undefined dependent variable!\", "
        "y_ := data1↓↓pos_), pos_ := VECTOR(position(ivars↓i_, vars), i_, 1, "
        "DIM(ivars)), IF(MEMBER?(0, pos_), "
        "RETURN \"Undefined independent variable(s)!\", m_ := data1↓↓pos_), "
        "ycpt := IF(MEMBER?(1, GRAD(eqq, parm)), 1, 0, 0), k_ :- ycpt, β_ := β0_, "
        "eq1 := VECTOR(SUBST(eqq, ivars, m_↓i), i, obs_), j_ := GRAD(eq1, parm)`, "
        "p_ := SUBST(eq1, parm, β_), r_ := y_ - p_, sse_ := r_·r_, t_ := IF(ycpt = 1, "
        "y_ - VECTOR(AVERAGE(y_), i, obs_), y_), sst_ := t_·t_, λ_ := 0.001, "
        "iter := [APPEND([0], [\" \"], β_, [sse_])], eps_ := 1, iter_ := 1, "
        "LOOP(IF(iter_ > n_ ∨ eps_ < cc_, exit), x_ := SUBST(j_, parm, β_), "
        "lsse_ := sse_, xpx := x_`·x_, diag_ := VECTOR(VECTOR(IF(q_ = s_, xpx↓q_↓s_, "
        "0), s_, k_ + ycpt), q_, k_ + ycpt), xpxi := 1/(xpx + λ_·diag_), "
        "δ_ := xpxi·x_`·r_, β_ :+ δ_, p_ := SUBST(eq1, parm, β_), r_ := y_ - p_, "
        "sse_ := r_·r_, iter := APPEND(iter, [APPEND([iter_], [λ_], β_, [sse_])]), "
        "IF(sse_ ≤ lsse_, PROG(eps_ := ABS((lsse_ - sse_)/(sse_ + 1/10^6)), λ_ :/ 10), "
        "PROG(β_ :- δ_, λ_ :* 10), RETURN \"Cannot resolve SSE test!\"), iter_ :+ 1), "
        "IF(iter_ > n_, msg := APPEND(\"Convergence failed after \", n_, "
        "\" iterations!\"), msg := \"Convergence criteria met!\"), "
        "dfe_ := obs_ - k_ - ycpt, mse_ := sse_/dfe_, se_ := √mse_, "
        "std_ := VECTOR(√(xpxi↓i_↓i_·mse_), i_, k_ + ycpt), "
        "t_ := VECTOR(β_↓i_/std_↓i_, i_, k_ + ycpt), "
        "prob_ := VECTOR(1 - STUDENT(t_↓i_, dfe_), i_, k_ + ycpt), "
        "ssr_ := sst_ - sse_, rsq := ssr_/sst_, "
        "arsq := 1 - sse_/(obs_ - k_ - ycpt)·((obs_ - ycpt)/sst_), "
        "fd_ := F_DISTRIBUTION(ssr_·dfe_/(sse_·k_), k_, dfe_), a1_ := [\"Source\", "
        "\"DF\", \"SS\", \"MS\", \"F\", \"Prob(F)\"], a2_ := [\"Reg\", k_, ssr_, "
        "ssr_/k_, ssr_·dfe_/(sse_·k_), IF(fd_ < 0, 0, fd_)], a3_ := [\"Error\", dfe_, "
        "sse_, sse_/dfe_, \" \", \" \"], a4_ := [\"Total\", dfe_ + k_, sst_, \" \", "
        "\" \", \" \"], anova := APPEND([a1_], [a2_], [a3_], [a4_]), "
        "feq := dvar = SUBST(eqq, parm, β_), iter := APPEND([APPEND([\"Iter\"], "
        "[\"λ\"], parm, [\"SSE\"])], iter), "
        "title := [\"Parm\"; \"Value\"; \"STD\"; APPEND(\"t(\", dfe_, "
        "\")\"); \"Prob(t)\"], output := APPEND_COLUMNS(title, APPEND([parm], [β_], "
        "[std_], [t_], [prob_]))`, stats := [\"SE\", \"R^2\", \"AdjR^2\"; se_, rsq, "
        "arsq], RETURN [\"Marquardt Method\"; \" \"; msg; \" \"; output; \" \"; anova; "
        "\" \"; stats; \" \"; feq])"
    ),
    (
        "Predicted_Values(x0, ci := 0.95):=PROG(prd := SUBST(RHS(feq), ivars1, x0), "
        "g_ := GRAD(eqq, parm1), gv := SUBST(g_, APPEND(parm1, ivars1), APPEND(β_, "
        "x0)), i_ := gv·xpxi·gv`, sep := se_·√i_, sef := se_·√(1 + i_), "
        "PROG(PrecisionDigits := 6, invt := RHS(NSOLVE(STUDENT(t, dfe_) - ci, t, 0, "
        "∞)), PrecisionDigits := 10, NotationDigits := 8), cyh1 := prd - sep·invt, "
        "cyh2 := prd + sep·invt, cy01 := prd - sef·invt, cy02 := prd + sef·invt, "
        "title := APPEND(STRING(ci·100), \"% Confidence Interval\"), "
        "[title; [\"Value\", prd, \" \"; \"Se_yhat/Se_Y0\", sep, sef; \"CI_yhat\", "
        "cyh1, cyh2; \"CI_Y0\", cy01, cy02]])"
    ),
]


#: The definitions of TaylorODE.
TAYLOR = [
    (
        "TaylorODE(f, x, y, o, t, h, n, init_vals, header? := 0, xy_only? := 0, vars, "
        "vars1, vars2, terms, co_effs, it_scheme, it_vars, "
        "table):=PROG(IF(DIM(init_vals) ≠ o + 1, "
        "RETURN \"Incorrect Number of Initial Values\"), "
        "vars := MAP_LIST(APPEND(STRING(y), r), r, 0, o - 1), vars1 := DELETE(vars, "
        "-1), vars2 := DELETE(vars, 1), lastvar := vars↓o, IF(o = 1, "
        "terms := APPEND(vars, ITERATES(f·∂(u, lastvar) + ∂(u, x), u, f, t - 1)), "
        "terms := APPEND(vars, ITERATES(GRAD(u, vars1) ⋅ vars2 + f·∂(u, "
        "lastvar) + ∂(u, x), u, f, t - 1))), co_effs := MAP_LIST(h^r/r!, r, 0, t), "
        "it_scheme := APPEND([x + h], MAP_LIST(co_effs ⋅ terms↓[r, ..., r + t], r, 1, "
        "o)), it_vars := ADJOIN(x, vars), table := ITERATES(it_scheme, it_vars, "
        "init_vals, n), IF(header? = 0, IF(xy_only? = 1, RETURN table COL [1, 2], "
        "RETURN table)), IF(xy_only? = 1, RETURN (APPEND([it_vars], table)) COL [1, "
        "2]), APPEND([it_vars], table))"
    ),
]


def test_a_package_is_not_part_of_the_library():
    session = Session()
    for name in ("Laplace", "InvLaplace", "GAUSS_NEWTON", "MARQUARDT", "TaylorODE"):
        assert name not in session.functions
    from rederive import library

    assert not {"Laplace", "InvLaplace", "TaylorODE"} & library.DEFINED
    assert not {"GAUSS_NEWTON", "MARQUARDT", "Predicted_Values"} & library.DEFINED


# -- LaplaceTransforms --------------------------------------------------------------


async def test_laplace_transforms(tmp_path):
    session = loaded(tmp_path, "LaplaceTransforms", LAPLACE)
    assert await answers(
        session,
        "Laplace(4*t^2*#e^(3*t)*SIN(4*t), t, s)",
        "Laplace(STEP(t - 2)*t^2, t, s)",
        "Laplace(COS(2*t), t, s)",
        "Laplace(t*#e^(-t), t, s)",
        # Division by t: the integral of the transform.
        "Laplace(SIN(t)/t, t, s)",
    ) == [
        "32*(3*s^2 - 18*s + 11)/(s^2 - 6*s + 25)^3",
        "2*(2*s^2 + 2*s + 1)*#e^(-2*s)/s^3",
        "s/(s^2 + 4)",
        "1/(s + 1)^2",
        "pi/2 - ATAN(s)",
    ]


async def test_a_shifted_step_times_a_sine(tmp_path):
    session = loaded(tmp_path, "LaplaceTransforms", LAPLACE)
    found, derives = await answers(
        session,
        "Laplace(STEP(t - 4)*t^2*SIN(3*t), t, s)",
        "2*#e^(-4*s)*(3*(8*s^4 + 8*s^3 + 147*s^2 + 72*s + 639)*COS(12) + (8*s^5 + "
        "4*s^4 + 145*s^3 + 621*s - 324)*SIN(12))/(s^2 + 9)^3",
    )
    assert (await session.simplify(f"{found} - ({derives})")).text == "0"


async def test_inverse_laplace_transforms(tmp_path):
    session = loaded(tmp_path, "LaplaceTransforms", LAPLACE)
    assert await answers(
        session,
        "InvLaplace(32*(3*s^2 - 18*s + 11)/(s^2 - 6*s + 25)^3, s, t)",
        "InvLaplace(2*#e^(-2*s)*(2*s^2 + 2*s + 1)/s^3, s, t)",
        "InvLaplace(1/(s^2 + 3*s + 2), s, t)",
        "InvLaplace(s/(s^2 + 4), s, t)",
    ) == [
        "4*t^2*SIN(4*t)*#e^(3*t)",
        "t^2*SIGN(t - 2)/2 + t^2/2",
        "#e^(-t) - #e^(-2*t)",
        "COS(2*t)",
    ]


async def test_an_inverse_transform_stores_what_simplifying_again_leaves(
    tmp_path, monkeypatch
):
    """The program taking the transform apart stores products of polynomials and
    complex exponentials, each simplified to a fixed point (see `stored_twice`)."""
    changed = stored_twice(monkeypatch)
    session = loaded(tmp_path, "LaplaceTransforms", LAPLACE)
    assert await answers(
        session, "InvLaplace(32*(3*s^2 - 18*s + 11)/(s^2 - 6*s + 25)^3, s, t)"
    ) == ["4*t^2*SIN(4*t)*#e^(3*t)"]
    assert changed == []


# -- NonlinearRegression ------------------------------------------------------------

#: The census of the United States, 1790 to 1970, in millions.
POPULATION = (
    "pop := [[p, y], [3.929, 1790], [5.308, 1800], [7.239, 1810], [9.638, 1820], "
    "[12.866, 1830], [17.069, 1840], [23.191, 1850], [31.443, 1860], [39.818, 1870], "
    "[50.155, 1880], [62.947, 1890], [75.994, 1900], [91.972, 1910], [105.71, 1920], "
    "[122.775, 1930], [131.669, 1940], [151.325, 1950], [179.323, 1960], "
    "[203.211, 1970]]"
)


async def test_a_gauss_newton_fit(tmp_path):
    session = loaded(tmp_path, "NonlinearRegression", NONLINEAR)
    (report,) = await answers(
        session,
        "> Precision := Approximate",
        "> NotationDigits := 8",
        f"> {POPULATION}",
        "GAUSS_NEWTON(p = c*NORMAL(a + b*(y - 1790)), [a, b, c], [-2.4, 0.012, 400], "
        "pop)",
    )
    assert report.startswith(
        '[["Gauss_Newton Method"], [" "], ["Convergence criteria met!"], [" "], '
        '[[["Parm", "Value", "STD", "t(16)", "Prob(t)"], '
        "[a, -2.3028183, 0.032845861, -70.109847, 1], "
        "[b, 0.012628143, 0.00094182436, 13.408172, 2.0290257*10^-10], "
        "[c, 407.10605, 60.663197, 6.7109231, 2.5009143*10^-6]]]"
    )


async def test_the_confidence_interval_of_a_prediction(tmp_path):
    session = loaded(tmp_path, "NonlinearRegression", NONLINEAR)
    report, predicted = await answers(
        session,
        "> Precision := Approximate",
        f"> {POPULATION}",
        "GAUSS_NEWTON(p = c*#e^(b*(y - 1790)) + a, [a, b, c], [10, 0.03, 20], pop)",
        "Predicted_Values([2010])",
    )
    assert "[b, 0.01152664956, 0.0006609969888, 17.43827848, " in report
    assert report.endswith("[p = 3.235913573*10^-8*#e^(0.01152664956*y) - 31.28072367]]")
    # Derive 6: 323.68166 and 360.21787, and 322.46847 and 361.43106 for Y0.
    assert predicted == (
        '[["95% Confidence Interval"], [[["Value", 341.94977, " "], '
        '["Se_yhat/Se_Y0", 10.463537, 11.158423], ["CI_yhat", 323.68165, 360.21788], '
        '["CI_Y0", 322.46846, 361.43107]]]]'
    )


async def test_a_marquardt_fit(tmp_path):
    session = loaded(tmp_path, "NonlinearRegression", NONLINEAR)
    (report,) = await answers(
        session,
        "> Precision := Approximate",
        "> NotationDigits := 8",
        f"> {POPULATION}",
        "MARQUARDT(p = c*NORMAL(a + b*(y - 1790)), [a, b, c], [-2.4, 0.012, 400], pop)",
    )
    assert report.startswith(
        '[["Marquardt Method"], [" "], ["Convergence criteria met!"], [" "], '
        '[[["Parm", "Value", "STD", "t(16)", "Prob(t)"], '
        "[a, -2.3028183, 0.032845831, "
    )


# -- TaylorODE ----------------------------------------------------------------------


async def test_a_fifth_order_equation_by_the_taylor_method(tmp_path):
    session = loaded(tmp_path, "TaylorODE", TAYLOR)
    assert (
        await session.approx(
            'TaylorODE(-4*"y1" - 4*"y0" + 4*#e^x*SIN(x), x, y, 5, 7, 0.2, 5, '
            "[0, 0, 1, 2, 2, 0], 1, 1)"
        )
    ).text == (
        '[[x, "y0"], [0, 0], [0.2, 0.2426552685], [0.4, 0.5809439002], '
        "[0.6, 1.028845664], [0.8, 1.596505336], [1, 2.287355278]]"
    )


async def test_the_taylor_method_loaded_by_load(tmp_path):
    package(tmp_path / "TaylorODE.mth", TAYLOR)
    session = Session()
    session.author('LOAD("TaylorODE.mth")')
    assert session.take_notice() is None
    found = await session.approx('TaylorODE(-"y0", x, y, 1, 4, 0.1, 3, [0, 1])')
    assert found.text == (
        "[[0, 1], [0.1, 0.9048375], [0.2, 0.8187309014], [0.3, 0.7408184220]]"
    )
    assert await answers(
        session, 'TaylorODE(-"y0", x, y, 1, 4, 0.1, 3, [0, 1, 2])'
    ) == ['"Incorrect Number of Initial Values"']
