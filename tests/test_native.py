"""Compiled programs: they give exactly what converting each node as it is gives.

The evaluator runs a program's statements compiled into closures, which compute its
integers, fractions, truth values and the vectors it writes into as Python values (see
`rederive.engine.native` and "compiled nodes" in `rederive.engine.evaluate`). Turning
compiling off (`evaluate.COMPILED`) must change no answer, no global a program leaves
and no count of steps. So the program tests run again with it off, as do the library's
quicker cases that run programs; and the cases here, which aim at where the two could
part (values shared between variables, writes out of range, values that stop being
native numbers midway, the ends of programs, programs at top level, whose variables are
globals, programs where values are not native: under Approximate, in a definition shown,
resumed in a suspended conditional, where nothing runs), run both ways, each transcript
compared with the other.

The benchmarks of the native values run here at a reduced size, for their answers.
"""

import pytest
from test_library import LIBRARY_CASES, _case_ids
from test_library import test_a_library_function_computes_its_specified_results as _library
from test_programs import *  # noqa: F403 - the program tests, run again uncompiled

from rederive.engine import evaluate
from rederive.engine.client import EngineError
from rederive.engine.context import Context
from rederive.engine.vectors import Vector
from rederive.model.session import Session
from rederive.syntax import write_expression


@pytest.fixture(autouse=True)
def uncompiled(monkeypatch, request):
    """Each test from another module runs here uncompiled."""
    if request.node.function.__module__ != __name__:
        monkeypatch.setattr(evaluate, "COMPILED", False)


#: The library cases that run programs natively and take little time; the others would
#: take too long to run twice.
NATIVE_LIBRARY = frozenset(
    """
    ADJOINT ALMOST_LIN ALMOST_LIN_GEN APPEND_COLUMNS AREA AREAY_OF_REVOLUTION
    AREA_CENTROID AREA_INERTIA AREA_OF_REVOLUTION ASSOCIATED_LEGENDRE_P AUTONOMOUS
    AreaBetweenCurves AreaOverCurve AreaUnderCurve BELL BERNOULLI BERNOULLI_ODE
    BERNOULLI_ODE_GEN BERNOULLI_POLY BESSEL_I_ASYMP BESSEL_I_SERIES BESSEL_J_ASYMP
    BESSEL_J_LIST BESSEL_J_SERIES BESSEL_K_ASYMP BESSEL_Y BESSEL_Y_ASYMP BESSEL_Y_SERIES
    BINOMIAL_DENSITY BINOMIAL_DISTRIBUTION CATALAN CENTERED CENTERED_CUBE CENTERED_HEX
    CENTERED_PYRAMID CENTER_OF_CURVATURE CHEBYCHEV_T CHEBYCHEV_T_LIST CHEBYCHEV_U
    CHEBYCHEV_U_LIST CLAIRAUT CLAIRAUT_DIF COFACTOR CONE CONTINUED_FRACTION CONVERGENTS
    COPROJECTION COVARIANT_METRIC_TENSOR CRT CURVATURE CYLINDER CYLINDRICAL_VOLUME
    DAWSON DEF_INT_PARTS DEF_INT_SUBST DIF2_DATA DIF_NUMERIC DIRECTION_FIELD DIVISORS
    DIVISOR_SIGMA DIVISOR_TAU DSOLVE1 DSOLVE2 DSOLVE2_BV EI1 EIF ELLIPTIC_E ELLIPTIC_F
    EN_ASYMP EULER_BETA EULER_ODE EULER_PHI EXACT EXACT2 EXACT_GEN EXTENDED_GCD
    EXTRACT_2_COLUMNS FAREY FIBONACCI FIXED_POINT FORCE0 FRESNEL_COS_SERIES
    FRESNEL_SIN_SERIES FUN_LIN_CCF FUN_LIN_CCF_GEN GAUSS GAUSS_SERIES GEGENBAUER_C
    GEGENBAUER_C_LIST GENERALIZED_LAGUERRE GEN_HOM GEN_HOM_GEN GEN_LUCAS GEOMETRIC1
    GEOMETRY_MATRIX GOODNESS_OF_FIT GRID_LINES GRID_POINTS HERMITE_H HERMITE_HE
    HERMITE_HE_LIST HERMITE_H_LIST HEX HOMOGENEOUS_GEN HORIZONTALS HURWITZ_ZETA
    HYPERGEOMETRIC_DENSITY HYPERGEOMETRIC_DISTRIBUTION HYPERGEOMETRIC_SERIES
    IMP_CENTER_OF_CURVATURE IMP_CURVATURE IMP_OSCULATING_CIRCLE IMP_PERPENDICULAR
    IMP_TANGENT INCOMPLETE_GAMMA_SERIES INT_DATA INT_PARTS INT_SUBST INVERSE ISOMETRIC
    JACOBI JACOBIAN JACOBI_P KI KRONECKER KUMMER_SERIES LAGUERRE_L LEFT_RIEMANN
    LEGENDRE_P LERCH_PHI LI LIM2 LIN1_DIFFERENCE LIN2_CCF_BV LINEAR1_GEN
    LINEAR_CORRELATION_COEFFICIENT LIOUVILLE LUCAS LUCAS_LEHMER MATPROD MERSENNE
    MERSENNE_DEGREE MINOR MOEBIUS_MU MONOMIAL_TEST NEWTON NEXT_MERSENNE_DEGREE
    NORMAL_LINE NTH_PRIME NULL_SPACE OCTAHEDRAL OSCULATING_CIRCLE OUTER
    PARA_CENTER_OF_CURVATURE PARA_CURVATURE PARA_DIF PARA_OSCULATING_CIRCLE
    PARA_PERPENDICULAR PARTITION PARTS_LIST PELL PENTATOPE PERFECT PERPENDICULAR PIVOT
    POCHHAMMER POISSON_DENSITY POISSON_DISTRIBUTION POLAR POLAR_AREA
    POLAR_CENTER_OF_CURVATURE POLAR_CURVATURE POLAR_DIF POLAR_OSCULATING_CIRCLE
    POLAR_PERPENDICULAR POLAR_SUM POLAR_TANGENT POLAR_TO_RECTANGULAR POLYGONAL
    POLYGONAL_PYRAMID POLYGON_FILL POLY_INTERPOLATE POLY_INTERPOLATE_EXPRESSION
    PRIME_POWER? PRIMITIVE_ROOT PROVE_SUM PlotInt RANDOM_MATRIX RANDOM_NORMAL
    RANDOM_POLY RANDOM_VECTOR RAYS RECTANGULAR_TO_POLAR RECURRENCE RECURRENCE1
    REMOVE_ELEMENTS RHOMBIC_DODECAHEDRAL ROTATE_X ROTATE_Z SCALE_ELEMENT SEPARABLE_GEN
    SMOOTH_COLUMN SMOOTH_VECTOR SOLVE_MOD SPHERE SPHERICAL_BESSEL_J_LIST
    SPHERICAL_VOLUME SQUAREFREE SQUARE_ROOT SQUARE_WAVE STAR STIRLING STIRLING1
    STIRLING2 STIRLING_CYCLE STIRLING_SUBSET SUBTRACT_ELEMENTS SURFACE_AREA
    SWAP_ELEMENTS TANGENT TANGENT_PLANE TAYLOR_ODES TETRAHEDRAL TORUS TRIANGULAR U_LUCAS
    U_MOD VOLUME VOLUMEY_OF_REVOLUTION VOLUME_CENTROID VOLUME_OF_REVOLUTION V_LUCAS
    V_MOD WEBER_D
    """.split()
)


_LIBRARY = {
    id_: case
    for case, id_ in zip(LIBRARY_CASES, _case_ids(LIBRARY_CASES), strict=True)
    if id_ in NATIVE_LIBRARY
}


@pytest.mark.parametrize("case", _LIBRARY.values(), ids=_LIBRARY.keys())
async def test_a_library_case_computes_its_results_uncompiled(monkeypatch, case):
    monkeypatch.setattr(evaluate, "COMPILED", False)
    await _library(case)


async def transcript(monkeypatch, lines, compiled, step_limit=-1):
    """What a fresh session answers to each of `lines` (one starting with `>` is
    entered, one starting with `?` asks for the variables it offers, which it finds
    where nothing runs), compiled or not, each answer with the steps it took."""
    monkeypatch.setattr(evaluate, "COMPILED", compiled)
    stores = []
    fresh = Context.fresh

    def kept(self):
        made = fresh(self)
        stores.append(made.store)
        return made

    monkeypatch.setattr(Context, "fresh", kept)
    session = Session(step_limit=step_limit)
    found = []
    for line in lines:
        if line.startswith("> "):
            session.author(line[2:])
            continue
        stores.clear()
        try:
            if line.startswith("? "):
                answer = repr(await session.variables(line[2:]))
            else:
                answer = (await session.simplify(line)).text
        except EngineError as error:
            answer = f"(error: {error})"
        found.append((line, answer, sum(store.steps for store in stores)))
    return found


CASES = {
    "aliasing": [
        "> al1(v, w) := PROG(v := [1, 2, 3], w := v, v SUB 1 := 0, w SUB 2 := 9, [v, w])",
        "al1()",
        "> al2(v, w, c) := PROG(v := [1, 2, 3], c := 1, w := IF(c = 1, v, 0), "
        "v SUB 1 := 0, [v, w])",
        "al2()",
        "> al3(v, w) := PROG(v := [1, 2, 3], w := [v, v], v SUB 1 := 0, [v, w])",
        "al3()",
        "> al4(v, w) := PROG(v := [1, 2, 3], v SUB 1 := 5, w := APPEND(v, [4]), "
        "v SUB 2 := 6, [v, w])",
        "al4()",
        "> setf(u) := PROG(u SUB 1 := 99, u)",
        "> al5(v, r) := PROG(v := [1, 2, 3], v SUB 3 := 7, r := setf(v), [v, r])",
        "al5()",
        "> al6(v, w) := PROG(v := [1, 2, 3], v SUB 1 := 4, w := LOOP(v, exit), "
        "v SUB 1 := 9, [v, w])",
        "al6()",
        "> al7(v, w) := PROG(v := [1, 2], v SUB 1 := 3, w := v, w := 5, v SUB 2 := 4, "
        "[v, w])",
        "al7()",
        "> al8(v) := PROG(v SUB 2 := 0, v)",
        "> vv := [5, 6, 7]",
        "[al8(vv), vv]",
        "> al9(v, w) := PROG(v := [1, 2, 3], w := v, v SUB 1 :+ 10, [v, w, w SUB 1])",
        "al9()",
        "> sh3(a, b) := PROG(a :+ 1, a)",
        "> al10(v, r) := PROG(v := [1, 2], r := sh3(v, v := 3), [v, r])",
        "al10()",
        "> sh4(a, b) := PROG(a SUB 1 := 9, a)",
        "> al11(v, r) := PROG(v := [1, 2], v SUB 2 := 0, r := sh4(v, v := [5, 6]), "
        "[v, r])",
        "al11()",
    ],
    "nested": [
        "> ne1(m, r) := PROG(m := [[1, 2], [3, 4]], m SUB 2 SUB 1 := 7, r := m SUB 1, "
        "r SUB 1 := 0, m SUB 1 SUB 2 :* 5, [m, r])",
        "ne1()",
        "> ne2(m) := PROG(m := IDENTITY_MATRIX(3), m SUB 2 SUB 3 := x, "
        "m SUB 1 := [7, 8, 9], m SUB 1 SUB 1 :+ 1, m)",
        "ne2()",
        "> ne3(m, r) := PROG(m := [[1, 2], [3, 4]], m SUB 1 SUB 1 := 5, r := m SUB 2, "
        "m SUB 2 SUB 2 := 6, [m, r])",
        "ne3()",
        "> ne4(m) := PROG(m := [[1, 2], [3, 4]], m SUB 1 := m SUB 2, m SUB 2 SUB 1 := 0, "
        "m)",
        "ne4()",
        "> ne5(m) := PROG(m := [[1, 2], 3], m SUB 2 SUB 1 := 0, m SUB 1 SUB 3 := 0, m)",
        "ne5()",
        "> ne6(m) := PROG(m := [[1, 2], [3, 4]], m SUB [1, 2] := [[5, 6], [7, 8]], m)",
        "ne6()",
    ],
    "places": [
        "> pl(v, k) := PROG(v := [1, 2, 3], v SUB 1 := 0, v SUB k := 8, [v, v SUB k])",
        "[pl(1, 3), pl(1, 4), pl(1, 0), pl(1, -1), pl(1, 1/2), pl(1, x), pl(1, -3)]",
        "> pu(v, k) := PROG(v := [1, 2, 3], v SUB 1 := 0, v SUB k :+ 8, [v, v SUB k])",
        "[pu(1, 3), pu(1, 4), pu(1, 0), pu(1, -1), pu(1, x)]",
        "> pr(v, k) := PROG(v := [1, 2, 3], v SUB 2 := 5, [v SUB k, DIM(v), FIRST(v)])",
        "[pr(1, 1), pr(1, -1), pr(1, -3), pr(1, -4), pr(1, 4), pr(1, 0), pr(1, [1, 3])]",
        "> pe(v) := PROG(v := [], [DIM(v), FIRST(v), v SUB 1])",
        "pe()",
        "> ps(s) := PROG(s := 123456, [s SUB 2, DIM(s), s SUB -1])",
        "ps()",
        "> pq(s) := PROG(s := \"hello\", [s SUB 2, DIM(s), IF(s = \"hello\", 1, 2)])",
        "pq()",
    ],
    "symbolic": [
        "> sy1(v) := PROG(v := [x, 1, 2], v SUB 2 :+ x, v SUB 1 :* 2, v SUB 3 := v SUB 3 "
        "+ y, v)",
        "sy1()",
        "> sy2(n, s, k) := PROG(s := 0, k := 0, LOOP(IF(k >= n, RETURN(s)), s :+ k*x, "
        "k :+ 1))",
        "[sy2(5), sy2(n)]",
        "> sy3(n, k) := PROG(k := 0, LOOP(IF(k > n, RETURN(k)), k :+ 1))",
        "sy3(y)",
        "> sy4(a, b) := PROG(IF(a > b, 1, 2, 3))",
        "[sy4(1, 2), sy4(x, 2), sy4(2, 1)]",
        "> sy5(a) := PROG(IF(a > 0, RETURN(1)), 2)",
        "[sy5(1), sy5(-1), sy5(z)]",
        "> sy6(v, k) := PROG(v := [1, 2, 3], v SUB 2 := 0, k := 0, LOOP(k :+ 1, "
        "IF(k > 3, RETURN([v, k])), v SUB k := v SUB k + z))",
        "sy6()",
    ],
    "numbers": [
        "> nu1(s, k) := PROG(s := 1, k := 0, LOOP(IF(s < 1/100, RETURN([s, k])), "
        "s := s/3, k :+ 1))",
        "nu1()",
        "> nu2(a, b) := [a/b, a^b, a^(-b), -a, ABS(a - b), FLOOR(a, b), CEILING(a, b), "
        "MOD(a, b), MODS(a, b), FLOOR(a), MOD(a)]",
        "> nu3(a, b) := PROG(a := a, b := b, [a/b, a^b, a^(-b), -a, ABS(a - b), "
        "FLOOR(a, b), CEILING(a, b), MOD(a, b), MODS(a, b), FLOOR(a), MOD(a), "
        "CEILING(a), |a - b|])",
        "[nu3(7, 3), nu3(-7, 3), nu3(7, -3), nu3(-7, -3), nu3(7, 0), nu3(0, 0)]",
        "[nu3(7/2, 2), nu3(-7/2, 3/4), nu3(0, -2), nu3(0, 3), nu3(2, 100)]",
        "[nu3(10^30, 7), nu3(-10^30, 10^20 + 1), nu3(2/3, 5), nu3(2, 9000)]",
        "nu3(x, 2)",
        "> nu4(k) := PROG(k := 2, k :/ 4, k :* 6, k :- 1/2, k :/ 0)",
        "nu4()",
        "> nu5(k) := PROG(k := 5, k :+ 1/2, k :+ 1/2, k :* 0, k :- 3, k)",
        "nu5()",
        "> nu6(k, s) := PROG(k := 1/0, s := 0/0, [k, s, k + 1, s = s, k > 1])",
        "nu6()",
        "> nu7(k, s) := PROG(k := inf, s := ?, [k + 1, s + 1, IF(s = 1, 1, 2, 3), "
        "IF(k > 1, 1, 2, 3)])",
        "nu7()",
        "> nu8(k) := PROG(k := 0.5, k :+ 0.25, [k, k*3, 2^k, k^2])",
        "nu8()",
        "> nu9(a, b) := PROG(a := 3, b := 4, [a = b, a /= b, a < b, a <= b, a > b, "
        "a >= b, 1/2 < 2/3, a = 3, a*b = 12])",
        "nu9()",
        "> nu10(k) := PROG(k := 2^1000, k :+ 1, MOD(k, 10^9 + 7))",
        "nu10()",
        "> nu11(k) := PROG(k := 0, k :+ #i, k :* 2, k)",
        "nu11()",
        "> nu12(k) := PROG(k := SQRT(2), k :* k, k :+ 1, k)",
        "nu12()",
        "> nu13(k) := PROG(k := 3, [k!, k%, DIM(k), k^(1/2), k^0, 0^k])",
        "nu13()",
    ],
    "logic": [
        "> lo1(a, b) := PROG(a := 3, b := 5, [a AND b, a OR b, a > 1 AND a < 5, "
        "a > 4 OR b > 4, a > 4 AND x, a < 4 OR x, NOT (a > 1), a < b < 6])",
        "lo1()",
        "> lo2(k, c) := PROG(k := 0, c := 0, LOOP(k :+ 1, IF(k > 20, RETURN(c)), "
        "IF(MOD(k, 2) = 0 AND MOD(k, 3) = 0 OR k = 7, c :+ 1)))",
        "lo2()",
        "> lo3(b) := PROG(b := 2 > 1, [b, b AND true, IF(b, 1, 2)])",
        "lo3()",
        "> lo4(k) := PROG(k := 3, IF(k), IF(k = 3), IF(k = 4), IF(k > 2, k))",
        "lo4()",
        "> lo5(k) := PROG(k := 3, [k > 2, k = x, IF(k = x, 1, 2, 3), RETURN(k = 3)])",
        "lo5()",
        "> lo6(k) := PROG(k := 3, IF(true AND k > 2, 1, 2) + IF(false OR k < 2, 1, 2))",
        "lo6()",
    ],
    "flow": [
        "> fl1(k, s) := PROG(k := 0, s := 0, LOOP(k :+ 1, IF(k > 10, exit), "
        "IF(MOD(k, 3) = 0, PROG(s :+ k, exit)), s :+ 100), [k, s])",
        "fl1()",
        "> fl2(k) := PROG(k := 0, LOOP(k :+ 1, IF(k = 5, exit)))",
        "fl2()",
        "> fl3(k) := PROG(k := 1, LOOP(IF(k > 3, exit), k :+ 1), k)",
        "fl3()",
        "> fl4(k) := PROG(k := 0, LOOP(k :+ 1, [k, IF(k > 2, exit)]))",
        "fl4()",
        "> fl5(k) := PROG(k := 0, EVERY(PROG(k :+ i, i < 4), i, 1, 9), k)",
        "fl5()",
        "> fl6(k) := PROG(k := 0, SOME(IF(i = 3, RETURN(k), PROG(k :+ i, false)), i, 1, 9))",
        "fl6()",
        "> fl7(k) := PROG(k := 5, exit, k :+ 1)",
        "fl7()",
        "> fl8(k) := PROG(k := 5, IF(k > 2, exit), 7)",
        "fl8()",
        "> fl9(k) := LOOP(k :+ 1, IF(k > 3, RETURN(k)))",
        "fl9(0)",
        "> fl10(v, k) := PROG(v := [0, 0, 0], k := 0, LOOP(k :+ 1, v SUB k := k^2, "
        "IF(k = 3, RETURN(v))))",
        "fl10()",
        "> fl11(k) := PROG(k := 2, IF(k = 2, [k :+ 1, k :+ 1], 0), k)",
        "fl11()",
    ],
    "recursion": [
        "> fib(n) := IF(n < 2, n, fib(n - 1) + fib(n - 2))",
        "fib(15)",
        "> fact(n, p, k) := PROG(p := 1, k := 1, LOOP(IF(k > n, RETURN(p)), p :* k, "
        "k :+ 1))",
        "[fact(20), fact(0), fact(m)]",
        "> rsum(n) := IF(n = 0, 0, n + rsum(n - 1))",
        "rsum(50)",
        "> deep(n, k) := PROG(k := n, IF(k <= 0, RETURN(0)), 1 + deep(k - 1))",
        "deep(30)",
        "> gcd3(a, b, t) := LOOP(IF(b = 0, RETURN(a)), t := b, b := MOD(a, b), a := t)",
        "[gcd3(1071, 462), gcd3(-12, 18), gcd3(x, 0)]",
    ],
    "globals": [
        "> cnt := 0",
        "> tot := 0",
        "> glob := 0",
        "> gv := 0",
        "> gl1(n, k) := PROG(k := 0, LOOP(IF(k >= n, RETURN(cnt)), cnt :+ k, k :+ 1))",
        "gl1(10)",
        "cnt",
        "> gl2(n, k) := PROG(k := 0, tot := [0, 0], LOOP(IF(k >= n, RETURN(tot)), "
        "tot SUB 1 :+ k, tot SUB 2 := k, k :+ 1))",
        "gl2(5)",
        "tot",
        "> gl3(k) := PROG(k := 3, glob := k, glob :+ 1, [glob, k])",
        "gl3()",
        "glob",
        "> gl4(v) := PROG(v := [1, 2], gv := v, v SUB 1 := 0, [v, gv])",
        "gl4()",
        "gv",
    ],
    "settings": [
        "> se1(s, k) := PROG(s := 0, k := 1, LOOP(IF(k > 6, RETURN(s)), s :+ 1/k, "
        "IF(k = 3, Precision := Approximate), k :+ 1))",
        "se1()",
        "Precision",
        "> Precision := Exact",
        "> se2(s) := PROG(s := 1/3, PrecisionDigits := 20, s := s + 1, Precision := "
        "Exact, s + 1/7)",
        "se2()",
        "> se3(s) := PROG(s := 1/3, Precision := Mixed, s :+ 1, s)",
        "se3()",
        "> Precision := Exact",
        "> se4(s) := PROG(s := 10, OutputBase := Binary, s :+ 1, [s, DIM(s), s SUB 1], "
        "OutputBase := Decimal, s)",
        "se4()",
        "> Precision := Approximate",
        "> se5(s, k) := PROG(s := 0, k := 1, LOOP(IF(k > 5, RETURN(s)), s :+ 1/k, k :+ 1))",
        "se5()",
        "> Precision := Exact",
    ],
    "mixed": [
        "> Precision := Mixed",
        "> mx1(n, k, s) := PROG(k := 0, s := 0, LOOP(IF(k >= n, RETURN(s)), "
        "s :+ MOD(k*k, 7)/3 + 1/(k + 1)^2, k :+ 1))",
        "mx1(1)",
        "mx1(20)",
        "> mx2(v, k) := PROG(v := [1/3, 2, SQRT(2)], v SUB 1 :+ 1/7, v SUB 3 :* 2, "
        "k := v SUB 1 + 1, [v, k, FLOOR(k), k > 1, ABS(-k)])",
        "mx2()",
        "> Precision := Exact",
    ],
    "binders": [
        "> bi1(v, k) := PROG(v := [3, 1, 4, 1, 5], v SUB 2 := 9, [EVERY(v SUB i > 0, i, 1, "
        "5), SOME(v SUB i > 8, i, 1, 5), EVERY(v SUB i > 1, i, 1, 5), SOME(v SUB i > 9, i, "
        "1, 5)])",
        "bi1()",
        "> bi2(n) := [EVERY(i > 0, i, 5, 1), SOME(i > 0, i, 5, 1), EVERY(MOD(i, 2) = 1, i, "
        "1, 9, 2), EVERY(i < 4, i, [1, 2, 3]), SOME(i = 2, i, {1, 2}), EVERY(i > 0, i, 3)]",
        "bi2(1)",
        "> bi2a(n) := PROG([EVERY(i > i, i, 1, n), SOME(i > 5, i, 1, n, 2), EVERY(i < 3, "
        "i, n), SOME(i = 2, i, 1/2, n, 1/2)])",
        "[bi2a(10^20), bi2a(30)]",
        "> bi3(n) := PROG([EVERY(i > 0, i, 1, n), SOME(i > x, i, 1, 3), EVERY(i > x, i, 1, "
        "3), SOME(i, i, 1, 3), EVERY(i = i, i, a, b)])",
        "[bi3(3), bi3(m)]",
        "> bi4(i) := PROG(EVERY(i > 0, i, 1, 3))",
        "[bi4(5), bi4(-1)]",
        "> bi5(i) := PROG(i := 5, [EVERY(i > 0, i, 1, 3), i])",
        "bi5()",
        "> i := 7",
        "> bi6(n) := PROG(EVERY(i < n, i, 1, 3))",
        "[bi6(4), bi6(3)]",
        "> i :=",
        "> bi7(n) := PROG(EVERY(SOME(i + j = n, j, 1, i), i, 1, 3), SOME(EVERY(j <= i, j, 1, "
        "i), i, 1, n))",
        "bi7(4)",
        "> bi8(v) := PROG(v := [0, 0, 0], EVERY(PROG(v SUB i := i, true), i, 1, 3), v)",
        "bi8()",
        "> bi9(v) := PROG(v := [x, 2, 3], v SUB 2 := 5, EVERY(v SUB i /= i, i, 1, 3))",
        "bi9()",
        "> bi10(f) := PROG(f := [1, 2], f SUB 1 := 3, MAP_LIST(f SUB i + i, i, 1, 2))",
        "bi10()",
        "> bi11(s) := PROG(s := 0, VECTOR(s :+ k, k, 1, 4), SUM(k^2, k, 1, s))",
        "bi11()",
        "> bi12(n, v) := PROG(v := VECTOR(k, k, 1, n), v SUB 2 := 0, "
        "SELECT(v SUB k > 0, k, 1, n))",
        "bi12(4)",
        "> bi13(n, v) := PROG(v := [1, 2, 3], v SUB 1 := x, EVERY(v SUB i > 0, i, 1, n))",
        "bi13(3)",
    ],
    "member": [
        "> me1(v, k) := PROG(v := [3, 1/2, true, \"a\", ?], v SUB 1 := 4, "
        "[MEMBER?(4, v), MEMBER?(3, v), MEMBER?(1/2, v), MEMBER?(true, v), "
        "MEMBER?(1, v), MEMBER?(false, v), MEMBER?(k, v)])",
        "[me1(1, 4), me1(1, 2)]",
        "> me2(k, w) := PROG(w := [1, x, [2, 3], (y + 1)^2 - y^2 - 2*y], "
        "[MEMBER?(k, w), MEMBER?(k + 1, w), MEMBER?(x, w), MEMBER?([2, 3], w), "
        "MEMBER?(k, {1, 2}), MEMBER?(k, [true]), MEMBER?(true, [1]), MEMBER?(k, 1), "
        "MEMBER?(k, [])])",
        "[me2(1), me2(0), me2(z)]",
        "> me3(v, k, c) := PROG(v := [1, 2, 3], v SUB 2 := x, c := 0, k := 0, "
        "LOOP(IF(k > 4, RETURN([c, v])), IF(MEMBER?(k, v), c :+ 1), k :+ 1, "
        "IF(k = 3, v SUB 2 := 4)))",
        "me3()",
        "> me4(v, c, i) := PROG(v := [0, 1, x, 2, y, 3], c := [], i := 0, LOOP(i :+ 1, "
        "IF(i > 6, RETURN(c)), c := APPEND(c, [MEMBER?(v SUB i, [1, 2, x])])))",
        "me4()",
        "> me5(n) := VECTOR(MEMBER?(3, [k, 3 - k]), k, 0, n)",
        "[me5(3), me5(m)]",
        "> qa := 1",
        "> me6(v) := PROG(v := [1, 'qa, 3], [MEMBER?(1, v), MEMBER?(2, v)])",
        "me6()",
        "PROG(q := 'qa, [MEMBER?(1, [q, 2]), MEMBER?(1, [1, 2]), MEMBER?(q, [1])])",
        "> mv := [5, 6]",
        "PROG(mk := 5, [MEMBER?(mk, mv), MEMBER?(mk + 2, mv), MEMBER?(mk, [mk])])",
    ],
    "roots": [
        "> ro1(a) := PROG(a := a, [SQRT(a), INTEGER?(SQRT(a)), SQRT(a)^2, SQRT(a) + 1])",
        "[ro1(0), ro1(1), ro1(16), ro1(12), ro1(1728), ro1(2), ro1(-4), ro1(-12)]",
        "[ro1(4/9), ro1(2/9), ro1(9/2), ro1(-4/9), ro1(10^40), ro1(10^41 + 3)]",
        "[ro1(12^21), ro1(2^61 - 1), ro1(3*(2^61 - 1)^2), ro1(x), ro1(x^2), ro1(1.44)]",
        "> ro2(n, k, c) := PROG(k := 1, c := 0, LOOP(IF(k > n, RETURN(c)), "
        "IF(INTEGER?(SQRT(k)), c :+ 1), k :+ 1))",
        "[ro2(100), ro2(-1)]",
        "PROG(rk := 0, rv := [], LOOP(IF(rk > 9, RETURN(rv)), "
        "rv := APPEND(rv, [SQRT(rk - 2)]), rk :+ 1))",
        "[rk, rv]",
        "> Branch := Real",
        "[ro1(-4), ro1(-4/9), ro1(8)]",
        "> Branch := Any",
        "[ro1(-4), ro1(-4/9), ro1(8)]",
        "> Branch := Principal",
        "> Precision := Approximate",
        "[ro1(16), ro1(12), ro1(4/9)]",
        "> Precision := Exact",
    ],
    "generated": [
        "> ge1(n) := PROG([VECTOR(k^2, k, n), VECTOR(k, k, 2, n), "
        "VECTOR(k, k, n, 1, -2), VECTOR(k, k, 1, n, 0), VECTOR(k, k, n, 1), "
        "VECTOR(k, k, 1/2, n), VECTOR(k, k, 1, n, 1/2), VECTOR(k, k, [3, x, 1/2]), "
        "VECTOR(k, k, {2, 1}), VECTOR(k, k), VECTOR(k, k, 1, 2, 3, 4)])",
        "[ge1(3), ge1(0), ge1(m)]",
        "> ge2(n, s) := PROG(s := 0, [VECTOR(s :+ k, k, 1, n), s, "
        "VECTOR(IF(k > 2, x, k), k, n), VECTOR(IF(k > y, 1, 0), k, n), "
        "VECTOR([k, k + 1], k, n)])",
        "ge2(4)",
        "> ge3(n) := PROG(VECTOR(IF(k = 3, RETURN(k*10), k), k, 1, n))",
        "[ge3(2), ge3(5)]",
        "> ge4(n) := PROG(VECTOR(VECTOR(i*j, j, 1, i), i, 1, n))",
        "ge4(3)",
        "> ge5(n) := PROG([EVERY(DIM(VECTOR(j, j, 1, i)) = i, i, 1, n), "
        "VECTOR(EVERY(j < i, j, 1, i - 1), i, 1, n), "
        "SOME(MEMBER?(i, VECTOR(j^2, j, 1, n)), i, 2, n)])",
        "ge5(4)",
        "> ge6(k) := PROG(VECTOR(k^2, k, 3))",
        "[ge6(5), ge6(y), ge6()]",
        "> ge7(k) := PROG(k := 10, [VECTOR(k + 1, k, 3), k])",
        "ge7()",
        "> ge8(x) := PROG(VECTOR(x^2, x, 3))",
        "[ge8(y), ge8(2)]",
        "> gk := 5",
        "> ge9(n) := PROG([VECTOR(gk + n, gk, 3), gk])",
        "ge9(1)",
        "[VECTOR(gk^2, gk, 2), PROG(VECTOR(gk + 1, gk, 2)), gk]",
        "> gk := z",
        "PROG(VECTOR(gk + 1, gk, 2))",
        "> gk :=",
        "> ge10(n, v) := PROG(v := VECTOR(0, k, n), v SUB 1 := 9, "
        "VECTOR(v SUB k + k, k, n))",
        "ge10(3)",
        "> ge11(v) := SUM(VECTOR(MOD(k*k, 7), k, 0, v - 1))",
        "[ge11(10), ge11(m)]",
        "PROG(gv := 0, [VECTOR(gv :+ k, k, 3), gv])",
        "PROG(VECTOR(PROG(gw := k, gw^2), k, 3), gw)",
        "> ge12(n) := PROG(VECTOR(1/(k - 2), k, n))",
        "ge12(3)",
        "> qa := 1",
        "> ge13(q) := PROG(q := 'qa, VECTOR(k + q, k, 2))",
        "ge13()",
    ],
    "iterated": [
        "> it1(n) := PROG([ITERATE(x + 1, x, 0, n), ITERATES(2*x, x, 1, n), "
        "ITERATE(x + 1, x, 0, -n), ITERATES(x + 1, x, 0, n/2), "
        "ITERATE(x/2 + 1, x, 2, inf), ITERATE(1/x, x, 2), "
        "ITERATES(MOD(x + 1, 3), x, 0), ITERATE(x + 1, x, 0, n, 9)])",
        "[it1(3), it1(0), it1(-2)]",
        "> it2(n) := PROG([ITERATE(IF(x > 2, y, x + 1), x, 0, n), "
        "ITERATES(x*y, x, 1, n), ITERATE([v SUB 2, v SUB 1 + v SUB 2], v, [0, 1], n), "
        "ITERATES(x/3, x, 1, n), ITERATE(x > 1, x, 2, n), "
        "ITERATE([j + k, j], [j, k], [1, 0], n)])",
        "it2(4)",
        "> it3(n, s) := PROG(s := 0, [ITERATE(PROG(s :+ x, x + 1), x, 1, n), s])",
        "it3(5)",
        "> it4(n) := PROG(ITERATE(IF(x = 3, RETURN(x*100), x + 1), x, 0, n))",
        "[it4(2), it4(9)]",
        "> it5(x) := PROG(ITERATE(x^2, x, 2, 3))",
        "[it5(5), it5(y), it5()]",
        "> it6(x) := PROG(x := 7, [ITERATE(x + 1, x, x, 2), x])",
        "it6()",
        "> it7(n) := PROG(ITERATE((x^2 - 1)/(x - 1), x, 1, n))",
        "it7(2)",
        "> it8(n) := PROG([EVERY(ITERATE(x + i, x, 0, 3) = 3*i, i, 1, n), "
        "VECTOR(ITERATE(x*i, x, 1, i), i, 1, n), ITERATE(VECTOR(x + j, j, 1, 2) SUB 1, "
        "x, 0, n)])",
        "it8(4)",
        "> it9(n) := ITERATE(MOD(x*x + 1, 1000003), x, 2, n)",
        "[it9(20), it9(0), it9(1/2)]",
        "> ix := 5",
        "> it10(n) := PROG([ITERATE(ix + n, ix, 1, 3), ix])",
        "it10(2)",
        "[ITERATE(ix*2, ix, 1, 3), PROG(ITERATES(ix*2, ix, 1, 3)), ix]",
        "> ix :=",
        "PROG(gi := 0, [ITERATE(PROG(gi :+ 1, x*2), x, 1, 4), gi])",
        "> it11(n) := PROG(ITERATE(x + 1/n, x, 0, n))",
        "[it11(3), it11(1/2)]",
        "> qa := 1",
        "> it12(q) := PROG(q := 'qa, ITERATE(x + q, x, 0, 2))",
        "it12()",
    ],
    "loops approximate": [
        "> Precision := Approximate",
        "> la1(n) := PROG([VECTOR(k/3, k, n), ITERATE(x/3 + 1, x, 1, n), "
        "MEMBER?(1/3, [1/3, 2])])",
        "la1(3)",
        "> Precision := Exact",
        "la1(3)",
        "> la2(n) := PROG(Precision := Approximate, [VECTOR(k/3, k, n), "
        "ITERATE(x/3 + 1, x, 1, n)])",
        "la2(3)",
        "> Precision := Exact",
    ],
    "commands": [
        "> co1(v) := PROG(v := SOLVE(x^2 = 4, x), v SUB 1)",
        "co1()",
        "> co2(v) := PROG(v := SOLVE(x^2 = 4, x), v SUB 1 := 0, v)",
        "co2()",
        "> co3(v) := PROG(v := [a := 1, 2], v SUB 2 := 3, v)",
        "co3()",
        "> co4(v) := PROG(v := FACTOR(360), v :+ 1, v)",
        "co4()",
        "> co5(v, k) := PROG(v := [1, 2], k := 2*(v SUB 1 := 5), [k, v])",
        "co5()",
        "> co6(s) := PROG(s := 1, DISPLAY(s), s :+ 1, WRITE(s), s)",
        "co6()",
        "> co7(v) := PROG(v := ITERATE(u + 1, u, 0, 5), v :+ 1)",
        "co7()",
        "> zz := 0",
        "> co8(k) := PROG(k := 3, ASSIGN(zz, k + 1), zz :+ k, zz)",
        "co8()",
        "> co9(v) := PROG(v := [1, 2, 3], v SUB 2 := 0, [APPEND(v, [4]), ADJOIN(0, v), "
        "REST(v), REVERSE(v), MEMBER?(0, v), v + 1, 2*v, v . v, SUM(v)])",
        "co9()",
    ],
    "omitted": [
        "> om1(a, b) := PROG(b :+ 1, [a, b])",
        "om1(1)",
        "> om2(a, b) := PROG(b SUB 1 := 1, [a, b])",
        "om2(1)",
        "> om3(a, b) := PROG(a := , a :+ 1, a)",
        "om3(1)",
        "> om4(a, v) := PROG(v := [1, 2], v :=, v SUB 1 := 3, v)",
        "om4(1)",
    ],
    "top level": [
        "PROG(k := 0, LOOP(IF(k > 5, RETURN k), k :+ 1))",
        "k",
        "PROG(s := 0, j := 0, LOOP(j :+ 1, IF(j > 10, RETURN s), s :+ MOD(j*j, 7)))",
        "> tv := [1, 2, 3]",
        "PROG(i := 1, LOOP(IF(i > 3, exit), tv SUB i := i^2, tv SUB 1 :+ i, i :+ 1), tv)",
        "[tv, tv SUB 2, tv SUB -1, tv SUB 5]",
        "tv SUB 2 := 7",
        "tv",
        "> a := 3",
        "a := a + 7",
        "a",
        "PROG(a := a + 7, a :+ 1, a)",
        "[PROG(x := 2, x^2 + y), x]",
        "PROG(w := [x, 2], w SUB 1 := w SUB 2 + 1, IF(w SUB 1 > 2, w, 0))",
        "LOOP(IF(a > 30, exit), a :* 2)",
        "> gk := 0",
        "> gs := 0",
        "LOOP(IF(gk >= 20, exit), gs :+ MOD(gk*gk, 7), gk :+ 1)",
        "[gk, gs]",
        "PROG(gk := 0, gk :+ 1)",
        "PROG(gk :+ 1, exit, 3)",
        "LOOP(gk :+ 1, IF(gk > 5, exit))",
        "[gk := 1, gk :+ 1, 2*(gk :+ 1), gk]",
        "PROG(gq := 1, LOOP(IF(gq > 10, RETURN [gq, gq := gq]), gq :* 3/2), gq)",
        "gq",
        "PROG(gz := 1, gz :+ x, gz :* 2, gz)",
        "PROG(gz := 2, gz :- 3, gz := gz, [gz, gz :+ 1])",
        "PROG(gb := 2 > 1, gb := gb AND gk > 2, gb)",
        "PROG(gn := 1, gn :epsilon Integer, gn :+ 1)",
        "PROG(gn := 1, [gn :+ 1, gn] SUB 2)",
        "PROG(gw := tv, tv SUB 1 := 100, gw SUB 2 :+ 1, [tv, gw])",
        "> gc := 0",
        "> gt := 0",
        "> gf(n) := PROG(gc :+ n, gc)",
        "PROG(gc := 0, gi := 0, LOOP(IF(gi >= 4, RETURN [gc, gi]), gc :+ gf(gi), "
        "gi :+ 1))",
        "[gc, gi]",
        "> gg(n, k) := PROG(k := 0, LOOP(IF(k >= n, RETURN gt), gt :+ k, k :+ 1))",
        "PROG(gt := 5, [gg(4), gt :+ 1])",
        "PROG(Precision := Approximate, gt :+ 1/3, gt)",
        "> Precision := Exact",
        "gt",
    ],
    "approximate": [
        "> Precision := Approximate",
        "> ap1(n, k, s) := PROG(k := 0, s := 0, LOOP(IF(k >= n, RETURN(s)), "
        "s :+ MOD(k*k, 7)/3 + 1/(k + 1), k :+ 1))",
        "ap1(30)",
        "> ap2(n, v, i, j) := PROG(v := VECTOR(1, i, 1, n), v SUB 1 := 0, i := 2, "
        "LOOP(IF(i*i > n, RETURN([v, SUM(v)])), IF(v SUB i = 1, PROG(j := i*i, "
        "LOOP(IF(j > n, exit), v SUB j := 1/3, j :+ i))), i :+ 1))",
        "ap2(30)",
        "PROG(k := 1/3, LOOP(IF(k > 2, RETURN [k, k = 7/3]), k :+ 1/3))",
        "> Precision := Mixed",
        "ap1(30)",
        "ap2(30)",
        "> ap3(v, k) := PROG(v := [1/3, SQRT(2), 2], k := 0, LOOP(k :+ 1, "
        "IF(k > 3, RETURN(v)), v SUB k :* 3, v SUB k :+ 1/7))",
        "ap3()",
        "> Precision := Exact",
    ],
    "approximate midway": [
        "> am1(v, k) := PROG(v := [1, 2, 3], k := 0, LOOP(k :+ 1, "
        "IF(k > 3, RETURN([v, k])), v SUB k :+ 1/3, IF(k = 2, Precision := Approximate), "
        "v SUB k :* 2))",
        "am1()",
        "Precision",
        "> Precision := Exact",
        "> am2(s, k) := PROG(s := 1/3, k := 5, Precision := Approximate, [s, k, s + k, "
        "IF(k > 4, s, k)])",
        "am2()",
        "> Precision := Exact",
        "> am3(s, Precision) := PROG(s := 1/3, s :+ 1/7, s)",
        "[am3(0, Approximate), am3(0, Exact)]",
    ],
    "suspended": [
        "> su1(n, k) := PROG(k := 2*n, IF(k > 5, k^2, -k))",
        "SUM(su1(j), j, 1, 4)",
        "> su2(n, v) := PROG(v := [1, 2, 3], v SUB 2 := 7, "
        "IF(n > v SUB 2, v SUB 1, v SUB 3))",
        "[VECTOR(su2(j), j, [5, 9]), su2(m)]",
        "> su3(n, k) := PROG(k := n + 1, k :+ 1, IF(n > 0, k, -k))",
        "[ITERATE(su3(u), u, 1, 2), SUBS(su3(x), x, 3)]",
    ],
    "definitions": [
        "> dc(n, k, s) := PROG(k := 0, s := 0, LOOP(IF(k >= n, RETURN(s)), s :+ k, "
        "k :+ 1))",
        "dd(y) := dc(5)*y + dc(y)",
        "dd(3)",
        "> dv(v, k) := PROG(v := [1, 2, 3], v SUB 2 := k, v)",
        "de(y) := dv(1, y) + dv(1, 4)",
        "df(y) := IF(y > 0, dc(4), dc(2))",
    ],
    "relation in a call": [
        "> hv(u, j) := PROG(u SUB j :* 2, [u, u SUB 1])",
        "> mn(p, q, a, b, k, v, w, m, c1, c2, c3) := PROG(k := 0, v := [1, 4, 9], "
        "c3 := 0, LOOP(c3 :+ 1, IF(c3 > 2, exit), v := hv(v, (1/2)), IF(a = 2, b :* a), "
        "v SUB 2 :* MOD(k > 4, 0), a := v SUB (MOD(c3, 3) + 1), Precision := Exact), v)",
        "mn(-2)",
    ],
    "nothing running": [
        "> nr(n, k) := PROG(k := n, LOOP(IF(k > 9, RETURN(k*y)), k :+ 1))",
        "nr(3)",
        "? nr(3) + x",
        "? PROG(k := 2, IF(k > z, w, k))",
        "> nv(v) := PROG(v := [1, 2], v SUB 1 := t, v)",
        "nv(0)",
        "? nv(0) + s",
        "? LOOP(q :+ 1, IF(q > 3, exit))",
    ],
}

#: Programs at top level, whose variables are globals, and functions that change
#: globals: their transcripts are also compared with the ones recorded in `RECORDED`.
TOP_LEVEL_CASES = {
    "global vectors": [
        "> gm := [[1, 2], [3, 4]]",
        "PROG(i := 1, LOOP(IF(i > 2, exit), gm SUB i SUB 1 := 10*i, "
        "gm SUB i SUB 2 :+ i, i :+ 1), gm)",
        "gm",
        "> gr := [1, 2, 3]",
        "PROG(gr SUB 4 := 9, gr SUB 0 := 9, gr SUB -1 := 9, gr SUB 2 := 5, gr)",
        "[gr, gr SUB 4 := 1, gr SUB 1 := 6, gr]",
        "PROG(gr SUB 1 := x, gr SUB 1 :+ 1, gr SUB 2 := y^2, gr SUB 3 :* 2, gr)",
        "gr",
        "> gwv := [0, 0, 0, 0]",
        "> gwf(k) := PROG(gwv SUB k :+ 10, gwv SUB k)",
        "PROG(i := 1, LOOP(IF(i > 4, RETURN gwv), gwv SUB i := i, gwv SUB i :+ gwf(i), "
        "i :+ 1))",
        "gwv",
        "> gwp(u) := PROG(u SUB 1 := 99, u)",
        "PROG(gwv SUB 2 := 7, [gwp(gwv), gwv])",
        "gwv",
        "> gwr(k) := gwv SUB k + 1",
        "PROG(i := 0, LOOP(i :+ 1, IF(i > 4, RETURN gwv), gwv SUB i := gwr(i)))",
        "PROG(sv := VECTOR(1, i, 1, 30), sv SUB 1 := 0, si := 2, LOOP(IF(si*si > 30, "
        "RETURN SUM(sv)), IF(sv SUB si = 1, PROG(sj := si*si, LOOP(IF(sj > 30, exit), "
        "sv SUB sj := 0, sj :+ si))), si :+ 1))",
        "[sv, si, sj]",
        "PROG(gk := 0, gs := 0, LOOP(IF(gk >= 30, RETURN gs), gs :+ MOD(gk*gk, 7), "
        "gk :+ 1))",
        "PROG(ga := [1, 2], gb := ga, ga SUB 1 := 5, gb SUB 2 := 6, [ga, gb])",
        "PROG(ga := [1, 2], ga SUB 1 := [3, 4], ga SUB 1 SUB 2 := 0, "
        "ga SUB 2 := ga SUB 1, ga SUB 1 SUB 1 := 9, ga)",
    ],
    "rollback": [
        "> rb := [1, 2, 3]",
        "> rs := 0",
        "> rf(n) := PROG(rs := 5, rb SUB 1 := 100, IF(n > 0, RETURN rb), 2)",
        "[rf(m), rs, rb]",
        "PROG(rb SUB 2 := 20, rb SUB 2 :+ 1, [rf(m), rb, rs])",
        "[rb, rs]",
        "> rg(n) := PROG(rb SUB 1 := 7, rb SUB 1 :+ 1, rb SUB 3 := 9, "
        "IF(n > 0, RETURN rb), 0)",
        "[rg(m), rb]",
        "PROG(rb SUB 1 := 0, [rg(m), rb, rg(1), rb])",
        "rb",
        "> ro(n, r) := PROG(rb SUB 1 := 50, r := rf(n), rb SUB 2 := 60, rb SUB 1 :+ 1, "
        "[r, rb, rs])",
        "ro(m)",
        "[rb, rs]",
        "PROG(rb SUB 3 := 1, [ro(m), rb, ro(2), rb])",
        "> rd(n) := PROG(rb SUB 1 := n, rb SUB 1 :+ 1, rs := rb SUB 1, rb SUB 1)",
        "rdd(y) := rd(y) + rd(2)",
        "[rb, rs]",
        "PROG(rb SUB 1 := 4, rdd(1), [rb, rs])",
    ],
    "statement values": [
        "PROG(x := 5, 0*(x :epsilon Integer) + exit)",
        "PROG(v := [1, 2, 3], 0*(v SUB 1 := 9) + exit)",
        "PROG(v SUB 1 := 5, 0*(v SUB 2 := 6) + exit)",
        "v",
        "PROG(xs := 5, 0*(xs :epsilon Real [7, 7]) + exit)",
        "PROG(xv := [1, 2], xv SUB 1 := 3, 0*(xv :epsilon Real [7, 7]) + exit)",
        "PROG(gk := 1, gk :+ 1)",
        "PROG(gv := [1, 2], gv SUB 2 :+ 1)",
        "[gk := 2, gv SUB 1 := gk, gv]",
    ],
    "expanding": [
        "> p := PROG(p := 5, p)",
        "p",
        "[p, p + 1]",
        "> lp := PROG(lk := 0, LOOP(IF(lk >= 3, RETURN lk*10), lk :+ 1))",
        "lp",
        "> lpf(n) := lp + n",
        "lpf(1)",
        "[lp, lk]",
        "PROG(lk := 7, [lp, lk])",
        "> lq := PROG(lv := [0, 0], lv SUB 1 := 3, lv)",
        "PROG(lv := [5, 5], [lq, lv])",
    ],
    "quotes": [
        "> qa := 1",
        "> qg := 'qa",
        "PROG(qa := 7, qg)",
        "PROG(qv := 'qa, qa := 5, qv + 1)",
        "> qf(u) := PROG(u := 'qa, qa := 9, u)",
        "qf()",
        "PROG(qw := ['qa, 2], qa := 3, qw SUB 1 + qw SUB 2)",
    ],
    "named": [
        'PROG("s1" := 1, LOOP(IF("s1" > 4, RETURN "s1"), "s1" := "s1" + 1))',
        '"s1"',
        'PROG(sk := 0, LOOP(IF(sk >= 4, exit), sk :+ 1, '
        'ASSIGN(IF(MOD(sk, 2) = 0, "sa", "sb"), sk*10)), ["sa", "sb", sk])',
        '["sa", "sb"]',
        'PROG(sk := 0, LOOP(IF(sk >= 3, RETURN ["sa", sk]), sk :+ 1, ASSIGN("sa", '
        '"sa" + sk)))',
    ],
    "session variables": [
        "> un := 5",
        "> um := -3",
        "> ut := true",
        "> uv := [1, 2, 3, 4, 5]",
        "> uw := [[1, 2], [-3, 4]]",
        "> ue := un + 1",
        "PROG(uk := 0, us := 0, LOOP(IF(uk >= un, RETURN us), uk :+ 1, "
        "us :+ uv SUB uk))",
        "> uf(k, s) := PROG(k := 0, s := 0, LOOP(IF(k >= un, RETURN s), k :+ 1, "
        "s :+ uv SUB k))",
        "uf()",
        "PROG(uk := 0, LOOP(IF(uk >= 3, RETURN [uk, um, ut, uw SUB 2 SUB 1, ue]), "
        "uk :+ 1))",
        "PROG(uv SUB 1 := 10, uk := 0, us := 0, LOOP(IF(uk >= un, RETURN [us, uv]), "
        "uk :+ 1, us :+ uv SUB uk))",
        "[uv, un]",
        "PROG(un := un + 1, un)",
        "PROG(uk := 0, LOOP(IF(uk >= 2, RETURN [ue, ut AND uk = 2]), uk :+ 1))",
        "> Precision := Approximate",
        "> ua := [1/3, -2, 0.5]",
        "PROG(uk := 0, LOOP(IF(uk >= 3, RETURN [ua, un + 1/3]), uk :+ 1))",
        "> Precision := Exact",
        "PROG(ux := un + ua SUB 1, Precision := Approximate, [un + ua SUB 1, ua, ux])",
        "> Precision := Exact",
        "> uz := [-1, 0.25, [-0.5]]",
        "PROG(uq := uz SUB 1, Precision := Approximate, [uz, uz SUB 2 + 1/3, uq, "
        "uz SUB 3 SUB 1 - um])",
        "> Precision := Exact",
    ],
    "read elsewhere": [
        "PROG(lastvar := [x, y] SUB 2, DIF(x^2*y, lastvar))",
        "PROG(d1 := (a1 - b1)/2, SOLVE(2*d1 = a1 - b1, d1))",
        "PROG(ec := 2, [APPROX(SQRT(ec) + 1), ec := ec + 1, APPROX(SQRT(ec))])",
        "PROG(vk := 3, [SUM(k^2, k, 1, vk), VECTOR(k, k, vk)])",
        "PROG(ws := [1, 2], ws SUB 1 := x, [DIF(ws, x), VARIABLES(ws)])",
    ],
    "precision midway": [
        "> pg := [1, 2, 3]",
        "PROG(pg SUB 1 :+ 1/3, Precision := Approximate, pg SUB 2 :+ 1/3, "
        "pg SUB 3 := pg SUB 1 + 1, pg)",
        "> Precision := Exact",
        "pg",
        "PROG(pk := 1/3, Precision := Approximate, pk :+ 1, Precision := Exact, "
        "pk :+ 1/7, pk)",
        "> Precision := Exact",
    ],
}

CASES.update(TOP_LEVEL_CASES)

#: The transcripts of `TOP_LEVEL_CASES` as recorded: each line, its answer and the
#: steps it took.
RECORDED = {
    "expanding": [
        ("p", "p", 4),
        ("[p, p + 1]", "[5, 6]", 4),
        ("lp", "30", 12),
        ("lpf(1)", "31", 14),
        ("[lp, lk]", "[30, 3]", 14),
        ("PROG(lk := 7, [lp, lk])", "[30, 3]", 16),
        ("PROG(lv := [5, 5], [lq, lv])", "[[3, 0], [3, 0]]", 9),
    ],
    "global vectors": [
        (
            "PROG(i := 1, LOOP(IF(i > 2, exit), gm SUB i SUB 1 := 10*i, gm SUB i SUB "
            "2 :+ i, i :+ 1), gm)",
            "[[10, 3], [20, 6]]",
            21,
        ),
        ("gm", "[[10, 3], [20, 6]]", 8),
        (
            "PROG(gr SUB 4 := 9, gr SUB 0 := 9, gr SUB -1 := 9, gr SUB 2 := 5, gr)",
            "[1, 5, 3]",
            10,
        ),
        (
            "[gr, gr SUB 4 := 1, gr SUB 1 := 6, gr]",
            "[[1, 5, 3], gr SUB 4 := 1, gr SUB 1 := 6, [6, 5, 3]]",
            9,
        ),
        (
            "PROG(gr SUB 1 := x, gr SUB 1 :+ 1, gr SUB 2 := y^2, gr SUB 3 :* 2, gr)",
            "[x + 1, y^2, 6]",
            10,
        ),
        ("gr", "[x + 1, y^2, 6]", 5),
        (
            "PROG(i := 1, LOOP(IF(i > 4, RETURN gwv), gwv SUB i := i, gwv SUB i :+ "
            "gwf(i), i :+ 1))",
            "[22, 24, 26, 28]",
            42,
        ),
        ("gwv", "[22, 24, 26, 28]", 6),
        (
            "PROG(gwv SUB 2 := 7, [gwp(gwv), gwv])",
            "[[99, 7, 26, 28], [22, 7, 26, 28]]",
            14,
        ),
        ("gwv", "[22, 7, 26, 28]", 6),
        (
            "PROG(i := 0, LOOP(i :+ 1, IF(i > 4, RETURN gwv), gwv SUB i := gwr(i)))",
            "[23, 8, 27, 29]",
            31,
        ),
        (
            "PROG(sv := VECTOR(1, i, 1, 30), sv SUB 1 := 0, si := 2, LOOP(IF(si*si > "
            "30, RETURN SUM(sv)), IF(sv SUB si = 1, PROG(sj := si*si, LOOP(IF(sj > "
            "30, exit), sv SUB sj := 0, sj :+ si))), si :+ 1))",
            "10",
            137,
        ),
        (
            "[sv, si, sj]",
            "[[0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, "
            "0, 0, 0, 0, 0, 1, 0], 6, 35]",
            37,
        ),
        (
            "PROG(gk := 0, gs := 0, LOOP(IF(gk >= 30, RETURN gs), gs :+ MOD(gk*gk, "
            "7), gk :+ 1))",
            "57",
            96,
        ),
        (
            "PROG(ga := [1, 2], gb := ga, ga SUB 1 := 5, gb SUB 2 := 6, [ga, gb])",
            "[[5, 2], [1, 6]]",
            8,
        ),
        (
            "PROG(ga := [1, 2], ga SUB 1 := [3, 4], ga SUB 1 SUB 2 := 0, ga SUB 2 := "
            "ga SUB 1, ga SUB 1 SUB 1 := 9, ga)",
            "[[9, 0], [3, 0]]",
            7,
        ),
    ],
    "named": [
        (
            "PROG(\"s1\" := 1, LOOP(IF(\"s1\" > 4, RETURN \"s1\"), \"s1\" := \"s1\" "
            "+ 1))",
            "5",
            13,
        ),
        ("\"s1\"", "5", 2),
        (
            "PROG(sk := 0, LOOP(IF(sk >= 4, exit), sk :+ 1, ASSIGN(IF(MOD(sk, 2) = "
            "0, \"sa\", \"sb\"), sk*10)), [\"sa\", \"sb\", sk])",
            "[20, 10, 4]",
            21,
        ),
        ("[\"sa\", \"sb\"]", "[20, 10]", 5),
        (
            "PROG(sk := 0, LOOP(IF(sk >= 3, RETURN [\"sa\", sk]), sk :+ 1, "
            "ASSIGN(\"sa\", \"sa\" + sk)))",
            "[20, 3]",
            15,
        ),
    ],
    "precision midway": [
        (
            "PROG(pg SUB 1 :+ 1/3, Precision := Approximate, pg SUB 2 :+ 1/3, pg SUB "
            "3 := pg SUB 1 + 1, pg)",
            "[1.333333333, 2.333333333, 2.333333333]",
            10,
        ),
        ("pg", "[4/3, 7/3, 7/3]", 5),
        (
            "PROG(pk := 1/3, Precision := Approximate, pk :+ 1, Precision := Exact, "
            "pk :+ 1/7, pk)",
            "31/21",
            7,
        ),
    ],
    "quotes": [
        ("PROG(qa := 7, qg)", "qa", 4),
        ("PROG(qv := 'qa, qa := 5, qv + 1)", "6", 5),
        ("qf()", "qa", 6),
        ("PROG(qw := ['qa, 2], qa := 3, qw SUB 1 + qw SUB 2)", "5", 6),
    ],
    "read elsewhere": [
        ("PROG(lastvar := [x, y] SUB 2, DIF(x^2*y, lastvar))", "x^2", 3),
        ("PROG(d1 := (a1 - b1)/2, SOLVE(2*d1 = a1 - b1, d1))", "d1 = (a - b)/2", 3),
        (
            "PROG(ec := 2, [APPROX(SQRT(ec) + 1), ec := ec + 1, APPROX(SQRT(ec))])",
            "[470832/195025, ec := 3, 262087/151316]",
            6,
        ),
        (
            "PROG(vk := 3, [SUM(k^2, k, 1, vk), VECTOR(k, k, vk)])",
            "[14, [1, 2, 3]]",
            11,
        ),
        (
            "PROG(ws := [1, 2], ws SUB 1 := x, [DIF(ws, x), VARIABLES(ws)])",
            "[[1, 0], [x]]",
            6,
        ),
    ],
    "rollback": [
        ("[rf(m), rs, rb]", "[rf(m), 0, [1, 2, 3]]", 18),
        (
            "PROG(rb SUB 2 := 20, rb SUB 2 :+ 1, [rf(m), rb, rs])",
            "[rf(m), [1, 21, 3], 0]",
            17,
        ),
        ("[rb, rs]", "[[1, 21, 3], 0]", 8),
        ("[rg(m), rb]", "[rg(m), [1, 21, 3]]", 17),
        (
            "PROG(rb SUB 1 := 0, [rg(m), rb, rg(1), rb])",
            "[rg(m), [0, 21, 3], [8, 21, 9], [8, 21, 9]]",
            24,
        ),
        ("rb", "[8, 21, 9]", 5),
        ("ro(m)", "[rf(m), [51, 60, 9], 0]", 21),
        ("[rb, rs]", "[[51, 60, 9], 0]", 8),
        (
            "PROG(rb SUB 3 := 1, [ro(m), rb, ro(2), rb])",
            "[[rf(m), [51, 60, 1], 0], [51, 60, 1], [[100, 60, 1], [101, 60, 1], 5], "
            "[101, 60, 1]]",
            43,
        ),
        ("rdd(y) := rd(y) + rd(2)", "rdd(y) := y + 4", 21),
        ("[rb, rs]", "[[101, 60, 1], 5]", 8),
        ("PROG(rb SUB 1 := 4, rdd(1), [rb, rs])", "[[4, 60, 1], 5]", 13),
    ],
    "session variables": [
        (
            "PROG(uk := 0, us := 0, LOOP(IF(uk >= un, RETURN us), uk :+ 1, us :+ uv "
            "SUB uk))",
            "15",
            28,
        ),
        ("uf()", "15", 30),
        (
            "PROG(uk := 0, LOOP(IF(uk >= 3, RETURN [uk, um, ut, uw SUB 2 SUB 1, "
            "ue]), uk :+ 1))",
            "[3, -3, true, -3, 6]",
            22,
        ),
        (
            "PROG(uv SUB 1 := 10, uk := 0, us := 0, LOOP(IF(uk >= un, RETURN [us, "
            "uv]), uk :+ 1, us :+ uv SUB uk))",
            "[24, [10, 2, 3, 4, 5]]",
            29,
        ),
        ("[uv, un]", "[[10, 2, 3, 4, 5], 5]", 10),
        ("PROG(un := un + 1, un)", "6", 4),
        (
            "PROG(uk := 0, LOOP(IF(uk >= 2, RETURN [ue, ut AND uk = 2]), uk :+ 1))",
            "[7, true]",
            12,
        ),
        (
            "PROG(uk := 0, LOOP(IF(uk >= 3, RETURN [ua, un + 1/3]), uk :+ 1))",
            "[[0.3333333333, -2, 0.5], 6.333333333]",
            16,
        ),
        (
            "PROG(ux := un + ua SUB 1, Precision := Approximate, [un + ua SUB 1, ua, "
            "ux])",
            "[6.333333333, [0.3333333333, -2, 0.5], 6.333333333]",
            16,
        ),
        (
            "PROG(uq := uz SUB 1, Precision := Approximate, [uz, uz SUB 2 + 1/3, uq, "
            "uz SUB 3 SUB 1 - um])",
            "[[-1, 0.25, [-0.5]], 0.5833333333, -1, 2.5]",
            14,
        ),
    ],
    "statement values": [
        ("PROG(x := 5, 0*(x :epsilon Integer) + exit)", "x := 5", 3),
        ("PROG(v := [1, 2, 3], 0*(v SUB 1 := 9) + exit)", "v := [1, 2, 3]", 3),
        ("PROG(v SUB 1 := 5, 0*(v SUB 2 := 6) + exit)", "v SUB 1 := 5", 7),
        ("v", "[5, 6, 3]", 5),
        ("PROG(xs := 5, 0*(xs :epsilon Real [7, 7]) + exit)", "xs := 5", 3),
        (
            "PROG(xv := [1, 2], xv SUB 1 := 3, 0*(xv :epsilon Real [7, 7]) + exit)",
            "xv SUB 1 := 3",
            4,
        ),
        ("PROG(gk := 1, gk :+ 1)", "gk := 2", 3),
        ("PROG(gv := [1, 2], gv SUB 2 :+ 1)", "gv SUB 2 := 3", 3),
        ("[gk := 2, gv SUB 1 := gk, gv]", "[gk := 2, gv SUB 1 := 2, [2, 3]]", 7),
    ],
}


@pytest.mark.parametrize("name", CASES)
async def test_compiling_changes_no_answer(monkeypatch, name):
    lines = CASES[name]
    statements = []
    ran = evaluate._statement

    def running(node):
        statements.append(node)
        return ran(node)

    monkeypatch.setattr(evaluate, "_statement", running)
    compiled = await transcript(monkeypatch, lines, True)
    assert statements, "nothing ran compiled"
    converted = await transcript(monkeypatch, lines, False)
    assert compiled == converted


async def test_compiling_takes_the_steps_the_limit_counts(monkeypatch):
    """The step limit stops a program at the same step either way: `cn(n)` takes
    `3*n + 8` steps."""
    lines = [
        "> cn(n, k, s) := PROG(k := 0, s := 0, LOOP(IF(k >= n, RETURN(s)), "
        "s :+ MOD(k*k, 7), k :+ 1))",
        "cn(100)",
        "cn(330)",
        "cn(331)",
    ]
    compiled = await transcript(monkeypatch, lines, True, step_limit=998)
    converted = await transcript(monkeypatch, lines, False, step_limit=998)
    assert compiled == converted
    assert [steps for _, _, steps in compiled] == [308, 998, 999]
    assert compiled[2][1] == "(error: Step limit exceeded)"


async def test_a_loop_over_values_stops_at_the_same_step(monkeypatch):
    """A `VECTOR` or `ITERATE` takes a step at each value either way, in a function and
    at top level, and the step limit stops it at the same one."""
    lines = [
        "> sv(n) := PROG(VECTOR(MOD(k*k, 7), k, 1, n))",
        "> si(n) := PROG(ITERATE(MOD(x*x + 1, 1000003), x, 2, n))",
        "> sm(n, k, c) := PROG(k := 0, c := 0, LOOP(IF(k >= n, RETURN(c)), "
        "IF(MEMBER?(MOD(k, 5), [1, 2]), c :+ 1), k :+ 1))",
        "DIM(sv(990))",
        "DIM(sv(997))",
        "si(990)",
        "si(997)",
        "sm(100)",
        "sm(300)",
        "PROG(sg := 0, DIM(VECTOR(sg :+ k, k, 1, 990)))",
        "PROG(sg := 0, DIM(VECTOR(sg :+ k, k, 1, 997)))",
        "sg",
        "[EVERY(DIM(VECTOR(j, j, i)) = i, i, 1, 40), "
        "EVERY(DIM(VECTOR(j, j, i)) = i, i, 1, 50)]",
    ]
    compiled = await transcript(monkeypatch, lines, True, step_limit=998)
    converted = await transcript(monkeypatch, lines, False, step_limit=998)
    assert compiled == converted
    answers = [answer for _, answer, _ in compiled]
    assert "(error: Step limit exceeded)" in answers
    assert "990" in answers


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "converted"])
async def test_a_definition_shown_takes_the_same_steps_every_time(monkeypatch, compiled):
    """Each call of a program in a definition shown runs in a context of its own, so
    each works out the value of the global `g` it reads: the steps don't depend on
    whether the second context happens to be made where the first one was, which runs
    repeated in one process make likely."""
    lines = [
        "> g := 1",
        "> mn(a) := PROG(a :+ g, a)",
        "dd(y) := mn(y) + mn(2)",
    ]
    counts = set()
    for _ in range(20):
        found = await transcript(monkeypatch, lines, compiled)
        assert found[0][1] == "dd(y) := y + 4"
        counts.add(found[0][2])
    assert counts == {11}


async def test_a_top_level_program_stops_at_the_same_step(monkeypatch):
    """A program at top level, whose variables are globals, stops at the step limit at
    the same step either way, and leaves the globals as it reached them."""
    lines = [
        "> k := 0",
        "> s := 0",
        "LOOP(IF(k >= 200, exit), s :+ k, k :+ 1)",
        "[k, s]",
        "LOOP(s :+ k, k :+ 1)",
        "[k, s]",
    ]
    compiled = await transcript(monkeypatch, lines, True, step_limit=998)
    converted = await transcript(monkeypatch, lines, False, step_limit=998)
    assert compiled == converted
    assert compiled[2][1] == "(error: Step limit exceeded)"


@pytest.mark.parametrize("name", TOP_LEVEL_CASES)
async def test_a_program_at_top_level_gives_the_recorded_answers(monkeypatch, name):
    assert await transcript(monkeypatch, TOP_LEVEL_CASES[name], True) == RECORDED[name]


async def test_a_command_hands_back_what_it_did_in_order(monkeypatch):
    """The assignments, definitions, declarations and settings a command made reach
    the session in the order they were last made, each global with its final value."""
    performed = []
    perform = Session._perform

    def recorded(self, result):
        performed.append([write_expression(node) for node in result.performed])
        perform(self, result)

    monkeypatch.setattr(Session, "_perform", recorded)
    session = Session(step_limit=-1)
    for line in ["ha := 0", "hb := 0", "hd := 1", "he := 2"]:
        session.author(line)
    entry = await session.simplify(
        "PROG(ha := 1, hb := [1, 2], hc :epsilon Integer, hf(x) := x + 1, ha := 3, "
        "PrecisionDigits := 12, hb SUB 1 := 5, hd :=, he := he, hg := 4, ha :+ 1, "
        "hd := hg)"
    )
    assert entry.text == "hd := 4"
    assert performed == [
        [
            "hc :epsilon Integer",
            "hf(x) := x + 1",
            "PrecisionDigits := 12",
            "hb := [5, 2]",
            "he :=",
            "hg := 4",
            "ha := 4",
            "hd := 4",
        ]
    ]
    assert list(session.assignments) == ["ha", "hb", "hd", "hg"]


async def test_a_top_level_vector_loop_stops_at_the_same_step(monkeypatch):
    """A program at top level writing elements of a global vector stops at the step
    limit at the same step either way, and the vector stays as the session had it."""
    lines = [
        "> sl := [0, 0, 0]",
        "LOOP(sl SUB 1 :+ 1, sl SUB 2 := sl SUB 1 * 2)",
        "sl",
        "PROG(sl := [0, 0], LOOP(sl SUB 2 :+ 1, IF(sl SUB 2 > 5, exit)), sl)",
    ]
    compiled = await transcript(monkeypatch, lines, True, step_limit=99)
    converted = await transcript(monkeypatch, lines, False, step_limit=99)
    assert compiled == converted
    assert compiled == [
        (lines[1], "(error: Step limit exceeded)", 100),
        ("sl", "[0, 0, 0]", 5),
        (lines[3], "[0, 6]", 17),
    ]


BENCHMARKS = [
    "> count(n,k,s):=PROG(k:=0, s:=0, LOOP(IF(k>=n, RETURN(s)), s:+MOD(k*k,7), k:+1))",
    "> sieve(n,v,i,j):=PROG(v:=VECTOR(1,i,1,n), v SUB 1:=0, i:=2, LOOP(IF(i*i>n, "
    "RETURN(SUM(v))), IF(v SUB i=1, PROG(j:=i*i, LOOP(IF(j>n, exit), v SUB j:=0, j:+i))), "
    "i:+1))",
    "> queens(n,c,q,r):=PROG(c:=0, q:=VECTOR(0,i,1,n), r:=1, LOOP(IF(r=0, RETURN(c)), "
    "q SUB r:+1, IF(q SUB r>n, [q SUB r:=0, r:-1], IF(EVERY(q SUB i/=q SUB r AND "
    "ABS(q SUB i-q SUB r)/=r-i, i, 1, r-1), IF(r=n, c:+1, r:+1)))))",
    "> mem(n,w,k,c):=PROG(w:=[3,5,7,11,13], k:=0, c:=0, LOOP(IF(k>=n, RETURN(c)), "
    "IF(MEMBER?(MOD(k,17),w), c:+1), k:+1))",
    "> vec(n):=SUM(VECTOR(MOD(k*k,7),k,0,n-1))",
    "> itr(n):=ITERATE(MOD(x*x+1,1000003),x,2,n)",
    "> sq(n,k,c):=PROG(k:=1, c:=0, LOOP(IF(k>n, RETURN(c)), IF(INTEGER?(SQRT(k)), c:+1), "
    "k:+1))",
]


async def test_the_benchmarks_of_native_values_compute_their_answers():
    session = Session()
    for line in BENCHMARKS:
        session.author(line[2:])
    assert (await session.simplify("count(20000)")).text == "39998"
    assert (await session.simplify("sieve(3000)")).text == "430"
    assert (await session.simplify("queens(6)")).text == "4"
    assert (await session.simplify("mem(20000)")).text == "5883"
    assert (await session.simplify("vec(20000)")).text == "39998"
    assert (await session.simplify("itr(20000)")).text == "803427"
    assert (await session.simplify("sq(5000)")).text == "70"


async def test_writing_an_element_of_a_long_vector_copies_no_vector(monkeypatch):
    """An element assignment writes the element in place, so the sieve builds its
    vector of `n` elements a few times in all, not once for each element it crosses
    out: about `n*LN(LN(n))` writes, which would be millions of elements copied."""
    built = [0]
    made = Vector.of.__func__

    def counted(cls, args):
        built[0] += len(args)
        return made(cls, args)

    monkeypatch.setattr(Vector, "of", classmethod(counted))
    session = Session()
    session.author(BENCHMARKS[1][2:])
    assert (await session.simplify("sieve(2000)")).text == "303"
    assert built[0] < 5 * 2000


async def test_writing_an_element_of_a_long_global_vector_copies_no_vector(monkeypatch):
    """The same sieve at top level, over a global vector, writes its elements in place
    too: past the vectors built to start and to hand the vector back, a few per
    element in all."""
    built = [0]
    made = Vector.of.__func__

    def counted(cls, args):
        built[0] += len(args)
        return made(cls, args)

    monkeypatch.setattr(Vector, "of", classmethod(counted))
    session = Session()
    sieve = (
        "PROG(v := VECTOR(1, i, 1, 2000), v SUB 1 := 0, i := 2, LOOP(IF(i*i > 2000, "
        "RETURN(SUM(v))), IF(v SUB i = 1, PROG(j := i*i, LOOP(IF(j > 2000, exit), "
        "v SUB j := 0, j :+ i))), i :+ 1))"
    )
    assert (await session.simplify(sieve)).text == "303"
    assert built[0] < 5 * 2000

