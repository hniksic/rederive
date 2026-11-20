"""The texts of the pre-defined utility library.

This module holds individual definitions in Derive's own language, most of them one-line
mathematical formulas and a few dozen of them short programs, taken from publicly
available material. It is not a copy of Derive's utility files: none of them is
reproduced with its header, comments, order or layout. The definitions are those of the
functions the Derive 6.1 online help documents for the utility files that Derive 6
loaded by itself, and of the helpers those definitions call, grouped by the file that
provided them.

The texts come from three public sources:

- the Derive User Group newsletter archive (https://www.austromath.at/dug/), whose
  published Derive 5 and 6 worksheets store, in their saved state, the definitions of
  the utility files the author's session had loaded;
- the Derive for DOS distributions published on the Internet Archive
  (https://archive.org/details/derive-for-dos-v-4,
  https://archive.org/details/derive314cas), for the files of which no worksheet copy
  exists: the text of the Derive for DOS 4.11 file of the same role;
- the Derive 6.1 online help (https://waluigibsod.github.io/derive6.1-online-help/),
  which documents each function's signature and meaning, for the few definitions written
  for Rederive.

Where the texts of each group come from:

- CombinatorialFunctions: Recovered from published Derive User Group newsletter
  worksheets (https://www.austromath.at/dug/): the saved state of 3 Derive 6 worksheets,
  for example 21/TITBITS7_new.dfw.
- NumberTheoryFunctions: Recovered from published Derive User Group newsletter
  worksheets (https://www.austromath.at/dug/): the saved state of 23 Derive 6
  worksheets, for example 100/Appendix 2A extended.dfw. Written for Rederive from the
  Derive 6.1 online help: DIVISORS, CONVERGENT, CONVERGENTS.
- EquationSolving: Recovered from published Derive User Group newsletter worksheets
  (https://www.austromath.at/dug/): the saved state of 6 Derive 6 worksheets, for
  example 121/newton.dfw.
- LinearAlgebra: Recovered from published Derive User Group newsletter worksheets
  (https://www.austromath.at/dug/): the saved state of 15 Derive 6 worksheets, for
  example 102/DGL_System.dfw.
- VectorMatrixFunctions: Recovered from published Derive User Group newsletter
  worksheets (https://www.austromath.at/dug/): the saved state of 17 Derive 6
  worksheets, for example 112/rational_hooks_impr.dfw. Written for Rederive from the
  Derive 6.1 online help: APPEND_COLUMNS.
- RationalApproximation: Recovered from published Derive User Group newsletter
  worksheets (https://www.austromath.at/dug/): the saved state of 1 Derive 6 worksheet,
  for example 08/COL_PADE.dfw. Written for Rederive from the Derive 6.1 online help:
  CHEBYSHEV_T_SERIES.
- MiscellaneousFunctions: Recovered from published Derive User Group newsletter
  worksheets (https://www.austromath.at/dug/): the saved state of 22 Derive 6
  worksheets, for example 103/Hill_new.dfw; INT_PARTS from the Derive 6.00 copy,
  52/titbit26_v6.dfw.
- DifferentiationApplications: Recovered from published Derive User Group newsletter
  worksheets (https://www.austromath.at/dug/): the saved state of 4 Derive 6 worksheets,
  for example 56/QUADOPT2.DFW. Changed from the source text: IMP_OSCULATING_CIRCLE.
- IntegrationApplications: Recovered from published Derive User Group newsletter
  worksheets (https://www.austromath.at/dug/): the saved state of 4 Derive 6 worksheets,
  for example 111/Beaudin_3.dfw.
- NumericalApproximation: Recovered from published Derive User Group newsletter
  worksheets (https://www.austromath.at/dug/): the saved state of 2 Derive 6 worksheets,
  for example 59/WEAK ACID.dfw.
- FirstOrderODEs: Recovered from published Derive User Group newsletter worksheets
  (https://www.austromath.at/dug/): the saved state of 7 Derive 6 worksheets, for
  example 07/MACKIE_V6.dfw.
- SecondOrderODEs: Recovered from published Derive User Group newsletter worksheets
  (https://www.austromath.at/dug/): the saved state of 10 Derive 6 worksheets, checked
  against the copy of the file published there as 20/fuster_ode2_derive6.mth.
- ODEApproximation: Recovered from published Derive User Group newsletter worksheets
  (https://www.austromath.at/dug/): the saved state of 18 Derive 6 worksheets, for
  example 109/SA_DERIVE_3D.dfw.
- RecurrenceEquations: The Derive for DOS 4.11 file RECUREQN.MTH, from the Derive for
  DOS distributions published on the Internet Archive
  (https://archive.org/details/derive-for-dos-v-4,
  https://archive.org/details/derive314cas).
- GraphicsFunctions: Recovered from published Derive User Group newsletter worksheets
  (https://www.austromath.at/dug/): the saved state of 21 Derive 6 worksheets, for
  example 120/Herrnhuter_new.dfw.
- BesselFunctions: Recovered from published Derive User Group newsletter worksheets
  (https://www.austromath.at/dug/): the saved state of 2 Derive 6 worksheets, for
  example 18/BESSEL_B_new.dfw.
- EllipticIntegrals: The Derive for DOS 4.11 file ELLIPTIC.MTH, from the Derive for DOS
  distributions published on the Internet Archive
  (https://archive.org/details/derive-for-dos-v-4,
  https://archive.org/details/derive314cas).
- ExponentialIntegrals: Recovered from published Derive User Group newsletter worksheets
  (https://www.austromath.at/dug/): the saved state of 10 Derive 6 worksheets, for
  example 108/GlockenspielSum4.dfw.
- FresnelIntegrals: Recovered from published Derive User Group newsletter worksheets
  (https://www.austromath.at/dug/): the saved state of 1 Derive 6 worksheet, for example
  90/Clothoids.dfw.
- HypergeometricFunctions: The Derive for DOS 4.11 file HYPERGEO.MTH, from the Derive
  for DOS distributions published on the Internet Archive
  (https://archive.org/details/derive-for-dos-v-4,
  https://archive.org/details/derive314cas). Changed from the source text:
  HYPERGEOMETRIC_SERIES.
- OrthogonalPolynomials: The Derive for DOS 4.11 file ORTH_POL.MTH, from the Derive for
  DOS distributions published on the Internet Archive
  (https://archive.org/details/derive-for-dos-v-4,
  https://archive.org/details/derive314cas).
- ProbabilityFunctions: Recovered from published Derive User Group newsletter worksheets
  (https://www.austromath.at/dug/): the saved state of 16 Derive 6 worksheets, for
  example 10/kozubik.dfw.
- ZetaFunctions: The Derive for DOS 4.11 file ZETA.MTH, from the Derive for DOS
  distributions published on the Internet Archive
  (https://archive.org/details/derive-for-dos-v-4,
  https://archive.org/details/derive314cas).

Each text is one definition, in linear notation as a math file holds it. The order of the
definitions does not matter: they call each other by name.
"""

#: The definitions of each group, by the name of the utility file that provided them.
TEXTS: dict[str, tuple[str, ...]] = {
    "CombinatorialFunctions": (
        "CATALAN(n):=COMB(2*n, n)/(n + 1)",
        "BERNOULLI(n):=IF(n = 0, 1, IF(n = 1, - 1/2, - n*ZETA(1 - n)))",
        (
            "BERNOULLI_POLY(n, x):=IF(n = 1, x - 1/2, n*LIM(SUM(k^(n - 1), k, 1, x_ "
            "- 1), x_, x) + BERNOULLI(n))"
        ),
        (
            "EULER_POLY(n, x):=2/(n + 1)*(BERNOULLI_POLY(n + 1, x) - 2^(n + "
            "1)*BERNOULLI_POLY(n + 1, x/2))"
        ),
        "EULER(n):=IF(MOD(n, 2) = 1, 0, 2^n*EULER_POLY(n, 1/2))",
        (
            "PARTS(n):=IF(n < 2, 1, FLOOR(APPROX(SUM(- "
            "1/pi*SQRT(k_/2)*SUM(IF(GCD(h_, k_) = 1, COS(pi*(SUM((i_/k_ - "
            "1/2)*(MOD(i_*h_/k_) - 1/2), i_, 1, k_ - 1) - 2*n*h_/k_)), 0), h_, 1, "
            "k_)*2*SQRT(6)*#e^(- pi*SQRT(24*n - 1)/(6*k_))*(#e^(pi*SQRT(24*n - "
            "1)/(3*k_))*(6*k_ - pi*SQRT(24*n - 1)) - 6*k_ - pi*SQRT(24*n - "
            "1))/(k_*(24*n - 1)^(3/2)), k_, 1, SQRT(n)/LOG(n, 11)), "
            "LOG(1/(4*n*SQRT(3))*#e^(pi*SQRT(2*n/3)), 10) + 5) + 1/2))"
        ),
        (
            "DISTINCT_PARTS(n):=SUM(IF(8*n - 16*i_ + 1 = FLOOR(SQRT(8*n - 16*i_ + "
            "1))^2, PARTS(i_), 0), i_, 0, n/2)"
        ),
        (
            "PARTS_LIST(n):=REVERSE((ITERATE(IF(n < DIM(p_), [p_, v_, m_], "
            "[APPEND([p_*v_], p_), APPEND(v_, [IF((3*m_^2 - m_)/2 = DIM(p_) + 1 OR "
            "(3*m_^2 + m_)/2 = DIM(p_) + 1, (-1)^(m_ + 1), 0)]), IF((3*m_^2 + m_)/2 "
            "= DIM(p_) + 1, m_ + 1, m_)]), [p_, v_, m_], [[1], [1], 1], n)) SUB 1)"
        ),
        (
            "BELL(n, b := [1], bc := [1]):=LOOP(IF(n = 0, RETURN b SUB 1), b := "
            "ADJOIN(b*bc, b), bc := ADJOIN(0, bc) + INSERT(0, bc, 0), n :- 1)"
        ),
        "RECURRENCE(u, v, v0, m):=REVERSE(ITERATE(APPEND([u], v), v, v0, m - DIM(v0)))",
        "POLYGONAL_PYRAMID(n, p, d):=COMB(n + d - 2, d - 1)*(n*(p - 2) - p + d + 2)/d",
        "POLYGONAL(n, p):=POLYGONAL_PYRAMID(n, p, 2)",
        "TRIANGULAR(n):=POLYGONAL(n, 3)",
        "TETRAHEDRAL(n):=POLYGONAL_PYRAMID(n, 3, 3)",
        "PENTATOPE(n):=POLYGONAL_PYRAMID(n, 3, 4)",
        "CENTERED_PYRAMID(n, p, d):=COMB(n + d - 3, d - 2) + p*COMB(n + d - 2, d)",
        "CENTERED(n, p):=CENTERED_PYRAMID(n, p, 2)",
        "HEX(n):=CENTERED(n, 6)",
        "STAR(n):=CENTERED(n, 12)",
        "OCTAHEDRAL(n):=CENTERED_PYRAMID(n, 4, 3)",
        "CENTERED_CUBE(n, d):=n^d + (n - 1)^d",
        "CENTERED_HEX(n, d):=n^(d + 1) - (n - 1)^(d + 1)",
        "RHOMBIC_DODECAHEDRAL(n):=CENTERED_HEX(n, 3)",
        (
            "STIRLING(n, k, k_ := 0, u_, v_ := [1]):=PROG(IF(n*k < 0, RETURN 0), "
            "IF(n > 0, PROG(IF(n <= k, RETURN MAX(0, n - k + 1)), u_ := "
            "VECTOR((-1)^(k - j_)*j_^n, j_, 0, k), n := k, LOOP(IF(k = 0, RETURN "
            "u_*v_/n!), v_ := ADJOIN(0, v_) + APPEND(v_, [0]), k :- 1))), n := -n, k "
            ":= -k, IF(n >= k, RETURN MAX(k - n + 1, 0)), IF(n = 0, RETURN 0), IF(n "
            "= 1, RETURN (k - 1)!), IF(n = k - 1, RETURN COMB(k, 2)), IF(n > k/2, "
            "PROG(LOOP(IF(k_ = k - n, exit), v_ := k_*APPEND(v_, [0]) + ADJOIN(0, "
            "v_), k_ :+ 1), LOOP(IF(k_ = n, exit), v_ := REST(k_*APPEND(v_, [0]) + "
            "ADJOIN(0, v_)), k_ :+ 1), LOOP(IF(k_ = k, RETURN FIRST(v_)), v_ := "
            "DELETE(REST(k_*APPEND(v_, [0]) + ADJOIN(0, v_)), -1), k_ :+ 1))), "
            "LOOP(IF(k_ = n, exit), v_ := k_*ADJOIN(0, v_) + APPEND(v_, [0]), k_ :+ "
            "1), LOOP(IF(k_ = k - n, exit), v_ := REST(k_*ADJOIN(0, v_) + APPEND(v_, "
            "[0])), k_ :+ 1), LOOP(IF(k_ = k, RETURN FIRST(v_)), v_ := "
            "DELETE(REST(k_*ADJOIN(0, v_) + APPEND(v_, [0])), -1), k_ :+ 1))"
        ),
        "STIRLING1(n, k):=STIRLING(-k, -n)",
        "STIRLING_CYCLE(n, k):=STIRLING(-k, -n)",
        "STIRLING2(n, k):=STIRLING(n, k)",
        "STIRLING_SUBSET(n, k):=STIRLING(n, k)",
    ),
    "NumberTheoryFunctions": (
        (
            "EXTENDED_GCD(a, b, q_, r_):=PROG(a := [a, [1, 0]], b := [b, [0, 1]], "
            "LOOP(IF(FIRST(b) = 0, exit), q_ := FIRST(a)/FIRST(b), q_ := "
            "ROUND(RE(q_)) + ROUND(IM(q_))*#i, r_ := a - q_*b, a := b, b := r_), "
            "IF(FIRST(a) = 0, RETURN [0, [0, 0]]), LOOP(IF(0 <= PHASE(FIRST(a)) < "
            "pi/2, RETURN a), a :* #i))"
        ),
        (
            "SOLVE_MOD(u, x, m, a_, b_, d_):=PROG(a_ := DIF(LHS(u) - RHS(u), x), b_ "
            ":= LIM(RHS(u) - LHS(u), x, 0), d_ := GCD(a_, m), IF(MOD(b_, d_) /= 0, "
            "RETURN []), m := m/d_, VECTOR(MOD(b_/d_*INVERSE_MOD(a_/d_, m), m) + "
            "k_*m, k_, 0, d_ - 1))"
        ),
        (
            "CRT(a, m, m_ := 1, x_ := 0):=LOOP(IF(m = [], RETURN x_), x_ := x_ + "
            "(FIRST(a) - x_)*m_*INVERSE_MOD(m_, FIRST(m)), m_ :* FIRST(m), x_ := "
            "MOD(x_, m_), a := REST(a), m := REST(m))"
        ),
        "NTH_PRIME(n):=ITERATE(NEXT_PRIME(k_), k_, 1, n)",
        (
            "PRIMEPI(x, d := 1, a := 1):=IF(d > 1, SUM(IF(PRIME?(d*n_ + a)), n_, 0, "
            "(x - a)/d), PRIMEPI(x, 6, 1) + PRIMEPI(x, 6, 5) + IF(x >= 3, 2, CHI(2, "
            "x, 3, 1)))"
        ),
        (
            "FAREY(n, a_ := 1, b_ := 1, l_ := [1]):=LOOP(IF(a_ = b_, IF(b_ = n, "
            "RETURN SORT(l_), [b_ :+ 1, a_ := 1])), IF(GCD(a_, b_) = 1, l_ := "
            "APPEND(l_, [a_/b_])), a_ :+ 1)"
        ),
        (
            "DIVISORS(n, d_ := [1]) := PROG(MAP(d_ := APPEND(VECTOR(d_*v_ SUB 1^k_, "
            "k_, 0, v_ SUB 2)), v_, FACTORS(n)), SORT(d_))"
        ),
        (
            "DIVISOR_SIGMA(k, n):=PRODUCT(SUM(v_ SUB 1^(k*j_), j_, 0, v_ SUB 2), v_, "
            "FACTORS(n))"
        ),
        "DIVISOR_TAU(n):=PRODUCT(v_ SUB 2 + 1, v_, FACTORS(n))",
        "EULER_PHI(n):=n*PRODUCT(1 - 1/v_ SUB 1, v_, FACTORS(n))",
        "MOEBIUS_MU(n):=PRODUCT(- MAX(2 - v_ SUB 2, 0), v_, FACTORS(n))",
        "SQUAREFREE(n):=EVERY(v_ SUB 2 = 1, v_, FACTORS(n))",
        "CYCLOTOMIC(n, x):=PRODUCT((x^(n/d_) - 1)^MOEBIUS_MU(d_), d_, DIVISORS(n))",
        "GEN_LUCAS(n, p, q, l0, l1):=([0, 1; -q, p]^n*[l0, l1]) SUB 1",
        "U_LUCAS(n, p, q):=GEN_LUCAS(n, p, q, 0, 1)",
        "V_LUCAS(n, p, q):=GEN_LUCAS(n, p, q, 2, p)",
        (
            "U_MOD(n, p, q, m, l_, ll_, q_ := 1):=PROG(m := IF(m > 0, m, 0, 0), l_ "
            ":= MOD(2, m), ll_ := p, OutputBase := Binary, n := "
            "NAME_TO_CODES(STRING(n)), OutputBase := Decimal, LOOP(IF(n = [], exit), "
            "IF(FIRST(n) = 48, PROG(ll_ := POLY_MOD(l_*ll_ - p*q_, m), l_ := "
            "POLY_MOD(l_^2 - 2*q_, m), q_ := MOD(q_^2, m)), PROG(l_ := "
            "POLY_MOD(l_*ll_ - p*q_, m), ll_ := POLY_MOD(ll_^2 - 2*q_*q, m), q_ := "
            "MOD(q*q_^2, m))), n := REST(n)), MOD((2*ll_ - p*l_)*INVERSE_MOD(p^2 - "
            "4*q, m), m))"
        ),
        (
            "V_MOD(n, p, q, m, l_, ll_, q_ := 1):=PROG(m := IF(m > 0, m, 0, 0), l_ "
            ":= MOD(2, m), ll_ := p, OutputBase := Binary, n := "
            "NAME_TO_CODES(STRING(n)), OutputBase := Decimal, LOOP(IF(n = [], RETURN "
            "l_), IF(FIRST(n) = 48, PROG(ll_ := POLY_MOD(l_*ll_ - p*q_, m), l_ := "
            "POLY_MOD(l_^2 - 2*q_, m), q_ := MOD(q_^2, m)), PROG(l_ := "
            "POLY_MOD(l_*ll_ - p*q_, m), ll_ := POLY_MOD(ll_^2 - 2*q_*q, m), q_ := "
            "MOD(q*q_^2, m))), n := REST(n)))"
        ),
        "LUCAS(n):=V_LUCAS(n, 1, -1)",
        "FIBONACCI(n):=GEN_LUCAS(n, 1, -1, 0, 1)",
        "PELL(n):=GEN_LUCAS(n, 2, -1, 1, 0)",
        (
            "LUCAS_LEHMER(p, m_):=PROG(m_ := 2^p - 1, SOLVE(ITERATE(MOD(s_^2 - 2, "
            "m_), s_, 4, p - 2) = 0))"
        ),
        (
            "NEXT_MERSENNE_DEGREE(n, p_):=IF(n < 2, 2, PROG(p_ := NEXT_PRIME(n), "
            "LOOP(IF(LUCAS_LEHMER(p_), RETURN p_), p_ := NEXT_PRIME(p_))))"
        ),
        "MERSENNE_LIST(n):=ITERATES(NEXT_MERSENNE_DEGREE(n_), n_, 2, n - 1)",
        (
            "MERSENNE_DEGREE(n):=[2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, "
            "607, 1279, 2203, 2281, 3217, 4253, 4423, 9689, 9941, 11213, 19937, "
            "21701, 23209, 44497, 86243, 110503, 132049, 216091, 756839, 859433, "
            "1257787, 1398269, 2976221, 3021377, 6972593, 13466917] SUB n"
        ),
        "MERSENNE(n):=2^MERSENNE_DEGREE(n) - 1",
        "PERFECT(n):=SUBST((m_ + 1)*m_/2, m_, MERSENNE(n))",
        "CONTINUED_FRACTION(u, n):=FLOOR(ITERATES(1/MOD(x_), x_, u, n))",
        (
            "CONVERGENT(x, k) := IF(INTEGER?(k) AND k >= 0, "
            "FIRST(REVERSE(CONVERGENTS(x, k))), ?)"
        ),
        (
            "JACOBI(a, b, s_ := 1, t_):=PROG(IF(b < 0 OR MOD(b, 2) = 0, RETURN ?), "
            "IF(GCD(a, b) > 1, RETURN 0), LOOP(IF(a = 1 OR b = 1, RETURN s_), a := "
            "MODS(a, b), IF(a < 0 AND MOD(b, 4) = 3, s_ := -s_), a := ABS(a), t_ := "
            "0, LOOP(IF(MOD(a, 2) = 1, exit), t_ :+ 1, a :/ 2), IF(MOD(t_, 2) = 1 "
            "AND MODS(b, 8) = \"+-\"3, s_ := -s_), t_ := a, a := b, b := t_, IF(MOD(a, "
            "4) = 3 AND MOD(b, 4) = 3, s_ := -s_)))"
        ),
        (
            "PRIME_POWER?(n, k_ := 1, t_ := 2):=LOOP(IF(t_ > n, RETURN false), "
            "IF(PRIME?(n^(1/k_)), exit), k_ :+ 1, t_ :* 2)"
        ),
        (
            "PRIMITIVE_ROOT(n, a_ := 2, n_, l_, p_):=IF(n < 5, n - 1, IF(MOD(n, 4) > "
            "0, IF(PRIME_POWER?(n/(2 - MOD(n, 2))), PROG(n_ := EULER_PHI(n), l_ := "
            "(FACTORS(n_)) SUB SUB 1, LOOP(IF(GCD(a_, n) = 1, "
            "IF(EVERY(MOD(a_^(n_/p_), n) /= 1, p_, l_), RETURN a_)), a_ :+ 1)))))"
        ),
        (
            "CONVERGENTS(x, k, a_, p_ := [1, 0], q_ := [0, 1], c_ := []) := "
            "IF(INTEGER?(k) AND k >= 0, LOOP(a_ := FLOOR(x), p_ := [a_*p_ SUB 1 + p_ "
            "SUB 2, p_ SUB 1], q_ := [a_*q_ SUB 1 + q_ SUB 2, q_ SUB 1], c_ := "
            "APPEND(c_, [p_ SUB 1/q_ SUB 1]), IF(x = a_ OR DIM(c_) > k, RETURN c_), "
            "x := 1/(x - a_)), ?)"
        ),
        (
            "SQUARE_ROOT(a, p, a_ := 2, b_, s_ := -2, t_):=PROG(IF(p = 2 OR MOD(a, "
            "p) = 0, RETURN MOD(a, p)), IF(JACOBI(a, p) = -1, RETURN ?), IF(MOD(p, "
            "4) = 3, RETURN MOD(a^((p + 1)/4), p)), IF(MOD(p, 8) = 5, IF(MODS(a^((p "
            "- 1)/4), p) = 1, RETURN MOD(a^((p + 3)/8), p), RETURN "
            "MOD(2*a*MOD((4*a)^((p - 5)/8), p), p))), LOOP(IF(JACOBI(a_, p) = -1, "
            "exit), a_ :+ 1), t_ := p - 1, LOOP(s_ :+ 1, t_ :/ 2, IF(MOD(t_, 2) = 1, "
            "exit)), a_ := MOD(a_^t_, p), b_ := MOD(a^((t_ + 1)/2), p), t_ := "
            "INVERSE_MOD(a, p), LOOP(IF(s_ < 0, RETURN b_), IF(MODS((b_^2*t_)^2^s_, "
            "p) = -1, b_ := MOD(a_*b_, p)), a_ := MOD(a_^2, p), s_ :- 1))"
        ),
    ),
    "EquationSolving": (
        (
            "NEWTON_AUX(a, x, x0, n):=ITERATES(xk - ELEMENT(ROW_REDUCE(LIM(a, x, "
            "xk))`, DIM(a) + 1), xk, x0, n)"
        ),
        "NEWTONS(u, x, x0, n):=NEWTON_AUX(APPEND(GRAD(u, x), [u])`, x, x0, n)",
        "NEWTON(u, x, x0, n):=ITERATES(x - u/DIF(u, x), x, x0, n)",
        "FIXED_POINT(g, x, x0, n):=ITERATES(LIM(g, x, xk), xk, x0, n)",
        (
            "TAYLOR_SOLVE_AUX2(x, y, x0, y0, n, aux):=y0 + SUM(LIM(ELEMENT(aux, k_), "
            "[y, x], [y0, x0])/k_!*(x - x0)^k_, k_, 1, n)"
        ),
        (
            "TAYLOR_SOLVE_AUX(x, y, x0, y0, n, v):=TAYLOR_SOLVE_AUX2(x, y, x0, y0, "
            "n, ITERATES(DIF(g_, y)*v + DIF(g_, x), g_, v, n - 1))"
        ),
        (
            "TAYLOR_SOLVE(u, x, y, x0, y0, n):=TAYLOR_SOLVE_AUX(x, y, x0, y0, n, - "
            "DIF(u, x)/DIF(u, y))"
        ),
        (
            "LAGRANGE_EXPANSION(u, x, y, x0, y0, n):=x0 + SUM((y - "
            "y0)^k_/k_!*LIM(DIF(((x - x0)/(u - y0))^k_, x, k_ - 1), x, x0), k_, 1, n)"
        ),
        (
            "TAYLOR_INVERSE(u, x, y, x0, n):=IF(LIM(DIF(u, x), x, x0) = 0, \"Taylor "
            "expansion of inverse does not exist\", LAGRANGE_EXPANSION(TAYLOR(u, x, "
            "x0, n), x, y, x0, LIM(u, x, x0), n), LAGRANGE_EXPANSION(TAYLOR(u, x, "
            "x0, n), x, y, x0, LIM(u, x, x0), n))"
        ),
    ),
    "LinearAlgebra": (
        "MINOR(M, i, j):=DELETE(DELETE(M, i)`, j)`",
        (
            "SWAP_ELEMENTS(v, i, j, t_):=PROG(t_ := v SUB i, v := REPLACE(v SUB j, "
            "v, i), v := REPLACE(t_, v, j))"
        ),
        "SCALE_ELEMENT(v, i, s):=REPLACE(s*v SUB i, v, i)",
        "SUBTRACT_ELEMENTS(v, i, j, s := 1):=REPLACE(v SUB i - s*v SUB j, v, i)",
        (
            "FORCE0(M, i, j, p):=REPLACE(M SUB i - M SUB i SUB j/M SUB p SUB j*M SUB "
            "p, M, i)"
        ),
        (
            "PIVOT(M, i, j):=VECTOR(IF(k_ <= i, M SUB k_, M SUB k_ - M SUB k_ SUB "
            "j/M SUB i SUB j*M SUB i), k_, DIM(M))"
        ),
        "COFACTOR(M, i, j):=COS(pi*(i + j))*DET(MINOR(M, i, j))",
        "ADJOINT(M):=VECTOR(VECTOR(COFACTOR(M, k_, n_), k_, DIM(M)), n_, DIM(M))",
        (
            "NULL_SPACE(M, R_, b_, t_, c_, k_, id_, i_ := 1, ns_ := []):=PROG(R_ := "
            "ROW_REDUCE(M), c_ := DIM(FIRST(M)), k_ := RANK(R_), IF(c_ = k_, RETURN "
            "[]), b_ := VECTOR(POSITION(1, R_ SUB t_), t_, 1, k_), id_ := - "
            "IDENTITY_MATRIX(c_ - k_), LOOP(IF(i_ > c_, RETURN REVERSE(ns_)), "
            "IF(MEMBER?(i_, b_), PROG(ns_ := ADJOIN(DELETE(FIRST(R_), b_), ns_), R_ "
            ":= REST(R_)), PROG(ns_ := ADJOIN(FIRST(id_), ns_), id_ := REST(id_))), "
            "i_ :+ 1))"
        ),
        (
            "EXACT_EIGENVECTOR(M, λ, r_, c_, trig_):=PROG(c_ := M - "
            "λ*IDENTITY_MATRIX(DIM(M)), r_ := NULL_SPACE(c_), IF(r_ = [], PROG(trig_ "
            ":= Trigonometry, Trigonometry := Collect, r_ := NULL_SPACE(c_), "
            "Trigonometry := trig_)), r_)"
        ),
        (
            "APPROX_EIGENVECTOR(M, μ):=SIGN(ROW_REDUCE(M - "
            "μ*IDENTITY_MATRIX(DIM(M)), VECTOR(3000*COS(pi*k_)/k_, k_, DIM(M)))` SUB "
            "(DIM(M) + 1))"
        ),
    ),
    "VectorMatrixFunctions": (
        "OUTER(v, w):=MAP_LIST(r*w, r, v)",
        "KRONECKER(i, j):=IF(i = j, 1, 0)",
        (
            "APPEND_COLUMNS args_ := IF(DIM(args_) = 1 AND VECTOR?(FIRST(args_)) AND "
            "EVERY(MATRIX?(a_), a_, FIRST(args_)), APPEND(MAP_LIST(a_`, a_, "
            "FIRST(args_)))`, IF(SOME(MATRIX?(a_), a_, args_), "
            "APPEND(MAP_LIST(IF(MATRIX?(a_), a_`, [a_]), a_, args_))`, APPEND(args_)))"
        ),
        (
            "PARTITION(v, n, d):=IF(NUMBER?(d) AND IM(d) = 0, MAP_LIST(MAP_LIST(v "
            "SUB m_, m_, n_, n_ + n - 1), n_, 1, 1 + d*FLOOR(DIM(v) - n, d), d), "
            "PARTITION(v, n, n))"
        ),
        "MATPROD(a, b, i, j):=PROG(b := b COL j, a := a SUB i, a*b)",
        "JACOBIAN(u, α):=MAP_LIST(GRAD(u SUB m_, α), m_, DIM(u))",
        "COVARIANT_METRIC_TENSOR(a):=a`*a",
        "GEOMETRY_MATRIX(α, g):=[α, MAP_LIST(SQRT(g SUB m_ SUB m_), m_, DIM(g))]",
        (
            "REMOVE_ELEMENTS(v, v_, places, m, c, r):=PROG(IF(NOT VECTOR?(v_), v_ := "
            "[v_]), places := SORT(APPEND(VECTOR(SELECT(v SUB r_ = v_ SUB s_, r_, 1, "
            "DIM(v)), s_, 1, DIM(v_)))), m := DIM(places), c := 1, r := 0, LOOP(IF(c "
            "> m, RETURN v), v := DELETE(v, places SUB c - r), c :+ 1, r :+ 1))"
        ),
        "RECTANGULAR_TO_POLAR(x, y):=[SQRT(x^2 + y^2), ATAN(y, x)]",
        "POLAR_TO_RECTANGULAR(r, θ):=[r*COS(θ), r*SIN(θ)]",
        (
            "POLAR_SUM(p, q):=[SQRT(2*p SUB 1*q SUB 1*COS(p SUB 2 - q SUB 2) + p SUB "
            "1^2 + q SUB 1^2), ATAN(p SUB 1*SIN(p SUB 2) + q SUB 1*SIN(q SUB 2), p "
            "SUB 1*COS(p SUB 2) + q SUB 1*COS(q SUB 2))]"
        ),
    ),
    "RationalApproximation": (
        (
            "PADE(y,x,x0,n,d,j_:=0,t_:=0):=PROG(LOOP(t_:+LIM(y,x,x0)*x^j_,j_:+1,IF(j_>"
            "n+d,exit),y:=DIF(y,x)/j_),t_:*SUM(STRING(k_+n)*x^k_,k_,1,d)+1,t_:-SUM(STR"
            "ING(k_)*x^k_,k_,0,n),y:=SUBST(ITERATES(DIF(u_,x),u_,t_,n+d),x,0),y:=FIRST"
            "(SOLUTIONS(y,STRING([0,...,n+d]))),SUM(y SUB "
            "(k_+1)*(x-x0)^k_,k_,0,n)/(SUM(y SUB (k_+n+1)*(x-x0)^k_,k_,1,d)+1))"
        ),
        (
            "CHEBYSHEV_T_SERIES(u, x, a, b, k, n, t_, c_, p_) := IF(k > n, \"k must "
            "be <= n\", PROG(c_ := VECTOR(((b - a)/2)^m_*LIM(DIF(u, x, m_), x, (a + "
            "b)/2)/m_!, m_, 0, n), p_ := ITERATES([w_ SUB 2, 2*t_*w_ SUB 2 - w_ SUB "
            "1], w_, [1, t_], k), EXPAND(SUBST(SUM(p_ SUB (j_ + 1) SUB "
            "1*SUM(IF(MOD(m_ - j_, 2) = 0, c_ SUB (m_ + 1)*2^(1 - m_)*COMB(m_, (m_ - "
            "j_)/2)*IF(j_ = 0, 1/2, 1), 0), m_, j_, n), j_, 0, k), t_, (2*x - a - "
            "b)/(b - a)), Rational)))"
        ),
    ),
    "MiscellaneousFunctions": (
        "RATIO_TEST(t, n):=LIM(ABS(LIM(t, n, n + 1)/t), n, inf)",
        "LIM2(u, x, y, x0, y0):=LIM(u, [y, x], [y0 + @1*(x - x0), x0])",
        (
            "LEFT_RIEMANN(u, x, a, b, n):=(b - a)/n*SUM(LIM(u, x, a + (b - a)*k_/n), "
            "k_, 0, n - 1)"
        ),
        "INT_PARTS(u, v, x):=u*INT(v, x) - INT(INT(v, x)*DIF(u, x), x)",
        (
            "DEF_INT_PARTS(u, v, x, a, b):=LIM(u*INT(v, x), x, b, -1) - LIM(u*INT(v, "
            "x), x, a, 1) - INT(INT(v, x)*DIF(u, x), x, a, b)"
        ),
        "INVERSE(u, x):=ITERATE(u, x, x, -1)",
        (
            "INT_SUBST(y, x, u):=LIM(INT(LIM(y, x, INVERSE(u, x))*DIF(INVERSE(u, x), "
            "x), x), x, u)"
        ),
        (
            "DEF_INT_SUBST(y, x, u, a, b):=INT(LIM(y, x, INVERSE(u, "
            "x))*DIF(INVERSE(u, x), x), x, LIM(u, x, a), LIM(u, x, b))"
        ),
        (
            "PROVE_SUM(t, k, a, n, s):=[LIM(t, k, a) - LIM(s, n, a), s + LIM(t, k, n "
            "+ 1) - LIM(s, n, n + 1)]"
        ),
        "POLY_COEFF(u, x, n):=SUBST(QUOTIENT(u, x^n, x), x, 0)",
        (
            "POLY_DEGREE(u, x):=PROG(IF(u = 0, RETURN -1), IF(IDENTICAL?(u, x), "
            "RETURN 1), IF(SUM?(u), RETURN MAX(VECTOR(POLY_DEGREE(v_, x), v_, "
            "TERMS(u)))), IF(POWER?(u) OR PRODUCT?(u), SUM(POLY_DEGREE(v_ SUB 1, "
            "x)*v_ SUB 2, v_, FACTORS(u)), 0))"
        ),
        "random_sign:=2*RANDOM(2) - 1",
        "RANDOM_POLY(x, d, s := 1):=x^d + SUM(random_sign*RANDOM(s)*x^j, j, 0, d - 1)",
        "RANDOM_VECTOR(n, s := 1):=VECTOR(random_sign*RANDOM(s), j, 1, n)",
        "RANDOM_MATRIX(m, n, s := 1):=VECTOR(RANDOM_VECTOR(n, s), k, 1, m)",
        "RANDOM_NORMAL(s, m):=s*SQRT(- 2*LOG(RANDOM(1)))*COS(2*pi*RANDOM(1)) + m",
        (
            "GOODNESS_OF_FIT(u, x, a):=SQRT(SUM((LIM(u, x, a SUB k SUB 1) - a SUB k "
            "SUB 2)^2, k, 1, DIM(a))/DIM(a))"
        ),
        (
            "LCC_AUX(x, y, n):=(n*x*y - SUM(x)*SUM(y))/SQRT((n*SUM(x SUB k_^2, k_, "
            "1, n) - SUM(x)^2)*(n*SUM(y SUB k_^2, k_, 1, n) - SUM(y)^2))"
        ),
        "LINEAR_CORRELATION_COEFFICIENT(a):=LCC_AUX(a` SUB 1, a` SUB 2, DIM(a))",
        (
            "POLY_INTERPOLATE_AUX(a, x, k, n):=PRODUCT((x - a SUB j_ SUB 1)/(a SUB k "
            "SUB 1 - a SUB j_ SUB 1), j_, 1, k - 1)*PRODUCT((x - a SUB j_ SUB 1)/(a "
            "SUB k SUB 1 - a SUB j_ SUB 1), j_, k + 1, n)"
        ),
        (
            "POLY_INTERPOLATE(a, x):=SUM(a SUB k_ SUB 2*POLY_INTERPOLATE_AUX(a, x, "
            "k_, DIM(a)), k_, 1, DIM(a))"
        ),
        (
            "POLY_INTERPOLATE_EXPRESSION(u, x, a):=POLY_INTERPOLATE(VECTOR([a SUB "
            "k_, LIM(u, x, a SUB k_)], k_, 1, DIM(a)), x)"
        ),
        "SQUARE_WAVE(x):=(-1)^FLOOR(x)",
    ),
    "DifferentiationApplications": (
        "CURVATURE_AUX(d1, d2):=d2/(1 + d1^2)^(3/2)",
        "CURVATURE(y, x):=CURVATURE_AUX(DIF(y, x), DIF(y, x, 2))",
        "CENTER_OF_CURVATURE_AUX(v, d1, d2):=v + [-d1, 1]*(1 + d1^2)/d2",
        (
            "CENTER_OF_CURVATURE(y, x):=CENTER_OF_CURVATURE_AUX([x, y], DIF(y, x), "
            "DIF(y, x, 2))"
        ),
        "LINE_AUX(x0, y0, d1, x):=y0 + d1*(x - x0)",
        (
            "TANGENT(y, x, x0):=LIM(LINE_AUX(x_, LIM(y, x, x_), "
            "LIM(IF(IDENTICAL?(ABS(x0), inf), y/x, DIF(y, x)), x, x0), x), x_, x0, 0)"
        ),
        (
            "PERPENDICULAR(y, x, x0):=LIM(LINE_AUX(x_, LIM(y, x, x_), - 1/LIM(DIF(y, "
            "x), x, x0), x), x_, x0, 0)"
        ),
        "CIRCLE_AUX(v0, r, α):=[v0 SUB 1 + r*COS(α), v0 SUB 2 + r*SIN(α)]",
        (
            "OSCULATING_CIRCLE(y, x, θ):=CIRCLE_AUX(CENTER_OF_CURVATURE(y, x), "
            "1/CURVATURE(y, x), θ)"
        ),
        "default1:=1",
        (
            "PARA_DIF(v, t, default1):=ITERATE(DIF(dk, t)/DIF(v SUB 1, t), dk, v SUB "
            "2, default1)"
        ),
        "PARA_CURVATURE(v, t):=CURVATURE_AUX(PARA_DIF(v, t, 1), PARA_DIF(v, t, 2))",
        (
            "PARA_CENTER_OF_CURVATURE(v, t):=CENTER_OF_CURVATURE_AUX(v, PARA_DIF(v, "
            "t, 1), PARA_DIF(v, t, 2))"
        ),
        (
            "PARA_TANGENT(v, t, t0, x):=LIM(LINE_AUX(LIM(v SUB 1, t, t_), LIM(v SUB "
            "2, t, t_), LIM(PARA_DIF(v, t, 1), t, t0), x), t_, t0, 0)"
        ),
        (
            "PARA_PERPENDICULAR(v, t, t0, x):=LIM(LINE_AUX(LIM(v SUB 1, t, t_), "
            "LIM(v SUB 2, t, t_), - 1/LIM(PARA_DIF(v, t, 1), t, t0), x), t_, t0, 0)"
        ),
        (
            "PARA_OSCULATING_CIRCLE(v, t, t0, "
            "φ):=CIRCLE_AUX(LIM(PARA_CENTER_OF_CURVATURE(v, t), t, t0), "
            "1/LIM(PARA_CURVATURE(v, t), t, t0), φ)"
        ),
        "POLAR_CURVATURE(r, θ):=PARA_CURVATURE([r*COS(θ), r*SIN(θ)], θ)",
        (
            "POLAR_CENTER_OF_CURVATURE(r, θ):=PARA_CENTER_OF_CURVATURE([r*COS(θ), "
            "r*SIN(θ)], θ)"
        ),
        "POLAR_TANGENT(r, θ, θ0, x):=PARA_TANGENT([r*COS(θ), r*SIN(θ)], θ, θ0, x)",
        (
            "POLAR_PERPENDICULAR(r, θ, θ0, x):=PARA_PERPENDICULAR([r*COS(θ), "
            "r*SIN(θ)], θ, θ0, x)"
        ),
        (
            "POLAR_OSCULATING_CIRCLE(r, θ, θ0, φ):=PARA_OSCULATING_CIRCLE([r*COS(θ), "
            "r*SIN(θ)], θ, θ0, φ)"
        ),
        "IMP_DIF_AUX(d1, x, y, n):=ITERATE(DIF(dk, x) + d1*DIF(dk, y), dk, d1, n - 1)",
        (
            "IMP_DIF(u, x, y, default1):=IMP_DIF_AUX(- DIF(u, x)/DIF(u, y), x, y, "
            "default1)"
        ),
        (
            "IMP_CURVATURE(u, x, y):=CURVATURE_AUX(IMP_DIF(u, x, y, 1), IMP_DIF(u, "
            "x, y, 2))"
        ),
        (
            "IMP_CENTER_OF_CURVATURE(u, x, y):=CENTER_OF_CURVATURE_AUX([x, y], "
            "IMP_DIF(u, x, y, 1), IMP_DIF(u, x, y, 2))"
        ),
        (
            "IMP_TANGENT(u, x, y, x0, y0):=LIM(LINE_AUX(x_, y_, LIM(IMP_DIF(u, x, y, "
            "1), [x, y], [x0, y0]), x), [x_, y_], [x0, y0])"
        ),
        (
            "IMP_PERPENDICULAR(u, x, y, x0, y0):=LIM(LINE_AUX(x_, y_, - "
            "LIM(IMP_DIF(u, x, y, 1), [x, y], [x0, y0])^(-1), x), [x_, y_], [x0, y0])"
        ),
        (
            "IMP_OSCULATING_CIRCLE(u, x, y, x0, y0, "
            "θ):=CIRCLE_AUX(LIM(IMP_CENTER_OF_CURVATURE(u, x, y), [x, y], [x0, y0]), "
            "LIM(IMP_CURVATURE(u, x, y), [x, y], [x0, y0])^(-1), θ)"
        ),
        "TANGENT_PLANE(u, v, v0):=LIM(GRAD(u, v), v, v0)*(v - v0)",
        "NORMAL_LINE(u, v, v0, t):=v0 + t*LIM(GRAD(u, v), v, v0)",
        "POLAR_DIF(r, θ, default1):=PARA_DIF([r*COS(θ), r*SIN(θ)], θ, default1)",
    ),
    "IntegrationApplications": (
        "LAPLACE(y, t, s):=INT(#e^(- s*t)*y, t, 0, inf)",
        (
            "FOURIER(y, t, t1, t2, n):=1/(t2 - t1)*INT(y, t, t1, t2) + 2/(t2 - "
            "t1)*SUM(COS(2*pi*m_*t/(t2 - t1))*INT(y*COS(2*pi*m_*t/(t2 - t1)), t, t1, "
            "t2) + SIN(2*pi*m_*t/(t2 - t1))*INT(y*SIN(2*pi*m_*t/(t2 - t1)), t, t1, "
            "t2), m_, 1, n)"
        ),
        (
            "PARA_ARC_LENGTH(v, t, t1, t2, default1):=INT(default1*SQRT(SUM(DIF(v "
            "SUB m_, t)^2, m_, 1, DIM(v))), t, t1, t2)"
        ),
        (
            "ARC_LENGTH(y, x, x1, x2, default1):=PARA_ARC_LENGTH([x, y], x, x1, x2, "
            "default1)"
        ),
        (
            "POLAR_ARC_LENGTH(r, θ, θ1, θ2, default1):=INT(default1*SQRT(r^2 + "
            "DIF(r, θ)^2), θ, θ1, θ2)"
        ),
        (
            "AREA(x, x1, x2, y, y1, y2, default1):=INT(INT(default1, y, y1, y2), x, "
            "x1, x2)"
        ),
        (
            "AREA_CENTROID(x, x1, x2, y, y1, y2, default1):=[AREA(x, x1, x2, y, y1, "
            "y2, x*default1), AREA(x, x1, x2, y, y1, y2, y*default1)]/AREA(x, x1, "
            "x2, y, y1, y2, default1)"
        ),
        (
            "AREA_INERTIA(x, x1, x2, y, y1, y2, default1):=[AREA(x, x1, x2, y, y1, "
            "y2, y^2*default1), AREA(x, x1, x2, y, y1, y2, - x*y*default1); AREA(x, "
            "x1, x2, y, y1, y2, - x*y*default1), AREA(x, x1, x2, y, y1, y2, "
            "x^2*default1)]/AREA(x, x1, x2, y, y1, y2, default1)"
        ),
        (
            "POLAR_AREA(r, r1, r2, θ, θ1, θ2, default1):=INT(INT(default1*r, r, r1, "
            "r2), θ, θ1, θ2)"
        ),
        (
            "SURFACE_AREA(z, x, x1, x2, y, y1, y2, "
            "default1):=INT(LIM(INT(default1*SQRT(1 + DIF(z, x)^2 + DIF(z, y)^2), y, "
            "LIM(y1, x, x_), LIM(y2, x, x_)), x_, x), x, x1, x2)"
        ),
        (
            "VOLUME(x, x1, x2, y, y1, y2, z, z1, z2, "
            "default1):=INT(INT(INT(default1, z, z1, z2), y, y1, y2), x, x1, x2)"
        ),
        (
            "VOLUME_CENTROID(x, x1, x2, y, y1, y2, z, z1, z2, default1):=[VOLUME(x, "
            "x1, x2, y, y1, y2, z, z1, z2, x*default1), VOLUME(x, x1, x2, y, y1, y2, "
            "z, z1, z2, y*default1), VOLUME(x, x1, x2, y, y1, y2, z, z1, z2, "
            "z*default1)]/VOLUME(x, x1, x2, y, y1, y2, z, z1, z2, default1)"
        ),
        (
            "VOLUME_INERTIA(x, x1, x2, y, y1, y2, z, z1, z2, default1):=[VOLUME(x, "
            "x1, x2, y, y1, y2, z, z1, z2, (y^2 + z^2)*default1), VOLUME(x, x1, x2, "
            "y, y1, y2, z, z1, z2, - x*y*default1), VOLUME(x, x1, x2, y, y1, y2, z, "
            "z1, z2, - x*z*default1); VOLUME(x, x1, x2, y, y1, y2, z, z1, z2, - "
            "x*y*default1), VOLUME(x, x1, x2, y, y1, y2, z, z1, z2, (x^2 + "
            "z^2)*default1), VOLUME(x, x1, x2, y, y1, y2, z, z1, z2, - "
            "y*z*default1); VOLUME(x, x1, x2, y, y1, y2, z, z1, z2, - x*z*default1), "
            "VOLUME(x, x1, x2, y, y1, y2, z, z1, z2, - y*z*default1), VOLUME(x, x1, "
            "x2, y, y1, y2, z, z1, z2, (x^2 + y^2)*default1)]/VOLUME(x, x1, x2, y, "
            "y1, y2, z, z1, z2, default1)"
        ),
        (
            "SPHERICAL_VOLUME(r, r1, r2, θ, θ1, θ2, φ, φ1, φ2, "
            "default1):=INT(INT(INT(default1*r^2*SIN(θ), φ, φ1, φ2), θ, θ1, θ2), r, "
            "r1, r2)"
        ),
        (
            "CYLINDRICAL_VOLUME(r, r1, r2, θ, θ1, θ2, z, z1, z2, "
            "default1):=INT(INT(INT(default1*r, r, r1, r2), θ, θ1, θ2), z, z1, z2)"
        ),
        (
            "VOLUME_OF_REVOLUTION(y, x, x1, x2):=CYLINDRICAL_VOLUME(r_, 0, y, θ_, 0, "
            "2*pi, x, x1, x2)"
        ),
        "AREA_OF_REVOLUTION(y, x, x1, x2):=2*pi*PARA_ARC_LENGTH([x, y], x, x1, x2, y)",
        "VOLUMEY_OF_REVOLUTION(y, x, x1, x2):=2*pi*INT(x*y, x, x1, x2)",
        "AREAY_OF_REVOLUTION(y, x, x1, x2):=2*pi*PARA_ARC_LENGTH([x, y], x, x1, x2, x)",
    ),
    "NumericalApproximation": (
        (
            "DIF_NUMERIC(y, x, x0, h, n):=IF(NUMBER(n), SUM((-1)^k*COMB(n, k)*LIM(y, "
            "x, x0 - n*h/2 + k*h), k, 0, n)/(-h)^n, DIF_NUMERIC(y, x, x0, h, 1), "
            "DIF_NUMERIC(y, x, x0, h, 1))"
        ),
        (
            "SMOOTH_VECTOR(v):=VECTOR(IF(m_ = 1 OR m_ = DIM(v), v SUB m_, (v SUB (m_ "
            "- 1) + v SUB m_ + v SUB (m_ + 1))/3), m_, DIM(v))"
        ),
        (
            "SMOOTH_COLUMN_AUX(a, j):=VECTOR(IF(m_ = j, SMOOTH_VECTOR(a SUB j), a "
            "SUB m_), m_, DIM(a))"
        ),
        "SMOOTH_COLUMN(a, j):=SMOOTH_COLUMN_AUX(a`, j)`",
        (
            "DIF_DATA_AUX(v1, v2):=(v1 SUB 1^2*v2 SUB 2 - v1 SUB 2*v2 SUB 1^2)/(v1 "
            "SUB 1*v2 SUB 1*(v1 SUB 1 - v2 SUB 1))"
        ),
        (
            "DIF_DATA(a):=VECTOR([a SUB m_ SUB 1, IF(m_ = 1, DIF_DATA_AUX(a SUB 2 - "
            "a SUB 1, a SUB 3 - a SUB 1), IF(m_ = DIM(a), DIF_DATA_AUX(a SUB (m_ - "
            "1) - a SUB m_, a SUB (m_ - 2) - a SUB m_), DIF_DATA_AUX(a SUB (m_ + 1) "
            "- a SUB m_, a SUB (m_ - 1) - a SUB m_)))], m_, DIM(a))"
        ),
        (
            "DIF2_DATA_AUX(v1, v2):=2*(v1 SUB 1*v2 SUB 2 - v1 SUB 2*v2 SUB 1)/(v1 "
            "SUB 1*v2 SUB 1*(v2 SUB 1 - v1 SUB 1))"
        ),
        (
            "DIF2_DATA(a):=VECTOR([a SUB m SUB 1, IF(m = 1, DIF2_DATA_AUX(a SUB 3 - "
            "a SUB 2, a SUB 1 - a SUB 2), IF(m = DIM(a), DIF2_DATA_AUX(a SUB m - a "
            "SUB (m - 1), a SUB (m - 2) - a SUB (m - 1)), DIF2_DATA_AUX(a SUB (m + "
            "1) - a SUB m, a SUB (m - 1) - a SUB m)))], m, DIM(a))"
        ),
        (
            "INT_DATA(a):=[ELEMENT(a`, 1), ELEMENT(ITERATES([1 + ELEMENT(v, 1), "
            "ELEMENT(v, 2) + (ELEMENT(a, ELEMENT(v, 1), 2) + ELEMENT(a, 1 + "
            "ELEMENT(v, 1), 2))*(ELEMENT(a, 1 + ELEMENT(v, 1), 1) - ELEMENT(a, "
            "ELEMENT(v, 1), 1))/2], v, [1, 0], DIM(a) - 1)`, 2)]`"
        ),
    ),
    "FirstOrderODEs": (
        (
            "LINEAR1(p, q, x, y, x0, y0):=y = (y0 + INT(q*#e^INT(p, x, x0, x), x, "
            "x0, x))/#e^INT(p, x, x0, x)"
        ),
        "LINEAR1_GEN(p, q, x, y, c):=y = (c + INT(q*#e^INT(p, x), x))/#e^INT(p, x)",
        "SEPARABLE(p, q, x, y, x0, y0):=INT(1/q, y, y0, y) = INT(p, x, x0, x)",
        "SEPARABLE_GEN(p, q, x, y, c):=INT(1/q, y) = INT(p, x) + c",
        (
            "HOMOGENEOUS(r, x, y, x0, y0):=IF(DIF(LIM(r, y, x*y), x), "
            "LIM(SEPARABLE(1/x, LIM(r, x, 1) - y, x, y, x0, y0/x0), y, y/x), "
            "\"inapplicable\", \"inapplicable\")"
        ),
        (
            "HOMOGENEOUS_GEN(r, x, y, c):=IF(DIF(LIM(r, y, x*y), x), "
            "LIM(SEPARABLE_GEN(1/x, LIM(r, x, 1) - y, x, y, c), y, y/x), "
            "\"inapplicable\", \"inapplicable\")"
        ),
        (
            "EXACT(p, q, x, y, x0, y0):=IF(DIF(q, x) = DIF(p, y), INT(LIM(q, x, x0), "
            "y, y0, y) + INT(p, x, x0, x) = 0, \"inapplicable\", \"inapplicable\")"
        ),
        "EXACT_GEN_AUX(s, q, y):=s + INT(q - DIF(s, y), y)",
        (
            "EXACT_GEN(p, q, x, y, c):=IF(DIF(q, x) = DIF(p, y), "
            "EXACT_GEN_AUX(INT(p, x), q, y) = c, \"inapplicable\", \"inapplicable\")"
        ),
        (
            "INTF_AUX(μ, p, q, x, y, x0, y0):=INT(LIM(μ*q, x, x0), y, y0, y) + "
            "INT(μ*p, x, x0, x) = 0"
        ),
        (
            "INTF_HLP(p, q, x, y, x0, y0):=IF(DIF((DIF(p, y) - DIF(q, x))/q, y), "
            "INTF_AUX(#e^INT((DIF(p, y) - DIF(q, x))/q, x), p, q, x, y, x0, y0), "
            "\"inapplicable\", \"inapplicable\")"
        ),
        (
            "INTEGRATING_FACTOR(p, q, x, y, x0, y0):=IF(DIF((DIF(q, x) - DIF(p, "
            "y))/p, x), INTF_AUX(#e^INT((DIF(q, x) - DIF(p, y))/p, y), p, q, x, y, "
            "x0, y0), INTF_HLP(p, q, x, y, x0, y0), INTF_HLP(p, q, x, y, x0, y0))"
        ),
        "INTF_GEN_AUX(μ, p, q, x, y):=EXACT_GEN_AUX(INT(μ*p, x), μ*q, y)",
        (
            "INTF_GEN_HLP(p, q, x, y, c):=IF(DIF((DIF(p, y) - DIF(q, x))/q, y), "
            "INTF_GEN_AUX(#e^INT((DIF(p, y) - DIF(q, x))/q, x), p, q, x, y) = c, "
            "\"inapplicable\", \"inapplicable\")"
        ),
        (
            "INTEGRATING_FACTOR_GEN(p, q, x, y, c):=IF(DIF((DIF(q, x) - DIF(p, "
            "y))/p, x), INTF_GEN_AUX(#e^INT((DIF(q, x) - DIF(p, y))/p, y), p, q, x, "
            "y) = c, INTF_GEN_HLP(p, q, x, y, c), INTF_GEN_HLP(p, q, x, y, c))"
        ),
        (
            "FUN_LIN_CCF(r, p, q, k, x, y, x0, y0):=LIM(SEPARABLE(1, p + q*LIM(r, y, "
            "(y - p*x - k)/q), x, y, x0, p*x0 + q*y0 + k), y, p*x + q*y + k)"
        ),
        (
            "FUN_LIN_CCF_GEN(r, p, q, k, x, y, c):=LIM(SEPARABLE_GEN(1, p + q*LIM(r, "
            "y, (y - p*x - k)/q), x, y, c), y, p*x + q*y + k)"
        ),
        (
            "LIN_FRAC_AUX(r, p, q, x, y, x0, y0):=LIM(HOMOGENEOUS(LIM(r, [y, x], [y "
            "+ q, x + p]), x, y, x0 - p, y0 - q), [y, x], [y - q, x - p])"
        ),
        (
            "LIN_FRAC(r, a, b, d, p, q, k, x, y, x0, y0):=LIN_FRAC_AUX(r, (b*k - "
            "d*q)/(a*q - b*p), (d*p - a*k)/(a*q - b*p), x, y, x0, y0)"
        ),
        (
            "LIN_FRAC_AUX_GEN(r, p, q, x, y, c):=LIM(HOMOGENEOUS_GEN(LIM(r, [y, x], "
            "[y + q, x + p]), x, y, c), [y, x], [y - q, x - p])"
        ),
        (
            "LIN_FRAC_GEN(r, a, b, d, p, q, k, x, y, c):=LIN_FRAC_AUX_GEN(r, (b*k - "
            "d*q)/(a*q - b*p), (d*p - a*k)/(a*q - b*p), x, y, c)"
        ),
        (
            "GEN_HOM_AUX(k, r, x, y, x0, y0):=IF(DIF(k, x) = 0 AND DIF(k, y) = 0, "
            "LIM(SEPARABLE(1/x, y*(k + LIM(r*x/y, y, y/x^k)), x, y, x0, y0*x0^k), y, "
            "y*x^k), \"inapplicable\", \"inapplicable\")"
        ),
        (
            "GEN_HOM(r, x, y, x0, y0):=GEN_HOM_AUX(x*DIF(x*r/y, x)/(y*DIF(x*r/y, "
            "y)), r, x, y, x0, y0)"
        ),
        (
            "GEN_HOM_AUX_GEN(k, r, x, y, c):=IF(DIF(k, x) = 0 AND DIF(k, y) = 0, "
            "LIM(SEPARABLE_GEN(1/x, y*(k + LIM(r*x/y, y, y/x^k)), x, y, c), y, "
            "y*x^k), \"inapplicable\", \"inapplicable\")"
        ),
        (
            "GEN_HOM_GEN(r, x, y, c):=GEN_HOM_AUX_GEN(x*DIF(x*r/y, x)/(y*DIF(x*r/y, "
            "y)), r, x, y, c)"
        ),
        (
            "ALMOST_LIN(r, b, p, q, x, y, x0, y0):=IF(DIF(DIF(b, y)/r, y), "
            "LIM(LINEAR1(p*DIF(b, y)/r, q*DIF(b, y)/r, x, y, x0, LIM(b, y, y0)), y, "
            "b), \"inapplicable\", \"inapplicable\")"
        ),
        (
            "ALMOST_LIN_GEN(r, b, p, q, x, y, c):=IF(DIF(DIF(b, y)/r, y), "
            "LIM(LINEAR1_GEN(p*DIF(b, y)/r, q*DIF(b, y)/r, x, y, c), y, b), "
            "\"inapplicable\", \"inapplicable\")"
        ),
        (
            "TESTM2(r, s, x, y):=IF(s SUB 1*r SUB 1 + s SUB 2*r SUB 2 - r SUB 3 = 0, "
            "x^s SUB 1*y^s SUB 2, \"inapplicable\", \"inapplicable\")"
        ),
        (
            "TESTM1(r, x, y):=TESTM2(r, ELEMENT(ROW_REDUCE([LIM(r, [y, x], [5, 2]), "
            "LIM(r, [y, x], [3, 7])])`, 3), x, y)"
        ),
        (
            "MONOMIAL_TEST(p, q, x, y):=TESTM1([y*q, - x*p, x*y*(DIF(p, y) - DIF(q, "
            "x))], x, y)"
        ),
        (
            "CLAIRAUT(p, q, x, y, v, c):=[LIM(p - q, v, c) = 0, x*LIM(DIF(LIM(p, y, "
            "x*v - y), y), y, x*v - y) - DIF(q, v) = 0]"
        ),
        "INDEP_OF(f, x):=IF(DIM(SELECT(i_ = x, i_, VARIABLES(f))), true, false)",
        (
            "SEPARATE_AUX(f, x, y):=ITERATE(IF(INDEP_OF(f SUB n, y), [p*f SUB n, q, "
            "n + 1], IF(INDEP_OF(f SUB n, x), [p, q*f SUB n, n + 1], [0, 0, n + "
            "1])), [p, q, n], [1, 1, 1], DIM(f))"
        ),
        (
            "SEPARATE(u, x, y):=SEPARATE_AUX(VECTOR(v_ SUB 1^v_ SUB 2, v_, "
            "FACTORS(u)), x, y)"
        ),
        (
            "SEP_GEN_HELP(s, x, y, c):=IF(s SUB 2, \"inapplicable\", SEPARABLE_GEN(s "
            "SUB 1, s SUB 2, x, y, c), SEPARABLE_GEN(s SUB 1, s SUB 2, x, y, c))"
        ),
        "SEP_GEN(p, q, x, y, c):=SEP_GEN_HELP(SEPARATE(- p/q, x, y), x, y, c)",
        (
            "DSOLVE1_GEN(p, q, x, y, c, a_):=IF(\"inapplicable\" = a_ := "
            "INTEGRATING_FACTOR_GEN(p, q, x, y, c), IF(\"inapplicable\" = a_ := "
            "HOMOGENEOUS_GEN(- p/q, x, y, c), IF(\"inapplicable\" = a_ := SEP_GEN(p, "
            "q, x, y, c), GEN_HOM_GEN(- p/q, x, y, c), a_, a_), a_, a_), a_, a_)"
        ),
        (
            "SEP_HELP(s, x, y, x0, y0):=IF(s SUB 2, \"inapplicable\", SEPARABLE(s SUB "
            "1, s SUB 2, x, y, x0, y0), SEPARABLE(s SUB 1, s SUB 2, x, y, x0, y0))"
        ),
        "SEP(p, q, x, y, x0, y0):=SEP_HELP(SEPARATE(- p/q, x, y), x, y, x0, y0)",
        (
            "DSOLVE1(p, q, x, y, x0, y0, a_):=IF(\"inapplicable\" = a_ := "
            "INTEGRATING_FACTOR(p, q, x, y, x0, y0), IF(\"inapplicable\" = a_ := "
            "HOMOGENEOUS(- p/q, x, y, x0, y0), IF(\"inapplicable\" = a_ := SEP(p, q, "
            "x, y, x0, y0), GEN_HOM(- p/q, x, y, x0, y0), a_, a_), a_, a_), a_, a_)"
        ),
        (
            "BERNOULLI_ODE(p, q, k, x, y, x0, y0):=LINEAR1((1 - k)*p, (1 - k)*q, x, "
            "y^(1 - k), x0, y0^(1 - k))"
        ),
        (
            "BERNOULLI_ODE_GEN(p, q, k, x, y, c):=LINEAR1_GEN((1 - k)*p, (1 - k)*q, "
            "x, y^(1 - k), c)"
        ),
    ),
    "SecondOrderODEs": (
        "LIN_SOLVE(a, b):=ELEMENT(ROW_REDUCE(a, b)`, 1 + DIM(a))",
        "DISC_AUX(p, q, x):=p^2/4 - q + DIF(p, x)/2",
        (
            "LIN2_HOM_AUX(d, x):=IF(d < 0, [COS(SQRT(-d)*x), SIN(SQRT(-d)*x)], "
            "[#e^(SQRT(d)*x), #e^(- SQRT(d)*x)], [#e^(SQRT(d)*x), #e^(- SQRT(d)*x)])"
        ),
        "LIN2_HOM(d, x):=IF(d = 0, [1, x], LIN2_HOM_AUX(d, x), LIN2_HOM_AUX(d, x))",
        (
            "LIN2_PARTIC(h, r, x):=h*INT([- ELEMENT(h, 2), ELEMENT(h, 1)]*r/DET([h, "
            "DIF(h, x)]), x)"
        ),
        "LIN2_AUX2(h, r, x, c):=c*h + LIN2_PARTIC(h, r, x)",
        (
            "LIN2_AUX(d, p, r, x, c):=IF(DIF(d, x) = 0, LIN2_AUX2(LIN2_HOM(d, "
            "x)*#e^(- INT(p, x)/2), r, x, c), \"inapplicable\", \"inapplicable\")"
        ),
        "DSOLVE2(p, q, r, x, c1, c2):=LIN2_AUX(DISC_AUX(p, q, x), p, r, x, [c1, c2])",
        (
            "LIN2_BV_AUX3(h, p, x, x0, y0, x2, y2):=p + h*LIN_SOLVE([LIM(h, x, x0), "
            "LIM(h, x, x2)], [y0 - LIM(p, x, x0), y2 - LIM(p, x, x2)])"
        ),
        (
            "LIN2_BV_AUX2(h, r, x, x0, y0, x2, y2):=LIN2_BV_AUX3(h, LIN2_PARTIC(h, "
            "r, x), x, x0, y0, x2, y2)"
        ),
        (
            "LIN2_BV_AUX(d, p, r, x, x0, y0, x2, y2):=IF(DIF(d, x) = 0, "
            "LIN2_BV_AUX2(LIN2_HOM(d, x)*#e^(- INT(p, x)/2), r, x, x0, y0, x2, y2), "
            "\"inapplicable\", \"inapplicable\")"
        ),
        (
            "DSOLVE2_BV(p, q, r, x, x0, y0, x2, y2):=LIN2_BV_AUX(DISC_AUX(p, q, x), "
            "p, r, x, x0, y0, x2, y2)"
        ),
        (
            "LIN2_IV_AUX3(h, p, x, x0, y0, v0):=p + h*LIN_SOLVE(LIM([h, DIF(h, x)], "
            "x, x0), LIM([y0 - p, v0 - DIF(p, x)], x, x0))"
        ),
        (
            "LIN2_IV_AUX2(h, r, x, x0, y0, v0):=LIN2_IV_AUX3(h, LIN2_PARTIC(h, r, "
            "x), x, x0, y0, v0)"
        ),
        (
            "LIN2_IV_AUX(d, p, r, x, x0, y0, v0):=IF(DIF(d, x) = 0, "
            "LIN2_IV_AUX2(LIN2_HOM(d, x)*#e^(- INT(p, x)/2), r, x, x0, y0, v0), "
            "\"inapplicable\", \"inapplicable\")"
        ),
        (
            "DSOLVE2_IV(p, q, r, x, x0, y0, v0):=LIN2_IV_AUX(DISC_AUX(p, q, x), p, "
            "r, x, x0, y0, v0)"
        ),
        (
            "AUTONOMOUS_CONSERVATIVE(q, x, y, x0, y0, v0):=x = x0 + IF(v0 = 0, 1, "
            "SIGN(v0))*INT((2*INT(q, y, y0, y) + v0^2)^(- 1/2), y, y0, y)"
        ),
        (
            "LIOUVILLE(p, q, x, y, c1, c2):=INT(#e^INT(q, y), y) - c1 + "
            "c2*INT(#e^INT(-p, x), x) = 0"
        ),
        "AUTONOMOUS(r, v):=r/v",
        "EXACT2_AUX3(r, u, x):=r + INT(u - DIF(r, x), x)",
        "EXACT2_AUX2(r, x, y, v):=EXACT2_AUX3(INT(DIF(r, v), y), LIM(r, v, 0), x)",
        (
            "EXACT2_AUX1(p, q, x, y, v):=p + EXACT2_AUX2(q - DIF(p, x) - v*DIF(p, "
            "y), x, y, v)"
        ),
        (
            "EXACT2(p, q, x, y, v, c):=IF((v*DIF(p, y, 2) + 2*DIF(DIF(p, x), y) - "
            "DIF(DIF(q, y), v))*v + DIF(p, x, 2) - DIF(DIF(q, x), v) + DIF(q, y) = 0 "
            "AND v*DIF(DIF(p, y), v) + DIF(DIF(p, x), v) + 2*DIF(p, y) - DIF(q, v, "
            "2) = 0, EXACT2_AUX1(INT(p, v), q, x, y, v) = c, \"inapplicable\", "
            "\"inapplicable\")"
        ),
    ),
    "ODEApproximation": (
        (
            "TAYLOR_ODE1(r, x, y, x0, y0, n, v_):=ELEMENT(ITERATE([v_ SUB 1 + 1, "
            "DIF(v_ SUB 2, x) + r*DIF(v_ SUB 2, y), v_ SUB 3 + (x - x0)^v_ SUB "
            "1*LIM(v_ SUB 2, [x, y], [x0, y0])/v_ SUB 1!], v_, [1, r, y0], n), 3)"
        ),
        "PICARD(r, p, x, y, x0, y0):=y0 + INT(LIM(r, y, p), x, x0, x)",
        (
            "TAYLOR_ODES(r, x, y, x0, y0, n):=ELEMENT(ITERATE([v_ SUB 1 + 1, DIF(v_ "
            "SUB 2, x) + r*GRAD(v_ SUB 2, y), v_ SUB 3 + (x - x0)^v_ SUB "
            "1*LIM(LIM(v_ SUB 2, y, y0), x, x0)/v_ SUB 1!], v_, [1, r, y0], n), 3)"
        ),
        (
            "TAYLOR_ODE2_AUX(x, y, v, x0, y0, v0, n, aux):=y0 + v0*(x - x0) + "
            "SUM(LIM(aux SUB (k_ - 1), [v, y, x], [v0, y0, x0])/k_!*(x - x0)^k_, k_, "
            "2, n)"
        ),
        (
            "TAYLOR_ODE2(r, x, y, v, x0, y0, v0, n):=TAYLOR_ODE2_AUX(x, y, v, x0, "
            "y0, v0, n, ITERATES(DIF(g_, v)*r + DIF(g_, y)*v + DIF(g_, x), g_, r, n "
            "- 2))"
        ),
        "EXTRACT_2_ELEMENTS(v, p, q):=[v SUB p, v SUB q]",
        "EXTRACT_2_COLUMNS(a, c1, c2):=EXTRACT_2_ELEMENTS(a`, c1, c2)`",
        "RK_AUX3(p, v, u_, c1, c2, c3):=(c1 + LIM(p, v, u_ + c3) + 2*(c2 + c3))/6",
        "RK_AUX2(p, v, u_, c1, c2):=RK_AUX3(p, v, u_, c1, c2, LIM(p, v, u_ + c2/2))",
        "RK_AUX1(p, v, u_, c1):=RK_AUX2(p, v, u_, c1, LIM(p, v, u_ + c1/2))",
        (
            "RK_AUX0(p, v, v0, n):=ITERATES(u_ + RK_AUX1(p, v, u_, LIM(p, v, u_)), "
            "u_, v0, n)"
        ),
        "RK(r, v, v0, h, n):=RK_AUX0(h*APPEND([1], r), v, v0, n)",
        (
            "SEG(rc, x, y, δx, δy):=IF(δy < ABS(rc)*δx, [x - δy/rc, y - δy; x + "
            "δy/rc, y + δy], [x - δx, y - rc*δx; x + δx, y + rc*δx], ?)"
        ),
        (
            "DIRECTION_FIELD(r, x, x0, xm, m, y, y0, yn, "
            "n):=VECTOR(VECTOR(SEG(LIM2(r, x, y, x0 + j*(xm - x0)/m, y0 + k*(yn - "
            "y0)/n), x0 + j*(xm - x0)/m, y0 + k*(yn - y0)/n, (xm - x0)/(4*m), (yn - "
            "y0)/(4*n)), j, 0, m), k, 0, n)"
        ),
        (
            "EULER_ODE(r, x, y, x0, y0, h, n):=ITERATES(v_ + h*[1, LIM(r, [x, y], "
            "v_)], v_, [x0, y0], n)"
        ),
    ),
    "RecurrenceEquations": (
        (
            "LIN1_DIFFERENCE(p,q,x,x0,y0):=PRODUCT(p,x,x0,x-1)*(y0+SUM(q/PRODUCT(p,x,x"
            "0,x),x,x0,x-1))"
        ),
        (
            "RECURRENCE1(r,x,y,x0,y0,n):=ITERATES([1+ELEMENT(v_,1),LIM(r,[x,y],v_)],v_"
            ",[x0,y0],n)"
        ),
        (
            "GEOMETRIC1(k,p,q,x,x0,y0):=LIM(LIN1_DIFFERENCE(LIM(p,x,k^x),LIM(q,x,k^x),"
            "x,LOG(x0,k),y0),x,LOG(x,k))"
        ),
        "CLAIRAUT_DIF(p,q,d,x,y,c):=LIM(p-q,d,c)=0",
        "LIN2_CCF_HOM_AUX(b,x):=IF(b<0, -COS(pi*x)*(-b)^x, b^x)",
        (
            "LIN2_CCF_HOM(p,q,d,x):= IF(d=0, [(-p/2)^x, x*(-p/2)^x], "
            "IF(d>0,[LIN2_CCF_HOM_AUX((-p+SQRT(d))/2,x), "
            "LIN2_CCF_HOM_AUX((-p-SQRT(d))/2,x)], "
            "((p^2-d)/4)^(x/2)*[SIN(x*ATAN(SQRT(-d),-p)),COS(x*ATAN(SQRT(-d),-p))]), "
            "((p^2-d)/4)^(x/2)*[SIN(x*ATAN(SQRT(-d),-p)),COS(x*ATAN(SQRT(-d),-p))])"
        ),
        (
            "LIN2_CCF_PARTIC(h,r,x):= h . SUM(r "
            "LIM([-ELEMENT(h,2),ELEMENT(h,1)]/DET([h,LIM(h,x,x+1)]),x,x+1),x)"
        ),
        "LIN2_CCF_AUX(h,r,x,c):= c . h + LIN2_CCF_PARTIC(h,r,x)",
        (
            "LIN2_CCF(p,q,r,x,c1,c2):= "
            "LIN2_CCF_AUX(LIN2_CCF_HOM(p,q,p^2-4q,x),r,x,[c1,c2])"
        ),
        (
            "LIN2_CCF_BV_AUX2(h,p,x,x0,y0,x2,y2):= p + h . "
            "LIN_SOLVE([LIM(h,x,x0),LIM(h,x,x2)],[y0-LIM(p,x,x0),y2-LIM(p,x,x2)])"
        ),
        (
            "LIN2_CCF_BV_AUX(h,r,x,x0,y0,x2,y2):= "
            "LIN2_CCF_BV_AUX2(h,LIN2_CCF_PARTIC(h,r,x),x,x0,y0,x2,y2)"
        ),
        (
            "LIN2_CCF_BV(p,q,r,x,x0,y0,x2,y2):= "
            "LIN2_CCF_BV_AUX(LIN2_CCF_HOM(p,q,p^2-4q,x),r,x,x0,y0,x2,y2)"
        ),
    ),
    "GraphicsFunctions": (
        "ROTATE_X(α):=[1, 0, 0; 0, COS(α), - SIN(α); 0, SIN(α), COS(α)]",
        "ROTATE_Y(β):=[COS(β), 0, SIN(β); 0, 1, 0; - SIN(β), 0, COS(β)]",
        "ROTATE_Z(σ):=[COS(σ), - SIN(σ), 0; SIN(σ), COS(σ), 0; 0, 0, 1]",
        "ISOMETRIC(v):=[v SUB 2 - v SUB 1, v SUB 3 - (v SUB 1 + v SUB 2)/2]",
        (
            "GRID(u, s, s0, sm, m, t, t0, tn, n):=VECTOR(VECTOR(LIM(u, [s, t], [s0 + "
            "j*(sm - s0)/m, t0 + k*(tn - t0)/n]), j, 0, m), k, 0, n)"
        ),
        (
            "ISOMETRICS(v, s, s0, sm, m, t, t0, tn, n):=GRID(ISOMETRIC(v), s, s0, "
            "sm, m, t, t0, tn, n)"
        ),
        "COPROJECTION(v):=VECTOR(VECTOR(u_ SUB n_, u_, v), n_, DIM(v SUB 1))",
        "SPHERE(r, θ, φ):=r*[SIN(φ)*COS(θ), SIN(φ)*SIN(θ), COS(φ)]",
        "CYLINDER(r, θ, z):=[r*COS(θ), r*SIN(θ), z]",
        "CONE(α, θ, z):=[z*SIN(α)*COS(θ), z*SIN(α)*SIN(θ), z]",
        (
            "TORUS(rc, rs, θ, φ):=[(rc - rs*SIN(φ))*COS(θ), (rc - rs*SIN(φ))*SIN(θ), "
            "rs*COS(φ)]"
        ),
        "RE_IM(w):=[RE(w), IM(w)]",
        (
            "RAYS(w, z, z00, zmn, m, n, x, y):=GRID([x, y; [x, y], [x, y] + "
            "RE_IM(LIM(w, z, x + #i*y))], x, RE(z00), RE(zmn), m, y, IM(z00), "
            "IM(zmn), n)"
        ),
        (
            "HORIZONTALS(w, z, z00, zmn, m, n, x, y):=GRID(RE_IM(LIM(w, z, x + "
            "#i*y)), x, RE(z00), RE(zmn), m, y, IM(z00), IM(zmn), n)"
        ),
        (
            "ARCS(w, z, r0, rm, m, θ0, θn, n, r, θ):=GRID(RE_IM(LIM(w, z, "
            "r*#e^(#i*θ))), θ, θ0, θn, n, r, r0, rm, m)"
        ),
        (
            "GRID_LINES(a, b, c := 1, o := [0, 0, 0]):=VECTOR([[a, i, o SUB 3; b, i, "
            "o SUB 3]; [a, o SUB 2, i; b, o SUB 2, i]; [o SUB 1, a, i; o SUB 1, b, "
            "i]; [i, a, o SUB 3; i, b, o SUB 3]; [i, o SUB 2, a; i, o SUB 2, b]; [o "
            "SUB 1, i, a; o SUB 1, i, b]], i, a, b, c)"
        ),
        (
            "POLYGON_FILL(u):=IF(DIM(u SUB 1) = 2, [APPEND(u, [u SUB 1]), "
            "VECTOR(SUBST(SUBST(((x <= 1 AND y <= 1 - x) AND 0 <= x) AND 0 <= y, [x, "
            "y], [x, y]*[u SUB (i + 1) - u SUB 1, u SUB (i + 2) - u SUB 1]^(-1)), "
            "[x, y], [x, y] - u SUB 1), i, 1, DIM(u) - 2)], [APPEND(u, [u SUB 1]), "
            "VECTOR([[0, 0, 0; 1, 0, 0]*[u SUB (i + 1) - u SUB 1, u SUB (i + 2) - u "
            "SUB 1, CROSS(u SUB (i + 1) - u SUB 1, u SUB (i + 2) - u SUB 1)] + [u "
            "SUB 1, u SUB 1], [0, 0, 0; 0, 1, 0]*[u SUB (i + 1) - u SUB 1, u SUB (i "
            "+ 2) - u SUB 1, CROSS(u SUB (i + 1) - u SUB 1, u SUB (i + 2) - u SUB "
            "1)] + [u SUB 1, u SUB 1]], i, 1, DIM(u) - 2)])"
        ),
        "AreaUnderCurve(u, x, a, b, y):=[u, a <= x <= b AND 0 < y < u]",
        "POLAR(r, θ):=r*[COS(θ), SIN(θ)]",
        (
            "GRID_POINTS(a, b, c := 1, o := [0, 0, 0]):=VECTOR(VECTOR([[s, t, o SUB "
            "3]; [s, o SUB 2, t]; [o SUB 1, s, t]], s, a, b, c), t, a, b, c)"
        ),
        "AreaOverCurve(u, x, a, b, y):=[u, a <= x <= b AND u < y < 0]",
        (
            "AreaBetweenCurves(u, v, x, a, b, y):=[u, v, a <= x <= b AND (v - y)*(y "
            "- u) > 0]"
        ),
        (
            "PlotInt(u, x, a, b, y):=IF(a < b, [u, a <= x <= b AND 0 < y < u, a <= x "
            "<= b AND u < y < 0], [u, b <= x <= a AND u < y < 0, b <= x <= a AND 0 < "
            "y < u])"
        ),
    ),
    "BesselFunctions": (
        "BESSEL_J2(v):=v/(v SUB 1 + 2*SUM(v SUB (2*k_ + 1), k_, 1, DIM(v)/2 - 1))",
        (
            "BESSEL_J1(n, z):=ITERATE(APPEND([(2*(n - DIM(v_)) + 4)/z*v_ SUB 1 - v_ "
            "SUB 2], v_), v_, [1, 0], n)"
        ),
        "BESSEL_J_LIST(n, z):=BESSEL_J2(BESSEL_J1(n, z))",
        (
            "BESSEL_J(n, z):=IF(z = 0, IF(n = 0, 1, 0), IF(MOD(n) = 0, IF(n < 0, "
            "(-1)^n, 1)*(BESSEL_J_LIST(ABS(n) + FLOOR(PrecisionDigits/2 + 4*ABS(z)) "
            "+ 4, z)) SUB (ABS(n) + 1), (z/2)^n/(SQRT(pi)*GAMMA(n + "
            "1/2))*INT(COS(z*COS(t_))*SIN(t_)^(2*n), t_, 0, pi)))"
        ),
        "BESSEL_J_SERIES(n, z, m):=(z/2)^n*SUM((- z^2/4)^k_/(k_!*(n + k_)!), k_, 0, m)",
        "BESSEL_J_ASYMP(n, z):=SQRT(2/(pi*z))*COS(z - (n + 1/2)*pi/2)",
        (
            "BESSELY0_AUX(z):=ITERATE(IF(b_ SUB 2 = b_ SUB 3, b_, [b_ SUB 1 + 1, b_ "
            "SUB 3, b_ SUB 3 + (- z^2/4)^b_ SUB 1/b_ SUB 1!^2*SUM(1/m_, m_, 1, b_ "
            "SUB 1)]), b_, [1, 1, 0])"
        ),
        (
            "BESSELY0(z):=2/pi*((LN(z/2) + euler_gamma)*BESSEL_J(0, z) - "
            "(BESSELY0_AUX(z)) SUB 2)"
        ),
        (
            "BESSELY1_AUX(z):=ITERATE(IF(b_ SUB 2 = b_ SUB 3, b_, [b_ SUB 1 + 1, b_ "
            "SUB 3, b_ SUB 3 + (- z^2/4)^b_ SUB 1/(b_ SUB 1!^2*(b_ SUB 1 + "
            "1))*(SUM(1/m_, m_, 1, b_ SUB 1) + 1/(2*(b_ SUB 1 + 1)))]), b_, [0, 1, 0])"
        ),
        (
            "BESSELY1(z):=- 2/pi*(1/z - (LN(z/2) + euler_gamma)*BESSEL_J(1, z) + "
            "z/2*(BESSELY1_AUX(z)) SUB 2)"
        ),
        (
            "BESSEL_Y1(n, z):=ITERATES([w_ SUB 1 + 1, w_ SUB 3, 2*w_ SUB 1/z*w_ SUB "
            "3 - w_ SUB 2], w_, [1, BESSELY0(z), BESSELY1(z)], n - 1)"
        ),
        (
            "BESSEL_Y(n, z):=IF(z = 0, -inf, IF(n < 0, (-1)^n, 1)*IF(n = 0, "
            "BESSELY0(z), IF(ABS(n) = 1, BESSELY1(z), (BESSEL_Y1(ABS(n), z)) SUB "
            "ABS(n) SUB 3)))"
        ),
        "HARMONIC_NUMBER(n):=SUM(1/k, k, 1, n)",
        (
            "BESSEL_Y_SERIES(n, z, m):=2/pi*((euler_gamma + "
            "LN(z/2))*BESSEL_J_SERIES(n, z, m) - (SUM((n - m_ - 1)!/m_!*(z/2)^(2*m_ "
            "- n), m_, 0, n - 1) + SUM(COS(pi*m_)*(z/2)^(n + 2*m_)/(m_!*(n + "
            "m_)!)*(HARMONIC_NUMBER(m_) + HARMONIC_NUMBER(n + m_)), m_, 0, m))/2)"
        ),
        "BESSEL_Y_ASYMP(n, z):=SQRT(2/(pi*z))*SIN(z - (n + 1/2)*pi/2)",
        "BESSEL_I(n, z):=INT(#e^(z*COS(t))*COS(n*t), t, 0, pi)/pi",
        "BESSEL_I_SERIES(n, z, m):=(z/2)^n*SUM((z^2/4)^k/(k!*(n + k)!), k, 0, m)",
        "BESSEL_I_ASYMP(n, z):=#e^z/SQRT(2*pi*z)*(1 - (4*n^2 - 1)/(8*z))",
        "BESSEL_K(n, z):=INT(#e^(- z*COSH(t))*COSH(n*t), t, 0, inf)",
        "BESSEL_K_ASYMP(n, z):=#e^(-z)*SQRT(pi/(2*z))*(1 + (4*n^2 - 1)/(8*z))",
        (
            "SPHERICAL_BESSEL_Y(n, z):=IF(n < 0, - (-1)^n*SPHERICAL_BESSEL_J(-n - 1, "
            "z), - z^n*LIM(ITERATE(- 1/z_*DIF(u_, z_), u_, COS(z_)/z_, n), z_, z, 0))"
        ),
        (
            "SPHERICAL_BESSEL_J(n, z):=IF(n < 0, (-1)^n*SPHERICAL_BESSEL_Y(-n - 1, "
            "z), z^n*LIM(ITERATE(- 1/z_*DIF(u_, z_), u_, SIN(z_)/z_, n), z_, z, 0))"
        ),
        "SPHERICAL_BESSEL_J2(v):=v/SQRT(SUM((2*k_ - 1)*v SUB k_^2, k_, 1, DIM(v)))",
        (
            "SPHERICAL_BESSEL_J1(n, z):=ITERATE(APPEND([(2*(n - DIM(v_)) + 5)/z*v_ "
            "SUB 1 - v_ SUB 2], v_), v_, [1, 0], n)"
        ),
        "SPHERICAL_BESSEL_J_LIST(n, z):=SPHERICAL_BESSEL_J2(SPHERICAL_BESSEL_J1(n, z))",
        "AI_AUX(z, m):=SUM(3^k*GAMMA(1/3 + k)*z^(3*k)/(3*k)!, k, 0, m)",
        "BI_AUX(z, m):=SUM(3^k*GAMMA(2/3 + k)*z^(3*k + 1)/(3*k + 1)!, k, 0, m)",
        (
            "AI_SERIES(z, m):=IF(z = 0, 3^(5/6)*(1/3)!/(2*pi), (3^(- 1/6)*AI_AUX(z, "
            "m) - 3^(1/6)*BI_AUX(z, m))/(2*pi))"
        ),
        (
            "BI_SERIES(z, m):=IF(z = 0, 3^(4/3)*(1/3)!/(2*pi), (3^(1/3)*AI_AUX(z, m) "
            "+ 3^(2/3)*BI_AUX(z, m))/(2*pi))"
        ),
    ),
    "EllipticIntegrals": (
        "ELLIPTIC_F(phi,m):=INT(1/SQRT(1-m*SIN(t_)^2),t_,0,phi)",
        "ELLIPTIC_E(phi,m):=INT(SQRT(1-m*SIN(t_)^2),t_,0,phi)",
        "ELLIPTIC_PI(phi,m,n):=INT(1/((1-n*SIN(t_)^2)*SQRT(1-m*SIN(t_)^2)),t_,0,phi)",
        (
            "JACOBI_AGM(m,n):=ITERATES([(v_ SUB 1+v_ SUB 2)/2,SQRT(v_ SUB 1*v_ SUB "
            "2),(v_ SUB 1-v_ SUB 2)/2],v_,[1,SQRT(1-m),SQRT(m)],n)"
        ),
        (
            "JACOBI_AUX(u,a,n):=ITERATE([ASIN(a SUB (v_ SUB 2) SUB 3*SIN(v_ SUB 1)/a "
            "SUB (v_ SUB 2) SUB 1)/2+v_ SUB 1/2,v_ SUB 2-1],v_,[2^n*a SUB n SUB "
            "1*u,n+1],n)"
        ),
        "JACOBI_AM(u,m,n):=(JACOBI_AUX(u,JACOBI_AGM(m,n),n)) SUB 1",
        "KI(m,n):=pi/(2*(JACOBI_AGM(m,n)) SUB (n+1) SUB 1)",
        "RF1(v,lambda):=[(v SUB 1+lambda)/4,(v SUB 2+lambda)/4,(v SUB 3+lambda)/4]",
        "RF2(v):=SQRT(v SUB 2*v SUB 3)+SQRT(v SUB 1*v SUB 3)+SQRT(v SUB 1*v SUB 2)",
        "RF3(x,y,z,n):=ITERATE(RF1(v_,RF2(v_)),v_,[x,y,z],n)",
        "RF4(ave,v):=[1,1,1]-v/ave",
        "RF5(v):=[v SUB 1*v SUB 2-v SUB 3*v SUB 3,v SUB 1*v SUB 2*v SUB 3]",
        "RF6(v,ave):=(1+(v SUB 1/24-1/10-3*v SUB 2/44)*v SUB 1+v SUB 2/14)/SQRT(ave)",
        "RF7(v,ave):=RF6(RF5(RF4(ave,v)),ave)",
        "RF8(v):=RF7(v,AVERAGE(v))",
        "RF9(x,y,z,n):=RF8(RF3(x,y,z,n))",
        "EIF(phi,m,n):=SIN(phi)*RF9(COS(phi)^2,1-m*SIN(phi)^2,1,n)",
    ),
    "ExponentialIntegrals": (
        "LI(x, m):=EI(LN(x), m)",
        "EN(n, z):=INT(#e^(- z*t_)/t_^n, t_, 1, inf)",
        "EN_ASYMP(n, z, m):=#e^(-z)/z*SUM(PERM(n + l_ - 1, l_)/(-z)^l_, l_, 0, m)",
        "EI1(z, m):=-euler_gamma - LN(z) - SUM((-z)^n_/(n_*n_!), n_, 1, m)",
    ),
    "FresnelIntegrals": (
        "DAWSON(x):=- SQRT(pi)*#i*#e^(- x^2)*ERF(#i*x)/2",
        "FRESNEL_SIN(z):=INT(SIN(pi*t_^2/2), t_, 0, z)",
        (
            "FRESNEL_SIN_SERIES(z, m):=SUM(COS(pi*k_)*(pi/2)^(2*k_ + 1)*z^(4*k_ + "
            "3)/((2*k_ + 1)!*(4*k_ + 3)), k_, 0, m)"
        ),
        "FRESNEL_COS(z):=INT(COS(pi*t_^2/2), t_, 0, z)",
        (
            "FRESNEL_COS_SERIES(z, m):=SUM(COS(pi*k_)*(pi/2)^(2*k_)*z^(4*k_ + "
            "1)/((2*k_)!*(4*k_ + 1)), k_, 0, m)"
        ),
        "FRESNEL_AUX_F(z):=1/(pi*z) - 3/(pi^3*z^5)",
        "FRESNEL_AUX_G(z):=1/(pi^2*z^3) - 15/(pi^4*z^7)",
        (
            "FRESNEL_SIN_ASYMP(z):=1/2 - FRESNEL_AUX_F(z)*COS(pi*z^2/2) - "
            "FRESNEL_AUX_G(z)*SIN(pi*z^2/2)"
        ),
        (
            "FRESNEL_COS_ASYMP(z):=1/2 + FRESNEL_AUX_F(z)*SIN(pi*z^2/2) - "
            "FRESNEL_AUX_G(z)*COS(pi*z^2/2)"
        ),
        "FRESNEL_COS_J_AUX(v, m):=SUM(v SUB (2*k_ + 1), k_, 0, m)",
        (
            "FRESNEL_COS_J(z, m):=FRESNEL_COS_J_AUX(z*SPHERICAL_BESSEL_J_LIST(2*m + "
            "2, pi*z^2/2), m)"
        ),
        "FRESNEL_SIN_J_AUX(v, m):=SUM(v SUB (2*k_ + 2), k_, 0, m)",
        (
            "FRESNEL_SIN_J(z, m):=FRESNEL_SIN_J_AUX(z*SPHERICAL_BESSEL_J_LIST(2*m + "
            "2, pi*z^2/2), m)"
        ),
    ),
    "HypergeometricFunctions": (
        (
            "HYPERGEOMETRIC_SERIES(plist,qlist,z,m):=SUM(PRODUCT(PERM(k_+plist SUB "
            "j_-1,k_),j_,1,DIMENSION(plist))*z^k_/(PRODUCT(PERM(k_+qlist SUB "
            "j_-1,k_),j_,1,DIMENSION(qlist))*k_!),k_,0,m)"
        ),
        "KUMMER_AUX(a,b,z):=(1/2)^a/a+INT(t^(a-1)*(#e^(z*t)*(1-t)^(b-a-1)-1),t,0,1/2)",
        (
            "KUMMER(a,b,z):=GAMMA(b)/(GAMMA(b-a)*GAMMA(a))*(KUMMER_AUX(a,b,z)+#e^z*KUM"
            "MER_AUX(b-a,b,-z))"
        ),
        "KUMMER_SERIES(a,b,z,m):=HYPERGEOMETRIC_SERIES([a],[b],z,m)",
        (
            "GAUSS_AUX(a,b,c,z):=(1/2)^b/b+INT(t^(b-1)*((1-t)^(c-b-1)*(1-t*z)^(-a)-1),"
            "t,0,1/2)"
        ),
        (
            "GAUSS(a,b,c,z):=IF(z=1,GAMMA(c)*GAMMA(c-a-b)/(GAMMA(c-a)*GAMMA(c-b)),GAMM"
            "A(c)/(GAMMA(b)*GAMMA(c-b))*(GAUSS_AUX(a,b,c,z)+(1-z)^(-a)*GAUSS_AUX(a,c-b"
            ",c,z/(z-1))),GAMMA(c)/(GAMMA(b)*GAMMA(c-b))*(GAUSS_AUX(a,b,c,z)+(1-z)^(-a"
            ")*GAUSS_AUX(a,c-b,c,z/(z-1))))"
        ),
        "GAUSS_SERIES(a,b,c,z,m):=HYPERGEOMETRIC_SERIES([a,b],[c],z,m)",
    ),
    "OrthogonalPolynomials": (
        (
            "CHEBYCHEV_T_SYMBOLIC(n,x):=IF(n=0,1,(SUM(ITERATES([v_ SUB 1*(2*v_ SUB "
            "2-n-1)*(2*v_ SUB 2-n-2)/(4*x^2*v_ SUB 2*(v_ SUB 2-n)),v_ SUB "
            "2+1],v_,[2^(n-1)*x^n,1],n/2))) SUB 1)"
        ),
        (
            "CHEBYCHEV_T_NUMERIC(n,x):=IF(n=0,1,IF(n=1,x,IF(MOD(n,2)=0,2*CHEBYCHEV_T_N"
            "UMERIC(n/2,x)^2-1,2*CHEBYCHEV_T_NUMERIC((n-1)/2,x)*CHEBYCHEV_T_NUMERIC((n"
            "+1)/2,x)-x)))"
        ),
        (
            "CHEBYCHEV_T(n,x):=IF(NUMBER(x),CHEBYCHEV_T_NUMERIC(n,x),CHEBYCHEV_T_SYMBO"
            "LIC(n,x),CHEBYCHEV_T_SYMBOLIC(n,x))"
        ),
        "CHEBYCHEV_T_LIST(n,x):=RECURRENCE(2*x*v_ SUB 1-v_ SUB 2,v_,[x,1],n)",
        (
            "CHEBYCHEV_U_SYMBOLIC(n,x):=IF(n=0,1,(SUM(ITERATES([(2*v_ SUB "
            "2-n-1)*(2*v_ SUB 2-n-2)/(4*v_ SUB 2*x^2*(v_ SUB 2-n-1))*v_ SUB 1,v_ SUB "
            "2+1],v_,[2^n*x^n,1],n/2))) SUB 1)"
        ),
        (
            "CHEBYCHEV_U_NUMERIC(n,x):=IF(n=0,1,IF(n=1,2*x,IF(MOD(n,2)=0,2*CHEBYCHEV_T"
            "_NUMERIC(n/2,x)*CHEBYCHEV_U_NUMERIC(n/2,x)-1,2*CHEBYCHEV_T_NUMERIC((n-1)/"
            "2,x)*CHEBYCHEV_U_NUMERIC((n+1)/2,x)-2*x)))"
        ),
        (
            "CHEBYCHEV_U(n,x):=IF(NUMBER(x),CHEBYCHEV_U_NUMERIC(n,x),CHEBYCHEV_U_SYMBO"
            "LIC(n,x),CHEBYCHEV_U_SYMBOLIC(n,x))"
        ),
        "CHEBYCHEV_U_LIST(n,x):=RECURRENCE(2*x*v_ SUB 1-v_ SUB 2,v_,[2*x,1],n)",
        (
            "LEGENDRE_P(n,x):=2^(-n)*(SUM(ITERATES([v_ SUB 1*(2*v_ SUB 2-n-1)*(2*v_ "
            "SUB 2-n-2)/(2*x^2*v_ SUB 2*(2*v_ SUB 2-2*n-1)),v_ SUB "
            "2+1],v_,[COMB(2*n,n)*x^n,1],n/2))) SUB 1"
        ),
        (
            "LEGENDRE_P_LIST(n,x):=RECURRENCE(2*x*v_ SUB 1-v_ SUB 2+(v_ SUB 2-x*v_ "
            "SUB 1)/DIMENSION(v_),v_,[x,1],n)"
        ),
        "ASSOCIATED_LEGENDRE_P(n,m,x):=(-1)^m*(1-x^2)^(m/2)*DIF(LEGENDRE_P(n,x),x,m)",
        (
            "HERMITE_H(n,x):=(SUM(ITERATES([-v_ SUB 1*(2*v_ SUB 2-n-1)*(2*v_ SUB "
            "2-n-2)/(4*x^2*v_ SUB 2),v_ SUB 2+1],v_,[2^n*x^n,1],n/2))) SUB 1"
        ),
        (
            "HERMITE_H_LIST(n,x):=RECURRENCE(2*x*v_ SUB 1-2*(DIMENSION(v_)-1)*v_ SUB "
            "2,v_,[2*x,1],n)"
        ),
        (
            "HERMITE_HE(n,x):=(SUM(ITERATES([-v_ SUB 1*(2*v_ SUB 2-n-1)*(2*v_ SUB "
            "2-n-2)/(2*x^2*v_ SUB 2),v_ SUB 2+1],v_,[x^n,1],n/2))) SUB 1"
        ),
        (
            "HERMITE_HE_LIST(n,x):=RECURRENCE(x*v_ SUB 1-(DIMENSION(v_)-1)*v_ SUB "
            "2,v_,[x,1],n)"
        ),
        "WEBER_D(n,x):=#e^(-x^2/4)*HERMITE_HE(n,x)",
        (
            "LAGUERRE_L(n,x):=(SUM(ITERATES([v_ SUB 1*x*(v_ SUB 2-n-1)/(v_ SUB 2*v_ "
            "SUB 2),v_ SUB 2+1],v_,[1,1],n))) SUB 1"
        ),
        (
            "LAGUERRE_L_LIST(n,x):=RECURRENCE(2*v_ SUB 1-v_ SUB 2+(v_ SUB 2-(x+1)*v_ "
            "SUB 1)/DIMENSION(v_),v_,[1-x,1],n)"
        ),
        (
            "GENERALIZED_LAGUERRE(n,a,x):=(SUM(ITERATES([v_ SUB 1*x*(v_ SUB "
            "2-n-1)/(v_ SUB 2*(a+v_ SUB 2)),v_ SUB "
            "2+1],v_,[PRODUCT(a+k_,k_,1,n)/n!,1],n))) SUB 1"
        ),
        (
            "GENERALIZED_LAGUERRE_LIST(n,a,x):=RECURRENCE(2*v_ SUB 1-v_ SUB "
            "2+((1-a)*v_ SUB 2-(1+x-a)*v_ SUB 1)/DIMENSION(v_),v_,[1-x+a,1],n)"
        ),
        (
            "JACOBI_P(n,a,b,x):=(SUM(ITERATES([v_ SUB 1*(a+b+v_ SUB 2+n)*(v_ SUB "
            "2-n-1)*(1-x)/(2*v_ SUB 2*(a+v_ SUB 2)),v_ SUB "
            "2+1],v_,[PRODUCT(a+k_,k_,1,n)/n!,1],n))) SUB 1"
        ),
        (
            "JACOBI_P_LIST(n,a,b,x):=RECURRENCE(((a+b+2*DIMENSION(v_)-1)*(x*(a+b+2*DIM"
            "ENSION(v_))*(a+b+2*(DIMENSION(v_)-1))+a^2-b^2)*v_ SUB "
            "1-2*(a+b+2*DIMENSION(v_))*(a+DIMENSION(v_)-1)*(b+DIMENSION(v_)-1)*v_ "
            "SUB "
            "2)/(2*DIMENSION(v_)*(a+b+DIMENSION(v_))*(a+b+2*(DIMENSION(v_)-1))),v_,[(x"
            "*(a+b+2)+a-b)/2,1],n)"
        ),
        (
            "GEGENBAUER_C(n,a,x):=(SUM(ITERATES([v_ SUB 1*(2*v_ SUB 2-n-1)*(2*v_ SUB "
            "2-n-2)/(4*x^2*v_ SUB 2*(v_ SUB 2-a-n)),v_ SUB "
            "2+1],v_,[(2*x)^n*IF(a=0,1/n,PRODUCT(a+k_-1,k_,1,n)/n!,PRODUCT(a+k_-1,k_,1"
            ",n)/n!),1],n/2))) SUB 1"
        ),
        (
            "GEGENBAUER_C_LIST(n,a,x):=RECURRENCE((2*x*(a+DIMENSION(v_)-1)*v_ SUB "
            "1-(2*a+DIMENSION(v_)-2)*v_ SUB 2)/DIMENSION(v_),v_,[2*a*x,1],n)"
        ),
    ),
    "ProbabilityFunctions": (
        "POCHHAMMER(a, x):=PERM(x + a - 1, x)",
        (
            "INCOMPLETE_GAMMA(z, w):=1/GAMMA(z)*(w^z/z + INT((#e^(-t_) - 1)*t_^(z - "
            "1), t_, 0, w))"
        ),
        (
            "INCOMPLETE_GAMMA_SERIES(z, w, m):=#e^(-w)*w^z*SUM(w^(m - n_)/GAMMA(z + "
            "m - n_ + 1), n_, 0, m)"
        ),
        "EULER_BETA(z, w):=GAMMA(z)*GAMMA(w)/GAMMA(z + w)",
        (
            "INCOMPLETE_BETA(x, z, w):=(x^z/z + (1 - (1 - x)^w)/w + INT(t_^(z - "
            "1)*((1 - t_)^(w - 1) - 1) - (1 - t_)^(w - 1), t_, 0, x))/EULER_BETA(z, w)"
        ),
        "POISSON_DENSITY(k, t):=#e^(-t)*t^k/k!",
        "POISSON_DISTRIBUTION(k, t):=SUM(POISSON_DENSITY(m_, t), m_, 0, k)",
        "BINOMIAL_DENSITY(k, n, p):=COMB(n, k)*p^k*(1 - p)^(n - k)",
        (
            "BINOMIAL_DISTRIBUTION(k, n, p):=SUM(BINOMIAL_DENSITY(m_, n, p), m_, 0, "
            "MIN(k, n))"
        ),
        "HYPERGEOMETRIC_DENSITY(k, n, m, j):=COMB(m, k)*COMB(j - m, n - k)/COMB(j, n)",
        (
            "HYPERGEOMETRIC_DISTRIBUTION(k, n, m, j):=SUM(HYPERGEOMETRIC_DENSITY(l_, "
            "n, m, j), l_, MAX(0, n - j + m), MIN(k, n, m))"
        ),
        (
            "STUDENT(t, v):=IF(t < 0, 1 - STUDENT(-t, v), 1 - "
            "1/2*INCOMPLETE_BETA(v/(v + t^2), v/2, 1/2))"
        ),
        "F_DISTRIBUTION(f_, v1, v2):=INCOMPLETE_BETA(v2/(v2 + v1*f_), v2/2, v1/2)",
        "CHI_SQUARE(x2, v):=INCOMPLETE_GAMMA(v/2, x2/2)",
        (
            "DIGAMMA_PSI(z):=LN(z) - 1/(2*z) - 2*INT(t_/((t_^2 + z^2)*(#e^(2*pi*t_) "
            "- 1)), t_, 0, inf)"
        ),
    ),
    "ZetaFunctions": (
        "HURWITZ_ZETA(s,a,m):=(m+a+1/2)^(1-s)/(s-1)+SUM((k+a)^(-s),k,0,m)",
        "LERCH_PHI(z,s,a,m):=SUM(z^k/(a+k)^s,k,0,m)",
    ),
}
