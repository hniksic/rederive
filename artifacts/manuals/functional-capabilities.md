# Derive: Functional Capabilities Reference

Research notes compiled for the "Derive remake" requirements effort. Derive was
a symbolic/numeric computer algebra system (CAS) originally developed by Soft
Warehouse Inc. (Honolulu, Hawaii) as a successor to muMATH, first released for
MS-DOS in 1988. Soft Warehouse was acquired by Texas Instruments in 1994; TI
continued Windows development through Derive 6.1 (final release, October
2004), and officially discontinued the product on 29 June 2007 in favor of the
TI-Nspire CAS (whose symbolic engine descends from Derive/muMATH, also used
earlier in the TI-89/TI-92 Plus/Voyage 200 calculators).

Primary source for this writeup is TI's own **Derive 6.1 online help**
(copyright 1988-2003 Texas Instruments), which is effectively a complete
command and function reference, mirrored offline in this repository at
`derive6.1-online-help/` (see manifest below). Historical/version details are
cross-checked against the Derive 3.14 User Manual (1996) and the Derive 5/6
"Introduction" booklets. Where noted, "limitations" material is drawn from
outside reviews since Derive's own documentation naturally doesn't advertise
its own gaps.

Derive's underlying implementation language was muLISP. It shipped as a single
integrated application: one algebra ("worksheet") window for symbolic/numeric
expressions plus separate 2D- and 3D-plot windows, all editable with a
structured "textbook-style" expression editor (no free-text command language
was required, though one existed).

---

## 1. Symbolic Algebra

### Simplification model
- Expressions are entered unevaluated and only transformed when the user
  issues a **Simplify** command (or the equivalent `Simplify` toolbar
  button) — merely typing/authoring an expression does not simplify it.
  This is a deliberate, distinctive design choice: Derive is not a
  "simplify-on-input" REPL like many other CASes; each worksheet line keeps
  its own simplify/approximate history, and prior lines/labels (`#n`) can be
  referenced by later ones.
- **Simplify > Basic** – general-purpose simplification to a "sufficiently
  simple form" (no superfluous variables/roots/functions, no reducible
  polynomial degrees). Deliberately conservative: a transformation is applied
  only when Derive can *prove* it is valid for the (declared) domain of the
  variables involved — e.g. `LN(x^2)/2 -> LN(x)` is only used if `x` is
  declared nonnegative.
- **Simplify > Expand** – polynomial expansion or rational-function partial
  fraction decomposition, with selectable "denominator/expression factoring
  type": Trivial, Squarefree, Rational, Radical, Complex (each a stronger,
  slower factoring criterion). Also expands Boolean expressions to
  disjunctive normal form. Built-in functions: `EXPAND(u)`, `EXPAND(u,x,...)`,
  `EXPAND(u,amount,x,...)`, `TERMS(u)`.
- **Simplify > Factor** – factor polynomials/numbers/matrices/Boolean
  expressions, with the same amount levels plus matrix-specific **Turing**
  (LU/row-echelon) and **Gram-Schmidt/Schmidt** (QR) factorization types.
  Functions: `FACTOR(u, amount, vars...)`, `FACTORS(...)` (returns
  base-degree pairs / prime-power pairs as a vector).
  - Number factoring uses trial division, perfect-power testing,
    Pollard/Brent rho, Pollard p-1, Williams p+1, and Lenstra elliptic-curve
    factoring, with a live "Display Steps" progress indicator.
- **Simplify > Variable Substitution** (`SUBST(u, v, w)`) — serial or
  parallel substitution of values/subexpressions for variables.
- **Simplify > Subexpression Substitution** — substitute directly for any
  highlighted subexpression (not just whole variables).
- **Simplify > Approximate** (`APPROX(u, n)`) — force numeric evaluation to
  n digits regardless of current precision mode.
- Expression ordering is controlled by a global, user-settable
  **VariableOrder** list ("more main" variables sort first); this materially
  affects whether subexpressions get expanded/collapsed and can be used
  deliberately to control expression blow-up (a very Derive-specific idiom).
- Display/format controls: Normal vs. Compressed spacing, multiplication
  operator glyph (Asterisk/Dot/Implicit), input/output radix base (2-36,
  with named Binary/Octal/Decimal/Hexadecimal shortcuts).
- **Step-by-Step / Display Steps mode** — a distinctive teaching feature:
  Derive can show the individual transformation *rules* applied during
  simplification (added/expanded in v6), and equations can be solved
  manually one algebraic step at a time by combining an equation with an
  expression via +, -, *, / (e.g. `(2x - 3 = 5) + 3`), or by applying a
  function to both sides of an equation via Edit > Object.

### Solving equations and inequalities
- **Solve > Expression** / `SOLVE(expr, var[, Real])` / `SOLUTIONS(...)` —
  exact algebraic solving. Returns results as a Boolean disjunction of
  relations (`SOLVE`) or as a vector of values (`SOLUTIONS`). Directly solves
  binomial/linear/quadratic/cubic/quartic equations and anything reducible to
  them by exact factoring; inequalities are solved into unions of intervals.
  A degenerate (always-true) equation returns `true`/`[@1]` (arbitrary-value
  placeholders, `@1`, `@2`, ... for free parameters).
  Restricting to `Real` filters out complex solutions.
- **Solve > System** / `SOLVE([...], [vars])` — systems of polynomial
  equations, internally reduced via **Gröbner basis** (`GROEBNER_BASIS`,
  lexicographic term order) to a univariate resolvent, back-substituted.
  Singular-but-consistent linear systems return arbitrary-parameterized
  solutions (`@n`); inconsistent systems return `false`/`[]`.
- **NSOLVE / NSOLUTIONS** — numeric-only solving for a single-variable
  equation (bisection/Newton-style search within an optional bracketing
  interval); falls back automatically from SOLVE in approximate mode.
- Non-polynomial systems and general iterative numeric solving are handled
  by the `EquationSolving.mth` utility file: multivariate **Newton's method**
  (`NEWTON`, `NEWTONS`), **fixed-point iteration** (`FIXED_POINT`), and
  **Taylor-series local solving** (`TAYLOR_SOLVE`, `TAYLOR_INVERSE`) for
  equations with no closed form.
- Distinctive idiom: `unit_circle` — a symbolic constant representing an
  arbitrary point on the complex unit circle, e.g. solving `|z|=2` for a
  complex-declared `z` yields `z = 2*unit_circle` rather than an artificial
  infinite disjunction.

### Domains, assumptions, and multivalued functions
- **Author > Variable Domain** (`variable :ε domain [interval]`) — the
  central "assumptions" mechanism. Declares a variable's data type (Integer,
  Real, Complex, Vector, Set, Logical) and, for Integer/Real, an interval
  (All / Positive / Negative / Nonnegative / Nonpositive / open / closed /
  half-open, with numeric bounds). Undeclared variables default to Real.
  Simplification rules are only invoked when provably valid for the declared
  domain (e.g. `SQRT(x^2) -> |x|` needs Real; `SQRT(x*y) -> SQRT(x)*SQRT(y)`
  needs both nonnegative). Substitution, unlike simplification, ignores
  domain declarations entirely ("at your own risk").
- Multivalued/branch handling: a global **Branch** mode setting controls how
  fractional powers/inverse trig of negative or complex arguments are
  resolved (principal branch conventions), independent of and orthogonal to
  variable-domain declarations.
- `IF(test, then, else, unknown)` expressions are the general mechanism for
  piecewise/conditional definitions and interact with domain declarations:
  if Derive cannot prove the test true or false from current declarations,
  the whole IF expression (or its `unknown` clause) is returned unevaluated
  rather than guessed.

---

## 2. Calculus

- **Limits**: `LIM(u, x, a)` (defaults `a=0`); one-sided limits via a 4th
  argument (+1 / -1 for right/left); vector form `LIM(u, [x,y], [a,b])`
  substitutes several variables at once (evaluated left-to-right per
  argument, not a true multivariate limit — a separate utility function
  `LIM2` computes the limit along a parametrized straight-line direction
  `@1` and can reveal direction-dependence). `LIM` differs from plain
  substitution (`SUBST`) in that it can resolve indeterminate forms (e.g.
  `LIM(SIN(x)/x, x, 0) = 1` vs. `SUBST(...) = ?`).
- **Differentiation**: `DIF(u, x)`, `DIF(u, x, n)` for nth order, nested
  `DIF(DIF(u,x),y)` for mixed partials; prime notation `F'(x)`, `F''(x)`
  for named user functions; negative order `DIF(u, x, -n)` computes nth
  antiderivatives. Full symbolic chain rule for arbitrary/undefined
  functions declared via `F(x) :=` (empty body). Cannot differentiate `!`
  (factorial) directly — done via Gamma/Digamma/polygamma instead.
- **Integration**: `INT(u, x)` (indefinite, no constant), `INT(u, x, c)`
  (with constant), `INT(u, x, a, b)` (definite); nested calls give
  iterated multiple integrals. Symbolic antiderivatives cover extended
  polynomials, products of polynomials with sinusoids/logs/exponentials
  linear in argument, piecewise-continuous functions, and many
  rational/square-root-of-linear-or-quadratic integrands. If precision is
  Approximate (or Mixed and no closed form is found), Derive falls back to
  an **extrapolated adaptive Simpson's rule** numeric integrator, flagging
  "Dubious accuracy" when it detects trouble. Improper integrals with
  infinite limits/endpoint singularities are handled via limits of the
  antiderivative; interior singularities are *not* detected automatically
  (documented risk of silently returning a Cauchy principal value instead of
  a correct/divergent result) — the manual explicitly tells users to find
  singularities themselves (e.g. via Solve on the denominator) and split
  the integral.
- **Series**: `TAYLOR(u, x, a, n)` — Taylor/Maclaurin polynomial
  approximation. No native symbolic Fourier series as a first-class
  command (Fourier series is a *utility-file* function, see below), and no
  built-in general power-series datatype/object — series are just
  ordinary truncated polynomials.
- **Sums and products**: `SUM(u, n)` (antidifference, analogous to an
  antiderivative, via telescoping), `SUM(u, n, k, m)` (definite sum),
  vector-domain sums `SUM(u, n, [list])`; `PRODUCT` is the multiplicative
  analogue (antiquotient). Both assume integer-valued index limits.
- **Differential equations (ODEs)** — no single generic "DSolve"-style
  black box; instead a rich library of *targeted* exact-method functions
  plus separate numeric/series fallbacks:
  - First order (`FirstOrderODEs.mth`): general-purpose `DSOLVE1_GEN` /
    `DSOLVE1` (handles exact, linear, separable, homogeneous,
    generalized-homogeneous, and integrating-factor-solvable equations
    automatically) plus named special-case solvers `SEPARABLE(_GEN)`,
    `LINEAR1(_GEN)`, `HOMOGENEOUS(_GEN)`, `EXACT(_GEN)`,
    `INTEGRATING_FACTOR(_GEN)`, and advanced-method solvers
    `MONOMIAL_TEST`, `BERNOULLI_ODE(_GEN)`, `GEN_HOM(_GEN)`,
    `FUN_LIN_CCF(_GEN)`, `LIN_FRAC(_GEN)`, `ALMOST_LIN(_GEN)`, and
    `CLAIRAUT` (returns both the general solution and a singular solution).
  - Second order (`SecondOrderODEs.mth`): `DSOLVE2` / `DSOLVE2_IV` /
    `DSOLVE2_BV` for constant/variable-coefficient linear equations
    `y''+p(x)y'+q(x)y=r(x)`, plus `AUTONOMOUS_CONSERVATIVE`, `LIOUVILLE`,
    `AUTONOMOUS`, `EXACT2` for particular nonlinear forms.
  - Recurrence/difference equations (`RecurrenceEquations.mth`):
    `LIN1_DIFFERENCE`, `RECURRENCE1`, `GEOMETRIC1`, `CLAIRAUT_DIF`,
    `LIN2_CCF(_BV)`.
  - Approximate/numeric ODE solving (`ODEApproximation.mth`), used when no
    exact method applies or the equation is transcendental: truncated
    Taylor-series solvers (`TAYLOR_ODE1`, `TAYLOR_ODES`, `TAYLOR_ODE2`),
    **Picard iteration** (`PICARD`), **Euler's method** (`EULER_ODE`),
    classic 4th-order **Runge-Kutta** (`RK`, supports systems and returns a
    plottable data matrix), and `DIRECTION_FIELD` for slope-field plotting.
    Higher-order ODEs/systems must be manually reduced to first-order
    systems (the help text walks through the standard substitution
    procedure) — there is no automatic order-reduction.
  - No native symbolic **PDE** (partial differential equation) solver of
    any kind.

---

## 3. Linear Algebra

- **Vectors and matrices** are Derive's only aggregate/array data type — an
  n×m matrix is literally stored and treated as a vector of n row-vectors
  (no distinct "matrix" object). Entered as `[a,b,c]` (vector) or
  `[a,b;c,d]` (matrix, semicolon-separated rows) via the expression line,
  or through Author > Vector / Author > Matrix dialogs.
- Element access/manipulation: `SUB` (`v SUB n`, displayed as a real
  subscript), double-`SUB`/`SUB SUB` for matrix rows vs. columns, `ROW` and
  `COL` infix operators, `ELEMENT(v,n)` functional form, plus a full
  Lisp-like vector toolkit: `FIRST`, `REST`, `ADJOIN`, `APPEND`, `DELETE`,
  `INSERT`, `REPLACE`, `REVERSE`, `POSITION`, `SELECT`, `SORT` (with
  optional custom comparator), `DIM`.
- Algebra: `+`, `-`, elementwise scalar `*`/`/`, dot-notation matrix/vector
  product (`A·v`, `v·A`, `A·B`), transpose postfix `` ` `` operator,
  `DET` (determinant), `TRACE`, integer matrix powers `A^n`, matrix inverse
  `A^(-1)` (undefined/unsimplified for singular matrices), dedicated
  `DOTPRODUCT` and `CROSSPRODUCT` infix operators (2- and 3-vector cross
  products; dot product conjugates the second operand for complex
  vectors). Declaring a variable's domain as **Vector** lets Derive apply
  vector/matrix algebra identities (associativity, `(AB)' = B'A'`, etc.)
  symbolically to undetermined vector-valued variables.
- **Row reduction / linear systems**: `ROW_REDUCE(A[, B])` computes
  (reduced) row echelon form, optionally of an augmented system, and
  handles singular/rank-deficient systems (returning arbitrary components
  where indeterminate) — the recommended general method for solving linear
  systems, in preference to explicit matrix inversion. `RANK(A)`.
  Turing factorization (`FACTOR(A, Turing)`) yields the full
  **P·L·D·U·R** decomposition in one call (permutation, unit-lower,
  diagonal, unit-upper, and row-echelon factors) and is the recommended
  way to locate symbolic special cases (where a parameterized system
  changes behavior). Gram-Schmidt **QR** factorization
  (`FACTOR(A, Schmidt)`) is also built in, for square, rectangular
  full-rank, and rank-deficient matrices.
- **Eigenvalues/eigenvectors**: `CHARPOLY(A, v)` (characteristic
  polynomial), `EIGENVALUES(A[, v])` (roots of the characteristic
  polynomial — exact only up to 4×4 via the quartic formula, otherwise
  practically requires numeric solving). Exact eigenvectors via
  `EXACT_EIGENVECTOR` (utility file; effectively limited to ≤4×4) and
  **inverse-iteration** numeric eigenvectors via `APPROX_EIGENVECTOR`
  (recommended above 3×3). `NULL_SPACE(M)` for the kernel/null space
  (rational basis). Row-reduction primitives for manual Gaussian/
  Gauss-Jordan elimination (`SCALE_ELEMENT`, `SWAP_ELEMENTS`,
  `SUBTRACT_ELEMENTS`, `FORCE0`, `PIVOT`) and cofactor-expansion primitives
  (`MINOR`, `COFACTOR`, `ADJOINT`) are provided in `LinearAlgebra.mth` for
  educational, step-at-a-time linear algebra.
- **Vector calculus**: differential (`DifferentialVectorCalculus`) and
  integral (`IntegralVectorCalculus`) operators over rectangular,
  cylindrical, and spherical coordinate systems via generic
  `JACOBIAN`/`COVARIANT_METRIC_TENSOR`/`GEOMETRY_MATRIX` machinery in
  `VectorMatrixFunctions.mth` (also defines convenience unit vectors `i_`,
  `j_`, `k_`, and coordinate-conversion functions
  `RECTANGULAR_TO_POLAR`/`POLAR_TO_RECTANGULAR`/`POLAR_SUM`).
- Tensor algebra beyond vector/matrix is *not* built in — only available as
  a **user-contributed** package (`TensorAlgebra.mth`), same for the
  **Simplex method**/linear programming (`SimplexMethod.mth`, contributed).

---

## 4. Number Systems, Precision, and Number Theory

- Internally, **all numbers are stored as integers or exact reduced
  rational fractions**; there is no native binary floating point.
  "Approximate" numbers are just rationals whose decimal expansion is cut
  off/rounded to the requested precision.
- **Precision modes** (`Options > Mode Settings > Simplification`,
  state variables `Precision` / `PrecisionDigits`):
  - **Exact** (factory default) — irrational values (radicals, `pi`, `#e`,
    etc.) are kept symbolic; arbitrary-size exact rationals are used
    throughout; nested-radical denesting is attempted
    (`SQRT(5+2*SQRT(6)) -> SQRT(3)+SQRT(2)`).
  - **Approximate** — irrationals and large rationals are rounded to the
    simplest rational approximating them to the requested number of
    significant digits (arbitrary precision, user-settable digit count;
    time roughly quadruples each time precision doubles).
  - **Mixed** — irrational values are approximated but *rational*
    arithmetic stays exact, avoiding roundoff from rational operations
    while still forcing numeric answers for irrational subexpressions;
    documented to often be more accurate than full Approximate mode.
  - Explicit `APPROX(u, n)` function / Simplify > Approximate command to
    force a one-off numeric evaluation independent of the ambient mode.
  - The manual explicitly discusses **catastrophic cancellation** (e.g.
    `SQRT(10001)-100`) as a known accuracy hazard of approximate mode and
    recommends Mixed/Exact or higher precision as mitigation.
- **Complex numbers**: `#i` is the built-in imaginary unit (`i` itself is a
  free variable, deliberately, so users can name currents/rates `i`).
  `ABS`, `SIGN` (point on unit circle for complex args), `RE`, `IM`,
  `CONJ`, `PHASE` (principal argument, respecting the Angular Unit mode).
  Complex-vs-real behavior is governed by the Branch-field mode setting and
  by variable domain declarations (Complex vs. Real).
- **Radix / number base control**: independent input and output radix base
  settings (`InputBase`/`OutputBase`), any base 2-36, with Binary / Octal /
  Decimal / Hexadecimal shortcuts; digits above 9 use letters A-Z with a
  leading `0` to disambiguate from variable names.
- **Number theory** — built-in: `GCD`, `LCM` (n-ary, vector/matrix-
  distributing), `INVERSE_MOD`, `POWER_MOD`, `PRIME?`, `NEXT_PRIME`,
  `PREVIOUS_PRIME`. Extended in `NumberTheoryFunctions.mth`:
  `EXTENDED_GCD`, `SOLVE_MOD` (linear congruences), `CRT` (Chinese
  Remainder Theorem), `NTH_PRIME`, `PRIMEPI` (prime counting in
  arithmetic progressions), `FAREY` (Farey sequences), `DIVISORS`,
  `DIVISOR_SIGMA`/`DIVISOR_TAU`, `EULER_PHI` (totient), `PRIME_POWER?`,
  `PRIMITIVE_ROOT`, `MOEBIUS_MU`, `SQUAREFREE`, `CYCLOTOMIC` polynomials,
  generalized/standard **Lucas sequences** (`GEN_LUCAS`, `U_LUCAS`,
  `V_LUCAS`, modular variants `U_MOD`/`V_MOD`), `LUCAS`, `FIBONACCI`,
  `PELL` numbers, **Lucas-Lehmer** Mersenne-prime testing
  (`LUCAS_LEHMER`, `NEXT_MERSENNE_DEGREE`, `MERSENNE_LIST`,
  `MERSENNE_DEGREE`, `MERSENNE`), `PERFECT` numbers, continued fractions
  (`CONTINUED_FRACTION`, `CONVERGENT(S)`), Jacobi symbol (`JACOBI`), and
  modular square roots (`SQUARE_ROOT(a,p)`).
- Piecewise/integer building blocks used throughout: `FLOOR`, `CEILING`,
  `ROUND`, `MOD` (with an efficient modular-power fast path),
  `MODS` (symmetric mod), `POLY_MOD`/`POLY_MODS` (polynomial coefficient
  reduction mod n), `MIN`/`MAX` (n-ary/vector/matrix distributing).

---

## 5. Statistics, Probability, and Combinatorics

- **Descriptive statistics** (n-ary / vector- and matrix-distributing):
  `AVERAGE`, `RMS`, `VARIANCE` (unbiased sample variance, /(n-1)),
  `STDEV`.
- **Curve/regression fitting**: `FIT(labelVector, dataMatrix)` — general
  least-squares fit of an expression that is *linear in its parameters*
  (but may be arbitrarily nonlinear in the data variables) to a data
  matrix; exact fit when #rows = #parameters, least-squares otherwise.
  `GOODNESS_OF_FIT`, `POLY_INTERPOLATE` / `POLY_INTERPOLATE_EXPRESSION`
  (exact polynomial interpolation) in the utility library. Dedicated
  user-contributed packages add **logistic regression** and **nonlinear
  regression** (Gauss-Newton / Marquardt methods).
- **Probability / combinatorics built-ins**: factorial `z!` (real/complex
  argument via Gamma), `GAMMA`, `DIGAMMA`, `PERM(m,n)`, `COMB(m,n)`
  (binomial coefficient), `RANDOM(n)` pseudo-random generator (linear
  congruential, seedable via negative argument or time-seeded via 0).
  Extended distributions/densities in `ProbabilityFunctions.mth`:
  incomplete gamma/beta, Euler beta, Pochhammer symbol, polygamma family
  (trigamma..hexagamma), **Poisson**, **binomial**, **hypergeometric**
  density/cumulative-distribution pairs, **Student's t**, **F**, and
  **chi-square** cumulative distributions.
- **Combinatorial number families** (`CombinatorialFunctions.mth`):
  Catalan, Stirling numbers of the first/second kind (`STIRLING`,
  `STIRLING1`/`STIRLING_CYCLE`, `STIRLING2`/`STIRLING_SUBSET`), Bernoulli
  numbers/polynomials, Euler numbers/polynomials, integer partitions
  (`PARTS`, `PARTS_LIST`, `DISTINCT_PARTS`), Bell numbers, general
  linear recurrences (`RECURRENCE`), and an extensive family of figurate
  numbers (triangular, tetrahedral, pentatope, p-gonal, p-gonal pyramidal,
  hex, star, centered-p-gonal, octahedral, centered pyramids/cubes,
  rhombic dodecahedral, centered hex numbers).
- **Financial functions** (present-value/future-value annuity math):
  `PVAL`, `FVAL`, `PMT`, `NPER`, `RATE`, all built around a shared annuity
  equation with an optional "payment timing" argument (end/beginning/
  fractional-period payments).

---

## 6. Plotting

Two dedicated window types, each with their own menu/toolbar sets, opened
alongside (and driven from) the algebra window by highlighting an expression
and using Insert > Plot (or F4/toolbar).

- **2D-plot window** — plot type is inferred from the *shape* of the
  highlighted expression:
  - `y = u(x)` or a bare univariate expression -> **explicit** function plot.
  - `u(x,y) = v(x,y)` -> **implicit** plot (triangle-based linear
    interpolation algorithm; Derive first tries to convert to explicit form
    for speed/accuracy, only falling back to the implicit algorithm when it
    can't solve for one variable).
  - Inequalities / Boolean combinations of inequalities (≤2 vars) -> filled
    **region** plots.
  - `[a, b]` or a matrix of such pairs -> **data-point** plots (optionally
    connected into polylines, with small/medium/large point sizes).
  - 2-element vector `[u(t), v(t)]` -> **parametric** plot.
  - **Polar** coordinates are a `Set > Coordinate System` mode toggle (not a
    separate plot type): the same explicit/implicit/parametric plot logic
    then interprets the primary/secondary variables as *r* and *θ*.
  - Multiple simultaneous plots: any 3+-element vector of expressions plots
    each element separately (2-element vectors are parametric, so this is a
    deliberate 2 vs. 3+ distinction); all curves in one window share scale
    and color-cycle; **Trace mode** and cross-hair coordinate readout
    supported, including per-curve traversal with up/down arrows and
    parametric/polar traversal by parameter value.
  - **Contour plots** of a bivariate function are produced by plotting a
    generated *vector* of implicit level-curve equations,
    `VECTOR(z = u(x,y), z, m, n, s)`.
  - Non-directly-plottable expressions can be approximated via `TABLE(u, x,
    min, max, step)` plus "Approximate Before Plotting".
- **3D-plot window** — explicit surfaces `z = u(x,y)`, data-point plots
  (isolated points / connected polylines / functional or fully parametric
  surfaces depending on matrix shape), and parametric surfaces
  `[u(s,t), v(s,t), w(s,t)]`. Coordinate systems: rectangular, cylindrical,
  spherical. Surfaces are rendered as a shaded grid of quadrilateral
  panels (configurable panel/grid-line density in each parametric
  direction); mouse-driven real-time rotation, multiple color schemes
  (Rainbow, Wire Grid, Gray Scale, Red & Blue Checker, Heat Wave, Auto Plot
  Color, or fully Custom top/bottom/mesh coloring by z-value, gradient, or
  checkerboard).
- Shared plot features: adjustable plot range/aspect ratio/axes, per-curve
  color assignment and a default color-cycling palette, text/annotation
  objects overlaid on plots, slider bars (added in v6) to interactively
  animate a parameter and watch the plot update live, background images,
  and OLE embedding of plot objects into the worksheet or external
  documents (Word, etc.).
- A **Graphics Functions** utility library (`GraphicsFunctions.mth`) adds
  higher-level plotting helpers: parametrically defined space curves
  (helixes etc.), filled-area-under-curve plots for illustrating
  integrals, complex-valued expression plotting (real/imaginary parts),
  grid-line and grid-point generation, and polygon filling.

---

## 7. Units and Physical Constants

Not a built-in "unit type" system with dimensional-analysis checking in the
kernel — instead implemented as ordinary **utility-file variable
assignments** that multiply out to base units on simplification:
- `EnglishUnits.mth` — assigns numeric values to a large set of English/
  Imperial units (e.g. `furlong`, `fortnight`) so expressions combining
  them simplify to a foot-pound-second (FPS) basis.
- `MetricUnits.mth` — same idea for SI/metric units, normalizing to a
  meter-gram-second (MGS) basis; loading both files (English then Metric,
  or vice versa) causes automatic English<->Metric conversion since later
  unit definitions are expressed in terms of the base units of the
  first-loaded file.
- `PhysicalConstants.mth` — high-precision assignments for fundamental
  physical constants (speed of light, Planck's constant, gravitational
  constant, etc.); exact set of names is only visible by loading the file
  as a Math File rather than a silent Utility File.
- Because units are just algebraic variables with assigned magnitudes,
  "unit checking" is whatever falls out of ordinary simplification —
  there's no separate dimensional-consistency checker or unit-typed
  quantity object.

---

## 8. Programming, Extensibility, and the ".MTH" Library

### User-defined functions and variables
- **Author > Variable Value** (`name := expr`) assigns a variable;
  `name :=` (empty) clears it back to an unassigned symbol. Assignments are
  substituted transitively (assigning `r:=s^2` then `s:=5` makes `r`
  simplify to 25, not `s^2`).
- **Author > Function Definition** (`f(x,y,...) := expr`) defines
  functions; `f(x,y,...) :=` with empty body declares an **arbitrary
  ("undefined") function** — a first-class idiom used constantly to permit
  symbolic differentiation/manipulation of a not-yet-specified function
  (`F(x) :=`, then `DIF(F(x)^3,x)` uses the chain rule symbolically to
  produce `3*F'(x)*F(x)^2`).
  - Functions support **default argument values** in the parameter list
    (`SUMSQ(x, y:=0, z:=0) := x^2+y^2+z^2`), variadic definitions (a single
    un-parenthesized formal parameter collects all call arguments into a
    vector), and a quote operator (`'`) to suppress evaluation of a
    sub-argument before it's passed in (Lisp-style).
- Recursive and mutually recursive definitions are fully supported (with
  worked "naive vs. accumulator-based Fibonacci" efficiency examples in
  the docs); doubly-recursive naive definitions are explicitly warned
  against as exponential-blowup traps, with the accumulator-argument
  rewrite as the documented idiom.
- **Procedural programming**: `PROG(...)` (sequential statement block,
  returns last statement or an explicit `RETURN`) and `LOOP(...)`
  (repeats until `EXIT`/`RETURN`), with local-variable scoping tied to the
  enclosing function call, and update operators `:=`, `:+`, `:-`, `:*`,
  `:/` for in-place variable/element updates (including subscripted
  vector/matrix element updates `A SUB n := u`).
- **General-purpose iteration primitives** `ITERATE`/`ITERATES` repeatedly
  apply an update formula `x <- u(x)` starting from `x0`, either until
  convergence (`ITERATES` returns the whole history vector; stops when a
  value repeats) or for a fixed count; used throughout the utility library
  to implement Newton's method, Fibonacci, continued fractions, etc.
  without explicit loops.
- `IF(test, then, else, unknown)` is the core conditional primitive
  (arbitrarily nestable), with a 2-argument Boolean short-circuit
  evaluation model for `AND`/`OR` inside test clauses, and a documented
  idiom of leaving `else`/`unknown` off to get an informative "stuck"
  unevaluated expression when Derive can't decide the condition (rather
  than guessing).
- Reserved system/state-variable names (mode-setting identifiers like
  `Precision`, `Exact`, `VariableOrder`, etc.) cannot be reused as user
  variable/function names, alongside all built-in function/constant names.

### State, files, and persistence
- Two native worksheet formats: **.DFW** (full worksheet: expressions,
  plots, embedded OLE objects, and all mode/state-variable settings) and
  **.MTH** (plain math expressions/annotations only, i.e. a "program" or
  library file, loadable via File > Load > Utility File or Math File).
- "Algebra state variables" (input/simplification/output mode settings —
  precision, notation, branch, angular unit, variable order, display
  format, etc.) are themselves ordinary Derive assignment statements
  (e.g. `Precision := Approximate`) and are saved/restored per-worksheet
  in `.DFW`/`.MTH` files (in Derive 6; earlier versions stored them
  globally in the `.ini` file only).
- `Derive6.ini` — global application preferences (window layout, colors,
  math-file directory, memory allocation) separate from per-worksheet math
  state.

### The .MTH Utility Library (bundled with Derive)
Loading a `.mth` utility file (`File > Load > Utility File`) makes its
functions/variables available exactly like built-ins; most are also
"autoloading" — referencing one of their functions loads the file
automatically. TI's own bundled library, as enumerated by the online help's
"Utility File Library" index:

| File | Provides |
|---|---|
| `CombinatorialFunctions.mth` | Catalan/Stirling/Bernoulli/Euler numbers & polynomials, partitions, Bell numbers, figurate-number families, generic linear recurrences |
| `EquationSolving.mth` | Newton's method (`NEWTON`/`NEWTONS`), fixed-point iteration, Taylor-series local equation solving/inversion |
| `LinearAlgebra.mth` | Manual row-reduction primitives, cofactor/adjoint/minor, null space, exact & approximate eigenvectors |
| `NumberTheoryFunctions.mth` | Extended GCD, modular congruences/CRT, prime-counting, Farey sequences, divisor functions, totient, Lucas/Fibonacci/Pell/Mersenne/perfect numbers, continued fractions, Jacobi symbol, modular square roots |
| `RationalApproximation.mth` | Padé rational approximation, Chebyshev series approximation |
| `VectorMatrixFunctions.mth` | Outer product, Kronecker delta, column-append, partition, Jacobian/metric-tensor/geometry-matrix machinery for curvilinear coordinates, polar<->rectangular conversions, unit vectors `i_ j_ k_`, cylindrical/spherical constants |
| `MiscellaneousFunctions.mth` | Series-convergence ratio test, multivariate directional limit (`LIM2`), Riemann-sum primitives, integration-by-parts/substitution helpers, function inversion, inductive-proof helper (`PROVE_SUM`), polynomial coefficient/degree extraction, random poly/vector/matrix/normal generators, goodness-of-fit, polynomial interpolation |
| `DifferentiationApplications.mth` | Curvature, center of curvature, tangent/perpendicular lines, osculating circles — for explicit, parametric, polar, and implicit curves, plus tangent planes/normal lines to implicit surfaces |
| `IntegrationApplications.mth` | Laplace transforms, Fourier series approximation, arc length, area, volume, centroids, moments/inertia tensors (rectangular, cylindrical, spherical) |
| `NumericalApproximation.mth` | Numerical-differentiation and (per cross-refs) related numeric calculus helpers |
| `FirstOrderODEs.mth` | Exact first-order ODE solvers (general + named special-case methods; see Calculus section above) |
| `SecondOrderODEs.mth` | Exact second/linear ODE solvers (see Calculus section above) |
| `ODEApproximation.mth` | Taylor-series, Picard, Euler, and Runge-Kutta ODE approximation; direction fields |
| `RecurrenceEquations.mth` | Exact solvers for first/second-order difference equations, linear-geometric recurrences |
| `GraphicsFunctions.mth` | Space-curve/parametric-surface plotting helpers, area-under-curve fill plots, complex-plane plotting, grid generation, polygon filling; defines the `axes` 3D constant |
| `EnglishUnits.mth` | English/Imperial unit-to-FPS-base conversions |
| `MetricUnits.mth` | Metric/SI unit-to-MGS-base conversions |
| `PhysicalConstants.mth` | High-precision fundamental physical constants |
| `BesselFunctions.mth` | Bessel J/Y/I/K (incl. series & asymptotic forms), spherical Bessel functions, Airy functions |
| `EllipticIntegrals.mth` | Elliptic integrals of the 1st/2nd/3rd kind, Jacobi elliptic amplitude |
| `ExponentialIntegrals.mth` | Exponential-integral family (referenced as the accuracy-discussion anchor for the special-function files) |
| `FresnelIntegrals.mth` | Fresnel sine/cosine integrals |
| `HypergeometricFunctions.mth` | Generalized hypergeometric series, Kummer's & Gauss's hypergeometric functions |
| `OrthogonalPolynomials.mth` | Classical orthogonal polynomial families |
| `ProbabilityFunctions.mth` | Incomplete gamma/beta, polygamma family, Poisson/binomial/hypergeometric/Student-t/F/chi-square distributions |
| `ZetaFunctions.mth` | Riemann/Hurwitz zeta, dilogarithm |

Beyond the shipped library, Derive 6 also bundled a `Users\...` tree of
**user-contributed** (community-authored, not TI-supported) math packages,
notably: Complex Analysis, Discrete Mathematics (error-correcting codes,
multi-valued logic, network-graph drawing), Equation Solving extras,
Integration extras, Linear Algebra extras, Number Theory extras, Plotting
extras, **Regression Analysis** (logistic & nonlinear regression),
**Simplex Method** (linear programming), Special Functions extras,
**Tensor Algebra**, and Laplace/Z **Transforms** — i.e., several capability
areas (tensors, linear programming, general complex analysis, coding
theory) existed only as optional/unofficial add-ons, not the core product.

---

## 9. Distinctive Derive Idioms (vs. other CAS)

- **Assumption-driven, not context-driven, simplification.** Rather than a
  global "assume" store applied opportunistically (Mathematica-style) or
  implicit generic-domain assumptions (many CAS default to "complex,
  generic"), Derive ties every domain-sensitive simplification rule
  directly to an explicit per-variable domain/interval declaration, and
  refuses the transformation otherwise — favoring "don't guess" over
  "simplify aggressively and let the user catch mistakes."
- **Explicit-vs-simplified worksheet lines.** Every line typed is inert
  until a Simplify (or Approximate/Expand/Factor) command is invoked on it;
  results become new numbered lines (`#n`) that can be referenced from
  later expressions — a "notebook of labeled steps" model rather than an
  eagerly-evaluating REPL.
- **No implicit floating point** — "approximate" numbers are still exact
  rationals under the hood, just displayed/rounded to N significant
  digits; this is why simple exact fractions often "pop out" even from
  approximate-mode computation (documented and touted as an advantage over
  ordinary binary floating point).
- **Vectors of vectors as the only aggregate type** — matrices are not a
  distinct type from vectors, which simplifies a lot of the vector-function
  library (row/column extraction is just one/two levels of subscripting)
  but also means "vector" domain declarations are needed to get
  matrix-algebra identities applied symbolically to opaque variables.
- **Arbitrary/undefined functions as first-class objects** (`F(x) :=`
  with no body) specifically to support symbolic calculus (chain rule,
  etc.) on not-yet-specified functions — a lightweight universal-function
  idiom rather than a separate "assumption" that a symbol is a function.
  Similarly, undeclared free variables default to Real rather than Complex.
- **`unit_circle` and `@n` arbitrary-parameter placeholders** as
  first-class ways to represent "a value that could be anything
  satisfying constraint X," rather than returning an explicit infinite
  disjunction or a generic existential.
- **Menu/dialog-first UI, with functional syntax as a fallback.** Nearly
  every capability (Simplify/Factor/Expand/Solve/Differentiate/Integrate/
  Taylor/Sum/Product/plots) is available both from a menu command *and* as
  an equivalent typed function (`DIF`, `INT`, `TAYLOR`, `SUM`, `SOLVE`,
  ...), and the online help consistently documents both. A frequently
  cited weakness (see Limitations) is that dozens of other useful
  functions (e.g. `GAMMA`, most utility-file functions) have *no* menu
  exposure at all and must be typed from memory or looked up in help.

---

## 10. Notable Limitations

Compiled both from the software's own documented boundaries (FAQ answers,
"inapplicable" fallback returns, explicit caveats in the help text) and from
outside reviews:

- **No unified command palette / GUI discoverability gap.** A contemporary
  review (Scientific Computing World, on Derive 6) singled out as its
  chief flaw that "most functions aren't accessible via the front menu" —
  many built-in and virtually all utility-library functions must be typed
  from memory or looked up, unlike competitors' more example/palette-driven
  interfaces. The same review called Derive "mathematically narrower than
  Maple or Mathematica," roughly comparable to "Mathematica Lite"
  (CalculationCenter) in scope, with a Simplify command that lacks the
  fine-grained customization of the larger packages.
- **Integration is elementary-functions-only.** Antiderivatives are
  returned only if expressible via elementary functions plus Gamma,
  Digamma, Error, Zeta, and Dilogarithm — e.g. the sine integral `Si(x)`
  has no closed form in Derive and the definite integral of `sin(t)/t`
  from 0 to x will not simplify to it (explicitly called out in Derive's
  own FAQ); only numeric approximation is available for such cases.
- **No automatic detection of interior singularities in definite
  integrals.** Derive computes definite integrals by subtracting limits of
  the antiderivative at the endpoints; if a singularity lies *inside* the
  interval this can silently yield a Cauchy principal value (or something
  even less justified) instead of flagging divergence — the manual
  explicitly warns the user must find and split at interior singularities
  themselves (e.g. via Solve on the denominator, or by plotting first).
- **No native tensor algebra, linear programming/simplex, or general
  complex-analysis toolkit** — only available as unsupported,
  community-contributed packages, not the core symbolic engine.
- **No partial differential equation solving** of any kind (ODEs only;
  even ODEs above 2nd order require manual reduction to a first-order
  system — there's no automatic order-reduction).
- **Eigenvalues/eigenvectors are only reliably exact up to 4×4 matrices**
  (quartic-formula blowup) — larger matrices effectively require numeric
  methods (`APPROX_EIGENVECTOR`, `NSOLVE` on the characteristic
  polynomial).
- **Symbolic matrix inversion/determinants of symbolic matrices grow
  combinatorially** with size — the docs explicitly recommend
  `ROW_REDUCE`/`COFACTOR`-based element-at-a-time computation instead of
  naive `DET`/`A^(-1)` for anything beyond small symbolic matrices, citing
  memory exhaustion risk.
- **Approximate-mode arithmetic is exact-rational-based, not IEEE
  floating point** — a documented strength for avoiding certain rounding
  artifacts, but it also means performance/memory characteristics differ
  substantially from typical numeric software (rational numerator/
  denominator size can explode; time roughly quadruples per doubling of
  requested precision).
- **No dimensional/units type system** — "units" are just pre-defined
  variable magnitudes; nothing prevents adding a length to a mass, for
  instance, beyond what falls out of incidental simplification.
- **Sums/products assume integer bounds** — Derive explicitly assumes
  variables in `SUM`/`PRODUCT` limits are integers (to avoid `FLOOR`
  cluttering results), which can silently produce wrong answers if a
  non-integer value is later substituted for a limit variable.
- **Multivariate limits are iterated, not intrinsically multivariate** —
  `LIM(u, [x,y], [x0,y0])` computes an iterated limit (first in x, then in
  y) which can differ from the true multivariate limit or from evaluating
  in the other order; true direction-dependence must be checked manually
  with the utility function `LIM2` or by other means.
- Historically (per Wikipedia and general CAS-history commentary), Derive
  was positioned as a lightweight, inexpensive, low-memory-footprint
  "CAS for the rest of us" (particularly for DOS-era and classroom/
  handheld-adjacent use), explicitly trading off some symbolic depth
  (compared to Macsyma/Maple/Mathematica-class systems) for approachability,
  small size, and tight integration with TI's graphing/CAS calculator line.

---

## 11. Version History Notes (from the bundled revision summaries)

- **Derive 1.x-3.x** (1988-1996): DOS-only releases; Derive 3.14 (1996) is
  the last widely archived DOS version (full manual available, see
  manifest).
- **Derive 4/5**: Windows ports; Derive 5 introduced the modern Windows
  worksheet/plot-window UI described throughout this document.
- **Derive 6.0 (2003)**: added a fully scalable Unicode font and Unicode
  input/text support, per-worksheet (not just global) saved state
  variables, optional multi-line expression editing, customizable menus/
  toolbars/shortcut keys, TI-89/TI-92+/Voyage 200 handheld worksheet
  exchange, animated slider-bar plots, mouse-driven 3D plot rotation, and
  `GROEBNER_BASIS` for polynomial systems. This is also the version whose
  online help introduced the **Display Steps** transformation-rule
  tracing feature.
- **Derive 6.01 (March 2004)**: bug-fix/robustness release — expanded
  Display Steps coverage (integration, elementary-function simplification,
  integer factoring), faster Mersenne-number factoring, a conjugating dot
  product operator, broadened closed-form integrand coverage, L'Hopital
  improvements, and various correctness fixes (financial functions at 0%
  rate, arctan-of-two-arguments simplification, degenerate matrix-product
  associativity, etc.).
- **Derive 6.10 (October 2004, final release)**: Windows 98/ME/2000/XP
  compatibility, TI-89 Titanium USB support (bundled TI-Connect 1.5),
  further Display Steps coverage (trig functions, definite integrals,
  combinatorics, string functions), improved inverse-trig-of-algebraic-
  number simplification (with step display), live partial-factorization
  progress display, and memory-exhaustion fixes for factoring/expanding
  large expressions.
- Derive was discontinued 29 June 2007 in favor of TI-Nspire CAS, whose
  symbolic engine is a descendant of the Derive/muMATH lineage (as used
  earlier in the TI-89 / TI-92 Plus / Voyage 200 calculators).

---

## 12. Manifest of Downloaded Manuals / Documentation

All files stored under `/home/hniksic/work/derive/artifacts/manuals/`.

| File | Size | Description | Source URL |
|---|---|---|---|
| `Derive_3.14_User_Manual.pdf` | 5.5 MB, 385 pp. | Full official *Derive Version 3.14 User Manual* (Soft Warehouse, DOS version, 1996) — the most complete single classic-era manual found; covers algebra, calculus, matrices, plotting, and the utility-file library as they existed pre-Windows. | https://archive.org/download/derivecas314manual/Derive%20Version%203.14%20User%20Manual.pdf (archive.org item `derivecas314manual`) |
| `Derive_5_Introduction.pdf` | 1.3 MB, 58 pp. | Official TI *Introduction to Derive 5* booklet (getting-started tutorial, not a full reference) — Windows-era UI walkthrough. | https://education.ti.com/html/eguides/discontinued/computer-software/EN/Derive-5-Introduction.pdf |
| `Derive_6_Introduction.pdf` | 2.4 MB, 108 pp. | Official TI *Introduction to Derive 6* booklet — the getting-started companion referenced by the Derive 6 online help ("Introduction to Derive 6 book makes learning Derive quick and easy"); larger and more detailed than the v5 edition. | https://education.ti.com/html/eguides/discontinued/computer-software/EN/Derive-6-Introduction.pdf |
| `CalculusConceptsUsingDerive.pdf` | 1.8 MB, 220 pp. | *Calculus Concepts Using Derive for Windows* — University of Hawai'i Mathematics Department lab manual/course book teaching calculus via Derive, including a chapter of utility-function documentation (`add-util.mth`) and many worked examples. | https://math.hawaii.edu/~ralph/CalculusConceptsUsingDerive.pdf |
| `derive6.1-online-help/` (directory, ~408 HTML pages, ~13 MB) | — | Complete static-HTML mirror of **TI's official Derive 6.1 online help system** — this is the single most valuable source used for this research: a full command reference (every menu command in the Algebra/2D-plot/3D-plot windows), full built-in function & constant index, full utility-file (.MTH) library reference, FAQ, and version revision summaries, copyright 1988-2003 Texas Instruments Incorporated. Cloned from a community GitHub Pages mirror of the original Derive 6.1 `.hlp`/CHM help content. | https://github.com/WaluigiBSOD/derive6.1-online-help (mirrored from https://waluigibsod.github.io/derive6.1-online-help/); start at `derive6.1-online-help/docs/index.html` or `TableofContents.html` |

### Sources consulted but not saved as files
- Wikipedia, [Derive (computer algebra system)](https://en.wikipedia.org/wiki/Derive_(computer_algebra_system)) — history/version overview.
- Wikipedia, [Comparison of computer algebra systems](https://en.wikipedia.org/wiki/Comparison_of_computer_algebra_systems) — Derive is listed only in the general product/history table, not the detailed capability matrix.
- Scientific Computing World, ["Derive 6: Far too good just for students"](https://www.scientific-computing.com/feature/derive-6-far-too-good-just-students) — contemporary review; source for the Limitations section's UI/discoverability and "narrower than Maple/Mathematica" commentary.
- Internet Archive software items [derive314cas](https://archive.org/details/derive314cas) (Derive 3.14 executable + manual pointer) and [derivecas162](https://archive.org/details/derivecas162) (Derive 1.62, 1988, no separate manual found) — used to confirm version/format history; the runnable DOSBox images themselves were not downloaded (out of scope — documentation only).
- WinWorld [Derive 3.x](https://winworldpc.com/product/derive/3x) and [Derive 2.x](https://appserv.winworldpc.com/product/derive/2x) product pages — version/date cross-checks.
- University of Hawai'i Mathematics Department, [Derive utilities](https://math.hawaii.edu/wordpress/derive-utilities/) and [Derive](https://math.hawaii.edu/wordpress/derive/) pages — supplementary utility-file context.
- Derive User Group (DUG) newsletter archive at [austromath.at/dug](https://www.austromath.at/dug/) — large (100+ issue) archive of user-contributed technique articles; browsed but not downloaded given the comprehensiveness of the official help mirror already obtained; could be revisited for specific edge-case techniques if needed later.
- TI knowledge-base article ["Availability of the Full Downloadable Guidebook for Derive 6"](https://education.ti.com/en/customer-support/knowledge-base/sofware-apps/product-usage/20389) — confirms TI does not host a single combined "full guidebook" PDF beyond the Introduction booklet and the online help system (both of which were obtained above).

### Gaps not filled
- No official **Quick Reference Card** (a single-page command/syntax
  cheat-sheet, common for classroom software of this era) was located;
  it may not have been published as a standalone artifact, or may only
  exist in the University Derive Newsletter (DUG) archive, in physical
  campus bookstore materials, or on vetusware.com (checked indirectly via
  search; no direct hit surfaced).
- No separate official "Doing Calculus with Derive" / "Doing Linear
  Algebra with Derive" TI-branded companion book was found — only
  third-party textbooks of similar titles (e.g. *Calculus and the Derive
  Program* by Gilligan, *Elementary Linear Algebra with DERIVE* by Hill &
  Keagy, *Learning Mathematics Through Derive* by Derry), which are
  commercial textbooks not freely downloadable and were not pursued
  further given the comprehensiveness of the official online help already
  captured.
- Derive 1.x/2.x manuals were not located as separate downloadable PDFs
  (only the 3.14 manual was found scanned); version-history detail for
  the earliest 1988-1993 releases relies on secondary sources
  (Wikipedia/WinWorld) rather than primary documentation.
