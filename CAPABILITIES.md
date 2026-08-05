# Capabilities

Rederive computes through sympy, whose CAS reaches well past the one Derive shipped with.
This file is about that surplus: the built-ins added to spell answers the original had no
name for, and the places where the engine computes what it refused.

## Derive's built-ins

The 110 names the original defines, all supported:

| | |
|---|---|
| Arithmetic, number theory | `ABS` `SIGN` `FLOOR` `MOD` `MODS` `GCD` `LCM` `NUMERATOR` `DENOMINATOR` `QUOTIENT` `REMAINDER` `NEXT_PRIME` `PRIME` `NUMBER` `RANDOM` `MAX` `MIN` |
| Elementary | `SIN` `COS` `TAN` `COT` `SEC` `CSC` and inverses, `SINH` … `CSCH` and inverses, `EXP` `LN` `LOG` `SQRT` |
| Special | `GAMMA` `ERF` `ERFC` `ZETA` `CHI` `STEP` |
| Complex | `RE` `IM` `CONJ` `PHASE` |
| Calculus | `DIF` `INT` `LIM` `SUM` `PRODUCT` `TAYLOR` |
| Algebra | `EXPAND` `FACTOR` `FACTORS` `SOLVE` `NORMAL` `TERMS` `APPROX` `POLY_GCD` `VARIABLES` |
| Vectors, matrices | `VECTOR` `ELEMENT` `DIMENSION` `APPEND` `DET` `TRACE` `CHARPOLY` `EIGENVALUES` `IDENTITY_MATRIX` `ROW_REDUCE` `CROSS` `DELETE_ELEMENT` `INSERT_ELEMENT` `REPLACE_ELEMENT` `REVERSE_VECTOR` `SELECT` |
| Vector calculus | `GRAD` `DIV` `CURL` `LAPLACIAN` `POTENTIAL` `VECTOR_POTENTIAL` |
| Statistics | `AVERAGE` `RMS` `VAR` `STDEV` `FIT` |
| Finance | `PVAL` `FVAL` `PMT` `NPER` `RATE` |
| Logic, control | `IF` `TRUTH_TABLE` `LHS` `RHS` `ITERATE` `ITERATES` |
| Combinatorics | `COMB` `PERM` |

Bound-variable heads are `DIF INT ITERATE ITERATES LIM PRODUCT SELECT SUM TAYLOR VECTOR`,
binding their second argument over the first.

## Added built-ins

Sympy answers in its own vocabulary, and some of what it returns has no name in the
original, like `AccumBounds`. Each such class has a built-in written for it, so the answer
has a spelling that can be read and typed back. The complete list:

| name | sympy class | reached by |
|---|---|---|
| `ROOT_SUM(p, t, u)` | `RootSum` | rational integrand whose denominator does not factor |
| `ROOT_OF(p, t, n)` | `CRootOf` | authored only |
| `INTERVAL(a, b)` | `AccumBounds` | bounded limit that does not converge |
| `COSH_INT`, `SINH_INT` | `Chi`, `Shi` | `INT(COSH(x)/x, x)` |

`ROOT_SUM` binds `t` across *all* its arguments, unlike the other bound heads, since `p`
and `u` share the variable.

Interval arithmetic propagates through `INTERVAL`: `INTERVAL(-1, 1) + 1` is
`INTERVAL(0, 2)`.

Beyond these, 118 sympy function classes are readable and printable under their
upper-cased name, so anything an answer produces has a spelling: `SI` `CI` `EI` `LI`
`FRESNELS` `FRESNELC` `POLYLOG` `LAMBERTW` `ELLIPTIC_K/F/E` `BESSELJ` `BESSELY` `AIRYAI`
`UPPERGAMMA` `LOWERGAMMA` `LERCHPHI` `POLYGAMMA` `DIRACDELTA` `FACTORIAL2` `EXP_POLAR`,
`HYPER`, `MEIJERG`. `LI` is the logarithmic integral `li`, not sympy's offset `Li`.

Derive shipped `SI` `CI` `EI` `LI` `ELLIPTIC_*` `BESSEL_*` as `.MTH` definitions; the
difference is that the engine now *produces* them.

## Where Rederive goes further

**Integration.** Nonelementary antiderivatives, refused by the original:

```
INT(SIN(x)/x, x)          SI(x)                          Derive: No elementary integral
INT(1/LN(x), x)           LI(x)                          Derive: No elementary integral
INT(SIN(x^2), x)          √2·√π·FRESNELS(√2·x/√π)/2      Derive: No elementary integral
INT(1/(x^5 - x - 1), x)   ROOT_SUM(2869·t^5 + …)         Derive: unevaluated
```

Definite integrals via Meijer G:

```
INT(#e^(-x^2)·COS(2x), x, 0, inf)   √π·#e^(-1)/2     Derive: unevaluated
INT(COS(x^2), x, 0, inf)            √2·√π/4          Derive: No elementary integral
INT(#e^(-x)·LN(x), x, 0, inf)       -euler_gamma     Derive: No elementary integral
INT(SIN(x)^2/x^2, x, 0, inf)        pi/2             Derive: ?   (i.e. wrong)
```

**Summation.** Hypergeometric closed forms, guarded by their convergence condition. All
three are left standing by the original:

```
SUM(x^k/k, k, 1, inf)         IF(x /= 1 AND ABS(x) <= 1, -LN(1 - x), …)
SUM(k^2·x^k, k, 1, inf)       IF(ABS(x) < 1, -x·(x + 1)/(x - 1)^3, …)
SUM(1/(k^2 + 1), k, 1, inf)   pi·(#e^(2·pi) + 1)/(2·#e^(2·pi) - 2) - 1/2
```

**Limits.** `LIM(SIN(1/x), x, 0)` and `LIM(SIN(x), x, inf)` are `INTERVAL(-1, 1)`; Derive
answers `SIN(∞)`.

**Approximation.** `APPROX` evaluates what has no exact answer: `APPROX(SI(2))`,
`APPROX(INT(SIN(x)/x, x, 1, 2))`, `APPROX(SUM(1/(k^3 + 1), k, 1, inf))`. Only known
infinities are refused.

**Size.** `EXPAND((1 + x + y + z)^n)`:

| n | terms | Rederive |
|---|---|---|
| 50 | 23 426 | 2 s |
| 60 | 39 711 | 24 s |
| 100 | 176 851 | 121 s |

Derive expands the first of these; the largest exceeds the memory it can address, and it
answers `Memory Full`.

## Plotting

The plotter is Rederive's own, over [pyqtgraph](https://www.pyqtgraph.org/), and sympy is
in it only to convert an expression into something `lambdify` can turn into a numeric
closure.

**What a shape is drawn as.** One command, `Plot`, reads the highlighted expression and
decides; the first matching row wins.

| shape | drawn as |
|---|---|
| scalar in 0 or 1 variables | a curve, the variable naming the horizontal axis |
| vector of ≥3 scalars in one common variable | one curve per element, colors cycling |
| 2-vector in exactly one variable | a parametric curve, over one turn by default |
| scalar in one variable, window in polar mode | `r = f(θ)` |
| `u = v` in at most two variables | the zero contour of `u - v`, by marching squares |
| inequality, or a boolean combination of them | the region it holds on, shaded |
| n x 2 matrix of constants | points, joined into a polyline on request |
| scalar in exactly two variables | a shaded surface, several to a window |
| vector of ≥2 scalars in the same two variables | one surface per element |
| equation with one side a lone variable | a surface, that variable naming the vertical axis |

**Sampling is redone at every scale.** Each plot keeps its lambdified closure, so a view
change re-evaluates over the new range instead of stretching pixels: 129 uniform samples,
then adaptive bisection until every chord sits within a quarter pixel of the curve, capped
at depth 12, and the same in the plane for parametric and polar curves. A spike narrower
than a pixel is therefore not lost - zoom towards it and it appears. Implicit contours and
regions are recomputed on the same schedule, so zooming into `x^2 + y^2 = 9` reveals its
detail rather than its polygon.

**Corner cases are drawn as what they are.** `TAN(x)` and `SIGN(x)` gap at their jumps
instead of bridging them with a vertical stroke. `SQRT(x)` and `x^(1/3)` are evaluated
over the complex plane and masked back to their real part, so the negative half is absent
rather than drawn as zero. A surface's mesh drops the faces that touch a non-real vertex,
which is why `SQRT(1 - x^2 - y^2)` stops cleanly at the unit circle instead of skirting
the floor.

**Nothing is silent.** An expression in too many variables is refused by name. A curve
with no real value in view says so and names the range it looked over. A surface whose
spikes would crush the box is clipped to the 1st-99th percentile and says that too. An
empty picture with no explanation is the named anti-goal.

**Numbers come off the closure, not off the pixels.** Trace rides a curve and reads out
`x` and `f(x)` at full precision, freshly evaluated at the exact x. `Tab` jumps to the
next root, local extremum or intersection with another plotted curve: candidates are found
by sign changes in the sampled arrays, then refined on the closure by bisection or a
parabolic fit. A sign change across a NaN gap is a pole and is never offered as a root.

**Deliberately absent.** No accuracy setting - sampling is always sub-pixel. No framing
dialogs; the mouse, and the stock context menu's per-axis min/max fields, are the whole
vocabulary. No persisted view state, so every new window frames the same [-5, 5] world and
a plot reads the same at a glance. `Options Plot` keeps only four preferences, and they
apply to the next window and the next plot rather than to what is already drawn.

**Where it runs.** Plotting needs a graphical display: on Linux, `DISPLAY` or
`WAYLAND_DISPLAY` has to be set. Where it is not, the Plot command stays on the menu - a
command that disappears is a command nobody learns - and refuses with `Plot: needs a
graphical display`. A broken install says so in its own words instead. The dependencies
are `pyside6`, `pyqtgraph`, `pyopengl` and `numpy`; none of them is imported by the
terminal program, which stays as free of Qt as it is of sympy. All the plotting lives in
one child process, spawned the first time something is plotted.
