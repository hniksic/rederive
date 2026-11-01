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
original, like `AccumBounds`. Each such class has a built-in written for it, so the
answer has a spelling that can be read and typed back. The complete list:

| name | sympy class | reached by |
|---|---|---|
| `ROOT_SUM(p, t, u)` | `RootSum` | rational integrand whose denominator does not factor |
| `ROOT_OF(p, t, n)` | `CRootOf` | authored only |
| `INTERVAL(a, b)` | `AccumBounds` | bounded limit that does not converge |
| `EI` | `Ei` | `INT(EXP(x)/x, x)` |
| `SI`, `CI` | `Si`, `Ci` | `INT(SIN(x)/x, x)`, `INT(COS(x)/x, x)` |
| `SINH_INT`, `COSH_INT` | `Shi`, `Chi` | `INT(SINH(x)/x, x)`, `INT(COSH(x)/x, x)` |
| `ERFI` | `erfi` | `INT(EXP(x^2), x)` |
| `DIGAMMA` | `polygamma(0, x)` | `DIF(GAMMA(x), x)` |
| `POLYGAMMA(n, x)` | `polygamma` | `DIF(DIGAMMA(x), x)` |
| `POLYLOG(s, x)` | `polylog` | `SUM(x^k/k^2, k, 1, inf)` |
| `LAMBERTW` | `LambertW` | authored only; differentiated and approximated |

`ROOT_SUM` binds `t` across *all* its arguments, unlike the other bound heads, since `p`
and `u` share the variable.

Interval arithmetic propagates through `INTERVAL`: `INTERVAL(-1, 1) + 1` is
`INTERVAL(0, 2)`.

The list is closed: an answer is written in these names and Derive's own, or not at all.
An operation whose answer would need any other function (Bessel, elliptic,
hypergeometric, ...) is left standing, as Derive leaves what it cannot do:
`INT(1/SQRT(1 - x^4), x)` stays as written. An answer sympy gives in a function that is
only another spelling of one of these is written in that one, so `INT(1/LN(x), x)` is
`EI(LN(x))`, not `li(x)`.

Derive shipped `SI` `CI` `EI` as `.MTH` definitions; the difference is that the engine
*produces* them.

## Where Rederive goes further

**Integration.** Nonelementary antiderivatives, refused by the original:

```
INT(SIN(x)/x, x)          SI(x)                          Derive: No elementary integral
INT(1/LN(x), x)           EI(LN(x))                      Derive: No elementary integral
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
