# Capabilities

What Rederive can compute, and where that differs from Derive 4.11.

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

## Added names

Rederive computes through sympy, so each added name is author notation for a sympy class.

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
`APPROX(INT(SIN(x)/x, x, 1, 2))`, `APPROX(PRODUCT(1 - 1/k^2, k, 2, inf))`. Only known
infinities are refused.

**Size.** `EXPAND((1 + x + y + z)^n)`:

| n | terms | Rederive |
|---|---|---|
| 50 | 23 426 | 2 s |
| 60 | 39 711 | 24 s |
| 100 | 176 851 | 121 s |

Derive expands the first of these; the largest exceeds the memory it can address, and it
answers `Memory Full`.

## Where Derive is still ahead

- Infinite products: `PRODUCT(1 - 1/k^2, k, 2, inf)` is `1/2` there, unevaluated here
  (`APPROX` gives `0.5`).
- Integer factoring: ECM given time, versus a bounded search here that gives a number back
  unfactored rather than run indefinitely.
- Casus irreducibilis: the real roots of an irreducible cubic are written trigonometrically
  there and as nested radicals over `#i` here, so a real eigenvalue of a symmetric matrix
  is spelled with an imaginary unit in it.
