# Functions

Names ignore case. The examples show what Simplify gives.

## Elementary

```examples
LOG(8, 2)             3
SIN(30 deg)           1/2
FLOOR(7/2)            3
ABS(3 + 4 #i)         5
```

| Function | Meaning |
|---|---|
| `EXP(z)`, `LN(z)`, `SQRT(z)` | #e^z, natural logarithm, square root |
| `LOG(z)`, `LOG(z, w)` | natural logarithm; logarithm to base w |
| `SIN`, `COS`, `TAN`, `COT`, `SEC`, `CSC` | sine, cosine, tangent, cotangent, secant, cosecant |
| `ASIN`, `ACOS`, `ATAN`, `ACOT`, `ASEC`, `ACSC` | inverses, in radians; `ATAN(y, x)` and `ACOT(x, y)` are the angle of the point (x, y) |
| `ARCSIN`, `ARCCOS`, `ARCTAN`, `ARCCOT`, `ARCSEC`, `ARCCSC` | inverses, in degrees; `ARCTAN(y, x)` and `ARCCOT(x, y)` likewise |
| `SINH`, `COSH`, `TANH`, `COTH`, `SECH`, `CSCH` | the hyperbolic functions |
| `ASINH`, `ACOSH`, `ATANH`, `ACOTH`, `ASECH`, `ACSCH` | their inverses |
| `ABS(z)` | magnitude |
| `SIGN(z)` | -1, 0 or 1 by the sign of z; `SIGN(0)` is ±1 |
| `STEP(z)` | 0 below zero, 1 above |
| `CHI(a, z, b)` | 1 between a and b, 0 outside; `CHI(a, z, b, c, d)` has the value c at a and d at b, and without d, 1 - c at b |
| `IF(test, u, v)` | u where the test holds, v where it does not |
| `MAX(...)`, `MIN(...)` | of the arguments, or of a vector's or set's elements |
| `FLOOR(u)`, `FLOOR(u, v)` | greatest integer not above u, or not above u/v |
| `CEILING(u)`, `CEILING(u, v)` | least integer not below u, or not below u/v |
| `ROUND(u)`, `ROUND(u, v)` | integer nearest u, or nearest u/v; a half rounds up |
| `MOD(u, v)` | remainder of u divided by v; `MOD(u)` is the fractional part |
| `MODS(u, v)` | the same, symmetric about zero |
| `RE(z)`, `IM(z)`, `CONJ(z)` | real part, imaginary part, complex conjugate |
| `PHASE(z)` | angle with the positive real axis |

Angles are in radians; write degrees with deg, as in `SIN(30 deg)`.

## Algebra and solving

```examples
FACTOR(x^2 - 1)           (x + 1) (x - 1)
EXPAND((x + 1)^3)         x^3 + 3x^2 + 3x + 1
SOLVE(x^2 = 4, x)         x = -2 OR x = 2
NSOLVE(COS(x) = x, x)     x = 0.7390851332
```

| Function | Meaning |
|---|---|
| `FACTOR(u)`, `FACTOR(u, x, ...)` | the Factor command, over all variables or those named; `FACTOR(u, amount, x, ...)` goes as far as the amount says: Trivial, Squarefree, Rational, raDical or Complex; for a matrix, see [Vectors and matrices](help:functions/vectors-and-matrices) |
| `EXPAND(u)`, `EXPAND(u, x, ...)` | the Expand command; `EXPAND(u, amount, x, ...)` takes any amount but Complex |
| `APPROX(u)`, `APPROX(u, n)` | the approX command, at the set precision or at n digits |
| `TERMS(u)`, `FACTORS(u)` | the terms of a sum, the factors as [base, exponent] pairs, as a vector; of a logical u, the disjuncts and the conjuncts of its normal forms; with further arguments `a, x, ...`, after expanding or factoring by the amount a in the variables x, ... |
| `NUMERATOR(u)`, `DENOMINATOR(u)` | the top and the bottom of a fraction |
| `VARIABLES(u)` | the free variables, the main one first |
| `SUBST(u, v, w)` | u with w in place of each v |
| `QUOTIENT(u, v)`, `QUOTIENT(u, v, x)` | polynomial quotient of u by v, in x |
| `REMAINDER(u, v)`, `REMAINDER(u, v, x)` | the remainder |
| `POLY_GCD(u, v)` | greatest common divisor of two polynomials |
| `POLY_MOD(u, m)`, `POLY_MODS(u, m)` | u with its coefficients reduced by MOD or MODS |
| `GROEBNER_BASIS(v, [x, y, ...])` | reduced Groebner basis of the polynomials in v, lexicographic with x > y > ... |
| `SOLVE(u, x)` | solutions of u for x: `x = -1 OR x = 1`; true if every x solves u, false if none does |
| `SOLUTIONS(u, x)` | solutions as a vector of values: `[1, -1]`; for a system, a matrix with a row per solution |
| `NSOLVE(u, x)`, `NSOLUTIONS(u, x)` | numerically: every root of a polynomial, one real root of anything else |
| `LHS(u)`, `RHS(u)` | sides of a relation, or of each relation in a vector |
| `ROOT_OF(p, t, n)` | in answers: root number n of the polynomial p in t, counted from zero, real roots first |
| `ROOT_SUM(p, t, u)` | in answers: the sum of u over every root t of the polynomial p |

```
SOLVE(v, [x, y])            a system: [x = 2 AND y = 1]
SOLVE(u, x, Real)           real solutions only
SOLVE(u, x, a, b)           real solutions between a and b
SOLVE(u)                    for all of u's variables
NSOLVE(u, x, a, b)          a root between a and b
SUBST(u, v1, w1, v2, w2)    one pair after the other
SUBST(u, [v1, v2], [w1, w2])
                            all at once
```

## Calculus

```examples
DIF(SIN(x)^2, x)                SIN(2x)
INT(#e^(-x^2), x, -inf, inf)    SQRT(pi)
LIM(SIN(x)/x, x, 0)             1
SUM(1/k^2, k, 1, inf)           pi^2/6
```

| Function | Meaning |
|---|---|
| `DIF(u, x)`, `DIF(u, x, n)` | derivative, of the first or the nth order |
| `INT(u, x)` | indefinite integral; `INT(u, x, c)` adds the constant c; `INT(u, x, a, b)` is the definite integral from a to b |
| `LIM(u, x, a)` | limit at a; `LIM(u, x, a, d)` from the right if d > 0, from the left if d < 0 |
| `SUM(u, k, a, b)` | sum over k from a to b; `SUM(u, k, a, b, s)` in steps of s; `SUM(u, k, c)` and `SUM(v)` over the elements of a vector or set |
| `PRODUCT(u, k, a, b)` | product, with the same forms as SUM |
| `TAYLOR(u, x, a, n)` | Taylor polynomial of degree n about a; `TAYLOR(u, x, n)` about 0 |
| `SUBST_DIFF(u, x, a, b)` | u at x = b minus u at x = a, by limits where u is undefined |
| `GRAD(u)`, `DIV(v)`, `CURL(v)`, `LAPLACIAN(u)` | gradient, divergence, curl, Laplacian |
| `POTENTIAL(v)`, `POTENTIAL(v, p)` | a scalar whose gradient is v |
| `VECTOR_POTENTIAL(v)`, `VECTOR_POTENTIAL(v, p)` | a vector whose curl is v |
| `SUBS(u, [x, ...], [a, ...])` | in answers: u with a in place of x, where that has to wait: the derivative of an undefined function at a point |
| `INTERVAL(a, b)` | in answers: a value known only to lie between a and b, such as the limit of SIN(1/x) at zero |

GRAD, DIV, CURL and LAPLACIAN take a vector of coordinates as an optional second argument,
for cylindrical or spherical coordinates.

## Vectors and matrices

```examples
VECTOR(k^2, k, 1, 5)      [1, 4, 9, 16, 25]
[a, b, c] SUB 2           b
[1, 2, 3] . [4, 5, 6]     32
DET([1, 2; 3, 4])         -2
```

| Function | Meaning |
|---|---|
| `VECTOR(u, k, ...)` | u at each value of k |
| `MAP_LIST(u, k, ...)` | the same; over a set, gives a set |
| `SELECT(u, k, ...)` | the values of k for which the test u holds |
| `TABLE(u, k, ...)` | the rows [k, u] at each value of k |
| `ITERATES(u, x, a, n)` | a, u(a), u(u(a)), ... as a vector; without n, until the value repeats |
| `ITERATE(u, x, a, n)` | the last element of that vector |

The values of k are given in the same ways to each of these:

```
VECTOR(u, k, n)         k from 1 to n
VECTOR(u, k, a, b)      k from a to b
VECTOR(u, k, a, b, s)   k from a to b in steps of s
VECTOR(u, k, v)         k over the elements of v
VECTOR(u, [x, y], [a, c], [b, d])
                        matrix: x from a to b, y from c to d
[a, b, ..., c]          from a in steps of b - a, up to c
```

| Function | Meaning |
|---|---|
| `DIM(v)` | number of elements |
| `ELEMENT(v, i)` | element i, also written `v SUB i`; counted from the right if i is negative; for a vector of indices, those elements; `ELEMENT(m, i, j)` is element j of row i |
| `FIRST(v)` | the first element |
| `REST(v)`, `REST(v, n)` | all but the first element, or the first n; for a negative n, all but the last -n |
| `ADJOIN(u, v)` | u put before the elements of v |
| `APPEND(v, w, ...)` | the elements of each, as one vector |
| `REVERSE(v)` | the elements in reverse order |
| `INSERT(u, v, n)` | v with u before its nth element |
| `DELETE(v, n)`, `DELETE(v, w)` | v without its nth element, or without those at the indices in w |
| `REPLACE(u, v, n)` | v with u as its nth element |
| `POSITION(u, v)`, `POSITION(u, v, n)` | index of the first u in v, from the nth element on; false if none |
| `SORT(v)` | ascending: numbers, then variables and strings by name, then other expressions; `SORT(v, p, x, y)` orders by the test p: x goes before y where p holds |

A matrix is a vector of its rows. These functions also work on a string, as a vector of
characters, and on an integer, as a vector of digits: `"Rederive" SUB 3` is `"d"`, and
`REVERSE(123)` is 321.

| Function | Meaning |
|---|---|
| `STRING(u)` | a name or number as text; otherwise the vector of u's top operator and its operands: `STRING(x + y)` is `["+", x, y]` |
| `NAME_TO_CODES(s)` | the character codes: `NAME_TO_CODES("ab")` is `[97, 98]` |
| `CODES_TO_NAME(v)` | the string with the codes in v |

| Function | Meaning |
|---|---|
| `IDENTITY_MATRIX(n)` | the n by n identity matrix |
| `DOTPRODUCT` | `u DOTPRODUCT v`, or `u • v`: dot product with the elements of v conjugated |
| `CROSS(u, v)`, `CROSSPRODUCT` | cross product of three-element vectors, also written `u CROSSPRODUCT v` |
| `ROW`, `COL` | `m ROW i`, `m COL j`: row i, column j; for a vector of indices, the matrix of those rows or columns |
| `DET(m)`, `TRACE(m)`, `RANK(m)` | determinant, trace, rank |
| `ROW_REDUCE(m)`, `ROW_REDUCE(m, b)` | reduced echelon form |
| `CHARPOLY(m, z)` | characteristic polynomial in z |
| `EIGENVALUES(m)` | the eigenvalues; `EIGENVALUES(m, z)` gives them as equations in z |

`u.v` is the matrix product, or the dot product of two vectors, and a backtick after a
matrix transposes it.

`FACTOR(A, Turing)` is the matrix A as P . L . D . U . R: permutation, unit lower
triangular, pivots, unit upper triangular, reduced echelon form. `FACTOR(A, Schmidt)` is
A as Q . R by Gram-Schmidt.

## Sets

```examples
{1, 2} UNION {2, 3}       {1, 2, 3}
POWER_SET({a, b})         {{}, {a}, {b}, {a, b}}
```

| Function | Meaning |
|---|---|
| `{u, v, ...}` | a set; `{}` is the empty set |
| `{a, b, ..., c}` | a, b and on in steps of b - a, up to c; without b, the step is 1 |
| `UNION`, `INTERSECTION` | `s UNION t`, `s INTERSECTION t`; also typed `∪`, `∩` |
| `\` | `s\t`: the elements of s not in t |
| `SUBSET` | `s SUBSET t`, or `s ⊆ t`: whether every element of s is in t |
| `MEMBER?(u, s)` | whether u is an element of s |
| `POWER_SET(s)`, `POWER_SET(s, n)` | all subsets, or those of n elements |

`s*t` and `s^n` are Cartesian products, as sets of vectors, and a backtick after a set is
its complement. DIM, FIRST, REST and ADJOIN work on sets as on vectors. Elements are
compared after simplifying, so x + x and 2x are one element. AND, OR and NOT on sets are
the intersection, union and complement.

## Logic and tests

NOT, AND, OR, XOR, IMP and IFF are under
[Operators and constants](help:entering/operators-and-constants).

| Function | Meaning |
|---|---|
| `EVERY(u, k, ...)`, `SOME(u, k, ...)` | whether the test u holds at every value of k, or at one of them; the values of k are given as for [VECTOR](name:VECTOR) |
| `TRUTH_TABLE(p, q, ..., u, ...)` | truth table of the expressions u over the variables p, q, ... |

Tests of form say whether u, once simplified, is:

| Test | True for |
|---|---|
| `INTEGER?(u)`, `RATIONAL?(u)` | an integer, a rational number |
| `EVEN?(u)`, `ODD?(u)` | an even or odd integer |
| `NUMBER?(u)` | a rational number, or a + b #i with a and b rational |
| `VARIABLE?(u)`, `STRING?(u)` | a variable, a string |
| `SUM?(u)`, `PRODUCT?(u)` | a sum, a product or quotient |
| `POWER?(u)` | a power with a numeric exponent |
| `VECTOR?(u)`, `MATRIX?(u)` | a vector, a vector of rows of one length |
| `SET?(u)`, `LOGICAL?(u)` | an explicit set, true or false |
| `IDENTICAL?(u, v)` | u and v that simplify to the same expression |

Tests of type say whether u, for every value of its variables, is:

| Test | True for |
|---|---|
| `INTEGER_TYPE?(u)`, `REAL_TYPE?(u)` | an integer, a real number |
| `NUMBER_TYPE?(u)` | a number, not a vector, set, string or truth value |
| `VECTOR_TYPE?(u)` | a vector, or a variable declared Vector |
| `SET_TYPE?(u)` | a set, or a variable declared Set |
| `LOGICAL_TYPE?(u)` | a truth value, relation or logical operation, or a variable declared Logical |

## Number theory

```examples
GCD(12, 18)                 6
NEXT_PRIME(100)             101
```

| Function | Meaning |
|---|---|
| `GCD(...)`, `LCM(...)` | greatest common divisor, least common multiple |
| `PRIME?(n)` | whether n is prime |
| `NEXT_PRIME(n)`, `PREVIOUS_PRIME(n)` | the first prime above n, the last prime below n |
| `INVERSE_MOD(a, m)` | the x below m with MOD(a x, m) = 1, or ? if none |
| `POWER_MOD(a, n, m)` | MOD(a^n, m), without computing a^n |
| `SHIFT(n, k)` | n shifted left k bits, right if k is negative |

## Probability and statistics

```examples
COMB(5, 2)            10
AVERAGE(1, 2, 6)      3
NORMAL(0)             1/2
```

| Function | Meaning |
|---|---|
| `COMB(n, k)`, `PERM(n, k)` | combinations, permutations of n things k at a time |
| `AVERAGE(...)`, `RMS(...)` | mean, root mean square of the arguments or of a vector's elements |
| `VARIANCE(...)`, `STDEV(...)` | variance, standard deviation |
| `NORMAL(z)`, `NORMAL(z, m, s)` | cumulative normal distribution |
| `FIT(v, A)` | least squares fit to the rows of the data matrix A; v lists the data variables, then the expression to fit, whose other variables are solved for; `FIT(A)` takes v from the first row of A |
| `RANDOM(n)` | for n above 1, a random integer from 0 to n - 1; `RANDOM(1)` is a random number from 0 up to, not including, 1; `RANDOM(-n)` restarts the random sequence from the seed n, and `RANDOM(0)` from the clock |

## Finance

| Function | Meaning |
|---|---|
| `PVAL(i, n, p, f, t)` | present value |
| `FVAL(i, n, p, v, t)` | future value |
| `PMT(i, n, v, f, t)` | payment per period |
| `NPER(i, p, v, f, t)` | number of periods |
| `RATE(n, p, v, f, t)` | interest rate per period, sought between 0 and 1; `RATE(n, p, v, f, t, a, b)` between a and b |

Here i is the interest rate per period, n the number of periods, p the payment per period,
v and f the present and future value, and t is 1 if payments are made at the start of each
period, 0 if at the end. Money received and money paid have opposite signs. Arguments left
out at the end count as zero. RATE works only on numbers.

## Special functions

```examples
GAMMA(5)              24
ZETA(2)               pi^2/6
INT(SIN(x)/x, x)      SI(x)
INT(#e^(x^2), x)      SQRT(pi) ERFI(x)/2
```

| Function | Meaning |
|---|---|
| `ERF(z)`, `ERFC(z)` | error function and its complement |
| `ERFI(z)` | imaginary error function, -#i ERF(#i z) |
| `GAMMA(z)`, `DIGAMMA(z)` | gamma function and its logarithmic derivative |
| `POLYGAMMA(n, z)` | nth derivative of DIGAMMA(z) |
| `ZETA(z)` | Riemann zeta function; `ZETA(s, z)` is the Hurwitz zeta function, the sum of 1/(k + z)^s from k = 0 |
| `DILOG(z)` | dilogarithm, POLYLOG(2, 1 - z) |
| `POLYLOG(s, z)` | polylogarithm, the sum of z^k/k^s from k = 1; `POLYLOG(s, z, m)` is the same sum to k = m |
| `EI(z)` | exponential integral, of EXP(t)/t; `EI(z, m)` is its series to m terms |
| `SI(z)`, `CI(z)` | sine and cosine integrals, of SIN(t)/t and COS(t)/t |
| `SINH_INT(z)`, `COSH_INT(z)` | hyperbolic sine and cosine integrals |
| `LAMBERTW(z)` | the w with w EXP(w) = z |

## Utility library

The functions of Derive's utility files are defined in every session, without loading
anything. A definition of your own under the same name replaces one. The table names the
main ones of about 400.

```examples
DIVISORS(12)              [1, 2, 3, 4, 6, 12]
TANGENT(x^2, x, 1)        2x - 1
```

| Area | Functions |
|---|---|
| Number theory | `DIVISORS(n)`, `EULER_PHI(n)`, `NTH_PRIME(n)`, `FIBONACCI(n)`, `CRT(a, m)` |
| Combinatorics | `CATALAN(n)`, `BERNOULLI(n)`, `BELL(n)`, `STIRLING2(n, k)` |
| Equations | `NEWTON(u, x, x0, n)`, `FIXED_POINT(g, x, x0, n)` |
| Linear algebra | `ADJOINT(M)`, `NULL_SPACE(M)`, `EXACT_EIGENVECTOR(M, λ)`, `JACOBIAN(u, α)` |
| Curves | `TANGENT(y, x, x0)`, `PERPENDICULAR(y, x, x0)`, `CURVATURE(y, x)`, `IMP_DIF(u, x, y)` |
| Integration | `ARC_LENGTH(y, x, x1, x2)`, `VOLUME_OF_REVOLUTION(y, x, x1, x2)`, `INT_PARTS(u, v, x)` |
| Transforms | `LAPLACE(y, t, s)`, `FOURIER(y, t, t1, t2, n)`, `PADE(y, x, x0, n, d)` |
| Data | `POLY_INTERPOLATE(A, x)`, `DIF_DATA(A)`, `INT_DATA(A)` |
| Differential equations | `DSOLVE1(p, q, x, y, x0, y0)`, `DSOLVE2(p, q, r, x, c1, c2)`, `RK(r, v, v0, h, n)` |
| Recurrences | `LIN1_DIFFERENCE(p, q, x, x0, y0)`, `LIN2_CCF(p, q, r, x, c1, c2)` |
| Plot shapes | `SPHERE(r, θ, φ)`, `CYLINDER(r, θ, z)`, `TORUS(rc, rs, θ, φ)` |
| Special functions | `BESSEL_J(n, z)`, `BESSEL_Y(n, z)`, `ELLIPTIC_F(phi, m)`, `FRESNEL_SIN(z)`, `KUMMER(a, b, z)`, `GAUSS(a, b, c, z)` |
| Orthogonal polynomials | `CHEBYCHEV_T(n, x)`, `LEGENDRE_P(n, x)`, `HERMITE_H(n, x)`, `LAGUERRE_L(n, x)` |
| Probability | `BINOMIAL_DENSITY(k, n, p)`, `POISSON_DENSITY(k, t)`, `STUDENT(t, v)`, `CHI_SQUARE(x2, v)` |
