# Ideas from Derive 6.1

Features of Derive 5 and 6 that Rederive might adopt, taken from the Derive 6.1 online
help (https://waluigibsod.github.io/derive6.1-online-help/). Each entry states what 6.1
specifies, quoting the help page named, what Rederive does today, and the suggestion.
The suggestions say what might be worth having, not how to build it; design questions
are left open, and so are the reasons given for not suggesting something. Where the
help says nothing beyond a changelog line, the entry says so.

One cost applies to every new builtin name: in Character mode it changes how runs of
letters split, as `CI` makes `velocity` read as `v*e*l*o*CI(t)*y`.

## Suggested

### Default parameter values

6.1 (ProceduralProgramming): "The default initial value of a user-defined function's
local variables can be specified by including assignment statements in the function's
formal argument list. If a function is called with fewer actual arguments than it has
local variables, the extra local variables are given their assigned values."

```
SUMSQ(x, y := 0, z := 0) := x^2 + y^2 + z^2
[SUMSQ(5), SUMSQ(5, 3), SUMSQ(5, 3, 2)]     [25, 34, 38]
```

A default may precede a parameter without one, and a call that stops short leaves the
rest unbound:

```
FOO(x, y := 3, z) :=
FOO(5)                                       FOO(5, 3, z)
```

The help's other example, `Rev(v, w := []) := IF(v = [], w, Rev(REST(v),
ADJOIN(FIRST(v), w)))`, uses a default to supply a recursion's accumulator.

Rederive: a parameter list cannot hold `:=`; the definition is a syntax error. Without
defaults, `[SUMSQ(5), SUMSQ(5, 3), SUMSQ(5, 3, 2)]` gives 6.1's partial result
`[y^2 + z^2 + 25, z^2 + 34, 38]`.

Suggestion: adopt.

Note: README.md lists "no local variables" among the language's limitations.

### IFF

6.1 (BooleanConstantsandOperators, Operators): `p IFF q`, displayed `p ↔ q`, is
logical equivalence, a binary infix operator with the lowest precedence of the Boolean
operators (NOT, AND, OR, XOR, IMP, IFF, in decreasing order). "If both p and q are true
or if both p and q are false, p IFF q simplifies to true. If p is true and q is false or
if p is false and q is true, p IFF q simplifies to false. If the truth values of p and
q are unknown, p IFF q simplifies to the equivalent Boolean expression (¬ p OR q) AND
(p OR ¬ q)."

Rederive: has NOT, AND, OR, XOR and IMP; `p IFF q` reads as `p*q*IF(f)`.

Suggestion: adopt.

### CEILING and ROUND

6.1 (PiecewiseContinuousFunctions):

- "CEILING(m) simplifies to the smallest integer greater than or equal to m.
  CEILING(m, n) is equivalent to CEILING(m/n) ... If m or n are not numbers,
  CEILING(m, n) simplifies to the equivalent expression -FLOOR(-m/n)."
- "ROUND(m) simplifies to the nearest integer to m. ROUND(m, n) is equivalent to
  ROUND(m/n)." The help does not say how a tie (m/n halfway between integers) rounds,
  nor what a nonnumeric ROUND simplifies to.

```
CEILING(5.73)       6
CEILING(573, 100)   6
ROUND(5.73)         6
ROUND(573, 100)     6
```

Rederive: has `FLOOR(m)` and `FLOOR(m, n)`, neither CEILING nor ROUND;
`CEILING(5.73)` reads as a product of letters. NUMBER.MTH defines
`CEILING(a, default1) := -FLOOR(-a, default1)` and
`ROUND(a, default1) := FLOOR(a/default1 + 1/2)`.

Suggestion: adopt both. The tie rule and the nonnumeric form of ROUND are open.

### PREVIOUS_PRIME

6.1 (NumberTheoryFunctionsBFC): `PREVIOUS_PRIME(n)` is the largest prime less than n;
`PREVIOUS_PRIME(1000)` is 997.

Rederive: has `NEXT_PRIME` only.

Suggestion: adopt.

### Renamed vector functions

6.1 renames five functions; arguments and meaning are unchanged:

| 6.1 | Rederive |
|---|---|
| `DIM(v)`, number of elements (rows of a matrix) | `DIMENSION(v)` |
| `DELETE(v, n)`, v without element n (row n of a matrix) | `DELETE_ELEMENT(v, n)` |
| `INSERT(u, v, n)`, u inserted before element n of v | `INSERT_ELEMENT(u, v, n)` |
| `REPLACE(u, v, n)`, element n of v replaced by u | `REPLACE_ELEMENT(u, v, n)` |
| `VARIANCE(z1, ..., zn)`, unbiased sample variance | `VAR(z1, ..., zn)` |

(VectorManipulationFunctions, StatisticalFunctions.) 6.1 also renames REVERSE_VECTOR to
`REVERSE(v)`; see the note under Rederive. 6.1 extends `DELETE(v, w)` to a vector w of
indices, deleting each: `DELETE([a, b, c, d], [2, -1])` is `[a, c]`.

Rederive: has only the old names; `DIM([1,2,3])` reads as `d*IM([1,2,3])`, and
`DELETE(...)` is a syntax error. The DOS utility file SPLINE.MTH defines its own
`REVERSE(m_)`, which reverses the first column of a matrix.

Suggestion: accept the new names, so 5/6-era files load. Whether to include REVERSE,
given SPLINE.MTH, and whether to take the vector-of-indices form of DELETE, are open.

### Negative indices

6.1 (VectorManipulationFunctions): "If an index is a negative integer n, it refers to
the -nth element from the right end of the vector." This applies to the vector
manipulation functions and operators that take an index (ELEMENT, SUB, DELETE, INSERT,
REPLACE).

```
DELETE([a, b, c, d], [2, -1])     [a, c]
```

Rederive: a negative index leaves the call standing (`ELEMENT([a, b, c, d], -1)`), and
`v SUB -1` is a syntax error (`v SUB (-1)` parses and stands).

Suggestion: adopt.

### Greek letter names

6.1 (EnteringGreekandMathCharacters, Input): every Greek letter can be typed by its
Latin name, lower case for the lower-case letter (`gamma` for γ) and capitalized for
the upper-case one (`Gamma` for Γ). Case is respected for Greek names even when case is
otherwise ignored. The exception is `pi`, which is the constant; the variable π can only
be entered as the glyph.

Rederive: knows ten lower-case names (alpha, beta, delta, epsilon, theta, mu, phi,
sigma, tau, omega); `gamma`, `zeta` and `chi` are functions. `lambda` reads as
`a^2*b*d*l*m`; `Delta` reads as `delta`.

Suggestion: add more names, `lambda` above all. Which names, and upper case, are open.

### Functions over vectors and equations

6.1: the changelogs say "Uniformly handle the case when nonscalar arguments (e.g.
equations, vectors, sets) are given to the elementary, special, statistical, integer,
and Calculus functions" (6.01); "If ... the TAYLOR function is given an equation or
vector, the Taylor series of the elements of the equation or vector are returned"
(5.04); "RE, IM, CONJ, and PHASE now distribute over the elements of vectors and
matrices" (5.06). The help gives no further rule.

Rederive, by function:

```
FLOOR([1.5, 2.5])            [1, 2]
DIF([x^2, x^3], x)           [2*x, 3*x^2]
INT([x, x^2], x, 0, 1)       [1/2, 1/3]
SIN([0, pi])                 SIN([0, pi])
SQRT([4, 9])                 SQRT([4, 9])
MOD([5, 7], 3)               MOD([5, 7], 3)
PHASE([#i, -1])              PHASE([#i, -1])
INT([x, x^2], x)             INT([x, x^2], x)
TAYLOR([SIN(x), COS(x)], x, 0, 3)   unchanged
SIN(x = y)                   SIN(x) = SIN(y)
TAYLOR(SIN(x) = x, x, 0, 3)  unchanged
```

Suggestion: make the behavior uniform.

### Integration constant

6.1 (Integrate): "The antiderivative of the expression u with respect to the variable x
with the integration constant c can be entered by typing in an expression of the form
INT(u, x, c)."

Rederive: `INT(x, x, c)` stays as written.

Suggestion: adopt.

### Antiderivatives of piecewise functions

6.1 (Integrate) lists "piecewise continuous functions" among the integrands it handles;
the 6.01 changelog says "Significantly expand the class of integrands for which
continuous antiderivatives can be found." No examples or rules are given.

Rederive: `INT(ABS(SIN(x)), x)` is `2*FLOOR(x/pi) - SIGN(SIN(x))*COS(x)`, but
`INT(ABS(x), x)`, `INT(ABS(x - 1), x)` and `INT(x*SIGN(x), x)` stay as written, while
`DIF(x*ABS(x), x)` is `2*ABS(x)`.

Suggestion: adopt. Which piecewise functions and arguments to cover is open, since 6.1
does not say.

### Real solutions only

6.1 (SolvingEquationsandInequalitiesA, SolvingEquationsNumerically): "If the third
argument is the constant Real, only solutions not having an imaginary component are
returned." `Real` is the only third argument documented for SOLVE, SOLUTIONS, NSOLVE and
NSOLUTIONS. As before, "The set of solutions returned is not restricted by the declared
domains of the solution variables."

```
SOLVE(x^3 + x^2 + x + 1, x)          x = -î ∨ x = î ∨ x = -1
SOLVE(x^3 + x^2 + x + 1, x, Real)    x = -1
```

Rederive: drops the third argument; `SOLVE(x^2 + 1 = 0, x, Real)` gives
`[x = #i, x = -#i]`. Solutions are not filtered by declared domain either, so there is
no way to ask for real solutions only.

Suggestion: adopt.

### Solving a conjunction

6.1 (SolvingSystemsofEquations): SOLVE accepts a system as a conjunction of equations
as well as a vector of them, and answers a conjunction in disjunctive normal form:

```
SOLVE(x^2 + y^2 = 2a^2 AND x - y = 0, [x, y])     (x = -a ∧ y = -a) ∨ (x = a ∧ y = a)
SOLVE([x^2 + y^2 = 2a^2, x - y = 0], [x, y])      [x = a ∧ y = a, x = -a ∧ y = -a]
```

A disjunction of equations as input is not documented.

Rederive: solves the vector form, answering a vector of vectors; the conjunction becomes
`[(x + y = 3 AND x - y = 1) = 0]` for `SOLVE(x+y=3 AND x-y=1, [x, y])`.

Suggestion: accept a conjunction as a system. The form of the answer is open.

### Inconsistent nonlinear systems

6.1 (SolvingSystemsofEquations): SOLVE reduces a system of polynomial equations with a
Gröbner basis. An inconsistent system answers `false` or `[]`, depending on how it is
given; the help's example is linear:

```
SOLVE(x + 3·y = 1 AND 2·x + 6·y = 3, [x, y])     false
SOLVE([x + 3·y = 1, 2·x + 6·y = 3], [x, y])      []
```

GrobnerBasis gives a nonlinear inconsistent system, whose basis is `[1]`:
`GROEBNER_BASIS([x^2+y^2-z+1, x^2+y^2-z+3], [x, y, z])`.

Rederive: answers `[]` for an inconsistent linear system, but returns a nonlinear one
unsolved: `SOLVE([x^2+y=1, x^2+y=2], [x,y])` is
`[[x^2 + y - 1 = 0, x^2 + y - 2 = 0]]`.

Suggestion: answer `[]` for inconsistent nonlinear systems too.

### Cube-root equations

6.1: the 5.02 changelog says the verification of tentative solutions "has been improved
to better recognize valid solutions. For example, version 5.02 can solve the equation
x+(x+1)^(1/3) = 2 for x."

Rederive: `SOLVE(x+(x+1)^(1/3)=2, x)` stays as written; the square-root analogue
solves.

Suggestion: adopt.

### All roots of a polynomial, numerically

6.1 (SolvingEquationsNumerically): "given a polynomial equation with numeric
coefficients, NSOLVE and NSOLUTIONS return all the real and complex solutions to the
equation. However, for any other type of equation, NSOLVE and NSOLUTIONS return only
one solution." And: "If Derive is in approximate mode and the functions SOLVE and
SOLUTIONS are unable to solve an equation using algebraic methods, they switch to
numerical methods."

Rederive: in Approximate precision `SOLVE(x^5 - x - 1 = 0, x)` stays as written; in
Mixed it gives the real root only. Bounded numeric solving takes a search interval.

Suggestion: in Approximate precision, return all roots of a numeric polynomial that the
algebraic solve leaves standing.

Note: README.md says "numeric root-finding requires a user-supplied search interval".

### Infinite bounds in numeric solving

6.1: the 6.01 changelog says "In the functions NSOLVE and NSOLUTIONS allow plus and/or
minus infinity to be used as valid limits to the solution range." The help gives no
example.

Rederive: allows infinite bounds for polynomials; for other equations it answers `[]`
(`COS(x) = x` on `-inf`, `inf`).

Suggestion: adopt for all equations.

### MOD

6.1 (PiecewiseContinuousFunctions):

- "MOD(m, n) simplifies to m modulo n (i.e. the nonnegative remainder of m/n).
  MOD(m, 0) simplifies to m. Since n defaults to 1, MOD(m) simplifies to the
  fractional-part of m." MODS likewise, with the remainder in [-n/2, n/2).
- "If m, n>0, and p>0 are integers, MOD(m^p, n) is computed efficiently using the
  power-mod algorithm." Same for MODS.
- The 5.05 changelog: "expressions involving FLOOR are converted to equivalent ones
  involving MOD, if the latter is more compact." The help no longer says that a
  nonnumeric MOD becomes FLOOR.

```
VECTOR(MOD(3^(10^n), 10007), n, 1, 10)    [9014, 4995, 6823, 5079, 2749, 8517, 594, 6074, 7480, 493]
```

Rederive: `MOD(2x + 3, 5)` is `-5*FLOOR(2*x/5 + 3/5) + 2*x + 3`; `MOD(m, n)` is
`m - n*FLOOR(m, n)`; `MOD(3^(10^7), 10007)` stays as
`3^10000000 - 10007*FLOOR(3^10000000/10007)`.

Suggestion: keep a nonnumeric MOD as MOD, and compute integer powers modulo n directly.

Contradicts current behavior: Rederive does this on purpose, following the DOS manual
(6.7 p.112: "a nonnumeric argument leaves the floor form behind"), which
tests/test_manual.py asserts as `MOD(m, n)` -> `m - n*FLOOR(m, n)`. Adopting this
suggestion means departing from the DOS manual.

### Financial functions at zero interest

6.1: the 6.01 changelog says "Make the financial functions properly handle the special
case when the interest rate is zero." The help gives no example. Its definition of the
contract, `v(1+i)^n + p(1+i·t)((1+i)^n - 1)/i + f = 0` (FinancialFunctions), has the
limit `v + n·p + f = 0` as i goes to 0.

Rederive: `PMT(0, 10, 1000)`, `PVAL(0, 12, 100)`, `FVAL(0, 12, 100)` and
`NPER(0, -100, 1000)` are `?`.

Suggestion: adopt.

Contradicts current behavior: Rederive does this on purpose, following DOS Derive.
tests/test_functions.py asserts `PMT(0, 10, 1000)` is `?`, and the comment there and in
to_sympy.py says the original "takes no limit to rescue it". Adopting this suggestion
means departing from DOS Derive.

### Logarithm Expand

6.1 (Logarithmfield): `LN(x) + LN(z) ⇔ LN(x·z)` is valid if x is nonnegative;
`k·LN(x) ⇔ LN(x^k)` is valid if k is rational and x is nonnegative or k lies in
(-1, 1]. Collect applies both to the right, Expand to the left, Auto picks the most
compact form.

```
LN(x·y^4 + 2·x·y^2 + x)/2
  Collect   LN(√x·(y^2 + 1))
  Expand    LN(x)/2 + LN(y^2 + 1)
```

Rederive: gives `LN(x*(y^4 + 2*y^2 + 1))/2` in all three modes. Given the factored
argument `LN(x*(y^2 + 1)^2)/2`, Expand gives `LN(x)/2 + LN(y^2 + 1)`.

Suggestion: adopt 6.1's Expand and Collect results for this example.

### Variable order as a setting line

6.1 (OrderingVariablesinExpressions): changing the variable order generates
`VariableOrder := [x1, x2, ..., xn]`, and "The variable ordering can also be changed by
entering on the expression entry line an expression of the above form using upper and
lower case for VariableOrder exactly as shown. [x, y, z] is the factory default."

Rederive: Manage Ordering sets the order but records no line; authoring
`VariableOrder := [y, x, z]` is an ordinary assignment with no effect. The state file
already writes the order as a `VariableOrder := [...]` line (model/state.py).

Suggestion: adopt.

Contradicts current behavior: Rederive does this on purpose. ui/app.py says ordering is
"done to the session rather than expressions derived from it" and appends nothing to
the history, and model/state.py says "nothing authors" the order list. Adopting this
suggestion means reversing that decision.

### ACOT branch

6.1 (InverseTrigonometricFunctions): "ACOT(z) simplifies to π/2 - ATAN(z) ... If x is
real, then ACOT(x) lies in the closed-interval [0, π]." This is not a 6.x change: the
DOS manual (section 6.4, p.108) gives the same identity.

Rederive: uses sympy's odd branch, so `ACOT(-1)` is `-pi/4` and `ACOT(x) + ATAN(x)` is
`pi*SIGN(x)/2`. The two branches agree for positive arguments and differ by π for
negative ones.

Suggestion: use the branch both Derive manuals specify.

Contradicts current behavior: Rederive departs from the manual here on purpose.
tests/test_manual.py records the manual's ACOT identity in NEVER_TO_HOLD, its list of
the manual's promises Rederive has decided not to keep, with the reason "ACOT is sympy's
odd branch; the identity is false on the negative reals". Adopting this suggestion means
reversing that decision.

### Inequality normalization

6.1 (RelationalOperators): "The Simplify commands independently simplify the left and
right sides of top-level equations ... However, the Simplify commands simplify
inequalities and non-top-level equations to a logically equivalent form with constants
on the right and variables on the left."

```
x + 2·x = c + x·x     3·x = x^2 + c
2·x + 3 < 5           x < 1
```

Rederive: simplifies only the sides of a single relation: `2x + 3 < 5` stays
`2*x + 3 < 5`, and so does `2x + 3 < 5 AND y > 0`. A conjunction in one variable is
normalized: `6 >= -2x AND 3x /= -9` is `x > -3`.

Suggestion: adopt.

### SUB with a vector of indices

6.1 (VectorManipulationFunctions): "If the right operand of SUB is a vector of indices, a
vector of the specified elements is returned."

```
[a, b, c, d] SUB [4, 2]                  [d, b]
[a, b, c; d, e, f; g, h, i] SUB [3, 1]   [[g, h, i], [a, b, c]]
```

Rederive: a vector of indices is a path into nested vectors:
`[[2, 3, 5], [7, 1, 4]] SUB [2, 3]` is 4.

Suggestion: adopt.

Contradicts current behavior: Rederive follows the DOS manual, which
tests/test_manual.py asserts as `[[2, 3, 5], [7, 1, 4]] SUB [2, 3]` -> `4`. Adopting this
suggestion means departing from the DOS manual.

### Degree mode

6.1 (AngularUnitfield): the Angular Unit setting selects "the unit (Degree or Radian)
used to express angles when expressions are simplified". Input is read the same way in
both modes: "in either angular unit mode, angles can be entered in radians using the
constant π, or in degrees using the operator °."

```
ASIN(1/2)          π/6 in Radian mode, 30° in Degree mode; approximates to 0.5235987755 in both
SIN(pi/4)          √2/2 in both modes
SIN(45 deg)        √2/2 in both modes
```

Rederive: in Degree mode, a bare number is read as degrees: `SIN(45)` is `SQRT(2)/2`.

Suggestion: adopt.

Contradicts current behavior: Rederive follows the DOS manual, which tests/test_manual.py
asserts as `SIN(45)` -> `SQRT(2)/2` in Degree mode. Adopting this suggestion means
departing from the DOS manual.

### POSITION and SORT

6.1 (VectorManipulationFunctions):

- "If e is an expression that occurs in the vector v, POSITION(e, v) simplifies to the
  position of the first occurrence of e. If e does not occur in v, POSITION(e, v)
  simplifies to false." `POSITION(e, v, n)` finds the first occurrence at or after
  element n.
- "Given a vector or set of numbers, SORT returns a vector of the numbers sorted in
  ascending numerical order of their real parts with ties broken on their imaginary
  parts. Variable names are sorted lexicographically after numbers." Vectors of vectors
  sort by first elements, ties broken by the next. `SORT(v, test, x, y)` sorts by a
  comparison of two variables.

```
POSITION(c, [a, b, c, d, c, e])        3
POSITION(f, [a, b, c, d, c, e])        false
POSITION(c, [a, b, c, d, c, e], 4)     5
SORT([3, 2, y, pi, #e, SQRT(2), z, x, 1, 0, 2, 0, 0, -3])
                                       [-3, 0, 0, 0, 1, √2, 2, 2, ê, 3, π, x, y, z]
SORT([-5, -2, 0, 3, 4], ABS(x) < ABS(y), x, y)
                                       [0, -2, 3, 4, -5]
```

Rederive: has neither; `sort` reads as `s OR t`.

Suggestion: adopt.

### Strings and integers as vectors

6.1 (StringProcessingFunctions): "All the Vector Manipulation Functions treat the
characters of strings and the digits of integers as if they were the elements of a
vector." STRING(u) converts a variable or number to a string; NAME_TO_CODES and
CODES_TO_NAME convert between strings and ASCII codes.

```
"Derive" SUB 3                     "r"
"Derive" SUB [1, 2, 6, 3]          "Deer"
123456789 SUB [1, 3, 5, -3, -1]    13579
DIM("Derive")                      6
REVERSE(1234567890)                987654321
APPEND("a", "b", "c", 3)           "abc3"
DELETE("Derive", 2)                "Drive"
```

Rederive: strings are inert; `"abc" SUB 2`, `DIMENSION("abc")` and `DIMENSION(123)`
stay as written.

Suggestion: adopt.

### DISPLAY, WRITE and LOAD

6.1 (ProgrammingFunctions, UtilityFileA):

- "WRITE(u) format the expression u as a single line of text, and displays the result on
  the Algebra window status line." For showing intermediate results, such as loop
  counters, while a function runs.
- "DISPLAY(u,n) creates a new text box containing the expression u formatted in 2
  dimensions using up to n lines of text. n defaults to 1."
- "Given the full path name of a mth file as a string variable, the LOAD function loads
  the function definitions and variable assignments in the file":
  `LOAD("C:\Derive6\Math\NumberTheoryFunctions.mth")`.

Rederive: has none of them. Utility files load only through Transfer Load, so one file
cannot load another.

Suggestion: adopt.

### SUBST and RANK

6.1:

- SUBST (VariableSubstitution): "SUBST(u, v, w) replaces all occurrences of v in u with
  w, simplifies, and then returns the result. The expression can be followed by as many
  variables and values as desired, in which case the substitutions are made serially.
  However, if the second argument of SUBST is a vector of variables and the third
  argument is a vector of values, the values are substituted for the variables in
  parallel."
- RANK (RowEchelonForm): "RANK(A) simplifies to the rank of matrix A":
  `RANK([2, 3, 5; 4, 6, 10; 1, 2, 3])` is 2.

Rederive:

- SUBST: Manage Substitute substitutes interactively; the notation function `SUBS(u, [x],
  [a])` substitutes in parallel inside an expression.
- RANK: VECTOR.MTH defines it, but only once loaded; otherwise `RANK([[1,2],[2,4]])` is
  a product of the letters r, a, n, k.

Suggestion: adopt.

### GROEBNER_BASIS

6.1 (GrobnerBasis): "GROEBNER_BASIS(polys, vars) simplifies to a vector (set) of
polynomials that form the Gröbner basis for polys based on the lexicographic ordering
of vars."

```
GROEBNER_BASIS([x^2+y^2+z^2-2, x^2+y^2-z, x^2+y^2-1], [x, y, z])    [z - 1, x^2 + y^2 - 1]
GROEBNER_BASIS([x^2+y^2-z+1, x^2+y^2-z+3], [x, y, z])               [1]
```

Rederive: has no such function.

Suggestion: adopt.

### Sets

6.1 (SetOperatorsandFunctions): sets are written in braces, `{1, 2, 3}`, with ellipses
for ranges (`{1, 3, ..., 10}` is `{1, 3, 5, 7, 9}`). Operators: complement ``s` `` (postfix
backquote, used only for algebraic simplification), intersection `∩` (INTERSECTION),
union `∪` (UNION), difference `\`. Functions: `MEMBER?(u, v)`, `POWER_SET(s)`,
`POWER_SET(s, n)`. A set of pairs acts as a function through SUB
(`{[2, 3], [3, 5], [1, 7]} SUB 2` is 3), and the product of two sets is their Cartesian
product. 6.1 also uses the backquote for matrix transpose (Operators); the help does not
say how a symbol's complement is told from its transpose.

```
{1,3,5,7,9} INTERSECTION {2,3,5,8,13}   {3,5}
{1,...,10} \ {2,3,5,7}                  {1,4,6,8,9,10}
POWER_SET({a, b, c}, 2)                 {{b, c}, {a, c}, {a, b}}
{a, b}*{1, 2}                           {[a, 1], [a, 2], [b, 1], [b, 2]}
```

Rederive: has no sets; `{1,2}` is a syntax error. The backquote is transpose.

Suggestion: adopt.

### ROW and COL

6.1 (VectorOperations): "If A is a matrix, A ROW n and A COL n return the nth row and
column of A as a vector, respectively. ... If the right operand of ROW or COL is a
vector of indices, a matrix of the specified rows or columns is returned."

```
[a, b, c; d, e, f; g, h, i] ROW 2     [d, e, f]
[a, b, c; d, e, f; g, h, i] COL 2     [b, e, h]
```

Rederive: rows through SUB, columns through transpose:
``[[a,b],[c,d]]` SUB 2`` is `[b, d]`. Keyword operators are found inside runs of letters
in Character mode (`sort` reads as `s OR t`), so ROW and COL might split runs such as
`grow` or `cold`.

Suggestion: adopt.

### Display Step

6.1 (DisplayStep, DisplayStepscheckbox): Simplify > Display Step performs the next step
of simplifying the highlighted expression and shows the result as a new line, optionally
with the rule applied ("Display Rules"). `DisplaySteps := true` makes every Simplify do
this. The help warns that some steps may not be shown and that factoring algorithms
have no intermediate steps.

Rederive: has no stepwise simplification.

Suggestion: adopt. Hard to implement: Rederive's simplifier works through sympy, which
has no notion of single steps.

### Autoloading utility files

6.1 (UtilityLibraryFiles): "When a function in any of the utility files listed below is
first used, all the definitions and assignments in the file are automatically loaded.
Then the function is evaluated and returns a value, just like it normally would."

Rederive: a utility file must be loaded with Transfer Load before its functions work;
until then a call such as `RANK(...)` is read as a product of letters.

Suggestion: adopt for the utility files Rederive supports.

## Not suggested

### Type predicates (INTEGER?, NUMBER?, MEMBER?, ...)
- 6.1 (ExpressionTypeFunctions): true or false according to the type of the argument.
- Reason: tests on the internal representation, plumbing for writing programs rather than
  mathematics. `NUMBER(u)` already covers the common case.

### Procedural programming: PROG, LOOP, RETURN, EXIT, ASSIGN
- 6.1 (ProceduralProgramming, ProgrammingFunctions): PROG evaluates statements in order
  and returns the last; LOOP repeats them until EXIT or RETURN; RETURN ends the function
  with a value; ASSIGN and the update operators `:+ :- :* :/` assign to variables.
- Reason: Lisp machinery (PROG and RETURN are Lisp's) exposed to users, not mathematics.

### FIRST, REST, ADJOIN
- 6.1 (VectorManipulationFunctions): first element, all but the first, and a new first
  element; the help says they process vectors "analogously to the way lists are
  processed in the LISP programming language".
- Reason: Lisp's CAR, CDR and CONS under other names. `ELEMENT(v, 1)`,
  `DELETE_ELEMENT(v, 1)` and `APPEND([u], v)` do the same.

### EVERY, SOME
- 6.1 (BooleanFunctions): whether a test holds for every, or for some, element of a
  vector or range, stopping at the first counterexample or example.
- Reason: Lisp's EVERY and SOME. Expressible through SUM of IF or SELECT.

### `'` as a quote operator
- 6.1 (ProceduralProgramming): `'(2+3)` simplifies to `2+3`.
- Reason: `'` already marks derivatives (`F'(x)`).

### `[a, b; c, d]` matrix rows
- 6.1 (Matrix): `;` separates the rows of a matrix.
- Reason: `;` starts a comment in .mth files, so the same text would mean different
  things on the Author line and in a file.

### MAP_LIST, SOLUTIONS, TABLE
- 6.1: builtins.
- Reason: each has an exact equivalent: `MAP_LIST(u, x, c)` is `VECTOR(u, x, c)`,
  `SOLUTIONS(...)` is `RHS(SOLVE(...))`, and `TABLE(u, k, ...)` is
  `VECTOR([k, u], k, ...)`.
