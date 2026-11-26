# Programming

A program is a function you define. The examples show what Simplify gives.

## Defining functions and variables

```
a := 3                  a^2 + 1 is now 10
a :=                    a has no value again
F(x) := x^2 + 1         F(3) is 10, F(t + 1) is t^2 + 2t + 2
G(x, y := 1) := x + y   y is 1 when left out: G(5) is 6
CNT v := DIM(v)         v takes all the arguments of a call,
                        as a vector: CNT(a, b, c) is 3
v SUB 2 := 7            replace element 2 of the vector v
k :+ 1                  k := k + 1; likewise :-, :* and :/
```

Only parameters are local; every other variable is global. For a local, add a parameter
that calls leave out.

Assigning to a [setting](help:settings/current-settings) changes it:
`Precision := Approximate`.

## Conditions and loops

A function's body is one expression. [IF](name:IF) chooses between values, and a function
can call itself:

```program
FIB(n) := IF(n < 2, n, FIB(n - 1) + FIB(n - 2))
```

```
FIB(10)               55
```

The test of an IF is a relation such as `n < 2`, relations joined by AND, OR and NOT, or
one of the [tests](help:functions/logic-and-tests) whose names end in a question mark.

PROG runs statements in order, and LOOP repeats them:

```program
ROOTS(a, b, c, d) :=
  PROG(
    d := b^2 - 4 a c,
    [(-b + SQRT(d))/(2 a), (-b - SQRT(d))/(2 a)])
```

```
ROOTS(1, -3, 2)       [2, 1]
```

```program
COLLATZ(n, k := 0) :=
  LOOP(
    IF(n = 1, RETURN k),
    n := IF(EVEN?(n), n/2, 3n + 1),
    k :+ 1)
```

```
COLLATZ(27)           111
```

The parameters d and k serve as locals.

| Statement | Meaning |
|---|---|
| `PROG(s, t, ...)` | run the statements in order; the value is the last one's |
| `LOOP(s, t, ...)` | run them repeatedly, until exit or RETURN |
| `MAP(u, k, ...)` | run u at each value of k; the value is true |
| `RETURN u` | end the function being called, with the value u |
| `exit` | leave the innermost LOOP or PROG, with the value of the statement that ran last |
| `ASSIGN(v, u)` | assign u to the variable that v holds: after `v := z`, `ASSIGN(v, 7)` sets z |
| `DISPLAY(u)` | add u to the worksheet as a line of text |
| `WRITE(u)` | show u on the message line while the program runs |
| `LOAD("file")` | read a file of definitions, as Transfer Load Utility does |

## Repeating over values

```examples
SQUARES(n) := VECTOR(k^2, k, 1, n)
                          SQUARES(5) is [1, 4, 9, 16, 25]
PRIMES(n) := SELECT(PRIME?(k), k, 1, n)
                          PRIMES(12) is [2, 3, 5, 7, 11]
SUM(k^2, k, 1, 4)         30
ITERATES(x/2, x, 16, 4)   [16, 8, 4, 2, 1]
EVERY(PRIME?(k), k, [2, 3, 5])
                          true
```

The values of k are given as for [VECTOR](name:VECTOR). [MAP](name:MAP) runs a statement
at each value.

## Examples

A polynomial reduced by a relation, applied until the polynomial stops changing.

```program
RED(u, v) :=
  ITERATE(RHS(v) QUOTIENT(p, LHS(v)) + REMAINDER(p, LHS(v)),
    p, u)
```

```
RED(x^10, x^2 = x + 1)      55x + 34
```

By Johann Wiesenbauer, Derive Newsletter 20.

The derivative of f with respect to a subexpression a, by way of a variable s put in its
place.

```program
NEWDER(f, a, s) := SUBST(DIF(SUBST(f, a, s), s), s, a)
```

```
NEWDER(SIN(x)^3 + 2 SIN(x), SIN(x), s)
                            3 SIN(x)^2 + 2
```

By Francisco Marcelo Fernández, Derive Newsletter 113.

The Wronskian of the functions in v, as the determinant of their derivatives.

```program
WRONSKIAN(v, x) :=
  DET(VECTOR(DIF(v, x, k), k, 0, DIM(v) - 1))
```

```
WRONSKIAN([SIN(x), COS(x)], x)      -1
```

A row of Pascal's triangle, each made by adding the row before to itself shifted by one.

```program
PASCAL_ROW(n) :=
  ITERATE(ADJOIN(0, v) + APPEND(v, [0]), v, [1], n)
```

```
PASCAL_ROW(6)         [1, 6, 15, 20, 15, 6, 1]
```

Adapted from Johann Wiesenbauer, Derive Newsletter 47.

The steps of Euclid's algorithm, as equations. The quote keeps each right side as written.

```program
EUCLID_STEPS(a, b, q, r, s := []) :=
  LOOP(
    IF(b = 0, RETURN s),
    q := FLOOR(a, b),
    r := a - q b,
    s := APPEND(s, ['(a = q b + r)]),
    [a := b, b := r])
```

```
EUCLID_STEPS(1071, 462)
    [1071 = 2*462 + 147, 462 = 3*147 + 21, 147 = 7*21 + 0]
```

Adapted from Johann Wiesenbauer, Derive Newsletter 37.
