# Conformance with the Derive 3.14 manual

What the printed manual promises, asked of rederive as tests, and what came back.

Source: `artifacts/manuals/derive-3.14-manual-transcript.md`.
Tests: `tests/test_manual.py`. Run with `uv run pytest tests/test_manual.py`.

## What was collected

Every chapter of the transcript was read for *actionable sessions* - anything a user can
do that has an observable, checkable outcome: type this expression and get that one,
press this key and see that happen, call this function of these arguments and get this
result. Each collected item records where it is in the transcript, what to type, what
the manual says appears, whether it is engine- or UI-level, and what a test author needs
to know (preconditions, settings it depends on, ambiguities, and - often - that the
manual states no outcome at all).

| Chapter | Items |
| --- | --- |
| 1. Introduction | 16 |
| 2. Fundamentals | 101 |
| 3. Arithmetic | 92 |
| 4. Algebra | 208 |
| 6. Functions and Constants | 173 |
| 7. Calculus | 58 |
| 8. Vectors and Matrices | 108 |
| 9. Utility Files | 320 |
| 10. Programming | 71 |
| Appendices | 163 |
| **Total** | **1310** |

Excluded as out of scope, since the project does not do plotting: all of chapter 5, the
2D and 3D plot Q&A, the plot window command and function key lists, section 9.19
GRAPHICS.MTH, and the plotting halves of exercises elsewhere (the computational halves
were kept). Also excluded: DOS hardware and printer trivia that has no software meaning
here.

## What was tested

`tests/test_manual.py` holds **394 cases** drawn from those sessions - the engine-level
ones with a stated outcome, plus the session-level behaviour the manual is explicit
about. The expected value is what the manual prints, translated only out of its typeset
form into author text: `·` becomes `*`, `√` becomes `SQRT`, raised exponents become `^`.
Where the manual states no outcome, there is no case.

Chapter 9 loads the original utility files from `artifacts/sessions/derive-3.14-dos/`, so
those cases exercise the real `.MTH` sources rather than a transcription of them.

## Result

**307 pass, 87 do not**, in about 24 seconds. It was 258 and 136 before the work below.

The 87 are marked `xfail` through `NOT_YET_HELD`, a manifest of test ids at the top of
`tests/test_manual.py`. They run rather than being skipped, so one that starts holding is
reported as an unexpected pass and can be struck off the list. The suite is therefore
green, and the manifest is the list of what the manual promises and the engine does not
do yet.

Three mistakes were in the tests rather than the engine, and are worth naming so the
number is not read as worse than it is. Two were found before counting: the
approximate-mode contexts did not let the precision carry the notation with it (section
3.9 says Options Precision moves it), and plus-or-minus was spelled `+-` where both the
manual and the program write `±`. The third was caught against the original - the `XOR`
case expected the fenced form section 4.16 prints, where Derive answers
`NOT p AND q OR p AND NOT q`, relying on AND binding tighter than OR.

## Two of the manual's own sessions do not terminate

`ITERATES(#e^(-x/20), x, 1)` approximated (10.1) and `INT_SUBST(t·SIN(t²), t, t²)` from
MISC.MTH (9.21) never come back. Left alone they occupied two test workers for 36 CPU
minutes each. `SIGALRM` does not break either one, so the loop is inside a C-level call
rather than in interruptible Python.

Both are now asked through `RemoteEngine` with a 15 second bound, which is the half of
the program Esc can stop, so they fail saying "no answer in 15 seconds" rather than
wedging the suite. They are counted among the 136.

## What the failures are

The split below is an eyeball reading of the assertion diffs, not a mechanical
classification. Roughly 40 are the same answer written differently and roughly 115 are
functional gaps.

### The same value, written differently

Sums are now written in the order list's order, so what remains here is narrower than it
was.

- Factor order in a *product*: `(x - 2)*(x + 2)` for the manual's `(x + 2)*(x - 2)`. Four
  cases, all differences of squares, and `(x + 2)*(x - 2)*(x + 1)^2` shows the exponent
  interacts with whatever the rule is. Too little evidence to write one.
- `ABS(x)` where the manual writes `|x|`. Left alone deliberately: `ABS(ABS(x) - y)` would
  print `|y - |x||`, which the bars cannot be read back out of.
- `-1/COS(x)^2` for `-TAN(x)^2 - 1`.
- The argument of a logarithm: `LN(z + SQRT(z^2 + 1))` for `LN(SQRT(z^2 + 1) + z)`.
  Ordering that one wants a compound factor to lend its variables outward, and the
  expansion `(x + 2*y + 1)^3` the manual states wants the opposite. Both cannot hold.
- Results computed correctly but grouped differently: BERNOULLI_POLY and EULER_POLY come
  back expanded where the manual gathers over a common denominator, and `NORMAL` gives
  `1/2 - ERF(...)/2` where the manual writes `(ERF(...) + 1)/2`.

### Transformations the manual states, not performed

Most of this list has been written; `ASEC`, `ACSC`, the two-argument `ACOT`, `ACOSH`,
`ACOTH`, `PERM`, `COMB`, `ERFC`, the two-argument `ERF`, `STEP`, `MIN` and `MAX` all hold
now, and `MOD` and `MODS` turned out to be the `FLOOR(m, n)` spelling rather than a
missing rule. `CHI` was already right. What is left is one rule and one caveat.

**`ACOT(z) -> pi/2 - ATAN(z)` is not done, and should not be.** This engine's `ACOT` is
sympy's odd branch, where `ACOT(-1)` is `-pi/4`; the manual's identity belongs to the
other branch and is false on the negative reals here. `pipeline.py` already documents
that divergence as deliberate, with tests asserting `ATAN(t) + ACOT(t)` is `pi*SIGN(t)/2`
rather than the manual's `pi/2`.

**The five arc rules of 6.4 are a table of equivalences, not a rewrite system.** Applying
`ACOS -> pi/2 - ASIN` and `ASEC -> ACOS(1/z)` together leaves `ASEC(z)` as
`pi/2 - ASIN(1/z)`, which is neither form the section prints, and the first answer would
re-simplify into something else on a second Simplify. The engine keeps `ASIN`, `ACOS` and
`ATAN` and writes the reciprocal arcs over them, which is what the manual's own printed
answer `ACOS(1/z)` shows Derive doing.

### Features that were absent

All five have been built. Where a case still fails, the reason is named.

- `FIT`, the least squares fit of section 6.10. Two of the manual's three examples hold
  exactly, the plane down to its last printed digit. The third computes the manual's
  coefficients and prints them in a different term order.
- The antidifference of `SUM` and the antiquotient of `PRODUCT`, both without limits, and
  with them the `GAMMA(z)` rule `PRODUCT(n^2, n)` was waiting on. Only hypergeometric
  summands are reachable, so the engine refuses where Derive would answer a general
  telescoping sum. `SUM(n^2, n)` comes back unfactored, which is the program's own form.
- All nine of the nonscalar algebra rules of section 8.8. A name declared nonscalar is a
  matrix now, so eight of the nine hold by construction, and the two distributive rules
  are applied whether or not they shorten anything. A scalar's transpose is the scalar,
  which section 8.5 says and nothing implemented.
- Boolean simplification, all four cases, in the program's spelling not the page's.
- Recursive user functions unfold: Simplify runs its pass again while a user call is still
  standing, bounded by the size guard `ITERATE` already used. `FACT(64)` arrives in well
  under a second. Where the bound is reached, what has been worked out so far stands as
  the answer, which is not what Derive does - it exhausts memory and returns nothing.

Two things inside the last of those were left deliberately:

- **The arms of an undecidable conditional are still simplified**, where Derive leaves
  them exactly as written. That needs unevaluated conversion, authored-order printing,
  and keeping the canonical rebuild away from a frozen conditional - three coupled changes
  to the printing path.
- **A derivative of a call not yet unfolded is taken too early.** `DIF(F(n - 1), mu)` sees
  an opaque head carrying no free `mu` and differentiates it to zero before a later pass
  can unfold it. An outside-in loop is structurally wrong for that shape, and deferring
  the calculus step does not terminate on its own, since the pre-pass unfolds
  unconditionally and a call is present at every pass in the branch about to be discarded.

### The utility files, again

The claim that most utility file functions follow from the recursion gap, "since they are
built on `ITERATE`, `ITERATES` and `IF`", did not hold. Of the thirteen named, one was a
recursion problem. NTH_PRIME answered correctly all along - `NTH_PRIME(1000)` is 7919 -
and PICARD differed from the manual only in term order.

Four of the real causes are fixed. Inert heads were re-read only once, so a nest resolved
one layer per pass; `Logical` was never re-read at all, which is why a test of
`PRIME(n) AND PRIME(n + 2)` froze where the same search without the `AND` succeeded;
frozen conditionals leaked as `IF0(1) + IF0(2)` wherever a finite sum expanded around
them; and `ITERATE` did not survive the spelling that holds a vector in one variable and
reads it back by subscript.

What stops the rest has nothing to do with any of that: `LIM` over a vector of variables
is unimplemented, which is what stops TAYLOR_SOLVE, TAYLOR_ODE1, TAYLOR_ODE2 and EULER;
`GRAD` misbinds its arguments, stopping JACOBIAN; and `FLOOR` does not map over a vector,
stopping CONTINUED_FRACTION.

### Three cases worth naming individually

These three were previously grouped under a heading calling them "documented
divergences". That was wrong: nothing had documented them, and the three are not alike.
Checked against the repository, they stand as follows.

**The initial Exponential direction: the manual is wrong, and the repository
already knew it.** `src/rederive/model/settings.py:377-381` records the
research: section 6.1 says the field starts on Collect where the identically
worded section 6.2 says Auto of Logarithm; Collect was the 1.x default, its
DERIVE.INI carries `*EXP-EXPD* |Collect|`, and the original's own DERIVE.INI
says `*EXP-EXPD* |Auto|`, as does its screen. So `Auto` is right and section
6.1 is a leftover. The test asserted the erratum; it now asserts `Auto` and
carries that reasoning, which is why 157 became 156. It is the only place in
the suite where the manual is treated as wrong rather than as unmet.

**`(-8)^(1/3)`: a printing difference, with the current form already pinned elsewhere.**
rederive answers `2*(-1)^(1/3)` where section 4.5 prints `1 + √3·î`. These are the same
number. `tests/test_simplify.py:510` already pins `2*(-1)^(1/3)` as the principal-branch
answer, and `src/rederive/engine/pipeline.py:1636` notes in passing that sympy hands back
that form. Neither takes a position on section 4.5's statement that complex results are
put in rectangular form, so what is untested is the printing promise, not the value.

**`∫(-1 to 2) 1/x³ dx`: a gap with no prior record.** The manual documents `3/8`, the
Cauchy principal value, and is explicit that this is what it means by it. rederive
answers `?`. Nothing in `src/` or `tests/` mentions Cauchy principal values at all. This
test is the first record of the difference.

## What the original does

Some of the above is about what Derive *prints*, which a typeset page cannot settle.
Those points follow the original's own output, which agrees with the manual
except where noted.

- **Operand order.** Authoring preserves what was typed; Simplify sorts. Variables order
  `x`, `y`, `z` first and the rest alphabetically, which is the `*VARIABLE-ORDER*`
  default; a term sorts on its leading variable, not a later one; `NOT` does not move a
  term, but a negated literal precedes a positive one on a tie. Sums are written this way
  now.
- **The author line is redrawn, not canonicalised.** `(NOT p) OR (q AND r)` is put up as
  `NOT p OR q AND r` and `x+y` as `x + y`, but `q OR p` stays `q OR p`: Derive
  pretty-prints the line it parsed without reordering it. An entry here now holds what it
  parsed rather than what was typed. Note that the *screen* was never the problem - the
  layout has always been drawn from the tree - it was the textual record behind Ctrl-C,
  F3, splicing and Transfer Save that echoed the keystrokes.
- **Nonscalars.** `a . a^-1` does not simplify at all, so no identity matrix of unknown
  dimension ever arises. An undeclared variable's transpose collapses at the first
  backquote: `x`` is `x`. The original right-nests `((a . b) . c) . d` to
  `a . (b . (c . d))` and prints the parentheses.
- **Closed forms.** `SUM(n^2, n)` is printed unfactored as `n^3/3 - n^2/2 + n/6`, not
  gathered over a denominator, which is the form the engine now reaches too.
  `PRODUCT(2, n)` is `2^n`, which does not agree with the base point
  `PRODUCT(n^2, n) -> (n - 1)!^2` implies; the engine gives `2^(n - 1)` and nothing tests
  it, so the program is inconsistent here and was not followed.
- **`IF(0, a, b)` is `a`** and `IF(5, a, b)` is `b`, section 10.3's rule that a
  non-relational test is read as `test = 0`. The engine answered `b` for the first and
  now does what the program does.
- **The arms of an undecidable conditional are untouched.** `IF(x > 0, 2 + 3, 4 + 5)`
  comes back with the arithmetic undone, not as `IF(x > 0, 5, 9)`.
- Confirmed in passing: `GAMMA(z)` is `(z - 1)!`, `SUM(IF(PRIME(n)), n, 1, 100)` is `25`,
  and `FACT(-1)` says `Memory Full` and adds no entry at all, leaving no partial result,
  as section 4.2 describes.
- **Function terms do not sort alphabetically.** The observed chain is
  `ABS < ASIN < ATAN < LN < COS < TAN < SIN`, an internal ordinal rather than a rule that
  can be read off a name, and the variable outranks the head: all `t` terms precede all
  `u` terms whatever the function. Those seven are placed; every other head sorts behind
  them alphabetically, which is an assumption and marked as one in the code.

### What is still worth asking the program

One case contradicts the readings above rather than merely going unmet.
COVARIANT_METRIC_TENSOR is printed `[[w^2 + v^2, ...]]` in section 9, but `w + a -> a + w`
shows `w` is off the order list and `b + a -> a + b` shows off-list names go
alphabetically, which together give `v^2 + w^2`. One of the three readings is not the rule
it looks like, and the tensor case was left failing rather than fitted with a rule made up
for it. Worth settling against the original.

Two smaller ones, both untested here: whether `(a . b)/2` is printed that way or as
`(1/2)*(a . b)`, and what governs factor order in a product, where the four cases in the
suite are all differences of squares and settle nothing on their own.

## Collected sessions

The per-chapter session inventories the tests were drawn from are working notes rather
than a deliverable; they live outside the repository, one file per chapter, at
`/tmp/claude-1000/-home-hniksic-work-rederive/de715675-be66-40c1-8f6b-54f3c8a639b0/scratchpad/sessions/`.
Say the word if they are worth keeping and they can be moved in.
