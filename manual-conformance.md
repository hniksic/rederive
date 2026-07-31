# Conformance with the Derive 3.14 manual

What the printed manual promises, asked of rederive as tests, and what came back.

Source: `artifacts/manuals/derive-3.14-manual-transcript.md`.
Tests: `tests/test_manual.py`. Run with `uv run pytest tests/test_manual.py`.

## What was collected

Every chapter of the transcript was read for *actionable sessions* - anything a user can
do that has an observable, checkable outcome: type this expression and get that one,
press this key and see that happen. Each collected item records where it is in the
transcript, what to type, what the manual says appears, whether it is engine- or
UI-level, and what a test author needs to know (preconditions, settings it depends on,
ambiguities, and - often - that the manual states no outcome at all).

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
were kept). Also excluded: DOS hardware and printer trivia with no software meaning here.

## What was tested

`tests/test_manual.py` holds **394 cases** drawn from those sessions - the engine-level
ones with a stated outcome, plus the session-level behaviour the manual is explicit
about. The expected value is what the manual prints, translated only out of its typeset
form into author text: `·` becomes `*`, `√` becomes `SQRT`, raised exponents become `^`.
Where the manual states no outcome, there is no case.

Chapter 9 loads the original utility files from `artifacts/sessions/derive-3.14-dos/`, so
those cases exercise the real `.MTH` sources rather than a transcription of them.

## Result

**307 pass, 87 do not**, in about 20 seconds. The 87 are marked `xfail` through
`NOT_YET_HELD`, a manifest of test ids at the top of `tests/test_manual.py`. They run
rather than being skipped, so one that starts holding is reported as an unexpected pass
and can be struck off the list. The suite is therefore green, and the manifest is the
list of what the manual promises and the engine does not do yet.

Two of the manual's own sessions never terminate: `ITERATES(#e^(-x/20), x, 1)`
approximated (10.1) and `INT_SUBST(t·SIN(t²), t, t²)` from MISC.MTH (9.21). The loop is
inside a C-level call that `SIGALRM` cannot break, so both are asked through
`RemoteEngine` with a 15 second bound - the half of the program Esc can stop - and fail
saying "no answer in 15 seconds" rather than wedging the suite.

## Why the 87 fail

Grouped by what stands in the way, not by chapter.

### Divergences that are deliberate

Nothing to implement here; these are decisions, recorded so they are not re-litigated.

- **`ACOT(z) -> pi/2 - ATAN(z)`.** This engine's `ACOT` is sympy's odd branch, where
  `ACOT(-1)` is `-pi/4`; the manual's identity belongs to the other branch and is false
  on the negative reals here. `pipeline.py` documents the divergence, with tests
  asserting `ATAN(t) + ACOT(t)` is `pi*SIGN(t)/2` rather than the manual's `pi/2`.
- **The five arc rules of 6.4 are a table of equivalences, not a rewrite system.**
  Applying `ACOS -> pi/2 - ASIN` and `ASEC -> ACOS(1/z)` together leaves `ASEC(z)` as
  `pi/2 - ASIN(1/z)`, which is neither form the section prints, and the result would
  re-simplify into something else on a second Simplify. The engine keeps `ASIN`, `ACOS`
  and `ATAN` and writes the reciprocal arcs over them - which is what the manual's own
  printed answer `ACOS(1/z)` shows Derive doing.
- **`ABS(x)` where the manual writes `|x|`.** `ABS(ABS(x) - y)` would print
  `|y - |x||`, which the bars cannot be read back out of.
- **A recursion that exceeds the size guard returns what was worked out so far.**
  Derive instead exhausts memory and returns nothing.

### The same value, written differently

- Factor order in a product: `(x - 2)*(x + 2)` for the manual's `(x + 2)*(x - 2)`. Four
  cases, all differences of squares, and `(x + 2)*(x - 2)*(x + 1)^2` shows the exponent
  interacts with whatever the rule is. Too little evidence to write one; see the open
  questions below.
- The argument of a logarithm: `LN(z + SQRT(z^2 + 1))` for `LN(SQRT(z^2 + 1) + z)`.
  Ordering that one wants a compound factor to lend its variables outward, and the
  expansion `(x + 2*y + 1)^3` the manual states wants the opposite. Both cannot hold.
- `-1/COS(x)^2` for `-TAN(x)^2 - 1`.
- Grouping: BERNOULLI_POLY and EULER_POLY come back expanded where the manual gathers
  over a common denominator; `NORMAL` gives `1/2 - ERF(...)/2` where the manual writes
  `(ERF(...) + 1)/2`; the third `FIT` example computes the manual's coefficients and
  prints them in a different term order (the other two hold exactly).
- `(-8)^(1/3)`: rederive answers `2*(-1)^(1/3)` where section 4.5 prints `1 + √3·î`.
  The same number, and `tests/test_simplify.py:510` pins the current form as the
  principal-branch answer. What is unmet is section 4.5's promise that complex results
  are put in rectangular form - a printing gap, not a value gap.

### Tractable gaps

Each of these is one misbehaving primitive with named casualties in chapter 9:

- `LIM` over a vector of variables is unimplemented - stops TAYLOR_SOLVE, TAYLOR_ODE1,
  TAYLOR_ODE2 and EULER.
- `GRAD` misbinds its arguments - stops JACOBIAN.
- `FLOOR` does not map over a vector - stops CONTINUED_FRACTION.

And one narrower limit: the antidifference of `SUM` (and antiquotient of `PRODUCT`)
reaches only hypergeometric summands, so the engine refuses where Derive would answer a
general telescoping sum.

### Structural problems

- **The arms of an undecidable conditional are simplified**, where Derive leaves them
  exactly as written. Fixing it needs unevaluated conversion, authored-order printing,
  and keeping the canonical rebuild away from a frozen conditional - three coupled
  changes to the printing path.
- **A derivative of a call not yet unfolded is taken too early.** `DIF(F(n - 1), mu)`
  sees an opaque head carrying no free `mu` and differentiates it to zero before a later
  pass can unfold it. An outside-in loop is structurally wrong for that shape, and
  deferring the calculus step does not terminate on its own, since the pre-pass unfolds
  unconditionally and a call is present at every pass in the branch about to be
  discarded.

### No design yet

- **Cauchy principal value.** The manual documents `∫(-1 to 2) 1/x³ dx` as `3/8`, and is
  explicit that the principal value is what it means. rederive answers `?`; nothing in
  `src/` or `tests/` mentions principal values at all.
- The two non-terminating sessions named under Result.

## Where the manual is wrong

The initial Exponential setting is the one place the suite treats the manual
as wrong rather than unmet. Section 6.1 says the field starts on Collect where
the identically worded section 6.2 says Auto of Logarithm.
`src/rederive/model/settings.py:377-381` records the research: Collect was the
1.x default, its DERIVE.INI carries `*EXP-EXPD* |Collect|`, and the original's
own DERIVE.INI and screen say `Auto`. So `Auto` is right, section 6.1 is a
leftover, and the test asserts `Auto`.

## What the original does

Some of the above is about what Derive *prints*, which a typeset page cannot settle.
Those points follow the original's own output, which agrees with the manual
except where noted.

- **Operand order.** Authoring preserves what was typed; Simplify sorts. Variables order
  `x`, `y`, `z` first and the rest alphabetically, which is the `*VARIABLE-ORDER*`
  default; a term sorts on its leading variable, not a later one; `NOT` does not move a
  term, but a negated literal precedes a positive one on a tie.
- **The author line is redrawn, not canonicalised.** `(NOT p) OR (q AND r)` is put up as
  `NOT p OR q AND r` and `x+y` as `x + y`, but `q OR p` stays `q OR p`: Derive
  pretty-prints the line it parsed without reordering it.
- **Nonscalars.** `a . a^-1` does not simplify at all, so no identity matrix of unknown
  dimension ever arises. An undeclared variable's transpose collapses at the first
  backquote: `x`` is `x`. The original right-nests `((a . b) . c) . d` to
  `a . (b . (c . d))` and prints the parentheses.
- **Closed forms.** `SUM(n^2, n)` is printed unfactored as `n^3/3 - n^2/2 + n/6`, not
  gathered over a denominator. `PRODUCT(2, n)` is `2^n`, which does not agree with the
  base point `PRODUCT(n^2, n) -> (n - 1)!^2` implies; the engine gives `2^(n - 1)` and
  nothing tests it - the program is inconsistent here and was not followed.
- **`IF(0, a, b)` is `a`** and `IF(5, a, b)` is `b`, section 10.3's rule that a
  non-relational test is read as `test = 0`.
- **The arms of an undecidable conditional are untouched.** `IF(x > 0, 2 + 3, 4 + 5)`
  comes back with the arithmetic undone, not as `IF(x > 0, 5, 9)`.
- **Function terms do not sort alphabetically.** The observed chain is
  `ABS < ASIN < ATAN < LN < COS < TAN < SIN`, an internal ordinal rather than a rule that
  can be read off a name, and the variable outranks the head: all `t` terms precede all
  `u` terms whatever the function. Those seven are placed; every other head sorts behind
  them alphabetically, which is an assumption and marked as one in the code.
- Confirmed in passing: `GAMMA(z)` is `(z - 1)!`, `SUM(IF(PRIME(n)), n, 1, 100)` is
  `25`, and `FACT(-1)` says `Memory Full` and adds no entry at all, leaving no partial
  result, as section 4.2 describes.

## Still to settle against the original

- COVARIANT_METRIC_TENSOR is printed `[[w^2 + v^2, ...]]` in section 9, but
  `w + a -> a + w` shows `w` is off the order list and `b + a -> a + b` shows off-list
  names go alphabetically, which together give `v^2 + w^2`. One of the three readings is
  not the rule it looks like; the tensor case was left failing rather than fitted with a
  rule made up for it.
- What governs factor order in a product - the four cases in the suite are all
  differences of squares and settle nothing on their own.
- Whether `(a . b)/2` is printed that way or as `(1/2)*(a . b)`.

## Collected sessions

The per-chapter session inventories the tests were drawn from are working notes rather
than a deliverable; they live outside the repository, one file per chapter, at
`/tmp/claude-1000/-home-hniksic-work-rederive/de715675-be66-40c1-8f6b-54f3c8a639b0/scratchpad/sessions/`.
Say the word if they are worth keeping and they can be moved in.
