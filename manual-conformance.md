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

**254 pass, 140 do not**, in about 17 seconds.

The 140 are marked `xfail` through `NOT_YET_HELD`, a manifest of test ids at the top of
`tests/test_manual.py`. They run rather than being skipped, so one that starts holding is
reported as an unexpected pass and can be struck off the list. The suite is therefore
green, and the manifest is the list of what the manual promises and the engine does not
do yet.

Two mistakes in the tests themselves were found and fixed before counting, and are worth
naming so the number is not read as worse than it is: the approximate-mode contexts were
built without letting the precision carry the notation with it (section 3.9 says Options
Precision moves it, and `Context.with_precision` implements that), and plus-or-minus was
spelled `+-` where both the manual and the program write `±`.

## Two of the manual's own sessions do not terminate

`ITERATES(#e^(-x/20), x, 1)` approximated (10.1) and `INT_SUBST(t·SIN(t²), t, t²)` from
MISC.MTH (9.21) never come back. Left alone they occupied two test workers for 36 CPU
minutes each. `SIGALRM` does not break either one, so the loop is inside a C-level call
rather than in interruptible Python.

Both are now asked through `RemoteEngine` with a 15 second bound, which is the half of
the program Esc can stop, so they fail saying "no answer in 15 seconds" rather than
wedging the suite. They are counted among the 140.

## What the failures are

The split below is an eyeball reading of the assertion diffs, not a mechanical
classification. Roughly 40 are the same answer written differently and roughly 115 are
functional gaps.

### The same value, written differently

- Factor order: `(x - 2)*(x + 2)` for the manual's `(x + 2)*(x - 2)`, and similarly for
  every other multi-factor result.
- Argument order under commutative operators: `#e^(w + z)` for `#e^(z + w)`,
  `#e^w*#e^z` for `#e^z*#e^w`.
- `ABS(x)` where the manual writes `|x|`; `FLOOR(m/n)` where it writes `FLOOR(m, n)`.
- `-1/COS(x)^2` for `-TAN(x)^2 - 1`; `1 - 1/2^m` for `1 - 2^(-m)`;
  `LN(z + SQRT(z^2 + 1))` for `LN(SQRT(z^2 + 1) + z)`.
- Results that are computed correctly but grouped differently, e.g. BERNOULLI_POLY and
  EULER_POLY come back expanded where the manual prints them over a common denominator.

### Transformations the manual states, not performed

The expression comes back unchanged.

- `ACOSH` and `ACOTH`, which do become logarithms but not the ones section 6.6 prints:
  `LN(SQRT(z - 1)*SQRT(z + 1) + z)` for `2*LN(SQRT(z + 1) + SQRT(z - 1)) - LN(2)`, and
  `LN(1 + 1/z)/2 - LN(1 - 1/z)/2` for `LN((z + 1)/(z - 1))/2`. The same value, an
  uncollected form.
- `ACOT` -> `pi/2 - ATAN`, `ACOS` -> `pi/2 - ASIN`, `ASEC` -> `ACOS(1/z)`,
  `ACSC` -> `ASIN(1/z)`. Note that the first two and the last two compose into each
  other: applying both leaves `ASEC(z)` as `pi/2 - ASIN(1/z)`, which is neither of the
  forms 6.4 prints, so what Derive does with the pair wants checking against the program
  rather than the page.
- `GAMMA(z)` -> `(z - 1)!`, `PERM`, `COMB`, `ERF(z, w)`, `ERFC`, `NORMAL`.
- The symbolic forms of `MIN`, `MAX`, `STEP` and `CHI`.

### Features that are absent

- `FIT`, the least squares fit of section 6.10, in all three of the manual's examples.
- The five financial functions of section 6.12.
- `RANDOM`.
- The antiquotient form of `PRODUCT`, and the antidifference form of `SUM`.
- All nine of the nonscalar algebra rules of section 8.8.
- Boolean simplification: `p OR NOT p`, and the `XOR` and `IMP` normal forms.
- Recursive user functions do not unfold: `FACT(5)` comes back as `5*FACT(4)`.
- Consequently most utility file functions, since they are built on `ITERATE`, `ITERATES`
  and `IF`: ADJOINT, JACOBIAN, GEOMETRY_MATRIX, TAYLOR_SOLVE, TAYLOR_ODE1, TAYLOR_ODE2,
  PICARD, EULER, LIN1_DIFFERENCE, CONVERGENT, CONTINUED_FRACTION, INVERSE, NTH_PRIME and
  others.

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

## Collected sessions

The per-chapter session inventories the tests were drawn from are working notes rather
than a deliverable; they live outside the repository, one file per chapter, at
`/tmp/claude-1000/-home-hniksic-work-rederive/de715675-be66-40c1-8f6b-54f3c8a639b0/scratchpad/sessions/`.
Say the word if they are worth keeping and they can be moved in.
