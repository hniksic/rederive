# Rederive: Requirements

Rederive is a fresh implementation of **Derive**, the computer algebra system
created by Soft Warehouse, Inc. (Honolulu) and sold from 1988 to 2007, from
1999 under Texas Instruments. The goal is to capture Derive's look, feel, and
mathematical coverage, not to clone its code, file formats, or bugs.

Version 1 is a terminal (TUI) application recreating DOS-era Derive (v1-v3,
1988-1996): the clean numbered worksheet, the mnemonic single-letter menus,
textbook-quality typesetting, and the "author it, then tell it what to do"
workflow. The expected implementation is Python with sympy doing the
symbolic/numeric work, but that is an implementation choice; this document
specifies behavior and user experience.

Everything here is backed by research material under `artifacts/`: the Derive
3.14 User Manual, TI's complete Derive 6.1 online help (mirrored), real
`.MTH`/`.DMO`/`.DFW` files from original installations, screenshots of both
eras, and period reviews and Usenet commentary. Section 9 maps the artifacts.

Requirement IDs (`R-ARCH1`, `R-HIST2`, ...) and priority tiers are used
throughout so an implementation plan can reference them directly:

- **Tier 1** - launch target; without these it doesn't feel like Derive.
- **Tier 2** - should follow soon after; rounds out the experience.
- **Tier 3** - stretch / extension material, much of which even real Derive
  shipped as optional utility files or unsupported add-ons.

---

## 1. Vision and scope

### 1.1 Vision

A menu-driven, textbook-typeset, worksheet-based math assistant that feels
like sitting down at Derive again: approachable rather than intimidating,
covering the algebra, calculus, linear algebra, and plotting that a student
or hobbyist actually reaches for.

### 1.2 Audience

The primary audience is **students and curious kids who should come away
amazed by what math can do**. Derive was a classroom tool, not a research
tool, and the remake keeps that position. **Engineers** are a real secondary
audience: quick, trustworthy symbolic/numeric answers and plotting. Research
mathematicians who need Mathematica- or Maple-class depth are explicitly not
the target.

### 1.3 Design principles

Drawn from what the research shows people actually valued about Derive
(`artifacts/forum-and-reviews/community-synthesis.md`):

1. **Worksheet, not REPL.** Expressions are authored, then explicitly told
   what to do (Simplify, Factor, Solve, ...). Nothing evaluates just because
   it was typed. Every result becomes a new, permanently numbered entry that
   later expressions can reference. This "notebook of labeled steps" model is
   Derive's most identity-defining trait.
2. **Looks like a textbook.** Built-up fractions, raised exponents, real
   radical/integral/sum/Greek glyphs; never linear ASCII like `SQRT(x)` in
   rendered output. Non-negotiable for "feels like Derive."
3. **Low intimidation, short learning curve.** Menu-driven operation with
   sensible defaults, so a first-time user gets a right answer without
   reading a manual. Typed function syntax exists as a power-user fallback,
   never as a requirement.
4. **Exact-first arithmetic.** Prefer exact rational/symbolic results;
   numeric approximation is an explicit action or mode, never the default.
5. **Don't guess.** Domain-sensitive simplifications (e.g. `sqrt(x^2) -> x`)
   fire only when justified by an explicit declaration about the variable.
   Favor a visibly unevaluated "stuck" expression over a silently wrong one.
6. **Small and fast.** Derive's appeal came partly from running on hardware
   too weak for its competitors. Stay light and responsive; bloat is a
   regression against the spirit of the product.
7. **Fix the one famous flaw.** The most common period criticism: "most
   functions aren't accessible via the front menu." Do better at
   discoverability (searchable command palette, autocomplete) without
   cluttering the primary menu.
8. **Engine decoupled from UI.** The math engine must be usable through a
   narrow, rendering-agnostic interface (section 3). This is what makes the
   v1 TUI replaceable later (GUI, web, richer terminal graphics) without
   rewriting the engine.
9. **Design for wonder, not just correctness.** For the primary audience, a
   beautiful 3D plot, a step-by-step simplification, or an instant answer to
   "what if?" matters as much as mathematical depth. Prioritize plotting and
   step display; favor an inviting first five minutes over a comprehensive
   manual.

### 1.4 Goals (v1)

- A TUI recreating DOS Derive's interaction model: pane types, mnemonic menu
  system, expression-history behavior, and typesetting style, including the
  original menu wording and single-letter keybindings (section 4).
- A math engine, built and exposed independently of the TUI, covering the
  domains in section 5: symbolic algebra, calculus, linear algebra, number
  theory basics, statistics, 2D/3D plotting, and a programmable
  function-definition layer.
- A native worksheet/session format for saving, reopening, and sharing work
  (section 7), designed fresh rather than reverse-engineered from Derive's.
- Derive-flavored aesthetics (menu wording, layout, status-line conventions,
  color scheme) as a homage, adapted to a modern terminal rather than
  pixel-cloned from a CRT.

### 1.5 Non-goals

- **Bug-for-bug compatibility.** Where Derive had a known wart (silent
  Cauchy-principal-value results for definite integrals with interior
  singularities; exact eigenvalues only up to 4x4; iterated rather than true
  multivariate limits; integer-assumed `SUM`/`PRODUCT` bounds - see
  `artifacts/manuals/functional-capabilities.md` §10), the remake is free to
  do better. It is equally free to be narrower where that is a conscious
  choice.
- **File compatibility.** The remake will not parse or write original
  `.mth`/`.dmo`/`.dfw` files. The archived originals are inspiration for
  syntax style, naming, and worksheet structure only. (`.dfw` turned out to
  be a binary tagged-record format mixing proprietary operator byte codes
  with embedded RTF, so compatibility would be substantial low-value work
  anyway; see `artifacts/sessions/session-files-notes.md`.)
- **Derive's internal architecture.** Nothing requires reproducing muLISP or
  Derive's internal representations (e.g. "a matrix is a vector of row
  vectors") except where user-visible behavior depends on it.
- **TI calculator integration** (Derive 6's TI-89/TI-92+/Voyage 200
  worksheet exchange).
- **A checked dimensional/units type system.** Like the original, units are
  ordinary variables with numeric magnitudes (section 5.10).
- **PDE solving, tensor algebra, linear programming** in the core product.
  These were unsupported community add-ons in real Derive too; candidates
  for later extensions.
- **DOS-era hardware constraints**: video-adapter selection, printer output,
  40-column mode. The DOS *aesthetic* is a core goal; the hardware
  limitations that shaped it are not.
- **A Windows-GUI front end in v1.** The Windows-era look (v5-v6: MDI
  windows, dialogs, sliders, mouse 3D rotation) is documented as a possible
  future front end (section 4.8), enabled by the architecture boundary in
  section 3, but not built now.
- **In-terminal graphics-protocol plot rendering in v1.** Plots are
  rasterized to an image file and opened externally (section 4.7);
  Sixel/Kitty/iTerm2 inline rendering is future work.

### 1.6 Success criteria

- **A 1990s Derive user** should, within minutes of opening the TUI,
  recognize: the numbered worksheet, the mnemonic menu (`A`uthor, so`L`ve,
  `S`implify, ...), the author-then-transform workflow, textbook typesetting,
  and separate 2D/3D plot panes. They should not need to know the engine is
  nothing like the original.
- **A student with no memory of the original** should find the first few
  minutes inviting rather than intimidating, and hit at least one genuine
  "whoa, it did that?" moment early: a plot, a step-by-step simplification,
  an instant answer to a "what if I try this?" question.

---

## 2. Historical background

Full, footnoted detail in `artifacts/history/history-and-timeline.md`; the
short version, as context for design decisions:

Derive descends from **muMATH** (1979-1983), the first CAS for personal
computers, written by **David R. Stoutemyer** (University of Hawaii
engineering professor) and **Albert D. Rich** (LISP hacker, ex-Navy), trading
as The Soft Warehouse in Honolulu. muMATH did symbolic algebra, calculus, and
linear algebra in under 100 KB of RAM on CP/M machines.

**Derive 1.0** (1988) was a ground-up, menu-driven rewrite of the muMATH
engine, designed to feel like an approachable tool rather than a programming
language. It became a European educational bestseller (Austria licensed it
country-wide for secondary schools in 1991; ~40,000 licenses worldwide by
1992) while running on a 286 with 512 KB of RAM. The DOS line ran through
v3.14 (1996) and a DOS-based v4.

**Texas Instruments** collaborated with Soft Warehouse from the early 1990s
on the CAS engine for the TI-92 (1995) and TI-89 (1998), then acquired the
company in August 1999. Derive continued as a Windows product (Derive 5,
2000; Derive 6, 2003-2004) with a modernized GUI, the "Display Steps"
teaching mode, slider-bar plots, and mouse 3D rotation. TI discontinued
Derive on 29 June 2007 in favor of TI-Nspire CAS, whose engine descends from
the same lineage.

What this history contributes to the remake: Derive's low-intimidation
reputation was its entire competitive identity against Mathematica and Maple
(chase approachability, not scope); its tiny footprint was a values
statement worth preserving as "simple and fast"; and its most consistent
criticism, poor menu discoverability of advanced functions, is a concrete
fixable target (principle 7).

---

## 3. Architecture: engine/UI separation

The most important structural requirement, not a UI detail:

- **R-ARCH1.** The TUI talks to the math engine only through a defined,
  narrow interface: author an expression, run an operation (Simplify, Solve,
  Differentiate, ...) on a given entry, request a plot, declare a variable's
  domain, and so on. No engine internals (sympy objects, solver internals)
  leak into rendering/input code; no terminal or keybinding concepts leak
  into the engine.
- **R-ARCH2.** The engine returns results as structured, rendering-agnostic
  data: an expression tree with enough structure to typeset, not a
  pre-formatted string of box-drawing characters. The TUI's renderer turns
  that into on-screen built-up math (section 4.5); a different front end
  would render the same data differently.
- **R-ARCH3.** This boundary is what makes a later front end (GUI, web,
  richer terminal graphics) an addition rather than a rewrite. Treat any
  shortcut that couples engine logic to terminal rendering as a design
  mistake, even under time pressure.

---

## 4. UI and interaction (v1 TUI)

The v1 target is DOS-era Derive specifically. Primary visual references, all
under `artifacts/screenshots/`: `dos-v2-algebra-window-selected-expression.png`,
`dos-v2.60-dosbox-emulator-view.png`, `dos-v3.14-algebra-window-limits-*.png`.
Menu wording and key bindings are cross-checked against the Derive 3.14 User
Manual (see `artifacts/screenshots/ui-look-and-feel-notes.md` §2).

**Reference versions.** Imitate the original in everything - layout, wording,
keys - except colors, which come from an earlier version, the last whose
factory defaults were colorful. The original ships monochrome, so a colorless
screen is not evidence of a wrong palette. Where the manual and the original
disagree, the original wins.

### 4.1 Screen anatomy

A pane is built from the same four horizontal bands the DOS version used:

1. **Work area** - the numbered expression history (section 4.4).
2. **Message line** - one line reporting what is happening or what input is
   expected (e.g. "Computing the slope of x^3 via a limit"), doubling as a
   breadcrumb of the last operation.
3. **Command menu** - two lines below a horizontal rule, labeled `COMMAND:`
   (or a context-specific label in plot panes), listing the pane's full menu
   with the current selection highlighted (section 4.3).
4. **Status line** - bottom line, split into fields: operation/edit-mode
   annotation on the left (e.g. `Simp(#3)`, `Ins`), a center field, and the
   pane type on the right (e.g. `Derive Algebra`, `Derive 2D-plot`). The
   original's center field showed `Free:100%` (muLISP heap); repurpose the
   slot for something useful today, e.g. a busy/elapsed-time indicator
   during long computations.

### 4.2 Color scheme

Derive colors eight slots, and the remake keeps them: a foreground and
background for the work area, and Frame, Option, Prompt, Status, Background
and Border for the chrome. A preset is therefore a table of eight color
numbers.

- **R-COL1.** Default theme: the original's factory colors - white
  expressions on black, red frame, yellow menu, red message line, green
  status line. Exact numbers in `src/rederive/ui/theme.py`.
- **R-COL2.** At least one alternate preset: green-phosphor (green on
  black).
- **R-COL3.** A monochrome preset, matching the original's later factory
  settings: white expressions on black, gray chrome.
- **R-COL4.** The selected expression is a solid inverse-video block (light
  background, dark text), as in the screenshots, in every theme.

### 4.3 Menus and keybindings

The heart of "feels like DOS Derive." Menus are per pane type; every command
has exactly one mnemonic letter, shown capitalized in the menu word, that
both hints and *invokes* the command (no separate confirm step). Tab/Space
and Shift-Tab/Backspace move the highlight without invoking; Enter invokes
the highlighted command.

- **R-MENU1.** `Unremove` appears in the Algebra menu only while there is
  something to restore. The removed-expression buffer is single-level and is
  cleared by the next Simplify/Approximate/Expand/Factor command, so
  `Unremove` disappears again at that point, not at end of session.
- **R-MENU2.** The mnemonic letter works whenever the menu is focused,
  without first tabbing to the item.

#### Algebra pane

```
COMMAND: Author Build Calculus Declare Expand Factor Help Jump soLve Manage
         Options Plot Quit Remove Simplify Transfer moVe Window approX
```

`A`uthor `B`uild `C`alculus `D`eclare `E`xpand `F`actor `H`elp `J`ump
so`L`ve `M`anage `O`ptions `P`lot `Q`uit `R`emove `S`implify `T`ransfer
`U`nremove (conditional, per R-MENU1) mo`V`e `W`indow appro`X`.

#### 2D-plot pane

```
COMMAND: Algebra Center Delete Help Move Options Plot Quit Range Scale
         Transfer Window aXes Zoom
```

This is the Derive 3.14 manual's command set. Earlier versions varied (a v2
screenshot shows `Ticks` in place of `Range`/`Transfer`); the manual's set is
the target.

#### 3D-plot pane

```
COMMAND: Algebra Center Eye Focal Grids Hide Length Options Plot Quit
         Transfer Window aXes Zoom
```

Note the 3D-specific verbs: **Eye** (viewpoint), **Focal** (perspective),
**Hide** (hidden-line toggle).

#### Function keys

| Key | Action |
|---|---|
| `F1` / `Shift-F1` | Next / previous pane |
| `F2` / `Shift-F2` | Next / previous overlaid pane (same slot) |
| `F3` | Copy highlighted expression to the author line (Algebra); toggle trace mode (2D-plot) |
| `F4` | Copy highlighted expression to the author line, parenthesized |
| `F5` | Toggle between the two most recent display/render modes |
| `F6` | Toggle author line between line-edit and subexpression-selection mode |
| `F7` / `Shift-F7` | Magnify plot vertically / horizontally |
| `F8` / `Shift-F8` | Shrink plot vertically / horizontally |
| `F9` / `Shift-F9` | Magnify plot both directions / export current pane as image |
| `F10` / `Shift-F10` | Shrink plot both directions / export whole screen as image |

The `Shift-F9`/`Shift-F10` image exports repurpose the original's
print-screen bindings; the original itself had `Ctrl-F9`/`Ctrl-F10` variants
that saved a TIFF instead of printing, so this has direct precedent.

Other bindings to keep: `Ctrl-Enter` authors *and* immediately simplifies;
typing an expression followed by `=` authors it and shows the simplified
result on the same combined line; `H` from an empty prompt opens Help.

#### Declare prominence

- **R-DECL1.** `D`eclare must remain a first-class, prominent menu entry,
  not a buried preference: declarations drive which simplifications are
  legal (principle 5). Behavior is specified in sections 5.1 and 6.

### 4.4 Expression history and selection

Engine-facing behavior, independent of which front end renders it:

- **R-HIST1.** Expressions are authored as inert input; nothing simplifies
  on entry. A separate action (Simplify, Approximate, Expand, Factor,
  Solve, ...) transforms an expression.
- **R-HIST2.** Every authored expression and every result becomes a new
  numbered entry (`#1`, `#2`, ...). Numbers are never *silently* reused;
  only an explicit user-invoked Renumber, or saving the worksheet,
  resequences labels (section 6, Manage > Renumber).
- **R-HIST3.** Later expressions can reference any earlier entry by number
  (`#n`). This is the provenance mechanism: there is no separate undo stack;
  the numbered history is the log.
- **R-HIST4.** The status line shows which operation produced the selected
  result and from what (e.g. `Simp(#3)`).
- **R-HIST5.** The selected expression, or selected subexpression for
  targeted operations like Substitute, is shown via the inverse-video
  highlight (R-COL4).

### 4.5 Math typesetting in the terminal

- **R-TYPE1.** Render true built-up mathematics using multi-row layout, as
  the original did: stacked fraction bars, raised/lowered baselines for
  exponents and limits, multi-line integral/summation bounds. Never linear
  notation like `sqrt(x)` or `x^2` in rendered output.
- **R-TYPE2.** Use Unicode to improve on the original's DOS line-drawing:
  box-drawing characters for fraction bars and brackets, `√`, `∫ Σ Π`,
  Greek letters, `∞`, `≤ ≥ ≠`, centered dot `·` for multiplication,
  superscript digits for simple integer exponents where they read cleanly.
  The multi-row layout *technique* is what must be preserved, not the
  original's exact glyphs.
- **R-TYPE3.** Fall back to the stacked layout whenever content doesn't fit
  an inline glyph, e.g. an exponent that is itself an expression, or a
  radical over a multi-term radicand.

### 4.6 Pane model

- **R-PANE1.** Three pane types: Algebra, 2D-plot, 3D-plot, each with its
  own menu (section 4.3).
- **R-PANE2.** Manual split (horizontal/vertical) and overlay commands,
  matching the original's `Window Split`/`Window Open`, plus a
  `Window Designate` equivalent to change a pane's type in place. Splitting
  an Algebra pane forks the derivation view: two paths can be pursued from a
  common point, or one expression kept frozen while another pane scrolls.
- **R-PANE3.** Each pane has a visible border and a pane number in a corner;
  the active pane's border/number is visually distinguished. `F1`/`Shift-F1`
  and `F2`/`Shift-F2` switch focus.

### 4.7 Plotting (v1: external image viewer)

- **R-PLOT1.** A Plot command computes the requested 2D or 3D plot and
  rasterizes it to an image file (e.g. PNG); the plotting library is an
  implementation choice.
- **R-PLOT2.** The TUI opens that image in an external viewer: a
  user-configured command or the OS default handler (e.g. `xdg-open`),
  spawned non-blocking so the TUI stays usable.
- **R-PLOT3.** A plot pane still exists in the TUI as a lightweight
  placeholder showing the plotted expressions, range/scale settings, and a
  note that the image is open externally. A low-resolution Unicode-block
  in-pane preview is a nice-to-have.
- **R-PLOT4 (future, not v1).** In-terminal rendering via a terminal
  graphics protocol (Sixel, Kitty, iTerm2), with per-terminal capability
  detection. Because v1 plots are static images, the Windows-era interactive
  features (slider-bar parameter dragging, mouse 3D rotation) are deferred
  with it.

### 4.8 Deferred: a Windows-era-style GUI

Context for the future, not a v1 requirement. Windows Derive (v5-v6) added:
an MDI main window with pull-down menus and toolbar; dialogs with sensible
defaults for every algebra command; mouse-first interaction including
drag-to-select subexpressions; rich worksheets mixing colored annotations,
headings, and inline plots; slider bars and mouse 3D rotation. Detail and
screenshots: `artifacts/screenshots/ui-look-and-feel-notes.md` §3.

"Display Steps" (showing the transformation rules applied, not just the
answer) is *not* GUI-specific: it is structured text output, belongs in the
TUI too, and is a priority per principle 9 (see section 5.1).

### 4.9 Out of scope for v1

Windows-era chrome and dialogs; in-terminal graphics-protocol rendering;
slider animation and mouse 3D rotation; DOS hardware-adapter selection,
printer output, 40-column mode; pixel-exact recreation of any screenshot.

---

## 5. Mathematical capabilities

Primary source: `artifacts/manuals/functional-capabilities.md`, built mainly
from TI's Derive 6.1 online help (mirrored at
`artifacts/manuals/derive6.1-online-help/`), plus the real `.MTH` libraries
under `artifacts/sessions/`.

Where sympy is already stronger than Derive's engine (e.g. much broader
closed-form integration), do not cripple the remake to match. The bar is "at
least what Derive could do, presented the way Derive presented it," not
"exactly what Derive could do and no more."

### 5.1 Symbolic algebra - Tier 1

- Simplify to a "sufficiently simple" form, as an explicit action (R-HIST1).
- Expand (polynomial expansion, partial fractions) and Factor (polynomials,
  integers, matrices), each with a selectable effort level (Derive's tiers:
  Trivial / Squarefree / Rational / Radical / Complex - keep the concept
  even if renamed).
- Substitute a value or expression for a variable (all variables
  substituted simultaneously, enabling clean swaps) or for a selected
  subexpression (exact-match occurrences only).
- Approximate as an action: one-off numeric evaluation to N digits,
  independent of the ambient precision mode.
- **Domain declarations gate simplification** (principle 5): a rule like
  `sqrt(x^2) -> x` fires only when a declaration makes it valid. Declarable:
  type (Integer, Real, Complex, nonscalar/vector) and, for Integer/Real, an
  interval (positive, nonnegative, bounded, ...). Undeclared variables
  default to Real. Tier 1, central to the product's identity.
- **Branch mode** (Principal / Real / Any) controls which root is returned
  for fractional powers of negative/complex numbers: `(-8)^(1/3)` is a
  complex principal-branch result by default; Real mode gives `-2` (the
  original recommended Real for most calculus work); Any permits the
  aggressive `(x^2)^(1/2) -> x`. Tier 1 alongside domain declarations.
- Equation/inequality solving: exact symbolic solving of single equations
  and systems (Groebner-basis-backed in Derive), numeric fallback when no
  closed form exists. Underdetermined solutions represent free parameters
  explicitly (Derive used `@1`, `@2`, ...) so "infinitely many solutions" is
  communicated rather than one being silently picked.
- Arbitrary/undefined functions as first-class objects: declare `f(x)` with
  no body, then differentiate/manipulate expressions containing it via the
  chain rule. Used constantly in real Derive worksheets and libraries.
- Expression ordering controlled by a user-settable variable-order list
  ("more main" variables sort first), as in the original's Manage >
  Ordering.
- **Step display (Tier 2):** show the individual transformation rules
  applied by a simplification/differentiation/integration, in the spirit of
  Derive 6's Display Steps. Engine-level structured output, rendered by any
  front end; a headline teaching feature per principle 9.

### 5.2 Calculus - Tier 1

- **Angle-unit mode**: trig functions operate in a selectable Degree/Radian
  mode (default Radian).
- **Limits**: two-sided by default with an explicit one-sided option;
  resolves classic indeterminate forms (`sin(x)/x` at 0), distinct from
  plain substitution.
- **Differentiation**: arbitrary order, mixed partials, prime notation for
  named functions, chain rule through arbitrary/undefined functions.
- **Integration**: indefinite (with and without explicit constant),
  definite, iterated multiple integrals. When no closed form exists, fall
  back to numeric integration and visibly flag low-confidence results
  (Derive's "Dubious accuracy" warning is a good model). Derive's known
  footgun - interior singularities silently producing a Cauchy principal
  value - is exactly what principle 5 forbids: detect (e.g. a singularity
  check on the interval) or at minimum warn.
- **Series**: Taylor/Maclaurin expansion to a requested order.
- **Sums and products**: symbolic (closed form/telescoping where possible)
  and definite/numeric evaluation.

### 5.3 Differential equations - Tier 2

- First-order ODE solver covering the standard named cases (separable,
  linear, homogeneous, exact, Bernoulli, integrating factor); sympy's
  `dsolve` already classifies these, so this is mostly "expose it well."
- Second-order linear ODE solver (constant and variable coefficient).
- Numeric fallback (at least a Runge-Kutta-family integrator) returning a
  plottable numeric solution.
- Direction-field plotting for first-order ODEs.
- Out of scope: automatic order-reduction above 2nd order (Derive required
  manual reduction too), and PDEs entirely.

### 5.4 Linear algebra - Tier 1

- Vector and matrix literals; elementwise and matrix arithmetic, matrix
  product, scalar product, transpose, determinant, trace; inverse with a
  clear "singular, no inverse" result rather than a crash.
- Row reduction (RREF) as the recommended general method for linear
  systems, including rank-deficient/parameterized systems with
  free-parameter placeholders (as in 5.1).
- Dot product and cross product (3-vector) as named operations.
- Whether matrices are a distinct type or "vectors of vectors" (Derive's
  model) is an implementation decision; only user-facing behavior
  (indexing, row/column extraction, algebra) is required. sympy's `Matrix`
  is a reasonable choice.

### 5.5 Linear algebra - Tier 2

- Eigenvalues and eigenvectors, exact for small matrices and numeric for
  larger ones (Derive was exact only up to 4x4; matching or exceeding that
  is fine).
- Null space / kernel.
- LU-style and QR-style factorization as named operations (Derive: "Turing"
  and "Gram-Schmidt"; the names need not survive, the capability should).
- Characteristic polynomial as a distinct, inspectable result.

### 5.6 Number systems and precision - Tier 1

- **Exact rational/symbolic arithmetic as the default**: results stay exact
  (fractions, radicals, `pi`, `e`) unless approximation is requested.
- Two precision modes at launch: **Exact** (default) and **Approximate**
  (requested significant digits). **Mixed** (approximate irrationals, keep
  rational arithmetic exact - documented as often more accurate than full
  Approximate) is Tier 2, cheap once the other two exist.
- Complex numbers: re/im extraction, conjugate, modulus, argument; a
  reserved imaginary-unit symbol that doesn't collide with a user variable
  named `i` (Derive: `#i` is the constant, `i` stays free).
- **Notation (Tier 2)**, distinct from Precision: Precision controls
  computation; Notation controls display (Rational, Decimal, Scientific, or
  Mixed, plus digit count) - an exactly computed `1/3` can display as
  `0.333333`. Default Notation mirrors the Precision mode (as the original
  did), overridable independently.
- **Radix base (Tier 3)**: independent input/output base 2-36 with
  Binary/Octal/Decimal/Hexadecimal shortcuts.

### 5.7 Number theory - Tier 2/3

Tier 2: gcd, lcm, modular inverse/power, primality testing, next/previous
prime, integer factorization.

Tier 3 (matches Derive's `NumberTheoryFunctions.mth`): extended gcd, linear
congruences, CRT, totient, Moebius, divisor functions, continued fractions,
Farey sequences, Fibonacci/Lucas/Pell, Lucas-Lehmer Mersenne testing.

### 5.8 Statistics, probability, combinatorics - Tier 2

- Descriptive statistics usable directly on a data vector: mean, RMS,
  variance, standard deviation.
- Least-squares curve/line fitting and polynomial interpolation.
- Basic combinatorics: factorial (real/complex via Gamma), permutations,
  combinations, a seedable pseudo-random generator.

Tier 3: named distributions (Poisson, binomial, hypergeometric, Student's
t, F, chi-square), combinatorial number families (Stirling, Bernoulli,
Euler, partitions, Bell), regression beyond least squares (logistic,
Gauss-Newton nonlinear).

### 5.9 Plotting - Tier 1

Paired with the UI requirements in section 4.7; v1 plots are static images,
which bounds what is realistic here.

- 2D at launch: explicit `y = f(x)`, parametric, multi-curve overlay, with
  explicit settable range and scale per axis (not just auto-fit).
- 3D at launch: explicit `z = f(x,y)` surfaces, with explicit settable
  range/box length per axis.

Tier 2: implicit plots, polar mode, inequality/region plots, data-point
plots, contour plots, parametric surfaces, cylindrical/spherical
coordinates, configurable 3D mesh density (the original's Grids setting),
plot accuracy/resolution setting.

Tier 3 / future (needs an interactive front end): slider-driven parameter
animation, mouse 3D rotation, in-terminal graphics rendering.

### 5.10 Units and physical constants - Tier 2/3

Match Derive's deliberately unambitious approach: units are ordinary
variables holding numeric magnitudes relative to a base system, so
expressions combining them simplify via ordinary arithmetic. A small
bundled set of common units and physical constants is a Tier 2/3 target; a
checked unit-type system is out of scope (section 1.5).

### 5.11 Programming and extensibility

Tier 1:

- User-defined variables (`name := expr`) and functions
  (`f(x, ...) := expr`), including recursive and mutually recursive
  definitions.
- Arbitrary/undefined functions (empty-body declarations) - see 5.1; needed
  early because calculus depends on it.
- Conditional/piecewise expressions (`if`) that return visibly unevaluated
  when the condition can't be decided under current declarations.
- **Sequence/table generation**: evaluate an expression over an index range
  or over a vector's elements to build a vector; nest to build a matrix
  (Derive's `VECTOR(u, k, m, n[, step])`). More fundamental than its
  original menu placement suggests: it underlies contour plots and data
  tables (5.9).

Tier 2:

- Default parameter values in function definitions.
- Procedural constructs analogous to Derive's `PROG`/`LOOP` (statement
  blocks, early return, looping).
- An iterate-to-convergence primitive (Derive's `ITERATE`/`ITERATES`), used
  pervasively in the original library (Newton's method, continued
  fractions) in place of explicit loops.

### 5.12 Utility-library coverage - Tier 3 (extension roadmap)

Real Derive shipped ~25 `.MTH` utility files (inventory:
`artifacts/manuals/functional-capabilities.md` §8; the files themselves are
under `artifacts/sessions/`): combinatorial number families, equation-solving
helpers, manual-step linear algebra, extended number theory, ODE solvers and
numeric approximation, recurrence equations, graphics helpers, unit
conversions, physical constants, and special functions (Bessel, elliptic,
Fresnel, hypergeometric, orthogonal polynomials, probability distributions,
zeta).

Treat as a post-launch roadmap, roughly in order: ODE/equation-solving
helpers and combinatorics first (highest everyday value), special functions
and units last. The module mechanism (section 7) should exist before most of
this, so extensions land without touching the core.

### 5.13 Out of scope

PDEs; tensor algebra, linear programming, general complex analysis, coding
theory (community add-ons even in real Derive); a checked units system;
automatic ODE order-reduction above 2nd order; TI worksheet exchange.

---

## 6. Command reference

Inventory of every DOS Derive menu command and the remake's disposition
toward it. Full original behavior is in the complete manual transcript,
`artifacts/manuals/derive-3.14-manual-transcript.md` (chapters 2-5 and the
Menu Commands appendix in particular).

Disposition legend: **Core** = engine behavior, specified in section 5;
**UI** = interaction behavior, specified here or in section 4; **File** =
worksheet/session behavior, specified in section 7; **Out** = not planned.

### 6.1 Algebra pane

| Key | Command | Original behavior | Disposition |
|---|---|---|---|
| `A` | Author | Type a new expression on the input line. | UI, Tier 1 (R-HIST1) |
| `B` | Build | Construct an expression step-by-step: operand, operator/function, next operand, ..., Done. | UI, Tier 2 |
| `C` | Calculus | Submenu: Differentiate, Integrate, Limit, Product, Sum, Taylor, Vector. | Core, Tier 1 (5.2; Vector: 5.11) |
| `D` | Declare | Submenu: Function, Variable (Value/Integer/Real/Complex/Nonscalar), Matrix, vectoR. | Core + UI, Tier 1 (5.1, 5.4, R-DECL1) |
| `E` | Expand | Expand w.r.t. some/all variables. | Core, Tier 1 (5.1) |
| `F` | Factor | Factor w.r.t. some/all variables. | Core, Tier 1 (5.1) |
| `H` | Help | Static help topics: editing, functions, per-window commands, utility files, state. | Tier 2, upgraded (below) |
| `J` | Jump | Move the highlight to a given expression number. | UI, Tier 1 |
| `L` | soLve | Solve an equation, inequality, or system. | Core, Tier 1 (5.1) |
| `M` | Manage | Submenu: see below. | mixed |
| `O` | Options | Submenu: see below. | mixed |
| `P` | Plot | Switch to (or open) a 2D- or 3D-plot pane. | UI, Tier 1 (R-PANE1) |
| `Q` | Quit | Exit; prompts `Abandon expressions (Y/N)?` if there is unsaved work. | UI, Tier 1, keep the prompt |
| `R` | Remove | Delete a contiguous block of expressions. | UI, Tier 1 |
| `S` | Simplify | Simplify the highlighted (sub)expression. | Core, Tier 1 (5.1) |
| `T` | Transfer | Submenu: Load, Save, Merge, Clear, Demo, Print. | mostly File (below) |
| `U` | Unremove | Restore the last-removed block. | UI, Tier 1 (R-MENU1) |
| `V` | moVe | Reorder a contiguous block of expressions. | Tier 3, deferred (below) |
| `W` | Window | Submenu: Close, Designate, Flip, Goto, Next, Open, Previous, Split. | UI (below) |
| `X` | approX | Force numeric evaluation. | Core, Tier 1 (5.6) |

**Help.** Per principle 7, this should be more than the original's static
topic tree: a searchable command/function palette with inline usage and
examples, drawing on the same content as the mirrored online help. Keep the
`H`-from-empty-prompt shortcut.

**Jump** prompts for a label number and moves the highlight there; the
natural companion to `#n` references (R-HIST3). The original also offered
subexpression-level navigation (descend into operands, step through the
terms of a sum, via arrow keys / a Ctrl-key "diamond"); expression-level
Jump is Tier 1, subexpression navigation Tier 2.

**Remove / Unremove.** Remove deletes a contiguous block (Start/End
fields). Unremove restores the most recently removed block; the buffer is
single-level and cleared by the next Simplify/Approximate/Expand/Factor
(hence R-MENU1).

**moVe** reordered a block while keeping label numbers, letting display
order go numerically out of sequence. This sits in tension with principle 1
(the append-only numbered history); defer it, and revisit only if real
worksheets need reordering for readability.

**Manage submenu:**

| Command | Original behavior | Disposition |
|---|---|---|
| Annotate | Edit the provenance tag attached to an expression (`Simp(#3)`). | UI, Tier 2. Hand-edited annotations + save was literally how original `.DMO` demos were authored; keep the same mechanism for the remake's demo files (section 7, R-WS6). |
| Branch | Principal / Real / Any root selection. | Core, Tier 1 (5.1) |
| Exponential, Logarithm | Auto/Collect/Expand direction for identities like `ln x + ln z <-> ln(x*z)`. | Core, Tier 2 (refines Simplify/Expand) |
| Ordering | Variable-order list controlling term/factor sort order. | Core, Tier 1 (5.1) |
| Renumber | Reassign fresh sequential labels matching physical order; updates `#n` references in annotations. Saving also auto-renumbered. | UI, Tier 2 (see R-HIST2, R-WS7) |
| Substitute | Simultaneous variable substitution, or exact-match subexpression substitution. | Core, Tier 1 (5.1) |
| Trigonometry | Auto/Collect/Expand direction, Sines/Cosines preference, and Degree/Radian angle mode. | Core: angle mode Tier 1 (5.2); directions Tier 2 |

**Options submenu:**

| Command | Original behavior | Disposition |
|---|---|---|
| Color | Menu/work-area color settings. | UI, Tier 1 (R-COL1-4) |
| Display | Text vs. graphics screen mode. | Superseded by R-TYPE1-3; no toggle needed |
| Execute | Shell out to DOS. | Out |
| Input | Character vs. Word naming mode; case sensitivity; default arrow-key mode. | Core, Tier 1. Both modes are implemented, with Derive's own defaults: Character mode (`xyz` parses as `x*y*z`) and case-insensitive. Word mode is a user option, not the default. Arrow-key default ties to the `F6` toggle. |
| Mute | Error beep on/off. | UI, Tier 3 |
| Notation | Number display style + digits. | Core, Tier 2 (5.6) |
| Output | Normal/Compressed spacing; multiplication glyph (asterisk/dot/implicit). | UI, Tier 3 |
| Precision | Exact/Approximate/Mixed + digits. | Core, Tier 1 (5.6) |
| Radix | Input/output base 2-36. | Core, Tier 3 (5.6) |

**Transfer submenu:**

| Command | Original behavior | Disposition |
|---|---|---|
| Load Derive / Merge | Replace with, or append, expressions from a `.MTH` file. | File, Tier 1 (R-WS5) |
| Load Utility | Load definitions without displaying them. | File, Tier 1 (R-LIB1-3) |
| Load daTa | Load numeric data arrays. | File, Tier 2 |
| Load/Save State | Settings persistence. | File, Tier 2 (per-worksheet settings, R-WS2) |
| Save Derive | Save the worksheet. | File, Tier 1 (section 7) |
| Save Basic/C/Fortran/Pascal | Export an expression as source code. | Tier 3, genuinely cheap: sympy ships code printers (`pycode`, `ccode`, `fcode`) |
| Clear All/Expressions/Functions/Variables | Granular state reset. | UI/File, Tier 2 (R-WS5) |
| Demo | Step through a `.DMO` file: show, auto-simplify, pause per keypress; Esc suspends, a later Demo resumes. | Tier 2 (R-WS6); the archived `.DMO` files make ready-made test content |
| Print | Printer output. | Out; its Print Screen variant is the ancestor of the `Shift-F9`/`F10` image export |

**Window submenu:** Split (horizontal/vertical, prompting for the split
position, default even), Open (overlay on the active pane), Flip/`F2`
between overlaid panes, Next/`F1`, Previous/`Shift-F1` are Tier 1
(R-PANE2/3). Close, Designate (change pane type in place), Goto (by window
number) are Tier 2.

### 6.2 2D-plot pane

Commands not shared with the Algebra pane:

| Key | Command | Original behavior | Disposition |
|---|---|---|---|
| `A` | Algebra | Switch to an algebra pane. | UI, Tier 1 |
| `C` | Center | Center the plot on the cross position. | UI, Tier 2 |
| `D` | Delete | Remove one or all expressions from the plot list (All/Butlast/First/Last). | UI, Tier 2 |
| `M` | Move | Move the trace cross to given coordinates. | UI, Tier 1 (companion to `F3` trace) |
| `P` | Plot | Render the expression highlighted in the algebra pane. | Core, Tier 1 (5.9, R-PLOT1-3) |
| `R` | Range | Set the plotted range. | Core, Tier 1 (5.9) |
| `S` | Scale | Set the plot scale. | Core, Tier 1 (5.9) |
| `X` | aXes | Aspect ratio, axis titles/labels. | UI, Tier 2 |
| `Z` | Zoom | Adjust scale in/out. | UI, Tier 1 (the `F7`-`F10` bindings) |
| | Options > Accuracy | Plotting accuracy/resolution. | Core, Tier 2 |
| | Options > Color | Plot-color cycling; plot/axes colors. | UI, Tier 2 |
| | Options > State | Coordinate display, follow mode, trace mode, point size. | UI, Tier 2 |

### 6.3 3D-plot pane

| Key | Command | Original behavior | Disposition |
|---|---|---|---|
| `C` | Center | Set the bounding-box center. | UI, Tier 2 |
| `E` | Eye | Set the viewer's eye coordinates. | UI, Tier 3 (deferred with interactive 3D) |
| `F` | Focal | Set the focal point (perspective). | UI, Tier 3 (same) |
| `G` | Grids | Set surface mesh density. | Core, Tier 2 (5.9) |
| `H` | Hide | Toggle hidden-line removal. | UI, Tier 2; may become a rendering-style toggle under a shaded default |
| `L` | Length | Set bounding-box side lengths (range per axis). | Core, Tier 1 (5.9) |
| `X` | aXes | Toggle axis display. | UI, Tier 2 |
| `Z` | Zoom | Adjust box side lengths in/out. | UI, Tier 2 |
| | Transfer > Acrospin | Export points for the AcroSpin animation viewer. | Out (1990s companion program) |

---

## 7. Worksheet and library files

Requirements for the remake's own, new formats. Informed by the real files
under `artifacts/sessions/` (analysis in `session-files-notes.md`), which are
inspiration, not a compatibility target (section 1.5).

What the originals looked like, in brief: `.MTH` library files were plain
ASCII - top-level expressions separated by blank lines, `NAME(args):=expr`
definitions, bare string literals serving as comments, `~` line
continuation, and mode settings as ordinary assignments
(`Precision:=Approximate`). `.DMO` demo scripts were the same style with
`;` comments and input expressions only. The numbered `#n` history was a
runtime concept, never stored in files. `.DFW` (Windows worksheets) was a
binary tagged-record container. The takeaway: keep the library format's
simplicity, and fix the worksheet format's complexity - a human-readable
worksheet format is a deliberate improvement.

### 7.1 Worksheet format

- **R-WS1.** Plain-text, human-readable, diffable: usable with version
  control and a text editor in a pinch.
- **R-WS2.** A worksheet represents, in order:
  - Numbered expression entries: input expression, optionally its result
    and the operation that produced it (persisting the `#n` history of
    R-HIST2-4, which the original never saved).
  - Free-form annotation blocks interleaved between expressions (headings,
    notes) - Windows-era worksheets used this heavily.
  - Embedded plot definitions, optionally with a cached rendered image so a
    reopened worksheet shows plots without recomputation.
  - Per-worksheet settings (precision, notation, variable ordering,
    declared domains). Derive 6 moved settings from global-only to
    per-worksheet; inherit that model.
  - Function/variable definitions authored in the worksheet.
- **R-WS3.** Concrete syntax and extension are an implementation decision: a
  lightweight custom text syntax, or a documented structured format
  (JSON/YAML schema, Markdown with math blocks). Prioritize what a text
  editor and `git diff` handle gracefully.
- **R-WS4.** Loading is a pure replay: author each stored expression and
  re-apply its recorded operation. No separate serialization of
  engine-internal state.
- **R-WS5.** Distinct operations for *replace* (load into a cleared view)
  and *append* (merge another worksheet onto the current one), plus
  granular reset (clear expressions / functions / variables / everything).
- **R-WS6 (Tier 2).** Demo mode: step through a worksheet's expressions,
  auto-running each recorded operation and pausing for a keypress
  (original: `Transfer Demo` on `.DMO` files). Directly testable against
  the archived `.DMO` files.
- **R-WS7 (Tier 3).** Renumber on save (and optionally on demand, as
  Manage > Renumber) so saved worksheets don't accumulate numbering gaps;
  `#n` references update to match.

### 7.2 Library / module mechanism

- **R-LIB1.** A way to package reusable definitions into a loadable unit,
  analogous to Derive's `.MTH` utility files; prerequisite for the Tier 3
  roadmap (5.12).
- **R-LIB2.** Implementation open: plain Python modules registering
  functions with the engine, or a text format sharing the worksheet's
  expression syntax. Favor Derive's simplicity (a flat sequence of
  definitions) over a heavyweight package system.
- **R-LIB3 (nice-to-have).** Autoloading: referencing a not-yet-loaded
  library function triggers loading its module, as in the original.

### 7.3 Not required

Reading or writing original `.mth`/`.dmo`/`.dfw` files; reproducing the
muLISP settings format; binary compatibility with anything.

---

## 8. Glossary

| Term | Meaning |
|---|---|
| **Algebra window** | The main worksheet pane: the scrolling, numbered list of authored expressions and results. |
| **Author** | Enter a new expression, assignment, or function definition (inert until acted on). |
| **Simplify / Approximate** | The explicit actions that transform an authored expression: symbolic simplification, or numeric evaluation to N digits. |
| **Declare** | Assert a variable's domain (Real, Integer, positive, ...), gating which simplifications are legal. |
| **`#n`** | The permanent sequential number of every worksheet entry; later expressions reference it. |
| **Branch mode** | Principal/Real/Any: which root is returned for fractional powers of negative/complex numbers. |
| **Precision vs. Notation** | Precision governs computation (Exact/Approximate/Mixed); Notation governs display (Rational/Decimal/Scientific/Mixed). |
| **Build** | Construct an expression step-by-step (operand, operator, operand, ..., Done) instead of typing syntax. |
| **Display Steps** | Derive 6 feature showing the transformation rules applied, not just the result. |
| **Slider bar** | Derive 6 control that live-redraws a plot as a parameter is dragged. |
| **soft-key menu** | The DOS two-line mnemonic command bar with one capitalized shortcut letter per word. |
| **`.MTH` / `.DMO` / `.DFW`** | Original file formats: plain-text library, plain-text demo script, binary Windows worksheet. |
| **muLISP / muMATH** | The LISP dialect Derive was written in, and its 1979-1983 predecessor CAS by the same authors. |

---

## 9. Research artifacts

| Directory | Contents | Consult for |
|---|---|---|
| `artifacts/history/` | Footnoted history and timeline plus ~14 downloaded primary/secondary sources. | Why Derive was positioned as it was; verifying historical claims. |
| `artifacts/screenshots/` | 32 images (DOS v1-v3, Windows v5-v6) plus `ui-look-and-feel-notes.md` (menu wording, key bindings, tiling mechanics, palette, typesetting). | Any look-and-feel question - check the image before assuming. |
| `artifacts/manuals/` | Derive 3.14 User Manual, TI Derive 5/6 Introduction booklets, a University of Hawaii calculus lab book, the complete Derive 6.1 online help (~408 HTML pages), `functional-capabilities.md`, `derive-3.14-manual-transcript.md` (full transcript of the 3.14 manual). | Exact function syntax and semantics; what any command actually did. |
| `artifacts/sessions/` | Real `.MTH`/`.DMO` files from original installations, a real `.DFW`, university course files, `session-files-notes.md` (format analysis). | Concrete syntax examples; how real files were structured. |
| `artifacts/forum-and-reviews/` | Usenet, magazine, forum, and academic commentary plus `community-synthesis.md`. | What users loved and hated - sanity-checking design against lived experience. |

Points the historical record leaves genuinely uncertain (detail in
`artifacts/history/history-and-timeline.md`): whether a Windows "Derive 4"
existed pre-TI (the evidence says v4 was still DOS; v5/v6 are "the Windows
era"); the exact TI acquisition closing date in 1999; original retail
pricing; and the "blue DOS screen" some people remember, which no screenshot
or manual corroborates - treat as folklore. None of these affect the
requirements.
