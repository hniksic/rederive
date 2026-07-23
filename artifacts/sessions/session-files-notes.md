# Derive session/worksheet files: research notes

Research for the Derive remake requirements document. Goal: locate actual Derive
files (`.mth` math libraries, `.dfw`/session worksheets) to document the real file
format and vocabulary, not just descriptions of it.

Result: substantial success. Two complete DOS Derive installations (v1.62 and
v3.14) were downloaded from archive.org and their entire `.MTH`/`.DMO` utility
library extracted (63 `.MTH` files, 12 `.DMO` demo files, 12 `.DOC` package docs).
In addition, real Windows-era files (one actual `.DFW` binary worksheet and three
`.MTH` files) were recovered from a still-live University of Hawaii mathematics
department course page - the same department where Derive's original authors
(David Stoutemyer et al., Soft Warehouse) and later course-book authors (Ralph
Freese, David Stegenga) taught.

No abandonware sites required registration, payment, or bypassing any wall; the
archive.org items are freely downloadable zips.

## Inventory of files saved

All paths below are under `/home/hniksic/work/derive/artifacts/sessions/`.

### `derive-1.62-dos/` - Derive Mathematical Assistant v1.62 (Soft Warehouse, 1988)

Source: https://archive.org/details/derivecas162 - file `derive.zip` (contains a
full runnable DOS diskette image: `DERIVE.EXE`, `.HLP`, `.INI`, and the `.MTH`
library). Only the plain-text files were kept (the `.EXE`/`.HLP`/`.COM` binaries
were discarded per the task's focus).

23 files, all plain 7-bit ASCII text with CRLF line endings:
`ALGEBRA.MTH`, `ANNUITY.MTH`, `APPROX.MTH`, `ARITH.MTH`, `CALCULUS.MTH`,
`COORD.MTH`, `DERIV.MTH`, `ENGLISH.MTH`, `FUNCTION.MTH`, `INTEGRAL.MTH`,
`MATRIX.MTH`, `METRIC.MTH`, `ODE1.MTH`, `ODE2.MTH`, `PHYSICAL.MTH`, `PLOT2D.MTH`,
`PLOT3D.MTH`, `PLOTPARA.MTH`, `RECUREQN.MTH`, `SPECIAL.MTH`, `TRIG.MTH`,
plus `DERIVE.INI` (settings file) and `README.txt` (renamed from `README`).
`SUPPORT.PAS` (Pascal numeric-routine listing, not itself Derive syntax) was
kept for reference.

### `derive-3.14-dos/` - Derive Mathematical Assistant v3.14 (Soft Warehouse, 1996)

Source: https://archive.org/details/derive314cas - file `DERIVE_3.14.zip`.

30 `.MTH` files (superset/evolution of the 1.62 set - names changed and grew):
`APPROX.MTH`, `BESSEL.MTH`, `DIF_APPS.MTH`, `ELLIPTIC.MTH`, `ENGLISH.MTH`,
`EXP_INT.MTH`, `FRESNEL.MTH`, `GRAPHICS.MTH`, `HYPERGEO.MTH`, `INT_APPS.MTH`,
`METRIC.MTH`, `MISC.MTH`, `NUMBER.MTH`, `NUMERIC.MTH`, `ODE1.MTH`, `ODE2.MTH`,
`ODE_APPR.MTH`, `ORTH_POL.MTH`, `PHYSICAL.MTH`, `PLOT2D.MTH`, `PLOT3D.MTH`,
`PLOTPARA.MTH`, `PROBABIL.MTH`, `RECUREQN.MTH`, `SOLVE.MTH`, `VECTOR.MTH`,
`ZETA.MTH`.

5 `.DMO` "demo" files - narrated example sessions meant to be stepped through
inside Derive (`ALGEBRA.DMO`, `ARITH.DMO`, `CALCULUS.DMO`, `FUNCTION.DMO`,
`MATRIX.DMO`, `TRIG.DMO`). These are the closest thing in this release to a
"session/worksheet" file, distinct from the `.MTH` function-library files.

Plus `DERIVE.INI`, `README.txt`, `SUPPORT.PAS`.

`USERS/` subdirectory - third-party contributed packages shipped on the same
diskette, each with matching `.MTH` (code), `.DOC` (plain-text documentation with
worked examples), and often `.DMO` (demo script): `CALLOPTN` (stock call
options), `CURV_FIT`, `FIT_`, `FUN`, `HERMITE`, `ODE`, `PSERIES`, `ROMB_INT`,
`SECANT`, `SPLINE`, `TAY_ODE`, `TENSOR` (+ `TENSOR1.DMO`/`TENSOR2.DMO`), plus a
`FIT_DEMO.BAT` DOS batch launcher.

### `hawaii-coursework/` - real Windows-era files, University of Hawaii

Source: http://math.hawaii.edu/lab/ (Derive Utilities page for the UH Calculus
Computer Lab, still live as of this research). These are course-authored files,
not Soft Warehouse originals, but they are genuine files created and used with
"Derive for Windows" in a real classroom, referenced from the book *Calculus
Concepts Using Derive for Windows* (Freese & Stegenga, Prentice-Hall 2000;
UH Math professors) - see https://math.hawaii.edu/RDPublishing/CalcLabBook/.

- `add-head.dfw` (http://math.hawaii.edu/lab/add-head.dfw) - an actual `.DFW`
  binary worksheet file, "DERIVE for Windows version 5.05", saved 01 Oct 2002.
  This is the only real `.dfw` obtained; see format notes below - it is NOT
  plain text like `.mth`.
- `add-util.mth` (http://math.hawaii.edu/lab/add-util.mth) - plain-text utility
  library of calculus helper functions (secant/tangent lines, curve fitting,
  splines, Newton's method, Euler/Runge-Kutta ODE solvers).
- `add-head.mth` (http://math.hawaii.edu/lab/add-head.mth) - a short loader
  file: comment + a literal vector of function-signature "stubs" (a crib sheet)
  + a `LOAD("add-util-ver4")` call.
- `add-util-ver4.mth` (http://math.hawaii.edu/lab/add-util-ver4.mth) - an
  earlier/parallel version of the utility library.

Other confirmed-real UH resources found but not downloaded (out of scope, PDFs
not source files): `derive_lab_manual.pdf` (full course book),
`CalculusConceptsUsingDerive.pdf`, per-chapter lab PDFs under
`math.hawaii.edu/lab/241/`.

### Also located, not downloaded (documentation, not source files)

- **Derive User Group (DUG) Newsletters** - a long-running paper newsletter for
  Derive users, with ~90+ issues scanned and hosted on archive.org, e.g.
  https://archive.org/details/Derive_User_Group_Newsletter_01 through at least
  issue 100 (search: `archive.org/advancedsearch.php?q="Derive User Group"`).
  These are scanned PDFs (not raw `.mth` text) but contain many printed
  transcripts of expressions and tips-and-tricks code that would be useful for
  a later, deeper syntax-mining pass if needed.
- https://archive.org/details/DavidStoutemyer-TheSoftWarehouse-Interview - an
  interview with Derive's chief architect; historical/background only.
- https://archive.org/details/studenteditionof0000unse - a scanned book (the
  Student Edition manual), not the software.
- WinWorld (winworldpc.com) only hosts Derive 2.x and 3.x for DOS - same
  vintage as what was already obtained from archive.org, so not re-fetched.
  No Derive 5/6 for Windows package was found as a redistributable archive on
  WinWorld, vetusware (requires registration - skipped per instructions), or
  in a quick discmaster.textfiles.com check.
- muMATH (Derive's 1980s predecessor, same authors) disk images exist at
  https://archive.org/details/muMATH-Disk1 and `-Disk2`; not pursued since it
  predates and does not use the `.MTH`/`.DFW` extensions/format themselves.

## File format observations

### `.MTH` files (DOS Derive, all versions checked: 1.62, 3.14, and the 2002
Windows-authored Hawaii files) - **plain ASCII text**, CRLF line endings.

- **Structure**: a sequence of top-level Derive *expressions*, each separated by
  a blank line. Loading a `.MTH` file via `Transfer Load` is equivalent to
  typing/Authoring each expression in turn at the interactive prompt.
- **Comments**: there is no dedicated comment syntax. Instead, a bare Derive
  *string literal* (double-quoted text) is used as a "comment expression" - it
  is a syntactically valid, self-evaluating expression that does nothing but
  display descriptive text when the file is stepped through. Example (from
  `ODE1.MTH`):
  ```
  "File:  ODE1.MTH   (c)          01/10/90        Soft Warehouse, Inc."

  "ELEMENTARY METHODS FOR SOLVING 1ST-ORDER ORDINARY DIFFERENTIAL EQUATIONS."

  "Solves the separable ODE  y' = p(x) q(y)  for  y(x0) = y0:"

  SEPARABLE(px,qy,x,y_,x0,y0):=INT(1/qy,y_,y0,y_)=INT(px,x,x0,x)
  ```
  The very first line of a file is conventionally a self-identifying banner
  string naming the file and its copyright holder.
- **Line-continuation**: Derive's DOS editor wrapped long lines at a fixed
  column; a line broken mid-expression ends with a tilde `~` and the next
  physical line continues the same logical expression, e.g.:
  ```
  BERNOULLI(px,qx,k,x,y_,x0,y0):=LIM(LINEAR1((1-k)*px,(1-k)*qx,x,y_,x0,y0),y_,y_~
  ^(1-k))
  ```
  (i.e. the `~` is stripped and the two physical lines concatenated to recover
  `...,y_,y_^(1-k))`.)
- **Function/variable definitions**: `NAME(arg1,arg2,...):=expression` for
  functions, `NAME:=expression` for variables/constants. No `let`/`var`
  keywords.
- **Notation quirks observed across many files**:
  - Trailing underscore on identifiers (`y_`, `x0`, `v_`, `n_`) is a widespread
    author convention to avoid colliding with Derive's own bound/special
    variable names, not a language requirement.
  - `SUB` is the (in)famous ASCII-only indexing/subscript operator:
    `m_ SUB k SUB 2` means `m_[k][2]` (element k, then element 2 of that row).
    Windows-era files sometimes render the same operator as a raised dot glyph
    when saved through the GUI, but `SUB` is what a plain-text/DOS file spells.
  - Backtick `` ` `` is the matrix/vector transpose postfix operator.
  - `#e` and `#i` are the reserved names for Euler's number and the imaginary
    unit; `pi` and `inf` are spelled out in full lowercase (no special glyph in
    the plain-text encoding).
  - `IF(condition, then, else)` is the general conditional form; some library
    functions supply a redundant third argument that repeats the "then" branch
    when there is no separate false case (an authoring idiom, not a language
    requirement), e.g. `LIN2_HOM(d,x):=IF(d=0,[1,x],LIN2_HOM_AUX(d,x),LIN2_HOM_AUX(d,x))`.
  - State/config assignments can appear inside a `.MTH` file too, not just
    function definitions, e.g. `SPLINE.MTH` (in `USERS/`) opens with
    `Precision:=Approximate`, `PrecisionDigits:=10`, `NotationDigits:=6` before
    any function definitions - i.e. a worksheet can set interpreter options as
    ordinary top-level statements.
- **No numbered history in `.MTH` files themselves.** The `#n:` numbering seen
  in Derive's on-screen "Algebra window" (each expression gets a permanent
  sequential number, `#1`, `#2`, ... referencable in later expressions) is a
  *runtime/session* concept, not something stored literally inside a `.MTH`
  utility file. It shows up only in prose documentation that transcribes a
  session, e.g. `SPLINE.DOC`:
  ```
  #29: SPLINE([[1, 1], [3, 2], [4, 2], [5, 3]], t)

  #30: [[2 t + 1, -0.521739 t^3 + 1.52173 t + 1],
        [t + 3  ,  0.413043 t^3 - 0.391304 t^2 - 0.0217391 t + 2],
        [t + 4  , -0.282608 t^3 + 0.847826 t^2 + 0.434782 t + 2]]
  ```
  This confirms the numbering behavior (permanent, sequential, referenced by
  later `#n` inputs and outputs) even though no literal `.MTH`/`.DFW` file
  encodes it as `#n:` text - the number is assigned by the running program
  when the expression is authored/simplified, and would live inside whatever
  internal binary state a saved worksheet keeps (see `.DFW`, below).

### `.DMO` files (demo scripts) - same plain-text conventions as `.MTH`, but
comments use a leading semicolon `;` instead of a quoted-string expression, and
the file is a flat list of *input* expressions only (no function
definitions) meant to be single-stepped and simplified live by the user, e.g.
(`ALGEBRA.DMO`):
```
; Expands as necessary to simplify expressions
(x+a)^2-2*a*x

; Unnecessary top-level expansion is avoided
2 (x^2 - y^2)^6 - (x^2 - y^2)^5(2 x^2 - 3)
```

### `DERIVE.INI` - plain-text settings file, CRLF, one setting per line, format
`*SETTING-NAME*  value`, where values are frequently literal muLISP/Lisp
S-expressions (Derive is implemented in muLISP), e.g.:
```
*PRECISION*  |Exact|
*NOTATION*  |Rational|
*VARIABLE-ORDER*  (\x \y \z)
*DEFAULT-DOMAIN*  (|Real| ((((1 INF) . -1) . T) (INF) . T))
```
`|Symbol|` pipe-quoting is muLISP's syntax for symbols, confirming the
config/state layer is just serialized Lisp data.

### `.DFW` files (Derive for Windows worksheet) - **binary**, not plain text.

The one real example obtained (`add-head.dfw`, saved by DfW 5.05) shows:
- A plain-text one-line banner header: `DERIVE for Windows version 5.05 DfW
  file saved on 01 Oct 2002`, terminated `\r\n` then a `SUB`/EOF byte `0x1A`.
- After the header, a sequence of binary-tagged records. Each function
  definition is stored as mostly-ASCII text (very close to `.MTH` syntax) but
  length-prefixed/tagged with binary bytes rather than blank-line-separated,
  and using **high-byte (0x80-0xFF) single-byte codes for math operators**
  instead of the ASCII spellings used in `.MTH` files - confirmed by hex
  inspection:
  - byte `0xA4` = the big Sigma summation operator (`SUM(...)`, spelled `SUM`
    in `.MTH` text but rendered as a literal Σ-glyph byte in `.DFW`)
  - byte `0xB7` = multiplication dot `·` (in place of `*`)
  - byte `0x99` = the `SUB` subscript operator, as a single glyph
  This is consistent with Derive's Windows GUI rendering expressions with a
  proprietary math font/character set, and the `.DFW` save format simply
  persisting those same byte codes rather than re-expanding them to ASCII
  keywords.
- Interleaved with the expression text are distinct **object records** with
  recognizable type tags, e.g. `CTextObj` (a rich-text annotation, storing an
  embedded **RTF** document - `{\rtf1\ansi\ansicpg1252...}` - for headings and
  free-form notes placed in the worksheet) and `CExpnObj` (an expression
  object, additionally carrying a `LOAD("path")` string that reproduces how
  the worksheet auto-loads its companion `.MTH` utility file on open, e.g.
  `LOAD("G:\Dfw5\M242L\add-util.mth")`).
- Net takeaway for the remake: `.DFW` is a **structured worksheet container**
  mixing (a) binary framing, (b) near-`.MTH`-syntax expression text using a
  proprietary single-byte math-operator encoding, and (c) embedded RTF text
  objects for annotations/headings - i.e. it is much closer to a lightweight
  "document with embedded rich text + math objects" format than to the flat,
  purely-textual `.MTH`/`.DMO` format. A from-scratch remake that wants a
  human-readable/diffable native format would be deviating from the original
  `.DFW` on purpose by choosing plain text (which matches the project's stated
  intent of *not* being format/bug-compatible).

## Requested library file list vs. what was actually found

Mapping the exact names mentioned in the task to what shipped historically
(names changed across versions; nothing called exactly `INT.MTH`,
`RECUR.MTH`, `GEOMETRY.MTH`, or `EQUATION.MTH` was ever shipped by Soft
Warehouse under those exact names):

| Requested (approximate) | Actual shipped name(s) found |
|---|---|
| INT.MTH | `INTEGRAL.MTH` (1.62); folded into core in 3.14, plus `INT_APPS.MTH` |
| VECTOR.MTH | `VECTOR.MTH` (3.14) |
| MISC.MTH | `MISC.MTH` (3.14) |
| RECUR.MTH | `RECUREQN.MTH` (1.62 and 3.14) |
| GEOMETRY.MTH | not found under this name; closest is `COORD.MTH` (1.62, coordinate systems) |
| EQUATION.MTH | not found under this name; closest is `SOLVE.MTH` (3.14) |
| ODE1.MTH / ODE2.MTH | present verbatim in both 1.62 and 3.14 |
| PROBABIL.MTH | `PROBABIL.MTH` (3.14) |
| GRAPHICS.MTH | `GRAPHICS.MTH` (3.14) |

## Sources (all free, no registration/payment required)

- https://archive.org/details/derivecas162 - Derive 1.62 (1988), `derive.zip`
- https://archive.org/details/derive314cas - Derive 3.14 (1996), `DERIVE_3.14.zip`
- http://math.hawaii.edu/lab/ (utilities page: `add-head.dfw`, `add-util.mth`,
  `add-head.mth`, `add-util-ver4.mth`)
- https://math.hawaii.edu/RDPublishing/CalcLabBook/ - context for the Hawaii
  course files (book: *Calculus Concepts Using Derive for Windows*)
- https://archive.org/details/Derive_User_Group_Newsletter_01 (and siblings
  `_02` .. `_100`) - DUG newsletters, documentation only, not re-downloaded
- https://winworldpc.com/product/derive/3x, `/2x` - checked, same DOS vintage
  already covered by archive.org copies above
- https://vetusware.com/download/Derive%204.11%204.11/?id=8610 - located but
  requires account registration; skipped per task instructions
