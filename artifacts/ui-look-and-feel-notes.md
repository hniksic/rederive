# Derive UI / Look-and-Feel Research Notes

Research pass over the visual look and feel of Soft Warehouse / Texas Instruments
"Derive" across its DOS text-mode era (v1-v3, roughly 1988-1996) and its Windows GUI
era (v5-v6, roughly 1998-2004). Compiled from publicly accessible archives (Wikipedia,
WinWorld, archive.org, a UK Derive reseller's marketing PDFs, a 1990s palmtop-computer
magazine, and a retrocomputing blog). All images referenced in the manifest below were
downloaded into this directory. No login-walled or paywalled material was used.

Primary authoritative source for the DOS section: the actual **Derive Version 3.14 User
Manual** (Soft Warehouse, 1996), OCR text retrieved from archive.org
(`derivecas314manual`). Quotations below are paraphrased/lightly cleaned from that OCR
text; section numbers refer to that manual.

---

## 1. Version timeline (as confirmed by sources)

| Version | Platform | Year | Notes |
|---|---|---|---|
| 1.0 - 1.62 | MS-DOS, text/CGA/EGA | 1988-1990 | First release, muLISP-based, successor to muMATH |
| 2.01 - 2.60 | MS-DOS | 1990-1994 | Same UI model, refined |
| 3.05 - 3.14 | MS-DOS | 1994-1996 | Last DOS line; runs in 512K RAM; still sold/supported into the late 1990s |
| 4.x | MS-DOS | ~1996-1998 | Still DOS-based (confirms the whole 1-4 line is DOS, not just 1-3) |
| 5.0x | Windows (native, 32-bit) | 1998-1999 | First Windows version, after/around the TI acquisition of Soft Warehouse (Aug 1999) |
| 6.0 - 6.1 | Windows 98/ME/2000/XP | 2000-2004 (6.1 revised through ~2004) | Final release; discontinued 29 June 2007 in favor of TI-Nspire CAS |

Correction versus the task brief's assumption: the task description mentions "v4/5/6" as
the Windows GUI line, but v4 is confirmed to still be a DOS release. The Windows GUI line
is properly v5 and v6 only.

---

## 2. The DOS text-mode UI

### 2.1 Overall screen anatomy

A DOS Derive screen (in its default "algebra window" state) is built from four
horizontal bands, top to bottom:

1. **Work area** - the numbered expression history (see 2.3).
2. **Message line** - a single line that reports what Derive is currently doing, or what
   it expects you to do next (e.g. `Enter option`, `A limit as x approaches infinity`,
   `Computing the slope of x^3 via a limit`). This line doubles as an annotated
   "breadcrumb" of the last operation performed, e.g. `Simp(#3)` meaning "this new
   expression is the simplification of expression #3" - the closest DOS-era analogue of
   an undo/provenance log.
3. **Command menu** ("soft-key" menu) - two lines, separated from the work area by a
   double horizontal rule. The word `COMMAND:` (or the context-specific title such as
   `PLOT:` inside a plot window) appears at the left margin, followed by the full list of
   top-level menu words wrapped over one or two lines. The **currently highlighted**
   option is shown in reverse video / a contrasting highlight box, and is moved with
   Tab/Space (forward) or Shift-Tab/Backspace (backward); typing the capitalized
   mnemonic letter jumps straight to that option and, for most options, invokes it. Every
   menu word has exactly one capitalized letter that serves as its keyboard shortcut (see
   2.2 below for the exact word list, which differs slightly by window type and by
   Derive version).
4. **Status line** - the bottom-most line, divided into left/center/right fields:
   left = editing mode annotation (`Ins` for insert mode, blank for overwrite) plus, at
   the very left edge, the operation annotation described above; center = `Free:100%`
   (percentage of the muLISP heap still available - deliberately only updated when
   Derive actually garbage-collects, not on every keystroke); right = the current
   window's type and product name, e.g. `Derive Algebra`, `Derive 2D-plot`,
   `Derive 3D-plot`.

Screenshots: `dos-v2.01-splash-screen-menu-bar.png`,
`dos-v3.14-algebra-window-limits-1/2/3-full.png`.

### 2.2 The soft-key / mnemonic menu - exact wording

Confirmed directly from a Derive 2.01 splash screen (WinWorld) and independently from
the Derive 3.14 User Manual OCR text. The **initial / default** top-level Algebra-window
command menu (v2.x-v3.x) reads, wrapped over two lines exactly as shown on screen:

```
COMMAND: Author Build Calculus Declare Expand Factor Help Jump soLve Manage
         Options Plot Quit Remove Simplify Transfer moVe Window approX
```

The 3.14 manual's own listing of the full command set adds one more item,
**Unremove**, between Remove/Simplify/Transfer and moVe/Window/approX:

```
COMMAND: Author Build Calculus Declare Expand Factor Help Jump soLve Manage
         Options Plot Quit Remove Simplify Transfer Unremove moVe Window approx
```

The likely explanation (not explicitly stated in the manual, but consistent with the
behavior of "undo remove" features of the era): **Unremove only appears in the menu
after at least one expression has been removed in the session** - i.e. it is a
conditionally-shown option, absent from the very first, pristine screen (which is why
the splash-screen capture doesn't show it) but present once there's something to
restore. A from-scratch remake should reproduce this conditional appearance rather than
hard-coding a fixed 19- or 20-word menu.

Mnemonic letters are capitalized inside the word to disambiguate clashes with an
earlier word starting with the same letter, e.g. so**L**ve (because **S**implify already
claims S), mo**V**e (because **V** isn't otherwise used... actually reserved because
**W**indow doesn't clash but moVe avoids clashing with nothing - historically it's
simply that "M" is taken by Manage), appro**X** (X is free). This is a genuine
"first-letter-unless-taken" mnemonic scheme, not arbitrary.

Different window types get **entirely different command menus** while keeping the same
visual chrome (same `COMMAND:`-style banner, same status-line layout):

- **Algebra window**: the 19/20-word list above.
- **2D-plot window**: `Algebra Center Delete Help Move Options Plot Quit Scale Ticks
  Window Zoom` (confirmed from a real screenshot) - an older variant seen elsewhere
  reads `Center Delete Help Move Options Plot Quit Range Scale Transfer Window Zoom`.
- **3D-plot window**: `Center Eye Focal Grids Hide Length Options Plot Quit Transfer`
  (confirmed from a real screenshot) - note the domain-specific verbs **Eye** (viewpoint),
  **Focal** (focal length/perspective), and **Hide** (hidden-line toggle) that only make
  sense for a 3D view.

This means the remake's menu system needs to be **contextual per window type**, not a
single static global menu - exactly like the original's window-designate model (see
2.4).

Screenshots: `dos-v2.01-splash-screen-menu-bar.png` (Algebra menu, pristine, no
Unremove), `dos-v2-algebra-and-2dplot-tiled-windows.png` and
`dos-2d-plot-window-circle-graphics-mode.gif` (2D-plot menu),
`dos-3d-2d-plot-split-windows-graphics-mode.gif` (3D-plot menu).

### 2.3 The numbered "algebra window" expression history

Every expression that is authored (typed) or produced by a command is appended to a
scrolling, permanently-numbered list inside the algebra window, formatted as:

```
#1:   <pretty-printed expression>
#2:   <pretty-printed expression>
...
```

Expression numbers are never reused within a window and are the universal way the UI
(and the user) refers back to prior results - e.g. typing `Calculus Differentiate` and
then referencing `#3` on the author line, or the automatic status-line annotation
`Simp(#3)` meaning "the newly-produced, currently-highlighted expression is expression
#3, simplified." The **currently selected/highlighted expression** is shown with an
inverted (light-on-dark, i.e. white-block) background rather than a colored box -
visible clearly in `dos-v2-algebra-window-selected-expression.png`, where expression #6
sits inside a solid white highlight rectangle while the rest of the screen is plain
black-background/white-text.

Two-line "before/after" pairs are common: an unsimplified input expression on one
numbered line, its simplified result on the next numbered line, e.g.:

```
#1:  lim  (x+h)^3 - x^3
     h->0  ----------
                h
#2:  3*x^2
```

The manual explicitly frames the design philosophy as "author your expression, then
tell Derive what to do with it" - matching a **transformation-log / worksheet** feel
rather than a REPL: every operation permanently appends a new numbered line rather than
mutating what came before.

### 2.4 Multi-window tiling model

Derive DOS supports multiple simultaneous windows of three types: **Algebra**,
**2D-plot**, and **3D-plot** (Section 2.13 of the manual, "Window Commands"). Key
mechanics, confirmed from the manual text:

- Windows can be **split** (`Window Split Horizontal` / `Window Split Vertical`),
  prompting for the row/column at which to split (default = an even half/half split).
  This is manual, explicit tiling - not automatic reflow.
- Windows can also be **overlaid** (`Window Open` stacks a new window on top of the
  active one; overlaid windows share the same window *number* and only the topmost is
  interactive) - i.e. a lightweight non-overlapping-numbering alternative to true
  overlapping windows, since DOS text mode can't do arbitrary z-ordered overlapping
  redraw cheaply.
- Each tile is outlined with a **drawn border** (line-drawing/extended-ASCII characters)
  and carries a **window number in its upper-left corner**; the active window's number
  is shown in reverse video/highlight.
- `Window Designate` changes an existing tile's *type* in place (Algebra / 2D-plot /
  3D-plot) via a selection-field prompt - so the remake should treat "window type" as a
  mutable property of a pane, not something fixed at creation.
- Window switching shortcuts: **F1** = next window, **Shift-F1** = previous window,
  **F2** = flip to next *overlaid* window, **Shift-F2** = flip to previous overlaid
  window.
- Rationale given in the manual for splitting *algebra* windows specifically: each split
  **forks the derivation history**, letting the user pursue two independent derivation
  paths from a common point, or freeze one important expression on screen while
  scrolling a second window.

Screenshot showing an actual tiled Algebra + 2D-plot pair, each with its own red window
border and its own distinct command menu, side by side:
`dos-v2-algebra-and-2dplot-tiled-windows.png`.

### 2.5 Text mode vs. graphics mode, and math typesetting

This is one of the more subtle and important facts for a faithful remake: **Derive DOS
runs its whole UI in either "Text" or "Graphics" display mode, and BOTH modes can show
math and plots** (Options Display command, Section 2.12) - it is not "text mode = no
plots, graphics mode = plots":

- **Text mode**: every plot pixel is literally the size of one text character cell (very
  coarse plots), but screen redraws are fast and character shapes can be crisper on some
  monitors. Fractions, roots, exponents, integral signs etc. are still "pretty-printed"
  using raised/lowered baseline positioning and box-drawing/extended-ASCII characters
  (built-up fractions with a horizontal rule made of dash/line characters, superscript
  exponents shifted up a text row, radicals approximated with available line-drawing
  glyphs). This is the style visible in all of the plain `dos-v3.14-algebra-window-*`
  and `dos-v2-*` screenshots in this set - pure monospace character-cell rendering, no
  antialiasing, but still clearly "typeset-looking" rather than linear ASCII (`SQRT(x)`
  is never shown literally; a real stacked fraction bar and raised exponent are used).
- **Graphics mode**: the same fractions/exponents/radicals/integral signs/Greek letters
  are drawn as actual bitmap glyphs from a custom font baked into Derive, giving properly
  curved radical signs, a real integral "∫", real Greek letters, etc., and plots become
  genuine pixel-resolution 2D/3D graphics (including hidden-line removal for 3D). The
  manual's own FAQ warns that on some adapters (AT&T 6300, Toshiba 3100) you must
  pre-load a `GRAFTABL.COM`/`GFTABLE.COM` "graphics table" TSR before starting Derive or
  these extended characters render as garbage - a nice period-accurate detail if the
  remake wants an in-universe Easter egg, but not otherwise relevant.
- Users pick **Mode** (Text/Graphics), **Resolution** (Medium 40-col / High 80-col),
  **Text size** (Large/Small - "Small" is half-height, EGA/VGA only, to fit more
  expressions on screen), **character Set** (Std/Extended ASCII), and **Adapter** (MDA,
  CGA, EGA, MCGA, VGA, Hercules, AT&T, T3100, 95LX, PCjr, Other) from one combined
  selection-field dialog (`Options Display`).
- Graphics-mode typeset examples (small crops from a 1990s HP-200LX palmtop-computer
  magazine review, showing the custom math font in isolation): built-up matrix with
  large square brackets (`dos-graphics-mode-matrix-determinant-typeset.gif`), a
  summation/limit pair with real Σ and lim-under-arrow (`dos-graphics-mode-sum-limit-
  typeset.gif`), a definite integral running to infinity with a real "∫...∞" glyph
  (`dos-graphics-mode-definite-integral-typeset.gif`), an indefinite integral with a
  genuine curved radical sign "√(1-x²)" (`dos-graphics-mode-indefinite-integral-
  radical.gif`), and an ODE solution mixing ERF, real square roots, and π
  (`dos-graphics-mode-ode-solution-typeset.gif`).
- Graphics-mode 3D plotting used **hidden-line removal** by default and rendered as a
  wireframe mesh with a caption showing the plotted expression underneath -
  `dos-3d-plot-hidden-line-surface-large.gif` is a good, if low-resolution, reference
  for the "look" of a DOS 3D plot (saddle-shaped surface, mesh grid, axis labels drawn in
  the same character font as the rest of the UI).

### 2.6 Colors

Derive supports up to 16 EGA/VGA colors (numbered 0-15), independently configurable via
two commands: `Options Color Work` (background/foreground used for the algebra,
2D-plot, and 3D-plot **work areas**) and `Options Color Menu` (background/foreground for
the **menu and border** areas) - i.e. the chrome and the content area are deliberately
themeable as two separate zones, not a single global palette choice. On monochrome
monitors, distinct color numbers instead map to distinct brightness levels.

In practice, screenshots in the wild show real variation:

- The clearest full-color reference (`dos-v2.01-splash-screen-menu-bar.png`, from
  WinWorld, presumably close to factory defaults) shows: **black background** overall;
  plain **white** text for the copyright/splash message; a **magenta/pink double
  horizontal rule** separating the menu from the work area; **yellow** for the
  `COMMAND:` label and the menu option text; the currently-highlighted menu option in an
  **orange/red inverse-highlight** box; **red** for the message-line text (`Enter
  option`); and **green** for the bottom-right product/window-type tag (`Derive
  Algebra`).
- Several other captures in this set show an all-**green-on-black** monochrome look
  (e.g. `dos-v3.14-algebra-window-limits-*.png`, `dos-v2-algebra-window-selected-
  expression.png`) - consistent with either a Hercules/monochrome adapter selection, a
  green-phosphor terminal emulator, or simply a screenshot tool that only captured
  luminance.
- The user brief's recollection of a "blue text-mode screen" was not corroborated by any
  primary source found (the manual text never mentions a default blue background, and
  no screenshot shows one); it's possible this conflates Derive with other blue-background
  DOS-era tools (WordPerfect, Turbo Pascal, etc.), or reflects a particular
  adapter/terminal configuration not captured in the sources found here. Recommend
  treating "black background, yellow/white/green/red/magenta 16-color EGA palette,
  fully user-remappable via Options Color" as the better-evidenced default, while still
  supporting a configurable palette in the remake so a "blue theme" preset is trivial to
  add if desired.

### 2.7 Function keys / keyboard shortcuts (from the official manual's own "Function Key
Commands" appendix, Section verbatim)

**Window switching** (Section 2.13):
- `F1` - switch to the next window
- `Shift-F1` - switch to the previous window
- `F2` - flip to the next overlaid window
- `Shift-F2` - flip to the previous overlaid window

**Line editing on the author line** (Section 3.3):
- `F1` - get help on line-editing commands or math functions while editing
- `F3` - copy the highlighted expression onto the author line
- `F4` - copy the highlighted expression onto the author line, enclosed in parentheses
- `F6` - toggle arrow-key behavior between *line-edit mode* (`Lin` shown on the status
  line) and *subexpression mode* (nothing shown) - i.e. whether arrow keys move the text
  cursor character-by-character, or move a selection cursor sub-expression-by-
  sub-expression through a previously-entered formula

**Display-mode switching** (Section 2.12):
- `F5` - toggle between the two most recent display modes (e.g. flip Text <-> Graphics)

**Plot-window commands** (Chapter 5):
- `F3` - toggle "trace mode" on/off (2D-plot window only) - moves a tracking cross along
  the curve while showing live coordinates on the status line
- `F7` / `Shift-F7` - magnify the plot vertically / horizontally
- `F8` / `Shift-F8` - shrink the plot vertically / horizontally
- `F9` / `Shift-F9` - wait, per manual: `F9` magnifies in both directions at once;
  `Shift-F9` sends a graphics image of the current window to the printer (dual-purpose
  key, context-independent binding overlap in the original product)
- `F10` - shrink the plot in both directions at once; `Shift-F10` sends a graphics image
  of the *entire* Derive screen (all windows + menu + status line) to the printer

**Printing/saving screen images** (Section 2.11):
- `Shift-F9` - print a graphics image of the current algebra/plot window
- `Shift-F10` - print a graphics image of the entire Derive screen
- `Ctrl-F9` - save a graphics image of the current window to a TIFF file

Other notable non-function-key bindings mentioned in the manual: `H` from the very
first screen invokes Help without going through the menu; `Ctrl-Enter`/`Ctrl-J` on the
author line authors *and* immediately simplifies an expression in one step (skipping the
separate unsimplified line); typing an expression followed by `=` and Enter shows both
the input and its simplified form together on one combined line.

---

## 3. The Windows GUI (v5 / v6)

### 3.1 Overall window structure

The Windows version replaces the single tiled text-mode screen with a conventional MDI
(multiple-document-interface)-style main window:

- A **main frame** titled `Derive 6` (or `Derive 5`) with a standard Windows menu bar:
  `File Edit Insert Author Simplify Solve Calculus Options Window Help` - notice
  **Author**, **Simplify**, **Solve**, and **Calculus** survive as *top-level menu
  names* directly descended from the DOS mnemonic menu (Author, Simplify, soLve,
  Calculus were all DOS command words), while Build/Declare/Expand/Factor/Manage/
  Transfer/moVe/approX have been folded into submenus, toolbar buttons, or dialog
  options instead of remaining top-level words.
- A **toolbar** row of icon buttons directly under the menu bar: new/open/save/print,
  cut/copy/paste, then a cluster of math-specific icons (square-root/simplify "=" style
  icon, an "approximate ≈" icon, integral "∫", summation "Σ", product "Π", plus
  vector/matrix template buttons) - the manual math functions are literally "on a mouse
  click from the Menu and Toolbar" per the vendor's own description.
- Inside the frame, one or more child windows: an **Algebra window** (titled e.g.
  `Algebra 1 - [book_cone.dfw]`, `.dfw` being the native "Derive for Windows" worksheet
  file extension), a **2D-plot window**, and/or a **3D-plot window**, freely tiled or
  overlapped exactly like any other 1990s MDI app (Word, Excel of the era). Each child
  window has its own title bar, its own minimize/maximize/close controls, and (for
  plot windows) its own small toolbar of view controls (zoom, cross-hair readout,
  rotate arrows for 3D).
- A **status bar** at the bottom of the main frame shows contextual info (e.g. cursor
  cross-hair coordinates `Cross: 1.944444, 1.313636`, current plot `Center:`/`Scale:`
  values, or a `Step(#3)` / elapsed-time readout during a long computation) - a direct,
  modernized descendant of the DOS status/message line duo.
- Classic Windows 98-era chrome throughout: gray 3D-beveled toolbar buttons, a light
  gray dialog background, small non-antialiased system font for menus, a visible Windows
  taskbar with a **Start** button in several captures (confirming these were taken on
  Windows 95/98).

Best single reference for the whole ensemble (menu bar + toolbar + a Solve dialog box +
tiled Algebra and 2D-plot child windows + Windows taskbar, all in one screenshot):
`win6-solve-dialog-algebra-2dplot-windows.gif`.

### 3.2 The Algebra window / worksheet model

Functionally the same numbered `#1:`, `#2:`, ... history as DOS, but now:

- It's a **rich worksheet**, not just a scrolling command log: free-form colored text
  annotations, headings, and instructions can be interleaved between numbered math
  lines (e.g. a red-bold instruction line, then black body text, then a `#1:` math
  line) - see `win6-worksheet-min-area-problem.gif` and `win6-worksheet-cone-integral-
  example.gif`, where explanatory paragraphs, a plotted graph, and derivation steps all
  coexist inside one continuous scrollable document.
- **2D and 3D plots can be embedded directly inline** in the algebra/worksheet window
  (not only in a separate tiled plot window) - see `win6-embedded-2dplot-in-worksheet.png`,
  where a small plot sits inline between two math lines, entirely within the Algebra
  child window.
- A distinct light-blue "task" callout box style is used at least once in vendor
  material to highlight an instruction/exercise prompt inline in the worksheet
  (`win6-task-highlight-log-rules-worksheet.png`) - i.e. a lightweight rich-text
  highlighting convention for "here is the question", separate from plain body text.
- Worksheets can be saved as `.dfw` files, and (per the v6.1 feature list) written out in
  Rich Text Format, embedding OLE objects (e.g. a MathType-authored inline formula) and
  raster plot images (TIFF/JPEG/BMP export supported).
- Derive 5 specifically (an older toolbar style than 6) is documented showing
  **dockable/floating toolbars** for entering Greek letters and other math symbols by
  point-and-click, plus a separate floating "plot tracing" mini-toolbar -
  `win5-floating-toolbars-greek-symbols.png` is a good reference for this feature,
  showing three separate floating palettes (Greek Symbols, Plot Tracing, Math Symbols)
  hovering over the algebra/plot windows.

### 3.3 Plot windows

- **2D-plot windows**: Cartesian axes with light dotted gridlines, colored curves (each
  plotted expression gets a distinct color, labeled directly on the plot next to the
  curve, e.g. `y = sin(x)` in red beside its curve and `y = sin(mx)` in green beside
  its own), a small cross-hair/status readout, and (new in v6) an optional floating
  **slider-bar control** (a tiny separate titled window with a horizontal slider and a
  numeric readout, e.g. `m = 4.00`, range `1` to `10`) for live-dragging a parameter and
  watching the curve redraw in real time. See `win6-2d-plot-with-slider-bar.gif` and
  `win6-algebra-2dplot-slider-transform-demo.gif` (the latter shows three stacked
  sliders for `a`, `b`, `c` simultaneously, each in its own small titled floating
  window).
- **3D-plot windows**: shaded (not just wireframe) surfaces by default, with solid fill
  colors per surface (e.g. blue top / red underside to indicate orientation), a
  bounding wireframe box with tick-labeled axes, and mouse- or arrow-button-driven
  real-time rotation; multiple 3D-plot child windows can be open side by side showing
  the same surface from different angles (`win6-3d-plot-dual-view-rotate.gif`). A more
  elaborate example shows a rainbow/heat-colored mesh surface for a bumpier function
  (`win6-3d-surface-colored-mesh.png`).
- Both plot types can be annotated (descriptive text, dimension callouts like
  `Height of cone = 10`, `Base radius = 4`) and exported as raster images, and both
  support embedding directly into the worksheet as described above.

### 3.4 Dialog boxes

The single clearest captured example is a **"Solve simultaneous equations" dialog**
(`win6-solve-dialog-algebra-2dplot-windows.gif`): a modal-looking child panel titled
"Solve 2 equation(s)" containing one text-entry row per equation, a "Solution
Variables" multi-select list box, and OK / Solve / Cancel buttons - opened while the
Algebra window above it already shows the red instructional heading "Solve simultaneous
equations algebraically and graphically" and the resulting `SOLVE(...)` expression.
This matches vendor documentation describing that **every algebra command
(Factor, Expand, Differentiate, Integrate, Solve, ...) pops a dialog box with several
options**, and that sensible defaults usually produce the expected answer with no
further input needed. Per the same vendor text, the full roster of "one mouse click
from the Menu and Toolbar" commands is:

> Author Expression, Author Vector, Author Matrix, Simplify, Expand, Factor,
> Approximate, Substitute, Solve Expression (algebraically or numerically), Solve
> System (algebraically or numerically), Differentiate, Integrate (definite or
> indefinite), Limit, Sum, Product, Taylor Series.

No standalone screenshots of the *Author Expression*, *Simplify* (precision-digits), or
*Calculate Integral* dialog boxes specifically were found in the sources searched (TI's
own Derive 5 "Introduction" PDF was retrieved but its screenshots were embedded as
non-extractable/encrypted image streams); the Solve dialog above is the best
first-hand evidence of the dialog style (plain gray Windows 95/98 dialog, simple
single-line text fields, standard OK/Cancel button row, no wizard-like multi-step flow).

### 3.5 Branding / packaging

The retail Derive 6 box (`win6-retail-box-ti-branding.png`) is a red box under the TI
(Texas Instruments) logo, titled "Derive 6 - Advanced Mathematics for Your PC", with
cover art of a torus/ring shape and a small windsurfing-board photo illustration -
useful purely as a branding/color reference (bold red, white, and TI-blue) rather than
as a UI reference. The DOS-era 3.5" floppy disk label
(`dos-v1.62-floppy-disk-label.jpg`) and the printed manual cover
(`dos-v3-manual-cover-branding.jpg`) show the earlier Soft Warehouse "handcrafted
software for the mind" identity: a bold arrow-chevron wordmark "DERIVE>>" in dark
red/maroon on a metallic-gold disk label, and a charcoal/red diagonal-striped design on
the manual, with the Soft Warehouse "oscilloscope-in-a-TV-box" logo. Neither is a
program screenshot, but both are useful for capturing period-accurate brand identity if
the remake wants nostalgic packaging/splash-screen art.

---

## 4. DOS vs. Windows: differences in feel

- **Menu depth and vocabulary.** DOS exposes a flat, always-visible, fully mnemonic
  20-word command bar that is *itself* the primary navigation UI (no icons, no mouse
  needed at all - the whole product is usable from the keyboard alone, including
  choosing plot ranges, colors, and printer options through the same Tab/Space-driven
  selection-field mechanism). Windows collapses most of that vocabulary into a
  standard pull-down `File Edit Insert Author Simplify Solve Calculus Options Window
  Help` bar plus a toolbar of icons, trading the "everything is one glance away" DOS
  bar for the far larger command surface a full Windows app can support (at the cost,
  per one contemporary review quoted in the research, that "most functions aren't
  accessible via the front menu" and require hunting through Help).
- **Single screen vs. MDI document model.** DOS Derive's "windows" are still just tiled
  regions of one 80x25 (or 40x25) text/graphics screen with hand-managed splits;
  Windows Derive is a real MDI app with freely overlapping, resizable, minimizable
  child windows and a saved `.dfw` document format - a proper "worksheet" you save,
  reopen, and print, rather than a live session tied to one terminal screen.
  Corollary: DOS has no true document format worth speaking of (state is saved via
  Transfer commands to `.mth`/similar rather than a rich mixed-content file); Windows
  worksheets are rich, mixed-media, RTF-based documents that can be handed to someone
  else.
  a
- **Typesetting fidelity.** Both eras render "real" built-up fractions, superscript
  exponents, and (in DOS graphics mode, and always in Windows) true radical/integral/
  Greek glyphs rather than linear ASCII notation - this core "looks like a textbook,
  not like a REPL" identity is consistent across both eras and should be treated as a
  non-negotiable identity trait of any remake. The main difference is resolution/
  smoothness (character-cell/bitmap-font DOS vs. presumably TrueType-quality, per the
  v6.1 feature list's mention of a "fully scaleable Derive Unicode font", Windows) and
  the addition, in Windows, of inline colored annotation text and embedded plots living
  in the *same* scrollable document as the math.
  a
- **Interactivity of plots.** DOS plotting is essentially "compute once, view, maybe
  zoom/trace with function keys, replot if you change something" - there is no
  live-dragging of a parameter. Windows v6 adds genuine **slider-bar widgets** that
  redraw the plot live as you drag, plus mouse-driven live 3D rotation - a real
  interactivity upgrade, not just a visual reskin.
  a
- **Color depth and identity.** DOS's palette is a hard 16-color EGA/VGA (or
  monochrome) affair, fully remappable via `Options Color Work`/`Options Color Menu`,
  and real-world screenshots show it was actually used with different color schemes on
  different machines (bright multi-color menu vs. flat green monochrome). Windows uses
  whatever the desktop theme/graphics card provides, with the app itself choosing
  fixed, tasteful colors per plotted curve/surface (red/green/blue) rather than
  exposing a global "menu color" vs. "work color" picker the way DOS did.
  a
- **Keyboard-only vs. mouse-first.** DOS is unapologetically **keyboard-only**
  (mnemonic letters, Tab/Space, function keys for absolutely everything including
  printing and zooming); Windows is **mouse-first** (toolbar icons, dialogs, drag-to-
  select subexpressions, drag sliders, drag-to-rotate 3D) while still keeping keyboard
  accelerators as an option. A remake aiming to "feel like both eras" either needs a
  genuine dual-mode input scheme, or should pick one era's input philosophy as primary
  and treat the other as a nostalgia skin.

---

## 5. Image manifest

All files are stored in `/home/hniksic/work/derive/artifacts/screenshots/`.

| File | Source URL | Description |
|---|---|---|
| `dos-v2.01-splash-screen-menu-bar.png` | https://winworldpc.com/product/derive/2x (image: `/res/img/screenshots/19bc0231...a19.png`) | Derive 2.01 startup/splash screen; the clearest full-color capture of the default 16-color palette and the complete pristine (no "Unremove") Algebra-window command menu. |
| `dos-v3-manual-cover-branding.jpg` | https://winworldpc.com/product/derive/3x (image: `/res/img/screenshots/60fa8bdc...a6.jpg`) | Scan of the printed "DERIVE Version 3 User Manual" cover; branding/identity reference only, not a program screenshot. |
| `dos-v1.62-floppy-disk-label.jpg` | https://archive.org/download/derivecas162/derive.jpg | Photo of the original 3.5" floppy disk label for Derive v1.62 (1988), Soft Warehouse branding/logo reference. |
| `dos-v3.14-algebra-window-limits-1.png` | https://archive.org/download/derive314cas/screenshot_00.png | DOSBox capture of Derive 3.14 Algebra window mid-derivation (expressions #1-#2), showing a built-up-fraction limit and its simplification; message line reads "Computing the slope of x^3 via a limit". |
| `dos-v3.14-algebra-window-limits-2.png` | https://archive.org/download/derive314cas/screenshot_02.png | Same session, expressions #1-#4, showing a `SIN(x)/x` limit example; message line "A famous limit of an indeterminate form 0/0". |
| `dos-v3.14-algebra-window-limits-3-full.png` | https://archive.org/download/derive314cas/screenshot_05.png | Same session, final state with expressions #1-#6 (adds a `lim x^(1/x)` example); message line "A limit as x approaches infinity". |
| `dos-v2-algebra-window-selected-expression.png` | https://darrengoossens.wordpress.com/2024/06/10/derive-calculus-and-algebra-software/ | Algebra window with expression #6 shown selected/highlighted via a solid white inverse-video block, illustrating DOS-era selection styling; status line shows `Int(3,x)`. |
| `dos-v2-algebra-and-2dplot-tiled-windows.png` | https://darrengoossens.wordpress.com/2024/06/10/derive-calculus-and-algebra-software/ | Two tiled, red-bordered, numbered windows side by side: an Algebra window (left, showing a series-of-fractions expression) and a 2D-plot window (right, showing the corresponding curves); demonstrates the different command-menu wording per window type. |
| `dos-v2.60-dosbox-emulator-view.png` | https://darrengoossens.wordpress.com/2024/06/10/derive-calculus-and-algebra-software/ | Derive 2.60 running inside a visible DOSBox 0.74-3 emulator window frame; useful as a "how people experience DOS Derive today" reference. |
| `dos-graphics-mode-matrix-determinant-typeset.gif` | https://www.palmtoppaper.com/ptphtml/48/48c0000c.htm | Small graphics-mode crop showing a symbolic 3x3 matrix in large square brackets and its computed determinant; illustrates bracket/matrix typesetting. |
| `dos-graphics-mode-sum-limit-typeset.gif` | https://www.palmtoppaper.com/ptphtml/48/48c0000c.htm | Graphics-mode crop showing a real Sigma summation formula and a `lim` expression, both with proper stacked bounds. |
| `dos-graphics-mode-definite-integral-typeset.gif` | https://www.palmtoppaper.com/ptphtml/48/48c0000c.htm | Graphics-mode crop of a definite integral with an infinite upper bound, showing the true integral-sign glyph. |
| `dos-graphics-mode-indefinite-integral-radical.gif` | https://www.palmtoppaper.com/ptphtml/48/48c0000c.htm | Graphics-mode crop showing a genuine curved radical (square-root) sign in an indefinite-integral result, `x*ASIN(x)/sqrt(1-x^2)`. |
| `dos-graphics-mode-ode-solution-typeset.gif` | https://www.palmtoppaper.com/ptphtml/48/48c0000c.htm | Graphics-mode crop of a 2nd-order ODE and its closed-form solution mixing ERF, square roots, and trig terms - a dense typesetting stress-test example. |
| `dos-2d-plot-window-circle-graphics-mode.gif` | https://www.palmtoppaper.com/ptphtml/48/48c0000c.htm | Graphics-mode 2D-plot window (a circle) with its own command menu visible at the bottom (`Center Delete Help Move Options Plot Quit Range Scale Transfer Window Zoom` variant). |
| `dos-3d-2d-plot-split-windows-graphics-mode.gif` | https://www.palmtoppaper.com/ptphtml/48/48c0000c.htm | Split screen: a 3D hidden-line dome plot (left) and a 2D ellipse plot (right); the visible command menu (`Center Eye Focal Grids Hide Length Options Plot Quit Transfer`) is the 3D-plot-window-specific menu. |
| `dos-3d-plot-hidden-line-surface-large.gif` | https://www.palmtoppaper.com/ptphtml/48/48c0000c.htm | Larger single 3D hidden-line wireframe surface plot with a captioned formula underneath, in the same character font as the rest of the UI. |
| `win5-floating-toolbars-greek-symbols.png` | https://www.chartwellyorke.com/derive/Derive6.pdf (p.1, embedded image) | Derive **5** (not 6) main window showing three separate floating/dockable toolbars - Greek Symbols, Plot Tracing, and Math Symbols - plus a worksheet with an ellipse/line intersection problem; Windows 98 taskbar visible. |
| `win6-solve-dialog-algebra-2dplot-windows.gif` | https://www.chartwellyorke.com/derive/images/in_action/bowers560/simultaneousequations.gif | Full Derive 6 main frame: menu bar, toolbar, a tiled 2D-plot window and Algebra window, and an open modal "Solve 2 equation(s)" dialog box with per-equation text fields and a solution-variables list; Windows taskbar visible. Single best "whole GUI at a glance" reference. |
| `win6-2d-plot-with-slider-bar.gif` | https://www.chartwellyorke.com/derive/images/in_action/bowers560/2D_slider.gif | 2D-plot window with two labeled curves (`y=sin(x)`, `y=sin(mx)`) and a small floating slider-bar control window (`m = 4.00`, range 1-10) for live parameter dragging. |
| `win6-3d-plot-dual-view-rotate.gif` | https://www.chartwellyorke.com/derive/images/in_action/bowers560/3D_rotate_new.gif | Two independent 3D-plot child windows showing the same saddle surface from two different rotations, plus an Algebra window below with the defining equation; illustrates mouse/arrow-driven 3D rotation. |
| `win6-worksheet-cone-integral-example.gif` | https://www.chartwellyorke.com/derive/images/in_action/bowers560/book_cone.gif | A full worksheet page: instructional text banner, an embedded labeled 3D cone diagram, and a step-by-step integral derivation of the cone's volume below it. |
| `win6-worksheet-min-area-problem.gif` | https://www.chartwellyorke.com/derive/images/in_action/bowers560/book_minarea.gif | A full worksheet page mixing explanatory paragraphs, an embedded small 2D plot, and a derivative/solve derivation, demonstrating the "book-like" worksheet layout style. |
| `win6-algebra-2dplot-slider-transform-demo.gif` | https://www.chartwellyorke.com/derive/images/in_action/550/sliders.gif | Tiled Algebra + 2D-plot windows with three stacked floating slider controls (`a`, `b`, `c`) for live parabola-transformation exploration; menu bar and toolbar visible; Windows taskbar visible. |
| `win6-integration-steps-worksheet.gif` | https://www.chartwellyorke.com/derive/images/in_action/550/integrationsteps.gif | Pure worksheet-content crop (no window chrome) showing Derive 6's "Display Steps" feature: an integration by parts worked out rule-by-rule. |
| `win6-differentiation-steps-worksheet.gif` | https://www.chartwellyorke.com/derive/images/in_action/550/steps2.gif | Pure worksheet-content crop showing step-by-step differentiation of `x^2 * e^x` via the product rule, each rule shown symbolically before being applied. |
| `win6-quotient-rule-steps-typeset.gif` | https://www.chartwellyorke.com/derive/images/in_action/bowers560/steps_diffquot.gif | Worksheet-content crop showing the quotient-rule differentiation of `SIN(x)/x`, rule-then-application, in the blue "rule" / black "result" two-color step convention. |
| `win6-integration-rules-steps-typeset.gif` | https://www.chartwellyorke.com/derive/images/in_action/bowers560/steps_int.gif | Worksheet-content crop showing basic integration rules (sum rule, power rule, constant rule) applied step by step to a simple polynomial. |
| `win6-retail-box-ti-branding.png` | https://www.chartwellyorke.com/derive/Derive6.pdf (p.2, embedded image) | Photo of the retail "Derive 6 - Advanced Mathematics for Your PC" box under the Texas Instruments logo; branding/packaging reference only. |
| `win6-task-highlight-log-rules-worksheet.png` | https://www.chartwellyorke.com/derive/Derive6.pdf (p.2, embedded image) | Worksheet crop showing a light-blue highlighted "Task:" callout box embedded among plain derivation lines - the rich-text instruction-highlighting convention. |
| `win6-embedded-2dplot-in-worksheet.png` | https://www.chartwellyorke.com/derive/Derive6.pdf (p.1, embedded image) | Worksheet crop showing a 2D plot embedded inline directly between math lines inside the Algebra/worksheet window itself (not a separate tiled plot window). |
| `win6-3d-surface-colored-mesh.png` | https://www.chartwellyorke.com/derive/Derive6.pdf (p.4, embedded image) | A bumpy 3D surface rendered with a rainbow/heat-mapped mesh coloring rather than the flat red/blue two-tone style seen elsewhere, showing an alternative 3D shading option. |

---

## 6. Sources consulted (including ones that yielded no usable images)

- https://en.wikipedia.org/wiki/Derive_(computer_algebra_system) - version history, no screenshots embedded in the article text as retrieved.
- https://winworldpc.com/product/derive/2x and /3x - DOS-only; source of the splash-screen and manual-cover images; no Windows-version entries exist on WinWorld for Derive.
- https://archive.org/details/derive314cas and https://archive.org/details/derivecas162 - in-browser-emulator software packages with bundled screenshot sets; source of most DOS Algebra-window images and the floppy-disk-label photo.
- https://archive.org/details/derivecas314manual (`Derive Version 3.14 User Manual`) - full OCR text used as the primary authoritative source for Section 2 of these notes (menu wording, function keys, window commands, display modes/colors).
- https://darrengoossens.wordpress.com/2024/06/10/derive-calculus-and-algebra-software/ - retro-computing blog post with original DOS screenshots (tiled windows, selection highlighting, DOSBox chrome).
- https://www.palmtoppaper.com/ptphtml/48/48c0000c.htm - a 1990s HP-200LX palmtop-computer magazine article reviewing Derive, with small graphics-mode typesetting crops and both 2D- and 3D-plot-window examples with visible menus.
- https://www.chartwellyorke.com/derive/deriveaction.html and https://www.chartwellyorke.com/derive/Derive6.pdf - a UK educational software reseller's marketing gallery and brochure; by far the richest source of genuine Windows GUI (v5 and v6) screenshots, including the only captured dialog box and the only floating-toolbar (v5) example.
- https://www.scientific-computing.com/feature/derive-6-far-too-good-just-students - a contemporary review with useful descriptive text about the Windows UI (confirmed the "most functions aren't accessible via the front menu" critique, the four-line optional input window, and the collapsible-tree help redesign) but its own embedded screenshot resolved to a dead internal file:// path, so no image was recoverable from it.
- https://education.ti.com/html/eguides/discontinued/computer-software/EN/Derive-5-Introduction.pdf - official TI "Introduction to Derive 5" guide; downloaded, but its figures are embedded as non-extractable encrypted image streams, so no screenshots could be pulled from it directly (text describes the same GUI elements documented above).
- https://waluigibsod.github.io/derive6.1-online-help/ - a fan-hosted mirror of the Derive 6.1 built-in help system; text-only (no images) in the pages checked, but confirmed menu names (Author > Variable Value / Function Definition / Variable Domain, Insert > OLE Object / Slider Bar).
- WinWorld's product search for vendor "Soft Warehouse, Inc." confirms no Windows-era Derive release is archived there.
- Derive.en.uptodown.com yielded five screenshots but all were tiny (~100px-tall) thumbnails duplicating UI already captured at much higher quality from Chartwell-Yorke; not included in the manifest.
- filehippo.com and derive.en.softonic.com download pages returned no usable screenshots (page failed to render / no images present in fetched content).
