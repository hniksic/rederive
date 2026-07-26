# Rederive: Vision

Rederive is a fresh implementation of **Derive**, the computer algebra system
created by Soft Warehouse, Inc. (Honolulu) and sold from 1988 to 2007, from
1999 under Texas Instruments. The goal is to capture Derive's look, feel, and
mathematical coverage, not to clone its code, file formats, or bugs.

Version 1 is a terminal application recreating DOS-era Derive (v1-v3,
1988-1996): the clean numbered worksheet, the mnemonic single-letter menus,
textbook-quality typesetting, and the "author it, then tell it what to do"
workflow.

This document says what Rederive is for and what it should feel like. It does
not specify behavior; that belongs in the specs that describe each part of the
system.

## Where Derive came from

Derive descended from muMATH, the first CAS for personal computers, and was a
ground-up menu-driven rewrite designed to feel like an approachable tool rather
than a programming language. It became a European educational bestseller while
running on a 286 with 512 KB of RAM, and its reputation rested on being
inviting where Mathematica and Maple were forbidding. Its most consistent
criticism was that its deeper functionality was hard to discover from the menus.

Three lessons carry over: chase approachability rather than scope, treat the
tiny footprint as a values statement, and fix the discoverability problem.

Full history, with sources, is in `artifacts/history/history-and-timeline.md`.

## Audience

The primary audience is **students and curious kids who should come away amazed
by what math can do**. Derive was a classroom tool, not a research tool, and the
remake keeps that position. **Engineers** are a real secondary audience: quick,
trustworthy symbolic and numeric answers, and plotting. Research mathematicians
who need Mathematica- or Maple-class depth are explicitly not the target.

## Design principles

1. **Worksheet, not REPL.** Expressions are authored, then explicitly told what
   to do. Nothing evaluates just because it was typed. Every result becomes a
   new, permanently numbered entry that later expressions can reference. This
   "notebook of labeled steps" model is Derive's most identity-defining trait.
2. **Looks like a textbook.** Built-up fractions, raised exponents, real
   radical, integral, sum and Greek glyphs; never linear ASCII in rendered
   output. Non-negotiable for "feels like Derive."
3. **Low intimidation, short learning curve.** Menu-driven operation with
   sensible defaults, so a first-time user gets a right answer without reading a
   manual. Typed function syntax is a power-user fallback, never a requirement.
4. **Exact-first arithmetic.** Prefer exact rational and symbolic results;
   numeric approximation is an explicit action or mode, never the default.
5. **Don't guess.** Simplifications that depend on a variable's domain fire only
   when an explicit declaration justifies them. Favor a visibly unevaluated
   "stuck" expression over a silently wrong one.
6. **Small and fast.** Derive's appeal came partly from running on hardware too
   weak for its competitors. Stay light and responsive; bloat is a regression
   against the spirit of the product.
7. **Discoverable.** Do better than the original at surfacing what the system
   can do - searchable palette, autocomplete, examples - without cluttering the
   primary menu.
8. **Engine decoupled from UI.** The math engine is usable through a narrow,
   rendering-agnostic interface, and returns structured results rather than
   preformatted text. This is what makes the terminal front end replaceable
   later without rewriting the engine.
9. **Design for wonder, not just correctness.** A beautiful plot, a step-by-step
   simplification, or an instant answer to "what if?" matters as much as
   mathematical depth. Favor an inviting first five minutes over a comprehensive
   manual.

## Goals

- A terminal UI that recreates DOS Derive's interaction model: pane types,
  mnemonic menus, numbered expression history, and typesetting style, using the
  original's menu wording and keybindings.
- A math engine, built and exposed independently of the UI, covering symbolic
  algebra, calculus, linear algebra, plotting, and user-defined functions - at
  least what Derive could do, presented the way Derive presented it.
- A plain-text worksheet format for saving, reopening, and sharing work,
  designed fresh rather than reverse-engineered.
- Derive-flavored aesthetics as homage, adapted to a modern terminal rather than
  pixel-cloned from a CRT.

## Non-goals

- **Compatibility of any kind with the original.** Not its file formats, not its
  internal architecture, not its bugs. The archived originals are inspiration
  for syntax style, naming, and structure only, and the remake is free to do
  better where Derive had known warts - or to be narrower where that is a
  conscious choice.
- **Research-grade breadth.** PDEs, tensor algebra, linear programming and the
  like stay out of the core; they were unsupported add-ons in real Derive too.
- **A checked units type system.** As in the original, units are ordinary
  variables holding numeric magnitudes.
- **DOS-era hardware constraints.** The DOS *aesthetic* is a core goal; the
  video-adapter selection, printer output and 40-column mode that shaped it are
  not.
- **A graphical front end, in v1.** The Windows-era look is a plausible future
  front end, enabled by the engine/UI boundary, but not built now.

## What success looks like

- **A 1990s Derive user** should, within minutes, recognize the numbered
  worksheet, the mnemonic menu, the author-then-transform workflow, the
  typesetting, and the separate plot panes. They should not need to know the
  engine is nothing like the original.
- **A student with no memory of the original** should find the first few minutes
  inviting rather than intimidating, and hit at least one genuine "whoa, it did
  that?" moment early.

## Research artifacts

Design decisions are backed by material under `artifacts/`: the Derive 3.14 User
Manual and full transcript, TI's mirrored Derive 6.1 online help, real
`.MTH`/`.DMO`/`.DFW` files from original installations, screenshots of both
eras, and period reviews and Usenet commentary. Each directory has a notes file
summarizing what it holds. Consult the artifacts before assuming anything about
how the original looked or behaved.
