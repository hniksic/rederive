# Rederive

A modern remake of the DOS-era *Derive* computer algebra system.

Rederive recreates DOS Derive as a terminal application: the numbered
worksheet, the mnemonic single-letter menus, built-up textbook typesetting, and
the workflow where an expression is authored first and told what to do second.
Nothing evaluates because it was typed. It is a fresh implementation, and
copies the original's look, wording and mathematical coverage rather than its
code, its file formats or its bugs.

## What it does

The Algebra menu carries the original's twenty commands, and all but one of
them work:

- `Author` and `Build` enter an expression, by typing it or by assembling it
  from operators a menu offers.
- `Simplify`, `approX`, `Expand`, `Factor` and `soLve` are the engine: exact
  results by default, approximation as an explicit act.
- `Calculus` differentiates, integrates, takes limits, products, sums, Taylor
  series and vectors.
- `Declare` says what a variable is - its domain or the interval it lies in -
  and defines functions, matrices and vectors.
- `Manage` annotates and renumbers expressions, sets the ordering of variables,
  substitutes, and holds the Branch, Exponential, Logarithm and Trigonometry
  transformation settings.
- `Options` sets Color, Input, Mute, Notation, Output, Precision and Radix.
- `Transfer` loads, merges and clears math, data, utility, state and demo
  files, saves the worksheet or the settings, and writes an expression out as
  C, Python, Rust or Julia source.
- `Window` splits, opens, closes and moves among panes, each holding a whole
  session of its own.
- `Jump`, `Remove`, `Unremove`, `moVe`, `Help` and `Quit` move around the
  worksheet and off it.

## Running

```sh
uv run rederive
```

Files named on the command line are read before the first frame, as the
original reads them. An extension says what a file is - `.mth` a math file,
`.dat` a data file, `.dmo` a demonstration - and a name given without one is
looked for under each in turn, so `uv run rederive arith` starts on
`arith.dmo`. A switch overrides that for the names after it, which is the only
way to say that a math file is to be read as a utility library:

```sh
uv run rederive -u number -m plot2d -d algebra
```

`-m` math, `-u` utility, `-t` data, `-d` demonstration. The original spelled
these after the name and behind a slash (`NUMBER/U`); a slash cannot mean that
here.

## Testing

```sh
uv run pytest
```

Simplifying the whole corpus - every shipped utility file, parser case and demo
script - takes two minutes where the rest of the suite takes five seconds, so
it is opt-in:

```sh
uv run pytest -m slow
```

Only the engine can regress it; a change to the UI or the session cannot move
it. Both runs spread across every core, and `-n0` turns that off, which a
single failing test is easier to read under.
