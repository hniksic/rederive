# Rederive

A modern remake of the DOS-era *Derive* computer algebra system.

This is milestone 1: a Textual TUI shell that looks and navigates like DOS
Derive but performs no mathematics at all. Expressions are parsed, validated
and typeset the way the original typesets them; nothing is ever simplified. It
exists so the look and feel - menu highlighting, Tab cycling, mnemonic letters,
expression input, and arrow-key movement across expressions and subexpressions
- can be evaluated before any engine work starts.

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

## Layout

- `src/rederive/model/` - pure Python, no Textual: the expression tree, the
  operators `Build` hangs one together from, the system control settings, the
  file formats a `Transfer` command reads or writes, the session, which holds
  the history, the parse state, each entry's render, and all navigation rules,
  and the windows the `Window` command splits, opens and closes - a tree of
  them, each holding a whole session of its own, plus where each one lands on
  the screen.
- `src/rederive/syntax/` - Derive expression text to expression trees, and back
  out again - as Derive notation, or as C, Python, Rust or Julia source. It
  does no mathematics: `2+3` is a sum node, never `5`.
- `src/rederive/display/` - expression trees to built-up, multi-row terminal
  renders, plus the rectangle each subexpression lands on.
- `src/rederive/engine/` - the mathematics: expression trees to sympy and back,
  and the commands built on that pair. `worker.py` is the child process the
  engine runs in and `remote.py` the proxy the app reaches it through, which is
  what makes a computation abortable with Esc and capped in the memory it may
  hold; a session given neither computes in this process, which is what the
  tests and every direct caller want.
- `src/rederive/memory.py` - what the program is holding, for the status line
  gauge: this process plus the engine worker, or nothing where the platform
  will not say.
- `src/rederive/ui/` - the Textual layer: theme, menu data, widgets, app.

## Dependencies

`textual` for the screen, `sympy` for the mathematics, and `psutil` for the
memory readings - the status line gauge, and the cap the engine worker is
watched against.
