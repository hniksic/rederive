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

## Testing

```sh
uv run pytest
```

## Layout

- `src/rederive/model/` - pure Python, no Textual: the expression tree, the
  system control settings, the file formats a `Transfer` command reads or
  writes, and the session, which holds the history, the parse state, each
  entry's render, and all navigation rules.
- `src/rederive/syntax/` - Derive expression text to expression trees, and back
  out again - as Derive notation, or as C, Python, Rust or Julia source. It
  does no mathematics: `2+3` is a sum node, never `5`.
- `src/rederive/display/` - expression trees to built-up, multi-row terminal
  renders, plus the rectangle each subexpression lands on.
- `src/rederive/ui/` - the Textual layer: theme, menu data, widgets, app.
