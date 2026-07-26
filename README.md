# Rederive

A modern remake of the DOS-era *Derive* computer algebra system.

This is milestone 1: a Textual TUI shell that looks and navigates like DOS
Derive but performs no mathematics at all. It exists so the look and feel -
menu highlighting, Tab cycling, mnemonic letters, expression input, and
arrow-key movement across expressions and subexpressions - can be evaluated
before any engine work starts.

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
  session (history plus selection), and all navigation rules.
- `src/rederive/placeholder_parser.py` - temporary scaffolding that gives
  authored text a shape to navigate. It will be deleted when the real engine
  lands.
- `src/rederive/display/` - expression trees to built-up, multi-row terminal
  renders, plus the rectangle each subexpression lands on.
- `src/rederive/ui/` - the Textual layer: theme, menu data, widgets, app.
