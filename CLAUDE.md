# Rederive

A computer algebra system inspired by the classic "Derive" program for DOS. It is a
terminal UIO over over a SymPy-based engine. Python 3.14+, managed with `uv`.

## Commands

- `uv run rederive` - run the app
- `uv run pytest` - the whole suite, in parallel (under a minute)
- `uv run pytest -n0 tests/test_simplify.py -k name` - one test, serially, for readable
  output

## Rules

- The app process must not import sympy. It talks to the engine through `engine.client`;
  only the worker, and code that computes in-process, imports `engine.computing`.
  `tests/test_packages.py` enforces this.
- Derive's behavior is used as a guideline, and we try to remain compatible with Derive's
  utility files and demos. Where Derive is buggy, limited or arbitrary, do what a good
  algebra system would do instead.
- `CAPABILITIES.md` is a short, high-level overview of where Rederive goes beyond Derive
  and the few places it deliberately differs. Almost no commit touches it: a fix, an
  internal rule or a design decision is not a capability. Edit it only for a new
  built-in or a difference a user would notice and want explained, in a line or two.
- Don't edit `README.md` unless asked to.
- Start a new worktree from the current branch's tip (typically local `master`), not
  from `origin/master`.
- An identity is applied only where it holds for every value the variables may take.
  An undeclared variable is real, so `SQRT(x^2)` is `ABS(x)`, not `x`.
- Arithmetic is exact: no floating point in computing or representing numbers.
- A symbolic answer rests on exact computation only. Numeric evaluation never decides
  a symbolic fact (that an expression is zero or constant, a sign, an equality,
  independence of a variable); what can't be decided exactly stays undecided. A numeric
  test may reject a candidate or pick what to try, provided the result is then proved
  exactly. Numbers are computed where the user asked for them: approximation, numeric
  solving, plotting.
