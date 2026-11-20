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
- Don't edit `README.md` unless asked to.
- Update a document where a change makes it wrong or incomplete by its own standard: a
  new built-in gets an entry where the built-ins are listed, a changed command gets its
  description corrected. Don't add a paragraph to give a change a mention: if the
  document has no existing place for the fact, it doesn't cover that kind of fact. What
  changed is described in the commit message.
- Worktrees start from the local HEAD, never from a remote ref (local branches have
  unpushed commits). Don't use the harness's worktree features (`isolation: "worktree"`,
  `EnterWorktree`, `--worktree`); run `git worktree add <path> -b <name> HEAD` yourself,
  and check that the worktree's HEAD matches yours before any agent works in it.
- Mind memory: SymPy processes are heavy and a runaway computation or too many parallel
  test runs can freeze the machine. When spawning several agents, limit how many run at
  once and tell each one to watch its memory use; when running a sweep or the suite
  alongside other work, do the same yourself.
- A test or docstring is not evidence of intent just by being there: tests often record
  what the code did when they were written, and a docstring often just restates the
  code. Give weight to a rationale in proportion to its argument and sources: one that
  cites a Derive transcript or manual, or makes a mathematical case you can check,
  counts; a bare assertion that the result is right, or that the behavior is deliberate,
  does not. If a change gives a better result than a test expects, update the test.
- An identity is applied only where it holds for every value the variables may take.
  An undeclared variable is real, so `SQRT(x^2)` is `ABS(x)`, not `x`.
- Arithmetic is exact: no floating point in computing or representing numbers.
- A symbolic answer rests on exact computation only. Numeric evaluation never decides
  a symbolic fact (that an expression is zero or constant, a sign, an equality,
  independence of a variable); what can't be decided exactly stays undecided. A numeric
  test may reject a candidate or pick what to try, provided the result is then proved
  exactly. Numbers are computed where the user asked for them: approximation, numeric
  solving, plotting.
