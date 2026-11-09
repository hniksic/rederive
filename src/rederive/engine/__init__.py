"""The math engine: expression trees in, expression trees out.

This file is documentation only and exports no names. The package has two halves, and
nothing outside it may import from inside except through one of them (nothing inside may
import from the UI):

* `client` imports no sympy and is all the app process holds: the vocabulary that
  crosses to the worker, the exceptions a dead worker raises, and the questions answered
  by reading a tree. `remote` is an implementation over it that runs a child process;
  `policy` handles recovery when a worker dies.
* `computing` is the mathematics and imports sympy. The worker uses it, as does anything
  that computes in-process (tests, direct callers). It re-exports everything in `client`,
  so such callers need one import.

The import line thus states what a module is allowed to do. `tests/test_packages.py`
enforces that the app side stays on its half.

A `Node` tree and a `Context` go in; a `Result` with the answer's text and reparsed tree
comes back. All commands share two converters:

* `to_sympy` translates a tree into sympy faithfully, simplifying nothing beyond sympy's
  automatic evaluation. It reads the precision mode, angle mode, input base and variable
  domains from the context.
* `from_sympy` writes an expression back as author notation and reparses it.

Both directions are total. Anything the engine has no mathematics for, including
constructs sympy rejects (a call with unusable arguments, a product of incompatible
shapes), becomes an inert head that keeps its operands and prints back as written.

`simplify(node, context)` implements Derive's Simplify: a sufficiently simple form (no
superfluous variables, roots, functions or reducible degrees) with every sum in normal
form, transforming as little else as necessary. It depends only on its `Context` and has
no side effects, so the same input always gives the same answer.

The normal form writes a sum as a rational function of its most main variable, as given
by the order list. So `(x + 1)^9 + y` becomes a ninth-degree polynomial in `x`, `(y +
1)^9 + x` is left alone, and `Manage Ordering` changes both. Only sums are normalized: a
product or power that isn't a sum is never distributed, so `2*x*(x - 3)^2` and `(x +
1)*(y + 1)` come back untouched.

Every command makes two promises:

* It never raises on parser output. A rewrite that fails is skipped and the previous
  form stands.
* It never guesses. A transformation that needs a variable to be real, positive or an
  integer fires only where a declaration says so. Undeclared variables are real, as in
  Derive.

`approx` is the same pipeline with precision set to Approximate.

`factor(node, context, amount, variables)` is Derive's Factor: Simplify followed by
factoring into a product. `amount` is Trivial, Squarefree, Rational, raDical or Complex,
each including the previous; `variables` names the factorization variables, empty meaning
all.

`expand(node, context, amount, variables)` is Derive's Expand: Simplify followed by
expanding into a sum. Everything free of `variables` (empty meaning all) is left alone,
which is how `(x + 2*y + 1)^3` expands about `x` in powers of `2*y + 1`. A ratio whose
denominator holds an expansion variable becomes partial fractions, and `amount` says how
far the denominator is factored; `Complex` is Factor-only.

`solve(node, context, variables, bounds)` is Derive's soLve: Simplify followed by solving.
It returns a *tuple* of results, one per solution, each a relation with the variable
alone on the left. `x^2 - 5*x + 6 = 0` gives two, a system gives one holding the solution
vector, and an inequality gives a chained relation such as `-2 < x < 2`. Three special
answers are deliberately distinct: the empty tuple means no solutions were found, an
identity answers `x = @1` (an arbitrary value from `Context.arbitrary_index`), and an
unsolvable equation answers its own residual, e.g. `COS(x) - x = 0`. None raises.

`variables` defaults to the single variable of a scalar or the most main of a system;
`bounds` limits a numeric search to an interval and applies only in Approximate mode.
Solve always works on a whole entry, since a solved subexpression can't be spliced back.

Factoring, expanding and solving live below the pipeline, because `FACTOR(u, amount, x,
...)`, `EXPAND(u, amount, x, ...)` and `SOLVE(u, x)` can be written in an expression and
Simplify evaluates them. `SOLVE` evaluates to a vector of relations, the Derive 3 and 4
shape that the shipped libraries use with `DIMENSION` and `RHS`.

`replace(node, replacements, state)` is Derive's Manage Substitute. It computes nothing:
it replaces every match of each subtree at once and returns the result unsimplified. This
differs from the evaluation other commands start with, which substitutes assigned values.

`expression_variables(node, context)` lists an expression's variables, most main first.
`relation_of(node, context)` gives the relation operator an expression evaluates to, if
any, so Plot can draw `IM(u = v)` as an equation.

Every command works on any subtree, so the session can act on a highlighted selection.

None of this bounds cost. `1000000!` simplifies eventually, `10^10^10` never does, and
sympy offers no cooperative cancellation. `remote.RemoteEngine` therefore runs the six
heavy calls, with the same signatures, in a killable child process, which makes Esc
abort and a memory cap enforceable. A session without one computes in-process, which is
what tests and direct callers want.
"""
