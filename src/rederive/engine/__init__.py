"""The math engine: expression trees in, expression trees out.

This file is documentation and nothing else - it exports no names, because the
package has two halves and a caller has to say which one it is entitled to.
Nothing inside the package may import from the UI, and nothing outside it may
import from inside except through one of the two:

* `client` is what anyone may hold: the `RemoteEngine` proxy, the vocabulary that
  crosses the pipe, and the questions a tree answers by itself. It imports no
  sympy, and the app process holds this and nothing more.
* `computing` is the mathematics, and importing it means importing sympy. The
  worker holds this, and so does anything that computes in place - every test,
  every direct caller. Everything `client` offers it offers too, so a caller who
  may compute writes one import.

The import line is therefore the whole of the rule, which is why there is no third
way in: a module that names a command has said in its imports that it is allowed
to. `tests/test_packages.py` is what holds the app side to its half.

What crosses the boundary is a `Node` tree and a `Context`; what comes back is a
`Result` carrying both the answer's text and its reparsed tree.

Two doors, shared by every command the engine will ever grow:

* `to_sympy` translates a tree into sympy faithfully. It maps each node to the
  object that means the same thing and simplifies nothing beyond sympy's own
  automatic evaluation. What it reads is the context: the precision mode, the
  angle mode, the input base, and the domains that say what a variable is.
* `from_sympy` writes an expression back as author notation and reparses it,
  which is how any result becomes a worksheet entry.

The mapping is total in both directions. Every node kind converts, and
anything the engine has no mathematics for becomes an inert head that carries
its operands, survives untouched and prints back to the notation it came from.
A construct sympy will not take - a call with arguments it cannot use, a
product of shapes it cannot multiply - becomes such a head as well, rather
than a guess at what was meant.

Commands are built on top of these two doors and the converters know nothing
about them. The first of them is Simplify.

`simplify(node, context)` promises Derive's Simplify: the sufficiently simple
form of an expression - no superfluous variables, roots, functions or reducible
degrees - with every sum in normal form, reached by transforming as little else
as necessary. It reads only its `Context`: precision, branch, the Trigonometry,
Trigpower, Exponential and Logarithm directions, angle measure, input base, the
variable order list, and the domains, assignments, function definitions and
labels the session has recorded. It has no other state and no side effects, so
the same tree and context always give the same answer.

The normal form is a sum written as a rational function of the most main
variable it holds, which the order list is what decides. So `(x + 1)^9 + y` is
a ninth-degree polynomial in `x`, `(y + 1)^9 + x` is left as it was written,
and `Manage Ordering` changes both answers. Only sums: a product or a power
that is not itself a sum is never distributed, which is why `2*x*(x - 3)^2` and
`(x + 1)*(y + 1)` come back untouched.

Two promises hold across every input:

* It never raises on anything the parser produced. A rewrite that fails is a
  rewrite not taken, and the previous form stands.
* It never guesses. A transformation that needs a variable to be real, or
  positive, or an integer, fires only where a declaration says so; where
  nothing says so the expression comes back as it went in. An undeclared
  variable is real, which is Derive's own default.

`approx` is the same pipeline with the precision mode set to Approximate, which
is what the manual says the approX command is.

`factor(node, context, amount, variables)` is Derive's Factor: the same
expression written as a product. It is Simplify and then factoring, which is
what the manual means by saying both commands reach a sufficiently simple form
and Factor goes further, and it makes the same two promises. `Amount` is how
far it goes - Trivial, Squarefree, Rational, raDical or Complex, each doing
everything the one before it does - and `variables` names the factorization
variables, empty meaning all of them.

`expand(node, context, amount, variables)` is Derive's Expand: the same
expression written as a sum. It is Simplify and then expanding, and it makes
the same two promises. `variables` names the expansion variables, empty
meaning all of them, and everything free of them is left alone - which is what
writes `(x + 2*y + 1)^3` about `x` in powers of `2*y + 1`. A ratio whose
denominator holds an expansion variable becomes partial fractions instead, and
`amount` says how far the denominator is factored on the way; Expand offers
four of the five amounts, `Complex` being Factor's alone.

`solve(node, context, variables, bounds)` is Derive's soLve, and it is the one
command whose answer is not one expression. It is Simplify and then solving,
and what comes back is a *tuple* of results, one per solution, each of them a
relation with the variable alone on the left: `x^2 - 5*x + 6 = 0` solves to two
of them, a system solves to one holding the whole solution vector, and an
interval solves to one chained relation `-2 < x < 2`. Three answers are not
solutions and are told apart on purpose - the empty tuple is "no solutions
found", an equation that holds everywhere answers with the arbitrary value
`x = @1` that `Context.arbitrary_index` mints, and an equation nothing can
solve answers with its own residual equation, `3^x - x^2 = 0`. None of them is
an exception: this makes the same two promises the others do.

`variables` names what to solve for, empty meaning the one variable of a
scalar or the most main of a system's; `bounds` confines a numeric search to an
interval, which is what Approximate precision asks the user for and what the
other two modes never do. Unlike every other command this one works on a whole
entry rather than on any subtree: solving half an expression produces something
there is no sensible way to splice back into the other half.

The mathematics of all three is a file of its own below the pipeline rather
than above, because `FACTOR(u, amount, x, y, ...)`, `EXPAND(u, amount, x, y,
...)` and `SOLVE(u, x)` are an authored line's own way of asking for the same
things, and Simplify is what evaluates those. `SOLVE` evaluates to a vector of
relations, which is the Derive 3 and 4 shape and the one the shipped libraries
count with `DIMENSION` and take apart with `RHS`.

`replace(node, replacements, state)` is Derive's Manage Substitute, and it is
the one command that computes nothing: each replacement is a subtree to look
for and what to write in its place, every match is replaced at once, and the
answer is written out unsimplified. It is not `substitute`, which is the
pre-pass every other command runs to write in what a name already stands for.

`expression_variables(node, context)` is what a command offers before it can
ask anything: the variables the expression holds, most main first.

Every command works on any subtree, not only on a whole authored line, so the
session can act on what the user has highlighted and put the answer back.

None of these promises covers what a computation costs. Simplify of `1000000!`
finishes eventually and `10^10^10` does not finish at all, and neither can be
interrupted from the inside, sympy having no cooperative cancellation to ask
for. So the engine also ships with a way to run it at arm's length:
`RemoteEngine` offers the six heavy calls with the signatures the session
already uses and answers them out of a child process that can be killed, which
is what makes Esc an abort and a memory cap enforceable. A session given one
computes remotely; a session given nothing computes here, which is what every
test and every direct caller wants.
"""
