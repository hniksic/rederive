"""The normal form of a sum: a rational function of that sum's primary variable.

Derive writes every sum, wherever it stands, as a rational function of one
variable: the most main one the sum holds, by the order list. The terms that
hold it are put over a common denominator, the numerator is multiplied out and
collected by the powers of that variable, and long division takes out whatever
part of it is a polynomial, so what is left over the denominator is proper.
Terms free of the variable are not touched by any of that - they are what the
constant term is added to, and that sum is written the same way about its own
primary variable, which is a variable further down the list.

So `(x + 1)^9 + y` is a ninth-degree polynomial in `x` plus `y`, while
`(y + 1)^9 + x` is `x + (y + 1)^9`: the ninth power is free of `x` and is left
exactly as it was written. That is the manual's section 4.3 demonstration, and
it is why `Manage Ordering` changes what Simplify answers.

Only sums. A product or a power that is not itself a sum is not distributed -
`2*x*(x - 3)^2` and `(x + 1)*(y + 1)` and `(x + 1)^2/y` come back as they were
written - which is what makes this a normal form for sums rather than a
wholesale expansion.

The mathematics is `expanding`'s, because writing a sum about one variable is
what Derive's Expand does about several: everything free of the variable is one
opaque quantity, and that is what leaves `3*x*(2*y + 1)^2` written in powers of
`2*y + 1` instead of multiplied out into `y`.

Two things the walk has to do that one pass over the tree does not:

* `sp.Add` is flat, so the part of the answer that is free of the primary
  variable is a sum the walk never sees as a node - `(x + 2*y + 1)^3 + z` comes
  out with `(2*y + 1)^3` and `z` side by side among the powers of `x`. They are
  written about their own primary variable in turn, which is what gives the
  original's `8*y^3 + 12*y^2 + 6*y + z + 1`.
* Collecting can build a sum that was not there before, and that one needs its
  normal form too. So the walk is repeated while it changes anything.

A sum this has written is never offered again, and that is not an optimization.
Writing one a second time takes back apart what the collecting found in common,
because the parts of a product a placeholder stands in for are multiplied
together to make it, and `2*(a + b)` multiplied out is `2*a + 2*b`.

Two things this does not do, both deliberate.

A common factor is never taken back out of the finished sum. `(x + 1)^2*y + y`
is `x^2*y + 2*x*y + 2*y` here and `y*(x^2 + 2*x + 2)` in the original, and that
is a difference in spelling and not in terms. It is not worth chasing, because
the original does not do it consistently: `y*x^2 + 2*y*x + 2*y` authored as it
stands comes back exactly as it stands, so Derive has two fixed points for the
one expression and which one it lands on depends on how the sum was written.
Both come from the original. Taking the factor out unconditionally is worse than
either - it turns `2*x + 2*y` into `2*(x + y)` and `x^2 + 2*x` into
`x*(x + 2)`, and the original leaves both alone.

And the primary is a variable here where the original's is a kernel. Derive
collects `LN(x)*(y + 1)^2 + LN(x)*z` on `LN(x)`, reaching
`(y^2 + 2*y + z + 1)*LN(x)`, because `LN(x)` is the most main thing the sum is
a polynomial in. Reproducing that means choosing a primary kernel rather than a
primary symbol - every maximal subexpression that is not a sum, a product or a
power, ordered by the most main variable it holds - and writing the sum as a
polynomial in that. Everything below would follow unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp

from rederive.engine.expanding import collected_by, held_about
from rederive.engine.ordering import ORDER_LIST, main_order

__all__ = ["normal_form"]

#: How many times the walk is repeated over its own answer. Each round can only
#: uncover sums one level further in - a coefficient the round before it
#: collected - so the count is a depth rather than an iteration limit, and the
#: deepest in the corpus is Derive's own `ODE` demonstration at five. The bound
#: is what keeps a rewrite that feeds itself from spinning.
_ROUNDS = 6


def normal_form(expression: sp.Basic, order: Sequence[str] = ORDER_LIST) -> sp.Basic:
    """Every sum in `expression` written about its own primary variable.

    `order` is the session's variable order list, which is what decides which
    variable that is.

    Total, like every other rewrite in the pipeline: what fails is not taken.
    """
    written: set[sp.Basic] = set()

    def pending(candidate: sp.Basic) -> bool:
        return _is_sum(candidate) and candidate not in written

    def step(total: sp.Basic) -> sp.Basic:
        answer = _as_rational(total, order)
        if _is_sum(answer):
            written.add(answer)
        return answer

    for _ in range(_ROUNDS):
        try:
            answer = expression.replace(pending, step, simultaneous=False)
        except Exception:
            return expression
        if answer == expression:
            return answer
        expression = answer
    return expression


def _is_sum(expression: sp.Basic) -> bool:
    return bool(expression.is_Add and expression.is_commutative)


def _as_rational(total: sp.Basic, order: Sequence[str]) -> sp.Basic:
    """One sum about its primary variable, and its free part about the next.

    Anything that is not a sum is already in normal form as far as this goes,
    which is what the recursion below rests on: a free part of one term is a
    product or a power, and writing it as a sum would multiply out something
    the original leaves folded.
    """
    if not _is_sum(total):
        return total
    primary = _primary(total, order)
    if primary is None:
        return total
    holding = [term for term in total.args if primary in term.free_symbols]
    free = [term for term in total.args if primary not in term.free_symbols]
    written = _rational_in(sp.Add(*holding), primary)
    if written is None:
        return total
    polynomial, proper = written
    # The polynomial's constant term is free of the primary variable, and so
    # belongs with the terms that were free of it all along: they are one sum,
    # and it is written about whichever variable is most main in it.
    terms = sp.Add.make_args(polynomial)
    powers = [term for term in terms if primary in term.free_symbols]
    ground = [term for term in terms if primary not in term.free_symbols] + free
    return _shared(sp.Add(*powers, _as_rational(sp.Add(*ground), order), proper))


def _shared(total: sp.Basic) -> sp.Basic:
    """A sum whose terms share a denominator, written over it.

    Which is what makes `(x + 1)^2/y + z` come out `(x^2 + 2*x + y*z + 1)/y`
    while `x/y + z` stays as it is: the first has three terms over `y` and the
    second has one, and a denominator only one term carries is that term's own
    business. Two are enough - `(u - v)/h*(x - a) + w` is
    `(x*(u - v) + a*(v - u) + h*w)/h`, where only two of the three terms
    arrived with an `h` under them.

    The proper part is never what is shared: its denominator holds the primary
    variable and no other term can carry it, having been divided by it already.
    """
    if not _is_sum(total):
        return total
    seen: dict[sp.Basic, int] = {}
    for term in total.args:
        # A sum can carry something that is no expression at all - a branch of
        # an undecided `IF` is a truth value - and asking that for a numerator
        # and a denominator is asking the wrong question.
        if not isinstance(term, sp.Expr):
            return total
        denominator = sp.fraction(term)[1]
        if denominator != 1:
            seen[denominator] = seen.get(denominator, 0) + 1
    if not any(count > 1 for count in seen.values()):
        return total
    combined = _attempt(total, sp.together)
    return total if combined is None else combined


def _attempt(expression: sp.Basic, rewrite) -> sp.Basic | None:
    try:
        candidate = rewrite(expression)
    except Exception:
        return None
    return candidate if isinstance(candidate, sp.Basic) else None


def _rational_in(
    expression: sp.Basic, primary: sp.Symbol
) -> tuple[sp.Basic, sp.Basic] | None:
    """`expression` as a polynomial in `primary` plus a proper part over it.

    Everything free of `primary` stands in for itself first, so that a
    denominator that does not hold the primary variable is invisible here: it
    belongs to a coefficient, and a coefficient is the business of the sum that
    coefficient turns out to be. That is what leaves `x/y + z` alone while
    `1/(x + 1) + 1/(x + 2) + y` reaches `(2*x + 3)/((x + 1)*(x + 2)) + y`.

    None where the shape is not a ratio of polynomials in the primary variable
    at all, which leaves the sum as it stands.
    """
    frozen, held = held_about(expression, primary)
    try:
        numerator, denominator = sp.fraction(sp.together(frozen))
        # The two rewrites that take a power apart are off, neither being
        # anything Derive does: `x^(a + b)` is not `x^a*x^b`.
        numerator = sp.expand(numerator, power_exp=False, power_base=False)
        if primary in denominator.free_symbols:
            polynomial, remainder = sp.div(numerator, denominator, primary)
        else:
            polynomial, remainder = _divided(numerator, denominator), sp.S.Zero
        polynomial = collected_by(polynomial, primary)
        remainder = collected_by(
            sp.expand(remainder, power_exp=False, power_base=False), primary
        )
    except Exception:
        return None
    proper = sp.S.Zero if remainder == 0 else remainder / denominator
    return polynomial.xreplace(held), proper.xreplace(held)


def _divided(numerator: sp.Basic, denominator: sp.Basic) -> sp.Basic:
    """A numerator over a denominator that holds no primary variable.

    The denominator is a coefficient like any other, so it is divided into each
    term rather than left outside: `(x + y*z)/y` is `x/y + z` and only then is
    it a polynomial in `x`.
    """
    if denominator == 1:
        return numerator
    return sp.expand(numerator / denominator, power_exp=False, power_base=False)


def _primary(total: sp.Basic, order: Sequence[str]) -> sp.Symbol | None:
    """The most main variable this sum holds, by the order list."""
    # `type is` rather than `isinstance`: a string literal is a `Symbol`
    # subclass, and a string is data rather than a variable to write about.
    named = {
        symbol.name: symbol
        for symbol in total.free_symbols
        if type(symbol) is sp.Symbol
    }
    if not named:
        return None
    return named[main_order(named, order)[0]]
