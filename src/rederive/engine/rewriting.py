"""A rewrite offered, and whether it is kept.

Every rewrite Simplify offers is offered inside a `try`, and a rewrite that raises is
one that was not worth having: `attempt` is that. `forced` keeps what a rewrite gives
whatever it costs, and `pays` is the gate a rewrite offered as an option has to pass.
"""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp

__all__ = ["Rewrite", "attempt", "forced", "pays"]


Rewrite = Callable[[sp.Basic], sp.Basic]


def forced(expression: sp.Basic, rewrite: Rewrite) -> sp.Basic:
    """`rewrite`, whatever it costs: a direction asked for, a fact, or a spelling."""
    candidate = attempt(expression, rewrite)
    return expression if candidate is None else candidate


def attempt(expression: sp.Basic, rewrite: Rewrite) -> sp.Basic | None:
    try:
        candidate = rewrite(expression)
    except Exception:
        return None
    return candidate if isinstance(candidate, sp.Basic) else None


def pays(candidate: sp.Basic, previous: sp.Basic) -> bool:
    """Whether a rewrite earned its keep.

    One variable fewer always counts, however long the answer: a superfluous
    variable is the first thing the manual says a simplified expression does
    not have. One denominator fewer under another counts the same way, being
    the other thing a simplified expression does not have. Failing both, the
    rewrite has to be shorter.
    """
    try:
        if candidate == previous:
            return False
        if len(candidate.free_symbols) < len(previous.free_symbols):
            return True
        if _compound(candidate) != _compound(previous):
            return _compound(previous)
        return sp.count_ops(candidate) < sp.count_ops(previous)
    except Exception:
        return False


def _compound(expression: sp.Basic) -> bool:
    """Whether a denominator in `expression` has a denominator of its own.

    Which is the one thing Derive never leaves standing whatever it costs:
    `DIF(ATAN(x/a), x)` is `1/(a + x^2/a)` to sympy and `a/(a^2 + x^2)` to the
    original, and the two are the same length, so nothing that counts
    operations can tell them apart.

    A number underneath is not a denominator of this kind. `x + 1/2` is a sum
    of a variable and a number wherever it stands, and putting it over two is
    the longer way to write it.
    """
    for power in expression.atoms(sp.Pow):
        if not power.exp.is_negative:
            continue
        for inner in power.base.atoms(sp.Pow):
            if inner.exp.is_negative and not inner.base.is_number:
                return True
    return False
