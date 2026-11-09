"""Applying rewrites and deciding whether to keep them.

Simplify tries every rewrite with `attempt`, which treats a rewrite that raises as not
applicable. `forced` keeps a rewrite's result regardless of cost, and `pays` decides
whether an optional rewrite is worth keeping.
"""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp

__all__ = ["Rewrite", "attempt", "forced", "pays"]


Rewrite = Callable[[sp.Basic], sp.Basic]


def forced(expression: sp.Basic, rewrite: Rewrite) -> sp.Basic:
    """Apply `rewrite` regardless of cost (for a requested direction, a fact, or a
    spelling), keeping `expression` if it fails."""
    candidate = attempt(expression, rewrite)
    return expression if candidate is None else candidate


def attempt(expression: sp.Basic, rewrite: Rewrite) -> sp.Basic | None:
    try:
        candidate = rewrite(expression)
    except Exception:
        return None
    return candidate if isinstance(candidate, sp.Basic) else None


def pays(candidate: sp.Basic, previous: sp.Basic) -> bool:
    """Whether a rewrite is worth keeping.

    Losing a variable always pays, however long the result, since the manual says a
    simplified expression has no superfluous variables. Likewise for removing a nested
    denominator. Otherwise the rewrite must be shorter.
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

    Derive always removes these: `DIF(ATAN(x/a), x)` is `1/(a + x^2/a)` to sympy and
    `a/(a^2 + x^2)` to Derive. Both have the same length, so counting operations can't
    tell them apart.

    Numeric inner denominators don't count: `x + 1/2` is shorter than putting it over
    two.
    """
    for power in expression.atoms(sp.Pow):
        if not power.exp.is_negative:
            continue
        for inner in power.base.atoms(sp.Pow):
            if inner.exp.is_negative and not inner.base.is_number:
                return True
    return False
