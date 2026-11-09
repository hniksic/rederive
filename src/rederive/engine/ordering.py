"""Derive's variable order list, and the ordering it induces.

Derive orders variables by a list it calls the order list, initially `x`, `y`, `z`. A
listed variable is more main than any later or unlisted one; unlisted variables are
ordered alphabetically.

The list applies wherever a command picks a main variable. Simplify's normal form writes
every sum as a rational function of its most main variable, so the list decides which
sums get multiplied out: `(x + 1)^9 + y` becomes a ninth-degree polynomial under `x y z`
but is left alone under `y x z`, where the ninth power is free of the main variable. The
list also orders the variables Factor and Expand offer (the first chosen becomes main),
the generators of an Expand about all variables, and the parameters of a `Declare
Function` definition.

It also determines how sums and products are written. A term sorts by its leading
variable, so `5*y^3 + 2*x^2 - 3*a*x` is written `2*x^2 - 3*a*x + 5*y^3`; a factor sorts
by its base, so `z^3*b^2*a*2*x^5*c` is written `2*a*b^2*c*x^5*z^3` (in products,
unlisted names go *before* listed ones). That is `printer`'s job, and is presentation
only, not part of the normal form. Operands the list doesn't distinguish keep sympy's
order.

`Manage Ordering` changes the list, so it is session state carried in the `Context`.
`ORDER_LIST` is the initial list, also used by callers without a session.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

__all__ = ["ORDER_LIST", "main_order"]

#: The variables Derive starts with on its order list, most main first.
ORDER_LIST: tuple[str, ...] = ("x", "y", "z")


def main_order(
    names: Iterable[str], order: Sequence[str] = ORDER_LIST
) -> tuple[str, ...]:
    """`names` from most main to least, without repeats.

    `z^2 - a^2` gives `z, a`: `z` is on the order list and `a` is not.
    """
    wanted = set(names)
    listed = [name for name in order if name in wanted]
    return tuple(listed + sorted(wanted.difference(listed)))
