"""Derive's variable order list, and the ordering it induces.

Sums and products are commutative, so the normal form has to pick an order for
their operands; Derive picks it from a list of variables it calls the order
list, which starts out as `x`, `y`, `z`. A variable on the list is more main
than one after it on the list and more main than one not on the list at all;
variables off the list are ordered among themselves alphabetically.

What the list reaches here is everywhere a command has to pick a variable to be
main: the order Factor and Expand offer their variables in, which makes the
first one chosen the primary variable; the generators an Expand about all of
them expands about; and the order a `Declare Function` definition's variables
become the function's parameters in. It does not reach the term order of a
simplified sum, which is sympy's beyond the two rules `printer` keeps - so the
manual's own demonstration of the list, resimplifying `(x + 1)^9 + y` under
`y x z` to keep it from expanding, has nothing to show in Rederive, whose
Simplify does not expand it under either order.

`Manage Ordering` rewrites the list, so the list is session state rather than a
constant: it travels in the `Context`, alongside everything else a command's
answer depends on. `ORDER_LIST` is only what a session starts with, and is what
a caller with no session of its own orders by.
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
