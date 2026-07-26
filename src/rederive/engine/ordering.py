"""Derive's variable order list, and the ordering it induces.

Sums and products are commutative, so the normal form has to pick an order for
their operands; Derive picks it from a list of variables it calls the order
list, which starts out as `x`, `y`, `z`. A variable on the list is more main
than one after it on the list and more main than one not on the list at all;
variables off the list are ordered among themselves alphabetically.

The ordering shows outside the normal form too: it is the order Factor and
Expand offer their variables in, which makes the first one chosen the primary
variable, and the order a `Declare Function` definition's variables become the
function's parameters in.

`Manage Ordering`, which lets the user rewrite the list, is not implemented, so
the list is the initial one and is a constant here.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = ["ORDER_LIST", "main_order"]

#: The variables Derive starts with on its order list, most main first.
ORDER_LIST: tuple[str, ...] = ("x", "y", "z")


def main_order(names: Iterable[str]) -> tuple[str, ...]:
    """`names` from most main to least, without repeats.

    `z^2 - a^2` gives `z, a`: `z` is on the order list and `a` is not.
    """
    wanted = set(names)
    listed = [name for name in ORDER_LIST if name in wanted]
    return tuple(listed + sorted(wanted.difference(listed)))
