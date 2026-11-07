"""Products of a real argument with its absolute value and its sign, folded.

`ABS(u)`, `SIGN(u)` and `u` are one real quantity written three ways, and
`folded_signs` writes a product holding more than one of them the shortest way.
"""

from __future__ import annotations

import sympy as sp

__all__ = ["folded_signs"]


#: Where the three ways of writing one real quantity sit in a product. Each is
#: a power of the other two - `ABS(u) = u*SIGN(u)` and `SIGN(u)^2 = 1` - so a
#: group is no more than the exponent every one of them carries.
_ABSOLUTE, _SIGNUM, _BARE = 0, 1, 2


def folded_signs(expression: sp.Basic) -> sp.Basic:
    """Every product of a real argument with its absolute value and its sign."""
    return expression.replace(
        lambda e: _folded_product(e) is not None,
        lambda e: _folded_product(e),
        simultaneous=False,
    )


def _folded_product(expression: sp.Basic) -> sp.Basic | None:
    """This product with `ABS(u)`, `SIGN(u)` and `u` collected into one another.

    For a real `u` the three are one quantity written three ways, so a product
    holding more than one of them says the same thing twice: `ABS(u)*SIGN(u)`
    and `ABS(u)/SIGN(u)` are both `u`, `u/ABS(u)` is `SIGN(u)`, `u*SIGN(u)` is
    `ABS(u)` and `SIGN(u)^2` is `1`.

    Zero is where the identities are a licence rather than a fact: `SIGN(0)` is
    zero, so `ABS(u)/SIGN(u)` is undefined there and `u` is not. Derive removes
    the singularity and so does this, on the same terms that make `u/u` come
    back `1`. Off the real line there is nothing to remove and nothing to fold:
    `SIGN(z)` is `z/|z|`, whose square is not one, so an argument nothing
    declares real is left as it was written.
    """
    groups, rest = _signed_powers(expression)
    if not any(_worth_folding(*exponents) for exponents in groups.values()):
        return None
    folded = [
        _shortest_spelling(argument, *exponents)
        if _worth_folding(*exponents)
        else _written(argument, *exponents)
        for argument, exponents in groups.items()
    ]
    candidate = sp.Mul(*rest, *folded)
    return None if candidate == expression else candidate


def _worth_folding(absolute: int, signum: int, bare: int) -> bool:
    """Whether this group says the same thing twice.

    One `ABS` or one `SIGN` by itself is what it is, and a bare power with
    neither is no business of this rule. What is left over is a group that
    holds two of the three, or a sign the argument carries more than once, or
    a sign under a division: all three collapse.
    """
    if not (absolute or signum):
        return False
    present = sum(1 for exponent in (absolute, signum, bare) if exponent)
    return present > 1 or abs(absolute) > 1 or signum not in (0, 1)


def _signed_powers(
    expression: sp.Basic,
) -> tuple[dict[sp.Basic, list[int]], list[sp.Basic]]:
    """This product read as groups of one real argument, and the factors that are not.

    A factor joins the group of the argument it is about - `ABS(u)^2` and
    `1/SIGN(u)` and `u` are all the group of `u` - and anything else, an
    argument sympy cannot call real or a power the exponents cannot be added
    up over, stays a factor of its own.

    A range is not one of these arguments. `SIGN(INTERVAL(-1, 1))` is the
    three-point set the interval's sign can be and not a number the identities
    hold for, so a bound stays the factor it was written as.
    """
    groups: dict[sp.Basic, list[int]] = {}
    rest: list[sp.Basic] = []
    for factor in sp.Mul.make_args(expression):
        base, exponent = factor.as_base_exp()
        if isinstance(base, sp.Abs):
            slot, argument = _ABSOLUTE, base.args[0]
        elif isinstance(base, sp.sign):
            slot, argument = _SIGNUM, base.args[0]
        else:
            slot, argument = _BARE, base
        if (
            exponent.is_Integer
            and argument.is_extended_real
            and not argument.has(sp.AccumBounds)
        ):
            groups.setdefault(argument, [0, 0, 0])[slot] += int(exponent)
        else:
            rest.append(factor)
    return groups, rest


def _written(argument: sp.Basic, absolute: int, signum: int, bare: int) -> sp.Basic:
    """This group as the author wrote it."""
    return sp.Abs(argument) ** absolute * sp.sign(argument) ** signum * argument**bare


def _shortest_spelling(
    argument: sp.Basic, absolute: int, signum: int, bare: int
) -> sp.Basic:
    """The shortest of the ways this group can be written.

    Trading one absolute value for the argument leaves a sign behind, and two
    signs are one, so the total power and the parity of the sign are all the
    group is: every `k` of `ABS(u)^k * u^(total - k) * SIGN(u)^(parity - k)`
    says it, with the last exponent taken to zero or one. The rule is therefore
    no direction to write `ABS` or to write `SIGN` but a choice of whichever
    `k` comes out shorter, and ties - the spelling the author wrote among them
    - go to what was already there.
    """
    total = absolute + bare
    parity = (absolute + signum) % 2
    spellings = [_written(argument, absolute, signum, bare)] + [
        sp.Abs(argument) ** count
        * argument ** (total - count)
        * sp.sign(argument) ** ((parity - count) % 2)
        for count in sorted({0, 1, absolute, total})
    ]
    return min(spellings, key=sp.count_ops)
