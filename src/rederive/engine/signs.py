"""Simplifying products of a real argument with its absolute value and sign.

For real `u`, `ABS(u)`, `SIGN(u)` and `u` are related, and `folded_signs` writes a
product containing more than one of them in the shortest way. For any `z`, the modulus
and phase are the parts of a logarithm, which `polar_logarithms` joins.
"""

from __future__ import annotations

import sympy as sp

__all__ = ["folded_signs", "polar_logarithms"]


#: Indices of the `ABS(u)`, `SIGN(u)` and `u` exponents in a group. Since `ABS(u) =
#: u*SIGN(u)` and `SIGN(u)^2 = 1`, a group is fully described by these three exponents.
_ABSOLUTE, _SIGNUM, _BARE = 0, 1, 2


def folded_signs(expression: sp.Basic) -> sp.Basic:
    """`expression` with every product of a real `u`, `ABS(u)` and `SIGN(u)` folded."""
    return expression.replace(
        lambda e: _folded_product(e) is not None,
        lambda e: _folded_product(e),
        simultaneous=False,
    )


def _folded_product(expression: sp.Basic) -> sp.Basic | None:
    """This product with `ABS(u)`, `SIGN(u)` and `u` folded together, or None.

    For real `u`, `ABS(u)*SIGN(u)` and `ABS(u)/SIGN(u)` are `u`, `u/ABS(u)` is
    `SIGN(u)`, `u*SIGN(u)` is `ABS(u)` and `SIGN(u)^2` is `1`.

    At zero these are not strictly true: `SIGN(0)` is zero, so `ABS(u)/SIGN(u)` is
    undefined there but `u` is not. Derive removes the singularity, as with `u/u` ->
    `1`, and so does this. For non-real `z`, `SIGN(z)` is `z/|z|`, whose square isn't
    one, so arguments not known to be real are left alone.
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
    """Whether this group can be folded.

    A lone `ABS` or `SIGN` can't, and a bare power without either is irrelevant. A
    group with two of the three, a repeated `ABS` or `SIGN`, or a `SIGN` in the
    denominator can.
    """
    if not (absolute or signum):
        return False
    present = sum(1 for exponent in (absolute, signum, bare) if exponent)
    return present > 1 or abs(absolute) > 1 or signum not in (0, 1)


def _signed_powers(
    expression: sp.Basic,
) -> tuple[dict[sp.Basic, list[int]], list[sp.Basic]]:
    """This product's factors grouped by real argument, plus the remaining factors.

    `ABS(u)^2`, `1/SIGN(u)` and `u` all join the group of `u`. Factors whose argument
    sympy can't prove real, or whose exponent isn't an integer, are kept separately.

    So are intervals: `SIGN(INTERVAL(-1, 1))` is a three-point set, not a number the
    identities hold for.
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
    """This group as originally written."""
    return sp.Abs(argument) ** absolute * sp.sign(argument) ** signum * argument**bare


def _shortest_spelling(
    argument: sp.Basic, absolute: int, signum: int, bare: int
) -> sp.Basic:
    """The shortest way to write this group.

    Replacing an `ABS(u)` with `u` leaves a `SIGN(u)`, and two signs cancel, so the
    group is determined by its total power and sign parity. Every `k` in `ABS(u)^k *
    u^(total - k) * SIGN(u)^(parity - k)` (last exponent reduced mod 2) is equivalent,
    and the shortest is chosen. Ties go to the form as written.
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


def polar_logarithms(expression: sp.Basic) -> sp.Basic:
    """Join the parts of a logarithm: `LN(ABS(z)) + #i*PHASE(z)` is `LN(z)`.

    This is the principal logarithm's definition, so it holds for every nonzero `z`, and
    with any common factor: `c*LN(ABS(z)) + c*#i*PHASE(z)` is `c*LN(z)`. Sympy's
    `solveset` writes the logarithm in the split form, as in the solution of `#e^y = x`.
    """

    def joined(total: sp.Add) -> sp.Basic:
        terms = list(total.args)
        changed = False
        for term in total.args:
            if term not in terms:
                continue
            for factor in sp.Mul.make_args(term):
                if not isinstance(factor, sp.log) or not isinstance(
                    factor.args[0], sp.Abs
                ):
                    continue
                argument = factor.args[0].args[0]
                coefficient = term / factor
                partner = sp.I * coefficient * sp.arg(argument)
                if partner in terms:
                    terms.remove(term)
                    terms.remove(partner)
                    terms.append(coefficient * sp.log(argument))
                    changed = True
                    break
        return sp.Add(*terms) if changed else total

    if not (expression.has(sp.arg) and expression.has(sp.Abs)):
        return expression
    return expression.replace(lambda e: isinstance(e, sp.Add), joined)
