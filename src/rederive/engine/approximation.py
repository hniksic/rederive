"""What an approximate number is: the simplest rational the precision allows.

Derive's approximate arithmetic is exact arithmetic over approximate numbers,
and an approximate number is a *rational* - the simplest one that stands for
the value to the precision asked for. Approximating π to six digits gives
355/113, which is why `Notation := Rational` shows exactly that, and why an
approximate answer is as exact a value as any other: it is just a different
value from the one it approximates.

Three consequences:

* A number that needs no digits does not get any. The simplest rational
  standing in for 5 is 5, and the one standing in for two thirds is two
  thirds - so `2/3` approximated is `2/3`, and shown to six digits it is
  `0.666666`, cut rather than rounded up to `0.666667`.
* A value keeps the digits it is shown by. `SQRT(3)` is `1.73205` and not the
  `1.73204` that a simpler rational a little below it would show.
* Every digit shown is the value's own. A number is worked out as if exactly,
  with as much working precision as that takes, and rounded only then - not
  rounded piece by piece and computed with afterwards, which lets the
  arithmetic carry the rounding of the pieces into the answer. The thousandth
  power of a root of 12, written in radicals, is 12 and not the `11.9994` that
  rounding its radicals first leaves, and `SQRT(10^200 + 1) - 10^100` is
  `5.00000*10^-101` and not the 0 of two rounded numbers cancelling. This is
  where the original is left behind on purpose: it rounds the pieces, and
  answers those two `12 - 0.000430980*#i` and 0.

Where this parts from the manual: 3.8 p.46 turns on the two fractions of
`SQRT(3422357/2313 - 1140443/771)` being rounded before they are subtracted,
which is what would leave Approximate mode short of the exactly 2/3 that Mixed
mode reaches. Both are quotients of whole numbers, so neither is rounded here
and both modes answer 2/3. Rounding every step instead does not reach the
manual's digits either, and what its 0.666622 really records is binary
floating point; `test_manual` holds the case and says so.

"To the precision asked for" is a tolerance, and it is one digit tighter than
the digits shown. The looser reading - the simplest rational that displays the
same digits - admits an error of a whole unit in the last digit shown, and
picks deliberately the coarsest value that gets away with it. That is the
worst possible choice for a number that is about to be computed with, because
the arithmetic around an approximate number is exact and so carries the whole
of that error forward. `SQRT(1000003) - SQRT(1000002)` is what it costs: both
roots display `1000.00`, the coarsest rational displaying that is 1000, and a
difference of about 0.0005 comes out 0 with nothing to say it did. The guard
digit is what keeps the substituted value close enough that a cancellation
between two of them still means something.

`approximated` is the other half: what a precision mode does to a whole
expression, which is this rule applied to every number in it that needs digits
and to nothing else. A number there is the largest one the expression holds:
in `x*SQRT(2) + pi` the numbers are `SQRT(2)` and `pi`, and in `x + SQRT(3) -
SQRT(2)` it is the difference of the two roots.

The answer is the simplest rational within the tolerance, found by walking the
continued fraction of the interval's ends: at each step the whole part they
share is the next term of the answer, and the walk stops at the first step
where an integer falls between them. That is what "simplest" means here, and
it is why π comes out 355/113 - not because a coarse tolerance let it, but
because π is approximated by 355/113 far better than a denominator of 113 has
any right to be.

The original can be asked what it does rather than reasoned about: under
Rational notation with NotationDigits at 32 it prints an approximation as the
ratio it is. Asked that way, the original confirms the tolerance above wherever
its own value is accurate - its π over three to nine digits is 333/106,
355/113, 103993/33102, 208341/66317, the first convergent inside 10^-(digits +
1) every time - and shows that where it is not, no tolerance would have helped.
Its SQRT(10001) at six digits is 180109/1801, where 20001/200 is simpler and
two hundred times nearer; its SQRT(2) at three digits is 379/268, where 239/169
is simpler and three times nearer. Those are not a choice of rational but the
error of its own square root, which works a digit or two past the precision
asked for. `test_manual`'s `NOT_YET_HELD` records which of the manual's pages
that costs.
"""

from __future__ import annotations

from collections.abc import Set
from decimal import Decimal
from fractions import Fraction

import sympy as sp

from rederive.engine.context import Context, Precision

__all__ = ["GUARD", "approximated", "simplest", "simplest_rational"]

#: Digits carried past the ones asked for while an irrational is evaluated, so
#: that the digits asked for are the value's own rather than a rounding of it.
GUARD = 5

#: The most digits a number is worked out to while its digits are found. Enough
#: to see through any cancellation short of an exact zero: `SQRT(10^200 + 1) -
#: 10^100` needs about two hundred and fifty, and a hidden zero exhausts it in
#: milliseconds.
MAX_DIGITS = 1000


def simplest(value: Fraction, digits: int) -> Fraction:
    """The simplest rational standing in for `value` at `digits` of precision."""
    if not value:
        return Fraction(0)
    sign = -1 if value < 0 else 1
    value = abs(value)
    tolerance = value * Fraction(1, 10 ** (digits + 1))
    low, high = value - tolerance, value + tolerance
    simplest_here = _simplest_between(low, high)
    # Simplest fixes the denominator, not the whole answer: where the interval
    # holds several rationals over that denominator, they are equally simple
    # and the one nearest the value is the one it stands for. `10^7*pi` turns
    # on it, the interval there holding six whole numbers of which 31415929 is
    # the value's own.
    denominator = simplest_here.denominator
    nearest = Fraction(round(value * denominator), denominator)
    return sign * (nearest if low <= nearest <= high else simplest_here)


def _simplest_between(low: Fraction, high: Fraction) -> Fraction:
    """The simplest rational in `[low, high]`, which must be positive.

    The whole parts of the two ends agree until the interval is wide enough to
    hold an integer, and everything they agree on is forced: any rational
    between them starts the same way. So each shared whole part is written
    down and the walk moves on to the reciprocals of what is left, where the
    ends trade places. The first integer to fall between them ends it, and the
    smallest such integer is the simplest thing the interval can hold.
    """
    whole = low.numerator // low.denominator
    if low.denominator == 1:
        return low
    if whole + 1 <= high:
        return Fraction(whole + 1)
    return whole + 1 / _simplest_between(1 / (high - whole), 1 / (low - whole))


def approximated(expression: sp.Basic, context: Context) -> sp.Basic:
    """What the precision mode does to an expression.

    The last step of every command's pipeline. Factor simplifies exactly, then
    factors, and only then rounds - so that radical factoring reaches `SQRT(2)`
    first and shows `1.41421` because of this, rather than factoring a number
    that has already been rounded.

    It is also what an iteration does to each iterate it hands on, an iterate
    being a value in its own right and the next step's number going in.
    """
    digits = context.precision_digits
    match context.precision:
        case Precision.APPROXIMATE:
            # Each number is approximated as a whole, and then what is left of
            # the arithmetic around them - a coefficient multiplied into a
            # rational, say - is rounded too.
            #
            # What the first pass already rounded is left alone by the second,
            # since rounding an approximate number again is not idempotent: the
            # tolerance is measured from wherever the value now stands, so a
            # second pass may reach a simpler rational a whole tolerance away
            # and cost the last digit shown.
            approximate: set[sp.Rational] = set()
            return _rounded(
                _approximated(expression, digits, approximate), digits, approximate
            )
        case Precision.MIXED:
            return _approximated(expression, digits)
    return expression


def _evalf(expression: sp.Basic, digits: int) -> sp.Basic:
    try:
        return expression.evalf(digits, maxn=MAX_DIGITS)
    except Exception:
        return expression


def _rounded(
    expression: sp.Basic, digits: int, approximate: Set[sp.Rational] = frozenset()
) -> sp.Basic:
    """Every number in `expression` as the simplest rational at this precision.

    Which is what Derive's approximate numbers are, so this is what makes
    `2*y*3` come out `6*y` in every precision mode - only the numbers that
    actually need digits get them - and what leaves an approximate answer an
    exact value that `Notation := Rational` can write out in full.

    A number in `approximate` is one this precision has already rounded, and it
    stands as it is.
    """
    try:
        return expression.replace(
            lambda part: isinstance(part, sp.Rational | sp.Float)
            and part not in approximate,
            lambda part: simplest_rational(part, digits),
            simultaneous=False,
        )
    except Exception:
        return expression


def simplest_rational(number: sp.Basic, digits: int) -> sp.Rational:
    """`simplest` for a sympy number: a rational, or a float evaluation produced.

    A float is read as the decimal digits it shows, not as the binary fraction
    it is stored as. The two differ past the float's precision, and the
    difference is no part of the value: `SQRT(10^200 + 1)` evaluates to a float
    that shows `1.0000000000e+100` and is stored as `10^100 - 1.73*10^87`. Both
    are within the tolerance, and the nearest of the equally simple candidates
    would be the stored one, which leaves the binary error in a number that is
    then computed with exactly.
    """
    if isinstance(number, sp.Float):
        value = Fraction(Decimal(str(number)))
    else:
        rational = sp.Rational(number)
        value = Fraction(rational.p, rational.q)
    return sp.Rational(simplest(value, digits))


def _approximated(
    expression: sp.Basic, digits: int, approximate: set[sp.Rational] | None = None
) -> sp.Basic:
    """Approximate each number in `expression` as a whole, outermost first.

    A number is worked out as if exactly and only then rounded, so every digit
    of the answer is the value's own: `(12^(1/1000)*w)^1000`, `w` a root of
    unity written in radicals, is 12, where rounding `12^(1/1000)` and the
    radicals first and raising them to the thousandth power exactly would carry
    their error a thousandfold into the answer.

    Outermost first is what reaches the whole number rather than its pieces,
    and a sum or a product that is only partly numeric has its numeric terms
    worked out together: `x + SQRT(1000003) - SQRT(1000002)` is `x + 0.000499999`
    and not two roots rounded and then subtracted.

    What each number is replaced by is a rational, since that is what an
    approximate number is. `approximate` collects those rationals for the
    caller, which is how the rounding of the finished answer knows what it has
    already rounded.
    """

    def approximation(part: sp.Basic) -> sp.Basic:
        value = _approximate(part, digits)
        if approximate is not None and value is not part:
            approximate.add(value)
        return value

    def walk(node: sp.Basic) -> sp.Basic:
        if _needs_digits(node):
            return approximation(node)
        if not isinstance(node, sp.Basic) or not node.args:
            return node
        if isinstance(node, sp.Add | sp.Mul):
            numeric = [arg for arg in node.args if arg.is_number]
            if 1 < len(numeric) < len(node.args):
                rest = [arg for arg in node.args if not arg.is_number]
                return node.func(walk(node.func(*numeric)), *map(walk, rest))
        args = tuple(map(walk, node.args))
        if args == node.args:
            return node
        return node.func(*args)

    try:
        return walk(expression)
    except Exception:
        return expression


def _needs_digits(expression: sp.Basic) -> bool:
    """Whether this number has to be worked out before it can be written down.

    Every number does, unless sympy can show that it is rational: a rational
    is already what an approximate number is, and anything else stands for
    digits nothing has computed yet.

    The question is which way round to ask it, and the answer is not the
    obvious one. `is_irrational` has a third answer besides yes and no, and
    that third answer is the common one - sympy leaves a product of surds open,
    since irrationals can multiply to a rational - so asking for a proof of
    irrationality leaves everything unprovable exact, and approximating a
    wholly numeric expression can then answer with radicals still standing in
    it. Asking for a proof of rationality instead sends the unprovable cases
    the other way, and a number can only come out a number.

    Finiteness is asked the same way round and for the same reason. Only what
    is known to be infinite is refused: `LN(0)` has no digits and never will,
    while `SI(2)`, `EI(2)` and `ERFI(2)` are finite numbers sympy simply does not
    prove finite, and demanding the proof left every one of them written as
    itself. An unevaluated integral or sum is the same case, and approximating
    one numerically is what the original does with what it cannot do exactly.
    Nothing is risked by asking loosely here: `_approximate` takes only what
    comes back a float or a rational, so a value that will not evaluate - `nan`,
    or a quadrature that does not converge - is left standing anyway.
    """
    return (
        expression.is_number
        and expression.is_rational is not True
        and expression.is_finite is not False
        and not isinstance(expression, sp.Float)
    )


def _approximate(part: sp.Basic, digits: int) -> sp.Basic:
    """The approximate number `part` stands for, or `part` where there is none.

    A number whose value is not real - `SQRT(-2)`, `(1 + #i)^10` - is approximated
    one part at a time, as the rational real part plus `#i` times the rational
    imaginary part, which is how Derive approximates it too.
    """
    value = _evaluated(part, digits + GUARD)
    if value is None:
        return part
    real, imaginary = value
    if not imaginary:
        return simplest_rational(real, digits)
    return simplest_rational(real, digits) + simplest_rational(imaginary, digits) * sp.I


def _evaluated(part: sp.Basic, digits: int) -> tuple[sp.Basic, sp.Basic] | None:
    """The real and imaginary parts of `part`, each correct to `digits`.

    Sympy raises its working precision until the digits asked for are the
    value's own, up to `MAX_DIGITS`. A value it cannot tell from zero by then
    comes back with no significant digits at all, and is zero: no nonzero value
    is that small next to the numbers it was computed from.

    The precision sympy guarantees is relative to the whole of a complex value,
    so a part far smaller than the other may be noise. Where it is below what
    the precision can see it is taken for zero, which is what makes
    `(12^(1/1000)*w)^1000` 12 rather than `12 + 5.06*10^-17*#i`. Where it is
    above, the value is worked out again with as many more digits as the part is
    smaller, so that its own digits are right too.
    """
    value = _evalf(part, digits)
    parts = _parts(value)
    if parts is None:
        return None
    size = max(abs(parts[0]), abs(parts[1]))
    if not size:
        return parts
    smaller = min(abs(parts[0]), abs(parts[1]))
    if smaller and smaller < size * sp.Rational(1, 10**digits):
        parts = tuple(p if abs(p) == size else sp.Integer(0) for p in parts)
    elif smaller:
        extra = int(sp.ceiling(sp.log(size / smaller, 10)))
        if extra > 0:
            parts = _parts(_evalf(part, digits + extra)) or parts
    return parts


def _parts(value: sp.Basic) -> tuple[sp.Basic, sp.Basic] | None:
    """`value`'s real and imaginary parts, where both are numbers with digits."""
    real, imaginary = value.as_real_imag()
    parts = []
    for number in (real, imaginary):
        if isinstance(number, sp.Float):
            # No significant digits: what sympy answers for a value it could
            # not tell from zero.
            parts.append(sp.Integer(0) if number._prec <= 1 else number)
        elif number.is_Rational:
            parts.append(number)
        else:
            return None
    return parts[0], parts[1]
