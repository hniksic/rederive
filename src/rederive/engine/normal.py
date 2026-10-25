"""Derive's normal form: every sum and quotient written partially factored, recursively.

Derive writes a sum, wherever it stands, as a rational function of one generator:
the most main one the sum holds. The terms that hold it go over one denominator, the
gcd is cancelled, long division takes out whatever part of it is a polynomial, and
that polynomial is collected by the powers of the generator. Everything free of the
generator is an opaque coefficient while that happens, and each coefficient is then a
sum of its own, written the same way about its own generator. Terms free of the
generator are not touched by any of it - they are what the constant term is added to,
and that sum is written about whichever generator is next.

So `(x + 1)^9 + y` is a ninth-degree polynomial in `x` plus `y`, while
`(y + 1)^9 + x` is `x + (y + 1)^9`: the ninth power is free of `x` and is left
exactly as it was written. That is the manual's section 4.3 demonstration, and it is
why `Manage Ordering` changes what Simplify answers. And it is what leaves
`3*x*(2*y + 1)^2` written in powers of `2*y + 1` instead of multiplied out into `y`.

A product is never distributed. Its numerator is cancelled against its denominator,
factor by factor, and that is all: `2*x*(x - 3)^2`, `(x + 1)*(y + 1)` and
`(x + 1)^2/y` come back as they were written, and `(x^2 + 2*x*y + y^2)/(x^2 - y^2)`
comes back `(x + y)/(x - y)` and not long-divided into `1 + 2*y/(x - y)`. Which is
what makes this a normal form rather than a wholesale expansion.

Two things decide what a sum looks like once it is written. A factor every term of
it carries is taken outside where some term carries a sum as a factor, which is how
`2*(x^2 - y^2)^6 - (x^2 - y^2)^5*(2*x^2 - 3)` comes back
`(x^2 - y^2)^5*(3 - 2*y^2)`; a sum of monomials keeps its terms. And a denominator
two or more terms carry that does not hold the generator - a coefficient's
denominator - puts the whole sum over it: `(x + 1)^2/y + z` is
`(x^2 + 2*x + y*z + 1)/y`, while `x/y + z` stays as it is.

The generator is a kernel and not only a variable: a maximal subexpression that is not
a sum, a product or a power with a numeric exponent, ranked by the most main variable
it holds, with plain variables first. `LN(x)*(y + 1)^2 + LN(x)*z` is collected on
`LN(x)`, reaching `(y^2 + 2*y + z + 1)*LN(x)`, because `LN(x)` is the most main thing
the sum is a polynomial in.

One walk, bottom up. Every argument is in normal form before the node above it is
written, and writing a node only ever builds sums out of parts already written, which
is what makes the answer a fixed point of the walk without repeating it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import reduce

import sympy as sp
from sympy.core.mul import _keep_coeff
from sympy.core.intfunc import integer_log

from rederive.engine.expanding import _BOUND
from rederive.engine.ordering import ORDER_LIST

__all__ = ["normal_form"]


def normal_form(
    expression: sp.Basic, order: Sequence[str] = ORDER_LIST, over_numbers: bool = True
) -> sp.Basic:
    """`expression` with every sum and quotient in it written in Derive's normal form.

    `order` is the session's variable order list, which is what decides which
    generator a sum is written about.

    `over_numbers` says whether a *number* several terms carry underneath is one the
    sum is written over. A quotient a computation made survives to the display in the
    original, and dividing by a coefficient is what solving does - `x = (1 - z)/2` and
    not `x = 1/2 - z/2`. A sum that was only ever added up keeps its terms apart,
    however many denominators they have in common: `x^2/6 + x/2` authored comes back
    as it stands, and so does `1/2 + SIN(x)/2`. So the pipeline asks for `False` and
    solving takes the default. A denominator holding a variable is combined either
    way, that being the rational function the normal form is about.

    Total, like every other rewrite in the pipeline: what fails is not taken.
    """
    return _Writer(tuple(order), over_numbers).written(expression)


class _Writer:
    """One walk of the normal form, with what it has written so far.

    The memo is a cache and nothing more: writing a sum is a function of the sum, and
    a coefficient that turns up in several places is written once.
    """

    def __init__(self, order: tuple[str, ...], over_numbers: bool) -> None:
        self.order = order
        self.over_numbers = over_numbers
        self.memo: dict[sp.Basic, sp.Basic] = {}

    def written(self, expression: sp.Basic) -> sp.Basic:
        try:
            return self.memo[expression]
        except (KeyError, TypeError):
            pass
        answer = self._written(expression)
        try:
            self.memo[expression] = answer
        except TypeError:
            pass
        return answer

    def _written(self, expression: sp.Basic) -> sp.Basic:
        if isinstance(expression, sp.MatrixBase):
            entries = _attempt(expression, lambda m: m.applyfunc(self.written))
            return expression if entries is None else entries
        if not isinstance(expression, sp.Basic) or expression.is_Atom:
            return expression
        if not expression.args or isinstance(expression, _BOUND):
            # A head that binds a variable of its own is not reached into: its
            # arguments are not all operands, and what is inside it is its business.
            return expression
        algebraic = isinstance(expression, sp.Expr) and bool(expression.is_commutative)
        if algebraic and not expression.free_symbols and not expression.has(*_BOUND):
            rebuilt = _attempt(expression, _rebuilt)
            return self._number(expression if rebuilt is None else rebuilt)
        arguments = [self.written(argument) for argument in expression.args]
        if algebraic and isinstance(expression, (sp.Add, sp.Mul)):
            return self._step(expression.func(*arguments))
        if all(new is old for new, old in zip(arguments, expression.args)):
            return expression
        rebuilt = _attempt(arguments, lambda parts: expression.func(*parts))
        return expression if rebuilt is None else rebuilt

    def _step(self, built: sp.Basic) -> sp.Basic:
        """A sum or a product of written parts, written itself.

        Asked by what came out of building it, which need not be what went in: a
        product of a number and a sum is a sum by the time sympy has built it.
        """
        if not (isinstance(built, sp.Expr) and built.is_commutative):
            return built
        if built.is_Add:
            rewrite = self._sum
        elif built.is_Mul:
            rewrite = self._product
        else:
            return built
        answer = _attempt(built, rewrite)
        return built if answer is None else answer

    def _number(self, number: sp.Expr) -> sp.Basic:
        """A number, which is the coefficient field and gets plain arithmetic.

        Multiplied out, rationalized and denested term by term, so that
        `(1 + 2*#i)*(3 + #i)` is `1 + 7*#i` and `(2 + #i)/(#i + 1)` is
        `3/2 - #i/2`: what is structure in a polynomial is only arithmetic here.

        Kept only where it is shorter. A number has no generator to be written
        about, and the arithmetic that leaves it longer is no simplification:
        `SQRT(5)*(2 + #i)/5` multiplied out is two terms each carrying the root, and
        `(#e^pi/2 - #e^(-pi)/2)/pi` is `SINH(pi)/pi` the way Simplify writes it.
        """
        value = _attempt(number, _arithmetic)
        if value is None:
            return number
        terms = []
        for term in sp.Add.make_args(value):
            denested = _attempt(term, sp.sqrtdenest)
            terms.append(term if denested is None else denested)
        value = sp.Add(*terms)
        try:
            shorter = sp.count_ops(value) < sp.count_ops(number)
        except Exception:
            shorter = False
        return value if shorter else number

    # -- products -----------------------------------------------------------

    def _product(self, product: sp.Basic) -> sp.Basic:
        """A product, cancelled and spelled, and never multiplied out.

        Whatever the numerator and the denominator have in common is cancelled,
        which is what `(x^2 + 2*x*y + y^2)/(x^2 - y^2)` needs to be `(x + y)/(x - y)`.
        The denominator keeps the factors it arrived with less what was cancelled:
        `1/((x^2 + y^2 + 6*y + 18)*(x^2 + y^2 - 6*y + 18))` is not a quartic.
        """
        if not product.is_Mul:
            return product
        # An undecidable four-argument `IF` carries a truth value, and sympy builds a
        # product out of that too; nothing that is not an expression has a numerator.
        if not all(isinstance(factor, sp.Expr) for factor in product.args):
            return product
        tops, bottoms = _fraction(product)
        if any(bottom.free_symbols for bottom in bottoms):
            cancelled = _cancelled(tops, bottoms)
            if cancelled is not None:
                # What is left of a factor a gcd went into is a polynomial nobody has
                # written yet.
                tops = [self.written(factor) for factor in cancelled[0]]
                bottoms = [self.written(factor) for factor in cancelled[1]]
        answer = _quotient(tops, bottoms)
        # Cancelling everything underneath can leave a number and a sum, which sympy
        # multiplies out as it builds them, and a sum is written as one.
        return self._step(answer) if answer.is_Add else _folded(answer)

    # -- sums ---------------------------------------------------------------

    def _sum(self, total: sp.Basic) -> sp.Basic:
        """One sum about its generator, and its free part about the next.

        The common factor of a sum of compound terms is asked for twice: of the sum as
        it arrives, and of the sum as it is written, since collecting can build one
        that was not there - `phi*x*y - y*(phi + omega)` - and an answer the rule
        would take apart is not a normal form.
        """
        if not total.is_Add:
            return total
        generator = self._main(total)
        if generator is None:
            return total
        common = self._common_factor(total, generator)
        if common is not None:
            return common
        answer = self._about(total, generator)
        if not answer.is_Add:
            return answer
        generator = self._main(answer)
        common = None if generator is None else self._common_factor(answer, generator)
        return answer if common is None else common

    def _about(self, total: sp.Add, generator: sp.Basic) -> sp.Basic:
        """`total` as a rational function of `generator`, and the rest about the next."""
        holding = [term for term in total.args if _holds(term, generator)]
        free = [term for term in total.args if not _holds(term, generator)]
        holder = _Holder()
        # A compound generator stands behind a symbol of its own while the algebra is
        # done, so that nothing reaches inside it: multiplying out reaches into an
        # integral and splits it in two.
        symbol = holder.standing_for(generator)
        frozen = sp.Add(*(_frozen(term, generator, symbol, holder) for term in holding))

        def thaw(part: sp.Basic) -> sp.Basic:
            return self.written(part.xreplace(holder.held))

        written = None
        if _thaws_back(frozen, holder, holding) and _expanded_terms(frozen) <= _EXPANSION:
            try:
                written = _rational_in(frozen, symbol, thaw)
            except Exception:
                written = None
        if written is None:
            # No rational function of the generator to write (`1/SQRT(x) + 1/x`,
            # `(x + 1)^n + y`), or one too large to multiply out: the terms holding
            # the generator stand as they are.
            terms = list(holding)
            rest = self.written(sp.Add(*free))
            return _over_one_denominator(self._shared(sp.Add(*terms, rest)))
        terms, constant = written
        # The polynomial's constant term is free of the generator, and so belongs with
        # the terms that were free of it all along: they are one sum, and it is
        # written about whichever generator is most main in it.
        terms.append(self.written(constant.xreplace(holder.held) + sp.Add(*free)))
        return self._shared(sp.Add(*terms))

    def _common_factor(self, total: sp.Add, generator: sp.Basic) -> sp.Basic | None:
        """A sum of compound products with a factor in common, written with it outside.

        `2*(x^2 - y^2)^6 - (x^2 - y^2)^5*(2*x^2 - 3)` is the fifth power times what is
        left of it, `(x^2 - y^2)^5*(3 - 2*y^2)`, which is what Derive answers and what
        multiplying the powers out about `x` would bury in a polynomial of degree
        twelve. A factor is in common where every term carries it, a power of it
        sharing the lower power.

        Only where some term carries a sum as a factor. A sum of monomials is a
        polynomial already, and taking its common factor out is factoring, which is
        the Factor command's business: `x^2 + 2*x`, `2*x + 2*y` and
        `y*x^2 + 2*y*x + 2*y` are all left as they stand.

        And never a power of the generator itself, which is what the sum is a
        polynomial in rather than a factor of it: `x^2 + 2*x*(a + 1)^10` is that
        polynomial, collected, and taking the `x` out would answer the sum's own normal
        form with a different one.
        """
        if not any(_is_compound(term) for term in total.args):
            return None
        shares = [_positive_powers(term) for term in total.args]
        common = {
            base: min(share[base] for share in shares)
            for base in shares[0]
            if base != generator and all(base in share for share in shares[1:])
        }
        if not common:
            return None
        factor = sp.Mul(*(base**exponent for base, exponent in common.items()))
        rest = self.written(sp.Add(*(term / factor for term in total.args)))
        return self._step(sp.Mul(factor, rest))

    def _shared(self, total: sp.Basic) -> sp.Basic:
        """A sum whose terms share a coefficient's denominator, written over it.

        Which is what makes `(x + 1)^2/y + z` come out `(x^2 + 2*x + y*z + 1)/y`
        while `x/y + z` stays as it is: the first has three terms over `y` and the
        second has one, and a denominator only one term carries is that term's own
        business. Two are enough - `(u - v)/h*(x - a) + w` is
        `(x*(u - v) - a*(u - v) + h*w)/h`, where only two of the three terms arrived
        with an `h` under them.

        A denominator holding the generator is never what is shared: that was the
        rational step's business, and what it left over such a denominator is proper.
        Without `over_numbers` a denominator that is a number is no denominator at all
        for this purpose - it is each term's own rational coefficient.
        """
        if not total.is_Add:
            return total
        generator = self._main(total)
        seen: dict[sp.Basic, int] = {}
        for term in total.args:
            if not isinstance(term, sp.Expr):
                return total
            denominator = _carried(term)[1]
            if denominator == 1:
                continue
            if denominator.free_symbols:
                if generator is not None and _holds(denominator, generator):
                    continue
            elif not self.over_numbers:
                continue
            seen[denominator] = seen.get(denominator, 0) + 1
        if not any(count > 1 for count in seen.values()):
            return total
        holder = _Holder()
        # The numerators frozen and the denominators in the open: they are what the
        # sum is combined over.
        parts = []
        symbol = None if generator is None else holder.standing_for(generator)
        for numerator, denominator in map(_carried, total.args):
            if generator is not None:
                numerator = _frozen(numerator, generator, symbol, holder)
            parts.append(numerator / denominator)
        numerator, denominator = sp.fraction(sp.together(sp.Add(*parts)))
        if denominator == 1:
            return total
        numerator = self.written(numerator.xreplace(holder.held))
        if not denominator.free_symbols and isinstance(numerator, sp.Add):
            # Written as a product with the number outside, which sympy would
            # otherwise multiply back over the sum as it builds the quotient.
            return _keep_coeff(1 / denominator, numerator)
        return _quotient(
            list(sp.Mul.make_args(numerator)), list(sp.Mul.make_args(denominator))
        )

    def _main(self, total: sp.Basic) -> sp.Basic | None:
        """The most main generator of `total`, by the order list.

        A kernel ranks by the most main variable it holds, and a plain variable before
        a compound kernel holding the same one: `x` leads `LN(x)`, which leads `y`.
        """
        best = None
        for kernel in _kernels(total):
            names = [s.name for s in kernel.free_symbols if type(s) is sp.Symbol]
            if not names:
                continue
            key = (min(map(self._rank, names)), not kernel.is_Symbol, str(kernel))
            if best is None or key < best[0]:
                best = (key, kernel)
        return None if best is None else best[1]

    def _rank(self, name: str) -> tuple[int, str]:
        if name in self.order:
            return self.order.index(name), ""
        return len(self.order), name


def _rational_in(
    frozen: sp.Expr, generator: sp.Basic, thaw: Callable[[sp.Basic], sp.Basic]
) -> tuple[list[sp.Basic], sp.Basic]:
    """The terms holding the generator, as a polynomial in it plus a proper part.

    Everything free of the generator stands in for itself, so that a denominator not
    holding the generator is invisible here: it belongs to a coefficient, and a
    coefficient is the business of the sum that coefficient turns out to be. That is
    what leaves `x/y + z` alone while `1/(x + 1) + 1/(x + 2) + y` reaches
    `(2*x + 3)/((x + 1)*(x + 2)) + y`.

    The answer is the written terms and the constant term, still frozen; the constant
    is what the terms free of the generator are added to. Raises where long division
    finds no ratio of polynomials in the generator at all, as in `SIN(x)/(x + 1)`.
    """
    numerator, denominator = sp.fraction(sp.together(frozen))
    numerator = _expanded(numerator)
    cancelled = _cancelled([numerator], list(sp.Mul.make_args(denominator)))
    if cancelled is not None:
        numerator = _expanded(sp.Mul(*cancelled[0]))
        denominator = sp.Mul(*cancelled[1])
    if denominator.has(generator):
        quotient, remainder = sp.div(numerator, denominator, generator)
    else:
        # A denominator free of the generator is a coefficient like any other, and
        # divides into every term: `(x + y*z)/y` is `x/y + z` and only then is it a
        # polynomial in `x`.
        quotient, remainder = _expanded(numerator / denominator), sp.S.Zero
    terms, constant = _collected(quotient, generator, thaw)
    if remainder != 0:
        rest, rest_constant = _collected(_expanded(remainder), generator, thaw)
        # Cancelled once more in the open, where the frozen gcd saw only the generator.
        tops = [sp.Add(*rest, thaw(rest_constant))]
        bottoms = [thaw(factor) for factor in sp.Mul.make_args(denominator)]
        cancelled = _cancelled(tops, bottoms)
        if cancelled is not None:
            tops, bottoms = ([thaw(factor) for factor in side] for side in cancelled)
        terms.append(_quotient(tops, bottoms))
    return terms, constant


def _collected(
    polynomial: sp.Basic, generator: sp.Basic, thaw: Callable[[sp.Basic], sp.Basic]
) -> tuple[list[sp.Basic], sp.Basic]:
    """A multiplied-out polynomial gathered back up by what each term holds of the
    generator.

    Mostly that is a power of it, and each power's coefficient is thawed and written
    keeping its numeric content inside, so that `6*x*y^2*z + 2*x*z^3` is
    `x*(6*y^2*z + 2*z^3)` - which is what the manual's own LAPLACIAN answers. A term
    can also hold the generator inside a kernel of it, `(1 + i)^n` or `LN(x)`, and the
    terms holding the same one are collected the same way: the annuity equation's
    `v*(1 + i)^n + p*t*(1 + i)^n - (p*t + v)*(1 + i)^n` is that kernel times nothing.
    The terms free of the generator come back apart, as the constant term.
    """
    groups: dict[sp.Basic, list[sp.Basic]] = {}
    constant = []
    for term in sp.Add.make_args(polynomial):
        if not term.has(generator):
            constant.append(term)
            continue
        holding, rest = [], []
        for factor in sp.Mul.make_args(term):
            (holding if factor.has(generator) else rest).append(factor)
        groups.setdefault(sp.Mul(*holding), []).append(sp.Mul(*rest))
    terms = [
        thaw(sp.Mul(sp.Add(*coefficients), key)) for key, coefficients in groups.items()
    ]
    return terms, sp.Add(*constant)


def _fraction(product: sp.Expr) -> tuple[list[sp.Expr], list[sp.Expr]]:
    """A product's factors over the line and under it.

    Kept apart rather than multiplied up into a numerator and a denominator, because
    sympy multiplies a number over a sum as it builds a product of the two: the
    numerator of `2*(a + b)/(n + 1)` built on its own is `2*a + 2*b`, and the quotient
    rebuilt from that is no longer the one that was written.
    """
    tops, bottoms = [], []
    for factor in sp.Mul.make_args(product):
        top, bottom = sp.fraction(factor)
        if top != 1:
            tops.append(top)
        if bottom != 1:
            bottoms.append(bottom)
    return tops, bottoms


def _cancelled(
    tops: list[sp.Expr], bottoms: list[sp.Expr]
) -> tuple[list[sp.Expr], list[sp.Expr]] | None:
    """What the factors over and under the line have in common, cancelled.

    Factor by factor, so that what is not cancelled keeps the shape it had: the
    denominator stays the product of the factors it arrived with, and
    `(x + 1)^3*(x + 2)/(x + 1)` is `(x + 1)^2*(x + 2)` rather than a cubic. A factor
    only part of which cancels is replaced by the part that is left. None where
    nothing cancels.

    A factor too large to multiply out is not asked: a gcd works on the expanded
    polynomial, and `(v + w + x + y + z)^60/q` is six hundred thousand terms to learn
    that nothing cancels.
    """
    highs = [list(factor.as_base_exp()) for factor in tops]
    lows = [list(factor.as_base_exp()) for factor in bottoms]
    changed = False
    for low in lows:
        if not _cancellable(low):
            continue
        for high in highs:
            while low[1] > 0 and high[1] > 0 and _cancellable(high):
                try:
                    common = sp.gcd(high[0], low[0])
                except Exception:
                    break
                if common == 1 or not common.free_symbols:
                    break
                for side, sides in ((high, highs), (low, lows)):
                    rest = sp.quo(side[0], common)
                    side[1] -= 1
                    if rest != 1:
                        sides.append([rest, sp.S.One])
                changed = True
    if not changed:
        return None
    return _powers(highs), _powers(lows)


def _cancellable(factor: list[sp.Basic]) -> bool:
    base, exponent = factor
    return (
        bool(exponent.is_Integer)
        and exponent > 0
        and bool(base.free_symbols)
        and _expanded_terms(base) <= _EXPANSION
    )


def _powers(factors: list[list[sp.Basic]]) -> list[sp.Expr]:
    return [base**exponent for base, exponent in factors if exponent != 0 and base != 1]


def _quotient(tops: list[sp.Expr], bottoms: list[sp.Expr]) -> sp.Expr:
    """Factors over factors, spelled the way Derive spells a quotient.

    A sum over the line has its common factor outside, which is the form a gcd leaves
    behind: `(n^2 + n)/(x + 1)` is `n*(n + 1)/(x + 1)`. Only a common factor and not a
    factorization - `(x^2 - 1)/(y + 1)` is printed as it stands.

    A number that is all a sum has beside it over the line is multiplied into it.
    That is how the quotient reads back: `2*(x + 1)/y` is read as `2*(x + 1)` over `y`,
    and sympy multiplies a number over a sum as it builds the product of the two, so
    the form that survives being printed is `(2*x + 2)/y`.
    """
    tops, bottoms = _cleared(tops, bottoms)
    if any(bottom.free_symbols for bottom in bottoms):
        tops = [part for top in tops for part in sp.Mul.make_args(_content(top))]
    if bottoms and len(tops) == 2 and sum(top.is_Number for top in tops) == 1:
        tops = [sp.Add(*map(_folded, sp.Add.make_args(sp.Mul(*tops))))]
    # A bracket whose every term carries a minus sign is written with the sign
    # outside, `-z*(b + 2)` and not `z*(-b - 2)`, and that is the product the text
    # reads back as.
    negated = [_all_negated(top) for top in tops]
    if any(negated):
        tops = [-top if flip else top for top, flip in zip(tops, negated)]
        tops.append(sp.Integer(-1) ** sum(negated))
    quotient = sp.Mul(*tops, *(sp.Pow(bottom, -1) for bottom in bottoms))
    return _coefficient_spelled(quotient)


def _folded(product: sp.Expr) -> sp.Expr:
    """A whole coefficient that is a power of a power's base, folded into the power.

    Sympy folds `2*2^n` into `2^(n + 1)` as it builds it and leaves `-2*2^n` alone,
    and a term written `- 2*2^n*u` is read back as the first of those. So the
    coefficient goes into the power whatever its sign: `-2^(n + 1)*u` is what the
    product reads back as, and `-4*2^n` would be `-2^(n + 2)` to the next Simplify.
    """
    if not product.is_Mul:
        return product
    coefficient, rest = product.as_coeff_Mul()
    if not (coefficient.is_Integer and abs(coefficient) > 1):
        return product
    factors = list(sp.Mul.make_args(rest))
    for index, factor in enumerate(factors):
        base, exponent = factor.as_base_exp()
        if base.is_Integer and base > 1 and not exponent.is_Number:
            power, exact = integer_log(abs(int(coefficient)), int(base))
            if exact:
                sign = 1 if coefficient > 0 else -1
                others = factors[:index] + factors[index + 1 :]
                return sp.Mul(sign, *others, sp.Pow(base, exponent + power))
    return product


def _cleared(
    tops: list[sp.Expr], bottoms: list[sp.Expr]
) -> tuple[list[sp.Expr], list[sp.Expr]]:
    """Factors over factors, with no sum underneath carrying fractions of its own.

    A denominator never has a denominator of its own, and a number is no exception
    there: `2/(u/2 + w/2)` is `4/(u + w)`, which is the same quotient and the only
    spelling of it, where the first is one of as many as there are ways to scale the
    sum underneath. Left alone, the two stand side by side as exponents that cancel
    and do not look as though they do.
    """
    cleared, scale = [], sp.S.One
    for bottom in bottoms:
        base, exponent = bottom.as_base_exp()
        if base.is_Add and exponent.is_Integer and exponent > 0:
            numbers = [term.as_coeff_Mul()[0] for term in base.args]
            if all(number.is_Rational for number in numbers):
                shared = reduce(sp.ilcm, (number.q for number in numbers), 1)
                if shared != 1:
                    base = sp.Add(*(shared * term for term in base.args))
                    scale *= sp.Integer(shared) ** exponent
                    bottom = base**exponent
        cleared.append(bottom)
    if scale == 1:
        return tops, bottoms
    number = sp.Mul(scale, *(top for top in tops if top.is_Number))
    return [number, *(top for top in tops if not top.is_Number)], cleared


def _all_negated(factor: sp.Expr) -> bool:
    return isinstance(factor, sp.Add) and all(
        term.could_extract_minus_sign() for term in factor.args
    )


def _content(top: sp.Expr) -> sp.Expr:
    if not isinstance(top, sp.Add):
        return top
    content = _attempt(top, sp.factor_terms)
    if isinstance(content, sp.Mul) and not content.as_coeff_Mul()[1].is_Add:
        return content
    return top


def _coefficient_spelled(product: sp.Expr) -> sp.Expr:
    """A product holding one sum, with its fraction inside the sum or in front of it.

    Inside where that gives some term a whole coefficient, the way a polynomial's
    coefficients are rational term by term: Derive's answer to the manual's difference
    equation is `(m^2/2 + m/2 + 1)*(m - 1)!`. In front where the fraction would land
    under every term - `x*(x - 1)/2` is that one, a denominator the whole sum shares.

    Decided from the product and not from how it was written: what the sum's terms
    have under them in common is taken out first, so `x*(x/2 - 1/2)` is `x*(x - 1)/2`
    as well, and the product has one spelling whichever way it arrives.
    """
    if not product.is_Mul:
        return product
    coefficient, rest = product.as_coeff_Mul()
    factors = sp.Mul.make_args(rest)
    sums = [factor for factor in factors if factor.is_Add]
    if not coefficient.is_Rational or len(factors) < 2 or len(sums) != 1:
        return product
    others = [factor for factor in factors if not factor.is_Add]
    # The sign stays in front: written into the sum it would turn every term round,
    # and a bracket whose every term carries a minus is written with it outside.
    sign = -1 if coefficient < 0 else 1
    coefficient = abs(coefficient)
    terms = list(sums[0].args)
    numbers = [term.as_coeff_Mul()[0] for term in terms]
    if not all(number.is_Rational for number in numbers):
        return product
    shared = reduce(sp.ilcm, (number.q for number in numbers), 1)
    if shared != 1:
        terms = [shared * term for term in terms]
        coefficient = coefficient / shared
    if coefficient.q == 1:
        if all(sp.fraction(other)[0] == 1 for other in others):
            # A whole number with nothing but a denominator beside it goes into the
            # sum, which is how `_quotient` writes it and how it reads back.
            if shared == 1 and coefficient == 1:
                return product
            spread = sp.Add(*(_folded(coefficient * term) for term in terms))
            return sp.Mul(sign, *others, spread)
        if shared == 1:
            return product
        return sp.Mul(sign * coefficient, *others, sp.Add(*terms))
    spread = [coefficient * term for term in terms]
    if any(term.as_coeff_Mul()[0].is_integer for term in spread):
        return sp.Mul(sign, *others, sp.Add(*spread))
    return sp.Mul(sign * coefficient, *others, sp.Add(*terms))


def _carried(term: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """A term's numerator and the denominator it carries, a coefficient's included.

    `b*(1 - 1/n)` carries an `n` as surely as `a/n` does, the one inside the
    coefficient it was collected with: beside `a/n` the two are one sum over `n`,
    `(a + b*(n - 1))/n`.
    """
    numerator, denominator = sp.fraction(term)
    inner = [
        factor
        for factor in sp.Mul.make_args(numerator)
        if factor.is_Add
        and any(sp.fraction(part)[1].free_symbols for part in factor.args)
    ]
    if not inner:
        return numerator, denominator
    combined = _attempt(term, sp.together)
    return (numerator, denominator) if combined is None else sp.fraction(combined)


def _over_one_denominator(total: sp.Basic) -> sp.Basic:
    """A sum whose every term is over powers of one thing, written over it.

    Where the sum is no rational function of its generator there is nothing to divide
    out, and a denominator standing twice is still worth combining: the lower power
    divides the higher, so what comes out is one ratio. `x^2/(1 - x^2)^(3/2) +
    1/SQRT(1 - x^2)` is `1/(1 - x^2)^(3/2)`, which is what Derive answers
    `DIF(x/SQRT(1 - x^2), x)` with - and authored as it stands, that same sum comes
    back as it stands. Two fixed points for the one expression; this takes the answer
    it gives to the question that was asked.

    Every term has to carry the denominator, and combining has to shorten the sum. A
    term with nothing underneath would have the denominator multiplied into it, which
    is what `b*SQRT(b^2 - 4*a*c)/(4*a) - c*LN(b + SQRT(b^2 - 4*a*c)) - b^2/(4*a)` -
    Derive's own answer to an integral - is here to rule out; and `1/SQRT(x) + 1/x`
    over `x^(3/2)` is longer than it was, which is why the original leaves that one
    alone as well.
    """
    if not total.is_Add:
        return total
    bases = set()
    for term in total.args:
        if not isinstance(term, sp.Expr):
            return total
        denominator = sp.fraction(term)[1]
        if denominator == 1:
            return total
        bases.add(denominator.as_base_exp()[0])
    if len(bases) != 1:
        return total
    combined = _attempt(total, sp.together)
    if combined is None or sp.count_ops(combined) >= sp.count_ops(total):
        return total
    return combined


# -- freezing what is free of the generator --------------------------------------


class _Holder:
    """The placeholders one sum's coefficients stand behind, and the way back.

    One placeholder per distinct coefficient, so that the same quantity in two terms
    is the same symbol to the gcd and to the collecting: `x*(a + b) + (a + b)` is a
    polynomial in `x` with one coefficient and not two that happen to be equal.
    """

    def __init__(self) -> None:
        self.held: dict[sp.Dummy, sp.Basic] = {}
        self.placeholders: dict[sp.Basic, sp.Dummy] = {}

    def hold(self, expression: sp.Basic) -> sp.Basic:
        # A number is not stood in for: `(2 + 3)^2` is 25 whatever is asked for.
        if expression.is_number:
            return expression
        placeholder = self.placeholders.get(expression)
        if placeholder is None:
            placeholder = self.placeholders[expression] = sp.Dummy()
            self.held[placeholder] = expression
        return placeholder

    def standing_for(self, generator: sp.Basic) -> sp.Symbol:
        """The symbol the algebra is done in: the generator, or one standing for it."""
        if isinstance(generator, sp.Symbol):
            return generator
        symbol = sp.Dummy()
        self.held[symbol] = generator
        return symbol


def _frozen(
    expression: sp.Basic, generator: sp.Basic, symbol: sp.Symbol, holder: _Holder
) -> sp.Basic:
    """`expression` with every maximal part free of the generator stood in for.

    A sum and a product take all their parts that are free of the generator together,
    as one placeholder: that is what leaves `x + 2*y + 1` as `x + d`, and so leaves
    `(x + 2*y + 1)^3` about `x` written in powers of `2*y + 1`. A product's numeric
    factor stays outside, being the coefficient the collecting is meant to find, and
    so does a power's numeric exponent. A head that binds a variable of its own is
    stood in for whole, generator and all - see `expanding._BOUND`.
    """
    if expression == generator:
        return symbol
    if not _holds(expression, generator) or isinstance(expression, _BOUND):
        return holder.hold(expression)

    def frozen(part: sp.Basic) -> sp.Basic:
        return _frozen(part, generator, symbol, holder)

    if isinstance(expression, (sp.Add, sp.Mul)):
        involved, free = [], []
        for argument in expression.args:
            (involved if _holds(argument, generator) else free).append(argument)
        parts = [frozen(argument) for argument in involved]
        if isinstance(expression, sp.Mul):
            parts.extend(argument for argument in free if argument.is_Number)
            free = [argument for argument in free if not argument.is_Number]
        if free:
            parts.append(holder.hold(expression.func(*free)))
        return expression.func(*parts)
    if isinstance(expression, sp.Pow):
        base, exponent = expression.args
        if _holds(exponent, generator):
            exponent = frozen(exponent)
        elif not exponent.is_Number:
            exponent = holder.hold(exponent)
        return sp.Pow(frozen(base), exponent)
    if not expression.args:
        return holder.hold(expression)
    return expression.func(*(frozen(argument) for argument in expression.args))


def _thaws_back(frozen: sp.Basic, holder: _Holder, terms: list[sp.Basic]) -> bool:
    """Whether putting the placeholders back gives the terms that were frozen.

    Checked rather than trusted: `_frozen` rebuilds a head it does not know from
    arguments it has rewritten, and a head that will not survive that loses whatever it
    was carrying - a `RootSum` rebuilt this way came back without its sum at all. Where
    it fails the terms are left standing, which is always the safe reading of a sum
    nobody can take apart.
    """
    try:
        return frozen.xreplace(holder.held) == sp.Add(*terms)
    except Exception:
        return False


def _holds(expression: sp.Basic, generator: sp.Basic) -> bool:
    if isinstance(generator, sp.Symbol):
        return generator in expression.free_symbols
    return expression.has(generator)


def _kernels(expression: sp.Basic) -> set[sp.Basic]:
    """The maximal parts of `expression` that are not sums, products or powers.

    A power with a numeric exponent is looked through, fractional or not: the kernel
    of `(x + 1)^(3/2)` is `x`, the same as that of `SQRT(x + 1)` beside it, which is
    what lets the two be written over one denominator. A power whose exponent is not
    a number is a kernel of its own - `(1 + i)^n` is no polynomial in anything.
    """
    if isinstance(expression, (sp.Add, sp.Mul)):
        return {kernel for argument in expression.args for kernel in _kernels(argument)}
    if isinstance(expression, sp.Pow) and expression.exp.is_Rational:
        return _kernels(expression.base)
    if not expression.free_symbols or isinstance(expression, sp.Dummy):
        return set()
    return {expression}


def _is_compound(term: sp.Basic) -> bool:
    """Whether a term carries a sum, or a whole power of one, as a factor."""
    for factor in sp.Mul.make_args(term):
        base, exponent = factor.as_base_exp()
        if base.is_Add and exponent.is_Integer and exponent > 0:
            return True
    return False


def _positive_powers(term: sp.Basic) -> dict[sp.Basic, sp.Basic]:
    """A term's factors with a variable in them, base to exponent, over the line only."""
    powers: dict[sp.Basic, sp.Basic] = {}
    for factor in sp.Mul.make_args(term):
        base, exponent = factor.as_base_exp()
        if not exponent.is_Number:
            # A power whose exponent is no number is a factor of its own, to be
            # shared whole: `(1 + i)^n` and not some power of `1 + i`.
            base, exponent = factor, sp.S.One
        if not base.free_symbols:
            continue
        if exponent.is_Rational and exponent > 0:
            powers[base] = powers.get(base, sp.S.Zero) + exponent
    return powers


# -- sizes ---------------------------------------------------------------------


def _expanded(expression: sp.Basic) -> sp.Basic:
    """Multiplied out, without the two rewrites that take a power apart.

    Neither is anything Derive does: `x^(a + b)` is not `x^a*x^b`, and `(x*y)^n` is
    not `x^n*y^n`.
    """
    return sp.expand(expression, power_exp=False, power_base=False)


#: How many terms an expansion may come to before the normal form declines to build
#: it. Derive returns `(v + w + x + y + z)^60 + 1` written about `x` and returns it at
#: once, which is sixty-one powers of `x`; frozen about `x` that is what it costs
#: here too. What this guards is the expansion freezing does not make small - a
#: dozen squared binomials in the generator come to half a million terms - and the
#: gcd of a numerator nobody asked to multiply out. Measured over every expression in
#: the corpus: the largest expansion any of them asks for is a hundred and twenty
#: terms, so this stands two orders of magnitude above anything a worksheet asks for.
_EXPANSION = 20_000


def _expanded_terms(expression: sp.Basic) -> int:
    """How many terms multiplying `expression` out would come to, at most.

    A bound and not a count: terms that would collect are counted apart, and over a
    limit whose purpose is to decline the expansions nobody asked for that is the safe
    direction to be wrong in. Counted no higher than the limit it is measured against,
    so that a power of a power cannot make the estimate the expensive thing.
    """
    if isinstance(expression, sp.Add):
        return min(sum(_expanded_terms(term) for term in expression.args), _CEILING)
    if isinstance(expression, sp.Mul):
        total = 1
        for factor in expression.args:
            total = min(total * _expanded_terms(factor), _CEILING)
        return total
    if isinstance(expression, sp.Pow):
        return _power_terms(expression)
    return 1


#: What `_expanded_terms` counts up to, one term past what it is asked about.
_CEILING = _EXPANSION + 1


def _power_terms(power: sp.Pow) -> int:
    """The terms a power of a sum comes to, one where it is not one.

    `(u_1 + ... + u_k)^n` has `C(n + k - 1, k - 1)` terms in it, built here as the
    product it is rather than as a binomial, so that a count already past the ceiling
    can stop where it gets there: the number itself is not wanted, only which side of
    the limit it falls.
    """
    terms = _expanded_terms(power.base)
    if terms <= 1 or not power.exp.is_Integer or power.exp <= 0:
        return 1
    exponent = int(power.exp)
    count = 1.0
    for index in range(1, terms):
        count *= (exponent + index) / index
        if count > _EXPANSION:
            return _CEILING
    return round(count)


def _arithmetic(number: sp.Expr) -> sp.Expr | None:
    """A number multiplied out and rationalized, or None where that is too large.

    Rationalizing a denominator multiplies the numerator by its conjugate, and over a
    least-squares fit's coefficients - quotients of sums of `SIN(4)` and `ATAN(4)` -
    what that comes to multiplied out is more terms than anyone will read.
    """
    if _expanded_terms(number) > _EXPANSION:
        return None
    rationalized = sp.radsimp(sp.expand(number, log=False))
    if _expanded_terms(rationalized) > _EXPANSION:
        return None
    return _expanded(rationalized)


def _rebuilt(expression: sp.Basic) -> sp.Basic:
    """Every sum and product in `expression` built again, as sympy would build it.

    A rewrite can hand back a product built with evaluation switched off -
    `(SQRT(2) + SQRT(2)*#i)/2`, a number in front of a sum - which is a form the
    notation cannot hold: reading it back gives the distributed sum.
    """
    if not expression.args or isinstance(expression, _BOUND):
        return expression
    arguments = [_rebuilt(argument) for argument in expression.args]
    if all(new is old for new, old in zip(arguments, expression.args)):
        if not isinstance(expression, (sp.Add, sp.Mul)):
            return expression
    return expression.func(*arguments)


def _attempt(expression, rewrite):
    try:
        candidate = rewrite(expression)
    except Exception:
        return None
    return candidate if isinstance(candidate, sp.Basic) else None
